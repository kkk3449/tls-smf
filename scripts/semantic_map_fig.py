#!/usr/bin/env python3
"""fig19: TOSM semantic-map visualization (paper Sec. 3) — place-layer
floor polygons + per-object class-colored segmented point clouds, in the
style of the lineage's explicit-model maps but with real TLS points.

(a) robot hall: SLIC place regions (pastel floor meshes) + room-scoped
    clusters colored by verified class (unverified = gray)
(b) cafeteria: class-colored clusters over a faint structure cloud

  .venv/bin/python scripts/semantic_map_fig.py
"""
import json
import os
import sys

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from render_instances import _autocrop, _orbit_camera, _pc_mat  # noqa: E402
import open3d.visualization.rendering as rendering  # noqa: E402

FIGS = os.environ.get(
    "MANUSCRIPT_FIGS",
    "/home/caselab/blk360_ros2_ws/docs/electronics2026/figs")
OUT = os.path.join(ROOT, "outputs", "paper_figs")

# fixed class palette (shared across panels; text tokens stay neutral)
CLASS_COLORS = {
    "chair": (0.17, 0.42, 0.69), "door": (0.55, 0.27, 0.07),
    "shelf": (0.55, 0.43, 0.78), "stair": (0.80, 0.47, 0.13),
    "ladder": (0.94, 0.65, 0.21), "motor": (0.75, 0.16, 0.16),
    "robotic arm": (0.86, 0.37, 0.34), "handrail": (0.20, 0.62, 0.36),
    "ramp": (0.61, 0.70, 0.26), "fire extinguisher": (0.89, 0.10, 0.11),
    "wall": (0.45, 0.45, 0.50), "machine": (0.12, 0.53, 0.53),
    "keyboard": (0.30, 0.30, 0.75), "water dispenser": (0.25, 0.60, 0.85),
    "toolbox": (0.72, 0.53, 0.10), "tv": (0.10, 0.10, 0.30),
    "junction box": (0.42, 0.56, 0.14), "steel beam": (0.35, 0.35, 0.35),
    "cable tray": (0.62, 0.44, 0.30), "control panel": (0.50, 0.20, 0.55),
    "ducting": (0.36, 0.54, 0.66), "electrical cabinet": (0.29, 0.33, 0.13),
    "plant": (0.28, 0.66, 0.28),
}
CLUTTER = (0.76, 0.70, 0.60)
UNVERIFIED = (0.55, 0.55, 0.55)


def load_lf(path):
    d = json.load(open(path))
    return {o["id"]: o for o in d["semanticObjects"]}


def cls_color(o):
    st = o["properties"].get("verificationStatus", "")
    t = o["type"].strip().lower()
    if not st.startswith("verified"):
        return UNVERIFIED, None
    if t == "clutter":
        return CLUTTER, "clutter"
    return CLASS_COLORS.get(t, CLUTTER), t


def poly_mesh(poly, z, color):
    v = np.array([[x, y, z] for x, y in poly], float)
    c = v.mean(0)
    verts = np.vstack([v, c])
    tris = [[i, (i + 1) % len(v), len(v)] for i in range(len(v))]
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(np.array(tris)))
    m.paint_uniform_color(color)
    m.compute_vertex_normals()
    return m


def crop_bg(img, pad=14, tol=6):
    corner = img[2, 2, :3].astype(int)
    mask = (np.abs(img[:, :, :3].astype(int) - corner) > tol).any(2)
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
    return img[y0:y1, x0:x1]


def render(geoms, size=(2400, 1600), azim=-120, elev=38, zoom=0.75):
    r = rendering.OffscreenRenderer(*size)
    r.scene.set_background([1, 1, 1, 1])
    pts = np.vstack([np.asarray(g.points) for g in geoms
                     if isinstance(g, o3d.geometry.PointCloud)])
    for i, g in enumerate(geoms):
        mat = _pc_mat(2.8)
        if isinstance(g, o3d.geometry.TriangleMesh):
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
        r.scene.add_geometry(f"g{i}", g, mat)
    lo, hi = pts.min(0), pts.max(0)
    _orbit_camera(r, (lo + hi) / 2, hi - lo, azim, elev, fov=42, zoom=zoom)
    img = np.asarray(r.render_to_image())
    return crop_bg(img)


def pc(xyz, color=None, colors=None):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(xyz)
    if colors is not None:
        p.colors = o3d.utility.Vector3dVector(colors)
    elif color is not None:
        p.paint_uniform_color(color)
    return p


def panel_a():
    lf = load_lf(os.path.join(
        ROOT, "outputs", "vis_n2_det_run1", "semanticObjects.lf_esc.json"))
    sel = open(os.path.join(
        ROOT, "outputs", "vis_n2_hires_select.txt")).read().split(",")
    src = json.load(open(os.path.join(
        ROOT, "outputs", "vis_n2_det_run1", "semanticObjects.json")))
    name2id = {o["name"]: o["id"] for o in src["semanticObjects"]}
    geoms, used = [], {}
    zs = []
    for nm in sel:
        i = name2id[nm]
        f = os.path.join(ROOT, "outputs", "vis_n2_det_hires_objs",
                         f"obj_{int(i):04d}.ply")
        if not os.path.exists(f):
            continue
        q = o3d.io.read_point_cloud(f)
        if len(q.points) == 0:
            continue
        col, label = cls_color(lf[i])
        q.paint_uniform_color(col)
        geoms.append(q)
        zs.append(np.asarray(q.points)[:, 2].min())
        if label:
            used[label] = CLASS_COLORS.get(label, CLUTTER)
    floor_z = float(np.percentile(zs, 10)) - 0.06
    places = json.load(open(os.path.join(
        ROOT, "outputs", "place_layer_T2_places.json")))["semanticPlaces"]
    pcolors = plt.get_cmap("Pastel1")
    pleg = []
    for k, plc in enumerate(places):
        col = pcolors(k % 9)[:3]
        geoms.insert(0, poly_mesh(plc["polygon"], floor_z, col))
        pleg.append((plc["name"], col))
    return render(geoms, azim=-115, elev=40, zoom=0.80), used, pleg


def panel_b():
    lf = load_lf(os.path.join(
        ROOT, "outputs", "cafe8f_objects", "semanticObjects.lf_esc.json"))
    geoms, used = [], {}
    env = o3d.io.read_point_cloud(os.path.join(
        ROOT, "outputs", "cafe8f_scope", "cafe_env_gamma.ply"))
    exyz = np.asarray(env.points)
    m = exyz[:, 2] < np.percentile(exyz[:, 2], 4)      # floor band only
    geoms.append(pc(exyz[m], color=(0.88, 0.87, 0.85)))
    for i, o in lf.items():
        f = os.path.join(ROOT, "outputs", "cafe8f_hires_objs",
                         f"obj_{int(i):04d}.ply")
        q = o3d.io.read_point_cloud(f)
        if len(q.points) == 0:
            continue
        col, label = cls_color(o)
        q.paint_uniform_color(col)
        geoms.append(q)
        if label:
            used[label] = CLASS_COLORS.get(label, CLUTTER)
    return render(geoms, azim=-60, elev=42, zoom=0.66), used


def main():
    img_a, used_a, pleg = panel_a()
    img_b, used_b = panel_b()
    fig = plt.figure(figsize=(13.6, 7.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.03,
                          left=0.005, right=0.995, top=0.93, bottom=0.24)
    for k, (img, title) in enumerate(
            [(img_a, "(a) robot hall: place layer + verified objects"),
             (img_b, "(b) cafeteria: verified objects over floor points")]):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=11.5, loc="left")
    # legends: objects (union of panels) + places
    allc = dict(sorted({**used_a, **used_b}.items()))
    allc.pop("clutter", None)
    handles = [Patch(fc=c, label=t) for t, c in allc.items()]
    handles += [Patch(fc=CLUTTER, label="clutter (verified)"),
                Patch(fc=UNVERIFIED, label="unverified (gated)")]
    leg1 = fig.legend(handles=handles, loc="lower left",
                      bbox_to_anchor=(0.01, 0.075), ncol=8, fontsize=8.2,
                      frameon=True, title="Objects", title_fontsize=9)
    ph = [Patch(fc=c, label=n) for n, c in pleg]
    fig.legend(handles=ph, loc="lower left", bbox_to_anchor=(0.01, -0.008),
               ncol=7, fontsize=8.2, frameon=True, title="Places (a)",
               title_fontsize=9)
    fig.add_artist(leg1)
    for d in (OUT, FIGS):
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "fig19_semantic_map.png")
        fig.savefig(p, dpi=185)
        print("->", p)


if __name__ == "__main__":
    main()
