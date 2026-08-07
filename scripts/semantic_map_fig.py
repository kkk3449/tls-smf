#!/usr/bin/env python3
"""fig19: TOSM semantic-map visualization (paper Sec. 3) — place-layer
floor + per-object class-colored segmented point clouds, with place-name
labels rendered on the map and per-panel legends.

(a) robot hall: SLIC place polygons + room-scoped clusters
(b) cafeteria: SLIC place cells (synthesized grid) + clusters

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
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from render_instances import _orbit_camera, _pc_mat  # noqa: E402
import open3d.visualization.rendering as rendering  # noqa: E402

FIGS = os.environ.get(
    "MANUSCRIPT_FIGS",
    "/home/caselab/blk360_ros2_ws/docs/electronics2026/figs")
OUT = os.path.join(ROOT, "outputs", "paper_figs")

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
    return {o["id"]: o
            for o in json.load(open(path))["semanticObjects"]}


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
    verts = np.vstack([v, v.mean(0)])
    tris = [[i, (i + 1) % len(v), len(v)] for i in range(len(v))]
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(np.array(tris)))
    m.paint_uniform_color(color)
    m.compute_vertex_normals()
    return m


def pc(xyz, color=None):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(xyz, float))
    if color is not None:
        p.paint_uniform_color(color)
    return p


def crop_bg(img, pad=14, tol=6):
    corner = img[2, 2, :3].astype(int)
    mask = (np.abs(img[:, :, :3].astype(int) - corner) > tol).any(2)
    if not mask.any():
        return img, (0, 0)
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
    return img[y0:y1, x0:x1], (y0, x0)


def render(geoms, labels3d, size=(2400, 1600), azim=-120, elev=38,
           zoom=0.78, fov=42, floor_idx=frozenset()):
    r = rendering.OffscreenRenderer(*size)
    r.scene.set_background([1, 1, 1, 1])
    pts = np.vstack([np.asarray(g.points) for g in geoms
                     if isinstance(g, o3d.geometry.PointCloud)])
    for i, g in enumerate(geoms):
        mat = _pc_mat(7.5 if i in floor_idx else 2.8)
        if isinstance(g, o3d.geometry.TriangleMesh):
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
        r.scene.add_geometry(f"g{i}", g, mat)
    lo, hi = pts.min(0), pts.max(0)
    center, extent = (lo + hi) / 2, hi - lo
    _orbit_camera(r, center, extent, azim, elev, fov=fov, zoom=zoom)
    img = np.asarray(r.render_to_image())
    img, (oy, ox) = crop_bg(img)

    # replicate the camera to project 3D label anchors to pixels
    az, el = np.radians(azim), np.radians(elev)
    eye = center + float(np.linalg.norm(extent)) * zoom * np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    f = center - eye
    f = f / np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up2 = np.cross(right, f)
    W, H = size
    t = np.tan(np.radians(fov) / 2)
    out = []
    for (x, y, z, txt) in labels3d:
        d = np.array([x, y, z]) - eye
        zc = d @ f
        px = (d @ right / zc) / (t * W / H)
        py = (d @ up2 / zc) / t
        out.append(((px + 1) / 2 * W - ox, (1 - (py + 1) / 2) * H - oy, txt))
    return img, out


def flat_cells(cells, z, color):
    return pc(np.array([[x, y, z] for x, y in cells]), color)


def pad_to_aspect(img, labels, aspect):
    """Center-pad with the background color to the exact target aspect."""
    h, w = img.shape[:2]
    bg = img[2, 2]
    if w / h < aspect:                     # too narrow -> pad width
        nw = int(round(h * aspect))
        px = (nw - w) // 2
        canvas = np.tile(bg, (h, nw, 1)).astype(img.dtype)
        canvas[:, px:px + w] = img
        labels = [(x + px, y, t) for x, y, t in labels]
    else:                                  # too wide -> pad height
        nh = int(round(w / aspect))
        py = (nh - h) // 2
        canvas = np.tile(bg, (nh, w, 1)).astype(img.dtype)
        canvas[py:py + h] = img
        labels = [(x, y + py, t) for x, y, t in labels]
    return canvas, labels


PHANTOM_TYPES = {"keyboard"}   # verified labels the owner refuted (no such
                               # object exists in either scene)
GATED_CALLOUT = {"door", "motor", "robotic arm", "tv", "machine",
                 "control panel", "plant"}   # gated clusters worth naming by their
                                    # own (withheld) top vote


def load_objs(lf, objdir, ids=None):
    geoms, used, zs, phantoms = [], {}, [], []
    for i, o in lf.items():
        if ids is not None and i not in ids:
            continue
        f = os.path.join(objdir, f"obj_{int(i):04d}.ply")
        if not os.path.exists(f):
            continue
        q = o3d.io.read_point_cloud(f)
        if len(q.points) == 0:
            continue
        col, label = cls_color(o)
        q.paint_uniform_color(col)
        geoms.append(q)
        pts = np.asarray(q.points)
        zs.append(pts[:, 2].min())
        st = o["properties"].get("verificationStatus", "")
        if not st.startswith("verified") and \
                o["type"].strip().lower() in GATED_CALLOUT:
            c = pts.mean(0)
            phantoms.append((c[0], c[1], pts[:, 2].max() + 0.12,
                             f"gated:{o['type'].strip().lower()}"))
        if label in PHANTOM_TYPES:
            c = pts.mean(0)
            phantoms.append((c[0], c[1], pts[:, 2].max() + 0.15,
                             f"phantom:{label}"))
        if label and label != "clutter":
            key = f"{label} (phantom)" if label in PHANTOM_TYPES else label
            used[key] = CLASS_COLORS.get(label, CLUTTER)
    return geoms, used, zs, phantoms


def panel_a():
    lf = load_lf(os.path.join(
        ROOT, "outputs", "vis_n2_det_run1", "semanticObjects.lf_esc.json"))
    sel = open(os.path.join(
        ROOT, "outputs", "vis_n2_hires_select.txt")).read().split(",")
    src = json.load(open(os.path.join(
        ROOT, "outputs", "vis_n2_det_run1", "semanticObjects.json")))
    ids = {o["id"] for o in src["semanticObjects"] if o["name"] in sel}
    geoms, used, zs, phantoms = load_objs(
        lf, os.path.join(ROOT, "outputs", "vis_n2_det_hires_objs"), ids)
    floor_z = float(np.percentile(zs, 10)) - 0.06
    places = json.load(open(os.path.join(
        ROOT, "outputs", "place_layer_T2_places.json")))["semanticPlaces"]
    pal = plt.get_cmap("Pastel1")
    labels3d = []
    for k, plc in enumerate(places):
        geoms.insert(0, poly_mesh(plc["polygon"], floor_z, pal(k % 9)[:3]))
        cx, cy = plc["centroid"]
        labels3d.append((cx, cy, floor_z, plc["name"]))
    labels3d += phantoms
    img, lab = render(geoms, labels3d, azim=-115, elev=40, zoom=0.80)
    nudge = {"displaying_section_003": (-300, 6),
             "tv_section_005": (70, 14),
             "hardware_section_004": (30, -30)}
    lab = [(x + nudge.get(t, (0, 0))[0], y + nudge.get(t, (0, 0))[1], t)
           for x, y, t in lab]
    return img, used, lab


def panel_b():
    lf = load_lf(os.path.join(
        ROOT, "outputs", "cafe8f_objects", "semanticObjects.lf_esc.json"))
    geoms, used, zs, phantoms = load_objs(
        lf, os.path.join(ROOT, "outputs", "cafe8f_hires_objs"))
    floor_z = float(np.percentile(zs, 10)) - 0.06
    places = json.load(open(os.path.join(
        ROOT, "outputs", "place_layer_CAFE_places.json")))["semanticPlaces"]
    pal = plt.get_cmap("Pastel1")
    labels3d = []
    for k, plc in enumerate(places):
        geoms.insert(0, flat_cells(plc["cells"], floor_z, pal(k % 9)[:3]))
        cx, cy = plc["centroid"]
        labels3d.append((cx, cy, floor_z, plc["name"]))
    labels3d += phantoms
    img, lab = render(geoms, labels3d, azim=-35, elev=50, zoom=0.80,
                      size=(2400, 2000),
                      floor_idx=frozenset(range(len(places))))
    nudge = {"gated:plant": (150, -55)}
    lab = [(x + nudge.get(t, (0, 0))[0], y + nudge.get(t, (0, 0))[1], t)
           for x, y, t in lab]
    return img, used, lab


def draw_panel(ax, img, labels, used, title):
    ax.imshow(img)
    placed = []
    def declash(x, y):
        for px, py in placed:
            if abs(x - px) < 260 and abs(y - py) < 46:
                y = py + 54
                x = px + 34
        placed.append((x, y))
        return x, y
    ax.axis("off")
    ax.set_title(title, fontsize=16.5, loc="left", pad=86)
    for x, y, txt in labels:
        if not (0 <= x <= img.shape[1] and 0 <= y <= img.shape[0]):
            continue
        if txt.startswith("phantom:"):
            ax.annotate("$\\times$", (x, y), ha="center", va="center",
                        fontsize=18, fontweight="bold", color="#c53030")
            continue
        if txt.startswith("gated:"):
            ax.annotate(f"{txt[6:]}?", (x, y), ha="center", va="center",
                        fontsize=10.5, style="italic", color="#555555",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="#aaaaaa", lw=0.5, alpha=0.8))
            continue
        ax.annotate(txt, (x, y), ha="center", fontsize=12.5,
                    style="italic", color="#333333",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="#999999", lw=0.5, alpha=0.85))
    xmark = Line2D([], [], marker="x", color="#c53030", ls="",
                   markersize=9, markeredgewidth=2.4)
    handles, labels_txt = [], []
    for t, c in sorted(used.items()):
        if "(phantom)" in t:
            handles.append((Patch(fc=c), xmark))
            labels_txt.append(t)
        else:
            handles.append(Patch(fc=c))
            labels_txt.append(t)
    handles += [Patch(fc=CLUTTER), Patch(fc=UNVERIFIED)]
    labels_txt += ["clutter (verified)", "unverified (gated)"]
    ax.legend(handles, labels_txt, loc="lower left",
              bbox_to_anchor=(0.0, 1.005), ncol=3, fontsize=12.5,
              frameon=False, borderaxespad=0, handlelength=1.4,
              columnspacing=0.8, handletextpad=0.5,
              handler_map={tuple: HandlerTuple(ndivide=None)})


def main():
    img_a, used_a, lab_a = panel_a()
    img_b, used_b, lab_b = panel_b()
    # identical panel size: pad both to a common aspect, equal widths
    asp_a = img_a.shape[1] / img_a.shape[0]
    asp_b = img_b.shape[1] / img_b.shape[0]
    print(f"aspects a={asp_a:.2f} b={asp_b:.2f}")
    target = max(asp_a, asp_b)
    img_a, lab_a = pad_to_aspect(img_a, lab_a, target)
    img_b, lab_b = pad_to_aspect(img_b, lab_b, target)
    fig = plt.figure(figsize=(13.8, 13.8 / 2 / target + 1.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.02,
                          left=0.004, right=0.996,
                          top=1 - 1.32 / (13.8 / 2 / target + 1.45),
                          bottom=0.01)
    for k, (img, lab, used, title) in enumerate(
            [(img_a, lab_a, used_a,
              "(a) robot hall: place layer + verified objects"),
             (img_b, lab_b, used_b,
              "(b) cafeteria: place layer + verified objects")]):
        ax = fig.add_subplot(gs[0, k])
        ax.set_anchor("NW" if k == 0 else "NE")
        draw_panel(ax, img, lab, used, title)
    for d in (OUT, FIGS):
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "fig19_semantic_map.png")
        fig.savefig(p, dpi=185, bbox_inches="tight")
        print("->", p)


if __name__ == "__main__":
    main()
