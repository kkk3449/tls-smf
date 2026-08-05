#!/usr/bin/env python3
"""Render the Chapter-6 figures for the dissertation.

  fig6_raw.png      — raw BLK360 E57 scan of the testroom (walls intact)
  fig6_nowall.png   — the same scan after structural-plane removal (RGB)
  fig6_seg.png      — 4-panel: (a) no-wall RGB, (b) DBSCAN+Uni3D, (c) SPFormer, (d) PTv3
  fig6_vlm.png      — VLM TOSM symbolic-correction gallery (before -> after)

EGL offscreen (headless). Run with the project venv.
"""
import os
import sys
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
sys.path.insert(0, ROOT)
from blk360seg import io as bio, preprocess  # noqa: E402

CYC = os.environ.get("BLK360_DATA_DIR", "data")
RAW_E57 = os.path.join(CYC, "testroom260601.e57")
NOWALL_PLY = os.path.join(CYC, "testroom_no_wall/stage2_no_wall.ply")


def _autocrop(img):
    mask = (img[:, :, :3] < 245).any(axis=2)
    if mask.any():
        ys, xs = np.where(mask)
        pad = 12
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
        img = img[y0:y1, x0:x1]
    return img


def render_pc(pc, w=1000, h=820, eye_dir=(0.8, -1.0, 0.7), dist_scale=0.95,
              point_size=6.5, crop=True):
    r = rendering.OffscreenRenderer(w, h)
    r.scene.set_background([1, 1, 1, 1])
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = point_size
    r.scene.add_geometry("pc", pc, mat)
    bb = pc.get_axis_aligned_bounding_box()
    c = bb.get_center()
    ext = bb.get_extent()
    d = float(np.linalg.norm(ext)) * dist_scale
    ed = np.array(eye_dir, dtype=float)
    ed = ed / np.linalg.norm(ed)
    eye = c + ed * d
    r.scene.camera.look_at(c, eye, [0, 0, 1])
    img = np.asarray(r.render_to_image())
    del r
    return _autocrop(img) if crop else img


def pc_from_xyzrgb(xyz, rgb):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0, 1).astype(np.float64))
    return pc


def load_raw(point_size_voxel=0.03):
    xyz, rgb = bio.load(RAW_E57)
    print(f"[raw] {len(xyz):,} points loaded")
    xyz, rgb = preprocess.voxel_downsample(xyz, rgb, point_size_voxel)
    print(f"[raw] {len(xyz):,} after voxel {point_size_voxel} m")
    return pc_from_xyzrgb(xyz, rgb)


def fig_raw():
    pc = load_raw()
    img = render_pc(pc, point_size=4.0, dist_scale=0.9)
    plt.figure(figsize=(7, 5.6))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    out = os.path.join(OUT, "fig6_raw.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print("wrote", out)


def fig_nowall():
    pc = o3d.io.read_point_cloud(NOWALL_PLY)
    img = render_pc(pc, point_size=4.5)
    plt.figure(figsize=(7, 5.6))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    out = os.path.join(OUT, "fig6_nowall.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print("wrote", out)


def fig_segmentation():
    nowall = o3d.io.read_point_cloud(NOWALL_PLY)
    panels = [
        ("RGB", nowall, "(a) Input scan (walls removed)\nraw RGB, 122,738 points"),
        ("PLY", os.path.join(OUT, "stage2_no_wall_objects_split/scene_classified.ply"),
         "(b) DBSCAN + Uni3D\n98.3% coverage, 13 classes"),
        ("PLY", os.path.join(OUT, "stage2_no_wall_spformer/scene_classified.ply"),
         "(c) SPFormer (ScanNet)\n51.1% coverage, furniture-biased"),
        ("PLY", os.path.join(OUT, "stage2_no_wall_ptv3/scene_ptv3_semseg.ply"),
         "(d) PTv3 (ScanNet)\n46% labeled 'wall' (none present)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    for ax, (kind, src, title) in zip(axes, panels):
        pc = src if kind == "RGB" else (
            o3d.io.read_point_cloud(src) if os.path.exists(src) else None)
        if pc is not None:
            ax.imshow(render_pc(pc))
        else:
            ax.text(0.5, 0.5, "missing", ha="center")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    plt.tight_layout()
    out = os.path.join(OUT, "fig6_seg.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print("wrote", out)


def fig_vlm():
    OD = os.path.join(OUT, "stage2_no_wall_objects_split")
    m = json.load(open(os.path.join(OD, "semanticObjects.annotated.json")))
    orig = {o["id"]: o["type"]
            for o in json.load(open(os.path.join(OD, "semanticObjects.json")))["semanticObjects"]}
    want = ["mobile_robot_002", "chair_005", "tv_039", "monitor_018",
            "chair_041", "machine_004"]
    objs = {o["name"]: o for o in m["semanticObjects"]}
    sel = [objs[n] for n in want if n in objs]
    cols = 3
    rows = (len(sel) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.3, rows * 3.5))
    axes = np.array(axes).flatten()
    for ax, o in zip(axes, sel):
        iv = o["properties"].get("imageViews", [])
        p = os.path.join(OD, iv[0]) if iv else None
        if p and os.path.exists(p):
            ax.imshow(mpimg.imread(p))
        before = orig.get(o["id"], "?")
        after = o["type"]
        tag = (f"{before}  ->  {after}" if before != after else f"{after} (kept)")
        ax.set_title(f"{tag}\nconf {o['confidence']}", fontsize=10)
        ax.axis("off")
    for j in range(len(sel), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    out = os.path.join(OUT, "fig6_vlm.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print("wrote", out)


def fig_instances():
    """Standalone instance-decomposition render (DBSCAN+Uni3D colors)."""
    src = os.path.join(OUT, "stage2_no_wall_objects_split/scene_classified.ply")
    pc = o3d.io.read_point_cloud(src)
    img = render_pc(pc, point_size=4.5)
    plt.figure(figsize=(7, 5.6))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    out = os.path.join(OUT, "fig6_instances.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print("wrote", out)


def fig_tosm(rows=2, cols=2, want=None, out_name="fig6_tosm.png",
             figsize=None, letter=True):
    """Completed TOSM semantic-object records: render + Symbolic/Explicit card."""
    OD = os.path.join(OUT, "stage2_no_wall_objects_split")
    m = json.load(open(os.path.join(OD, "semanticObjects.annotated.json")))
    objs = {o["name"]: o for o in m["semanticObjects"]}
    want = want or ["mobile_robot_002", "machine_004", "tv_039", "chair_005"]
    sel = [objs[n] for n in want if n in objs]

    def card_text(o):
        d = o["dimensions"]
        pr = o["properties"]
        verified = pr.get("verification",
                          str(pr.get("symbolicVerified", False)).lower())
        if len(verified) > 24 and " (" in verified:  # wrap long manual note
            head, tail = verified.rsplit(" (", 1)
            ver_lines = [("  verified", head), ("", "(" + tail)]
        else:
            ver_lines = [("  verified", verified)]
        lines = [
            ("Symbolic", None),
            ("  type", o["type"]),
            ("  name", o["name"]),
            ("  id", str(o["id"])),
            *ver_lines,
            ("Explicit", None),
            ("  pose (map)", f"x {o['poseX']:.2f}   y {o['poseY']:.2f}"),
            ("", f"z {pr['poseZ']:.2f} m"),
            ("  theta", f"{o['poseTheta']:.2f} rad"),
            ("  dimensions", f"{d['length']:.2f} x {d['width']:.2f} x "
                             f"{d['height']:.2f} m"),
            ("", "(l x w x h)"),
            ("  color", o["color"]),
            ("  confidence", f"{o['confidence']:.2f}"),
            ("Implicit", None),
            ("  isKeyObject", str(pr.get("isKeyObject", False)).lower()),
            ("  isMovable", str(pr.get("isMovable", False)).lower()),
        ]
        return lines

    fig = plt.figure(figsize=figsize or (cols * 6.6, rows * 4.6))
    gs = fig.add_gridspec(rows, cols, wspace=0.04, hspace=0.16)
    for k, o in enumerate(sel):
        sub = gs[k // cols, k % cols].subgridspec(1, 2, width_ratios=[1.0, 1.15],
                                                  wspace=0.02)
        ax_im = fig.add_subplot(sub[0, 0])
        ax_tx = fig.add_subplot(sub[0, 1])
        iv = o["properties"].get("imageViews", [])
        p = os.path.join(OD, iv[0]) if iv else None
        if p and os.path.exists(p):
            img = mpimg.imread(p)
            ax_im.imshow(_autocrop((img * 255).astype(np.uint8)
                                   if img.dtype != np.uint8 else img))
        head = f"({chr(97 + k)}) {o['name']}" if letter else o["name"]
        ax_im.set_title(head, fontsize=12, loc="left")
        ax_im.axis("off")
        ax_tx.axis("off")
        y = 0.97
        for key, val in card_text(o):
            if val is None:  # section header
                ax_tx.text(0.02, y, key, fontsize=10.5, fontweight="bold",
                           family="monospace", va="top", color="#1a4f8b")
            else:
                ax_tx.text(0.04, y, f"{key:<13}", fontsize=9.5,
                           family="monospace", va="top", color="0.35")
                fs = 9.5 if len(val) <= 32 else 8.6
                ax_tx.text(0.455, y, val, fontsize=fs, family="monospace",
                           va="top", color="0.05")
            y -= 0.06
        for spine in ax_tx.spines.values():
            spine.set_visible(True)
            spine.set_color("0.75")
    plt.tight_layout()
    out = os.path.join(OUT, out_name)
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print("wrote", out)


if __name__ == "__main__":
    targets = sys.argv[1:] or ["raw", "nowall", "seg", "vlm"]
    if "raw" in targets:
        fig_raw()
    if "nowall" in targets:
        fig_nowall()
    if "seg" in targets:
        fig_segmentation()
    if "instances" in targets:
        fig_instances()
    if "vlm" in targets:
        fig_vlm()
    if "tosm" in targets:
        fig_tosm()
    if "tosm_strip" in targets:
        fig_tosm(rows=1, cols=2, want=["mobile_robot_002", "machine_004"],
                 out_name="fig6_tosm_strip.png", letter=False)
