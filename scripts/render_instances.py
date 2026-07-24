#!/usr/bin/env python3
"""Publication-quality renders of an instance-segmentation result.

Reproduces the common paper figure: each segmented object drawn in its original
RGB with a red 3D bounding box, plus the full-scene reference.

  CUDA_HOME=/path/to/cuda \
  python scripts/render_instances.py \
      --objects-dir outputs/stage2_no_wall_objects \
      --scene ../testroom_no_wall/stage2_no_wall.e57 \
      --classification outputs/stage2_no_wall_objects/classification.csv

Writes to <objects-dir>/figs/:
  instances_overview.png   all objects (RGB) + red bboxes, oblique view
  scene_reference.png      full original RGB scene
  instances_gallery.png    grid of individual objects, each w/ bbox + class label
"""
import argparse
import glob
import os
import sys

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _autocrop(img, bg=250, pad=18):
    """Trim near-white borders so the figure is tight (paper-ready)."""
    mask = (img[:, :, :3] < bg).any(2)
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
    return img[y0:y1, x0:x1]


def _pc_mat(point_size=3.0):
    m = rendering.MaterialRecord()
    m.shader = "defaultUnlit"
    m.point_size = point_size
    return m


def _line_mat(width=4.0):
    m = rendering.MaterialRecord()
    m.shader = "unlitLine"
    m.line_width = width
    return m


def _aabb_lines(pts, color=(0.9, 0.05, 0.05), pad=0.02):
    aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(pts.astype(np.float64)))
    aabb.min_bound = aabb.min_bound - pad
    aabb.max_bound = aabb.max_bound + pad
    ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
    ls.paint_uniform_color(color)
    return ls


def _orbit_camera(renderer, center, extent, azim_deg, elev_deg, fov=55.0, zoom=1.9):
    az, el = np.radians(azim_deg), np.radians(elev_deg)
    r = float(np.linalg.norm(extent)) * zoom
    eye = np.array(center) + r * np.array([
        np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    renderer.setup_camera(fov, center.astype(np.float64),
                          eye.astype(np.float64), np.array([0.0, 0.0, 1.0]))


def render_geometries(geoms, W, H, azim, elev, bg=(1, 1, 1, 1), point_size=3.0,
                      box_width=4.0):
    r = rendering.OffscreenRenderer(W, H)
    r.scene.set_background(list(bg))
    r.scene.scene.set_sun_light([0.3, 0.3, -1.0], [1, 1, 1], 60000)
    r.scene.scene.enable_sun_light(False)
    allpts = []
    for name, g in geoms:
        if isinstance(g, o3d.geometry.LineSet):
            r.scene.add_geometry(name, g, _line_mat(box_width))
            allpts.append(np.asarray(g.points))
        else:
            r.scene.add_geometry(name, g, _pc_mat(point_size))
            allpts.append(np.asarray(g.points))
    pts = np.concatenate(allpts, 0)
    center = pts.mean(0)
    extent = pts.max(0) - pts.min(0)
    _orbit_camera(r, center, extent, azim, elev)
    img = np.asarray(r.render_to_image())
    return img


def render_object_views(pc, azimuths=(0, 90, 180, 270), elev=25, size=512,
                        box=True):
    """Render one object from several azimuths -> list of cropped RGB arrays.

    Multi-view input lets the Stage-B VLM disambiguate look-alikes (a chair from
    one angle can read as 'stair'); 4 views around the object fix most of that.
    """
    pts = np.asarray(pc.points)
    imgs = []
    for az in azimuths:
        geoms = [("o", pc)]
        if box:
            geoms.append(("b", _aabb_lines(pts)))
        imgs.append(_autocrop(render_geometries(geoms, size, size, az, elev,
                                                 point_size=4.0)))
    return imgs


def load_objects(objects_dir):
    files = sorted(glob.glob(os.path.join(objects_dir, "obj_*.ply")))
    objs = []
    for f in files:
        pc = o3d.io.read_point_cloud(f)
        xyz = np.asarray(pc.points)
        if len(xyz) == 0:
            continue
        objs.append({"file": os.path.basename(f), "pc": pc, "xyz": xyz})
    return objs


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--objects-dir", required=True)
    ap.add_argument("--scene", default=None, help="original e57/ply for the reference panel")
    ap.add_argument("--classification", default=None, help="classification.csv for labels")
    ap.add_argument("--azim", type=float, default=-60)
    ap.add_argument("--elev", type=float, default=35)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1100)
    ap.add_argument("--gallery-n", type=int, default=12)
    args = ap.parse_args()

    out = os.path.join(args.objects_dir, "figs")
    os.makedirs(out, exist_ok=True)
    objs = load_objects(args.objects_dir)
    print(f"[render] {len(objs)} objects from {args.objects_dir}")

    labels = {}
    if args.classification and os.path.exists(args.classification):
        cdf = pd.read_csv(args.classification)
        for _, row in cdf.iterrows():
            labels[row["file"]] = (str(row.get("top1", "")), float(row.get("score1", 0)))

    # ---- overview: all objects (RGB) + red bboxes ----
    geoms = []
    for i, o in enumerate(objs):
        geoms.append((f"o{i}", o["pc"]))
        geoms.append((f"b{i}", _aabb_lines(o["xyz"])))
    ov = _autocrop(render_geometries(geoms, args.width, args.height, args.azim, args.elev))
    plt.imsave(os.path.join(out, "instances_overview.png"), ov)
    print(f"[render] wrote instances_overview.png  ({ov.shape[1]}x{ov.shape[0]})")

    # ---- scene reference (original RGB) ----
    if args.scene:
        from blk360seg import io, preprocess
        import yaml
        cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "default.yaml")))
        sx, sr = io.load(args.scene)
        sx, sr = preprocess.voxel_downsample(sx, sr, cfg["preprocess"]["voxel_size_m"])
        if sr.max() > 1.5:
            sr = sr / 255.0
        spc = o3d.geometry.PointCloud()
        spc.points = o3d.utility.Vector3dVector(sx.astype(np.float64))
        spc.colors = o3d.utility.Vector3dVector(np.clip(sr, 0, 1).astype(np.float64))
        sc = _autocrop(render_geometries([("scene", spc)], args.width, args.height,
                                         args.azim, args.elev, point_size=2.5))
        plt.imsave(os.path.join(out, "scene_reference.png"), sc)
        print("[render] wrote scene_reference.png")

    # ---- gallery: top-N individual objects, each w/ bbox + label ----
    objs_sorted = sorted(objs, key=lambda o: -len(o["xyz"]))[:args.gallery_n]
    tiles = []
    for o in objs_sorted:
        img = _autocrop(render_geometries([("o", o["pc"]), ("b", _aabb_lines(o["xyz"]))],
                                          640, 640, args.azim, args.elev, point_size=4.0))
        cap = o["file"]
        if o["file"] in labels:
            cls, sc_ = labels[o["file"]]
            cap = f"{cls} ({sc_:.2f})"
        tiles.append((img, cap, len(o["xyz"])))
    ncol = 4
    nrow = int(np.ceil(len(tiles) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.2, nrow * 3.4))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (img, cap, n) in zip(np.atleast_1d(axes).ravel(), tiles):
        ax.imshow(img)
        ax.set_title(f"{cap}\n{n:,} pts", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "instances_gallery.png"), dpi=120, bbox_inches="tight")
    print(f"[render] wrote instances_gallery.png ({len(tiles)} tiles)")


if __name__ == "__main__":
    main()
