#!/usr/bin/env python3
"""Yaw-align the 8F cafeteria scan to its dominant wall direction and render
a gridded top-down view for crop-bound confirmation.

  1. load E57 -> 3cm voxel downsample
  2. wall-normal histogram (near-horizontal normals, angle mod 90deg) -> yaw
  3. rotate by -yaw about z, save aligned.ply
  4. top-down render with 1m grid + optional crop box overlay

  python scripts/cafe_scope.py --input <e57> [--crop xmin xmax ymin ymax]
  python scripts/cafe_scope.py --render-only [--crop ...]   # reuse aligned.ply
With --crop also writes cafe_crop.ply (points inside the box, full z range).
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blk360seg import io  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "cafe8f_scope")


def estimate_yaw(pc):
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.15, max_nn=30))
    n = np.asarray(pc.normals)
    horiz = np.abs(n[:, 2]) < 0.2          # wall-ish normals
    ang = np.arctan2(n[horiz, 1], n[horiz, 0])
    ang90 = np.mod(ang, np.pi / 2)          # fold to [0, 90deg)
    hist, edges = np.histogram(ang90, bins=180)
    k = int(np.argmax(hist))
    yaw = float((edges[k] + edges[k + 1]) / 2)
    if yaw > np.pi / 4:
        yaw -= np.pi / 2                    # smallest rotation
    return yaw, int(hist[k]), int(horiz.sum())


def render(xyz, rgb, out_png, crop=None, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(14, 14))
    order = np.argsort(xyz[:, 2])           # paint low first, high on top
    ax.scatter(xyz[order, 0], xyz[order, 1], s=0.05,
               c=rgb[order] if rgb is not None else xyz[order, 2],
               linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    xt = np.arange(np.floor(xyz[:, 0].min()), np.ceil(xyz[:, 0].max()) + 1)
    yt = np.arange(np.floor(xyz[:, 1].min()), np.ceil(xyz[:, 1].max()) + 1)
    ax.set_xticks(xt[::2]); ax.set_yticks(yt[::2])
    ax.grid(True, lw=0.3, alpha=0.5)
    if crop:
        x0, x1, y0, y1 = crop
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               fill=False, ec="red", lw=2))
        title += f"  crop=[{x0},{x1}]x[{y0},{y1}]"
    ax.set_title(title)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"render -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--crop", nargs=4, type=float, default=None,
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    ap.add_argument("--voxel", type=float, default=0.03)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    aligned = os.path.join(OUT, "aligned.ply")

    if a.render_only:
        pc = o3d.io.read_point_cloud(aligned)
        xyz = np.asarray(pc.points)
        rgb = np.asarray(pc.colors) if pc.has_colors() else None
    else:
        xyz, rgb = io.load(a.input)
        print(f"loaded {len(xyz):,} pts")
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(xyz)
        if rgb is not None:
            pc.colors = o3d.utility.Vector3dVector(rgb / 255.0
                                                   if rgb.max() > 1 else rgb)
        pc = pc.voxel_down_sample(a.voxel)
        print(f"downsampled -> {len(pc.points):,} pts @ {a.voxel}m")
        yaw, peak, nwall = estimate_yaw(pc)
        print(f"yaw = {np.degrees(yaw):.2f} deg "
              f"(peak {peak} of {nwall} wall normals)")
        R = pc.get_rotation_matrix_from_axis_angle([0, 0, -yaw])
        pc.rotate(R, center=(0, 0, 0))
        o3d.io.write_point_cloud(aligned, pc)
        np.save(os.path.join(OUT, "yaw.npy"), np.array([yaw]))
        print(f"aligned cloud -> {aligned}")
        xyz = np.asarray(pc.points)
        rgb = np.asarray(pc.colors) if pc.has_colors() else None

    # mid-height slab render hides ceiling so furniture is visible
    z0, z1 = np.percentile(xyz[:, 2], [1, 99])
    slab = (xyz[:, 2] > z0 + 0.1) & (xyz[:, 2] < z0 + 0.75 * (z1 - z0))
    render(xyz[slab], None if rgb is None else rgb[slab],
           os.path.join(OUT, "topdown.png"), crop=a.crop,
           title=f"8F cafe aligned top-down (slab {z0 + 0.1:.1f}"
                 f"-{z0 + 0.75 * (z1 - z0):.1f}m)")

    if a.crop:
        x0, x1, y0, y1 = a.crop
        m = ((xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) &
             (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1))
        cp = o3d.geometry.PointCloud()
        cp.points = o3d.utility.Vector3dVector(xyz[m])
        if rgb is not None:
            cp.colors = o3d.utility.Vector3dVector(rgb[m])
        out = os.path.join(OUT, "cafe_crop.ply")
        o3d.io.write_point_cloud(out, cp)
        print(f"crop {m.sum():,} pts -> {out}")


if __name__ == "__main__":
    main()
