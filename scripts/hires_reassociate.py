#!/usr/bin/env python3
"""Re-associate full-resolution points to Stage-A clusters (render-source
ablation, paper Sec. 5.8): nearest working-cloud cluster point within
--radius owns each full-res point.

  .venv/bin/python scripts/hires_reassociate.py \
      --objects-dir outputs/stage2_no_wall_objects_split \
      --source ../testroom_no_wall/stage2_no_wall.e57 \
      --out outputs/s2_split_hires_objs
"""
import argparse
import glob
import os
import sys

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blk360seg import io  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects-dir", required=True)
    ap.add_argument("--source", required=True, help=".e57 or .ply full cloud")
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=float, default=0.045)
    ap.add_argument("--max-pts", type=int, default=400_000)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.objects_dir, "obj_*.ply")))
    pts, lab = [], []
    for i, f in enumerate(files):
        p = np.asarray(o3d.io.read_point_cloud(f).points)
        pts.append(p)
        lab.append(np.full(len(p), i))
    seed = np.vstack(pts)
    seedlab = np.concatenate(lab)
    tree = cKDTree(seed)
    print(f"seed {len(seed):,} pts from {len(files)} clusters")

    xyz, rgb = io.load(a.source)
    print(f"source {len(xyz):,} pts")
    if rgb is not None and rgb.max() > 1:
        rgb = rgb / 255.0

    owner = np.full(len(xyz), -1, np.int32)
    CH = 4_000_000
    for s in range(0, len(xyz), CH):
        d, idx = tree.query(xyz[s:s + CH], k=1,
                            distance_upper_bound=a.radius, workers=-1)
        ok = np.isfinite(d)
        owner[s:s + CH][ok] = seedlab[idx[ok]]

    os.makedirs(a.out, exist_ok=True)
    for i, f in enumerate(files):
        m = owner == i
        q = o3d.geometry.PointCloud()
        x, c = xyz[m], (rgb[m] if rgb is not None else None)
        if len(x) > a.max_pts:
            sel = np.random.RandomState(0).choice(len(x), a.max_pts,
                                                  replace=False)
            x, c = x[sel], (c[sel] if c is not None else None)
        q.points = o3d.utility.Vector3dVector(x)
        if c is not None:
            q.colors = o3d.utility.Vector3dVector(c)
        o3d.io.write_point_cloud(os.path.join(a.out, os.path.basename(f)), q)
        print(os.path.basename(f), int(m.sum()), "->", len(x))


if __name__ == "__main__":
    main()
