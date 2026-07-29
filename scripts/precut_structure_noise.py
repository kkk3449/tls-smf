#!/usr/bin/env python3
"""Point-level structure-noise pre-cut, ahead of re-segmentation.

Two cuts that are safe at point level (they cannot eat wall-flush
furniture, unlike a wall-line band cut):
  - ceiling band:  z above floor + --ceil-above  (fluorescent fixtures,
                   ceiling noise, upper wall bands; tallest real object in
                   the room is ~2.2 m)
  - outside room:  points beyond the exploration grid map's outer room
                   polygon (door bleed-through)
Wall sheets are left to the cluster-level hybrid filter after DBSCAN.

  .venv/bin/python scripts/precut_structure_noise.py \
      --input outputs/vis_sota_det/clean.ply \
      --transform outputs/vissota_to_visn2_T.npy \
      --map /home/caselab/ammr_twin/map_vis_n2_1.yaml \
      --out outputs/vis_sota_det2/clean_precut.ply
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.structure_noise_filter import load_gridmap  # noqa: E402


def main():
    from scipy.ndimage import binary_fill_holes
    import open3d as o3d

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--transform", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--ceil-above", type=float, default=2.4,
                    help="cut points more than this above the floor (m)")
    ap.add_argument("--floor-pct", type=float, default=1.0,
                    help="floor = this z-percentile of the input cloud")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pc = o3d.io.read_point_cloud(args.input)
    xyz = np.asarray(pc.points)
    rgb = np.asarray(pc.colors)
    n0 = len(xyz)
    floor_z = float(np.percentile(xyz[:, 2], args.floor_pct))

    T = np.load(args.transform)
    img, res, ox, oy = load_gridmap(args.map)
    occ = img < 100
    free = img > 200
    room = binary_fill_holes(free | occ)

    m = (T[:3, :3] @ xyz.T).T + T[:3, 3]
    ci = np.clip(((m[:, 0] - ox) / res).astype(int), 0, room.shape[1] - 1)
    cj = np.clip(((m[:, 1] - oy) / res).astype(int), 0, room.shape[0] - 1)
    inside = room[cj, ci]

    ceil_cut = xyz[:, 2] > floor_z + args.ceil_above
    keep = inside & ~ceil_cut
    print(f"floor z={floor_z:.3f}; cut ceiling-band {int(ceil_cut.sum()):,} "
          f"+ outside-room {int((~inside).sum()):,} "
          f"-> keep {int(keep.sum()):,}/{n0:,}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(xyz[keep].astype(np.float64))
    out.colors = o3d.utility.Vector3dVector(rgb[keep].astype(np.float64))
    o3d.io.write_point_cloud(args.out, out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
