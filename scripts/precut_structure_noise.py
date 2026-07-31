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
      --room-bounds outputs/vis_n2_room_bounds.json \
      --out outputs/vis_sota_det2/clean_precut.ply

--room-bounds is the STRICT test-room cut (owner 2026-08-01: room scoping
belongs at the point level, before DBSCAN — without it, corridor/glazing
bleed forms dozens of out-of-room clusters that every later stage must
re-filter).
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
    ap.add_argument("--room-bounds", default=None,
                    help="room-mask json (cells + cell_m); points outside "
                         "the mask dilated by --room-margin are cut (door/"
                         "window bleed past the room shell)")
    ap.add_argument("--room-margin", type=float, default=0.35,
                    help="dilation (m) of the room mask, so the wall shell "
                         "itself survives the cut")
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

    if args.room_bounds:
        import json
        from scipy.ndimage import binary_dilation
        rb = json.load(open(args.room_bounds))
        cells = np.array(rb["cells"], dtype=int)
        cm = rb["cell_m"]
        x0, y0 = cells[:, 0].min() - 4, cells[:, 1].min() - 4
        mask = np.zeros((cells[:, 1].max() - y0 + 5,
                         cells[:, 0].max() - x0 + 5), dtype=bool)
        mask[cells[:, 1] - y0, cells[:, 0] - x0] = True
        it = max(1, int(round(args.room_margin / cm)))
        mask = binary_dilation(mask, iterations=it)
        mi = np.clip((m[:, 0] / cm).astype(int) - x0, 0, mask.shape[1] - 1)
        mj = np.clip((m[:, 1] / cm).astype(int) - y0, 0, mask.shape[0] - 1)
        in_room = mask[mj, mi]
        print(f"room-bounds cut: {int((keep & ~in_room).sum()):,} points "
              f"beyond the room shell (+{args.room_margin} m margin)")
        keep &= in_room

    print(f"floor z={floor_z:.3f}; cut ceiling-band {int(ceil_cut.sum()):,} "
          f"+ outside-grid {int((~inside).sum()):,} "
          f"-> keep {int(keep.sum()):,}/{n0:,}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(xyz[keep].astype(np.float64))
    out.colors = o3d.utility.Vector3dVector(rgb[keep].astype(np.float64))
    o3d.io.write_point_cloud(args.out, out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
