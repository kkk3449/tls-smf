#!/usr/bin/env python3
"""Automatic place segmentation of the room occupancy grid via SLIC.

Follows the DK-SMF place-modeling recipe: the free-space occupancy grid is
partitioned into a parameterized number of superpixels with SLIC
(Achanta et al., TPAMI 2012); each superpixel becomes a candidate place
region. Object membership and verified-only ring codes are computed per
epoch on the shared segmentation, so place identity is stable across scans.

  .venv/bin/python scripts/place_slic_segment.py \
      --bounds outputs/vis_n2_room_bounds.json --n-segments 7 \
      --epoch T3 --objects "outputs/vis_sota_det/semanticObjects.lf_esc.visn2frame.room.json" \
      --out outputs/place_layer_T3_slic.json
"""
import argparse
import json
import math

import numpy as np
from skimage.segmentation import slic

PALETTE = ["21, 128, 16", "185, 28, 28", "37, 99, 235", "217, 119, 6",
           "124, 58, 237", "13, 148, 136", "219, 39, 119", "101, 163, 13",
           "234, 88, 12", "8, 145, 178"]


def load_gridmap(yaml_path):
    """Nav-stack occupancy grid (pgm+yaml) -> free-space mask + geometry."""
    import os
    import re
    meta = {}
    for line in open(yaml_path):
        m = re.match(r"(\w+):\s*(.+)", line.strip())
        if m:
            meta[m.group(1)] = m.group(2)
    res = float(meta["resolution"])
    ox, oy = [float(v) for v in
              meta["origin"].strip("[]").split(",")[:2]]
    pgm = os.path.join(os.path.dirname(yaml_path), meta["image"])
    from PIL import Image
    img = np.asarray(Image.open(pgm))
    free = img >= 250          # trinary: 254 free / 205 unknown / 0 occupied
    free = np.flipud(free)     # pgm row 0 = top; map row 0 = bottom
    return free, res, ox, oy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True,
                    help="nav-stack occupancy grid .yaml (LiDAR-grade map)")
    ap.add_argument("--bounds", required=True,
                    help="room mask json used to scope the map to the room")
    ap.add_argument("--objects", required=True)
    ap.add_argument("--epoch", required=True)
    ap.add_argument("--n-segments", type=int, default=7,
                    help="parameterized superpixel count (DK-SMF style)")
    ap.add_argument("--compactness", type=float, default=0.05)
    ap.add_argument("--min-area-m2", type=float, default=1.5,
                    help="discard regions/fragments smaller than this")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    free, res, ox, oy = load_gridmap(args.map)
    b = json.load(open(args.bounds))
    bcm = b["cell_m"]
    room = {tuple(c) for c in b["cells"]}

    H, W = free.shape
    ys, xs = np.nonzero(free)
    keep = np.zeros_like(free)
    for y, x in zip(ys, xs):
        wx, wy = ox + (x + 0.5) * res, oy + (y + 0.5) * res
        if (int(round(wx / bcm)), int(round(wy / bcm))) in room:
            keep[y, x] = True
    mask = keep
    cm = res
    x0 = y0 = 0            # world coords come from map origin below
    img = mask.astype(float)

    def cell_world(x_idx, y_idx):
        return ox + (x_idx + 0.5) * res, oy + (y_idx + 0.5) * res

    seg = slic(img, n_segments=args.n_segments, compactness=args.compactness,
               mask=mask, channel_axis=None, start_label=1)

    # DK-SMF post-processing: keep connected free-space regions, discard any
    # region (or disconnected fragment) whose area falls below the threshold.
    from skimage.measure import label as cc_label
    min_cells = int(args.min_area_m2 / (res * res))
    clean = np.zeros_like(seg)
    nxt = 1
    for lb in sorted(set(seg[seg > 0].tolist())):
        comp = cc_label(seg == lb)
        for c in range(1, comp.max() + 1):
            m = comp == c
            if m.sum() >= min_cells:
                clean[m] = nxt
                nxt += 1
    seg = clean
    labels = sorted(set(seg[seg > 0].tolist()))

    objs = json.load(open(args.objects))["semanticObjects"]
    lab_ys, lab_xs = np.nonzero(seg)

    def region_of(x, y):
        ci = int(round((x - ox) / res - 0.5))
        cj = int(round((y - oy) / res - 0.5))
        ci = min(max(ci, 0), W - 1)
        cj = min(max(cj, 0), H - 1)
        if seg[cj, ci] > 0:
            return seg[cj, ci]
        # nearest labeled cell (objects can anchor on top of obstacles)
        k = np.argmin((lab_xs - ci) ** 2 + (lab_ys - cj) ** 2)
        return seg[lab_ys[k], lab_xs[k]]

    members = {lb: [] for lb in labels}
    for o in objs:
        members[region_of(o["poseX"], o["poseY"])].append(o)

    out = {"semanticPlaces": [], "cell_m": cm,
           "method": f"SLIC superpixels (n_segments={args.n_segments}, "
                     f"compactness={args.compactness}) over the nav-stack "
                     f"occupancy grid free space ({res} m)"}
    for i, lb in enumerate(labels):
        ys2, xs2 = np.nonzero(seg == lb)
        cel = np.stack([ox + (xs2 + 0.5) * res, oy + (ys2 + 0.5) * res],
                       axis=1)
        cx, cy = cel.mean(0)
        ring = []
        for o in members[lb]:
            st = o["properties"].get("verificationStatus", "")
            if not st.startswith("verified"):
                continue
            brg = math.degrees(math.atan2(o["poseX"] - cx, o["poseY"] - cy)) % 360
            ring.append({"bearing": round(brg, 1), "type": o["type"],
                         "name": o["name"]})
        ring.sort(key=lambda r: r["bearing"])
        out["semanticPlaces"].append({
            "id": str(lb), "name": f"region_{lb:03d}",
            "color": PALETTE[i % len(PALETTE)],
            "centroid": [round(float(cx), 3), round(float(cy), 3)],
            "cells": cel.round(3).tolist(), "ringCode": ring,
            "memberCount": len(members[lb]),
            "verifiedMemberCount": len(ring)})
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}: {len(labels)} regions, "
          f"members {[len(members[lb]) for lb in labels]}")


if __name__ == "__main__":
    main()
