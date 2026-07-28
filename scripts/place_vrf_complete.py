#!/usr/bin/env python3
"""Complete the manual section polygons to a full partition of the room.

Voronoi-style completion in the spirit of occupancy-grid room segmentation
(Bormann et al., ICRA 2016; Luperto et al., RA-L 2022): every free cell of
the room mask is assigned to a place -- point-in-polygon where covered,
nearest polygon boundary elsewhere -- so the place layer tiles the room with
no OUTSIDE residue. Object membership and verified-only ring codes are then
recomputed under the completed partition.

  .venv/bin/python scripts/place_vrf_complete.py \
      --places outputs/place_layer_T3_places.json \
      --objects "outputs/vis_sota_det/semanticObjects.lf_esc.visn2frame.room.json" \
      --bounds outputs/vis_n2_room_bounds.json \
      --out outputs/place_layer_T3_vrf.json
"""
import argparse
import json
import math

import numpy as np


def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def seg_dist(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def poly_dist(x, y, poly):
    return min(seg_dist(x, y, *poly[i], *poly[(i + 1) % len(poly)])
               for i in range(len(poly)))


def assign(x, y, polys):
    for k, poly in enumerate(polys):
        if point_in_poly(x, y, poly):
            return k
    return int(np.argmin([poly_dist(x, y, poly) for poly in polys]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--places", required=True)
    ap.add_argument("--objects", required=True)
    ap.add_argument("--bounds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    places = json.load(open(args.places))["semanticPlaces"]
    objs = json.load(open(args.objects))["semanticObjects"]
    b = json.load(open(args.bounds))
    polys = [p["polygon"] for p in places]

    cells = np.array(b["cells"], dtype=float) * b["cell_m"]
    cell_place = [assign(cx, cy, polys) for cx, cy in cells]

    members = {k: [] for k in range(len(places))}
    changed = []
    for o in objs:
        k = assign(o["poseX"], o["poseY"], polys)
        members[k].append(o)
        if not point_in_poly(o["poseX"], o["poseY"], polys[k]):
            changed.append((o["name"], places[k]["name"]))

    out = {"semanticPlaces": [], "cell_m": b["cell_m"],
           "completion": "nearest-polygon Voronoi over room mask"}
    for k, p in enumerate(places):
        cel = cells[[i for i, cp in enumerate(cell_place) if cp == k]]
        cx, cy = (cel.mean(0) if len(cel) else p["centroid"])
        ring = []
        for o in members[k]:
            st = o["properties"].get("verificationStatus", "")
            if not st.startswith("verified"):
                continue
            brg = math.degrees(math.atan2(o["poseX"] - cx, o["poseY"] - cy)) % 360
            ring.append({"bearing": round(brg, 1), "type": o["type"],
                         "name": o["name"]})
        ring.sort(key=lambda r: r["bearing"])
        out["semanticPlaces"].append({
            "id": p["id"], "name": p["name"], "color": p["color"],
            "polygon": p["polygon"], "centroid": [round(cx, 3), round(cy, 3)],
            "cells": cel.round(3).tolist(), "ringCode": ring,
            "memberCount": len(members[k]),
            "verifiedMemberCount": len(ring)})
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}: cells {len(cells)}, newly captured objects "
          f"{len(changed)}: {[c[0] for c in changed]}")


if __name__ == "__main__":
    main()
