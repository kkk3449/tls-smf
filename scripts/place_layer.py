#!/usr/bin/env python3
"""TOSM place layer: TIPS semanticPlace polygons + object membership + ring codes.

Implements the lab's place-modeling scheme on top of the verified object
layer: manually defined section polygons (TIPS semanticPlace.json, registered
into the scan frame), automatic object membership (point-in-polygon ->
isInsideOf), and per-place ring codes (member objects sorted by bearing
around the place centroid). Place semantics are derived from VERIFIED
objects only — the confidence gate propagates from the object layer to the
place layer.

  .venv/bin/python scripts/place_layer.py \
      --places "/path/semanticPlace.json" --transform map2visn2.npz \
      --objects outputs/vis_n2_det_filt/semanticObjects.lf_esc.room.json \
      --out-prefix outputs/place_layer_T2
"""
import argparse
import json
import math

import numpy as np


def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n - 1):
        x1, y1 = poly[i]
        x2, y2 = poly[i + 1]
        if (y1 > y) != (y2 > y):
            xt = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xt:
                inside = not inside
    return inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--places", required=True, help="TIPS semanticPlace.json")
    ap.add_argument("--transform", default=None,
                    help="npz with R,t,mean mapping map frame -> scan frame")
    ap.add_argument("--objects", required=True,
                    help="semanticObjects json (scan frame)")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--verified-only-ring", action="store_true", default=True)
    args = ap.parse_args()

    pd_ = json.load(open(args.places))
    if args.transform:
        z = np.load(args.transform)
        R, t, mean = z["R"], z["t"], z["mean"]
        tf = lambda p: ((np.asarray(p) - mean) @ R.T + t)
    else:
        tf = lambda p: np.asarray(p)

    places = []
    for p in pd_["semanticPlaces"]:
        poly = tf([[v["x"], v["y"]] for v in p["pose"]])
        places.append({"id": p["id"], "name": p["name"], "color": p["color"],
                       "polygon": [[round(float(a), 3), round(float(b), 3)]
                                   for a, b in poly],
                       "centroid": [round(float(poly[:-1, 0].mean()), 3),
                                    round(float(poly[:-1, 1].mean()), 3)]})

    d = json.load(open(args.objects))
    objs = d["semanticObjects"] if isinstance(d, dict) and \
        "semanticObjects" in d else d

    for o in objs:
        o["isInsideOf"] = None
        for p in places:
            if point_in_poly(o["poseX"], o["poseY"], p["polygon"]):
                o["isInsideOf"] = p["name"]
                break

    # ring codes: member objects sorted by bearing around the place centroid
    for p in places:
        members = [o for o in objs if o["isInsideOf"] == p["name"]]
        ring = []
        for o in members:
            st = o["properties"].get("verificationStatus", "")
            if args.verified_only_ring and not st.startswith("verified"):
                continue
            br = math.degrees(math.atan2(o["poseY"] - p["centroid"][1],
                                         o["poseX"] - p["centroid"][0])) % 360
            ring.append((br, o["type"], o["name"]))
        ring.sort()
        p["ringCode"] = [{"bearing": round(b, 1), "type": t, "name": n}
                         for b, t, n in ring]
        p["memberCount"] = len(members)
        p["verifiedMemberCount"] = len(ring)

    json.dump({"semanticPlaces": places},
              open(args.out_prefix + "_places.json", "w"),
              indent=1, ensure_ascii=False)

    # VDA5050-style exports (schema of the TIPS files)
    hdr = {"headerId": 0, "timestamp": "2026-07-22T12:00:00Z",
           "version": "1.0.0", "manufacturer": "CASELAB",
           "serialNumber": "BLK360-TOSM"}
    json.dump({**hdr, "semanticPlaces": [
        {"id": p["id"], "name": p["name"], "color": p["color"],
         "pose": [{"x": a, "y": b} for a, b in p["polygon"]]}
        for p in places]},
        open(args.out_prefix + "_semanticPlace.json", "w"),
        indent=1, ensure_ascii=False)
    json.dump({**hdr, "semanticObjects": [
        {"type": o["type"], "id": o["id"], "name": o["name"],
         "color": o.get("color"), "poseX": o["poseX"], "poseY": o["poseY"],
         "isInsideOf": o["isInsideOf"]} for o in objs]},
        open(args.out_prefix + "_semanticObject.json", "w"),
        indent=1, ensure_ascii=False)

    from collections import Counter
    print("membership:", dict(Counter(o["isInsideOf"] or "OUTSIDE"
                                      for o in objs)))
    for p in places:
        rc = ", ".join(f"{r['type']}" for r in p["ringCode"])
        print(f"  {p['name']}: {p['verifiedMemberCount']}/{p['memberCount']} "
              f"verified ring = [{rc}]")


if __name__ == "__main__":
    main()
