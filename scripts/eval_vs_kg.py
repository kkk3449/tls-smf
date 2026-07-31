#!/usr/bin/env python3
"""Evaluate a fresh pipeline run against the owner-corrected KG.

The live graph (rev14, 46 present nodes) embodies the accumulated owner
ground truth — restorations, refutations, label and geometry corrections.
This scores a re-run's detections against it:

  recall     which GT objects have a matching detection (centroid < --dist,
             no dimension test by default: fresh segmentation extents vary)
  precision  which detections match some GT object (rest = over-detections
             or structure noise the GT has refuted)
  type acc   canonical-synonym type match on the recalled pairs

  .venv/bin/python scripts/eval_vs_kg.py \
      --detections outputs/vis_sota_det5/semanticObjects.lf_esc.json \
      [--frame sota|n2] [--dist 0.6]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.build_det4 import canon  # noqa: E402

KG = os.path.join(ROOT, "outputs", "testroom_epochs_kg.json")
TRANSFORM = os.path.join(ROOT, "outputs", "vissota_to_visn2_T.npy")
ROOM_BOUNDS = os.path.join(ROOT, "outputs", "vis_n2_room_bounds.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", required=True)
    ap.add_argument("--frame", default="sota", choices=("sota", "n2"))
    ap.add_argument("--dist", type=float, default=0.6)
    args = ap.parse_args()

    det = json.load(open(args.detections))
    det = det["semanticObjects"] if isinstance(det, dict) else det
    T = np.load(TRANSFORM)
    P = []
    for r in det:
        p = np.array([r["poseX"], r["poseY"],
                      r.get("properties", {}).get("poseZ", 0.0)])
        if args.frame == "sota":
            p = T[:3, :3] @ p + T[:3, 3]
        P.append(p[:2])
    P = np.array(P)
    rb = json.load(open(ROOM_BOUNDS))
    cells = {tuple(c) for c in rb["cells"]}
    cm = rb["cell_m"]
    inroom = np.array([(int(np.floor(x / cm)), int(np.floor(y / cm)))
                       in cells for x, y in P])
    det_in = [(r, P[i]) for i, r in enumerate(det) if inroom[i]]
    print(f"{len(det)} detections, {len(det_in)} inside the test room")

    g = json.load(open(KG))
    gt = [n for n in g["nodes"] if n.get("presence") == "present"]
    recalled, type_ok, miss = [], 0, []
    used = set()
    for n in gt:
        c = np.array([n["pose"]["x"], n["pose"]["y"]])
        best = None
        for i, (r, p) in enumerate(det_in):
            if i in used:
                continue
            d = np.linalg.norm(c - p)
            if d < args.dist and (best is None or d < best[0]):
                best = (d, i, r)
        if best:
            used.add(best[1])
            r = best[2]
            ok = canon(r["type"]) == canon(n["type"])
            type_ok += ok
            recalled.append((n["name"], n["type"], r["name"], r["type"],
                             round(best[0], 2), ok))
        else:
            miss.append((n["name"], n["type"], n.get("status")))
    prec_match = len(used)
    print(f"\nGT recall: {len(recalled)}/{len(gt)} "
          f"({100*len(recalled)/len(gt):.0f}%)  "
          f"type-correct on recalled: {type_ok}/{len(recalled)}")
    print(f"detection precision (in-room): {prec_match}/{len(det_in)} "
          f"({100*prec_match/max(1,len(det_in)):.0f}%)")
    print("\nmissed GT objects:")
    for nm, ty, st in miss:
        print(f"  {nm:24s} {ty:18s} {st}")
    print("\nrecalled with WRONG type:")
    for nm, ty, rn, rt, d, ok in recalled:
        if not ok:
            print(f"  {nm:24s} GT={ty:16s} det={rn}({rt}) @{d}m")
    out = {"recall": [list(x) for x in recalled],
           "missed": [list(x) for x in miss],
           "n_det_inroom": len(det_in), "n_gt": len(gt)}
    op = args.detections.replace(".json", ".eval_vs_kg.json")
    json.dump(out, open(op, "w"), indent=1)
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()
