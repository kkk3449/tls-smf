#!/usr/bin/env python3
"""Cheap tuning sweep for the DBSCAN placeholder + class-set choice.

Loads the cleaned cloud ONCE and Uni3D ONCE, then for each (eps, min_points)
re-clusters in memory, classifies every object, and reports how the clutter
rate / object count / mean confidence move. Also tries the class set with and
without the "clutter" catch-all.

  python scripts/tune_segmentation.py --input ../testroom_no_wall/stage2_no_wall.e57

Nothing is written to disk except a sweep_results.csv summary; pick the winning
row and re-run extract_objects.py + classify_objects.py with it.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from blk360seg import io, preprocess, postprocess, objects, classes  # noqa: E402
from blk360seg.uni3d_classifier import Uni3DClassifier               # noqa: E402

EPS_GRID = [0.12, 0.15, 0.20, 0.25, 0.30]
MINPTS_GRID = [100, 150]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default=os.path.join(here, "..", "configs", "default.yaml"))
    ap.add_argument("--uni3d-ckpt", default=os.path.join(ROOT, "weights", "uni3d-b.pt"))
    ap.add_argument("--clip-ckpt", default=os.path.join(ROOT, "weights", "eva02_e_clip.bin"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    print(f"[tune] loading + downsampling {args.input}")
    xyz, rgb = io.load(args.input)
    xyz, rgb = preprocess.voxel_downsample(xyz, rgb, cfg["preprocess"]["voxel_size_m"])
    print(f"[tune] {len(xyz):,} points after downsample (no remove-structure)")

    print("[tune] loading Uni3D + CLIP (once)...")
    clf = Uni3DClassifier(args.uni3d_ckpt, args.clip_ckpt, device=args.device)
    cls_full = classes.get_classes("industrial")
    cls_noclutter = [c for c in cls_full if c != "clutter"]

    rows = []
    for eps in EPS_GRID:
        for mp in MINPTS_GRID:
            inst = postprocess.cluster_all(xyz, eps_m=eps, min_points=mp)
            objs = objects.extract_objects(xyz, rgb, inst, min_points=mp)
            nobj = len(objs)
            if nobj == 0:
                continue
            covered = sum(o["n"] for o in objs)
            for setname, cset in (("with_clutter", cls_full),
                                  ("no_clutter", cls_noclutter)):
                clf.set_classes(cset)
                top1, scores = [], []
                for o in objs:
                    preds = clf.classify(o["xyz"], o["rgb"], topk=1)
                    top1.append(preds[0][0]); scores.append(preds[0][1])
                clutter = sum(1 for t in top1 if t == "clutter")
                ndistinct = len(set(top1))
                rows.append({
                    "eps": eps, "min_pts": mp, "class_set": setname,
                    "n_obj": nobj, "coverage_pts": covered,
                    "clutter": clutter,
                    "clutter_pct": round(100 * clutter / nobj, 1),
                    "distinct_classes": ndistinct,
                    "mean_score": round(float(np.mean(scores)), 3),
                })
                print(f"  eps={eps:<4} mp={mp:<3} {setname:<12} "
                      f"n_obj={nobj:<3} clutter={clutter:<3} "
                      f"({rows[-1]['clutter_pct']:>5}%) distinct={ndistinct:<2} "
                      f"mean={rows[-1]['mean_score']}")

    out = os.path.join(ROOT, "outputs", "sweep_results.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n[tune] wrote {out}")
    df = pd.DataFrame(rows)
    print("\n[tune] best by lowest clutter%% (with_clutter set):")
    sub = df[df.class_set == "with_clutter"].sort_values("clutter_pct")
    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
