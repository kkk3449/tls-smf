#!/usr/bin/env python3
"""Segmentation parameter sensitivity (paper Sec. 5.6 / reviewer defense of
the 2.5 m giant-split rule and DBSCAN parameters).

Grid: eps x min_points x split_footprint, clustered on the SAME clean cloud
the audited run used (exact point identity), scored against the audited
instance set (obj_*.ply of the kept clusters, restricted to GT-scored ids):

  recovered : one cluster covers the instance with IoU >= 0.5
  split     : >= 2 clusters each hold >= 20 %% of the instance's points
  merged    : best cluster also holds >= 20 %% of another instance
  missing   : neither
  precision : matched clusters / all clusters (>= min_points)
  recall    : recovered / instances

  .venv/bin/python scripts/sensitivity_sweep.py \
      --clean outputs/showroom_det_filt/clean.ply \
      --ref-objects outputs/showroom_det_filt \
      --gt outputs/showroom_gt.json \
      --out outputs/sensitivity_showroom.csv
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from blk360seg import io, postprocess  # noqa: E402

EPS_GRID = (0.25, 0.30, 0.35)
MINPTS_GRID = (50, 100, 150)
FOOT_GRID = (1.5, 2.0, 2.5, 3.0)


def key_of(xyz):
    """Exact-coordinate hash per point (clean cloud is shared verbatim)."""
    q = np.round(np.asarray(xyz, dtype=np.float64) * 1e4).astype(np.int64)
    return q[:, 0] * 73856093 ^ q[:, 1] * 19349663 ^ q[:, 2] * 83492791


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--ref-objects", required=True,
                    help="dir with the audited obj_*.ply instances")
    ap.add_argument("--gt", required=True,
                    help="GT json; instances limited to scored ids "
                         "(not artifact / not out_of_scope / not unknown)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    xyz, rgb = io.load(args.clean)
    keys = key_of(xyz)
    order = np.argsort(keys)
    skeys = keys[order]

    gd = json.load(open(args.gt))
    glist = gd["objects"] if isinstance(gd, dict) else gd
    scored = {str(g["id"]) for g in glist
              if not g.get("structure_artifact") and not g.get("out_of_scope")
              and g.get("gt_quality") in ("confirmed", "plausible")}

    import open3d as o3d
    inst_of = np.full(len(xyz), -1, dtype=np.int32)   # point -> instance idx
    inst_ids = []
    for f in sorted(os.listdir(args.ref_objects)):
        if not (f.startswith("obj_") and f.endswith(".ply")):
            continue
        oid = str(int(f[4:8]))
        if oid not in scored:
            continue
        op = np.asarray(o3d.io.read_point_cloud(
            os.path.join(args.ref_objects, f)).points)
        ok = key_of(op)
        pos = np.searchsorted(skeys, ok)
        pos = np.clip(pos, 0, len(skeys) - 1)
        hit = skeys[pos] == ok
        inst_of[order[pos[hit]]] = len(inst_ids)
        inst_ids.append(oid)
    n_inst = len(inst_ids)
    print(f"[sweep] {n_inst} scored reference instances, "
          f"{(inst_of >= 0).sum():,} labeled points / {len(xyz):,}")

    rows = []
    for eps in EPS_GRID:
        for mp in MINPTS_GRID:
            for foot in FOOT_GRID:
                t0 = time.time()
                inst, n_obj, n_split = postprocess.cluster_all_split(
                    xyz, eps_m=eps, min_points=mp, split_footprint_m=foot,
                    split_eps_m=0.20, split_min_points=mp)
                dt = time.time() - t0
                labs = np.unique(inst[inst >= 0])
                # overlap table cluster x instance
                stats = {i: {} for i in range(n_inst)}
                for c in labs:
                    m = inst == c
                    if int(m.sum()) < mp:
                        continue
                    gi, cnt = np.unique(inst_of[m], return_counts=True)
                    for g, n in zip(gi, cnt):
                        if g >= 0:
                            stats[int(g)][int(c)] = int(n)
                inst_pts = np.bincount(inst_of[inst_of >= 0],
                                       minlength=n_inst)
                cl_pts = {int(c): int((inst == c).sum()) for c in labs}
                rec = spl = mer = mis = 0
                matched_clusters = set()
                for g in range(n_inst):
                    ov = stats[g]
                    tot = inst_pts[g]
                    if not ov or tot == 0:
                        mis += 1
                        continue
                    big = [c for c, n in ov.items() if n >= 0.2 * tot]
                    best_c, best_n = max(ov.items(), key=lambda kv: kv[1])
                    iou = best_n / (tot + cl_pts[best_c] - best_n)
                    other = any(n >= 0.2 * inst_pts[g2]
                                for g2 in range(n_inst) if g2 != g
                                for c2, n in stats[g2].items()
                                if c2 == best_c)
                    if len(big) >= 2:
                        spl += 1
                    elif other:
                        mer += 1
                    elif iou >= 0.5:
                        rec += 1
                        matched_clusters.add(best_c)
                    else:
                        mis += 1
                n_clusters = len([c for c in labs if cl_pts[int(c)] >= mp])
                rows.append({
                    "eps": eps, "min_points": mp, "footprint": foot,
                    "n_objects": n_obj, "n_split_ops": n_split,
                    "recovered": rec, "split": spl, "merged": mer,
                    "missing": mis,
                    "recall": round(rec / n_inst, 3),
                    "precision": round(len(matched_clusters)
                                       / max(n_clusters, 1), 3),
                    "coverage_pct": round(100 * (inst >= 0).mean(), 1),
                    "runtime_s": round(dt, 2),
                })
                print(f"  eps={eps} mp={mp} foot={foot}: obj={n_obj} "
                      f"rec={rec}/{n_inst} spl={spl} mer={mer} mis={mis} "
                      f"({dt:.1f}s)", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[sweep] wrote {args.out}")


if __name__ == "__main__":
    main()
