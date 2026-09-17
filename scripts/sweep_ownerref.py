#!/usr/bin/env python3
"""Revision-1: DBSCAN sensitivity sweep re-scored against an INDEPENDENT,
owner-grounded object-level reference (reviewer: the original sweep scored
against the operating-point segmentation's own instance set -> circular).

Reference construction (per scene):
  * reference OBJECT LIST = the owner-audited inventory: vis_n2_gt.json /
    showroom_gt.json entries with gt_quality in {confirmed, plausible}, not
    structure_artifact, not out_of_scope (room-scoped).  These entries carry
    owner-audited poses + oriented dimensions.
  * point support, two variants scored side by side:
      - owner_clusters : the audited cluster point files
          vis_n2   -> outputs/vis_n2_det_run1/obj_*.ply  (42 objects)
          showroom -> outputs/stage2_no_wall_objects_split/obj_*.ply
                      (the owner-audited 57-set; 48 scored objects -- 15 MORE
                      than the 33 the original sweep could score, because the
                      original was limited to the operating run's own files)
      - owner_box : membership-independent support = ALL clean-cloud points
          inside the owner-audited oriented box (center = audited poseX/Y/Z,
          size = audited length/width/height + 2 cm margin, yaw = PCA yaw of
          the audited cluster, i.e. the same yaw convention under which the
          owner audited the dimensions; a point in several boxes goes to the
          smallest-volume box).  This support is NOT a cluster membership set:
          it can contain points the operating point dropped as noise or gave
          to a neighbour, so no parameter combination is trivially IoU = 1.

Sweep grid + scoring thresholds are IDENTICAL to scripts/sensitivity_sweep.py
(eps 0.25/0.30/0.35 x min_points 50/100/150 x footprint 1.5/2.0/2.5/3.0;
recovered IoU >= 0.5, fragments >= 20 % -> split/merged), so the tables are
directly comparable.

  .venv/bin/python scripts/sweep_ownerref.py \
      --out-json outputs/revision1_tables/sweep_ownerref.json \
      --out-csv  outputs/revision1_tables/sweep_ownerref.csv

Deterministic: clustering is Open3D DBSCAN on a cached clean cloud (no RNG);
numpy seed fixed anyway.
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
from blk360seg.semantic_object import yaw_and_footprint  # noqa: E402

EPS_GRID = (0.25, 0.30, 0.35)
MINPTS_GRID = (50, 100, 150)
FOOT_GRID = (1.5, 2.0, 2.5, 3.0)
OP = (0.30, 100, 2.5)          # operating point
BOX_MARGIN = 0.02              # m, on every owner-box face

SCENES = {
    "vis_n2": {
        "clean": "outputs/vis_n2_det_filt/clean.ply",
        "ref_objects": "outputs/vis_n2_det_run1",
        "gt": "outputs/vis_n2_gt.json",
    },
    "showroom": {
        "clean": "outputs/showroom_det_filt/clean.ply",
        "ref_objects": "outputs/stage2_no_wall_objects_split",
        "gt": "outputs/showroom_gt.json",
    },
}


def key_of(xyz):
    q = np.round(np.asarray(xyz, dtype=np.float64) * 1e4).astype(np.int64)
    return q[:, 0] * 73856093 ^ q[:, 1] * 19349663 ^ q[:, 2] * 83492791


def scored_entries(gt_path):
    gd = json.load(open(gt_path))
    glist = gd["objects"] if isinstance(gd, dict) else gd
    return [g for g in glist
            if not g.get("structure_artifact") and not g.get("out_of_scope")
            and g.get("gt_quality") in ("confirmed", "plausible")]


def score(inst, inst_of, n_inst, mp):
    """EXACT replica of the scoring loop in scripts/sensitivity_sweep.py."""
    labs = np.unique(inst[inst >= 0])
    stats = {i: {} for i in range(n_inst)}
    for c in labs:
        m = inst == c
        if int(m.sum()) < mp:
            continue
        gi, cnt = np.unique(inst_of[m], return_counts=True)
        for g, n in zip(gi, cnt):
            if g >= 0:
                stats[int(g)][int(c)] = int(n)
    inst_pts = np.bincount(inst_of[inst_of >= 0], minlength=n_inst)
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
    return dict(recovered=rec, split=spl, merged=mer, missing=mis,
                recall=round(rec / n_inst, 3),
                precision=round(len(matched_clusters) / max(n_clusters, 1), 3),
                n_clusters=n_clusters)


def build_references(scene, paths):
    import open3d as o3d
    xyz, _ = io.load(os.path.join(ROOT, paths["clean"]))
    keys = key_of(xyz)
    order = np.argsort(keys)
    skeys = keys[order]

    ents = scored_entries(os.path.join(ROOT, paths["gt"]))
    inst_cl = np.full(len(xyz), -1, dtype=np.int32)   # owner_clusters support
    inst_bx = np.full(len(xyz), -1, dtype=np.int32)   # owner_box support
    box_vol = np.full(len(xyz), np.inf)               # smallest-box tiebreak
    meta = []
    for gi, g in enumerate(sorted(ents, key=lambda e: int(e["id"]))):
        oid = int(g["id"])
        f = os.path.join(ROOT, paths["ref_objects"], f"obj_{oid:04d}.ply")
        op = np.asarray(o3d.io.read_point_cloud(f).points)
        ok = key_of(op)
        pos = np.clip(np.searchsorted(skeys, ok), 0, len(skeys) - 1)
        hit = skeys[pos] == ok
        inst_cl[order[pos[hit]]] = gi

        # owner box: audited pose/dims, yaw refit with the audit's convention
        yaw, fit_l, fit_w = yaw_and_footprint(op)
        L = g["dimensions"]["length"] + 2 * BOX_MARGIN
        W = g["dimensions"]["width"] + 2 * BOX_MARGIN
        H = g["dimensions"]["height"] + 2 * BOX_MARGIN
        cx, cy = g["poseX"], g["poseY"]
        cz = g.get("properties", {}).get("poseZ", float(op[:, 2].mean()))
        ca, sa = np.cos(-yaw), np.sin(-yaw)
        dx, dy = xyz[:, 0] - cx, xyz[:, 1] - cy
        u = dx * ca - dy * sa
        v = dx * sa + dy * ca
        inside = ((np.abs(u) <= L / 2) & (np.abs(v) <= W / 2)
                  & (np.abs(xyz[:, 2] - cz) <= H / 2))
        vol = L * W * H
        take = inside & (vol < box_vol)
        inst_bx[take] = gi
        box_vol[take] = vol
        meta.append({
            "gt_id": oid, "gt_type": g.get("gt_type"),
            "gt_quality": g.get("gt_quality"),
            "n_cluster_pts": int(hit.sum()),
            "n_box_pts": int(inside.sum()),
            "dims_owner": [round(g["dimensions"]["length"], 3),
                           round(g["dimensions"]["width"], 3),
                           round(g["dimensions"]["height"], 3)],
            "dims_refit_lw": [round(fit_l, 3), round(fit_w, 3)],
        })
    for m in meta:  # box support after smallest-box assignment
        pass
    counts = np.bincount(inst_bx[inst_bx >= 0], minlength=len(meta))
    for gi, m in enumerate(meta):
        m["n_box_pts_assigned"] = int(counts[gi])
    return xyz, {"owner_clusters": inst_cl, "owner_box": inst_bx}, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json",
                    default="outputs/revision1_tables/sweep_ownerref.json")
    ap.add_argument("--out-csv",
                    default="outputs/revision1_tables/sweep_ownerref.csv")
    ap.add_argument("--scenes", nargs="+", default=["vis_n2", "showroom"])
    args = ap.parse_args()
    np.random.seed(0)
    oj = os.path.join(ROOT, args.out_json)
    oc = os.path.join(ROOT, args.out_csv)
    for p in (oj, oc):
        if os.path.exists(p):
            sys.exit(f"[ownerref] refusing to overwrite existing {p}")

    rows, meta_all = [], {}
    for scene in args.scenes:
        paths = SCENES[scene]
        xyz, refs, meta = build_references(scene, paths)
        meta_all[scene] = meta
        n_inst = len(meta)
        print(f"[ownerref] {scene}: {n_inst} owner-audited reference objects, "
              f"clean cloud {len(xyz):,} pts")
        for name, inst_of in refs.items():
            print(f"    {name}: {(inst_of >= 0).sum():,} support points")
        for eps in EPS_GRID:
            for mp in MINPTS_GRID:
                for foot in FOOT_GRID:
                    t0 = time.time()
                    inst, n_obj, n_split = postprocess.cluster_all_split(
                        xyz, eps_m=eps, min_points=mp,
                        split_footprint_m=foot,
                        split_eps_m=0.20, split_min_points=mp)
                    dt = time.time() - t0
                    for ref_name, inst_of in refs.items():
                        s = score(inst, inst_of, n_inst, mp)
                        rows.append({
                            "scene": scene, "reference": ref_name,
                            "n_ref": n_inst,
                            "eps": eps, "min_points": mp, "footprint": foot,
                            "n_objects": n_obj, "n_split_ops": n_split,
                            **s,
                            "coverage_pct": round(100 * (inst >= 0).mean(), 1),
                            "runtime_s": round(dt, 2),
                        })
                    r = [x for x in rows[-2:]]
                    print(f"  eps={eps} mp={mp} foot={foot}: obj={n_obj} | "
                          + " | ".join(
                              f"{x['reference']}: rec={x['recovered']}/"
                              f"{n_inst} spl={x['split']} mer={x['merged']} "
                              f"mis={x['missing']}" for x in r)
                          + f" ({dt:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(oc), exist_ok=True)
    df.to_csv(oc, index=False)
    out = {
        "date": time.strftime("%Y-%m-%d"),
        "construction": {
            "object_list": "owner-audited inventory: gt_quality in "
                           "{confirmed, plausible}, in-scope, non-artifact",
            "owner_clusters": "point support = audited cluster files "
                              "(vis_n2: vis_n2_det_run1; showroom: the "
                              "audited 57-set stage2_no_wall_objects_split)",
            "owner_box": "point support = clean-cloud points inside the "
                         "owner-audited oriented box (pose + dims + "
                         f"{BOX_MARGIN*100:.0f} cm margin, PCA yaw); "
                         "multi-box points -> smallest box; membership-"
                         "independent of any clustering run",
            "scoring": "identical to scripts/sensitivity_sweep.py: "
                       "IoU>=0.5 recovered; >=20% fragments split/merged",
            "grid": {"eps": EPS_GRID, "min_points": MINPTS_GRID,
                     "footprint": FOOT_GRID},
            "operating_point": {"eps": OP[0], "min_points": OP[1],
                                "footprint": OP[2]},
        },
        "reference_objects": meta_all,
        "rows": rows,
    }
    json.dump(out, open(oj, "w"), indent=1)
    print(f"[ownerref] wrote {oc}\n[ownerref] wrote {oj}")


if __name__ == "__main__":
    main()
