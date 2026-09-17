#!/usr/bin/env python3
"""Revision-1: voxel-resolution and registration-noise robustness of the
Stage-A geometric backbone on the robot-hall scan (vis_n2).

Reviewer: "controlled experiments regarding registration noise, voxel
resolution ... should be included" (geometric axis; colour is covered by the
render-source ablation elsewhere).

(a) voxel sweep   : full backbone at voxel_size 0.02 / 0.03 (baseline) / 0.05 m
(b) noise sweep   : additive per-point iid Gaussian noise (seeded),
                    sigma = 5 / 10 / 20 mm applied to the RAW cloud BEFORE the
                    backbone, backbone at baseline parameters.

    Noise regimes probed (stated per row in the output):
      sigma  5 mm ~ 1x the BLK360 per-scan bundle registration error
                    (4-5 mm, the within-bundle multi-setup regime);
      sigma 10 mm ~ 2x bundle error;
      sigma 20 mm ~ 4x bundle error -- still well BELOW the ~5 cm inter-epoch
                    ICP residual, which the pipeline handles at the KG object-
                    matching level, not at the point level.

Backbone per run (identical to the audited vis_n2_det pipeline; verified: the
baseline run's 0.03 m downsample reproduces outputs/vis_n2_det_filt/
ceilingref.ply EXACTLY, 989,021 pts, 100 % coordinate-key overlap):
  voxel_downsample -> seeded RANSAC structure removal -> DBSCAN
  cluster_all_split (eps 0.30, min_points 100, footprint 2.5, split 0.20/100)
  -> ceiling-soffit remnant filter (pre-removal cloud as reference).

Scoring: same owner-grounded reference + thresholds as
scripts/sweep_ownerref.py (owner_clusters variant: the 42 owner-audited
confirmed+plausible in-scope objects of outputs/vis_n2_gt.json with point
support from outputs/vis_n2_det_run1). Because non-baseline runs have
different point coordinates, reference labels are transferred by nearest
neighbour on the pre-removal reference cloud (label -1 = structure/background
competes, limiting bleed) with tolerance
  tol = max(1.5 * voxel, 0.045) + 2 * sigma   [m]
then recovered / split / merged / missing exactly as the sensitivity sweep
(IoU >= 0.5 recovered, >= 20 % fragments split/merged).

  .venv/bin/python scripts/robustness_voxel_noise.py \
      --raw "/home/caselab/Downloads/DataSend/vis_n2 1.e57"

Seeded end to end: RANSAC seed 0 (pipeline default), noise rng seed
20260901 + sigma_mm; DBSCAN + voxel grid are deterministic.
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
from blk360seg import io, preprocess, postprocess, objects, structure_filter  # noqa: E402

OP = dict(eps_m=0.30, min_points=100, split_footprint_m=2.5,
          split_eps_m=0.20, split_min_points=100)
NOISE_SEED_BASE = 20260901
REF_CLOUD = "outputs/vis_n2_det_filt/ceilingref.ply"   # pre-removal, 0.03 m
REF_OBJECTS = "outputs/vis_n2_det_run1"
GT = "outputs/vis_n2_gt.json"

REGIME = {
    0: "clean baseline (no added noise)",
    5: "~1x BLK360 per-scan bundle registration error (4-5 mm, "
       "within-bundle multi-setup regime)",
    10: "~2x per-scan bundle registration error",
    20: "~4x per-scan bundle registration error; still below the ~5 cm "
        "inter-epoch ICP residual handled at the KG matching level",
}


def key_of(xyz):
    q = np.round(np.asarray(xyz, dtype=np.float64) * 1e4).astype(np.int64)
    return q[:, 0] * 73856093 ^ q[:, 1] * 19349663 ^ q[:, 2] * 83492791


def build_reference():
    """Owner-grounded reference labels on the pre-removal 0.03 m cloud."""
    import open3d as o3d
    ref = np.asarray(o3d.io.read_point_cloud(
        os.path.join(ROOT, REF_CLOUD)).points)
    keys = key_of(ref)
    order = np.argsort(keys)
    skeys = keys[order]
    gd = json.load(open(os.path.join(ROOT, GT)))
    ents = [g for g in gd["objects"]
            if not g.get("structure_artifact") and not g.get("out_of_scope")
            and g.get("gt_quality") in ("confirmed", "plausible")]
    lab = np.full(len(ref), -1, dtype=np.int32)
    ids = []
    for gi, g in enumerate(sorted(ents, key=lambda e: int(e["id"]))):
        oid = int(g["id"])
        op = np.asarray(o3d.io.read_point_cloud(os.path.join(
            ROOT, REF_OBJECTS, f"obj_{oid:04d}.ply")).points)
        ok = key_of(op)
        pos = np.clip(np.searchsorted(skeys, ok), 0, len(skeys) - 1)
        hit = skeys[pos] == ok
        assert hit.all(), f"obj {oid}: {int((~hit).sum())} pts not in ref cloud"
        lab[order[pos[hit]]] = gi
        ids.append(oid)
    print(f"[robust] reference: {len(ids)} owner-audited objects, "
          f"{(lab >= 0).sum():,} support pts on {len(ref):,} ref pts")
    return ref, lab, ids


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
    matched = set()
    detail = []
    for g in range(n_inst):
        ov = stats[g]
        tot = inst_pts[g]
        if not ov or tot == 0:
            mis += 1
            detail.append("missing")
            continue
        big = [c for c, n in ov.items() if n >= 0.2 * tot]
        best_c, best_n = max(ov.items(), key=lambda kv: kv[1])
        iou = best_n / (tot + cl_pts[best_c] - best_n)
        other = any(n >= 0.2 * inst_pts[g2]
                    for g2 in range(n_inst) if g2 != g
                    for c2, n in stats[g2].items() if c2 == best_c)
        if len(big) >= 2:
            spl += 1
            detail.append("split")
        elif other:
            mer += 1
            detail.append("merged")
        elif iou >= 0.5:
            rec += 1
            matched.add(best_c)
            detail.append("recovered")
        else:
            mis += 1
            detail.append("missing")
    n_clusters = len([c for c in labs if cl_pts[int(c)] >= mp])
    return dict(recovered=rec, split=spl, merged=mer, missing=mis,
                recall=round(rec / n_inst, 3),
                precision=round(len(matched) / max(n_clusters, 1), 3),
                n_clusters=n_clusters), detail


def run_backbone(raw, voxel, sigma_mm):
    """One full Stage-A geometric run. Returns (clean_xyz, inst, stats)."""
    st = {}
    xyz = raw
    if sigma_mm > 0:
        t0 = time.time()
        rng = np.random.default_rng(NOISE_SEED_BASE + sigma_mm)
        xyz = raw.copy()
        # chunked so the float32 noise never doubles peak RAM
        s = np.float32(sigma_mm / 1000.0)
        for i in range(0, len(xyz), 8_000_000):
            xyz[i:i + 8_000_000] += rng.standard_normal(
                xyz[i:i + 8_000_000].shape, dtype=np.float32) * s
        st["t_noise_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    ds, _ = preprocess.voxel_downsample(xyz, np.zeros_like(xyz), voxel)
    st["n_downsampled"] = len(ds)
    st["t_downsample_s"] = round(time.time() - t0, 1)
    if sigma_mm > 0:
        del xyz
    t0 = time.time()
    ceiling_ref = ds  # pre-removal cloud (still has the ceiling)
    clean, _, nrm = preprocess.remove_structure(
        ds, np.zeros_like(ds))            # seeded RANSAC (seed=0 default)
    st["n_planes_removed"] = int(nrm)
    st["n_clean"] = len(clean)
    st["t_ransac_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    inst, n_obj, n_split = postprocess.cluster_all_split(clean, **OP)
    st["n_objects"] = int(n_obj)
    st["n_split_ops"] = int(n_split)
    st["t_cluster_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    objs = objects.extract_objects(clean, np.zeros_like(clean), inst,
                                   min_points=OP["min_points"])
    flags = structure_filter.find_ceiling_remnants(objs, ceiling_ref)
    drop = {f["id"] for f in flags}
    if drop:
        inst = inst.copy()
        inst[np.isin(inst, list(drop))] = -1
    st["n_remnants_dropped"] = len(drop)
    st["n_objects_final"] = int(n_obj) - len(drop)
    st["t_remnant_filter_s"] = round(time.time() - t0, 1)
    return clean, inst, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/home/caselab/Downloads/DataSend/"
                                     "vis_n2 1.e57")
    ap.add_argument("--raw-cache", default=None,
                    help="optional .npy float32 xyz cache of the raw cloud")
    ap.add_argument("--out-json",
                    default="outputs/revision1_tables/"
                            "robustness_voxel_noise.json")
    ap.add_argument("--out-csv",
                    default="outputs/revision1_tables/"
                            "robustness_voxel_noise.csv")
    args = ap.parse_args()
    np.random.seed(0)
    oj = os.path.join(ROOT, args.out_json)
    oc = os.path.join(ROOT, args.out_csv)
    for p in (oj, oc):
        if os.path.exists(p):
            sys.exit(f"[robust] refusing to overwrite existing {p}")

    from scipy.spatial import cKDTree
    ref_xyz, ref_lab, ref_ids = build_reference()
    tree = cKDTree(ref_xyz)
    n_inst = int(ref_lab.max()) + 1

    t0 = time.time()
    if args.raw_cache and os.path.exists(args.raw_cache):
        raw = np.load(args.raw_cache)
        print(f"[robust] raw cache: {len(raw):,} pts "
              f"({time.time()-t0:.1f}s)")
    else:
        raw, _ = io.load(args.raw)
        print(f"[robust] loaded raw {args.raw}: {len(raw):,} pts "
              f"({time.time()-t0:.1f}s)")

    runs = [("voxel", 0.02, 0), ("voxel", 0.03, 0), ("voxel", 0.05, 0),
            ("noise", 0.03, 5), ("noise", 0.03, 10), ("noise", 0.03, 20)]
    rows, details = [], {}
    for kind, voxel, sigma in runs:
        tag = (f"voxel_{voxel:.2f}" if kind == "voxel"
               else f"noise_{sigma}mm")
        print(f"[robust] === {tag} (voxel={voxel} m, sigma={sigma} mm) ===",
              flush=True)
        t_run = time.time()
        clean, inst, st = run_backbone(raw, voxel, sigma)
        tol = max(1.5 * voxel, 0.045) + 2 * sigma / 1000.0
        t0 = time.time()
        d, idx = tree.query(clean, k=1, workers=-1)
        inst_of = np.where(d <= tol, ref_lab[idx], -1).astype(np.int32)
        st["t_label_transfer_s"] = round(time.time() - t0, 1)
        s, detail = score(inst, inst_of, n_inst, OP["min_points"])
        wall = round(time.time() - t_run, 1)
        row = {"run": tag, "kind": kind, "voxel_m": voxel,
               "sigma_mm": sigma, "noise_regime": REGIME[sigma],
               "tol_m": round(tol, 3), "n_raw": len(raw), **st, **s,
               "support_pts": int((inst_of >= 0).sum()),
               "wall_s": wall}
        rows.append(row)
        details[tag] = {str(ref_ids[i]): detail[i] for i in range(n_inst)}
        print(f"    ds={st['n_downsampled']:,} clean={st['n_clean']:,} "
              f"obj={st['n_objects_final']} | rec={s['recovered']}/{n_inst} "
              f"spl={s['split']} mer={s['merged']} mis={s['missing']} "
              f"recall={s['recall']} | wall {wall}s", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(oc), exist_ok=True)
    df.drop(columns=["noise_regime"]).to_csv(oc, index=False)
    out = {
        "date": time.strftime("%Y-%m-%d"),
        "scene": "vis_n2 robot hall",
        "raw_scan": args.raw,
        "backbone": "voxel_downsample -> seeded RANSAC structure removal "
                    "(seed 0) -> cluster_all_split eps 0.30 / min_points 100 "
                    "/ footprint 2.5 / split 0.20/100 -> ceiling-remnant "
                    "filter (= audited vis_n2_det pipeline; baseline "
                    "downsample reproduces ceilingref.ply exactly)",
        "reference": "owner-audited inventory (42 confirmed+plausible "
                     "in-scope objects of outputs/vis_n2_gt.json, point "
                     "support outputs/vis_n2_det_run1), labels transferred "
                     "by NN on the pre-removal reference cloud, "
                     "tol = max(1.5*voxel, 0.045) + 2*sigma",
        "scoring": "identical to scripts/sensitivity_sweep.py: IoU>=0.5 "
                   "recovered; >=20% fragments split/merged",
        "noise_model": "additive per-point iid Gaussian on the raw cloud "
                       "BEFORE the backbone, per-axis sigma as stated, "
                       "rng seed 20260901 + sigma_mm",
        "noise_regimes": REGIME,
        "rows": rows,
        "per_object_outcome": details,
    }
    json.dump(out, open(oj, "w"), indent=1)
    print(f"[robust] wrote {oc}\n[robust] wrote {oj}")


if __name__ == "__main__":
    main()
