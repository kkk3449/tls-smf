#!/usr/bin/env python3
"""Segmentation pilot (paper Sec. 5.2): Point-SAM refinement of DBSCAN
merged-cluster failures.

For every oversized cluster that the conservative giant-split could not
separate (footprint > threshold after split), prompt Point-SAM with K
farthest-point-sampled single-point prompts, take each prompt's best mask,
greedily de-overlap them, and emit the resulting sub-clusters as
sub_<name>_<i>.ply for visual/GT audit and re-classification.

Pre-registered adoption rule (fixed in the outline): adopt Point-SAM as an
optional refinement iff >=50% of merged-cluster failures are correctly
separated AND final label accuracy improves >=5 %p; otherwise report the
negative result and keep the geometric-only front end.

  .venv/bin/python scripts/pointsam_pilot.py \
      --objects-dir outputs/vis_n2_det_run1 \
      --semantic outputs/vis_n2_det_run1/semanticObjects.json \
      --ckpt weights/pointsam_vitl.safetensors \
      --out outputs/pointsam_pilot_vis_n2
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PSAM = os.path.join(ROOT, "third_party", "Point-SAM")
sys.path.insert(0, ROOT)
sys.path.insert(0, PSAM)


def fps(xyz, k, seed=0):
    """Deterministic farthest point sampling (first seed = closest to mean)."""
    n = len(xyz)
    sel = [int(np.argmin(((xyz - xyz.mean(0)) ** 2).sum(1)))]
    d = ((xyz - xyz[sel[0]]) ** 2).sum(1)
    for _ in range(k - 1):
        sel.append(int(np.argmax(d)))
        d = np.minimum(d, ((xyz - xyz[sel[-1]]) ** 2).sum(1))
    return np.array(sel)


def main():
    import torch
    import open3d as o3d
    import hydra
    from omegaconf import OmegaConf
    from safetensors.torch import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--objects-dir", required=True)
    ap.add_argument("--semantic", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="large")
    ap.add_argument("--footprint", type=float, default=2.5,
                    help="select objects whose max XY extent exceeds this")
    ap.add_argument("--min-aspect", type=float, default=0.35,
                    help="skip thin elongated objects (rails/pipes): require "
                         "minor/major XY extent ratio above this OR footprint "
                         "> 2x threshold")
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--min-points", type=int, default=100)
    ap.add_argument("--select", default=None)
    args = ap.parse_args()

    d = json.load(open(args.semantic))
    objs = d["semanticObjects"] if isinstance(d, dict) and \
        "semanticObjects" in d else d
    sel = set(args.select.split(",")) if args.select else None
    cands = []
    for o in objs:
        dim = o["dimensions"]
        major = max(dim["length"], dim["width"])
        minor = min(dim["length"], dim["width"])
        if sel is not None:
            if o["name"] in sel:
                cands.append(o)
            continue
        if major > args.footprint and (minor / major >= args.min_aspect
                                       or major > 2 * args.footprint):
            cands.append(o)
    print(f"{len(cands)} merged-cluster candidates: "
          f"{[c['name'] for c in cands]}")

    # ---- model ----
    with hydra.initialize_config_dir(os.path.join(PSAM, "configs"),
                                     version_base=None):
        cfg = hydra.compose(config_name=args.config)
        OmegaConf.resolve(cfg)
    model = hydra.utils.instantiate(cfg.model)
    load_model(model, args.ckpt)
    model.eval().cuda()

    os.makedirs(args.out, exist_ok=True)
    report = []
    for o in cands:
        f = os.path.join(args.objects_dir, o["properties"]["imageFile"])
        pc = o3d.io.read_point_cloud(f)
        xyz = np.asarray(pc.points, dtype=np.float32)
        rgb = np.asarray(pc.colors, dtype=np.float32)
        n = len(xyz)

        center = xyz.mean(0)
        coords = xyz - center
        scale = float(np.linalg.norm(coords, axis=1).max())
        coords_n = coords / scale

        seeds = fps(xyz, args.prompts)
        tc = torch.tensor(coords_n, device="cuda")[None]
        tf = torch.tensor(rgb, device="cuda")[None]

        masks_np = []
        with torch.no_grad():
            for s in seeds:
                pcoord = tc[:, s: s + 1, :][:, None]      # [B, M=1, 1, 3]
                plabel = torch.ones(1, 1, 1, dtype=torch.long,
                                    device="cuda")
                logits, ious = model.predict_masks(
                    tc, tf, pcoord.squeeze(1), plabel.squeeze(1))
                best = int(torch.argmax(ious[0]))
                masks_np.append((float(ious[0, best]),
                                 (logits[0, best] > 0).cpu().numpy()))

        # greedy NMS: keep high-iou masks that add novel coverage
        masks_np.sort(key=lambda t: -t[0])
        keep, covered = [], np.zeros(n, dtype=bool)
        for iou, m in masks_np:
            novel = m & ~covered
            if novel.sum() < args.min_points:
                continue
            keep.append(novel)
            covered |= m
        rest = ~covered
        if rest.sum() >= args.min_points:
            keep.append(rest)

        subs = []
        for i, m in enumerate(keep):
            sub = o3d.geometry.PointCloud()
            sub.points = o3d.utility.Vector3dVector(xyz[m].astype(np.float64))
            sub.colors = o3d.utility.Vector3dVector(rgb[m].astype(np.float64))
            sf = os.path.join(args.out, f"sub_{o['name']}_{i}.ply")
            o3d.io.write_point_cloud(sf, sub)
            subs.append({"file": os.path.basename(sf),
                         "n_points": int(m.sum())})
        report.append({"name": o["name"], "n_points": n,
                       "dimensions": o["dimensions"],
                       "n_subclusters": len(subs), "subs": subs})
        print(f"  {o['name']}: {n:,} pts -> {len(subs)} sub-clusters "
              f"({[s['n_points'] for s in subs]})")

    json.dump(report, open(os.path.join(args.out, "pilot_report.json"), "w"),
              indent=1)
    print(f"report -> {args.out}/pilot_report.json")


if __name__ == "__main__":
    main()
