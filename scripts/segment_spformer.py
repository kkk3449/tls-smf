#!/usr/bin/env python3
"""Segment a BLK360 cloud with SPFormer (ScanNet-pretrained, Mask3D substitute).

  CUDA_HOME=/path/to/cuda \
  LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib:$LD_LIBRARY_PATH \
  python scripts/segment_spformer.py --input ../testroom_no_wall/stage2_no_wall.e57

Writes obj_*.ply + objects.csv (same layout as extract_objects.py) plus
spformer_instances.csv (each instance's ScanNet class + confidence) so the same
Stage-B classifier / comparison can run on SPFormer objects.
"""
import argparse
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from blk360seg import io, preprocess, objects                       # noqa: E402

REPO = os.path.join(ROOT, "third_party", "SPFormer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    ap.add_argument("--spf-config", default=os.path.join(REPO, "configs", "spf_scannet.yaml"))
    ap.add_argument("--ckpt", default=os.path.join(REPO, "checkpoints", "spf_scannet_512.pth"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--knn", type=int, default=16)
    ap.add_argument("--k-thresh", type=float, default=0.04)
    ap.add_argument("--min-points", type=int, default=50)
    args = ap.parse_args()

    from blk360seg.spformer_segmenter import SPFormerSegmenter

    cfg = yaml.safe_load(open(args.config))
    name = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.out or os.path.join(ROOT, "outputs", f"{name}_spformer")

    print(f"[spf] loading {args.input}")
    xyz, rgb = io.load(args.input)
    xyz, rgb = preprocess.voxel_downsample(xyz, rgb, cfg["preprocess"]["voxel_size_m"])
    print(f"[spf] {len(xyz):,} points after downsample")

    seg = SPFormerSegmenter(REPO, args.ckpt, args.spf_config)
    print("[spf] generating superpoints + running SPFormer...")
    inst_labels, meta, superpoint = seg.segment(xyz, rgb, knn=args.knn, k_thresh=args.k_thresh)
    n_sp = int(superpoint.max()) + 1
    print(f"[spf] {n_sp} superpoints -> {len(meta)} instances "
          f"(score_thr/npoint_thr from config)")

    # save instances as objects (filter tiny)
    objs = objects.extract_objects(xyz, rgb, inst_labels, min_points=args.min_points)
    objects.save_objects(objs, out_dir)
    print(f"[spf] {len(objs)} objects (>= {args.min_points} pts) -> {out_dir}/")

    import pandas as pd
    pd.DataFrame(meta).to_csv(os.path.join(out_dir, "spformer_instances.csv"), index=False)
    if meta:
        import collections
        cc = collections.Counter(m["class"] for m in meta)
        print("[spf] SPFormer ScanNet-class counts:\n  "
              + "  ".join(f"{k}:{v}" for k, v in cc.most_common()))
    cov = (inst_labels >= 0).mean() * 100
    print(f"[spf] point coverage by instances: {cov:.1f}%")


if __name__ == "__main__":
    main()
