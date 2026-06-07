#!/usr/bin/env python3
"""Smoke-test CLI: load a BLK360 cloud -> preprocess -> segment -> save/visualize.

Runs end-to-end with the geometric baseline (no GPU/weights) so IO, preprocessing
and visualization can be validated on real data before the DL pipelines are wired.

  python scripts/segment.py --input ../testroom260601.e57 --viz
"""
import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blk360seg import io, preprocess, classes, segmenter, postprocess, viz  # noqa: E402


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help=".csv or .e57 cloud")
    ap.add_argument("--config", default=os.path.join(here, "..", "configs", "default.yaml"))
    ap.add_argument("--out", default=os.path.join(here, "..", "outputs"))
    ap.add_argument("--viz", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    os.makedirs(args.out, exist_ok=True)

    print(f"[seg] loading {args.input}")
    xyz, rgb = io.load(args.input)
    print(f"[seg] {len(xyz):,} points")

    pp = cfg["preprocess"]
    xyz, rgb = preprocess.voxel_downsample(xyz, rgb, pp["voxel_size_m"])
    print(f"[seg] downsampled to {len(xyz):,} @ {pp['voxel_size_m']} m")
    if pp.get("normalize", True):
        xyz = preprocess.normalize(xyz)

    seg = segmenter.build(cfg)
    print(f"[seg] segmenting with {type(seg).__name__}")
    labels = seg.segment(xyz, rgb)

    cls = classes.get_classes(cfg.get("classes", "s3dis"))
    po = cfg["postprocess"]
    inst, ninst = postprocess.cluster_instances(
        xyz, labels, eps_m=po["instance_eps_m"], min_points=po["instance_min_points"])
    print(f"[seg] {ninst} instances. class counts:")
    for c in np.unique(labels):
        name = cls[c] if c < len(cls) else str(c)
        print(f"    {name:>10}: {(labels == c).sum():,}")

    base = os.path.splitext(os.path.basename(args.input))[0]
    viz.save_labeled_csv(os.path.join(args.out, base + "_labeled.csv"), xyz, labels, inst)
    colors = viz.color_by_label(labels, len(cls))
    viz.save_ply(os.path.join(args.out, base + "_labeled.ply"), xyz, colors)
    print(f"[seg] wrote {args.out}/{base}_labeled.(csv|ply)")
    if args.viz:
        viz.show(xyz, colors)


if __name__ == "__main__":
    main()
