#!/usr/bin/env python3
"""Semantic-segment a BLK360 cloud with PTv3 (ScanNet-pretrained, semantic-only).

  CUDA_HOME=/path/to/cuda \
  LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib:$LD_LIBRARY_PATH \
  PYTHONPATH=third_party/Pointcept \
  python scripts/segment_ptv3.py --input ../testroom_no_wall/stage2_no_wall.e57

PTv3 predicts a per-point ScanNet-20 semantic class (no instances). Writes:
  - scene_ptv3_semseg.ply  (points colored by predicted class)
  - ptv3_semseg.csv        (per-class point count + mean confidence)
to outputs/<name>_ptv3/.
"""
import argparse
import collections
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "Pointcept"))
from blk360seg import io, preprocess, classes                    # noqa: E402

REPO = os.path.join(ROOT, "third_party", "Pointcept")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    ap.add_argument("--ckpt", default=os.path.join(REPO, "checkpoints", "ptv3_scannet_base.pth"))
    ap.add_argument("--grid-size", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from blk360seg.ptv3_segmenter import PTv3Segmenter, SCANNET20

    cfg = yaml.safe_load(open(args.config))
    name = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.out or os.path.join(ROOT, "outputs", f"{name}_ptv3")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[ptv3] loading {args.input}")
    xyz, rgb = io.load(args.input)
    xyz, rgb = preprocess.voxel_downsample(xyz, rgb, cfg["preprocess"]["voxel_size_m"])
    print(f"[ptv3] {len(xyz):,} points after downsample")

    seg = PTv3Segmenter(REPO, args.ckpt, grid_size=args.grid_size)
    print("[ptv3] running PTv3 semantic segmentation...")
    labels, conf = seg.segment(xyz, rgb)

    # per-class summary
    cc = collections.Counter(labels.tolist())
    rows = []
    for c, n in cc.most_common():
        m = float(conf[labels == c].mean())
        rows.append({"class_id": int(c), "class": SCANNET20[int(c)],
                     "n_points": int(n), "frac": n / len(labels), "mean_conf": round(m, 3)})
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "ptv3_semseg.csv"), index=False)

    print("[ptv3] ScanNet-20 semantic distribution:")
    for r in rows:
        print(f"  {r['class']:<16} {r['n_points']:>7} pts  {100*r['frac']:5.1f}%  conf={r['mean_conf']}")

    # colored point cloud by class
    import open3d as o3d
    pal = classes.palette(len(SCANNET20))
    col = np.array([pal[int(c)] for c in labels], dtype=np.float64)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(col)
    o3d.io.write_point_cloud(os.path.join(out_dir, "scene_ptv3_semseg.ply"), pc)
    print(f"[ptv3] wrote {out_dir}/scene_ptv3_semseg.ply + ptv3_semseg.csv")
    print(f"[ptv3] distinct classes: {len(cc)}  mean conf: {conf.mean():.3f}")


if __name__ == "__main__":
    main()
