#!/usr/bin/env python3
"""Compare Uni3D vs PointCLIP V2 open-vocab classification on the SAME objects.

Both use the same EVA02-E CLIP text encoder + the same templates, so the only
difference is the point representation: Uni3D encodes the raw 3D cloud, PointCLIP
V2 renders 10 depth views and encodes them with CLIP's image tower.

  python scripts/compare_classifiers.py --objects-dir outputs/stage2_no_wall_objects

Writes comparison.csv (per object: each method's top-1 + score + agreement) and
prints class-distribution + agreement stats.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from blk360seg import classes                                       # noqa: E402
from blk360seg.uni3d_classifier import Uni3DClassifier              # noqa: E402
from blk360seg.pointclipv2_classifier import PointCLIPv2Classifier  # noqa: E402


def load_ply(path):
    import open3d as o3d
    pc = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pc.points, dtype=np.float32)
    rgb = np.asarray(pc.colors, dtype=np.float32)
    if rgb.shape[0] != xyz.shape[0] or rgb.size == 0:
        rgb = np.full_like(xyz, 0.4)
    return xyz, rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects-dir", required=True)
    ap.add_argument("--uni3d-ckpt", default=os.path.join(ROOT, "weights", "uni3d-b.pt"))
    ap.add_argument("--clip-ckpt", default=os.path.join(ROOT, "weights", "eva02_e_clip.bin"))
    ap.add_argument("--classes", default="industrial")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    cls = classes.get_classes(args.classes)
    files = sorted(glob.glob(os.path.join(args.objects_dir, "obj_*.ply")))
    objs = [(os.path.basename(f), *load_ply(f)) for f in files]
    print(f"[cmp] {len(cls)} classes, {len(objs)} objects")

    # EVA02-E CLIP is huge; run the two methods sequentially so only one big
    # model is resident at a time (two copies OOM a 24 GB card).
    print("[cmp] pass 1/2: Uni3D...")
    uni = Uni3DClassifier(args.uni3d_ckpt, args.clip_ckpt, device=args.device)
    uni.set_classes(cls)
    uni_pred = {name: uni.classify(xyz, rgb, topk=1)[0] for name, xyz, rgb in objs}
    del uni
    torch.cuda.empty_cache()

    print("[cmp] pass 2/2: PointCLIP V2 (same CLIP)...")
    pc2 = PointCLIPv2Classifier(args.clip_ckpt, device=args.device)
    pc2.set_classes(cls)
    pcv2_pred = {name: pc2.classify(xyz, topk=1)[0] for name, xyz, rgb in objs}
    del pc2
    torch.cuda.empty_cache()

    print(f"\n{'object':>14} | {'Uni3D':>20} | {'PointCLIP V2':>20} | agree")
    print("-" * 70)
    rows = []
    for name, xyz, rgb in objs:
        u, p = uni_pred[name], pcv2_pred[name]
        agree = (u[0] == p[0])
        print(f"{name:>14} | {u[0]:>13} {u[1]:>5.2f} | "
              f"{p[0]:>13} {p[1]:>5.2f} | {'YES' if agree else ''}")
        rows.append({"file": name, "n_points": len(xyz),
                     "uni3d_top1": u[0], "uni3d_score": round(u[1], 3),
                     "pcv2_top1": p[0], "pcv2_score": round(p[1], 3),
                     "agree": agree})

    df = pd.DataFrame(rows)
    out = os.path.join(args.objects_dir, "comparison.csv")
    df.to_csv(out, index=False)
    print("\n" + "=" * 60)
    print(f"agreement: {df.agree.sum()}/{len(df)} ({100*df.agree.mean():.0f}%)")
    print(f"mean conf  Uni3D={df.uni3d_score.mean():.3f}  "
          f"PointCLIP V2={df.pcv2_score.mean():.3f}")
    print("\nUni3D class counts:\n" + df.uni3d_top1.value_counts().to_string())
    print("\nPointCLIP V2 class counts:\n" + df.pcv2_top1.value_counts().to_string())
    print(f"\n[cmp] wrote {out}")


if __name__ == "__main__":
    main()
