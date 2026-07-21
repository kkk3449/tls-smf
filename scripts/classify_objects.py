#!/usr/bin/env python3
"""Stage B: classify each Stage-A object (obj_*.ply) with Uni3D open-vocab.

  python scripts/classify_objects.py \
      --objects-dir outputs/stage2_no_wall_objects \
      --classes industrial

Writes classification.csv (per object: top-1..k class + score) and a combined
scene_classified.ply colored by predicted class.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from blk360seg import classes, viz                              # noqa: E402
from blk360seg.uni3d_classifier import Uni3DClassifier          # noqa: E402


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
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # named set (s3dis|scannet|industrial) or a literal comma-separated list
    # of user keywords ("chair,desk,monitor,...") — scene-curated vocabularies
    cls = ([c.strip() for c in args.classes.split(",") if c.strip()]
           if "," in args.classes else classes.get_classes(args.classes))
    print(f"[B] {len(cls)} classes: {cls}")
    print("[B] loading Uni3D + EVA02-E CLIP (first run is slow)...")
    clf = Uni3DClassifier(args.uni3d_ckpt, args.clip_ckpt, device=args.device)
    clf.set_classes(cls)

    files = sorted(glob.glob(os.path.join(args.objects_dir, "obj_*.ply")))
    print(f"[B] classifying {len(files)} objects")
    pal = classes.palette(len(cls))
    rows, scene_xyz, scene_rgb = [], [], []
    for f in files:
        xyz, rgb = load_ply(f)
        preds = clf.classify(xyz, rgb, topk=args.topk)
        top1, s1 = preds[0]
        print(f"  {os.path.basename(f):>16}  ->  {top1:<18} ({s1:.2f})   "
              + " ".join(f"{c}:{s:.2f}" for c, s in preds[1:]))
        row = {"file": os.path.basename(f), "n_points": len(xyz),
               "top1": top1, "score1": round(s1, 3)}
        for j, (c, s) in enumerate(preds[1:], start=2):
            row[f"top{j}"] = c
            row[f"score{j}"] = round(s, 3)
        rows.append(row)
        scene_xyz.append(xyz)
        scene_rgb.append(np.tile(pal[cls.index(top1)], (len(xyz), 1)))

    out_csv = os.path.join(args.objects_dir, "classification.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    if scene_xyz:
        viz.save_ply(os.path.join(args.objects_dir, "scene_classified.ply"),
                     np.vstack(scene_xyz), np.vstack(scene_rgb))
    print(f"[B] wrote {out_csv} + scene_classified.ply")
    counts = pd.DataFrame(rows)["top1"].value_counts()
    print("[B] class counts:\n" + counts.to_string())


if __name__ == "__main__":
    main()
