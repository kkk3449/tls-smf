#!/usr/bin/env python3
"""Extract SPFormer's `chair` instances for visual ground-truth verification.

The showroom genuinely contains chairs/desks/robots, so SPFormer's 67 chair
instances are a mix of *real chairs* and *industrial objects mis-labeled chair*.
This pulls them out so a human can eyeball which is which:

  outputs/<name>_spformer_chairs/
    chairs_highlight.ply   full scene; chairs colored by conf tier, rest light gray
    chairs/chair_<id>_c<conf>.ply   each chair instance, isolated
    chairs.csv             id, conf, n_points, centroid, bbox size

Run with the SPFormer CUDA env (same as scripts/segment_spformer.py):
  CUDA_HOME=/home/caselab/cuda128 \
  LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib:$LD_LIBRARY_PATH \
  python scripts/extract_spformer_chairs.py --input ../testroom_no_wall/stage2_no_wall.e57
"""
import argparse
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from blk360seg import io, preprocess                              # noqa: E402

REPO = os.path.join(ROOT, "third_party", "SPFormer")

# conf -> RGB tier (high=red, mid=orange, low=yellow); non-chair = light gray
GRAY = (0.82, 0.82, 0.82)


def tier_color(conf):
    if conf >= 0.6:
        return (0.85, 0.10, 0.10)   # red  — high confidence
    if conf >= 0.3:
        return (0.95, 0.55, 0.10)   # orange — medium
    return (0.95, 0.90, 0.20)       # yellow — low


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    ap.add_argument("--spf-config", default=os.path.join(REPO, "configs", "spf_scannet.yaml"))
    ap.add_argument("--ckpt", default=os.path.join(REPO, "checkpoints", "spf_scannet_512.pth"))
    ap.add_argument("--knn", type=int, default=16)
    ap.add_argument("--k-thresh", type=float, default=0.04)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import open3d as o3d
    import pandas as pd
    from blk360seg.spformer_segmenter import SPFormerSegmenter, SCANNET_INST

    cfg = yaml.safe_load(open(args.config))
    name = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.out or os.path.join(ROOT, "outputs", f"{name}_spformer_chairs")
    os.makedirs(os.path.join(out_dir, "chairs"), exist_ok=True)

    print(f"[chairs] loading {args.input}")
    xyz, rgb = io.load(args.input)
    xyz, rgb = preprocess.voxel_downsample(xyz, rgb, cfg["preprocess"]["voxel_size_m"])
    print(f"[chairs] {len(xyz):,} points")

    seg = SPFormerSegmenter(REPO, args.ckpt, args.spf_config)
    print("[chairs] running SPFormer...")
    inst_labels, meta, _ = seg.segment(xyz, rgb, knn=args.knn, k_thresh=args.k_thresh)

    chair_ids = [m["id"] for m in meta if m["class"] == "chair"]
    by_id = {m["id"]: m for m in meta}
    print(f"[chairs] {len(chair_ids)} chair instances")

    # full-scene highlight
    col = np.tile(np.array(GRAY, dtype=np.float64), (len(xyz), 1))
    rows = []
    for cid in chair_ids:
        m = inst_labels == cid
        n = int(m.sum())
        if n == 0:
            continue
        conf = by_id[cid]["conf"]
        col[m] = tier_color(conf)
        oxyz = xyz[m]
        c = oxyz.mean(0)
        size = oxyz.max(0) - oxyz.min(0)
        rows.append({"id": cid, "conf": round(conf, 3), "n_points": n,
                     "cx": round(float(c[0]), 3), "cy": round(float(c[1]), 3),
                     "cz": round(float(c[2]), 3),
                     "size_x": round(float(size[0]), 3),
                     "size_y": round(float(size[1]), 3),
                     "size_z": round(float(size[2]), 3)})
        # isolated instance ply (keep original color so shape is readable)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(oxyz.astype(np.float64))
        pc.colors = o3d.utility.Vector3dVector(rgb[m].astype(np.float64))
        o3d.io.write_point_cloud(
            os.path.join(out_dir, "chairs", f"chair_{cid:02d}_c{conf:.2f}.ply"), pc)

    scene = o3d.geometry.PointCloud()
    scene.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    scene.colors = o3d.utility.Vector3dVector(col)
    o3d.io.write_point_cloud(os.path.join(out_dir, "chairs_highlight.ply"), scene)

    df = pd.DataFrame(rows).sort_values("conf", ascending=False)
    df.to_csv(os.path.join(out_dir, "chairs.csv"), index=False)
    print(f"[chairs] wrote {out_dir}/chairs_highlight.ply + chairs/ ({len(rows)} plys) + chairs.csv")
    hi = (df["conf"] >= 0.6).sum()
    print(f"[chairs] tiers — red(>=0.6): {hi}  orange(0.3-0.6): {((df.conf>=0.3)&(df.conf<0.6)).sum()}  "
          f"yellow(<0.3): {(df.conf<0.3).sum()}")
    print("[chairs] top-conf chairs (likely real / strongest claims):")
    for _, r in df.head(12).iterrows():
        print(f"  chair_{int(r['id']):02d}  conf={r['conf']:.2f}  n={int(r['n_points'])}  "
              f"size={r['size_x']:.2f}x{r['size_y']:.2f}x{r['size_z']:.2f}m  @({r['cx']},{r['cy']},{r['cz']})")


if __name__ == "__main__":
    main()
