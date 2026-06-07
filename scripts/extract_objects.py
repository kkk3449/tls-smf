#!/usr/bin/env python3
"""Stage A: cloud -> object instances saved as individual point clouds + manifest.

  load (csv/e57) -> downsample -> [auto-remove floor/walls] -> DBSCAN into objects
  -> save obj_*.ply + objects.csv

Works whether or not the input still has walls/floor (remove_structure auto-strips
the big structural planes; it's a no-op when they're already gone). The per-object
clouds feed Stage B (Uni3D open-vocab classification).

  python scripts/extract_objects.py --input ../testroom_no_wall/stage2_no_wall.e57
"""
import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blk360seg import io, preprocess, postprocess, objects  # noqa: E402


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default=os.path.join(here, "..", "configs", "default.yaml"))
    ap.add_argument("--out", default=None, help="default: outputs/<name>_objects/")
    ap.add_argument("--no-remove-structure", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    name = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.out or os.path.join(here, "..", "outputs", f"{name}_objects")

    print(f"[A] loading {args.input}")
    xyz, rgb = io.load(args.input)
    print(f"[A] {len(xyz):,} points")

    pp = cfg["preprocess"]
    xyz, rgb = preprocess.voxel_downsample(xyz, rgb, pp["voxel_size_m"])
    print(f"[A] downsampled to {len(xyz):,} @ {pp['voxel_size_m']} m")

    if cfg.get("remove_structure", True) and not args.no_remove_structure:
        n0 = len(xyz)
        xyz, rgb, nrm = preprocess.remove_structure(xyz, rgb)
        print(f"[A] removed {nrm} structural plane(s) (floor/wall/ceiling): "
              f"{n0:,} -> {len(xyz):,} points")

    oc = cfg["objects"]
    inst = postprocess.cluster_all(xyz, eps_m=oc["cluster_eps_m"],
                                   min_points=oc["min_points"])
    objs = objects.extract_objects(xyz, rgb, inst, min_points=oc["min_points"])
    manifest = objects.save_objects(objs, out_dir)
    print(f"[A] {len(objs)} objects -> {out_dir}/")
    for o in objs[:12]:
        s = o["bbox_max"] - o["bbox_min"]
        print(f"    obj_{o['id']:04d}: {o['n']:>6,} pts  size {s[0]:.2f}x{s[1]:.2f}x{s[2]:.2f} m")
    if len(objs) > 12:
        print(f"    ... (+{len(objs) - 12} more)")
    print(f"[A] manifest: {manifest}")


if __name__ == "__main__":
    main()
