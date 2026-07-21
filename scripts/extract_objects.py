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
    ap.add_argument("--save-clean", action="store_true",
                    help="also write the preprocessed (downsampled + "
                         "structure-removed) cloud to <out>/clean.ply")
    ap.add_argument("--input-clean", action="store_true",
                    help="input already IS a preprocessed clean cloud "
                         "(skip downsample + structure removal entirely; "
                         "used by the synthetic re-scan pipeline so that "
                         "untouched clusters stay byte-identical)")
    ap.add_argument("--ceiling-ref", default=None,
                    help="cloud (e57/ply) that still contains the ceiling; "
                         "enables the ceiling-soffit remnant filter. When "
                         "structure removal runs in-pipeline the pre-removal "
                         "cloud is used automatically and this is not needed")
    ap.add_argument("--keep-remnants", action="store_true",
                    help="disable the structure-remnant filter")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    name = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.out or os.path.join(here, "..", "outputs", f"{name}_objects")

    print(f"[A] loading {args.input}")
    xyz, rgb = io.load(args.input)
    print(f"[A] {len(xyz):,} points")

    ceiling_ref = None
    if args.input_clean:
        print("[A] --input-clean: skipping downsample + structure removal")
    else:
        pp = cfg["preprocess"]
        xyz, rgb = preprocess.voxel_downsample(xyz, rgb, pp["voxel_size_m"])
        print(f"[A] downsampled to {len(xyz):,} @ {pp['voxel_size_m']} m")

        if cfg.get("remove_structure", True) and not args.no_remove_structure:
            n0 = len(xyz)
            ceiling_ref = xyz.copy()   # pre-removal cloud still has the ceiling
            xyz, rgb, nrm = preprocess.remove_structure(xyz, rgb)
            print(f"[A] removed {nrm} structural plane(s) (floor/wall/ceiling): "
                  f"{n0:,} -> {len(xyz):,} points")

    if args.ceiling_ref:
        rx, _ = io.load(args.ceiling_ref)
        ceiling_ref = rx
        print(f"[A] ceiling reference: {args.ceiling_ref} ({len(rx):,} points)")

    if args.save_clean:
        import open3d as o3d
        os.makedirs(out_dir, exist_ok=True)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
        o3d.io.write_point_cloud(os.path.join(out_dir, "clean.ply"), pc)
        print(f"[A] clean cloud -> {out_dir}/clean.ply")
        if ceiling_ref is not None and not args.ceiling_ref:
            rp = o3d.geometry.PointCloud()
            rp.points = o3d.utility.Vector3dVector(ceiling_ref.astype(np.float64))
            o3d.io.write_point_cloud(os.path.join(out_dir, "ceilingref.ply"), rp)
            print(f"[A] pre-removal cloud -> {out_dir}/ceilingref.ply "
                  f"(reuse via --ceiling-ref for --input-clean re-runs)")

    oc = cfg["objects"]
    if oc.get("split_oversized"):
        inst, n_obj, n_split = postprocess.cluster_all_split(
            xyz, eps_m=oc["cluster_eps_m"], min_points=oc["min_points"],
            split_footprint_m=oc["split_footprint_m"],
            split_eps_m=oc["split_eps_m"], split_min_points=oc["split_min_points"])
        print(f"[A] clustered into {n_obj} objects "
              f"(giant-split re-split {n_split} over-merged cluster(s))")
    else:
        inst = postprocess.cluster_all(xyz, eps_m=oc["cluster_eps_m"],
                                       min_points=oc["min_points"])
    objs = objects.extract_objects(xyz, rgb, inst, min_points=oc["min_points"])

    if ceiling_ref is not None and not args.keep_remnants:
        from blk360seg import structure_filter
        flags = structure_filter.find_ceiling_remnants(objs, ceiling_ref)
        if flags:
            import pandas as pd
            drop = {f["id"] for f in flags}
            objs = [o for o in objs if o["id"] not in drop]
            os.makedirs(out_dir, exist_ok=True)
            rem_csv = os.path.join(out_dir, "structure_remnants.csv")
            pd.DataFrame(flags).to_csv(rem_csv, index=False)
            print(f"[A] structure-remnant filter: dropped {len(flags)} "
                  f"ceiling-soffit cluster(s) -> {rem_csv}")
            for f in flags:
                print(f"    obj_{f['id']:04d}: z=[{f['zmin']},{f['zmax']}] "
                      f"{f['above_floor']} m above floor, "
                      f"{f['ceiling_pts_above']} ceiling pts above")

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
