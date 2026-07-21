#!/usr/bin/env python3
"""Build a synthetic re-scan cloud by editing objects in a clean scene cloud.

Simulates a post-manipulation re-scan for the incremental-update experiment
(paper Sec. 5.5): remove / move / duplicate-insert whole object clusters in
the preprocessed scene cloud. The edited cloud is then re-run through the
same deterministic segmentation (extract_objects.py --input-clean), so every
untouched cluster reproduces byte-identically and only the edited regions
change — mirroring what a real re-scan of a mostly-static scene would yield.
(Disclosed in the paper as a synthetic manipulation of the real TLS scan.)

Edit spec (JSON):
  {"remove": ["chair_005"],
   "move":   [{"name": "clutter_011", "dx": 2.0, "dy": 0.5}],
   "insert": [{"copy_of": "chair_036", "dx": 3.0, "dy": -2.0}]}

Object membership comes from the run's obj_*.ply files: each object's points
are located in the scene cloud by exact nearest-neighbor match (the object
clouds are subsets of the clean cloud).

  .venv/bin/python scripts/synthetic_rescan.py \
      --clean outputs/showroom_det/clean.ply \
      --objects-dir outputs/showroom_det \
      --semantic outputs/showroom_det/semanticObjects.json \
      --edits configs/rescan_edits.json \
      --out outputs/showroom_rescan_clean.ply
"""
import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def obj_indices(scene_xyz, tree, obj_ply):
    """Indices of an object's points inside the scene cloud (exact subset)."""
    opts = np.asarray(o3d.io.read_point_cloud(obj_ply).points)
    idx = np.empty(len(opts), dtype=np.int64)
    for i, p in enumerate(opts):
        _, nn, d2 = tree.search_knn_vector_3d(p, 1)
        if d2[0] > 1e-10:
            raise RuntimeError(f"{obj_ply}: point {i} not found in scene "
                               f"(d2={d2[0]:.2e}) — wrong clean cloud?")
        idx[i] = nn[0]
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True, help="clean.ply of the base run")
    ap.add_argument("--objects-dir", required=True)
    ap.add_argument("--semantic", required=True,
                    help="semanticObjects.json (maps names -> obj files)")
    ap.add_argument("--edits", required=True, help="edit spec json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pc = o3d.io.read_point_cloud(args.clean)
    xyz = np.asarray(pc.points)
    rgb = np.asarray(pc.colors)
    tree = o3d.geometry.KDTreeFlann(pc)

    d = json.load(open(args.semantic))
    objs = d["semanticObjects"] if isinstance(d, dict) and \
        "semanticObjects" in d else d
    by_name = {o["name"]: o for o in objs}

    def ply_of(name):
        f = by_name[name]["properties"]["imageFile"]
        return os.path.join(args.objects_dir, f)

    edits = json.load(open(args.edits))
    drop = np.zeros(len(xyz), dtype=bool)
    add_xyz, add_rgb = [], []

    for name in edits.get("remove", []):
        idx = obj_indices(xyz, tree, ply_of(name))
        drop[idx] = True
        print(f"[edit] remove {name}: {len(idx):,} pts")

    for mv in edits.get("move", []):
        idx = obj_indices(xyz, tree, ply_of(mv["name"]))
        drop[idx] = True
        moved = xyz[idx] + np.array([mv.get("dx", 0.0), mv.get("dy", 0.0),
                                     mv.get("dz", 0.0)])
        add_xyz.append(moved)
        add_rgb.append(rgb[idx])
        print(f"[edit] move {mv['name']}: {len(idx):,} pts by "
              f"({mv.get('dx', 0)}, {mv.get('dy', 0)}, {mv.get('dz', 0)})")

    for ins in edits.get("insert", []):
        idx = obj_indices(xyz, tree, ply_of(ins["copy_of"]))
        placed = xyz[idx] + np.array([ins.get("dx", 0.0), ins.get("dy", 0.0),
                                      ins.get("dz", 0.0)])
        add_xyz.append(placed)
        add_rgb.append(rgb[idx])
        print(f"[edit] insert copy of {ins['copy_of']}: {len(idx):,} pts at "
              f"offset ({ins.get('dx', 0)}, {ins.get('dy', 0)})")

    keep_xyz, keep_rgb = xyz[~drop], rgb[~drop]
    out_xyz = np.concatenate([keep_xyz] + add_xyz) if add_xyz else keep_xyz
    out_rgb = np.concatenate([keep_rgb] + add_rgb) if add_rgb else keep_rgb

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(out_xyz)
    out.colors = o3d.utility.Vector3dVector(out_rgb)
    o3d.io.write_point_cloud(args.out, out)
    print(f"[edit] {len(xyz):,} -> {len(out_xyz):,} pts -> {args.out}")


if __name__ == "__main__":
    main()
