"""Stage A output: extract per-instance object point clouds + a manifest.

These per-object clouds are what Stage B (Uni3D open-vocab classification)
consumes -- one object at a time.
"""
import os

import numpy as np


def extract_objects(xyz, rgb, instances, min_points=50):
    """instances: per-point instance id (-1 = noise). Returns a list of dicts."""
    objs = []
    for iid in np.unique(instances):
        if iid < 0:
            continue
        m = instances == iid
        if int(m.sum()) < min_points:
            continue
        oxyz, orgb = xyz[m], rgb[m]
        objs.append({
            "id": int(iid),
            "xyz": oxyz,
            "rgb": orgb,
            "n": int(m.sum()),
            "centroid": oxyz.mean(axis=0),
            "bbox_min": oxyz.min(axis=0),
            "bbox_max": oxyz.max(axis=0),
        })
    return objs


def save_objects(objects, out_dir):
    """Write each object as obj_<id>.ply and an objects.csv manifest."""
    import glob
    import open3d as o3d
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(out_dir, "obj_*.ply")):
        os.remove(stale)   # avoid leaving a previous run's objects behind
    rows = []
    for o in objects:
        fn = f"obj_{o['id']:04d}.ply"
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(o["xyz"].astype(np.float64))
        pc.colors = o3d.utility.Vector3dVector(o["rgb"].astype(np.float64))
        o3d.io.write_point_cloud(os.path.join(out_dir, fn), pc)
        size = o["bbox_max"] - o["bbox_min"]
        rows.append({
            "id": o["id"], "file": fn, "n_points": o["n"],
            "cx": round(float(o["centroid"][0]), 3),
            "cy": round(float(o["centroid"][1]), 3),
            "cz": round(float(o["centroid"][2]), 3),
            "size_x": round(float(size[0]), 3),
            "size_y": round(float(size[1]), 3),
            "size_z": round(float(size[2]), 3),
        })
    manifest = os.path.join(out_dir, "objects.csv")
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest
