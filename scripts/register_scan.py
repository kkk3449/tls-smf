#!/usr/bin/env python3
"""Register a new scan's clean.ply into an existing epoch's frame (CLI form of
register_epoch_scan.py: FPFH + RANSAC global registration, point-to-plane ICP).

  .venv/bin/python scripts/register_scan.py --src outputs/X/clean.ply \
      --dst outputs/vis_n2_det_filt/clean.ply --out outputs/X/to_kg_T.npy
"""
import argparse, json
import numpy as np
import open3d as o3d

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--voxel", type=float, default=0.08)
a = ap.parse_args()
V = a.voxel

def prep(path):
    p = o3d.io.read_point_cloud(path)
    d = p.voxel_down_sample(V)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=V * 2.5, max_nn=40))
    f = o3d.pipelines.registration.compute_fpfh_feature(
        d, o3d.geometry.KDTreeSearchParamHybrid(radius=V * 5, max_nn=100))
    return p, d, f

src, sd, sf = prep(a.src); dst, dd, df = prep(a.dst)
o3d.utility.random.seed(0)
res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    sd, dd, sf, df, True, V * 1.5,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
    [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
     o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(V * 1.5)],
    o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 0.9999))
icp = o3d.pipelines.registration.registration_icp(
    sd, dd, V * 1.2, res.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane())
T = np.asarray(icp.transformation)
yaw = float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
np.save(a.out, T)
json.dump({"ransac_fitness": res.fitness, "icp_fitness": icp.fitness,
           "icp_rmse_m": icp.inlier_rmse, "yaw_deg": yaw,
           "t": T[:3, 3].tolist()}, open(a.out.replace(".npy", ".json"), "w"), indent=1)
print(f"RANSAC fitness {res.fitness:.3f}; ICP fitness {icp.fitness:.3f} rmse {icp.inlier_rmse*100:.1f} cm; "
      f"yaw {yaw:.1f} deg t=({T[0,3]:.2f},{T[1,3]:.2f},{T[2,3]:.2f}) -> {a.out}")
