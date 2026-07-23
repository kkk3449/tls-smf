#!/usr/bin/env python3
"""Register the T3 scan (vis_sota) into the vis_n2 (T2) frame.

FPFH + RANSAC global registration on downsampled clouds, refined with
point-to-plane ICP. Same recipe that registered the June testroom scan
(yaw 42.6 deg, RMSE 5.1 cm). Saves a 4x4 transform to
outputs/vissota_to_visn2_T.npy.
"""
import numpy as np
import open3d as o3d

SRC = "outputs/vis_sota_det/clean.ply"   # T3, to be moved
DST = "outputs/vis_n2_det_filt/clean.ply"  # T2, canonical frame
OUT = "outputs/vissota_to_visn2_T.npy"
VOXEL = 0.08

def prep(path):
    p = o3d.io.read_point_cloud(path)
    d = p.voxel_down_sample(VOXEL)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 2.5, max_nn=40))
    f = o3d.pipelines.registration.compute_fpfh_feature(
        d, o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 5, max_nn=100))
    return p, d, f

src, sd, sf = prep(SRC)
dst, dd, df = prep(DST)
print(f"src {len(src.points)} pts (down {len(sd.points)}), dst {len(dst.points)} (down {len(dd.points)})")

o3d.utility.random.seed(0)
res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    sd, dd, sf, df, True, VOXEL * 1.5,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
    [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
     o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(VOXEL * 1.5)],
    o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 0.9999))
print(f"RANSAC fitness {res.fitness:.3f} rmse {res.inlier_rmse:.3f}")

icp = o3d.pipelines.registration.registration_icp(
    sd, dd, VOXEL * 1.2, res.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane())
T = icp.transformation
print(f"ICP fitness {icp.fitness:.3f} rmse {icp.inlier_rmse*100:.1f} cm")
yaw = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
print(f"yaw {yaw:.1f} deg  t = ({T[0,3]:.2f}, {T[1,3]:.2f}, {T[2,3]:.2f})")

np.save(OUT, np.asarray(T))
print("saved", OUT)
