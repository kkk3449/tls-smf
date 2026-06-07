"""Downsample / normalize a point cloud before segmentation."""
import numpy as np


def voxel_downsample(xyz, rgb, voxel_size_m):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    pc = pc.voxel_down_sample(voxel_size_m)
    return np.asarray(pc.points, dtype=np.float32), np.asarray(pc.colors, dtype=np.float32)


def normalize(xyz):
    """Center XY at the mean, shift Z so the floor sits near z=0."""
    out = xyz.astype(np.float32).copy()
    out[:, 0] -= xyz[:, 0].mean()
    out[:, 1] -= xyz[:, 1].mean()
    out[:, 2] -= xyz[:, 2].min()
    return out


def remove_structure(xyz, rgb, dist_thresh=0.05, min_plane_frac=0.04,
                     max_planes=10, horiz_normal=0.85, vert_normal=0.2):
    """Iteratively RANSAC-remove large floor/ceiling (horizontal) and wall
    (vertical) planes, keeping the rest as objects. No-op-ish if the scene has no
    big structural planes (e.g. data with walls/floor already removed). Returns
    (xyz_obj, rgb_obj, n_removed)."""
    import open3d as o3d
    pts, cols = xyz.copy(), rgb.copy()
    removed = 0
    for _ in range(max_planes):
        if len(pts) < 2000:
            break
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        model, inliers = pc.segment_plane(dist_thresh, ransac_n=3, num_iterations=300)
        if len(inliers) < min_plane_frac * len(pts):
            break  # no large plane left
        a, b, c, d = model
        nz = abs(c) / (np.sqrt(a * a + b * b + c * c) + 1e-9)
        is_structure = nz >= horiz_normal or nz <= vert_normal
        if not is_structure:
            break  # biggest remaining plane is object-like (slanted) -> stop
        keep = np.ones(len(pts), dtype=bool)
        keep[inliers] = False
        pts, cols = pts[keep], cols[keep]
        removed += 1
    return pts, cols, removed
