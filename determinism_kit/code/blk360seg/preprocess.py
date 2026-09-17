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


def _ransac_plane(pts, dist_thresh, num_iterations, rng):
    """Seeded plane RANSAC (Open3D's segment_plane has no seed -> re-runs
    differ; a fixed rng makes the whole preprocessing chain reproducible,
    which the knowledge-graph diffing relies on). Returns (model, inliers)
    like segment_plane: model = (a, b, c, d), inlier index array."""
    n = len(pts)
    triples = rng.integers(0, n, size=(num_iterations, 3))
    best_count, best = -1, None
    for i0, i1, i2 in triples:
        p0 = pts[i0]
        nvec = np.cross(pts[i1] - p0, pts[i2] - p0)
        norm = np.linalg.norm(nvec)
        if norm < 1e-9:
            continue
        nvec = nvec / norm
        dist = np.abs((pts - p0) @ nvec)
        count = int((dist < dist_thresh).sum())
        if count > best_count:
            best_count, best = count, (nvec, p0)
    if best is None:
        return (0.0, 0.0, 1.0, 0.0), np.empty(0, dtype=np.int64)
    nvec, p0 = best
    inl = np.abs((pts - p0) @ nvec) < dist_thresh
    # one least-squares refit on the inliers (SVD -> deterministic), then final
    # inlier set from the refit plane, mirroring segment_plane's behavior
    q = pts[inl]
    c = q.mean(0)
    nvec = np.linalg.svd(q - c, full_matrices=False)[2][-1]
    inliers = np.where(np.abs((pts - c) @ nvec) < dist_thresh)[0]
    return (float(nvec[0]), float(nvec[1]), float(nvec[2]),
            float(-nvec @ c)), inliers


def remove_structure(xyz, rgb, dist_thresh=0.05, min_plane_frac=0.04,
                     max_planes=10, horiz_normal=0.85, vert_normal=0.2,
                     seed=0):
    """Iteratively RANSAC-remove large floor/ceiling (horizontal) and wall
    (vertical) planes, keeping the rest as objects. No-op-ish if the scene has no
    big structural planes (e.g. data with walls/floor already removed). Returns
    (xyz_obj, rgb_obj, n_removed). Seeded RANSAC: identical re-runs yield
    identical outputs."""
    pts, cols = xyz.copy(), rgb.copy()
    rng = np.random.default_rng(seed)
    removed = 0
    for _ in range(max_planes):
        if len(pts) < 2000:
            break
        model, inliers = _ransac_plane(pts.astype(np.float64), dist_thresh,
                                       num_iterations=300, rng=rng)
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
