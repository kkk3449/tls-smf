"""Cluster points into object instances (DBSCAN)."""
import numpy as np


def cluster_all(xyz, eps_m=0.15, min_points=100):
    """Cluster the WHOLE cloud into objects (for wall/floor-removed scans, where
    each cluster is an object). Returns per-point instance id (-1 = noise)."""
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    return np.asarray(pc.cluster_dbscan(eps=eps_m, min_points=min_points))


def cluster_instances(xyz, labels, eps_m=0.12, min_points=40):
    import open3d as o3d
    inst = np.full(len(xyz), -1, dtype=np.int64)
    nid = 0
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if len(idx) < min_points:
            continue
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(xyz[idx].astype(np.float64))
        lab = np.asarray(pc.cluster_dbscan(eps=eps_m, min_points=min_points))
        for c in np.unique(lab[lab >= 0]):
            inst[idx[lab == c]] = nid
            nid += 1
    return inst, nid
