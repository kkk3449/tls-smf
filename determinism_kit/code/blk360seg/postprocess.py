"""Cluster points into object instances (DBSCAN)."""
import numpy as np


def cluster_all(xyz, eps_m=0.15, min_points=100):
    """Cluster the WHOLE cloud into objects (for wall/floor-removed scans, where
    each cluster is an object). Returns per-point instance id (-1 = noise)."""
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    return np.asarray(pc.cluster_dbscan(eps=eps_m, min_points=min_points))


def cluster_all_split(xyz, eps_m=0.30, min_points=100,
                      split_footprint_m=1.2, split_eps_m=0.12, split_min_points=60):
    """DBSCAN at eps_m, then CONDITIONALLY re-split only the over-merged clusters.

    A cluster whose horizontal footprint (max of x/y extent) exceeds
    `split_footprint_m` is re-clustered at the tighter `split_eps_m`; the split is
    accepted only if it yields >=2 sub-clusters each >= `split_min_points`
    (otherwise the original cluster is kept intact). This separates e.g. two
    chairs merged into one 2 m blob without shattering a genuinely large object.
    """
    import open3d as o3d
    base = cluster_all(xyz, eps_m=eps_m, min_points=min_points)
    out = np.full(len(xyz), -1, dtype=np.int64)
    nid = 0
    n_split = 0
    for cls in np.unique(base[base >= 0]):
        idx = np.where(base == cls)[0]
        sub = xyz[idx]
        foot = float(max(np.ptp(sub[:, 0]), np.ptp(sub[:, 1])))
        if foot <= split_footprint_m:
            out[idx] = nid
            nid += 1
            continue
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(sub.astype(np.float64))
        lab = np.asarray(pc.cluster_dbscan(eps=split_eps_m, min_points=split_min_points))
        good = [c for c in np.unique(lab[lab >= 0])
                if int((lab == c).sum()) >= split_min_points]
        if len(good) >= 2:                          # accept the split
            n_split += 1
            for c in good:
                out[idx[lab == c]] = nid
                nid += 1
        else:                                       # keep intact
            out[idx] = nid
            nid += 1
    return out, nid, n_split


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
