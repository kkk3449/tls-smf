"""Per-point semantic segmentation backends.

`Segmenter.segment(xyz, rgb) -> int label per point` (label = index into the
active class set, see classes.py).
"""
import numpy as np

from .classes import BASELINE_LABELS


class Segmenter:
    def segment(self, xyz, rgb):
        raise NotImplementedError


class GeometricBaseline(Segmenter):
    """Runnable placeholder until the DL model is wired. Classifies by surface
    orientation (normals) + height into ceiling / floor / wall / clutter."""

    def __init__(self, floor_distance_m=0.04, cluster_eps_m=0.10,
                 cluster_min_points=50, horiz_normal=0.85, vert_normal=0.35):
        self.band = max(0.3, floor_distance_m * 5)
        self.horiz = horiz_normal
        self.vert = vert_normal

    def segment(self, xyz, rgb):
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pc.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
        nz = np.abs(np.asarray(pc.normals)[:, 2])
        z = xyz[:, 2]
        zmin, zmax = z.min(), z.max()
        labels = np.full(len(xyz), BASELINE_LABELS["clutter"], dtype=np.int64)
        labels[nz <= self.vert] = BASELINE_LABELS["wall"]
        labels[(nz >= self.horiz) & (z <= zmin + self.band)] = BASELINE_LABELS["floor"]
        labels[(nz >= self.horiz) & (z >= zmax - self.band)] = BASELINE_LABELS["ceiling"]
        return labels


class DLSegmenter(Segmenter):
    """STUB. Wrap a pretrained point-cloud semantic-seg model (recommended:
    Pointcept PTv3 / SparseUNet pretrained on ScanNet or S3DIS)."""

    def __init__(self, framework="pointcept", weights="", device="cuda"):
        self.framework, self.weights, self.device = framework, weights, device

    def segment(self, xyz, rgb):
        raise NotImplementedError(
            "DLSegmenter is not wired yet. To enable:\n"
            "  1) pip install a CUDA torch build + the framework (e.g. Pointcept)\n"
            "  2) load a pretrained ScanNet/S3DIS checkpoint in __init__\n"
            "  3) tile + normalize points to the model input, run inference, and\n"
            "     scatter predictions back to per-point labels here.\n"
            "Until then set segmenter.backend=baseline in the config.")


def build(cfg):
    seg = cfg.get("segmenter", {})
    if seg.get("backend", "baseline") == "dl":
        return DLSegmenter(**seg.get("dl", {}))
    return GeometricBaseline(**seg.get("baseline", {}))
