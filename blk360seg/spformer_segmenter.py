"""SPFormer instance segmentation on raw BLK360 clouds (Mask3D substitute).

SPFormer (AAAI'23, sunjiahao1999/SPFormer) is an off-the-shelf ScanNet-pretrained
superpoint mask-transformer. It normally consumes mesh-derived superpoints; our
data is a raw point cloud, so we generate point-based superpoints with a
Felzenszwalb-Huttenlocher graph over-segmentation on a kNN graph weighted by
normal dissimilarity (the point-cloud analogue of ScanNet's mesh `segmentator`).

Runtime needs the conda CUDA libs on LD_LIBRARY_PATH (set by scripts/segment_spformer.py).
"""
import os
import sys

import numpy as np
import torch

# ScanNet 18 instance classes (num_class=18; predict() returns 1-indexed labels).
SCANNET_INST = [
    "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator", "shower curtain",
    "toilet", "sink", "bathtub", "otherfurniture",
]


def estimate_normals(xyz, radius=0.1, max_nn=30):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return np.asarray(pc.normals, dtype=np.float32)


def generate_superpoints(xyz, knn=16, k_thresh=0.04, min_size=20):
    """Felzenszwalb-Huttenlocher over-segmentation on a kNN graph.

    Edge weight = normal dissimilarity (1 - |n_i . n_j|) + small spatial term, so
    points on one smooth surface merge. `k_thresh` controls superpoint size
    (bigger -> larger superpoints). Returns an int superpoint id per point.
    """
    from scipy.spatial import cKDTree
    n = len(xyz)
    normals = estimate_normals(xyz)
    tree = cKDTree(xyz)
    dist, idx = tree.query(xyz, k=knn + 1)        # includes self
    dist, idx = dist[:, 1:], idx[:, 1:]
    scale = np.median(dist) + 1e-9

    # build undirected edge list
    src = np.repeat(np.arange(n), knn)
    dst = idx.reshape(-1)
    ndot = np.abs((normals[src] * normals[dst]).sum(1)).clip(0, 1)
    w = (1.0 - ndot) + 0.2 * (dist.reshape(-1) / scale)
    keep = src < dst                              # dedup undirected
    src, dst, w = src[keep], dst[keep], w[keep]
    order = np.argsort(w, kind="stable")
    src, dst, w = src[order], dst[order], w[order]

    parent = np.arange(n)
    size = np.ones(n, dtype=np.int64)
    intd = np.zeros(n, dtype=np.float32)          # internal difference

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    for a, b, weight in zip(src.tolist(), dst.tolist(), w.tolist()):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if weight <= min(intd[ra] + k_thresh / size[ra],
                         intd[rb] + k_thresh / size[rb]):
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]
            intd[ra] = max(intd[ra], intd[rb], weight)

    # merge tiny segments into their nearest-neighbor segment
    roots = np.array([find(i) for i in range(n)])
    for a, b in zip(src.tolist(), dst.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb and (size[ra] < min_size or size[rb] < min_size):
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]
    roots = np.array([find(i) for i in range(n)])
    return np.unique(roots, return_inverse=True)[1].astype(np.int64)


class SPFormerSegmenter:
    def __init__(self, repo_dir, ckpt, config, device="cuda", scale=50, mode=4):
        sys.path.insert(0, os.path.abspath(repo_dir))
        import gorilla
        import pointgroup_ops                                       # noqa: F401
        from spformer.model import SPFormer

        self.pgops = pointgroup_ops
        self.device = device
        self.scale = scale
        self.mode = mode
        cfg = gorilla.Config.fromfile(config)
        self.spatial_shape = cfg.model.test_cfg  # unused; keep cfg around
        self.voxel_spatial = cfg.data.test.voxel_cfg.spatial_shape
        model = SPFormer(**cfg.model).to(device)
        model.eval()                              # note: SPFormer.train() returns None
        ck = torch.load(ckpt, map_location="cpu")
        sd = ck["model"] if "model" in ck else ck
        model.load_state_dict(sd, strict=False)
        self.model = model

    def _build_batch(self, xyz, rgb, superpoint):
        # replicate transform_test + collate_fn for a single scene
        xyz_middle = xyz.astype(np.float32)
        coord = (xyz_middle * self.scale)
        coord -= coord.min(0)
        coord = np.ascontiguousarray(coord)
        superpoint = np.unique(superpoint, return_inverse=True)[1]

        coord_t = torch.from_numpy(coord).long()
        coord_float = torch.from_numpy(xyz_middle).float()
        feat = torch.from_numpy(rgb.astype(np.float32))
        sp = torch.from_numpy(superpoint).long()

        coords = torch.cat([torch.zeros(coord_t.shape[0], 1, dtype=torch.long), coord_t], 1)
        feats = torch.cat((feat, coord_float), dim=1)   # use_xyz=True
        spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(),
                                self.voxel_spatial[0], None)
        voxel_coords, p2v_map, v2p_map = self.pgops.voxelization_idx(coords, 1, self.mode)
        return {
            "scan_ids": ["blk360"],
            "voxel_coords": voxel_coords,
            "p2v_map": p2v_map,
            "v2p_map": v2p_map,
            "spatial_shape": spatial_shape,
            "feats": feats,
            "superpoints": sp,
            "batch_offsets": torch.tensor([0, int(sp.max().item()) + 1], dtype=torch.int),
            "insts": [_DummyInst()],
        }

    @torch.no_grad()
    def segment(self, xyz, rgb, superpoint=None, knn=16, k_thresh=0.04):
        if superpoint is None:
            superpoint = generate_superpoints(xyz, knn=knn, k_thresh=k_thresh)
        batch = self._build_batch(xyz, rgb, superpoint)
        ret = self.model(batch, mode="predict")
        n = len(xyz)
        inst_labels = np.full(n, -1, dtype=np.int64)
        meta = []
        # paint instances by descending confidence so high-conf wins overlaps
        from spformer.utils.mask_encoder import rle_decode
        preds = sorted(ret["pred_instances"], key=lambda p: -p["conf"])
        for iid, p in enumerate(preds):
            mask = rle_decode(p["pred_mask"]).astype(bool)
            inst_labels[mask] = iid
            meta.append({"id": iid, "label_id": int(p["label_id"]),
                         "class": SCANNET_INST[int(p["label_id"]) - 1],
                         "conf": float(p["conf"]), "n": int(mask.sum())})
        return inst_labels, meta, superpoint


class _DummyInst:
    gt_instances = np.zeros(0, dtype=np.int64)
