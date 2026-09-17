"""PTv3 (Point Transformer v3) semantic segmentation on raw BLK360 clouds.

PTv3 (Wu et al., CVPR'24, Pointcept) is a ScanNet-pretrained *semantic* segmenter
(per-point class among 20 ScanNet classes — no instances). Like SPFormer it is a
closed-set indoor model; we run it to add a third, semantic, arm to the benchmark
and show the same out-of-distribution failure (industrial geometry forced onto
furniture classes).

We replicate the Pointcept eval transform for a single scene:
  CenterShift(apply_z=True) -> GridSample(0.02, test) -> CenterShift(apply_z=False)
  -> feat = [color/255, normal]  (in_channels=6)
The model returns per-voxel features; we scatter the argmax back to every point
through the voxelization inverse map.

Runtime needs the conda CUDA libs on LD_LIBRARY_PATH (set by scripts/segment_ptv3.py).
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# ScanNet 20 semantic classes (DefaultSegmentorV2 num_classes=20).
SCANNET20 = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub",
    "otherfurniture",
]

# Exact backbone config from configs/scannet/semseg-pt-v3m1-0-base.py.
_BACKBONE = dict(
    in_channels=6,
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    dec_depths=(2, 2, 2, 2),
    dec_channels=(64, 64, 128, 256),
    dec_num_head=(4, 4, 8, 16),
    dec_patch_size=(1024, 1024, 1024, 1024),
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.0,            # eval
    shuffle_orders=True,
    pre_norm=True,
    enable_rpe=False,
    enable_flash=False,       # no flash-attn on this env -> SDPA fallback
    upcast_attention=False,
    upcast_softmax=False,
    enc_mode=False,
    pdnorm_bn=False,
    pdnorm_ln=False,
    pdnorm_decouple=True,
    pdnorm_adaptive=False,
    pdnorm_affine=True,
    pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
)


def estimate_normals(xyz, radius=0.1, max_nn=30):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pc.normalize_normals()
    return np.asarray(pc.normals, dtype=np.float32)


def _center_shift(coord, apply_z):
    x_min, y_min, z_min = coord.min(0)
    x_max, y_max, _ = coord.max(0)
    shift = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2,
                      z_min if apply_z else 0.0], dtype=np.float32)
    return coord - shift


def _voxelize(coord, grid_size):
    """GridSample test-mode: one representative point per voxel + inverse map."""
    grid = np.floor(coord / grid_size).astype(np.int64)
    grid -= grid.min(0)
    keys = grid.astype(np.int64)
    # hash voxel coords to a unique key, then unique with inverse
    k = (keys[:, 0].astype(np.int64) * 1_000_003 + keys[:, 1]) * 1_000_003 + keys[:, 2]
    _, first_idx, inverse = np.unique(k, return_index=True, return_inverse=True)
    return first_idx, inverse, grid


class PTv3Segmenter:
    def __init__(self, repo_dir, ckpt, device="cuda", grid_size=0.02):
        sys.path.insert(0, os.path.abspath(repo_dir))
        from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
            PointTransformerV3,
        )
        self.device = device
        self.grid_size = grid_size
        self.backbone = PointTransformerV3(**_BACKBONE).to(device)
        self.seg_head = nn.Linear(64, len(SCANNET20)).to(device)

        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = ck["state_dict"] if "state_dict" in ck else ck
        bb, head = {}, {}
        for k, v in sd.items():
            k = k[len("module."):] if k.startswith("module.") else k
            if k.startswith("backbone."):
                bb[k[len("backbone."):]] = v
            elif k.startswith("seg_head."):
                head[k[len("seg_head."):]] = v
        mb = self.backbone.load_state_dict(bb, strict=False)
        mh = self.seg_head.load_state_dict(head, strict=False)
        print(f"[ptv3] backbone missing={len(mb.missing_keys)} unexpected={len(mb.unexpected_keys)}; "
              f"seg_head loaded={len(head)}")
        self.backbone.eval()
        self.seg_head.eval()

    @torch.no_grad()
    def segment(self, xyz, rgb):
        """Return per-point ScanNet-20 labels (int) for every input point."""
        xyz = xyz.astype(np.float32)
        rgb = rgb.astype(np.float32)
        if rgb.max() > 1.5:          # NormalizeColor expects [0,255] -> [0,1]
            rgb = rgb / 255.0
        normals = estimate_normals(xyz)

        coord = _center_shift(xyz, apply_z=True)
        first_idx, inverse, _ = _voxelize(coord, self.grid_size)
        coord_v = coord[first_idx]
        coord_v = _center_shift(coord_v, apply_z=False)
        grid_v = np.floor(coord[first_idx] / self.grid_size).astype(np.int64)
        grid_v -= grid_v.min(0)
        feat_v = np.concatenate([rgb[first_idx], normals[first_idx]], axis=1)

        nv = len(first_idx)
        data = {
            "coord": torch.from_numpy(np.ascontiguousarray(coord_v)).float().to(self.device),
            "grid_coord": torch.from_numpy(np.ascontiguousarray(grid_v)).int().to(self.device),
            "feat": torch.from_numpy(np.ascontiguousarray(feat_v)).float().to(self.device),
            "offset": torch.tensor([nv], dtype=torch.long, device=self.device),
        }
        point = self.backbone(data)
        feat = point.feat if hasattr(point, "feat") else point["feat"]
        logits = self.seg_head(feat)                 # [Nv, 20]
        pred_v = logits.argmax(1).cpu().numpy()      # per voxel
        conf_v = torch.softmax(logits, 1).max(1).values.cpu().numpy()
        labels = pred_v[inverse]                     # scatter back to all points
        conf = conf_v[inverse]
        return labels.astype(np.int64), conf.astype(np.float32)
