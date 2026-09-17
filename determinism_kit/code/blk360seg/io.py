"""Load BLK360 clouds from .csv (x,y,z,r,g,b) or raw .e57 -> (xyz Nx3, rgb Nx3)."""
import numpy as np


def load_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    xyz = df.iloc[:, :3].to_numpy(dtype=np.float32)
    if df.shape[1] >= 6:
        rgb = df.iloc[:, 3:6].to_numpy(dtype=np.float32)
        if rgb.max() > 1.5:            # 0-255 -> 0-1
            rgb /= 255.0
    else:
        rgb = np.zeros((len(df), 3), dtype=np.float32)
    return _clean(xyz, np.clip(rgb, 0.0, 1.0))


def load_e57(path, scan_index=None):
    """scan_index=None merges every setup in the file (poses applied by
    pye57); an int loads that single setup only. Multi-setup files come
    straight from Cyclone without merge-on-export."""
    import pye57
    e = pye57.E57(str(path))
    idxs = range(e.scan_count) if scan_index is None else [scan_index]
    xs, cs = [], []
    for i in idxs:
        d = e.read_scan(i, ignore_missing_fields=True, colors=True)
        xyz = np.column_stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]]).astype(np.float32)
        if "colorRed" in d:
            rgb = np.column_stack([d["colorRed"], d["colorGreen"], d["colorBlue"]]).astype(np.float32)
            if rgb.max() > 1.5:
                rgb /= 255.0
        else:
            rgb = np.zeros_like(xyz)
        xs.append(xyz)
        cs.append(np.clip(rgb, 0.0, 1.0))
    return _clean(np.concatenate(xs), np.concatenate(cs))


def load_ply(path):
    import open3d as o3d
    pc = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pc.points, dtype=np.float32)
    rgb = (np.asarray(pc.colors, dtype=np.float32) if pc.has_colors()
           else np.zeros_like(xyz))
    return _clean(xyz, np.clip(rgb, 0.0, 1.0))


def load(path, **kw):
    p = str(path).lower()
    if p.endswith(".e57"):
        return load_e57(path, **kw)
    if p.endswith(".ply"):
        return load_ply(path)
    return load_csv(path)


def _clean(xyz, rgb):
    m = np.isfinite(xyz).all(axis=1)
    return xyz[m], rgb[m]
