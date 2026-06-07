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


def load_e57(path, scan_index=0):
    import pye57
    e = pye57.E57(str(path))
    d = e.read_scan(scan_index, ignore_missing_fields=True, colors=True)
    xyz = np.column_stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]]).astype(np.float32)
    if "colorRed" in d:
        rgb = np.column_stack([d["colorRed"], d["colorGreen"], d["colorBlue"]]).astype(np.float32)
        if rgb.max() > 1.5:
            rgb /= 255.0
    else:
        rgb = np.zeros_like(xyz)
    return _clean(xyz, np.clip(rgb, 0.0, 1.0))


def load(path, **kw):
    return load_e57(path, **kw) if str(path).lower().endswith(".e57") else load_csv(path)


def _clean(xyz, rgb):
    m = np.isfinite(xyz).all(axis=1)
    return xyz[m], rgb[m]
