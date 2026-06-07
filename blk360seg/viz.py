"""Color points by label and save / show."""
import numpy as np

from .classes import palette


def color_by_label(labels, n_classes):
    pal = palette(max(n_classes, int(labels.max()) + 1))
    return pal[np.clip(labels, 0, len(pal) - 1)]


def save_ply(path, xyz, colors):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(path, pc)


def save_labeled_csv(path, xyz, labels, instances=None):
    import pandas as pd
    d = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "label": labels}
    if instances is not None:
        d["instance"] = instances
    pd.DataFrame(d).to_csv(path, index=False)


def show(xyz, colors):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.visualization.draw_geometries([pc])
