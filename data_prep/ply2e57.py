#!/usr/bin/env python3
"""PLY → E57 변환 (좌표 + RGB 색상 보존).

사용:
  .venv/bin/python ply2e57.py stage2_no_wall.ply stage2_no_wall.e57
"""
import sys
import numpy as np
import open3d as o3d
import pye57

src = sys.argv[1] if len(sys.argv) > 1 else "stage2_no_wall.ply"
dst = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".e57"

pcd = o3d.io.read_point_cloud(src)
pts = np.asarray(pcd.points)
print(f"[load] {src}: {len(pts):,} points")

data = {
    "cartesianX": np.ascontiguousarray(pts[:, 0]),
    "cartesianY": np.ascontiguousarray(pts[:, 1]),
    "cartesianZ": np.ascontiguousarray(pts[:, 2]),
}

if pcd.has_colors():
    rgb = (np.asarray(pcd.colors) * 255.0).round().clip(0, 255).astype(np.uint8)
    data["colorRed"] = np.ascontiguousarray(rgb[:, 0])
    data["colorGreen"] = np.ascontiguousarray(rgb[:, 1])
    data["colorBlue"] = np.ascontiguousarray(rgb[:, 2])
    print("[info] RGB 색상 포함")

e57 = pye57.E57(dst, mode="w")
e57.write_scan_raw(data, name="stage2_no_wall")
e57.close()
print(f"[save] {dst}")
