"""e57 파일 메타데이터 검사: 스캔 개수, 점 개수, 좌표 범위(bounding box)."""
import sys
import numpy as np
import pye57

PATH = "testroom260601.e57"

e57 = pye57.E57(PATH)
n_scans = e57.scan_count
print(f"scan_count = {n_scans}")

# 전체 점 범위 누적 (스캔별로 읽어 메모리 절약)
gmin = np.array([np.inf, np.inf, np.inf])
gmax = np.array([-np.inf, -np.inf, -np.inf])
total_pts = 0

for i in range(n_scans):
    header = e57.get_header(i)
    print(f"\n--- scan {i} ---")
    print("  point_count:", header.point_count)
    try:
        print("  fields:", header.point_fields)
    except Exception:
        pass

    data = e57.read_scan(i, ignore_missing_fields=True, colors=True, intensity=True)
    x = np.asarray(data["cartesianX"])
    y = np.asarray(data["cartesianY"])
    z = np.asarray(data["cartesianZ"])
    pts = np.vstack([x, y, z]).T
    # 유효(유한) 점만
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    total_pts += pts.shape[0]

    smin = pts.min(axis=0)
    smax = pts.max(axis=0)
    gmin = np.minimum(gmin, smin)
    gmax = np.maximum(gmax, smax)
    print(f"  bbox min: {smin}")
    print(f"  bbox max: {smax}")
    print(f"  available keys: {list(data.keys())}")

print("\n===== GLOBAL =====")
print(f"total valid points = {total_pts:,}")
print(f"global bbox min = {gmin}")
print(f"global bbox max = {gmax}")
print(f"size (X,Y,Z) = {gmax - gmin}")
print(f"center = {(gmin + gmax) / 2}")
