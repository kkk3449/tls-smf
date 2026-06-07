#!/usr/bin/env python3
"""Cyclone360 테스트룸 포인트클라우드 전처리.

1단계: 슬라이싱(AABB 크롭) + 통계적 이상치 제거(노이즈 제거)
2단계: RANSAC 평면 분할로 벽/천장/바닥 제거

사용 예:
  .venv/bin/python preprocess.py --stage all
  .venv/bin/python preprocess.py --stage 1 --voxel 0.01
  .venv/bin/python preprocess.py --stage 2 --in stage1_clean.ply
"""
import argparse
import numpy as np
import open3d as o3d
import pye57

SRC = "testroom260601.e57"

# --- e57 검사로 결정한 슬라이싱 박스 (단위: m, Z가 높이축) ---
CROP_MIN = (-6.1, -6.0, -1.05)   # 바닥 살짝 아래
CROP_MAX = (9.7, 4.6, 2.55)      # 천장 살짝 위


def load_e57(path):
    """e57 → Open3D PointCloud (좌표 + RGB). 무효(NaN/Inf) 점 제거."""
    print(f"[load] reading {path} ...")
    e57 = pye57.E57(path)
    d = e57.read_scan(0, ignore_missing_fields=True, colors=True, intensity=False)
    pts = np.vstack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]]).T
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if "colorRed" in d:
        rgb = np.vstack([d["colorRed"], d["colorGreen"], d["colorBlue"]]).T[finite]
        if rgb.max() > 1.0:          # 0..255 → 0..1
            rgb = rgb / 255.0
        pcd.colors = o3d.utility.Vector3dVector(rgb)

    print(f"[load] {len(pcd.points):,} points")
    return pcd


def stage1(pcd, voxel=0.0, nb_neighbors=20, std_ratio=2.0):
    """슬라이싱 + (옵션) 다운샘플 + 통계적 이상치 제거."""
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=CROP_MIN, max_bound=CROP_MAX)
    pcd = pcd.crop(aabb)
    print(f"[stage1] after crop: {len(pcd.points):,}")

    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
        print(f"[stage1] after voxel({voxel}m): {len(pcd.points):,}")

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"[stage1] after SOR: {len(pcd.points):,}")
    return pcd


def stage2(pcd, dist_ceil=0.03, wall_margin=0.02, floor_margin=0.02,
           max_planes=20, min_abs=150_000, margin_frac=0.12, wall_margins=None):
    """RANSAC으로 벽/바닥 평면을 찾아 'half-space 슬라이스'로 깔끔히 잘라낸다.

    동작:
    - 평면 검출을 반복하며 평면 종류를 분류한다.
      * 천장(수평면, 위쪽)  → 두께 dist_ceil 로 제거(단차 천장 대응)
      * 바닥(수평면, 아래쪽) → 경계 평면으로 등록
      * 외벽(수직면, 외곽)   → 경계 평면으로 등록
      * 내부 수직면/기울어진 면 → 보존
    - 마지막에 등록된 바닥·벽 평면마다 '방 안쪽으로 margin 이상 들어간 점만 유지'
      하는 슬라이스를 적용한다. 평면과 그 바깥쪽은 통째로 잘려 벽면 잔재가 남지
      않고, 벽에서 margin 이상 떨어진 가구·거치대·TV는 100% 보존된다.

    wall_margin / floor_margin: 벽·바닥 평면에서 안쪽으로 이만큼까지 잘라낸다.
    wall_margins: 벽별 마진 덮어쓰기 dict. 키는 'X+','X-','Y+','Y-'
        (X/Y축의 높은 쪽/낮은 쪽 벽). 지정 안 된 벽은 wall_margin 사용.
    """
    wall_margins = wall_margins or {}
    detect_dist = 0.03           # 평면 검출 임계값(천장 두께와 분리)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    pts_all = np.asarray(pcd.points)
    lo = np.percentile(pts_all, 1, axis=0)    # 이상치 배제한 방 경계
    hi = np.percentile(pts_all, 99, axis=0)
    margin = (hi - lo) * margin_frac
    z_mid = (lo[2] + hi[2]) / 2          # 천장/바닥 구분 기준 높이
    room_center = (lo + hi) / 2          # 안/밖 판정 기준점
    print(f"[stage2] room bbox lo={lo.round(2)} hi={hi.round(2)}  z_mid={z_mid:.2f}")

    remaining = pcd
    kept = o3d.geometry.PointCloud()     # 보존(내부 수직면/기울어진 면)
    sliced = []                          # 슬라이스한 경계 종류 기록
    done = set()                         # 이미 슬라이스한 방향(바닥/+X/-X/+Y/-Y)

    for i in range(max_planes):
        if len(remaining.points) < min_abs:
            break
        model, inliers = remaining.segment_plane(
            distance_threshold=detect_dist, ransac_n=3, num_iterations=1000)
        if len(inliers) < min_abs:
            print(f"[stage2] plane {i}: only {len(inliers):,} pts (< {min_abs:,}), stop")
            break

        a, b, c, d = model           # 단위 법선(a²+b²+c²=1)
        n = np.array([a, b, c])
        seg = remaining.select_by_index(inliers)
        centroid = np.asarray(seg.points).mean(axis=0)

        if abs(n[2]) > 0.9 and centroid[2] > z_mid:
            # 천장 → 두께 제거(단차 대응)
            pts = np.asarray(remaining.points)
            sel = np.where(np.abs(pts @ n + d) <= dist_ceil)[0]
            print(f"[stage2] plane {i}: 천장 제거 {len(sel):,} pts "
                  f"(두께 {dist_ceil*100:.0f}cm), n={n.round(2)}")
            remaining = remaining.select_by_index(sel, invert=True)
            continue

        if abs(n[2]) > 0.9:
            kind, m, key = "바닥", floor_margin, "floor"
        elif abs(n[2]) < 0.2:
            ax = int(np.argmax(np.abs(n[:2])))
            at_edge = (centroid[ax] < lo[ax] + margin[ax]) or \
                      (centroid[ax] > hi[ax] - margin[ax])
            if not at_edge:
                print(f"[stage2] plane {i}: 내부수직면 보존 {len(inliers):,} pts")
                kept += seg
                remaining = remaining.select_by_index(inliers, invert=True)
                continue
            side = '+' if centroid[ax] > room_center[ax] else '-'
            wlabel = f"{'XY'[ax]}{side}"          # 예: 'Y+', 'X-'
            m = wall_margins.get(wlabel, wall_margin)
            kind, key = f"외벽({wlabel})", ("wall", ax, side)
        else:
            print(f"[stage2] plane {i}: 기울어진면 보존 {len(inliers):,} pts")
            kept += seg
            remaining = remaining.select_by_index(inliers, invert=True)
            continue

        # 같은 방향을 이미 슬라이스한 경우의 재검출 처리.
        if key in done:
            # 바닥 높이 근처(±10cm)의 수평면 = 기울어진 바닥 잔재 → 제거.
            # 그보다 높은 수평면 = 가구 상판/선반, 벽 근처 잔여면 = 물체 → 보존.
            floor_remnant = (key == "floor" and centroid[2] < lo[2] + 0.10)
            if floor_remnant:
                print(f"[stage2] plane {i}: 바닥 잔재 제거 {len(inliers):,} pts "
                      f"(z={centroid[2]:.2f})")
            else:
                print(f"[stage2] plane {i}: {kind} 재검출 → 물체로 보존 {len(inliers):,} pts")
                kept += seg
            remaining = remaining.select_by_index(inliers, invert=True)
            continue

        # 방향별 첫(=가장 큰) 평면만 슬라이스. 평면에서 방 안쪽 m 미만(평면+바깥)을 제거.
        pts = np.asarray(remaining.points)
        s = np.sign(room_center @ n + d) or 1.0
        inside = (pts @ n + d) * s
        sel = np.where(inside < m)[0]
        done.add(key)
        sliced.append(kind)
        print(f"[stage2] plane {i}: {kind} 슬라이스 제거 {len(sel):,} pts "
              f"(margin {m*100:.0f}cm), n={n.round(2)}")
        remaining = remaining.select_by_index(sel, invert=True)

    result = remaining + kept
    print(f"[stage2] 슬라이스 {len(sliced)}회({', '.join(sliced)}); "
          f"남은 점 {len(result.points):,}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["1", "2", "all"], default="all")
    ap.add_argument("--in", dest="infile", default=None, help="입력 ply (stage2 단독 실행용)")
    ap.add_argument("--voxel", type=float, default=0.0, help="voxel 크기(m), 0이면 다운샘플 안함")
    ap.add_argument("--dist-ceil", type=float, default=0.03, help="천장 제거 두께(m)")
    ap.add_argument("--wall-margin", type=float, default=0.02,
                    help="모든 벽 기본 마진(m). 작을수록 벽 근처 물체 보존")
    ap.add_argument("--wall-margins", default=None,
                    help="벽별 마진 덮어쓰기. 예: 'Y+=0.05,X-=0.03' "
                         "(X/Y축의 높은 쪽=+, 낮은 쪽=-). 미지정 벽은 --wall-margin 사용")
    ap.add_argument("--floor-margin", type=float, default=0.02,
                    help="바닥 평면에서 위로 잘라낼 거리(m)")
    ap.add_argument("--min-abs", type=int, default=150_000, help="평면 최소 점 개수")
    ap.add_argument("--max-planes", type=int, default=20, help="검출할 최대 평면 수")
    args = ap.parse_args()

    wall_margins = {}
    if args.wall_margins:
        for item in args.wall_margins.split(","):
            k, v = item.split("=")
            wall_margins[k.strip()] = float(v)

    if args.stage in ("1", "all"):
        pcd = load_e57(SRC)
        s1 = stage1(pcd, voxel=args.voxel)
        o3d.io.write_point_cloud("stage1_clean.ply", s1)
        print("[save] stage1_clean.ply")
    if args.stage == "2":
        s1 = o3d.io.read_point_cloud(args.infile or "stage1_clean.ply")
    if args.stage in ("2", "all"):
        s2 = stage2(s1, dist_ceil=args.dist_ceil, wall_margin=args.wall_margin,
                    floor_margin=args.floor_margin, min_abs=args.min_abs,
                    max_planes=args.max_planes, wall_margins=wall_margins)
        o3d.io.write_point_cloud("stage2_no_wall.ply", s2)
        print("[save] stage2_no_wall.ply")


if __name__ == "__main__":
    main()
