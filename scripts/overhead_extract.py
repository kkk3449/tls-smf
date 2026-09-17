#!/usr/bin/env python3
"""Overhead-structure recognition pass (ceiling-mounted objects).

The floor pass (extract_objects.py) removes the ceiling planes and therefore
discards everything that hangs from them.  This complementary pass runs on the
*pre-removal* reference cloud (ceilingref.ply) and recovers ceiling-mounted /
overhead content as its own object set, without touching the floor pass:

  1. floor z from the reference cloud (densest low z-bin);
  2. ceiling levels: iterative seeded RANSAC for near-horizontal planes above
     the driving band; points within +-ceil_band of a level are 'ceiling';
  3. walls: seeded RANSAC for near-vertical planes in the overhead band;
     points within wall_band of a wall plane are 'wall';
  4. candidates = overhead-band points that are neither ceiling nor wall and
     do not belong to the upper part of an already-detected floor object
     (footprint test against the floor-pass semanticObjects);
  5. fixed-parameter DBSCAN (same eps/min-points family as the floor pass,
     with the conservative giant-split) -> overhead clusters;
  6. explicit model per cluster (same routine as the floor pass) plus
     overhead attributes: clearanceBelow (bottom above floor), attachment
     (ceiling-hung / wall-mounted / suspended), heightLevel = "high";
  7. writes obj_*.ply + objects.csv + semanticObjects.json (Stage-A style)
     -> Stage B renders from below + late-fusion verification as usual.

  .venv/bin/python scripts/overhead_extract.py \
      --ref outputs/vis_sota_det/ceilingref.ply \
      --raw "/home/caselab/Downloads/DataSend/vis_sota_1.e57" \
      --floor-objects outputs/vis_sota_det5/semanticObjects.lf_esc.json \
      --out outputs/vis_sota_overhead
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from blk360seg import io, preprocess, postprocess, objects  # noqa: E402
from blk360seg.preprocess import _ransac_plane  # noqa: E402
from blk360seg.structure_filter import estimate_floor_z  # noqa: E402
from blk360seg.semantic_object import (object_geometry,  # noqa: E402
                                       nearest_color, build_message)


def horizontal_levels(pts, rng, dist=0.05, max_planes=4, min_frac=0.03,
                      nz_min=0.85):
    """Iteratively find near-horizontal planes; returns list of (z, n_inl)."""
    levels, keep = [], np.ones(len(pts), bool)
    for _ in range(max_planes):
        cur = pts[keep]
        if len(cur) < 2000:
            break
        model, inl = _ransac_plane(cur.astype(np.float64), dist, 300, rng)
        if len(inl) < min_frac * len(cur):
            break
        a, b, c, d = model
        nz = abs(c) / (np.sqrt(a * a + b * b + c * c) + 1e-9)
        if nz < nz_min:          # biggest plane left is not horizontal
            # remove it anyway from the search pool but do not record
            idx = np.where(keep)[0][inl]
            keep[idx] = False
            continue
        z = float(np.median(cur[inl, 2]))
        levels.append((z, int(len(inl))))
        idx = np.where(keep)[0][inl]
        keep[idx] = False
    return sorted(levels)


def vertical_planes(pts, rng, dist=0.05, max_planes=8, min_frac=0.03,
                    nz_max=0.2):
    """Iteratively find near-vertical planes; returns list of (n, p0)."""
    planes, keep = [], np.ones(len(pts), bool)
    for _ in range(max_planes):
        cur = pts[keep]
        if len(cur) < 2000:
            break
        model, inl = _ransac_plane(cur.astype(np.float64), dist, 300, rng)
        if len(inl) < min_frac * len(cur):
            break
        a, b, c, d = model
        n = np.array([a, b, c]); n = n / (np.linalg.norm(n) + 1e-9)
        idx = np.where(keep)[0][inl]
        keep[idx] = False
        if abs(n[2]) <= nz_max:
            planes.append((n, float(d)))
    return planes


def dist_to_planes(pts, planes):
    if not planes:
        return np.full(len(pts), np.inf)
    d = np.full(len(pts), np.inf)
    for n, dd in planes:
        d = np.minimum(d, np.abs(pts @ n + dd))
    return d


def inside_footprint(xy, o, pad):
    """Point-in-rotated-rectangle test for a floor-pass record."""
    cx, cy, th = o["poseX"], o["poseY"], o.get("poseTheta", 0.0)
    L = o["dimensions"]["length"] / 2 + pad
    W = o["dimensions"]["width"] / 2 + pad
    q = xy - np.array([cx, cy])
    ca, sa = np.cos(-th), np.sin(-th)
    x = q[:, 0] * ca - q[:, 1] * sa
    y = q[:, 0] * sa + q[:, 1] * ca
    return (np.abs(x) <= L) & (np.abs(y) <= W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="ceilingref.ply (pre-removal)")
    ap.add_argument("--raw", default=None,
                    help="raw e57; if given, rebuilds the colored downsampled "
                         "cloud (deterministic) so overhead objects keep RGB")
    ap.add_argument("--conn-eps", type=float, default=0.20,
                    help="DBSCAN eps for the floor-connectivity test only")
    ap.add_argument("--clean", default=None,
                    help="floor-pass clean.ply (structure removed); when given, "
                         "candidates are clustered together with the lower cloud "
                         "and a cluster is overhead only if it is NOT connected "
                         "down into the driving band (zmin > floor + drive_top - 0.1)")
    ap.add_argument("--floor-objects", default=None,
                    help="floor-pass semanticObjects json in the SAME frame; "
                         "upper parts of those objects are excluded")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    ap.add_argument("--drive-top", type=float, default=1.8,
                    help="top of the driving band above floor [m]")
    ap.add_argument("--ceil-band", type=float, default=0.06)
    ap.add_argument("--wall-band", type=float, default=0.12)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--min-points", type=int, default=80)
    ap.add_argument("--floor-pad", type=float, default=0.10)
    ap.add_argument("--max-floor-footprint", type=float, default=3.0,
                    help="floor-pass objects wider than this are treated as "
                         "merged clusters and do not exclude overhead points")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    oc = cfg["objects"]
    eps = args.eps or oc["cluster_eps_m"]
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    xyz, rgb = io.load(args.ref)
    print(f"[O] reference cloud {len(xyz):,} pts (colors: {bool(rgb.any())})")
    if args.raw and not rgb.any():
        rx, rc = io.load(args.raw)
        rx, rc = preprocess.voxel_downsample(rx, rc, cfg["preprocess"]["voxel_size_m"])
        print(f"[O] rebuilt colored cloud from raw: {len(rx):,} pts")
        if len(rx) == len(xyz) and np.allclose(rx, xyz, atol=1e-4):
            rgb = rc
            print("[O] byte-consistent with the reference; colors attached")
        else:  # nearest-neighbour color transfer (should not be needed)
            from scipy.spatial import cKDTree
            _, ii = cKDTree(rx).query(xyz, k=1)
            rgb = rc[ii]
            print("[O] colors transferred by nearest neighbour")
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
        o3d.io.write_point_cloud(os.path.join(args.out, "ceilingref_rgb.ply"), pc)

    floor_z = estimate_floor_z(xyz)
    band = xyz[:, 2] > floor_z + args.drive_top
    P, C = xyz[band], rgb[band]
    print(f"[O] floor z = {floor_z:.3f}; overhead band (> +{args.drive_top} m): "
          f"{len(P):,} pts")

    levels = horizontal_levels(P, rng)
    print("[O] ceiling levels (z, inliers): " +
          ", ".join(f"{z:.2f} ({n})" for z, n in levels))
    zc = np.array([z for z, _ in levels]) if levels else np.array([P[:, 2].max()])
    is_ceiling = np.min(np.abs(P[:, 2][:, None] - zc[None, :]), 1) < args.ceil_band
    # anything above the lowest ceiling level that is not itself a ceiling
    # level (content seen through openings) is out of scope
    above = P[:, 2] > zc.min() + args.ceil_band
    walls = vertical_planes(P[~is_ceiling], rng)
    dwall = dist_to_planes(P, walls)
    is_wall = dwall < args.wall_band
    cand = ~is_ceiling & ~is_wall & ~(above & ~is_ceiling & (P[:, 2] > zc.max() + args.ceil_band))
    print(f"[O] ceiling pts {is_ceiling.sum():,}, wall pts {is_wall.sum():,} "
          f"({len(walls)} planes), candidates {cand.sum():,}")

    floor_objs = []
    if args.floor_objects:
        d = json.load(open(args.floor_objects))
        floor_objs = d["semanticObjects"] if isinstance(d, dict) else d
        excl = np.zeros(len(P), bool)
        for o in floor_objs:
            top = o["properties"].get("poseZ", 0.0) + o["dimensions"]["height"] / 2
            if top < floor_z + args.drive_top - 0.3:
                continue          # object entirely inside the driving band
            if max(o["dimensions"]["length"], o["dimensions"]["width"]) > args.max_floor_footprint:
                continue          # merged mega-cluster: its footprint is not an object
            m = inside_footprint(P[:, :2], o, args.floor_pad) & (P[:, 2] <= top + 0.15)
            excl |= m
        print(f"[O] excluded {excl.sum():,} pts belonging to the upper parts of "
              f"{len(floor_objs)} floor-pass objects")
        cand &= ~excl

    Q, QC, Qd = P[cand], C[cand], dwall[cand]
    if args.clean:
        lx, lc = io.load(args.clean)
        lm = (lx[:, 2] > floor_z + 0.10) & (lx[:, 2] <= floor_z + args.drive_top + 0.05)
        lx, lc = lx[lm], lc[lm]
        n_up = len(Q)
        U = np.vstack([Q, lx]); UC = np.vstack([QC, lc])
        inst_u = postprocess.cluster_all(U, eps_m=args.conn_eps, min_points=20)
        # a cluster that reaches down into the driving band is a (tall) floor
        # object -> drop all of its overhead points from the candidate set
        keep = np.ones(n_up, bool)
        zmin_of = {}
        for lab in np.unique(inst_u):
            if lab < 0:
                continue
            m = inst_u == lab
            zmin_of[lab] = float(U[m, 2].min())
        for i in range(n_up):
            lab = inst_u[i]
            if lab >= 0 and zmin_of[lab] < floor_z + args.drive_top - 0.10:
                keep[i] = False
        print(f"[O] connectivity test vs lower cloud ({len(lx):,} pts): "
              f"{(~keep).sum():,} candidate pts belong to floor-connected clusters")
        Q, QC, Qd = Q[keep], QC[keep], Qd[keep]
    if oc.get("split_oversized"):
        inst, n_obj, n_split = postprocess.cluster_all_split(
            Q, eps_m=eps, min_points=args.min_points,
            split_footprint_m=oc["split_footprint_m"],
            split_eps_m=oc["split_eps_m"], split_min_points=args.min_points)
        print(f"[O] {n_obj} overhead clusters (giant-split {n_split})")
    else:
        inst = postprocess.cluster_all(Q, eps_m=eps, min_points=args.min_points)
    objs = objects.extract_objects(Q, QC, inst, min_points=args.min_points)

    n_cut = sum(1 for o in objs if float(o["bbox_min"][2]) < floor_z + args.drive_top + 0.05)
    objs = [o for o in objs if float(o["bbox_min"][2]) >= floor_z + args.drive_top + 0.05]
    print(f"[O] dropped {n_cut} cluster(s) whose bottom lies at the band cut "
          f"(truncated continuation of something below)")
    recs, rows = [], []
    for o in objs:
        g = object_geometry(o["xyz"])
        zmin, zmax = float(o["bbox_min"][2]), float(o["bbox_max"][2])
        near_c = float(np.min(np.abs(zc - zmax)))
        m = inst == o["id"]
        wall_frac = float((Qd[m] < args.wall_band + 0.10).mean())
        if near_c < 0.15:
            attach = "ceiling-hung"
        elif wall_frac > 0.25:
            attach = "wall-mounted"
        else:
            attach = "suspended"
        rec = {
            "type": "", "id": str(o["id"]), "name": f"overhead_{o['id']:03d}",
            "poseX": round(g["poseX"], 4), "poseY": round(g["poseY"], 4),
            "poseTheta": round(g["poseTheta"], 4),
            "dimensions": {"length": round(g["length"], 3),
                           "width": round(g["width"], 3),
                           "height": round(g["height"], 3)},
            "color": nearest_color(o["rgb"].mean(0)) if o["rgb"].any() else "gray",
            "properties": {
                "poseZ": round(g["poseZ"], 4), "coordinateFrame": "map",
                "source": "overhead-pass(DBSCAN)", "symbolicVerified": False,
                "imageFile": f"obj_{o['id']:04d}.ply",
                "heightLevel": "high",
                "clearanceBelow": round(zmin - floor_z, 3),
                "bottomAboveFloor": round(zmin - floor_z, 3),
                "topBelowCeiling": round(near_c, 3),
                "attachment": attach, "nPoints": int(o["n"]),
            }}
        recs.append(rec)
    manifest = objects.save_objects(objs, args.out)
    msg = build_message(recs)
    msg["overhead"] = {"floorZ": round(floor_z, 3),
                       "ceilingLevels": [round(z, 3) for z in zc.tolist()],
                       "driveTop": args.drive_top, "eps": eps,
                       "minPoints": args.min_points, "nWallPlanes": len(walls)}
    json.dump(msg, open(os.path.join(args.out, "semanticObjects.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"[O] {len(recs)} overhead objects -> {args.out}/ (manifest {manifest})")
    for r in recs:
        dd = r["dimensions"]
        print(f"    {r['name']}: {r['properties']['nPoints']:>6} pts  "
              f"{dd['length']:.2f}x{dd['width']:.2f}x{dd['height']:.2f} m  "
              f"bottom +{r['properties']['bottomAboveFloor']:.2f} m  {r['properties']['attachment']}")


if __name__ == "__main__":
    main()
