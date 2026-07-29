#!/usr/bin/env python3
"""Hybrid structure-noise filter: vertical-component detection x lidar
boundary-map registration.

RANSAC removes the dominant wall/ceiling planes, but residual sheets (a
7.4 m wall band, ceiling patches, fluorescent fixtures, outside-door
bleed) survive and get verified as phantom objects. This filter scores
every detected cluster against the exploration-scan occupancy grid — the
lidar-drawn room boundary is an independent, drift-free wall prior — and
against its own PCA shape:

  wall remnant     points hug the grid wall line (median dist < eps) AND
                   the cluster is a thin sheet (minor PCA extent small).
                   Wall-flush real objects (a monitor bank protruding
                   0.1-0.5 m) fail the hug test and survive.
  ceiling fixture  cluster bottom high above the floor (> ceil_frac of
                   room height) — catches lights and ceiling noise that
                   the soffit band filter (tuned for deep soffits) missed.
  outside noise    majority of points fall outside the room's free-space
                   boundary in the grid map (door bleed-through).

  .venv/bin/python scripts/structure_noise_filter.py \
      --objects outputs/vis_sota_det/semanticObjects.lf_esc.visn2frame.room.json \
      --objects-dir outputs/vis_sota_det \
      --transform outputs/vissota_to_visn2_T.npy \
      --map /home/caselab/ammr_twin/map_vis_n2_1.yaml \
      --gt outputs/t3_owner_gt.json \
      --out outputs/t3_structure_noise_filter.json
"""
import argparse
import json
import os
import re

import numpy as np


def load_gridmap(yaml_path):
    meta = dict(re.findall(r"(\w+):\s*(.+)", open(yaml_path).read()))
    res = float(meta["resolution"])
    ox, oy = [float(v) for v in meta["origin"].strip("[]").split(",")[:2]]
    pgm = os.path.join(os.path.dirname(yaml_path), meta["image"])
    with open(pgm, "rb") as f:
        assert f.readline().strip() == b"P5"
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        f.readline()
        img = np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w)
    img = np.flipud(img)                       # row 0 = ymin
    return img, res, ox, oy


def main():
    from scipy.ndimage import distance_transform_edt, binary_fill_holes

    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", required=True)
    ap.add_argument("--objects-dir", required=True)
    ap.add_argument("--transform", required=True,
                    help="4x4 .npy mapping object/scan frame -> map frame")
    ap.add_argument("--map", required=True)
    ap.add_argument("--gt", default=None,
                    help="owner GT json: evaluate flag decisions against it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wall-eps", type=float, default=0.15,
                    help="wall-line hug distance (m)")
    ap.add_argument("--wall-hug-frac", type=float, default=0.6)
    ap.add_argument("--sheet-thin", type=float, default=0.22,
                    help="max minor-axis extent (m) for a wall sheet")
    ap.add_argument("--ceil-bottom", type=float, default=2.0,
                    help="cluster bottom above floor (m) => ceiling fixture")
    ap.add_argument("--outside-frac", type=float, default=0.5)
    ap.add_argument("--full-height", type=float, default=2.5,
                    help="z-span (m) above which a wall-hugging band is "
                         "treated as wall regardless of global bbox width")
    args = ap.parse_args()

    import open3d as o3d
    T = np.load(args.transform)
    img, res, ox, oy = load_gridmap(args.map)
    occ = img < 100                                   # occupied = wall
    free = img > 200
    # room polygon INCLUDING the wall band: wall-flush objects (a monitor
    # bank against the wall) must not count as outside; only points beyond
    # the outer boundary (door bleed-through) are outside
    room = binary_fill_holes(free | occ)
    d_wall = distance_transform_edt(~occ) * res       # to nearest wall cell
    d_outside = distance_transform_edt(room) * res    # 0 outside, >0 inside

    def grid_at(field, xy):
        ci = np.clip(((xy[:, 0] - ox) / res).astype(int), 0, occ.shape[1] - 1)
        cj = np.clip(((xy[:, 1] - oy) / res).astype(int), 0, occ.shape[0] - 1)
        return field[cj, ci]

    d = json.load(open(args.objects))
    objs = d["semanticObjects"] if isinstance(d, dict) else d
    floor_z = sorted(o["properties"]["poseZ"] - o["dimensions"]["height"] / 2
                     for o in objs)[len(objs) // 4]

    report = []
    for o in objs:
        f = os.path.join(args.objects_dir, o["properties"]["imageFile"])
        pc = o3d.io.read_point_cloud(f)
        xyz = np.asarray(pc.points)
        xyz = (T[:3, :3] @ xyz.T).T + T[:3, 3]        # -> map frame
        xy = xyz[:, :2]

        dw = grid_at(d_wall, xy)
        hug = float((dw < args.wall_eps).mean())
        med_dw = float(np.median(dw))
        out_frac = float((grid_at(d_outside, xy) <= 0).mean())

        c = xyz - xyz.mean(0)
        ew, evec = np.linalg.eigh(np.cov(c.T))        # ascending
        minor_ext = float(4.0 * np.sqrt(max(ew[0], 0.0)))   # ~2 sigma each way
        # a wall sheet stands vertically: its thin axis is horizontal
        minor_horizontal = abs(evec[2, 0]) < 0.4
        bottom = float(xyz[:, 2].min() - floor_z)
        zspan = float(xyz[:, 2].max() - xyz[:, 2].min())

        # local sheet thickness: a long wavy wall band is thick under global
        # PCA but thin in every 1 m segment; a monitor bank / desk row has
        # real depth in most segments
        major_dir = evec[:2, 2] / (np.linalg.norm(evec[:2, 2]) or 1.0)
        t = xy @ major_dir
        local = []
        for b0 in np.arange(t.min(), t.max(), 1.0):
            m = (t >= b0) & (t < b0 + 1.0)
            if m.sum() >= 30:
                ewl = np.linalg.eigvalsh(np.cov((xyz[m] - xyz[m].mean(0)).T))
                local.append(4.0 * np.sqrt(max(ewl[0], 0.0)))
        local_thin = float(np.median(local)) if local else minor_ext

        why = None
        if out_frac > args.outside_frac:
            why = f"outside room ({out_frac:.0%} of points)"
        elif bottom > args.ceil_bottom:
            why = f"ceiling fixture/noise (bottom {bottom:.2f} m above floor)"
        elif (hug > args.wall_hug_frac and minor_ext < args.sheet_thin
              and minor_horizontal):
            why = (f"wall remnant (hug {hug:.0%} within {args.wall_eps} m, "
                   f"vertical sheet {minor_ext:.2f} m thin)")
        elif (hug > args.wall_hug_frac and zspan > args.full_height
              and local_thin < 0.3
              and float(t.max() - t.min()) > 2.0):
            # LONG floor-to-ceiling band hugging the wall line: a wall,
            # whatever its global bbox says. A real wall-flush object like a
            # monitor bank stays under ~2 m tall with depth in its segments;
            # a wall-mounted panel/conduit column is full-height but narrow
            # (< 2 m along the wall), so the length condition spares it.
            why = (f"full-height wall band (z-span {zspan:.1f} m, "
                   f"{t.max() - t.min():.1f} m long, local thickness "
                   f"{local_thin:.2f} m)")
        report.append({"name": o["name"], "type": o["type"],
                       "flag": bool(why), "why": why,
                       "wall_hug_frac": round(hug, 3),
                       "median_wall_dist_m": round(med_dw, 3),
                       "outside_frac": round(out_frac, 3),
                       "minor_extent_m": round(minor_ext, 3),
                       "local_thickness_m": round(local_thin, 3),
                       "z_span_m": round(zspan, 2),
                       "bottom_above_floor_m": round(bottom, 2)})
        if why:
            print(f"  FLAG {o['name']:24s} {why}")

    out = {"params": {k: getattr(args, k) for k in
                      ("wall_eps", "wall_hug_frac", "sheet_thin",
                       "ceil_bottom", "outside_frac")},
           "floor_z": floor_z, "flagged": sum(r["flag"] for r in report),
           "clusters": report}

    if args.gt:
        gt = {g["det"]: g for g in json.load(open(args.gt))["objects"]}
        noise = {n for n, g in gt.items()
                 if g["category"] == "structure_noise"
                 or (g["owner_gt"] or "").startswith("형광등")}
        tp = [r["name"] for r in report if r["flag"] and r["name"] in noise]
        fp = [r["name"] for r in report if r["flag"] and r["name"] not in noise]
        fn = [r["name"] for r in report if not r["flag"] and r["name"] in noise]
        out["eval"] = {"gt_noise": sorted(noise), "tp": tp, "fp": fp, "fn": fn,
                       "recall": round(len(tp) / max(len(noise), 1), 3),
                       "precision": round(len(tp) / max(len(tp) + len(fp), 1), 3)}
        print(f"\nvs owner GT: recall {len(tp)}/{len(noise)}, "
              f"false positives {fp or 'none'}, missed {fn or 'none'}")

    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {args.out} ({out['flagged']}/{len(report)} flagged)")


if __name__ == "__main__":
    main()
