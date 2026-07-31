#!/usr/bin/env python3
"""Structure-condition ensemble: detect under three structure-removal
conditions and fuse (owner idea, 2026-07-31: imperfect wall/ceiling
removal both amputates real objects and leaves phantom sheets, so vary
the removal and judge across runs).

  A raw        clean.ply untouched (walls + ceiling in)
  B no-ceil    clean.ply minus the ceiling band (walls in)
  C removed    the production det4 view (ceiling + walls removed)

A and B are geometry-only DBSCAN passes. Components larger than
--blob-area are structure blobs: they get the resplit-v2 treatment
(map wall-line strip + erosion split + support-plane compound split), so
wall-flush content that removal would amputate still becomes candidates.

Fusion:
  - every det4 object gets a support count: how many conditions re-find
    it (centroid < 0.5 m, extents within 30% / 0.15 m — the KG matcher's
    tolerance). Stable objects are cross-condition confirmed.
  - A/B candidates with no det4 counterpart are ensemble DISCOVERIES;
    with --vlm the top --max-vlm get the two-pass consensus (open vs
    domain vocabulary, height level as hard constraint).

  .venv/bin/python scripts/structure_ensemble.py          # geometry + report
  .venv/bin/python scripts/structure_ensemble.py --vlm    # + classify finds
"""
import argparse
import json
import os
import sys

import numpy as np
from plyfile import PlyData, PlyElement
from scipy import ndimage
from sklearn.cluster import DBSCAN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.structure_noise_filter import load_gridmap          # noqa: E402
from scripts.diffuse_resplit_v2 import (fill_area, height_level,  # noqa: E402
                                        split_compound)

CLEAN = os.path.join(ROOT, "outputs", "vis_sota_det2", "clean.ply")
DET4 = os.path.join(ROOT, "outputs", "vis_sota_det4",
                    "semanticObjects.lf_esc.json")
OUT = os.path.join(ROOT, "outputs", "ensemble3")
MAP = "/home/caselab/ammr_twin/map_vis_n2_1.yaml"
TRANSFORM = os.path.join(ROOT, "outputs", "vissota_to_visn2_T.npy")
EV_PTS, EV_AREA, EV_H = 250, 5.0, 0.15
DOMAIN = ["mobile robot", "robot manipulator", "desk", "monitor",
          "charging station", "conveyor belt", "machine", "chair",
          "cabinet", "cardboard box", "poster stand", "column", "plant"]


def load(p):
    v = PlyData.read(p)["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], 1)


def wall_tools():
    img, res, ox, oy = load_gridmap(MAP)
    occ = img < 50
    room = ndimage.binary_erosion(
        ndimage.binary_fill_holes(ndimage.binary_dilation(
            occ, iterations=3)), iterations=3)
    boundary = ndimage.binary_dilation(
        room ^ ndimage.binary_erosion(room, iterations=4), iterations=2)
    d_wall = ndimage.distance_transform_edt(~(occ & boundary)) * res

    def sample(field, p):
        ci = np.clip(((p[:, 0] - ox) / res).astype(int), 0,
                     img.shape[1] - 1)
        cj = np.clip(((p[:, 1] - oy) / res).astype(int), 0,
                     img.shape[0] - 1)
        return field[cj, ci]
    return room, d_wall, sample


def erosion_components(kp, min_pts=250, core_pts=8, core_nb=3):
    xy = kp[:, :2]
    lo = xy.min(0)
    ij = np.floor((xy - lo) / 0.10).astype(int)
    H = np.zeros(ij.max(0) + 1, dtype=int)
    np.add.at(H, (ij[:, 0], ij[:, 1]), 1)
    o2 = H > 0
    nb = ndimage.convolve(o2.astype(int), np.ones((3, 3), int),
                          mode="constant") - o2.astype(int)
    core = (H >= core_pts) & (nb >= core_nb)
    comp, n = ndimage.label(core, structure=np.ones((3, 3), int))
    cc = comp[ij[:, 0], ij[:, 1]]
    return [np.where(cc == i)[0] for i in range(1, n + 1)
            if (cc == i).sum() >= min_pts]


def condition_candidates(pts_s, pts_n2, room, d_wall, sample, blob_area,
                         tag):
    """DBSCAN a condition's cloud; structure blobs get wall strip + split.
    Returns candidate dicts in the sota frame."""
    lab = DBSCAN(eps=0.30, min_samples=60).fit_predict(pts_s)
    cands = []
    for li in sorted(set(lab) - {-1}):
        idx = np.where(lab == li)[0]
        if len(idx) < EV_PTS:
            continue
        sp = pts_s[idx]
        fill, area = fill_area(sp[:, :2])
        if area <= blob_area:
            ext = sp.max(0) - sp.min(0)
            if ext[2] >= EV_H:
                cands.append((f"{tag}_d{li}", idx))
            continue
        # structure blob: strip wall band / outside-room, then split
        pn = pts_n2[idx]
        ins = sample(room, pn)
        dw = sample(d_wall, pn)
        keep = np.where(ins & (dw > 0.10))[0]
        if len(keep) < EV_PTS:
            continue
        kp = pn[keep]
        floor_z = np.percentile(kp[:, 2], 1)
        for ci, cm in enumerate(erosion_components(kp)):
            sub = kp[cm]
            _, carea = fill_area(sub[:, :2])
            if carea > 4.0:
                for sfx, mm in split_compound(sub, floor_z,
                                              f"{tag}_b{li}c{ci}"):
                    cands.append((sfx, idx[keep][cm][mm]))
            else:
                cands.append((f"{tag}_b{li}c{ci}", idx[keep][cm]))
    out = []
    for name, gidx in cands:
        sp = pts_s[gidx]
        fill, area = fill_area(sp[:, :2])
        ext = sp.max(0) - sp.min(0)
        amax = 12.0 if name.endswith("_desk") else EV_AREA
        if len(sp) < EV_PTS or area > amax or ext[2] < EV_H:
            continue
        pn = pts_n2[gidx]
        fz = np.percentile(pts_n2[:, 2], 1)
        out.append({"name": name, "n": int(len(sp)),
                    "cx": float(sp[:, 0].mean()),
                    "cy": float(sp[:, 1].mean()),
                    "cz": float(sp[:, 2].mean()),
                    "dims": [float(e) for e in ext],
                    "fill": round(fill, 2),
                    "heightLevel": height_level(
                        float(pn[:, 2].min() - fz),
                        float(pn[:, 2].max() - fz)),
                    "idx": gidx})
    return out


def match(c, o):
    """kg._matches-style tolerance between a candidate and a det4 record."""
    oz = o["properties"].get("poseZ", 0.0)
    d = np.linalg.norm([c["cx"] - o["poseX"], c["cy"] - o["poseY"],
                        c["cz"] - oz])
    if d > 0.5:
        return False
    od = [o["dimensions"]["length"], o["dimensions"]["width"],
          o["dimensions"]["height"]]
    for a, b in zip(c["dims"], od):
        if abs(a - b) > max(0.3 * max(a, b), 0.15):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", action="store_true")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--blob-area", type=float, default=5.0)
    ap.add_argument("--max-vlm", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    pts = load(CLEAN)
    T = np.load(TRANSFORM)
    pn_all = (T[:3, :3] @ pts.T).T + T[:3, 3]
    room, d_wall, sample = wall_tools()
    zc = np.percentile(pn_all[:, 2], 99)      # ceiling height (n2)
    det4 = json.load(open(DET4))["semanticObjects"]

    conds = {}
    for tag, mask in (("A", np.ones(len(pts), bool)),
                      ("B", pn_all[:, 2] < zc - 0.35)):
        c = condition_candidates(pts[mask], pn_all[mask], room, d_wall,
                                 sample, args.blob_area, tag)
        conds[tag] = c
        print(f"condition {tag}: {len(c)} candidates")

    support = []
    for o in det4:
        s = {"name": o["name"], "type": o["type"],
             "status": o["properties"].get("verificationStatus",
                                           "unverified"),
             "found_in": [t for t, cl in conds.items()
                          if any(match(c, o) for c in cl)]}
        support.append(s)
    stable = [s for s in support if len(s["found_in"]) == 2]
    print(f"det4 support: {len(stable)}/{len(det4)} found in both A and B")

    matched = set()
    for t, cl in conds.items():
        for c in cl:
            if any(match(c, o) for o in det4):
                matched.add(c["name"])
    discoveries = []
    for t, cl in conds.items():
        for c in cl:
            if c["name"] in matched:
                continue
            # cross-check against already collected discoveries (dedup A/B)
            dup = None
            for d in discoveries:
                if (np.hypot(c["cx"] - d["cx"], c["cy"] - d["cy"]) < 0.5
                        and all(abs(a - b) <= max(0.3 * max(a, b), 0.15)
                                for a, b in zip(c["dims"], d["dims"]))):
                    dup = d
                    break
            if dup:
                dup["conditions"].append(t)
            else:
                c2 = dict(c)
                c2["conditions"] = [t]
                discoveries.append(c2)
    # a discovery near an existing object's centroid is an extent VARIANT
    # (the condition merged/cut it differently), not a new find
    for d in discoveries:
        near = [o["name"] for o in det4
                if np.hypot(d["cx"] - o["poseX"], d["cy"] - o["poseY"])
                < 0.7]
        d["variant_of"] = near[0] if near else None
    new = [d for d in discoveries if not d["variant_of"]]
    new.sort(key=lambda d: (-len(d["conditions"]), -d["n"]))
    discoveries = new + [d for d in discoveries if d["variant_of"]]
    print(f"{len(discoveries)} ensemble discoveries: {len(new)} new, "
          f"{len(discoveries) - len(new)} extent-variants of known objects")
    for d in new[:20]:
        print(f"  {d['name']:16s} conds={d['conditions']} pts={d['n']:6d} "
              f"({d['cx']:6.2f},{d['cy']:6.2f}) "
              f"L{d['dims'][0]:.2f} W{d['dims'][1]:.2f} H{d['dims'][2]:.2f} "
              f"lvl={d['heightLevel']}")

    report = {"det4_support": support,
              "discoveries": [{k: v for k, v in d.items() if k != "idx"}
                              for d in discoveries]}

    if args.vlm and new:
        recs = []
        for d in new[:args.max_vlm]:
            sp = pts[d["idx"]]
            fn = f"obj_{d['name']}.ply"
            el = PlyElement.describe(
                np.array([tuple(p) for p in sp],
                         dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]),
                "vertex")
            PlyData([el]).write(os.path.join(OUT, fn))
            recs.append({"name": d["name"], "type": "clutter", "id": 0,
                         "poseX": d["cx"], "poseY": d["cy"],
                         "dimensions": {"length": d["dims"][0],
                                        "width": d["dims"][1],
                                        "height": d["dims"][2]},
                         "properties": {"imageFile": fn, "poseZ": d["cz"],
                                        "nPoints": d["n"],
                                        "heightLevel": d["heightLevel"]}})
        json.dump({"semanticObjects": recs},
                  open(os.path.join(OUT, "semanticObjects.json"), "w"),
                  indent=1)
        import subprocess
        subprocess.run([sys.executable,
                        os.path.join(ROOT, "scripts",
                                     "render_object_views.py"),
                        "--semantic", os.path.join(OUT,
                                                   "semanticObjects.json"),
                        "--objects-dir", OUT, "--out", OUT,
                        "--size", "1024", "--point-size", "14"],
                       check=True)
        from blk360seg import vlm_stage_b as vb
        from scripts.clutter_reverify import _SYS_RECOVERY
        vlm = vb.SemanticVLM(model=args.model)
        for r in recs:
            views = [os.path.join(OUT, "views", r["name"],
                                  f"view_{a:03d}.png")
                     for a in (0, 90, 180, 270)]
            views = [v for v in views if os.path.exists(v)]
            out = {}
            for tg, cands in (("open", []), ("domain", DOMAIN)):
                content = vlm.build_symbolic_request(
                    r, views, allowed_types=cands,
                    height_level=r["properties"]["heightLevel"])
                out[tg] = vlm._call(_SYS_RECOVERY, content,
                                    vb.SYMBOLIC_TOOL)
            a, b = out["open"]["corrected_type"], \
                out["domain"]["corrected_type"]
            agree = (a == b) or (a in b) or (b in a)
            r["vlm"] = {"open": out["open"], "domain": out["domain"],
                        "adopted": a if (agree and a != "clutter")
                        else None}
            print(f"  VLM {r['name']:16s} open={a:18s} domain={b:18s} "
                  f"-> {r['vlm']['adopted'] or 'clutter'}")
        report["vlm"] = {"usage": vlm.usage_summary(), "records": recs}
        print(f"VLM cost ${report['vlm']['usage']['cost_usd']}")

    json.dump(report, open(os.path.join(OUT, "ensemble_report.json"), "w"),
              indent=1, default=str)
    print(f"wrote {OUT}/ensemble_report.json")


if __name__ == "__main__":
    main()
