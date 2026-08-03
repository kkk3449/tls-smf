#!/usr/bin/env python3
"""Track-A pilot: diffuseness gate -> eps-down re-split -> evidence filter
-> (optional) two-pass VLM with consensus adoption.

Diffuse blobs (2D 10 cm fill ratio < 0.3 AND footprint area > 5 m^2) are
scatter/dust aggregates that DBSCAN glued together at the operating eps
(det2 examples: room_013 95K pts over 80 m^2, clutter_001 12.7K over 12 m^2).
Real objects in the same scene measure fill 0.5-0.85, so the gate separates
them cleanly. Flagged blobs are re-split at eps 0.12 (the machine_012
refinement recipe); only subs with real evidence (>=300 pts, fill >= 0.4,
area <= 5 m^2) survive to VLM; the rest is discarded as noise.

VLM stage mirrors the proven recovery consensus: pass A open-vocabulary,
pass B with lab-domain candidates; a label is adopted only when both passes
agree (candidate anchoring showed 50% precision without the gate).

  .venv/bin/python scripts/diffuse_resplit_pilot.py            # split only
  .venv/bin/python scripts/diffuse_resplit_pilot.py --vlm      # + classify
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.cluster import DBSCAN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DET = os.path.join(ROOT, "outputs", "vis_sota_det2")
OUT = os.path.join(ROOT, "outputs", "trackA_pilot")
FILL_MAX, AREA_MIN = 0.30, 5.0          # diffuseness gate
SUB_EPS, SUB_MIN = 0.12, 10             # re-split
EV_PTS, EV_FILL, EV_AREA = 300, 0.40, 5.0   # sub evidence gate
DOMAIN = ["mobile robot", "robot manipulator", "charging station",
          "conveyor belt", "machine", "chair", "cabinet", "cardboard box"]


def load(p):
    v = PlyData.read(p)["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], 1)


def fill_area(xy):
    lo, hi = xy.min(0), xy.max(0)
    area = max(1e-3, (hi - lo)[0] * (hi - lo)[1])
    vox = set(map(tuple, np.floor((xy - lo) / 0.10).astype(int)))
    cells = max(1, int(np.ceil((hi - lo)[0] / 0.10))
                * int(np.ceil((hi - lo)[1] / 0.10)))
    return len(vox) / cells, area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", action="store_true")
    ap.add_argument("--model", default="claude-sonnet-5")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    det = json.load(open(os.path.join(
        DET, "semanticObjects.lf_esc.json")))["semanticObjects"]
    files = sorted(glob.glob(os.path.join(DET, "obj_*.ply")))
    clouds = {f: load(f) for f in files}
    cents = {f: c[:, :2].mean(0) for f, c in clouds.items()}

    # record -> ply by nearest centroid (file index order differs)
    flagged, used = [], set()
    for rec in det:
        tgt = np.array([rec["poseX"], rec["poseY"]])
        f = min((f for f in files if f not in used),
                key=lambda f: np.linalg.norm(cents[f] - tgt))
        used.add(f)
        fill, area = fill_area(clouds[f][:, :2])
        if fill < FILL_MAX and area > AREA_MIN:
            flagged.append((rec, f, fill, area))
    print("flagged diffuse blobs:")
    for rec, f, fill, area in flagged:
        print(f"  {rec['name']:14s} {os.path.basename(f)} "
              f"pts={len(clouds[f])} area={area:.1f} fill={fill:.2f}")

    # Erosion split: these blobs are densely CONNECTED low-fill webs (median
    # NN spacing ~1.8 cm keeps them one DBSCAN component at any usable eps),
    # so density clustering cannot cut them. Instead: 10 cm occupancy grid ->
    # keep "core" cells (enough points AND enough occupied neighbours; 1-cell
    # strips die) -> 8-connected components of core cells become candidate
    # sub-objects; strip residue is discarded as structure scatter.
    from scipy import ndimage
    subs, discarded_pts = [], 0
    for rec, f, _, _ in flagged:
        pts = clouds[f]
        xy = pts[:, :2]
        lo = xy.min(0)
        ij = np.floor((xy - lo) / 0.10).astype(int)
        H = np.zeros(ij.max(0) + 1, dtype=int)
        np.add.at(H, (ij[:, 0], ij[:, 1]), 1)
        occ = H > 0
        nb = ndimage.convolve(occ.astype(int), np.ones((3, 3), int),
                              mode="constant") - occ.astype(int)
        core = (H >= 12) & (nb >= 4)
        comp, ncomp = ndimage.label(core, structure=np.ones((3, 3), int))
        cell_comp = comp[ij[:, 0], ij[:, 1]]
        discarded_pts += int((cell_comp == 0).sum())
        kept_here = 0
        for k in range(1, ncomp + 1):
            m = cell_comp == k
            sp = pts[m]
            fill, area = fill_area(sp[:, :2])
            if len(sp) < EV_PTS or fill < EV_FILL or area > EV_AREA:
                discarded_pts += int(m.sum())
                continue
            name = f"{rec['name']}_s{kept_here}"
            kept_here += 1
            fn = f"obj_{name}.ply"
            el = PlyElement.describe(
                np.array([tuple(p) for p in sp],
                         dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]),
                "vertex")
            PlyData([el]).write(os.path.join(OUT, fn))
            lo, hi = sp.min(0), sp.max(0)
            subs.append({
                "name": name, "type": "clutter", "id": 0,
                "poseX": float(sp[:, 0].mean()),
                "poseY": float(sp[:, 1].mean()),
                "dimensions": {"length": float(hi[0] - lo[0]),
                               "width": float(hi[1] - lo[1]),
                               "height": float(hi[2] - lo[2])},
                "properties": {"imageFile": fn,
                               "poseZ": float(sp[:, 2].mean()),
                               "parent": rec["name"],
                               "nPoints": int(len(sp)),
                               "fill": round(fill, 2)}})
    json.dump({"semanticObjects": subs},
              open(os.path.join(OUT, "semanticObjects.json"), "w"), indent=1)
    total_flagged = sum(len(clouds[f]) for _, f, _, _ in flagged)
    kept_pts = total_flagged - discarded_pts
    print(f"\nsubs surviving evidence gate: {len(subs)} "
          f"({kept_pts}/{total_flagged} pts kept, "
          f"{discarded_pts} discarded as scatter)")
    for s in subs:
        print(f"  {s['name']:22s} pts={s['properties']['nPoints']:6d} "
              f"fill={s['properties']['fill']:.2f} "
              f"h={s['dimensions']['height']:.2f}")

    if not subs:
        return
    subprocess.run([sys.executable,
                    os.path.join(ROOT, "scripts", "render_object_views.py"),
                    "--semantic", os.path.join(OUT, "semanticObjects.json"),
                    "--objects-dir", OUT, "--out", OUT], check=True)

    if not args.vlm:
        print("(render done; rerun with --vlm to classify)")
        return

    from blk360seg import vlm_stage_b as vb
    from clutter_reverify import _SYS_RECOVERY
    bottoms = sorted(o["properties"]["poseZ"] - o["dimensions"]["height"] / 2
                     for o in det)
    floor_z = bottoms[len(bottoms) // 4]
    vlm = vb.SemanticVLM(model=args.model)
    results = []
    for s in subs:
        views = [os.path.join(OUT, "views", s["name"], f"view_{a:03d}.png")
                 for a in (0, 90, 180, 270)]
        out = {}
        for tag, cands in (("open", []), ("domain", DOMAIN)):
            content = vlm.build_symbolic_request(s, views,
                                                 allowed_types=cands,
                                                 floor_z=floor_z)
            out[tag] = vlm._call(_SYS_RECOVERY, content, vb.SYMBOLIC_TOOL)
        a, b = out["open"]["corrected_type"], out["domain"]["corrected_type"]
        agree = (a == b) or (a in b) or (b in a)
        adopted = a if (agree and a != "clutter") else None
        results.append({"name": s["name"], "parent": s["properties"]["parent"],
                        "open": out["open"], "domain": out["domain"],
                        "adopted": adopted})
        print(f"  {s['name']:22s} open={a:18s} domain={b:18s} "
              f"-> {'ADOPT ' + adopted if adopted else 'keep clutter'}")
    usage = vlm.usage_summary()
    json.dump({"model": args.model, "gate": {"fill": FILL_MAX,
                                             "area": AREA_MIN},
               "usage": usage, "subs": subs, "results": results},
              open(os.path.join(OUT, "pilot_report.json"), "w"), indent=1)
    print(f"wrote {OUT}/pilot_report.json; cost ${usage['cost_usd']}")


if __name__ == "__main__":
    main()
