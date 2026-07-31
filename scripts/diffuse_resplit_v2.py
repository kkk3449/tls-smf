#!/usr/bin/env python3
"""Re-split v2: recover real objects from the residual the trackA pilot
discarded (the >5 m^2 erosion components — robot arm on a desk, part of a
mobile robot, wall noise — that the evidence gate rejected wholesale).

Owner ground truth showed the pilot's discard was hiding content, not
cleaning it: only ~7% of the discarded points were true mixed-pixel
scatter; the rest were wall-hugging sheets plus a desk/arm/mobile-robot
compound. v2 therefore separates by CAUSE instead of gating by size:

  1. residual  = flagged diffuse blob minus the pilot's adopted subs
  2. wall strip (map frame): the exploration grid's outer wall line is a
     drift-free prior. Points inside the ROOM polygon (fill of the wall
     outline — NOT the free-space polygon, which would drop wall-flush
     furniture) but within --wall-band of the wall line are structure
     noise. Points outside the room polygon are bleed-through.
  3. erosion components (10 cm grid, core >=8 pts & >=3 neighbours) of
     what survives; thin sheets that still hug the wall (median wall dist
     < 0.18 m AND PCA minor extent < 0.15 m) are late wall remnants.
  4. compound split for components larger than --compound-area: extract
     thin vertical columns (15 cm cells spanning >1.3 m -> poles/stands),
     find the support plane (z-histogram peak 0.5-0.95 m above floor),
     cut into above / slab / below layers, DBSCAN each layer in XY, and
     re-attach under-slab clusters whose footprint lies under the slab
     (>=50% 10 cm-cell overlap with the dilated slab) to the desk.
  5. evidence gate (>=250 pts, footprint <= 5 m^2, height >= 0.15 m)
  6. height level (low/mid/high vs the 2D-lidar scan plane / ceiling)
     is computed per sub and passed to the VLM as a hard constraint.
  7. two-pass VLM consensus (open vocabulary vs domain candidates) at
     1024 px; adopt only on agreement, as in the recovery pipeline.

Geometry runs in the vis_n2 map frame (where the wall prior lives);
outputs (plys, poses) stay in the det2/sota frame for pipeline
consistency.

  .venv/bin/python scripts/diffuse_resplit_v2.py            # split only
  .venv/bin/python scripts/diffuse_resplit_v2.py --vlm      # + classify
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
from plyfile import PlyData, PlyElement
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.structure_noise_filter import load_gridmap  # noqa: E402

DET = os.path.join(ROOT, "outputs", "vis_sota_det2")
PILOT = os.path.join(ROOT, "outputs", "trackA_pilot")
OUT = os.path.join(ROOT, "outputs", "trackA_v2")
MAP = "/home/caselab/ammr_twin/map_vis_n2_1.yaml"
TRANSFORM = os.path.join(ROOT, "outputs", "vissota_to_visn2_T.npy")
FILL_MAX, AREA_MIN = 0.30, 5.0            # diffuseness gate (as pilot)
EV_PTS, EV_AREA, EV_H = 250, 5.0, 0.15    # evidence gate v2
LIDAR_Z, CEIL_REL = 0.35, 1.8             # height-level thresholds (rel floor)
DOMAIN = ["mobile robot", "robot manipulator", "desk", "monitor",
          "charging station", "conveyor belt", "machine", "chair",
          "cabinet", "cardboard box", "poster stand", "column"]


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


def cellset(xy, dilate=0):
    s = set(map(tuple, np.floor(xy / 0.10).astype(int)))
    for _ in range(dilate):
        s |= {(i + di, j + dj) for i, j in s
              for di in (-1, 0, 1) for dj in (-1, 0, 1)}
    return s


def height_level(zlo_rel, zhi_rel):
    if zlo_rel > CEIL_REL:
        return "high"
    if zlo_rel < 0.25 and zhi_rel > 0.5 * LIDAR_Z:
        return "low"
    return "mid"


def split_compound(pts, floor_z, tag):
    """pts (n2 frame) -> list of (name_suffix, mask arrays into pts)."""
    out = []
    # thin vertical columns first (they span every z layer)
    ij = np.floor((pts[:, :2] - pts[:, :2].min(0)) / 0.15).astype(int)
    keys = ij[:, 0] * 100000 + ij[:, 1]
    polemask = np.zeros(len(pts), bool)
    for k in np.unique(keys):
        m = keys == k
        if pts[m, 2].max() - pts[m, 2].min() > 1.3:
            polemask |= m
    if polemask.sum() > 100:
        lab = DBSCAN(eps=0.25, min_samples=20).fit_predict(
            pts[polemask, :2])
        idx = np.where(polemask)[0]
        for li in set(lab) - {-1}:
            out.append((f"{tag}_pole{li}", idx[lab == li]))
    rest_idx = np.where(~polemask)[0]
    rp = pts[rest_idx]
    # support plane: strongest z peak 0.5-0.95 m above floor
    h, e = np.histogram(rp[:, 2], bins=60)
    zc = (e[:-1] + e[1:]) / 2
    band = (zc > floor_z + 0.5) & (zc < floor_z + 0.95)
    if band.any() and h[band].max() > 0.03 * len(rp):
        zd = zc[band][np.argmax(h[band])]
        above = rp[:, 2] > zd + 0.08
        slab = (rp[:, 2] >= zd - 0.12) & (rp[:, 2] <= zd + 0.08)
        below = rp[:, 2] < zd - 0.12
        # clean the slab to its largest connected cell component — the z
        # band also catches desk-height spill over neighbouring objects
        # (e.g. the mobile robot's top edge), and those stray cells must
        # not vote in the under-desk merge test
        sxy = rp[slab, :2]
        slo = sxy.min(0)
        sij = np.floor((sxy - slo) / 0.10).astype(int)
        sg = np.zeros(sij.max(0) + 1, bool)
        sg[sij[:, 0], sij[:, 1]] = True
        scomp, sn = ndimage.label(sg, structure=np.ones((3, 3), int))
        if sn > 1:
            sizes = ndimage.sum(sg, scomp, range(1, sn + 1))
            main = 1 + int(np.argmax(sizes))
            keep_slab = scomp[sij[:, 0], sij[:, 1]] == main
        else:
            keep_slab = np.ones(len(sxy), bool)
        slab_idx = rest_idx[slab][keep_slab]
        slabcells = cellset(sxy[keep_slab])
        desk_idx = [slab_idx]
        for lname, lmask, eps, mn in (("above", above, 0.18, 25),
                                      ("below", below, 0.18, 30)):
            if not lmask.any():
                continue
            lab = DBSCAN(eps=eps, min_samples=mn).fit_predict(
                rp[lmask, :2])
            lidx = rest_idx[lmask]
            for li in set(lab) - {-1}:
                m = lab == li
                if m.sum() < 150:
                    continue
                if lname == "below":
                    cs = cellset(rp[lmask][m, :2])
                    ov = len(cs & slabcells) / max(1, len(cs))
                    top = rp[lmask][m, 2].max()
                    if ov >= 0.6 and top >= zd - 0.30:
                        desk_idx.append(lidx[m])   # under-desk storage
                        continue
                out.append((f"{tag}_{lname}{li}", lidx[m]))
        out.append((f"{tag}_desk", np.concatenate(desk_idx)))
    else:
        lab = DBSCAN(eps=0.18, min_samples=30).fit_predict(rp[:, :2])
        for li in set(lab) - {-1}:
            m = lab == li
            if m.sum() >= 150:
                out.append((f"{tag}_c{li}", rest_idx[m]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", action="store_true")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--wall-band", type=float, default=0.10)
    ap.add_argument("--compound-area", type=float, default=4.0)
    ap.add_argument("--render-size", type=int, default=1024)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    det = json.load(open(os.path.join(
        DET, "semanticObjects.lf_esc.json")))["semanticObjects"]
    files = sorted(glob.glob(os.path.join(DET, "obj_*.ply")))
    cents = {f: load(f)[:, :2].mean(0) for f in files}
    used, m = set(), {}
    for rec in det:
        tgt = np.array([rec["poseX"], rec["poseY"]])
        f = min((f for f in files if f not in used),
                key=lambda f: np.linalg.norm(cents[f] - tgt))
        used.add(f)
        m[rec["name"]] = f

    T = np.load(TRANSFORM)
    img, res, ox, oy = load_gridmap(MAP)
    occ = img < 50
    room = ndimage.binary_erosion(
        ndimage.binary_fill_holes(ndimage.binary_dilation(
            occ, iterations=3)), iterations=3)
    boundary = ndimage.binary_dilation(
        room ^ ndimage.binary_erosion(room, iterations=4), iterations=2)
    d_wall = ndimage.distance_transform_edt(~(occ & boundary)) * res

    def sample(field, p):
        ci = np.clip(((p[:, 0] - ox) / res).astype(int), 0, img.shape[1] - 1)
        cj = np.clip(((p[:, 1] - oy) / res).astype(int), 0, img.shape[0] - 1)
        return field[cj, ci]

    subs, report = [], []
    for rec in det:
        pts_s = load(m[rec["name"]])            # sota frame (output frame)
        fill, area = fill_area(pts_s[:, :2])
        if not (fill < FILL_MAX and area > AREA_MIN):
            continue
        pil = [load(sp) for sp in glob.glob(
            os.path.join(PILOT, f"obj_{rec['name']}_s*.ply"))]
        if pil:
            d, _ = cKDTree(np.vstack(pil)).query(pts_s, k=1, workers=8)
            pts_s = pts_s[d > 1e-4]
        pts = (T[:3, :3] @ pts_s.T).T + T[:3, 3]   # n2 frame (geometry)
        ins = sample(room, pts)
        dw = sample(d_wall, pts)
        keep = ins & (dw > args.wall_band)
        report.append({"parent": rec["name"], "residual": int(len(pts)),
                       "wall_band": int(((dw <= args.wall_band) & ins).sum()),
                       "outside_room": int((~ins).sum())})
        k = np.where(keep)[0]
        if len(k) < EV_PTS:
            continue
        kp = pts[k]
        floor_z = np.percentile(kp[:, 2], 1)
        xy = kp[:, :2]
        lo = xy.min(0)
        ij = np.floor((xy - lo) / 0.10).astype(int)
        H = np.zeros(ij.max(0) + 1, dtype=int)
        np.add.at(H, (ij[:, 0], ij[:, 1]), 1)
        o2 = H > 0
        nb = ndimage.convolve(o2.astype(int), np.ones((3, 3), int),
                              mode="constant") - o2.astype(int)
        core = (H >= 8) & (nb >= 3)
        comp, n = ndimage.label(core, structure=np.ones((3, 3), int))
        cc = comp[ij[:, 0], ij[:, 1]]
        pieces = []                     # (suffix, indices into kp)
        for i in range(1, n + 1):
            cm = np.where(cc == i)[0]
            if len(cm) < EV_PTS:
                continue
            sp = kp[cm]
            md = float(np.median(dw[k][cm]))
            c2 = sp[:, :2] - sp[:, :2].mean(0)
            ev = np.linalg.eigvalsh(np.cov(c2.T))
            minor = 2.5 * float(np.sqrt(max(ev[0], 1e-9)))
            if md < 0.18 and minor < 0.15:
                continue                # late wall sheet
            _, carea = fill_area(sp[:, :2])
            if carea > args.compound_area:
                pieces += [(sfx, cm[sub])
                           for sfx, sub in split_compound(sp, floor_z,
                                                          f"c{i}")]
            else:
                pieces.append((f"c{i}", cm))
        for sfx, cm in pieces:
            sp = kp[cm]
            fill, carea = fill_area(sp[:, :2])
            ext = sp.max(0) - sp.min(0)
            # a support-plane desk is structurally validated by its slab;
            # an L-shaped desk bank legitimately exceeds the web-residue
            # bbox cap, so it gets a looser one
            amax = 12.0 if sfx.endswith("_desk") else EV_AREA
            if len(sp) < EV_PTS or carea > amax or ext[2] < EV_H:
                continue
            zlo_rel = float(sp[:, 2].min() - floor_z)
            zhi_rel = float(sp[:, 2].max() - floor_z)
            lvl = height_level(zlo_rel, zhi_rel)
            name = f"{rec['name']}_v2_{sfx}"
            sps = pts_s[k][cm]           # same rows, sota frame
            fn = f"obj_{name}.ply"
            el = PlyElement.describe(
                np.array([tuple(p) for p in sps],
                         dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]),
                "vertex")
            PlyData([el]).write(os.path.join(OUT, fn))
            slo, shi = sps.min(0), sps.max(0)
            subs.append({
                "name": name, "type": "clutter", "id": 0,
                "poseX": float(sps[:, 0].mean()),
                "poseY": float(sps[:, 1].mean()),
                "dimensions": {"length": float(shi[0] - slo[0]),
                               "width": float(shi[1] - slo[1]),
                               "height": float(shi[2] - slo[2])},
                "properties": {"imageFile": fn,
                               "poseZ": float(sps[:, 2].mean()),
                               "parent": rec["name"],
                               "nPoints": int(len(sps)),
                               "fill": round(fill, 2),
                               "heightLevel": lvl,
                               "bottomAboveFloor": round(zlo_rel, 2),
                               "n2Pose": [round(float(sp[:, 0].mean()), 3),
                                          round(float(sp[:, 1].mean()), 3)]}})
            print(f"  {name:26s} pts={len(sps):6d} fill={fill:.2f} "
                  f"L{ext[0]:.2f} W{ext[1]:.2f} H{ext[2]:.2f} "
                  f"lvl={lvl} bot={zlo_rel:.2f}")
    json.dump({"semanticObjects": subs},
              open(os.path.join(OUT, "semanticObjects.json"), "w"), indent=1)
    print(f"\n{len(subs)} v2 subs; residual accounting:")
    for r in report:
        print(f"  {r['parent']}: residual={r['residual']} "
              f"wall_band={r['wall_band']} outside={r['outside_room']}")

    if not subs:
        return
    subprocess.run([sys.executable,
                    os.path.join(ROOT, "scripts", "render_object_views.py"),
                    "--semantic", os.path.join(OUT, "semanticObjects.json"),
                    "--objects-dir", OUT, "--out", OUT,
                    "--size", str(args.render_size)], check=True)
    if not args.vlm:
        print("(render done; rerun with --vlm to classify)")
        return

    from blk360seg import vlm_stage_b as vb
    from scripts.clutter_reverify import _SYS_RECOVERY
    vlm = vb.SemanticVLM(model=args.model)
    results = []
    for s in subs:
        views = [os.path.join(OUT, "views", s["name"], f"view_{a:03d}.png")
                 for a in (0, 90, 180, 270)]
        views = [v for v in views if os.path.exists(v)]
        out = {}
        for tag, cands in (("open", []), ("domain", DOMAIN)):
            content = vlm.build_symbolic_request(
                s, views, allowed_types=cands, floor_z=None,
                height_level=s["properties"]["heightLevel"])
            out[tag] = vlm._call(_SYS_RECOVERY, content, vb.SYMBOLIC_TOOL)
        a, b = out["open"]["corrected_type"], out["domain"]["corrected_type"]
        agree = (a == b) or (a in b) or (b in a)
        adopted = a if (agree and a != "clutter") else None
        results.append({"name": s["name"],
                        "parent": s["properties"]["parent"],
                        "open": out["open"], "domain": out["domain"],
                        "adopted": adopted})
        print(f"  {s['name']:26s} open={a:20s} domain={b:20s} "
              f"-> {'ADOPT ' + adopted if adopted else 'keep clutter'}")
    usage = vlm.usage_summary()
    json.dump({"model": args.model, "render_size": args.render_size,
               "usage": usage, "residual_accounting": report,
               "subs": subs, "results": results},
              open(os.path.join(OUT, "v2_report.json"), "w"), indent=1)
    print(f"wrote {OUT}/v2_report.json; cost ${usage['cost_usd']}")


if __name__ == "__main__":
    main()
