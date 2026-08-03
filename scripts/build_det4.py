#!/usr/bin/env python3
"""Adjudicate the trackA v2 sub-objects and build det4 + KG rev7.

Adjudication layers on top of the two-pass consensus, fixing the two
failure modes the v2 run exposed:
  1. synonym-blind agreement: "robot_arm" vs "robot manipulator" is a
     semantic match the substring test missed -> canonical label table.
  2. non-clutter disagreement (table vs machine): a focused tie-break
     pass with exactly the two disputed labels (+"neither") resolves it
     without re-opening candidate anchoring.
  3. owner declarations: the owner's ground-truth statement (2026-07-31,
     "the discarded compound is a robot arm on a desk + part of a mobile
     robot + wall noise") types subs the VLM cannot confidently read
     from a partial scan — same highest-trust-evidence path as the
     fire-extinguisher restoration, recorded in provenance.

det4 (sota frame) = det3 records + adjudicated v2 subs (plys copied).
KG rev7 = upsert of det3's in-room vis_n2 records + transformed v2 subs
+ the owner-restored fire extinguisher (so it is not diffed to absent).
Every record gains properties.heightLevel (low/mid/high).

  .venv/bin/python scripts/build_det4.py            # adjudicate only
  .venv/bin/python scripts/build_det4.py --apply    # + det4 + KG rev7
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DET3 = os.path.join(ROOT, "outputs", "vis_sota_det3")
V2 = os.path.join(ROOT, "outputs", "trackA_v2")
DET4 = os.path.join(ROOT, "outputs", "vis_sota_det4")
KG_PATH = os.path.join(ROOT, "outputs", "testroom_epochs_kg.json")
TRANSFORM = os.path.join(ROOT, "outputs", "vissota_to_visn2_T.npy")
LIDAR_Z, CEIL_REL = 0.35, 1.8

SYN = {
    "robot arm": "robot manipulator", "robotic arm": "robot manipulator",
    "manipulator": "robot manipulator", "robot_arm": "robot manipulator",
    "table": "desk", "work table": "desk", "workbench": "desk",
    "pillar": "column", "pole": "column",
    "drawer unit": "cabinet", "drawers": "cabinet", "shelf": "cabinet",
    "agv": "mobile robot",
}
OWNER_DECL = {
    # owner 2026-07-31: the compound is arm-on-desk + partial mobile robot
    "room_013_v2_c2_below0": ("mobile robot",
                              "owner declaration 2026-07-31: partially "
                              "scanned mobile robot parked beside the desk"),
    "room_013_v2_c2_desk": ("desk",
                            "owner declaration 2026-07-31: desk carrying "
                            "the robot manipulator"),
    "room_013_v2_c2_pole1": ("robot manipulator",
                             "owner declaration 2026-07-31: robot arm "
                             "mounted at the desk"),
}


def canon(label):
    s = str(label).strip().lower().replace("_", " ")
    return SYN.get(s, s)


def height_level(bot_rel, top_rel):
    if bot_rel > CEIL_REL:
        return "high"
    if bot_rel < 0.25 and top_rel > 0.5 * LIDAR_Z:
        return "low"
    return "mid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--timestamp", default="2026-07-31T14:00:00")
    args = ap.parse_args()

    rep = json.load(open(os.path.join(V2, "v2_report.json")))
    subs = {s["name"]: s for s in rep["subs"]}
    tiebreak_cost = 0.0
    decisions = []
    need_tiebreak = []
    for r in rep["results"]:
        a, b = canon(r["open"]["corrected_type"]), \
            canon(r["domain"]["corrected_type"])
        agree = (a == b) or (a in b) or (b in a)
        if agree and a != "clutter":
            decisions.append((r["name"], a, "verified_recovery",
                              max(r["open"].get("confidence", 0.7),
                                  r["domain"].get("confidence", 0.7)),
                              f"v2 consensus (canonical): open={a} "
                              f"domain={b}"))
        elif a != "clutter" and b != "clutter":
            need_tiebreak.append((r, a, b))
        else:
            decisions.append((r["name"], None, None, None, None))

    if need_tiebreak:
        from blk360seg import vlm_stage_b as vb
        from scripts.clutter_reverify import _SYS_RECOVERY
        vlm = vb.SemanticVLM(model=args.model)
        for r, a, b in need_tiebreak:
            s = subs[r["name"]]
            views = [os.path.join(V2, "views", s["name"],
                                  f"view_{az:03d}.png")
                     for az in (0, 90, 180, 270)]
            views = [v for v in views if os.path.exists(v)]
            content = vlm.build_symbolic_request(
                s, views, allowed_types=[a, b, "neither of these"],
                height_level=s["properties"]["heightLevel"])
            out = vlm._call(_SYS_RECOVERY, content, vb.SYMBOLIC_TOOL)
            c = canon(out["corrected_type"])
            conf = out.get("confidence", 0.0)
            print(f"tie-break {r['name']}: {a} vs {b} -> {c} ({conf})")
            if c in (a, b) and conf >= 0.6:
                decisions.append((r["name"], c, "verified_recovery", conf,
                                  f"v2 tie-break: {a} vs {b} -> {c}"))
            else:
                decisions.append((r["name"], None, None, None, None))
        tiebreak_cost = vlm.usage_summary()["cost_usd"]
        print(f"tie-break cost ${tiebreak_cost}")

    final = {}
    for name, typ, st, conf, reason in decisions:
        if name in OWNER_DECL and st is None:
            typ, reason = OWNER_DECL[name]
            st, conf = "verified_owner", 1.0
        elif name in OWNER_DECL and canon(typ) != canon(OWNER_DECL[name][0]):
            # VLM and owner disagree -> owner wins, keep both in the reason
            vlm_typ = typ
            typ, reason = OWNER_DECL[name]
            reason += f" (VLM consensus said: {vlm_typ})"
            st, conf = "verified_owner", 1.0
        final[name] = (typ, st, conf, reason)
        print(f"{name:28s} -> {typ or 'clutter':18s} "
              f"{st or 'unverified'} {reason or ''}")

    if not args.apply:
        return

    det3 = json.load(open(os.path.join(
        DET3, "semanticObjects.lf_esc.json")))["semanticObjects"]
    det3_n2 = json.load(open(os.path.join(
        DET3, "semanticObjects.lf_esc.visn2frame.room.json")))[
        "semanticObjects"]
    T = np.load(TRANSFORM)
    yaw = float(np.arctan2(T[1, 0], T[0, 0]))

    # height level for existing det3 records (per-frame floor: 25th pct of
    # record bottoms, as in the recovery pipeline)
    def add_levels(records):
        bots = sorted(r["properties"].get("poseZ", 0.0)
                      - r["dimensions"]["height"] / 2 for r in records)
        floor = bots[len(bots) // 4]
        for r in records:
            z = r["properties"].get("poseZ", 0.0)
            h = r["dimensions"]["height"]
            r["properties"]["heightLevel"] = height_level(
                z - h / 2 - floor, z + h / 2 - floor)

    add_levels(det3)
    add_levels(det3_n2)

    v2recs, v2recs_n2 = [], []
    for s in rep["subs"]:
        typ, st, conf, reason = final[s["name"]]
        r = dict(s)
        r["type"] = typ or "clutter"
        r["confidence"] = conf if conf else 0.3
        pr = dict(r["properties"])
        pr["source"] = "resplit-v2 (wall-strip + support-plane split)"
        pr["verificationStatus"] = st or "unverified"
        pr["symbolicReason"] = reason or "v2 sub, no consensus"
        pr["isKeyObject"] = typ in ("mobile robot", "robot manipulator")
        pr["isMovable"] = typ not in ("column", "desk")
        r["properties"] = pr
        v2recs.append(r)
        n = json.loads(json.dumps(r))
        p = T @ np.array([r["poseX"], r["poseY"],
                          pr.get("poseZ", 0.0), 1.0])
        n["poseX"], n["poseY"] = round(float(p[0]), 4), round(float(p[1]), 4)
        n["properties"]["poseZ"] = round(float(p[2]), 4)
        n["poseTheta"] = round(float(np.arctan2(
            np.sin(r.get("poseTheta", 0.0) + yaw),
            np.cos(r.get("poseTheta", 0.0) + yaw))), 4)
        v2recs_n2.append(n)

    os.makedirs(DET4, exist_ok=True)
    for f in glob.glob(os.path.join(DET3, "obj_*.ply")):
        shutil.copy(f, DET4)
    for f in glob.glob(os.path.join(V2, "obj_*.ply")):
        shutil.copy(f, DET4)
    json.dump({"semanticObjects": det3 + v2recs},
              open(os.path.join(DET4, "semanticObjects.lf_esc.json"), "w"),
              indent=1)
    json.dump({"semanticObjects": det3_n2 + v2recs_n2},
              open(os.path.join(
                  DET4, "semanticObjects.lf_esc.visn2frame.room.json"),
                  "w"), indent=1)
    print(f"det4: {len(det3)} det3 + {len(v2recs)} v2 -> {DET4}")

    # KG rev7: full record set incl. the owner-restored fire extinguisher
    kg_graph = json.load(open(KG_PATH))
    fe = [n for n in kg_graph["nodes"]
          if n["name"] == "fire_extinguisher_060"][0]
    fe_rec = {
        "type": fe["type"], "id": fe["name"], "name": fe["name"],
        "poseX": fe["pose"]["x"], "poseY": fe["pose"]["y"],
        "poseTheta": fe["pose"].get("theta", 0.0),
        "dimensions": fe["dimensions"], "color": fe.get("color"),
        "confidence": fe["confidence"],
        "properties": {"poseZ": fe["pose"].get("z", 0.0),
                       "verificationStatus": fe["status"],
                       "source": fe["provenance"].get("source"),
                       "symbolicReason": fe["provenance"].get("reason"),
                       "isKeyObject": fe["implicit"].get("isKeyObject"),
                       "isMovable": fe["implicit"].get("isMovable"),
                       "heightLevel": "low"},
    }
    from blk360seg import kg
    from blk360seg.spatial_relations import compute_relations
    records = det3_n2 + v2recs_n2 + [fe_rec]
    rels = compute_relations(records)
    edges = [(r["subject"], r["predicate"], r["object"]) for r in rels]
    shutil.copy(KG_PATH, KG_PATH.replace(".json", ".rev6.bak.json"))
    diff = kg.upsert(kg_graph, records, "testroom", args.timestamp,
                     edges=edges)
    json.dump(kg_graph, open(KG_PATH, "w"), indent=1, ensure_ascii=False)
    print(f"KG rev{kg_graph['revision']}: "
          + ", ".join(f"{k}={len(v) if isinstance(v, list) else v}"
                      for k, v in diff.items() if k != "timestamp"))


if __name__ == "__main__":
    main()
