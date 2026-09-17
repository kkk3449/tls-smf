#!/usr/bin/env python3
"""Overhead layer across epochs: upsert T2 then T3 overhead records into a
copy of the knowledge graph and report the identity outcome (matched / moved /
inserted / absent), plus a merged floor+overhead export for the twin.

  .venv/bin/python scripts/overhead_kg_epochs.py
"""
import json, os, shutil, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from blk360seg import kg  # noqa: E402

T = np.load(f"{ROOT}/outputs/vissota_to_visn2_T.npy")
yaw = float(np.arctan2(T[1, 0], T[0, 0]))


def to_kg_frame(recs):
    out = []
    for o in recs:
        o = json.loads(json.dumps(o))
        p = T @ np.array([o["poseX"], o["poseY"], o["properties"]["poseZ"], 1.0])
        o["poseX"], o["poseY"] = round(float(p[0]), 4), round(float(p[1]), 4)
        o["properties"]["poseZ"] = round(float(p[2]), 4)
        o["poseTheta"] = round(float((o.get("poseTheta", 0.0) + yaw + np.pi) % (2 * np.pi) - np.pi), 4)
        o["properties"]["coordinateFrame"] = "vis_n2"
        out.append(o)
    return out


def load(p):
    d = json.load(open(p))
    return d["semanticObjects"] if isinstance(d, dict) else d


t2 = load(f"{ROOT}/outputs/vis_n2_overhead_c0.15/semanticObjects.room.lf_esc.json")
t3 = load(f"{ROOT}/outputs/vis_sota_overhead_c0.15/semanticObjects.room.carried.json") + \
     load(f"{ROOT}/outputs/vis_sota_overhead_c0.15/semanticObjects.room.new.lf_esc.json")
t3 = to_kg_frame(t3)
for r in t2 + t3:
    r["properties"]["layer"] = "overhead"

g = kg.new_graph("vis_n2_overhead")
d2 = kg.upsert(g, t2, "vis_n2_overhead", "2026-07-14T15:00:00", edges=[])
d3 = kg.upsert(g, t3, "vis_n2_overhead", "2026-07-23T15:00:00", edges=[])
summ = {k: {c: len(v[c]) for c in ("updated", "unchanged", "moved", "inserted", "absent")}
        for k, v in (("T2", d2), ("T3", d3))}
print(json.dumps(summ, indent=1))
for n in g["nodes"]:
    print(f"  {n['id'][:10]} {n.get('name'):<24} {n.get('status'):<19} presence={n.get('presence')} "
          f"revisions={[h['revision'] for h in n.get('history', [])]}")
json.dump(g, open(f"{ROOT}/outputs/overhead_epochs_kg.json", "w"), indent=1, ensure_ascii=False)
json.dump(summ, open(f"{ROOT}/outputs/overhead_epochs_summary.json", "w"), indent=1)
