#!/usr/bin/env python3
"""Transform a semanticObjects json by a 4x4 (optional) and room-scope it with a
bounds json (cells at cell_m). Writes <stem>.kgframe.json and
<stem>.kgframe.room.json; prints the in-room names (comma-separated) on the
last line for vlm_late_fusion --select.

  .venv/bin/python scripts/roomscope_any.py --src X.json [--T T.npy] \
      --bounds outputs/vis_n2_room_bounds.json [--tol 0.0] [--frame vis_n2]
"""
import argparse, json, math
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--T", default=None)
ap.add_argument("--bounds", required=True); ap.add_argument("--tol", type=float, default=0.0)
ap.add_argument("--frame", default="vis_n2"); ap.add_argument("--suffix", default="kgframe")
a = ap.parse_args()
T = np.load(a.T) if a.T else np.eye(4)
yaw = float(np.arctan2(T[1, 0], T[0, 0]))
rb = json.load(open(a.bounds)); cells = np.array(rb["cells"]); cm = rb["cell_m"]
doc = json.load(open(a.src)); objs = doc["semanticObjects"]
inside = []
for o in objs:
    p = T @ np.array([o["poseX"], o["poseY"], o["properties"].get("poseZ", 0.0), 1.0])
    o["poseX"], o["poseY"] = round(float(p[0]), 4), round(float(p[1]), 4)
    o["properties"]["poseZ"] = round(float(p[2]), 4)
    o["poseTheta"] = round(float((o.get("poseTheta", 0.0) + yaw + np.pi) % (2 * np.pi) - np.pi), 4)
    o["properties"]["coordinateFrame"] = a.frame
    c = np.array([math.floor(p[0] / cm), math.floor(p[1] / cm)])
    dmin = float(np.min(np.hypot(cells[:, 0] - c[0], cells[:, 1] - c[1])) * cm)
    o["properties"]["roomDistance"] = round(dmin, 2)
    if dmin <= a.tol:
        inside.append(o)
stem = a.src[:-5]
json.dump(doc, open(f"{stem}.{a.suffix}.json", "w"), indent=1, ensure_ascii=False)
d2 = dict(doc); d2["semanticObjects"] = inside
json.dump(d2, open(f"{stem}.{a.suffix}.room.json", "w"), indent=1, ensure_ascii=False)
print(f"in-room {len(inside)} / {len(objs)}")
print(",".join(o["name"] for o in inside))
