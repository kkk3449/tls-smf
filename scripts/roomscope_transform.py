#!/usr/bin/env python3
"""Transform T3 (vis_sota) semanticObjects into the vis_n2 frame and keep
only objects inside the test-room mask (vis_n2_room_bounds.json cells).

Writes <stem>.visn2frame.json (all, transformed) and
<stem>.visn2frame.room.json (in-room only) and prints the in-room name
list for vlm_late_fusion --select.
"""
import json
import sys

import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else \
    "outputs/vis_sota_det/semanticObjects.json"
T = np.load("outputs/vissota_to_visn2_T.npy")
yaw = float(np.arctan2(T[1, 0], T[0, 0]))
rb = json.load(open("outputs/vis_n2_room_bounds.json"))
cells = {tuple(c) for c in rb["cells"]}
cell_m = rb["cell_m"]

doc = json.load(open(SRC))
objs = doc["semanticObjects"]
inside, outside = [], []
for o in objs:
    p = T @ np.array([o["poseX"], o["poseY"],
                      o["properties"].get("poseZ", 0.0), 1.0])
    o["poseX"], o["poseY"] = round(float(p[0]), 4), round(float(p[1]), 4)
    o["properties"]["poseZ"] = round(float(p[2]), 4)
    o["poseTheta"] = round(float((o.get("poseTheta", 0.0) + yaw + np.pi)
                                 % (2 * np.pi) - np.pi), 4)
    o["properties"]["coordinateFrame"] = "vis_n2"
    cell = (int(np.floor(p[0] / cell_m)), int(np.floor(p[1] / cell_m)))
    (inside if cell in cells else outside).append(o)

stem = SRC[:-5]
doc["semanticObjects"] = inside + outside
json.dump(doc, open(stem + ".visn2frame.json", "w"), indent=1,
          ensure_ascii=False)
doc["semanticObjects"] = inside
json.dump(doc, open(stem + ".visn2frame.room.json", "w"), indent=1,
          ensure_ascii=False)
print(f"in-room {len(inside)} / outside {len(outside)}")
print("outside:", [o["name"] for o in outside])
print("SELECT:", ",".join(o["name"] for o in inside))
