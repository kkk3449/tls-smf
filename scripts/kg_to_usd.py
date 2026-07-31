#!/usr/bin/env python3
"""Build the Isaac Sim scene DIRECTLY from the semantic KG json — the KG is
the single semantic DB; the web UI / mediator already read it live, and this
makes the USD a pure derivation of the same file (owner decision 2026-08-01:
one DB, no drift between Isaac and the Gazebo UI).

- objects  = present KG nodes (name/type/status/confidence/implicit incl.
  heightLevel & labelStability, provenance reason). Nodes carry a
  `pointCloud` ply reference (assets dir); nodes without one render as
  extent boxes — the existing placeholder convention.
- edges come from the KG itself (no separate relations.json).
- FRAME: everything is emitted in the KG/map (vis_n2) frame — the same
  frame as the nav map, mediator goals, and the web UI. Point clouds and
  the environment ply (sota frame on disk) are transformed on load.

  .venv/bin/python scripts/kg_to_usd.py                # build once
  .venv/bin/python scripts/kg_to_usd.py --watch        # rebuild on KG change
  (Isaac: File > Open t4_kg_scene.usda; File > Reload after a rebuild)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from blk360seg import usd_export as ux  # noqa: E402

KG = os.path.join(ROOT, "outputs", "testroom_epochs_kg.json")
ASSETS = os.path.join(ROOT, "outputs", "vis_sota_det4")
ENV = os.path.join(ROOT, "outputs", "vis_sota_det2", "env_room_ceil.ply")
TRANSFORM = os.path.join(ROOT, "outputs", "vissota_to_visn2_T.npy")
OUT = os.path.join(ROOT, "outputs", "vis_sota_det4", "t4_kg_scene.usda")
FLOOR_N2 = -1.0


def node_to_record(n):
    imp = n.get("implicit", {})
    return {
        "type": n["type"], "id": n["name"], "name": n["name"],
        "poseX": n["pose"]["x"], "poseY": n["pose"]["y"],
        "poseTheta": n["pose"].get("theta", 0.0),
        "dimensions": n["dimensions"], "color": n.get("color"),
        "confidence": n.get("confidence", 0.0),
        "properties": {
            "poseZ": n["pose"].get("z", 0.0),
            "imageFile": n.get("pointCloud"),
            "verificationStatus": n.get("status"),
            "symbolicReason": n.get("provenance", {}).get("reason", ""),
            **{k: v for k, v in imp.items() if v is not None},
        },
    }


def build():
    import open3d as o3d
    g = json.load(open(KG))
    T = np.load(TRANSFORM)
    nodes = [n for n in g["nodes"] if n.get("presence") == "present"]
    objects = [node_to_record(n) for n in nodes]
    names = {o["name"] for o in objects}
    relations = [{"subject": e["subject"], "predicate": e["predicate"],
                  "object": e["object"]} for e in g.get("edges", [])
                 if e.get("subject") in names and e.get("object") in names]
    xs = [o["poseX"] for o in objects]
    ys = [o["poseY"] for o in objects]
    places = [{"id": "place_1", "name": "testroom", "type": "room",
               "centroid": [float(np.mean(xs)), float(np.mean(ys))],
               "area_m2": 0.0, "synthesized": True,
               "bbox": [min(xs) - 1, min(ys) - 1, max(xs) + 1, max(ys) + 1],
               "polygon": [[min(xs) - 1, min(ys) - 1],
                           [max(xs) + 1, min(ys) - 1],
                           [max(xs) + 1, max(ys) + 1],
                           [min(xs) - 1, max(ys) + 1]]}]
    graph = {"objects": objects, "places": places, "relations": relations}

    clouds = {}
    for o in objects:
        f = o["properties"].get("imageFile")
        if not f:
            continue
        p = os.path.join(ASSETS, f)
        if not os.path.exists(p):
            continue
        pc = o3d.io.read_point_cloud(p)
        xyz = np.asarray(pc.points)
        xyz = (T[:3, :3] @ xyz.T).T + T[:3, 3]      # sota -> map frame
        rgb = np.asarray(pc.colors) if pc.has_colors() else \
            np.full_like(xyz, 0.6)
        if len(xyz) > 4000:
            idx = np.linspace(0, len(xyz) - 1, 4000).astype(int)
            xyz, rgb = xyz[idx], rgb[idx]
        clouds[o["name"]] = (xyz, rgb)

    env = None
    if os.path.exists(ENV):
        pc = o3d.io.read_point_cloud(ENV)
        xyz = np.asarray(pc.points)
        xyz = (T[:3, :3] @ xyz.T).T + T[:3, 3]
        rgb = np.asarray(pc.colors) if pc.has_colors() else \
            np.full_like(xyz, 0.6)
        if len(xyz) > 300000:
            idx = np.linspace(0, len(xyz) - 1, 300000).astype(int)
            xyz, rgb = xyz[idx], rgb[idx]
        env = (xyz, rgb)

    usda = ux.to_usda(graph, clouds=clouds, environment=env,
                      floor_offset=-FLOOR_N2, point_width=0.02)
    with open(OUT, "w") as f:
        f.write(usda)
    print(f"[KG->USD] rev{g.get('revision')} {len(objects)} objects "
          f"({len(clouds)} with clouds) + {len(relations)} relations "
          f"-> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true",
                    help="poll the KG mtime and rebuild on change")
    args = ap.parse_args()
    build()
    if not args.watch:
        return
    last = os.path.getmtime(KG)
    print("[KG->USD] watching for KG changes (Ctrl-C to stop; "
          "reload the stage in Isaac after each rebuild)")
    while True:
        time.sleep(3)
        mt = os.path.getmtime(KG)
        if mt != last:
            last = mt
            time.sleep(1)          # let the writer finish
            try:
                build()
            except Exception as ex:
                print(f"[KG->USD] rebuild failed: {ex}")


if __name__ == "__main__":
    main()
