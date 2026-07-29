#!/usr/bin/env python3
"""TOSM robot layer: build semanticRobot record(s) from the robot's URDF +
nav config, and ingest into the knowledge graph.

Mirrors the semanticObject three-attribute structure:
  symbolic  what the robot is (model, drive type)
  explicit  measured geometry/kinematics/sensors (from URDF + nav params)
  implicit  affordances/state the planner reasons over (canManipulate,
            battery, isDocked)

Robots live in a separate top-level "robots" array of the KG json (they are
agents, not detections), each carrying its containing place resolved from
the place layer.

  .venv/bin/python scripts/build_semantic_robot.py \
      --urdf /home/caselab/blk360_ros2_ws/src/ammr_description/urdf/ammr.urdf \
      --nav-params /home/caselab/blk360_ros2_ws/src/blk360_bringup/config/nav2/nav2_params.yaml \
      --places outputs/place_layer_T3_slic.json \
      --pose 0 0 0 \
      --kg outputs/testroom_epochs_kg.json \
      --out outputs/semanticRobot.json
"""
import argparse
import json
import math
import re
import xml.etree.ElementTree as ET

import numpy as np


def parse_urdf(path):
    r = ET.parse(path).getroot()
    spec = {"model": r.get("name"), "links": len(r.findall("link"))}
    for l in r.findall("link"):
        if l.get("name") == "base_link":
            g = l.find("collision/geometry/box")
            if g is not None:
                L, W, H = map(float, g.get("size").split())
                spec["footprint"] = {"length": L, "width": W, "height": H}
    wheels, steers, sensors = [], [], []
    for j in r.findall("joint"):
        n = j.get("name", "")
        if "wheel" in n and j.get("type") == "continuous":
            wheels.append(n)
        if "steer" in n and j.get("type") == "continuous":
            steers.append(n)
        if any(s in n for s in ("lidar", "laser", "camera", "imu")):
            o = j.find("origin")
            sensors.append({"joint": n,
                            "child": j.find("child").get("link"),
                            "xyz": [float(v) for v in
                                    (o.get("xyz") or "0 0 0").split()]})
    for l in r.findall("link"):
        n = l.get("name", "")
        if "wheel" in n:
            c = l.find("collision/geometry/cylinder")
            if c is not None:
                spec["wheel_radius"] = float(c.get("radius"))
            break
    spec["drive"] = ("twin_steer" if steers else
                     "differential" if len(wheels) == 2 else "unknown")
    spec["driven_wheels"] = len(wheels)
    spec["steer_joints"] = len(steers)
    spec["sensors"] = sensors
    spec["has_manipulator"] = any("arm" in l.get("name", "")
                                  for l in r.findall("link"))
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--nav-params", default=None)
    ap.add_argument("--places", default=None)
    ap.add_argument("--pose", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                    metavar=("X", "Y", "THETA"),
                    help="initial pose in the map frame")
    ap.add_argument("--name", default=None)
    ap.add_argument("--kg", default=None, help="ingest into this KG json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = parse_urdf(args.urdf)

    limits = {}
    if args.nav_params:
        txt = open(args.nav_params).read()
        for key, pat in (("max_lin_vel_mps", r"vx_max:\s*([\d.]+)"),
                         ("max_ang_vel_rps", r"wz_max:\s*([\d.]+)"),
                         ("nav_radius_m", r"robot_radius:\s*([\d.]+)")):
            m = re.search(pat, txt)
            if m:
                limits[key] = float(m.group(1))

    x, y, th = args.pose
    place = None
    if args.places:
        d = json.load(open(args.places))
        best, bd = None, 1e9
        for p in d["semanticPlaces"]:
            cel = np.array(p["cells"])
            dd = float(np.min((cel[:, 0] - x) ** 2 + (cel[:, 1] - y) ** 2))
            if dd < bd:
                best, bd = p["name"], dd
        place = best

    name = args.name or f"{spec['model']}_001"
    robot = {
        "id": f"rob-{spec['model']}-001",
        "name": name,
        "mapId": "testroom",
        "symbolic": {
            "type": "mobile_manipulator" if spec["has_manipulator"]
                    else "mobile_robot",
            "model": spec["model"].upper(),
            "drive": spec["drive"],
        },
        "explicit": {
            "pose": {"x": x, "y": y, "theta": th},
            "footprint": spec.get("footprint"),
            "wheel_radius_m": spec.get("wheel_radius"),
            "driven_wheels": spec["driven_wheels"],
            "steer_joints": spec["steer_joints"],
            "sensors": spec["sensors"],
            "limits": limits,
        },
        "implicit": {
            "isMovable": True,
            "canManipulate": spec["has_manipulator"],
            "canNavigate": True,
            "isDocked": False,
            "battery": {"level": 1.0, "low_threshold": 0.2},
        },
        "isInsideOf": place,
        "source": {"urdf": args.urdf, "nav_params": args.nav_params},
    }
    json.dump({"semanticRobots": [robot]}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}: {name} ({robot['symbolic']['type']}, "
          f"{spec['drive']}, place={place})")

    if args.kg:
        g = json.load(open(args.kg))
        robots = g.setdefault("robots", [])
        robots[:] = [r for r in robots if r["id"] != robot["id"]]
        robots.append(robot)
        json.dump(g, open(args.kg, "w"), indent=1)
        print(f"ingested into {args.kg} (rev {g.get('revision')}, "
              f"robots={len(robots)})")


if __name__ == "__main__":
    main()
