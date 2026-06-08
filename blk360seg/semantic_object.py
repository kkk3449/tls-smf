"""Stage A — deterministic builder for the VDA5050-style semanticObject message.

From DBSCAN+Uni3D output (obj_*.ply + objects.csv [+ classification.csv]) this
fills the **explicit model** (pose, size, color — pure point-cloud geometry,
never touched by the VLM) and a **provisional symbolic model** (type/name/id
from Uni3D). The **implicit model** (isKeyObject, isMovable, isOpen, canBeOpen)
is left for the VLM stage (Stage B).

TOSM (Joo & Kuc, 2020, Table 3) -> VDA5050 JSON mapping:
  symbolic.name/ID      -> name / id
  (object class)        -> type
  explicit.pose         -> poseX, poseY, poseTheta(yaw);  poseZ -> properties
  explicit.size         -> dimensions {length, width, height}
  explicit.color        -> color (nearest CSS-ish name from mean RGB)
  explicit.coordFrame   -> properties.coordinateFrame (or mapId)
  implicit.*            -> filled in Stage B (VLM)

A candidate name set may be supplied (e.g. {chair, TV, monitor, Mobile Robot,
desk}); it is recorded top-level as `allowedTypes` so Stage B constrains the
VLM's corrected `type` to it. If empty, the pipeline stays open-vocabulary.
"""
import glob
import os

import numpy as np

# Coarse color vocabulary -> the `color` string field (TOSM color is r,g,b).
BASE_COLORS = {
    "red": (0.60, 0.10, 0.10), "orange": (0.85, 0.45, 0.10),
    "yellow": (0.85, 0.75, 0.10), "green": (0.15, 0.50, 0.20),
    "blue": (0.10, 0.25, 0.60), "purple": (0.40, 0.15, 0.50),
    "brown": (0.40, 0.27, 0.16), "white": (0.85, 0.85, 0.85),
    "gray": (0.50, 0.50, 0.50), "black": (0.12, 0.12, 0.12),
}


def nearest_color(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    best, bd = None, 1e9
    for name, c in BASE_COLORS.items():
        d = float(np.sum((rgb - np.array(c)) ** 2))
        if d < bd:
            best, bd = name, d
    return best


def yaw_and_footprint(xyz):
    """PCA on XY -> yaw (rad) + tight length/width in the yaw-aligned frame."""
    xy = xyz[:, :2].astype(np.float64)
    c = xy.mean(0)
    q = xy - c
    if len(q) < 3:
        return 0.0, float(np.ptp(xy[:, 0]) if len(xy) else 0.0), \
               float(np.ptp(xy[:, 1]) if len(xy) else 0.0)
    cov = q.T @ q / len(q)
    w, v = np.linalg.eigh(cov)
    major = v[:, int(np.argmax(w))]
    yaw = float(np.arctan2(major[1], major[0]))
    ca, sa = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[ca, -sa], [sa, ca]])
    pr = q @ R.T
    length = float(np.ptp(pr[:, 0]))
    width = float(np.ptp(pr[:, 1]))
    return yaw, length, width


def object_geometry(xyz):
    mn, mx = xyz.min(0), xyz.max(0)
    cx, cy, cz = (mn + mx) / 2.0
    yaw, length, width = yaw_and_footprint(xyz)
    height = float(mx[2] - mn[2])
    return dict(poseX=float(cx), poseY=float(cy), poseZ=float(cz),
                poseTheta=yaw, length=length, width=width, height=height)


def build_objects(objects_dir, classification=None, map_id=None,
                  coordinate_frame="map", source="DBSCAN+Uni3D"):
    import open3d as o3d
    import pandas as pd

    odf = pd.read_csv(os.path.join(objects_dir, "objects.csv"))
    cls = {}
    if classification and os.path.exists(classification):
        cdf = pd.read_csv(classification)
        for _, r in cdf.iterrows():
            cls[r["file"]] = (str(r["top1"]), float(r["score1"]))

    objs = []
    for _, row in odf.iterrows():
        f = row["file"]
        pc = o3d.io.read_point_cloud(os.path.join(objects_dir, f))
        xyz = np.asarray(pc.points)
        rgb = np.asarray(pc.colors)
        if len(xyz) == 0:
            continue
        g = object_geometry(xyz)
        typ, conf = cls.get(f, ("", None))
        oid = str(int(row["id"]))
        name = f"{typ}_{int(row['id']):03d}" if typ else f"object_{int(row['id']):03d}"

        obj = {
            "type": typ,
            "id": oid,
            "name": name,
            "poseX": round(g["poseX"], 4),
            "poseY": round(g["poseY"], 4),
            "poseTheta": round(g["poseTheta"], 4),
            "dimensions": {
                "length": round(g["length"], 3),
                "width": round(g["width"], 3),
                "height": round(g["height"], 3),
            },
        }
        if rgb.size:
            obj["color"] = nearest_color(rgb.mean(0))
        if conf is not None:
            obj["confidence"] = round(conf, 3)
        if map_id:
            obj["mapId"] = map_id
        # explicit overflow (3D) + provenance; implicit fields added in Stage B
        obj["properties"] = {
            "poseZ": round(g["poseZ"], 4),
            "coordinateFrame": coordinate_frame,
            "source": source,
            "symbolicVerified": False,   # Stage B (VLM) sets True after prompt 1
            "imageFile": f,
        }
        objs.append(obj)
    return objs


def build_message(objs, candidate_names=None, manufacturer="CASELAB",
                  serial="SERIALNUMBER", version="0.0.0", header_id=0,
                  timestamp=None):
    if timestamp is None:
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-4] + "Z"
    msg = {
        "headerId": header_id,
        "timestamp": timestamp,
        "version": version,
        "manufacturer": manufacturer,
        "serialNumber": serial,
        "semanticObjects": objs,
    }
    if candidate_names:
        # controlled vocabulary for Stage B (VLM prompt 1 allowed `type` set).
        msg["allowedTypes"] = list(candidate_names)
    return msg
