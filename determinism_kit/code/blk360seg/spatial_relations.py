"""TOSM spatialRelation modeling from object global poses (deterministic).

Given the semantic-object records (poseX, poseY, properties.poseZ, poseTheta,
dimensions = length/width/height in the object frame, z = gravity-aligned), this
computes the object-object spatial relations of the TOSM ontology (Joo & Kuc 2020,
Table 2) as a knowledge graph of triples (subject, predicate, object):

  isNextTo   (symmetric)  horizontal footprints close, at the same vertical level
  isOn       (A on B)     A rests on B: A's base ≈ B's top, footprints overlap
  isAboveOf  (A above B)  A clearly above B with footprint overlap (not resting)
  isInFrontOf / isBehindOf (A rel. B, orientation-dependent, OPTIONAL)

These are pure functions of pose + oriented bounding box, so they are exact,
reproducible and free — no VLM needed. The orientation-dependent pair uses
poseTheta (PCA yaw), whose facing sign is ambiguous; it is off by default and
emitted with low confidence when enabled (resolve facing with a VLM/heuristic
later if needed). Place-level relations (isInsideOf object→place, isConnectedTo
place→place) need place segmentation and are out of scope here.
"""
import numpy as np


def _obb_corners_2d(cx, cy, length, width, theta):
    """Footprint corners of an oriented box (length along local x, width local y)."""
    hx, hy = length / 2.0, width / 2.0
    c, s = np.cos(theta), np.sin(theta)
    local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([cx, cy])


def _obb_overlap_2d(a, b):
    """Separating-axis test for two convex quads -> True if they overlap in XY."""
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            norm = np.hypot(*axis)
            if norm < 1e-9:
                continue
            axis /= norm
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() - 1e-9 or pb.max() < pa.min() - 1e-9:
                return False
    return True


def _obj_geom(o):
    d = o["dimensions"]
    z = o["properties"].get("poseZ", 0.0)
    h = d["height"]
    return {
        "name": o["name"],
        "cx": o["poseX"], "cy": o["poseY"], "cz": z,
        "length": d["length"], "width": d["width"], "height": h,
        "theta": o.get("poseTheta", 0.0),
        "bottom": z - h / 2.0, "top": z + h / 2.0,
        "radius": 0.5 * np.hypot(d["length"], d["width"]),  # footprint circumradius
        "corners": _obb_corners_2d(o["poseX"], o["poseY"],
                                   d["length"], d["width"], o.get("poseTheta", 0.0)),
    }


def _vertical_overlap(a, b):
    return min(a["top"], b["top"]) - max(a["bottom"], b["bottom"])


def compute_relations(objects, next_to_gap=0.40, on_tol=0.15, above_gap=0.10,
                      orientation=False, front_halfangle_deg=45.0):
    """Return a list of relation triples computed from object geometry.

    Each triple: {subject, predicate, object, ...evidence}. isNextTo is emitted
    once per unordered pair (symmetric); the others are directional.
    """
    g = [_obj_geom(o) for o in objects]
    rels = []
    n = len(g)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = g[i], g[j]
            dxy = np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])
            overlap_xy = _obb_overlap_2d(a["corners"], b["corners"])

            # --- isOn (A on B): A's base near B's top, footprints overlap ---
            if (overlap_xy and a["cz"] > b["cz"]
                    and abs(a["bottom"] - b["top"]) <= on_tol):
                rels.append({"subject": a["name"], "predicate": "isOn",
                             "object": b["name"],
                             "z_gap": round(float(a["bottom"] - b["top"]), 3)})
            # --- isAboveOf (A above B): clear vertical gap, footprints overlap ---
            elif overlap_xy and a["bottom"] > b["top"] + above_gap:
                rels.append({"subject": a["name"], "predicate": "isAboveOf",
                             "object": b["name"],
                             "z_gap": round(float(a["bottom"] - b["top"]), 3)})

            # --- isInFrontOf / isBehindOf (directional, optional) ---
            if orientation and i != j:
                # bearing of A in B's local frame (B faces local +x = poseTheta)
                vx, vy = a["cx"] - b["cx"], a["cy"] - b["cy"]
                ang = np.degrees(np.arctan2(vy, vx) - b["theta"])
                ang = (ang + 180.0) % 360.0 - 180.0   # wrap to [-180,180]
                if abs(ang) <= front_halfangle_deg:
                    rels.append({"subject": a["name"], "predicate": "isInFrontOf",
                                 "object": b["name"], "bearing_deg": round(float(ang), 1),
                                 "confidence": 0.5,
                                 "note": "theta facing-sign ambiguous"})
                elif abs(ang) >= 180.0 - front_halfangle_deg:
                    rels.append({"subject": a["name"], "predicate": "isBehindOf",
                                 "object": b["name"], "bearing_deg": round(float(ang), 1),
                                 "confidence": 0.5,
                                 "note": "theta facing-sign ambiguous"})

        # --- isNextTo (symmetric, once per unordered pair) ---
        for j in range(i + 1, n):
            a, b = g[i], g[j]
            gap = np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]) \
                - (a["radius"] + b["radius"])
            if gap <= next_to_gap and _vertical_overlap(a, b) > 0:
                rels.append({"subject": a["name"], "predicate": "isNextTo",
                             "object": b["name"], "gap_m": round(float(max(gap, 0.0)), 3),
                             "symmetric": True})
    return rels


def build_relation_message(src_message, objects, **kw):
    """Wrap computed relations in a JSON-serializable dict that mirrors the
    semanticObjects header so it slots into the same VDA 5050 / digital-twin
    consumer (Isaac Sim, knowledge graph)."""
    rels = compute_relations(objects, **kw)
    counts = {}
    for r in rels:
        counts[r["predicate"]] = counts.get(r["predicate"], 0) + 1
    return {
        "headerId": src_message.get("headerId", 0),
        "timestamp": src_message.get("timestamp"),
        "manufacturer": src_message.get("manufacturer", "CASELAB"),
        "mapId": src_message.get("mapId"),
        "coordinateFrame": "map",
        "relationCounts": counts,
        "objectRelations": rels,
    }
