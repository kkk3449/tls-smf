"""TOSM knowledge-graph layer: stable IDs, confidence-gated upsert, diffing.

The graph is a plain JSON document (nodes + edges + a per-node change log), so
it is diffable, versionable, and independent of any DB; `to_neo4j_cypher`
emits the same content as Cypher for Neo4j storage/querying.

Design (paper Sec. 4.5):
- Stable object ID = spatial anchor hash: quantized (x, y, z, l, w, h) + map
  ID. Deterministic segmentation -> identical re-runs produce identical IDs.
- Upsert: match incoming records to existing nodes by spatial overlap;
  matched -> attribute update (logged), missing -> marked `absent` (not
  deleted), new -> inserted. Node-level incremental update, no rebuild.
- Gating: verificationStatus 'verified' / 'verified_escalated' ingest as
  verified; 'unverified' ingests queryable but confidence-tagged.
"""
import hashlib
import json
import math


ANCHOR_Q = 0.25  # m — quantization step for the spatial anchor hash


def stable_id(rec, map_id=None, q=ANCHOR_Q):
    """Deterministic object ID from quantized pose + extents.

    Quantization makes the ID robust to sub-step jitter, while a moved or
    resized object gets a NEW id (its old node is then marked absent —
    movement is an observable event, not a silent mutation)."""
    d = rec["dimensions"]
    z = rec.get("properties", {}).get("poseZ", 0.0)
    key = (map_id or rec.get("mapId", "map"),
           *(int(math.floor(v / q)) for v in
             (rec["poseX"], rec["poseY"], z,
              d["length"], d["width"], d["height"])))
    h = hashlib.sha1(repr(key).encode()).hexdigest()[:10]
    return f"obj-{key[0]}-{h}"


def _gate(rec):
    """Map a Stage-B record's verification status to KG ingestion status."""
    st = rec.get("properties", {}).get("verificationStatus")
    if st in ("verified", "verified_escalated", "verified_majority"):
        return st
    if rec.get("properties", {}).get("symbolicVerified") and st is None:
        return "verified"          # pre-escalation-era records
    return "unverified"


def _node_from_record(rec, map_id, now):
    pr = rec.get("properties", {})
    return {
        "id": stable_id(rec, map_id),
        "name": rec["name"],
        "type": rec["type"],
        "mapId": map_id,
        "pose": {"x": rec["poseX"], "y": rec["poseY"],
                 "z": pr.get("poseZ", 0.0),
                 "theta": rec.get("poseTheta", 0.0)},
        "dimensions": rec["dimensions"],
        "color": rec.get("color"),
        "confidence": rec.get("confidence", 0.0),
        "status": _gate(rec),               # verified | verified_escalated | unverified
        "presence": "present",              # present | absent
        "implicit": {k: pr.get(k) for k in
                     ("isKeyObject", "isMovable", "isOpen", "canBeOpen")
                     if pr.get(k) is not None},
        "provenance": {"source": pr.get("source", "DBSCAN+Uni3D+VLM"),
                       "voteShare": pr.get("voteShare"),
                       "escalated": pr.get("escalated", False),
                       "viewVotes": pr.get("viewVotes"),
                       "reason": pr.get("symbolicReason", "")},
        "firstSeen": now,
        "lastSeen": now,
        "history": [],
    }


def _matches(node, rec, max_center_dist=0.5):
    """Spatial match: same map, centers within max_center_dist, and extents
    within 30% (or 0.15 m) on every axis."""
    pr = rec.get("properties", {})
    dx = node["pose"]["x"] - rec["poseX"]
    dy = node["pose"]["y"] - rec["poseY"]
    dz = node["pose"]["z"] - pr.get("poseZ", 0.0)
    if math.sqrt(dx * dx + dy * dy + dz * dz) > max_center_dist:
        return False
    nd, rd = node["dimensions"], rec["dimensions"]
    for k in ("length", "width", "height"):
        tol = max(0.3 * max(nd[k], rd[k]), 0.15)
        if abs(nd[k] - rd[k]) > tol:
            return False
    return True


def new_graph(map_id):
    return {"mapId": map_id, "nodes": [], "edges": [], "revision": 0,
            "updates": []}


# labels too generic to carry identity: two distinct piles of clutter with
# similar extents are NOT evidence of the same object having moved
# (cross-epoch experiment finding: clutter<->clutter "moves" of 2-5 m were
# unverifiable, while fire-extinguisher/chair moves were credible)
_GENERIC_TYPES = {"clutter", "unknown", "misc", "object"}


def _same_object_moved(node, rec, max_move=6.0):
    """Second-pass match for MOVED objects: same specific type, same extents
    (within the _matches tolerance), different position. Conservative: only
    fires when the node was about to go absent and the record was about to
    be a fresh insert, and never for generic type labels."""
    t = node["type"].strip().lower()
    if t != rec["type"].strip().lower() or t in _GENERIC_TYPES:
        return False
    nd, rd = node["dimensions"], rec["dimensions"]
    for k in ("length", "width", "height"):
        tol = max(0.3 * max(nd[k], rd[k]), 0.15)
        if abs(nd[k] - rd[k]) > tol:
            return False
    dx = node["pose"]["x"] - rec["poseX"]
    dy = node["pose"]["y"] - rec["poseY"]
    return math.hypot(dx, dy) <= max_move


def upsert(graph, records, map_id, timestamp, edges=None):
    """Node-level incremental update. Returns a diff summary dict.

    records: Stage-B semanticObjects list for the (re-)scan.
    edges:   optional [(subj_name, predicate, obj_name), ...] regenerated for
             the current scan (edge set is recomputed, nodes persist).

    Diff classes: unchanged / updated / moved / inserted / absent. A node's id
    is its anchor hash at FIRST sight (birth anchor) — identity persists
    across moves; matching is geometric, never id-based."""
    graph["revision"] += 1
    rev, now = graph["revision"], timestamp
    diff = {"revision": rev, "timestamp": now, "updated": [], "unchanged": [],
            "moved": [], "inserted": [], "absent": []}
    live = {n["id"]: n for n in graph["nodes"] if n["presence"] == "present"}
    claimed = set()
    unmatched_recs = []

    for rec in records:
        cand = [n for n in live.values()
                if n["id"] not in claimed and _matches(n, rec)]
        if cand:
            node = min(cand, key=lambda n:
                       (n["pose"]["x"] - rec["poseX"]) ** 2
                       + (n["pose"]["y"] - rec["poseY"]) ** 2)
            claimed.add(node["id"])
            fresh = _node_from_record(rec, map_id, now)
            changes = {}
            for k in ("type", "confidence", "status", "implicit", "color",
                      "pose", "dimensions"):
                if node.get(k) != fresh[k]:
                    changes[k] = {"old": node.get(k), "new": fresh[k]}
                    node[k] = fresh[k]
            node["provenance"] = fresh["provenance"]
            node["lastSeen"] = now
            if changes:
                node["history"].append({"revision": rev, "timestamp": now,
                                        "changes": changes})
                diff["updated"].append(node["id"])
            else:
                diff["unchanged"].append(node["id"])
        else:
            unmatched_recs.append(rec)

    # ---- second pass: moved objects (would-be absent x would-be insert) ----
    leftovers = []
    for rec in unmatched_recs:
        cand = [n for nid, n in live.items() if nid not in claimed
                and _same_object_moved(n, rec)]
        if cand:
            node = min(cand, key=lambda n:
                       (n["pose"]["x"] - rec["poseX"]) ** 2
                       + (n["pose"]["y"] - rec["poseY"]) ** 2)
            claimed.add(node["id"])
            fresh = _node_from_record(rec, map_id, now)
            old_pose = node["pose"]
            changes = {"pose": {"old": old_pose, "new": fresh["pose"]}}
            node["pose"] = fresh["pose"]
            for k in ("type", "confidence", "status", "implicit", "color",
                      "dimensions"):
                if node.get(k) != fresh[k]:
                    changes[k] = {"old": node.get(k), "new": fresh[k]}
                    node[k] = fresh[k]
            node["provenance"] = fresh["provenance"]
            node["lastSeen"] = now
            node["history"].append({"revision": rev, "timestamp": now,
                                    "event": "moved", "changes": changes})
            diff["moved"].append(node["id"])
        else:
            leftovers.append(rec)

    for rec in leftovers:
        node = _node_from_record(rec, map_id, now)
        if any(n["id"] == node["id"] for n in graph["nodes"]):
            # returned object (same anchor as an absent node) -> revive
            old = next(n for n in graph["nodes"] if n["id"] == node["id"])
            old.update({k: node[k] for k in
                        ("type", "confidence", "status", "implicit",
                         "provenance")})
            old["presence"] = "present"
            old["lastSeen"] = now
            old["history"].append({"revision": rev, "timestamp": now,
                                   "changes": {"presence":
                                               {"old": "absent",
                                                "new": "present"}}})
            diff["updated"].append(old["id"])
            claimed.add(old["id"])
        else:
            node["history"].append({"revision": rev, "timestamp": now,
                                    "changes": {"presence":
                                                {"old": None,
                                                 "new": "present"}}})
            graph["nodes"].append(node)
            diff["inserted"].append(node["id"])
            claimed.add(node["id"])

    for nid, node in live.items():
        if nid not in claimed:
            node["presence"] = "absent"
            node["history"].append({"revision": rev, "timestamp": now,
                                    "changes": {"presence":
                                                {"old": "present",
                                                 "new": "absent"}}})
            diff["absent"].append(nid)

    if edges is not None:
        name_to_id = {n["name"]: n["id"] for n in graph["nodes"]
                      if n["presence"] == "present"}
        graph["edges"] = [
            {"subj": name_to_id[s], "pred": p, "obj": name_to_id[o],
             "revision": rev}
            for s, p, o in edges
            if s in name_to_id and o in name_to_id]
    graph["updates"].append({k: (v if isinstance(v, (int, str)) else len(v))
                             for k, v in diff.items()})
    return diff


def to_neo4j_cypher(graph):
    """Emit MERGE statements mirroring the JSON graph (id is the merge key)."""
    out = [f"MATCH (n:TosmObject {{mapId: {json.dumps(graph['mapId'])}}}) "
           f"DETACH DELETE n;"]
    for n in graph["nodes"]:
        props = {
            "id": n["id"], "name": n["name"], "type": n["type"],
            "mapId": n["mapId"], "status": n["status"],
            "presence": n["presence"], "confidence": n["confidence"],
            "x": n["pose"]["x"], "y": n["pose"]["y"], "z": n["pose"]["z"],
            "theta": n["pose"]["theta"],
            "length": n["dimensions"]["length"],
            "width": n["dimensions"]["width"],
            "height": n["dimensions"]["height"],
            "color": n.get("color"),
            **{f"i_{k}": v for k, v in n.get("implicit", {}).items()},
        }
        props = {k: v for k, v in props.items() if v is not None}
        # Cypher map literals use bare identifiers as keys (JSON quoting is
        # a syntax error); values keep JSON encoding.
        body = ", ".join(f"{k}: {json.dumps(v, ensure_ascii=False)}"
                         for k, v in props.items())
        out.append(f"MERGE (n:TosmObject {{id: {json.dumps(n['id'])}}}) "
                   f"SET n = {{{body}}};")
    for e in graph["edges"]:
        out.append(
            f"MATCH (a:TosmObject {{id: {json.dumps(e['subj'])}}}), "
            f"(b:TosmObject {{id: {json.dumps(e['obj'])}}}) "
            f"MERGE (a)-[:{e['pred'].upper()}]->(b);")
    return "\n".join(out)
