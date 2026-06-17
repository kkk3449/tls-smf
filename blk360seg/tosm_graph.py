"""Merge object / relation / place artifacts into one unified TOSM graph.

Combines the three deterministic outputs of the pipeline

  semanticObjects(.annotated).json            object nodes (symbolic+explicit+implicit)
  ...relations.json   (spatial_relations.py)  object-object edges (isNextTo/isOn/isAboveOf)
  ...places.json      (place_modeling.py)      place nodes + isConnectedTo + isInsideOf

into a single knowledge graph (nodes + one unified edge list of TOSM triples),
ready for visualization or a digital-twin (Isaac Sim) importer. Keeps the
VDA 5050-style header.

If no place artifact is given, a single enclosing place (the room envelope) is
synthesized from the object extent so isInsideOf is still populated — the common
case for an open single-room scan.
"""
import numpy as np

_OBJ_OBJ = {"isNextTo", "isOn", "isAboveOf", "isInFrontOf", "isBehindOf"}


def synthesize_enclosing_place(objects, name="showroom", margin=0.3):
    """One place = bounding envelope of all object poses (open single room)."""
    xs = [o["poseX"] for o in objects]
    ys = [o["poseY"] for o in objects]
    x0, x1 = min(xs) - margin, max(xs) + margin
    y0, y1 = min(ys) - margin, max(ys) + margin
    place = {
        "id": "place_1", "name": name, "type": "room",
        "centroid": [round(float(np.mean(xs)), 3), round(float(np.mean(ys)), 3)],
        "area_m2": round(float((x1 - x0) * (y1 - y0)), 2),
        "bbox": [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)],
        "polygon": [[round(x0, 3), round(y0, 3)], [round(x1, 3), round(y0, 3)],
                    [round(x1, 3), round(y1, 3)], [round(x0, 3), round(y1, 3)]],
        "synthesized": True,
    }
    inside = [{"subject": o["name"], "predicate": "isInsideOf", "object": "place_1"}
              for o in objects]
    return [place], inside


def merge(objects_msg, relations_msg=None, places_msg=None,
          synthesize_place=True):
    """Build the unified TOSM graph dict from the component messages."""
    objects = objects_msg["semanticObjects"]
    edges = []

    # object-object spatial relations
    if relations_msg:
        for r in relations_msg.get("objectRelations", []):
            edges.append(r)

    # places + place/object relations
    if places_msg and places_msg.get("places"):
        places = places_msg["places"]
        edges.extend(places_msg.get("placeRelations", []))     # isConnectedTo
        edges.extend(places_msg.get("objectPlaces", []))       # isInsideOf
    elif synthesize_place:
        places, inside = synthesize_enclosing_place(objects)
        edges.extend(inside)
    else:
        places = []

    edge_counts = {}
    for e in edges:
        edge_counts[e["predicate"]] = edge_counts.get(e["predicate"], 0) + 1

    return {
        "headerId": objects_msg.get("headerId", 0),
        "timestamp": objects_msg.get("timestamp"),
        "version": objects_msg.get("version"),
        "manufacturer": objects_msg.get("manufacturer", "CASELAB"),
        "serialNumber": objects_msg.get("serialNumber"),
        "coordinateFrame": "map",
        "graph": {
            "nodeCounts": {"objects": len(objects), "places": len(places)},
            "edgeCounts": edge_counts,
        },
        "objects": objects,
        "places": places,
        "relations": edges,    # unified TOSM triple list (all predicates)
    }
