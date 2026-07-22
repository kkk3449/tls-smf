#!/usr/bin/env python3
"""Minimal TOSM place layer on top of the object graph.

TOSM models object / place / robot. This paper focuses on the object layer;
the place layer is included minimally and *derived from data*: the room
boundary is the floor-connectivity mask used for room scoping (Sec. room
scope), the gateway is the audited door object. Adds to the graph json a
`places` section and writes a Cypher snippet that creates:

  (:TosmPlace {id, name, type:'room', mapId, area_m2})
  (obj)-[:isIn]->(place)          for every present object node
  (place)-[:connectsTo {via: door_name}]->(:TosmPlace corridor stub)

  .venv/bin/python scripts/add_place_layer.py \
      --graph outputs/vis_n2_kg.json --bounds outputs/vis_n2_room_bounds.json \
      --place-name testroom --gateway door_013 \
      --outside-name corridor --cypher outputs/vis_n2_place_layer.cypher
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--bounds", required=True)
    ap.add_argument("--place-name", default="testroom")
    ap.add_argument("--gateway", default=None,
                    help="object node name acting as the doorway")
    ap.add_argument("--outside-name", default="corridor")
    ap.add_argument("--cypher", required=True)
    args = ap.parse_args()

    g = json.load(open(args.graph))
    b = json.load(open(args.bounds))
    cells = np.array(b["cells"], dtype=float) * b["cell_m"]
    area = len(b["cells"]) * b["cell_m"] ** 2
    lo, hi = cells.min(0), cells.max(0)
    map_id = g["mapId"]
    place_id = f"place-{map_id}-{args.place_name}"
    outside_id = f"place-{map_id}-{args.outside_name}"

    g["places"] = [{
        "id": place_id, "name": args.place_name, "type": "room",
        "mapId": map_id, "area_m2": round(area, 1),
        "bboxMin": [round(float(lo[0]), 2), round(float(lo[1]), 2)],
        "bboxMax": [round(float(hi[0]), 2), round(float(hi[1]), 2)],
        "boundary": "floor-connectivity mask (vis_n2_room_bounds.json)",
        "gateway": args.gateway,
        "connectsTo": args.outside_name,
    }, {
        "id": outside_id, "name": args.outside_name, "type": "corridor",
        "mapId": map_id, "modeled": False,
        "note": "outside the modeling scope; stub for topology only",
    }]
    json.dump(g, open(args.graph, "w"), indent=1, ensure_ascii=False)

    lines = [
        f"MERGE (p:TosmPlace {{id: '{place_id}'}}) "
        f"SET p.name='{args.place_name}', p.type='room', p.mapId='{map_id}', "
        f"p.area_m2={round(area, 1)};",
        f"MERGE (c:TosmPlace {{id: '{outside_id}'}}) "
        f"SET c.name='{args.outside_name}', c.type='corridor', "
        f"c.mapId='{map_id}', c.modeled=false;",
        f"MATCH (o:TosmObject {{mapId: '{map_id}'}}), "
        f"(p:TosmPlace {{id: '{place_id}'}}) "
        f"WHERE o.presence <> 'absent' OR o.presence IS NULL "
        f"MERGE (o)-[:isIn]->(p);",
    ]
    if args.gateway:
        lines.append(
            f"MATCH (p:TosmPlace {{id: '{place_id}'}}), "
            f"(c:TosmPlace {{id: '{outside_id}'}}) "
            f"MERGE (p)-[:connectsTo {{via: '{args.gateway}'}}]->(c);")
    open(args.cypher, "w").write("\n".join(lines) + "\n")
    n_obj = sum(1 for n in g["nodes"] if n.get("presence") != "absent")
    print(f"place layer: {args.place_name} ({area:.1f} m2), {n_obj} isIn "
          f"edges, gateway={args.gateway} -> {args.outside_name}")
    print(f"cypher -> {args.cypher}")


if __name__ == "__main__":
    main()
