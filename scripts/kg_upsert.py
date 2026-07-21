#!/usr/bin/env python3
"""Ingest a (re-)scan's Stage-B records into the TOSM knowledge graph.

First run creates the graph; later runs upsert into it (matched -> update,
missing -> absent, new -> insert) and print the diff summary. Edges
(geometric relations) are recomputed from the current present nodes.

  .venv/bin/python scripts/kg_upsert.py \
      --input outputs/vis_n2_det_run1/semanticObjects.lf_esc.json \
      --graph outputs/vis_n2_kg.json --map-id vis_n2 \
      --timestamp 2026-07-20T15:00:00 [--cypher outputs/vis_n2_kg.cypher]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from blk360seg import kg  # noqa: E402
from blk360seg.spatial_relations import compute_relations  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Stage-B semanticObjects json")
    ap.add_argument("--graph", required=True, help="graph json (created if absent)")
    ap.add_argument("--map-id", required=True)
    ap.add_argument("--timestamp", required=True,
                    help="ISO timestamp of the scan (explicit -> reproducible)")
    ap.add_argument("--cypher", default=None, help="also write Neo4j Cypher here")
    args = ap.parse_args()

    d = json.load(open(args.input))
    records = d["semanticObjects"] if isinstance(d, dict) and \
        "semanticObjects" in d else d

    graph = (json.load(open(args.graph)) if os.path.exists(args.graph)
             else kg.new_graph(args.map_id))

    rels = compute_relations(records)
    edges = [(r["subject"], r["predicate"], r["object"]) for r in rels]
    diff = kg.upsert(graph, records, args.map_id, args.timestamp, edges=edges)

    json.dump(graph, open(args.graph, "w"), indent=1, ensure_ascii=False)
    if args.cypher:
        open(args.cypher, "w").write(kg.to_neo4j_cypher(graph))

    ns = [n for n in graph["nodes"] if n["presence"] == "present"]
    by_status = {}
    for n in ns:
        by_status[n["status"]] = by_status.get(n["status"], 0) + 1
    print(json.dumps({
        "revision": diff["revision"],
        "updated": len(diff["updated"]), "unchanged": len(diff["unchanged"]),
        "moved": len(diff["moved"]), "inserted": len(diff["inserted"]),
        "absent": len(diff["absent"]),
        "present_nodes": len(ns), "edges": len(graph["edges"]),
        "status_counts": by_status}, indent=1))


if __name__ == "__main__":
    main()
