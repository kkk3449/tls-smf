#!/usr/bin/env python3
"""Merge objects + relations + places into one unified TOSM graph JSON.

  build_tosm_graph.py --objects semanticObjects.annotated.json
                      [--relations ...relations.json] [--places ...places.json]
                      [--out tosm_graph.json] [--no-synth-place]

Missing --relations/--places are simply omitted; with neither a place artifact
nor --no-synth-place, a single enclosing "showroom" place is synthesized so
isInsideOf is populated.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blk360seg import tosm_graph as tg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", required=True)
    ap.add_argument("--relations", default=None)
    ap.add_argument("--places", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-synth-place", action="store_true")
    args = ap.parse_args()

    objects_msg = json.load(open(args.objects))
    rel = json.load(open(args.relations)) if args.relations else None
    plc = json.load(open(args.places)) if args.places else None

    graph = tg.merge(objects_msg, rel, plc,
                     synthesize_place=not args.no_synth_place)

    out = args.out or args.objects.replace(".json", ".tosm_graph.json") \
        .replace(".annotated", "")
    with open(out, "w") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    g = graph["graph"]
    print(f"[TOSM] nodes: {g['nodeCounts']} | edges: {g['edgeCounts']}")
    print(f"[TOSM] wrote {out}")


if __name__ == "__main__":
    main()
