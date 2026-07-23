#!/usr/bin/env python3
"""Name places from their ring codes (verified objects sorted by bearing).

The ring code is the place's object signature; a single LLM call per place
turns it into a functional name. Only verified-tier objects are in the ring,
so the confidence gate propagates into place semantics.

  .venv/bin/python scripts/place_ring_naming.py \
      --places outputs/place_layer_T3_places.json \
      --epoch T3 --out outputs/place_ring_naming.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from blk360seg.vlm_stage_b import SemanticVLM

TOOL = {
    "name": "name_place",
    "description": "Assign a short functional English name to a room section",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string",
                     "description": "snake_case functional name, e.g. seating_area"},
            "reason": {"type": "string"},
        },
        "required": ["name", "reason"],
    },
}

SYSTEM = (
    "You name sections of an industrial test room for a robot semantic map. "
    "You are given the section's ring code: its member objects (verified "
    "detections only) listed clockwise by bearing around the section centroid, "
    "with object types. Reply with a short snake_case functional name that a "
    "robot task planner could use. Prefer function over furniture inventory."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--places", required=True)
    ap.add_argument("--epoch", required=True, help="key in the output json, e.g. T3")
    ap.add_argument("--out", default="outputs/place_ring_naming.json")
    args = ap.parse_args()

    places = json.load(open(args.places))["semanticPlaces"]
    vlm = SemanticVLM(max_tokens=256)
    book = json.load(open(args.out)) if os.path.exists(args.out) else {}
    ep = book.setdefault(args.epoch, {})
    for p in places:
        ring = [r["type"] for r in p.get("ringCode", [])]
        if not ring:
            ep[p["name"]] = {"name": "unknown_area", "reason": "empty ring"}
            continue
        content = (f"Section '{p['name']}' ring code (clockwise): "
                   f"{', '.join(ring)}. Members total {p['memberCount']}, "
                   f"verified {p['verifiedMemberCount']}.")
        r = vlm._call(SYSTEM, content, TOOL)
        ep[p["name"]] = r
        print(f"{p['name']:<24} -> {r['name']}  ({r['reason'][:70]})")
    book.setdefault("_usage", {})[args.epoch] = vlm.usage_summary()
    json.dump(book, open(args.out, "w"), indent=1, ensure_ascii=False)
    print("cost:", round(vlm.cost_usd(), 4), "USD ->", args.out)


if __name__ == "__main__":
    main()
