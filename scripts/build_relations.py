#!/usr/bin/env python3
"""Build the TOSM object-relation knowledge graph from semantic-object poses.

  build_relations.py --input <semanticObjects(.annotated).json> [--out ...]
                     [--orientation] [--next-to-gap 0.4] [--on-tol 0.15]

Deterministic geometry only (no VLM). Writes a JSON of (subject, predicate,
object) triples that a digital twin / knowledge graph (Isaac Sim) can consume.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blk360seg import spatial_relations as sr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="semanticObjects(.annotated).json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--next-to-gap", type=float, default=0.40)
    ap.add_argument("--on-tol", type=float, default=0.15)
    ap.add_argument("--above-gap", type=float, default=0.10)
    ap.add_argument("--orientation", action="store_true",
                    help="also emit isInFrontOf/isBehindOf (low-confidence, "
                         "theta facing-sign ambiguous)")
    args = ap.parse_args()

    msg = json.load(open(args.input))
    objs = msg["semanticObjects"]
    out_msg = sr.build_relation_message(
        msg, objs, next_to_gap=args.next_to_gap, on_tol=args.on_tol,
        above_gap=args.above_gap, orientation=args.orientation)

    out = args.out or args.input.replace(".json", ".relations.json")
    with open(out, "w") as f:
        json.dump(out_msg, f, indent=2, ensure_ascii=False)
    c = out_msg["relationCounts"]
    print(f"[REL] {len(objs)} objects -> {len(out_msg['objectRelations'])} relations "
          f"{c}")
    print(f"[REL] wrote {out}")


if __name__ == "__main__":
    main()
