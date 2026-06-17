#!/usr/bin/env python3
"""Convert a unified TOSM graph JSON to a USD (.usda) scene for Isaac Sim.

  build_usd.py <tosm_graph.json> [out.usda]

Text-USD output (no pxr needed). Open the .usda in Isaac Sim or usdview.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blk360seg import usd_export as ux  # noqa: E402


def main():
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else path.replace(".json", ".usda")
    graph = json.load(open(path))
    usda = ux.to_usda(graph)
    with open(out, "w") as f:
        f.write(usda)
    nobj = len(graph["objects"])
    nplace = len(graph.get("places", []))
    nrel = len(graph.get("relations", []))
    print(f"[USD] {nobj} objects + {nplace} places + {nrel} relations -> {out}")


if __name__ == "__main__":
    main()
