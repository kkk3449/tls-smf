#!/usr/bin/env python3
"""Render the unified TOSM graph as a spatially-grounded node-edge diagram.

Object nodes are placed at their real global pose (poseX, poseY) and the place
node at its centroid, so the knowledge graph is overlaid on the actual metric
layout (the advantage of a 3D model over the recognition-based 2D graphs of
prior TOSM work). Edges are colored per predicate.

  render_tosm_graph.py <tosm_graph.json> [out.png]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

EDGE_STYLE = {            # predicate -> (color, directed?, label)
    "isNextTo":     ("#7f7f7f", False, "isNextTo"),
    "isOn":         ("#1f77b4", True,  "isOn"),
    "isAboveOf":    ("#9467bd", True,  "isAboveOf"),
    "isInsideOf":   ("#cccccc", True,  "isInsideOf"),
    "isConnectedTo": ("#d62728", False, "isConnectedTo"),
}
# node fill per (coarse) object type
TYPE_COLOR = {
    "chair": "#2ca02c", "monitor": "#17becf", "TV": "#17becf",
    "Mobile Robot": "#ff7f0e", "Robot": "#ff7f0e", "machine": "#ff7f0e",
    "control panel": "#8c564b", "control_panel": "#8c564b",
}
DEFAULT_NODE = "#bbbbbb"


def main():
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else path.replace(".json", ".png")
    g = json.load(open(path))

    pos = {o["name"]: (o["poseX"], o["poseY"]) for o in g["objects"]}
    typ = {o["name"]: o.get("type", "") for o in g["objects"]}
    key = {o["name"]: o["properties"].get("isKeyObject", False)
           for o in g["objects"]}
    for p in g["places"]:
        pos[p["id"]] = tuple(p["centroid"])

    fig, ax = plt.subplots(figsize=(13, 9))

    # place envelope(s)
    for p in g["places"]:
        if p.get("bbox"):
            x0, y0, x1, y1 = p["bbox"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=True,
                                   fc="#f3f0e8", ec="#b0a080", lw=1.5,
                                   alpha=0.5, zorder=0))
            ax.text(x0 + 0.1, y1 - 0.1, f"{p['id']}: {p['name']}",
                    fontsize=11, fontweight="bold", color="#7a6a40",
                    va="top", zorder=6)

    # edges
    for e in g["relations"]:
        s, o = e["subject"], e["object"]
        if s not in pos or o not in pos:
            continue
        color, directed, _ = EDGE_STYLE.get(e["predicate"], ("#999999", False, ""))
        x0, y0 = pos[s]
        x1, y1 = pos[o]
        if e["predicate"] == "isInsideOf":
            # containment is already shown by the place envelope; keep these
            # edges very faint so the object-object relations stay legible.
            ax.plot([x0, x1], [y0, y1], "-", color=color, lw=0.3,
                    alpha=0.12, zorder=1)
        elif directed:
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), color=color,
                         lw=1.4, alpha=0.9, zorder=3,
                         arrowstyle="-|>", mutation_scale=12,
                         shrinkA=5, shrinkB=5))
        else:
            ax.plot([x0, x1], [y0, y1], "-", color=color, lw=1.0,
                    alpha=0.6, zorder=2)

    # object nodes
    for o in g["objects"]:
        x, y = pos[o["name"]]
        t = typ[o["name"]]
        c = next((v for k, v in TYPE_COLOR.items() if k in t), DEFAULT_NODE)
        ax.plot(x, y, "o", ms=12 if key[o["name"]] else 8, color=c,
                mec="k", mew=1.3 if key[o["name"]] else 0.5, zorder=4)
        ax.annotate(o["name"], (x, y), fontsize=6.0, zorder=5,
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    color="0.15")

    # place node marker
    for p in g["places"]:
        x, y = pos[p["id"]]
        ax.plot(x, y, "s", ms=16, color="#8c6d3f", mec="k", mew=1.5, zorder=5)

    # legends
    eleg = [Line2D([0], [0], color=c, lw=2,
                   label=lbl + (" →" if d else ""))
            for (c, d, lbl) in EDGE_STYLE.values()]
    nleg = [Line2D([0], [0], marker="o", ls="", mfc=c, mec="k", ms=9, label=k)
            for k, c in [("chair", "#2ca02c"), ("monitor/TV", "#17becf"),
                         ("robot/machine", "#ff7f0e"),
                         ("control panel", "#8c564b"), ("other", DEFAULT_NODE)]]
    nleg.append(Line2D([0], [0], marker="o", ls="", mfc="w", mec="k", mew=1.6,
                       ms=11, label="key object (bold)"))
    leg1 = ax.legend(handles=eleg, title="relations", loc="upper left",
                     fontsize=8, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=nleg, title="object type", loc="upper right",
              fontsize=8, framealpha=0.9)

    gc = g["graph"]
    ax.set_title("Unified TOSM knowledge graph "
                 f"({gc['nodeCounts']['objects']} objects, "
                 f"{gc['nodeCounts']['places']} place; "
                 f"{sum(gc['edgeCounts'].values())} relations) "
                 "— nodes at true global pose", fontsize=12)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
