#!/usr/bin/env python3
"""Build TOSM Places (room/corridor segmentation) + isConnectedTo / isInsideOf.

  build_places.py --map <map.pgm> --yaml <map.yaml>
                  [--objects semanticObjects.json] [--out places.json]
                  [--door-radius 0.6] [--min-room 2.0] [--render places.png]

Deterministic morphological room segmentation on the occupancy grid (no VLM).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blk360seg import place_modeling as pm  # noqa: E402


def render(labels, msg, out_png, origin, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, W = labels.shape
    ext = [origin[0], origin[0] + W * res, origin[1], origin[1] + H * res]
    fig, ax = plt.subplots(figsize=(9, 6))
    disp = np.ma.masked_where(labels == 0, labels)
    ax.imshow(disp, cmap="tab20", origin="lower", interpolation="nearest",
              extent=ext, alpha=0.85)
    for p in msg["places"]:
        cx, cy = p["centroid"]
        ax.text(cx, cy, p["id"].replace("place_", "P") + f"\n{p['type']}\n"
                f"{p['area_m2']:.0f} m²", ha="center", va="center",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
    for r in msg["placeRelations"]:
        dx, dy = r["doorway"]
        ax.plot(dx, dy, "kv", ms=11, mec="k", mfc="yellow", zorder=5)
        a = next(p for p in msg["places"] if p["id"] == r["subject"])["centroid"]
        b = next(p for p in msg["places"] if p["id"] == r["object"])["centroid"]
        ax.plot([a[0], dx, b[0]], [a[1], dy, b[1]], "k--", lw=1.0, alpha=0.6)
    for op in msg["objectPlaces"]:
        pass
    ax.set_title(f"TOSM places: {msg['placeCounts']['places']} places, "
                 f"{msg['placeCounts']['isConnectedTo']} isConnectedTo "
                 f"(▼ doorway), {msg['placeCounts']['isInsideOf']} isInsideOf")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print("[PLACE] wrote", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--objects", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--render", default=None)
    ap.add_argument("--door-radius", type=float, default=0.6)
    ap.add_argument("--min-room", type=float, default=2.0)
    ap.add_argument("--merge-width", type=float, default=2.0,
                    help="adjacent regions joined by an opening wider than this "
                         "(m) are merged into one place (else isConnectedTo)")
    args = ap.parse_args()

    objects = header = None
    if args.objects:
        m = json.load(open(args.objects))
        objects = m["semanticObjects"]
        header = m

    msg, labels = pm.build_place_message(
        args.map, args.yaml, objects=objects, header=header,
        door_radius_m=args.door_radius, min_room_m2=args.min_room,
        merge_width_m=args.merge_width)

    out = args.out or (args.objects or args.map).replace(
        ".json", ".places.json").replace(".pgm", ".places.json")
    with open(out, "w") as f:
        json.dump(msg, f, indent=2, ensure_ascii=False)
    print(f"[PLACE] {msg['placeCounts']} -> {out}")
    for p in msg["places"]:
        print(f"   {p['id']} {p['type']:<8} area={p['area_m2']:>6.1f} m^2 "
              f"centroid={p['centroid']}")
    for r in msg["placeRelations"]:
        print(f"   {r['subject']} <-isConnectedTo-> {r['object']} @ doorway {r['doorway']}")
    if args.render:
        _, res, origin = pm.load_occupancy(args.map, args.yaml)
        render(labels, msg, args.render, origin, res)


if __name__ == "__main__":
    main()
