#!/usr/bin/env python3
"""Place-scoped TOSM relation layer.

Object-object proximity is only meaningful within a place: isNextTo edges
are kept when both endpoints share a place; vertical relations (isOn /
isAboveOf) are kept regardless. Each place designates a key object (VLM
implicit isKeyObject among verified members, else its most confident
specific-type verified member), and places themselves carry adjacency
edges derived from shared region boundaries.

  .venv/bin/python scripts/place_scoped_relations.py \
      --kg outputs/testroom_epochs_kg.json \
      --places outputs/place_layer_T3_slic.json \
      --out outputs/t3_place_scoped_relations.json
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--places", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refuted", default="outputs/owner_refuted.json",
                    help="owner ground-truth refutations; excluded from "
                         "key-object designation (nodes stay in the graph)")
    args = ap.parse_args()

    refuted = set()
    if args.refuted and __import__("os").path.exists(args.refuted):
        refuted = {r["name"]
                   for r in json.load(open(args.refuted))["refuted"]}

    g = json.load(open(args.kg))
    rev = g["revision"]
    nodes = {n["id"]: n for n in g["nodes"] if n.get("presence") != "absent"}
    d = json.load(open(args.places))
    places = d["semanticPlaces"]
    cm = d["cell_m"]

    # rasterize region cells once for membership + adjacency
    allc = np.concatenate([np.array(p["cells"]) for p in places])
    xmin, ymin = allc.min(0)
    idx = {}
    for k, p in enumerate(places):
        for x, y in p["cells"]:
            idx[(int(round((x - xmin) / cm)), int(round((y - ymin) / cm)))] = k

    def place_of(x, y):
        ci = int(round((x - xmin) / cm))
        cj = int(round((y - ymin) / cm))
        if (ci, cj) in idx:
            return idx[(ci, cj)]
        keys = np.array(list(idx.keys()))
        k = np.argmin((keys[:, 0] - ci) ** 2 + (keys[:, 1] - cj) ** 2)
        return idx[tuple(keys[k])]

    member_place = {nid: place_of(n["pose"]["x"], n["pose"]["y"])
                    for nid, n in nodes.items()}

    def obb(n):
        import math
        cx, cy = n["pose"]["x"], n["pose"]["y"]
        th = n["pose"].get("theta", 0.0)
        hl, hw = n["dimensions"]["length"] / 2, n["dimensions"]["width"] / 2
        c, si = math.cos(th), math.sin(th)
        return [(cx + sx * hl * c - sy * hw * si,
                 cy + sx * hl * si + sy * hw * c)
                for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]

    def footprints_overlap(a, b, margin=0.1):
        # separating-axis test on the two yaw-rotated footprints
        import math
        pa, pb = obb(a), obb(b)
        for poly in (pa, pb):
            for i in range(4):
                ex = poly[(i + 1) % 4][0] - poly[i][0]
                ey = poly[(i + 1) % 4][1] - poly[i][1]
                ax_, ay_ = -ey, ex
                L = math.hypot(ax_, ay_) or 1.0
                ax_, ay_ = ax_ / L, ay_ / L
                p1 = [ax_ * x + ay_ * y for x, y in pa]
                p2 = [ax_ * x + ay_ * y for x, y in pb]
                if min(p1) > max(p2) + margin or min(p2) > max(p1) + margin:
                    return False
        return True

    edges = [e for e in g["edges"] if e.get("revision") == rev
             and e["subj"] in nodes and e["obj"] in nodes]
    intra, vertical, dropped, dropped_vert = [], [], 0, []
    for e in edges:
        if e["pred"] in ("isOn", "isAboveOf"):
            # vertical relations require the two footprints to actually
            # overlap; oversized merged clusters otherwise produce spurious
            # long-range stacked relations
            if footprints_overlap(nodes[e["subj"]], nodes[e["obj"]]):
                vertical.append(e)
            else:
                dropped_vert.append((nodes[e["subj"]]["name"],
                                     e["pred"], nodes[e["obj"]]["name"]))
        elif member_place[e["subj"]] == member_place[e["obj"]]:
            intra.append(e)
        else:
            dropped += 1

    # key object per place
    key = {}
    for k, p in enumerate(places):
        cand = [n for nid, n in nodes.items() if member_place[nid] == k
                and str(n.get("status", "")).startswith("verified")
                and n["name"] not in refuted]
        flagged = [n for n in cand if n.get("implicit", {}).get("isKeyObject")]
        pool = flagged or [n for n in cand
                           if n.get("type") not in ("clutter", "unknown")] or cand
        if pool:
            key[p["name"]] = max(pool, key=lambda n: n.get("confidence", 0))["name"]

    # place adjacency from shared 4-neighbor boundaries
    adj = set()
    for (ci, cj), k in idx.items():
        for dci, dcj in ((1, 0), (0, 1)):
            k2 = idx.get((ci + dci, cj + dcj))
            if k2 is not None and k2 != k:
                adj.add(tuple(sorted((k, k2))))

    out = {"revision": rev,
           "members": {places[k]["name"]: [nodes[nid]["name"] for nid, pk
                                           in member_place.items() if pk == k]
                       for k in range(len(places))},
           "keyObjects": key,
           "ownerRefuted": sorted(refuted),
           "edges_intra_nextTo": intra,
           "edges_vertical": vertical,
           "dropped_cross_place_nextTo": dropped,
           "dropped_vertical_no_overlap": dropped_vert,
           "place_adjacency": [[places[a]["name"], places[b]["name"]]
                               for a, b in sorted(adj)]}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}: intra {len(intra)} / vertical {len(vertical)} "
          f"/ dropped cross-place {dropped} / dropped vertical {dropped_vert} "
          f"/ adjacency {len(adj)} / "
          f"key objects {key}")


if __name__ == "__main__":
    main()
