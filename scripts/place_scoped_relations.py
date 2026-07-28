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
    args = ap.parse_args()

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

    edges = [e for e in g["edges"] if e.get("revision") == rev
             and e["subj"] in nodes and e["obj"] in nodes]
    intra, vertical, dropped = [], [], 0
    for e in edges:
        if e["pred"] in ("isOn", "isAboveOf"):
            vertical.append(e)
        elif member_place[e["subj"]] == member_place[e["obj"]]:
            intra.append(e)
        else:
            dropped += 1

    # key object per place
    key = {}
    for k, p in enumerate(places):
        cand = [n for nid, n in nodes.items() if member_place[nid] == k
                and str(n.get("status", "")).startswith("verified")]
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
           "edges_intra_nextTo": intra,
           "edges_vertical": vertical,
           "dropped_cross_place_nextTo": dropped,
           "place_adjacency": [[places[a]["name"], places[b]["name"]]
                               for a, b in sorted(adj)]}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}: intra {len(intra)} / vertical {len(vertical)} "
          f"/ dropped cross-place {dropped} / adjacency {len(adj)} / "
          f"key objects {key}")


if __name__ == "__main__":
    main()
