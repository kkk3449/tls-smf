#!/usr/bin/env python3
"""Cafe place layer: synthesize a nav-style occupancy grid from the scan's
floor points (no SLAM run exists for the cafe), then reuse the exact SLIC
place pipeline of the main campaign.

  1. floor-band points -> 0.05 m grid, closing + hole fill -> free mask
  2. write cafe_map.pgm/.yaml + room-bounds json (0.25 m cells)
  3. place_slic_segment (n_segments=4) -> place_layer_CAFE_places.json
  4. merge ring-naming results into the places json (run naming separately)

  .venv/bin/python scripts/cafe_place_prep.py [--merge-names]
"""
import argparse
import json
import os

import numpy as np
import open3d as o3d
from scipy.ndimage import binary_closing, binary_fill_holes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
RES = 0.05


def build_map():
    pc = o3d.io.read_point_cloud(os.path.join(OUT, "cafe8f_scope",
                                              "cafe_fullres.ply"))
    xyz = np.asarray(pc.points)
    z0 = np.percentile(xyz[:, 2], 1)
    fl = xyz[(xyz[:, 2] > z0 - 0.05) & (xyz[:, 2] < z0 + 0.12)]
    ox, oy = fl[:, 0].min() - 0.1, fl[:, 1].min() - 0.1
    W = int((fl[:, 0].max() - ox) / RES) + 3
    H = int((fl[:, 1].max() - oy) / RES) + 3
    g = np.zeros((H, W), int)
    ci = ((fl[:, 0] - ox) / RES).astype(int)
    cj = ((fl[:, 1] - oy) / RES).astype(int)
    np.add.at(g, (cj, ci), 1)
    free = binary_fill_holes(binary_closing(g >= 2, iterations=3))
    img = np.where(free, 254, 205).astype(np.uint8)
    pgm = os.path.join(OUT, "cafe_map.pgm")
    with open(pgm, "wb") as f:
        f.write(f"P5\n{W} {H}\n255\n".encode())
        f.write(img[::-1].tobytes())          # pgm rows top-down
    yml = os.path.join(OUT, "cafe_map.yaml")
    open(yml, "w").write(
        f"image: cafe_map.pgm\nresolution: {RES}\n"
        f"origin: [{ox:.3f}, {oy:.3f}, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    cm = 0.25
    cells = set()
    jj, ii = np.nonzero(free)
    for x, y in zip(ox + (ii + 0.5) * RES, oy + (jj + 0.5) * RES):
        cells.add((int(round(x / cm)), int(round(y / cm))))
    jb = os.path.join(OUT, "cafe_room_bounds.json")
    json.dump({"cell_m": cm, "cells": sorted(cells)}, open(jb, "w"))
    print(f"map {W}x{H} free={int(free.sum())} cells={len(cells)}")
    print("->", yml, "\n->", jb)


def merge_names():
    places = json.load(open(os.path.join(
        OUT, "place_layer_CAFE_places.json")))
    naming = json.load(open(os.path.join(OUT, "place_ring_naming.json")))
    key = "CAFE"
    named = naming.get(key, naming) if isinstance(naming, dict) else naming
    by_id = {str(e.get("id")): e.get("name") for e in named} \
        if isinstance(named, list) else {}
    if not by_id and isinstance(named, dict):
        by_id = {str(k): v.get("name", v) if isinstance(v, dict) else v
                 for k, v in named.items()}
    for p in places["semanticPlaces"]:
        nm = by_id.get(str(p["id"]))
        if nm:
            p["name"] = f"{nm}_{int(p['id']):03d}"
    json.dump(places, open(os.path.join(
        OUT, "place_layer_CAFE_places.json"), "w"), indent=1)
    print("names merged:", [p["name"] for p in places["semanticPlaces"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-names", action="store_true")
    a = ap.parse_args()
    merge_names() if a.merge_names else build_map()
