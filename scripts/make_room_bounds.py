#!/usr/bin/env python3
"""Room-bounds json (cells at cell_m) from a nav map's free space — the
generator that vis_n2_room_bounds.json never had. Keeps the largest connected
free region (the room the scanner stood in).

  .venv/bin/python scripts/make_room_bounds.py --map ~/ammr_twin/map_X.yaml --out outputs/X_room_bounds.json
"""
import argparse, json, os
import numpy as np, yaml
from PIL import Image
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("--map", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cell-m", type=float, default=0.25)
ap.add_argument("--pad", type=float, default=0.5, help="grow the room mask by this much [m]")
a = ap.parse_args()
y = yaml.safe_load(open(a.map))
img = np.asarray(Image.open(os.path.join(os.path.dirname(a.map), y["image"])).convert("L"))
res = float(y["resolution"]); ox, oy = float(y["origin"][0]), float(y["origin"][1]); H = img.shape[0]
free = img >= 250 if not y.get("negate", 0) else img <= 5
# close small gaps, then keep the largest connected free component
free = ndimage.binary_opening(free, iterations=1)
lab, n = ndimage.label(free)
if n == 0:
    raise SystemExit("no free space in map")
sizes = ndimage.sum(free, lab, range(1, n + 1)); keep = lab == (int(np.argmax(sizes)) + 1)
# objects standing in the room are holes in its free space: fill them, then
# grow by --pad so that wall-flush objects (centroid inside the wall band)
# still count as in-room
keep = ndimage.binary_fill_holes(keep)
pad_c = int(round(a.pad / res))
if pad_c > 0:
    keep = ndimage.binary_dilation(keep, iterations=pad_c)
rows, cols = np.nonzero(keep)
x = ox + (cols + 0.5) * res; yv = oy + (H - 1 - rows + 0.5) * res
cells = sorted({(int(np.floor(xx / a.cell_m)), int(np.floor(yy / a.cell_m))) for xx, yy in zip(x, yv)})
json.dump({"cell_m": a.cell_m, "source_map": os.path.abspath(a.map), "method": f"largest free component, holes filled, padded {a.pad} m",
           "cells": [list(c) for c in cells]}, open(a.out, "w"))
print(f"{len(cells)} cells @ {a.cell_m} m -> {a.out}")
