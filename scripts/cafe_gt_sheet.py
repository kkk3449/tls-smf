#!/usr/bin/env python3
"""Contact sheet for owner GT confirmation of the 8F cafe run.

One row per object: two renders (base az chosen by widest xy face + its zoom)
plus Stage-A label, VLM verdict, tier, and vote trail. Split across pages.

  .venv/bin/python scripts/cafe_gt_sheet.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBJ = os.path.join(ROOT, "outputs", "cafe8f_objects")
VIEWS = os.path.join(ROOT, "outputs", "cafe8f_hiresviews", "views")
PER_PAGE = 7

doc = json.load(open(os.path.join(OBJ, "semanticObjects.lf_esc.json")))
objs = doc["semanticObjects"]
# LF renames objects after fusion; recover the render-dir name via id
orig = json.load(open(os.path.join(OBJ, "semanticObjects.json")))
by_id = {o["id"]: o["name"] for o in orig["semanticObjects"]}
for o in objs:
    o["properties"]["sourceName"] = by_id.get(o["id"], o["name"])

pages = [objs[i:i + PER_PAGE] for i in range(0, len(objs), PER_PAGE)]
for pi, page in enumerate(pages):
    fig, axes = plt.subplots(len(page), 3, figsize=(11, 3.1 * len(page)),
                             gridspec_kw={"width_ratios": [1, 1, 1.6]})
    if len(page) == 1:
        axes = [axes]
    for r, o in enumerate(page):
        name = o["properties"].get("sourceName", o["name"])
        vdir = os.path.join(VIEWS, name)
        base = os.path.join(vdir, "view_000.png")
        zoom = os.path.join(vdir, "view_090.png")
        for c, img in enumerate([base, zoom]):
            ax = axes[r][c]
            if os.path.exists(img):
                ax.imshow(mpimg.imread(img))
            ax.axis("off")
        p = o["properties"]
        d = o["dimensions"]
        txt = (f"[{pi * PER_PAGE + r}] {name}\n"
               f"VLM: {o['type']}  (conf {o.get('confidence', '?')})\n"
               f"status: {p.get('verificationStatus', '?')}"
               f"  esc: {p.get('escalated', False)}\n"
               f"size {d['length']:.2f} x {d['width']:.2f}"
               f" x {d['height']:.2f} m\n"
               f"pos ({o['poseX']:.1f}, {o['poseY']:.1f})\n"
               f"GT: ______________")
        ax = axes[r][2]
        ax.text(0.02, 0.95, txt, va="top", fontsize=11, family="monospace")
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join(OBJ, f"gt_sheet_{pi + 1}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("->", out)
