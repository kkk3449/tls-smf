#!/usr/bin/env python3
"""fig17: cafeteria stress scene composite (paper Sec. 5.8).

(a) instance top-down with error-structure annotations
(b) same-cluster render pairs: 3cm-voxel vs full-resolution source
(c) model x render-source grouped bars (escalated accuracy, unanimous prec.)

  .venv/bin/python scripts/cafe_fig.py

Reads esc_analysis_cafe8f*.json for panel (c); copies the png to the
manuscript figs dir when MANUSCRIPT_FIGS is set (default ws path).
"""
import glob
import json
import os

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "paper_figs")
FIGS = os.environ.get(
    "MANUSCRIPT_FIGS",
    "/home/caselab/blk360_ros2_ws/docs/electronics2026/figs")

C46 = "#2b6cb0"   # Sonnet 4-6 (paper deterministic blue)
CS5 = "#dd6b20"   # Sonnet 5   (paper stochastic orange)
BAD = "#c53030"
OK = "#2f855a"

SOFA = {3, 7, 9, 10, 11, 12, 13}          # sofa fragments + merges
NOISE = {2, 5, 6, 8, 14, 15, 25}          # carpet floor-noise clusters
LABELS = {26: "door", 24: "glass door +\nfire ext.", 1: "vending row",
          0: "ad panel", 17: "chair", 19: "chair", 18: "chair",
          22: "planter", 4: "planter", 23: "table+chairs"}


def summary(name):
    s = json.load(open(os.path.join(ROOT, "outputs", name)))["summary"]
    return (100 * s["esc_acc_strict"][0], 100 * s["unanimity"]["acc"])


def main():
    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(13.8, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 0.95, 0.9],
                          wspace=0.12,
                          left=0.015, right=0.99, top=0.90, bottom=0.09)

    # ---------- (a) instance map ----------
    ax = fig.add_subplot(gs[0, 0])
    cmap = plt.get_cmap("tab20")
    for f in sorted(glob.glob(os.path.join(
            ROOT, "outputs", "cafe8f_objects", "obj_*.ply"))):
        i = int(f.split("_")[-1][:4])
        p = np.asarray(o3d.io.read_point_cloud(f).points)
        col = cmap(i % 20)
        ax.scatter(p[:, 0], p[:, 1], s=0.5, color=col, linewidths=0,
                   rasterized=True)
        cx, cy = p[:, 0].mean(), p[:, 1].mean()
        if i in SOFA:
            ax.annotate("", (cx, cy), (cx, cy))  # anchor
            ax.scatter([cx], [cy], marker="x", s=46, color=BAD, lw=1.6,
                       zorder=5)
        if i in LABELS:
            ax.annotate(LABELS[i], (cx, cy), textcoords="offset points",
                        xytext=(0, 7), fontsize=7.6, ha="center",
                        color="#222222", zorder=6)
    for i in NOISE:                     # small dot marker for carpet noise
        f = os.path.join(ROOT, "outputs", "cafe8f_objects",
                         f"obj_{i:04d}.ply")
        p = np.asarray(o3d.io.read_point_cloud(f).points)
        ax.scatter([p[:, 0].mean()], [p[:, 1].mean()], marker="o",
                   facecolors="none", edgecolors="#777777", s=52, lw=1.1,
                   zorder=5)
    ax.scatter([], [], marker="x", s=46, color=BAD, lw=1.6,
               label="sofa fragment / merge (7, all absorbed)")
    ax.scatter([], [], marker="o", facecolors="none", edgecolors="#777777",
               s=52, lw=1.1, label="carpet floor-noise cluster (7)")
    ax.legend(loc="upper left", fontsize=8.4, frameon=False)
    ax.set_ylim(-4.4, 4.8)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("(a) cafeteria clusters (28) after yaw alignment and "
                 "room crop", fontsize=10.5, loc="left")

    # ---------- (b) render-source pairs ----------
    gsb = gs[0, 1].subgridspec(2, 2, hspace=0.05, wspace=0.02)
    pairs = [("clutter_019", "chair\n(recovered)"),
             ("clutter_013", "sofa fragment\n(absorbed in both)")]
    srcs = [("cafe8f_hdviews", "3 cm voxel source"),
            ("cafe8f_hiresviews512", "full-resolution source")]
    for r, (name, rowlab) in enumerate(pairs):
        for c, (d, collab) in enumerate(srcs):
            axi = fig.add_subplot(gsb[r, c])
            img = os.path.join(ROOT, "outputs", d, "views", name,
                               "view_000.png")
            im = mpimg.imread(img)
            k = im.shape[0] // 512
            axi.imshow(im[110 * k:405 * k, 110 * k:405 * k])
            axi.set_xticks([]); axi.set_yticks([])
            for s in axi.spines.values():
                s.set_color("#cccccc")
            if r == 0:
                axi.set_title(collab, fontsize=9)
            if c == 0:
                axi.set_ylabel(rowlab, fontsize=8.6)
    fig.text(gs[0, 1].get_position(fig).x0 - 0.005,
             0.955, "(b) render source only", fontsize=10.5)

    # ---------- (c) 2x2 grouped bars ----------
    axc = fig.add_subplot(gs[0, 2])
    acc46 = [summary("esc_analysis_cafe8f.json"),
             summary("esc_analysis_cafe8f_hires.json")]
    accs5 = [summary("esc_analysis_cafe8f_s5.json"),
             summary("esc_analysis_cafe8f_hires_s5.json")]
    groups = ["escalated\naccuracy", "unanimous\nprecision"]
    cells = [("4-6 / voxel", acc46[0], C46, 0.55),
             ("4-6 / full-res", acc46[1], C46, 1.0),
             ("S5 / voxel", accs5[0], CS5, 0.55),
             ("S5 / full-res", accs5[1], CS5, 1.0)]
    for gi, gname in enumerate(groups):
        for bi, (name, vals, col, alpha) in enumerate(cells):
            x = gi * 1.15 + bi * 0.2
            v = vals[gi]
            axc.bar(x, v, width=0.16, color=col, alpha=alpha)
            axc.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.6,
                     color="#333333")
    axc.set_xticks([0.3 + i * 1.15 for i in range(2)])
    axc.set_xticklabels(groups, fontsize=9)
    axc.set_ylim(0, 100)
    axc.set_ylabel("% (confirmed GT)", fontsize=9)
    axc.spines[["top", "right"]].set_visible(False)
    axc.grid(axis="y", lw=0.3, alpha=0.4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C46, alpha=0.55),
               plt.Rectangle((0, 0), 1, 1, color=C46),
               plt.Rectangle((0, 0), 1, 1, color=CS5, alpha=0.55),
               plt.Rectangle((0, 0), 1, 1, color=CS5)]
    axc.legend(handles, ["4-6 / voxel", "4-6 / full-res",
                         "S5 / voxel", "S5 / full-res"],
               fontsize=7.8, frameon=False, loc="upper left")
    axc.set_title("(c) verifier $\\times$ render source", fontsize=10.5,
                  loc="left")

    for path in (os.path.join(OUT, "fig17_cafe_stress.png"),
                 os.path.join(FIGS, "fig17_cafe_stress.png")):
        fig.savefig(path, dpi=200)
        print("->", path)


if __name__ == "__main__":
    main()
