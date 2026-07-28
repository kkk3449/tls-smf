#!/usr/bin/env python3
"""Publication-draft figures for the Electronics paper (outline r3 plan).

  .venv/bin/python scripts/make_paper_figs.py [--only fig6,fig7] \
      [--out outputs/paper_figs]

Fig 1  framework architecture (deterministic vs stochastic, loops)
Fig 2  geometric pipeline stages + ceiling-soffit remnant inset
Fig 3  multi-view verification + escalation views + vote table
Fig 4  verified vs unverified object record cards
Fig 5  identity matching / upsert state diagram
Fig 6  escalation accuracy-cost points (both scenes)
Fig 7  4-condition query benchmark (accuracy, hallucination, traps)
Fig 8  incremental update before/after map (vis_n2 room)
Fig 10 error-case taxonomy panels
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
O = lambda *p: os.path.join(ROOT, "outputs", *p)

DET = "#2b6cb0"     # deterministic
STO = "#dd6b20"     # stochastic (VLM/Uni3D)
OK = "#2f855a"
BAD = "#c53030"


def _save(fig, out, name):
    p = os.path.join(out, name)
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ------------------------------------------------------------------ fig 1 ---
def fig1(out):
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                    fc="white", ec=color, lw=1.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color="black")

    def arrow(x1, y1, x2, y2, color="black", style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=13, color=color,
                                     lw=1.4, linestyle=ls))

    y, h = 0.52, 0.30
    box(0.010, y, 0.105, h, "Registered\nTLS scan\n(E57)", "gray")
    box(0.135, y, 0.140, h, "Preprocessing\nvoxel + seeded\nRANSAC planes", DET)
    box(0.295, y, 0.150, h, "Object decomposition\nDBSCAN + giant-split\n+ soffit-remnant filter", DET)
    box(0.465, y, 0.140, h, "Explicit modeling\npose / extents / color\n+ Uni3D provisional", DET)
    ax.text(0.535, y - 0.045, "provisional label: stochastic", ha="center",
            fontsize=7.6, color=STO)
    box(0.625, y, 0.150, h, "Multi-view VLM\nverification\n4-view late fusion", STO)
    box(0.795, y, 0.095, h, "Confidence-\ngated upsert\n(mint-once ID)", DET)
    box(0.905, y, 0.088, h, "Queryable\nTOSM KG\n(Neo4j)", "gray")

    for x1, x2 in [(0.115, 0.135), (0.275, 0.295), (0.445, 0.465),
                   (0.605, 0.625), (0.775, 0.795), (0.890, 0.905)]:
        arrow(x1, y + h / 2, x2, y + h / 2)

    # escalation loop
    box(0.625, 0.08, 0.150, 0.22,
        "Automatic escalation\n8-view + zoom re-query\nsplit vote only", STO)
    arrow(0.680, y, 0.680, 0.30, STO)
    arrow(0.740, 0.30, 0.740, y, STO)
    ax.text(0.700, 0.035, "unresolved → ingested as 'unverified' "
            "(confidence exposed)", fontsize=8, color=STO)

    # update loop
    arrow(0.950, y + h, 0.950, 0.955, DET)
    arrow(0.950, 0.955, 0.070, 0.955, DET)
    arrow(0.070, 0.955, 0.070, y + h, DET)
    ax.text(0.50, 0.975, "incremental update: re-scan → identical backbone → "
            "match (mint-once IDs) → upsert diff (unchanged / updated / moved / "
            "inserted / absent)", ha="center", fontsize=8.6, color=DET)
    # legend
    ax.plot([], [], color=DET, lw=3, label="deterministic (seeded, byte-identical re-runs)")
    ax.plot([], [], color=STO, lw=3, label="stochastic (controlled by voting + gating)")
    ax.legend(loc="lower left", fontsize=8.4, frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _save(fig, out, "fig1_architecture.png")


# ------------------------------------------------------------------ fig 2 ---
def fig2(out):
    import open3d as o3d
    from PIL import Image
    stages = [("fig6_raw.png", "(a) registered TLS scan"),
              ("fig6_nowall.png", "(b) structure removed"),
              ("fig6_seg.png", "(c) object instances"),
              ("fig6_tosm.png", "(d) TOSM knowledge graph")]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.15]})
    for ax, (f, t) in zip(axes, stages):
        ax.imshow(Image.open(O(f)))
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    # soffit inset: front view of showroom clean cloud, strips red
    ax = axes[4]
    pc = o3d.io.read_point_cloud(O("showroom_det", "clean.ply"))
    p = np.asarray(pc.points)
    rng = np.random.default_rng(0)
    s = rng.choice(len(p), 90000, replace=False)
    ax.scatter(p[s, 0], p[s, 2], s=0.25, c="lightgray", linewidths=0)
    for i in (27, 28, 29, 30, 31, 41, 43):
        q = np.asarray(o3d.io.read_point_cloud(
            O("showroom_det", f"obj_{i:04d}.ply")).points)
        ax.scatter(q[:, 0], q[:, 2], s=1.2, c=BAD, linewidths=0)
    ax.set_title("(e) ceiling-soffit remnants\n(filtered at Stage A)",
                 fontsize=10, color=BAD)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    _save(fig, out, "fig2_pipeline.png")


# ------------------------------------------------------------------ fig 3 ---
def fig3(out):
    from PIL import Image
    name = "clutter_040"
    d = json.load(open(O("stage2_no_wall_objects_split",
                         "semanticObjects.lf_esc.filtered.json")))
    rec = [o for o in d["semanticObjects"] if o["id"] == "40"][0]
    votes = rec["properties"]["viewVotes"]
    vd = O("s2_split_hdviews", "views", name)
    base = [(az, f"view_{az:03d}.png") for az in (0, 90, 180, 270)]
    esc = [(az, f"view_{az:03d}.png") for az in (45, 135, 225, 315)]
    zoom = [(az, f"view_{az:03d}_z.png") for az in (0, 90, 180, 270)]
    fig, axes = plt.subplots(3, 4, figsize=(10.5, 8.2))
    rows = [("base 4-view", base), ("escalation +4 diagonal", esc),
            ("escalation +4 zoom", zoom)]
    for r, (label, items) in enumerate(rows):
        for c, (az, f) in enumerate(items):
            ax = axes[r][c]
            fp = os.path.join(vd, f)
            if os.path.exists(fp):
                ax.imshow(Image.open(fp))
            v = [v for v in votes
                 if v["azimuth"] == az and bool(v.get("zoom")) == ("_z" in f)]
            if v:
                v = v[0]
                ax.set_title(f"az {az}°: {v['type']} ({v['confidence']:.2f})",
                             fontsize=8.4)
            ax.axis("off")
            if c == 0:
                ax.text(-0.12, 0.5, label, rotation=90, transform=ax.transAxes,
                        va="center", fontsize=10)
    fig.suptitle(
        f"{rec['name']}  —  split base vote → automatic escalation → "
        f"fused '{rec['type']}' (share {rec['properties']['voteShare']}, "
        f"{rec['properties']['verificationStatus']})", fontsize=11)
    _save(fig, out, "fig3_multiview_escalation.png")


# ------------------------------------------------------------------ fig 4 ---
def fig4(out):
    """TOSM record cards: render + Symbolic / Explicit / Implicit panels."""
    from PIL import Image
    d = json.load(open(O("stage2_no_wall_objects_split",
                         "semanticObjects.lf_esc.filtered.json")))
    by = {o["id"]: o for o in d["semanticObjects"]}

    def fields(o):
        pr = o["properties"]
        ver = pr.get("verificationStatus", "?")
        sym = [("type", o["type"] if ver != "unverified"
                else f"unknown (gated; was '{o['type']}')"),
               ("name", o["name"]),
               ("verified", {"verified": "true (VLM 4-view unanimous)",
                             "verified_escalated": "true (escalated 12-view)",
                             "unverified": "false → demoted at query time"}
                .get(ver, ver)),
               ("voteShare", str(pr.get("voteShare", "—")))]
        dm = o["dimensions"]
        exp = [("pose", f"x {o['poseX']:.2f}  y {o['poseY']:.2f}  z "
                        f"{pr.get('poseZ', 0.0):.2f} m  (map)"),
               ("theta", f"{o.get('poseTheta', 0.0):.2f} rad"),
               ("dims", f"{dm['length']:.2f} x {dm['width']:.2f} x "
                        f"{dm['height']:.2f} m"),
               ("color", str(o.get("color", "—"))),
               ("conf.", f"{o.get('confidence', 0):.2f}")]
        imp = [("keyObject", str(pr.get("isKeyObject", False)).lower()),
               ("movable", str(pr.get("isMovable", False)).lower())]
        return [("Symbolic", sym), ("Explicit", exp), ("Implicit", imp)]

    cards = [("7", "control panel_007", "(a) verified object record", OK),
             ("46", "ladder_046", "(b) unverified record (gated)", BAD)]
    fig = plt.figure(figsize=(11.6, 4.6))
    for ci, (oid, vdir, title, col) in enumerate(cards):
        o = by[oid]
        axi = fig.add_axes([0.015 + ci * 0.5, 0.06, 0.185, 0.78])
        fp = O("s2_split_hdviews", "views", vdir, "view_000.png")
        if os.path.exists(fp):
            axi.imshow(Image.open(fp))
        axi.axis("off")
        axi.set_title(title, fontsize=10.5, color=col, loc="left")
        axt = fig.add_axes([0.215 + ci * 0.5, 0.02, 0.27, 0.88])
        axt.axis("off")
        y = 0.98
        for section, rows in fields(o):
            axt.text(0.0, y, section, fontsize=10, color="#1a56a0",
                     style="italic", weight="bold", va="top")
            y -= 0.085
            for k, v in rows:
                axt.text(0.06, y, k, fontsize=8.6, color="#333333",
                         family="monospace", va="top")
                axt.text(0.34, y, v, fontsize=8.6, color="#111111",
                         family="monospace", va="top", wrap=True)
                y -= 0.072
            y -= 0.030
    _save(fig, out, "fig4_record_cards.png")


# ------------------------------------------------------------------ fig 5 ---
def fig5(out):
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                    fc="white", ec=color, lw=1.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, text="", color="black"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, color=color, lw=1.3))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.03, text, ha="center",
                    fontsize=7.8, color=color)

    box(0.01, 0.40, 0.16, 0.24, "re-scan object\n(deterministic\nbackbone output)", DET)
    box(0.24, 0.40, 0.20, 0.24, "match vs existing nodes\npass 1: spatial\n"
        "(center 0.5 m, extents 30%)", "black")
    box(0.24, 0.05, 0.20, 0.24, "pass 2: moved?\nsame type+extents\nwithin 6 m", "black")
    box(0.52, 0.70, 0.17, 0.22, "UNCHANGED /\nUPDATED\nkeep node ID", OK)
    box(0.52, 0.40, 0.17, 0.22, "MOVED\nkeep node ID\nupdate pose", OK)
    box(0.52, 0.10, 0.17, 0.22, "INSERTED\nmint new\nimmutable ID", DET)
    box(0.76, 0.40, 0.21, 0.24, "nodes not seen\nin this re-scan\n→ ABSENT "
        "(kept with\nhistory, revivable)", BAD)
    arrow(0.17, 0.52, 0.24, 0.52)
    arrow(0.44, 0.58, 0.52, 0.79, "matched")
    arrow(0.44, 0.46, 0.44, 0.29)
    arrow(0.44, 0.17, 0.52, 0.19, "no match")
    arrow(0.44, 0.24, 0.52, 0.49, "type+dims match")
    ax.text(0.50, 0.985, "mint-once identity: the ID is issued at first "
            "insertion (birth-anchor fingerprint) and never changes; "
            "matching, not the hash, carries identity across scans",
            ha="center", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _save(fig, out, "fig5_identity_upsert.png")


# ------------------------------------------------------------------ fig 6 ---
def fig6(out):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for f, scene, color, n_gt in [
            (O("esc_analysis_showroom.json"), "Showroom", DET, None),
            (O("esc_analysis_vis_n2.json"), "Robot hall", STO, None)]:
        s = json.load(open(f))["summary"]
        n = s["objects_gt"]
        base_calls = 5 * n                    # 4 views + 1 implicit per object
        esc_calls = base_calls + s["escalation_calls"]
        acc_b = s["base4_acc_strict"][0] * 100
        acc_e = s["esc_acc_strict"][0] * 100
        ax.plot([base_calls / n, esc_calls / n], [acc_b, acc_e], "-o",
                color=color, label=f"{scene} (n={s['base4_acc_strict'][1]} confirmed GT)")
        ax.annotate("4-view", (base_calls / n, acc_b), textcoords="offset points",
                    xytext=(4, -12), fontsize=8.5, color=color)
        ax.annotate(f"+escalation ({s['trigger_rate']*100:.0f}% trigger)",
                    (esc_calls / n, acc_e), textcoords="offset points",
                    xytext=(-8, 8), fontsize=8.5, color=color, ha="right")
    ax.set_xlabel("VLM calls per object")
    ax.set_ylabel("top-1 label accuracy [%] (confirmed GT)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.6)
    _save(fig, out, "fig6_escalation_cost.png")


# ------------------------------------------------------------------ fig 7 ---
def fig7(out):
    conds = ["llm_only", "ungated", "verified_only", "gated"]
    labels = ["LLM-only", "ungated KG", "verified-only KG", "gated KG"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, f, scene in [(axes[0], O("query_bench_showroom_v5.json"), "Showroom"),
                         (axes[1], O("query_bench_vis_n2.json"), "Robot hall")]:
        s = json.load(open(f))["summary"]
        acc = [s[c]["accuracy"] * 100 for c in conds]
        hal = [s[c]["hallucination_rate"] * 100 for c in conds]
        trap = [s[c]["per_type"]["trap"] for c in conds]
        x = np.arange(len(conds))
        ax.bar(x - 0.19, acc, 0.36, color=DET, label="accuracy")
        ax.bar(x + 0.19, hal, 0.36, color=BAD, label="hallucination rate")
        for i, t in enumerate(trap):
            ax.text(i, max(acc[i], hal[i]) + 2.5,
                    f"traps {t['correct']}/{t['n']}", ha="center", fontsize=8.4)
        ax.set_xticks(x, labels, fontsize=8.6)
        ax.set_ylim(0, 100)
        ax.set_ylabel("[%]")
        ax.set_title(f"{scene} (n={s[conds[0]]['n']} queries)", fontsize=10.5)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8.4)
    _save(fig, out, "fig7_query_benchmark.png")


# ------------------------------------------------------------------ fig 8 ---
def _room_outline(ax, b, color="#555555", lw=1.4, tag=None):
    """Boundary contour of the room-mask cell grid (the test-room wall line)."""
    cells = np.array(b["cells"], dtype=int)
    cm = b["cell_m"]
    x0, y0 = cells[:, 0].min() - 2, cells[:, 1].min() - 2
    gx, gy = cells[:, 0] - x0, cells[:, 1] - y0
    grid = np.zeros((gy.max() + 3, gx.max() + 3))
    grid[gy, gx] = 1
    X = (np.arange(grid.shape[1]) + x0) * cm
    Y = (np.arange(grid.shape[0]) + y0) * cm
    ax.contour(*np.meshgrid(X, Y), grid, levels=[0.5], colors=color,
               linewidths=lw, zorder=2)
    if tag:
        ax.text(X[0] + 0.3, Y.max() - 0.2, tag, fontsize=7.2, color=color,
                style="italic", va="top")


def _rbox(ax, x, y, L, W, th, ec, lw=1.0, ls="-", fc="none", z=5, alpha=1.0):
    """Object footprint rectangle rotated by the explicit model's yaw."""
    import math
    c, s = math.cos(th), math.sin(th)
    cx = x + (-L / 2) * c - (-W / 2) * s
    cy = y + (-L / 2) * s + (-W / 2) * c
    ax.add_patch(Rectangle((cx, cy), L, W, angle=math.degrees(th),
                           facecolor=fc, edgecolor=ec, lw=lw, linestyle=ls,
                           zorder=z, alpha=alpha))


def _seg_points(objs, objdir, max_pts=320, T=None):
    """name -> downsampled xy points of the segmented cluster (top-down)."""
    import open3d as o3d
    rng = np.random.default_rng(0)
    pts = {}
    for o in objs:
        try:
            idx = int(o["name"].rsplit("_", 1)[1])
        except ValueError:
            continue
        f = os.path.join(objdir, f"obj_{idx:04d}.ply")
        if not os.path.exists(f):
            continue
        p = np.asarray(o3d.io.read_point_cloud(f).points)
        if T is not None:
            p = p @ T[:3, :3].T + T[:3, 3]
        if len(p) > max_pts:
            p = p[rng.choice(len(p), max_pts, replace=False)]
        pts[o["name"]] = p[:, :2]
    return pts


def _pastel(i):
    c = np.array(plt.get_cmap("tab20")(i % 20)[:3])
    return tuple(0.55 * c + 0.45)


_SHORT = {"fire_extinguisher": "fire_ext", "industrial_robot": "ind_robot"}


def _label(ax, o, fs=4.8, color="#333333", z=7):
    L, W = o["dimensions"]["length"], o["dimensions"]["width"]
    if o.get("type") in ("clutter", "unknown") and L * W < 1.5:
        return
    if L * W < 0.30:
        return
    nm = o["name"]
    for k, v in _SHORT.items():
        nm = nm.replace(k, v)
    ax.text(o["poseX"], o["poseY"], nm, fontsize=fs, color=color,
            ha="center", va="center", zorder=z)


def fig8(out):
    """Scenario panels A-D over the real segmentation.

    Each object's segmented points are drawn in its own muted color with the
    yaw-rotated explicit-model footprint and its node name; the test-room
    boundary is the dark contour. Edited objects carry callouts with the
    matcher's decision.
    """
    t1 = json.load(open(O("vis_n2_det_filt", "semanticObjects.lf_esc.room.json")))["semanticObjects"]
    pre = {o["name"]: o for o in
           json.load(open(O("vis_n2_det_run1", "semanticObjects.json")))["semanticObjects"]}
    b = json.load(open(O("vis_n2_room_bounds.json")))
    seg = _seg_points(t1, O("vis_n2_det_filt"))

    chair, toolbox, donor = pre["chair_011"], pre["toolbox_013"], pre["clutter_012"]
    seg_pre = _seg_points([chair, toolbox, donor], O("vis_n2_det_filt"))
    ins = {"poseX": donor["poseX"] + 5.0, "poseY": donor["poseY"] - 2.0,
           "dimensions": donor["dimensions"], "name": "inserted object",
           "type": "clutter"}

    def base(ax, skip=(), labels=True):
        _room_outline(ax, b)
        skip_xy = [(pre[n]["poseX"], pre[n]["poseY"]) for n in skip]
        for i, o in enumerate(t1):
            if any(abs(o["poseX"] - x) < .3 and abs(o["poseY"] - y) < .3
                   for x, y in skip_xy):
                continue
            p = seg.get(o["name"])
            if p is not None:
                ax.scatter(p[:, 0], p[:, 1], s=0.45, color=_pastel(i),
                           linewidths=0, zorder=3)
            _rbox(ax, o["poseX"], o["poseY"], o["dimensions"]["length"],
                  o["dimensions"]["width"], o.get("poseTheta", 0.0),
                  "#c8c8c8", lw=0.4, z=4)
            if labels:
                _label(ax, o)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-11.7, 7.0); ax.set_ylim(-10.4, 7.9)
        for s in ax.spines.values():
            s.set_color("#cccccc")

    def box(ax, o, fc, ec, ls="-", lw=1.7, dx=0.0, dy=0.0):
        _rbox(ax, o["poseX"] + dx, o["poseY"] + dy,
              o["dimensions"]["length"], o["dimensions"]["width"],
              o.get("poseTheta", 0.0), ec, lw=lw, ls=ls, fc=fc, z=5)

    def note(ax, xy, text, at, color):
        ax.annotate(text, xy=xy, xytext=at, textcoords="axes fraction",
                    fontsize=8.4, color=color, ha="center", va="center", zorder=8,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.9,
                                    shrinkB=2),
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color,
                              lw=0.9))

    def verdict(ax, text, color, fc):
        ax.text(0.5, 0.045, text, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9.2, color=color, zorder=9,
                bbox=dict(boxstyle="round,pad=0.4", fc=fc, ec=color, lw=1.1))

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 9.4))
    fig.suptitle("Controlled re-scan scenarios, room-scoped robot hall "
                 "(55 objects; colored points = segmented clusters, "
                 "thin boxes = explicit models)", fontsize=11.5, y=0.98)

    # --- (A) unchanged repeat ------------------------------------------------
    ax = axes[0][0]
    base(ax)
    ax.set_title("(A) re-scan, nothing changed", fontsize=10.5, loc="left")
    verdict(ax, "all 55 verdicts carried over\n0 VLM calls  ·  0 identity switches",
            OK, "#f0f9f4")

    # --- (B) single move -----------------------------------------------------
    ax = axes[0][1]
    base(ax, skip=("chair_011",))
    ax.set_title("(B) one chair moved 3.6 m", fontsize=10.5, loc="left")
    box(ax, chair, "none", "#909090", ls=":", lw=1.4)
    box(ax, chair, "none", DET, dx=-3.0, dy=2.0)
    pc = seg_pre.get("chair_011")
    if pc is not None:
        ax.scatter(pc[:, 0] - 3.0, pc[:, 1] + 2.0, s=1.0, color=DET,
                   linewidths=0, zorder=6)
    ax.add_patch(FancyArrowPatch(
        (chair["poseX"], chair["poseY"]),
        (chair["poseX"] - 3.0, chair["poseY"] + 2.0),
        arrowstyle="-|>", mutation_scale=15, color=DET, lw=1.7, zorder=6))
    note(ax, (chair["poseX"], chair["poseY"] - 0.4), "old position",
         (0.86, 0.30), "#707070")
    note(ax, (chair["poseX"] - 3.0, chair["poseY"] + 2.4),
         "recognized as the SAME chair\n(node ID preserved)", (0.30, 0.88), DET)
    verdict(ax, "diff: 1 moved  ·  54 carried  ·  no false events",
            DET, "#eef4fb")

    # --- (C) remove + insert -------------------------------------------------
    ax = axes[1][0]
    base(ax)
    ax.set_title("(C) one object removed, one new object placed",
                 fontsize=10.5, loc="left")
    box(ax, toolbox, "none", BAD, ls="--", lw=1.8)
    ax.plot(toolbox["poseX"], toolbox["poseY"], "x", color=BAD, ms=10,
            mew=2.2, zorder=6)
    box(ax, ins, "none", OK)
    pd_ = seg_pre.get("clutter_012")
    if pd_ is not None:
        ax.scatter(pd_[:, 0] + 5.0, pd_[:, 1] - 2.0, s=1.0, color=OK,
                   linewidths=0, zorder=6)
    note(ax, (toolbox["poseX"] + 0.3, toolbox["poseY"] - 0.2),
         "gone → marked absent\n(node kept in graph, flagged)",
         (0.80, 0.34), BAD)
    note(ax, (ins["poseX"], ins["poseY"] + 0.4),
         "new → inserted, new ID\n(only this cluster re-verified: $0.23)",
         (0.26, 0.63), OK)
    verdict(ax, "diff exact: 1 absent + 1 inserted  ·  53 carried",
            OK, "#f0f9f4")

    # --- (D) occlusion stress ------------------------------------------------
    ax = axes[1][1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("(D) partial occlusion — where matching breaks",
                 fontsize=10.5, loc="left")

    def occl_row(y, frac, ok_flag, verdict_text):
        w, h = 2.1, 1.25
        x0 = 0.55
        col = OK if ok_flag else BAD
        ax.add_patch(Rectangle((x0, y), w, h, facecolor="#dce8f7",
                               edgecolor=DET, lw=1.5))
        ax.add_patch(Rectangle((x0 + w * (1 - frac), y), w * frac, h,
                               facecolor="white", edgecolor=BAD,
                               hatch="////", lw=1.1))
        ax.text(x0 + w / 2, y - 0.55, f"{int(frac * 100)}% of points cropped",
                ha="center", fontsize=8.6, color="#444444")
        ax.text(x0 + w + 0.55, y + h / 2, verdict_text, ha="left", va="center",
                fontsize=9.4, color=col,
                bbox=dict(boxstyle="round,pad=0.42", fc="white", ec=col,
                          lw=1.1))

    occl_row(7.0, 0.2, True,
             "still matched → updated\nlabels and IDs kept  (2/2 objects)")
    occl_row(3.4, 0.4, False,
             "match fails → 2 false-absent\n+ 2 false-new (IDs lost)")
    ax.text(0.5, 0.9, "breaking point lies between 20% and 40% occlusion;\n"
            "motivates overlap-based matching features (future work)",
            ha="left", fontsize=8.8, color="#555555", style="italic")

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    _save(fig, out, "fig8_incremental_update.png")


# ----------------------------------------------------------------- fig 11 ---
def fig11(out):
    """Semantic place map (T3), DK-SMF style: dark map, saturated regions,
    white object-type labels; SLIC superpixel regions on the nav grid map."""
    import re as _re
    from PIL import Image
    import matplotlib.patheffects as pe

    d = json.load(open(O("place_layer_T3_slic.json")))
    places = d["semanticPlaces"]
    cm = d["cell_m"]
    try:
        names = json.load(open(O("place_ring_naming.json")))["T3_slic"]
    except KeyError:
        names = {}
    t3 = json.load(open(O("vis_sota_det",
                          "semanticObjects.lf_esc.visn2frame.room.json")))["semanticObjects"]

    # nav-stack grid map for obstacle/unknown context
    ymap = os.environ.get("NAV_MAP_YAML",
                          "/home/caselab/ammr_twin/map_vis_n2_1.yaml")
    meta = dict(_re.findall(r"(\w+):\s*(.+)", open(ymap).read()))
    res = float(meta["resolution"])
    ox, oy = [float(v) for v in meta["origin"].strip("[]").split(",")[:2]]
    gm = np.flipud(np.asarray(Image.open(
        os.path.join(os.path.dirname(ymap), meta["image"]))))

    allc = np.concatenate([np.array(p["cells"]) for p in places])
    pad = 0.8
    xmin, ymin = allc.min(0) - pad
    xmax, ymax = allc.max(0) + pad
    i0, i1 = int((xmin - ox) / res), int((xmax - ox) / res) + 1
    j0, j1 = int((ymin - oy) / res), int((ymax - oy) / res) + 1
    crop = gm[j0:j1, i0:i1]

    from scipy.ndimage import gaussian_filter, binary_fill_holes

    H, W = crop.shape
    lab = np.zeros((H, W), dtype=int)
    for k, p in enumerate(places, start=1):
        cel = np.array(p["cells"])
        xi = np.round((cel[:, 0] - ox) / res - 0.5).astype(int) - i0
        yi = np.round((cel[:, 1] - oy) / res - 0.5).astype(int) - j0
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        lab[yi[ok], xi[ok]] = k

    # visualization-only smoothing: 2x upsample + gaussian soft voting turns
    # the stair-stepped cell boundaries into smooth region borders
    UP, SIG = 2, 4.0
    labu = np.kron(lab, np.ones((UP, UP), dtype=int))
    room = binary_fill_holes(
        gaussian_filter((labu > 0).astype(float), SIG) > 0.5)
    votes = np.stack([gaussian_filter((labu == k).astype(float), SIG)
                      for k in range(1, len(places) + 1)])
    smooth = votes.argmax(0) + 1
    smooth[~room] = 0

    Hs, Ws = smooth.shape
    img = np.zeros((Hs, Ws, 3))
    img[:] = 0.16                       # outside the room: dark gray
    for k, p in enumerate(places, start=1):
        rgb = [int(c) / 255 for c in p["color"].split(",")]
        img[smooth == k] = rgb

    ext = (ox + i0 * res, ox + i1 * res, oy + j0 * res, oy + j1 * res)
    fig, ax = plt.subplots(figsize=(9.4, 7.6))
    ax.imshow(img, origin="lower", extent=ext, interpolation="bilinear",
              zorder=1)
    gx = np.linspace(ext[0], ext[1], Ws)
    gy = np.linspace(ext[2], ext[3], Hs)
    ax.contour(gx, gy, room.astype(float), levels=[0.5], colors="black",
               linewidths=3.2, zorder=3)

    stroke = [pe.withStroke(linewidth=2.2, foreground="black")]
    for p in places:
        nm = names.get(p["name"], {}).get("name", p["name"])
        cx, cy = p["centroid"]
        ax.text(cx, cy, nm.replace("_", "\n"), fontsize=9.4, ha="center",
                va="center", color="white", weight="bold", zorder=6,
                path_effects=stroke)
    for o in t3:
        st = o["properties"].get("verificationStatus", "")
        x, y = o["poseX"], o["poseY"]
        if st.startswith("verified"):
            ax.plot(x, y, "o", ms=5.5, color="#ffd400",
                    mec="black", mew=0.6, zorder=5)
            if o["type"] not in ("clutter", "unknown"):
                ax.annotate(o["type"], (x, y), textcoords="offset points",
                            xytext=(4, 4), fontsize=6.2, color="white",
                            zorder=6, path_effects=stroke)
        else:
            ax.plot(x, y, "o", ms=4.5, color="#4da3ff",
                    mec="black", mew=0.6, zorder=5)
    ax.plot([], [], "o", ms=6, color="#ffd400", mec="black",
            label="verified object (type labeled)")
    ax.plot([], [], "o", ms=5, color="#4da3ff", mec="black",
            label="unverified (excluded from ring codes)")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    leg = ax.legend(fontsize=8.4, loc="lower right", framealpha=0.9)
    ax.set_title("object-grounded place layer (T3): SLIC superpixel regions"
                 " over the nav-stack grid map;\nLLM-derived names from"
                 " verified-object ring codes", fontsize=10.5)
    _save(fig, out, "fig11_place_map.png")


# ----------------------------------------------------------------- fig 12 ---
def fig12(out):
    """Relation map on the TOSM hierarchy: place regions (isInsideOf) as the
    base layer, object nodes and isNextTo/isOn/isAboveOf edges on top."""
    import re as _re
    from PIL import Image
    import matplotlib.patheffects as pe
    from scipy.ndimage import gaussian_filter, binary_fill_holes

    d = json.load(open(O("place_layer_T3_slic.json")))
    places = d["semanticPlaces"]
    try:
        names = json.load(open(O("place_ring_naming.json")))["T3_slic"]
    except KeyError:
        names = {}
    g = json.load(open(O("testroom_epochs_kg.json")))
    rev = g["revision"]
    nodes = {n["id"]: n for n in g["nodes"] if n.get("presence") != "absent"}
    edges = [e for e in g["edges"] if e.get("revision") == rev
             and e["subj"] in nodes and e["obj"] in nodes]

    # pastel place background (same smoothing as fig11, lightened)
    ymap = os.environ.get("NAV_MAP_YAML",
                          "/home/caselab/ammr_twin/map_vis_n2_1.yaml")
    meta = dict(_re.findall(r"(\w+):\s*(.+)", open(ymap).read()))
    res = float(meta["resolution"])
    ox, oy = [float(v) for v in meta["origin"].strip("[]").split(",")[:2]]
    allc = np.concatenate([np.array(p["cells"]) for p in places])
    pad = 0.8
    xmin, ymin = allc.min(0) - pad
    xmax, ymax = allc.max(0) + pad
    i0, i1 = int((xmin - ox) / res), int((xmax - ox) / res) + 1
    j0, j1 = int((ymin - oy) / res), int((ymax - oy) / res) + 1
    W, H = i1 - i0, j1 - j0
    lab = np.zeros((H, W), dtype=int)
    for k, p in enumerate(places, start=1):
        cel = np.array(p["cells"])
        xi = np.round((cel[:, 0] - ox) / res - 0.5).astype(int) - i0
        yi = np.round((cel[:, 1] - oy) / res - 0.5).astype(int) - j0
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        lab[yi[ok], xi[ok]] = k
    UP, SIG = 2, 4.0
    labu = np.kron(lab, np.ones((UP, UP), dtype=int))
    room = binary_fill_holes(
        gaussian_filter((labu > 0).astype(float), SIG) > 0.5)
    votes = np.stack([gaussian_filter((labu == k).astype(float), SIG)
                      for k in range(1, len(places) + 1)])
    smooth = votes.argmax(0) + 1
    smooth[~room] = 0
    Hs, Ws = smooth.shape
    img = np.ones((Hs, Ws, 3))
    for k, p in enumerate(places, start=1):
        rgb = np.array([int(c) / 255 for c in p["color"].split(",")])
        img[smooth == k] = 0.45 * rgb + 0.55       # pastel wash
    ext = (ox + i0 * res, ox + i1 * res, oy + j0 * res, oy + j1 * res)

    fig, ax = plt.subplots(figsize=(9.6, 7.8))
    ax.imshow(img, origin="lower", extent=ext, interpolation="bilinear",
              zorder=1)
    gx = np.linspace(ext[0], ext[1], Ws)
    gy = np.linspace(ext[2], ext[3], Hs)
    ax.contour(gx, gy, room.astype(float), levels=[0.5], colors="black",
               linewidths=2.6, zorder=2)
    stroke = [pe.withStroke(linewidth=2.0, foreground="white")]
    for p in places:
        nm = names.get(p["name"], {}).get("name", p["name"])
        cx, cy = p["centroid"]
        ax.text(cx, cy, nm.replace("_", "\n"), fontsize=7.8, ha="center",
                va="center", color="#444444", style="italic", zorder=3,
                path_effects=stroke)

    sc = json.load(open(O("t3_place_scoped_relations.json")))
    colors = {"isNextTo": "#5b7ba6", "isOn": "#dd6b20", "isAboveOf": "#7c3aed"}
    widths = {"isNextTo": 1.1, "isOn": 2.4, "isAboveOf": 2.4}
    for e in sc["edges_intra_nextTo"] + sc["edges_vertical"]:
        a, bb = nodes[e["subj"]], nodes[e["obj"]]
        ax.plot([a["pose"]["x"], bb["pose"]["x"]],
                [a["pose"]["y"], bb["pose"]["y"]],
                color=colors[e["pred"]], lw=widths[e["pred"]],
                alpha=0.9 if e["pred"] != "isNextTo" else 0.6, zorder=4)
    cent = {p["name"]: p["centroid"] for p in places}
    for pa, pb in sc["place_adjacency"]:
        (x1, y1), (x2, y2) = cent[pa], cent[pb]
        ax.plot([x1, x2], [y1, y2], ls="--", color="#333333", lw=1.4,
                alpha=0.75, zorder=3)
    for p in places:
        ax.plot(*p["centroid"], "s", ms=6, color="#333333", zorder=4)
    keyset = set(sc["keyObjects"].values())
    byname = {n["name"]: n for n in nodes.values()}
    for n in nodes.values():
        ver = str(n.get("status", "")).startswith("verified")
        ax.plot(n["pose"]["x"], n["pose"]["y"], "o",
                ms=6 if ver else 5, color="#f2c200" if ver else "white",
                mec="#333333", mew=0.7, zorder=5)
        if n.get("type") not in ("clutter", "unknown") and ver:
            ax.annotate(n["type"], (n["pose"]["x"], n["pose"]["y"]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=6.0, color="#111111", zorder=6,
                        path_effects=stroke)
    for nm in keyset:
        if nm in byname:
            n = byname[nm]
            ax.plot(n["pose"]["x"], n["pose"]["y"], "*", ms=15,
                    color="#f2c200", mec="#b91c1c", mew=1.3, zorder=7)
    for pred, c in colors.items():
        n_e = sum(1 for e in sc["edges_intra_nextTo"] + sc["edges_vertical"]
                  if e["pred"] == pred)
        lbl = "isNextTo (intra-place, " if pred == "isNextTo" else f"{pred} ("
        ax.plot([], [], color=c, lw=widths[pred], label=f"{lbl}{n_e})")
    ax.plot([], [], ls="--", color="#333333", lw=1.4,
            label=f"place isAdjacentTo ({len(sc['place_adjacency'])})")
    ax.plot([], [], "*", ms=12, color="#f2c200", mec="#b91c1c",
            label="place key object")
    ax.plot([], [], "o", ms=6, color="#f2c200", mec="#333333",
            label="verified object")
    ax.plot([], [], "o", ms=5, color="white", mec="#333333",
            label="unverified")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8.0, loc="lower right", framealpha=0.95)
    ax.set_title(f"place-scoped TOSM relation map (T3, rev {rev}): intra-place"
                 " object edges, place adjacency,\nand per-place key objects"
                 " (isInsideOf = the region an object lies in)", fontsize=10.5)
    _save(fig, out, "fig12_relations_map.png")


# ----------------------------------------------------------------- fig 13 ---
def fig13(out):
    """Baselines figure: Uni3D vocabulary ablation + Point-SAM refinement."""
    from PIL import Image
    fig = plt.figure(figsize=(11.8, 4.4))

    # (a) Uni3D provisional accuracy vs vocabulary size (Sec 5.6 numbers)
    ax = fig.add_axes([0.05, 0.14, 0.20, 0.72])
    vocab = ["generic-30", "curated-20", "owner-18"]
    acc = [66.7, 48.7, 33.3]
    ax.bar(vocab, acc, 0.55, color=STO)
    for i, a in enumerate(acc):
        ax.text(i, a + 1.5, f"{a}%", ha="center", fontsize=8.6)
    ax.set_ylim(0, 80)
    ax.set_ylabel("Uni3D top-1 [%]", fontsize=9)
    ax.tick_params(axis="x", labelsize=7.6, rotation=12)
    ax.set_title("(a) Uni3D provisional label:\naccuracy drops as vocabulary\n"
                 "narrows to the true classes", fontsize=9.2)
    ax.grid(axis="y", alpha=0.3)

    # (b) Point-SAM: merged parent cluster
    axp = fig.add_axes([0.30, 0.12, 0.20, 0.72])
    axp.imshow(Image.open(O("pointsam_pilot_showroom", "orig_clutter_015.png")))
    axp.axis("off")
    axp.set_title("(b) flagged merged cluster\nclutter_015 (parent LF: "
                  "'clutter')", fontsize=9.2)

    # (c) Point-SAM sub-clusters with their own 4-view LF labels
    sub_lbl = ["machine (1.00)", "office chair (0.53)", "clutter",
               "clutter", "chair (0.72)", "clutter"]
    for i in range(6):
        r, c = divmod(i, 3)
        axs = fig.add_axes([0.53 + c * 0.155, 0.50 - r * 0.40, 0.148, 0.36])
        fp = O("pointsam_pilot_showroom", f"sub_clutter_015_{i}.png")
        if os.path.exists(fp):
            axs.imshow(Image.open(fp))
        axs.axis("off")
        good = "clutter" not in sub_lbl[i]
        axs.set_title(f"sub {i}: {sub_lbl[i]}", fontsize=7.4,
                      color=OK if good else "#666666",
                      weight="bold" if good else "normal")
    fig.text(0.765, 0.95, "(c) Point-SAM subdivision: a machine, an office chair, "
             "and a chair re-emerge from one 'clutter' blob",
             ha="center", fontsize=9.2)
    _save(fig, out, "fig13_pointsam_uni3d.png")


# ----------------------------------------------------------------- fig 10 ---
def fig10(out):
    import open3d as o3d
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6))
    rng = np.random.default_rng(0)

    # (a) showroom soffit strips in scene section (strips run along y at
    # x~0.2, between the z=2.48 and z=2.18 ceiling levels)
    ax = axes[0][0]
    raw = np.asarray(o3d.io.read_point_cloud(O("_raw_cache.ply")).points)
    m = (raw[:, 0] > -2.0) & (raw[:, 0] < 2.5) & \
        (raw[:, 2] > -1.6) & (raw[:, 2] < 2.9) & \
        (raw[:, 1] > -3.0) & (raw[:, 1] < 5.0)
    p = raw[m]
    s = rng.choice(len(p), min(len(p), 80000), replace=False)
    ax.scatter(p[s, 1], p[s, 2], s=0.3, c="lightgray", linewidths=0)
    for i in (27, 28, 29, 30, 31):
        q = np.asarray(o3d.io.read_point_cloud(
            O("showroom_det", f"obj_{i:04d}.ply")).points)
        ax.scatter(q[:, 1], q[:, 2], s=2.0, c=BAD, linewidths=0)
    ax.set_title("(a) multi-level ceiling soffits\n→ 'keyboard' phantoms "
                 "(escalation-verified)", fontsize=9.6)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    # (b) vis_n2 hanging strip verified as door handle
    ax = axes[0][1]
    raw2 = np.asarray(o3d.io.read_point_cloud(
        os.environ.get("VIS_N2_HD_PLY", "data/scene_vis_n2_hd.ply")).points)
    q = np.asarray(o3d.io.read_point_cloud(
        O("vis_n2_det_run1", "obj_0083.ply")).points)
    c = q.mean(0)
    m = (np.abs(raw2[:, 0] - c[0]) < 1.5) & (np.abs(raw2[:, 1] - c[1]) < 1.5)
    p = raw2[m]
    if len(p) > 60000:
        p = p[rng.choice(len(p), 60000, replace=False)]
    ax.scatter(p[:, 0], p[:, 2], s=0.5, c="lightgray", linewidths=0)
    ax.scatter(q[:, 0], q[:, 2], s=3.0, c=BAD, linewidths=0)
    ax.set_title("(b) hanging chain/ghost returns\n→ 'door handle' "
                 "(unanimous verified)", fontsize=9.6)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    # (c) clutter absorption: red robot
    from PIL import Image
    ax = axes[1][0]
    fp = O("vis_n2_det_hdviews", "views", "clutter_068", "view_000.png")
    if os.path.exists(fp):
        ax.imshow(Image.open(fp))
    ax.set_title("(c) clutter absorption: partially\ndisassembled robot "
                 "→ 'clutter' (unanimous)", fontsize=9.6)
    ax.axis("off")

    # (d) banner mislabeled but correctly gated
    ax = axes[1][1]
    fp = O("vis_n2_det_hdviews", "views", "tank_050", "view_000.png")
    if not os.path.exists(fp):
        for cand in os.listdir(O("vis_n2_det_hdviews", "views")):
            if cand.endswith("_050"):
                fp = O("vis_n2_det_hdviews", "views", cand, "view_000.png")
                break
    if os.path.exists(fp):
        ax.imshow(Image.open(fp))
    ax.set_title("(d) exhibition banner → 'refrigerator'\nsplit vote "
                 "→ correctly gated unverified", fontsize=9.6)
    ax.axis("off")
    _save(fig, out, "fig10_error_taxonomy.png")


# ------------------------------------------------------------------ fig 9 ---
def fig9(out):
    """Real multi-epoch update over the T3 segmentation.

    Underlay: every T3 cluster's segmented points in its own muted color
    (what the deterministic backbone produced from the newest scan), plus the
    test-room boundary. Overlay: yaw-rotated node footprints colored by the
    diff state, with node names on the specific-type objects.
    """
    g = json.load(open(O("testroom_epochs_kg.json")))
    b = json.load(open(O("vis_n2_room_bounds.json")))
    t3 = json.load(open(O("vis_sota_det",
                          "semanticObjects.lf_esc.visn2frame.room.json")))["semanticObjects"]
    T = np.load(O("vissota_to_visn2_T.npy"))
    seg = _seg_points(t3, O("vis_sota_det"), max_pts=420, T=T)

    fig, ax = plt.subplots(figsize=(9.4, 7.6))
    _room_outline(ax, b, tag="test-room boundary")
    for i, o in enumerate(t3):
        p = seg.get(o["name"])
        if p is not None:
            ax.scatter(p[:, 0], p[:, 1], s=0.5, color=_pastel(i),
                       linewidths=0, zorder=3)

    def rbox(x, y, d, th, color, lw=1.3, ls="-"):
        # huge merged clusters would dominate the map: fade their footprints
        big = d["length"] * d["width"] > 18.0
        _rbox(ax, x, y, d["length"], d["width"], th, color,
              lw=0.9 if big else lw, ls=ls, z=5, alpha=0.30 if big else 1.0)

    rev = g.get("revision", 2)
    counts = {"moved": 0, "inserted": 0, "absent": 0, "updated": 0}
    for n in g["nodes"]:
        hist = n.get("history", [])
        born = hist[0]["revision"] if hist else 1
        d, th = n["dimensions"], n["pose"].get("theta", 0.0)
        o_lbl = {"poseX": n["pose"]["x"], "poseY": n["pose"]["y"],
                 "dimensions": d, "name": n["name"], "type": n.get("type")}
        if n.get("presence") == "absent":
            # only nodes that BECAME absent at the latest revision; earlier
            # departures would bury the current transition under old boxes
            if not any(h["revision"] == rev and
                       h["changes"].get("presence", {}).get("new") == "absent"
                       for h in hist):
                continue
            big = d["length"] * d["width"] > 18.0
            _rbox(ax, n["pose"]["x"], n["pose"]["y"], d["length"],
                  d["width"], th, BAD, lw=0.8, ls="--", z=5,
                  alpha=0.25 if big else 0.55)
            counts["absent"] += 1
        elif born == rev:
            rbox(n["pose"]["x"], n["pose"]["y"], d, th, OK, 1.3)
            if n["name"] == "machine_012":
                ax.text(n["pose"]["x"], n["pose"]["y"],
                        "machine_012\n(merged mega-cluster, Sec. 5.6)",
                        fontsize=5.8, color="#1e5e40", ha="center",
                        va="center", style="italic", zorder=7)
            else:
                _label(ax, o_lbl, fs=5.4, color="#1e5e40")
            counts["inserted"] += 1
        elif any(h.get("event") == "moved" and h["revision"] == rev
                 for h in hist):
            ev = [h for h in hist
                  if h.get("event") == "moved" and h["revision"] == rev][-1]
            old, new = ev["changes"]["pose"]["old"], ev["changes"]["pose"]["new"]
            rbox(old["x"], old["y"], d, th, "#888888", 0.9, ":")
            rbox(new["x"], new["y"], d, th, DET, 2.0)
            ax.add_patch(FancyArrowPatch((old["x"], old["y"]),
                         (new["x"], new["y"]), arrowstyle="-|>",
                         mutation_scale=13, color=DET, lw=1.6, zorder=6))
            ax.annotate(n["name"] + " (ID kept)", xy=(new["x"], new["y"]),
                        xytext=(new["x"] + 1.0, new["y"] + 1.6), fontsize=6.4,
                        color=DET, zorder=8,
                        arrowprops=dict(arrowstyle="-", color=DET, lw=0.7),
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec=DET, lw=0.8))
            counts["moved"] += 1
        else:
            rbox(n["pose"]["x"], n["pose"]["y"], d, th, "#b08c2e", 1.5)
            _label(ax, o_lbl, fs=5.4, color="#7a5f14")
            counts["updated"] += 1
    for c, ls, lbl in [
            ("#b08c2e", "-", f"matched in place ({counts['updated']})"),
            (DET, "-", f"moved, ID preserved ({counts['moved']})"),
            (OK, "-", f"inserted ({counts['inserted']})"),
            (BAD, "--", f"absent, kept in graph ({counts['absent']})")]:
        ax.plot([], [], color=c, ls=ls, label=lbl)
    ax.scatter([], [], s=12, color=_pastel(4),
               label="segmented clusters (T3 scan)")
    ax.legend(fontsize=8.4, loc="lower right", framealpha=0.95)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"real multi-epoch update of the same room, latest "
                 f"transition (rev {rev-1} \u2192 {rev})\n(cross-scan "
                 "registration 4.2\u20135.1 cm RMSE; identity carried for "
                 "specific-type objects)", fontsize=10)
    _save(fig, out, "fig9_two_epoch_update.png")


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5,
        "fig6": fig6, "fig7": fig7, "fig8": fig8, "fig9": fig9,
        "fig10": fig10, "fig11": fig11, "fig12": fig12, "fig13": fig13}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=O("paper_figs"))
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names = args.only.split(",") if args.only else list(FIGS)
    for n in names:
        try:
            FIGS[n](args.out)
        except Exception as e:
            print(f"[FAIL] {n}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
