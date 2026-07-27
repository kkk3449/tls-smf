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
    d = json.load(open(O("stage2_no_wall_objects_split",
                         "semanticObjects.lf_esc.filtered.json")))
    by = {o["id"]: o for o in d["semanticObjects"]}
    def card(o):
        pr = o["properties"]
        return json.dumps({
            "name": o["name"], "type": o["type"],
            "pose": {"x": o["poseX"], "y": o["poseY"],
                     "z": pr.get("poseZ"), "theta": o.get("poseTheta")},
            "dimensions": o["dimensions"], "color": o.get("color"),
            "verificationStatus": pr.get("verificationStatus"),
            "voteShare": pr.get("voteShare"),
            "voteScores": pr.get("voteScores"),
            "escalated": pr.get("escalated"),
            "isMovable": pr.get("isMovable"),
            "isKeyObject": pr.get("isKeyObject"),
        }, indent=1, ensure_ascii=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, oid, title, col in [
            (axes[0], "0", "verified record", OK),
            (axes[1], "46", "unverified record (gated at query time)", BAD)]:
        ax.axis("off")
        ax.set_title(title, fontsize=11, color=col)
        ax.text(0.02, 0.97, card(by[oid]), family="monospace", fontsize=7.6,
                va="top", transform=ax.transAxes)
        for s in ax.spines.values():
            s.set_visible(True)
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
            (O("esc_analysis_showroom.json"), "showroom", DET, None),
            (O("esc_analysis_vis_n2.json"), "vis_n2", STO, None)]:
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
    for ax, f, scene in [(axes[0], O("query_bench_showroom_v5.json"), "showroom"),
                         (axes[1], O("query_bench_vis_n2.json"), "vis_n2")]:
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
def fig8(out):
    """Scenario panels A-D: one story per panel instead of a wall of boxes.

    Unchanged objects are soft filled shapes (background furniture); every
    edited object gets a labeled callout saying what the matcher decided.
    """
    t1 = json.load(open(O("vis_n2_det_filt", "semanticObjects.lf_esc.room.json")))["semanticObjects"]
    pre = {o["name"]: o for o in
           json.load(open(O("vis_n2_det_run1", "semanticObjects.json")))["semanticObjects"]}
    b = json.load(open(O("vis_n2_room_bounds.json")))
    carr = np.array(b["cells"]) * b["cell_m"]

    chair, toolbox, donor = pre["chair_011"], pre["toolbox_013"], pre["clutter_012"]
    ins = {"poseX": donor["poseX"] + 5.0, "poseY": donor["poseY"] - 2.0,
           "dimensions": donor["dimensions"]}

    def base(ax, skip=()):
        ax.scatter(carr[:, 0], carr[:, 1], s=1.4, c="#f4f4f4", linewidths=0)
        skip_xy = [(pre[n]["poseX"], pre[n]["poseY"]) for n in skip]
        for o in t1:
            if any(abs(o["poseX"] - x) < .3 and abs(o["poseY"] - y) < .3
                   for x, y in skip_xy):
                continue
            L, W = o["dimensions"]["length"], o["dimensions"]["width"]
            ax.add_patch(Rectangle((o["poseX"] - L / 2, o["poseY"] - W / 2), L, W,
                                   facecolor="#e9e9e9", edgecolor="#d2d2d2", lw=0.4))
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-11.7, 7.0); ax.set_ylim(-10.4, 7.9)
        for s in ax.spines.values():
            s.set_color("#cccccc")

    def box(ax, o, fc, ec, ls="-", lw=1.7, dx=0.0, dy=0.0):
        L, W = o["dimensions"]["length"], o["dimensions"]["width"]
        ax.add_patch(Rectangle((o["poseX"] + dx - L / 2, o["poseY"] + dy - W / 2),
                               L, W, facecolor=fc, edgecolor=ec, lw=lw,
                               linestyle=ls, zorder=5))

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
                 "(55 objects; gray = unchanged furniture)", fontsize=11.5,
                 y=0.98)

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
    box(ax, chair, "#dce8f7", DET, dx=-3.0, dy=2.0)
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
    box(ax, ins, "#e6f4ec", OK)
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
    """Real two-epoch update: June manual scan -> July exploration scan."""
    g = json.load(open(O("testroom_epochs_kg.json")))
    b = json.load(open(O("vis_n2_room_bounds.json")))
    cells = np.array(b["cells"]) * b["cell_m"]
    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    ax.scatter(cells[:, 0], cells[:, 1], s=2.5, c="#f4f4f4", linewidths=0)

    def rect(x, y, L, W, color, lw=1.2, ls="-"):
        ax.add_patch(Rectangle((x - L / 2, y - W / 2), L, W, fill=False,
                               ec=color, lw=lw, linestyle=ls))

    rev = g.get("revision", 2)
    counts = {"moved": 0, "inserted": 0, "absent": 0, "updated": 0}
    for n in g["nodes"]:
        hist = n.get("history", [])
        born = hist[0]["revision"] if hist else 1
        d = n["dimensions"]
        if n.get("presence") == "absent":
            # only nodes that BECAME absent at the latest revision; earlier
            # departures would bury the current transition under old boxes
            if not any(h["revision"] == rev and
                       h["changes"].get("presence", {}).get("new") == "absent"
                       for h in hist):
                continue
            rect(n["pose"]["x"], n["pose"]["y"], d["length"], d["width"],
                 BAD, 1.1, "--")
            counts["absent"] += 1
        elif born == rev:
            rect(n["pose"]["x"], n["pose"]["y"], d["length"], d["width"],
                 OK, 1.1)
            counts["inserted"] += 1
        elif any(h.get("event") == "moved" and h["revision"] == rev
                 for h in hist):
            ev = [h for h in hist
                  if h.get("event") == "moved" and h["revision"] == rev][-1]
            old, new = ev["changes"]["pose"]["old"], ev["changes"]["pose"]["new"]
            rect(old["x"], old["y"], d["length"], d["width"], "#888", 0.9, ":")
            rect(new["x"], new["y"], d["length"], d["width"], DET, 1.9)
            ax.add_patch(FancyArrowPatch((old["x"], old["y"]),
                         (new["x"], new["y"]), arrowstyle="-|>",
                         mutation_scale=13, color=DET, lw=1.6))
            counts["moved"] += 1
        else:
            rect(n["pose"]["x"], n["pose"]["y"], d["length"], d["width"],
                 "#b08c2e", 1.4)
            counts["updated"] += 1
    for c, lbl in [("#b08c2e", f"matched in place ({counts['updated']})"),
                   (DET, f"moved, ID preserved ({counts['moved']})"),
                   (OK, f"inserted ({counts['inserted']})"),
                   (BAD, f"absent, kept in graph ({counts['absent']})")]:
        ax.plot([], [], color=c, label=lbl)
    ax.legend(fontsize=8.6, loc="lower right")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"real multi-epoch update of the same room, latest "
                 f"transition (rev {rev-1} \u2192 {rev})\n(cross-scan "
                 "registration 4.2\u20135.1 cm RMSE; identity carried for "
                 "specific-type objects)", fontsize=10)
    _save(fig, out, "fig9_two_epoch_update.png")


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5,
        "fig6": fig6, "fig7": fig7, "fig8": fig8, "fig9": fig9, "fig10": fig10}


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
