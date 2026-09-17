#!/usr/bin/env python3
"""Resolve-share threshold sensitivity sweep (journal revision, "why 0.6?").

Re-scores the STORED per-view votes of the three campaign runs offline --
no VLM/network calls.  For every escalated object the post-escalation fused
share is recomputed with the same fuse() logic as analyze_escalation.py /
vlm_late_fusion.py, then resolve_share is swept 0.40..0.90 (step 0.05) and
the verification tiers are re-assigned:

  unanimous base vote          -> verified            (threshold-independent)
  escalated, share >= t        -> verified_escalated
  escalated, share <  t        -> unverified

Per scene x threshold we report tier sizes (over all objects in the run),
precision of the verified_escalated tier and of the combined
verified+verified_escalated set on confirmed GT (synonym-tolerant, same
type_match as analyze_escalation.py), and recall/F1 of the combined set
against all confirmed-GT ("GT-scoreable") objects.

At t=0.60 the numbers are asserted against the published esc_analysis_*.json
files (canonical run+GT pairs verified to reproduce their rows exactly).

  .venv/bin/python scripts/threshold_sweep.py
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from analyze_escalation import fuse, type_match  # noqa: E402  (reuse, not reimplement)

# canonical inputs: the run+GT pairs that reproduce esc_analysis_*.json
# rows byte-for-byte (checked in verify_at_060 below against by_status /
# esc_acc_strict).  cafe8f main GT (n=20 confirmed) is the published 35.0%
# cell of Table tab:cafe; variantB (fragment-excluded, n=15) is its 46.7%
# column and is verified too but not swept.
SCENES = [
    ("showroom",
     "outputs/stage2_no_wall_objects_split/semanticObjects.lf_esc.json",
     "outputs/showroom_gt.json",
     "outputs/esc_analysis_showroom.json"),
    ("vis_n2",
     "outputs/vis_n2_det_run1/semanticObjects.lf_esc.json",
     "outputs/vis_n2_gt.json",
     "outputs/esc_analysis_vis_n2.json"),
    ("cafe8f",
     "outputs/cafe8f_objects/semanticObjects.lf_esc.json",
     "outputs/cafe8f_gt.json",
     "outputs/esc_analysis_cafe8f.json"),
]
CAFE_VARIANT_B_GT = "outputs/cafe8f_gt_variantB.json"

THRESHOLDS = [round(0.40 + 0.05 * i, 2) for i in range(11)]  # 0.40 .. 0.90
EPS = 1e-9  # share is rounded to 3 decimals like vlm_late_fusion.fuse()


def load_gt(path):
    gd = json.load(open(path))
    glist = gd["objects"] if isinstance(gd, dict) and "objects" in gd else gd
    return {str(g["id"]): g for g in glist
            if not g.get("structure_artifact") and not g.get("out_of_scope")}


def load_objects(path):
    d = json.load(open(path))
    return d["semanticObjects"] if isinstance(d, dict) and \
        "semanticObjects" in d else d


def score_objects(lf_path, gt_path):
    """One record per object in the run, with the recomputed fused share."""
    gt = load_gt(gt_path)
    recs = []
    for o in load_objects(lf_path):
        p = o["properties"]
        votes = p.get("viewVotes", [])
        if votes:
            ftype, fshare, _ = fuse(votes)       # all stored votes (4 or 12)
            fshare = round(fshare, 3)            # mirror vlm_late_fusion.fuse
        else:                                    # no votes retained: keep run
            ftype, fshare = o["type"], 1.0
        g = gt.get(str(o["id"]))
        recs.append({
            "id": o["id"], "name": o["name"],
            "escalated": bool(p.get("escalated", False)),
            "stored_status": p.get("verificationStatus"),
            "final_type": ftype, "final_share": fshare,
            "gt_type": g["gt_type"] if g else None,
            "gt_quality": g["gt_quality"] if g else None,
            "gt_ok": type_match(ftype, g["gt_type"]) if g else None,
        })
    return recs


def tier_at(rec, t):
    if not rec["escalated"]:
        return rec["stored_status"] or "verified"  # unanimous -> verified
    return ("verified_escalated" if rec["final_share"] + EPS >= t
            else "unverified")


def prec(recs):
    """Synonym-tolerant precision on confirmed GT; (value, n)."""
    sel = [r for r in recs if r["gt_quality"] == "confirmed"]
    if not sel:
        return None, 0
    return round(sum(r["gt_ok"] for r in sel) / len(sel), 3), len(sel)


def sweep_scene(name, recs):
    gt_scoreable = [r for r in recs if r["gt_quality"] == "confirmed"]
    n_gt = len(gt_scoreable)
    rows = []
    for t in THRESHOLDS:
        tiers = {}
        for r in recs:
            tiers.setdefault(tier_at(r, t), []).append(r)
        ver = tiers.get("verified", [])
        ve = tiers.get("verified_escalated", [])
        unv = tiers.get("unverified", [])
        comb = ver + ve
        p_ve, n_ve_conf = prec(ve)
        p_comb, n_comb_conf = prec(comb)
        tp = sum(r["gt_ok"] for r in comb if r["gt_quality"] == "confirmed")
        recall = round(tp / n_gt, 3) if n_gt else None
        f1 = (round(2 * p_comb * recall / (p_comb + recall), 3)
              if p_comb and recall and (p_comb + recall) else None)
        rows.append({
            "scene": name, "threshold": t,
            "n_verified": len(ver), "n_verified_escalated": len(ve),
            "n_unverified": len(unv),
            "prec_verified_escalated": p_ve,
            "n_ve_confirmed": n_ve_conf,
            "prec_combined": p_comb, "n_combined_confirmed": n_comb_conf,
            "recall_combined": recall, "f1_combined": f1,
            "n_gt_scoreable": n_gt,
        })
    # sensitive band: escalated objects whose tier flips between 0.5 and 0.7
    band = [r for r in recs if r["escalated"]
            and tier_at(r, 0.5) != tier_at(r, 0.7)]
    band_info = {
        "n": len(band),
        "objects": [{"name": r["name"], "share": r["final_share"],
                     "gt_type": r["gt_type"], "gt_quality": r["gt_quality"],
                     "final_type": r["final_type"], "gt_ok": r["gt_ok"]}
                    for r in sorted(band, key=lambda r: r["final_share"])],
    }
    return rows, band_info


def verify_at_060(name, recs, esc_path):
    """Reproduce the published numbers from esc_analysis_*.json at t=0.6."""
    pub = json.load(open(esc_path))["summary"]
    gt_rows = [r for r in recs if r["gt_quality"] is not None]
    esc_gt = [r for r in gt_rows if r["escalated"]]
    trigger = round(len(esc_gt) / len(gt_rows), 3)
    esc_acc, n_conf = prec(gt_rows)
    checks = {
        "trigger_rate": (trigger, pub["trigger_rate"]),
        "esc_acc_strict": ((esc_acc, n_conf), tuple(pub["esc_acc_strict"])),
    }
    for st in ("verified", "verified_escalated", "unverified"):
        sel = [r for r in gt_rows if tier_at(r, 0.6) == st]
        checks[f"n_{st}(gt)"] = (len(sel), pub["by_status"][st]["n"])
        checks[f"acc_{st}"] = (prec(sel)[0],
                               pub["by_status"][st]["acc_confirmed"])
    # stored statuses must be reproduced exactly at t=0.6 (all objects)
    mism = [r["name"] for r in recs if tier_at(r, 0.6) != r["stored_status"]]
    checks["status_mismatches_at_0.6"] = (mism, [])
    ok = all(a == b for a, b in checks.values())
    return ok, checks


def main():
    os.chdir(ROOT)
    all_rows, result = [], {"thresholds": THRESHOLDS, "scenes": {}}

    for name, lf, gtf, escf in SCENES:
        recs = score_objects(lf, gtf)
        ok, checks = verify_at_060(name, recs, escf)
        status = "OK" if ok else "MISMATCH"
        print(f"[verify @0.60] {name}: {status}")
        for k, (got, want) in checks.items():
            flag = "" if got == want else "   <-- MISMATCH"
            print(f"   {k:28s} got={got}  published={want}{flag}")
        if not ok:
            sys.exit(f"{name}: published numbers not reproduced -- check "
                     f"canonical inputs before trusting the sweep.")
        rows, band = sweep_scene(name, recs)
        all_rows += rows
        result["scenes"][name] = {
            "lf": lf, "gt": gtf, "esc_analysis": escf,
            "n_objects": len(recs),
            "n_escalated": sum(r["escalated"] for r in recs),
            "verified_at_060_reproduced": ok,
            "sweep": rows, "sensitive_band_0.5_to_0.7": band,
            "escalated_shares": sorted(r["final_share"] for r in recs
                                       if r["escalated"]),
        }

    # cafe8f variantB (fragment-excluded) reproduction check only
    recs_vb = score_objects(SCENES[2][1], CAFE_VARIANT_B_GT)
    acc_vb, n_vb = prec([r for r in recs_vb if r["gt_quality"] is not None])
    print(f"[verify @0.60] cafe8f variantB (fragment-excluded): "
          f"esc_acc={acc_vb} n={n_vb} (published 0.467, n=15)")
    result["cafe8f_variantB_check"] = {
        "gt": CAFE_VARIANT_B_GT, "esc_acc_strict": [acc_vb, n_vb],
        "published": [0.467, 15]}

    with open("outputs/threshold_sweep.json", "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    cols = ["scene", "threshold", "n_verified", "n_verified_escalated",
            "n_unverified", "prec_verified_escalated", "n_ve_confirmed",
            "prec_combined", "n_combined_confirmed", "recall_combined",
            "f1_combined", "n_gt_scoreable"]
    with open("outputs/threshold_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)

    # report table
    fmt = ("{scene:9s} {threshold:.2f}  V={n_verified:3d} "
           "VE={n_verified_escalated:3d} U={n_unverified:3d}  "
           "P_ve={p_ve:>6s}  P_comb={p_c:>6s}  R={r:>6s}  F1={f1:>6s}")
    print("\nscene     thr   tiers (all objects)        "
          "P(esc tier)  P(comb)  recall  F1   [confirmed GT]")
    for row in all_rows:
        print(fmt.format(scene=row["scene"], threshold=row["threshold"],
                         n_verified=row["n_verified"],
                         n_verified_escalated=row["n_verified_escalated"],
                         n_unverified=row["n_unverified"],
                         p_ve="-" if row["prec_verified_escalated"] is None
                         else f'{row["prec_verified_escalated"]:.3f}',
                         p_c=f'{row["prec_combined"]:.3f}',
                         r=f'{row["recall_combined"]:.3f}',
                         f1="-" if row["f1_combined"] is None
                         else f'{row["f1_combined"]:.3f}'))
    print()
    for name in result["scenes"]:
        band = result["scenes"][name]["sensitive_band_0.5_to_0.7"]
        objs = ", ".join(f'{o["name"]}({o["share"]:.2f},'
                         f'{"ok" if o["gt_ok"] else "wrong" if o["gt_ok"] is False else "noGT"})'
                         for o in band["objects"])
        print(f"sensitive band 0.50->0.70 {name}: {band['n']} objects: {objs}")
    print("\nwrote outputs/threshold_sweep.json / outputs/threshold_sweep.csv")


if __name__ == "__main__":
    main()
