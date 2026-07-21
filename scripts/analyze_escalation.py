#!/usr/bin/env python3
"""Escalation accuracy-cost analysis (paper Sec. 5.3).

Reads an lf_esc.json (per-object viewVotes retained) + a GT file and reports:
  - base 4-view late fusion vs. post-escalation fusion accuracy
    (counterfactuals recomputed offline from the stored votes)
  - escalation trigger rate; recoveries (wrong->right) vs. regressions
  - verification-status precision (verified / verified_escalated / unverified)
  - accuracy-cost points: 4-view only, +escalation, at several resolve-share
    thresholds

GT file: list of {id, name, gt_type, gt_quality(confirmed|plausible|unknown)}.
Accuracy is reported on confirmed GT only (strict) and confirmed+plausible
(lenient); unknown-GT objects are excluded.

  .venv/bin/python scripts/analyze_escalation.py \
      --lf outputs/stage2_no_wall_objects_split/semanticObjects.lf_esc.json \
      --gt outputs/showroom_gt_draft.json --out outputs/esc_analysis_showroom.json
"""
import argparse
import json


def fuse(votes):
    score = {}
    for v in votes:
        t = v["type"].strip().lower()
        score[t] = score.get(t, 0.0) + float(v["confidence"])
    total = sum(score.values()) or 1.0
    top = max(score, key=score.get)
    return top, score[top] / total, len(score) == 1


def type_match(pred, gt):
    p, g = pred.strip().lower(), gt.strip().lower()
    if p == g:
        return True
    # tolerate common synonyms so the analysis measures verification, not
    # vocabulary; the pairs mirror the audit's grading practice
    syn = [{"tv", "television", "display", "monitor", "screen"},
           {"mobile robot", "agv", "robot"},
           # owner: mobile manipulator with occluded wheels — robot arm ok
           {"mobile manipulator", "robot arm", "robotic arm", "robot"},
           {"control panel", "panel", "control cabinet"},
           {"cabinet", "locker"},
           {"junction box", "electrical box", "distribution box"}]
    return any(p in s and g in s for s in syn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lf", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--esc-call-cost", type=float, default=0.0091,
                    help="avg USD per escalation call (from usage json)")
    args = ap.parse_args()

    d = json.load(open(args.lf))
    objs = d["semanticObjects"] if isinstance(d, dict) and \
        "semanticObjects" in d else d
    gd = json.load(open(args.gt))
    glist = gd["objects"] if isinstance(gd, dict) and "objects" in gd else gd
    # excluded from evaluation: structure artifacts (not semantic objects)
    # and out-of-scope objects (outside the modeled room)
    gt = {str(g["id"]): g for g in glist
          if not g.get("structure_artifact") and not g.get("out_of_scope")}

    rows = []
    for o in objs:
        g = gt.get(str(o["id"]))
        if not g:
            continue
        votes = o["properties"].get("viewVotes", [])
        base = [v for v in votes if not v.get("zoom")
                and v["azimuth"] in (0, 90, 180, 270)][:4]
        bt, bs, bu = fuse(base) if base else (o["type"], 1.0, True)
        ft, fs, fu = fuse(votes) if votes else (bt, bs, bu)
        rows.append({
            "id": o["id"], "name": o["name"], "gt": g["gt_type"],
            "gt_quality": g["gt_quality"],
            "base_type": bt, "base_share": round(bs, 3), "unanimous": bu,
            "final_type": ft, "final_share": round(fs, 3),
            "status": o["properties"].get("verificationStatus"),
            "escalated": o["properties"].get("escalated", False),
            "n_votes": len(votes),
            "base_ok": type_match(bt, g["gt_type"]),
            "final_ok": type_match(ft, g["gt_type"]),
        })

    def acc(rs, key, quality):
        sel = [r for r in rs if r["gt_quality"] in quality]
        return (round(sum(r[key] for r in sel) / len(sel), 3), len(sel)) \
            if sel else (None, 0)

    conf = ("confirmed",)
    len_q = ("confirmed", "plausible")
    esc_rows = [r for r in rows if r["escalated"]]
    recover = [r["name"] for r in esc_rows if not r["base_ok"] and r["final_ok"]]
    regress = [r["name"] for r in esc_rows if r["base_ok"] and not r["final_ok"]]

    by_status = {}
    for st in ("verified", "verified_escalated", "unverified"):
        rs = [r for r in rows if r["status"] == st]
        by_status[st] = {"n": len(rs),
                         "acc_confirmed": acc(rs, "final_ok", conf)[0]}

    esc_calls = sum(r["n_votes"] - 4 for r in rows)
    summary = {
        "objects_gt": len(rows),
        "trigger_rate": round(len(esc_rows) / len(rows), 3) if rows else None,
        "base4_acc_strict": acc(rows, "base_ok", conf),
        "base4_acc_lenient": acc(rows, "base_ok", len_q),
        "esc_acc_strict": acc(rows, "final_ok", conf),
        "esc_acc_lenient": acc(rows, "final_ok", len_q),
        "recoveries": recover, "regressions": regress,
        "unanimity": {
            "n": sum(r["unanimous"] for r in rows),
            "acc": acc([r for r in rows if r["unanimous"]],
                       "base_ok", conf)[0]},
        "split_base_acc": acc([r for r in rows if not r["unanimous"]],
                              "base_ok", conf)[0],
        "by_status": by_status,
        "escalation_calls": esc_calls,
        "escalation_cost_usd": round(esc_calls * args.esc_call_cost, 2),
    }
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"),
              indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
