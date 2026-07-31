#!/usr/bin/env python3
"""Clutter-recovery re-verification pilot (candidate-shortlist prompt).

For every verified-clutter KG object at the target epoch (plus Point-SAM
sub-clusters of the ones that decomposed, plus named non-clutter rescue
targets such as an unverified monitor), re-ask the Stage-B verifier with a
RECOVERY prompt: 'clutter' is treated as *unidentified*, not as a valid
final class. The Uni3D top-k (minus 'clutter') is offered as a candidate
shortlist, and keeping 'clutter' is allowed only when nothing fits — the
anti-hallucination rules (size sanity, sparse-render caution, multi-view
consistency) stay in force.

Report-only: results go to a JSON for audit; nothing is written back to the
KG here.

  ANTHROPIC_API_KEY=... .venv/bin/python scripts/clutter_reverify.py \
      --targets outputs/clutter_recovery_targets.json \
      --det-dir outputs/vis_sota_det \
      --subs-dir outputs/clutter_recovery_t3/subs \
      --out outputs/clutter_recovery_t3/reverify_results.json
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from blk360seg import vlm_stage_b as vb                          # noqa: E402

_SYS_RECOVERY = vb._SYS_SYMBOLIC.replace(
    "DECISION RULE (follow exactly):\n",
    "RECOVERY MODE: the current `uni3d_type` is 'clutter' (or an unverified "
    "guess), which here means UNIDENTIFIED, not a valid final class. Your job "
    "is to try hard to identify the object. Rule 1 (keep-if-correct) does NOT "
    "protect 'clutter': only answer 'clutter' if, after honestly checking "
    "every candidate and your own open vocabulary, no class is clearly "
    "supported by the visible structure. All cautions below still apply — a "
    "specific label must be EARNED by clearly outlined shape, never guessed "
    "from speckle.\n\nDECISION RULE (follow exactly):\n")


def topk_from_csv(path, k=5):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            cands = []
            for i in range(1, k + 1):
                c, s = row.get(f"top{i}"), row.get(f"score{i}")
                if c:
                    cands.append((c, float(s)))
            out[row["file"]] = cands
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--det-dir", required=True)
    ap.add_argument("--subs-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--cands-csv", default=None,
                    help="classification.csv to draw candidate shortlists "
                         "from (default: <det-dir>/classification.csv)")
    ap.add_argument("--extra-cands", default=None,
                    help="comma-separated domain-knowledge classes appended "
                         "to every candidate shortlist (site metadata)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    det = json.load(open(os.path.join(
        args.det_dir, "semanticObjects.lf_esc.visn2frame.room.json")))
    det = det["semanticObjects"] if isinstance(det, dict) else det
    byname = {o["name"]: o for o in det}
    pre = json.load(open(os.path.join(args.det_dir, "semanticObjects.json")))
    pre = pre["semanticObjects"] if isinstance(pre, dict) else pre
    f2pre = {o["properties"]["imageFile"]: o["name"] for o in pre}
    det_cands = topk_from_csv(args.cands_csv or
                              os.path.join(args.det_dir, "classification.csv"))
    # robust floor estimate: most objects rest on the floor, so the low
    # quartile of cluster bottoms tracks it; a plain min() gets dragged down
    # by wall-remnant outliers
    bottoms = sorted(o["properties"]["poseZ"] - o["dimensions"]["height"] / 2
                     for o in det)
    floor_z = bottoms[len(bottoms) // 4]

    def views(vdir):
        return [os.path.join(vdir, f"view_{a:03d}.png")
                for a in (0, 90, 180, 270)]

    extra = ([c.strip() for c in args.extra_cands.split(",") if c.strip()]
             if args.extra_cands else [])
    jobs = []
    for t in json.load(open(args.targets)):
        o = byname[t["det"]]
        cands = [c for c, s in det_cands.get(t["file"], []) if c != "clutter"]
        cands = cands[:4] + [c for c in extra if c not in cands[:4]]
        jobs.append({"key": t["kg"], "det": t["det"], "obj": o,
                     "cands": cands,
                     "views": views(os.path.join(args.det_dir, "views",
                                                 f2pre[t["file"]]))})
    # rescue target: the unverified monitor (large wall TV) — only when the
    # targets file doesn't already include it
    if "monitor_006" in byname and \
            not any(j["det"] == "monitor_006" for j in jobs):
        mon = byname["monitor_006"]
        jobs.append({"key": "monitor_006", "det": "monitor_006", "obj": mon,
                     "cands": ["tv", "monitor", "display panel", "whiteboard"],
                     "views": views(os.path.join(args.det_dir, "views",
                                                 f2pre["obj_0006.ply"]))})
    # Point-SAM sub-clusters
    if args.subs_dir and os.path.isdir(args.subs_dir):
        sub_cands = topk_from_csv(
            os.path.join(args.subs_dir, "classification.csv"))
        for f, cl in sorted(sub_cands.items()):
            name = f[len("obj_"):-len(".ply")]
            vdir = os.path.join(args.subs_dir, "views", name)
            if not os.path.isdir(vdir):
                continue
            parent = byname["_".join(name.split("_")[:-1])]
            jobs.append({"key": f"sub:{name}", "det": name,
                         "obj": {"type": "clutter", "dimensions": {},
                                 "properties": {
                                     "poseZ": parent["properties"]["poseZ"]}},
                         "cands": [c for c, s in cl if c != "clutter"][:4],
                         "views": views(vdir)})

    print(f"{len(jobs)} re-verification jobs, floor_z={floor_z:.3f}")
    if args.dry_run:
        for j in jobs:
            print(f"  {j['key']:24s} cands={j['cands']} "
                  f"views={len([v for v in j['views'] if os.path.exists(v)])}")
        return

    vlm = vb.SemanticVLM(model=args.model)
    results = []
    for j in jobs:
        content = vlm.build_symbolic_request(j["obj"], j["views"],
                                             allowed_types=j["cands"],
                                             floor_z=floor_z)
        r = vlm._call(_SYS_RECOVERY, content, vb.SYMBOLIC_TOOL)
        results.append({"key": j["key"], "det": j["det"],
                        "old_type": j["obj"].get("type"),
                        "candidates": j["cands"], **r})
        print(f"  {j['key']:24s} {j['obj'].get('type')} -> "
              f"{r['corrected_type']} ({r['confidence']:.2f}) "
              f"{r['reason'][:70]}")

    usage = vlm.usage_summary()
    json.dump({"model": args.model, "floor_z": floor_z, "usage": usage,
               "results": results}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}; cost ${usage['cost_usd']}")


if __name__ == "__main__":
    main()
