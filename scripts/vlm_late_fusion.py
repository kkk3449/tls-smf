#!/usr/bin/env python3
"""Stage B, multi-view late fusion with automatic escalation.

Per object:
  1. Classify each base view independently (one forced-tool-use call per view).
  2. Confidence-weighted vote over the per-view types.
     - unanimous vote  -> status "verified"
  3. Split vote + --escalate -> automatic re-query on the escalation views
     (4 diagonal azimuths + 4 zoomed-in cardinals), then re-vote over ALL votes.
     - fused share >= --resolve-share -> status "verified_escalated"
     - else                           -> status "unverified" (kept, confidence
                                         exposed; no human review anywhere)
  4. One implicit-attribute call on the base views with the fused type.

Every record keeps properties.viewVotes / voteShare / verificationStatus, so
the knowledge-graph gate (upsert) can decide ingestion without re-running the
VLM. Usage/cost is logged next to the output json.

  .venv/bin/python scripts/vlm_late_fusion.py \
      --input outputs/vis_n2_objects/semanticObjects.json \
      --views-dir outputs/vis_n2_hdviews --escalate \
      --out outputs/vis_n2_objects/semanticObjects.lf_esc.json

Resume: --resume-from <prior lf json> reuses its base-view votes verbatim and
only runs what is missing (e.g. adding escalation to an earlier 4-view run).
"""
import argparse
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from blk360seg.vlm_stage_b import (SemanticVLM, apply_implicit,  # noqa: E402
                                   apply_symbolic, slug)

BASE_AZIMS = (0, 90, 180, 270)
ESC_AZIMS = (45, 135, 225, 315)          # extra orientations (normal zoom)
ESC_ZOOM_AZIMS = (0, 90, 180, 270)       # zoomed-in close-ups (suffix _z)


def view_path(views_dir, name, az, zoom=False):
    return os.path.join(views_dir, "views", name,
                        f"view_{az:03d}{'_z' if zoom else ''}.png")


def fuse(votes):
    """Confidence-weighted vote -> (fused_type, share, unanimous, score_table)."""
    score = {}
    for v in votes:
        t = v["type"].strip().lower()
        score[t] = score.get(t, 0.0) + float(v["confidence"])
    total = sum(score.values()) or 1.0
    top = max(score, key=score.get)
    share = score[top] / total
    unanimous = len(score) == 1
    return top, round(share, 3), unanimous, {k: round(v, 3) for k, v in score.items()}


def classify_view(vlm, obj, img, az, zoom, floor_z=None):
    r = vlm.verify_symbolic(obj, [img], floor_z=floor_z)
    return {"azimuth": az, "zoom": bool(zoom),
            "type": r["corrected_type"],
            "confidence": round(float(r["confidence"]), 3),
            "reason": r.get("reason", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--views-dir", required=True,
                    help="root containing views/<object_name>/view_XXX.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--escalate", action="store_true")
    ap.add_argument("--resolve-share", type=float, default=0.6,
                    help="post-escalation fused share needed for "
                         "verified_escalated (else unverified)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--select", default=None)
    ap.add_argument("--resume-from", default=None,
                    help="prior lf json; reuse its base viewVotes")
    ap.add_argument("--floor-z", type=float, default=None,
                    help="scene floor height in the record frame; enables "
                         "mounting-height context in the symbolic prompt")
    ap.add_argument("--workers", type=int, default=1,
                    help="objects processed in parallel (each object's own "
                         "calls stay sequential); one VLM client per worker")
    args = ap.parse_args()

    d = json.load(open(args.input))
    wrapped = isinstance(d, dict) and "semanticObjects" in d
    objs = copy.deepcopy(d["semanticObjects"] if wrapped else d)
    sel = set(args.select.split(",")) if args.select else None

    prior = {}
    if args.resume_from:
        pd = json.load(open(args.resume_from))
        pobjs = pd["semanticObjects"] if isinstance(pd, dict) and \
            "semanticObjects" in pd else pd
        prior = {o["id"]: o for o in pobjs}

    import threading
    from concurrent.futures import ThreadPoolExecutor

    # crash-safe checkpoint: every finished object is appended to
    # <out>.partial.jsonl; a restart skips those objects (no re-spend)
    part_path = args.out + ".partial.jsonl"
    ckpt = {}
    if os.path.exists(part_path):
        for line in open(part_path):
            try:
                r = json.loads(line)
                ckpt[str(r["id"])] = r
            except json.JSONDecodeError:
                pass  # torn last line from a crash
        print(f"[resume] {len(ckpt)} objects from {part_path}",
              file=sys.stderr)
    part_lock = threading.Lock()
    part_f = open(part_path, "a")

    work = []
    for obj in objs:
        if sel and obj["name"] not in sel and str(obj["id"]) not in sel:
            continue
        if args.limit is not None and len(work) >= args.limit:
            break
        c = ckpt.get(str(obj["id"]))
        if c is not None:
            obj.clear()
            obj.update(c)
            continue
        base_imgs = [view_path(args.views_dir, obj["name"], az)
                     for az in BASE_AZIMS]
        if not all(os.path.exists(p) for p in base_imgs):
            print(f"[skip] {obj['name']}: base views missing", file=sys.stderr)
            continue
        work.append((obj, base_imgs))

    tlocal = threading.local()
    lock = threading.Lock()
    clients = []
    stage_calls = {"base": 0, "escalation": 0, "implicit": 0}
    tallies = {"esc": 0, "unres": 0, "done": 0}

    def get_vlm():
        v = getattr(tlocal, "vlm", None)
        if v is None:
            v = SemanticVLM(model=args.model)
            tlocal.vlm = v
            with lock:
                clients.append(v)
        return v

    def process(item):
        obj, base_imgs = item
        vlm = get_vlm()
        name = obj["name"]
        calls = {"base": 0, "escalation": 0, "implicit": 0}

        po = prior.get(obj["id"], {}).get("properties", {})
        votes = [v for v in po.get("viewVotes", []) if not v.get("zoom")
                 and v.get("azimuth") in BASE_AZIMS][:4]
        if len(votes) != 4:
            votes = [classify_view(vlm, obj, p, az, False,
                                   floor_z=args.floor_z)
                     for az, p in zip(BASE_AZIMS, base_imgs)]
            calls["base"] = 4

        ftype, share, unanimous, score = fuse(votes)
        status, escalated = "verified", False
        if not unanimous:
            if args.escalate:
                escalated = True
                esc = ([(az, view_path(args.views_dir, name, az), False)
                        for az in ESC_AZIMS] +
                       [(az, view_path(args.views_dir, name, az, zoom=True),
                         True) for az in ESC_ZOOM_AZIMS])
                for az, p, zoom in esc:
                    if not os.path.exists(p):
                        print(f"[warn] {name}: missing esc view {p}",
                              file=sys.stderr)
                        continue
                    votes.append(classify_view(vlm, obj, p, az, zoom,
                                               floor_z=args.floor_z))
                    calls["escalation"] += 1
                ftype, share, unanimous, score = fuse(votes)
                status = ("verified_escalated" if share >= args.resolve_share
                          else "unverified")
            else:
                status = ("verified_majority" if share >= args.resolve_share
                          else "unverified")

        apply_symbolic(obj, {"corrected_type": ftype, "confidence": share,
                             "reason": f"late fusion of {len(votes)} views "
                                       f"(share {share}"
                                       f"{', unanimous' if unanimous else ''}"
                                       f"{', escalated' if escalated else ''})"})
        imp = vlm.infer_implicit(ftype, base_imgs)
        apply_implicit(obj, imp)
        calls["implicit"] = 1

        pr = obj["properties"]
        pr["viewVotes"] = votes
        pr["voteShare"] = share
        pr["voteScores"] = score
        pr["verificationStatus"] = status
        pr["escalated"] = escalated
        with part_lock:
            part_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            part_f.flush()
        with lock:
            for k in calls:
                stage_calls[k] += calls[k]
            tallies["esc"] += int(escalated)
            tallies["unres"] += int(status == "unverified")
            tallies["done"] += 1
            print(f"[{tallies['done']}/{len(work)}] {name}: {obj['type']} "
                  f"share={share} {status} ({len(votes)} votes)", flush=True)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(process, work))
    else:
        for item in work:
            process(item)

    part_f.close()
    out = {"semanticObjects": objs} if wrapped else objs
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    if os.path.exists(part_path):
        os.remove(part_path)  # run completed; checkpoint no longer needed
    usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
             "cache_read_tokens": 0, "cache_write_tokens": 0}
    cost = 0.0
    for v in clients:
        for k in usage:
            usage[k] += v.usage[k]
        cost += v.cost_usd()
    summary = {**usage, "cost_usd": round(cost, 4),
               "stage_calls": stage_calls, "objects": tallies["done"],
               "escalated": tallies["esc"], "unverified": tallies["unres"],
               "resolve_share": args.resolve_share, "model": args.model,
               "workers": args.workers}
    json.dump(summary, open(args.out.replace(".json", "") + ".usage.json", "w"),
              indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
