#!/usr/bin/env python3
"""Gated vs. ungated vs. LLM-only querying benchmark (paper Sec. 5.4).

Generates a query set with programmatic ground truth from a GT-labeled
graph, then runs each query under three knowledge conditions:

  llm_only : the raw Stage-A record dump (provisional open-vocab labels,
             no verification, no relations) is the only context.
  ungated  : the TOSM graph serialized WITHOUT verification status —
             every label presented as fact.
  gated    : the TOSM graph WITH verificationStatus + confidence; the
             model is told unverified labels may be wrong and that it may
             abstain ("uncertain").

The answering model must reply through a forced tool call
(answer / abstain / confidence), so scoring is deterministic:
  correct        — answer matches GT
  wrong          — answer contradicts GT (counted as hallucination when
                   asserted with abstain=false)
  abstained      — model declined; never a hallucination
Per-type and aggregate accuracy / precision / recall / hallucination rate.

GT file: a json with {"objects": [{name, type(final GT), pose, dimensions,
color, isMovable, isKeyObject, [isOpen]}], "relations": [[subj, pred, obj]]}
— built separately from the human-audited labels.

  .venv/bin/python scripts/kg_query_benchmark.py \
      --gt outputs/showroom_gt.json \
      --stage-a outputs/stage2_no_wall_objects_split/semanticObjects.json \
      --graph outputs/showroom_kg.json \
      --out outputs/query_bench_showroom.json [--dry-run]
"""
import argparse
import copy
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ANSWER_TOOL = {
    "name": "report_answer",
    "description": "Report the answer to the environment query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string",
                       "description": "Direct answer. Counts: bare number. "
                                      "Yes/no: 'yes' or 'no'. Object: its "
                                      "name. List: comma-separated names. "
                                      "Dimension/position: numbers in meters "
                                      "as specified in the question."},
            "abstain": {"type": "boolean",
                        "description": "True if the knowledge provided is "
                                       "insufficient or too unreliable to "
                                       "answer confidently."},
            "confidence": {"type": "number"},
        },
        "required": ["answer", "abstain", "confidence"],
    },
}

_SYS = ("You answer questions about an indoor environment strictly from the "
        "provided knowledge context. Do not use outside knowledge about what "
        "rooms usually contain. If the context cannot support a confident "
        "answer, set abstain=true.")

_GATED_NOTE = ("\nNote: records carry verificationStatus. 'verified' and "
               "'verified_escalated' labels passed multi-view VLM "
               "verification. 'unverified' labels FAILED verification: the "
               "object exists at that pose with the measured geometry, but "
               "its TYPE LABEL is unreliable and often wrong. Rules: "
               "(1) never assert the presence, count, or identity of an "
               "object type when every record carrying that label is "
               "'unverified' — abstain or answer negatively instead; "
               "(2) geometry (pose, dimensions) of unverified records is "
               "still trustworthy and may be used.")


# ---------------------------------------------------------------- queries ---
def _fmt(v):
    return f"{v:.2f}"


COUNT_SKIP = {"clutter", "display", "tv", "monitor", "keyboard", "shelf",
              "ladder", "panel", "wall panel", "stair"}


def gen_queries(gt, trap_types=()):
    """~40+ queries with programmatic GT across the four types.

    GT objects carry gt_type + gt_quality (confirmed/plausible/unknown).
    Count/existence GT uses only types whose instances are ALL confirmed and
    that no unknown-GT object could plausibly be (COUNT_SKIP family).

    trap_types: labels that exist ONLY on unverified (verification-failed)
    records and are known-wrong or unconfirmable. A trap query asks about
    such a type; asserting its presence is a hallucination, while abstaining
    or denying is correct. These queries measure the value of gating."""
    objs = gt["objects"]
    rels = gt.get("relations", [])
    for o in objs:
        o.setdefault("type", o.get("gt_type"))
    by_type = {}
    for o in objs:
        by_type.setdefault(o["type"].lower(), []).append(o)
    qs = []

    # --- symbolic (labels / existence / counting) ---
    for t, members in sorted(by_type.items()):
        if t in COUNT_SKIP:
            continue
        if any(m.get("gt_quality") != "confirmed" for m in members):
            continue
        qs.append({"type": "symbolic",
                   "q": f"How many objects of type '{t}' are in the room? "
                        f"Answer with a bare number.",
                   "gt": str(len(members)), "kind": "count", "topic": t})
    present = sorted(t for t in by_type if t not in COUNT_SKIP
                     and all(m.get("gt_quality") == "confirmed"
                             for m in by_type[t]))
    absent = [t for t in ("sofa", "refrigerator", "bicycle", "printer",
                          "microwave oven", "potted plant")
              if t not in by_type][:4]
    for t in present[:4]:
        qs.append({"type": "symbolic",
                   "q": f"Is there a {t} in the room? Answer yes or no.",
                   "gt": "yes", "kind": "yesno", "topic": t})
    for t in absent:
        qs.append({"type": "symbolic",
                   "q": f"Is there a {t} in the room? Answer yes or no.",
                   "gt": "no", "kind": "yesno", "topic": t})

    # --- explicit (pose / dimensions / metric comparisons) ---
    tallest = max(objs, key=lambda o: o["dimensions"]["height"])
    qs.append({"type": "explicit",
               "q": "Which single object is the tallest? Answer with its name.",
               "gt": tallest["name"], "kind": "name", "topic": "tallest"})
    largest = max(objs, key=lambda o: o["dimensions"]["length"]
                  * o["dimensions"]["width"])
    qs.append({"type": "explicit",
               "q": "Which single object has the largest footprint "
                    "(length x width)? Answer with its name.",
               "gt": largest["name"], "kind": "name", "topic": "largest"})
    for o in objs[::max(1, len(objs) // 6)][:6]:
        qs.append({"type": "explicit",
                   "q": f"What is the height of {o['name']} in meters? "
                        f"Answer with a bare number.",
                   "gt": _fmt(o["dimensions"]["height"]), "kind": "number",
                   "topic": o["name"]})
    for o in objs[1::max(1, len(objs) // 4)][:4]:
        qs.append({"type": "explicit",
                   "q": f"What is the (x, y) position of {o['name']} in "
                        f"meters? Answer as 'x, y'.",
                   "gt": f"{_fmt(o['poseX'])}, {_fmt(o['poseY'])}",
                   "kind": "xy", "topic": o["name"]})

    # --- implicit (movability / key objects / openability) ---
    movs = [o for o in objs if o.get("isMovable") is True][:4]
    fixed = [o for o in objs if o.get("isMovable") is False][:4]
    for o in movs:
        qs.append({"type": "implicit",
                   "q": f"Is {o['name']} movable? Answer yes or no.",
                   "gt": "yes", "kind": "yesno", "topic": o["name"]})
    for o in fixed:
        qs.append({"type": "implicit",
                   "q": f"Is {o['name']} movable? Answer yes or no.",
                   "gt": "no", "kind": "yesno", "topic": o["name"]})
    keys = sorted(o["name"] for o in objs if o.get("isKeyObject"))
    if keys:
        qs.append({"type": "implicit",
                   "q": "List the names of all key objects (landmark-grade "
                        "objects for navigation), comma-separated.",
                   "gt": ",".join(keys), "kind": "set", "topic": "keyObjects"})

    # --- scene-level (relations / spatial aggregation) ---
    rel_by_subj = {}
    for s, p, o2 in rels:
        rel_by_subj.setdefault((s, p), []).append(o2)
    n_added = 0
    for (s, p), tgts in sorted(rel_by_subj.items()):
        if p == "isNextTo" and n_added < 4:
            qs.append({"type": "scene",
                       "q": f"Which objects are next to {s}? Answer with "
                            f"comma-separated names.",
                       "gt": ",".join(sorted(tgts)), "kind": "set",
                       "topic": s})
            n_added += 1
        if p in ("isOn", "isAboveOf") and n_added < 8:
            qs.append({"type": "scene",
                       "q": f"What is {s} on top of or above? Answer with "
                            f"the object name(s), comma-separated.",
                       "gt": ",".join(sorted(tgts)), "kind": "set",
                       "topic": s})
            n_added += 1
    # --- trap queries (gating value): unverified-only labels ---
    for t in trap_types:
        qs.append({"type": "trap",
                   "q": f"Is there a {t} in the room? Answer yes or no.",
                   "gt": "no-or-abstain", "kind": "trap", "topic": t})
        qs.append({"type": "trap",
                   "q": f"How many objects of type '{t}' are in the room? "
                        f"Answer with a bare number.",
                   "gt": "0-or-abstain", "kind": "trap", "topic": t})

    anchor = objs[len(objs) // 2]
    near = sorted(o["name"] for o in objs if o is not anchor and math.hypot(
        o["poseX"] - anchor["poseX"], o["poseY"] - anchor["poseY"]) <= 1.5)
    qs.append({"type": "scene",
               "q": f"Which objects are within 1.5 m (center distance, XY) of "
                    f"{anchor['name']}? Answer with comma-separated names.",
               "gt": ",".join(near) if near else "none", "kind": "set",
               "topic": anchor["name"]})
    return qs


# ---------------------------------------------------------------- scoring ---
def norm_name(s):
    return s.strip().lower().replace(" ", "_")


def score(q, resp):
    if q["kind"] == "trap":
        # correct = refusing to assert the phantom object (abstain, "no", "0",
        # "none"); asserting it ("yes" / a positive count) = hallucination
        if resp["abstain"]:
            return "correct"
        a = str(resp["answer"]).strip().lower()
        neg = a.startswith(("no", "none", "0", "uncertain", "unknown"))
        return "correct" if neg else "wrong"
    if resp["abstain"]:
        return "abstained"
    a = str(resp["answer"]).strip().lower()
    gt = q["gt"].lower()
    if q["kind"] == "yesno":
        ok = a.startswith(gt)
    elif q["kind"] in ("count",):
        ok = a.split()[0].rstrip(".") == gt if a else False
    elif q["kind"] == "number":
        try:
            ok = abs(float(a.split()[0].rstrip(".")) - float(gt)) <= \
                max(0.05, 0.1 * float(gt))
        except ValueError:
            ok = False
    elif q["kind"] == "xy":
        try:
            gx, gy = (float(v) for v in gt.split(","))
            parts = a.replace("(", " ").replace(")", " ").split(",")
            ax, ay = float(parts[0]), float(parts[1])
            ok = math.hypot(ax - gx, ay - gy) <= 0.3
        except (ValueError, IndexError):
            ok = False
    elif q["kind"] == "name":
        ok = norm_name(gt) in norm_name(a)
    elif q["kind"] == "set":
        gset = {norm_name(x) for x in gt.split(",") if x} - {"none"}
        aset = {norm_name(x) for x in a.split(",") if x.strip()} - {"none"}
        inter = len(gset & aset)
        prec = inter / len(aset) if aset else (1.0 if not gset else 0.0)
        rec = inter / len(gset) if gset else (1.0 if not aset else 0.0)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        ok = f1 >= 0.8
    else:
        ok = a == gt
    return "correct" if ok else "wrong"


# --------------------------------------------------------------- contexts ---
def ctx_llm_only(stage_a_records):
    recs = []
    for o in stage_a_records:
        recs.append({"name": o["name"], "type": o["type"],
                     "confidence": o.get("confidence"),
                     "poseX": o["poseX"], "poseY": o["poseY"],
                     "dimensions": o["dimensions"], "color": o.get("color")})
    return ("Raw object records from an automated scan pipeline "
            "(open-vocabulary classifier output, unverified):\n"
            + json.dumps(recs, ensure_ascii=False))


def ctx_graph(graph, gated):
    nodes = []
    for n in graph["nodes"]:
        if n.get("presence") == "absent":
            continue
        node = {"name": n["name"], "type": n["type"],
                "pose": n["pose"], "dimensions": n["dimensions"],
                "color": n.get("color"), **n.get("implicit", {})}
        if gated:
            node["verificationStatus"] = n["status"]
            node["confidence"] = n["confidence"]
        nodes.append(node)
    id2name = {n["id"]: n["name"] for n in graph["nodes"]}
    edges = [[id2name[e["subj"]], e["pred"], id2name[e["obj"]]]
             for e in graph["edges"]]
    txt = ("TOSM knowledge graph of the room.\nNodes:\n"
           + json.dumps(nodes, ensure_ascii=False)
           + "\nEdges (subject, predicate, object):\n"
           + json.dumps(edges, ensure_ascii=False))
    return txt + (_GATED_NOTE if gated else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--stage-a", required=True,
                    help="raw Stage-A semanticObjects.json (llm_only context)")
    ap.add_argument("--graph", required=True, help="kg_upsert graph json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--conditions", default="llm_only,ungated,gated")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the query set and exit (no API calls)")
    ap.add_argument("--trap-types", default=None,
                    help="comma-sep audit-established phantom labels")
    args = ap.parse_args()

    gt = json.load(open(args.gt))
    graph = json.load(open(args.graph))

    # trap types: labels established as wrong/unconfirmable by the human
    # audit that nonetheless appear on (unverified) records. Hand-curated —
    # auto-derivation confuses synonyms (display vs tv) with phantoms.
    trap = [t for t in (args.trap_types.split(",") if args.trap_types else [])
            if t]
    print(f"trap types: {trap}")

    queries = gen_queries(gt, trap_types=trap)
    print(f"{len(queries)} queries "
          f"({', '.join(t + ':' + str(sum(1 for q in queries if q['type'] == t)) for t in ('symbolic', 'explicit', 'implicit', 'scene'))})")
    if args.dry_run:
        for q in queries:
            print(f"  [{q['type']}] {q['q']}  => {q['gt']}")
        return

    d = json.load(open(args.stage_a))
    stage_a = d["semanticObjects"] if isinstance(d, dict) and \
        "semanticObjects" in d else d
    contexts = {"llm_only": ctx_llm_only(stage_a),
                "ungated": ctx_graph(graph, gated=False),
                "gated": ctx_graph(graph, gated=True)}

    from blk360seg.vlm_stage_b import SemanticVLM
    vlm = SemanticVLM(model=args.model, max_tokens=1024)

    def call_cached(ctx, question):
        """Context as a cached system block: 44 queries share one prefix."""
        resp = vlm.client.messages.create(
            model=vlm.model, max_tokens=vlm.max_tokens,
            system=[{"type": "text", "text": _SYS},
                    {"type": "text", "text": ctx,
                     "cache_control": {"type": "ephemeral"}}],
            tools=[ANSWER_TOOL],
            tool_choice={"type": "tool", "name": ANSWER_TOOL["name"]},
            messages=[{"role": "user", "content": question}],
        )
        u = resp.usage
        vlm.usage["calls"] += 1
        vlm.usage["input_tokens"] += u.input_tokens or 0
        vlm.usage["output_tokens"] += u.output_tokens or 0
        vlm.usage["cache_read_tokens"] += \
            getattr(u, "cache_read_input_tokens", 0) or 0
        vlm.usage["cache_write_tokens"] += \
            getattr(u, "cache_creation_input_tokens", 0) or 0
        for block in resp.content:
            if block.type == "tool_use":
                return dict(block.input)
        raise RuntimeError("no tool_use in response")

    # crash-safe checkpoint: each answered query appended to a partial file
    part_path = args.out + ".partial.jsonl"
    done = {}
    if os.path.exists(part_path):
        for line in open(part_path):
            try:
                r = json.loads(line)
                done[(r["condition"], r["qi"])] = r
            except json.JSONDecodeError:
                pass
        print(f"[resume] {len(done)} answers from {part_path}")
    part_f = open(part_path, "a")

    results = []
    for cond in args.conditions.split(","):
        ctx = contexts[cond]
        for i, q in enumerate(queries):
            r = done.get((cond, i))
            if r is None:
                resp = call_cached(ctx, "Question: " + q["q"])
                verdict = score(q, resp)
                r = {**{k: q[k] for k in ("type", "q", "gt", "kind")},
                     "qi": i, "condition": cond, "answer": resp["answer"],
                     "abstain": resp["abstain"],
                     "confidence": resp.get("confidence"),
                     "verdict": verdict}
                part_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                part_f.flush()
            results.append(r)
            print(f"[{cond} {i + 1}/{len(queries)}] {q['type']:>8} "
                  f"{r['verdict']:<9} {q['q'][:60]}", flush=True)

    summary = {}
    for cond in args.conditions.split(","):
        rs = [r for r in results if r["condition"] == cond]
        n = len(rs)
        c = sum(r["verdict"] == "correct" for r in rs)
        w = sum(r["verdict"] == "wrong" for r in rs)
        a = sum(r["verdict"] == "abstained" for r in rs)
        answered = c + w
        summary[cond] = {
            "n": n, "correct": c, "wrong": w, "abstained": a,
            "accuracy": round(c / n, 3),
            "precision_answered": round(c / answered, 3) if answered else None,
            "recall": round(c / n, 3),
            "hallucination_rate": round(w / n, 3),
            "per_type": {t: {
                "n": len([r for r in rs if r["type"] == t]),
                "correct": sum(r["verdict"] == "correct" for r in rs
                               if r["type"] == t),
                "wrong": sum(r["verdict"] == "wrong" for r in rs
                             if r["type"] == t)}
                for t in ("symbolic", "explicit", "implicit", "scene", "trap")},
        }
    part_f.close()
    json.dump({"summary": summary, "results": results,
               "usage": vlm.usage_summary()},
              open(args.out, "w"), indent=1, ensure_ascii=False)
    if os.path.exists(part_path):
        os.remove(part_path)
    print(json.dumps({"summary": summary,
                      "cost_usd": vlm.usage_summary()["cost_usd"]}, indent=1))


if __name__ == "__main__":
    main()
