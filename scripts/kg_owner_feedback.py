#!/usr/bin/env python3
"""Apply owner feedback + agreement-gated recovery labels as a KG revision.

Owner refutations and the recovery pass are treated as a first-class update
source, symmetric with a rescan: structure-noise refutations mark the node
absent (the cluster was wall/ceiling/outside noise, not an object), label
refutations and agreement-gated recoveries correct the type in place, and
every change is appended to the node's history under the new revision.
Edges of the previous revision are carried forward, dropping any that touch
a newly-absent node.

  .venv/bin/python scripts/kg_owner_feedback.py \
      --kg outputs/testroom_epochs_kg.json \
      --refuted outputs/owner_refuted.json \
      --adopted outputs/clutter_recovery_t3/recovery_adopted.json \
      --det-map outputs/clutter_recovery_targets_v2.json \
      --timestamp 2026-07-29T10:00:00
"""
import argparse
import copy
import json

# owner-refuted nodes that were real objects with a wrong label (stay
# present, type corrected) rather than structure noise (marked absent)
LABEL_CORRECTIONS = {"fire_extinguisher_051": "poster stand"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--refuted", required=True)
    ap.add_argument("--adopted", required=True)
    ap.add_argument("--det-map", required=True,
                    help="targets json mapping det names -> kg names")
    ap.add_argument("--timestamp", required=True)
    ap.add_argument("--out", default=None, help="default: in-place")
    args = ap.parse_args()

    g = json.load(open(args.kg))
    prev_rev = g["revision"]
    rev = prev_rev + 1
    byname = {}
    for n in g["nodes"]:
        byname.setdefault(n["name"], []).append(n)

    def node(name):
        cands = [n for n in byname.get(name, [])
                 if n.get("presence") != "absent"]
        assert len(cands) == 1, f"{name}: {len(cands)} present nodes"
        return cands[0]

    stats = {"absent": [], "label_corrected": [], "recovered": []}

    for r in json.load(open(args.refuted))["refuted"]:
        try:
            n = node(r["name"])
        except AssertionError:
            continue                       # unverified duplicates etc.
        hist = n.setdefault("history", [])
        if r["name"] in LABEL_CORRECTIONS:
            new_type = LABEL_CORRECTIONS[r["name"]]
            hist.append({"revision": rev, "change": "owner_label_correction",
                         "from": n["type"], "to": new_type,
                         "evidence": r["true_identity"]})
            n["type"] = new_type
            n["status"] = "verified_owner"
            n["confidence"] = 1.0
            stats["label_corrected"].append(r["name"])
        else:
            hist.append({"revision": rev, "change": "owner_refuted_absent",
                         "was": f"{n['type']} ({n['status']})",
                         "evidence": r["true_identity"]})
            n["presence"] = "absent"
            n["status"] = "refuted_structure"
            stats["absent"].append(r["name"])

    det2kg = {t["det"]: t["kg"]
              for t in json.load(open(args.det_map))}
    for det, a in json.load(open(args.adopted)).items():
        n = node(det2kg.get(det, det))
        n.setdefault("history", []).append(
            {"revision": rev, "change": "recovery_adopted",
             "from": n["type"], "to": a["type"], "rule": a["rule"]})
        n["type"] = a["type"]
        n["status"] = "verified_recovery"
        n["confidence"] = round(float(a["confidence"]), 3)
        stats["recovered"].append(f"{n['name']}->{a['type']}")

    # carry the previous revision's edges, minus any touching absent nodes
    present_ids = {n["id"] for n in g["nodes"]
                   if n.get("presence") != "absent"}
    carried = 0
    for e in [e for e in g["edges"] if e.get("revision") == prev_rev]:
        if e["subj"] in present_ids and e["obj"] in present_ids:
            e2 = copy.deepcopy(e)
            e2["revision"] = rev
            g["edges"].append(e2)
            carried += 1

    g["revision"] = rev
    g["updates"].append({
        "revision": rev, "timestamp": args.timestamp,
        "source": "owner_feedback+agreement_gated_recovery",
        "refuted_absent": len(stats["absent"]),
        "label_corrected": len(stats["label_corrected"]),
        "recovered": len(stats["recovered"]),
        "edges_carried": carried})
    json.dump(g, open(args.out or args.kg, "w"), indent=1)
    print(f"rev{rev}: absent {stats['absent']}")
    print(f"      corrected {stats['label_corrected']}")
    print(f"      recovered {stats['recovered']}")
    print(f"      edges carried {carried}")


if __name__ == "__main__":
    main()
