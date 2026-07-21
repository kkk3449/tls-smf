#!/usr/bin/env python3
"""Diff-driven selective re-verification for incremental updates (Sec. 5.5).

After a re-scan, deterministic segmentation reproduces every untouched
cluster byte-identically. This script matches the new Stage-A records to a
previous verified run BY CLOUD CONTENT HASH and:
  - identical cluster -> carries the previous VLM verdict over verbatim
    (type, confidence, viewVotes, verificationStatus, implicit attributes);
  - changed / new cluster -> left provisional and listed for Stage-B.

So a re-scan only pays VLM cost proportional to the actual scene change.

  .venv/bin/python scripts/selective_reverify.py \
      --new outputs/showroom_rescan/semanticObjects.json \
      --new-objects-dir outputs/showroom_rescan \
      --prev outputs/showroom_det/semanticObjects.lf_esc.json \
      --prev-objects-dir outputs/showroom_det \
      --out outputs/showroom_rescan/semanticObjects.carried.json

Then run Stage-B only on the printed remainder:
  scripts/vlm_late_fusion.py ... --select <printed names> \
      --resume-from outputs/showroom_rescan/semanticObjects.carried.json
"""
import argparse
import hashlib
import json
import os


CARRY_PROPS = ("symbolicVerified", "symbolicReason", "viewVotes", "voteShare",
               "voteScores", "verificationStatus", "escalated", "isKeyObject",
               "isMovable", "isOpen", "canBeOpen", "implicitReason")


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def load(path):
    d = json.load(open(path))
    wrapped = isinstance(d, dict) and "semanticObjects" in d
    return d, (d["semanticObjects"] if wrapped else d), wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--new-objects-dir", required=True)
    ap.add_argument("--prev", required=True)
    ap.add_argument("--prev-objects-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    _, prev_objs, _ = load(args.prev)
    prev_by_hash = {}
    for o in prev_objs:
        f = os.path.join(args.prev_objects_dir,
                         o.get("properties", {}).get("imageFile", ""))
        if os.path.exists(f):
            prev_by_hash[md5(f)] = o

    doc, new_objs, wrapped = load(args.new)
    carried, pending = [], []
    for o in new_objs:
        f = os.path.join(args.new_objects_dir,
                         o.get("properties", {}).get("imageFile", ""))
        p = prev_by_hash.get(md5(f)) if os.path.exists(f) else None
        if p is None:
            pending.append(o["name"])
            continue
        o["type"] = p["type"]
        o["confidence"] = p["confidence"]
        keep_img = o["properties"]["imageFile"]
        o["name"] = p["name"]
        for k in CARRY_PROPS:
            if k in p.get("properties", {}):
                o["properties"][k] = p["properties"][k]
        o["properties"]["imageFile"] = keep_img
        o["properties"]["carriedOver"] = True
        carried.append(o["name"])

    json.dump(doc if wrapped else new_objs, open(args.out, "w"), indent=1,
              ensure_ascii=False)
    print(json.dumps({"carried_over": len(carried),
                      "needs_vlm": len(pending)}, indent=1))
    if pending:
        print("VLM select list:")
        print(",".join(pending))


if __name__ == "__main__":
    main()
