#!/usr/bin/env python3
"""Stage B — VLM-refine semanticObjects.json with Claude Sonnet 4.6.

Per object: prompt 1 (verify/correct type) -> prompt 2 (infer implicit model),
both forced-tool-use on the object's 4-view render.

  ANTHROPIC_API_KEY=sk-... \
  python scripts/vlm_annotate.py \
      --input outputs/stage2_no_wall_objects/semanticObjects.json \
      --objects-dir outputs/stage2_no_wall_objects

Options:
  --dry-run     build + print the request for object 0, call NO API (no key needed)
  --limit N     only process the first N objects (cost control)
  --model       default claude-sonnet-4-6
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from blk360seg import vlm_stage_b as vb                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="semanticObjects.json from Stage A")
    ap.add_argument("--objects-dir", required=True, help="dir holding views/ (image paths are relative to it)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--select", default=None,
                    help="comma-sep names/ids/imageFiles to process only (e.g. stair_008,obj_0001.ply)")
    ap.add_argument("--num-views", type=int, default=1,
                    help="how many of the rendered views to send per object "
                         "(1 for preliminary/예심, 4 for final/본심). Default 1.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    msg = json.load(open(args.input))
    objs = msg["semanticObjects"]
    allowed = msg.get("allowedTypes")
    if args.select:
        want = {s.strip() for s in args.select.split(",") if s.strip()}
        objs = [o for o in objs
                if want & {o.get("name"), o.get("id"),
                           o.get("properties", {}).get("imageFile")}]
        print(f"[B] selected {len(objs)} objects matching {want}")
    if args.limit:
        objs = objs[:args.limit]
    print(f"[B] {len(objs)} objects | allowedTypes={allowed or '(open-vocab)'} | model={args.model}")

    def img_paths(obj):
        views = obj.get("properties", {}).get("imageViews", [])[:args.num_views]
        return [os.path.join(args.objects_dir, v) for v in views]

    if args.dry_run:
        # build + show the prompt-1 request for the first object, call nothing
        vlm = vb.SemanticVLM.__new__(vb.SemanticVLM)   # no client/key needed
        o = objs[0]
        paths = img_paths(o)
        content = vb.SemanticVLM.build_symbolic_request(vlm, o, paths, allowed)
        preview = [("image" if b.get("type") == "image" else b.get("text")) for b in content]
        print("\n=== DRY RUN — prompt 1 (symbolic) for object 0 ===")
        print("system:\n", vb._SYS_SYMBOLIC[:400], "...")
        print("\nuser content blocks:", preview)
        print("\ntools:", [vb.SYMBOLIC_TOOL["name"], vb.IMPLICIT_TOOL["name"]],
              "| tool_choice = forced")
        print(f"\nimages found for object 0: {len(paths)} -> {[os.path.basename(p) for p in paths]}")
        missing = [p for p in paths if not os.path.exists(p)]
        print("missing image files:", missing or "none")
        return

    vlm = vb.SemanticVLM(model=args.model)
    for i, o in enumerate(objs):
        paths = img_paths(o)
        if not paths:
            print(f"  [{i}] {o['name']}: no images, skipped")
            continue
        before = o["type"]
        s = vlm.verify_symbolic(o, paths, allowed)
        vb.apply_symbolic(o, s)
        im = vlm.infer_implicit(o["type"], paths)
        vb.apply_implicit(o, im)
        flag = f"  (was '{before}')" if s.get("changed") else ""
        print(f"  [{i}] {o['name']:<22} type={o['type']:<14} conf={o['confidence']:.2f}{flag} "
              f"| key={o['properties']['isKeyObject']} movable={o['properties']['isMovable']}")

    out = args.out or args.input.replace(".json", ".annotated.json")
    with open(out, "w") as f:
        json.dump(msg, f, indent=2, ensure_ascii=False)
    print(f"[B] wrote {out}")


if __name__ == "__main__":
    main()
