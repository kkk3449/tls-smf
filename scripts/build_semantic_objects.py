#!/usr/bin/env python3
"""Stage A — build the VDA5050 semanticObject JSON from DBSCAN+Uni3D output.

Deterministic: fills explicit (pose/size/color) + provisional symbolic
(type/name/id). Implicit fields are added later by the VLM stage (Stage B).

  CUDA_HOME=/home/caselab/cuda128 \
  python scripts/build_semantic_objects.py \
      --objects-dir outputs/stage2_no_wall_objects \
      --classification outputs/stage2_no_wall_objects/classification.csv \
      --candidate-names "chair,TV,monitor,Mobile Robot,desk" \
      --render-views

`--candidate-names` is OPTIONAL — the representative class set, recorded as
`allowedTypes` in the message and used by Stage B to constrain the VLM's
corrected type. Leave it out (blank) to stay fully open-vocabulary.

Writes <objects-dir>/semanticObjects.json and, with --render-views,
<objects-dir>/views/<name>/view_<azim>.png (4 views per object for the VLM).
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from blk360seg import semantic_object as so                      # noqa: E402


def parse_candidates(s, path):
    names = []
    if path and os.path.exists(path):
        with open(path) as f:
            names += [ln.strip() for ln in f if ln.strip()]
    if s:
        names += [t.strip() for t in s.split(",") if t.strip()]
    # dedup preserving order
    seen, out = set(), []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects-dir", required=True)
    ap.add_argument("--classification", default=None,
                    help="Uni3D classification.csv (provisional type/confidence)")
    ap.add_argument("--candidate-names", default=None,
                    help='representative class set, e.g. "chair,TV,monitor,Mobile Robot,desk"')
    ap.add_argument("--candidate-names-file", default=None,
                    help="text file, one candidate name per line")
    ap.add_argument("--map-id", default=None)
    ap.add_argument("--coordinate-frame", default="map")
    ap.add_argument("--manufacturer", default="CASELAB")
    ap.add_argument("--serial", default="SERIALNUMBER")
    ap.add_argument("--version", default="0.0.0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--render-views", action="store_true",
                    help="render 4 azimuth views per object for the VLM stage")
    ap.add_argument("--views", type=int, nargs="+", default=[0, 90, 180, 270])
    ap.add_argument("--view-size", type=int, default=512)
    args = ap.parse_args()

    candidates = parse_candidates(args.candidate_names, args.candidate_names_file)
    print(f"[A] candidate names ({len(candidates)}): {candidates or '(none -> open-vocab)'}")

    classification = args.classification
    if classification is None:
        guess = os.path.join(args.objects_dir, "classification.csv")
        classification = guess if os.path.exists(guess) else None
    print(f"[A] classification: {classification or '(none -> empty type)'}")

    objs = so.build_objects(args.objects_dir, classification=classification,
                            map_id=args.map_id, coordinate_frame=args.coordinate_frame)
    print(f"[A] built {len(objs)} semantic objects")

    # optional 4-view render per object (for VLM)
    if args.render_views:
        import open3d as o3d
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from render_instances import render_object_views
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        vroot = os.path.join(args.objects_dir, "views")
        for obj in objs:
            f = obj["properties"]["imageFile"]
            pc = o3d.io.read_point_cloud(os.path.join(args.objects_dir, f))
            vd = os.path.join(vroot, obj["name"])
            os.makedirs(vd, exist_ok=True)
            imgs = render_object_views(pc, azimuths=tuple(args.views),
                                       size=args.view_size)
            paths = []
            for az, im in zip(args.views, imgs):
                p = os.path.join(vd, f"view_{int(az):03d}.png")
                plt.imsave(p, im)
                paths.append(os.path.relpath(p, args.objects_dir))
            obj["properties"]["imageViews"] = paths
        print(f"[A] rendered {len(objs)}x{len(args.views)} views -> {vroot}/")

    msg = so.build_message(objs, candidate_names=candidates,
                           manufacturer=args.manufacturer, serial=args.serial,
                           version=args.version)
    out = args.out or os.path.join(args.objects_dir, "semanticObjects.json")
    with open(out, "w") as f:
        json.dump(msg, f, indent=2, ensure_ascii=False)
    print(f"[A] wrote {out}  ({len(objs)} objects)")
    # quick peek
    import collections
    cc = collections.Counter(o["type"] for o in objs)
    print("[A] provisional type counts:", dict(cc.most_common()))


if __name__ == "__main__":
    main()
