#!/usr/bin/env python3
"""Batch-render per-object view sets for Stage-B VLM verification.

Writes <out>/views/<object_name>/view_XXX.png for every object in a Stage-A
semanticObjects.json. XXX = azimuth in degrees (000/090/180/270 for the
default 4-view set; 45-degree steps with --views 8 for the escalation set).

Replaces the ad-hoc session script that produced outputs/s2_split_hdviews.
Defaults reproduce that HD convention: 512x512, elev 25, point_size 8.

Examples
  # standard 4-view HD set (Stage-B late fusion input)
  .venv/bin/python scripts/render_object_views.py \
      --semantic outputs/vis_n2_objects/semanticObjects.json \
      --objects-dir outputs/vis_n2_objects --out outputs/vis_n2_hdviews

  # escalation set: 8 azimuths + zoomed-in close-ups
  .venv/bin/python scripts/render_object_views.py \
      --semantic outputs/vis_n2_objects/semanticObjects.json \
      --objects-dir outputs/vis_n2_objects --out outputs/vis_n2_escviews \
      --views 8 --zoom 1.2 --suffix _z
"""
import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from render_instances import (_aabb_lines, _autocrop, _line_mat,  # noqa: E402
                              _pc_mat, _orbit_camera)
import open3d.visualization.rendering as rendering  # noqa: E402


def render_views(pc, azimuths, elev, size, point_size, zoom, box=True):
    pts = np.asarray(pc.points)
    imgs = []
    for az in azimuths:
        r = rendering.OffscreenRenderer(size, size)
        r.scene.set_background([1, 1, 1, 1])
        r.scene.scene.enable_sun_light(False)
        r.scene.add_geometry("o", pc, _pc_mat(point_size))
        allpts = [pts]
        if box:
            ls = _aabb_lines(pts)
            r.scene.add_geometry("b", ls, _line_mat(4.0))
            allpts.append(np.asarray(ls.points))
        ap = np.concatenate(allpts, 0)
        _orbit_camera(r, ap.mean(0), ap.max(0) - ap.min(0), az, elev, zoom=zoom)
        imgs.append((az, _autocrop(np.asarray(r.render_to_image()))))
        del r
    return imgs


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--semantic", required=True,
                    help="Stage-A semanticObjects.json (names + imageFile)")
    ap.add_argument("--objects-dir", required=True, help="dir with obj_*.ply")
    ap.add_argument("--out", required=True,
                    help="output root; views go to <out>/views/<name>/")
    ap.add_argument("--views", type=int, default=4, choices=(4, 8))
    ap.add_argument("--elev", type=float, default=25)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--point-size", type=float, default=8.0)
    ap.add_argument("--zoom", type=float, default=1.9,
                    help="camera distance factor (1.9 = default framing; "
                         "lower = closer, e.g. 1.2 for escalation close-ups)")
    ap.add_argument("--suffix", default="",
                    help="filename suffix, e.g. _z for zoom sets so they can "
                         "coexist with the normal set in one folder")
    ap.add_argument("--select", default=None,
                    help="comma-sep object names to render only")
    args = ap.parse_args()

    d = json.load(open(args.semantic))
    objs = d["semanticObjects"] if isinstance(d, dict) else d
    sel = set(args.select.split(",")) if args.select else None
    azimuths = list(range(0, 360, 360 // args.views))

    n_done = 0
    for o in objs:
        name = o["name"]
        if sel and name not in sel:
            continue
        ply = os.path.join(args.objects_dir,
                           o.get("properties", {}).get("imageFile",
                                                       f"obj_{o['id']:0>4}.ply"))
        if not os.path.exists(ply):
            print(f"[skip] {name}: {ply} missing", file=sys.stderr)
            continue
        pc = o3d.io.read_point_cloud(ply)
        if len(pc.points) == 0:
            print(f"[skip] {name}: empty cloud", file=sys.stderr)
            continue
        od = os.path.join(args.out, "views", name)
        os.makedirs(od, exist_ok=True)
        for az, img in render_views(pc, azimuths, args.elev, args.size,
                                    args.point_size, args.zoom):
            plt.imsave(os.path.join(od, f"view_{az:03d}{args.suffix}.png"), img)
        n_done += 1
        if n_done % 10 == 0:
            print(f"  {n_done} objects rendered...")
    print(f"done: {n_done} objects -> {args.out}/views/")


if __name__ == "__main__":
    main()
