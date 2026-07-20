#!/usr/bin/env python3
"""Convert a unified TOSM graph JSON to a USD (.usda) scene for Isaac Sim.

  build_usd.py <tosm_graph.json> [out.usda]

Text-USD output (no pxr needed). Open the .usda in Isaac Sim or usdview.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blk360seg import usd_export as ux  # noqa: E402


def load_clouds(graph, objects_dir, max_pts=4000):
    """Load each object's segmented point cloud (xyz, rgb) from its imageFile PLY."""
    import numpy as np
    import open3d as o3d
    clouds = {}
    for o in graph["objects"]:
        f = o.get("properties", {}).get("imageFile")
        if not f:
            continue
        p = os.path.join(objects_dir, f)
        if not os.path.exists(p):
            continue
        pc = o3d.io.read_point_cloud(p)
        xyz = np.asarray(pc.points)
        rgb = np.asarray(pc.colors) if pc.has_colors() else \
            np.full_like(xyz, 0.6)
        if len(xyz) > max_pts:                       # cap for file size
            idx = np.linspace(0, len(xyz) - 1, max_pts).astype(int)
            xyz, rgb = xyz[idx], rgb[idx]
        clouds[o["name"]] = (xyz, rgb)
    return clouds


def load_environment(graph, ply, voxel=0.05, margin=3.0,
                     z_range=(-1.1, 2.2), max_pts=200000,
                     exclude_names=()):
    """Raw E57 background (walls+floor): crop to the object extent + margin,
    voxel-downsample, return (xyz, rgb). Drops the through-window outliers."""
    import numpy as np
    import open3d as o3d
    pc = o3d.io.read_point_cloud(ply)
    if voxel > 0:
        pc = pc.voxel_down_sample(voxel)
    xyz = np.asarray(pc.points)
    rgb = np.asarray(pc.colors) if pc.has_colors() else np.full_like(xyz, 0.6)
    xs = [o["poseX"] for o in graph["objects"]]
    ys = [o["poseY"] for o in graph["objects"]]
    x0, x1 = min(xs) - margin, max(xs) + margin
    y0, y1 = min(ys) - margin, max(ys) + margin
    m = ((xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) & (xyz[:, 1] >= y0)
         & (xyz[:, 1] <= y1) & (xyz[:, 2] >= z_range[0]) & (xyz[:, 2] <= z_range[1]))
    xyz, rgb = xyz[m], rgb[m]
    for o in graph["objects"]:          # e.g. the robot itself caught in the scan
        if o["name"] not in exclude_names:
            continue
        d = o.get("dimensions", {})
        hx = (max(d.get("length", 0), d.get("width", 0)) / 2 + 0.15)
        hz = d.get("height", 2.0) / 2 + 0.25
        cz = o.get("properties", {}).get("poseZ", 0.0)
        keep = ~((np.abs(xyz[:, 0] - o["poseX"]) < hx)
                 & (np.abs(xyz[:, 1] - o["poseY"]) < hx)
                 & (np.abs(xyz[:, 2] - cz) < hz))
        print(f"[USD] excluded {int((~keep).sum())} env pts inside {o['name']}")
        xyz, rgb = xyz[keep], rgb[keep]
    if len(xyz) > max_pts:
        idx = np.linspace(0, len(xyz) - 1, max_pts).astype(int)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz, rgb


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--points", default=None,
                    help="objects dir with per-object PLYs; embed real "
                         "segmented geometry instead of placeholder cubes")
    ap.add_argument("--environment", default=None,
                    help="raw E57/PLY (walls+floor) to embed as background")
    ap.add_argument("--env-voxel", type=float, default=0.05)
    ap.add_argument("--env-max-pts", type=int, default=200000)
    ap.add_argument("--env-z", nargs=2, type=float, default=[-1.1, 2.2],
                    help="environment z crop in the input cloud frame "
                         "(raise the max to keep the ceiling)")
    ap.add_argument("--floor-offset", type=float, default=1.014,
                    help="added to embedded point z (E57 floor -> z=0)")
    ap.add_argument("--point-width", type=float, default=0.02,
                    help="rendered point size (m); ~0.8x the cloud voxel size")
    ap.add_argument("--obj-max-pts", type=int, default=4000,
                    help="per-object embedded point cap")
    ap.add_argument("--exclude-object", action="append", default=[],
                    help="object name(s) to OMIT entirely (prim + env points), "
                         "e.g. the robot itself caught in the scan")
    args = ap.parse_args()

    out = args.out or args.graph.replace(".json", ".usda")
    graph = json.load(open(args.graph))
    env = load_environment(graph, args.environment, voxel=args.env_voxel,
                           z_range=tuple(args.env_z),
                           max_pts=args.env_max_pts,
                           exclude_names=set(args.exclude_object)) \
        if args.environment else None
    if args.exclude_object:              # drop the prim(s) too
        n0 = len(graph["objects"])
        graph["objects"] = [o for o in graph["objects"]
                            if o["name"] not in set(args.exclude_object)]
        print(f"[USD] excluded objects: {n0 - len(graph['objects'])}")
    clouds = load_clouds(graph, args.points, max_pts=args.obj_max_pts) \
        if args.points else None
    usda = ux.to_usda(graph, clouds=clouds, environment=env,
                      floor_offset=args.floor_offset
                      if (clouds or env) else 0.0,
                      point_width=args.point_width)
    with open(out, "w") as f:
        f.write(usda)
    nobj = len(graph["objects"])
    nplace = len(graph.get("places", []))
    nrel = len(graph.get("relations", []))
    mode = f"points ({sum(len(c[0]) for c in clouds.values())} pts)" \
        if clouds else "cube placeholders"
    envtxt = f" + environment ({len(env[0])} pts)" if env is not None else ""
    print(f"[USD] {nobj} objects + {nplace} places + {nrel} relations "
          f"[{mode}{envtxt}] -> {out}")


if __name__ == "__main__":
    main()
