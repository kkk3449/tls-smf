#!/usr/bin/env python3
"""Hash every Stage A output cluster + record platform info -> report JSON.

Report fields:
  files            {filename: sha256} for obj_*.ply, objects.csv,
                   structure_remnants.csv (if present)
  aggregate_sha256 sha256 over the sorted "name:hash" lines -> one value to eyeball
  summary          cluster count, total retained points, dropped remnants
  platform         OS / CPU / glibc / python + package versions
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import platform
import sys


def sha256_file(path, buf=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def pkg_versions():
    from importlib.metadata import version
    out = {}
    for dist in ("numpy", "open3d", "pye57", "pandas", "PyYAML"):
        try:
            out[dist] = version(dist)
        except Exception as e:  # noqa: BLE001
            out[dist] = f"unavailable: {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects-dir", required=True)
    ap.add_argument("--input", required=True, help="the .e57 that was processed")
    ap.add_argument("--wall-seconds", type=int, default=-1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = {}
    for p in sorted(glob.glob(os.path.join(args.objects_dir, "obj_*.ply"))):
        files[os.path.basename(p)] = sha256_file(p)
    n_clusters = len(files)
    for extra in ("objects.csv", "structure_remnants.csv"):
        p = os.path.join(args.objects_dir, extra)
        if os.path.exists(p):
            files[extra] = sha256_file(p)

    total_points = 0
    with open(os.path.join(args.objects_dir, "objects.csv")) as f:
        for row in csv.DictReader(f):
            total_points += int(row["n_points"])

    n_dropped = 0
    rem = os.path.join(args.objects_dir, "structure_remnants.csv")
    if os.path.exists(rem):
        with open(rem) as f:
            n_dropped = sum(1 for _ in csv.DictReader(f))

    agg = hashlib.sha256()
    for name in sorted(files):
        agg.update(f"{name}:{files[name]}\n".encode())

    uname = platform.uname()
    try:
        libc = os.confstr("CS_GNU_LIBC_VERSION")
    except (ValueError, OSError):
        libc = " ".join(platform.libc_ver())

    report = {
        "kit": "TLS-SMF Stage A determinism verification",
        "input": {
            "file": os.path.basename(args.input),
            "size_bytes": os.path.getsize(args.input),
            "sha256": sha256_file(args.input),
        },
        "summary": {
            "n_clusters_retained": n_clusters,
            "n_soffit_remnants_dropped": n_dropped,
            "n_clusters_prefilter": n_clusters + n_dropped,
            "total_retained_points": total_points,
            "stageA_wall_seconds": args.wall_seconds,
        },
        "aggregate_sha256": agg.hexdigest(),
        "files": files,
        "platform": {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "cpu_model": cpu_model(),
            "cpu_count": os.cpu_count(),
            "glibc": libc,
            "python": sys.version.split()[0],
            "in_container": os.path.exists("/.dockerenv"),
            "packages": pkg_versions(),
        },
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[kit] report -> {args.out}")
    print(f"[kit] clusters retained: {n_clusters} "
          f"(pre-filter {n_clusters + n_dropped}, dropped {n_dropped}), "
          f"points {total_points:,}")
    print(f"[kit] aggregate_sha256: {report['aggregate_sha256']}")


if __name__ == "__main__":
    main()
