#!/usr/bin/env python3
"""Compare two determinism reports (reference vs. new run).

  python3 compare_checksums.py reference_checksums.json determinism_report.json

Exit code 0 = byte-identical clusters, 1 = any difference, 2 = usage error.
"""
import json
import sys


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a = json.load(open(sys.argv[1]))
    b = json.load(open(sys.argv[2]))

    print(f"reference : {a['platform']['cpu_model']} / glibc {a['platform']['glibc']} / "
          f"python {a['platform']['python']} / container={a['platform']['in_container']}")
    print(f"this run  : {b['platform']['cpu_model']} / glibc {b['platform']['glibc']} / "
          f"python {b['platform']['python']} / container={b['platform']['in_container']}")

    if a["input"]["sha256"] != b["input"]["sha256"]:
        print("\nWARNING: the input .e57 differs between the two runs "
              "(different scan file) — the comparison below is meaningless.")

    fa, fb = a["files"], b["files"]
    only_a = sorted(set(fa) - set(fb))
    only_b = sorted(set(fb) - set(fa))
    diff = sorted(k for k in set(fa) & set(fb) if fa[k] != fb[k])

    sa, sb = a["summary"], b["summary"]
    print(f"\nclusters  : ref {sa['n_clusters_retained']} "
          f"(pre-filter {sa['n_clusters_prefilter']}) vs "
          f"run {sb['n_clusters_retained']} (pre-filter {sb['n_clusters_prefilter']})")
    print(f"points    : ref {sa['total_retained_points']:,} vs run {sb['total_retained_points']:,}")
    print(f"aggregate : ref {a['aggregate_sha256'][:16]}... vs run {b['aggregate_sha256'][:16]}...")

    if not only_a and not only_b and not diff:
        print("\nRESULT: MATCH — every retained cluster (and manifest) is byte-identical.")
        return 0

    print("\nRESULT: MISMATCH")
    if only_a:
        print(f"  files only in reference ({len(only_a)}): {', '.join(only_a[:10])}"
              + (" ..." if len(only_a) > 10 else ""))
    if only_b:
        print(f"  files only in this run  ({len(only_b)}): {', '.join(only_b[:10])}"
              + (" ..." if len(only_b) > 10 else ""))
    if diff:
        print(f"  differing files ({len(diff)}): {', '.join(diff[:10])}"
              + (" ..." if len(diff) > 10 else ""))
    return 1


if __name__ == "__main__":
    sys.exit(main())
