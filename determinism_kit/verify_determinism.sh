#!/bin/sh
# Runs the deterministic geometric backbone (Stage A) on an input .e57 and
# writes /data/determinism_report.json with SHA256 of every retained cluster.
#   usage: verify_determinism.sh [/data/input.e57] [/data/report_name.json]
set -eu

IN="${1:-/data/input.e57}"
REPORT="${2:-/data/determinism_report.json}"
WORK=/tmp/stageA_out

if [ ! -f "$IN" ]; then
    echo "ERROR: input scan not found: $IN" >&2
    echo "Mount the directory containing the .e57 at /data, e.g.:" >&2
    echo "  docker run --rm -v /path/to/dir:/data tlssmf-determinism \"/data/vis_n2 1.e57\"" >&2
    exit 2
fi

echo "[kit] input : $IN"
echo "[kit] report: $REPORT"
rm -rf "$WORK"

T0=$(date +%s)
python /opt/tlssmf/scripts/extract_objects.py --input "$IN" --out "$WORK"
T1=$(date +%s)

python /opt/tlssmf/make_report.py \
    --objects-dir "$WORK" --input "$IN" \
    --wall-seconds "$((T1 - T0))" --out "$REPORT"

echo "[kit] done. Compare with the reference:"
echo "      python3 compare_checksums.py reference_checksums.json determinism_report.json"
