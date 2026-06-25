#!/usr/bin/env bash
#
# Sweep --dist from 1..16 with 32 ranks, capture logs, build a CSV.
#
# Usage:
#   ./run.sh [BINARY] [NRANKS] [DIST_MAX] [EXTRA_ARGS...]
#
# Defaults: ./mini, 32 ranks, dist 1..16

set -euo pipefail

BIN="${1:-./../../minimal}"
NRANKS="${2:-32}"
DIST_MAX="${3:-16}"
shift || true; shift || true; shift || true
EXTRA_ARGS=("$@")

OUTDIR="results_$(date +%Y%m%d_%H%M%S)"
LOGDIR="$OUTDIR/logs"
CSV="$OUTDIR/metrics.csv"
mkdir -p "$LOGDIR"

PY="$(dirname "$0")/parse_logs.py"

echo "binary    : $BIN"
echo "ranks     : $NRANKS"
echo "dist range: 1..$DIST_MAX"
echo "outdir    : $OUTDIR"
echo "extra args: ${EXTRA_ARGS[*]:-<none>}"
echo

# Write CSV header once (python script appends rows).
echo "dist,nranks,recv,isend,send_b,send_h,sent,delivered,forwarded,in_flight" > "$CSV"

for D in $(seq 1 "$DIST_MAX"); do
    LOG="$LOGDIR/dist_${D}.log"
    echo "=== running --dist $D ==="
    # Merge stderr→stdout so we capture whatever stream DSTM logs on.
    mpirun -n "$NRANKS" "$BIN" --dist "$D" "${EXTRA_ARGS[@]}" \
        > "$LOG" 2>&1 || {
            echo "  run failed (exit $?); continuing"
        }
    python3 "$PY" --dist "$D" --nranks "$NRANKS" --log "$LOG" --csv "$CSV"
done

echo
echo "done. CSV: $CSV"
column -s, -t < "$CSV" || cat "$CSV"
