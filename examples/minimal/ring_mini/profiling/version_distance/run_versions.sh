#!/usr/bin/env bash
#
# Sweep --dist from 1..DIST_MAX, capturing version activity via parse_versions.py.
#
# For each dist value we:
#   - run the binary with --info to get version event logs
#   - pipe output into parse_versions.py
#   - append the Global active-versions stats to a summary CSV
#   - save the per-rank detail CSV into a details/ dir
#
# Usage:
#   ./run_versions.sh [BINARY] [NRANKS] [DIST_MAX] [EXTRA_ARGS...]
#
# Defaults: ../../minimal, 32 ranks, dist 1..16

set -uo pipefail

BIN="${1:-./../../minimal}"
NRANKS="${2:-32}"
DIST_MAX="${3:-16}"
shift || true; shift || true; shift || true
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARSER="$SCRIPT_DIR/parse_versions.py"

PYTHON_BIN="$(command -v python3 || command -v python)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: no python found on PATH" >&2
    exit 1
fi
if [[ ! -f "$PARSER" ]]; then
    echo "ERROR: parser not found: $PARSER" >&2
    exit 1
fi
if [[ ! -x "$BIN" ]]; then
    echo "ERROR: binary not found or not executable: $BIN" >&2
    exit 1
fi

OUTDIR="$SCRIPT_DIR/results_$(date +%Y%m%d_%H%M%S)"
CSV="$OUTDIR/versions_summary.csv"
DETAILS_DIR="$OUTDIR/details"
mkdir -p "$DETAILS_DIR"

echo "binary    : $BIN"
echo "ranks     : $NRANKS"
echo "dist range: 1..$DIST_MAX"
echo "parser    : $PARSER"
echo "python    : $PYTHON_BIN"
echo "outdir    : $OUTDIR"
echo "extra args: ${EXTRA_ARGS[*]:-<none>}"
echo

echo "dist,nranks,v_min,v_max,v_avg,n_samples,status" > "$CSV"

for D in $(seq 1 "$DIST_MAX"); do
    echo "=== running --dist $D ==="

    local_workdir=$(mktemp -d "$OUTDIR/.vrun_XXXXXX")
    parser_log="$local_workdir/parser.out"

    mpirun -n "$NRANKS" "$BIN" --dist "$D" --info "${EXTRA_ARGS[@]}" \
        2>&1 \
        | (cd "$local_workdir" && "$PYTHON_BIN" "$PARSER" -) > "$parser_log" 2>&1
    pstat=( "${PIPESTATUS[@]}" )
    mpi_rc=${pstat[0]:-0}
    parse_rc=${pstat[1]:-0}

    status="ok"
    if (( mpi_rc != 0 )); then
        status="mpi_exit_${mpi_rc}"
    elif (( parse_rc != 0 )); then
        status="parse_exit_${parse_rc}"
    fi

    # Parse: Global active-versions: min=0 max=4 avg=1.23  (across 512 rank-tick samples)
    v_min="" v_max="" v_avg="" n_samples=""
    gline=$(grep -E '^Global active-versions:' "$parser_log" | tail -n1)
    if [[ -n "$gline" ]]; then
        v_min=$(  echo "$gline" | sed -nE 's/.*min=([0-9]+).*/\1/p')
        v_max=$(  echo "$gline" | sed -nE 's/.*max=([0-9]+).*/\1/p')
        v_avg=$(  echo "$gline" | sed -nE 's/.*avg=([0-9.]+).*/\1/p')
        n_samples=$(echo "$gline" | sed -nE 's/.*across ([0-9]+) rank-tick.*/\1/p')
        if [[ -z "$v_min" || -z "$v_max" || -z "$v_avg" ]]; then
            status="${status}+parse_global_fail"
        fi
    else
        if grep -q 'No version events found' "$parser_log"; then
            status="${status}+no_versions"
        else
            status="${status}+no_global_line"
        fi
    fi

    # Move per-rank detail CSV into details/ dir.
    per_rank_src="$local_workdir/stdin.versions.per_rank.csv"
    per_rank_dst="$DETAILS_DIR/dist_${D}.per_rank.csv"
    if [[ -f "$per_rank_src" ]]; then
        mv "$per_rank_src" "$per_rank_dst"
    fi
    rm -f "$local_workdir/stdin.versions.csv"

    echo "${D},${NRANKS},${v_min},${v_max},${v_avg},${n_samples},${status}" >> "$CSV"

    if [[ "$status" == "ok" ]]; then
        rm -rf "$local_workdir"
    else
        echo "  kept $local_workdir for inspection (status=$status)"
    fi
done

echo
echo "done."
echo "Summary CSV: $CSV"
echo "Details dir: $DETAILS_DIR"
echo
column -s, -t < "$CSV" || cat "$CSV"
