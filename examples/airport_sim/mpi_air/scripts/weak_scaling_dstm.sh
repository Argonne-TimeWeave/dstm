#!/usr/bin/env bash
# Multi-node weak-scaling sweep for the DSTM airports build only.
# Each rank owns a BLOCK_ROWS x BLOCK_COLS block of airports; total airports = block * P.
#
# For every run, the dump log is parsed by analyze_dump.py and the
# per-rank avg time and % of elapsed for get_or_default and put are
# appended as extra CSV columns.
#
# Assumes a Cobalt allocation (qsub -I -n $NODES -q skylake_8180 ...).
#
# Build expected at:  ../airports        (relative to this script)
# Analyzer expected at: ./analyze_dump.py (same dir as this script)
#
# Output: weak_scaling_dstm_<timestamp>.csv in this scripts/ directory.

set -u

# ---------- Config ----------
RANKS=(1 2 4 8 16 32 64 128 256)
NODES=8
BLOCK_ROWS=4
BLOCK_COLS=4
TIMESTEPS=32
TRIALS=2
WORK=100000
SEED=42

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIRPORTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$AIRPORTS_DIR/airports"
ANALYZER="$SCRIPT_DIR/analyze_dump.py"

# ---------- Sanity checks ----------
if [[ -z "${COBALT_NODEFILE:-}" ]]; then
    echo "ERROR: \$COBALT_NODEFILE not set. Run inside a Cobalt allocation." >&2
    exit 1
fi

ALLOCATED_NODES=$(wc -l < "$COBALT_NODEFILE")
if (( ALLOCATED_NODES < NODES )); then
    echo "ERROR: only $ALLOCATED_NODES node(s) allocated, need $NODES." >&2
    exit 1
fi

echo "Allocation: $ALLOCATED_NODES nodes"
cat "$COBALT_NODEFILE"
echo

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: missing or non-executable build: $BIN" >&2
    exit 1
fi
if [[ ! -f "$ANALYZER" ]]; then
    echo "ERROR: analyzer script not found: $ANALYZER" >&2
    exit 1
fi

MPI_PREFIX="$(dirname "$(dirname "$(which mpirun)")")"
echo "MPI prefix: $MPI_PREFIX"
echo "Binary:     $BIN"
echo "Analyzer:   $ANALYZER"
echo

TS="$(date +%Y%m%d_%H%M%S)"
CSV="$SCRIPT_DIR/weak_scaling_dstm_${TS}.csv"
LOG_DIR="$SCRIPT_DIR/logs_${TS}"
mkdir -p "$LOG_DIR"

# ---------- CSV header (time-only metrics) ----------
echo "ranks,ppn,nodes_used,block_rows,block_cols,total_airports,timesteps,trial,elapsed_s,steps_per_s,all_avg_s,all_pct" > "$CSV"

# Compute ranks-per-node and nodes-used for a given total rank count.
compute_placement() {
    local p="$1"
    if (( p < NODES )); then
        PPN=1
        NODES_USED=$p
    else
        PPN=$(( p / NODES ))
        NODES_USED=$NODES
        if (( p % NODES != 0 )); then
            echo "WARNING: P=$p not divisible by NODES=$NODES; ppr mapping may fail" >&2
        fi
    fi
}

# Extract one field from the analyzer's output.
# Usage: parse_analyzer "<analyzer stdout>" <metric> <field>
#   metric = all
#   field  = per-rank-avg  |  pct
# Returns empty string on failure.
parse_analyzer() {
    local out="$1" metric="$2" field="$3"
    if [[ "$metric" == "all_to_allv" ]]; then
        local line; line=$(echo "$out" | grep -E '^all_to_allv:')
    fi
    [[ -z "$line" ]] && { echo ""; return; }

    if [[ "$field" == "per-rank-avg" ]]; then
        # "... per-rank avg=0.1129 s, 86.83% of elapsed"
        echo "$line" | sed -n 's/.*per-rank avg=\([0-9.eE+-]*\) s.*/\1/p'
    else
        # pct
        echo "$line" | sed -n 's/.*s, \([0-9.]*\)% of elapsed.*/\1/p'
    fi
}

run_one() {
    local p="$1"
    local trial="$2"

    compute_placement "$p"

    local total_airports=$(( BLOCK_ROWS * BLOCK_COLS * p ))
    local logf="$LOG_DIR/dstm_p${p}_t${trial}.log"

    echo ">>> [dstm] ranks=$p ppn=$PPN nodes=$NODES_USED trial=$trial  (airports=$total_airports)"

    local map_arg=""
    if (( p > 1 )) && (( p % NODES_USED == 0 )); then
        map_arg="--map-by ppr:${PPN}:node"
    fi

    mpirun -n "$p" --hostfile "$COBALT_NODEFILE" \
        --prefix "$MPI_PREFIX" \
        $map_arg \
        -x LD_LIBRARY_PATH -x PATH \
        "$BIN" \
        --block-rows "$BLOCK_ROWS" \
        --block-cols "$BLOCK_COLS" \
        --time "$TIMESTEPS" \
        --seed "$SEED" \
        --work "$WORK" \
        --quiet \
        --logging \
        > "$logf" 2>&1
    local rc=$?

    local elapsed="" sps=""
    local allv_avg="" allv_pct="" 

    if [[ $rc -eq 0 ]]; then
        elapsed=$(grep -E '^Elapsed time:'     "$logf" | awk '{print $3}')
        sps=$(    grep -E '^Timesteps/second:' "$logf" | awk '{print $2}')

        # Run analyzer; pass --ranks explicitly so it doesn't depend on
        # the log header format.
        local analyzer_out
        if analyzer_out=$(python3 "$ANALYZER" "$logf" --ranks "$p" 2>/dev/null); then
            all_avg=$(parse_analyzer "$analyzer_out" all_to_allv per-rank-avg)
            all_pct=$(parse_analyzer "$analyzer_out" all_to_allv pct)
        else
            echo "  WARN: analyzer failed on $logf" >&2
        fi
    else
        echo "  WARN: mpirun exited with rc=$rc (see $logf)" >&2
    fi

    echo "${p},${PPN},${NODES_USED},${BLOCK_ROWS},${BLOCK_COLS},${total_airports},${TIMESTEPS},${trial},${elapsed},${sps},${all_avg},${all_pct}" >> "$CSV"
}

# ---------- Sweep ----------
for p in "${RANKS[@]}"; do
    for trial in $(seq 1 "$TRIALS"); do
        run_one "$p" "$trial"
    done
done

echo
echo "Done."
echo "CSV : $CSV"
echo "Logs: $LOG_DIR"
