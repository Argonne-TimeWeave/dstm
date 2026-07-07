#!/usr/bin/env bash
# Weak-scaling sweep comparing:
#   - DSTM airport sim (BSP)
#   - ROSS airport sim, conservative  (--synch=2 --cons-lookahead=1e-9)
#   - ROSS airport sim, optimistic    (--synch=3)
#
# Per-rank airport count (--na) is held constant across all P.
# Total airport count grows linearly: TOTAL_AIRPORTS = AIRPORTS_PER_RANK * P.

set -u

# ---------- Config ----------
RANKS=(2 4 8 16 32 64 128 256)
NODES=8
AIRPORTS_PER_RANK=64
TIMESTEPS=64
TRIALS=2
WORK=25000
SEED=42
LAYOVER=3
EXTRAMEM=200000
CONS_LOOKAHEAD=0.000005

# When set to 1 (or pass --keep-logs), save every run's stdout+stderr.
KEEP_LOGS=0
for a in "$@"; do
    [[ "$a" == "--keep-logs" ]] && KEEP_LOGS=1
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

declare -A BUILDS=(
    [dstm]="$EXAMPLES_DIR/airports/airports"
    [ross]="airport"
)

# We iterate engines in this order for each (P, trial): dstm, ross-cons, ross-opt.
# Format: "<name>:<sync_mode>"  (sync_mode is only used for ross)
ENGINE_ORDER=(
    "dstm:bsp"
#    "ross:cons"
    "ross:opt"
)

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

for spec in "${ENGINE_ORDER[@]}"; do
    name="${spec%%:*}"
    bin="${BUILDS[$name]}"
    if [[ ! -x "$bin" ]]; then
        echo "ERROR: missing or non-executable build: $name -> $bin" >&2
        exit 1
    fi
done

MPI_PREFIX="$(dirname "$(dirname "$(which mpirun)")")"
echo "MPI prefix              : $MPI_PREFIX"
echo "Airports per rank (fix) : $AIRPORTS_PER_RANK"
echo "Timesteps / end-time    : $TIMESTEPS"
echo "Work iters/departure    : $WORK"
echo "Trials per point        : $TRIALS"
echo "Rank sweep              : ${RANKS[*]}"
echo "Cons lookahead          : $CONS_LOOKAHEAD"
echo "Keep per-run logs       : $KEEP_LOGS"
echo

factor_mesh() {
    local n="$1"
    local best_r=1 r
    for (( r=1; r*r<=n; r++ )); do
        if (( n % r == 0 )); then best_r=$r; fi
    done
    ROWS=$best_r
    COLS=$(( n / best_r ))
}

# ---------- Output filenames ----------
RANKS_MIN=${RANKS[0]}
RANKS_MAX=${RANKS[-1]}
RANKS_COUNT=${#RANKS[@]}

BASE="weak_n${NODES}_apr${AIRPORTS_PER_RANK}_t${TIMESTEPS}_w${WORK}_tr${TRIALS}_s${SEED}_r${RANKS_MIN}-${RANKS_MAX}x${RANKS_COUNT}"
CSV="$SCRIPT_DIR/${BASE}.csv"

if [[ -e "$CSV" ]]; then
    i=2
    while [[ -e "$SCRIPT_DIR/${BASE}_${i}.csv" ]]; do i=$(( i + 1 )); done
    CSV="$SCRIPT_DIR/${BASE}_${i}.csv"
fi

LOG_DIR="$SCRIPT_DIR/${BASE}_logs"
mkdir -p "$LOG_DIR"

echo "CSV output : $CSV"
echo "Log dir    : $LOG_DIR (populated on failure or --keep-logs)"

# NEW: 'sync' column added right after 'build'
echo "build,sync,ranks,ppn,nodes_used,na,total_airports,timesteps,work,trial,elapsed_s,steps_per_s,departures,arrivals,connections,ross_events,ross_rollbacks,ross_efficiency,status" > "$CSV"

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

maybe_keep_log() {
    local tmpf="$1" name="$2" sync="$3" p="$4" trial="$5" status="$6"
    if [[ "$KEEP_LOGS" == "1" ]] || [[ "$status" != "ok" ]]; then
        local dest="$LOG_DIR/${name}_${sync}_p${p}_trial${trial}_${status}.log"
        cp "$tmpf" "$dest"
        echo "    (log saved: $dest)"
    fi
    rm -f "$tmpf"
}

# --------------------------------------------------------------------------
run_dstm() {
    local bin="$1"; local p="$2"; local trial="$3"; local total="$4"
    local sync="bsp"

    echo ">>> [dstm/${sync}] ranks=$p ppn=$PPN nodes=$NODES_USED trial=$trial  (na=$AIRPORTS_PER_RANK total=$total)"

    local map_arg=""
    if (( p > 1 )) && (( p % NODES_USED == 0 )); then
        map_arg="--map-by ppr:${PPN}:node"
    fi

    local tmpf; tmpf=$(mktemp)

    mpirun -n "$p" --hostfile "$COBALT_NODEFILE" \
        --prefix "$MPI_PREFIX" \
        $map_arg \
        -x LD_LIBRARY_PATH -x PATH \
        "$bin" \
        --na "$AIRPORTS_PER_RANK" \
        --time "$TIMESTEPS" \
        --seed "$SEED" \
        --work "$WORK" \
        --quiet \
        > "$tmpf" 2>&1
    local rc=$?

    local elapsed="" sps="" dep="" arr="" conn="" status="ok"

    if [[ $rc -ne 0 ]]; then
        status="exit_${rc}"
    else
        elapsed=$(grep -E '^Elapsed time:'      "$tmpf" | awk '{print $3}')
        sps=$(    grep -E '^Timesteps/second:'  "$tmpf" | awk '{print $2}')
        dep=$(    grep -E '^Total departures:'  "$tmpf" | awk '{print $3}')
        arr=$(    grep -E '^Total arrivals:'    "$tmpf" | awk '{print $3}')
        conn=$(   grep -E '^Total connections:' "$tmpf" | awk '{print $3}')
        [[ -z "$elapsed" ]] && status="parse_fail"
    fi

    echo "dstm,${sync},${p},${PPN},${NODES_USED},${AIRPORTS_PER_RANK},${total},${TIMESTEPS},${WORK},${trial},${elapsed},${sps},${dep},${arr},${conn},,,,${status}" >> "$CSV"

    maybe_keep_log "$tmpf" "dstm" "$sync" "$p" "$trial" "$status"
}

# --------------------------------------------------------------------------
# run_ross <bin> <p> <trial> <total> <rows> <cols> <sync_mode>
#   sync_mode ∈ {cons, opt}
# --------------------------------------------------------------------------
run_ross() {
    local bin="$1"; local p="$2"; local trial="$3"
    local total="$4"; local rows="$5"; local cols="$6"
    local sync="$7"

    local synch_flag=""
    local extra_flags=()
    case "$sync" in
        cons)
            synch_flag="--synch=2"
            extra_flags=( "--cons-lookahead=${CONS_LOOKAHEAD}" )
            ;;
        opt)
            synch_flag="--synch=3"
            extra_flags=( "--extramem=${EXTRAMEM}" )
            ;;
        *)
            echo "ERROR: unknown ROSS sync mode '$sync'" >&2
            return 1
            ;;
    esac

    echo ">>> [ross/${sync}] ranks=$p ppn=$PPN nodes=$NODES_USED trial=$trial  (nairports=$total mesh=${rows}x${cols})"

    local map_arg=""
    if (( p > 1 )) && (( p % NODES_USED == 0 )); then
        map_arg="--map-by ppr:${PPN}:node"
    fi

    local tmpf; tmpf=$(mktemp)

    mpirun -n "$p" --hostfile "$COBALT_NODEFILE" \
        --prefix "$MPI_PREFIX" \
        $map_arg \
        -x LD_LIBRARY_PATH -x PATH \
        "$bin" \
        "$synch_flag" \
        "${extra_flags[@]}" \
        --na="$AIRPORTS_PER_RANK" \
        --rows="$rows" \
        --cols="$cols" \
        --layover="$LAYOVER" \
        --end="$TIMESTEPS" \
        --work="$WORK" \
        --report-interval=0 \
        > "$tmpf" 2>&1
    local rc=$?

    local elapsed="" sps="" dep="" arr="" conn="" status="ok"
    local ross_events="" ross_rollbacks="" ross_eff=""

    if [[ $rc -ne 0 ]]; then
        status="exit_${rc}"
    else
        elapsed=$(grep -E 'Running Time[[:space:]]*=' "$tmpf" \
                  | awk -F'=' '{print $2}' | awk '{print $1}' | tail -1)

        dep=$(  grep -E '^Total departures:'    "$tmpf" | awk '{print $3}')
        arr=$(  grep -E '^Total arrivals:'      "$tmpf" | awk '{print $3}')
        conn=$( grep -E '^Total connections:'   "$tmpf" | awk '{print $3}')

        ross_events=$(   grep -E 'Total Events Processed' "$tmpf" | awk '{print $NF}' | head -1)
        ross_rollbacks=$(grep -E '^[[:space:]]*Total Roll Backs' "$tmpf" | awk '{print $NF}')
        ross_eff=$(      grep -E '^[[:space:]]*Efficiency'       "$tmpf" | awk '{print $(NF-1)}')

        if [[ "$elapsed" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            sps=$(awk -v t="$TIMESTEPS" -v e="$elapsed" \
                  'BEGIN { if (e+0 > 0) printf "%.2f", t/e; else print "" }')
        fi
        [[ -z "$elapsed" ]] && status="parse_fail"
    fi

    echo "ross,${sync},${p},${PPN},${NODES_USED},${AIRPORTS_PER_RANK},${total},${TIMESTEPS},${WORK},${trial},${elapsed},${sps},${dep},${arr},${conn},${ross_events},${ross_rollbacks},${ross_eff},${status}" >> "$CSV"

    maybe_keep_log "$tmpf" "ross" "$sync" "$p" "$trial" "$status"
}

# --------------------------------------------------------------------------
for p in "${RANKS[@]}"; do
    compute_placement "$p"
    TOTAL=$(( AIRPORTS_PER_RANK * p ))
    factor_mesh "$p"
    ROSS_ROWS=$ROWS
    ROSS_COLS=$COLS

    echo
    echo "=== P=$p  na=$AIRPORTS_PER_RANK  total_airports=$TOTAL  ross_mesh=${ROSS_ROWS}x${ROSS_COLS} ==="

    for spec in "${ENGINE_ORDER[@]}"; do
        name="${spec%%:*}"
        sync="${spec##*:}"
        bin="${BUILDS[$name]}"
        for trial in $(seq 1 "$TRIALS"); do
            case "$name" in
                dstm) run_dstm "$bin" "$p" "$trial" "$TOTAL" ;;
                ross) run_ross "$bin" "$p" "$trial" "$TOTAL" "$ROSS_ROWS" "$ROSS_COLS" "$sync" ;;
                *)    echo "ERROR: unknown engine $name" >&2; exit 1 ;;
            esac
        done
    done
done

rm -f ross.csv
rm -rf ross_all_lp_stats

echo
echo "Done."
echo "CSV     : $CSV"
echo "Logs    : $LOG_DIR"
