#!/usr/bin/env bash
# Weak-scaling sweep comparing (with FAIR timing normalization):
#   - DSTM airport sim (BSP)
#   - ROSS airport sim, conservative  (--synch=2)
#   - ROSS airport sim, optimistic    (--synch=3)
#
# WEAK SCALING: per-rank airport count is HELD CONSTANT.
#   AIRPORTS_PER_RANK = 64
#   TOTAL_AIRPORTS(P) = 64 * P
#
# FAIR TIMING NOTES
# -----------------
# ROSS and DSTM do NOT measure the same "time unit":
#   * DSTM: integer barrier timesteps; every airport tries to depart once per step.
#   * ROSS: continuous virtual time with Exp(MEAN_DEPARTURE=30) inter-departures
#           and rollback-based speculation.
# So "seconds per --time=128" vs "seconds per --end=128" is meaningless.
#
# This script normalizes on the atomic unit both perform: the PLANE HOP
# (one airport -> airport transfer). We report:
#   hops_per_sec               = committed_departures / wall_time
#   hops_per_sec_per_rank      = hops_per_sec / P
#
# We also MATCH THE WORKLOAD so both engines are asked to perform ~the same
# number of hops:
#   Expected DSTM hops  ~= TIMESTEPS * total_airports * PLANES
#   Expected ROSS hops  ~= (ROSS_END / MEAN_DEPARTURE) * total_airports * PLANES
# so we set ROSS_END = TIMESTEPS * MEAN_DEPARTURE.
#
# Finally, the first trial per (engine, P) is treated as WARM-UP and dropped
# from the summary (still logged in the raw CSV with trial=0).

set -u

# ---------- Config ----------
RANKS=(4 16 64 256)
NODES=8
AIRPORTS_PER_RANK=64
TIMESTEPS=128                # DSTM --time (bulk rounds)
PLANES=64                    # ROSS --nplanes and DSTM --planes
TRIALS=2                     # measured trials (a warm-up trial runs before these)
SEED=42
EXTRAMEM=200000

# Must match MEAN_DEPARTURE in ROSS's airport.h. If you change it there,
# change it here.
ROSS_MEAN_DEPARTURE=30.0

# Derived: give ROSS enough virtual time to depart each plane ~TIMESTEPS times,
# matching the total plane-hop workload DSTM performs.
ROSS_END=$(awk -v t="$TIMESTEPS" -v m="$ROSS_MEAN_DEPARTURE" \
           'BEGIN { printf "%.4f", t*m }')

# When set to 1 (or pass --keep-logs), save every run's stdout+stderr.
KEEP_LOGS=0
for a in "$@"; do
    [[ "$a" == "--keep-logs" ]] && KEEP_LOGS=1
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

declare -A BUILDS=(
    [dstm]="$EXAMPLES_DIR/simple_dstm/airports"
    [ross]="$SCRIPT_DIR/airport"
)

# Format: "<name>:<sync_mode>"  (sync_mode is only used for ross)
ENGINE_ORDER=(
    "dstm:bsp"
    "ross:cons"
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
echo "Airports per rank       : $AIRPORTS_PER_RANK   (weak scaling, held constant)"
echo "DSTM timesteps          : $TIMESTEPS"
echo "ROSS end virtual time   : $ROSS_END   (= $TIMESTEPS * MEAN_DEPARTURE=$ROSS_MEAN_DEPARTURE)"
echo "Planes per airport      : $PLANES"
echo "Measured trials / point : $TRIALS  (+1 discarded warm-up)"
echo "Rank sweep              : ${RANKS[*]}"
echo "Keep per-run logs       : $KEEP_LOGS"
echo

# Verify every P in the sweep yields a perfect-square total airport count.
is_perfect_square() {
    local n="$1"
    local s
    s=$(awk -v n="$n" 'BEGIN { s=int(sqrt(n)+0.5); print (s*s==n)?1:0 }')
    [[ "$s" == "1" ]]
}
for p in "${RANKS[@]}"; do
    total=$(( AIRPORTS_PER_RANK * p ))
    if ! is_perfect_square "$total"; then
        echo "ERROR: P=$p * apr=$AIRPORTS_PER_RANK = $total is not a perfect square." >&2
        exit 1
    fi
done

# ---------- Output filenames ----------
RANKS_MIN=${RANKS[0]}
RANKS_MAX=${RANKS[-1]}
RANKS_COUNT=${#RANKS[@]}

BASE="weak_n${NODES}_apr${AIRPORTS_PER_RANK}_t${TIMESTEPS}_p${PLANES}_tr${TRIALS}_s${SEED}_r${RANKS_MIN}-${RANKS_MAX}x${RANKS_COUNT}_fair"
CSV="$SCRIPT_DIR/${BASE}.csv"

if [[ -e "$CSV" ]]; then
    i=2
    while [[ -e "$SCRIPT_DIR/${BASE}_${i}.csv" ]]; do i=$(( i + 1 )); done
    CSV="$SCRIPT_DIR/${BASE}_${i}.csv"
fi
SUMMARY_CSV="${CSV%.csv}.summary.csv"

LOG_DIR="$SCRIPT_DIR/${BASE}_logs"
mkdir -p "$LOG_DIR"

echo "CSV output  : $CSV"
echo "Summary CSV : $SUMMARY_CSV"
echo "Log dir     : $LOG_DIR (populated on failure or --keep-logs)"

# Raw per-run CSV. trial=0 means warm-up (excluded from summary).
echo "build,sync,ranks,ppn,nodes_used,na,total_airports,grid_side,timesteps,ross_end,planes,trial,warmup,elapsed_s,committed_hops,hops_per_sec,hops_per_sec_per_rank,ross_events,ross_rollbacks,ross_efficiency,status" > "$CSV"

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

# fmt_rate: safe division, prints "" on non-numeric or zero-time input.
fmt_rate() {
    local num="$1" den="$2"
    awk -v n="$num" -v d="$den" 'BEGIN {
        if (n+0 <= 0 || d+0 <= 0) { print ""; exit }
        printf "%.2f", n/d
    }'
}

# --------------------------------------------------------------------------
run_dstm() {
    local bin="$1"; local p="$2"; local trial="$3"; local warmup="$4"
    local na="$5"; local total="$6"; local grid="$7"
    local sync="bsp"

    local tag="dstm/${sync}"
    (( warmup )) && tag="${tag}[warmup]"
    echo ">>> [${tag}] ranks=$p ppn=$PPN nodes=$NODES_USED trial=$trial  (na=$na total=$total grid=${grid}x${grid})"

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
        --na "$na" \
        --time "$TIMESTEPS" \
        --planes "$PLANES" \
        --seed "$SEED" \
        --quiet \
        > "$tmpf" 2>&1
    local rc=$?

    local elapsed="" dep="" land="" hops="" hps="" hpspr="" status="ok"

    if [[ $rc -ne 0 ]]; then
        status="exit_${rc}"
    else
        elapsed=$(grep -E '^Elapsed time:'     "$tmpf" | awk '{print $3}')
        dep=$(    grep -E '^Total departures:' "$tmpf" | awk '{print $3}')
        land=$(   grep -E '^Total landings:'   "$tmpf" | awk '{print $3}')

        # For DSTM the atomic hop count is exactly total_departures.
        hops="$dep"
        hps=$(fmt_rate "$hops" "$elapsed")
        hpspr=$(fmt_rate "$hps" "$p")

        [[ -z "$elapsed" ]] && status="parse_fail"
    fi

    echo "dstm,${sync},${p},${PPN},${NODES_USED},${na},${total},${grid},${TIMESTEPS},${ROSS_END},${PLANES},${trial},${warmup},${elapsed},${hops},${hps},${hpspr},,,,${status}" >> "$CSV"

    maybe_keep_log "$tmpf" "dstm" "$sync" "$p" "$trial" "$status"
}

# --------------------------------------------------------------------------
# run_ross <bin> <p> <trial> <warmup> <total_airports> <grid_side> <sync_mode>
#   sync_mode ∈ {cons, opt}
# --------------------------------------------------------------------------
run_ross() {
    local bin="$1"; local p="$2"; local trial="$3"; local warmup="$4"
    local total="$5"; local grid="$6"; local sync="$7"

    local synch_flag=""
    local extra_flags=()
    case "$sync" in
        cons)
            synch_flag="--synch=2"
            extra_flags=( "--lookahead=1.00" )
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

    local tag="ross/${sync}"
    (( warmup )) && tag="${tag}[warmup]"
    echo ">>> [${tag}] ranks=$p ppn=$PPN nodes=$NODES_USED trial=$trial  (nairports=$total grid=${grid}x${grid} end=$ROSS_END)"

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
        --nairports="$total" \
        --end="$ROSS_END" \
        --nplanes="$PLANES" \
        > "$tmpf" 2>&1
    local rc=$?

    local elapsed="" status="ok"
    local ross_events="" ross_rollbacks="" ross_eff=""
    local hops="" hps="" hpspr=""

    if [[ $rc -ne 0 ]]; then
        status="exit_${rc}"
    else
        elapsed=$(grep -E 'Running Time[[:space:]]*=' "$tmpf" \
                  | awk -F'=' '{print $2}' | awk '{print $1}' | tail -1)

        ross_events=$(     grep -E 'Total Events Processed'       "$tmpf" | awk '{print $NF}' | head -1)
        ross_net_events=$( grep -E 'Net Events Processed'         "$tmpf" | awk '{print $NF}')
        ross_events_rb=$(  grep -E 'Events Rolled Back'           "$tmpf" | awk '{print $NF}')
        ross_rollbacks=$(  grep -E '^[[:space:]]*Total Roll Backs' "$tmpf" | awk '{print $NF}')
        ross_eff=$(        grep -E '^[[:space:]]*Efficiency'      "$tmpf" | awk '{print $(NF-1)}')

        # Each plane hop = DEPARTURE + ARRIVAL + LAND, so 3 committed events per hop.
        # ROSS already computed Net = Total - EventsRolledBack; trust its number.
        if [[ -n "$ross_net_events" ]]; then
            hops=$(awk -v c="$ross_net_events" 'BEGIN { printf "%d", int(c/3.0 + 0.5) }')
        fi

        hps=$(fmt_rate "$hops" "$elapsed")
        hpspr=$(fmt_rate "$hps" "$p")

        [[ -z "$elapsed" ]] && status="parse_fail"
    fi

    echo "ross,${sync},${p},${PPN},${NODES_USED},,${total},${grid},${TIMESTEPS},${ROSS_END},${PLANES},${trial},${warmup},${elapsed},${hops},${hps},${hpspr},${ross_events},${ross_rollbacks},${ross_eff},${status}" >> "$CSV"

    maybe_keep_log "$tmpf" "ross" "$sync" "$p" "$trial" "$status"
}

# --------------------------------------------------------------------------
# Main sweep: for each P and each engine, run 1 warm-up + TRIALS measured runs.
# --------------------------------------------------------------------------
for p in "${RANKS[@]}"; do
    NA=$AIRPORTS_PER_RANK
    TOTAL=$(( NA * p ))
    GRID=$(awk -v n="$TOTAL" 'BEGIN { printf "%d", int(sqrt(n)+0.5) }')

    compute_placement "$p"

    echo
    echo "=== P=$p  na=$NA  total_airports=$TOTAL  grid=${GRID}x${GRID} ==="

    for spec in "${ENGINE_ORDER[@]}"; do
        name="${spec%%:*}"
        sync="${spec##*:}"
        bin="${BUILDS[$name]}"

        # Warm-up (trial=0, warmup=1) — excluded from summary.
        case "$name" in
            dstm) run_dstm "$bin" "$p" 0 1 "$NA" "$TOTAL" "$GRID" ;;
            ross) run_ross "$bin" "$p" 0 1 "$TOTAL" "$GRID" "$sync" ;;
        esac

        # Measured trials.
        for trial in $(seq 1 "$TRIALS"); do
            case "$name" in
                dstm) run_dstm "$bin" "$p" "$trial" 0 "$NA" "$TOTAL" "$GRID" ;;
                ross) run_ross "$bin" "$p" "$trial" 0 "$TOTAL" "$GRID" "$sync" ;;
                *)    echo "ERROR: unknown engine $name" >&2; exit 1 ;;
            esac
        done
    done
done

# --------------------------------------------------------------------------
# Summarize: median hops/sec per (build, sync, ranks), excluding warm-ups.
# --------------------------------------------------------------------------
echo "build,sync,ranks,n_trials,median_elapsed_s,median_hops,median_hops_per_sec,median_hops_per_sec_per_rank" > "$SUMMARY_CSV"

awk -F, '
    NR == 1 { next }                          # skip header
    $13 == "1" { next }                       # skip warm-up rows
    $21 != "ok" { next }                      # skip failed rows
    {
        key = $1 "," $2 "," $3
        el[key]  = el[key]  " " $14
        hp[key]  = hp[key]  " " $15
        hps[key] = hps[key] " " $16
        hpspr[key] = hpspr[key] " " $17
        cnt[key]++
    }
    function median(list,   a, n, i, j, k, v) {
        n = split(list, a, " ")
        # split leaves a[1]="" from the leading space; compact it.
        j = 0
        for (i = 1; i <= n; i++) if (a[i] != "") { j++; a[j] = a[i]+0 }
        n = j
        if (n == 0) return ""
        # insertion sort (n is tiny)
        for (i = 2; i <= n; i++) {
            v = a[i]; k = i - 1
            while (k >= 1 && a[k] > v) { a[k+1] = a[k]; k-- }
            a[k+1] = v
        }
        if (n % 2) return a[(n+1)/2]
        return (a[n/2] + a[n/2+1]) / 2.0
    }
    END {
        for (k in cnt) {
            printf "%s,%d,%.6f,%.0f,%.2f,%.2f\n",
                k, cnt[k],
                median(el[k])  + 0,
                median(hp[k])  + 0,
                median(hps[k]) + 0,
                median(hpspr[k]) + 0
        }
    }
' "$CSV" | sort -t, -k1,1 -k2,2 -k3,3n >> "$SUMMARY_CSV"

rm -f ross.csv
rm -rf ross_all_lp_stats

echo
echo "Done."
echo "Raw CSV     : $CSV"
echo "Summary CSV : $SUMMARY_CSV"
echo "Logs        : $LOG_DIR"
echo
echo "Primary comparison metric: median_hops_per_sec_per_rank in $SUMMARY_CSV"
