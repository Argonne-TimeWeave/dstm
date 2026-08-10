#!/usr/bin/env bash
# 2D sweep: PLANES x RANKS, comparing DSTM (BSP) vs ROSS (optimistic/conservative).
#
# Purpose: find and visualize the crossover where DSTM starts beating
# ROSS as plane density per airport increases.
#
# FAIR TIMING NOTES  (see weak_scale.sh for full rationale)
# ---------------------------------------------------------
# ROSS and DSTM don't share a "time unit":
#   * DSTM  : integer barrier timesteps; every airport departs every
#             plane on the ground each step, up to LANDINGS_PER_STEP
#             landings per step (matches ROSS's MEAN_LAND queue).
#   * ROSS  : continuous virtual time with Exp(MEAN_DEPARTURE=30) hops,
#             Exp(MEAN_LAND=10) runway service, and rollback-based
#             speculation.
# So "seconds per --time=100" vs "seconds per --end=100" is meaningless.
#
# We normalize on the atomic unit both perform: the PLANE HOP.
#   hops_per_sec           = committed_departures / wall_time
#   hops_per_sec_per_rank  = hops_per_sec / P
#
# We match the workload by setting ROSS_END = TIMESTEPS * MEAN_DEPARTURE
# so ROSS's expected departure count per airport equals DSTM's
# (both scale identically with `planes`).
#
# ROSS hop count comes from `Net Events Processed` (Total - EventsRolledBack,
# computed by ROSS itself) divided by 3 (each hop = DEPARTURE+ARRIVAL+LAND).
#
# First trial per (engine, P, planes) is a warm-up; excluded from summary.

set -u

# ---------- Config ----------
RANKS=(4 16 64 256)
PLANES=(1 4 16 64 256 1024 4096 16384)
NODES=1
TOTAL_AIRPORTS=1024
TIMESTEPS=100
TRIALS=3                            # measured trials (+1 discarded warm-up)
SEED=42
EXTRAMEM=200000

# Must match MEAN_DEPARTURE in ROSS's airport.h.
ROSS_MEAN_DEPARTURE=30.0

# Derived: match total plane-hop workload across engines.
ROSS_END=$(awk -v t="$TIMESTEPS" -v m="$ROSS_MEAN_DEPARTURE" \
           'BEGIN { printf "%.4f", t*m }')

KEEP_LOGS=0
for a in "$@"; do
    [[ "$a" == "--keep-logs" ]] && KEEP_LOGS=1
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

declare -A BUILDS=(
    [dstm]="$EXAMPLES_DIR/simple_dstm/airports"
    [ross]="$EXAMPLES_DIR/ROSS/airport"
)

# Engines to run at every (P, planes) point.
ENGINE_ORDER=(
    "dstm:bsp"
    "ross:opt"
    "ross:cons"
)

# ---------- Sanity ----------
if [[ -z "${COBALT_NODEFILE:-}" ]]; then
    echo "ERROR: \$COBALT_NODEFILE not set. Run inside a Cobalt allocation." >&2
    exit 1
fi

ALLOCATED_NODES=$(wc -l < "$COBALT_NODEFILE")
if (( ALLOCATED_NODES < NODES )); then
    echo "ERROR: only $ALLOCATED_NODES node(s) allocated, need $NODES." >&2
    exit 1
fi

for spec in "${ENGINE_ORDER[@]}"; do
    name="${spec%%:*}"
    bin="${BUILDS[$name]}"
    if [[ ! -x "$bin" ]]; then
        echo "ERROR: missing or non-executable build: $name -> $bin" >&2
        exit 1
    fi
done

# TOTAL_AIRPORTS must be a perfect square (torus).
is_perfect_square() {
    local n="$1"
    local s
    s=$(awk -v n="$n" 'BEGIN { s=int(sqrt(n)+0.5); print (s*s==n)?1:0 }')
    [[ "$s" == "1" ]]
}
if ! is_perfect_square "$TOTAL_AIRPORTS"; then
    echo "ERROR: TOTAL_AIRPORTS=$TOTAL_AIRPORTS is not a perfect square." >&2
    exit 1
fi
GRID=$(awk -v n="$TOTAL_AIRPORTS" 'BEGIN { printf "%d", int(sqrt(n)+0.5) }')

MPI_PREFIX="$(dirname "$(dirname "$(which mpirun)")")"

# ---------- Output filenames ----------
PLANES_MIN=${PLANES[0]}
PLANES_MAX=${PLANES[-1]}
PLANES_COUNT=${#PLANES[@]}
RANKS_COUNT=${#RANKS[@]}

BASE="crossover_a${TOTAL_AIRPORTS}_t${TIMESTEPS}_tr${TRIALS}_s${SEED}"
BASE="${BASE}_r${RANKS_COUNT}pts_pl${PLANES_MIN}-${PLANES_MAX}x${PLANES_COUNT}_fair"
CSV="$SCRIPT_DIR/${BASE}.csv"

if [[ -e "$CSV" ]]; then
    i=2
    while [[ -e "$SCRIPT_DIR/${BASE}_${i}.csv" ]]; do i=$(( i + 1 )); done
    CSV="$SCRIPT_DIR/${BASE}_${i}.csv"
fi
SUMMARY_CSV="${CSV%.csv}.summary.csv"

LOG_DIR="$SCRIPT_DIR/${BASE}_logs"
mkdir -p "$LOG_DIR"

TOTAL_RUNS=$(( ${#RANKS[@]} * ${#PLANES[@]} * ${#ENGINE_ORDER[@]} * (TRIALS + 1) ))

echo "=================================================================="
echo "PLANES x RANKS crossover sweep  (fair-timing / hops-per-sec)"
echo "=================================================================="
echo "Allocation           : $ALLOCATED_NODES nodes"
echo "Total airports       : $TOTAL_AIRPORTS  (${GRID}x${GRID} torus, fixed)"
echo "DSTM timesteps       : $TIMESTEPS"
echo "ROSS end virt time   : $ROSS_END   (= $TIMESTEPS * MEAN_DEPARTURE=$ROSS_MEAN_DEPARTURE)"
echo "Measured trials      : $TRIALS  (+1 discarded warm-up per point)"
echo "Ranks sweep          : ${RANKS[*]}"
echo "Planes sweep         : ${PLANES[*]}"
echo "Engines              : ${ENGINE_ORDER[*]}"
echo "CSV output           : $CSV"
echo "Summary CSV          : $SUMMARY_CSV"
echo "Log dir              : $LOG_DIR"
echo "Total mpirun runs    : $TOTAL_RUNS"
echo "=================================================================="
echo

# Raw per-run CSV. trial=0 rows are warm-ups (warmup=1), excluded from summary.
echo "build,sync,ranks,ppn,nodes_used,na,total_airports,grid_side,timesteps,ross_end,planes,trial,warmup,elapsed_s,committed_hops,hops_per_sec,hops_per_sec_per_rank,ross_events,ross_net_events,ross_events_rb,ross_rollbacks,ross_efficiency,status" > "$CSV"

# ---------- Helpers ----------
compute_placement() {
    local p="$1"
    if (( p < NODES )); then
        PPN=1
        NODES_USED=$p
    else
        PPN=$(( p / NODES ))
        NODES_USED=$NODES
    fi
}

maybe_keep_log() {
    local tmpf="$1" name="$2" sync="$3" p="$4" planes="$5" trial="$6" status="$7"
    if [[ "$KEEP_LOGS" == "1" ]] || [[ "$status" != "ok" ]]; then
        local dest="$LOG_DIR/${name}_${sync}_p${p}_pl${planes}_trial${trial}_${status}.log"
        cp "$tmpf" "$dest"
        echo "    (log saved: $dest)"
    fi
    rm -f "$tmpf"
}

fmt_rate() {
    local num="$1" den="$2"
    awk -v n="$num" -v d="$den" 'BEGIN {
        if (n+0 <= 0 || d+0 <= 0) { print ""; exit }
        printf "%.2f", n/d
    }'
}

# ---------- Runners ----------
run_dstm() {
    local bin="$1"; local p="$2"; local trial="$3"; local warmup="$4"
    local na="$5"; local planes="$6"
    local sync="bsp"

    local tag="dstm/${sync}"
    (( warmup )) && tag="${tag}[warmup]"
    echo ">>> [${tag}] P=$p planes=$planes trial=$trial (na=$na)"

    local map_arg=""
    (( p > 1 )) && (( p % NODES_USED == 0 )) && map_arg="--map-by ppr:${PPN}:node"

    local tmpf; tmpf=$(mktemp)

    mpirun -n "$p" --hostfile "$COBALT_NODEFILE" \
        --prefix "$MPI_PREFIX" $map_arg \
        -x LD_LIBRARY_PATH -x PATH \
        "$bin" \
        --na "$na" --time "$TIMESTEPS" \
        --planes "$planes" --seed "$SEED" --quiet \
        > "$tmpf" 2>&1
    local rc=$?

    local elapsed="" dep="" land="" hops="" hps="" hpspr="" status="ok"

    if [[ $rc -ne 0 ]]; then
        status="exit_${rc}"
    else
        elapsed=$(grep -E '^Elapsed time:'     "$tmpf" | awk '{print $3}')
        dep=$(    grep -E '^Total departures:' "$tmpf" | awk '{print $3}')
        land=$(   grep -E '^Total landings:'   "$tmpf" | awk '{print $3}')

        hops="$dep"
        hps=$(fmt_rate "$hops" "$elapsed")
        hpspr=$(fmt_rate "$hps" "$p")

        [[ -z "$elapsed" ]] && status="parse_fail"
    fi

    echo "dstm,${sync},${p},${PPN},${NODES_USED},${na},${TOTAL_AIRPORTS},${GRID},${TIMESTEPS},${ROSS_END},${planes},${trial},${warmup},${elapsed},${hops},${hps},${hpspr},,,,,,${status}" >> "$CSV"

    maybe_keep_log "$tmpf" "dstm" "$sync" "$p" "$planes" "$trial" "$status"
}

run_ross() {
    local bin="$1"; local p="$2"; local trial="$3"; local warmup="$4"
    local planes="$5"; local sync="$6"

    local synch_flag="" ; local extra_flags=()
    case "$sync" in
        cons) synch_flag="--synch=2"; extra_flags=( "--lookahead=1.00" ) ;;
        opt)  synch_flag="--synch=3"; extra_flags=( "--extramem=${EXTRAMEM}" ) ;;
        *)    echo "ERROR: unknown ROSS sync '$sync'" >&2; return 1 ;;
    esac

    local tag="ross/${sync}"
    (( warmup )) && tag="${tag}[warmup]"
    echo ">>> [${tag}] P=$p planes=$planes trial=$trial (end=$ROSS_END)"

    local map_arg=""
    (( p > 1 )) && (( p % NODES_USED == 0 )) && map_arg="--map-by ppr:${PPN}:node"

    local tmpf; tmpf=$(mktemp)

    mpirun -n "$p" --hostfile "$COBALT_NODEFILE" \
        --prefix "$MPI_PREFIX" $map_arg \
        -x LD_LIBRARY_PATH -x PATH \
        "$bin" "$synch_flag" "${extra_flags[@]}" \
        --nairports="$TOTAL_AIRPORTS" \
        --end="$ROSS_END" --nplanes="$planes" \
        > "$tmpf" 2>&1
    local rc=$?

    local elapsed="" status="ok"
    local ross_events="" ross_net_events="" ross_events_rb=""
    local ross_rollbacks="" ross_eff=""
    local hops="" hps="" hpspr=""

    if [[ $rc -ne 0 ]]; then
        status="exit_${rc}"
    else
        elapsed=$(grep -E 'Running Time[[:space:]]*=' "$tmpf" \
                  | awk -F'=' '{print $2}' | awk '{print $1}' | tail -1)

        ross_events=$(     grep -E 'Total Events Processed'         "$tmpf" | awk '{print $NF}' | head -1)
        ross_net_events=$( grep -E 'Net Events Processed'           "$tmpf" | awk '{print $NF}')
        ross_events_rb=$(  grep -E 'Events Rolled Back'             "$tmpf" | awk '{print $NF}')
        ross_rollbacks=$(  grep -E '^[[:space:]]*Total Roll Backs'  "$tmpf" | awk '{print $NF}')
        ross_eff=$(        grep -E '^[[:space:]]*Efficiency'        "$tmpf" | awk '{print $(NF-1)}')

        # Committed hops = Net Events Processed / 3
        # (each plane hop = DEPARTURE + ARRIVAL + LAND).
        # ROSS already computed Net = Total - EventsRolledBack; trust its number.
        if [[ -n "$ross_net_events" ]]; then
            hops=$(awk -v c="$ross_net_events" 'BEGIN { printf "%d", int(c/3.0 + 0.5) }')
        fi

        hps=$(fmt_rate "$hops" "$elapsed")
        hpspr=$(fmt_rate "$hps" "$p")

        [[ -z "$elapsed" ]] && status="parse_fail"
    fi

    echo "ross,${sync},${p},${PPN},${NODES_USED},,${TOTAL_AIRPORTS},${GRID},${TIMESTEPS},${ROSS_END},${planes},${trial},${warmup},${elapsed},${hops},${hps},${hpspr},${ross_events},${ross_net_events},${ross_events_rb},${ross_rollbacks},${ross_eff},${status}" >> "$CSV"

    maybe_keep_log "$tmpf" "ross" "$sync" "$p" "$planes" "$trial" "$status"
}

# ---------- Main sweep ----------
# Outer:  RANKS  (placement setup)
# Middle: PLANES
# Inner:  trial, then engine (interleave so noise hits all engines equally)
# For each (P, planes) we do 1 warm-up per engine first, then TRIALS measured.
for p in "${RANKS[@]}"; do
    if (( TOTAL_AIRPORTS % p != 0 )); then
        echo "SKIP P=$p: does not divide TOTAL_AIRPORTS=$TOTAL_AIRPORTS"
        continue
    fi
    NA=$(( TOTAL_AIRPORTS / p ))
    compute_placement "$p"

    echo
    echo "########## P=$p  na=$NA ##########"

    for planes in "${PLANES[@]}"; do
        echo
        echo "---- planes=$planes ----"

        # Warm-up trial per engine (trial=0, warmup=1) — excluded from summary.
        for spec in "${ENGINE_ORDER[@]}"; do
            name="${spec%%:*}"; sync="${spec##*:}"
            bin="${BUILDS[$name]}"
            case "$name" in
                dstm) run_dstm "$bin" "$p" 0 1 "$NA" "$planes" ;;
                ross) run_ross "$bin" "$p" 0 1 "$planes" "$sync" ;;
            esac
        done

        # Measured trials, interleaved by engine.
        for trial in $(seq 1 "$TRIALS"); do
            for spec in "${ENGINE_ORDER[@]}"; do
                name="${spec%%:*}"; sync="${spec##*:}"
                bin="${BUILDS[$name]}"
                case "$name" in
                    dstm) run_dstm "$bin" "$p" "$trial" 0 "$NA" "$planes" ;;
                    ross) run_ross "$bin" "$p" "$trial" 0 "$planes" "$sync" ;;
                esac
            done
        done
    done
done

# ----------------------------------------------------------------------
# Summarize: median per (build, sync, ranks, planes), excluding warm-ups.
# Field indices (must match the CSV header above):
#   1  build          8  grid_side       15 committed_hops       22 ross_efficiency
#   2  sync           9  timesteps       16 hops_per_sec         23 status
#   3  ranks         10  ross_end        17 hops_per_sec_per_rank
#   4  ppn           11  planes          18 ross_events
#   5  nodes_used    12  trial           19 ross_net_events
#   6  na            13  warmup          20 ross_events_rb
#   7  total_air     14  elapsed_s       21 ross_rollbacks
# ----------------------------------------------------------------------
echo "build,sync,ranks,planes,n_trials,median_elapsed_s,median_hops,median_hops_per_sec,median_hops_per_sec_per_rank" > "$SUMMARY_CSV"

awk -F, '
    NR == 1 { next }
    $13 == "1" { next }        # skip warm-ups
    $23 != "ok" { next }        # skip failed rows (status is now field 23)
    {
        # key: build,sync,ranks,planes
        key = $1 "," $2 "," $3 "," $11
        el[key]    = el[key]    " " $14
        hp[key]    = hp[key]    " " $15
        hps[key]   = hps[key]   " " $16
        hpspr[key] = hpspr[key] " " $17
        cnt[key]++
    }
    function median(list,   a, n, i, j, kk, v) {
        n = split(list, a, " ")
        j = 0
        for (i = 1; i <= n; i++) if (a[i] != "") { j++; a[j] = a[i]+0 }
        n = j
        if (n == 0) return ""
        for (i = 2; i <= n; i++) {
            v = a[i]; kk = i - 1
            while (kk >= 1 && a[kk] > v) { a[kk+1] = a[kk]; kk-- }
            a[kk+1] = v
        }
        if (n % 2) return a[(n+1)/2]
        return (a[n/2] + a[n/2+1]) / 2.0
    }
    END {
        for (k in cnt) {
            printf "%s,%d,%.6f,%.0f,%.2f,%.2f\n",
                k, cnt[k],
                median(el[k])    + 0,
                median(hp[k])    + 0,
                median(hps[k])   + 0,
                median(hpspr[k]) + 0
        }
    }
' "$CSV" | sort -t, -k1,1 -k2,2 -k3,3n -k4,4n >> "$SUMMARY_CSV"

rm -f ross.csv
rm -rf ross_all_lp_stats

echo
echo "=================================================================="
echo "Done."
echo "Raw CSV     : $CSV"
echo "Summary CSV : $SUMMARY_CSV"
echo "Logs        : $LOG_DIR"
echo
echo "Primary metric: median_hops_per_sec (and median_hops_per_sec_per_rank)"
echo "To plot:"
echo "  python3 $SCRIPT_DIR/plot_crossover.py $SUMMARY_CSV"
echo "=================================================================="
