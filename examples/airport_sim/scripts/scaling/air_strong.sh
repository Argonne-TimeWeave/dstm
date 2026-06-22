#!/usr/bin/env bash
# Strong-scaling sweep comparing the DSTM and MPI airport simulators.
#
# Total airport count is held constant across all P. Per-rank airport count
# (--na) shrinks as P grows: NA = TOTAL_AIRPORTS / P.
#
# Builds expected at:
#   ../airports/airports     (DSTM)
#   ../mpi_air/airports      (MPI send/recv)
#
# Output: strong_scaling_<timestamp>.csv next to this script.

set -u

# ---------- Config ----------
RANKS=(1 2 4 8 16 32 64 128 256)
NODES=8
TOTAL_AIRPORTS=16384
TIMESTEPS=32
TRIALS=3
WORK=100000
SEED=42

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

declare -A BUILDS=(
    [dstm]="$EXAMPLES_DIR/airports/airports"
)

# Order to iterate the builds in (associative arrays don't preserve order).
BUILD_ORDER=(dstm)

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

for name in "${BUILD_ORDER[@]}"; do
    bin="${BUILDS[$name]}"
    if [[ ! -x "$bin" ]]; then
        echo "ERROR: missing or non-executable build: $name -> $bin" >&2
        exit 1
    fi
done

# Verify total airports divides evenly by every requested P.
for p in "${RANKS[@]}"; do
    if (( TOTAL_AIRPORTS % p != 0 )); then
        echo "ERROR: TOTAL_AIRPORTS=$TOTAL_AIRPORTS not divisible by P=$p" >&2
        exit 1
    fi
done

MPI_PREFIX="$(dirname "$(dirname "$(which mpirun)")")"
echo "MPI prefix: $MPI_PREFIX"
echo "Total airports (fixed): $TOTAL_AIRPORTS"
echo

TS="$(date +%Y%m%d_%H%M%S)"
CSV="$SCRIPT_DIR/strong_scaling_${TS}.csv"

# ---------- CSV header ----------
echo "build,ranks,ppn,nodes_used,na,total_airports,timesteps,work,trial,elapsed_s,steps_per_s,departures,arrivals,connections,status" > "$CSV"

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

run_one() {
    local name="$1"
    local bin="$2"
    local p="$3"
    local trial="$4"

    compute_placement "$p"

    local na=$(( TOTAL_AIRPORTS / p ))

    echo ">>> [$name] ranks=$p ppn=$PPN nodes=$NODES_USED trial=$trial  (na=$na)"

    local map_arg=""
    if (( p > 1 )) && (( p % NODES_USED == 0 )); then
        map_arg="--map-by ppr:${PPN}:node"
    fi

    local tmpf
    tmpf=$(mktemp)

    mpirun -n "$p" --hostfile "$COBALT_NODEFILE" \
        --prefix "$MPI_PREFIX" \
        $map_arg \
        -x LD_LIBRARY_PATH -x PATH \
        "$bin" \
        --na "$na" \
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
        if [[ -z "$elapsed" ]]; then
            status="parse_fail"
        fi
    fi

    rm -f "$tmpf"

    echo "${name},${p},${PPN},${NODES_USED},${na},${TOTAL_AIRPORTS},${TIMESTEPS},${WORK},${trial},${elapsed},${sps},${dep},${arr},${conn},${status}" >> "$CSV"
}

# ---------- Sweep ----------
for p in "${RANKS[@]}"; do
    for name in "${BUILD_ORDER[@]}"; do
        bin="${BUILDS[$name]}"
        for trial in $(seq 1 "$TRIALS"); do
            run_one "$name" "$bin" "$p" "$trial"
        done
    done
done

echo
echo "Done."
echo "CSV : $CSV"
