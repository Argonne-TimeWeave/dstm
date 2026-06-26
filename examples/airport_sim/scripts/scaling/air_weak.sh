#!/usr/bin/env bash
# Weak-scaling sweep comparing the DSTM and MPI airport simulators.
#
# Per-rank problem size is held constant across all P. Total airport count
# grows linearly with P:
#     NA           = AIRPORTS_PER_RANK  (constant)
#     total        = AIRPORTS_PER_RANK * P
#
# Ideal weak-scaling: elapsed time stays flat as P grows.
#
# Builds expected at:
#   ../airports/airports     (DSTM)
#   ../mpi_air/airports      (MPI send/recv)
#
# Output: weak_scaling_<timestamp>.csv next to this script.

set -u

# ---------- Config ----------
RANKS=(1 2 4 8 16 32 64 128 256)
NODES=8
AIRPORTS_PER_RANK=64    # fixed per-rank workload
TIMESTEPS=64
TRIALS=1
WORK=2500
SEED=42

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

declare -A BUILDS=(
    [dstm]="$EXAMPLES_DIR/airports/airports"
    [mpi]="$EXAMPLES_DIR/mpi_air/airports"
)

# Order to iterate the builds in (associative arrays don't preserve order).
BUILD_ORDER=(dstm mpi)

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

MPI_PREFIX="$(dirname "$(dirname "$(which mpirun)")")"
echo "MPI prefix: $MPI_PREFIX"
echo "Airports per rank (fixed): $AIRPORTS_PER_RANK"
echo "Largest run total airports: $(( AIRPORTS_PER_RANK * ${RANKS[-1]} ))"
echo

# ---------- Output filename from parameters ----------
# Compact, sorted, deterministic. Same params -> same filename.
# Format:
#   weak_n<NODES>_apr<APR>_t<TIMESTEPS>_w<WORK>_tr<TRIALS>_s<SEED>_r<MIN>-<MAX>x<COUNT>.csv
RANKS_MIN=${RANKS[0]}
RANKS_MAX=${RANKS[-1]}
RANKS_COUNT=${#RANKS[@]}

BASE="weak_n${NODES}_apr${AIRPORTS_PER_RANK}_t${TIMESTEPS}_w${WORK}_tr${TRIALS}_s${SEED}_r${RANKS_MIN}-${RANKS_MAX}x${RANKS_COUNT}"
CSV="$SCRIPT_DIR/${BASE}.csv"

# Avoid clobbering an earlier run with identical params: append _2, _3, ...
if [[ -e "$CSV" ]]; then
    i=2
    while [[ -e "$SCRIPT_DIR/${BASE}_${i}.csv" ]]; do
        i=$(( i + 1 ))
    done
    CSV="$SCRIPT_DIR/${BASE}_${i}.csv"
fi

echo "CSV output: $CSV"

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

    local na=$AIRPORTS_PER_RANK
    local total=$(( na * p ))

    echo ">>> [$name] ranks=$p ppn=$PPN nodes=$NODES_USED trial=$trial  (na=$na, total=$total)"

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

    echo "${name},${p},${PPN},${NODES_USED},${na},${total},${TIMESTEPS},${WORK},${trial},${elapsed},${sps},${dep},${arr},${conn},${status}" >> "$CSV"
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
