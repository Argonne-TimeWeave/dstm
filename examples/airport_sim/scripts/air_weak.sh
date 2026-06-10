#!/usr/bin/env bash
# Multi-node weak-scaling sweep for the three airport simulator builds.
# Each rank owns a BLOCK_ROWS x BLOCK_COLS block of airports; total airports = block * P.
#
# Assumes a Cobalt allocation of 4 nodes (qsub -I -n 4 -q skylake_8180 ...).
# Ranks are distributed evenly across all 4 nodes once P >= 4.
#
# Builds expected at:
#   ../airports/airports          (DSTM)
#   ../mpi_air/airports           (MPI Alltoallv)
#   ../single_mpi_air/airports    (MPI Sendrecv per-pair)
#
# Output: weak_scaling_<timestamp>.csv in this scripts/ directory.

set -u  # don't use -e: we want to keep going past individual failures

# ---------- Config ----------
RANKS=(1 2 4 8 16 32 64 128 256)
NODES=8                       # number of nodes in the allocation
BLOCK_ROWS=4
BLOCK_COLS=4
TIMESTEPS=32
TRIALS=2
WORK=100000
SEED=42

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

declare -A BUILDS=(
    [dstm]="$EXAMPLES_DIR/airports/airports"
    [mpi_alltoall]="$EXAMPLES_DIR/mpi_air/airports"
    [mpi_sendrecv]="$EXAMPLES_DIR/single_mpi_air/airports"
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

for name in "${!BUILDS[@]}"; do
    bin="${BUILDS[$name]}"
    if [[ ! -x "$bin" ]]; then
        echo "ERROR: missing or non-executable build: $name -> $bin" >&2
        exit 1
    fi
done

MPI_PREFIX="$(dirname "$(dirname "$(which mpirun)")")"
echo "MPI prefix: $MPI_PREFIX"
echo

TS="$(date +%Y%m%d_%H%M%S)"
CSV="$SCRIPT_DIR/weak_scaling_${TS}.csv"
LOG_DIR="$SCRIPT_DIR/logs_${TS}"
mkdir -p "$LOG_DIR"

# ---------- CSV header ----------
echo "build,ranks,ppn,nodes_used,block_rows,block_cols,total_airports,timesteps,work,trial,elapsed_s,steps_per_s,departures,arrivals,connections,status" > "$CSV"

# Compute ranks-per-node and nodes-used for a given total rank count.
# Strategy: spread evenly across all $NODES whenever possible.
#   P  <  NODES -> 1 ppn, P nodes
#   P >=  NODES -> ceil(P/NODES) ppn, NODES nodes (requires P % NODES == 0 for ppr)
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

    local total_airports=$(( BLOCK_ROWS * BLOCK_COLS * p ))
    local logf="$LOG_DIR/${name}_p${p}_t${trial}.log"

    echo ">>> [$name] ranks=$p ppn=$PPN nodes=$NODES_USED trial=$trial  (airports=$total_airports)"

    # Build mpirun command. Only use --map-by ppr when it makes sense:
    #   * P > 1 (ppr:1:node with P=1 is fine but pointless)
    #   * P is divisible by NODES_USED (ppr requires exact fit)
    local map_arg=""
    if (( p > 1 )) && (( p % NODES_USED == 0 )); then
        map_arg="--map-by ppr:${PPN}:node"
    fi

    mpirun -n "$p" --hostfile "$COBALT_NODEFILE" \
        --prefix "$MPI_PREFIX" \
        $map_arg \
        -x LD_LIBRARY_PATH -x PATH \
        "$bin" \
        --block-rows "$BLOCK_ROWS" \
        --block-cols "$BLOCK_COLS" \
        --time "$TIMESTEPS" \
        --seed "$SEED" \
        --work "$WORK" \
        --quiet \
        > "$logf" 2>&1
    local rc=$?

    local elapsed="" sps="" dep="" arr="" conn="" status="ok"

    if [[ $rc -ne 0 ]]; then
        status="exit_${rc}"
    else
        elapsed=$(grep -E '^Elapsed time:'      "$logf" | awk '{print $3}')
        sps=$(    grep -E '^Timesteps/second:'  "$logf" | awk '{print $2}')
        dep=$(    grep -E '^Total departures:'  "$logf" | awk '{print $3}')
        arr=$(    grep -E '^Total arrivals:'    "$logf" | awk '{print $3}')
        conn=$(   grep -E '^Total connections:' "$logf" | awk '{print $3}')
        if [[ -z "$elapsed" ]]; then
            status="parse_fail"
        fi
    fi

    echo "${name},${p},${PPN},${NODES_USED},${BLOCK_ROWS},${BLOCK_COLS},${total_airports},${TIMESTEPS},${WORK},${trial},${elapsed},${sps},${dep},${arr},${conn},${status}" >> "$CSV"
}

# ---------- Sweep ----------
# Order: for each rank count, run all builds back-to-back, all trials.
# Keeps node/cache/network state more comparable within a rank-count group.
for p in "${RANKS[@]}"; do
    for name in dstm mpi_alltoall mpi_sendrecv; do
        bin="${BUILDS[$name]}"
        for trial in $(seq 1 "$TRIALS"); do
            run_one "$name" "$bin" "$p" "$trial"
        done
    done
done

echo
echo "Done."
echo "CSV : $CSV"
echo "Logs: $LOG_DIR"
