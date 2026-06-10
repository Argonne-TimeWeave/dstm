#!/usr/bin/env bash
# Multi-node STRONG-scaling sweep for the three airport simulator builds.
#
# Total problem size is held constant across all P. Per-rank block size
# shrinks as P grows.
#
# Total airports = TOTAL_AIRPORTS (must be divisible by every P in RANKS).
# Per-rank block = TOTAL_AIRPORTS / P, factored into rows x cols.
#
# Builds expected at:
#   ../airports/airports          (DSTM)
#   ../mpi_air/airports           (MPI Alltoallv)
#   ../single_mpi_air/airports    (MPI Sendrecv per-pair)
#
# Output: strong_scaling_<timestamp>.csv in this scripts/ directory.

set -u

# ---------- Config ----------
RANKS=(1 2 4 8 16 32 64 128 256)
NODES=8
TOTAL_AIRPORTS=8192
TIMESTEPS=8
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
echo "build,ranks,ppn,nodes_used,block_rows,block_cols,airports_per_rank,total_airports,timesteps,work,trial,elapsed_s,steps_per_s,departures,arrivals,connections,status" > "$CSV"

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

# Factor N into rows*cols, as square-ish as possible.
# Sets globals BLOCK_ROWS, BLOCK_COLS.
factor_block() {
    local n="$1"
    local r=1
    for ((i = 1; i * i <= n; i++)); do
        if (( n % i == 0 )); then r=$i; fi
    done
    BLOCK_ROWS=$r
    BLOCK_COLS=$(( n / r ))
}

run_one() {
    local name="$1"
    local bin="$2"
    local p="$3"
    local trial="$4"

    compute_placement "$p"

    local airports_per_rank=$(( TOTAL_AIRPORTS / p ))
    factor_block "$airports_per_rank"

    echo ">>> [$name] ranks=$p ppn=$PPN nodes=$NODES_USED trial=$trial  (block=${BLOCK_ROWS}x${BLOCK_COLS}, ${airports_per_rank} airports/rank)"

    local map_arg=""
    if (( p > 1 )) && (( p % NODES_USED == 0 )); then
        map_arg="--map-by ppr:${PPN}:node"
    fi

    # Capture output to a tmpfile we'll delete after parsing.
    local tmpf
    tmpf=$(mktemp)

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

    echo "${name},${p},${PPN},${NODES_USED},${BLOCK_ROWS},${BLOCK_COLS},${airports_per_rank},${TOTAL_AIRPORTS},${TIMESTEPS},${WORK},${trial},${elapsed},${sps},${dep},${arr},${conn},${status}" >> "$CSV"
}

# ---------- Sweep ----------
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
