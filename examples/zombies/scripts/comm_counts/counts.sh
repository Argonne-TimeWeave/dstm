#!/usr/bin/env bash
# Weak-scaling sweep that captures *communication count logs* (not perf) from
# the DSTM and MPI zombie simulators, and tallies comm op counts per run.
#
# Per-rank problem size is held constant across all P:
#     NL = LOCATIONS_PER_RANK (constant)
#     NH = HUMANS_PER_RANK    (constant)
#
# Output dir contains:
#   params.txt        - parameters of this sweep
#   index.csv         - one row per run, pointing at its log file
#   comm_counts.csv   - one row per run, with parsed Recv/Send/... totals
#   <build>_pNNNN.log - raw stdout+stderr from each run

set -u

# ---------- Config ----------
RANKS=(1 2 4 8 16 32 64 128 256)
NODES=8
LOCATIONS_PER_RANK=128
HUMANS_PER_RANK=512
TIMESTEPS=72
WORK=1000
SEED=42

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

declare -A BUILDS=(
    [dstm]="$EXAMPLES_DIR/dstm_zombies/zombies"
    [mpi]="$EXAMPLES_DIR/mpi_zombies/zombies"
)
BUILD_ORDER=(dstm mpi)

PARSER="$SCRIPT_DIR/parse.py"
if [[ ! -f "$PARSER" ]]; then
    echo "ERROR: missing parser: $PARSER" >&2
    exit 1
fi

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
echo "Locations per rank (fixed): $LOCATIONS_PER_RANK"
echo "Humans per rank    (fixed): $HUMANS_PER_RANK"
echo "Largest run total humans  : $(( HUMANS_PER_RANK * ${RANKS[-1]} ))"
echo

# ---------- Output layout ----------
RUN_TAG="weak_nl${LOCATIONS_PER_RANK}_nh${HUMANS_PER_RANK}_t${TIMESTEPS}_w${WORK}_s${SEED}_n${NODES}"
OUT_DIR="$SCRIPT_DIR/logs_${RUN_TAG}"

if [[ -e "$OUT_DIR" ]]; then
    echo "ERROR: output dir already exists: $OUT_DIR" >&2
    echo "Remove it or change the parameters and rerun." >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

INDEX_CSV="$OUT_DIR/index.csv"
echo "build,ranks,ppn,nodes_used,nl,nh,total_humans,timesteps,work,seed,log_file,status" > "$INDEX_CSV"

COUNTS_CSV="$OUT_DIR/comm_counts.csv"
PARSER_HEADER=$(python3 "$PARSER" --csv /dev/null 2>/dev/null | head -n 1)
if [[ -z "$PARSER_HEADER" ]]; then
    echo "ERROR: could not get header from parser ($PARSER --csv)" >&2
    exit 1
fi
echo "build,ranks,ppn,nodes_used,nl,nh,total_humans,timesteps,work,seed,status,${PARSER_HEADER}" > "$COUNTS_CSV"

# Number of comma-separated count columns, used to build an empty fallback row.
NCOUNT_COLS=$(awk -F, '{print NF}' <<< "$PARSER_HEADER")
EMPTY_COUNTS=$(printf ',%.0s' $(seq 1 $((NCOUNT_COLS - 1))))

{
    echo "RANKS=(${RANKS[*]})"
    echo "NODES=$NODES"
    echo "LOCATIONS_PER_RANK=$LOCATIONS_PER_RANK"
    echo "HUMANS_PER_RANK=$HUMANS_PER_RANK"
    echo "TIMESTEPS=$TIMESTEPS"
    echo "WORK=$WORK"
    echo "SEED=$SEED"
    echo "BUILDS:"
    for name in "${BUILD_ORDER[@]}"; do
        echo "  $name = ${BUILDS[$name]}"
    done
    echo "MPI_PREFIX=$MPI_PREFIX"
    echo "started=$(date -Iseconds)"
} > "$OUT_DIR/params.txt"

echo "Output dir: $OUT_DIR"
echo

# ---------- Helpers ----------
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

    compute_placement "$p"

    local nl=$LOCATIONS_PER_RANK
    local nh=$HUMANS_PER_RANK
    local total=$(( nh * p ))

    local p_pad
    p_pad=$(printf "%04d" "$p")
    local log_rel="${name}_p${p_pad}.log"
    local log_path="$OUT_DIR/$log_rel"

    echo ">>> [$name] ranks=$p ppn=$PPN nodes=$NODES_USED  (nl=$nl, nh=$nh, total_humans=$total) -> $log_rel"

    local map_arg=""
    if (( p > 1 )) && (( p % NODES_USED == 0 )); then
        map_arg="--map-by ppr:${PPN}:node"
    fi

    {
        echo "# build=$name"
        echo "# ranks=$p ppn=$PPN nodes_used=$NODES_USED"
        echo "# nl=$nl nh=$nh total_humans=$total"
        echo "# timesteps=$TIMESTEPS work=$WORK seed=$SEED"
        echo "# bin=$bin"
        echo "# started=$(date -Iseconds)"
        echo "# ---- begin output ----"
    } > "$log_path"

    mpirun -n "$p" --hostfile "$COBALT_NODEFILE" \
        --prefix "$MPI_PREFIX" \
        $map_arg \
        -x LD_LIBRARY_PATH -x PATH \
        "$bin" \
        --nl "$nl" \
        --nh "$nh" \
        --time "$TIMESTEPS" \
        --seed "$SEED" \
        --work "$WORK" \
        --quiet \
        >> "$log_path" 2>&1
    local rc=$?

    local status="ok"
    if [[ $rc -ne 0 ]]; then
        status="exit_${rc}"
    fi

    {
        echo "# ---- end output ----"
        echo "# rc=$rc status=$status"
        echo "# finished=$(date -Iseconds)"
    } >> "$log_path"

    echo "${name},${p},${PPN},${NODES_USED},${nl},${nh},${total},${TIMESTEPS},${WORK},${SEED},${log_rel},${status}" >> "$INDEX_CSV"

    # Parse comm counts from the log and append a row to the counts CSV.
    local counts_line=""
    if [[ "$status" == "ok" ]]; then
        # --csv prints: header on line 1, then one data line per file
        counts_line=$(python3 "$PARSER" --csv "$log_path" 2>/dev/null | tail -n 1)
        # Guard: if tail gave us the header (e.g. parser printed nothing else), discard it.
        if [[ "$counts_line" == "$PARSER_HEADER" ]]; then
            counts_line=""
        fi
    fi
    if [[ -z "$counts_line" ]]; then
        counts_line="$EMPTY_COUNTS"
    fi
    echo "${name},${p},${PPN},${NODES_USED},${nl},${nh},${total},${TIMESTEPS},${WORK},${SEED},${status},${counts_line}" >> "$COUNTS_CSV"
}

# ---------- Sweep ----------
for p in "${RANKS[@]}"; do
    for name in "${BUILD_ORDER[@]}"; do
        bin="${BUILDS[$name]}"
        run_one "$name" "$bin" "$p"
    done
done

echo
echo "Done."
echo "Logs        : $OUT_DIR"
echo "Index       : $INDEX_CSV"
echo "Comm counts : $COUNTS_CSV"
