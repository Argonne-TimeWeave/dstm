#!/usr/bin/env bash
# Weak-scaling sweep recording DSTM version activity per run.
#
# Per-rank load held constant; total airports grow with rank count.
# For each run we pipe mpirun output through parse_versions.py and record:
#   - one row in the summary CSV with global min/max/avg active versions
#   - one detailed per-(rank,t) CSV in details/ for later analysis
#
# Only the DSTM build is run; MPI has no version concept.
#
# Builds expected at:
#   ../../airports/airports
#
# Output:
#   weak_versions_<params>.csv     (summary, one row per run)
#   details/<run_tag>.csv          (per-(rank,t) detail, one file per run)

set -u

# ---------- Config ----------
RANKS=(1 2 4 8 16 32 64 128 256)
NODES=8
AIRPORTS_PER_RANK=64
TIMESTEPS=64
TRIALS=1
WORK=2500
SEED=42

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

DSTM_BIN="$EXAMPLES_DIR/airports/airports"
PARSER="$SCRIPT_DIR/parse_versions.py"

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

if [[ ! -x "$DSTM_BIN" ]]; then
    echo "ERROR: missing or non-executable build: $DSTM_BIN" >&2
    exit 1
fi
if [[ ! -f "$PARSER" ]]; then
    echo "ERROR: parser not found: $PARSER" >&2
    exit 1
fi

# Prefer the active python (works inside the (dstm) env you're in).
PYTHON_BIN="$(command -v python3 || command -v python)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: no python found on PATH" >&2
    exit 1
fi

MPI_PREFIX="$(dirname "$(dirname "$(which mpirun)")")"
echo "MPI prefix:  $MPI_PREFIX"
echo "Python:      $PYTHON_BIN"
echo "Parser:      $PARSER"
echo "Airports per rank (fixed): $AIRPORTS_PER_RANK"
echo "Largest run total airports: $(( AIRPORTS_PER_RANK * ${RANKS[-1]} ))"
echo

# ---------- Output paths ----------
RANKS_MIN=${RANKS[0]}
RANKS_MAX=${RANKS[-1]}
RANKS_COUNT=${#RANKS[@]}
BASE="weak_versions_n${NODES}_apr${AIRPORTS_PER_RANK}_t${TIMESTEPS}_w${WORK}_tr${TRIALS}_s${SEED}_r${RANKS_MIN}-${RANKS_MAX}x${RANKS_COUNT}"
CSV="$SCRIPT_DIR/${BASE}.csv"
if [[ -e "$CSV" ]]; then
    i=2
    while [[ -e "$SCRIPT_DIR/${BASE}_${i}.csv" ]]; do i=$(( i + 1 )); done
    CSV="$SCRIPT_DIR/${BASE}_${i}.csv"
fi

DETAILS_DIR="$SCRIPT_DIR/details_${BASE}"
mkdir -p "$DETAILS_DIR"

echo "Summary CSV: $CSV"
echo "Details dir: $DETAILS_DIR"
echo

echo "build,ranks,ppn,nodes_used,na,total_airports,timesteps,work,trial,v_min,v_max,v_avg,n_samples,status" > "$CSV"

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
    local p="$1"
    local trial="$2"

    compute_placement "$p"
    local na=$AIRPORTS_PER_RANK
    local total=$(( na * p ))
    local tag="dstm_r${p}_ppn${PPN}_n${NODES_USED}_tr${trial}"

    echo ">>> [dstm] ranks=$p ppn=$PPN nodes=$NODES_USED trial=$trial  (na=$na, total=$total)"

    local map_arg=""
    if (( p > 1 )) && (( p % NODES_USED == 0 )); then
        map_arg="--map-by ppr:${PPN}:node"
    fi

    # Workdir so parse_versions.py's default "stdin.versions.csv" lands here.
    local workdir
    workdir=$(mktemp -d "$SCRIPT_DIR/.vrun_XXXXXX")
    local parser_log="$workdir/parser.out"
    local mpirun_err="$workdir/mpirun.err"

    # Pipe mpirun stdout+stderr straight into the parser to avoid materializing
    # the full --info log. mpirun stderr also goes through the parser; non-prefix
    # lines are ignored by it. We keep a separate stderr copy for diagnostics
    # via a tee on the err stream.
    set +e
    mpirun -n "$p" --hostfile "$COBALT_NODEFILE" \
        --prefix "$MPI_PREFIX" \
        $map_arg \
        -x LD_LIBRARY_PATH -x PATH \
        "$DSTM_BIN" \
        --na "$na" \
        --time "$TIMESTEPS" \
        --seed "$SEED" \
        --work "$WORK" \
        --info \
        2> >(tee "$mpirun_err" >&2) \
        | (cd "$workdir" && "$PYTHON_BIN" "$PARSER" -) > "$parser_log" 2>&1
    # PIPESTATUS[0] = mpirun rc, PIPESTATUS[1] = parser rc
    local -a pstat=( "${PIPESTATUS[@]}" )
    local mpi_rc=${pstat[0]:-0}
    local parse_rc=${pstat[1]:-0}

    local status="ok"
    if (( mpi_rc != 0 )); then
        status="mpi_exit_${mpi_rc}"
    elif (( parse_rc != 0 )); then
        status="parse_exit_${parse_rc}"
    fi

    # Parse the parser's stdout for the Global active-versions line, e.g.:
    #   Global active-versions: min=0 max=4 avg=1.23  (across 512 rank-tick samples)
    local v_min="" v_max="" v_avg="" n_samples=""
    local gline
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
        # parser may have emitted "No version events found" to stderr;
        # treat that as a measurement failure, not a run failure.
        if grep -q 'No version events found' "$parser_log"; then
            status="${status}+no_versions"
        else
            status="${status}+no_global_line"
        fi
    fi

    # Move the detail CSV (written by parser as stdin.versions.csv in workdir)
    # into the details dir under a descriptive name.
    local per_rank_src="$workdir/stdin.versions.per_rank.csv"
    local per_rank_dst="$DETAILS_DIR/${tag}.per_rank.csv"
    if [[ -f "$per_rank_src" ]]; then
        mv "$per_rank_src" "$per_rank_dst"
    fi
    rm -f "$workdir/stdin.versions.csv"

    echo "dstm,${p},${PPN},${NODES_USED},${na},${total},${TIMESTEPS},${WORK},${trial},${v_min},${v_max},${v_avg},${n_samples},${status}" >> "$CSV"

    if [[ "$status" == "ok" ]]; then
        rm -rf "$workdir"
    else
        echo "    kept $workdir for inspection (status=$status)"
    fi
}

# ---------- Sweep ----------
for p in "${RANKS[@]}"; do
    for trial in $(seq 1 "$TRIALS"); do
        run_one "$p" "$trial"
    done
done

echo
echo "Done."
echo "Summary CSV: $CSV"
echo "Details dir: $DETAILS_DIR"
