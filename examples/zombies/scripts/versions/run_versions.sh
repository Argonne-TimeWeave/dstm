#!/usr/bin/env bash
# Weak-scaling sweep recording version activity for the zombies sim.
#
# Per-rank load held constant; total work grows with rank count.
# For each run we pipe mpirun output through parse_versions.py and record:
#   - one row in the summary CSV with global min/max/avg active versions
#   - one per-rank summary CSV in details/ for later analysis
#
# Runs both dstm_fixed and dstm_zombies (both have the versions concept).
# The MPI build is skipped (no versions).
#
# Builds expected at:
#   ../../dstm_fixed/zombies
#   ../../dstm_zombies/zombies
#
# Output:
#   weak_versions_zombies_<params>.csv      (summary, one row per run)
#   details_<params>/<run_tag>.per_rank.csv (per-rank detail, one file per run)

set -u

# ---------- Config ----------
TRIALS="${TRIALS:-1}"

# Per-rank workload (held constant for weak scaling).
NL=128
NH=512
WORK=0
TIME=256
SEED=42

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FIXED_BIN="$SCRIPT_DIR/../../dstm_fixed/zombies"
DSTM_BIN="$SCRIPT_DIR/../../dstm_zombies/zombies"
PARSER="$SCRIPT_DIR/parse_versions.py"

declare -A BUILDS=(
    [fixed]="$FIXED_BIN"
    [dstm]="$DSTM_BIN"
)
BUILD_ORDER=(fixed dstm)

# ranks  ppn  nodes
CONFIGS=(
    "2     1    2"
    "4     1    4"
    "8     1    8"
    "16    2    8"
    "32    4    8"
    "64    8    8"
    "128   16   8"
    "256   32   8"
)

# ---------- Sanity checks ----------
if [[ -z "${COBALT_NODEFILE:-}" ]]; then
    echo "ERROR: \$COBALT_NODEFILE not set. Run inside a Cobalt allocation." >&2
    exit 1
fi
if [[ ! -r "$COBALT_NODEFILE" ]]; then
    echo "ERROR: COBALT_NODEFILE=$COBALT_NODEFILE not readable." >&2
    exit 1
fi

ALLOCATED_NODES=$(wc -l < "$COBALT_NODEFILE")
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
if [[ ! -f "$PARSER" ]]; then
    echo "ERROR: parser not found: $PARSER" >&2
    exit 1
fi

PYTHON_BIN="$(command -v python3 || command -v python)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: no python found on PATH" >&2
    exit 1
fi

MPI_PREFIX="$(dirname "$(dirname "$(which mpirun)")")"
echo "MPI prefix:  $MPI_PREFIX"
echo "Python:      $PYTHON_BIN"
echo "Parser:      $PARSER"
echo "Per-rank nl=$NL nh=$NH work=$WORK time=$TIME seed=$SEED"
echo

# ---------- Output paths ----------
RANKS_LIST=()
for cfg in "${CONFIGS[@]}"; do
    read -r R _ _ <<<"$cfg"
    RANKS_LIST+=("$R")
done
RANKS_MIN=${RANKS_LIST[0]}
RANKS_MAX=${RANKS_LIST[-1]}
RANKS_COUNT=${#RANKS_LIST[@]}

BASE="weak_versions_zombies_nl${NL}_nh${NH}_t${TIME}_w${WORK}_tr${TRIALS}_s${SEED}_r${RANKS_MIN}-${RANKS_MAX}x${RANKS_COUNT}"
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

echo "build,ranks,ppn,nodes_used,nl,nh,work,timesteps,trial,v_min,v_max,v_avg,n_samples,status" > "$CSV"

# ---------- Helpers ----------
run_one() {
    local name="$1"
    local bin="$2"
    local ranks="$3"
    local ppn="$4"
    local nodes="$5"
    local trial="$6"

    local tag="${name}_r${ranks}_ppn${ppn}_n${nodes}_tr${trial}"

    echo ">>> [$name] ranks=$ranks ppn=$ppn nodes=$nodes trial=$trial"

    local workdir
    workdir=$(mktemp -d "$SCRIPT_DIR/.vrun_XXXXXX")
    local parser_log="$workdir/parser.out"
    local mpirun_err="$workdir/mpirun.err"

    # Pipe mpirun stdout+stderr into the parser to avoid materializing the
    # full --info log on disk. mpirun stderr is also tee'd for diagnostics.
    mpirun -n "$ranks" --hostfile "$COBALT_NODEFILE" \
        --prefix "$MPI_PREFIX" \
        --map-by "ppr:${ppn}:node" \
        -x LD_LIBRARY_PATH -x PATH \
        "$bin" \
        --seed "$SEED" \
        --time "$TIME" \
        --nl "$NL" \
        --nh "$NH" \
        --work "$WORK" \
        --info \
        2> >(tee "$mpirun_err" >&2) \
        | (cd "$workdir" && "$PYTHON_BIN" "$PARSER" -) > "$parser_log" 2>&1
    local -a pstat=( "${PIPESTATUS[@]}" )
    local mpi_rc=${pstat[0]:-0}
    local parse_rc=${pstat[1]:-0}

    local status="ok"
    if (( mpi_rc != 0 )); then
        status="mpi_exit_${mpi_rc}"
    elif (( parse_rc != 0 )); then
        status="parse_exit_${parse_rc}"
    fi

    # Parse the Global active-versions line from parser stdout, e.g.:
    #   Global active-versions: min=0 max=4 avg=1.23  (across 512 rank-tick samples)
    local v_min="" v_max="" v_avg="" n_samples=""
    local gline
    gline=$(grep -E '^Global active-versions:' "$parser_log" | tail -n1)
    if [[ -n "$gline" ]]; then
        v_min=$(    echo "$gline" | sed -nE 's/.*min=([0-9]+).*/\1/p')
        v_max=$(    echo "$gline" | sed -nE 's/.*max=([0-9]+).*/\1/p')
        v_avg=$(    echo "$gline" | sed -nE 's/.*avg=([0-9.]+).*/\1/p')
        n_samples=$(echo "$gline" | sed -nE 's/.*across ([0-9]+) rank-tick.*/\1/p')
        if [[ -z "$v_min" || -z "$v_max" || -z "$v_avg" ]]; then
            status="${status}+parse_global_fail"
        fi
    else
        if grep -q 'No version events found' "$parser_log"; then
            status="${status}+no_versions"
        else
            status="${status}+no_global_line"
        fi
    fi

    # Keep per-rank summary, discard per-(rank, t) detail.
    local per_rank_src="$workdir/stdin.versions.per_rank.csv"
    local per_rank_dst="$DETAILS_DIR/${tag}.per_rank.csv"
    if [[ -f "$per_rank_src" ]]; then
        mv "$per_rank_src" "$per_rank_dst"
    fi
    rm -f "$workdir/stdin.versions.csv"

    echo "${name},${ranks},${ppn},${nodes},${NL},${NH},${WORK},${TIME},${trial},${v_min},${v_max},${v_avg},${n_samples},${status}" >> "$CSV"

    if [[ "$status" == "ok" ]]; then
        rm -rf "$workdir"
    else
        echo "    kept $workdir for inspection (status=$status)"
    fi
}

# ---------- Sweep ----------
for cfg in "${CONFIGS[@]}"; do
    read -r RANKS PPN NODES <<<"$cfg"
    for trial in $(seq 1 "$TRIALS"); do
        for name in "${BUILD_ORDER[@]}"; do
            bin="${BUILDS[$name]}"
            run_one "$name" "$bin" "$RANKS" "$PPN" "$NODES" "$trial"
        done
    done
done

echo
echo "Done."
echo "Summary CSV: $CSV"
echo "Details dir: $DETAILS_DIR"
