#!/usr/bin/env bash
# Weak scaling experiment: zombies DSTM vs MPI.
# Per-rank load is held constant; total work grows with rank count.
#
# Output: weak_scaling_zombies.csv
#
# Usage:
#   ./run_weak_scaling.sh                  # default 2 trials
#   TRIALS=3 ./run_weak_scaling.sh         # override trials
#   OUT=foo.csv ./run_weak_scaling.sh      # override output path
#
# Requires: $COBALT_NODEFILE pointing at >= 8 nodes, mpirun in PATH.

set -u  # don't set -e: we want to keep going even if one config fails

# ---------- config ----------
TRIALS="${TRIALS:-2}"
OUT="${OUT:-weak_scaling_zombies.csv}"

# Per-rank workload (held constant for weak scaling).
NL=64
NH=32
WORK=1000
TIME=72
SEED=42

# Path to the two binaries.
DSTM_BIN="./../../dstm_zombies/zombies"
MPI_BIN="./../../mpi_zombies/zombies"

# Rank counts and matching (ranks, ppn, nodes_used) — mirrors airport sweep.
#   ranks  ppn  nodes
CONFIGS=(
    "1     1    1"
    "2     1    2"
    "4     1    4"
    "8     1    8"
    "16    2    8"
    "32    4    8"
    "64    8    8"
    "128   16   8"
    "256   32   8"
)

# ---------- sanity ----------
if [[ -z "${COBALT_NODEFILE:-}" ]]; then
    echo "ERROR: COBALT_NODEFILE not set. Run inside a Cobalt allocation." >&2
    exit 1
fi
if [[ ! -r "$COBALT_NODEFILE" ]]; then
    echo "ERROR: COBALT_NODEFILE=$COBALT_NODEFILE not readable." >&2
    exit 1
fi
for bin in "$DSTM_BIN" "$MPI_BIN"; do
    if [[ ! -x "$bin" ]]; then
        echo "ERROR: binary not found or not executable: $bin" >&2
        exit 1
    fi
done

MPI_PREFIX="$(dirname "$(dirname "$(which mpirun)")")"

# ---------- output ----------
echo "build,ranks,ppn,nodes_used,nl,nh,work,timesteps,trial,elapsed_s,steps_per_s,status" > "$OUT"
echo "writing results to $OUT"

# ---------- helpers ----------
run_one() {
    local build="$1"     # dstm | mpi
    local bin="$2"
    local ranks="$3"
    local ppn="$4"
    local nodes="$5"
    local trial="$6"

    local logfile
    logfile="$(mktemp -t zombies_${build}_r${ranks}_t${trial}.XXXXXX.log)"

    echo ">> [$build] ranks=$ranks ppn=$ppn nodes=$nodes trial=$trial"

    # Build the mpirun command per TODO.md conventions.
    # --map-by ppr:PPN:node ensures PPN ranks per node.
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
        --quiet \
        > "$logfile" 2>&1
    local rc=$?

    local elapsed="" steps="" status="ok"
    if [[ $rc -ne 0 ]]; then
        status="exit_${rc}"
    fi

    # Parse "Elapsed time:       0.0130 seconds"
    elapsed="$(grep -E '^Elapsed time:' "$logfile" \
               | tail -n1 \
               | awk '{print $3}')"
    # Parse "Timesteps/second:   9852.79"
    steps="$(grep -E '^Timesteps/second:' "$logfile" \
             | tail -n1 \
             | awk '{print $2}')"

    if [[ -z "$elapsed" || -z "$steps" ]]; then
        status="parse_fail"
        elapsed="${elapsed:-NA}"
        steps="${steps:-NA}"
        echo "   !! parse failed; see $logfile" >&2
    else
        # keep logs on failure only; remove on success to save space
        rm -f "$logfile"
    fi

    echo "${build},${ranks},${ppn},${nodes},${NL},${NH},${WORK},${TIME},${trial},${elapsed},${steps},${status}" \
        >> "$OUT"
}

# ---------- main sweep ----------
for cfg in "${CONFIGS[@]}"; do
    read -r RANKS PPN NODES <<<"$cfg"
    for trial in $(seq 1 "$TRIALS"); do
        run_one "dstm" "$DSTM_BIN" "$RANKS" "$PPN" "$NODES" "$trial"
        run_one "mpi"  "$MPI_BIN"  "$RANKS" "$PPN" "$NODES" "$trial"
    done
done

echo
echo "done. results in $OUT"
