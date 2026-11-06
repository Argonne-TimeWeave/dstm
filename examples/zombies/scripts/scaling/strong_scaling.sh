#!/usr/bin/env bash
# Strong scaling experiment: zombies DSTM vs MPI.
# Total problem size is held constant; per-rank load shrinks as ranks grow.
#
# Output: strong_scaling_zombies.csv
#
# Usage:
#   ./run_strong_scaling.sh                # default 4 trials
#   TRIALS=3 ./run_strong_scaling.sh       # override trials
#   OUT=foo.csv ./run_strong_scaling.sh    # override output path
#   NL_TOTAL=8192 NH_TOTAL=32768 ./run_strong_scaling.sh   # override size
#
# Requires: $COBALT_NODEFILE pointing at >= 8 nodes, mpirun in PATH.

set -u  # don't set -e: we want to keep going even if one config fails

# ---------- config ----------
TRIALS="${TRIALS:-4}"
OUT="${OUT:-strong_scaling_zombies.csv}"

# Total problem size (held constant for strong scaling).
# Defaults chosen so that at the largest rank count in CONFIGS (256), the
# per-rank values still match the original weak-scaling per-rank values
# (NL=128, NH=512), i.e. NL_TOTAL = 128 * 256, NH_TOTAL = 512 * 256.
NL_TOTAL="${NL_TOTAL:-4096}"
NH_TOTAL="${NH_TOTAL:-16384}"
WORK=0
TIME=256
SEED=42

# Path to the three binaries.
FIXED_BIN="./../../dstm_fixed/zombies"
DSTM_BIN="./../../dstm_zombies/zombies"
MPI_BIN="./../../mpi_zombies/zombies"

# Rank counts and matching (ranks, ppn, nodes_used) — mirrors weak sweep.
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
for bin in "$DSTM_BIN" "$MPI_BIN" "$FIXED_BIN"; do
    if [[ ! -x "$bin" ]]; then
        echo "ERROR: binary not found or not executable: $bin" >&2
        exit 1
    fi
done

# Sanity-check that the largest rank count divides the totals evenly.
LAST_CFG="${CONFIGS[-1]}"
read -r MAX_RANKS _ _ <<<"$LAST_CFG"
if (( NL_TOTAL % MAX_RANKS != 0 )); then
    echo "WARNING: NL_TOTAL=$NL_TOTAL not divisible by max ranks=$MAX_RANKS" >&2
fi
if (( NH_TOTAL % MAX_RANKS != 0 )); then
    echo "WARNING: NH_TOTAL=$NH_TOTAL not divisible by max ranks=$MAX_RANKS" >&2
fi

MPI_PREFIX="$(dirname "$(dirname "$(which mpirun)")")"

# ---------- output ----------
# Note: nl/nh columns hold the PER-RANK values used for that run.
# nl_total / nh_total record the fixed global problem size.
echo "build,ranks,ppn,nodes_used,nl,nh,nl_total,nh_total,work,timesteps,trial,elapsed_s,steps_per_s,status" > "$OUT"
echo "writing results to $OUT"
echo "strong scaling: NL_TOTAL=$NL_TOTAL  NH_TOTAL=$NH_TOTAL"

# ---------- helpers ----------
run_one() {
    local build="$1"     # fixed | dstm | mpi
    local bin="$2"
    local ranks="$3"
    local ppn="$4"
    local nodes="$5"
    local trial="$6"

    # Per-rank slice of the fixed total problem. Floor at 1 so tiny totals
    # don't degenerate to zero.
    local nl=$(( NL_TOTAL / ranks ))
    local nh=$(( NH_TOTAL / ranks ))
    (( nl < 1 )) && nl=1
    (( nh < 1 )) && nh=1

    local logfile
    logfile="$(mktemp -t zombies_${build}_r${ranks}_t${trial}.XXXXXX.log)"

    echo ">> [$build] ranks=$ranks ppn=$ppn nodes=$nodes trial=$trial  (nl=$nl nh=$nh)"

    mpirun -n "$ranks" --hostfile "$COBALT_NODEFILE" \
        --prefix "$MPI_PREFIX" \
        --map-by "ppr:${ppn}:node" \
        -x LD_LIBRARY_PATH -x PATH \
        "$bin" \
        --seed "$SEED" \
        --time "$TIME" \
        --nl "$nl" \
        --nh "$nh" \
        --work "$WORK" \
        --quiet \
        > "$logfile" 2>&1
    local rc=$?

    local elapsed="" steps="" status="ok"
    if [[ $rc -ne 0 ]]; then
        status="exit_${rc}"
    fi

    elapsed="$(grep -E '^Elapsed time:' "$logfile" \
               | tail -n1 \
               | awk '{print $3}')"
    steps="$(grep -E '^Timesteps/second:' "$logfile" \
             | tail -n1 \
             | awk '{print $2}')"

    if [[ -z "$elapsed" || -z "$steps" ]]; then
        status="parse_fail"
        elapsed="${elapsed:-NA}"
        steps="${steps:-NA}"
        echo "   !! parse failed; see $logfile" >&2
    else
        rm -f "$logfile"
    fi

    echo "${build},${ranks},${ppn},${nodes},${nl},${nh},${NL_TOTAL},${NH_TOTAL},${WORK},${TIME},${trial},${elapsed},${steps},${status}" \
        >> "$OUT"
}

# ---------- main sweep ----------
for cfg in "${CONFIGS[@]}"; do
    read -r RANKS PPN NODES <<<"$cfg"
    for trial in $(seq 1 "$TRIALS"); do
        run_one "fixed" "$FIXED_BIN" "$RANKS" "$PPN" "$NODES" "$trial"
        run_one "dstm"  "$DSTM_BIN"  "$RANKS" "$PPN" "$NODES" "$trial"
        run_one "mpi"   "$MPI_BIN"   "$RANKS" "$PPN" "$NODES" "$trial"
    done
done

echo
echo "done. results in $OUT"
