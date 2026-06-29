#!/usr/bin/env bash
# Weak-scaling memory sweep: zombies fixed / DSTM / MPI.
#
# Per-rank load held constant; total work grows with rank count.
# Records peak resident memory per rank, aggregated as:
#   rss_max_mb : max peak RSS across all ranks      (worst single rank)
#   rss_sum_mb : sum of peak RSS across all ranks   (total footprint)
# Plus the same for VmPeak (virtual size) as a sanity check.
#
# Measurement: no binary modification. A lightweight sampler runs on every
# node (one per node) polling /proc/<pid>/status for every zombies process
# and tracking each PID's peak VmHWM and VmPeak. After the run, per-node
# files are aggregated.
#
# Usage:
#   ./weak_memory_zombies.sh
#   OUT=foo.csv ./weak_memory_zombies.sh
#
# Requires: $COBALT_NODEFILE pointing at >= 8 nodes, mpirun in PATH.

set -u

# ---------- Config ----------
# Per-rank workload (held constant for weak scaling).
NL=128
NH=512
WORK=0
TIME=256
SEED=42

SAMPLE_MS=50           # sampler poll interval in ms
BIN_COMM="zombies"     # /proc/<pid>/status Name: field to match

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FIXED_BIN="$SCRIPT_DIR/../../dstm_fixed/zombies"
DSTM_BIN="$SCRIPT_DIR/../../dstm_zombies/zombies"
MPI_BIN="$SCRIPT_DIR/../../mpi_zombies/zombies"

declare -A BUILDS=(
    [fixed]="$FIXED_BIN"
    [dstm]="$DSTM_BIN"
    [mpi]="$MPI_BIN"
)
BUILD_ORDER=(fixed dstm mpi)

# ranks  ppn  nodes
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

# ---------- Sanity ----------
if [[ -z "${COBALT_NODEFILE:-}" ]]; then
    echo "ERROR: COBALT_NODEFILE not set. Run inside a Cobalt allocation." >&2
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

MPI_PREFIX="$(dirname "$(dirname "$(which mpirun)")")"
echo "MPI prefix: $MPI_PREFIX"
echo "Per-rank nl=$NL nh=$NH work=$WORK time=$TIME seed=$SEED"
echo "Sampler interval: ${SAMPLE_MS} ms, matching comm '${BIN_COMM}'"
echo

# ---------- Output filename ----------
# Mirror the airport memory script's naming style, with config-derived suffix.
RANKS_LIST=()
for cfg in "${CONFIGS[@]}"; do
    read -r R _ _ <<<"$cfg"
    RANKS_LIST+=("$R")
done
RANKS_MIN=${RANKS_LIST[0]}
RANKS_MAX=${RANKS_LIST[-1]}
RANKS_COUNT=${#RANKS_LIST[@]}

if [[ -n "${OUT:-}" ]]; then
    CSV="$OUT"
else
    BASE="weak_mem_zombies_nl${NL}_nh${NH}_t${TIME}_w${WORK}_s${SEED}_r${RANKS_MIN}-${RANKS_MAX}x${RANKS_COUNT}"
    CSV="$SCRIPT_DIR/${BASE}.csv"
    if [[ -e "$CSV" ]]; then
        i=2
        while [[ -e "$SCRIPT_DIR/${BASE}_${i}.csv" ]]; do i=$(( i + 1 )); done
        CSV="$SCRIPT_DIR/${BASE}_${i}.csv"
    fi
fi
echo "CSV output: $CSV"

echo "build,ranks,ppn,nodes_used,nl,nh,work,timesteps,n_procs_seen,rss_max_mb,rss_sum_mb,vsz_max_mb,vsz_sum_mb,status" > "$CSV"

# ---------- Per-node sampler ----------
# Writes one file per node: $OUTDIR/mem_<hostname>.txt
# Each line:   <pid> <peak_vmhwm_kb> <peak_vmpeak_kb>
# Must live on a shared FS so every node can exec it.
SAMPLER_SCRIPT="$SCRIPT_DIR/.sampler_$$.sh"
cat > "$SAMPLER_SCRIPT" <<'SAMPLER'
#!/usr/bin/env bash
# args: <outdir> <sample_ms> <bin_comm>
set -u
OUTDIR="$1"
SAMPLE_MS="$2"
BIN_COMM="$3"
HOST="$(hostname -s)"
OUT="$OUTDIR/mem_${HOST}.txt"
: > "$OUT"

declare -A HWM
declare -A VPK

flush_pid() {
    local pid="$1"
    if [[ -n "${HWM[$pid]:-}" ]]; then
        printf '%s %s %s\n' "$pid" "${HWM[$pid]}" "${VPK[$pid]:-0}" >> "$OUT"
        unset 'HWM[$pid]'
        unset 'VPK[$pid]'
    fi
}

cleanup() {
    for pid in "${!HWM[@]}"; do flush_pid "$pid"; done
    exit 0
}
trap cleanup TERM INT

SLEEP_S=$(awk -v ms="$SAMPLE_MS" 'BEGIN{printf "%.4f", ms/1000.0}')

while :; do
    declare -A SEEN=()
    for d in /proc/[0-9]*; do
        [[ -r "$d/status" ]] || continue
        read -r _ name < "$d/status" 2>/dev/null || continue
        [[ "$name" == "$BIN_COMM" ]] || continue
        pid="${d#/proc/}"
        SEEN[$pid]=1

        hwm=0; vpk=0
        while IFS= read -r line; do
            case "$line" in
                VmHWM:*)  hwm=${line//[^0-9]/} ;;
                VmPeak:*) vpk=${line//[^0-9]/} ;;
            esac
        done < "$d/status"

        if [[ -z "${HWM[$pid]:-}" ]] || (( hwm > HWM[$pid] )); then
            HWM[$pid]=$hwm
        fi
        if [[ -z "${VPK[$pid]:-}" ]] || (( vpk > VPK[$pid] )); then
            VPK[$pid]=$vpk
        fi
    done

    for pid in "${!HWM[@]}"; do
        [[ -z "${SEEN[$pid]:-}" ]] && flush_pid "$pid"
    done

    sleep "$SLEEP_S"
done
SAMPLER
chmod +x "$SAMPLER_SCRIPT"

run_one() {
    local name="$1"
    local bin="$2"
    local ranks="$3"
    local ppn="$4"
    local nodes="$5"

    echo ">>> [$name] ranks=$ranks ppn=$ppn nodes=$nodes"

    local rundir
    rundir=$(mktemp -d "$SCRIPT_DIR/.memrun_XXXXXX")

    # Launch one sampler per node, in background.
    mpirun -n "$nodes" --hostfile "$COBALT_NODEFILE" \
        --prefix "$MPI_PREFIX" \
        --map-by ppr:1:node \
        -x LD_LIBRARY_PATH -x PATH \
        "$SAMPLER_SCRIPT" "$rundir" "$SAMPLE_MS" "$BIN_COMM" \
        > "$rundir/sampler.log" 2>&1 &
    local sampler_pid=$!

    # Give samplers a beat to start before the real workload.
    sleep 0.3

    local runlog="$rundir/run.log"
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
        > "$runlog" 2>&1
    local rc=$?

    # Stop samplers (TERM triggers their cleanup/flush).
    mpirun -n "$nodes" --hostfile "$COBALT_NODEFILE" \
        --prefix "$MPI_PREFIX" \
        --map-by ppr:1:node \
        -x LD_LIBRARY_PATH -x PATH \
        bash -c "pkill -TERM -f '$SAMPLER_SCRIPT' >/dev/null 2>&1; sleep 0.2; pkill -KILL -f '$SAMPLER_SCRIPT' >/dev/null 2>&1; true" \
        >> "$rundir/sampler.log" 2>&1 || true

    wait "$sampler_pid" 2>/dev/null || true

    # Aggregate per-node files.
    local n_procs=0
    local rss_max_kb=0 rss_sum_kb=0 vsz_max_kb=0 vsz_sum_kb=0
    shopt -s nullglob
    for f in "$rundir"/mem_*.txt; do
        while read -r pid hwm vpk; do
            [[ -z "$pid" ]] && continue
            n_procs=$(( n_procs + 1 ))
            rss_sum_kb=$(( rss_sum_kb + hwm ))
            vsz_sum_kb=$(( vsz_sum_kb + vpk ))
            (( hwm > rss_max_kb )) && rss_max_kb=$hwm
            (( vpk > vsz_max_kb )) && vsz_max_kb=$vpk
        done < "$f"
    done
    shopt -u nullglob

    local status="ok"
    if [[ $rc -ne 0 ]]; then
        status="exit_${rc}"
    elif (( n_procs != ranks )); then
        status="proc_count_mismatch_${n_procs}_vs_${ranks}"
    fi

    local rss_max_mb rss_sum_mb vsz_max_mb vsz_sum_mb
    rss_max_mb=$(awk -v k="$rss_max_kb" 'BEGIN{printf "%.2f", k/1024.0}')
    rss_sum_mb=$(awk -v k="$rss_sum_kb" 'BEGIN{printf "%.2f", k/1024.0}')
    vsz_max_mb=$(awk -v k="$vsz_max_kb" 'BEGIN{printf "%.2f", k/1024.0}')
    vsz_sum_mb=$(awk -v k="$vsz_sum_kb" 'BEGIN{printf "%.2f", k/1024.0}')

    echo "${name},${ranks},${ppn},${nodes},${NL},${NH},${WORK},${TIME},${n_procs},${rss_max_mb},${rss_sum_mb},${vsz_max_mb},${vsz_sum_mb},${status}" >> "$CSV"

    if [[ "$status" == "ok" ]]; then
        rm -rf "$rundir"
    else
        echo "    kept $rundir for inspection (status=$status)"
    fi
}

cleanup_all() {
    rm -f "$SAMPLER_SCRIPT"
    rm -f "$SCRIPT_DIR"/.sampler_*.sh 2>/dev/null || true
}
trap cleanup_all EXIT

# ---------- Sweep ----------
for cfg in "${CONFIGS[@]}"; do
    read -r RANKS PPN NODES <<<"$cfg"
    for name in "${BUILD_ORDER[@]}"; do
        bin="${BUILDS[$name]}"
        run_one "$name" "$bin" "$RANKS" "$PPN" "$NODES"
    done
done

echo
echo "Done."
echo "CSV : $CSV"
