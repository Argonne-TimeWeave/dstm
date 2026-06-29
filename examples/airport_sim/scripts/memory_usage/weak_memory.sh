#!/usr/bin/env bash
# Weak-scaling memory sweep for the DSTM and MPI airport simulators.
#
# Same sweep shape as the timing script, but instead of elapsed time we
# record peak resident memory per rank, aggregated as:
#   rss_max_mb : max peak RSS across all ranks      (worst single rank)
#   rss_sum_mb : sum of peak RSS across all ranks   (total footprint)
# Plus the same for VmPeak (virtual size) as a sanity check.
#
# Measurement strategy: no binary modification. A lightweight sampler is
# launched on every node (one per node) that polls /proc/<pid>/status for
# every airports process on that node and tracks each PID's peak VmHWM
# and VmPeak. After the run, per-node files are aggregated.
#
# Builds expected at:
#   ../airports/airports     (DSTM)
#   ../mpi_air/airports      (MPI send/recv)
#
# Output: weak_mem_<params>.csv next to this script.

set -u

# ---------- Config ----------
RANKS=(1 2 4 8 16 32 64 128 256)
NODES=8
AIRPORTS_PER_RANK=64
TIMESTEPS=64
TRIALS=1
WORK=2500
SEED=42

SAMPLE_MS=50           # sampler poll interval in ms
BIN_COMM="airports"    # /proc/<pid>/status Name: field to match

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

declare -A BUILDS=(
    [dstm]="$EXAMPLES_DIR/airports/airports"
    [mpi]="$EXAMPLES_DIR/mpi_air/airports"
)
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
echo "Sampler interval: ${SAMPLE_MS} ms, matching comm '${BIN_COMM}'"
echo

# ---------- Output filename ----------
RANKS_MIN=${RANKS[0]}
RANKS_MAX=${RANKS[-1]}
RANKS_COUNT=${#RANKS[@]}
BASE="weak_mem_n${NODES}_apr${AIRPORTS_PER_RANK}_t${TIMESTEPS}_w${WORK}_tr${TRIALS}_s${SEED}_r${RANKS_MIN}-${RANKS_MAX}x${RANKS_COUNT}"
CSV="$SCRIPT_DIR/${BASE}.csv"
if [[ -e "$CSV" ]]; then
    i=2
    while [[ -e "$SCRIPT_DIR/${BASE}_${i}.csv" ]]; do i=$(( i + 1 )); done
    CSV="$SCRIPT_DIR/${BASE}_${i}.csv"
fi
echo "CSV output: $CSV"

echo "build,ranks,ppn,nodes_used,na,total_airports,timesteps,work,trial,n_procs_seen,rss_max_mb,rss_sum_mb,vsz_max_mb,vsz_sum_mb,status" > "$CSV"

# ---------- Per-node sampler ----------
# Writes one file per node: $OUTDIR/mem_<hostname>.txt
# Each line:   <pid> <peak_vmhwm_kb> <peak_vmpeak_kb>
# Strategy: track per-PID running max of VmHWM and VmPeak across samples.
# When a PID disappears, flush its final values. On exit, flush all remaining.
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

# sleep duration in seconds (floating point)
SLEEP_S=$(awk -v ms="$SAMPLE_MS" 'BEGIN{printf "%.4f", ms/1000.0}')

while :; do
    # Snapshot current matching pids
    declare -A SEEN=()
    for d in /proc/[0-9]*; do
        [[ -r "$d/status" ]] || continue
        # Quick name check on first line of status
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

    # Flush any PIDs that disappeared since last sample
    for pid in "${!HWM[@]}"; do
        [[ -z "${SEEN[$pid]:-}" ]] && flush_pid "$pid"
    done

    sleep "$SLEEP_S"
done
SAMPLER
chmod +x "$SAMPLER_SCRIPT"

compute_placement() {
    local p="$1"
    if (( p < NODES )); then
        PPN=1
        NODES_USED=$p
    else
        PPN=$(( p / NODES ))
        NODES_USED=$NODES
        if (( p % NODES != 0 )); then
            echo "WARNING: P=$p not divisible by NODES=$NODES" >&2
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

    echo ">>> [$name] ranks=$p ppn=$PPN nodes=$NODES_USED trial=$trial (na=$na, total=$total)"

    local map_arg=""
    if (( p > 1 )) && (( p % NODES_USED == 0 )); then
        map_arg="--map-by ppr:${PPN}:node"
    fi

    # Per-run shared dir for sampler output. Must be visible from every node;
    # script dir is assumed to be on a shared filesystem (same assumption as
    # the timing script, which writes the CSV there).
    local rundir
    rundir=$(mktemp -d "$SCRIPT_DIR/.memrun_XXXXXX")

    # Launch one sampler per node, in background.
    mpirun -n "$NODES_USED" --hostfile "$COBALT_NODEFILE" \
        --prefix "$MPI_PREFIX" \
        --map-by ppr:1:node \
        -x LD_LIBRARY_PATH -x PATH \
        "$SAMPLER_SCRIPT" "$rundir" "$SAMPLE_MS" "$BIN_COMM" \
        > "$rundir/sampler.log" 2>&1 &
    local sampler_pid=$!

    # Give samplers a beat to start before the real workload.
    sleep 0.3

    local runlog="$rundir/run.log"
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
        > "$runlog" 2>&1
    local rc=$?

    # Stop samplers (TERM triggers their cleanup/flush). Use pkill via a
    # one-shot mpirun to hit every node.
    mpirun -n "$NODES_USED" --hostfile "$COBALT_NODEFILE" \
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
    elif (( n_procs != p )); then
        status="proc_count_mismatch_${n_procs}_vs_${p}"
    fi

    # KB -> MB with 2 decimals
    local rss_max_mb rss_sum_mb vsz_max_mb vsz_sum_mb
    rss_max_mb=$(awk -v k="$rss_max_kb" 'BEGIN{printf "%.2f", k/1024.0}')
    rss_sum_mb=$(awk -v k="$rss_sum_kb" 'BEGIN{printf "%.2f", k/1024.0}')
    vsz_max_mb=$(awk -v k="$vsz_max_kb" 'BEGIN{printf "%.2f", k/1024.0}')
    vsz_sum_mb=$(awk -v k="$vsz_sum_kb" 'BEGIN{printf "%.2f", k/1024.0}')

    echo "${name},${p},${PPN},${NODES_USED},${na},${total},${TIMESTEPS},${WORK},${trial},${n_procs},${rss_max_mb},${rss_sum_mb},${vsz_max_mb},${vsz_sum_mb},${status}" >> "$CSV"

    # Keep rundir on non-ok so you can inspect; otherwise clean up.
    if [[ "$status" == "ok" ]]; then
        rm -rf "$rundir"
    else
        echo "    kept $rundir for inspection (status=$status)"
    fi
}

cleanup_all() {
    rm -f "$SAMPLER_SCRIPT"
    # Also clean any stragglers from prior interrupted runs in this dir.
    rm -f "$SCRIPT_DIR"/.sampler_*.sh 2>/dev/null || true
}
trap cleanup_all EXIT

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
