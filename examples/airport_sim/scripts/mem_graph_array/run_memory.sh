#!/usr/bin/env bash
#PBS -N airport_memory
#PBS -l select=8:ncpus=64:mpiprocs=32
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o airport_memory.log
#
# ---- set your queue/project before submitting -----------------------------
##PBS -q <queue>
##PBS -A <project>
#
# run_memory.sh -- 8-node memory-footprint comparison of the two airport_sim
#                  backends (dstm_darray "simple_dstm" vs dstm_dgraph
#                  "graph_dstm").
#
# This is the memory twin of scripts/graph_v_array/run_scaling.sh: identical
# ladders, identical sizing rules, identical launch recipe, identical header
# patching -- but instead of timing the run we sample peak resident memory.
#
# MEASUREMENT
#   No binary modification. One lightweight sampler process per node polls
#   /proc/<pid>/status for every process whose comm matches ${BIN_COMM} and
#   tracks each PID's high-water VmHWM (peak RSS) and VmPeak (peak virtual).
#   After the workload exits, the per-node files are aggregated into:
#     rss_max_mb  : worst single rank         (what a node must provision)
#     rss_sum_mb  : sum over all ranks        (total job footprint)
#     rss_mean_mb : rss_sum_mb / n_procs_seen
#   VmPeak is carried along as a sanity check; it includes the MPI/RDMA
#   mappings and is not a meaningful "how much did DSTM allocate" number.
#
#   Because the sampler is a poller, the workload must live long enough to be
#   seen. Airport runs at small na finish in ~20 ms, so TIMESTEPS defaults far
#   higher here than in run_scaling.sh, and any run where n_procs_seen < ranks
#   is automatically retried once with ${RETRY_MULT}x the timesteps. Steady
#   state footprint is allocated up front, so more timesteps do not inflate it.
#
# FAIRNESS
#   Footprint is dominated by gate capacity, so the same coupling used in
#   run_scaling.sh is enforced here and recorded in every CSV row:
#       EDGE_GATE_CAPACITY = CAP_PER_EDGE            (graph, per edge)
#       GATE_CAPACITY      = CAP_PER_EDGE * na       (darray, per neighbor rank)
#   na changes per rung in strong mode, so both headers are rewritten and both
#   binaries rebuilt before each rung. Originals are restored on exit.
#
# LAUNCH RECIPE (matches ROSS/weak_scale.sh, which is known to work here)
#   mpirun -n P --hostfile $NODEFILE --prefix $MPI_PREFIX \
#          --map-by ppr:PPN:node -x LD_LIBRARY_PATH -x PATH ./airports ...
#
# USAGE
#   qsub run_memory.sh                          # PBS Pro (directives above)
#   qsub -n 8 -t 180 ./run_memory.sh            # Cobalt
#   ./run_memory.sh --dry-run
#   ./run_memory.sh --mode weak --reps 1
#   ./run_memory.sh --cap-per-edge 8 --sample-ms 10
#
# OUTPUT
#   results/<stamp>/raw/*.log     full stdout+stderr per run
#   results/<stamp>/mem.csv       one row per run
#   results/<stamp>/summary.txt   median-of-reps tables + head-to-head
#
set -uo pipefail

# ==========================================================================
# Paths
# ==========================================================================
if   [[ -n "${PBS_O_WORKDIR:-}" ]]; then
        cd "${PBS_O_WORKDIR}" || exit 1
        SCRIPT_DIR="${PBS_O_WORKDIR}"
elif [[ -n "${COBALT_STARTDIR:-}" ]]; then
        cd "${COBALT_STARTDIR}" || exit 1
        SCRIPT_DIR="${COBALT_STARTDIR}"
else
        SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"          # airport_sim/

DARRAY_EXE="${DARRAY_EXE:-${ROOT}/simple_dstm/airports}"
DGRAPH_EXE="${DGRAPH_EXE:-${ROOT}/graph_dstm/airports}"

# ==========================================================================
# Defaults
# ==========================================================================
MPIRUN="${MPIRUN:-mpirun}"
MPI_FLAVOR="${MPI_FLAVOR:-auto}"        # auto | openmpi | mpich
MPI_EXTRA="${MPI_EXTRA:-}"
MPI_PREFIX="${MPI_PREFIX:-}"
MPI_ENV_VARS="${MPI_ENV_VARS:-LD_LIBRARY_PATH PATH}"
MAX_PPN="${MAX_PPN:-32}"
MAX_NODES="${MAX_NODES:-8}"
PLACEMENT="${PLACEMENT:-pack}"          # pack | spread

WEAK_LADDER="${WEAK_LADDER:-4:1:4 16:1:16 64:2:32 256:8:32}"
STRONG_LADDER="${STRONG_LADDER:-4:1:4 16:1:16 32:1:32 64:2:32 128:4:32 256:8:32}"

NA_PER_RANK="${NA_PER_RANK:-64}"
TOTAL_AIRPORTS="${TOTAL_AIRPORTS:-4096}"
PLANES="${PLANES:-4}"
TIMESTEPS="${TIMESTEPS:-4096}"          # long enough for the poller to see it
SEED="${SEED:-42}"
REPS="${REPS:-2}"                       # footprint is near-deterministic
TIMEOUT_S="${TIMEOUT_S:-900}"
MODES="weak strong"
IMPLS="darray graph"
VARIANTS="base"
VERBOSITY="--quiet"
DRY_RUN=0
OUTROOT="${OUTROOT:-${SCRIPT_DIR}/results}"

# ---- sampler knobs -------------------------------------------------------
SAMPLE_MS="${SAMPLE_MS:-20}"            # poll interval, milliseconds
BIN_COMM="${BIN_COMM:-airports}"        # /proc/<pid>/status Name: to match
SAMPLER_IDLE_MS="${SAMPLER_IDLE_MS:-1500}"  # quiet time before sampler self-exits
SAMPLER_GRACE="${SAMPLER_GRACE:-120}"       # s to wait for sampler mpirun to return
SAMPLER_WARMUP="${SAMPLER_WARMUP:-0.5}" # seconds before the workload starts
RETRY_MULT="${RETRY_MULT:-8}"           # timestep multiplier on a missed run
KEEP_RUNDIR="${KEEP_RUNDIR:-0}"         # 1 = never delete per-run scratch

# ==========================================================================
# Gate capacity (fairness) -- see header comment
# ==========================================================================
CAP_PER_EDGE="${CAP_PER_EDGE:-8}"
DARRAY_HDR="${DARRAY_HDR:-${ROOT}/simple_dstm/airport.h}"
DGRAPH_HDR="${DGRAPH_HDR:-${ROOT}/graph_dstm/airport.h}"
DARRAY_DIR="$(dirname "${DARRAY_EXE}")"
DGRAPH_DIR="$(dirname "${DGRAPH_EXE}")"
MAKE_CMD="${MAKE_CMD:-make}"
MAKE_FLAGS="${MAKE_FLAGS:--B}"
MAKE_TARGET="${MAKE_TARGET:-}"

usage() {
        sed -n '7,58p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        cat <<EOF

options:
  --mode {weak|strong|both}    default: both
  --impl {darray|graph|both}   default: both
  --variants "a b c"           graph only: base congestion novstate
  --ladder "np:nodes:ppn ..."  override both ladders
  --weak-ladder   "..."        override weak ladder only
  --strong-ladder "..."        override strong ladder only
  --placement {pack|spread}    pack = fill nodes to ppn; spread = even fan-out
  --max-ppn N                  ranks per node cap        (default ${MAX_PPN})
  --max-nodes N                node cap                  (default ${MAX_NODES})
  --na-per-rank N              weak airports/rank        (default ${NA_PER_RANK})
  --total N                    strong total airports     (default ${TOTAL_AIRPORTS})
  --planes P / --time T / --seed S / --reps R
  --timeout S                  per-run seconds, 0 = off  (default ${TIMEOUT_S})
  --sample-ms N                sampler poll interval     (default ${SAMPLE_MS})
  --comm NAME                  /proc Name: to match      (default ${BIN_COMM})
  --retry-mult N               timestep x N on a missed run (default ${RETRY_MULT})
  --keep-rundir                keep per-run sampler scratch dirs
  --mpirun CMD                 launcher                  (default ${MPIRUN})
  --mpi-flavor {auto|openmpi|mpich}
  --mpi-prefix DIR             Open MPI install root for --prefix
  --mpi-env "A B C"            vars to forward with -x   (default: ${MPI_ENV_VARS})
  --mpi-extra "ARGS"           extra launcher args
  --cap-per-edge N             sets EDGE_GATE_CAPACITY=N and GATE_CAPACITY=N*na,
                               rebuilding both before each rung
                                                         (default ${CAP_PER_EDGE})
  --make CMD / --make-flags "ARGS" / --make-target NAME
  --info | --loud              raise DSTM log level
  --out DIR / --dry-run / -h
EOF
}

while [[ $# -gt 0 ]]; do
        case "$1" in
        --mode)          MODES="$2"; [[ "$2" == both ]] && MODES="weak strong"; shift 2 ;;
        --impl)          IMPLS="$2"; [[ "$2" == both ]] && IMPLS="darray graph"; shift 2 ;;
        --variants)      VARIANTS="$2"; shift 2 ;;
        --ladder)        WEAK_LADDER="$2"; STRONG_LADDER="$2"; shift 2 ;;
        --weak-ladder)   WEAK_LADDER="$2"; shift 2 ;;
        --strong-ladder) STRONG_LADDER="$2"; shift 2 ;;
        --placement)     PLACEMENT="$2"; shift 2 ;;
        --max-ppn)       MAX_PPN="$2"; shift 2 ;;
        --max-nodes)     MAX_NODES="$2"; shift 2 ;;
        --na-per-rank)   NA_PER_RANK="$2"; shift 2 ;;
        --total)         TOTAL_AIRPORTS="$2"; shift 2 ;;
        --planes)        PLANES="$2"; shift 2 ;;
        --time)          TIMESTEPS="$2"; shift 2 ;;
        --seed)          SEED="$2"; shift 2 ;;
        --reps)          REPS="$2"; shift 2 ;;
        --timeout)       TIMEOUT_S="$2"; shift 2 ;;
        --sample-ms)     SAMPLE_MS="$2"; shift 2 ;;
        --comm)          BIN_COMM="$2"; shift 2 ;;
        --retry-mult)    RETRY_MULT="$2"; shift 2 ;;
        --keep-rundir)   KEEP_RUNDIR=1; shift ;;
        --mpirun)        MPIRUN="$2"; shift 2 ;;
        --mpi-flavor)    MPI_FLAVOR="$2"; shift 2 ;;
        --mpi-prefix)    MPI_PREFIX="$2"; shift 2 ;;
        --mpi-env)       MPI_ENV_VARS="$2"; shift 2 ;;
        --mpi-extra)     MPI_EXTRA="$2"; shift 2 ;;
        --cap-per-edge)  CAP_PER_EDGE="$2"; shift 2 ;;
        --make)          MAKE_CMD="$2"; shift 2 ;;
        --make-flags)    MAKE_FLAGS="$2"; shift 2 ;;
        --make-target)   MAKE_TARGET="$2"; shift 2 ;;
        --info)          VERBOSITY="--info"; shift ;;
        --loud)          VERBOSITY=""; shift ;;
        --out)           OUTROOT="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
        esac
done

# ==========================================================================
# Allocation discovery
# ==========================================================================
ALLOC_NODEFILE=""
ALLOC_SRC="none"

if   [[ -n "${COBALT_NODEFILE:-}" && -r "${COBALT_NODEFILE}" ]]; then
        ALLOC_NODEFILE="${COBALT_NODEFILE}"; ALLOC_SRC="COBALT_NODEFILE"
elif [[ -n "${PBS_NODEFILE:-}"    && -r "${PBS_NODEFILE}"    ]]; then
        ALLOC_NODEFILE="${PBS_NODEFILE}";    ALLOC_SRC="PBS_NODEFILE"
elif [[ -n "${HOSTFILE:-}"        && -r "${HOSTFILE}"        ]]; then
        ALLOC_NODEFILE="${HOSTFILE}";        ALLOC_SRC="HOSTFILE"
elif [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
        ALLOC_NODEFILE="$(mktemp)"
        scontrol show hostnames "${SLURM_JOB_NODELIST}" > "${ALLOC_NODEFILE}"
        ALLOC_SRC="SLURM"
fi

NODELIST=()
if [[ -n "${ALLOC_NODEFILE}" ]]; then
        mapfile -t NODELIST < <(sort -u "${ALLOC_NODEFILE}")
else
        NODELIST=("$(hostname)")
        ALLOC_SRC="localhost"
fi

AVAIL_NODES=${#NODELIST[@]}
(( AVAIL_NODES > MAX_NODES )) && AVAIL_NODES=${MAX_NODES}

if [[ "${ALLOC_SRC}" == "localhost" && ${DRY_RUN} -eq 0 ]]; then
        echo "WARNING: no batch allocation detected; running on $(hostname) only." >&2
        echo "         Multi-node rungs will be skipped." >&2
fi

# ==========================================================================
# MPI flavor + prefix
# ==========================================================================
if [[ "${MPI_FLAVOR}" == auto ]]; then
        if "${MPIRUN}" --version 2>&1 | grep -qi "open mpi\|open-rte"; then
                MPI_FLAVOR=openmpi
        elif "${MPIRUN}" --version 2>&1 | grep -qi "hydra\|mpich\|intel(r) mpi"; then
                MPI_FLAVOR=mpich
        else
                MPI_FLAVOR=openmpi
        fi
fi

if [[ -z "${MPI_PREFIX}" ]]; then
        _mpirun_path="$(command -v "${MPIRUN}" 2>/dev/null)"
        if [[ -n "${_mpirun_path}" ]]; then
                MPI_PREFIX="$(dirname "$(dirname "${_mpirun_path}")")"
        fi
fi

MPI_X_FLAGS=()
if [[ "${MPI_FLAVOR}" == openmpi ]]; then
        for _v in ${MPI_ENV_VARS}; do
                MPI_X_FLAGS+=(-x "${_v}")
        done
fi

# ==========================================================================
# Helpers
# ==========================================================================
isqrt_exact() { awk -v n="$1" 'BEGIN { r = int(sqrt(n) + 0.5); print (r * r == n) ? r : 0 }'; }

median() {
        sort -g | awk '
                { a[NR] = $1 }
                END {
                        if (NR == 0) { print "nan"; exit }
                        if (NR % 2)  { printf "%.4f\n", a[(NR + 1) / 2] }
                        else         { printf "%.4f\n", (a[NR/2] + a[NR/2 + 1]) / 2 }
                }'
}

exe_for() { case "$1" in darray) echo "${DARRAY_EXE}";; graph) echo "${DGRAPH_EXE}";; esac; }

variant_flags() {
        [[ "$1" != graph ]] && { echo ""; return; }
        case "$2" in
        congestion) echo "--congestion" ;;
        novstate)   echo "--no-vstate" ;;
        *)          echo "" ;;
        esac
}

hostfile_for() {
        local nodes="$1" hf i
        if [[ -n "${ALLOC_NODEFILE}" && "${nodes}" -eq "${#NODELIST[@]}" ]]; then
                echo "${ALLOC_NODEFILE}"
                return
        fi
        hf="${HFDIR}/hosts-n${nodes}.txt"
        if [[ ! -f "${hf}" ]]; then
                : > "${hf}"
                for (( i = 0; i < nodes; i++ )); do
                        echo "${NODELIST[$i]}" >> "${hf}"
                done
        fi
        echo "${hf}"
}

# --------------------------------------------------------------------------
# Header patching + rebuild
# --------------------------------------------------------------------------
HDR_BACKUPS=()
SAMPLER_SCRIPT=""

backup_headers() {
        local h
        for h in "${DARRAY_HDR}" "${DGRAPH_HDR}"; do
                [[ -f "${h}" ]] || continue
                cp -p "${h}" "${h}.memory.bak" || {
                        echo "ERROR: cannot back up ${h}" >&2
                        exit 1
                }
                HDR_BACKUPS+=("${h}")
        done
}

cleanup_all() {
        local h
        for h in ${HDR_BACKUPS[@]+"${HDR_BACKUPS[@]}"}; do
                [[ -f "${h}.memory.bak" ]] || continue
                mv -f "${h}.memory.bak" "${h}"
        done
        [[ -n "${SAMPLER_SCRIPT}" && -f "${SAMPLER_SCRIPT}" ]] && rm -f "${SAMPLER_SCRIPT}"
        return 0
}
trap cleanup_all EXIT INT TERM

patch_define() {
        local file="$1" name="$2" value="$3"
        grep -Eq "^[[:space:]]*#[[:space:]]*define[[:space:]]+${name}[[:space:]]+" "${file}" || {
                echo "ERROR: no '#define ${name}' found in ${file}" >&2
                return 1
        }
        sed -i -E "s|^([[:space:]]*#[[:space:]]*define[[:space:]]+${name}[[:space:]]+).*$|\1${value}|" "${file}" || return 1
        grep -Eq "^[[:space:]]*#[[:space:]]*define[[:space:]]+${name}[[:space:]]+${value}[[:space:]]*$" "${file}" || {
                echo "ERROR: ${name} did not take in ${file}" >&2
                return 1
        }
        return 0
}

CUR_CAP_NA=""
GATE_CAP=0
EDGE_CAP=0
set_capacity() {
        local na="$1" impl dir rc

        EDGE_CAP=${CAP_PER_EDGE}
        GATE_CAP=$(( CAP_PER_EDGE * na ))

        [[ "${CUR_CAP_NA}" == "${na}" ]] && return 0

        echo "  capacity: EDGE_GATE_CAPACITY=${EDGE_CAP}  GATE_CAPACITY=${GATE_CAP}  (na=${na})"

        if [[ ${DRY_RUN} -eq 1 ]]; then
                echo "      would patch ${DGRAPH_HDR} and ${DARRAY_HDR}, then rebuild"
                CUR_CAP_NA="${na}"
                return 0
        fi

        for impl in ${IMPLS}; do
                case "${impl}" in
                darray) patch_define "${DARRAY_HDR}" "GATE_CAPACITY" "${GATE_CAP}" || return 1 ;;
                graph)  patch_define "${DGRAPH_HDR}" "EDGE_GATE_CAPACITY" "${EDGE_CAP}" || return 1 ;;
                esac
        done

        for impl in ${IMPLS}; do
                case "${impl}" in
                darray) dir="${DARRAY_DIR}" ;;
                graph)  dir="${DGRAPH_DIR}" ;;
                *)      continue ;;
                esac
                # shellcheck disable=SC2086
                ${MAKE_CMD} -C "${dir}" ${MAKE_FLAGS} ${MAKE_TARGET} \
                        > "${OUTDIR}/build-${impl}-na${na}.log" 2>&1
                rc=$?
                if [[ ${rc} -ne 0 ]]; then
                        echo "  BUILD FAILED (${impl}, na=${na}) -> build-${impl}-na${na}.log" >&2
                        return 1
                fi
        done

        CUR_CAP_NA="${na}"
        return 0
}

# ==========================================================================
# Preflight
# ==========================================================================
missing=0
for impl in ${IMPLS}; do
        exe="$(exe_for "${impl}")"
        [[ -x "${exe}" ]] || { echo "ERROR: ${impl} exe missing or not executable: ${exe}" >&2; missing=1; }
done
[[ ${missing} -eq 1 && ${DRY_RUN} -eq 0 ]] && exit 1

command -v "${MPIRUN}" >/dev/null 2>&1 || {
        echo "ERROR: launcher '${MPIRUN}' not on PATH" >&2
        [[ ${DRY_RUN} -eq 0 ]] && exit 1
}

if [[ "${MPI_FLAVOR}" == openmpi && ! -x "${MPI_PREFIX}/bin/orted" && ${DRY_RUN} -eq 0 ]]; then
        echo "WARNING: ${MPI_PREFIX}/bin/orted not found." >&2
        echo "         --prefix may be wrong; override with --mpi-prefix DIR." >&2
fi

for _h in "${DARRAY_HDR}" "${DGRAPH_HDR}"; do
        if [[ ! -w "${_h}" && ${DRY_RUN} -eq 0 ]]; then
                echo "ERROR: header not writable: ${_h}" >&2
                exit 1
        fi
done
[[ ${DRY_RUN} -eq 0 ]] && backup_headers

TIMEOUT_CMD=()
if [[ "${TIMEOUT_S}" != "0" ]] && command -v timeout >/dev/null 2>&1; then
        TIMEOUT_CMD=(timeout --signal=TERM --kill-after=30 "${TIMEOUT_S}")
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUTDIR="${OUTROOT}/${STAMP}"
RAWDIR="${OUTDIR}/raw"
HFDIR="${OUTDIR}/hostfiles"
SCRATCH="${OUTDIR}/scratch"
CSV="${OUTDIR}/mem.csv"
SUMMARY="${OUTDIR}/summary.txt"
mkdir -p "${RAWDIR}" "${HFDIR}" "${SCRATCH}"

{
        echo "host:         $(hostname)"
        echo "date:         $(date -Is)"
        echo "job id:       ${COBALT_JOBID:-${PBS_JOBID:-${SLURM_JOB_ID:-interactive}}}"
        echo "alloc source: ${ALLOC_SRC}  (${#NODELIST[@]} nodes, using <= ${AVAIL_NODES})"
        echo "nodefile:     ${ALLOC_NODEFILE:-<none>}"
        echo "nodes:        ${NODELIST[*]:0:${AVAIL_NODES}}"
        echo "launcher:     ${MPIRUN} [${MPI_FLAVOR}] ${MPI_EXTRA}"
        echo "mpi prefix:   ${MPI_PREFIX:-<unset>}"
        echo "forwarded:    ${MPI_ENV_VARS}"
        echo "placement:    ${PLACEMENT}   max ppn ${MAX_PPN}"
        echo "darray exe:   ${DARRAY_EXE}"
        echo "dgraph exe:   ${DGRAPH_EXE}"
        echo "modes:        ${MODES}"
        echo "impls:        ${IMPLS}    variants: ${VARIANTS}"
        echo "weak ladder:  ${WEAK_LADDER}"
        echo "strong ladr:  ${STRONG_LADDER}"
        echo "na/rank:      ${NA_PER_RANK} (weak)   total: ${TOTAL_AIRPORTS} (strong)"
        echo "planes:       ${PLANES}    timesteps: ${TIMESTEPS}    reps: ${REPS}"
        echo "cap/edge:     ${CAP_PER_EDGE}  (GATE_CAPACITY = cap * na)"
        echo "sampler:      ${SAMPLE_MS} ms, comm '${BIN_COMM}', warmup ${SAMPLER_WARMUP}s"
        echo "retry mult:   ${RETRY_MULT}x timesteps if ranks are missed"
        echo "timeout:      ${TIMEOUT_S}s"
} | tee "${OUTDIR}/config.txt"
echo

echo "impl,variant,mode,ranks,nodes,ppn,na,total,grid,planes,timesteps,cap_per_edge,gate_capacity,edge_gate_capacity,rep,status,n_procs_seen,rss_max_mb,rss_sum_mb,rss_mean_mb,vsz_max_mb,vsz_sum_mb,elapsed,tsps" > "${CSV}"

# ==========================================================================
# Per-node sampler
#   Written once to a shared-FS path so every node can exec it.
#   Output: <rundir>/mem_<host>.txt, one line per PID:
#       <pid> <peak_vmhwm_kb> <peak_vmpeak_kb>
# ==========================================================================
SAMPLER_SCRIPT="${SCRATCH}/.sampler_$$.sh"

cat > "${SAMPLER_SCRIPT}" <<'SAMPLER'
#!/usr/bin/env bash
# args: <outdir> <sample_ms> <bin_comm> <max_seconds> <idle_ms>
#
# Polls /proc for processes whose comm == <bin_comm>, tracking each PID's
# high-water VmHWM / VmPeak. Self-terminating: once at least one target has
# been seen, the sampler exits after <idle_ms> with zero targets alive. A
# hard <max_seconds> deadline guarantees it never outlives its mpirun.
set -u
OUTDIR="$1"
SAMPLE_MS="$2"
BIN_COMM="$3"
MAX_S="$4"
IDLE_MS="$5"

# The kernel truncates comm to 15 chars; compare against the same prefix.
WANT="${BIN_COMM:0:15}"

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
    local pid
    for pid in "${!HWM[@]}"; do flush_pid "$pid"; done
    exit 0
}
trap cleanup TERM INT

SLEEP_S=$(awk -v ms="$SAMPLE_MS" 'BEGIN { printf "%.4f", ms / 1000.0 }')
IDLE_LIMIT=$(awk -v i="$IDLE_MS" -v s="$SAMPLE_MS" \
                 'BEGIN { n = int(i / s); print (n < 3) ? 3 : n }')
DEADLINE=$(( SECONDS + MAX_S ))

saw_any=0
idle=0

while :; do
    (( SECONDS >= DEADLINE )) && break

    declare -A SEEN=()
    live=0

    for d in /proc/[0-9]*; do
        [[ -r "$d/status" ]] || continue

        # First line is "Name:\t<comm>". Exact match only -- a glob here
        # would let an empty/racing name match every process on the node.
        read -r _ name < "$d/status" 2>/dev/null || continue
        [[ -n "$name" ]]        || continue
        [[ "$name" == "$WANT" ]] || continue

        pid="${d#/proc/}"
        SEEN[$pid]=1
        live=$(( live + 1 ))

        hwm=0; vpk=0
        # VmPeak precedes VmHWM in /proc/<pid>/status; stop once both seen.
        while IFS= read -r line; do
            case "$line" in
                VmPeak:*) vpk=${line//[^0-9]/} ;;
                VmHWM:*)  hwm=${line//[^0-9]/}; break ;;
            esac
        done < "$d/status" 2>/dev/null

        (( hwm == 0 )) && continue          # exited mid-read; ignore sample

        if [[ -z "${HWM[$pid]:-}" ]] || (( hwm > HWM[$pid] )); then
            HWM[$pid]=$hwm
        fi
        if [[ -z "${VPK[$pid]:-}" ]] || (( vpk > VPK[$pid] )); then
            VPK[$pid]=$vpk
        fi
    done

    # A tracked PID that vanished has exited: flush its high-water marks.
    for pid in "${!HWM[@]}"; do
        [[ -z "${SEEN[$pid]:-}" ]] && flush_pid "$pid"
    done

    if (( live > 0 )); then
        saw_any=1
        idle=0
    elif (( saw_any )); then
        idle=$(( idle + 1 ))
        (( idle >= IDLE_LIMIT )) && break
    fi

    sleep "$SLEEP_S"
done

cleanup
SAMPLER
chmod +x "${SAMPLER_SCRIPT}"

# ==========================================================================
# Launch helpers
# ==========================================================================
build_launch() {   # build_launch <np> <nodes> <ppn> <hostfile> -> LAUNCH[]
        local np="$1" nodes="$2" ppn="$3" hf="$4"
        LAUNCH=()
        if [[ "${MPI_FLAVOR}" == openmpi ]]; then
                LAUNCH=("${MPIRUN}" -n "${np}" --hostfile "${hf}")
                [[ -n "${MPI_PREFIX}" ]] && LAUNCH+=(--prefix "${MPI_PREFIX}")
                if (( np > 1 )) && (( np % nodes == 0 )); then
                        LAUNCH+=(--map-by "ppr:${ppn}:node")
                fi
                LAUNCH+=("${MPI_X_FLAGS[@]}")
        else
                LAUNCH=("${MPIRUN}" -n "${np}" -f "${hf}" -ppn "${ppn}")
        fi
}

start_samplers() { # start_samplers <nodes> <hostfile> <rundir> -> SAMPLER_PID
        local nodes="$1" hf="$2" rundir="$3"
        local launch=()
        if [[ "${MPI_FLAVOR}" == openmpi ]]; then
                launch=("${MPIRUN}" -n "${nodes}" --hostfile "${hf}")
                [[ -n "${MPI_PREFIX}" ]] && launch+=(--prefix "${MPI_PREFIX}")
                launch+=(--map-by "ppr:1:node" "${MPI_X_FLAGS[@]}")
        else
                launch=("${MPIRUN}" -n "${nodes}" -f "${hf}" -ppn 1)
        fi
        local max_s=$(( ${TIMEOUT_S:-900} + 120 ))
        (( max_s < 180 )) && max_s=180
        "${launch[@]}" "${SAMPLER_SCRIPT}" "${rundir}" "${SAMPLE_MS}" "${BIN_COMM}" \
                       "${max_s}" "${SAMPLER_IDLE_MS}" \
                > "${rundir}/sampler.log" 2>&1 &
        SAMPLER_PID=$!
}

# Wait for the self-terminating samplers to drain. No second launcher
# invocation: spawning a third mpirun at 8x32 to pkill remote samplers is
# what segfaulted the launcher. If the sampler job overstays its welcome,
# TERM the local mpirun -- orted propagates it and the remote traps flush.
wait_samplers() {  # wait_samplers <rundir> <sampler_pid>
        local rundir="$1" spid="$2" waited=0

        while kill -0 "${spid}" 2>/dev/null && (( waited < SAMPLER_GRACE )); do
                sleep 1
                waited=$(( waited + 1 ))
        done

        if kill -0 "${spid}" 2>/dev/null; then
                echo "sampler mpirun alive after ${SAMPLER_GRACE}s; terminating" \
                        >> "${rundir}/sampler.log"
                kill -TERM "${spid}" 2>/dev/null
                sleep 3
                kill -KILL "${spid}" 2>/dev/null
        fi
        wait "${spid}" 2>/dev/null || true
}

# ==========================================================================
# One measured run
#   Sets: R_STATUS R_NPROCS R_RSS_MAX R_RSS_SUM R_RSS_MEAN R_VSZ_MAX R_VSZ_SUM
#         R_ELAPSED R_TSPS
# ==========================================================================
measure_once() {
        local impl="$1" variant="$2" np="$3" nodes="$4" ppn="$5"
        local na="$6" steps="$7" log="$8"
        local exe flags hf rc rundir

        exe="$(exe_for "${impl}")"
        flags="$(variant_flags "${impl}" "${variant}")"
        hf="$(hostfile_for "${nodes}")"

        rundir="$(mktemp -d "${SCRATCH}/run_XXXXXX")"

        build_launch "${np}" "${nodes}" "${ppn}" "${hf}"
        # shellcheck disable=SC2206
        local cmd=("${LAUNCH[@]}" ${MPI_EXTRA} "${exe}"
                   --na "${na}" --time "${steps}"
                   --planes "${PLANES}" --seed "${SEED}" ${flags} ${VERBOSITY})

        if [[ ${DRY_RUN} -eq 1 ]]; then
                printf '\n      sampler: %s -n %s ... %s %s %s %s\n' \
                       "${MPIRUN}" "${nodes}" "${SAMPLER_SCRIPT}" "<rundir>" \
                       "${SAMPLE_MS}" "${BIN_COMM}"
                printf '      run    : %s\n' "${cmd[*]}"
                rmdir "${rundir}" 2>/dev/null
                R_STATUS="dry"; R_NPROCS=0
                R_RSS_MAX=0; R_RSS_SUM=0; R_RSS_MEAN=0; R_VSZ_MAX=0; R_VSZ_SUM=0
                R_ELAPSED=0; R_TSPS=0
                return 0
        fi

        start_samplers "${nodes}" "${hf}" "${rundir}"
        sleep "${SAMPLER_WARMUP}"

        "${TIMEOUT_CMD[@]}" "${cmd[@]}" > "${log}" 2>&1
        rc=$?

        wait_samplers "${rundir}" "${SAMPLER_PID}"

        # ---- aggregate per-node sampler files ----------------------------
        local n=0 rss_max_kb=0 rss_sum_kb=0 vsz_max_kb=0 vsz_sum_kb=0
        shopt -s nullglob
        local f pid hwm vpk
        for f in "${rundir}"/mem_*.txt; do
                while read -r pid hwm vpk; do
                        [[ -z "${pid}" ]] && continue
                        n=$(( n + 1 ))
                        rss_sum_kb=$(( rss_sum_kb + hwm ))
                        vsz_sum_kb=$(( vsz_sum_kb + vpk ))
                        (( hwm > rss_max_kb )) && rss_max_kb=${hwm}
                        (( vpk > vsz_max_kb )) && vsz_max_kb=${vpk}
                done < "${f}"
        done
        shopt -u nullglob

        R_STATUS="ok"
        if   [[ ${rc} -eq 124 || ${rc} -eq 137 ]]; then R_STATUS="timeout"
        elif [[ ${rc} -ne 0 ]];                   then R_STATUS="fail(${rc})"
        fi
        if grep -q "orted: command not found\|ORTE was unable to reliably start" "${log}" 2>/dev/null; then
                R_STATUS="launch_err"
        fi
        # Undercount = poller too slow (retryable). Overcount = comm matching
        # is wrong (not retryable; retrying just burns the allocation).
        if [[ "${R_STATUS}" == ok ]]; then
                if   (( n < np )); then R_STATUS="missed_${n}of${np}"
                elif (( n > np )); then R_STATUS="overcount_${n}of${np}"
                fi
        fi

        R_NPROCS=${n}
        R_RSS_MAX=$(awk -v k="${rss_max_kb}" 'BEGIN { printf "%.2f", k / 1024.0 }')
        R_RSS_SUM=$(awk -v k="${rss_sum_kb}" 'BEGIN { printf "%.2f", k / 1024.0 }')
        R_RSS_MEAN=$(awk -v k="${rss_sum_kb}" -v n="${n}" \
                        'BEGIN { printf "%.2f", (n > 0) ? k / n / 1024.0 : 0 }')
        R_VSZ_MAX=$(awk -v k="${vsz_max_kb}" 'BEGIN { printf "%.2f", k / 1024.0 }')
        R_VSZ_SUM=$(awk -v k="${vsz_sum_kb}" 'BEGIN { printf "%.2f", k / 1024.0 }')

        R_ELAPSED=$(awk '/^Elapsed time:/      { print $3 }' "${log}" | tail -1)
        R_TSPS=$(   awk '/^Timesteps\/second:/ { print $2 }' "${log}" | tail -1)
        : "${R_ELAPSED:=nan}" "${R_TSPS:=nan}"

        if [[ "${R_STATUS}" == ok && ${KEEP_RUNDIR} -eq 0 ]]; then
                rm -rf "${rundir}"
        else
                echo "      kept ${rundir}"
        fi
        return 0
}

run_one() {
        local impl="$1" variant="$2" mode="$3" np="$4" nodes="$5" ppn="$6"
        local na="$7" total="$8" grid="$9" rep="${10}"
        local tag log steps

        steps=${TIMESTEPS}
        tag="${impl}-${variant}-${mode}-np${np}-n${nodes}x${ppn}-na${na}-r${rep}"
        log="${RAWDIR}/${tag}.log"

        printf '  %-6s %-10s np=%-4s %sx%-3s na=%-6s total=%-7s rep=%s ... ' \
               "${impl}" "${variant}" "${np}" "${nodes}" "${ppn}" \
               "${na}" "${total}" "${rep}"

        measure_once "${impl}" "${variant}" "${np}" "${nodes}" "${ppn}" \
                     "${na}" "${steps}" "${log}"

        # The sampler is a poller: a run that finished between two polls shows
        # up as fewer PIDs than ranks. Stretch it and try once more.
        if [[ "${R_STATUS}" == missed_* ]] && (( RETRY_MULT > 1 )); then
                steps=$(( TIMESTEPS * RETRY_MULT ))
                printf '\n      %s -> retry with time=%s ... ' "${R_STATUS}" "${steps}"
                log="${RAWDIR}/${tag}-retry.log"
                measure_once "${impl}" "${variant}" "${np}" "${nodes}" "${ppn}" \
                             "${na}" "${steps}" "${log}"
        fi

        [[ ${DRY_RUN} -eq 1 ]] && return 0

        echo "${impl},${variant},${mode},${np},${nodes},${ppn},${na},${total},${grid},${PLANES},${steps},${CAP_PER_EDGE},${GATE_CAP},${EDGE_CAP},${rep},${R_STATUS},${R_NPROCS},${R_RSS_MAX},${R_RSS_SUM},${R_RSS_MEAN},${R_VSZ_MAX},${R_VSZ_SUM},${R_ELAPSED},${R_TSPS}" >> "${CSV}"

        if [[ "${R_STATUS}" == ok ]]; then
                printf 'rss max %8s MB  sum %10s MB  mean %8s MB\n' \
                       "${R_RSS_MAX}" "${R_RSS_SUM}" "${R_RSS_MEAN}"
        else
                printf '%s (%s)\n' "${R_STATUS}" "$(basename "${log}")"
        fi
}

# ==========================================================================
# Sweep
# ==========================================================================
ladder_for() { case "$1" in weak) echo "${WEAK_LADDER}";; strong) echo "${STRONG_LADDER}";; esac; }

for mode in ${MODES}; do
        echo "===== ${mode} scaling (memory) ====="
        for rung in $(ladder_for "${mode}"); do
                np="${rung%%:*}"
                rest="${rung#*:}"
                nodes="${rest%%:*}"
                ppn="${rest##*:}"

                if [[ "${PLACEMENT}" == spread ]]; then
                        nodes=${AVAIL_NODES}
                        (( nodes > np )) && nodes=${np}
                        ppn=$(( (np + nodes - 1) / nodes ))
                fi

                if (( ppn > MAX_PPN )); then
                        echo "  skip np=${np}: ppn ${ppn} exceeds cap ${MAX_PPN}"; continue
                fi
                if (( nodes > AVAIL_NODES )); then
                        echo "  skip np=${np}: needs ${nodes} nodes, have ${AVAIL_NODES}"; continue
                fi
                if (( nodes * ppn < np )); then
                        echo "  skip np=${np}: ${nodes}x${ppn} cannot host ${np} ranks"; continue
                fi

                if [[ "${mode}" == weak ]]; then
                        na=${NA_PER_RANK}
                        total=$(( na * np ))
                else
                        if (( TOTAL_AIRPORTS % np != 0 )); then
                                echo "  skip np=${np}: ${TOTAL_AIRPORTS} not divisible by ranks"; continue
                        fi
                        na=$(( TOTAL_AIRPORTS / np ))
                        total=${TOTAL_AIRPORTS}
                fi

                grid=$(isqrt_exact "${total}")
                if [[ "${grid}" == "0" ]]; then
                        echo "  skip np=${np}: total ${total} is not a perfect square"; continue
                fi
                if (( grid < 2 )); then
                        echo "  skip np=${np}: grid ${grid} < 2 (dgraph main rejects it)"; continue
                fi

                if ! set_capacity "${na}"; then
                        echo "  skip np=${np}: capacity build failed for na=${na}"; continue
                fi

                for impl in ${IMPLS}; do
                        vlist="${VARIANTS}"
                        [[ "${impl}" == darray ]] && vlist="base"
                        for variant in ${vlist}; do
                                for (( rep = 1; rep <= REPS; rep++ )); do
                                        run_one "${impl}" "${variant}" "${mode}" \
                                                "${np}" "${nodes}" "${ppn}" \
                                                "${na}" "${total}" "${grid}" "${rep}"
                                done
                        done
                done
        done
        echo
done

[[ ${DRY_RUN} -eq 1 ]] && { echo "dry run complete"; exit 0; }

# ==========================================================================
# Summary
#   CSV columns:
#     1 impl  2 variant  3 mode  4 ranks  5 nodes  6 ppn  7 na  8 total
#     9 grid 10 planes 11 timesteps 12 cap_per_edge 13 gate_capacity
#    14 edge_gate_capacity 15 rep 16 status 17 n_procs_seen
#    18 rss_max_mb 19 rss_sum_mb 20 rss_mean_mb 21 vsz_max_mb 22 vsz_sum_mb
#    23 elapsed 24 tsps
# ==========================================================================
np_list_for() {
        awk -F, -v m="$1" 'NR > 1 && $3 == m { print $4 }' "${CSV}" | sort -n -u
}

med_col() {   # med_col <mode> <impl> <variant> <np> <col>
        awk -F, -v m="$1" -v i="$2" -v v="$3" -v n="$4" -v c="$5" \
            '$3==m && $1==i && $2==v && $4==n && $16=="ok" { print $c }' \
            "${CSV}" | median
}

summarize() {
        local mode="$1" keys key impl variant np
        local rmax rsum rmean vmax na tot nodes ppn gcap ecap

        keys=$(awk -F, -v m="${mode}" 'NR > 1 && $3 == m { print $1 "|" $2 }' "${CSV}" | sort -u)
        [[ -z "${keys}" ]] && return

        echo "===== ${mode} scaling: peak memory (median of ${REPS} reps) ====="
        printf '%-8s %-10s %6s %8s %7s %8s %8s %8s %11s %11s %11s %12s\n' \
               impl variant ranks layout na total gate edge \
               "rss/rank" "rss max" "rss total" "B/airport"
        printf -- '--------------------------------------------------------------------------------------------------------------------------\n'

        for key in ${keys}; do
                impl="${key%%|*}"; variant="${key##*|}"
                for np in $(np_list_for "${mode}"); do
                        rmean=$(med_col "${mode}" "${impl}" "${variant}" "${np}" 20)
                        [[ "${rmean}" == "nan" ]] && continue
                        rmax=$( med_col "${mode}" "${impl}" "${variant}" "${np}" 18)
                        rsum=$( med_col "${mode}" "${impl}" "${variant}" "${np}" 19)
                        vmax=$( med_col "${mode}" "${impl}" "${variant}" "${np}" 21)

                        read -r nodes ppn na tot gcap ecap < <(awk -F, -v m="${mode}" \
                                -v i="${impl}" -v v="${variant}" -v n="${np}" \
                                '$3==m && $1==i && $2==v && $4==n { print $5, $6, $7, $8, $13, $14; exit }' "${CSV}")

                        awk -v i="${impl}" -v v="${variant}" -v n="${np}" \
                            -v lay="${nodes}x${ppn}" -v na="${na}" -v tot="${tot}" \
                            -v g="${gcap}" -v e="${ecap}" \
                            -v rmean="${rmean}" -v rmax="${rmax}" -v rsum="${rsum}" '
                        BEGIN {
                                bpa = (tot > 0) ? rsum * 1048576.0 / tot : 0;
                                printf "%-8s %-10s %6s %8s %7s %8s %8s %8s %8.1f MB %8.1f MB %8.1f MB %10.0f B\n",
                                       i, v, n, lay, na, tot, g, e, rmean, rmax, rsum, bpa;
                        }'
                done
                echo
        done

        if [[ "${IMPLS}" == *darray* && "${IMPLS}" == *graph* ]]; then
                echo "  head to head (peak RSS per rank; > 1.00 means dgraph uses more)"
                printf '  %6s %8s %12s %12s %10s %14s\n' \
                       ranks layout darray dgraph ratio "delta/rank"
                for np in $(np_list_for "${mode}"); do
                        local d g
                        d=$(med_col "${mode}" darray base "${np}" 20)
                        g=$(med_col "${mode}" graph  base "${np}" 20)
                        [[ "${d}" == "nan" || "${g}" == "nan" ]] && continue
                        lay=$(awk -F, -v m="${mode}" -v n="${np}" \
                                '$3==m && $4==n { print $5 "x" $6; exit }' "${CSV}")
                        awk -v n="${np}" -v l="${lay}" -v d="${d}" -v g="${g}" '
                        BEGIN {
                                printf "  %6s %8s %9.1f MB %9.1f MB %9.2fx %11.1f MB\n",
                                       n, l, d, g, (d > 0 ? g / d : 0), g - d;
                        }'
                done
                echo
        fi
}

{
        cat "${OUTDIR}/config.txt"
        echo
        for mode in ${MODES}; do summarize "${mode}"; done

        total_runs=$(( $(wc -l < "${CSV}") - 1 ))
        bad=$(awk -F, 'NR > 1 && $16 != "ok"' "${CSV}" | wc -l)
        tmo=$(awk -F, 'NR > 1 && $16 == "timeout"' "${CSV}" | wc -l)
        lerr=$(awk -F, 'NR > 1 && $16 == "launch_err"' "${CSV}" | wc -l)
        mis=$(awk -F, 'NR > 1 && $16 ~ /^missed_/' "${CSV}" | wc -l)

        echo "runs recorded:      ${total_runs}"
        echo "failed:             ${bad}  (timeouts: ${tmo}, launcher: ${lerr}, undersampled: ${mis})"
        if (( lerr > 0 )); then
                echo
                echo "NOTE: ${lerr} run(s) failed in the MPI launcher, not the application."
                echo "      Check --mpi-prefix (currently '${MPI_PREFIX}') and --mpi-env."
        fi
        if (( mis > 0 )); then
                echo
                echo "NOTE: ${mis} run(s) were shorter than the sampler could resolve even"
                echo "      after retry. Raise --time or lower --sample-ms and rerun those"
                echo "      points; their rss_* values are a lower bound only."
        fi
        if (( bad > 0 )); then
                echo
                echo "non-ok runs:"
                awk -F, 'NR > 1 && $16 != "ok" { printf "  %-6s %-10s %-6s np=%-4s %sx%-3s na=%-6s -> %s\n", $1,$2,$3,$4,$5,$6,$7,$16 }' "${CSV}"
        fi
} | tee "${SUMMARY}"

echo
echo "csv:     ${CSV}"
echo "summary: ${SUMMARY}"
echo "logs:    ${RAWDIR}"
