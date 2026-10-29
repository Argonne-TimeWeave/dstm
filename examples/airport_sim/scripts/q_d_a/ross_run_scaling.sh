#!/usr/bin/env bash
#PBS -N airport_all
#PBS -l select=8:ncpus=64:mpiprocs=32
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o airport_all.log
#
# ---- set your queue/project before submitting -----------------------------
##PBS -q <queue>
##PBS -A <project>
#
# run_all.sh -- MERGED scaling + MPI-activity comparison of the airport_sim
#   backends, producing ONE runs.csv and ONE summary.txt with ALL columns.
#
#     DSTM backends (timestep/BSP):
#       darray    -> simple_dstm/airports   (dstm_darray)
#       graph     -> graph_dstm/airports    (dstm_dgraph)
#       queue     -> queue_dstm/airports    (dstm_dqueue)
#
#     ROSS backends (optimistic PDES, continuous virtual time):
#       ross_opt  -> ross_airport  --synch=3  (optimistic + suspend LP)
#       ross_cons -> ross_airport  --synch=2  (conservative)
#
#   The single ROSS executable (built from ROSS/models/airport with PMPI
#   mpi_wrap.c linked in) is used for BOTH ross_opt and ross_cons; only the
#   --synch flag differs.
#
# TWO PASSES PER CONFIG
#   pass 1 (timing)   : run WITHOUT MPI_WRAP_OUTDIR; parse stdout for the
#                       throughput metric (committed plane-hops / second).
#   pass 2 (activity) : run WITH -x MPI_WRAP_OUTDIR=<dir>; collect per-rank
#                       mpi_counts_rank*.txt dumps (from mpi_wrap.c) and
#                       aggregate PMPI call counts / bytes per rank per tick.
#   Both passes write into the SAME CSV row for that config.
#
#   PREREQUISITE: all backends (DSTM *and* ROSS) must be linked with
#   mpi_wrap.o so pass 2 produces dumps. The script verifies dumps appear
#   and marks activity columns -1 (no_dumps) if they don't.
#
# FAIR THROUGHPUT METRIC (hops/sec), identical for every backend:
#       committed plane-hops / elapsed_seconds
#   DSTM hop  = one committed "Total departures:" entry.
#   ROSS hop  = Net Events Processed / ROSS_EVENTS_PER_HOP (=3:
#               DEPARTURE+ARRIVAL+LAND).
#   We do NOT use the DSTM "Timesteps/second:" line -- that is timesteps/s,
#   ~10000x smaller than hops/s, and would make DSTM look falsely slow.
#
# FAIRNESS -- TIME UNIT (ROSS vs DSTM), per weak_scale.sh:
#       ROSS --end = TIMESTEPS * ROSS_MEAN_DEPARTURE (=30.0)
#   so both engines do the same intended plane-hop workload.
#
# FAIRNESS -- CAPACITY (DSTM only):
#       EDGE_GATE_CAPACITY = CAP_PER_EDGE                 (graph header)
#       GATE_CAPACITY      = CAP_PER_EDGE * na            (darray header)
#       queue --cap        = CAP_PER_EDGE * na            (runtime flag)
#   darray/graph headers are rewritten + rebuilt per rung; queue takes --cap
#   at runtime; ROSS has no such knob and is never patched/rebuilt.
#
# HARD CONSTRAINT (all mains): na*num_ranks must be a perfect square; grid>=2.
#
# USAGE
#   qsub run_all.sh
#   ./run_all.sh --dry-run
#   ./run_all.sh --mode strong --reps 5
#   ./run_all.sh --impl "darray queue ross_opt"
#   ./run_all.sh --impl all                  # darray graph queue ross_opt ross_cons
#   ./run_all.sh --no-activity               # timing pass only (skip pass 2)
#   ./run_all.sh --no-timing                 # activity pass only (skip pass 1)
#
# OUTPUT
#   results/<stamp>/raw/*.log                      stdout+stderr per pass
#   results/<stamp>/dumps/<tag>/mpi_counts_rank*.txt  per-rank PMPI dumps
#   results/<stamp>/runs.csv                       one row per config (both passes)
#   results/<stamp>/summary.txt                    timing + activity tables
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
DQUEUE_EXE="${DQUEUE_EXE:-${ROOT}/queue_dstm/airports}"
ROSS_EXE="${ROSS_EXE:-${ROOT}/ross_airport}"

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

WEAK_LADDER="${WEAK_LADDER:-16:1:16 64:2:32 256:8:32}"
STRONG_LADDER="${STRONG_LADDER:-16:1:16 32:1:32 64:2:32 128:4:32 256:8:32}"

NA_PER_RANK="${NA_PER_RANK:-256}"
TOTAL_AIRPORTS="${TOTAL_AIRPORTS:-16384}"
PLANES="${PLANES:-4}"
TIMESTEPS="${TIMESTEPS:-512}"
SEED="${SEED:-42}"
REPS="${REPS:-3}"
TIMEOUT_S="${TIMEOUT_S:-600}"
MODES="weak strong"
IMPLS="darray graph queue ross_opt ross_cons"
VARIANTS="base"                         # base | congestion | novstate (graph only)
VERBOSITY="--quiet"
DRY_RUN=0
DO_TIMING=1                             # pass 1
DO_ACTIVITY=1                           # pass 2
OUTROOT="${OUTROOT:-${SCRIPT_DIR}/results}"

# ---- ROSS normalization knobs (match ROSS/weak_scale.sh) -----------------
ROSS_MEAN_DEPARTURE="${ROSS_MEAN_DEPARTURE:-30.0}"   # must match airport.h
ROSS_EVENTS_PER_HOP="${ROSS_EVENTS_PER_HOP:-3}"      # DEPARTURE+ARRIVAL+LAND
ROSS_EXTRAMEM="${ROSS_EXTRAMEM:-200000}"             # optimistic rollback mem
ROSS_LOOKAHEAD="${ROSS_LOOKAHEAD:-1.00}"             # conservative lookahead
ROSS_END="$(awk -v t="${TIMESTEPS}" -v m="${ROSS_MEAN_DEPARTURE}" \
            'BEGIN { printf "%.4f", t*m }')"

# ==========================================================================
# Capacity (fairness) -- DSTM only
# ==========================================================================
CAP_PER_EDGE="${CAP_PER_EDGE:-8}"
DARRAY_HDR="${DARRAY_HDR:-${ROOT}/simple_dstm/airport.h}"
DGRAPH_HDR="${DGRAPH_HDR:-${ROOT}/graph_dstm/airport.h}"
DARRAY_DIR="$(dirname "${DARRAY_EXE}")"
DGRAPH_DIR="$(dirname "${DGRAPH_EXE}")"
MAKE_CMD="${MAKE_CMD:-make}"
MAKE_FLAGS="${MAKE_FLAGS:-}"
MAKE_TARGET="${MAKE_TARGET:-}"
MAKE_NO_AUTOREGEN="${MAKE_NO_AUTOREGEN:-AUTOCONF=true AUTOHEADER=true AUTOMAKE=true ACLOCAL=true MAINTAINER_MODE=no}"

is_ross() { [[ "$1" == ross_opt || "$1" == ross_cons ]]; }

usage() {
        sed -n '7,86p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        cat <<EOF

options:
  --mode {weak|strong|both}         default: both
  --impl {darray|graph|queue|ross_opt|ross_cons|both|dstm|ross|all}
                                    both/dstm = darray graph; ross = ross_opt
                                    ross_cons; all = all five (default: all).
                                    Also accepts a quoted subset.
  --variants "a b c"                graph only: base congestion novstate
  --ladder / --weak-ladder / --strong-ladder "np:nodes:ppn ..."
  --placement {pack|spread}
  --max-ppn N / --max-nodes N
  --na-per-rank N                   weak airports/rank  (default ${NA_PER_RANK})
  --total N                         strong total        (default ${TOTAL_AIRPORTS})
  --planes P / --time T / --seed S / --reps R
  --timeout S                       per-run seconds, 0 = off (default ${TIMEOUT_S})
  --no-timing                       skip pass 1 (activity only)
  --no-activity                     skip pass 2 (timing only)
  --mpirun CMD / --mpi-flavor {auto|openmpi|mpich}
  --mpi-prefix DIR / --mpi-env "A B C" / --mpi-extra "ARGS"
  --cap-per-edge N                  DSTM only: GATE=N*na, EDGE=N, queue --cap=N*na
  --ross-exe PATH / --ross-mean-departure F / --ross-events-per-hop N
  --ross-extramem N / --ross-lookahead F
  --make CMD / --make-flags "ARGS" / --make-target NAME
  --info | --loud
  --out DIR / --dry-run / -h
EOF
}

while [[ $# -gt 0 ]]; do
        case "$1" in
        --mode)          MODES="$2"; [[ "$2" == both ]] && MODES="weak strong"; shift 2 ;;
        --impl)
                case "$2" in
                both|dstm) IMPLS="darray graph" ;;
                ross)      IMPLS="ross_opt ross_cons" ;;
                all)       IMPLS="darray graph queue ross_opt ross_cons" ;;
                *)         IMPLS="$2" ;;
                esac
                shift 2 ;;
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
        --no-timing)     DO_TIMING=0; shift ;;
        --no-activity)   DO_ACTIVITY=0; shift ;;
        --mpirun)        MPIRUN="$2"; shift 2 ;;
        --mpi-flavor)    MPI_FLAVOR="$2"; shift 2 ;;
        --mpi-prefix)    MPI_PREFIX="$2"; shift 2 ;;
        --mpi-env)       MPI_ENV_VARS="$2"; shift 2 ;;
        --mpi-extra)     MPI_EXTRA="$2"; shift 2 ;;
        --cap-per-edge)  CAP_PER_EDGE="$2"; shift 2 ;;
        --ross-exe)              ROSS_EXE="$2"; shift 2 ;;
        --ross-mean-departure)   ROSS_MEAN_DEPARTURE="$2"; shift 2 ;;
        --ross-events-per-hop)   ROSS_EVENTS_PER_HOP="$2"; shift 2 ;;
        --ross-extramem)         ROSS_EXTRAMEM="$2"; shift 2 ;;
        --ross-lookahead)        ROSS_LOOKAHEAD="$2"; shift 2 ;;
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

if (( DO_TIMING == 0 && DO_ACTIVITY == 0 )); then
        echo "ERROR: --no-timing and --no-activity together leave nothing to do." >&2
        exit 2
fi

# Recompute ROSS_END in case --time / --ross-mean-departure changed it.
ROSS_END="$(awk -v t="${TIMESTEPS}" -v m="${ROSS_MEAN_DEPARTURE}" \
            'BEGIN { printf "%.4f", t*m }')"

# MPI_WRAP_OUTDIR is forwarded per-run (pass 2 only). Keep user's list intact.
case " ${MPI_ENV_VARS} " in
        *" MPI_WRAP_OUTDIR "*) : ;;
        *) MPI_ENV_VARS="${MPI_ENV_VARS} MPI_WRAP_OUTDIR" ;;
esac

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

# Shared-FS assumption for pass-2 dumps: the dump dir must be visible under
# the SAME path on every node. On Cascade/JLSE $HOME is shared, so OUTROOT
# under the script dir is fine. For node-local disk, point --out at a shared
# path or activity aggregation will see only rank-0's node.

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
        [[ -n "${_mpirun_path}" ]] && MPI_PREFIX="$(dirname "$(dirname "${_mpirun_path}")")"
fi
# Base -x flags (everything EXCEPT MPI_WRAP_OUTDIR, which is appended per-run).
MPI_X_FLAGS=()
if [[ "${MPI_FLAVOR}" == openmpi ]]; then
        for _v in ${MPI_ENV_VARS}; do
                [[ "${_v}" == MPI_WRAP_OUTDIR ]] && continue
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
                        if (NR % 2)  { printf "%.6f\n", a[(NR + 1) / 2] }
                        else         { printf "%.6f\n", (a[NR/2] + a[NR/2 + 1]) / 2 }
                }'
}

exe_for() {
        case "$1" in
        darray)              echo "${DARRAY_EXE}" ;;
        graph)               echo "${DGRAPH_EXE}" ;;
        queue)               echo "${DQUEUE_EXE}" ;;
        ross_opt|ross_cons)  echo "${ROSS_EXE}" ;;
        esac
}

variant_flags() {
        [[ "$1" != graph ]] && { echo ""; return; }
        case "$2" in
        congestion) echo "--congestion" ;;
        novstate)   echo "--no-vstate" ;;
        *)          echo "" ;;
        esac
}

impl_extra_flags() {
        local impl="$1" na="$2"
        if [[ "${impl}" == queue ]]; then
                echo "--cap $(( CAP_PER_EDGE * na ))"
        else
                echo ""
        fi
}

ross_sync_flags() {
        case "$1" in
        ross_opt)  echo "--synch=3 --extramem=${ROSS_EXTRAMEM}" ;;
        ross_cons) echo "--synch=2 --lookahead=${ROSS_LOOKAHEAD}" ;;
        esac
}

hostfile_for() {
        local nodes="$1" hf i
        if [[ -n "${ALLOC_NODEFILE}" && "${nodes}" -eq "${#NODELIST[@]}" ]]; then
                echo "${ALLOC_NODEFILE}"; return
        fi
        hf="${HFDIR}/hosts-n${nodes}.txt"
        if [[ ! -f "${hf}" ]]; then
                : > "${hf}"
                for (( i = 0; i < nodes; i++ )); do echo "${NODELIST[$i]}" >> "${hf}"; done
        fi
        echo "${hf}"
}

# Build the launcher prefix array (no app, no MPI_WRAP_OUTDIR).
# $1=np $2=nodes $3=ppn ; result in global array LAUNCH[].
build_launch() {
        local np="$1" nodes="$2" ppn="$3"
        LAUNCH=()
        if [[ "${MPI_FLAVOR}" == openmpi ]]; then
                LAUNCH=("${MPIRUN}" -n "${np}" --hostfile "$(hostfile_for "${nodes}")")
                [[ -n "${MPI_PREFIX}" ]] && LAUNCH+=(--prefix "${MPI_PREFIX}")
                if (( np > 1 )) && (( np % nodes == 0 )); then
                        LAUNCH+=(--map-by "ppr:${ppn}:node")
                fi
                LAUNCH+=("${MPI_X_FLAGS[@]}")
        else
                LAUNCH=("${MPIRUN}" -n "${np}" -f "$(hostfile_for "${nodes}")" -ppn "${ppn}")
        fi
}

# Build the application argv for a backend. result in global array APPCMD[].
# $1=impl $2=variant $3=na $4=total
build_appcmd() {
        local impl="$1" variant="$2" na="$3" total="$4"
        local exe; exe="$(exe_for "${impl}")"
        APPCMD=()
        if is_ross "${impl}"; then
                # shellcheck disable=SC2206
                local sflags=( $(ross_sync_flags "${impl}") )
                APPCMD=("${exe}" "${sflags[@]}"
                        "--nairports=${total}" "--end=${ROSS_END}" "--nplanes=${PLANES}")
        else
                local flags extra
                flags="$(variant_flags "${impl}" "${variant}")"
                extra="$(impl_extra_flags "${impl}" "${na}")"
                # shellcheck disable=SC2206
                APPCMD=("${exe}" --na "${na}" --time "${TIMESTEPS}"
                        --planes "${PLANES}" --seed "${SEED}" ${flags} ${extra} ${VERBOSITY})
        fi
}

# --------------------------------------------------------------------------
# Header patching + rebuild (darray/graph only; queue=runtime, ross=never)
# --------------------------------------------------------------------------
HDR_BACKUPS=()
backup_headers() {
        local h
        for h in "${DARRAY_HDR}" "${DGRAPH_HDR}"; do
                [[ -f "${h}" ]] || continue
                cp -p "${h}" "${h}.runall.bak" || { echo "ERROR: cannot back up ${h}" >&2; exit 1; }
                HDR_BACKUPS+=("${h}")
        done
}
restore_headers() {
        local h
        for h in ${HDR_BACKUPS[@]+"${HDR_BACKUPS[@]}"}; do
                [[ -f "${h}.runall.bak" ]] || continue
                mv -f "${h}.runall.bak" "${h}"
        done
}
trap restore_headers EXIT INT TERM

patch_define() {
        local file="$1" name="$2" value="$3"
        grep -Eq "^[[:space:]]*#[[:space:]]*define[[:space:]]+${name}[[:space:]]+" "${file}" || {
                echo "ERROR: no '#define ${name}' found in ${file}" >&2; return 1; }
        sed -i -E "s|^([[:space:]]*#[[:space:]]*define[[:space:]]+${name}[[:space:]]+).*$|\1${value}|" "${file}" || return 1
        grep -Eq "^[[:space:]]*#[[:space:]]*define[[:space:]]+${name}[[:space:]]+${value}[[:space:]]*$" "${file}" || {
                echo "ERROR: ${name} did not take in ${file}" >&2; return 1; }
        return 0
}
needs_capacity_build() { [[ "${IMPLS}" == *darray* || "${IMPLS}" == *graph* ]]; }

CUR_CAP_NA=""
set_capacity() {
        local na="$1" gate_cap edge_cap impl dir rc
        needs_capacity_build || return 0
        edge_cap=${CAP_PER_EDGE}
        gate_cap=$(( CAP_PER_EDGE * na ))
        [[ "${CUR_CAP_NA}" == "${na}" ]] && return 0
        echo "  capacity: EDGE_GATE_CAPACITY=${edge_cap}  GATE_CAPACITY=${gate_cap}  queue --cap=${gate_cap}  (na=${na})"
        if [[ ${DRY_RUN} -eq 1 ]]; then
                echo "      would patch darray/graph headers and rebuild"
                CUR_CAP_NA="${na}"; return 0
        fi
        for impl in ${IMPLS}; do
                case "${impl}" in
                darray) patch_define "${DARRAY_HDR}" "GATE_CAPACITY" "${gate_cap}" || return 1 ;;
                graph)  patch_define "${DGRAPH_HDR}" "EDGE_GATE_CAPACITY" "${edge_cap}" || return 1 ;;
                esac
        done
        for impl in ${IMPLS}; do
                case "${impl}" in
                darray) dir="${DARRAY_DIR}" ;;
                graph)  dir="${DGRAPH_DIR}" ;;
                *)      continue ;;
                esac
                # shellcheck disable=SC2086
                ${MAKE_CMD} -C "${dir}" ${MAKE_NO_AUTOREGEN} ${MAKE_FLAGS} ${MAKE_TARGET} \
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

if needs_capacity_build; then
        for _h in "${DARRAY_HDR}" "${DGRAPH_HDR}"; do
                if [[ ! -w "${_h}" && ${DRY_RUN} -eq 0 ]]; then
                        echo "ERROR: header not writable: ${_h}" >&2
                        exit 1
                fi
        done
        [[ ${DRY_RUN} -eq 0 ]] && backup_headers
fi

TIMEOUT_CMD=()
if [[ "${TIMEOUT_S}" != "0" ]] && command -v timeout >/dev/null 2>&1; then
        TIMEOUT_CMD=(timeout --signal=TERM --kill-after=30 "${TIMEOUT_S}")
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUTDIR="${OUTROOT}/${STAMP}"
RAWDIR="${OUTDIR}/raw"
DUMPROOT="${OUTDIR}/dumps"
HFDIR="${OUTDIR}/hostfiles"
CSV="${OUTDIR}/runs.csv"
SUMMARY="${OUTDIR}/summary.txt"
mkdir -p "${RAWDIR}" "${DUMPROOT}" "${HFDIR}"

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
        echo "passes:       timing=${DO_TIMING}  activity=${DO_ACTIVITY}"
        echo "darray exe:   ${DARRAY_EXE}"
        echo "dgraph exe:   ${DGRAPH_EXE}"
        echo "dqueue exe:   ${DQUEUE_EXE}"
        echo "ross exe:     ${ROSS_EXE}"
        echo "modes:        ${MODES}"
        echo "impls:        ${IMPLS}    variants: ${VARIANTS}"
        echo "weak ladder:  ${WEAK_LADDER}"
        echo "strong ladr:  ${STRONG_LADDER}"
        echo "na/rank:      ${NA_PER_RANK} (weak)   total: ${TOTAL_AIRPORTS} (strong)"
        echo "cap/edge:     ${CAP_PER_EDGE}  (GATE/queue cap = cap/edge * na; DSTM only)"
        echo "planes:       ${PLANES}    timesteps: ${TIMESTEPS}    reps: ${REPS}"
        echo "ross end vt:  ${ROSS_END}  (= ${TIMESTEPS} * mean_dep=${ROSS_MEAN_DEPARTURE})"
        echo "ross hops:    net_events / ${ROSS_EVENTS_PER_HOP}"
        echo "ross opt:     --synch=3 --extramem=${ROSS_EXTRAMEM}"
        echo "ross cons:    --synch=2 --lookahead=${ROSS_LOOKAHEAD}"
        echo "dumps under:  ${DUMPROOT}"
        echo "timeout:      ${TIMEOUT_S}s"
} | tee "${OUTDIR}/config.txt"
echo

# CSV schema: identity + timing block + activity block.
# Timing block (from pass 1): elapsed tsps departures landings overflows
#   misroutes conservation expected cons_ok  (ROSS repurposes some, see NOTE).
# Activity block (from pass 2): dumps_found calls_total calls_pertick_mean
#   calls_pertick_max p2p_pertick_mean coll_pertick_mean completion_pertick_mean
#   isend_pt irecv_pt send_pt recv_pt wait_pt waitall_pt test_pt barrier_pt
#   reduce_pt allreduce_pt bcast_pt gather_pt allgather_pt alltoall_pt
#   bytes_sent_total bytes_recvd_total bytes_sent_pertick_mean
echo "impl,variant,mode,ranks,nodes,ppn,placement,na,total,grid,planes,timesteps,rep,status,\
elapsed,tsps,departures,landings,overflows,misroutes,conservation,expected,cons_ok,\
act_status,dumps_found,calls_total,calls_pertick_mean,calls_pertick_max,\
p2p_pertick_mean,coll_pertick_mean,completion_pertick_mean,\
isend_pt,irecv_pt,send_pt,recv_pt,wait_pt,waitall_pt,test_pt,\
barrier_pt,reduce_pt,allreduce_pt,bcast_pt,gather_pt,allgather_pt,alltoall_pt,\
bytes_sent_total,bytes_recvd_total,bytes_sent_pertick_mean" > "${CSV}"

# Number of fields in the activity block (dumps_found .. bytes_sent_pertick_mean).
ACT_NFIELDS=25
act_blank() { local i out="-1"; for (( i=2; i<=ACT_NFIELDS; i++ )); do out="${out},-1"; done; echo "${out}"; }

# ==========================================================================
# Aggregate one run's per-rank dumps -> comma-joined block
#   (dumps_found, calls_total, calls_pertick_mean, calls_pertick_max,
#    p2p_pt, coll_pt, completion_pt, <14 per-op pt>, bsent, brecvd, bsent_pt)
# $1=dumpdir $2=ranks $3=ticks
# ==========================================================================
aggregate_dumps() {
        local dumpdir="$1" ranks="$2" ticks="$3"
        local nf
        nf=$(find "${dumpdir}" -maxdepth 1 -name 'mpi_counts_rank*.txt' 2>/dev/null | wc -l)
        if [[ "${nf}" -eq 0 ]]; then echo "NODUMPS"; return; fi

        awk -v ranks="${ranks}" -v ticks="${ticks}" '
        /^rank / { seen++; cur = $2; }
        /^isend / {
                isend += $2; irecv += $4; send += $6; recv += $8;
                c[cur] += $2 + $4 + $6 + $8;
        }
        /^wait / {
                wait += $2; waitall += $4; testc += $6;
                c[cur] += $2 + $4 + $6;
        }
        /^barrier / {
                barrier += $2; reduce += $4; allreduce += $6;
                bcast += $8; gather += $10; allgather += $12; alltoall += $14;
                c[cur] += $2 + $4 + $6 + $8 + $10 + $12 + $14;
        }
        /^bytes_sent / { bsent += $2; brecvd += $4 }
        END {
                if (seen == 0) { print "NODUMPS"; exit }
                calls_total = isend+irecv+send+recv + wait+waitall+testc + \
                              barrier+reduce+allreduce+bcast+gather+allgather+alltoall;
                cmax = 0; for (r in c) if (c[r] > cmax) cmax = c[r];
                tk = (ticks > 0) ? ticks : 1;
                rk = (ranks > 0) ? ranks : seen;

                calls_pt_mean = (calls_total / rk) / tk;
                calls_pt_max  = cmax / tk;
                p2p   = isend+irecv+send+recv;
                coll  = barrier+reduce+allreduce+bcast+gather+allgather+alltoall;
                completion = wait+waitall+testc;
                p2p_pt   = (p2p/rk)/tk;  coll_pt = (coll/rk)/tk;  compl_pt = (completion/rk)/tk;

                isend_pt=(isend/rk)/tk; irecv_pt=(irecv/rk)/tk; send_pt=(send/rk)/tk; recv_pt=(recv/rk)/tk;
                wait_pt=(wait/rk)/tk; waitall_pt=(waitall/rk)/tk; test_pt=(testc/rk)/tk;
                barrier_pt=(barrier/rk)/tk; reduce_pt=(reduce/rk)/tk; allreduce_pt=(allreduce/rk)/tk;
                bcast_pt=(bcast/rk)/tk; gather_pt=(gather/rk)/tk; allgather_pt=(allgather/rk)/tk; alltoall_pt=(alltoall/rk)/tk;
                bsent_pt = (bsent/rk)/tk;

                printf "%d,%.0f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.0f,%.0f,%.2f",
                       seen, calls_total, calls_pt_mean, calls_pt_max, p2p_pt, coll_pt, compl_pt,
                       isend_pt, irecv_pt, send_pt, recv_pt, wait_pt, waitall_pt, test_pt,
                       barrier_pt, reduce_pt, allreduce_pt, bcast_pt, gather_pt, allgather_pt, alltoall_pt,
                       bsent, brecvd, bsent_pt;
        }' "${dumpdir}"/mpi_counts_rank*.txt 2>/dev/null
}

# ==========================================================================
# Parse the timing block from a pass-1 log into globals:
#   T_elapsed T_tsps T_dep T_land T_ovf T_mis T_cons T_exp T_consok
# $1=impl $2=log $3=wallclock_fallback ; sets T_status_note ("" or "parse_fail")
# ==========================================================================
parse_timing() {
        local impl="$1" log="$2" wall="$3"
        T_elapsed=""; T_tsps="nan"; T_dep="-1"; T_land="-1"; T_ovf="-1"
        T_mis="-1"; T_cons="-1"; T_exp="-1"; T_consok="n/a"; T_status_note=""

        if is_ross "${impl}"; then
                T_elapsed=$(awk -F'=' '/Running Time[[:space:]]*=/ { v=$2 } END { print v }' "${log}" \
                            | awk '{ print $1 }')
                local net
                net=$(awk '/Net Events Processed/ { v=$NF } END { print v }' "${log}")
                if [[ -n "${net}" ]]; then
                        T_dep=$(awk -v c="${net}" -v e="${ROSS_EVENTS_PER_HOP}" \
                                'BEGIN { printf "%d", int(c/e + 0.5) }')
                fi
                if [[ -n "${T_elapsed}" && "${T_dep}" != "-1" ]]; then
                        T_tsps=$(awk -v h="${T_dep}" -v t="${T_elapsed}" \
                                 'BEGIN { if (t+0>0) printf "%.2f", h/t; else print "nan" }')
                fi
                T_mis=$(awk '/Events Rolled Back/ { v=$NF } END { print (v==""?"-1":v) }' "${log}")
                local eff
                eff=$(awk '/^[[:space:]]*Efficiency/ { v=$(NF-1) } END { print v }' "${log}")
                [[ -n "${eff}" ]] && { T_cons="${eff}"; T_exp="pct"; T_consok="eff"; }
        else
                T_elapsed=$(awk '/^Elapsed time:/      { print $3 }' "${log}" | tail -1)
                T_dep=$(    awk '/^Total departures:/  { print $3 }' "${log}" | tail -1)
                T_land=$(   awk '/^Total landings:/    { print $3 }' "${log}" | tail -1)
                # Throughput = committed hops/sec (NOT the model's Timesteps/second line).
                if [[ -n "${T_elapsed}" && -n "${T_dep}" ]]; then
                        T_tsps=$(awk -v h="${T_dep}" -v t="${T_elapsed}" \
                                 'BEGIN { if (t+0>0) printf "%.2f", h/t; else print "nan" }')
                else
                        T_tsps="nan"
                fi
                T_ovf=$(   awk '/^Gate overflows:/ || /^Queue overflows:/ { print $3 }' "${log}" | tail -1)
                T_mis=$(   awk '/^Misrouted flights:/ { print $3 }' "${log}" | tail -1)
                T_cons=$(  awk '/^Conservation:/      { print $2 }' "${log}" | tail -1)
                T_exp=$(   awk '/^Conservation:/      { print $4 }' "${log}" | tail -1)
                T_consok=$(awk '/^Conservation:/      { print $5 }' "${log}" | tail -1)
        fi

        if [[ -z "${T_elapsed}" ]]; then
                T_elapsed="${wall}"
                T_status_note="parse_fail"
        fi
        : "${T_tsps:=nan}" "${T_dep:=-1}" "${T_land:=-1}" "${T_ovf:=-1}"
        : "${T_mis:=-1}" "${T_cons:=-1}" "${T_exp:=-1}" "${T_consok:=n/a}"
}

# Run one mpirun invocation of APPCMD; returns rc, writes $1=log.
# $2.. = extra launcher args inserted right after LAUNCH (e.g. -x VAR=VAL).
# Uses a subshell cd for ROSS so ross.csv/ross_all_lp_stats stay contained.
exec_run() {
        local log="$1" impl="$2" tag="$3"; shift 3
        local extra_launch=( "$@" )
        local runcwd="${PWD}"
        if is_ross "${impl}"; then
                runcwd="${RAWDIR}/${tag}.d"
                mkdir -p "${runcwd}"
        fi
        local full=( "${LAUNCH[@]}" "${extra_launch[@]}" ${MPI_EXTRA} "${APPCMD[@]}" )
        ( cd "${runcwd}" && "${TIMEOUT_CMD[@]}" "${full[@]}" ) > "${log}" 2>&1
        return $?
}

classify_rc() {
        local rc="$1" log="$2"
        local st="ok"
        if   [[ ${rc} -eq 124 || ${rc} -eq 137 ]]; then st="timeout"
        elif [[ ${rc} -ne 0 ]];                   then st="fail(${rc})"
        fi
        if grep -q "orted: command not found\|ORTE was unable to reliably start" "${log}" 2>/dev/null; then
                st="launch_err"
        fi
        echo "${st}"
}

# ==========================================================================
# One config (two passes) -> one CSV row
# ==========================================================================
run_one() {
        local impl="$1" variant="$2" mode="$3" np="$4" nodes="$5" ppn="$6"
        local na="$7" total="$8" grid="$9" rep="${10}"
        local tag="${impl}-${variant}-${mode}-np${np}-n${nodes}x${ppn}-na${na}-r${rep}"

        build_launch "${np}" "${nodes}" "${ppn}"
        build_appcmd "${impl}" "${variant}" "${na}" "${total}"

        printf '  %-9s %-10s np=%-4s %sx%-3s na=%-6s total=%-7s rep=%s ... ' \
               "${impl}" "${variant}" "${np}" "${nodes}" "${ppn}" "${na}" "${total}" "${rep}"

        if [[ ${DRY_RUN} -eq 1 ]]; then
                printf '\n'
                (( DO_TIMING ))   && printf '      [timing]   %s %s\n' "${LAUNCH[*]}" "${APPCMD[*]}"
                (( DO_ACTIVITY )) && printf '      [activity] %s -x MPI_WRAP_OUTDIR=%s %s\n' \
                        "${LAUNCH[*]}" "${DUMPROOT}/${tag}" "${APPCMD[*]}"
                return 0
        fi

        local status="ok" act_status="skip"
        local elapsed tsps dep land ovf mis cons exp cons_ok
        elapsed="-1"; tsps="nan"; dep="-1"; land="-1"; ovf="-1"; mis="-1"
        cons="-1"; exp="-1"; cons_ok="n/a"
        local act_block; act_block="$(act_blank)"   # dumps_found=-1 ... = skipped

        # ---------------- Pass 1: timing (no wrapper outdir) --------------
        if (( DO_TIMING )); then
                local tlog="${RAWDIR}/${tag}.timing.log"
                local t0 t1 wall rc
                t0=$(date +%s.%N)
                exec_run "${tlog}" "${impl}" "${tag}"
                rc=$?
                t1=$(date +%s.%N)
                wall=$(awk -v a="${t0}" -v b="${t1}" 'BEGIN { printf "%.4f", b - a }')
                status="$(classify_rc "${rc}" "${tlog}")"

                parse_timing "${impl}" "${tlog}" "${wall}"
                elapsed="${T_elapsed}"; tsps="${T_tsps}"; dep="${T_dep}"; land="${T_land}"
                ovf="${T_ovf}"; mis="${T_mis}"; cons="${T_cons}"; exp="${T_exp}"; cons_ok="${T_consok}"
                [[ "${status}" == ok && -n "${T_status_note}" ]] && status="${T_status_note}"
        fi

        # ---------------- Pass 2: activity (wrapper dumps) ----------------
        if (( DO_ACTIVITY )); then
                local alog="${RAWDIR}/${tag}.activity.log"
                local dumpdir="${DUMPROOT}/${tag}"
                mkdir -p "${dumpdir}"
                export MPI_WRAP_OUTDIR="${dumpdir}"

                local xarg=()
                if [[ "${MPI_FLAVOR}" == openmpi ]]; then
                        xarg=(-x "MPI_WRAP_OUTDIR=${dumpdir}")
                else
                        xarg=(-genv MPI_WRAP_OUTDIR "${dumpdir}")
                fi

                local rc2
                exec_run "${alog}" "${impl}" "${tag}" "${xarg[@]}"
                rc2=$?
                act_status="$(classify_rc "${rc2}" "${alog}")"

                # If timing was skipped, use activity run for the overall status.
                if (( DO_TIMING == 0 )); then status="${act_status}"; fi

                if [[ "${act_status}" == ok ]]; then
                        local waited=0 nf=0
                        while (( waited < 30 )); do
                                nf=$(find "${dumpdir}" -maxdepth 1 \
                                     -name 'mpi_counts_rank*.txt' 2>/dev/null | wc -l)
                                (( nf >= np )) && break
                                sleep 1; (( waited++ ))
                        done
                        local agg; agg="$(aggregate_dumps "${dumpdir}" "${np}" "${TIMESTEPS}")"
                        if [[ -z "${agg}" || "${agg}" == "NODUMPS" ]]; then
                                act_status="no_dumps"
                                act_block="$(act_blank)"
                        else
                                act_block="${agg}"
                        fi
                fi
                [[ "${act_status}" != ok ]] && act_block="$(act_blank)"
                unset MPI_WRAP_OUTDIR
        fi

        echo "${impl},${variant},${mode},${np},${nodes},${ppn},${PLACEMENT},${na},${total},${grid},${PLANES},${TIMESTEPS},${rep},${status},${elapsed},${tsps},${dep},${land},${ovf},${mis},${cons},${exp},${cons_ok},${act_status},${act_block}" >> "${CSV}"

        # ---- console line ----
        if [[ "${status}" == ok || "${status}" == parse_fail ]]; then
                local cpt dumps_found
                dumps_found="${act_block%%,*}"
                cpt="$(echo "${act_block}" | awk -F, '{ print $3 }')"
                printf '%9ss %12s hops/s  act=%-8s dumps=%-4s calls/t=%s\n' \
                       "${elapsed}" "${tsps}" "${act_status}" "${dumps_found}" "${cpt}"
        else
                printf '%s (%s)\n' "${status}" "${tag}"
        fi
}

# ==========================================================================
# Sweep
# ==========================================================================
ladder_for() { case "$1" in weak) echo "${WEAK_LADDER}";; strong) echo "${STRONG_LADDER}";; esac; }

for mode in ${MODES}; do
        echo "===== ${mode} scaling + activity ====="
        for rung in $(ladder_for "${mode}"); do
                np="${rung%%:*}"; rest="${rung#*:}"; nodes="${rest%%:*}"; ppn="${rest##*:}"

                if [[ "${PLACEMENT}" == spread ]]; then
                        nodes=${AVAIL_NODES}; (( nodes > np )) && nodes=${np}
                        ppn=$(( (np + nodes - 1) / nodes ))
                fi
                (( ppn > MAX_PPN ))       && { echo "  skip np=${np}: ppn ${ppn} > cap ${MAX_PPN}"; continue; }
                (( nodes > AVAIL_NODES )) && { echo "  skip np=${np}: needs ${nodes} nodes, have ${AVAIL_NODES}"; continue; }
                (( nodes * ppn < np ))    && { echo "  skip np=${np}: ${nodes}x${ppn} cannot host ${np}"; continue; }

                if [[ "${mode}" == weak ]]; then
                        na=${NA_PER_RANK}; total=$(( na * np ))
                else
                        (( TOTAL_AIRPORTS % np != 0 )) && { echo "  skip np=${np}: ${TOTAL_AIRPORTS} not divisible by ranks"; continue; }
                        na=$(( TOTAL_AIRPORTS / np )); total=${TOTAL_AIRPORTS}
                fi

                grid=$(isqrt_exact "${total}")
                [[ "${grid}" == "0" ]] && { echo "  skip np=${np}: total ${total} not a perfect square"; continue; }
                (( grid < 2 ))         && { echo "  skip np=${np}: grid ${grid} < 2 (mains reject it)"; continue; }

                if ! set_capacity "${na}"; then
                        echo "  skip np=${np}: capacity build failed for na=${na}"
                        continue
                fi

                for impl in ${IMPLS}; do
                        vlist="${VARIANTS}"
                        [[ "${impl}" != graph ]] && vlist="base"
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
# ==========================================================================
# CSV column index reference (1-based):
#   1 impl 2 variant 3 mode 4 ranks 5 nodes 6 ppn 7 placement 8 na 9 total
#   10 grid 11 planes 12 timesteps 13 rep 14 status
#   15 elapsed 16 tsps 17 departures 18 landings 19 overflows 20 misroutes
#   21 conservation 22 expected 23 cons_ok
#   24 act_status 25 dumps_found 26 calls_total 27 calls_pertick_mean
#   28 calls_pertick_max 29 p2p_pertick_mean 30 coll_pertick_mean
#   31 completion_pertick_mean 32 isend_pt 33 irecv_pt 34 send_pt 35 recv_pt
#   36 wait_pt 37 waitall_pt 38 test_pt 39 barrier_pt 40 reduce_pt
#   41 allreduce_pt 42 bcast_pt 43 gather_pt 44 allgather_pt 45 alltoall_pt
#   46 bytes_sent_total 47 bytes_recvd_total 48 bytes_sent_pertick_mean
# ==========================================================================
np_list_for() { awk -F, -v m="$1" 'NR > 1 && $3 == m { print $4 }' "${CSV}" | sort -n -u; }

# median of CSV column $5 for (mode,impl,variant,np) over timing-ok rows.
med_timing() {
        awk -F, -v m="$1" -v i="$2" -v v="$3" -v n="$4" -v c="$5" \
            '$3==m && $1==i && $2==v && $4==n && $14=="ok" { print $c }' "${CSV}" | median
}
# median of CSV column $5 for (mode,impl,variant,np) over activity-ok rows.
med_act() {
        awk -F, -v m="$1" -v i="$2" -v v="$3" -v n="$4" -v c="$5" \
            '$3==m && $1==i && $2==v && $4==n && $24=="ok" { print $c }' "${CSV}" | median
}

summarize_timing() {
        local mode="$1" keys key impl variant np med base_med first_np na tot ts nodes ppn
        keys=$(awk -F, -v m="${mode}" 'NR > 1 && $3 == m { print $1 "|" $2 }' "${CSV}" | sort -u)
        [[ -z "${keys}" ]] && return

        echo "===== ${mode} scaling -- throughput (median of ${REPS} reps) ====="
        printf '%-10s %-10s %6s %8s %7s %8s %11s %13s %9s %8s\n' \
               impl variant ranks layout na total "elapsed(s)" "hops/s" speedup eff
        printf -- '----------------------------------------------------------------------------------------------------------\n'

        for key in ${keys}; do
                impl="${key%%|*}"; variant="${key##*|}"
                base_med=""; first_np=""
                for np in $(np_list_for "${mode}"); do
                        med=$(med_timing "${mode}" "${impl}" "${variant}" "${np}" 15)
                        [[ "${med}" == "nan" ]] && continue
                        ts=$(med_timing "${mode}" "${impl}" "${variant}" "${np}" 16)
                        read -r nodes ppn na tot < <(awk -F, -v m="${mode}" -v i="${impl}" \
                                -v v="${variant}" -v n="${np}" \
                                '$3==m && $1==i && $2==v && $4==n { print $5, $6, $8, $9; exit }' "${CSV}")
                        if [[ -z "${base_med}" ]]; then base_med="${med}"; first_np="${np}"; fi
                        awk -v i="${impl}" -v v="${variant}" -v n="${np}" \
                            -v lay="${nodes}x${ppn}" -v na="${na}" -v tot="${tot}" \
                            -v e="${med}" -v ts="${ts}" -v b="${base_med}" \
                            -v bn="${first_np}" -v mode="${mode}" '
                        BEGIN {
                                sp  = (e > 0) ? b / e : 0;
                                eff = (mode == "weak") ? sp : sp / (n / bn);
                                printf "%-10s %-10s %6s %8s %7s %8s %11.4f %13.2f %8.2fx %7.0f%%\n",
                                       i, v, n, lay, na, tot, e, ts, sp, eff * 100;
                        }'
                done
                echo
        done

        # ---- head to head: median elapsed(s), base variant ----------------
        local cols=() c
        for c in darray graph queue ross_opt ross_cons; do
                [[ "${IMPLS}" == *"${c}"* ]] && cols+=("${c}")
        done
        (( ${#cols[@]} < 2 )) && return

        echo "  head to head: median elapsed(s), base variant.  'best' = fastest backend."
        printf '  %6s %8s' ranks layout
        for c in "${cols[@]}"; do printf ' %12s' "${c}"; done
        printf '   %-10s\n' best

        for np in $(np_list_for "${mode}"); do
                local vals=() lay all_nan=1 m2
                for c in "${cols[@]}"; do
                        m2=$(med_timing "${mode}" "${c}" base "${np}" 15)
                        vals+=("${m2}")
                        [[ "${m2}" != "nan" ]] && all_nan=0
                done
                (( all_nan )) && continue
                lay=$(awk -F, -v m="${mode}" -v n="${np}" '$3==m && $4==n { print $5 "x" $6; exit }' "${CSV}")
                {
                        printf '%s\t%s' "${np}" "${lay}"
                        for i in "${!cols[@]}"; do printf '\t%s\t%s' "${cols[$i]}" "${vals[$i]}"; done
                        printf '\n'
                } | awk -F'\t' '
                function fmt(x) { return (x == "nan" || x == "") ? "           -" : sprintf("%12.4f", x) }
                {
                        n=$1; lay=$2; printf "  %6s %8s", n, lay;
                        best="-"; bv=0; first=1;
                        for (i=3; i<=NF; i+=2) {
                                name=$i; val=$(i+1); printf " %s", fmt(val);
                                if (val != "nan" && val != "") {
                                        if (first || val+0 < bv) { bv=val+0; best=name; first=0 }
                                }
                        }
                        printf "   %-10s\n", best;
                }'
        done
        echo
}

summarize_activity() {
        local mode="$1" keys key impl variant np
        # Only report if any activity rows exist for this mode.
        local any
        any=$(awk -F, -v m="${mode}" 'NR>1 && $3==m && $24=="ok"' "${CSV}" | wc -l)
        (( any == 0 )) && return

        keys=$(awk -F, -v m="${mode}" 'NR>1 && $3==m && $24=="ok" { print $1 "|" $2 }' "${CSV}" | sort -u)
        [[ -z "${keys}" ]] && return

        echo "===== ${mode} MPI activity -- calls per rank per tick (median of ${REPS}) ====="
        printf '%-10s %-10s %6s %7s %12s %12s %10s %10s %13s\n' \
               impl variant ranks na "calls/t mean" "calls/t max" "p2p/t" "coll/t" "bytes_sent/t"
        printf -- '-------------------------------------------------------------------------------------------------------------\n'
        for key in ${keys}; do
                impl="${key%%|*}"; variant="${key##*|}"
                for np in $(np_list_for "${mode}"); do
                        local cm cx p2 co by na
                        cm=$(med_act "${mode}" "${impl}" "${variant}" "${np}" 27)
                        [[ "${cm}" == "nan" ]] && continue
                        cx=$(med_act "${mode}" "${impl}" "${variant}" "${np}" 28)
                        p2=$(med_act "${mode}" "${impl}" "${variant}" "${np}" 29)
                        co=$(med_act "${mode}" "${impl}" "${variant}" "${np}" 30)
                        by=$(med_act "${mode}" "${impl}" "${variant}" "${np}" 48)
                        na=$(awk -F, -v m="${mode}" -v i="${impl}" -v v="${variant}" -v n="${np}" \
                                '$3==m && $1==i && $2==v && $4==n { print $8; exit }' "${CSV}")
                        printf '%-10s %-10s %6s %7s %12s %12s %10s %10s %13s\n' \
                               "${impl}" "${variant}" "${np}" "${na}" "${cm}" "${cx}" "${p2}" "${co}" "${by}"
                done
                echo
        done

        echo "===== ${mode} MPI activity -- per-category calls per rank per tick (base variant) ====="
        printf '%-10s %6s %9s %9s %8s %8s %8s %8s %8s %8s %8s %9s %8s %8s %8s\n' \
               impl ranks isend irecv send recv wait waitall test barrier reduce allred bcast gather allgat
        printf -- '-------------------------------------------------------------------------------------------------------------------------------------\n'
        for impl in ${IMPLS}; do
                for np in $(np_list_for "${mode}"); do
                        local v1
                        v1=$(med_act "${mode}" "${impl}" base "${np}" 32)
                        [[ "${v1}" == "nan" ]] && continue
                        printf '%-10s %6s %9s %9s %8s %8s %8s %8s %8s %8s %8s %9s %8s %8s %8s\n' \
                               "${impl}" "${np}" \
                               "$(med_act "${mode}" "${impl}" base "${np}" 32)" \
                               "$(med_act "${mode}" "${impl}" base "${np}" 33)" \
                               "$(med_act "${mode}" "${impl}" base "${np}" 34)" \
                               "$(med_act "${mode}" "${impl}" base "${np}" 35)" \
                               "$(med_act "${mode}" "${impl}" base "${np}" 36)" \
                               "$(med_act "${mode}" "${impl}" base "${np}" 37)" \
                               "$(med_act "${mode}" "${impl}" base "${np}" 38)" \
                               "$(med_act "${mode}" "${impl}" base "${np}" 39)" \
                               "$(med_act "${mode}" "${impl}" base "${np}" 40)" \
                               "$(med_act "${mode}" "${impl}" base "${np}" 41)" \
                               "$(med_act "${mode}" "${impl}" base "${np}" 42)" \
                               "$(med_act "${mode}" "${impl}" base "${np}" 43)" \
                               "$(med_act "${mode}" "${impl}" base "${np}" 44)"
                done
                echo
        done
}

{
        cat "${OUTDIR}/config.txt"
        echo
        echo "NOTE: throughput metric = committed plane-hops / second."
        echo "  DSTM hop = one 'Total departures:' entry."
        echo "  ROSS hop = Net Events Processed / ${ROSS_EVENTS_PER_HOP}."
        echo "NOTE: for ROSS timing rows, repurposed columns:"
        echo "  departures = committed hops ;  misroutes = Events Rolled Back ;"
        echo "  conservation = Efficiency %% ;  landings/overflows = -1 (n/a)."
        echo "NOTE: activity numbers are per-rank-per-tick means from mpi_wrap.c"
        echo "  dumps (pass 2). Rows with act_status != ok show -1."
        echo

        if (( DO_TIMING )); then
                for mode in ${MODES}; do summarize_timing "${mode}"; done
        else
                echo "(timing pass skipped: --no-timing)"
                echo
        fi

        if (( DO_ACTIVITY )); then
                for mode in ${MODES}; do summarize_activity "${mode}"; done
        else
                echo "(activity pass skipped: --no-activity)"
                echo
        fi

        total_runs=$(( $(wc -l < "${CSV}") - 1 ))
        bad=$(awk -F, 'NR > 1 && $14 != "ok"' "${CSV}" | wc -l)
        tmo=$(awk -F, 'NR > 1 && $14 == "timeout"' "${CSV}" | wc -l)
        lerr=$(awk -F, 'NR > 1 && $14 == "launch_err"' "${CSV}" | wc -l)
        nod=$(awk -F, 'NR > 1 && $24 == "no_dumps"' "${CSV}" | wc -l)
        acterr=$(awk -F, 'NR > 1 && $24 != "ok" && $24 != "skip"' "${CSV}" | wc -l)

        echo "runs recorded:           ${total_runs}"
        echo "timing non-ok:           ${bad}  (timeouts: ${tmo}, launcher: ${lerr})"
        echo "activity non-ok:         ${acterr}  (no_dumps: ${nod})"
        if (( nod > 0 )); then
                echo
                echo "NOTE: ${nod} run(s) produced NO per-rank dumps in pass 2. Either the"
                echo "      binary was not linked with mpi_wrap.o, or MPI_WRAP_OUTDIR did not"
                echo "      reach the ranks (check -x/-genv forwarding and shared-FS path)."
        fi
        if (( lerr > 0 )); then
                echo
                echo "NOTE: ${lerr} run(s) failed in the MPI launcher, not the application."
                echo "      Check --mpi-prefix (currently '${MPI_PREFIX}') and --mpi-env."
        fi
        if (( bad > 0 )); then
                echo
                echo "non-ok timing runs:"
                awk -F, 'NR > 1 && $14 != "ok" { printf "  %-9s %-10s %-6s np=%-4s %sx%-3s -> %s\n", $1,$2,$3,$4,$5,$6,$14 }' "${CSV}"
        fi
} | tee "${SUMMARY}"

echo
echo "csv:     ${CSV}"
echo "summary: ${SUMMARY}"
echo "dumps:   ${DUMPROOT}"
echo "logs:    ${RAWDIR}"
