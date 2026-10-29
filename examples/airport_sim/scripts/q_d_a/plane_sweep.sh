#!/usr/bin/env bash
#PBS -N airport_planes_sweep
#PBS -l select=8:ncpus=64:mpiprocs=32
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o airport_planes_sweep.log
#
# ---- set your queue/project before submitting -----------------------------
##PBS -q <queue>
##PBS -A <project>
#
# run_planes_sweep.sh -- PLANES sweep at FIXED ranks=256, for all airport_sim
#   backends (darray/graph/queue/ross_opt/ross_cons). Everything except the
#   plane count is held at the run_all.sh defaults (ranks=256, na=256,
#   total=65536, timesteps=512, seed=42, cap/edge=8, reps=3).
#
#   Swept axis:
#       PLANES in { 1 4 16 64 256 1024 4096 16384 }
#
#   Two passes per (impl, planes) config, exactly like run_all.sh:
#       pass 1 (timing)   : no MPI_WRAP_OUTDIR; hops/sec throughput.
#       pass 2 (activity) : -x MPI_WRAP_OUTDIR=<dir>; per-rank PMPI dumps.
#
#   All backends must be linked with mpi_wrap.o so pass 2 produces dumps.
#
# RANKS ARE FIXED
#   This script runs ONLY the single rung  256:8:32  (np=256, 8 nodes x 32 ppn),
#   weak-mode sizing -> na=256, total=65536 (256x256 torus, a perfect square).
#   It does NOT sweep ranks and does NOT run strong mode.
#
# THROUGHPUT METRIC (fair, per backend): committed plane-hops / second.
#   DSTM hop = one 'Total departures:' entry.
#   ROSS hop = Net Events Processed / ROSS_EVENTS_PER_HOP (=3).
#   ROSS --end = TIMESTEPS * ROSS_MEAN_DEPARTURE (=30.0) for matched workload.
#
# OUTPUT (discernible planes-sweep naming)
#   results_planes_sweep/<stamp>/raw/<tag>.{timing,activity}.log
#   results_planes_sweep/<stamp>/dumps/<tag>/mpi_counts_rank*.txt
#       where <tag> includes  pl<planes>  so dump dirs are per-plane-count.
#   results_planes_sweep/<stamp>/runs.csv      one row per (impl, planes, rep)
#   results_planes_sweep/<stamp>/summary.txt   per-planes throughput + activity
#
# USAGE
#   qsub run_planes_sweep.sh
#   ./run_planes_sweep.sh --dry-run
#   ./run_planes_sweep.sh --impl "darray ross_opt" --reps 1
#   ./run_planes_sweep.sh --planes-list "1 16 256 4096"
#   ./run_planes_sweep.sh --no-activity
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
DQUEUE_BATCH_EXE="${DQUEUE_BATCH_EXE:-${ROOT}/batch_queue/airports}"
ROSS_EXE="${ROSS_EXE:-${ROOT}/ross_airport}"

# ==========================================================================
# Defaults  (everything but PLANES held fixed)
# ==========================================================================
MPIRUN="${MPIRUN:-mpirun}"
MPI_FLAVOR="${MPI_FLAVOR:-auto}"        # auto | openmpi | mpich
MPI_EXTRA="${MPI_EXTRA:-}"
MPI_PREFIX="${MPI_PREFIX:-}"
MPI_ENV_VARS="${MPI_ENV_VARS:-LD_LIBRARY_PATH PATH}"
MAX_PPN="${MAX_PPN:-32}"
MAX_NODES="${MAX_NODES:-8}"
PLACEMENT="${PLACEMENT:-pack}"          # pack | spread

# FIXED rung: np=256 on 8 nodes x 32 ppn.
RUNG="${RUNG:-256:8:32}"

# Weak-mode sizing at np=256 -> na=256, total=65536 (256x256 perfect square).
NA_PER_RANK="${NA_PER_RANK:-256}"

# Fixed capacity for the batched queue (planes-independent on purpose).
QUEUE_BATCH_CAP="${QUEUE_BATCH_CAP:-32}"

# ---- The swept axis ------------------------------------------------------
PLANES_LIST="${PLANES_LIST:-1 2 4 8 16 32 64 128 256}"

TIMESTEPS="${TIMESTEPS:-512}"
SEED="${SEED:-42}"
REPS="${REPS:-3}"
TIMEOUT_S="${TIMEOUT_S:-100}"
IMPLS="darray graph queue queue_batch ross_opt ross_cons"
VARIANTS="base"                         # base | congestion | novstate (graph only)
VERBOSITY="--quiet"
DRY_RUN=0
DO_TIMING=1
DO_ACTIVITY=1
OUTROOT="${OUTROOT:-${SCRIPT_DIR}/results_planes_sweep}"

# ---- ROSS normalization knobs (match run_all.sh / weak_scale.sh) ---------
ROSS_MEAN_DEPARTURE="${ROSS_MEAN_DEPARTURE:-30.0}"
ROSS_EVENTS_PER_HOP="${ROSS_EVENTS_PER_HOP:-3}"
ROSS_EXTRAMEM="${ROSS_EXTRAMEM:-200000}"
ROSS_LOOKAHEAD="${ROSS_LOOKAHEAD:-1.00}"
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
        sed -n '7,55p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        cat <<EOF

options:
  --planes-list "1 4 16 ..."     plane counts to sweep (default: ${PLANES_LIST})
  --rung np:nodes:ppn            fixed rank rung (default: ${RUNG})
  --na-per-rank N                airports/rank at the fixed rung (default: ${NA_PER_RANK})
  --impl {darray|graph|queue|ross_opt|ross_cons|both|dstm|ross|all}
                                 default: all five; accepts a quoted subset
  --variants "a b c"             graph only: base congestion novstate
  --time T / --seed S / --reps R
  --timeout S                    per-run seconds, 0 = off (default ${TIMEOUT_S})
  --no-timing / --no-activity
  --placement {pack|spread}  --max-ppn N / --max-nodes N
  --mpirun CMD / --mpi-flavor {auto|openmpi|mpich}
  --mpi-prefix DIR / --mpi-env "A B C" / --mpi-extra "ARGS"
  --cap-per-edge N
  --ross-exe PATH / --ross-mean-departure F / --ross-events-per-hop N
  --ross-extramem N / --ross-lookahead F
  --make CMD / --make-flags "ARGS" / --make-target NAME
  --info | --loud  --out DIR / --dry-run / -h
EOF
}

while [[ $# -gt 0 ]]; do
        case "$1" in
        --planes-list)   PLANES_LIST="$2"; shift 2 ;;
        --rung)          RUNG="$2"; shift 2 ;;
        --na-per-rank)   NA_PER_RANK="$2"; shift 2 ;;
        --impl)
                case "$2" in
                both|dstm) IMPLS="darray graph" ;;
                ross)      IMPLS="ross_opt ross_cons" ;;
                all)       IMPLS="darray graph queue queue_batch ross_opt ross_cons" ;;
                *)         IMPLS="$2" ;;
                esac
                shift 2 ;;
        --variants)      VARIANTS="$2"; shift 2 ;;
        --placement)     PLACEMENT="$2"; shift 2 ;;
        --max-ppn)       MAX_PPN="$2"; shift 2 ;;
        --max-nodes)     MAX_NODES="$2"; shift 2 ;;
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

ROSS_END="$(awk -v t="${TIMESTEPS}" -v m="${ROSS_MEAN_DEPARTURE}" \
            'BEGIN { printf "%.4f", t*m }')"

case " ${MPI_ENV_VARS} " in
        *" MPI_WRAP_OUTDIR "*) : ;;
        *) MPI_ENV_VARS="${MPI_ENV_VARS} MPI_WRAP_OUTDIR" ;;
esac

# Fixed rung decomposition.
NP="${RUNG%%:*}"; _rest="${RUNG#*:}"; NODES="${_rest%%:*}"; PPN="${_rest##*:}"

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
        echo "         The fixed np=${NP} rung needs ${NODES} nodes and will be skipped." >&2
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
        [[ -n "${_mpirun_path}" ]] && MPI_PREFIX="$(dirname "$(dirname "${_mpirun_path}")")"
fi
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
        queue_batch)         echo "${DQUEUE_BATCH_EXE}" ;;
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
        case "${impl}" in
        queue)
                # original queue: capacity ~ cap/edge * na (as before)
                echo "--cap $(( ${CAP_PER_EDGE} * na ))"
                ;;
        queue_batch)
                # batched queue: one element = one batch; a slot holds at
                # most (#neighbors + self) batches per tick, so a small
                # FIXED cap is correct and gives real backpressure.
                echo "--cap ${QUEUE_BATCH_CAP}"
                ;;
        *)
                echo ""
                ;;
        esac
}

ross_sync_flags() {
        local em=$(( ROSS_EXTRAMEM * ( $2 > 0 ? $2 : 1 ) ))
        case "$1" in
        ross_opt)  echo "--synch=3 --extramem=${ROSS_EXTRAMEM}" ;;
        ross_cons) echo "--synch=2 --lookahead=${ROSS_LOOKAHEAD} --extramem=${em}" ;;
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

# Launcher prefix array in LAUNCH[]. $1=np $2=nodes $3=ppn.
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

# Application argv in APPCMD[]. $1=impl $2=variant $3=na $4=total $5=planes.
build_appcmd() {
        local impl="$1" variant="$2" na="$3" total="$4" planes="$5"
        local exe; exe="$(exe_for "${impl}")"
        APPCMD=()
        if is_ross "${impl}"; then
                # shellcheck disable=SC2206
                local sflags=( $(ross_sync_flags "${impl}" "${planes}") )
                APPCMD=("${exe}" "${sflags[@]}"
                        "--nairports=${total}" "--end=${ROSS_END}" "--nplanes=${planes}")
        else
                local flags extra
                flags="$(variant_flags "${impl}" "${variant}")"
                extra="$(impl_extra_flags "${impl}" "${na}" "${planes}")"
                # shellcheck disable=SC2206
                APPCMD=("${exe}" --na "${na}" --time "${TIMESTEPS}"
                        --planes "${planes}" --seed "${SEED}" ${flags} ${extra} ${VERBOSITY})
        fi
}

# --------------------------------------------------------------------------
# Header patching + rebuild (darray/graph only)
# --------------------------------------------------------------------------
HDR_BACKUPS=()
backup_headers() {
        local h
        for h in "${DARRAY_HDR}" "${DGRAPH_HDR}"; do
                [[ -f "${h}" ]] || continue
                cp -p "${h}" "${h}.planes.bak" || { echo "ERROR: cannot back up ${h}" >&2; exit 1; }
                HDR_BACKUPS+=("${h}")
        done
}
restore_headers() {
        local h
        for h in ${HDR_BACKUPS[@]+"${HDR_BACKUPS[@]}"}; do
                [[ -f "${h}.planes.bak" ]] || continue
                mv -f "${h}.planes.bak" "${h}"
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

# na is FIXED for the whole sweep (ranks fixed), so set capacity ONCE.
CAP_DONE=0
set_capacity_once() {
        local na="$1" gate_cap edge_cap impl dir rc
        needs_capacity_build || return 0
        (( CAP_DONE )) && return 0
        edge_cap=${CAP_PER_EDGE}
        gate_cap=$(( CAP_PER_EDGE * na ))
        echo "  capacity: EDGE_GATE_CAPACITY=${edge_cap}  GATE_CAPACITY=${gate_cap}  queue --cap=${gate_cap}  (na=${na})"
        if [[ ${DRY_RUN} -eq 1 ]]; then
                echo "      would patch darray/graph headers and rebuild (once)"
                CAP_DONE=1; return 0
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
        CAP_DONE=1
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

# Discernible planes-sweep stamp.
STAMP="planes_sweep_np${NP}_$(date +%Y%m%d-%H%M%S)"
OUTDIR="${OUTROOT}/${STAMP}"
RAWDIR="${OUTDIR}/raw"
DUMPROOT="${OUTDIR}/dumps"
HFDIR="${OUTDIR}/hostfiles"
CSV="${OUTDIR}/runs.csv"
SUMMARY="${OUTDIR}/summary.txt"
mkdir -p "${RAWDIR}" "${DUMPROOT}" "${HFDIR}"

# Sizing at the fixed rung (weak-mode style: na fixed, total grows with np).
NA="${NA_PER_RANK}"
TOTAL=$(( NA * NP ))
GRID="$(isqrt_exact "${TOTAL}")"

{
        echo "ANALYSIS:     PLANES sweep at FIXED ranks=${NP}"
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
        echo "fixed rung:   np=${NP}  (${NODES} nodes x ${PPN} ppn)"
        echo "fixed na:     ${NA}     total_airports=${TOTAL} (${GRID}x${GRID})"
        echo "SWEPT planes: ${PLANES_LIST}"
        echo "darray exe:   ${DARRAY_EXE}"
        echo "dgraph exe:   ${DGRAPH_EXE}"
        echo "dqueue exe:   ${DQUEUE_EXE}"
        echo "dqueue-batch: ${DQUEUE_BATCH_EXE}"
        echo "ross exe:     ${ROSS_EXE}"
        echo "impls:        ${IMPLS}    variants: ${VARIANTS}"
        echo "cap/edge:     ${CAP_PER_EDGE}  (GATE/queue cap = cap/edge * na; DSTM only)"
        echo "timesteps:    ${TIMESTEPS}    reps: ${REPS}"
        echo "ross end vt:  ${ROSS_END}  (= ${TIMESTEPS} * mean_dep=${ROSS_MEAN_DEPARTURE})"
        echo "ross hops:    net_events / ${ROSS_EVENTS_PER_HOP}"
        echo "ross opt:     --synch=3 --extramem=${ROSS_EXTRAMEM}"
        echo "ross cons:    --synch=2 --lookahead=${ROSS_LOOKAHEAD}"
        echo "dumps under:  ${DUMPROOT}"
        echo "timeout:      ${TIMEOUT_S}s"
} | tee "${OUTDIR}/config.txt"
echo

# Validate the fixed rung up front.
if [[ "${GRID}" == "0" ]]; then
        echo "ERROR: total_airports=${TOTAL} (na=${NA} * np=${NP}) is not a perfect square." >&2
        exit 1
fi
if (( GRID < 2 )); then
        echo "ERROR: grid ${GRID} < 2; mains reject it." >&2
        exit 1
fi

# CSV schema identical to run_all.sh so the same plot.py / tooling works.
echo "impl,variant,mode,ranks,nodes,ppn,placement,na,total,grid,planes,timesteps,rep,status,\
elapsed,tsps,departures,landings,overflows,misroutes,conservation,expected,cons_ok,\
act_status,dumps_found,calls_total,calls_pertick_mean,calls_pertick_max,\
p2p_pertick_mean,coll_pertick_mean,completion_pertick_mean,\
isend_pt,irecv_pt,send_pt,recv_pt,wait_pt,waitall_pt,test_pt,\
barrier_pt,reduce_pt,allreduce_pt,bcast_pt,gather_pt,allgather_pt,alltoall_pt,\
bytes_sent_total,bytes_recvd_total,bytes_sent_pertick_mean" > "${CSV}"

ACT_NFIELDS=25
act_blank() { local i out="-1"; for (( i=2; i<=ACT_NFIELDS; i++ )); do out="${out},-1"; done; echo "${out}"; }

# ==========================================================================
# Aggregate per-rank dumps -> comma-joined block (as in run_all.sh)
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
                p2p_pt=(p2p/rk)/tk; coll_pt=(coll/rk)/tk; compl_pt=(completion/rk)/tk;
                isend_pt=(isend/rk)/tk; irecv_pt=(irecv/rk)/tk; send_pt=(send/rk)/tk; recv_pt=(recv/rk)/tk;
                wait_pt=(wait/rk)/tk; waitall_pt=(waitall/rk)/tk; test_pt=(testc/rk)/tk;
                barrier_pt=(barrier/rk)/tk; reduce_pt=(reduce/rk)/tk; allreduce_pt=(allreduce/rk)/tk;
                bcast_pt=(bcast/rk)/tk; gather_pt=(gather/rk)/tk; allgather_pt=(allgather/rk)/tk; alltoall_pt=(alltoall/rk)/tk;
                bsent_pt=(bsent/rk)/tk;
                printf "%d,%.0f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.0f,%.0f,%.2f",
                       seen, calls_total, calls_pt_mean, calls_pt_max, p2p_pt, coll_pt, compl_pt,
                       isend_pt, irecv_pt, send_pt, recv_pt, wait_pt, waitall_pt, test_pt,
                       barrier_pt, reduce_pt, allreduce_pt, bcast_pt, gather_pt, allgather_pt, alltoall_pt,
                       bsent, brecvd, bsent_pt;
        }' "${dumpdir}"/mpi_counts_rank*.txt 2>/dev/null
}

# ==========================================================================
# Parse timing block -> globals T_*  (as in run_all.sh)
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
        if [[ -z "${T_elapsed}" ]]; then T_elapsed="${wall}"; T_status_note="parse_fail"; fi
        : "${T_tsps:=nan}" "${T_dep:=-1}" "${T_land:=-1}" "${T_ovf:=-1}"
        : "${T_mis:=-1}" "${T_cons:=-1}" "${T_exp:=-1}" "${T_consok:=n/a}"
}

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
        local rc="$1" log="$2" st="ok"
        if   [[ ${rc} -eq 124 || ${rc} -eq 137 ]]; then st="timeout"
        elif [[ ${rc} -ne 0 ]];                   then st="fail(${rc})"
        fi
        if grep -q "orted: command not found\|ORTE was unable to reliably start" "${log}" 2>/dev/null; then
                st="launch_err"
        fi
        echo "${st}"
}

# ==========================================================================
# One config (fixed np, one planes value) -> one CSV row
# ==========================================================================
run_one() {
        local impl="$1" variant="$2" planes="$3" rep="$4"
        # mode label "planes" so rows are clearly from this analysis, and so
        # existing plot.py (which filters on mode) can target --mode planes.
        local mode="planes"
        local tag="${impl}-${variant}-pl${planes}-np${NP}-n${NODES}x${PPN}-na${NA}-r${rep}"

        build_launch "${NP}" "${NODES}" "${PPN}"
        build_appcmd "${impl}" "${variant}" "${NA}" "${TOTAL}" "${planes}"

        printf '  %-9s %-10s planes=%-6s np=%-4s na=%-6s total=%-7s rep=%s ... ' \
               "${impl}" "${variant}" "${planes}" "${NP}" "${NA}" "${TOTAL}" "${rep}"

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
        local act_block; act_block="$(act_blank)"

        # ---- Pass 1: timing ----
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

        # ---- Pass 2: activity ----
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
                (( DO_TIMING == 0 )) && status="${act_status}"
                if [[ "${act_status}" == ok ]]; then
                        local waited=0 nf=0
                        while (( waited < 30 )); do
                                nf=$(find "${dumpdir}" -maxdepth 1 \
                                     -name 'mpi_counts_rank*.txt' 2>/dev/null | wc -l)
                                (( nf >= NP )) && break
                                sleep 1; (( waited++ ))
                        done
                        local agg; agg="$(aggregate_dumps "${dumpdir}" "${NP}" "${TIMESTEPS}")"
                        if [[ -z "${agg}" || "${agg}" == "NODUMPS" ]]; then
                                act_status="no_dumps"; act_block="$(act_blank)"
                        else
                                act_block="${agg}"
                        fi
                fi
                [[ "${act_status}" != ok ]] && act_block="$(act_blank)"
                unset MPI_WRAP_OUTDIR
        fi

        echo "${impl},${variant},${mode},${NP},${NODES},${PPN},${PLACEMENT},${NA},${TOTAL},${GRID},${planes},${TIMESTEPS},${rep},${status},${elapsed},${tsps},${dep},${land},${ovf},${mis},${cons},${exp},${cons_ok},${act_status},${act_block}" >> "${CSV}"

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
# Sweep: fixed rung, loop over planes
# ==========================================================================
# Guard: the fixed rung must fit the allocation.
if (( PPN > MAX_PPN )); then
        echo "ERROR: rung ppn ${PPN} exceeds cap ${MAX_PPN}." >&2; exit 1
fi
if (( NODES > AVAIL_NODES )); then
        if [[ ${DRY_RUN} -eq 1 ]]; then
                echo "WARNING (dry-run): rung needs ${NODES} nodes, have ${AVAIL_NODES}."
        else
                echo "ERROR: rung needs ${NODES} nodes, have ${AVAIL_NODES}." >&2; exit 1
        fi
fi
if (( NODES * PPN < NP )); then
        echo "ERROR: ${NODES}x${PPN} cannot host ${NP} ranks." >&2; exit 1
fi

if ! set_capacity_once "${NA}"; then
        echo "ERROR: capacity build failed for na=${NA}." >&2
        exit 1
fi

echo "===== planes sweep @ np=${NP} (na=${NA}, total=${TOTAL}) ====="
for planes in ${PLANES_LIST}; do
        echo
        echo "---- planes=${planes} ----"
        for impl in ${IMPLS}; do
                vlist="${VARIANTS}"
                [[ "${impl}" != graph ]] && vlist="base"
                for variant in ${vlist}; do
                        for (( rep = 1; rep <= REPS; rep++ )); do
                                run_one "${impl}" "${variant}" "${planes}" "${rep}"
                        done
                done
        done
done
echo

[[ ${DRY_RUN} -eq 1 ]] && { echo "dry run complete"; exit 0; }

# ==========================================================================
# Summary -- per-planes throughput + activity (median over reps)
# CSV col refs: 11 planes  14 status  16 tsps  24 act_status
#   27 calls_pertick_mean  29 p2p_pertick_mean  30 coll_pertick_mean
#   48 bytes_sent_pertick_mean
# ==========================================================================
planes_list_sorted() { awk -F, 'NR>1 { print $11 }' "${CSV}" | sort -n -u; }

med_timing() {   # $1=impl $2=variant $3=planes $4=col
        awk -F, -v i="$1" -v v="$2" -v p="$3" -v c="$4" \
            '$1==i && $2==v && $11==p && $14=="ok" { print $c }' "${CSV}" | median
}
med_act() {      # $1=impl $2=variant $3=planes $4=col
        awk -F, -v i="$1" -v v="$2" -v p="$3" -v c="$4" \
            '$1==i && $2==v && $11==p && $24=="ok" { print $c }' "${CSV}" | median
}

{
        cat "${OUTDIR}/config.txt"
        echo
        echo "NOTE: throughput metric = committed plane-hops / second."
        echo "  DSTM hop = one 'Total departures:' entry."
        echo "  ROSS hop = Net Events Processed / ${ROSS_EVENTS_PER_HOP}."
        echo "NOTE: ranks are FIXED at ${NP}; the x-axis of this analysis is PLANES."
        echo "NOTE: activity numbers are per-rank-per-tick means from mpi_wrap.c."
        echo

        echo "===== throughput vs planes @ np=${NP} (median of ${REPS} reps) ====="
        printf '%-10s %-10s %8s %13s %13s\n' impl variant planes "elapsed(s)" "hops/s"
        printf -- '--------------------------------------------------------------\n'
        for impl in ${IMPLS}; do
                vlist="${VARIANTS}"; [[ "${impl}" != graph ]] && vlist="base"
                for variant in ${vlist}; do
                        for p in $(planes_list_sorted); do
                                e=$(med_timing "${impl}" "${variant}" "${p}" 15)
                                [[ "${e}" == "nan" ]] && continue
                                ts=$(med_timing "${impl}" "${variant}" "${p}" 16)
                                awk -v i="${impl}" -v v="${variant}" -v p="${p}" \
                                    -v e="${e}" -v ts="${ts}" 'BEGIN {
                                        printf "%-10s %-10s %8s %13.4f %13.2f\n", i, v, p, e, ts }'
                        done
                        echo
                done
        done

        echo "===== MPI activity vs planes @ np=${NP} (median of ${REPS} reps, base) ====="
        printf '%-10s %8s %14s %12s %12s %14s\n' \
               impl planes "calls/t mean" "p2p/t" "coll/t" "bytes_sent/t"
        printf -- '----------------------------------------------------------------------------\n'
        for impl in ${IMPLS}; do
                for p in $(planes_list_sorted); do
                        cm=$(med_act "${impl}" base "${p}" 27)
                        [[ "${cm}" == "nan" ]] && continue
                        p2=$(med_act "${impl}" base "${p}" 29)
                        co=$(med_act "${impl}" base "${p}" 30)
                        by=$(med_act "${impl}" base "${p}" 48)
                        printf '%-10s %8s %14s %12s %12s %14s\n' "${impl}" "${p}" "${cm}" "${p2}" "${co}" "${by}"
                done
                echo
        done

        total_runs=$(( $(wc -l < "${CSV}") - 1 ))
        bad=$(awk -F, 'NR>1 && $14!="ok"' "${CSV}" | wc -l)
        nod=$(awk -F, 'NR>1 && $24=="no_dumps"' "${CSV}" | wc -l)
        echo "runs recorded: ${total_runs}   timing non-ok: ${bad}   no_dumps: ${nod}"
        if (( bad > 0 )); then
                echo
                echo "non-ok timing runs:"
                awk -F, 'NR>1 && $14!="ok" { printf "  %-9s %-10s planes=%-6s -> %s\n", $1,$2,$11,$14 }' "${CSV}"
        fi
} | tee "${SUMMARY}"

echo
echo "csv:     ${CSV}"
echo "summary: ${SUMMARY}"
echo "dumps:   ${DUMPROOT}"
echo "logs:    ${RAWDIR}"
