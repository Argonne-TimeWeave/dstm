#!/usr/bin/env bash
#PBS -N airport_mpi_activity
#PBS -l select=8:ncpus=64:mpiprocs=32
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o airport_mpi_activity.log
#
# ---- set your queue/project before submitting -----------------------------
##PBS -q <queue>
##PBS -A <project>
#
# run_mpi_activity.sh -- three-way MPI-activity comparison of the airport_sim
#   backends (darray / graph / queue), capturing PMPI call counts and bytes
#   from mpi_wrap.c instead of runtime.
#
# PREREQUISITE
#   All three ./airports binaries must be LINKED WITH mpi_wrap.o so each rank
#   dumps ${MPI_WRAP_OUTDIR}/mpi_counts_rank<R>.txt at MPI_Finalize. Build e.g.
#       mpicc -c -fPIC examples/mpi_wrap.c -o examples/mpi_wrap.o
#   then add mpi_wrap.o to each backend's link line and rebuild. This script
#   verifies the dumps appear and fails loudly if they do not.
#
# WHAT IT MEASURES  (per mpi_wrap.c fields)
#   point-to-point : isend irecv send recv
#   completion     : wait waitall test
#   collectives    : barrier reduce allreduce bcast gather allgather alltoall
#   volume         : bytes_sent bytes_recvd
#
#   For each run it aggregates the per-rank dumps into:
#     - total over all ranks
#     - mean per rank         (total / ranks)
#     - max over ranks        (busiest rank -- bounds the tick)
#     - mean per rank per tick (mean / timesteps)
#     - max  per rank per tick (max  / timesteps)
#   The per-rank-per-tick figures are the ones to plot, matching the
#   MPI-activity figures in the workshop paper.
#
# LAUNCH RECIPE  -- identical to run_scaling.sh, plus -x MPI_WRAP_OUTDIR so
#   every rank writes its dump into this run's private directory (not /tmp,
#   which would collide across ranks/runs/nodes on a shared filesystem).
#
# USAGE
#   qsub run_mpi_activity.sh
#   ./run_mpi_activity.sh --dry-run
#   ./run_mpi_activity.sh --mode weak --impl all --reps 1
#   ./run_mpi_activity.sh --ladder "16:1:16"
#
# OUTPUT
#   results/<stamp>/raw/<tag>.log              stdout+stderr per run
#   results/<stamp>/dumps/<tag>/mpi_counts_rank*.txt   raw per-rank dumps
#   results/<stamp>/mpi_runs.csv               one row per run (aggregated)
#   results/<stamp>/mpi_summary.txt            per-rank-per-tick tables
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

# ==========================================================================
# Defaults
# ==========================================================================
MPIRUN="${MPIRUN:-mpirun}"
MPI_FLAVOR="${MPI_FLAVOR:-auto}"
MPI_EXTRA="${MPI_EXTRA:-}"
MPI_PREFIX="${MPI_PREFIX:-}"
# MPI_WRAP_OUTDIR is appended automatically; include it in forwarded vars.
MPI_ENV_VARS="${MPI_ENV_VARS:-LD_LIBRARY_PATH PATH}"
MAX_PPN="${MAX_PPN:-32}"
MAX_NODES="${MAX_NODES:-8}"
PLACEMENT="${PLACEMENT:-pack}"

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
IMPLS="darray graph queue"
VARIANTS="base"
VERBOSITY="--quiet"
DRY_RUN=0
OUTROOT="${OUTROOT:-${SCRIPT_DIR}/results_mpi}"

# Capacity (fairness) -- same policy as run_scaling.sh
CAP_PER_EDGE="${CAP_PER_EDGE:-8}"
DARRAY_HDR="${DARRAY_HDR:-${ROOT}/simple_dstm/airport.h}"
DGRAPH_HDR="${DGRAPH_HDR:-${ROOT}/graph_dstm/airport.h}"
DARRAY_DIR="$(dirname "${DARRAY_EXE}")"
DGRAPH_DIR="$(dirname "${DGRAPH_EXE}")"
MAKE_CMD="${MAKE_CMD:-make}"
MAKE_FLAGS="${MAKE_FLAGS:-}"
MAKE_TARGET="${MAKE_TARGET:-}"
MAKE_NO_AUTOREGEN="${MAKE_NO_AUTOREGEN:-AUTOCONF=true AUTOHEADER=true AUTOMAKE=true ACLOCAL=true MAINTAINER_MODE=no}"

usage() {
        cat <<EOF
run_mpi_activity.sh -- capture MPI call counts/bytes (via mpi_wrap.c) for the
three airport_sim backends, reduced to per-rank-per-tick figures.

options:
  --mode {weak|strong|both}         default: both
  --impl {darray|graph|queue|both|all}  default: all ("both"=darray graph)
  --variants "a b c"                graph only
  --ladder / --weak-ladder / --strong-ladder "np:nodes:ppn ..."
  --placement {pack|spread}
  --max-ppn N / --max-nodes N
  --na-per-rank N / --total N / --planes P / --time T / --seed S / --reps R
  --timeout S                       per-run seconds, 0=off (default ${TIMEOUT_S})
  --mpirun CMD / --mpi-flavor {auto|openmpi|mpich}
  --mpi-prefix DIR / --mpi-env "A B C" / --mpi-extra "ARGS"
  --cap-per-edge N                  GATE/queue cap = N*na  (default ${CAP_PER_EDGE})
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
                both) IMPLS="darray graph" ;;
                all)  IMPLS="darray graph queue" ;;
                *)    IMPLS="$2" ;;
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

# MPI_WRAP_OUTDIR must be forwarded to every rank (Open MPI -x). Append it
# once so the user's --mpi-env list is preserved.
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

# IMPORTANT (shared-filesystem assumption): the dump directory must be
# reachable by every node under the SAME path, so per-rank files from all
# nodes aggregate here. On Cascade/JLSE $HOME is shared, so OUTROOT under the
# script dir is fine. If you run where ranks only see node-local disk, point
# --out at a shared path, or the aggregation will see only rank-0's dumps.

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
                [[ "${_v}" == MPI_WRAP_OUTDIR ]] && continue   # passed per-run as -x VAR=VALUE
                MPI_X_FLAGS+=(-x "${_v}")
        done
fi

# ==========================================================================
# Helpers
# ==========================================================================
isqrt_exact() { awk -v n="$1" 'BEGIN { r = int(sqrt(n) + 0.5); print (r * r == n) ? r : 0 }'; }

exe_for() {
        case "$1" in
        darray) echo "${DARRAY_EXE}" ;;
        graph)  echo "${DGRAPH_EXE}" ;;
        queue)  echo "${DQUEUE_EXE}" ;;
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
        [[ "${impl}" == queue ]] && echo "--cap $(( CAP_PER_EDGE * na ))" || echo ""
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

# --------------------------------------------------------------------------
# Header patching + rebuild (darray/graph only; queue is runtime --cap)
# --------------------------------------------------------------------------
HDR_BACKUPS=()
backup_headers() {
        local h
        for h in "${DARRAY_HDR}" "${DGRAPH_HDR}"; do
                [[ -f "${h}" ]] || continue
                cp -p "${h}" "${h}.mpiact.bak" || { echo "ERROR: cannot back up ${h}" >&2; exit 1; }
                HDR_BACKUPS+=("${h}")
        done
}
restore_headers() {
        local h
        for h in ${HDR_BACKUPS[@]+"${HDR_BACKUPS[@]}"}; do
                [[ -f "${h}.mpiact.bak" ]] || continue
                mv -f "${h}.mpiact.bak" "${h}"
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
        if [[ ${DRY_RUN} -eq 1 ]]; then CUR_CAP_NA="${na}"; return 0; fi
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
                [[ ${rc} -ne 0 ]] && { echo "  BUILD FAILED (${impl}, na=${na})" >&2; return 1; }
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
        [[ -x "${exe}" ]] || { echo "ERROR: ${impl} exe missing: ${exe}" >&2; missing=1; }
done
[[ ${missing} -eq 1 && ${DRY_RUN} -eq 0 ]] && exit 1

command -v "${MPIRUN}" >/dev/null 2>&1 || { echo "ERROR: launcher '${MPIRUN}' not on PATH" >&2; [[ ${DRY_RUN} -eq 0 ]] && exit 1; }

if needs_capacity_build; then
        for _h in "${DARRAY_HDR}" "${DGRAPH_HDR}"; do
                [[ ! -w "${_h}" && ${DRY_RUN} -eq 0 ]] && { echo "ERROR: header not writable: ${_h}" >&2; exit 1; }
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
CSV="${OUTDIR}/mpi_runs.csv"
SUMMARY="${OUTDIR}/mpi_summary.txt"
mkdir -p "${RAWDIR}" "${DUMPROOT}" "${HFDIR}"

{
        echo "host:         $(hostname)"
        echo "date:         $(date -Is)"
        echo "job id:       ${COBALT_JOBID:-${PBS_JOBID:-${SLURM_JOB_ID:-interactive}}}"
        echo "alloc source: ${ALLOC_SRC}  (${#NODELIST[@]} nodes, using <= ${AVAIL_NODES})"
        echo "launcher:     ${MPIRUN} [${MPI_FLAVOR}]"
        echo "mpi prefix:   ${MPI_PREFIX:-<unset>}"
        echo "forwarded:    ${MPI_ENV_VARS}"
        echo "darray exe:   ${DARRAY_EXE}"
        echo "dgraph exe:   ${DGRAPH_EXE}"
        echo "dqueue exe:   ${DQUEUE_EXE}"
        echo "modes:        ${MODES}"
        echo "impls:        ${IMPLS}    variants: ${VARIANTS}"
        echo "na/rank:      ${NA_PER_RANK} (weak)   total: ${TOTAL_AIRPORTS} (strong)"
        echo "cap/edge:     ${CAP_PER_EDGE}"
        echo "planes:       ${PLANES}    timesteps: ${TIMESTEPS}    reps: ${REPS}"
        echo "dumps under:  ${DUMPROOT}"
} | tee "${OUTDIR}/config.txt"
echo

# CSV schema: identity + aggregated MPI activity.
# p2p   = isend+irecv+send+recv ; coll = barrier+reduce+allreduce+bcast+
#         gather+allgather+alltoall ; completion = wait+waitall+test.
# For every metric X we record: X_total, X_permax (max over ranks),
# X_pertick_mean, X_pertick_max.  Category breakdown is per-rank-per-tick mean.
echo "impl,variant,mode,ranks,nodes,ppn,na,total,timesteps,rep,status,dumps_found,\
calls_total,calls_pertick_mean,calls_pertick_max,\
p2p_pertick_mean,coll_pertick_mean,completion_pertick_mean,\
isend_pt,irecv_pt,send_pt,recv_pt,wait_pt,waitall_pt,test_pt,\
barrier_pt,reduce_pt,allreduce_pt,bcast_pt,gather_pt,allgather_pt,alltoall_pt,\
bytes_sent_total,bytes_recvd_total,bytes_sent_pertick_mean" > "${CSV}"

# ==========================================================================
# Aggregate one run's per-rank dumps -> a single CSV field block.
# Reads every mpi_counts_rank*.txt in $1; emits a comma-joined block matching
# the CSV columns from calls_total onward. $2 = ranks, $3 = timesteps.
# ==========================================================================
aggregate_dumps() {
        local dumpdir="$1" ranks="$2" ticks="$3"

        # Guard: if the glob matched nothing, bail before invoking awk so we
        # never feed awk a literal unexpanded pattern.
        local nf
        nf=$(find "${dumpdir}" -maxdepth 1 -name 'mpi_counts_rank*.txt' 2>/dev/null | wc -l)
        if [[ "${nf}" -eq 0 ]]; then
                echo "NODUMPS"
                return
        fi

        awk -v ranks="${ranks}" -v ticks="${ticks}" '
        /^rank / {
                seen++;
                cur = $2;          # current rank id for the per-rank max
        }
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

                calls_total = isend+irecv+send+recv + wait+waitall+testc + barrier+reduce+allreduce+bcast+gather+allgather+alltoall;

                cmax = 0;
                for (r in c) if (c[r] > cmax) cmax = c[r];

                tk = (ticks > 0) ? ticks : 1;
                rk = (ranks > 0) ? ranks : seen;

                calls_pt_mean = (calls_total / rk) / tk;
                calls_pt_max  = cmax / tk;

                p2p   = isend+irecv+send+recv;
                coll  = barrier+reduce+allreduce+bcast+gather+allgather+alltoall;
                completion = wait+waitall+testc;

                p2p_pt   = (p2p   / rk) / tk;
                coll_pt  = (coll  / rk) / tk;
                compl_pt = (completion / rk) / tk;

                isend_pt     = (isend/rk)/tk;      irecv_pt     = (irecv/rk)/tk;
                send_pt      = (send/rk)/tk;       recv_pt      = (recv/rk)/tk;
                wait_pt      = (wait/rk)/tk;       waitall_pt   = (waitall/rk)/tk;
                test_pt      = (testc/rk)/tk;
                barrier_pt   = (barrier/rk)/tk;    reduce_pt    = (reduce/rk)/tk;
                allreduce_pt = (allreduce/rk)/tk;  bcast_pt     = (bcast/rk)/tk;
                gather_pt    = (gather/rk)/tk;     allgather_pt = (allgather/rk)/tk;
                alltoall_pt  = (alltoall/rk)/tk;

                bsent_pt = (bsent/rk)/tk;

                printf "%d,%.0f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.0f,%.0f,%.2f", seen, calls_total, calls_pt_mean, calls_pt_max, p2p_pt, coll_pt, completion_pt, isend_pt, irecv_pt, send_pt, recv_pt, wait_pt, waitall_pt, test_pt, barrier_pt, reduce_pt, allreduce_pt, bcast_pt, gather_pt, allgather_pt, alltoall_pt, bsent, brecvd, bsent_pt;
        }
        ' "${dumpdir}"/mpi_counts_rank*.txt  2>/dev/null
}

# ==========================================================================
# One run
# ==========================================================================
run_one() {
        local impl="$1" variant="$2" mode="$3" np="$4" nodes="$5" ppn="$6"
        local na="$7" total="$8" rep="${9}"
        local exe flags extra log rc hf tag dumpdir

        exe="$(exe_for "${impl}")"
        flags="$(variant_flags "${impl}" "${variant}")"
        extra="$(impl_extra_flags "${impl}" "${na}")"
        tag="${impl}-${variant}-${mode}-np${np}-n${nodes}x${ppn}-na${na}-r${rep}"
        log="${RAWDIR}/${tag}.log"
        hf="$(hostfile_for "${nodes}")"
        dumpdir="${DUMPROOT}/${tag}"
        mkdir -p "${dumpdir}"

        local launch=()
        if [[ "${MPI_FLAVOR}" == openmpi ]]; then
                launch=("${MPIRUN}" -n "${np}" --hostfile "${hf}")
                [[ -n "${MPI_PREFIX}" ]] && launch+=(--prefix "${MPI_PREFIX}")
                (( np > 1 )) && (( np % nodes == 0 )) && launch+=(--map-by "ppr:${ppn}:node")
                launch+=("${MPI_X_FLAGS[@]}")
                launch+=(-x "MPI_WRAP_OUTDIR=${dumpdir}")
        else
                # MPICH/Hydra: forward MPI_WRAP_OUTDIR explicitly via -genv.
                launch=("${MPIRUN}" -n "${np}" -f "${hf}" -ppn "${ppn}"
                        -genv MPI_WRAP_OUTDIR "${dumpdir}")
        fi

        # shellcheck disable=SC2206
        local cmd=("${launch[@]}" ${MPI_EXTRA} "${exe}"
                   --na "${na}" --time "${TIMESTEPS}"
                   --planes "${PLANES}" --seed "${SEED}" ${flags} ${extra} ${VERBOSITY})

        printf '  %-6s %-10s np=%-4s %sx%-3s na=%-6s total=%-7s rep=%s ... ' \
               "${impl}" "${variant}" "${np}" "${nodes}" "${ppn}" "${na}" "${total}" "${rep}"

        if [[ ${DRY_RUN} -eq 1 ]]; then
                printf '\n      MPI_WRAP_OUTDIR=%s %s\n' "${dumpdir}" "${cmd[*]}"
                return 0
        fi

        # Export for Open MPI's -x pickup (it forwards the *current* value).
        export MPI_WRAP_OUTDIR="${dumpdir}"

        "${TIMEOUT_CMD[@]}" "${cmd[@]}" > "${log}" 2>&1
        rc=$?

        local status="ok"
        if   [[ ${rc} -eq 124 || ${rc} -eq 137 ]]; then status="timeout"
        elif [[ ${rc} -ne 0 ]];                   then status="fail(${rc})"
        fi
        if grep -q "orted: command not found\|ORTE was unable to reliably start" "${log}" 2>/dev/null; then
                status="launch_err"
        fi

        # Pull conservation out of the app log as a sanity gate (optional).
        local cons_ok
        cons_ok=$(awk '/^Conservation:/ { print $5 }' "${log}" | tail -1)
        : "${cons_ok:=n/a}"

        # Wait for all per-rank dumps to land (shared FS lags mpirun return).
        if [[ "${status}" == ok ]]; then
                local waited=0 nf=0
                while (( waited < 30 )); do
                        nf=$(find "${dumpdir}" -maxdepth 1 \
                             -name 'mpi_counts_rank*.txt' 2>/dev/null | wc -l)
                        (( nf >= np )) && break
                        sleep 1; (( waited++ ))
                done
        fi

        # Aggregate the per-rank dumps.
        local agg
        agg="$(aggregate_dumps "${dumpdir}" "${np}" "${TIMESTEPS}")"

        local dumps_found=0
        if [[ -z "${agg}" || "${agg}" == "NODUMPS" ]]; then
                [[ "${status}" == ok ]] && status="no_dumps"
                # Fill the metric block with -1 so the row width stays constant.
                agg="0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1"
        else
                dumps_found="${agg%%,*}"
        fi

        echo "${impl},${variant},${mode},${np},${nodes},${ppn},${na},${total},${TIMESTEPS},${rep},${status},${agg}" >> "${CSV}"

        if [[ "${status}" == ok ]]; then
                # Echo calls/tick mean (col after dumps_found,calls_total).
                local cpt
                cpt="$(echo "${agg}" | awk -F, '{ print $3 }')"
                printf 'dumps=%-4s calls/rank/tick=%-10s cons=%s\n' "${dumps_found}" "${cpt}" "${cons_ok}"
        else
                printf '%s (%s)\n' "${status}" "$(basename "${log}")"
        fi
}

# ==========================================================================
# Sweep
# ==========================================================================
ladder_for() { case "$1" in weak) echo "${WEAK_LADDER}";; strong) echo "${STRONG_LADDER}";; esac; }

for mode in ${MODES}; do
        echo "===== ${mode} MPI-activity sweep ====="
        for rung in $(ladder_for "${mode}"); do
                np="${rung%%:*}"; rest="${rung#*:}"; nodes="${rest%%:*}"; ppn="${rest##*:}"

                if [[ "${PLACEMENT}" == spread ]]; then
                        nodes=${AVAIL_NODES}; (( nodes > np )) && nodes=${np}
                        ppn=$(( (np + nodes - 1) / nodes ))
                fi
                (( ppn > MAX_PPN ))    && { echo "  skip np=${np}: ppn ${ppn} > cap ${MAX_PPN}"; continue; }
                (( nodes > AVAIL_NODES )) && { echo "  skip np=${np}: needs ${nodes} nodes, have ${AVAIL_NODES}"; continue; }
                (( nodes * ppn < np )) && { echo "  skip np=${np}: ${nodes}x${ppn} cannot host ${np}"; continue; }

                if [[ "${mode}" == weak ]]; then
                        na=${NA_PER_RANK}; total=$(( na * np ))
                else
                        (( TOTAL_AIRPORTS % np != 0 )) && { echo "  skip np=${np}: ${TOTAL_AIRPORTS} not divisible"; continue; }
                        na=$(( TOTAL_AIRPORTS / np )); total=${TOTAL_AIRPORTS}
                fi
                grid=$(isqrt_exact "${total}")
                [[ "${grid}" == "0" ]] && { echo "  skip np=${np}: total ${total} not a perfect square"; continue; }
                (( grid < 2 ))         && { echo "  skip np=${np}: grid ${grid} < 2"; continue; }

                set_capacity "${na}" || { echo "  skip np=${np}: capacity build failed"; continue; }

                for impl in ${IMPLS}; do
                        vlist="${VARIANTS}"; [[ "${impl}" != graph ]] && vlist="base"
                        for variant in ${vlist}; do
                                for (( rep = 1; rep <= REPS; rep++ )); do
                                        run_one "${impl}" "${variant}" "${mode}" \
                                                "${np}" "${nodes}" "${ppn}" \
                                                "${na}" "${total}" "${rep}"
                                done
                        done
                done
        done
        echo
done

[[ ${DRY_RUN} -eq 1 ]] && { echo "dry run complete"; exit 0; }

# ==========================================================================
# Summary -- per-rank-per-tick, median over reps
# ==========================================================================
np_list_for() { awk -F, -v m="$1" 'NR>1 && $3==m { print $4 }' "${CSV}" | sort -n -u; }
median() {
        sort -g | awk '{a[NR]=$1} END {
                if(NR==0){print "nan";exit}
                if(NR%2){printf "%.4f\n",a[(NR+1)/2]}
                else{printf "%.4f\n",(a[NR/2]+a[NR/2+1])/2}}'
}

# Pull median of CSV column $5 for (mode,impl,variant,np), ok rows only.
med_col() {
        local mode="$1" impl="$2" variant="$3" np="$4" col="$5"
        awk -F, -v m="${mode}" -v i="${impl}" -v v="${variant}" -v n="${np}" -v c="${col}" \
            '$3==m && $1==i && $2==v && $4==n && $11=="ok" { print $c }' "${CSV}" | median
}

summarize() {
        local mode="$1" keys key impl variant np
        keys=$(awk -F, -v m="${mode}" 'NR>1 && $3==m { print $1 "|" $2 }' "${CSV}" | sort -u)
        [[ -z "${keys}" ]] && return

        echo "===== ${mode}: MPI calls per rank per tick (median of ${REPS}) ====="
        printf '%-8s %-10s %6s %7s %12s %12s %10s %10s %12s\n' \
               impl variant ranks na "calls/t mean" "calls/t max" "p2p/t" "coll/t" "bytes_sent/t"
        printf -- '--------------------------------------------------------------------------------------------------\n'
        for key in ${keys}; do
                impl="${key%%|*}"; variant="${key##*|}"
                for np in $(np_list_for "${mode}"); do
                        local cm cx p2 co by na
                        cm=$(med_col "${mode}" "${impl}" "${variant}" "${np}" 14)  # calls_pertick_mean
                        [[ "${cm}" == "nan" ]] && continue
                        cx=$(med_col "${mode}" "${impl}" "${variant}" "${np}" 15)  # calls_pertick_max
                        p2=$(med_col "${mode}" "${impl}" "${variant}" "${np}" 16)  # p2p_pertick_mean
                        co=$(med_col "${mode}" "${impl}" "${variant}" "${np}" 17)  # coll_pertick_mean
                        by=$(med_col "${mode}" "${impl}" "${variant}" "${np}" 33)  # bytes_sent_pertick_mean
                        na=$(awk -F, -v m="${mode}" -v i="${impl}" -v v="${variant}" -v n="${np}" \
                                '$3==m && $1==i && $2==v && $4==n { print $7; exit }' "${CSV}")
                        printf '%-8s %-10s %6s %7s %12s %12s %10s %10s %12s\n' \
                               "${impl}" "${variant}" "${np}" "${na}" "${cm}" "${cx}" "${p2}" "${co}" "${by}"
                done
                echo
        done

        echo "===== ${mode}: per-category calls per rank per tick (median of ${REPS}, base variant) ====="
        printf '%-8s %6s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n' \
               impl ranks isend irecv send recv wait waitall test barrier reduce allred bcast gather allgat
        printf -- '-----------------------------------------------------------------------------------------------------------------------------------\n'
        for impl in ${IMPLS}; do
                for np in $(np_list_for "${mode}"); do
                        local v1
                        v1=$(med_col "${mode}" "${impl}" base "${np}" 18)
                        [[ "${v1}" == "nan" ]] && continue
                        printf '%-8s %6s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n' \
                               "${impl}" "${np}" \
                               "$(med_col "${mode}" "${impl}" base "${np}" 18)" \
                               "$(med_col "${mode}" "${impl}" base "${np}" 19)" \
                               "$(med_col "${mode}" "${impl}" base "${np}" 20)" \
                               "$(med_col "${mode}" "${impl}" base "${np}" 21)" \
                               "$(med_col "${mode}" "${impl}" base "${np}" 22)" \
                               "$(med_col "${mode}" "${impl}" base "${np}" 23)" \
                               "$(med_col "${mode}" "${impl}" base "${np}" 24)" \
                               "$(med_col "${mode}" "${impl}" base "${np}" 25)" \
                               "$(med_col "${mode}" "${impl}" base "${np}" 26)" \
                               "$(med_col "${mode}" "${impl}" base "${np}" 27)" \
                               "$(med_col "${mode}" "${impl}" base "${np}" 28)" \
                               "$(med_col "${mode}" "${impl}" base "${np}" 29)" \
                               "$(med_col "${mode}" "${impl}" base "${np}" 30)"
                done
                echo
        done
}

{
        cat "${OUTDIR}/config.txt"
        echo
        for mode in ${MODES}; do summarize "${mode}"; done

        total_runs=$(( $(wc -l < "${CSV}") - 1 ))
        bad=$(awk -F, 'NR>1 && $11!="ok"' "${CSV}" | wc -l)
        nod=$(awk -F, 'NR>1 && $11=="no_dumps"' "${CSV}" | wc -l)
        echo "runs recorded: ${total_runs}   non-ok: ${bad}   no_dumps: ${nod}"
        if (( nod > 0 )); then
                echo
                echo "NOTE: ${nod} run(s) produced NO per-rank dumps. Either the binary was"
                echo "      not linked with mpi_wrap.o, or MPI_WRAP_OUTDIR did not reach the"
                echo "      ranks (check -x/-genv forwarding and that the path is shared)."
        fi
        if (( bad > 0 )); then
                echo
                echo "non-ok runs:"
                awk -F, 'NR>1 && $11!="ok" { printf "  %-6s %-10s %-6s np=%-4s -> %s\n", $1,$2,$3,$4,$11 }' "${CSV}"
        fi
} | tee "${SUMMARY}"

echo
echo "csv:     ${CSV}"
echo "summary: ${SUMMARY}"
echo "dumps:   ${DUMPROOT}"
echo "logs:    ${RAWDIR}"
