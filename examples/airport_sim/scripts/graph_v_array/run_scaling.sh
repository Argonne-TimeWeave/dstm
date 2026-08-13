#!/usr/bin/env bash
#PBS -N airport_scaling
#PBS -l select=8:ncpus=64:mpiprocs=32
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o airport_scaling.log
#
# ---- set your queue/project before submitting -----------------------------
##PBS -q <queue>
##PBS -A <project>
#
# run_scaling.sh -- 8-node scaling comparison of the two airport_sim backends
#                   (dstm_darray "simple_dstm" vs dstm_dgraph "graph_dstm").
#
# Both executables are named ./airports and live in their own directories:
#
#   airport_sim/
#     simple_dstm/airports      <- darray backend
#     graph_dstm/airports       <- dgraph backend
#     scripts/graph_v_array/run_scaling.sh
#
# LAUNCH RECIPE (matches ROSS/weak_scale.sh, which is known to work here)
#   mpirun -n P --hostfile $COBALT_NODEFILE --prefix $MPI_PREFIX \
#          --map-by ppr:PPN:node -x LD_LIBRARY_PATH -x PATH ./airports ...
#   The --prefix / -x pair is what keeps orted findable on the compute nodes;
#   without it Open MPI dies with "bash: orted: command not found".
#
# TOPOLOGY LADDER
#   Up to 8 nodes x 32 ranks/node = 256 ranks. Each rung is "np:nodes:ppn";
#   rungs needing more nodes than the allocation provides are skipped.
#
# HARD CONSTRAINT (both mains)
#   na * num_ranks must be a perfect square; the dgraph main additionally
#   requires grid >= 2. Every point is validated before launch.
#     weak   : na fixed  -> np must itself be a perfect square (1,4,16,64,256)
#     strong : total fixed -> np need only divide total
#
# USAGE
#   qsub -n 8 -t 180 ./run_scaling.sh        # Cobalt
#   qsub run_scaling.sh                      # PBS Pro (directives above)
#   ./run_scaling.sh --dry-run               # print command lines, no MPI
#   ./run_scaling.sh --mode strong --reps 5
#   ./run_scaling.sh --ladder "16:1:16 16:4:4"   # intra- vs inter-node
#
# OUTPUT
#   results/<stamp>/raw/*.log    full stdout+stderr per run
#   results/<stamp>/runs.csv     one row per run
#   results/<stamp>/summary.txt  median-of-reps tables + head-to-head
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
MPI_EXTRA="${MPI_EXTRA:-}"              # site-specific extras
MPI_PREFIX="${MPI_PREFIX:-}"            # auto-derived from `which mpirun`
# Environment forwarded to every rank (Open MPI -x). Add site vars here.
MPI_ENV_VARS="${MPI_ENV_VARS:-LD_LIBRARY_PATH PATH}"
MAX_PPN="${MAX_PPN:-32}"                # hard cap: 32 ranks per node
MAX_NODES="${MAX_NODES:-8}"
PLACEMENT="${PLACEMENT:-pack}"          # pack | spread

# np:nodes:ppn -- np must be a perfect square for weak scaling
WEAK_LADDER="${WEAK_LADDER:-4:1:4 16:1:16 64:2:32 256:8:32}"
# strong scaling only needs np | total
STRONG_LADDER="${STRONG_LADDER:-4:1:4 16:1:16 32:1:32 64:2:32 128:4:32 256:8:32}"

NA_PER_RANK="${NA_PER_RANK:-64}"        # weak: airports/rank (keep it square)
TOTAL_AIRPORTS="${TOTAL_AIRPORTS:-4096}" # strong: fixed size, 64x64 torus
PLANES="${PLANES:-4}"
TIMESTEPS="${TIMESTEPS:-256}"
SEED="${SEED:-42}"
REPS="${REPS:-3}"
TIMEOUT_S="${TIMEOUT_S:-600}"           # per-run guard; 0 disables
MODES="weak strong"
IMPLS="darray graph"
VARIANTS="base"                         # base | congestion | novstate (graph)
VERBOSITY="--quiet"
DRY_RUN=0
OUTROOT="${OUTROOT:-${SCRIPT_DIR}/results}"

# ==========================================================================
# Gate capacity (fairness)
#
#   simple_dstm keeps ONE gate_slot_t per neighbor *rank*, shared by every
#   local airport that sends that way. graph_dstm keeps one per *edge*, i.e.
#   per (airport, direction). To give both the same aggregate headroom per
#   rank pair the two macros must be tied together by na:
#
#       EDGE_GATE_CAPACITY = CAP_PER_EDGE
#       GATE_CAPACITY      = CAP_PER_EDGE * na
#
#   na changes per rung in strong mode, so the headers are rewritten and the
#   binaries rebuilt before each rung. Originals are restored on exit.
# ==========================================================================
CAP_PER_EDGE="${CAP_PER_EDGE:-8}"
DARRAY_HDR="${DARRAY_HDR:-${ROOT}/simple_dstm/airport.h}"
DGRAPH_HDR="${DGRAPH_HDR:-${ROOT}/graph_dstm/airport.h}"
DARRAY_DIR="$(dirname "${DARRAY_EXE}")"
DGRAPH_DIR="$(dirname "${DGRAPH_EXE}")"
MAKE_CMD="${MAKE_CMD:-make}"
MAKE_FLAGS="${MAKE_FLAGS:--B}"          # -B: force, headers may not be tracked
MAKE_TARGET="${MAKE_TARGET:-}"

usage() {
        sed -n '7,52p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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
  --mpirun CMD                 launcher                  (default ${MPIRUN})
  --mpi-flavor {auto|openmpi|mpich}
  --mpi-prefix DIR             Open MPI install root for --prefix
  --mpi-env "A B C"            vars to forward with -x   (default: ${MPI_ENV_VARS})
  --mpi-extra "ARGS"           extra launcher args
  --cap-per-edge N             flights per airport per direction; sets
                               EDGE_GATE_CAPACITY=N and GATE_CAPACITY=N*na,
                               rebuilding both before each rung
                                                         (default ${CAP_PER_EDGE})
  --make CMD                   build command             (default ${MAKE_CMD})
  --make-flags "ARGS"          build flags               (default ${MAKE_FLAGS})
  --make-target NAME           build target              (default: <default>)
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
#   Prefer the scheduler's own nodefile and hand it to mpirun verbatim --
#   that is what the known-good ROSS script does. Subset hostfiles are only
#   generated when a rung asks for fewer nodes than we hold.
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
#   MPI_PREFIX mirrors weak_scale.sh: two dirnames up from `which mpirun`,
#   i.e. <root>/bin/mpirun -> <root>. Passed as --prefix so orted resolves
#   on nodes whose login shell lacks the MPI module.
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

# Open MPI -x flags, built once.
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
                        if (NR % 2)  { printf "%.6f\n", a[(NR + 1) / 2] }
                        else         { printf "%.6f\n", (a[NR/2] + a[NR/2 + 1]) / 2 }
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

# Echo the hostfile to use for a given node count. The scheduler's own file
# is used unchanged when the rung wants every node we hold; otherwise a
# truncated copy (bare hostnames, same format) is cached under hostfiles/.
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

backup_headers() {
        local h
        for h in "${DARRAY_HDR}" "${DGRAPH_HDR}"; do
                [[ -f "${h}" ]] || continue
                cp -p "${h}" "${h}.scaling.bak" || {
                        echo "ERROR: cannot back up ${h}" >&2
                        exit 1
                }
                HDR_BACKUPS+=("${h}")
        done
}

restore_headers() {
        local h
        for h in ${HDR_BACKUPS[@]+"${HDR_BACKUPS[@]}"}; do
                [[ -f "${h}.scaling.bak" ]] || continue
                mv -f "${h}.scaling.bak" "${h}"
        done
}
trap restore_headers EXIT INT TERM

# Rewrite one "#define NAME value" in place. Anchored on the macro name with
# a trailing space so GATE_CAPACITY never matches EDGE_GATE_CAPACITY.
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
set_capacity() {
        local na="$1" gate_cap edge_cap impl dir rc

        edge_cap=${CAP_PER_EDGE}
        gate_cap=$(( CAP_PER_EDGE * na ))

        if [[ "${CUR_CAP_NA}" == "${na}" ]]; then
                return 0
        fi

        echo "  capacity: EDGE_GATE_CAPACITY=${edge_cap}  GATE_CAPACITY=${gate_cap}  (na=${na})"

        if [[ ${DRY_RUN} -eq 1 ]]; then
                echo "      would patch ${DGRAPH_HDR} and ${DARRAY_HDR}, then rebuild"
                CUR_CAP_NA="${na}"
                return 0
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
CSV="${OUTDIR}/runs.csv"
SUMMARY="${OUTDIR}/summary.txt"
mkdir -p "${RAWDIR}" "${HFDIR}"

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
        echo "timeout:      ${TIMEOUT_S}s"
} | tee "${OUTDIR}/config.txt"
echo

echo "impl,variant,mode,ranks,nodes,ppn,placement,na,total,grid,planes,timesteps,rep,status,elapsed,tsps,departures,landings,overflows,misroutes,conservation,expected,cons_ok" > "${CSV}"

# ==========================================================================
# One run
# ==========================================================================
run_one() {
        local impl="$1" variant="$2" mode="$3" np="$4" nodes="$5" ppn="$6"
        local na="$7" total="$8" grid="$9" rep="${10}"
        local exe flags log rc hf tag

        exe="$(exe_for "${impl}")"
        flags="$(variant_flags "${impl}" "${variant}")"
        tag="${impl}-${variant}-${mode}-np${np}-n${nodes}x${ppn}-na${na}-r${rep}"
        log="${RAWDIR}/${tag}.log"
        hf="$(hostfile_for "${nodes}")"

        # ---- launch line -------------------------------------------------
        # Open MPI: --prefix + -x PATH/-x LD_LIBRARY_PATH are what make orted
        # resolvable on the compute nodes. --map-by ppr:N:node is only added
        # when the ranks divide evenly across the nodes, matching weak_scale.sh.
        local launch=()
        if [[ "${MPI_FLAVOR}" == openmpi ]]; then
                launch=("${MPIRUN}" -n "${np}" --hostfile "${hf}")
                [[ -n "${MPI_PREFIX}" ]] && launch+=(--prefix "${MPI_PREFIX}")
                if (( np > 1 )) && (( np % nodes == 0 )); then
                        launch+=(--map-by "ppr:${ppn}:node")
                fi
                launch+=("${MPI_X_FLAGS[@]}")
        else
                launch=("${MPIRUN}" -n "${np}" -f "${hf}" -ppn "${ppn}")
        fi

        # shellcheck disable=SC2206
        local cmd=("${launch[@]}" ${MPI_EXTRA} "${exe}"
                   --na "${na}" --time "${TIMESTEPS}"
                   --planes "${PLANES}" --seed "${SEED}" ${flags} ${VERBOSITY})

        printf '  %-6s %-10s np=%-4s %sx%-3s na=%-6s total=%-7s rep=%s ... ' \
               "${impl}" "${variant}" "${np}" "${nodes}" "${ppn}" \
               "${na}" "${total}" "${rep}"

        if [[ ${DRY_RUN} -eq 1 ]]; then
                printf '\n      %s\n' "${cmd[*]}"
                return 0
        fi

        local t0 t1 wall
        t0=$(date +%s.%N)
        "${TIMEOUT_CMD[@]}" "${cmd[@]}" > "${log}" 2>&1
        rc=$?
        t1=$(date +%s.%N)
        wall=$(awk -v a="${t0}" -v b="${t1}" 'BEGIN { printf "%.4f", b - a }')

        local status="ok"
        if   [[ ${rc} -eq 124 || ${rc} -eq 137 ]]; then status="timeout"
        elif [[ ${rc} -ne 0 ]];                   then status="fail(${rc})"
        fi

        # Launcher-level failures are distinct from application failures;
        # tag them so they don't read as DSTM bugs in the summary.
        if grep -q "orted: command not found\|ORTE was unable to reliably start" "${log}" 2>/dev/null; then
                status="launch_err"
        fi

        local elapsed tsps dep land ovf mis cons exp cons_ok
        elapsed=$(awk '/^Elapsed time:/      { print $3 }' "${log}" | tail -1)
        tsps=$(   awk '/^Timesteps\/second:/ { print $2 }' "${log}" | tail -1)
        dep=$(    awk '/^Total departures:/  { print $3 }' "${log}" | tail -1)
        land=$(   awk '/^Total landings:/    { print $3 }' "${log}" | tail -1)
        ovf=$(    awk '/^Gate overflows:/    { print $3 }' "${log}" | tail -1)
        mis=$(    awk '/^Misrouted flights:/ { print $3 }' "${log}" | tail -1)
        cons=$(   awk '/^Conservation:/      { print $2 }' "${log}" | tail -1)
        exp=$(    awk '/^Conservation:/      { print $4 }' "${log}" | tail -1)
        cons_ok=$(awk '/^Conservation:/      { print $5 }' "${log}" | tail -1)

        [[ -z "${elapsed}" ]] && { elapsed="${wall}"
                                   [[ "${status}" == ok ]] && status="parse_fail"; }
        : "${tsps:=nan}" "${dep:=-1}" "${land:=-1}" "${ovf:=-1}"
        : "${mis:=-1}" "${cons:=-1}" "${exp:=-1}" "${cons_ok:=n/a}"

        echo "${impl},${variant},${mode},${np},${nodes},${ppn},${PLACEMENT},${na},${total},${grid},${PLANES},${TIMESTEPS},${rep},${status},${elapsed},${tsps},${dep},${land},${ovf},${mis},${cons},${exp},${cons_ok}" >> "${CSV}"

        if [[ "${status}" == ok ]]; then
                printf '%9ss %9s ts/s  %s\n' "${elapsed}" "${tsps}" "${cons_ok}"
        else
                printf '%s (%s)\n' "${status}" "$(basename "${log}")"
        fi
}

# ==========================================================================
# Sweep
# ==========================================================================
ladder_for() { case "$1" in weak) echo "${WEAK_LADDER}";; strong) echo "${STRONG_LADDER}";; esac; }

for mode in ${MODES}; do
        echo "===== ${mode} scaling ====="
        for rung in $(ladder_for "${mode}"); do
                np="${rung%%:*}"
                rest="${rung#*:}"
                nodes="${rest%%:*}"
                ppn="${rest##*:}"

                # --- placement rewrite -------------------------------------
                if [[ "${PLACEMENT}" == spread ]]; then
                        nodes=${AVAIL_NODES}
                        (( nodes > np )) && nodes=${np}
                        ppn=$(( (np + nodes - 1) / nodes ))
                fi

                # --- allocation / cap checks --------------------------------
                if (( ppn > MAX_PPN )); then
                        echo "  skip np=${np}: ppn ${ppn} exceeds cap ${MAX_PPN}"
                        continue
                fi
                if (( nodes > AVAIL_NODES )); then
                        echo "  skip np=${np}: needs ${nodes} nodes, have ${AVAIL_NODES}"
                        continue
                fi
                if (( nodes * ppn < np )); then
                        echo "  skip np=${np}: ${nodes}x${ppn} cannot host ${np} ranks"
                        continue
                fi

                # --- problem sizing ----------------------------------------
                if [[ "${mode}" == weak ]]; then
                        na=${NA_PER_RANK}
                        total=$(( na * np ))
                else
                        if (( TOTAL_AIRPORTS % np != 0 )); then
                                echo "  skip np=${np}: ${TOTAL_AIRPORTS} not divisible by ranks"
                                continue
                        fi
                        na=$(( TOTAL_AIRPORTS / np ))
                        total=${TOTAL_AIRPORTS}
                fi

                grid=$(isqrt_exact "${total}")
                if [[ "${grid}" == "0" ]]; then
                        echo "  skip np=${np}: total ${total} is not a perfect square"
                        continue
                fi
                if (( grid < 2 )); then
                        echo "  skip np=${np}: grid ${grid} < 2 (dgraph main rejects it)"
                        continue
                fi

                # --- capacity: patch headers + rebuild for this na ----------
                if ! set_capacity "${na}"; then
                        echo "  skip np=${np}: capacity build failed for na=${na}"
                        continue
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
# ==========================================================================
np_list_for() {
        awk -F, -v m="$1" 'NR > 1 && $3 == m { print $4 }' "${CSV}" | sort -n -u
}

summarize() {
        local mode="$1" keys key impl variant np med base_med first_np na tot ts nodes ppn

        keys=$(awk -F, -v m="${mode}" 'NR > 1 && $3 == m { print $1 "|" $2 }' "${CSV}" | sort -u)
        [[ -z "${keys}" ]] && return

        echo "===== ${mode} scaling (median of ${REPS} reps) ====="
        printf '%-8s %-10s %6s %8s %7s %8s %11s %10s %9s %8s\n' \
               impl variant ranks layout na total "elapsed(s)" "ts/s" speedup eff
        printf -- '-------------------------------------------------------------------------------------------------\n'

        for key in ${keys}; do
                impl="${key%%|*}"; variant="${key##*|}"
                base_med=""; first_np=""
                for np in $(np_list_for "${mode}"); do
                        med=$(awk -F, -v m="${mode}" -v i="${impl}" -v v="${variant}" -v n="${np}" \
                                  '$3==m && $1==i && $2==v && $4==n && $14=="ok" { print $15 }' \
                                  "${CSV}" | median)
                        [[ "${med}" == "nan" ]] && continue

                        ts=$(awk -F, -v m="${mode}" -v i="${impl}" -v v="${variant}" -v n="${np}" \
                                 '$3==m && $1==i && $2==v && $4==n && $14=="ok" { print $16 }' \
                                 "${CSV}" | median)
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
                                printf "%-8s %-10s %6s %8s %7s %8s %11.4f %10.2f %8.2fx %7.0f%%\n",
                                       i, v, n, lay, na, tot, e, ts, sp, eff * 100;
                        }'
                done
                echo
        done

        if [[ "${IMPLS}" == *darray* && "${IMPLS}" == *graph* ]]; then
                echo "  head to head (darray / dgraph elapsed; > 1.00 means dgraph is faster)"
                printf '  %6s %8s %12s %12s %10s\n' ranks layout darray dgraph ratio
                for np in $(np_list_for "${mode}"); do
                        local d gph lay
                        d=$(awk -F, -v m="${mode}" -v n="${np}" \
                                '$3==m && $1=="darray" && $4==n && $14=="ok" { print $15 }' "${CSV}" | median)
                        gph=$(awk -F, -v m="${mode}" -v n="${np}" \
                                '$3==m && $1=="graph" && $2=="base" && $4==n && $14=="ok" { print $15 }' "${CSV}" | median)
                        [[ "${d}" == "nan" || "${gph}" == "nan" ]] && continue
                        lay=$(awk -F, -v m="${mode}" -v n="${np}" \
                                '$3==m && $4==n { print $5 "x" $6; exit }' "${CSV}")
                        awk -v n="${np}" -v l="${lay}" -v d="${d}" -v g="${gph}" \
                            'BEGIN { printf "  %6s %8s %12.4f %12.4f %9.2fx\n", n, l, d, g, (g > 0 ? d / g : 0) }'
                done
                echo
        fi
}

{
        cat "${OUTDIR}/config.txt"
        echo
        for mode in ${MODES}; do summarize "${mode}"; done

        total_runs=$(( $(wc -l < "${CSV}") - 1 ))
        bad=$(awk -F, 'NR > 1 && $14 != "ok"' "${CSV}" | wc -l)
        tmo=$(awk -F, 'NR > 1 && $14 == "timeout"' "${CSV}" | wc -l)
        lerr=$(awk -F, 'NR > 1 && $14 == "launch_err"' "${CSV}" | wc -l)
        mism=$(awk -F, 'NR > 1 && $23 == "MISMATCH"' "${CSV}" | wc -l)

        echo "runs recorded:           ${total_runs}"
        echo "failed:                  ${bad}  (timeouts: ${tmo}, launcher: ${lerr})"
        echo "conservation mismatches: ${mism}"
        if (( lerr > 0 )); then
                echo
                echo "NOTE: ${lerr} run(s) failed in the MPI launcher, not the application."
                echo "      Check --mpi-prefix (currently '${MPI_PREFIX}') and --mpi-env."
        fi
        if (( bad > 0 )); then
                echo
                echo "non-ok runs:"
                awk -F, 'NR > 1 && $14 != "ok" { printf "  %-6s %-10s %-6s np=%-4s %sx%-3s -> %s\n", $1,$2,$3,$4,$5,$6,$14 }' "${CSV}"
        fi
} | tee "${SUMMARY}"

echo
echo "csv:     ${CSV}"
echo "summary: ${SUMMARY}"
echo "logs:    ${RAWDIR}"
