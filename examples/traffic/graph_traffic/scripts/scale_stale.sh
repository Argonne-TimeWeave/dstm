#!/bin/bash
# scale_stale.sh -- fix ranks, sweep --stale-delay for graph_traffic.
#
#   For a single fixed rank count, run ./traffic --replica-mode stale over
#   a ladder of --stale-delay values and record trip/staleness stats to a
#   CSV. One row per (stale_delay, rep).
#
#   Only the STALE replica mode reads --stale-delay (oracle syncs every
#   tick, blind never merges), so this sweep runs stale mode only.
#
# LAUNCH RECIPE (matches scale_replica.sh / run_scaling.sh)
#   Under a batch allocation:
#     mpirun -n P --hostfile $NODEFILE --prefix $MPI_PREFIX \
#            --map-by ppr:PPN:node -x LD_LIBRARY_PATH -x PATH ./traffic ...
#   Interactively: mpirun -n P ./traffic ...
#
# USAGE
#   ./scale_stale.sh                              # default ladder @ 256 ranks
#   RANKS=128 ./scale_stale.sh
#   STALE_DELAYS="1 2 4 8 16 32" ./scale_stale.sh
#   ./scale_stale.sh --ranks 128 --stale-delays "1 3 5 10 20" --reps 3
#   ./scale_stale.sh --dry-run
#
# OUTPUT
#   results/<stamp>/runs.csv      one row per run
#   results/<stamp>/raw/*.log     full stdout+stderr per run
#   results/<stamp>/config.txt    what was launched
set -uo pipefail

# --------------------------------------------------------------------------
# Locate ourselves; graph_traffic dir is one up from scripts/.
# --------------------------------------------------------------------------
if   [[ -n "${PBS_O_WORKDIR:-}"   ]]; then cd "${PBS_O_WORKDIR}"   || exit 1; SCRIPT_DIR="${PBS_O_WORKDIR}"
elif [[ -n "${COBALT_STARTDIR:-}" ]]; then cd "${COBALT_STARTDIR}" || exit 1; SCRIPT_DIR="${COBALT_STARTDIR}"
else SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
# scripts/ lives inside graph_traffic; the exe is one level up.
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXE="${EXE:-${ROOT}/traffic}"

# --------------------------------------------------------------------------
# Defaults (override via env or flags)
# --------------------------------------------------------------------------
MPIRUN="${MPIRUN:-mpirun}"
MPI_PREFIX="${MPI_PREFIX:-}"                 # auto-derived from `which mpirun`
MPI_ENV_VARS="${MPI_ENV_VARS:-LD_LIBRARY_PATH PATH}"
MPI_EXTRA="${MPI_EXTRA:-}"
MAX_PPN="${MAX_PPN:-32}"                     # 32 ranks/node on this machine
MAX_NODES="${MAX_NODES:-8}"

DATA="${DATA:-${ROOT}/data/bloomington/}"
RANKS="${RANKS:-256}"                        # FIXED rank count for this sweep
MODE="${MODE:-stale}"                        # only stale reads --stale-delay
STALE_DELAYS="${STALE_DELAYS:-1 2 4 8 16 32}"
ORACLE_BASELINE="${ORACLE_BASELINE:-1}"      # also run oracle REPS times as an
                                             # always-fresh upper-bound baseline
BLIND_BASELINE="${BLIND_BASELINE:-1}"        # also run blind REPS times as a
                                             # never-merge lower-bound baseline

STEPS="${STEPS:-1024}"
CARS="${CARS:-256}"
RATE="${RATE:-256}"
CAP="${CAP:-32}"
SEED="${SEED:-32}"
REPS="${REPS:-1}"
TIMEOUT_S="${TIMEOUT_S:-600}"               # per-run guard; 0 disables
VERBOSITY="${VERBOSITY:---quiet}"
DRY_RUN=0
OUTROOT="${OUTROOT:-${SCRIPT_DIR}/stale-results}"

usage() {
        cat <<EOF
scale_stale.sh -- fix ranks, sweep --stale-delay for graph_traffic (stale mode)

  --data DIR              network dir (node.csv/link.csv)  (default ${DATA})
  --ranks N              fixed rank count                 (default ${RANKS})
  --mode M              replica mode (only stale reacts)  (default ${MODE})
  --stale-delays "1 2 4" stale-delay ladder               (default "${STALE_DELAYS}")
  --oracle-baseline 0|1  also run oracle REPS times       (default ${ORACLE_BASELINE})
  --no-oracle           shorthand for --oracle-baseline 0
  --blind-baseline 0|1   also run blind REPS times        (default ${BLIND_BASELINE})
  --no-blind            shorthand for --blind-baseline 0
  --steps N              ticks                            (default ${STEPS})
  --cars N              burst spawn count                (default ${CARS})
  --rate N              continuous spawn rate            (default ${RATE})
  --cap N               per-edge car cap                 (default ${CAP})
  --seed N              RNG seed                         (default ${SEED})
  --reps N              repetitions per point            (default ${REPS})
  --timeout S           per-run seconds, 0=off           (default ${TIMEOUT_S})
  --max-ppn N           ranks per node cap               (default ${MAX_PPN})
  --max-nodes N         node cap                         (default ${MAX_NODES})
  --exe PATH            traffic binary                   (default ${EXE})
  --mpirun CMD / --mpi-prefix DIR / --mpi-env "A B" / --mpi-extra "ARGS"
  --loud                drop --quiet
  --out DIR / --dry-run / -h
EOF
}

while [[ $# -gt 0 ]]; do
        case "$1" in
        --data)         DATA="$2"; shift 2 ;;
        --ranks)        RANKS="$2"; shift 2 ;;
        --mode)         MODE="$2"; shift 2 ;;
        --stale-delays) STALE_DELAYS="$2"; shift 2 ;;
        --oracle-baseline) ORACLE_BASELINE="$2"; shift 2 ;;
        --no-oracle)    ORACLE_BASELINE=0; shift ;;
        --blind-baseline) BLIND_BASELINE="$2"; shift 2 ;;
        --no-blind)     BLIND_BASELINE=0; shift ;;
        --steps)        STEPS="$2"; shift 2 ;;
        --cars)         CARS="$2"; shift 2 ;;
        --rate)         RATE="$2"; shift 2 ;;
        --cap)          CAP="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --reps)         REPS="$2"; shift 2 ;;
        --timeout)      TIMEOUT_S="$2"; shift 2 ;;
        --max-ppn)      MAX_PPN="$2"; shift 2 ;;
        --max-nodes)    MAX_NODES="$2"; shift 2 ;;
        --exe)          EXE="$2"; shift 2 ;;
        --mpirun)       MPIRUN="$2"; shift 2 ;;
        --mpi-prefix)   MPI_PREFIX="$2"; shift 2 ;;
        --mpi-env)      MPI_ENV_VARS="$2"; shift 2 ;;
        --mpi-extra)    MPI_EXTRA="$2"; shift 2 ;;
        --loud)         VERBOSITY=""; shift ;;
        --out)          OUTROOT="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
        esac
done

# --------------------------------------------------------------------------
# Allocation discovery (same precedence as scale_replica.sh).
# --------------------------------------------------------------------------
ALLOC_NODEFILE=""; ALLOC_SRC="none"
if   [[ -n "${COBALT_NODEFILE:-}" && -r "${COBALT_NODEFILE}" ]]; then ALLOC_NODEFILE="${COBALT_NODEFILE}"; ALLOC_SRC="COBALT_NODEFILE"
elif [[ -n "${PBS_NODEFILE:-}"    && -r "${PBS_NODEFILE}"    ]]; then ALLOC_NODEFILE="${PBS_NODEFILE}";    ALLOC_SRC="PBS_NODEFILE"
elif [[ -n "${HOSTFILE:-}"        && -r "${HOSTFILE}"        ]]; then ALLOC_NODEFILE="${HOSTFILE}";        ALLOC_SRC="HOSTFILE"
elif [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
        ALLOC_NODEFILE="$(mktemp)"; scontrol show hostnames "${SLURM_JOB_NODELIST}" > "${ALLOC_NODEFILE}"; ALLOC_SRC="SLURM"
fi

NODELIST=()
if [[ -n "${ALLOC_NODEFILE}" ]]; then mapfile -t NODELIST < <(sort -u "${ALLOC_NODEFILE}")
else NODELIST=("$(hostname)"); ALLOC_SRC="localhost"
fi
AVAIL_NODES=${#NODELIST[@]}
(( AVAIL_NODES > MAX_NODES )) && AVAIL_NODES=${MAX_NODES}

# --------------------------------------------------------------------------
# MPI prefix (two dirnames up from `which mpirun`), for --prefix / orted.
# --------------------------------------------------------------------------
if [[ -z "${MPI_PREFIX}" ]]; then
        _mp="$(command -v "${MPIRUN}" 2>/dev/null)"
        [[ -n "${_mp}" ]] && MPI_PREFIX="$(dirname "$(dirname "${_mp}")")"
fi
MPI_X_FLAGS=()
for _v in ${MPI_ENV_VARS}; do MPI_X_FLAGS+=(-x "${_v}"); done

# --------------------------------------------------------------------------
# Preflight.
# --------------------------------------------------------------------------
[[ -x "${EXE}" || ${DRY_RUN} -eq 1 ]] || { echo "ERROR: traffic exe missing/not executable: ${EXE}" >&2; exit 1; }
command -v "${MPIRUN}" >/dev/null 2>&1 || { echo "ERROR: launcher '${MPIRUN}' not on PATH" >&2; [[ ${DRY_RUN} -eq 0 ]] && exit 1; }
if [[ ! -d "${ROOT}/${DATA}" && ! -d "${DATA}" && ${DRY_RUN} -eq 0 ]]; then
        echo "WARNING: data dir not found relative to ${ROOT} or CWD: ${DATA}" >&2
fi

TIMEOUT_CMD=()
if [[ "${TIMEOUT_S}" != "0" ]] && command -v timeout >/dev/null 2>&1; then
        TIMEOUT_CMD=(timeout --signal=TERM --kill-after=30 "${TIMEOUT_S}")
fi

# Pick nodes/ppn for a rank count: pack onto as few nodes as possible,
# <= MAX_PPN per node. Returns "nodes ppn" or "" if it won't fit.
place_for() {
        local np="$1" nodes ppn
        nodes=$(( (np + MAX_PPN - 1) / MAX_PPN ))
        (( nodes < 1 )) && nodes=1
        (( nodes > AVAIL_NODES )) && { echo ""; return; }
        ppn=$(( (np + nodes - 1) / nodes ))
        (( ppn > MAX_PPN )) && { echo ""; return; }
        echo "${nodes} ${ppn}"
}

# --------------------------------------------------------------------------
# Output tree.
# --------------------------------------------------------------------------
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTDIR="${OUTROOT}/${STAMP}"; RAWDIR="${OUTDIR}/raw"
CSV="${OUTDIR}/runs.csv"; CONFIG="${OUTDIR}/config.txt"
mkdir -p "${RAWDIR}"

# Resolve placement once: rank count is fixed for the whole sweep.
read -r NODES PPN < <(place_for "${RANKS}")
if [[ -z "${NODES:-}" ]]; then
        echo "ERROR: np=${RANKS} needs > ${AVAIL_NODES} nodes at ${MAX_PPN} ppn" >&2
        [[ ${DRY_RUN} -eq 0 ]] && exit 1
        NODES=1; PPN="${RANKS}"   # let dry-run still print something
fi

{
        echo "host:         $(hostname)"
        echo "date:         $(date -Is)"
        echo "job id:       ${COBALT_JOBID:-${PBS_JOBID:-${SLURM_JOB_ID:-interactive}}}"
        echo "alloc source: ${ALLOC_SRC}  (${#NODELIST[@]} nodes, using <= ${AVAIL_NODES})"
        echo "nodefile:     ${ALLOC_NODEFILE:-<none>}"
        echo "exe:          ${EXE}"
        echo "data:         ${DATA}"
        echo "mpi prefix:   ${MPI_PREFIX:-<unset>}   forwarded: ${MPI_ENV_VARS}"
        echo "ranks:        ${RANKS}  (${NODES}x${PPN})   mode: ${MODE}"
        echo "stale delays: ${STALE_DELAYS}"
        echo "oracle base:  $([[ ${ORACLE_BASELINE} -eq 1 ]] && echo "yes (REPS=${REPS})" || echo "no")"
        echo "blind base:   $([[ ${BLIND_BASELINE} -eq 1 ]] && echo "yes (REPS=${REPS})" || echo "no")"
        echo "steps:        ${STEPS}   cars: ${CARS}   rate: ${RATE}   cap: ${CAP}   seed: ${SEED}"
        echo "reps:         ${REPS}    timeout: ${TIMEOUT_S}s   max ppn: ${MAX_PPN}"
} | tee "${CONFIG}"
echo

# Requested columns: mean_dur, mode, rank(s), stale_mx, age (+ arrived).
# Plus the swept variable stale_delay and rep so rows are identifiable.
echo "ranks,mode,stale_delay,rep,status,mean_dur,arrived,mean_age,stale_max" > "${CSV}"

# --------------------------------------------------------------------------
# One run.
# --------------------------------------------------------------------------
run_one() {
        local np="$1" nodes="$2" ppn="$3" mode="$4" sd="$5" rep="$6"
        local tag log launch=() cmd=() rc status
        local sd_csv sd_arg

        # Oracle and blind ignore --stale-delay: record "na" in the CSV
        # column, but still pass a valid integer (1) to the binary so
        # run_one stays uniform. Stale uses the real sweep value for both.
        if [[ "${mode}" == "oracle" || "${mode}" == "blind" ]]; then
                sd_csv="na"; sd_arg="1"
        else
                sd_csv="${sd}"; sd_arg="${sd}"
        fi

        tag="np${np}-n${nodes}x${ppn}-${mode}-sd${sd_csv}-r${rep}"
        log="${RAWDIR}/${tag}.log"

        # Batch allocation -> full recipe; interactive single node -> plain -n.
        if [[ "${ALLOC_SRC}" != "localhost" ]]; then
                launch=("${MPIRUN}" -n "${np}" --hostfile "${ALLOC_NODEFILE}")
                [[ -n "${MPI_PREFIX}" ]] && launch+=(--prefix "${MPI_PREFIX}")
                (( np > 1 && np % nodes == 0 )) && launch+=(--map-by "ppr:${ppn}:node")
                launch+=("${MPI_X_FLAGS[@]}")
        else
                launch=("${MPIRUN}" -n "${np}")
        fi

        # shellcheck disable=SC2206
        cmd=("${launch[@]}" ${MPI_EXTRA} "${EXE}"
             --data "${DATA}" --steps "${STEPS}" --cars "${CARS}"
             --rate "${RATE}" --cap "${CAP}" --seed "${SEED}"
             --replica-mode "${mode}" --stale-delay "${sd_arg}" ${VERBOSITY})

        printf '  np=%-4s %sx%-3s %-7s sd=%-3s rep=%s ... ' "${np}" "${nodes}" "${ppn}" "${mode}" "${sd_csv}" "${rep}"

        if [[ ${DRY_RUN} -eq 1 ]]; then
                printf '\n      %s\n' "${cmd[*]}"
                return 0
        fi

        "${TIMEOUT_CMD[@]}" "${cmd[@]}" > "${log}" 2>&1
        rc=$?

        status="ok"
        if   [[ ${rc} -eq 124 || ${rc} -eq 137 ]]; then status="timeout"
        elif [[ ${rc} -ne 0 ]];                   then status="fail(${rc})"
        fi
        if grep -q "orted: command not found\|ORTE was unable to reliably start" "${log}" 2>/dev/null; then
                status="launch_err"
        fi

        # Parse only the fields we keep: mean_dur, arrived, mean_age, stale_max.
        local dur arr age smax
        dur=$(  sed -n 's/.*mean duration=\([0-9.]*\).*/\1/p'              "${log}" | tail -1)
        arr=$(  sed -n 's/^spawned=[0-9]* arrived=\([0-9]*\).*/\1/p'       "${log}" | tail -1)
        age=$(  sed -n 's/.*consulted entry=\([0-9.]*\).*/\1/p'            "${log}" | tail -1)
        smax=$( sed -n 's/.*staleness.*max=\([0-9]*\).*/\1/p'              "${log}" | tail -1)

        [[ -z "${dur}" && "${status}" == ok ]] && status="parse_fail"
        : "${dur:=nan}" "${arr:=-1}" "${age:=nan}" "${smax:=-1}"

        echo "${np},${mode},${sd_csv},${rep},${status},${dur},${arr},${age},${smax}" >> "${CSV}"

        if [[ "${status}" == ok ]]; then
                printf 'dur=%-8s arr=%-7s age=%-6s stale_mx=%s\n' "${dur}" "${arr}" "${age}" "${smax}"
        else
                printf '%s (%s)\n' "${status}" "$(basename "${log}")"
        fi
}

# --------------------------------------------------------------------------
# Sweep: stale_delay (outer) x reps. Rank count is fixed.
# --------------------------------------------------------------------------
echo "===== ranks=${RANKS}  (${NODES}x${PPN})  mode=${MODE} ====="
for sd in ${STALE_DELAYS}; do
        for (( rep = 1; rep <= REPS; rep++ )); do
                run_one "${RANKS}" "${NODES}" "${PPN}" "${MODE}" "${sd}" "${rep}"
        done
done
echo

# Always-fresh upper bound: oracle syncs every tick, so it does not depend
# on stale_delay. Run it REPS times once, not once per sweep point.
if [[ "${ORACLE_BASELINE}" -eq 1 && "${MODE}" != "oracle" ]]; then
        echo "===== oracle baseline (ranks=${RANKS}, REPS=${REPS}) ====="
        for (( rep = 1; rep <= REPS; rep++ )); do
                run_one "${RANKS}" "${NODES}" "${PPN}" "oracle" "na" "${rep}"
        done
        echo
fi

# Never-merge lower bound: blind never reads --stale-delay, so like oracle
# it does not depend on stale_delay. Run it REPS times once.
if [[ "${BLIND_BASELINE}" -eq 1 && "${MODE}" != "blind" ]]; then
        echo "===== blind baseline (ranks=${RANKS}, REPS=${REPS}) ====="
        for (( rep = 1; rep <= REPS; rep++ )); do
                run_one "${RANKS}" "${NODES}" "${PPN}" "blind" "na" "${rep}"
        done
        echo
fi

[[ ${DRY_RUN} -eq 1 ]] && { echo "dry run complete"; exit 0; }

echo "csv:  ${CSV}"
echo "logs: ${RAWDIR}"
echo
echo "quick median-of-reps view (by stale_delay; oracle baseline last):"
awk -F, 'NR>1 && $5=="ok" {
        # For stale rows key on stale_delay; for the na-keyed baselines
        # (oracle/blind) key on the mode so they do not collide.
        key = ($3 == "na") ? $2 : $3;
        dur[key]=dur[key]" "$6; arr[key]=arr[key]" "$7; age[key]=age[key]" "$8; sm[key]=sm[key]" "$9;
        if ($3 == "na") { base[$2]=1 } else { seen[key]=1 }   # keep na out of numeric sort
}
function med(s,   a,n,i,j,t) { n=split(s,a," "); if(!n) return "nan";
        for(i=1;i<=n;i++) for(j=i+1;j<=n;j++) if(a[j]<a[i]){t=a[i];a[i]=a[j];a[j]=t};
        return (n%2)? a[(n+1)/2] : (a[n/2]+a[n/2+1])/2 }
END {
        printf "%-12s %10s %10s %8s %9s\n","stale_delay","mean_dur","arrived","age","stale_mx";
        n=asorti(seen,sk,"@val_num_asc");
        for(i=1;i<=n;i++){ k=sk[i];
                printf "%-12s %10s %10s %8s %9s\n", k, med(dur[k]), med(arr[k]), med(age[k]), med(sm[k]) }
        if ("oracle" in base)
                printf "%-12s %10s %10s %8s %9s   <- always-fresh upper bound\n", \
                       "oracle", med(dur["oracle"]), med(arr["oracle"]), med(age["oracle"]), med(sm["oracle"]);
        if ("blind" in base)
                printf "%-12s %10s %10s %8s %9s   <- never-merge lower bound\n", \
                       "blind", med(dur["blind"]), med(arr["blind"]), med(age["blind"]), med(sm["blind"]);
}' "${CSV}" 2>/dev/null || echo "  (install gawk for the summary view; raw CSV is complete)"
