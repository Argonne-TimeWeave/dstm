#!/bin/bash
# scale_replica.sh -- scale ranks x replica-mode sweep for graph_traffic.
#
#   For each rank count in the ladder and each replica mode (blind stale
#   oracle), run ./traffic --data DIR and record runtime + trip stats to a
#   CSV. One row per (ranks, mode, rep).
#
# LAUNCH RECIPE (matches run_scaling.sh / ROSS weak_scale.sh)
#   Under a batch allocation we launch with:
#     mpirun -n P --hostfile $NODEFILE --prefix $MPI_PREFIX \
#            --map-by ppr:PPN:node -x LD_LIBRARY_PATH -x PATH ./traffic ...
#   Interactively (no allocation) we fall back to: mpirun -n P ./traffic ...
#
# USAGE
#   ./scale_replica.sh                         # default ladder, both @ bloomington
#   RANKS="1 2 4 8" ./scale_replica.sh
#   ./scale_replica.sh --data data/bloomington/ --steps 200 --rate 1 --cap 16
#   ./scale_replica.sh --cars 2048 --rate 1 --reps 3
#   ./scale_replica.sh --dry-run
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
RANKS="${RANKS:-4 8 16 32 64 128 256}"
MODES="${MODES:-stale oracle}"

STEPS="${STEPS:-256}"
CARS="${CARS:-256}"
RATE="${RATE:-256}"
CAP="${CAP:-2}"
SEED="${SEED:-32}"
REPS="${REPS:-2}"
TIMEOUT_S="${TIMEOUT_S:-600}"               # per-run guard; 0 disables
VERBOSITY="${VERBOSITY:---quiet}"
DRY_RUN=0
OUTROOT="${OUTROOT:-${SCRIPT_DIR}/results}"

usage() {
        cat <<EOF
scale_replica.sh -- scale ranks x {blind,stale,oracle} for graph_traffic

  --data DIR        network dir (node.csv/link.csv)  (default ${DATA})
  --ranks "1 2 4"   rank ladder                      (default "${RANKS}")
  --modes "a b c"   replica modes                    (default "${MODES}")
  --steps N         ticks                            (default ${STEPS})
  --cars N          burst spawn count                (default ${CARS})
  --rate N          continuous spawn rate            (default ${RATE})
  --cap N           per-edge car cap                 (default ${CAP})
  --seed N          RNG seed                         (default ${SEED})
  --reps N          repetitions per point            (default ${REPS})
  --timeout S       per-run seconds, 0=off           (default ${TIMEOUT_S})
  --max-ppn N       ranks per node cap               (default ${MAX_PPN})
  --max-nodes N     node cap                         (default ${MAX_NODES})
  --exe PATH        traffic binary                   (default ${EXE})
  --mpirun CMD / --mpi-prefix DIR / --mpi-env "A B" / --mpi-extra "ARGS"
  --loud            drop --quiet
  --out DIR / --dry-run / -h
EOF
}

while [[ $# -gt 0 ]]; do
        case "$1" in
        --data)       DATA="$2"; shift 2 ;;
        --ranks)      RANKS="$2"; shift 2 ;;
        --modes)      MODES="$2"; shift 2 ;;
        --steps)      STEPS="$2"; shift 2 ;;
        --cars)       CARS="$2"; shift 2 ;;
        --rate)       RATE="$2"; shift 2 ;;
        --cap)        CAP="$2"; shift 2 ;;
        --seed)       SEED="$2"; shift 2 ;;
        --reps)       REPS="$2"; shift 2 ;;
        --timeout)    TIMEOUT_S="$2"; shift 2 ;;
        --max-ppn)    MAX_PPN="$2"; shift 2 ;;
        --max-nodes)  MAX_NODES="$2"; shift 2 ;;
        --exe)        EXE="$2"; shift 2 ;;
        --mpirun)     MPIRUN="$2"; shift 2 ;;
        --mpi-prefix) MPI_PREFIX="$2"; shift 2 ;;
        --mpi-env)    MPI_ENV_VARS="$2"; shift 2 ;;
        --mpi-extra)  MPI_EXTRA="$2"; shift 2 ;;
        --loud)       VERBOSITY=""; shift ;;
        --out)        OUTROOT="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
        esac
done

# --------------------------------------------------------------------------
# Allocation discovery (same precedence as run_scaling.sh).
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

{
        echo "host:         $(hostname)"
        echo "date:         $(date -Is)"
        echo "job id:       ${COBALT_JOBID:-${PBS_JOBID:-${SLURM_JOB_ID:-interactive}}}"
        echo "alloc source: ${ALLOC_SRC}  (${#NODELIST[@]} nodes, using <= ${AVAIL_NODES})"
        echo "nodefile:     ${ALLOC_NODEFILE:-<none>}"
        echo "exe:          ${EXE}"
        echo "data:         ${DATA}"
        echo "mpi prefix:   ${MPI_PREFIX:-<unset>}   forwarded: ${MPI_ENV_VARS}"
        echo "ranks:        ${RANKS}"
        echo "modes:        ${MODES}"
        echo "steps:        ${STEPS}   cars: ${CARS}   rate: ${RATE}   cap: ${CAP}   seed: ${SEED}"
        echo "reps:         ${REPS}    timeout: ${TIMEOUT_S}s   max ppn: ${MAX_PPN}"
} | tee "${CONFIG}"
echo

echo "ranks,nodes,ppn,mode,rep,status,run_time,mean_dur,mean_hops,arrived,spawned,live,lost,blocked_pct,crossings,mean_age,stale_mean,stale_max,seed,steps,cars,rate,cap" > "${CSV}"

# --------------------------------------------------------------------------
# One run.
# --------------------------------------------------------------------------
run_one() {
        local np="$1" nodes="$2" ppn="$3" mode="$4" rep="$5"
        local tag log launch=() cmd=() rc t0 t1 wall status

        tag="np${np}-n${nodes}x${ppn}-${mode}-r${rep}"
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
             --replica-mode "${mode}" ${VERBOSITY})

        printf '  np=%-4s %sx%-3s %-7s rep=%s ... ' "${np}" "${nodes}" "${ppn}" "${mode}" "${rep}"

        if [[ ${DRY_RUN} -eq 1 ]]; then
                printf '\n      %s\n' "${cmd[*]}"
                return 0
        fi

        t0=$(date +%s.%N)
        "${TIMEOUT_CMD[@]}" "${cmd[@]}" > "${log}" 2>&1
        rc=$?
        t1=$(date +%s.%N)
        wall=$(awk -v a="${t0}" -v b="${t1}" 'BEGIN { printf "%.4f", b - a }')

        status="ok"
        if   [[ ${rc} -eq 124 || ${rc} -eq 137 ]]; then status="timeout"
        elif [[ ${rc} -ne 0 ]];                   then status="fail(${rc})"
        fi
        if grep -q "orted: command not found\|ORTE was unable to reliably start" "${log}" 2>/dev/null; then
                status="launch_err"
        fi

        # Parse the traffic summary block (same sed/awk style as compare.sh).
        local dur hops arr spawn live lost blk cross age smean smax prog_time
        prog_time=$(sed -n 's/^total run time: \([0-9.]*\).*/\1/p'              "${log}" | tail -1)
        dur=$(  sed -n 's/.*mean duration=\([0-9.]*\).*/\1/p'                    "${log}" | tail -1)
        hops=$( sed -n 's/.*mean hops=\([0-9.]*\).*/\1/p'                        "${log}" | tail -1)
        arr=$(  sed -n 's/^spawned=[0-9]* arrived=\([0-9]*\).*/\1/p'             "${log}" | tail -1)
        spawn=$(sed -n 's/^spawned=\([0-9]*\).*/\1/p'                            "${log}" | tail -1)
        live=$( sed -n 's/^spawned=[0-9]* arrived=[0-9]* live=\([0-9]*\).*/\1/p' "${log}" | tail -1)
        lost=$( sed -n 's/.*lost=\([0-9]*\).*/\1/p'                              "${log}" | tail -1)
        blk=$(  sed -n 's/.*blocked=[0-9]* (\([0-9.]*\)%.*/\1/p'                 "${log}" | tail -1)
        cross=$(sed -n 's/^crossings=\([0-9]*\).*/\1/p'                          "${log}" | tail -1)
        age=$(  sed -n 's/.*consulted entry=\([0-9.]*\).*/\1/p'                  "${log}" | tail -1)
        smean=$(sed -n 's/.*staleness at t=[0-9]*: mean=\([0-9.]*\).*/\1/p'      "${log}" | tail -1)
        smax=$( sed -n 's/.*staleness.*max=\([0-9]*\).*/\1/p'                    "${log}" | tail -1)

        [[ -z "${dur}" && "${status}" == ok ]] && status="parse_fail"
        : "${dur:=nan}" "${hops:=nan}" "${arr:=-1}" "${spawn:=-1}" "${live:=-1}"
        : "${lost:=-1}" "${blk:=nan}" "${cross:=-1}" "${age:=nan}" "${smean:=nan}" "${smax:=-1}"

        # Prefer the program's own reported time; fall back to launcher wall clock.
        : "${prog_time:=${wall}}"

        echo "${np},${nodes},${ppn},${mode},${rep},${status},${prog_time},${dur},${hops},${arr},${spawn},${live},${lost},${blk},${cross},${age},${smean},${smax},${SEED},${STEPS},${CARS},${RATE},${CAP}" >> "${CSV}"

        if [[ "${status}" == ok ]]; then
                printf '%8ss  dur=%-7s hops=%-6s arr=%-7s blk=%-5s%%\n' "${prog_time}" "${dur}" "${hops}" "${arr}" "${blk}"
        else
                printf '%s (%s)\n' "${status}" "$(basename "${log}")"
        fi
}

# --------------------------------------------------------------------------
# Sweep: ranks (outer) x modes x reps.
# --------------------------------------------------------------------------
for np in ${RANKS}; do
        read -r nodes ppn < <(place_for "${np}")
        if [[ -z "${nodes:-}" ]]; then
                echo "  skip np=${np}: needs > ${AVAIL_NODES} nodes at ${MAX_PPN} ppn"
                continue
        fi
        echo "===== ranks=${np}  (${nodes}x${ppn}) ====="
        for mode in ${MODES}; do
                for (( rep = 1; rep <= REPS; rep++ )); do
                        run_one "${np}" "${nodes}" "${ppn}" "${mode}" "${rep}"
                done
        done
        echo
done

[[ ${DRY_RUN} -eq 1 ]] && { echo "dry run complete"; exit 0; }

echo "csv:  ${CSV}"
echo "logs: ${RAWDIR}"
echo
echo "quick median-of-reps view (arrived / mean_dur by ranks x mode):"
awk -F, 'NR>1 && $6=="ok" {
        key=$1"|"$4;
        dur[key]=dur[key]" "$8; arr[key]=arr[key]" "$10; age[key]=age[key]" "$16; sm[key]=sm[key]" "$18;
        seen[$1]=1; modes[$4]=1;
}
function med(s,   a,n,i) { n=split(s,a," "); if(!n) return "nan";
        for(i=1;i<=n;i++) for(j=i+1;j<=n;j++) if(a[j]<a[i]){t=a[i];a[i]=a[j];a[j]=t};
        return (n%2)? a[(n+1)/2] : (a[n/2]+a[n/2+1])/2 }
END {
        printf "%-6s %-7s %10s %10s %8s %8s\n","ranks","mode","mean_dur","arrived","age","stale_mx";
        n=asorti(seen,rk,"@val_num_asc");
        for(i=1;i<=n;i++){ r=rk[i];
                for(m in modes){ k=r"|"m;
                        if(k in dur) printf "%-6s %-7s %10s %10s %8s %8s\n", r,m,med(dur[k]),med(arr[k]),med(age[k]),med(sm[k]) }
        }
}' "${CSV}" 2>/dev/null || echo "  (install gawk for the summary view; raw CSV is complete)"
