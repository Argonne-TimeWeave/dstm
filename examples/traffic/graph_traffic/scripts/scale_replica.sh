#!/bin/bash
# scale_replica.sh -- scale ranks x replica-mode sweep for graph_traffic.
#
#   For each rank count in the ladder and each replica mode (blind stale
#   oracle), run ./traffic --data DIR and record runtime, early-departure
#   cohort routing metrics, and overall simulation totals to a CSV.
#   One row per (ranks, mode, stale_delay, rep).
#
#   Routing metrics are computed by traffic for cars departing during the
#   first 10% of simulation timestamps:
#     mean_dur, mean_hops, arrived_pct, cohort_arrived, cohort_spawned.
#
#   For mode=stale the run is repeated once per value in STALE_DELAYS
#   (default "2 8 32"); blind and oracle ignore --stale-delay and run once
#   (stale_delay logged as "na").
#
# LAUNCH RECIPE (matches run_scaling.sh / ROSS weak_scale.sh)
#   Under a batch allocation we launch with:
#     mpirun -n P --hostfile $NODEFILE --prefix $MPI_PREFIX \
#            --map-by ppr:PPN:node -x LD_LIBRARY_PATH -x PATH ./traffic ...
#   Interactively (no allocation) we fall back to: mpirun -n P ./traffic ...
#
# USAGE
#   ./scale_replica.sh                         # default ladder, all modes @ bloomington
#   RANKS="1 2 4 8" ./scale_replica.sh
#   ./scale_replica.sh --data data/bloomington/ --steps 200 --rate 1 --cap 16
#   ./scale_replica.sh --cars 2048 --rate 1 --reps 3
#   ./scale_replica.sh --stale-delays "1 4 16"
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
MODES="${MODES:-blind stale oracle}"
STALE_DELAYS="${STALE_DELAYS:-2 8 32}"       # only swept when mode=stale

STEPS="${STEPS:-256}"
CARS="${CARS:-256}"
RATE="${RATE:-256}"
CAP="${CAP:-32}"
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
  --stale-delays "2 8 32"  stale-delay sweep (stale mode only)  (default "${STALE_DELAYS}")
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
        --data)         DATA="$2"; shift 2 ;;
        --ranks)        RANKS="$2"; shift 2 ;;
        --modes)        MODES="$2"; shift 2 ;;
        --stale-delays) STALE_DELAYS="$2"; shift 2 ;;
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
        echo "stale delays: ${STALE_DELAYS}  (stale mode only)"
        echo "steps:        ${STEPS}   cars: ${CARS}   rate: ${RATE}   cap: ${CAP}   seed: ${SEED}"
        echo "reps:         ${REPS}    timeout: ${TIMEOUT_S}s   max ppn: ${MAX_PPN}"
} | tee "${CONFIG}"
echo

echo "ranks,nodes,ppn,mode,stale_delay,rep,status,run_time,mean_dur,mean_hops,arrived_pct,cohort_arrived,cohort_spawned,arrived,spawned,live,lost,blocked_pct,crossings,mean_age,stale_mean,stale_max,mib_tick_total,mib_tick_per_rank,seed,steps,cars,rate,cap" > "${CSV}"

# --------------------------------------------------------------------------
# One run.
# --------------------------------------------------------------------------
run_one() {
        local np="$1" nodes="$2" ppn="$3" mode="$4" delay="$5" rep="$6"
        local tag log launch=() cmd=() rc t0 t1 wall status delay_flag=() delay_lbl

        # Only stale mode consumes --stale-delay; others log "na".
        if [[ "${mode}" == "stale" ]]; then
                delay_flag=(--stale-delay "${delay}")
                delay_lbl="${delay}"
                tag="np${np}-n${nodes}x${ppn}-${mode}-d${delay}-r${rep}"
        else
                delay_lbl="na"
                tag="np${np}-n${nodes}x${ppn}-${mode}-r${rep}"
        fi
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
             --replica-mode "${mode}" "${delay_flag[@]}" ${VERBOSITY})

        printf '  np=%-4s %sx%-3s %-7s d=%-3s rep=%s ... ' "${np}" "${nodes}" "${ppn}" "${mode}" "${delay_lbl}" "${rep}"

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

                # Parse the routing-metric cohort and the overall simulation totals.
        #
        # Expected cohort line:
        #   routing metric: cohort_spawned=N cohort_arrived=N
        #   arrived_pct=P mean_dur=D mean_hops=H
        #
        # [^ ]* is used for floating-point metrics so "nan" is accepted when
        # no member of the cohort has arrived.
        local dur hops arrived_pct cohort_arr cohort_spawn
        local arr spawn live lost blk cross age smean smax prog_time mibt mibt_pr

        prog_time=$(sed -n 's/^total run time: \([0-9.]*\).*/\1/p'              "${log}" | tail -1)

        dur=$(sed -n \
                's/^routing metric: .*mean_dur=\([^ ]*\).*/\1/p'                \
                "${log}" | tail -1)
        hops=$(sed -n \
                's/^routing metric: .*mean_hops=\([^ ]*\).*/\1/p'               \
                "${log}" | tail -1)
        arrived_pct=$(sed -n \
                's/^routing metric: .*arrived_pct=\([^ ]*\).*/\1/p'             \
                "${log}" | tail -1)
        cohort_arr=$(sed -n \
                's/^routing metric: .*cohort_arrived=\([0-9]*\).*/\1/p'         \
                "${log}" | tail -1)
        cohort_spawn=$(sed -n \
                's/^routing metric: cohort_spawned=\([0-9]*\).*/\1/p'           \
                "${log}" | tail -1)

        # Overall simulation totals, not restricted to the metric cohort.
        arr=$(  sed -n 's/^spawned=[0-9]* arrived=\([0-9]*\).*/\1/p'             "${log}" | tail -1)
        spawn=$(sed -n 's/^spawned=\([0-9]*\).*/\1/p'                            "${log}" | tail -1)
        live=$( sed -n 's/^spawned=[0-9]* arrived=[0-9]* live=\([0-9]*\).*/\1/p' "${log}" | tail -1)
        lost=$( sed -n 's/.*lost=\([0-9]*\).*/\1/p'                              "${log}" | tail -1)
        blk=$(  sed -n 's/.*blocked=[0-9]* (\([0-9.]*\)%.*/\1/p'                 "${log}" | tail -1)
        cross=$(sed -n 's/^crossings=\([0-9]*\).*/\1/p'                          "${log}" | tail -1)
        age=$(  sed -n 's/.*consulted entry=\([0-9.]*\).*/\1/p'                  "${log}" | tail -1)
        smean=$(sed -n 's/.*staleness at t=[0-9]*: mean=\([0-9.]*\).*/\1/p'      "${log}" | tail -1)
        smax=$( sed -n 's/.*staleness.*max=\([0-9]*\).*/\1/p'                    "${log}" | tail -1)
        mibt=$( sed -n 's/.*(\([0-9.]*\) MiB\/tick across all ranks).*/\1/p'     "${log}" | tail -1)

        if [[ "${status}" == ok &&
              ( -z "${dur}" ||
                -z "${hops}" ||
                -z "${arrived_pct}" ||
                -z "${cohort_arr}" ||
                -z "${cohort_spawn}" ) ]]; then
                status="parse_fail"
        fi

        : "${dur:=nan}" "${hops:=nan}" "${arrived_pct:=nan}"
        : "${cohort_arr:=-1}" "${cohort_spawn:=-1}"
        : "${arr:=-1}" "${spawn:=-1}" "${live:=-1}" "${lost:=-1}"
        : "${blk:=nan}" "${cross:=-1}" "${age:=nan}"
        : "${smean:=nan}" "${smax:=-1}" "${mibt:=nan}"

        # Aggregate MiB/tick -> per-rank.
        if [[ "${mibt}" != "nan" ]]; then
                mibt_pr=$(awk -v t="${mibt}" -v n="${np}" 'BEGIN { if (n>0) printf "%.6f", t/n; else print "nan" }')
        else
                mibt_pr="nan"
        fi

        # Prefer the program's own reported time; fall back to launcher wall clock.
        : "${prog_time:=${wall}}"

        echo "${np},${nodes},${ppn},${mode},${delay_lbl},${rep},${status},${prog_time},${dur},${hops},${arrived_pct},${cohort_arr},${cohort_spawn},${arr},${spawn},${live},${lost},${blk},${cross},${age},${smean},${smax},${mibt},${mibt_pr},${SEED},${STEPS},${CARS},${RATE},${CAP}" >> "${CSV}"

        if [[ "${status}" == ok ]]; then
                printf '%8ss  dur=%-7s hops=%-6s cohort=%s/%s (%s%%) blk=%-5s%% mib/t=%-7s\n' \
                        "${prog_time}" "${dur}" "${hops}" \
                        "${cohort_arr}" "${cohort_spawn}" "${arrived_pct}" \
                        "${blk}" "${mibt}"
        else
                printf '%s (%s)\n' "${status}" "$(basename "${log}")"
        fi
}

# --------------------------------------------------------------------------
# Sweep: ranks (outer) x modes x stale_delays x reps.
#   stale mode -> one pass per value in STALE_DELAYS
#   other modes -> single pass (delay ignored)
# --------------------------------------------------------------------------
for np in ${RANKS}; do
        read -r nodes ppn < <(place_for "${np}")
        if [[ -z "${nodes:-}" ]]; then
                echo "  skip np=${np}: needs > ${AVAIL_NODES} nodes at ${MAX_PPN} ppn"
                continue
        fi
        echo "===== ranks=${np}  (${nodes}x${ppn}) ====="
        for mode in ${MODES}; do
                if [[ "${mode}" == "stale" ]]; then
                        delays="${STALE_DELAYS}"
                else
                        delays="na"        # sentinel: one pass, delay unused
                fi
                for delay in ${delays}; do
                        for (( rep = 1; rep <= REPS; rep++ )); do
                                run_one "${np}" "${nodes}" "${ppn}" "${mode}" "${delay}" "${rep}"
                        done
                done
        done
        echo
done

[[ ${DRY_RUN} -eq 1 ]] && { echo "dry run complete"; exit 0; }

echo "csv:  ${CSV}"
echo "logs: ${RAWDIR}"
echo
echo "quick median-of-reps view (early-departure cohort by ranks x mode x delay):"
awk -F, 'NR>1 && $7=="ok" {
        key=$1"|"$4"|"$5;

        dur[key]=dur[key]" "$9;
        hops[key]=hops[key]" "$10;
        pct[key]=pct[key]" "$11;
        age[key]=age[key]" "$20;
        smax[key]=smax[key]" "$22;
        mib[key]=mib[key]" "$23;

        seen[$1]=1;
        combo[$4"|"$5]=1;
}
function med(s,   a,n,i,j,t) {
        n=split(s,a," ");
        if(!n) return "nan";
        for(i=1;i<=n;i++)
                for(j=i+1;j<=n;j++)
                        if(a[j]<a[i]) {
                                t=a[i];
                                a[i]=a[j];
                                a[j]=t
                        }
        return (n%2) ? a[(n+1)/2] : (a[n/2]+a[n/2+1])/2
}
END {
        printf "%-6s %-7s %-6s %10s %10s %11s %8s %8s %10s\n",
               "ranks", "mode", "delay",
               "mean_dur", "mean_hops", "arrived_pct",
               "age", "stale_mx", "mib/tick";

        n=asorti(seen,rk,"@val_num_asc");
        for(i=1;i<=n;i++) {
                r=rk[i];
                for(c in combo) {
                        split(c,cc,"|");
                        k=r"|"c;
                        if(k in dur)
                                printf "%-6s %-7s %-6s %10s %10s %11s %8s %8s %10s\n",
                                       r, cc[1], cc[2],
                                       med(dur[k]), med(hops[k]), med(pct[k]),
                                       med(age[k]), med(smax[k]), med(mib[k])
                }
        }
}' "${CSV}" 2>/dev/null ||
        echo "  (install gawk for the summary view; raw CSV is complete)"
