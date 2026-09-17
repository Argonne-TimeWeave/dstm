#!/bin/bash
# scale_stale.sh -- fix ranks, sweep --stale-delay for graph_traffic.
#
#   For a single fixed rank count, run ./traffic --replica-mode stale over
#   a ladder of --stale-delay values and record early-departure cohort
#   routing metrics, overall arrivals, and staleness stats to a CSV.
#   One row per (stale_delay, rep).
#
#   Routing metrics are computed by traffic for cars departing during the
#   first 10% of simulation timestamps:
#     mean_dur, mean_hops, arrived_pct, cohort_arrived, cohort_spawned.
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
#   stale-results/<stamp>/runs.csv    one aggregate row per run
#   stale-results/<stamp>/raw/*.log   full stdout+stderr per run
#   stale-results/<stamp>/trips/*.csv one row per simulated trip
#   stale-results/<stamp>/config.txt  what was launched
#
# Trip tables contain the fixed trip ID, initial routed travel-time
# estimate, experienced travel time, prediction gap, distance, VMT, and
# VHT. Matching trip IDs can be compared across stale delays and against
# the oracle/blind baselines.
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

STEPS="${STEPS:-2048}"
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

# Pick nodes/ppn for a rank count: spread across as many allocated nodes as
# possible, up to MAX_NODES, before placing multiple ranks on each node.
#
# With at least 8 allocated nodes and MAX_NODES=8:
#   np=4   -> 4 nodes x 1 rank/node
#   np=8   -> 8 nodes x 1 rank/node
#   np=16  -> 8 nodes x 2 ranks/node
#   np=256 -> 8 nodes x 32 ranks/node
#
# Returns "nodes ppn" or "" if the rank count cannot fit.
place_for() {
        local np="$1" nodes ppn capacity

        # Use one node per rank until all available nodes are occupied.
        nodes="${np}"
        (( nodes > AVAIL_NODES )) && nodes="${AVAIL_NODES}"
        (( nodes > MAX_NODES )) && nodes="${MAX_NODES}"
        (( nodes < 1 )) && nodes=1

        # Then distribute additional ranks evenly across those nodes.
        ppn=$(( (np + nodes - 1) / nodes ))

        capacity=$(( nodes * MAX_PPN ))
        (( np > capacity || ppn > MAX_PPN )) && {
                echo ""
                return
        }

        echo "${nodes} ${ppn}"
}

# --------------------------------------------------------------------------
# Output tree.
# --------------------------------------------------------------------------
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTDIR="${OUTROOT}/${STAMP}"
RAWDIR="${OUTDIR}/raw"
TRIPDIR="${OUTDIR}/trips"
CSV="${OUTDIR}/runs.csv"
CONFIG="${OUTDIR}/config.txt"

mkdir -p "${RAWDIR}" "${TRIPDIR}"

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
        echo "trip tables:  ${TRIPDIR}"
        echo "mpi prefix:   ${MPI_PREFIX:-<unset>}   forwarded: ${MPI_ENV_VARS}"
        echo "ranks:        ${RANKS}  (${NODES}x${PPN})   mode: ${MODE}"
        echo "stale delays: ${STALE_DELAYS}"
        echo "oracle base:  $([[ ${ORACLE_BASELINE} -eq 1 ]] && echo "yes (REPS=${REPS})" || echo "no")"
        echo "blind base:   $([[ ${BLIND_BASELINE} -eq 1 ]] && echo "yes (REPS=${REPS})" || echo "no")"
        echo "steps:        ${STEPS}   global cars: ${CARS}   global rate: ${RATE}   cap: ${CAP}   seed: ${SEED}"
        echo "reps:         ${REPS}    timeout: ${TIMEOUT_S}s   max ppn: ${MAX_PPN}"
} | tee "${CONFIG}"
echo

echo "ranks,mode,stale_delay,rep,status,mean_dur,mean_hops,arrived_pct,cohort_arrived,cohort_spawned,arrived,mean_age,stale_max,run_time,mib_tick_total,mib_tick_per_rank,trip_table" > "${CSV}"

# --------------------------------------------------------------------------
# One run.
# --------------------------------------------------------------------------
run_one() {
        local np="$1" nodes="$2" ppn="$3" mode="$4" sd="$5" rep="$6"
        local tag log trip_file launch=() cmd=() rc status
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
        trip_file="${TRIPDIR}/${tag}.csv"


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
             --replica-mode "${mode}" --stale-delay "${sd_arg}"
             --trip-output "${trip_file}" ${VERBOSITY})

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

                # Parse the early-departure routing cohort, overall arrivals,
        # staleness, runtime, and communication volume.
        #
        # Expected cohort line:
        #   routing metric: cohort_spawned=N cohort_arrived=N
        #   arrived_pct=P mean_dur=D mean_hops=H
        #
        # Floating-point fields use [^ ]* so "nan" is accepted when no
        # member of the cohort has arrived.
        local dur hops arrived_pct cohort_arr cohort_spawn
        local arr age smax rt mibt mibt_pr

        dur=$(sed -n \
                's/^routing metric: .*mean_dur=\([^ ]*\).*/\1/p'             \
                "${log}" | tail -1)
        hops=$(sed -n \
                's/^routing metric: .*mean_hops=\([^ ]*\).*/\1/p'            \
                "${log}" | tail -1)
        arrived_pct=$(sed -n \
                's/^routing metric: .*arrived_pct=\([^ ]*\).*/\1/p'          \
                "${log}" | tail -1)
        cohort_arr=$(sed -n \
                's/^routing metric: .*cohort_arrived=\([0-9]*\).*/\1/p'      \
                "${log}" | tail -1)
        cohort_spawn=$(sed -n \
                's/^routing metric: cohort_spawned=\([0-9]*\).*/\1/p'        \
                "${log}" | tail -1)

        # Overall arrival count, not restricted to the routing cohort.
        arr=$(sed -n \
                's/^spawned=[0-9]* arrived=\([0-9]*\).*/\1/p'                \
                "${log}" | tail -1)

        age=$(sed -n \
                's/.*consulted entry=\([0-9.]*\).*/\1/p'                     \
                "${log}" | tail -1)
        smax=$(sed -n \
                's/.*staleness.*max=\([0-9]*\).*/\1/p'                       \
                "${log}" | tail -1)
        rt=$(sed -n \
                's/^total run time:[[:space:]]*\([0-9.]*\).*/\1/p'           \
                "${log}" | tail -1)
        mibt=$(sed -n \
                's/.*(\([0-9.]*\) MiB\/tick across all ranks).*/\1/p'        \
                "${log}" | tail -1)

        # Derive per-rank MiB/tick = aggregate / rank count.
        if [[ -n "${mibt}" ]]; then
                mibt_pr=$(awk -v t="${mibt}" -v n="${np}" \
                        'BEGIN { if (n > 0) printf "%.4f", t/n; else print "nan" }')
        fi

        # A successful run must contain the complete routing-metric line.
        # "nan" is a valid parsed value; an empty value indicates failure.
        if [[ "${status}" == ok &&
              ( -z "${dur}" ||
                -z "${hops}" ||
                -z "${arrived_pct}" ||
                -z "${cohort_arr}" ||
                -z "${cohort_spawn}" ) ]]; then
                status="parse_fail"
        fi

        # The per-trip table is now part of every successful result. A
        # nonempty file must contain at least its CSV header. Mark the run
        # separately from an aggregate-output parse failure when it is
        # absent.
        if [[ "${status}" == ok && ! -s "${trip_file}" ]]; then
                status="trip_output_missing"
        fi

        : "${dur:=nan}" "${hops:=nan}" "${arrived_pct:=nan}"
        : "${cohort_arr:=-1}" "${cohort_spawn:=-1}" "${arr:=-1}"
        : "${age:=nan}" "${smax:=-1}" "${rt:=nan}"
        : "${mibt:=nan}" "${mibt_pr:=nan}"

        echo "${np},${mode},${sd_csv},${rep},${status},${dur},${hops},${arrived_pct},${cohort_arr},${cohort_spawn},${arr},${age},${smax},${rt},${mibt},${mibt_pr},trips/${tag}.csv" >> "${CSV}"

        if [[ "${status}" == ok ]]; then
                printf 'dur=%-8s hops=%-7s cohort=%s/%s (%s%%) age=%-6s stale_mx=%-3s rt=%-8s MiB/tick=%-8s per_rank=%s\n' \
                       "${dur}" "${hops}" \
                       "${cohort_arr}" "${cohort_spawn}" "${arrived_pct}" \
                       "${age}" "${smax}" "${rt}" "${mibt}" "${mibt_pr}"
        else
                printf '%s (log=%s trips=%s)\n' \
                       "${status}" \
                       "$(basename "${log}")" \
                       "$(basename "${trip_file}")"
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
echo "trips: ${TRIPDIR}"
echo
echo "quick median-of-reps view (early-departure cohort by stale delay):"
awk -F, 'NR > 1 && $5 == "ok" {
        # For stale rows, key on stale_delay. For na-keyed oracle/blind
        # baselines, key on mode so the two baselines do not collide.
        key = ($3 == "na") ? $2 : $3;

        dur[key]  = dur[key]  " " $6;
        hops[key] = hops[key] " " $7;
        pct[key]  = pct[key]  " " $8;
        age[key]  = age[key]  " " $12;
        smax[key] = smax[key] " " $13;
        rt[key]   = rt[key]   " " $14;
        mibt[key] = mibt[key] " " $15;
        mibp[key] = mibp[key] " " $16;

        if ($3 == "na")
                base[$2] = 1;
        else
                seen[key] = 1;
}
function med(s,   a,v,n,m,i,j,t) {
        # Ignore "nan" values when calculating a median. This matters when
        # a cohort has no arrivals and therefore has no defined duration or
        # hop mean.
        n = split(s, a, " ");
        m = 0;
        for (i = 1; i <= n; i++) {
                if (a[i] == "" || tolower(a[i]) == "nan")
                        continue;
                v[++m] = a[i] + 0;
        }

        if (m == 0)
                return "nan";

        for (i = 1; i <= m; i++)
                for (j = i + 1; j <= m; j++)
                        if (v[j] < v[i]) {
                                t = v[i];
                                v[i] = v[j];
                                v[j] = t;
                        }

        return (m % 2) ? v[(m + 1) / 2]
                       : (v[m / 2] + v[m / 2 + 1]) / 2;
}
function print_row(label, note) {
        printf "%-12s %10s %10s %11s %8s %9s %10s %12s %12s%s\n",
               label,
               med(dur[label]),
               med(hops[label]),
               med(pct[label]),
               med(age[label]),
               med(smax[label]),
               med(rt[label]),
               med(mibt[label]),
               med(mibp[label]),
               note;
}
END {
        printf "%-12s %10s %10s %11s %8s %9s %10s %12s %12s\n",
               "stale_delay",
               "mean_dur",
               "mean_hops",
               "arrived_pct",
               "age",
               "stale_mx",
               "run_time",
               "MiB/tick",
               "per_rank";

        n = asorti(seen, sk, "@val_num_asc");
        for (i = 1; i <= n; i++)
                print_row(sk[i], "");

        if ("oracle" in base)
                print_row("oracle", "   <- always-fresh upper bound");

        if ("blind" in base)
                print_row("blind", "   <- never-merge lower bound");
}' "${CSV}" 2>/dev/null ||
        echo "  (install gawk for the summary view; raw CSV is complete)"
