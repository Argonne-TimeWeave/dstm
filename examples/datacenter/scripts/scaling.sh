#!/usr/bin/env bash
set -uo pipefail

# Usage:
#   ./scripts/scaling.sh strong
#   ./scripts/scaling.sh weak
#   ./scripts/scaling.sh both
#
# Optional environment variables:
#   REPEATS=3 STEPS=300 STRONG_JOBS=1250 WEAK_JOBS_PER_RANK=125
#   MPIRUN_ARGS="--oversubscribe"
#
# RANKS denotes simulation ranks:
#
#   dstm         : RANKS total MPI ranks
#   sst          : RANKS total MPI ranks
#   opt_dstm : RANKS thermal ranks + 1 dedicated scheduler rank
#   opt_sst  : RANKS thermal ranks + 1 dedicated scheduler rank
#
# opt_dstm reserves one node exclusively for scheduler rank 0.

MODE="${1:-both}"
#RANKS=(2 4 8 16 32 64 128 256)
#RANKS=(128)
# Example: RANKS_OVERRIDE="32"
read -r -a RANKS <<< "${RANKS_OVERRIDE:-2 4 8 16 32 64 128 256}"

# Space-separated subset: "dstm sst opt_dstm opt_sst"
SYSTEMS="${SYSTEMS:-dstm sst opt_dstm opt_sst}"

enabled()
{
    [[ " ${SYSTEMS} " == *" $1 "* ]]
}

REPEATS="${REPEATS:-2}"
STEPS="${STEPS:-1000}"
NODES_PER_RACK="${NODES_PER_RACK:-32}"

SCHEDULE_WINDOW="${SCHEDULE_WINDOW:-60}"
CONTROL_LATENCY="${CONTROL_LATENCY:-30}"
TREE_FANOUT="${TREE_FANOUT:-8}"   # K=8 gives depth 1-3 for 2..256 ranks; all divide 30

STRONG_RACKS="${STRONG_RACKS:-512}"
STRONG_JOBS="${STRONG_JOBS:-32768}"

WEAK_RACKS_PER_RANK="${WEAK_RACKS_PER_RANK:-8}"
WEAK_JOBS_PER_RANK="${WEAK_JOBS_PER_RANK:-512}"

export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-0}"

MPIRUN="${MPIRUN:-/opt/cray/pals/1.8/bin/mpiexec}"
MPIRUN_ARGS="${MPIRUN_ARGS:-}"
MPI_FLAVOR="${MPI_FLAVOR:-pals}"

MAX_PPN="${MAX_PPN:-32}"

# Maximum nodes used by simulation/thermal ranks. opt_dstm may use one
# additional node for its dedicated scheduler.
MAX_WORKER_NODES="${MAX_WORKER_NODES:-${MAX_NODES:-8}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DSTM_EXE="${ROOT_DIR}/dstm/datacenter"
OPT_DSTM_EXE="${ROOT_DIR}/opt_dstm/datacenter"

SST_DIR="${ROOT_DIR}/sst"
OPT_SST_DIR="${ROOT_DIR}/opt_sst"

GENERATOR="${ROOT_DIR}/jobs/gen.py"

RESULT_DIR="${SCRIPT_DIR}/results"
TRACE_DIR="${RESULT_DIR}/traces"
LOG_DIR="${RESULT_DIR}/logs"
RESULT_CSV="${RESULT_DIR}/scaling.csv"

# Discover the nodes assigned by the batch scheduler. The precedence matches
# the launch setup used by the other scaling scripts.
ALLOC_NODEFILE=""
ALLOC_SRC="none"

if [[ -n "${COBALT_NODEFILE:-}" &&
      -r "${COBALT_NODEFILE}" ]]; then
    ALLOC_NODEFILE="${COBALT_NODEFILE}"
    ALLOC_SRC="COBALT_NODEFILE"
elif [[ -n "${PBS_NODEFILE:-}" &&
        -r "${PBS_NODEFILE}" ]]; then
    ALLOC_NODEFILE="${PBS_NODEFILE}"
    ALLOC_SRC="PBS_NODEFILE"
elif [[ -n "${HOSTFILE:-}" &&
        -r "${HOSTFILE}" ]]; then
    ALLOC_NODEFILE="${HOSTFILE}"
    ALLOC_SRC="HOSTFILE"
elif [[ -n "${SLURM_JOB_NODELIST:-}" ]] &&
     command -v scontrol >/dev/null 2>&1; then
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

ALLOCATED_NODES=${#NODELIST[@]}

if (( ALLOCATED_NODES < MAX_WORKER_NODES + 1 )); then
    echo "Warning: allocated ${ALLOCATED_NODES} nodes; opt_dstm reserves one scheduler node and can use at most $((ALLOCATED_NODES - 1)) worker nodes" >&2
fi

# Baseline runs use at most MAX_WORKER_NODES. The epoch implementation may
# use one additional allocated node for its scheduler.
AVAIL_WORKER_NODES="${ALLOCATED_NODES}"
(( AVAIL_WORKER_NODES > MAX_WORKER_NODES )) &&
    AVAIL_WORKER_NODES="${MAX_WORKER_NODES}"

# PALS obtains its launch environment from global --env/--envlist options.

# Select the number of nodes and ranks per node for a rank count.
#
# With at least 8 allocated nodes and MAX_NODES=8:
#   ranks=4  -> 4 nodes x 1 rank/node
#   ranks=8  -> 8 nodes x 1 rank/node
#   ranks=16 -> 8 nodes x 2 ranks/node
#   ranks=32 -> 8 nodes x 4 ranks/node
place_for()
{
    local ranks="$1"
    local available_nodes="$2"
    local node_limit="$3"
    local nodes ppn capacity

    nodes="${ranks}"
    (( nodes > available_nodes )) && nodes="${available_nodes}"
    (( nodes > node_limit )) && nodes="${node_limit}"
    (( nodes < 1 )) && nodes=1

    ppn=$(( (ranks + nodes - 1) / nodes ))
    capacity=$(( nodes * MAX_PPN ))

    if (( ranks > capacity || ppn > MAX_PPN )); then
        echo ""
        return
    fi

    echo "${nodes} ${ppn}"
}

# Construct the MPI launcher for one rank count. MPI_LAUNCH is an array so
# hostfile, prefix, mapping, and environment-forwarding arguments retain
# their argument boundaries.
MPI_LAUNCH=()

make_mpi_launch()
{
    local ranks="$1"
    local model_options="${2:-}"
    local nodes ppn
    local mpi_args=()

    read -r nodes ppn < <(
        place_for "${ranks}" \
            "${AVAIL_WORKER_NODES}" \
            "${MAX_WORKER_NODES}"
    )

    [[ -n "${nodes}" ]] || return 1

    # MPIRUN_ARGS must contain PALS global options only.
    # shellcheck disable=SC2206
    mpi_args=( ${MPIRUN_ARGS} )

    MPI_LAUNCH=(
        "${MPIRUN}"
        "${mpi_args[@]}"
        --hostfile "${ALLOC_NODEFILE}"
        --envlist "PATH,LD_LIBRARY_PATH"
        --env "MPICH_GPU_SUPPORT_ENABLED=0"
    )

    if [[ -n "${model_options}" ]]; then
        MPI_LAUNCH+=(
            --env "SST_MODEL_OPTIONS=${model_options}"
        )
    fi

    MPI_LAUNCH+=(
        -n "${ranks}"
        --ppn "${ppn}"
    )
}

OPT_MPI_LAUNCH=()
OPT_MPI_SCHED=()
OPT_MPI_WORKER=()

make_opt_mpi_launch()
{
    local worker_ranks="$1"
    local worker_nodes ppn capacity
    local mpi_args=()
    local hosts
    local i

    if (( ALLOCATED_NODES < 2 )); then
        echo "Optimized runs require at least two allocated nodes" >&2
        return 1
    fi

    worker_nodes="${worker_ranks}"
    (( worker_nodes > MAX_WORKER_NODES )) &&
        worker_nodes="${MAX_WORKER_NODES}"
    (( worker_nodes > ALLOCATED_NODES - 1 )) &&
        worker_nodes=$((ALLOCATED_NODES - 1))
    (( worker_nodes < 1 )) && worker_nodes=1

    ppn=$(( (worker_ranks + worker_nodes - 1) / worker_nodes ))
    capacity=$((worker_nodes * MAX_PPN))

    if (( worker_ranks > capacity || ppn > MAX_PPN )); then
        echo "Cannot fit ${worker_ranks} workers on ${worker_nodes} nodes at ${MAX_PPN} ranks/node" >&2
        return 1
    fi

    hosts="${NODELIST[0]}"
    for ((i = 1; i <= worker_nodes; i++)); do
        hosts+=",${NODELIST[i]}"
    done

    # MPIRUN_ARGS must contain PALS global options only.
    # shellcheck disable=SC2206
    mpi_args=( ${MPIRUN_ARGS} )

    OPT_MPI_LAUNCH=(
        "${MPIRUN}"
        "${mpi_args[@]}"
        --hosts "${hosts}"
        --envlist "PATH,LD_LIBRARY_PATH"
        --env "MPICH_GPU_SUPPORT_ENABLED=0"
    )

    # PALS MPMD command groups. The first group consumes NODELIST[0];
    # the worker group continues on NODELIST[1] and later nodes.
    OPT_MPI_SCHED=(-n 1 --ppn 1)
    OPT_MPI_WORKER=(-n "${worker_ranks}" --ppn "${ppn}")
}

mkdir -p "${TRACE_DIR}" "${LOG_DIR}"

if [[ ! -x "${DSTM_EXE}" ]]; then
    echo "Missing executable: ${DSTM_EXE}" >&2
    exit 1
fi

if [[ ! -x "${OPT_DSTM_EXE}" ]]; then
    echo "Missing executable: ${OPT_DSTM_EXE}" >&2
    exit 1
fi

if [[ ! -f "${SST_DIR}/run.py" ]]; then
    echo "Missing SST model: ${SST_DIR}/run.py" >&2
    exit 1
fi

if [[ ! -f "${OPT_SST_DIR}/run.py" ]]; then
    echo "Missing optimized SST model: ${OPT_SST_DIR}/run.py" >&2
    exit 1
fi

if [[ ! -f "${GENERATOR}" ]]; then
    echo "Missing job generator: ${GENERATOR}" >&2
    exit 1
fi

if [[ "${MODE}" != "strong" &&
      "${MODE}" != "weak" &&
      "${MODE}" != "both" ]]; then
    echo "Usage: $0 {strong|weak|both}" >&2
    exit 1
fi

printf '%s\n' \
    "mode,system,ranks,mpi_ranks,racks,jobs,steps,repeat,runtime_seconds,status" \
    > "${RESULT_CSV}"

generate_trace()
{
    local output="$1"
    local jobs="$2"

    python3 "${GENERATOR}" \
        --output "${output}" \
        --jobs "${jobs}" \
        --simulation-seconds "${STEPS}" \
        --max-nodes 8 \
        --seed 42 \
        >/dev/null
}

# Epoch-based DSTM: 1 scheduler rank + N thermal ranks (MPMD).
# usage: run_epoch_dstm SYSTEM EXE MODE RANKS RACKS JOBS TRACE REPEAT [extra args...]
run_epoch_dstm()
{
    local system="$1" exe="$2" scaling_mode="$3" worker_ranks="$4"
    local racks="$5" jobs="$6" trace="$7" repeat="$8"
    shift 8
    local extra=("$@")

    local mpi_ranks=$((worker_ranks + 1))
    local log="${LOG_DIR}/${scaling_mode}_${system}_r${worker_ranks}_run${repeat}.log"
    local elapsed="nan"
    local status
    local args=( --racks "${racks}" --nodes-per-rack "${NODES_PER_RACK}"
                 --steps "${STEPS}" --schedule-window "${SCHEDULE_WINDOW}"
                 --control-latency "${CONTROL_LATENCY}" --jobs "${trace}"
                 "${extra[@]}" )

    if ! make_opt_mpi_launch "${worker_ranks}"; then
        status="placement_failed"
    elif "${OPT_MPI_LAUNCH[@]}" "${OPT_MPI_SCHED[@]}" "${exe}" "${args[@]}" \
         : "${OPT_MPI_WORKER[@]}" "${exe}" "${args[@]}" >"${log}" 2>&1; then
        if elapsed="$(
            awk '
                /^(simulation runtime|wall-clock runtime)[[:space:]]*:/ {
                    value = $(NF - 1); found = 1 }
                END { if (!found) exit 1; printf "%.9f\n", value }
            ' "${log}"
        )"; then status="ok"; else status="missing_timing"; fi
    else
        status="failed"
    fi

    printf '%s,%s,%d,%d,%d,%d,%d,%d,%s,%s\n' \
        "${scaling_mode}" "${system}" "${worker_ranks}" "${mpi_ranks}" \
        "${racks}" "${jobs}" "${STEPS}" "${repeat}" "${elapsed}" "${status}" \
        >> "${RESULT_CSV}"

    echo "${scaling_mode}, ${system}, thermal_ranks=${worker_ranks}, mpi_ranks=${mpi_ranks}, run=${repeat}: ${elapsed}s (${status})"
    [[ "${status}" == "ok" ]]
}

run_dstm()     { run_epoch_dstm dstm     "${DSTM_EXE}"     "$@"; }
run_opt_dstm() { run_epoch_dstm opt_dstm "${OPT_DSTM_EXE}" "$@" --tree-fanout "${TREE_FANOUT}"; }

# Epoch-based SST: 1 scheduler rank + N domains.
# usage: run_epoch_sst SYSTEM DIR MODE RANKS RACKS JOBS TRACE REPEAT
run_epoch_sst()
{
    local system="$1" dir="$2" scaling_mode="$3" worker_ranks="$4"
    local racks="$5" jobs="$6" trace="$7" repeat="$8"

    local mpi_ranks=$((worker_ranks + 1))
    local log="${LOG_DIR}/${scaling_mode}_${system}_r${worker_ranks}_run${repeat}.log"
    local elapsed="nan"
    local status

    local options="--racks ${racks} --domains ${worker_ranks} --sst-ranks ${mpi_ranks}"
    options+=" --nodes-per-rack ${NODES_PER_RACK} --steps ${STEPS}"
    options+=" --schedule-window ${SCHEDULE_WINDOW} --control-latency ${CONTROL_LATENCY}"
    options+=" --jobs ${trace}"

    if ! make_opt_mpi_launch "${worker_ranks}"; then
        status="placement_failed"
    elif (
        cd "${dir}" &&
        "${OPT_MPI_LAUNCH[@]}" --env "SST_MODEL_OPTIONS=${options}" \
            "${OPT_MPI_SCHED[@]}" sst run.py \
            : "${OPT_MPI_WORKER[@]}" sst run.py
    ) >"${log}" 2>&1; then
        if elapsed="$(
            awk '
                { for (i = 1; i <= NF; i++)
                    if ($i ~ /^simulation_runtime_seconds=/) {
                        split($i, v, "="); t = v[2] + 0.0
                        if (!found || t > m) m = t
                        found = 1 } }
                END { if (!found) exit 1; printf "%.9f\n", m }
            ' "${log}"
        )"; then status="ok"; else status="missing_timing"; fi
    else
        status="failed"
    fi

    printf '%s,%s,%d,%d,%d,%d,%d,%d,%s,%s\n' \
        "${scaling_mode}" "${system}" "${worker_ranks}" "${mpi_ranks}" \
        "${racks}" "${jobs}" "${STEPS}" "${repeat}" "${elapsed}" "${status}" \
        >> "${RESULT_CSV}"

    echo "${scaling_mode}, ${system}, thermal_ranks=${worker_ranks}, mpi_ranks=${mpi_ranks}, run=${repeat}: ${elapsed}s (${status})"
    [[ "${status}" == "ok" ]]
}

run_sst()     { run_epoch_sst sst     "${SST_DIR}"     "$@"; }
run_opt_sst() { run_epoch_sst opt_sst "${OPT_SST_DIR}" "$@"; }

run_case()
{
    local scaling_mode="$1"
    local ranks="$2"
    local racks jobs trace

    if [[ "${scaling_mode}" == "strong" ]]; then
        racks="${STRONG_RACKS}"
        jobs="${STRONG_JOBS}"
        trace="${TRACE_DIR}/strong.csv"
    else
        racks=$((WEAK_RACKS_PER_RANK * ranks))
        jobs=$((WEAK_JOBS_PER_RANK * ranks))
        trace="${TRACE_DIR}/weak_r${ranks}.csv"
    fi

    if (( racks % ranks != 0 )); then
        echo "Rack count ${racks} is not divisible by ${ranks}" >&2
        exit 1
    fi

    generate_trace "${trace}" "${jobs}"

    for ((repeat = 1; repeat <= REPEATS; repeat++)); do
        enabled dstm && run_dstm \
            "${scaling_mode}" "${ranks}" "${racks}" \
            "${jobs}" "${trace}" "${repeat}" || true

        enabled opt_dstm && run_opt_dstm \
            "${scaling_mode}" "${ranks}" "${racks}" \
            "${jobs}" "${trace}" "${repeat}" || true

        enabled sst && run_sst \
            "${scaling_mode}" "${ranks}" "${racks}" \
            "${jobs}" "${trace}" "${repeat}" || true

        enabled opt_sst && run_opt_sst \
            "${scaling_mode}" "${ranks}" "${racks}" \
            "${jobs}" "${trace}" "${repeat}" || true
    done
}

for ranks in "${RANKS[@]}"; do
    if [[ "${MODE}" == "strong" || "${MODE}" == "both" ]]; then
        run_case strong "${ranks}"
    fi

    if [[ "${MODE}" == "weak" || "${MODE}" == "both" ]]; then
        run_case weak "${ranks}"
    fi
done

echo "Results: ${RESULT_CSV}"
echo "Logs:    ${LOG_DIR}"
