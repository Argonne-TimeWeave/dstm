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
#
# opt_dstm reserves one node exclusively for scheduler rank 0.

MODE="${1:-both}"
RANKS=(2 4 8 16 32 64 128 256)
#RANKS=(128)

REPEATS="${REPEATS:-2}"
STEPS="${STEPS:-1000}"
NODES_PER_RACK="${NODES_PER_RACK:-32}"

SCHEDULE_WINDOW="${SCHEDULE_WINDOW:-60}"
CONTROL_LATENCY="${CONTROL_LATENCY:-30}"

STRONG_RACKS="${STRONG_RACKS:-512}"
STRONG_JOBS="${STRONG_JOBS:-32768}"

WEAK_RACKS_PER_RANK="${WEAK_RACKS_PER_RANK:-8}"
WEAK_JOBS_PER_RANK="${WEAK_JOBS_PER_RANK:-512}"

MPIRUN="${MPIRUN:-mpirun}"
MPIRUN_ARGS="${MPIRUN_ARGS:-}"

MPI_PREFIX="${MPI_PREFIX:-}"
MPI_ENV_VARS="${MPI_ENV_VARS:-LD_LIBRARY_PATH PATH}"
MAX_PPN="${MAX_PPN:-32}"

# Maximum nodes used by simulation/thermal ranks. opt_dstm may use one
# additional node for its dedicated scheduler.
MAX_WORKER_NODES="${MAX_WORKER_NODES:-${MAX_NODES:-8}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DSTM_EXE="${ROOT_DIR}/dstm/datacenter"
AUCTION_DSTM_EXE="${ROOT_DIR}/opt_dstm/datacenter"
SST_DIR="${ROOT_DIR}/sst"
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

# Open MPI needs its installation prefix on remote nodes so it can locate
# its helper executables, including orted/prted.
if [[ -z "${MPI_PREFIX}" ]]; then
    _mp="$(command -v "${MPIRUN}" 2>/dev/null)"
    if [[ -n "${_mp}" ]]; then
        MPI_PREFIX="$(dirname "$(dirname "${_mp}")")"
    fi
fi

MPI_X_FLAGS=()
for _v in ${MPI_ENV_VARS}; do
    MPI_X_FLAGS+=(-x "${_v}")
done

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
    local nodes="" ppn=""
    local mpi_args=()

    read -r nodes ppn < <(place_for "${ranks}" "${AVAIL_WORKER_NODES}" "${MAX_WORKER_NODES}")

    if [[ -z "${nodes}" ]]; then
        echo "Rank count ${ranks} cannot fit on ${AVAIL_WORKER_NODES} worker nodes at ${MAX_PPN} ranks/node" >&2
        exit 1
    fi

    # Preserve the existing MPIRUN_ARGS environment variable.
    # shellcheck disable=SC2206
    mpi_args=( ${MPIRUN_ARGS} )

    MPI_LAUNCH=("${MPIRUN}" "${mpi_args[@]}" -n "${ranks}")

    if [[ "${ALLOC_SRC}" != "localhost" ]]; then
        MPI_LAUNCH+=(--hostfile "${ALLOC_NODEFILE}")

        if [[ -n "${MPI_PREFIX}" ]]; then
            MPI_LAUNCH+=(--prefix "${MPI_PREFIX}")
        fi

        if (( ranks > 1 && ranks % nodes == 0 )); then
            MPI_LAUNCH+=(--map-by "ppr:${ppn}:node")
        fi

        MPI_LAUNCH+=("${MPI_X_FLAGS[@]}")
    fi
}

AUCTION_MPI_LAUNCH=()
AUCTION_RANKFILE=""

make_opt_mpi_launch()
{
    local worker_ranks="$1"
    local worker_nodes ppn capacity
    local total_ranks
    local mpi_args=()
    local worker worker_node_index worker_slot mpi_rank

    if (( ALLOCATED_NODES < 2 )); then
        echo "opt_dstm requires at least two allocated nodes: one scheduler node and one or more worker nodes" >&2
        return 1
    fi

    worker_nodes="${worker_ranks}"

    (( worker_nodes > MAX_WORKER_NODES )) &&
        worker_nodes="${MAX_WORKER_NODES}"

    # NODELIST[0] is reserved exclusively for scheduler rank 0.
    (( worker_nodes > ALLOCATED_NODES - 1 )) &&
        worker_nodes=$((ALLOCATED_NODES - 1))

    (( worker_nodes < 1 )) && worker_nodes=1

    ppn=$(( (worker_ranks + worker_nodes - 1) / worker_nodes ))
    capacity=$(( worker_nodes * MAX_PPN ))

    if (( worker_ranks > capacity || ppn > MAX_PPN )); then
        echo "opt_dstm cannot fit ${worker_ranks} thermal ranks on ${worker_nodes} worker nodes at ${MAX_PPN} ranks/node" >&2
        return 1
    fi

    total_ranks=$((worker_ranks + 1))

    AUCTION_RANKFILE="${RESULT_DIR}/opt_r${worker_ranks}.rankfile"
    : > "${AUCTION_RANKFILE}"

    # Scheduler rank 0 is alone on the first allocated node.
    printf 'rank 0=%s slot=0\n' \
        "${NODELIST[0]}" \
        >> "${AUCTION_RANKFILE}"

    # Spread thermal ranks across nodes 1..worker_nodes. No thermal rank is
    # placed on the scheduler node.
    for ((worker = 0; worker < worker_ranks; worker++)); do
        mpi_rank=$((worker + 1))
        worker_node_index=$((1 + worker % worker_nodes))
        worker_slot=$((worker / worker_nodes))

        printf 'rank %d=%s slot=%d\n' \
            "${mpi_rank}" \
            "${NODELIST[worker_node_index]}" \
            "${worker_slot}" \
            >> "${AUCTION_RANKFILE}"
    done

    # Preserve the existing MPIRUN_ARGS environment variable.
    # shellcheck disable=SC2206
    mpi_args=( ${MPIRUN_ARGS} )

    AUCTION_MPI_LAUNCH=(
        "${MPIRUN}"
        "${mpi_args[@]}"
        -n "${total_ranks}"
        --rankfile "${AUCTION_RANKFILE}"
    )

    if [[ "${ALLOC_SRC}" != "localhost" ]]; then
        AUCTION_MPI_LAUNCH+=(--hostfile "${ALLOC_NODEFILE}")

        if [[ -n "${MPI_PREFIX}" ]]; then
            AUCTION_MPI_LAUNCH+=(--prefix "${MPI_PREFIX}")
        fi

        AUCTION_MPI_LAUNCH+=("${MPI_X_FLAGS[@]}")
    fi
}

mkdir -p "${TRACE_DIR}" "${LOG_DIR}"

if [[ ! -x "${DSTM_EXE}" ]]; then
    echo "Missing executable: ${DSTM_EXE}" >&2
    exit 1
fi

if [[ ! -x "${AUCTION_DSTM_EXE}" ]]; then
    echo "Missing executable: ${AUCTION_DSTM_EXE}" >&2
    exit 1
fi

if [[ ! -f "${SST_DIR}/run.py" ]]; then
    echo "Missing SST model: ${SST_DIR}/run.py" >&2
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

run_dstm()
{
    local scaling_mode="$1"
    local ranks="$2"
    local racks="$3"
    local jobs="$4"
    local trace="$5"
    local repeat="$6"
    local log="${LOG_DIR}/${scaling_mode}_dstm_r${ranks}_run${repeat}.log"

    local elapsed="nan"
    local status

    make_mpi_launch "${ranks}"

    if "${MPI_LAUNCH[@]}" \
        "${DSTM_EXE}" \
        --racks "${racks}" \
        --nodes-per-rack "${NODES_PER_RACK}" \
        --steps "${STEPS}" \
        --jobs "${trace}" >"${log}" 2>&1; then

        if elapsed="$(
            awk '
                /^simulation runtime[[:space:]]*:/ {
                    value = $(NF - 1)
                    found = 1
                }
                END {
                    if (!found)
                        exit 1
                    printf "%.9f\n", value
                }
            ' "${log}"
        )"; then
            status="ok"
        else
            status="missing_timing"
        fi
    else
        status="failed"
    fi

    printf '%s,%s,%d,%d,%d,%d,%d,%d,%s,%s\n' \
        "${scaling_mode}" "dstm" \
        "${ranks}" "${ranks}" \
        "${racks}" "${jobs}" "${STEPS}" "${repeat}" \
        "${elapsed}" "${status}" \
        >> "${RESULT_CSV}"

    echo "${scaling_mode}, dstm, ranks=${ranks}, run=${repeat}: ${elapsed}s (${status})"

    [[ "${status}" == "ok" ]]
}

run_opt_dstm()
{
    local scaling_mode="$1"
    local worker_ranks="$2"
    local racks="$3"
    local jobs="$4"
    local trace="$5"
    local repeat="$6"

    local mpi_ranks=$((worker_ranks + 1))
    local log="${LOG_DIR}/${scaling_mode}_opt_dstm_r${worker_ranks}_run${repeat}.log"

    local elapsed="nan"
    local status

    if ! make_opt_mpi_launch "${worker_ranks}"; then
        status="placement_failed"

        printf '%s,%s,%d,%d,%d,%d,%d,%d,%s,%s\n' \
            "${scaling_mode}" "opt_dstm" \
            "${worker_ranks}" "${mpi_ranks}" \
            "${racks}" "${jobs}" "${STEPS}" "${repeat}" \
            "${elapsed}" "${status}" \
            >> "${RESULT_CSV}"

        echo "${scaling_mode}, opt_dstm, thermal_ranks=${worker_ranks}, mpi_ranks=${mpi_ranks}, run=${repeat}: ${elapsed}s (${status})"
        return 1
    fi

    if "${AUCTION_MPI_LAUNCH[@]}" \
        "${AUCTION_DSTM_EXE}" \
        --racks "${racks}" \
        --nodes-per-rack "${NODES_PER_RACK}" \
        --steps "${STEPS}" \
        --schedule-window "${SCHEDULE_WINDOW}" \
        --control-latency "${CONTROL_LATENCY}" \
        --jobs "${trace}" >"${log}" 2>&1; then

        if elapsed="$(
            awk '
                /^(simulation runtime|wall-clock runtime)[[:space:]]*:/ {
                    value = $(NF - 1)
                    found = 1
                }
                END {
                    if (!found)
                        exit 1
                    printf "%.9f\n", value
                }
            ' "${log}"
        )"; then
            status="ok"
        else
            status="missing_timing"
        fi
    else
        status="failed"
    fi

    printf '%s,%s,%d,%d,%d,%d,%d,%d,%s,%s\n' \
        "${scaling_mode}" "opt_dstm" \
        "${worker_ranks}" "${mpi_ranks}" \
        "${racks}" "${jobs}" "${STEPS}" "${repeat}" \
        "${elapsed}" "${status}" \
        >> "${RESULT_CSV}"

    echo "${scaling_mode}, opt_dstm, thermal_ranks=${worker_ranks}, mpi_ranks=${mpi_ranks}, run=${repeat}: ${elapsed}s (${status})"

    [[ "${status}" == "ok" ]]
}

run_sst()
{
    local scaling_mode="$1"
    local ranks="$2"
    local racks="$3"
    local jobs="$4"
    local trace="$5"
    local repeat="$6"
    local log="${LOG_DIR}/${scaling_mode}_sst_r${ranks}_run${repeat}.log"

    local options
    options="--racks ${racks} --domains ${ranks} --sst-ranks ${ranks}"
    options+=" --nodes-per-rack ${NODES_PER_RACK}"
    options+=" --steps ${STEPS} --jobs ${trace}"

    local elapsed="nan"
    local status

    make_mpi_launch "${ranks}"

    if (
        cd "${SST_DIR}" &&
        SST_MODEL_OPTIONS="${options}" \
            "${MPI_LAUNCH[@]}" sst run.py
    ) >"${log}" 2>&1; then
        
        if elapsed="$(
            awk '
                {
                    for (i = 1; i <= NF; i++) {
                        if ($i ~ /^simulation_runtime_seconds=/) {
                            split($i, value, "=")
                            time = value[2] + 0.0

                            if (!found || time > maximum)
                                maximum = time

                            found = 1
                        }
                    }
                }
                END {
                    if (!found)
                        exit 1
                    printf "%.9f\n", maximum
                }
            ' "${log}"
        )"; then
            status="ok"
        else
            status="missing_timing"
        fi
    else
        status="failed"
    fi

    printf '%s,%s,%d,%d,%d,%d,%d,%d,%s,%s\n' \
        "${scaling_mode}" "sst" \
        "${ranks}" "${ranks}" \
        "${racks}" "${jobs}" "${STEPS}" "${repeat}" \
        "${elapsed}" "${status}" \
        >> "${RESULT_CSV}"

    echo "${scaling_mode}, sst, ranks=${ranks}, run=${repeat}: ${elapsed}s (${status})"

    [[ "${status}" == "ok" ]]
}

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
        run_dstm \
            "${scaling_mode}" "${ranks}" "${racks}" \
            "${jobs}" "${trace}" "${repeat}" || true

        run_opt_dstm \
            "${scaling_mode}" "${ranks}" "${racks}" \
            "${jobs}" "${trace}" "${repeat}" || true

        run_sst \
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
