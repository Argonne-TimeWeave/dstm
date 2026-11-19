#!/usr/bin/env bash
#
# Arithmetic-intensity sweep. Fixes ranks and total racks, varies only
# --nodes-per-rack, so per-tick compute grows with NO change to message
# count or cadence. Tests the hypothesis that DSTM overtakes SST once
# per-tick work is large enough to amortize sync overhead on a fast
# interconnect (it already wins on the slower JLSE fabric).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${SCRIPT_DIR}/results/intensity"
COMBINED="${RESULT_DIR}/intensity.csv"

mkdir -p "${RESULT_DIR}"

printf '%s\n' \
    "nodes_per_rack,mode,system,ranks,mpi_ranks,racks,jobs,steps,repeat,runtime_seconds,status" \
    > "${COMBINED}"

export RANKS_OVERRIDE="${RANKS_OVERRIDE:-32}"
export SYSTEMS="${SYSTEMS:-opt_dstm opt_sst}"
export REPEATS="${REPEATS:-5}"

# Long runs so subsecond timer noise is negligible.
export STEPS="${STEPS:-10000}"

# Fixed problem size; only the per-rack node count changes.
export STRONG_RACKS="${STRONG_RACKS:-512}"
export STRONG_JOBS="${STRONG_JOBS:-32768}"
export TREE_FANOUT="${TREE_FANOUT:-8}"
export MAX_WORKER_NODES="${MAX_WORKER_NODES:-8}"

# The intensity axis. Override with e.g. NPR_LIST="32 128 512".
NPR_LIST="${NPR_LIST:-32 64 128 256 512}"

for npr in ${NPR_LIST}; do
    echo
    echo "=== nodes_per_rack=${npr} (ranks=${RANKS_OVERRIDE}, racks=${STRONG_RACKS}) ==="

    export NODES_PER_RACK="${npr}"

    "${SCRIPT_DIR}/scaling.sh" strong

    output="${RESULT_DIR}/intensity_npr${npr}.csv"
    cp "${SCRIPT_DIR}/results/scaling.csv" "${output}"

    awk -F, -v OFS=, -v npr="${npr}" \
        'NR > 1 { print npr, $0 }' \
        "${output}" >> "${COMBINED}"
done

echo
echo "Combined results: ${COMBINED}"
