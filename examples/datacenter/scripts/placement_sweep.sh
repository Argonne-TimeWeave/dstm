#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${SCRIPT_DIR}/results/placement"
COMBINED="${RESULT_DIR}/placement.csv"

mkdir -p "${RESULT_DIR}"

printf '%s\n' \
    "worker_nodes,ppn,mode,system,ranks,mpi_ranks,racks,jobs,steps,repeat,runtime_seconds,status" \
    > "${COMBINED}"

export RANKS_OVERRIDE="${RANKS_OVERRIDE:-32}"
export SYSTEMS="${SYSTEMS:-opt_dstm opt_sst}"
export REPEATS="${REPEATS:-5}"

# Longer execution makes subsecond timer noise less significant.
export STEPS="${STEPS:-10000}"
export STRONG_RACKS="${STRONG_RACKS:-512}"
export STRONG_JOBS="${STRONG_JOBS:-32768}"
export MAX_PPN="${MAX_PPN:-32}"

ranks="${RANKS_OVERRIDE}"

for worker_nodes in 1 2 4 8; do
    if (( ranks % worker_nodes != 0 )); then
        echo "Skipping ${worker_nodes} nodes: ${ranks} ranks not divisible"
        continue
    fi

    ppn=$((ranks / worker_nodes))

    if (( ppn > MAX_PPN )); then
        echo "Skipping ${worker_nodes} nodes: ${ppn} ppn exceeds ${MAX_PPN}"
        continue
    fi

    echo
    echo "=== ${ranks} ranks: ${worker_nodes} nodes x ${ppn} ranks/node ==="

    export MAX_WORKER_NODES="${worker_nodes}"

    "${SCRIPT_DIR}/scaling.sh" strong

    output="${RESULT_DIR}/placement_n${worker_nodes}_ppn${ppn}.csv"
    cp "${SCRIPT_DIR}/results/scaling.csv" "${output}"

    awk -F, -v OFS=, \
        -v nodes="${worker_nodes}" \
        -v ppn="${ppn}" \
        'NR > 1 { print nodes, ppn, $0 }' \
        "${output}" >> "${COMBINED}"
done

echo
echo "Combined results: ${COMBINED}"
