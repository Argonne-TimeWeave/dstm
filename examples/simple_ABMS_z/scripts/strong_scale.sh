#!/usr/bin/env bash
# strong_scale.sh — fixed global agent count, sweep ranks
#
# usage: ./strong_scale.sh [N_TOTAL] [TIME]

set -uo pipefail
N=${1:-4000}
T=${2:-200}

# Same rank set as your earlier sweeps, adjust to what your hostfile allows.
SWEEP_CONFIG="${SWEEP_CONFIG:-4:4:1 8:8:1 16:16:1 32:32:1 64:32:2 128:32:4 256:32:8}"

for entry in $SWEEP_CONFIG; do
    IFS=: read -r ranks ppn nodes <<< "$entry"
    per_rank_total=$(( N / ranks ))
    (( per_rank_total < 1 )) && per_rank_total=1
    nz=5
    nh=$(( per_rank_total - nz ))
    (( nh < 0 )) && nh=0

    bash compare.sh \
        --hostfile "$COBALT_NODEFILE" \
        --sweep-config "$entry" \
        --nh "$nh" \
        --nz "$nz" \
        --time "$T" \
        --density 1.0

    # Rename the run's summary.csv so we can concatenate later.
    latest=$(ls -td results/*/ | head -1)
    mv "$latest/summary.csv" "$latest/strong_r${ranks}.csv"
done
