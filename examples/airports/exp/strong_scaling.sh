#!/usr/bin/env bash
#
# Strong scaling: total problem size fixed (default 32x32 = 1024 airports),
# split across increasing rank counts.
#
# The grid is decomposed as rank_rows x rank_cols (most-square factoring of
# num_ranks). We pick block_rows = GRID_ROWS / rank_rows and block_cols =
# GRID_COLS / rank_cols so total airports stays constant.
#
set -euo pipefail

MPIRUN=${MPIRUN:-mpirun}
EXE=${EXE:-./airports}

# Fixed total grid (must be divisible by every rank decomposition we use)
GRID_ROWS=${GRID_ROWS:-16}
GRID_COLS=${GRID_COLS:-16}
TIMESTEPS=${TIMESTEPS:-256}
SEED=${SEED:-42}
REPS=${REPS:-1}

RANK_COUNTS=(1 2 4 8 16 32)

OUT_CSV=exp/${OUT_CSV:-strong_scaling.csv}
LOG_DIR=exp/${LOG_DIR:-logs_strong}
cd ../
mkdir -p "$LOG_DIR"

echo "rank_count,block_rows,block_cols,airports_per_rank,total_airports,rank_rows,rank_cols,timesteps,rep,elapsed_sec,timesteps_per_sec" > "$OUT_CSV"

# Same most-square factoring the C code uses
factor_ranks() {
    local n=$1
    local cols=$n
    local r=1
    while (( r * r <= n )); do
        if (( n % r == 0 )); then
            cols=$(( n / r ))
        fi
        r=$(( r + 1 ))
    done
    local rows=$(( n / cols ))
    echo "$rows $cols"
}

total_airports=$(( GRID_ROWS * GRID_COLS ))

for n in "${RANK_COUNTS[@]}"; do
    read -r rank_rows rank_cols < <(factor_ranks "$n")

    if (( GRID_ROWS % rank_rows != 0 || GRID_COLS % rank_cols != 0 )); then
        echo "SKIP n=$n: ${GRID_ROWS}x${GRID_COLS} not divisible by ${rank_rows}x${rank_cols}" >&2
        continue
    fi

    block_rows=$(( GRID_ROWS / rank_rows ))
    block_cols=$(( GRID_COLS / rank_cols ))
    airports_per_rank=$(( block_rows * block_cols ))

    for rep in $(seq 1 "$REPS"); do
        log="$LOG_DIR/strong_n${n}_r${rep}.log"
        echo ">>> STRONG n=$n rep=$rep ranks=${rank_rows}x${rank_cols} block=${block_rows}x${block_cols} total=${total_airports}"

        $MPIRUN -n "$n" "$EXE" \
            --block-rows "$block_rows" \
            --block-cols "$block_cols" \
            --time "$TIMESTEPS" \
            --seed "$SEED" \
            --quiet \
            > "$log" 2>&1

        elapsed=$(grep -E "^Elapsed time:" "$log" | awk '{print $3}')
        tps=$(grep -E "^Timesteps/second:" "$log" | awk '{print $2}')

        if [[ -z "$elapsed" || -z "$tps" ]]; then
            echo "WARNING: could not parse results from $log" >&2
            elapsed="NA"
            tps="NA"
        fi

        echo "${n},${block_rows},${block_cols},${airports_per_rank},${total_airports},${rank_rows},${rank_cols},${TIMESTEPS},${rep},${elapsed},${tps}" >> "$OUT_CSV"
    done
done

echo
echo "Strong scaling complete. Results in $OUT_CSV"
