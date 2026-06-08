#!/usr/bin/env bash
#
# Weak scaling: each rank owns a fixed block of airports (default 8x8 = 64).
# Total problem size grows linearly with rank count.
#
set -euo pipefail

MPIRUN=${MPIRUN:-mpirun}
EXE=${EXE:-./airports}

BLOCK_ROWS=${BLOCK_ROWS:-16}
BLOCK_COLS=${BLOCK_COLS:-16}
TIMESTEPS=${TIMESTEPS:-128}
SEED=${SEED:-42}
REPS=${REPS:-1}

RANK_COUNTS=(1 2 4 8 16 32)

OUT_CSV=exp/${OUT_CSV:-weak_scaling.csv}
LOG_DIR=exp/${LOG_DIR:-logs_weak}
cd ../
mkdir -p "$LOG_DIR"

echo "rank_count,block_rows,block_cols,airports_per_rank,total_airports,timesteps,rep,elapsed_sec,timesteps_per_sec" > "$OUT_CSV"

airports_per_rank=$(( BLOCK_ROWS * BLOCK_COLS ))

for n in "${RANK_COUNTS[@]}"; do
    total_airports=$(( airports_per_rank * n ))

    for rep in $(seq 1 "$REPS"); do
        log="$LOG_DIR/weak_n${n}_r${rep}.log"
        echo ">>> WEAK n=$n rep=$rep block=${BLOCK_ROWS}x${BLOCK_COLS} total=${total_airports}"

        $MPIRUN -n "$n" "$EXE" \
            --block-rows "$BLOCK_ROWS" \
            --block-cols "$BLOCK_COLS" \
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

        echo "${n},${BLOCK_ROWS},${BLOCK_COLS},${airports_per_rank},${total_airports},${TIMESTEPS},${rep},${elapsed},${tps}" >> "$OUT_CSV"
    done
done

echo
echo "Weak scaling complete. Results in $OUT_CSV"
