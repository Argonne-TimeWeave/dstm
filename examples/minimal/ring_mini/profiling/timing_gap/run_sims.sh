#!/bin/bash

OUTPUT_CSV="results.csv"
EXE_PATH="./../../minimal"
LOG_DIR="data"
RUNS_PER_CONFIG=5

mkdir -p "$LOG_DIR"

echo "n,p,avg_max_distance,max_of_runs" > "$OUTPUT_CSV"

Ns=(4 8 12 16 20)
Ps=(10 25 30 45 60 75 90)

for n in "${Ns[@]}"; do
    for p in "${Ps[@]}"; do
        total_distance=0
        max_run_distance=0
        
        for run in $(seq 1 $RUNS_PER_CONFIG); do
            log_file="$LOG_DIR/mini_${n}_${p}_run${run}.log"
            
            mpirun -n $n "$EXE_PATH" --time 32 --prob $p > "$log_file"
            
            max_distance=$(python3 parse.py "$log_file" | grep "Maximum time step distance:" | awk '{print $5}')
            
            if [ -z "$max_distance" ]; then
                max_distance=0
            fi
            
            total_distance=$(echo "$total_distance + $max_distance" | bc)

            if (( $(echo "$max_distance > $max_run_distance" | bc -l) )); then
                max_run_distance=$max_distance
            fi

            if [ $run -gt 1 ]; then
                rm "$log_file"
            fi
            
            echo "Completed: n=$n, p=$p, run=$run, max_distance=$max_distance, max=$max_run_distance"
        done
        
        avg_distance=$(echo "scale=2; $total_distance / $RUNS_PER_CONFIG" | bc)
        
        echo "$n,$p,$avg_distance,$max_run_distance" >> "$OUTPUT_CSV"
        
        echo "Finished: n=$n, p=$p, avg=$avg_distance, max=$max_run_distance"
    done
done

echo "Results saved to $OUTPUT_CSV"
