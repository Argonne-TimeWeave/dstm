#!/bin/bash

OUTPUT_CSV="results.csv"
EXE_PATH="./../minimal"
TOPO_DIR="./../topos"
LOG_DIR="data"
RUNS_PER_CONFIG=5

mkdir -p "$LOG_DIR"

echo "topo,n,p,avg_max_distance,max_of_runs" > "$OUTPUT_CSV"

# List of topology files (relative to TOPO_DIR). Add/remove as needed.
TOPOS=(topo5.txt small_cluster.txt med_cluster.txt topo24.txt)
Ps=(75)

# Extract N from a topology file: first non-blank, non-comment line.
read_n_from_topo() {
    local file="$1"
    # strip comments (everything after '#'), trim, drop blanks, take first line
    sed 's/#.*//' "$file" \
        | awk 'NF { print $1; exit }'
}

for topo in "${TOPOS[@]}"; do
    topo_path="$TOPO_DIR/$topo"
    if [ ! -f "$topo_path" ]; then
        echo "WARNING: topology file not found: $topo_path — skipping"
        continue
    fi

    n=$(read_n_from_topo "$topo_path")
    if [ -z "$n" ] || ! [[ "$n" =~ ^[0-9]+$ ]]; then
        echo "WARNING: could not parse N from $topo_path — skipping"
        continue
    fi

    topo_name="${topo%.txt}"   # strip extension for tidy log/CSV names

    for p in "${Ps[@]}"; do
        total_distance=0
        max_run_distance=0

        for run in $(seq 1 $RUNS_PER_CONFIG); do
            log_file="$LOG_DIR/${topo_name}_n${n}_p${p}_run${run}.log"

            mpirun -n "$n" "$EXE_PATH" \
                --topo "$topo_path" \
                --time $((n * 8)) \
                --prob "$p" \
                > "$log_file"

            max_distance=$(python3 parse.py "$log_file" \
                           | grep "Maximum time step distance:" \
                           | awk '{print $5}')

            if [ -z "$max_distance" ]; then
                max_distance=0
            fi

            total_distance=$(echo "$total_distance + $max_distance" | bc)

            if (( $(echo "$max_distance > $max_run_distance" | bc -l) )); then
                max_run_distance=$max_distance
            fi

            # keep only the first run's log for inspection
            if [ "$run" -gt 1 ]; then
                rm "$log_file"
            fi

            echo "Completed: topo=$topo_name, n=$n, p=$p, run=$run, " \
                 "max_distance=$max_distance, max=$max_run_distance"
        done

        avg_distance=$(echo "scale=2; $total_distance / $RUNS_PER_CONFIG" | bc)

        echo "$topo_name,$n,$p,$avg_distance,$max_run_distance" >> "$OUTPUT_CSV"

        echo "Finished: topo=$topo_name, n=$n, p=$p, " \
             "avg=$avg_distance, max=$max_run_distance"
    done
done

echo "Results saved to $OUTPUT_CSV"
