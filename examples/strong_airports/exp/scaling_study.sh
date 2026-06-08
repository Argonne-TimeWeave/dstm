#!/bin/bash

# Strong Scaling Study for Airport Simulation
# Usage: ./scaling_study.sh [timesteps] [seed]

TIMESTEPS=${1:-1028}
SEED=${2:-42}
OUTPUT_CSV="scaling_results.csv"
EXECUTABLE="./strong_airports"

cd ../

# MPI options to suppress InfiniBand warnings
MPI_OPTS="--mca btl ^openib"

# Rank counts to test
RANKS=(1 2 4 8 16 32)

# Check if executable exists
if [ ! -x "$EXECUTABLE" ]; then
    echo "Error: $EXECUTABLE not found or not executable"
    echo "Run 'make' first"
    exit 1
fi

# Create CSV header
echo "ranks,elapsed_seconds,timesteps_per_second,speedup" > "$OUTPUT_CSV"

echo "========================================"
echo "   STRONG SCALING STUDY"
echo "========================================"
echo "Timesteps: $TIMESTEPS"
echo "Seed: $SEED"
echo "Output: $OUTPUT_CSV"
echo "========================================"
echo ""

# Store baseline time for speedup calculation
BASELINE_TIME=""

for NRANKS in "${RANKS[@]}"; do
    echo "Running with $NRANKS rank(s)..."
    
    # Run simulation and capture output
    OUTPUT=$(mpirun $MPI_OPTS -n $NRANKS $EXECUTABLE --time $TIMESTEPS --seed $SEED --quiet 2>/dev/null)
    
    # Check if run succeeded
    if [ $? -ne 0 ]; then
        echo "  FAILED (skipping)"
        echo "$NRANKS,FAILED,,," >> "$OUTPUT_CSV"
        continue
    fi
    
    # Parse output
    ELAPSED=$(echo "$OUTPUT" | grep "Elapsed time:" | awk '{print $3}')
    TIMESTEPS_SEC=$(echo "$OUTPUT" | grep "Timesteps/second:" | awk '{print $2}')
    
    # Store baseline for speedup calculation
    if [ -z "$BASELINE_TIME" ]; then
        BASELINE_TIME=$ELAPSED
    fi
    
    # Calculate speedup and efficiency
    SPEEDUP=$(echo "scale=4; $BASELINE_TIME / $ELAPSED" | bc)
    
    # Write to CSV
    echo "$NRANKS,$ELAPSED,$TIMESTEPS_SEC,$SPEEDUP" >> "$OUTPUT_CSV"
    
    # Print summary
    printf "  Ranks: %2d | Time: %8.4fs | Speedup: %6.2fx \n" \
           "$NRANKS" "$ELAPSED" "$SPEEDUP"
done

mv $OUTPUT_CSV exp/$OUTPUT_CSV

echo ""
echo "========================================"
echo "Results written to: $OUTPUT_CSV"
echo "========================================"
echo ""
cat "exp/$OUTPUT_CSV" | column -t -s','
