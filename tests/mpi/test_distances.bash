#!/bin/bash

PROGRAM="./distances"
TIMEOUT=10
MPI_RANKS=9
OUTPUT_FILE="stalled_tests.log"
RUNS_PER_BINARY=5

# Clear/create output file
> "$OUTPUT_FILE"

echo "Running all test combinations (0-255) x $RUNS_PER_BINARY runs each..."
echo "Logging results to: $OUTPUT_FILE"
echo ""

CONSISTENT_STALL=0
INTERMITTENT_STALL=0
PASS_COUNT=0

for i in {0..255}; do
    # Simpler binary conversion
    binary=$(python3 -c "print(format($i, '08b'))")
    
    stall_count=0
    
    # Run 5 times
    for run in {1..5}; do
        timeout ${TIMEOUT}s mpirun -n ${MPI_RANKS} "$PROGRAM" "$binary" > /dev/null 2>&1
        exit_code=$?
        
        if [ $exit_code -eq 124 ]; then
            ((stall_count++))
        fi
    done
    
    # Analyze results
    if [ $stall_count -eq 5 ]; then
        echo "CONSISTENT STALL: $binary (mask=$i) - stalled all 5 runs"
        echo "CONSISTENT STALL: $binary (mask=$i)" >> "$OUTPUT_FILE"
        ((CONSISTENT_STALL++))
    elif [ $stall_count -gt 0 ]; then
        echo "INTERMITTENT STALL: $binary (mask=$i) - stalled $stall_count/5 runs"
        echo "INTERMITTENT STALL: $binary (mask=$i) - stalled $stall_count/5 runs" >> "$OUTPUT_FILE"
        ((INTERMITTENT_STALL++))
    else
        echo "PASS: $binary (mask=$i) - all 5 runs passed"
        ((PASS_COUNT++))
    fi
done

echo ""
echo "====== SUMMARY ======"
echo "Passed (all 5 runs): $PASS_COUNT"
echo "Consistent stalls (5/5 runs): $CONSISTENT_STALL"
echo "Intermittent stalls (race condition): $INTERMITTENT_STALL"
echo "Results logged to: $OUTPUT_FILE"
