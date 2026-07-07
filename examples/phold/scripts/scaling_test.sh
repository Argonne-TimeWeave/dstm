#!/bin/bash
# scaling_test.sh - Comparative scaling test: ROSS PHOLD vs DSTM PHOLD
# Run from: ~/dstm/examples/phold/scripts/
# Designed for Cobalt-scheduled systems with hostfile-based MPI launch.

#set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROSS_BIN="$SCRIPT_DIR/ross-phold"
DSTM_BIN="$SCRIPT_DIR/../complete/phold"
OUTPUT_CSV="$SCRIPT_DIR/scaling_results.csv"
RAW_DIR="$SCRIPT_DIR/raw_output"

# Detect MPI prefix (use mpirun's own prefix if not set)
if [ -z "${MPI_PREFIX:-}" ]; then
    MPI_PREFIX="$(dirname "$(dirname "$(which mpirun)")")"
fi

#RANKS="1 2 4 8 16 32 64 128 256"
RANKS="1 2 4 8 16 32 64"
NUM_TRIALS=4

# --- Shared parameters ---
LPS_PER_RANK=8
REMOTE_PROB=0.25
MEAN_DELAY=1.0

# --- ROSS settings ---
ROSS_END=500.00

# --- DSTM settings ---
DSTM_TIMESTEPS=500
DSTM_INIT_EVENTS=1
DSTM_SEED=42

# --- Hostfile ---
# Use COBALT_NODEFILE if available, otherwise fall back to localhost
if [ -n "${COBALT_NODEFILE:-}" ]; then
    HOSTFILE="$COBALT_NODEFILE"
else
    echo "WARNING: COBALT_NODEFILE not set, using localhost"
    HOSTFILE=""
fi

mkdir -p "$RAW_DIR"

# Write CSV header
echo "framework,ranks,trial,lps_per_rank,total_lps,runtime_sec,events_processed,throughput_eps,efficiency_eps_rank" > "$OUTPUT_CSV"

# ============================================================
# Helper: build mpirun command with proper host mapping
# ============================================================
build_mpi_cmd() {
    local np="$1"
    shift
    # Remaining args are the binary + its arguments

    local cmd="mpirun -n $np"
    cmd+=" --prefix $MPI_PREFIX"

    if [ -n "$HOSTFILE" ]; then
        cmd+=" --hostfile $HOSTFILE"
        # Map by node for multi-node runs to spread ranks across nodes
        if [ "$np" -gt 1 ]; then
            cmd+=" --map-by node"
        fi
    else
        cmd+=" --oversubscribe"
    fi

    cmd+=" -x LD_LIBRARY_PATH -x PATH"
    cmd+=" $*"

    echo "$cmd"
}

# ============================================================
# ROSS PHOLD  (conservative, --synch=2)
# ============================================================
echo "########################################"
echo "# ROSS PHOLD Scaling Test"
echo "# Hostfile: ${HOSTFILE:-localhost}"
echo "# MPI prefix: $MPI_PREFIX"
echo "########################################"

for np in $RANKS; do
    total_lps=$((np * LPS_PER_RANK))
    for trial in $(seq 1 $NUM_TRIALS); do
        echo "=== ROSS: ranks=$np, trial=$trial ==="

        RAWFILE="$RAW_DIR/ross_np${np}_t${trial}.txt"

        mpi_cmd=$(build_mpi_cmd "$np" \
            "$ROSS_BIN" \
            --synch=2 \
            --nlp="$LPS_PER_RANK" \
            --end="$ROSS_END" \
            --remote="$REMOTE_PROB" \
            --mean="$MEAN_DELAY")

        echo "CMD: $mpi_cmd"
        eval "$mpi_cmd" > "$RAWFILE" 2>&1
        rc=$?

        if [ $rc -ne 0 ]; then
            echo "WARNING: ROSS exited with rc=$rc"
            cat "$RAWFILE"
        fi

        # Parse runtime
        runtime=$(grep -i "running time" "$RAWFILE" | grep -oP '[\d]+\.[\d]+' | head -1)

        # Parse events
        events=$(grep -i "events processed" "$RAWFILE" | grep -oP '[\d]+' | head -1)
        if [ -z "${events:-}" ]; then
            events=$(grep -i "total events" "$RAWFILE" | grep -oP '[\d]+' | head -1)
        fi

        runtime=${runtime:-"N/A"}
        events=${events:-"N/A"}

        throughput="N/A"
        efficiency="N/A"
        if [[ "$runtime" != "N/A" && "$events" != "N/A" && "$runtime" != "0" ]]; then
            throughput=$(awk "BEGIN {printf \"%.2f\", $events / $runtime}")
            efficiency=$(awk "BEGIN {printf \"%.2f\", $events / $runtime / $np}")
        fi

        echo "ROSS,$np,$trial,$LPS_PER_RANK,$total_lps,$runtime,$events,$throughput,$efficiency" >> "$OUTPUT_CSV"
        echo "--- ROSS done: ranks=$np, trial=$trial, runtime=${runtime}s, events=$events ---"
        echo ""
    done
done

# ============================================================
# DSTM PHOLD
# ============================================================
echo "########################################"
echo "# DSTM PHOLD Scaling Test"
echo "########################################"

for np in $RANKS; do
    total_lps=$((np * LPS_PER_RANK))
    topo="complete"

    for trial in $(seq 1 $NUM_TRIALS); do
        echo "=== DSTM: ranks=$np, trial=$trial ==="

        RAWFILE="$RAW_DIR/dstm_np${np}_t${trial}.txt"

        mpi_cmd=$(build_mpi_cmd "$np" \
            "$DSTM_BIN" \
            --lps-per-rank "$LPS_PER_RANK" \
            --init-events "$DSTM_INIT_EVENTS" \
            --timesteps "$DSTM_TIMESTEPS" \
            --remote "$REMOTE_PROB" \
            --mean "$MEAN_DELAY" \
            --seed "$DSTM_SEED" \
            --topology "$topo" \
            --quiet)

        echo "CMD: $mpi_cmd"
        eval "$mpi_cmd" > "$RAWFILE" 2>&1
        rc=$?

        if [ $rc -ne 0 ]; then
            echo "WARNING: DSTM exited with rc=$rc"
            cat "$RAWFILE"
        fi

        # Parse DSTM output
        events=$(grep "events_processed" "$RAWFILE" | grep -oP '[\d]+' | head -1)
        runtime=$(grep "wall_time" "$RAWFILE" | grep -oP '[\d]+\.[\d]+' | head -1)
        throughput=$(grep "throughput" "$RAWFILE" | grep -oP '[\d]+' | head -1)
        efficiency=$(grep "efficiency" "$RAWFILE" | grep -oP '[\d]+' | head -1)

        events=${events:-"N/A"}
        runtime=${runtime:-"N/A"}
        throughput=${throughput:-"N/A"}
        efficiency=${efficiency:-"N/A"}

        echo "DSTM,$np,$trial,$LPS_PER_RANK,$total_lps,$runtime,$events,$throughput,$efficiency" >> "$OUTPUT_CSV"
        echo "--- DSTM done: ranks=$np, trial=$trial, runtime=${runtime}s, events=$events ---"
        echo ""
    done
done

echo "========================================"
echo "All tests complete. Results: $OUTPUT_CSV"
echo "Raw output saved to: $RAW_DIR/"
echo "========================================"
