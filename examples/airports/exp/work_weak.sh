for w in 0 10000 50000 100000 250000 500000; do
    echo "=== WORK=$w ==="
    mpirun -n 1 ./../airports --block-rows 16 --block-cols 8 --time 64 --work $w --quiet 2>&1 | grep -E "Elapsed|Timesteps"
    mpirun -n 32 ./../airports --block-rows 16 --block-cols 8 --time 64 --work $w --quiet 2>&1 | grep -E "Elapsed|Timesteps"
done
