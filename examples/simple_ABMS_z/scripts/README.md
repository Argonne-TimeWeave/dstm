# DSTM vs Repast

```
# One matched run: 4 ranks, 500 humans + 5 zombies per rank, 100 ticks
bash compare.sh

# Just DSTM, larger population
bash compare.sh --only dstm --nh 5000 --time 200

# Strong-scaling sweep on process count (fixed problem size)
bash compare.sh --sweep-nprocs "1 2 4 8 16" --nh 1000 --time 200 --repeats 3

# Weak-scaling-ish sweep (nh per rank is constant → total work grows)
bash compare.sh --sweep-nprocs "1 4 16" --nh 500 --time 200

# Full 2-D sweep: nprocs × human population
bash compare.sh --sweep-nprocs "1 2 4 8" --sweep-nh "100 500 2000" --repeats 3

bash compare.sh --sweep-density "0.5 1.0 5.0" --nh 500 --time 200
```

## Multi-node

```
./compare.sh \
    --hostfile "$COBALT_NODEFILE" \
    --sweep-config "1:1:1 2:1:2 4:1:4 8:1:8 16:2:8 32:4:8 64:8:8 128:16:8 256:32:8" \
    --nh 512 --time 256 --density 1.0 --repeats 4

./compare.sh \
    --hostfile "$COBALT_NODEFILE" \
    --sweep-config "16:2:8 64:8:8 256:32:8" \
    --sweep-density "0.1 1.0 10.0" \
    --nh 512 --time 200

./compare.sh --only dstm \
    --hostfile "$COBALT_NODEFILE" \
    --sweep-config "1:1:1 4:1:4 16:2:8 64:8:8 256:32:8" \
    --nh 1024
```
