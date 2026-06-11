# airports sim

A basic simulation of a network of airports using DSTM. Details on the simulation are located in ../README.md

## Suggestions and future

We currently get from each gate one by one. It would be nice to have a darray function like dstm\_darray\_range\_get() so that I could get all of my gates and then locally unload and handle them (some aggregation on the gets). I know that dstm\_get() does this but programmers may prefer to use a wrapper like dstm\_darray\_get() provides (ease of use). 

**Current approach:** Sequential GET operations for each gate:

```c
for (int g = 0; g < num_ranks; g++) {
    dstm_darray_get_or_default(gates, idx + g, ...);
}
```

Could implement weather delays to add some congestion / randomization in the future

## Running the simulation:

```
# Weak scaling: each rank gets 8x8 = 64 airports, total grows with ranks
mpirun -n 4  ./airports --block-rows 8 --block-cols 8 --time 64
mpirun -n 16 ./airports --block-rows 8 --block-cols 8 --time 64

# Strong scaling: fix total at, say, 32x32 = 1024 airports
mpirun -n 1  ./airports --block-rows 32 --block-cols 32 --time 64   # 1 rank, 32x32 block
mpirun -n 4  ./airports --block-rows 16 --block-cols 16 --time 64   # 2x2 ranks, 16x16 blocks
mpirun -n 16 ./airports --block-rows 8  --block-cols 8  --time 64   # 4x4 ranks, 8x8 blocks
mpirun -n 64 ./airports --block-rows 4  --block-cols 4  --time 64   # 8x8 ranks, 4x4 blocks
```

### Please note

In the summaries it appears that more plans depart then land. That made me nervous at first but I realized it is because some flights depart and then the simulation ends while they are traveling to their destination. 
