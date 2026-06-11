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
each rank has 8 airports
mpirun -n 4  ./airports --na 8 --time 64
mpirun -n 16 ./airports --na 8 --time 64
```

### Please note

In the summaries it appears that more plans depart then land. That made me nervous at first but I realized it is because some flights depart and then the simulation ends while they are traveling to their destination. 
