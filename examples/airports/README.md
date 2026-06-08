# airports sim

A basic simulation of a network of airports using DSTM

## topology / structure

This simulation uses a mesh topology where each rank is an airport. Airports can be local, regional, or hub airports. Hub airports are the most likely to produce, connect, and receive flights. This creates a natural congestion. 

Upon mesh creation airports are created with this distribution:

```
**Hub airports** - 15% of total
**Regional airports** - 35%
**Local airports** - 50% 
```

## departures and arrivals

Each airport has 24 gates. These gates have slots that can hold incoming flights. In the rare occurrance that a gate is full, it is handled as an edge case and delayed. Flights can only travel a darray distance of 3 before having to *stop for fuel*. We only have 24 gates because that is how many airports can reach us via a 3 distance hop. We make sure that each gate can only be written to by one airport (right now just a simple hash function)  

## layovers

If the destination is larger than a darray of distance 3 away we need to *stop for fuel*. We pick a place to stop that is (1) closer to our destination, (2) and preferrably a hub airport. This adds some congestion to the hub airports and also makes the synchronizations more local. 

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
mpirun -n 16 ./airport_sim --time 20 --seed 42
```

### Please note

In the summaries it appears that more plans depart then land. That made me nervous at first but I realized it is because some flights depart and then the simulation ends while they are traveling to their destination. 
