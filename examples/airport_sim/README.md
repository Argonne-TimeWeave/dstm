# airports sim

A basic simulation of a network of airports 

Three directories contain three different implementations of the same distributed simulation (DSTM, MPI\_All (mpi\_air), and MPI\_Send/Recv (single\_mpi\_air). The scripts directory compares performance results from all three.

## topology / structure

This simulation uses a mesh topology where each rank is an airport. Airports can be local, regional, or hub airports. Hub airports are the most likely to produce, connect, and receive flights. This creates a natural congestion. 

Upon mesh creation airports are created with this distribution:

```
**Hub airports** - 15% of total
**Regional airports** - 35%
**Local airports** - 50% 
```

Now each node contains --na number of airports. airplanes do not travel locally, layover distance is calculated by node not by local airport. 

Ranks represent airport regions with this notation (\<rank or region>.\<local airport index>)

## departures and arrivals

Each airport has 24 gates. These gates have slots that can hold incoming flights. In the rare occurrance that a gate is full, it is handled as an edge case and delayed. Flights can only travel a darray distance of 3 before having to *stop for fuel*. We only have 24 gates because that is how many airports can reach us via a 3 distance hop. We make sure that each gate can only be written to by one airport (right now just a simple hash function)  

## layovers

If the destination is larger than a darray of distance 3 away we need to *stop for fuel*. We pick a place to stop that is (1) closer to our destination, (2) and preferrably a hub airport. This adds some congestion to the hub airports and also makes the synchronizations more local. 

## fake work

This simulation does not contain a lot of computational effort, therefore the --work option allows you to input some fake work.

## scripts 

The scripts directory contains some experiments that compare results from all three implementations. Such as weak scaling and strong scaling. 
