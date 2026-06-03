# Minimal Network Simulation

A distributed network simulator that demonstrates packet routing around a ring topology.

## TODO

This iteration of the simulation does not really use DSTM for it's intended purpose. Right now every time step there is a send and receive for each node. Therefore, communication every step. If our intention is to show use cases where ranks can work independently for a while... this does not do that.

Need to add some randomness to where packets come and go from (across larger distances as well; consider putting and getting a distance away instead of always putting 1 in the future (idk to simulate the distance it would take to do?))

Or we can consider having each rank own multiple src/dest (but then this begins to look like the traffic example. 

Currently each rank has some area for local packets, and we go over them one by one with the timestamp. This should be changed if we are going to not send somethign each and every time step. 

Can try with different topologies in the future (dragonfly for network most likely)

A summary printout that describes what occurred

## Overview

This simulation models a network where packets are transmitted between nodes over discrete time steps. Each process (rank) represents a network node that sends packets to its neighbor and receives from its predecessor.

## What It Simulates

Nodes are arranged in a unidirectional ring:

Rank 0 → Rank 1 → Rank 2 → ... → Rank 0

At each time step:
- Every rank sends a packet to the next rank (arrives 1 step later)
- Every rank receives packets from the previous rank
- Packets contain the sender's rank and the time they were sent

### LOCAL\_PACKET\_SLOTS

LOCAL\_PACKET\_SLOTS defines how many packet slots each rank owns in the shared distributed buffer. The distributed array is sized to num\_ranks * LOCAL\_PACKET\_SLOTS, and each rank’s local region is rank * LOCAL\_PACKET\_SLOTS through (rank+1) * LOCAL\_PACKET\_SLOTS - 1. At each time step the code picks a slot using current\_time % LOCAL\_PACKET\_SLOTS, so the same rank can hold multiple packets across different slot indices.

## Running

mpirun -n 3 ./minimal

## Output

The logs show when packets are sent and received. Example:

```
Rank 0 sent packet to rank 1
Rank 0 received: Packet from rank 2 at time 5
```

The simulation runs for 10 time steps and then exits cleanly.
