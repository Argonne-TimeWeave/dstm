# Minimal Network Simulation

A distributed network simulator that demonstrates packet routing around a ring topology.

## Overview

This iteration of the simulation: Cut-through messaging with left right mailboxes.

Each rank owns a mailbox [ left , right ] with incoming messages.

Each timestep:
 - First, fill slots with empty values (bad)
 - Then, get each slot (just print it out)
 - Then, send randomly, delta is distance (pick slot by shortest path)

## Problems \ TODO

Multiple sends to one slot on the same time stamp write over each other.

*Possible Solve*: No cut-through (more realistic as well). Mailbox slots need to be able to handle multiple packets, most preferrably the rank should be able to read them all in one get and then parse out each packet to see if it has arrived at it's location or needs to be passed on to the next rank. More realistically, it can read multiple locations in each slot of the mailbox and then read each location until it reaches an empty slot (in that case there is no more to be read). 

*Further More on that Solve*: We could have multiple locations in each slot of the mailbox but deciding the size of that would be difficult. (too short we would lose data and too large we would be wasting memory space. 

## Running

mpirun -n 3 ./minimal

The simulation runs for 10 time steps and then exits cleanly.
