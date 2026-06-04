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

*Possible Solve*: No cut-through (more realistic as well). Mailbox slots need to be able to handle multiple packets. Realistically each slot can contain multiple locations and then a rank could read each location until it reaches an empty slot (in that case there is no more to be read). 

 left | right

[ n | n ]

left: [p, p, d, d, d]; n = 5; recv 2 packets (rest are defaults)

*Further More on that Solve*: We could have multiple locations in each slot of the mailbox but deciding the size of that would be difficult. (too short we would lose data and too large we would be wasting memory space). And then it may be fine since each mailbox slot is only being touched by one node (left or right) therefore we shouldn't have an race conditions/overwritting behavior. The sending node would just need to put into each slot iteratively. Just fixed sized slots for now. 

## Running

mpirun -n 3 ./minimal

The simulation runs for 10 time steps and then exits cleanly.
