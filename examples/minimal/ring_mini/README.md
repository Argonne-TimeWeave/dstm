# Minimal Network Simulation

A distributed network simulator that demonstrates packet routing around a ring topology.

## Overview

This iteration of the simulation: Forward Passing messages with left right mailboxes.

Each rank owns a mailbox [ left *slot*  , right *slot* ] with incoming messages.

Each slot is a fixed size. Holding multiple packets. *Padded with empty packets*.

 left | right

[ n | n ]

left: [p, p, d, d, d]; n = 5; recv 2 packets (rest are defaults)

Each timestep:
 - First, fill slots with empty values (bad)
 - Then, get each slot. Either (1) print out slot [it arrived at it's destination] or (2) pass it on to the next rank [has not arrived at it's destination]
 - Then, select to send or not randomly, destination is also random (if it does decide to send)

**Comments are very detailed if you are more interested** *most details are here though*

## Problems \ TODO

We drop packets that do not fit within our mailbox slot (because we are at a fixed size). I don't know if dynammically resizing is viable

Test other topologies

Get a better understanding of out debug messages (do some profiling)

Maybe try higher intensity per unpack

## Running

mpirun -n 3 ./minimal --time t --prob p [--quiet]

--time is number of timesteps

--prob is the likelihood for a rank to produce a new packet to be sent each timestep
