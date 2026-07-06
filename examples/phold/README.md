# PHOLD Benchmark

This benchmark is a classic PDES benchmark. A unfortunately is an example of a very bad usecase for DSTM.

## Why is it bad?

This benchmark (comparing to the ROSS verison here) requires all ranks to be connected to one another with a distance of 1. Therefore, DSTM has a very difficult time creating any meaningful drift. 

DSTM is tick-stepping while the ROSS PHOLD benchmark is event-driven. Therefore in the DSTM case we are looping over O(timesteps) even when no events exist for a given tick. And DSTM does all of the work in a timestep (getting/putting in inboxes/outboxes) even when nothing is happening. 

Our DSTM version of PHOLD is a mailbox holding style. This causes for a large amount of empty communications per step. This typically is not a large problem, but in the PHOLD case with no computation and event-driven timing it is no good.
