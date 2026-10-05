# Epoch-Based Datacenter Simulation
The simulation separates scheduling from thermal execution:

 - Rank 0 is a dedicated global job scheduler.
 - All other ranks own rack partitions and execute the power and thermal model.
 - Execution is divided into configurable scheduling epochs, such as 60 simulated seconds.

At each epoch boundary, thermal ranks send a compact snapshot of their rack temperatures, cooling state, and power to the scheduler. After a configurable control delay, the scheduler uses this feedback, job arrivals, completions, and global resource availability to compute the next epoch’s schedule.

The scheduler sends each thermal rank one schedule block containing its node utilization for every tick in the upcoming epoch. These blocks are written to future DSTM versions, allowing thermal ranks to execute independently until the next epoch.

This design:

 - preserves deterministic global FCFS scheduling;
 - enables temperature-aware global placement;
 - reduces schedule communication to one block per rank per epoch;
 - keeps per-tick thermal communication sparse and local;
 - allows ranks to drift between scheduling boundaries without rollback or global per-tick barriers.
