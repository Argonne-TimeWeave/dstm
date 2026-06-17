# zombie TODO 

figure out correct value for cell\_capcity

zombies now wander towards the local cell with the highest population
    or "POPULAR" locations
        humans do not avoid zombies currently...

there exists a bug that after a certain timestep the infected/zombie counts never change
    this is concerning because infected are not even turning into zombies 
        and that should just be a count to 3 days thing
For example: 
```
ncassarino@skylake07:~/dstm/examples/zombies> cat dump2.log | tail -n 25
zombie_sim:     normal:10/32:   Rank 10 [t=168]: cell 10.16 pop=1 same_cont=1 near_cont=2
zombie_sim:     normal:10/32:   Rank 10 [t=168]: cell 10.17 pop=2 same_cont=2 near_cont=1
zombie_sim:     normal:10/32:   Rank 10 [t=168]: cell 10.19 pop=1 same_cont=1 near_cont=0
zombie_sim:     normal:10/32:   Rank 10 [t=168]: ROSTER total=25 (limbo=0)

=============== INFECTION SUMMARY ===============
Status          Count
Healthy             4
Infected          361
Zombie            435
-------------------------
Total             800

High-risk population:   117 / 800  (14.6%)
High-risk infected:     117 / 117  (100.0% of HR)
=================================================

=============== RESULTS ===============
Total departures:   16500
Total arrivals:     16500
Total infections:   795

Elapsed time:       0.1651 seconds
Timesteps/second:   1023.45
========================================
ncassarino@skylake07:~/dstm/examples/zombies> cat dump.log | tail -n 25
zombie_sim:     normal:7/32:    Rank 7 [t=999]: cell 7.14 pop=1 same_cont=1 near_cont=0
zombie_sim:     normal:7/32:    Rank 7 [t=999]: cell 7.16 pop=16 same_cont=16 near_cont=0
zombie_sim:     normal:7/32:    Rank 7 [t=999]: cell 7.18 pop=1 same_cont=1 near_cont=0
zombie_sim:     normal:7/32:    Rank 7 [t=999]: ROSTER total=25 (limbo=0)

=============== INFECTION SUMMARY ===============
Status          Count
Healthy             3
Infected          362
Zombie            435
-------------------------
Total             800

High-risk population:   117 / 800  (14.6%)
High-risk infected:     117 / 117  (100.0% of HR)
=================================================

=============== RESULTS ===============
Total departures:   64070
Total arrivals:     64070
Total infections:   796

Elapsed time:       0.7191 seconds
Timesteps/second:   1390.68
========================================
```

when increasing the timestep from '--time 170' to '--time 1000' we get only a difference in one in infected counts. and more shockingly none of the infected turn into zombies 

# How to run on multi-node
## This is 8 nodes; 4 ranks per node
mpirun -n 32 --hostfile "$COBALT\_NODEFILE" \
    --prefix "$(dirname "$(dirname "$(which mpirun)")")" \
    --map-by ppr:4:node \
    -x LD\_LIBRARY\_PATH -x PATH \
    ./zombies
