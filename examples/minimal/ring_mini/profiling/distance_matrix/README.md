# distance experiment

This experiment focuses on a specific feature of the distance matrix. DSTM only sends heartbeats to ranks within the distance matrix that are greater than 0. In our experiment we prove that. Before the simulation runs we convert anything with a distance over \<--dist> to be -1. 

We observe lower heartbeat communications with lower \<--dist> values. 
