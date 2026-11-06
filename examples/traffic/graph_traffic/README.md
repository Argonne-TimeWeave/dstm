# Stale data traffic sim
The goal is to implement some routing decisions that use delayed congestion information about far away nodes. 

DSTM graph representation is only needed for communication.

The routing requires each rank to own a representation of the entire (traffic) graph which contains congestion information for each vertex or edge. 
Each time step the data is updated with local congestion information (nodes/edges that our rank owns) and then send it to neighbors.

This naturally causes further vertices to hold more state information. It's a diffusion of state information. This information can be sent over edges, so that then a rank recieves the *num of incoming edges* updated states from neighbors and then can reduce or simplify it into it's most accurate and recent congestion information. 
(By congestion information i mean whole graph! in the future it may be possible to just send local information and be able to update a global state (within each rank) based on that (less data movement) for now we are just holding the full state)

The car (agent) makes a decision about the next step in it's path based on this congestion information. (dijkstra's)

Since the car is crossing over the same edge that the congestion state is traveling we will need to partition this out

[[ congestion state ] [ cars crossing ]] 

With a pointer to where the cars crossing buffer starts (congestion state should be the same size every time which is good but the cars crossing may have a certain cap for amount of cars that are able to cross on each edge at each timestep)
