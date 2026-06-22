# profiling within minimal

The profiling within minimal is indeed also minimal.

## what do we do

We have a bash script that runs our minimal simulation with different parameters (the varying ones being n [num ranks] and p [probability of injecting a new packet]). It runs each configuration of parameters multiple times (5) and then calls a parsing script and outputs our metrics into a results.csv. 

## metric

Right now the metric is just: the largest distance (in time) between any two ranks at any given simulation time. We just want to know if our ranks are getting ahead of each other or not. A small distance means that we practically have a timestep synched simulation. While a large distance is in our case good as it means that the simulation is utilizing the strengths of the DSTM API. 


