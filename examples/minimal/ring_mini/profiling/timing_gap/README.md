# profiling within minimal

The profiling within minimal is indeed also minimal.

## what do we do

We have a bash script that runs our minimal simulation with different parameters (the varying ones being n [num ranks] and p [probability of injecting a new packet]). It runs each configuration of parameters multiple times (5) and then calls a parsing script and outputs our metrics into a results.csv. 

## metric

Right now the metric is just: the largest distance (in time) between any two ranks at any given simulation time. We just want to know if our ranks are getting ahead of each other or not. A small distance means that we practically have a timestep synched simulation. While a large distance is in our case good as it means that the simulation is utilizing the strengths of the DSTM API. 

### Example output

```
n,p,avg_max_distance,max_of_runs
...
4,45,2.60,3
...
8,45,4.00,4
...
12,45,5.00,6
...
16,45,7.20,8
...
20,45,7.80,9
...
```

max\_of\_runs is just the max max\_distance across all runs of that configuration. In this ring topology we observe that this almost always reaches (n / 2), or at least approaches it.  

It is nice that we can observe the avg\_max\_distance increasing with n. It seems that p and avg\_max\_distance do not correlate in any significant way.

## Limitations / TODO

Can we trust the logs that DSTM is writting? The ordering could be varied depending on if ranks write at the same time, and our simulation is not very arithmatically intense at all so we can expect ranks to be writing over each other frequently. 

What other important metrics can we extract from our current logs?

How can we use the hierarchical logging to our advantage in the future? 
