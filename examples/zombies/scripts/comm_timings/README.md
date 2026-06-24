# comm timing results

following the scaling results, i tried to measure some communication timings to get a better understanding of the performance difference between DSTM and MPI (zombie simulation). 

DSTM is spending much more time communicating, mainly with *get* operations growing exponentially.

## dicussing our prior hypothesis

It does seem like the lack of a *gate system* is to blame for the performance difference. Put time is also growing, so we can not disregard the *packing hypothesis*. But, it is strongly overshadowed by the growth of *get*. 
