# comm counts results

if you are confused by the chronology of these experiments: scaling (hypothesis), comm\_timings, comm\_counts

so this is the latest experiment in the zombies benchmark

## hypothesis discussion

these results really back-up the *gates* hypothesis. In the airport example the communication per-rank ratio was 3670:110 (DSTM:MPI on 256 ranks), while in the zombie example it is 36694:372 (DSTM:MPI on 256 ranks). That is pretty drastic. 

```
Airport: ~36:1
Zombie : ~100:1
```

Therefore, when programming with DSTM it is very important to keep this aspect in mind. Accepting gets/puts from local nodes only (and shaping your darray/communication around that) is extremely important for performance. 
