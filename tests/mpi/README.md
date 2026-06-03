# MPI Tests

## Distances Test

A race condition exists within the current distances test that stalls the test. In hopes of finding this stall, the distances test now takes in an argument that denotes which tests within the file to run. For example, 00000001 only runs the first test and 11111111 runs all of the tests within the file. A bash script: test\_distances.bash was also created in order to explore the stall (outputs exist within the stalled\_tests.log file). Here follows stalls that I have observed from my run of test\_distances.bash:

```
INTERMITTENT STALL: 01101110 (mask=110) - stalled 1/5 runs
INTERMITTENT STALL: 01111111 (mask=127) - stalled 1/5 runs
INTERMITTENT STALL: 10011101 (mask=157) - stalled 1/5 runs
CONSISTENT STALL: 10111111 (mask=191)
CONSISTENT STALL: 11111111 (mask=255)
```

For now this know issue stays as a TODO
