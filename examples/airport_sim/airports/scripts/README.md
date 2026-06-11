# scripts for DSTM airports

The bash script located in this directory runs a weak scaling test with the --logging option on. The metrics of our interest are how long DSTM\_get and DSTM\_put take. The included python files either parse our dumps (analyize\_dump.py) or plot our data in a meaningful way. 

Included is a python file that uses data located in ../../mpi\_air/scripts (compare\_to\_mpi.py). This requires the same metric gathering to be performed in that directory first.
