# single\_mpi\_air (send and recv)

This is a airport simulation that uses individual MPI send and recv communications. The intention is to compare results from this with our DSTM implementation

# limitations

In it's current state I am not 100% confident on how closely the communication structure resembles the DSTM version. But the intention is there. It has very similar performance to MPI all to all, I believe this is due to the sparsity of our communications? It could also be due to poor programming. I do not know. 
