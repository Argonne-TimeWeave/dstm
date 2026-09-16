/* mpi_wrap.c — PMPI wrapper to count MPI calls and bytes.
 * Link into either binary; dumps /tmp/mpi_counts_rank<R>.txt on MPI_Finalize.
 *
 * Build:  mpicc -c -fPIC mpi_wrap.c -o mpi_wrap.o
 * Link:   ... mpi_wrap.o ...
 *
 * Counts every point-to-point send/recv, every collective, and total bytes
 * moved.  Sufficient to compare "amount of MPI activity per tick" between
 * two frameworks running the same workload.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long long n_isend = 0, n_irecv = 0, n_send = 0, n_recv = 0;
static long long n_wait = 0, n_waitall = 0, n_test = 0;
static long long n_barrier = 0, n_reduce = 0, n_allreduce = 0;
static long long n_bcast = 0, n_gather = 0, n_allgather = 0, n_alltoall = 0;
static long long bytes_sent = 0, bytes_recvd = 0;

static double t_start = 0.0;
static int wrap_rank = -1;

static int type_size(MPI_Datatype dt) {
    int s = 0;
    PMPI_Type_size(dt, &s);
    return s;
}

int MPI_Init(int *argc, char ***argv) {
    int rc = PMPI_Init(argc, argv);
    PMPI_Comm_rank(MPI_COMM_WORLD, &wrap_rank);
    t_start = PMPI_Wtime();
    return rc;
}

int MPI_Isend(const void *buf, int count, MPI_Datatype dt, int dest,
              int tag, MPI_Comm comm, MPI_Request *req) {
    n_isend++;
    bytes_sent += (long long)count * type_size(dt);
    return PMPI_Isend(buf, count, dt, dest, tag, comm, req);
}
int MPI_Irecv(void *buf, int count, MPI_Datatype dt, int src,
              int tag, MPI_Comm comm, MPI_Request *req) {
    n_irecv++;
    return PMPI_Irecv(buf, count, dt, src, tag, comm, req);
}
int MPI_Send(const void *buf, int count, MPI_Datatype dt, int dest,
             int tag, MPI_Comm comm) {
    n_send++;
    bytes_sent += (long long)count * type_size(dt);
    return PMPI_Send(buf, count, dt, dest, tag, comm);
}
int MPI_Recv(void *buf, int count, MPI_Datatype dt, int src,
             int tag, MPI_Comm comm, MPI_Status *st) {
    n_recv++;
    int rc = PMPI_Recv(buf, count, dt, src, tag, comm, st);
    bytes_recvd += (long long)count * type_size(dt);
    return rc;
}
int MPI_Wait(MPI_Request *req, MPI_Status *st) { n_wait++;    return PMPI_Wait(req, st); }
int MPI_Waitall(int c, MPI_Request *r, MPI_Status *s) { n_waitall++; return PMPI_Waitall(c, r, s); }
int MPI_Test(MPI_Request *r, int *f, MPI_Status *s) { n_test++;   return PMPI_Test(r, f, s); }

int MPI_Barrier(MPI_Comm c) { n_barrier++; return PMPI_Barrier(c); }
int MPI_Reduce(const void *sb, void *rb, int cnt, MPI_Datatype dt,
               MPI_Op op, int root, MPI_Comm c) {
    n_reduce++; bytes_sent += (long long)cnt * type_size(dt);
    return PMPI_Reduce(sb, rb, cnt, dt, op, root, c);
}
int MPI_Allreduce(const void *sb, void *rb, int cnt, MPI_Datatype dt,
                  MPI_Op op, MPI_Comm c) {
    n_allreduce++; bytes_sent += (long long)cnt * type_size(dt);
    return PMPI_Allreduce(sb, rb, cnt, dt, op, c);
}
int MPI_Bcast(void *b, int cnt, MPI_Datatype dt, int root, MPI_Comm c) {
    n_bcast++; bytes_sent += (long long)cnt * type_size(dt);
    return PMPI_Bcast(b, cnt, dt, root, c);
}
int MPI_Gather(const void *sb, int sc, MPI_Datatype sdt,
               void *rb, int rc, MPI_Datatype rdt, int root, MPI_Comm c) {
    n_gather++; bytes_sent += (long long)sc * type_size(sdt);
    return PMPI_Gather(sb, sc, sdt, rb, rc, rdt, root, c);
}
int MPI_Allgather(const void *sb, int sc, MPI_Datatype sdt,
                  void *rb, int rc, MPI_Datatype rdt, MPI_Comm c) {
    n_allgather++; bytes_sent += (long long)sc * type_size(sdt);
    return PMPI_Allgather(sb, sc, sdt, rb, rc, rdt, c);
}
int MPI_Alltoall(const void *sb, int sc, MPI_Datatype sdt,
                 void *rb, int rc, MPI_Datatype rdt, MPI_Comm c) {
    n_alltoall++; bytes_sent += (long long)sc * type_size(sdt);
    return PMPI_Alltoall(sb, sc, sdt, rb, rc, rdt, c);
}

int MPI_Finalize(void) {
    double t_end = PMPI_Wtime();
    char path[256];
    const char *dir = getenv("MPI_WRAP_OUTDIR");
    if (!dir) dir = "/tmp";
    snprintf(path, sizeof path, "%s/mpi_counts_rank%d.txt", dir, wrap_rank);
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "rank %d  elapsed %.6f\n", wrap_rank, t_end - t_start);
        fprintf(f, "isend %lld irecv %lld send %lld recv %lld\n",
                n_isend, n_irecv, n_send, n_recv);
        fprintf(f, "wait %lld waitall %lld test %lld\n",
                n_wait, n_waitall, n_test);
        fprintf(f, "barrier %lld reduce %lld allreduce %lld "
                   "bcast %lld gather %lld allgather %lld alltoall %lld\n",
                n_barrier, n_reduce, n_allreduce,
                n_bcast, n_gather, n_allgather, n_alltoall);
        fprintf(f, "bytes_sent %lld bytes_recvd %lld\n",
                bytes_sent, bytes_recvd);
        fclose(f);
    }
    return PMPI_Finalize();
}
