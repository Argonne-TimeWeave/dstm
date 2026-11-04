#include <assert.h>
#include <mpi.h>
#include "dstm.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    dstm_init(&argc, &argv);
    dstm_log_setlevel(DSTM_LOG_DEBUG);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    dstm_distance_t *dist = NULL;
    assert(dstm_distance_create_complete(&dist, size) == DSTM_SUCCESS);
    assert(dstm_start(dist) == DSTM_SUCCESS);

    fprintf(stderr, "before dstm_fini\n");
    dstm_fini(0);
    fprintf(stderr, "after dstm_fini\n");

    dstm_distance_destroy(&dist);
    fprintf(stderr, "after distance destroy\n");

    dstm_finalize();
    fprintf(stderr, "after dstm_finalize\n");

    MPI_Finalize();
    return 0;
}
