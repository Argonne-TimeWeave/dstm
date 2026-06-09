#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "simulation.h"

static void usage(const char *prog)
{
    fprintf(stderr, "usage: %s [--seed S] [--time T] [--quiet]\n", prog);
    fprintf(stderr, "              [--block-rows R] [--block-cols C] [--work W]\n");
}

int main(int argc, char *argv[])
{
    int seed = 42, quiet = 0, timesteps = 16;
    int block_rows = 8, block_cols = 8, work = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc)       seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--time") && i + 1 < argc)  timesteps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--block-rows") && i+1<argc) block_rows = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--block-cols") && i+1<argc) block_cols = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quiet"))                  quiet = 1;
        else if (!strcmp(argv[i], "--work") && i + 1 < argc)   work = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]); return 0;
        }
    }

    g_work_iterations = work;
    g_log_quiet = quiet;

    if (block_rows <= 0 || block_cols <= 0) {
        fprintf(stderr, "Error: --block-rows and --block-cols must be positive\n");
        return -1;
    }

    int rank, num_ranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    simulation_state_t sim;
    if (simulation_init(&sim, rank, num_ranks, block_rows, block_cols, seed) != 0) {
        fprintf(stderr, "[Rank %d] simulation_init failed\n", rank);
        MPI_Finalize();
        return -1;
    }

    if (rank == 0) {
        printf("=== AIRPORTS SCALING SIMULATION (MPI Alltoall) ===\n");
        printf("Grid: %dx%d = %d airports\n", sim.grid_rows, sim.grid_cols, sim.num_airports);
        printf("Ranks: %d (%dx%d rank grid)\n", num_ranks, sim.rank_rows, sim.rank_cols);
        printf("Airports per rank: %d (%dx%d block)\n",
               sim.airports_per_rank, sim.block_rows, sim.block_cols);
        printf("Gates per airport: %d\n", NUM_GATES);
        printf("Layover threshold: %d\n", LAYOVER_THRESHOLD);
        printf("Timesteps: %d\n", timesteps);
        printf("Seed: %d\n\n", seed);

        printf("Airport tiers:\n");
        for (int r = 0; r < sim.grid_rows; r++) {
            printf("  ");
            for (int c = 0; c < sim.grid_cols; c++) {
                int id = r * sim.grid_cols + c;
                char tc = (sim.tiers[id] == TIER_HUB) ? 'H' :
                          (sim.tiers[id] == TIER_REGIONAL) ? 'R' : 'L';
                printf("%c ", tc);
            }
            printf("\n");
        }
        printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (dstm_time_t t = 0; t < timesteps; t++) {
        simulation_timestep(&sim, t, quiet);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    int local_dep, local_arr, local_conn;
    simulation_gather_stats(&sim, &local_dep, &local_arr, &local_conn);

    int total_dep, total_arr, total_conn;
    MPI_Reduce(&local_dep,  &total_dep,  1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_arr,  &total_arr,  1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_conn, &total_conn, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n=============== RESULTS ===============\n");
        printf("Total departures:   %d\n", total_dep);
        printf("Total arrivals:     %d\n", total_arr);
        printf("Total connections:  %d\n", total_conn);
        printf("\nElapsed time:       %.4f seconds\n", max_elapsed);
        printf("Timesteps/second:   %.2f\n", timesteps / max_elapsed);
        printf("========================================\n");
    }

    simulation_destroy(&sim);
    MPI_Finalize();
    return 0;
}
