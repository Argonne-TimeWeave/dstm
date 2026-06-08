#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <dstm.h>
#include "simulation.h"

static void usage(const char *prog)
{
    fprintf(stderr, "usage: %s [--seed S] [--time T] [--quiet]\n", prog);
    fprintf(stderr, "\nStrong scaling: 64 airports (8x8 grid), variable ranks\n");
    fprintf(stderr, "Valid rank counts: 1, 2, 4, 8, 16, 32, 64\n");
}

int main(int argc, char *argv[])
{
    int seed      = 42;
    int quiet     = 0;
    int timesteps = 16;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--time") && i + 1 < argc)
            timesteps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quiet"))
            quiet = 1;
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        }
    }

     int rank, num_ranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    fprintf(stderr, "[Rank %d] MPI initialized, num_ranks=%d\n", rank, num_ranks);
    fflush(stderr);

    dstm_init(&argc, &argv);
    fprintf(stderr, "[Rank %d] DSTM initialized\n", rank);
    fflush(stderr);

    dstm_log_init(stdout, "airport_scaling");

    if (quiet)
        dstm_log_setlevel(DSTM_LOG_QUIET);
    else
        dstm_log_setlevel(DSTM_LOG_NORMAL);

    /* Validate rank count */
    if (NUM_AIRPORTS % num_ranks != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: num_ranks (%d) must divide %d airports evenly\n",
                    num_ranks, NUM_AIRPORTS);
        }
        MPI_Finalize();
        return -1;
    }

    fprintf(stderr, "[Rank %d] Validation passed\n", rank);
    fflush(stderr);

    /* Create topology based on rank grid */
    dstm_distance_t *topology;
    int status = dstm_distance_create_mesh(&topology, num_ranks);
    if (status != DSTM_SUCCESS) {
        fprintf(stderr, "[Rank %d] Failed to create mesh topology, status=%d\n", rank, status);
        dstm_log_error("Failed to create mesh topology");
        MPI_Finalize();
        return -1;
    }

    fprintf(stderr, "[Rank %d] Topology created\n", rank);
    fflush(stderr);

    status = dstm_start(topology);
    if (status != DSTM_SUCCESS) {
        fprintf(stderr, "[Rank %d] Failed to start DSTM, status=%d\n", rank, status);
        dstm_log_error("Failed to start DSTM");
        MPI_Finalize();
        return -1;
    }

    fprintf(stderr, "[Rank %d] DSTM started\n", rank);
    fflush(stderr);

    /* Create gate array */
    dstm_darray_t *gates;
    status = dstm_darray_create_equal(&gates,
                                      sizeof(gate_slot_t),
                                      (size_t)NUM_AIRPORTS * NUM_GATES);
    if (status != DSTM_SUCCESS) {
        fprintf(stderr, "[Rank %d] Failed to create gates array, status=%d\n", rank, status);
        dstm_log_error("Failed to create gates array");
        MPI_Finalize();
        return -1;
    }

    fprintf(stderr, "[Rank %d] Gates array created (size=%zu)\n", rank, (size_t)NUM_AIRPORTS * NUM_GATES);
    fflush(stderr);

    simulation_state_t sim;
    status = simulation_init(&sim, rank, num_ranks, seed, gates);
    if (status != 0) {
        fprintf(stderr, "[Rank %d] Failed to initialize simulation, status=%d\n", rank, status);
        dstm_log_error("Failed to initialize simulation");
        MPI_Finalize();
        return -1;
    }

    fprintf(stderr, "[Rank %d] Simulation initialized, owning %d airports\n", rank, sim.num_my_airports);
    fflush(stderr);

    if (rank == 0) {
        printf("=== STRONG SCALING AIRPORTS SIMULATION ===\n");
        printf("Grid: %dx%d = %d airports\n", GRID_ROWS, GRID_COLS, NUM_AIRPORTS);
        printf("Ranks: %d (%dx%d rank grid)\n", num_ranks, sim.rank_rows, sim.rank_cols);
        printf("Airports per rank: %d (%dx%d block)\n",
               sim.num_my_airports, sim.airports_per_rank_row, sim.airports_per_rank_col);
        printf("Gates per airport: %d\n", NUM_GATES);
        printf("Layover threshold: %d\n", LAYOVER_THRESHOLD);
        printf("Timesteps: %d\n", timesteps);
        printf("Seed: %d\n\n", seed);

        printf("Airport tiers:\n");
        for (int r = 0; r < GRID_ROWS; r++) {
            printf("  ");
            for (int c = 0; c < GRID_COLS; c++) {
                int id = r * GRID_COLS + c;
                char tier_char = (sim.tiers[id] == TIER_HUB) ? 'H' :
                                 (sim.tiers[id] == TIER_REGIONAL) ? 'R' : 'L';
                printf("%c ", tier_char);
            }
            printf("\n");
        }
        printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    /* Main simulation loop */
    for (dstm_time_t t = 0; t < timesteps; t++) {
        simulation_timestep(&sim, t, quiet);
    }

    dstm_time_t last_time = timesteps + 1;
    status = dstm_barrier(last_time);
    if (status != DSTM_SUCCESS)
        dstm_log_error("Barrier failed");

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    /* Gather statistics */
    int local_dep, local_arr, local_conn;
    simulation_gather_stats(&sim, &local_dep, &local_arr, &local_conn);

    int total_dep, total_arr, total_conn;
    MPI_Reduce(&local_dep, &total_dep, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_arr, &total_arr, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_conn, &total_conn, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n=============== RESULTS ===============\n");
        printf("Total departures:   %d\n", total_dep);
        printf("Total arrivals:     %d\n", total_arr);
        printf("Total connections:  %d\n", total_conn);
        printf("\n");
        printf("Elapsed time:       %.4f seconds\n", max_elapsed);
        printf("Timesteps/second:   %.2f\n", timesteps / max_elapsed);
        printf("========================================\n");
    }

    simulation_destroy(&sim);
    dstm_darray_destroy(&gates);

    status = dstm_fini(last_time);
    if (status != DSTM_SUCCESS)
        dstm_log_error("Failed to finalize DSTM");

    dstm_distance_destroy(&topology);
    dstm_finalize();
    MPI_Finalize();
    return 0;
}
