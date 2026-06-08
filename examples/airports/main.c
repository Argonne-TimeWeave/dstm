#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <dstm.h>
#include "airports.h"

static void usage(const char *prog)
{
    fprintf(stderr, "usage: %s [--seed S] [--time T] [--quiet]\n", prog);
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

    dstm_init(&argc, &argv);
    dstm_log_init(stdout, "airport_sim");

    if (quiet)
        dstm_log_setlevel(DSTM_LOG_QUIET);
    else
        dstm_log_setlevel(DSTM_LOG_NORMAL);

    dstm_distance_t *topology;
    int status = dstm_distance_create_mesh(&topology, num_ranks);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to create mesh topology");
        MPI_Finalize();
        return -1;
    }

    int cols = num_ranks;
    for (int r = 1; r * r <= num_ranks; r++) {
        if (num_ranks % r == 0)
            cols = num_ranks / r;
    }
    int rows = num_ranks / cols;

    /* Assign tiers (deterministic from seed, same on all ranks) */
    srand(seed);
    airport_tier_t *tiers = malloc(sizeof(airport_tier_t) * num_ranks);
    for (int i = 0; i < num_ranks; i++) {
        int r = rand() % 100;
        if (r < 15)
            tiers[i] = TIER_HUB;
        else if (r < 50)
            tiers[i] = TIER_REGIONAL;
        else
            tiers[i] = TIER_LOCAL;
    }

    if (rank == 0) {
        printf("Mesh topology: %d airports (%dx%d grid), "
                        "%d gates/airport, layover threshold=%d\n",
                        num_ranks, rows, cols, NUM_GATES, LAYOVER_THRESHOLD);
        for (int i = 0; i < num_ranks; i++)
            printf("  Airport %d: %s\n", i, tier_name[tiers[i]]);
    }

    status = dstm_start(topology);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to start DSTM");
        free(tiers);
        MPI_Finalize();
        return -1;
    }

    dstm_darray_t *gates;
    status = dstm_darray_create_equal(&gates,
                                      sizeof(gate_slot_t),
                                      (size_t)num_ranks * NUM_GATES);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to create gates array");
        free(tiers);
        MPI_Finalize();
        return -1;
    }

    /* Per-rank RNG */
    srand(seed + rank);

    airport_state_t state;
    airport_init(&state, rank, num_ranks, tiers, topology, gates);

    /* Main simulation loop */
    for (dstm_time_t t = 0; t < timesteps; t++) {
        airport_timestep(&state, t, quiet);
    }

    dstm_time_t last_time = timesteps + 1;
    status = dstm_barrier(last_time);
    if (status != DSTM_SUCCESS)
        dstm_log_error("Barrier failed");

    /* Gather and print summary */
    int *all_departures  = NULL;
    int *all_arrivals    = NULL;
    int *all_connections = NULL;
    if (rank == 0) {
        all_departures  = malloc(sizeof(int) * num_ranks);
        all_arrivals    = malloc(sizeof(int) * num_ranks);
        all_connections = malloc(sizeof(int) * num_ranks);
    }

    MPI_Gather(&state.departures, 1, MPI_INT,
               all_departures, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&state.arrivals, 1, MPI_INT,
               all_arrivals, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&state.connections, 1, MPI_INT,
               all_connections, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int total_dep = 0, total_arr = 0, total_conn = 0;
        fprintf(stdout,
                "\n=============== FLIGHT SUMMARY ===============\n"
                "%-8s %-10s %10s %10s %12s\n",
                "Airport", "Tier", "Departures", "Arrivals", "Connections");
        fprintf(stdout,
                "-------- ---------- ---------- ---------- ------------\n");
        for (int i = 0; i < num_ranks; i++) {
            fprintf(stdout, "%-8d %-10s %10d %10d %12d\n",
                    i, tier_name[tiers[i]],
                    all_departures[i], all_arrivals[i], all_connections[i]);
            total_dep  += all_departures[i];
            total_arr  += all_arrivals[i];
            total_conn += all_connections[i];
        }
        fprintf(stdout,
                "-------- ---------- ---------- ---------- ------------\n"
                "%-8s %-10s %10d %10d %12d\n"
                "==============================================\n",
                "TOTAL", "", total_dep, total_arr, total_conn);
        free(all_departures);
        free(all_arrivals);
        free(all_connections);
    }

    airport_destroy(&state);
    dstm_darray_destroy(&gates);

    status = dstm_fini(last_time);
    if (status != DSTM_SUCCESS)
        dstm_log_error("Failed to finalize DSTM");

    dstm_distance_destroy(&topology);
    free(tiers);
    dstm_finalize();
    MPI_Finalize();
    return status;
}
