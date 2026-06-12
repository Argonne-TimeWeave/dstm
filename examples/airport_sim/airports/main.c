#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <dstm.h>
#include "airport.h"

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s [--seed S] [--time T] [--na N] [--quiet]\n", prog);
}

int main(int argc, char *argv[])
{
    int seed      = 42;
    int quiet     = 0;
    int timesteps = 16;
    int na        = 1;   /* airports per rank */
    int work      = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--time") && i + 1 < argc)
            timesteps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--na") && i + 1 < argc)
            na = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--work") && i + 1 < argc)
            work = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quiet"))
            quiet = 1;
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        }
    }

    if (na < 1) na = 1;
    if (work < 0) work = 0;

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

    int total_airports = num_ranks * na;

    /* Assign tiers per local airport (deterministic from seed, same on all ranks) */
    srand(seed);
    airport_tier_t *tiers = malloc(sizeof(airport_tier_t) * total_airports);
    for (int i = 0; i < total_airports; i++) {
        int r = rand() % 100;
        if (r < 15)
            tiers[i] = TIER_HUB;
        else if (r < 50)
            tiers[i] = TIER_REGIONAL;
        else
            tiers[i] = TIER_LOCAL;
    }

    if (rank == 0) {
        printf("Mesh topology: %d ranks (%dx%d grid), %d airports/rank "
               "(%d total), %d gates/rank, layover threshold=%d, work=%d\n",
               num_ranks, rows, cols, na, total_airports,
               NUM_GATES, LAYOVER_THRESHOLD, work);
        /*
        for (int r = 0; r < num_ranks; r++) {
            for (int l = 0; l < na; l++) {
                int g = airport_global(r, l, na);
                printf("  Airport %d.%d: %s\n", r, l, tier_name[tiers[g]]);
            }
        }
        */
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
    airport_init(&state, rank, num_ranks, na, work, tiers, topology, gates);

    /* Sync all ranks before timing the simulation loop */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* Main simulation loop */
    for (dstm_time_t t = 0; t < timesteps; t++) {
        airport_timestep(&state, t, quiet);
    }

    dstm_time_t last_time = timesteps + 1;
    status = dstm_barrier(last_time);
    if (status != DSTM_SUCCESS)
        dstm_log_error("Barrier failed");

    double t_end = MPI_Wtime();
    double elapsed_local = t_end - t_start;
    double elapsed = 0.0;

    MPI_Reduce(&elapsed_local, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Gather per-local-airport counters from all ranks.
     * Each rank contributes `na` ints. */
    int *all_departures  = NULL;
    int *all_arrivals    = NULL;
    int *all_connections = NULL;
    if (rank == 0) {
        all_departures  = malloc(sizeof(int) * total_airports);
        all_arrivals    = malloc(sizeof(int) * total_airports);
        all_connections = malloc(sizeof(int) * total_airports);
    }

    MPI_Gather(state.departures, na, MPI_INT,
               all_departures, na, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(state.arrivals, na, MPI_INT,
               all_arrivals, na, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(state.connections, na, MPI_INT,
               all_connections, na, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int total_dep = 0, total_arr = 0, total_conn = 0;
        fprintf(stdout,
                "\n=============== FLIGHT SUMMARY ===============\n"
                "%-10s %-10s %10s %10s %12s\n",
                "Airport", "Tier", "Departures", "Arrivals", "Connections");
        fprintf(stdout,
                "---------- ---------- ---------- ---------- ------------\n");
        for (int r = 0; r < num_ranks; r++) {
            for (int l = 0; l < na; l++) {
                int g = airport_global(r, l, na);
                char label[16];
                /*
                snprintf(label, sizeof(label), "%d.%d", r, l);
                fprintf(stdout, "%-10s %-10s %10d %10d %12d\n",
                        label, tier_name[tiers[g]],
                        all_departures[g],
                        all_arrivals[g],
                        all_connections[g]);
                */
                total_dep  += all_departures[g];
                total_arr  += all_arrivals[g];
                total_conn += all_connections[g];
            }
        }
        fprintf(stdout,
                "---------- ---------- ---------- ---------- ------------\n"
                "%-10s %-10s %10d %10d %12d\n"
                "==============================================\n",
                "TOTAL", "", total_dep, total_arr, total_conn);

        double tsps = (elapsed > 0.0) ? (timesteps / elapsed) : 0.0;
        fprintf(stdout,
                "\n=============== RESULTS ===============\n"
                "Total departures:   %d\n"
                "Total arrivals:     %d\n"
                "Total connections:  %d\n"
                "\n"
                "Elapsed time:       %.4f seconds\n"
                "Timesteps/second:   %.2f\n"
                "========================================\n",
                total_dep, total_arr, total_conn,
                elapsed, tsps);

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
