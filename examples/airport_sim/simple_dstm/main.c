#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <dstm.h>
#include "airport.h"

#include "internal/distance.h"

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s [--seed S] [--time T] [--na N] [--planes P] [--quiet]\n"
            "  --na N       airports per rank (total airports = N * num_ranks;\n"
            "               total must be a perfect square for the torus)\n"
            "  --planes P   initial planes per airport (default 1)\n",
            prog);
}

/*
 * Prune the mesh distance matrix so only ranks that host at least one
 * torus-neighbor pair are marked reachable (delta=1). All other entries -1.
 *
 * We consider two ranks connected iff there exists at least one airport on
 * rank A whose torus neighbor (N/S/E/W wrap) lives on rank B. This is what
 * DSTM needs: rank-level reachability derived from the airport-level torus.
 */
static void build_rank_adjacency(dstm_distance_t *topology,
                                 int num_ranks, int na, int grid)
{
    /* zero out (mark unreachable) */
    for (int i = 0; i < num_ranks; i++)
        for (int j = 0; j < num_ranks; j++)
            topology->distances[i * num_ranks + j] = -1;

    /* Walk all airports; for each neighbor cross-rank edge, set delta=1. */
    int total = na * num_ranks;
    for (int gid = 0; gid < total; gid++) {
        int owner = gid / na;
        int row = gid / grid;
        int col = gid % grid;
        int neigh[4] = {
            ((row - 1 + grid) % grid) * grid + col,
            ((row + 1) % grid)         * grid + col,
              row * grid + ((col + 1) % grid),
              row * grid + ((col - 1 + grid) % grid),
        };
        for (int k = 0; k < 4; k++) {
            int n_owner = neigh[k] / na;
            if (n_owner == owner)
                continue;
            topology->distances[owner * num_ranks + n_owner] = 1;
            topology->distances[n_owner * num_ranks + owner] = 1;
        }
    }
}

int main(int argc, char *argv[])
{
    int seed               = 42;
    int quiet              = 0;
    int timesteps          = 16;
    int na                 = 1;   /* airports per rank */
    int planes_per_airport = 1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--time") && i + 1 < argc)
            timesteps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--na") && i + 1 < argc)
            na = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--planes") && i + 1 < argc)
            planes_per_airport = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quiet"))
            quiet = 1;
        else if (!strcmp(argv[i], "--info"))
            quiet = 2;
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        }
    }
    if (na < 1) na = 1;
    if (planes_per_airport < 1) planes_per_airport = 1;

    int rank, num_ranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    dstm_init(&argc, &argv);
    dstm_log_init(stdout, "airport_sim");

    if (quiet) {
        dstm_log_setlevel(DSTM_LOG_QUIET);
        if (quiet == 2)
            dstm_log_setlevel(DSTM_LOG_INFO);
    } else {
        dstm_log_setlevel(DSTM_LOG_NORMAL);
    }

    /* Total airports must be a perfect square for the torus. */
    int total_airports = na * num_ranks;
    int grid = (int)(sqrt((double)total_airports) + 0.5);
    if (grid * grid != total_airports) {
        if (rank == 0)
            fprintf(stderr,
                    "ERROR: total airports (na * num_ranks) = %d is not a "
                    "perfect square. Use e.g. na=1024, np=1, or na=64 np=16.\n",
                    total_airports);
        MPI_Finalize();
        return -1;
    }

    dstm_distance_t *topology;
    int status = dstm_distance_create_mesh(&topology, num_ranks);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to create mesh topology");
        MPI_Finalize();
        return -1;
    }
    build_rank_adjacency(topology, num_ranks, na, grid);

    if (rank == 0) {
        printf("Airport torus: %d airports (%dx%d), %d ranks, "
               "na=%d, planes/airport=%d, timesteps=%d\n",
               total_airports, grid, grid, num_ranks, na,
               planes_per_airport, timesteps);
    }

    status = dstm_start(topology);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to start DSTM");
        MPI_Finalize();
        return -1;
    }

    dstm_darray_t *gates;
    status = dstm_darray_create_equal(&gates,
                                      sizeof(gate_slot_t),
                                      (size_t)num_ranks * num_ranks);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to create gates array");
        MPI_Finalize();
        return -1;
    }

    srand(seed + rank);

    airport_state_t state;
    airport_init(&state, rank, num_ranks, na, grid,
                 planes_per_airport, topology, gates);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* Prime sentinel state at t=0 so first GETs don't stall. */
    for (dstm_time_t t = 0; t < timesteps; t++) {
        for (int g = 0; g < num_ranks; g++) {
            int delta = dstm_distance_get(topology, rank, g);
            if (delta == -1)
                continue;
            size_t idx = (size_t)rank * num_ranks + g;
            dstm_darray_put(state.gates, idx, &empty_gate, 0, t);
        }
    }

    for (dstm_time_t t = 0; t < timesteps; t++) {
        airport_timestep(&state, t);
    }

    dstm_time_t last_time = timesteps + 1;
    status = dstm_barrier(last_time);
    if (status != DSTM_SUCCESS)
        dstm_log_error("Barrier failed");

    double t_end = MPI_Wtime();
    double elapsed_local = t_end - t_start;
    double elapsed = 0.0;
    MPI_Reduce(&elapsed_local, &elapsed, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    long long my_landings = 0;
    for (int l = 0; l < na; l++) my_landings += state.landings[l];
    long long total_landings = 0, total_departures = 0;
    MPI_Reduce(&my_landings,        &total_landings,   1, MPI_LONG_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&state.departures,   &total_departures, 1, MPI_LONG_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double tsps = (elapsed > 0.0) ? (timesteps / elapsed) : 0.0;
        fprintf(stdout,
                "\n=============== RESULTS ===============\n"
                "Airports:           %d  (%d x %d torus)\n"
                "Ranks:              %d   (na=%d)\n"
                "Planes per airport: %d\n"
                "Timesteps:          %d\n"
                "Total departures:   %lld\n"
                "Total landings:     %lld\n"
                "\n"
                "Elapsed time:       %.4f seconds\n"
                "Timesteps/second:   %.2f\n"
                "========================================\n",
                total_airports, grid, grid,
                num_ranks, na, planes_per_airport, timesteps,
                total_departures, total_landings,
                elapsed, tsps);
    }

    airport_destroy(&state);
    dstm_darray_destroy(&gates);

    status = dstm_fini(last_time);
    if (status != DSTM_SUCCESS)
        dstm_log_error("Failed to finalize DSTM");

    dstm_distance_destroy(&topology);
    dstm_finalize();
    MPI_Finalize();
    return status;
}
