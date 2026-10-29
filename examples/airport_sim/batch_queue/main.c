#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <dstm.h>
#include "airport.h"

#include "internal/distance.h"

/*
 * Airport torus benchmark on dstm_dqueue.
 *
 * The queue twin of simple_dstm (dstm_darray) and graph_dstm (dstm_dgraph).
 * Planes route by owning rank rather than by rank-pair slot (darray) or
 * directed edge (dgraph); all producers append into the owner's single
 * versioned slot, so there is no per-neighbor mailbox to index.
 *
 * The rank-level distance matrix is the SAME torus adjacency the other two
 * backends build: delta=1 between ranks sharing a corridor, -1 otherwise,
 * 0 on the diagonal. It bounds who can ever push to whom.
 */

static void usage(const char *prog)
{
        fprintf(stderr,
                "usage: %s [--seed S] [--time T] [--na N] [--planes P]\n"
                "          [--cap C] [--quiet | --info]\n"
                "  --na N       airports per rank (total = N * num_ranks;\n"
                "               total must be a perfect square for the torus)\n"
                "  --planes P   initial planes per airport (default 1)\n"
                "  --cap C      force queue capacity (default: derived)\n",
                prog);
}

/*
 * Rank-level torus adjacency, identical in result to simple_dstm's
 * build_rank_adjacency() and graph_dstm's distance_from_topology(): two ranks
 * are reachable (delta=1) iff some airport on one has a torus neighbor on the
 * other. Purely local and closed-form, so every rank derives the same matrix.
 */
static void build_rank_adjacency(dstm_distance_t *topology,
                                 int num_ranks, int na, int grid)
{
        for (int i = 0; i < num_ranks; i++)
                for (int j = 0; j < num_ranks; j++)
                        topology->distances[i * num_ranks + j] = -1;
        for (int i = 0; i < num_ranks; i++)
                topology->distances[i * num_ranks + i] = 0;

        int total = na * num_ranks;
        for (int gid = 0; gid < total; gid++) {
                int owner = gid / na;
                int row   = gid / grid;
                int col   = gid % grid;
                int neigh[4] = {
                        ((row - 1 + grid) % grid) * grid + col,
                        ((row + 1) % grid)        * grid + col,
                          row * grid + ((col + 1) % grid),
                          row * grid + ((col - 1 + grid) % grid),
                };
                for (int k = 0; k < 4; k++) {
                        int n_owner = neigh[k] / na;
                        if (n_owner == owner)
                                continue;
                        topology->distances[owner   * num_ranks + n_owner] = 1;
                        topology->distances[n_owner * num_ranks + owner]   = 1;
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
        long force_cap         = 0;

        for (int i = 1; i < argc; i++) {
                if (!strcmp(argv[i], "--seed") && i + 1 < argc)
                        seed = atoi(argv[++i]);
                else if (!strcmp(argv[i], "--time") && i + 1 < argc)
                        timesteps = atoi(argv[++i]);
                else if (!strcmp(argv[i], "--na") && i + 1 < argc)
                        na = atoi(argv[++i]);
                else if (!strcmp(argv[i], "--planes") && i + 1 < argc)
                        planes_per_airport = atoi(argv[++i]);
                else if (!strcmp(argv[i], "--cap") && i + 1 < argc)
                        force_cap = atol(argv[++i]);
                else if (!strcmp(argv[i], "--quiet"))
                        quiet = 1;
                else if (!strcmp(argv[i], "--info"))
                        quiet = 2;
                else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
                        usage(argv[0]);
                        return 0;
                }
        }
        if (na < 1)                 na = 1;
        if (planes_per_airport < 1) planes_per_airport = 1;
        if (timesteps < 1)          timesteps = 1;

        int rank, num_ranks;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

        dstm_init(&argc, &argv);
        dstm_log_init(stdout, "airport_queue");
        dstm_log_setlevel(quiet == 0 ? DSTM_LOG_NORMAL
                        : quiet == 2 ? DSTM_LOG_INFO
                                     : DSTM_LOG_QUIET);

        int total_airports = na * num_ranks;
        int grid = (int)(sqrt((double)total_airports) + 0.5);
        if (grid * grid != total_airports || grid < 2) {
                if (rank == 0)
                        fprintf(stderr,
                                "ERROR: na * num_ranks = %d must be a perfect "
                                "square >= 4 (e.g. na=1024 np=1, na=64 np=16).\n",
                                total_airports);
                dstm_finalize();
                MPI_Finalize();
                return -1;
        }

        /* --- Distance matrix (torus rank adjacency) ----------------------- */
        dstm_distance_t *topology = NULL;
        int status = dstm_distance_create_mesh(&topology, num_ranks);
        if (status != DSTM_SUCCESS) {
                dstm_log_error("Failed to create mesh topology");
                dstm_finalize();
                MPI_Finalize();
                return -1;
        }
        build_rank_adjacency(topology, num_ranks, na, grid);

        /* One element = one batch. At most (#neighbor ranks + self seed)
         * batches land in a slot per tick, independent of planes. */
        size_t derived_cap = QUEUE_CAPACITY_FLOOR;   /* fixed, >= 5 */
        size_t capacity = force_cap > 0 ? (size_t)force_cap : derived_cap;

        if (rank == 0) {
                printf("Airport torus (dqueue): %d airports (%dx%d), %d ranks, "
                       "na=%d, planes/airport=%d, timesteps=%d\n"
                       "  element %zu B, queue capacity %zu elements/rank\n",
                       total_airports, grid, grid, num_ranks, na,
                       planes_per_airport, timesteps,
                       sizeof(plane_msg_t), capacity);
        }

        status = dstm_start(topology);
        if (status != DSTM_SUCCESS) {
                dstm_log_error("Failed to start DSTM");
                dstm_distance_destroy(&topology);
                dstm_finalize();
                MPI_Finalize();
                return -1;
        }

        /* --- Create the distributed queue over the torus neighbor set ----- */
        dstm_dqueue_t *queue = NULL;
        status = dstm_dqueue_create(&queue, sizeof(plane_msg_t),
                                    capacity, topology);
        if (status != DSTM_SUCCESS) {
                dstm_log_error("Failed to create dqueue: %d", status);
                dstm_fini(timesteps + 1);
                dstm_distance_destroy(&topology);
                dstm_finalize();
                MPI_Finalize();
                return -1;
        }

        /* FAIRNESS: identical seeding to the other two backends. Must come
         * after dstm_start/create so nothing reseeds afterward. */
        srand(seed + rank);

        airport_state_t state;
        airport_init(&state, rank, num_ranks, na, grid,
                     planes_per_airport, topology, queue);

        long long local_initial  =
                (long long)na * (long long)planes_per_airport;
        long long global_initial = 0;
        MPI_Allreduce(&local_initial, &global_initial, 1, MPI_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();

        /* No per-tick barrier: ranks drift and the queue's version
         * visibility orders the transport. */
        for (dstm_time_t t = 0; t < timesteps; t++)
                airport_timestep(&state, t);

        /* Final drain: the last tick's push published version `timesteps`;
         * pop it so planes still in flight are counted for conservation. */
        dstm_time_t t_last = (dstm_time_t)timesteps;
        long long local_inflight = airport_drain_inflight(&state, t_last);

        dstm_time_t last_time = timesteps + 1;
        if (dstm_barrier(last_time) != DSTM_SUCCESS)
                dstm_log_error("Barrier failed");

        double elapsed_local = MPI_Wtime() - t_start;

        /* --- Reduce & report --------------------------------------------- */
        long long my_land = 0, my_ground = 0, my_pending = 0;
        for (int l = 0; l < na; l++) {
                my_land    += state.landings[l];
                my_ground  += state.planes_on_the_ground[l];
                my_pending += state.pending_landings[l];
        }

        /* ground + pending covers every plane this rank currently holds;
         * the just-drained in-flight planes were added to pending. */
        long long lsend[6] = { my_land, state.departures, state.arrivals,
                               my_ground + my_pending, state.overflows,
                               state.misroutes };
        long long lrecv[6] = { 0 };
        double elapsed = 0.0;

        MPI_Reduce(&elapsed_local, &elapsed, 1, MPI_DOUBLE, MPI_MAX,
                   0, MPI_COMM_WORLD);
        MPI_Reduce(lsend, lrecv, 6, MPI_LONG_LONG, MPI_SUM,
                   0, MPI_COMM_WORLD);

        long long global_inflight = 0;
        MPI_Reduce(&local_inflight, &global_inflight, 1, MPI_LONG_LONG,
                   MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
                long long accounted = lrecv[3];   /* ground + pending */
                double tsps = elapsed > 0.0 ? timesteps / elapsed : 0.0;

                fprintf(stdout,
                "\n=============== RESULTS ===============\n"
                "Backend:            dstm_dqueue\n"
                "Airports:           %d  (%d x %d torus)\n"
                "Ranks:              %d   (na=%d)\n"
                "Planes per airport: %d\n"
                "Queue capacity:     %zu elements/rank\n"
                "Timesteps:          %d\n"
                "Total departures:   %lld\n"
                "Total landings:     %lld\n"
                "Cross-rank arrivals:%lld\n"
                "Queue overflows:    %lld\n"
                "Misrouted flights:  %lld\n"
                "Final drained:      %lld\n"
                "Conservation:       %lld / %lld  %s\n"
                "\n"
                "Elapsed time:       %.4f seconds\n"
                "Timesteps/second:   %.2f\n"
                "========================================\n",
                total_airports, grid, grid,
                num_ranks, na, planes_per_airport, capacity, timesteps,
                lrecv[1], lrecv[0], lrecv[2], lrecv[4], lrecv[5],
                global_inflight,
                accounted, global_initial,
                accounted == global_initial ? "OK" : "MISMATCH",
                elapsed, tsps);
        }

        airport_destroy(&state);
        dstm_dqueue_destroy(&queue);

        status = dstm_fini(last_time);
        if (status != DSTM_SUCCESS)
                dstm_log_error("Failed to finalize DSTM");

        dstm_distance_destroy(&topology);
        dstm_finalize();
        MPI_Finalize();
        return status == DSTM_SUCCESS ? 0 : -1;
}
