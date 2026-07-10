/*
 * Zombie simulation on top of DSTM.
 *
 * Simplified to match the RepastHPC "Zombies" demo model:
 *   - integer ticks (no hour-of-day / day-of concept)
 *   - N humans + M zombies per rank, seeded at random local cells
 *   - each tick: zombies move toward neighboring cell with most humans;
 *                humans move toward neighboring cell with fewest zombies;
 *                a zombie infects ONE healthy human sharing its cell;
 *                infected humans become zombies after 50 ticks
 *
 * Input arguments:
 *   --seed S     RNG seed                 (default 42)
 *   --time T     ticks to simulate        (default 100)
 *   --nl   N     local cells per rank     (default 4; snapped up to
 *                                          next perfect square for the
 *                                          Moore neighborhood)
 *   --nh   N     humans per rank          (default 500)  -- Repast: human.count
 *   --nz   N     zombies per rank         (default 5)    -- Repast: zombie.count
 *   --work W     synthetic work knob      (default 0)
 *   --infcsv F   per-tick infection CSV   (default ./infection.csv, rank 0 only)
 *   --quiet | --info
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <dstm.h>

#include "internal/distance.h"

#include "human.h"
#include "zombie.h"

static void usage(const char *prog)
{
    fprintf(stderr,
        "usage: %s [--seed S] [--time T] [--nl N] [--nh N] [--nz N] "
        "[--work W] [--infcsv F] [--quiet|--info]\n", prog);
}

int main(int argc, char *argv[])
{
    int seed      = 42;
    int quiet     = 0;
    int timesteps = 100;
    int nl        = 4;
    int nh        = 500;
    int nz        = 5;
    int work      = 0;
    const char *infcsv_path = "infection.csv";

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--seed")   && i+1 < argc) seed        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--time")   && i+1 < argc) timesteps   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nl")     && i+1 < argc) nl          = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nh")     && i+1 < argc) nh          = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nz")     && i+1 < argc) nz          = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--work")   && i+1 < argc) work        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--infcsv") && i+1 < argc) infcsv_path = argv[++i];
        else if (!strcmp(argv[i], "--quiet"))                quiet       = 1;
        else if (!strcmp(argv[i], "--info"))                 quiet       = 2;
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]); return 0;
        }
    }

    if (nh   < 0) nh   = 0;
    if (nz   < 0) nz   = 0;
    if (nh + nz < 1) nh = 1;
    if (work < 0) work = 0;
    if (nl   < 1) nl   = 1;

    int rank, num_ranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    dstm_init(&argc, &argv);
    dstm_log_init(stdout, "zombie_sim");

    if (quiet) {
        dstm_log_setlevel(DSTM_LOG_QUIET);
        if (quiet == 2)
            dstm_log_setlevel(DSTM_LOG_INFO);
    } else {
        dstm_log_setlevel(DSTM_LOG_NORMAL);
    }

    /* Build mesh topology. */
    dstm_distance_t *topology;
    int status = dstm_distance_create_mesh(&topology, num_ranks);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to create mesh topology");
        MPI_Finalize();
        return -1;
    }

    /* Factor num_ranks into a rows x cols rank grid (as square as possible). */
    int cols = num_ranks;
    for (int r = 1; r * r <= num_ranks; r++) {
        if (num_ranks % r == 0) cols = num_ranks / r;
    }
    int rows = num_ranks / cols;

    /* Rewrite the distance matrix so that Moore-adjacent ranks (the 8
     * neighbors in the rows x cols rank grid, including diagonals) are
     * at distance 1, and every other rank is unreachable (-1). Self
     * stays at 0. This matches Repast HPC's default 2-D Moore ghost
     * topology: each rank exchanges with up to 8 neighbors per tick. */
    for (int i = 0; i < num_ranks; i++) {
        int ir = i / cols, ic = i % cols;
        for (int j = 0; j < num_ranks; j++) {
            int jr = j / cols, jc = j % cols;

            int dr = ir - jr; if (dr < 0) dr = -dr;
            int dc = ic - jc; if (dc < 0) dc = -dc;

            int d;
            if (i == j) {
                d = 0;                    /* self */
            } else if (dr <= 1 && dc <= 1) {
                d = 1;                    /* Moore neighbor (edge or diagonal) */
            } else {
                d = -1;                   /* unreachable in this topology */
            }
            topology->distances[i * num_ranks + j] = d;

            if (rank == 0)
                printf("%3d ", d);
        }
        if (rank == 0)
            printf("\n");
    }

    /* Compute the actual nl the sim will use (perfect square). */
    int cell_side = 1;
    while (cell_side * cell_side < nl) cell_side++;
    int nl_actual = cell_side * cell_side;

    int total_agents_per_rank = nh + nz;
    int total_cells           = num_ranks * nl_actual;
    int total_humans          = num_ranks * nh;
    int total_zombies         = num_ranks * nz;

    if (rank == 0) {
        printf("Zombie sim: %d ranks (%dx%d), %d locals/rank (%dx%d, %d total)\n"
                "Per rank: %d humans + %d zombies (%d agents)\n"
                "Totals:   %d humans + %d zombies\n"
                "Ticks: %d, cell cap=%d, incubation=%d ticks, "
                "zombie_step=%d, human_step=%d, work=%d\n",
                num_ranks, rows, cols, nl_actual, cell_side, cell_side, total_cells,
                nh, nz, total_agents_per_rank,
                total_humans, total_zombies,
                timesteps, CELL_CAPACITY, INFECT_TICKS_TO_ZOMBIE,
                ZOMBIE_STEP, HUMAN_STEP, work);
    }

    status = dstm_start(topology);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to start DSTM");
        MPI_Finalize();
        return -1;
    }

    /* Global darray — one bundle per (dest_rank, src_rank). */
    dstm_darray_t *cells;
    status = dstm_darray_create_equal(&cells,
            sizeof(cell_slot_t),
            (size_t)num_ranks * (size_t)num_ranks);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to create cells darray");
        MPI_Finalize();
        return -1;
    }

    /* Per-rank RNG seed (kept for any legacy callers of rand()). */
    srand(seed + rank);

    /* Open the per-tick infection CSV on rank 0 only. */
    FILE *infection_csv = NULL;
    if (rank == 0) {
        infection_csv = fopen(infcsv_path, "w");
        if (infection_csv) {
            fprintf(infection_csv, "tick,number_infected\n");
        } else {
            dstm_log_warning("Could not open %s for infection CSV; skipping",
                    infcsv_path);
        }
    }

    zombie_state_t state;
    zombie_init(&state, rank, num_ranks, nl, nh, nz,
            work, seed + rank, topology, cells, infection_csv);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    double start = 0;
    double end   = 0;

    if (LOGGING) start = MPI_Wtime();

    for (dstm_time_t t = 0; t < timesteps; t++) {
        size_t idx = bundle_index(rank, rank, num_ranks);
        dstm_darray_put(state.cells, idx, &empty_cell, 0, t);
    }

    if (LOGGING) {
        end = MPI_Wtime();
        printf("_PUT: %f\n", end - start);
    }

    for (dstm_time_t t = 0; t < timesteps; t++) {
        zombie_timestep(&state, t);
    }

    dstm_time_t last_time = timesteps + 1;
    status = dstm_barrier(last_time);
    if (status != DSTM_SUCCESS) dstm_log_error("Barrier failed");

    double t_end = MPI_Wtime();
    double elapsed_local = t_end - t_start;
    double elapsed = 0.0;
    MPI_Reduce(&elapsed_local, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Gather counters + final population stats. */
    int *all_dep = NULL, *all_arr = NULL, *all_inf = NULL;
    int local_healthy = 0, local_infected = 0, local_zombie = 0;
    for (int i = 0; i < state.nh; i++) {
        human_t *h = &state.humans[i];
        switch (h->state) {
            case HSTATE_HEALTHY:  local_healthy++;  break;
            case HSTATE_INFECTED: local_infected++; break;
            case HSTATE_ZOMBIE:   local_zombie++;   break;
        }
    }

    if (rank == 0) {
        all_dep = malloc(sizeof(int) * total_cells);
        all_arr = malloc(sizeof(int) * total_cells);
        all_inf = malloc(sizeof(int) * total_cells);
    }
    MPI_Gather(state.departures, state.nl, MPI_INT, all_dep, state.nl, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(state.arrivals,   state.nl, MPI_INT, all_arr, state.nl, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(state.infections, state.nl, MPI_INT, all_inf, state.nl, MPI_INT, 0, MPI_COMM_WORLD);

    int tot_healthy = 0, tot_infected = 0, tot_zombie = 0;
    MPI_Reduce(&local_healthy,  &tot_healthy,  1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_infected, &tot_infected, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_zombie,   &tot_zombie,   1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int total_dep = 0, total_arr = 0, total_inf = 0;
        for (int i = 0; i < total_cells; i++) {
            total_dep += all_dep[i];
            total_arr += all_arr[i];
            total_inf += all_inf[i];
        }

        int total_agents = tot_healthy + tot_infected + tot_zombie;

        printf("\n=============== INFECTION SUMMARY ===============\n"
                "%-10s %10s\n"
                "Healthy    %10d\n"
                "Infected   %10d\n"
                "Zombie     %10d\n"
                "-------------------------\n"
                "Total      %10d\n"
                "=================================================\n",
                "Status", "Count",
                tot_healthy, tot_infected, tot_zombie, total_agents);

        double tsps = (elapsed > 0.0) ? (timesteps / elapsed) : 0.0;
        printf("\n=============== RESULTS ===============\n"
                "Total departures:   %d\n"
                "Total arrivals:     %d\n"
                "Total infections:   %d\n"
                "\n"
                "Elapsed time:       %.4f seconds\n"
                "Timesteps/second:   %.2f\n"
                "========================================\n",
                total_dep, total_arr, total_inf, elapsed, tsps);

        free(all_dep); free(all_arr); free(all_inf);
    }

    if (infection_csv) fclose(infection_csv);

    zombie_destroy(&state);
    dstm_darray_destroy(&cells);

    int fini_status = dstm_fini(last_time);
    if (fini_status != DSTM_SUCCESS) dstm_log_error("Failed to finalize DSTM");

    dstm_distance_destroy(&topology);
    dstm_finalize();
    MPI_Finalize();
    return 0;
}
