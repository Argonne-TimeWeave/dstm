/*
 * Zombie / infection simulation on top of DSTM.
 *
 * Input arguments:
 *   --seed S         RNG seed (default 42)
 *   --time T         number of HOURS to simulate (default 72 = 3 days)
 *   --nl  N          local locations per rank (must be power of 2, default 4)
 *   --nh  N          humans per rank (default 8)
 *   --work W         synthetic work knob (default 0)
 *   --quiet          suppress per-hour logging
 *
 * One random human is infected at t=0.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <dstm.h>

#include "event.h"
#include "human.h"
#include "zombie.h"

static void usage(const char *prog)
{
    fprintf(stderr,
        "usage: %s [--seed S] [--time T] [--nl N] [--nh N] [--work W] [--quiet]\n",
        prog);
}

int main(int argc, char *argv[])
{
    int seed      = 42;
    int quiet     = 0;
    int timesteps = 72;   /* hours */
    int nl        = 4;    /* local cells per rank, power of 2 */
    int nh        = 8;    /* humans per rank */
    int work      = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--seed") && i+1 < argc) seed      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--time") && i+1 < argc) timesteps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nl")   && i+1 < argc) nl        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nh")   && i+1 < argc) nh        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--work") && i+1 < argc) work      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quiet"))               quiet     = 1;
        else if (!strcmp(argv[i], "--info"))                quiet     = 2;
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]); return 0;
        }
    }

    if (nh < 1)   nh = 1;
    if (work < 0) work = 0;

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
    } else
        dstm_log_setlevel(DSTM_LOG_NORMAL);

    /* Build mesh topology. */
    dstm_distance_t *topology;
    int status = dstm_distance_create_mesh(&topology, num_ranks);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to create mesh topology");
        MPI_Finalize();
        return -1;
    }

    int cols = num_ranks;
    for (int r = 1; r * r <= num_ranks; r++) {
        if (num_ranks % r == 0) cols = num_ranks / r;
    }
    int rows = num_ranks / cols;

    int total_cells  = num_ranks * nl;
    int total_humans = num_ranks * nh;

    /* Deterministic tier assignment, identical on every rank. */
    srand(seed);
    loc_tier_t *loc_tiers = malloc(sizeof(loc_tier_t) * total_cells);
    for (int i = 0; i < total_cells; i++) {
        int r = rand() % 100;
        if      (r < 20) loc_tiers[i] = LOC_POPULAR;
        else if (r < 60) loc_tiers[i] = LOC_NORMAL;
        else             loc_tiers[i] = LOC_QUIET;
    }

    /* Pick global patient zero deterministically on every rank. */
    int patient_zero_global = rand() % total_humans;
    int patient_zero_rank   = patient_zero_global / nh;
    int patient_zero_local  = patient_zero_global % nh;

    if (rank == 0) {
        printf("Zombie sim: %d ranks (%dx%d), %d locals/rank (%d total), "
               "%d humans/rank (%d total), cell cap=%d, "
               "incubation=%d days, work=%d\n"
               "Risk model: %d%% high-risk, near-cell radius=%d "
               "(same=%d%%, near=%d%%)\n"
               "Patient zero: rank=%d local-human=%d\n",
               num_ranks, rows, cols, nl, total_cells,
               nh, total_humans, CELL_CAPACITY,
               INFECT_DAYS_TO_ZOMBIE, work,
               HIGH_RISK_FRACTION_PCT, HIGH_RISK_RADIUS,
               INFECT_PROB_SAME_PCT, INFECT_PROB_NEAR_PCT,
               patient_zero_rank, patient_zero_local);
    }

    status = dstm_start(topology);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to start DSTM");
        free(loc_tiers);
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
        free(loc_tiers);
        MPI_Finalize();
        return -1;
    }

    /* Per-rank RNG seed (after the deterministic phase). */
    srand(seed + rank);

    zombie_state_t state;
    zombie_init(&state, rank, num_ranks, nl, nh, work, seed + rank,
                loc_tiers, topology, cells);

    /* Patient zero — owner-side state mutation, plus seed the bundle so
     * step 0 already sees them as contagious in their home cell. */
    if (rank == patient_zero_rank) {
        human_t *pz = &state.humans[patient_zero_local];
        pz->state           = HSTATE_INFECTED;
        pz->days_infected   = 0;
        pz->last_infect_day = 0;
        dstm_log_normal("Rank %d: patient zero is local human %d "
                        "(home %d.%d)\n",
                        rank, patient_zero_local,
                        pz->home_node, pz->home_local);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    double start = 0;
    double end = 0;

    if (LOGGING)
        start = MPI_Wtime();

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
    int local_hr_total = 0, local_hr_infected_or_worse = 0;
    for (int i = 0; i < nh; i++) {
        human_t *h = &state.humans[i];
        switch (h->state) {
            case HSTATE_HEALTHY:  local_healthy++;  break;
            case HSTATE_INFECTED: local_infected++; break;
            case HSTATE_ZOMBIE:   local_zombie++;   break;
        }
        if (h->risk == HRISK_HIGH) {
            local_hr_total++;
            if (h->state != HSTATE_HEALTHY) local_hr_infected_or_worse++;
        }
    }

    if (rank == 0) {
        all_dep = malloc(sizeof(int) * total_cells);
        all_arr = malloc(sizeof(int) * total_cells);
        all_inf = malloc(sizeof(int) * total_cells);
    }
    MPI_Gather(state.departures, nl, MPI_INT, all_dep, nl, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(state.arrivals,   nl, MPI_INT, all_arr, nl, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(state.infections, nl, MPI_INT, all_inf, nl, MPI_INT, 0, MPI_COMM_WORLD);

    int tot_healthy = 0, tot_infected = 0, tot_zombie = 0;
    MPI_Reduce(&local_healthy,  &tot_healthy,  1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_infected, &tot_infected, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_zombie,   &tot_zombie,   1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    int tot_hr_total = 0, tot_hr_infected = 0;
    MPI_Reduce(&local_hr_total,             &tot_hr_total,    1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_hr_infected_or_worse, &tot_hr_infected, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int total_dep = 0, total_arr = 0, total_inf = 0;
        for (int i = 0; i < total_cells; i++) {
            total_dep += all_dep[i];
            total_arr += all_arr[i];
            total_inf += all_inf[i];
        }

        printf("\n=============== INFECTION SUMMARY ===============\n"
               "%-10s %10s\n"
               "Healthy    %10d\n"
               "Infected   %10d\n"
               "Zombie     %10d\n"
               "-------------------------\n"
               "Total      %10d\n"
               "\n"
               "High-risk population:   %d / %d  (%.1f%%)\n"
               "High-risk infected:     %d / %d  (%.1f%% of HR)\n"
               "=================================================\n",
               "Status", "Count",
               tot_healthy, tot_infected, tot_zombie,
               tot_healthy + tot_infected + tot_zombie,
               tot_hr_total, total_humans,
               100.0 * tot_hr_total / total_humans,
               tot_hr_infected, tot_hr_total,
               tot_hr_total > 0 ? 100.0 * tot_hr_infected / tot_hr_total : 0.0);

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

    zombie_destroy(&state);
    dstm_darray_destroy(&cells);

    int fini_status = dstm_fini(last_time);
    if (fini_status != DSTM_SUCCESS) dstm_log_error("Failed to finalize DSTM");

    dstm_distance_destroy(&topology);
    free(loc_tiers);
    dstm_finalize();
    MPI_Finalize();
    return 0;
}
