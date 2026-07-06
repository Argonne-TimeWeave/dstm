/*******************************************************************************
 * phold.c — PHOLD (Parallel HOLD) PDES Benchmark for DSTM
 *
 * Outbox-based approach: each rank writes its own outbox slots with
 * delta=0 (local PUT), and readers fetch from remote ranks' outbox
 * slots with delta=-distance (remote GET into the past).
 *
 * This eliminates sentinel pre-fill entirely: every slot has a version
 * for every tick because the owning rank always writes it.
 *
 * Outbox layout: num_ranks × num_ranks slots
 *   Index: source_rank * num_ranks + dest_rank
 *   Ownership: rank owns slots [rank*num_ranks .. (rank+1)*num_ranks)
 *   Writer: rank writes its own row (delta=0, local)
 *   Reader: dest_rank reads source_rank's column (delta=-dist, remote)
 *
 * Copyright 2025 UChicago Argonne, LLC.
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "dstm.h"

/*******************************************************************************
 * Tuning constants
 ******************************************************************************/

#define INBOX_CAPACITY 64
#define LP_QUEUE_CAPACITY 512

/*******************************************************************************
 * Data types
 ******************************************************************************/

typedef struct {
    int         valid;
    int         source_lp;
    int         dest_lp;
    dstm_time_t timestamp;
    int         generation;
} phold_event_t;

typedef struct {
    int            n_events;
    phold_event_t  events[INBOX_CAPACITY];
} inbox_slot_t;

typedef struct {
    int            size;
    phold_event_t  heap[LP_QUEUE_CAPACITY];
} event_queue_t;

typedef struct {
    int              rank;
    int              num_ranks;
    int              lps_per_rank;
    int              total_lps;
    double           remote_prob;
    double           mean_delay;
    double           lookahead;
    int              timesteps;
    int              quiet;
    unsigned int     rng_state;

    dstm_distance_t *topology;
    dstm_darray_t   *outboxes;       /**< sender-owned outbox array */

    event_queue_t   *queues;

    long             events_processed;
} phold_state_t;

static const inbox_slot_t empty_inbox = { .n_events = 0 };

/*******************************************************************************
 * PRNG helpers
 ******************************************************************************/

static unsigned int xorshift32(unsigned int *state)
{
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static double rand_uniform(unsigned int *state)
{
    return (double)xorshift32(state) / 4294967296.0;
}

static double rand_exponential(unsigned int *state, double mean)
{
    double u;
    do {
        u = rand_uniform(state);
    } while (u == 0.0);
    return -mean * log(u);
}

/*******************************************************************************
 * Priority queue (min-heap by timestamp)
 ******************************************************************************/

static void pq_init(event_queue_t *q)
{
    q->size = 0;
}

static void pq_swap(phold_event_t *a, phold_event_t *b)
{
    phold_event_t tmp = *a;
    *a = *b;
    *b = tmp;
}

static void pq_push(event_queue_t *q, phold_event_t e)
{
    if (q->size >= LP_QUEUE_CAPACITY) {
        fprintf(stderr, "ERROR: LP queue overflow (capacity=%d)\n",
                LP_QUEUE_CAPACITY);
        return;
    }
    q->heap[q->size] = e;
    int i = q->size++;
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (q->heap[parent].timestamp <= q->heap[i].timestamp)
            break;
        pq_swap(&q->heap[parent], &q->heap[i]);
        i = parent;
    }
}

static phold_event_t pq_pop(event_queue_t *q)
{
    phold_event_t top = q->heap[0];
    q->heap[0] = q->heap[--q->size];
    int i = 0;
    for (;;) {
        int left  = 2 * i + 1;
        int right = 2 * i + 2;
        int smallest = i;
        if (left < q->size &&
            q->heap[left].timestamp < q->heap[smallest].timestamp)
            smallest = left;
        if (right < q->size &&
            q->heap[right].timestamp < q->heap[smallest].timestamp)
            smallest = right;
        if (smallest == i)
            break;
        pq_swap(&q->heap[i], &q->heap[smallest]);
        i = smallest;
    }
    return top;
}

static int pq_empty(event_queue_t *q)
{
    return q->size == 0;
}

static dstm_time_t pq_peek_time(event_queue_t *q)
{
    return q->heap[0].timestamp;
}

/*******************************************************************************
 * LP / rank mapping helpers
 ******************************************************************************/

static int lp_to_rank(int global_lp, int lps_per_rank)
{
    return global_lp / lps_per_rank;
}

static int lp_to_local(int global_lp, int lps_per_rank)
{
    return global_lp % lps_per_rank;
}

static int lp_from_local(int rank, int local_lp, int lps_per_rank)
{
    return rank * lps_per_rank + local_lp;
}

/*******************************************************************************
 * Outbox index helper
 *
 * Layout: num_ranks × num_ranks
 * Index:  source_rank * num_ranks + dest_rank
 * Ownership: rank owns its entire row (source_rank == rank)
 ******************************************************************************/

static size_t outbox_index(int source_rank, int dest_rank, int num_ranks)
{
    return (size_t)source_rank * num_ranks + dest_rank;
}

/*******************************************************************************
 * PHOLD event generation (matches ROSS RNG draw order)
 ******************************************************************************/

static phold_event_t phold_generate_event(phold_state_t *s,
                                          int source_lp,
                                          dstm_time_t now,
                                          int generation)
{
    int dest_lp;

    if (rand_uniform(&s->rng_state) <= s->remote_prob) {
        dest_lp = (int)(rand_uniform(&s->rng_state) * s->total_lps);
        if (dest_lp >= s->total_lps) dest_lp = s->total_lps - 1;
    } else {
        dest_lp = source_lp;
    }

    double delay_f;
    if (s->mean_delay > 0.0)
        delay_f = rand_exponential(&s->rng_state, s->mean_delay) + s->lookahead;
    else
        delay_f = s->lookahead;

    dstm_time_t delay = (dstm_time_t)ceil(delay_f);
    if (delay < 1) delay = 1;

    phold_event_t ev = {
        .valid      = 1,
        .source_lp  = source_lp,
        .dest_lp    = dest_lp,
        .timestamp  = now + delay,
        .generation = generation,
    };
    return ev;
}

/*******************************************************************************
 * Usage
 ******************************************************************************/

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s [OPTIONS]\n"
            "  --lps-per-rank N   LPs per MPI rank          (default: 8)\n"
            "  --init-events  N   initial events per LP      (default: 1)\n"
            "  --timesteps    N   simulation timesteps        (default: 1000)\n"
            "  --remote       F   remote event probability    (default: 0.25)\n"
            "  --mean         F   mean exponential delay      (default: 1.0)\n"
            "  --lookahead    L   minimum timestamp increment (default: 1.0)\n"
            "  --seed         S   base PRNG seed              (default: 42)\n"
            "  --quiet            suppress per-event logging\n"
            "  --help, -h         this message\n",
            prog);
}

/*******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char *argv[])
{
    int    lps_per_rank = 8;
    int    init_events  = 1;
    int    timesteps    = 1000;
    double remote_prob  = 0.25;
    double mean_delay   = 1.0;
    double lookahead    = 1.0;
    int    seed         = 42;
    int    quiet        = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--lps-per-rank") && i + 1 < argc)
            lps_per_rank = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--init-events") && i + 1 < argc)
            init_events = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--timesteps") && i + 1 < argc)
            timesteps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--remote") && i + 1 < argc)
            remote_prob = atof(argv[++i]);
        else if (!strcmp(argv[i], "--mean") && i + 1 < argc)
            mean_delay = atof(argv[++i]);
        else if (!strcmp(argv[i], "--lookahead") && i + 1 < argc)
            lookahead = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            seed = atoi(argv[++i]);
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
    dstm_log_init(stdout, "phold");

    if (quiet)
        dstm_log_setlevel(DSTM_LOG_QUIET);
    else
        dstm_log_setlevel(DSTM_LOG_NORMAL);

    dstm_distance_t *topology;
    int status;
    status = dstm_distance_create_complete(&topology, num_ranks);

    if (status != DSTM_SUCCESS) {
        MPI_Finalize();
        return 1;
    }

    status = dstm_start(topology);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("dstm_start failed");
        MPI_Finalize();
        return 1;
    }

    phold_state_t state;
    memset(&state, 0, sizeof(state));
    state.rank         = rank;
    state.num_ranks    = num_ranks;
    state.lps_per_rank = lps_per_rank;
    state.total_lps    = num_ranks * lps_per_rank;
    state.remote_prob  = remote_prob;
    state.lookahead    = lookahead;
    double adjusted_mean = mean_delay - lookahead;
    if (adjusted_mean < 0.0) adjusted_mean = 0.0;
    state.mean_delay   = adjusted_mean;
    state.timesteps    = timesteps;
    state.quiet        = quiet;
    state.topology     = topology;
    state.rng_state    = (unsigned int)(seed + rank + 1);

    if (rank == 0) {
        printf("==== PHOLD Benchmark (DSTM) ====\n");
        printf("  ranks          = %d\n", num_ranks);
        printf("  lps_per_rank   = %d\n", lps_per_rank);
        printf("  total_lps      = %d\n", state.total_lps);
        printf("  init_events/LP = %d\n", init_events);
        printf("  timesteps      = %d\n", timesteps);
        printf("  remote_prob    = %.2f\n", remote_prob);
        printf("  mean_delay     = %.2f (adjusted: %.2f)\n", mean_delay, adjusted_mean);
        printf("  lookahead      = %.2f\n", lookahead);
        printf("  seed           = %d\n", seed);
        printf("  outbox layout  = %d slots (%d x %d ranks)\n",
               num_ranks * num_ranks, num_ranks, num_ranks);
        printf("  sentinel PUTs  = 0 (outbox pattern)\n");
        printf("================================\n");
    }

    /* ----- distributed outbox array -----
     *
     * Layout: num_ranks × num_ranks slots
     * Index:  source_rank * num_ranks + dest_rank
     *
     * Each rank OWNS its entire row (slots source_rank*num_ranks
     * through (source_rank+1)*num_ranks - 1).
     *
     * Writer: rank writes its own row with PUT(time=t, delta=0).
     *         This is a local write — no MPI, just versioned memcpy.
     *
     * Reader: dest_rank reads source_rank's slot with
     *         GET(time=t, delta=-distance(dest_rank, source_rank)).
     *         This reads the version the source wrote `distance` ticks ago.
     *
     * Since the owner always writes every tick (even if empty),
     * NO sentinel pre-fill is needed.
     */
    status = dstm_darray_create(&state.outboxes,
                                sizeof(inbox_slot_t),
                                (size_t)num_ranks);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to create outbox darray");
        MPI_Finalize();
        return 1;
    }

    /* ----- local priority queues ----- */
    state.queues = (event_queue_t *)calloc(lps_per_rank, sizeof(event_queue_t));
    if (!state.queues) {
        dstm_log_error("Failed to allocate local queues");
        MPI_Finalize();
        return 1;
    }
    for (int l = 0; l < lps_per_rank; l++)
        pq_init(&state.queues[l]);

    /* ----- seed initial events (matches ROSS: self-events) ----- */
    for (int l = 0; l < lps_per_rank; l++) {
        int global_lp = lp_from_local(rank, l, lps_per_rank);
        for (int e = 0; e < init_events; e++) {
            double delay_f;
            if (adjusted_mean > 0.0)
                delay_f = rand_exponential(&state.rng_state, adjusted_mean) + lookahead;
            else
                delay_f = lookahead;
            dstm_time_t ts = (dstm_time_t)ceil(delay_f);
            if (ts < 1) ts = 1;

            phold_event_t ev = {
                .valid      = 1,
                .source_lp  = global_lp,
                .dest_lp    = global_lp,
                .timestamp  = ts,
                .generation = 0,
            };
            pq_push(&state.queues[l], ev);
        }
    }

    /* NO sentinel pre-fill needed — the outbox owner writes every tick */

     int max_distance = 0;
    for (int r = 0; r < num_ranks; r++) {
        if (r == rank) continue;
        int d = dstm_distance_get(topology, rank, r);
        if (d > max_distance) max_distance = d;
    }

    for (dstm_time_t bt = 0; bt < max_distance; bt++) {
        for (int dr = 0; dr < num_ranks; dr++) {
            if (dr == rank) continue;
            size_t idx = outbox_index(rank, dr, num_ranks);
            dstm_darray_put(state.outboxes, idx,
                            (void *)&empty_inbox, bt, 0);
        }
    }

    /* ----- per-destination-rank outbox scratch (reused each tick) ----- */
    inbox_slot_t *tick_outboxes = (inbox_slot_t *)calloc(num_ranks,
                                                         sizeof(inbox_slot_t));
    if (!tick_outboxes) {
        dstm_log_error("Failed to allocate tick_outboxes");
        MPI_Finalize();
        return 1;
    }

    /* ----- query darray layout for local slot access ----- */
    size_t local_offset = dstm_darray_global_offset(state.outboxes);

    /* ================================================================
     * MAIN SIMULATION LOOP
     *
     * Each logical timestep t:
     *
     *   PHASE 1 — WRITE (PUT, delta=0):
     *     Clear all outbox slots for this tick, process events,
     *     generate successors, batch remote events into outbox slots,
     *     then PUT all slots (including empty ones) into the darray.
     *     These are local writes (delta=0, we own the slots).
     *
     *   PHASE 2 — READ (GET, delta=-distance):
     *     Read each remote rank's outbox slot addressed to us.
     *     The GET uses a negative delta to look into the past,
     *     reading what the remote rank wrote `distance` ticks ago.
     *     Scatter received events into local LP queues.
     *
     *   PUT-before-GET ordering: all local PUTs at (t, delta=0) happen
     *   before any local GETs at (t, delta=0). Since our GETs use
     *   delta=-distance (non-zero), the phase rule doesn't apply to
     *   them, but we still PUT first for clean separation.
     * ================================================================ */

    double wall_start = MPI_Wtime();

        for (dstm_time_t t = 1; t <= timesteps; t++) {

        /* --- clear tick outboxes --- */
        for (int r = 0; r < num_ranks; r++)
            tick_outboxes[r] = empty_inbox;

        /* --- PHASE 1: GET from each remote rank's outbox ---
         *
         * Read what remote ranks wrote at tick (t - distance).
         * Merge received events into local LP queues BEFORE processing.
         */
        for (int sr = 0; sr < num_ranks; sr++) {
            if (sr == rank) continue;

            int dist = dstm_distance_get(topology, rank, sr);
            if (dist <= 0 || dist == -1) continue;

            size_t idx = outbox_index(sr, rank, num_ranks);
            inbox_slot_t recv;

            status = dstm_darray_get_or_default(state.outboxes, idx,
                                                &recv, t, -dist,
                                                &empty_inbox);
            if (status != DSTM_SUCCESS) {
                dstm_log_error("GET outbox[%d->%d] at t=%ld delta=%d "
                               "failed (%d)", sr, rank, t, -dist, status);
                continue;
            }

            if (recv.n_events == 0) continue;

            dstm_log_normal("Rank %d [t=%ld]: received %d event(s) "
                            "from rank %d (written at t=%ld)\n",
                            rank, t, recv.n_events, sr, t - dist);

            for (int e = 0; e < recv.n_events; e++) {
                phold_event_t ev = recv.events[e];
                if (!ev.valid) continue;

                int local_idx = lp_to_local(ev.dest_lp, lps_per_rank);
                pq_push(&state.queues[local_idx], ev);

                dstm_log_normal("Rank %d [t=%ld]: LP %d received event "
                                "from LP %d (ts=%ld gen=%d)\n",
                                rank, t, ev.dest_lp, ev.source_lp,
                                ev.timestamp, ev.generation);
            }
        }

        /* --- PHASE 2: process events and generate successors --- */
        for (int l = 0; l < lps_per_rank; l++) {
            int glp = lp_from_local(rank, l, lps_per_rank);
            event_queue_t *q = &state.queues[l];

            while (!pq_empty(q) && pq_peek_time(q) <= t) {
                phold_event_t ev = pq_pop(q);
                state.events_processed++;

                dstm_log_normal("Rank %d [t=%ld]: LP %d PROCESS event "
                                "(src=%d ts=%ld gen=%d)\n",
                                rank, t, glp, ev.source_lp,
                                ev.timestamp, ev.generation);

                phold_event_t new_ev = phold_generate_event(
                    &state, glp, t, ev.generation + 1);

                int dest_rank = lp_to_rank(new_ev.dest_lp, lps_per_rank);

                if (dest_rank == rank) {
                    pq_push(&state.queues[lp_to_local(new_ev.dest_lp,
                                                      lps_per_rank)],
                            new_ev);

                    dstm_log_normal("Rank %d [t=%ld]: LP %d -> LP %d "
                                    "LOCAL (ts=%ld)\n",
                                    rank, t, glp, new_ev.dest_lp,
                                    new_ev.timestamp);
                } else {
                    inbox_slot_t *ob = &tick_outboxes[dest_rank];
                    if (ob->n_events < INBOX_CAPACITY) {
                        ob->events[ob->n_events++] = new_ev;

                        dstm_log_normal("Rank %d [t=%ld]: LP %d -> LP %d "
                                        "REMOTE rank %d (ts=%ld)\n",
                                        rank, t, glp, new_ev.dest_lp,
                                        dest_rank, new_ev.timestamp);
                    } else {
                        dstm_log_warning("Rank %d [t=%ld]: outbox to rank %d "
                                         "full (%d/%d) — event DROPPED\n",
                                         rank, t, dest_rank,
                                         ob->n_events, INBOX_CAPACITY);
                    }
                }
            }
        }

        /* --- PHASE 3: PUT all outbox slots (local, delta=0) --- */
        for (int dr = 0; dr < num_ranks; dr++) {
            if (dr == rank) continue;
            size_t idx = outbox_index(rank, dr, num_ranks);
            status = dstm_darray_put(state.outboxes, idx,
                                     &tick_outboxes[dr], t, 0);
            if (status != DSTM_SUCCESS) {
                dstm_log_error("PUT outbox[%d->%d] at t=%ld failed (%d)",
                               rank, dr, t, status);
            }
        }
    }

    double wall_end = MPI_Wtime();
    double wall_elapsed = wall_end - wall_start;

    /* ----- drain and finalize ----- */
    dstm_time_t last_time = timesteps + 2;

    status = dstm_barrier(last_time);
    if (status != DSTM_SUCCESS)
        dstm_log_error("dstm_barrier failed");

    dstm_darray_destroy(&state.outboxes);

    status = dstm_fini(last_time);
    if (status != DSTM_SUCCESS)
        dstm_log_error("dstm_fini failed");

    dstm_distance_destroy(&topology);

    long local_metrics[1] = {
        state.events_processed,
    };
    long global_metrics[1] = {0};
    MPI_Reduce(local_metrics, global_metrics, 1, MPI_LONG, MPI_SUM,
               0, MPI_COMM_WORLD);

    double max_wall = 0.0;
    MPI_Reduce(&wall_elapsed, &max_wall, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        long total_events = global_metrics[0];
        printf("\n==== PHOLD GLOBAL METRICS ====\n");
        printf("  events_processed = %ld\n", total_events);
        printf("  wall_time        = %.4f s\n", max_wall);
        printf("  throughput       = %.0f events/s\n",
               max_wall > 0 ? total_events / max_wall : 0.0);
        printf("  efficiency       = %.0f events/s/rank\n",
               max_wall > 0 ? total_events / max_wall / num_ranks : 0.0);
        printf("==============================\n");
    }

    free(state.queues);
    free(tick_outboxes);

    dstm_finalize();
    MPI_Finalize();
    return 0;
}
