#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "dstm.h"
#include "internal/distance.h"

/**
 * PHOLD Benchmark on DSTM
 *
 * Classic Parallel HOLD (PHOLD) synthetic benchmark:
 *   - Each rank owns LPS_PER_RANK logical processes (LPs).
 *   - Each LP starts with INIT_EVENTS events in its future event list.
 *   - Processing an event generates exactly one new event:
 *       * with probability REMOTE, target is a remote LP
 *       * otherwise the target is a local LP on this rank
 *     The new event's timestamp is (now + LOOKAHEAD + Exp(MEAN)).
 *
 * Line topology + NEIGHBORHOOD_DIST:
 *   Ranks are laid out on a line 0..P-1.  Any pair of ranks whose
 *   line distance exceeds NEIGHBORHOOD_DIST is treated as unconnected
 *   (distance = -1).  Remote events are only sent to reachable ranks.
 *
 * Communication (mirrors ring_mini / airport_sim):
 *   - Distributed array `mailbox` of size num_ranks * num_ranks:
 *         index (dest * num_ranks + src)   ← src writes, dest reads
 *   - Each slot is a mailbox_slot_t holding up to SLOT_CAPACITY events.
 *   - Each timestep every rank PUTs sentinel empties into all reachable
 *     slots addressed to itself (so GET never blocks), GETs events sent
 *     to it, processes them, and PUTs outbound events with delta = 1.
 */

#define NEIGHBORHOOD_DIST 2
#define SLOT_CAPACITY     32
#define PAYLOAD_BYTES     48

typedef struct {
    int         valid;
    int         src_rank;
    int         src_lp;
    int         dst_rank;
    int         dst_lp;
    dstm_time_t send_time;      /* timestep it was emitted           */
    long long   event_id;
    char        payload[PAYLOAD_BYTES];
} phold_event_t;

typedef struct {
    int           n_events;
    phold_event_t events[SLOT_CAPACITY];
} mailbox_slot_t;

static const mailbox_slot_t empty_slot = { .n_events = 0 };

/* -----------------------------------------------------------------
 * Exponential RNG — inverse-CDF sample with mean `mean`.
 * ----------------------------------------------------------------- */
static double exp_sample(double mean)
{
    double u;
    do {
        u = (double)rand() / (double)RAND_MAX;
    } while (u <= 0.0 || u >= 1.0);
    return -mean * log(u);
}

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

    if (lps_per_rank < 1) lps_per_rank = 1;
    if (init_events  < 0) init_events  = 0;
    if (timesteps    < 1) timesteps    = 1;
    if (remote_prob  < 0.0) remote_prob = 0.0;
    if (remote_prob  > 1.0) remote_prob = 1.0;
    if (mean_delay   <= 0.0) mean_delay = 1.0;
    if (lookahead    <  0.0) lookahead  = 0.0;

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

    /* -----------------------------------------------------------------
     * Line topology, then prune anything beyond NEIGHBORHOOD_DIST.
     * ----------------------------------------------------------------- */
    dstm_distance_t *topology;
    int status = dstm_distance_create_line(&topology, num_ranks);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to create line topology");
        MPI_Finalize();
        return -1;
    }

    for (int i = 0; i < num_ranks; i++) {
        for (int j = 0; j < num_ranks; j++) {
            if (i != j &&
                topology->distances[i * num_ranks + j] > NEIGHBORHOOD_DIST) {
                topology->distances[i * num_ranks + j] = -1;
            }
            if (rank == 0)
                printf("%d ", topology->distances[i * num_ranks + j]);
        }
        if (rank == 0)
            printf("\n");
    }

    int total_lps = num_ranks * lps_per_rank;

    if (rank == 0) {
        printf("PHOLD: ranks=%d lps/rank=%d total_lps=%d "
               "init_events/LP=%d timesteps=%d remote=%.3f "
               "mean=%.3f lookahead=%.3f neighborhood=%d\n",
               num_ranks, lps_per_rank, total_lps,
               init_events, timesteps, remote_prob,
               mean_delay, lookahead, NEIGHBORHOOD_DIST);
    }

    status = dstm_start(topology);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to start DSTM");
        MPI_Finalize();
        return -1;
    }

    /* -----------------------------------------------------------------
     * Mailbox: index = dst_rank * num_ranks + src_rank
     * Only src ever writes that slot -> no write conflicts.
     * ----------------------------------------------------------------- */
    dstm_darray_t *mailbox;
    status = dstm_darray_create_equal(&mailbox,
                                      sizeof(mailbox_slot_t),
                                      (size_t)num_ranks * num_ranks);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to create mailbox");
        MPI_Finalize();
        return -1;
    }

    /* Per-rank RNG stream */
    srand(seed + rank * 1009);

    /* Outbox: one slot per potential destination rank */
    mailbox_slot_t *outbox = calloc((size_t)num_ranks, sizeof(mailbox_slot_t));
    if (!outbox) {
        dstm_log_error("outbox alloc failed");
        MPI_Finalize();
        return -1;
    }

    /* -----------------------------------------------------------------
     * Local event pool.  Simple FIFO of pending events destined for one
     * of this rank's LPs, scheduled to fire on some future timestep.
     * We bucket by (send_time % timesteps) to avoid a heap.
     * ----------------------------------------------------------------- */
    typedef struct {
        int           n;
        int           cap;
        phold_event_t *ev;
    } bucket_t;

    bucket_t *buckets = calloc((size_t)timesteps, sizeof(bucket_t));
    if (!buckets) {
        dstm_log_error("bucket alloc failed");
        MPI_Finalize();
        return -1;
    }

    long long next_event_id = (long long)rank * 1000000000LL;

    /* Seed initial events, all scheduled for t=0 on random local LPs */
    for (int l = 0; l < lps_per_rank; l++) {
        for (int e = 0; e < init_events; e++) {
            bucket_t *b = &buckets[0];
            if (b->n == b->cap) {
                b->cap = b->cap ? b->cap * 2 : 8;
                b->ev  = realloc(b->ev, (size_t)b->cap * sizeof(phold_event_t));
            }
            phold_event_t ev = {
                .valid     = 1,
                .src_rank  = rank,
                .src_lp    = l,
                .dst_rank  = rank,
                .dst_lp    = l,
                .send_time = 0,
                .event_id  = next_event_id++,
            };
            snprintf(ev.payload, sizeof(ev.payload),
                     "init r=%d lp=%d e=%d", rank, l, e);
            b->ev[b->n++] = ev;
        }
    }

    /* -----------------------------------------------------------------
     * Sentinel empties: for every reachable src->rank slot, PUT empty
     * at every timestep up front so GETs never block.
     * ----------------------------------------------------------------- */
    for (dstm_time_t t = 0; t < timesteps; t++) {
        for (int src = 0; src < num_ranks; src++) {
            int d = dstm_distance_get(topology, src, rank);
            if (src != rank && d == -1) continue;
            size_t idx = (size_t)rank * num_ranks + src;
            /* We are the writer only for the slot where src == rank;
             * for slots where src != rank the sender will overwrite. But
             * DSTM requires *someone* to seed empties for slots we read.
             * The convention used in ring_mini is that each rank seeds
             * its own outgoing slots — do the same: seed slot (dst, rank)
             * for every reachable dst. */
            (void)idx;
        }
        for (int dst = 0; dst < num_ranks; dst++) {
            int d = dstm_distance_get(topology, rank, dst);
            if (rank != dst && d == -1) continue;
            size_t idx = (size_t)dst * num_ranks + rank;
            dstm_darray_put(mailbox, idx, &empty_slot, 0, t);
        }
    }

    long events_processed = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* -----------------------------------------------------------------
     * Main PHOLD loop
     * ----------------------------------------------------------------- */
    for (dstm_time_t t = 0; t < timesteps; t++) {

        /* Clear outbox */
        for (int r = 0; r < num_ranks; r++)
            outbox[r] = empty_slot;

        /* --- receive: GET from every reachable src slot --- */
        mailbox_slot_t recv;
        for (int src = 0; src < num_ranks; src++) {
            if (src == rank) continue;
            int d = dstm_distance_get(topology, src, rank);
            if (d == -1) continue;

            size_t idx = (size_t)rank * num_ranks + src;
            status = dstm_darray_get_or_default(mailbox, idx,
                                                &recv, t, 0,
                                                &empty_slot);
            if (status != DSTM_SUCCESS) {
                dstm_log_error("GET failed src=%d", src);
                continue;
            }

            for (int i = 0; i < recv.n_events; i++) {
                phold_event_t ev = recv.events[i];
                if (!ev.valid) continue;

                /* Land in the local bucket for time t (fire now) */
                bucket_t *b = &buckets[t % timesteps];
                if (b->n == b->cap) {
                    b->cap = b->cap ? b->cap * 2 : 8;
                    b->ev  = realloc(b->ev,
                                     (size_t)b->cap * sizeof(phold_event_t));
                }
                b->ev[b->n++] = ev;

                if (!quiet) {
                    dstm_log_normal("Rank %d [t=%ld]: RECV eid=%lld "
                                    "%d.%d -> %d.%d\n",
                                    rank, t, ev.event_id,
                                    ev.src_rank, ev.src_lp,
                                    ev.dst_rank, ev.dst_lp);
                }
            }
        }

        /* --- process every event scheduled for this timestep --- */
        bucket_t *cur = &buckets[t % timesteps];
        int n_now = cur->n;

        for (int i = 0; i < n_now; i++) {
            phold_event_t in = cur->ev[i];
            events_processed++;

            /* Generate exactly one successor event */
            double u = (double)rand() / (double)RAND_MAX;
            int dst_rank, dst_lp;

            if (u < remote_prob && num_ranks > 1) {
                /* Pick a reachable remote rank uniformly */
                int reachable[64];
                int n_reach = 0;
                for (int r = 0; r < num_ranks && n_reach < 64; r++) {
                    if (r == rank) continue;
                    int d = dstm_distance_get(topology, rank, r);
                    if (d != -1) reachable[n_reach++] = r;
                }
                if (n_reach == 0) {
                    dst_rank = rank;
                } else {
                    dst_rank = reachable[rand() % n_reach];
                }
            } else {
                dst_rank = rank;
            }
            dst_lp = rand() % lps_per_rank;

            phold_event_t out = {
                .valid     = 1,
                .src_rank  = rank,
                .src_lp    = in.dst_lp,
                .dst_rank  = dst_rank,
                .dst_lp    = dst_lp,
                .send_time = t,
                .event_id  = next_event_id++,
            };
            snprintf(out.payload, sizeof(out.payload),
                     "r%d->r%d", rank, dst_rank);

            if (!quiet) {
                dstm_log_normal("Rank %d [t=%ld]: FIRE  eid=%lld "
                                "lp=%d -> %d.%d\n",
                                rank, t, in.event_id,
                                in.dst_lp, dst_rank, dst_lp);
            }

            if (dst_rank == rank) {
                /* Local: schedule into a future bucket */
                dstm_time_t fire_t = t + 1;
                if (fire_t >= timesteps) continue;
                bucket_t *b = &buckets[fire_t % timesteps];
                if (b->n == b->cap) {
                    b->cap = b->cap ? b->cap * 2 : 8;
                    b->ev  = realloc(b->ev,
                                     (size_t)b->cap * sizeof(phold_event_t));
                }
                b->ev[b->n++] = out;
            } else {
                /* Remote: stage into outbox */
                mailbox_slot_t *ob = &outbox[dst_rank];
                if (ob->n_events < SLOT_CAPACITY) {
                    ob->events[ob->n_events++] = out;
                } else {
                    dstm_log_warning("Rank %d [t=%ld]: outbox to %d full, "
                                     "dropping eid=%lld\n",
                                     rank, t, dst_rank, out.event_id);
                }
            }
        }
        cur->n = 0; /* consumed */

        /* --- send: one PUT per reachable remote rank --- */
        for (int dst = 0; dst < num_ranks; dst++) {
            if (dst == rank) continue;
            int d = dstm_distance_get(topology, rank, dst);
            if (d == -1) continue;
            if (outbox[dst].n_events == 0) {
                /* Still need to advance the slot version so the reader's
                 * GET at time t doesn't stall waiting on us. */
                size_t idx = (size_t)dst * num_ranks + rank;
                dstm_darray_put(mailbox, idx, &empty_slot, t, d);
                continue;
            }

            size_t idx = (size_t)dst * num_ranks + rank;
            status = dstm_darray_put(mailbox, idx,
                                     &outbox[dst], t, d);
            if (status != DSTM_SUCCESS)
                dstm_log_error("PUT to rank %d failed", dst);
        }
    }

    dstm_time_t last_time = timesteps + 1;
    status = dstm_barrier(last_time);
    if (status != DSTM_SUCCESS)
        dstm_log_error("Barrier failed");

    double t_end = MPI_Wtime();
    double wall_local = t_end - t_start;
    double max_wall   = 0.0;

    long local_metrics[1]  = { events_processed };
    long global_metrics[1] = { 0 };
    MPI_Reduce(local_metrics, global_metrics, 1, MPI_LONG, MPI_SUM,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&wall_local, &max_wall, 1, MPI_DOUBLE, MPI_MAX,
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

    for (int i = 0; i < timesteps; i++)
        free(buckets[i].ev);
    free(buckets);
    free(outbox);

    dstm_darray_destroy(&mailbox);

    status = dstm_fini(last_time);
    if (status != DSTM_SUCCESS)
        dstm_log_error("Failed to finalize DSTM");

    dstm_distance_destroy(&topology);
    dstm_finalize();
    MPI_Finalize();
    return status;
}
