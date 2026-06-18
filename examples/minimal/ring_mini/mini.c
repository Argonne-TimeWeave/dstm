#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "dstm.h"

/**
 * Minimal Network Simulation with DSTM
 *
 * Simulates a multi-node network where packets are routed hop-by-hop
 * around a ring topology using store-and-forward.
 *
 * Each rank owns a mailbox with two slots:
 *   slot 0 (SLOT_LEFT)  — written exclusively by the left neighbor
 *   slot 1 (SLOT_RIGHT) — written exclusively by the right neighbor
 *
 * Each slot holds up to SLOT_CAPACITY packets.  The first `n_packets`
 * entries are valid; the rest are padding.  Because only one neighbor
 * ever writes each slot there are no write conflicts.
 *
 * Each timestep:
 *   1. Write empty slot into own mailbox (sentinel so GET never blocks)
 *   2. GET both slots from own mailbox
 *   3. For each received packet:
 *        - if dest == rank: deliver (print)
 *        - else:            forward one hop toward dest (PUT, delta=1)
 *   4. Randomly inject new packets into the left or right neighbor's slot
 */

#define PACKET_SIZE    256
#define SLOT_CAPACITY  8    /* max packets per slot per timestep */

#define MAILBOX_SLOTS  2
#define SLOT_LEFT      0
#define SLOT_RIGHT     1

typedef struct {
    int         valid;
    int         source_rank;
    int         dest_rank;
    int         hops_taken;       /* incremented at each forwarding hop */
    dstm_time_t origin_time;
    char        payload[PACKET_SIZE - 24];
} mailbox_entry_t;

/*
 * One slot in the distributed array.
 * Written by exactly one neighbor (left XOR right), so no conflicts.
 */
typedef struct {
    int            n_packets;              /* how many entries are valid   */
    mailbox_entry_t packets[SLOT_CAPACITY];
} mailbox_slot_t;

static const mailbox_slot_t empty_slot = { .n_packets = 0 };

static void usage(const char *prog)
{
    fprintf(stderr, "usage: %s [--seed S] [--time T] [--prob %] [--quiet]\n", prog);
}

int main(int argc, char *argv[])
{
    int seed  = 42;
    int quiet = 0;
    int timesteps = 16;
    int send_prob = 60; // check each rank sends each step

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--time"))
            timesteps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prob"))
            send_prob = atoi(argv[++i]);
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
    dstm_log_init(stdout, "network_sim");
    
    if (quiet) 
        dstm_log_setlevel(DSTM_LOG_QUIET);
    else 
        dstm_log_setlevel(DSTM_LOG_NORMAL);

    dstm_distance_t *topology;
    int status = dstm_distance_create_ring(&topology, num_ranks);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to create topology");
        MPI_Finalize();
        return -1;
    }

    if (rank == 0)
        dstm_log_normal("Ring topology: %d nodes, store-and-forward, "
               "%d packets/slot\n", num_ranks, SLOT_CAPACITY);

    status = dstm_start(topology);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to start DSTM");
        return -1;
    }

    /*
     * Distributed array layout — one mailbox_slot_t per (rank, direction):
     *
     *   index = rank * MAILBOX_SLOTS + slot
     *
     *   [r0·LEFT][r0·RIGHT][r1·LEFT][r1·RIGHT] ...
     *       ↑         ↑
     *       │         └─ written only by right_rank (rank+1)
     *       └─────────── written only by left_rank  (rank-1)
     */
    dstm_darray_t *mailbox;
    status = dstm_darray_create_equal(&mailbox,
                                      sizeof(mailbox_slot_t),
                                      (size_t)num_ranks * MAILBOX_SLOTS);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to create mailbox");
        return -1;
    }

    srand(seed + rank);

    int left_rank  = (rank - 1 + num_ranks) % num_ranks;
    int right_rank = (rank + 1) % num_ranks;

    /* pending outbox: packets this rank is forwarding or injecting this step.
     * Indexed by direction: [SLOT_LEFT] = going left, [SLOT_RIGHT] = going right.
     * We accumulate into these before a single PUT per direction. */
    mailbox_slot_t outbox[MAILBOX_SLOTS];

    mailbox_slot_t recv_slot;

    for (dstm_time_t t = 0; t < timesteps; t++) {
        dstm_log_normal("Rank %d: time step %ld\n", rank, t);

        /* ----------------------------------------------------------------
         * STEP 1 — sentinel PUT
         * Write empty_slot into our own slots so that any GET at this
         * timestep always finds a version and never blocks.
         * A real packet PUT by a neighbor (delta=1 from t-1) will have
         * been written before we GET, so it wins over this sentinel.
         * ---------------------------------------------------------------- */
        for (int s = 0; s < MAILBOX_SLOTS; s++) {
            size_t idx = (size_t)rank * MAILBOX_SLOTS + s;
            dstm_darray_put(mailbox, idx, &empty_slot, t, 0);
        }

        /* ----------------------------------------------------------------
         * STEP 2 — RECEIVE: read both inbox slots
         * ---------------------------------------------------------------- */
        outbox[SLOT_LEFT]  = empty_slot;
        outbox[SLOT_RIGHT] = empty_slot;

        for (int s = 0; s < MAILBOX_SLOTS; s++) {
            size_t idx = (size_t)rank * MAILBOX_SLOTS + s;

            status = dstm_darray_get_or_default(mailbox, idx,
                                                &recv_slot, t, 0,
                                                &empty_slot);
            if (status != DSTM_SUCCESS) {
                dstm_log_error("GET failed on slot %d", s);
                continue;
            }

            if (!quiet)
                dstm_log_normal("Rank %d [t=%ld]: %s slot has %d packet(s)\n",
                          rank, t,
                          s == SLOT_LEFT ? "left" : "right",
                          recv_slot.n_packets);

            /* ----------------------------------------------------------------
             * STEP 3 — process each packet in the slot
             * ---------------------------------------------------------------- */
            for (int p = 0; p < recv_slot.n_packets; p++) {
                mailbox_entry_t pkt = recv_slot.packets[p];

                if (pkt.dest_rank == rank) {
                    /* DELIVER */
                    dstm_log_normal("Rank %d [t=%ld]: DELIVERED — "
                           "src=%d origin_t=%ld hops=%d payload=\"%s\"\n",
                           rank, t,
                           pkt.source_rank, pkt.origin_time,
                           pkt.hops_taken, pkt.payload);
                } else {
                    /* FORWARD one hop toward dest.
                     *
                     * Direction: clockwise distance to dest tells us whether
                     * to pass right or left.  We only move one hop (delta=1)
                     * which is pure store-and-forward — no cut-through.
                     *
                     * The neighbor's slot we write into is the one that faces
                     * back toward us:
                     *   - we go right → write into neighbor's SLOT_LEFT
                     *   - we go left  → write into neighbor's SLOT_RIGHT
                     */
                    int dist_right = (pkt.dest_rank - rank + num_ranks) % num_ranks;
                    int go_right   = (dist_right <= num_ranks - dist_right);

                    int fwd_dir        = go_right ? SLOT_RIGHT : SLOT_LEFT;
                    int neighbor_slot  = go_right ? SLOT_LEFT  : SLOT_RIGHT;
                    int neighbor_rank  = go_right ? right_rank : left_rank;
                    (void)neighbor_rank; /* rank implicit from fwd_dir outbox */

                    pkt.hops_taken++;

                    mailbox_slot_t *ob = &outbox[fwd_dir];
                    if (ob->n_packets < SLOT_CAPACITY) {
                        ob->packets[ob->n_packets++] = pkt;
                        dstm_log_normal("Rank %d [t=%ld]: FORWARD src=%d dst=%d "
                                      "via %s (hop %d)\n",
                                      rank, t, pkt.source_rank, pkt.dest_rank,
                                      go_right ? "right" : "left",
                                      pkt.hops_taken);
                    } else {
                        dstm_log_warning("Rank %d [t=%ld]: DROP src=%d dst=%d "
                                         "— outbox full\n",
                                         rank, t,
                                         pkt.source_rank, pkt.dest_rank);
                    }

                    (void)neighbor_slot; /* used implicitly: outbox[fwd_dir]
                                           maps to neighbor's opposite slot  */
                }
            }
        }

        /* ----------------------------------------------------------------
         * STEP 4 — INJECT: randomly create a new packet this timestep
         * ---------------------------------------------------------------- */
        if ((rand() % 100) < send_prob) {
            int dest = rand() % (num_ranks - 1);
            if (dest >= rank) dest++;

            int dist_right = (dest - rank + num_ranks) % num_ranks;
            int go_right   = (dist_right <= num_ranks - dist_right);
            int fwd_dir    = go_right ? SLOT_RIGHT : SLOT_LEFT;

            mailbox_slot_t *ob = &outbox[fwd_dir];
            if (ob->n_packets < SLOT_CAPACITY) {
                mailbox_entry_t pkt = {
                    .valid       = 1,
                    .source_rank = rank,
                    .dest_rank   = dest,
                    .hops_taken  = 0,
                    .origin_time = t,
                };
                snprintf(pkt.payload, sizeof(pkt.payload),
                         "src=%d dst=%d t=%ld", rank, dest, t);
                ob->packets[ob->n_packets++] = pkt;

                dstm_log_normal("Rank %d [t=%ld]: INJECT dst=%d via %s\n",
                              rank, t, dest, go_right ? "right" : "left");
            }
        }

        /* ----------------------------------------------------------------
         * STEP 5 — SEND outbox slots to neighbors (delta=1, one hop)
         *
         * outbox[SLOT_RIGHT] → right_rank's SLOT_LEFT  (they recv from left)
         * outbox[SLOT_LEFT]  → left_rank's  SLOT_RIGHT (they recv from right)
         * ---------------------------------------------------------------- */
        if (outbox[SLOT_RIGHT].n_packets > 0) {
            size_t idx = (size_t)right_rank * MAILBOX_SLOTS + SLOT_LEFT;
            status = dstm_darray_put(mailbox, idx,
                                     &outbox[SLOT_RIGHT], t, 1);
            if (status != DSTM_SUCCESS)
                dstm_log_error("PUT right→left failed");
        }

        if (outbox[SLOT_LEFT].n_packets > 0) {
            size_t idx = (size_t)left_rank * MAILBOX_SLOTS + SLOT_RIGHT;
            status = dstm_darray_put(mailbox, idx,
                                     &outbox[SLOT_LEFT], t, 1);
            if (status != DSTM_SUCCESS)
                dstm_log_error("PUT left→right failed");
        }
    }

    dstm_time_t last_time = timesteps + 1;
    status = dstm_barrier(last_time);
    if (status != DSTM_SUCCESS)
        dstm_log_error("Barrier failed");

    dstm_darray_destroy(&mailbox);

    status = dstm_fini(last_time);
    if (status != DSTM_SUCCESS)
        dstm_log_error("Failed to finalize DSTM");

    dstm_distance_destroy(&topology);
    dstm_finalize();
    MPI_Finalize();
    return status;
}
