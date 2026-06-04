#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "dstm.h"

/**
 * Minimal Network Simulation with DSTM
 * 
 * Simulates a multi-node network where packets are transmitted
 * between nodes over discrete time steps.
 */

#define PACKET_SIZE 256
#define NUM_TIME_STEPS 10

/*
 * A single in-flight packet.  Each rank has exactly 2 mailbox slots
 * in the distributed array: slot 0 = from left neighbor (rank-1),
 *                           slot 1 = from right neighbor (rank+1).
 * "Left" and "right" refer to ring direction only; the actual data
 * can originate anywhere because we use cut-through routing.
 */
#define MAILBOX_SLOTS  2          /* [0] = left inbox, [1] = right inbox  */
#define SLOT_LEFT      0
#define SLOT_RIGHT     1

typedef struct {
    int          valid;           /* 1 = a real packet is here            */
    int          source_rank;     /* original sender                      */
    int          dest_rank;       /* final destination                    */
    dstm_time_t  origin_time;     /* logical time the packet was injected */
    char         payload[PACKET_SIZE - 20];
} mailbox_entry_t;

#define SEND_PROBABILITY 60       /* % chance each rank sends each step   */

static void usage(const char *prog)
{
	fprintf(stderr,
	        "usage: %s [--seed S] [--quiet]\n",
	        prog);
}

int main(int argc, char *argv[]) {
    int seed = 42;
    int quiet = 0;

    for (int i = 1; i < argc; i++) {
	if (!strcmp(argv[i], "--seed") && i + 1 < argc)
	    seed = atoi(argv[++i]);
	else if (!strcmp(argv[i], "--quiet"))
            quiet = 1;
	else if (!strcmp(argv[i], "--help") ||
		 !strcmp(argv[i], "-h")) {
	    usage(argv[0]);
	    return 0;
	}
    }

    int rank, num_ranks;
    dstm_distance_t *topology;

    dstm_darray_t *mailbox;
    mailbox_entry_t *local_slots;   /* points into our 2 local slots */
    dstm_time_t current_time;

    mailbox_entry_t recv_entry;
    mailbox_entry_t empty_entry = { .valid = 0, .source_rank = -1 };

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    /* Initialize DSTM logging and runtime */
    dstm_init(&argc, &argv);
    dstm_log_init(stdout, "network_sim");

    if (quiet)
	dstm_log_setlevel(DSTM_LOG_QUIET);
    else
	dstm_log_setlevel(DSTM_LOG_INFO);

    /* Create network topology (ring in this example) */
    if (rank == 0) {
        printf("Creating ring topology with %d nodes\n", num_ranks);
    }
    
    int status = dstm_distance_create_ring(&topology, num_ranks);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to create topology");
        MPI_Finalize();
        return -1;
    }

    /* Start DSTM with the topology */
    status = dstm_start(topology);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to start DSTM");
        return -1;
    }

    /*
     * Total elements = num_ranks * MAILBOX_SLOTS (2 per rank).
     * Each rank owns exactly 2 entries: its left inbox and right inbox.
     * Global index layout:
     *   rank r, slot s  →  r * MAILBOX_SLOTS + s
     */
    status = dstm_darray_create_equal(&mailbox,
                                  sizeof(mailbox_entry_t),
                                  num_ranks * MAILBOX_SLOTS);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to create mailbox");
        return -1;
    }

    local_slots = (mailbox_entry_t *)dstm_darray_get_local_ptr(mailbox);

    srand(seed + rank);   /* each rank gets a different but reproducible seed */

    int left_rank  = (rank - 1 + num_ranks) % num_ranks;
    int right_rank = (rank + 1) % num_ranks;

    for (current_time = 0; current_time < NUM_TIME_STEPS; current_time++) {
        dstm_log_info("Rank %d: time step %ld\n", rank, current_time);

        /* Each rank writes empty_entry to its OWN slots each timestep as a default,
         * before the send phase. Real packets from senders will overwrite these. */
        //TODO lazy fix
        for (int slot = 0; slot < MAILBOX_SLOTS; slot++) {
            size_t my_index = (size_t)rank * MAILBOX_SLOTS + slot;
            dstm_darray_put(mailbox, my_index, &empty_entry, current_time, 0);
        }

        /* ---------------------------------------------------------------
         * RECEIVE PHASE
         * Check both inbox slots.  Slot SLOT_LEFT holds packets arriving
         * from our left neighbor's direction; SLOT_RIGHT from the right.
         * We GET at (current_time, delta=0) — the packet was PUT with a
         * delta equal to the hop distance, so it arrives exactly now.
         * --------------------------------------------------------------- */
        for (int slot = 0; slot < MAILBOX_SLOTS; slot++) {
            size_t my_index = (size_t)rank * MAILBOX_SLOTS + slot;

            status = dstm_darray_get_or_default(mailbox,
                                                my_index,
                                                &recv_entry,
                                                current_time,
                                                0,        /* delta = 0: read current time */
                                                &empty_entry);
        
            if (status != DSTM_SUCCESS) {
                dstm_log_error("GET failed on slot %d", slot);
                continue;
            }

            if (!recv_entry.valid) continue;  /* inbox slot is empty */

            if (!quiet) {
                printf("Rank %d [t=%ld]: RECEIVED via %s slot — "
                       "from rank %d, sent at t=%ld: \"%s\"\n",
                       rank, current_time,
                       slot == SLOT_LEFT ? "left" : "right",
                       recv_entry.source_rank,
                       recv_entry.origin_time,
                       recv_entry.payload);
            }
        }

        /* ---------------------------------------------------------------
         * SEND PHASE
         * Randomly decide whether to send, and to whom.
         * We use cut-through routing: PUT directly into the destination's
         * mailbox with delta = hop distance (from the ring topology).
         * The destination's slot is chosen by which side the packet
         * approaches from:
         *   - If we are to the RIGHT of dest  → packet enters dest's SLOT_LEFT
         *   - If we are to the LEFT  of dest  → packet enters dest's SLOT_RIGHT
         * --------------------------------------------------------------- */
        if ((rand() % 100) < SEND_PROBABILITY) {

            /* pick a random destination that is not ourselves */
            int dest = rand() % (num_ranks - 1);
            if (dest >= rank) dest++;

            /* clockwise (right) distance from us to dest */
            int dist_right = (dest - rank + num_ranks) % num_ranks;
            /* counter-clockwise (left) distance */
            int dist_left  = num_ranks - dist_right;

            /*
             * dstm_distance_get returns the shortest-path distance between two nodes
             * as encoded in the topology matrix.
             */
            int hops = dstm_distance_get(topology, rank, dest);

            /* Direction is still derived geometrically — only used to pick the slot,
             * not to compute the distance. */
            int dest_slot  = (dist_right <= num_ranks - dist_right) ? SLOT_LEFT : SLOT_RIGHT;

            size_t dest_index = (size_t)dest * MAILBOX_SLOTS + dest_slot;

            mailbox_entry_t pkt = {
                .valid       = 1,
                .source_rank = rank,
                .dest_rank   = dest,
                .origin_time = current_time,
            };
            snprintf(pkt.payload, sizeof(pkt.payload),
                     "pkt src=%d dst=%d hops=%d t=%ld",
                     rank, dest, hops, current_time);

            /* cut-through PUT: delta = hop count, arrives at dest at t+hops */
            status = dstm_darray_put(mailbox, dest_index, &pkt,
                                     current_time, (dstm_time_t)hops);
            if (status != DSTM_SUCCESS) {
                dstm_log_error("PUT failed to rank %d slot %d", dest, dest_slot);
            } else {
                dstm_log_info("Rank %d [t=%ld]: SENT to rank %d "
                              "(%d hops, arrives slot %s at t=%ld)\n",
                              rank, current_time, dest, hops,
                              dest_slot == SLOT_LEFT ? "left" : "right",
                              current_time + hops);
            }
        }
    }
    
    /* Barrier to ensure all packets are delivered */
    dstm_time_t last_time = NUM_TIME_STEPS + 1;
    status = dstm_barrier(last_time);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Barrier failed");
    }

    dstm_darray_destroy(&mailbox);

    /* Finalize DSTM */
    status = dstm_fini(last_time);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to finalize DSTM");
    }

    /* Cleanup */
    dstm_distance_destroy(&topology);
    dstm_finalize();

    MPI_Finalize();
    return status;
}
