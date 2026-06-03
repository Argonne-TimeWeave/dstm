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

#define LOCAL_PACKET_SLOTS 8
#define NUM_TIME_STEPS 10
#define PACKET_SIZE 256

/* Simple packet structure */
typedef struct {
    int source_rank;
    int dest_rank;
    dstm_time_t timestamp;
    char payload[PACKET_SIZE - 16];
} network_packet_t;

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
    dstm_darray_t *packet_buffer;
    network_packet_t *local_packets;
    network_packet_t recv_packet;
    dstm_time_t current_time;
    int status;

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
	dstm_log_setlevel(DSTM_LOG_DEBUG);

    /* Create network topology (ring in this example) */
    if (rank == 0) {
        printf("Creating ring topology with %d nodes\n", num_ranks);
    }
    
    status = dstm_distance_create_ring(&topology, num_ranks);
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

    /* Create distributed packet buffer (one packet per rank) */
    status = dstm_darray_create_equal(&packet_buffer,
                                      sizeof(network_packet_t),
                                      num_ranks * LOCAL_PACKET_SLOTS);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Failed to create packet buffer");
        return -1;
    }

    local_packets = (network_packet_t *)dstm_darray_get_local_ptr(packet_buffer);

    /* Simulation main loop */
    for (current_time = 0; current_time < NUM_TIME_STEPS; current_time++) {
        dstm_log_info("Rank %d: Time step %ld", rank, current_time);

        /* Each rank creates a packet to send to its ring neighbor */
        int next_rank = (rank + 1) % num_ranks;
	size_t slot = current_time % LOCAL_PACKET_SLOTS;

        /* Prepare packet */
	local_packets[slot].source_rank = rank;
        local_packets[slot].dest_rank = next_rank;
        local_packets[slot].timestamp = current_time;
        snprintf(local_packets[slot].payload, sizeof(local_packets[slot].payload),
             "Packet from rank %d at time %ld", rank, current_time);

        size_t send_index = next_rank * LOCAL_PACKET_SLOTS + slot;
        size_t recv_index = rank * LOCAL_PACKET_SLOTS + slot;

        /* PUT: Send packet to next rank's buffer (lookahead = 1 time step) */
        status = dstm_darray_put(packet_buffer,
                                send_index,
                                &local_packets[slot],
                                current_time,
                                1);  /* delta = 1: packet arrives next time step */
        
        if (status != DSTM_SUCCESS) {
            dstm_log_error("PUT failed");
            continue;
        }

        dstm_log_info("Rank %d sent packet to rank %d", rank, next_rank);

        /* GET: Receive packet from previous rank (if available) */
	if (current_time > 0) {
	    status = dstm_darray_get_or_default(packet_buffer,
					   recv_index,
					   &local_packets[slot],
					   current_time,
					   0,  /* delta = 0: get current version */
					   (const void *)&local_packets[slot]);
	
	    if (status == DSTM_SUCCESS) {
	        if (local_packets[slot].source_rank >= 0) {
		    dstm_log_info("Rank %d received: %s",
			    rank, recv_packet.payload);
	        }
	    }
	}
    }

    /* Barrier to ensure all packets are delivered */
    dstm_time_t last_time = NUM_TIME_STEPS + 1;
    status = dstm_barrier(last_time);
    if (status != DSTM_SUCCESS) {
        dstm_log_error("Barrier failed");
    }

    dstm_darray_destroy(&packet_buffer);

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
