#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "simulation.h"

#define TAG_COUNT   1001
#define TAG_PAYLOAD 1002

static void compute_ownership(simulation_state_t *sim)
{
    int rank = sim->rank;
    int num_ranks = sim->num_ranks;

    int rank_cols = num_ranks;
    for (int r = 1; r * r <= num_ranks; r++) {
        if (num_ranks % r == 0) rank_cols = num_ranks / r;
    }
    int rank_rows = num_ranks / rank_cols;

    sim->rank_rows = rank_rows;
    sim->rank_cols = rank_cols;
    sim->grid_rows = sim->block_rows * rank_rows;
    sim->grid_cols = sim->block_cols * rank_cols;
    sim->num_airports = sim->grid_rows * sim->grid_cols;

    sim->num_my_airports = 0;
    for (int r = 0; r < sim->grid_rows; r++) {
        for (int c = 0; c < sim->grid_cols; c++) {
            int airport_id = r * sim->grid_cols + c;
            int owner_rank_row = r / sim->block_rows;
            int owner_rank_col = c / sim->block_cols;
            int owner = owner_rank_row * rank_cols + owner_rank_col;
            sim->airport_owner[airport_id] = owner;
            if (owner == rank) {
                sim->my_airports[sim->num_my_airports++] = airport_id;
            }
        }
    }
}

void simulation_gather_stats(simulation_state_t *sim,
                             int *total_departures,
                             int *total_arrivals,
                             int *total_connections)
{
    *total_departures = 0;
    *total_arrivals = 0;
    *total_connections = 0;
    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        *total_departures  += sim->airports[airport_id].departures;
        *total_arrivals    += sim->airports[airport_id].arrivals;
        *total_connections += sim->airports[airport_id].connections;
    }
}

static void build_flight_mpi_type(MPI_Datatype *out)
{
    MPI_Type_contiguous(sizeof(flight_t), MPI_BYTE, out);
    MPI_Type_commit(out);
}

int simulation_init(simulation_state_t *sim, int rank, int num_ranks,
                    int block_rows, int block_cols, int seed)
{
    memset(sim, 0, sizeof(*sim));
    sim->rank = rank;
    sim->num_ranks = num_ranks;
    sim->block_rows = block_rows;
    sim->block_cols = block_cols;
    sim->airports_per_rank = block_rows * block_cols;

    int rank_cols = num_ranks;
    for (int r = 1; r * r <= num_ranks; r++) {
        if (num_ranks % r == 0) rank_cols = num_ranks / r;
    }
    int rank_rows = num_ranks / rank_cols;
    int grid_rows = block_rows * rank_rows;
    int grid_cols = block_cols * rank_cols;
    int num_airports = grid_rows * grid_cols;

    sim->my_airports   = malloc(num_airports * sizeof(int));
    sim->airport_owner = malloc(num_airports * sizeof(int));
    sim->airports      = calloc(num_airports, sizeof(airport_state_t));
    sim->tiers         = malloc(num_airports * sizeof(airport_tier_t));
    sim->local_inbox   = calloc(num_airports, sizeof(gate_slot_t *));

    if (!sim->my_airports || !sim->airport_owner || !sim->airports ||
        !sim->tiers || !sim->local_inbox) {
        fprintf(stderr, "Failed to allocate simulation memory\n");
        return -1;
    }

    compute_ownership(sim);

    /* Tier assignment must be identical on all ranks */
    srand(seed);
    for (int i = 0; i < sim->num_airports; i++) {
        int r = rand() % 100;
        if (r < 15)      sim->tiers[i] = TIER_HUB;
        else if (r < 50) sim->tiers[i] = TIER_REGIONAL;
        else             sim->tiers[i] = TIER_LOCAL;
    }

    /* Per-rank diverged RNG for flight injection */
    srand(seed + rank);

    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        airport_init(&sim->airports[airport_id], airport_id, sim->num_airports);
        sim->local_inbox[airport_id] = calloc(NUM_GATES, sizeof(gate_slot_t));
    }

    /* Per-destination-rank send bucket bookkeeping.
       We reuse send_counts/recv_counts/send_displs from the struct, but the
       payload itself is staged as one bucket per destination rank so we can
       fire individual Sends. */
    sim->send_counts = calloc(num_ranks, sizeof(int));
    sim->recv_counts = calloc(num_ranks, sizeof(int));
    sim->send_displs = calloc(num_ranks, sizeof(int));
    sim->recv_displs = calloc(num_ranks, sizeof(int));

    sim->send_buf_cap = 1024;
    sim->recv_buf_cap = 1024;
    sim->send_buf = malloc(sim->send_buf_cap * sizeof(flight_t));
    sim->recv_buf = malloc(sim->recv_buf_cap * sizeof(flight_t));

    build_flight_mpi_type(&sim->flight_type);
    return 0;
}

void simulation_destroy(simulation_state_t *sim)
{
    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        airport_destroy(&sim->airports[airport_id]);
        free(sim->local_inbox[airport_id]);
    }
    free(sim->local_inbox);
    free(sim->my_airports);
    free(sim->airport_owner);
    free(sim->airports);
    free(sim->tiers);

    free(sim->send_counts);
    free(sim->recv_counts);
    free(sim->send_displs);
    free(sim->recv_displs);
    free(sim->send_buf);
    free(sim->recv_buf);

    MPI_Type_free(&sim->flight_type);
}

static void merge_into_gate(gate_slot_t *dest, gate_slot_t *src)
{
    for (int f = 0; f < src->n_flights && dest->n_flights < GATE_CAPACITY; f++) {
        dest->flights[dest->n_flights++] = src->flights[f];
    }
}

static void ensure_send_cap(simulation_state_t *sim, int needed)
{
    if (needed <= sim->send_buf_cap) return;
    int cap = sim->send_buf_cap;
    while (cap < needed) cap *= 2;
    sim->send_buf = realloc(sim->send_buf, cap * sizeof(flight_t));
    sim->send_buf_cap = cap;
}

static void ensure_recv_cap(simulation_state_t *sim, int needed)
{
    if (needed <= sim->recv_buf_cap) return;
    int cap = sim->recv_buf_cap;
    while (cap < needed) cap *= 2;
    sim->recv_buf = realloc(sim->recv_buf, cap * sizeof(flight_t));
    sim->recv_buf_cap = cap;
}

void simulation_timestep(simulation_state_t *sim, dstm_time_t t, int quiet)
{
    int grid_cols = sim->grid_cols;
    int num_airports = sim->num_airports;
    int num_ranks = sim->num_ranks;
    int my_rank = sim->rank;
    (void)quiet;

    /* 1. Clear outboxes */
    for (int i = 0; i < sim->num_my_airports; i++) {
        airport_clear_outbox(&sim->airports[sim->my_airports[i]]);
    }

    /* 2. Process any flights deposited locally during previous step */
    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        for (int g = 0; g < NUM_GATES; g++) {
            gate_slot_t *gs = &sim->local_inbox[airport_id][g];
            if (gs->n_flights > 0) {
                airport_process_incoming(&sim->airports[airport_id], gs,
                                         grid_cols, sim->tiers, t);
                *gs = empty_gate;
            }
        }
    }

    /* 3. Inject new flights */
    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        airport_inject_flight(&sim->airports[airport_id], grid_cols, sim->tiers, t);
    }

    /* 4. Bucket outbox by destination rank; route same-rank flights directly */
    for (int r = 0; r < num_ranks; r++) sim->send_counts[r] = 0;

    /* Count */
    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        airport_state_t *ap = &sim->airports[airport_id];
        for (int dest = 0; dest < num_airports; dest++) {
            if (dest == airport_id) continue;
            int n = ap->outbox[dest].n_flights;
            if (n == 0) continue;
            int dest_owner = sim->airport_owner[dest];
            if (dest_owner == my_rank) continue;
            sim->send_counts[dest_owner] += n;
        }
    }

    int total_send = 0;
    for (int r = 0; r < num_ranks; r++) {
        sim->send_displs[r] = total_send;
        total_send += sim->send_counts[r];
    }
    ensure_send_cap(sim, total_send);

    /* Pack send buffer + handle local routes */
    int *cursor = calloc(num_ranks, sizeof(int));
    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        airport_state_t *ap = &sim->airports[airport_id];
        for (int dest = 0; dest < num_airports; dest++) {
            if (dest == airport_id) continue;
            gate_slot_t *gs = &ap->outbox[dest];
            if (gs->n_flights == 0) continue;
            int dest_owner = sim->airport_owner[dest];

            if (dest_owner == my_rank) {
                int gate = (airport_id * 49) % NUM_GATES;
                merge_into_gate(&sim->local_inbox[dest][gate], gs);
                dstm_log_normal("Airport %d [t=%ld]: LOCAL send %d flights to airport %d\n",
                                airport_id, t, gs->n_flights, dest);
            } else {
                int off = sim->send_displs[dest_owner] + cursor[dest_owner];
                for (int f = 0; f < gs->n_flights; f++) {
                    sim->send_buf[off + f] = gs->flights[f];
                }
                cursor[dest_owner] += gs->n_flights;
            }
        }
    }
    free(cursor);

    /* 5. Exchange counts via per-pair Sendrecv (one with every other rank).
          This is deliberately O(P) per rank — the naive approach. */
    for (int r = 0; r < num_ranks; r++) {
        if (r == my_rank) {
            sim->recv_counts[r] = 0;
            continue;
        }
        MPI_Sendrecv(&sim->send_counts[r], 1, MPI_INT, r, TAG_COUNT,
                     &sim->recv_counts[r], 1, MPI_INT, r, TAG_COUNT,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int total_recv = 0;
    for (int r = 0; r < num_ranks; r++) {
        sim->recv_displs[r] = total_recv;
        total_recv += sim->recv_counts[r];
    }
    ensure_recv_cap(sim, total_recv);

    /* 6. Exchange payloads with each peer that has nonzero traffic in either
          direction. Use Sendrecv to avoid deadlock; skip pairs where both
          counts are zero to save calls. This is the "harsh" pattern: P-1
          point-to-point exchanges per rank per timestep. */
    for (int r = 0; r < num_ranks; r++) {
        if (r == my_rank) continue;
        int sc = sim->send_counts[r];
        int rc = sim->recv_counts[r];
        if (sc == 0 && rc == 0) continue;

        flight_t *sbuf = sim->send_buf + sim->send_displs[r];
        flight_t *rbuf = sim->recv_buf + sim->recv_displs[r];

        MPI_Sendrecv(sbuf, sc, sim->flight_type, r, TAG_PAYLOAD,
                     rbuf, rc, sim->flight_type, r, TAG_PAYLOAD,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* 7. Deposit received flights into local_inbox */
    for (int k = 0; k < total_recv; k++) {
        flight_t *flt = &sim->recv_buf[k];
        int landing = flt->next_hop;
        int source  = flt->origin_airport;
        int gate = (source * 49) % NUM_GATES;
        gate_slot_t *gs = &sim->local_inbox[landing][gate];
        if (gs->n_flights < GATE_CAPACITY) {
            gs->flights[gs->n_flights++] = *flt;
        } else {
            dstm_log_warning("Airport %d [t=%ld]: RECV gate full, dropping flight %d\n",
                             landing, t, flt->flight_id);
        }
    }

    /* 8. Lock-step barrier */
    MPI_Barrier(MPI_COMM_WORLD);
}
