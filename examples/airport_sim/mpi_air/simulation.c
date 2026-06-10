#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "simulation.h"

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
    /* Safe & portable: just send it as a byte blob. flight_t has no pointers. */
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

    /* local_inbox[airport_id] only valid for owned airports, but we
       allocate the top-level pointer array for all to keep indexing simple. */
    sim->local_inbox = calloc(num_airports, sizeof(gate_slot_t *));

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

    /* MPI exchange buffers */
    sim->send_counts = calloc(num_ranks, sizeof(int));
    sim->recv_counts = calloc(num_ranks, sizeof(int));
    sim->send_displs = calloc(num_ranks, sizeof(int));
    sim->recv_displs = calloc(num_ranks, sizeof(int));

    /* Initial capacity: generous guess, grown on demand */
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

void simulation_timestep(simulation_state_t *sim, dstm_time_t t, int logging, int quiet)
{
    int grid_cols = sim->grid_cols;
    int num_airports = sim->num_airports;
    int num_ranks = sim->num_ranks;
    (void)quiet;

    /* 1. Clear outboxes */
    for (int i = 0; i < sim->num_my_airports; i++) {
        airport_clear_outbox(&sim->airports[sim->my_airports[i]]);
    }

    /* 2. Process any flights that arrived (in local_inbox) from previous step.
          On t==0 these are all empty. */
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

    /* 3. Inject new flights from each owned airport */
    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        airport_inject_flight(&sim->airports[airport_id], grid_cols, sim->tiers, t);
    }

    /* 4. Bucket outbox by destination rank (and route local sends directly) */
    for (int r = 0; r < num_ranks; r++) sim->send_counts[r] = 0;

    /* First pass: count */
    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        airport_state_t *ap = &sim->airports[airport_id];
        for (int dest = 0; dest < num_airports; dest++) {
            if (dest == airport_id) continue;
            int n = ap->outbox[dest].n_flights;
            if (n == 0) continue;
            int dest_owner = sim->airport_owner[dest];
            if (dest_owner == sim->rank) continue; /* handled locally below */
            sim->send_counts[dest_owner] += n;
        }
    }

    int total_send = 0;
    for (int r = 0; r < num_ranks; r++) {
        sim->send_displs[r] = total_send;
        total_send += sim->send_counts[r];
    }
    ensure_send_cap(sim, total_send);

    /* Second pass: fill send buffer + handle local routes */
    int *cursor = calloc(num_ranks, sizeof(int)); /* per-rank write offset */
    for (int i = 0; i < sim->num_my_airports; i++) {
        int airport_id = sim->my_airports[i];
        airport_state_t *ap = &sim->airports[airport_id];
        for (int dest = 0; dest < num_airports; dest++) {
            if (dest == airport_id) continue;
            gate_slot_t *gs = &ap->outbox[dest];
            if (gs->n_flights == 0) continue;
            int dest_owner = sim->airport_owner[dest];

            if (dest_owner == sim->rank) {
                int gate = (airport_id * 49) % NUM_GATES;
                merge_into_gate(&sim->local_inbox[dest][gate], gs);
                dstm_log_normal("Airport %d [t=%ld]: LOCAL send %d flights to airport %d\n",
                                airport_id, t, gs->n_flights, dest);
            } else {
                int off = sim->send_displs[dest_owner] + cursor[dest_owner];
                for (int f = 0; f < gs->n_flights; f++) {
                    sim->send_buf[off + f] = gs->flights[f];
                    /* stash final destination via dest_airport (already in flight) */
                }
                cursor[dest_owner] += gs->n_flights;
            }
        }
    }
    free(cursor);
    
    double start, end;

    if (logging)
        start = MPI_Wtime();

    /* 5. Exchange counts, then payload */
    MPI_Alltoall(sim->send_counts, 1, MPI_INT,
                 sim->recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

    int total_recv = 0;
    for (int r = 0; r < num_ranks; r++) {
        sim->recv_displs[r] = total_recv;
        total_recv += sim->recv_counts[r];
    }
    ensure_recv_cap(sim, total_recv);

    MPI_Alltoallv(sim->send_buf, sim->send_counts, sim->send_displs, sim->flight_type,
                  sim->recv_buf, sim->recv_counts, sim->recv_displs, sim->flight_type,
                  MPI_COMM_WORLD);
    
    if (logging) {
        end = MPI_Wtime();
        printf("all_to_allv: %.7f\n", end - start);
    }

    /* 6. Deposit received flights into local_inbox to be processed next step.
          Bucket by destination airport's gate (using origin airport id, same
          formula as the DSTM version). */
    for (int k = 0; k < total_recv; k++) {
        flight_t *flt = &sim->recv_buf[k];
        /* The flight's immediate destination on this rank: it was sent toward
           its next_hop, but next_hop is only known to the sender. We need to
           recompute landing airport: the sender placed it in outbox[next_hop],
           and next_hop's owner is this rank. We didn't transmit next_hop
           separately, so encode it: the sender already stored next_hop in
           flt->next_hop. */
        int landing = flt->next_hop;
        int source  = flt->origin_airport; /* good enough for gate hashing parity */
        /* For parity with DSTM version, gate is based on the *previous*
           airport that sent it. On leg 0 that's origin; on later legs we
           don't track it explicitly — using origin is acceptable for a
           perf-comparison harness. */
        int gate = (source * 49) % NUM_GATES;
        gate_slot_t *gs = &sim->local_inbox[landing][gate];
        if (gs->n_flights < GATE_CAPACITY) {
            gs->flights[gs->n_flights++] = *flt;
        } else {
            dstm_log_warning("Airport %d [t=%ld]: RECV gate full, dropping flight %d\n",
                             landing, t, flt->flight_id);
        }
    }

    /* 7. Barrier to keep timesteps lock-step*/
    MPI_Barrier(MPI_COMM_WORLD);
}
