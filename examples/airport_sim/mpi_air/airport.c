#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> 
#include <mpi.h>

#include "airport.h"

const char *tier_name[] = { "HUB", "REGIONAL", "LOCAL" };
const int tier_send_prob[] = { 80, 40, 15 };
const int tier_recv_weight[] = { 8, 3, 1 };
const gate_slot_t empty_gate = { .n_flights = 0 };

/* ---------- Pending-flight queue (per-rank, in-transit storage) ----------
 *
 * Mirrors DSTM's delta semantics: a flight sent at time t with delta=d
 * becomes visible at time t+d. We park flights here on the receiver
 * side and only "land" them once their arrive_time matches the current
 * timestep. */
typedef struct {
    flight_t *flights;
    int       n;
    int       cap;
} pending_queue_t;

static void pending_push(pending_queue_t *q, const flight_t *flt)
{
    if (q->n == q->cap) {
        int new_cap = q->cap ? q->cap * 2 : 64;
        q->flights = realloc(q->flights, sizeof(flight_t) * new_cap);
        q->cap = new_cap;
    }
    q->flights[q->n++] = *flt;
}

/* Remove element at index i by swapping last element down. */
static void pending_remove_at(pending_queue_t *q, int i)
{
    q->flights[i] = q->flights[q->n - 1];
    q->n--;
}

/*
 * Synthetic per-arrival workload. Does `iterations` of a cheap floating-point
 * computation. We mark the accumulator volatile and return it so the
 * compiler can't optimize the loop away.
 */
double airport_do_work(int iterations)
{
    if (iterations <= 0)
        return 0.0;

    volatile double acc = 1.0;
    for (int i = 1; i <= iterations; i++) {
        /* Mix of mul/add/sqrt — keeps the FPU honest without allocating. */
        acc = acc + sqrt((double)i) * 1.0000001;
        acc = acc - (acc * 0.5);
    }
    return acc;
}

void airport_init(airport_state_t *state,
                  int rank,
                  int num_ranks,
                  int na,
                  int work,
                  airport_tier_t *tiers,
                  dstm_distance_t *topology,
                  dstm_darray_t *gates)
{
    state->rank           = rank;
    state->num_ranks      = num_ranks;
    state->na             = na;
    state->work           = work;
    state->tiers          = tiers;
    state->topology       = topology;
    state->gates          = gates;
    state->outbox         = calloc(num_ranks, sizeof(gate_slot_t));
    state->flight_counter = 0;
    state->departures     = calloc(na, sizeof(int));
    state->arrivals       = calloc(na, sizeof(int));
    state->connections    = calloc(na, sizeof(int));
}

void airport_destroy(airport_state_t *state)
{
    free(state->outbox);
    free(state->departures);
    free(state->arrivals);
    free(state->connections);
    state->outbox      = NULL;
    state->departures  = NULL;
    state->arrivals    = NULL;
    state->connections = NULL;
}

/*
 * Pick a destination airport, biased by the tier of the candidate.
 * Destination must not live on origin_rank (no local-to-local flights).
 */
void airport_pick_dest(airport_state_t *state,
                       int origin_rank,
                       int *dest_rank,
                       int *dest_local)
{
    int na = state->na;
    int total_weight = 0;

    for (int r = 0; r < state->num_ranks; r++) {
        if (r == origin_rank)
            continue;
        for (int l = 0; l < na; l++) {
            int g = airport_global(r, l, na);
            total_weight += tier_recv_weight[state->tiers[g]];
        }
    }

    if (total_weight <= 0) {
        /* Degenerate (only one rank); caller should have guarded. */
        *dest_rank  = (origin_rank + 1) % state->num_ranks;
        *dest_local = 0;
        return;
    }

    int rnd = rand() % total_weight;
    int accum = 0;
    for (int r = 0; r < state->num_ranks; r++) {
        if (r == origin_rank)
            continue;
        for (int l = 0; l < na; l++) {
            int g = airport_global(r, l, na);
            accum += tier_recv_weight[state->tiers[g]];
            if (rnd < accum) {
                *dest_rank  = r;
                *dest_local = l;
                return;
            }
        }
    }

    /* Fallback */
    *dest_rank  = (origin_rank + 1) % state->num_ranks;
    *dest_local = 0;
}

/*
 * Find next-hop RANK from origin_rank toward dest_rank.
 * Distance is per-rank (node-level). Prefers ranks that host at least
 * one HUB-tier airport when picking a layover.
 */
int airport_find_next_hop(airport_state_t *state, int origin_rank, int dest_rank)
{
    int dist_to_dest = dstm_distance_get(state->topology, origin_rank, dest_rank);

    if (dist_to_dest <= LAYOVER_THRESHOLD)
        return dest_rank;

    int best_hop = -1;
    int best_remaining = dist_to_dest;
    int best_is_hub = 0;
    int na = state->na;

    for (int r = 0; r < state->num_ranks; r++) {
        if (r == origin_rank || r == dest_rank)
            continue;

        int dist_to_hop = dstm_distance_get(state->topology, origin_rank, r);
        if (dist_to_hop > LAYOVER_THRESHOLD)
            continue;

        int remaining = dstm_distance_get(state->topology, r, dest_rank);
        if (remaining >= dist_to_dest)
            continue;

        /* Rank "is a hub" if any of its local airports is a HUB tier */
        int is_hub = 0;
        for (int l = 0; l < na; l++) {
            if (state->tiers[airport_global(r, l, na)] == TIER_HUB) {
                is_hub = 1;
                break;
            }
        }

        if (best_hop == -1 ||
            (is_hub && !best_is_hub) ||
            (is_hub == best_is_hub && remaining < best_remaining)) {
            best_hop = r;
            best_remaining = remaining;
            best_is_hub = is_hub;
        }
    }

    if (best_hop == -1)
        return dest_rank;

    return best_hop;
}

#define AIRPORT_MPI_TAG_BASE 1000

void airport_timestep(airport_state_t *state, dstm_time_t t, int quiet)
{
    int rank = state->rank;
    int num_ranks = state->num_ranks;
    int na = state->na;

    dstm_log_normal("Rank %d [t=%ld]: === timestep ===\n", rank, t);

    /* Per-rank persistent state. Single-threaded per rank, so static is fine. */
    static pending_queue_t pending = { NULL, 0, 0 };
    static gate_slot_t    *saved_inbox = NULL;

    if (saved_inbox == NULL)
        saved_inbox = calloc(num_ranks, sizeof(gate_slot_t));

    /* Drain saved_inbox from previous timestep's exchange into pending,
     * preserving each flight's arrive_time stamped by the sender. */
    for (int src = 0; src < num_ranks; src++) {
        if (src == rank) continue;
        gate_slot_t *gs = &saved_inbox[src];
        for (int f = 0; f < gs->n_flights; f++)
            pending_push(&pending, &gs->flights[f]);
        gs->n_flights = 0;
    }

    /* Clear outbox; we'll rebuild it from this timestep's processing. */
    for (int r = 0; r < num_ranks; r++)
        state->outbox[r] = empty_gate;

    /* Allocate a fresh inbox for this step's exchange. */
    gate_slot_t *inbox = malloc(sizeof(gate_slot_t) * num_ranks);
    for (int r = 0; r < num_ranks; r++)
        inbox[r] = empty_gate;

    /* ---------- STEP 3: process pending flights due at time t ---------- */
    int i = 0;
    while (i < pending.n) {
        flight_t *flt_p = &pending.flights[i];
        if (flt_p->arrive_time != t) {
            i++;
            continue;
        }
        flight_t flt = *flt_p;
        pending_remove_at(&pending, i);
        /* Don't increment i; the swap put a new element at position i. */

        if (flt.dest_rank == rank) {
            int dl = flt.dest_local;
            if (dl < 0 || dl >= na) {
                dstm_log_error("Bad dest_local=%d on flight %d", dl, flt.flight_id);
                continue;
            }
            state->arrivals[dl]++;
            volatile double sink = airport_do_work(state->work);
            (void)sink;

            int og = airport_global(flt.origin_rank, flt.origin_local, na);
            dstm_log_normal("Rank %d [t=%ld]: LANDED flight %d at airport "
                            "%d.%d — origin=%d.%d (%s) dep_t=%ld legs=%d \"%s\"\n",
                            rank, t, flt.flight_id, rank, dl,
                            flt.origin_rank, flt.origin_local,
                            tier_name[state->tiers[og]],
                            flt.departure_time, flt.legs_completed + 1,
                            flt.payload);
        } else {
            /* Connecting flight — forward to next hop with its own delta. */
            int local_hub = (flt.origin_rank * 49) % NUM_GATES % na;
            state->connections[local_hub]++;
            flt.legs_completed++;

            int next_hop = airport_find_next_hop(state, rank, flt.dest_rank);
            int delta = dstm_distance_get(state->topology, rank, next_hop);
            if (delta < 1) delta = 1;  /* never schedule into the past */

            gate_slot_t *ob = &state->outbox[next_hop];
            if (ob->n_flights < GATE_CAPACITY) {
                flt.next_hop_rank = next_hop;
                flt.arrive_time = t + delta;
                ob->flights[ob->n_flights++] = flt;

                if (next_hop == flt.dest_rank) {
                    dstm_log_normal("Rank %d [t=%ld]: CONNECTING flight %d "
                                    "-> final dst rank=%d (airport %d.%d) "
                                    "delta=%d arrive_t=%ld (leg %d)\n",
                                    rank, t, flt.flight_id, next_hop,
                                    flt.dest_rank, flt.dest_local, delta,
                                    flt.arrive_time, flt.legs_completed);
                } else {
                    dstm_log_normal("Rank %d [t=%ld]: CONNECTING flight %d "
                                    "-> hop rank=%d toward dst=%d.%d delta=%d "
                                    "arrive_t=%ld (leg %d)\n",
                                    rank, t, flt.flight_id, next_hop,
                                    flt.dest_rank, flt.dest_local, delta,
                                    flt.arrive_time, flt.legs_completed);
                }
            } else {
                dstm_log_warning("Rank %d [t=%ld]: CONNECTION DELAYED "
                                 "flight %d — outbox to rank %d full\n",
                                 rank, t, flt.flight_id, next_hop);
            }
        }
    }

    /* ---------- STEP 4: inject new departures into outbox ---------- */
    if (num_ranks > 1) {
        for (int l = 0; l < na; l++) {
            int og = airport_global(rank, l, na);
            airport_tier_t my_tier = state->tiers[og];
            if ((rand() % 100) >= tier_send_prob[my_tier]) continue;

            int dest_rank, dest_local;
            airport_pick_dest(state, rank, &dest_rank, &dest_local);
            int next_hop = airport_find_next_hop(state, rank, dest_rank);
            int delta = dstm_distance_get(state->topology, rank, next_hop);
            if (delta < 1) delta = 1;

            gate_slot_t *ob = &state->outbox[next_hop];
            if (ob->n_flights >= GATE_CAPACITY) {
                dstm_log_warning("Rank %d [t=%ld]: DEPART DROPPED from %d.%d "
                                 "— outbox to rank %d full\n",
                                 rank, t, rank, l, next_hop);
                continue;
            }

            int dg = airport_global(dest_rank, dest_local, na);
            flight_t flt = {
                .valid = 1, .origin_rank = rank, .origin_local = l,
                .dest_rank = dest_rank, .dest_local = dest_local,
                .next_hop_rank = next_hop, .legs_completed = 0,
                .departure_time = t,
                .arrive_time = t + delta,   /* NEW */
                .flight_id = (rank * na + l) * 100000 + state->flight_counter++,
            };
            int is_direct = (next_hop == dest_rank);
            snprintf(flt.payload, sizeof(flt.payload),
                     "%s(%d.%d)->%s(%d.%d)%s t=%ld",
                     tier_name[my_tier], rank, l,
                     tier_name[state->tiers[dg]], dest_rank, dest_local,
                     is_direct ? "" : " [LAYOVER]", t);
            ob->flights[ob->n_flights++] = flt;
            state->departures[l]++;

            if (is_direct) {
                dstm_log_normal("Rank %d [t=%ld]: DEPART flight %d from %d.%d "
                                "dst=%d.%d (%s) DIRECT delta=%d arrive_t=%ld\n",
                                rank, t, flt.flight_id, rank, l,
                                dest_rank, dest_local,
                                tier_name[state->tiers[dg]], delta,
                                flt.arrive_time);
            } else {
                dstm_log_normal("Rank %d [t=%ld]: DEPART flight %d from %d.%d "
                                "dst=%d.%d via rank %d delta=%d arrive_t=%ld\n",
                                rank, t, flt.flight_id, rank, l,
                                dest_rank, dest_local, next_hop, delta,
                                flt.arrive_time);
            }
        }
    }

    /* ---------- STEP 5: exchange (always one round-trip per step) ---------- */
    MPI_Request *recv_reqs = malloc(sizeof(MPI_Request) * num_ranks);
    MPI_Request *send_reqs = malloc(sizeof(MPI_Request) * num_ranks);
    int n_recv = 0, n_send = 0;

    for (int src = 0; src < num_ranks; src++) {
        if (src == rank) continue;
        MPI_Irecv(&inbox[src], (int)sizeof(gate_slot_t), MPI_BYTE,
                  src, AIRPORT_MPI_TAG_BASE + (int)t,
                  MPI_COMM_WORLD, &recv_reqs[n_recv++]);
    }
    for (int dest = 0; dest < num_ranks; dest++) {
        if (dest == rank) continue;
        MPI_Isend(&state->outbox[dest], (int)sizeof(gate_slot_t), MPI_BYTE,
                  dest, AIRPORT_MPI_TAG_BASE + (int)t,
                  MPI_COMM_WORLD, &send_reqs[n_send++]);
    }
    if (n_recv > 0) MPI_Waitall(n_recv, recv_reqs, MPI_STATUSES_IGNORE);
    if (n_send > 0) MPI_Waitall(n_send, send_reqs, MPI_STATUSES_IGNORE);
    free(recv_reqs);
    free(send_reqs);

    /* Stash this step's received gates for next step's pending-queue ingest. */
    memcpy(saved_inbox, inbox, sizeof(gate_slot_t) * num_ranks);

    free(inbox);
    MPI_Barrier(MPI_COMM_WORLD);
}  
