#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> 
#include <limits.h>

#include "airport.h"

const char *tier_name[] = { "HUB", "REGIONAL", "LOCAL" };
const int tier_send_prob[] = { 80, 40, 15 };
const int tier_recv_weight[] = { 8, 3, 1 };
const gate_slot_t empty_gate = { .n_flights = 0 };

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
                  int cols,
                  int work,
                  airport_tier_t *tiers,
                  dstm_distance_t *topology,
                  dstm_darray_t *gates)
{
    state->rank           = rank;
    state->num_ranks      = num_ranks;
    state->na             = na;
    state->cols           = cols;
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
 * Find next-hop RANK from origin_rank toward dest_rank on a rows x cols mesh.
 *
 * Distance matrix convention (after pruning):
 *   d >= 0  : reachable in one hop
 *   d == -1 : not reachable in one hop
 *
 * Selection rules, in priority order:
 *   1. Direct hop if the edge exists.
 *   2. Among reachable hops, prefer those that strictly reduce Manhattan
 *      distance to dest in (row, col) — this replaces the old
 *      "remaining < dist_to_dest" monotonic-progress rule.
 *   3. Prefer ranks hosting at least one HUB-tier airport (unchanged).
 *   4. Among equally-good Manhattan progress, prefer purely cardinal
 *      moves (row XOR col matches dest) over diagonal-ish ones.
 *   5. Smaller hop distance breaks any remaining ties.
 *
 * If no progress-making hop exists, fall back to returning dest_rank
 * (mirrors the original behavior).
 */
int airport_find_next_hop(airport_state_t *state, int origin_rank, int dest_rank)
{
    int cols = state->cols;
    int na   = state->na;

    /* 1) Direct edge wins. */
    int dist_to_dest_edge = dstm_distance_get(state->topology, origin_rank, dest_rank);
    if (dist_to_dest_edge >= 0)
        return dest_rank;

    /* Grid coordinates */
    int o_row = origin_rank / cols, o_col = origin_rank % cols;
    int d_row = dest_rank   / cols, d_col = dest_rank   % cols;

    int manh_origin_to_dest = abs(d_row - o_row) + abs(d_col - o_col);

    int best_hop       = -1;
    int best_manh      = manh_origin_to_dest;  /* must strictly improve */
    int best_is_hub    = 0;
    int best_cardinal  = 0;                    /* 1 if hop is purely N/S/E/W */
    int best_dist      = INT_MAX;

    for (int r = 0; r < state->num_ranks; r++) {
        if (r == origin_rank || r == dest_rank)
            continue;

        int dist_to_hop = dstm_distance_get(state->topology, origin_rank, r);
        if (dist_to_hop < 0)
            continue;  /* unreachable in one hop */

        int h_row = r / cols, h_col = r % cols;
        int manh_hop_to_dest = abs(d_row - h_row) + abs(d_col - h_col);

        /* Strict progress in Manhattan distance — analog of the original
         * "remaining >= dist_to_dest" prune. */
        if (manh_hop_to_dest >= manh_origin_to_dest)
            continue;

        /* Hub preference (unchanged from original) */
        int is_hub = 0;
        for (int l = 0; l < na; l++) {
            if (state->tiers[airport_global(r, l, na)] == TIER_HUB) {
                is_hub = 1;
                break;
            }
        }

        /* Purely cardinal step toward dest: aligns on one axis with origin
         * and moves only on the other axis. */
        int is_cardinal = (h_row == o_row) || (h_col == o_col);

        /* Lexicographic preference:
         *   smaller manh > is_hub > is_cardinal > smaller hop distance */
        int take = 0;
        if (best_hop == -1) {
            take = 1;
        } else if (manh_hop_to_dest != best_manh) {
            take = (manh_hop_to_dest < best_manh);
        } else if (is_hub != best_is_hub) {
            take = is_hub;
        } else if (is_cardinal != best_cardinal) {
            take = is_cardinal;
        } else {
            take = (dist_to_hop < best_dist);
        }

        if (take) {
            best_hop      = r;
            best_manh     = manh_hop_to_dest;
            best_is_hub   = is_hub;
            best_cardinal = is_cardinal;
            best_dist     = dist_to_hop;
        }
    }

    if (best_hop == -1)
        return dest_rank;   /* original fallback behavior */

    return best_hop;
}

void airport_timestep(airport_state_t *state, dstm_time_t t, int quiet)
{
    int rank = state->rank;
    int num_ranks = state->num_ranks;
    int na = state->na;
    int status;
    double start, end, sum = 0.0;

    dstm_log_normal("Rank %d [t=%ld]: === timestep ===\n", rank, t);

    /* STEP 2 — clear outbox and receive */
    for (int r = 0; r < num_ranks; r++)
        state->outbox[r] = empty_gate;

    gate_slot_t recv_gate;

    for (int g = 0; g < num_ranks; g++) {
        size_t idx = (size_t)rank * num_ranks + g;
        int delta = dstm_distance_get(state->topology, rank, g);
        if (delta == -1) {
            continue;
        }

        if (LOGGING)
            start = MPI_Wtime();

        status = dstm_darray_get_or_default(state->gates, idx,
                                            &recv_gate, t, 0,
                                            &empty_gate);
        
        if (LOGGING) {
            end = MPI_Wtime();
            sum += end - start;
        }

        if (status != DSTM_SUCCESS) {
            dstm_log_error("GET failed on gate %d", g);
            continue;
        }

        if (recv_gate.n_flights == 0)
            continue;

        dstm_log_normal("Rank %d [t=%ld]: gate %d has %d flight(s)\n",
                        rank, t, g, recv_gate.n_flights);

        /* STEP 3 — process flights */
        for (int f = 0; f < recv_gate.n_flights; f++) {
            flight_t flt = recv_gate.flights[f];

            if (flt.dest_rank == rank) {
                /* Final destination is on this rank — deliver to local airport */
                int dl = flt.dest_local;
                if (dl < 0 || dl >= na) {
                    dstm_log_error("Bad dest_local=%d on flight %lld", dl, flt.flight_id);
                    continue;
                }

                state->arrivals[dl]++;

                int og = airport_global(flt.origin_rank, flt.origin_local, na);
                dstm_log_normal("Rank %d [t=%ld]: LANDED flight %lld at airport "
                                "%d.%d — origin=%d.%d (%s) dep_t=%ld legs=%d \"%s\"\n",
                                rank, t, flt.flight_id,
                                rank, dl,
                                flt.origin_rank, flt.origin_local,
                                tier_name[state->tiers[og]],
                                flt.departure_time,
                                flt.legs_completed + 1,
                                flt.payload);
            } else {
                /* Connecting flight — find next hop RANK toward dest_rank.
                 * Attribute the connection to whichever local airport hashed
                 * to this gate (use gate-hash-style: any local airport on
                 * this rank acts as the connecting hub; pick local 0 for
                 * the counter to keep things simple, or distribute round-robin). */
                int local_hub = g % na;  /* spread connection credit */
                state->connections[local_hub]++;
                flt.legs_completed++;

                int next_hop = airport_find_next_hop(state, rank, flt.dest_rank);
                int delta = dstm_distance_get(state->topology, rank, next_hop);

                gate_slot_t *ob = &state->outbox[next_hop];
                if (ob->n_flights < GATE_CAPACITY) {
                    flt.next_hop_rank = next_hop;
                    ob->flights[ob->n_flights++] = flt;

                    if (next_hop == flt.dest_rank) {
                        dstm_log_normal("Rank %d [t=%ld]: CONNECTING flight %lld "
                                        "-> final dst rank=%d (airport %d.%d) "
                                        "delta=%d (leg %d)\n",
                                        rank, t, flt.flight_id,
                                        next_hop, flt.dest_rank, flt.dest_local,
                                        delta, flt.legs_completed);
                    } else {
                        dstm_log_normal("Rank %d [t=%ld]: CONNECTING flight %lld "
                                        "-> hop rank=%d toward dst=%d.%d delta=%d (leg %d)\n",
                                        rank, t, flt.flight_id,
                                        next_hop, flt.dest_rank, flt.dest_local,
                                        delta, flt.legs_completed);
                    }
                } else {
                    dstm_log_warning("Rank %d [t=%ld]: CONNECTION DELAYED "
                                     "flight %lld — outbox to rank %d full\n",
                                     rank, t, flt.flight_id, next_hop);
                }
            }
        }
    }
    
    if (LOGGING)
        dstm_log_normal("GET %f\n", sum);

    /* STEP 4 — each local airport independently decides to inject a flight */
    if (num_ranks > 1) {
        for (int l = 0; l < na; l++) {
            int og = airport_global(rank, l, na);
            airport_tier_t my_tier = state->tiers[og];

            if ((rand() % 100) >= tier_send_prob[my_tier])
                continue;

            int dest_rank, dest_local;
            airport_pick_dest(state, rank, &dest_rank, &dest_local);

            int next_hop = airport_find_next_hop(state, rank, dest_rank);
            int delta = dstm_distance_get(state->topology, rank, next_hop);

            gate_slot_t *ob = &state->outbox[next_hop];
            if (ob->n_flights >= GATE_CAPACITY) {
                dstm_log_warning("Rank %d [t=%ld]: DEPART DROPPED from %d.%d "
                                 "— outbox to rank %d full\n",
                                 rank, t, rank, l, next_hop);
                continue;
            }

            int dg = airport_global(dest_rank, dest_local, na);
            flight_t flt = {
                .valid          = 1,
                .origin_rank    = rank,
                .origin_local   = l,
                .dest_rank      = dest_rank,
                .dest_local     = dest_local,
                .next_hop_rank  = next_hop,
                .legs_completed = 0,
                .departure_time = t,
                .flight_id      = (long long)(rank * na + l) * 100000 + state->flight_counter++,
            };

            int is_direct = (next_hop == dest_rank);
            snprintf(flt.payload, sizeof(flt.payload),
                     "%s(%d.%d)->%s(%d.%d)%s t=%ld",
                     tier_name[my_tier], rank, l,
                     tier_name[state->tiers[dg]], dest_rank, dest_local,
                     is_direct ? "" : " [LAYOVER]",
                     t);
            ob->flights[ob->n_flights++] = flt;

            state->departures[l]++;

            volatile double sink = airport_do_work(state->work);
            (void)sink;

            if (is_direct) {
                dstm_log_normal("Rank %d [t=%ld]: DEPART flight %lld from %d.%d "
                                "dst=%d.%d (%s) DIRECT delta=%d\n",
                                rank, t, flt.flight_id, rank, l,
                                dest_rank, dest_local,
                                tier_name[state->tiers[dg]], delta);
            } else {
                dstm_log_normal("Rank %d [t=%ld]: DEPART flight %lld from %d.%d "
                                "dst=%d.%d via rank %d delta=%d\n",
                                rank, t, flt.flight_id, rank, l,
                                dest_rank, dest_local, next_hop, delta);
            }
        }
    }

    if (LOGGING)
        sum = 0.0;

    /* STEP 5 — send outbox (one PUT per remote rank that has flights) */
    for (int dest = 0; dest < num_ranks; dest++) {
        if (dest == rank)
            continue;
        if (state->outbox[dest].n_flights == 0)
            continue;

        size_t idx = (size_t)dest * num_ranks + rank;
        int delta = dstm_distance_get(state->topology, rank, dest);

        if (delta == -1)
            continue; 

        if (LOGGING)
            start = MPI_Wtime();

        status = dstm_darray_put(state->gates, idx,
                                 &state->outbox[dest], t, delta);
        
        if (LOGGING) {
            end = MPI_Wtime();
            sum += end - start;
        }

        if (status != DSTM_SUCCESS)
            dstm_log_error("PUT to rank %d gate failed", dest);
    }
    
    if (LOGGING)
        dstm_log_normal("PUT %f\n", sum);
}  
