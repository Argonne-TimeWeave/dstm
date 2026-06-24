#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <dstm.h>

#include "zombie.h"

const char *loc_tier_name[]   = { "POPULAR", "NORMAL", "QUIET" };
const int   loc_tier_weight[] = {       60,       30,      10 };

cell_slot_t empty_cell = { .n_humans = 0 };

#define MAX_TRAVEL_HOURS   2
#define EVENT_MIN_HOURS    2

static void pending_bundle_push(pending_bundle_queue_t *q,
                                const cell_slot_t *bundle, int src)
{
    if (q->n == q->cap) {
        int new_cap = q->cap ? q->cap * 2 : 64;
        q->bundles = realloc(q->bundles, sizeof(pending_bundle_t) * new_cap);
        q->cap = new_cap;
    }
    q->bundles[q->n].bundle      = *bundle;
    q->bundles[q->n].src         = src;
    q->bundles[q->n].arrive_time = bundle->arrive_time;
    q->n++;
}

/* Remove element at index i by swapping last element down. */
static void pending_bundle_remove_at(pending_bundle_queue_t *q, int i)
{
    q->bundles[i] = q->bundles[q->n - 1];
    q->n--;
}

/* ---------- generic helpers ---------- */

double zombie_do_work(int work)
{
    double s = 0.0;
    for (int i = 0; i < work; i++) s += sin((double)i);
    return s;
}

int human_find_next_hop(zombie_state_t *st, int from_rank, int to_rank)
{
    if (from_rank == to_rank) return to_rank;

    int cols = st->cols;
    int fr = from_rank / cols, fc = from_rank % cols;
    int tr = to_rank   / cols, tc = to_rank   % cols;

    if (fc != tc) {
        int nc = fc + ((tc > fc) ? 1 : -1);
        return fr * cols + nc;
    } else {
        int nr = fr + ((tr > fr) ? 1 : -1);
        return nr * cols + fc;
    }
}

/* Pick an event cell that the human can actually reach from home_node
 * within max_travel_hours. Weighted by tier popularity, restricted to
 * the reachable set. Falls back to (home_node, home_local) if nothing
 * qualifies (shouldn't happen unless max_travel_hours == 0 and the
 * home cell is somehow filtered out). */
void pick_event_location(zombie_state_t *st, event_slot_t slot,
                         int home_node, int home_local,
                         int max_travel_hours,
                         int *out_node, int *out_local)
{
    (void)slot;

    int total_cells = st->num_ranks * st->nl;

    /* First pass: total weight across reachable cells only. */
    long total_weight = 0;
    for (int i = 0; i < total_cells; i++) {
        int node = i / st->nl;
        int cost = (node == home_node)
                    ? 0
                    : dstm_distance_get(st->topology, home_node, node);
        if (cost <= max_travel_hours)
            total_weight += loc_tier_weight[st->loc_tiers[i]];
    }

    if (total_weight <= 0) {
        /* No reachable cell — stay home. */
        *out_node  = home_node;
        *out_local = home_local;
        return;
    }

    long r = (long)rand() % total_weight;
    long acc = 0;
    for (int i = 0; i < total_cells; i++) {
        int node = i / st->nl;
        int cost = (node == home_node)
                    ? 0
                    : dstm_distance_get(st->topology, home_node, node);
        if (cost > max_travel_hours) continue;

        acc += loc_tier_weight[st->loc_tiers[i]];
        if (r < acc) {
            *out_node  = node;
            *out_local = i % st->nl;
            return;
        }
    }

    /* Shouldn't reach here, but be safe. */
    *out_node  = home_node;
    *out_local = home_local;
}

static void make_event(zombie_state_t *st, event_slot_t slot,
                       int home_node, int home_local,
                       event_t *out)
{
    int start_min, end_max;
    if (slot == SLOT_MORNING) {
        start_min = MORNING_START_MIN; end_max = MORNING_END_MAX;
    } else {
        start_min = AFTERNOON_START_MIN; end_max = AFTERNOON_END_MAX;
    }

    /* Pick a start such that an event of at least EVENT_MIN_HOURS fits
     * inside the slot window. */
    int latest_start = end_max - EVENT_MIN_HOURS;
    if (latest_start < start_min) latest_start = start_min;
    int span_start = latest_start - start_min + 1;
    int start = start_min + (rand() % span_start);

    /* Length: clamp into [EVENT_MIN_HOURS, EVENT_MAX_HOURS] and into
     * what's left in the slot window. */
    int max_len = end_max - start;
    if (max_len > EVENT_MAX_HOURS) max_len = EVENT_MAX_HOURS;
    if (max_len < EVENT_MIN_HOURS) max_len = EVENT_MIN_HOURS;
    int len_span = max_len - EVENT_MIN_HOURS + 1;
    int len = EVENT_MIN_HOURS + (rand() % len_span);

    out->time_start = start;
    out->time_end   = start + len;

    /* Budget: humans should arrive before the event ENDS, ideally before
     * it starts. We require travel <= MAX_TRAVEL_HOURS, which is also
     * <= EVENT_MIN_HOURS, so even a late-arriving human gets on-site time. */
    pick_event_location(st, slot, home_node, home_local,
                        MAX_TRAVEL_HOURS,
                        &out->location_node, &out->location_local);
}

/* ---------- lifecycle ---------- */

void zombie_init(zombie_state_t *st,
                 int rank, int num_ranks, int nl, int nh,
                 int work, int seed,
                 loc_tier_t *loc_tiers,
                 dstm_distance_t *topology,
                 dstm_darray_t   *cells)
{
    st->rank      = rank;
    st->num_ranks = num_ranks;
    st->nl        = nl;
    st->nh        = nh;
    st->work      = work;
    st->loc_tiers = loc_tiers;
    st->topology  = topology;
    st->cells     = cells;

    int cols = num_ranks;
    for (int r = 1; r * r <= num_ranks; r++)
        if (num_ranks % r == 0) cols = num_ranks / r;
    st->cols = cols;
    st->rows = num_ranks / cols;

    st->humans     = calloc(nh,        sizeof(human_t));
    st->outbox     = calloc(num_ranks, sizeof(cell_slot_t));
    st->departures = calloc(nl,        sizeof(int));
    st->arrivals   = calloc(nl,        sizeof(int));
    st->infections = calloc(nl,        sizeof(int));

    /* Carryover sized for the worst case: every human in the entire
     * simulation could end up resident on this rank. */
    st->carryover_cap = num_ranks * nh;
    st->carryover     = calloc(st->carryover_cap, sizeof(human_t));
    st->n_carryover   = 0;

    srand((unsigned)seed);

    for (int i = 0; i < nh; i++) {
        human_t *h = &st->humans[i];
        h->valid           = 1;
        h->human_id        = rank * nh + i;
        h->state           = HSTATE_HEALTHY;

        {
            unsigned hash = (unsigned)(h->human_id * 2654435761u) ^ (unsigned)seed;
            h->risk = ((hash % 100) < HIGH_RISK_FRACTION_PCT)
                ? HRISK_HIGH : HRISK_LOW;
        }           

        h->days_infected   = 0;
        h->last_infect_day = -1;
        h->home_node       = rank;
        h->home_local      = rand() % nl;
        h->cur_node        = h->home_node;
        h->cur_local       = h->home_local;
        h->dest_node       = h->home_node;
        h->dest_local      = h->home_local;
        h->next_hop_rank   = rank;
        h->legs_completed  = 0;
        make_event(st, SLOT_MORNING, h->home_node, h->home_local, &h->morning_event);
        make_event(st, SLOT_AFTERNOON, h->home_node, h->home_local, &h->afternoon_event);
        snprintf(h->payload, PAYLOAD_MAX, "H%d@%d.%d",
                 h->human_id, h->home_node, h->home_local);
    }

    st->last_population = calloc(nl, sizeof(int));
    st->last_zombie_count = calloc(nl, sizeof(int));
}

void zombie_destroy(zombie_state_t *st)
{
    free(st->humans);
    free(st->outbox);
    free(st->carryover);
    free(st->departures);
    free(st->arrivals);
    free(st->infections);
    free(st->last_population);
    free(st->last_zombie_count);
}

/* ---------- per-human scheduling ---------- */

static void desired_cell(zombie_state_t *st, human_t *h, int hod,
                         int *want_node, int *want_local)
{
    if (h->state == HSTATE_ZOMBIE) {
        /* Try hunting first: find the local cell with the largest crowd. */
        int best_local = -1;
        int best_pop   = 0;

        for (int l = 0; l < st->nl; l++) {
            if (st->last_population[l] > best_pop) {
                best_pop   = st->last_population[l];
                best_local = l;
            }
        }

        if (best_local >= 0) {
            /* Found prey — chase. */
            *want_node  = st->rank;
            *want_local = best_local;
            return;
        }

        /* No prey nearby — wander to a POPULAR cell on any rank. */
        int at_target = (h->cur_node == h->dest_node &&
                         h->cur_local == h->dest_local);
        if (at_target || h->dest_node < 0) {
            int total_cells = st->num_ranks * st->nl;
            int popular[total_cells];
            int n_popular = 0;
            for (int i = 0; i < total_cells; i++)
                if (st->loc_tiers[i] == LOC_POPULAR)
                    popular[n_popular++] = i;

            if (n_popular > 0) {
                int pick = popular[rand() % n_popular];
                *want_node  = pick / st->nl;
                *want_local = pick % st->nl;
            } else {
                *want_node  = h->cur_node;
                *want_local = h->cur_local;
            }
        } else {
            *want_node  = h->dest_node;
            *want_local = h->dest_local;
        }
        return;
    }
     
    /* --- HUMAN LOGIC (healthy or infected) --- */
    event_t *m = &h->morning_event;
    event_t *a = &h->afternoon_event;

    int base_node, base_local;
    if (hod >= m->time_start && hod < m->time_end) {
        base_node = m->location_node; base_local = m->location_local;
    } else if (hod >= a->time_start && hod < a->time_end) {
        base_node = a->location_node; base_local = a->location_local;
    } else {
        base_node = h->home_node; base_local = h->home_local;
    }

    /* ZOMBIE AVOIDANCE: If currently on this rank, check for nearby zombies */
    if (h->cur_node == st->rank) {
        int dominated_by_zombie = 0;
        
        /* Check if destination cell has zombies */
        if (base_node == st->rank && st->last_zombie_count[base_local] > 0) {
            dominated_by_zombie = 1;
        }
        
        /* Also check current cell — if zombies here, flee immediately */
        if (h->cur_local >= 0 && h->cur_local < st->nl &&
            st->last_zombie_count[h->cur_local] > 0) {
            dominated_by_zombie = 1;
        }
        
        if (dominated_by_zombie) {
            /* Find safest local cell: no zombies, lowest population */
            int best_safe = -1;
            int best_safe_pop = INT_MAX;
            
            for (int l = 0; l < st->nl; l++) {
                if (st->last_zombie_count[l] == 0) {
                    if (st->last_population[l] < best_safe_pop) {
                        best_safe_pop = st->last_population[l];
                        best_safe = l;
                    }
                }
            }
            
            if (best_safe >= 0) {
                *want_node  = st->rank;
                *want_local = best_safe;
                return;
            }
            
            /* All local cells have zombies — flee to another rank */
            int total_cells = st->num_ranks * st->nl;
            int quiet_remote[total_cells];
            int n_quiet = 0;
            for (int i = 0; i < total_cells; i++) {
                int node = i / st->nl;
                if (node != st->rank && st->loc_tiers[i] == LOC_QUIET) {
                    quiet_remote[n_quiet++] = i;
                }
            }
            if (n_quiet > 0) {
                int pick = quiet_remote[rand() % n_quiet];
                *want_node  = pick / st->nl;
                *want_local = pick % st->nl;
                return;
            }
        }
    }

    *want_node = base_node;
    *want_local = base_local;
}

/* Push a human into the local carryover list, growing as needed. */
static void carryover_push(zombie_state_t *st, const human_t *h)
{
    if (st->n_carryover >= st->carryover_cap) {
        st->carryover_cap *= 2;
        st->carryover = realloc(st->carryover,
                                st->carryover_cap * sizeof(human_t));
    }
    st->carryover[st->n_carryover++] = *h;
}

#define ZOMBIE_MPI_TAG_BASE 1000

/* ---------- main timestep (one HOUR) ----------
 *
 * MPI port of zombie_timestep. Mirrors the airport MPI baseline's
 * structure:
 *   - Alltoall header exchange to discover nonzero senders
 *   - Sparse Isend/Irecv for actual payloads
 *   - Per-rank `pending_bundles` queue holds delta-delayed inbound
 *     bundles until their arrive_time hits
 *   - saved_inbox / saved_recv_counts drain last step's received
 *     bundles into pending at the top of each step
 *
 * Self-bundle is never sent (DSTM had a sentinel PUT; here we just
 * skip rank==self in both send and recv loops).
 */

void zombie_timestep(zombie_state_t *st, dstm_time_t t)
{
    int rank      = st->rank;
    int num_ranks = st->num_ranks;
    int nl        = st->nl;
    int hod       = hour_of_day((int)t);
    int dod       = day_of((int)t);

    dstm_log_normal("Rank %d [t=%ld d=%d h=%02d]: === step ===\n",
                    rank, t, dod, hod);

    /* ---- Per-rank persistent state across timesteps ---- */
    static pending_bundle_queue_t pending = { NULL, 0, 0 };
    static cell_slot_t *saved_inbox       = NULL;
    static int         *saved_recv_counts = NULL;

    /* Hoisted scratch buffers for the exchange (allocated once). */
    static int         *send_counts = NULL;
    static int         *recv_counts = NULL;
    static cell_slot_t *inbox       = NULL;
    static MPI_Request *recv_reqs   = NULL;
    static MPI_Request *send_reqs   = NULL;

    if (saved_inbox == NULL) {
        saved_inbox       = calloc(num_ranks, sizeof(cell_slot_t));
        saved_recv_counts = calloc(num_ranks, sizeof(int));
        send_counts       = malloc(sizeof(int)          * num_ranks);
        recv_counts       = malloc(sizeof(int)          * num_ranks);
        inbox             = malloc(sizeof(cell_slot_t)  * num_ranks);
        recv_reqs         = malloc(sizeof(MPI_Request)  * num_ranks);
        send_reqs         = malloc(sizeof(MPI_Request)  * num_ranks);
    }

    /* ---- Drain last step's saved_inbox into the pending queue ----
     * Each saved bundle was sent at some t_send with delta, so it is
     * scheduled to land at t_send + delta. The sender doesn't tell us
     * arrive_time over the wire; we recompute it as t (current step,
     * which is when the previous step's exchange completed) plus the
     * sender's delta. To preserve DSTM's semantics exactly we instead
     * stamp arrive_time = t_send + delta on the SENDER side and ship
     * it as part of the cell_slot_t header. We assume cell_slot_t has
     * an `arrive_time` field added for that purpose (one int64). */
    for (int src = 0; src < num_ranks; src++) {
        if (src == rank) continue;
        if (saved_recv_counts[src] == 0) continue;
        pending_bundle_push(&pending, &saved_inbox[src], src);
        saved_inbox[src]       = empty_cell;
        saved_recv_counts[src] = 0;
    }

    /* STEP 0 — day boundary: ticks + schedule re-roll (owner-side).
     * Identical to DSTM version. */
    if (hod == 0) {
        /* (block over st->humans[] left commented out, as in original) */
        for (int i = 0; i < st->n_carryover; i++) {
            human_t *h = &st->carryover[i];
            if (h->state == HSTATE_INFECTED && h->last_infect_day != dod) {
                h->days_infected++;
                h->last_infect_day = dod;
                if (h->days_infected >= INFECT_DAYS_TO_ZOMBIE) {
                    h->state = HSTATE_ZOMBIE;
                    dstm_log_normal("Rank %d [t=%ld]: human %d TURNED ZOMBIE "
                                    "at %d.%d (away from home)\n",
                                    rank, t, h->human_id,
                                    h->cur_node, h->cur_local);
                }
            }
            if (h->state != HSTATE_ZOMBIE) {
                make_event(st, SLOT_MORNING,   h->home_node, h->home_local,
                           &h->morning_event);
                make_event(st, SLOT_AFTERNOON, h->home_node, h->home_local,
                           &h->afternoon_event);
            }
        }
    }

    /* STEP 2 — clear outbox. (STEP 1 sentinel-PUT is unneeded in MPI.) */
    for (int r = 0; r < num_ranks; r++)
        st->outbox[r] = empty_cell;

    /* STEP 3 — build roster from BOTH streams.
     *
     * Roster cells 0..nl-1 are local cells; nl is LIMBO (in-transit). */
    int LIMBO     = nl;
    int n_buckets = nl + 1;

    human_t **bucket     = calloc(n_buckets, sizeof(human_t *));
    int      *bucket_n   = calloc(n_buckets, sizeof(int));
    int      *bucket_cap = calloc(n_buckets, sizeof(int));
    int      *contagious = calloc(n_buckets, sizeof(int));

    #define BUCKET_PUSH(b, hp) do {                                    \
        int _b = (b);                                                  \
        if (bucket_n[_b] >= bucket_cap[_b]) {                          \
            bucket_cap[_b] = bucket_cap[_b] ? bucket_cap[_b] * 2 : 16; \
            bucket[_b] = realloc(bucket[_b],                           \
                                 bucket_cap[_b] * sizeof(human_t));    \
        }                                                              \
        bucket[_b][bucket_n[_b]++] = *(hp);                            \
    } while (0)

    /* 3a — inbound bundles from the pending queue whose arrive_time
     * matches the current step. Replaces the DSTM darray GET loop. */
    {
        int i = 0;
        while (i < pending.n) {
            pending_bundle_t *pb = &pending.bundles[i];
            if (pb->arrive_time != t) { i++; continue; }

            cell_slot_t recv = pb->bundle;
            int          src = pb->src;
            pending_bundle_remove_at(&pending, i);  /* don't increment i */

            if (recv.n_humans == 0) continue;

            dstm_log_normal("Rank %d [t=%ld]: bundle src=%d carries %d "
                            "human(s)\n",
                            rank, t, src, recv.n_humans);

            for (int j = 0; j < recv.n_humans; j++) {
                human_t h = recv.humans[j];

                /* Tick infection for humans who were in transit during
                 * the day boundary (missed STEP 0). Same as DSTM. */
                if (h.state == HSTATE_INFECTED && h.last_infect_day != dod) {
                    h.days_infected++;
                    h.last_infect_day = dod;
                    if (h.days_infected >= INFECT_DAYS_TO_ZOMBIE) {
                        h.state = HSTATE_ZOMBIE;
                        dstm_log_normal("Rank %d [t=%ld]: human %d TURNED "
                                        "ZOMBIE on arrival at %d.%d "
                                        "(was in transit)\n",
                                        rank, t, h.human_id,
                                        h.cur_node, h.cur_local);
                    }
                }

                int b;
                if (h.cur_node == rank && h.dest_node == rank &&
                    h.cur_local >= 0 && h.cur_local < nl) {
                    b = h.cur_local;
                    st->arrivals[b]++;
                    if (h.state == HSTATE_ZOMBIE || h.state == HSTATE_INFECTED)
                        contagious[b]++;
                } else {
                    b = LIMBO;
                }
                BUCKET_PUSH(b, &h);
            }
        }
    }

    /* 3b — carryover (humans who were here last hour). Same as DSTM. */
    for (int i = 0; i < st->n_carryover; i++) {
        human_t h = st->carryover[i];
        int b;
        if (h.cur_node == rank && h.dest_node == rank &&
            h.cur_local >= 0 && h.cur_local < nl) {
            b = h.cur_local;
            if (h.state == HSTATE_ZOMBIE || h.state == HSTATE_INFECTED)
                contagious[b]++;
        } else {
            b = LIMBO;
        }
        BUCKET_PUSH(b, &h);
    }
    st->n_carryover = 0;

    /* 3c — bootstrap at t=0. Same as DSTM. */
    if (t == 0) {
        for (int i = 0; i < st->nh; i++) {
            human_t *h = &st->humans[i];
            if (!h->valid) continue;
            int b;
            if (h->cur_node == rank && h->dest_node == rank &&
                h->cur_local >= 0 && h->cur_local < nl) {
                b = h->cur_local;
                if (h->state == HSTATE_ZOMBIE || h->state == HSTATE_INFECTED)
                    contagious[b]++;
            } else {
                b = LIMBO;
            }
            BUCKET_PUSH(b, h);
        }
    }

    /* STEP 4 — INFECTION per cell. (Pure local CPU; unchanged.) */
    for (int l = 0; l < nl; l++) {
        if (bucket_n[l] == 0) continue;

        int same_cont = contagious[l];

        int near_cont = 0;
        for (int ll = 0; ll < nl; ll++) {
            if (ll == l) continue;
            int d = ll - l;
            if (d < 0) d = -d;
            if (d <= HIGH_RISK_RADIUS)
                near_cont += contagious[ll];
        }

        dstm_log_normal("Rank %d [t=%ld]: cell %d.%d pop=%d "
                        "same_cont=%d near_cont=%d\n",
                        rank, t, rank, l, bucket_n[l],
                        same_cont, near_cont);

        if (same_cont == 0 && near_cont == 0) continue;

        for (int i = 0; i < bucket_n[l]; i++) {
            human_t *h = &bucket[l][i];
            if (h->state != HSTATE_HEALTHY) continue;

            int infected_now = 0;

            for (int k = 0; k < same_cont; k++) {
                if ((rand() % 100) < INFECT_PROB_SAME_PCT) {
                    infected_now = 1;
                    break;
                }
            }

            if (!infected_now && h->risk == HRISK_HIGH && near_cont > 0) {
                for (int k = 0; k < near_cont; k++) {
                    if ((rand() % 100) < INFECT_PROB_NEAR_PCT) {
                        infected_now = 1;
                        break;
                    }
                }
            }

            if (!infected_now) continue;

            h->state           = HSTATE_INFECTED;
            h->days_infected   = 0;
            h->last_infect_day = dod;
            st->infections[l]++;

            dstm_log_normal("Rank %d [t=%ld]: INFECTED human %d at %d.%d "
                            "(%s risk, %s exposure)\n",
                            rank, t, h->human_id, rank, l,
                            h->risk == HRISK_HIGH ? "HIGH" : "low",
                            same_cont > 0 ? "same-cell" : "nearby-cell");
        }
    }

    for (int l = 0; l < nl; l++) {
        st->last_population[l] = bucket_n[l];

        int zcount = 0;
        for (int j = 0; j < bucket_n[l]; j++) {
            if (bucket[l][j].state == HSTATE_ZOMBIE)
                zcount++;
        }
        st->last_zombie_count[l] = zcount;
    }

    /* STEP 5 — ADVANCE & route every roster human. (Pure local;
     * unchanged from DSTM version.) */
    int total_present = 0;
    for (int b = 0; b < n_buckets; b++) total_present += bucket_n[b];

    dstm_log_normal("Rank %d [t=%ld]: ROSTER total=%d (limbo=%d)\n",
                    rank, t, total_present, bucket_n[LIMBO]);

    for (int b = 0; b < n_buckets; b++) {
        for (int j = 0; j < bucket_n[b]; j++) {
            human_t *h = &bucket[b][j];

            int want_node, want_local;
            desired_cell(st, h, hod, &want_node, &want_local);
            h->dest_node  = want_node;
            h->dest_local = want_local;

            int from_local = h->cur_local;
            int dest_rank;

            if (h->cur_node == want_node) {
                dest_rank    = rank;
                h->cur_local = want_local;
            } else {
                dest_rank = human_find_next_hop(st, rank, want_node);
                h->cur_node = dest_rank;
                if (dest_rank == want_node)
                    h->cur_local = want_local;
            }
            h->next_hop_rank = dest_rank;
            h->legs_completed++;

            if (dest_rank == rank) {
                carryover_push(st, h);
            } else {
                cell_slot_t *ob = &st->outbox[dest_rank];
                if (ob->n_humans >= CELL_CAPACITY) {
                    h->cur_node  = rank;
                    h->cur_local = from_local;
                    carryover_push(st, h);
                    dstm_log_warning("Rank %d [t=%ld]: outbox→%d FULL, "
                                     "human %d held over at %d.%d\n",
                                     rank, t, dest_rank,
                                     h->human_id, rank, from_local);
                } else {
                    ob->humans[ob->n_humans++] = *h;
                    if (b != LIMBO) st->departures[from_local]++;
                    dstm_log_normal("Rank %d [t=%ld]: SHIP human %d "
                                    "from %d.%d toward %d.%d via rank %d "
                                    "(state=%d)\n",
                                    rank, t, h->human_id,
                                    rank, from_local,
                                    want_node, want_local, dest_rank,
                                    h->state);
                }
            }

            if (h->home_node == rank) {
                int li = h->human_id - rank * st->nh;
                if (li >= 0 && li < st->nh)
                    st->humans[li] = *h;
            }
        }
    }

    volatile double sink = zombie_do_work(st->work);
    (void)sink;

    /* Cleanup roster. */
    for (int b = 0; b < n_buckets; b++) free(bucket[b]);
    free(bucket);
    free(bucket_n);
    free(bucket_cap);
    free(contagious);

    /* ====================================================================
     * STEP 6 — sparsity-aware two-phase exchange. Same pattern as the
     * airport MPI baseline:
     *   Phase A: MPI_Alltoall on one int per peer (header = n_humans)
     *   Phase B: Isend/Irecv full cell_slot_t only where header > 0
     *
     * Sender stamps arrive_time = t + delta into the cell_slot_t header
     * so receiver can park it in `pending` until that step arrives.
     * ==================================================================== */

    /* Stamp arrive_time on each non-empty outbox bundle. */
    for (int dest = 0; dest < num_ranks; dest++) {
        if (dest == rank) continue;
        if (st->outbox[dest].n_humans == 0) continue;
        int delta = dstm_distance_get(st->topology, rank, dest);
        if (delta < 1) delta = 1;
        st->outbox[dest].arrive_time = t + delta;
    }

    /* Phase A: header exchange. */
    for (int r = 0; r < num_ranks; r++) {
        send_counts[r] = st->outbox[r].n_humans;
        recv_counts[r] = 0;
    }
    MPI_Alltoall(send_counts, 1, MPI_INT,
                 recv_counts, 1, MPI_INT,
                 MPI_COMM_WORLD);

    /* Phase B: sparse payload exchange. */
    for (int r = 0; r < num_ranks; r++)
        inbox[r] = empty_cell;

    int n_recv = 0, n_send = 0;
    for (int src = 0; src < num_ranks; src++) {
        if (src == rank) continue;
        if (recv_counts[src] == 0) continue;
        MPI_Irecv(&inbox[src], (int)sizeof(cell_slot_t), MPI_BYTE,
                  src, ZOMBIE_MPI_TAG_BASE + (int)t,
                  MPI_COMM_WORLD, &recv_reqs[n_recv++]);
    }
    for (int dest = 0; dest < num_ranks; dest++) {
        if (dest == rank) continue;
        if (send_counts[dest] == 0) continue;
        MPI_Isend(&st->outbox[dest], (int)sizeof(cell_slot_t), MPI_BYTE,
                  dest, ZOMBIE_MPI_TAG_BASE + (int)t,
                  MPI_COMM_WORLD, &send_reqs[n_send++]);
    }

    if (n_recv > 0) MPI_Waitall(n_recv, recv_reqs, MPI_STATUSES_IGNORE);
    if (n_send > 0) MPI_Waitall(n_send, send_reqs, MPI_STATUSES_IGNORE);

    /* Stash for next step's drain. inbox already carries arrive_time
     * inside each cell_slot_t, so saved_inbox + saved_recv_counts is
     * sufficient — we don't need a separate arrive_time array. */
    memcpy(saved_inbox, inbox, sizeof(cell_slot_t) * num_ranks);
    memcpy(saved_recv_counts, recv_counts, sizeof(int) * num_ranks);

    #undef BUCKET_PUSH
}
