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

void pick_event_location(zombie_state_t *st, event_slot_t slot,
                         int *out_node, int *out_local)
{
    (void)slot;

    int total_cells = st->num_ranks * st->nl;
    long total_weight = 0;
    for (int i = 0; i < total_cells; i++)
        total_weight += loc_tier_weight[st->loc_tiers[i]];

    long r = (long)rand() % (total_weight > 0 ? total_weight : 1);
    long acc = 0;
    for (int i = 0; i < total_cells; i++) {
        acc += loc_tier_weight[st->loc_tiers[i]];
        if (r < acc) {
            *out_node  = i / st->nl;
            *out_local = i % st->nl;
            return;
        }
    }
    *out_node  = 0;
    *out_local = 0;
}

static void make_event(zombie_state_t *st, event_slot_t slot, event_t *out)
{
    int start_min, end_max;
    if (slot == SLOT_MORNING) {
        start_min = MORNING_START_MIN; end_max = MORNING_END_MAX;
    } else {
        start_min = AFTERNOON_START_MIN; end_max = AFTERNOON_END_MAX;
    }
    int span_start = end_max - start_min;
    int start = start_min + (rand() % (span_start > 0 ? span_start : 1));
    int max_len = end_max - start;
    if (max_len < 1) max_len = 1;
    if (max_len > EVENT_MAX_HOURS) max_len = EVENT_MAX_HOURS;
    int len = 1 + (rand() % max_len);

    out->time_start = start;
    out->time_end   = start + len;
    pick_event_location(st, slot, &out->location_node, &out->location_local);
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
        make_event(st, SLOT_MORNING,   &h->morning_event);
        make_event(st, SLOT_AFTERNOON, &h->afternoon_event);
        snprintf(h->payload, PAYLOAD_MAX, "H%d@%d.%d",
                 h->human_id, h->home_node, h->home_local);
    }
}

void zombie_destroy(zombie_state_t *st)
{
    free(st->humans);
    free(st->outbox);
    free(st->carryover);
    free(st->departures);
    free(st->arrivals);
    free(st->infections);
}

/* ---------- per-human scheduling ---------- */

static void desired_cell(human_t *h, int hod,
                         int *want_node, int *want_local)
{
    if (h->state == HSTATE_ZOMBIE) {
        *want_node  = h->cur_node;
        *want_local = h->cur_local;
        return;
    }

    event_t *m = &h->morning_event;
    event_t *a = &h->afternoon_event;

    if (hod >= m->time_start && hod < m->time_end) {
        *want_node = m->location_node; *want_local = m->location_local;
    } else if (hod >= a->time_start && hod < a->time_end) {
        *want_node = a->location_node; *want_local = a->location_local;
    } else {
        *want_node = h->home_node; *want_local = h->home_local;
    }
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

/* ---------- main timestep (one HOUR) ----------
 *
 * Two streams of humans feed into each hour's roster:
 *   (A) inbound bundles from the darray — humans who traveled here
 *   (B) carryover from last hour — humans physically here who
 *       stayed put or just took a local hop. NEVER crosses MPI.
 *
 * The darray only ever carries inter-rank traffic. The self-bundle
 * is sentineled (empty) but never used to ship humans, so there's
 * no capacity contention there. */

void zombie_timestep(zombie_state_t *st, dstm_time_t t, int quiet)
{
    int rank      = st->rank;
    int num_ranks = st->num_ranks;
    int nl        = st->nl;
    int status;
    int hod       = hour_of_day((int)t);
    int dod       = day_of((int)t);

    if (!quiet)
        dstm_log_normal("Rank %d [t=%ld d=%d h=%02d]: === step ===\n",
                        rank, t, dod, hod);

    /* STEP 0 — day boundary: ticks + schedule re-roll (owner-side). */
    if (hod == 0) {
        for (int i = 0; i < st->nh; i++) {
            human_t *h = &st->humans[i];
            if (h->state == HSTATE_INFECTED && h->last_infect_day != dod) {
                h->days_infected++;
                h->last_infect_day = dod;
                if (h->days_infected >= INFECT_DAYS_TO_ZOMBIE) {
                    h->state = HSTATE_ZOMBIE;
                    dstm_log_normal("Rank %d [t=%ld]: human %d TURNED ZOMBIE "
                                    "at %d.%d\n",
                                    rank, t, h->human_id,
                                    h->cur_node, h->cur_local);
                }
            }
            if (h->state != HSTATE_ZOMBIE) {
                make_event(st, SLOT_MORNING,   &h->morning_event);
                make_event(st, SLOT_AFTERNOON, &h->afternoon_event);
            }
        }
        /* Also tick any infected human in carryover (they may be away
         * from home — their authoritative state lives in carryover). */
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
                make_event(st, SLOT_MORNING,   &h->morning_event);
                make_event(st, SLOT_AFTERNOON, &h->afternoon_event);
            }
        }
    }

    /* STEP 1 — sentinel PUT for our self-bundle. We never write real
     * humans into self; this just makes sure GETs don't block. */
    {
        size_t idx = bundle_index(rank, rank, num_ranks);
        dstm_darray_put(st->cells, idx, &empty_cell, t, 0);
    }

    /* STEP 2 — clear outbox. */
    for (int r = 0; r < num_ranks; r++)
        st->outbox[r] = empty_cell;

    /* STEP 3 — build roster from BOTH streams.
     *
     * Roster cells 0..nl-1 are local cells; nl is LIMBO (in-transit).
     * Each roster cell can hold an unbounded number of humans, so we
     * use heap-allocated growable buffers. */
    int LIMBO = nl;
    int n_buckets = nl + 1;

    human_t **bucket      = calloc(n_buckets, sizeof(human_t *));
    int      *bucket_n    = calloc(n_buckets, sizeof(int));
    int      *bucket_cap  = calloc(n_buckets, sizeof(int));
    int      *contagious  = calloc(n_buckets, sizeof(int));

    #define BUCKET_PUSH(b, hp) do {                                    \
        int _b = (b);                                                  \
        if (bucket_n[_b] >= bucket_cap[_b]) {                          \
            bucket_cap[_b] = bucket_cap[_b] ? bucket_cap[_b] * 2 : 16; \
            bucket[_b] = realloc(bucket[_b],                           \
                                 bucket_cap[_b] * sizeof(human_t));    \
        }                                                              \
        bucket[_b][bucket_n[_b]++] = *(hp);                            \
    } while (0)

    /* 3a — inbound bundles (humans who actually traveled to us). */
    for (int src = 0; src < num_ranks; src++) {
        if (src == rank) continue; /* self-bundle is always sentinel-only */

        size_t idx = bundle_index(rank, src, num_ranks);
        cell_slot_t recv;
        status = dstm_darray_get_or_default(st->cells, idx,
                                            &recv, t, 0, &empty_cell);
        if (status != DSTM_SUCCESS) {
            dstm_log_error("GET failed on bundle (dest=%d,src=%d)", rank, src);
            continue;
        }
        if (recv.n_humans == 0) continue;

        if (!quiet)
            dstm_log_normal("Rank %d [t=%ld]: bundle src=%d carries %d human(s)\n",
                            rank, t, src, recv.n_humans);

        for (int i = 0; i < recv.n_humans; i++) {
            human_t h = recv.humans[i];
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

    /* 3b — carryover (humans who were here last hour and stayed/moved
     * locally). This stream is authoritative for any human it contains;
     * the owner record may be stale for humans away from home. */
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
    st->n_carryover = 0; /* consumed; we'll rebuild it in STEP 5 */

    /* 3c — bootstrap: at t=0 the carryover is empty, so seed from
     * owner records. After t=0 every owned human is accounted for
     * either via 3a (came home from a trip) or via 3b (stayed). */
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

    /* STEP 4 — INFECTION per cell. */
    for (int l = 0; l < nl; l++) {
        if (bucket_n[l] == 0) continue;
        int cont = contagious[l];

        if (!quiet)
            dstm_log_normal("Rank %d [t=%ld]: cell %d.%d pop=%d contagious=%d\n",
                            rank, t, rank, l, bucket_n[l], cont);
        if (cont == 0) continue;

        for (int i = 0; i < bucket_n[l]; i++) {
            human_t *h = &bucket[l][i];
            if (h->state != HSTATE_HEALTHY) continue;

            int infected_now = 0;
            for (int k = 0; k < cont; k++) {
                if ((rand() % 100) < 30) { infected_now = 1; break; }
            }
            if (!infected_now) continue;

            h->state           = HSTATE_INFECTED;
            h->days_infected   = 0;
            h->last_infect_day = dod;
            st->infections[l]++;

            dstm_log_normal("Rank %d [t=%ld]: INFECTED human %d at %d.%d\n",
                            rank, t, h->human_id, rank, l);
        }
    }

    /* STEP 5 — ADVANCE & route every roster human.
     *
     * Local hops and stay-puts go into the carryover list (no MPI).
     * Inter-rank hops go into the outbox.
     * Owner records (st->humans[i]) are updated in lockstep for any
     * human we own so the final summary is accurate. */
    int total_present = 0;
    for (int b = 0; b < n_buckets; b++) total_present += bucket_n[b];

    if (!quiet)
        dstm_log_normal("Rank %d [t=%ld]: ROSTER total=%d (limbo=%d)\n",
                        rank, t, total_present, bucket_n[LIMBO]);

    for (int b = 0; b < n_buckets; b++) {
        for (int j = 0; j < bucket_n[b]; j++) {
            human_t *h = &bucket[b][j];

            int want_node, want_local;
            desired_cell(h, hod, &want_node, &want_local);
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
                /* Local: drop into carryover; NEVER fails. */
                carryover_push(st, h);
                if (!quiet)
                    dstm_log_normal("Rank %d [t=%ld]: STAY human %d at %d.%d "
                                    "(state=%d)\n",
                                    rank, t, h->human_id,
                                    rank, h->cur_local, h->state);
            } else {
                /* Inter-rank: outbox. Bounded by CELL_CAPACITY. */
                cell_slot_t *ob = &st->outbox[dest_rank];
                if (ob->n_humans >= CELL_CAPACITY) {
                    /* Don't drop — keep them here in carryover so they
                     * try again next hour. Roll back cur_node so they
                     * don't think they've already moved. */
                    h->cur_node = rank;
                    /* cur_local stays as it was (from_local on a stay,
                     * or want_local if dest_rank was want_node — but
                     * either way they're still on our rank). */
                    h->cur_local = from_local;
                    carryover_push(st, h);
                    dstm_log_warning("Rank %d [t=%ld]: outbox→%d FULL, "
                                     "human %d held over at %d.%d\n",
                                     rank, t, dest_rank,
                                     h->human_id, rank, from_local);
                } else {
                    ob->humans[ob->n_humans++] = *h;
                    if (b != LIMBO) st->departures[from_local]++;
                    if (!quiet)
                        dstm_log_normal("Rank %d [t=%ld]: SHIP human %d "
                                        "from %d.%d toward %d.%d via rank %d "
                                        "(state=%d)\n",
                                        rank, t, h->human_id,
                                        rank, from_local,
                                        want_node, want_local, dest_rank,
                                        h->state);
                }
            }

            /* Mirror to owner record so the final summary is accurate. */
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

    /* STEP 6 — PUT non-empty inter-rank bundles. */
    for (int dest = 0; dest < num_ranks; dest++) {
        if (dest == rank) continue;
        if (st->outbox[dest].n_humans == 0) continue;

        size_t idx = bundle_index(dest, rank, num_ranks);
        int delta = dstm_distance_get(st->topology, rank, dest);
        if (delta < 1) delta = 1;

        status = dstm_darray_put(st->cells, idx,
                                 &st->outbox[dest], t, delta);
        if (status != DSTM_SUCCESS)
            dstm_log_error("PUT to rank %d failed", dest);
    }

    #undef BUCKET_PUSH
}
