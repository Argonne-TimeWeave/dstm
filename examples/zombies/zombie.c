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

/* ---------- main timestep (one HOUR) ----------
 *
 * Phases:
 *   STEP 0  — owner-only: day-boundary clock ticks + schedule re-roll
 *   STEP 1  — sentinel PUT on our self-bundle
 *   STEP 2  — clear outbox
 *   STEP 3  — RECEIVE: read all (rank, src) bundles into a per-cell
 *             roster keyed by cur_local. Bootstrap with our owned
 *             humans whose cur_node == rank but who never showed up
 *             in any bundle (e.g. patient zero on t=0, or anyone
 *             staying home through a sentinel hour).
 *   STEP 4  — INFECTION: per local cell, contagious carriers may
 *             infect healthy occupants. Apply state to roster
 *             entries; mirror into owner records for our humans.
 *   STEP 5  — ADVANCE & SHIP: for each human present, compute
 *             desired cell, route one hop, drop in outbox.
 *   STEP 6  — PUT each non-empty outbox bundle.
 */

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

    /* STEP 0 */
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
    }

    /* STEP 1 — sentinel PUT for our (rank, rank) self-bundle. */
    {
        size_t idx = bundle_index(rank, rank, num_ranks);
        dstm_darray_put(st->cells, idx, &empty_cell, t, 0);
    }

    /* STEP 2 — clear outbox. */
    for (int r = 0; r < num_ranks; r++)
        st->outbox[r] = empty_cell;

    /* STEP 3 — RECEIVE & build roster.
     *
     * The roster is sized nl + 1: cells 0..nl-1 are local cells, and
     * cell nl is a "limbo" bucket for in-transit humans (cur_node !=
     * dest_node, or cur_node != rank). Limbo humans don't participate
     * in infection — they're just being forwarded. */
    int LIMBO = nl;
    cell_slot_t *roster = calloc(nl + 1, sizeof(cell_slot_t));
    int *contagious_per_cell = calloc(nl, sizeof(int));
    char *owned_seen = calloc(st->nh, sizeof(char));

    /* 3a — pull every bundle addressed to us. */
    for (int src = 0; src < num_ranks; src++) {
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

            /* Decide bucket: are they fully here, or just passing through? */
            int bucket;
            if (h.cur_node == rank && h.dest_node == rank &&
                h.cur_local >= 0 && h.cur_local < nl) {
                bucket = h.cur_local;
            } else {
                bucket = LIMBO;
            }

            cell_slot_t *bk = &roster[bucket];
            if (bk->n_humans >= CELL_CAPACITY) {
                dstm_log_warning("Rank %d [t=%ld]: roster bucket %d FULL, "
                                 "dropping human %d\n",
                                 rank, t, bucket, h.human_id);
                continue;
            }
            bk->humans[bk->n_humans++] = h;
            if (bucket != LIMBO) {
                st->arrivals[bucket]++;
                if (h.state == HSTATE_ZOMBIE || h.state == HSTATE_INFECTED)
                    contagious_per_cell[bucket]++;
            }
            if (h.home_node == rank) {
                int li = h.human_id - rank * st->nh;
                if (li >= 0 && li < st->nh) owned_seen[li] = 1;
            }
        }
    }

    /* 3b — bootstrap roster with owned humans who didn't appear in any
     * bundle. These are humans physically on our rank (cur_node==rank)
     * who weren't transmitted via the darray this hour — either because
     * it's t=0, or because they stayed put across an empty sentinel.
     *
     * This is the critical fix: previously, patient zero never entered
     * any roster, so was invisible to infection. */
    for (int i = 0; i < st->nh; i++) {
        if (owned_seen[i]) continue;
        human_t *h = &st->humans[i];
        if (!h->valid) continue;
        if (h->cur_node != rank) continue; /* somewhere else right now */

        int bucket;
        if (h->dest_node == rank &&
            h->cur_local >= 0 && h->cur_local < nl) {
            bucket = h->cur_local;
        } else {
            bucket = LIMBO;
        }

        cell_slot_t *bk = &roster[bucket];
        if (bk->n_humans >= CELL_CAPACITY) {
            dstm_log_warning("Rank %d [t=%ld]: roster bucket %d FULL on "
                             "bootstrap, dropping owned human %d\n",
                             rank, t, bucket, h->human_id);
            continue;
        }
        bk->humans[bk->n_humans++] = *h;
        if (bucket != LIMBO) {
            /* Don't count as arrival — they didn't travel. */
            if (h->state == HSTATE_ZOMBIE || h->state == HSTATE_INFECTED)
                contagious_per_cell[bucket]++;
        }
        owned_seen[i] = 1;
    }

    /* STEP 4 — INFECTION per cell. */
    for (int l = 0; l < nl; l++) {
        cell_slot_t *bk = &roster[l];
        if (bk->n_humans == 0) continue;
        int cont = contagious_per_cell[l];

        if (!quiet)
            dstm_log_normal("Rank %d [t=%ld]: cell %d.%d pop=%d contagious=%d\n",
                            rank, t, rank, l, bk->n_humans, cont);
        if (cont == 0) continue;

        for (int i = 0; i < bk->n_humans; i++) {
            human_t *h = &bk->humans[i];
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

            if (h->home_node == rank) {
                int li = h->human_id - rank * st->nh;
                if (li >= 0 && li < st->nh)
                    st->humans[li] = *h;
            }
        }
    }

    /* STEP 5 — ADVANCE & SHIP every roster human (incl. limbo). */
    for (int b = 0; b <= LIMBO; b++) {
        cell_slot_t *bk = &roster[b];
        for (int j = 0; j < bk->n_humans; j++) {
            human_t *h = &bk->humans[j];

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

            cell_slot_t *ob = &st->outbox[dest_rank];
            if (ob->n_humans >= CELL_CAPACITY) {
                dstm_log_warning("Rank %d [t=%ld]: TRAVEL DROPPED human %d "
                                 "— outbox to rank %d full\n",
                                 rank, t, h->human_id, dest_rank);
            } else {
                ob->humans[ob->n_humans++] = *h;
                if (dest_rank != rank && b != LIMBO) {
                    st->departures[from_local]++;
                    if (!quiet)
                        dstm_log_normal("Rank %d [t=%ld]: SHIP human %d "
                                        "from %d.%d toward %d.%d via rank %d "
                                        "(state=%d)\n",
                                        rank, t, h->human_id,
                                        rank, from_local,
                                        want_node, want_local, dest_rank,
                                        h->state);
                } else if (dest_rank == rank && !quiet) {
                    dstm_log_normal("Rank %d [t=%ld]: STAY human %d at %d.%d "
                                    "(state=%d)\n",
                                    rank, t, h->human_id,
                                    rank, h->cur_local, h->state);
                }
            }

            /* Mirror back to owner record. */
            if (h->home_node == rank) {
                int li = h->human_id - rank * st->nh;
                if (li >= 0 && li < st->nh)
                    st->humans[li] = *h;
            }
        }
    }

    volatile double sink = zombie_do_work(st->work);
    (void)sink;

    free(roster);
    free(contagious_per_cell);
    free(owned_seen);

    /* STEP 6 — PUT non-empty bundles. */
    for (int dest = 0; dest < num_ranks; dest++) {
        if (st->outbox[dest].n_humans == 0) continue;

        size_t idx = bundle_index(dest, rank, num_ranks);
        int delta;
        if (dest == rank) {
            delta = 1;
        } else {
            delta = dstm_distance_get(st->topology, rank, dest);
            if (delta < 1) delta = 1;
        }

        status = dstm_darray_put(st->cells, idx,
                                 &st->outbox[dest], t, delta);
        if (status != DSTM_SUCCESS)
            dstm_log_error("PUT to rank %d failed", dest);
    }
}
