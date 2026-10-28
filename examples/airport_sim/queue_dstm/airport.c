#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "airport.h"

/* ------------------------------------------------------------------ */
/* RNG / FAIRNESS                                                       */
/*                                                                     */
/* Identical to simple_dstm and graph_dstm: srand(seed+rank) is issued  */
/* in main() (not here), and every plane consumes exactly one rand()    */
/* used as `rand() & 3` for its direction. The consumption order is     */
/* airport ascending, then plane ascending, one draw per plane, so all  */
/* three backends fly the identical set of flights for a given --seed.  */
/* Same-rank planes still consume a draw (as in both other backends),   */
/* keeping the generator position aligned across substrates.            */
/* ------------------------------------------------------------------ */

/* Torus neighbor in direction dir (0=N 1=S 2=E 3=W). Byte-for-byte the
 * same as simple_dstm's neighbor_of(). */
int airport_neighbor_of(int gid, int grid, int dir)
{
        int row = gid / grid;
        int col = gid % grid;
        switch (dir) {
        case 0: row = (row - 1 + grid) % grid; break;
        case 1: row = (row + 1) % grid;        break;
        case 2: col = (col + 1) % grid;        break;
        case 3: col = (col - 1 + grid) % grid; break;
        }
        return row * grid + col;
}

void airport_init(airport_state_t *state,
                  int rank,
                  int num_ranks,
                  int na,
                  int grid,
                  int planes_per_airport,
                  dstm_distance_t *topology,
                  dstm_dqueue_t *queue)
{
        memset(state, 0, sizeof(*state));
        state->rank      = rank;
        state->num_ranks = num_ranks;
        state->na        = na;
        state->grid      = grid;
        state->topology  = topology;
        state->queue     = queue;

        state->landings             = calloc((size_t)na, sizeof(long long));
        state->planes_on_the_ground = calloc((size_t)na, sizeof(int));
        state->pending_landings     = calloc((size_t)na, sizeof(int));
        for (int l = 0; l < na; l++)
                state->planes_on_the_ground[l] = planes_per_airport;

        /* Size the pop scratch buffer once, up front: the whole incoming
         * queue is ONE slot holding at most capacity elements. */
        state->inbox_cap = dstm_dqueue_max_incoming(queue);
        state->inbox     = calloc(state->inbox_cap ? state->inbox_cap : 1,
                                  sizeof(plane_msg_t));
}

void airport_destroy(airport_state_t *state)
{
        if (!state)
                return;
        free(state->landings);
        free(state->planes_on_the_ground);
        free(state->pending_landings);
        free(state->inbox);
        state->landings             = NULL;
        state->planes_on_the_ground = NULL;
        state->pending_landings     = NULL;
        state->inbox                = NULL;
}

/* Deposit a plane bound for a LOCAL airport into its pending-landing queue.
 * Mirrors simple_dstm's "arrivals go into pending_landings, not the ground". */
static inline void deliver_local(airport_state_t *s, int dst_global)
{
        int local = dst_global - s->rank * s->na;
        if (local < 0 || local >= s->na) {
                s->misroutes++;
                return;
        }
        s->pending_landings[local]++;
}

void airport_timestep(airport_state_t *state, dstm_time_t t)
{
        int rank = state->rank;
        int na   = state->na;
        int grid = state->grid;
        int rc;

        /* --- STEP 0: seed our own next version -------------------------
         * dqueue has no flush; versions exist only where an append landed.
         * If no neighbor sends us anything at tick t+1, our slot's version
         * t+1 would never be created and our get at (t+1,0) would block
         * forever. It would also leave our originating clock from stalling,
         * preventing neighbors' versions from ever becoming readable.
         *
         * So every tick each rank self-pushes ONE sentinel into its OWN
         * slot at (t, +1): it guarantees version t+1 exists, advances this
         * rank's clock past t for every neighbor, and is dropped on drain.
         * This is the queue analogue of simple_dstm's t=0 priming loop and
         * graph_dstm's airport_prime, done incrementally. */
        {
                plane_msg_t seed = {
                        .flight_id  = -1,
                        .dst_global = PLANE_SENTINEL,
                        .src_global = (int32_t)(rank * na),
                };
                rc = dstm_dqueue_push(state->queue, rank, &seed, t, 1);
                if (rc != DSTM_SUCCESS)
                        dstm_log_error("seed self-push t=%lld: %d",
                                       (long long)t, rc);
        }

        /* --- STEP 1: drain arrivals from the queue ------------------- */
        if (t >= 1) {
                size_t count = state->inbox_cap;
                rc = dstm_dqueue_pop_incoming(state->queue, state->inbox,
                                              &count, t, 0);
                if (rc != DSTM_SUCCESS) {
                        dstm_log_error("pop_incoming t=%lld failed: %d",
                                       (long long)t, rc);
                        count = 0;
                }
                for (size_t k = 0; k < count; k++) {
                        if (state->inbox[k].dst_global == PLANE_SENTINEL)
                                continue;                 /* seed, not a plane */
                        if (airport_owner(state->inbox[k].dst_global, na)
                            != rank) {
                                state->misroutes++;
                                continue;
                        }
                        deliver_local(state, state->inbox[k].dst_global);
                        state->arrivals++;
                }
        }

        /* --- STEP 1b: drain the per-airport runway queue -----------------
         * Each airport lands at most LANDINGS_PER_STEP planes per tick.
         * Identical to the other two backends. */
        for (int l = 0; l < na; l++) {
                int budget   = LANDINGS_PER_STEP;
                int can_land = state->pending_landings[l] < budget
                             ? state->pending_landings[l] : budget;
                if (can_land == 0)
                        continue;
                state->pending_landings[l]     -= can_land;
                state->planes_on_the_ground[l] += can_land;
                state->landings[l]             += can_land;
        }

        /* --- STEP 2 & 3 fused: depart every grounded plane ---------------
         * Same-rank destinations are delivered locally in place (never enter
         * the queue). Cross-rank destinations are pushed immediately to the
         * owning rank at (t, +1) so they become visible at tick t+1 -- the
         * future push the dqueue requires for remote traffic.
         *
         * Unlike darray/dgraph we do not stage an outbox: dstm_dqueue_push
         * is itself an append issued immediately, so one push per cross-rank
         * plane is the natural form here. */
        for (int l = 0; l < na; l++) {
                int planes_here = state->planes_on_the_ground[l];
                if (planes_here == 0)
                        continue;

                int my_gid = rank * na + l;

                for (int p = 0; p < planes_here; p++) {
                        /* FAIRNESS: exactly one rand() per plane, matching
                         * simple_dstm/graph_dstm's stream position. */
                        int dir       = rand() & 3;
                        int dst_gid   = airport_neighbor_of(my_gid, grid, dir);
                        int dst_rank  = airport_owner(dst_gid, na);

                        if (dst_rank == rank) {
                                /* Stays on this rank: deliver in-process. */
                                deliver_local(state, dst_gid);
                                state->planes_on_the_ground[l]--;
                                state->departures++;
                                continue;
                        }

                        plane_msg_t msg = {
                                .flight_id  = (int64_t)my_gid * 1000000LL
                                            + state->flight_counter++,
                                .dst_global = (int32_t)dst_gid,
                                .src_global = (int32_t)my_gid,
                        };

                        /* Future push: remote delta must be > 0. */
                        rc = dstm_dqueue_push(state->queue, dst_rank, &msg,
                                              t, 1);
                        if (rc == -DSTM_ENOMEM) {
                                /* Queue full: this plane and every plane
                                 * behind it on this airport stay on the
                                 * ground and retry next tick. Mirrors the
                                 * gate-full `break` in the other backends.
                                 * The flight_counter draw already happened,
                                 * but the plane did not depart, so roll the
                                 * ground/departure accounting back out by
                                 * simply not decrementing. */
                                state->overflows += planes_here - p;
                                break;
                        }
                        if (rc != DSTM_SUCCESS) {
                                dstm_log_error("push to rank %d t=%lld: %d",
                                               dst_rank, (long long)t, rc);
                                continue;
                        }

                        state->planes_on_the_ground[l]--;
                        state->departures++;
                }
        }
}

long long airport_drain_inflight(airport_state_t *state, dstm_time_t t_last)
{
        int rank = state->rank;
        int na   = state->na;
        int rc;

        size_t count = state->inbox_cap;
        rc = dstm_dqueue_pop_incoming(state->queue, state->inbox,
                                      &count, t_last, 0);
        if (rc != DSTM_SUCCESS) {
                dstm_log_error("final drain t=%lld failed: %d",
                               (long long)t_last, rc);
                return 0;
        }

        for (size_t k = 0; k < count; k++) {
                if (state->inbox[k].dst_global == PLANE_SENTINEL)
                        continue;                         /* seed, not a plane */
                if (airport_owner(state->inbox[k].dst_global, na) != rank) {
                        state->misroutes++;
                        continue;
                }
                deliver_local(state, state->inbox[k].dst_global);
                state->arrivals++;
        }

        return (long long)count;
}
