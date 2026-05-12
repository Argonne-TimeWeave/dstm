/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

/*
 * Traffic agent-based simulation example for DSTM.
 *
 * Demonstrates how to:
 *   - map an agent-based simulation onto a conservative discrete-event
 *     engine driven by DSTM heartbeats;
 *   - share routing-relevant global state (per-intersection congestion)
 *     through a dstm_darray with topology-aware lookahead;
 *   - migrate agents between ranks at region boundaries through per-pair
 *     "inbox" slots written with positive lookahead and read at the
 *     receiver's current time;
 *   - identify simulation virtual time with dstm logical time using a
 *     1:1 mapping (1 dstm_time_t tick == 1 simulation second).
 *
 * The city is a 16x16 grid + 2 cross-city highways + a 4x4 dense downtown
 * block.  It is partitioned across exactly 5 MPI ranks:
 *
 *   rank 0 = downtown (hot)
 *   ranks 1..4 = NW/NE/SW/SE quadrants (cold)
 *
 * Roughly 60% of trips terminate in downtown, producing a clearly visible
 * work imbalance favoring rank 0.
 *
 * Run with:  mpirun -n 5 ./traffic [--steps N] [--vehicles M] [--seed S]
 */

#include <assert.h>
#include <inttypes.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dstm.h"

#include "agent.h"
#include "city.h"

/*******************************************************************************
 * Constants and shared types
 ******************************************************************************/

/* Number of records per inbox cell.  Each cell is owned by a single
 * sender and read by a single receiver; multiple migrations from the
 * same (sender, receiver) targeting the same arrive_tick are packed
 * into the cell's records[] array.
 *
 * MAX_OUTBOX_LOOKAHEAD bounds how far in the future a sender can have
 * a pending cell waiting to be flushed.  arrive_tick - sim_time at the
 * moment of send_migration is at most max_distance plus any extra
 * travel time beyond the lookahead; we use a generous value. */
#define RECORDS_PER_CELL     16
#define MAX_OUTBOX_LOOKAHEAD 8

/* Per-intersection congestion record (lives in DSTM darray). */
struct cong_record {
	int32_t queue_len;
	int32_t last_update_tick;
};

/* Migration record (in inbox ring slots). */
struct migration_record {
	int32_t valid;
	int32_t agent_id;
	int32_t origin;
	int32_t destination;
	int32_t dst_intersection;
	int32_t depart_tick;
	int32_t arrive_tick;
	int32_t hops;
};

/*
 * Inbox layout
 * ============
 *
 * The "inbox" is a single dstm_darray whose element is an
 * inbox_cell containing RECORDS_PER_CELL migration records.  Each rank
 * owns EXPECTED_RANKS cells, one per possible sender:
 *
 *   index = receiver * EXPECTED_RANKS + sender
 *
 * Each cell is owned (write-side) by exactly one sender and read by
 * exactly one receiver.  Within a single simulation tick, a sender may
 * place several outbound migration records aimed at the same receiver
 * and the same arrive_tick into the cell's records[] array.  At the
 * end of the tick the sender flushes the populated cell into the
 * version that corresponds to arrive_tick using a single dstm_put.
 *
 * To support multiple distinct arrive_tick values pending at the same
 * time the sender keeps a local outbox indexed by (receiver,
 * arrive_tick - sim_time) so each future tick has its own pending
 * cell.  See struct outbox below.
 */
struct inbox_cell {
	struct migration_record records[RECORDS_PER_CELL];
};

/*******************************************************************************
 * Per-rank simulation context
 ******************************************************************************/

/*
 * Per-tick outbox: while processing events, send_migration accumulates
 * records into cells indexed by (receiver, arrive_offset) where
 * arrive_offset = arrive_tick - sim_time and is in [0, MAX_OUTBOX_LOOKAHEAD).
 * cell_count[r][off] is how many records are currently in
 * pending[r][off].records[].  Cells are flushed to DSTM at the end of
 * each tick by flush_outbox().
 */
struct outbox {
	struct inbox_cell pending[EXPECTED_RANKS][MAX_OUTBOX_LOOKAHEAD];
	int cell_count[EXPECTED_RANKS][MAX_OUTBOX_LOOKAHEAD];
};

struct sim {
	int rank;
	int world_size;

	struct city city;

	dstm_distance_t *dist;
	dstm_darray_t *cong;  /* NUM_INTERSECTIONS records, distributed */
	dstm_darray_t *inbox; /* EXPECTED_RANKS cells per rank: inbox[r,s] */

	struct event_pq pq;
	struct agent_pool pool;
	struct outbox ob;

	dstm_time_t sim_time;
	dstm_time_t sim_end;
	int seed;

	/* Stats */
	uint64_t events_processed;
	uint64_t routing_remote_reads;
	uint64_t inbox_overflows;
	uint64_t spawned;

	/* Local mirror of congestion for our owned intersections, kept
	 * consistent with what we publish to DSTM. */
	int32_t local_queue[NUM_INTERSECTIONS];
};

static struct sim S;

/*******************************************************************************
 * Helpers
 ******************************************************************************/

static int rank_owner_of(int intersection)
{
	return city_owner(&S.city, intersection);
}

static unsigned int xrand(unsigned int *state)
{
	unsigned int x = *state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*state = x ? x : 1u;
	return *state;
}

/* Inbox darray index for cell owned by (receiver r, sender s). */
static size_t inbox_index(int sender, int receiver)
{
	/* Each rank owns EXPECTED_RANKS consecutive cells. */
	return (size_t)receiver * EXPECTED_RANKS + (size_t)sender;
}

/*******************************************************************************
 * Congestion: read remote / publish local
 ******************************************************************************/

/* Publish current congestion of one of our owned intersections at sim_time.
 * Issues a fresh put on the version at time = sim_time so that any peers
 * reading us at (sim_time + d, delta = -d) for d = distance see the
 * latest queue length within this tick. */
static void publish_congestion(int intersection)
{
	assert(rank_owner_of(intersection) == S.rank);
	struct cong_record rec = {
	    .queue_len = S.local_queue[intersection],
	    .last_update_tick = (int32_t)S.sim_time,
	};
	int rc = dstm_darray_put(S.cong,
	                         (size_t)S.city.cong_index[intersection], &rec,
	                         S.sim_time, /*delta=*/0);
	assert(rc == DSTM_SUCCESS);
}

/* Read the congestion of an intersection (any rank) for routing decisions.
 * Reads use the safe "stale" delta = -distance(self, owner) so the GET
 * never blocks.  This is the standard conservative-PDES tradeoff: we may
 * see slightly out-of-date numbers, but no rollback is ever needed.
 *
 * For the early ticks where sim_time < distance the effective target is
 * negative; dstm_darray_get_or_default short-circuits and returns the
 * supplied "empty" record so we don't deadlock on non-existent history. */
static int read_congestion(int intersection)
{
	int owner = rank_owner_of(intersection);
	if (owner == S.rank)
		return S.local_queue[intersection];

	struct cong_record rec = {0, 0};
	struct cong_record empty = {0, -1};
	int d = dstm_distance_get(S.dist, S.rank, owner);
	int rc = dstm_darray_get_or_default(
	    S.cong, (size_t)S.city.cong_index[intersection], &rec,
	    S.sim_time, /*delta=*/-d, &empty);
	if (rc != DSTM_SUCCESS) {
		/* On error, fall back to neutral congestion 0. */
		return 0;
	}
	S.routing_remote_reads++;
	return rec.queue_len;
}

/* Publish the current queue length for every intersection we own at
 * sim_time with delta=0.  This serves three purposes:
 *
 *   1. Creates the local DSTM region version at time sim_time so that
 *      the subsequent self-region inbox drain (delta=0) will not block
 *      forever waiting for a non-existent version.
 *   2. Triggers DSTM's automatic heartbeats to all neighbors,
 *      advancing the distributed time horizon by one tick.
 *   3. Makes the per-tick congestion snapshot visible to peers reading
 *      with delta = -distance(self, peer).
 */
static void publish_owned_congestion(void)
{
	int n = S.city.rank_node_count[S.rank];
	const int *ids = S.city.rank_intersections[S.rank];
	for (int i = 0; i < n; i++) {
		int intersection = ids[i];
		struct cong_record rec = {
		    .queue_len = S.local_queue[intersection],
		    .last_update_tick = (int32_t)S.sim_time,
		};
		int rc = dstm_darray_put(
		    S.cong, (size_t)S.city.cong_index[intersection], &rec,
		    S.sim_time, /*delta=*/0);
		assert(rc == DSTM_SUCCESS);
	}
}

/*******************************************************************************
 * Migration: send / drain
 ******************************************************************************/

/* Queue a migration record for delivery to dst_rank at arrive_tick.
 * The record is held in the local outbox and flushed to DSTM at the
 * end of the current tick by flush_outbox().
 *
 * Caller must ensure arrive_tick - sim_time is in
 * [dstm_distance(self, dst_rank), MAX_OUTBOX_LOOKAHEAD).
 * pick_next_hop() enforces the lower bound; the upper bound is a
 * configuration limit of this example. */
static void send_migration(int dst_rank, struct migration_record m,
                           int arrive_tick)
{
	int travel = arrive_tick - (int)S.sim_time;
	int min_travel = dstm_distance_get(S.dist, S.rank, dst_rank);
	assert(travel >= min_travel);
	assert(travel >= 0 && travel < MAX_OUTBOX_LOOKAHEAD);

	struct inbox_cell *cell = &S.ob.pending[dst_rank][travel];
	int *cnt = &S.ob.cell_count[dst_rank][travel];
	if (*cnt >= RECORDS_PER_CELL) {
		/* Cell is full: this migration cannot be batched in this
		 * tick.  Increment overflow counter and drop the
		 * migration. */
		S.inbox_overflows++;
		return;
	}
	m.arrive_tick = arrive_tick;
	m.valid = 1;
	cell->records[*cnt] = m;
	(*cnt)++;
	S.pool.migrated_out_count++;
}

/* Flush the outbox at the end of a tick: for every (receiver, travel)
 * cell that has any records, write the entire cell into the inbox
 * darray with delta=travel so the cell lands in the receiver's region
 * version sim_time+travel.  Reset the cell counters afterwards. */
static void flush_outbox(void)
{
	for (int r = 0; r < EXPECTED_RANKS; r++) {
		if (r == S.rank)
			continue;
		for (int t = 0; t < MAX_OUTBOX_LOOKAHEAD; t++) {
			int cnt = S.ob.cell_count[r][t];
			if (cnt == 0)
				continue;
			/* Zero-fill the unused tail of the cell so the
			 * receiver sees clean valid=0 markers. */
			for (int k = cnt; k < RECORDS_PER_CELL; k++)
				memset(&S.ob.pending[r][t].records[k], 0,
				       sizeof(struct migration_record));
			int rc = dstm_darray_put(
			    S.inbox, inbox_index(S.rank, r),
			    &S.ob.pending[r][t], S.sim_time,
			    /*delta=*/(dstm_time_t)t);
			assert(rc == DSTM_SUCCESS);
			S.ob.cell_count[r][t] = 0;
		}
	}
}

/* Drain my inboxes from every sender at the current sim_time.
 *
 * The local self-region get with delta=0 requires that version sim_time
 * already exists locally; publish_owned_congestion() at the start of the
 * tick guarantees this.  We use get_or_default with an all-zero default
 * as a belt-and-braces against any future code path that drains before
 * publishing. */
static void drain_inbox(void)
{
	struct inbox_cell empty;
	memset(&empty, 0, sizeof(empty));
	for (int sender = 0; sender < S.world_size; sender++) {
		if (sender == S.rank)
			continue;
		struct inbox_cell cell;
		memset(&cell, 0, sizeof(cell));
		int rc = dstm_darray_get_or_default(
		    S.inbox, inbox_index(sender, S.rank), &cell,
		    S.sim_time, /*delta=*/0, &empty);
		if (rc != DSTM_SUCCESS)
			continue;

		for (int k = 0; k < RECORDS_PER_CELL; k++) {
			struct migration_record *m = &cell.records[k];
			if (!m->valid)
				continue;
			if (m->arrive_tick != (int32_t)S.sim_time)
				continue; /* not for this tick */

			struct vehicle v = {
			    .id = m->agent_id,
			    .origin = m->origin,
			    .destination = m->destination,
			    .current_intersection = m->dst_intersection,
			    .depart_tick = m->depart_tick,
			    .hops = m->hops,
			};
			int idx = pool_insert(&S.pool, v);
			S.pool.migrated_in_count++;
			struct event ev = {
			    .time = S.sim_time,
			    .type = EV_ARRIVE,
			    .vehicle_idx = idx,
			    .intersection = m->dst_intersection,
			};
			pq_push(&S.pq, ev);
		}
	}
}

/*******************************************************************************
 * Routing
 ******************************************************************************/

/* Greedy router: pick the neighbor that minimises
 *   manhattan_distance_to_destination + congestion_weight * queue_len.
 *
 * For cross-rank next hops, the returned travel time is bumped up to
 * the DSTM region distance between the current rank and the next-hop
 * owner, so that the ensuing migration's lookahead always satisfies
 * DSTM's conservative consistency rule.
 *
 * Returns -1 if there are no neighbors (dead end). */
static int pick_next_hop(int current, int destination, int *out_travel_time)
{
	const struct intersection *here = &S.city.nodes[current];
	if (here->neighbor_count == 0)
		return -1;

	int dst_row = S.city.nodes[destination].row;
	int dst_col = S.city.nodes[destination].col;

	int best = -1;
	int best_score = (1 << 30);
	int best_travel = 1;

	for (int k = 0; k < here->neighbor_count; k++) {
		int n = here->neighbors[k];
		int travel = here->travel_time[k];
		int nrow = S.city.nodes[n].row;
		int ncol = S.city.nodes[n].col;
		int dist = abs(nrow - dst_row) + abs(ncol - dst_col);
		int q = read_congestion(n);
		int score = dist * 4 + q; /* weight congestion vs distance */
		if (n == destination)
			score = -1; /* always prefer reaching destination */
		if (score < best_score) {
			best_score = score;
			best = n;
			best_travel = travel;
		}
	}

	/* Enforce travel >= dstm_distance(self, owner_of_next) for
	 * cross-rank hops so send_migration's lookahead invariant
	 * holds. */
	if (best >= 0) {
		int next_owner = rank_owner_of(best);
		if (next_owner != S.rank) {
			int min_travel =
			    dstm_distance_get(S.dist, S.rank, next_owner);
			if (best_travel < min_travel)
				best_travel = min_travel;
		}
	}

	if (out_travel_time)
		*out_travel_time = best_travel;
	return best;
}

/*******************************************************************************
 * Event handlers
 ******************************************************************************/

static void handle_arrive(struct event ev)
{
	struct vehicle *v = &S.pool.vehicles[ev.vehicle_idx];
	if (v->id < 0)
		return; /* freed */

	v->current_intersection = ev.intersection;
	v->hops++;

	if (v->current_intersection == v->destination) {
		S.pool.arrived_count++;
		pool_remove(&S.pool, ev.vehicle_idx);
		return;
	}

	/* Update local congestion (vehicle joins the queue) and publish. */
	if (rank_owner_of(ev.intersection) == S.rank) {
		S.local_queue[ev.intersection]++;
		publish_congestion(ev.intersection);
	}

	/* Schedule a departure from this intersection one tick later. */
	struct event dep = {
	    .time = S.sim_time + 1,
	    .type = EV_DEPART,
	    .vehicle_idx = ev.vehicle_idx,
	    .intersection = ev.intersection,
	};
	pq_push(&S.pq, dep);
}

static void handle_depart(struct event ev)
{
	struct vehicle *v = &S.pool.vehicles[ev.vehicle_idx];
	if (v->id < 0)
		return;
	if (v->current_intersection != ev.intersection)
		return; /* stale */

	int travel = 1;
	int next = pick_next_hop(ev.intersection, v->destination, &travel);
	if (next < 0)
		return; /* no exits */

	/* Vehicle leaves current intersection: decrement local queue. */
	if (rank_owner_of(ev.intersection) == S.rank &&
	    S.local_queue[ev.intersection] > 0) {
		S.local_queue[ev.intersection]--;
		publish_congestion(ev.intersection);
	}

	int next_owner = rank_owner_of(next);
	if (next_owner == S.rank) {
		/* Local hop: schedule ARRIVE at sim_time + travel. */
		struct event ar = {
		    .time = S.sim_time + travel,
		    .type = EV_ARRIVE,
		    .vehicle_idx = ev.vehicle_idx,
		    .intersection = next,
		};
		pq_push(&S.pq, ar);
	} else {
		/* Cross-rank migration: bundle vehicle as a migration
		 * record and free its local slot.  The receiver will
		 * recreate the vehicle on the other side. */
		struct migration_record m = {
		    .valid = 1,
		    .agent_id = v->id,
		    .origin = v->origin,
		    .destination = v->destination,
		    .dst_intersection = next,
		    .depart_tick = v->depart_tick,
		    .arrive_tick = (int32_t)S.sim_time + travel,
		    .hops = v->hops,
		};
		int min_travel = dstm_distance_get(S.dist, S.rank, next_owner);
		int arrive = (int)S.sim_time +
		             (travel > min_travel ? travel : min_travel);
		send_migration(next_owner, m, arrive);
		pool_remove(&S.pool, ev.vehicle_idx);
	}
}

/*******************************************************************************
 * Initial spawn
 ******************************************************************************/

static void spawn_initial_vehicles(int per_rank)
{
	unsigned int rng = (unsigned int)(S.seed * 7919u + S.rank * 31u + 1u);
	int next_id = S.rank * per_rank * 1000;

	for (int i = 0; i < per_rank; i++) {
		/* Pick an origin that is owned by this rank. */
		int origin;
		int tries = 0;
		do {
			origin = (int)(xrand(&rng) %
			               (unsigned int)NUM_INTERSECTIONS);
			tries++;
		} while (rank_owner_of(origin) != S.rank && tries < 256);
		if (rank_owner_of(origin) != S.rank)
			continue;

		int dest = city_pick_destination(&S.city, origin, &rng);

		struct vehicle v = {
		    .id = next_id++,
		    .origin = origin,
		    .destination = dest,
		    .current_intersection = origin,
		    .depart_tick = 0,
		    .hops = 0,
		};
		int idx = pool_insert(&S.pool, v);
		S.spawned++;

		/* Initial event: ARRIVE at origin at t=0 — this places the
		 * vehicle into the intersection queue and triggers a DEPART
		 * at t=1. */
		struct event ev = {
		    .time = 0,
		    .type = EV_ARRIVE,
		    .vehicle_idx = idx,
		    .intersection = origin,
		};
		pq_push(&S.pq, ev);
	}
}

/*******************************************************************************
 * Main loop
 ******************************************************************************/

static void run_step(void)
{
	/* 1. Publish: write our owned-intersection congestion at sim_time
	 *    with delta=0.  This guarantees the local DSTM region version
	 *    sim_time exists (so the upcoming self-region inbox drain
	 *    won't deadlock waiting for it) AND emits heartbeats to all
	 *    neighbors so the distributed time horizon advances. */
	publish_owned_congestion();

	/* 2. Drain my inboxes for migrations arriving at this tick. */
	drain_inbox();

	/* 3. Process all local events at <= sim_time.  Cross-rank
	 *    departures accumulate into the local outbox. */
	struct event ev;
	while (pq_peek(&S.pq, &ev) && ev.time <= S.sim_time) {
		pq_pop(&S.pq, &ev);
		S.events_processed++;
		switch (ev.type) {
		case EV_ARRIVE:
			handle_arrive(ev);
			break;
		case EV_DEPART:
			handle_depart(ev);
			break;
		}
	}

	/* 4. Flush all outbox cells that accumulated migrations this
	 *    tick.  One dstm_put per (receiver, future tick) cell. */
	flush_outbox();
}

/*******************************************************************************
 * Setup / teardown
 ******************************************************************************/

static int setup(int steps)
{
	MPI_Comm_rank(MPI_COMM_WORLD, &S.rank);
	MPI_Comm_size(MPI_COMM_WORLD, &S.world_size);

	if (S.world_size != EXPECTED_RANKS) {
		if (S.rank == 0)
			fprintf(stderr,
			        "ERROR: traffic example requires exactly %d "
			        "ranks (got %d)\n",
			        EXPECTED_RANKS, S.world_size);
		return -1;
	}

	city_build(&S.city);

	/* Build a DSTM distance matrix from the city's region adjacency. */
	int matrix[EXPECTED_RANKS * EXPECTED_RANKS];
	city_distance_matrix(&S.city, matrix);
	int rc = dstm_distance_create_from_matrix(&S.dist, EXPECTED_RANKS,
	                                          matrix);
	if (rc != DSTM_SUCCESS) {
		fprintf(stderr, "rank %d: dstm_distance_create_from_matrix "
		                "failed: %d\n",
		        S.rank, rc);
		return -1;
	}

	rc = dstm_start(S.dist);
	if (rc != DSTM_SUCCESS) {
		fprintf(stderr, "rank %d: dstm_start failed: %d\n", S.rank,
		        rc);
		return -1;
	}

	/* Allocate congestion darray.  Each rank declares it owns
	 * city.rank_node_count[rank] contiguous global indices; the
	 * intersection-id -> global-index translation is held in
	 * city.cong_index[] so that index N..N+count-1 of the darray
	 * coincides with this rank's owned intersections. */
	size_t mine = (size_t)S.city.rank_node_count[S.rank];
	rc = dstm_darray_create(&S.cong, sizeof(struct cong_record), mine);
	if (rc != DSTM_SUCCESS) {
		fprintf(stderr, "rank %d: dstm_darray_create(cong) failed\n",
		        S.rank);
		return -1;
	}

	/* Allocate inbox darray: each rank owns EXPECTED_RANKS cells,
	 * one per possible sender.  Each cell is a fixed-size array of
	 * RECORDS_PER_CELL migration records.  Each cell is owned by
	 * exactly one sender and read by exactly one receiver, so no
	 * concurrent writers can clobber it in a single version. */
	rc = dstm_darray_create(&S.inbox, sizeof(struct inbox_cell),
	                        EXPECTED_RANKS);
	if (rc != DSTM_SUCCESS) {
		fprintf(stderr, "rank %d: dstm_darray_create(inbox) failed\n",
		        S.rank);
		return -1;
	}

	pq_init(&S.pq);
	pool_init(&S.pool);
	memset(S.local_queue, 0, sizeof(S.local_queue));
	memset(&S.ob, 0, sizeof(S.ob));
	S.sim_time = 0;
	S.sim_end = steps;

	if (S.rank == 0)
		city_print_summary(&S.city);

	return 0;
}

/* Destroy DSTM-owned darrays.  MUST be called BEFORE dstm_fini, since
 * dstm_fini destroys the local region that the darrays' allocations
 * live in. */
static void teardown_dstm_objects(void)
{
	dstm_darray_destroy(&S.inbox);
	dstm_darray_destroy(&S.cong);
}

/* Free local-only state.  Safe to call any time after the simulation
 * loop terminates. */
static void teardown_local(void)
{
	pq_destroy(&S.pq);
	pool_destroy(&S.pool);
	dstm_distance_destroy(&S.dist);
}

/*******************************************************************************
 * Stats reporting
 ******************************************************************************/

struct rank_stats {
	uint64_t events_processed;
	uint64_t spawned;
	uint64_t arrived;
	uint64_t migrated_out;
	uint64_t migrated_in;
	uint64_t routing_remote_reads;
	uint64_t outbox_overflows;
	uint64_t live_count;
};

static void collect_and_print_stats(void)
{
	struct rank_stats mine = {
	    .events_processed = S.events_processed,
	    .spawned = S.spawned,
	    .arrived = (uint64_t)S.pool.arrived_count,
	    .migrated_out = (uint64_t)S.pool.migrated_out_count,
	    .migrated_in = (uint64_t)S.pool.migrated_in_count,
	    .routing_remote_reads = S.routing_remote_reads,
	    .outbox_overflows = S.inbox_overflows,
	    .live_count = (uint64_t)S.pool.live_count,
	};

	struct rank_stats *all = NULL;
	if (S.rank == 0) {
		all = malloc(sizeof(*all) * (size_t)S.world_size);
		assert(all != NULL);
	}
	MPI_Gather(&mine, sizeof(mine), MPI_BYTE, all, sizeof(mine), MPI_BYTE,
	           0, MPI_COMM_WORLD);
	if (S.rank != 0)
		return;

	const char *names[EXPECTED_RANKS] = {"downtown", "NW", "NE", "SW",
	                                     "SE"};
	uint64_t total_spawned = 0, total_arrived = 0, total_events = 0;
	uint64_t total_mout = 0, total_min = 0, total_overflow = 0;
	uint64_t total_live = 0;
	printf("\n=== Per-rank simulation stats ===\n");
	printf("%-10s %8s %8s %6s %8s %10s %8s %8s %9s\n", "rank", "spawned",
	       "arrived", "live", "events", "remote_rd", "mig_out", "mig_in",
	       "ob_overfl");
	for (int r = 0; r < S.world_size; r++) {
		printf("%-10s %8" PRIu64 " %8" PRIu64 " %6" PRIu64
		       " %8" PRIu64 " %10" PRIu64 " %8" PRIu64 " %8" PRIu64
		       " %9" PRIu64 "\n",
		       names[r], all[r].spawned, all[r].arrived,
		       all[r].live_count, all[r].events_processed,
		       all[r].routing_remote_reads, all[r].migrated_out,
		       all[r].migrated_in, all[r].outbox_overflows);
		total_spawned += all[r].spawned;
		total_arrived += all[r].arrived;
		total_events += all[r].events_processed;
		total_mout += all[r].migrated_out;
		total_min += all[r].migrated_in;
		total_overflow += all[r].outbox_overflows;
		total_live += all[r].live_count;
	}
	printf("--- totals ---\n");
	printf("spawned=%" PRIu64 " arrived=%" PRIu64 " live=%" PRIu64
	       " in_transit=%" PRIu64 " events=%" PRIu64
	       " migrations(sent=%" PRIu64 ",recv=%" PRIu64
	       ") outbox_overflows=%" PRIu64 "\n",
	       total_spawned, total_arrived, total_live,
	       total_spawned - total_arrived - total_live - total_overflow,
	       total_events, total_mout, total_min, total_overflow);
	printf("event imbalance: rank0(downtown) processed %.2fx avg; "
	       "remote_rd 0=%" PRIu64 " avg=%.0f\n",
	       (double)all[0].events_processed * (double)S.world_size /
	           (double)(total_events ? total_events : 1),
	       all[0].routing_remote_reads,
	       (double)(all[0].routing_remote_reads + all[1].routing_remote_reads +
	                all[2].routing_remote_reads + all[3].routing_remote_reads +
	                all[4].routing_remote_reads) /
	           (double)S.world_size);
	free(all);
}

/*******************************************************************************
 * Argument parsing
 ******************************************************************************/

static void usage(const char *prog)
{
	fprintf(stderr,
	        "usage: %s [--steps N] [--vehicles M] [--seed S] [--quiet]\n",
	        prog);
}

int main(int argc, char *argv[])
{
	int steps = 80;
	int vehicles_per_rank = 200;
	int seed = 42;
	int quiet = 0;

	MPI_Init(&argc, &argv);
	dstm_init(&argc, &argv);

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--steps") && i + 1 < argc)
			steps = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--vehicles") && i + 1 < argc)
			vehicles_per_rank = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
			seed = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--quiet"))
			quiet = 1;
		else if (!strcmp(argv[i], "--help") ||
		         !strcmp(argv[i], "-h")) {
			usage(argv[0]);
			MPI_Finalize();
			return 0;
		}
	}

	if (quiet)
		dstm_log_setlevel(DSTM_LOG_QUIET);
	else
		dstm_log_setlevel(DSTM_LOG_DEBUG);

	S.seed = seed;
	if (setup(steps) < 0) {
		dstm_finalize();
		MPI_Finalize();
		return 1;
	}

	spawn_initial_vehicles(vehicles_per_rank);

	for (S.sim_time = 0; S.sim_time <= S.sim_end; S.sim_time++)
		run_step();

	/* Compute the highest effective timestep we ever wrote so dstm_fini
	 * can correctly drain.  The largest is the last possible migration
	 * issued at sim_end with delta = max_distance. */
	int max_d = 0;
	for (int i = 0; i < EXPECTED_RANKS; i++) {
		int d = dstm_distance_get(S.dist, S.rank, i);
		if (d > max_d)
			max_d = d;
	}
	dstm_time_t last_time = (dstm_time_t)(S.sim_end + max_d + 1);

	/* Destroy DSTM-allocated darrays BEFORE dstm_fini, while the
	 * region they were allocated in still exists. */
	teardown_dstm_objects();

	dstm_fini(last_time);

	collect_and_print_stats();

	teardown_local();
	dstm_finalize();
	MPI_Finalize();
	return 0;
}
