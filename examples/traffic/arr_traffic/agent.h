/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef TRAFFIC_AGENT_H
#define TRAFFIC_AGENT_H 1

#include <stddef.h>
#include <stdint.h>

#include "dstm.h"

/*
 * Vehicle / agent representation.
 *
 * A vehicle is a small POD struct.  It is stored locally on whichever
 * rank currently owns the intersection at which the vehicle is queued.
 * Agents migrate between ranks via DSTM "inbox" puts when they cross
 * a region boundary.
 */
struct vehicle {
	int32_t id;
	int32_t origin;
	int32_t destination;
	int32_t current_intersection;
	int32_t depart_tick;
	int32_t hops;
};

/* Event types in the local event queue. */
enum event_type {
	EV_DEPART = 1, /* vehicle is queued at intersection, picks next hop */
	EV_ARRIVE = 2, /* vehicle arrives at an intersection */
};

struct event {
	dstm_time_t time;
	enum event_type type;
	int32_t vehicle_idx; /* index into agent_pool */
	int32_t intersection;
};

/*
 * Min-heap event priority queue.  Implementation lives in agent.c.
 */
struct event_pq {
	struct event *data;
	size_t size;
	size_t capacity;
};

void pq_init(struct event_pq *pq);
void pq_destroy(struct event_pq *pq);
void pq_push(struct event_pq *pq, struct event ev);
int pq_pop(struct event_pq *pq, struct event *out);
int pq_peek(const struct event_pq *pq, struct event *out);
size_t pq_size(const struct event_pq *pq);

/*
 * Agent pool.  Vehicles are stored in a flat dynamic array; the index
 * into this array is what events refer to.  A vehicle slot is "free"
 * if its id is -1.
 */
struct agent_pool {
	struct vehicle *vehicles;
	size_t size;
	size_t capacity;
	size_t live_count;
	size_t arrived_count;
	size_t migrated_out_count;
	size_t migrated_in_count;
};

void pool_init(struct agent_pool *p);
void pool_destroy(struct agent_pool *p);
/* Insert a vehicle, returning its index. */
int pool_insert(struct agent_pool *p, struct vehicle v);
/* Remove a vehicle (mark slot free). */
void pool_remove(struct agent_pool *p, int idx);

#endif /* TRAFFIC_AGENT_H */
