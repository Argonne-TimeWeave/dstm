/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "agent.h"

/*******************************************************************************
 * Event priority queue (binary min-heap)
 ******************************************************************************/

void pq_init(struct event_pq *pq)
{
	pq->capacity = 64;
	pq->size = 0;
	pq->data = malloc(sizeof(struct event) * pq->capacity);
	assert(pq->data != NULL);
}

void pq_destroy(struct event_pq *pq)
{
	free(pq->data);
	pq->data = NULL;
	pq->size = 0;
	pq->capacity = 0;
}

static int ev_less(const struct event *a, const struct event *b)
{
	if (a->time != b->time)
		return a->time < b->time;
	/* Tie-break: ARRIVE before DEPART (so an arriving vehicle is
	 * registered at the intersection before any DEPART executing this
	 * tick examines congestion). */
	return a->type < b->type;
}

static void heap_swap(struct event *a, struct event *b)
{
	struct event tmp = *a;
	*a = *b;
	*b = tmp;
}

void pq_push(struct event_pq *pq, struct event ev)
{
	if (pq->size == pq->capacity) {
		pq->capacity *= 2;
		pq->data = realloc(pq->data,
		                   sizeof(struct event) * pq->capacity);
		assert(pq->data != NULL);
	}
	pq->data[pq->size] = ev;
	size_t i = pq->size++;
	while (i > 0) {
		size_t parent = (i - 1) / 2;
		if (ev_less(&pq->data[i], &pq->data[parent])) {
			heap_swap(&pq->data[i], &pq->data[parent]);
			i = parent;
		} else {
			break;
		}
	}
}

int pq_peek(const struct event_pq *pq, struct event *out)
{
	if (pq->size == 0)
		return 0;
	*out = pq->data[0];
	return 1;
}

int pq_pop(struct event_pq *pq, struct event *out)
{
	if (pq->size == 0)
		return 0;
	*out = pq->data[0];
	pq->data[0] = pq->data[--pq->size];
	size_t i = 0;
	for (;;) {
		size_t left = 2 * i + 1;
		size_t right = 2 * i + 2;
		size_t smallest = i;
		if (left < pq->size &&
		    ev_less(&pq->data[left], &pq->data[smallest]))
			smallest = left;
		if (right < pq->size &&
		    ev_less(&pq->data[right], &pq->data[smallest]))
			smallest = right;
		if (smallest == i)
			break;
		heap_swap(&pq->data[i], &pq->data[smallest]);
		i = smallest;
	}
	return 1;
}

size_t pq_size(const struct event_pq *pq)
{
	return pq->size;
}

/*******************************************************************************
 * Agent pool
 ******************************************************************************/

void pool_init(struct agent_pool *p)
{
	p->capacity = 256;
	p->size = 0;
	p->live_count = 0;
	p->arrived_count = 0;
	p->migrated_out_count = 0;
	p->migrated_in_count = 0;
	p->vehicles = malloc(sizeof(struct vehicle) * p->capacity);
	assert(p->vehicles != NULL);
}

void pool_destroy(struct agent_pool *p)
{
	free(p->vehicles);
	p->vehicles = NULL;
	p->size = 0;
	p->capacity = 0;
}

int pool_insert(struct agent_pool *p, struct vehicle v)
{
	/* Linear scan for a free slot first; fall back to append. */
	for (size_t i = 0; i < p->size; i++) {
		if (p->vehicles[i].id < 0) {
			p->vehicles[i] = v;
			p->live_count++;
			return (int)i;
		}
	}
	if (p->size == p->capacity) {
		p->capacity *= 2;
		p->vehicles = realloc(p->vehicles,
		                      sizeof(struct vehicle) * p->capacity);
		assert(p->vehicles != NULL);
	}
	int idx = (int)p->size;
	p->vehicles[idx] = v;
	p->size++;
	p->live_count++;
	return idx;
}

void pool_remove(struct agent_pool *p, int idx)
{
	assert(idx >= 0 && (size_t)idx < p->size);
	if (p->vehicles[idx].id < 0)
		return; /* already free */
	p->vehicles[idx].id = -1;
	p->live_count--;
}
