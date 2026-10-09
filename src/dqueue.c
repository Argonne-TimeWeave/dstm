/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "config.h"

#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "dstm.h"

#include "internal/globals.h"
#include "internal/region.h"

/*******************************************************************************
 * Internal layout (single append-based queue per (rank, timestamp))
 *
 * Each rank owns ONE region holding ONE slot.  Every (rank, timestamp)
 * is a distinct version of that slot, managed by the region version list.
 * The slot is:
 *
 *     [ uint64_t byte_count | element_size * capacity bytes ]
 *
 * Producers APPEND into the slot via dstm_append: all producers targeting
 * the same (owner, timestamp) accumulate into the same version buffer in
 * MPI arrival order.  The owning rank is the concatenation point, so there
 * are no per-neighbor blocks and no wholesale-overwrite collisions.
 *
 *   push(dest, elem)      -> stage into the local outbox for dest
 *   flush(time, delta)    -> one dstm_append per destination with staged
 *                            data (delta >= 0; remote requires delta > 0)
 *   pop_incoming(t, 0)    -> one local dstm_get of this rank's slot, then
 *                            read byte_count/element_size elements
 *
 * CAPACITY SIZING RULE.  Because every producer appends into the SAME
 * slot, `capacity` must be sized for the SUM over all producers of their
 * maximum per-tick contribution, NOT one producer's worth.  push() and
 * flush() bound-check against this and return -DSTM_ENOMEM on overflow.
 *
 * ORDERING.  Concatenation order is MPI arrival order and is therefore
 * nondeterministic.  Elements must be order-independent, or carry their
 * own key and be sorted by the consumer after pop.  This dqueue does NOT
 * sort.
 *
 * VERSION EXISTENCE.  A consumer's pop reads one version of the slot.  To
 * guarantee that version exists even when no producer sent anything this
 * tick, flush() always performs a local self-append of ZERO elements to
 * the owner's own slot, which creates (and keeps WRITABLE until the phase
 * rule fires) the (self, time+delta) version.  Combined with the
 * early-tick fallback (effective time < 0 -> empty), pop never blocks on a
 * missing version.
 ******************************************************************************/

/* The slot's count field counts BYTES (what dstm_append maintains). */
typedef struct {
        uint64_t byte_count;
        /* payload[element_size * capacity] follows */
} dqueue_slot_hdr_t;

struct dstm_dqueue_s {
        int    region_id;        /* this rank's local region id */
        dstm_ptr_t slot;         /* ptr to the slot base (count field) */

        size_t element_size;
        size_t capacity;         /* max elements in the aggregate slot */
        size_t slot_bytes;       /* sizeof(hdr) + element_size*capacity */

        int world_size;
        int self;

        /* Per-rank region id of every owner's queue region, so we can
         * address a remote owner's slot (region ids are a per-rank
         * namespace, unrelated to MPI rank). */
        int *rank_region_id;     /* [world_size] */

        /* Per-rank byte offset of the slot inside that rank's region. */
        intptr_t *rank_slot_off; /* [world_size] */
};

/* ---------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */

static inline dstm_ptr_t slot_ptr_of(const dstm_dqueue_t *q, int rank)
{
        return dstm_ptr_make(q->rank_region_id[rank], q->rank_slot_off[rank]);
}

/* ---------------------------------------------------------------------------
 * Teardown
 * ------------------------------------------------------------------------- */

static void dqueue_free_all(dstm_dqueue_t *q)
{
        free(q->rank_region_id);
        free(q->rank_slot_off);
        free(q);
}

/* ---------------------------------------------------------------------------
 * Creation
 * ------------------------------------------------------------------------- */

int dstm_dqueue_create_on_region(dstm_dqueue_t **ret,
                                 size_t element_size,
                                 size_t capacity,
                                 dstm_distance_t *dist,
                                 int region_id)
{
        if (ret == NULL || element_size == 0 || capacity == 0 || dist == NULL)
                return -DSTM_EINVAL;
        *ret = NULL;

        int world_size = dstm_global_context.world_size;
        int self       = dstm_global_context.rank;
        MPI_Comm comm  = dstm_global_context.comm;

        if (dstm_distance_size(dist) != (size_t)world_size)
                return -DSTM_EINVAL;

        dstm_region_t *region = dstm_get_local_region(region_id);
        if (region == NULL)
                return -DSTM_EINVAL;

        dstm_dqueue_t *q = calloc(1, sizeof(*q));
        if (q == NULL)
                return -DSTM_ENOMEM;

        q->region_id    = region_id;
        q->element_size = element_size;
        q->capacity     = capacity;
        q->slot_bytes   = sizeof(dqueue_slot_hdr_t) +
                          element_size * capacity;
        q->world_size   = world_size;
        q->self         = self;

        q->rank_region_id = calloc((size_t)world_size, sizeof(int));
        q->rank_slot_off  = calloc((size_t)world_size, sizeof(intptr_t));
        if (!q->rank_region_id || !q->rank_slot_off) {
                dqueue_free_all(q);
                return -DSTM_ENOMEM;
        }

        /* Allocate this rank's single slot in its region. */
        dstm_ptr_t slot = dstm_region_malloc(region, q->slot_bytes);
        if (dstm_ptr_isnull(slot)) {
                dqueue_free_all(q);
                return -DSTM_ENOMEM;
        }
        q->slot = slot;

        /* Exchange every rank's region id and slot offset so remote
         * appends can address the right owner's slot.  slot.offset is an
         * intptr_t; exchange as int64_t for width independence. */
        {
                int rc = MPI_Allgather(&region_id, 1, MPI_INT,
                                       q->rank_region_id, 1, MPI_INT, comm);
                if (rc != MPI_SUCCESS) {
                        dstm_region_free(region, slot);
                        dqueue_free_all(q);
                        return -DSTM_FAILURE;
                }

                int64_t  mine = (int64_t)slot.offset;
                int64_t *offs = calloc((size_t)world_size, sizeof(int64_t));
                if (offs == NULL) {
                        dstm_region_free(region, slot);
                        dqueue_free_all(q);
                        return -DSTM_ENOMEM;
                }
                rc = MPI_Allgather(&mine, 1, MPI_INT64_T,
                                   offs, 1, MPI_INT64_T, comm);
                if (rc != MPI_SUCCESS) {
                        free(offs);
                        dstm_region_free(region, slot);
                        dqueue_free_all(q);
                        return -DSTM_FAILURE;
                }
                for (int r = 0; r < world_size; r++)
                        q->rank_slot_off[r] = (intptr_t)offs[r];
                free(offs);
        }

        *ret = q;
        return DSTM_SUCCESS;
}

int dstm_dqueue_create(dstm_dqueue_t **ret,
                       size_t element_size,
                       size_t capacity,
                       dstm_distance_t *dist)
{
        return dstm_dqueue_create_on_region(ret, element_size, capacity,
                                            dist, dstm_get_local_region_id());
}

void dstm_dqueue_destroy(dstm_dqueue_t **dqueue)
{
        if (dqueue == NULL || *dqueue == NULL)
                return;
        dstm_dqueue_t *q = *dqueue;
        dstm_region_t *region = dstm_get_local_region(q->region_id);
        if (region != NULL && !dstm_ptr_isnull(q->slot))
                dstm_region_free(region, q->slot);
        dqueue_free_all(q);
        *dqueue = NULL;
}

/* ---------------------------------------------------------------------------
 * Introspection
 * ------------------------------------------------------------------------- */

size_t dstm_dqueue_element_size(dstm_dqueue_t *q)
{
        return q ? q->element_size : 0;
}

size_t dstm_dqueue_capacity(dstm_dqueue_t *q)
{
        return q ? q->capacity : 0;
}

/* One queue, one slot: the whole queue holds at most `capacity`
 * elements, so max_incoming collapses to capacity. */
size_t dstm_dqueue_max_incoming(dstm_dqueue_t *q)
{
        return q ? q->capacity : 0;
}

/* ---------------------------------------------------------------------------
 * Push  (== append, issued immediately)
 * ------------------------------------------------------------------------- */

int dstm_dqueue_push(dstm_dqueue_t *q, int dest_rank, const void *element,
                     dstm_time_t time, dstm_time_t delta)
{
        if (q == NULL || element == NULL)
                return -DSTM_EINVAL;
        if (dest_rank < 0 || dest_rank >= q->world_size)
                return -DSTM_EINVAL;

        /* Up-front sanity guard: a single element must at least fit the
         * slot.  True aggregate overflow (sum over producers) is enforced
         * by the append handler on the owning rank. */
        if (q->capacity == 0)
                return -DSTM_ENOMEM;

        dstm_ptr_t dst = slot_ptr_of(q, dest_rank);
        return dstm_append(dest_rank, q->rank_region_id[dest_rank],
                           time, delta, dst, (void *)element,
                           q->element_size);
}

/* ---------------------------------------------------------------------------
 * Pop (== get) : single local get of the slot
 * ------------------------------------------------------------------------- */

int dstm_dqueue_pop_incoming(dstm_dqueue_t *q, void *out_elements,
                             size_t *io_count,
                             dstm_time_t time, dstm_time_t delta)
{
        if (q == NULL || io_count == NULL)
                return -DSTM_EINVAL;

        /* Early-tick fallback: version predates history -> empty. */
        if (time + delta < 0) {
                *io_count = 0;
                return DSTM_SUCCESS;
        }

        size_t cap = *io_count;    /* caller buffer capacity, in elements */

        /* One local get of this rank's slot at (time + delta).  The owner
         * always seeds this version in flush(), so it exists. */
        dqueue_slot_hdr_t *hdr = calloc(1, q->slot_bytes);
        if (hdr == NULL)
                return -DSTM_ENOMEM;

        dstm_ptr_t src = slot_ptr_of(q, q->self);
        int rc = dstm_get(q->self, q->region_id, time, delta,
                          hdr, src, q->slot_bytes);
        if (rc != DSTM_SUCCESS) {
                free(hdr);
                return rc;
        }

        uint64_t bytes = hdr->byte_count;
        size_t   count = (size_t)(bytes / q->element_size);
        if (count > q->capacity)
                count = q->capacity;   /* corrupt-slot guard */

        if (out_elements == NULL || count > cap) {
                *io_count = count;
                free(hdr);
                return (out_elements == NULL && cap == 0)
                               ? -DSTM_ENOMEM   /* count-only query */
                               : (count > cap ? -DSTM_ENOMEM : DSTM_SUCCESS);
        }

        memcpy(out_elements,
               (char *)hdr + sizeof(dqueue_slot_hdr_t),
               count * q->element_size);
        *io_count = count;
        free(hdr);
        return DSTM_SUCCESS;
}
