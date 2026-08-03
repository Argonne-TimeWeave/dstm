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
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "dstm.h"
#include "dstm/dgraph.h"

#include "internal/globals.h"
#include "internal/region.h"

/*******************************************************************************
 * Internal layout (static topology)
 *
 * One dstm_region_t partitioned into three fixed-size slabs at create time:
 *
 *   [ vertex_state_slab | edge_row_ptr_slab | edge_slot_slab ]
 *
 * vertex_state_slab
 *   local_vertices entries of vertex_state_size bytes, indexed by
 *   local id (global_id - local_vertex_offset). Empty if size == 0.
 *   Versioned; mutated by dstm_dgraph_vertex_put.
 *
 * edge_row_ptr_slab
 *   (local_vertices + 1) uint64_t CSR row pointers. Entry i is the
 *   first outgoing edge slot for local vertex i; entry local_vertices
 *   is a sentinel == local_edge_count. Outgoing edges of vertex i
 *   occupy slots [row_ptr[i], row_ptr[i+1]).
 *
 *   Written once at time 0 and never mutated (static topology). Kept
 *   inside the region so remote row pointers are readable via the
 *   same dstm_get path as everything else.
 *
 * edge_slot_slab
 *   local_edge_count packed entries of edge_slot_stride bytes:
 *   (dst_vertex_id, user_state[edge_state_size]). Within each
 *   vertex's window, slots are sorted by ascending dst, enabling
 *   binary search in dstm_dgraph_edge_{get,put}. dst is immutable
 *   after construction; only user_state is mutated by edge_put.
 *
 * All three slabs share one region, so vertex and edge state use the
 * same (time, delta) versioning. Row pointers and dst fields are
 * effectively immutable but cost only one snapshot at t=0.
 *
 * Slab base offsets are cached in vertex_state_off / edge_row_ptr_off
 * / edge_slot_off so dstm_ptr_t values can be built by addition.
 ******************************************************************************/

/* Fixed-width type for CSR row pointers and destination ids in the
 * region, so on-region layout is independent of host size_t width. */
typedef uint64_t dgraph_vid_t;

struct dstm_dgraph_s {
        /* Backing storage */
        dstm_region_t *region;
        int            region_id;

        /* Element sizes */
        size_t vertex_state_size;
        size_t edge_state_size;
        size_t edge_slot_stride;

        /* Global partitioning */
        int      world_size;
        int      local_rank;
        size_t   total_vertices;
        size_t   local_vertices;
        size_t   local_vertex_offset;
        size_t  *rank_vertex_offsets;   /* [world_size + 1] prefix sum */

        /* Edge counts */
        size_t total_edges;
        size_t local_edge_count;

        /* Host-cached CSR row pointers, length local_vertices + 1. */
        dgraph_vid_t *row_ptrs;

        /* Slab byte offsets within the local region. */
        intptr_t vertex_state_off;
        intptr_t edge_row_ptr_off;
        intptr_t edge_slot_off;

        /* Slab byte sizes. */
        size_t vertex_state_bytes;
        size_t edge_row_ptr_bytes;
        size_t edge_slot_bytes;
        size_t region_bytes;

        /* Per-rank layout, length world_size. Needed so we can build
         * dstm_ptr_t values into remote ranks' slabs. */
        size_t   *rank_local_vertices;
        size_t   *rank_local_edge_count;
        intptr_t *rank_vertex_state_off;
        intptr_t *rank_edge_row_ptr_off;
        intptr_t *rank_edge_slot_off;
        size_t   *rank_edge_slot_stride;

        /* Fully-replicated topology cache: on every rank, the sorted
         * dst arrays for every rank's local edges. Indexed as
         * rank_edge_dsts[r][slot] where slot is in [0, rank_local_edge_count[r]).
         * Topology is immutable, so this is built once at create time
         * from local data + MPI_Allgatherv, avoiding the need to do
         * DSTM gets on topology after t=0 ages out. */
        dgraph_vid_t **rank_edge_dsts;

        /* Fully-replicated row pointer cache. Indexed as
         * rank_row_ptrs[r][i] for i in [0, rank_local_vertices[r]]. */
        dgraph_vid_t **rank_row_ptrs;
};

/* ---------------------------------------------------------------------------
 * Layout helpers
 * ------------------------------------------------------------------------- */

/* Alignment used for slab boundaries and per-slot padding. 8 bytes is
 * sufficient for dgraph_vid_t (uint64_t) and any typical user state. */
#define DGRAPH_ALIGN ((size_t)8)

/* Align x up to the next multiple of a (a must be a power of 2). */
static inline size_t align_up(size_t x, size_t a)
{
        return (x + (a - 1)) & ~(a - 1);
}

/* Compute the padded stride of a single edge slot:
 *   [ dgraph_vid_t dst | user_state[edge_state_size] | pad ]
 * padded up to DGRAPH_ALIGN so the next slot's dst stays aligned. */
static inline size_t compute_edge_slot_stride(size_t edge_state_size)
{
        size_t raw = sizeof(dgraph_vid_t) + edge_state_size;
        return align_up(raw, DGRAPH_ALIGN);
}

/* Build a dstm_ptr_t pointing into `region_id` at byte offset `off`. */
static inline dstm_ptr_t make_ptr(int region_id, intptr_t off)
{
        dstm_ptr_t p;
        p.region = region_id;
        p.offset = off;
        return p;
}

/* ---------------------------------------------------------------------------
 * Ownership resolution
 * ------------------------------------------------------------------------- */

/* Binary search: return rank r such that
 *   rank_vertex_offsets[r] <= vertex_id < rank_vertex_offsets[r+1]. */
static int dgraph_find_rank(dstm_dgraph_t *g, size_t vertex_id)
{
        int lo = 0;
        int hi = g->world_size - 1;
        while (lo < hi) {
                int mid = lo + ((hi - lo + 1) / 2);
                if (g->rank_vertex_offsets[mid] <= vertex_id)
                        lo = mid;
                else
                        hi = mid - 1;
        }
        return lo;
}

int dstm_dgraph_vertex_owner(dstm_dgraph_t *g, size_t vertex_id)
{
        if (g == NULL || vertex_id >= g->total_vertices)
                return -DSTM_EINVAL;
        return dgraph_find_rank(g, vertex_id);
}

/* ---------------------------------------------------------------------------
 * Introspection
 * ------------------------------------------------------------------------- */

size_t dstm_dgraph_vertex_count(dstm_dgraph_t *g)
{
        return g ? g->total_vertices : 0;
}

size_t dstm_dgraph_local_vertex_count(dstm_dgraph_t *g)
{
        return g ? g->local_vertices : 0;
}

size_t dstm_dgraph_local_vertex_offset(dstm_dgraph_t *g)
{
        return g ? g->local_vertex_offset : 0;
}

size_t dstm_dgraph_edge_count(dstm_dgraph_t *g)
{
        return g ? g->total_edges : 0;
}

size_t dstm_dgraph_local_edge_count(dstm_dgraph_t *g)
{
        return g ? g->local_edge_count : 0;
}

/* ---------------------------------------------------------------------------
 * Creation
 * ------------------------------------------------------------------------- */

/* Validate per-vertex neighbor slice: strictly ascending, all in
 * [0, total_vertices). Returns 0 on success, -DSTM_EINVAL on violation. */
static int validate_neighbors(const size_t *dsts, size_t deg,
                              size_t total_vertices)
{
        for (size_t i = 0; i < deg; i++) {
                if (dsts[i] >= total_vertices)
                        return -DSTM_EINVAL;
                if (i > 0 && dsts[i] <= dsts[i - 1])
                        return -DSTM_EINVAL;
        }
        return 0;
}

int dstm_dgraph_create(dstm_dgraph_t **ret,
                       size_t vertex_state_size,
                       size_t edge_state_size,
                       size_t local_vertices,
                       const size_t *local_out_degrees,
                       const size_t *local_edge_dsts,
                       const void *local_vertex_states,
                       const void *local_edge_states)
{
        if (ret == NULL || local_vertices == 0 || local_out_degrees == NULL) {
                if (ret) *ret = NULL;
                return -DSTM_EINVAL;
        }

        int err = 0;
        int world_size = dstm_global_context.world_size;
        int self       = dstm_global_context.rank;
        MPI_Comm comm  = dstm_global_context.comm;

        dstm_dgraph_t *g = calloc(1, sizeof(*g));
        if (g == NULL) { *ret = NULL; return -DSTM_ENOMEM; }

        g->region_id  = dstm_get_local_region_id();
        g->region     = dstm_get_local_region();
        g->world_size = world_size;
        g->local_rank = self;
        g->vertex_state_size = vertex_state_size;
        g->edge_state_size   = edge_state_size;
        g->edge_slot_stride  = compute_edge_slot_stride(edge_state_size);
        g->local_vertices    = local_vertices;

        /* --- Compute local_edge_count ------------------------------------- */
        size_t local_edge_count = 0;
        for (size_t i = 0; i < local_vertices; i++)
                local_edge_count += local_out_degrees[i];
        g->local_edge_count = local_edge_count;

        /* --- Per-rank arrays --------------------------------------------- */
        g->rank_local_vertices    = calloc(world_size, sizeof(size_t));
        g->rank_local_edge_count  = calloc(world_size, sizeof(size_t));
        g->rank_vertex_state_off  = calloc(world_size, sizeof(intptr_t));
        g->rank_edge_row_ptr_off  = calloc(world_size, sizeof(intptr_t));
        g->rank_edge_slot_off     = calloc(world_size, sizeof(intptr_t));
        g->rank_edge_slot_stride  = calloc(world_size, sizeof(size_t));
        g->rank_vertex_offsets     = calloc((size_t)world_size + 1,
                                            sizeof(size_t));

        if (!g->rank_local_vertices || !g->rank_local_edge_count ||
            !g->rank_vertex_state_off || !g->rank_edge_row_ptr_off ||
            !g->rank_edge_slot_off   || !g->rank_edge_slot_stride  ||
            !g->rank_vertex_offsets) {
                err = -DSTM_ENOMEM;
                goto err_free_arrays;
        }

        /* --- Exchange vertex counts to build global partitioning ---------- */
        err = MPI_Allgather(&local_vertices, 1, MPI_UNSIGNED_LONG_LONG,
                            g->rank_local_vertices, 1,
                            MPI_UNSIGNED_LONG_LONG, comm);
        if (err != MPI_SUCCESS) { err = -DSTM_FAILURE; goto err_free_arrays; }

        err = MPI_Allgather(&local_edge_count, 1, MPI_UNSIGNED_LONG_LONG,
                            g->rank_local_edge_count, 1,
                            MPI_UNSIGNED_LONG_LONG, comm);
        if (err != MPI_SUCCESS) { err = -DSTM_FAILURE; goto err_free_arrays; }

        /* Build prefix sum of vertex counts (length world_size + 1). */
        size_t total_v = 0, total_e = 0;
        g->rank_vertex_offsets[0] = 0;
        for (int r = 0; r < world_size; r++) {
                g->rank_vertex_offsets[r] = total_v;
                total_v += g->rank_local_vertices[r];
                total_e += g->rank_local_edge_count[r];
        }
        g->rank_vertex_offsets[world_size] = total_v;
        g->total_vertices      = total_v;
        g->total_edges         = total_e;
        g->local_vertex_offset = g->rank_vertex_offsets[self];

        /* --- Validate local topology now that total_vertices is known ---- */
        {
                const size_t *cursor = local_edge_dsts;
                for (size_t i = 0; i < local_vertices; i++) {
                        size_t deg = local_out_degrees[i];
                        if (deg > 0) {
                                if (cursor == NULL) {
                                        err = -DSTM_EINVAL;
                                        goto err_free_arrays;
                                }
                                int rc = validate_neighbors(cursor, deg,
                                                            total_v);
                                if (rc != 0) { err = rc; goto err_free_arrays; }
                        }
                        cursor += deg;
                }
        }

        /* --- Compute slab sizes on this rank ----------------------------- */
        g->vertex_state_bytes  = local_vertices * vertex_state_size;
        g->edge_row_ptr_bytes  = (local_vertices + 1) * sizeof(dgraph_vid_t);
        g->edge_slot_bytes     = local_edge_count * g->edge_slot_stride;

        /* Pad each slab up so the next one starts aligned. */
        size_t vs_padded  = align_up(g->vertex_state_bytes, DGRAPH_ALIGN);
        size_t rp_padded  = align_up(g->edge_row_ptr_bytes, DGRAPH_ALIGN);
        size_t es_padded  = align_up(g->edge_slot_bytes,    DGRAPH_ALIGN);
        g->region_bytes   = vs_padded + rp_padded + es_padded;

        /* --- Single region allocation ------------------------------------ */
        dstm_ptr_t base = dstm_region_malloc(g->region,
                                             g->region_bytes ? g->region_bytes
                                                             : DGRAPH_ALIGN);
        if (dstm_ptr_isnull(base)) {
                err = -DSTM_ENOMEM;
                goto err_free_arrays;
        }

        g->vertex_state_off = base.offset;
        g->edge_row_ptr_off = base.offset + (intptr_t)vs_padded;
        g->edge_slot_off    = g->edge_row_ptr_off + (intptr_t)rp_padded;

        /* --- Host-side cached row_ptrs ----------------------------------- */
        g->row_ptrs = calloc(local_vertices + 1, sizeof(dgraph_vid_t));
        if (g->row_ptrs == NULL) {
                err = -DSTM_ENOMEM;
                goto err_free_region;
        }
        {
                dgraph_vid_t acc = 0;
                for (size_t i = 0; i < local_vertices; i++) {
                        g->row_ptrs[i] = acc;
                        acc += (dgraph_vid_t)local_out_degrees[i];
                }
                g->row_ptrs[local_vertices] = acc;
        }

        /* --- Exchange each rank's slab base offsets --------------------- */
        err = MPI_Allgather(&g->vertex_state_off, 1, MPI_AINT,
                            g->rank_vertex_state_off, 1, MPI_AINT, comm);
        if (err != MPI_SUCCESS) { err = -DSTM_FAILURE; goto err_free_rowptrs; }

        err = MPI_Allgather(&g->edge_row_ptr_off, 1, MPI_AINT,
                            g->rank_edge_row_ptr_off, 1, MPI_AINT, comm);
        if (err != MPI_SUCCESS) { err = -DSTM_FAILURE; goto err_free_rowptrs; }

        err = MPI_Allgather(&g->edge_slot_off, 1, MPI_AINT,
                            g->rank_edge_slot_off, 1, MPI_AINT, comm);
        if (err != MPI_SUCCESS) { err = -DSTM_FAILURE; goto err_free_rowptrs; }

        err = MPI_Allgather(&g->edge_slot_stride, 1, MPI_UNSIGNED_LONG_LONG,
                            g->rank_edge_slot_stride, 1,
                            MPI_UNSIGNED_LONG_LONG, comm);
        if (err != MPI_SUCCESS) { err = -DSTM_FAILURE; goto err_free_rowptrs; }

                /* --- Replicate topology on every rank ---------------------------- */
        g->rank_row_ptrs  = calloc(world_size, sizeof(dgraph_vid_t *));
        g->rank_edge_dsts = calloc(world_size, sizeof(dgraph_vid_t *));
        if (!g->rank_row_ptrs || !g->rank_edge_dsts) {
                err = -DSTM_ENOMEM;
                goto err_free_topocache;
        }

        /* Row pointers: build displacements and counts for Allgatherv. */
        {
                int *rp_counts = malloc((size_t)world_size * sizeof(int));
                int *rp_displs = malloc((size_t)world_size * sizeof(int));
                if (!rp_counts || !rp_displs) {
                        free(rp_counts); free(rp_displs);
                        err = -DSTM_ENOMEM;
                        goto err_free_topocache;
                }
                size_t total_rp = 0;
                for (int r = 0; r < world_size; r++) {
                        rp_counts[r] = (int)(g->rank_local_vertices[r] + 1);
                        rp_displs[r] = (int)total_rp;
                        total_rp += (size_t)rp_counts[r];
                }
                dgraph_vid_t *rp_all = malloc(total_rp * sizeof(dgraph_vid_t));
                if (rp_all == NULL) {
                        free(rp_counts); free(rp_displs);
                        err = -DSTM_ENOMEM;
                        goto err_free_topocache;
                }
                err = MPI_Allgatherv(g->row_ptrs,
                                     (int)(local_vertices + 1),
                                     MPI_UINT64_T,
                                     rp_all, rp_counts, rp_displs,
                                     MPI_UINT64_T, comm);
                if (err != MPI_SUCCESS) {
                        free(rp_counts); free(rp_displs); free(rp_all);
                        err = -DSTM_FAILURE;
                        goto err_free_topocache;
                }
                for (int r = 0; r < world_size; r++) {
                        size_t n = g->rank_local_vertices[r] + 1;
                        g->rank_row_ptrs[r] =
                                malloc(n * sizeof(dgraph_vid_t));
                        if (g->rank_row_ptrs[r] == NULL) {
                                free(rp_counts); free(rp_displs); free(rp_all);
                                err = -DSTM_ENOMEM;
                                goto err_free_topocache;
                        }
                        memcpy(g->rank_row_ptrs[r],
                               rp_all + rp_displs[r],
                               n * sizeof(dgraph_vid_t));
                }
                free(rp_counts); free(rp_displs); free(rp_all);
        }

        /* Edge dsts: same pattern. */
        {
                int *ed_counts = malloc((size_t)world_size * sizeof(int));
                int *ed_displs = malloc((size_t)world_size * sizeof(int));
                if (!ed_counts || !ed_displs) {
                        free(ed_counts); free(ed_displs);
                        err = -DSTM_ENOMEM;
                        goto err_free_topocache;
                }
                size_t total_ed = 0;
                for (int r = 0; r < world_size; r++) {
                        ed_counts[r] = (int)g->rank_local_edge_count[r];
                        ed_displs[r] = (int)total_ed;
                        total_ed += (size_t)ed_counts[r];
                }
                dgraph_vid_t *local_dsts =
                        malloc(local_edge_count * sizeof(dgraph_vid_t));
                if (local_edge_count > 0 && local_dsts == NULL) {
                        free(ed_counts); free(ed_displs);
                        err = -DSTM_ENOMEM;
                        goto err_free_topocache;
                }
                for (size_t s = 0; s < local_edge_count; s++)
                        local_dsts[s] = (dgraph_vid_t)local_edge_dsts[s];

                dgraph_vid_t *ed_all = total_ed
                        ? malloc(total_ed * sizeof(dgraph_vid_t))
                        : NULL;
                if (total_ed > 0 && ed_all == NULL) {
                        free(ed_counts); free(ed_displs); free(local_dsts);
                        err = -DSTM_ENOMEM;
                        goto err_free_topocache;
                }
                err = MPI_Allgatherv(local_dsts, (int)local_edge_count,
                                     MPI_UINT64_T,
                                     ed_all, ed_counts, ed_displs,
                                     MPI_UINT64_T, comm);
                free(local_dsts);
                if (err != MPI_SUCCESS) {
                        free(ed_counts); free(ed_displs); free(ed_all);
                        err = -DSTM_FAILURE;
                        goto err_free_topocache;
                }
                for (int r = 0; r < world_size; r++) {
                        size_t n = g->rank_local_edge_count[r];
                        if (n == 0) {
                                g->rank_edge_dsts[r] = NULL;
                                continue;
                        }
                        g->rank_edge_dsts[r] =
                                malloc(n * sizeof(dgraph_vid_t));
                        if (g->rank_edge_dsts[r] == NULL) {
                                free(ed_counts); free(ed_displs); free(ed_all);
                                err = -DSTM_ENOMEM;
                                goto err_free_topocache;
                        }
                        memcpy(g->rank_edge_dsts[r],
                               ed_all + ed_displs[r],
                               n * sizeof(dgraph_vid_t));
                }
                free(ed_counts); free(ed_displs); free(ed_all);
        }

        /* --- Publish initial state at time 0 ------------------------------ */
        /* Row pointers. */
        {
                dstm_ptr_t rp_ptr = make_ptr(g->region_id, g->edge_row_ptr_off);
                int rc = dstm_put(self, g->region_id, 0, 0, rp_ptr,
                                  g->row_ptrs, g->edge_row_ptr_bytes);
                if (rc != DSTM_SUCCESS) { err = rc; goto err_free_rowptrs; }
        }

        /* Vertex state. */
        if (g->vertex_state_bytes > 0) {
                void *vbuf = local_vertex_states
                             ? (void *)local_vertex_states
                             : calloc(1, g->vertex_state_bytes);
                if (vbuf == NULL) {
                        err = -DSTM_ENOMEM;
                        goto err_free_rowptrs;
                }
                dstm_ptr_t vp = make_ptr(g->region_id, g->vertex_state_off);
                int rc = dstm_put(self, g->region_id, 0, 0, vp, vbuf,
                                  g->vertex_state_bytes);
                if (vbuf != local_vertex_states) free(vbuf);
                if (rc != DSTM_SUCCESS) { err = rc; goto err_free_rowptrs; }
        }

        /* Edge slots: build one contiguous packed buffer of stride entries
         * with [dst | user_state | pad], then a single put. */
        if (local_edge_count > 0) {
                char *ebuf = calloc(local_edge_count, g->edge_slot_stride);
                if (ebuf == NULL) {
                        err = -DSTM_ENOMEM;
                        goto err_free_rowptrs;
                }
                const size_t *dcursor = local_edge_dsts;
                const char *scursor   = (const char *)local_edge_states;
                for (size_t slot = 0; slot < local_edge_count; slot++) {
                        char *entry = ebuf + slot * g->edge_slot_stride;
                        dgraph_vid_t d = (dgraph_vid_t)dcursor[slot];
                        memcpy(entry, &d, sizeof(dgraph_vid_t));
                        if (edge_state_size > 0) {
                                if (scursor) {
                                        memcpy(entry + sizeof(dgraph_vid_t),
                                               scursor + slot * edge_state_size,
                                               edge_state_size);
                                }
                                /* else: already zeroed by calloc */
                        }
                }
                dstm_ptr_t ep = make_ptr(g->region_id, g->edge_slot_off);
                int rc = dstm_put(self, g->region_id, 0, 0, ep, ebuf,
                                  g->edge_slot_bytes);
                free(ebuf);
                if (rc != DSTM_SUCCESS) { err = rc; goto err_free_rowptrs; }
        }

        *ret = g;
        return DSTM_SUCCESS;

err_free_topocache:
        if (g->rank_row_ptrs) {
                for (int r = 0; r < world_size; r++)
                        free(g->rank_row_ptrs[r]);
                free(g->rank_row_ptrs);
        }
        if (g->rank_edge_dsts) {
                for (int r = 0; r < world_size; r++)
                        free(g->rank_edge_dsts[r]);
                free(g->rank_edge_dsts);
        }
        free(g->row_ptrs);
        goto err_free_region_only;
err_free_region_only:
err_free_rowptrs:
        free(g->row_ptrs);
err_free_region:
        /* We cannot easily rebuild the base dstm_ptr_t here because we
         * discarded it; reconstruct from vertex_state_off. */
        {
                dstm_ptr_t base_p = make_ptr(g->region_id, g->vertex_state_off);
                dstm_region_free(g->region, base_p);
        }
err_free_arrays:
        free(g->rank_local_vertices);
        free(g->rank_local_edge_count);
        free(g->rank_vertex_state_off);
        free(g->rank_edge_row_ptr_off);
        free(g->rank_edge_slot_off);
        free(g->rank_edge_slot_stride);
        free(g->rank_vertex_offsets);
        free(g);
        *ret = NULL;
        return err;
}

int dstm_dgraph_create_equal(dstm_dgraph_t **ret,
                             size_t vertex_state_size,
                             size_t edge_state_size,
                             size_t total_vertices,
                             const size_t *local_out_degrees,
                             const size_t *local_edge_dsts,
                             const void *local_vertex_states,
                             const void *local_edge_states)
{
        int world_size = dstm_global_context.world_size;
        int self       = dstm_global_context.rank;

        size_t local_vertices = total_vertices / (size_t)world_size;
        size_t remainder      = total_vertices % (size_t)world_size;
        if (self < (int)remainder)
                local_vertices++;

        return dstm_dgraph_create(ret, vertex_state_size, edge_state_size,
                                  local_vertices, local_out_degrees,
                                  local_edge_dsts, local_vertex_states,
                                  local_edge_states);
}

void dstm_dgraph_destroy(dstm_dgraph_t **dgraph)
{
        if (dgraph == NULL || *dgraph == NULL)
                return;
        dstm_dgraph_t *g = *dgraph;

        dstm_ptr_t base_p = make_ptr(g->region_id, g->vertex_state_off);
        dstm_region_free(g->region, base_p);

        if (g->rank_row_ptrs) {
                for (int r = 0; r < g->world_size; r++)
                        free(g->rank_row_ptrs[r]);
                free(g->rank_row_ptrs);
        }
        if (g->rank_edge_dsts) {
                for (int r = 0; r < g->world_size; r++)
                        free(g->rank_edge_dsts[r]);
                free(g->rank_edge_dsts);
        }

        free(g->row_ptrs);
        free(g->rank_vertex_offsets);
        free(g->rank_local_vertices);
        free(g->rank_local_edge_count);
        free(g->rank_vertex_state_off);
        free(g->rank_edge_row_ptr_off);
        free(g->rank_edge_slot_off);
        free(g->rank_edge_slot_stride);
        free(g);
        *dgraph = NULL;
}

/* ---------------------------------------------------------------------------
 * Vertex state operations
 * ------------------------------------------------------------------------- */

int dstm_dgraph_vertex_get(dstm_dgraph_t *g, size_t vertex_id, void *dst,
                           dstm_time_t time, dstm_time_t delta)
{
        if (g == NULL || dst == NULL) return -DSTM_EINVAL;
        if (vertex_id >= g->total_vertices) return -DSTM_EINVAL;
        if (g->vertex_state_size == 0) return DSTM_SUCCESS;

        int rank = dgraph_find_rank(g, vertex_id);
        size_t local_idx = vertex_id - g->rank_vertex_offsets[rank];
        intptr_t off = g->rank_vertex_state_off[rank]
                       + (intptr_t)(local_idx * g->vertex_state_size);
        dstm_ptr_t src = make_ptr(rank, off);
        return dstm_get(rank, rank, time, delta, dst, src,
                        g->vertex_state_size);
}

int dstm_dgraph_vertex_put(dstm_dgraph_t *g, size_t vertex_id, void *src,
                           dstm_time_t time, dstm_time_t delta)
{
        if (g == NULL || src == NULL) return -DSTM_EINVAL;
        if (vertex_id >= g->total_vertices) return -DSTM_EINVAL;
        if (g->vertex_state_size == 0) return DSTM_SUCCESS;

        int rank = dgraph_find_rank(g, vertex_id);
        size_t local_idx = vertex_id - g->rank_vertex_offsets[rank];
        intptr_t off = g->rank_vertex_state_off[rank]
                       + (intptr_t)(local_idx * g->vertex_state_size);
        dstm_ptr_t dst = make_ptr(rank, off);
        return dstm_put(rank, rank, time, delta, dst, src,
                        g->vertex_state_size);
}

int dstm_dgraph_vertex_get_or_default(dstm_dgraph_t *g, size_t vertex_id,
                                      void *dst, dstm_time_t time,
                                      dstm_time_t delta,
                                      const void *default_value)
{
        if (g == NULL || dst == NULL || default_value == NULL)
                return -DSTM_EINVAL;
        if (time + delta < 0) {
                if (g->vertex_state_size > 0)
                        memcpy(dst, default_value, g->vertex_state_size);
                return DSTM_SUCCESS;
        }
        return dstm_dgraph_vertex_get(g, vertex_id, dst, time, delta);
}

/* ---------------------------------------------------------------------------
 * Edge lookup helpers
 *
 * fetch_row_range: return [row_start, row_end) for local vertex `local_idx`
 * on rank `rank`, using the host cache for the local rank and dstm_get on
 * the row-pointer slab for remote ranks.
 * ------------------------------------------------------------------------- */

/* Topology is fully replicated on every rank, so no DSTM traffic is
 * needed here. (time, delta) are ignored -- topology is immutable. */
static int fetch_row_range(dstm_dgraph_t *g, int rank, size_t local_idx,
                           dstm_time_t time, dstm_time_t delta,
                           dgraph_vid_t *out_start, dgraph_vid_t *out_end)
{
        (void)time;
        (void)delta;
        *out_start = g->rank_row_ptrs[rank][local_idx];
        *out_end   = g->rank_row_ptrs[rank][local_idx + 1];
        return DSTM_SUCCESS;
}

/* Binary search in the fully-replicated dst cache; no DSTM traffic. */
static int find_edge_slot(dstm_dgraph_t *g, int rank,
                          dgraph_vid_t row_start, dgraph_vid_t row_end,
                          size_t v, dstm_time_t time, dstm_time_t delta,
                          dgraph_vid_t *out_slot)
{
        (void)time;
        (void)delta;
        const dgraph_vid_t *dsts = g->rank_edge_dsts[rank];
        dgraph_vid_t lo = row_start;
        dgraph_vid_t hi = row_end;
        dgraph_vid_t target = (dgraph_vid_t)v;

        while (lo < hi) {
                dgraph_vid_t mid = lo + (hi - lo) / 2;
                dgraph_vid_t mid_dst = dsts[mid];
                if (mid_dst == target) {
                        *out_slot = mid;
                        return DSTM_SUCCESS;
                }
                if (mid_dst < target)
                        lo = mid + 1;
                else
                        hi = mid;
        }
        return -DSTM_ENOTFOUND;
}

/* ---------------------------------------------------------------------------
 * Edge state operations
 * ------------------------------------------------------------------------- */

int dstm_dgraph_edge_get(dstm_dgraph_t *g, size_t u, size_t v, void *dst,
                         dstm_time_t time, dstm_time_t delta)
{
        if (g == NULL || dst == NULL) return -DSTM_EINVAL;
        if (u >= g->total_vertices || v >= g->total_vertices)
                return -DSTM_EINVAL;

        int rank = dgraph_find_rank(g, u);
        size_t local_idx = u - g->rank_vertex_offsets[rank];

        dgraph_vid_t rs, re;
        int rc = fetch_row_range(g, rank, local_idx, time, delta, &rs, &re);
        if (rc != DSTM_SUCCESS) return rc;

        dgraph_vid_t slot;
        rc = find_edge_slot(g, rank, rs, re, v, time, delta, &slot);
        if (rc != DSTM_SUCCESS) return rc;

        if (g->edge_state_size == 0) return DSTM_SUCCESS;

        size_t stride = g->rank_edge_slot_stride[rank];
        intptr_t off = g->rank_edge_slot_off[rank]
                       + (intptr_t)((size_t)slot * stride)
                       + (intptr_t)sizeof(dgraph_vid_t);
        dstm_ptr_t src = make_ptr(rank, off);
        return dstm_get(rank, rank, time, delta, dst, src,
                        g->edge_state_size);
}

int dstm_dgraph_edge_put(dstm_dgraph_t *g, size_t u, size_t v, void *src,
                         dstm_time_t time, dstm_time_t delta)
{
        if (g == NULL || src == NULL) return -DSTM_EINVAL;
        if (u >= g->total_vertices || v >= g->total_vertices)
                return -DSTM_EINVAL;

        int rank = dgraph_find_rank(g, u);
        size_t local_idx = u - g->rank_vertex_offsets[rank];

        dgraph_vid_t rs, re;
        /* Topology (row pointers and dst fields) is immutable and was
         * written at (t=0, delta=0). Reads must use delta <= 0, so we
         * always use (0, 0) regardless of the caller's state-put
         * (time, delta), which may have delta > 0. */
        int rc = fetch_row_range(g, rank, local_idx, time, delta, &rs, &re);
        if (rc != DSTM_SUCCESS) return rc;

        dgraph_vid_t slot;
        rc = find_edge_slot(g, rank, rs, re, v, time, 0, &slot);
        if (rc != DSTM_SUCCESS) return rc;

        if (g->edge_state_size == 0) return DSTM_SUCCESS;

        size_t stride = g->rank_edge_slot_stride[rank];
        intptr_t off = g->rank_edge_slot_off[rank]
                       + (intptr_t)((size_t)slot * stride)
                       + (intptr_t)sizeof(dgraph_vid_t);
        dstm_ptr_t dst = make_ptr(rank, off);
        return dstm_put(rank, rank, time, delta, dst, src,
                        g->edge_state_size);
}

/* ---------------------------------------------------------------------------
 * Neighborhood queries
 * ------------------------------------------------------------------------- */

int dstm_dgraph_degree_get(dstm_dgraph_t *g, size_t u, size_t *out_degree)
{
        if (g == NULL || out_degree == NULL) return -DSTM_EINVAL;
        if (u >= g->total_vertices) return -DSTM_EINVAL;

        int rank = dgraph_find_rank(g, u);
        size_t local_idx = u - g->rank_vertex_offsets[rank];
        dgraph_vid_t rs, re;
        /* Topology is time-invariant; use (0, 0) for the underlying read
         * so remote gets go against the guaranteed-present t=0 snapshot. */
        int rc = fetch_row_range(g, rank, local_idx, 0, 0, &rs, &re);
        if (rc != DSTM_SUCCESS) return rc;
        *out_degree = (size_t)(re - rs);
        return DSTM_SUCCESS;
}

int dstm_dgraph_neighbors_get(dstm_dgraph_t *g, size_t u,
                              size_t *out_neighbors, size_t *io_count)
{
        if (g == NULL || io_count == NULL) return -DSTM_EINVAL;
        if (u >= g->total_vertices) return -DSTM_EINVAL;

        int rank = dgraph_find_rank(g, u);
        size_t local_idx = u - g->rank_vertex_offsets[rank];

        dgraph_vid_t rs, re;
        int rc = fetch_row_range(g, rank, local_idx, 0, 0, &rs, &re);
        if (rc != DSTM_SUCCESS) return rc;
        size_t deg = (size_t)(re - rs);

        if (out_neighbors == NULL || *io_count < deg) {
                *io_count = deg;
                return (out_neighbors == NULL && *io_count == deg && deg == 0)
                       ? DSTM_SUCCESS : -DSTM_ENOMEM;
        }

        if (deg == 0) {
                *io_count = 0;
                return DSTM_SUCCESS;
        }

        /* Read straight from the replicated topology cache. */
        const dgraph_vid_t *dsts = g->rank_edge_dsts[rank];
        for (size_t i = 0; i < deg; i++)
                out_neighbors[i] = (size_t)dsts[(size_t)rs + i];
        *io_count = deg;
        return DSTM_SUCCESS;
}


/* ---------------------------------------------------------------------------
 * Distance matrix derivation
 * ------------------------------------------------------------------------- */

int dstm_dgraph_derive_distance(dstm_dgraph_t *g, dstm_distance_t **ret,
                                int default_delta)
{
        if (g == NULL || ret == NULL || default_delta <= 0)
                return -DSTM_EINVAL;

        int world_size = g->world_size;
        int self       = g->local_rank;
        MPI_Comm comm  = dstm_global_context.comm;

        /* Local row of the rank-level adjacency: which remote ranks does
         * this rank touch via any of its outgoing edges? */
        int *local_row = calloc(world_size, sizeof(int));
        if (local_row == NULL) return -DSTM_ENOMEM;

        const dgraph_vid_t *local_dsts = g->rank_edge_dsts[self];
        for (size_t i = 0; i < g->local_vertices; i++) {
                dgraph_vid_t rs = g->row_ptrs[i];
                dgraph_vid_t re = g->row_ptrs[i + 1];
                for (dgraph_vid_t s = rs; s < re; s++) {
                        int dst_rank = dgraph_find_rank(g,
                                                        (size_t)local_dsts[s]);
                        local_row[dst_rank] = 1;
                }
        }

        /* Gather every rank's row into a full world_size * world_size
         * boolean adjacency matrix. */
        int *full = calloc((size_t)world_size * (size_t)world_size,
                           sizeof(int));
        if (full == NULL) { free(local_row); return -DSTM_ENOMEM; }

        int err = MPI_Allgather(local_row, world_size, MPI_INT,
                                full, world_size, MPI_INT, comm);
        free(local_row);
        if (err != MPI_SUCCESS) {
                free(full);
                return -DSTM_FAILURE;
        }

        /* Symmetrize (reachable is undirected for topology purposes) and
         * convert to the distance-matrix encoding: 0 on diagonal,
         * default_delta on reachable off-diagonal pairs, -1 elsewhere. */
        int *matrix = malloc((size_t)world_size * (size_t)world_size
                             * sizeof(int));
        if (matrix == NULL) { free(full); return -DSTM_ENOMEM; }

        for (int i = 0; i < world_size; i++) {
                for (int j = 0; j < world_size; j++) {
                        int v;
                        if (i == j) {
                                v = 0;
                        } else {
                                int reachable = full[i * world_size + j]
                                                || full[j * world_size + i];
                                v = reachable ? default_delta : -1;
                        }
                        matrix[i * world_size + j] = v;
                }
        }
        free(full);

        int rc = dstm_distance_create_from_matrix(ret, (size_t)world_size,
                                                  matrix);
        free(matrix);
        return rc;
}
