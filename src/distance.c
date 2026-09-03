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

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "dstm.h"

#include "internal/distance.h"

/*******************************************************************************
 * Helper Functions
 ******************************************************************************/

/**
 * Compute all-pairs shortest paths using Floyd-Warshall algorithm.
 * Modifies the distance matrix in-place to contain shortest path distances.
 *
 * @param distances: flattened 2D array (size x size)
 * @param size: number of nodes
 */
static void compute_shortest_paths(int *distances, size_t size)
{
	for (size_t k = 0; k < size; k++)
		for (size_t i = 0; i < size; i++)
			for (size_t j = 0; j < size; j++) {
				int ik = distances[i * size + k];
				int kj = distances[k * size + j];
				if (ik == INT_MAX || kj == INT_MAX)
					continue;
				int via_k = ik + kj;
				if (via_k < distances[i * size + j])
					distances[i * size + j] = via_k;
			}
}

/**
 * Initialize distance matrix with infinity and diagonal zeros.
 * This prepares the matrix for shortest path computation.
 *
 * @param distances: flattened 2D array (size x size)
 * @param size: number of nodes
 */
static void init_distance_matrix(int *distances, size_t size)
{
	for (size_t i = 0; i < size * size; i++)
		distances[i] = INT_MAX;
	for (size_t i = 0; i < size; i++)
		distances[i * size + i] = 0;
}

/**
 * Allocate and initialise a dstm_distance_t with a zeroed distance matrix.
 *
 * Validates parameters, allocates the structure and its distance array, and
 * stores the result in *ret.  On failure, sets dstm_errno and returns a
 * negative error code.
 *
 * @param[out] ret       pointer to distance structure pointer (must be non-NULL)
 * @param[in]  size      number of nodes
 * @param[in]  min_size  minimum acceptable size (topology-dependent)
 * @return DSTM_SUCCESS on success, negative error code on failure
 */
static int dstm_distance_alloc(dstm_distance_t **ret, size_t size,
                               size_t min_size)
{
	dstm_distance_t *dist;

	if (ret == NULL)
		goto err_inval;

	if (size < min_size || size > DSTM_DISTANCE_MAX_SIZE)
		goto err_inval;

	dist = malloc(sizeof(*dist));
	if (dist == NULL)
		goto err_nomem;

	dist->distances = calloc(size * size, sizeof(int));
	if (dist->distances == NULL)
		goto err_free_dist;

	dist->size = size;
	*ret = dist;
	return DSTM_SUCCESS;

err_free_dist:
	free(dist);
err_nomem:
	dstm_errno = DSTM_ENOMEM;
	return -DSTM_ENOMEM;
err_inval:
	dstm_errno = DSTM_EINVAL;
	return -DSTM_EINVAL;
}

/*******************************************************************************
 * Common Topologies
 ******************************************************************************/

/**
 * Create ring/circular topology distance matrix.
 *
 * In a ring topology, each node is connected to exactly two neighbors
 * in a circular pattern: node i connects to (i-1) mod N and (i+1) mod N.
 * The distance between any two nodes is the minimum number of hops
 * around the ring.
 *
 * Example for 4 nodes:
 *   0 -- 1
 *   |    |
 *   3 -- 2
 *
 * Distance matrix (hop counts):
 *     0  1  2  3
 *   0[0, 1, 2, 1]
 *   1[1, 0, 1, 2]
 *   2[2, 1, 0, 1]
 *   3[1, 2, 1, 0]
 *
 * @param[out] ret: pointer to distance structure pointer (must be non-NULL)
 * @param[in] size: number of nodes (must be >= 3 for a valid ring)
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL for invalid parameters,
 *         -DSTM_ENOMEM for allocation failures
 */
int dstm_distance_create_ring(dstm_distance_t **ret, size_t size)
{
	int err = dstm_distance_alloc(ret, size, 3);
	if (err)
		return err;

	dstm_distance_t *dist = *ret;
	init_distance_matrix(dist->distances, size);

	/* Set ring edges (bidirectional) */
	for (size_t i = 0; i < size; i++) {
		size_t next = (i + 1) % size;
		size_t prev = (i + size - 1) % size;
		dist->distances[i * size + next] = 1;
		dist->distances[i * size + prev] = 1;
	}

	compute_shortest_paths(dist->distances, size);
	return DSTM_SUCCESS;
}

/**
 * Create fully connected (complete graph) topology distance matrix.
 *
 * In a complete graph, every node has a direct connection to every other
 * node. All non-diagonal distances are 1, and diagonal (self) distances
 * are 0. This represents the lowest-latency topology but requires the
 * most connections.
 *
 * Example for 4 nodes (all pairs connected):
 *     0 --- 1
 *     |\   /|
 *     | \ / |
 *     | / \ |
 *     |/   \|
 *     3 --- 2
 *
 * Distance matrix:
 *     0  1  2  3
 *   0[0, 1, 1, 1]
 *   1[1, 0, 1, 1]
 *   2[1, 1, 0, 1]
 *   3[1, 1, 1, 0]
 *
 * @param[out] ret: pointer to distance structure pointer (must be non-NULL)
 * @param[in] size: number of nodes (must be >= 1)
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL for invalid parameters,
 *         -DSTM_ENOMEM for allocation failures
 */
int dstm_distance_create_complete(dstm_distance_t **ret, size_t size)
{
	int err = dstm_distance_alloc(ret, size, 1);
	if (err)
		return err;

	dstm_distance_t *dist = *ret;

	/* In a complete graph, all nodes are directly connected */
	for (size_t i = 0; i < size; i++)
		for (size_t j = 0; j < size; j++)
			dist->distances[i * size + j] = (i == j) ? 0 : 1;

	return DSTM_SUCCESS;
}

/**
 * Create star topology distance matrix.
 *
 * In a star topology, node 0 is the central hub connected to all other
 * nodes. Non-central nodes are only connected to the center, not to each
 * other. Communication between non-central nodes requires 2 hops through
 * the center.
 *
 * Example for 5 nodes:
 *       1
 *       |
 *   2---0---3
 *       |
 *       4
 *
 * Distance matrix:
 *     0  1  2  3  4
 *   0[0, 1, 1, 1, 1]
 *   1[1, 0, 2, 2, 2]
 *   2[1, 2, 0, 2, 2]
 *   3[1, 2, 2, 0, 2]
 *   4[1, 2, 2, 2, 0]
 *
 * @param[out] ret: pointer to distance structure pointer (must be non-NULL)
 * @param[in] size: number of nodes (must be >= 2 for a valid star)
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL for invalid parameters,
 *         -DSTM_ENOMEM for allocation failures
 */
int dstm_distance_create_star(dstm_distance_t **ret, size_t size)
{
	int err = dstm_distance_alloc(ret, size, 2);
	if (err)
		return err;

	dstm_distance_t *dist = *ret;
	init_distance_matrix(dist->distances, size);

	/* Set edges: center (node 0) to all others */
	for (size_t i = 1; i < size; i++) {
		dist->distances[0 * size + i] = 1;
		dist->distances[i * size + 0] = 1;
	}

	compute_shortest_paths(dist->distances, size);
	return DSTM_SUCCESS;
}

/**
 * Create line/chain topology distance matrix.
 *
 * In a line topology, nodes are connected in a linear sequence:
 * 0 -- 1 -- 2 -- ... -- (N-1). Each node (except endpoints) has exactly
 * two neighbors. The distance between nodes is the absolute difference
 * of their indices.
 *
 * Example for 5 nodes:
 *   0 -- 1 -- 2 -- 3 -- 4
 *
 * Distance matrix:
 *     0  1  2  3  4
 *   0[0, 1, 2, 3, 4]
 *   1[1, 0, 1, 2, 3]
 *   2[2, 1, 0, 1, 2]
 *   3[3, 2, 1, 0, 1]
 *   4[4, 3, 2, 1, 0]
 *
 * @param[out] ret: pointer to distance structure pointer (must be non-NULL)
 * @param[in] size: number of nodes (must be >= 1)
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL for invalid parameters,
 *         -DSTM_ENOMEM for allocation failures
 */
int dstm_distance_create_line(dstm_distance_t **ret, size_t size)
{
	int err = dstm_distance_alloc(ret, size, 1);
	if (err)
		return err;

	dstm_distance_t *dist = *ret;

	/* For a line topology, distance is simply |i - j| */
	for (size_t i = 0; i < size; i++)
		for (size_t j = 0; j < size; j++) {
			int diff = (int)i - (int)j;
			dist->distances[i * size + j] = diff < 0 ? -diff : diff;
		}

	return DSTM_SUCCESS;
}

/**
 * Create 2D mesh/grid topology distance matrix.
 *
 * In a 2D mesh, nodes are arranged in a rectangular grid with each interior
 * node connected to its 4 neighbors (up, down, left, right). Edge nodes have
 * fewer connections. The distance is the Manhattan distance on the grid.
 *
 * Grid dimensions are automatically computed to be as square as possible.
 * For example, 12 nodes creates a 3x4 grid, 16 nodes creates a 4x4 grid.
 *
 * Example for 12 nodes (3x4 grid):
 *   0 -- 1 -- 2 -- 3
 *   |    |    |    |
 *   4 -- 5 -- 6 -- 7
 *   |    |    |    |
 *   8 -- 9 --10 --11
 *
 * Node indexing: node i is at position (i / cols, i % cols)
 *
 * Distance is Manhattan: |row_i - row_j| + |col_i - col_j|
 *
 * @param[out] ret: pointer to distance structure pointer (must be non-NULL)
 * @param[in] size: number of nodes (must be >= 1)
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL for invalid parameters,
 *         -DSTM_ENOMEM for allocation failures
 */
int dstm_distance_create_mesh(dstm_distance_t **ret, size_t size)
{
	int err = dstm_distance_alloc(ret, size, 1);
	if (err)
		return err;

	dstm_distance_t *dist = *ret;

	/* Compute grid dimensions (as square as possible) */
	size_t cols = size;
	for (size_t r = 1; r * r <= size; r++) {
		if (size % r == 0)
			cols = size / r;
	}

	/* Compute Manhattan distances */
	for (size_t i = 0; i < size; i++) {
		size_t row_i = i / cols;
		size_t col_i = i % cols;
		for (size_t j = 0; j < size; j++) {
			size_t row_j = j / cols;
			size_t col_j = j % cols;
			int row_dist = (int)row_i - (int)row_j;
			int col_dist = (int)col_i - (int)col_j;
			dist->distances[i * size + j] =
			        (row_dist < 0 ? -row_dist : row_dist) +
			        (col_dist < 0 ? -col_dist : col_dist);
		}
	}

	return DSTM_SUCCESS;
}

/**
 * Create 2D torus topology distance matrix.
 *
 * In a 2D torus, nodes are arranged in a rectangular grid with wraparound
 * edges, so each node has exactly 4 neighbors. This is like a mesh where
 * the edges wrap around (periodic boundary conditions).
 *
 * Grid dimensions are automatically computed to be as square as possible.
 * The distance is the shortest Manhattan distance considering wraparound.
 *
 * Example for 12 nodes (3x4 torus):
 *   The grid wraps: leftmost column connects to rightmost,
 *   top row connects to bottom row.
 *
 * Distance: min over both direct and wraparound paths
 *
 * @param[out] ret: pointer to distance structure pointer (must be non-NULL)
 * @param[in] size: number of nodes (must be >= 4 for a valid torus)
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL for invalid parameters,
 *         -DSTM_ENOMEM for allocation failures
 */
int dstm_distance_create_torus(dstm_distance_t **ret, size_t size)
{
	int err = dstm_distance_alloc(ret, size, 4);
	if (err)
		return err;

	dstm_distance_t *dist = *ret;

	/* Compute grid dimensions (as square as possible) */
	size_t rows = 1, cols = size;
	for (size_t r = 1; r * r <= size; r++) {
		if (size % r == 0) {
			rows = r;
			cols = size / r;
		}
	}

	/* Compute toroidal distances (with wraparound) */
	for (size_t i = 0; i < size; i++) {
		size_t row_i = i / cols;
		size_t col_i = i % cols;
		for (size_t j = 0; j < size; j++) {
			size_t row_j = j / cols;
			size_t col_j = j % cols;

			/* Compute distance with wraparound */
			int row_diff = (int)row_i - (int)row_j;
			int col_diff = (int)col_i - (int)col_j;
			if (row_diff < 0)
				row_diff = -row_diff;
			if (col_diff < 0)
				col_diff = -col_diff;

			/* Take shorter path (direct or wraparound) */
			int row_dist = row_diff < (int)rows - row_diff ?
			                       row_diff :
			                       (int)rows - row_diff;
			int col_dist = col_diff < (int)cols - col_diff ?
			                       col_diff :
			                       (int)cols - col_diff;

			dist->distances[i * size + j] = row_dist + col_dist;
		}
	}

	return DSTM_SUCCESS;
}

/**
 * Compute the tree distance between two nodes in a binary tree.
 *
 * The distance is the path length through the lowest common ancestor (LCA).
 * Node i has children at (2*i + 1) and (2*i + 2), parent at (i-1)/2.
 *
 * @param i: first node index
 * @param j: second node index
 * @return distance between nodes i and j
 */
static int tree_distance(size_t i, size_t j)
{
	if (i == j)
		return 0;

	/* Build path from i to root */
	size_t path_i[64];
	size_t len_i = 0;
	size_t node = i;
	while (len_i < 64) {
		path_i[len_i++] = node;
		if (node == 0)
			break;
		node = (node - 1) / 2;
	}

	/* Build path from j to root */
	size_t path_j[64];
	size_t len_j = 0;
	node = j;
	while (len_j < 64) {
		path_j[len_j++] = node;
		if (node == 0)
			break;
		node = (node - 1) / 2;
	}

	/* Find LCA (first common node) */
	for (size_t pi = 0; pi < len_i; pi++)
		for (size_t pj = 0; pj < len_j; pj++)
			if (path_i[pi] == path_j[pj])
				return (int)pi + (int)pj;

	return 0;
}

/**
 * Create binary tree topology distance matrix.
 *
 * In a binary tree, nodes are arranged hierarchically with node 0 as root.
 * Node i has children at indices (2*i + 1) and (2*i + 2), and parent at
 * index (i-1)/2. Distance is the path length through the tree.
 *
 * Example for 7 nodes (complete binary tree):
 *         0
 *        / \
 *       1   2
 *      / \ / \
 *     3 4 5 6
 *
 * Distance between nodes requires going up to lowest common ancestor (LCA),
 * then down to the target.
 *
 * @param[out] ret: pointer to distance structure pointer (must be non-NULL)
 * @param[in] size: number of nodes (must be >= 1)
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL for invalid parameters,
 *         -DSTM_ENOMEM for allocation failures
 */
int dstm_distance_create_tree(dstm_distance_t **ret, size_t size)
{
	int err = dstm_distance_alloc(ret, size, 1);
	if (err)
		return err;

	dstm_distance_t *dist = *ret;

	for (size_t i = 0; i < size; i++)
		for (size_t j = 0; j < size; j++)
			dist->distances[i * size + j] = tree_distance(i, j);

	return DSTM_SUCCESS;
}

/**
 * Create hypercube topology distance matrix.
 *
 * In a hypercube, size must be a power of 2 (2^d for d dimensions).
 * Each node is connected to d neighbors (one per dimension).
 * Distance is the Hamming distance between node indices.
 *
 * Example for 8 nodes (3D hypercube):
 *   Nodes labeled 000, 001, 010, 011, 100, 101, 110, 111
 *   Distance = number of bit positions that differ
 *
 * Properties:
 *   - 8 nodes = 3D cube (each node has 3 neighbors)
 *   - 16 nodes = 4D hypercube (each node has 4 neighbors)
 *
 * @param[out] ret: pointer to distance structure pointer (must be non-NULL)
 * @param[in] size: number of nodes (must be power of 2, >= 2)
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL for invalid parameters,
 *         -DSTM_ENOMEM for allocation failures
 */
int dstm_distance_create_hypercube(dstm_distance_t **ret, size_t size)
{
	if (ret == NULL || (size & (size - 1)) != 0) {
		dstm_errno = DSTM_EINVAL;
		return -DSTM_EINVAL;
	}

	int err = dstm_distance_alloc(ret, size, 2);
	if (err)
		return err;

	dstm_distance_t *dist = *ret;

	/* Compute Hamming distances */
	for (size_t i = 0; i < size; i++) {
		for (size_t j = 0; j < size; j++) {
			/* Distance = popcount(i XOR j) */
			size_t xor_val = i ^ j;
			int hamming_dist = 0;
			while (xor_val) {
				hamming_dist += xor_val & 1;
				xor_val >>= 1;
			}
			dist->distances[i * size + j] = hamming_dist;
		}
	}

	return DSTM_SUCCESS;
}

/**
 * Create dragonfly topology distance matrix.
 *
 * Dragonfly is a hierarchical topology used in modern HPC systems.
 * Nodes are organized into groups, with high-bandwidth local connections
 * within groups and sparse global connections between groups.
 *
 * The size is automatically factored into groups. Each group is fully
 * connected internally (complete graph), and groups are connected via
 * a subset of global links.
 *
 * Distance properties:
 *   - Within same group: 1 hop
 *   - Between different groups: 2-3 hops (via global links)
 *
 * For simplicity, we use a minimal dragonfly where each group connects
 * to all other groups through at least one global link.
 *
 * @param[out] ret: pointer to distance structure pointer (must be non-NULL)
 * @param[in] size: number of nodes (must be >= 4)
 * @return DSTM_SUCCESS on success, -DSTM_EINVAL for invalid parameters,
 *         -DSTM_ENOMEM for allocation failures
 */
int dstm_distance_create_dragonfly(dstm_distance_t **ret, size_t size)
{
	int err = dstm_distance_alloc(ret, size, 4);
	if (err)
		return err;

	dstm_distance_t *dist = *ret;

	/* Determine number of groups (aim for sqrt(size) nodes per group) */
	size_t group_size = 1;
	for (size_t gs = 2; gs * gs <= size; gs++) {
		if (size % gs == 0)
			group_size = gs;
	}

	/* Compute dragonfly distances */
	for (size_t i = 0; i < size; i++) {
		size_t group_i = i / group_size;
		for (size_t j = 0; j < size; j++) {
			size_t group_j = j / group_size;

			if (i == j) {
				dist->distances[i * size + j] = 0;
			} else if (group_i == group_j) {
				/* Same group: 1 hop (intra-group complete
				 * graph) */
				dist->distances[i * size + j] = 1;
			} else {
				/* Different groups: 3 hops
				 * (local -> global link -> remote global link
				 * -> remote node) In a real dragonfly this can
				 * be 2 hops if nodes have direct global links,
				 * but 3 is worst case
				 */
				dist->distances[i * size + j] = 3;
			}
		}
	}

	return DSTM_SUCCESS;
}

/*******************************************************************************
 * Cleanup
 ******************************************************************************/

/**
 * Destroy a distance matrix and free all associated resources.
 *
 * This function safely deallocates a distance matrix created by any of
 * the dstm_distance_create_* functions. It is safe to call this function
 * with a NULL pointer or on an already-freed distance matrix.
 *
 * After this call, *dist will be set to NULL.
 *
 * @param[in,out] dist: pointer to distance structure pointer
 */
void dstm_distance_destroy(dstm_distance_t **dist)
{
	if (dist == NULL || *dist == NULL)
		return;

	free((*dist)->distances);
	free(*dist);
	*dist = NULL;
}

/*******************************************************************************
 * Extra functions
 ******************************************************************************/

int dstm_distance_create_from_matrix(dstm_distance_t **ret, size_t size,
                                     const int *matrix)
{
	if (ret == NULL || matrix == NULL)
		goto err_inval;

	if (size < 1 || size > DSTM_DISTANCE_MAX_SIZE)
		goto err_inval;

	/* Validate: diagonal must be zero, all entries non-negative. */
	for (size_t i = 0; i < size; i++) {
		if (matrix[i * size + i] != 0)
			goto err_inval;
		for (size_t j = 0; j < size; j++) {
			if (matrix[i * size + j] < -1) //-1 represents two ranks with no edge
				goto err_inval;
		}
	}

	dstm_distance_t *dist = malloc(sizeof(*dist));
	if (dist == NULL)
		goto err_nomem;

	dist->distances = calloc(size * size, sizeof(int));
	if (dist->distances == NULL)
		goto err_free_dist;

	dist->size = size;
	memcpy(dist->distances, matrix, size * size * sizeof(int));
	*ret = dist;
	return DSTM_SUCCESS;

err_free_dist:
	free(dist);
err_nomem:
	dstm_errno = DSTM_ENOMEM;
	return -DSTM_ENOMEM;
err_inval:
	dstm_errno = DSTM_EINVAL;
	return -DSTM_EINVAL;
}

int dstm_distance_create_copy(dstm_distance_t **dst, dstm_distance_t *src)
{
	if (dst == NULL || src == NULL || src->size == 0 ||
	    src->distances == NULL)
		return -DSTM_EINVAL;

	size_t size = src->size;
	dstm_distance_t *dist = malloc(sizeof(*dist));
	if (dist == NULL)
		return -DSTM_ENOMEM;

	dist->distances = calloc(size * size, sizeof(int));
	if (dist->distances == NULL) {
		free(dist);
		return -DSTM_ENOMEM;
	}

	dist->size = size;
	memcpy(dist->distances, src->distances,
	       dist->size * dist->size * sizeof(int));
	*dst = dist;
	return DSTM_SUCCESS;
}

int dstm_distance_combine(dstm_distance_t **ret,
                          const dstm_distance_t *const *mats,
                          size_t n)
{
	if (ret == NULL || mats == NULL || n == 0)
		goto err_inval;

	*ret = NULL;

	/* All inputs must be non-NULL and share a common size. */
	const dstm_distance_t *first = mats[0];
	if (first == NULL || first->distances == NULL || first->size == 0)
		goto err_inval;

	size_t size = first->size;
	for (size_t k = 1; k < n; k++) {
		if (mats[k] == NULL || mats[k]->distances == NULL ||
		    mats[k]->size != size)
			goto err_inval;
	}

	int *matrix = malloc(size * size * sizeof(int));
	if (matrix == NULL)
		goto err_nomem;

    /* Per-cell union reduction.  Only -1 is the "no edge / unreachable"
	 * sentinel: a distance of 0 is a LEGITIMATE edge (all-to-all sync at
	 * every timestep) and must survive the union, otherwise a structure
	 * that uses distance 0 across a pair would see that pair marked
	 * unreachable in the session matrix and have its versions evicted
	 * prematurely.  We take the minimum over all real (>= 0) values so the
	 * result reflects the most-constrained (tightest) synchronization any
	 * structure requires.  The result is -1 only when EVERY input is -1. */
	for (size_t i = 0; i < size; i++) {
		for (size_t j = 0; j < size; j++) {
			int v;

			if (i == j) {
				v = 0;
			} else {
				int best = -1; /* -1 == no real edge seen yet */

				for (size_t k = 0; k < n; k++) {
					int cur = mats[k]->distances[i * size
					                             + j];
					if (cur < 0)
						continue; /* -1 = unreachable */
					if (best < 0 || cur < best)
						best = cur;
				}

				v = best; /* -1 if no input had a real edge */
			}

			matrix[i * size + j] = v;
		}
	}

	int rc = dstm_distance_create_from_matrix(ret, size, matrix);
	free(matrix);
	return rc;

err_nomem:
	dstm_errno = DSTM_ENOMEM;
	return -DSTM_ENOMEM;
err_inval:
	dstm_errno = DSTM_EINVAL;
	return -DSTM_EINVAL;
}

int dstm_distance_get(dstm_distance_t *dst, int i, int j)
{
	return dst->distances[dst->size * i + j];
}

size_t dstm_distance_size(dstm_distance_t *dst)
{
	return dst->size;
}

/*******************************************************************************
 * Formatting helpers
 ******************************************************************************/

json_t *dstm_distance_to_json(dstm_distance_t *d)
{
	if (d == NULL)
		return json_null();

	json_t *obj = json_pack("{s:I}", "size", (json_int_t)d->size);

	/* Include the first row of the distance matrix as a representative
	 * sample.  Including the full NxN matrix would be excessive for
	 * non-trivial process counts. */
	if (d->distances != NULL && d->size > 0) {
		json_t *row = json_array();
		for (size_t j = 0; j < d->size; j++)
			json_array_append_new(row, json_integer(d->distances[j]));
		json_object_set_new(obj, "row0", row);
	}

	return obj;
}
