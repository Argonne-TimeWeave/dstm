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
				if (ik != INT_MAX && kj != INT_MAX) {
					int via_k = ik + kj;
					if (via_k < distances[i * size + j])
						distances[i * size + j] = via_k;
				}
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

/*******************************************************************************
 * Common Topologies
 ******************************************************************************/

/**
 * Create a identity topology distance matrix
 *
 * Each node is connected to every other node at distance 1.
 */
int dstm_distance_create_identity(dstm_distance_t **ret, size_t size)
{
	if (ret == NULL) {
		return -DSTM_EINVAL;
	}

	if (size < 1 || size > DSTM_DISTANCE_MAX_SIZE) {
		return -DSTM_EINVAL;
	}

	dstm_distance_t *dist = malloc(sizeof(*dist));
	if (dist == NULL) {
		return -DSTM_ENOMEM;
	}

	dist->distances = calloc(size * size, sizeof(int));
	if (dist->distances == NULL) {
		free(dist);
		return -DSTM_ENOMEM;
	}

	dist->size = size;

	for (size_t i = 0; i < size * size; i++)
		dist->distances[i] = 1;

	*ret = dist;
	return DSTM_SUCCESS;
}

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
	if (ret == NULL) {
		dstm_errno = DSTM_EINVAL;
		return -DSTM_EINVAL;
	}

	if (size < 3 || size > DSTM_DISTANCE_MAX_SIZE) {
		dstm_errno = DSTM_EINVAL;
		return -DSTM_EINVAL;
	}

	dstm_distance_t *dist = malloc(sizeof(*dist));
	if (dist == NULL) {
		dstm_errno = DSTM_ENOMEM;
		return -DSTM_ENOMEM;
	}

	dist->distances = calloc(size * size, sizeof(int));
	if (dist->distances == NULL) {
		free(dist);
		dstm_errno = DSTM_ENOMEM;
		return -DSTM_ENOMEM;
	}

	dist->size = size;
	init_distance_matrix(dist->distances, size);

	/* Set ring edges (bidirectional) */
	for (size_t i = 0; i < size; i++) {
		size_t next = (i + 1) % size;
		size_t prev = (i + size - 1) % size;
		dist->distances[i * size + next] = 1;
		dist->distances[i * size + prev] = 1;
	}

	compute_shortest_paths(dist->distances, size);

	*ret = dist;
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
	if (ret == NULL) {
		dstm_errno = DSTM_EINVAL;
		return -DSTM_EINVAL;
	}

	if (size < 1 || size > DSTM_DISTANCE_MAX_SIZE) {
		dstm_errno = DSTM_EINVAL;
		return -DSTM_EINVAL;
	}

	dstm_distance_t *dist = malloc(sizeof(*dist));
	if (dist == NULL) {
		dstm_errno = DSTM_ENOMEM;
		return -DSTM_ENOMEM;
	}

	dist->distances = calloc(size * size, sizeof(int));
	if (dist->distances == NULL) {
		free(dist);
		dstm_errno = DSTM_ENOMEM;
		return -DSTM_ENOMEM;
	}

	dist->size = size;

	/* In a complete graph, all nodes are directly connected */
	for (size_t i = 0; i < size; i++)
		for (size_t j = 0; j < size; j++)
			dist->distances[i * size + j] = (i == j) ? 0 : 1;

	*ret = dist;
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
	if (ret == NULL) {
		dstm_errno = DSTM_EINVAL;
		return -DSTM_EINVAL;
	}

	if (size < 2 || size > DSTM_DISTANCE_MAX_SIZE) {
		dstm_errno = DSTM_EINVAL;
		return -DSTM_EINVAL;
	}

	dstm_distance_t *dist = malloc(sizeof(*dist));
	if (dist == NULL) {
		dstm_errno = DSTM_ENOMEM;
		return -DSTM_ENOMEM;
	}

	dist->distances = calloc(size * size, sizeof(int));
	if (dist->distances == NULL) {
		free(dist);
		dstm_errno = DSTM_ENOMEM;
		return -DSTM_ENOMEM;
	}

	dist->size = size;
	init_distance_matrix(dist->distances, size);

	/* Set edges: center (node 0) to all others */
	for (size_t i = 1; i < size; i++) {
		dist->distances[0 * size + i] = 1;
		dist->distances[i * size + 0] = 1;
	}

	compute_shortest_paths(dist->distances, size);

	*ret = dist;
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
	if (ret == NULL) {
		dstm_errno = DSTM_EINVAL;
		return -DSTM_EINVAL;
	}

	if (size < 1 || size > DSTM_DISTANCE_MAX_SIZE) {
		dstm_errno = DSTM_EINVAL;
		return -DSTM_EINVAL;
	}

	dstm_distance_t *dist = malloc(sizeof(*dist));
	if (dist == NULL) {
		dstm_errno = DSTM_ENOMEM;
		return -DSTM_ENOMEM;
	}

	dist->distances = calloc(size * size, sizeof(int));
	if (dist->distances == NULL) {
		free(dist);
		dstm_errno = DSTM_ENOMEM;
		return -DSTM_ENOMEM;
	}

	dist->size = size;

	/* For a line topology, distance is simply |i - j| */
	for (size_t i = 0; i < size; i++)
		for (size_t j = 0; j < size; j++) {
			int diff = (int)i - (int)j;
			dist->distances[i * size + j] = diff < 0 ? -diff : diff;
		}

	*ret = dist;
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

	if ((*dist)->distances != NULL)
		free((*dist)->distances);

	free(*dist);
	*dist = NULL;
}

/*******************************************************************************
 * Extra functions
 ******************************************************************************/

int dstm_distance_create_copy(dstm_distance_t **dst, dstm_distance_t *src)
{
	if (dst == NULL || src == NULL || src->size == 0 || src->distances == NULL)
		return -DSTM_EINVAL;

	size_t size = src->size;
	dstm_distance_t *dist = malloc(sizeof(*dist));
	if (dist == NULL) {
		return -DSTM_ENOMEM;
	}

	dist->distances = calloc(size * size, sizeof(int));
	if (dist->distances == NULL) {
		free(dist);
		return -DSTM_ENOMEM;
	}

	dist->size = size;

	memcpy(dist->distances, src->distances, dist->size * dist->size * sizeof(int));
	*dst = dist;
	return DSTM_SUCCESS;
}

int dstm_distance_get(dstm_distance_t *dst, int i, int j)
{
	return dst->distances[dst->size * i + j];
}

size_t dstm_distance_size(dstm_distance_t *dst)
{
	return dst->size;
}
