#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <dstm.h>

#include "airport.h"

/*
 * Airport torus benchmark on dstm_dgraph.
 *
 * Airports are vertices, corridors are edges, in-flight manifests are edge
 * state, airport load is vertex state. No internal DSTM headers required:
 * the rank-level distance matrix is derived from the same CSR arrays that
 * define the graph.
 */

static void usage(const char *prog)
{
	fprintf(stderr,
	"usage: %s [options]\n"
	"  --seed S        RNG seed (default 42)\n"
	"  --time T        timesteps (default 16)\n"
	"  --na N          airports per rank; N * num_ranks must be a\n"
	"                  perfect square >= 4 (default 1)\n"
	"  --planes P      initial planes per airport (default 1)\n"
	"  --congestion    route toward the least-loaded neighbor using\n"
	"                  remote vertex-state reads (default: uniform random)\n"
	"  --no-vstate     skip publishing vertex state each tick\n"
	"  --region-mb M   force local region size in MiB (default: derived)\n"
	"  --quiet | --info\n",
	prog);
}

static size_t next_pow2(size_t x)
{
	size_t p = 1;
	while (p < x)
		p <<= 1;
	return p;
}

int main(int argc, char *argv[])
{
	int seed       = 42;
	int quiet      = 0;
	int timesteps  = 16;
	int na         = 1;
	int planes     = 1;
	int congestion = 0;
	int publish    = 1;
	long region_mb = 0;

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--seed") && i + 1 < argc)
			seed = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--time") && i + 1 < argc)
			timesteps = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--na") && i + 1 < argc)
			na = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--planes") && i + 1 < argc)
			planes = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--region-mb") && i + 1 < argc)
			region_mb = atol(argv[++i]);
		else if (!strcmp(argv[i], "--congestion"))
			congestion = 1;
		else if (!strcmp(argv[i], "--no-vstate"))
			publish = 0;
		else if (!strcmp(argv[i], "--quiet"))
			quiet = 1;
		else if (!strcmp(argv[i], "--info"))
			quiet = 2;
		else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
			usage(argv[0]);
			return 0;
		}
	}
	if (na < 1)     na = 1;
	if (planes < 1) planes = 1;
	if (timesteps < 1) timesteps = 1;
    if (congestion) publish = 1;

	int rank, num_ranks;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

	dstm_init(&argc, &argv);
	dstm_log_init(stdout, "airport_dgraph");
	dstm_log_setlevel(quiet == 0 ? DSTM_LOG_NORMAL
	                 : quiet == 2 ? DSTM_LOG_INFO
	                              : DSTM_LOG_QUIET);

	int total_airports = na * num_ranks;
	int grid = (int)(sqrt((double)total_airports) + 0.5);
	if (grid * grid != total_airports || grid < 2) {
		if (rank == 0)
			fprintf(stderr,
			        "ERROR: na * num_ranks = %d must be a perfect "
			        "square >= 4 (e.g. na=1024 np=1, na=64 np=16).\n",
			        total_airports);
		dstm_finalize();
		MPI_Finalize();
		return -1;
	}

	/* --- 1. Local CSR slice ------------------------------------------
	 * Built before dstm_start, because the distance matrix that starts
	 * the session is derived from it. */
	airport_topology_t topo;
	if (airport_topology_build(&topo, rank, na, grid) != 0) {
		dstm_log_error("topology build failed");
		dstm_finalize();
		MPI_Finalize();
		return -1;
	}

	/* --- 2. Distance matrix straight from the topology ----------------
	 * Replaces build_rank_adjacency(): same result (delta 1 between ranks
	 * sharing a corridor, -1 otherwise, 0 on the diagonal) with no
	 * internal/distance.h and no hand-walked global airport loop.
	 *
	 * Note the ordering constraint: dstm_dgraph_create() publishes state
	 * and therefore needs a live session, so the graph's own
	 * derive_distance() can never be the matrix that starts its session.
	 * This entry point exists for exactly that reason. */
	dstm_distance_t *distance = NULL;
	int status = dstm_dgraph_distance_from_topology(&distance,
	                                                (size_t)na,
	                                                topo.out_degrees,
	                                                topo.edge_dsts,
	                                                1);
	if (status != DSTM_SUCCESS) {
		dstm_log_error("distance_from_topology failed: %d", status);
		goto out_topo;
	}

	/* --- 3. Region sizing ---------------------------------------------
	 * DSTM keeps a full-size buffer per version, so the region must hold
	 * the whole graph. Undersizing shows up as an ENOMEM out of
	 * dstm_dgraph_create, not as a slow run. */
	{
		size_t need = airport_topology_region_bytes(
			&topo, sizeof(airport_vertex_t), sizeof(gate_slot_t));
		size_t want = region_mb > 0
			    ? (size_t)region_mb * 1024u * 1024u
			    : next_pow2(2 * need + (1u << 16));
		if (want < (1u << 16))
			want = 1u << 16;

		unsigned long long lw = (unsigned long long)want, gw = 0;
		MPI_Allreduce(&lw, &gw, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
		              MPI_COMM_WORLD);

		if (rank == 0) {
			printf("Airport torus (dgraph): %d airports (%dx%d), "
			       "%d ranks, na=%d, planes/airport=%d, "
			       "timesteps=%d\n"
			       "  vertex state %zu B, edge state %zu B "
			       "(cap %d), graph %.2f MiB/rank, "
			       "region %.0f MiB/rank%s\n",
			       total_airports, grid, grid, num_ranks, na,
			       planes, timesteps,
			       sizeof(airport_vertex_t), sizeof(gate_slot_t),
			       EDGE_GATE_CAPACITY,
			       (double)need / (1024.0 * 1024.0),
			       (double)gw / (1024.0 * 1024.0),
			       congestion ? ", congestion routing" : "");
		}

		dstm_opts_t opts = dstm_default_options;
		opts.region_size = (size_t)gw;

		status = dstm_start_custom(distance, opts);
		if (status != DSTM_SUCCESS) {
			dstm_log_error("dstm_start_custom failed: %d", status);
			goto out_dist;
		}
	}

	/* --- 4. Build the graph -------------------------------------------
	 * local_vertices == na on every rank, so contiguous rank-order
	 * partitioning gives vertex_offset == rank * na, matching the torus
	 * ids we generated. */
	airport_vertex_t *v0 = calloc((size_t)na, sizeof(airport_vertex_t));
	for (int l = 0; l < na; l++)
		v0[l].planes_on_ground = planes;

	dstm_dgraph_t *g = NULL;
	status = dstm_dgraph_create(&g,
	                            sizeof(airport_vertex_t),
	                            sizeof(gate_slot_t),
	                            (size_t)na,
	                            topo.out_degrees,
	                            topo.edge_dsts,
	                            v0,
	                            NULL /* edges start empty */);
	free(v0);
	if (status != DSTM_SUCCESS) {
		dstm_log_error("dgraph_create failed: %d", status);
		goto out_fini;
	}

	if (rank == 0 && quiet != 1) {
		size_t nranks = 0;
		dstm_dgraph_neighbor_ranks_get(g, NULL, &nranks);
		printf("  graph: %zu vertices, %zu edges, "
		       "%zu neighbor ranks on rank 0\n",
		       dstm_dgraph_vertex_count(g),
		       dstm_dgraph_edge_count(g), nranks);
	}

	airport_state_t state;
	if (airport_init(&state, rank, num_ranks, planes, seed,
	                 congestion, publish, &topo, g) != 0) {
		dstm_log_error("airport_init failed");
		goto out_graph;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double t_start = MPI_Wtime();

	for (dstm_time_t t = 0; t < timesteps; t++)
		airport_timestep(&state, t);

	dstm_time_t last_time = timesteps + 1;
	if (dstm_barrier(last_time) != DSTM_SUCCESS)
		dstm_log_error("barrier failed");

	double elapsed_local = MPI_Wtime() - t_start;

	/* --- 5. Reduce ---------------------------------------------------- */
	{
		long long my_land = 0, my_ground = 0, my_pending = 0;
		for (int l = 0; l < na; l++) {
			my_land    += state.landings[l];
			my_ground  += state.planes_on_ground[l];
			my_pending += state.pending_landings[l];
		}

		long long lsend[6] = { my_land, state.departures,
		                       state.arrivals, my_ground + my_pending,
		                       state.overflows, state.misroutes };
		long long lrecv[6] = { 0 };
		double elapsed = 0.0;

		MPI_Reduce(&elapsed_local, &elapsed, 1, MPI_DOUBLE, MPI_MAX,
		           0, MPI_COMM_WORLD);
		MPI_Reduce(lsend, lrecv, 6, MPI_LONG_LONG, MPI_SUM,
		           0, MPI_COMM_WORLD);

		if (rank == 0) {
			long long in_flight = lrecv[1] - lrecv[2];
			long long accounted = lrecv[3] + in_flight;
			long long expected  = (long long)total_airports * planes;
			double tsps = elapsed > 0.0 ? timesteps / elapsed : 0.0;

			fprintf(stdout,
			"\n=============== RESULTS ===============\n"
			"Backend:            dstm_dgraph\n"
			"Airports:           %d  (%d x %d torus)\n"
			"Corridors (edges):  %zu\n"
			"Ranks:              %d   (na=%d)\n"
			"Planes per airport: %d\n"
			"Routing:            %s\n"
			"Timesteps:          %d\n"
			"Total departures:   %lld\n"
			"Total landings:     %lld\n"
			"Gate overflows:     %lld\n"
			"Misrouted flights:  %lld\n"
			"Conservation:       %lld / %lld  %s\n"
			"\n"
			"Elapsed time:       %.4f seconds\n"
			"Timesteps/second:   %.2f\n"
			"Edge ops/second:    %.2f\n"
			"========================================\n",
			total_airports, grid, grid,
			dstm_dgraph_edge_count(g),
			num_ranks, na, planes,
			congestion ? "least-loaded neighbor (t-1)"
			           : "uniform random",
			timesteps, lrecv[1], lrecv[0], lrecv[4], lrecv[5],
			accounted, expected,
			accounted == expected ? "OK" : "MISMATCH",
			elapsed, tsps,
			elapsed > 0.0
			    ? (double)dstm_dgraph_edge_count(g) * timesteps
			      * 2.0 / elapsed
			    : 0.0);
		}
	}

out_graph:
	dstm_dgraph_destroy(&g);
out_fini:
	if (dstm_fini(timesteps + 1) != DSTM_SUCCESS)
		dstm_log_error("dstm_fini failed");
out_dist:
	dstm_distance_destroy(&distance);
out_topo:
	airport_topology_destroy(&topo);
	dstm_finalize();
	MPI_Finalize();
	return status == DSTM_SUCCESS ? 0 : -1;
}
