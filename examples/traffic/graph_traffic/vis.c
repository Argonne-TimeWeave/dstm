#include "vis.h"

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
 * Common visualization helpers
 ******************************************************************************/

void vis_init(struct vis *vis)
{
        if (vis == NULL)
                return;

        memset(vis, 0, sizeof(*vis));
}

const char *vis_corner_name(int corner)
{
        switch (corner) {
        case 0:
                return "southwest";
        case 1:
                return "northwest";
        case 2:
                return "southeast";
        case 3:
                return "northeast";
        default:
                return "unknown";
        }
}

/*******************************************************************************
 * Replica staleness snapshot
 *
 * A congestion replica contains one timestamp per directed edge rather than
 * one timestamp per vertex. The dump therefore contains:
 *
 *   edge records: exact stamp, age, and stored value
 *   node records: oldest known outgoing-edge stamp for each vertex
 *
 * Node age is the maximum age among the node's known outgoing edges.
 * known_edges/total_edges allows consumers to distinguish a completely
 * observed node from a partially observed node.
 *
 * A stamp or age of -1 means no information has been observed.
 ******************************************************************************/

void vis_write_staleness_snapshot(const struct vis *vis,
                                  const struct city *city,
                                  const struct cong_replica *replica,
                                  int rank,
                                  int32_t t,
                                  const char *replica_mode)
{
        char path[4096];

        if (vis == NULL || city == NULL || replica == NULL)
                return;

        if (!vis->snapshot_enabled ||
            t != vis->snapshot_tick ||
            rank != 0)
                return;

        if (replica_mode == NULL)
                replica_mode = "unknown";

        snprintf(path, sizeof(path),
                 "staleness-%s-t%d-rank0.csv",
                 replica_mode, (int)t);

        FILE *f = fopen(path, "w");
        if (f == NULL) {
                perror(path);
                return;
        }

        /*
         * Node records carry the authoritative dense-ID mapping and
         * coordinates. Edge records use the same dense IDs.
         */
        fprintf(f,
                "record_type,"
                "node_id,original_node_id,x,y,owner,"
                "edge_id,src,dst,"
                "stamp,age,value,"
                "known_edges,total_edges,partially_known\n");

        for (size_t u = 0; u < city->nv; u++) {
                int32_t oldest_stamp = -1;
                int64_t maximum_age = -1;
                size_t known_edges = 0;
                size_t total_edges =
                        city->row_ptr[u + 1] - city->row_ptr[u];

                for (size_t ge = city->row_ptr[u];
                     ge < city->row_ptr[u + 1];
                     ge++) {
                        const struct cong_entry *entry =
                                &replica->e[ge];

                        if (entry->stamp < 0)
                                continue;

                        int64_t age =
                                (int64_t)t -
                                (int64_t)entry->stamp;

                        known_edges++;

                        if (oldest_stamp < 0 ||
                            entry->stamp < oldest_stamp)
                                oldest_stamp = entry->stamp;

                        if (maximum_age < 0 || age > maximum_age)
                                maximum_age = age;
                }

                int partially_known =
                        known_edges < total_edges;

                fprintf(f,
                        "node,"
                        "%zu,%ld,%.17g,%.17g,%d,"
                        "-1,-1,-1,"
                        "%d,%" PRId64 ",nan,"
                        "%zu,%zu,%d\n",
                        u,
                        city->dense2orig[u],
                        city->x[u],
                        city->y[u],
                        city_owner(city, u),
                        (int)oldest_stamp,
                        maximum_age,
                        known_edges,
                        total_edges,
                        partially_known);
        }

        /*
         * Emit every directed graph edge. src and dst are dense IDs and
         * therefore index directly into the node records above.
         */
        for (size_t ge = 0; ge < city->ne; ge++) {
                const struct cong_entry *entry =
                        &replica->e[ge];

                int known = entry->stamp >= 0;
                int64_t age = known
                        ? (int64_t)t -
                          (int64_t)entry->stamp
                        : -1;

                fprintf(f,
                        "edge,"
                        "-1,-1,nan,nan,-1,"
                        "%zu,%zu,%zu,"
                        "%d,%" PRId64 ",%.9g,"
                        "%d,1,%d\n",
                        ge,
                        city->esrc[ge],
                        city->dst[ge],
                        (int)entry->stamp,
                        age,
                        entry->value,
                        known,
                        known ? 0 : 1);
        }

        if (fclose(f) != 0) {
                perror(path);
                return;
        }

        printf("staleness snapshot: %s\n", path);
        fflush(stdout);
}

/*******************************************************************************
 * Traffic-visualization viewpoints
 *
 * Select the vertex nearest each corner of the coordinate bounding box.
 * Selection is deterministic and performed independently by every rank over
 * the replicated city data.
 ******************************************************************************/

int vis_select_traffic_corners(struct vis *vis,
                               const struct city *city,
                               int rank)
{
        if (vis == NULL || city == NULL)
                return -1;

        if (!vis->traffic_enabled)
                return 0;

        if (city->nv < VIS_TRAFFIC_CORNERS) {
                if (rank == 0) {
                        fprintf(stderr,
                                "traffic visualization requires at least "
                                "%d vertices\n",
                                VIS_TRAFFIC_CORNERS);
                }
                return -1;
        }

        double xmin = city->x[0];
        double xmax = city->x[0];
        double ymin = city->y[0];
        double ymax = city->y[0];

        for (size_t v = 1; v < city->nv; v++) {
                if (city->x[v] < xmin)
                        xmin = city->x[v];
                if (city->x[v] > xmax)
                        xmax = city->x[v];
                if (city->y[v] < ymin)
                        ymin = city->y[v];
                if (city->y[v] > ymax)
                        ymax = city->y[v];
        }

        double xspan = xmax - xmin;
        double yspan = ymax - ymin;

        if (!(xspan > 0.0))
                xspan = 1.0;
        if (!(yspan > 0.0))
                yspan = 1.0;

        const double target_x[VIS_TRAFFIC_CORNERS] = {
                xmin, xmin, xmax, xmax
        };

        const double target_y[VIS_TRAFFIC_CORNERS] = {
                ymin, ymax, ymin, ymax
        };

        unsigned char *selected =
                calloc(city->nv, sizeof(*selected));

        if (selected == NULL)
                return -1;

        for (int corner = 0;
             corner < VIS_TRAFFIC_CORNERS;
             corner++) {
                size_t best = (size_t)-1;
                double best_distance = INFINITY;

                for (size_t v = 0; v < city->nv; v++) {
                        if (selected[v])
                                continue;

                        double dx =
                                (city->x[v] - target_x[corner]) /
                                xspan;

                        double dy =
                                (city->y[v] - target_y[corner]) /
                                yspan;

                        double distance =
                                dx * dx + dy * dy;

                        if (best == (size_t)-1 ||
                            distance < best_distance ||
                            (distance == best_distance &&
                             v < best)) {
                                best = v;
                                best_distance = distance;
                        }
                }

                assert(best != (size_t)-1);

                selected[best] = 1;
                vis->traffic_node[corner] = best;
                vis->traffic_owner[corner] =
                        city_owner(city, best);
        }

        free(selected);

        if (rank == 0) {
                printf("=== Traffic visualization viewpoints ===\n");

                for (int corner = 0;
                     corner < VIS_TRAFFIC_CORNERS;
                     corner++) {
                        size_t v = vis->traffic_node[corner];

                        printf("  %-9s: dense=%zu original=%ld "
                               "owner=%d x=%.3f y=%.3f\n",
                               vis_corner_name(corner),
                               v,
                               city->dense2orig[v],
                               vis->traffic_owner[corner],
                               city->x[v],
                               city->y[v]);
                }

                printf("  warmup timestamps: [0, %d)\n",
                       (int)vis->traffic_warmup);
                printf("==========================================\n");
                fflush(stdout);
        }

        return 0;
}

/*******************************************************************************
 * Traffic-visualization files
 ******************************************************************************/

static void close_opened_traffic_files(struct vis *vis)
{
        for (int corner = 0;
             corner < VIS_TRAFFIC_CORNERS;
             corner++) {
                FILE *f = vis->traffic_file[corner];

                if (f == NULL)
                        continue;

                fclose(f);
                vis->traffic_file[corner] = NULL;
        }
}

int vis_open_traffic_files(struct vis *vis,
                           const struct city *city,
                           int rank,
                           const char *replica_mode,
                           int stale_delay)
{
        int local_error = 0;

        if (vis == NULL || city == NULL)
                return -1;

        if (!vis->traffic_enabled)
                return 0;

        if (replica_mode == NULL)
                replica_mode = "unknown";

        for (int corner = 0;
             corner < VIS_TRAFFIC_CORNERS;
             corner++) {
                if (vis->traffic_owner[corner] != rank)
                        continue;

                size_t viewpoint =
                        vis->traffic_node[corner];

                char path[4096];

                snprintf(path, sizeof(path),
                         "traffic-view-%s-node%zu-rank%d-%s-d%d.csv",
                         vis_corner_name(corner),
                         viewpoint,
                         rank,
                         replica_mode,
                         stale_delay);

                FILE *f = fopen(path, "w");
                if (f == NULL) {
                        perror(path);
                        local_error = 1;
                        continue;
                }

                vis->traffic_file[corner] = f;

                /*
                 * Node rows describe topology and viewpoint metadata.
                 * Edge rows describe perceived congestion at one tick.
                 */
                fprintf(f,
                        "record_type,tick,"
                        "corner,view_node,view_original_id,view_rank,"
                        "node_id,original_node_id,x,y,node_owner,"
                        "edge_id,src,dst,src_owner,dst_owner,"
                        "stamp,age,known,"
                        "congestion_ewma,effective_congestion,"
                        "capacity,volume_capacity_ratio,"
                        "free_flow_ticks,route_penalty,"
                        "perceived_route_cost\n");

                /*
                 * Topology does not change, so each node is written once.
                 * tick=-1 identifies static topology records.
                 */
                for (size_t node = 0;
                     node < city->nv;
                     node++) {
                        fprintf(f,
                                "node,-1,"
                                "%s,%zu,%ld,%d,"
                                "%zu,%ld,%.17g,%.17g,%d,"
                                "-1,-1,-1,-1,-1,"
                                "-1,-1,0,"
                                "nan,nan,nan,nan,"
                                "nan,nan,nan\n",
                                vis_corner_name(corner),
                                viewpoint,
                                city->dense2orig[viewpoint],
                                rank,
                                node,
                                city->dense2orig[node],
                                city->x[node],
                                city->y[node],
                                city_owner(city, node));
                }

                fflush(f);

                printf("rank %d traffic visualization: %s\n",
                       rank, path);
        }

        /*
         * Ensure every rank makes the same setup decision. Otherwise one
         * rank could leave setup while other ranks continue into collective
         * or DSTM operations.
         */
        int global_error = 0;

        MPI_Allreduce(&local_error, &global_error,
                      1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (global_error) {
                close_opened_traffic_files(vis);
                return -1;
        }

        return 0;
}

/*******************************************************************************
 * Warmup traffic-visualization output
 *
 * Each selected viewpoint records its owner rank's congestion replica.
 * Call this after incoming replica merges and before routing so the recorded
 * replica is exactly the perspective used by the routing decision.
 ******************************************************************************/

void vis_write_traffic_tick(struct vis *vis,
                            const struct city *city,
                            const struct cong_replica *replica,
                            int rank,
                            int32_t t,
                            float cong_half_life,
                            float route_delay_factor)
{
        if (vis == NULL || city == NULL || replica == NULL)
                return;

        if (!vis->traffic_enabled ||
            t < 0 ||
            t >= vis->traffic_warmup)
                return;

        for (int corner = 0;
             corner < VIS_TRAFFIC_CORNERS;
             corner++) {
                FILE *f = vis->traffic_file[corner];

                if (f == NULL)
                        continue;

                size_t viewpoint =
                        vis->traffic_node[corner];

                for (size_t ge = 0;
                     ge < city->ne;
                     ge++) {
                        const struct cong_entry *entry =
                                &replica->e[ge];

                        size_t src = city->esrc[ge];
                        size_t dst = city->dst[ge];

                        int known = entry->stamp >= 0;

                        int64_t age = known
                                ? (int64_t)t -
                                  (int64_t)entry->stamp
                                : -1;

                        /*
                         * The stored value is the last observed EWMA.
                         * effective_congestion applies the same age decay
                         * used by the routing cost calculation.
                         */
                        float stored = known
                                ? entry->value
                                : NAN;

                        float effective = known
                                ? cong_effective(entry,
                                                 t,
                                                 cong_half_life,
                                                 0.0f)
                                : 0.0f;

                        float capacity =
                                city->cap_tick[ge];

                        float ratio =
                                effective / capacity;

                        float penalty =
                                ratio * route_delay_factor;

                        float perceived_cost =
                                (float)city->ff_ticks[ge] +
                                penalty;

                        fprintf(f,
                                "edge,%d,"
                                "%s,%zu,%ld,%d,"
                                "-1,-1,nan,nan,-1,"
                                "%zu,%zu,%zu,%d,%d,"
                                "%d,%" PRId64 ",%d,"
                                "%.9g,%.9g,%.9g,%.9g,"
                                "%d,%.9g,%.9g\n",
                                (int)t,
                                vis_corner_name(corner),
                                viewpoint,
                                city->dense2orig[viewpoint],
                                rank,
                                ge,
                                src,
                                dst,
                                city_owner(city, src),
                                city_owner(city, dst),
                                (int)entry->stamp,
                                age,
                                known,
                                stored,
                                effective,
                                capacity,
                                ratio,
                                city->ff_ticks[ge],
                                penalty,
                                perceived_cost);
                }

                /*
                 * Flush at frame boundaries so an interrupted run still
                 * contains complete animation frames.
                 */
                fflush(f);
        }
}

/*******************************************************************************
 * Teardown
 ******************************************************************************/

void vis_close(struct vis *vis, int rank)
{
        if (vis == NULL)
                return;

        for (int corner = 0;
             corner < VIS_TRAFFIC_CORNERS;
             corner++) {
                FILE *f = vis->traffic_file[corner];

                if (f == NULL)
                        continue;

                if (fclose(f) != 0) {
                        fprintf(stderr,
                                "rank %d: failed to close traffic "
                                "visualization file for %s\n",
                                rank,
                                vis_corner_name(corner));
                }

                vis->traffic_file[corner] = NULL;
        }
}
