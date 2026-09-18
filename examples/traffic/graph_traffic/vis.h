#ifndef VIS_H
#define VIS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "chan.h"
#include "city.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VIS_TRAFFIC_CORNERS 4

/*
 * Visualization state.
 *
 * The configuration fields are public so command-line parsing can populate
 * them before setup:
 *
 *     vis.snapshot_enabled
 *     vis.snapshot_tick
 *     vis.traffic_enabled
 *     vis.traffic_warmup
 *
 * The remaining fields are managed by the vis_* functions.
 */
struct vis {
        int     snapshot_enabled;
        int32_t snapshot_tick;

        int     traffic_enabled;
        int32_t traffic_warmup;

        size_t traffic_node[VIS_TRAFFIC_CORNERS];
        int    traffic_owner[VIS_TRAFFIC_CORNERS];
        FILE  *traffic_file[VIS_TRAFFIC_CORNERS];
};

/*
 * Initializes visualization state. This is optional for a statically
 * zero-initialized struct, but is recommended for non-static instances.
 */
void vis_init(struct vis *vis);

/*
 * Returns the stable name used for one of the four traffic viewpoints:
 *
 *     0 southwest
 *     1 northwest
 *     2 southeast
 *     3 northeast
 */
const char *vis_corner_name(int corner);

/*
 * Selects the geographically extreme traffic viewpoints.
 *
 * Must be called on every rank because every rank needs the same selected
 * node and owner metadata. The city topology must be replicated and its
 * partition ownership information must already be initialized.
 */
int vis_select_traffic_corners(struct vis *vis,
                               const struct city *city,
                               int rank);

/*
 * Opens traffic-view CSV files on the ranks owning the selected viewpoints.
 *
 * This function performs an MPI_Allreduce so it must be called collectively
 * by every rank when MPI_COMM_WORLD is active.
 */
int vis_open_traffic_files(struct vis *vis,
                           const struct city *city,
                           int rank,
                           const char *replica_mode,
                           int stale_delay);

/*
 * Writes the optional rank-zero replica snapshot for tick t.
 */
void vis_write_staleness_snapshot(const struct vis *vis,
                                  const struct city *city,
                                  const struct cong_replica *replica,
                                  int rank,
                                  int32_t t,
                                  const char *replica_mode);

/*
 * Writes one warmup traffic frame from the current rank's replica.
 *
 * Call this after incoming replica merges and immediately before routing.
 * cong_half_life and route_delay_factor must match the values used by the
 * routing cost calculation.
 */
void vis_write_traffic_tick(struct vis *vis,
                            const struct city *city,
                            const struct cong_replica *replica,
                            int rank,
                            int32_t t,
                            float cong_half_life,
                            float route_delay_factor);

/*
 * Closes all traffic-view files owned by this rank.
 */
void vis_close(struct vis *vis, int rank);

#ifdef __cplusplus
}
#endif

#endif /* VIS_H */
