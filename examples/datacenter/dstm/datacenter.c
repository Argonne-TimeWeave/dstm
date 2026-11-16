/*******************************************************************************
 * DSTM push-based datacenter job/power/thermal simulation
 *
 * Job trace format, without a header:
 *
 *   arrival,duration,nodes,cpu_per_node,gpu_per_node,cpu_util,gpu_util
 *
 * Push protocols
 * ==============
 *
 * Heat, for physical source rack u and destination rack v:
 *
 *   source:
 *       dstm_dgraph_edge_put(heat_graph[delay],
 *                            vertex(v), vertex(u),
 *                            message, t, delay)
 *
 *   destination:
 *       dstm_dgraph_edge_get(heat_graph[delay],
 *                            vertex(v), vertex(u),
 *                            message, t, 0)
 *
 * The first endpoint is v, so the edge is owned by the receiver. Each
 * delay class has its own graph and DSTM region. Thus, every heat-region
 * version is populated by operations with one common delta.
 *
 * Feedback:
 *
 *   worker pushes feedback generated at B into a scheduler-owned slot
 *   with delta control_latency. Scheduler reads that slot locally at
 *   B + control_latency.
 *
 * Schedule:
 *
 *   scheduler pushes a schedule generated at B + control_latency into a
 *   worker-owned slot with delta schedule_window - control_latency.
 *   Worker reads it locally at B + schedule_window.
 *
 * All communication state classes use separate DSTM regions.
 ******************************************************************************/

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>

#include "dstm.h"

/*******************************************************************************
 * Constants
 ******************************************************************************/

#define MAX_NODES_PER_RACK 32
#define MAX_JOB_NODES      64
#define MAX_THERMAL_DELAY   3

#define DEFAULT_RACKS             16
#define DEFAULT_NODES_PER_RACK     8
#define DEFAULT_STEPS            300
#define DEFAULT_SCHEDULE_WINDOW   60
#define DEFAULT_CONTROL_LATENCY   30

#define SCHEDULER_RANK 0

#define NODE_CPU_CAPACITY 64
#define NODE_GPU_CAPACITY  8

#define NODE_IDLE_POWER_W    180.0
#define CPU_DYNAMIC_POWER_W  320.0
#define GPU_DYNAMIC_POWER_W 2400.0

#define DT_SECONDS 1.0

#define AMBIENT_TEMP_C         22.0
#define INITIAL_COOLANT_TEMP_C 18.0
#define INITIAL_OUTLET_TEMP_C  24.0
#define INITIAL_SETPOINT_C     18.0
#define TARGET_INLET_TEMP_C    24.0

#define RACK_HEAT_CAPACITY_J_K    120000.0
#define COOLANT_HEAT_CAPACITY_J_K 500000.0

#define AIRFLOW_HEAT_CAPACITY_W_K 8000.0
#define RACK_TO_COOLANT_UA_W_K    6000.0

#define COOLING_SETPOINT_RESET_C_C 0.10
#define COOLING_SETPOINT_RATE_C_S  0.25
#define MIN_COOLING_SETPOINT_C    12.0
#define MAX_COOLING_SETPOINT_C    24.0

#define COOLING_PROPORTIONAL_W_K 12000.0
#define MAX_COOLING_REMOVAL_W   100000.0
#define COOLING_ACTUATOR_TIME_S     20.0
#define COOLING_COP                  4.0

#define HEAT_RECIRCULATION_FRACTION 0.04

#define MIN_REGION_SIZE (1ULL << 16)

/*******************************************************************************
 * Error handling
 ******************************************************************************/

static int g_rank;
static int g_world_size;

#define CHECK(expr, what)                                                     \
        do {                                                                  \
                int _rc = (expr);                                             \
                if (_rc != DSTM_SUCCESS) {                                    \
                        fprintf(stderr,                                       \
                                "[rank %d] %s failed: rc=%d at %s:%d\n",      \
                                g_rank, (what), _rc, __FILE__, __LINE__);     \
                        fflush(stderr);                                       \
                        MPI_Abort(MPI_COMM_WORLD, 1);                         \
                }                                                             \
        } while (0)

#define REQUIRE(cond, what)                                                   \
        do {                                                                  \
                if (!(cond)) {                                                \
                        fprintf(stderr,                                       \
                                "[rank %d] requirement failed: %s "           \
                                "at %s:%d\n",                                 \
                                g_rank, (what), __FILE__, __LINE__);          \
                        fflush(stderr);                                       \
                        MPI_Abort(MPI_COMM_WORLD, 1);                         \
                }                                                             \
        } while (0)

/*******************************************************************************
 * Model state
 ******************************************************************************/

enum job_status {
        JOB_UNUSED = 0,
        JOB_FUTURE,
        JOB_QUEUED,
        JOB_RUNNING,
        JOB_COMPLETE,
};

struct node_state {
        int cpu_capacity;
        int gpu_capacity;

        int cpu_used;
        int gpu_used;

        double cpu_utilization;
        double gpu_utilization;
        double power_watts;

        int running_jobs;
};

struct rack_state {
        int rack_id;
        int node_count;

        struct node_state nodes[MAX_NODES_PER_RACK];

        double total_power_watts;

        double inlet_temp_c;
        double outlet_temp_c;

        double coolant_temp_c;
        double coolant_flow;
        double cooling_setpoint_c;
        double cooling_removal_w;
};

struct allocation {
        int global_rack;
        int node;
};

struct job_state {
        int valid;
        int id;

        int arrival_time;
        int duration;

        int requested_nodes;
        int cpu_per_node;
        int gpu_per_node;

        double cpu_utilization;
        double gpu_utilization;

        enum job_status status;

        int start_time;
        int completion_time;

        int allocation_count;
        struct allocation allocations[MAX_JOB_NODES];
};

struct heat_message {
        int source_rack;
        int destination_rack;
        int generation_time;
        int arrival_time;

        double heat_watts;
};

struct scheduled_node {
        int cpu_used;
        int gpu_used;
        int running_jobs;

        double cpu_utilization;
        double gpu_utilization;
};

struct schedule_block_header {
        int generation;
        int start_time;
        int tick_count;

        int destination_rank;
        int rack_count;
        int nodes_per_rack;

        int event_count;
        int event_capacity;
};

struct schedule_event {
        int tick_offset;
        int local_node_index;

        struct scheduled_node state;
};

struct rack_feedback {
        int global_rack;

        double inlet_temp_c;
        double outlet_temp_c;
        double coolant_temp_c;
        double cooling_removal_w;
        double total_power_watts;
};

struct feedback_block_header {
        int snapshot_time;
        int source_rank;
        int rack_count;
};

/*******************************************************************************
 * Simulation context
 ******************************************************************************/

struct sim {
        int total_racks;
        int nodes_per_rack;
        int steps;

        int schedule_window;
        int control_latency;

        int thermal_rank_count;
        int is_scheduler;
        int thermal_rank;

        size_t local_racks;
        size_t rack_offset;

        struct rack_state *racks;

        struct job_state *jobs;
        size_t job_count;

        /*
         * Jobs are sorted by (arrival_time, id). These cursors prevent
         * rescanning the full trace at every simulated tick.
         */
        size_t next_arrival_job;
        size_t next_queued_job;

        /*
         * Min-heap of indices into jobs[], ordered by completion_time.
         */
        size_t *completion_heap;
        size_t completion_heap_size;
        size_t completion_heap_capacity;

        struct node_state *scheduler_nodes;
        struct rack_feedback *scheduler_feedback;

        dstm_distance_t *distance;

        /*
         * heat_graph[d] is used only by channels whose physical delay is d.
         * Index zero is unused.
         */
        dstm_dgraph_t *heat_graph[MAX_THERMAL_DELAY + 1];

        /*
         * Region zero backs schedule_array.
         *
         * feedback_region_id backs feedback_array.
         * heat_region_id[d] backs heat_graph[d].
         */
        int schedule_region_id;
        int feedback_region_id;
        int heat_region_id[MAX_THERMAL_DELAY + 1];

        dstm_darray_t *schedule_array;
        dstm_darray_t *feedback_array;

        size_t schedule_global_offset;
        size_t feedback_global_offset;
        size_t feedback_scheduler_offset;

        void *schedule_buffer;
        void *feedback_buffer;

        /*
         * Maximum number of node-state changes in one worker's schedule
         * block. The darray element size is fixed at setup.
         */
        size_t schedule_event_capacity;
        size_t schedule_event_cursor;

        size_t thermal_vertex_count;
        struct rack_state *thermal_vertices;

        /*
         * One independent CSR topology per delay class.
         */
        size_t *heat_degrees[MAX_THERMAL_DELAY + 1];
        size_t *heat_destinations[MAX_THERMAL_DELAY + 1];

        uint64_t jobs_started;
        uint64_t jobs_completed;

        double scheduler_planning_seconds;
        double scheduler_publication_seconds;

        uint64_t schedule_events_emitted;
        uint64_t schedule_event_capacity_total;

        double it_energy_joules;
        double cooling_energy_joules;

        double max_inlet_temp_c;
        double max_outlet_temp_c;
};

/*******************************************************************************
 * Utility functions
 ******************************************************************************/

static double clamp_double(double value, double low, double high)
{
        if (value < low)
                return low;
        if (value > high)
                return high;
        return value;
}

static size_t next_power_of_two(size_t value)
{
        size_t result = 1;

        if (value <= 1)
                return 1;

        while (result < value) {
                REQUIRE(result <= ((size_t)-1) / 2,
                        "power-of-two size overflow");
                result <<= 1;
        }

        return result;
}

static size_t padded_region_size(size_t payload)
{
        /*
         * Buddy allocations include headers and independently round large
         * graph/array slabs. Keep four times the expected payload and impose
         * a small lower bound.
         */
        REQUIRE(payload <= ((size_t)-1) / 4,
                "region payload overflow");

        size_t size = payload * 4;

        if (size < MIN_REGION_SIZE)
                size = MIN_REGION_SIZE;

        return next_power_of_two(size);
}

static int rack_owner(const struct sim *s, size_t rack)
{
        REQUIRE(rack < (size_t)s->total_racks,
                "invalid physical rack");

        return 1 + (int)(rack / s->local_racks);
}

static size_t rack_graph_vertex(size_t physical_rack)
{
        /*
         * Rank zero contributes graph vertex zero. Physical rack r maps to
         * graph vertex r + 1.
         */
        return physical_rack + 1;
}

static size_t worker_node_count(const struct sim *s)
{
        return s->local_racks *
               (size_t)s->nodes_per_rack;
}

static size_t schedule_block_size(const struct sim *s)
{
        return sizeof(struct schedule_block_header) +
               s->schedule_event_capacity *
               sizeof(struct schedule_event);
}

static size_t feedback_block_size(const struct sim *s)
{
        return sizeof(struct feedback_block_header) +
               s->local_racks *
               sizeof(struct rack_feedback);
}

static struct schedule_event *schedule_block_events(void *buffer)
{
        return (struct schedule_event *)
                ((unsigned char *)buffer +
                 sizeof(struct schedule_block_header));
}

static const struct schedule_event *schedule_block_events_const(
        const void *buffer)
{
        return (const struct schedule_event *)
                ((const unsigned char *)buffer +
                 sizeof(struct schedule_block_header));
}

static struct rack_feedback *feedback_block_racks(void *buffer)
{
        return (struct rack_feedback *)
                ((unsigned char *)buffer +
                 sizeof(struct feedback_block_header));
}

static const struct rack_feedback *feedback_block_racks_const(
        const void *buffer)
{
        return (const struct rack_feedback *)
                ((const unsigned char *)buffer +
                 sizeof(struct feedback_block_header));
}

static struct node_state *scheduler_node(
        struct sim *s,
        size_t global_rack,
        int node)
{
        REQUIRE(s->is_scheduler,
                "scheduler node accessed by thermal rank");

        REQUIRE(global_rack < (size_t)s->total_racks,
                "invalid scheduler rack");

        REQUIRE(node >= 0 &&
                node < s->nodes_per_rack,
                "invalid scheduler node");

        return &s->scheduler_nodes[
                global_rack *
                (size_t)s->nodes_per_rack +
                (size_t)node];
}

static size_t ring_left(const struct sim *s, size_t rack)
{
        return (rack +
                (size_t)s->total_racks -
                1) %
               (size_t)s->total_racks;
}

static size_t ring_right(const struct sim *s, size_t rack)
{
        return (rack + 1) %
               (size_t)s->total_racks;
}

static int thermal_delay(size_t a, size_t b)
{
        size_t lo = a < b ? a : b;
        size_t hi = a < b ? b : a;

        uint64_t hash =
                (uint64_t)lo * 2654435761ULL +
                (uint64_t)hi * 2246822519ULL;

        return 1 + (int)(hash % MAX_THERMAL_DELAY);
}

static struct scheduled_node make_scheduled_node(
        const struct node_state *node)
{
        return (struct scheduled_node) {
                .cpu_used = node->cpu_used,
                .gpu_used = node->gpu_used,
                .running_jobs = node->running_jobs,
                .cpu_utilization = node->cpu_utilization,
                .gpu_utilization = node->gpu_utilization,
        };
}

static int scheduled_node_equal(
        const struct scheduled_node *a,
        const struct scheduled_node *b)
{
        return a->cpu_used == b->cpu_used &&
               a->gpu_used == b->gpu_used &&
               a->running_jobs == b->running_jobs &&
               a->cpu_utilization == b->cpu_utilization &&
               a->gpu_utilization == b->gpu_utilization;
}

/*******************************************************************************
 * Job loading
 ******************************************************************************/

static int compare_jobs_by_arrival(
        const void *lhs,
        const void *rhs)
{
        const struct job_state *a =
                (const struct job_state *)lhs;

        const struct job_state *b =
                (const struct job_state *)rhs;

        if (a->arrival_time < b->arrival_time)
                return -1;

        if (a->arrival_time > b->arrival_time)
                return 1;

        if (a->id < b->id)
                return -1;

        if (a->id > b->id)
                return 1;

        return 0;
}

static int load_scheduler_jobs(struct sim *s, const char *path)
{
        if (!s->is_scheduler)
                return 0;

        FILE *file = fopen(path, "r");

        if (file == NULL) {
                fprintf(stderr,
                        "[rank %d] cannot open job trace %s\n",
                        g_rank,
                        path);
                return -1;
        }

        struct job_state *jobs = NULL;
        size_t count = 0;
        size_t capacity = 0;
        int id = 0;

        for (;;) {
                int arrival;
                int duration;
                int nodes;
                int cpu;
                int gpu;
                double cpu_util;
                double gpu_util;

                int n = fscanf(file,
                               " %d,%d,%d,%d,%d,%lf,%lf",
                               &arrival,
                               &duration,
                               &nodes,
                               &cpu,
                               &gpu,
                               &cpu_util,
                               &gpu_util);

                if (n == EOF)
                        break;

                if (n != 7) {
                        fprintf(stderr,
                                "[rank %d] malformed job record "
                                "near id %d\n",
                                g_rank,
                                id);
                        fclose(file);
                        free(jobs);
                        return -1;
                }

                REQUIRE(arrival >= 0,
                        "negative arrival time");

                REQUIRE(duration > 0,
                        "job duration must be positive");

                REQUIRE(nodes > 0 &&
                        nodes <= MAX_JOB_NODES,
                        "requested node count exceeds MAX_JOB_NODES");

                REQUIRE(cpu >= 0 &&
                        cpu <= NODE_CPU_CAPACITY,
                        "invalid CPU request");

                REQUIRE(gpu >= 0 &&
                        gpu <= NODE_GPU_CAPACITY,
                        "invalid GPU request");

                if (count == capacity) {
                        size_t next_capacity =
                                capacity ?
                                capacity * 2 :
                                32;

                        struct job_state *replacement =
                                realloc(jobs,
                                        next_capacity *
                                        sizeof(*jobs));

                        REQUIRE(replacement != NULL,
                                "job allocation");

                        jobs = replacement;
                        capacity = next_capacity;
                }

                jobs[count++] = (struct job_state) {
                        .valid = 1,
                        .id = id,
                        .arrival_time = arrival,
                        .duration = duration,
                        .requested_nodes = nodes,
                        .cpu_per_node = cpu,
                        .gpu_per_node = gpu,
                        .cpu_utilization =
                                clamp_double(cpu_util,
                                             0.0,
                                             1.0),
                        .gpu_utilization =
                                clamp_double(gpu_util,
                                             0.0,
                                             1.0),
                        .status = JOB_FUTURE,
                        .start_time = -1,
                        .completion_time = -1,
                        .allocation_count = 0,
                };

                id++;
        }

        fclose(file);

        /*
         * Preserve deterministic FCFS ordering while permitting cursor-based
         * arrival and queue processing.
         */
        qsort(jobs,
              count,
              sizeof(*jobs),
              compare_jobs_by_arrival);

        s->jobs = jobs;
        s->job_count = count;

        s->next_arrival_job = 0;
        s->next_queued_job = 0;

        s->completion_heap_capacity = count;
        s->completion_heap_size = 0;

        if (count > 0) {
                s->completion_heap =
                        malloc(count *
                               sizeof(*s->completion_heap));

                REQUIRE(s->completion_heap != NULL,
                        "completion heap allocation");
        }

        return 0;
}

/*******************************************************************************
 * Initial model state
 ******************************************************************************/

static void initialize_racks(struct sim *s)
{
        if (s->is_scheduler)
                return;

        s->racks =
                calloc(s->local_racks,
                       sizeof(*s->racks));

        REQUIRE(s->racks != NULL,
                "rack allocation");

        for (size_t i = 0;
             i < s->local_racks;
             i++) {
                struct rack_state *rack =
                        &s->racks[i];

                rack->rack_id =
                        (int)(s->rack_offset + i);

                rack->node_count =
                        s->nodes_per_rack;

                rack->inlet_temp_c =
                        AMBIENT_TEMP_C;

                rack->outlet_temp_c =
                        INITIAL_OUTLET_TEMP_C;

                rack->coolant_temp_c =
                        INITIAL_COOLANT_TEMP_C;

                rack->coolant_flow = 1.0;

                rack->cooling_setpoint_c =
                        INITIAL_SETPOINT_C;

                for (int n = 0;
                     n < rack->node_count;
                     n++) {
                        rack->nodes[n].cpu_capacity =
                                NODE_CPU_CAPACITY;

                        rack->nodes[n].gpu_capacity =
                                NODE_GPU_CAPACITY;

                        rack->nodes[n].power_watts =
                                NODE_IDLE_POWER_W;
                }
        }
}

static void initialize_scheduler_state(struct sim *s)
{
        if (!s->is_scheduler)
                return;

        size_t node_count =
                (size_t)s->total_racks *
                (size_t)s->nodes_per_rack;

        s->scheduler_nodes =
                calloc(node_count,
                       sizeof(*s->scheduler_nodes));

        s->scheduler_feedback =
                calloc((size_t)s->total_racks,
                       sizeof(*s->scheduler_feedback));

        REQUIRE(s->scheduler_nodes != NULL,
                "scheduler node allocation");

        REQUIRE(s->scheduler_feedback != NULL,
                "scheduler feedback allocation");

        for (size_t i = 0;
             i < node_count;
             i++) {
                s->scheduler_nodes[i].cpu_capacity =
                        NODE_CPU_CAPACITY;

                s->scheduler_nodes[i].gpu_capacity =
                        NODE_GPU_CAPACITY;

                s->scheduler_nodes[i].power_watts =
                        NODE_IDLE_POWER_W;
        }

        for (int rack = 0;
             rack < s->total_racks;
             rack++) {
                s->scheduler_feedback[rack] =
                        (struct rack_feedback) {
                                .global_rack = rack,
                                .inlet_temp_c =
                                        AMBIENT_TEMP_C,
                                .outlet_temp_c =
                                        INITIAL_OUTLET_TEMP_C,
                                .coolant_temp_c =
                                        INITIAL_COOLANT_TEMP_C,
                                .cooling_removal_w = 0.0,
                                .total_power_watts =
                                        NODE_IDLE_POWER_W *
                                        s->nodes_per_rack,
                        };
        }
}

/*******************************************************************************
 * Delay-separated graph topology
 ******************************************************************************/

static void build_heat_topologies(struct sim *s)
{
        s->thermal_vertex_count =
                s->is_scheduler ?
                1 :
                s->local_racks;

        if (s->is_scheduler) {
                s->thermal_vertices =
                        calloc(1,
                               sizeof(*s->thermal_vertices));

                REQUIRE(s->thermal_vertices != NULL,
                        "dummy thermal vertex allocation");
        } else {
                s->thermal_vertices = s->racks;
        }

        for (int delay = 1;
             delay <= MAX_THERMAL_DELAY;
             delay++) {
                size_t vertices =
                        s->thermal_vertex_count;

                s->heat_degrees[delay] =
                        calloc(vertices,
                               sizeof(size_t));

                /*
                 * At most two outgoing ring edges per physical rack.
                 * Rank zero has one dummy vertex and no edges.
                 */
                size_t capacity =
                        s->is_scheduler ?
                        1 :
                        2 * s->local_racks;

                s->heat_destinations[delay] =
                        calloc(capacity,
                               sizeof(size_t));

                REQUIRE(s->heat_degrees[delay] != NULL,
                        "heat degree allocation");

                REQUIRE(s->heat_destinations[delay] != NULL,
                        "heat destination allocation");

                if (s->is_scheduler)
                        continue;

                size_t edge_count = 0;

                for (size_t i = 0;
                     i < s->local_racks;
                     i++) {
                        size_t receiver =
                                s->rack_offset + i;

                        size_t sources[2] = {
                                ring_left(s, receiver),
                                ring_right(s, receiver),
                        };

                        size_t row[2];
                        size_t row_count = 0;

                        for (int n = 0; n < 2; n++) {
                                size_t source =
                                        sources[n];

                                if (thermal_delay(source,
                                                  receiver) !=
                                    delay) {
                                        continue;
                                }

                                /*
                                 * Receiver-owned edge:
                                 *
                                 *     receiver -> source
                                 */
                                row[row_count++] =
                                        rack_graph_vertex(source);
                        }

                        if (row_count == 2 &&
                            row[0] > row[1]) {
                                size_t tmp = row[0];
                                row[0] = row[1];
                                row[1] = tmp;
                        }

                        s->heat_degrees[delay][i] =
                                row_count;

                        for (size_t n = 0;
                             n < row_count;
                             n++) {
                                s->heat_destinations[delay]
                                                    [edge_count++] =
                                        row[n];
                        }
                }
        }
}

/*******************************************************************************
 * Session-global distance matrix
 ******************************************************************************/

static int create_matrix(
        dstm_distance_t **ret,
        int world_size,
        const int *matrix)
{
        return dstm_distance_create_from_matrix(
                ret,
                (size_t)world_size,
                matrix);
}

static int build_distance_matrix(struct sim *s)
{
        int ws = g_world_size;

        int *matrix =
                malloc((size_t)ws *
                       (size_t)ws *
                       sizeof(*matrix));

        if (matrix == NULL)
                return -1;

        for (int i = 0; i < ws; i++) {
                for (int j = 0; j < ws; j++) {
                        matrix[(size_t)i * ws + j] =
                                (i == j) ?
                                0 :
                                -1;
                }
        }

        /*
         * Thermal rank pairs.
         *
         * Use the smallest possible heat delay across each rank boundary,
         * matching the runtime's readable-bound semantics. Delay-separated
         * regions prevent different heat deltas from sharing one local
         * region state machine.
         */
        for (size_t u = 0;
             u < (size_t)s->total_racks;
             u++) {
                size_t neighbors[2] = {
                        ring_left(s, u),
                        ring_right(s, u),
                };

                for (int n = 0; n < 2; n++) {
                        size_t v = neighbors[n];

                        int ru = rack_owner(s, u);
                        int rv = rack_owner(s, v);

                        if (ru == rv)
                                continue;

                        int delay =
                                thermal_delay(u, v);

                        int *a =
                                &matrix[
                                        (size_t)ru *
                                        ws +
                                        rv];

                        int *b =
                                &matrix[
                                        (size_t)rv *
                                        ws +
                                        ru];

                        if (*a < 0 || delay < *a)
                                *a = delay;

                        if (*b < 0 || delay < *b)
                                *b = delay;
                }
        }

        /*
         * Scheduler-worker control channels. Both directions are recorded
         * because feedback flows worker -> scheduler and schedules flow
         * scheduler -> worker.
         */
        int schedule_delay =
                s->schedule_window -
                s->control_latency;

        for (int rank = 1;
             rank < ws;
             rank++) {
                int control_bound =
                        s->control_latency <
                        schedule_delay ?
                        s->control_latency :
                        schedule_delay;

                int *a =
                        &matrix[
                                (size_t)SCHEDULER_RANK *
                                ws +
                                rank];

                int *b =
                        &matrix[
                                (size_t)rank *
                                ws +
                                SCHEDULER_RANK];

                if (*a < 0 || control_bound < *a)
                        *a = control_bound;

                if (*b < 0 || control_bound < *b)
                        *b = control_bound;
        }

        int rc = create_matrix(
                &s->distance,
                ws,
                matrix);

        free(matrix);

        return rc == DSTM_SUCCESS ?
               0 :
               -1;
}

static int completion_job_less(
        const struct sim *s,
        size_t lhs_index,
        size_t rhs_index)
{
        const struct job_state *lhs =
                &s->jobs[lhs_index];

        const struct job_state *rhs =
                &s->jobs[rhs_index];

        if (lhs->completion_time <
            rhs->completion_time) {
                return 1;
        }

        if (lhs->completion_time >
            rhs->completion_time) {
                return 0;
        }

        return lhs->id < rhs->id;
}

static void completion_heap_push(
        struct sim *s,
        size_t job_index)
{
        REQUIRE(s->completion_heap_size <
                        s->completion_heap_capacity,
                "completion heap capacity exceeded");

        size_t position =
                s->completion_heap_size++;

        s->completion_heap[position] =
                job_index;

        while (position > 0) {
                size_t parent =
                        (position - 1) / 2;

                if (!completion_job_less(
                            s,
                            s->completion_heap[position],
                            s->completion_heap[parent])) {
                        break;
                }

                size_t temporary =
                        s->completion_heap[parent];

                s->completion_heap[parent] =
                        s->completion_heap[position];

                s->completion_heap[position] =
                        temporary;

                position = parent;
        }
}

static size_t completion_heap_pop(
        struct sim *s)
{
        REQUIRE(s->completion_heap_size > 0,
                "completion heap underflow");

        size_t result =
                s->completion_heap[0];

        s->completion_heap_size--;

        if (s->completion_heap_size == 0)
                return result;

        s->completion_heap[0] =
                s->completion_heap[
                        s->completion_heap_size];

        size_t position = 0;

        for (;;) {
                size_t left =
                        position * 2 + 1;

                size_t right =
                        left + 1;

                size_t smallest =
                        position;

                if (left <
                            s->completion_heap_size &&
                    completion_job_less(
                            s,
                            s->completion_heap[left],
                            s->completion_heap[smallest])) {
                        smallest = left;
                }

                if (right <
                            s->completion_heap_size &&
                    completion_job_less(
                            s,
                            s->completion_heap[right],
                            s->completion_heap[smallest])) {
                        smallest = right;
                }

                if (smallest == position)
                        break;

                size_t temporary =
                        s->completion_heap[position];

                s->completion_heap[position] =
                        s->completion_heap[smallest];

                s->completion_heap[smallest] =
                        temporary;

                position = smallest;
        }

        return result;
}

static double job_cpu_utilization_contribution(
        const struct node_state *node,
        const struct job_state *job)
{
        return ((double)job->cpu_per_node /
                (double)node->cpu_capacity) *
               job->cpu_utilization;
}

static double job_gpu_utilization_contribution(
        const struct node_state *node,
        const struct job_state *job)
{
        return ((double)job->gpu_per_node /
                (double)node->gpu_capacity) *
               job->gpu_utilization;
}

static void add_job_to_node(
        struct node_state *node,
        const struct job_state *job)
{
        node->cpu_used +=
                job->cpu_per_node;

        node->gpu_used +=
                job->gpu_per_node;

        node->running_jobs++;

        node->cpu_utilization +=
                job_cpu_utilization_contribution(
                        node,
                        job);

        node->gpu_utilization +=
                job_gpu_utilization_contribution(
                        node,
                        job);

        node->cpu_utilization =
                clamp_double(
                        node->cpu_utilization,
                        0.0,
                        1.0);

        node->gpu_utilization =
                clamp_double(
                        node->gpu_utilization,
                        0.0,
                        1.0);
}

static void remove_job_from_node(
        struct node_state *node,
        const struct job_state *job)
{
        node->cpu_utilization -=
                job_cpu_utilization_contribution(
                        node,
                        job);

        node->gpu_utilization -=
                job_gpu_utilization_contribution(
                        node,
                        job);

        node->cpu_used -=
                job->cpu_per_node;

        node->gpu_used -=
                job->gpu_per_node;

        node->running_jobs--;

        /*
         * Remove small negative values introduced by floating-point
         * subtraction.
         */
        node->cpu_utilization =
                clamp_double(
                        node->cpu_utilization,
                        0.0,
                        1.0);

        node->gpu_utilization =
                clamp_double(
                        node->gpu_utilization,
                        0.0,
                        1.0);

        assert(node->cpu_used >= 0);
        assert(node->gpu_used >= 0);
        assert(node->running_jobs >= 0);
}

/*******************************************************************************
 * Scheduler job processing
 ******************************************************************************/

static void release_completed_jobs(
        struct sim *s,
        int effective_time)
{
        REQUIRE(s->is_scheduler,
                "job release outside scheduler");

        while (s->completion_heap_size > 0) {
                size_t job_index =
                        s->completion_heap[0];

                struct job_state *job =
                        &s->jobs[job_index];

                if (job->completion_time >
                    effective_time) {
                        break;
                }

                size_t popped_index =
                        completion_heap_pop(s);

                REQUIRE(popped_index == job_index,
                        "completion heap root changed");

                REQUIRE(job->status == JOB_RUNNING,
                        "completed heap job is not running");

                for (int a = 0;
                     a < job->allocation_count;
                     a++) {
                        int rack =
                                job->allocations[a]
                                   .global_rack;

                        int node_index =
                                job->allocations[a]
                                   .node;

                        struct node_state *node =
                                scheduler_node(
                                        s,
                                        (size_t)rack,
                                        node_index);

                        remove_job_from_node(
                                node,
                                job);
                }

                job->allocation_count = 0;
                job->status = JOB_COMPLETE;

                s->jobs_completed++;
        }
}

static void admit_arrivals(
        struct sim *s,
        int effective_time)
{
        REQUIRE(s->is_scheduler,
                "job admission outside scheduler");

        while (s->next_arrival_job <
               s->job_count) {
                struct job_state *job =
                        &s->jobs[
                                s->next_arrival_job];

                if (job->arrival_time >
                    effective_time) {
                        break;
                }

                REQUIRE(job->status == JOB_FUTURE,
                        "arrival cursor found non-future job");

                job->status = JOB_QUEUED;

                s->next_arrival_job++;
        }
}

static int node_can_host(
        const struct node_state *node,
        const struct job_state *job)
{
        int cpu_free =
                node->cpu_capacity -
                node->cpu_used;

        int gpu_free =
                node->gpu_capacity -
                node->gpu_used;

        return cpu_free >=
                        job->cpu_per_node &&
               gpu_free >=
                        job->gpu_per_node;
}

static int try_start_job(
        struct sim *s,
        struct job_state *job,
        int effective_time)
{
        REQUIRE(s->is_scheduler,
                "job placement outside scheduler");

        struct allocation candidates[MAX_JOB_NODES];
        int candidate_count = 0;

        /*
         * The previous implementation rescanned every node once for each
         * requested node in order to find the lexicographically smallest
         * unselected feasible node.
         *
         * A single rack-major scan produces exactly the same candidate
         * sequence because node capacity is not modified until all
         * candidates have been found.
         */
        for (int rack = 0;
             rack < s->total_racks &&
                     candidate_count < job->requested_nodes;
             rack++) {
                for (int node_index = 0;
                     node_index < s->nodes_per_rack &&
                             candidate_count < job->requested_nodes;
                     node_index++) {
                        struct node_state *node =
                                scheduler_node(
                                        s,
                                        (size_t)rack,
                                        node_index);

                        if (!node_can_host(
                                    node,
                                    job)) {
                                continue;
                        }

                        candidates[candidate_count++] =
                                (struct allocation) {
                                        .global_rack =
                                                rack,
                                        .node =
                                                node_index,
                                };
                }
        }

        if (candidate_count !=
            job->requested_nodes) {
                return 0;
        }

        for (int a = 0;
             a < candidate_count;
             a++) {
                struct allocation allocation =
                        candidates[a];

                struct node_state *node =
                        scheduler_node(
                                s,
                                (size_t)allocation.global_rack,
                                allocation.node);

                add_job_to_node(
                        node,
                        job);

                job->allocations[a] =
                        allocation;
        }

        job->allocation_count =
                candidate_count;

        job->start_time =
                effective_time;

        job->completion_time =
                effective_time +
                job->duration;

        job->status =
                JOB_RUNNING;

        size_t job_index =
                (size_t)(job - s->jobs);

        REQUIRE(job_index < s->job_count,
                "invalid started job index");

        completion_heap_push(
                s,
                job_index);

        s->jobs_started++;

        return 1;
}

static void schedule_queued_jobs(
        struct sim *s,
        int effective_time)
{
        REQUIRE(s->is_scheduler,
                "job scheduling outside scheduler");

        while (s->next_queued_job <
               s->job_count) {
                struct job_state *job =
                        &s->jobs[
                                s->next_queued_job];

                /*
                 * This job has not arrived yet. Since jobs are sorted by
                 * arrival time, no later job can be ready.
                 */
                if (job->status == JOB_FUTURE)
                        break;

                REQUIRE(job->status == JOB_QUEUED,
                        "queue cursor found invalid job state");

                /*
                 * Preserve the original strict FCFS behavior: if the first
                 * queued job does not fit, do not backfill later jobs.
                 */
                if (!try_start_job(
                            s,
                            job,
                            effective_time)) {
                        break;
                }

                s->next_queued_job++;
        }
}

/*******************************************************************************
 * Push feedback channel
 ******************************************************************************/

static void publish_feedback(
        struct sim *s,
        dstm_time_t snapshot_time)
{
        REQUIRE(!s->is_scheduler,
                "scheduler cannot publish feedback");

        memset(s->feedback_buffer,
               0,
               feedback_block_size(s));

        struct feedback_block_header *header =
                (struct feedback_block_header *)
                s->feedback_buffer;

        header->snapshot_time =
                (int)snapshot_time;

        header->source_rank =
                g_rank;

        header->rack_count =
                (int)s->local_racks;

        struct rack_feedback *output =
                feedback_block_racks(
                        s->feedback_buffer);

        for (size_t i = 0;
             i < s->local_racks;
             i++) {
                const struct rack_state *rack =
                        &s->racks[i];

                output[i] =
                        (struct rack_feedback) {
                                .global_rack =
                                        rack->rack_id,
                                .inlet_temp_c =
                                        rack->inlet_temp_c,
                                .outlet_temp_c =
                                        rack->outlet_temp_c,
                                .coolant_temp_c =
                                        rack->coolant_temp_c,
                                .cooling_removal_w =
                                        rack->cooling_removal_w,
                                .total_power_watts =
                                        rack->total_power_watts,
                        };
        }

        /*
         * Rank zero owns slots:
         *
         *     feedback_scheduler_offset + thermal_rank
         */
        size_t scheduler_slot =
                s->feedback_scheduler_offset +
                (size_t)s->thermal_rank;

        CHECK(dstm_darray_put(
                      s->feedback_array,
                      scheduler_slot,
                      s->feedback_buffer,
                      snapshot_time,
                      s->control_latency),
              "future feedback push");

        /*
         * Advance this worker's unused local feedback element so its local
         * feedback region participates in the rank timeline.
         */
        if (snapshot_time > 0) {
                CHECK(dstm_darray_put(
                              s->feedback_array,
                              s->feedback_global_offset,
                              s->feedback_buffer,
                              snapshot_time,
                              0),
                      "feedback dummy put");
        }
}

static void collect_feedback(
        struct sim *s,
        dstm_time_t scheduler_time,
        int snapshot_time)
{
        REQUIRE(s->is_scheduler,
                "feedback collection outside scheduler");

        for (int source_rank = 1;
             source_rank < g_world_size;
             source_rank++) {
                memset(s->feedback_buffer,
                       0,
                       feedback_block_size(s));

                size_t slot =
                        s->feedback_scheduler_offset +
                        (size_t)(source_rank - 1);

                CHECK(dstm_darray_get(
                              s->feedback_array,
                              slot,
                              s->feedback_buffer,
                              scheduler_time,
                              0),
                      "local feedback get");

                const struct feedback_block_header *header =
                        (const struct feedback_block_header *)
                        s->feedback_buffer;

                REQUIRE(header->snapshot_time ==
                                snapshot_time,
                        "feedback timestamp mismatch");

                REQUIRE(header->source_rank ==
                                source_rank,
                        "feedback source mismatch");

                REQUIRE(header->rack_count ==
                                (int)s->local_racks,
                        "feedback rack count mismatch");

                const struct rack_feedback *input =
                        feedback_block_racks_const(
                                s->feedback_buffer);

                for (size_t i = 0;
                     i < s->local_racks;
                     i++) {
                        int rack =
                                input[i].global_rack;

                        REQUIRE(rack >= 0 &&
                                rack <
                                        s->total_racks,
                                "invalid feedback rack");

                        REQUIRE(rack_owner(
                                        s,
                                        (size_t)rack) ==
                                        source_rank,
                                "feedback rack ownership mismatch");

                        s->scheduler_feedback[rack] =
                                input[i];
                }
        }
}

/*******************************************************************************
 * Push schedule channel
 ******************************************************************************/

static void append_worker_schedule_events(
        struct sim *s,
        void *buffer,
        int destination_rank,
        int tick_offset,
        struct scheduled_node *previous)
{
        struct schedule_block_header *header =
                (struct schedule_block_header *)buffer;

        struct schedule_event *events =
                schedule_block_events(buffer);

        size_t first_rack =
                (size_t)(destination_rank - 1) *
                s->local_racks;

        size_t local_node_index = 0;

        for (size_t local_rack = 0;
             local_rack < s->local_racks;
             local_rack++) {
                size_t global_rack =
                        first_rack + local_rack;

                for (int node_index = 0;
                     node_index < s->nodes_per_rack;
                     node_index++, local_node_index++) {
                        size_t global_node_index =
                                global_rack *
                                (size_t)s->nodes_per_rack +
                                (size_t)node_index;

                        const struct node_state *node =
                                scheduler_node(
                                        s,
                                        global_rack,
                                        node_index);

                        struct scheduled_node current =
                                make_scheduled_node(node);

                        if (scheduled_node_equal(
                                    &current,
                                    &previous[global_node_index])) {
                                continue;
                        }

                        REQUIRE(header->event_count <
                                        header->event_capacity,
                                "schedule event capacity exceeded");

                        events[header->event_count++] =
                                (struct schedule_event) {
                                        .tick_offset = tick_offset,
                                        .local_node_index =
                                                (int)local_node_index,
                                        .state = current,
                                };

                        previous[global_node_index] = current;
                }
        }
}

static void plan_and_publish_epoch(
        struct sim *s,
        dstm_time_t scheduler_time,
        int epoch_start)
{
        REQUIRE(s->is_scheduler,
                "epoch planning outside scheduler");

        int tick_count =
                s->schedule_window;

        if (epoch_start + tick_count >
            s->steps) {
                tick_count =
                        s->steps -
                        epoch_start;
        }

        if (tick_count <= 0)
                return;

        size_t bytes =
                schedule_block_size(s);

        void **blocks =
                calloc((size_t)g_world_size,
                       sizeof(*blocks));

        REQUIRE(blocks != NULL,
                "schedule block pointer allocation");

        for (int destination_rank = 1;
             destination_rank < g_world_size;
             destination_rank++) {
                blocks[destination_rank] =
                        calloc(1, bytes);

                REQUIRE(blocks[destination_rank] != NULL,
                        "schedule block allocation");

                struct schedule_block_header *header =
                        (struct schedule_block_header *)
                        blocks[destination_rank];

                header->generation =
                        epoch_start /
                        s->schedule_window;

                header->start_time =
                        epoch_start;

                header->tick_count =
                        tick_count;

                header->destination_rank =
                        destination_rank;

                header->rack_count =
                        (int)s->local_racks;

                header->nodes_per_rack =
                        s->nodes_per_rack;
                
                header->event_count = 0;
                header->event_capacity = (int)s->schedule_event_capacity;
        }

        size_t total_node_count =
                (size_t)s->total_racks *
                (size_t)s->nodes_per_rack;

        struct scheduled_node *previous =
                calloc(total_node_count, sizeof(*previous));

        REQUIRE(previous != NULL,
                "previous schedule state allocation");

        /*
         * Initialize the comparison state from the state at the end of the
         * preceding epoch. Workers already retain this state locally.
         */
        for (size_t global_rack = 0;
             global_rack < (size_t)s->total_racks;
             global_rack++) {
                for (int node_index = 0;
                     node_index < s->nodes_per_rack;
                     node_index++) {
                        size_t global_node_index =
                                global_rack *
                                (size_t)s->nodes_per_rack +
                                (size_t)node_index;

                        previous[global_node_index] =
                                make_scheduled_node(
                                        scheduler_node(
                                                s,
                                                global_rack,
                                                node_index));
                }
        }

        double planning_start =
                MPI_Wtime();

        for (int offset = 0;
             offset < tick_count;
             offset++) {
                int effective_time =
                        epoch_start + offset;

                release_completed_jobs(
                        s,
                        effective_time);

                admit_arrivals(
                        s,
                        effective_time);

                schedule_queued_jobs(
                        s,
                        effective_time);

                for (int destination_rank = 1;
                     destination_rank < g_world_size;
                     destination_rank++) {
                        append_worker_schedule_events(
                                s,
                                blocks[destination_rank],
                                destination_rank,
                                offset,
                                previous);
                }
        }

        free(previous);

        s->scheduler_planning_seconds +=
                MPI_Wtime() -
                planning_start;

        double publication_start =
                MPI_Wtime();

        int schedule_delay =
                epoch_start -
                (int)scheduler_time;

        REQUIRE(schedule_delay > 0,
                "schedule must target future time");

        for (int destination_rank = 1;
             destination_rank < g_world_size;
             destination_rank++) {
                /*
                 * Every rank owns one schedule element, so the worker's MPI
                 * rank is its global schedule-array index.
                 */

                const struct schedule_block_header *published_header =
                        (const struct schedule_block_header *)
                        blocks[destination_rank];

                s->schedule_events_emitted +=
                        (uint64_t)published_header->event_count;

                s->schedule_event_capacity_total +=
                        (uint64_t)published_header->event_capacity;                

                CHECK(dstm_darray_put(
                              s->schedule_array,
                              (size_t)destination_rank,
                              blocks[destination_rank],
                              scheduler_time,
                              schedule_delay),
                      "future schedule push");

                free(blocks[destination_rank]);
        }

        free(blocks);

        s->scheduler_publication_seconds +=
                MPI_Wtime() -
                publication_start;
}

static void receive_schedule_block(
        struct sim *s,
        dstm_time_t epoch_start)
{
        REQUIRE(!s->is_scheduler,
                "scheduler cannot receive schedule");

        memset(s->schedule_buffer,
               0,
               schedule_block_size(s));

        CHECK(dstm_darray_get(
                      s->schedule_array,
                      s->schedule_global_offset,
                      s->schedule_buffer,
                      epoch_start,
                      0),
              "local schedule get");

        const struct schedule_block_header *header =
                (const struct schedule_block_header *)
                s->schedule_buffer;

        REQUIRE(header->start_time ==
                        (int)epoch_start,
                "schedule start mismatch");

        REQUIRE(header->destination_rank ==
                        g_rank,
                "schedule destination mismatch");

        REQUIRE(header->rack_count ==
                        (int)s->local_racks,
                "schedule rack count mismatch");

        REQUIRE(header->nodes_per_rack ==
                        s->nodes_per_rack,
                "schedule node count mismatch");

        REQUIRE(header->tick_count > 0 &&
                header->tick_count <=
                        s->schedule_window,
                "invalid schedule tick count");

        REQUIRE(header->event_count >= 0 &&
                header->event_count <=
                        header->event_capacity,
                "invalid schedule event count");

        REQUIRE(header->event_capacity ==
                        (int)s->schedule_event_capacity,
                "schedule event capacity mismatch");

        s->schedule_event_cursor = 0;
}

static void apply_schedule_tick(
        struct sim *s,
        dstm_time_t t)
{
        REQUIRE(!s->is_scheduler,
                "scheduler cannot apply schedule");

        if (t < s->schedule_window)
                return;

        const struct schedule_block_header *header =
                (const struct schedule_block_header *)
                s->schedule_buffer;

        int offset =
                (int)t -
                header->start_time;

        REQUIRE(offset >= 0 &&
                offset < header->tick_count,
                "schedule tick outside block");

        const struct schedule_event *events =
                schedule_block_events_const(
                        s->schedule_buffer);

        while (s->schedule_event_cursor <
               (size_t)header->event_count) {
                const struct schedule_event *event =
                        &events[s->schedule_event_cursor];

                if (event->tick_offset > offset)
                        break;

                REQUIRE(event->tick_offset == offset,
                        "missed schedule event");

                REQUIRE(event->local_node_index >= 0 &&
                        (size_t)event->local_node_index <
                                worker_node_count(s),
                        "invalid schedule event node");

                size_t local_rack =
                        (size_t)event->local_node_index /
                        (size_t)s->nodes_per_rack;

                int node_index =
                        event->local_node_index %
                        s->nodes_per_rack;

                struct node_state *node =
                        &s->racks[local_rack].nodes[node_index];

                node->cpu_used =
                        event->state.cpu_used;

                node->gpu_used =
                        event->state.gpu_used;

                node->running_jobs =
                        event->state.running_jobs;

                node->cpu_utilization =
                        event->state.cpu_utilization;

                node->gpu_utilization =
                        event->state.gpu_utilization;

                s->schedule_event_cursor++;
        }
}

/*******************************************************************************
 * Rack-state versioning
 *
 * The delay-one heat graph also stores rack vertex state. This avoids a
 * fourth graph while preserving versioned rack state.
 ******************************************************************************/

static void read_current_racks(
        struct sim *s,
        dstm_time_t t)
{
        REQUIRE(!s->is_scheduler,
                "scheduler cannot read racks");

        for (size_t i = 0;
             i < s->local_racks;
             i++) {
                size_t vertex =
                        rack_graph_vertex(
                                s->rack_offset +
                                i);

                CHECK(dstm_dgraph_vertex_get(
                              s->heat_graph[1],
                              vertex,
                              &s->racks[i],
                              t,
                              0),
                      "rack vertex get");
        }
}

static void publish_next_rack_state(
        struct sim *s,
        dstm_time_t t)
{
        REQUIRE(!s->is_scheduler,
                "scheduler cannot publish racks");

        for (size_t i = 0;
             i < s->local_racks;
             i++) {
                size_t vertex =
                        rack_graph_vertex(
                                s->rack_offset +
                                i);

                CHECK(dstm_dgraph_vertex_put(
                              s->heat_graph[1],
                              vertex,
                              &s->racks[i],
                              t,
                              1),
                      "future rack vertex put");
        }
}

/*******************************************************************************
 * Push heat channel
 ******************************************************************************/

static double receive_neighbor_heat(
        struct sim *s,
        size_t local_rack,
        dstm_time_t t)
{
        size_t destination =
                s->rack_offset +
                local_rack;

        size_t sources[2] = {
                ring_left(s, destination),
                ring_right(s, destination),
        };

        double incoming_heat = 0.0;

        for (int n = 0;
             n < 2;
             n++) {
                size_t source =
                        sources[n];

                int delay =
                        thermal_delay(
                                source,
                                destination);

                /*
                 * No message can have arrived before its physical delay.
                 * Skipping these timestamps removes the need for future
                 * pipeline-prefill puts during setup.
                 */
                if (t < (dstm_time_t)delay)
                        continue;

                struct heat_message message =
                        {0};

                CHECK(dstm_dgraph_edge_get(
                              s->heat_graph[delay],
                              rack_graph_vertex(destination),
                              rack_graph_vertex(source),
                              &message,
                              t,
                              0),
                      "local heat edge get");

                if (message.source_rack ==
                            (int)source &&
                    message.destination_rack ==
                            (int)destination &&
                    message.generation_time ==
                            (int)t - delay &&
                    message.arrival_time ==
                            (int)t) {
                        incoming_heat +=
                                message.heat_watts;
                }
        }

        return incoming_heat;
}

static void send_delayed_heat(
        struct sim *s,
        dstm_time_t t)
{
        REQUIRE(!s->is_scheduler,
                "scheduler cannot send heat");

        for (size_t i = 0;
             i < s->local_racks;
             i++) {
                size_t source =
                        s->rack_offset + i;

                size_t destinations[2] = {
                        ring_left(s, source),
                        ring_right(s, source),
                };

                double exhaust_heat =
                        AIRFLOW_HEAT_CAPACITY_W_K *
                        fmax(s->racks[i]
                                     .outlet_temp_c -
                                     AMBIENT_TEMP_C,
                             0.0);

                double heat_per_neighbor =
                        HEAT_RECIRCULATION_FRACTION *
                        exhaust_heat /
                        2.0;

                for (int n = 0;
                     n < 2;
                     n++) {
                        size_t destination =
                                destinations[n];

                        int delay =
                                thermal_delay(
                                        source,
                                        destination);

                        struct heat_message message = {
                                .source_rack =
                                        (int)source,
                                .destination_rack =
                                        (int)destination,
                                .generation_time =
                                        (int)t,
                                .arrival_time =
                                        (int)t + delay,
                                .heat_watts =
                                        heat_per_neighbor,
                        };

                        /*
                         * Receiver-owned edge (destination, source).
                         */
                        CHECK(dstm_dgraph_edge_put(
                                      s->heat_graph[delay],
                                      rack_graph_vertex(
                                              destination),
                                      rack_graph_vertex(
                                              source),
                                      &message,
                                      t,
                                      delay),
                              "future heat edge push");
                }
        }
}

/*******************************************************************************
 * Power and thermal model
 ******************************************************************************/

static void update_local_power(struct sim *s)
{
        REQUIRE(!s->is_scheduler,
                "scheduler cannot update power");

        for (size_t r = 0;
             r < s->local_racks;
             r++) {
                struct rack_state *rack =
                        &s->racks[r];

                rack->total_power_watts =
                        0.0;

                for (int n = 0;
                     n < rack->node_count;
                     n++) {
                        struct node_state *node =
                                &rack->nodes[n];

                        node->cpu_utilization =
                                clamp_double(
                                        node->cpu_utilization,
                                        0.0,
                                        1.0);

                        node->gpu_utilization =
                                clamp_double(
                                        node->gpu_utilization,
                                        0.0,
                                        1.0);

                        node->power_watts =
                                NODE_IDLE_POWER_W +
                                CPU_DYNAMIC_POWER_W *
                                        node->cpu_utilization +
                                GPU_DYNAMIC_POWER_W *
                                        node->gpu_utilization;

                        rack->total_power_watts +=
                                node->power_watts;
                }

                s->it_energy_joules +=
                        rack->total_power_watts *
                        DT_SECONDS;
        }
}

static double integrate_rack_thermal(
        struct rack_state *rack,
        double incoming_heat_watts)
{
        double next_inlet =
                AMBIENT_TEMP_C +
                incoming_heat_watts /
                        AIRFLOW_HEAT_CAPACITY_W_K;

        double heat_to_air =
                AIRFLOW_HEAT_CAPACITY_W_K *
                (rack->outlet_temp_c -
                 next_inlet);

        double heat_to_coolant =
                RACK_TO_COOLANT_UA_W_K *
                (rack->outlet_temp_c -
                 rack->coolant_temp_c);

        double outlet_derivative =
                (rack->total_power_watts -
                 heat_to_air -
                 heat_to_coolant) /
                RACK_HEAT_CAPACITY_J_K;

        double next_outlet =
                rack->outlet_temp_c +
                DT_SECONDS *
                outlet_derivative;

        double temperature_error =
                next_inlet -
                TARGET_INLET_TEMP_C;

        double requested_setpoint =
                INITIAL_SETPOINT_C -
                COOLING_SETPOINT_RESET_C_C *
                        temperature_error;

        requested_setpoint =
                clamp_double(
                        requested_setpoint,
                        MIN_COOLING_SETPOINT_C,
                        MAX_COOLING_SETPOINT_C);

        double setpoint_delta =
                requested_setpoint -
                rack->cooling_setpoint_c;

        setpoint_delta =
                clamp_double(
                        setpoint_delta,
                        -COOLING_SETPOINT_RATE_C_S *
                                DT_SECONDS,
                        COOLING_SETPOINT_RATE_C_S *
                                DT_SECONDS);

        double next_setpoint =
                rack->cooling_setpoint_c +
                setpoint_delta;

        double requested_cooling =
                COOLING_PROPORTIONAL_W_K *
                rack->coolant_flow *
                fmax(rack->coolant_temp_c -
                             next_setpoint,
                     0.0);

        requested_cooling =
                clamp_double(
                        requested_cooling,
                        0.0,
                        MAX_COOLING_REMOVAL_W);

        double previous_cooling =
                rack->cooling_removal_w;

        double cooling_derivative =
                (requested_cooling -
                 previous_cooling) /
                COOLING_ACTUATOR_TIME_S;

        double next_cooling =
                previous_cooling +
                DT_SECONDS *
                cooling_derivative;

        next_cooling =
                clamp_double(
                        next_cooling,
                        0.0,
                        MAX_COOLING_REMOVAL_W);

        double coolant_derivative =
                (heat_to_coolant -
                 previous_cooling) /
                COOLANT_HEAT_CAPACITY_J_K;

        double next_coolant =
                rack->coolant_temp_c +
                DT_SECONDS *
                coolant_derivative;

        double average_cooling =
                0.5 *
                (previous_cooling +
                 next_cooling);

        rack->inlet_temp_c =
                next_inlet;

        rack->outlet_temp_c =
                next_outlet;

        rack->coolant_temp_c =
                next_coolant;

        rack->cooling_setpoint_c =
                next_setpoint;

        rack->cooling_removal_w =
                next_cooling;

        return (average_cooling /
                COOLING_COP) *
               DT_SECONDS;
}

/*******************************************************************************
 * Simulation ticks
 ******************************************************************************/

static void run_scheduler_step(
        struct sim *s,
        dstm_time_t t)
{
        REQUIRE(s->is_scheduler,
                "scheduler step on thermal rank");

        if (t < s->control_latency)
                return;

        int snapshot_time =
                (int)t -
                s->control_latency;

        if (snapshot_time %
                    s->schedule_window !=
            0) {
                return;
        }

        int epoch_start =
                snapshot_time +
                s->schedule_window;

        collect_feedback(
                s,
                t,
                snapshot_time);

        if (epoch_start >= s->steps)
                return;

        plan_and_publish_epoch(
                s,
                t,
                epoch_start);
}

static void run_thermal_step(
        struct sim *s,
        dstm_time_t t)
{
        REQUIRE(!s->is_scheduler,
                "thermal step on scheduler");

        /*
         * Feedback uses its own region, so it may be pushed before graph
         * reads without participating in a graph version's phase.
         */
        if (t %
                    s->schedule_window ==
            0) {
                publish_feedback(s, t);
        }

        /*
         * Version t of the primary heat graph contains:
         *
         *   - rack vertex state published at t-1 with delta 1;
         *   - delay-one heat published at t-1 with delta 1.
         */
        read_current_racks(s, t);

        if (t >=
                    s->schedule_window &&
            t %
                    s->schedule_window ==
            0) {
                receive_schedule_block(
                        s,
                        t);
        }

        apply_schedule_tick(s, t);
        update_local_power(s);

        for (size_t r = 0;
             r < s->local_racks;
             r++) {
                double heat =
                        receive_neighbor_heat(
                                s,
                                r,
                                t);

                s->cooling_energy_joules +=
                        integrate_rack_thermal(
                                &s->racks[r],
                                heat);

                if (s->racks[r].inlet_temp_c >
                    s->max_inlet_temp_c) {
                        s->max_inlet_temp_c =
                                s->racks[r]
                                 .inlet_temp_c;
                }

                if (s->racks[r].outlet_temp_c >
                    s->max_outlet_temp_c) {
                        s->max_outlet_temp_c =
                                s->racks[r]
                                 .outlet_temp_c;
                }
        }

        /*
         * Publish all future state only after consuming current state.
         */
        publish_next_rack_state(s, t);
        send_delayed_heat(s, t);
}

static void run_step(
        struct sim *s,
        dstm_time_t t)
{
        if (s->is_scheduler)
                run_scheduler_step(s, t);
        else
                run_thermal_step(s, t);
}

/*******************************************************************************
 * DSTM setup
 ******************************************************************************/

static size_t heat_graph_payload_estimate(
        const struct sim *s,
        int delay)
{
        size_t local_edges = 0;

        for (size_t i = 0;
             i < s->thermal_vertex_count;
             i++) {
                local_edges +=
                        s->heat_degrees[delay][i];
        }

        size_t vertex_bytes =
                delay == 1 ?
                s->thermal_vertex_count *
                        sizeof(struct rack_state) :
                0;

        /*
         * Dgraph edge slots contain a uint64_t destination, payload, and
         * alignment padding. This is a conservative estimate.
         */
        size_t edge_stride =
                sizeof(uint64_t) +
                sizeof(struct heat_message) +
                16;

        size_t row_bytes =
                (s->thermal_vertex_count + 1) *
                sizeof(uint64_t);

        return vertex_bytes +
               row_bytes +
               local_edges *
                        edge_stride +
               4096;
}

static int setup(
        struct sim *s,
        const char *job_path)
{
        REQUIRE(g_world_size >= 2,
                "requires scheduler and thermal ranks");

        s->is_scheduler =
                g_rank == SCHEDULER_RANK;

        s->thermal_rank_count =
                g_world_size - 1;

        s->thermal_rank =
                s->is_scheduler ?
                -1 :
                g_rank - 1;

        REQUIRE(s->total_racks >= 3,
                "ring requires at least three racks");

        REQUIRE(s->total_racks %
                        s->thermal_rank_count ==
                0,
                "racks must divide thermal ranks");

        REQUIRE(s->nodes_per_rack > 0 &&
                s->nodes_per_rack <=
                        MAX_NODES_PER_RACK,
                "invalid nodes-per-rack");

        REQUIRE(s->schedule_window > 0,
                "schedule window must be positive");

        REQUIRE(s->control_latency > 0,
                "control latency must be positive");

        REQUIRE(2 * s->control_latency <=
                        s->schedule_window,
                "control round trip exceeds window");

        s->local_racks =
                (size_t)s->total_racks /
                (size_t)s->thermal_rank_count;

                s->rack_offset =
                s->is_scheduler ?
                0 :
                s->local_racks *
                        (size_t)s->thermal_rank;

        /*
         * append_worker_schedule_events() emits at most one event per node
         * per tick. Therefore this is the strict worst-case capacity for one
         * worker's scheduling window.
         */
        REQUIRE((size_t)s->schedule_window <=
                        SIZE_MAX / worker_node_count(s),
                "schedule event capacity overflow");

        s->schedule_event_capacity =
                (size_t)s->schedule_window *
                worker_node_count(s);

        REQUIRE(s->schedule_event_capacity <= (size_t)INT_MAX,
                "schedule event capacity exceeds header representation");

        s->schedule_event_cursor = 0;

        s->max_inlet_temp_c =
                -INFINITY;

        s->max_outlet_temp_c =
                -INFINITY;

        initialize_racks(s);
        initialize_scheduler_state(s);
        build_heat_topologies(s);

        if (load_scheduler_jobs(
                    s,
                    job_path) != 0) {
                return -1;
        }

        if (build_distance_matrix(s) != 0)
                return -1;

        /*
         * Region zero is the schedule region. Each rank owns exactly one
         * schedule element.
         */
        s->schedule_region_id =
                dstm_get_local_region_id();

        size_t schedule_payload =
                schedule_block_size(s) +
                4096;

        dstm_opts_t opts =
                dstm_default_options;

        opts.region_size =
                padded_region_size(
                        schedule_payload);

        CHECK(dstm_start_custom(
                      s->distance,
                      opts),
              "dstm_start_custom");

        /*
         * Register all additional regions before any DSTM operation can
         * advance the rank heartbeat.
         *
         * Feedback ownership:
         *   scheduler owns thermal_rank_count slots;
         * every thermal rank owns one unused dummy slot.
         */
        size_t feedback_slots =
                s->is_scheduler ?
                (size_t)s->thermal_rank_count :
                1;

        size_t feedback_payload =
                feedback_slots *
                feedback_block_size(s) +
                4096;

        CHECK(dstm_region_register(
                      padded_region_size(
                              feedback_payload),
                      &s->feedback_region_id),
              "feedback region register");

        for (int delay = 1;
             delay <= MAX_THERMAL_DELAY;
             delay++) {
                CHECK(dstm_region_register(
                              padded_region_size(
                                      heat_graph_payload_estimate(
                                              s,
                                              delay)),
                              &s->heat_region_id[delay]),
                      "heat region register");
        }

        /*
         * Create schedule and feedback arrays collectively.
         */
        CHECK(dstm_darray_create_on_region(
                      &s->schedule_array,
                      schedule_block_size(s),
                      1,
                      s->schedule_region_id),
              "schedule array create");

        CHECK(dstm_darray_create_on_region(
                      &s->feedback_array,
                      feedback_block_size(s),
                      feedback_slots,
                      s->feedback_region_id),
              "feedback array create");

        s->schedule_global_offset =
                dstm_darray_global_offset(
                        s->schedule_array);

        s->feedback_global_offset =
                dstm_darray_global_offset(
                        s->feedback_array);

        REQUIRE(s->schedule_global_offset ==
                        (size_t)g_rank,
                "schedule rank/index mapping");

        s->feedback_scheduler_offset = 0;

        if (s->is_scheduler) {
                REQUIRE(s->feedback_global_offset == 0,
                        "scheduler feedback offset");
        } else {
                size_t expected =
                        (size_t)s->thermal_rank_count +
                        (size_t)s->thermal_rank;

                REQUIRE(s->feedback_global_offset ==
                                expected,
                        "worker feedback dummy offset");
        }

        /*
         * Each delay graph uses the same vertex partition but a topology
         * containing only edges of that delay.
         *
         * Delay-one also carries rack vertex state.
         */
        for (int delay = 1;
             delay <= MAX_THERMAL_DELAY;
             delay++) {
                size_t vertex_state_size =
                        delay == 1 ?
                        sizeof(struct rack_state) :
                        0;

                const void *initial_vertices =
                        delay == 1 ?
                        s->thermal_vertices :
                        NULL;

                CHECK(dstm_dgraph_create_on_region(
                              &s->heat_graph[delay],
                              vertex_state_size,
                              sizeof(struct heat_message),
                              s->thermal_vertex_count,
                              s->heat_degrees[delay],
                              s->heat_destinations[delay],
                              initial_vertices,
                              NULL,
                              s->heat_region_id[delay]),
                      "delay heat graph create");
        }

        s->schedule_buffer =
                calloc(1,
                       schedule_block_size(s));

        s->feedback_buffer =
                calloc(1,
                       feedback_block_size(s));

        REQUIRE(s->schedule_buffer != NULL,
                "schedule buffer allocation");

        REQUIRE(s->feedback_buffer != NULL,
                "feedback buffer allocation");

        /*
         * Seed every locally owned schedule element at time zero.
         */
        CHECK(dstm_darray_put(
                      s->schedule_array,
                      s->schedule_global_offset,
                      s->schedule_buffer,
                      0,
                      0),
              "initial schedule put");

        /*
         * Seed scheduler-owned feedback slots. Worker dummy elements are
         * also seeded locally.
         */
        if (s->is_scheduler) {
                for (int thermal_rank = 0;
                     thermal_rank <
                             s->thermal_rank_count;
                     thermal_rank++) {
                        CHECK(dstm_darray_put(
                                      s->feedback_array,
                                      s->feedback_scheduler_offset +
                                              (size_t)thermal_rank,
                                      s->feedback_buffer,
                                      0,
                                      0),
                              "initial scheduler feedback put");
                }
        } else {
                CHECK(dstm_darray_put(
                              s->feedback_array,
                              s->feedback_global_offset,
                              s->feedback_buffer,
                              0,
                              0),
                      "initial feedback dummy put");
        }

        return 0;
}

/*******************************************************************************
 * Teardown
 ******************************************************************************/

static void teardown_dstm(struct sim *s)
{
        if (s->feedback_array != NULL)
                dstm_darray_destroy(
                        &s->feedback_array);

        if (s->schedule_array != NULL)
                dstm_darray_destroy(
                        &s->schedule_array);

        for (int delay = 1;
             delay <= MAX_THERMAL_DELAY;
             delay++) {
                if (s->heat_graph[delay] != NULL) {
                        dstm_dgraph_destroy(
                                &s->heat_graph[delay]);
                }
        }
}

static void teardown_local(struct sim *s)
{
        dstm_distance_destroy(
                &s->distance);

        for (int delay = 1;
             delay <= MAX_THERMAL_DELAY;
             delay++) {
                free(s->heat_degrees[delay]);
                free(s->heat_destinations[delay]);
        }

        if (s->is_scheduler)
                free(s->thermal_vertices);

        free(s->racks);
        free(s->jobs);
        free(s->completion_heap);

        free(s->scheduler_nodes);
        free(s->scheduler_feedback);

        free(s->schedule_buffer);
        free(s->feedback_buffer);
}

/*******************************************************************************
 * Statistics
 ******************************************************************************/

static void print_statistics(const struct sim *s)
{
        uint64_t local_counts[2] = {
                s->jobs_started,
                s->jobs_completed,
        };

        uint64_t global_counts[2] =
                {0, 0};

        MPI_Reduce(local_counts,
                   global_counts,
                   2,
                   MPI_UINT64_T,
                   MPI_SUM,
                   0,
                   MPI_COMM_WORLD);

        double total_it_energy = 0.0;
        double total_cooling_energy = 0.0;
        double max_inlet = 0.0;
        double max_outlet = 0.0;

        double local_max_inlet =
                s->is_scheduler ?
                -INFINITY :
                s->max_inlet_temp_c;

        double local_max_outlet =
                s->is_scheduler ?
                -INFINITY :
                s->max_outlet_temp_c;

        MPI_Reduce(&s->it_energy_joules,
                   &total_it_energy,
                   1,
                   MPI_DOUBLE,
                   MPI_SUM,
                   0,
                   MPI_COMM_WORLD);

        MPI_Reduce(&s->cooling_energy_joules,
                   &total_cooling_energy,
                   1,
                   MPI_DOUBLE,
                   MPI_SUM,
                   0,
                   MPI_COMM_WORLD);

        MPI_Reduce(&local_max_inlet,
                   &max_inlet,
                   1,
                   MPI_DOUBLE,
                   MPI_MAX,
                   0,
                   MPI_COMM_WORLD);

        MPI_Reduce(&local_max_outlet,
                   &max_outlet,
                   1,
                   MPI_DOUBLE,
                   MPI_MAX,
                   0,
                   MPI_COMM_WORLD);

        if (g_rank == 0) {
                printf("\n=== DSTM datacenter results ===\n");

                printf("jobs started       : %" PRIu64 "\n",
                       global_counts[0]);

                printf("jobs completed     : %" PRIu64 "\n",
                       global_counts[1]);

                printf("IT energy          : %.3f kWh\n",
                       total_it_energy /
                               3600000.0);

                printf("cooling energy     : %.3f kWh\n",
                       total_cooling_energy /
                               3600000.0);

                printf("facility energy    : %.3f kWh\n",
                       (total_it_energy +
                        total_cooling_energy) /
                               3600000.0);

                printf("maximum inlet temp : %.3f C\n",
                       max_inlet);

                printf("maximum outlet temp: %.3f C\n",
                       max_outlet);

                printf("scheduler planning : %.6f seconds\n",
                       s->scheduler_planning_seconds);

                printf("schedule publishing: %.6f seconds\n",
                       s->scheduler_publication_seconds);

                printf("schedule events    : %" PRIu64 "\n",
                       s->schedule_events_emitted);

                printf("event capacity     : %" PRIu64 "\n",
                       s->schedule_event_capacity_total);

                if (s->schedule_event_capacity_total > 0) {
                        printf("event occupancy    : %.3f%%\n",
                               100.0 *
                               (double)s->schedule_events_emitted /
                               (double)s->schedule_event_capacity_total);
                }

                printf("===============================\n");
        }
}

/*******************************************************************************
 * Command line
 ******************************************************************************/

static void usage(const char *program)
{
        fprintf(stderr,
                "usage: %s --jobs FILE [options]\n"
                "\n"
                "options:\n"
                "  --jobs FILE           CSV job trace, required\n"
                "  --racks N             total racks, default %d\n"
                "  --nodes-per-rack N    nodes per rack, default %d\n"
                "  --steps N             simulated seconds, default %d\n"
                "  --schedule-window N   scheduling epoch, default %d\n"
                "  --control-latency N   one-way control delay, default %d\n"
                "  --verbose-dstm        enable DSTM debug logging\n"
                "  --help                show this message\n",
                program,
                DEFAULT_RACKS,
                DEFAULT_NODES_PER_RACK,
                DEFAULT_STEPS,
                DEFAULT_SCHEDULE_WINDOW,
                DEFAULT_CONTROL_LATENCY);
}

/*******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char **argv)
{
        struct sim sim;
        const char *job_path = NULL;
        int verbose_dstm = 0;

        memset(&sim, 0, sizeof(sim));

        sim.total_racks =
                DEFAULT_RACKS;

        sim.nodes_per_rack =
                DEFAULT_NODES_PER_RACK;

        sim.steps =
                DEFAULT_STEPS;

        sim.schedule_window =
                DEFAULT_SCHEDULE_WINDOW;

        sim.control_latency =
                DEFAULT_CONTROL_LATENCY;

        MPI_Init(&argc, &argv);
        dstm_init(&argc, &argv);

        MPI_Comm_rank(
                MPI_COMM_WORLD,
                &g_rank);

        MPI_Comm_size(
                MPI_COMM_WORLD,
                &g_world_size);

        for (int i = 1;
             i < argc;
             i++) {
                if (!strcmp(argv[i], "--jobs") &&
                    i + 1 < argc) {
                        job_path =
                                argv[++i];
                } else if (!strcmp(argv[i], "--racks") &&
                           i + 1 < argc) {
                        sim.total_racks =
                                atoi(argv[++i]);
                } else if (!strcmp(argv[i], "--nodes-per-rack") &&
                           i + 1 < argc) {
                        sim.nodes_per_rack =
                                atoi(argv[++i]);
                } else if (!strcmp(argv[i], "--steps") &&
                           i + 1 < argc) {
                        sim.steps =
                                atoi(argv[++i]);
                } else if (!strcmp(argv[i], "--schedule-window") &&
                           i + 1 < argc) {
                        sim.schedule_window =
                                atoi(argv[++i]);
                } else if (!strcmp(argv[i], "--control-latency") &&
                           i + 1 < argc) {
                        sim.control_latency =
                                atoi(argv[++i]);
                } else if (!strcmp(argv[i], "--verbose-dstm")) {
                        verbose_dstm = 1;
                } else if (!strcmp(argv[i], "--help") ||
                           !strcmp(argv[i], "-h")) {
                        if (g_rank == 0)
                                usage(argv[0]);

                        dstm_finalize();
                        MPI_Finalize();
                        return 0;
                } else {
                        if (g_rank == 0) {
                                fprintf(stderr,
                                        "unknown argument: %s\n",
                                        argv[i]);
                        }

                        dstm_finalize();
                        MPI_Finalize();
                        return 1;
                }
        }

        if (job_path == NULL) {
                if (g_rank == 0) {
                        fprintf(stderr,
                                "--jobs FILE is required\n\n");
                        usage(argv[0]);
                }

                dstm_finalize();
                MPI_Finalize();
                return 1;
        }

        if (sim.steps < 1)
                sim.steps = 1;

        dstm_log_setlevel(
                verbose_dstm ?
                DSTM_LOG_DEBUG :
                DSTM_LOG_ERROR);

        if (setup(&sim, job_path) != 0) {
                fprintf(stderr,
                        "[rank %d] setup failed\n",
                        g_rank);

                MPI_Abort(
                        MPI_COMM_WORLD,
                        1);
        }

        if (g_rank == 0) {
                printf("DSTM push-based datacenter simulation\n");
                printf("MPI ranks       : %d\n",
                       g_world_size);
                printf("scheduler ranks : 1\n");
                printf("thermal ranks   : %d\n",
                       sim.thermal_rank_count);
                printf("racks           : %d\n",
                       sim.total_racks);
                printf("nodes/rack      : %d\n",
                       sim.nodes_per_rack);
                printf("seconds         : %d\n",
                       sim.steps);
                printf("schedule window : %d seconds\n",
                       sim.schedule_window);
                printf("control latency : %d seconds\n",
                       sim.control_latency);
                printf("thermal delays  : 1..%d seconds\n",
                       MAX_THERMAL_DELAY);
                printf("warm-up interval: [0, %d)\n",
                       sim.schedule_window);
                printf("heat regions    : %d delay-separated regions\n",
                       MAX_THERMAL_DELAY);
                fflush(stdout);
        }

        double start =
                MPI_Wtime();

        for (dstm_time_t t = 0;
             t < sim.steps;
             t++) {
                run_step(&sim, t);
        }

        /*
         * Last heat:
         *
         *     origin    = steps - 1
         *     effective = steps - 1 + MAX_THERMAL_DELAY
         *
         * Last control publication may extend by schedule_window. Use a
         * conservative common final timestamp for the drain.
         */
        dstm_time_t last_time =
                (dstm_time_t)sim.steps +
                sim.schedule_window +
                MAX_THERMAL_DELAY +
                1;

        /*
         * Match the lifecycle used by the working graph-traffic example:
         * destroy graph/array handles before dstm_fini destroys regions.
         */
        teardown_dstm(&sim);

        CHECK(dstm_fini(last_time),
              "dstm_fini");

        double end =
                MPI_Wtime();

        print_statistics(&sim);

        if (g_rank == 0) {
                printf("wall-clock runtime  : %.9f seconds\n",
                       end - start);
        }

        teardown_local(&sim);

        dstm_finalize();
        MPI_Finalize();

        return 0;
}
