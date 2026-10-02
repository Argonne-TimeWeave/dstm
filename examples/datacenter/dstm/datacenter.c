/*******************************************************************************
 * DSTM datacenter job/power/thermal example
 *
 * Job trace format, without a header:
 *
 *   arrival,duration,nodes,cpu_per_node,gpu_per_node,cpu_util,gpu_util
 *
 * Example:
 *
 *   0,20,2,16,1,0.75,0.90
 *   5,30,1,32,4,0.95,0.80
 *
 * Run:
 *
 *   mpirun -n 4 ./datacenter \
 *       --racks 16 \
 *       --nodes-per-rack 8 \
 *       --steps 300 \
 *       --jobs jobs.csv
 *
 * Assumptions:
 *
 *   - one tick is one simulated second;
 *   - racks are arranged in a ring;
 *   - each rank owns a contiguous set of racks;
 *   - job id modulo world_size selects its scheduling/cooling domain;
 *   - jobs are placed only within their assigned domain;
 *   - heat channels use receiver-owned DSTM edges:
 *
 *         source u:
 *             dstm_dgraph_edge_put(g, v, u, ..., t, delay)
 *
 *         destination v:
 *             dstm_dgraph_edge_get(g, v, u, ..., t, 0)
 *
 *   - every channel has exactly one writer and one reader;
 *   - every heat channel is populated at every effective timestamp.
 ******************************************************************************/

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dstm.h"

/*******************************************************************************
 * Model limits
 ******************************************************************************/

#define MAX_NODES_PER_RACK  32
#define MAX_JOB_NODES       64
#define MAX_THERMAL_DELAY   3

#define DEFAULT_RACKS          16
#define DEFAULT_NODES_PER_RACK 8
#define DEFAULT_STEPS          300

#define NODE_CPU_CAPACITY       64
#define NODE_GPU_CAPACITY        8

/*
 * Benchmark model references:
 *
 * Power:
 *   Fan, Weber, and Barroso, "Power Provisioning for a
 *   Warehouse-sized Computer," ISCA 2007.
 *
 * Thermal formulation:
 *   ASHRAE TC 9.9, Thermal Guidelines for Data Processing
 *   Environments. The implementation uses the standard lumped-capacitance
 *   energy balance:
 *
 *     C dT/dt = sum(heat inputs) - sum(heat outputs).
 *
 * SST does not prescribe these equations. The same equations and parameters
 * must be copied independently into the SST rack component.
 *
 * The numerical values below are benchmark parameters. They must be fitted
 * against measurements before results can be described as calibrated.
 */

#define NODE_IDLE_POWER_W       180.0
#define CPU_DYNAMIC_POWER_W     320.0
#define GPU_DYNAMIC_POWER_W    2400.0

#define DT_SECONDS                 1.0

#define AMBIENT_TEMP_C            22.0
#define INITIAL_COOLANT_TEMP_C    18.0
#define INITIAL_OUTLET_TEMP_C     24.0
#define INITIAL_SETPOINT_C        18.0
#define TARGET_INLET_TEMP_C       24.0

/* Effective thermal masses, J/K. */
#define RACK_HEAT_CAPACITY_J_K       120000.0
#define COOLANT_HEAT_CAPACITY_J_K    500000.0

/* Heat-transfer coefficients, W/K. */
#define AIRFLOW_HEAT_CAPACITY_W_K      8000.0
#define RACK_TO_COOLANT_UA_W_K         6000.0

/*
 * Cooling controller and actuator.
 *
 * The coolant setpoint is reset from inlet-temperature error. Actual cooling
 * removal approaches the requested removal with a first-order actuator.
 */
#define COOLING_SETPOINT_RESET_C_C        0.10
#define COOLING_SETPOINT_RATE_C_S         0.25
#define MIN_COOLING_SETPOINT_C           12.0
#define MAX_COOLING_SETPOINT_C           24.0

#define COOLING_PROPORTIONAL_W_K      12000.0
#define MAX_COOLING_REMOVAL_W        100000.0
#define COOLING_ACTUATOR_TIME_S          20.0

/*
 * Constant cooling coefficient of performance:
 *
 *     cooling electrical power = removed heat / COP
 *
 * A calibrated implementation may replace this with a measured
 * temperature-dependent chiller curve, but both DSTM and SST must use the
 * exact same curve.
 */
#define COOLING_COP                       4.0

/* Fraction of exhaust enthalpy routed to neighboring rack inlets. */
#define HEAT_RECIRCULATION_FRACTION       0.04

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
 * Versioned state
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

        /*
         * Actual thermal power currently removed by the cooling system.
         * This is versioned because the cooling plant has a finite response
         * time.
         */
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

/*
 * Rank 0 is the only runtime writer of schedule slots.
 *
 * Each rank owns one dstm_darray element. Rank 0 writes element r at
 * (t, +1), and rank r reads its local element at (t + 1, 0).
 */
struct scheduled_node {
        int cpu_used;
        int gpu_used;
        int running_jobs;

        double cpu_utilization;
        double gpu_utilization;
};

struct schedule_header {
        int effective_time;
        int rack_count;
        int nodes_per_rack;
};

/*
 * The schedule element has variable-sized trailing storage:
 *
 *   struct schedule_header
 *   struct scheduled_node nodes[local_racks * nodes_per_rack]
 */

/*******************************************************************************
 * Simulation context
 ******************************************************************************/

struct sim {
        int total_racks;
        int nodes_per_rack;
        int steps;

        size_t local_racks;
        size_t rack_offset;

        struct rack_state *racks;

        /*
         * Only rank 0 has real jobs. Other ranks have no job records.
         */
        struct job_state *jobs;
        size_t job_count;

        /*
         * Authoritative reservation model, present only on rank 0.
         * Indexed by:
         *
         *   global_rack * nodes_per_rack + node
         */
        struct node_state *scheduler_nodes;

        dstm_distance_t *distance;
        dstm_dgraph_t *thermal_graph;

        /*
         * One fixed-size schedule element per rank.
         */
        dstm_darray_t *schedule_array;
        size_t schedule_global_offset;
        void *schedule_buffer;

        size_t *degrees;
        size_t *destinations;

        uint64_t jobs_started;
        uint64_t jobs_completed;

        double it_energy_joules;
        double cooling_energy_joules;

        double max_inlet_temp_c;
        double max_outlet_temp_c;
};

/*******************************************************************************
 * Utility helpers
 ******************************************************************************/

static double clamp_double(double value, double low, double high)
{
        if (value < low)
                return low;
        if (value > high)
                return high;
        return value;
}

static int rack_owner(const struct sim *s, size_t rack)
{
        return (int)(rack / s->local_racks);
}

static size_t ring_left(const struct sim *s, size_t rack)
{
        return (rack + (size_t)s->total_racks - 1) %
               (size_t)s->total_racks;
}

static size_t ring_right(const struct sim *s, size_t rack)
{
        return (rack + 1) % (size_t)s->total_racks;
}

/*
 * A deterministic physical propagation delay of one to three seconds.
 * Both directions between a pair use the same delay.
 */
static int thermal_delay(size_t a, size_t b)
{
        size_t lo = a < b ? a : b;
        size_t hi = a < b ? b : a;
        uint64_t hash = (uint64_t)lo * 2654435761ULL +
                        (uint64_t)hi * 2246822519ULL;

        return 1 + (int)(hash % MAX_THERMAL_DELAY);
}

static size_t schedule_node_count(const struct sim *s)
{
        return s->local_racks * (size_t)s->nodes_per_rack;
}

static size_t schedule_element_size(const struct sim *s)
{
        return sizeof(struct schedule_header) +
               schedule_node_count(s) * sizeof(struct scheduled_node);
}

static struct scheduled_node *schedule_nodes(void *buffer)
{
        return (struct scheduled_node *)
                ((unsigned char *)buffer +
                 sizeof(struct schedule_header));
}

static const struct scheduled_node *schedule_nodes_const(
        const void *buffer)
{
        return (const struct scheduled_node *)
                ((const unsigned char *)buffer +
                 sizeof(struct schedule_header));
}

static struct node_state *scheduler_node(
        struct sim *s,
        size_t global_rack,
        int node)
{
        REQUIRE(g_rank == 0, "scheduler state accessed outside rank 0");
        REQUIRE(global_rack < (size_t)s->total_racks,
                "invalid global scheduler rack");
        REQUIRE(node >= 0 && node < s->nodes_per_rack,
                "invalid global scheduler node");

        return &s->scheduler_nodes[
                global_rack * (size_t)s->nodes_per_rack +
                (size_t)node];
}

static void initialize_global_scheduler(struct sim *s)
{
        if (g_rank != 0)
                return;

        size_t count =
                (size_t)s->total_racks *
                (size_t)s->nodes_per_rack;

        s->scheduler_nodes =
                calloc(count, sizeof(*s->scheduler_nodes));

        REQUIRE(s->scheduler_nodes != NULL,
                "global scheduler node allocation");

        for (size_t i = 0; i < count; i++) {
                s->scheduler_nodes[i].cpu_capacity =
                        NODE_CPU_CAPACITY;
                s->scheduler_nodes[i].gpu_capacity =
                        NODE_GPU_CAPACITY;
                s->scheduler_nodes[i].power_watts =
                        NODE_IDLE_POWER_W;
        }
}

/*******************************************************************************
 * Job trace loading
 ******************************************************************************/

static int load_local_jobs(struct sim *s, const char *path)
{
        FILE *file = fopen(path, "r");
        if (file == NULL) {
                fprintf(stderr, "[rank %d] cannot open job trace %s\n",
                        g_rank, path);
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

                int n = fscanf(file, " %d,%d,%d,%d,%d,%lf,%lf",
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
                                "[rank %d] malformed job record near id %d\n",
                                g_rank, id);
                        fclose(file);
                        free(jobs);
                        return -1;
                }

                REQUIRE(arrival >= 0, "negative arrival time");
                REQUIRE(duration > 0, "job duration must be positive");
                REQUIRE(nodes > 0 && nodes <= MAX_JOB_NODES,
                        "requested node count exceeds MAX_JOB_NODES");
                REQUIRE(cpu >= 0 && cpu <= NODE_CPU_CAPACITY,
                        "invalid CPU request");
                REQUIRE(gpu >= 0 && gpu <= NODE_GPU_CAPACITY,
                        "invalid GPU request");

                /*
                 * Deterministically assign each job to one rank's cooling
                 * domain. This avoids a centralized scheduler becoming a
                 * synchronization bottleneck while preserving explicit
                 * arrival/allocation/start/completion semantics.
                 */
                if (g_rank == 0) {
                        if (count == capacity) {
                                capacity = capacity ? capacity * 2 : 32;
                                jobs = realloc(jobs,
                                               capacity * sizeof(*jobs));
                                REQUIRE(jobs != NULL, "job allocation");
                        }

                        jobs[count] = (struct job_state) {
                                .valid = 1,
                                .id = id,
                                .arrival_time = arrival,
                                .duration = duration,
                                .requested_nodes = nodes,
                                .cpu_per_node = cpu,
                                .gpu_per_node = gpu,
                                .cpu_utilization =
                                        clamp_double(cpu_util, 0.0, 1.0),
                                .gpu_utilization =
                                        clamp_double(gpu_util, 0.0, 1.0),
                                .status = JOB_FUTURE,
                                .start_time = -1,
                                .completion_time = -1,
                                .allocation_count = 0,
                        };

                        count++;
                }

                id++;
        }

        fclose(file);

        s->job_count = count;

        if (count > 0) {
                s->jobs = malloc(count * sizeof(*s->jobs));
                REQUIRE(s->jobs != NULL, "global job storage");
                memcpy(s->jobs, jobs, count * sizeof(*jobs));
        }

        free(jobs);
        return 0;
}

/*******************************************************************************
 * Rack topology
 ******************************************************************************/

static void build_local_thermal_topology(struct sim *s)
{
        s->degrees = malloc(s->local_racks * sizeof(*s->degrees));
        s->destinations =
                malloc(s->local_racks * 2 * sizeof(*s->destinations));

        REQUIRE(s->degrees != NULL, "degree allocation");
        REQUIRE(s->destinations != NULL, "destination allocation");

        for (size_t i = 0; i < s->local_racks; i++) {
                size_t rack = s->rack_offset + i;
                size_t a = ring_left(s, rack);
                size_t b = ring_right(s, rack);

                s->degrees[i] = 2;

                /* Dgraph requires strictly ascending destinations. */
                if (a < b) {
                        s->destinations[i * 2] = a;
                        s->destinations[i * 2 + 1] = b;
                } else {
                        s->destinations[i * 2] = b;
                        s->destinations[i * 2 + 1] = a;
                }
        }
}

/*******************************************************************************
 * Distance matrix
 *
 * One matrix is constructed for every physical delay value, then all are
 * combined. This preserves:
 *
 *   readable bound  = minimum reachable delay;
 *   discard bound   = maximum reachable delay.
 ******************************************************************************/

static int build_distance_matrix(struct sim *s)
{
        dstm_distance_t *parts[MAX_THERMAL_DELAY + 1] = {0};
        int *matrices[MAX_THERMAL_DELAY] = {0};
        int ws = g_world_size;

        for (int d = 1; d <= MAX_THERMAL_DELAY; d++) {
                int *m = malloc((size_t)ws * (size_t)ws * sizeof(*m));
                if (m == NULL)
                        return -1;

                matrices[d - 1] = m;

                for (int i = 0; i < ws; i++) {
                        for (int j = 0; j < ws; j++)
                                m[(size_t)i * ws + j] =
                                        (i == j) ? 0 : -1;
                }
        }

        for (size_t u = 0; u < (size_t)s->total_racks; u++) {
                size_t neighbors[2] = {
                        ring_left(s, u),
                        ring_right(s, u),
                };

                for (int k = 0; k < 2; k++) {
                        size_t v = neighbors[k];
                        int ru = rack_owner(s, u);
                        int rv = rack_owner(s, v);

                        if (ru == rv)
                                continue;

                        int delay = thermal_delay(u, v);
                        int *m = matrices[delay - 1];

                        m[(size_t)ru * ws + rv] = delay;
                        m[(size_t)rv * ws + ru] = delay;
                }
        }

        for (int d = 1; d <= MAX_THERMAL_DELAY; d++) {
                int rc = dstm_distance_create_from_matrix(
                        &parts[d - 1],
                        (size_t)ws,
                        matrices[d - 1]);

                free(matrices[d - 1]);

                if (rc != DSTM_SUCCESS)
                        return -1;
        }

        /*
         * Global scheduler control network.
         *
         * Rank 0 must be able to write every rank's schedule with a
         * one-tick delay.
         */
        int *control =
                malloc((size_t)ws * (size_t)ws * sizeof(*control));

        if (control == NULL)
                return -1;

        for (int i = 0; i < ws; i++) {
                for (int j = 0; j < ws; j++) {
                        control[(size_t)i * ws + j] =
                                (i == j) ? 0 : 1;
                }
        }

        int control_rc = dstm_distance_create_from_matrix(
                &parts[MAX_THERMAL_DELAY],
                (size_t)ws,
                control);

        free(control);

        if (control_rc != DSTM_SUCCESS)
                return -1;

        const dstm_distance_t *inputs[MAX_THERMAL_DELAY + 1];

        for (int i = 0; i < MAX_THERMAL_DELAY + 1; i++)
                inputs[i] = parts[i];

        int rc = dstm_distance_combine(
                &s->distance,
                inputs,
                MAX_THERMAL_DELAY + 1);

        for (int i = 0; i < MAX_THERMAL_DELAY + 1; i++)
                dstm_distance_destroy(&parts[i]);

        return rc == DSTM_SUCCESS ? 0 : -1;
}

/*******************************************************************************
 * Initial state
 ******************************************************************************/

static void initialize_racks(struct sim *s)
{
        s->racks = calloc(s->local_racks, sizeof(*s->racks));
        REQUIRE(s->racks != NULL, "rack allocation");

        for (size_t i = 0; i < s->local_racks; i++) {
                struct rack_state *rack = &s->racks[i];

                rack->rack_id = (int)(s->rack_offset + i);
                rack->node_count = s->nodes_per_rack;

                rack->inlet_temp_c = AMBIENT_TEMP_C;
                rack->outlet_temp_c = INITIAL_OUTLET_TEMP_C;
                rack->coolant_temp_c = INITIAL_COOLANT_TEMP_C;
                rack->coolant_flow = 1.0;
                rack->cooling_setpoint_c = INITIAL_SETPOINT_C;

                for (int n = 0; n < rack->node_count; n++) {
                        rack->nodes[n].cpu_capacity = NODE_CPU_CAPACITY;
                        rack->nodes[n].gpu_capacity = NODE_GPU_CAPACITY;
                        rack->nodes[n].power_watts =
                                NODE_IDLE_POWER_W;
                }
        }
}

/*
 * Dgraph construction seeds zero edge state at effective time zero, but an
 * edge with delay d will not receive its first runtime message until time d.
 * Populate versions 1 through d-1 so receivers have a value at every tick.
 */
static void prefill_heat_pipeline(struct sim *s)
{
        struct heat_message zero = {0};

        for (size_t i = 0; i < s->local_racks; i++) {
                size_t source = s->rack_offset + i;
                size_t neighbors[2] = {
                        ring_left(s, source),
                        ring_right(s, source),
                };

                for (int n = 0; n < 2; n++) {
                        size_t destination = neighbors[n];
                        int delay = thermal_delay(source, destination);

                        zero.source_rack = (int)source;
                        zero.destination_rack = (int)destination;
                        zero.generation_time = 0;
                        zero.heat_watts = 0.0;

                        for (int effective = 1;
                             effective < delay;
                             effective++) {
                                zero.arrival_time = effective;

                                CHECK(dstm_dgraph_edge_put(
                                              s->thermal_graph,
                                              destination,
                                              source,
                                              &zero,
                                              0,
                                              effective),
                                      "prefill heat pipeline");
                        }
                }
        }
}

/*******************************************************************************
 * Version reads
 ******************************************************************************/

static void read_current_racks(struct sim *s, dstm_time_t t)
{
        for (size_t i = 0; i < s->local_racks; i++) {
                size_t rack_id = s->rack_offset + i;

                CHECK(dstm_dgraph_vertex_get(s->thermal_graph,
                                             rack_id,
                                             &s->racks[i],
                                             t,
                                             0),
                      "rack vertex get");
        }
}

static void receive_global_schedule(struct sim *s, dstm_time_t t)
{
        memset(s->schedule_buffer, 0,
               schedule_element_size(s));

        CHECK(dstm_darray_get(
                      s->schedule_array,
                      s->schedule_global_offset,
                      s->schedule_buffer,
                      t,
                      0),
              "schedule get");

        const struct schedule_header *header =
                (const struct schedule_header *)s->schedule_buffer;

        /*
         * Ignore a stale/default version. Rank 0 populates every effective
         * timestamp after time zero.
         */
        if (header->effective_time != (int)t)
                return;

        REQUIRE(header->rack_count == (int)s->local_racks,
                "schedule rack count mismatch");
        REQUIRE(header->nodes_per_rack == s->nodes_per_rack,
                "schedule node count mismatch");

        const struct scheduled_node *input =
                schedule_nodes_const(s->schedule_buffer);

        size_t index = 0;

        for (size_t r = 0; r < s->local_racks; r++) {
                for (int n = 0; n < s->nodes_per_rack; n++, index++) {
                        struct node_state *node =
                                &s->racks[r].nodes[n];

                        node->cpu_used =
                                input[index].cpu_used;
                        node->gpu_used =
                                input[index].gpu_used;
                        node->running_jobs =
                                input[index].running_jobs;

                        node->cpu_utilization =
                                input[index].cpu_utilization;
                        node->gpu_utilization =
                                input[index].gpu_utilization;
                }
        }
}

static void publish_global_schedule(struct sim *s, dstm_time_t t)
{
        REQUIRE(g_rank == 0,
                "only rank 0 may publish schedules");

        int effective_time = (int)t + 1;
        size_t bytes = schedule_element_size(s);
        void *buffer = calloc(1, bytes);

        REQUIRE(buffer != NULL,
                "schedule publication buffer");

        struct schedule_header *header =
                (struct schedule_header *)buffer;

        header->effective_time = effective_time;
        header->rack_count = (int)s->local_racks;
        header->nodes_per_rack = s->nodes_per_rack;

        struct scheduled_node *output =
                schedule_nodes(buffer);

        for (int destination_rank = 0;
             destination_rank < g_world_size;
             destination_rank++) {
                size_t first_global_rack =
                        (size_t)destination_rank *
                        s->local_racks;

                size_t index = 0;

                for (size_t r = 0; r < s->local_racks; r++) {
                        size_t global_rack =
                                first_global_rack + r;

                        for (int n = 0;
                             n < s->nodes_per_rack;
                             n++, index++) {
                                const struct node_state *node =
                                        scheduler_node(
                                                s,
                                                global_rack,
                                                n);

                                output[index].cpu_used =
                                        node->cpu_used;
                                output[index].gpu_used =
                                        node->gpu_used;
                                output[index].running_jobs =
                                        node->running_jobs;

                                output[index].cpu_utilization =
                                        node->cpu_utilization;
                                output[index].gpu_utilization =
                                        node->gpu_utilization;
                        }
                }

                /*
                 * Every rank owns one element, and every rank contributes
                 * exactly one element, so destination_rank is its global
                 * array index.
                 */
                CHECK(dstm_darray_put(
                              s->schedule_array,
                              (size_t)destination_rank,
                              buffer,
                              t,
                              1),
                      "global schedule put");
        }

        free(buffer);
}

/*******************************************************************************
 * Scheduler
 ******************************************************************************/

static void release_completed_jobs(struct sim *s, int effective_time)
{
        REQUIRE(g_rank == 0,
                "only rank 0 may release global jobs");

        for (size_t j = 0; j < s->job_count; j++) {
                struct job_state *job = &s->jobs[j];

                if (job->status != JOB_RUNNING)
                        continue;
                if (job->completion_time > effective_time)
                        continue;

                for (int a = 0; a < job->allocation_count; a++) {
                        int rack = job->allocations[a].global_rack;
                        int node_index = job->allocations[a].node;

                        struct node_state *node =
                                scheduler_node(s,
                                               (size_t)rack,
                                               node_index);

                        node->cpu_used -= job->cpu_per_node;
                        node->gpu_used -= job->gpu_per_node;
                        node->running_jobs--;

                        assert(node->cpu_used >= 0);
                        assert(node->gpu_used >= 0);
                        assert(node->running_jobs >= 0);
                }

                job->allocation_count = 0;
                job->status = JOB_COMPLETE;
                s->jobs_completed++;
        }
}

static void admit_arrivals(struct sim *s, int arrival_time)
{
        REQUIRE(g_rank == 0,
                "only rank 0 may admit global jobs");

        for (size_t j = 0; j < s->job_count; j++) {
                struct job_state *job = &s->jobs[j];

                if (job->status == JOB_FUTURE &&
                    job->arrival_time <= arrival_time)
                        job->status = JOB_QUEUED;
        }
}

static int node_can_host(const struct node_state *node,
                         const struct job_state *job)
{
        int cpu_free = node->cpu_capacity - node->cpu_used;
        int gpu_free = node->gpu_capacity - node->gpu_used;

        return cpu_free >= job->cpu_per_node &&
               gpu_free >= job->gpu_per_node;
}

static int try_start_job(struct sim *s,
                         struct job_state *job,
                         int effective_time)
{
        struct allocation candidates[MAX_JOB_NODES];
        int candidate_count = 0;

        REQUIRE(g_rank == 0,
                "only rank 0 may place global jobs");

        /*
         * Deterministic global first-fit ordering:
         *
         *   global rack 0..N-1
         *   node 0..nodes_per_rack-1
         */
        for (size_t rack = 0;
             rack < (size_t)s->total_racks &&
             candidate_count < job->requested_nodes;
             rack++) {
                for (int node_index = 0;
                     node_index < s->nodes_per_rack &&
                     candidate_count < job->requested_nodes;
                     node_index++) {
                        struct node_state *node =
                                scheduler_node(s, rack, node_index);

                        if (!node_can_host(node, job))
                                continue;

                        candidates[candidate_count++] =
                                (struct allocation) {
                                        .global_rack = (int)rack,
                                        .node = node_index,
                                };
                }
        }

        if (candidate_count != job->requested_nodes)
                return 0;

        for (int a = 0; a < candidate_count; a++) {
                struct allocation allocation = candidates[a];

                struct node_state *node =
                        scheduler_node(
                                s,
                                (size_t)allocation.global_rack,
                                allocation.node);

                node->cpu_used += job->cpu_per_node;
                node->gpu_used += job->gpu_per_node;
                node->running_jobs++;

                job->allocations[a] = allocation;
        }

        job->allocation_count = candidate_count;
        job->start_time = effective_time;
        job->completion_time =
                effective_time + job->duration;
        job->status = JOB_RUNNING;

        s->jobs_started++;
        return 1;
}

static void schedule_queued_jobs(struct sim *s, int effective_time)
{
        REQUIRE(g_rank == 0,
                "only rank 0 may schedule global jobs");

        for (size_t j = 0; j < s->job_count; j++) {
                struct job_state *job = &s->jobs[j];

                if (job->status != JOB_QUEUED)
                        continue;

                if (!try_start_job(s, job, effective_time))
                        break;
        }
}

/*******************************************************************************
 * Utilization and power
 ******************************************************************************/

static void rebuild_global_scheduler_utilization(struct sim *s)
{
        REQUIRE(g_rank == 0,
                "only rank 0 may rebuild global utilization");

        size_t node_count =
                (size_t)s->total_racks *
                (size_t)s->nodes_per_rack;

        for (size_t i = 0; i < node_count; i++) {
                s->scheduler_nodes[i].cpu_utilization = 0.0;
                s->scheduler_nodes[i].gpu_utilization = 0.0;
        }

        for (size_t j = 0; j < s->job_count; j++) {
                const struct job_state *job = &s->jobs[j];

                if (job->status != JOB_RUNNING)
                        continue;

                for (int a = 0; a < job->allocation_count; a++) {
                        int rack = job->allocations[a].global_rack;
                        int node_index = job->allocations[a].node;

                        struct node_state *node =
                                scheduler_node(s,
                                               (size_t)rack,
                                               node_index);

                        node->cpu_utilization +=
                                ((double)job->cpu_per_node /
                                 (double)node->cpu_capacity) *
                                job->cpu_utilization;

                        node->gpu_utilization +=
                                ((double)job->gpu_per_node /
                                 (double)node->gpu_capacity) *
                                job->gpu_utilization;
                }
        }

        for (size_t i = 0; i < node_count; i++) {
                struct node_state *node = &s->scheduler_nodes[i];

                node->cpu_utilization =
                        clamp_double(node->cpu_utilization,
                                     0.0, 1.0);

                node->gpu_utilization =
                        clamp_double(node->gpu_utilization,
                                     0.0, 1.0);
        }
}

static void update_local_power(struct sim *s)
{
        for (size_t r = 0; r < s->local_racks; r++) {
                struct rack_state *rack = &s->racks[r];

                rack->total_power_watts = 0.0;

                for (int n = 0; n < rack->node_count; n++) {
                        struct node_state *node = &rack->nodes[n];

                        node->power_watts =
                                NODE_IDLE_POWER_W +
                                CPU_DYNAMIC_POWER_W *
                                        node->cpu_utilization +
                                GPU_DYNAMIC_POWER_W *
                                        node->gpu_utilization;

                        rack->total_power_watts += node->power_watts;
                }

                s->it_energy_joules +=
                        rack->total_power_watts * DT_SECONDS;
        }
}

/*******************************************************************************
 * Heat reception and RC integration
 ******************************************************************************/

static double receive_neighbor_heat(struct sim *s,
                                    size_t local_rack,
                                    dstm_time_t t)
{
        size_t destination = s->rack_offset + local_rack;
        size_t neighbors[2] = {
                ring_left(s, destination),
                ring_right(s, destination),
        };

        double incoming_heat = 0.0;

        for (int n = 0; n < 2; n++) {
                struct heat_message message = {0};
                size_t source = neighbors[n];

                CHECK(dstm_dgraph_edge_get(s->thermal_graph,
                                           destination,
                                           source,
                                           &message,
                                           t,
                                           0),
                      "heat edge get");

                /*
                 * The timestamp guard prevents an accidentally reused or
                 * malformed channel value from being counted at the wrong
                 * effective timestamp.
                 */
                if (message.arrival_time == t)
                        incoming_heat += message.heat_watts;
        }

        return incoming_heat;
}

static double integrate_rack_thermal(struct rack_state *rack,
                                     double incoming_heat_watts)
{
        /*
         * Delayed recirculated exhaust heat raises the rack inlet above
         * ambient:
         *
         *     T_in = T_ambient + Q_recirculated / (m_dot * c_p)
         *
         * AIRFLOW_HEAT_CAPACITY_W_K represents m_dot * c_p.
         */
        double next_inlet =
                AMBIENT_TEMP_C +
                incoming_heat_watts /
                        AIRFLOW_HEAT_CAPACITY_W_K;

        /*
         * Heat transported out of the rack by its airflow:
         *
         *     Q_air = (m_dot * c_p) * (T_out - T_in)
         */
        double heat_to_air =
                AIRFLOW_HEAT_CAPACITY_W_K *
                (rack->outlet_temp_c - next_inlet);

        /*
         * Conductive/convective heat transfer to the coolant loop:
         *
         *     Q_coolant = UA * (T_out - T_coolant)
         *
         * This value is intentionally signed to preserve the energy balance.
         */
        double heat_to_coolant =
                RACK_TO_COOLANT_UA_W_K *
                (rack->outlet_temp_c -
                 rack->coolant_temp_c);

        /*
         * Rack energy balance:
         *
         *     C_rack dT_out/dt =
         *         P_IT - Q_air - Q_coolant
         *
         * Recirculated heat is not added separately here: it has already
         * entered the balance through T_in. Adding it again would count the
         * same heat twice.
         */
        double outlet_derivative =
                (rack->total_power_watts -
                 heat_to_air -
                 heat_to_coolant) /
                RACK_HEAT_CAPACITY_J_K;

        double next_outlet =
                rack->outlet_temp_c +
                DT_SECONDS * outlet_derivative;

        /*
         * Reset the coolant setpoint from the inlet-temperature error.
         *
         * This is an absolute reset around INITIAL_SETPOINT_C rather than an
         * accumulation into the previous setpoint. Consequently, a constant
         * error does not cause unintended integral windup.
         */
        double temperature_error =
                next_inlet - TARGET_INLET_TEMP_C;

        double requested_setpoint =
                INITIAL_SETPOINT_C -
                COOLING_SETPOINT_RESET_C_C *
                        temperature_error;

        requested_setpoint =
                clamp_double(requested_setpoint,
                             MIN_COOLING_SETPOINT_C,
                             MAX_COOLING_SETPOINT_C);

        double setpoint_delta =
                requested_setpoint -
                rack->cooling_setpoint_c;

        setpoint_delta =
                clamp_double(
                        setpoint_delta,
                        -COOLING_SETPOINT_RATE_C_S * DT_SECONDS,
                        COOLING_SETPOINT_RATE_C_S * DT_SECONDS);

        double next_setpoint =
                rack->cooling_setpoint_c +
                setpoint_delta;

        /*
         * Requested cooling removal is proportional to coolant temperature
         * above its setpoint and is limited by plant capacity.
         */
        double requested_cooling =
                COOLING_PROPORTIONAL_W_K *
                rack->coolant_flow *
                fmax(rack->coolant_temp_c -
                     next_setpoint,
                     0.0);

        requested_cooling =
                clamp_double(requested_cooling,
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
                DT_SECONDS * cooling_derivative;

        next_cooling =
                clamp_double(next_cooling,
                             0.0,
                             MAX_COOLING_REMOVAL_W);

        double coolant_derivative =
                (heat_to_coolant -
                 previous_cooling) /
                COOLANT_HEAT_CAPACITY_J_K;

        double next_coolant =
                rack->coolant_temp_c +
                DT_SECONDS * coolant_derivative;

        double average_cooling =
                0.5 * (previous_cooling +
                       next_cooling);

        rack->inlet_temp_c = next_inlet;
        rack->outlet_temp_c = next_outlet;
        rack->coolant_temp_c = next_coolant;
        rack->cooling_setpoint_c = next_setpoint;
        rack->cooling_removal_w = next_cooling;

        return (average_cooling / COOLING_COP) *
               DT_SECONDS;
}

/*******************************************************************************
 * Future-state publication
 ******************************************************************************/

static void publish_next_state(struct sim *s, dstm_time_t t)
{
        for (size_t i = 0; i < s->local_racks; i++) {
                size_t rack_id = s->rack_offset + i;

                CHECK(dstm_dgraph_vertex_put(s->thermal_graph,
                                             rack_id,
                                             &s->racks[i],
                                             t,
                                             1),
                      "rack vertex put");
        }
}

static void send_delayed_heat(struct sim *s, dstm_time_t t)
{
        for (size_t i = 0; i < s->local_racks; i++) {
                size_t source = s->rack_offset + i;
                size_t neighbors[2] = {
                        ring_left(s, source),
                        ring_right(s, source),
                };

                /*
                 * Excess exhaust-air enthalpy relative to ambient:
                 *
                 *     Q_exhaust = (m_dot * c_p) *
                 *                 max(T_out - T_ambient, 0)
                 *
                 * A fixed fraction is recirculated and divided evenly
                 * between the two neighboring racks. The DSTM edge delay
                 * represents its physical propagation time.
                 */
                double exhaust_heat =
                        AIRFLOW_HEAT_CAPACITY_W_K *
                        fmax(s->racks[i].outlet_temp_c -
                             AMBIENT_TEMP_C,
                             0.0);

                double heat_per_neighbor =
                        HEAT_RECIRCULATION_FRACTION *
                        exhaust_heat / 2.0;

                for (int n = 0; n < 2; n++) {
                        size_t destination = neighbors[n];
                        int delay = thermal_delay(source, destination);

                        struct heat_message message = {
                                .source_rack = (int)source,
                                .destination_rack = (int)destination,
                                .generation_time = (int)t,
                                .arrival_time = (int)t + delay,
                                .heat_watts = heat_per_neighbor,
                        };

                        CHECK(dstm_dgraph_edge_put(s->thermal_graph,
                                                   destination,
                                                   source,
                                                   &message,
                                                   t,
                                                   delay),
                              "heat edge put");
                }
        }
}

/*******************************************************************************
 * One simulation tick
 ******************************************************************************/

static void run_step(struct sim *s, dstm_time_t t)
{
        /*
         * First load this rank's thermal state and the schedule that rank 0
         * published during the preceding tick.
         */
        read_current_racks(s, t);
        receive_global_schedule(s, t);

        /*
         * The received utilization drives local IT power and thermal state.
         */
        update_local_power(s);

        for (size_t r = 0; r < s->local_racks; r++) {
                double heat = receive_neighbor_heat(s, r, t);

                s->cooling_energy_joules +=
                        integrate_rack_thermal(&s->racks[r], heat);

                if (s->racks[r].inlet_temp_c >
                    s->max_inlet_temp_c)
                        s->max_inlet_temp_c =
                                s->racks[r].inlet_temp_c;

                if (s->racks[r].outlet_temp_c >
                    s->max_outlet_temp_c)
                        s->max_outlet_temp_c =
                                s->racks[r].outlet_temp_c;
        }

        /*
         * Rank 0 computes the schedule for effective time t+1.
         *
         * Jobs arriving at t therefore begin consuming resources at t+1.
         * A duration-D job started at S is absent from the snapshot for
         * effective time S+D.
         */
        if (g_rank == 0) {
                int effective_time = (int)t + 1;

                release_completed_jobs(s, effective_time);
                admit_arrivals(s, (int)t);
                schedule_queued_jobs(s, effective_time);
                rebuild_global_scheduler_utilization(s);
                publish_global_schedule(s, t);
        }

        publish_next_state(s, t);
        send_delayed_heat(s, t);
}

/*******************************************************************************
 * Setup and teardown
 ******************************************************************************/

static int setup(struct sim *s, const char *job_path)
{
        REQUIRE(s->total_racks >= 3,
                "ring topology requires at least three racks");

        REQUIRE(s->total_racks % g_world_size == 0,
                "rack count must be divisible by MPI world size");

        REQUIRE(s->nodes_per_rack > 0 &&
                s->nodes_per_rack <= MAX_NODES_PER_RACK,
                "invalid nodes-per-rack");

        s->local_racks =
                (size_t)s->total_racks / (size_t)g_world_size;

        s->rack_offset =
                s->local_racks * (size_t)g_rank;

        s->max_inlet_temp_c = -INFINITY;
        s->max_outlet_temp_c = -INFINITY;

        initialize_racks(s);
        initialize_global_scheduler(s);
        build_local_thermal_topology(s);

        if (load_local_jobs(s, job_path) != 0)
                return -1;

        if (build_distance_matrix(s) != 0)
                return -1;

        dstm_opts_t opts = dstm_default_options;

        /*
         * Increase this if rack/job state or version history exceeds the
         * library default region.
         */
        opts.region_size = 1ULL << 28;

        CHECK(dstm_start_custom(s->distance, opts),
              "dstm_start_custom");

        CHECK(dstm_dgraph_create(
                      &s->thermal_graph,
                      sizeof(struct rack_state),
                      sizeof(struct heat_message),
                      s->local_racks,
                      s->degrees,
                      s->destinations,
                      s->racks,
                      NULL),
              "thermal graph create");

        /*
         * Every rank owns exactly one schedule element, so the global
         * index of rank r's element is r.
         */
        CHECK(dstm_darray_create(
                      &s->schedule_array,
                      schedule_element_size(s),
                      1),
              "schedule array create");

        s->schedule_global_offset =
                dstm_darray_global_offset(s->schedule_array);

        REQUIRE(s->schedule_global_offset == (size_t)g_rank,
                "schedule array rank/index mapping");

        s->schedule_buffer =
                calloc(1, schedule_element_size(s));

        REQUIRE(s->schedule_buffer != NULL,
                "schedule buffer allocation");

        prefill_heat_pipeline(s);

        struct schedule_header *initial =
                (struct schedule_header *)s->schedule_buffer;

        initial->effective_time = 0;
        initial->rack_count = (int)s->local_racks;
        initial->nodes_per_rack = s->nodes_per_rack;

        CHECK(dstm_darray_put(
                      s->schedule_array,
                      s->schedule_global_offset,
                      s->schedule_buffer,
                      0,
                      0),
              "initial schedule put");

        return 0;
}

static void teardown_dstm(struct sim *s)
{
        if (s->schedule_array != NULL)
                dstm_darray_destroy(&s->schedule_array);

        if (s->thermal_graph != NULL)
                dstm_dgraph_destroy(&s->thermal_graph);
}

static void teardown_local(struct sim *s)
{
        dstm_distance_destroy(&s->distance);

        free(s->degrees);
        free(s->destinations);
        free(s->racks);
        free(s->jobs);
        free(s->scheduler_nodes);
        free(s->schedule_buffer);
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

        uint64_t global_counts[2] = {0, 0};

        MPI_Reduce(local_counts,
                   global_counts,
                   2,
                   MPI_UINT64_T,
                   MPI_SUM,
                   0,
                   MPI_COMM_WORLD);

        double local_values[4] = {
                s->it_energy_joules,
                s->cooling_energy_joules,
                s->max_inlet_temp_c,
                s->max_outlet_temp_c,
        };

        double total_it_energy = 0.0;
        double total_cooling_energy = 0.0;
        double max_inlet = 0.0;
        double max_outlet = 0.0;

        MPI_Reduce(&local_values[0],
                   &total_it_energy,
                   1,
                   MPI_DOUBLE,
                   MPI_SUM,
                   0,
                   MPI_COMM_WORLD);

        MPI_Reduce(&local_values[1],
                   &total_cooling_energy,
                   1,
                   MPI_DOUBLE,
                   MPI_SUM,
                   0,
                   MPI_COMM_WORLD);

        MPI_Reduce(&local_values[2],
                   &max_inlet,
                   1,
                   MPI_DOUBLE,
                   MPI_MAX,
                   0,
                   MPI_COMM_WORLD);

        MPI_Reduce(&local_values[3],
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
                       total_it_energy / 3600000.0);
                printf("cooling energy     : %.3f kWh\n",
                       total_cooling_energy / 3600000.0);
                printf("facility energy    : %.3f kWh\n",
                       (total_it_energy +
                        total_cooling_energy) / 3600000.0);
                printf("maximum inlet temp : %.3f C\n",
                       max_inlet);
                printf("maximum outlet temp: %.3f C\n",
                       max_outlet);
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
                "  --verbose-dstm        enable DSTM informational logging\n"
                "  --help                show this message\n",
                program,
                DEFAULT_RACKS,
                DEFAULT_NODES_PER_RACK,
                DEFAULT_STEPS);
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

        sim.total_racks = DEFAULT_RACKS;
        sim.nodes_per_rack = DEFAULT_NODES_PER_RACK;
        sim.steps = DEFAULT_STEPS;

        MPI_Init(&argc, &argv);
        dstm_init(&argc, &argv);

        MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &g_world_size);

        for (int i = 1; i < argc; i++) {
                if (!strcmp(argv[i], "--jobs") && i + 1 < argc) {
                        job_path = argv[++i];
                } else if (!strcmp(argv[i], "--racks") &&
                           i + 1 < argc) {
                        sim.total_racks = atoi(argv[++i]);
                } else if (!strcmp(argv[i], "--nodes-per-rack") &&
                           i + 1 < argc) {
                        sim.nodes_per_rack = atoi(argv[++i]);
                } else if (!strcmp(argv[i], "--steps") &&
                           i + 1 < argc) {
                        sim.steps = atoi(argv[++i]);
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
                        if (g_rank == 0)
                                fprintf(stderr,
                                        "unknown argument: %s\n",
                                        argv[i]);

                        dstm_finalize();
                        MPI_Finalize();
                        return 1;
                }
        }

        if (job_path == NULL) {
                if (g_rank == 0) {
                        fprintf(stderr, "--jobs FILE is required\n\n");
                        usage(argv[0]);
                }

                dstm_finalize();
                MPI_Finalize();
                return 1;
        }

        if (sim.steps < 1)
                sim.steps = 1;

        dstm_log_setlevel(verbose_dstm
                          ? DSTM_LOG_INFO
                          : DSTM_LOG_QUIET);

        if (setup(&sim, job_path) != 0) {
                fprintf(stderr, "[rank %d] setup failed\n", g_rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (g_rank == 0) {
                printf("DSTM datacenter simulation\n");
                printf("ranks          : %d\n", g_world_size);
                printf("racks          : %d\n", sim.total_racks);
                printf("nodes/rack     : %d\n", sim.nodes_per_rack);
                printf("seconds        : %d\n", sim.steps);
                printf("thermal delays : 1..%d seconds\n",
                       MAX_THERMAL_DELAY);
                fflush(stdout);
        }

        
        double start = MPI_Wtime();

        for (dstm_time_t t = 0; t < sim.steps; t++)
                run_step(&sim, t);

        /*
         * The final runtime heat puts originate at steps-1 and may target
         * steps-1+MAX_THERMAL_DELAY. Advance beyond that effective time.
         */
        dstm_time_t last_time =
                (dstm_time_t)sim.steps +
                MAX_THERMAL_DELAY +
                1;

        /*
         * Follow the lifecycle used by the supplied DSTM examples:
         * destroy DSTM-backed containers before dstm_fini destroys regions.
         */
        teardown_dstm(&sim);
        CHECK(dstm_fini(last_time), "dstm_fini");

        double end = MPI_Wtime();

        print_statistics(&sim);

        if (g_rank == 0)
                printf("simulation runtime  : %.6f seconds\n", end - start);

        teardown_local(&sim);

        dstm_finalize();
        MPI_Finalize();
        return 0;
}
