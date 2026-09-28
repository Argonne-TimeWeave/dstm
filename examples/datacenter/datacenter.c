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

/*******************************************************************************
 * Node/power parameters
 ******************************************************************************/

#define NODE_CPU_CAPACITY       64
#define NODE_GPU_CAPACITY        8

#define NODE_IDLE_POWER_W    180.0
#define CPU_DYNAMIC_POWER_W  320.0
#define GPU_DYNAMIC_POWER_W 2400.0

/*******************************************************************************
 * Rack/thermal parameters
 *
 * These are illustrative lumped-RC values and should be calibrated against
 * exaDigit or facility measurements for a production experiment.
 ******************************************************************************/

#define DT_SECONDS                 1.0
#define AMBIENT_TEMP_C            22.0
#define INITIAL_COOLANT_TEMP_C    18.0
#define INITIAL_OUTLET_TEMP_C     24.0
#define INITIAL_SETPOINT_C        18.0
#define TARGET_INLET_TEMP_C       24.0

/* Rack thermal capacitance, J/K. */
#define RACK_HEAT_CAPACITY_J_K 120000.0

/* Coolant-loop thermal capacitance, J/K. */
#define COOLANT_HEAT_CAPACITY_J_K 500000.0

/* Rack-to-coolant thermal conductance, W/K. */
#define RACK_UA_W_K                6000.0

/* Effective airflow heat-capacity rate, W/K. */
#define AIRFLOW_W_K                8000.0

/* Cooling-system response. */
#define COOLING_GAIN_W_K          12000.0
#define COOLING_CONTROL_KP            0.10
#define COOLING_SETPOINT_RATE_C_S      0.25
#define MIN_COOLING_SETPOINT_C        12.0
#define MAX_COOLING_SETPOINT_C        24.0

/* Fraction of source-rack electrical power recirculated to neighbors. */
#define HEAT_RECIRCULATION_FRACTION    0.04

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
};

struct allocation {
        int local_rack;
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

        struct job_state *jobs;
        size_t local_job_slots;
        size_t local_real_jobs;

        dstm_distance_t *distance;
        dstm_dgraph_t *thermal_graph;
        dstm_darray_t *job_array;

        size_t *degrees;
        size_t *destinations;

        size_t job_global_offset;

        uint64_t jobs_started;
        uint64_t jobs_completed;

        double energy_joules;
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
                if ((id % g_world_size) == g_rank) {
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

        /*
         * dstm_darray_create requires every rank to own at least one element.
         * A rank with no real jobs therefore receives one unused slot.
         */
        s->local_real_jobs = count;
        s->local_job_slots = count ? count : 1;
        s->jobs = calloc(s->local_job_slots, sizeof(*s->jobs));
        REQUIRE(s->jobs != NULL, "local job storage");

        if (count > 0)
                memcpy(s->jobs, jobs, count * sizeof(*jobs));

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
        dstm_distance_t *parts[MAX_THERMAL_DELAY] = {0};
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

        const dstm_distance_t *inputs[MAX_THERMAL_DELAY];

        for (int i = 0; i < MAX_THERMAL_DELAY; i++)
                inputs[i] = parts[i];

        int rc = dstm_distance_combine(
                &s->distance,
                inputs,
                MAX_THERMAL_DELAY);

        for (int i = 0; i < MAX_THERMAL_DELAY; i++)
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

static void publish_initial_jobs(struct sim *s)
{
        for (size_t i = 0; i < s->local_job_slots; i++) {
                size_t global_index = s->job_global_offset + i;

                CHECK(dstm_darray_put(s->job_array,
                                      global_index,
                                      &s->jobs[i],
                                      0,
                                      0),
                      "initial job put");
        }
}

/*******************************************************************************
 * Version reads
 ******************************************************************************/

static void read_current_versions(struct sim *s, dstm_time_t t)
{
        /*
         * Read rack versions owned by this rank. A delta-zero access is
         * legal because every vertex is local.
         */
        for (size_t i = 0; i < s->local_racks; i++) {
                size_t rack_id = s->rack_offset + i;

                CHECK(dstm_dgraph_vertex_get(s->thermal_graph,
                                             rack_id,
                                             &s->racks[i],
                                             t,
                                             0),
                      "rack vertex get");
        }

        for (size_t i = 0; i < s->local_job_slots; i++) {
                size_t global_index = s->job_global_offset + i;

                CHECK(dstm_darray_get(s->job_array,
                                      global_index,
                                      &s->jobs[i],
                                      t,
                                      0),
                      "job state get");
        }
}

/*******************************************************************************
 * Scheduler
 ******************************************************************************/

static void release_completed_jobs(struct sim *s, int t)
{
        for (size_t j = 0; j < s->local_real_jobs; j++) {
                struct job_state *job = &s->jobs[j];

                if (job->status != JOB_RUNNING)
                        continue;
                if (job->completion_time > t)
                        continue;

                for (int a = 0; a < job->allocation_count; a++) {
                        int r = job->allocations[a].local_rack;
                        int n = job->allocations[a].node;

                        struct node_state *node =
                                &s->racks[r].nodes[n];

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

static void admit_arrivals(struct sim *s, int t)
{
        for (size_t j = 0; j < s->local_real_jobs; j++) {
                struct job_state *job = &s->jobs[j];

                if (job->status == JOB_FUTURE &&
                    job->arrival_time <= t)
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

static int try_start_job(struct sim *s, struct job_state *job, int t)
{
        struct allocation candidates[MAX_JOB_NODES];
        int candidate_count = 0;

        for (size_t r = 0;
             r < s->local_racks &&
             candidate_count < job->requested_nodes;
             r++) {
                struct rack_state *rack = &s->racks[r];

                for (int n = 0;
                     n < rack->node_count &&
                     candidate_count < job->requested_nodes;
                     n++) {
                        if (!node_can_host(&rack->nodes[n], job))
                                continue;

                        candidates[candidate_count++] =
                                (struct allocation) {
                                        .local_rack = (int)r,
                                        .node = n,
                                };
                }
        }

        if (candidate_count != job->requested_nodes)
                return 0;

        for (int a = 0; a < candidate_count; a++) {
                int r = candidates[a].local_rack;
                int n = candidates[a].node;
                struct node_state *node = &s->racks[r].nodes[n];

                node->cpu_used += job->cpu_per_node;
                node->gpu_used += job->gpu_per_node;
                node->running_jobs++;

                job->allocations[a] = candidates[a];
        }

        job->allocation_count = candidate_count;
        job->start_time = t;
        job->completion_time = t + job->duration;
        job->status = JOB_RUNNING;

        s->jobs_started++;
        return 1;
}

static void schedule_queued_jobs(struct sim *s, int t)
{
        /*
         * SST-style first-fit event semantics:
         *
         *   arrival -> queued -> allocation/start -> timed completion.
         *
         * A production implementation can replace this loop with SST 2024's
         * scheduler/allocation policy while retaining the DSTM state flow.
         */
        for (size_t j = 0; j < s->local_real_jobs; j++) {
                if (s->jobs[j].status == JOB_QUEUED)
                        (void)try_start_job(s, &s->jobs[j], t);
        }
}

/*******************************************************************************
 * Utilization and power
 ******************************************************************************/

static void update_utilization_and_power(struct sim *s)
{
        /*
         * Capacity reservation and utilization are kept separate:
         *
         *   cpu_used       reserves scheduler capacity;
         *   utilization    describes the active workload fraction;
         *   power          is calculated from utilization.
         */
        for (size_t r = 0; r < s->local_racks; r++) {
                struct rack_state *rack = &s->racks[r];

                for (int n = 0; n < rack->node_count; n++) {
                        rack->nodes[n].cpu_utilization = 0.0;
                        rack->nodes[n].gpu_utilization = 0.0;
                }
        }

        for (size_t j = 0; j < s->local_real_jobs; j++) {
                const struct job_state *job = &s->jobs[j];

                if (job->status != JOB_RUNNING)
                        continue;

                for (int a = 0; a < job->allocation_count; a++) {
                        int r = job->allocations[a].local_rack;
                        int n = job->allocations[a].node;
                        struct node_state *node = &s->racks[r].nodes[n];

                        double cpu_share =
                                ((double)job->cpu_per_node /
                                 (double)node->cpu_capacity) *
                                job->cpu_utilization;

                        double gpu_share =
                                ((double)job->gpu_per_node /
                                 (double)node->gpu_capacity) *
                                job->gpu_utilization;

                        node->cpu_utilization += cpu_share;
                        node->gpu_utilization += gpu_share;
                }
        }

        for (size_t r = 0; r < s->local_racks; r++) {
                struct rack_state *rack = &s->racks[r];

                rack->total_power_watts = 0.0;

                for (int n = 0; n < rack->node_count; n++) {
                        struct node_state *node = &rack->nodes[n];

                        node->cpu_utilization =
                                clamp_double(node->cpu_utilization,
                                             0.0, 1.0);

                        node->gpu_utilization =
                                clamp_double(node->gpu_utilization,
                                             0.0, 1.0);

                        node->power_watts =
                                NODE_IDLE_POWER_W +
                                CPU_DYNAMIC_POWER_W *
                                        node->cpu_utilization +
                                GPU_DYNAMIC_POWER_W *
                                        node->gpu_utilization;

                        rack->total_power_watts += node->power_watts;
                }

                s->energy_joules +=
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

static void integrate_rack_thermal(struct rack_state *rack,
                                   double incoming_heat_watts)
{
        /*
         * Inlet temperature from ambient air plus delayed recirculated heat.
         */
        double inlet_target =
                AMBIENT_TEMP_C +
                incoming_heat_watts / AIRFLOW_W_K;

        /*
         * Rack outlet lumped-RC equation:
         *
         * C_rack dT_out/dt =
         *      electrical_power
         *    + incoming_heat
         *    - UA (T_out - T_coolant)
         */
        double rack_removed =
                RACK_UA_W_K *
                (rack->outlet_temp_c - rack->coolant_temp_c);

        double outlet_derivative =
                (rack->total_power_watts +
                 incoming_heat_watts -
                 rack_removed) /
                RACK_HEAT_CAPACITY_J_K;

        double next_outlet =
                rack->outlet_temp_c +
                DT_SECONDS * outlet_derivative;

        /*
         * First-order inlet response. This avoids an instantaneous jump
         * while still using a one-second integration step.
         */
        double next_inlet =
                rack->inlet_temp_c +
                0.5 * (inlet_target - rack->inlet_temp_c);

        /*
         * Cooling controller:
         *
         * A hot inlet lowers the coolant setpoint. The rate limiter models
         * finite cooling-system response.
         */
        double temperature_error =
                next_inlet - TARGET_INLET_TEMP_C;

        double requested_setpoint =
                rack->cooling_setpoint_c -
                COOLING_CONTROL_KP * temperature_error;

        requested_setpoint =
                clamp_double(requested_setpoint,
                             MIN_COOLING_SETPOINT_C,
                             MAX_COOLING_SETPOINT_C);

        double setpoint_delta =
                requested_setpoint - rack->cooling_setpoint_c;

        setpoint_delta =
                clamp_double(setpoint_delta,
                             -COOLING_SETPOINT_RATE_C_S * DT_SECONDS,
                             COOLING_SETPOINT_RATE_C_S * DT_SECONDS);

        double next_setpoint =
                rack->cooling_setpoint_c + setpoint_delta;

        /*
         * Coolant-loop RC equation:
         *
         * C_coolant dT_coolant/dt =
         *      rack_to_coolant_heat
         *    - cooling_system_removal
         */
        double cooling_removed =
                COOLING_GAIN_W_K *
                rack->coolant_flow *
                fmax(rack->coolant_temp_c - next_setpoint, 0.0);

        double coolant_derivative =
                (rack_removed - cooling_removed) /
                COOLANT_HEAT_CAPACITY_J_K;

        double next_coolant =
                rack->coolant_temp_c +
                DT_SECONDS * coolant_derivative;

        rack->inlet_temp_c = next_inlet;
        rack->outlet_temp_c = next_outlet;
        rack->coolant_temp_c = next_coolant;
        rack->cooling_setpoint_c = next_setpoint;
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

        for (size_t i = 0; i < s->local_job_slots; i++) {
                size_t global_index = s->job_global_offset + i;

                CHECK(dstm_darray_put(s->job_array,
                                      global_index,
                                      &s->jobs[i],
                                      t,
                                      1),
                      "job state put");
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
                 * Split recirculated power evenly between both neighboring
                 * racks.
                 */
                double heat_per_neighbor =
                        HEAT_RECIRCULATION_FRACTION *
                        s->racks[i].total_power_watts / 2.0;

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
         * All reads target effective time t. All state writes target t+1,
         * while heat writes target t+physical_delay. This preserves DSTM's
         * put-before-get rule for each effective timestamp.
         */
        read_current_versions(s, t);

        /* SST-style event order. */
        release_completed_jobs(s, (int)t);
        admit_arrivals(s, (int)t);
        schedule_queued_jobs(s, (int)t);

        update_utilization_and_power(s);

        for (size_t r = 0; r < s->local_racks; r++) {
                double heat = receive_neighbor_heat(s, r, t);

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

        CHECK(dstm_darray_create(
                      &s->job_array,
                      sizeof(struct job_state),
                      s->local_job_slots),
              "job array create");

        s->job_global_offset =
                dstm_darray_global_offset(s->job_array);

        publish_initial_jobs(s);
        prefill_heat_pipeline(s);

        return 0;
}

static void teardown_dstm(struct sim *s)
{
        if (s->job_array != NULL)
                dstm_darray_destroy(&s->job_array);

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

        double local_values[3] = {
                s->energy_joules,
                s->max_inlet_temp_c,
                s->max_outlet_temp_c,
        };

        double total_energy = 0.0;
        double max_inlet = 0.0;
        double max_outlet = 0.0;

        MPI_Reduce(&local_values[0],
                   &total_energy,
                   1,
                   MPI_DOUBLE,
                   MPI_SUM,
                   0,
                   MPI_COMM_WORLD);

        MPI_Reduce(&local_values[1],
                   &max_inlet,
                   1,
                   MPI_DOUBLE,
                   MPI_MAX,
                   0,
                   MPI_COMM_WORLD);

        MPI_Reduce(&local_values[2],
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
                printf("energy             : %.3f kWh\n",
                       total_energy / 3600000.0);
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
                printf("wall-clock runtime  : %.6f seconds\n", end - start);

        teardown_local(&sim);

        dstm_finalize();
        MPI_Finalize();
        return 0;
}
