#include <sst/core/component.h>
#include <sst/core/event.h>
#include <sst/core/link.h>
#include <sst/core/params.h>
#include <sst/core/serialization/serialize.h>

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace dcbench {

/*******************************************************************************
 * Constants
 ******************************************************************************/

static constexpr int MAX_THERMAL_DELAY = 3;
static constexpr int MAX_JOB_NODES = 64;

static constexpr int NODE_CPU_CAPACITY = 64;
static constexpr int NODE_GPU_CAPACITY = 8;

static constexpr double NODE_IDLE_POWER_W = 180.0;
static constexpr double CPU_DYNAMIC_POWER_W = 320.0;
static constexpr double GPU_DYNAMIC_POWER_W = 2400.0;

static constexpr double DT_SECONDS = 1.0;

static constexpr double AMBIENT_TEMP_C = 22.0;
static constexpr double INITIAL_COOLANT_TEMP_C = 18.0;
static constexpr double INITIAL_OUTLET_TEMP_C = 24.0;
static constexpr double INITIAL_SETPOINT_C = 18.0;
static constexpr double TARGET_INLET_TEMP_C = 24.0;

static constexpr double RACK_HEAT_CAPACITY_J_K = 120000.0;
static constexpr double COOLANT_HEAT_CAPACITY_J_K = 500000.0;

static constexpr double AIRFLOW_HEAT_CAPACITY_W_K = 8000.0;
static constexpr double RACK_TO_COOLANT_UA_W_K = 6000.0;

static constexpr double COOLING_SETPOINT_RESET_C_C = 0.10;
static constexpr double COOLING_SETPOINT_RATE_C_S = 0.25;
static constexpr double MIN_COOLING_SETPOINT_C = 12.0;
static constexpr double MAX_COOLING_SETPOINT_C = 24.0;

static constexpr double COOLING_PROPORTIONAL_W_K = 12000.0;
static constexpr double MAX_COOLING_REMOVAL_W = 100000.0;
static constexpr double COOLING_ACTUATOR_TIME_S = 20.0;
static constexpr double COOLING_COP = 4.0;

static constexpr double HEAT_RECIRCULATION_FRACTION = 0.04;

/*******************************************************************************
 * Utility functions
 ******************************************************************************/

static double clampDouble(double value, double low, double high)
{
    if (value < low)
        return low;

    if (value > high)
        return high;

    return value;
}

static int thermalDelay(int a, int b)
{
    uint64_t lo =
        static_cast<uint64_t>(std::min(a, b));

    uint64_t hi =
        static_cast<uint64_t>(std::max(a, b));

    uint64_t hash =
        lo * 2654435761ULL +
        hi * 2246822519ULL;

    return 1 +
        static_cast<int>(
            hash %
            static_cast<uint64_t>(MAX_THERMAL_DELAY));
}

/*******************************************************************************
 * Relay-tree topology over MPI ranks.
 *
 * Tree node 0 is the scheduler; tree node d+1 is thermal domain d. This
 * mirrors the DSTM opt_dstm tree: parent(t) = (t-1)/K, children are
 * K*t+1 .. K*t+K. Fanout K >= (node count - 1) collapses to the old star
 * (depth 1), which reproduces the dedicated-scheduler baseline exactly.
 ******************************************************************************/

static int treeParent(int node, int fanout)
{
    return node == 0 ? -1 : (node - 1) / fanout;
}

static int treeFirstChild(int node, int fanout)
{
    return fanout * node + 1;
}

static int treeChildCount(int node, int fanout, int nodeCount)
{
    int first = treeFirstChild(node, fanout);
    int last = nodeCount - 1;

    if (first > last)
        return 0;

    int n = last - first + 1;

    return n > fanout ? fanout : n;
}

static int treeDepthOf(int node, int fanout)
{
    int depth = 0;

    while (node > 0) {
        node = (node - 1) / fanout;
        ++depth;
    }

    return depth;
}

static int treeMaxDepth(int nodeCount, int fanout)
{
    return treeDepthOf(nodeCount - 1, fanout);
}

/* The child of `node` whose subtree contains `target` (target != node). */
static int treeChildToward(int node, int target, int fanout)
{
    while (treeParent(target, fanout) != node)
        target = treeParent(target, fanout);

    return target;
}

/*******************************************************************************
 * Shared model structures
 ******************************************************************************/

struct ScheduledNode {
    int cpuUsed = 0;
    int gpuUsed = 0;
    int runningJobs = 0;

    double cpuUtilization = 0.0;
    double gpuUtilization = 0.0;
};

static bool scheduledNodeEqual(
    const ScheduledNode& a,
    const ScheduledNode& b)
{
    return
        a.cpuUsed == b.cpuUsed &&
        a.gpuUsed == b.gpuUsed &&
        a.runningJobs == b.runningJobs &&
        a.cpuUtilization == b.cpuUtilization &&
        a.gpuUtilization == b.gpuUtilization;
}

struct RackFeedback {
    int globalRack = 0;

    double inletTempC = AMBIENT_TEMP_C;
    double outletTempC = INITIAL_OUTLET_TEMP_C;
    double coolantTempC = INITIAL_COOLANT_TEMP_C;
    double coolingRemovalW = 0.0;
    double totalPowerW = 0.0;
};

struct NodeState {
    int cpuUsed = 0;
    int gpuUsed = 0;
    int runningJobs = 0;

    double cpuUtilization = 0.0;
    double gpuUtilization = 0.0;
    double powerW = NODE_IDLE_POWER_W;
};

struct RackState {
    int globalRack = 0;

    std::vector<NodeState> nodes;

    double totalPowerW = 0.0;

    double inletTempC = AMBIENT_TEMP_C;
    double outletTempC = INITIAL_OUTLET_TEMP_C;

    double coolantTempC = INITIAL_COOLANT_TEMP_C;
    double coolantFlow = 1.0;
    double coolingSetpointC = INITIAL_SETPOINT_C;
    double coolingRemovalW = 0.0;
};

struct Allocation {
    int globalRack = 0;
    int node = 0;
};

enum class JobStatus {
    FUTURE,
    QUEUED,
    RUNNING,
    COMPLETE
};

struct JobState {
    int id = 0;

    int arrivalTime = 0;
    int duration = 0;

    int requestedNodes = 0;
    int cpuPerNode = 0;
    int gpuPerNode = 0;

    double cpuUtilization = 0.0;
    double gpuUtilization = 0.0;

    JobStatus status = JobStatus::FUTURE;

    int startTime = -1;
    int completionTime = -1;

    std::vector<Allocation> allocations;
};

/*******************************************************************************
 * SST events
 ******************************************************************************/

class HeatEvent : public SST::Event {
public:
    int sourceRack = 0;
    int destinationRack = 0;
    int generationTime = 0;
    int arrivalTime = 0;

    double heatW = 0.0;

    static constexpr int EVENT_PRIORITY =
        CLOCKPRIORITY - 1;

    HeatEvent()
    {
        setPriority(EVENT_PRIORITY);
    }

    HeatEvent(
        int source,
        int destination,
        int generation,
        int arrival,
        double heat) :
        sourceRack(source),
        destinationRack(destination),
        generationTime(generation),
        arrivalTime(arrival),
        heatW(heat)
    {
        setPriority(EVENT_PRIORITY);
    }

    void serialize_order(
        SST::Core::Serialization::serializer& serializer) override
    {
        SST::Event::serialize_order(serializer);

        serializer.primitive(sourceRack);
        serializer.primitive(destinationRack);
        serializer.primitive(generationTime);
        serializer.primitive(arrivalTime);
        serializer.primitive(heatW);
    }

    ImplementSerializable(HeatEvent);
};

class FeedbackEvent : public SST::Event {
public:
    int snapshotTime = 0;
    int sourceDomain = 0;
    int rackCount = 0;

    std::vector<int> globalRack;

    std::vector<double> inletTempC;
    std::vector<double> outletTempC;
    std::vector<double> coolantTempC;
    std::vector<double> coolingRemovalW;
    std::vector<double> totalPowerW;

    static constexpr int EVENT_PRIORITY =
        CLOCKPRIORITY - 1;

    FeedbackEvent()
    {
        setPriority(EVENT_PRIORITY);
    }

    FeedbackEvent(
        int snapshot,
        int source,
        int count) :
        snapshotTime(snapshot),
        sourceDomain(source),
        rackCount(count)
    {
        setPriority(EVENT_PRIORITY);
    }

    void serialize_order(
        SST::Core::Serialization::serializer& ser) override
    {
        SST::Event::serialize_order(ser);

        ser.primitive(snapshotTime);
        ser.primitive(sourceDomain);
        ser.primitive(rackCount);

        SST_SER(globalRack);
        SST_SER(inletTempC);
        SST_SER(outletTempC);
        SST_SER(coolantTempC);
        SST_SER(coolingRemovalW);
        SST_SER(totalPowerW);
    }

    ImplementSerializable(FeedbackEvent);
};

class ScheduleBlockEvent : public SST::Event {
public:
    int generation = 0;
    int startTime = 0;
    int tickCount = 0;

    int destinationDomain = 0;
    int rackCount = 0;
    int nodesPerRack = 0;

    /*
     * Parallel primitive arrays represent sparse ScheduleChange records.
     * Every array must have the same length.
     */
    std::vector<int> changeTickOffset;
    std::vector<int> changeLocalNodeIndex;

    std::vector<int> changeCpuUsed;
    std::vector<int> changeGpuUsed;
    std::vector<int> changeRunningJobs;

    std::vector<double> changeCpuUtilization;
    std::vector<double> changeGpuUtilization;

    static constexpr int EVENT_PRIORITY =
        CLOCKPRIORITY - 1;

    ScheduleBlockEvent()
    {
        setPriority(EVENT_PRIORITY);
    }

    ScheduleBlockEvent(
        int scheduleGeneration,
        int scheduleStart,
        int scheduleTicks,
        int destination,
        int racks,
        int nodes) :
        generation(scheduleGeneration),
        startTime(scheduleStart),
        tickCount(scheduleTicks),
        destinationDomain(destination),
        rackCount(racks),
        nodesPerRack(nodes)
    {
        setPriority(EVENT_PRIORITY);
    }

    size_t changeCount() const
    {
        return changeTickOffset.size();
    }

    bool hasValidChangeArrays() const
    {
        size_t count = changeTickOffset.size();

        return
            changeLocalNodeIndex.size() == count &&
            changeCpuUsed.size() == count &&
            changeGpuUsed.size() == count &&
            changeRunningJobs.size() == count &&
            changeCpuUtilization.size() == count &&
            changeGpuUtilization.size() == count;
    }

    void reserveChanges(size_t count)
    {
        changeTickOffset.reserve(count);
        changeLocalNodeIndex.reserve(count);
        changeCpuUsed.reserve(count);
        changeGpuUsed.reserve(count);
        changeRunningJobs.reserve(count);
        changeCpuUtilization.reserve(count);
        changeGpuUtilization.reserve(count);
    }

    void appendChange(
        int tickOffset,
        int localNodeIndex,
        const ScheduledNode& state)
    {
        changeTickOffset.push_back(tickOffset);
        changeLocalNodeIndex.push_back(localNodeIndex);

        changeCpuUsed.push_back(state.cpuUsed);
        changeGpuUsed.push_back(state.gpuUsed);
        changeRunningJobs.push_back(state.runningJobs);

        changeCpuUtilization.push_back(
            state.cpuUtilization);

        changeGpuUtilization.push_back(
            state.gpuUtilization);
    }

    void serialize_order(
        SST::Core::Serialization::serializer& ser) override
    {
        SST::Event::serialize_order(ser);

        ser.primitive(generation);
        ser.primitive(startTime);
        ser.primitive(tickCount);
        ser.primitive(destinationDomain);
        ser.primitive(rackCount);
        ser.primitive(nodesPerRack);

        SST_SER(changeTickOffset);
        SST_SER(changeLocalNodeIndex);
        SST_SER(changeCpuUsed);
        SST_SER(changeGpuUsed);
        SST_SER(changeRunningJobs);
        SST_SER(changeCpuUtilization);
        SST_SER(changeGpuUtilization);
    }

    ImplementSerializable(ScheduleBlockEvent);
};

/*******************************************************************************
 * Dedicated scheduler component
 ******************************************************************************/

class Scheduler : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(
        Scheduler,
        "opt_dcbench",
        "scheduler",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "Dedicated datacenter scheduling component",
        COMPONENT_CATEGORY_PROCESSOR
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"domains", "Number of thermal domains", "1"},
        {"racks", "Total rack count", "16"},
        {"nodes_per_rack", "Nodes per rack", "8"},
        {"steps", "Simulation seconds", "300"},
        {"schedule_window", "Scheduling epoch length", "60"},
        {"control_latency", "Feedback latency", "30"},
        {"tree_fanout", "Relay-tree fanout K", "1073741824"},
        {"jobs", "CSV job trace", "jobs.csv"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"dn_fb%(domains)d",
         "Aggregated feedback from a child subtree",
         {"dcbench.FeedbackEvent"}},

        {"dn_sched%(domains)d",
         "Schedule blocks sent down to a child subtree",
         {"dcbench.ScheduleBlockEvent"}}
    )

    Scheduler(
        SST::ComponentId_t id,
        SST::Params& params) :
        SST::Component(id)
    {
        domains_ =
            params.find<int>("domains", 1);

        totalRacks_ =
            params.find<int>("racks", 16);

        nodesPerRack_ =
            params.find<int>("nodes_per_rack", 8);

        steps_ =
            params.find<int>("steps", 300);

        scheduleWindow_ =
            params.find<int>("schedule_window", 60);

        controlLatency_ =
            params.find<int>("control_latency", 30);

        jobPath_ =
            params.find<std::string>("jobs", "jobs.csv");

        treeFanout_ =
            params.find<int>("tree_fanout", 1 << 30);

        if (treeFanout_ < 2)
            treeFanout_ = 2;

        if (domains_ < 1) {
            fatal("domains must be positive");
        }

        if (totalRacks_ < 3 ||
            totalRacks_ % domains_ != 0) {
            fatal("racks must be >= 3 and divisible by domains");
        }

        if (nodesPerRack_ <= 0) {
            fatal("nodes_per_rack must be positive");
        }

        if (steps_ < 1) {
            fatal("steps must be positive");
        }

        if (scheduleWindow_ <= 0) {
            fatal("schedule_window must be positive");
        }

        if (controlLatency_ <= 0) {
            fatal("control_latency must be positive");
        }

        if (2 * controlLatency_ > scheduleWindow_) {
            fatal("control round trip exceeds schedule window");
        }

        localRacks_ =
            totalRacks_ / domains_;

        nodeCount_ = domains_ + 1;

        treeDepth_ =
            treeMaxDepth(nodeCount_, treeFanout_);

        if (treeDepth_ < 1)
            treeDepth_ = 1;

        if (controlLatency_ % treeDepth_ != 0 ||
            (scheduleWindow_ - controlLatency_) %
                    treeDepth_ != 0) {
            fatal(
                "tree depth must divide control latency "
                "and schedule delay");
        }

        schedulerNodes_.resize(
            static_cast<size_t>(totalRacks_) *
            static_cast<size_t>(nodesPerRack_));

        schedulerFeedback_.resize(
            static_cast<size_t>(totalRacks_));

        childCount_ =
            treeChildCount(0, treeFanout_, nodeCount_);

        childFeedbackLinks_.resize(
            static_cast<size_t>(childCount_),
            nullptr);

        childScheduleLinks_.resize(
            static_cast<size_t>(childCount_),
            nullptr);

        for (int c = 0; c < childCount_; ++c) {
            childFeedbackLinks_[c] =
                configureLink(
                    "dn_fb" + std::to_string(c),
                    new SST::Event::Handler2<
                        Scheduler,
                        &Scheduler::handleFeedback>(this));

            childScheduleLinks_[c] =
                configureLink(
                    "dn_sched" + std::to_string(c));

            if (childFeedbackLinks_[c] == nullptr ||
                childScheduleLinks_[c] == nullptr) {
                fatal(
                    "scheduler child links are missing for child " +
                    std::to_string(c));
            }
        }

        loadJobs();

        registerAsPrimaryComponent();
        primaryComponentDoNotEndSim();

        registerClock(
            "1s",
            new SST::Clock::Handler2<
                Scheduler,
                &Scheduler::clockTick>(this));
    }

    void finish() override
    {
        double runtime =
            timerStopped_
                ? timerEnd_ - timerStart_
                : 0.0;

        double occupancy = 0.0;

        if (scheduleEventCapacityTotal_ > 0) {
            occupancy =
                100.0 *
                static_cast<double>(
                    scheduleEventsEmitted_) /
                static_cast<double>(
                    scheduleEventCapacityTotal_);
        }

        std::printf(
            "role=scheduler "
            "simulation_runtime_seconds=%.9f "
            "jobs_started=%llu "
            "jobs_completed=%llu "
            "scheduler_planning_seconds=%.9f "
            "schedule_publishing_seconds=%.9f "
            "schedule_events=%llu "
            "event_capacity=%llu "
            "event_occupancy_percent=%.6f\n",
            runtime,
            static_cast<unsigned long long>(
                jobsStarted_),
            static_cast<unsigned long long>(
                jobsCompleted_),
            schedulerPlanningSeconds_,
            schedulerPublicationSeconds_,
            static_cast<unsigned long long>(
                scheduleEventsEmitted_),
            static_cast<unsigned long long>(
                scheduleEventCapacityTotal_),
            occupancy);
    }

private:
    using CompletionEntry =
        std::tuple<int, int, size_t>;

    int domains_ = 0;
    int totalRacks_ = 0;
    int localRacks_ = 0;
    int nodesPerRack_ = 0;
    int steps_ = 0;

    int scheduleWindow_ = 0;
    int controlLatency_ = 0;

    int currentTime_ = 0;

    std::string jobPath_;

    std::vector<NodeState> schedulerNodes_;
    std::vector<RackFeedback> schedulerFeedback_;

    std::vector<JobState> jobs_;

    size_t nextArrivalJob_ = 0;
    size_t nextQueuedJob_ = 0;

    std::priority_queue<
        CompletionEntry,
        std::vector<CompletionEntry>,
        std::greater<CompletionEntry>>
        completionHeap_;

    std::vector<SST::Link*> childFeedbackLinks_;
    std::vector<SST::Link*> childScheduleLinks_;

    int treeFanout_ = 1;
    int nodeCount_ = 0;
    int treeDepth_ = 1;
    int childCount_ = 0;

    std::map<int, int> feedbackRackCount_;

    uint64_t jobsStarted_ = 0;
    uint64_t jobsCompleted_ = 0;

    uint64_t scheduleEventsEmitted_ = 0;
    uint64_t scheduleEventCapacityTotal_ = 0;

    double schedulerPlanningSeconds_ = 0.0;
    double schedulerPublicationSeconds_ = 0.0;

    bool timerStarted_ = false;
    bool timerStopped_ = false;

    double timerStart_ = 0.0;
    double timerEnd_ = 0.0;

    [[noreturn]] void fatal(
        const std::string& message) const
    {
        getSimulationOutput().fatal(
            CALL_INFO,
            1,
            "%s\n",
            message.c_str());

        std::abort();
    }

    void loadJobs()
    {
        std::ifstream input(jobPath_);

        if (!input) {
            fatal(
                "cannot open job trace " +
                jobPath_);
        }

        std::string line;
        int id = 0;

        while (std::getline(input, line)) {
            if (line.empty() ||
                line[0] == '#') {
                continue;
            }

            std::replace(
                line.begin(),
                line.end(),
                ',',
                ' ');

            std::istringstream stream(line);

            JobState job;
            job.id = id++;

            if (!(stream >>
                  job.arrivalTime >>
                  job.duration >>
                  job.requestedNodes >>
                  job.cpuPerNode >>
                  job.gpuPerNode >>
                  job.cpuUtilization >>
                  job.gpuUtilization)) {
                fatal(
                    "malformed job record: " +
                    line);
            }

            if (job.arrivalTime < 0) {
                fatal("negative arrival time");
            }

            if (job.duration <= 0) {
                fatal("job duration must be positive");
            }

            if (job.requestedNodes <= 0 ||
                job.requestedNodes > MAX_JOB_NODES) {
                fatal("invalid requested node count");
            }

            if (job.cpuPerNode < 0 ||
                job.cpuPerNode >
                    NODE_CPU_CAPACITY) {
                fatal("invalid CPU request");
            }

            if (job.gpuPerNode < 0 ||
                job.gpuPerNode >
                    NODE_GPU_CAPACITY) {
                fatal("invalid GPU request");
            }

            job.cpuUtilization =
                clampDouble(
                    job.cpuUtilization,
                    0.0,
                    1.0);

            job.gpuUtilization =
                clampDouble(
                    job.gpuUtilization,
                    0.0,
                    1.0);

            jobs_.push_back(std::move(job));
        }

        std::sort(
            jobs_.begin(),
            jobs_.end(),
            [](const JobState& a,
               const JobState& b) {
                if (a.arrivalTime !=
                    b.arrivalTime) {
                    return
                        a.arrivalTime <
                        b.arrivalTime;
                }

                return a.id < b.id;
            });
    }

    NodeState& schedulerNode(
        int globalRack,
        int node)
    {
        size_t index =
            static_cast<size_t>(globalRack) *
                static_cast<size_t>(
                    nodesPerRack_) +
            static_cast<size_t>(node);

        return schedulerNodes_[index];
    }

    static ScheduledNode makeScheduledNode(
        const NodeState& node)
    {
        ScheduledNode result;

        result.cpuUsed =
            node.cpuUsed;

        result.gpuUsed =
            node.gpuUsed;

        result.runningJobs =
            node.runningJobs;

        result.cpuUtilization =
            node.cpuUtilization;

        result.gpuUtilization =
            node.gpuUtilization;

        return result;
    }

    static double cpuContribution(
        const JobState& job)
    {
        return
            (static_cast<double>(
                 job.cpuPerNode) /
             static_cast<double>(
                 NODE_CPU_CAPACITY)) *
            job.cpuUtilization;
    }

    static double gpuContribution(
        const JobState& job)
    {
        return
            (static_cast<double>(
                 job.gpuPerNode) /
             static_cast<double>(
                 NODE_GPU_CAPACITY)) *
            job.gpuUtilization;
    }

    static void addJobToNode(
        NodeState& node,
        const JobState& job)
    {
        node.cpuUsed +=
            job.cpuPerNode;

        node.gpuUsed +=
            job.gpuPerNode;

        node.runningJobs++;

        node.cpuUtilization =
            clampDouble(
                node.cpuUtilization +
                    cpuContribution(job),
                0.0,
                1.0);

        node.gpuUtilization =
            clampDouble(
                node.gpuUtilization +
                    gpuContribution(job),
                0.0,
                1.0);
    }

    static void removeJobFromNode(
        NodeState& node,
        const JobState& job)
    {
        node.cpuUtilization =
            clampDouble(
                node.cpuUtilization -
                    cpuContribution(job),
                0.0,
                1.0);

        node.gpuUtilization =
            clampDouble(
                node.gpuUtilization -
                    gpuContribution(job),
                0.0,
                1.0);

        node.cpuUsed -=
            job.cpuPerNode;

        node.gpuUsed -=
            job.gpuPerNode;

        node.runningJobs--;

        if (node.cpuUsed < 0 ||
            node.gpuUsed < 0 ||
            node.runningJobs < 0) {
            std::abort();
        }
    }

    static bool nodeCanHost(
        const NodeState& node,
        const JobState& job)
    {
        int cpuFree =
            NODE_CPU_CAPACITY -
            node.cpuUsed;

        int gpuFree =
            NODE_GPU_CAPACITY -
            node.gpuUsed;

        return
            cpuFree >= job.cpuPerNode &&
            gpuFree >= job.gpuPerNode;
    }

    void releaseCompletedJobs(
        int effectiveTime)
    {
        while (!completionHeap_.empty()) {
            const CompletionEntry& entry =
                completionHeap_.top();

            int completionTime =
                std::get<0>(entry);

            if (completionTime >
                effectiveTime) {
                break;
            }

            size_t jobIndex =
                std::get<2>(entry);

            completionHeap_.pop();

            if (jobIndex >= jobs_.size()) {
                fatal(
                    "invalid completion heap job index");
            }

            JobState& job =
                jobs_[jobIndex];

            if (job.status !=
                JobStatus::RUNNING) {
                fatal(
                    "completion heap job is not running");
            }

            for (const Allocation& allocation :
                 job.allocations) {
                removeJobFromNode(
                    schedulerNode(
                        allocation.globalRack,
                        allocation.node),
                    job);
            }

            job.allocations.clear();
            job.status =
                JobStatus::COMPLETE;

            jobsCompleted_++;
        }
    }

    void admitArrivals(
        int effectiveTime)
    {
        while (nextArrivalJob_ <
               jobs_.size()) {
            JobState& job =
                jobs_[nextArrivalJob_];

            if (job.arrivalTime >
                effectiveTime) {
                break;
            }

            if (job.status !=
                JobStatus::FUTURE) {
                fatal(
                    "arrival cursor found "
                    "non-future job");
            }

            job.status =
                JobStatus::QUEUED;

            nextArrivalJob_++;
        }
    }

    bool tryStartJob(
        JobState& job,
        int effectiveTime)
    {
        std::vector<Allocation> candidates;

        candidates.reserve(
            static_cast<size_t>(
                job.requestedNodes));

        for (int rack = 0;
             rack < totalRacks_ &&
             static_cast<int>(
                 candidates.size()) <
                 job.requestedNodes;
             ++rack) {
            for (int node = 0;
                 node < nodesPerRack_ &&
                 static_cast<int>(
                     candidates.size()) <
                     job.requestedNodes;
                 ++node) {
                if (!nodeCanHost(
                        schedulerNode(rack, node),
                        job)) {
                    continue;
                }

                candidates.push_back(
                    Allocation{rack, node});
            }
        }

        if (static_cast<int>(
                candidates.size()) !=
            job.requestedNodes) {
            return false;
        }

        for (const Allocation& allocation :
             candidates) {
            addJobToNode(
                schedulerNode(
                    allocation.globalRack,
                    allocation.node),
                job);
        }

        job.allocations =
            std::move(candidates);

        job.startTime =
            effectiveTime;

        job.completionTime =
            effectiveTime +
            job.duration;

        job.status =
            JobStatus::RUNNING;

        size_t jobIndex =
            static_cast<size_t>(
                &job - jobs_.data());

        completionHeap_.emplace(
            job.completionTime,
            job.id,
            jobIndex);

        jobsStarted_++;

        return true;
    }

    void scheduleQueuedJobs(
        int effectiveTime)
    {
        while (nextQueuedJob_ <
               jobs_.size()) {
            JobState& job =
                jobs_[nextQueuedJob_];

            if (job.status ==
                JobStatus::FUTURE) {
                break;
            }

            if (job.status !=
                JobStatus::QUEUED) {
                fatal(
                    "queue cursor found "
                    "invalid job state");
            }

            if (!tryStartJob(
                    job,
                    effectiveTime)) {
                break;
            }

            nextQueuedJob_++;
        }
    }

    void appendDomainChanges(
        ScheduleBlockEvent& block,
        int destinationDomain,
        int tickOffset,
        std::vector<ScheduledNode>& previous)
    {
        int firstRack =
            destinationDomain *
            localRacks_;

        int localNodeIndex = 0;

        for (int localRack = 0;
             localRack < localRacks_;
             ++localRack) {
            int globalRack =
                firstRack +
                localRack;

            for (int node = 0;
                 node < nodesPerRack_;
                 ++node, ++localNodeIndex) {
                size_t globalNodeIndex =
                    static_cast<size_t>(
                        globalRack) *
                        static_cast<size_t>(
                            nodesPerRack_) +
                    static_cast<size_t>(node);

                ScheduledNode current =
                    makeScheduledNode(
                        schedulerNode(
                            globalRack,
                            node));

                if (scheduledNodeEqual(
                        current,
                        previous[
                            globalNodeIndex])) {
                    continue;
                }

                block.appendChange(
                    tickOffset,
                    localNodeIndex,
                    current);

                previous[globalNodeIndex] =
                    current;
            }
        }
    }

    void planAndPublishEpoch(
        int schedulerTime,
        int epochStart)
    {
        int tickCount =
            scheduleWindow_;

        if (epochStart + tickCount >
            steps_) {
            tickCount =
                steps_ -
                epochStart;
        }

        if (tickCount <= 0)
            return;

        std::vector<ScheduleBlockEvent*> blocks(
            static_cast<size_t>(domains_),
            nullptr);

        size_t workerNodeCount =
            static_cast<size_t>(
                localRacks_) *
            static_cast<size_t>(
                nodesPerRack_);

        size_t eventCapacity =
            static_cast<size_t>(
                scheduleWindow_) *
            workerNodeCount;

        for (int destination = 0;
             destination < domains_;
             ++destination) {
            blocks[destination] =
                new ScheduleBlockEvent(
                    epochStart /
                        scheduleWindow_,
                    epochStart,
                    tickCount,
                    destination,
                    localRacks_,
                    nodesPerRack_);

            blocks[destination]
                ->reserveChanges(
                    std::min(
                        eventCapacity,
                        workerNodeCount));
        }

        std::vector<ScheduledNode> previous(
            schedulerNodes_.size());

        for (size_t i = 0;
             i < schedulerNodes_.size();
             ++i) {
            previous[i] =
                makeScheduledNode(
                    schedulerNodes_[i]);
        }

        double planningStart =
            MPI_Wtime();

        for (int offset = 0;
             offset < tickCount;
             ++offset) {
            int effectiveTime =
                epochStart +
                offset;

            releaseCompletedJobs(
                effectiveTime);

            admitArrivals(
                effectiveTime);

            scheduleQueuedJobs(
                effectiveTime);

            for (int destination = 0;
                 destination < domains_;
                 ++destination) {
                appendDomainChanges(
                    *blocks[destination],
                    destination,
                    offset,
                    previous);
            }
        }

        schedulerPlanningSeconds_ +=
            MPI_Wtime() -
            planningStart;

        int scheduleDelay =
            epochStart -
            schedulerTime;

        if (scheduleDelay <= 0) {
            fatal(
                "schedule does not target "
                "a future timestamp");
        }

        /*
         * Link latency is configured by run.py as:
         *
         *     schedule_window - control_latency
         *
         * Therefore no additional send delay is required.
         */
        double publicationStart =
            MPI_Wtime();

        for (int destination = 0;
             destination < domains_;
             ++destination) {
            ScheduleBlockEvent* block =
                blocks[destination];

            scheduleEventsEmitted_ +=
                static_cast<uint64_t>(
                    block->changeCount());

            scheduleEventCapacityTotal_ +=
                static_cast<uint64_t>(
                    eventCapacity);

            int child =
                treeChildToward(
                    0,
                    destination + 1,
                    treeFanout_) - 1;

            childScheduleLinks_[child]
                ->send(0, block);
        }

        schedulerPublicationSeconds_ +=
            MPI_Wtime() -
            publicationStart;
    }

    void handleFeedback(
        SST::Event* event)
    {
        auto* feedback =
            dynamic_cast<FeedbackEvent*>(event);

        if (feedback == nullptr) {
            delete event;
            fatal(
                "scheduler received "
                "an invalid feedback event");
        }

        size_t count =
            feedback->globalRack.size();

        if (static_cast<size_t>(feedback->rackCount) != count ||
            feedback->inletTempC.size() != count ||
            feedback->outletTempC.size() != count ||
            feedback->coolantTempC.size() != count ||
            feedback->coolingRemovalW.size() != count ||
            feedback->totalPowerW.size() != count) {
            delete feedback;
            fatal(
                "invalid feedback array sizes");
        }

        for (size_t i = 0; i < count; ++i) {
            int globalRack =
                feedback->globalRack[i];

            if (globalRack < 0 ||
                globalRack >= totalRacks_) {
                delete feedback;
                fatal(
                    "feedback rack out of range");
            }

            RackFeedback rack;

            rack.globalRack = globalRack;
            rack.inletTempC = feedback->inletTempC[i];
            rack.outletTempC = feedback->outletTempC[i];
            rack.coolantTempC = feedback->coolantTempC[i];
            rack.coolingRemovalW = feedback->coolingRemovalW[i];
            rack.totalPowerW = feedback->totalPowerW[i];

            schedulerFeedback_[
                static_cast<size_t>(globalRack)] =
                rack;
        }

        feedbackRackCount_[feedback->snapshotTime] +=
            static_cast<int>(count);

        delete feedback;
    }

    void collectFeedback(
        int snapshotTime)
    {
        auto it =
            feedbackRackCount_.find(snapshotTime);

        if (it == feedbackRackCount_.end() ||
            it->second != totalRacks_) {
            fatal(
                "feedback incomplete at snapshot " +
                std::to_string(snapshotTime));
        }

        feedbackRackCount_.erase(it);
    }

    bool clockTick(
        SST::Cycle_t)
    {
        if (!timerStarted_) {
            timerStart_ =
                MPI_Wtime();

            timerStarted_ = true;
        }

        if (currentTime_ < steps_ &&
            currentTime_ >=
                controlLatency_) {
            int snapshotTime =
                currentTime_ -
                controlLatency_;

            if (snapshotTime %
                    scheduleWindow_ ==
                0) {
                int epochStart =
                    snapshotTime +
                    scheduleWindow_;

                collectFeedback(
                    snapshotTime);

                if (epochStart <
                    steps_) {
                    planAndPublishEpoch(
                        currentTime_,
                        epochStart);
                }
            }
        }

        currentTime_++;

        int finalTime =
            steps_ +
            scheduleWindow_ +
            MAX_THERMAL_DELAY +
            1;

        if (currentTime_ >=
            finalTime) {
            timerEnd_ =
                MPI_Wtime();

            timerStopped_ = true;

            primaryComponentOKToEndSim();
            return true;
        }

        return false;
    }
};

/*******************************************************************************
 * Thermal worker component
 ******************************************************************************/

class ThermalDomain : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(
        ThermalDomain,
        "opt_dcbench",
        "thermal",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "Datacenter thermal worker component",
        COMPONENT_CATEGORY_PROCESSOR
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"domain", "Thermal-domain identifier", "0"},
        {"domains", "Number of thermal domains", "1"},
        {"racks", "Total rack count", "16"},
        {"nodes_per_rack", "Nodes per rack", "8"},
        {"steps", "Simulation seconds", "300"},
        {"schedule_window", "Scheduling epoch length", "60"},
        {"control_latency", "Feedback latency", "30"},
        {"tree_fanout", "Relay-tree fanout K", "1073741824"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"left",
         "Left thermal-domain neighbor",
         {"dcbench.HeatEvent"}},

        {"right",
         "Right thermal-domain neighbor",
         {"dcbench.HeatEvent"}},

        {"up_fb",
         "Aggregated feedback to parent",
         {"dcbench.FeedbackEvent"}},

        {"up_sched",
         "Schedule blocks from parent",
         {"dcbench.ScheduleBlockEvent"}},

        {"dn_fb%(domains)d",
         "Aggregated feedback from a child subtree",
         {"dcbench.FeedbackEvent"}},

        {"dn_sched%(domains)d",
         "Schedule blocks sent down to a child subtree",
         {"dcbench.ScheduleBlockEvent"}}
    )

    ThermalDomain(
        SST::ComponentId_t id,
        SST::Params& params) :
        SST::Component(id)
    {
        domain_ =
            params.find<int>("domain", 0);

        domains_ =
            params.find<int>("domains", 1);

        totalRacks_ =
            params.find<int>("racks", 16);

        nodesPerRack_ =
            params.find<int>("nodes_per_rack", 8);

        steps_ =
            params.find<int>("steps", 300);

        scheduleWindow_ =
            params.find<int>("schedule_window", 60);

        controlLatency_ =
            params.find<int>("control_latency", 30);

        if (domains_ < 1 ||
            domain_ < 0 ||
            domain_ >= domains_) {
            fatal("invalid thermal domain");
        }

        if (totalRacks_ < 3 ||
            totalRacks_ % domains_ != 0) {
            fatal(
                "racks must be >= 3 and "
                "divisible by domains");
        }

        if (nodesPerRack_ <= 0) {
            fatal(
                "nodes_per_rack must be positive");
        }

        if (scheduleWindow_ <= 0 ||
            controlLatency_ <= 0 ||
            2 * controlLatency_ >
                scheduleWindow_) {
            fatal(
                "invalid control timing");
        }

        localRacks_ =
            totalRacks_ /
            domains_;

        rackOffset_ =
            domain_ *
            localRacks_;

        treeFanout_ =
            params.find<int>("tree_fanout", 1 << 30);

        if (treeFanout_ < 2)
            treeFanout_ = 2;

        treeNode_ = domain_ + 1;
        nodeCount_ = domains_ + 1;
        parentNode_ = treeParent(treeNode_, treeFanout_);
        isParentScheduler_ = (parentNode_ == 0);

        childCount_ =
            treeChildCount(
                treeNode_, treeFanout_, nodeCount_);

        treeDepth_ =
            treeMaxDepth(nodeCount_, treeFanout_);

        if (treeDepth_ < 1)
            treeDepth_ = 1;

        if (controlLatency_ % treeDepth_ != 0 ||
            (scheduleWindow_ - controlLatency_) %
                    treeDepth_ != 0) {
            fatal(
                "tree depth must divide control latency "
                "and schedule delay");
        }

        parentFeedbackLink_ =
            configureLink("up_fb");

        parentScheduleLink_ =
            configureLink(
                "up_sched",
                new SST::Event::Handler2<
                    ThermalDomain,
                    &ThermalDomain::handleSchedule>(
                        this));

        if (parentFeedbackLink_ == nullptr ||
            parentScheduleLink_ == nullptr) {
            fatal(
                "parent control links are required");
        }

        childFeedbackLinks_.resize(
            static_cast<size_t>(childCount_), nullptr);

        childScheduleLinks_.resize(
            static_cast<size_t>(childCount_), nullptr);

        for (int c = 0; c < childCount_; ++c) {
            childFeedbackLinks_[c] =
                configureLink(
                    "dn_fb" + std::to_string(c),
                    new SST::Event::Handler2<
                        ThermalDomain,
                        &ThermalDomain::handleChildFeedback>(
                            this));

            childScheduleLinks_[c] =
                configureLink(
                    "dn_sched" + std::to_string(c));

            if (childFeedbackLinks_[c] == nullptr ||
                childScheduleLinks_[c] == nullptr) {
                fatal(
                    "child control links are required");
            }
        }

        if (domains_ > 1) {
            leftLink_ =
                configureLink(
                    "left",
                    new SST::Event::Handler2<
                        ThermalDomain,
                        &ThermalDomain::handleHeat>(
                            this));

            rightLink_ =
                configureLink(
                    "right",
                    new SST::Event::Handler2<
                        ThermalDomain,
                        &ThermalDomain::handleHeat>(
                            this));

            if (leftLink_ == nullptr ||
                rightLink_ == nullptr) {
                fatal(
                    "thermal neighbor links "
                    "are required");
            }
        }

        racks_.resize(
            static_cast<size_t>(
                localRacks_));

        for (int localRack = 0;
             localRack < localRacks_;
             ++localRack) {
            RackState& rack =
                racks_[
                    static_cast<size_t>(
                        localRack)];

            rack.globalRack =
                rackOffset_ +
                localRack;

            rack.nodes.resize(
                static_cast<size_t>(
                    nodesPerRack_));

            rack.totalPowerW =
                NODE_IDLE_POWER_W *
                static_cast<double>(
                    nodesPerRack_);
        }

        size_t pendingTimes =
            static_cast<size_t>(
                steps_ +
                scheduleWindow_ +
                MAX_THERMAL_DELAY +
                2);

        pendingHeat_.resize(
            pendingTimes,
            std::vector<double>(
                static_cast<size_t>(
                    localRacks_),
                0.0));

        registerAsPrimaryComponent();
        primaryComponentDoNotEndSim();

        registerClock(
            "1s",
            new SST::Clock::Handler2<
                ThermalDomain,
                &ThermalDomain::clockTick>(
                    this));
    }

    void finish() override
    {
        double runtime =
            timerStopped_
                ? timerEnd_ - timerStart_
                : 0.0;

        std::printf(
            "role=thermal "
            "domain=%d "
            "simulation_runtime_seconds=%.9f "
            "it_energy_kwh=%.9f "
            "cooling_energy_kwh=%.9f "
            "facility_energy_kwh=%.9f "
            "max_inlet=%.6f "
            "max_outlet=%.6f\n",
            domain_,
            runtime,
            itEnergyJ_ / 3600000.0,
            coolingEnergyJ_ / 3600000.0,
            (itEnergyJ_ +
             coolingEnergyJ_) /
                3600000.0,
            maxInletTempC_,
            maxOutletTempC_);
    }

private:
    int domain_ = 0;
    int domains_ = 0;

    int totalRacks_ = 0;
    int localRacks_ = 0;
    int rackOffset_ = 0;

    int nodesPerRack_ = 0;
    int steps_ = 0;

    int scheduleWindow_ = 0;
    int controlLatency_ = 0;

    int currentTime_ = 0;

    SST::Link* leftLink_ = nullptr;
    SST::Link* rightLink_ = nullptr;

    SST::Link* parentFeedbackLink_ = nullptr;
    SST::Link* parentScheduleLink_ = nullptr;

    std::vector<SST::Link*> childFeedbackLinks_;
    std::vector<SST::Link*> childScheduleLinks_;

    int treeFanout_ = 1;
    int treeNode_ = 0;
    int parentNode_ = -1;
    int nodeCount_ = 0;
    int childCount_ = 0;
    int treeDepth_ = 1;
    bool isParentScheduler_ = true;

    struct FeedbackAgg {
        std::vector<RackFeedback> racks;
        int childrenReceived = 0;
        bool ownStashed = false;
    };

    std::map<int, FeedbackAgg> feedbackAgg_;

    std::map<int, ScheduleBlockEvent*> pendingSchedule_;

    std::vector<RackState> racks_;

    std::vector<std::vector<double>>
        pendingHeat_;

    ScheduleBlockEvent* scheduleBlock_ =
        nullptr;

    size_t scheduleCursor_ = 0;

    double itEnergyJ_ = 0.0;
    double coolingEnergyJ_ = 0.0;

    double maxInletTempC_ =
        -std::numeric_limits<double>
            ::infinity();

    double maxOutletTempC_ =
        -std::numeric_limits<double>
            ::infinity();

    bool timerStarted_ = false;
    bool timerStopped_ = false;

    double timerStart_ = 0.0;
    double timerEnd_ = 0.0;

    [[noreturn]] void fatal(
        const std::string& message) const
    {
        getSimulationOutput().fatal(
            CALL_INFO,
            1,
            "%s\n",
            message.c_str());

        std::abort();
    }

    int ringLeft(int rack) const
    {
        return
            (rack +
             totalRacks_ -
             1) %
            totalRacks_;
    }

    int ringRight(int rack) const
    {
        return
            (rack + 1) %
            totalRacks_;
    }

    void handleHeat(
        SST::Event* event)
    {
        auto* heat =
            dynamic_cast<HeatEvent*>(event);

        if (heat == nullptr) {
            delete event;
            fatal(
                "received invalid heat event");
        }

        int localRack =
            heat->destinationRack -
            rackOffset_;

        if (localRack < 0 ||
            localRack >= localRacks_) {
            delete heat;
            fatal(
                "heat event delivered to "
                "the wrong thermal domain");
        }

        if (heat->arrivalTime < 0 ||
            static_cast<size_t>(
                heat->arrivalTime) >=
                pendingHeat_.size()) {
            delete heat;
            fatal(
                "heat arrival time out of range");
        }

        int expectedDelay =
            thermalDelay(
                heat->sourceRack,
                heat->destinationRack);

        if (heat->generationTime +
                expectedDelay !=
            heat->arrivalTime) {
            delete heat;
            fatal(
                "heat timestamp mismatch");
        }

        pendingHeat_[
            static_cast<size_t>(
                heat->arrivalTime)]
            [static_cast<size_t>(
                localRack)] +=
            heat->heatW;

        delete heat;
    }

    void handleSchedule(
        SST::Event* event)
    {
        auto* block =
            dynamic_cast<ScheduleBlockEvent*>(
                event);

        if (block == nullptr) {
            delete event;
            fatal(
                "received invalid schedule event");
        }

        if (!block->hasValidChangeArrays()) {
            delete block;
            fatal(
                "inconsistent schedule change arrays");
        }

        /*
         * Not ours: relay down toward the child subtree that owns the
         * destination domain. One hop of hop_down latency per level.
         */
        if (block->destinationDomain != domain_) {
            int target =
                block->destinationDomain + 1;

            int child =
                treeChildToward(
                    treeNode_, target, treeFanout_) -
                treeFirstChild(
                    treeNode_, treeFanout_);

            if (child < 0 ||
                child >= childCount_) {
                delete block;
                fatal(
                    "schedule has no route "
                    "to destination");
            }

            childScheduleLinks_[child]->send(0, block);
            return;
        }

        if (block->rackCount != localRacks_ ||
            block->nodesPerRack != nodesPerRack_) {
            delete block;
            fatal(
                "schedule topology mismatch");
        }

        /*
         * Ours. Shallow nodes receive it before the epoch starts; hold it
         * (keyed by start time) until applyScheduleTick consumes it.
         */
        if (block->startTime < currentTime_) {
            delete block;
            fatal(
                "schedule arrived after its "
                "epoch start");
        }

        if (block->tickCount <= 0 ||
            block->tickCount > scheduleWindow_) {
            delete block;
            fatal(
                "invalid schedule tick count");
        }

        if (pendingSchedule_.count(block->startTime)) {
            delete block;
            fatal(
                "duplicate schedule block for epoch");
        }

        pendingSchedule_[block->startTime] = block;
    }

    void stashOwnFeedback(
        int snapshotTime)
    {
        FeedbackAgg& agg =
            feedbackAgg_[snapshotTime];

        for (const RackState& rack : racks_) {
            RackFeedback fb;

            fb.globalRack = rack.globalRack;
            fb.inletTempC = rack.inletTempC;
            fb.outletTempC = rack.outletTempC;
            fb.coolantTempC = rack.coolantTempC;
            fb.coolingRemovalW = rack.coolingRemovalW;
            fb.totalPowerW = rack.totalPowerW;

            agg.racks.push_back(fb);
        }

        agg.ownStashed = true;

        tryFlushFeedback(snapshotTime);
    }

    void handleChildFeedback(
        SST::Event* event)
    {
        auto* feedback =
            dynamic_cast<FeedbackEvent*>(event);

        if (feedback == nullptr) {
            delete event;
            fatal(
                "received invalid child "
                "feedback event");
        }

        size_t count =
            feedback->globalRack.size();

        FeedbackAgg& agg =
            feedbackAgg_[feedback->snapshotTime];

        for (size_t i = 0; i < count; ++i) {
            RackFeedback fb;

            fb.globalRack = feedback->globalRack[i];
            fb.inletTempC = feedback->inletTempC[i];
            fb.outletTempC = feedback->outletTempC[i];
            fb.coolantTempC = feedback->coolantTempC[i];
            fb.coolingRemovalW = feedback->coolingRemovalW[i];
            fb.totalPowerW = feedback->totalPowerW[i];

            agg.racks.push_back(fb);
        }

        agg.childrenReceived += 1;

        int snapshot = feedback->snapshotTime;

        delete feedback;

        tryFlushFeedback(snapshot);
    }

    void tryFlushFeedback(
        int snapshotTime)
    {
        auto it =
            feedbackAgg_.find(snapshotTime);

        if (it == feedbackAgg_.end())
            return;

        FeedbackAgg& agg = it->second;

        if (!agg.ownStashed ||
            agg.childrenReceived < childCount_) {
            return;
        }

        size_t count = agg.racks.size();

        auto* event =
            new FeedbackEvent(
                snapshotTime,
                domain_,
                static_cast<int>(count));

        event->globalRack.reserve(count);
        event->inletTempC.reserve(count);
        event->outletTempC.reserve(count);
        event->coolantTempC.reserve(count);
        event->coolingRemovalW.reserve(count);
        event->totalPowerW.reserve(count);

        for (const RackFeedback& fb : agg.racks) {
            event->globalRack.push_back(fb.globalRack);
            event->inletTempC.push_back(fb.inletTempC);
            event->outletTempC.push_back(fb.outletTempC);
            event->coolantTempC.push_back(fb.coolantTempC);
            event->coolingRemovalW.push_back(fb.coolingRemovalW);
            event->totalPowerW.push_back(fb.totalPowerW);
        }

        /* One hop up; the parent link carries hop_up latency. */
        parentFeedbackLink_->send(0, event);

        feedbackAgg_.erase(it);
    }

    void applyScheduleTick(
        int time)
    {
        if (time <
            scheduleWindow_) {
            return;
        }

        int epochStart =
            (time / scheduleWindow_) *
            scheduleWindow_;

        if (scheduleBlock_ == nullptr ||
            scheduleBlock_->startTime != epochStart) {
            auto it =
                pendingSchedule_.find(epochStart);

            if (it == pendingSchedule_.end()) {
                fatal(
                    "missing schedule block at time " +
                    std::to_string(time));
            }

            if (scheduleBlock_ != nullptr)
                delete scheduleBlock_;

            scheduleBlock_ = it->second;
            scheduleCursor_ = 0;
            pendingSchedule_.erase(it);
        }

        int offset =
            time -
            scheduleBlock_->startTime;

        if (offset < 0 ||
            offset >=
                scheduleBlock_->tickCount) {
            fatal(
                "schedule tick lies outside block");
        }

        while (scheduleCursor_ <
               scheduleBlock_->changeCount()) {
            size_t changeIndex =
                scheduleCursor_;

            int changeTickOffset =
                scheduleBlock_->
                    changeTickOffset[
                        changeIndex];

            if (changeTickOffset >
                offset) {
                break;
            }

            if (changeTickOffset !=
                offset) {
                fatal(
                    "missed schedule change");
            }

            int localNodeIndex =
                scheduleBlock_->
                    changeLocalNodeIndex[
                        changeIndex];

            int localNodeCount =
                localRacks_ *
                nodesPerRack_;

            if (localNodeIndex < 0 ||
                localNodeIndex >=
                    localNodeCount) {
                fatal(
                    "invalid schedule node index");
            }

            int localRack =
                localNodeIndex /
                nodesPerRack_;

            int nodeIndex =
                localNodeIndex %
                nodesPerRack_;

            NodeState& node =
                racks_[
                    static_cast<size_t>(
                        localRack)]
                    .nodes[
                        static_cast<size_t>(
                            nodeIndex)];

            node.cpuUsed =
                scheduleBlock_->
                    changeCpuUsed[
                        changeIndex];

            node.gpuUsed =
                scheduleBlock_->
                    changeGpuUsed[
                        changeIndex];

            node.runningJobs =
                scheduleBlock_->
                    changeRunningJobs[
                        changeIndex];

            node.cpuUtilization =
                scheduleBlock_->
                    changeCpuUtilization[
                        changeIndex];

            node.gpuUtilization =
                scheduleBlock_->
                    changeGpuUtilization[
                        changeIndex];

            scheduleCursor_++;
        }

        if (offset ==
            scheduleBlock_->tickCount - 1) {
            if (scheduleCursor_ !=
                scheduleBlock_->changeCount()) {
                fatal(
                    "unconsumed schedule changes");
            }

            delete scheduleBlock_;
            scheduleBlock_ = nullptr;
            scheduleCursor_ = 0;
        }
    }

    void updateLocalPower()
    {
        for (RackState& rack :
             racks_) {
            rack.totalPowerW = 0.0;

            for (NodeState& node :
                 rack.nodes) {
                node.cpuUtilization =
                    clampDouble(
                        node.cpuUtilization,
                        0.0,
                        1.0);

                node.gpuUtilization =
                    clampDouble(
                        node.gpuUtilization,
                        0.0,
                        1.0);

                node.powerW =
                    NODE_IDLE_POWER_W +
                    CPU_DYNAMIC_POWER_W *
                        node.cpuUtilization +
                    GPU_DYNAMIC_POWER_W *
                        node.gpuUtilization;

                rack.totalPowerW +=
                    node.powerW;
            }

            itEnergyJ_ +=
                rack.totalPowerW *
                DT_SECONDS;
        }
    }

    double integrateRackThermal(
        RackState& rack,
        double incomingHeatW)
    {
        double nextInlet =
            AMBIENT_TEMP_C +
            incomingHeatW /
                AIRFLOW_HEAT_CAPACITY_W_K;

        double heatToAir =
            AIRFLOW_HEAT_CAPACITY_W_K *
            (rack.outletTempC -
             nextInlet);

        double heatToCoolant =
            RACK_TO_COOLANT_UA_W_K *
            (rack.outletTempC -
             rack.coolantTempC);

        double outletDerivative =
            (rack.totalPowerW -
             heatToAir -
             heatToCoolant) /
            RACK_HEAT_CAPACITY_J_K;

        double nextOutlet =
            rack.outletTempC +
            DT_SECONDS *
                outletDerivative;

        double temperatureError =
            nextInlet -
            TARGET_INLET_TEMP_C;

        double requestedSetpoint =
            INITIAL_SETPOINT_C -
            COOLING_SETPOINT_RESET_C_C *
                temperatureError;

        requestedSetpoint =
            clampDouble(
                requestedSetpoint,
                MIN_COOLING_SETPOINT_C,
                MAX_COOLING_SETPOINT_C);

        double setpointDelta =
            requestedSetpoint -
            rack.coolingSetpointC;

        setpointDelta =
            clampDouble(
                setpointDelta,
                -COOLING_SETPOINT_RATE_C_S *
                    DT_SECONDS,
                COOLING_SETPOINT_RATE_C_S *
                    DT_SECONDS);

        double nextSetpoint =
            rack.coolingSetpointC +
            setpointDelta;

        double requestedCooling =
            COOLING_PROPORTIONAL_W_K *
            rack.coolantFlow *
            std::max(
                rack.coolantTempC -
                    nextSetpoint,
                0.0);

        requestedCooling =
            clampDouble(
                requestedCooling,
                0.0,
                MAX_COOLING_REMOVAL_W);

        double previousCooling =
            rack.coolingRemovalW;

        double coolingDerivative =
            (requestedCooling -
             previousCooling) /
            COOLING_ACTUATOR_TIME_S;

        double nextCooling =
            previousCooling +
            DT_SECONDS *
                coolingDerivative;

        nextCooling =
            clampDouble(
                nextCooling,
                0.0,
                MAX_COOLING_REMOVAL_W);

        double coolantDerivative =
            (heatToCoolant -
             previousCooling) /
            COOLANT_HEAT_CAPACITY_J_K;

        double nextCoolant =
            rack.coolantTempC +
            DT_SECONDS *
                coolantDerivative;

        double averageCooling =
            0.5 *
            (previousCooling +
             nextCooling);

        rack.inletTempC =
            nextInlet;

        rack.outletTempC =
            nextOutlet;

        rack.coolantTempC =
            nextCoolant;

        rack.coolingSetpointC =
            nextSetpoint;

        rack.coolingRemovalW =
            nextCooling;

        return
            (averageCooling /
             COOLING_COP) *
            DT_SECONDS;
    }

    void sendHeat(
        int localSource,
        int destination,
        int time)
    {
        RackState& sourceRack =
            racks_[
                static_cast<size_t>(
                    localSource)];

        int source =
            sourceRack.globalRack;

        int delay =
            thermalDelay(
                source,
                destination);

        int arrivalTime =
            time +
            delay;

        double exhaustHeat =
            AIRFLOW_HEAT_CAPACITY_W_K *
            std::max(
                sourceRack.outletTempC -
                    AMBIENT_TEMP_C,
                0.0);

        double heatPerNeighbor =
            HEAT_RECIRCULATION_FRACTION *
            exhaustHeat /
            2.0;

        if (destination >=
                rackOffset_ &&
            destination <
                rackOffset_ +
                localRacks_) {
            int localDestination =
                destination -
                rackOffset_;

            pendingHeat_[
                static_cast<size_t>(
                    arrivalTime)]
                [static_cast<size_t>(
                    localDestination)] +=
                heatPerNeighbor;

            return;
        }

        auto* event =
            new HeatEvent(
                source,
                destination,
                time,
                arrivalTime,
                heatPerNeighbor);

        int left =
            ringLeft(source);

        if (destination == left) {
            leftLink_->send(
                delay - 1,
                event);
        } else {
            rightLink_->send(
                delay - 1,
                event);
        }
    }

    void sendDelayedHeat(
        int time)
    {
        for (int localRack = 0;
             localRack < localRacks_;
             ++localRack) {
            int source =
                racks_[
                    static_cast<size_t>(
                        localRack)]
                    .globalRack;

            sendHeat(
                localRack,
                ringLeft(source),
                time);

            sendHeat(
                localRack,
                ringRight(source),
                time);
        }
    }

    bool clockTick(
        SST::Cycle_t)
    {
        if (!timerStarted_) {
            timerStart_ =
                MPI_Wtime();

            timerStarted_ = true;
        }

        if (currentTime_ < steps_) {
            /*
             * As in DSTM, feedback is generated before the
             * current timestamp's thermal integration.
             */
            if (currentTime_ %
                    scheduleWindow_ ==
                0) {
                stashOwnFeedback(
                    currentTime_);
            }

            applyScheduleTick(
                currentTime_);

            updateLocalPower();

            for (int localRack = 0;
                 localRack < localRacks_;
                 ++localRack) {
                double incomingHeat =
                    pendingHeat_[
                        static_cast<size_t>(
                            currentTime_)]
                        [static_cast<size_t>(
                            localRack)];

                coolingEnergyJ_ +=
                    integrateRackThermal(
                        racks_[
                            static_cast<size_t>(
                                localRack)],
                        incomingHeat);

                maxInletTempC_ =
                    std::max(
                        maxInletTempC_,
                        racks_[
                            static_cast<size_t>(
                                localRack)]
                            .inletTempC);

                maxOutletTempC_ =
                    std::max(
                        maxOutletTempC_,
                        racks_[
                            static_cast<size_t>(
                                localRack)]
                            .outletTempC);
            }

            sendDelayedHeat(
                currentTime_);
        }

        currentTime_++;

        int finalTime =
            steps_ +
            scheduleWindow_ +
            MAX_THERMAL_DELAY +
            1;

        if (currentTime_ >=
            finalTime) {
            timerEnd_ =
                MPI_Wtime();

            timerStopped_ = true;

            primaryComponentOKToEndSim();
            return true;
        }

        return false;
    }
};

} // namespace dcbench
