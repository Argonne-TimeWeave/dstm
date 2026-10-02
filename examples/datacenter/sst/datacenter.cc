#include <sst/core/component.h>
#include <sst/core/event.h>
#include <sst/core/link.h>
#include <sst/core/params.h>
#include <sst/core/serialization/serialize.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

namespace dcbench {

static constexpr int MAX_THERMAL_DELAY = 3;

static constexpr int CPU_CAPACITY = 64;
static constexpr int GPU_CAPACITY = 8;

static constexpr double IDLE_POWER_W = 180.0;
static constexpr double CPU_POWER_W = 320.0;
static constexpr double GPU_POWER_W = 2400.0;

static constexpr double AMBIENT_C = 22.0;
static constexpr double INITIAL_COOLANT_C = 18.0;
static constexpr double INITIAL_OUTLET_C = 24.0;
static constexpr double INITIAL_SETPOINT_C = 18.0;
static constexpr double TARGET_INLET_C = 24.0;

static constexpr double RACK_CAPACITY = 120000.0;
static constexpr double COOLANT_CAPACITY = 500000.0;
static constexpr double RACK_UA = 6000.0;
static constexpr double AIRFLOW = 8000.0;
static constexpr double COOLING_GAIN = 12000.0;
static constexpr double CONTROL_KP = 0.10;
static constexpr double SETPOINT_RATE = 0.25;

static constexpr double MAX_COOLING_W = 100000.0;
static constexpr double COOLING_ACTUATOR_TIME_S = 20.0;
static constexpr double COOLING_COP = 4.0;

static constexpr double RECIRCULATION = 0.04;

double simulationStart_ = 0.0;
double simulationEnd_ = 0.0;
bool simulationTimerStarted_ = false;
bool simulationTimerStopped_ = false;

static double clamp(double x, double low, double high)
{
    return std::max(low, std::min(x, high));
}

static int thermalDelay(int a, int b)
{
    uint64_t lo = static_cast<uint64_t>(std::min(a, b));
    uint64_t hi = static_cast<uint64_t>(std::max(a, b));
    uint64_t hash = lo * 2654435761ULL + hi * 2246822519ULL;

    return 1 + static_cast<int>(hash % MAX_THERMAL_DELAY);
}

class HeatEvent : public SST::Event {
public:
    int source = 0;
    int destination = 0;
    int arrival = 0;
    double heat = 0.0;

    static constexpr int HEAT_EVENT_PRIORITY = CLOCKPRIORITY - 1;

    HeatEvent()
    {
        setPriority(HEAT_EVENT_PRIORITY);
    }

    HeatEvent(int src, int dst, int when, double value) :
        source(src),
        destination(dst),
        arrival(when),
        heat(value)
    {
        setPriority(HEAT_EVENT_PRIORITY);
    }

    void serialize_order(
        SST::Core::Serialization::serializer& ser) override
    {
        SST::Event::serialize_order(ser);
        ser.primitive(source);
        ser.primitive(destination);
        ser.primitive(arrival);
        ser.primitive(heat);
    }

    ImplementSerializable(HeatEvent);
};

struct ScheduledNode {
    int cpuUsed = 0;
    int gpuUsed = 0;
    int runningJobs = 0;

    double cpuUtil = 0.0;
    double gpuUtil = 0.0;
};

class ScheduleEvent : public SST::Event {
public:
    int destinationDomain = 0;
    int effectiveTime = 0;
    int rackCount = 0;
    int nodesPerRack = 0;

    std::vector<int> cpuUsed;
    std::vector<int> gpuUsed;
    std::vector<int> runningJobs;
    std::vector<double> cpuUtil;
    std::vector<double> gpuUtil;

    static constexpr int SCHEDULE_EVENT_PRIORITY =
        CLOCKPRIORITY - 1;

    ScheduleEvent()
    {
        setPriority(SCHEDULE_EVENT_PRIORITY);
    }

    ScheduleEvent(int destination,
                  int effective,
                  int racks,
                  int nodes) :
        destinationDomain(destination),
        effectiveTime(effective),
        rackCount(racks),
        nodesPerRack(nodes)
    {
        setPriority(SCHEDULE_EVENT_PRIORITY);
    }

    void serialize_order(
        SST::Core::Serialization::serializer& ser) override
    {
        SST::Event::serialize_order(ser);

        ser.primitive(destinationDomain);
        ser.primitive(effectiveTime);
        ser.primitive(rackCount);
        ser.primitive(nodesPerRack);

        SST_SER(cpuUsed);
        SST_SER(gpuUsed);
        SST_SER(runningJobs);
        SST_SER(cpuUtil);
        SST_SER(gpuUtil);
    }

    ImplementSerializable(ScheduleEvent);
};

enum class JobStatus {
    FUTURE,
    QUEUED,
    RUNNING,
    COMPLETE
};

struct Node {
    int cpuUsed = 0;
    int gpuUsed = 0;
    int runningJobs = 0;

    double cpuUtil = 0.0;
    double gpuUtil = 0.0;
    double power = IDLE_POWER_W;
};

struct Rack {
    int globalId = 0;
    std::vector<Node> nodes;

    double power = 0.0;
    double inlet = AMBIENT_C;
    double outlet = INITIAL_OUTLET_C;
    double coolant = INITIAL_COOLANT_C;
    double setpoint = INITIAL_SETPOINT_C;
    double coolingRemoval = 0.0;
};

struct Allocation {
    int globalRack = 0;
    int node = 0;
};

struct Job {
    int id = 0;
    int arrival = 0;
    int duration = 0;
    int requestedNodes = 0;
    int cpuPerNode = 0;
    int gpuPerNode = 0;

    double cpuUtil = 0.0;
    double gpuUtil = 0.0;

    JobStatus status = JobStatus::FUTURE;
    int completion = -1;

    std::vector<Allocation> allocations;
};

class Datacenter : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(
        Datacenter,
        "dcbench",
        "datacenter",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "Minimal datacenter PDES benchmark",
        COMPONENT_CATEGORY_PROCESSOR
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"domain",         "Domain identifier", "0"},
        {"domains",        "Number of domains", "1"},
        {"racks",          "Total rack count", "16"},
        {"nodes_per_rack", "Nodes per rack", "8"},
        {"steps",          "Simulation seconds", "300"},
        {"jobs",           "CSV job trace", "jobs.csv"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"left",
         "Left neighboring thermal domain",
         {"dcbench.HeatEvent"}},

        {"right",
         "Right neighboring thermal domain",
         {"dcbench.HeatEvent"}},

        /*
         * Domain 0 uses schedule_1, schedule_2, ...
         * Every nonzero domain receives on schedule_0.
         */
        {"schedule%(domains)d",
         "Central scheduler control link",
         {"dcbench.ScheduleEvent"}}
    )

    Datacenter(SST::ComponentId_t id, SST::Params& params) :
        SST::Component(id)
    {
        domain_ = params.find<int>("domain", 0);
        domains_ = params.find<int>("domains", 1);
        totalRacks_ = params.find<int>("racks", 16);
        nodesPerRack_ = params.find<int>("nodes_per_rack", 8);
        steps_ = params.find<int>("steps", 300);
        jobPath_ = params.find<std::string>("jobs", "jobs.csv");

        if (domains_ < 1 || totalRacks_ < 3 ||
            totalRacks_ % domains_ != 0) {
            getSimulationOutput().fatal(
                CALL_INFO, 1,
                "racks must be >= 3 and divisible by domains\n");
        }

        localRacks_ = totalRacks_ / domains_;
        rackOffset_ = domain_ * localRacks_;

        left_ = configureLink(
            "left",
            new SST::Event::Handler2<
                Datacenter, &Datacenter::handleHeat>(this));

        right_ = configureLink(
            "right",
            new SST::Event::Handler2<
                Datacenter, &Datacenter::handleHeat>(this));

        if (domains_ > 1 && (!left_ || !right_)) {
            getSimulationOutput().fatal(
                CALL_INFO, 1, "left/right links are required\n");
        }

        scheduleLinks_.resize(
            static_cast<size_t>(domains_), nullptr);

        if (domain_ == 0) {
            for (int destination = 1;
                 destination < domains_;
                 ++destination) {
                std::string port =
                    "schedule" + std::to_string(destination);

                scheduleLinks_[destination] = configureLink(
                    port,
                    new SST::Event::Handler2<
                        Datacenter,
                        &Datacenter::handleSchedule>(this));

                if (!scheduleLinks_[destination]) {
                    getSimulationOutput().fatal(
                        CALL_INFO, 1,
                        "scheduler link %s is required\n",
                        port.c_str());
                }
            }
        } else {
            scheduleLinks_[0] = configureLink(
                "schedule0",
                new SST::Event::Handler2<
                    Datacenter,
                    &Datacenter::handleSchedule>(this));

            if (!scheduleLinks_[0]) {
                getSimulationOutput().fatal(
                    CALL_INFO, 1,
                    "schedule_0 link is required on domain %d\n",
                    domain_);
            }
        }

        racks_.resize(localRacks_);

        for (int r = 0; r < localRacks_; ++r) {
            racks_[r].globalId = rackOffset_ + r;
            racks_[r].nodes.resize(nodesPerRack_);
        }

        pendingSchedules_.resize(
            static_cast<size_t>(steps_ + MAX_THERMAL_DELAY + 2));

        scheduleReady_.resize(
            static_cast<size_t>(steps_ + MAX_THERMAL_DELAY + 2),
            false);

        if (domain_ == 0) {
            schedulerNodes_.resize(
                static_cast<size_t>(totalRacks_) *
                static_cast<size_t>(nodesPerRack_));
        }

        pendingHeat_.resize(
            static_cast<size_t>(steps_ + MAX_THERMAL_DELAY + 2),
            std::vector<double>(localRacks_, 0.0));

        loadJobs();

        registerAsPrimaryComponent();
        primaryComponentDoNotEndSim();

        registerClock(
            "1s",
            new SST::Clock::Handler2<
                Datacenter, &Datacenter::clockTick>(this));
    }

    void finish() override
    {
        double simulationSeconds = simulationTimerStopped_
            ? simulationEnd_ - simulationStart_
            : 0.0;

        std::printf(
            "domain=%d simulation_runtime_seconds=%.9f\n",
            domain_,
            simulationSeconds);

        std::printf(
            "domain=%d jobs_started=%llu jobs_completed=%llu "
            "it_energy_kwh=%.6f cooling_energy_kwh=%.6f "
            "facility_energy_kwh=%.6f "
            "max_inlet=%.3f max_outlet=%.3f\n",
            domain_,
            static_cast<unsigned long long>(jobsStarted_),
            static_cast<unsigned long long>(jobsCompleted_),
            energyJ_ / 3600000.0,
            coolingEnergyJ_ / 3600000.0,
            (energyJ_ + coolingEnergyJ_) / 3600000.0,
            maxInlet_,
            maxOutlet_);
    }

private:
    int domain_ = 0;
    int domains_ = 1;
    int totalRacks_ = 0;
    int localRacks_ = 0;
    int rackOffset_ = 0;
    int nodesPerRack_ = 0;
    int steps_ = 0;
    int currentStep_ = 0;

    std::string jobPath_;

    SST::Link* left_ = nullptr;
    SST::Link* right_ = nullptr;

    std::vector<Rack> racks_;
    std::vector<Job> jobs_;
    std::vector<std::vector<double>> pendingHeat_;

    /*
     * Only domain 0 owns and modifies these nodes. They represent the
     * authoritative resource state for every rack in the datacenter.
     */
    std::vector<Node> schedulerNodes_;

    /*
     * Domain 0 has scheduleLinks_[d] for every d != 0.
     * Other domains only have scheduleLinks_[0].
     */
    std::vector<SST::Link*> scheduleLinks_;

    /*
     * Schedules are indexed by their logical effective time.
     */
    std::vector<std::vector<ScheduledNode>> pendingSchedules_;
    std::vector<bool> scheduleReady_;

    uint64_t jobsStarted_ = 0;
    uint64_t jobsCompleted_ = 0;

    double energyJ_ = 0.0;
    double coolingEnergyJ_ = 0.0;
    double maxInlet_ = -INFINITY;
    double maxOutlet_ = -INFINITY;

    void loadJobs()
    {
        if (domain_ != 0)
            return;
        
        std::ifstream input(jobPath_);

        if (!input) {
            getSimulationOutput().fatal(
                CALL_INFO, 1, "cannot open job trace %s\n",
                jobPath_.c_str());
        }

        std::string line;
        int id = 0;

        while (std::getline(input, line)) {
            if (line.empty() || line[0] == '#')
                continue;

            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream stream(line);

            Job job;
            job.id = id++;

            if (!(stream >> job.arrival
                         >> job.duration
                         >> job.requestedNodes
                         >> job.cpuPerNode
                         >> job.gpuPerNode
                         >> job.cpuUtil
                         >> job.gpuUtil)) {
                getSimulationOutput().fatal(
                    CALL_INFO, 1,
                    "malformed job record: %s\n", line.c_str());
            }

            job.cpuUtil = clamp(job.cpuUtil, 0.0, 1.0);
            job.gpuUtil = clamp(job.gpuUtil, 0.0, 1.0);

            jobs_.push_back(job);
        }
    }

    Node& schedulerNode(int globalRack, int node)
    {
        size_t index =
            static_cast<size_t>(globalRack) *
                static_cast<size_t>(nodesPerRack_) +
            static_cast<size_t>(node);

        return schedulerNodes_[index];
    }

    const Node& schedulerNode(int globalRack, int node) const
    {
        size_t index =
            static_cast<size_t>(globalRack) *
                static_cast<size_t>(nodesPerRack_) +
            static_cast<size_t>(node);

        return schedulerNodes_[index];
    }

    void handleHeat(SST::Event* event)
    {
        auto* heat = dynamic_cast<HeatEvent*>(event);

        if (!heat) {
            delete event;
            return;
        }

        int localRack = heat->destination - rackOffset_;

        if (heat->arrival >= 0 &&
            heat->arrival < static_cast<int>(pendingHeat_.size()) &&
            localRack >= 0 &&
            localRack < localRacks_) {
            pendingHeat_[heat->arrival][localRack] += heat->heat;
        }

        delete heat;
    }

    void releaseCompletedJobs(int effectiveTime)
    {
        for (Job& job : jobs_) {
            if (job.status != JobStatus::RUNNING ||
                job.completion > effectiveTime) {
                continue;
            }

            for (const Allocation& allocation : job.allocations) {
                Node& node =
                    schedulerNode(
                        allocation.globalRack,
                        allocation.node);

                node.cpuUsed -= job.cpuPerNode;
                node.gpuUsed -= job.gpuPerNode;
                node.runningJobs--;

                if (node.cpuUsed < 0 ||
                    node.gpuUsed < 0 ||
                    node.runningJobs < 0) {
                    getSimulationOutput().fatal(
                        CALL_INFO, 1,
                        "negative scheduler reservation state\n");
                }
            }

            job.allocations.clear();
            job.status = JobStatus::COMPLETE;
            jobsCompleted_++;
        }
    }

    void admitJobs(int arrivalTime)
    {
        for (Job& job : jobs_) {
            if (job.status == JobStatus::FUTURE &&
                job.arrival <= arrivalTime) {
                job.status = JobStatus::QUEUED;
            }
        }
    }

    bool canHost(const Node& node, const Job& job) const
    {
        return CPU_CAPACITY - node.cpuUsed >= job.cpuPerNode &&
               GPU_CAPACITY - node.gpuUsed >= job.gpuPerNode;
    }

    bool startJob(Job& job, int effectiveTime)
    {
        std::vector<Allocation> candidates;

        /*
         * Same deterministic first-fit order as the centralized DSTM
         * scheduler:
         *
         *   global rack 0..totalRacks-1
         *   node 0..nodesPerRack-1
         */
        for (int rack = 0;
             rack < totalRacks_ &&
             static_cast<int>(candidates.size()) <
                 job.requestedNodes;
             ++rack) {
            for (int nodeIndex = 0;
                 nodeIndex < nodesPerRack_ &&
                 static_cast<int>(candidates.size()) <
                     job.requestedNodes;
                 ++nodeIndex) {
                if (canHost(
                        schedulerNode(rack, nodeIndex),
                        job)) {
                    candidates.push_back(
                        {rack, nodeIndex});
                }
            }
        }

        if (static_cast<int>(candidates.size()) !=
            job.requestedNodes) {
            return false;
        }

        for (const Allocation& allocation : candidates) {
            Node& node =
                schedulerNode(
                    allocation.globalRack,
                    allocation.node);

            node.cpuUsed += job.cpuPerNode;
            node.gpuUsed += job.gpuPerNode;
            node.runningJobs++;
        }

        job.allocations = std::move(candidates);
        job.completion = effectiveTime + job.duration;
        job.status = JobStatus::RUNNING;
        jobsStarted_++;

        return true;
    }

    void scheduleJobs(int effectiveTime)
    {
        for (Job& job : jobs_) {
            if (job.status != JobStatus::QUEUED)
                continue;

            if (!startJob(job, effectiveTime))
                break;
        }
    }

    void updatePower()
    {
        for (Rack& rack : racks_) {
            rack.power = 0.0;

            for (Node& node : rack.nodes) {
                node.cpuUtil =
                    clamp(node.cpuUtil, 0.0, 1.0);

                node.gpuUtil =
                    clamp(node.gpuUtil, 0.0, 1.0);

                node.power =
                    IDLE_POWER_W +
                    CPU_POWER_W * node.cpuUtil +
                    GPU_POWER_W * node.gpuUtil;

                rack.power += node.power;
            }

            energyJ_ += rack.power;
        }
    }

    void rebuildSchedulerUtilization()
    {
        for (Node& node : schedulerNodes_) {
            node.cpuUtil = 0.0;
            node.gpuUtil = 0.0;
        }

        for (const Job& job : jobs_) {
            if (job.status != JobStatus::RUNNING)
                continue;

            for (const Allocation& allocation : job.allocations) {
                Node& node =
                    schedulerNode(
                        allocation.globalRack,
                        allocation.node);

                node.cpuUtil +=
                    (static_cast<double>(job.cpuPerNode) /
                     CPU_CAPACITY) *
                    job.cpuUtil;

                node.gpuUtil +=
                    (static_cast<double>(job.gpuPerNode) /
                     GPU_CAPACITY) *
                    job.gpuUtil;
            }
        }

        for (Node& node : schedulerNodes_) {
            node.cpuUtil = clamp(node.cpuUtil, 0.0, 1.0);
            node.gpuUtil = clamp(node.gpuUtil, 0.0, 1.0);
        }
    }

    void handleSchedule(SST::Event* event)
    {
        auto* schedule =
            dynamic_cast<ScheduleEvent*>(event);

        if (!schedule) {
            delete event;
            return;
        }

        if (schedule->destinationDomain != domain_) {
            getSimulationOutput().fatal(
                CALL_INFO, 1,
                "schedule for domain %d delivered to domain %d\n",
                schedule->destinationDomain,
                domain_);
        }

        size_t expected =
            static_cast<size_t>(localRacks_) *
            static_cast<size_t>(nodesPerRack_);

        if (schedule->rackCount != localRacks_ ||
            schedule->nodesPerRack != nodesPerRack_ ||
            schedule->cpuUsed.size() != expected ||
            schedule->gpuUsed.size() != expected ||
            schedule->runningJobs.size() != expected ||
            schedule->cpuUtil.size() != expected ||
            schedule->gpuUtil.size() != expected) {
            getSimulationOutput().fatal(
                CALL_INFO, 1,
                "invalid schedule payload for domain %d\n",
                domain_);
        }

        if (schedule->effectiveTime < 0 ||
            schedule->effectiveTime >=
                static_cast<int>(pendingSchedules_.size())) {
            getSimulationOutput().fatal(
                CALL_INFO, 1,
                "invalid schedule effective time %d\n",
                schedule->effectiveTime);
        }

        std::vector<ScheduledNode>& snapshot =
            pendingSchedules_[schedule->effectiveTime];

        snapshot.resize(expected);

        for (size_t i = 0; i < expected; ++i) {
            snapshot[i].cpuUsed = schedule->cpuUsed[i];
            snapshot[i].gpuUsed = schedule->gpuUsed[i];
            snapshot[i].runningJobs =
                schedule->runningJobs[i];
            snapshot[i].cpuUtil = schedule->cpuUtil[i];
            snapshot[i].gpuUtil = schedule->gpuUtil[i];
        }

        scheduleReady_[schedule->effectiveTime] = true;

        delete schedule;
    }

    void applySchedule(int effectiveTime)
    {
        /*
         * Time zero uses the initial empty Node state. Every later modeled
         * timestamp must have a schedule produced during the previous tick.
         */
        if (effectiveTime == 0)
            return;

        if (effectiveTime < 0 ||
            effectiveTime >=
                static_cast<int>(scheduleReady_.size()) ||
            !scheduleReady_[effectiveTime]) {
            getSimulationOutput().fatal(
                CALL_INFO, 1,
                "domain %d has no schedule for time %d\n",
                domain_,
                effectiveTime);
        }

        const std::vector<ScheduledNode>& snapshot =
            pendingSchedules_[effectiveTime];

        size_t index = 0;

        for (int rack = 0; rack < localRacks_; ++rack) {
            for (int nodeIndex = 0;
                 nodeIndex < nodesPerRack_;
                 ++nodeIndex, ++index) {
                Node& node =
                    racks_[rack].nodes[nodeIndex];

                node.cpuUsed = snapshot[index].cpuUsed;
                node.gpuUsed = snapshot[index].gpuUsed;
                node.runningJobs =
                    snapshot[index].runningJobs;
                node.cpuUtil = snapshot[index].cpuUtil;
                node.gpuUtil = snapshot[index].gpuUtil;
            }
        }

        scheduleReady_[effectiveTime] = false;
        pendingSchedules_[effectiveTime].clear();
    }

    std::vector<ScheduledNode> buildScheduleForDomain(
        int destinationDomain) const
    {
        std::vector<ScheduledNode> snapshot;

        size_t count =
            static_cast<size_t>(localRacks_) *
            static_cast<size_t>(nodesPerRack_);

        snapshot.reserve(count);

        int firstRack =
            destinationDomain * localRacks_;

        for (int localRack = 0;
             localRack < localRacks_;
             ++localRack) {
            int globalRack = firstRack + localRack;

            for (int nodeIndex = 0;
                 nodeIndex < nodesPerRack_;
                 ++nodeIndex) {
                const Node& node =
                    schedulerNode(globalRack, nodeIndex);

                snapshot.push_back({
                    node.cpuUsed,
                    node.gpuUsed,
                    node.runningJobs,
                    node.cpuUtil,
                    node.gpuUtil
                });
            }
        }

        return snapshot;
    }

    void publishSchedules(int effectiveTime)
    {
        for (int destination = 0;
             destination < domains_;
             ++destination) {
            std::vector<ScheduledNode> snapshot =
                buildScheduleForDomain(destination);

            if (destination == 0) {
                /*
                 * DSTM's write to rank 0's own schedule slot is local.
                 * Preserve the same logical one-tick delay without sending
                 * an SST event to ourselves.
                 */
                pendingSchedules_[effectiveTime] =
                    std::move(snapshot);

                scheduleReady_[effectiveTime] = true;
                continue;
            }

            auto* event = new ScheduleEvent(
                destination,
                effectiveTime,
                localRacks_,
                nodesPerRack_);

            size_t count = snapshot.size();

            event->cpuUsed.reserve(count);
            event->gpuUsed.reserve(count);
            event->runningJobs.reserve(count);
            event->cpuUtil.reserve(count);
            event->gpuUtil.reserve(count);

            for (const ScheduledNode& node : snapshot) {
                event->cpuUsed.push_back(node.cpuUsed);
                event->gpuUsed.push_back(node.gpuUsed);
                event->runningJobs.push_back(
                    node.runningJobs);
                event->cpuUtil.push_back(node.cpuUtil);
                event->gpuUtil.push_back(node.gpuUtil);
            }

            /*
             * The SST link itself has one second of latency. A zero
             * additional delay therefore delivers at logical time t+1.
             */
            scheduleLinks_[destination]->send(0, event);
        }
    }

    void integrateThermal(Rack& rack, double incomingHeat)
    {
        double nextInlet =
            AMBIENT_C + incomingHeat / AIRFLOW;

        double heatToAir =
            AIRFLOW * (rack.outlet - nextInlet);

        double heatToCoolant =
            RACK_UA * (rack.outlet - rack.coolant);

        double nextOutlet =
            rack.outlet +
            (rack.power - heatToAir - heatToCoolant) /
                RACK_CAPACITY;

        /*
         * Absolute reset around INITIAL_SETPOINT_C.
         */
        double requestedSetpoint =
            INITIAL_SETPOINT_C -
            CONTROL_KP * (nextInlet - TARGET_INLET_C);

        requestedSetpoint =
            clamp(requestedSetpoint, 12.0, 24.0);

        double setpointDelta =
            clamp(requestedSetpoint - rack.setpoint,
                  -SETPOINT_RATE,
                  SETPOINT_RATE);

        double nextSetpoint = rack.setpoint + setpointDelta;

        double requestedCooling =
            COOLING_GAIN *
            std::max(rack.coolant - nextSetpoint, 0.0);

        requestedCooling =
            clamp(requestedCooling, 0.0, MAX_COOLING_W);

        double previousCooling = rack.coolingRemoval;

        double nextCooling =
            previousCooling +
            (requestedCooling - previousCooling) /
                COOLING_ACTUATOR_TIME_S;

        nextCooling =
            clamp(nextCooling, 0.0, MAX_COOLING_W);

        /*
         * DSTM uses the previous actuator output for this tick's
         * coolant integration.
         */
        double nextCoolant =
            rack.coolant +
            (heatToCoolant - previousCooling) /
                COOLANT_CAPACITY;

        double averageCooling =
            0.5 * (previousCooling + nextCooling);

        coolingEnergyJ_ += averageCooling / COOLING_COP;

        rack.inlet = nextInlet;
        rack.outlet = nextOutlet;
        rack.coolant = nextCoolant;
        rack.setpoint = nextSetpoint;
        rack.coolingRemoval = nextCooling;

        maxInlet_ = std::max(maxInlet_, rack.inlet);
        maxOutlet_ = std::max(maxOutlet_, rack.outlet);
    }

    void sendHeat(int localSource, int destination)
    {
        Rack& sourceRack = racks_[localSource];
        int source = sourceRack.globalId;
        int delay = thermalDelay(source, destination);
        int arrival = currentStep_ + delay;

        double exhaustHeat =
            AIRFLOW *
            std::max(sourceRack.outlet - AMBIENT_C, 0.0);

        double heat =
            RECIRCULATION * exhaustHeat / 2.0;

        if (destination >= rackOffset_ &&
            destination < rackOffset_ + localRacks_) {
            int localDestination = destination - rackOffset_;

            if (arrival < static_cast<int>(pendingHeat_.size()))
                pendingHeat_[arrival][localDestination] += heat;

            return;
        }

        auto* event =
            new HeatEvent(source, destination, arrival, heat);

        /*
         * Every link has one second of latency. The additional delay is
         * therefore physical_delay - 1.
         */
        if (destination ==
            (source + totalRacks_ - 1) % totalRacks_) {
            left_->send(delay - 1, event);
        } else {
            right_->send(delay - 1, event);
        }
    }

    void publishHeat()
    {
        for (int r = 0; r < localRacks_; ++r) {
            int source = racks_[r].globalId;
            int left =
                (source + totalRacks_ - 1) % totalRacks_;
            int right =
                (source + 1) % totalRacks_;

            sendHeat(r, left);
            sendHeat(r, right);
        }
    }

    bool clockTick(SST::Cycle_t)
    {
        if (!simulationTimerStarted_) {
            simulationStart_ = MPI_Wtime();
            simulationTimerStarted_ = true;
        }

        if (currentStep_ < steps_) {
            /*
             * Consume the schedule published by domain 0 during the
             * preceding logical second.
             */
            applySchedule(currentStep_);

            /*
             * The received utilization drives this timestamp's power and
             * thermal integration.
             */
            updatePower();

            for (int r = 0; r < localRacks_; ++r) {
                integrateThermal(
                    racks_[r],
                    pendingHeat_[currentStep_][r]);
            }

            publishHeat();

            /*
             * Domain 0 computes the global schedule for time t+1 after
             * modeling time t.
             */
            if (domain_ == 0) {
                int effectiveTime = currentStep_ + 1;

                releaseCompletedJobs(effectiveTime);
                admitJobs(currentStep_);
                scheduleJobs(effectiveTime);
                rebuildSchedulerUtilization();
                publishSchedules(effectiveTime);
            }

            if (currentStep_ == steps_ - 1) {
                simulationEnd_ = MPI_Wtime();
                simulationTimerStopped_ = true;
            }
        }

        currentStep_++;

        /*
         * Drain both heat and final schedule events.
         */
        if (currentStep_ >= steps_ + MAX_THERMAL_DELAY + 1) {
            primaryComponentOKToEndSim();
            return true;
        }

        return false;
    }
};

} // namespace dcbench
