#include <sst/core/component.h>
#include <sst/core/event.h>
#include <sst/core/link.h>
#include <sst/core/params.h>
#include <sst/core/serialization/serializer.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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
static constexpr double RECIRCULATION = 0.04;

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

    HeatEvent() = default;

    HeatEvent(int src, int dst, int when, double value) :
        source(src),
        destination(dst),
        arrival(when),
        heat(value)
    {}

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
};

struct Allocation {
    int rack = 0;
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
        {"left",  "Left neighboring domain",  {"dcbench.HeatEvent"}},
        {"right", "Right neighboring domain", {"dcbench.HeatEvent"}}
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

        racks_.resize(localRacks_);

        for (int r = 0; r < localRacks_; ++r) {
            racks_[r].globalId = rackOffset_ + r;
            racks_[r].nodes.resize(nodesPerRack_);
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
        std::printf(
            "domain=%d jobs_started=%llu jobs_completed=%llu "
            "energy_kwh=%.6f max_inlet=%.3f max_outlet=%.3f\n",
            domain_,
            static_cast<unsigned long long>(jobsStarted_),
            static_cast<unsigned long long>(jobsCompleted_),
            energyJ_ / 3600000.0,
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

    uint64_t jobsStarted_ = 0;
    uint64_t jobsCompleted_ = 0;

    double energyJ_ = 0.0;
    double maxInlet_ = -INFINITY;
    double maxOutlet_ = -INFINITY;

    void loadJobs()
    {
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

            if ((job.id % domains_) != domain_)
                continue;

            job.cpuUtil = clamp(job.cpuUtil, 0.0, 1.0);
            job.gpuUtil = clamp(job.gpuUtil, 0.0, 1.0);

            jobs_.push_back(job);
        }
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

    void releaseCompletedJobs()
    {
        for (Job& job : jobs_) {
            if (job.status != JobStatus::RUNNING ||
                job.completion > currentStep_) {
                continue;
            }

            for (const Allocation& allocation : job.allocations) {
                Node& node =
                    racks_[allocation.rack].nodes[allocation.node];

                node.cpuUsed -= job.cpuPerNode;
                node.gpuUsed -= job.gpuPerNode;
                node.runningJobs--;
            }

            job.allocations.clear();
            job.status = JobStatus::COMPLETE;
            jobsCompleted_++;
        }
    }

    void admitJobs()
    {
        for (Job& job : jobs_) {
            if (job.status == JobStatus::FUTURE &&
                job.arrival <= currentStep_) {
                job.status = JobStatus::QUEUED;
            }
        }
    }

    bool canHost(const Node& node, const Job& job) const
    {
        return CPU_CAPACITY - node.cpuUsed >= job.cpuPerNode &&
               GPU_CAPACITY - node.gpuUsed >= job.gpuPerNode;
    }

    bool startJob(Job& job)
    {
        std::vector<Allocation> candidates;

        for (int r = 0;
             r < localRacks_ &&
             static_cast<int>(candidates.size()) < job.requestedNodes;
             ++r) {
            for (int n = 0;
                 n < nodesPerRack_ &&
                 static_cast<int>(candidates.size()) <
                     job.requestedNodes;
                 ++n) {
                if (canHost(racks_[r].nodes[n], job))
                    candidates.push_back({r, n});
            }
        }

        if (static_cast<int>(candidates.size()) !=
            job.requestedNodes) {
            return false;
        }

        for (const Allocation& allocation : candidates) {
            Node& node =
                racks_[allocation.rack].nodes[allocation.node];

            node.cpuUsed += job.cpuPerNode;
            node.gpuUsed += job.gpuPerNode;
            node.runningJobs++;
        }

        job.allocations = std::move(candidates);
        job.completion = currentStep_ + job.duration;
        job.status = JobStatus::RUNNING;
        jobsStarted_++;

        return true;
    }

    void scheduleJobs()
    {
        for (Job& job : jobs_) {
            if (job.status == JobStatus::QUEUED)
                startJob(job);
        }
    }

    void updatePower()
    {
        for (Rack& rack : racks_) {
            rack.power = 0.0;

            for (Node& node : rack.nodes) {
                node.cpuUtil = 0.0;
                node.gpuUtil = 0.0;
            }
        }

        for (const Job& job : jobs_) {
            if (job.status != JobStatus::RUNNING)
                continue;

            for (const Allocation& allocation : job.allocations) {
                Node& node =
                    racks_[allocation.rack].nodes[allocation.node];

                node.cpuUtil +=
                    (static_cast<double>(job.cpuPerNode) /
                     CPU_CAPACITY) * job.cpuUtil;

                node.gpuUtil +=
                    (static_cast<double>(job.gpuPerNode) /
                     GPU_CAPACITY) * job.gpuUtil;
            }
        }

        for (Rack& rack : racks_) {
            for (Node& node : rack.nodes) {
                node.cpuUtil = clamp(node.cpuUtil, 0.0, 1.0);
                node.gpuUtil = clamp(node.gpuUtil, 0.0, 1.0);

                node.power =
                    IDLE_POWER_W +
                    CPU_POWER_W * node.cpuUtil +
                    GPU_POWER_W * node.gpuUtil;

                rack.power += node.power;
            }

            energyJ_ += rack.power;
        }
    }

    void integrateThermal(Rack& rack, double incomingHeat)
    {
        double inletTarget =
            AMBIENT_C + incomingHeat / AIRFLOW;

        double rackRemoved =
            RACK_UA * (rack.outlet - rack.coolant);

        double nextOutlet =
            rack.outlet +
            (rack.power + incomingHeat - rackRemoved) /
                RACK_CAPACITY;

        double nextInlet =
            rack.inlet + 0.5 * (inletTarget - rack.inlet);

        double requestedSetpoint =
            rack.setpoint -
            CONTROL_KP * (nextInlet - TARGET_INLET_C);

        requestedSetpoint =
            clamp(requestedSetpoint, 12.0, 24.0);

        double setpointDelta =
            clamp(requestedSetpoint - rack.setpoint,
                  -SETPOINT_RATE,
                  SETPOINT_RATE);

        double nextSetpoint = rack.setpoint + setpointDelta;

        double coolingRemoved =
            COOLING_GAIN *
            std::max(rack.coolant - nextSetpoint, 0.0);

        double nextCoolant =
            rack.coolant +
            (rackRemoved - coolingRemoved) / COOLANT_CAPACITY;

        rack.inlet = nextInlet;
        rack.outlet = nextOutlet;
        rack.coolant = nextCoolant;
        rack.setpoint = nextSetpoint;

        maxInlet_ = std::max(maxInlet_, rack.inlet);
        maxOutlet_ = std::max(maxOutlet_, rack.outlet);
    }

    void sendHeat(int localSource, int destination)
    {
        Rack& sourceRack = racks_[localSource];
        int source = sourceRack.globalId;
        int delay = thermalDelay(source, destination);
        int arrival = currentStep_ + delay;

        double heat = RECIRCULATION * sourceRack.power / 2.0;

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
        if (currentStep_ < steps_) {
            releaseCompletedJobs();
            admitJobs();
            scheduleJobs();
            updatePower();

            for (int r = 0; r < localRacks_; ++r) {
                integrateThermal(
                    racks_[r],
                    pendingHeat_[currentStep_][r]);
            }

            publishHeat();
        }

        currentStep_++;

        /*
         * Drain delayed events after the final simulated second.
         */
        if (currentStep_ >= steps_ + MAX_THERMAL_DELAY + 1) {
            primaryComponentOKToEndSim();
            return true;
        }

        return false;
    }
};

} // namespace dcbench
