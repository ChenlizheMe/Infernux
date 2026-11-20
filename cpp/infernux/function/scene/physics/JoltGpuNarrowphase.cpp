#include "JoltGpuNarrowphase.h"
#include "JoltGpuConstraints.h"
#include "JoltGpuGeometryStore.h"
#include "JoltGpuProgramSources.h"
#include <function/scene/physics/gpu/ConvexGeometryTypes.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace infernux
{
namespace
{
using namespace JPH::GPU;

enum Port : uint8_t
{
    Params,
    Work,
    Shapes,
    States,
    Patches,
    Contacts,
    Geometry,
    Scratch,
    Models,
    SweepBodies,
    SweepChoice,
    Joints,
    Jobs,
    Arguments,
    SweepClock,
    PortCount
};
enum Entry : uint8_t
{
    Initialize,
    QueueConvex,
    PublishJobs,
    Advance,
    Progress,
    EventContactProgress,
    AdvanceSweeps,
    InitializeSweep,
    SweepProgress,
    FlattenRoots,
    MoveToEvent,
    DiscoverEvent,
    JoinEventJoints,
    LinkEvent,
    LinkEventJoints,
    SolveEvent,
    SolveJointEvent,
    EventProgress,
    LinkedEventProgress,
    ReportEvent,
    ResumeMotion,
    EntryCount
};
constexpr std::array<std::string_view, EntryCount> EntryNames{"narrowphase_init_jobs",
                                                              "narrowphase_queue_jobs",
                                                              "narrowphase_publish_jobs",
                                                              "narrowphase_advance_jobs",
                                                              "narrowphase_job_progress",
                                                              "sweep_event_contact_progress",
                                                              "sweep_advance_jobs",
                                                              "sweep_initialize_jobs",
                                                              "sweep_job_progress",
                                                              "sweep_flatten_parents",
                                                              "sweep_move_to_event",
                                                              "sweep_event_contacts",
                                                              "sweep_join_event_joints",
                                                              "sweep_link_event_islands",
                                                              "sweep_link_event_joints",
                                                              "sweep_solve_event_islands",
                                                              "sweep_solve_event_joint_islands",
                                                              "sweep_event_progress",
                                                              "sweep_linked_event_progress",
                                                              "sweep_report_event",
                                                              "sweep_resume_motion"};
constexpr uint32_t Mask(Port port)
{
    return 1u << port;
}
constexpr uint32_t JobPublicationPorts = Mask(Scratch) | Mask(Jobs) | Mask(Arguments);
constexpr std::array<uint32_t, EntryCount> EntryPorts{Mask(Work) | Mask(Scratch) | Mask(Jobs),
                                                      Mask(Params) | Mask(Work) | Mask(Shapes) | Mask(Patches) |
                                                          Mask(Jobs),
                                                      JobPublicationPorts,
                                                      0xfdu | Mask(Work) | Mask(Jobs),
                                                      JobPublicationPorts,
                                                      0x81u | Mask(Arguments),
                                                      0x7d7u | Mask(SweepClock),
                                                      0x483u | Mask(SweepClock),
                                                      0x483u | Mask(Patches) | Mask(Arguments) | Mask(SweepClock),
                                                      0x603u,
                                                      0x202u | Mask(SweepClock),
                                                      0x7f5u,
                                                      0xd03u,
                                                      0x523u,
                                                      0xd03u,
                                                      0x721u,
                                                      0xf21u,
                                                      0x481u | Mask(Arguments) | Mask(SweepClock),
                                                      0x481u | Mask(Arguments) | Mask(SweepClock),
                                                      0x333u,
                                                      0x303u | Mask(SweepClock)};
constexpr uint32_t HeaderWords = 4u, MaxSlots = 32u;
constexpr uint32_t AdvancesPerPacket = JoltGpuNarrowphase::Workspace::AdvancesPerPacket;
constexpr uint32_t ResolveDispatchesPerPacket = 4u;
constexpr uint32_t MaxSweepDispatches = 16u;
constexpr uint64_t WordBytes = sizeof(UInt4);
static_assert(SweepMotionChunk == WorkItemLimit(Domain::Bodies));

struct PortLayout
{
    std::string_view name;
    uint64_t stride;
    std::array<std::string_view, 8> fields;
    uint32_t integers;
    bool readOnly;
};
constexpr std::array<PortLayout, PortCount> Ports{{
    {"step_params",
     sizeof(StepParams),
     {"counts", "solve", "gravityDt", "contact", "restitution", "selection", "selectedCounts"},
     0x63u,
     true},
    {"stage_work", sizeof(StageWork), {"range"}, 1u, true},
    {"shape_models",
     sizeof(ShapeModel),
     {"meta", "geometry", "localPositionCOM", "localRotation", "material", "surface", "source", "geometryReference"},
     0xe1u,
     true},
    {"body_states", sizeof(BodyState), {"positionCOM", "rotation", "linearVelocity", "angularVelocity"}, 0u, true},
    {"contact_patches", sizeof(ContactPatch), {"leaves"}, 1u, false},
    {"contacts", sizeof(Contact), {"meta", "normalDepth", "localAnchorA", "localAnchorB", "impulse"}, 1u, false},
    {"convex_geometry", WordBytes, {}, 0u, true},
    {"convex_scratch", WordBytes, {}, 0u, false},
    {"body_models", sizeof(BodyModel), {"meta", "mass", "invInertiaDiagonal", "inertiaRotation", "damping"}, 1u, true},
    {"sweep_bodies",
     sizeof(SweepBody),
     {"positionCOM", "rotation", "linearVelocity", "angularVelocity", "displacement", "linearResponse",
      "angularResponse"},
     0u,
     false},
    {"sweep_choice", sizeof(uint32_t), {}, 1u, false},
    {"joint_stream", sizeof(QueryRecordWord), {"value"}, 1u, false},
    {"convex_jobs", sizeof(uint32_t), {}, 1u, false},
    {"convex_dispatch_args", WordBytes, {}, 0u, false},
    {"sweep_clock", WordBytes, {}, 0u, false},
}};

uint32_t Groups(uint32_t count)
{
    return count / 64u + uint32_t(count % 64u != 0u);
}

void ValidateLayout(Port port, const ShaderResourceLayout &resource)
{
    const auto &expected = Ports[port];
    const bool uniform = port == Params || port == Work;
    const auto &value = resource.value;
    if (resource.set != 0u ||
        resource.binding.type != (uniform ? rhi::BindingType::UniformBuffer : rhi::BindingType::StorageBuffer) ||
        (uniform ? value.byteSize : resource.elementStride) != expected.stride ||
        (!uniform && resource.binding.readOnlyStorage != expected.readOnly))
        throw std::invalid_argument("GPU narrowphase binding differs from shared POD: " + resource.name);
    if (port == Geometry || port == Scratch || port == Arguments || port == SweepClock) {
        if (value.kind != ShaderValueKind::Vector || value.scalar != ShaderScalarKind::UInt32 || value.rows != 1u ||
            value.columns != 4u || value.byteSize != WordBytes || !value.fields.empty())
            throw std::invalid_argument("GPU narrowphase geometry/scratch requires raw uint4 storage");
        return;
    }
    if (port == SweepChoice || port == Jobs) {
        if (value.kind != ShaderValueKind::Scalar || value.scalar != ShaderScalarKind::UInt32 ||
            value.byteSize != sizeof(uint32_t))
            throw std::invalid_argument("GPU narrowphase index storage must use unsigned 32-bit scalars");
        return;
    }
    if (value.kind != ShaderValueKind::Struct || value.fields.size() != expected.stride / WordBytes)
        throw std::invalid_argument("GPU narrowphase struct differs from shared POD");
    for (size_t i = 0; i < value.fields.size(); ++i) {
        const auto &field = value.fields[i];
        const auto scalar = expected.integers & (1u << i) ? ShaderScalarKind::UInt32 : ShaderScalarKind::Float32;
        if (field.name != expected.fields[i] || field.offset != i * WordBytes || field.byteSize != WordBytes ||
            field.kind != ShaderValueKind::Vector || field.scalar != scalar || field.rows != 1u || field.columns != 4u)
            throw std::invalid_argument("GPU narrowphase field differs from shared POD");
    }
}
} // namespace

class JoltGpuNarrowphase::Impl
{
  public:
    struct Program
    {
        std::shared_ptr<rhi::ComputeKernel> kernel;
        std::vector<Port> ports;
    };
    struct Execution;

    Impl(rhi::ComputeHost &source, const ShaderTargetCode &target) : host(source)
    {
        if (target.target != ShaderCodeTarget::SpirV && target.target != ShaderCodeTarget::Wgsl)
            throw std::invalid_argument("GPU narrowphase requires the engine SPIR-V/WGSL target");
        for (uint8_t i = 0; i < EntryCount; ++i) {
            const auto code = std::find_if(target.entries.begin(), target.entries.end(),
                                           [&](const auto &item) { return item.name == EntryNames[i]; });
            const uint32_t width = i == Advance ? ConvexJobGroupSize : 64u;
            if (code == target.entries.end() || code->stage != rhi::ShaderStage::Compute ||
                code->workgroupSize != std::array<uint32_t, 3>{width, 1u, 1u})
                throw std::invalid_argument("GPU narrowphase entry/workgroup differs from module");
            auto &program = entries[i];
            std::vector<rhi::ComputeBufferBinding> bindings;
            uint32_t mask = 0;
            for (const auto &resource : target.resources) {
                const std::array<uint32_t, 2> key{resource.set, resource.binding.binding};
                if (std::find(code->bindings.begin(), code->bindings.end(), key) == code->bindings.end())
                    continue;
                const auto port = std::find_if(Ports.begin(), Ports.end(),
                                               [&](const auto &item) { return item.name == resource.name; });
                if (port == Ports.end())
                    throw std::invalid_argument("GPU narrowphase entry contains an unknown resource");
                const auto index = static_cast<Port>(port - Ports.begin());
                ValidateLayout(index, resource);
                if (mask & (1u << index))
                    throw std::invalid_argument("GPU narrowphase entry repeats a resource");
                mask |= 1u << index;
                program.ports.push_back(index);
                bindings.push_back({resource.binding.binding, resource.binding.type, resource.binding.readOnlyStorage});
            }
            if (mask != EntryPorts[i] || code->bindings.size() != program.ports.size())
                throw std::invalid_argument("GPU narrowphase resource mask differs for " + std::string(EntryNames[i]) +
                                            ": " + std::to_string(mask));
            program.kernel = PrepareJoltGpuKernel(host, target.target, *code, std::move(bindings));
        }
    }
    rhi::ComputeHost host;
    std::array<Program, EntryCount> entries;
    std::array<double, 4> advanceNanoseconds{};
};

struct JoltGpuNarrowphase::Impl::Execution : std::enable_shared_from_this<Execution>
{
    enum class Phase
    {
        InitializeJobs,
        QueueJobs,
        AdvanceJobs,
        MoveEvent,
        InitializeEvent,
        DiscoverEvent,
        LinkEvent,
        SolveEvent,
        FinishEvent,
        Done
    };
    using SweepOutcome = JoltGpuSweepOutcome;
    enum class ResolutionDispatch
    {
        Direct,
        EventLinks,
        Motion
    };
    struct ResolveCursor
    {
        Entry entry = EntryCount;
        uint32_t first = 0, parentPasses = 0;
    };
    struct SweepTail
    {
        std::optional<ResolveCursor> links, motion;
        bool linkedSolve = false;
        uint32_t integratedBodies = 0;

        bool Recorded() const noexcept
        {
            return links || motion || integratedBodies;
        }
    };

    Execution(std::shared_ptr<Impl> implementation, const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
              FrameWorkOwner &budget, std::function<void(JoltGpuSweepResult)> completed,
              JoltGpuPrograms *stagePrograms = nullptr, bool continuing = false, bool reportImpacts = false,
              Workspace *resident = nullptr)
        : implementation(std::move(implementation)), storage(storage), stepInputs(inputs), stagePrograms(stagePrograms),
          contactLease(stagePrograms ? inputs.GetSweepContacts() : inputs.GetContacts()),
          geometry(inputs.GetGeometry()), budget(budget), completed(std::move(completed)),
          swept(stagePrograms != nullptr), continuing(continuing), reportImpacts(reportImpacts), resident(resident)
    {
        if (!this->implementation->host.SharesServicesWith(storage.GetHost()) || !contactLease || !this->completed)
            throw std::invalid_argument("GPU narrowphase requires the accepted Tick host/contact workset");
        frozen[Params] = contactLease->parameters;
        frozen[Shapes] = inputs.GetCaptured(Resource::ShapeModels);
        frozen[States] = storage.Get(Resource::BodyStates);
        frozen[Patches] = contactLease->patches;
        frozen[Contacts] = contactLease->contacts;
        if (!geometry || !geometry->buffer)
            throw std::logic_error("GPU convex candidates require their captured geometry lease");
        frozen[Geometry] = geometry->buffer;
        if (swept) {
            frozen[Models] = inputs.GetCaptured(Resource::BodyModels);
            frozen[SweepBodies] = inputs.GetSweepBodies();
            frozen[SweepChoice] = inputs.GetSweepChoices();
            frozen[SweepClock] = inputs.GetSweepClock();
            bodyCount = inputs.GetParameters().selectedCounts.x;
            jointCount = inputs.GetParameters().selectedCounts.z;
            if (jointCount)
                frozen[Joints] = storage.Get(Resource::Joints);
        }
        const auto &caps = this->implementation->host.device.GetCapabilities();
        if (swept && bodyCount > uint64_t(caps.limits.maxComputeWorkgroupCount[0]) * 64u)
            throw std::length_error("GPU sweep choices exceed device dispatch limits");
        const uint64_t words = std::min(caps.portable.maxStorageBufferBinding / WordBytes,
                                        uint64_t(caps.limits.maxComputeWorkgroupCount[0]) * 64u);
        if (words < HeaderWords + ConvexJobSlotWords)
            throw std::length_error("Device cannot bind and initialize one convex job slot");
        slots = static_cast<uint32_t>(std::min(
            {uint64_t(MaxSlots), uint64_t(contactLease->patchSlots), (words - HeaderWords) / ConvexJobSlotWords}));
        scratchWords = HeaderWords + slots * ConvexJobSlotWords;
        if (resident) {
            workspace = std::exchange(*resident, {});
            scratch = std::move(workspace.scratch);
            if (scratch && scratch->GetByteSize() < uint64_t(scratchWords) * WordBytes)
                scratch.reset();
            frozen[Jobs] = std::move(workspace.jobs);
            if (frozen[Jobs] && frozen[Jobs]->GetByteSize() < JobBytes())
                frozen[Jobs].reset();
            frozen[Arguments] = std::move(workspace.arguments);
        }
    }

    JoltGpuControlWork Retain(JoltGpuControlWork work) const
    {
        work.geometry = geometry;
        return work;
    }

    uint64_t JobBytes() const
    {
        return (uint64_t(contactLease->patchSlots) + 1u) * sizeof(uint32_t);
    }

    bool Allocate(std::shared_ptr<rhi::ComputeBuffer> &buffer, uint64_t bytes,
                  rhi::BufferUsageFlags usage = rhi::BufferUsageFlags::None)
    {
        if (!buffer)
            buffer = rhi::ComputeBuffer::TryCreatePrivate(implementation->host,
                                                          {bytes / 4u, rhi::ComputeScalarType::UInt32, 1u, usage});
        return buffer != nullptr;
    }

    bool AllocateScratch()
    {
        if (scratch)
            return true;
        // Choose concurrency before any job has started. Each grant runs the
        // same strided jobs; reducing slots never restarts numerical work.
        uint32_t count = slots;
        while (true) {
            const uint32_t words = HeaderWords + count * ConvexJobSlotWords;
            if (Allocate(scratch, uint64_t(words) * WordBytes)) {
                slots = count;
                scratchWords = words;
                return true;
            }
            if (count <= 1u)
                return false;
            count = (count + 1u) / 2u;
        }
    }

    rhi::ComputeDispatchDesc Dispatch(Entry entry, uint32_t count = 1u,
                                      const std::shared_ptr<rhi::ComputeBuffer> &uniform = {},
                                      const StageWork *initialization = nullptr) const
    {
        const auto &program = implementation->entries[entry];
        rhi::ComputeDispatchDesc dispatch;
        dispatch.kernel = program.kernel;
        const bool jobInit = entry == InitializeSweep;
        const uint32_t initWords = std::max(scratchWords, bodyCount);
        dispatch.groupCountX = Groups(jobInit ? initWords : count);
        if (entry == Advance) {
            dispatch.groupCountX = (slots + ConvexJobGroupSize - 1u) / ConvexJobGroupSize;
            dispatch.indirectArguments = rhi::ComputeIndirectArguments{frozen[Arguments], 0u};
        }
        rhi::ComputeDispatchProof proof;
        for (const auto port : program.ports) {
            const auto &buffer = port == Scratch ? scratch
                                 : port == Work  ? (uniform ? uniform : workUniform)
                                                 : frozen[port];
            const uint32_t binding = static_cast<uint32_t>(dispatch.buffers.size());
            dispatch.buffers.push_back(buffer);
            uint64_t completeWrite = 0u;
            const bool startsClock = jobInit && phase == Phase::InitializeJobs && !continuing && port == SweepClock;
            if ((jobInit && (port == Scratch || port == SweepChoice)) || port == Arguments || startsClock)
                completeWrite = buffer->GetByteSize();
            else if (initialization && port == Scratch)
                completeWrite = uint64_t(initialization->range.x) * WordBytes;
            else if (initialization && port == Jobs)
                completeWrite = uint64_t(initialization->range.y) * sizeof(uint32_t);
            const bool writesMotion =
                entry == MoveToEvent || entry == SolveEvent || entry == SolveJointEvent || entry == ResumeMotion;
            const bool mutablePort = port == Scratch || (port == Patches && entry != QueueConvex) ||
                                     (port == Jobs && entry == QueueConvex) || port == Contacts ||
                                     port == SweepChoice || (writesMotion && port == SweepBodies) ||
                                     (entry == SolveJointEvent && port == Joints) ||
                                     (entry == SweepProgress && port == SweepClock);
            dispatch.bufferAccesses.push_back(completeWrite ? rhi::ComputeBufferAccess::Write
                                              : mutablePort ? rhi::ComputeBufferAccess::ReadWrite
                                                            : rhi::ComputeBufferAccess::Read);
            if (completeWrite)
                proof.writes.push_back({binding, 0u, completeWrite});
            else if (port == Geometry) {
                for (const auto &range : geometry->wordRanges)
                    proof.reads.push_back({binding, range.first * WordBytes, (range.second - range.first) * WordBytes});
            } else
                proof.reads.push_back({binding, 0u, buffer->GetByteSize()});
        }
        dispatch.proof = std::move(proof);
        return dispatch;
    }

    void Finish(SweepOutcome outcome = SweepOutcome::NoImpact, uint32_t integratedBodies = 0u)
    {
        if (resident) {
            workspace.scratch = std::move(scratch);
            workspace.jobs = std::move(frozen[Jobs]);
            workspace.arguments = std::move(frozen[Arguments]);
            *resident = std::move(workspace);
        }
        phase = Phase::Done;
        auto callback = std::move(completed);
        callback({outcome, integratedBodies, reportedCandidates});
    }

    uint32_t AdvanceRounds(size_t costIndex, uint32_t maximum) const
    {
        const auto unitCost = swept ? implementation->advanceNanoseconds[costIndex] : workspace.advanceNanoseconds;
        if (unitCost <= 0.0)
            return std::min(swept ? 1u : 4u, maximum);
        const auto available = budget.ResourceRemaining(FrameWorkResource::Gpu);
        const auto window = available ? available : budget.ResourceState(FrameWorkResource::Gpu).limit;
        return static_cast<uint32_t>(std::clamp(double(window) / unitCost, 1.0, double(maximum)));
    }

    bool AppendAdvance(uint32_t rounds, std::vector<rhi::ComputeBufferUpdate> &uploads,
                       std::vector<rhi::ComputeDispatchDesc> &dispatches)
    {
        auto &uniform = workspace.advanceRanges[rounds - 1u];
        if (!Allocate(uniform, sizeof(StageWork)))
            return false;
        if (!uniform->HasInitializedRange(0u, sizeof(StageWork))) {
            const StageWork range{{rounds, 0u, 0u, 0u}};
            std::vector<uint8_t> bytes(sizeof(range));
            std::memcpy(bytes.data(), &range, sizeof(range));
            uploads.push_back({uniform, 0u, std::move(bytes)});
        }
        dispatches.push_back(Dispatch(Advance, 1u, uniform));
        return true;
    }

    JoltGpuControlWork PrepareJobs(size_t maximum = Workspace::DispatchesPerPacket)
    {
        if (!Allocate(frozen[Jobs], JobBytes()) || !AllocateScratch() ||
            !Allocate(frozen[Arguments], WordBytes, rhi::BufferUsageFlags::Indirect))
            return Retain({nullptr, [self = shared_from_this()] { return self->PrepareJobs(); }});

        std::vector<rhi::ComputeBufferUpdate> uploads;
        std::vector<rhi::ComputeDispatchDesc> dispatches;
        const auto appendRange = [&](Entry entry, const StageWork &range) {
            auto &uniform = workspace.ranges[dispatches.size()];
            if (!Allocate(uniform, sizeof(StageWork)))
                return false;
            std::vector<uint8_t> bytes(sizeof(StageWork));
            std::memcpy(bytes.data(), &range, sizeof(range));
            uploads.push_back({uniform, 0u, std::move(bytes)});
            const bool initializing = entry == Initialize;
            const uint32_t count = initializing ? std::max(range.range.x, range.range.y) : range.range.y;
            auto dispatch = Dispatch(entry, count, uniform, initializing ? &range : nullptr);
            if (entry == QueueConvex)
                dispatch.indirectArguments = contactLease->Dispatch(JoltGpuContactWorkset::Range::Pairs, range.range.x);
            dispatches.push_back(std::move(dispatch));
            return true;
        };
        if (phase == Phase::InitializeJobs) {
            const auto &jobs = frozen[Jobs];
            const uint32_t clearScratch = scratch->HasInitializedRange(0u, scratch->GetByteSize())
                                              ? 0u
                                              : static_cast<uint32_t>(scratch->GetByteSize() / WordBytes);
            const uint32_t clearJobs = jobs->HasInitializedRange(0u, jobs->GetByteSize())
                                           ? 1u
                                           : static_cast<uint32_t>(jobs->GetByteSize() / sizeof(uint32_t));
            if (!appendRange(Initialize, StageWork{{clearScratch, clearJobs, 0u, 0u}}))
                return Retain({nullptr, [self = shared_from_this()] { return self->PrepareJobs(); }});
        }

        uint32_t first = queueFirst;
        const uint32_t capacity = contactLease->patchSlots;
        const uint32_t chunkSize = contactLease->countedArguments ? contactLease->patchesPerDispatch : 2048u;
        while (first < capacity && dispatches.size() < maximum) {
            const uint32_t chunk = std::min(chunkSize, capacity - first);
            if (!appendRange(QueueConvex, StageWork{{first, chunk, 0u, 0u}}))
                break;
            first += chunk;
        }
        const bool publishes = first == capacity && dispatches.size() < maximum;
        if (publishes)
            dispatches.push_back(Dispatch(PublishJobs));
        const uint32_t rounds = AdvanceRounds(0u, AdvancesPerPacket);
        const auto started = std::chrono::steady_clock::now();
        const bool advances =
            publishes && dispatches.size() + 2u <= maximum && AppendAdvance(rounds, uploads, dispatches);
        std::vector<rhi::ComputeBufferRead> reads;
        JoltGpuContactTopologyStart::ObservePrefix observedTopology;
        if (advances) {
            dispatches.push_back(Dispatch(Progress));
            if (topology.appendAfterCollision)
                observedTopology = topology.appendAfterCollision(scratch, uploads, dispatches);
            reads.push_back({scratch, 0u, WordBytes});
        }
        if (dispatches.empty())
            return Retain({nullptr, [self = shared_from_this()] { return self->PrepareJobs(); }});

        const bool canSplit = dispatches.size() > 1;
        auto packet =
            rhi::PrepareComputeBatch(implementation->host, std::move(uploads), std::move(dispatches), std::move(reads));
        auto work = Retain(
            {packet,
             [self = shared_from_this(), packet, first, publishes, advances, rounds, started,
              observedTopology = std::move(observedTopology)] {
                 self->queueFirst = first;
                 self->phase = publishes ? Phase::AdvanceJobs : Phase::QueueJobs;
                 return advances ? self->ContinueProgress(*packet, rounds, started, 0u, observedTopology, {})
                                 : self->Next();
             },
             advances ? JoltGpuControlWork::ResumeAfter::Completion : JoltGpuControlWork::ResumeAfter::Acceptance});
        work.numericalWorkUnits = advances ? rounds * ConvexJobAdvanceQuantum : 0u;
        if (canSplit)
            work.prepareMinimum = [self = shared_from_this()] { return self->PrepareJobs(1u); };
        return work;
    }

    JoltGpuControlWork ContinueProgress(rhi::PreparedComputeBatch &packet, uint32_t rounds,
                                        std::chrono::steady_clock::time_point started, size_t costIndex,
                                        const JoltGpuContactTopologyStart::ObservePrefix &observedTopology,
                                        const SweepTail &tail)
    {
        UInt4 result{};
        packet.Readback(0u)->ReadInto(&result, sizeof(result));
        if (result.x == SweepCandidatesIncomplete)
            return retryCandidates();
        const bool linkedSolve = tail.linkedSolve && result.w == SweepSolverReceipt &&
                                 (result.x == QueryPending || result.x == QuerySuccess);
        if (linkedSolve)
            phase = Phase::SolveEvent;
        if (!linkedSolve &&
            (result.x == QueryPending || (swept ? !tail.Recorded() : result.y != 0u && !observedTopology))) {
            // Tick sizes the next admitted span from actual accepted work.
            // Completed packets which also consumed the result have mixed costs.
            const double observed = packet.GetCompletion().GetDeviceElapsedNanoseconds().value_or(
                std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - started).count());
            auto &estimate = swept ? implementation->advanceNanoseconds[costIndex] : workspace.advanceNanoseconds;
            estimate = std::max(observed / rounds, estimate * .75);
        }
        if (result.x == QueryPending) {
            if (observedTopology)
                observedTopology(false);
            return Next();
        }
        if (result.x != QuerySuccess) {
            if (swept && result.w == SweepUnsettled)
                throw std::runtime_error("Continuous impacts did not settle at the current time");
            if (swept && result.w == SweepMissingContact)
                throw std::runtime_error("GPU continuous cast reached its impact time without a contact");
            throw std::runtime_error(std::string(phase == Phase::SolveEvent ? "GPU sweep event solver"
                                                 : swept                    ? "GPU linear sweep"
                                                                            : "GPU convex narrowphase") +
                                     " failed at candidate " + std::to_string(result.z) + " (reason " +
                                     std::to_string(result.w) + ")");
        }
        if (phase == Phase::SolveEvent) {
            integratedBodies = tail.integratedBodies;
            phase = Phase::FinishEvent;
            return AcceptResolution(tail.motion.value_or(ResolveCursor{reportImpacts ? ReportEvent : ResumeMotion}));
        }
        if (swept && result.y != slots)
            throw std::runtime_error("GPU convex completion omitted an active job slot");
        if (phase == Phase::DiscoverEvent) {
            phase = Phase::LinkEvent;
            return AcceptResolution(tail.links ? *tail.links : EventLinks());
        }
        if (swept) {
            sweepOutcome = static_cast<SweepOutcome>(result.z);
            if (sweepOutcome != SweepOutcome::NoImpact) {
                reportedCandidates = result.w;
                phase = Phase::MoveEvent;
                return AcceptResolution(tail.motion.value_or(ResolveCursor{MoveToEvent}));
            }
            Finish(sweepOutcome, tail.integratedBodies);
            return {};
        }
        Finish();
        if (observedTopology)
            return observedTopology(true);
        return topology.prepare ? topology.prepare() : JoltGpuControlWork{};
    }

    ResolveCursor EventLinks() const
    {
        ResolveCursor cursor{jointCount ? JoinEventJoints : bodyCount > 1u ? FlattenRoots : LinkEvent};
        for (uint32_t links = bodyCount; links > 1u; links = 1u + (links - 1u) / 32u)
            ++cursor.parentPasses;
        if (jointCount)
            cursor.parentPasses = std::max(1u, cursor.parentPasses);
        return cursor;
    }

    void AppendResolution(ResolveCursor &next, uint32_t maximum, ResolutionDispatch mode,
                          std::vector<rhi::ComputeBufferUpdate> &uploads,
                          std::vector<rhi::ComputeDispatchDesc> &dispatches, size_t firstUniform = 0u)
    {
        auto &[entry, first, parentPasses] = next;
        for (uint32_t slot = 0; slot < maximum && entry != EntryCount; ++slot) {
            // The producer emits a command for the first bounded chunk of
            // each consumer. A wider range resumes from its accepted cursor.
            if (mode == ResolutionDispatch::Motion && first != 0u)
                break;
            auto &uniform = resolveUniforms[firstUniform + slot];
            if (!Allocate(uniform, sizeof(StageWork)))
                break;
            const bool patchDomain = entry == LinkEvent || entry == ReportEvent;
            const bool jointDomain = entry == JoinEventJoints || entry == LinkEventJoints;
            const uint32_t length = patchDomain ? contactLease->patchSlots : jointDomain ? jointCount : bodyCount;
            const uint32_t count = std::min(length - first, WorkItemLimit(Domain::Bodies));
            StageWork range{{first, count, 0u, 0u}};
            if (entry == FlattenRoots && parentPasses == 1u && jointCount)
                range.range.z = 1u;
            std::vector<uint8_t> bytes(sizeof(range));
            std::memcpy(bytes.data(), &range, sizeof(range));
            uploads.push_back({uniform, 0u, std::move(bytes)});
            auto dispatch = Dispatch(entry, entry == LinkEvent || jointDomain ? 1u : count, uniform);
            if (mode != ResolutionDispatch::Direct) {
                const auto record = mode == ResolutionDispatch::EventLinks
                                        ? (entry == FlattenRoots ? SweepFlattenDispatch : SweepLinkDispatch)
                                        : (entry == ReportEvent ? SweepReportDispatch : SweepMotionDispatch);
                dispatch.indirectArguments = rhi::ComputeIndirectArguments{frozen[Arguments], record * WordBytes};
            }
            dispatches.push_back(std::move(dispatch));
            first += count;
            if (first == length) {
                first = 0u;
                if (entry == MoveToEvent || entry == LinkEventJoints || entry == ResumeMotion)
                    entry = EntryCount;
                else if (entry == JoinEventJoints)
                    entry = parentPasses ? FlattenRoots : LinkEvent;
                else if (entry == LinkEvent)
                    entry = jointCount ? LinkEventJoints : EntryCount;
                else if (entry == ReportEvent)
                    entry = ResumeMotion;
                else if (entry == FlattenRoots) {
                    if (--parentPasses == 0u)
                        entry = contactLease->patchSlots ? LinkEvent : LinkEventJoints;
                }
            }
        }
    }

    JoltGpuControlWork AcceptResolution(ResolveCursor next)
    {
        if (next.entry != EntryCount) {
            resolution = next;
            return Resolve();
        }
        if (phase == Phase::MoveEvent) {
            phase = Phase::InitializeEvent;
            return Next();
        }
        if (phase == Phase::LinkEvent) {
            phase = Phase::SolveEvent;
            return Next();
        }
        Finish(sweepOutcome, integratedBodies);
        return {};
    }

    JoltGpuControlWork Resolve(uint32_t maximum = ResolveDispatchesPerPacket)
    {
        auto next = resolution;
        std::vector<rhi::ComputeBufferUpdate> uploads;
        std::vector<rhi::ComputeDispatchDesc> dispatches;
        AppendResolution(next, maximum, ResolutionDispatch::Direct, uploads, dispatches);
        if (dispatches.empty())
            return Retain({nullptr, [self = shared_from_this(), maximum] { return self->Resolve(maximum); }});
        const bool canSplit = dispatches.size() > 1;
        auto packet = rhi::PrepareComputeBatch(implementation->host, std::move(uploads), std::move(dispatches));
        auto work = Retain({packet, [self = shared_from_this(), next] { return self->AcceptResolution(next); },
                            JoltGpuControlWork::ResumeAfter::Acceptance});
        if (canSplit)
            work.prepareMinimum = [self = shared_from_this()] { return self->Resolve(1u); };
        return work;
    }

    void AppendSweepTail(SweepTail &result, bool solving, size_t firstUniform,
                         std::vector<rhi::ComputeBufferUpdate> &uploads,
                         std::vector<rhi::ComputeDispatchDesc> &dispatches)
    {
        ResolveCursor next{solving ? (reportImpacts ? ReportEvent : ResumeMotion) : MoveToEvent};
        const auto first = dispatches.size();
        const auto slots = static_cast<uint32_t>(
            std::min({size_t{2u}, MaxSweepDispatches - first, resolveUniforms.size() - firstUniform}));
        AppendResolution(next, slots, ResolutionDispatch::Motion, uploads, dispatches, firstUniform);
        if (dispatches.size() != first)
            result.motion = next;
        // An empty cast can integrate immediately. A last impact needs its
        // entire motion applied first and any requested report consumed.
        if (dispatches.size() < MaxSweepDispatches &&
            (!solving || (sweepOutcome == SweepOutcome::LastImpact && !reportImpacts && next.entry == EntryCount))) {
            const auto count = std::min(bodyCount, SweepMotionChunk);
            const auto base = stepInputs.GetParameters().selection.y;
            stagePrograms->AppendStages(storage, stepInputs, {{Stage::IntegrateSwept, {{base, count, 0u, 0u}}}},
                                        uploads, dispatches);
            dispatches.back().indirectArguments =
                rhi::ComputeIndirectArguments{frozen[Arguments], SweepIntegrateDispatch * WordBytes};
            result.integratedBodies = count;
        }
    }

    void AppendEventContactTail(SweepTail &result, std::vector<rhi::ComputeBufferUpdate> &uploads,
                                std::vector<rhi::ComputeDispatchDesc> &dispatches)
    {
        auto next = EventLinks();
        const auto first = dispatches.size();
        AppendResolution(next, ResolveDispatchesPerPacket, ResolutionDispatch::EventLinks, uploads, dispatches);
        const auto usedUniforms = dispatches.size() - first;
        if (usedUniforms == 0u)
            return;
        result.links = next;
        if (next.entry != EntryCount || dispatches.size() + 2u > MaxSweepDispatches)
            return;
        auto solve = Dispatch(jointCount ? SolveJointEvent : SolveEvent, bodyCount);
        solve.indirectArguments = rhi::ComputeIndirectArguments{frozen[Arguments], SweepSolveDispatch * WordBytes};
        dispatches.push_back(std::move(solve));
        dispatches.push_back(Dispatch(LinkedEventProgress));
        result.linkedSolve = true;
        AppendSweepTail(result, true, usedUniforms, uploads, dispatches);
    }

    JoltGpuControlWork Next(uint32_t maximumAdvances = AdvancesPerPacket)
    {
        if (phase == Phase::Done)
            return {};
        if (phase == Phase::MoveEvent || phase == Phase::LinkEvent || phase == Phase::FinishEvent)
            return Resolve();
        if (contactLease->patchSlots == 0u && phase == Phase::InitializeJobs) {
            Finish();
            return {};
        }
        if (!swept && phase != Phase::AdvanceJobs)
            return PrepareJobs();
        if (swept) {
            const uint64_t choiceWords = 5ull * bodyCount + 3u + contactLease->patchSlots + jointCount;
            if (frozen[SweepChoice] && frozen[SweepChoice]->GetByteSize() < choiceWords * sizeof(uint32_t))
                frozen[SweepChoice].reset();
            if (!Allocate(workUniform, sizeof(StageWork)) ||
                !Allocate(frozen[SweepChoice], choiceWords * sizeof(uint32_t)) ||
                !Allocate(frozen[SweepClock], WordBytes) ||
                !Allocate(frozen[Arguments], SweepArgumentWords * WordBytes, rhi::BufferUsageFlags::Indirect) ||
                !AllocateScratch())
                return Retain({nullptr, [self = shared_from_this()] { return self->Next(); }});
            retainChoices(frozen[SweepChoice]);
            retainClock(frozen[SweepClock]);
        }
        const bool eventInitialize = phase == Phase::InitializeEvent;
        const bool initializesSweep = phase == Phase::InitializeJobs || eventInitialize;
        auto preparedPhase = phase;
        std::vector<rhi::ComputeDispatchDesc> dispatches;
        std::vector<rhi::ComputeBufferUpdate> uploads;
        if (initializesSweep) {
            const uint32_t mode = eventInitialize ? SweepEventContacts : continuing ? SweepContinuation : SweepInitial;
            StageWork work{{0u, contactLease->patchSlots, mode, uint32_t(reportImpacts)}};
            std::vector<uint8_t> bytes(sizeof(work));
            std::memcpy(bytes.data(), &work, sizeof(work));
            uploads.push_back({workUniform, 0u, std::move(bytes)});
            dispatches.push_back(Dispatch(InitializeSweep));
            preparedPhase = eventInitialize ? Phase::DiscoverEvent : Phase::AdvanceJobs;
            if (maximumAdvances == 1u) {
                auto packet = rhi::PrepareComputeBatch(implementation->host, std::move(uploads), std::move(dispatches));
                return Retain({packet,
                               [self = shared_from_this(), preparedPhase] {
                                   self->phase = preparedPhase;
                                   return self->Next();
                               },
                               JoltGpuControlWork::ResumeAfter::Acceptance});
            }
        }
        // Each invocation advances a fixed number of real support/EPA/clip
        // operations. The persisted cursor, not Host retry count, determines
        // remaining work. Only this small completion record crosses to CPU.
        const bool solvingEvent = preparedPhase == Phase::SolveEvent;
        const bool findingContacts = preparedPhase == Phase::DiscoverEvent;
        const size_t costIndex = solvingEvent ? 3u : findingContacts ? 2u : size_t(swept);
        const uint32_t count = AdvanceRounds(costIndex, maximumAdvances);
        const auto started = std::chrono::steady_clock::now();
        if (swept) {
            const auto step = solvingEvent      ? (jointCount ? SolveJointEvent : SolveEvent)
                              : findingContacts ? DiscoverEvent
                                                : AdvanceSweeps;
            for (uint32_t i = 0; i < count; ++i)
                dispatches.push_back(Dispatch(step, solvingEvent ? bodyCount : 1u));
        } else if (!AppendAdvance(count, uploads, dispatches)) {
            return Retain(
                {nullptr, [self = shared_from_this(), maximumAdvances] { return self->Next(maximumAdvances); }});
        }
        dispatches.push_back(Dispatch(solvingEvent      ? EventProgress
                                      : findingContacts ? EventContactProgress
                                      : swept           ? SweepProgress
                                                        : Progress));
        JoltGpuContactTopologyStart::ObservePrefix observedTopology;
        if (maximumAdvances > 1u && topology.appendAfterCollision)
            observedTopology = topology.appendAfterCollision(scratch, uploads, dispatches);
        const bool hasTopology = bool(observedTopology);
        SweepTail tail;
        if (swept && maximumAdvances > 1u) {
            if (findingContacts)
                AppendEventContactTail(tail, uploads, dispatches);
            else
                AppendSweepTail(tail, solvingEvent, 0u, uploads, dispatches);
        }
        auto packet = rhi::PrepareComputeBatch(implementation->host, std::move(uploads), std::move(dispatches),
                                               {{scratch, 0u, WordBytes}});
        auto work =
            Retain({packet, [self = shared_from_this(), packet, started, count, costIndex, preparedPhase,
                             observedTopology = std::move(observedTopology), tail] {
                        self->phase = preparedPhase;
                        return self->ContinueProgress(*packet, count, started, costIndex, observedTopology, tail);
                    }});
        work.numericalWorkUnits = swept ? 0u : count * ConvexJobAdvanceQuantum;
        if (initializesSweep || count > 1u || hasTopology || tail.Recorded())
            work.prepareMinimum = [self = shared_from_this()] { return self->Next(1u); };
        return work;
    }

    std::shared_ptr<Impl> implementation;
    const JoltGpuStorage &storage;
    const JoltGpuStepInputs &stepInputs;
    JoltGpuPrograms *stagePrograms;
    std::shared_ptr<const JoltGpuContactWorkset> contactLease;
    std::shared_ptr<const JoltGpuGeometryLease> geometry;
    JoltGpuContactTopologyStart topology;
    FrameWorkOwner &budget;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> frozen;
    std::shared_ptr<rhi::ComputeBuffer> scratch, workUniform;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, ResolveDispatchesPerPacket + 2u> resolveUniforms;
    std::function<void(JoltGpuSweepResult)> completed;
    std::function<JoltGpuControlWork()> retryCandidates;
    std::function<void(const std::shared_ptr<rhi::ComputeBuffer> &)> retainChoices, retainClock;
    const bool swept;
    const bool continuing;
    const bool reportImpacts;
    Workspace *resident;
    Workspace workspace;
    uint32_t queueFirst = 0u;
    uint32_t integratedBodies = 0u, reportedCandidates = 0u;
    SweepOutcome sweepOutcome = SweepOutcome::NoImpact;
    Phase phase = Phase::InitializeJobs;
    ResolveCursor resolution;
    uint32_t slots = 0u, scratchWords = 0u, bodyCount = 0u, jointCount = 0u;
};

JoltGpuNarrowphase::JoltGpuNarrowphase(rhi::ComputeHost &host, const ShaderTargetCode &target)
    : m_impl(std::make_shared<Impl>(host, target))
{
}

JoltGpuControlWork JoltGpuNarrowphase::Prepare(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                                               FrameWorkOwner &budget, Workspace &resident,
                                               JoltGpuContactTopologyStart topology, std::function<void()> completed)
{
    auto execution = std::make_shared<Impl::Execution>(
        m_impl, storage, inputs, budget, [done = std::move(completed)](JoltGpuSweepResult) { done(); }, nullptr, false,
        false, &resident);
    execution->topology = std::move(topology);
    return execution->Next();
}

JoltGpuControlWork JoltGpuNarrowphase::PrepareSweeps(const JoltGpuStorage &storage, JoltGpuStepInputs &inputs,
                                                     JoltGpuPrograms &programs, FrameWorkOwner &budget, bool continuing,
                                                     bool reportImpacts,
                                                     std::function<JoltGpuControlWork()> retryCandidates,
                                                     std::function<void(JoltGpuSweepResult)> completed)
{
    auto execution = std::make_shared<Impl::Execution>(m_impl, storage, inputs, budget, std::move(completed), &programs,
                                                       continuing, reportImpacts);
    execution->retryCandidates = std::move(retryCandidates);
    execution->retainChoices = [&inputs](const auto &choices) { inputs.SetSweepChoices(choices); };
    execution->retainClock = [&inputs](const auto &clock) { inputs.SetSweepClock(clock); };
    return execution->Next();
}

} // namespace infernux
