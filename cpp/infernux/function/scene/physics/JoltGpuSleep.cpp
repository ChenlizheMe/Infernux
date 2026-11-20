#include "JoltGpuSleep.h"
#include <Jolt/GPU/ConstraintTypes.h>
#include "JoltGpuProgramSources.h"
#include "JoltGpuSleepLayout.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

namespace infernux
{
namespace
{
using namespace JPH::GPU;
enum Port : uint8_t
{
    Params,
    Work,
    Models,
    States,
    Forces,
    Sleeps,
    Shapes,
    Patches,
    Joints,
    Schedule,
    Roots,
    Bounds,
    Snapshot,
    Requests,
    Results,
    PortCount
};
enum Entry : uint8_t
{
    Clear,
    SeedBodies,
    SeedJoints,
    SeedPatches,
    Wake,
    ForceWake,
    Update,
    Commit,
    RetireHinge,
    CaptureBounds,
    BoundsWake,
    Export,
    Query,
    PrescribedCapture,
    PrescribedWake,
    NewBoundsWake,
    EntryCount
};
constexpr std::array<std::string_view, EntryCount> Names{"sleep_clear_votes",
                                                         "sleep_seed_awake_bodies",
                                                         "sleep_seed_joint_endpoints",
                                                         "sleep_seed_patch_endpoints",
                                                         "sleep_wake_components",
                                                         "sleep_wake_force_input",
                                                         "sleep_update_history",
                                                         "sleep_commit_components",
                                                         "sleep_retire_hinge_momentum",
                                                         "sleep_capture_old_bounds",
                                                         "sleep_wake_bounds",
                                                         "sleep_export_status",
                                                         "query_sleep_status",
                                                         "sleep_capture_prescribed_bounds",
                                                         "sleep_wake_prescribed_bounds",
                                                         "sleep_wake_new_bounds"};
constexpr uint32_t Bit(Port p)
{
    return 1u << p;
}
constexpr uint32_t P = Bit(Params), W = Bit(Work), B = Bit(Models), S = Bit(Sleeps), R = Bit(Roots), C = Bit(Schedule);
constexpr std::array<uint32_t, EntryCount> Masks{P | W | R,
                                                 P | W | B | S | R | C,
                                                 P | W | B | S | R | C | Bit(Joints),
                                                 P | W | B | S | R | C | Bit(Patches) | Bit(Shapes),
                                                 P | W | B | S | R | C | Bit(States),
                                                 P | W | B | S | Bit(States) | Bit(Forces),
                                                 P | W | B | S | R | C | Bit(States) | Bit(Forces),
                                                 P | W | B | S | R | C | Bit(States),
                                                 P | W | B | S | R | Bit(Joints),
                                                 P | B | S | R | Bit(States) | Bit(Bounds),
                                                 P | W | B | S | R | Bit(States) | Bit(Bounds),
                                                 P | W | B | S | Bit(Snapshot),
                                                 P | W | R | Bit(Snapshot) | Bit(Requests) | Bit(Results),
                                                 P | W | B | S | R | Bit(States) | Bit(Bounds) | Bit(Requests),
                                                 P | W | B | S | R | Bit(States) | Bit(Bounds) | Bit(Requests),
                                                 P | W | B | S | R | Bit(States) | Bit(Requests)};
struct Layout
{
    std::string_view name;
    uint32_t stride;
    uint32_t integers;
    bool readOnly;
};
constexpr std::array<Layout, PortCount> Layouts{{{"sleep_params", sizeof(SleepWorkParameters), 14u, true},
                                                 {"stage_work", sizeof(StageWork), 1u, true},
                                                 {"body_models", sizeof(BodyModel), 1u, true},
                                                 {"body_states", sizeof(BodyState), 0u, false},
                                                 {"body_forces", sizeof(BodyForces), 0u, false},
                                                 {"body_sleep", sizeof(BodySleepRecord), 0u, false},
                                                 {"shape_models", sizeof(ShapeModel), 225u, true},
                                                 {"contact_patches", sizeof(ContactPatch), 1u, false},
                                                 {"joint_stream", 16u, 1u, false},
                                                 {"constraint_schedule", 16u, 1u, true},
                                                 {"body_sleep_roots", 4u, 1u, false},
                                                 {"sleep_aabbs", 16u, 0u, false},
                                                 {"sleep_snapshot", 16u, 1u, false},
                                                 {"sleep_requests", 4u, 1u, true},
                                                 {"sleep_results", 16u, 1u, false}}};
constexpr std::array<std::array<std::string_view, 8>, PortCount> Fields{
    {{"timing", "selected", "layout", "operation"},
     {"range"},
     {"meta", "mass", "invInertiaDiagonal", "inertiaRotation", "damping"},
     {"positionCOM", "rotation", "linearVelocity", "angularVelocity"},
     {"force", "torque"},
     {"author", "runtime"},
     {"meta", "geometry", "localPositionCOM", "localRotation", "material", "surface", "source", "geometryReference"},
     {"leaves"},
     {"value"},
     {"value"},
     {},
     {},
     {},
     {},
     {}}};
template <class T> std::vector<uint8_t> Bytes(const T &v)
{
    std::vector<uint8_t> b(sizeof(T));
    std::memcpy(b.data(), &v, sizeof(T));
    return b;
}
uint32_t Groups(uint32_t n)
{
    return n / 64u + uint32_t(n % 64u != 0);
}
void Validate(Port p, const ShaderResourceLayout &r)
{
    const auto &layout = Layouts[p];
    bool uniform = p == Params || p == Work;
    if (r.set != 0 || r.binding.type != (uniform ? rhi::BindingType::UniformBuffer : rhi::BindingType::StorageBuffer) ||
        (uniform ? r.value.byteSize : r.elementStride) != layout.stride ||
        (!uniform && r.binding.readOnlyStorage != layout.readOnly))
        throw std::invalid_argument("GPU sleep resource differs from its shared ABI: " + std::string(layout.name));
    if (p == Sleeps) {
        ValidateJoltGpuSleepLayout(r.value);
        return;
    }
    if (p == Roots || p == Requests) {
        if (r.value.kind != ShaderValueKind::Scalar || r.value.scalar != ShaderScalarKind::UInt32 ||
            r.value.byteSize != 4u)
            throw std::invalid_argument("GPU sleep scalar storage differs from uint32");
        return;
    }
    if (p == Bounds || p == Snapshot || p == Results) {
        if (r.value.kind != ShaderValueKind::Vector || r.value.columns != 4u || r.value.rows != 1u ||
            r.value.scalar != (p == Bounds ? ShaderScalarKind::Float32 : ShaderScalarKind::UInt32))
            throw std::invalid_argument("GPU sleep vector storage differs from shared ABI");
        return;
    }
    if (r.value.kind != ShaderValueKind::Struct || r.value.fields.size() != layout.stride / 16u)
        throw std::invalid_argument("GPU sleep struct field count differs from shared ABI");
    for (size_t i = 0; i < r.value.fields.size(); ++i) {
        const auto &f = r.value.fields[i];
        if (f.name != Fields[p][i] || f.offset != i * 16u || f.byteSize != 16u || f.kind != ShaderValueKind::Vector ||
            f.rows != 1u || f.columns != 4u ||
            f.scalar != ((layout.integers & (1u << i)) ? ShaderScalarKind::UInt32 : ShaderScalarKind::Float32))
            throw std::invalid_argument("GPU sleep struct field differs from shared ABI");
    }
}
SleepWorkParameters Selection(const JoltGpuStorage &storage, const StepParams &p)
{
    SleepWorkParameters result{};
    result.selected = {p.selection.x, p.selection.y, p.selectedCounts.x, storage.GetBodyCount()};
    result.operation.x = UINT32_MAX;
    if (result.selected.y > result.selected.w || result.selected.z > result.selected.w - result.selected.y)
        throw std::invalid_argument("GPU sleep selection exceeds accepted body slots");
    return result;
}
} // namespace

class JoltGpuSleep::Impl
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
            throw std::invalid_argument("GPU sleep requires the shared SPIR-V/WGSL artifact");
        for (uint32_t i = 0; i < EntryCount; ++i) {
            auto code = std::find_if(target.entries.begin(), target.entries.end(),
                                     [&](const auto &e) { return e.name == Names[i]; });
            if (code == target.entries.end() || code->stage != rhi::ShaderStage::Compute ||
                code->workgroupSize != std::array<uint32_t, 3>{64, 1, 1})
                throw std::invalid_argument("GPU sleep artifact is missing its bounded compute entry");
            auto &program = entries[i];
            uint32_t mask = 0;
            std::vector<rhi::ComputeBufferBinding> bindings;
            for (const auto &r : target.resources) {
                std::array<uint32_t, 2> key{r.set, r.binding.binding};
                if (std::find(code->bindings.begin(), code->bindings.end(), key) == code->bindings.end())
                    continue;
                auto found =
                    std::find_if(Layouts.begin(), Layouts.end(), [&](const auto &v) { return v.name == r.name; });
                if (found == Layouts.end())
                    throw std::invalid_argument("GPU sleep entry has unknown resource");
                Port port = static_cast<Port>(found - Layouts.begin());
                Validate(port, r);
                if (mask & Bit(port))
                    throw std::invalid_argument("GPU sleep entry repeats resource");
                mask |= Bit(port);
                program.ports.push_back(port);
                bindings.push_back({r.binding.binding, r.binding.type, r.binding.readOnlyStorage});
            }
            if (mask != Masks[i] || code->bindings.size() != program.ports.size())
                throw std::invalid_argument("GPU sleep entry differs from its actual resource mask: " +
                                            std::string(Names[i]));
            program.kernel = PrepareJoltGpuKernel(host, target.target, *code, std::move(bindings));
        }
    }
    rhi::ComputeHost host;
    std::array<Program, EntryCount> entries;
};

struct JoltGpuSleep::Impl::Execution : std::enable_shared_from_this<Execution>
{
    struct Operation
    {
        Entry entry;
        uint32_t first, count;
        SleepWorkParameters parameters;
    };
    explicit Execution(std::shared_ptr<Impl> owner) : owner(std::move(owner))
    {
    }
    void Bind(const JoltGpuStorage &storage)
    {
        if (!owner->host.SharesServicesWith(storage.GetHost()))
            throw std::invalid_argument("GPU sleep crosses compute owner");
        buffers[Models] = storage.Get(Resource::BodyModels);
        buffers[States] = storage.Get(Resource::BodyStates);
        buffers[Forces] = storage.Get(Resource::BodyForces);
        buffers[Sleeps] = storage.Get(Resource::SleepStates);
        buffers[Joints] = storage.Get(Resource::Joints);
    }
    void Add(Entry entry, uint32_t first, uint32_t count, const SleepWorkParameters &p)
    {
        if (count)
            plan.push_back({entry, first, count, p});
    }
    void SetRequests(const std::vector<uint32_t> &requests)
    {
        requestBytes.resize(requests.size() * sizeof(uint32_t));
        if (!requests.empty())
            std::memcpy(requestBytes.data(), requests.data(), requestBytes.size());
    }
    template <class F> void OnComplete(F done)
    {
        completed = [done = std::move(done)](Execution &work) {
            done(work);
            return JoltGpuControlWork{};
        };
    }
    bool Allocate(std::shared_ptr<rhi::ComputeBuffer> &buffer, uint64_t bytes)
    {
        if (!bytes || bytes % 4u || bytes > owner->host.device.GetCapabilities().portable.maxStorageBufferBinding)
            throw std::length_error("GPU sleep control allocation exceeds the actual device binding range");
        if (!buffer)
            buffer = rhi::ComputeBuffer::TryCreatePrivate(
                owner->host, {bytes / 4u, rhi::ComputeScalarType::UInt32, 1u, rhi::BufferUsageFlags::None});
        return buffer != nullptr;
    }

    static void ReadStatus(rhi::PreparedComputeBatch &packet)
    {
        uint32_t status = 0;
        packet.Readback(0)->ReadInto(&status, sizeof(status));
        if (status & ConstraintNotReady)
            throw std::runtime_error("GPU constraint topology did not validate (status " +
                                     std::to_string(status & ~ConstraintNotReady) + ")");
        if (status)
            throw std::runtime_error("GPU sleep failed state/root validation (status " + std::to_string(status) + ")");
    }

    rhi::ComputeDispatchDesc Dispatch(const Operation &op, uint32_t offset, uint32_t count,
                                      const std::shared_ptr<rhi::ComputeBuffer> &params,
                                      const std::shared_ptr<rhi::ComputeBuffer> &work) const
    {
        rhi::ComputeDispatchDesc dispatch;
        dispatch.kernel = owner->entries[op.entry].kernel;
        dispatch.groupCountX = Groups(count);
        if (wakeArguments)
            dispatch.indirectArguments = rhi::ComputeIndirectArguments{
                wakeArguments, uint64_t(dispatch.groupCountX - 1u) * sizeof(ConstraintScheduleWord)};
        rhi::ComputeDispatchProof proof;
        for (Port port : owner->entries[op.entry].ports) {
            auto buffer = port == Params ? params : port == Work ? work : buffers[port];
            if (!buffer)
                throw std::logic_error("GPU sleep dispatch lacks its exact resource");
            uint32_t binding = static_cast<uint32_t>(dispatch.buffers.size());
            dispatch.buffers.push_back(buffer);
            const bool complete = (port == Bounds && (op.entry == CaptureBounds || op.entry == PrescribedCapture)) ||
                                  (port == Snapshot && op.entry == Export) || port == Results ||
                                  (port == Roots && op.entry == Clear);
            const bool writer = port == Roots ||
                                (port == Sleeps && (op.entry == Wake || op.entry == ForceWake || op.entry == Update ||
                                                    op.entry == Commit || op.entry == BoundsWake ||
                                                    op.entry == PrescribedWake || op.entry == NewBoundsWake)) ||
                                (port == Forces && op.entry == Update) || (port == States && op.entry == Commit) ||
                                (port == Joints && op.entry == RetireHinge);
            dispatch.bufferAccesses.push_back(complete ? rhi::ComputeBufferAccess::Write
                                              : writer ? rhi::ComputeBufferAccess::ReadWrite
                                                       : rhi::ComputeBufferAccess::Read);
            if (complete) {
                uint64_t first = op.first + offset;
                if (port == Roots) {
                    first -= op.parameters.selected.y;
                    proof.writes.push_back({binding, first * 4u, uint64_t(count) * 4u});
                    if (first == 0u)
                        proof.writes.push_back({binding, uint64_t(statusIndex) * 4u, 4u});
                } else if (port == Bounds) {
                    if (op.entry == PrescribedCapture)
                        proof.writes.push_back({binding, first * 64u, uint64_t(count) * 64u});
                    else
                        proof.writes.push_back({binding, 0u, 64u});
                } else {
                    if (port == Snapshot)
                        first -= op.parameters.selected.y;
                    proof.writes.push_back({binding, first * 16u, uint64_t(count) * 16u});
                }
            } else
                proof.reads.push_back({binding, 0u, buffer->GetByteSize()});
        }
        dispatch.proof = std::move(proof);
        return dispatch;
    }

    JoltGpuControlWork Next(uint32_t maximumDispatches = 16u)
    {
        if (cursor == plan.size()) {
            auto callback = std::move(completed);
            return callback(*this);
        }
        if (reportsStatus && !buffers[Roots]) {
            auto &status = boundsValidation ? boundsValidation->status : buffers[Roots];
            if (!Allocate(status, sizeof(uint32_t)))
                return {nullptr, [self = shared_from_this()] { return self->Next(); }};
            buffers[Roots] = status;
        }
        if (resultBytes && !Allocate(buffers[resultPort], resultBytes))
            return {nullptr, [self = shared_from_this()] { return self->Next(); }};
        if (!requestBytes.empty() && !Allocate(buffers[Requests], requestBytes.size()))
            return {nullptr, [self = shared_from_this()] { return self->Next(); }};
        size_t next = cursor;
        uint32_t offset = cursorOffset;
        std::vector<rhi::ComputeBufferUpdate> uploads;
        std::vector<rhi::ComputeDispatchDesc> dispatches;
        if (!started) {
            if (reportsStatus && initializeStatus &&
                (!boundsValidation || !buffers[Roots]->HasInitializedRange(0u, sizeof(uint32_t))))
                uploads.push_back({buffers[Roots], uint64_t(statusIndex) * 4u, Bytes(uint32_t(0))});
            if (!requestBytes.empty())
                uploads.push_back({buffers[Requests], 0u, requestBytes});
            if (!boundsBytes.empty())
                uploads.push_back({buffers[Bounds], 0u, boundsBytes});
        }
        const size_t firstUniform = uploads.size();
        const auto uniform = [&](const auto &value) -> std::shared_ptr<rhi::ComputeBuffer> {
            auto bytes = Bytes(value);
            for (size_t i = firstUniform; i < uploads.size(); ++i)
                if (uploads[i].data == bytes)
                    return uploads[i].buffer;
            auto buffer = rhi::ComputeBuffer::TryCreatePrivate(
                owner->host, {sizeof(value) / 4u, rhi::ComputeScalarType::UInt32, 1u, rhi::BufferUsageFlags::None});
            if (buffer)
                uploads.push_back({buffer, 0u, std::move(bytes)});
            return buffer;
        };
        while (next < plan.size() && dispatches.size() < maximumDispatches) {
            const auto &op = plan[next];
            uint32_t count = std::min(JoltGpuPoseWake::BodiesPerDispatch, op.count - offset);
            auto params = uniform(op.parameters);
            StageWork range{{op.first + offset, count, 0u, 0u}};
            auto work = uniform(range);
            if (!params || !work)
                break;
            dispatches.push_back(Dispatch(op, offset, count, params, work));
            offset += count;
            if (offset == op.count) {
                offset = 0u;
                ++next;
            }
        }
        if (dispatches.empty())
            return {nullptr, [self = shared_from_this()] { return self->Next(); }};
        bool final = next == plan.size();
        const bool readStatus = final && reportsStatus && !boundsValidation;
        const bool deferStatus = readStatus && !query && bool(deferValidation);
        std::vector<rhi::ComputeBufferRead> reads;
        if (readStatus) {
            reads.push_back({buffers[Roots], uint64_t(statusIndex) * 4u, 4u});
            if (query)
                reads.push_back({buffers[Results], 0u, resultBytes});
        }
        const bool multi = dispatches.size() > 1u;
        auto packet =
            rhi::PrepareComputeBatch(owner->host, std::move(uploads), std::move(dispatches), std::move(reads));
        JoltGpuControlWork work{
            packet,
            [self = shared_from_this(), packet, next, offset, readStatus, deferStatus]() -> JoltGpuControlWork {
                self->started = true;
                self->cursor = next;
                self->cursorOffset = offset;
                if (readStatus) {
                    if (deferStatus)
                        self->deferValidation(packet, [packet] { ReadStatus(*packet); });
                    else
                        ReadStatus(*packet);
                    if (self->query) {
                        self->queryResults.resize(self->resultBytes / sizeof(UInt4));
                        packet->Readback(1)->ReadInto(self->queryResults.data(), self->resultBytes);
                    }
                }
                return self->Next();
            },
            deferStatus || !final ? JoltGpuControlWork::ResumeAfter::Acceptance : finalResumeAfter};
        if (multi)
            work.prepareMinimum = [self = shared_from_this()] { return self->Next(1u); };
        return work;
    }
    std::shared_ptr<Impl> owner;
    std::shared_ptr<rhi::ComputeBuffer> wakeArguments;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> buffers;
    std::shared_ptr<BoundsValidation> boundsValidation;
    std::vector<Operation> plan;
    size_t cursor = 0;
    uint32_t cursorOffset = 0, statusIndex = 0;
    bool initializeStatus = true, started = false, query = false, reportsStatus = true;
    JoltGpuControlWork::ResumeAfter finalResumeAfter = JoltGpuControlWork::ResumeAfter::Completion;
    Port resultPort = Results;
    uint64_t resultBytes = 0;
    std::vector<uint8_t> requestBytes, boundsBytes;
    std::vector<UInt4> queryResults;
    std::function<JoltGpuControlWork(Execution &)> completed;
    DeferredValidation deferValidation;
};

JoltGpuSleep::JoltGpuSleep(rhi::ComputeHost &host, const ShaderTargetCode &target)
    : m_impl(std::make_shared<Impl>(host, target))
{
}

bool JoltGpuSleep::TryPrepareStartupPrograms()
{
    bool ready = true;
    for (const auto entry : {Clear, Update, Commit, Export})
        ready &= m_impl->entries[entry].kernel->TryPreparePipeline();
    return ready;
}

JoltGpuControlWork JoltGpuSleep::PrepareWake(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                                             const JoltGpuConstraintTopology &topology, std::function<void()> done,
                                             DeferredValidation deferValidation)
{
    const auto &selected = inputs.GetParameters();
    const auto &contacts = inputs.GetContacts();
    auto p = Selection(storage, selected);
    if (!topology.schedule || !topology.rootVotes || topology.bodyFirst != p.selected.y ||
        topology.bodyCount != p.selected.z || !contacts)
        throw std::logic_error("GPU sleep wake requires its accepted substep topology");
    if (selected.selectedCounts.z == 0 && contacts->patchCount == 0) {
        // Without edges, validated Dynamic roots are their own bodies. An
        // awake body votes only for itself, and an asleep body gets no vote.
        done();
        return {};
    }
    auto e = std::make_shared<Impl::Execution>(m_impl);
    e->Bind(storage);
    e->buffers[Schedule] = topology.schedule;
    e->deferValidation = std::move(deferValidation);
    e->buffers[Roots] = topology.rootVotes;
    e->statusIndex = p.selected.z;
    p.layout = {topology.bodyRootFirst, e->statusIndex, 0u, 0u};
    e->initializeStatus = false;
    e->buffers[Shapes] = inputs.GetCaptured(Resource::ShapeModels);
    e->buffers[Patches] = contacts->patches;
    e->wakeArguments = topology.wakeArguments;
    if (!e->wakeArguments)
        e->Add(Clear, p.selected.y, p.selected.z, p);
    e->Add(SeedBodies, p.selected.y, p.selected.z, p);
    p.layout.z = selected.selection.w;
    p.layout.w = selected.selectedCounts.z;
    e->Add(SeedJoints, p.layout.z, p.layout.w, p);
    p.layout.z = 0u;
    p.layout.w = contacts->patchCount;
    e->Add(SeedPatches, 0u, p.layout.w, p);
    e->Add(Wake, p.selected.y, p.selected.z, p);
    e->OnComplete([done = std::move(done)](auto &) { done(); });
    return e->Next();
}
JoltGpuControlWork JoltGpuSleep::PrepareHistory(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                                                const BodySleepParameters &settings, std::function<void()> done,
                                                DeferredValidation deferValidation)
{
    if (!std::isfinite(settings.timing.x) || settings.timing.x <= 0 || !std::isfinite(settings.timing.y) ||
        settings.timing.y < 0 || !std::isfinite(settings.timing.z) || settings.timing.z <= 0 || settings.timing.w != 0)
        throw std::invalid_argument("GPU sleep needs finite physical time and configured thresholds");
    auto e = std::make_shared<Impl::Execution>(m_impl);
    e->Bind(storage);
    auto p = Selection(storage, inputs.GetParameters());
    auto plan = inputs.GetConstraints();
    if (!plan || plan->bodyFirst != p.selected.y || plan->bodyCount != p.selected.z)
        throw std::logic_error("GPU sleep history requires its completed numerical plan");
    e->buffers[Schedule] = plan->schedule;
    e->buffers[Roots] = plan->rootVotes;
    e->statusIndex = p.selected.z;
    e->initializeStatus = false;
    p.layout = {plan->bodyRootFirst, e->statusIndex, 0u, 0u};
    p.timing = settings.timing;
    p.operation.z = 1u;
    e->Add(Clear, p.selected.y, p.selected.z, p);
    e->Add(Update, p.selected.y, p.selected.z, p);
    const auto &step = inputs.GetParameters();
    e->deferValidation = std::move(deferValidation);
    e->Add(Commit, p.selected.y, p.selected.z, p);
    p.layout.z = step.selection.w;
    p.layout.w = step.selectedCounts.z;
    e->Add(RetireHinge, p.layout.z, p.layout.w, p);
    e->OnComplete([done = std::move(done)](auto &) { done(); });
    return e->Next();
}
JoltGpuControlWork JoltGpuSleep::PrepareForceWake(const JoltGpuStorage &s, const StepParams &selection,
                                                  std::function<void()> done, DeferredValidation deferValidation)
{
    auto e = std::make_shared<Impl::Execution>(m_impl);
    e->Bind(s);
    auto p = Selection(s, selection);
    e->Add(ForceWake, p.selected.y, p.selected.z, p);
    e->deferValidation = std::move(deferValidation);
    e->OnComplete([done = std::move(done)](auto &) { done(); });
    return e->Next();
}
JoltGpuControlWork JoltGpuSleep::PreparePrescribedCapture(
    const JoltGpuStorage &s, const StepParams &selection, std::vector<uint32_t> sources,
    std::function<void(std::shared_ptr<const JoltGpuPrescribedSleepBounds>)> done, DeferredValidation deferValidation)
{
    if (sources.empty()) {
        done({});
        return {};
    }
    if (sources.size() > UINT32_MAX)
        throw std::length_error("Prescribed source directory exceeds uint32");
    auto e = std::make_shared<Impl::Execution>(m_impl);
    e->Bind(s);
    auto p = Selection(s, selection);
    p.operation.w = static_cast<uint32_t>(sources.size());
    for (auto source : sources)
        if (source < p.selected.y || source - p.selected.y >= p.selected.z)
            throw std::out_of_range("Prescribed source crosses selected World slots");
    e->SetRequests(sources);
    e->resultPort = Bounds;
    e->resultBytes = uint64_t(sources.size()) * 64u;
    e->Add(PrescribedCapture, 0u, p.operation.w, p);
    e->deferValidation = std::move(deferValidation);
    e->OnComplete([done = std::move(done), count = p.operation.w](auto &self) {
        auto result = std::make_shared<JoltGpuPrescribedSleepBounds>();
        result->count = count;
        result->sources = self.buffers[Requests]->Freeze();
        result->bounds = self.buffers[Bounds]->Freeze();
        done(std::move(result));
    });
    return e->Next();
}
JoltGpuControlWork JoltGpuSleep::PreparePrescribedWake(const JoltGpuStorage &s, const StepParams &selection,
                                                       const JoltGpuPrescribedSleepBounds &old,
                                                       std::function<void()> done, DeferredValidation deferValidation)
{
    if (!old.count || !old.sources || !old.bounds || old.sources->GetByteSize() != uint64_t(old.count) * 4u ||
        old.bounds->GetByteSize() != uint64_t(old.count) * 64u ||
        !old.sources->HasInitializedRange(0u, old.sources->GetByteSize()) ||
        !old.bounds->HasInitializedRange(0u, old.bounds->GetByteSize()))
        throw std::invalid_argument("Prescribed wake lost its completed exact source capture");
    auto e = std::make_shared<Impl::Execution>(m_impl);
    e->Bind(s);
    auto p = Selection(s, selection);
    e->buffers[Requests] = old.sources;
    e->deferValidation = std::move(deferValidation);
    e->buffers[Bounds] = old.bounds;
    p.operation.w = old.count;
    for (uint32_t source = 0; source < old.count; ++source) {
        p.layout.z = source;
        e->Add(PrescribedWake, p.selected.y, p.selected.z, p);
    }
    e->OnComplete([done = std::move(done)](auto &) { done(); });
    return e->Next();
}
JoltGpuControlWork JoltGpuSleep::PrepareSnapshot(const JoltGpuStorage &s, const StepParams &selection,
                                                 std::function<void(std::shared_ptr<rhi::ComputeBuffer>)> done)
{
    auto e = std::make_shared<Impl::Execution>(m_impl);
    e->Bind(s);
    auto p = Selection(s, selection);
    e->resultPort = Snapshot;
    e->resultBytes = uint64_t(p.selected.z) * 16u;
    // Export has no CPU status consumer. Its successor freezes the accepted
    // GPU destination; the enclosing Tick retains the publication receipts.
    e->reportsStatus = false;
    e->finalResumeAfter = JoltGpuControlWork::ResumeAfter::Acceptance;
    e->Add(Export, p.selected.y, p.selected.z, p);
    e->OnComplete([done = std::move(done)](auto &self) {
        done(self.buffers[Snapshot] ? self.buffers[Snapshot]->Freeze() : nullptr);
    });
    return e->Next();
}
JoltGpuControlWork JoltGpuSleep::PrepareOldBounds(const JoltGpuStorage &s, const StepParams &selection, uint32_t source,
                                                  std::shared_ptr<BoundsValidation> validation,
                                                  std::function<void(std::shared_ptr<rhi::ComputeBuffer>)> done)
{
    auto e = std::make_shared<Impl::Execution>(m_impl);
    e->Bind(s);
    e->boundsValidation = std::move(validation);
    e->finalResumeAfter = JoltGpuControlWork::ResumeAfter::Acceptance;
    auto p = Selection(s, selection);
    p.operation.x = source;
    e->resultPort = Bounds;
    e->resultBytes = 64u;
    e->Add(CaptureBounds, 0u, 1u, p);
    e->OnComplete([done = std::move(done)](auto &self) { done(self.buffers[Bounds]->Freeze()); });
    return e->Next();
}
JoltGpuControlWork JoltGpuSleep::PrepareBoundsWake(const JoltGpuStorage &s, const StepParams &selection,
                                                   std::shared_ptr<rhi::ComputeBuffer> old, uint32_t source,
                                                   std::shared_ptr<BoundsValidation> validation)
{
    if (!old || old->GetByteSize() != 64u || !old->HasInitializedRange(0u, 64u))
        throw std::invalid_argument("GPU bounds wake requires its exact completed old-bound capture");
    auto e = std::make_shared<Impl::Execution>(m_impl);
    e->Bind(s);
    e->boundsValidation = std::move(validation);
    e->finalResumeAfter = JoltGpuControlWork::ResumeAfter::Acceptance;
    auto p = Selection(s, selection);
    p.operation.x = source;
    p.operation.y = 1u | (source == UINT32_MAX ? 0u : 2u);
    e->buffers[Bounds] = std::move(old);
    e->Add(BoundsWake, p.selected.y, p.selected.z, p);
    e->OnComplete([](auto &) {});
    return e->Next();
}
JoltGpuControlWork JoltGpuSleep::PrepareSourceBoundsWake(const JoltGpuStorage &s, const StepParams &selection,
                                                         std::vector<uint32_t> sources,
                                                         std::shared_ptr<BoundsValidation> validation,
                                                         std::vector<std::pair<uint32_t, uint32_t>> exclusions)
{
    auto e = std::make_shared<Impl::Execution>(m_impl);
    e->Bind(s);
    const auto sourceCount = static_cast<uint32_t>(sources.size());
    std::sort(exclusions.begin(), exclusions.end());
    sources.reserve(sources.size() + 2 * exclusions.size());
    for (const auto &[source, target] : exclusions) {
        sources.push_back(source);
        sources.push_back(target);
    }
    e->SetRequests(sources);
    e->boundsValidation = std::move(validation);
    e->finalResumeAfter = JoltGpuControlWork::ResumeAfter::Acceptance;
    auto p = Selection(s, selection);
    p.layout.z = sourceCount;
    p.layout.w = static_cast<uint32_t>(exclusions.size());
    // Bound the comparisons in each dispatch in both dimensions. The same
    // immutable source list is used by all segments of this topology edit.
    constexpr uint32_t SourcesPerDispatch = 64;
    for (uint32_t first = 0; first < sourceCount; first += SourcesPerDispatch) {
        p.operation = {UINT32_MAX, 0u, first, std::min(SourcesPerDispatch, sourceCount - first)};
        e->Add(NewBoundsWake, p.selected.y, p.selected.z, p);
    }
    e->OnComplete([](auto &) {});
    return e->Next();
}
JoltGpuControlWork JoltGpuSleep::PrepareAuthoredBoundsWake(const JoltGpuStorage &s, const StepParams &selection,
                                                           Float4 low, Float4 high,
                                                           std::shared_ptr<BoundsValidation> validation)
{
    if (!std::isfinite(low.x) || !std::isfinite(low.y) || !std::isfinite(low.z) || low.w != 0 ||
        !std::isfinite(high.x) || !std::isfinite(high.y) || !std::isfinite(high.z) || high.w != 0 || low.x > high.x ||
        low.y > high.y || low.z > high.z)
        throw std::invalid_argument("Wake bounds must be finite ordered world coordinates");
    auto e = std::make_shared<Impl::Execution>(m_impl);
    e->Bind(s);
    e->boundsValidation = std::move(validation);
    e->finalResumeAfter = JoltGpuControlWork::ResumeAfter::Acceptance;
    auto p = Selection(s, selection);
    p.operation.y = 1u;
    e->resultPort = Bounds;
    e->resultBytes = 32u;
    e->boundsBytes = Bytes(std::array<Float4, 2>{low, high});
    e->Add(BoundsWake, p.selected.y, p.selected.z, p);
    e->OnComplete([](auto &) {});
    return e->Next();
}

JoltGpuPoseWake JoltGpuSleep::PreparePoseWake(const JoltGpuStorage &s, const StepParams &selection,
                                              BoundsValidation &validation)
{
    auto e = std::make_shared<Impl::Execution>(m_impl);
    e->Bind(s);
    const auto p = Selection(s, selection);
    const auto makeBuffer = [](rhi::ComputeHost &host, size_t bytes) {
        return std::make_shared<rhi::ComputeBuffer>(
            host, rhi::ComputeBufferDesc{bytes / 4u, rhi::ComputeScalarType::UInt32, 1});
    };
    if (!validation.status)
        validation.status = makeBuffer(m_impl->host, sizeof(uint32_t));
    e->buffers[Roots] = validation.status;
    e->buffers[Bounds] = makeBuffer(m_impl->host, 4u * sizeof(Float4));

    return {p.selected.z,
            [e, p, makeBuffer](const JoltGpuPoseWake::Work &piece, std::vector<rhi::ComputeBufferUpdate> &uploads,
                               std::vector<rhi::ComputeDispatchDesc> &dispatches) {
                const auto &status = e->buffers[Roots];
                if (!status->HasInitializedRange(0u, sizeof(uint32_t)) &&
                    std::none_of(uploads.begin(), uploads.end(),
                                 [&](const auto &upload) { return upload.buffer == status; }))
                    uploads.push_back({status, 0u, Bytes(uint32_t(0))});
                const auto uniform = [&](const auto &value) {
                    auto bytes = Bytes(value);
                    auto found = std::find_if(uploads.begin(), uploads.end(),
                                              [&](const auto &upload) { return upload.data == bytes; });
                    if (found != uploads.end())
                        return found->buffer;
                    auto buffer = makeBuffer(e->owner->host, bytes.size());
                    uploads.push_back({buffer, 0u, std::move(bytes)});
                    return buffer;
                };
                auto parameters = p;
                parameters.operation = {piece.source, 7u, 0u, 0u}; // Old and new bounds, only if pose changes.
                const uint32_t first = p.selected.y + piece.first;
                const auto params = uniform(parameters);
                const auto work = piece.capture ? nullptr : uniform(StageWork{{first, piece.count, 0u, 0u}});
                const Impl::Execution::Operation op{piece.capture ? CaptureBounds : BoundsWake, first, piece.count,
                                                    parameters};
                dispatches.push_back(e->Dispatch(op, 0u, piece.count, params, work));
            }};
}

JoltGpuControlWork JoltGpuSleep::ValidateBoundsEdits(const BoundsValidation &validation)
{
    if (!validation.status)
        return {};
    auto packet = rhi::PrepareComputeBatch(m_impl->host, {}, {}, {{validation.status, 0u, sizeof(uint32_t)}});
    return {packet, [packet] {
                Impl::Execution::ReadStatus(*packet);
                return JoltGpuControlWork{};
            }};
}
JoltGpuControlWork JoltGpuSleep::PrepareQuery(std::shared_ptr<rhi::ComputeBuffer> snapshot, uint32_t count,
                                              std::vector<uint32_t> requests,
                                              std::function<void(std::vector<UInt4>)> done)
{
    if (requests.size() > UINT32_MAX || !snapshot || snapshot->GetByteSize() != uint64_t(count) * 16u ||
        !snapshot->HasInitializedRange(0u, snapshot->GetByteSize()) ||
        !m_impl->host.SharesServicesWith(snapshot->GetHost()))
        throw std::invalid_argument("GPU sleep query requires its exact completed publication");
    for (auto index : requests)
        if (index >= count)
            throw std::out_of_range("GPU sleep query body exceeds publication");
    auto e = std::make_shared<Impl::Execution>(m_impl);
    e->buffers[Snapshot] = std::move(snapshot);
    e->query = true;
    e->resultBytes = uint64_t(requests.size()) * 16u;
    e->SetRequests(requests);
    SleepWorkParameters p{};
    p.selected.z = count;
    p.operation.w = static_cast<uint32_t>(requests.size());
    e->Add(Query, 0u, p.operation.w, p);
    e->OnComplete([done = std::move(done)](auto &self) { done(std::move(self.queryResults)); });
    return e->Next();
}
} // namespace infernux
