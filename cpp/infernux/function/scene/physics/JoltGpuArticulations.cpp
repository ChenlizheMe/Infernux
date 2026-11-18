#include "JoltGpuArticulations.h"
#include "JoltGpuArticulationForest.h"
#include "JoltGpuArticulationNumerics.h"
#include "JoltGpuProgramSources.h"
#include "JoltGpuSleepLayout.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string_view>

namespace infernux
{
namespace
{
using namespace JPH::GPU;
enum Port : uint8_t
{
    Params,
    Words,
    Models,
    Sleeps,
    Targets,
    Joints,
    Schedule,
    PortCount
};
enum Entry : uint8_t
{
    Rotors,
    Begin,
    Bodies,
    RuntimeJoints,
    References,
    Finalize,
    EntryCount
};
constexpr std::array<std::string_view, EntryCount> Names{
    "articulation_capture_rotors",  "articulation_classify_begin", "articulation_classify_bodies",
    "articulation_classify_joints", "articulation_classify_refs",  "articulation_classify_finalize"};
constexpr std::array<std::string_view, PortCount> Ports{"articulation_params", "articulation_words", "body_models",
                                                        "body_sleep",          "kinematic_targets",  "joint_stream",
                                                        "constraint_schedule"};
constexpr uint32_t P = 1u << Params, W = 1u << Words;
constexpr std::array<uint32_t, EntryCount> Masks{P | W | (1u << Joints),
                                                 P | W,
                                                 P | W | (1u << Models) | (1u << Sleeps) | (1u << Targets),
                                                 P | W | (1u << Joints),
                                                 P | W | (1u << Models) | (1u << Schedule),
                                                 P | W};
template <class T> std::vector<uint8_t> Bytes(const T &v)
{
    std::vector<uint8_t> bytes(sizeof(T));
    std::memcpy(bytes.data(), &v, sizeof(T));
    return bytes;
}
void Validate(Port port, const ShaderResourceLayout &r)
{
    constexpr uint32_t Strides[]{32, 4, 80, 128, 64, 16, 16};
    const bool uniform = port == Params;
    const bool readOnly = port == Models || port == Schedule;
    if (r.set != 0 || r.binding.type != (uniform ? rhi::BindingType::UniformBuffer : rhi::BindingType::StorageBuffer) ||
        (uniform ? r.value.byteSize : r.elementStride) != Strides[port] ||
        (!uniform && r.binding.readOnlyStorage != readOnly))
        throw std::invalid_argument("GPU articulation control resource differs from shared ABI");
    if (port == Sleeps) {
        ValidateJoltGpuSleepLayout(r.value);
        return;
    }
    if (port == Words) {
        if (r.value.kind != ShaderValueKind::Scalar || r.value.scalar != ShaderScalarKind::UInt32 ||
            r.value.byteSize != 4)
            throw std::invalid_argument("GPU articulation atomic word is not uint32");
        return;
    }
    constexpr std::array<std::array<std::string_view, 5>, PortCount> Fields{
        {{"range", "references"},
         {},
         {"meta", "mass", "invInertiaDiagonal", "inertiaRotation", "damping"},
         {},
         {"positionSpeed", "rotation", "rootCOMDt", "state"},
         {"value"},
         {"value"}}};
    const uint32_t integerMask = port == Params ? 3u : port == Models ? 1u : port == Targets ? 8u : 1u;
    if (r.value.kind != ShaderValueKind::Struct || r.value.fields.size() != Strides[port] / 16u)
        throw std::invalid_argument("GPU articulation control field count differs from shared ABI");
    for (size_t i = 0; i < r.value.fields.size(); ++i) {
        const auto &f = r.value.fields[i];
        if (f.name != Fields[port][i] || f.offset != i * 16u || f.byteSize != 16 || f.kind != ShaderValueKind::Vector ||
            f.rows != 1 || f.columns != 4 ||
            f.scalar != ((integerMask & (1u << i)) ? ShaderScalarKind::UInt32 : ShaderScalarKind::Float32))
            throw std::invalid_argument("GPU articulation control field differs from shared ABI");
    }
}
} // namespace

class JoltGpuArticulations::Impl
{
  public:
    struct Program
    {
        std::shared_ptr<rhi::ComputeKernel> kernel;
        std::vector<Port> ports;
    };
    struct Initialization;
    struct Execution;
    Impl(rhi::ComputeHost &source, const ShaderTargetCode &target, std::shared_ptr<rhi::ComputeBuffer> canonicalEmpty)
        : host(source), empty(std::move(canonicalEmpty))
    {
        if (target.target != ShaderCodeTarget::SpirV && target.target != ShaderCodeTarget::Wgsl)
            throw std::invalid_argument("GPU articulation control requires SPIR-V/WGSL");
        for (uint32_t i = 0; i < EntryCount; ++i) {
            const auto code = std::find_if(target.entries.begin(), target.entries.end(),
                                           [&](const auto &e) { return e.name == Names[i]; });
            if (code == target.entries.end() || code->stage != rhi::ShaderStage::Compute ||
                code->workgroupSize != std::array<uint32_t, 3>{64, 1, 1})
                throw std::invalid_argument("GPU articulation control artifact lacks bounded entry");
            uint32_t mask = 0;
            auto &program = entries[i];
            std::vector<rhi::ComputeBufferBinding> bindings;
            for (const auto &r : target.resources) {
                if (std::find(code->bindings.begin(), code->bindings.end(),
                              std::array<uint32_t, 2>{r.set, r.binding.binding}) == code->bindings.end())
                    continue;
                const auto found = std::find(Ports.begin(), Ports.end(), r.name);
                if (found == Ports.end())
                    throw std::invalid_argument("GPU articulation control entry has unknown resource");
                const auto port = static_cast<Port>(found - Ports.begin());
                Validate(port, r);
                if (mask & (1u << port))
                    throw std::invalid_argument("GPU articulation control repeats resource");
                mask |= 1u << port;
                program.ports.push_back(port);
                bindings.push_back({r.binding.binding, r.binding.type, r.binding.readOnlyStorage});
            }
            if (mask != Masks[i] || code->bindings.size() != program.ports.size())
                throw std::invalid_argument(
                    "GPU articulation control entry resource mask differs from actual artifact");
            program.kernel = PrepareJoltGpuKernel(host, target.target, *code, std::move(bindings));
        }
    }
    std::shared_ptr<rhi::ComputeBuffer> Allocate(uint64_t bytes)
    {
        if (!bytes || bytes % 4u)
            throw std::logic_error("GPU articulation allocation is not a word range");
        if (bytes > host.device.GetCapabilities().portable.maxStorageBufferBinding || bytes > UINT32_MAX)
            return {};
        return rhi::ComputeBuffer::TryCreatePrivate(
            host, {bytes / 4u, rhi::ComputeScalarType::UInt32, 1u, rhi::BufferUsageFlags::None});
    }
    rhi::ComputeHost host;
    std::array<Program, EntryCount> entries;
    std::shared_ptr<rhi::ComputeBuffer> empty;
    std::shared_ptr<JoltGpuArticulationNumerics> numeric;
};

// One bounded upload per packet. Existing Work budgets may advance multiple
// packets in one Host; 16KiB is a fragment size, never a per-frame quota.
struct JoltGpuArticulations::Impl::Initialization : std::enable_shared_from_this<Initialization>
{
    std::shared_ptr<Impl> owner;
    std::shared_ptr<JoltGpuArticulationState> state;
    std::function<JoltGpuControlWork(std::shared_ptr<JoltGpuArticulationState>)> completed;
    std::function<JoltGpuControlWork()> declined;
    uint32_t part = 0;
    uint64_t offset = 0;
    JoltGpuControlWork Decline()
    {
        state.reset();
        auto next = std::move(declined);
        if (!next)
            throw std::logic_error("Canonical articulation header lost its existing allocation");
        return next();
    }
    JoltGpuControlWork Next()
    {
        if (!state->control && !(state->control = owner->Allocate(state->header.storage.w)))
            return Decline();
        if (!state->links.empty()) {
            if (!state->linkBuffer &&
                !(state->linkBuffer = owner->Allocate(state->links.size() * sizeof(ArticulationLink))))
                return Decline();
            if (!state->treeBuffer &&
                !(state->treeBuffer = owner->Allocate(state->trees.size() * sizeof(ArticulationTree))))
                return Decline();
        }
        const std::array<std::shared_ptr<rhi::ComputeBuffer>, 3> buffers{state->control, state->linkBuffer,
                                                                         state->treeBuffer};
        const std::array<uint64_t, 3> sizes{state->prefix.size(), state->links.size() * sizeof(ArticulationLink),
                                            state->trees.size() * sizeof(ArticulationTree)};
        while (part < 3 && offset == sizes[part]) {
            ++part;
            offset = 0;
        }
        if (part == 3) {
            auto next = std::move(completed);
            return next(std::move(state));
        }
        const auto *source = part == 0   ? state->prefix.data()
                             : part == 1 ? reinterpret_cast<const uint8_t *>(state->links.data())
                                         : reinterpret_cast<const uint8_t *>(state->trees.data());
        const uint64_t bytes = std::min<uint64_t>(16384u, sizes[part] - offset);
        std::vector<uint8_t> chunk(source + offset, source + offset + bytes);
        auto packet = rhi::PrepareComputeBatch(owner->host, {{buffers[part], offset, std::move(chunk)}}, {}, {});
        return {packet,
                [self = shared_from_this(), bytes] {
                    self->offset += bytes;
                    return self->Next();
                },
                JoltGpuControlWork::ResumeAfter::Acceptance};
    }
};

struct JoltGpuArticulations::Impl::Execution : std::enable_shared_from_this<Execution>
{
    struct Operation
    {
        Entry entry;
        ArticulationClassifyWork parameters;
    };
    std::shared_ptr<Impl> owner;
    std::shared_ptr<JoltGpuArticulationState> state;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> buffers;
    std::vector<Operation> plan;
    std::function<void()> completed;
    size_t cursor = 0;
    uint32_t offset = 0;
    void Add(Entry entry, uint32_t first, uint32_t count, uint32_t world, UInt4 refs = {}, uint32_t kind = 0)
    {
        if (count)
            plan.push_back({entry, {{first, count, world, kind}, refs}});
    }
    JoltGpuControlWork Next(uint32_t maximum = 8)
    {
        if (cursor == plan.size()) {
            auto done = std::move(completed);
            done();
            return {};
        }
        size_t next = cursor;
        uint32_t nextOffset = offset;
        std::vector<rhi::ComputeBufferUpdate> uploads;
        std::vector<rhi::ComputeDispatchDesc> dispatches;
        const uint64_t dispatchLimit =
            uint64_t(owner->host.device.GetCapabilities().limits.maxComputeWorkgroupCount[0]) * 64u;
        const uint32_t quantum = uint32_t(std::min<uint64_t>(2048u, dispatchLimit));
        if (!quantum)
            throw std::length_error("GPU articulation control has no dispatch capacity");
        while (next < plan.size() && dispatches.size() < maximum) {
            const auto &operation = plan[next];
            auto parameters = operation.parameters;
            const uint32_t count = std::min(quantum, parameters.range.y - nextOffset);
            parameters.range.x += nextOffset;
            parameters.range.y = count;
            if (!state->numeric || !state->numeric->uniforms[dispatches.size()])
                throw std::logic_error("Articulation control lost its cold-admitted parameter pool");
            auto uniform = state->numeric->uniforms[dispatches.size()];
            uploads.push_back({uniform, 0, Bytes(parameters)});
            rhi::ComputeDispatchDesc dispatch;
            dispatch.kernel = owner->entries[operation.entry].kernel;
            dispatch.groupCountX = (count + 63u) / 64u;
            rhi::ComputeDispatchProof proof;
            for (const auto port : owner->entries[operation.entry].ports) {
                const auto buffer = port == Params ? uniform : buffers[port];
                if (!buffer)
                    throw std::logic_error("GPU articulation classification lacks its accepted source");
                const uint32_t binding = uint32_t(dispatch.buffers.size());
                dispatch.buffers.push_back(buffer);
                dispatch.bufferAccesses.push_back(port == Words ? rhi::ComputeBufferAccess::ReadWrite
                                                                : rhi::ComputeBufferAccess::Read);
                // Control prefix only. The same allocation's uninitialized numeric
                // tail is NEVER claimed readable or written by this foundation.
                proof.reads.push_back({binding, 0, port == Words ? state->header.storage.z : buffer->GetByteSize()});
                if (port == Words) {
                    const auto &h = state->header;
                    if (operation.entry == Begin || operation.entry == Finalize) {
                        proof.writes.push_back(
                            {binding, uint64_t(h.maps.w) + uint64_t(parameters.range.x) * 4u, uint64_t(count) * 4u});
                        if (operation.entry == Begin)
                            proof.writes.push_back({binding, uint64_t(h.storage.x) + uint64_t(parameters.range.x) * 4u,
                                                    uint64_t(count) * 4u});
                    }
                    // Rotors skip unmapped/non-Hinge rows; body/ref passes OR
                    // only incident reasons. They preserve an initialized prefix
                    // and make NO false unconditional complete-store claim.
                }
            }
            dispatch.proof = std::move(proof);
            dispatches.push_back(std::move(dispatch));
            nextOffset += count;
            if (nextOffset == operation.parameters.range.y) {
                ++next;
                nextOffset = 0;
            }
        }
        if (dispatches.empty())
            return {nullptr, [self = shared_from_this()] { return self->Next(); }};
        const bool multi = dispatches.size() > 1u;
        auto packet = rhi::PrepareComputeBatch(owner->host, std::move(uploads), std::move(dispatches), {});
        JoltGpuControlWork work{packet,
                                [self = shared_from_this(), next, nextOffset] {
                                    self->cursor = next;
                                    self->offset = nextOffset;
                                    return self->Next();
                                },
                                JoltGpuControlWork::ResumeAfter::Acceptance};
        if (multi)
            work.prepareMinimum = [self = shared_from_this()] { return self->Next(1); };
        return work;
    }
};

JoltGpuArticulations::JoltGpuArticulations(rhi::ComputeHost &host, const ShaderTargetCode &target,
                                           std::shared_ptr<rhi::ComputeBuffer> empty, const ShaderTargetCode *capture,
                                           const ShaderTargetCode *dynamics)
    : m_impl(std::make_shared<Impl>(host, target, std::move(empty)))
{
    if (bool(capture) != bool(dynamics))
        throw std::invalid_argument("Reduced owner requires both numeric artifacts");
    if (capture)
        m_impl->numeric = std::make_shared<JoltGpuArticulationNumerics>(host, target, *capture, *dynamics);
}

JoltGpuControlWork
JoltGpuArticulations::PrepareAccepted(const JoltGpuStorage &storage, uint32_t world, std::shared_ptr<const void> key,
                                      std::shared_ptr<JoltGpuArticulationState> previous,
                                      std::function<void(std::shared_ptr<JoltGpuArticulationState>)> completed,
                                      bool acceptedStateIntent)
{
    if (!m_impl->host.SharesServicesWith(storage.GetHost()))
        throw std::invalid_argument("GPU articulation crosses compute owner");
    if (previous && previous->authorKey == key) {
        if (acceptedStateIntent && m_impl->numeric && !previous->resourceDeclined)
            return m_impl->numeric->PrepareDirty(previous,
                                                 [previous, completed = std::move(completed)] { completed(previous); });
        completed(std::move(previous));
        return {};
    }
    // One cold decision per accepted author/layout generation. A managed-null
    // optional allocation declines lowering here; it never suspends an otherwise
    // valid maximal World indefinitely or creates a future-frame retry source.
    const auto prepareDeclined = [owner = m_impl, &storage, world, key, previous, completed]() -> JoltGpuControlWork {
        auto empty = BuildJoltGpuArticulationForest(storage.GetModel(), world, 48u);
        empty->resourceDeclined = true;
        empty->authorKey = key;
        empty->control = owner->empty;
        auto init = std::make_shared<Impl::Initialization>();
        init->owner = owner;
        init->state = empty;
        if (empty->control->HasInitializedRange(0, ArtControlHeaderBytes))
            init->part = 1;
        init->completed = [owner, &storage, previous, completed](auto value) {
            if (owner->numeric)
                return owner->numeric->PrepareDeclined(storage, value, previous,
                                                       [value, completed] { completed(value); });
            completed(value);
            return JoltGpuControlWork{};
        };
        return init->Next();
    };
    auto state = BuildJoltGpuArticulationForest(
        storage.GetModel(), world,
        m_impl->numeric
            ? std::min<uint64_t>(UINT32_MAX, m_impl->host.device.GetCapabilities().portable.maxStorageBufferBinding)
            : 48u);
    if (state->trees.empty()) {
        // The canonical descriptor is initialized once, not one dummy per World/tick.
        state->control = m_impl->empty;
    }
    state->authorKey = key;
    auto execution = std::make_shared<Impl::Initialization>();
    execution->owner = m_impl;
    execution->state = state;
    execution->declined = prepareDeclined;
    execution->completed = [owner = m_impl, &storage, previous, completed, prepareDeclined](auto value) {
        if (!owner->numeric) {
            completed(value);
            return JoltGpuControlWork{};
        }
        if (value->trees.empty())
            return owner->numeric->PrepareDeclined(storage, value, previous, [value, completed] { completed(value); });
        return owner->numeric->PrepareInitialize(storage, value, previous,
                                                 [value, completed, prepareDeclined](bool accepted) {
                                                     if (accepted) {
                                                         completed(value);
                                                         return JoltGpuControlWork{};
                                                     }
                                                     value->numeric.reset();
                                                     value->control.reset();
                                                     value->linkBuffer.reset();
                                                     value->treeBuffer.reset();
                                                     return prepareDeclined();
                                                 });
    };
    if (state->trees.empty() && state->control->HasInitializedRange(0, ArtControlHeaderBytes))
        execution->part = 1;
    return execution->Next();
}

JoltGpuControlWork
JoltGpuArticulations::PreparePressureDeclined(const JoltGpuStorage &storage,
                                              std::shared_ptr<JoltGpuArticulationState> previous, bool capturedWitness,
                                              std::function<void(std::shared_ptr<JoltGpuArticulationState>)> completed)
{
    if (!previous || previous->resourceDeclined) {
        completed(std::move(previous));
        return {};
    }
    auto empty = BuildJoltGpuArticulationForest(storage.GetModel(), previous->worldRow, 48u);
    empty->resourceDeclined = true;
    empty->authorKey = previous->authorKey;
    empty->control = m_impl->empty;
    auto init = std::make_shared<Impl::Initialization>();
    init->owner = m_impl;
    init->state = empty;
    if (empty->control->HasInitializedRange(0, ArtControlHeaderBytes))
        init->part = 1;
    init->completed = [owner = m_impl, &storage, previous, capturedWitness,
                       completed = std::move(completed)](auto value) {
        if (owner->numeric)
            return owner->numeric->PrepareDeclined(
                storage, value, previous, [value, completed] { completed(value); }, capturedWitness);
        completed(value);
        return JoltGpuControlWork{};
    };
    return init->Next();
}

JoltGpuControlWork JoltGpuArticulations::PrepareRotors(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                                                       std::function<void()> completed)
{
    auto state = inputs.GetArticulations();
    auto after = [owner = m_impl, &storage, &inputs, completed = std::move(completed)] {
        if (owner->numeric)
            return owner->numeric->PrepareBeforeForces(storage, inputs, completed);
        completed();
        return JoltGpuControlWork{};
    };
    if (!state || state->trees.empty())
        return after();
    auto e = std::make_shared<Impl::Execution>();
    e->owner = m_impl;
    e->state = state;
    e->completed = [] {};
    e->buffers[Words] = state->control;
    e->buffers[Joints] = storage.Get(Resource::Joints);
    e->Add(Rotors, state->header.selected.z, state->header.selected.w, inputs.GetParameters().selection.x);
    return ContinueJoltGpuControlWork(e->Next(), std::move(after));
}
JoltGpuControlWork JoltGpuArticulations::PrepareClassification(const JoltGpuStorage &storage,
                                                               const JoltGpuStepInputs &inputs,
                                                               const std::vector<uint32_t> *participantBodies,
                                                               std::function<void()> completed)
{
    auto state = inputs.GetArticulations();
    auto after = [owner = m_impl, &storage, &inputs, completed = std::move(completed)] {
        if (owner->numeric)
            return owner->numeric->PrepareCandidate(storage, inputs, completed);
        completed();
        return JoltGpuControlWork{};
    };
    if (!state || state->trees.empty())
        return after();
    const auto &plan = inputs.GetConstraints();
    if (!plan)
        throw std::logic_error("GPU articulation classification precedes validated numerical workset");
    auto e = std::make_shared<Impl::Execution>();
    e->owner = m_impl;
    e->state = state;
    e->completed = [] {};
    e->buffers[Words] = state->control;
    e->buffers[Models] = inputs.GetCaptured(Resource::BodyModels);
    e->buffers[Sleeps] = storage.Get(Resource::SleepStates);
    e->buffers[Targets] = storage.Get(Resource::KinematicTargets);
    e->buffers[Joints] = storage.Get(Resource::Joints);
    e->buffers[Schedule] = plan->schedule;
    const uint32_t world = inputs.GetParameters().selection.x;
    const auto &h = state->header;
    const bool sweepsDynamicPairs = storage.HasDynamicSweeps(world) || inputs.NeedsCoupledSweep();
    // Cross-device contact rows share the same maximal velocity representation as ordinary external contacts.
    std::vector<uint32_t> contactedTrees;
    const auto addTree = [&](uint32_t slot) {
        if (slot < h.selected.x || slot - h.selected.x >= h.selected.y)
            return;
        uint32_t tree;
        std::memcpy(&tree, state->prefix.data() + h.maps.y + 4u * (slot - h.selected.x), sizeof(tree));
        if (tree != ArtControlInvalid)
            contactedTrees.push_back(tree);
    };
    for (const auto slot : inputs.GetExternalContactBodies())
        addTree(slot);
    if (participantBodies)
        for (const auto slot : *participantBodies)
            addTree(slot);
    std::sort(contactedTrees.begin(), contactedTrees.end());
    contactedTrees.erase(std::unique(contactedTrees.begin(), contactedTrees.end()), contactedTrees.end());
    uint32_t first = 0;
    for (size_t i = 0; i < contactedTrees.size();) {
        const uint32_t start = contactedTrees[i];
        uint32_t end = start + 1;
        for (++i; i < contactedTrees.size() && contactedTrees[i] == end; ++i)
            ++end;
        e->Add(Begin, first, start - first, world, {}, sweepsDynamicPairs ? ArtRefusedContact : 0u);
        e->Add(Begin, start, end - start, world, {}, ArtRefusedContact);
        first = end;
    }
    // A dynamic sweep can hit a link after classification. Use the representation
    // that accepts the swept impulse and integrates the affected rigid links.
    e->Add(Begin, first, h.maps.x - first, world, {}, sweepsDynamicPairs ? ArtRefusedContact : 0u);
    e->Add(Bodies, h.selected.x, h.selected.y, world);
    e->Add(RuntimeJoints, h.selected.z, h.selected.w, world);
    const UInt4 refs{plan->jointRefFirst, plan->jointRefCount, plan->patchRefFirst, plan->patchRefCount};
    e->Add(References, 0, refs.y, world, refs, 0);
    e->Add(References, 0, refs.w, world, refs, 1);
    e->Add(Finalize, 0, h.maps.x, world, {}, inputs.GetParameters().counts.w == 0u ? 1u : 0u);
    return ContinueJoltGpuControlWork(e->Next(), std::move(after));
}
JoltGpuControlWork JoltGpuArticulations::PreparePose(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                                                     std::function<void()> completed)
{
    if (m_impl->numeric)
        return m_impl->numeric->PreparePose(storage, inputs, std::move(completed));
    completed();
    return {};
}
} // namespace infernux
