#include "JoltGpuConstraints.h"
#include <Jolt/GPU/ConstraintTypes.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Build structural schedules on the Tick source from its accepted inputs.
// Submission, completion and resource admission remain engine-owned.
namespace infernux
{
namespace
{
using namespace JPH::GPU;

enum Port : uint8_t
{
    Params,
    Work,
    Bodies,
    Shapes,
    Worlds,
    Joints,
    Patches,
    Contacts,
    LabelsOld,
    LabelsNext,
    Schedule,
    PortCount
};
enum Entry : uint8_t
{
    InitWords,
    Seed,
    BuildJointRefs,
    BuildPatchRefs,
    InitLabels,
    ShortcutLabels,
    PropagateEdges,
    CheckRound,
    CommitRound,
    ValidateLabels,
    ValidateEdges,
    AcceptComponents,
    ScanBlocks,
    ReduceBlocks,
    AddOffsets,
    SeedSort,
    MergeItems,
    ClearRootFlags,
    MarkRoots,
    BuildRootRows,
    WriteSources,
    WriteIslands,
    Finalize,
    EntryCount
};
constexpr std::array<std::string_view, EntryCount> EntryNames{
    "constraint_init_words",       "constraint_seed",
    "constraint_build_joint_refs", "constraint_build_patch_refs",
    "constraint_init_labels",      "constraint_shortcut_labels",
    "constraint_propagate_edges",  "constraint_check_round",
    "constraint_commit_round",     "constraint_validate_labels",
    "constraint_validate_edges",   "constraint_accept_components",
    "constraint_scan_blocks",      "constraint_reduce_blocks",
    "constraint_add_scan_offsets", "constraint_seed_sort",
    "constraint_merge_items",      "constraint_clear_root_flags",
    "constraint_mark_roots",       "constraint_build_root_rows",
    "constraint_write_sources",    "constraint_write_islands",
    "constraint_finalize"};
constexpr uint32_t Bit(Port port)
{
    return 1u << port;
}
constexpr uint32_t P = Bit(Params), K = Bit(Work), S = Bit(Schedule), B = Bit(Bodies), W = Bit(Worlds),
                   O = Bit(LabelsOld), N = Bit(LabelsNext);
constexpr std::array<uint32_t, EntryCount> EntryPorts{P | K | S,
                                                      P | S,
                                                      P | K | S | B | W | Bit(Joints),
                                                      P | K | S | B | W | Bit(Shapes) | Bit(Patches) | Bit(Contacts),
                                                      P | K | B | W | O | N,
                                                      P | K | S | B | W | O | N,
                                                      P | K | S | B | W | O | N,
                                                      P | K | S | B | W | O | N,
                                                      P | S,
                                                      P | K | S | B | W | O,
                                                      P | K | S | B | W | O,
                                                      P | S,
                                                      P | K | S,
                                                      P | K | S,
                                                      P | K | S,
                                                      P | K | S | O,
                                                      P | K | S,
                                                      P | K | N,
                                                      P | K | S | O | N,
                                                      P | K | S | O | N,
                                                      P | K | S,
                                                      P | K | S | B,
                                                      P | S};

struct PortLayout
{
    std::string_view name;
    uint64_t stride;
    std::array<std::string_view, 12> fields;
    uint32_t integers;
    bool readOnly;
};
constexpr PortLayout Ports[] = {
    {"constraint_params",
     sizeof(ConstraintParams),
     {"counts", "limits", "layout", "references", "sources", "islands", "scan", "scanParent", "merge", "select",
      "summary", "round"},
     0xfffu,
     true},
    {"stage_work", sizeof(StageWork), {"range"}, 1u, true},
    {"body_models", sizeof(BodyModel), {"meta", "mass", "invInertiaDiagonal", "inertiaRotation", "damping"}, 1u, true},
    {"shape_models",
     sizeof(ShapeModel),
     {"meta", "geometry", "localPositionCOM", "localRotation", "material", "surface", "source"},
     (1u << 0) | (1u << 5) | (1u << 6),
     true},
    {"worlds", sizeof(WorldRange), {"bodies", "leaves"}, 3u, true},
    {"joint_stream", sizeof(QueryRecordWord), {"value"}, 1u, true},
    {"contact_patches", sizeof(ContactPatch), {"leaves"}, 1u, true},
    {"contacts", sizeof(Contact), {"meta", "normalDepth", "localAnchorA", "localAnchorB", "impulse"}, 1u, true},
    {"labels_old", sizeof(uint32_t), {}, 0u, false},
    {"labels_next", sizeof(uint32_t), {}, 0u, false},
    {"constraint_schedule", sizeof(ConstraintScheduleWord), {"value"}, 1u, false}};
static_assert(std::size(Ports) == PortCount);
constexpr uint32_t MaxDispatchesPerPacket = 16u;
constexpr uint64_t WordBytes = sizeof(ConstraintScheduleWord);

uint32_t Checked(uint64_t n)
{
    if (n > UINT32_MAX)
        throw std::overflow_error("GPU constraint indexing exceeds uint32");
    return static_cast<uint32_t>(n);
}
uint32_t Blocks(uint32_t n)
{
    return n / 64u + uint32_t(n % 64u != 0u);
}
bool Collective(Entry e)
{
    return e == ScanBlocks || e == ReduceBlocks;
}
bool Summary(Entry e)
{
    return e == Seed || e == CommitRound || e == AcceptComponents || e == Finalize;
}
template <class T> std::vector<uint8_t> Bytes(const T &v)
{
    std::vector<uint8_t> result(sizeof(T));
    std::memcpy(result.data(), &v, sizeof(T));
    return result;
}

void ValidateLayout(Port port, const ShaderResourceLayout &resource)
{
    const auto &expected = Ports[port];
    const bool uniform = port == Params || port == Work;
    const auto &value = resource.value;
    if (resource.binding.type != (uniform ? rhi::BindingType::UniformBuffer : rhi::BindingType::StorageBuffer) ||
        (uniform ? value.byteSize : resource.elementStride) != expected.stride ||
        (!uniform && resource.binding.readOnlyStorage != expected.readOnly))
        throw std::invalid_argument("GPU constraint binding differs from shared POD");
    if (port == LabelsOld || port == LabelsNext) {
        // Scalar uint atomics require this layout, not a UInt4 struct field.
        if (value.kind != ShaderValueKind::Scalar || value.scalar != ShaderScalarKind::UInt32 || value.byteSize != 4u ||
            value.rows != 1u || value.columns != 1u || !value.fields.empty())
            throw std::invalid_argument("GPU constraint labels require scalar uint32 stride4 storage");
        return;
    }
    if (value.kind != ShaderValueKind::Struct || value.fields.size() != expected.stride / WordBytes)
        throw std::invalid_argument("GPU constraint struct differs from shared POD");
    for (size_t i = 0; i < value.fields.size(); ++i) {
        const auto &field = value.fields[i];
        const auto scalar = expected.integers & (1u << i) ? ShaderScalarKind::UInt32 : ShaderScalarKind::Float32;
        if (field.name != expected.fields[i] || field.offset != i * WordBytes || field.byteSize != WordBytes ||
            field.kind != ShaderValueKind::Vector || field.scalar != scalar || field.rows != 1u || field.columns != 4u)
            throw std::invalid_argument("GPU constraint field differs from shared POD");
    }
}
} // namespace

class JoltGpuConstraints::Impl
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
            throw std::invalid_argument("GPU constraints require the engine SPIR-V/WGSL target");
        for (uint8_t i = 0; i < EntryCount; ++i) {
            const auto entry = static_cast<Entry>(i);
            const auto code = std::find_if(target.entries.begin(), target.entries.end(),
                                           [&](const auto &item) { return item.name == EntryNames[i]; });
            if (code == target.entries.end() || code->stage != rhi::ShaderStage::Compute ||
                code->workgroupSize != std::array<uint32_t, 3>{Summary(entry) ? 1u : 64u, 1u, 1u})
                throw std::invalid_argument("GPU constraint entry/workgroup differs from frozen module");
            auto &program = programs[i];
            std::vector<rhi::ComputeBufferBinding> bindings;
            uint32_t actual = 0, storageCount = 0;
            for (const auto &resource : target.resources) {
                const std::array<uint32_t, 2> key{resource.set, resource.binding.binding};
                if (std::find(code->bindings.begin(), code->bindings.end(), key) == code->bindings.end())
                    continue;
                const auto found = std::find_if(std::begin(Ports), std::end(Ports),
                                                [&](const auto &port) { return port.name == resource.name; });
                if (found == std::end(Ports) || resource.set != 0u)
                    throw std::invalid_argument("GPU constraint entry contains an unknown resource");
                const auto port = static_cast<Port>(found - std::begin(Ports));
                if (actual & Bit(port))
                    throw std::invalid_argument("GPU constraint entry repeats a resource");
                ValidateLayout(port, resource);
                actual |= Bit(port);
                storageCount += resource.binding.type == rhi::BindingType::StorageBuffer;
                program.ports.push_back(port);
                bindings.push_back({resource.binding.binding, resource.binding.type, resource.binding.readOnlyStorage});
            }
            if (actual != EntryPorts[i] || code->bindings.size() != program.ports.size() || storageCount > 8u)
                throw std::invalid_argument("GPU constraint entry differs from frozen resource mask");
            if (target.target == ShaderCodeTarget::SpirV) {
                if (code->code.empty() || code->code.size() % sizeof(uint32_t))
                    throw std::invalid_argument("GPU constraint SPIR-V has invalid word extent");
                std::vector<uint32_t> words(code->code.size() / sizeof(uint32_t));
                std::memcpy(words.data(), code->code.data(), code->code.size());
                program.kernel = std::make_shared<rhi::ComputeKernel>(
                    host, rhi::ShaderModuleDesc::FromSpirV(words.data(), words.size()), code->name, std::move(bindings),
                    0);
            } else {
                program.kernel = std::make_shared<rhi::ComputeKernel>(
                    host, rhi::ShaderModuleDesc::FromWgsl(code->code.data(), code->code.size()), code->name,
                    std::move(bindings), 0);
            }
        }
    }
    rhi::ComputeHost host; // Copy of the shared services, never a new queue/device.
    std::array<Program, EntryCount> programs;
};

struct JoltGpuConstraints::Impl::Execution : std::enable_shared_from_this<Execution>
{
    enum class Phase
    {
        Initialize,
        RoundWindow,
        Assemble,
        Done
    };
    struct Operation
    {
        Entry entry;
        uint32_t length, oldBank;
        ConstraintParams config;
    };
    struct Cursor
    {
        size_t operation = 0;
        uint32_t first = 0;
    };
    struct ScanLevel
    {
        uint32_t prefix, totals, capacity;
    };

    Execution(std::shared_ptr<Impl> owner, const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
              const StepParams &parameters,
              std::function<void(std::shared_ptr<const JoltGpuConstraintWorkset>)> callback)
        : implementation(std::move(owner)), contactLease(inputs.GetContacts()), completed(std::move(callback))
    {
        if (!implementation->host.SharesServicesWith(storage.GetHost()) || !completed || !contactLease)
            throw std::invalid_argument("GPU constraints require the accepted tick host/contact workset");
        const uint32_t b = storage.GetBodyCount(), j = storage.GetJointCount(), p = contactLease->patchCount,
                       w = storage.GetWorldCount();
        // Empty scenes bypass the solver owner; a real body domain can contain
        // zero responsive constraints and still yields a validated empty schedule.
        if (!b || !w || parameters.counts.x != b || parameters.counts.y != w ||
            parameters.counts.z != storage.GetShapeCount())
            throw std::invalid_argument("GPU constraint parameters differ from accepted model domain");
        Checked(2ull * (uint64_t(j) + p));
        Checked(4ull * p);
        Checked(12ull * j);
        Checked(2ull * b);
        const uint32_t allRefs = Checked(uint64_t(j) + p);
        frozen[Bodies] = inputs.GetCaptured(Resource::BodyModels);
        frozen[Shapes] = inputs.GetCaptured(Resource::ShapeModels);
        frozen[Worlds] = contactLease->worlds;
        frozen[Joints] = storage.Get(Resource::Joints)->Freeze();
        frozen[Patches] = contactLease->patches ? contactLease->patches->Freeze() : nullptr;
        frozen[Contacts] = contactLease->contacts ? contactLease->contacts->Freeze() : nullptr;
        for (auto port : {Bodies, Shapes, Worlds, Joints, Patches, Contacts})
            ValidateInput(frozen[port]);
        RequireBytes(Bodies, uint64_t(b) * sizeof(BodyModel));
        RequireBytes(Shapes, uint64_t(std::max(1u, storage.GetShapeCount())) * sizeof(ShapeModel));
        RequireBytes(Worlds, uint64_t(w) * sizeof(WorldRange));
        RequireBytes(Joints, (1ull + 2ull * JointRecordWords * j) * sizeof(QueryRecordWord));
        RequireBytes(Contacts, uint64_t(std::max(1u, Checked(4ull * p))) * sizeof(Contact));
        if (frozen[Patches]->GetByteSize() < WordBytes)
            throw std::invalid_argument("GPU constraints require a candidate-bank header");
        config.counts = {b, j, p, w};
        config.limits.y = storage.GetShapeCount();
        config.limits.z = Checked(4ull * p);
        uint64_t words = 0;
        const auto reserve = [&](uint64_t n) {
            const auto base = Checked(words);
            words += n;
            Checked(words);
            return base;
        };
        config.layout.x = reserve(ConstraintScheduleHeaderWords);
        config.layout.y = reserve(1u);
        config.layout.z = reserve(b);
        config.layout.w = reserve(allRefs);
        config.references.x = reserve(2ull * j);
        config.references.y = reserve(2ull * p);
        jointSort = {reserve(j), reserve(j)};
        patchSort = {reserve(p), reserve(p)};
        config.sources = {reserve(j), reserve(p), reserve(b), reserve(b)};
        config.islands = {reserve(2ull * b), reserve(b), reserve(b), reserve(b)};
        config.summary = {reserve(1u), reserve(1u), reserve(1u), reserve(1u)};
        config.round.x = reserve(1u);
        // One reusable hierarchy. Persistent summaries above never alias it.
        uint32_t count = std::max(b, allRefs);
        do {
            const uint32_t blocks = Blocks(count);
            // Level0 exclusive-prefix storage is config.islands.z. Only
            // higher levels need a scratch prefix; reductions need none.
            const uint32_t prefix = hierarchy.empty() ? 0u : reserve(count);
            hierarchy.push_back({prefix, reserve(blocks), count});
            if (blocks == 1u)
                break;
            count = blocks;
        } while (true);
        config.limits.x = Checked(words);
        // references.z/w are set to the actual final merge bank BEFORE any
        // recipe is emitted; seed/finalize see precisely the frozen layout.
        config.references.z = FinalBank(jointSort, j);
        config.references.w = FinalBank(patchSort, p);
        BuildInitialization();
    }

    void ValidateInput(const std::shared_ptr<rhi::ComputeBuffer> &buffer) const
    {
        if (!buffer || !buffer->IsFrozen() || !implementation->host.SharesServicesWith(buffer->GetHost()))
            throw std::invalid_argument("GPU constraints require frozen accepted input leases");
    }
    void RequireBytes(Port p, uint64_t bytes) const
    {
        if (frozen[p]->GetByteSize() != bytes)
            throw std::invalid_argument("GPU constraint input extent differs from accepted model");
    }
    std::shared_ptr<rhi::ComputeBuffer> Allocate(uint64_t bytes)
    {
        if (!bytes || bytes % 4u || bytes > std::numeric_limits<size_t>::max() ||
            !implementation->host.device.GetCapabilities().IsStorageBufferRangeValid(0, bytes))
            throw std::length_error("GPU constraint storage exceeds host/device range");
        return rhi::ComputeBuffer::TryCreatePrivate(
            implementation->host, {bytes / 4u, rhi::ComputeScalarType::UInt32, 1, rhi::BufferUsageFlags::None});
    }
    bool Endpoint(std::shared_ptr<rhi::ComputeBuffer> &buffer, uint64_t bytes)
    {
        if (!buffer)
            buffer = Allocate(bytes);
        return buffer != nullptr;
    }
    bool Capacity()
    {
        return Endpoint(schedule, uint64_t(config.limits.x) * WordBytes) &&
               Endpoint(labels[0], uint64_t(config.counts.x) * 4u) &&
               Endpoint(labels[1], uint64_t(config.counts.x) * 4u);
    }

    static uint32_t FinalBank(std::array<uint32_t, 2> banks, uint32_t count)
    {
        uint32_t bank = 0;
        for (uint64_t width = 1u; width < count; width *= 2u)
            bank ^= 1u;
        return banks[bank];
    }
    void Add(Entry entry, uint32_t length, const ConstraintParams &value, uint32_t bank)
    {
        if (length)
            plan.push_back({entry, length, bank, value});
    }
    void Add(Entry entry, uint32_t length)
    {
        Add(entry, length, config, planBank);
    }
    void Kind(Entry entry, uint32_t kind)
    {
        auto value = config;
        value.select.x = kind;
        Add(entry, kind == ConstraintJoint ? config.counts.y : config.counts.z, value, planBank);
    }

    // Only logarithmically many operation descriptors are materialized. A large
    // domain is sliced by the packet cursor; no CPU arrays of labels/refs/ranges.
    // Empty input leaves its distinct, already initialized total word zero.
    void Reduce(uint32_t input, uint32_t count, uint32_t mode, uint32_t destination)
    {
        if (!count)
            return;
        uint32_t level = 0;
        while (true) {
            auto value = config;
            const uint32_t blocks = Blocks(count);
            const uint32_t total = blocks == 1u ? destination : hierarchy[level].totals;
            value.scan = {input, 0u, total, count};
            value.scanParent.y = mode;
            Add(ReduceBlocks, blocks, value, planBank);
            if (blocks == 1u)
                return;
            input = total;
            count = blocks;
            ++level;
        }
    }
    void ExclusiveScan(uint32_t input, uint32_t count, uint32_t prefix, uint32_t destination)
    {
        if (!count)
            return;
        struct Level
        {
            uint32_t count, input, prefix, total;
        };
        std::vector<Level> scan;
        uint32_t level = 0;
        while (true) {
            const uint32_t blocks = Blocks(count), total = blocks == 1u ? destination : hierarchy[level].totals;
            scan.push_back({count, input, level == 0u ? prefix : hierarchy[level].prefix, total});
            if (blocks == 1u)
                break;
            input = total;
            count = blocks;
            ++level;
        }
        for (const auto &l : scan) {
            auto value = config;
            value.scan = {l.input, l.prefix, l.total, l.count};
            value.scanParent.y = 0u;
            Add(ScanBlocks, Blocks(l.count), value, planBank);
        }
        for (size_t i = scan.size() - 1u; i != 0u; --i) {
            const auto &child = scan[i - 1u];
            auto value = config;
            value.scan = {child.input, child.prefix, child.total, child.count};
            value.scanParent = {scan[i].prefix, 0u, 0u, 0u};
            Add(AddOffsets, child.count, value, planBank);
        }
    }
    void Sort(uint32_t kind, std::array<uint32_t, 2> banks, uint32_t count)
    {
        auto value = config;
        value.select = {kind, banks[0], 0u, 0u};
        Add(SeedSort, count, value, planBank);
        uint32_t source = 0;
        for (uint64_t width = 1u; width < count; width *= 2u) {
            value.merge = {banks[source], banks[source ^ 1u], count, Checked(width)};
            Add(MergeItems, count, value, planBank);
            source ^= 1u;
        }
    }
    void ResetPlan(Phase next)
    {
        phase = next;
        plan.clear();
        cursor = {};
        planBank = oldBank;
    }
    void BuildInitialization()
    {
        ResetPlan(Phase::Initialize);
        Add(InitWords, config.limits.x);
        Add(Seed, 1u);
        Add(BuildJointRefs, config.counts.y);
        Add(BuildPatchRefs, config.counts.z);
        Add(InitLabels, config.counts.x);
    }
    void BuildRoundWindow()
    {
        ResetPlan(Phase::RoundWindow);
        const uint32_t remaining = config.counts.x - scheduledRounds;
        windowCount = std::min(scheduledRounds == 0u ? 1u : ConstraintRoundWindow, remaining);
        if (!windowCount)
            throw std::runtime_error("GPU constraint CC exhausted its proven body-slot bound");
        for (uint32_t round = 0; round < windowCount; ++round) {
            Add(ShortcutLabels, config.counts.x);
            Kind(PropagateEdges, ConstraintJoint);
            Kind(PropagateEdges, ConstraintPatch);
            Add(CheckRound, config.counts.x);
            Reduce(config.layout.z, config.counts.x, 0u, config.round.x);
            Add(CommitRound, 1u);
            // This records the binding parity for subsequent operations only.
            // The committed cursor/parity changes after the real receipt.
            planBank ^= 1u;
        }
    }
    void BuildAssembly()
    {
        ResetPlan(Phase::Assemble);
        Add(ValidateLabels, config.counts.x);
        Reduce(config.layout.z, config.counts.x, 0u, config.summary.w);
        Kind(ValidateEdges, ConstraintJoint);
        Kind(ValidateEdges, ConstraintPatch);
        const uint32_t refs = Checked(uint64_t(config.counts.y) + config.counts.z);
        Reduce(config.layout.w, refs, 0u, config.summary.z);
        Add(AcceptComponents, 1u);
        Sort(ConstraintJoint, jointSort, config.counts.y);
        Sort(ConstraintPatch, patchSort, config.counts.z);
        Add(ClearRootFlags, config.counts.x);
        Kind(MarkRoots, ConstraintJoint);
        Kind(MarkRoots, ConstraintPatch);
        Add(BuildRootRows, config.counts.x);
        ExclusiveScan(config.islands.y, config.counts.x, config.islands.z, config.summary.x);
        Kind(WriteSources, ConstraintJoint);
        Kind(WriteSources, ConstraintPatch);
        Add(WriteIslands, config.counts.x);
        Reduce(config.islands.w, config.counts.x, 1u, config.summary.y);
        Add(Finalize, 1u);
    }

    bool Emit(const Operation &op, uint32_t first, uint32_t count, std::vector<rhi::ComputeBufferUpdate> &uploads,
              std::vector<rhi::ComputeDispatchDesc> &dispatches)
    {
        const bool collective = Collective(op.entry);
        // The block limit is <=64; each packet uses at most 32 blocks.
        const uint32_t limit = collective ? ConstraintBlockLimit : Summary(op.entry) ? 1u : ConstraintItemLimit;
        if (!count || count > limit || first >= op.length || count > op.length - first)
            throw std::logic_error("GPU constraints emitted an invalid bounded chunk");
        // Keep partial uniform acquisition across managed pressure. No cursor
        // or phase advances on a null-packet continuation.
        if (!Endpoint(pendingParams, sizeof(ConstraintParams)) ||
            ((EntryPorts[op.entry] & K) && !Endpoint(pendingWork, sizeof(StageWork))))
            return false;
        uploads.push_back({pendingParams, 0u, Bytes(op.config)});
        const StageWork work{{first, count, 0u, 0u}};
        if (EntryPorts[op.entry] & K)
            uploads.push_back({pendingWork, 0u, Bytes(work)});
        rhi::ComputeDispatchDesc dispatch;
        const auto &program = implementation->programs[op.entry];
        dispatch.kernel = program.kernel;
        dispatch.groupCountX = collective ? count : Summary(op.entry) ? 1u : Blocks(count);
        if (dispatch.groupCountX > implementation->host.device.GetCapabilities().limits.maxComputeWorkgroupCount[0])
            throw std::length_error("GPU constraint chunk exceeds dispatch dimensions");
        rhi::ComputeDispatchProof proof;
        for (const auto port : program.ports) {
            const auto &buffer = port == Params       ? pendingParams
                                 : port == Work       ? pendingWork
                                 : port == Schedule   ? schedule
                                 : port == LabelsOld  ? labels[op.oldBank]
                                 : port == LabelsNext ? labels[op.oldBank ^ 1u]
                                                      : frozen[port];
            if (!buffer)
                throw std::logic_error("GPU constraint dispatch has an absent endpoint");
            const uint32_t binding = static_cast<uint32_t>(dispatch.buffers.size());
            dispatch.buffers.push_back(buffer);
            const bool initSchedule = op.entry == InitWords && port == Schedule;
            const bool initLabel = op.entry == InitLabels && (port == LabelsOld || port == LabelsNext);
            const bool uniqueLabel = port == LabelsNext && (op.entry == ShortcutLabels || op.entry == ClearRootFlags);
            const bool writes = initSchedule || initLabel || uniqueLabel;
            // After full label initialization the read/write version is live
            // throughout CC. Reads never Freeze it, so ping-pong is not COW.
            const bool labelReadOnly = port == LabelsOld && op.entry != InitLabels;
            const bool nextReadOnly = port == LabelsNext && (op.entry == CheckRound || op.entry == BuildRootRows);
            const bool scheduleReadOnly =
                port == Schedule && (op.entry == ShortcutLabels || op.entry == PropagateEdges || op.entry == MarkRoots);
            const bool mutablePort = port == Schedule || port == LabelsOld || port == LabelsNext;
            dispatch.bufferAccesses.push_back(writes ? rhi::ComputeBufferAccess::Write
                                              : !mutablePort || labelReadOnly || nextReadOnly || scheduleReadOnly
                                                  ? rhi::ComputeBufferAccess::Read
                                                  : rhi::ComputeBufferAccess::ReadWrite);
            if (writes) {
                const uint64_t stride = port == Schedule ? WordBytes : 4u;
                proof.writes.push_back({binding, uint64_t(first) * stride, uint64_t(count) * stride});
            } else
                proof.reads.push_back({binding, 0u, buffer->GetByteSize()});
        }
        dispatch.proof = std::move(proof);
        dispatches.push_back(std::move(dispatch));
        pendingParams.reset();
        pendingWork.reset();
        return true;
    }

    void DecodeRound(const std::shared_ptr<rhi::PreparedComputeBatch> &packet)
    {
        UInt4 control{};
        auto read = packet->Readback(0u);
        if (!read || !read->IsComplete())
            throw std::logic_error("GPU CC control needs its completed source receipt");
        read->ReadInto(&control, sizeof(control));
        const uint32_t previousScheduled = scheduledRounds;
        scheduledRounds += windowCount;
        oldBank = planBank;
        if (control.z || control.w || control.x > config.counts.x || control.y > scheduledRounds ||
            control.y <= previousScheduled || (control.x && control.y != scheduledRounds))
            throw std::runtime_error("GPU constraint CC failed its monotone convergence/status contract (status " +
                                     std::to_string(control.z) + ")");
        if (control.x == 0u)
            BuildAssembly();
        else
            BuildRoundWindow();
    }
    void Install(const std::shared_ptr<rhi::PreparedComputeBatch> &packet)
    {
        std::array<UInt4, 2> result{};
        auto read = packet->Readback(0u);
        if (!read || !read->IsComplete())
            throw std::logic_error("GPU schedule install needs its completed source receipt");
        read->ReadInto(result.data(), sizeof(result));
        const auto &counts = result[0], &items = result[1];
        if (counts.w || counts.x > config.counts.x || counts.y > counts.x || counts.z > counts.x ||
            items.x > config.counts.y || items.y > config.counts.z || items.z > items.x || items.w > items.y ||
            (counts.y == 0u) != (items.x == 0u) || (counts.z == 0u) != (items.y == 0u) || counts.y > items.x ||
            counts.z > items.y || (items.z == 0u) != (items.x == 0u) || (items.w == 0u) != (items.y == 0u))
            throw std::runtime_error("GPU constraint schedule failed structural acceptance (status " +
                                     std::to_string(counts.w) + ")");
        auto value = std::make_shared<JoltGpuConstraintWorkset>();
        value->islandCount = counts.x;
        value->jointIslandCount = counts.y;
        value->contactIslandCount = counts.z;
        value->activeJoints = items.x;
        value->activePatches = items.y;
        value->maxJointItems = items.z;
        value->maxPatchItems = items.w;
        value->schedule = schedule->Freeze();
        phase = Phase::Done;
        auto callback = std::move(completed);
        callback(std::move(value));
    }
    JoltGpuControlWork Next()
    {
        if (phase == Phase::Done)
            return {};
        if (!Capacity())
            return {nullptr, [self = shared_from_this()] { return self->Next(); }};
        Cursor next = cursor;
        std::vector<rhi::ComputeBufferUpdate> uploads;
        std::vector<rhi::ComputeDispatchDesc> dispatches;
        while (next.operation < plan.size() && dispatches.size() < MaxDispatchesPerPacket) {
            const auto &op = plan[next.operation];
            const uint32_t limit = Collective(op.entry) ? ConstraintBlockLimit
                                   : Summary(op.entry)  ? 1u
                                                        : ConstraintItemLimit;
            const uint32_t count = std::min(limit, op.length - next.first);
            if (!Emit(op, next.first, count, uploads, dispatches))
                break;
            next.first += count;
            if (next.first == op.length) {
                next.first = 0u;
                ++next.operation;
            }
        }
        if (dispatches.empty())
            return {nullptr, [self = shared_from_this()] { return self->Next(); }};
        const bool boundary = next.operation == plan.size();
        const auto submittedPhase = phase;
        std::vector<rhi::ComputeBufferRead> reads;
        if (boundary && phase == Phase::RoundWindow)
            reads.push_back({schedule, uint64_t(config.layout.y) * WordBytes, WordBytes});
        if (boundary && phase == Phase::Assemble)
            reads.push_back({schedule, uint64_t(config.layout.x) * WordBytes, 2u * WordBytes});
        auto packet =
            rhi::PrepareComputeBatch(implementation->host, std::move(uploads), std::move(dispatches), std::move(reads));
        return {packet, [self = shared_from_this(), packet, next, boundary, submittedPhase]() -> JoltGpuControlWork {
                    // The existing owner calls once after this exact packet's receipt.
                    // No advancement or repeated Setup/Warm occurs on busy admission.
                    self->cursor = next;
                    if (boundary) {
                        if (submittedPhase == Phase::Initialize)
                            self->BuildRoundWindow();
                        else if (submittedPhase == Phase::RoundWindow)
                            self->DecodeRound(packet);
                        else if (submittedPhase == Phase::Assemble) {
                            self->Install(packet);
                            return {};
                        }
                    }
                    return self->Next();
                }};
    }

    std::shared_ptr<Impl> implementation;
    std::shared_ptr<const JoltGpuContactWorkset> contactLease;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> frozen;
    std::shared_ptr<rhi::ComputeBuffer> schedule, pendingParams, pendingWork;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, 2> labels;
    std::function<void(std::shared_ptr<const JoltGpuConstraintWorkset>)> completed;
    ConstraintParams config{};
    std::array<uint32_t, 2> jointSort{}, patchSort{};
    std::vector<ScanLevel> hierarchy;
    std::vector<Operation> plan;
    Cursor cursor;
    Phase phase = Phase::Initialize;
    uint32_t oldBank = 0u, planBank = 0u, scheduledRounds = 0u, windowCount = 0u;
};

JoltGpuConstraints::JoltGpuConstraints(rhi::ComputeHost &host, const ShaderTargetCode &target)
    : m_impl(std::make_shared<Impl>(host, target))
{
}
JoltGpuControlWork
JoltGpuConstraints::Prepare(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                            const JPH::GPU::StepParams &parameters,
                            std::function<void(std::shared_ptr<const JoltGpuConstraintWorkset>)> completed)
{
    return std::make_shared<Impl::Execution>(m_impl, storage, inputs, parameters, std::move(completed))->Next();
}
} // namespace infernux
