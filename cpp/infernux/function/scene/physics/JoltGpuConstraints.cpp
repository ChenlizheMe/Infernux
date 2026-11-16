#include "JoltGpuConstraints.h"
#include <Jolt/GPU/ConstraintTypes.h>
#include "JoltGpuSleepLayout.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>
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
    Sleeps,
    TopologyArguments,
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
    WriteBodyRoots,
    FinalizeTopology,
    CommitRoundAndGateTopology,
    EntryCount
};
constexpr std::array<std::string_view, EntryCount> EntryNames{
    "constraint_init_words",        "constraint_seed",
    "constraint_build_joint_refs",  "constraint_build_patch_refs",
    "constraint_init_labels",       "constraint_shortcut_labels",
    "constraint_propagate_edges",   "constraint_check_round",
    "constraint_commit_round",      "constraint_validate_labels",
    "constraint_validate_edges",    "constraint_accept_components",
    "constraint_scan_blocks",       "constraint_reduce_blocks",
    "constraint_add_scan_offsets",  "constraint_seed_sort",
    "constraint_merge_items",       "constraint_clear_root_flags",
    "constraint_mark_roots",        "constraint_build_root_rows",
    "constraint_write_sources",     "constraint_write_islands",
    "constraint_finalize",          "constraint_write_body_roots",
    "constraint_finalize_topology", "constraint_commit_round_and_gate_topology"};
constexpr uint32_t Bit(Port port)
{
    return 1u << port;
}
constexpr uint32_t P = Bit(Params), K = Bit(Work), S = Bit(Schedule), B = Bit(Bodies), W = Bit(Worlds),
                   O = Bit(LabelsOld), N = Bit(LabelsNext);
constexpr std::array<uint32_t, EntryCount> EntryPorts{P | K | S,
                                                      P | S,
                                                      P | K | S | B | W | Bit(Joints) | Bit(Sleeps),
                                                      P | K | S | B | W | Bit(Shapes) | Bit(Patches) | Bit(Contacts) |
                                                          Bit(Sleeps),
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
                                                      P | S,
                                                      P | K | S | O,
                                                      P | S,
                                                      P | K | S | Bit(TopologyArguments)};

struct PortLayout
{
    std::string_view name;
    uint64_t stride;
    std::array<std::string_view, 14> fields;
    uint32_t integers;
    bool readOnly;
};
constexpr PortLayout Ports[] = {
    {"constraint_params",
     sizeof(ConstraintParams),
     {"counts", "limits", "layout", "references", "sources", "islands", "scan", "scanParent", "merge", "select",
      "summary", "round", "selection", "selectedCounts"},
     0x3fffu,
     true},
    {"stage_work", sizeof(StageWork), {"range"}, 1u, true},
    {"body_models", sizeof(BodyModel), {"meta", "mass", "invInertiaDiagonal", "inertiaRotation", "damping"}, 1u, true},
    {"shape_models",
     sizeof(ShapeModel),
     {"meta", "geometry", "localPositionCOM", "localRotation", "material", "surface", "source", "geometryReference"},
     (1u << 0) | (1u << 5) | (1u << 6) | (1u << 7),
     true},
    {"worlds", sizeof(WorldRange), {"bodies", "leaves"}, 3u, true},
    {"joint_stream", sizeof(QueryRecordWord), {"value"}, 1u, true},
    {"contact_patches", sizeof(ContactPatch), {"leaves"}, 1u, true},
    {"contacts", sizeof(Contact), {"meta", "normalDepth", "localAnchorA", "localAnchorB", "impulse"}, 1u, true},
    {"labels_old", sizeof(uint32_t), {}, 0u, false},
    {"labels_next", sizeof(uint32_t), {}, 0u, false},
    {"constraint_schedule", sizeof(ConstraintScheduleWord), {"value"}, 1u, false},
    {"body_sleep", sizeof(BodySleepRecord), {"author", "runtime"}, 0u, true},
    {"topology_arguments", sizeof(ConstraintScheduleWord), {"value"}, 1u, false}};
static_assert(std::size(Ports) == PortCount);
constexpr uint32_t MaxDispatchesPerPacket = 16u;
constexpr uint32_t MaxTopologyArguments = 15u;
constexpr uint64_t WordBytes = sizeof(ConstraintScheduleWord);
static_assert(MaxDispatchesPerPacket - 1u == MaxTopologyArguments);

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
    return e == Seed || e == CommitRound || e == AcceptComponents || e == Finalize || e == FinalizeTopology ||
           e == CommitRoundAndGateTopology;
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
    if (port == Sleeps) {
        if (resource.binding.type != rhi::BindingType::StorageBuffer || !resource.binding.readOnlyStorage ||
            resource.elementStride != sizeof(BodySleepRecord))
            throw std::invalid_argument("GPU constraint sleep input differs from its shared record");
        ValidateJoltGpuSleepLayout(value);
        return;
    }
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
        RoundWindow,
        Assemble,
        TopologyReady,
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
    struct Uniforms
    {
        std::shared_ptr<rhi::ComputeBuffer> parameters, work;
    };

    Execution(std::shared_ptr<Impl> owner, const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
              const StepParams &parameters,
              std::function<void(std::shared_ptr<const JoltGpuConstraintWorkset>)> callback, bool onlyTopology = false,
              std::function<void(std::shared_ptr<JoltGpuConstraintTopology>)> topologyCallback = {})
        : implementation(std::move(owner)), contactLease(inputs.GetContacts()), completed(std::move(callback)),
          completedTopology(std::move(topologyCallback)), topologyOnly(onlyTopology)
    {
        if (!implementation->host.SharesServicesWith(storage.GetHost()) || (!completed && !completedTopology) ||
            !contactLease)
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
        // Joint directory/activity flags stay fixed under this arena writer.
        // Warm/rotor retirement changes neither field read by the schedule.
        frozen[Joints] = storage.Get(Resource::Joints);
        frozen[Patches] = contactLease->patches ? contactLease->patches->Freeze() : nullptr;
        frozen[Contacts] = contactLease->contacts ? contactLease->contacts->Freeze() : nullptr;
        // Topology mode does not read runtime activity. Borrow this live source
        // endpoint so waking after CC cannot COW a stale captured reader.
        frozen[Sleeps] = storage.Get(Resource::SleepStates);
        if (!frozen[Joints] || !implementation->host.SharesServicesWith(frozen[Joints]->GetHost()))
            throw std::invalid_argument("GPU constraints require their exclusive writer joint stream");
        for (auto port : {Bodies, Shapes, Worlds, Patches, Contacts})
            ValidateInput(frozen[port]);
        RequireBytes(Bodies, uint64_t(b) * sizeof(BodyModel));
        RequireBytes(Shapes, uint64_t(std::max(1u, storage.GetShapeCount())) * sizeof(ShapeModel));
        RequireBytes(Worlds, uint64_t(w) * sizeof(WorldRange));
        RequireBytes(Joints, storage.GetJointLayout()->ByteSize());
        if (frozen[Contacts]->GetByteSize() < uint64_t(std::max(1u, Checked(4ull * p))) * sizeof(Contact) ||
            frozen[Contacts]->GetByteSize() % sizeof(Contact))
            throw std::invalid_argument("GPU constraints require initialized contact capacity covering logical 4C");
        if (!frozen[Sleeps] || !implementation->host.SharesServicesWith(frozen[Sleeps]->GetHost()))
            throw std::invalid_argument("GPU constraints require the selected sleep source");
        RequireBytes(Sleeps, uint64_t(b) * sizeof(BodySleepRecord));
        if (frozen[Patches]->GetByteSize() < WordBytes)
            throw std::invalid_argument("GPU constraints require a candidate-bank header");
        config.counts = {b, j, p, w};
        config.selection = parameters.selection;
        config.selectedCounts = parameters.selectedCounts;
        config.limits.y = storage.GetShapeCount();
        config.limits.z = Checked(4ull * p);
        config.select.z = topologyOnly ? 0u : 1u;
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
        config.limits.w = reserve(b);
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
        BuildRoundWindow();
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
    std::shared_ptr<rhi::ComputeBuffer> Allocate(uint64_t bytes,
                                                 rhi::BufferUsageFlags usage = rhi::BufferUsageFlags::None)
    {
        if (!bytes || bytes % 4u || bytes > std::numeric_limits<size_t>::max() ||
            !implementation->host.device.GetCapabilities().IsStorageBufferRangeValid(0, bytes))
            throw std::length_error("GPU constraint storage exceeds host/device range");
        return rhi::ComputeBuffer::TryCreatePrivate(implementation->host,
                                                    {bytes / 4u, rhi::ComputeScalarType::UInt32, 1, usage});
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
               Endpoint(labels[0], (uint64_t(config.counts.x) + 1u) * 4u) &&
               Endpoint(labels[1], (uint64_t(config.counts.x) + 1u) * 4u);
    }

    static uint32_t FinalBank(std::array<uint32_t, 2> banks, uint32_t count)
    {
        uint32_t bank = 0;
        for (uint64_t width = 1u; width < count; width *= 2u)
            bank ^= 1u;
        return banks[bank];
    }
    void Append(std::vector<Operation> &operations, Entry entry, uint32_t length, const ConstraintParams &value,
                uint32_t bank)
    {
        if (length)
            operations.push_back({entry, length, bank, value});
    }
    void Add(Entry entry, uint32_t length, const ConstraintParams &value, uint32_t bank)
    {
        Append(plan, entry, length, value, bank);
    }
    void Add(Entry entry, uint32_t length)
    {
        Add(entry, length, config, planBank);
    }
    void AppendKind(std::vector<Operation> &operations, Entry entry, uint32_t kind)
    {
        auto value = config;
        value.select.x = kind;
        Append(operations, entry, kind == ConstraintJoint ? config.counts.y : config.counts.z, value, planBank);
    }
    void Kind(Entry entry, uint32_t kind)
    {
        AppendKind(plan, entry, kind);
    }

    // Only logarithmically many operation descriptors are materialized. A large
    // domain is sliced by the packet cursor; no CPU arrays of labels/refs/ranges.
    // Empty input leaves its distinct, already initialized total word zero.
    void AppendReduction(std::vector<Operation> &operations, uint32_t input, uint32_t count, uint32_t mode,
                         uint32_t destination)
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
            Append(operations, ReduceBlocks, blocks, value, planBank);
            if (blocks == 1u)
                return;
            input = total;
            count = blocks;
            ++level;
        }
    }
    void Reduce(uint32_t input, uint32_t count, uint32_t mode, uint32_t destination)
    {
        AppendReduction(plan, input, count, mode, destination);
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
    void BuildRoundWindow()
    {
        ResetPlan(Phase::RoundWindow);
        const bool firstWindow = scheduledRounds == 0u;
        const uint32_t remaining = config.counts.x - scheduledRounds;
        windowCount = std::min(firstWindow ? 1u : ConstraintRoundWindow, remaining);
        if (!windowCount)
            throw std::runtime_error("GPU constraint CC exhausted its proven body-slot bound");
        if (firstWindow) {
            // The first round can consume the initialization writes in queue
            // order. Only the convergence record needs a host receipt.
            Add(InitWords, config.limits.x);
            Add(Seed, 1u);
            Add(BuildJointRefs, config.counts.y);
            Add(BuildPatchRefs, config.counts.z);
            Add(InitLabels, config.counts.x);
        }
        for (uint32_t round = 0; round < windowCount; ++round) {
            Add(ShortcutLabels, config.counts.x);
            Kind(PropagateEdges, ConstraintJoint);
            Kind(PropagateEdges, ConstraintPatch);
            Add(CheckRound, config.counts.x);
            Reduce(config.layout.z, config.counts.x, 0u, config.round.x);
            Add(CommitRound, 1u);
            // This records the binding parity for subsequent operations only.
            // Committed CC parity changes only after the completed summary.
            planBank ^= 1u;
        }
    }
    void AppendComponentValidation(std::vector<Operation> &operations)
    {
        Append(operations, ValidateLabels, config.counts.x, config, planBank);
        AppendReduction(operations, config.layout.z, config.counts.x, 0u, config.summary.w);
        AppendKind(operations, ValidateEdges, ConstraintJoint);
        AppendKind(operations, ValidateEdges, ConstraintPatch);
        const uint32_t refs = Checked(uint64_t(config.counts.y) + config.counts.z);
        AppendReduction(operations, config.layout.w, refs, 0u, config.summary.z);
        Append(operations, AcceptComponents, 1u, config, planBank);
        Append(operations, WriteBodyRoots, config.counts.x, config, planBank);
        if (topologyOnly)
            Append(operations, FinalizeTopology, 1u, config, planBank);
    }
    void BuildAssembly()
    {
        ResetPlan(Phase::Assemble);
        if (resumingNumerical) {
            Add(BuildJointRefs, config.counts.y);
            Add(BuildPatchRefs, config.counts.z);
        }
        AppendComponentValidation(plan);
        if (topologyOnly)
            return;
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

    static uint32_t ChunkLimit(Entry entry)
    {
        return Collective(entry) ? ConstraintBlockLimit : Summary(entry) ? 1u : ConstraintItemLimit;
    }

    bool Emit(const Operation &op, uint32_t first, uint32_t count, Uniforms &uniforms,
              std::vector<rhi::ComputeBufferUpdate> &uploads, std::vector<rhi::ComputeDispatchDesc> &dispatches,
              const std::shared_ptr<rhi::ComputeBuffer> &arguments = {})
    {
        const bool collective = Collective(op.entry);
        // The block limit is <=64; each packet uses at most 32 blocks.
        const uint32_t limit = ChunkLimit(op.entry);
        if (!count || count > limit || first >= op.length || count > op.length - first)
            throw std::logic_error("GPU constraints emitted an invalid bounded chunk");
        // Keep partial uniform acquisition across managed pressure. No cursor
        // or phase advances on a null-packet continuation.
        if (!Endpoint(uniforms.parameters, sizeof(ConstraintParams)) ||
            ((EntryPorts[op.entry] & K) && !Endpoint(uniforms.work, sizeof(StageWork))))
            return false;
        uploads.push_back({uniforms.parameters, 0u, Bytes(op.config)});
        StageWork work{{first, count, 0u, 0u}};
        if (arguments)
            work.range = {scheduledRounds, static_cast<uint32_t>(arguments->GetByteSize() / WordBytes),
                          scheduledRounds + windowCount, 0u};
        if (EntryPorts[op.entry] & K)
            uploads.push_back({uniforms.work, 0u, Bytes(work)});
        rhi::ComputeDispatchDesc dispatch;
        const auto &program = implementation->programs[op.entry];
        dispatch.kernel = program.kernel;
        dispatch.groupCountX = collective ? count : Summary(op.entry) ? 1u : Blocks(count);
        if (dispatch.groupCountX > implementation->host.device.GetCapabilities().limits.maxComputeWorkgroupCount[0])
            throw std::length_error("GPU constraint chunk exceeds dispatch dimensions");
        rhi::ComputeDispatchProof proof;
        for (const auto port : program.ports) {
            const auto &buffer = port == Params              ? uniforms.parameters
                                 : port == Work              ? uniforms.work
                                 : port == Schedule          ? schedule
                                 : port == LabelsOld         ? labels[op.oldBank]
                                 : port == LabelsNext        ? labels[op.oldBank ^ 1u]
                                 : port == TopologyArguments ? arguments
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
            const bool mutablePort =
                port == Schedule || port == LabelsOld || port == LabelsNext || port == TopologyArguments;
            dispatch.bufferAccesses.push_back(writes ? rhi::ComputeBufferAccess::Write
                                              : !mutablePort || labelReadOnly || nextReadOnly || scheduleReadOnly
                                                  ? rhi::ComputeBufferAccess::Read
                                                  : rhi::ComputeBufferAccess::ReadWrite);
            if (writes) {
                const uint64_t stride = port == Schedule ? WordBytes : 4u;
                proof.writes.push_back({binding, uint64_t(first) * stride, uint64_t(count) * stride});
            } else
                proof.reads.push_back(
                    {binding, 0u,
                     port == LabelsOld || port == LabelsNext ? uint64_t(config.counts.x) * 4u : buffer->GetByteSize()});
        }
        dispatch.proof = std::move(proof);
        dispatches.push_back(std::move(dispatch));
        uniforms = {};
        return true;
    }

    bool TryAppendTopologyTail(const Operation &commit, size_t dispatchLimit,
                               std::vector<rhi::ComputeBufferUpdate> &uploads,
                               std::vector<rhi::ComputeDispatchDesc> &dispatches)
    {
        if (!topologyOnly || dispatches.size() + 1u >= dispatchLimit)
            return false;
        std::vector<Operation> tail;
        AppendComponentValidation(tail);
        struct Chunk
        {
            const Operation *operation;
            uint32_t first, count;
        };
        std::array<Chunk, MaxTopologyArguments> chunks;
        const size_t available = std::min(chunks.size(), dispatchLimit - 1u - dispatches.size());
        size_t argumentCount = 0;
        for (const auto &op : tail) {
            const uint32_t limit = ChunkLimit(op.entry);
            for (uint32_t first = 0u; first < op.length;) {
                if (argumentCount == available)
                    return false;
                const uint32_t count = std::min(limit, op.length - first);
                chunks[argumentCount++] = {&op, first, count};
                first += count;
            }
        }
        const auto arguments =
            Allocate(argumentCount * WordBytes, rhi::BufferUsageFlags::Storage | rhi::BufferUsageFlags::Indirect);
        if (!arguments)
            return false;

        // Allocation refusal leaves the ordinary commit and its exact cursor
        // available. Only append after the entire optional suffix is ready.
        Uniforms uniforms;
        std::vector<rhi::ComputeBufferUpdate> tailUploads;
        std::vector<rhi::ComputeDispatchDesc> tailDispatches;
        if (!Emit({CommitRoundAndGateTopology, commit.length, commit.oldBank, commit.config}, 0u, 1u, uniforms,
                  tailUploads, tailDispatches, arguments))
            return false;
        std::vector<uint8_t> bytes(argumentCount * WordBytes);
        for (size_t i = 0; i < argumentCount; ++i) {
            const auto &chunk = chunks[i];
            if (!Emit(*chunk.operation, chunk.first, chunk.count, uniforms, tailUploads, tailDispatches))
                return false;
            auto &unit = tailDispatches.back();
            const auto offset = i * WordBytes;
            unit.indirectArguments = rhi::ComputeIndirectArguments{arguments, offset};
            // Only the GPU gate can turn on the bounded group count.
            const UInt4 record{0u, 1u, 1u, unit.groupCountX};
            std::memcpy(bytes.data() + offset, &record, sizeof(record));
        }
        tailUploads.push_back({arguments, 0u, std::move(bytes)});
        uploads.insert(uploads.end(), std::make_move_iterator(tailUploads.begin()),
                       std::make_move_iterator(tailUploads.end()));
        dispatches.insert(dispatches.end(), std::make_move_iterator(tailDispatches.begin()),
                          std::make_move_iterator(tailDispatches.end()));
        return true;
    }

    void AcceptRoundControl(const UInt4 &control, bool gated = false)
    {
        const uint32_t proposed = scheduledRounds + windowCount;
        const bool topologyAccepted = gated && control.x == 0u && control.w == 1u;
        if (control.z || control.w != uint32_t(topologyAccepted) || control.x > config.counts.x ||
            control.y > proposed || control.y <= scheduledRounds || (control.x && control.y != proposed))
            throw std::runtime_error("GPU constraint CC failed its monotone convergence/status contract (status " +
                                     std::to_string(control.z) + ")");
        if (gated && control.x == 0u && !topologyAccepted)
            throw std::runtime_error("GPU sleep topology failed its independent CC acceptance");
        scheduledRounds = proposed;
        oldBank = planBank;
    }
    void DecodeRound(const std::shared_ptr<rhi::PreparedComputeBatch> &packet)
    {
        UInt4 control{};
        auto read = packet->Readback(0u);
        if (!read || !read->IsComplete())
            throw std::logic_error("GPU CC control needs its completed source receipt");
        read->ReadInto(&control, sizeof(control));
        AcceptRoundControl(control);
        if (control.x == 0u)
            BuildAssembly();
        else
            BuildRoundWindow();
    }
    static void ValidateTopologyHeader(const UInt4 &counts, const UInt4 &items)
    {
        if (counts.w != ConstraintNotReady || counts.x || counts.y || counts.z || items.x || items.y || items.z ||
            items.w)
            throw std::runtime_error("GPU sleep topology failed its independent CC acceptance");
    }
    void DecodeGatedRound(const std::shared_ptr<rhi::PreparedComputeBatch> &packet)
    {
        std::array<UInt4, ConstraintScheduleHeaderWords + 1u> result{};
        auto read = packet->Readback(0u);
        if (!read || !read->IsComplete())
            throw std::logic_error("GPU CC topology needs its completed source receipt");
        read->ReadInto(result.data(), sizeof(result));
        const auto &control = result.back();
        // AcceptComponents preserves x/y and only ORs z. Its w=1 distinguishes
        // the accepted tail from the identical seeded topology header.
        ValidateTopologyHeader(result[0], result[1]);
        AcceptRoundControl(control, true);
        if (control.x)
            BuildRoundWindow();
        else
            PublishTopology();
    }
    void PublishTopology()
    {
        auto value = std::make_shared<JoltGpuConstraintTopology>();
        value->bodyRootFirst = config.limits.w;
        value->bodySlots = config.counts.x;
        value->schedule = schedule;
        value->rootVotes = labels[oldBank ^ 1u];
        value->execution = shared_from_this();
        // Setup must write/read its actual new generations.
        frozen[Joints].reset();
        frozen[Contacts].reset();
        phase = Phase::TopologyReady;
        auto callback = std::move(completedTopology);
        callback(std::move(value));
    }
    void Install(const UInt4 &counts, const UInt4 &items)
    {
        if (topologyOnly) {
            ValidateTopologyHeader(counts, items);
            PublishTopology();
            return;
        }
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
        value->bodyRootFirst = config.limits.w;
        value->bodySlots = config.counts.x;
        value->rootVotes = labels[oldBank ^ 1u];
        frozen[Joints].reset();
        frozen[Contacts].reset();
        frozen[Sleeps].reset();
        phase = Phase::Done;
        auto callback = std::move(completed);
        callback(std::move(value));
    }
    void DecodeAssembly(const std::shared_ptr<rhi::PreparedComputeBatch> &packet)
    {
        std::array<UInt4, 2> result{};
        auto read = packet->Readback(0u);
        if (!read || !read->IsComplete())
            throw std::logic_error("GPU schedule install needs its completed source receipt");
        read->ReadInto(result.data(), sizeof(result));
        Install(result[0], result[1]);
    }
    JoltGpuControlWork Next(size_t dispatchLimit = MaxDispatchesPerPacket)
    {
        if (phase == Phase::Done || phase == Phase::TopologyReady)
            return {};
        if (!Capacity())
            return {nullptr, [self = shared_from_this(), dispatchLimit] { return self->Next(dispatchLimit); }};
        Cursor next = cursor;
        std::vector<rhi::ComputeBufferUpdate> uploads;
        std::vector<rhi::ComputeDispatchDesc> dispatches;
        bool gated = false;
        while (next.operation < plan.size() && dispatches.size() < dispatchLimit) {
            const auto &op = plan[next.operation];
            if (op.entry == CommitRound && next.operation + 1u == plan.size() &&
                TryAppendTopologyTail(op, dispatchLimit, uploads, dispatches)) {
                gated = true;
                next = {plan.size(), 0u};
                break;
            }
            const uint32_t count = std::min(ChunkLimit(op.entry), op.length - next.first);
            if (!Emit(op, next.first, count, pendingUniforms, uploads, dispatches))
                break;
            next.first += count;
            if (next.first == op.length) {
                next.first = 0u;
                ++next.operation;
            }
        }
        if (dispatches.empty())
            return {nullptr, [self = shared_from_this(), dispatchLimit] { return self->Next(dispatchLimit); }};
        const bool canContract = dispatches.size() > 1;
        const bool boundary = next.operation == plan.size();
        const auto submittedPhase = phase;
        std::vector<rhi::ComputeBufferRead> reads;
        if (boundary) {
            if (gated)
                reads.push_back({schedule, uint64_t(config.layout.x) * WordBytes,
                                 (ConstraintScheduleHeaderWords + 1ull) * WordBytes});
            else if (submittedPhase == Phase::RoundWindow)
                reads.push_back({schedule, uint64_t(config.layout.y) * WordBytes, WordBytes});
            else
                reads.push_back({schedule, uint64_t(config.layout.x) * WordBytes, 2u * WordBytes});
        }
        const auto resumeAfter =
            boundary ? JoltGpuControlWork::ResumeAfter::Completion : JoltGpuControlWork::ResumeAfter::Acceptance;
        auto packet =
            rhi::PrepareComputeBatch(implementation->host, std::move(uploads), std::move(dispatches), std::move(reads));
        JoltGpuControlWork work{
            packet,
            [self = shared_from_this(), packet, next, boundary, submittedPhase, gated]() -> JoltGpuControlWork {
                // Cursor changes follow acceptance. Boundary decode also needs completion.
                self->cursor = next;
                if (boundary) {
                    if (gated)
                        self->DecodeGatedRound(packet);
                    else if (submittedPhase == Phase::RoundWindow)
                        self->DecodeRound(packet);
                    else {
                        self->DecodeAssembly(packet);
                        return {};
                    }
                }
                return self->Next();
            },
            resumeAfter};
        if (canContract)
            work.prepareMinimum = [self = shared_from_this()] { return self->Next(1); };
        return work;
    }

    std::shared_ptr<Impl> implementation;
    std::shared_ptr<const JoltGpuContactWorkset> contactLease;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> frozen;
    std::shared_ptr<rhi::ComputeBuffer> schedule;
    Uniforms pendingUniforms;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, 2> labels;
    std::function<void(std::shared_ptr<const JoltGpuConstraintWorkset>)> completed;
    std::function<void(std::shared_ptr<JoltGpuConstraintTopology>)> completedTopology;
    bool topologyOnly = false, resumingNumerical = false;
    ConstraintParams config{};
    std::array<uint32_t, 2> jointSort{}, patchSort{};
    std::vector<ScanLevel> hierarchy;
    std::vector<Operation> plan;
    Cursor cursor;
    Phase phase = Phase::RoundWindow;
    uint32_t oldBank = 0u, planBank = 0u, scheduledRounds = 0u, windowCount = 0u;
};

JoltGpuConstraints::JoltGpuConstraints(rhi::ComputeHost &host, const ShaderTargetCode &target)
    : m_impl(std::make_shared<Impl>(host, target))
{
}
JoltGpuControlWork
JoltGpuConstraints::PrepareTopology(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                                    const JPH::GPU::StepParams &parameters,
                                    std::function<void(std::shared_ptr<JoltGpuConstraintTopology>)> completed)
{
    return std::make_shared<Impl::Execution>(m_impl, storage, inputs, parameters,
                                             std::function<void(std::shared_ptr<const JoltGpuConstraintWorkset>)>{},
                                             true, std::move(completed))
        ->Next();
}
JoltGpuControlWork
JoltGpuConstraints::PrepareNumericalPlan(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                                         std::shared_ptr<JoltGpuConstraintTopology> topology,
                                         std::function<void(std::shared_ptr<const JoltGpuConstraintWorkset>)> completed)
{
    if (!topology || !topology->execution || !completed)
        throw std::invalid_argument("GPU numerical plan requires its one accepted topology continuation");
    auto execution = std::static_pointer_cast<Impl::Execution>(topology->execution);
    if (execution->implementation != m_impl || execution->phase != Impl::Execution::Phase::TopologyReady ||
        execution->contactLease != inputs.GetContacts() || storage.GetBodyCount() != topology->bodySlots)
        throw std::logic_error("GPU numerical plan crosses its source/topology generation");
    // Setup has completed its real accepted writes. Observe this writer's live
    // directory/activity flags; history retirement does not change those flags.
    execution->frozen[Joints] = storage.Get(JPH::GPU::Resource::Joints);
    execution->frozen[Contacts] = inputs.GetContacts()->contacts->Freeze();
    execution->frozen[Sleeps] = storage.Get(JPH::GPU::Resource::SleepStates)->Freeze();
    if (!execution->frozen[Joints] || !storage.GetHost().SharesServicesWith(execution->frozen[Joints]->GetHost()))
        throw std::invalid_argument("GPU constraints require their exclusive writer joint stream");
    execution->ValidateInput(execution->frozen[Contacts]);
    execution->ValidateInput(execution->frozen[Sleeps]);
    execution->config.select.z = 1u;
    execution->topologyOnly = false;
    execution->resumingNumerical = true;
    execution->completed = std::move(completed);
    topology->execution.reset();
    execution->BuildAssembly();
    return execution->Next();
}
} // namespace infernux
