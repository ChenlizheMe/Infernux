#include "JoltGpuConstraints.h"
#include <Jolt/GPU/ConstraintTypes.h>
#include "JoltGpuProgramSources.h"
#include "JoltGpuSleep.h"
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
    Joints,
    Patches,
    Contacts,
    Parents,
    RootVotes,
    Schedule,
    Sleeps,
    TopologyArguments,
    StepStatus,
    PortCount
};
enum Entry : uint8_t
{
    InitWords,
    Seed,
    BuildJointRefs,
    BuildPatchRefs,
    InitParents,
    JoinEdges,
    FlattenParents,
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
    BuildLocalTopology,
    BuildLocalSchedule,
    EntryCount
};
constexpr std::array<std::string_view, EntryCount> EntryNames{"constraint_init_words",
                                                              "constraint_seed",
                                                              "constraint_build_joint_refs",
                                                              "constraint_build_patch_refs",
                                                              "constraint_init_parents",
                                                              "constraint_join_edges",
                                                              "constraint_flatten_parents",
                                                              "constraint_validate_labels",
                                                              "constraint_validate_edges",
                                                              "constraint_accept_components",
                                                              "constraint_scan_blocks",
                                                              "constraint_reduce_blocks",
                                                              "constraint_add_scan_offsets",
                                                              "constraint_seed_sort",
                                                              "constraint_merge_items",
                                                              "constraint_clear_root_flags",
                                                              "constraint_mark_roots",
                                                              "constraint_build_root_rows",
                                                              "constraint_write_sources",
                                                              "constraint_write_islands",
                                                              "constraint_finalize",
                                                              "constraint_build_local_topology",
                                                              "constraint_build_local_schedule"};
constexpr uint32_t Bit(Port port)
{
    return 1u << port;
}
constexpr uint32_t P = Bit(Params), K = Bit(Work), S = Bit(Schedule), B = Bit(Bodies), C = Bit(Parents),
                   V = Bit(RootVotes);
constexpr std::array<uint32_t, EntryCount> EntryPorts{
    P | K | S,
    P | S,
    P | K | S | B | Bit(Joints) | Bit(Sleeps),
    P | K | S | B | Bit(Shapes) | Bit(Patches) | Bit(Contacts) | Bit(Sleeps),
    P | K | B | C,
    P | K | S | C,
    P | K | C | V,
    P | K | S | B | C,
    P | K | S | B | C,
    P | S | Bit(TopologyArguments) | Bit(StepStatus),
    P | K | S,
    P | K | S,
    P | K | S,
    P | K | S | C,
    P | K | S,
    P | K | V,
    P | K | S | C | V,
    P | K | S | C | V,
    P | K | S,
    P | K | S,
    P | S,
    P | S | B | C | V | Bit(Patches) | Bit(TopologyArguments) | Bit(StepStatus),
    P | S | B | C | Bit(Patches) | Bit(TopologyArguments) | Bit(StepStatus)};

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
      "summary", "selection"},
     0xfffu,
     true},
    {"stage_work", sizeof(StageWork), {"range"}, 1u, true},
    {"body_models", sizeof(BodyModel), {"meta", "mass", "invInertiaDiagonal", "inertiaRotation", "damping"}, 1u, true},
    {"shape_models",
     sizeof(ShapeModel),
     {"meta", "geometry", "localPositionCOM", "localRotation", "material", "surface", "source", "geometryReference"},
     (1u << 0) | (1u << 5) | (1u << 6) | (1u << 7),
     true},
    {"joint_stream", sizeof(QueryRecordWord), {"value"}, 1u, true},
    {"contact_patches", sizeof(ContactPatch), {"leaves"}, 1u, true},
    {"contacts", sizeof(Contact), {"meta", "normalDepth", "localAnchorA", "localAnchorB", "impulse"}, 1u, true},
    {"component_parents", sizeof(uint32_t), {}, 0u, false},
    {"root_votes", sizeof(uint32_t), {}, 0u, false},
    {"constraint_schedule", sizeof(ConstraintScheduleWord), {"value"}, 1u, false},
    {"body_sleep", sizeof(BodySleepRecord), {"author", "runtime"}, 0u, true},
    {"topology_arguments", sizeof(ConstraintScheduleWord), {"value"}, 1u, false},
    {"step_status", sizeof(uint32_t), {}, 0u, false}};
static_assert(std::size(Ports) == PortCount);
constexpr size_t MaxDispatchesPerPacket = JoltGpuConstraints::Workspace::MaxDispatches;
constexpr uint64_t WordBytes = sizeof(ConstraintScheduleWord);
constexpr uint64_t TopologyArgumentBytes = ConstraintBlockLimit * WordBytes;
static_assert(ConstraintItemLimit == ConstraintBlockSize * ConstraintBlockLimit);

uint32_t Checked(uint64_t n)
{
    if (n > UINT32_MAX)
        throw std::overflow_error("GPU constraint indexing exceeds uint32");
    return static_cast<uint32_t>(n);
}
uint32_t Blocks(uint32_t n)
{
    return n / ConstraintBlockSize + uint32_t(n % ConstraintBlockSize != 0u);
}
bool Collective(Entry e)
{
    return e == ScanBlocks || e == ReduceBlocks;
}
bool Summary(Entry e)
{
    return e == Seed || e == AcceptComponents || e == Finalize;
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
    if (port == Parents || port == RootVotes || port == StepStatus) {
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
            program.kernel = PrepareJoltGpuKernel(host, target.target, *code, std::move(bindings));
        }
    }
    rhi::ComputeHost host; // Copy of the shared services, never a new queue/device.
    std::array<Program, EntryCount> programs;
};

struct JoltGpuConstraints::Impl::Execution : std::enable_shared_from_this<Execution>
{
    enum class Phase
    {
        Connectivity,
        Assemble,
        TopologyReady,
        Done
    };
    struct Operation
    {
        Entry entry;
        uint32_t length;
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
              JoltGpuPrograms &programs, const StepParams &parameters,
              const std::shared_ptr<rhi::ComputeBuffer> &status, Workspace &resident,
              std::optional<JoltGpuConstraintSetup> setup,
              std::function<void(std::shared_ptr<JoltGpuConstraintTopology>, bool)> callback)
        : implementation(std::move(owner)), rigidStorage(storage), stepInputs(inputs), stagePrograms(programs),
          resident(resident), contactLease(inputs.GetContacts()), stepStatus(status), setup(std::move(setup)),
          completedTopology(std::move(callback))
    {
        if (!implementation->host.SharesServicesWith(storage.GetHost()) || !completedTopology || !contactLease)
            throw std::invalid_argument("GPU constraints require the accepted tick host/contact workset");
        const uint32_t world = parameters.selection.x;
        const auto &range = storage.GetModel().worlds.at(world);
        const uint32_t b = range.bodies.y, j = range.leaves.w, p = contactLease->patchSlots;
        const uint32_t totalBodies = storage.GetBodyCount();
        if (!b || parameters.counts.x != totalBodies || parameters.counts.y != storage.GetWorldCount() ||
            parameters.counts.z != storage.GetShapeCount())
            throw std::invalid_argument("GPU constraint parameters differ from accepted model domain");
        config.counts = {b, j, p, storage.GetJointCount()};
        config.selection = {world, range.bodies.x, range.leaves.x, range.leaves.z};
        // The accepted patch count and joint range establish that every body
        // can run its sleep update independently.
        isolatedTopology = j == 0u && p == 0u;
        if (isolatedTopology) {
            resident = {};
            return;
        }

        Checked(2ull * (uint64_t(j) + p));
        Checked(4ull * p);
        Checked(12ull * j);
        Checked(2ull * b);
        const uint32_t allRefs = Checked(uint64_t(j) + p);
        localTopology = ConstraintLocalTopologyFits(b, j, p);
        deviceSchedule = ConstraintDeviceScheduleFits(b, j, p);
        frozen[Bodies] = inputs.GetCaptured(Resource::BodyModels);
        frozen[Shapes] = inputs.GetCaptured(Resource::ShapeModels);
        // Joint directory/activity flags stay fixed under this arena writer.
        // Warm/rotor retirement changes neither field read by the schedule.
        frozen[Joints] = storage.Get(Resource::Joints);
        frozen[Patches] = contactLease->patches ? contactLease->patches->Freeze() : nullptr;
        // These live endpoints belong to the solver writer for this substep.
        // The packet orders topology reads, setup writes and assembly reads.
        frozen[Contacts] = contactLease->contacts;
        frozen[Sleeps] = storage.Get(Resource::SleepStates);
        if (!frozen[Joints] || !implementation->host.SharesServicesWith(frozen[Joints]->GetHost()))
            throw std::invalid_argument("GPU constraints require their exclusive writer joint stream");
        for (auto port : {Bodies, Shapes, Patches})
            ValidateInput(frozen[port]);
        RequireBytes(Bodies, uint64_t(totalBodies) * sizeof(BodyModel));
        RequireBytes(Shapes, uint64_t(std::max(1u, storage.GetShapeCount())) * sizeof(ShapeModel));
        RequireBytes(Joints, storage.GetJointLayout()->ByteSize());
        if (!frozen[Contacts] || !storage.GetHost().SharesServicesWith(frozen[Contacts]->GetHost()) ||
            frozen[Contacts]->GetByteSize() < uint64_t(std::max(1u, Checked(4ull * p))) * sizeof(Contact) ||
            frozen[Contacts]->GetByteSize() % sizeof(Contact))
            throw std::invalid_argument("GPU constraints require initialized contact capacity covering logical 4C");
        if (!frozen[Sleeps] || !implementation->host.SharesServicesWith(frozen[Sleeps]->GetHost()))
            throw std::invalid_argument("GPU constraints require the selected sleep source");
        RequireBytes(Sleeps, uint64_t(totalBodies) * sizeof(BodySleepRecord));
        if (frozen[Patches]->GetByteSize() < WordBytes)
            throw std::invalid_argument("GPU constraints require a candidate-bank header");
        config.limits.y = range.leaves.y;
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
        config.references.x = reserve(2ull * j);
        config.references.y = reserve(2ull * p);
        // Every numerical island uses at least one active reference. Each
        // per-kind table has the same bound for its own reference stream.
        config.sources = {reserve(j), reserve(p), reserve(std::min(b, j)), reserve(std::min(b, p))};
        config.islands.x = reserve(2ull * std::min(b, allRefs));
        config.limits.w = reserve(b);
        if (!localTopology) {
            config.select.w = 1u;
            config.layout.z = reserve(b);
            config.layout.w = reserve(allRefs);
            jointSort = {reserve(j), reserve(j)};
            patchSort = {reserve(p), reserve(p)};
            config.islands.y = reserve(b);
            config.islands.z = reserve(b);
            config.islands.w = reserve(b);
            config.summary = {reserve(1u), reserve(1u), reserve(1u), reserve(1u)};
            // One reusable hierarchy for multi-workgroup scans and reductions.
            uint32_t count = std::max(b, allRefs);
            do {
                const uint32_t blocks = Blocks(count);
                const uint32_t prefix = hierarchy.empty() ? 0u : reserve(count);
                hierarchy.push_back({prefix, reserve(blocks), count});
                if (blocks == 1u)
                    break;
                count = blocks;
            } while (true);
        }
        config.limits.x = Checked(words);
        workspace = std::exchange(resident, {});
        const auto scheduleBytes = uint64_t(config.limits.x) * WordBytes;
        if (workspace.schedule && (workspace.schedule->GetByteSize() < scheduleBytes ||
                                   workspace.schedule->GetByteSize() / 2u > scheduleBytes))
            workspace.schedule.reset();
        const auto keepExact = [](auto &buffer, uint64_t bytes) {
            if (buffer && buffer->GetByteSize() != bytes)
                buffer.reset();
        };
        keepExact(workspace.parents, uint64_t(b) * sizeof(uint32_t));
        keepExact(workspace.rootVotes, uint64_t(b) * sizeof(uint32_t));
        keepExact(workspace.arguments, ArgumentBytes());
        BuildConnectivity();
    }

    uint64_t ArgumentBytes() const
    {
        return localTopology ? ConstraintLocalArgumentWords * WordBytes : TopologyArgumentBytes;
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
        const uint64_t bodyBytes = uint64_t(config.counts.x) * sizeof(uint32_t);
        if (!Endpoint(workspace.schedule, uint64_t(config.limits.x) * WordBytes) ||
            !Endpoint(workspace.rootVotes, bodyBytes))
            return false;
        if (phase != Phase::Connectivity)
            return true;
        if (!Endpoint(workspace.parents, bodyBytes))
            return false;
        if (!workspace.arguments)
            workspace.arguments =
                Allocate(ArgumentBytes(), rhi::BufferUsageFlags::Storage | rhi::BufferUsageFlags::Indirect);
        return workspace.arguments != nullptr;
    }

    void Add(Entry entry, uint32_t length, const ConstraintParams &value)
    {
        if (length)
            plan.push_back({entry, length, value});
    }
    void Add(Entry entry, uint32_t length)
    {
        Add(entry, length, config);
    }
    static void AddReferences(std::vector<Operation> &operations, const ConstraintParams &value)
    {
        if (value.counts.y)
            operations.push_back({BuildJointRefs, value.counts.y, value});
        if (value.counts.z)
            operations.push_back({BuildPatchRefs, value.counts.z, value});
    }
    void Kind(Entry entry, uint32_t kind)
    {
        auto value = config;
        value.select.x = kind;
        Add(entry, kind == ConstraintJoint ? config.counts.y : config.counts.z, value);
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
            Add(ReduceBlocks, blocks, value);
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
            Add(ScanBlocks, Blocks(l.count), value);
        }
        for (size_t i = scan.size() - 1u; i != 0u; --i) {
            const auto &child = scan[i - 1u];
            auto value = config;
            value.scan = {child.input, child.prefix, child.total, child.count};
            value.scanParent = {scan[i].prefix, 0u, 0u, 0u};
            Add(AddOffsets, child.count, value);
        }
    }
    uint32_t Sort(uint32_t kind, std::array<uint32_t, 2> banks, uint32_t count)
    {
        auto value = config;
        value.select = {kind, banks[0], 0u, 0u};
        Add(SeedSort, count, value);
        uint32_t source = 0;
        for (uint64_t width = ConstraintBlockSize; width < count; width *= 2u) {
            value.merge = {banks[source], banks[source ^ 1u], count, Checked(width)};
            Add(MergeItems, count, value);
            source ^= 1u;
        }
        return banks[source];
    }
    void ResetPlan(Phase next)
    {
        phase = next;
        plan.clear();
        cursor = {};
    }
    void BuildConnectivity()
    {
        ResetPlan(Phase::Connectivity);
        if (!workspace.schedule || !workspace.schedule->HasInitializedRange(0, workspace.schedule->GetByteSize()))
            Add(InitWords, config.limits.x);
        Add(Seed, 1u);
        AddReferences(plan, config);
        if (localTopology) {
            Add(BuildLocalTopology, 1u);
            return;
        }
        Add(InitParents, config.counts.x);
        Kind(JoinEdges, ConstraintJoint);
        Kind(JoinEdges, ConstraintPatch);
        Add(FlattenParents, config.counts.x);
        Add(ValidateLabels, config.counts.x);
        Reduce(config.layout.z, config.counts.x, 0u, config.summary.w);
        ValidateReferences();
    }
    void ValidateReferences()
    {
        Kind(ValidateEdges, ConstraintJoint);
        Kind(ValidateEdges, ConstraintPatch);
        const uint32_t refs = Checked(uint64_t(config.counts.y) + config.counts.z);
        Reduce(config.layout.w, refs, 0u, config.summary.z);
        Add(AcceptComponents, 1u);
    }
    void BuildAssembly()
    {
        ResetPlan(Phase::Assemble);
        AddReferences(plan, config);
        // Wake changes the active reference subset. The parent forest and
        // its published body roots are still the accepted substep's inputs.
        if (localTopology) {
            Add(BuildLocalSchedule, 1u);
            return;
        }
        ValidateReferences();
        config.references.z = Sort(ConstraintJoint, jointSort, config.counts.y);
        config.references.w = Sort(ConstraintPatch, patchSort, config.counts.z);
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

    uint32_t ChunkLimit(Entry entry) const
    {
        if (localTopology) {
            if (entry == InitWords)
                return config.limits.x;
            if (entry == BuildPatchRefs)
                return config.counts.z;
        }
        return Collective(entry) ? ConstraintBlockLimit : Summary(entry) ? 1u : ConstraintItemLimit;
    }

    bool Emit(const Operation &op, uint32_t first, uint32_t count, std::vector<rhi::ComputeBufferUpdate> &uploads,
              std::vector<rhi::ComputeDispatchDesc> &dispatches)
    {
        const bool collective = Collective(op.entry);
        // Each dispatch covers at most ConstraintBlockLimit workgroups.
        const uint32_t limit = ChunkLimit(op.entry);
        if (!count || count > limit || first >= op.length || count > op.length - first)
            throw std::logic_error("GPU constraints emitted an invalid bounded chunk");
        const bool usesWork = (EntryPorts[op.entry] & K) != 0u;
        auto parameterBytes = Bytes(op.config);
        auto workBytes = usesWork ? Bytes(StageWork{{first, count, 0u, 0u}}) : std::vector<uint8_t>{};
        // Uniform inputs are immutable within this packet. Adjacent operators
        // often use the same configuration or range, so share their one upload.
        const auto reuse = [&](const auto &bytes) -> std::shared_ptr<rhi::ComputeBuffer> {
            const auto found =
                std::find_if(uploads.begin(), uploads.end(), [&](const auto &upload) { return upload.data == bytes; });
            return found == uploads.end() ? nullptr : found->buffer;
        };
        auto parameters = reuse(parameterBytes);
        auto work = usesWork ? reuse(workBytes) : nullptr;
        // Each slot is written at most once before this packet records. The
        // next packet reuses it after the prior readers have entered the FIFO.
        auto &uniforms = workspace.uniforms[dispatches.size()];
        if ((!parameters && !Endpoint(uniforms.parameters, sizeof(ConstraintParams))) ||
            (usesWork && !work && !Endpoint(uniforms.work, sizeof(StageWork))))
            return false;
        if (!parameters) {
            parameters = uniforms.parameters;
            uploads.push_back({parameters, 0u, std::move(parameterBytes)});
        }
        if (usesWork && !work) {
            work = uniforms.work;
            uploads.push_back({work, 0u, std::move(workBytes)});
        }
        if (op.entry == BuildLocalTopology || op.entry == AcceptComponents)
            JoltGpuSleep::InitializeStepStatus(stepStatus, uploads);
        rhi::ComputeDispatchDesc dispatch;
        const auto &program = implementation->programs[op.entry];
        dispatch.kernel = program.kernel;
        dispatch.groupCountX = collective ? count : Summary(op.entry) ? 1u : Blocks(count);
        if (dispatch.groupCountX > implementation->host.device.GetCapabilities().limits.maxComputeWorkgroupCount[0])
            throw std::length_error("GPU constraint chunk exceeds dispatch dimensions");
        rhi::ComputeDispatchProof proof;
        for (const auto port : program.ports) {
            const auto &buffer = port == Params              ? parameters
                                 : port == Work              ? work
                                 : port == Schedule          ? workspace.schedule
                                 : port == Parents           ? workspace.parents
                                 : port == RootVotes         ? workspace.rootVotes
                                 : port == TopologyArguments ? workspace.arguments
                                 : port == StepStatus        ? stepStatus
                                                             : frozen[port];
            if (!buffer)
                throw std::logic_error("GPU constraint dispatch has an absent endpoint");
            const uint32_t binding = static_cast<uint32_t>(dispatch.buffers.size());
            dispatch.buffers.push_back(buffer);
            const bool initSchedule = op.entry == InitWords && port == Schedule;
            const bool initParent = op.entry == InitParents && port == Parents;
            const bool initVotes = port == RootVotes && (op.entry == ClearRootFlags || op.entry == FlattenParents);
            if (op.entry == BuildLocalTopology && (port == Parents || port == RootVotes)) {
                dispatch.bufferAccesses.push_back(rhi::ComputeBufferAccess::Write);
                proof.writes.push_back({binding, 0u, buffer->GetByteSize()});
            } else if (initSchedule || initParent || initVotes) {
                const uint64_t stride = port == Schedule ? WordBytes : 4u;
                dispatch.bufferAccesses.push_back(rhi::ComputeBufferAccess::Write);
                proof.writes.push_back({binding, uint64_t(first) * stride, uint64_t(count) * stride});
            } else if (port == TopologyArguments) {
                dispatch.bufferAccesses.push_back(rhi::ComputeBufferAccess::Write);
                if (op.entry == BuildLocalTopology) {
                    const auto groups = Blocks(std::max({config.counts.x, config.counts.y, config.counts.z}));
                    proof.writes.push_back({binding, 0u, groups * WordBytes});
                    proof.writes.push_back({binding, ConstraintLocalWakePatchDispatch * WordBytes, WordBytes});
                } else if (op.entry == BuildLocalSchedule) {
                    proof.writes.push_back({binding, ConstraintLocalJointDispatch * WordBytes, 2u * WordBytes});
                } else {
                    proof.writes.push_back({binding, 0u, buffer->GetByteSize()});
                }
            } else {
                const bool mayWrite = port == StepStatus ||
                                      (port == Schedule && op.entry != MarkRoots && op.entry != JoinEdges) ||
                                      (port == Parents && (op.entry == JoinEdges || op.entry == FlattenParents)) ||
                                      (port == RootVotes && op.entry == MarkRoots);
                dispatch.bufferAccesses.push_back(mayWrite ? rhi::ComputeBufferAccess::ReadWrite
                                                           : rhi::ComputeBufferAccess::Read);
                proof.reads.push_back({binding, 0u, buffer->GetByteSize()});
            }
        }
        dispatch.proof = std::move(proof);
        if (op.entry == BuildPatchRefs)
            dispatch.indirectArguments = contactLease->patchDispatch;
        dispatches.push_back(std::move(dispatch));
        return true;
    }

    JoltGpuConstraintTopology TopologyView() const
    {
        JoltGpuConstraintTopology value;
        value.bodyRootFirst = config.limits.w;
        value.bodyFirst = config.selection.y;
        value.bodyCount = config.counts.x;
        value.schedule = workspace.schedule;
        value.rootVotes = workspace.rootVotes;
        value.wakeArguments = workspace.arguments;
        if (localTopology)
            value.wakePatchDispatch =
                rhi::ComputeIndirectArguments{workspace.arguments, ConstraintLocalWakePatchDispatch * WordBytes};
        return value;
    }
    void PublishTopology(bool setupAccepted = false)
    {
        auto value = std::make_shared<JoltGpuConstraintTopology>(TopologyView());
        value->execution = shared_from_this();
        // Setup must write/read its actual new generations.
        frozen[Joints].reset();
        frozen[Contacts].reset();
        phase = Phase::TopologyReady;
        setup.reset();
        auto callback = std::move(completedTopology);
        callback(std::move(value), setupAccepted);
    }
    void Install(const UInt4 &counts, const UInt4 &items)
    {
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
        // The assembly and all its consumers use this owner's Compute FIFO.
        // The next substep may write these addresses after all readers
        // have been accepted. Queue receipts retain their actual allocations.
        value->schedule = workspace.schedule;
        if (deviceSchedule)
            value->solveArguments = workspace.arguments;
        value->bodyRootFirst = config.limits.w;
        value->bodyFirst = config.selection.y;
        value->bodyCount = config.counts.x;
        value->jointRefFirst = config.references.x;
        value->jointRefCount = config.counts.y;
        value->patchRefFirst = config.references.y;
        value->patchRefCount = config.counts.z;
        value->rootVotes = workspace.rootVotes;
        if (!isolatedTopology)
            resident = std::move(workspace);
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

    void AppendPrefix(const std::vector<Operation> &operations, Cursor &next, size_t dispatchLimit,
                      std::vector<rhi::ComputeBufferUpdate> &uploads, std::vector<rhi::ComputeDispatchDesc> &dispatches)
    {
        while (next.operation < operations.size() && dispatches.size() < dispatchLimit) {
            const auto &op = operations[next.operation];
            const uint32_t count = std::min(ChunkLimit(op.entry), op.length - next.first);
            if (!Emit(op, next.first, count, uploads, dispatches))
                return;
            next.first += count;
            if (next.first == op.length) {
                next.first = 0u;
                ++next.operation;
            }
        }
    }

    bool AppendSetup(size_t dispatchLimit, std::vector<rhi::ComputeBufferUpdate> &uploads,
                     std::vector<rhi::ComputeDispatchDesc> &dispatches)
    {
        if (!setup || setup->stages.size() > dispatchLimit - dispatches.size())
            return false;
        const auto prefixUploads = uploads.size(), prefixDispatches = dispatches.size();
        if (!stagePrograms.GetSleep().AppendWake(rigidStorage, stepInputs, TopologyView(), stepStatus,
                                                 dispatchLimit - setup->stages.size(), uploads, dispatches)) {
            uploads.resize(prefixUploads);
            dispatches.resize(prefixDispatches);
            return false;
        }
        stagePrograms.AppendStages(rigidStorage, stepInputs, setup->stages, uploads, dispatches);
        return true;
    }

    bool AppendLocalAssembly(size_t dispatchLimit, std::vector<rhi::ComputeBufferUpdate> &uploads,
                             std::vector<rhi::ComputeDispatchDesc> &dispatches)
    {
        if (!deviceSchedule || !setup->assembled)
            return false;
        auto numerical = config;
        numerical.select.z = 1u;
        std::vector<Operation> operations;
        AddReferences(operations, numerical);
        operations.push_back({BuildLocalSchedule, 1u, numerical});
        const auto prefixUploads = uploads.size(), prefixDispatches = dispatches.size();
        Cursor next;
        AppendPrefix(operations, next, dispatchLimit, uploads, dispatches);
        if (next.operation == operations.size())
            return true;
        uploads.resize(prefixUploads);
        dispatches.resize(prefixDispatches);
        return false;
    }

    JoltGpuControlWork Next(size_t dispatchLimit = MaxDispatchesPerPacket)
    {
        if (phase == Phase::Done || phase == Phase::TopologyReady)
            return {};
        if (isolatedTopology) {
            PublishTopology();
            return {};
        }
        if (!Capacity())
            return {nullptr, [self = shared_from_this(), dispatchLimit] { return self->Next(dispatchLimit); }};
        Cursor next = cursor;
        std::vector<rhi::ComputeBufferUpdate> uploads;
        std::vector<rhi::ComputeDispatchDesc> dispatches;
        AppendPrefix(plan, next, dispatchLimit, uploads, dispatches);
        if (dispatches.empty())
            return {nullptr, [self = shared_from_this(), dispatchLimit] { return self->Next(dispatchLimit); }};
        const bool boundary = next.operation == plan.size();
        const auto submittedPhase = phase;
        const bool includesSetup =
            boundary && submittedPhase == Phase::Connectivity && AppendSetup(dispatchLimit, uploads, dispatches);
        const bool includesAssembly = includesSetup && AppendLocalAssembly(dispatchLimit, uploads, dispatches);
        const bool canContract = dispatches.size() > 1;
        std::vector<rhi::ComputeBufferRead> reads;
        if (boundary && submittedPhase == Phase::Assemble && !deviceSchedule)
            reads.push_back({workspace.schedule, uint64_t(config.layout.x) * WordBytes, 2u * WordBytes});
        const auto resumeAfter =
            reads.empty() ? JoltGpuControlWork::ResumeAfter::Acceptance : JoltGpuControlWork::ResumeAfter::Completion;
        auto packet =
            rhi::PrepareComputeBatch(implementation->host, std::move(uploads), std::move(dispatches), std::move(reads));
        JoltGpuControlWork work{packet,
                                [self = shared_from_this(), packet, next, boundary, submittedPhase, includesSetup,
                                 includesAssembly]() -> JoltGpuControlWork {
                                    // Cursor changes follow acceptance. Boundary decode also needs completion.
                                    self->cursor = next;
                                    if (boundary) {
                                        if (submittedPhase == Phase::Connectivity) {
                                            if (includesAssembly) {
                                                self->completed = std::move(self->setup->assembled);
                                                self->setup.reset();
                                                self->completedTopology = {};
                                                self->Install({}, {});
                                            } else
                                                self->PublishTopology(includesSetup);
                                            return {};
                                        }
                                        if (self->deviceSchedule)
                                            self->Install({}, {});
                                        else
                                            self->DecodeAssembly(packet);
                                        return {};
                                    }
                                    return self->Next();
                                },
                                resumeAfter};
        if (canContract)
            work.prepareMinimum = [self = shared_from_this()] { return self->Next(1); };
        return work;
    }

    std::shared_ptr<Impl> implementation;
    const JoltGpuStorage &rigidStorage;
    const JoltGpuStepInputs &stepInputs;
    JoltGpuPrograms &stagePrograms;
    Workspace &resident;
    Workspace workspace;
    std::shared_ptr<const JoltGpuContactWorkset> contactLease;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> frozen;
    std::shared_ptr<rhi::ComputeBuffer> stepStatus;
    std::optional<JoltGpuConstraintSetup> setup;
    std::function<void(std::shared_ptr<const JoltGpuConstraintWorkset>)> completed;
    std::function<void(std::shared_ptr<JoltGpuConstraintTopology>, bool)> completedTopology;
    bool isolatedTopology = false;
    bool localTopology = false;
    bool deviceSchedule = false;
    ConstraintParams config{};
    std::array<uint32_t, 2> jointSort{}, patchSort{};
    std::vector<ScanLevel> hierarchy;
    std::vector<Operation> plan;
    Cursor cursor;
    Phase phase = Phase::Connectivity;
};

JoltGpuConstraints::JoltGpuConstraints(rhi::ComputeHost &host, const ShaderTargetCode &target)
    : m_impl(std::make_shared<Impl>(host, target))
{
}

bool JoltGpuConstraints::TryPrepareStartupPrograms()
{
    bool ready = true;
    for (const auto entry : {InitWords, Seed, BuildPatchRefs, BuildLocalTopology, BuildLocalSchedule})
        ready &= m_impl->programs[entry].kernel->TryPreparePipeline();
    return ready;
}

JoltGpuControlWork
JoltGpuConstraints::PrepareTopology(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                                    JoltGpuPrograms &programs, const JPH::GPU::StepParams &parameters,
                                    const std::shared_ptr<rhi::ComputeBuffer> &stepStatus, Workspace &workspace,
                                    std::optional<JoltGpuConstraintSetup> setup,
                                    std::function<void(std::shared_ptr<JoltGpuConstraintTopology>, bool)> completed)
{
    return std::make_shared<Impl::Execution>(m_impl, storage, inputs, programs, parameters, stepStatus, workspace,
                                             std::move(setup), std::move(completed))
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
    const auto &parameters = inputs.GetParameters();
    if (execution->implementation != m_impl || execution->phase != Impl::Execution::Phase::TopologyReady ||
        execution->contactLease != inputs.GetContacts() || topology->bodyFirst != parameters.selection.y ||
        topology->bodyCount != parameters.selectedCounts.x)
        throw std::logic_error("GPU numerical plan crosses its source/topology generation");
    if (execution->isolatedTopology) {
        execution->completed = std::move(completed);
        topology->execution.reset();
        execution->Install({}, {});
        return {};
    }
    // Setup has completed its real accepted writes. Observe this writer's live
    // directory/activity flags; history retirement does not change those flags.
    execution->frozen[Joints] = storage.Get(JPH::GPU::Resource::Joints);
    execution->frozen[Contacts] = inputs.GetContacts()->contacts;
    execution->frozen[Sleeps] = storage.Get(JPH::GPU::Resource::SleepStates);
    if (!execution->frozen[Joints] || !storage.GetHost().SharesServicesWith(execution->frozen[Joints]->GetHost()))
        throw std::invalid_argument("GPU constraints require their exclusive writer joint stream");
    execution->config.select.z = 1u;
    execution->completed = std::move(completed);
    topology->execution.reset();
    execution->BuildAssembly();
    return execution->Next();
}
} // namespace infernux
