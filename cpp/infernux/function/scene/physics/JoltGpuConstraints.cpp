#include "JoltGpuConstraints.h"
#include <Jolt/GPU/ConstraintTypes.h>
#include "JoltGpuProgramSources.h"
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
    PortCount
};
enum Entry : uint8_t
{
    InitWords,
    Seed,
    BuildJointRefs,
    BuildPatchRefs,
    InitParents,
    ResetConnectivity,
    JoinEdges,
    FlattenParents,
    GateValidation,
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
    InitializeIsolatedTopology,
    BuildLocalTopology,
    EntryCount
};
constexpr std::array<std::string_view, EntryCount> EntryNames{"constraint_init_words",
                                                              "constraint_seed",
                                                              "constraint_build_joint_refs",
                                                              "constraint_build_patch_refs",
                                                              "constraint_init_parents",
                                                              "constraint_reset_connectivity",
                                                              "constraint_join_edges",
                                                              "constraint_flatten_parents",
                                                              "constraint_gate_validation",
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
                                                              "constraint_write_body_roots",
                                                              "constraint_initialize_isolated_topology",
                                                              "constraint_build_local_topology"};
constexpr uint32_t Bit(Port port)
{
    return 1u << port;
}
constexpr uint32_t P = Bit(Params), K = Bit(Work), S = Bit(Schedule), B = Bit(Bodies), C = Bit(Parents),
                   V = Bit(RootVotes);
constexpr std::array<uint32_t, EntryCount> EntryPorts{P | K | S,
                                                      P | S,
                                                      P | K | S | B | Bit(Joints) | Bit(Sleeps),
                                                      P | K | S | B | Bit(Shapes) | Bit(Patches) | Bit(Contacts) |
                                                          Bit(Sleeps),
                                                      P | K | B | C,
                                                      P | V,
                                                      P | K | S | C | V,
                                                      P | K | C | V,
                                                      P | S | V | Bit(TopologyArguments),
                                                      P | K | S | B | C,
                                                      P | K | S | B | C,
                                                      P | S,
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
                                                      P | K | S | C,
                                                      P | K | S | B | V,
                                                      P | S | B | C | V | Bit(TopologyArguments)};

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
    {"topology_arguments", sizeof(ConstraintScheduleWord), {"value"}, 1u, false}};
static_assert(std::size(Ports) == PortCount);
constexpr uint32_t MaxDispatchesPerPacket = 16u;
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
    return e == Seed || e == ResetConnectivity || e == AcceptComponents || e == Finalize;
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
    if (port == Parents || port == RootVotes) {
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
    struct Uniforms
    {
        std::shared_ptr<rhi::ComputeBuffer> parameters, work;
    };

    Execution(std::shared_ptr<Impl> owner, const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
              const StepParams &parameters, std::function<void(std::shared_ptr<JoltGpuConstraintTopology>)> callback)
        : implementation(std::move(owner)), contactLease(inputs.GetContacts()), completedTopology(std::move(callback))
    {
        if (!implementation->host.SharesServicesWith(storage.GetHost()) || !completedTopology || !contactLease)
            throw std::invalid_argument("GPU constraints require the accepted tick host/contact workset");
        const uint32_t world = parameters.selection.x;
        const auto &range = storage.GetModel().worlds.at(world);
        const uint32_t b = range.bodies.y, j = range.leaves.w, p = contactLease->patchCount;
        const uint32_t totalBodies = storage.GetBodyCount();
        // Empty scenes bypass the solver owner; a real body domain can contain
        // zero responsive constraints and still yields a validated empty schedule.
        if (!b || parameters.counts.x != totalBodies || parameters.counts.y != storage.GetWorldCount() ||
            parameters.counts.z != storage.GetShapeCount())
            throw std::invalid_argument("GPU constraint parameters differ from accepted model domain");
        Checked(2ull * (uint64_t(j) + p));
        Checked(4ull * p);
        Checked(12ull * j);
        Checked(2ull * b);
        const uint32_t allRefs = Checked(uint64_t(j) + p);
        localTopology = b <= ConstraintLocalBodies && allRefs <= ConstraintLocalRefs;
        frozen[Bodies] = inputs.GetCaptured(Resource::BodyModels);
        frozen[Shapes] = inputs.GetCaptured(Resource::ShapeModels);
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
        for (auto port : {Bodies, Shapes, Patches, Contacts})
            ValidateInput(frozen[port]);
        RequireBytes(Bodies, uint64_t(totalBodies) * sizeof(BodyModel));
        RequireBytes(Shapes, uint64_t(std::max(1u, storage.GetShapeCount())) * sizeof(ShapeModel));
        RequireBytes(Joints, storage.GetJointLayout()->ByteSize());
        if (frozen[Contacts]->GetByteSize() < uint64_t(std::max(1u, Checked(4ull * p))) * sizeof(Contact) ||
            frozen[Contacts]->GetByteSize() % sizeof(Contact))
            throw std::invalid_argument("GPU constraints require initialized contact capacity covering logical 4C");
        if (!frozen[Sleeps] || !implementation->host.SharesServicesWith(frozen[Sleeps]->GetHost()))
            throw std::invalid_argument("GPU constraints require the selected sleep source");
        RequireBytes(Sleeps, uint64_t(totalBodies) * sizeof(BodySleepRecord));
        if (frozen[Patches]->GetByteSize() < WordBytes)
            throw std::invalid_argument("GPU constraints require a candidate-bank header");
        config.counts = {b, j, p, storage.GetJointCount()};
        config.selection = {world, range.bodies.x, range.leaves.x, range.leaves.z};
        config.limits.y = range.leaves.y;
        config.limits.z = Checked(4ull * p);
        // The completed broadphase and accepted selected joint range prove an empty numerical domain.
        isolatedTopology = j == 0u && p == 0u;
        if (isolatedTopology) {
            config.limits.w = ConstraintScheduleHeaderWords;
            config.limits.x = Checked(uint64_t(config.limits.w) + b);
            phase = Phase::Assemble;
            Add(InitializeIsolatedTopology, b);
            return;
        }
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
        BuildConnectivity();
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
        const uint64_t voteBytes = (uint64_t(config.counts.x) + 1u) * sizeof(uint32_t);
        if (!Endpoint(schedule, uint64_t(config.limits.x) * WordBytes) || !Endpoint(rootVotes, voteBytes))
            return false;
        if (isolatedTopology || phase != Phase::Connectivity)
            return true;
        if (!Endpoint(parents, voteBytes - sizeof(uint32_t)))
            return false;
        if (!topologyArguments) {
            const auto groups = Blocks(std::max({config.counts.x, config.counts.y, config.counts.z}));
            const auto bytes = localTopology ? uint64_t(groups) * WordBytes : TopologyArgumentBytes;
            topologyArguments = Allocate(bytes, rhi::BufferUsageFlags::Storage | rhi::BufferUsageFlags::Indirect);
        }
        return topologyArguments != nullptr;
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
    void BuildConnectivity(bool linkEdges = true)
    {
        ResetPlan(Phase::Connectivity);
        if (!connectivityPass) {
            Add(InitWords, config.limits.x);
            Add(Seed, 1u);
            Add(BuildJointRefs, config.counts.y);
            Add(BuildPatchRefs, config.counts.z);
            if (localTopology) {
                Add(BuildLocalTopology, 1u);
                validationStart = plan.size();
                return;
            }
            Add(InitParents, config.counts.x);
        }
        Add(ResetConnectivity, 1u);
        if (linkEdges) {
            Kind(JoinEdges, ConstraintJoint);
            Kind(JoinEdges, ConstraintPatch);
        }
        Add(FlattenParents, config.counts.x);
        Add(GateValidation, 1u);
        validationStart = plan.size();
        Add(ValidateLabels, config.counts.x);
        Reduce(config.layout.z, config.counts.x, 0u, config.summary.w);
        ValidateReferences();
        Add(WriteBodyRoots, config.counts.x);
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
        Add(BuildJointRefs, config.counts.y);
        Add(BuildPatchRefs, config.counts.z);
        // Wake changes the active reference subset. The parent forest and
        // its published body roots are still the accepted substep's inputs.
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

    static uint32_t ChunkLimit(Entry entry)
    {
        return Collective(entry) ? ConstraintBlockLimit : Summary(entry) ? 1u : ConstraintItemLimit;
    }

    bool Emit(const Operation &op, uint32_t first, uint32_t count, Uniforms &uniforms,
              std::vector<rhi::ComputeBufferUpdate> &uploads, std::vector<rhi::ComputeDispatchDesc> &dispatches)
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
        // Only new, unpublished allocations survive pressure. A packet-local
        // hit must not become the next packet's writable pending endpoint.
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
                                 : port == Schedule          ? schedule
                                 : port == Parents           ? parents
                                 : port == RootVotes         ? rootVotes
                                 : port == TopologyArguments ? topologyArguments
                                                             : frozen[port];
            if (!buffer)
                throw std::logic_error("GPU constraint dispatch has an absent endpoint");
            const uint32_t binding = static_cast<uint32_t>(dispatch.buffers.size());
            dispatch.buffers.push_back(buffer);
            if (op.entry == InitializeIsolatedTopology && (port == Schedule || port == RootVotes)) {
                dispatch.bufferAccesses.push_back(rhi::ComputeBufferAccess::Write);
                if (port == Schedule) {
                    const uint64_t rootFirst = config.limits.w;
                    const uint64_t start = first ? rootFirst + first : 0u;
                    const uint64_t length = count + (first ? 0u : rootFirst);
                    proof.writes.push_back({binding, start * WordBytes, length * WordBytes});
                } else {
                    const uint64_t length = uint64_t(count) + (first + count == config.counts.x);
                    proof.writes.push_back({binding, uint64_t(first) * sizeof(uint32_t), length * sizeof(uint32_t)});
                }
                continue;
            }
            const bool initSchedule = op.entry == InitWords && port == Schedule;
            const bool initParent = op.entry == InitParents && port == Parents;
            const bool initVotes = op.entry == ClearRootFlags && port == RootVotes;
            const bool statusOnly = port == RootVotes && (op.entry == ResetConnectivity || op.entry == JoinEdges ||
                                                          op.entry == FlattenParents || op.entry == GateValidation);
            const uint64_t voteStatusOffset = uint64_t(config.counts.x) * sizeof(uint32_t);
            if (op.entry == BuildLocalTopology && (port == Parents || port == RootVotes)) {
                dispatch.bufferAccesses.push_back(rhi::ComputeBufferAccess::Write);
                proof.writes.push_back({binding, 0u, buffer->GetByteSize()});
            } else if (initSchedule || initParent || initVotes) {
                const uint64_t stride = port == Schedule ? WordBytes : 4u;
                dispatch.bufferAccesses.push_back(rhi::ComputeBufferAccess::Write);
                proof.writes.push_back({binding, uint64_t(first) * stride, uint64_t(count) * stride});
            } else if (port == TopologyArguments || (statusOnly && op.entry == ResetConnectivity)) {
                dispatch.bufferAccesses.push_back(rhi::ComputeBufferAccess::Write);
                proof.writes.push_back({binding, statusOnly ? voteStatusOffset : 0u,
                                        statusOnly ? sizeof(uint32_t) : buffer->GetByteSize()});
            } else {
                const bool mayWrite = (port == Schedule && op.entry != MarkRoots) ||
                                      (port == Parents && (op.entry == JoinEdges || op.entry == FlattenParents)) ||
                                      (port == RootVotes && op.entry != BuildRootRows && op.entry != GateValidation);
                dispatch.bufferAccesses.push_back(mayWrite ? rhi::ComputeBufferAccess::ReadWrite
                                                           : rhi::ComputeBufferAccess::Read);
                const uint64_t bytes =
                    port == RootVotes ? (statusOnly ? sizeof(uint32_t) : voteStatusOffset) : buffer->GetByteSize();
                proof.reads.push_back({binding, statusOnly ? voteStatusOffset : 0u, bytes});
            }
        }
        dispatch.proof = std::move(proof);
        dispatches.push_back(std::move(dispatch));
        uniforms = {};
        return true;
    }

    void DecodeConnectivity(const std::shared_ptr<rhi::PreparedComputeBatch> &packet)
    {
        UInt4 control{};
        packet->Readback(0u)->ReadInto(&control, sizeof(control));
        const bool ready = control.x == 0u;
        if (control.y != ++connectivityPass || control.z ||
            (control.x & ~(ConstraintLinksPending | ConstraintRootsPending)) || control.w != uint32_t(ready))
            throw std::runtime_error("GPU constraint topology did not validate (status " + std::to_string(control.z) +
                                     ")");
        if (ready) {
            PublishTopology();
            return;
        }
        // An edge can descend at most 2*(B-1) links, and flattening needs
        // at most B parent reads per body across the remaining passes.
        const auto quanta = [](uint64_t links) {
            return (links + ConstraintConnectivityQuantum - 1u) / ConstraintConnectivityQuantum;
        };
        if (connectivityPass >= quanta(2ull * (config.counts.x - 1u)) + quanta(config.counts.x))
            throw std::runtime_error("GPU constraint topology did not converge within its parent-chain bound");
        BuildConnectivity((control.x & ConstraintLinksPending) != 0u);
    }
    void PublishTopology()
    {
        auto value = std::make_shared<JoltGpuConstraintTopology>();
        value->bodyRootFirst = config.limits.w;
        value->bodyFirst = config.selection.y;
        value->bodyCount = config.counts.x;
        value->schedule = schedule;
        value->rootVotes = rootVotes;
        value->execution = shared_from_this();
        if (localTopology)
            value->wakeArguments = std::move(topologyArguments);
        else
            topologyArguments.reset();
        // Setup must write/read its actual new generations.
        frozen[Joints].reset();
        frozen[Contacts].reset();
        phase = Phase::TopologyReady;
        auto callback = std::move(completedTopology);
        callback(std::move(value));
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
        value->schedule = schedule->Freeze();
        value->bodyRootFirst = config.limits.w;
        value->bodyFirst = config.selection.y;
        value->bodyCount = config.counts.x;
        value->jointRefFirst = config.references.x;
        value->jointRefCount = config.counts.y;
        value->patchRefFirst = config.references.y;
        value->patchRefCount = config.counts.z;
        value->rootVotes = rootVotes;
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
        while (next.operation < plan.size() && dispatches.size() < dispatchLimit) {
            const auto &op = plan[next.operation];
            const uint32_t count = std::min(ChunkLimit(op.entry), op.length - next.first);
            if (!Emit(op, next.first, count, pendingUniforms, uploads, dispatches))
                break;
            if (phase == Phase::Connectivity && next.operation >= validationStart) {
                auto &unit = dispatches.back();
                unit.indirectArguments =
                    rhi::ComputeIndirectArguments{topologyArguments, (unit.groupCountX - 1u) * WordBytes};
            }
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
        if (boundary && !isolatedTopology) {
            if (submittedPhase == Phase::Connectivity && !localTopology)
                reads.push_back({schedule, uint64_t(config.layout.y) * WordBytes, WordBytes});
            else if (submittedPhase == Phase::Assemble)
                reads.push_back({schedule, uint64_t(config.layout.x) * WordBytes, 2u * WordBytes});
        }
        const auto resumeAfter =
            reads.empty() ? JoltGpuControlWork::ResumeAfter::Acceptance : JoltGpuControlWork::ResumeAfter::Completion;
        auto packet =
            rhi::PrepareComputeBatch(implementation->host, std::move(uploads), std::move(dispatches), std::move(reads));
        JoltGpuControlWork work{
            packet,
            [self = shared_from_this(), packet, next, boundary, submittedPhase]() -> JoltGpuControlWork {
                // Cursor changes follow acceptance. Boundary decode also needs completion.
                self->cursor = next;
                if (boundary) {
                    if (self->isolatedTopology || (submittedPhase == Phase::Connectivity && self->localTopology)) {
                        self->PublishTopology();
                        return {};
                    }
                    if (submittedPhase == Phase::Connectivity)
                        self->DecodeConnectivity(packet);
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
    std::shared_ptr<rhi::ComputeBuffer> schedule, parents, rootVotes, topologyArguments;
    Uniforms pendingUniforms;
    std::function<void(std::shared_ptr<const JoltGpuConstraintWorkset>)> completed;
    std::function<void(std::shared_ptr<JoltGpuConstraintTopology>)> completedTopology;
    bool isolatedTopology = false;
    bool localTopology = false;
    ConstraintParams config{};
    std::array<uint32_t, 2> jointSort{}, patchSort{};
    std::vector<ScanLevel> hierarchy;
    std::vector<Operation> plan;
    Cursor cursor;
    Phase phase = Phase::Connectivity;
    size_t validationStart = 0u;
    uint32_t connectivityPass = 0u;
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
    return std::make_shared<Impl::Execution>(m_impl, storage, inputs, parameters, std::move(completed))->Next();
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
        // The accepted broadphase and joint range prove that no numerical rows
        // exist. Keep the queued root write as the sleep pass's dependency.
        execution->completed = std::move(completed);
        topology->execution.reset();
        execution->Install({}, {});
        return {};
    }
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
    execution->completed = std::move(completed);
    topology->execution.reset();
    execution->BuildAssembly();
    return execution->Next();
}
} // namespace infernux
