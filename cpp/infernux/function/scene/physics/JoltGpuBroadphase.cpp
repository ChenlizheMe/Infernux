#include "JoltGpuBroadphase.h"
#include "JoltGpuProgramSources.h"

#include <Jolt/GPU/BroadphaseTypes.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <optional>
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
    Parameters,
    Work,
    Bodies,
    Shapes,
    AuthorWorlds,
    Bounds,
    Joints,
    RuntimeWorlds,
    Arena,
    Arguments,
    ColliderPairs,
    ResidentStep,
    ResidentContacts,
    CandidateArena,
    PortCount
};
enum Entry : uint8_t
{
    InitWords,
    SeedActive,
    PrepareIndirect,
    BuildSort,
    BuildSuppression,
    Merge,
    SortCandidateTiles,
    FindEnds,
    CountRound,
    FillRound,
    ScanBlocks,
    ReduceBlocks,
    AddOffsets,
    CompactActive,
    AccumulateStatus,
    PublishActive,
    MaterializeCounts,
    StoreOffsets,
    PublishCount,
    BuildWorldRanges,
    ResetFill,
    CheckFill,
    PublishFill,
    SetCandidateBank,
    InitContacts,
    PublishResidentCount,
    InitCandidates,
    BuildSmallPairs,
    EntryCount
};
constexpr std::array<std::string_view, EntryCount> EntryNames{"init_words",
                                                              "seed_active",
                                                              "prepare_indirect_arguments",
                                                              "build_sort_records",
                                                              "build_suppression",
                                                              "merge_items",
                                                              "sort_candidate_tiles",
                                                              "find_sap_ends",
                                                              "count_round",
                                                              "fill_round",
                                                              "scan_blocks",
                                                              "reduce_blocks",
                                                              "add_scan_offsets",
                                                              "compact_active",
                                                              "accumulate_status_summary",
                                                              "publish_active_summary",
                                                              "materialize_counts",
                                                              "store_rank_offsets",
                                                              "publish_count_summary",
                                                              "build_world_ranges",
                                                              "reset_fill_ranks",
                                                              "check_fill",
                                                              "publish_fill_summary",
                                                              "set_candidate_bank",
                                                              "init_contacts",
                                                              "publish_resident_count",
                                                              "init_candidates",
                                                              "build_small_pairs"};

constexpr uint32_t Bit(Port port)
{
    return 1u << port;
}
constexpr uint32_t A = Bit(Arena), P = Bit(Parameters), K = Bit(Work);
constexpr std::array<uint32_t, EntryCount> EntryPorts{
    A | P | K,
    A | P,
    A | Bit(Arguments) | P | K,
    A | P | K | Bit(Shapes) | Bit(Bodies) | Bit(AuthorWorlds) | Bit(Bounds),
    A | P | K | Bit(Joints) | Bit(Bodies) | Bit(AuthorWorlds),
    A | P | K | Bit(CandidateArena),
    A | P | K | Bit(CandidateArena),
    A | P | K | Bit(Bounds),
    A | P | K | Bit(Shapes) | Bit(Bodies) | Bit(Bounds) | Bit(ColliderPairs) | Bit(CandidateArena),
    A | P | K | Bit(Shapes) | Bit(Bodies) | Bit(Bounds) | Bit(ColliderPairs) | Bit(CandidateArena),
    A | P | K,
    A | P | K,
    A | P | K,
    A | P | K,
    A | P,
    A | P,
    A | P | K,
    A | P | K,
    A | P,
    A | P | K | Bit(AuthorWorlds) | Bit(RuntimeWorlds),
    A | P | K,
    A | P | K,
    A | P,
    A | P | Bit(CandidateArena),
    P | K | Bit(ResidentContacts),
    A | P | Bit(ResidentStep),
    P | K | Bit(CandidateArena),
    A | P | Bit(Bodies) | Bit(Shapes) | Bit(AuthorWorlds) | Bit(Bounds) | Bit(Joints) | Bit(ColliderPairs) |
        Bit(CandidateArena)};

struct PortLayout
{
    std::string_view name;
    uint64_t stride;
    std::array<std::string_view, 14> fields;
    uint32_t integers;
    bool readOnly;
};
constexpr PortLayout Ports[] = {
    {"broadphase",
     sizeof(BroadphaseParams),
     {"counts", "workspace", "rankRows", "activeQueues", "round", "candidates", "merge", "mergeKey", "scan",
      "scanParent", "activeState", "summary", "selection", "selectedCounts"},
     0x3fffu,
     true},
    {"stage_work", sizeof(StageWork), {"range"}, 1u, true},
    {"body_models", sizeof(BodyModel), {"meta", "mass", "invInertiaDiagonal", "inertiaRotation", "damping"}, 1u, true},
    {"shape_models",
     sizeof(ShapeModel),
     {"meta", "geometry", "localPositionCOM", "localRotation", "material", "surface", "source", "geometryReference"},
     (1u << 0) | (1u << 5) | (1u << 6) | (1u << 7),
     true},
    {"author_worlds", sizeof(WorldRange), {"bodies", "leaves"}, 3u, true},
    {"bounds", sizeof(Aabb), {"minimum", "maximum"}, 0u, true},
    {"joint_stream", sizeof(QueryRecordWord), {"value"}, 1u, true},
    {"runtime_worlds", sizeof(WorldRange), {"bodies", "leaves"}, 3u, false},
    {"contact_patches", sizeof(ContactPatch), {"leaves"}, 1u, false},
    {"indirect_arguments", sizeof(QueryRecordWord), {"value"}, 1u, false},
    {"collider_pair_ignores", sizeof(ColliderPairKey), {"a", "b"}, 3u, true},
    {"resident_step", sizeof(QueryRecordWord), {"value"}, 1u, false},
    {"resident_contacts",
     sizeof(Contact),
     {"meta", "normalDepth", "localAnchorA", "localAnchorB", "impulse"},
     1u,
     false},
    {"resident_patches", sizeof(ContactPatch), {"leaves"}, 1u, false}};

constexpr uint32_t MaxDispatchesPerPacket = 16;
constexpr uint32_t ActivePassesPerWindow = 4;
constexpr uint32_t NeighborTestsPerDispatch = BroadphaseItemLimit * 128u;
constexpr uint64_t WordBytes = sizeof(UInt4);

template <class T> std::vector<uint8_t> Bytes(const T &value)
{
    std::vector<uint8_t> result(sizeof(T));
    std::memcpy(result.data(), &value, sizeof(T));
    return result;
}

uint32_t Blocks(uint32_t count)
{
    return count / BroadphaseBlockSize + static_cast<uint32_t>(count % BroadphaseBlockSize != 0);
}

void ValidateLayout(Port port, const ShaderResourceLayout &resource)
{
    const auto &expected = Ports[port];
    const bool uniform = port == Parameters || port == Work;
    const auto &value = resource.value;
    if (resource.binding.type != (uniform ? rhi::BindingType::UniformBuffer : rhi::BindingType::StorageBuffer) ||
        (uniform ? value.byteSize : resource.elementStride) != expected.stride ||
        (!uniform && resource.binding.readOnlyStorage != expected.readOnly) || value.kind != ShaderValueKind::Struct ||
        value.fields.size() != expected.stride / WordBytes)
        throw std::invalid_argument("Jolt GPU broadphase target differs from its shared layout");
    for (size_t index = 0; index < value.fields.size(); ++index) {
        const auto &field = value.fields[index];
        const auto scalar = (expected.integers & (1u << index)) ? ShaderScalarKind::UInt32 : ShaderScalarKind::Float32;
        if (field.name != expected.fields[index] || field.offset != index * WordBytes || field.byteSize != WordBytes ||
            field.kind != ShaderValueKind::Vector || field.scalar != scalar || field.rows != 1 || field.columns != 4)
            throw std::invalid_argument("Jolt GPU broadphase target field differs from its shared layout");
    }
}

bool SummaryEntry(Entry entry)
{
    return entry == SeedActive || entry == AccumulateStatus || entry == PublishActive || entry == PublishCount ||
           entry == PublishFill || entry == SetCandidateBank || entry == PublishResidentCount;
}
} // namespace

class JoltGpuBroadphase::Impl
{
  public:
    struct Program
    {
        std::shared_ptr<rhi::ComputeKernel> kernel;
        std::vector<Port> ports;
    };
    struct Execution;
    explicit Impl(rhi::ComputeHost &source, const ShaderTargetCode &target) : host(source)
    {
        for (uint8_t index = 0; index < EntryCount; ++index) {
            const auto entry = static_cast<Entry>(index);
            const auto code = std::find_if(target.entries.begin(), target.entries.end(),
                                           [&](const auto &item) { return item.name == EntryNames[index]; });
            const uint32_t threads = SummaryEntry(entry) ? 1u : BroadphaseBlockSize;
            if (code == target.entries.end() || code->stage != rhi::ShaderStage::Compute ||
                code->workgroupSize != std::array<uint32_t, 3>{threads, 1u, 1u})
                throw std::invalid_argument("Jolt GPU broadphase target is missing its shared compute entry");
            auto &program = programs[index];
            std::vector<rhi::ComputeBufferBinding> bindings;
            uint32_t actual = 0, storageCount = 0;
            for (const auto &resource : target.resources) {
                const std::array<uint32_t, 2> key{resource.set, resource.binding.binding};
                if (std::find(code->bindings.begin(), code->bindings.end(), key) == code->bindings.end())
                    continue;
                const auto found = std::find_if(std::begin(Ports), std::end(Ports),
                                                [&](const auto &port) { return port.name == resource.name; });
                if (found == std::end(Ports) || resource.set != 0)
                    throw std::invalid_argument("Jolt GPU broadphase entry contains an unknown resource");
                const auto port = static_cast<Port>(found - std::begin(Ports));
                if (actual & Bit(port))
                    throw std::invalid_argument("Jolt GPU broadphase entry repeats a resource");
                ValidateLayout(port, resource);
                actual |= Bit(port);
                storageCount += resource.binding.type == rhi::BindingType::StorageBuffer;
                program.ports.push_back(port);
                bindings.push_back({resource.binding.binding, resource.binding.type, resource.binding.readOnlyStorage});
            }
            if (actual != EntryPorts[index] || code->bindings.size() != program.ports.size() || storageCount > 8u)
                throw std::invalid_argument("Jolt GPU broadphase entry differs from its shared resource set");
            program.kernel = PrepareJoltGpuKernel(host, target.target, *code, std::move(bindings));
        }
    }
    // A numeric capacity hint owns no optional GPU allocations. The current
    // workset alone retains the values still required by its real consumers.
    std::array<uint32_t, 2> recentCapacity{};
    rhi::ComputeHost host;
    std::array<Program, EntryCount> programs;
};

struct JoltGpuBroadphase::Impl::Execution : std::enable_shared_from_this<Execution>
{
    enum class Phase
    {
        InitialCapacity,
        Initialize,
        InitializeCandidates,
        SmallPairs,
        Suppression,
        SuppressionStatus,
        SuppressionStatusPublish,
        SuppressionSort,
        SortRecords,
        SapSort,
        Ends,
        Count,
        Counts,
        CountScan,
        Offsets,
        CountPublish,
        Worlds,
        CountBoundary,
        FinalCapacity,
        FinalInitialize,
        FinalContacts,
        ResidentPublish,
        Reset,
        Fill,
        FillCheck,
        FillReduce,
        FillPublish,
        ActiveBoundary,
        CandidateSort,
        CandidatePublish,
        Collide,
        Done
    };
    struct ScanLevel
    {
        uint32_t input, prefix, total, capacity;
    };
    struct ScanCursor
    {
        bool active = false, exclusive = false, descending = false, dynamic = false;
        uint32_t originalInput = 0, level = 0, first = 0, root = 0;
        std::vector<uint32_t> counts;
    };
    struct SweepTail
    {
        uint32_t ranks, neighbors;
    };

    Execution(std::shared_ptr<Impl> owner, const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
              JoltGpuPrograms &programs, const StepParams &settings,
              std::function<void(std::shared_ptr<const JoltGpuContactWorkset>)> callback, bool swept)
        : implementation(std::move(owner)), stagePrograms(programs), storageOwner(storage), acceptedInputs(inputs),
          parameters(settings), completed(std::move(callback))
    {
        if (!implementation->host.SharesServicesWith(storage.GetHost()) || !completed)
            throw std::invalid_argument("Jolt GPU broadphase requires its tick owner and completion callback");
        const uint32_t n = storage.GetShapeCount(), j = storage.GetJointCount(), w = storage.GetWorldCount();
        if (!storage.GetBodyCount() || !w)
            throw std::invalid_argument("An empty GPU model does not require broadphase work");
        settingsPorts[Bodies] = inputs.GetCaptured(Resource::BodyModels);
        settingsPorts[Shapes] = inputs.GetCaptured(Resource::ShapeModels);
        settingsPorts[AuthorWorlds] = inputs.GetCaptured(Resource::Worlds);
        settingsPorts[Bounds] = storage.Get(Resource::Bounds) ? storage.Get(Resource::Bounds)->Freeze() : nullptr;
        // The exclusive arena writer keeps this topology fixed. Broadphase
        // reads only joint directory/author metadata, so representation-history
        // retirement may update runtime words without retaining a COW reader.
        settingsPorts[Joints] = storage.Get(Resource::Joints);
        if (!settingsPorts[Joints] || !implementation->host.SharesServicesWith(settingsPorts[Joints]->GetHost()))
            throw std::invalid_argument("GPU broadphase requires its exclusive writer joint stream");
        for (auto port : {Bodies, Shapes, AuthorWorlds})
            ValidateInput(settingsPorts[port]);
        if (n)
            ValidateInput(settingsPorts[Bounds]);
        const auto requireBytes = [&](Port port, uint64_t bytes) {
            if (!settingsPorts[port] || settingsPorts[port]->GetByteSize() != bytes)
                throw std::invalid_argument("Jolt GPU broadphase input extent differs from the accepted model");
        };
        requireBytes(Bodies, uint64_t(storage.GetBodyCount()) * sizeof(BodyModel));
        requireBytes(Shapes, uint64_t(std::max(1u, n)) * sizeof(ShapeModel));
        requireBytes(AuthorWorlds, uint64_t(w) * sizeof(WorldRange));
        requireBytes(Joints, storage.GetJointLayout()->ByteSize());
        if (n)
            requireBytes(Bounds, uint64_t(n) * sizeof(Aabb));
        config.counts = {n, j, w, storage.GetBodyCount()};
        config.selection = settings.selection;
        config.selectedCounts = settings.selectedCounts;
        const auto &pairs = inputs.GetColliderPairs();
        if (!pairs || pairs->keys.size() > UINT32_MAX)
            throw std::invalid_argument("Broadphase requires its accepted Collider policy");
        settingsPorts[ColliderPairs] = pairs->buffer;
        ValidateInput(settingsPorts[ColliderPairs]);
        requireBytes(ColliderPairs, uint64_t(std::max<size_t>(1, pairs->keys.size())) * sizeof(ColliderPairKey));
        config.selectedCounts.w = static_cast<uint32_t>(pairs->keys.size());
        const uint32_t leaves = config.selectedCounts.y;
        const uint32_t joints = config.selectedCounts.z;
        const bool smallPairs = leaves && leaves <= BroadphaseSmallSetLimit && joints <= BroadphaseSmallSetLimit;
        config.round.w = uint32_t(smallPairs);
        config.round.y = uint32_t(swept);
        uint64_t words = 0;
        const auto reserve = [&](uint64_t size) {
            if (size > UINT32_MAX || words > UINT32_MAX - size)
                throw std::overflow_error("Jolt GPU broadphase workspace exceeds device word indexing");
            const auto result = static_cast<uint32_t>(words);
            words += size;
            return result;
        };
        config.workspace.y = reserve(1);
        if (!smallPairs) {
            zeroRecord = reserve(1);
            sap = {reserve(leaves), reserve(leaves)};
            suppression = {reserve(joints), reserve(joints)};
            config.rankRows = {reserve(leaves), reserve(leaves), 0, 0};
            queues = {reserve(leaves), reserve(leaves)};
            activeMeta = {reserve(1), reserve(1)};
            uint32_t count = std::max({leaves, joints, 1u});
            do {
                const uint32_t input = reserve(count), prefix = reserve(count);
                levels.push_back({input, prefix, 0, count});
                const uint32_t next = Blocks(count);
                if (next == 1)
                    break;
                count = next;
            } while (true);
            const uint32_t terminal = reserve(1);
            for (size_t level = 0; level < levels.size(); ++level)
                levels[level].total = level + 1 < levels.size() ? levels[level + 1].input : terminal;
            config.workspace.z = sap[0];
            config.workspace.w = suppression[0];
            config.scan.x = levels[0].input;
        }
        scratchWords = static_cast<uint32_t>(words);
        config.workspace.x = scratchWords;
        parameters.counts.x = storage.GetBodyCount();
        parameters.counts.y = w;
        parameters.counts.z = n;
        parameters.solve.z = parameters.solve.w = 0;
        workset = std::make_shared<JoltGpuContactWorkset>();
        // A single group knows its complete pair upper bound from this World.
        ConfigureCapacity(smallPairs ? std::max(1u, leaves * (leaves - 1u) / 2u)
                                     : implementation->recentCapacity[config.round.y]);
    }

    uint32_t PointsPerPair() const
    {
        return config.round.y ? 1u : MaxContactPointsPerPair;
    }

    void ConfigureCapacity(uint32_t value)
    {
        if (value > UINT32_MAX / MaxContactPointsPerPair || 1ull + 2ull * value > UINT32_MAX)
            throw std::overflow_error("Resident broadphase capacity exceeds checked indexing");
        capacity = value;
        config.workspace.x = scratchWords;
        config.candidates = {1u, 1u + value, value, 0};
    }

    void ValidateInput(const std::shared_ptr<rhi::ComputeBuffer> &buffer) const
    {
        if (!buffer || !buffer->IsFrozen() || !implementation->host.SharesServicesWith(buffer->GetHost()))
            throw std::invalid_argument("Jolt GPU broadphase requires the frozen accepted input prefix");
    }

    std::shared_ptr<rhi::ComputeBuffer> Allocate(uint64_t bytes,
                                                 rhi::BufferUsageFlags usage = rhi::BufferUsageFlags::None)
    {
        if (!bytes || bytes % sizeof(uint32_t) || bytes > std::numeric_limits<size_t>::max() ||
            !implementation->host.device.GetCapabilities().IsStorageBufferRangeValid(0, bytes))
            throw std::length_error("Jolt GPU broadphase requested storage exceeds host or device range: " +
                                    std::to_string(bytes));
        auto buffer = rhi::ComputeBuffer::TryCreatePrivate(
            implementation->host, {bytes / sizeof(uint32_t), rhi::ComputeScalarType::UInt32, 1, usage});
        return buffer;
    }

    bool Endpoint(std::shared_ptr<rhi::ComputeBuffer> &buffer, uint64_t bytes,
                  rhi::BufferUsageFlags usage = rhi::BufferUsageFlags::None)
    {
        if (!buffer)
            buffer = Allocate(bytes, usage);
        if (buffer)
            return true;
        allocationBlocked = true;
        return false;
    }

    bool Capacity()
    {
        if (phase == Phase::InitialCapacity) {
            // Admit the finite count workspace before the optional historical
            // K hint. Until this branch succeeds no packet/cursor was accepted.
            for (uint32_t slot = 0; slot < MaxDispatchesPerPacket; ++slot)
                if (!Endpoint(parameterSlots[slot], sizeof(BroadphaseParams)) ||
                    !Endpoint(workSlots[slot], sizeof(StageWork)))
                    return false;
            if (!Endpoint(arena, uint64_t(scratchWords) * WordBytes) ||
                !Endpoint(workset->parameters, sizeof(StepParams)) ||
                !Endpoint(runtimeWorlds, uint64_t(config.counts.z) * sizeof(WorldRange)) ||
                !Endpoint(arguments, (MaxDispatchesPerPacket - 1u) * WordBytes,
                          rhi::BufferUsageFlags::Storage | rhi::BufferUsageFlags::Indirect))
                return false;
            if (!Endpoint(candidateArena, (1ull + 2ull * capacity) * WordBytes) ||
                !Endpoint(workset->contacts, uint64_t(std::max(1u, capacity * PointsPerPair())) * sizeof(Contact))) {
                if (capacity && !config.round.w) {
                    // A historical K is not a mandatory lower bound. Drop all
                    // partial optimistic owners, then let the SAME execution
                    // run its cold CountBoundary with K=0. Nothing is replayed.
                    candidateArena.reset();
                    workset->patches.reset();
                    workset->contacts.reset();
                    ConfigureCapacity(0u);
                }
                return false;
            }
            workset->patches = candidateArena;
            initialUploads = true;
            phase = Phase::Initialize;
        } else if (phase == Phase::FinalCapacity) {
            if (!Endpoint(candidateArena, (1ull + 2ull * capacity) * WordBytes) ||
                !Endpoint(workset->contacts, uint64_t(std::max(1u, capacity * PointsPerPair())) * sizeof(Contact)) ||
                !Endpoint(workset->parameters, sizeof(StepParams))) {
                // Historical growth slack is optional. Under real allocation
                // pressure release the partial grown owner, then retry exactly
                // the C already returned by this frozen count prefix.
                const uint32_t exact = std::max(1u, requiredCapacity);
                if (capacity > exact) {
                    workset->contacts.reset();
                    workset->patches.reset();
                    candidateArena.reset();
                    ConfigureCapacity(exact);
                }
                return false;
            }
            workset->patches = candidateArena;
            phase = Phase::FinalInitialize;
            item = 0;
        }
        return true;
    }

    bool EmitWork(Entry entry, const StageWork &work, const BroadphaseParams &value,
                  std::vector<rhi::ComputeBufferUpdate> &uploads, std::vector<rhi::ComputeDispatchDesc> &dispatches,
                  std::optional<rhi::ComputeIndirectArguments> indirect = {})
    {
        const bool block = entry == ScanBlocks || entry == ReduceBlocks;
        const uint32_t first = work.range.x, count = work.range.y;
        const uint32_t limit = entry == PrepareIndirect ? MaxDispatchesPerPacket - 1u
                               : block                  ? BroadphaseBlockLimit
                                                        : BroadphaseItemLimit;
        if (!count || count > limit)
            throw std::logic_error("Jolt GPU broadphase emitted an invalid bounded work unit");
        const auto slot = dispatches.size();
        if (slot >= packetLimit || !parameterSlots[slot] || !workSlots[slot])
            throw std::logic_error("Resident broadphase requires its admitted bounded uniform workspace");
        const auto &uniform = parameterSlots[slot];
        const auto &workUniform = workSlots[slot];
        uploads.push_back({uniform, 0, Bytes(value)});
        if (EntryPorts[entry] & K)
            uploads.push_back({workUniform, 0, Bytes(work)});
        const auto &program = implementation->programs[entry];
        rhi::ComputeDispatchDesc dispatch;
        dispatch.kernel = program.kernel;
        dispatch.groupCountX = block ? count : SummaryEntry(entry) ? 1u : Blocks(count);
        dispatch.indirectArguments = std::move(indirect);
        if (dispatch.groupCountX > implementation->host.device.GetCapabilities().limits.maxComputeWorkgroupCount[0])
            throw std::length_error("Jolt GPU broadphase bounded work exceeds dispatch limits");
        rhi::ComputeDispatchProof proof;
        for (const auto port : program.ports) {
            const auto binding = static_cast<uint32_t>(dispatch.buffers.size());
            const auto &buffer = port == Parameters         ? uniform
                                 : port == Work             ? workUniform
                                 : port == Arena            ? arena
                                 : port == RuntimeWorlds    ? runtimeWorlds
                                 : port == Arguments        ? arguments
                                 : port == ResidentContacts ? workset->contacts
                                 : port == ResidentStep     ? workset->parameters
                                 : port == CandidateArena   ? candidateArena
                                                            : settingsPorts[port];
            if (!buffer)
                throw std::logic_error("Jolt GPU broadphase work has an absent resource");
            dispatch.buffers.push_back(buffer);
            const bool initialize = (port == Arena && entry == InitWords) ||
                                    (port == ResidentContacts && entry == InitContacts) ||
                                    (port == CandidateArena && entry == InitCandidates);
            const bool worldWrite = port == RuntimeWorlds;
            const bool argumentWrite = port == Arguments;
            dispatch.bufferAccesses.push_back(
                initialize || worldWrite || argumentWrite                         ? rhi::ComputeBufferAccess::Write
                : port == Arena || port == CandidateArena || port == ResidentStep ? rhi::ComputeBufferAccess::ReadWrite
                                                                                  : rhi::ComputeBufferAccess::Read);
            if (initialize)
                proof.writes.push_back({binding,
                                        uint64_t(first) * (port == ResidentContacts ? sizeof(Contact) : WordBytes),
                                        uint64_t(count) * (port == ResidentContacts ? sizeof(Contact) : WordBytes)});
            else if (worldWrite)
                proof.writes.push_back(
                    {binding, uint64_t(first) * sizeof(WorldRange), uint64_t(count) * sizeof(WorldRange)});
            else if (argumentWrite)
                proof.writes.push_back({binding, 0, uint64_t(count) * WordBytes});
            else if (port == Bounds) {
                const uint64_t bytes = uint64_t(config.selectedCounts.y) * sizeof(Aabb);
                if (bytes)
                    proof.reads.push_back({binding, uint64_t(config.selection.z) * sizeof(Aabb), bytes});
            } else
                proof.reads.push_back({binding, 0, buffer->GetByteSize()});
        }
        dispatch.proof = std::move(proof);
        dispatches.push_back(std::move(dispatch));
        return true;
    }

    bool Emit(Entry entry, uint32_t first, uint32_t count, const BroadphaseParams &value,
              std::vector<rhi::ComputeBufferUpdate> &uploads, std::vector<rhi::ComputeDispatchDesc> &dispatches)
    {
        StageWork work{{first, count, 0, 0}};
        return EmitWork(entry, work, value, uploads, dispatches);
    }

    bool EmitIndirect(Entry entry, uint32_t first, uint32_t count, uint32_t size, const BroadphaseParams &value,
                      std::vector<rhi::ComputeBufferUpdate> &uploads, std::vector<rhi::ComputeDispatchDesc> &dispatches)
    {
        const bool block = entry == ScanBlocks || entry == ReduceBlocks;
        const uint32_t limit = block ? BroadphaseBlockLimit : BroadphaseItemLimit;
        const bool matching = indirectEntry == entry && indirectMeta == value.activeState.x &&
                              indirectLevel == value.activeState.z && first >= indirectFirst &&
                              (first - indirectFirst) % limit == 0 && (first - indirectFirst) / limit < indirectCount;
        if (!matching) {
            const uint32_t available = packetLimit - static_cast<uint32_t>(dispatches.size());
            if (available < 2u) {
                packetFull = true;
                return false;
            }
            const uint32_t remaining = (size - first) / limit + static_cast<uint32_t>((size - first) % limit != 0);
            const uint32_t records = std::min(available - 1u, remaining);
            if (!records || uint64_t(first) + uint64_t(records - 1u) * limit >= size)
                throw std::logic_error("Jolt GPU indirect chunk plan exceeds its bounded domain");
            const StageWork producer{{first, records, value.mergeKey.y == 1u ? 1u : 0u, block ? 1u : 0u}};
            // A producer is recorded here, after earlier queue/meta publishers;
            // it must not be hoisted to the front of the packet.
            if (!EmitWork(PrepareIndirect, producer, value, uploads, dispatches))
                return false;
            indirectEntry = entry;
            indirectMeta = value.activeState.x;
            indirectLevel = value.activeState.z;
            indirectFirst = first;
            indirectCount = records;
        }
        const uint32_t record = (first - indirectFirst) / limit;
        const StageWork work{{first, count, 0, 0}};
        if (!EmitWork(entry, work, value, uploads, dispatches,
                      rhi::ComputeIndirectArguments{arguments, uint64_t(record) * WordBytes}))
            return false;
        return true;
    }

    bool Range(Entry entry, uint32_t size, std::vector<rhi::ComputeBufferUpdate> &uploads,
               std::vector<rhi::ComputeDispatchDesc> &dispatches)
    {
        if (item == size) {
            item = 0;
            return false;
        }
        const uint32_t count = std::min(BroadphaseItemLimit, size - item);
        if (!Emit(entry, item, count, config, uploads, dispatches))
            return true;
        item += count;
        return true;
    }

    void BeginScan(uint32_t input, uint32_t count, bool exclusive, bool dynamic = false)
    {
        scan = {};
        scan.active = count != 0;
        scan.exclusive = exclusive;
        scan.originalInput = input;
        scan.dynamic = dynamic;
        scan.root = zeroRecord;
        if (count) {
            do {
                scan.counts.push_back(count);
                count = Blocks(count);
            } while (count > 1);
        }
    }

    bool Scan(std::vector<rhi::ComputeBufferUpdate> &uploads, std::vector<rhi::ComputeDispatchDesc> &dispatches)
    {
        if (!scan.active)
            return false;
        auto value = config;
        const uint32_t level = scan.level, size = scan.counts[level];
        const auto &addresses = levels[level];
        value.activeState.z = level;
        value.activeState.w = scan.dynamic ? 1u : 0u;
        value.scan = {level ? levels[level].input : scan.originalInput, addresses.prefix, addresses.total, size};
        if (!scan.descending) {
            const uint32_t count = std::min(BroadphaseBlockLimit, Blocks(size) - scan.first);
            const auto entry = scan.exclusive ? ScanBlocks : ReduceBlocks;
            const bool emitted = scan.dynamic
                                     ? EmitIndirect(entry, scan.first, count, Blocks(size), value, uploads, dispatches)
                                     : Emit(entry, scan.first, count, value, uploads, dispatches);
            if (!emitted)
                return true;
            scan.first += count;
            if (scan.first == Blocks(size)) {
                scan.first = 0;
                if (level + 1u < scan.counts.size())
                    ++scan.level;
                else {
                    scan.root = addresses.total;
                    if (scan.exclusive && level) {
                        scan.descending = true;
                        --scan.level;
                    } else
                        scan.active = false;
                }
            }
        } else {
            value.scanParent.x = levels[level + 1u].prefix;
            const uint32_t count = std::min(BroadphaseItemLimit, size - scan.first);
            const bool emitted = scan.dynamic
                                     ? EmitIndirect(AddOffsets, scan.first, count, size, value, uploads, dispatches)
                                     : Emit(AddOffsets, scan.first, count, value, uploads, dispatches);
            if (!emitted)
                return true;
            scan.first += count;
            if (scan.first == size) {
                scan.first = 0;
                if (level)
                    --scan.level;
                else
                    scan.active = false;
            }
        }
        return true;
    }

    void BeginSort(uint32_t first, uint32_t second, uint32_t count, uint32_t key)
    {
        sortSource = first;
        sortTarget = second;
        sortCount = count;
        sortKey = key;
        // SAP and suppression records are sorted when seeded. Filled pairs
        // enter as unsorted device-counted rows and seed their own tiles.
        sortWidth = key == 2u ? 1u : BroadphaseBlockSize;
        item = 0;
    }

    bool Sort(std::vector<rhi::ComputeBufferUpdate> &uploads, std::vector<rhi::ComputeDispatchDesc> &dispatches)
    {
        if (sortWidth >= sortCount)
            return false;
        auto value = config;
        value.merge = {sortSource, sortTarget, sortCount, sortWidth};
        value.mergeKey.x = sortKey;
        const uint32_t count = std::min(BroadphaseItemLimit, sortCount - item);
        value.mergeKey.y = sortKey == 2u ? 1u : 0u;
        const bool seed = sortWidth == 1u;
        const auto entry = seed ? SortCandidateTiles : Merge;
        if (!(sortKey == 2u ? EmitIndirect(entry, item, count, sortCount, value, uploads, dispatches)
                            : Emit(Merge, item, count, value, uploads, dispatches)))
            return true;
        item += count;
        if (item == sortCount) {
            item = 0;
            if (seed) {
                sortWidth = BroadphaseBlockSize;
            } else {
                std::swap(sortSource, sortTarget);
                // A final width of 2^31 completes every uint32-sized domain.
                sortWidth = sortWidth > UINT32_MAX / 2u ? sortCount : sortWidth * 2u;
            }
        }
        return true;
    }

    enum class ActiveStage
    {
        Seed,
        Round,
        Scan,
        Compact,
        Publish
    };

    void BeginActive(bool fill)
    {
        activeFilling = fill;
        config.round.z = fill ? 1u : 0u;
        activeStage = ActiveStage::Seed;
        activeBound = config.selectedCounts.y;
        neighborsRemaining = std::max(1u, activeBound) - 1u;
        windowPasses = 0;
        fillTail = 0;
        config.activeQueues = {queues[0], queues[1], 0, 0};
        config.activeState = {activeMeta[0], activeMeta[1], 0, 0};
    }

    void ContinueSweep(SweepTail tail)
    {
        activeBound = tail.ranks;
        neighborsRemaining = tail.neighbors;
        windowPasses = 0;
        phase = activeFilling ? Phase::Fill : Phase::Count;
    }

    bool ActiveRange(Entry entry, std::vector<rhi::ComputeBufferUpdate> &uploads,
                     std::vector<rhi::ComputeDispatchDesc> &dispatches)
    {
        if (item == activeBound) {
            item = 0;
            return false;
        }
        auto value = config;
        value.activeState.z = 0;
        value.activeState.w = 1;
        const uint32_t count = std::min(BroadphaseItemLimit, activeBound - item);
        if (!EmitIndirect(entry, item, count, activeBound, value, uploads, dispatches))
            return true;
        item += count;
        return true;
    }

    bool ActiveRounds(std::vector<rhi::ComputeBufferUpdate> &uploads, std::vector<rhi::ComputeDispatchDesc> &dispatches)
    {
        while (true) {
            switch (activeStage) {
            case ActiveStage::Seed:
                if (!Emit(SeedActive, 0, 1, config, uploads, dispatches))
                    return true;
                activeStage = ActiveStage::Round;
                return true;
            case ActiveStage::Round: {
                config.scan.x = levels[0].input;
                if (!neighborsRemaining) {
                    // The final count/fill reduction already verifies every cursor.
                    phase = activeFilling ? Phase::FillCheck : Phase::Counts;
                    return false;
                }
                const uint32_t ranksPerDispatch = std::min(activeBound, BroadphaseItemLimit);
                // Give shorter queues more work per rank within the same dispatch limit.
                config.round.x = std::min(
                    {neighborsRemaining, BroadphaseNeighborLimit, NeighborTestsPerDispatch / ranksPerDispatch});
                if (ActiveRange(activeFilling ? FillRound : CountRound, uploads, dispatches))
                    return true;
                neighborsRemaining -= config.round.x;
                if (!neighborsRemaining || ++windowPasses < ActivePassesPerWindow)
                    break;
                // Fill traverses the same frozen spans. Its last window can
                // also skip compaction when count proved the queue exhausted.
                if (activeFilling && countTails[fillTail].ranks == 0) {
                    neighborsRemaining = 0;
                    break;
                }
                BeginScan(levels[0].input, activeBound, true, true);
                activeStage = ActiveStage::Scan;
                break;
            }
            case ActiveStage::Scan:
                if (Scan(uploads, dispatches))
                    return true;
                config.scan.x = levels[0].input;
                config.scan.y = levels[0].prefix;
                config.summary.x = scan.root;
                activeStage = ActiveStage::Compact;
                break;
            case ActiveStage::Compact:
                if (ActiveRange(CompactActive, uploads, dispatches))
                    return true;
                activeStage = ActiveStage::Publish;
                break;
            case ActiveStage::Publish:
                if (!Emit(PublishActive, 0, 1, config, uploads, dispatches))
                    return true;
                std::swap(config.activeQueues.x, config.activeQueues.y);
                std::swap(config.activeState.x, config.activeState.y);
                activeStage = ActiveStage::Round;
                if (activeFilling)
                    ContinueSweep(countTails[fillTail++]);
                else
                    phase = Phase::ActiveBoundary;
                return true;
            }
        }
    }

    void DecodeActive(const std::shared_ptr<rhi::PreparedComputeBatch> &packet)
    {
        UInt4 value{};
        const auto read = packet->Readback(0);
        if (!read || !read->IsComplete())
            throw std::logic_error("Jolt GPU active queue requires its completed source receipt");
        read->ReadInto(&value, sizeof(value));
        if (value.z)
            throw std::runtime_error("Jolt GPU active queue failed numerical validation (status " +
                                     std::to_string(value.z) + ")");
        if (value.x > activeBound || value.y > neighborsRemaining || (value.x == 0) != (value.y == 0))
            throw std::runtime_error("Jolt GPU active queue exceeded its remaining sweep range");
        countTails.push_back({value.x, value.y});
        ContinueSweep(countTails.back());
    }

    bool NextDispatch(std::vector<rhi::ComputeBufferUpdate> &uploads, std::vector<rhi::ComputeDispatchDesc> &dispatches)
    {
        // Only bounded phase transitions run here; item work is emitted lazily.
        while (true) {
            switch (phase) {
            case Phase::Initialize:
                // Scratch is reset each step. Capacity suffix needs a real
                // direct initializer only when this allocation is new.
                if (Range(InitWords, scratchWords, uploads, dispatches))
                    return true;
                item = candidateArena->HasInitializedRange(0, candidateArena->GetByteSize())
                           ? static_cast<uint32_t>(1ull + 2ull * capacity)
                           : 0u;
                phase = Phase::InitializeCandidates;
                break;
            case Phase::InitializeCandidates:
                if (Range(InitCandidates, static_cast<uint32_t>(1ull + 2ull * capacity), uploads, dispatches))
                    return true;
                afterContacts = config.round.w ? Phase::SmallPairs : Phase::Suppression;
                phase = workset->contacts->HasInitializedRange(0, workset->contacts->GetByteSize())
                            ? afterContacts
                            : Phase::FinalContacts;
                break;
            case Phase::SmallPairs:
                if (!Emit(BuildSmallPairs, 0, 1, config, uploads, dispatches))
                    return true;
                phase = Phase::Worlds;
                return true;
            case Phase::Suppression:
                config.merge.x = suppression[0];
                if (Range(BuildSuppression, config.selectedCounts.z, uploads, dispatches))
                    return true;
                BeginScan(config.scan.x, config.selectedCounts.z, false);
                phase = Phase::SuppressionStatus;
                break;
            case Phase::SuppressionStatus:
                if (Scan(uploads, dispatches))
                    return true;
                config.summary.x = scan.root;
                phase = Phase::SuppressionStatusPublish;
                break;
            case Phase::SuppressionStatusPublish:
                if (!Emit(AccumulateStatus, 0, 1, config, uploads, dispatches))
                    return true;
                BeginSort(suppression[0], suppression[1], config.selectedCounts.z, 1);
                phase = Phase::SuppressionSort;
                return true;
            case Phase::SuppressionSort:
                if (Sort(uploads, dispatches))
                    return true;
                config.workspace.w = sortSource;
                phase = Phase::SortRecords;
                break;
            case Phase::SortRecords:
                config.merge.x = sap[0];
                if (Range(BuildSort, config.selectedCounts.y, uploads, dispatches))
                    return true;
                BeginSort(sap[0], sap[1], config.selectedCounts.y, 0);
                phase = Phase::SapSort;
                break;
            case Phase::SapSort:
                if (Sort(uploads, dispatches))
                    return true;
                config.workspace.z = sortSource;
                BeginActive(false);
                phase = Phase::Ends;
                break;
            case Phase::Ends:
                if (Range(FindEnds, config.selectedCounts.y, uploads, dispatches))
                    return true;
                phase = Phase::Count;
                break;
            case Phase::Count:
            case Phase::Fill:
                if (ActiveRounds(uploads, dispatches))
                    return true;
                break;
            case Phase::Counts:
                if (Range(MaterializeCounts, config.selectedCounts.y, uploads, dispatches))
                    return true;
                BeginScan(config.scan.x, config.selectedCounts.y, true);
                phase = Phase::CountScan;
                break;
            case Phase::CountScan:
                if (Scan(uploads, dispatches))
                    return true;
                config.scan.y = levels[0].prefix;
                config.summary.x = scan.root;
                phase = Phase::Offsets;
                break;
            case Phase::Offsets:
                if (Range(StoreOffsets, config.selectedCounts.y, uploads, dispatches))
                    return true;
                phase = Phase::CountPublish;
                break;
            case Phase::CountPublish:
                if (!Emit(PublishCount, 0, 1, config, uploads, dispatches))
                    return true;
                phase = Phase::Worlds;
                return true;
            case Phase::Worlds:
                if (Range(BuildWorldRanges, config.counts.z, uploads, dispatches))
                    return true;
                phase = capacity ? Phase::ResidentPublish : Phase::CountBoundary;
                break;
            case Phase::FinalInitialize:
                if (Range(InitCandidates, static_cast<uint32_t>(1ull + 2ull * capacity), uploads, dispatches))
                    return true;
                afterContacts = Phase::ResidentPublish;
                phase = Phase::FinalContacts;
                break;
            case Phase::FinalContacts:
                if (Range(InitContacts, std::max(1u, capacity * PointsPerPair()), uploads, dispatches))
                    return true;
                phase = afterContacts;
                break;
            case Phase::ResidentPublish:
                if (!Emit(PublishResidentCount, 0, 1, config, uploads, dispatches))
                    return true;
                if (config.round.w) {
                    phase = config.round.y ? Phase::Done : Phase::Collide;
                    return true;
                }
                BeginActive(true);
                phase = Phase::Reset;
                return true;
            case Phase::Reset:
                if (Range(ResetFill, config.selectedCounts.y, uploads, dispatches))
                    return true;
                phase = Phase::Fill;
                break;
            case Phase::FillCheck:
                if (Range(CheckFill, config.selectedCounts.y, uploads, dispatches))
                    return true;
                BeginScan(config.scan.x, config.selectedCounts.y, false);
                phase = Phase::FillReduce;
                break;
            case Phase::FillReduce:
                if (Scan(uploads, dispatches))
                    return true;
                config.summary.x = scan.root;
                phase = Phase::FillPublish;
                break;
            case Phase::FillPublish:
                if (!Emit(PublishFill, 0, 1, config, uploads, dispatches))
                    return true;
                BeginSort(config.candidates.x, config.candidates.y, capacity, 2);
                phase = Phase::CandidateSort;
                return true;
            case Phase::CandidateSort:
                if (Sort(uploads, dispatches))
                    return true;
                config.merge.x = sortSource;
                phase = Phase::CandidatePublish;
                break;
            case Phase::CandidatePublish:
                if (!Emit(SetCandidateBank, 0, 1, config, uploads, dispatches))
                    return true;
                item = 0;
                phase = capacity && !config.round.y ? Phase::Collide : Phase::Done;
                return true;
            case Phase::Collide: {
                if (item == capacity) {
                    item = 0;
                    phase = Phase::Done;
                    break;
                }
                auto value = config;
                value.mergeKey.y = 1u;
                // The same producer/record contract as Merge; Collide's
                // arguments are consumed by the original stage kernel.
                const bool matching = indirectEntry == EntryCount && indirectCount && item >= indirectFirst &&
                                      (item - indirectFirst) % BroadphaseItemLimit == 0u &&
                                      (item - indirectFirst) / BroadphaseItemLimit < indirectCount;
                if (!matching) {
                    const uint32_t available = packetLimit - static_cast<uint32_t>(dispatches.size());
                    if (available < 2u) {
                        packetFull = true;
                        return true;
                    }
                    const uint32_t remaining = (capacity - item) / BroadphaseItemLimit +
                                               static_cast<uint32_t>((capacity - item) % BroadphaseItemLimit != 0u);
                    const uint32_t records = std::min(available - 1u, remaining);
                    const StageWork producer{{item, records, 1u, 0u}};
                    if (!EmitWork(PrepareIndirect, producer, value, uploads, dispatches))
                        return true;
                    indirectEntry = EntryCount;
                    indirectFirst = item;
                    indirectCount = records;
                }
                const uint32_t count = std::min(BroadphaseItemLimit, capacity - item);
                const uint32_t record = (item - indirectFirst) / BroadphaseItemLimit;
                stagePrograms.AppendResidentCollide(storageOwner, acceptedInputs, workset, capacity,
                                                    StageWork{{item, count, 0u, 0u}}, workSlots[dispatches.size()],
                                                    {arguments, uint64_t(record) * WordBytes}, uploads, dispatches);
                item += count;
                if (item == capacity) {
                    item = 0;
                    phase = Phase::Done;
                }
                return true;
            }
            default:
                return false;
            }
        }
    }

    void Decode(const std::shared_ptr<rhi::PreparedComputeBatch> &packet, bool count)
    {
        const auto read = packet->Readback(0);
        if (!read || !read->IsComplete())
            throw std::logic_error("Jolt GPU broadphase metadata requires its completed source receipt");
        UInt4 result{};
        read->ReadInto(&result, sizeof(result));
        constexpr uint32_t CapacityNeeded = 128u;
        if (result.z & ~CapacityNeeded)
            throw std::runtime_error("Jolt GPU broadphase numerical validation failed (status " +
                                     std::to_string(result.z) + ")");
        if (result.y || result.x > UINT32_MAX / MaxContactPointsPerPair || result.w > result.x ||
            (result.x == 0 && result.w != 0))
            throw std::runtime_error("Jolt GPU broadphase returned invalid candidate metadata");
        if (count || (result.z & CapacityNeeded)) {
            if (!count && result.x <= capacity)
                throw std::runtime_error("Resident broadphase reported a false capacity gap");
            workset->patchCount = result.x;
            workset->maxPatchesPerWorld = result.w;
            requiredCapacity = result.x;
            // Grow only after a real GPU C>K result; never derive from B^2.
            const uint64_t previousCapacity = capacity;
            const uint64_t grown =
                std::max<uint64_t>(std::max(1u, result.x), previousCapacity + (previousCapacity + 1u) / 2u);
            const uint64_t binding = implementation->host.device.GetCapabilities().portable.maxStorageBufferBinding;
            const uint64_t contactLimit = binding / (uint64_t(PointsPerPair()) * sizeof(Contact));
            const uint64_t candidateLimit = binding < WordBytes ? 0u : (binding / WordBytes - 1u) / 2u;
            const uint64_t maximum = std::min<uint64_t>(
                {UINT32_MAX / MaxContactPointsPerPair, (uint64_t(UINT32_MAX) - 1u) / 2u, contactLimit, candidateLimit});
            ConfigureCapacity(static_cast<uint32_t>(std::min(grown, maximum)));
            if (capacity < result.x)
                throw std::length_error("Resident broadphase cannot admit required candidate capacity");
            // Old Contact values are never a suffix input. Release them before
            // growth instead of retaining our own obsolete capacity owner.
            workset->contacts.reset();
            workset->patches.reset();
            candidateArena.reset();
            phase = Phase::FinalCapacity;
        } else {
            workset->patchCount = result.x;
            workset->maxPatchesPerWorld = result.w;
            if (!config.round.w)
                implementation->recentCapacity[config.round.y] =
                    result.x < capacity / 4u ? std::max(1u, capacity / 2u) : capacity;
        }
    }

    void Restore(const Execution &s)
    {
        settingsPorts = s.settingsPorts;
        arena = s.arena;
        runtimeWorlds = s.runtimeWorlds;
        arguments = s.arguments;
        candidateArena = s.candidateArena;
        parameterSlots = s.parameterSlots;
        workSlots = s.workSlots;
        workset = s.workset;
        parameters = s.parameters;
        config = s.config;
        completed = s.completed;
        levels = s.levels;
        scan = s.scan;
        phase = s.phase;
        sap = s.sap;
        suppression = s.suppression;
        queues = s.queues;
        activeMeta = s.activeMeta;
        scratchWords = s.scratchWords;
        zeroRecord = s.zeroRecord;
        item = s.item;
        sortSource = s.sortSource;
        sortTarget = s.sortTarget;
        sortCount = s.sortCount;
        sortKey = s.sortKey;
        sortWidth = s.sortWidth;
        activeStage = s.activeStage;
        indirectEntry = s.indirectEntry;
        activeBound = s.activeBound;
        neighborsRemaining = s.neighborsRemaining;
        windowPasses = s.windowPasses;
        fillTail = s.fillTail;
        indirectMeta = s.indirectMeta;
        indirectLevel = s.indirectLevel;
        indirectFirst = s.indirectFirst;
        indirectCount = s.indirectCount;
        initialUploads = s.initialUploads;
        allocationBlocked = s.allocationBlocked;
        packetFull = s.packetFull;
        activeFilling = s.activeFilling;
        capacity = s.capacity;
        requiredCapacity = s.requiredCapacity;
        afterContacts = s.afterContacts;
    }

    JoltGpuControlWork Next(uint32_t limit = MaxDispatchesPerPacket)
    {
        // Only an unaccepted prefix may contract. The checkpoint contains the
        // producer cursor, never a replay of already accepted work.
        packetLimit = limit;
        allocationBlocked = packetFull = false;
        indirectEntry = EntryCount;
        indirectCount = 0;
        if (!Capacity())
            return {nullptr, [self = shared_from_this()] { return self->Next(); }};
        auto checkpoint = std::make_shared<Execution>(*this);
        std::vector<rhi::ComputeBufferUpdate> uploads;
        std::vector<rhi::ComputeDispatchDesc> dispatches;
        if (initialUploads) {
            uploads.push_back({workset->parameters, 0, Bytes(parameters)});
            initialUploads = false;
        }
        while (dispatches.size() < packetLimit) {
            if (!NextDispatch(uploads, dispatches))
                break;
            if (allocationBlocked || packetFull)
                break;
            if (phase == Phase::CountBoundary || phase == Phase::ActiveBoundary || phase == Phase::Done)
                break;
        }
        const auto boundary = phase;
        std::vector<rhi::ComputeBufferRead> reads;
        if (boundary == Phase::CountBoundary || boundary == Phase::Done)
            reads.push_back({arena, uint64_t(config.workspace.y) * WordBytes, WordBytes});
        if (boundary == Phase::ActiveBoundary)
            reads.push_back({arena, uint64_t(config.activeState.x) * WordBytes, WordBytes});
        if (dispatches.empty() && uploads.empty() && allocationBlocked)
            return {nullptr, [self = shared_from_this()] { return self->Next(); }};
        if (dispatches.empty() && uploads.empty() && reads.empty() && packetFull && limit == 1u) {
            Restore(*checkpoint);
            return Next(2u); // One indivisible indirect producer/consumer pair.
        }
        if (dispatches.empty() && uploads.empty() && reads.empty())
            throw std::logic_error("Jolt GPU broadphase continuation produced an empty packet");
        // Pure cursor transitions need acceptance, not CPU fence
        // observation. Metadata decode and final workset handoff still wait.
        const auto resumeAfter = reads.empty() && boundary != Phase::Done ? JoltGpuControlWork::ResumeAfter::Acceptance
                                                                          : JoltGpuControlWork::ResumeAfter::Completion;
        auto packet =
            rhi::PrepareComputeBatch(implementation->host, std::move(uploads), std::move(dispatches), std::move(reads));
        JoltGpuControlWork result{packet,
                                  [self = shared_from_this(), packet, boundary]() -> JoltGpuControlWork {
                                      if (boundary == Phase::CountBoundary || boundary == Phase::Done)
                                          self->Decode(packet, boundary == Phase::CountBoundary);
                                      if (boundary == Phase::ActiveBoundary)
                                          self->DecodeActive(packet);
                                      if (boundary == Phase::Done && self->phase == Phase::Done) {
                                          // No optional scratch/args/uniform owner escapes
                                          // this completed execution. The workset retains only
                                          // the values needed by generic/topology/solver work.
                                          self->workset->parameters = self->workset->parameters->Freeze();
                                          self->workset->worlds = self->runtimeWorlds->Freeze();
                                          self->workset->patches = self->candidateArena;
                                          auto callback = std::move(self->completed);
                                          callback(self->workset);
                                          return {};
                                      }
                                      return self->Next();
                                  },
                                  resumeAfter};
        if (limit > 2u)
            result.prepareMinimum = [self = shared_from_this(), packet, checkpoint] {
                packet->Cancel();
                self->Restore(*checkpoint);
                return self->Next(1u);
            };
        return result;
    }

    std::shared_ptr<Impl> implementation;
    JoltGpuPrograms &stagePrograms;
    const JoltGpuStorage &storageOwner;
    const JoltGpuStepInputs &acceptedInputs;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> settingsPorts;
    std::shared_ptr<rhi::ComputeBuffer> arena, runtimeWorlds, arguments, candidateArena;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, MaxDispatchesPerPacket> parameterSlots, workSlots;
    std::shared_ptr<JoltGpuContactWorkset> workset;
    StepParams parameters;
    BroadphaseParams config{};
    std::function<void(std::shared_ptr<const JoltGpuContactWorkset>)> completed;
    std::vector<ScanLevel> levels;
    ScanCursor scan;
    // Appended only after a count receipt completes. Packet preparation and
    // its rollback never change these bounds for the subsequent fill passes.
    std::vector<SweepTail> countTails;
    Phase phase = Phase::InitialCapacity;
    std::array<uint32_t, 2> sap{}, suppression{}, queues{}, activeMeta{};
    uint32_t scratchWords = 0, zeroRecord = 0, item = 0;
    uint32_t sortSource = 0, sortTarget = 0, sortCount = 0, sortKey = 0, sortWidth = 1;
    ActiveStage activeStage = ActiveStage::Seed;
    Entry indirectEntry = EntryCount;
    uint32_t activeBound = 0, neighborsRemaining = 0, windowPasses = 0;
    size_t fillTail = 0;
    uint32_t indirectMeta = 0, indirectLevel = 0, indirectFirst = 0, indirectCount = 0;
    uint32_t capacity = 0, requiredCapacity = 0, packetLimit = MaxDispatchesPerPacket;
    Phase afterContacts = Phase::Suppression;
    bool initialUploads = false;
    bool allocationBlocked = false, packetFull = false, activeFilling = false;
};

JoltGpuBroadphase::JoltGpuBroadphase(rhi::ComputeHost &host, const ShaderTargetCode &target)
    : m_impl(std::make_shared<Impl>(host, target))
{
}

bool JoltGpuBroadphase::TryPrepareSmallPairPipeline()
{
    return m_impl->programs[BuildSmallPairs].kernel->TryPreparePipeline();
}

JoltGpuControlWork
JoltGpuBroadphase::Prepare(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs, JoltGpuPrograms &programs,
                           const StepParams &parameters,
                           std::function<void(std::shared_ptr<const JoltGpuContactWorkset>)> completed, bool swept)
{
    return std::make_shared<Impl::Execution>(m_impl, storage, inputs, programs, parameters, std::move(completed), swept)
        ->Next();
}

void JoltGpuBroadphase::ResetCapacityHint() noexcept
{
    m_impl->recentCapacity = {};
}

} // namespace infernux
