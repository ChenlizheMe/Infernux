#include "JoltGpuBroadphase.h"
#include "JoltGpuProgramSources.h"
#include "JoltGpuSleep.h"
#include <core/log/InxLog.h>

#include <function/scene/physics/gpu/BroadphaseTypes.h>
#include <function/scene/physics/gpu/ConstraintTypes.h>

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
    Arena,
    Arguments,
    ColliderPairs,
    ResidentStep,
    ResidentContacts,
    CandidateArena,
    PairCount,
    StepStatus,
    PortCount
};
enum Entry : uint8_t
{
    InitWords,
    SeedSweep,
    PrepareIndirect,
    PrepareSweepArguments,
    BuildSort,
    BuildSuppression,
    Merge,
    SortCandidateTiles,
    FindEnds,
    ProjectActivePairs,
    CollectSweep,
    ScanBlocks,
    ReduceBlocks,
    AddOffsets,
    AccumulateStatus,
    PublishCount,
    InitContacts,
    PublishCandidates,
    InitCandidates,
    BuildSmallPairs,
    EntryCount
};
constexpr std::array<std::string_view, EntryCount> EntryNames{"init_words",
                                                              "seed_sweep",
                                                              "prepare_indirect_arguments",
                                                              "prepare_sweep_arguments",
                                                              "build_sort_records",
                                                              "build_suppression",
                                                              "merge_items",
                                                              "sort_candidate_tiles",
                                                              "find_sap_ends",
                                                              "project_active_pairs",
                                                              "collect_sap_pairs",
                                                              "scan_blocks",
                                                              "reduce_blocks",
                                                              "add_scan_offsets",
                                                              "accumulate_status_summary",
                                                              "publish_count_summary",
                                                              "init_contacts",
                                                              "publish_candidates",
                                                              "init_candidates",
                                                              "build_small_pairs"};

constexpr uint32_t Bit(Port port)
{
    return 1u << port;
}
constexpr uint32_t A = Bit(Arena), P = Bit(Parameters), K = Bit(Work);
constexpr std::array<uint32_t, EntryCount> EntryPorts{
    A | P | K,
    A | P | Bit(PairCount),
    A | Bit(Arguments) | P | K,
    A | Bit(Arguments) | P | K | Bit(PairCount),
    A | P | K | Bit(Shapes) | Bit(Bodies) | Bit(AuthorWorlds) | Bit(Bounds),
    A | P | K | Bit(Joints) | Bit(Bodies) | Bit(AuthorWorlds),
    A | P | K | Bit(CandidateArena),
    A | P | K | Bit(CandidateArena),
    A | P | K | Bit(Bounds) | Bit(Shapes) | Bit(Bodies),
    A | P | K,
    A | P | K | Bit(Shapes) | Bit(Bodies) | Bit(Bounds) | Bit(ColliderPairs) | Bit(CandidateArena) | Bit(PairCount),
    A | P | K,
    A | P | K,
    A | P | K,
    A | P,
    A | P | K | Bit(PairCount),
    P | K | Bit(ResidentContacts),
    A | P | K | Bit(ResidentStep) | Bit(CandidateArena) | Bit(Arguments) | Bit(StepStatus),
    P | K | Bit(CandidateArena),
    A | P | Bit(Bodies) | Bit(Shapes) | Bit(AuthorWorlds) | Bit(Bounds) | Bit(Joints) | Bit(ColliderPairs) |
        Bit(CandidateArena)};

struct PortLayout
{
    std::string_view name;
    uint64_t stride;
    std::array<std::string_view, 11> fields;
    uint32_t integers;
    bool readOnly;
};
constexpr PortLayout Ports[] = {
    {"broadphase",
     sizeof(BroadphaseParams),
     {"counts", "workspace", "round", "candidates", "merge", "mergeKey", "scan", "scanParent", "summary", "selection",
      "selectedCounts"},
     0x7ffu,
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
    {"contact_patches", sizeof(ContactPatch), {"leaves"}, 1u, false},
    {"indirect_arguments", sizeof(QueryRecordWord), {"value"}, 1u, false},
    {"collider_pair_ignores", sizeof(ColliderPairKey), {"a", "b"}, 3u, true},
    {"resident_step", sizeof(QueryRecordWord), {"value"}, 1u, false},
    {"resident_contacts",
     sizeof(Contact),
     {"meta", "normalDepth", "localAnchorA", "localAnchorB", "impulse"},
     1u,
     false},
    {"resident_patches", sizeof(ContactPatch), {"leaves"}, 1u, false},
    {"pair_count", sizeof(uint32_t), {}, 1u, false},
    {"step_status", sizeof(uint32_t), {}, 1u, false}};

constexpr uint32_t MaxDispatchesPerPacket = JoltGpuBroadphase::Workspace::DispatchesPerPacket;
constexpr uint32_t CapacityNeeded = 128u;
constexpr uint32_t SweepPending = 256u;
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
        (!uniform && resource.binding.readOnlyStorage != expected.readOnly))
        throw std::invalid_argument("Jolt GPU broadphase target differs from its shared layout");
    if (port == PairCount || port == StepStatus) {
        if (value.kind != ShaderValueKind::Scalar || value.scalar != ShaderScalarKind::UInt32 ||
            value.byteSize != sizeof(uint32_t) || value.rows != 1 || value.columns != 1 || !value.fields.empty())
            throw std::invalid_argument("Jolt GPU broadphase requires scalar uint32 atomic storage");
        return;
    }
    if (value.kind != ShaderValueKind::Struct || value.fields.size() != expected.stride / WordBytes)
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
    return entry == SeedSweep || entry == AccumulateStatus || entry == PublishCount;
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
        ActiveScan,
        ProjectionRows,
        ProjectionScan,
        SeedSweep,
        Sweep,
        CountPublish,
        CountBoundary,
        FinalCapacity,
        FinalInitialize,
        FinalContacts,
        ResidentPublish,
        CandidateSort,
        Collide,
        Done
    };
    struct ScanLevel
    {
        uint32_t input, prefix, total, capacity;
    };
    struct ScanCursor
    {
        bool active = false, exclusive = false, descending = false;
        uint32_t originalInput = 0, level = 0, first = 0, root = 0;
        std::vector<uint32_t> counts;
    };
    struct PacketCursor
    {
        BroadphaseParams config;
        ScanCursor scan;
        uint64_t sweepCursor;
        Phase phase, afterContacts;
        uint32_t item, sortSource, sortTarget, sortCount, sortKey, sortWidth;
        bool initialUploads;
    };
    Execution(std::shared_ptr<Impl> owner, const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
              JoltGpuPrograms &programs, const StepParams &settings,
              const std::shared_ptr<rhi::ComputeBuffer> &stepStatus, Workspace &workspace,
              std::function<void(std::shared_ptr<const JoltGpuContactWorkset>)> callback, Mode mode,
              SweepConsumer sweep = {})
        : implementation(std::move(owner)), stagePrograms(programs), storageOwner(storage), acceptedInputs(inputs),
          workspace(workspace), parameters(settings), completed(std::move(callback)), consumeSweep(std::move(sweep))
    {
        const bool swept = mode == Mode::Swept;
        if (!implementation->host.SharesServicesWith(storage.GetHost()) || (swept ? !consumeSweep : !completed))
            throw std::invalid_argument("Jolt GPU broadphase requires its tick owner and completion callback");
        const uint32_t n = storage.GetShapeCount(), j = storage.GetJointCount(), w = storage.GetWorldCount();
        if (!storage.GetBodyCount() || !w)
            throw std::invalid_argument("An empty GPU model does not require broadphase work");
        settingsPorts[Bodies] = inputs.GetCaptured(Resource::BodyModels);
        settingsPorts[Shapes] = inputs.GetCaptured(Resource::ShapeModels);
        settingsPorts[AuthorWorlds] = inputs.GetCaptured(Resource::Worlds);
        // BuildBounds and broadphase use the same World leaf range. The tick
        // owns that range through its last broadphase consumer, while peer
        // ticks write only their own range. Keep the allocation alive without
        // freezing the entire packed scratch buffer on each World.
        settingsPorts[Bounds] = storage.Get(Resource::Bounds);
        settingsPorts[StepStatus] = stepStatus;
        // The exclusive arena writer keeps this topology fixed. Broadphase
        // reads only joint directory/author metadata, so representation-history
        // retirement may update runtime words without retaining a COW reader.
        settingsPorts[Joints] = storage.Get(Resource::Joints);
        if (!settingsPorts[Joints] || !implementation->host.SharesServicesWith(settingsPorts[Joints]->GetHost()))
            throw std::invalid_argument("GPU broadphase requires its exclusive writer joint stream");
        for (auto port : {Bodies, Shapes, AuthorWorlds})
            ValidateInput(settingsPorts[port]);
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
        const bool smallPairs = leaves && leaves <= BroadphaseSmallSetLimit && joints <= BroadphaseSmallJointLimit;
        const auto &potential = storage.GetDiscretePairLimits(config.selection.x);
        const bool projectActivePairs = !smallPairs && (swept || potential.activeLeaves < leaves);
        sweepUpperBound = leaves ? uint64_t(leaves) * (leaves - 1u) / 2u : 0u;
        if (projectActivePairs)
            sweepUpperBound = potential.projections;
        config.round.x = uint32_t(projectActivePairs);
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
        config.workspace.y = reserve(smallPairs ? 1u : 2u);
        if (!smallPairs) {
            zeroRecord = reserve(1);
            sap = {reserve(leaves), reserve(leaves)};
            suppression = {reserve(joints), reserve(joints)};
            if (projectActivePairs) {
                config.summary.y = reserve(leaves); // One row per sorted projection interval.
                config.summary.z = reserve(leaves); // Active rank -> sorted leaf rank.
            }
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
        const uint32_t previousPatches = swept ? 0u : storage.GetContactHistory(settings.selection.x).patchCount;
        // Use the preceding substep's accepted pair count to avoid allocating
        // quadratic contact storage for sparse single-group work.
        uint32_t pairLimit = smallPairs ? static_cast<uint32_t>(sweepUpperBound) : 0u;
        // A bounded contact stream can feed the complete patch solver table
        // without a host count. Body slots don't set that stream's growth rate.
        const bool boundedDiscrete = mode == Mode::Discrete && sweepUpperBound <= BroadphaseSweepJobs &&
                                     potential.candidates <= IslandPatchRoundVisits;
        const bool deviceCounts = swept || boundedDiscrete;
        config.round.z = uint32_t(deviceCounts);
        if (boundedDiscrete) {
            pairLimit = static_cast<uint32_t>(potential.candidates);
            workset->patchSlots = pairLimit;
            if (ConstraintDeviceScheduleFits(settings.selectedCounts.x, joints, pairLimit))
                collisionChunk = ConstraintLocalRefs;
        }
        auto previous = std::exchange(workspace, {});
        parameterSlots = std::move(previous.parameterSlots);
        workSlots = std::move(previous.workSlots);
        sweepLimit = std::min(sweepUpperBound, std::max<uint64_t>(previous.sweepJobs, BroadphaseSweepJobs));
        if (swept) {
            const uint32_t seed = std::min(leaves, BroadphaseItemLimit);
            ConfigureCapacity(static_cast<uint32_t>(std::max<uint64_t>(
                1u, std::min<uint64_t>(potential.candidates, std::max(seed, previous.nextCapacity)))));
        } else {
            ConfigureCapacity(boundedDiscrete ? std::max(1u, pairLimit)
                              : smallPairs    ? std::min(std::max(leaves, previousPatches), std::max(1u, pairLimit))
                                              : previous.nextCapacity);
        }
        if (previous.arena && previous.arena->GetByteSize() == uint64_t(scratchWords) * WordBytes)
            arena = std::move(previous.arena);
        arguments = std::move(previous.arguments);
        workset->parameters = std::move(previous.parameters);
        if (!smallPairs)
            pairCount = std::move(previous.pairCount);
        if (previous.patchCapacity == capacity) {
            candidateArena = std::move(previous.patches);
            workset->contacts = std::move(previous.contacts);
        }
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

    bool ArgumentCapacity()
    {
        const uint64_t bytes =
            uint64_t(std::max(MaxDispatchesPerPacket - 1u, CollisionChunks() * RecordsPerChunk())) * WordBytes;
        if (arguments && arguments->GetByteSize() < bytes)
            arguments.reset();
        return Endpoint(arguments, bytes, rhi::BufferUsageFlags::Storage | rhi::BufferUsageFlags::Indirect);
    }

    uint32_t CollisionChunkSize() const
    {
        return collisionChunk;
    }

    uint32_t CollisionChunks() const
    {
        const auto size = CollisionChunkSize();
        return capacity / size + uint32_t(capacity % size != 0u);
    }

    uint32_t RecordsPerChunk() const
    {
        return config.round.z && !config.round.y ? uint32_t(JoltGpuContactWorkset::Range::RangeCount) : 1u;
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
                (!config.round.w && !Endpoint(pairCount, 2u * sizeof(uint32_t))) || !ArgumentCapacity())
                return false;
            if (!Endpoint(candidateArena, (1ull + 2ull * capacity) * WordBytes) ||
                !Endpoint(workset->contacts, uint64_t(std::max(1u, capacity * PointsPerPair())) * sizeof(Contact))) {
                const uint32_t minimum = config.round.y ? 1u : 0u;
                if (capacity > minimum && !config.round.w && (!config.round.z || config.round.y)) {
                    // A historical K is not a mandatory lower bound. Drop all
                    // partial optimistic owners, then let the SAME execution
                    // run its cold CountBoundary with K=0. Nothing is replayed.
                    candidateArena.reset();
                    workset->patches.reset();
                    workset->contacts.reset();
                    ConfigureCapacity(minimum);
                }
                return false;
            }
            workset->patches = candidateArena;
            initialUploads = true;
            phase = Phase::Initialize;
        } else if (phase == Phase::FinalCapacity) {
            if (!Endpoint(candidateArena, (1ull + 2ull * capacity) * WordBytes) ||
                !Endpoint(workset->contacts, uint64_t(std::max(1u, capacity * PointsPerPair())) * sizeof(Contact)) ||
                !Endpoint(workset->parameters, sizeof(StepParams)) || !ArgumentCapacity()) {
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
        const bool sweepArguments = entry == PrepareSweepArguments;
        const uint32_t first = work.range.x, count = work.range.y;
        const bool publish = entry == PublishCandidates;
        const uint32_t limit = entry == PrepareIndirect || sweepArguments ? MaxDispatchesPerPacket - 1u
                               : entry == CollectSweep                    ? BroadphaseSweepJobs
                               : publish                                  ? CollisionChunks()
                               : block                                    ? BroadphaseBlockLimit
                                                                          : BroadphaseItemLimit;
        if (!count || count > limit)
            throw std::logic_error("Jolt GPU broadphase emitted an invalid bounded work unit");
        const auto slot = dispatches.size();
        if (slot >= packetLimit || !parameterSlots[slot] || !workSlots[slot])
            throw std::logic_error("Resident broadphase requires its admitted bounded uniform workspace");
        auto uniform = parameterSlots[slot];
        const auto &workUniform = workSlots[slot];
        auto parameterBytes = Bytes(value);
        const auto same = std::find_if(uploads.begin(), uploads.end(), [&](const auto &upload) {
            return upload.offset == 0 && upload.data == parameterBytes &&
                   std::find(parameterSlots.begin(), parameterSlots.end(), upload.buffer) != parameterSlots.end();
        });
        if (same != uploads.end())
            uniform = same->buffer;
        else
            uploads.push_back({uniform, 0, std::move(parameterBytes)});
        if (EntryPorts[entry] & K)
            uploads.push_back({workUniform, 0, Bytes(work)});
        if (publish)
            JoltGpuSleep::InitializeStepStatus(settingsPorts[StepStatus], uploads);
        const auto &program = implementation->programs[entry];
        rhi::ComputeDispatchDesc dispatch;
        dispatch.kernel = program.kernel;
        dispatch.groupCountX = block ? count : SummaryEntry(entry) || publish || sweepArguments ? 1u : Blocks(count);
        dispatch.indirectArguments = std::move(indirect);
        if (dispatch.groupCountX > implementation->host.device.GetCapabilities().limits.maxComputeWorkgroupCount[0])
            throw std::length_error("Jolt GPU broadphase bounded work exceeds dispatch limits");
        rhi::ComputeDispatchProof proof;
        for (const auto port : program.ports) {
            const auto binding = static_cast<uint32_t>(dispatch.buffers.size());
            const auto &buffer = port == Parameters         ? uniform
                                 : port == Work             ? workUniform
                                 : port == Arena            ? arena
                                 : port == Arguments        ? arguments
                                 : port == ResidentContacts ? workset->contacts
                                 : port == ResidentStep     ? workset->parameters
                                 : port == CandidateArena   ? candidateArena
                                 : port == PairCount        ? pairCount
                                                            : settingsPorts[port];
            if (!buffer)
                throw std::logic_error("Jolt GPU broadphase work has an absent resource");
            dispatch.buffers.push_back(buffer);
            const bool initialize = (port == Arena && (entry == InitWords || entry == BuildSmallPairs)) ||
                                    (port == ResidentContacts && entry == InitContacts) ||
                                    (port == CandidateArena && entry == InitCandidates);
            const bool argumentWrite = port == Arguments;
            const bool resetCount = port == PairCount && entry == SeedSweep;
            dispatch.bufferAccesses.push_back(initialize || argumentWrite || resetCount
                                                  ? rhi::ComputeBufferAccess::Write
                                              : port == Arena || port == CandidateArena || port == ResidentStep ||
                                                      port == StepStatus || (port == PairCount && entry == CollectSweep)
                                                  ? rhi::ComputeBufferAccess::ReadWrite
                                                  : rhi::ComputeBufferAccess::Read);
            if (initialize)
                proof.writes.push_back({binding,
                                        uint64_t(first) * (port == ResidentContacts ? sizeof(Contact) : WordBytes),
                                        uint64_t(count) * (port == ResidentContacts ? sizeof(Contact) : WordBytes)});
            else if (argumentWrite) {
                const auto records = count * (entry == PublishCandidates ? work.range.z : 1u);
                proof.writes.push_back({binding, 0, uint64_t(records) * WordBytes});
            } else if (resetCount)
                proof.writes.push_back({binding, 0, 2u * sizeof(uint32_t)});
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
        constexpr auto limit = BroadphaseItemLimit;
        const bool matching = indirectEntry == entry && first >= indirectFirst &&
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
            const StageWork producer{{first, records, 0u, 0u}};
            if (!EmitWork(PrepareIndirect, producer, value, uploads, dispatches))
                return false;
            indirectEntry = entry;
            indirectFirst = first;
            indirectCount = records;
        }
        const uint64_t record = (first - indirectFirst) / limit;
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

    void BeginScan(uint32_t input, uint32_t count, bool exclusive)
    {
        scan = {};
        scan.active = count != 0;
        scan.exclusive = exclusive;
        scan.originalInput = input;
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
        value.scan = {level ? levels[level].input : scan.originalInput, addresses.prefix, addresses.total, size};
        if (!scan.descending) {
            const uint32_t count = std::min(BroadphaseBlockLimit, Blocks(size) - scan.first);
            const auto entry = scan.exclusive ? ScanBlocks : ReduceBlocks;
            if (!Emit(entry, scan.first, count, value, uploads, dispatches))
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
            if (!Emit(AddOffsets, scan.first, count, value, uploads, dispatches))
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

    static StageWork SweepWork(uint64_t first, uint32_t count)
    {
        return {{static_cast<uint32_t>(first), count, static_cast<uint32_t>(first >> 32u), 0u}};
    }

    bool Sweep(std::vector<rhi::ComputeBufferUpdate> &uploads, std::vector<rhi::ComputeDispatchDesc> &dispatches)
    {
        if (sweepCursor >= sweepLimit)
            return false;
        if (indirectEntry != CollectSweep) {
            const auto available = packetLimit - static_cast<uint32_t>(dispatches.size());
            if (available < 2u) {
                packetFull = true;
                return true;
            }
            const auto jobs = sweepLimit - sweepCursor;
            const auto windows = (jobs + BroadphaseSweepJobs - 1u) / BroadphaseSweepJobs;
            const auto records = static_cast<uint32_t>(std::min<uint64_t>(available - 1u, windows));
            auto argumentsWork = SweepWork(sweepCursor, records);
            argumentsWork.range.w = static_cast<uint32_t>(
                std::min<uint64_t>(BroadphaseSweepJobs, jobs - uint64_t(records - 1u) * BroadphaseSweepJobs));
            if (!EmitWork(PrepareSweepArguments, argumentsWork, config, uploads, dispatches))
                return true;
            indirectEntry = CollectSweep;
            indirectFirst = sweepCursor;
        }
        const auto record = (sweepCursor - indirectFirst) / BroadphaseSweepJobs;
        const auto count = static_cast<uint32_t>(std::min<uint64_t>(BroadphaseSweepJobs, sweepLimit - sweepCursor));
        if (!EmitWork(CollectSweep, SweepWork(sweepCursor, count), config, uploads, dispatches,
                      rhi::ComputeIndirectArguments{arguments, record * WordBytes}))
            return true;
        sweepCursor += count;
        return true;
    }

    bool NextDispatch(std::vector<rhi::ComputeBufferUpdate> &uploads, std::vector<rhi::ComputeDispatchDesc> &dispatches)
    {
        // Only bounded phase transitions run here; item work is emitted lazily.
        while (true) {
            switch (phase) {
            case Phase::Initialize:
                // Scratch is reset each step. Capacity suffix needs a real
                // direct initializer only when this allocation is new.
                if (!config.round.w && Range(InitWords, scratchWords, uploads, dispatches))
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
                if (!Emit(BuildSmallPairs, 0, 1u, config, uploads, dispatches))
                    return true;
                config.merge.x = config.candidates.x;
                phase = Phase::ResidentPublish;
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
                config.scan.x = levels[0].input;
                phase = Phase::Ends;
                break;
            case Phase::Ends:
                if (Range(FindEnds, config.selectedCounts.y, uploads, dispatches))
                    return true;
                BeginScan(config.scan.x, config.selectedCounts.y, true);
                phase = config.round.x ? Phase::ActiveScan : Phase::ProjectionScan;
                break;
            case Phase::ActiveScan:
                if (Scan(uploads, dispatches))
                    return true;
                config.summary.x = scan.root;
                config.scan.y = levels[0].prefix;
                phase = Phase::ProjectionRows;
                break;
            case Phase::ProjectionRows:
                if (Range(ProjectActivePairs, config.selectedCounts.y, uploads, dispatches))
                    return true;
                BeginScan(config.scan.x, config.selectedCounts.y, true);
                phase = Phase::ProjectionScan;
                break;
            case Phase::ProjectionScan:
                if (Scan(uploads, dispatches))
                    return true;
                config.summary.x = scan.root;
                config.scan.y = levels[0].prefix;
                phase = Phase::SeedSweep;
                break;
            case Phase::SeedSweep:
                if (!Emit(SeedSweep, 0, 1u, config, uploads, dispatches))
                    return true;
                sweepCursor = 0u;
                phase = Phase::Sweep;
                return true;
            case Phase::Sweep:
                if (Sweep(uploads, dispatches))
                    return true;
                phase = Phase::CountPublish;
                break;
            case Phase::CountPublish:
                if (!EmitWork(PublishCount, SweepWork(sweepCursor, 1u), config, uploads, dispatches))
                    return true;
                if (capacity) {
                    BeginSort(config.candidates.x, config.candidates.y, capacity, 2);
                    phase = Phase::CandidateSort;
                } else
                    phase = Phase::CountBoundary;
                return true;
            case Phase::FinalInitialize:
                if (Range(InitCandidates, static_cast<uint32_t>(1ull + 2ull * capacity), uploads, dispatches))
                    return true;
                if (config.round.w)
                    afterContacts = Phase::SmallPairs;
                else if (requiredCapacity) {
                    // Refill from the retained projection prefix when storage grows.
                    afterContacts = Phase::SeedSweep;
                } else {
                    BeginSort(config.candidates.x, config.candidates.y, capacity, 2);
                    afterContacts = Phase::CandidateSort;
                }
                phase = Phase::FinalContacts;
                break;
            case Phase::FinalContacts:
                if (Range(InitContacts, std::max(1u, capacity * PointsPerPair()), uploads, dispatches))
                    return true;
                phase = afterContacts;
                break;
            case Phase::CandidateSort:
                if (Sort(uploads, dispatches))
                    return true;
                config.merge.x = sortSource;
                phase = Phase::ResidentPublish;
                break;
            case Phase::ResidentPublish: {
                const StageWork publication{{CollisionChunkSize(), CollisionChunks(), RecordsPerChunk(), 0u}};
                if (!EmitWork(PublishCandidates, publication, config, uploads, dispatches))
                    return true;
                phase = config.round.y ? Phase::Done : Phase::Collide;
                return true;
            }
            case Phase::Collide: {
                const uint32_t count = std::min(CollisionChunkSize(), capacity - item);
                const uint64_t record = uint64_t(item / CollisionChunkSize()) * RecordsPerChunk();
                stagePrograms.AppendResidentCollide(storageOwner, acceptedInputs, workset, capacity,
                                                    StageWork{{item, count, 0u, 0u}}, workSlots[dispatches.size()],
                                                    {arguments, record * WordBytes}, uploads, dispatches);
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
        std::array<UInt4, 2> records{};
        read->ReadInto(records.data(), config.round.w ? WordBytes : 2u * WordBytes);
        const auto &result = records[0];
        if (result.z & ~(CapacityNeeded | SweepPending))
            throw std::runtime_error("Jolt GPU broadphase numerical validation failed (status " +
                                     std::to_string(result.z) + ")");
        if (result.y || result.x > UINT32_MAX / MaxContactPointsPerPair || result.w > result.x ||
            (result.x == 0 && result.w != 0))
            throw std::runtime_error("Jolt GPU broadphase returned invalid candidate metadata");
        const uint64_t projected = (uint64_t(records[1].y) << 32u) | records[1].x;
        if (projected > sweepUpperBound || bool(result.z & SweepPending) != (projected > sweepCursor))
            throw std::runtime_error("Jolt GPU broadphase returned invalid projection metadata");
        sweepLimit = projected;
        if (result.z & SweepPending) {
            // Resume the existing projection and append counter at the first
            // untested job. A changed command budget never repeats a window.
            phase = Phase::Sweep;
            return;
        }
        if (count || (result.z & CapacityNeeded)) {
            if (!count && result.x <= capacity)
                throw std::runtime_error("Resident broadphase reported a false capacity gap");
            workset->patchSlots = result.x;
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
            workset->patchSlots = result.x;
        }
    }

    void RetainWorkspace(uint32_t nextCapacity)
    {
        workspace = {capacity,       nextCapacity,      sweepLimit, arena,          arguments, workset->parameters,
                     candidateArena, workset->contacts, pairCount,  parameterSlots, workSlots};
    }

    JoltGpuControlWork ResumeSweep()
    {
        // The consumer received a deferred candidate header and did not
        // change motion. Read the producer's exact count/projection record
        // only now, retaining the accepted sweep cursor and projection scan.
        workspace = {};
        auto packet = rhi::PrepareComputeBatch(
            implementation->host, {}, {},
            {{arena, uint64_t(config.workspace.y) * WordBytes, config.round.w ? WordBytes : 2u * WordBytes}});
        return {packet, [self = shared_from_this(), packet] {
                    self->Decode(packet, false);
                    if (INXLOG_GET_LEVEL() == LOG_DEBUG) {
                        try {
                            INXLOG_DIAGNOSTIC("GPU swept candidate continuation: world_row=", self->config.selection.x,
                                              " reason=", self->phase == Phase::Sweep ? "projection" : "capacity",
                                              " accepted_jobs=", self->sweepCursor,
                                              " projected_jobs=", self->sweepLimit,
                                              " required_pairs=", self->requiredCapacity, " capacity=", self->capacity);
                        } catch (...) {
                            // Diagnostics cannot invalidate the accepted cursor.
                        }
                    }
                    return self->Next();
                }};
    }

    JoltGpuControlWork PublishSweep()
    {
        RetainWorkspace(capacity);
        auto view = std::make_shared<JoltGpuContactWorkset>(*workset);
        view->patchSlots = capacity;
        view->parameters = view->parameters->Freeze();
        return consumeSweep(std::move(view), [self = shared_from_this()] { return self->ResumeSweep(); });
    }

    void Restore(const PacketCursor &s)
    {
        config = s.config;
        scan = s.scan;
        phase = s.phase;
        afterContacts = s.afterContacts;
        item = s.item;
        sortSource = s.sortSource;
        sortTarget = s.sortTarget;
        sortCount = s.sortCount;
        sortKey = s.sortKey;
        sortWidth = s.sortWidth;
        sweepCursor = s.sweepCursor;
        initialUploads = s.initialUploads;
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
        PacketCursor checkpoint{config,     scan,       sweepCursor, phase,   afterContacts, item,
                                sortSource, sortTarget, sortCount,   sortKey, sortWidth,     initialUploads};
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
            if (phase == Phase::CountBoundary || phase == Phase::Done)
                break;
        }
        const auto boundary = phase;
        std::vector<rhi::ComputeBufferRead> reads;
        const bool readCount = boundary == Phase::CountBoundary || (boundary == Phase::Done && !config.round.z);
        if (readCount)
            reads.push_back(
                {arena, uint64_t(config.workspace.y) * WordBytes, config.round.w ? WordBytes : 2u * WordBytes});
        if (dispatches.empty() && uploads.empty() && allocationBlocked)
            return {nullptr, [self = shared_from_this()] { return self->Next(); }};
        if (dispatches.empty() && uploads.empty() && reads.empty() && packetFull && limit == 1u) {
            Restore(checkpoint);
            return Next(2u); // One indivisible indirect producer/consumer pair.
        }
        if (dispatches.empty() && uploads.empty() && reads.empty())
            throw std::logic_error("Jolt GPU broadphase continuation produced an empty packet");
        const auto resumeAfter =
            reads.empty() ? JoltGpuControlWork::ResumeAfter::Acceptance : JoltGpuControlWork::ResumeAfter::Completion;
        auto packet =
            rhi::PrepareComputeBatch(implementation->host, std::move(uploads), std::move(dispatches), std::move(reads));
        JoltGpuControlWork result{packet,
                                  [self = shared_from_this(), packet, boundary, readCount]() -> JoltGpuControlWork {
                                      if (readCount)
                                          self->Decode(packet, boundary == Phase::CountBoundary);
                                      if (boundary == Phase::Done && self->phase == Phase::Done) {
                                          if (self->config.round.y)
                                              return self->PublishSweep();
                                          const auto capacity = self->capacity;
                                          const auto nextCapacity = self->workset->patchSlots < capacity / 4u
                                                                        ? std::max(1u, capacity / 2u)
                                                                        : capacity;
                                          self->RetainWorkspace(nextCapacity);
                                          self->workset->parameters = self->workset->parameters->Freeze();
                                          self->workset->patches = self->candidateArena;
                                          if (self->config.round.z) {
                                              self->workset->countedArguments = self->arguments;
                                              self->workset->patchesPerDispatch = self->CollisionChunkSize();
                                          }
                                          auto callback = std::move(self->completed);
                                          callback(self->workset);
                                          return {};
                                      }
                                      return self->Next();
                                  },
                                  resumeAfter};
        if (limit > 2u)
            result.prepareMinimum = [self = shared_from_this(), packet, checkpoint = std::move(checkpoint)] {
                packet->Cancel();
                self->Restore(checkpoint);
                return self->Next(1u);
            };
        return result;
    }

    std::shared_ptr<Impl> implementation;
    JoltGpuPrograms &stagePrograms;
    const JoltGpuStorage &storageOwner;
    const JoltGpuStepInputs &acceptedInputs;
    Workspace &workspace;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> settingsPorts;
    std::shared_ptr<rhi::ComputeBuffer> arena, arguments, candidateArena, pairCount;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, MaxDispatchesPerPacket> parameterSlots, workSlots;
    std::shared_ptr<JoltGpuContactWorkset> workset;
    StepParams parameters;
    BroadphaseParams config{};
    std::function<void(std::shared_ptr<const JoltGpuContactWorkset>)> completed;
    SweepConsumer consumeSweep;
    std::vector<ScanLevel> levels;
    ScanCursor scan;
    Phase phase = Phase::InitialCapacity;
    std::array<uint32_t, 2> sap{}, suppression{};
    uint32_t scratchWords = 0, zeroRecord = 0, item = 0;
    uint32_t sortSource = 0, sortTarget = 0, sortCount = 0, sortKey = 0, sortWidth = 1;
    Entry indirectEntry = EntryCount;
    uint64_t indirectFirst = 0, sweepUpperBound = 0, sweepLimit = 0, sweepCursor = 0;
    uint32_t indirectCount = 0;
    uint32_t capacity = 0, requiredCapacity = 0, packetLimit = MaxDispatchesPerPacket;
    uint32_t collisionChunk = BroadphaseItemLimit;
    Phase afterContacts = Phase::Suppression;
    bool initialUploads = false;
    bool allocationBlocked = false, packetFull = false;
};

JoltGpuBroadphase::JoltGpuBroadphase(rhi::ComputeHost &host, const ShaderTargetCode &target)
    : m_impl(std::make_shared<Impl>(host, target))
{
}

bool JoltGpuBroadphase::TryPrepareStartupPrograms()
{
    bool ready = true;
    for (const auto entry : {InitWords, InitCandidates, InitContacts, BuildSmallPairs, PublishCandidates})
        ready &= m_impl->programs[entry].kernel->TryPreparePipeline();
    return ready;
}

JoltGpuControlWork
JoltGpuBroadphase::Prepare(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs, JoltGpuPrograms &programs,
                           const StepParams &parameters, const std::shared_ptr<rhi::ComputeBuffer> &stepStatus,
                           Workspace &workspace,
                           std::function<void(std::shared_ptr<const JoltGpuContactWorkset>)> completed, Mode mode)
{
    return std::make_shared<Impl::Execution>(m_impl, storage, inputs, programs, parameters, stepStatus, workspace,
                                             std::move(completed), mode)
        ->Next();
}

JoltGpuControlWork JoltGpuBroadphase::PrepareSweep(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                                                   JoltGpuPrograms &programs, const StepParams &parameters,
                                                   const std::shared_ptr<rhi::ComputeBuffer> &stepStatus,
                                                   Workspace &workspace, SweepConsumer consumer)
{
    return std::make_shared<Impl::Execution>(m_impl, storage, inputs, programs, parameters, stepStatus, workspace,
                                             nullptr, Mode::Swept, std::move(consumer))
        ->Next();
}

} // namespace infernux
