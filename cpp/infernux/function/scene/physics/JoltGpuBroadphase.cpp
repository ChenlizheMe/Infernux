#include "JoltGpuBroadphase.h"

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
    PublishWorld,
    ResetFill,
    CheckFill,
    PublishFill,
    SetCandidateBank,
    EntryCount
};
constexpr std::array<std::string_view, EntryCount> EntryNames{"init_words",
                                                              "seed_active",
                                                              "prepare_indirect_arguments",
                                                              "build_sort_records",
                                                              "build_suppression",
                                                              "merge_items",
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
                                                              "publish_world_summary",
                                                              "reset_fill_ranks",
                                                              "check_fill",
                                                              "publish_fill_summary",
                                                              "set_candidate_bank"};

constexpr uint32_t Bit(Port port)
{
    return 1u << port;
}
constexpr uint32_t A = Bit(Arena), P = Bit(Parameters), K = Bit(Work);
constexpr std::array<uint32_t, EntryCount> EntryPorts{A | P | K,
                                                      A | P,
                                                      A | Bit(Arguments) | P | K,
                                                      A | P | K | Bit(Shapes) | Bit(Bodies) | Bit(AuthorWorlds) |
                                                          Bit(Bounds),
                                                      A | P | K | Bit(Joints) | Bit(Bodies) | Bit(AuthorWorlds),
                                                      A | P | K,
                                                      A | P | K | Bit(Bounds),
                                                      A | P | K | Bit(Shapes) | Bit(Bodies) | Bit(Bounds),
                                                      A | P | K | Bit(Shapes) | Bit(Bodies) | Bit(Bounds),
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
                                                      A | P,
                                                      A | P | K,
                                                      A | P | K,
                                                      A | P,
                                                      A | P};

struct PortLayout
{
    std::string_view name;
    uint64_t stride;
    std::array<std::string_view, 12> fields;
    uint32_t integers;
    bool readOnly;
};
constexpr PortLayout Ports[] = {
    {"broadphase",
     sizeof(BroadphaseParams),
     {"counts", "workspace", "rankRows", "activeQueues", "round", "candidates", "merge", "mergeKey", "scan",
      "scanParent", "activeState", "summary"},
     0xfffu,
     true},
    {"stage_work", sizeof(StageWork), {"range"}, 1u, true},
    {"body_models", sizeof(BodyModel), {"meta", "mass", "invInertiaDiagonal", "inertiaRotation", "damping"}, 1u, true},
    {"shape_models",
     sizeof(ShapeModel),
     {"meta", "geometry", "localPositionCOM", "localRotation", "material", "surface", "source"},
     (1u << 0) | (1u << 5) | (1u << 6),
     true},
    {"author_worlds", sizeof(WorldRange), {"bodies", "leaves"}, 3u, true},
    {"bounds", sizeof(Aabb), {"minimum", "maximum"}, 0u, true},
    {"joint_stream", sizeof(QueryRecordWord), {"value"}, 1u, true},
    {"runtime_worlds", sizeof(WorldRange), {"bodies", "leaves"}, 3u, false},
    {"contact_patches", sizeof(ContactPatch), {"leaves"}, 1u, false},
    {"indirect_arguments", sizeof(QueryRecordWord), {"value"}, 1u, false}};

constexpr uint32_t MaxDispatchesPerPacket = 16;
constexpr uint32_t ActiveRoundsPerWindow = 4;
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
           entry == PublishWorld || entry == PublishFill || entry == SetCandidateBank;
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
            if (target.target == ShaderCodeTarget::SpirV) {
                if (code->code.empty() || code->code.size() % sizeof(uint32_t))
                    throw std::invalid_argument("Jolt GPU broadphase target contains invalid SPIR-V");
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
    rhi::ComputeHost host;
    std::array<Program, EntryCount> programs;
};

struct JoltGpuBroadphase::Impl::Execution : std::enable_shared_from_this<Execution>
{
    enum class Phase
    {
        InitialCapacity,
        Initialize,
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
        WorldReduce,
        WorldPublish,
        CountBoundary,
        FinalCapacity,
        Copy,
        FinalInitialize,
        Reset,
        Fill,
        FillCheck,
        FillReduce,
        FillPublish,
        FillBoundary,
        ActiveBoundary,
        CandidateSort,
        CandidatePublish,
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

    Execution(std::shared_ptr<Impl> owner, const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
              const StepParams &settings, std::function<void(std::shared_ptr<const JoltGpuContactWorkset>)> callback)
        : implementation(std::move(owner)), parameters(settings), completed(std::move(callback))
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
        settingsPorts[Joints] = storage.Get(Resource::Joints)->Freeze();
        for (auto port : {Bodies, Shapes, AuthorWorlds, Joints})
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
        requireBytes(Joints, (1ull + 2ull * JointRecordWords * j) * sizeof(QueryRecordWord));
        if (n)
            requireBytes(Bounds, uint64_t(n) * sizeof(Aabb));
        config.counts = {n, j, w, storage.GetBodyCount()};
        config.round = {BroadphaseNeighborLimit, 0, 0, 0};
        rounds = n > 1 ? (n - 2u) / BroadphaseNeighborLimit + 1u : 1u;
        uint64_t words = 0;
        const auto reserve = [&](uint64_t size) {
            if (size > UINT32_MAX || words > UINT32_MAX - size)
                throw std::overflow_error("Jolt GPU broadphase workspace exceeds device word indexing");
            const auto result = static_cast<uint32_t>(words);
            words += size;
            return result;
        };
        config.workspace.y = reserve(2);
        zeroRecord = reserve(1);
        sap[0] = reserve(n);
        sap[1] = reserve(n);
        suppression[0] = reserve(j);
        suppression[1] = reserve(j);
        config.rankRows = {reserve(n), reserve(n), reserve(j), 0};
        queues = {reserve(n), reserve(n)};
        activeMeta = {reserve(1), reserve(1)};
        config.activeQueues = {queues[0], queues[1], n, 0};
        config.activeState = {activeMeta[0], activeMeta[1], 0, 0};
        uint32_t count = std::max({n, j, w, 1u});
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
        scratchWords = static_cast<uint32_t>(words);
        config.workspace.x = scratchWords;
        config.workspace.z = sap[0];
        config.workspace.w = suppression[0];
        config.scan.x = levels[0].input;
        config.summary.y = levels[0].input;
        config.summary.z = levels[0].input;
        parameters.counts.x = storage.GetBodyCount();
        parameters.counts.y = w;
        parameters.counts.z = n;
        parameters.solve.z = parameters.solve.w = 0;
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
            if (!Endpoint(arena, uint64_t(scratchWords) * WordBytes) ||
                !Endpoint(runtimeWorlds, uint64_t(config.counts.z) * sizeof(WorldRange)) ||
                !Endpoint(arguments, (MaxDispatchesPerPacket - 1u) * WordBytes,
                          rhi::BufferUsageFlags::Storage | rhi::BufferUsageFlags::Indirect))
                return false;
            phase = Phase::Initialize;
        } else if (phase == Phase::FinalCapacity) {
            if (!Endpoint(finalArena, uint64_t(config.workspace.x) * WordBytes) ||
                !Endpoint(workset->contacts,
                          uint64_t(std::max(1u, workset->patchCount * MaxContactPointsPerPair)) * sizeof(Contact)) ||
                !Endpoint(workset->parameters, sizeof(StepParams)))
                return false;
            oldArena = arena->Freeze();
            arena = std::move(finalArena);
            phase = Phase::Copy;
            copyFirst = 0;
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
        // Keep acquired uniforms across a temporary managed-capacity deferral.
        if (!Endpoint(pendingUniform, sizeof(BroadphaseParams)) ||
            ((EntryPorts[entry] & K) && !Endpoint(pendingWorkUniform, sizeof(StageWork))))
            return false;
        const auto uniform = pendingUniform, workUniform = pendingWorkUniform;
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
            const auto &buffer = port == Parameters      ? uniform
                                 : port == Work          ? workUniform
                                 : port == Arena         ? arena
                                 : port == RuntimeWorlds ? runtimeWorlds
                                 : port == Arguments     ? arguments
                                                         : settingsPorts[port];
            if (!buffer)
                throw std::logic_error("Jolt GPU broadphase work has an absent resource");
            dispatch.buffers.push_back(buffer);
            const bool initialize = port == Arena && entry == InitWords;
            const bool worldWrite = port == RuntimeWorlds;
            const bool argumentWrite = port == Arguments;
            dispatch.bufferAccesses.push_back(initialize || worldWrite || argumentWrite
                                                  ? rhi::ComputeBufferAccess::Write
                                              : port == Arena ? rhi::ComputeBufferAccess::ReadWrite
                                                              : rhi::ComputeBufferAccess::Read);
            if (initialize)
                proof.writes.push_back({binding, uint64_t(first) * WordBytes, uint64_t(count) * WordBytes});
            else if (worldWrite)
                proof.writes.push_back(
                    {binding, uint64_t(first) * sizeof(WorldRange), uint64_t(count) * sizeof(WorldRange)});
            else if (argumentWrite)
                proof.writes.push_back({binding, 0, uint64_t(count) * WordBytes});
            else
                proof.reads.push_back({binding, 0, buffer->GetByteSize()});
        }
        dispatch.proof = std::move(proof);
        dispatches.push_back(std::move(dispatch));
        pendingUniform.reset();
        pendingWorkUniform.reset();
        return true;
    }

    bool Emit(Entry entry, uint32_t first, uint32_t count, const BroadphaseParams &value,
              std::vector<rhi::ComputeBufferUpdate> &uploads, std::vector<rhi::ComputeDispatchDesc> &dispatches)
    {
        StageWork work{{first, count, 0, 0}};
        if (entry == BuildWorldRanges)
            work.range = {first, count, first, count};
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
            const uint32_t available = MaxDispatchesPerPacket - static_cast<uint32_t>(dispatches.size());
            if (available < 2u) {
                packetFull = true;
                return false;
            }
            const uint32_t remaining = (size - first) / limit + static_cast<uint32_t>((size - first) % limit != 0);
            const uint32_t records = std::min(available - 1u, remaining);
            if (!records || uint64_t(first) + uint64_t(records - 1u) * limit >= size)
                throw std::logic_error("Jolt GPU indirect chunk plan exceeds its bounded domain");
            const StageWork producer{{first, records, 0, block ? 1u : 0u}};
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
        sortWidth = 1;
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
        if (!Emit(Merge, item, count, value, uploads, dispatches))
            return true;
        item += count;
        if (item == sortCount) {
            item = 0;
            std::swap(sortSource, sortTarget);
            // A final width of 2^31 completes every uint32-sized domain.
            sortWidth = sortWidth > UINT32_MAX / 2u ? sortCount : sortWidth * 2u;
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
        activeStage = ActiveStage::Seed;
        activeBound = config.counts.x;
        round = windowRounds = 0;
        config.activeQueues = {queues[0], queues[1], activeBound, 0};
        config.activeState = {activeMeta[0], activeMeta[1], 0, 0};
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
            case ActiveStage::Round:
                config.scan.x = levels[0].input;
                if (ActiveRange(activeFilling ? FillRound : CountRound, uploads, dispatches))
                    return true;
                ++round;
                ++windowRounds;
                // A small window amortizes global scan/compaction. Each pass
                // retains the same input queue and bounded neighbor quantum;
                // a rank whose cursor reached its end performs no more tests.
                if (windowRounds < ActiveRoundsPerWindow && round < rounds)
                    break;
                BeginScan(levels[0].input, activeBound, true, true);
                activeStage = ActiveStage::Scan;
                break;
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
            case ActiveStage::Publish: {
                auto value = config;
                value.round.y = windowRounds;
                if (!Emit(PublishActive, 0, 1, value, uploads, dispatches))
                    return true;
                std::swap(config.activeQueues.x, config.activeQueues.y);
                std::swap(config.activeState.x, config.activeState.y);
                activeStage = ActiveStage::Round;
                if (round == rounds) {
                    // The full rank validation below already checks every
                    // cursor/status and publishes the count/fill receipt. At
                    // the algorithmic final round, share that boundary instead
                    // of adding an earlier host map of the active metadata.
                    config.scan.x = levels[0].input;
                    phase = activeFilling ? Phase::FillCheck : Phase::Counts;
                } else if (windowRounds == ActiveRoundsPerWindow)
                    phase = Phase::ActiveBoundary;
                return true;
            }
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
        if (value.x > activeBound || value.y > rounds || value.w || (value.x && (round == rounds || value.y != round)))
            throw std::runtime_error("Jolt GPU active queue failed its monotone bounded-round contract");
        item = windowRounds = 0;
        if (value.x) {
            activeBound = value.x;
            config.activeQueues.z = activeBound;
            phase = activeFilling ? Phase::Fill : Phase::Count;
        } else {
            config.scan.x = levels[0].input;
            phase = activeFilling ? Phase::FillCheck : Phase::Counts;
        }
    }

    bool NextDispatch(std::vector<rhi::ComputeBufferUpdate> &uploads, std::vector<rhi::ComputeDispatchDesc> &dispatches)
    {
        // Only bounded phase transitions run here; item work is emitted lazily.
        while (true) {
            switch (phase) {
            case Phase::Initialize:
                if (Range(InitWords, scratchWords, uploads, dispatches))
                    return true;
                phase = Phase::Suppression;
                break;
            case Phase::Suppression:
                config.merge.x = suppression[0];
                if (Range(BuildSuppression, config.counts.y, uploads, dispatches))
                    return true;
                BeginScan(config.rankRows.z, config.counts.y, false);
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
                BeginSort(suppression[0], suppression[1], config.counts.y, 1);
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
                if (Range(BuildSort, config.counts.x, uploads, dispatches))
                    return true;
                BeginSort(sap[0], sap[1], config.counts.x, 0);
                phase = Phase::SapSort;
                break;
            case Phase::SapSort:
                if (Sort(uploads, dispatches))
                    return true;
                config.workspace.z = sortSource;
                phase = Phase::Ends;
                break;
            case Phase::Ends:
                if (Range(FindEnds, config.counts.x, uploads, dispatches))
                    return true;
                BeginActive(false);
                phase = Phase::Count;
                break;
            case Phase::Count:
            case Phase::Fill:
                return ActiveRounds(uploads, dispatches);
            case Phase::Counts:
                if (Range(MaterializeCounts, config.counts.x, uploads, dispatches))
                    return true;
                BeginScan(levels[0].input, config.counts.x, true);
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
                if (Range(StoreOffsets, config.counts.x, uploads, dispatches))
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
                BeginScan(levels[0].input, config.counts.z, false);
                phase = Phase::WorldReduce;
                break;
            case Phase::WorldReduce:
                if (Scan(uploads, dispatches))
                    return true;
                config.summary.x = scan.root;
                phase = Phase::WorldPublish;
                break;
            case Phase::WorldPublish:
                if (!Emit(PublishWorld, 0, 1, config, uploads, dispatches))
                    return true;
                phase = Phase::CountBoundary;
                return true;
            case Phase::FinalInitialize:
                if (Range(InitWords, config.workspace.x, uploads, dispatches))
                    return true;
                BeginActive(true);
                phase = Phase::Reset;
                break;
            case Phase::Reset:
                if (Range(ResetFill, config.counts.x, uploads, dispatches))
                    return true;
                phase = Phase::Fill;
                break;
            case Phase::FillCheck:
                if (Range(CheckFill, config.counts.x, uploads, dispatches))
                    return true;
                BeginScan(config.summary.z, config.counts.x, false);
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
                phase = Phase::FillBoundary;
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
                phase = Phase::Done;
                return true;
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
        if (result.z)
            throw std::runtime_error("Jolt GPU broadphase failed numerical validation (status " +
                                     std::to_string(result.z) + ", required candidates " +
                                     std::to_string((uint64_t(result.y) << 32u) | result.x) + ")");
        if (result.y || result.x > UINT32_MAX / MaxContactPointsPerPair || result.w > result.x ||
            (result.x == 0 && result.w != 0))
            throw std::runtime_error("Jolt GPU broadphase returned invalid candidate metadata");
        if (count) {
            workset = std::make_shared<JoltGpuContactWorkset>();
            workset->patchCount = result.x;
            workset->maxPatchesPerWorld = result.w;
            const uint64_t words = uint64_t(scratchWords) + 2ull * result.x;
            if (words > UINT32_MAX)
                throw std::overflow_error("Jolt GPU broadphase candidate arena exceeds device word indexing");
            config.workspace.x = static_cast<uint32_t>(words);
            config.candidates = {scratchWords, scratchWords + result.x, result.x, 0};
            parameters.solve.z = result.x * MaxContactPointsPerPair;
            parameters.solve.w = result.x;
            phase = Phase::FinalCapacity;
        } else {
            if (result.x != workset->patchCount || result.w != workset->maxPatchesPerWorld)
                throw std::runtime_error("Jolt GPU broadphase count/fill metadata differs");
            BeginSort(config.candidates.x, config.candidates.y, workset->patchCount, 2);
            phase = Phase::CandidateSort;
        }
    }

    JoltGpuControlWork Next()
    {
        allocationBlocked = packetFull = false;
        indirectEntry = EntryCount;
        indirectCount = 0;
        if (!Capacity())
            return {nullptr, [self = shared_from_this()] { return self->Next(); }};
        std::vector<rhi::ComputeBufferUpdate> uploads;
        std::vector<rhi::ComputeDispatchDesc> dispatches;
        std::vector<rhi::ComputeBufferCopy> copies;
        if (phase == Phase::Copy) {
            const uint64_t total = uint64_t(scratchWords) * WordBytes;
            for (uint32_t index = 0; index < MaxDispatchesPerPacket && copyFirst < total; ++index) {
                const uint64_t bytes = std::min(uint64_t(BroadphaseItemLimit) * WordBytes, total - copyFirst);
                copies.push_back({oldArena, arena, {copyFirst, copyFirst, bytes}});
                copyFirst += bytes;
            }
            if (copyFirst == total) {
                phase = Phase::FinalInitialize;
                // Preserve the copied scratch: only the new exact 2C suffix is cleared.
                item = scratchWords;
                finalUploads = true;
            }
        } else {
            if (finalUploads) {
                uploads.push_back({workset->parameters, 0, Bytes(parameters)});
                if (!workset->patchCount)
                    uploads.push_back({workset->contacts, 0, Bytes(Contact{})});
                finalUploads = false;
            }
            while (dispatches.size() < MaxDispatchesPerPacket) {
                if (!NextDispatch(uploads, dispatches))
                    break;
                if (allocationBlocked || packetFull)
                    break;
                if (phase == Phase::CountBoundary || phase == Phase::FillBoundary || phase == Phase::ActiveBoundary ||
                    phase == Phase::Done)
                    break;
            }
        }
        const auto boundary = phase;
        std::vector<rhi::ComputeBufferRead> reads;
        if (boundary == Phase::CountBoundary || boundary == Phase::FillBoundary)
            reads.push_back({arena, uint64_t(config.workspace.y + 1u) * WordBytes, WordBytes});
        if (boundary == Phase::ActiveBoundary)
            reads.push_back({arena, uint64_t(config.activeState.x) * WordBytes, WordBytes});
        if (dispatches.empty() && copies.empty() && uploads.empty() && allocationBlocked)
            return {nullptr, [self = shared_from_this()] { return self->Next(); }};
        if (dispatches.empty() && copies.empty() && uploads.empty())
            throw std::logic_error("Jolt GPU broadphase continuation produced an empty packet");
        auto packet = rhi::PrepareComputeBatch(implementation->host, std::move(uploads), std::move(dispatches),
                                               std::move(reads), std::move(copies));
        return {packet, [self = shared_from_this(), packet, boundary]() -> JoltGpuControlWork {
                    if (boundary == Phase::FinalInitialize)
                        self->oldArena.reset();
                    if (boundary == Phase::CountBoundary || boundary == Phase::FillBoundary)
                        self->Decode(packet, boundary == Phase::CountBoundary);
                    if (boundary == Phase::ActiveBoundary)
                        self->DecodeActive(packet);
                    if (boundary == Phase::Done) {
                        self->workset->parameters = self->workset->parameters->Freeze();
                        self->workset->worlds = self->runtimeWorlds->Freeze();
                        self->workset->patches = self->arena;
                        auto callback = std::move(self->completed);
                        callback(self->workset);
                        return {};
                    }
                    return self->Next();
                }};
    }

    std::shared_ptr<Impl> implementation;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> settingsPorts;
    std::shared_ptr<rhi::ComputeBuffer> arena, oldArena, runtimeWorlds, finalArena, arguments;
    std::shared_ptr<rhi::ComputeBuffer> pendingUniform, pendingWorkUniform;
    std::shared_ptr<JoltGpuContactWorkset> workset;
    StepParams parameters;
    BroadphaseParams config{};
    std::function<void(std::shared_ptr<const JoltGpuContactWorkset>)> completed;
    std::vector<ScanLevel> levels;
    ScanCursor scan;
    Phase phase = Phase::InitialCapacity;
    std::array<uint32_t, 2> sap{}, suppression{}, queues{}, activeMeta{};
    uint32_t scratchWords = 0, zeroRecord = 0, item = 0, rounds = 0, round = 0;
    uint32_t sortSource = 0, sortTarget = 0, sortCount = 0, sortKey = 0, sortWidth = 1;
    uint64_t copyFirst = 0;
    ActiveStage activeStage = ActiveStage::Seed;
    Entry indirectEntry = EntryCount;
    uint32_t activeBound = 0, windowRounds = 0;
    uint32_t indirectMeta = 0, indirectLevel = 0, indirectFirst = 0, indirectCount = 0;
    bool finalUploads = false, allocationBlocked = false, packetFull = false, activeFilling = false;
};

JoltGpuBroadphase::JoltGpuBroadphase(rhi::ComputeHost &host, const ShaderTargetCode &target)
    : m_impl(std::make_shared<Impl>(host, target))
{
}

JoltGpuControlWork
JoltGpuBroadphase::Prepare(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs, const StepParams &parameters,
                           std::function<void(std::shared_ptr<const JoltGpuContactWorkset>)> completed)
{
    return std::make_shared<Impl::Execution>(m_impl, storage, inputs, parameters, std::move(completed))->Next();
}

} // namespace infernux
