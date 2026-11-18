#include "RayWorldTailCache.h"

#include "RayTailCacheSource.h"
#include "RayWorldGeometry.h"
#include "RayWorldMaterials.h"
#include <core/error/InxError.h>
#include <core/threading/JobSystem.h>
#include <function/renderer/RenderWorld.h>
#include <function/renderer/rhi/RhiBuffer.h>
#include <function/renderer/rhi/RhiCommand.h>
#include <function/renderer/shader/DeviceShaderProgram.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <shared_mutex>
#include <stdexcept>
#include <vector>

namespace infernux::raytracing
{
namespace
{
using Buffer = std::shared_ptr<const rhi::BufferResource>;
using Clock = std::chrono::steady_clock;
constexpr uint64_t CloneWindowBytes = 256 * 1024;
constexpr uint64_t SampleNanoseconds = 2000;
constexpr uint64_t CloneNanoseconds = 4;
struct alignas(16) Control
{
    std::array<uint32_t, 16> words{};
};
static_assert(sizeof(Control) == 64);

template <class A, class B> bool SameOwner(const A &a, const B &b)
{
    return !a.owner_before(b) && !b.owner_before(a);
}

struct Key
{
    uint64_t world, epoch, lighting;
    uint32_t visibility;
    Buffer geometry, meshes;
    rhi::SubmissionTicket producer;
    std::shared_ptr<const RayWorldMaterialPublication> materials;

    bool operator==(const Key &other) const noexcept
    {
        return world == other.world && epoch == other.epoch && lighting == other.lighting &&
               visibility == other.visibility && SameOwner(geometry, other.geometry) &&
               SameOwner(meshes, other.meshes) && producer == other.producer && SameOwner(materials, other.materials);
    }
};

Buffer Allocate(rhi::Device &device, uint64_t bytes, bool upload)
{
    if (!device.GetCapabilities().IsStorageBufferRangeValid(0, bytes))
        throw std::length_error("Directional ray cache exceeds the device storage range");
    rhi::BufferDesc desc;
    desc.byteSize = bytes;
    desc.usage = rhi::BufferUsageFlags::Storage | rhi::BufferUsageFlags::TransferSource |
                 rhi::BufferUsageFlags::TransferDestination;
    desc.memory = upload ? rhi::BufferMemory::Upload : rhi::BufferMemory::DeviceLocal;
    desc.queueAccess = rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute;
    const auto value = device.CreateBuffer(desc);
    if (!value.IsValid())
        throw std::runtime_error("Could not allocate directional ray cache storage");
    return std::make_shared<rhi::BufferResource>(device, value, bytes);
}

uint64_t Elapsed(Clock::time_point start)
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}
} // namespace

class RayTailCacheProgram final
{
  public:
    struct Entry
    {
        std::unique_ptr<DeviceShaderProgram> shader;
        rhi::ComputePipelineHandle pipeline;
    };

    RayTailCacheProgram(rhi::Device &owner, const ShaderCompilation &compiled)
        : device(owner), lifetime(owner.GetLifetime())
    {
        auto code = compiled.targets.front();
        for (auto &resource : code.resources)
            if (resource.name == "tail_samples")
                resource.binding.sampleType = rhi::TextureSampleType::UnfilterableFloat;
        for (size_t index = 0; index < 2; ++index) {
            const char *name = index ? "merge_main" : "prepare_main";
            const auto it = std::find_if(code.entries.begin(), code.entries.end(),
                                         [name](const auto &item) { return item.name == name; });
            if (it == code.entries.end())
                throw std::logic_error("Directional cache compute entry is missing");
            auto &entry = entries[index];
            entry.shader = std::make_unique<DeviceShaderProgram>(
                device, code, std::vector<uint32_t>{uint32_t(it - code.entries.begin())});
            rhi::ComputePipelineDesc desc;
            entry.shader->Apply(desc);
            entry.pipeline = device.CreateComputePipeline(desc);
            if (!entry.pipeline.IsValid())
                throw std::runtime_error("Directional cache compute pipeline creation failed");
        }
    }

    ~RayTailCacheProgram()
    {
        std::shared_lock lock(lifetime->gate);
        if (lifetime->alive.load(std::memory_order_acquire))
            for (const auto &entry : entries)
                device.Release(entry.pipeline);
    }

    rhi::BindGroupHandle Bind(size_t index, const std::array<Buffer, 4> &buffers,
                              const std::array<rhi::TextureViewHandle, 4> &images) const
    {
        static constexpr std::array<const char *, 4> names{"cache_read", "cache_write", "cache_requests",
                                                           "cache_control"};
        rhi::BindGroupDesc desc;
        desc.layout = entries[index].shader->GetBindingLayouts().front();
        for (const auto &resource : entries[index].shader->GetResources()) {
            if (resource.binding.type == rhi::BindingType::StorageBuffer) {
                const auto it = std::find(names.begin(), names.end(), resource.name);
                if (it == names.end())
                    throw std::logic_error("Unexpected directional cache buffer");
                const auto &buffer = buffers[it - names.begin()];
                desc.buffers[desc.bufferCount++] = {resource.binding.binding, rhi::BindingType::StorageBuffer,
                                                    buffer->GetBuffer(), 0, buffer->GetByteSize()};
            } else if (resource.binding.type == rhi::BindingType::SampledTexture) {
                const auto image = index ? 0u : static_cast<uint32_t>(resource.name.back() - '0');
                desc.textures[desc.textureCount++] = {
                    resource.binding.binding, rhi::BindingType::SampledTexture, images[image], {}, false};
            }
        }
        const auto group = device.CreateBindGroup(desc);
        if (!group.IsValid())
            throw std::runtime_error("Directional cache binding creation failed");
        return group;
    }

    rhi::Device &device;
    std::shared_ptr<rhi::DeviceLifetime> lifetime;
    std::array<Entry, 2> entries;
};

class RayTailCacheRequest final
{
  public:
    RayTailCacheRequest(Key identity, uint32_t token) : key(std::move(identity)), epoch(token)
    {
    }
    const Key key;
    const uint32_t epoch;
    uint32_t nextGeneration = 1;
    uint32_t cloneCursor = 0;
    uint32_t sourceCursor = 0;
    uint64_t lastAcquire = 0;
    Buffer write;
    Buffer idleControl;
    std::shared_ptr<const RayTailCachePublication> published;
    std::weak_ptr<RayTailCacheBatch> leased;
    std::shared_ptr<RayTailCacheBatch> pending;
};

struct RayTailCacheBatch::State
{
    std::shared_ptr<RayTailCacheRequest> request;
    std::shared_ptr<RayTailCacheProgram> program;
    std::shared_ptr<const RayTailCachePublication> read;
    std::array<Buffer, 4> buffers;
    std::shared_ptr<rhi::WorkCompletionReceipt> receipt = std::make_shared<rhi::WorkCompletionReceipt>();
    Control control;
    rhi::RenderViewId view = 0;
    std::vector<rhi::BindGroupHandle> groups;
    uint64_t prediction = 0;
    uint64_t opportunity = 0, gpuGrant = 0, transferGrant = 0, transferBytes = 0;
    std::function<std::optional<rhi::GpuTimestampFrame>()> gpuTiming;
    bool prepareRecorded = false, mergeRecorded = false, accepted = false;
};

RayTailCacheBatch::RayTailCacheBatch(std::unique_ptr<State> state) : m_state(std::move(state))
{
}
RayTailCacheBatch::~RayTailCacheBatch()
{
    std::shared_lock lock(m_state->program->lifetime->gate);
    if (m_state->program->lifetime->alive.load(std::memory_order_acquire))
        for (const auto group : m_state->groups)
            m_state->program->device.Release(group);
}
bool RayTailCacheBatch::Allows(uint32_t gate) const noexcept
{
    return gate == 0 || (!m_state->accepted && (gate == 1 || m_state->control.words[6] == 1));
}
bool RayTailCacheBatch::Accepted() const noexcept
{
    return m_state->accepted;
}
const std::shared_ptr<rhi::WorkCompletionReceipt> &RayTailCacheBatch::Receipt() const noexcept
{
    return m_state->receipt;
}
const std::array<Buffer, 4> &RayTailCacheBatch::Buffers() const noexcept
{
    return m_state->buffers;
}
void RayTailCacheBatch::SetGpuTiming(std::function<std::optional<rhi::GpuTimestampFrame>()> timing)
{
    m_state->gpuTiming = std::move(timing);
}
uint32_t RayTailCacheBatch::RecordPrepare(const rhi::ComputeCommandEncoder &commands,
                                          const std::array<rhi::TextureViewHandle, 2> &requests)
{
    const auto &program = *m_state->program;
    const auto group = program.Bind(0, m_state->buffers, {requests[0], requests[1], {}, {}});
    m_state->groups.push_back(group);
    commands.BindPipeline(program.entries[0].pipeline);
    commands.BindGroup(program.entries[0].pipeline, 0, group);
    const auto count = m_state->control.words[6] == 0 ? m_state->control.words[5] : 1u;
    const auto groups = (count + 63u) / 64u;
    commands.Dispatch(groups);
    m_state->prepareRecorded = true;
    return groups;
}
uint32_t RayTailCacheBatch::RecordMerge(const rhi::ComputeCommandEncoder &commands, rhi::TextureViewHandle samples)
{
    const auto &program = *m_state->program;
    const auto group = program.Bind(1, m_state->buffers, {samples, {}, {}, {}});
    m_state->groups.push_back(group);
    commands.BindPipeline(program.entries[1].pipeline);
    commands.BindGroup(program.entries[1].pipeline, 0, group);
    const auto groups = (m_state->control.words[1] + 63u) / 64u;
    commands.Dispatch(groups);
    m_state->mergeRecorded = true;
    return groups;
}

struct RayWorldTailCache::State
{
    State(FrameWorkOwner &frameOwner, rhi::Device &rhiDevice, RayTailCacheConfig settings)
        : owner(frameOwner), device(rhiDevice), config(settings)
    {
    }
    FrameWorkOwner &owner;
    rhi::Device &device;
    RayTailCacheConfig config;
    FrameWorkOwner::SourceId source = 0;
    uint32_t epoch = 0;
    Buffer empty;
    Buffer idleWrite, idleRequests;
    std::vector<std::weak_ptr<RayTailCacheRequest>> requests;
    std::weak_ptr<RayTailCacheBatch> leased;
    size_t nextRequest = 0;
    std::vector<std::weak_ptr<const rhi::BufferResource>> allocations;
    struct Compile
    {
        std::optional<ShaderCompilation> value;
        std::exception_ptr failure;
        uint64_t elapsed = 0;
    };
    std::shared_ptr<Compile> compile;
    JobHandle compilation;
    std::shared_ptr<RayTailCacheProgram> program;
    std::vector<std::shared_ptr<RayTailCacheBatch>> timings;
    uint64_t sampleNanoseconds = SampleNanoseconds, cloneNanoseconds = CloneNanoseconds;
    uint64_t compileOpportunity = 0, compileGrant = 0;
    bool failed = false;
};

RayWorldTailCache::RayWorldTailCache(FrameWorkOwner &owner, rhi::Device &device, RayTailCacheConfig config)
    : m_state(std::make_unique<State>(owner, device, config))
{
    auto &s = *m_state;
    if (!config.capacity || !config.batchSize || config.capacity > 0x3fffffffu || config.cellSize <= 0 ||
        !config.directionGrid || config.maxRayDistance <= 0 ||
        uint64_t(config.capacity) * 128 + 32 > config.residentBytes)
        throw std::invalid_argument("Directional ray cache has an invalid fixed resource configuration");
    s.empty = Allocate(device, 32 + uint64_t(config.capacity) * 128, true);
    s.idleWrite = Allocate(device, 32 + uint64_t(config.capacity) * 128, false);
    s.idleRequests = Allocate(device, 64 + uint64_t(config.batchSize) * 64, true);
    s.allocations = {s.empty, s.idleWrite, s.idleRequests};
    std::array<uint32_t, 8> cold{};
    std::memcpy(&cold[4], &config.cellSize, sizeof(float));
    cold[5] = config.directionGrid;
    std::memcpy(&cold[6], &config.maxRayDistance, sizeof(float));
    if (!device.WriteBuffer(s.empty->GetBuffer(), 0, cold.data(), sizeof(cold)))
        throw std::runtime_error("Could not initialize the empty directional ray cache");
    const auto coldGrant = owner.RecordAcceptedResource(FrameWorkResource::Transfer, 32);
    owner.ObserveResource(FrameWorkResource::Transfer, 32, owner.Opportunity(), coldGrant);
    FrameWorkSource source;
    source.observe = [this] { Observe(); };
    source.refresh = source.observe;
    source.describe = [this] { return Describe(); };
    source.advance = [this](bool, uint64_t) { return Advance(); };
    s.source = owner.AddSource(std::move(source));
}
RayWorldTailCache::~RayWorldTailCache()
{
    m_state->owner.RemoveSource(m_state->source);
    for (const auto &entry : m_state->requests)
        if (auto request = entry.lock())
            request->pending.reset();
}

uint64_t RayWorldTailCache::AvailableBytes()
{
    auto &s = *m_state;
    uint64_t remaining = s.config.residentBytes;
    s.allocations.erase(std::remove_if(s.allocations.begin(), s.allocations.end(),
                                       [&](const auto &entry) {
                                           const auto value = entry.lock();
                                           if (!value)
                                               return true;
                                           remaining -= std::min(remaining, value->GetByteSize());
                                           return false;
                                       }),
                        s.allocations.end());
    return remaining;
}

RayTailCacheBinding RayWorldTailCache::Acquire(const std::shared_ptr<const RenderWorldFrame> &world,
                                               const RayWorldGeometryBinding &geometry,
                                               const std::shared_ptr<const RayWorldMaterialPublication> &materials,
                                               std::optional<rhi::RenderViewId> updateView,
                                               std::shared_ptr<RayTailCacheRequest> &request)
{
    auto &s = *m_state;
    const Key key{world->WorldId(),
                  world->SimulationEpoch(),
                  world->Lighting() ? world->Lighting()->generation : 0,
                  geometry.visibilitySerial,
                  geometry.buffer,
                  geometry.meshes,
                  geometry.producer,
                  materials};
    if (!request || !(request->key == key)) {
        request.reset();
        for (const auto &entry : s.requests)
            if (auto item = entry.lock(); item && item->key == key) {
                request = std::move(item);
                break;
            }
        if (!request && updateView) {
            request = std::make_shared<RayTailCacheRequest>(key, ++s.epoch);
            Control control;
            control.words[0] = s.config.capacity;
            control.words[2] = request->epoch;
            std::memcpy(&control.words[8], &s.config.cellSize, sizeof(float));
            control.words[9] = s.config.directionGrid;
            control.words[10] = 1;
            std::memcpy(&control.words[14], &s.config.maxRayDistance, sizeof(float));
            request->idleControl = Allocate(s.device, sizeof(Control), true);
            if (!s.device.WriteBuffer(request->idleControl->GetBuffer(), 0, &control, sizeof(control)))
                throw std::runtime_error("Could not initialize directional ray cache controls");
            const auto grant = s.owner.RecordAcceptedResource(FrameWorkResource::Transfer, sizeof(control));
            s.owner.ObserveResource(FrameWorkResource::Transfer, sizeof(control), s.owner.Opportunity(), grant);
            s.allocations.emplace_back(request->idleControl);
            s.requests.emplace_back(request);
        }
        if (updateView)
            s.owner.Wake(s.source);
    }
    if (!updateView) {
        // A view may consume another view's completed cache without owning any work.
        auto publication = request ? request->published : nullptr;
        RayTailCacheBinding result{};
        result.buffers[0] = publication ? publication->entries : s.empty;
        result.byteSizes[0] = 32 + uint64_t(s.config.capacity) * 128;
        result.publication = std::move(publication);
        return result;
    }
    auto batch = PrepareBatch(request, *updateView);
    if (!request->write && !request->pending)
        s.owner.Wake(s.source);
    const auto read = request->published ? request->published->entries : s.empty;
    return {batch ? batch->Buffers() : std::array{read, s.idleWrite, s.idleRequests, request->idleControl},
            request->published,
            std::move(batch),
            {32 + uint64_t(s.config.capacity) * 128, 32 + uint64_t(s.config.capacity) * 128,
             64 + uint64_t(s.config.batchSize) * 64, sizeof(Control)}};
}

std::shared_ptr<RayTailCacheBatch> RayWorldTailCache::PrepareBatch(const std::shared_ptr<RayTailCacheRequest> &request,
                                                                   rhi::RenderViewId view)
{
    auto &s = *m_state;
    request->lastAcquire = s.owner.Opportunity();
    if (request->pending || !s.program || !request->write || s.owner.IsGpuForegroundBlocked())
        return {};
    if (auto leased = s.leased.lock(); leased && !leased->Receipt()->complete && !leased->Receipt()->failed)
        return !leased->Accepted() && leased->m_state->request == request && leased->m_state->view == view ? leased
                                                                                                           : nullptr;
    for (size_t i = 0; i < s.requests.size(); ++i) {
        const auto index = (s.nextRequest + i) % s.requests.size();
        const auto candidate = s.requests[index].lock();
        if (!candidate || !candidate->write || candidate->pending || candidate->lastAcquire + 1 < s.owner.Opportunity())
            continue;
        if (candidate != request)
            return {};
        break;
    }
    const bool cloning = request->cloneCursor < s.config.capacity;
    const auto transferRemaining = s.owner.ResourceRemaining(FrameWorkResource::Transfer);
    const auto gpuRemaining = s.owner.ResourceRemaining(FrameWorkResource::Gpu);
    if (transferRemaining < 2 * sizeof(Control) || !gpuRemaining)
        return {};
    const auto unit = cloning ? s.cloneNanoseconds : s.sampleNanoseconds;
    const auto gpuCount = std::max<uint64_t>(1, gpuRemaining / unit);
    const auto count = cloning ? std::min<uint64_t>({s.config.capacity - request->cloneCursor, CloneWindowBytes / 128,
                                                     (transferRemaining - 2 * sizeof(Control)) / 128, gpuCount})
                               : std::min<uint64_t>(s.config.batchSize, gpuCount);
    const auto prediction = count * (cloning ? s.cloneNanoseconds : s.sampleNanoseconds);
    const uint64_t transfer = 2 * sizeof(Control) + (cloning ? count * 128 : 0);
    const uint64_t bytes = sizeof(Control) + 64 + uint64_t(s.config.batchSize) * 64;
    if (!count || !s.owner.CanAdmitResource(FrameWorkResource::Gpu, prediction, count == 1) ||
        !s.owner.CanAdmitResource(FrameWorkResource::Transfer, transfer) || AvailableBytes() < bytes)
        return {};
    auto state = std::make_unique<RayTailCacheBatch::State>();
    state->request = request;
    state->program = s.program;
    state->read = request->published;
    state->view = view;
    state->prediction = prediction;
    auto &c = state->control.words;
    c = {s.config.capacity,
         cloning ? s.config.batchSize : uint32_t(count),
         request->epoch,
         request->nextGeneration,
         request->cloneCursor,
         cloning ? uint32_t(count) : 0,
         cloning ? 0u : 1u,
         request->sourceCursor,
         0,
         s.config.directionGrid,
         1,
         request->published ? request->published->generation : 0,
         uint32_t(request->key.world),
         uint32_t(request->key.world >> 32),
         0,
         0};
    std::memcpy(&c[8], &s.config.cellSize, sizeof(float));
    std::memcpy(&c[14], &s.config.maxRayDistance, sizeof(float));
    auto requests = Allocate(s.device, 64 + uint64_t(s.config.batchSize) * 64, true);
    auto control = Allocate(s.device, sizeof(Control), true);
    if (!s.device.WriteBuffer(control->GetBuffer(), 0, &state->control, sizeof(Control)))
        throw std::runtime_error("Could not publish directional ray cache batch controls");
    if (!s.device.WriteBuffer(requests->GetBuffer(), 0, &state->control, sizeof(Control)))
        throw std::runtime_error("Could not initialize directional ray request controls");
    const auto uploadGrant = s.owner.RecordAcceptedResource(FrameWorkResource::Transfer, 2 * sizeof(Control));
    s.owner.ObserveResource(FrameWorkResource::Transfer, 2 * sizeof(Control), s.owner.Opportunity(), uploadGrant);
    s.allocations.emplace_back(requests);
    s.allocations.emplace_back(control);
    state->buffers = {request->published ? request->published->entries : s.empty, request->write, requests, control};
    auto batch = std::shared_ptr<RayTailCacheBatch>(new RayTailCacheBatch(std::move(state)));
    request->leased = batch;
    s.leased = batch;
    return batch;
}

bool RayWorldTailCache::AcceptBatch(const std::shared_ptr<RayTailCacheBatch> &batch) noexcept
{
    auto &b = *batch->m_state;
    if (b.accepted || !b.prepareRecorded || !b.mergeRecorded || !b.receipt->submission.IsValid())
        return false;
    b.accepted = true;
    b.request->pending = batch;
    for (size_t i = 0; i < m_state->requests.size(); ++i)
        if (m_state->requests[i].lock() == b.request) {
            m_state->nextRequest = i + 1;
            break;
        }
    b.opportunity = m_state->owner.Opportunity();
    b.gpuGrant = m_state->owner.RecordAcceptedResource(FrameWorkResource::Gpu, b.prediction);
    if (b.control.words[6] == 0) {
        b.transferBytes = uint64_t(b.control.words[5]) * 128;
        b.transferGrant = m_state->owner.RecordAcceptedResource(FrameWorkResource::Transfer, b.transferBytes);
    }
    m_state->owner.Wake(m_state->source);
    return true;
}

void RayWorldTailCache::Observe()
{
    auto &s = *m_state;
    if (s.compilation.IsValid() && s.compilation.IsComplete()) {
        s.owner.ObserveResource(FrameWorkResource::Cpu, s.compile->elapsed, s.compileOpportunity, s.compileGrant);
        s.compilation = {};
        if (s.compile->failure) {
            try {
                std::rethrow_exception(s.compile->failure);
            } catch (const std::exception &error) {
                INXLOG_ERROR("Directional ray cache shader failed: ", error.what());
            }
            s.failed = true;
            s.compile.reset();
        }
        s.owner.ReportProgress();
    }
    for (const auto &entry : s.requests) {
        const auto request = entry.lock();
        if (!request || !request->pending)
            continue;
        const auto batch = request->pending;
        const auto &b = *batch->m_state;
        if (!b.receipt->complete && !b.receipt->failed)
            continue;
        if (b.receipt->complete && b.transferBytes)
            s.owner.ObserveResource(FrameWorkResource::Transfer, b.transferBytes, b.opportunity, b.transferGrant);
        if (b.receipt->complete && b.gpuTiming)
            s.timings.push_back(batch);
        if (b.receipt->failed) {
            request->write.reset();
            request->cloneCursor = 0;
        } else if (b.control.words[6] == 0) {
            request->cloneCursor += b.control.words[5];
        } else {
            auto published = std::make_shared<RayTailCachePublication>();
            published->entries = std::move(request->write);
            published->generation = request->nextGeneration++;
            published->producer = b.receipt->submission;
            request->published = std::move(published);
            request->cloneCursor = 0;
            request->sourceCursor += b.control.words[1];
        }
        request->pending.reset();
        request->leased.reset();
        s.owner.ReportProgress();
    }
    for (auto it = s.timings.begin(); it != s.timings.end();) {
        auto &batch = *(*it)->m_state;
        const auto result = batch.gpuTiming();
        if (!result) {
            ++it;
            continue;
        }
        double nanoseconds = 0;
        for (uint32_t i = 0; i < result->sampleCount; ++i)
            nanoseconds += result->samples[i].milliseconds * 1.0e6;
        const auto actual = static_cast<uint64_t>(nanoseconds);
        s.owner.ObserveResource(FrameWorkResource::Gpu, actual, batch.opportunity, batch.gpuGrant);
        const bool clone = batch.control.words[6] == 0;
        const auto count = clone ? batch.control.words[5] : batch.control.words[1];
        auto &prediction = clone ? s.cloneNanoseconds : s.sampleNanoseconds;
        prediction = std::max<uint64_t>(1, (actual + count - 1) / count);
        batch.gpuTiming = {};
        it = s.timings.erase(it);
    }
    s.requests.erase(std::remove_if(s.requests.begin(), s.requests.end(), [](const auto &v) { return v.expired(); }),
                     s.requests.end());
}

FrameWorkFacts RayWorldTailCache::Describe()
{
    auto &s = *m_state;
    FrameWorkFacts facts;
    if (s.failed || s.requests.empty())
        return facts;
    facts.pending = !s.program;
    facts.observe = !s.timings.empty();
    facts.pending |= !s.timings.empty();
    facts.workerBusy = s.compilation.IsValid();
    if (!facts.workerBusy && !s.program)
        facts.host.ready = true;
    const bool canAllocate = AvailableBytes() >= 32 + uint64_t(s.config.capacity) * 128;
    for (const auto &entry : s.requests)
        if (auto request = entry.lock()) {
            facts.gpuBusy |= bool(request->pending);
            facts.observe |= bool(request->pending);
            facts.host.ready |= canAllocate && s.program && !request->write && !request->pending;
            facts.pending |= bool(request->pending) || (canAllocate && !request->write && !request->pending);
        }
    facts.host.ready &= s.owner.CanAdmitResource(FrameWorkResource::Cpu, 0);
    return facts;
}

bool RayWorldTailCache::Advance()
{
    auto &s = *m_state;
    if (s.failed)
        return false;
    if (!s.program) {
        if (s.compilation.IsValid())
            return false;
        if (!s.compile) {
            s.compile = std::make_shared<State::Compile>();
            const auto target = s.device.GetShaderSourceLanguage() == rhi::ShaderSourceLanguage::SpirV
                                    ? ShaderCodeTarget::SpirV
                                    : ShaderCodeTarget::Wgsl;
            const auto run = [value = s.compile, target] {
                const auto start = Clock::now();
                try {
                    value->value = SlangCompiler::Get().Compile(
                        {"inx_ray_tail_update", "<infernux/shader/ray_tail_update>", RayTailCacheSource()}, {},
                        {target}, true);
                } catch (...) {
                    value->failure = std::current_exception();
                }
                value->elapsed = Elapsed(start);
            };
            s.compileOpportunity = s.owner.Opportunity();
            s.compileGrant = s.owner.RecordAcceptedResource(FrameWorkResource::Cpu, 0);
            if (JobSystem::Get().IsInline()) {
                run();
                s.owner.ObserveResource(FrameWorkResource::Cpu, s.compile->elapsed, s.compileOpportunity,
                                        s.compileGrant);
            } else {
                s.compilation =
                    JobSystem::Get().Schedule(run, JobDomain::Render, JobPriority::Low, JobAffinity::WorkerOnly);
                return true;
            }
        }
        if (s.compile->failure) {
            s.failed = true;
            return true;
        }
        const auto started = Clock::now();
        const auto grant = s.owner.RecordAcceptedResource(FrameWorkResource::Cpu, 0);
        s.program = std::make_shared<RayTailCacheProgram>(s.device, *s.compile->value);
        s.compile.reset();
        s.owner.ObserveResource(FrameWorkResource::Cpu, Elapsed(started), s.owner.Opportunity(), grant);
        return true;
    }
    const uint64_t bytes = 32 + uint64_t(s.config.capacity) * 128;
    if (AvailableBytes() < bytes)
        return false;
    for (const auto &entry : s.requests)
        if (auto request = entry.lock(); request && !request->write && !request->pending) {
            const auto start = Clock::now();
            const auto grant = s.owner.RecordAcceptedResource(FrameWorkResource::Cpu, 0);
            request->write = Allocate(s.device, bytes, false);
            s.allocations.emplace_back(request->write);
            s.owner.ObserveResource(FrameWorkResource::Cpu, Elapsed(start), s.owner.Opportunity(), grant);
            return true;
        }
    return false;
}
} // namespace infernux::raytracing
