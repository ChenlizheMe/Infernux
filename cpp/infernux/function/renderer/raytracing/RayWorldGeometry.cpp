#include "RayWorldGeometry.h"

#include "RayGeometry.h"

#include <core/error/InxError.h>
#include <function/renderer/RenderWorld.h>
#include <function/renderer/rhi/RhiBuffer.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMesh/InxMesh.h>
#include <function/resources/InxMesh/MeshArtifactReader.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>

namespace infernux::raytracing
{
namespace
{
using WorldRevision = std::array<uint64_t, 6>;
constexpr uint32_t FullBuild = UINT32_MAX;
constexpr uint32_t InlineWorkWindow = 1024;
constexpr uint64_t UploadWindow = 256 * 1024;

WorldRevision GetRevision(const RenderWorldFrame &world)
{
    return {world.WorldId(),           world.SimulationEpoch(), world.StructuralRevision(),
            world.TransformRevision(), world.ContentRevision(), world.PoseRevision()};
}

uint64_t AddStorage(uint64_t bytes, uint64_t extra)
{
    const auto maximum = (std::numeric_limits<uint64_t>::max)();
    return bytes > maximum - extra ? maximum : bytes + extra;
}

std::shared_ptr<const rhi::BufferResource> AllocatePage(rhi::Device &device, uint64_t bytes)
{
    if (!device.GetCapabilities().IsStorageBufferRangeValid(0, bytes))
        throw std::length_error("World ray geometry exceeds the device storage range");
    rhi::BufferDesc desc;
    desc.byteSize = bytes;
    desc.usage = rhi::BufferUsageFlags::Storage;
    desc.memory = rhi::BufferMemory::Upload;
    const auto allocation = device.CreateBuffer(desc);
    if (!allocation.IsValid())
        throw std::runtime_error("Could not allocate the world's ray geometry page");
    return std::make_shared<rhi::BufferResource>(device, allocation, bytes);
}

void WritePage(rhi::Device &device, const rhi::BufferResource &buffer, uint64_t offset, const void *data,
               uint64_t bytes)
{
    if (!device.WriteBuffer(buffer.GetBuffer(), offset, data, bytes))
        throw std::runtime_error("Could not publish the world's ray geometry page");
}
} // namespace

class RayWorldGeometryPublication final
{
  public:
    explicit RayWorldGeometryPublication(std::shared_ptr<const RenderWorldFrame> frame)
        : revision(GetRevision(*frame)), world(std::move(frame))
    {
    }

    const WorldRevision revision;
    std::shared_ptr<const RenderWorldFrame> world;
    std::shared_ptr<const rhi::BufferResource> buffer;
    RayGeometryStatus status = RayGeometryStatus::Pending;
};

class RayWorldGeometryBuild final
{
  public:
    enum class Step : uint8_t
    {
        Plan,
        Open,
        Decode,
        BuildRecovered,
        BuildResident,
        CollectInstances,
        BuildPage,
        Upload,
    };

    bool Plan(uint32_t window);
    void Run(uint32_t window);
    [[nodiscard]] bool Upload(rhi::Device &device, uint64_t bytes);
    [[nodiscard]] FrameWorkResource Resource() const;
    [[nodiscard]] uint64_t UploadBytes(uint64_t remaining) const;
    [[nodiscard]] uint64_t DecodeWorkspace() const
    {
        return reader->WorkspaceBytes();
    }

    Step step = Step::Plan;
    std::weak_ptr<RayWorldGeometryPublication> publication;
    std::shared_ptr<const RenderWorldFrame> world;
    std::shared_ptr<FrameCpuStorageClaim> storage;
    std::shared_ptr<FrameCpuStorageClaim> decodeStorage;
    std::shared_ptr<const rhi::BufferResource> buffer;
    std::exception_ptr failure;
    uint64_t workspace = 2 * sizeof(RayGeometryWord);
    uint64_t elapsed = 0;

  private:
    using SliceKey = std::tuple<size_t, uintptr_t, uintptr_t, MeshGeometryView, uint32_t, uint32_t, int32_t>;

    struct Slice
    {
        const DrawCall *draw;
        size_t mesh;
    };

    struct Recovery
    {
        MeshArtifactReference artifact;
        std::vector<Slice> slices;
    };

    bool BuildSlices(const std::vector<Slice> &work, uint32_t window);

    std::map<SliceKey, size_t> sliceIndices;
    std::map<std::pair<std::string, uint64_t>, size_t> artifactIndices;
    size_t nextDraw = 0;
    std::vector<RayMesh> meshes;
    std::vector<Slice> instances;
    std::vector<Slice> resident;
    std::vector<Recovery> recovery;
    size_t nextArtifact = 0;
    std::unique_ptr<MeshArtifactReader> reader;
    std::shared_ptr<InxMesh> decodedMesh;
    size_t nextSlice = 0;
    std::unique_ptr<RayMeshBuild> meshBuild;
    std::vector<RayMeshInstance> rays;
    std::unique_ptr<RayGeometryPageBuild> pageBuild;
    RayGeometryPage page;
    std::optional<RayGeometryPageLayout> layout;
    uint64_t uploadedBytes = 0;
    size_t uploadSection = 0;
    uint64_t sectionOffset = 0;
};

bool RayWorldGeometryBuild::Plan(uint32_t window)
{
    // Matches the exact draw ranges of the frozen raster world. Ray queries
    // also need geometry outside the current camera frustum.
    const auto &draws = world->DrawCalls().drawCalls;
    const auto end = nextDraw + std::min<size_t>(window, draws.size() - nextDraw);
    for (; nextDraw < end; ++nextDraw) {
        const auto &draw = draws[nextDraw];
        if (!draw.indexCount)
            continue;
        if (!draw.surfaceId || draw.meshVertexBuffer || draw.devicePose.HasValue() || draw.skinBoneMatrices)
            return false;

        size_t artifact = 0;
        const bool hasStreams = draw.meshDataOwner && draw.meshVertices && draw.meshIndices &&
                                !draw.meshVertices->empty() && !draw.meshIndices->empty();
        if (!hasStreams) {
            if (draw.meshUsesDetachedStorage || draw.meshAssetGuid.empty())
                return false;
            auto [entry, inserted] =
                artifactIndices.try_emplace({draw.meshAssetGuid, draw.meshRuntimeVersion}, recovery.size() + 1);
            if (inserted) {
                auto reference =
                    AssetRegistry::Instance().CaptureMeshArtifact(draw.meshAssetGuid, draw.meshRuntimeVersion);
                if (!reference)
                    return false;
                recovery.push_back({std::move(*reference), {}});
            }
            artifact = entry->second;
        }
        const auto *vertices = hasStreams ? draw.meshVertices : nullptr;
        const auto *indices = hasStreams ? draw.meshIndices : nullptr;
        const SliceKey key{artifact,
                           reinterpret_cast<uintptr_t>(vertices),
                           reinterpret_cast<uintptr_t>(indices),
                           draw.meshGeometryView,
                           draw.indexStart,
                           draw.indexCount,
                           draw.vertexStart};
        auto [entry, inserted] = sliceIndices.try_emplace(key, sliceIndices.size());
        const Slice slice{&draw, entry->second};
        instances.push_back(slice);
        workspace = AddStorage(workspace, 512);
        if (inserted) {
            // Covers the compact-index map, BLAS, and its contribution to the final page.
            workspace = AddStorage(workspace, uint64_t{draw.indexCount} * 352);
            meshes.emplace_back();
            auto &work = hasStreams ? resident : recovery[artifact - 1].slices;
            work.push_back(slice);
        }
    }
    if (nextDraw == draws.size()) {
        rays.reserve(instances.size());
        step = recovery.empty() ? Step::BuildResident : Step::Open;
    }
    return true;
}

FrameWorkResource RayWorldGeometryBuild::Resource() const
{
    switch (step) {
    case Step::Open:
    case Step::Decode:
        return FrameWorkResource::Io;
    case Step::Plan:
    case Step::BuildRecovered:
    case Step::BuildResident:
    case Step::CollectInstances:
    case Step::BuildPage:
        return FrameWorkResource::Cpu;
    case Step::Upload:
        return FrameWorkResource::Transfer;
    }
    std::terminate();
}

bool RayWorldGeometryBuild::BuildSlices(const std::vector<Slice> &work, uint32_t window)
{
    while (nextSlice < work.size()) {
        const auto &slice = work[nextSlice];
        if (!meshBuild) {
            const auto &draw = *slice.draw;
            const auto *vertices = draw.meshVertices;
            const auto *indices = draw.meshIndices;
            if (decodedMesh) {
                const auto geometry = draw.meshGeometryView == MeshGeometryView::NodeLocal
                                          ? decodedMesh->GetModelSourceGeometry()
                                          : decodedMesh->GetGeometrySnapshot();
                if (!geometry)
                    throw std::runtime_error("The cooked mesh does not contain the frozen geometry view");
                vertices = &geometry->vertices;
                indices = &geometry->indices;
            }
            meshBuild =
                std::make_unique<RayMeshBuild>(*vertices, *indices, draw.indexStart, draw.indexCount, draw.vertexStart);
        }
        if (meshBuild->Advance(window)) {
            meshes[slice.mesh] = meshBuild->Take();
            meshBuild.reset();
            ++nextSlice;
        }
        if (window != FullBuild)
            break;
    }
    if (nextSlice != work.size())
        return false;
    nextSlice = 0;
    return true;
}

void RayWorldGeometryBuild::Run(uint32_t window)
{
    const auto start = std::chrono::steady_clock::now();
    try {
        switch (step) {
        case Step::Plan:
            std::terminate();
        case Step::Open:
            reader = std::make_unique<MeshArtifactReader>(recovery[nextArtifact].artifact);
            step = Step::Decode;
            break;
        case Step::Decode:
            decodedMesh = reader->Read();
            reader.reset();
            step = Step::BuildRecovered;
            break;
        case Step::BuildRecovered:
            if (BuildSlices(recovery[nextArtifact].slices, window)) {
                decodedMesh.reset();
                step = ++nextArtifact == recovery.size() ? Step::BuildResident : Step::Open;
            }
            break;
        case Step::BuildResident:
            if (!BuildSlices(resident, window))
                break;
            step = Step::CollectInstances;
            if (window != FullBuild)
                break;
            [[fallthrough]];
        case Step::CollectInstances: {
            const auto end = rays.size() + std::min<size_t>(window, instances.size() - rays.size());
            for (; rays.size() < end;) {
                const auto &slice = instances[rays.size()];
                const auto &draw = *slice.draw;
                rays.push_back({meshes[slice.mesh], draw.worldMatrix, draw.surfaceId, draw.layerMask});
            }
            if (rays.size() != instances.size())
                break;
            pageBuild = std::make_unique<RayGeometryPageBuild>(rays);
            step = Step::BuildPage;
            if (window != FullBuild)
                break;
            [[fallthrough]];
        }
        case Step::BuildPage:
            do {
                if (pageBuild->Advance(window)) {
                    page = pageBuild->Take();
                    pageBuild.reset();
                    layout.emplace(page, RayGeometryStatus::Ready);
                    step = Step::Upload;
                    break;
                }
            } while (window == FullBuild);
            break;
        case Step::Upload:
            std::terminate();
        }
    } catch (...) {
        failure = std::current_exception();
    }
    elapsed = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

uint64_t RayWorldGeometryBuild::UploadBytes(uint64_t remaining) const
{
    const auto capacity = std::max<uint64_t>(sizeof(RayGeometryWord), remaining);
    const auto window = std::min<uint64_t>(capacity, UploadWindow);
    return std::min(layout->bytes - uploadedBytes, window / sizeof(RayGeometryWord) * sizeof(RayGeometryWord));
}

bool RayWorldGeometryBuild::Upload(rhi::Device &device, uint64_t bytes)
{
    if (!buffer)
        buffer = AllocatePage(device, layout->bytes);
    const auto sections = layout->Sections();
    while (bytes) {
        const auto &section = sections[uploadSection];
        const auto part = std::min<uint64_t>(bytes, section.bytes - sectionOffset);
        if (part) {
            const auto *data = static_cast<const std::byte *>(section.data) + sectionOffset;
            WritePage(device, *buffer, uploadedBytes, data, part);
            uploadedBytes += part;
            sectionOffset += part;
            bytes -= part;
        }
        if (sectionOffset == section.bytes) {
            ++uploadSection;
            sectionOffset = 0;
        }
    }
    return uploadedBytes == layout->bytes;
}

RayWorldGeometry::RayWorldGeometry(FrameWorkOwner &owner, rhi::Device &device) : m_owner(owner), m_device(device)
{
    FrameWorkSource source;
    source.observe = [this] { Observe(); };
    source.refresh = source.observe;
    source.describe = [this] { return Describe(); };
    source.advance = [this](bool, uint64_t) { return Advance(); };
    m_source = owner.AddSource(std::move(source));
}

RayWorldGeometry::~RayWorldGeometry()
{
    m_owner.RemoveSource(m_source);
    if (m_job.IsValid()) {
        m_job.Cancel();
        JobSystem::Get().WaitPassiveCompletion(m_job);
    }
}

RayWorldGeometryBinding RayWorldGeometry::Acquire(const std::shared_ptr<const RenderWorldFrame> &world,
                                                  std::shared_ptr<const RayWorldGeometryPublication> &publication)
{
    const auto revision = GetRevision(*world);
    if (!publication || publication->revision != revision) {
        publication.reset();
        m_publications.erase(std::remove_if(m_publications.begin(), m_publications.end(),
                                            [](const auto &entry) { return entry.expired(); }),
                             m_publications.end());
        for (const auto &entry : m_publications) {
            if (auto value = entry.lock(); value && value->revision == revision) {
                publication = std::move(value);
                break;
            }
        }
        if (!publication) {
            auto value = std::make_shared<RayWorldGeometryPublication>(world);
            m_publications.emplace_back(value);
            publication = std::move(value);
            m_owner.Wake(m_source);
        }
    }
    if (publication->buffer)
        return {publication->buffer, false};
    const bool pending = publication->status == RayGeometryStatus::Pending;
    auto &header = pending ? m_pendingBuffer : m_unavailableBuffer;
    if (!header) {
        const RayGeometryPage empty;
        const RayGeometryPageLayout layout(empty, publication->status);
        auto buffer = AllocatePage(m_device, layout.bytes);
        WritePage(m_device, *buffer, 0, layout.header.data(), layout.bytes);
        header = std::move(buffer);
    }
    return {header, pending};
}

std::shared_ptr<RayWorldGeometryPublication> RayWorldGeometry::FindPending() const
{
    for (const auto &entry : m_publications) {
        auto value = entry.lock();
        if (value && value->world)
            return value;
    }
    return {};
}

void RayWorldGeometry::Observe()
{
    if (!m_job.IsValid() || !m_job.IsComplete())
        return;
    m_job = {};
    CollectReceipt();
}

void RayWorldGeometry::CollectReceipt()
{
    m_owner.ObserveResource(m_workResource, m_active->elapsed, m_submittedOpportunity, m_workGrant);
    if (m_active->step != RayWorldGeometryBuild::Step::BuildRecovered)
        m_active->decodeStorage.reset();
}

FrameWorkFacts RayWorldGeometry::Describe()
{
    FrameWorkFacts facts;
    if (m_active) {
        facts.pending = true;
        facts.workerBusy = m_job.IsValid();
        if (!facts.workerBusy) {
            const auto resource = m_active->Resource();
            const auto prediction = resource == FrameWorkResource::Transfer
                                        ? m_active->UploadBytes(m_owner.ResourceRemaining(resource))
                                        : 0;
            facts.host.ready = m_active->publication.expired() || m_active->failure ||
                               m_owner.CanAdmitResource(resource, prediction, true);
            facts.ioReady = facts.host.ready && resource == FrameWorkResource::Io;
        }
    } else if (FindPending()) {
        facts.pending = true;
        facts.host.ready = m_owner.CanAdmitResource(FrameWorkResource::Cpu, 0);
    }
    return facts;
}

bool RayWorldGeometry::Advance()
{
    if (m_job.IsValid())
        return false;
    const auto publication = m_active ? m_active->publication.lock() : FindPending();
    if (!publication) {
        const bool wasActive = static_cast<bool>(m_active);
        m_active.reset();
        return wasActive;
    }
    try {
        if (!m_active) {
            auto build = std::make_shared<RayWorldGeometryBuild>();
            build->publication = publication;
            build->world = std::move(publication->world);
            m_active = std::move(build);
        }
        if (m_active->failure)
            std::rethrow_exception(m_active->failure);
        const auto resource = m_active->Resource();
        const auto bytes =
            resource == FrameWorkResource::Transfer ? m_active->UploadBytes(m_owner.ResourceRemaining(resource)) : 0;
        if (!m_owner.CanAdmitResource(resource, bytes, true))
            return false;
        if (resource == FrameWorkResource::Transfer) {
            const bool done = m_active->Upload(m_device, bytes);
            m_owner.RecordAcceptedResource(resource, bytes);
            if (done) {
                publication->buffer = std::move(m_active->buffer);
                publication->status = RayGeometryStatus::Ready;
                m_active.reset();
            }
            return true;
        }
        auto &jobs = JobSystem::Get();
        if (m_active->step == RayWorldGeometryBuild::Step::Plan) {
            const auto opportunity = m_owner.Opportunity();
            const auto grant = m_owner.RecordAcceptedResource(resource, 0);
            const auto started = std::chrono::steady_clock::now();
            const bool supported = m_active->Plan(jobs.IsInline() ? InlineWorkWindow : FullBuild);
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started);
            m_owner.ObserveResource(resource, static_cast<uint64_t>(elapsed.count()), opportunity, grant);
            if (!supported) {
                publication->status = RayGeometryStatus::Unavailable;
                m_active.reset();
            }
            return true;
        }
        const auto decode = m_active->step == RayWorldGeometryBuild::Step::Decode ? m_active->DecodeWorkspace() : 0;
        if (AddStorage(m_active->workspace, decode) > m_owner.CpuStorageLimit())
            throw std::length_error("Build and mesh decode exceed the CPU storage limit");
        if (!m_active->storage) {
            m_active->storage = m_owner.TryReserveCpuStorage(m_active->workspace);
            if (!m_active->storage)
                return false;
        }
        if (decode) {
            m_active->decodeStorage = m_owner.TryReserveCpuStorage(decode);
            if (!m_active->decodeStorage)
                return false;
        }
        if (!jobs.IsInline())
            m_job = jobs.Schedule([build = m_active] { build->Run(FullBuild); },
                                  resource == FrameWorkResource::Io ? JobDomain::IO : JobDomain::Render,
                                  JobPriority::Low, JobAffinity::WorkerOnly);
        m_workResource = resource;
        m_submittedOpportunity = m_owner.Opportunity();
        m_workGrant = m_owner.RecordAcceptedResource(resource, 0);
        if (jobs.IsInline()) {
            m_active->Run(InlineWorkWindow);
            CollectReceipt();
        }
    } catch (const std::exception &error) {
        INXLOG_ERROR("World ray geometry preparation failed: ", error.what());
        publication->world.reset();
        publication->status = RayGeometryStatus::Unavailable;
        m_active.reset();
    }
    return true;
}
} // namespace infernux::raytracing
