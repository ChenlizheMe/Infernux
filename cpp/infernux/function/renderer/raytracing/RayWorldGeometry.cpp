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
#include <stdexcept>
#include <tuple>

namespace infernux::raytracing
{
namespace
{
using WorldRevision = std::array<uint64_t, 6>;

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

uint64_t PageSize(const std::vector<RayGeometryWord> &page)
{
    return page.size() * sizeof(RayGeometryWord);
}

std::shared_ptr<const rhi::BufferResource> UploadPage(rhi::Device &device, const std::vector<RayGeometryWord> &page)
{
    const auto bytes = PageSize(page);
    if (!device.GetCapabilities().IsStorageBufferRangeValid(0, bytes))
        throw std::length_error("World ray geometry exceeds the device storage range");
    rhi::BufferDesc desc;
    desc.byteSize = bytes;
    desc.usage = rhi::BufferUsageFlags::Storage;
    desc.memory = rhi::BufferMemory::Upload;
    const auto allocation = device.CreateBuffer(desc);
    const auto buffer = std::make_shared<rhi::BufferResource>(device, allocation, bytes);
    if (!allocation.IsValid() || !device.WriteBuffer(allocation, 0, page.data(), bytes))
        throw std::runtime_error("Could not publish the world's ray geometry");
    return buffer;
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
        Open,
        Decode,
        BuildRecovered,
        Build,
        Upload,
    };

    bool Plan();
    void Run();
    [[nodiscard]] FrameWorkResource Resource() const;
    [[nodiscard]] uint64_t DecodeWorkspace() const
    {
        return reader->WorkspaceBytes();
    }

    Step step = Step::Build;
    std::weak_ptr<RayWorldGeometryPublication> publication;
    std::shared_ptr<const RenderWorldFrame> world;
    std::shared_ptr<FrameCpuStorageClaim> storage;
    std::shared_ptr<FrameCpuStorageClaim> decodeStorage;
    std::vector<RayGeometryWord> words;
    std::exception_ptr failure;
    uint64_t workspace = 2 * sizeof(RayGeometryWord);
    uint64_t elapsed = 0;

  private:
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

    std::vector<RayMesh> meshes;
    std::vector<Slice> instances;
    std::vector<Slice> resident;
    std::vector<Recovery> recovery;
    size_t nextArtifact = 0;
    std::unique_ptr<MeshArtifactReader> reader;
    std::shared_ptr<InxMesh> decodedMesh;
};

bool RayWorldGeometryBuild::Plan()
{
    // Matches the exact draw ranges of the frozen raster world. Ray queries
    // also need geometry outside the current camera frustum.
    using Key = std::tuple<size_t, uintptr_t, uintptr_t, MeshGeometryView, uint32_t, uint32_t, int32_t>;
    std::map<Key, size_t> slices;
    std::map<std::pair<std::string, uint64_t>, size_t> artifacts;
    for (const auto &draw : world->DrawCalls().drawCalls) {
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
                artifacts.try_emplace({draw.meshAssetGuid, draw.meshRuntimeVersion}, recovery.size() + 1);
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
        const Key key{artifact,
                      reinterpret_cast<uintptr_t>(vertices),
                      reinterpret_cast<uintptr_t>(indices),
                      draw.meshGeometryView,
                      draw.indexStart,
                      draw.indexCount,
                      draw.vertexStart};
        auto [entry, inserted] = slices.try_emplace(key, slices.size());
        const Slice slice{&draw, entry->second};
        instances.push_back(slice);
        workspace = AddStorage(workspace, 512);
        if (inserted) {
            // Covers the compact-index map, BLAS, and its contribution to the final page.
            workspace = AddStorage(workspace, uint64_t{draw.indexCount} * 256);
            auto &work = hasStreams ? resident : recovery[artifact - 1].slices;
            work.push_back(slice);
        }
    }
    meshes.resize(slices.size());
    step = recovery.empty() ? Step::Build : Step::Open;
    return true;
}

FrameWorkResource RayWorldGeometryBuild::Resource() const
{
    switch (step) {
    case Step::Open:
    case Step::Decode:
        return FrameWorkResource::Io;
    case Step::BuildRecovered:
    case Step::Build:
        return FrameWorkResource::Cpu;
    case Step::Upload:
        return FrameWorkResource::Transfer;
    }
    std::terminate();
}

void RayWorldGeometryBuild::Run()
{
    const auto start = std::chrono::steady_clock::now();
    const auto buildSlice = [this](const Slice &slice, const auto &vertices, const auto &indices) {
        const auto &draw = *slice.draw;
        meshes[slice.mesh] = BuildRayMesh(vertices, indices, draw.indexStart, draw.indexCount, draw.vertexStart);
    };
    try {
        switch (step) {
        case Step::Open:
            reader = std::make_unique<MeshArtifactReader>(recovery[nextArtifact].artifact);
            step = Step::Decode;
            break;
        case Step::Decode:
            decodedMesh = reader->Read();
            reader.reset();
            step = Step::BuildRecovered;
            break;
        case Step::BuildRecovered: {
            const auto merged = decodedMesh->GetGeometrySnapshot();
            const auto nodeLocal = decodedMesh->GetModelSourceGeometry();
            for (const auto &slice : recovery[nextArtifact].slices) {
                const auto &geometry = slice.draw->meshGeometryView == MeshGeometryView::NodeLocal ? nodeLocal : merged;
                if (!geometry)
                    throw std::runtime_error("The cooked mesh does not contain the frozen geometry view");
                buildSlice(slice, geometry->vertices, geometry->indices);
            }
            decodedMesh.reset();
            step = ++nextArtifact == recovery.size() ? Step::Build : Step::Open;
            break;
        }
        case Step::Build: {
            for (const auto &slice : resident)
                buildSlice(slice, *slice.draw->meshVertices, *slice.draw->meshIndices);
            std::vector<RayMeshInstance> rays;
            rays.reserve(instances.size());
            for (const auto &slice : instances) {
                const auto &draw = *slice.draw;
                rays.push_back({meshes[slice.mesh], draw.worldMatrix, draw.surfaceId, draw.layerMask});
            }
            words = EncodeRayGeometryPage(BuildRayGeometryPage(rays), RayGeometryStatus::Ready);
            step = Step::Upload;
            break;
        }
        case Step::Upload:
            std::terminate();
        }
    } catch (...) {
        failure = std::current_exception();
    }
    elapsed = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
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
    if (!header)
        header = UploadPage(m_device, EncodeRayGeometryPage({}, publication->status));
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
    m_owner.ObserveResource(m_workResource, m_active->elapsed, m_submittedOpportunity, m_workGrant);
    m_job = {};
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
            const auto prediction = resource == FrameWorkResource::Transfer ? PageSize(m_active->words) : 0;
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
            if (build->Plan())
                m_active = std::move(build);
            else
                publication->status = RayGeometryStatus::Unavailable;
            return true;
        }
        if (m_active->failure)
            std::rethrow_exception(m_active->failure);
        const auto resource = m_active->Resource();
        const auto bytes = resource == FrameWorkResource::Transfer ? PageSize(m_active->words) : 0;
        if (!m_owner.CanAdmitResource(resource, bytes, true))
            return false;
        if (resource == FrameWorkResource::Transfer) {
            publication->buffer = UploadPage(m_device, m_active->words);
            publication->status = RayGeometryStatus::Ready;
            m_owner.RecordAcceptedResource(resource, bytes);
            m_active.reset();
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
        m_job = JobSystem::Get().Schedule([build = m_active] { build->Run(); },
                                          resource == FrameWorkResource::Io ? JobDomain::IO : JobDomain::Render,
                                          JobPriority::Low, JobAffinity::WorkerOnly);
        m_workResource = resource;
        m_submittedOpportunity = m_owner.Opportunity();
        m_workGrant = m_owner.RecordAcceptedResource(resource, 0);
    } catch (const std::exception &error) {
        INXLOG_ERROR("World ray geometry preparation failed: ", error.what());
        publication->world.reset();
        publication->status = RayGeometryStatus::Unavailable;
        m_active.reset();
    }
    return true;
}
} // namespace infernux::raytracing
