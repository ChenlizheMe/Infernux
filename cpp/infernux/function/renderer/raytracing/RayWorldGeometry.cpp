#include "RayWorldGeometry.h"

#include "RayGeometry.h"

#include <core/error/InxError.h>
#include <function/renderer/RenderWorld.h>
#include <function/renderer/rhi/RhiBuffer.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <stdexcept>

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

uint64_t EstimateBuildStorage(const DrawCallResult &geometry)
{
    // Reserve for the source-index remap, BVH build workspace and packed output.
    // Count repeated slices conservatively until the builder deduplicates them.
    uint64_t bytes = 2 * sizeof(RayGeometryWord);
    for (const auto &draw : geometry.drawCalls) {
        const uint64_t drawBytes = 512 + uint64_t{draw.indexCount} * 256;
        if (bytes > (std::numeric_limits<uint64_t>::max)() - drawBytes)
            return (std::numeric_limits<uint64_t>::max)();
        bytes += drawBytes;
    }
    return bytes;
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
        : revision(GetRevision(*frame)), workspace(EstimateBuildStorage(frame->DrawCalls())), world(std::move(frame))
    {
    }

    const WorldRevision revision;
    const uint64_t workspace;
    std::shared_ptr<const RenderWorldFrame> world;
    std::shared_ptr<const rhi::BufferResource> buffer;
    RayGeometryStatus status = RayGeometryStatus::Pending;
};

class RayWorldGeometryBuild final
{
  public:
    std::weak_ptr<RayWorldGeometryPublication> publication;
    std::shared_ptr<const RenderWorldFrame> world;
    std::shared_ptr<FrameCpuStorageClaim> storage;
    std::vector<RayGeometryWord> words;
    uint64_t elapsed = 0;
};

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
    m_owner.ObserveResource(FrameWorkResource::Cpu, m_active->elapsed, m_submittedOpportunity, m_cpuGrant);
    m_job = {};
    m_active->world.reset();
    const auto publication = m_active->publication.lock();
    if (!publication || m_active->words.empty() ||
        m_active->words[1][1] == static_cast<uint32_t>(RayGeometryStatus::Unavailable)) {
        if (publication)
            publication->status = RayGeometryStatus::Unavailable;
        m_active.reset();
    }
}

FrameWorkFacts RayWorldGeometry::Describe()
{
    FrameWorkFacts facts;
    if (m_active) {
        facts.pending = true;
        facts.workerBusy = m_job.IsValid();
        facts.host.ready = !facts.workerBusy &&
                           (m_active->publication.expired() ||
                            m_owner.CanAdmitResource(FrameWorkResource::Transfer, PageSize(m_active->words), true));
    } else if (const auto next = FindPending()) {
        facts.pending = true;
        facts.host.ready =
            next->workspace > m_owner.CpuStorageLimit() || m_owner.CanAdmitResource(FrameWorkResource::Cpu, 0);
    }
    return facts;
}

bool RayWorldGeometry::Advance()
{
    if (m_active) {
        if (m_job.IsValid())
            return false;
        const auto publication = m_active->publication.lock();
        if (!publication) {
            m_active.reset();
            return true;
        }
        const auto bytes = PageSize(m_active->words);
        if (!m_owner.CanAdmitResource(FrameWorkResource::Transfer, bytes, true))
            return false;
        try {
            publication->buffer = UploadPage(m_device, m_active->words);
            publication->status = RayGeometryStatus::Ready;
            m_owner.RecordAcceptedResource(FrameWorkResource::Transfer, bytes);
        } catch (const std::exception &error) {
            INXLOG_ERROR("World ray geometry upload failed: ", error.what());
            publication->status = RayGeometryStatus::Unavailable;
        }
        m_active.reset();
        return true;
    }
    auto next = FindPending();
    if (!next)
        return false;
    if (next->workspace > m_owner.CpuStorageLimit()) {
        INXLOG_ERROR("World ray geometry build exceeds the CPU storage limit");
        next->world.reset();
        next->status = RayGeometryStatus::Unavailable;
        return true;
    }
    if (!m_owner.CanAdmitResource(FrameWorkResource::Cpu, 0))
        return false;
    auto storage = m_owner.TryReserveCpuStorage(next->workspace);
    if (!storage)
        return false;
    auto build = std::make_shared<RayWorldGeometryBuild>();
    build->publication = next;
    build->world = next->world;
    build->storage = std::move(storage);
    m_job = JobSystem::Get().Schedule(
        [build] {
            const auto start = std::chrono::steady_clock::now();
            try {
                build->words = BuildWorldRayGeometry(build->world->DrawCalls());
            } catch (const std::exception &error) {
                INXLOG_ERROR("World ray geometry build failed: ", error.what());
            }
            build->elapsed = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
        },
        JobDomain::Render, JobPriority::Low, JobAffinity::WorkerOnly);
    next->world.reset();
    m_active = std::move(build);
    m_submittedOpportunity = m_owner.Opportunity();
    m_cpuGrant = m_owner.RecordAcceptedResource(FrameWorkResource::Cpu, 0);
    return true;
}
} // namespace infernux::raytracing
