#include "RayWorldMaterials.h"

#include <core/error/InxError.h>
#include <function/renderer/RenderWorld.h>
#include <function/renderer/rhi/RhiBuffer.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <stdexcept>
#include <string>

namespace infernux::raytracing
{
namespace
{

constexpr uint64_t TransferWindow = 256 * 1024;
constexpr uint32_t CopyCommandLimit = 64;
using MaterialRevision = std::array<uint64_t, 5>;

MaterialRevision RevisionOf(const RenderWorldFrame &world, uint64_t textureContentGeneration) noexcept
{
    return {world.WorldId(), world.SimulationEpoch(), world.StructuralRevision(), world.ContentRevision(),
            textureContentGeneration};
}

std::shared_ptr<const rhi::BufferResource> AllocateTable(rhi::Device &device, uint64_t bytes)
{
    if (!device.GetCapabilities().IsStorageBufferRangeValid(0, bytes))
        throw std::length_error("Ray material table exceeds the device storage range");
    rhi::BufferDesc desc;
    desc.byteSize = bytes;
    desc.usage = rhi::BufferUsageFlags::Storage;
    desc.memory = rhi::BufferMemory::Upload;
    const auto buffer = device.CreateBuffer(desc);
    if (!buffer.IsValid())
        throw std::runtime_error("Could not allocate a ray material table");
    return std::make_shared<rhi::BufferResource>(device, buffer, bytes);
}

const char *PageIssue(RayTexturePageIssue issue) noexcept
{
    switch (issue) {
    case RayTexturePageIssue::SourceUnavailable:
        return "has no resident allocation";
    case RayTexturePageIssue::NeedsConversion:
        return "needs texture page format, size, or mip conversion";
    case RayTexturePageIssue::DeviceLimit:
        return "exceeds the texture page device limits";
    case RayTexturePageIssue::ByteBudget:
        return "exceeds the texture page storage budget";
    case RayTexturePageIssue::None:
        return "has no texture page plan";
    }
    return "cannot be placed in a ray texture page";
}

} // namespace

rhi::TextureUploadRequest RayPlaceholderTextureUpload(RayTexturePageClass pageClass) noexcept
{
    static constexpr std::array<uint8_t, 4> texel{0, 0, 0, 0};
    static constexpr rhi::TextureSubresourceUpload subresource{texel.data(), texel.size(), 0, 0, 1, 1, 1, 1, 4, 4};
    rhi::TextureUploadRequest request;
    request.texture.format =
        pageClass == RayTexturePageClass::Color ? rhi::PixelFormat::RGBA8Srgb : rhi::PixelFormat::RGBA8UNorm;
    request.texture.usage = rhi::TextureUsageFlags::Sampled | rhi::TextureUsageFlags::TransferDestination;
    request.view.dimension = rhi::TextureViewDimension::Texture2DArray;
    request.view.format = request.texture.format;
    request.view.usage = rhi::TextureUsageFlags::Sampled;
    request.sampler.addressU = request.sampler.addressV = request.sampler.addressW = rhi::AddressMode::ClampToEdge;
    request.sampler.minFilter = request.sampler.magFilter = request.sampler.mipFilter = rhi::FilterMode::Linear;
    request.subresources = &subresource;
    request.subresourceCount = 1;
    return request;
}

class RayWorldMaterialRequest final
{
  public:
    RayWorldMaterialRequest(std::shared_ptr<const RenderWorldFrame> frame,
                            std::shared_ptr<const RayLitMaterialLayout> materialLayout, MaterialRevision key)
        : revision(key), layout(std::move(materialLayout)), world(std::move(frame))
    {
    }

    const MaterialRevision revision;
    const std::shared_ptr<const RayLitMaterialLayout> layout;
    std::shared_ptr<const RenderWorldFrame> world;
    std::shared_ptr<const RayWorldMaterialPublication> value;
};

class RayWorldMaterialBuild final
{
  public:
    explicit RayWorldMaterialBuild(const std::shared_ptr<RayWorldMaterialRequest> &value) : request(value)
    {
    }

    struct Table
    {
        std::vector<std::byte> bytes;
        std::shared_ptr<const rhi::BufferResource> buffer;
    };

    [[nodiscard]] bool UploadPending() const noexcept
    {
        return candidate && !candidate->Failed() && nextTable < tables.size();
    }

    [[nodiscard]] uint64_t UploadBytes(uint64_t allowance) const noexcept
    {
        const uint64_t window = (std::min)(allowance, TransferWindow);
        return (std::min)(uint64_t{tables[nextTable].bytes.size()} - tableOffset, window & ~uint64_t{3});
    }

    void Upload(rhi::Device &device, uint64_t bytes)
    {
        auto &table = tables[nextTable];
        if (!table.buffer)
            table.buffer = AllocateTable(device, table.bytes.size());
        if (!device.WriteBuffer(table.buffer->GetBuffer(), tableOffset, table.bytes.data() + tableOffset, bytes))
            throw std::runtime_error("Could not upload a ray material table");
        tableOffset += bytes;
        if (tableOffset == table.bytes.size()) {
            storage->ReduceTo(storage->Bytes() - table.bytes.size());
            std::vector<std::byte>{}.swap(table.bytes);
            tableOffset = 0;
            ++nextTable;
        }
    }

    std::weak_ptr<RayWorldMaterialRequest> request;
    std::optional<RayTexturePagePlan> plan;
    std::shared_ptr<RayTexturePageCandidate> candidate;
    std::array<Table, 2> tables;
    std::shared_ptr<FrameCpuStorageClaim> storage;
    std::shared_ptr<rhi::WorkCompletionReceipt> receipt;
    uint64_t atlasBytes = 0;
    uint64_t copyBytes = 0;
    uint64_t copyOpportunity = 0;
    uint64_t retryOpportunity = 0;
    uint64_t tableOffset = 0;
    std::size_t nextTable = 0;
};

RayWorldMaterials::RayWorldMaterials(FrameWorkOwner &owner, rhi::Device &device, ResolveRayLitTexture resolver,
                                     ResolveRayPlaceholderTexture placeholders, uint64_t atlasByteBudget)
    : m_owner(owner), m_device(device), m_resolve(std::move(resolver)), m_placeholder(std::move(placeholders)),
      m_atlasByteBudget(atlasByteBudget)
{
    FrameWorkSource source;
    source.observe = [this] { Observe(); };
    source.refresh = source.observe;
    source.describe = [this] { return Describe(); };
    source.advance = [this](bool, uint64_t) { return Advance(); };
    m_source = owner.AddSource(std::move(source));
}

RayWorldMaterials::~RayWorldMaterials()
{
    m_owner.RemoveSource(m_source);
}

RayWorldMaterialBinding RayWorldMaterials::Acquire(const std::shared_ptr<const RenderWorldFrame> &world,
                                                   const std::shared_ptr<const RayLitMaterialLayout> &layout,
                                                   uint64_t textureContentGeneration,
                                                   std::shared_ptr<const RayWorldMaterialRequest> &request)
{
    assert(world && layout);
    const auto key = RevisionOf(*world, textureContentGeneration);
    if (!request || request->revision != key || request->layout != layout) {
        request.reset();
        m_requests.erase(
            std::remove_if(m_requests.begin(), m_requests.end(), [](const auto &entry) { return entry.expired(); }),
            m_requests.end());
        for (const auto &entry : m_requests) {
            if (auto item = entry.lock(); item && item->revision == key && item->layout == layout) {
                request = std::move(item);
                break;
            }
        }
        if (!request) {
            auto item = std::make_shared<RayWorldMaterialRequest>(world, layout, key);
            m_requests.emplace_back(item);
            m_waiting.emplace_back(item);
            request = std::move(item);
            m_owner.Wake(m_source);
        }
    }
    return {request->value, static_cast<bool>(request->world)};
}

std::shared_ptr<RayTexturePageBatch> RayWorldMaterials::PrepareCopyBatch()
{
    constexpr auto resource = FrameWorkResource::Transfer;
    if (!m_active || !m_active->candidate || m_active->UploadPending() || m_active->request.expired())
        return {};
    const auto bytes = (std::min)(m_owner.ResourceRemaining(resource), TransferWindow);
    if (!m_owner.CanAdmitResource(resource, bytes))
        return {};
    return m_active->candidate->PrepareBatch(bytes, CopyCommandLimit);
}

bool RayWorldMaterials::AcceptCopyBatch(const RayTexturePageBatch &batch) noexcept
{
    if (!m_active || !m_active->candidate || !m_active->candidate->Accept(batch))
        return false;
    m_active->receipt = batch.Receipt();
    m_active->copyOpportunity = m_owner.Opportunity();
    m_active->copyBytes = m_owner.RecordAcceptedResource(FrameWorkResource::Transfer, batch.ByteSize());
    return true;
}

void RayWorldMaterials::FailCopyBatch() noexcept
{
    assert(m_active && m_active->candidate && !m_active->receipt);
    m_active->candidate->Fail();
    m_owner.ReportProgress();
}

void RayWorldMaterials::Observe()
{
    if (!m_active)
        return;
    auto &build = *m_active;
    if (build.receipt) {
        if (!build.candidate->Observe())
            return;
        if (!build.receipt->failed)
            m_owner.ObserveResource(FrameWorkResource::Transfer, build.copyBytes, build.copyOpportunity,
                                    build.copyBytes);
        build.receipt.reset();
    }
    if (build.request.expired()) {
        m_active.reset();
        m_owner.ReportProgress();
    }
}

FrameWorkFacts RayWorldMaterials::Describe()
{
    while (!m_waiting.empty() && m_waiting.front().expired())
        m_waiting.pop_front();

    FrameWorkFacts facts;
    facts.pending = m_active || !m_waiting.empty();
    if (!facts.pending)
        return facts;
    if (m_active && m_active->UploadPending()) {
        constexpr auto resource = FrameWorkResource::Transfer;
        const auto bytes = m_active->UploadBytes(m_owner.ResourceRemaining(resource));
        facts.host.ready = bytes && m_owner.CanAdmitResource(resource, bytes);
    } else if (!m_active ||
               ((!m_active->candidate || m_active->candidate->Publication() || m_active->candidate->Failed()) &&
                m_active->retryOpportunity != m_owner.Opportunity())) {
        facts.host.ready = m_owner.CanAdmitResource(FrameWorkResource::Cpu, 0);
    }
    return facts;
}

bool RayWorldMaterials::Advance()
{
    if (!m_active && m_waiting.empty())
        return false;
    auto request = m_active ? m_active->request.lock() : m_waiting.front().lock();
    if (!request)
        return false;
    if (!m_active) {
        m_active = std::make_unique<RayWorldMaterialBuild>(request);
        m_waiting.pop_front();
    }
    auto &build = *m_active;
    try {
        if (build.UploadPending()) {
            constexpr auto resource = FrameWorkResource::Transfer;
            const auto bytes = build.UploadBytes(m_owner.ResourceRemaining(resource));
            if (!bytes || !m_owner.CanAdmitResource(resource, bytes))
                return false;
            build.Upload(m_device, bytes);
            const auto grant = m_owner.RecordAcceptedResource(resource, bytes);
            m_owner.ObserveResource(resource, bytes, m_owner.Opportunity(), grant);
            return true;
        }
        constexpr auto resource = FrameWorkResource::Cpu;
        if (!m_owner.CanAdmitResource(resource, 0))
            return false;
        const auto grant = m_owner.RecordAcceptedResource(resource, 0);
        const auto started = std::chrono::steady_clock::now();
        const bool progressed = AdvanceCpu(request);
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started);
        m_owner.ObserveResource(resource, static_cast<uint64_t>(elapsed.count()), m_owner.Opportunity(), grant);
        if (!progressed)
            build.retryOpportunity = m_owner.Opportunity();
        return progressed;
    } catch (const std::exception &error) {
        INXLOG_ERROR("World ray material preparation failed: ", error.what());
        request->world.reset();
        m_active.reset();
        return true;
    }
}

bool RayWorldMaterials::AdvanceCpu(const std::shared_ptr<RayWorldMaterialRequest> &request)
{
    auto &build = *m_active;
    if (!build.candidate) {
        if (!build.plan) {
            auto input = request->layout->Capture(request->world->DrawCalls(), m_resolve);
            if (!input)
                return false;
            auto prepared = PrepareRayTexturePages(m_device.GetDeviceId(), m_device.GetCapabilities().limits,
                                                   m_atlasByteBudget, input->sources);
            if (!prepared.plan)
                throw std::invalid_argument("Ray Lit texture source " + std::to_string(prepared.sourceIndex) + " " +
                                            PageIssue(prepared.issue));
            auto descriptors = request->layout->PackDescriptors(*input, prepared.plan->slots);
            if (descriptors.empty())
                descriptors.resize(2 * sizeof(uint32_t)); // The empty scene still binds one complete descriptor.
            build.tables[0].bytes = std::move(input->materials);
            build.tables[1].bytes = std::move(descriptors);
            build.storage =
                m_owner.TrackOwnedCpuStorage(uint64_t{build.tables[0].bytes.size()} + build.tables[1].bytes.size());
            build.plan = std::move(prepared.plan);
        }
        if (build.plan->byteSize > AvailableAtlasBytes())
            return false;
        for (std::size_t index = 0; index < m_emptyPages.size(); ++index) {
            if (!build.plan->pages[index] && !m_emptyPages[index]) {
                m_emptyPages[index] = m_placeholder(static_cast<RayTexturePageClass>(index));
                if (!m_emptyPages[index])
                    return false;
            }
        }
        build.atlasBytes = build.plan->byteSize;
        build.candidate = RayTexturePageCandidate::Create(
            m_device, std::move(*build.plan), "world/ray_materials/" + std::to_string(request->world->WorldId()),
            ++m_revision);
        build.plan.reset();
        return true;
    }
    if (build.candidate->Failed())
        throw std::runtime_error("Ray texture page submission failed");
    const auto &textures = build.candidate->Publication();
    if (!textures)
        return false;

    auto value = std::make_shared<RayWorldMaterialPublication>();
    value->materials = std::move(build.tables[0].buffer);
    value->descriptors = std::move(build.tables[1].buffer);
    value->atlasBytes = build.atlasBytes;
    for (std::size_t index = 0; index < value->pages.size(); ++index)
        value->pages[index] = textures->pages[index] ? textures->pages[index] : m_emptyPages[index];
    m_published.emplace_back(value);
    request->value = std::move(value);
    request->world.reset();
    m_active.reset();
    return true;
}

uint64_t RayWorldMaterials::AvailableAtlasBytes()
{
    uint64_t available = m_atlasByteBudget;
    m_published.erase(std::remove_if(m_published.begin(), m_published.end(),
                                     [&](const auto &entry) {
                                         const auto value = entry.lock();
                                         if (!value)
                                             return true;
                                         available -= (std::min)(available, value->atlasBytes);
                                         return false;
                                     }),
                      m_published.end());
    return available;
}

} // namespace infernux::raytracing
