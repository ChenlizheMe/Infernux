#include "RayTexturePages.h"

#include <function/renderer/rhi/RhiCapabilities.h>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <tuple>

namespace infernux::raytracing
{
namespace
{

constexpr std::size_t PageIndex(RayTexturePageClass pageClass) noexcept
{
    return static_cast<std::size_t>(pageClass);
}

constexpr rhi::PixelFormat PageFormat(RayTexturePageClass pageClass) noexcept
{
    return pageClass == RayTexturePageClass::Color ? rhi::PixelFormat::RGBA8Srgb : rhi::PixelFormat::RGBA8UNorm;
}

constexpr uint32_t AtMip(uint32_t size, uint32_t mip) noexcept
{
    return std::max(1u, size >> mip);
}

uint32_t MaxMipCount(uint32_t width, uint32_t height) noexcept
{
    uint32_t count = 0;
    for (auto extent = std::max(width, height); extent; extent >>= 1)
        ++count;
    return count;
}

auto SourceKey(const RayTexturePageSource &source) noexcept
{
    const auto &view = source.view->GetViewDesc();
    // The same exact asset can have independent resident allocations for different material samplers.
    return std::make_tuple(source.pageClass, std::string_view(source.view->GetSourceId()), source.view->GetRevision(),
                           source.contentGeneration, view.baseLayer, view.baseMip, view.mipCount);
}

bool CanCopy(const RayTexturePageSource &source) noexcept
{
    const auto &allocation = source.allocation;
    const auto &view = source.view->GetViewDesc();
    const auto format = PageFormat(source.pageClass);
    return allocation.dimension == rhi::TextureDimension::Texture2D && allocation.samples == rhi::SampleCount::One &&
           allocation.format == format && source.view->GetFormat() == format &&
           (view.format == rhi::PixelFormat::Undefined || view.format == format) &&
           rhi::HasTextureUsage(allocation.usage, rhi::TextureUsageFlags::TransferSource) && allocation.width > 0 &&
           allocation.height > 0 && allocation.mipLevels <= MaxMipCount(allocation.width, allocation.height) &&
           view.dimension == rhi::TextureViewDimension::Texture2D && view.aspect == rhi::TextureAspect::Color &&
           view.layerCount == 1 && view.baseLayer < allocation.depthOrLayers && view.mipCount > 0 &&
           view.baseMip < allocation.mipLevels && view.mipCount <= allocation.mipLevels - view.baseMip;
}

bool ReserveLayerBytes(const rhi::TextureDesc &texture, uint64_t &budget) noexcept
{
    const auto layout = rhi::GetPixelFormatLayout(texture.format);
    for (uint32_t mip = 0; mip < texture.mipLevels; ++mip) {
        const auto rowBytes = layout.RowBytes(AtMip(texture.width, mip));
        const auto rows = AtMip(texture.height, mip);
        if (rows > budget / rowBytes)
            return false;
        budget -= rowBytes * rows;
    }
    return true;
}

std::shared_ptr<rhi::TextureResource> AllocatePage(rhi::Device &device, const rhi::TextureDesc &desc, uint64_t bytes)
{
    rhi::TextureViewDesc viewDesc;
    viewDesc.dimension = rhi::TextureViewDimension::Texture2DArray;
    viewDesc.format = desc.format;
    viewDesc.mipCount = desc.mipLevels;
    viewDesc.layerCount = desc.depthOrLayers;
    viewDesc.usage = rhi::TextureUsageFlags::Sampled;
    rhi::TextureViewHandle view;
    rhi::SamplerHandle sampler;
    try {
        viewDesc.texture = device.CreateTexture(desc);
        if (!viewDesc.texture.IsValid())
            throw std::runtime_error("Ray material texture page allocation failed");
        view = device.CreateTextureView(viewDesc);
        if (!view.IsValid())
            throw std::runtime_error("Ray material texture page view allocation failed");
        rhi::SamplerDesc settings;
        settings.addressU = settings.addressV = settings.addressW = rhi::AddressMode::ClampToEdge;
        settings.minFilter = settings.magFilter = settings.mipFilter = rhi::FilterMode::Linear;
        settings.maxLod = static_cast<float>(desc.mipLevels - 1);
        sampler = device.CreateSampler(settings);
        if (!sampler.IsValid())
            throw std::runtime_error("Ray material texture page sampler allocation failed");
        return std::make_shared<rhi::TextureResource>(device, viewDesc.texture, view, sampler, bytes, desc.format,
                                                      viewDesc);
    } catch (...) {
        device.Release(sampler);
        device.Release(view);
        device.Release(viewDesc.texture);
        throw;
    }
}

} // namespace

RayTexturePagePreparation PrepareRayTexturePages(rhi::DeviceId device, const rhi::DeviceLimits &limits,
                                                 uint64_t byteBudget, const std::vector<RayTexturePageSource> &sources)
{
    using Issue = RayTexturePageIssue;
    for (std::size_t i = 0; i < sources.size(); ++i) {
        const auto &source = sources[i];
        if (!source.view || !source.view->IsValid() || !source.view->GetTexture().IsValid() ||
            source.view->GetTexture().Device() != device || source.allocation.format == rhi::PixelFormat::Undefined)
            return {{}, Issue::SourceUnavailable, i};
    }

    std::vector<std::size_t> order(sources.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        const auto leftKey = SourceKey(sources[left]);
        const auto rightKey = SourceKey(sources[right]);
        if (leftKey != rightKey)
            return leftKey < rightKey;
        const auto a = sources[left].view->GetTexture();
        const auto b = sources[right].view->GetTexture();
        return std::tie(a.index, a.generation) < std::tie(b.index, b.generation);
    });

    RayTexturePagePlan plan;
    plan.device = device;
    plan.slots.resize(sources.size());
    uint64_t remaining = byteBudget;
    for (const auto index : order) {
        const auto &source = sources[index];
        if (!CanCopy(source))
            return {{}, Issue::NeedsConversion, index};
        const auto &view = source.view->GetViewDesc();
        const auto width = AtMip(source.allocation.width, view.baseMip);
        const auto height = AtMip(source.allocation.height, view.baseMip);
        auto &page = plan.pages[PageIndex(source.pageClass)];
        if (!page) {
            if (width > limits.maxTextureDimension2D || height > limits.maxTextureDimension2D)
                return {{}, Issue::DeviceLimit, index};
            page.emplace();
            page->width = width;
            page->height = height;
            page->depthOrLayers = 0;
            page->mipLevels = view.mipCount;
            page->format = PageFormat(source.pageClass);
            page->usage = rhi::TextureUsageFlags::Sampled | rhi::TextureUsageFlags::TransferDestination;
        }
        if (width != page->width || height != page->height || view.mipCount != page->mipLevels)
            return {{}, Issue::NeedsConversion, index};

        if (!plan.layers.empty() && SourceKey(plan.layers.back().source) == SourceKey(source)) {
            plan.slots[index] = plan.layers.back().slot;
            continue;
        }
        if (page->depthOrLayers == limits.maxTextureArrayLayers)
            return {{}, Issue::DeviceLimit, index};
        const uint64_t before = remaining;
        if (!ReserveLayerBytes(*page, remaining))
            return {{}, Issue::ByteBudget, index};
        plan.pageBytes[PageIndex(source.pageClass)] += before - remaining;
        const RayTexturePageSlot slot{source.pageClass, page->depthOrLayers++};
        plan.slots[index] = slot;
        plan.layers.push_back({source, slot});
    }
    plan.byteSize = byteBudget - remaining;
    return {std::move(plan)};
}

bool RayTexturePageCursor::Done() const noexcept
{
    return m_layer == m_plan.layers.size();
}

uint64_t RayTexturePageCursor::MinimumBytes() const noexcept
{
    if (Done())
        return 0;
    const auto &page = *m_plan.pages[PageIndex(m_plan.layers[m_layer].slot.pageClass)];
    return rhi::GetPixelFormatLayout(page.format).RowBytes(AtMip(page.width, m_mip));
}

std::optional<RayTexturePageCopy> RayTexturePageCursor::Peek(uint64_t byteAllowance) const noexcept
{
    const auto rowBytes = MinimumBytes();
    if (rowBytes == 0 || byteAllowance < rowBytes)
        return std::nullopt;
    const auto &layer = m_plan.layers[m_layer];
    const auto &view = layer.source.view->GetViewDesc();
    const auto &page = *m_plan.pages[PageIndex(layer.slot.pageClass)];
    const auto rows =
        static_cast<uint32_t>(std::min<uint64_t>(AtMip(page.height, m_mip) - m_row, byteAllowance / rowBytes));
    rhi::TextureCopyRegion region;
    region.sourceMip = view.baseMip + m_mip;
    region.sourceLayer = view.baseLayer;
    region.destinationMip = m_mip;
    region.destinationLayer = layer.slot.layer;
    region.width = AtMip(page.width, m_mip);
    region.height = rows;
    region.sourceY = region.destinationY = m_row;
    return RayTexturePageCopy{m_layer, region, rowBytes * rows};
}

void RayTexturePageCursor::Advance(uint32_t rows) noexcept
{
    assert(!Done() && rows != 0);
    const auto &page = *m_plan.pages[PageIndex(m_plan.layers[m_layer].slot.pageClass)];
    const auto height = AtMip(page.height, m_mip);
    assert(rows <= height - m_row);
    m_row += rows;
    if (m_row == height) {
        m_row = 0;
        ++m_mip;
        if (m_mip == page.mipLevels) {
            m_mip = 0;
            ++m_layer;
        }
    }
}

RayTexturePageBatch::RayTexturePageBatch(std::shared_ptr<RayTexturePageCandidate> owner, uint64_t offset)
    : m_owner(std::move(owner)), m_receipt(std::make_shared<rhi::WorkCompletionReceipt>()), m_offset(offset)
{
}

bool RayTexturePageBatch::Record(const rhi::TransferCommandEncoder &encoder)
{
    if (!encoder.HasCompleteDispatch())
        return false;
    for (const auto &copy : m_transfers)
        encoder.CopyTexture(copy.source, copy.destination, copy.region);
    m_recorded = true;
    return true;
}

std::shared_ptr<RayTexturePageCandidate> RayTexturePageCandidate::Create(rhi::Device &device, RayTexturePagePlan plan,
                                                                         std::string sourceId, uint64_t revision)
{
    if (plan.device != device.GetDeviceId() || sourceId.empty() || revision == 0)
        throw std::invalid_argument("Ray material texture candidate requires its device and a revisioned source");
    auto candidate = std::shared_ptr<RayTexturePageCandidate>(
        new RayTexturePageCandidate(device, std::move(plan), std::move(sourceId), revision));
    if (candidate->m_cursor.Done())
        candidate->Publish();
    return candidate;
}

RayTexturePageCandidate::RayTexturePageCandidate(rhi::Device &device, RayTexturePagePlan plan, std::string sourceId,
                                                 uint64_t revision)
    : m_plan(std::move(plan)), m_cursor(m_plan), m_sourceId(std::move(sourceId)), m_revision(revision)
{
    for (std::size_t i = 0; i < m_plan.pages.size(); ++i) {
        if (m_plan.pages[i])
            m_resources[i] = AllocatePage(device, *m_plan.pages[i], m_plan.pageBytes[i]);
    }
}

std::shared_ptr<RayTexturePageBatch> RayTexturePageCandidate::PrepareBatch(uint64_t byteAllowance, uint32_t maxCopies)
{
    if (m_failed || m_pending || m_publication || !maxCopies || byteAllowance < m_cursor.MinimumBytes())
        return {};
    auto batch = std::shared_ptr<RayTexturePageBatch>(new RayTexturePageBatch(shared_from_this(), m_acceptedBytes));
    auto cursor = m_cursor;
    for (uint32_t i = 0; i < maxCopies; ++i) {
        const auto part = cursor.Peek(byteAllowance);
        if (!part)
            break;
        const auto &layer = m_plan.layers[part->layerIndex];
        const auto page = PageIndex(layer.slot.pageClass);
        const auto &source = *layer.source.view;
        batch->m_transfers.push_back(
            {source.GetTexture(), m_resources[page]->GetTexture(), part->region, source.GetUploadSubmission()});
        batch->m_pageBytes[page] += part->bytes;
        byteAllowance -= part->bytes;
        cursor.Advance(part->region.height);
    }
    assert(!batch->m_transfers.empty());
    return batch;
}

bool RayTexturePageCandidate::Accept(const RayTexturePageBatch &batch) noexcept
{
    const auto &receipt = *batch.m_receipt;
    if (m_failed || m_pending || m_publication || batch.m_owner.get() != this || batch.m_offset != m_acceptedBytes ||
        !batch.m_recorded || !receipt.submission.IsValid() || receipt.submission.device != m_plan.device)
        return false;
    for (const auto &copy : batch.m_transfers)
        m_cursor.Advance(copy.region.height);
    m_acceptedBytes += batch.ByteSize();
    m_pending = batch.m_receipt;
    return true;
}

bool RayTexturePageCandidate::Observe()
{
    if (!m_pending || (!m_pending->complete && !m_pending->failed))
        return false;
    m_failed |= m_pending->failed;
    m_completion = m_pending->submission;
    if (!m_failed && m_cursor.Done())
        Publish();
    m_pending.reset();
    return true;
}

void RayTexturePageCandidate::Fail() noexcept
{
    assert(!m_publication);
    m_failed = true;
}

void RayTexturePageCandidate::Publish()
{
    static constexpr const char *suffix[] = {"/color", "/data"};
    auto publication = std::make_shared<RayTexturePagePublication>();
    publication->revision = m_revision;
    publication->completion = m_completion;
    publication->pageBytes = m_plan.pageBytes;
    for (std::size_t i = 0; i < m_resources.size(); ++i) {
        if (m_resources[i]) {
            m_resources[i]->SetUploadSubmission(m_completion);
            publication->pages[i] =
                std::make_shared<const rhi::TextureGpuView>(m_sourceId + suffix[i], m_revision, m_resources[i]);
        }
    }
    publication->slots = std::move(m_plan.slots);
    m_plan.layers.clear();
    m_publication = std::move(publication);
}

} // namespace infernux::raytracing
