#include "RayTexturePages.h"
#include "RayTexturePageRenderer.h"

#include <function/renderer/rhi/RhiCapabilities.h>

#include <algorithm>
#include <cassert>
#include <cmath>
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

constexpr uint32_t MaxMipCount(uint32_t width, uint32_t height) noexcept
{
    uint32_t count = 0;
    for (auto size = std::max(width, height); size; size >>= 1u)
        ++count;
    return count;
}

auto SourceKey(const RayTexturePageSource &source) noexcept
{
    const auto &view = source.view->GetViewDesc();
    // A source view is independent of the material sampler applied to its normalized page.
    return std::make_tuple(source.pageClass, std::string_view(source.view->GetSourceId()), source.view->GetRevision(),
                           source.contentGeneration, view.baseLayer, view.baseMip, view.mipCount, view.format);
}

bool CanSample(const RayTexturePageSource &source, const rhi::DeviceCaps &caps) noexcept
{
    const auto &allocation = source.allocation;
    const auto &view = source.view->GetViewDesc();
    const auto format = view.format == rhi::PixelFormat::Undefined ? source.view->GetFormat() : view.format;
    const auto usage = view.usage == rhi::TextureUsageFlags::None ? allocation.usage : view.usage;
    return allocation.dimension == rhi::TextureDimension::Texture2D && allocation.samples == rhi::SampleCount::One &&
           caps.CheckFormat(format, rhi::FormatFeature::Sampled | rhi::FormatFeature::FilterLinear).IsSupported() &&
           rhi::HasTextureUsage(usage, rhi::TextureUsageFlags::Sampled) && allocation.width > 0 &&
           allocation.height > 0 && allocation.mipLevels <= MaxMipCount(allocation.width, allocation.height) &&
           view.dimension == rhi::TextureViewDimension::Texture2D && view.aspect == rhi::TextureAspect::Color &&
           view.layerCount == 1 && view.baseLayer < allocation.depthOrLayers && view.mipCount > 0 &&
           view.baseMip < allocation.mipLevels && view.mipCount <= allocation.mipLevels - view.baseMip;
}

bool SetPageCosts(RayTexturePagePlan &plan, uint64_t budget) noexcept
{
    uint64_t remaining = budget;
    for (std::size_t i = 0; i < plan.pages.size(); ++i) {
        auto &page = plan.pages[i];
        if (!page)
            continue;
        page->mipLevels = MaxMipCount(page->width, page->height);
        auto layerRoom = remaining / page->depthOrLayers;
        const auto startRoom = layerRoom;
        for (uint32_t mip = 0; mip < page->mipLevels; ++mip) {
            const auto rowBytes = uint64_t{AtMip(page->width, mip)} * 4;
            const auto rows = AtMip(page->height, mip);
            if (rows > layerRoom / rowBytes)
                return false;
            layerRoom -= rowBytes * rows;
        }
        plan.pageBytes[i] = (startRoom - layerRoom) * page->depthOrLayers;
        remaining -= plan.pageBytes[i];
    }
    plan.byteSize = budget - remaining;
    return true;
}

float SourceLod(const RayTexturePageSource &source, uint32_t width, uint32_t height) noexcept
{
    const auto &view = source.view->GetViewDesc();
    const float x = static_cast<float>(AtMip(source.allocation.width, view.baseMip)) / static_cast<float>(width);
    const float y = static_cast<float>(AtMip(source.allocation.height, view.baseMip)) / static_cast<float>(height);
    return std::clamp(std::log2(std::max(x, y)), 0.0f, static_cast<float>(view.mipCount - 1));
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

RayTexturePagePreparation PrepareRayTexturePages(rhi::DeviceId device, const rhi::DeviceCaps &caps, uint64_t byteBudget,
                                                 uint64_t maxRowBytes, const std::vector<RayTexturePageSource> &sources)
{
    using Issue = RayTexturePageIssue;
    for (std::size_t i = 0; i < sources.size(); ++i) {
        const auto &source = sources[i];
        if (!source.view || !source.view->IsValid() || !source.view->GetTexture().IsValid() ||
            source.view->GetTexture().Device() != device || source.allocation.format == rhi::PixelFormat::Undefined)
            return {{}, Issue::SourceUnavailable, i};
        if (!CanSample(source, caps))
            return {{}, Issue::UnsupportedSource, i};
    }
    const auto &limits = caps.limits;
    const auto rowWidth = std::min<uint64_t>(limits.maxTextureDimension2D, maxRowBytes / 4);
    if (!sources.empty() && !rowWidth)
        return {{}, Issue::DeviceLimit, 0};

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
    for (const auto index : order) {
        const auto &source = sources[index];
        if (!plan.layers.empty() && SourceKey(plan.layers.back().source) == SourceKey(source)) {
            plan.slots[index] = plan.layers.back().slot;
            continue;
        }
        auto &page = plan.pages[PageIndex(source.pageClass)];
        if (!page) {
            constexpr auto features =
                rhi::FormatFeature::Sampled | rhi::FormatFeature::FilterLinear | rhi::FormatFeature::ColorAttachment;
            const auto format = PageFormat(source.pageClass);
            if (!caps.CheckFormat(format, features).IsSupported())
                return {{}, Issue::DeviceLimit, index};
            page.emplace();
            page->depthOrLayers = 0;
            page->format = format;
            page->usage = rhi::TextureUsageFlags::Sampled | rhi::TextureUsageFlags::ColorAttachment;
        }
        if (page->depthOrLayers == limits.maxTextureArrayLayers)
            return {{}, Issue::DeviceLimit, index};
        const auto &view = source.view->GetViewDesc();
        page->width = std::max(page->width, AtMip(source.allocation.width, view.baseMip));
        page->height = std::max(page->height, AtMip(source.allocation.height, view.baseMip));
        const RayTexturePageSlot slot{source.pageClass, page->depthOrLayers++};
        plan.slots[index] = slot;
        plan.layers.push_back({source, slot});
    }
    for (auto &page : plan.pages) {
        if (!page)
            continue;
        while (page->width > rowWidth || page->height > limits.maxTextureDimension2D) {
            page->width = AtMip(page->width, 1);
            page->height = AtMip(page->height, 1);
        }
    }
    while (!SetPageCosts(plan, byteBudget)) {
        bool shrink = false;
        for (auto &page : plan.pages) {
            if (page) {
                shrink |= page->width > 1 || page->height > 1;
                page->width = AtMip(page->width, 1);
                page->height = AtMip(page->height, 1);
            }
        }
        if (!shrink)
            return {{}, Issue::ByteBudget, 0};
    }
    return {std::move(plan)};
}

bool RayTexturePageCursor::Done() const noexcept
{
    return m_layer == m_plan.layers.size();
}

uint64_t RayTexturePageCursor::MinimumBytes() const noexcept
{
    return Done() ? 0 : uint64_t{Width()} * 4;
}

uint32_t RayTexturePageCursor::Width() const noexcept
{
    return AtMip(m_plan.pages[PageIndex(m_plan.layers[m_layer].slot.pageClass)]->width, m_mip);
}

uint32_t RayTexturePageCursor::Height() const noexcept
{
    return AtMip(m_plan.pages[PageIndex(m_plan.layers[m_layer].slot.pageClass)]->height, m_mip);
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

std::shared_ptr<RayTexturePageBatch> RayTexturePageCandidate::PrepareBatch(RayTexturePageRenderer &renderer,
                                                                           uint64_t byteAllowance, uint32_t maxSlices)
{
    if (m_failed || m_pending || m_publication || !maxSlices || byteAllowance < m_cursor.MinimumBytes())
        return {};
    auto batch = std::shared_ptr<RayTexturePageBatch>(new RayTexturePageBatch(shared_from_this(), m_acceptedBytes));
    auto cursor = m_cursor;
    for (uint32_t i = 0; i < maxSlices && !cursor.Done(); ++i) {
        const auto rowBytes = cursor.MinimumBytes();
        if (rowBytes > byteAllowance)
            break;
        const auto &layer = m_plan.layers[cursor.m_layer];
        const auto page = PageIndex(layer.slot.pageClass);
        const auto width = cursor.Width();
        const auto height = cursor.Height();
        const auto rows = static_cast<uint32_t>(std::min<uint64_t>(height - cursor.m_row, byteAllowance / rowBytes));
        const auto &destination = m_resources[page];
        auto draw = cursor.m_row ? m_continuedDraw
                                 : renderer.Prepare(layer.source.view, destination, width, height, cursor.m_mip,
                                                    layer.slot.layer, SourceLod(layer.source, width, height));
        assert(draw);
        batch->m_slices.push_back({std::move(draw), destination->GetTexture(), layer.source.view->GetUploadSubmission(),
                                   layer.slot.pageClass, cursor.m_mip, layer.slot.layer, width, height, cursor.m_row,
                                   rows});
        const auto bytes = rowBytes * rows;
        batch->m_pageBytes[page] += bytes;
        byteAllowance -= bytes;
        cursor.Advance(rows);
    }
    assert(!batch->m_slices.empty());
    return batch;
}

bool RayTexturePageCandidate::Accept(const RayTexturePageBatch &batch) noexcept
{
    const auto &receipt = *batch.m_receipt;
    if (m_failed || m_pending || m_publication || batch.m_owner.get() != this || batch.m_offset != m_acceptedBytes ||
        !batch.m_recorded || !receipt.submission.IsValid() || receipt.submission.device != m_plan.device)
        return false;
    for (const auto &slice : batch.m_slices)
        m_cursor.Advance(slice.rowCount);
    const auto &last = batch.m_slices.back();
    m_continuedDraw = last.LastStripe() ? nullptr : last.draw;
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
