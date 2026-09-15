#include "RhiRenderTexture.h"
#include <cassert>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace infernux::rhi
{
namespace
{
uint32_t TexelBytes(PixelFormat format)
{
    switch (format) {
    case PixelFormat::R8UNorm:
        return 1;
    case PixelFormat::RG8UNorm:
    case PixelFormat::R16SFloat:
    case PixelFormat::RGBA4UNormPack16:
        return 2;
    case PixelFormat::RGBA8UNorm:
    case PixelFormat::RGBA8Srgb:
    case PixelFormat::BGRA8UNorm:
    case PixelFormat::BGRA8Srgb:
    case PixelFormat::RG16SFloat:
    case PixelFormat::R32SFloat:
    case PixelFormat::RGB10A2UNorm:
    case PixelFormat::D32SFloat:
    case PixelFormat::D24UNormS8UInt:
        return 4;
    case PixelFormat::RGBA16SFloat:
    case PixelFormat::RGBA16UNorm:
    case PixelFormat::RG32UInt:
        return 8;
    case PixelFormat::RGBA32SFloat:
        return 16;
    default:
        throw std::invalid_argument("RenderTexture requires an uncompressed attachment format");
    }
}

uint32_t RelativeExtent(uint32_t reference, float scale)
{
    if (reference == 0 || !std::isfinite(scale) || scale <= 0)
        throw std::invalid_argument("Relative RenderTexture requires positive reference dimensions and finite scales");
    const double value = std::ceil(static_cast<double>(reference) * scale);
    if (value > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("RenderTexture dimensions overflow");
    return static_cast<uint32_t>(value);
}

bool SameDescription(const RenderTextureDesc &a, const RenderTextureDesc &b)
{
    return a.sizeMode == b.sizeMode && a.width == b.width && a.height == b.height && a.widthScale == b.widthScale &&
           a.heightScale == b.heightScale && a.colorFormat == b.colorFormat && a.depthFormat == b.depthFormat &&
           a.samples == b.samples && a.filter == b.filter && a.storage == b.storage && a.sampledDepth == b.sampledDepth;
}

void RequireFormat(const DeviceCaps &caps, PixelFormat format, FormatFeature features, SampleCount samples)
{
    if (!caps.CheckFormat(format, features).IsSupported())
        throw std::invalid_argument(
            "RenderTexture format does not support the requested attachment/sampling/storage use");
    if (!caps.CheckSampleCount(format, samples).IsSupported())
        throw std::invalid_argument("RenderTexture format does not support the requested sample count");
}

std::shared_ptr<TextureResource> CreateAttachment(Device &device, const TextureDesc &desc, FilterMode filter)
{
    TextureHandle texture;
    TextureViewHandle view;
    SamplerHandle sampler;
    try {
        texture = device.CreateTexture(desc);
        if (!texture.IsValid())
            throw std::runtime_error("RenderTexture image allocation failed");
        TextureViewDesc viewDesc;
        viewDesc.texture = texture;
        viewDesc.format = desc.format;
        viewDesc.aspect = IsStencilFormat(desc.format) ? TextureAspect::DepthStencil
                          : IsDepthFormat(desc.format) ? TextureAspect::Depth
                                                       : TextureAspect::Color;
        view = device.CreateTextureView(viewDesc);
        if (!view.IsValid())
            throw std::runtime_error("RenderTexture image view allocation failed");
        // Multisampled texelFetch reads sample indices, not filtered UVs.
        if (desc.samples == SampleCount::One && HasTextureUsage(desc.usage, TextureUsageFlags::Sampled) &&
            !IsDepthFormat(desc.format)) {
            SamplerDesc samplerDesc;
            samplerDesc.minFilter = samplerDesc.magFilter = samplerDesc.mipFilter = filter;
            samplerDesc.addressU = samplerDesc.addressV = samplerDesc.addressW = AddressMode::ClampToEdge;
            sampler = device.CreateSampler(samplerDesc);
            if (!sampler.IsValid())
                throw std::runtime_error("RenderTexture sampler allocation failed");
        }
        const uint64_t bytes =
            uint64_t{desc.width} * desc.height * TexelBytes(desc.format) * static_cast<uint8_t>(desc.samples);
        return std::make_shared<TextureResource>(device, texture, view, sampler, bytes, desc.format, viewDesc);
    } catch (...) {
        // Partial construction owns exactly these handles; no alternate format
        // or old-content execution path is attempted after allocation failure.
        device.Release(sampler);
        device.Release(view);
        device.Release(texture);
        throw;
    }
}
} // namespace

void ValidateRenderTextureDescription(const RenderTextureDesc &description)
{
    if (description.sizeMode == RenderTextureSizeMode::Absolute) {
        if (description.width == 0 || description.height == 0)
            throw std::invalid_argument("RenderTexture pixel dimensions must be positive");
    } else if (description.sizeMode == RenderTextureSizeMode::Relative) {
        if (!std::isfinite(description.widthScale) || !std::isfinite(description.heightScale) ||
            description.widthScale <= 0 || description.heightScale <= 0)
            throw std::invalid_argument("RenderTexture scales must be positive finite numbers");
    } else {
        throw std::invalid_argument("Invalid RenderTexture size mode");
    }
    const auto samples = static_cast<unsigned>(description.samples);
    if (samples != 1 && samples != 2 && samples != 4 && samples != 8)
        throw std::invalid_argument("RenderTexture samples must be 1, 2, 4, or 8");
    if (IsDepthFormat(description.colorFormat) ||
        (description.depthFormat != PixelFormat::Undefined && !IsDepthFormat(description.depthFormat)))
        throw std::invalid_argument("RenderTexture color/depth formats have incompatible attachment roles");
    if (description.filter != FilterMode::Linear && description.filter != FilterMode::Nearest)
        throw std::invalid_argument("Invalid RenderTexture filter");
    TexelBytes(description.colorFormat);
    if (description.depthFormat != PixelFormat::Undefined)
        TexelBytes(description.depthFormat);
    if (description.sampledDepth && description.depthFormat == PixelFormat::Undefined)
        throw std::invalid_argument("Sampled depth requires a depth attachment format");
}

RenderTexture::RenderTexture(Device &device, std::string sourceId, const RenderTextureDesc &description,
                             uint32_t referenceWidth, uint32_t referenceHeight, std::string assetGuid)
    : m_device(&device), m_lifetime(device.GetLifetime()), m_sourceId(std::move(sourceId)),
      m_assetGuid(std::move(assetGuid))
{
    if (m_sourceId.empty())
        throw std::invalid_argument("RenderTexture requires a stable resource identity");
    m_sampledColorSlot = std::make_shared<TextureGpuViewSlot>(m_sourceId);
    Reconfigure(description, referenceWidth, referenceHeight);
}

std::shared_ptr<const RenderTextureGeneration> RenderTexture::Acquire() const noexcept
{
    return std::atomic_load_explicit(&m_current, std::memory_order_acquire);
}

void RenderTexture::RetainDepthAttachment()
{
    if (!Acquire()->depth)
        throw std::invalid_argument("Camera requires a RenderTexture depth attachment");
    ++m_depthConsumers;
}

void RenderTexture::ReleaseDepthAttachment() noexcept
{
    assert(m_depthConsumers > 0);
    --m_depthConsumers;
}

bool RenderTexture::Reconfigure(const RenderTextureDesc &description, uint32_t referenceWidth, uint32_t referenceHeight)
{
    auto next = PrepareGeneration(description, referenceWidth, referenceHeight);
    if (next == Acquire())
        return false;
    PublishGeneration(std::move(next));
    return true;
}

void RenderTexture::ReconfigureReferenceSize(const std::vector<std::shared_ptr<RenderTexture>> &targets,
                                             uint32_t referenceWidth, uint32_t referenceHeight)
{
    std::vector<std::shared_ptr<const RenderTextureGeneration>> next;
    next.reserve(targets.size());
    for (const auto &target : targets)
        next.push_back(target->PrepareGeneration(target->Acquire()->description, referenceWidth, referenceHeight));
    for (size_t i = 0; i < targets.size(); ++i) {
        if (next[i] != targets[i]->Acquire())
            targets[i]->PublishGeneration(std::move(next[i]));
    }
}

std::shared_ptr<const RenderTextureGeneration> RenderTexture::PrepareGeneration(const RenderTextureDesc &description,
                                                                                uint32_t referenceWidth,
                                                                                uint32_t referenceHeight)
{
    if (m_lifetime && !m_lifetime->alive.load(std::memory_order_acquire))
        throw std::runtime_error("RenderTexture device has been destroyed");
    uint32_t width = description.width, height = description.height;
    if (description.sizeMode == RenderTextureSizeMode::Relative) {
        width = RelativeExtent(referenceWidth, description.widthScale);
        height = RelativeExtent(referenceHeight, description.heightScale);
    } else if (description.sizeMode != RenderTextureSizeMode::Absolute) {
        throw std::invalid_argument("Invalid RenderTexture size mode");
    }
    const auto current = Acquire();
    if (current && current->width == width && current->height == height &&
        SameDescription(current->description, description))
        return current;

    ValidateRenderTextureDescription(description);
    if (m_depthConsumers && description.depthFormat == PixelFormat::Undefined)
        throw std::invalid_argument("RenderTexture is bound to a Camera that requires a depth attachment");
    const auto &caps = m_device->GetCapabilities();
    if (width == 0 || height == 0 || width > caps.limits.maxTextureDimension2D ||
        height > caps.limits.maxTextureDimension2D)
        throw std::invalid_argument("RenderTexture dimensions exceed the device's 2D texture limits");
    auto features = FormatFeature::Sampled | FormatFeature::ColorAttachment | FormatFeature::TransferSource |
                    FormatFeature::TransferDestination;
    if (description.filter == FilterMode::Linear)
        features |= FormatFeature::FilterLinear;
    if (description.storage)
        features |= FormatFeature::Storage;
    RequireFormat(caps, description.colorFormat, features, SampleCount::One);
    RequireFormat(caps, description.colorFormat,
                  FormatFeature::ColorAttachment | FormatFeature::TransferSource | FormatFeature::Sampled,
                  description.samples);
    if (description.depthFormat != PixelFormat::Undefined) {
        auto depthFeatures =
            FormatFeature::DepthStencilAttachment | FormatFeature::TransferSource | FormatFeature::TransferDestination;
        if (description.sampledDepth)
            depthFeatures |= FormatFeature::Sampled;
        RequireFormat(caps, description.depthFormat, depthFeatures, description.samples);
    }

    auto next = std::make_shared<RenderTextureGeneration>();
    next->revision = current ? current->revision + 1 : 1;
    next->description = description;
    next->width = width;
    next->height = height;
    TextureDesc image;
    image.width = width;
    image.height = height;
    image.format = description.colorFormat;
    image.usage = TextureUsageFlags::Sampled | TextureUsageFlags::ColorAttachment | TextureUsageFlags::TransferSource |
                  TextureUsageFlags::TransferDestination;
    if (description.storage)
        image.usage = image.usage | TextureUsageFlags::Storage;
    next->color = CreateAttachment(*m_device, image, description.filter);
    if (description.samples != SampleCount::One) {
        image.samples = description.samples;
        image.usage =
            TextureUsageFlags::ColorAttachment | TextureUsageFlags::TransferSource | TextureUsageFlags::Sampled;
        next->multisampleColor = CreateAttachment(*m_device, image, description.filter);
    }
    if (description.depthFormat != PixelFormat::Undefined) {
        image.format = description.depthFormat;
        image.samples = description.samples;
        image.usage = TextureUsageFlags::DepthStencilAttachment | TextureUsageFlags::TransferSource |
                      TextureUsageFlags::TransferDestination;
        if (description.sampledDepth)
            image.usage = image.usage | TextureUsageFlags::Sampled;
        next->depth = CreateAttachment(*m_device, image, description.filter);
    }
    next->sampledColor = std::make_shared<const TextureGpuView>(m_sourceId, next->revision, next->color);
    return next;
}

void RenderTexture::PublishGeneration(std::shared_ptr<const RenderTextureGeneration> generation)
{
    (void)m_sampledColorSlot->TryPublish(generation->sampledColor);
    std::atomic_store_explicit(&m_current, std::move(generation), std::memory_order_release);
}
} // namespace infernux::rhi
