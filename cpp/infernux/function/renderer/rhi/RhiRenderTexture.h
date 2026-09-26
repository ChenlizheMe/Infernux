#pragma once

#include "RhiTexture.h"
#include <vector>

namespace infernux::rhi
{

enum class RenderTextureSizeMode : uint8_t
{
    Absolute,
    Relative,
};

/// Persistent 2D target description. Relative dimensions round up to pixels.
/// Contents start undefined; a producer must write/clear before any sampling.
/// One mip is provided here; mip generation is a render-graph operation, not
/// an implicit side effect of allocation or publication.
struct RenderTextureDesc
{
    RenderTextureSizeMode sizeMode = RenderTextureSizeMode::Absolute;
    uint32_t width = 256;
    uint32_t height = 256;
    float widthScale = 1.0f;
    float heightScale = 1.0f;
    PixelFormat colorFormat = PixelFormat::RGBA8UNorm;
    PixelFormat depthFormat = PixelFormat::Undefined;
    SampleCount samples = SampleCount::One;
    FilterMode filter = FilterMode::Linear;
    bool storage = false;
    bool sampledDepth = false; ///< Depth sampling is explicit; no implicit filtering/comparison sampler.
};

/// Device-independent authoring contract. Device capabilities are checked at allocation.
void ValidateRenderTextureDescription(const RenderTextureDesc &description);

/// A complete immutable allocation generation. Hold this while recording or
/// consuming a target; never cache individual handles across resize. Color is
/// always single-sample and sampleable. Raster writes ColorAttachment(), then
/// resolves to color when MSAA is enabled. Depth matches raster sample count.
struct RenderTextureGeneration final
{
    uint64_t revision = 0;
    RenderTextureDesc description;
    uint32_t width = 0;
    uint32_t height = 0;
    std::shared_ptr<TextureResource> color;
    std::shared_ptr<TextureResource> multisampleColor;
    std::shared_ptr<TextureResource> depth;
    std::shared_ptr<const TextureGpuView> sampledColor;

    // Device-owner-thread recording state, shared by every borrowing graph.
    // This tracks the initial layout publication, not whether pixels are valid.
    // A resize publishes a new generation with a fresh initialization state.
    mutable bool graphLayoutsInitialized = false;

    [[nodiscard]] const TextureResource &ColorAttachment() const noexcept
    {
        return *(multisampleColor ? multisampleColor : color);
    }
    /// Texel payload, not driver allocation padding or descriptor overhead.
    [[nodiscard]] uint64_t GetResidentBytes() const noexcept
    {
        return color->GetResidentBytes() + (multisampleColor ? multisampleColor->GetResidentBytes() : 0) +
               (depth ? depth->GetResidentBytes() : 0);
    }
};

/// Engine-owned persistent target; no Vulkan/ImGui/editor dependencies.
/// Reconfigure runs on the device owner thread, as does device teardown.
/// Acquire may be used by readers; it publishes attachments together, not a
/// mixture of old depth/new color. Backend Release owns in-flight retirement.
class RenderTexture final
{
  public:
    RenderTexture(Device &device, std::string sourceId, const RenderTextureDesc &description,
                  uint32_t referenceWidth = 0, uint32_t referenceHeight = 0, std::string assetGuid = {});
    RenderTexture(const RenderTexture &) = delete;
    RenderTexture &operator=(const RenderTexture &) = delete;

    /// Returns false when the resolved allocation/description is unchanged.
    /// Failure reports an error and does not publish incomplete attachments.
    bool Reconfigure(const RenderTextureDesc &description, uint32_t referenceWidth = 0, uint32_t referenceHeight = 0);
    /// Prepare all relative allocations before publishing any. The caller owns
    /// this device-thread size transition; unchanged extents allocate nothing.
    static void ReconfigureReferenceSize(const std::vector<std::shared_ptr<RenderTexture>> &targets,
                                         uint32_t referenceWidth, uint32_t referenceHeight);
    [[nodiscard]] std::shared_ptr<const RenderTextureGeneration> Acquire() const noexcept;
    [[nodiscard]] const std::string &GetAssetGuid() const noexcept
    {
        return m_assetGuid;
    }
    /// Camera bindings retain this requirement even while disabled.
    void RetainDepthAttachment();
    void ReleaseDepthAttachment() noexcept;
    /// Stable sampled-color publication shared by material/UI consumers.
    /// Consumers retain a view while recording; resize publishes its successor.
    [[nodiscard]] const std::shared_ptr<TextureGpuViewSlot> &GetSampledColorSlot() const noexcept
    {
        return m_sampledColorSlot;
    }

  private:
    [[nodiscard]] std::shared_ptr<const RenderTextureGeneration>
    PrepareGeneration(const RenderTextureDesc &description, uint32_t referenceWidth, uint32_t referenceHeight);
    void PublishGeneration(std::shared_ptr<const RenderTextureGeneration> generation);
    Device *m_device;
    std::shared_ptr<DeviceLifetime> m_lifetime;
    std::string m_sourceId;
    std::string m_assetGuid;
    uint32_t m_depthConsumers = 0;
    std::shared_ptr<TextureGpuViewSlot> m_sampledColorSlot;
    std::shared_ptr<const RenderTextureGeneration> m_current;
};

} // namespace infernux::rhi
