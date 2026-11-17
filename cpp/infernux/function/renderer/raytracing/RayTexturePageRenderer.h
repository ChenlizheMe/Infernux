#pragma once

#include <function/renderer/rhi/RhiRasterPass.h>

#include <cstdint>
#include <memory>

namespace infernux::rhi
{
class Device;
class GraphicsCommandEncoder;
class TextureGpuView;
class TextureResource;
} // namespace infernux::rhi

namespace infernux::raytracing
{

struct RayTexturePageProgram;

/// Prepared normalization draw for one destination mip and array layer.
/// Retain the draw through the accepted graphics submission.
class RayTexturePageDraw final
{
  public:
    ~RayTexturePageDraw();

    [[nodiscard]] const rhi::RasterPassDescription &Raster() const noexcept;
    [[nodiscard]] bool Record(const rhi::GraphicsCommandEncoder &encoder) const;

  private:
    friend class RayTexturePageRenderer;

    RayTexturePageDraw(std::shared_ptr<RayTexturePageProgram> program,
                       std::shared_ptr<const rhi::TextureGpuView> source,
                       std::shared_ptr<const rhi::TextureResource> destination) noexcept;
    RayTexturePageDraw(const RayTexturePageDraw &) = delete;
    RayTexturePageDraw &operator=(const RayTexturePageDraw &) = delete;

    std::shared_ptr<RayTexturePageProgram> m_program;
    std::shared_ptr<const rhi::TextureGpuView> m_source;
    std::shared_ptr<const rhi::TextureResource> m_destination;
    rhi::RasterPassDescription m_raster;
    rhi::GraphicsPipelineHandle m_pipeline;
    rhi::BufferHandle m_constants;
    rhi::BindGroupHandle m_group;
};

/// Device-local normalization program shared by one world material owner.
/// The device outlives the renderer and its retained draw submissions.
class RayTexturePageRenderer final
{
  public:
    explicit RayTexturePageRenderer(rhi::Device &device) noexcept;

    RayTexturePageRenderer(const RayTexturePageRenderer &) = delete;
    RayTexturePageRenderer &operator=(const RayTexturePageRenderer &) = delete;

    /// Width and height cover the full destination mip. The recorder sets the
    /// scissor for its row range. LOD is relative to the frozen source view.
    [[nodiscard]] std::shared_ptr<const RayTexturePageDraw>
    Prepare(std::shared_ptr<const rhi::TextureGpuView> source, std::shared_ptr<const rhi::TextureResource> destination,
            uint32_t width, uint32_t height, uint32_t mip, uint32_t layer, float sourceLod);

  private:
    rhi::Device &m_device;
    std::shared_ptr<RayTexturePageProgram> m_program;
};

} // namespace infernux::raytracing
