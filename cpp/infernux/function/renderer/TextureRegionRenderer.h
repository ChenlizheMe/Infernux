#pragma once

#include "rhi/RenderCoreApi.h"
#include "rhi/RhiCommand.h"
#include "rhi/RhiTexture.h"
#include <memory>
#include <string>

namespace infernux
{
struct ShaderTargetCode;

/// Exact texel/sample transfer through rasterization, including depth regions
/// on devices whose transfer queue only permits complete depth subresources.
/// The caller owns attachment transitions, viewport/scissor and pass lifetime.
class TextureRegionRenderer
{
  public:
    INFERNUX_RENDER_CORE_API TextureRegionRenderer(rhi::Device &device, const ShaderTargetCode &code,
                                                   rhi::PixelFormat format, rhi::SampleCount samples);
    INFERNUX_RENDER_CORE_API ~TextureRegionRenderer();
    INFERNUX_RENDER_CORE_API void ClearBindings();
    [[nodiscard]] INFERNUX_RENDER_CORE_API static std::string ShaderSource(bool depth, bool multisample);
    INFERNUX_RENDER_CORE_API void Record(const rhi::GraphicsCommandEncoder &commands,
                                         std::shared_ptr<const rhi::TextureResource> source, int32_t sourceX,
                                         int32_t sourceY, int32_t destinationX, int32_t destinationY);

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
} // namespace infernux
