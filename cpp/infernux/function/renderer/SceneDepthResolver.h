#pragma once

#include "rhi/RenderCoreApi.h"
#include "rhi/RhiCommand.h"
#include "rhi/RhiDevice.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace infernux
{

struct SceneDepthResolveProgram;

/// RHI compute implementation of the scene-depth sampling contract. It
/// converts a multisampled depth attachment into a single-sample R32F texture
/// without exposing backend-specific resolve commands to particle rendering.
class SceneDepthResolver
{
  public:
    SceneDepthResolver() = default;
    INFERNUX_RENDER_CORE_API ~SceneDepthResolver();

    SceneDepthResolver(const SceneDepthResolver &) = delete;
    SceneDepthResolver &operator=(const SceneDepthResolver &) = delete;

    [[nodiscard]] INFERNUX_RENDER_CORE_API bool Initialize(rhi::Device &device, const uint32_t *spirv,
                                                           size_t wordCount);
    [[nodiscard]] INFERNUX_RENDER_CORE_API static std::shared_ptr<const SceneDepthResolveProgram>
    CreateProgram(rhi::Device &device, const uint32_t *spirv, size_t wordCount);
    [[nodiscard]] INFERNUX_RENDER_CORE_API bool Initialize(std::shared_ptr<const SceneDepthResolveProgram> program);
    INFERNUX_RENDER_CORE_API void Destroy() noexcept;

    [[nodiscard]] INFERNUX_RENDER_CORE_API bool IsValid() const noexcept;
    [[nodiscard]] INFERNUX_RENDER_CORE_API bool Record(const rhi::ComputeCommandEncoder &encoder,
                                                       rhi::TextureViewHandle sourceDepth,
                                                       rhi::TextureViewHandle resolvedDepth, uint32_t width,
                                                       uint32_t height, uint32_t sampleCount);

    /// Detach graph-specific descriptor groups so the caller can retire them
    /// after in-flight command buffers finish using the previous graph.
    [[nodiscard]] INFERNUX_RENDER_CORE_API std::vector<rhi::BindGroupHandle> TakeBindGroups();

    [[nodiscard]] INFERNUX_RENDER_CORE_API static std::string_view ShaderSource() noexcept;

  private:
    struct BindingEntry
    {
        rhi::TextureViewHandle sourceDepth;
        rhi::TextureViewHandle resolvedDepth;
        rhi::BindGroupHandle group;
    };

    [[nodiscard]] rhi::BindGroupHandle ResolveBindGroup(rhi::TextureViewHandle sourceDepth,
                                                        rhi::TextureViewHandle resolvedDepth);

    rhi::Device *m_device = nullptr;
    std::shared_ptr<const SceneDepthResolveProgram> m_program;
    std::vector<BindingEntry> m_bindings;
};

} // namespace infernux
