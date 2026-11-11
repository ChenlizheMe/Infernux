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
struct ShaderTargetCode;
class GpuRetirementQueue;

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

    [[nodiscard]] INFERNUX_RENDER_CORE_API static std::shared_ptr<const SceneDepthResolveProgram>
    CreateProgram(rhi::Device &device, const ShaderTargetCode &code);
    INFERNUX_RENDER_CORE_API void Initialize(std::shared_ptr<const SceneDepthResolveProgram> program);
    INFERNUX_RENDER_CORE_API void Destroy() noexcept;

    [[nodiscard]] INFERNUX_RENDER_CORE_API bool IsValid() const noexcept;
    INFERNUX_RENDER_CORE_API void Record(const rhi::ComputeCommandEncoder &encoder, rhi::TextureViewHandle sourceDepth,
                                         rhi::TextureViewHandle resolvedDepth, uint32_t width, uint32_t height,
                                         uint32_t sampleCount);

    /// Groups and their immutable constants share the previous graph's GPU lifetime.
    INFERNUX_RENDER_CORE_API void RetireBindings(GpuRetirementQueue &queue);

    [[nodiscard]] INFERNUX_RENDER_CORE_API static std::string_view ShaderSource() noexcept;

  private:
    struct BindingEntry
    {
        rhi::TextureViewHandle sourceDepth;
        rhi::TextureViewHandle resolvedDepth;
        uint32_t width;
        uint32_t height;
        uint32_t sampleCount;
        rhi::BindGroupHandle group;
        rhi::BufferHandle constants;
    };

    [[nodiscard]] rhi::BindGroupHandle ResolveBindGroup(rhi::TextureViewHandle sourceDepth,
                                                        rhi::TextureViewHandle resolvedDepth, uint32_t width,
                                                        uint32_t height, uint32_t sampleCount);

    rhi::Device *m_device = nullptr;
    std::shared_ptr<const SceneDepthResolveProgram> m_program;
    std::vector<BindingEntry> m_bindings;
};

} // namespace infernux
