#pragma once

#include "ShaderProgramCode.h"
#include <function/renderer/rhi/RenderCoreApi.h>

#include <memory>

namespace infernux::rhi
{
class Device;
}

namespace infernux
{

/// A concrete selection of entry points, with only their used bindings.
/// Device publication owns modules and layouts; it never invokes a compiler
/// or consults authoring source. Pipelines retain the backend objects.
/// Shared layouts are borrowed by set index; their owner outlives this program.
class DeviceShaderProgram
{
  public:
    INFERNUX_RENDER_CORE_API DeviceShaderProgram(rhi::Device &device, const ShaderTargetCode &code,
                                                 const std::vector<uint32_t> &entries,
                                                 const std::vector<rhi::BindingLayoutHandle> &sharedLayouts = {});
    INFERNUX_RENDER_CORE_API ~DeviceShaderProgram();
    DeviceShaderProgram(const DeviceShaderProgram &) = delete;
    DeviceShaderProgram &operator=(const DeviceShaderProgram &) = delete;

    INFERNUX_RENDER_CORE_API void Apply(rhi::GraphicsPipelineDesc &pipeline) const;
    INFERNUX_RENDER_CORE_API void Apply(rhi::ComputePipelineDesc &pipeline) const;
    [[nodiscard]] INFERNUX_RENDER_CORE_API const std::vector<ShaderResourceLayout> &GetResources() const noexcept;
    [[nodiscard]] INFERNUX_RENDER_CORE_API const std::vector<rhi::BindingLayoutHandle> &
    GetBindingLayouts() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace infernux
