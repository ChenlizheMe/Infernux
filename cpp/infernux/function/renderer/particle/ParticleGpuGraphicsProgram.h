#pragma once

#include "ParticleGpuSurfaceBinding.h"
#include <function/renderer/MaterialPassPipeline.h>
#include <function/renderer/shader/DeviceShaderProgram.h>
#include <function/renderer/shader/ShaderViewBindings.h>

#include <memory>
#include <vector>

namespace infernux::particle
{

/// Device publication shared by billboard, mesh and ribbon outputs. Geometry
/// and authored surface keep separate compiled modules and compose actual
/// entries. Per-view bindings remain owned by their camera/light view.
class ParticleGpuGraphicsProgram final
{
  public:
    ParticleGpuGraphicsProgram(rhi::Device &device, std::shared_ptr<const ShaderCompilation> geometry,
                               std::shared_ptr<const ShaderProgramArtifact> surface,
                               rhi::BindingLayoutHandle materialLayout, rhi::CullMode cullMode);
    ~ParticleGpuGraphicsProgram();
    ParticleGpuGraphicsProgram(const ParticleGpuGraphicsProgram &) = delete;
    ParticleGpuGraphicsProgram &operator=(const ParticleGpuGraphicsProgram &) = delete;

    [[nodiscard]] rhi::GraphicsPipelineHandle Get(const MaterialPassPipelineDescriptor &pass,
                                                  const GpuBillboardMaterialState &materialState,
                                                  const ShaderViewBindings &perView,
                                                  rhi::BindingLayoutHandle geometryLayout, uint8_t softDepthKind);

    void ReleaseView(rhi::BufferHandle globals) noexcept;

  private:
    struct Pipeline
    {
        MaterialPassPipelineDescriptor pass;
        rhi::BindingLayoutHandle perViewLayout;
        rhi::BindingLayoutHandle geometryLayout;
        uint8_t state;
        uint8_t softDepthKind;
        std::unique_ptr<DeviceShaderProgram> program;
        rhi::GraphicsPipelineHandle handle;
        std::vector<rhi::BufferHandle> viewGlobals;
    };

    rhi::Device &m_device;
    std::shared_ptr<const ShaderCompilation> m_geometry;
    std::shared_ptr<const ShaderProgramArtifact> m_surface;
    const ShaderTargetCode *m_geometryTarget;
    const ShaderTargetCode *m_surfaceTarget;
    rhi::BindingLayoutHandle m_materialLayout;
    rhi::CullMode m_cullMode;
    std::vector<Pipeline> m_pipelines;
};

} // namespace infernux::particle
