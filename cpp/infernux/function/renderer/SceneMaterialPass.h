#pragma once

#include "MaterialPassPipeline.h"
#include "rhi/RhiCommand.h"
#include "rhi/RhiDevice.h"
#include "shader/DeviceShaderProgram.h"
#include "shader/MaterialUniformLayout.h"

#include <functional>

namespace infernux
{

/// One authored material/pass on a device. The compiler owns all field and
/// binding layouts; scene execution only supplies accepted resource values.
/// No shader-name classification or backend-specific material ABI lives here.
class SceneMaterialPass final
{
  public:
    using ResolveResource = std::function<bool(const ShaderResourceLayout &, rhi::BindGroupDesc &)>;

    SceneMaterialPass(rhi::Device &device, std::shared_ptr<const ShaderProgramArtifact> artifact,
                      const RenderState &state, const MaterialPassPipelineDescriptor &pass);
    ~SceneMaterialPass();
    SceneMaterialPass(const SceneMaterialPass &) = delete;
    SceneMaterialPass &operator=(const SceneMaterialPass &) = delete;

    /// Values change without compiling shaders or creating pipelines. Binding
    /// groups change only when their actual resource allocations change.
    bool Prepare(std::shared_ptr<const InxMaterial> material, std::shared_ptr<const RendererParameterBlock> parameters,
                 const ResolveResource &resolve, uint32_t view = 0);
    void Bind(const rhi::GraphicsCommandEncoder &encoder, uint32_t view = 0) const;

  private:
    rhi::Device &m_device;
    std::shared_ptr<const ShaderProgramArtifact> m_artifact;
    std::unique_ptr<DeviceShaderProgram> m_program;
    rhi::GraphicsPipelineHandle m_pipeline;
    struct Uniform
    {
        ShaderResourceLayout resource;
        MaterialUniformLayout layout;
        rhi::BufferHandle buffer;
        std::vector<std::byte> bytes;
    };
    struct Group
    {
        rhi::BindGroupDesc description;
        rhi::BindGroupHandle handle;
    };
    std::vector<Uniform> m_uniforms;
    std::vector<std::vector<Group>> m_views;
    uint64_t m_valueVersion = UINT64_MAX;
    std::shared_ptr<const InxMaterial> m_values;
    std::shared_ptr<const RendererParameterBlock> m_parameters;
};

} // namespace infernux
