#include "SceneMaterialPass.h"

#include "MaterialPipelineState.h"
#include "MeshVertexLayout.h"

#include <algorithm>
#include <stdexcept>

namespace infernux
{
namespace
{
bool SameResources(const rhi::BindGroupDesc &a, const rhi::BindGroupDesc &b)
{
    if (a.layout != b.layout || a.bufferCount != b.bufferCount || a.textureCount != b.textureCount)
        return false;
    for (uint32_t i = 0; i < a.bufferCount; ++i) {
        const auto &x = a.buffers[i];
        const auto &y = b.buffers[i];
        if (x.binding != y.binding || x.type != y.type || x.buffer != y.buffer || x.offset != y.offset ||
            x.byteSize != y.byteSize)
            return false;
    }
    for (uint32_t i = 0; i < a.textureCount; ++i) {
        const auto &x = a.textures[i];
        const auto &y = b.textures[i];
        if (x.binding != y.binding || x.type != y.type || x.texture != y.texture || x.sampler != y.sampler ||
            x.depthRead != y.depthRead)
            return false;
    }
    return true;
}
} // namespace

SceneMaterialPass::SceneMaterialPass(rhi::Device &device, std::shared_ptr<const ShaderProgramArtifact> artifact,
                                     const RenderState &state, const MaterialPassPipelineDescriptor &pass)
    : m_device(device), m_artifact(std::move(artifact))
{
    if (!m_artifact)
        throw ShaderProgramSelectionError("Scene material must be prepared before device publication");
    const auto target = device.GetShaderSourceLanguage() == rhi::ShaderSourceLanguage::Wgsl ? ShaderCodeTarget::Wgsl
                                                                                            : ShaderCodeTarget::SpirV;
    const auto entries = SelectMaterialShaderEntries(*m_artifact, pass.target, target);
    m_program = std::make_unique<DeviceShaderProgram>(device, entries);
    auto pipeline = MaterialPipelineDescription(state, pass);
    m_program->Apply(pipeline);
    for (const auto &entry : entries) {
        const auto &code = entry.program.entries[entry.entry];
        if (code.stage == rhi::ShaderStage::Vertex)
            pipeline.vertexInput = MeshVertexLayout(code.inputs);
    }
    m_pipeline = device.CreateGraphicsPipeline(pipeline);
    if (!m_pipeline.IsValid())
        throw std::runtime_error("Scene material pipeline creation failed: " + m_artifact->key.stages.ToString());
    for (const auto &resource : m_program->GetResources()) {
        if (resource.set != 0 || resource.binding.type != rhi::BindingType::UniformBuffer)
            continue;
        Uniform value{resource, MaterialUniformLayout(resource.value.fields), {}, {}};
        value.bytes.resize(resource.value.byteSize);
        m_uniforms.push_back(std::move(value));
    }
}

SceneMaterialPass::~SceneMaterialPass()
{
    for (const auto &view : m_views)
        for (const auto &group : view)
            m_device.Release(group.handle);
    for (const auto &uniform : m_uniforms)
        m_device.Release(uniform.buffer);
    m_device.Release(m_pipeline);
}

bool SceneMaterialPass::Prepare(std::shared_ptr<const InxMaterial> material,
                                std::shared_ptr<const RendererParameterBlock> parameters,
                                const ResolveResource &resolve, uint32_t view)
{
    const bool changed = m_values != material || m_valueVersion != material->GetVersion() || m_parameters != parameters;
    if (m_views.size() <= view)
        m_views.resize(view + 1);
    auto &previousGroups = m_views[view];
    previousGroups.resize(m_program->GetBindingLayouts().size());
    std::vector<rhi::BindGroupDesc> groups(previousGroups.size());
    for (size_t i = 0; i < groups.size(); ++i)
        groups[i].layout = m_program->GetBindingLayouts()[i];
    for (auto &uniform : m_uniforms) {
        if (changed) {
            std::fill(uniform.bytes.begin(), uniform.bytes.end(), std::byte{});
            uniform.layout.Write(uniform.bytes.data(), *material);
            if (parameters)
                uniform.layout.Apply(uniform.bytes.data(), *material, *parameters);
            const float alpha =
                material->GetRenderState().alphaClipEnabled ? material->GetRenderState().alphaClipThreshold : 0.0f;
            uniform.layout.WriteValue(uniform.bytes.data(), "_alpha_clip_threshold", &alpha);
            // A newer material publication may arrive while an earlier frame
            // still reads this value. Publish new storage; the device retires
            // the old registration after its accepted submissions complete.
            rhi::BufferDesc description;
            description.byteSize = uniform.bytes.size();
            description.memory = rhi::BufferMemory::Upload;
            description.usage = rhi::BufferUsageFlags::Uniform | rhi::BufferUsageFlags::TransferDestination;
            description.initialData = uniform.bytes.data();
            description.initialDataBytes = uniform.bytes.size();
            const auto replacement = m_device.CreateBuffer(description);
            if (!replacement.IsValid())
                throw std::runtime_error("Scene material value publication failed");
            m_device.Release(uniform.buffer);
            uniform.buffer = replacement;
        }
        auto &group = groups[uniform.resource.set];
        group.buffers[group.bufferCount++] = {uniform.resource.binding.binding, rhi::BindingType::UniformBuffer,
                                              uniform.buffer, 0, uniform.bytes.size()};
    }
    m_valueVersion = material->GetVersion();
    m_values = std::move(material);
    m_parameters = parameters;
    for (const auto &resource : m_program->GetResources()) {
        if ((resource.set != 0 || resource.binding.type != rhi::BindingType::UniformBuffer) &&
            !resolve(resource, groups[resource.set]))
            return false;
    }
    for (size_t i = 0; i < groups.size(); ++i) {
        auto &previous = previousGroups[i];
        if (previous.handle.IsValid() && SameResources(previous.description, groups[i]))
            continue;
        auto next = m_device.CreateBindGroup(groups[i]);
        if (!next.IsValid())
            throw std::runtime_error("Scene material resource binding failed");
        m_device.Release(previous.handle);
        previous = {groups[i], next};
    }
    return true;
}

void SceneMaterialPass::Bind(const rhi::GraphicsCommandEncoder &encoder, uint32_t view) const
{
    encoder.BindPipeline(m_pipeline);
    const auto &groups = m_views[view];
    for (uint32_t i = 0; i < groups.size(); ++i)
        encoder.BindGroup(m_pipeline, i, groups[i].handle);
}
} // namespace infernux
