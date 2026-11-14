#include "ParticleGpuRenderer.h"

#include <function/renderer/EngineGlobals.h>
#include <function/renderer/particle/ParticleGpuGraphicsProgram.h>
#include <function/renderer/rhi/RhiDevice.h>

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace infernux::particle
{

ParticleGpuRenderer::ParticleGpuRenderer() = default;

ParticleGpuRenderer::~ParticleGpuRenderer()
{
    Destroy();
}

bool ParticleGpuRenderer::Create(rhi::Device &device, const GpuParticleRendererDesc &desc)
{
    Destroy();
    m_device = &device;
    m_type = desc.type;
    m_semantics = desc.semantics;
    m_shaderProgram = desc.surface;
    m_meshOwner = desc.meshOwner;
    m_ribbon = desc.ribbon;
    m_columns = desc.flipbookColumns;
    m_rows = desc.flipbookRows;
    m_uvMode = desc.ribbonUvMode;
    m_uvScale = desc.ribbonUvScale;
    // Establish the prepared material contract before consuming its metadata.
    // Cold publication and hot refresh enter through this same owner.
    if (!desc.geometry || !m_surface.Create(device, desc.surface, desc.material, desc.materialState, desc.semantics,
                                            desc.textureResolver, desc.retirement)) {
        Destroy();
        return false;
    }
    m_buffers = {desc.instances, desc.renderIndices, desc.meshVertices.buffer, desc.meshIndices.buffer};
    m_viewIndexSlot = m_type == GpuParticleOutputType::Ribbon ? 2 : 1;
    m_vertexCount = m_type == GpuParticleOutputType::Mesh ? desc.indexCount : 6;
    if (m_type == GpuParticleOutputType::Ribbon) {
        m_buffers = {m_ribbon->InstanceBuffer(), m_ribbon->SortedIndexBuffer(), m_ribbon->SortedIndexBuffer(),
                     m_ribbon->SourceIndirectBuffer()};
        m_semantics.sortMode = ParticleSortMode::None;
    } else if (m_type == GpuParticleOutputType::Mesh) {
        m_staticBuffers = {desc.meshVertices, desc.meshIndices};
    }

    const auto target = device.GetShaderSourceLanguage() == rhi::ShaderSourceLanguage::SpirV ? ShaderCodeTarget::SpirV
                                                                                             : ShaderCodeTarget::Wgsl;
    const auto geometry = std::find_if(desc.geometry->targets.begin(), desc.geometry->targets.end(),
                                       [target](const auto &code) { return code.target == target; });
    if (geometry == desc.geometry->targets.end())
        throw std::invalid_argument("Particle geometry has no target for the selected device");
    const std::unordered_map<std::string, uint8_t> sources = {
        {"instances", 0},        {"render_indices", 1}, {"sorted_indices", 1},   {"mesh_vertices", 2},
        {"visible_segments", 2}, {"mesh_indices", 3},   {"source_arguments", 3},
    };
    rhi::BindingLayoutDesc layout;
    layout.entries[layout.entryCount++] = {0, rhi::BindingType::UniformBuffer,
                                           rhi::ShaderStage::Vertex | rhi::ShaderStage::Fragment};
    for (const auto &resource : geometry->resources) {
        const std::array<uint32_t, 2> key{resource.set, resource.binding.binding};
        const bool used = std::any_of(geometry->entries.begin(), geometry->entries.end(), [&](const auto &entry) {
            return entry.stage == rhi::ShaderStage::Vertex &&
                   std::find(entry.bindings.begin(), entry.bindings.end(), key) != entry.bindings.end();
        });
        if (resource.set != 2 || !used || resource.binding.binding == 0)
            continue;
        m_bufferSlots.push_back({resource.binding.binding, sources.at(resource.name)});
        layout.entries.at(layout.entryCount++) = resource.binding;
    }
    m_geometryLayouts[0] = device.CreateBindingLayout(layout);
    if (m_semantics.softParticles) {
        const auto surface =
            std::find_if(desc.surface->compilation->targets.begin(), desc.surface->compilation->targets.end(),
                         [target](const auto &code) { return code.target == target; });
        const std::array<std::array<std::string_view, 2>, 2> names{{
            {"fragment_soft_depth", "fragment_forward_plus_soft_depth"},
            {"fragment_soft_color", "fragment_forward_plus_soft_color"},
        }};
        for (size_t index = 0; index < names.size(); ++index) {
            auto softLayout = layout;
            for (const auto name : names[index]) {
                const auto entry = std::find_if(surface->entries.begin(), surface->entries.end(),
                                                [&](const auto &value) { return value.name == name; });
                if (entry == surface->entries.end()) {
                    Destroy();
                    return false;
                }
                for (const auto &key : entry->bindings) {
                    const auto mapped =
                        std::find_if(softLayout.entries.begin(), softLayout.entries.begin() + softLayout.entryCount,
                                     [&](const auto &value) { return value.binding == key[1]; });
                    if (key[0] != 2 || mapped != softLayout.entries.begin() + softLayout.entryCount)
                        continue;
                    const auto resource =
                        std::find_if(surface->resources.begin(), surface->resources.end(), [&](const auto &value) {
                            return value.set == key[0] && value.binding.binding == key[1];
                        });
                    if (resource == surface->resources.end() ||
                        resource->binding.type != rhi::BindingType::SampledTexture || m_depthReads[index]) {
                        Destroy();
                        return false;
                    }
                    softLayout.entries.at(softLayout.entryCount++) = resource->binding;
                    m_depthReads[index] = true;
                    m_depthBindings[index] = resource->binding.binding;
                }
            }
            m_geometryLayouts[index + 1] = device.CreateBindingLayout(softLayout);
        }
    }
    if (!m_geometryLayouts[0].IsValid()) {
        Destroy();
        return false;
    }
    m_program = std::make_unique<ParticleGpuGraphicsProgram>(
        device, desc.geometry, desc.surface, m_surface.Layout(),
        m_type == GpuParticleOutputType::Mesh ? rhi::CullMode::Back : rhi::CullMode::None);
    return true;
}

void ParticleGpuRenderer::Destroy() noexcept
{
    m_program.reset();
    if (m_device) {
        for (const auto &entry : m_viewGroups)
            m_device->Release(entry.group);
        for (const auto layout : m_geometryLayouts)
            m_device->Release(layout);
    }
    m_surface.Destroy();
    m_device = nullptr;
    m_shaderProgram.reset();
    m_meshOwner.reset();
    m_ribbon.reset();
    m_buffers = {};
    m_bufferSlots.clear();
    m_staticBuffers.clear();
    m_viewGroups.clear();
    m_geometryLayouts = {};
    m_depthReads = {};
    m_depthBindings = {};
}

bool ParticleGpuRenderer::IsValid() const noexcept
{
    return m_device && m_program && m_geometryLayouts[0].IsValid() && m_surface.IsValid();
}

int32_t ParticleGpuRenderer::RenderQueue() const noexcept
{
    return m_surface.ResolveMaterialState().renderQueue;
}

bool ParticleGpuRenderer::RequiresSceneDepth() const noexcept
{
    return m_semantics.softParticles && (m_depthReads[0] || m_depthReads[1]);
}

bool ParticleGpuRenderer::CanCastShadows() const noexcept
{
    return m_shaderProgram && m_shaderProgram->FindVariant(ShaderCompileTarget::Shadow);
}

uint32_t ParticleGpuRenderer::VertexCount() const noexcept
{
    return m_vertexCount;
}

rhi::BufferHandle ParticleGpuRenderer::InstanceBuffer() const noexcept
{
    return m_buffers[0];
}

rhi::BufferHandle ParticleGpuRenderer::RenderIndexBuffer() const noexcept
{
    return m_buffers[1];
}

size_t ParticleGpuRenderer::ViewBindingCount() const noexcept
{
    return m_viewGroups.size();
}

const std::vector<GpuParticleStaticBuffer> &ParticleGpuRenderer::StaticVertexStorageBuffers() const noexcept
{
    return m_staticBuffers;
}

rhi::BindGroupHandle ParticleGpuRenderer::GeometryGroup(rhi::BufferHandle indices, rhi::BufferHandle globals,
                                                        rhi::TextureViewHandle sceneDepth, uint8_t depthKind)
{
    if (!indices.IsValid())
        indices = RenderIndexBuffer();
    const auto found = std::find_if(m_viewGroups.begin(), m_viewGroups.end(),
                                    [indices, globals, sceneDepth, depthKind](const auto &entry) {
                                        return entry.indices == indices && entry.globals == globals &&
                                               entry.sceneDepth == sceneDepth && entry.depthKind == depthKind;
                                    });
    if (found != m_viewGroups.end())
        return found->group;
    auto buffers = m_buffers;
    buffers[m_viewIndexSlot] = indices;
    rhi::BindGroupDesc group;
    group.layout = m_geometryLayouts[depthKind];
    group.buffers[group.bufferCount++] = {0, rhi::BindingType::UniformBuffer, globals, 0, sizeof(EngineGlobalsUBO)};
    for (const auto &slot : m_bufferSlots)
        group.buffers[group.bufferCount++] = {slot.binding, rhi::BindingType::StorageBuffer, buffers[slot.source], 0,
                                              0};
    if (depthKind != 0 && m_depthReads[depthKind - 1]) {
        group.textures[0] = {
            m_depthBindings[depthKind - 1], rhi::BindingType::SampledTexture, sceneDepth, {}, depthKind == 1};
        group.textureCount = 1;
    }
    const auto handle = m_device->CreateBindGroup(group);
    if (handle.IsValid())
        m_viewGroups.push_back({indices, globals, sceneDepth, depthKind, handle});
    return handle;
}

void ParticleGpuRenderer::ReleaseViewBinding(rhi::BufferHandle indices) noexcept
{
    m_viewGroups.erase(std::remove_if(m_viewGroups.begin(), m_viewGroups.end(),
                                      [&](const auto &entry) {
                                          if (entry.indices != indices)
                                              return false;
                                          m_device->Release(entry.group);
                                          return true;
                                      }),
                       m_viewGroups.end());
}

void ParticleGpuRenderer::ReleaseView(rhi::BufferHandle globals) noexcept
{
    m_viewGroups.erase(std::remove_if(m_viewGroups.begin(), m_viewGroups.end(),
                                      [&](const auto &entry) {
                                          if (entry.globals != globals)
                                              return false;
                                          m_device->Release(entry.group);
                                          return true;
                                      }),
                       m_viewGroups.end());
    m_program->ReleaseView(globals);
}

GpuParticleViewConstants ParticleGpuRenderer::DrawConstants(const GpuParticleViewConstants &view,
                                                            ShaderCompileTarget pass, rhi::BufferHandle indices) const
{
    auto result = view;
    result.cameraRight[3] = m_semantics.softDistance;
    const bool color = pass == ShaderCompileTarget::Forward || pass == ShaderCompileTarget::ForwardPlus;
    result.cameraUp[3] = color && m_semantics.softParticles ? 1.0f : 0.0f;
    result.lightingControl[0] =
        pass == ShaderCompileTarget::ForwardPlus && m_semantics.receiveSceneLighting ? 1.0f : 0.0f;
    result.lightingControl[1] = m_semantics.sortMode != ParticleSortMode::None ? 1.0f : 0.0f;
    result.lightingControl[2] = m_surface.ResolveMaterialFloat("softness", 0.18f);
    result.lightingControl[3] = m_semantics.receiveSceneLighting ? 1.0f : 0.0f;
    result.renderingControl[0] = m_semantics.receiveShadows ? 1.0f : 0.0f;
    result.renderingControl[1] = m_surface.ResolveMaterialState().premultipliedAlpha ? 1.0f : 0.0f;
    result.renderingControl[2] = static_cast<float>(m_columns);
    result.renderingControl[3] = static_cast<float>(m_rows);
    if (m_type == GpuParticleOutputType::Ribbon) {
        result.lightingControl[1] = 0.0f;
        result.alignmentReference = {m_uvScale, m_uvMode == ParticleRibbonUvMode::Repeat ? 1.0f : 0.0f,
                                     indices.IsValid() && indices != RenderIndexBuffer() ? 1.0f : 0.0f, -1.0f};
    } else {
        result.alignmentReference[3] = static_cast<float>(m_semantics.spriteAlignment);
        if (m_semantics.spriteAlignment == ParticleSpriteAlignment::Axis)
            std::copy(m_semantics.alignmentAxis.begin(), m_semantics.alignmentAxis.end(),
                      result.alignmentReference.begin());
    }
    return result;
}

bool ParticleGpuRenderer::RecordDraw(const rhi::GraphicsCommandEncoder &encoder,
                                     const MaterialPassPipelineDescriptor &pass, rhi::BufferHandle indirectArguments,
                                     const GpuParticleViewConstants &view, const ShaderViewBindings &perView,
                                     rhi::BufferHandle indices, rhi::TextureViewHandle sceneDepth,
                                     bool sceneDepthIsDepth)
{
    const bool color = pass.target == ShaderCompileTarget::Forward || pass.target == ShaderCompileTarget::ForwardPlus;
    const uint8_t depthKind = color && m_semantics.softParticles ? (sceneDepthIsDepth ? 1 : 2) : 0;
    const bool readsDepth = depthKind != 0 && m_depthReads[depthKind - 1];
    if (!m_surface.RefreshMaterialBuffer(false) || !m_surface.RefreshTextureBindings(false) ||
        (readsDepth && !sceneDepth.IsValid()))
        return false;
    auto materialPass = pass;
    if (pass.target == ShaderCompileTarget::ForwardPlus && !m_semantics.receiveSceneLighting)
        materialPass.target = ShaderCompileTarget::Forward;
    const auto pipeline = m_program->Get(materialPass, m_surface.ResolveMaterialState(), perView,
                                         m_geometryLayouts[depthKind], depthKind);
    const auto geometry =
        GeometryGroup(indices, perView.globals, readsDepth ? sceneDepth : rhi::TextureViewHandle{}, depthKind);
    const auto surface = m_surface.ResolveBindGroup();
    if (!pipeline.IsValid() || !geometry.IsValid() || !surface.IsValid())
        return false;
    const auto constants = DrawConstants(view, pass.target, indices);
    encoder.BindPipeline(pipeline);
    encoder.BindGroup(pipeline, 0, surface);
    encoder.BindGroup(pipeline, 1, perView.group);
    encoder.BindGroup(pipeline, 2, geometry);
    encoder.PushConstants(pipeline, rhi::ShaderStage::Vertex | rhi::ShaderStage::Fragment, sizeof(constants),
                          &constants);
    encoder.DrawIndirect(indirectArguments);
    return true;
}

bool ParticleGpuRenderer::RecordPickingDraw(const rhi::GraphicsCommandEncoder &encoder,
                                            const MaterialPassPipelineDescriptor &pass,
                                            rhi::BufferHandle indirectArguments, const GpuParticleViewConstants &view,
                                            uint64_t ownerObjectId, const ShaderViewBindings &perView,
                                            rhi::BufferHandle indices)
{
    auto selected = view;
    selected.objectData[0] = static_cast<uint32_t>(ownerObjectId);
    selected.objectData[1] = static_cast<uint32_t>(ownerObjectId >> 32u);
    return RecordDraw(encoder, pass, indirectArguments, selected, perView, indices);
}

} // namespace infernux::particle
