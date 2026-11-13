#include "ParticleGpuRibbonRenderer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace infernux::particle
{

namespace
{

constexpr std::string_view RibbonGeometrySource = R"slang(struct ParticleRenderInstance {
    float4 position_size;
    float4 color;
    float4 rotation_custom;
    float4 scale_custom;
    uint4 ribbon_data;
    float4 custom_data;
    float4 previous_position_history;
};
struct DrawArguments {
    uint vertex_count;
    uint instance_count;
    uint first_vertex;
    uint first_instance;
};
[[vk::binding(0, 0)]] StructuredBuffer<ParticleRenderInstance> instances;
[[vk::binding(1, 0)]] StructuredBuffer<uint> sorted_indices;
[[vk::binding(2, 0)]] StructuredBuffer<uint> visible_segments;
[[vk::binding(3, 0)]] StructuredBuffer<DrawArguments> source_arguments;
struct ViewConstants {
    column_major float4x4 view_projection;
    column_major float4x4 previous_view_projection;
    float4 camera_right;
    float4 camera_up;
    float4 material_tint;
    float4 depth_reconstruct;
    float4 lighting_control;
    float4 rendering_control;
    float4 alignment_reference;
};
INX_ROOT_CONSTANTS ConstantBuffer<ViewConstants> view;
static const uint endpoint_for_vertex[6] = {0u, 0u, 1u, 0u, 1u, 1u};
static const float side_for_vertex[6] = {-1.0, 1.0, 1.0, -1.0, 1.0, -1.0};
bool ribbon_points_connected(uint4 first, uint4 second) {
    return first.x == second.x && (second.z & 1u) == 0u;
}
float3 ribbon_side(float3 first, float3 second, float3 fallback_side) {
    float3 tangent = second - first;
    float tangent_length = length(tangent);
    tangent = tangent_length > 1e-7 ? tangent / tangent_length : normalize(view.camera_up.xyz);
    float3 camera_forward = cross(view.camera_right.xyz, view.camera_up.xyz);
    float3 side = cross(camera_forward, tangent);
    float side_length = length(side);
    return side_length > 1e-7 ? side / side_length : fallback_side;
}
float3 ribbon_join_offset(float3 first_side, float3 second_side, float half_width) {
    float3 sum = first_side + second_side;
    float sum_length = length(sum);
    if (sum_length <= 1e-5) return second_side * half_width;
    float3 join = sum / sum_length;
    float projection = max(abs(dot(join, second_side)), 1e-4);
    return join * (half_width * min(1.0 / projection, 2.0));
}
struct RibbonSegment {
    ParticleRenderInstance first;
    ParticleRenderInstance second;
    ParticleRenderInstance neighbor;
    uint endpoint;
    uint corner;
    bool connected;
    bool neighbor_connected;
};
RibbonSegment ribbon_segment(uint vertex, uint instance) {
    bool compacted = view.alignment_reference.z > 0.5;
    uint segment = compacted ? visible_segments[instance] : vertex / 6u;
    RibbonSegment result;
    result.corner = vertex % 6u;
    result.endpoint = endpoint_for_vertex[result.corner];
    result.first = instances[sorted_indices[segment]];
    result.second = instances[sorted_indices[segment + 1u]];
    result.connected = ribbon_points_connected(result.first.ribbon_data, result.second.ribbon_data);
    result.neighbor = result.first;
    result.neighbor_connected = false;
    if (result.connected && result.endpoint == 0u && segment > 0u) {
        result.neighbor = instances[sorted_indices[segment - 1u]];
        result.neighbor_connected = ribbon_points_connected(result.neighbor.ribbon_data, result.first.ribbon_data);
    } else if (result.connected && result.endpoint == 1u && segment + 2u < source_arguments[0].instance_count) {
        result.neighbor = instances[sorted_indices[segment + 2u]];
        result.neighbor_connected = ribbon_points_connected(result.second.ribbon_data, result.neighbor.ribbon_data);
    }
    return result;
}
float3 ribbon_point_position(ParticleRenderInstance point, bool previous) {
    return previous ? point.previous_position_history.xyz : point.position_size.xyz;
}
struct RibbonFrame {
    float3 position;
    float3 tangent;
    float3 side;
};
RibbonFrame ribbon_frame(RibbonSegment segment, bool previous) {
    float3 first = ribbon_point_position(segment.first, previous);
    float3 second = segment.connected ? ribbon_point_position(segment.second, previous) : first;
    ParticleRenderInstance point = segment.endpoint == 0u ? segment.first : segment.second;
    float3 center = segment.endpoint == 0u ? first : second;
    float half_width = abs(point.position_size.w) * 0.5;
    RibbonFrame result;
    result.side = ribbon_side(first, second, normalize(view.camera_right.xyz));
    result.tangent = normalize(second - first + normalize(view.camera_up.xyz) * 1e-8);
    float3 offset = result.side * half_width;
    if (segment.neighbor_connected) {
        float3 neighbor = ribbon_point_position(segment.neighbor, previous);
        if (segment.endpoint == 0u)
            offset = ribbon_join_offset(ribbon_side(neighbor, first, result.side), result.side, half_width);
        else
            offset = ribbon_join_offset(result.side, ribbon_side(second, neighbor, result.side), half_width);
    }
    result.position = center + offset * side_for_vertex[segment.corner];
    return result;
}
)slang";

const std::string VertexSource = std::string(RibbonGeometrySource) + R"slang(struct VertexOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_position : TEXCOORD0;
    [[vk::location(1)]] float3 normal : TEXCOORD1;
    [[vk::location(2)]] float4 tangent : TEXCOORD2;
    [[vk::location(3)]] float3 color : TEXCOORD3;
    [[vk::location(4)]] float2 uv : TEXCOORD4;
    [[vk::location(5)]] float view_depth : TEXCOORD5;
    [[vk::location(6)]] float4 line_color : TEXCOORD6;
    [[vk::location(9)]] float2 particle_local_uv : TEXCOORD9;
    [[vk::location(10)]] float2 particle_next_uv : TEXCOORD10;
    [[vk::location(11)]] float particle_blend : TEXCOORD11;
    [[vk::location(12)]] float particle_age : TEXCOORD12;
    [[vk::location(13)]] nointerpolation uint particle_id : TEXCOORD13;
    [[vk::location(14)]] float particle_alpha : TEXCOORD14;
    [[vk::location(15)]] nointerpolation uint layer_mask : TEXCOORD15;
};
[shader("vertex")]
VertexOutput main(uint vertex : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID) {
    RibbonSegment segment = ribbon_segment(vertex, instance);
    RibbonFrame frame = ribbon_frame(segment, false);
    ParticleRenderInstance point = segment.endpoint == 0u ? segment.first : segment.second;
    VertexOutput output;
    output.position = mul(view.view_projection, float4(frame.position, 1.0));
    output.position.y = -output.position.y;
    float order_coordinate = float(point.ribbon_data.y);
    float uv_scale = max(view.alignment_reference.x, 1e-6);
    output.uv = float2(view.alignment_reference.y > 0.5 ? order_coordinate * uv_scale : order_coordinate / uv_scale,
                       side_for_vertex[segment.corner] > 0.0 ? 1.0 : 0.0);
    output.world_position = frame.position;
    output.normal = normalize(cross(frame.side, frame.tangent));
    output.tangent = float4(frame.tangent, 1.0);
    output.color = point.color.rgb;
    output.view_depth = output.position.w;
    output.line_color = float4(1.0);
    output.particle_local_uv = float2(float(segment.endpoint), side_for_vertex[segment.corner] > 0.0 ? 1.0 : 0.0);
    output.particle_next_uv = output.particle_local_uv;
    output.particle_blend = 0.0;
    output.particle_age = saturate(point.scale_custom.w);
    output.particle_id = point.ribbon_data.w;
    output.particle_alpha = segment.connected ? point.color.a : 0.0;
    output.layer_mask = asuint(view.lighting_control.w);
    return output;
}
)slang";

const std::string PickingFragmentSource = std::string(RibbonGeometrySource) + R"slang([shader("fragment")]
uint2 main([[vk::location(14)]] float particle_alpha : TEXCOORD14) : SV_Target0 {
    if (!(particle_alpha > 0.0)) discard;
    return uint2(asuint(view.material_tint.x), asuint(view.material_tint.y));
}
)slang";

const std::string MotionVertexSource = std::string(RibbonGeometrySource) + R"slang(struct MotionOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float4 color : TEXCOORD0;
    [[vk::location(1)]] float2 motion : TEXCOORD1;
};
[shader("vertex")]
MotionOutput main(uint vertex : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID) {
    RibbonSegment segment = ribbon_segment(vertex, instance);
    RibbonFrame current = ribbon_frame(segment, false);
    RibbonFrame previous = ribbon_frame(segment, true);
    ParticleRenderInstance point = segment.endpoint == 0u ? segment.first : segment.second;
    float4 current_clip = mul(view.view_projection, float4(current.position, 1.0));
    float4 previous_clip = mul(view.previous_view_projection, float4(previous.position, 1.0));
    MotionOutput output;
    output.position = current_clip;
    output.position.y = -output.position.y;
    float2 current_ndc = current_clip.xy / max(abs(current_clip.w), 1e-6);
    float2 previous_ndc = previous_clip.xy / max(abs(previous_clip.w), 1e-6);
    output.motion = (current_ndc - previous_ndc) * float2(0.5, -0.5);
    output.color = point.color * view.material_tint;
    if (!segment.connected) output.color.a = 0.0;
    return output;
}
)slang";

constexpr std::string_view MotionFragmentSource = R"slang(struct MotionInput {
    [[vk::location(0)]] float4 color : TEXCOORD0;
    [[vk::location(1)]] float2 motion : TEXCOORD1;
};
[shader("fragment")]
float2 main(MotionInput input) : SV_Target0 {
    if (input.color.a <= 0.0) discard;
    return input.motion;
}
)slang";

std::vector<uint32_t> CopySpirvWords(const std::vector<char> &bytes)
{
    if (bytes.empty() || bytes.size() % sizeof(uint32_t) != 0)
        return {};
    std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
    std::memcpy(words.data(), bytes.data(), bytes.size());
    return words;
}

uint8_t PipelineStateSignature(const GpuBillboardMaterialState &state) noexcept
{
    return static_cast<uint8_t>((state.blendEnabled ? 1u : 0u) | (state.depthTestEnabled ? 2u : 0u) |
                                (state.depthWriteEnabled ? 4u : 0u) | (state.premultipliedAlpha ? 8u : 0u));
}

} // namespace

std::string_view GpuParticleRibbonRenderShaderSources::Vertex() noexcept
{
    return VertexSource;
}

std::string_view GpuParticleRibbonRenderShaderSources::PickingFragment() noexcept
{
    return PickingFragmentSource;
}

std::string_view GpuParticleRibbonRenderShaderSources::MotionVertex() noexcept
{
    return MotionVertexSource;
}

std::string_view GpuParticleRibbonRenderShaderSources::MotionFragment() noexcept
{
    return MotionFragmentSource;
}

bool GpuParticleRibbonRenderProgram::IsValid() const noexcept
{
    return vertex.IsSpirV() && pickingFragment.IsSpirV() && motionVertex.IsSpirV() && motionFragment.IsSpirV();
}

bool GpuParticleRibbonRenderProgramStorage::Assign(const GpuParticleRibbonRenderProgram &program)
{
    if (!program.IsValid())
        return false;
    const std::array<ShaderBytecode, 4> sources = {program.vertex, program.pickingFragment, program.motionVertex,
                                                   program.motionFragment};
    std::array<ShaderBytecodeStorage, 4> candidate;
    for (size_t index = 0; index < sources.size(); ++index)
        candidate[index].Assign(sources[index]);
    shaders = std::move(candidate);
    return true;
}

bool GpuParticleRibbonRenderProgramStorage::IsValid() const noexcept
{
    return View().IsValid();
}

GpuParticleRibbonRenderProgram GpuParticleRibbonRenderProgramStorage::View() const noexcept
{
    return {
        shaders[0].View(),
        shaders[1].View(),
        shaders[2].View(),
        shaders[3].View(),
    };
}

ParticleGpuRibbonRenderer::~ParticleGpuRibbonRenderer()
{
    Destroy();
}

bool ParticleGpuRibbonRenderer::Create(rhi::Device &device, const GpuRibbonRendererDesc &desc)
{
    Destroy();
    if (!desc.program.IsValid() || !desc.topology || !desc.topology->IsValid() || !std::isfinite(desc.uvScale) ||
        desc.uvScale <= 0.0f || !desc.semantics.IsValid() || desc.semantics.castShadows || !desc.shaderProgram ||
        !desc.shaderProgram->IsValid() || desc.shaderProgram->domain != ShaderProgramDomain::ParticleSprite ||
        !desc.textureResolver) {
        return false;
    }
    const auto *linkedVariant = desc.shaderProgram->FindVariant(ShaderCompileTarget::Forward);
    const auto *linkedForwardPlusVariant = desc.shaderProgram->FindVariant(ShaderCompileTarget::ForwardPlus);
    if (!linkedVariant || (desc.semantics.receiveSceneLighting && !linkedForwardPlusVariant))
        return false;
    const auto linkedFragmentWords = CopySpirvWords(linkedVariant->fragmentSpirv);
    const auto linkedForwardPlusFragmentWords =
        linkedForwardPlusVariant ? CopySpirvWords(linkedForwardPlusVariant->fragmentSpirv) : std::vector<uint32_t>{};
    if (linkedFragmentWords.empty() || (desc.semantics.receiveSceneLighting && linkedForwardPlusFragmentWords.empty()))
        return false;

    m_device = &device;
    m_topology = desc.topology;
    m_shaderProgram = desc.shaderProgram;
    m_semantics = desc.semantics;
    m_semantics.sortMode = ParticleSortMode::None;
    m_uvMode = desc.uvMode;
    m_uvScale = desc.uvScale;
    m_vertexShader = device.CreateShaderModule(
        rhi::ShaderModuleDesc::FromSpirV(desc.program.vertex.words, desc.program.vertex.wordCount));
    m_fragmentShader = device.CreateShaderModule(
        rhi::ShaderModuleDesc::FromSpirV(linkedFragmentWords.data(), linkedFragmentWords.size()));
    if (m_semantics.receiveSceneLighting) {
        m_forwardPlusFragmentShader = device.CreateShaderModule(rhi::ShaderModuleDesc::FromSpirV(
            linkedForwardPlusFragmentWords.data(), linkedForwardPlusFragmentWords.size()));
    }
    m_pickingFragmentShader = device.CreateShaderModule(
        rhi::ShaderModuleDesc::FromSpirV(desc.program.pickingFragment.words, desc.program.pickingFragment.wordCount));
    m_motionVertexShader = device.CreateShaderModule(
        rhi::ShaderModuleDesc::FromSpirV(desc.program.motionVertex.words, desc.program.motionVertex.wordCount));
    m_motionFragmentShader = device.CreateShaderModule(
        rhi::ShaderModuleDesc::FromSpirV(desc.program.motionFragment.words, desc.program.motionFragment.wordCount));
    if (!m_vertexShader.IsValid() || !m_fragmentShader.IsValid() || !m_pickingFragmentShader.IsValid() ||
        !m_motionVertexShader.IsValid() || !m_motionFragmentShader.IsValid() ||
        (m_semantics.receiveSceneLighting && !m_forwardPlusFragmentShader.IsValid())) {
        Destroy();
        return false;
    }
    rhi::BindingLayoutDesc layout;
    layout.entries[layout.entryCount++] = {0, rhi::BindingType::StorageBuffer, rhi::ShaderStage::Vertex, 1};
    layout.entries[layout.entryCount++] = {1, rhi::BindingType::StorageBuffer, rhi::ShaderStage::Vertex, 1};
    layout.entries[layout.entryCount++] = {2, rhi::BindingType::StorageBuffer, rhi::ShaderStage::Vertex, 1};
    layout.entries[layout.entryCount++] = {3, rhi::BindingType::StorageBuffer, rhi::ShaderStage::Vertex, 1};
    for (uint32_t index = 0; index < layout.entryCount; ++index)
        layout.entries[index].readOnlyStorage = true;
    m_geometryLayout = device.CreateBindingLayout(layout);
    m_emptyLayout = device.CreateBindingLayout({});
    rhi::BindGroupDesc emptyGroupDesc;
    emptyGroupDesc.layout = m_emptyLayout;
    m_emptyGroup = device.CreateBindGroup(emptyGroupDesc);
    if (!m_geometryLayout.IsValid() || !m_emptyLayout.IsValid() || !m_emptyGroup.IsValid() ||
        !m_surface.Create(device, desc.shaderProgram, desc.material, desc.fallbackMaterial, desc.semantics,
                          desc.textureResolver, desc.deletionQueue)) {
        Destroy();
        return false;
    }
    m_geometryGroup = CreateGeometryGroup(m_topology->SortedIndexBuffer());
    if (!m_geometryGroup.IsValid()) {
        Destroy();
        return false;
    }
    return true;
}

void ParticleGpuRibbonRenderer::Destroy() noexcept
{
    if (m_device) {
        for (const auto &entry : m_pipelines)
            m_device->Release(entry.pipeline);
        for (const auto &entry : m_viewGroups)
            m_device->Release(entry.group);
        m_device->Release(m_geometryGroup);
        m_device->Release(m_emptyGroup);
        m_device->Release(m_geometryLayout);
        m_device->Release(m_emptyLayout);
        m_device->Release(m_pickingFragmentShader);
        m_device->Release(m_motionFragmentShader);
        m_device->Release(m_motionVertexShader);
        m_device->Release(m_forwardPlusFragmentShader);
        m_device->Release(m_fragmentShader);
        m_device->Release(m_vertexShader);
    }
    m_surface.Destroy();
    m_device = nullptr;
    m_topology.reset();
    m_shaderProgram.reset();
    m_semantics = {};
    m_uvMode = ParticleRibbonUvMode::Stretch;
    m_uvScale = 1.0f;
    m_vertexShader = {};
    m_fragmentShader = {};
    m_forwardPlusFragmentShader = {};
    m_pickingFragmentShader = {};
    m_motionVertexShader = {};
    m_motionFragmentShader = {};
    m_geometryLayout = {};
    m_geometryGroup = {};
    m_emptyLayout = {};
    m_emptyGroup = {};
    m_viewGroups.clear();
    m_pipelines.clear();
}

bool ParticleGpuRibbonRenderer::IsValid() const noexcept
{
    return m_device && m_topology && m_topology->IsValid() && m_vertexShader.IsValid() && m_fragmentShader.IsValid() &&
           m_pickingFragmentShader.IsValid() && m_motionVertexShader.IsValid() && m_motionFragmentShader.IsValid() &&
           m_geometryLayout.IsValid() && m_geometryGroup.IsValid() && m_emptyLayout.IsValid() &&
           m_emptyGroup.IsValid() && m_surface.IsValid() &&
           (!m_semantics.receiveSceneLighting || m_forwardPlusFragmentShader.IsValid());
}

int32_t ParticleGpuRibbonRenderer::RenderQueue() const noexcept
{
    return m_surface.ResolveMaterialState().renderQueue;
}

rhi::BufferHandle ParticleGpuRibbonRenderer::InstanceBuffer() const noexcept
{
    return m_topology ? m_topology->InstanceBuffer() : rhi::BufferHandle{};
}

rhi::BufferHandle ParticleGpuRibbonRenderer::RenderIndexBuffer() const noexcept
{
    return m_topology ? m_topology->SortedIndexBuffer() : rhi::BufferHandle{};
}

rhi::BindGroupHandle ParticleGpuRibbonRenderer::CreateGeometryGroup(rhi::BufferHandle renderIndices) const
{
    if (!m_device || !m_geometryLayout.IsValid() || !InstanceBuffer().IsValid() || !renderIndices.IsValid() ||
        !m_topology || !m_topology->SourceIndirectBuffer().IsValid())
        return {};
    rhi::BindGroupDesc group;
    group.layout = m_geometryLayout;
    group.buffers[group.bufferCount++] = {0, rhi::BindingType::StorageBuffer, InstanceBuffer(), 0, 0};
    group.buffers[group.bufferCount++] = {1, rhi::BindingType::StorageBuffer, RenderIndexBuffer(), 0, 0};
    group.buffers[group.bufferCount++] = {2, rhi::BindingType::StorageBuffer, renderIndices, 0, 0};
    group.buffers[group.bufferCount++] = {3, rhi::BindingType::StorageBuffer, m_topology->SourceIndirectBuffer(), 0, 0};
    return m_device->CreateBindGroup(group);
}

void ParticleGpuRibbonRenderer::ReleaseViewBinding(rhi::BufferHandle renderIndices) noexcept
{
    const auto binding = std::find_if(m_viewGroups.begin(), m_viewGroups.end(),
                                      [&](const auto &entry) { return entry.renderIndices == renderIndices; });
    if (binding == m_viewGroups.end())
        return;
    m_device->Release(binding->group);
    m_viewGroups.erase(binding);
}

rhi::BindGroupHandle ParticleGpuRibbonRenderer::ResolveGeometryGroup(rhi::BufferHandle renderIndices)
{
    if (!renderIndices.IsValid() || renderIndices == RenderIndexBuffer())
        return m_geometryGroup;
    const auto found = std::find_if(m_viewGroups.begin(), m_viewGroups.end(),
                                    [&](const auto &entry) { return entry.renderIndices == renderIndices; });
    if (found != m_viewGroups.end())
        return found->group;
    const auto group = CreateGeometryGroup(renderIndices);
    if (group.IsValid())
        m_viewGroups.push_back({renderIndices, group});
    return group;
}

bool ParticleGpuRibbonRenderer::RecordDraw(const rhi::GraphicsCommandEncoder &encoder,
                                           const MaterialPassPipelineDescriptor &pass,
                                           rhi::BufferHandle indirectArguments, const GpuParticleViewConstants &view,
                                           rhi::BufferHandle renderIndices, rhi::TextureViewHandle sceneDepth,
                                           bool sceneDepthIsDepth, const GpuParticlePerViewBindings &perView)
{
    if (!IsValid() || !encoder.IsValid() || !indirectArguments.IsValid() || !m_surface.RefreshMaterialBuffer(false) ||
        !m_surface.RefreshTextureBindings(false))
        return false;
    if (m_semantics.softParticles && !sceneDepth.IsValid())
        return false;
    const bool usesForwardPlusLighting =
        pass.target == ShaderCompileTarget::ForwardPlus && m_semantics.receiveSceneLighting;
    const bool usesPerViewBindings =
        pass.target == ShaderCompileTarget::Forward || pass.target == ShaderCompileTarget::ForwardPlus;
    if (usesPerViewBindings && !perView.IsValid())
        return false;
    const auto pipeline = GetOrCreatePipeline(pass, usesPerViewBindings ? perView.layout : rhi::BindingLayoutHandle{});
    const auto geometryGroup = ResolveGeometryGroup(renderIndices);
    const auto surfaceGroup = m_surface.ResolveBindGroup(sceneDepth, sceneDepthIsDepth);
    const bool usesBindlessTextures =
        m_surface.UsesBindlessTextures() &&
        (pass.target == ShaderCompileTarget::Forward || pass.target == ShaderCompileTarget::ForwardPlus);
    const auto bindlessTable = m_surface.BindlessTableBinding();
    if (!pipeline.IsValid() || !geometryGroup.IsValid() || !surfaceGroup.IsValid() ||
        (usesBindlessTextures && !bindlessTable.IsValid()))
        return false;
    auto constants = view;
    constants.materialTint = {1.0f, 1.0f, 1.0f, 1.0f};
    constants.cameraRight[3] = m_semantics.softDistance;
    constants.cameraUp[3] = m_semantics.softParticles ? 1.0f : 0.0f;
    constants.lightingControl[0] = usesForwardPlusLighting ? 1.0f : 0.0f;
    constants.lightingControl[1] = 0.0f;
    constants.lightingControl[2] = m_surface.ResolveMaterialFloat("softness", 0.18f);
    constants.renderingControl[0] = m_semantics.receiveShadows ? 1.0f : 0.0f;
    constants.renderingControl[1] = m_surface.ResolveMaterialState().premultipliedAlpha ? 1.0f : 0.0f;
    constants.renderingControl[2] = 1.0f;
    constants.renderingControl[3] = 1.0f;
    constants.alignmentReference[0] = m_uvScale;
    constants.alignmentReference[1] = m_uvMode == ParticleRibbonUvMode::Repeat ? 1.0f : 0.0f;
    constants.alignmentReference[2] = renderIndices.IsValid() && renderIndices != RenderIndexBuffer() ? 1.0f : 0.0f;
    constants.alignmentReference[3] = -1.0f;
    encoder.BindPipeline(pipeline);
    encoder.BindGroup(pipeline, 0, geometryGroup);
    encoder.BindGroup(pipeline, 1, usesPerViewBindings ? perView.group : m_emptyGroup);
    encoder.BindGroup(pipeline, 2, surfaceGroup);
    if (usesBindlessTextures) {
        encoder.BindGroup(pipeline, 3, bindlessTable.group);
        m_surface.MarkBindlessTexturesUsed();
    }
    encoder.PushConstants(pipeline, rhi::ShaderStage::Vertex | rhi::ShaderStage::Fragment, sizeof(constants),
                          &constants);
    encoder.DrawIndirect(indirectArguments);
    return true;
}

bool ParticleGpuRibbonRenderer::RecordPickingDraw(const rhi::GraphicsCommandEncoder &encoder,
                                                  const MaterialPassPipelineDescriptor &pass,
                                                  rhi::BufferHandle indirectArguments,
                                                  const GpuParticleViewConstants &view, uint64_t ownerObjectId,
                                                  rhi::BufferHandle renderIndices)
{
    if (!IsValid() || !encoder.IsValid() || !indirectArguments.IsValid() || ownerObjectId == 0)
        return false;
    const auto pipeline = GetOrCreatePipeline(pass);
    const auto geometryGroup = ResolveGeometryGroup(renderIndices);
    const auto surfaceGroup = m_surface.ResolveBindGroup();
    if (!pipeline.IsValid() || !geometryGroup.IsValid() || !surfaceGroup.IsValid())
        return false;
    auto constants = view;
    const std::array<uint32_t, 4> objectId = {
        static_cast<uint32_t>(ownerObjectId),
        static_cast<uint32_t>(ownerObjectId >> 32u),
        0u,
        0x3f800000u,
    };
    std::memcpy(constants.materialTint.data(), objectId.data(), sizeof(objectId));
    constants.renderingControl[1] = 0.0f;
    constants.renderingControl[2] = 1.0f;
    constants.renderingControl[3] = 1.0f;
    constants.alignmentReference[0] = m_uvScale;
    constants.alignmentReference[1] = m_uvMode == ParticleRibbonUvMode::Repeat ? 1.0f : 0.0f;
    constants.alignmentReference[2] = renderIndices.IsValid() && renderIndices != RenderIndexBuffer() ? 1.0f : 0.0f;
    constants.alignmentReference[3] = -1.0f;
    encoder.BindPipeline(pipeline);
    encoder.BindGroup(pipeline, 0, geometryGroup);
    encoder.BindGroup(pipeline, 1, m_emptyGroup);
    encoder.BindGroup(pipeline, 2, surfaceGroup);
    encoder.PushConstants(pipeline, rhi::ShaderStage::Vertex | rhi::ShaderStage::Fragment, sizeof(constants),
                          &constants);
    encoder.DrawIndirect(indirectArguments);
    return true;
}

rhi::GraphicsPipelineHandle ParticleGpuRibbonRenderer::GetOrCreatePipeline(const MaterialPassPipelineDescriptor &pass,
                                                                           rhi::BindingLayoutHandle perViewLayout)
{
    if (!pass.IsValid() ||
        (pass.target != ShaderCompileTarget::Forward && pass.target != ShaderCompileTarget::ForwardPlus &&
         pass.target != ShaderCompileTarget::Picking && pass.target != ShaderCompileTarget::Motion))
        return {};
    const bool picking = pass.target == ShaderCompileTarget::Picking;
    const bool motion = pass.target == ShaderCompileTarget::Motion;
    const bool usesForwardPlusLighting =
        pass.target == ShaderCompileTarget::ForwardPlus && m_semantics.receiveSceneLighting;
    const bool usesPerViewBindings =
        pass.target == ShaderCompileTarget::Forward || pass.target == ShaderCompileTarget::ForwardPlus;
    if (usesPerViewBindings && !perViewLayout.IsValid())
        return {};
    const auto state = picking  ? GpuBillboardMaterialState{3000, false, true, true}
                       : motion ? GpuBillboardMaterialState{3000, false, true, false}
                                : m_surface.ResolveMaterialState();
    const uint8_t signature = PipelineStateSignature(state);
    const auto found = std::find_if(m_pipelines.begin(), m_pipelines.end(), [&](const auto &entry) {
        return entry.pass == pass && entry.perViewLayout == perViewLayout && entry.materialStateSignature == signature;
    });
    if (found != m_pipelines.end())
        return found->pipeline;

    rhi::GraphicsPipelineDesc desc;
    desc.vertexShader = motion ? m_motionVertexShader : m_vertexShader;
    desc.fragmentShader = picking  ? m_pickingFragmentShader
                          : motion ? m_motionFragmentShader
                                   : (usesForwardPlusLighting ? m_forwardPlusFragmentShader : m_fragmentShader);
    pass.ApplyRenderingContract(desc);
    desc.raster.cullMode = rhi::CullMode::None;
    pass.ApplyRasterContract(desc);
    desc.depth.testEnabled = state.depthTestEnabled && pass.depthFormat != rhi::PixelFormat::Undefined;
    desc.depth.writeEnabled =
        state.depthWriteEnabled && !pass.depthReadOnly && pass.depthFormat != rhi::PixelFormat::Undefined;
    desc.samples = pass.samples;
    for (size_t index = 0; index < pass.colorFormats.size(); ++index) {
        desc.colorTargets[index].format = pass.colorFormats[index];
        desc.colorTargets[index].blendEnabled = state.blendEnabled;
        desc.colorTargets[index].premultipliedAlpha = state.premultipliedAlpha;
    }
    desc.colorTargetCount = static_cast<uint32_t>(pass.colorFormats.size());
    desc.bindingLayouts[0] = m_geometryLayout;
    desc.bindingLayouts[1] = usesPerViewBindings ? perViewLayout : m_emptyLayout;
    desc.bindingLayouts[2] = m_surface.Layout();
    if (!picking && !motion && m_surface.UsesBindlessTextures())
        desc.bindingLayouts[3] = m_surface.BindlessTableBinding().layout;
    desc.bindingLayoutCount = !picking && !motion && m_surface.UsesBindlessTextures() ? 4 : 3;
    desc.pushConstantStages = rhi::ShaderStage::Vertex | rhi::ShaderStage::Fragment;
    desc.pushConstantBytes = sizeof(GpuParticleViewConstants);
    const auto pipeline = m_device->CreateGraphicsPipeline(desc);
    if (pipeline.IsValid())
        m_pipelines.push_back({pass, perViewLayout, signature, pipeline});
    return pipeline;
}

} // namespace infernux::particle
