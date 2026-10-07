#include "ForwardPlusLightGrid.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace infernux::lighting
{

namespace
{

constexpr uint64_t TileHeaderBytes = 16;

uint64_t GrowCapacity(uint64_t required)
{
    uint64_t capacity = 256;
    while (capacity < required) {
        if (capacity > std::numeric_limits<uint64_t>::max() / 2u)
            return required;
        capacity *= 2u;
    }
    return capacity;
}

bool MultiplyFits(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
        return false;
    result = lhs * rhs;
    return true;
}

} // namespace

bool ForwardPlusGridConfig::IsValid() const noexcept
{
    return width > 0 && height > 0 && tileCountX > 0 && tileCountY > 0 && tileCount > 0 && maskWordStride > 0 &&
           domainMask != 0 && headerBytes >= (static_cast<uint64_t>(tileCount) + 1u) * TileHeaderBytes &&
           maskBytes >= static_cast<uint64_t>(tileCount) * maskWordStride * sizeof(uint32_t);
}

ForwardPlusGridConfig BuildForwardPlusGridConfig(uint32_t width, uint32_t height, uint32_t localLightCount,
                                                 uint32_t domainMask)
{
    ForwardPlusGridConfig config;
    if (width == 0 || height == 0 || domainMask == 0)
        return config;

    config.width = width;
    config.height = height;
    config.tileCountX = (width + ForwardPlusLightGrid::TileSize - 1u) / ForwardPlusLightGrid::TileSize;
    config.tileCountY = (height + ForwardPlusLightGrid::TileSize - 1u) / ForwardPlusLightGrid::TileSize;
    const uint64_t tileCount = static_cast<uint64_t>(config.tileCountX) * config.tileCountY;
    if (tileCount > std::numeric_limits<uint32_t>::max())
        return {};

    config.tileCount = static_cast<uint32_t>(tileCount);
    config.localLightCount = localLightCount;
    // One bit represents one local light. A zero-light view still owns one
    // word, which keeps the descriptor ABI valid without a dummy resource.
    const uint32_t maskWords = localLightCount / 32u + (localLightCount % 32u != 0u ? 1u : 0u);
    config.maskWordStride = std::max(maskWords, 1u);
    config.domainMask = domainMask;
    if (!MultiplyFits(tileCount + 1u, TileHeaderBytes, config.headerBytes))
        return {};
    uint64_t maskWordCount = 0;
    if (!MultiplyFits(tileCount, config.maskWordStride, maskWordCount) ||
        !MultiplyFits(maskWordCount, sizeof(uint32_t), config.maskBytes)) {
        return {};
    }
    return config;
}

bool ForwardPlusGridProgram::IsValid() const noexcept
{
    return words && wordCount >= 5 && words[0] == 0x07230203u;
}

ForwardPlusLightGrid::~ForwardPlusLightGrid()
{
    Shutdown();
}

bool ForwardPlusLightGrid::Initialize(rhi::Device &device, uint32_t framesInFlight,
                                      const ForwardPlusGridProgram &program)
{
    Shutdown();
    if (framesInFlight == 0 || !program.IsValid())
        return false;

    m_device = &device;
    rhi::BindingLayoutDesc layoutDesc;
    for (uint32_t binding = 0; binding < 3; ++binding)
        layoutDesc.entries[binding] = {binding, rhi::BindingType::StorageBuffer, rhi::ShaderStage::Compute, 1};
    layoutDesc.entryCount = 3;
    m_layout = device.CreateBindingLayout(layoutDesc);
    rhi::BindingLayoutDesc consumerLayoutDesc;
    for (uint32_t binding = 0; binding < 3; ++binding) {
        consumerLayoutDesc.entries[binding] = {binding, rhi::BindingType::StorageBuffer, rhi::ShaderStage::Fragment, 1};
    }
    consumerLayoutDesc.entryCount = 3;
    m_consumerLayout = device.CreateBindingLayout(consumerLayoutDesc);

    const auto shader = device.CreateShaderModule(rhi::ShaderModuleDesc::FromSpirV(program.words, program.wordCount));
    if (m_layout.IsValid() && m_consumerLayout.IsValid() && shader.IsValid()) {
        rhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.computeShader = shader;
        pipelineDesc.bindingLayouts[0] = m_layout;
        pipelineDesc.bindingLayoutCount = 1;
        pipelineDesc.pushConstantBytes = sizeof(ForwardPlusGridConstants);
        m_pipeline = device.CreateComputePipeline(pipelineDesc);
    }
    device.Release(shader);
    if (!m_layout.IsValid() || !m_pipeline.IsValid()) {
        Shutdown();
        return false;
    }

    m_frames.resize(framesInFlight);
    return true;
}

void ForwardPlusLightGrid::Shutdown() noexcept
{
    if (m_device) {
        for (const auto &retired : m_retired) {
            m_device->Release(retired.bindGroup);
            m_device->Release(retired.consumerBindGroup);
            m_device->Release(retired.lightMasks);
            m_device->Release(retired.headers);
        }
        for (auto &frame : m_frames) {
            m_device->Release(frame.bindGroup);
            m_device->Release(frame.consumerBindGroup);
            m_device->Release(frame.lightMasks);
            m_device->Release(frame.headers);
        }
        m_device->Release(m_pipeline);
        m_device->Release(m_layout);
        m_device->Release(m_consumerLayout);
    }
    m_frames.clear();
    m_retired.clear();
    m_pipeline = {};
    m_layout = {};
    m_consumerLayout = {};
    m_device = nullptr;
}

bool ForwardPlusLightGrid::PrepareFrame(uint32_t frameIndex, uint32_t width, uint32_t height, uint32_t localLightCount,
                                        uint32_t domainMask, rhi::BufferHandle canonicalLights)
{
    if (!IsValid() || frameIndex >= m_frames.size() || !canonicalLights.IsValid())
        return false;

    const ForwardPlusGridConfig config = BuildForwardPlusGridConfig(width, height, localLightCount, domainMask);
    if (!config.IsValid())
        return false;

    auto &frame = m_frames[frameIndex];
    bool resourcesChanged = false;
    if (!frame.headers.IsValid() || frame.headerCapacityBytes < config.headerBytes) {
        const uint64_t capacity = GrowCapacity(config.headerBytes);
        const auto replacement = m_device->CreateBuffer({capacity, rhi::BufferUsageFlags::Storage});
        if (!replacement.IsValid())
            return false;
        m_retired.push_back({frame.bindGroup, frame.consumerBindGroup, frame.headers, {}});
        frame.bindGroup = {};
        frame.consumerBindGroup = {};
        frame.headers = replacement;
        frame.headerCapacityBytes = capacity;
        resourcesChanged = true;
    }
    if (!frame.lightMasks.IsValid() || frame.maskCapacityBytes < config.maskBytes) {
        const uint64_t capacity = GrowCapacity(config.maskBytes);
        const auto replacement = m_device->CreateBuffer({capacity, rhi::BufferUsageFlags::Storage});
        if (!replacement.IsValid())
            return false;
        if (frame.bindGroup.IsValid())
            m_retired.push_back({frame.bindGroup, frame.consumerBindGroup, {}, {}});
        frame.bindGroup = {};
        frame.consumerBindGroup = {};
        if (frame.lightMasks.IsValid())
            m_retired.push_back({{}, {}, {}, frame.lightMasks});
        frame.lightMasks = replacement;
        frame.maskCapacityBytes = capacity;
        resourcesChanged = true;
    }

    frame.config = config;
    if (resourcesChanged || frame.canonicalLights != canonicalLights || !frame.bindGroup.IsValid() ||
        !frame.consumerBindGroup.IsValid())
        return RebuildBindGroup(frame, canonicalLights);
    return true;
}

void ForwardPlusLightGrid::Record(uint32_t frameIndex, const rhi::ComputeCommandEncoder &encoder,
                                  const ForwardPlusGridConstants &constants) const
{
    if (!IsValid() || !encoder.IsValid() || frameIndex >= m_frames.size())
        return;
    const auto &frame = m_frames[frameIndex];
    if (!frame.config.IsValid() || !frame.bindGroup.IsValid())
        return;

    ForwardPlusGridConstants resolved = constants;
    resolved.viewportAndProjectionScale[0] = static_cast<float>(frame.config.width);
    resolved.viewportAndProjectionScale[1] = static_cast<float>(frame.config.height);
    resolved.gridAndLights[0] = frame.config.tileCountX;
    resolved.gridAndLights[1] = frame.config.tileCountY;
    resolved.gridAndLights[2] = frame.config.localLightCount;
    resolved.gridAndLights[3] = TileSize;
    resolved.domainAndMaskWords[0] = frame.config.domainMask;
    resolved.domainAndMaskWords[1] = frame.config.maskWordStride;

    encoder.BindPipeline(m_pipeline);
    encoder.BindGroup(m_pipeline, 0, frame.bindGroup);
    encoder.PushConstants(m_pipeline, sizeof(resolved), &resolved);
    encoder.Dispatch(frame.config.tileCountX, frame.config.tileCountY, 1);
}

bool ForwardPlusLightGrid::IsValid() const noexcept
{
    return m_device && m_layout.IsValid() && m_consumerLayout.IsValid() && m_pipeline.IsValid() && !m_frames.empty();
}

uint32_t ForwardPlusLightGrid::FrameCount() const noexcept
{
    return static_cast<uint32_t>(m_frames.size());
}

const ForwardPlusGridFrame &ForwardPlusLightGrid::Frame(uint32_t frameIndex) const
{
    if (frameIndex >= m_frames.size())
        throw std::out_of_range("Forward+ light-grid frame index is out of range");
    return m_frames[frameIndex];
}

std::vector<ForwardPlusRetiredResources> ForwardPlusLightGrid::TakeRetiredResources()
{
    std::vector<ForwardPlusRetiredResources> result;
    result.swap(m_retired);
    return result;
}

bool ForwardPlusLightGrid::RebuildBindGroup(ForwardPlusGridFrame &frame, rhi::BufferHandle canonicalLights)
{
    if (frame.bindGroup.IsValid())
        m_retired.push_back({frame.bindGroup, frame.consumerBindGroup, {}, {}});
    frame.bindGroup = {};
    frame.consumerBindGroup = {};
    rhi::BindGroupDesc groupDesc;
    groupDesc.layout = m_layout;
    const rhi::BufferHandle buffers[] = {canonicalLights, frame.headers, frame.lightMasks};
    for (uint32_t binding = 0; binding < 3; ++binding)
        groupDesc.buffers[binding] = {binding, rhi::BindingType::StorageBuffer, buffers[binding], 0, 0};
    groupDesc.bufferCount = 3;
    frame.bindGroup = m_device->CreateBindGroup(groupDesc);
    if (!frame.bindGroup.IsValid())
        return false;

    groupDesc.layout = m_consumerLayout;
    frame.consumerBindGroup = m_device->CreateBindGroup(groupDesc);
    if (!frame.consumerBindGroup.IsValid()) {
        m_device->Release(frame.bindGroup);
        frame.bindGroup = {};
        return false;
    }
    frame.canonicalLights = canonicalLights;
    return true;
}

std::string_view ForwardPlusLightGrid::ShaderSource() noexcept
{
    return R"glsl(#version 450
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct CanonicalLightData {
    vec4 position_range;
    vec4 direction_spot;
    vec4 color_intensity;
    vec4 attenuation;
    vec4 area_right_width;
    vec4 area_up_height;
    uvec4 metadata;
    uvec4 identity_shadow;
};

layout(std430, set = 0, binding = 0) readonly buffer CanonicalLights {
    uvec4 counts_generation;
    CanonicalLightData lights[];
};
layout(std430, set = 0, binding = 1) buffer TileHeaders { uvec4 tile_headers[]; };
layout(std430, set = 0, binding = 2) buffer TileLightMasks { uint tile_light_masks[]; };

layout(push_constant) uniform ForwardPlusGridConstants {
    mat4 view_projection;
    vec4 viewport_projection_scale;
    uvec4 grid_lights;
    uvec4 domain_mask_words;
} pc;

// Pull the tile's clip inequalities back to world-space planes. This avoids
// dividing a sphere's center by w and guessing a symmetric projected radius,
// which can exclude lit pixels for off-axis or shifted projections. All light
// lanes share the six planes and their normal lengths, computed once per tile.
shared vec4 tile_planes[6];
shared float tile_plane_lengths[6];

void prepare_tile_plane(uint index, uvec2 tile) {
    mat4 rows = transpose(pc.view_projection);
    vec2 viewport = pc.viewport_projection_scale.xy;
    vec2 pixel_min = vec2(tile) * float(pc.grid_lights.w);
    vec2 pixel_max = min(pixel_min + float(pc.grid_lights.w), viewport);
    vec2 ndc_min = 2.0 * pixel_min / viewport - 1.0;
    vec2 ndc_max = 2.0 * pixel_max / viewport - 1.0;
    vec4 plane;
    if (index == 0u) plane = rows[0] - ndc_min.x * rows[3];
    else if (index == 1u) plane = ndc_max.x * rows[3] - rows[0];
    else if (index == 2u) plane = rows[1] - ndc_min.y * rows[3];
    else if (index == 3u) plane = ndc_max.y * rows[3] - rows[1];
    // Vulkan clip depth is [0,w], also for reversed or infinite-far matrices.
    else if (index == 4u) plane = rows[2];
    else plane = rows[3] - rows[2];
    tile_planes[index] = plane;
    // Do not normalize: an infinite far plane can have a zero normal.
    tile_plane_lengths[index] = length(plane.xyz);
}

bool overlaps_tile(CanonicalLightData light) {
    vec4 center = vec4(light.position_range.xyz, 1.0);
    float radius = max(light.position_range.w, 0.0);
    if (light.metadata.x == 3u) {
        radius += 0.5 * length(vec2(light.area_right_width.w, light.area_up_height.w));
    }
    for (uint index = 0u; index < 6u; ++index) {
        vec4 plane = tile_planes[index];
        float support = radius * tile_plane_lengths[index];
        // Outward rounding for float dot products keeps a tangent sphere in
        // the list; this is relative to the terms, not a world-unit padding.
        float roundoff = 0.000001 * (dot(abs(plane), abs(center)) + support);
        if (dot(plane, center) + support < -roundoff) return false;
    }
    return true;
}

void main() {
    uvec2 tile = gl_WorkGroupID.xy;
    uint tile_index = tile.y * pc.grid_lights.x + tile.x;
    uint offset = tile_index * pc.domain_mask_words.y;
    if (gl_LocalInvocationIndex < 6u) prepare_tile_plane(gl_LocalInvocationIndex, tile);
    for (uint word = gl_LocalInvocationIndex; word < pc.domain_mask_words.y; word += gl_WorkGroupSize.x) {
        tile_light_masks[offset + word] = 0u;
    }
    // barrier() only synchronizes invocation progress. SSBO writes require an
    // explicit buffer-memory barrier before another invocation in the same
    // workgroup can safely feed the cleared words to atomicOr. Desktop drivers
    // commonly hide this omission; tiled mobile GPUs do not.
    memoryBarrierBuffer();
    barrier();

    uint directional_count = counts_generation.x;
    uint available_local_count = min(counts_generation.y, pc.grid_lights.z);
    for (uint local_index = gl_LocalInvocationIndex; local_index < available_local_count;
         local_index += gl_WorkGroupSize.x) {
        CanonicalLightData light = lights[directional_count + local_index];
        if ((light.metadata.w & pc.domain_mask_words.x) == 0u || !overlaps_tile(light)) continue;
        atomicOr(tile_light_masks[offset + (local_index >> 5u)], 1u << (local_index & 31u));
    }
    memoryBarrierBuffer();
    barrier();
    if (gl_LocalInvocationIndex == 0u) {
        if (tile_index == 0u) {
            tile_headers[0] = uvec4(pc.grid_lights.xy, pc.grid_lights.w, pc.domain_mask_words.x);
        }
        tile_headers[tile_index + 1u] =
            uvec4(offset, pc.domain_mask_words.y, available_local_count, pc.domain_mask_words.x);
    }
}
)glsl";
}

} // namespace infernux::lighting
