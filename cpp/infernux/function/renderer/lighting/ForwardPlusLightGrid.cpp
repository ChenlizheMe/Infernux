#include "ForwardPlusLightGrid.h"
#include <function/renderer/shader/DeviceShaderProgram.h>
#include <function/renderer/shader/ShaderValueCopy.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace infernux::lighting
{

// Geometry and particle views share code and layout, never mutable buffers.
struct ForwardPlusGridPipeline
{
    ForwardPlusGridPipeline(rhi::Device &owner, const ShaderTargetCode &code) : device(owner), shader(owner, code, {0})
    {
        for (const auto &resource : shader.GetResources()) {
            if (resource.name == "pc") {
                constantsBinding = resource.binding.binding;
                constantBytes = resource.value.byteSize;
                constantCopy = std::make_unique<ShaderValueCopy>(resource.value);
            } else if (resource.name == "canonical_lights")
                lightsBinding = resource.binding.binding;
            else if (resource.name == "tile_headers")
                headersBinding = resource.binding.binding;
            else if (resource.name == "tile_light_masks")
                masksBinding = resource.binding.binding;
        }
        rhi::ComputePipelineDesc description;
        shader.Apply(description);
        pipeline = device.CreateComputePipeline(description);
        if (!pipeline.IsValid())
            throw std::runtime_error("Forward+ grid pipeline creation failed");
        // Until the mesh author path is migrated, its published fragment
        // program consumes the canonical three-buffer light-table interface.
        rhi::BindingLayoutDesc consumer;
        for (uint32_t binding = 0; binding < 3; ++binding) {
            consumer.entries[binding] = {binding, rhi::BindingType::StorageBuffer, rhi::ShaderStage::Fragment, 1};
            consumer.entries[binding].readOnlyStorage = true;
        }
        consumer.entryCount = 3;
        consumerLayout = device.CreateBindingLayout(consumer);
    }
    ~ForwardPlusGridPipeline()
    {
        device.Release(pipeline);
        device.Release(consumerLayout);
    }
    rhi::Device &device;
    DeviceShaderProgram shader;
    rhi::BindingLayoutHandle consumerLayout;
    rhi::ComputePipelineHandle pipeline;
    uint32_t constantsBinding = 0;
    uint32_t lightsBinding = 0;
    uint32_t headersBinding = 0;
    uint32_t masksBinding = 0;
    uint64_t constantBytes = 0;
    std::unique_ptr<ShaderValueCopy> constantCopy;
};

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

ForwardPlusLightGrid::~ForwardPlusLightGrid()
{
    Shutdown();
}

std::shared_ptr<const ForwardPlusGridPipeline> ForwardPlusLightGrid::CreateProgram(rhi::Device &device,
                                                                                   const ShaderTargetCode &program)
{
    return std::make_shared<ForwardPlusGridPipeline>(device, program);
}

bool ForwardPlusLightGrid::Initialize(rhi::Device &device, uint32_t framesInFlight, const ShaderTargetCode &program)
{
    if (framesInFlight == 0) {
        Shutdown();
        return false;
    }
    return Initialize(CreateProgram(device, program), framesInFlight);
}

bool ForwardPlusLightGrid::Initialize(std::shared_ptr<const ForwardPlusGridPipeline> program, uint32_t framesInFlight)
{
    Shutdown();
    if (!program || framesInFlight == 0)
        return false;
    m_program = std::move(program);
    m_device = &m_program->device;
    m_frames.resize(framesInFlight);
    for (auto &frame : m_frames) {
        frame.constantBytes.resize(m_program->constantBytes);
        rhi::BufferDesc description;
        description.byteSize = frame.constantBytes.size();
        description.usage = rhi::BufferUsageFlags::Uniform;
        description.memory = rhi::BufferMemory::Upload;
        frame.constants = m_device->CreateBuffer(description);
    }
    return true;
}

rhi::BindingLayoutHandle ForwardPlusLightGrid::ConsumerLayout() const noexcept
{
    return m_program ? m_program->consumerLayout : rhi::BindingLayoutHandle{};
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
            m_device->Release(frame.constants);
        }
    }
    m_frames.clear();
    m_retired.clear();
    m_program.reset();
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
                                  const ForwardPlusGridConstants &constants)
{
    auto &frame = m_frames[frameIndex];

    ForwardPlusGridConstants resolved = constants;
    resolved.viewportAndProjectionScale[0] = static_cast<float>(frame.config.width);
    resolved.viewportAndProjectionScale[1] = static_cast<float>(frame.config.height);
    resolved.gridAndLights[0] = frame.config.tileCountX;
    resolved.gridAndLights[1] = frame.config.tileCountY;
    resolved.gridAndLights[2] = frame.config.localLightCount;
    resolved.gridAndLights[3] = TileSize;
    resolved.domainAndMaskWords[0] = frame.config.domainMask;
    resolved.domainAndMaskWords[1] = frame.config.maskWordStride;

    m_program->constantCopy->Write(frame.constantBytes.data(), &resolved);
    m_device->WriteBuffer(frame.constants, 0, frame.constantBytes.data(), frame.constantBytes.size());
    encoder.BindPipeline(m_program->pipeline);
    encoder.BindGroup(m_program->pipeline, 0, frame.bindGroup);
    encoder.Dispatch(frame.config.tileCountX, frame.config.tileCountY, 1);
}

bool ForwardPlusLightGrid::IsValid() const noexcept
{
    return m_device && m_program && !m_frames.empty();
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
    groupDesc.layout = m_program->shader.GetBindingLayouts()[0];
    const rhi::BufferHandle buffers[] = {canonicalLights, frame.headers, frame.lightMasks};
    const uint32_t bindings[] = {m_program->lightsBinding, m_program->headersBinding, m_program->masksBinding};
    for (uint32_t index = 0; index < 3; ++index)
        groupDesc.buffers[index] = {bindings[index], rhi::BindingType::StorageBuffer, buffers[index], 0, 0};
    groupDesc.buffers[3] = {m_program->constantsBinding, rhi::BindingType::UniformBuffer, frame.constants, 0,
                            m_program->constantBytes};
    groupDesc.bufferCount = 4;
    frame.bindGroup = m_device->CreateBindGroup(groupDesc);
    if (!frame.bindGroup.IsValid())
        return false;

    groupDesc.layout = m_program->consumerLayout;
    groupDesc.bufferCount = 3;
    for (uint32_t binding = 0; binding < 3; ++binding)
        groupDesc.buffers[binding] = {binding, rhi::BindingType::StorageBuffer, buffers[binding], 0, 0};
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
    return R"(

struct CanonicalLightData {
    float4 position_range;
    float4 direction_spot;
    float4 color_intensity;
    float4 attenuation;
    float4 area_right_width;
    float4 area_up_height;
    uint4 metadata;
    uint4 identity_shadow;
};

ByteAddressBuffer canonical_lights;
RWStructuredBuffer<uint4> tile_headers;
RWStructuredBuffer<Atomic<uint>> tile_light_masks;

struct ForwardPlusGridConstants {
    // The canonical host camera stores column vectors. Keep that data
    // contract explicit and independent of target matrix storage rules.
    float4 view_projection_columns[4];
    float4 viewport_projection_scale;
    uint4 grid_lights;
    uint4 domain_mask_words;
};
ConstantBuffer<ForwardPlusGridConstants> pc;

// Pull the tile's clip inequalities back to world-space planes. This avoids
// dividing a sphere's center by w and guessing a symmetric projected radius,
// which can exclude lit pixels for off-axis or shifted projections. All light
// lanes share the six planes and their normal lengths, computed once per tile.
groupshared float4 tile_planes[6];
groupshared float tile_plane_lengths[6];

void prepare_tile_plane(uint index, uint2 tile) {
    float4x4 rows = transpose(float4x4(pc.view_projection_columns[0], pc.view_projection_columns[1],
                                      pc.view_projection_columns[2], pc.view_projection_columns[3]));
    float2 viewport = pc.viewport_projection_scale.xy;
    float2 pixel_min = float2(tile) * float(pc.grid_lights.w);
    float2 pixel_max = min(pixel_min + float(pc.grid_lights.w), viewport);
    float2 ndc_min = 2.0 * pixel_min / viewport - 1.0;
    float2 ndc_max = 2.0 * pixel_max / viewport - 1.0;
    float4 plane;
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
    float4 center = float4(light.position_range.xyz, 1.0);
    float radius = max(light.position_range.w, 0.0);
    if (light.metadata.x == 3u) {
        radius += 0.5 * length(float2(light.area_right_width.w, light.area_up_height.w));
    }
    for (uint index = 0u; index < 6u; ++index) {
        float4 plane = tile_planes[index];
        float support = radius * tile_plane_lengths[index];
        // Outward rounding for float dot products keeps a tangent sphere in
        // the list; this is relative to the terms, not a world-unit padding.
        float roundoff = 0.000001 * (dot(abs(plane), abs(center)) + support);
        if (dot(plane, center) + support < -roundoff) return false;
    }
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void build_light_grid(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    uint2 tile = group.xy;
    uint tile_index = tile.y * pc.grid_lights.x + tile.x;
    uint offset = tile_index * pc.domain_mask_words.y;
    if (lane < 6u) prepare_tile_plane(lane, tile);
    for (uint word = lane; word < pc.domain_mask_words.y; word += 64u) {
        tile_light_masks[offset + word].store(0u);
    }
    // Both shared planes and storage mask words must be visible before the
    // other lanes test lights and atomically accumulate into the cleared mask.
    AllMemoryBarrierWithGroupSync();

    uint4 counts_generation = canonical_lights.Load<uint4>(0);
    uint directional_count = counts_generation.x;
    uint available_local_count = min(counts_generation.y, pc.grid_lights.z);
    for (uint local_index = lane; local_index < available_local_count; local_index += 64u) {
        CanonicalLightData light = canonical_lights.Load<CanonicalLightData>(
            16u + (directional_count + local_index) * sizeof(CanonicalLightData));
        if ((light.metadata.w & pc.domain_mask_words.x) == 0u || !overlaps_tile(light)) continue;
        tile_light_masks[offset + (local_index >> 5u)].or(1u << (local_index & 31u));
    }
    AllMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        if (tile_index == 0u) {
            tile_headers[0] = uint4(pc.grid_lights.xy, pc.grid_lights.w, pc.domain_mask_words.x);
        }
        tile_headers[tile_index + 1u] =
            uint4(offset, pc.domain_mask_words.y, available_local_count, pc.domain_mask_words.x);
    }
}
)";
}

} // namespace infernux::lighting
