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

} // namespace infernux::lighting
