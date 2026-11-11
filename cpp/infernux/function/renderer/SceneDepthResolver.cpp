#include "SceneDepthResolver.h"

#include "rhi/GpuRetirementQueue.h"
#include "shader/DeviceShaderProgram.h"
#include "shader/ShaderValueCopy.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace infernux
{

// One immutable program per device. Views own their binding publications.
struct SceneDepthResolveProgram
{
    SceneDepthResolveProgram(rhi::Device &owner, const ShaderTargetCode &code) : device(owner), shader(owner, code, {0})
    {
        for (const auto &resource : shader.GetResources()) {
            if (resource.name == "parameters") {
                constantsBinding = resource.binding.binding;
                constantBytes = resource.value.byteSize;
                constantsCopy = std::make_unique<ShaderValueCopy>(resource.value);
            } else if (resource.name == "source_depth")
                sourceBinding = resource.binding.binding;
            else if (resource.name == "resolved_depth")
                destinationBinding = resource.binding.binding;
        }
        rhi::ComputePipelineDesc description;
        shader.Apply(description);
        pipeline = device.CreateComputePipeline(description);
        if (!pipeline.IsValid())
            throw std::runtime_error("Scene-depth resolve pipeline creation failed");
    }
    ~SceneDepthResolveProgram()
    {
        device.Release(pipeline);
    }
    rhi::Device &device;
    DeviceShaderProgram shader;
    rhi::ComputePipelineHandle pipeline;
    uint32_t sourceBinding = 0;
    uint32_t destinationBinding = 0;
    uint32_t constantsBinding = 0;
    uint64_t constantBytes = 0;
    std::unique_ptr<ShaderValueCopy> constantsCopy;
};

SceneDepthResolver::~SceneDepthResolver()
{
    Destroy();
}

std::shared_ptr<const SceneDepthResolveProgram> SceneDepthResolver::CreateProgram(rhi::Device &device,
                                                                                  const ShaderTargetCode &code)
{
    return std::make_shared<SceneDepthResolveProgram>(device, code);
}

void SceneDepthResolver::Initialize(std::shared_ptr<const SceneDepthResolveProgram> program)
{
    Destroy();
    m_program = std::move(program);
    m_device = &m_program->device;
}

void SceneDepthResolver::Destroy() noexcept
{
    if (m_device) {
        for (const auto &binding : m_bindings) {
            m_device->Release(binding.group);
            m_device->Release(binding.constants);
        }
    }
    m_bindings.clear();
    m_program.reset();
    m_device = nullptr;
}

bool SceneDepthResolver::IsValid() const noexcept
{
    return static_cast<bool>(m_program);
}

void SceneDepthResolver::Record(const rhi::ComputeCommandEncoder &encoder, rhi::TextureViewHandle sourceDepth,
                                rhi::TextureViewHandle resolvedDepth, uint32_t width, uint32_t height,
                                uint32_t sampleCount)
{
    const auto group = ResolveBindGroup(sourceDepth, resolvedDepth, width, height, sampleCount);
    encoder.BindPipeline(m_program->pipeline);
    encoder.BindGroup(m_program->pipeline, 0, group);
    encoder.Dispatch((width + 7u) / 8u, (height + 7u) / 8u, 1);
}

void SceneDepthResolver::RetireBindings(GpuRetirementQueue &queue)
{
    if (m_bindings.empty())
        return;
    queue.Retire([device = m_device, bindings = std::exchange(m_bindings, {})] {
        for (const auto &binding : bindings) {
            device->Release(binding.group);
            device->Release(binding.constants);
        }
    });
}

std::string_view SceneDepthResolver::ShaderSource() noexcept
{
    return R"(
struct ResolveConstants {
    uint width;
    uint height;
    uint sample_count;
    uint reserved;
};
ConstantBuffer<ResolveConstants> parameters;
DepthTexture2DMS source_depth;
[format("r32f")] WTexture2D<float> resolved_depth;

[shader("compute")]
[numthreads(8, 8, 1)]
void resolve_depth(uint3 id : SV_DispatchThreadID)
{
    uint2 coordinate = id.xy;
    if (coordinate.x >= parameters.width || coordinate.y >= parameters.height) return;
    int2 pixel = int2(coordinate);
    float depth = source_depth.Load(pixel, 0);
    for (uint sample_index = 1; sample_index < parameters.sample_count; ++sample_index)
        // Reversed-Z: retain the nearest covered surface for soft-particle edges.
        depth = max(depth, source_depth.Load(pixel, int(sample_index)));
    resolved_depth.Store(pixel, depth);
}
)";
}

rhi::BindGroupHandle SceneDepthResolver::ResolveBindGroup(rhi::TextureViewHandle sourceDepth,
                                                          rhi::TextureViewHandle resolvedDepth, uint32_t width,
                                                          uint32_t height, uint32_t sampleCount)
{
    const auto existing = std::find_if(m_bindings.begin(), m_bindings.end(), [&](const auto &entry) {
        return entry.sourceDepth == sourceDepth && entry.resolvedDepth == resolvedDepth && entry.width == width &&
               entry.height == height && entry.sampleCount == sampleCount;
    });
    if (existing != m_bindings.end())
        return existing->group;

    const std::array<uint32_t, 4> values{width, height, sampleCount, 0};
    std::vector<char> bytes(m_program->constantBytes);
    m_program->constantsCopy->Write(bytes.data(), values.data());
    rhi::BufferDesc constants;
    constants.byteSize = bytes.size();
    constants.usage = rhi::BufferUsageFlags::Uniform;
    constants.memory = rhi::BufferMemory::Upload;
    constants.initialData = bytes.data();
    constants.initialDataBytes = bytes.size();
    const auto buffer = m_device->CreateBuffer(constants);
    if (!buffer.IsValid())
        throw std::runtime_error("Scene-depth resolve constants allocation failed");

    rhi::BindGroupDesc group;
    group.layout = m_program->shader.GetBindingLayouts()[0];
    group.textures[0] = {m_program->sourceBinding, rhi::BindingType::SampledTexture, sourceDepth, {}, true};
    group.textures[1] = {m_program->destinationBinding, rhi::BindingType::StorageTexture, resolvedDepth};
    group.textureCount = 2;
    group.buffers[0] = {m_program->constantsBinding, rhi::BindingType::UniformBuffer, buffer, 0, bytes.size()};
    group.bufferCount = 1;
    const auto binding = m_device->CreateBindGroup(group);
    if (!binding.IsValid()) {
        m_device->Release(buffer);
        throw std::runtime_error("Scene-depth resolve binding creation failed");
    }
    m_bindings.push_back({sourceDepth, resolvedDepth, width, height, sampleCount, binding, buffer});
    return binding;
}

} // namespace infernux
