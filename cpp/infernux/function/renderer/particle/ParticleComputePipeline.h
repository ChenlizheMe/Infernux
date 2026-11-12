#pragma once

#include <function/renderer/rhi/RhiDevice.h>
#include <function/renderer/shader/DeviceShaderProgram.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace infernux::particle
{

inline std::vector<rhi::BufferBinding> ParticleBufferBindings(std::initializer_list<rhi::BufferHandle> buffers)
{
    std::vector<rhi::BufferBinding> result;
    result.reserve(buffers.size());
    for (auto buffer : buffers)
        result.push_back({static_cast<uint32_t>(result.size()), rhi::BindingType::StorageBuffer, buffer});
    return result;
}

/// A prepared built-in kernel. Reflection selects the entry's actual ports,
/// including read-only storage, before any frame records this program.
class ParticleComputePipeline
{
  public:
    ParticleComputePipeline(rhi::Device &device, const ShaderTargetCode &code)
        : m_device(device), m_shader(device, code, {0})
    {
        rhi::ComputePipelineDesc description;
        m_shader.Apply(description);
        m_pipeline = device.CreateComputePipeline(description);
        if (!m_pipeline.IsValid())
            throw std::runtime_error("Particle compute pipeline creation failed");
    }

    ~ParticleComputePipeline()
    {
        for (auto group : m_groups)
            m_device.Release(group);
        m_device.Release(m_pipeline);
    }

    [[nodiscard]] rhi::BindGroupHandle CreateBindings(const std::vector<rhi::BufferBinding> &buffers) const
    {
        rhi::BindGroupDesc description;
        description.layout = m_shader.GetBindingLayouts().at(0);
        for (const auto &resource : m_shader.GetResources()) {
            const auto binding = resource.binding.binding;
            const auto buffer = std::find_if(buffers.begin(), buffers.end(),
                                             [binding](const auto &value) { return value.binding == binding; });
            if (buffer == buffers.end())
                throw std::runtime_error("Particle compute input is missing: " + resource.name);
            description.buffers[description.bufferCount++] = *buffer;
        }
        const auto group = m_device.CreateBindGroup(description);
        if (!group.IsValid())
            throw std::runtime_error("Particle compute bindings creation failed");
        return group;
    }

    void AddBindings(const std::vector<rhi::BufferBinding> &buffers)
    {
        m_groups.push_back(CreateBindings(buffers));
    }

    template <typename Constants>
    void Bind(const rhi::ComputeCommandEncoder &encoder, const Constants &constants, uint32_t variant = 0) const
    {
        BindGroup(encoder, m_groups[variant], constants);
    }

    template <typename Constants>
    void BindGroup(const rhi::ComputeCommandEncoder &encoder, rhi::BindGroupHandle group,
                   const Constants &constants) const
    {
        encoder.BindPipeline(m_pipeline);
        encoder.BindGroup(m_pipeline, 0, group);
        encoder.PushConstants(m_pipeline, sizeof(constants), &constants);
    }

  private:
    rhi::Device &m_device;
    DeviceShaderProgram m_shader;
    rhi::ComputePipelineHandle m_pipeline;
    std::vector<rhi::BindGroupHandle> m_groups;
};

} // namespace infernux::particle
