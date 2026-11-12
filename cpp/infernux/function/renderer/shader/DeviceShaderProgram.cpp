#include "DeviceShaderProgram.h"

#include <function/renderer/rhi/RhiDevice.h>

#include <algorithm>
#include <stdexcept>

namespace infernux
{

struct DeviceShaderProgram::Impl
{
    struct Entry
    {
        std::string name;
        rhi::ShaderStage stage;
        rhi::ShaderModuleHandle module;
    };
    rhi::Device &device;
    std::vector<Entry> entries;
    std::vector<ShaderResourceLayout> resources;
    std::vector<rhi::BindingLayoutHandle> layouts;
    uint32_t rootConstantBytes = 0;

    explicit Impl(rhi::Device &owner) : device(owner)
    {
    }
    ~Impl()
    {
        for (const auto &entry : entries)
            device.Release(entry.module);
        for (const auto layout : layouts)
            device.Release(layout);
    }
};

DeviceShaderProgram::DeviceShaderProgram(rhi::Device &device, const ShaderTargetCode &code,
                                         const std::vector<uint32_t> &entries)
    : m_impl(std::make_unique<Impl>(device))
{
    m_impl->rootConstantBytes = code.rootConstants ? static_cast<uint32_t>(code.rootConstants->byteSize) : 0;
    for (const auto index : entries) {
        const auto &entry = code.entries.at(index);
        const auto language =
            code.target == ShaderCodeTarget::SpirV ? rhi::ShaderSourceLanguage::SpirV : rhi::ShaderSourceLanguage::Wgsl;
        const auto module = device.CreateShaderModule({language, entry.code.data(), entry.code.size()});
        if (!module.IsValid())
            throw std::runtime_error("Device shader module creation failed: " + entry.name);
        m_impl->entries.push_back({entry.name, entry.stage, module});
    }
    std::vector<rhi::BindingLayoutDesc> descriptions;
    for (auto resource : code.resources) {
        auto &binding = resource.binding;
        binding.visibility = rhi::ShaderStage::None;
        for (const auto index : entries) {
            const auto &entry = code.entries[index];
            const std::array<uint32_t, 2> key{resource.set, binding.binding};
            if (std::find(entry.bindings.begin(), entry.bindings.end(), key) != entry.bindings.end())
                binding.visibility = binding.visibility | entry.stage;
        }
        if (binding.visibility == rhi::ShaderStage::None)
            continue;
        if (resource.set >= rhi::GraphicsPipelineDesc::MaxBindingLayouts)
            throw std::runtime_error("Shader exceeds the device program binding-set capacity: " + resource.name);
        descriptions.resize(std::max(descriptions.size(), static_cast<size_t>(resource.set) + 1));
        auto &set = descriptions[resource.set];
        if (set.entryCount == set.entries.size())
            throw std::runtime_error("Shader exceeds the device binding-set capacity: " + resource.name);
        set.entries[set.entryCount++] = binding;
        m_impl->resources.push_back(std::move(resource));
    }
    for (const auto &description : descriptions) {
        const auto layout = device.CreateBindingLayout(description);
        if (!layout.IsValid())
            throw std::runtime_error("Device shader binding layout creation failed");
        m_impl->layouts.push_back(layout);
    }
}

DeviceShaderProgram::~DeviceShaderProgram() = default;

void DeviceShaderProgram::Apply(rhi::GraphicsPipelineDesc &pipeline) const
{
    for (const auto &entry : m_impl->entries) {
        if (m_impl->rootConstantBytes != 0)
            pipeline.pushConstantStages = pipeline.pushConstantStages | entry.stage;
        if (entry.stage == rhi::ShaderStage::Vertex) {
            pipeline.vertexShader = entry.module;
            pipeline.vertexEntryPoint = entry.name.c_str();
        } else if (entry.stage == rhi::ShaderStage::Fragment) {
            pipeline.fragmentShader = entry.module;
            pipeline.fragmentEntryPoint = entry.name.c_str();
        } else {
            throw std::logic_error("A graphics pipeline cannot contain a Compute entry");
        }
    }
    pipeline.bindingLayoutCount = static_cast<uint32_t>(m_impl->layouts.size());
    pipeline.pushConstantBytes = m_impl->rootConstantBytes;
    std::copy(m_impl->layouts.begin(), m_impl->layouts.end(), pipeline.bindingLayouts.begin());
}

void DeviceShaderProgram::Apply(rhi::ComputePipelineDesc &pipeline) const
{
    if (m_impl->entries.size() != 1 || m_impl->entries[0].stage != rhi::ShaderStage::Compute)
        throw std::logic_error("A Compute pipeline selects exactly one Compute entry");
    pipeline.computeShader = m_impl->entries[0].module;
    pipeline.entryPoint = m_impl->entries[0].name.c_str();
    pipeline.bindingLayoutCount = static_cast<uint32_t>(m_impl->layouts.size());
    pipeline.pushConstantBytes = m_impl->rootConstantBytes;
    std::copy(m_impl->layouts.begin(), m_impl->layouts.end(), pipeline.bindingLayouts.begin());
}

const std::vector<ShaderResourceLayout> &DeviceShaderProgram::GetResources() const noexcept
{
    return m_impl->resources;
}

const std::vector<rhi::BindingLayoutHandle> &DeviceShaderProgram::GetBindingLayouts() const noexcept
{
    return m_impl->layouts;
}

} // namespace infernux
