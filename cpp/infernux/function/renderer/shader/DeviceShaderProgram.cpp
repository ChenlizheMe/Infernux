#include "DeviceShaderProgram.h"

#include <function/renderer/rhi/RhiDevice.h>

#include <algorithm>
#include <stdexcept>
#include <tuple>

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
    std::vector<rhi::BindingLayoutHandle> ownedLayouts;
    uint32_t rootConstantBytes = 0;

    explicit Impl(rhi::Device &owner) : device(owner)
    {
    }
    ~Impl()
    {
        for (const auto &entry : entries)
            device.Release(entry.module);
        for (const auto layout : ownedLayouts)
            device.Release(layout);
    }
};

namespace
{
auto ResourceBindingKey(const rhi::BindingLayoutEntry &value)
{
    return std::tie(value.binding, value.type, value.count, value.depthRead, value.readOnlyStorage, value.viewDimension,
                    value.sampleType, value.multisampled, value.storageFormat, value.storageAccess, value.samplerType,
                    value.minBufferBindingSize);
}

std::vector<ShaderStageSelection> SelectEntries(const ShaderTargetCode &program, const std::vector<uint32_t> &entries)
{
    std::vector<ShaderStageSelection> result;
    result.reserve(entries.size());
    for (const auto index : entries)
        result.push_back({program, index});
    return result;
}
} // namespace

DeviceShaderProgram::DeviceShaderProgram(rhi::Device &device, const ShaderTargetCode &code,
                                         const std::vector<uint32_t> &entries,
                                         const std::vector<rhi::BindingLayoutHandle> &sharedLayouts)
    : DeviceShaderProgram(device, SelectEntries(code, entries), sharedLayouts)
{
}

DeviceShaderProgram::DeviceShaderProgram(rhi::Device &device, const std::vector<ShaderStageSelection> &entries,
                                         const std::vector<rhi::BindingLayoutHandle> &sharedLayouts)
    : m_impl(std::make_unique<Impl>(device))
{
    const ShaderValueLayout *rootLayout = nullptr;
    for (const auto &selection : entries) {
        const auto &code = selection.program;
        const auto &entry = code.entries.at(selection.entry);
        if (entry.sampleRateShading && !device.GetCapabilities().features.sampleRateShading)
            throw std::runtime_error("Device does not support per-sample fragment inputs: " + entry.name);
        const auto language =
            code.target == ShaderCodeTarget::SpirV ? rhi::ShaderSourceLanguage::SpirV : rhi::ShaderSourceLanguage::Wgsl;
        const auto module = device.CreateShaderModule({language, entry.code.data(), entry.code.size()});
        if (!module.IsValid())
            throw std::runtime_error("Device shader module creation failed: " + entry.name);
        const auto rootBytes = entry.rootConstants ? static_cast<uint32_t>(entry.rootConstants->byteSize) : 0;
        m_impl->entries.push_back({entry.name, entry.stage, module});
        if (entry.rootConstants) {
            if (rootLayout && !(*rootLayout == *entry.rootConstants))
                throw std::invalid_argument("Shader stages disagree on per-execution values: " + entry.name);
            rootLayout = &*entry.rootConstants;
        }
        m_impl->rootConstantBytes = std::max(m_impl->rootConstantBytes, rootBytes);

        for (const auto &resource : code.resources) {
            const std::array<uint32_t, 2> key{resource.set, resource.binding.binding};
            if (std::find(entry.bindings.begin(), entry.bindings.end(), key) == entry.bindings.end())
                continue;
            const auto existing =
                std::find_if(m_impl->resources.begin(), m_impl->resources.end(), [&](const auto &value) {
                    return value.set == resource.set && value.binding.binding == resource.binding.binding;
                });
            if (existing == m_impl->resources.end()) {
                m_impl->resources.push_back(resource);
                m_impl->resources.back().binding.visibility = entry.stage;
                continue;
            }
            if (ResourceBindingKey(existing->binding) != ResourceBindingKey(resource.binding) ||
                (resource.binding.type == rhi::BindingType::UniformBuffer && !(existing->value == resource.value)))
                throw std::invalid_argument("Shader stages disagree on resource binding " +
                                            std::to_string(resource.set) + "/" +
                                            std::to_string(resource.binding.binding) + ": " + resource.name);
            existing->binding.visibility = existing->binding.visibility | entry.stage;
        }
    }
    std::sort(m_impl->resources.begin(), m_impl->resources.end(), [](const auto &first, const auto &second) {
        return std::tie(first.set, first.binding.binding) < std::tie(second.set, second.binding.binding);
    });
    std::vector<rhi::BindingLayoutDesc> descriptions;
    for (const auto &resource : m_impl->resources) {
        if (resource.set >= rhi::GraphicsPipelineDesc::MaxBindingLayouts)
            throw std::runtime_error("Shader exceeds the device program binding-set capacity: " + resource.name);
        descriptions.resize(std::max(descriptions.size(), static_cast<size_t>(resource.set) + 1));
        auto &set = descriptions[resource.set];
        if (set.entryCount == set.entries.size())
            throw std::runtime_error("Shader exceeds the device binding-set capacity: " + resource.name);
        set.entries[set.entryCount++] = resource.binding;
    }
    for (size_t index = 0; index < descriptions.size(); ++index) {
        const bool shared = index < sharedLayouts.size() && sharedLayouts[index].IsValid();
        const auto layout = shared ? sharedLayouts[index] : device.CreateBindingLayout(descriptions[index]);
        if (!layout.IsValid())
            throw std::runtime_error("Device shader binding layout creation failed");
        m_impl->layouts.push_back(layout);
        if (!shared)
            m_impl->ownedLayouts.push_back(layout);
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
