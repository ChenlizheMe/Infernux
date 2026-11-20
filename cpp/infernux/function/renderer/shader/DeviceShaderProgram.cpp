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
    return std::tie(value.binding, value.type, value.count, value.depthRead, value.viewDimension, value.sampleType,
                    value.multisampled, value.storageFormat, value.storageAccess, value.samplerType,
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

std::vector<rhi::BindingLayoutDesc> BuildBindingLayouts(const std::vector<ShaderResourceLayout> &resources)
{
    std::vector<rhi::BindingLayoutDesc> descriptions;
    for (const auto &resource : resources) {
        if (resource.set >= rhi::GraphicsPipelineDesc::MaxBindingLayouts)
            throw std::invalid_argument("Shader exceeds the device program binding-set capacity: " + resource.name);
        descriptions.resize(std::max(descriptions.size(), static_cast<size_t>(resource.set) + 1));
        auto &set = descriptions[resource.set];
        if (set.entryCount == set.entries.size())
            throw std::invalid_argument("Shader binding set " + std::to_string(resource.set) + " exceeds " +
                                        std::to_string(set.entries.size()) + " entries at resource '" + resource.name +
                                        "'");
        set.entries[set.entryCount++] = resource.binding;
    }
    return descriptions;
}
} // namespace

std::vector<ShaderStageSelection> SelectGraphicsShaderEntries(const ShaderTargetCode &code)
{
    std::vector<ShaderStageSelection> result;
    for (uint32_t index = 0; index < code.entries.size(); ++index)
        if (code.entries[index].stage != rhi::ShaderStage::Compute)
            result.push_back({code, index});
    return result;
}

void ValidateDeviceShaderBindings(const std::vector<ShaderStageSelection> &entries)
{
    (void)BuildBindingLayouts(LinkShaderStages(entries).resources);
}

ShaderStageLayout LinkShaderStages(const std::vector<ShaderStageSelection> &entries)
{
    ShaderStageLayout result;
    const ShaderEntryPointCode *vertex = nullptr;
    const ShaderEntryPointCode *fragment = nullptr;
    for (const auto &selection : entries) {
        const auto &code = selection.program;
        const auto &entry = code.entries.at(selection.entry);
        if (entry.stage == rhi::ShaderStage::Vertex)
            vertex = &entry;
        if (entry.stage == rhi::ShaderStage::Fragment)
            fragment = &entry;
        if (entry.rootConstants) {
            if (result.rootConstants && !(*result.rootConstants == *entry.rootConstants))
                throw std::invalid_argument("Shader stages disagree on per-execution values: " + entry.name);
            result.rootConstants = entry.rootConstants;
        }
        for (const auto &resource : code.resources) {
            const std::array<uint32_t, 2> key{resource.set, resource.binding.binding};
            if (std::find(entry.bindings.begin(), entry.bindings.end(), key) == entry.bindings.end())
                continue;
            const auto existing =
                std::find_if(result.resources.begin(), result.resources.end(), [&](const auto &value) {
                    return value.set == resource.set && value.binding.binding == resource.binding.binding;
                });
            if (existing == result.resources.end()) {
                result.resources.push_back(resource);
                result.resources.back().binding.visibility = entry.stage;
                continue;
            }
            if (ResourceBindingKey(existing->binding) != ResourceBindingKey(resource.binding) ||
                existing->name != resource.name ||
                (resource.binding.type == rhi::BindingType::UniformBuffer && !(existing->value == resource.value)))
                throw std::invalid_argument("Shader stages disagree on resource binding " +
                                            std::to_string(resource.set) + "/" +
                                            std::to_string(resource.binding.binding) + ": " + resource.name);
            existing->binding.visibility = existing->binding.visibility | entry.stage;
            existing->binding.readOnlyStorage &= resource.binding.readOnlyStorage;
        }
    }
    if (vertex && fragment) {
        for (const auto &input : fragment->inputs) {
            const auto output = std::find_if(vertex->outputs.begin(), vertex->outputs.end(),
                                             [&](const auto &value) { return value.location == input.location; });
            const auto context = "Fragment input '" + input.name + "' at location " + std::to_string(input.location);
            if (output == vertex->outputs.end())
                throw std::invalid_argument(context + " has no matching vertex output (" + vertex->name + " -> " +
                                            fragment->name + ")");
            if (output->scalar != input.scalar || output->components != input.components)
                throw std::invalid_argument(context + " does not match vertex output '" + output->name +
                                            "': scalar type or component count differs");
        }
    }
    std::sort(result.resources.begin(), result.resources.end(), [](const auto &first, const auto &second) {
        return std::tie(first.set, first.binding.binding) < std::tie(second.set, second.binding.binding);
    });
    return result;
}

std::vector<ShaderStageSelection> SelectMaterialShaderEntries(const ShaderProgramArtifact &program,
                                                              ShaderCompileTarget pass, ShaderCodeTarget target)
{
    const auto *code = program.FindPassCode(pass, target);
    if (!code)
        throw std::invalid_argument("Material has no compiled pass for this device: " + program.key.ToString());
    std::vector<ShaderStageSelection> result;
    result.reserve(code->vertex ? 2 : 1);
    const auto indexOf = [&](const ShaderEntryPointCode *entry) {
        return static_cast<uint32_t>(entry - code->module->entries.data());
    };
    if (code->vertex)
        result.push_back({*code->module, indexOf(code->vertex)});
    result.push_back({*code->module, indexOf(code->fragment)});
    return result;
}

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
    auto linked = LinkShaderStages(entries);
    m_impl->resources = std::move(linked.resources);
    m_impl->rootConstantBytes = linked.rootConstants ? static_cast<uint32_t>(linked.rootConstants->byteSize) : 0;
    const auto descriptions = BuildBindingLayouts(m_impl->resources);
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
        m_impl->entries.push_back({entry.name, entry.stage, module});
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
