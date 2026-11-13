#include "FullscreenRenderer.h"

#include "rhi/RhiDevice.h"
#include "shader/DeviceShaderProgram.h"
#include "shader/ShaderValueCopy.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace infernux
{
namespace
{
void StoreParameter(char *destination, ShaderScalarKind scalar, double value)
{
    switch (scalar) {
    case ShaderScalarKind::Float32: {
        const auto converted = static_cast<float>(value);
        std::memcpy(destination, &converted, sizeof(converted));
        break;
    }
    case ShaderScalarKind::Int32: {
        const auto converted = static_cast<int32_t>(value);
        std::memcpy(destination, &converted, sizeof(converted));
        break;
    }
    case ShaderScalarKind::UInt32: {
        const auto converted = static_cast<uint32_t>(value);
        std::memcpy(destination, &converted, sizeof(converted));
        break;
    }
    default:
        throw std::logic_error("Fullscreen parameter has no scalar upload contract");
    }
}

void ReadParameterComponents(const nlohmann::json &value, std::vector<double> &components)
{
    if (value.is_array()) {
        for (const auto &element : value)
            ReadParameterComponents(element, components);
    } else
        components.push_back(value.is_boolean() ? (value.get<bool>() ? 1.0 : 0.0) : value.get<double>());
}

void PrepareInputs(FullscreenPipelineEntry &entry, const std::string &schemaText)
{
    const auto schema = nlohmann::json::parse(schemaText);
    if (schema.at("domain") != "fullscreen")
        throw std::invalid_argument("Fullscreen pass requires a shader.fullscreen author program");
    for (const auto &input : schema.at("inputs"))
        entry.inputNames.push_back(input.at("name").get<std::string>());
    for (const auto &resource : entry.program->GetResources()) {
        if (resource.binding.type == rhi::BindingType::SampledTexture ||
            resource.binding.type == rhi::BindingType::Sampler) {
            const auto end = resource.name.find('.');
            const auto name = resource.name.substr(4, end - 4); // Generated inx_<logical port>.
            const auto input = std::find(entry.inputNames.begin(), entry.inputNames.end(), name);
            if (input == entry.inputNames.end())
                throw std::logic_error("Compiled fullscreen resource has no logical input: " + resource.name);
            entry.resourceInputs.push_back(static_cast<int32_t>(input - entry.inputNames.begin()));
        } else {
            static const std::unordered_map<std::string, int32_t> uniformSources = {
                {"inx_parameters", FullscreenPipelineEntry::Parameters},
                {"inx_globals", FullscreenPipelineEntry::Globals},
                {"inx_view", FullscreenPipelineEntry::Camera},
            };
            const auto source = uniformSources.find(resource.name);
            if (resource.binding.type != rhi::BindingType::UniformBuffer || source == uniformSources.end())
                throw std::logic_error("Fullscreen author resource has no graph binding contract: " + resource.name);
            entry.resourceInputs.push_back(source->second);
        }
    }
}
} // namespace

bool FullscreenPipelineKey::operator==(const FullscreenPipelineKey &other) const noexcept
{
    return shaderName == other.shaderName && renderTargetLayout == other.renderTargetLayout &&
           samples == other.samples && colorFormat == other.colorFormat && depthFormat == other.depthFormat &&
           depth.testEnabled == other.depth.testEnabled && depth.writeEnabled == other.depth.writeEnabled &&
           depth.compare == other.depth.compare && alphaBlend == other.alphaBlend &&
           inputResourceCount == other.inputResourceCount && depthInputMask == other.depthInputMask &&
           inputBufferMask == other.inputBufferMask && useDynamicRendering == other.useDynamicRendering &&
           (program ? program->code.get() : nullptr) == (other.program ? other.program->code.get() : nullptr);
}

size_t FullscreenPipelineKeyHash::operator()(const FullscreenPipelineKey &key) const noexcept
{
    size_t hash = std::hash<std::string>{}(key.shaderName);
    const auto combine = [&hash](size_t value) { hash ^= value + 0x9e3779b9U + (hash << 6) + (hash >> 2); };
    combine(rhi::HandleHash<rhi::RenderTargetLayoutTag>{}(key.renderTargetLayout));
    combine(static_cast<size_t>(key.samples));
    combine(static_cast<size_t>(key.colorFormat));
    combine(static_cast<size_t>(key.depthFormat));
    combine(key.depth.testEnabled);
    combine(key.depth.writeEnabled);
    combine(static_cast<size_t>(key.depth.compare));
    combine(key.alphaBlend);
    combine(key.inputResourceCount);
    combine(key.inputBufferMask);
    combine(key.depthInputMask);
    combine(key.useDynamicRendering ? 1U : 0U);
    combine(std::hash<const ShaderTargetCode *>{}(key.program ? key.program->code.get() : nullptr));
    return hash;
}

struct FullscreenPipelineCache::Impl
{
    struct PipelineEntry
    {
        FullscreenPipelineEntry rhi;
        rhi::BindingLayoutHandle emptyGapLayout;
        uint64_t lastUse = 0;
    };

    std::shared_ptr<FullscreenRendererHost> host;
    rhi::Device *device = nullptr;
    std::unordered_map<FullscreenPipelineKey, PipelineEntry, FullscreenPipelineKeyHash> pipelines;
    size_t capacity = 0;
    uint64_t useSequence = 0;

    void DestroyPipeline(PipelineEntry &entry)
    {
        if (device) {
            device->Release(entry.rhi.pipeline);
            device->Release(entry.rhi.inputLayout);
            device->Release(entry.emptyGapLayout);
        }
        entry = {};
    }

    PipelineEntry CreatePipeline(const FullscreenPipelineKey &key)
    {
        PipelineEntry entry;
        if (!host || !device || key.inputResourceCount > rhi::BindingLayoutDesc::MaxEntries ||
            (!key.useDynamicRendering && !key.renderTargetLayout.IsValid()))
            return entry;

        rhi::GraphicsPipelineDesc desc;
        desc.useDynamicRendering = key.useDynamicRendering;
        desc.renderTargetLayout = key.renderTargetLayout;
        desc.topology = rhi::PrimitiveTopology::TriangleList;
        desc.raster.cullMode = rhi::CullMode::None;
        desc.raster.frontFace = rhi::FrontFace::Clockwise;
        desc.samples = key.samples;
        desc.colorTargetCount = 1;
        desc.colorTargets[0].format = key.colorFormat;
        desc.colorTargets[0].blendEnabled = key.alphaBlend;
        desc.depth = key.depth;
        if (key.useDynamicRendering) {
            desc.renderingSignature.colorFormatCount = 1;
            desc.renderingSignature.colorFormats[0] = key.colorFormat;
            desc.renderingSignature.samples = key.samples;
            desc.renderingSignature.depthFormat = key.depthFormat;
            if (rhi::IsStencilFormat(key.depthFormat))
                desc.renderingSignature.stencilFormat = key.depthFormat;
        }
        if (key.program) {
            std::vector<uint32_t> selected;
            for (uint32_t index = 0; index < key.program->code->entries.size(); ++index)
                if (key.program->code->entries[index].stage != rhi::ShaderStage::Compute)
                    selected.push_back(index);
            entry.rhi.program = std::make_shared<DeviceShaderProgram>(*device, *key.program->code, selected);
            entry.rhi.program->Apply(desc);
            PrepareInputs(entry.rhi, key.program->schema);
            entry.rhi.pipeline = device->CreateGraphicsPipeline(desc);
            if (!entry.rhi.pipeline.IsValid())
                throw std::runtime_error("Fullscreen pipeline creation failed: " + key.shaderName);
            return entry;
        }

        rhi::BindingLayoutDesc inputLayoutDesc;
        inputLayoutDesc.entryCount = key.inputResourceCount;
        for (uint32_t index = 0; index < key.inputResourceCount; ++index) {
            auto &binding = inputLayoutDesc.entries[index];
            binding.binding = index;
            binding.type = (key.inputBufferMask & (1u << index)) != 0 ? rhi::BindingType::StorageBuffer
                                                                      : rhi::BindingType::CombinedTextureSampler;
            binding.visibility = rhi::ShaderStage::Fragment;
            binding.depthRead = (key.depthInputMask & (1u << index)) != 0;
            binding.readOnlyStorage = binding.type == rhi::BindingType::StorageBuffer;
        }
        entry.rhi.inputLayout = device->CreateBindingLayout(inputLayoutDesc);
        if (!entry.rhi.inputLayout.IsValid())
            return entry;

        desc.bindingLayouts[desc.bindingLayoutCount++] = entry.rhi.inputLayout;
        desc.pushConstantStages = rhi::ShaderStage::Fragment;
        desc.pushConstantBytes = sizeof(FullscreenPushConstants);

        const auto perViewLayout = host->GetPerViewLayout();
        const auto globalsLayout = host->GetGlobalsLayout();
        if (perViewLayout.IsValid() || globalsLayout.IsValid()) {
            if (perViewLayout.IsValid()) {
                desc.bindingLayouts[desc.bindingLayoutCount++] = perViewLayout;
            } else {
                entry.emptyGapLayout = device->CreateBindingLayout({});
                if (!entry.emptyGapLayout.IsValid()) {
                    DestroyPipeline(entry);
                    return {};
                }
                desc.bindingLayouts[desc.bindingLayoutCount++] = entry.emptyGapLayout;
            }
        }
        if (globalsLayout.IsValid())
            desc.bindingLayouts[desc.bindingLayoutCount++] = globalsLayout;

        desc.vertexShader = host->AcquireShaderModule("Fullscreen Triangle", rhi::ShaderStage::Vertex, 0, 0);
        desc.fragmentShader = host->AcquireShaderModule(key.shaderName, rhi::ShaderStage::Fragment,
                                                        key.inputResourceCount, key.inputBufferMask);
        if (!desc.vertexShader.IsValid() || !desc.fragmentShader.IsValid()) {
            host->ReportError("FullscreenRenderer: missing shader modules for '" + key.shaderName + "'");
            device->Release(desc.vertexShader);
            device->Release(desc.fragmentShader);
            DestroyPipeline(entry);
            return {};
        }

        entry.rhi.pipeline = device->CreateGraphicsPipeline(desc);
        device->Release(desc.vertexShader);
        device->Release(desc.fragmentShader);
        if (!entry.rhi.pipeline.IsValid()) {
            host->ReportError("FullscreenRenderer: failed to create pipeline for '" + key.shaderName + "'");
            DestroyPipeline(entry);
            return {};
        }
        entry.rhi.hasPerView = perViewLayout.IsValid();
        entry.rhi.hasGlobals = globalsLayout.IsValid();
        return entry;
    }
};

FullscreenPipelineCache::FullscreenPipelineCache(std::shared_ptr<FullscreenRendererHost> host, size_t capacity)
{
    if (!host || capacity == 0)
        throw std::invalid_argument("Fullscreen pipeline cache requires a host and positive capacity");
    m_impl = std::make_unique<Impl>();
    m_impl->host = std::move(host);
    m_impl->device = &m_impl->host->GetRhiDevice();
    m_impl->capacity = capacity;
}

FullscreenPipelineCache::~FullscreenPipelineCache()
{
    for (auto &[key, entry] : m_impl->pipelines)
        m_impl->DestroyPipeline(entry);
}

rhi::Device &FullscreenPipelineCache::GetDevice() const noexcept
{
    return *m_impl->device;
}

bool FullscreenPipelineCache::HasShader(const std::string &name, rhi::ShaderStage stage) const
{
    return std::any_of(m_impl->pipelines.begin(), m_impl->pipelines.end(), [&](const auto &item) {
        const auto &[key, pipeline] = item;
        if (!key.program || (key.shaderName != name && key.program->shaderId != name && key.program->assetGuid != name))
            return false;
        return pipeline.rhi.pipeline.IsValid() &&
               std::any_of(key.program->code->entries.begin(), key.program->code->entries.end(),
                           [&](const auto &entry) { return entry.stage == stage; });
    });
}

struct FullscreenRenderer::Impl
{
    struct ParameterBuffer
    {
        rhi::BufferHandle buffer;
        uint64_t capacity = 0;
    };
    struct FrameParameters
    {
        std::vector<ParameterBuffer> buffers;
        size_t next = 0;
        std::vector<char> scratch;
    };
    std::shared_ptr<FullscreenRendererHost> host;
    std::shared_ptr<FullscreenPipelineCache> pipelines;
    rhi::Device *device = nullptr;
    rhi::SamplerHandle linearSampler;
    rhi::SamplerHandle nearestSampler;
    std::vector<std::vector<rhi::BindGroupHandle>> frameBindGroups;
    std::vector<FrameParameters> frameParameters;

    [[nodiscard]] uint32_t CurrentFrame() const noexcept
    {
        if (!host || frameBindGroups.empty())
            return 0;
        return host->GetCurrentFrame() % static_cast<uint32_t>(frameBindGroups.size());
    }

    void ReleaseBindGroups(uint32_t frame)
    {
        if (!device || frame >= frameBindGroups.size())
            return;
        for (const auto handle : frameBindGroups[frame])
            device->Release(handle);
        frameBindGroups[frame].clear();
    }

    rhi::BufferHandle UploadParameters(uint32_t frame, const void *data, uint64_t bytes)
    {
        auto &parameters = frameParameters[frame];
        if (parameters.next == parameters.buffers.size())
            parameters.buffers.emplace_back();
        auto &slot = parameters.buffers[parameters.next++];
        if (slot.capacity < bytes) {
            device->Release(slot.buffer);
            rhi::BufferDesc description;
            description.byteSize = bytes;
            description.usage = rhi::BufferUsageFlags::Uniform;
            description.memory = rhi::BufferMemory::Upload;
            slot.buffer = device->CreateBuffer(description);
            slot.capacity = bytes;
            if (!slot.buffer.IsValid())
                throw std::runtime_error("Fullscreen parameter buffer allocation failed");
        }
        if (!device->WriteBuffer(slot.buffer, 0, data, bytes))
            throw std::runtime_error("Fullscreen parameter upload failed");
        return slot.buffer;
    }
};

FullscreenRenderer::FullscreenRenderer() = default;
FullscreenRenderer::~FullscreenRenderer()
{
    Destroy();
}

void FullscreenRenderer::Initialize(std::shared_ptr<FullscreenRendererHost> host,
                                    std::shared_ptr<FullscreenPipelineCache> pipelines)
{
    Destroy();
    if (!host)
        return;

    m_impl = std::make_unique<Impl>();
    m_impl->host = std::move(host);
    m_impl->device = &m_impl->host->GetRhiDevice();
    if (pipelines && &pipelines->GetDevice() != m_impl->device)
        throw std::invalid_argument("Fullscreen pipeline cache belongs to another device");
    m_impl->pipelines = pipelines ? std::move(pipelines) : std::make_shared<FullscreenPipelineCache>(m_impl->host);

    rhi::SamplerDesc linear;
    linear.addressU = rhi::AddressMode::ClampToEdge;
    linear.addressV = rhi::AddressMode::ClampToEdge;
    linear.addressW = rhi::AddressMode::ClampToEdge;
    rhi::SamplerDesc nearest = linear;
    nearest.minFilter = rhi::FilterMode::Nearest;
    nearest.magFilter = rhi::FilterMode::Nearest;
    nearest.mipFilter = rhi::FilterMode::Nearest;
    m_impl->linearSampler = m_impl->device->CreateSampler(linear);
    m_impl->nearestSampler = m_impl->device->CreateSampler(nearest);
    if (!m_impl->linearSampler.IsValid() || !m_impl->nearestSampler.IsValid()) {
        m_impl->host->ReportError("FullscreenRenderer: failed to create RHI samplers");
        Destroy();
        return;
    }

    const uint32_t frames = std::max(1u, m_impl->host->GetFrameCount());
    m_impl->frameBindGroups.resize(frames);
    m_impl->frameParameters.resize(frames);
}

void FullscreenRenderer::Destroy()
{
    if (!m_impl)
        return;
    for (uint32_t frame = 0; frame < m_impl->frameBindGroups.size(); ++frame)
        m_impl->ReleaseBindGroups(frame);
    for (auto &frame : m_impl->frameParameters)
        for (const auto &slot : frame.buffers)
            m_impl->device->Release(slot.buffer);
    if (m_impl->device) {
        m_impl->device->Release(m_impl->linearSampler);
        m_impl->device->Release(m_impl->nearestSampler);
    }
    m_impl.reset();
}

const FullscreenPipelineEntry &FullscreenRenderer::EnsurePipeline(const FullscreenPipelineKey &key)
{
    static const FullscreenPipelineEntry invalid;
    if (!m_impl)
        return invalid;
    return m_impl->pipelines->EnsurePipeline(key);
}

const FullscreenPipelineEntry &FullscreenPipelineCache::EnsurePipeline(const FullscreenPipelineKey &key)
{
    const auto found = m_impl->pipelines.find(key);
    if (found != m_impl->pipelines.end()) {
        found->second.lastUse = ++m_impl->useSequence;
        return found->second.rhi;
    }
    if (m_impl->pipelines.size() == m_impl->capacity) {
        const auto oldest = std::min_element(
            m_impl->pipelines.begin(), m_impl->pipelines.end(),
            [](const auto &left, const auto &right) { return left.second.lastUse < right.second.lastUse; });
        m_impl->DestroyPipeline(oldest->second);
        m_impl->pipelines.erase(oldest);
    }
    auto [it, inserted] = m_impl->pipelines.emplace(key, m_impl->CreatePipeline(key));
    it->second.lastUse = ++m_impl->useSequence;
    return it->second.rhi;
}

void FullscreenRenderer::InvalidateShader(const std::string &shaderName)
{
    if (m_impl)
        m_impl->pipelines->InvalidateShader(shaderName);
}

void FullscreenPipelineCache::InvalidateShader(const std::string &shaderName)
{
    if (shaderName.empty())
        return;

    size_t retired = 0;
    for (auto entry = m_impl->pipelines.begin(); entry != m_impl->pipelines.end();) {
        const bool authorAsset = entry->first.program && entry->first.program->assetGuid == shaderName;
        if (!authorAsset && entry->first.shaderName != shaderName && shaderName != "Fullscreen Triangle") {
            ++entry;
            continue;
        }
        auto pipeline = std::move(entry->second);
        entry = m_impl->pipelines.erase(entry);
        m_impl->DestroyPipeline(pipeline);
        ++retired;
    }
    if (retired > 0) {
        m_impl->host->ReportInfo("FullscreenRenderer: retired " + std::to_string(retired) +
                                 " pipeline revision(s) for shader '" + shaderName + "'");
    }
}

rhi::BindGroupHandle FullscreenRenderer::AllocateBindGroup(rhi::BindingLayoutHandle layout,
                                                           const FullscreenResourceInput *inputs, uint32_t inputCount,
                                                           rhi::SamplerHandle colorSampler)
{
    if (!m_impl || !m_impl->device || m_impl->frameBindGroups.empty())
        return {};
    const uint32_t frame = m_impl->CurrentFrame();

    rhi::BindGroupDesc groupDesc;
    groupDesc.layout = layout;
    groupDesc.lifetime = rhi::BindGroupLifetime::FrameTransient;
    if (!layout.IsValid() || !inputs || inputCount > rhi::BindingLayoutDesc::MaxEntries)
        return {};
    for (uint32_t i = 0; i < inputCount; ++i) {
        const auto &input = inputs[i];
        if (input.buffer.IsValid()) {
            if (groupDesc.bufferCount >= groupDesc.buffers.size())
                return {};
            auto &buffer = groupDesc.buffers[groupDesc.bufferCount++];
            buffer.binding = i;
            buffer.type = rhi::BindingType::StorageBuffer;
            buffer.buffer = input.buffer;
            buffer.offset = input.byteOffset;
            buffer.byteSize = input.byteSize;
            continue;
        }
        if (groupDesc.textureCount >= groupDesc.textures.size())
            return {};
        const bool nearestSampling = input.depthRead || rhi::IsIntegerFormat(input.format);
        auto &texture = groupDesc.textures[groupDesc.textureCount++];
        texture.binding = i;
        texture.type = rhi::BindingType::CombinedTextureSampler;
        texture.texture = input.view;
        texture.sampler =
            input.sampler.IsValid() ? input.sampler : (nearestSampling ? m_impl->nearestSampler : colorSampler);
        texture.depthRead = input.depthRead;
        if (!texture.texture.IsValid() || !texture.sampler.IsValid())
            return {};
    }

    const auto handle = m_impl->device->CreateBindGroup(groupDesc);
    if (handle.IsValid())
        m_impl->frameBindGroups[frame].push_back(handle);
    return handle;
}

void FullscreenRenderer::Draw(rhi::GraphicsCommandEncoder &encoder, const FullscreenPipelineEntry &entry,
                              rhi::BindGroupHandle inputGroup, rhi::BindGroupHandle perViewGroup,
                              rhi::BindGroupHandle globalsGroup, const FullscreenPushConstants &pushConstants,
                              uint32_t pushConstantSize)
{
    if (!m_impl || !entry.pipeline.IsValid())
        return;
    encoder.BindPipeline(entry.pipeline);
    if (inputGroup.IsValid())
        encoder.BindGroup(entry.pipeline, 0, inputGroup);
    if (entry.hasPerView && perViewGroup.IsValid())
        encoder.BindGroup(entry.pipeline, 1, perViewGroup);
    if (entry.hasGlobals)
        encoder.BindGroup(entry.pipeline, 2, globalsGroup);
    encoder.PushConstants(entry.pipeline, rhi::ShaderStage::Fragment, pushConstantSize, pushConstants.values);
    encoder.Draw(3);
}

void FullscreenRenderer::ResetPool()
{
    if (!m_impl || m_impl->frameBindGroups.empty())
        return;
    m_impl->ReleaseBindGroups(m_impl->CurrentFrame());
    m_impl->frameParameters[m_impl->CurrentFrame()].next = 0;
}

std::shared_ptr<const FullscreenProgram> FullscreenRenderer::AcquireProgram(const std::string &name)
{
    return m_impl->host->AcquireProgram(name);
}

FullscreenParameters
FullscreenRenderer::PrepareParameters(const FullscreenProgram &program,
                                      const std::vector<std::pair<std::string, ShaderParameterValue>> &parameters)
{
    const auto schema = nlohmann::json::parse(program.schema);
    if (schema.at("domain") != "fullscreen")
        throw std::invalid_argument("Fullscreen pass requires a shader.fullscreen author program");
    FullscreenParameters result;
    const auto &resources = program.code->resources;
    const auto uniform = std::find_if(resources.begin(), resources.end(),
                                      [](const auto &item) { return item.name == "inx_parameters"; });
    if (uniform == resources.end()) {
        if (!parameters.empty())
            throw std::invalid_argument("Fullscreen program declares no parameters");
        return result;
    }
    result.defaults.resize(uniform->value.byteSize);
    const auto &fields = uniform->value.fields;
    const auto fieldByName = [&](const std::string &name) -> const ShaderValueLayout & {
        const auto field =
            std::find_if(fields.begin(), fields.end(), [&](const auto &item) { return item.name == name; });
        if (field == fields.end())
            throw std::invalid_argument("Fullscreen parameter is not declared: " + name);
        return *field;
    };
    const auto &declarations = schema.at("parameters");
    const auto &inputs = schema.at("inputs");
    for (uint32_t index = 0; index < inputs.size(); ++index) {
        const auto &input = inputs[index];
        if (input.at("kind") == "History2D")
            result.histories.push_back(
                {index, static_cast<uint32_t>(
                            fieldByName("_inx_history_valid_" + input.at("name").get<std::string>()).offset)});
    }
    for (const auto &parameter : declarations) {
        const auto &field = fieldByName(parameter.at("name").get<std::string>());
        const ShaderValueCopy copy(field);
        std::vector<double> components;
        ReadParameterComponents(parameter.at("default"), components);
        if (components.size() * sizeof(uint32_t) != copy.GetSourceBytes())
            throw std::logic_error("Shader parameter schema does not match its compiled layout: " + field.name);
        std::vector<char> source(copy.GetSourceBytes());
        for (size_t index = 0; index < components.size(); ++index)
            StoreParameter(source.data() + index * sizeof(uint32_t), field.scalar, components[index]);
        copy.Write(result.defaults.data(), source.data());
    }
    using Kind = ShaderParameterKind;
    static const std::unordered_map<std::string, Kind> kinds = {
        {"float", Kind::Float},  {"int", Kind::Int},      {"uint", Kind::UInt},    {"bool", Kind::Bool},
        {"vec2", Kind::Vector2}, {"vec3", Kind::Vector3}, {"vec4", Kind::Vector4}, {"color", Kind::Vector4},
        {"mat3", Kind::Matrix3}, {"mat4", Kind::Matrix4},
    };
    for (const auto &parameter : parameters) {
        const auto &name = parameter.first;
        const auto declaration = std::find_if(declarations.begin(), declarations.end(),
                                              [&](const auto &item) { return item.at("name") == name; });
        if (declaration == declarations.end())
            throw std::invalid_argument("Fullscreen parameter is not declared: " + name);
        result.values.push_back(parameter.second.ConvertTo(kinds.at(declaration->at("kind").get<std::string>())));
        result.copies.emplace_back(fieldByName(name));
        result.copies.back().Write(result.defaults.data(), result.values.back().Data());
    }
    return result;
}

void FullscreenRenderer::DrawProgram(rhi::GraphicsCommandEncoder &encoder, const FullscreenPipelineEntry &entry,
                                     const FullscreenResourceInput *inputs, const FullscreenParameters &parameters,
                                     const ShaderParameterValue *values, const FullscreenViewResources &view)
{
    const auto frame = m_impl->CurrentFrame();
    auto &bytes = m_impl->frameParameters[frame].scratch;
    bytes.assign(parameters.defaults.begin(), parameters.defaults.end());
    if (values)
        for (size_t index = 0; index < parameters.copies.size(); ++index)
            parameters.copies[index].Write(bytes.data(), values[index].Data());
    for (const auto &history : parameters.histories) {
        const uint32_t valid = inputs[history.input].historyValid ? 1u : 0u;
        std::memcpy(bytes.data() + history.offset, &valid, sizeof(valid));
    }
    const auto uniform =
        bytes.empty() ? rhi::BufferHandle{} : m_impl->UploadParameters(frame, bytes.data(), bytes.size());
    const std::array<rhi::BufferBinding, 3> uniforms = {{
        {0, rhi::BindingType::UniformBuffer, uniform, 0, bytes.size()},
        view.globals,
        view.camera,
    }};
    const auto &layouts = entry.program->GetBindingLayouts();
    std::array<rhi::BindGroupDesc, rhi::GraphicsPipelineDesc::MaxBindingLayouts> groups;
    for (size_t index = 0; index < layouts.size(); ++index) {
        groups[index].layout = layouts[index];
        groups[index].lifetime = rhi::BindGroupLifetime::FrameTransient;
    }
    const auto &resources = entry.program->GetResources();
    for (size_t index = 0; index < resources.size(); ++index) {
        const auto &resource = resources[index];
        auto &group = groups[resource.set];
        const auto &binding = resource.binding;
        if (binding.type == rhi::BindingType::UniformBuffer) {
            auto source = uniforms[static_cast<size_t>(-1 - entry.resourceInputs[index])];
            source.binding = binding.binding;
            group.buffers[group.bufferCount++] = source;
        } else {
            const auto &input = inputs[entry.resourceInputs[index]];
            const auto sampler = binding.samplerType == rhi::SamplerBindingType::NonFiltering ? m_impl->nearestSampler
                                 : input.sampler.IsValid()                                    ? input.sampler
                                                                                              : m_impl->linearSampler;
            group.textures[group.textureCount++] = {binding.binding, binding.type, input.view, sampler,
                                                    input.depthRead};
        }
    }
    encoder.BindPipeline(entry.pipeline);
    for (uint32_t index = 0; index < layouts.size(); ++index) {
        const auto group = m_impl->device->CreateBindGroup(groups[index]);
        if (!group.IsValid())
            throw std::runtime_error("Fullscreen bind group creation failed");
        m_impl->frameBindGroups[frame].push_back(group);
        encoder.BindGroup(entry.pipeline, index, group);
    }
    encoder.Draw(3);
}

rhi::SamplerHandle FullscreenRenderer::GetLinearSampler() const noexcept
{
    return m_impl ? m_impl->linearSampler : rhi::SamplerHandle{};
}

} // namespace infernux
