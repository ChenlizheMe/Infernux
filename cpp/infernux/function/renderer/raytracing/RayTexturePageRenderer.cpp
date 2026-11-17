#include "RayTexturePageRenderer.h"

#include <function/renderer/rhi/RhiCommand.h>
#include <function/renderer/rhi/RhiTexture.h>
#include <function/renderer/shader/DeviceShaderProgram.h>
#include <function/renderer/shader/FullscreenTriangleSource.h>
#include <function/renderer/shader/ShaderValueCopy.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>

#include <algorithm>
#include <array>
#include <shared_mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infernux::raytracing
{
namespace
{

constexpr std::string_view NormalizeFragmentSource = R"(
struct NormalizeParameters { float4 inverse_size_and_lod; };
ConstantBuffer<NormalizeParameters> parameters;
Texture2D<float4> source_texture;
SamplerState source_sampler;

[shader("fragment")]
float4 fragment_main(float4 position : SV_Position) : SV_Target0 {
    let sample_parameters = parameters.inverse_size_and_lod;
    return source_texture.SampleLevel(source_sampler, position.xy * sample_parameters.xy, sample_parameters.z);
}
)";

ShaderTargetCode CompileProgram(rhi::Device &device)
{
    auto source = std::string(FullscreenTriangleVertexSource);
    source += NormalizeFragmentSource;
    const auto target = device.GetShaderSourceLanguage() == rhi::ShaderSourceLanguage::SpirV ? ShaderCodeTarget::SpirV
                                                                                             : ShaderCodeTarget::Wgsl;
    auto compiled = SlangCompiler::Get().Compile(
        {"inx_ray_texture_page", "Infernux/RayTexturePage.slang", std::move(source)}, {}, {target});
    return std::move(compiled.targets.at(0));
}

uint32_t Entry(const ShaderTargetCode &code, rhi::ShaderStage stage)
{
    const auto found = std::find_if(code.entries.begin(), code.entries.end(),
                                    [stage](const auto &entry) { return entry.stage == stage; });
    if (found == code.entries.end())
        throw std::logic_error("Ray texture normalization shader is missing a graphics stage");
    return static_cast<uint32_t>(found - code.entries.begin());
}

const ShaderResourceLayout &Binding(const DeviceShaderProgram &shader, std::string_view name, rhi::BindingType type)
{
    const auto &resources = shader.GetResources();
    const auto found = std::find_if(resources.begin(), resources.end(), [&](const auto &resource) {
        return resource.name == name && resource.set == 0 && resource.binding.type == type;
    });
    if (found == resources.end())
        throw std::logic_error("Ray texture normalization shader is missing a binding: " + std::string(name));
    return *found;
}

} // namespace

struct RayTexturePageProgram final
{
    RayTexturePageProgram(rhi::Device &device, const ShaderTargetCode &code)
        : device(device), lifetime(device.GetLifetime()),
          shader(device, code, {Entry(code, rhi::ShaderStage::Vertex), Entry(code, rhi::ShaderStage::Fragment)}),
          parameters(Binding(shader, "parameters", rhi::BindingType::UniformBuffer)), parameterCopy(parameters.value),
          sourceBinding(Binding(shader, "source_texture", rhi::BindingType::SampledTexture).binding.binding),
          samplerBinding(Binding(shader, "source_sampler", rhi::BindingType::Sampler).binding.binding)
    {
        if (shader.GetBindingLayouts().size() != 1 || parameterCopy.GetSourceBytes() != 4 * sizeof(float))
            throw std::logic_error("Ray texture normalization has an unexpected reflected layout");

        rhi::SamplerDesc settings;
        settings.addressU = settings.addressV = settings.addressW = rhi::AddressMode::ClampToEdge;
        settings.minFilter = settings.magFilter = settings.mipFilter = rhi::FilterMode::Linear;
        settings.maxLod = 32.0f;
        sampler = device.CreateSampler(settings);
        if (!sampler.IsValid())
            throw std::runtime_error("Ray texture normalization sampler creation failed");
    }

    ~RayTexturePageProgram()
    {
        std::shared_lock lock(lifetime->gate);
        if (!lifetime->alive.load(std::memory_order_acquire))
            return;
        for (auto pipeline : pipelines)
            device.Release(pipeline);
        device.Release(sampler);
    }

    rhi::GraphicsPipelineHandle GetPipeline(rhi::PixelFormat format)
    {
        auto &pipeline = pipelines[format == rhi::PixelFormat::RGBA8Srgb ? 0 : 1];
        if (!pipeline.IsValid()) {
            rhi::GraphicsPipelineDesc description;
            shader.Apply(description);
            description.useDynamicRendering = true;
            description.raster.cullMode = rhi::CullMode::None;
            description.colorTargetCount = description.renderingSignature.colorFormatCount = 1;
            description.colorTargets[0].format = description.renderingSignature.colorFormats[0] = format;
            pipeline = device.CreateGraphicsPipeline(description);
            if (!pipeline.IsValid())
                throw std::runtime_error("Ray texture normalization pipeline creation failed");
        }
        return pipeline;
    }

    rhi::Device &device;
    std::shared_ptr<rhi::DeviceLifetime> lifetime;
    DeviceShaderProgram shader;
    const ShaderResourceLayout &parameters;
    ShaderValueCopy parameterCopy;
    uint32_t sourceBinding;
    uint32_t samplerBinding;
    rhi::SamplerHandle sampler;
    std::array<rhi::GraphicsPipelineHandle, 2> pipelines{};
};

RayTexturePageDraw::RayTexturePageDraw(std::shared_ptr<RayTexturePageProgram> program,
                                       std::shared_ptr<const rhi::TextureGpuView> source,
                                       std::shared_ptr<const rhi::TextureResource> destination) noexcept
    : m_program(std::move(program)), m_source(std::move(source)), m_destination(std::move(destination))
{
}

RayTexturePageDraw::~RayTexturePageDraw()
{
    const auto release = [&] {
        m_program->device.Release(m_group);
        m_program->device.Release(m_constants);
        m_program->device.Release(m_raster.colors[0].view);
    };
    if (const auto &lifetime = m_program->lifetime) {
        std::shared_lock lock(lifetime->gate);
        if (lifetime->alive.load(std::memory_order_acquire))
            release();
    } else {
        release();
    }
}

const rhi::RasterPassDescription &RayTexturePageDraw::Raster() const noexcept
{
    return m_raster;
}

bool RayTexturePageDraw::Record(const rhi::GraphicsCommandEncoder &encoder) const
{
    if (!encoder.HasCompleteDispatch())
        return false;
    encoder.BindPipeline(m_pipeline);
    encoder.BindGroup(m_pipeline, 0, m_group);
    encoder.Draw(3);
    return true;
}

RayTexturePageRenderer::RayTexturePageRenderer(rhi::Device &device) noexcept : m_device(device)
{
}

std::shared_ptr<const RayTexturePageDraw>
RayTexturePageRenderer::Prepare(std::shared_ptr<const rhi::TextureGpuView> source,
                                std::shared_ptr<const rhi::TextureResource> destination, uint32_t width,
                                uint32_t height, uint32_t mip, uint32_t layer, float sourceLod)
{
    if (!source || !source->IsValid() || !destination || !destination->IsValid() || !width || !height ||
        source->GetTexture().Device() != m_device.GetDeviceId() ||
        destination->GetTexture().Device() != m_device.GetDeviceId() ||
        source->GetTexture() == destination->GetTexture())
        throw std::invalid_argument(
            "Ray texture normalization requires separate resident source and destination images");

    const auto &sourceView = source->GetViewDesc();
    const auto &destinationView = destination->GetViewDesc();
    const auto format = destination->GetFormat();
    if (sourceView.dimension != rhi::TextureViewDimension::Texture2D ||
        sourceView.aspect != rhi::TextureAspect::Color || !sourceView.mipCount ||
        !(sourceLod >= 0.0f && sourceLod <= static_cast<float>(sourceView.mipCount - 1)) ||
        destinationView.dimension != rhi::TextureViewDimension::Texture2DArray || mip < destinationView.baseMip ||
        mip - destinationView.baseMip >= destinationView.mipCount || layer < destinationView.baseLayer ||
        layer - destinationView.baseLayer >= destinationView.layerCount ||
        (format != rhi::PixelFormat::RGBA8Srgb && format != rhi::PixelFormat::RGBA8UNorm))
        throw std::invalid_argument("Ray texture normalization requires a valid source LOD and RGBA8 array slice");

    if (!m_program)
        m_program = std::make_shared<RayTexturePageProgram>(m_device, CompileProgram(m_device));
    auto draw = std::shared_ptr<RayTexturePageDraw>(
        new RayTexturePageDraw(m_program, std::move(source), std::move(destination)));
    draw->m_pipeline = m_program->GetPipeline(format);
    draw->m_raster.width = width;
    draw->m_raster.height = height;
    draw->m_raster.colorCount = 1;

    rhi::TextureViewDesc attachment;
    attachment.texture = draw->m_destination->GetTexture();
    attachment.format = format;
    attachment.baseMip = mip;
    attachment.baseLayer = layer;
    attachment.usage = rhi::TextureUsageFlags::ColorAttachment;
    auto &color = draw->m_raster.colors[0];
    color.view = m_device.CreateTextureView(attachment);
    color.format = format;
    if (!color.view.IsValid())
        throw std::runtime_error("Ray texture normalization slice view creation failed");

    const std::array<float, 4> values{1.0f / static_cast<float>(width), 1.0f / static_cast<float>(height), sourceLod,
                                      0.0f};
    std::vector<char> bytes(m_program->parameters.value.byteSize);
    m_program->parameterCopy.Write(bytes.data(), values.data());
    rhi::BufferDesc buffer;
    buffer.byteSize = buffer.initialDataBytes = bytes.size();
    buffer.initialData = bytes.data();
    buffer.usage = rhi::BufferUsageFlags::Uniform;
    buffer.memory = rhi::BufferMemory::Upload;
    draw->m_constants = m_device.CreateBuffer(buffer);
    if (!draw->m_constants.IsValid())
        throw std::runtime_error("Ray texture normalization uniform buffer creation failed");

    rhi::BindGroupDesc group;
    group.layout = m_program->shader.GetBindingLayouts()[0];
    group.buffers[0] = {m_program->parameters.binding.binding, rhi::BindingType::UniformBuffer, draw->m_constants, 0,
                        bytes.size()};
    group.bufferCount = 1;
    group.textures[0] = {
        m_program->sourceBinding, rhi::BindingType::SampledTexture, draw->m_source->GetView(), {}, false};
    group.textures[1] = {m_program->samplerBinding, rhi::BindingType::Sampler, {}, m_program->sampler, false};
    group.textureCount = 2;
    draw->m_group = m_device.CreateBindGroup(group);
    if (!draw->m_group.IsValid())
        throw std::runtime_error("Ray texture normalization binding creation failed");
    return draw;
}

} // namespace infernux::raytracing
