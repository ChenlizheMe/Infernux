#include "TextureRegionRenderer.h"
#include "rhi/RhiDevice.h"
#include "shader/DeviceShaderProgram.h"
#include "shader/FullscreenTriangleSource.h"
#include "shader/ShaderValueCopy.h"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace infernux
{
struct TextureRegionRenderer::Impl
{
    struct Binding
    {
        std::shared_ptr<const rhi::TextureResource> source;
        std::array<int32_t, 4> offset;
        rhi::BufferHandle constants;
        rhi::BindGroupHandle group;
    };
    rhi::Device &device;
    DeviceShaderProgram shader;
    rhi::GraphicsPipelineHandle pipeline;
    uint32_t textureBinding = 0, constantsBinding = 0;
    std::unique_ptr<ShaderValueCopy> copy;
    uint64_t constantBytes = 0;
    std::vector<Binding> bindings;

    Impl(rhi::Device &owner, const ShaderTargetCode &code, rhi::PixelFormat format, rhi::SampleCount samples)
        : device(owner), shader(owner, code, {0, 1})
    {
        for (const auto &resource : shader.GetResources()) {
            if (resource.name == "source_texture")
                textureBinding = resource.binding.binding;
            else if (resource.name == "parameters") {
                constantsBinding = resource.binding.binding;
                constantBytes = resource.value.byteSize;
                copy = std::make_unique<ShaderValueCopy>(resource.value);
            }
        }
        rhi::GraphicsPipelineDesc description;
        shader.Apply(description);
        description.useDynamicRendering = true;
        description.samples = description.renderingSignature.samples = samples;
        description.raster.cullMode = rhi::CullMode::None;
        if (rhi::IsDepthFormat(format)) {
            description.renderingSignature.depthFormat = format;
            if (rhi::IsStencilFormat(format))
                description.renderingSignature.stencilFormat = format;
            description.depth.testEnabled = description.depth.writeEnabled = true;
            description.depth.compare = rhi::CompareFunction::Always;
        } else {
            description.colorTargetCount = description.renderingSignature.colorFormatCount = 1;
            description.colorTargets[0].format = description.renderingSignature.colorFormats[0] = format;
        }
        pipeline = device.CreateGraphicsPipeline(description);
        if (!pipeline.IsValid())
            throw std::runtime_error("Texture-region transfer pipeline creation failed");
    }
    void ClearBindings()
    {
        for (const auto &binding : bindings) {
            device.Release(binding.group);
            device.Release(binding.constants);
        }
        bindings.clear();
    }
    ~Impl()
    {
        ClearBindings();
        device.Release(pipeline);
    }
};

TextureRegionRenderer::TextureRegionRenderer(rhi::Device &device, const ShaderTargetCode &code, rhi::PixelFormat format,
                                             rhi::SampleCount samples)
    : m_impl(std::make_unique<Impl>(device, code, format, samples))
{
}
TextureRegionRenderer::~TextureRegionRenderer() = default;
void TextureRegionRenderer::ClearBindings()
{
    m_impl->ClearBindings();
}

std::string TextureRegionRenderer::ShaderSource(bool depth, bool multisample)
{
    std::string source = R"(
struct Region { int4 offset; };
ConstantBuffer<Region> parameters;)";
    source += FullscreenTriangleVertexSource;
    source += depth ? (multisample ? "DepthTexture2DMS" : "DepthTexture2D")
                    : (multisample ? "Texture2DMS<float4>" : "Texture2D<float4>");
    source += " source_texture;\n[shader(\"fragment\")]\n";
    source += depth ? "float" : "float4";
    source += " fragment_main(float4 position : SV_Position";
    if (multisample)
        source += ", uint sample_index : SV_SampleIndex";
    source += depth ? ") : SV_Depth {\n" : ") : SV_Target0 {\n";
    source += "int2 pixel = int2(position.xy) + parameters.offset.xy;\nreturn source_texture.Load(";
    source += multisample ? "pixel, int(sample_index));\n}\n" : "int3(pixel, 0));\n}\n";
    return source;
}

void TextureRegionRenderer::Record(const rhi::GraphicsCommandEncoder &commands,
                                   std::shared_ptr<const rhi::TextureResource> source, int32_t sourceX, int32_t sourceY,
                                   int32_t destinationX, int32_t destinationY)
{
    auto &s = *m_impl;
    const std::array<int32_t, 4> offset{sourceX - destinationX, sourceY - destinationY, 0, 0};
    auto found = std::find_if(s.bindings.begin(), s.bindings.end(), [&](const auto &binding) {
        return binding.source == source && binding.offset == offset;
    });
    if (found == s.bindings.end()) {
        std::vector<char> bytes(s.constantBytes);
        s.copy->Write(bytes.data(), offset.data());
        rhi::BufferDesc description;
        description.byteSize = description.initialDataBytes = bytes.size();
        description.initialData = bytes.data();
        description.usage = rhi::BufferUsageFlags::Uniform;
        description.memory = rhi::BufferMemory::Upload;
        const auto constants = s.device.CreateBuffer(description);
        rhi::BindGroupDesc group;
        group.layout = s.shader.GetBindingLayouts()[0];
        group.buffers[0] = {s.constantsBinding, rhi::BindingType::UniformBuffer, constants, 0, bytes.size()};
        group.bufferCount = 1;
        group.textures[0] = {s.textureBinding,
                             rhi::BindingType::SampledTexture,
                             source->GetView(),
                             {},
                             rhi::IsDepthFormat(source->GetFormat())};
        group.textureCount = 1;
        s.bindings.push_back({std::move(source), offset, constants, s.device.CreateBindGroup(group)});
        found = std::prev(s.bindings.end());
    }
    commands.BindPipeline(s.pipeline);
    commands.BindGroup(s.pipeline, 0, found->group);
    commands.Draw(3);
}
} // namespace infernux
