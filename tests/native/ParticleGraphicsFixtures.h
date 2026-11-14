#pragma once

#include <function/renderer/EngineGlobals.h>
#include <function/renderer/InxRenderStruct.h>
#include <function/renderer/particle/ParticleGpuOutputRenderer.h>
#include <function/renderer/rhi/RhiDevice.h>
#include <function/resources/ShaderAsset/ShaderAuthorMaterial.h>
#include <function/resources/ShaderAsset/SlangParticleModule.h>
#include <nlohmann/json.hpp>

namespace infernux::particle_test
{

inline const particle::GpuParticleGraphicsPrograms &GeometryPrograms()
{
    static const auto programs = [] {
        particle::GpuParticleGraphicsPrograms result;
        const auto &modules = GetSlangParticleGraphicsModules();
        for (size_t index = 0; index < result.size(); ++index)
            result[index] = std::make_shared<const ShaderCompilation>(SlangCompiler::Get().Compile(
                modules[index].module, {}, {ShaderCodeTarget::SpirV, ShaderCodeTarget::Wgsl}));
        return result;
    }();
    return programs;
}

inline std::shared_ptr<const ShaderProgramArtifact> CreateSurface(bool textured = false, bool zeroCoverage = false)
{
    const std::string texture = textured ? R"slang(
struct ColorImage { Texture2D<float4> texture; SamplerState sampler; };
[[vk::binding(1, 0)]] ColorImage inx_albedo_map;
)slang"
                                         : "";
    const std::string sample = textured ? " * inx_albedo_map.texture.Sample(inx_albedo_map.sampler, input.uv0)" : "";
    const std::string value =
        zeroCoverage ? "float4(0.0)" : "(float4(input.color, input.alpha) + inx_parameters.color)" + sample;
    const auto softValue = [zeroCoverage](const char *depthPlane) {
        return zeroCoverage ? std::string("evaluate(input)")
                            : "inx_particle_apply_soft_coverage(evaluate(input), inx_particle_soft_coverage<" +
                                  std::string(depthPlane) + ">(input))";
    };
    const std::string source = R"slang(import inx_particle;
struct Parameters { float4 color; };
[[vk::binding(0, 0)]] ConstantBuffer<Parameters> inx_parameters;
)slang" + texture + R"slang(
float4 evaluate(InxParticleInput input) {
    return )slang" + value + R"slang(;
}
[shader("fragment")] float4 fragment_main(InxParticleInput input) : SV_Target0 { return evaluate(input); }
[shader("fragment")] float4 fragment_forward_plus(InxParticleInput input) : SV_Target0 { return evaluate(input); }
[shader("fragment")] float4 fragment_soft_depth(InxParticleInput input) : SV_Target0 {
    return )slang" + softValue("true") +
                               R"slang(;
}
[shader("fragment")] float4 fragment_soft_color(InxParticleInput input) : SV_Target0 {
    return )slang" + softValue("false") +
                               R"slang(;
}
[shader("fragment")] float4 fragment_forward_plus_soft_depth(InxParticleInput input) : SV_Target0 {
    return )slang" + softValue("true") +
                               R"slang(;
}
[shader("fragment")] float4 fragment_forward_plus_soft_color(InxParticleInput input) : SV_Target0 {
    return )slang" + softValue("false") +
                               R"slang(;
}
[shader("fragment")] uint2 fragment_picking(InxParticleInput input) : SV_Target0 {
    clip(evaluate(input).a - 0.0001); return inx_particle_owner_id();
}
[shader("fragment")] float2 fragment_motion(InxParticleInput input) : SV_Target0 {
    clip(evaluate(input).a - 0.0001); return input.motion_vector;
}
[shader("fragment")] void fragment_depth(InxParticleInput input) { clip(evaluate(input).a - 0.0001); }
)slang";
    using nlohmann::json;
    json parameters = json::array({{{"name", "color"},
                                    {"kind", "vec4"},
                                    {"default", {0, 0, 0, 0}},
                                    {"stages", {"fragment"}},
                                    {"hdr", false},
                                    {"range", nullptr}}});
    if (textured)
        parameters.push_back(
            {{"name", "albedo_map"}, {"kind", "texture_2d"}, {"default", "white"}, {"stages", {"fragment"}}});
    const json schema{
        {"domain", "particle"}, {"surface", false}, {"parameters", parameters}, {"info", {{"cast_shadows", true}}}};
    const auto compilation = std::make_shared<const ShaderCompilation>(
        SlangCompiler::Get().Compile({"particle_test_surface", "Tests/ParticleSurface.slang", source}, {},
                                     {ShaderCodeTarget::SpirV, ShaderCodeTarget::Wgsl}));
    return std::make_shared<const ShaderProgramArtifact>(
        CreateShaderAuthorMaterial("Tests/ParticleSurface", compilation, schema.dump(), ""));
}

inline const std::shared_ptr<const ShaderProgramArtifact> &Surface(bool textured = false)
{
    static const std::array<std::shared_ptr<const ShaderProgramArtifact>, 2> programs{CreateSurface(false),
                                                                                      CreateSurface(true)};
    return programs[textured ? 1 : 0];
}

struct ViewResources
{
    explicit ViewResources(rhi::Device &owner) : device(owner)
    {
        UniformBufferObject camera{};
        camera.model = camera.view = camera.proj = camera.previousViewProj = camera.inverseViewProj = glm::mat4(1.0f);
        EngineGlobalsUBO frame{};
        const std::array<float, 8> shadow{0, 0, 1, 1, 0, 0, 0, 0};
        const auto upload = [&](const void *data, size_t bytes) {
            rhi::BufferDesc descriptor;
            descriptor.byteSize = bytes;
            descriptor.usage = rhi::BufferUsageFlags::Uniform;
            descriptor.memory = rhi::BufferMemory::Upload;
            descriptor.initialData = data;
            descriptor.initialDataBytes = bytes;
            return device.CreateBuffer(descriptor);
        };
        cameraBuffer = upload(&camera, sizeof(camera));
        globals = upload(&frame, sizeof(frame));
        shadowBuffer = upload(shadow.data(), sizeof(shadow));
        rhi::BindingLayoutDesc layout;
        layout.entries[0] = {5, rhi::BindingType::UniformBuffer, rhi::ShaderStage::Vertex | rhi::ShaderStage::Fragment};
        layout.entries[1] = {7, rhi::BindingType::UniformBuffer, rhi::ShaderStage::Vertex};
        layout.entryCount = 2;
        bindings.layout = device.CreateBindingLayout(layout);
        rhi::BindGroupDesc group;
        group.layout = bindings.layout;
        group.buffers[0] = {5, rhi::BindingType::UniformBuffer, cameraBuffer, 0, sizeof(camera)};
        group.buffers[1] = {7, rhi::BindingType::UniformBuffer, shadowBuffer, 0, sizeof(shadow)};
        group.bufferCount = 2;
        bindings.group = device.CreateBindGroup(group);
        bindings.globals = globals;
    }
    ~ViewResources()
    {
        device.Release(bindings.group);
        device.Release(bindings.layout);
        device.Release(cameraBuffer);
        device.Release(shadowBuffer);
        device.Release(globals);
    }
    rhi::Device &device;
    ShaderViewBindings bindings;
    rhi::BufferHandle cameraBuffer, shadowBuffer, globals;
};

} // namespace infernux::particle_test
