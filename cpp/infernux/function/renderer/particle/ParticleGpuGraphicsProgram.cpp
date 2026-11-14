#include "ParticleGpuGraphicsProgram.h"

#include <function/renderer/rhi/RhiDevice.h>

#include <algorithm>
#include <stdexcept>

namespace infernux::particle
{
namespace
{

const ShaderTargetCode &DeviceTarget(const ShaderCompilation &compilation, ShaderCodeTarget target)
{
    const auto found = std::find_if(compilation.targets.begin(), compilation.targets.end(),
                                    [target](const auto &code) { return code.target == target; });
    if (found == compilation.targets.end())
        throw std::invalid_argument("Particle graphics program has no code for the selected device");
    return *found;
}

uint32_t EntryIndex(const ShaderTargetCode &program, std::string_view name)
{
    const auto found = std::find_if(program.entries.begin(), program.entries.end(),
                                    [name](const auto &entry) { return entry.name == name; });
    if (found == program.entries.end())
        throw std::invalid_argument("Particle graphics program has no compiled entry: " + std::string(name));
    return static_cast<uint32_t>(found - program.entries.begin());
}

uint8_t StateSignature(const GpuBillboardMaterialState &state)
{
    return static_cast<uint8_t>((state.blendEnabled ? 1u : 0u) | (state.depthTestEnabled ? 2u : 0u) |
                                (state.depthWriteEnabled ? 4u : 0u) | (state.premultipliedAlpha ? 8u : 0u));
}

GpuBillboardMaterialState PassState(ShaderCompileTarget target, const GpuBillboardMaterialState &surface)
{
    switch (target) {
    case ShaderCompileTarget::Forward:
    case ShaderCompileTarget::ForwardPlus:
        return surface;
    case ShaderCompileTarget::Depth:
    case ShaderCompileTarget::Shadow:
    case ShaderCompileTarget::Picking:
        return {3000, false, true, true, false};
    case ShaderCompileTarget::Motion:
        return {3000, false, true, false, false};
    default:
        throw std::invalid_argument("Particle output does not define the requested material pass");
    }
}

} // namespace

ParticleGpuGraphicsProgram::ParticleGpuGraphicsProgram(rhi::Device &device,
                                                       std::shared_ptr<const ShaderCompilation> geometry,
                                                       std::shared_ptr<const ShaderProgramArtifact> surface,
                                                       rhi::BindingLayoutHandle materialLayout, rhi::CullMode cullMode)
    : m_device(device), m_geometry(std::move(geometry)), m_surface(std::move(surface)),
      m_materialLayout(materialLayout), m_cullMode(cullMode)
{
    const auto target = device.GetShaderSourceLanguage() == rhi::ShaderSourceLanguage::SpirV ? ShaderCodeTarget::SpirV
                                                                                             : ShaderCodeTarget::Wgsl;
    m_geometryTarget = &DeviceTarget(*m_geometry, target);
    m_surfaceTarget = &DeviceTarget(*m_surface->compilation, target);
}

ParticleGpuGraphicsProgram::~ParticleGpuGraphicsProgram()
{
    // RHI release retains backend objects consumed by accepted submissions.
    // Keep stage entry names and modules alive through pipeline retirement.
    for (const auto &pipeline : m_pipelines)
        m_device.Release(pipeline.handle);
}

rhi::GraphicsPipelineHandle ParticleGpuGraphicsProgram::Get(const MaterialPassPipelineDescriptor &pass,
                                                            const GpuBillboardMaterialState &materialState,
                                                            const ShaderViewBindings &perView,
                                                            rhi::BindingLayoutHandle geometryLayout,
                                                            uint8_t softDepthKind)
{
    const auto state = PassState(pass.target, materialState);
    const auto signature = StateSignature(state);
    const auto existing = std::find_if(m_pipelines.begin(), m_pipelines.end(), [&](const auto &pipeline) {
        return pipeline.pass == pass && pipeline.perViewLayout == perView.layout &&
               pipeline.geometryLayout == geometryLayout && pipeline.state == signature &&
               pipeline.softDepthKind == softDepthKind;
    });
    if (existing != m_pipelines.end()) {
        auto &owners = existing->viewGlobals;
        if (std::find(owners.begin(), owners.end(), perView.globals) == owners.end())
            owners.push_back(perView.globals);
        return existing->handle;
    }

    const auto *variant = m_surface->FindVariant(pass.target);
    if (!variant)
        return {}; // This material has not declared this optional pass.
    const char *vertex = pass.target == ShaderCompileTarget::Motion   ? "vertex_motion"
                         : pass.target == ShaderCompileTarget::Shadow ? "vertex_shadow"
                                                                      : "vertex_main";
    const std::vector<ShaderStageSelection> entries = {
        {*m_geometryTarget, EntryIndex(*m_geometryTarget, vertex)},
        {*m_surfaceTarget, EntryIndex(*m_surfaceTarget, softDepthKind == 1   ? "fragment_soft_depth"
                                                        : softDepthKind == 2 ? "fragment_soft_color"
                                                                             : variant->fragmentEntryPoint)},
    };
    auto program = std::make_unique<DeviceShaderProgram>(
        m_device, entries, std::vector<rhi::BindingLayoutHandle>{m_materialLayout, perView.layout, geometryLayout});
    rhi::GraphicsPipelineDesc descriptor;
    program->Apply(descriptor);
    pass.ApplyRenderingContract(descriptor);
    descriptor.raster.cullMode = m_cullMode;
    descriptor.raster.frontFace = rhi::FrontFace::Clockwise;
    pass.ApplyRasterContract(descriptor);
    descriptor.depth.testEnabled = state.depthTestEnabled && pass.depthFormat != rhi::PixelFormat::Undefined;
    descriptor.depth.writeEnabled =
        state.depthWriteEnabled && !pass.depthReadOnly && pass.depthFormat != rhi::PixelFormat::Undefined;
    descriptor.samples = pass.samples;
    descriptor.colorTargetCount = static_cast<uint32_t>(pass.colorFormats.size());
    for (uint32_t index = 0; index < descriptor.colorTargetCount; ++index) {
        descriptor.colorTargets[index].format = pass.colorFormats[index];
        descriptor.colorTargets[index].blendEnabled = state.blendEnabled;
        descriptor.colorTargets[index].premultipliedAlpha = state.premultipliedAlpha;
    }
    const auto pipeline = m_device.CreateGraphicsPipeline(descriptor);
    if (!pipeline.IsValid())
        throw std::runtime_error("Particle graphics pipeline creation failed");
    m_pipelines.push_back({pass,
                           perView.layout,
                           geometryLayout,
                           signature,
                           softDepthKind,
                           std::move(program),
                           pipeline,
                           {perView.globals}});
    return pipeline;
}

void ParticleGpuGraphicsProgram::ReleaseView(rhi::BufferHandle globals) noexcept
{
    m_pipelines.erase(std::remove_if(m_pipelines.begin(), m_pipelines.end(),
                                     [&](auto &pipeline) {
                                         auto &owners = pipeline.viewGlobals;
                                         owners.erase(std::remove(owners.begin(), owners.end(), globals), owners.end());
                                         if (!owners.empty())
                                             return false;
                                         m_device.Release(pipeline.handle);
                                         return true;
                                     }),
                      m_pipelines.end());
}

} // namespace infernux::particle
