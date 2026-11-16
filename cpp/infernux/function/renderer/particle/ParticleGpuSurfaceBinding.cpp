#include "ParticleGpuSurfaceBinding.h"

#include <core/config/EngineConfig.h>
#include <core/types/ColorSpace.h>
#include <function/renderer/rhi/GpuRetirementQueue.h>
#include <function/renderer/shader/MaterialValueCopy.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <stdexcept>

#include <algorithm>
#include <cstring>

namespace infernux::particle
{

ParticleGpuSurfaceBinding::~ParticleGpuSurfaceBinding()
{
    Destroy();
}

bool ParticleGpuSurfaceBinding::Create(rhi::Device &device, std::shared_ptr<const ShaderProgramArtifact> shaderProgram,
                                       std::shared_ptr<InxMaterial> material,
                                       GpuBillboardMaterialState fallbackMaterial, ParticleOutputSemantics semantics,
                                       GpuBillboardTextureResolver textureResolver, GpuRetirementQueue *deletionQueue)
{
    Destroy();
    if (!shaderProgram || !shaderProgram->IsValid() || !shaderProgram->compilation ||
        shaderProgram->domain != ShaderProgramDomain::ParticleSprite)
        return false;

    const auto fail = [this]() {
        Destroy();
        return false;
    };

    m_device = &device;
    m_shaderProgram = std::move(shaderProgram);
    m_material = std::move(material);
    m_fallbackMaterial = fallbackMaterial;
    m_semantics = semantics;
    m_textureResolver = std::move(textureResolver);
    m_deletionQueue = deletionQueue;
    const auto codeTarget = device.GetShaderSourceLanguage() == rhi::ShaderSourceLanguage::SpirV
                                ? ShaderCodeTarget::SpirV
                                : ShaderCodeTarget::Wgsl;
    const auto &targets = m_shaderProgram->compilation->targets;
    const auto target = std::find_if(targets.begin(), targets.end(),
                                     [codeTarget](const auto &value) { return value.target == codeTarget; });
    if (target == targets.end())
        throw std::invalid_argument("Particle material has no target for the active device");
    m_target = &*target;
    if (!m_material)
        m_material = std::make_shared<InxMaterial>();
    m_material->SynchronizeShaderPropertyDefaults(*m_shaderProgram);

    rhi::BindingLayoutDesc layoutDesc;
    for (const auto &resource : m_target->resources) {
        if (resource.set != 0)
            continue;
        layoutDesc.entries.at(layoutDesc.entryCount++) = resource.binding;
        if (resource.binding.type == rhi::BindingType::UniformBuffer && resource.binding.binding == 0) {
            m_materialBufferSize = static_cast<uint32_t>((resource.value.byteSize + 15u) & ~uint64_t{15u});
            for (const auto &field : resource.value.fields)
                m_memberCopies.emplace_back(field.name, ShaderValueCopy(field, ShaderMatrixOrder::ColumnMajor));
        }
    }
    for (const auto &property : m_shaderProgram->properties) {
        if (!property.IsTexture())
            continue;
        const auto image =
            std::find_if(m_target->resources.begin(), m_target->resources.end(), [&](const auto &resource) {
                return resource.set == 0 && resource.name == property.name &&
                       resource.binding.type == rhi::BindingType::SampledTexture;
            });
        const auto sampler =
            std::find_if(m_target->resources.begin(), m_target->resources.end(), [&](const auto &resource) {
                return resource.set == 0 && resource.name == property.name &&
                       resource.binding.type == rhi::BindingType::Sampler;
            });
        if (image == m_target->resources.end() || sampler == m_target->resources.end())
            throw std::logic_error("Particle texture is absent from the compiled target: " + property.name);
        m_textures.push_back(
            {image->binding.binding, sampler->binding.binding, property.name, property.textureDefault});
    }
    m_layout = device.CreateBindingLayout(layoutDesc);
    if (!m_layout.IsValid())
        return fail();
    if (!RefreshMaterialBuffer(true) || !RefreshTextureBindings(true)) {
        Destroy();
        return false;
    }
    return true;
}

void ParticleGpuSurfaceBinding::Destroy() noexcept
{
    if (m_device) {
        RetireBindGroup(m_group);
        RetireBuffer(m_materialBuffer);
        m_device->Release(m_layout);
    }
    m_device = nullptr;
    m_shaderProgram.reset();
    m_material.reset();
    m_fallbackMaterial = {};
    m_semantics = {};
    m_textureResolver = {};
    m_deletionQueue = nullptr;
    m_layout = {};
    m_group = {};
    m_materialBuffer = {};
    m_materialBytes.clear();
    m_target = nullptr;
    m_materialBufferSize = 0;
    m_memberCopies.clear();
    m_textures.clear();
    m_materialVersion = 0;
    m_materialVersionInitialized = false;
}

bool ParticleGpuSurfaceBinding::IsValid() const noexcept
{
    return m_device && m_shaderProgram && m_layout.IsValid() && m_group.IsValid();
}

GpuBillboardMaterialState ParticleGpuSurfaceBinding::ResolveMaterialState() const noexcept
{
    GpuBillboardMaterialState state = m_fallbackMaterial;
    if (m_material && !m_material->IsDeleted()) {
        const auto &renderState = m_material->GetRenderState();
        state = {renderState.renderQueue, renderState.blendEnable, renderState.depthTestEnable,
                 renderState.depthWriteEnable,
                 renderState.srcColorBlendFactor == MaterialBlendFactor::One &&
                     renderState.dstColorBlendFactor == MaterialBlendFactor::OneMinusSourceAlpha};
    }
    if (m_semantics.softParticles) {
        state.renderQueue = std::max(state.renderQueue, EngineConfig::Get().transparentQueueMin);
        state.blendEnabled = true;
        state.depthWriteEnabled = false;
    }
    return state;
}

float ParticleGpuSurfaceBinding::ResolveMaterialFloat(const char *name, float fallback) const noexcept
{
    if (!name || !m_material || m_material->IsDeleted())
        return fallback;
    const auto *property = m_material->GetProperty(name);
    if (!property || property->type != MaterialPropertyType::Float)
        return fallback;
    const auto *value = std::get_if<float>(&property->value);
    return value ? *value : fallback;
}

std::string ParticleGpuSurfaceBinding::ResolveMaterialTextureGuid(const TextureBindingState &binding) const
{
    if (!m_material || m_material->IsDeleted())
        return binding.defaultGuid;
    const auto *property = m_material->GetProperty(binding.name);
    if (!property || property->type != MaterialPropertyType::Texture2D)
        return binding.defaultGuid;
    const auto *value = std::get_if<std::string>(&property->value);
    return value && !value->empty() ? *value : binding.defaultGuid;
}

void ParticleGpuSurfaceBinding::RetireBindGroup(rhi::BindGroupHandle group)
{
    if (!m_device || !group.IsValid())
        return;
    auto release = [device = m_device, group] { device->Release(group); };
    if (m_deletionQueue)
        m_deletionQueue->Retire(std::move(release));
    else
        release();
}

void ParticleGpuSurfaceBinding::RetireBuffer(rhi::BufferHandle buffer)
{
    if (!m_device || !buffer.IsValid())
        return;
    auto release = [device = m_device, buffer] { device->Release(buffer); };
    if (m_deletionQueue)
        m_deletionQueue->Retire(std::move(release));
    else
        release();
}

void ParticleGpuSurfaceBinding::RetireTexture(std::shared_ptr<const rhi::TextureGpuView> gpuView)
{
    if (!gpuView)
        return;
    auto release = [gpuView = std::move(gpuView)]() mutable { gpuView.reset(); };
    if (m_deletionQueue)
        m_deletionQueue->Retire(std::move(release));
    else
        release();
}

bool ParticleGpuSurfaceBinding::RefreshMaterialBuffer(bool force)
{
    if (m_materialBufferSize == 0)
        return true;
    if (!m_device || !m_layout.IsValid())
        return false;
    const uint64_t version = m_material && !m_material->IsDeleted() ? m_material->GetVersion() : 0;
    if (!force && m_materialVersionInitialized && version == m_materialVersion)
        return true;

    std::vector<uint8_t> bytes(m_materialBufferSize, 0);
    for (const auto &[name, copy] : m_memberCopies) {
        const auto &property = *m_material->GetProperty(name);
        WriteMaterialValue(copy, bytes.data(), property.type, property.value, name);
    }
    if (m_materialBuffer.IsValid() && bytes == m_materialBytes) {
        m_materialVersion = version;
        m_materialVersionInitialized = true;
        return true;
    }
    rhi::BufferDesc desc;
    desc.byteSize = bytes.size();
    desc.usage = rhi::BufferUsageFlags::Uniform;
    desc.memory = rhi::BufferMemory::Upload;
    desc.initialData = bytes.data();
    desc.initialDataBytes = bytes.size();
    const auto buffer = m_device->CreateBuffer(desc);
    if (!buffer.IsValid())
        return false;
    if (m_group.IsValid()) {
        const auto group = CreateBindGroup(m_textures, buffer);
        if (!group.IsValid()) {
            m_device->Release(buffer);
            return false;
        }
        RetireBindGroup(m_group);
        m_group = group;
    }
    RetireBuffer(m_materialBuffer);
    m_materialBuffer = buffer;
    m_materialBytes = std::move(bytes);
    m_materialVersion = version;
    m_materialVersionInitialized = true;
    return true;
}

rhi::BindGroupHandle ParticleGpuSurfaceBinding::CreateBindGroup(const std::vector<TextureBindingState> &textures,
                                                                rhi::BufferHandle materialBuffer) const
{
    rhi::BindGroupDesc descriptor;
    descriptor.layout = m_layout;
    if (materialBuffer.IsValid())
        descriptor.buffers[descriptor.bufferCount++] = {0, rhi::BindingType::UniformBuffer, materialBuffer, 0,
                                                        m_materialBufferSize};
    for (const auto &binding : textures) {
        descriptor.textures[descriptor.textureCount++] = {binding.binding, rhi::BindingType::SampledTexture,
                                                          binding.texture};
        descriptor.textures[descriptor.textureCount++] = {
            binding.samplerBinding, rhi::BindingType::Sampler, {}, binding.sampler};
    }
    return m_device->CreateBindGroup(descriptor);
}

bool ParticleGpuSurfaceBinding::RebuildBindGroup()
{
    const auto group = CreateBindGroup(m_textures, m_materialBuffer);
    if (!group.IsValid())
        return false;
    RetireBindGroup(m_group);
    m_group = group;
    return true;
}

bool ParticleGpuSurfaceBinding::RefreshTextureBindings(bool force)
{
    if (m_textures.empty())
        return m_group.IsValid() || RebuildBindGroup();
    if (!m_textureResolver)
        return false;

    auto candidate = m_textures;
    const uint64_t textureAssetRevision =
        m_material && !m_material->IsDeleted() ? m_material->GetTextureAssetRevision() : 0;
    std::vector<size_t> changed;
    changed.reserve(candidate.size());
    for (size_t index = 0; index < candidate.size(); ++index) {
        auto &binding = candidate[index];
        const std::string textureGuid = ResolveMaterialTextureGuid(binding);
        GpuBillboardTextureLease lease;
        bool usingFallback = false;
        if (!force && (!binding.pending || !binding.fallback) && textureGuid == binding.requestedGuid &&
            binding.gpuSlot && (!binding.fallback || binding.requestedTextureAssetRevision == textureAssetRevision) &&
            !binding.gpuSlot->NeedsRefresh()) {
            auto published = binding.gpuSlot->Acquire();
            if (published && published->IsValid()) {
                binding.pending = false;
                if (binding.gpuView && binding.gpuView->GetRevision() == published->GetRevision() &&
                    binding.gpuView->GetSourceId() == published->GetSourceId())
                    continue;
                lease.status = GpuBillboardTextureStatus::Ready;
                lease.texture = published->GetView();
                lease.sampler = published->GetSampler();
                lease.gpuSlot = binding.gpuSlot;
                lease.gpuView = std::move(published);
                usingFallback = binding.fallback;
            }
        }
        if (lease.status != GpuBillboardTextureStatus::Ready)
            lease = m_textureResolver(textureGuid, binding.name, GpuParticleTextureRequest::Poll);
        const bool pending = lease.status == GpuBillboardTextureStatus::Pending;
        if (pending && textureGuid == binding.requestedGuid && binding.gpuView && binding.gpuView->IsValid() &&
            binding.texture.IsValid() && binding.sampler.IsValid()) {
            binding.pending = true;
            continue;
        }
        if (lease.status != GpuBillboardTextureStatus::Ready || !lease.texture.IsValid() || !lease.sampler.IsValid() ||
            !lease.gpuView || !lease.gpuView->IsValid()) {
            const std::string fallbackGuid = !binding.defaultGuid.empty() && binding.defaultGuid != textureGuid
                                                 ? binding.defaultGuid
                                                 : std::string{};
            lease = m_textureResolver(fallbackGuid, binding.name, GpuParticleTextureRequest::Poll);
            usingFallback = true;
        }
        if (lease.status != GpuBillboardTextureStatus::Ready || !lease.texture.IsValid() || !lease.sampler.IsValid() ||
            !lease.gpuView || !lease.gpuView->IsValid()) {
            if (binding.texture.IsValid() && binding.sampler.IsValid()) {
                binding.pending = pending;
                continue;
            }
            return m_group.IsValid();
        }
        // The displayed fallback's revision does not describe the requested
        // asset. Remember a failed request until its texture is invalidated,
        // without rebuilding identical descriptors or retrying every frame.
        binding.pending = pending;
        binding.fallback = usingFallback;
        binding.requestedTextureAssetRevision = textureAssetRevision;
        if (!force && textureGuid == binding.requestedGuid && binding.gpuView &&
            binding.gpuView->GetSourceId() == lease.gpuView->GetSourceId() &&
            binding.gpuView->GetRevision() == lease.gpuView->GetRevision())
            continue;

        binding.texture = lease.texture;
        binding.sampler = lease.sampler;
        binding.gpuSlot = std::move(lease.gpuSlot);
        binding.gpuView = std::move(lease.gpuView);
        binding.requestedGuid = textureGuid;
        changed.push_back(index);
    }
    if (changed.empty()) {
        m_textures = std::move(candidate);
        return m_group.IsValid();
    }

    const auto group = CreateBindGroup(candidate, m_materialBuffer);
    if (!group.IsValid())
        return false;
    RetireBindGroup(m_group);
    m_group = group;
    for (const size_t index : changed)
        RetireTexture(std::move(m_textures[index].gpuView));
    m_textures = std::move(candidate);
    return true;
}

} // namespace infernux::particle
