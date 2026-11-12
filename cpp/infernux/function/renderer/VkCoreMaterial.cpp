/**
 * @file VkCoreMaterial.cpp
 * @brief InxVkCoreModular — Material system, lighting, and buffer accessors
 *
 * Split from InxVkCoreModular.cpp for maintainability.
 * Contains: UpdateMaterialUBO, CreateBuffer,
 *           InitializeMaterialSystem, RefreshMaterialPipeline,
 *           SetAmbientColor, UpdateLightingUBO,
 *           GetObjectBuffer, GetUniformBuffer, GetShaderModule.
 */

#include "InxError.h"
#include "InxVkCoreModular.h"
#include "MsaaPolicy.h"
#include "VertexInputFilter.h"
#include "gui/GPUMaterialPreview.h"
#include "gui/GPUMeshPreview.h"
#include "rhi/RhiComputeBuffer.h"
#include "rhi/RhiRenderTexture.h"
#include "vk/DescriptorBindTrace.h"
#include "vk/MaterialRenderStateVulkan.h"
#include "vk/RhiVulkanTypes.h"
#include "vk/VkPipelineHelpers.h"
#include "vk/VkRenderUtils.h"

#include <function/renderer/shader/ShaderProgram.h>
#include <function/renderer/shader/ShaderReflection.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxFileLoader/InxShaderLoader.hpp>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxTexture/InxTexture.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>

#include <algorithm>
#include <array>
#include <glm/glm.hpp>
#include <unordered_set>

namespace infernux
{

namespace
{

ShadowViewGpuData PackShadowView(const lighting::ShadowView &view, uint32_t atlasSize)
{
    ShadowViewGpuData result{};
    result.viewProjection = view.viewProjection;
    result.atlasScaleOffset = view.atlas.ScaleOffset(atlasSize);
    result.depthTexel = glm::vec4(view.nearPlane, view.farPlane, view.worldUnitsPerTexel, view.filterRadiusTexels);
    result.splitData = glm::vec4(view.splitNear, view.splitFar, 0.0f, 0.0f);
    result.metadata = glm::uvec4(static_cast<uint32_t>(view.type), view.subView, 0u, 0u);
    return result;
}

} // namespace

// ============================================================================
// Material UBO Management
// ============================================================================

void InxVkCoreModular::UpdateMaterialUBO(InxMaterial &material)
{
    const bool hasPendingTextures = m_materialPipelineManagerInitialized &&
                                    m_materialPipelineManager.HasPendingTextureProperties(material.GetMaterialKey());
    if (!material.IsPropertiesDirty() && !hasPendingTextures) {
        return;
    }

    if (m_materialPipelineManagerInitialized) {
        MaterialRenderData *renderData = m_materialPipelineManager.GetRenderData(material.GetMaterialKey());
        if (renderData && renderData->isValid && renderData->shaderProgram &&
            renderData->descriptorSet != VK_NULL_HANDLE &&
            m_materialPipelineManager.IsDescriptorSetLive(renderData->descriptorSet)) {
            const bool published =
                m_materialPipelineManager.UpdateMaterialProperties(material.GetMaterialKey(), material);
            if (published && !m_materialPipelineManager.HasPendingTextureProperties(material.GetMaterialKey())) {
                material.ClearPropertiesDirty();
            }
            return;
        }
    }

    // Material GPU data is owned exclusively by the descriptor manager. A
    // material that has not acquired valid render data must keep its dirty
    // state until its first complete publication; no shared scratch-UBO writes.
}

// ============================================================================
// Material / Pipeline System
// ============================================================================

void InxVkCoreModular::CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                                    VkBuffer &buffer, VmaAllocation &allocation)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocator allocator = m_backend.Device().GetVmaAllocator();
    VmaAllocationCreateInfo allocCreateInfo{};

    if (properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
        if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
            allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            allocCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        } else {
            allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        }
    } else if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocCreateInfo.requiredFlags = properties;
        if (usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) {
            allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        } else {
            allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        }
    } else {
        allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    }

    VkResult result = vmaCreateBuffer(allocator, &bufferInfo, &allocCreateInfo, &buffer, &allocation, nullptr);
    if (result != VK_SUCCESS) {
        INXLOG_ERROR("CreateBuffer: failed to create buffer via VMA");
        buffer = VK_NULL_HANDLE;
        allocation = VK_NULL_HANDLE;
    }
}

void InxVkCoreModular::InitializeMaterialSystem()
{
    if (m_materialSystemInitialized) {
        return;
    }

    AssetRegistry::Instance().InitializeBuiltinMaterials();

    if (!m_materialPipelineManagerInitialized) {
        // A retry/reinitialization must not carry shader-program publications
        // owned by the previous manager generation into the new one.
        ReleaseMaterialPassResolutionCache();

        // Use SceneRenderTarget-compatible formats: HDR R16G16B16A16_SFLOAT color + device depth format
        VkFormat colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        VkFormat depthFormat = m_backend.Device().FindDepthFormat();
        const auto colorSamples = m_backend.Device().GetImageSampleCountMask(
            colorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        const auto resolveSamples = m_backend.Device().GetImageSampleCountMask(
            colorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        const auto depthSamples =
            m_backend.Device().GetImageSampleCountMask(depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
        const auto supportedSamples = GetSceneTargetSampleCountMask(colorSamples, resolveSamples, depthSamples);
        const int requestedSamples = static_cast<int>(m_msaaSampleCount);
        if (!SupportsMsaaSampleCount(supportedSamples, requestedSamples)) {
            const int fallbackSamples = SelectSupportedMsaaAtOrBelow(supportedSamples, requestedSamples);
            if (fallbackSamples == 0) {
                throw std::runtime_error(
                    "No common MSAA sample count is supported by the scene color and depth formats");
            }
            INXLOG_WARN("Default ", requestedSamples,
                        "x MSAA is unsupported for the HDR color/depth target pair; using ", fallbackSamples,
                        "x. Runtime requests remain strict and will not silently downgrade.");
            m_msaaSampleCount = rhi::ToVkSampleCount(ToRhiSampleCount(fallbackSamples));
        }
        m_materialPipelineManager.Initialize(
            m_backend.Device().GetVmaAllocator(), GetDevice(), GetPhysicalDevice(), colorFormat, depthFormat,
            m_msaaSampleCount, m_shaderCache.GetProgramCache(), &m_deletionQueue,
            m_backend.Device().IsDescriptorIndexingEnabled(), &m_backend.Device().GetRhiDevice().GetDescriptorManager(),
            rhi::ComputeDeviceShaderContractKey(m_backend.Device().GetCapabilities().portable,
                                                m_backend.Device().GetCapabilities().BackendName()));
        m_materialPipelineManagerInitialized = true;

        auto &materialDescriptors = m_materialPipelineManager.GetDescriptorManager();
        auto &rhiDevice = m_backend.Device().GetRhiDevice();
        materialDescriptors.SetBindlessMaterialMode(rhiDevice.GetBindlessTextureTableBinding().IsValid());
        materialDescriptors.SetBindlessTextureResolver(
            [device = &rhiDevice](const std::shared_ptr<const rhi::TextureGpuView> &view) -> rhi::ResourceIndex {
                return device->PublishBindlessTexture(view);
            });
        materialDescriptors.SetBufferResolver(
            [device =
                 &rhiDevice](const std::shared_ptr<rhi::ComputeBufferAllocation> &buffer) -> VkDescriptorBufferInfo {
                VkDescriptorBufferInfo result{};
                if (!buffer || buffer->GetBuffer().Device() != device->GetDeviceId())
                    return result;
                result.buffer = device->Resolve(buffer->GetBuffer());
                result.offset = 0;
                result.range = buffer->GetByteSize();
                return result;
            });
        materialDescriptors.SetBufferCapture([this](const std::shared_ptr<rhi::ComputeBuffer> &buffer) {
            if (!buffer)
                return std::shared_ptr<rhi::ComputeBufferAllocation>{};
            if (!m_drawBufferBindings)
                return buffer->GetAllocation(); // Standalone material and preview preparation.
            const auto found = m_drawBufferBindings->find(buffer.get());
            if (found == m_drawBufferBindings->end())
                throw std::logic_error("Material compute input was not captured by its render view");
            return found->second;
        });

        auto whiteSlot = m_textureCache.Find("white", m_ensureFrameCounter);
        auto whiteTex = whiteSlot ? whiteSlot->Acquire() : nullptr;
        if (whiteTex) {
            auto &rhiDevice = m_backend.Device().GetRhiDevice();
            m_materialPipelineManager.SetDefaultTexture(rhiDevice.Resolve(whiteTex->GetView()),
                                                        rhiDevice.Resolve(whiteTex->GetSampler()), whiteTex);
        }

        auto normalSlot = m_textureCache.Find("_default_normal", m_ensureFrameCounter);
        auto normalTex = normalSlot ? normalSlot->Acquire() : nullptr;
        if (normalTex) {
            auto &rhiDevice = m_backend.Device().GetRhiDevice();
            m_materialPipelineManager.SetDefaultNormalTexture(rhiDevice.Resolve(normalTex->GetView()),
                                                              rhiDevice.Resolve(normalTex->GetSampler()), normalTex);
        }

        auto blackSlot = m_textureCache.Find("_default_black", m_ensureFrameCounter);
        auto blackTex = blackSlot ? blackSlot->Acquire() : nullptr;
        if (blackTex) {
            auto &rhiDevice = m_backend.Device().GetRhiDevice();
            m_materialPipelineManager.SetDefaultBlackTexture(rhiDevice.Resolve(blackTex->GetView()),
                                                             rhiDevice.Resolve(blackTex->GetSampler()), blackTex);
        }

        // Set up texture resolver for material Texture2D properties
        // Delegates to ResolveTextureForMaterial which uses GUID-based cache keys.
        m_materialPipelineManager.SetTextureResolver(
            [this](const std::string &textureRef, const std::string &bindingName,
                   const MaterialTextureSampler *sampler) -> TextureResolveResult {
                return ResolveTextureForMaterial(textureRef, bindingName, sampler);
            });
        m_materialPipelineManager.GetDescriptorManager().SetRenderTextureResolver(
            [this](const std::shared_ptr<rhi::RenderTexture> &texture) {
                MaterialDescriptorSet::TextureBinding binding;
                binding.gpuSlot = texture->GetSampledColorSlot();
                binding.gpuView = binding.gpuSlot->Acquire();
                auto &device = m_backend.Device().GetRhiDevice();
                binding.imageView = device.Resolve(binding.gpuView->GetView());
                binding.sampler = device.Resolve(binding.gpuView->GetSampler());
                return binding;
            });
    }

    // Pre-build error material pipeline (unlit magenta-black checkerboard).
    auto errorMaterial = AssetRegistry::Instance().GetBuiltinMaterial("ErrorMaterial");
    if (errorMaterial) {
        const std::string &errVertId = errorMaterial->GetVertShaderName();
        const std::string &errFragId = errorMaterial->GetFragShaderName();

        const auto *errVertCode = m_shaderCache.FindVertCode(errVertId);
        const auto *errFragCode = m_shaderCache.FindFragCode(errFragId);

        if (errVertCode && errFragCode) {
            auto *renderData = m_materialPipelineManager.GetOrCreateRenderDataWithReflection(
                errorMaterial, *errVertCode, *errFragCode, ShaderProgramKey{{errVertId, errFragId}, 0});
            if (renderData && renderData->isValid) {
                INXLOG_INFO("Error material pipeline created successfully (shaders: ", errVertId, "/", errFragId, ")");
            } else {
                INXLOG_WARN("InitializeMaterialSystem: error material pipeline deferred to lazy build");
            }
        } else {
            INXLOG_DEBUG("InitializeMaterialSystem: error shader SPIR-V not yet in cache "
                         "(vert='",
                         errVertId, "', frag='", errFragId, "'), will be built lazily on first use");
        }
    }

    m_materialSystemInitialized = true;
}

bool InxVkCoreModular::CommitMaterialPipelineGeneration(VkSampleCountFlagBits newSampleCount)
{
    if (!m_materialPipelineManagerInitialized) {
        m_msaaSampleCount = newSampleCount;
        return true;
    }

    if (!m_materialPipelineManager.ReconfigureSampleCount(newSampleCount))
        return false;
    m_msaaSampleCount = newSampleCount;

    // Preview render targets cache a render pass / framebuffer that must stay
    // compatible with the material pipelines' MSAA sample count. Keep the old
    // previews alive until every queue that may reference them has completed.
    if (m_gpuMaterialPreview) {
        std::shared_ptr<GPUMaterialPreview> retired(m_gpuMaterialPreview.release());
        m_deletionQueue.Retire([retired = std::move(retired)] {});
    }
    if (m_gpuMeshPreview) {
        std::shared_ptr<GPUMeshPreview> retired(m_gpuMeshPreview.release());
        m_deletionQueue.Retire([retired = std::move(retired)] {});
    }
    if (m_gpuAnimationPreview) {
        std::shared_ptr<GPUMeshPreview> retired(m_gpuAnimationPreview.release());
        m_deletionQueue.Retire([retired = std::move(retired)] {});
    }
    return true;
}

void InxVkCoreModular::PrepareMaterialRenderState(const std::shared_ptr<InxMaterial> &material)
{
    if (!material)
        return;
    if (!material->IsPipelineDirty() && m_materialPipelineManager.GetRenderData(material->GetMaterialKey()))
        return;
    const auto stages = material->GetShaderStages();
    const std::string rejectionKey = material->GetMaterialKey() + "|" + stages.ToString() + "|Mesh";
    if (m_rejectedGeometryMaterialPrograms.count(rejectionKey))
        return;
    const ShaderProgramArtifact *artifact;
    try {
        // Queue classification is part of geometry publication. A rejected
        // shader must not change blending, depth or routing on the retained
        // valid pipeline before ResolveMeshMaterial reports the bad selection.
        artifact = ResolveShaderProgramArtifact(material, stages, ShaderProgramDomain::Mesh);
    } catch (const ShaderProgramDomainMismatch &) {
        return;
    }
    const std::string &fragment = artifact ? stages.fragmentKey : material->GetFragShaderName();
    if (const auto *meta = m_shaderCache.GetRenderMeta(fragment)) {
        material->ApplyShaderRenderMeta(meta->cullMode, meta->depthWrite, meta->depthTest, meta->blend, meta->queue,
                                        meta->passTag, meta->stencil, meta->alphaClip);
    }
}

bool InxVkCoreModular::RefreshMaterialPipeline(std::shared_ptr<InxMaterial> material)
{
    return RefreshPreviewMaterialPipeline(std::move(material));
}

void InxVkCoreModular::ReleaseGpuPreviews()
{
    m_resourceManager.DrainAsyncGraphicsSubmissions();
    m_gpuAnimationPreview.reset();
    m_gpuMeshPreview.reset();
    m_gpuMaterialPreview.reset();
}

MaterialRenderData *InxVkCoreModular::ResolveMeshMaterial(const std::shared_ptr<InxMaterial> &material)
{
    if (!material)
        return nullptr;
    const std::string &materialKey = material->GetMaterialKey();
    auto *forward = m_materialPipelineManager.GetRenderData(materialKey);
    if (forward && forward->descriptorSet != VK_NULL_HANDLE &&
        !m_materialPipelineManager.IsDescriptorSetLive(forward->descriptorSet)) {
        m_materialPipelineManager.RemoveRenderData(materialKey);
        forward = nullptr;
    }
    if (forward && !material->IsPipelineDirty()) {
        if ((!material->GetBuffers().empty() || material->IsPropertiesDirty()) &&
            !m_materialPipelineManager.RefreshMaterialBindings(*material))
            return nullptr;
        if (!forward->isValid || forward->descriptorSet == VK_NULL_HANDLE)
            return nullptr;
        UpdateMaterialUBO(*material);
        return forward;
    }

    const auto requestedStages = material->GetShaderStages();
    const std::string rejectionKey = materialKey + "|" + requestedStages.ToString() + "|Mesh";
    const auto *requestedArtifact = m_shaderCache.FindProgramArtifact(requestedStages);
    bool domainRejected =
        m_rejectedGeometryMaterialPrograms.find(rejectionKey) != m_rejectedGeometryMaterialPrograms.end();
    if (!domainRejected && !requestedArtifact && (!forward || material->IsPipelineDirty()) &&
        m_shaderProgramArtifactResolver) {
        try {
            m_shaderProgramArtifactResolver(material, ShaderProgramDomain::Mesh);
            requestedArtifact = m_shaderCache.FindProgramArtifact(requestedStages);
        } catch (const ShaderProgramDomainMismatch &) {
            domainRejected = true;
        }
        forward = m_materialPipelineManager.GetRenderData(materialKey);
    }
    if (domainRejected || (requestedArtifact && requestedArtifact->domain != ShaderProgramDomain::Mesh)) {
        if (m_rejectedGeometryMaterialPrograms.insert(rejectionKey).second)
            RefreshMaterialPipeline(material);
        // A rejected edit cannot replace an already committed geometry ABI.
        material->ClearPipelineDirty();
    } else if (!forward || material->IsPipelineDirty()) {
        if (requestedStages.fragmentKey.empty() || !RefreshMaterialPipeline(material))
            return nullptr;
        forward = m_materialPipelineManager.GetRenderData(materialKey);
    }
    if (!forward || !forward->isValid || forward->descriptorSet == VK_NULL_HANDLE ||
        !m_materialPipelineManager.IsDescriptorSetLive(forward->descriptorSet))
        return nullptr;
    UpdateMaterialUBO(*material);
    return forward;
}

bool InxVkCoreModular::RefreshPreviewMaterialPipeline(std::shared_ptr<InxMaterial> material, bool reportDomainMismatch)
{
    if (!material) {
        return false;
    }
    PrepareMaterialTextureAssets(material);

    const auto stages = material->GetShaderStages();
    const ShaderProgramArtifact *artifact = nullptr;
    try {
        // Direct Fullscreen stages have no linked artifact. Resolve the declared
        // domain before constructing a geometry layout, including on first use.
        // A hot edit can also change domain while an older Mesh artifact exists.
        if (m_shaderCache.FindProgramArtifact(stages) && m_shaderProgramArtifactResolver)
            m_shaderProgramArtifactResolver(material, ShaderProgramDomain::Mesh);
        artifact = ResolveShaderProgramArtifact(material, stages, ShaderProgramDomain::Mesh);
    } catch (const ShaderProgramDomainMismatch &error) {
        if (reportDomainMismatch)
            m_rejectedGeometryMaterialPrograms.insert(material->GetMaterialKey() + "|" + stages.ToString() + "|Mesh");
        MaterialRenderData *previous = m_materialPipelineManager.GetRenderData(material->GetMaterialKey());
        const bool hasLastKnownGood = previous && previous->isValid && previous->pipeline != VK_NULL_HANDLE &&
                                      previous->pipelineLayout != VK_NULL_HANDLE &&
                                      previous->descriptorSet != VK_NULL_HANDLE &&
                                      m_materialPipelineManager.IsDescriptorSetLive(previous->descriptorSet);

        if (reportDomainMismatch) {
            static std::unordered_set<std::string> reportedDomainMismatches;
            const std::string materialName =
                material->GetName().empty() ? material->GetMaterialKey() : material->GetName();
            const std::string failureKey = material->GetMaterialKey() + "|" + stages.ToString() + "|Mesh";
            if (reportedDomainMismatches.insert(failureKey).second) {
                INXLOG_ERROR("Material shader domain mismatch: material '", materialName, "' uses domain '",
                             ShaderProgramDomainName(error.actual),
                             "', but MeshRenderer geometry requires domain 'Mesh'. Use a Mesh-domain vertex/fragment "
                             "shader pair. ",
                             hasLastKnownGood ? "The rejected rebuild did not replace the previous valid GPU pipeline."
                                              : "This material cannot be rendered by MeshRenderer.");
            }
        }

        // Shader selection dirties the material before its domain is known. A
        // previous geometry generation remains valid and is intentionally kept
        // active until the user selects another shader pair.
        if (hasLastKnownGood) {
            material->ClearPipelineDirty();
            return true;
        }
        return false;
    }

    // Material documents intentionally store only authored values. Resolve
    // omitted values from the linked shader contract before any descriptor or
    // preview pipeline observes the material, so passive Project thumbnails,
    // Inspector previews and scene rendering all start from the same snapshot.
    if (artifact)
        material->SynchronizeShaderPropertyDefaults(*artifact);

    // Apply shader render-state annotations to the material before pipeline creation.
    // Fragment shader annotations take priority (they define the surface behaviour).
    const auto *renderMeta = m_shaderCache.GetRenderMeta(artifact ? stages.fragmentKey : material->GetFragShaderName());
    if (renderMeta) {
        material->ApplyShaderRenderMeta(renderMeta->cullMode, renderMeta->depthWrite, renderMeta->depthTest,
                                        renderMeta->blend, renderMeta->queue, renderMeta->passTag, renderMeta->stencil,
                                        renderMeta->alphaClip);
    }

    const auto *forward = artifact ? artifact->FindVariant(ShaderCompileTarget::Forward) : nullptr;
    const auto *vertCode = forward ? &forward->vertexSpirv : m_shaderCache.FindVertCode(material->GetVertShaderName());
    const auto *fragCode =
        forward ? &forward->fragmentSpirv : m_shaderCache.FindFragCode(material->GetFragShaderName());
    const ShaderProgramKey programKey = artifact ? artifact->key : ShaderProgramKey{stages, 0};

    if (vertCode && fragCode && m_materialPipelineManagerInitialized) {
        auto *renderData = m_materialPipelineManager.GetOrCreateRenderDataWithReflection(
            material, *vertCode, *fragCode, programKey,
            {artifact ? artifact->domain : ShaderProgramDomain::Mesh, forward ? forward->vertexEntryPoint : "main",
             forward ? forward->fragmentEntryPoint : "main", forward ? forward->vertexTarget.get() : nullptr,
             forward ? forward->fragmentTarget.get() : nullptr, artifact ? &artifact->properties : nullptr});

        bool forwardOk = renderData && renderData->isValid;
        if (forwardOk)
            m_rejectedGeometryMaterialPrograms.erase(material->GetMaterialKey() + "|" + stages.ToString() + "|Mesh");

        if (forwardOk && artifact) {
            // Optional programs are materialized by their first real pass.
            for (const auto target :
                 {ShaderCompileTarget::ForwardPlus, ShaderCompileTarget::GBuffer, ShaderCompileTarget::Shadow,
                  ShaderCompileTarget::Depth, ShaderCompileTarget::Picking, ShaderCompileTarget::Motion,
                  ShaderCompileTarget::Normal, ShaderCompileTarget::BaseColor})
                material->SetPassShaderProgram(target, nullptr);
        }

        if (forwardOk && m_shadowPipelineReady) {
            // Shadow resources are created lazily by DrawShadowCasters. Eagerly
            // allocating here makes every transient/runtime material consume a
            // descriptor even when it never reaches a shadow pass, and repeated
            // 2D animation material refreshes should not touch shadow resources.
            // The next real shadow draw resolves the shared pipeline and binding cache.
            material->SetPassPipeline(ShaderCompileTarget::Shadow, VK_NULL_HANDLE);
            material->SetPassDescriptorSet(ShaderCompileTarget::Shadow, VK_NULL_HANDLE);
        }

        return forwardOk;
    }

    return false;
}

MaterialPassRenderData *InxVkCoreModular::GetOrCreatePreviewMaterialPass(std::shared_ptr<InxMaterial> material)
{
    if (!material)
        return nullptr;

    auto &pipelineManager = GetMaterialPipelineManager();
    auto *forward = pipelineManager.GetRenderData(material->GetMaterialKey());
    if (!forward || !forward->isValid || !forward->shaderProgram)
        return nullptr;

    auto descriptor = pipelineManager.GetDefaultPassPipelineDescriptor(ShaderCompileTarget::Forward);
    return pipelineManager.GetOrCreatePassRenderData(std::move(material), forward->shaderProgram, descriptor);
}

// ============================================================================
// Lighting System
// ============================================================================

void InxVkCoreModular::SetAmbientColor(const glm::vec3 &color, float intensity)
{
    m_lightCollector.SetAmbientColor(color, intensity);
}

void InxVkCoreModular::UpdateLightingState()
{
    // Environment lighting follows the active scene's settings (Unity-style
    // Lighting > Environment): ambient is derived from the skybox material,
    // an explicit gradient, or a flat color.
    Scene *activeScene = SceneManager::Instance().GetActiveScene();
    const SceneEnvironmentSettings env = activeScene ? activeScene->GetEnvironment() : SceneEnvironmentSettings{};
    using AmbientSource = SceneEnvironmentSettings::AmbientSource;

    // The builtin procedural sky has no backing asset — its parameters are
    // scene data. Push them onto the shared builtin material so the skybox
    // draw and the ambient sync below both see the scene's values (only
    // writing on change to avoid dirtying the material UBO every frame).
    if (env.skyboxMaterialGuid.empty()) {
        if (auto builtinSky = AssetRegistry::Instance().GetBuiltinMaterial("SkyboxProcedural")) {
            const auto syncColor = [&](const char *name, const glm::vec3 &value) {
                const glm::vec4 desired(value, 1.0f);
                const auto *prop = builtinSky->GetProperty(name);
                if (!prop || !std::holds_alternative<glm::vec4>(prop->value) ||
                    std::get<glm::vec4>(prop->value) != desired)
                    builtinSky->SetColor(name, desired);
            };
            syncColor("skyTopColor", env.skyTopColor);
            syncColor("skyHorizonColor", env.skyHorizonColor);
            syncColor("groundColor", env.skyGroundColor);
            const auto *exposureProp = builtinSky->GetProperty("exposure");
            if (!exposureProp || !std::holds_alternative<float>(exposureProp->value) ||
                std::get<float>(exposureProp->value) != env.skyExposure)
                builtinSky->SetFloat("exposure", env.skyExposure);
        }
    }

    switch (static_cast<AmbientSource>(env.ambientSource)) {
    case AmbientSource::Gradient:
        m_lightCollector.SetAmbientGradient(env.ambientSkyColor, env.ambientEquatorColor, env.ambientGroundColor,
                                            env.ambientIntensity);
        break;
    case AmbientSource::Color:
        m_lightCollector.SetAmbientColor(env.ambientColor, env.ambientIntensity);
        break;
    case AmbientSource::Skybox:
    default: {
        std::shared_ptr<InxMaterial> skyMat = activeScene
                                                  ? activeScene->ResolveSkyboxMaterial()
                                                  : AssetRegistry::Instance().GetBuiltinMaterial("SkyboxProcedural");
        bool applied = false;
        if (skyMat) {
            const auto *skyTopProp = skyMat->GetProperty("skyTopColor");
            const auto *horizonProp = skyMat->GetProperty("skyHorizonColor");
            const auto *groundProp = skyMat->GetProperty("groundColor");
            const auto *exposureProp = skyMat->GetProperty("exposure");
            if (skyTopProp && groundProp) {
                const glm::vec3 skyTop = glm::vec3(std::get<glm::vec4>(skyTopProp->value));
                const glm::vec3 ground = glm::vec3(std::get<glm::vec4>(groundProp->value));
                const glm::vec3 equator =
                    horizonProp ? glm::vec3(std::get<glm::vec4>(horizonProp->value)) : glm::mix(ground, skyTop, 0.5f);
                float exposure = 1.0f;
                if (exposureProp && std::holds_alternative<float>(exposureProp->value))
                    exposure = std::get<float>(exposureProp->value);
                m_lightCollector.SetAmbientGradient(skyTop, equator, ground, exposure * env.ambientIntensity);
                applied = true;
            }
        }
        // Custom sky materials without the procedural color properties fall
        // back to a neutral gradient scaled by the ambient intensity.
        if (!applied)
            m_lightCollector.SetAmbientGradient(glm::vec3(0.5f), glm::vec3(0.4f), glm::vec3(0.25f),
                                                env.ambientIntensity);
        break;
    }
    }

    // Build the shader-compatible UBO from collected lights
    m_lightCollector.BuildShaderLightingUBO();
    const uint32_t frameIndex = m_currentFrame % m_maxFramesInFlight;
    if (!m_canonicalLightGpuBuffer.Update(frameIndex, m_lightCollector.GetCanonicalLightSnapshot()))
        INXLOG_ERROR("Failed to upload canonical light snapshot for frame slot ", frameIndex);
}

// ============================================================================
// Buffer / Shader Accessors (for OutlineRenderer)
// ============================================================================

bool InxVkCoreModular::BindObjectGeometry(VkCommandBuffer commands, uint64_t objectId) const
{
    const auto it = m_perObjectBuffers.find(objectId);
    if (it == m_perObjectBuffers.end() || !it->second.HasVertexBuffer() || !it->second.indexBuffer)
        return false;
    const auto &geometry = it->second;
    if (auto *reads = vk::SubmissionReadScope::Current(commands)) {
        if (!geometry.residentVertexHandle)
            reads->Add(geometry.vertexBuffer->GetUploadSubmission(), VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
        reads->Add(geometry.indexBuffer->GetUploadSubmission(), VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
    }
    const VkBuffer vertices = geometry.GetVertexHandle();
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(commands, 0, 1, &vertices, &offset);
    vkCmdBindIndexBuffer(commands, geometry.indexBuffer->GetBuffer(), 0,
                         geometry.indexFormat == MeshIndexFormat::UInt16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
    return true;
}

void InxVkCoreModular::PrepareMaterialTextureAssets(const std::shared_ptr<InxMaterial> &material)
{
    if (!material)
        return;
    const auto graph = AssetDependencyGraph::Current();
    material->BindDependencyGraph(graph);
    if (!material->NeedsTextureAssetResolution())
        return;
    auto *database = AssetRegistry::Instance().GetAssetDatabase();
    std::unordered_map<std::string, std::shared_ptr<rhi::RenderTexture>> textures;
    std::unordered_set<std::string> dependencies;
    for (const auto &[name, property] : material->GetAllProperties()) {
        if (property.type != MaterialPropertyType::Texture2D || material->HasRuntimeTextureOverride(name))
            continue;
        const auto &guid = std::get<std::string>(property.value);
        if (guid.empty() || guid == "white" || guid == "black" || guid == "normal")
            continue;
        dependencies.insert(guid); // Missing references still receive restore notifications.
        const auto metadata = database ? database->GetMetaByGuid(guid) : nullptr;
        if (metadata && metadata->GetResourceType() == ResourceType::RenderTexture)
            textures[name] = m_renderTextureAssetLoader(guid);
    }
    // All target allocations succeed before changing this material's bindings.
    material->PublishTextureAssets(std::move(textures));
    const auto owner = material->GetTextureDependencyOwner();
    graph->ClearRuntimeDependenciesOf(owner);
    if (dependencies.empty()) {
        m_materialTextureOwners.erase(owner);
    } else {
        m_materialTextureOwners[owner] = material;
        for (const auto &guid : dependencies)
            graph->AddRuntimeDependency(owner, guid);
    }
}

bool InxVkCoreModular::InvalidateMaterialTextureAssets(const std::string &owner, const std::string &guid, bool deleted)
{
    const auto found = m_materialTextureOwners.find(owner);
    if (found == m_materialTextureOwners.end())
        return false;
    if (auto material = found->second.lock())
        material->InvalidateTextureAssets(guid, deleted);
    else {
        AssetDependencyGraph::Instance().ClearRuntimeDependenciesOf(owner);
        m_materialTextureOwners.erase(found);
    }
    return true;
}

void InxVkCoreModular::CollectUnusedMaterialTextureOwners()
{
    for (auto owner = m_materialTextureOwners.begin(); owner != m_materialTextureOwners.end();) {
        if (owner->second.expired()) {
            AssetDependencyGraph::Instance().ClearRuntimeDependenciesOf(owner->first);
            owner = m_materialTextureOwners.erase(owner);
        } else
            ++owner;
    }
}

VkBuffer InxVkCoreModular::GetFallbackMaterialUbo() const
{
    return m_materialUbo ? m_materialUbo->GetBuffer() : VK_NULL_HANDLE;
}

VkBuffer InxVkCoreModular::GetInstanceSSBO(size_t index) const
{
    if (index < m_instanceBuffers.size() && m_instanceBuffers[index].buffer)
        return m_instanceBuffers[index].buffer->GetBuffer();
    return VK_NULL_HANDLE;
}

VkShaderModule InxVkCoreModular::GetShaderModule(const std::string &name, const std::string &type)
{
    if (!EnsureShaderAvailable(name, type))
        return VK_NULL_HANDLE;
    return m_shaderCache.GetModule(name, type);
}

// ============================================================================
// Shared shadow material bindings and pipeline creation
// ============================================================================

InxVkCoreModular::ShadowDescriptorAllocation InxVkCoreModular::AllocateShadowMaterialDescriptorSet()
{
    if (m_shadowMaterialDescSetLayout == VK_NULL_HANDLE)
        return {};
    ShadowDescriptorAllocation allocation{};
    allocation.descriptorLease = m_backend.Device().GetRhiDevice().GetDescriptorManager().Allocate(
        m_shadowMaterialDescSetLayout, vk::DescriptorArena::Persistent);
    allocation.descriptorSet = allocation.descriptorLease.set;
    return allocation;
}

bool InxVkCoreModular::EnsureShadowMaterialDummyDescriptorSet()
{
    if (m_shadowMaterialDummyDescSet != VK_NULL_HANDLE)
        return true;
    if (m_shadowMaterialDescSetLayout == VK_NULL_HANDLE)
        return false;
    auto defaultSlot = m_textureCache.Find("white", m_ensureFrameCounter);
    auto defaultTex = defaultSlot ? defaultSlot->Acquire() : nullptr;
    if (!defaultTex)
        return false;
    auto &rhiDevice = m_backend.Device().GetRhiDevice();
    const VkImageView defaultView = rhiDevice.Resolve(defaultTex->GetView());
    const VkSampler defaultSampler = rhiDevice.Resolve(defaultTex->GetSampler());
    if (defaultView == VK_NULL_HANDLE || defaultSampler == VK_NULL_HANDLE)
        return false;
    if (!m_materialUbo)
        return false;
    const ShadowDescriptorAllocation allocation = AllocateShadowMaterialDescriptorSet();
    if (allocation.descriptorSet == VK_NULL_HANDLE)
        return false;
    const VkDescriptorSet set = allocation.descriptorSet;
    std::vector<VkDescriptorImageInfo> imageInfos(kMaxShadowMaterialTextures);
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(kMaxShadowMaterialTextures + 2);
    for (uint32_t i = 0; i < kMaxShadowMaterialTextures; ++i) {
        imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfos[i].imageView = defaultView;
        imageInfos[i].sampler = defaultSampler;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = set;
        w.dstBinding = i;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &imageInfos[i];
        writes.push_back(w);
    }
    VkDescriptorBufferInfo fragBi{};
    fragBi.buffer = m_materialUbo->GetBuffer();
    fragBi.offset = 0;
    fragBi.range = 16;
    VkWriteDescriptorSet wf{};
    wf.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wf.dstSet = set;
    wf.dstBinding = kMaxShadowMaterialTextures;
    wf.descriptorCount = 1;
    wf.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    wf.pBufferInfo = &fragBi;
    VkDescriptorBufferInfo vtxBi{};
    vtxBi.buffer = m_materialUbo->GetBuffer();
    vtxBi.offset = 0;
    vtxBi.range = 16;
    VkWriteDescriptorSet wv{};
    wv.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wv.dstSet = set;
    wv.dstBinding = 14;
    wv.descriptorCount = 1;
    wv.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    wv.pBufferInfo = &vtxBi;
    writes.push_back(wf);
    writes.push_back(wv);
    vkUpdateDescriptorSets(GetDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    m_shadowMaterialDummyDescSet = set;
    m_shadowMaterialDummyLease = allocation.descriptorLease;
    return true;
}

void InxVkCoreModular::RetireShadowMaterialBinding(ShadowMaterialBindingEntry entry)
{
    if (entry.descriptorSet == VK_NULL_HANDLE || entry.descriptorSet == m_shadowMaterialDummyDescSet ||
        !entry.descriptorLease.IsValid()) {
        return;
    }

    m_backend.Device().GetRhiDevice().GetDescriptorManager().Retire(entry.descriptorLease);
    ++m_shadowMaterialBindingRetirements;
    m_deletionQueue.Retire([keepAlive = std::move(entry.textureKeepAlive)]() mutable { keepAlive.clear(); });
}

void InxVkCoreModular::CollectUnusedShadowMaterialBindings()
{
    for (auto it = m_shadowMaterialBindingCache.begin(); it != m_shadowMaterialBindingCache.end();) {
        if (!it->second.owner.expired()) {
            ++it;
            continue;
        }
        ShadowMaterialBindingEntry retired = std::move(it->second);
        it = m_shadowMaterialBindingCache.erase(it);
        RetireShadowMaterialBinding(std::move(retired));
    }
}

VkDescriptorSet InxVkCoreModular::EnsureShadowMaterialBinding(const std::shared_ptr<InxMaterial> &material,
                                                              const MaterialDescriptorSet *forwardMaterialDesc,
                                                              const ShaderProgram *forwardProgram,
                                                              const ShaderProgram *shadowProgram)
{
    if (!material)
        return VK_NULL_HANDLE;

    const bool hasVertexMaterialUBO = shadowProgram ? shadowProgram->HasVertexMaterialUBO()
                                                    : (forwardProgram && forwardProgram->HasVertexMaterialUBO());
    const bool hasVertexMaterialTextures =
        shadowProgram &&
        std::any_of(shadowProgram->GetDescriptorBindings().begin(), shadowProgram->GetDescriptorBindings().end(),
                    [](const MergedDescriptorBinding &binding) {
                        return binding.set == 2 && binding.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
                               (binding.stageFlags & VK_SHADER_STAGE_VERTEX_BIT) != 0;
                    });
    // Even an opaque material must bind the reflected threshold uniform:
    // the same compiled surface can enable alpha clipping at runtime.
    const bool hasFragmentMaterialUBO = shadowProgram && shadowProgram->HasMaterialUBO();
    const bool usesBindlessTextures = shadowProgram && shadowProgram->UsesBindlessTextureABI();
    const bool needsDescriptor =
        hasVertexMaterialUBO || hasVertexMaterialTextures || hasFragmentMaterialUBO || usesBindlessTextures;
    const InxMaterial *identity = material.get();

    const auto markBindlessShadowResourcesUsed = [&]() {
        if (usesBindlessTextures) {
            m_backend.Device().GetRhiDevice().MarkBindlessTexturesUsed(
                forwardMaterialDesc->bindlessTextureIndices.data(), forwardMaterialDesc->bindlessTextureIndices.size());
        }
    };
    auto existing = m_shadowMaterialBindingCache.find(identity);
    // Forward resource replacements publish a new, never-reused descriptor lease.
    // Numeric edits and texture residency changes use this same publication path.
    if (existing != m_shadowMaterialBindingCache.end() && forwardMaterialDesc &&
        existing->second.forwardDescriptorId == forwardMaterialDesc->descriptorLease.id &&
        existing->second.program.get() == shadowProgram) {
        ++m_shadowMaterialBindingCacheHits;
        markBindlessShadowResourcesUsed();
        return existing->second.descriptorSet;
    }

    if (!needsDescriptor) {
        if (existing != m_shadowMaterialBindingCache.end()) {
            ShadowMaterialBindingEntry retired = std::move(existing->second);
            m_shadowMaterialBindingCache.erase(existing);
            RetireShadowMaterialBinding(std::move(retired));
        }
        return EnsureShadowMaterialDummyDescriptorSet() ? m_shadowMaterialDummyDescSet : VK_NULL_HANDLE;
    }

    if (!forwardMaterialDesc || !forwardMaterialDesc->isValid) {
        INXLOG_WARN("EnsureShadowMaterialBinding: forward material resources are unavailable for '",
                    material->GetName(), "'");
        return VK_NULL_HANDLE;
    }
    // Forward merges an identical vertex/fragment MaterialProperties layout
    // into its canonical material UBO. Shadow remaps those stages to separate
    // bindings, but both bindings still consume that same authored buffer.
    const MaterialUBO *vertexUbo = hasVertexMaterialUBO ? (forwardProgram && forwardProgram->HasVertexMaterialUBO()
                                                               ? forwardMaterialDesc->vertexMaterialUBO.get()
                                                               : forwardMaterialDesc->materialUBO.get())
                                                        : nullptr;
    if (hasVertexMaterialUBO && (!vertexUbo || !vertexUbo->IsValid())) {
        INXLOG_WARN("EnsureShadowMaterialBinding: missing vertex material UBO for '", material->GetName(), "'");
        return VK_NULL_HANDLE;
    }
    if (hasFragmentMaterialUBO && (!forwardMaterialDesc->materialUBO || !forwardMaterialDesc->materialUBO->IsValid())) {
        INXLOG_WARN("EnsureShadowMaterialBinding: missing alpha-clip material UBO for '", material->GetName(), "'");
        return VK_NULL_HANDLE;
    }
    if (usesBindlessTextures &&
        (!forwardMaterialDesc->usesBindlessTextureABI || !forwardMaterialDesc->textureIndexUBO ||
         !forwardMaterialDesc->textureIndexUBO->IsValid())) {
        INXLOG_WARN("EnsureShadowMaterialBinding: missing bindless texture indices for '", material->GetName(), "'");
        return VK_NULL_HANDLE;
    }

    auto defaultSlot = m_textureCache.Find("white", m_ensureFrameCounter);
    auto defaultTex = defaultSlot ? defaultSlot->Acquire() : nullptr;
    if (!defaultTex || !m_materialUbo) {
        return VK_NULL_HANDLE;
    }
    auto &rhiDevice = m_backend.Device().GetRhiDevice();
    const VkImageView defaultView = rhiDevice.Resolve(defaultTex->GetView());
    const VkSampler defaultSampler = rhiDevice.Resolve(defaultTex->GetSampler());
    if (defaultView == VK_NULL_HANDLE || defaultSampler == VK_NULL_HANDLE)
        return VK_NULL_HANDLE;

    std::vector<std::pair<uint32_t, MaterialDescriptorSet::TextureBinding>> sortedTextures;
    if (hasFragmentMaterialUBO || hasVertexMaterialTextures) {
        sortedTextures.assign(forwardMaterialDesc->textureBindings.begin(), forwardMaterialDesc->textureBindings.end());
        std::sort(sortedTextures.begin(), sortedTextures.end(),
                  [](const auto &left, const auto &right) { return left.first < right.first; });
        if (sortedTextures.size() > kMaxShadowMaterialTextures) {
            INXLOG_ERROR("EnsureShadowMaterialBinding: material '", material->GetName(), "' requires ",
                         sortedTextures.size(), " shadow texture slots, maximum is ", kMaxShadowMaterialTextures);
            return VK_NULL_HANDLE;
        }
    }

    ++m_shadowMaterialBindingCacheMisses;
    const ShadowDescriptorAllocation allocation = AllocateShadowMaterialDescriptorSet();
    if (allocation.descriptorSet == VK_NULL_HANDLE) {
        INXLOG_WARN("EnsureShadowMaterialBinding: descriptor allocation failed for '", material->GetName(), "'");
        return VK_NULL_HANDLE;
    }

    std::vector<VkDescriptorImageInfo> imageInfos(kMaxShadowMaterialTextures);
    for (auto &imageInfo : imageInfos) {
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = defaultView;
        imageInfo.sampler = defaultSampler;
    }
    for (size_t index = 0; index < sortedTextures.size(); ++index) {
        const auto &texture = sortedTextures[index].second;
        if (texture.imageView != VK_NULL_HANDLE && texture.sampler != VK_NULL_HANDLE) {
            imageInfos[index].imageView = texture.imageView;
            imageInfos[index].sampler = texture.sampler;
        }
    }

    const MaterialUBO *fragmentUbo = hasFragmentMaterialUBO ? forwardMaterialDesc->materialUBO.get() : nullptr;
    VkDescriptorBufferInfo fragmentBuffer{};
    fragmentBuffer.buffer =
        fragmentUbo ? fragmentUbo->GetBuffer() : (vertexUbo ? vertexUbo->GetBuffer() : m_materialUbo->GetBuffer());
    fragmentBuffer.offset = 0;
    fragmentBuffer.range = fragmentUbo ? fragmentUbo->GetSize() : (vertexUbo ? vertexUbo->GetSize() : 16);
    VkDescriptorBufferInfo vertexBuffer{};
    vertexBuffer.buffer = vertexUbo ? vertexUbo->GetBuffer() : fragmentBuffer.buffer;
    vertexBuffer.offset = 0;
    vertexBuffer.range = vertexUbo ? vertexUbo->GetSize() : fragmentBuffer.range;

    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(kMaxShadowMaterialTextures + 3);
    for (uint32_t index = 0; index < kMaxShadowMaterialTextures; ++index) {
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = allocation.descriptorSet;
        write.dstBinding = index;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &imageInfos[index];
        writes.push_back(write);
    }
    VkWriteDescriptorSet fragmentWrite{};
    fragmentWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    fragmentWrite.dstSet = allocation.descriptorSet;
    fragmentWrite.dstBinding = kMaxShadowMaterialTextures;
    fragmentWrite.descriptorCount = 1;
    fragmentWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    fragmentWrite.pBufferInfo = &fragmentBuffer;
    writes.push_back(fragmentWrite);
    VkWriteDescriptorSet vertexWrite{};
    vertexWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vertexWrite.dstSet = allocation.descriptorSet;
    vertexWrite.dstBinding = 14;
    vertexWrite.descriptorCount = 1;
    vertexWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vertexWrite.pBufferInfo = &vertexBuffer;
    writes.push_back(vertexWrite);
    VkDescriptorBufferInfo textureIndexBuffer{};
    VkWriteDescriptorSet textureIndexWrite{};
    if (usesBindlessTextures) {
        textureIndexBuffer.buffer = forwardMaterialDesc->textureIndexUBO->GetBuffer();
        textureIndexBuffer.offset = 0;
        textureIndexBuffer.range = forwardMaterialDesc->textureIndexUBO->GetSize();
        textureIndexWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        textureIndexWrite.dstSet = allocation.descriptorSet;
        textureIndexWrite.dstBinding = ShaderProgram::MaterialTextureIndexBinding;
        textureIndexWrite.descriptorCount = 1;
        textureIndexWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        textureIndexWrite.pBufferInfo = &textureIndexBuffer;
        writes.push_back(textureIndexWrite);
    }
    vkUpdateDescriptorSets(GetDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    auto uploadSources = forwardMaterialDesc->GetTextureUploadSources();
    uploadSources.Add(defaultTex->GetUploadSubmission());
    rhiDevice.GetDescriptorManager().SetUploadSources(allocation.descriptorLease, uploadSources);
    markBindlessShadowResourcesUsed();

    ShadowMaterialBindingEntry replacement{};
    replacement.owner = material;
    replacement.program = material->GetPassShaderProgramPublication(ShaderCompileTarget::Shadow);
    replacement.forwardDescriptorId = forwardMaterialDesc->descriptorLease.id;
    replacement.descriptorSet = allocation.descriptorSet;
    replacement.descriptorLease = allocation.descriptorLease;
    replacement.textureKeepAlive.push_back(defaultTex);
    for (const auto &[binding, texture] : sortedTextures) {
        (void)binding;
        if (texture.gpuView)
            replacement.textureKeepAlive.push_back(texture.gpuView);
    }

    if (existing != m_shadowMaterialBindingCache.end()) {
        ShadowMaterialBindingEntry retired = std::move(existing->second);
        existing->second = std::move(replacement);
        RetireShadowMaterialBinding(std::move(retired));
    } else {
        m_shadowMaterialBindingCache.emplace(identity, std::move(replacement));
    }
    return allocation.descriptorSet;
}

VkDescriptorSet InxVkCoreModular::EnsureMaterialShadowPipeline(const std::shared_ptr<InxMaterial> &material,
                                                               VkFormat depthFormat)
{
    // Shared shadow resources must be ready
    if (m_shadowPipelineLayout == VK_NULL_HANDLE || depthFormat == VK_FORMAT_UNDEFINED)
        return VK_NULL_HANDLE;

    if (!material)
        return VK_NULL_HANDLE;

    VkDevice device = GetDevice();

    MaterialRenderData *forwardRenderData = ResolveMeshMaterial(material);
    if (!forwardRenderData)
        return VK_NULL_HANDLE;
    const ShaderStagePair &stagePair = forwardRenderData->programKey.stages;
    if (!material->GetPassShaderProgram(ShaderCompileTarget::Shadow)) {
        const auto *artifact = m_shaderCache.FindProgramArtifact(stagePair);
        if (!artifact || !artifact->FindVariant(ShaderCompileTarget::Shadow))
            return VK_NULL_HANDLE;
        auto publication = m_shaderCache.MaterializeProgramVariant(stagePair, ShaderCompileTarget::Shadow);
        if (!publication) {
            INXLOG_ERROR("Failed to materialize Shadow variant for '", stagePair.ToString(), "'");
            return VK_NULL_HANDLE;
        }
        // Shader and stage-pair invalidation clear this publication on the material.
        material->SetPassShaderProgram(ShaderCompileTarget::Shadow, std::move(publication));
    }
    MaterialDescriptorSet *forwardMaterialDesc = forwardRenderData->materialDescSet;
    const ShaderProgram *forwardProgram = forwardRenderData->shaderProgram.get();
    const auto &linkedShadowPublication = material->GetPassShaderProgramPublication(ShaderCompileTarget::Shadow);
    const ShaderProgram *linkedShadowProgram = linkedShadowPublication.get();
    material->SetPassDescriptorSet(ShaderCompileTarget::Shadow, VK_NULL_HANDLE);

    const VkShaderModule vertModule = linkedShadowProgram->GetVertexModule();
    const VkShaderModule fragModule = linkedShadowProgram->GetFragmentModule();
    if (vertModule == VK_NULL_HANDLE || fragModule == VK_NULL_HANDLE) {
        INXLOG_ERROR("EnsureMaterialShadowPipeline: linked Shadow modules are unavailable for material '",
                     material->GetName(), "'");
        return VK_NULL_HANDLE;
    }

    const VkCullModeFlags matCullMode = vk::ToVkCullMode(material->GetRenderState().cullMode);
    const ShadowPipelineKey shadowShaderKey{linkedShadowPublication, matCullMode, depthFormat};
    auto cacheIt = m_shadowPipelineCache.find(shadowShaderKey);
    if (cacheIt != m_shadowPipelineCache.end()) {
        material->SetPassPipeline(ShaderCompileTarget::Shadow, cacheIt->second);
        material->SetPassPipelineLayout(ShaderCompileTarget::Shadow, m_shadowPipelineLayout);
        return EnsureShadowMaterialBinding(material, forwardMaterialDesc, forwardProgram, linkedShadowProgram);
    }

    // Shader stages
    auto shaderStages = vkrender::MakeVertFragStages(vertModule, fragModule);

    // Vertex input — only attributes consumed by the shadow vertex shader (full mesh buffer still bound).
    auto bindingDesc = vk::GetVertexBindingDescription();
    const ShaderReflection &shadowVertRefl = linkedShadowProgram->GetVertexReflection();
    std::vector<VkVertexInputAttributeDescription> attrDescs = FilterVertexAttributesForReflection(shadowVertRefl);

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = attrDescs.empty() ? 0u : 1u;
    vertexInput.pVertexBindingDescriptions = attrDescs.empty() ? nullptr : &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size());
    vertexInput.pVertexAttributeDescriptions = attrDescs.empty() ? nullptr : attrDescs.data();

    auto inputAssembly = vkrender::MakeTriangleListInputAssembly();

    vkrender::DynamicViewportScissorState dynVpScissor;
    const std::array<VkDynamicState, 3> shadowDynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_DEPTH_BIAS,
    };
    dynVpScissor.dynamicState.dynamicStateCount = static_cast<uint32_t>(shadowDynamicStates.size());
    dynVpScissor.dynamicState.pDynamicStates = shadowDynamicStates.data();

    // Directional depth bias is applied dynamically in raster space. Unlike a
    // world-space caster translation this follows polygon slope without
    // detaching contact shadows by a full cascade texel. Local lights retain
    // their perspective-scaled vertex bias and set the dynamic values to zero.
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = static_cast<VkCullModeFlags>(matCullMode);
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_TRUE;
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasSlopeFactor = 0.0f;
    rasterizer.depthBiasClamp = 0.0f;

    auto multisampling = vkrender::MakeMultisampleState();

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 0;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &dynVpScissor.viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlend;
    pipelineInfo.pDynamicState = &dynVpScissor.dynamicState;
    pipelineInfo.layout = m_shadowPipelineLayout;
    std::array<VkFormat, rhi::GraphicsRenderingSignature::MaxColorTargets> dynamicColorFormats{};
    VkPipelineRenderingCreateInfo dynamicRenderingInfo{};
    rhi::GraphicsRenderingSignature signature;
    signature.depthFormat = rhi::FromVkFormat(depthFormat);
    signature.stencilFormat =
        rhi::IsStencilFormat(signature.depthFormat) ? signature.depthFormat : rhi::PixelFormat::Undefined;
    signature.samples = rhi::SampleCount::One;
    if (!rhi::BuildVkPipelineRenderingInfo(signature, dynamicColorFormats, dynamicRenderingInfo)) {
        INXLOG_WARN("Failed to build Dynamic Rendering signature for shadow pipeline");
        return VK_NULL_HANDLE;
    }
    pipelineInfo.pNext = &dynamicRenderingInfo;
    pipelineInfo.renderPass = VK_NULL_HANDLE;
    pipelineInfo.subpass = 0;

    VkPipeline shadowPipeline = VK_NULL_HANDLE;
    VkPipelineCache pipelineCache = m_materialPipelineManager.GetVkPipelineCache();
    if (vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineInfo, nullptr, &shadowPipeline) != VK_SUCCESS) {
        INXLOG_WARN("Failed to create linked shadow pipeline for '", material->GetName(), "' (program='",
                    stagePair.ToString(), "')");
        return VK_NULL_HANDLE;
    }

    m_shadowPipelineCache[shadowShaderKey] = shadowPipeline;
    material->SetPassPipeline(ShaderCompileTarget::Shadow, shadowPipeline);
    material->SetPassPipelineLayout(ShaderCompileTarget::Shadow, m_shadowPipelineLayout);
    return EnsureShadowMaterialBinding(material, forwardMaterialDesc, forwardProgram, linkedShadowProgram);
}

// ============================================================================
// GPU Material Preview
// ============================================================================

std::shared_ptr<PreviewImage> InxVkCoreModular::BeginMaterialPreviewGPU(const std::shared_ptr<InxMaterial> &material,
                                                                        int size, bool &progressed,
                                                                        bool *texturePending)
{
    if (!material || size <= 0 || !m_materialPipelineManagerInitialized)
        return nullptr;

    if (!m_gpuMaterialPreview) {
        m_gpuMaterialPreview = std::make_unique<GPUMaterialPreview>(this);
        progressed = true;
    }

    return m_gpuMaterialPreview->BeginRender(*material, size, progressed, texturePending);
}

bool InxVkCoreModular::PrepareMeshPreviewMaterials(const std::vector<std::shared_ptr<InxMaterial>> &sources,
                                                   std::vector<std::shared_ptr<InxMaterial>> &prepared,
                                                   bool &progressed)
{
    if (!m_materialPipelineManagerInitialized)
        return false;
    if (!m_gpuMeshPreview) {
        m_gpuMeshPreview = std::make_unique<GPUMeshPreview>(this);
        progressed = true;
    }
    return m_gpuMeshPreview->PrepareMaterials(sources, prepared, progressed);
}

std::shared_ptr<PreviewImage>
InxVkCoreModular::BeginMeshPreviewGPU(const InxMesh &mesh, const std::vector<std::shared_ptr<InxMaterial>> &materials,
                                      int size, std::shared_ptr<PreviewGeometryUpload> &geometry, bool &progressed,
                                      bool &materialsChanged)
{
    if (!m_gpuMeshPreview) {
        materialsChanged = true;
        return nullptr;
    }
    return m_gpuMeshPreview->BeginRender(mesh, materials, size, geometry, progressed, materialsChanged);
}

uint64_t InxVkCoreModular::RenderMeshPreviewGPUImGuiCamera(const InxMesh &mesh,
                                                           const std::vector<std::shared_ptr<InxMaterial>> &materials,
                                                           int size, const glm::mat4 &view, const glm::mat4 &proj,
                                                           const glm::vec3 &cameraPos,
                                                           std::shared_ptr<PreviewGeometryUpload> &geometry,
                                                           bool &progressed, bool cloneMaterials)
{
    if (size <= 0 || !m_materialPipelineManagerInitialized)
        return 0;

    if (!m_gpuMeshPreview) {
        m_gpuMeshPreview = std::make_unique<GPUMeshPreview>(this);
        progressed = true;
    }

    return m_gpuMeshPreview->RenderToImGuiTextureCamera(mesh, materials, size, view, proj, cameraPos, geometry,
                                                        progressed, cloneMaterials);
}

uint64_t InxVkCoreModular::GetMeshPreviewDisplayTextureId() const
{
    return m_gpuMeshPreview ? m_gpuMeshPreview->GetDisplayTextureId() : 0;
}

MeshIndexFormat InxVkCoreModular::GetObjectIndexFormat(uint64_t objectId) const
{
    const auto it = m_perObjectBuffers.find(objectId);
    if (it == m_perObjectBuffers.end() || !it->second.indexBuffer)
        throw std::out_of_range("Object has no published GPU index buffer");
    return it->second.indexFormat;
}

uint64_t InxVkCoreModular::GetObjectIndexBufferBytes(uint64_t objectId) const
{
    const auto it = m_perObjectBuffers.find(objectId);
    if (it == m_perObjectBuffers.end() || !it->second.indexBuffer)
        return 0;
    return it->second.indexBuffer->GetSize();
}

uint64_t InxVkCoreModular::RenderModelAnimationPreview(const std::shared_ptr<InxMesh> &mesh, const std::string &take,
                                                       float seconds, int size, uint64_t dependencyRevision)
{
    if (!m_materialPipelineManagerInitialized)
        return 0;
    if (!m_gpuAnimationPreview)
        m_gpuAnimationPreview = std::make_unique<GPUMeshPreview>(this);
    return m_gpuAnimationPreview->RenderAnimation(mesh, take, seconds, size, dependencyRevision);
}

} // namespace infernux
