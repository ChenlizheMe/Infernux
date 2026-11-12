#include <core/math/DepthConvention.h>
/**
 * @file GPUMaterialPreview.cpp
 * @brief GPU material preview — renders a sphere using the material's real
 *        Vulkan pipeline and produces independent GPU textures for editor thumbnails.
 */

#include "GPUMaterialPreview.h"
#include "InxError.h"
#include "PreviewDisplay.h"
#include "PreviewResources.h"
#include <function/renderer/EngineGlobals.h>
#include <function/renderer/InxRenderStruct.h>
#include <function/renderer/InxVkCoreModular.h>
#include <function/renderer/MaterialPipelineManager.h>
#include <function/renderer/RenderInstanceHistory.h>
#include <function/renderer/rhi/RhiComputeBuffer.h>
#include <function/renderer/shader/ShaderProgram.h>
#include <function/renderer/vk/DescriptorBindTrace.h>
#include <function/renderer/vk/RhiVulkanTypes.h>
#include <function/renderer/vk/VkRenderUtils.h>
#include <function/renderer/vk/VkResourceManager.h>
#include <function/renderer/vk/VulkanBackendContext.h>
#include <function/renderer/vk/VulkanComputeQueue.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/scene/Light.h>
#include <function/scene/LightingData.h>
#include <function/scene/PrimitiveMeshes.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <core/log/InxLog.h>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>

namespace infernux
{

namespace
{
constexpr float kPreviewMatteR = 0.0f;
constexpr float kPreviewMatteG = 0.0f;
constexpr float kPreviewMatteB = 0.0f;
constexpr float kPreviewMatteA = 0.0f;
constexpr float kPreviewCameraDistance = 1.82f; // slightly closer so the material sphere fills a bit more
constexpr float kPreviewModelScale = 1.28f;
constexpr int kPreviewSupersampleFactor = 2;

} // namespace

// ============================================================================
// Construction / Destruction
// ============================================================================

GPUMaterialPreview::GPUMaterialPreview(InxVkCoreModular *vkCore) : m_vkCore(vkCore)
{
    m_renderView.id = rhi::AllocateRenderViewId();
    m_renderView.device = vkCore ? vkCore->GetDeviceContext().GetDeviceId() : rhi::InvalidDeviceId;
    m_renderView.kind = rhi::RenderViewKind::Preview;
    m_renderView.output = rhi::RenderOutputKind::OffscreenTexture;
    m_renderView.revision = 1;
    if (vkCore)
        m_display = std::make_unique<PreviewDisplay>(*vkCore, m_renderView.id);
}

GPUMaterialPreview::~GPUMaterialPreview()
{
    if (!m_vkCore)
        return;

    DestroyAttachments();

    m_sphereVBO.reset();
    m_sphereIBO.reset();
    DestroyViewResources();
}

// ============================================================================
// BeginRender — main entry point
// ============================================================================

std::shared_ptr<PreviewImage> GPUMaterialPreview::BeginRender(InxMaterial &material, int size, bool &progressed,
                                                              bool *texturePending)
{
    if (texturePending)
        *texturePending = false;
    if (!m_vkCore)
        throw std::logic_error("material preview requires its initialized Vulkan renderer");
    if (size <= 0)
        throw std::invalid_argument("material preview size must be positive");
    if (m_activeSubmission && !m_activeSubmission->IsComplete())
        return nullptr;
    if (m_activeSubmission && m_activeSubmission->HasExecutionTiming())
        m_gpuNanoseconds = m_activeSubmission->GetExecutionNanoseconds();
    m_activeSubmission.reset();

    const int renderSize = std::max(size, size * kPreviewSupersampleFactor);
    const glm::mat4 previewModel = glm::scale(glm::mat4(1.0f), glm::vec3(kPreviewModelScale));
    const glm::mat4 previewNormal = glm::transpose(glm::inverse(previewModel));

    std::vector<std::shared_ptr<InxMaterial>> ownedPreviewMaterials;
    std::vector<std::shared_ptr<InxMaterial>> previewPassMaterials;

    // Always isolate preview rendering from the caller's material instance.
    // Even opaque previews can otherwise overwrite or invalidate the live
    // material's forward pass handles while the main scene is still drawing.
    auto basePreviewMaterial = material.Clone();
    if (!basePreviewMaterial)
        throw std::runtime_error("failed to clone the preview material");
    std::vector<rhi::SubmissionTicket> producers;
    for (const auto &[name, buffer] : material.GetBuffers()) {
        auto value = buffer->Freeze();
        const auto producer = value->GetLastWriteSubmission();
        if (producer.IsValid())
            producers.push_back(producer);
        basePreviewMaterial->SetBuffer(name, std::move(value));
    }
    basePreviewMaterial->ClearAllPassPipelines();
    if (!PrepareSphereBuffers(progressed))
        return nullptr;
    if (!EnsureResources(renderSize, progressed))
        throw std::runtime_error("failed to allocate material preview resources");
    if (!m_vkCore->RefreshPreviewMaterialPipeline(basePreviewMaterial, false)) {
        return nullptr;
    }
    ownedPreviewMaterials.push_back(basePreviewMaterial);
    previewPassMaterials.push_back(basePreviewMaterial);

    const RenderState &baseState = basePreviewMaterial->GetRenderState();
    if (baseState.blendEnable) {
        auto buildPreviewPass = [&](MaterialCullMode cullMode, bool overrideCull) -> std::shared_ptr<InxMaterial> {
            // Preview passes must not reuse the source material's GUID/name key,
            // otherwise descriptor-cache replacement can invalidate the live set
            // still used by the main scene.
            auto passMaterial = basePreviewMaterial->Clone();
            RenderState state = passMaterial->GetRenderState();
            if (overrideCull)
                state.cullMode = cullMode;

            // Preview path uses transparent clear color (alpha=0), so alpha must
            // be written by the material pass instead of preserving destination alpha.
            state.srcAlphaBlendFactor = MaterialBlendFactor::One;
            state.dstAlphaBlendFactor = MaterialBlendFactor::OneMinusSourceAlpha;
            state.alphaBlendOp = MaterialBlendOp::Add;

            passMaterial->SetRenderState(state);
            passMaterial->MarkOverride(RenderStateOverride::BlendMode);
            if (overrideCull)
                passMaterial->MarkOverride(RenderStateOverride::CullMode);
            passMaterial->ClearAllPassPipelines();

            if (!m_vkCore->RefreshPreviewMaterialPipeline(passMaterial, false)) {
                return nullptr;
            }
            return passMaterial;
        };

        if (!baseState.depthWriteEnable && baseState.cullMode == MaterialCullMode::None) {
            auto backFacePass = buildPreviewPass(MaterialCullMode::Front, true);
            auto frontFacePass = buildPreviewPass(MaterialCullMode::Back, true);
            if (backFacePass && frontFacePass) {
                ownedPreviewMaterials.push_back(backFacePass);
                ownedPreviewMaterials.push_back(frontFacePass);
                previewPassMaterials.clear();
                previewPassMaterials.push_back(backFacePass);
                previewPassMaterials.push_back(frontFacePass);
            } else {
                return nullptr;
            }
        } else {
            auto singlePass = buildPreviewPass(baseState.cullMode, false);
            if (singlePass) {
                ownedPreviewMaterials.push_back(singlePass);
                previewPassMaterials.clear();
                previewPassMaterials.push_back(singlePass);
            } else {
                return nullptr;
            }
        }
    }

    struct PreviewPassBinding
    {
        std::shared_ptr<InxMaterial> material;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSet materialDescSet = VK_NULL_HANDLE;
        const ShaderProgram *program = nullptr;
    };

    auto refreshPassBinding = [&](PreviewPassBinding &binding) -> bool {
        InxMaterial *passMat = binding.material.get();
        if (!passMat) {
            return false;
        }

        MaterialPassRenderData *rd = m_vkCore->GetOrCreatePreviewMaterialPass(binding.material);
        if (!rd || !rd->isValid || rd->descriptorSet == VK_NULL_HANDLE) {
            if (!m_vkCore->RefreshPreviewMaterialPipeline(binding.material, false)) {
                return false;
            }
            rd = m_vkCore->GetOrCreatePreviewMaterialPass(binding.material);
        }

        if (rd && rd->isValid && rd->descriptorSet != VK_NULL_HANDLE) {
            passMat->SetPassPipeline(ShaderCompileTarget::Forward, rd->pipeline);
            passMat->SetPassPipelineLayout(ShaderCompileTarget::Forward, rd->pipelineLayout);
            passMat->SetPassDescriptorSet(ShaderCompileTarget::Forward, rd->descriptorSet);
            passMat->SetPassShaderProgram(ShaderCompileTarget::Forward, rd->shaderProgram);
        }

        binding.pipeline = passMat->GetPassPipeline(ShaderCompileTarget::Forward);
        binding.pipelineLayout = passMat->GetPassPipelineLayout(ShaderCompileTarget::Forward);
        binding.materialDescSet = passMat->GetPassDescriptorSet(ShaderCompileTarget::Forward);
        binding.program = passMat->GetPassShaderProgram(ShaderCompileTarget::Forward);

        return binding.pipeline != VK_NULL_HANDLE && binding.pipelineLayout != VK_NULL_HANDLE &&
               binding.materialDescSet != VK_NULL_HANDLE && binding.program != nullptr;
    };

    std::vector<PreviewPassBinding> passBindings;
    passBindings.reserve(previewPassMaterials.size());
    for (const auto &passMat : previewPassMaterials) {
        PreviewPassBinding binding{};
        binding.material = passMat;

        // Material property synchronization may publish a replacement set 0
        // (for example after resolving a texture). Do it before capturing the
        // immutable preview binding, otherwise this command buffer can retain
        // the descriptor generation that was just retired.
        m_vkCore->UpdateMaterialUBO(*binding.material);
        if (m_vkCore->GetMaterialPipelineManager().HasPendingTextureProperties(binding.material->GetMaterialKey())) {
            if (texturePending)
                *texturePending = true;
            return nullptr;
        }
        if (!refreshPassBinding(binding)) {
            INXLOG_WARN("GPUMaterialPreview: preview pass pipeline not ready");
            return nullptr;
        }
        passBindings.push_back(binding);
    }

    // ------------------------------------------------------------------
    // Prepare preview scene UBO (camera looking at sphere)
    // ------------------------------------------------------------------
    float aspect = 1.0f;
    glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, kPreviewCameraDistance), glm::vec3(0.0f, 0.0f, 0.0f),
                                 glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 100.0f, 0.1f);
    proj[1][1] *= -1.0f; // Vulkan Y-flip

    UniformBufferObject sceneUBO{};
    sceneUBO.model = previewModel;
    sceneUBO.view = view;
    sceneUBO.proj = proj;
    sceneUBO.previousViewProj = proj * view;
    sceneUBO.inverseViewProj = glm::inverse(proj * view);
    sceneUBO.projectionParams = glm::vec4(0.1f, 100.0f, 0.01f, 0.001f);
    sceneUBO.zBufferParams = depth::LinearizationParams(0.1f, 100.0f);

    // ------------------------------------------------------------------
    // Prepare preview lighting UBO.
    // Single key light from above plus the ambient probe — the probe already
    // keeps unlit regions readable, and a lower fill light made previews
    // look lit from below.
    // ------------------------------------------------------------------
    ShaderLightingUBO lightingUBO{};
    memset(&lightingUBO, 0, sizeof(lightingUBO));
    lightingUBO.lightCounts = glm::ivec4(1, 0, 0, 0);
    lightingUBO.ambientColor = glm::vec4(0.06f, 0.06f, 0.07f, 1.0f);
    lightingUBO.ambientSkyColor = glm::vec4(0.18f, 0.19f, 0.22f, 0.55f);
    lightingUBO.ambientEquatorColor = glm::vec4(0.09f, 0.10f, 0.12f, 1.0f);
    lightingUBO.ambientGroundColor = glm::vec4(0.04f, 0.035f, 0.03f, 0.30f);
    lightingUBO.cameraPos = glm::vec4(0.0f, 0.0f, kPreviewCameraDistance, 1.0f);

    // Main key light — top-right-front
    lightingUBO.directionalLights[0].direction = glm::vec4(glm::normalize(glm::vec3(-0.8f, -1.0f, -0.6f)), 0.0f);
    // color.rgb = color * intensity, color.a = intensity
    lightingUBO.directionalLights[0].color = glm::vec4(2.0f * 1.0f, 2.0f * 0.95f, 2.0f * 0.9f, 2.0f);
    lightingUBO.directionalLights[0].metadata =
        glm::uvec4(~0u, static_cast<uint32_t>(LightInfluenceDomain::Geometry), 0u, 0u);

    // ------------------------------------------------------------------
    // Prepare engine globals UBO (minimal)
    // ------------------------------------------------------------------
    EngineGlobalsUBO globalsUBO{};
    globalsUBO.projectionParams = sceneUBO.projectionParams;
    globalsUBO.zBufferParams = sceneUBO.zBufferParams;
    globalsUBO.screenParams =
        glm::vec4(static_cast<float>(renderSize), static_cast<float>(renderSize), 1.0f / renderSize, 1.0f / renderSize);
    globalsUBO.worldSpaceCameraPos = glm::vec4(0.0f, 0.0f, kPreviewCameraDistance, 1.0f);

    const VkBuffer sceneUBOBuf = m_previewSceneUbo->GetBuffer();
    const VkBuffer lightingUBOBuf = m_previewLightingUbo->GetBuffer();
    const VkBuffer globalsUBOBuf = m_previewGlobalsUbo->GetBuffer();
    const VkBuffer instanceSSBOBuf = m_previewInstanceBuffer->GetBuffer();
    VkDescriptorSet shadowDesc = VK_NULL_HANDLE;
    VkDescriptorSet globalsDesc = VK_NULL_HANDLE;

    const bool requiresPerViewSet = std::any_of(passBindings.begin(), passBindings.end(), [](const auto &binding) {
        return binding.program && binding.program->HasDeclaredDescriptorSet(1);
    });
    const bool requiresGlobalsSet = std::any_of(passBindings.begin(), passBindings.end(), [](const auto &binding) {
        return binding.program && binding.program->HasDeclaredDescriptorSet(2);
    });

    if (requiresPerViewSet) {
        if (!m_fallbackShadowDescLease.IsValid()) {
            m_fallbackShadowDescLease = m_vkCore->AllocatePerViewDescriptorLease();
            m_fallbackShadowDescSet = m_fallbackShadowDescLease.set;
            if (m_fallbackShadowDescLease.IsValid()) {
                m_vkCore->UpdatePerViewLightingBuffer(m_fallbackShadowDescSet, lightingUBOBuf,
                                                      sizeof(ShaderLightingUBO));
                m_vkCore->UpdatePerViewCameraBuffer(m_fallbackShadowDescSet, sceneUBOBuf, sizeof(UniformBufferObject));
            }
        }
        shadowDesc = m_fallbackShadowDescSet;
        if (shadowDesc == VK_NULL_HANDLE) {
            INXLOG_WARN("GPUMaterialPreview: shader requires set 1 but no shadow descriptor is available");
            return nullptr;
        }
    }

    if (requiresGlobalsSet)
        globalsDesc = m_previewGlobalsSet;

    const bool multisampled = m_sampleCount != VK_SAMPLE_COUNT_1_BIT;
    auto output =
        m_display->Prepare(multisampled ? m_resolveColor.GetView() : m_msaaColor.GetView(), m_colorFormat, size);
    if (!output) {
        if (texturePending)
            *texturePending = true;
        return nullptr;
    }
    constexpr VkDeviceSize transferBytes = sizeof(sceneUBO) + sizeof(lightingUBO) + sizeof(globalsUBO) +
                                           sizeof(previewModel) + sizeof(std::array<uint32_t, 4>) + sizeof(glm::mat4) +
                                           sizeof(GPUInstanceAuxData);
    PreviewCommands commands(*m_vkCore, transferBytes, m_gpuNanoseconds);
    const VkCommandBuffer cmd = commands.Get();
    if (cmd == VK_NULL_HANDLE)
        return nullptr;

    vkCmdUpdateBuffer(cmd, sceneUBOBuf, 0, sizeof(sceneUBO), &sceneUBO);
    vkCmdUpdateBuffer(cmd, lightingUBOBuf, 0, sizeof(lightingUBO), &lightingUBO);
    vkCmdUpdateBuffer(cmd, globalsUBOBuf, 0, sizeof(globalsUBO), &globalsUBO);
    vkCmdUpdateBuffer(cmd, instanceSSBOBuf, 0, sizeof(previewModel), &previewModel);
    const std::array<uint32_t, 4> emptySkinInstance{};
    const glm::mat4 identityBone(1.0f);
    vkCmdUpdateBuffer(cmd, m_previewSkinInstanceBuffer->GetBuffer(), 0, sizeof(emptySkinInstance),
                      emptySkinInstance.data());
    vkCmdUpdateBuffer(cmd, m_previewSkinPaletteBuffer->GetBuffer(), 0, sizeof(identityBone), &identityBone);
    GPUInstanceAuxData instanceAux{};
    instanceAux.previousModel = previewModel;
    instanceAux.layerMask = ~0u;
    vkCmdUpdateBuffer(cmd, m_previewInstanceAuxBuffer->GetBuffer(), 0, sizeof(instanceAux), &instanceAux);

    // Memory barrier: make UBO writes visible to shaders
    VkMemoryBarrier uboBarrier{};
    uboBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    uboBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    uboBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &uboBarrier,
                         0, nullptr, 0, nullptr);

    // ------------------------------------------------------------------
    // Begin the RHI-selected graphics attachment path.
    // ------------------------------------------------------------------
    VkClearValue clearValues[2];
    clearValues[0].color = {{kPreviewMatteR, kPreviewMatteG, kPreviewMatteB, kPreviewMatteA}};
    clearValues[1].depthStencil = {0.0f, 0};

    vkrender::TransitionTrackedImageLayout(cmd, m_msaaColor.GetImage(), VK_IMAGE_ASPECT_COLOR_BIT, m_msaaColorLayout,
                                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    if (m_depth.HasView()) {
        vkrender::TransitionTrackedImageLayout(cmd, m_depth.GetImage(), rhi::ToVkImageAspectMask(m_depthFormat),
                                               m_depthLayout, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = m_msaaColor.GetView();
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue = clearValues[0];

    VkRenderingAttachmentInfo depthAttachment{};
    if (m_depth.HasView()) {
        depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAttachment.imageView = m_depth.GetView();
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.clearValue = clearValues[1];
    }

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.extent = {static_cast<uint32_t>(renderSize), static_cast<uint32_t>(renderSize)};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = m_depth.HasView() ? &depthAttachment : nullptr;
    m_dynamicRenderingCommands.begin(cmd, &renderingInfo);

    // Viewport & scissor
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(renderSize);
    viewport.height = static_cast<float>(renderSize);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {static_cast<uint32_t>(renderSize), static_cast<uint32_t>(renderSize)};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Bind sphere geometry
    VkBuffer vbo = m_sphereVBO->GetBuffer();
    VkDeviceSize offsets[] = {0};
    commands.ReadGeometry(*m_sphereVBO, *m_sphereIBO);
    vkCmdBindVertexBuffers(cmd, 0, 1, &vbo, offsets);
    vkCmdBindIndexBuffer(cmd, m_sphereIBO->GetBuffer(), 0, VK_INDEX_TYPE_UINT32);

    // Push constants (identity model + normalMat)
    struct PushConstants
    {
        glm::mat4 model;
        glm::mat4 normalMat;
    };
    PushConstants pushData{};
    pushData.model = previewModel;
    pushData.normalMat = previewNormal;
    bool anyDrawn = false;
    for (auto &binding : passBindings) {
        const uint64_t descRaw = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(binding.materialDescSet));
        const auto *descriptor = m_vkCore->GetMaterialPipelineManager().GetDescriptorManager().FindLiveDescriptorSet(
            binding.materialDescSet);
        if (!descriptor) {
            static int deadPreviewDescWarnCount = 0;
            if (deadPreviewDescWarnCount++ < 8) {
                INXLOG_WARN("GPUMaterialPreview: DEAD set0 descriptor=", descRaw, " material='",
                            binding.material ? binding.material->GetMaterialKey() : std::string(), "' - skipping pass");
            }
            continue;
        }

        commands.ReadMaterial(*descriptor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, binding.pipeline);

        // Set 0 — material (scene UBO + lighting UBO + material UBO + textures)
        vkdebug::CmdBindDescriptorSetsTracked("GPUMaterialPreview.RenderToPixels.Set0", cmd,
                                              VK_PIPELINE_BIND_POINT_GRAPHICS, binding.pipelineLayout, 0, 1,
                                              &binding.materialDescSet, 0, nullptr);

        // Set 1 — shadow (if the shader declares it)
        if (binding.program->HasDeclaredDescriptorSet(1)) {
            vkdebug::CmdBindDescriptorSetsTracked("GPUMaterialPreview.RenderToPixels.Set1", cmd,
                                                  VK_PIPELINE_BIND_POINT_GRAPHICS, binding.pipelineLayout, 1, 1,
                                                  &shadowDesc, 0, nullptr);
        }

        // Set 2 — engine globals (if the shader declares it)
        if (binding.program->HasDeclaredDescriptorSet(2)) {
            vkdebug::CmdBindDescriptorSetsTracked("GPUMaterialPreview.RenderToPixels.Set2", cmd,
                                                  VK_PIPELINE_BIND_POINT_GRAPHICS, binding.pipelineLayout, 2, 1,
                                                  &globalsDesc, 0, nullptr);
        }

        if (binding.program->UsesBindlessTextureABI()) {
            auto &rhiDevice = m_vkCore->GetDeviceContext().GetRhiDevice();
            const auto bindlessBinding = rhiDevice.GetBindlessTextureTableBinding();
            const VkDescriptorSet bindlessSet = rhiDevice.Resolve(bindlessBinding.group);
            if (bindlessSet == VK_NULL_HANDLE) {
                static int missingBindlessSetErrorCount = 0;
                if (missingBindlessSetErrorCount++ < 8)
                    INXLOG_ERROR(
                        "Material preview pass skipped because the device-global texture table is unavailable");
                continue;
            } else {
                vkdebug::CmdBindDescriptorSetsTracked("GPUMaterialPreview.RenderToPixels.BindlessTextures", cmd,
                                                      VK_PIPELINE_BIND_POINT_GRAPHICS, binding.pipelineLayout,
                                                      ShaderProgram::BindlessTextureSet, 1, &bindlessSet, 0, nullptr);
                if (const auto *indices =
                        m_vkCore->GetMaterialPipelineManager().GetDescriptorManager().GetBindlessTextureIndices(
                            binding.material->GetMaterialKey())) {
                    rhiDevice.MarkBindlessTexturesUsed(indices->empty() ? nullptr : indices->data(), indices->size());
                }
            }
        }

        if (binding.program->HasVertexPushConstants())
            vkCmdPushConstants(cmd, binding.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants),
                               &pushData);
        vkCmdDrawIndexed(cmd, m_sphereIndexCount, 1, 0, 0, 0);
        anyDrawn = true;
    }

    m_dynamicRenderingCommands.end(cmd);
    m_msaaColorLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    if (m_depth.HasView())
        m_depthLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // ------------------------------------------------------------------
    // Resolve MSAA → single-sample resolve image
    // ------------------------------------------------------------------
    if (m_sampleCount != VK_SAMPLE_COUNT_1_BIT) {
        // Transition MSAA color → TRANSFER_SRC
        vkrender::TransitionTrackedImageLayout(cmd, m_msaaColor.GetImage(), VK_IMAGE_ASPECT_COLOR_BIT,
                                               m_msaaColorLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        // Transition resolve → TRANSFER_DST
        vkrender::TransitionTrackedImageLayout(cmd, m_resolveColor.GetImage(), VK_IMAGE_ASPECT_COLOR_BIT,
                                               m_resolveColorLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        // Resolve
        VkImageResolve resolveRegion{};
        resolveRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        resolveRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        resolveRegion.extent = {static_cast<uint32_t>(renderSize), static_cast<uint32_t>(renderSize), 1};
        vkCmdResolveImage(cmd, m_msaaColor.GetImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_resolveColor.GetImage(),
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &resolveRegion);
    }

    const float sphereRadius = 0.5f * kPreviewModelScale;
    const float radius01 = 0.5f / std::tan(glm::radians(45.0f) * 0.5f) * sphereRadius /
                           std::sqrt(kPreviewCameraDistance * kPreviewCameraDistance - sphereRadius * sphereRadius);
    m_display->Record(cmd, *output, multisampled ? m_resolveColor.GetImage() : m_msaaColor.GetImage(),
                      multisampled ? m_resolveColorLayout : m_msaaColorLayout, size, radius01);
    for (const auto &producer : producers) {
        const auto point = m_vkCore->GetBackendContext().Queues().DependencyFor(producer, rhi::QueueRole::Graphics);
        commands.AddDependency({point.semaphore, point.value,
                                VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT});
    }
    m_activeSubmission = commands.Submit([materials = std::move(ownedPreviewMaterials), finalView = output->texture,
                                          vertices = m_sphereVBO, indices = m_sphereIBO] {});
    progressed = true;
    output->submission = m_activeSubmission;
    // Submit even an empty draw so tracked layouts and recording ownership
    // remain correct, but never publish its transparent clear as a thumbnail.
    return anyDrawn ? output : nullptr;
}

// ============================================================================
// Resource management
// ============================================================================

bool GPUMaterialPreview::EnsureViewResources(bool &progressed)
{
    if (m_previewGlobalsSet != VK_NULL_HANDLE)
        return true;

    auto &rm = m_vkCore->GetResourceManager();
    m_previewSceneUbo = rm.CreateUniformBuffer(sizeof(UniformBufferObject));
    m_previewLightingUbo = rm.CreateUniformBuffer(sizeof(ShaderLightingUBO));
    m_previewGlobalsUbo = rm.CreateUniformBuffer(sizeof(EngineGlobalsUBO));
    m_previewInstanceBuffer = rm.CreateStorageBuffer(sizeof(glm::mat4), false);
    m_previewSkinInstanceBuffer = rm.CreateStorageBuffer(64, false);
    m_previewSkinPaletteBuffer = rm.CreateStorageBuffer(sizeof(glm::mat4), false);
    m_previewInstanceAuxBuffer = rm.CreateStorageBuffer(sizeof(GPUInstanceAuxData), false);
    if (!m_previewSceneUbo || !m_previewLightingUbo || !m_previewGlobalsUbo || !m_previewInstanceBuffer ||
        !m_previewSkinInstanceBuffer || !m_previewSkinPaletteBuffer || !m_previewInstanceAuxBuffer)
        throw std::runtime_error("Failed to allocate isolated material-preview view buffers");

    const VkDevice device = m_vkCore->GetDevice();
    const VkDescriptorSetLayout layout = m_vkCore->GetGlobalsDescSetLayout();
    if (layout == VK_NULL_HANDLE)
        throw std::logic_error("Material preview requires the renderer globals descriptor layout");
    m_previewGlobalsLease = m_vkCore->GetDeviceContext().GetRhiDevice().GetDescriptorManager().Allocate(
        layout, vk::DescriptorArena::ViewPersistent);
    if (!m_previewGlobalsLease.IsValid())
        throw std::runtime_error("Failed to allocate isolated material-preview globals descriptor set");
    m_previewGlobalsSet = m_previewGlobalsLease.set;

    VkDescriptorBufferInfo buffers[5]{};
    buffers[0] = {m_previewGlobalsUbo->GetBuffer(), 0, sizeof(EngineGlobalsUBO)};
    buffers[1] = {m_previewInstanceBuffer->GetBuffer(), 0, VK_WHOLE_SIZE};
    buffers[2] = {m_previewSkinInstanceBuffer->GetBuffer(), 0, VK_WHOLE_SIZE};
    buffers[3] = {m_previewSkinPaletteBuffer->GetBuffer(), 0, VK_WHOLE_SIZE};
    buffers[4] = {m_previewInstanceAuxBuffer->GetBuffer(), 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet writes[5]{};
    for (uint32_t binding = 0; binding < 5; ++binding) {
        writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[binding].dstSet = m_previewGlobalsSet;
        writes[binding].dstBinding = binding;
        writes[binding].descriptorCount = 1;
        writes[binding].descriptorType =
            binding == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[binding].pBufferInfo = &buffers[binding];
    }
    vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);
    progressed = true;
    return true;
}

void GPUMaterialPreview::DestroyViewResources()
{
    m_activeSubmission.reset();
    if (m_vkCore) {
        auto &descriptorManager = m_vkCore->GetDeviceContext().GetRhiDevice().GetDescriptorManager();
        descriptorManager.Retire(m_fallbackShadowDescLease);
        descriptorManager.Retire(m_previewGlobalsLease);
    }
    m_fallbackShadowDescLease = {};
    m_fallbackShadowDescSet = VK_NULL_HANDLE;
    m_previewGlobalsLease = {};
    m_previewGlobalsSet = VK_NULL_HANDLE;
    m_previewSkinPaletteBuffer.reset();
    m_previewSkinInstanceBuffer.reset();
    m_previewInstanceAuxBuffer.reset();
    m_previewInstanceBuffer.reset();
    m_previewGlobalsUbo.reset();
    m_previewLightingUbo.reset();
    m_previewSceneUbo.reset();
}

bool GPUMaterialPreview::EnsureResources(int size, bool &progressed)
{
    // Cache format info from MaterialPipelineManager
    auto &mpm = m_vkCore->GetMaterialPipelineManager();
    VkFormat colorFormat = mpm.GetColorFormat();
    VkFormat depthFormat = mpm.GetDepthFormat();
    VkSampleCountFlagBits sampleCount = mpm.GetSampleCount();
    const auto dynamicCommands = rhi::ResolveDynamicRenderingCommands(m_vkCore->GetDevice());
    if (!m_vkCore->GetDeviceContext().GetRhiDevice().GetVulkanFeatures().dynamicRendering.enabled ||
        !dynamicCommands.IsValid()) {
        INXLOG_ERROR("GPUMaterialPreview: dynamic rendering is required for material previews");
        return false;
    }

    const bool renderConfigChanged =
        (m_currentSize != 0) &&
        (colorFormat != m_colorFormat || depthFormat != m_depthFormat || sampleCount != m_sampleCount);

    if (renderConfigChanged) {
        DestroyAttachments();
        progressed = true;
    }

    m_colorFormat = colorFormat;
    m_depthFormat = depthFormat;
    m_sampleCount = sampleCount;
    m_dynamicRenderingCommands = dynamicCommands;

    if (m_currentSize != size) {
        DestroyAttachments();
        CreateAttachments(size);
        m_currentSize = size;
        progressed = true;
        if (m_msaaColor.IsValid())
            PublishRenderView();
    }

    return m_msaaColor.IsValid() && EnsureViewResources(progressed);
}

void GPUMaterialPreview::CreateAttachments(int size)
{
    VkDevice device = m_vkCore->GetDevice();
    VmaAllocator allocator = m_vkCore->GetDeviceContext().GetVmaAllocator();
    uint32_t w = static_cast<uint32_t>(size);
    uint32_t h = static_cast<uint32_t>(size);

    // MSAA color image
    m_msaaColor.Create(allocator, device, w, h, m_colorFormat, VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                           (m_sampleCount == VK_SAMPLE_COUNT_1_BIT ? VK_IMAGE_USAGE_SAMPLED_BIT : 0),
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_sampleCount);
    m_msaaColor.CreateView(m_colorFormat, VK_IMAGE_ASPECT_COLOR_BIT);

    // Resolve image (1x sample, for MSAA resolve + display)
    if (m_sampleCount != VK_SAMPLE_COUNT_1_BIT) {
        m_resolveColor.Create(allocator, device, w, h, m_colorFormat, VK_IMAGE_TILING_OPTIMAL,
                              VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                  VK_IMAGE_USAGE_SAMPLED_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_resolveColor.CreateView(m_colorFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    // Depth image (MSAA)
    if (m_depthFormat != VK_FORMAT_UNDEFINED) {
        m_depth.Create(allocator, device, w, h, m_depthFormat, VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_sampleCount);
        m_depth.CreateView(m_depthFormat, rhi::ToVkImageAspectMask(m_depthFormat));
    }
}

void GPUMaterialPreview::DestroyAttachments()
{
    UnpublishRenderView();
    RetirePreviewAttachments(*m_vkCore, m_msaaColor, m_resolveColor, m_depth);
    m_msaaColorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    m_resolveColorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    m_depthLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    m_currentSize = 0;
}

void GPUMaterialPreview::PublishRenderView()
{
    if (!m_vkCore || !m_msaaColor.IsValid())
        return;
    auto &device = m_vkCore->GetDeviceContext().GetRhiDevice();
    if (m_renderView.color.IsValid())
        device.Release(m_renderView.color);
    if (m_renderView.depth.IsValid())
        device.Release(m_renderView.depth);

    m_renderView.device = m_vkCore->GetDeviceContext().GetDeviceId();
    m_renderView.width = static_cast<uint32_t>(m_currentSize);
    m_renderView.height = static_cast<uint32_t>(m_currentSize);
    m_renderView.colorFormat = rhi::FromVkFormat(m_colorFormat);
    m_renderView.depthFormat = rhi::FromVkFormat(m_depthFormat);
    m_renderView.samples = rhi::FromVkSampleCount(m_sampleCount);
    m_renderView.color = device.RegisterTextureView(m_msaaColor.GetView());
    m_renderView.depth = m_depth.HasView() ? device.RegisterTextureView(m_depth.GetView()) : rhi::TextureViewHandle{};
    ++m_renderView.revision;
}

void GPUMaterialPreview::UnpublishRenderView()
{
    if (!m_vkCore)
        return;
    auto &device = m_vkCore->GetDeviceContext().GetRhiDevice();
    if (m_renderView.color.IsValid())
        device.Release(m_renderView.color);
    if (m_renderView.depth.IsValid())
        device.Release(m_renderView.depth);
    const bool wasPublished = m_renderView.color.IsValid() || m_renderView.depth.IsValid() || m_renderView.width != 0;
    m_renderView.color = {};
    m_renderView.depth = {};
    m_renderView.width = 0;
    m_renderView.height = 0;
    if (wasPublished)
        ++m_renderView.revision;
}

bool GPUMaterialPreview::PrepareSphereBuffers(bool &progressed)
{
    const auto &vertices = PrimitiveMeshes::GetSphereVertices();
    const auto &indices = PrimitiveMeshes::GetSphereIndices();

    if (vertices.empty() || indices.empty()) {
        throw std::logic_error("Material preview sphere geometry is empty");
    }

    if (!PreparePreviewBuffer(*m_vkCore, m_sphereVertexUpload, vertices.data(), vertices.size() * sizeof(Vertex),
                              rhi::BufferUsageFlags::Vertex, progressed))
        return false;
    if (!PreparePreviewBuffer(*m_vkCore, m_sphereIndexUpload, indices.data(), indices.size() * sizeof(uint32_t),
                              rhi::BufferUsageFlags::Index, progressed))
        return false;
    m_sphereVBO = m_sphereVertexUpload->GetBuffer();
    m_sphereIBO = m_sphereIndexUpload->GetBuffer();
    m_sphereIndexCount = static_cast<uint32_t>(indices.size());
    return true;
}

} // namespace infernux
