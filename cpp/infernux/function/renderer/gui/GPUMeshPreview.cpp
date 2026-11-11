#include <core/math/DepthConvention.h>
/**
 * @file GPUMeshPreview.cpp
 * @brief GPU mesh preview — renders an arbitrary InxMesh with per-submesh
 *        materials into independent GPU textures for editor thumbnails
 * (model / prefab previews).
 */

#include "GPUMeshPreview.h"
#include "InxError.h"
#include "PreviewDisplay.h"
#include "PreviewResources.h"
#include <backends/imgui_impl_vulkan.h>
#include <function/renderer/EngineGlobals.h>
#include <function/renderer/InxRenderStruct.h>
#include <function/renderer/InxVkCoreModular.h>
#include <function/renderer/MaterialPipelineManager.h>
#include <function/renderer/RenderInstanceHistory.h>
#include <function/renderer/shader/ShaderProgram.h>
#include <function/renderer/vk/DescriptorBindTrace.h>
#include <function/renderer/vk/RhiVulkanTypes.h>
#include <function/renderer/vk/VkRenderUtils.h>
#include <function/renderer/vk/VkResourceManager.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxSkinnedMesh/InxSkinnedMesh.h>
#include <function/scene/Light.h>
#include <function/scene/LightingData.h>

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
constexpr float kMeshPreviewFovDeg = 30.0f;
constexpr int kPreviewSupersampleFactor = 2;

/// @brief Compute camera transform that fits a bounding box into the viewport.
struct FitCameraResult
{
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec3 cameraPos;
};

void SetPreviewDepthParams(UniformBufferObject &ubo, const glm::mat4 &projection)
{
    const auto inverse = glm::inverse(glm::dmat4(projection));
    const auto nearPoint = inverse * glm::dvec4(0, 0, depth::Near, 1);
    const auto farPoint = inverse * glm::dvec4(0, 0, depth::Far, 1);
    const float nearClip = static_cast<float>(std::abs(nearPoint.z / nearPoint.w));
    const float farClip = static_cast<float>(std::abs(farPoint.z / farPoint.w));
    ubo.projectionParams = {nearClip, farClip, 1.0f / farClip, nearClip / farClip};
    ubo.zBufferParams = depth::LinearizationParams(nearClip, farClip);
}

FitCameraResult FitCameraToBounds(const glm::vec3 &boundsMin, const glm::vec3 &boundsMax, float fovDeg)
{
    const glm::vec3 center = (boundsMin + boundsMax) * 0.5f;
    const glm::vec3 extent = boundsMax - boundsMin;
    const float maxExtent = std::max({extent.x, extent.y, extent.z, 0.001f});

    // Camera looks from front-top-right, raised higher above center
    const glm::vec3 viewDir = glm::normalize(glm::vec3(-0.6f, -0.5f, -0.7f));
    const float halfFov = glm::radians(fovDeg) * 0.5f;
    // Bounding sphere radius from actual diagonal, with padding
    const float radius = glm::length(extent) * 0.5f;
    // Tighter framing (was 1.15) so model / prefab thumbnails sit closer and larger.
    const float distance = (radius / std::sin(halfFov)) * 0.98f;

    // Shift look-at target slightly upward so the model isn't bottom-heavy
    const glm::vec3 lookAt = center + glm::vec3(0.0f, maxExtent * 0.05f, 0.0f);
    const glm::vec3 cameraPos = lookAt - viewDir * distance;

    FitCameraResult result;
    result.cameraPos = cameraPos;
    result.view = glm::lookAt(cameraPos, lookAt, glm::vec3(0.0f, 1.0f, 0.0f));
    result.proj = glm::perspective(glm::radians(fovDeg), 1.0f, distance * 4.0f, distance * 0.01f);
    result.proj[1][1] *= -1.0f; // Vulkan Y-flip
    return result;
}

} // anonymous namespace

// ============================================================================
// Constructor / Destructor
// ============================================================================

GPUMeshPreview::GPUMeshPreview(InxVkCoreModular *vkCore) : m_vkCore(vkCore)
{
    m_renderView.id = rhi::AllocateRenderViewId();
    m_renderView.device = vkCore ? vkCore->GetDeviceContext().GetDeviceId() : rhi::InvalidDeviceId;
    m_renderView.kind = rhi::RenderViewKind::Preview;
    m_renderView.output = rhi::RenderOutputKind::OffscreenTexture;
    m_renderView.revision = 1;
    if (vkCore)
        m_display = std::make_unique<PreviewDisplay>(*vkCore, m_renderView.id);
}

GPUMeshPreview::~GPUMeshPreview()
{
    if (!m_vkCore)
        return;
    DestroyImGuiDisplayDescriptor();
    DestroyAttachments();
    DestroyViewResources();
}

// ============================================================================
// GPU thumbnail
// ============================================================================

bool GPUMeshPreview::PrepareGeometry(const InxMesh &mesh, std::shared_ptr<PreviewGeometryUpload> &upload,
                                     bool &progressed, const std::shared_ptr<const InxSkinnedMesh> &skin)
{
    auto &resources = m_vkCore->GetResourceManager();
    const auto geometry = skin ? std::shared_ptr<const MeshGeometry>() : mesh.GetGeometrySnapshot();
    const auto format = skin ? skin->indexFormat : mesh.GetIndexFormat();
    const auto &vertices = skin ? skin->baseVertices : geometry->vertices;
    const auto &indices = skin ? skin->indices : geometry->indices;
    if (!upload || upload->geometry != geometry || upload->skin != skin || upload->format != format) {
        const auto resolved = ResolveMeshIndexFormat(format, vertices.size(), indices);
        const auto indexType = resolved == MeshIndexFormat::UInt16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
        auto vertexUpload =
            resources.CreateBufferUpload(vertices.size() * sizeof(Vertex), rhi::BufferUsageFlags::Vertex);
        auto indexUpload = resources.CreateBufferUpload(
            indices.size() * (indexType == VK_INDEX_TYPE_UINT16 ? sizeof(uint16_t) : sizeof(uint32_t)),
            rhi::BufferUsageFlags::Index);
        upload = std::make_shared<PreviewGeometryUpload>(
            PreviewGeometryUpload{geometry, skin, format, indexType, std::move(vertexUpload), std::move(indexUpload)});
        progressed = true;
    }
    // One bounded copy per advance, never a fence wait. A later call observes
    // the original ticket; it cannot re-submit a block still in flight.
    auto *owner = ActivePreviewOwner(*m_vkCore);
    for (const auto &ticket : {upload->vertices, upload->indices}) {
        const bool published = ticket->IsPublished();
        if (resources.TryPublishBufferUpload(ticket)) {
            progressed |= !published;
            continue;
        }
        if (ticket->IsPending())
            return false;
        const auto offset = ticket->GetSubmittedBytes();
        const VkDeviceSize unit = ticket == upload->vertices                  ? sizeof(float)
                                  : upload->indexType == VK_INDEX_TYPE_UINT16 ? sizeof(uint16_t)
                                                                              : sizeof(uint32_t);
        const auto bytes = PreviewUploadBytes(owner, ticket->GetSize() - offset, unit);
        if (!bytes)
            return false;
        if (ticket != upload->vertices && upload->indexType == VK_INDEX_TYPE_UINT16) {
            progressed |= resources.TryAppendIndex16Upload(ticket, indices.data() + offset / unit, bytes, owner);
        } else {
            const auto *source = ticket == upload->vertices ? reinterpret_cast<const std::byte *>(vertices.data())
                                                            : reinterpret_cast<const std::byte *>(indices.data());
            progressed |=
                resources.TryAppendBufferUpload(ticket, source + offset, bytes, owner, bytes == unit, unit,
                                                {ticket == upload->vertices ? upload->indices.get() : nullptr});
        }
        return false;
    }
    return true;
}

bool GPUMeshPreview::PrepareMaterials(const std::vector<std::shared_ptr<InxMaterial>> &sources,
                                      std::vector<std::shared_ptr<InxMaterial>> &prepared, bool &progressed)
{
    if (prepared.empty() && !sources.empty()) {
        const auto defaultMaterial = AssetRegistry::Instance().GetBuiltinMaterial("DefaultLit");
        const auto prepare = [this](const std::shared_ptr<InxMaterial> &source) -> std::shared_ptr<InxMaterial> {
            if (!source)
                return {};
            auto material = source->Clone();
            if (!m_vkCore->RefreshPreviewMaterialPipeline(material, material->GetVertShaderName(),
                                                          material->GetFragShaderName(), false))
                return {};
            return material;
        };
        prepared.reserve(sources.size());
        for (const auto &source : sources) {
            auto material = prepare(source);
            if (!material && source && source != defaultMaterial)
                material = prepare(defaultMaterial);
            prepared.push_back(std::move(material));
        }
        progressed = true;
    }

    bool ready = true;
    for (auto &material : prepared) {
        if (!material)
            continue;
        if (material->NeedsTextureAssetResolution() &&
            !m_vkCore->RefreshPreviewMaterialPipeline(material, material->GetVertShaderName(),
                                                      material->GetFragShaderName(), false)) {
            material.reset();
            continue;
        }
        m_vkCore->UpdateMaterialUBO(*material);
        if (m_vkCore->GetMaterialPipelineManager().HasPendingTextureProperties(material->GetMaterialKey())) {
            ready = false;
            continue;
        }
        const auto *pass = m_vkCore->GetOrCreatePreviewMaterialPass(material);
        if (!pass || !pass->isValid || pass->descriptorSet == VK_NULL_HANDLE)
            material.reset();
    }
    return ready;
}

std::shared_ptr<PreviewImage> GPUMeshPreview::BeginRender(const InxMesh &mesh,
                                                          const std::vector<std::shared_ptr<InxMaterial>> &materials,
                                                          int size, std::shared_ptr<PreviewGeometryUpload> &geometry,
                                                          bool &progressed, bool &materialsChanged)
{
    if (!m_vkCore || size <= 0)
        return nullptr;
    const auto camera = FitCameraToBounds(mesh.GetBoundsMin(), mesh.GetBoundsMax(), kMeshPreviewFovDeg);
    const auto &view = camera.view;
    const auto &proj = camera.proj;
    const auto &cameraPos = camera.cameraPos;
    if (m_activeSubmission && !m_activeSubmission->IsComplete())
        return nullptr;
    if (m_activeSubmission && m_activeSubmission->HasExecutionTiming())
        m_gpuNanoseconds = m_activeSubmission->GetExecutionNanoseconds();
    m_activeSubmission.reset();

    const auto &vertices = mesh.GetVertices();
    const auto &indices = mesh.GetIndices();
    if (vertices.empty() || indices.empty())
        return nullptr;

    const int renderSize = std::max(size, size * kPreviewSupersampleFactor);

    if (!EnsureResources(renderSize, progressed))
        return nullptr;

    struct SubmeshBinding
    {
        const SubMesh *submesh;
        std::shared_ptr<InxMaterial> ownedMaterial;
        VkPipeline pipeline;
        VkPipelineLayout pipelineLayout;
        VkDescriptorSet materialDescSet;
        const ShaderProgram *program;
    };

    std::vector<SubmeshBinding> bindings;
    bindings.reserve(mesh.GetSubMeshCount());
    for (const auto &submesh : mesh.GetSubMeshes()) {
        if (!submesh.indexCount)
            continue;
        const auto &material = materials.at(submesh.materialSlot);
        if (!material)
            continue;
        // The preparation owns the descriptors and their resolved texture slots.
        // A changed slot invalidates this candidate before any geometry is used.
        if (material->NeedsTextureAssetResolution() ||
            m_vkCore->GetMaterialPipelineManager().HasPendingTextureProperties(material->GetMaterialKey())) {
            materialsChanged = true;
            return nullptr;
        }
        auto *pass = m_vkCore->GetOrCreatePreviewMaterialPass(material);
        if (!pass || !pass->isValid || pass->descriptorSet == VK_NULL_HANDLE) {
            materialsChanged = true;
            return nullptr;
        }
        bindings.push_back(
            {&submesh, material, pass->pipeline, pass->pipelineLayout, pass->descriptorSet, pass->shaderProgram.get()});
    }
    if (bindings.empty())
        return nullptr;

    // Allocate geometry only after the material textures are ready; waiting
    // for an upload must not recreate vertex/index buffers every frame.
    if (!PrepareGeometry(mesh, geometry, progressed))
        return nullptr;
    auto vbo = geometry->vertices->GetBuffer();
    auto ibo = geometry->indices->GetBuffer();

    // ── Scene UBO ────────────────────────────────────────────────────
    // Use identity model matrix; the mesh vertices are in local space
    // and the camera is positioned to look at the mesh bounds.
    glm::mat4 modelMat = glm::mat4(1.0f);

    UniformBufferObject sceneUBO{};
    sceneUBO.model = modelMat;
    sceneUBO.view = view;
    sceneUBO.proj = proj;
    sceneUBO.previousViewProj = proj * view;
    sceneUBO.inverseViewProj = glm::inverse(proj * view);
    SetPreviewDepthParams(sceneUBO, proj);

    // ── Lighting UBO ─────────────────────────────────────────────────
    ShaderLightingUBO lightingUBO{};
    memset(&lightingUBO, 0, sizeof(lightingUBO));
    lightingUBO.lightCounts = glm::ivec4(2, 0, 0, 0);
    lightingUBO.ambientColor = glm::vec4(0.08f, 0.08f, 0.09f, 1.0f);
    lightingUBO.ambientSkyColor = glm::vec4(0.20f, 0.22f, 0.26f, 0.55f);
    lightingUBO.ambientEquatorColor = glm::vec4(0.10f, 0.11f, 0.13f, 1.0f);
    lightingUBO.ambientGroundColor = glm::vec4(0.05f, 0.04f, 0.035f, 0.30f);
    lightingUBO.cameraPos = glm::vec4(cameraPos, 1.0f);

    lightingUBO.directionalLights[0].direction = glm::vec4(glm::normalize(glm::vec3(-0.7f, -1.0f, -0.5f)), 0.0f);
    lightingUBO.directionalLights[0].color = glm::vec4(1.8f, 1.71f, 1.62f, 1.8f);
    lightingUBO.directionalLights[0].metadata =
        glm::uvec4(~0u, static_cast<uint32_t>(LightInfluenceDomain::Geometry), 0u, 0u);

    lightingUBO.directionalLights[1].direction = glm::vec4(glm::normalize(glm::vec3(0.5f, 0.3f, -0.7f)), 0.0f);
    lightingUBO.directionalLights[1].color = glm::vec4(0.36f, 0.42f, 0.51f, 0.6f);
    lightingUBO.directionalLights[1].metadata =
        glm::uvec4(~0u, static_cast<uint32_t>(LightInfluenceDomain::Geometry), 0u, 0u);

    // ── Engine globals UBO ───────────────────────────────────────────
    EngineGlobalsUBO globalsUBO{};
    globalsUBO.projectionParams = sceneUBO.projectionParams;
    globalsUBO.zBufferParams = sceneUBO.zBufferParams;
    globalsUBO.screenParams =
        glm::vec4(static_cast<float>(renderSize), static_cast<float>(renderSize), 1.0f / renderSize, 1.0f / renderSize);
    globalsUBO.worldSpaceCameraPos = glm::vec4(cameraPos, 1.0f);

    const VkBuffer sceneUBOBuf = m_previewSceneUbo->GetBuffer();
    const VkBuffer lightingUBOBuf = m_previewLightingUbo->GetBuffer();
    const VkBuffer globalsUBOBuf = m_previewGlobalsUbo->GetBuffer();
    const VkBuffer instanceSSBOBuf = m_previewInstanceBuffer->GetBuffer();

    VkDescriptorSet shadowDesc = VK_NULL_HANDLE;
    VkDescriptorSet globalsDesc = VK_NULL_HANDLE;
    const bool requiresPerViewSet = std::any_of(bindings.begin(), bindings.end(), [](const auto &binding) {
        return binding.program && binding.program->HasDeclaredDescriptorSet(1);
    });
    const bool requiresGlobalsSet = std::any_of(bindings.begin(), bindings.end(), [](const auto &binding) {
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
        if (shadowDesc == VK_NULL_HANDLE)
            return nullptr;
    }

    if (requiresGlobalsSet) {
        globalsDesc = m_previewGlobalsSet;
    }

    // ── Record command buffer ────────────────────────────────────────
    const bool multisampled = m_sampleCount != VK_SAMPLE_COUNT_1_BIT;
    auto output =
        m_display->Prepare(multisampled ? m_resolveColor.GetView() : m_msaaColor.GetView(), m_colorFormat, size);
    if (!output)
        return nullptr;
    constexpr VkDeviceSize transferBytes = sizeof(sceneUBO) + sizeof(lightingUBO) + sizeof(globalsUBO) +
                                           sizeof(modelMat) + sizeof(std::array<uint32_t, 4>) + sizeof(glm::mat4) +
                                           sizeof(GPUInstanceAuxData);
    PreviewCommands commands(*m_vkCore, transferBytes, m_gpuNanoseconds);
    const VkCommandBuffer cmd = commands.Get();
    if (cmd == VK_NULL_HANDLE)
        return nullptr;

    vkCmdUpdateBuffer(cmd, sceneUBOBuf, 0, sizeof(sceneUBO), &sceneUBO);
    vkCmdUpdateBuffer(cmd, lightingUBOBuf, 0, sizeof(lightingUBO), &lightingUBO);
    vkCmdUpdateBuffer(cmd, globalsUBOBuf, 0, sizeof(globalsUBO), &globalsUBO);
    vkCmdUpdateBuffer(cmd, instanceSSBOBuf, 0, sizeof(modelMat), &modelMat);
    const std::array<uint32_t, 4> emptySkinInstance{};
    const glm::mat4 identityBone(1.0f);
    vkCmdUpdateBuffer(cmd, m_previewSkinInstanceBuffer->GetBuffer(), 0, sizeof(emptySkinInstance),
                      emptySkinInstance.data());
    vkCmdUpdateBuffer(cmd, m_previewSkinPaletteBuffer->GetBuffer(), 0, sizeof(identityBone), &identityBone);
    GPUInstanceAuxData instanceAux{};
    instanceAux.previousModel = modelMat;
    instanceAux.layerMask = ~0u;
    vkCmdUpdateBuffer(cmd, m_previewInstanceAuxBuffer->GetBuffer(), 0, sizeof(instanceAux), &instanceAux);

    // Barrier: make UBO writes visible
    VkMemoryBarrier uboBarrier{};
    uboBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    uboBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    uboBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &uboBarrier,
                         0, nullptr, 0, nullptr);

    // ── Begin the RHI-selected graphics attachment path ─────────────
    VkClearValue clearValues[2];
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
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

    VkViewport viewport{};
    viewport.width = static_cast<float>(renderSize);
    viewport.height = static_cast<float>(renderSize);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = {static_cast<uint32_t>(renderSize), static_cast<uint32_t>(renderSize)};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Bind mesh geometry
    VkBuffer vboBuf = vbo->GetBuffer();
    VkDeviceSize offsets[] = {0};
    commands.ReadGeometry(*vbo, *ibo);
    vkCmdBindVertexBuffers(cmd, 0, 1, &vboBuf, offsets);
    vkCmdBindIndexBuffer(cmd, ibo->GetBuffer(), 0, geometry->indexType);

    // Push constants
    struct PushConstants
    {
        glm::mat4 model;
        glm::mat4 normalMat;
    };
    PushConstants pushData{};
    pushData.model = modelMat;
    pushData.normalMat = glm::transpose(glm::inverse(modelMat));

    // ── Draw each submesh ────────────────────────────────────────────
    bool anyDrawn = false;
    for (auto &b : bindings) {
        const auto *descriptor =
            m_vkCore->GetMaterialPipelineManager().GetDescriptorManager().FindLiveDescriptorSet(b.materialDescSet);
        if (!descriptor)
            continue;

        commands.ReadMaterial(*descriptor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, b.pipeline);

        vkdebug::CmdBindDescriptorSetsTracked("GPUMeshPreview.Set0", cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                              b.pipelineLayout, 0, 1, &b.materialDescSet, 0, nullptr);

        if (b.program->HasDeclaredDescriptorSet(1) && shadowDesc != VK_NULL_HANDLE) {
            vkdebug::CmdBindDescriptorSetsTracked("GPUMeshPreview.Set1", cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                  b.pipelineLayout, 1, 1, &shadowDesc, 0, nullptr);
        }

        if (b.program->HasDeclaredDescriptorSet(2) && globalsDesc != VK_NULL_HANDLE) {
            vkdebug::CmdBindDescriptorSetsTracked("GPUMeshPreview.Set2", cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                  b.pipelineLayout, 2, 1, &globalsDesc, 0, nullptr);
        }

        if (b.program->UsesBindlessTextureABI()) {
            auto &rhiDevice = m_vkCore->GetDeviceContext().GetRhiDevice();
            const auto bindlessBinding = rhiDevice.GetBindlessTextureTableBinding();
            const VkDescriptorSet bindlessSet = rhiDevice.Resolve(bindlessBinding.group);
            if (bindlessSet == VK_NULL_HANDLE) {
                static int missingBindlessSetErrorCount = 0;
                if (missingBindlessSetErrorCount++ < 8)
                    INXLOG_ERROR("Mesh preview pass skipped because the device-global texture table is unavailable");
                continue;
            } else {
                vkdebug::CmdBindDescriptorSetsTracked("GPUMeshPreview.BindlessTextures", cmd,
                                                      VK_PIPELINE_BIND_POINT_GRAPHICS, b.pipelineLayout,
                                                      ShaderProgram::BindlessTextureSet, 1, &bindlessSet, 0, nullptr);
                if (const auto *indices =
                        m_vkCore->GetMaterialPipelineManager().GetDescriptorManager().GetBindlessTextureIndices(
                            b.ownedMaterial->GetMaterialKey())) {
                    rhiDevice.MarkBindlessTexturesUsed(indices->empty() ? nullptr : indices->data(), indices->size());
                }
            }
        }

        if (b.program->HasVertexPushConstants())
            vkCmdPushConstants(cmd, b.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &pushData);

        vkCmdDrawIndexed(cmd, b.submesh->indexCount, 1, b.submesh->indexStart, 0, 0);
        anyDrawn = true;
    }

    m_dynamicRenderingCommands.end(cmd);
    m_msaaColorLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    if (m_depth.HasView())
        m_depthLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // ── MSAA resolve + display ──────────────────────────────────────
    if (m_sampleCount != VK_SAMPLE_COUNT_1_BIT) {
        vkrender::TransitionTrackedImageLayout(cmd, m_msaaColor.GetImage(), VK_IMAGE_ASPECT_COLOR_BIT,
                                               m_msaaColorLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        vkrender::TransitionTrackedImageLayout(cmd, m_resolveColor.GetImage(), VK_IMAGE_ASPECT_COLOR_BIT,
                                               m_resolveColorLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkImageResolve resolveRegion{};
        resolveRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        resolveRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        resolveRegion.extent = {static_cast<uint32_t>(renderSize), static_cast<uint32_t>(renderSize), 1};
        vkCmdResolveImage(cmd, m_msaaColor.GetImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_resolveColor.GetImage(),
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &resolveRegion);
    }

    m_display->Record(cmd, *output, multisampled ? m_resolveColor.GetImage() : m_msaaColor.GetImage(),
                      multisampled ? m_resolveColorLayout : m_msaaColorLayout, size);

    auto vertexLease = std::shared_ptr<vk::VkBufferHandle>(std::move(vbo));
    auto indexLease = std::shared_ptr<vk::VkBufferHandle>(std::move(ibo));
    std::vector<std::shared_ptr<InxMaterial>> materialLeases;
    materialLeases.reserve(bindings.size());
    for (const auto &binding : bindings)
        materialLeases.push_back(binding.ownedMaterial);
    m_activeSubmission =
        commands.Submit([vertexLease = std::move(vertexLease), indexLease = std::move(indexLease),
                         materialLeases = std::move(materialLeases), finalView = output->texture]() mutable {
            materialLeases.clear();
            indexLease.reset();
            vertexLease.reset();
        });
    // A descriptor can still be retired between binding preparation and
    // command recording. Finish the submission so command-buffer and resource
    // ownership remain valid, but let the caller retry instead of publishing
    // the transparent clear as a successful thumbnail.
    output->submission = m_activeSubmission;
    progressed = true;
    return anyDrawn ? output : nullptr;
}

// ============================================================================
// Resource management (mirrors GPUMaterialPreview)
// ============================================================================

bool GPUMeshPreview::EnsureResources(int size, bool &progressed)
{
    auto &mpm = m_vkCore->GetMaterialPipelineManager();
    VkFormat colorFormat = mpm.GetColorFormat();
    VkFormat depthFormat = mpm.GetDepthFormat();
    VkSampleCountFlagBits sampleCount = mpm.GetSampleCount();
    const auto dynamicCommands = rhi::ResolveDynamicRenderingCommands(m_vkCore->GetDevice());
    if (!m_vkCore->GetDeviceContext().GetRhiDevice().GetVulkanFeatures().dynamicRendering.enabled ||
        !dynamicCommands.IsValid()) {
        INXLOG_ERROR("GPUMeshPreview: dynamic rendering is required for mesh previews");
        return false;
    }

    const bool renderConfigChanged =
        (m_currentSize != 0) &&
        (colorFormat != m_colorFormat || depthFormat != m_depthFormat || sampleCount != m_sampleCount);

    if (renderConfigChanged) {
        DestroyImGuiDisplayDescriptor();
        DestroyAttachments();
        progressed = true;
    }

    m_colorFormat = colorFormat;
    m_depthFormat = depthFormat;
    m_sampleCount = sampleCount;
    m_dynamicRenderingCommands = dynamicCommands;

    if (m_currentSize != size) {
        DestroyImGuiDisplayDescriptor();
        DestroyAttachments();
        CreateAttachments(size);
        m_currentSize = size;
        progressed = true;
        if (m_msaaColor.IsValid())
            PublishRenderView();
    }

    return m_msaaColor.IsValid() && EnsureViewResources(progressed);
}

bool GPUMeshPreview::EnsureViewResources(bool &progressed)
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
        throw std::runtime_error("Failed to allocate isolated mesh-preview view buffers");

    const VkDevice device = m_vkCore->GetDevice();
    const VkDescriptorSetLayout layout = m_vkCore->GetGlobalsDescSetLayout();
    if (layout == VK_NULL_HANDLE)
        throw std::logic_error("Mesh preview requires the renderer globals descriptor layout");
    m_previewGlobalsLease = m_vkCore->GetDeviceContext().GetRhiDevice().GetDescriptorManager().Allocate(
        layout, vk::DescriptorArena::ViewPersistent);
    if (!m_previewGlobalsLease.IsValid())
        throw std::runtime_error("Failed to allocate isolated mesh-preview globals descriptor set");
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

void GPUMeshPreview::DestroyViewResources()
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

void GPUMeshPreview::CreateAttachments(int size)
{
    VkDevice device = m_vkCore->GetDevice();
    VmaAllocator allocator = m_vkCore->GetDeviceContext().GetVmaAllocator();
    uint32_t w = static_cast<uint32_t>(size);
    uint32_t h = static_cast<uint32_t>(size);

    m_msaaColor.Create(allocator, device, w, h, m_colorFormat, VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                           (m_sampleCount == VK_SAMPLE_COUNT_1_BIT ? VK_IMAGE_USAGE_SAMPLED_BIT : 0),
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_sampleCount);
    m_msaaColor.CreateView(m_colorFormat, VK_IMAGE_ASPECT_COLOR_BIT);

    if (m_sampleCount != VK_SAMPLE_COUNT_1_BIT) {
        m_resolveColor.Create(allocator, device, w, h, m_colorFormat, VK_IMAGE_TILING_OPTIMAL,
                              VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                  VK_IMAGE_USAGE_SAMPLED_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_resolveColor.CreateView(m_colorFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    if (m_depthFormat != VK_FORMAT_UNDEFINED) {
        m_depth.Create(allocator, device, w, h, m_depthFormat, VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_sampleCount);
        m_depth.CreateView(m_depthFormat, rhi::ToVkImageAspectMask(m_depthFormat));
    }
}

void GPUMeshPreview::DestroyAttachments()
{
    UnpublishRenderView();
    RetirePreviewAttachments(*m_vkCore, m_msaaColor, m_resolveColor, m_depth);
    m_msaaColorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    m_resolveColorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    m_depthLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    m_currentSize = 0;
    m_displayImageShaderReady = false;
}

void GPUMeshPreview::PublishRenderView()
{
    if (!m_vkCore || !m_msaaColor.IsValid())
        return;
    auto &device = m_vkCore->GetDeviceContext().GetRhiDevice();
    if (m_renderView.color.IsValid())
        device.Release(m_renderView.color);
    if (m_renderView.depth.IsValid())
        device.Release(m_renderView.depth);

    const VkImageView displayView =
        m_sampleCount != VK_SAMPLE_COUNT_1_BIT ? m_resolveColor.GetView() : m_msaaColor.GetView();
    m_renderView.device = m_vkCore->GetDeviceContext().GetDeviceId();
    m_renderView.width = static_cast<uint32_t>(m_currentSize);
    m_renderView.height = static_cast<uint32_t>(m_currentSize);
    m_renderView.colorFormat = rhi::FromVkFormat(m_colorFormat);
    m_renderView.depthFormat = rhi::FromVkFormat(m_depthFormat);
    m_renderView.samples = rhi::FromVkSampleCount(m_sampleCount);
    m_renderView.color = device.RegisterTextureView(displayView);
    m_renderView.depth = m_depth.HasView() ? device.RegisterTextureView(m_depth.GetView()) : rhi::TextureViewHandle{};
    ++m_renderView.revision;
}

void GPUMeshPreview::UnpublishRenderView()
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

void GPUMeshPreview::DestroyImGuiDisplayDescriptor()
{
    if (!m_vkCore)
        return;
    VkDevice device = m_vkCore->GetDevice();
    // In-flight GPU frames (and UI code holding the previously returned
    // ImTextureID) may still reference this descriptor for a few frames.
    // Freeing it immediately makes the ImGui backend bind a destroyed
    // VkDescriptorSet — validation errors and intermittent crashes — so both
    // the descriptor and its sampler retire through the submission retirement queue.
    if (m_displayDescriptorSet != VK_NULL_HANDLE || m_displaySampler != VK_NULL_HANDLE) {
        VkDescriptorSet retiredSet = m_displayDescriptorSet;
        VkSampler retiredSampler = m_displaySampler;
        m_vkCore->RetireGpuResource([device, retiredSet, retiredSampler]() {
            if (retiredSet != VK_NULL_HANDLE && ImGui::GetCurrentContext() != nullptr &&
                ImGui::GetIO().BackendRendererUserData != nullptr)
                ImGui_ImplVulkan_RemoveTexture(retiredSet);
            if (retiredSampler != VK_NULL_HANDLE)
                vkDestroySampler(device, retiredSampler, nullptr);
        });
        m_displayDescriptorSet = VK_NULL_HANDLE;
        m_displaySampler = VK_NULL_HANDLE;
    }
    m_displayImageShaderReady = false;
}

void GPUMeshPreview::EnsureImGuiDisplayDescriptor()
{
    if (m_displayDescriptorSet != VK_NULL_HANDLE)
        return;

    VkDevice device = m_vkCore->GetDevice();
    VkImageView displayView = VK_NULL_HANDLE;
    if (m_sampleCount != VK_SAMPLE_COUNT_1_BIT)
        displayView = m_resolveColor.GetView();
    else
        displayView = m_msaaColor.GetView();
    if (displayView == VK_NULL_HANDLE)
        return;

    auto samplerInfo = vkrender::MakeLinearClampSamplerInfo();
    if (vkCreateSampler(device, &samplerInfo, nullptr, &m_displaySampler) != VK_SUCCESS)
        return;

    m_displayDescriptorSet =
        ImGui_ImplVulkan_AddTexture(m_displaySampler, displayView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

uint64_t GPUMeshPreview::RenderToImGuiTextureCamera(const InxMesh &mesh,
                                                    const std::vector<std::shared_ptr<InxMaterial>> &materials,
                                                    int size, const glm::mat4 &view, const glm::mat4 &proj,
                                                    const glm::vec3 &cameraPos,
                                                    std::shared_ptr<PreviewGeometryUpload> &geometry, bool &progressed,
                                                    bool cloneMaterials, const std::vector<glm::mat4> *bonePalette)
{
    if (!m_vkCore || size <= 0)
        return 0;
    if (m_activeSubmission && !m_activeSubmission->IsComplete()) {
        return 0;
    }
    if (m_activeSubmission && m_activeSubmission->HasExecutionTiming())
        m_gpuNanoseconds = m_activeSubmission->GetExecutionNanoseconds();
    m_activeSubmission.reset();

    const auto skin = bonePalette ? mesh.GetSkinnedData() : nullptr;
    const auto &vertices = skin ? skin->baseVertices : mesh.GetVertices();
    const auto &indices = skin ? skin->indices : mesh.GetIndices();
    const auto &submeshes = skin ? skin->subMeshes : mesh.GetSubMeshes();
    if (vertices.empty() || indices.empty())
        return 0;

    const int renderSize = size;
    if (!EnsureResources(renderSize, progressed))
        return 0;

    if (!PrepareGeometry(mesh, geometry, progressed, skin))
        return 0;
    auto &rm = m_vkCore->GetResourceManager();
    auto vbo = geometry->vertices->GetBuffer();
    auto ibo = geometry->indices->GetBuffer();

    const VkDeviceSize paletteBytes = bonePalette ? bonePalette->size() * sizeof(glm::mat4) : sizeof(glm::mat4);
    if (m_previewSkinPaletteBuffer->GetSize() < paletteBytes) {
        // The prior preview submission completed above; no shader still reads
        // this descriptor. Geometry and GUI image consumers have separate leases.
        m_previewSkinPaletteBuffer = rm.CreateStorageBuffer(paletteBytes, false);
        if (!m_previewSkinPaletteBuffer)
            throw std::runtime_error("Failed to allocate animation preview palette");
        VkDescriptorBufferInfo buffer{m_previewSkinPaletteBuffer->GetBuffer(), 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = m_previewGlobalsSet;
        write.dstBinding = 3;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buffer;
        vkUpdateDescriptorSets(m_vkCore->GetDevice(), 1, &write, 0, nullptr);
    }

    auto defaultMat = AssetRegistry::Instance().GetBuiltinMaterial("DefaultLit");
    struct SubmeshBinding
    {
        const SubMesh *submesh = nullptr;
        std::shared_ptr<InxMaterial> ownedMaterial;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSet materialDescSet = VK_NULL_HANDLE;
        const ShaderProgram *program = nullptr;
    };
    std::vector<SubmeshBinding> bindings;
    bindings.reserve(submeshes.size());
    for (const SubMesh &sm : submeshes) {
        if (sm.indexCount == 0)
            continue;
        std::shared_ptr<InxMaterial> srcMat;
        if (sm.materialSlot < materials.size() && materials[sm.materialSlot])
            srcMat = materials[sm.materialSlot];
        else
            srcMat = defaultMat;
        if (!srcMat)
            continue;
        std::shared_ptr<InxMaterial> previewMat = cloneMaterials ? srcMat->Clone() : srcMat;
        if (!previewMat)
            continue;
        if (cloneMaterials) {
            previewMat->ClearAllPassPipelines();
        }
        bool pipelineReady = m_vkCore->RefreshPreviewMaterialPipeline(previewMat, previewMat->GetVertShaderName(),
                                                                      previewMat->GetFragShaderName(), false);
        if (!pipelineReady && srcMat != defaultMat && defaultMat) {
            previewMat = cloneMaterials ? defaultMat->Clone() : defaultMat;
            if (previewMat && cloneMaterials)
                previewMat->ClearAllPassPipelines();
            pipelineReady =
                previewMat && m_vkCore->RefreshPreviewMaterialPipeline(previewMat, previewMat->GetVertShaderName(),
                                                                       previewMat->GetFragShaderName(), false);
        }
        if (!pipelineReady || !previewMat)
            continue;
        MaterialPassRenderData *rd = m_vkCore->GetOrCreatePreviewMaterialPass(previewMat);
        if (!rd || !rd->isValid || rd->descriptorSet == VK_NULL_HANDLE)
            continue;
        previewMat->SetPassPipeline(ShaderCompileTarget::Forward, rd->pipeline);
        previewMat->SetPassPipelineLayout(ShaderCompileTarget::Forward, rd->pipelineLayout);
        previewMat->SetPassDescriptorSet(ShaderCompileTarget::Forward, rd->descriptorSet);
        previewMat->SetPassShaderProgram(ShaderCompileTarget::Forward, rd->shaderProgram);
        SubmeshBinding b;
        b.submesh = &sm;
        b.ownedMaterial = previewMat;
        b.pipeline = rd->pipeline;
        b.pipelineLayout = rd->pipelineLayout;
        b.materialDescSet = rd->descriptorSet;
        b.program = rd->shaderProgram.get();
        bindings.push_back(std::move(b));
    }
    if (bindings.empty())
        return 0;

    for (auto &b : bindings) {
        m_vkCore->UpdateMaterialUBO(*b.ownedMaterial);
        if (m_vkCore->GetMaterialPipelineManager().HasPendingTextureProperties(b.ownedMaterial->GetMaterialKey()))
            return 0;
    }

    bindings.erase(std::remove_if(bindings.begin(), bindings.end(),
                                  [&](SubmeshBinding &binding) {
                                      MaterialPassRenderData *rd =
                                          m_vkCore->GetOrCreatePreviewMaterialPass(binding.ownedMaterial);
                                      if (!rd || !rd->isValid || rd->pipeline == VK_NULL_HANDLE ||
                                          rd->pipelineLayout == VK_NULL_HANDLE || rd->descriptorSet == VK_NULL_HANDLE ||
                                          !rd->shaderProgram)
                                          return true;
                                      binding.pipeline = rd->pipeline;
                                      binding.pipelineLayout = rd->pipelineLayout;
                                      binding.materialDescSet = rd->descriptorSet;
                                      binding.program = rd->shaderProgram.get();
                                      return false;
                                  }),
                   bindings.end());
    if (bindings.empty())
        return 0;

    const glm::mat4 modelMat = glm::mat4(1.0f);
    UniformBufferObject sceneUBO{};
    sceneUBO.model = modelMat;
    sceneUBO.view = view;
    sceneUBO.proj = proj;
    sceneUBO.previousViewProj = proj * view;
    sceneUBO.inverseViewProj = glm::inverse(proj * view);
    SetPreviewDepthParams(sceneUBO, proj);

    ShaderLightingUBO lightingUBO{};
    memset(&lightingUBO, 0, sizeof(lightingUBO));
    lightingUBO.lightCounts = glm::ivec4(2, 0, 0, 0);
    lightingUBO.ambientColor = glm::vec4(0.08f, 0.08f, 0.09f, 1.0f);
    lightingUBO.ambientSkyColor = glm::vec4(0.20f, 0.22f, 0.26f, 0.55f);
    lightingUBO.ambientEquatorColor = glm::vec4(0.10f, 0.11f, 0.13f, 1.0f);
    lightingUBO.ambientGroundColor = glm::vec4(0.05f, 0.04f, 0.035f, 0.30f);
    lightingUBO.cameraPos = glm::vec4(cameraPos, 1.0f);
    lightingUBO.directionalLights[0].direction = glm::vec4(glm::normalize(glm::vec3(-0.7f, -1.0f, -0.5f)), 0.0f);
    lightingUBO.directionalLights[0].color = glm::vec4(1.8f, 1.71f, 1.62f, 1.8f);
    lightingUBO.directionalLights[0].metadata =
        glm::uvec4(~0u, static_cast<uint32_t>(LightInfluenceDomain::Geometry), 0u, 0u);
    lightingUBO.directionalLights[1].direction = glm::vec4(glm::normalize(glm::vec3(0.5f, 0.3f, -0.7f)), 0.0f);
    lightingUBO.directionalLights[1].color = glm::vec4(0.36f, 0.42f, 0.51f, 0.6f);
    lightingUBO.directionalLights[1].metadata =
        glm::uvec4(~0u, static_cast<uint32_t>(LightInfluenceDomain::Geometry), 0u, 0u);

    EngineGlobalsUBO globalsUBO{};
    globalsUBO.projectionParams = sceneUBO.projectionParams;
    globalsUBO.zBufferParams = sceneUBO.zBufferParams;
    globalsUBO.screenParams =
        glm::vec4(static_cast<float>(renderSize), static_cast<float>(renderSize), 1.0f / renderSize, 1.0f / renderSize);
    globalsUBO.worldSpaceCameraPos = glm::vec4(cameraPos, 1.0f);

    const VkBuffer sceneUBOBuf = m_previewSceneUbo->GetBuffer();
    const VkBuffer lightingUBOBuf = m_previewLightingUbo->GetBuffer();
    const VkBuffer globalsUBOBuf = m_previewGlobalsUbo->GetBuffer();
    const VkBuffer instanceSSBOBuf = m_previewInstanceBuffer->GetBuffer();

    VkDescriptorSet shadowDesc = VK_NULL_HANDLE;
    VkDescriptorSet globalsDesc = VK_NULL_HANDLE;
    const bool requiresPerViewSet = std::any_of(bindings.begin(), bindings.end(), [](const auto &binding) {
        return binding.program && binding.program->HasDeclaredDescriptorSet(1);
    });
    const bool requiresGlobalsSet = std::any_of(bindings.begin(), bindings.end(), [](const auto &binding) {
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
        if (shadowDesc == VK_NULL_HANDLE)
            return 0;
    }
    if (requiresGlobalsSet) {
        globalsDesc = m_previewGlobalsSet;
    }

    const VkDeviceSize transferBytes = sizeof(sceneUBO) + sizeof(lightingUBO) + sizeof(globalsUBO) + sizeof(modelMat) +
                                       sizeof(std::array<uint32_t, 4>) + paletteBytes + sizeof(GPUInstanceAuxData);
    PreviewCommands commands(*m_vkCore, transferBytes, m_gpuNanoseconds);
    const VkCommandBuffer cmd = commands.Get();
    if (cmd == VK_NULL_HANDLE)
        return 0;

    VkImage displayImage =
        (m_sampleCount != VK_SAMPLE_COUNT_1_BIT) ? m_resolveColor.GetImage() : m_msaaColor.GetImage();
    if (m_sampleCount != VK_SAMPLE_COUNT_1_BIT) {
        vkrender::TransitionTrackedImageLayout(cmd, displayImage, VK_IMAGE_ASPECT_COLOR_BIT, m_resolveColorLayout,
                                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    }

    vkCmdUpdateBuffer(cmd, sceneUBOBuf, 0, sizeof(sceneUBO), &sceneUBO);
    vkCmdUpdateBuffer(cmd, lightingUBOBuf, 0, sizeof(lightingUBO), &lightingUBO);
    vkCmdUpdateBuffer(cmd, globalsUBOBuf, 0, sizeof(globalsUBO), &globalsUBO);
    vkCmdUpdateBuffer(cmd, instanceSSBOBuf, 0, sizeof(modelMat), &modelMat);
    const std::array<uint32_t, 4> skinInstance{0u, bonePalette ? static_cast<uint32_t>(bonePalette->size()) : 0u,
                                               bonePalette ? 1u : 0u, 0u};
    const glm::mat4 identityBone(1.0f);
    vkCmdUpdateBuffer(cmd, m_previewSkinInstanceBuffer->GetBuffer(), 0, sizeof(skinInstance), skinInstance.data());
    const auto *paletteData =
        reinterpret_cast<const unsigned char *>(bonePalette ? bonePalette->data() : &identityBone);
    for (VkDeviceSize offset = 0; offset < paletteBytes; offset += 65536) {
        const auto bytes = std::min<VkDeviceSize>(65536, paletteBytes - offset);
        vkCmdUpdateBuffer(cmd, m_previewSkinPaletteBuffer->GetBuffer(), offset, bytes, paletteData + offset);
    }
    GPUInstanceAuxData instanceAux{};
    instanceAux.previousModel = modelMat;
    instanceAux.layerMask = ~0u;
    vkCmdUpdateBuffer(cmd, m_previewInstanceAuxBuffer->GetBuffer(), 0, sizeof(instanceAux), &instanceAux);

    VkMemoryBarrier uboBarrier{};
    uboBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    uboBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    uboBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &uboBarrier,
                         0, nullptr, 0, nullptr);

    VkClearValue clearValues[2];
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
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

    VkViewport viewport{};
    viewport.width = static_cast<float>(renderSize);
    viewport.height = static_cast<float>(renderSize);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor{};
    scissor.extent = {static_cast<uint32_t>(renderSize), static_cast<uint32_t>(renderSize)};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    VkBuffer vboBuf = vbo->GetBuffer();
    VkDeviceSize offsets[] = {0};
    commands.ReadGeometry(*vbo, *ibo);
    vkCmdBindVertexBuffers(cmd, 0, 1, &vboBuf, offsets);
    vkCmdBindIndexBuffer(cmd, ibo->GetBuffer(), 0, geometry->indexType);

    struct PushConstants
    {
        glm::mat4 model;
        glm::mat4 normalMat;
    };
    PushConstants pushData{};
    pushData.model = modelMat;
    pushData.normalMat = glm::transpose(glm::inverse(modelMat));

    for (auto &b : bindings) {
        const auto *descriptor =
            m_vkCore->GetMaterialPipelineManager().GetDescriptorManager().FindLiveDescriptorSet(b.materialDescSet);
        if (!descriptor)
            continue;
        commands.ReadMaterial(*descriptor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, b.pipeline);
        vkdebug::CmdBindDescriptorSetsTracked("GPUMeshPreview.Live.Set0", cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                              b.pipelineLayout, 0, 1, &b.materialDescSet, 0, nullptr);
        if (b.program->HasDeclaredDescriptorSet(1) && shadowDesc != VK_NULL_HANDLE)
            vkdebug::CmdBindDescriptorSetsTracked("GPUMeshPreview.Live.Set1", cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                  b.pipelineLayout, 1, 1, &shadowDesc, 0, nullptr);
        if (b.program->HasDeclaredDescriptorSet(2) && globalsDesc != VK_NULL_HANDLE)
            vkdebug::CmdBindDescriptorSetsTracked("GPUMeshPreview.Live.Set2", cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                  b.pipelineLayout, 2, 1, &globalsDesc, 0, nullptr);
        if (b.program->UsesBindlessTextureABI()) {
            auto &rhiDevice = m_vkCore->GetDeviceContext().GetRhiDevice();
            const auto bindlessBinding = rhiDevice.GetBindlessTextureTableBinding();
            const VkDescriptorSet bindlessSet = rhiDevice.Resolve(bindlessBinding.group);
            if (bindlessSet == VK_NULL_HANDLE) {
                static int missingLiveBindlessSetErrorCount = 0;
                if (missingLiveBindlessSetErrorCount++ < 8)
                    INXLOG_ERROR(
                        "Live mesh preview pass skipped because the device-global texture table is unavailable");
                continue;
            } else {
                vkdebug::CmdBindDescriptorSetsTracked("GPUMeshPreview.Live.BindlessTextures", cmd,
                                                      VK_PIPELINE_BIND_POINT_GRAPHICS, b.pipelineLayout,
                                                      ShaderProgram::BindlessTextureSet, 1, &bindlessSet, 0, nullptr);
                if (const auto *indices =
                        m_vkCore->GetMaterialPipelineManager().GetDescriptorManager().GetBindlessTextureIndices(
                            b.ownedMaterial->GetMaterialKey())) {
                    rhiDevice.MarkBindlessTexturesUsed(indices->empty() ? nullptr : indices->data(), indices->size());
                }
            }
        }
        if (b.program->HasVertexPushConstants())
            vkCmdPushConstants(cmd, b.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &pushData);
        vkCmdDrawIndexed(cmd, b.submesh->indexCount, 1, b.submesh->indexStart, 0, 0);
    }
    m_dynamicRenderingCommands.end(cmd);
    m_msaaColorLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    if (m_depth.HasView())
        m_depthLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    if (m_sampleCount != VK_SAMPLE_COUNT_1_BIT) {
        vkrender::TransitionTrackedImageLayout(cmd, m_msaaColor.GetImage(), VK_IMAGE_ASPECT_COLOR_BIT,
                                               m_msaaColorLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkImageResolve resolveRegion{};
        resolveRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        resolveRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        resolveRegion.extent = {static_cast<uint32_t>(renderSize), static_cast<uint32_t>(renderSize), 1};
        vkCmdResolveImage(cmd, m_msaaColor.GetImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_resolveColor.GetImage(),
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &resolveRegion);
    }

    if (m_sampleCount != VK_SAMPLE_COUNT_1_BIT) {
        vkrender::TransitionTrackedImageLayout(cmd, displayImage, VK_IMAGE_ASPECT_COLOR_BIT, m_resolveColorLayout,
                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    } else {
        vkrender::TransitionTrackedImageLayout(cmd, displayImage, VK_IMAGE_ASPECT_COLOR_BIT, m_msaaColorLayout,
                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    auto vertexLease = std::shared_ptr<vk::VkBufferHandle>(std::move(vbo));
    auto indexLease = std::shared_ptr<vk::VkBufferHandle>(std::move(ibo));
    std::vector<std::shared_ptr<InxMaterial>> materialLeases;
    materialLeases.reserve(bindings.size());
    for (const auto &binding : bindings)
        materialLeases.push_back(binding.ownedMaterial);
    m_activeSubmission = commands.Submit([vertexLease = std::move(vertexLease), indexLease = std::move(indexLease),
                                          materialLeases = std::move(materialLeases)]() mutable {
        materialLeases.clear();
        indexLease.reset();
        vertexLease.reset();
    });
    m_displayImageShaderReady = true;
    progressed = true;
    EnsureImGuiDisplayDescriptor();
    return reinterpret_cast<uint64_t>(m_displayDescriptorSet);
}

uint64_t GPUMeshPreview::RenderAnimation(const std::shared_ptr<InxMesh> &mesh, const std::string &take, float seconds,
                                         int size, uint64_t dependencyRevision)
{
    if (!mesh || !std::isfinite(seconds) || seconds < 0.0f || size < 32 || size > 1024)
        throw std::invalid_argument("Animation preview requires a mesh, finite nonnegative time and size 32..1024");
    const auto &skin = mesh->GetSkinnedData();
    if (!skin || !skin->IsValid())
        throw std::invalid_argument("Animation preview requires a renderable skinned model");
    if (!skin->FindAnimation(take))
        throw std::invalid_argument("Animation preview clip is not present in the published model");
    const bool sourceChanged = mesh != m_animationMesh || mesh->GetGeneration() != m_animationGeneration ||
                               dependencyRevision != m_animationRevision || !m_animationMaterialLoads.IsCurrent();
    if (sourceChanged) {
        m_animationMesh = mesh;
        m_animationGeneration = mesh->GetGeneration();
        m_animationRevision = dependencyRevision;
        m_renderedSeconds = -1.0f;
        m_animationMaterials.clear();
        m_animationMaterialLoads.Clear();
        m_animationMaterialsReady = false;
    }
    const bool changed = !m_animationMaterialsReady;
    bool progressed = false;
    if (changed) {
        std::vector<std::shared_ptr<InxMaterial>> sources;
        if (!TryPrepareModelPreviewMaterials(*mesh, skin->subMeshes, m_animationMaterialLoads, sources, progressed))
            return 0;
        for (const auto &material : sources)
            m_animationMaterials.push_back(material->Clone());
        m_animationMaterialsReady = true;
    }
    if (take == m_renderedTake && seconds == m_renderedSeconds && size == m_renderedSize && GetDisplayTextureId())
        return GetDisplayTextureId();

    SkinnedSampleRequest request;
    request.takeName = take;
    request.timeSeconds = seconds;
    request.loop = false; // Scrubbing to duration must show the final pose.
    if (changed || take != m_framedTake) {
        auto firstPose = request;
        firstPose.timeSeconds = 0.0f;
        const auto firstPalette = skin->BuildGpuBonePalette(firstPose);
        m_animationFrameMin = mesh->GetBoundsMin();
        m_animationFrameMax = mesh->GetBoundsMax();
        glm::vec3 poseMin, poseMax;
        if (skin->ComputeSkinnedBounds(firstPalette, poseMin, poseMax)) {
            m_animationFrameMin = poseMin;
            m_animationFrameMax = poseMax;
        }
        m_framedTake = take;
    }
    // Keep the starting pose in view even when a clip starts away from the
    // bind pose. Framing stays fixed during playback, so root motion remains
    // visible without CPU skinning every frame.
    const auto camera = FitCameraToBounds(m_animationFrameMin, m_animationFrameMax, kMeshPreviewFovDeg);
    const auto palette = skin->BuildGpuBonePalette(request);
    const uint64_t texture =
        RenderToImGuiTextureCamera(*mesh, m_animationMaterials, size, camera.view, camera.proj, camera.cameraPos,
                                   m_animationGeometry, progressed, false, &palette);
    if (texture) {
        m_renderedTake = take;
        m_renderedSeconds = seconds;
        m_renderedSize = size;
    }
    // GPU backpressure retains the last image of this source, not another model.
    return m_renderedSeconds >= 0.0f ? GetDisplayTextureId() : 0;
}

} // namespace infernux
