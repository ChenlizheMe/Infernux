/**
 * @file GPUMeshPreview.cpp
 * @brief GPU mesh preview — renders an arbitrary InxMesh with per-submesh
 *        materials into a small offscreen framebuffer and reads back RGBA8
 *        pixels for editor thumbnails (model / prefab previews).
 */

#include "GPUMeshPreview.h"
#include "InxError.h"
#include <function/renderer/EngineGlobals.h>
#include <function/renderer/InxRenderStruct.h>
#include <function/renderer/InxVkCoreModular.h>
#include <function/renderer/MaterialPipelineManager.h>
#include <function/renderer/shader/ShaderProgram.h>
#include <function/renderer/vk/DescriptorBindTrace.h>
#include <function/renderer/vk/VkRenderUtils.h>
#include <function/renderer/vk/VkResourceManager.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/scene/LightingData.h>

#include <algorithm>
#include <cmath>
#include <core/log/InxLog.h>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace infernux
{

namespace
{
constexpr float kMeshPreviewFovDeg = 30.0f;
constexpr int kPreviewSupersampleFactor = 2;

/// @brief Downsample RGBA image from srcSize to dstSize using box filter.
void DownsampleRGBABox(const std::vector<unsigned char> &srcPixels, int srcSize, int dstSize,
                       std::vector<unsigned char> &dstPixels)
{
    if (srcSize <= 0 || dstSize <= 0 || srcPixels.empty()) {
        dstPixels.clear();
        return;
    }
    if (srcSize == dstSize) {
        dstPixels = srcPixels;
        return;
    }

    dstPixels.resize(static_cast<size_t>(dstSize) * dstSize * 4);
    const float scale = static_cast<float>(srcSize) / static_cast<float>(dstSize);

    for (int dy = 0; dy < dstSize; ++dy) {
        for (int dx = 0; dx < dstSize; ++dx) {
            const int sx0 = static_cast<int>(dx * scale);
            const int sy0 = static_cast<int>(dy * scale);
            const int sx1 = std::min(static_cast<int>((dx + 1) * scale), srcSize);
            const int sy1 = std::min(static_cast<int>((dy + 1) * scale), srcSize);
            float r = 0, g = 0, b = 0, a = 0;
            int count = 0;
            for (int sy = sy0; sy < sy1; ++sy) {
                for (int sx = sx0; sx < sx1; ++sx) {
                    const size_t idx = (static_cast<size_t>(sy) * srcSize + sx) * 4;
                    r += srcPixels[idx + 0];
                    g += srcPixels[idx + 1];
                    b += srcPixels[idx + 2];
                    a += srcPixels[idx + 3];
                    ++count;
                }
            }
            if (count > 0) {
                const float inv = 1.0f / count;
                const size_t dstIdx = (static_cast<size_t>(dy) * dstSize + dx) * 4;
                dstPixels[dstIdx + 0] = static_cast<unsigned char>(r * inv + 0.5f);
                dstPixels[dstIdx + 1] = static_cast<unsigned char>(g * inv + 0.5f);
                dstPixels[dstIdx + 2] = static_cast<unsigned char>(b * inv + 0.5f);
                dstPixels[dstIdx + 3] = static_cast<unsigned char>(a * inv + 0.5f);
            }
        }
    }
}

/// @brief Compute camera transform that fits a bounding box into the viewport.
struct FitCameraResult
{
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec3 cameraPos;
};

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
    const float distance = (radius / std::sin(halfFov)) * 1.15f;

    // Shift look-at target slightly upward so the model isn't bottom-heavy
    const glm::vec3 lookAt = center + glm::vec3(0.0f, maxExtent * 0.05f, 0.0f);
    const glm::vec3 cameraPos = lookAt - viewDir * distance;

    FitCameraResult result;
    result.cameraPos = cameraPos;
    result.view = glm::lookAt(cameraPos, lookAt, glm::vec3(0.0f, 1.0f, 0.0f));
    result.proj = glm::perspective(glm::radians(fovDeg), 1.0f, distance * 0.01f, distance * 4.0f);
    result.proj[1][1] *= -1.0f; // Vulkan Y-flip
    return result;
}

} // anonymous namespace

// ============================================================================
// Constructor / Destructor
// ============================================================================

GPUMeshPreview::GPUMeshPreview(InxVkCoreModular *vkCore) : m_vkCore(vkCore)
{
}

GPUMeshPreview::~GPUMeshPreview()
{
    if (!m_vkCore)
        return;
    VkDevice device = m_vkCore->GetDevice();
    vkDeviceWaitIdle(device);
    DestroyFramebuffer();
    if (m_renderPass != VK_NULL_HANDLE)
        vkDestroyRenderPass(device, m_renderPass, nullptr);
}

// ============================================================================
// RenderToPixels
// ============================================================================

bool GPUMeshPreview::RenderToPixels(const InxMesh &mesh, const std::vector<std::shared_ptr<InxMaterial>> &materials,
                                    int size, std::vector<unsigned char> &outPixels)
{
    if (!m_vkCore || size <= 0)
        return false;

    const auto &vertices = mesh.GetVertices();
    const auto &indices = mesh.GetIndices();
    if (vertices.empty() || indices.empty())
        return false;

    const int renderSize = std::max(size, size * kPreviewSupersampleFactor);

    // ── Upload mesh geometry to temporary GPU buffers ────────────────
    auto &rm = m_vkCore->GetResourceManager();
    auto vbo = rm.CreateVertexBuffer(vertices.data(), vertices.size() * sizeof(Vertex));
    auto ibo = rm.CreateIndexBuffer(indices.data(), indices.size() * sizeof(uint32_t));
    if (!vbo || !ibo)
        return false;

    // ── Auto-fit camera to mesh bounds ───────────────────────────────
    auto cam = FitCameraToBounds(mesh.GetBoundsMin(), mesh.GetBoundsMax(), kMeshPreviewFovDeg);

    // ── Prepare per-submesh material pipelines ───────────────────────
    // Get a default material for submeshes without an assigned material.
    auto defaultMat = AssetRegistry::Instance().GetBuiltinMaterial("DefaultLit");

    struct SubmeshBinding
    {
        const SubMesh *submesh = nullptr;
        std::shared_ptr<InxMaterial> ownedMaterial; // keep-alive
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSet materialDescSet = VK_NULL_HANDLE;
        ShaderProgram *program = nullptr;
    };

    std::vector<SubmeshBinding> bindings;
    bindings.reserve(mesh.GetSubMeshCount());

    for (uint32_t si = 0; si < mesh.GetSubMeshCount(); ++si) {
        const SubMesh &sm = mesh.GetSubMesh(si);
        if (sm.indexCount == 0)
            continue;

        // Pick material for this submesh
        std::shared_ptr<InxMaterial> srcMat;
        if (sm.materialSlot < materials.size() && materials[sm.materialSlot])
            srcMat = materials[sm.materialSlot];
        else
            srcMat = defaultMat;

        if (!srcMat)
            continue;

        // Clone + prepare pipeline (isolate from live scene materials)
        auto previewMat = srcMat->Clone();
        if (!previewMat)
            continue;
        previewMat->ClearAllPassPipelines();
        if (!m_vkCore->RefreshMaterialPipeline(previewMat, previewMat->GetVertShaderName(),
                                               previewMat->GetFragShaderName()))
            continue;

        MaterialRenderData *rd = m_vkCore->GetMaterialPipelineManager().GetRenderData(previewMat->GetMaterialKey());
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
        b.program = rd->shaderProgram;
        bindings.push_back(std::move(b));
    }

    if (bindings.empty())
        return false;

    if (!EnsureResources(renderSize))
        return false;

    // Update UBO data for each preview material
    for (auto &b : bindings)
        m_vkCore->UpdateMaterialUBO(*b.ownedMaterial);

    // ── Scene UBO ────────────────────────────────────────────────────
    // Use identity model matrix; the mesh vertices are in local space
    // and the camera is positioned to look at the mesh bounds.
    glm::mat4 modelMat = glm::mat4(1.0f);

    UniformBufferObject sceneUBO{};
    sceneUBO.model = modelMat;
    sceneUBO.view = cam.view;
    sceneUBO.proj = cam.proj;

    // ── Lighting UBO ─────────────────────────────────────────────────
    ShaderLightingUBO lightingUBO{};
    memset(&lightingUBO, 0, sizeof(lightingUBO));
    lightingUBO.lightCounts = glm::ivec4(2, 0, 0, 0);
    lightingUBO.ambientColor = glm::vec4(0.08f, 0.08f, 0.09f, 1.0f);
    lightingUBO.ambientSkyColor = glm::vec4(0.20f, 0.22f, 0.26f, 0.55f);
    lightingUBO.ambientEquatorColor = glm::vec4(0.10f, 0.11f, 0.13f, 1.0f);
    lightingUBO.ambientGroundColor = glm::vec4(0.05f, 0.04f, 0.035f, 0.30f);
    lightingUBO.cameraPos = glm::vec4(cam.cameraPos, 1.0f);

    lightingUBO.directionalLights[0].direction = glm::vec4(glm::normalize(glm::vec3(-0.7f, -1.0f, -0.5f)), 0.0f);
    lightingUBO.directionalLights[0].color = glm::vec4(1.8f, 1.71f, 1.62f, 1.8f);

    lightingUBO.directionalLights[1].direction = glm::vec4(glm::normalize(glm::vec3(0.5f, 0.3f, -0.7f)), 0.0f);
    lightingUBO.directionalLights[1].color = glm::vec4(0.36f, 0.42f, 0.51f, 0.6f);

    // ── Engine globals UBO ───────────────────────────────────────────
    EngineGlobalsUBO globalsUBO{};
    memset(&globalsUBO, 0, sizeof(globalsUBO));
    globalsUBO.screenParams =
        glm::vec4(static_cast<float>(renderSize), static_cast<float>(renderSize), 1.0f / renderSize, 1.0f / renderSize);
    globalsUBO.worldSpaceCameraPos = glm::vec4(cam.cameraPos, 1.0f);

    // ── Buffer indexing (same rationale as GPUMaterialPreview) ────────
    const uint32_t frameIndex =
        m_vkCore->GetSwapchain().GetCurrentFrame() % std::max(1u, m_vkCore->GetMaxFramesInFlight());

    VkBuffer sceneUBOBuf = m_vkCore->GetUniformBuffer(0);
    VkBuffer lightingUBOBuf = m_vkCore->GetLightingUBO(0);
    VkBuffer globalsUBOBuf = m_vkCore->GetGlobalsBuffer(frameIndex);
    VkBuffer instanceSSBOBuf = m_vkCore->GetInstanceSSBO(frameIndex);

    if (sceneUBOBuf == VK_NULL_HANDLE || lightingUBOBuf == VK_NULL_HANDLE)
        return false;

    // Resolve optional descriptor sets from the first binding's shader
    ShaderProgram *primaryProgram = bindings.front().program;
    VkDescriptorSet shadowDesc = VK_NULL_HANDLE;
    VkDescriptorSet globalsDesc = VK_NULL_HANDLE;

    if (primaryProgram->HasDeclaredDescriptorSet(1)) {
        shadowDesc = m_vkCore->GetActiveShadowDescriptorSet();
        if (shadowDesc == VK_NULL_HANDLE) {
            if (m_fallbackShadowDescSet == VK_NULL_HANDLE)
                m_fallbackShadowDescSet = m_vkCore->AllocatePerViewDescriptorSet();
            shadowDesc = m_fallbackShadowDescSet;
        }
        if (shadowDesc == VK_NULL_HANDLE)
            return false;
    }

    if (primaryProgram->HasDeclaredDescriptorSet(2)) {
        if (globalsUBOBuf == VK_NULL_HANDLE || instanceSSBOBuf == VK_NULL_HANDLE)
            return false;
        globalsDesc = m_vkCore->GetCurrentGlobalsDescSet();
        if (globalsDesc == VK_NULL_HANDLE)
            return false;
    }

    // Write identity instance matrix at slot 0
    if (instanceSSBOBuf != VK_NULL_HANDLE) {
        if (!m_vkCore->WriteInstanceMatrix(frameIndex, 0, modelMat))
            return false;
    }

    // ── Record command buffer ────────────────────────────────────────
    VkCommandBuffer cmd = m_vkCore->BeginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE)
        return false;

    vkCmdUpdateBuffer(cmd, sceneUBOBuf, 0, sizeof(sceneUBO), &sceneUBO);
    vkCmdUpdateBuffer(cmd, lightingUBOBuf, 0, sizeof(lightingUBO), &lightingUBO);
    if (globalsUBOBuf != VK_NULL_HANDLE)
        vkCmdUpdateBuffer(cmd, globalsUBOBuf, 0, sizeof(globalsUBO), &globalsUBO);

    // Barrier: make UBO writes visible
    VkMemoryBarrier uboBarrier{};
    uboBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    uboBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    uboBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &uboBarrier,
                         0, nullptr, 0, nullptr);

    // ── Begin render pass ────────────────────────────────────────────
    VkClearValue clearValues[2];
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpBegin{};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass = m_renderPass;
    rpBegin.framebuffer = m_framebuffer;
    rpBegin.renderArea.offset = {0, 0};
    rpBegin.renderArea.extent = {static_cast<uint32_t>(renderSize), static_cast<uint32_t>(renderSize)};
    rpBegin.clearValueCount = 2;
    rpBegin.pClearValues = clearValues;

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

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
    vkCmdBindVertexBuffers(cmd, 0, 1, &vboBuf, offsets);
    vkCmdBindIndexBuffer(cmd, ibo->GetBuffer(), 0, VK_INDEX_TYPE_UINT32);

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
    for (auto &b : bindings) {
        if (!m_vkCore->GetMaterialPipelineManager().IsDescriptorSetLive(b.materialDescSet))
            continue;

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

        vkCmdPushConstants(cmd, b.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &pushData);

        vkCmdDrawIndexed(cmd, b.submesh->indexCount, 1, b.submesh->indexStart, 0, 0);
    }

    vkCmdEndRenderPass(cmd);

    // ── MSAA resolve + readback ──────────────────────────────────────
    if (m_sampleCount != VK_SAMPLE_COUNT_1_BIT) {
        VkImageMemoryBarrier msaaBarrier = vkrender::MakeImageBarrier(
            m_msaaColor.GetImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &msaaBarrier);

        VkImageMemoryBarrier resolveBarrier = vkrender::MakeImageBarrier(
            m_resolveColor.GetImage(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &resolveBarrier);

        VkImageResolve resolveRegion{};
        resolveRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        resolveRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        resolveRegion.extent = {static_cast<uint32_t>(renderSize), static_cast<uint32_t>(renderSize), 1};
        vkCmdResolveImage(cmd, m_msaaColor.GetImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_resolveColor.GetImage(),
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &resolveRegion);

        VkImageMemoryBarrier readbackBarrier = vkrender::MakeImageBarrier(
            m_resolveColor.GetImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &readbackBarrier);
    } else {
        VkImageMemoryBarrier barrier = vkrender::MakeImageBarrier(
            m_msaaColor.GetImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);
    }

    VkImage srcImage = (m_sampleCount != VK_SAMPLE_COUNT_1_BIT) ? m_resolveColor.GetImage() : m_msaaColor.GetImage();

    VkBufferImageCopy copyRegion{};
    copyRegion.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copyRegion.imageExtent = {static_cast<uint32_t>(renderSize), static_cast<uint32_t>(renderSize), 1};
    vkCmdCopyImageToBuffer(cmd, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_staging.GetBuffer(), 1, &copyRegion);

    m_vkCore->EndSingleTimeCommands(cmd);

    // ── Readback pixels ──────────────────────────────────────────────
    const int pixelCount = renderSize * renderSize;
    std::vector<unsigned char> renderPixels(static_cast<size_t>(pixelCount) * 4, 0);

    void *mapped = m_staging.Map();
    if (!mapped)
        return false;

    if (m_colorFormat == VK_FORMAT_R16G16B16A16_SFLOAT) {
        const uint16_t *src = static_cast<const uint16_t *>(mapped);
        auto halfToFloat = [](uint16_t h) -> float {
            uint32_t sign = (h >> 15) & 0x1;
            uint32_t exponent = (h >> 10) & 0x1F;
            uint32_t mantissa = h & 0x3FF;
            if (exponent == 0) {
                if (mantissa == 0)
                    return sign ? -0.0f : 0.0f;
                float val = (mantissa / 1024.0f) * std::pow(2.0f, -14.0f);
                return sign ? -val : val;
            }
            if (exponent == 31)
                return mantissa ? 0.0f : (sign ? -1e30f : 1e30f);
            float val = std::pow(2.0f, static_cast<float>(exponent) - 15.0f) * (1.0f + mantissa / 1024.0f);
            return sign ? -val : val;
        };

        auto linearToSrgb = [](float c) -> float {
            if (c <= 0.0031308f)
                return c * 12.92f;
            return 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
        };

        for (int i = 0; i < pixelCount; ++i) {
            float r = halfToFloat(src[i * 4 + 0]);
            float g = halfToFloat(src[i * 4 + 1]);
            float b = halfToFloat(src[i * 4 + 2]);
            float a = std::clamp(halfToFloat(src[i * 4 + 3]), 0.0f, 1.0f);

            // Reinhard tonemap
            r = r / (1.0f + r);
            g = g / (1.0f + g);
            b = b / (1.0f + b);

            r = linearToSrgb(r);
            g = linearToSrgb(g);
            b = linearToSrgb(b);

            renderPixels[i * 4 + 0] = static_cast<unsigned char>(std::clamp(r, 0.0f, 1.0f) * 255.0f + 0.5f);
            renderPixels[i * 4 + 1] = static_cast<unsigned char>(std::clamp(g, 0.0f, 1.0f) * 255.0f + 0.5f);
            renderPixels[i * 4 + 2] = static_cast<unsigned char>(std::clamp(b, 0.0f, 1.0f) * 255.0f + 0.5f);
            renderPixels[i * 4 + 3] = static_cast<unsigned char>(a * 255.0f + 0.5f);
        }
    } else {
        std::memcpy(renderPixels.data(), mapped, renderPixels.size());
    }

    m_staging.Unmap();
    DownsampleRGBABox(renderPixels, renderSize, size, outPixels);
    return true;
}

// ============================================================================
// Resource management (mirrors GPUMaterialPreview)
// ============================================================================

bool GPUMeshPreview::EnsureResources(int size)
{
    auto &mpm = m_vkCore->GetMaterialPipelineManager();
    VkFormat colorFormat = mpm.GetColorFormat();
    VkFormat depthFormat = mpm.GetDepthFormat();
    VkSampleCountFlagBits sampleCount = mpm.GetSampleCount();

    const bool renderConfigChanged =
        (m_renderPass != VK_NULL_HANDLE) &&
        (colorFormat != m_colorFormat || depthFormat != m_depthFormat || sampleCount != m_sampleCount);

    if (renderConfigChanged) {
        DestroyFramebuffer();
        vkDestroyRenderPass(m_vkCore->GetDevice(), m_renderPass, nullptr);
        m_renderPass = VK_NULL_HANDLE;
    }

    m_colorFormat = colorFormat;
    m_depthFormat = depthFormat;
    m_sampleCount = sampleCount;

    if (m_renderPass == VK_NULL_HANDLE)
        CreateRenderPass();
    if (m_renderPass == VK_NULL_HANDLE)
        return false;

    if (m_currentSize != size) {
        DestroyFramebuffer();
        CreateFramebuffer(size);
        m_currentSize = size;
    }

    return m_framebuffer != VK_NULL_HANDLE;
}

void GPUMeshPreview::CreateRenderPass()
{
    VkDevice device = m_vkCore->GetDevice();

    std::vector<VkAttachmentDescription> attachments;
    std::vector<VkAttachmentReference> colorRefs;
    VkAttachmentReference depthRef{};

    VkAttachmentDescription colorAtt{};
    colorAtt.format = m_colorFormat;
    colorAtt.samples = m_sampleCount;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAtt.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments.push_back(colorAtt);

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorRefs.push_back(colorRef);

    bool hasDepth = (m_depthFormat != VK_FORMAT_UNDEFINED);
    if (hasDepth) {
        VkAttachmentDescription depthAtt{};
        depthAtt.format = m_depthFormat;
        depthAtt.samples = m_sampleCount;
        depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAtt.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachments.push_back(depthAtt);

        depthRef.attachment = 1;
        depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = static_cast<uint32_t>(colorRefs.size());
    subpass.pColorAttachments = colorRefs.data();
    subpass.pDepthStencilAttachment = hasDepth ? &depthRef : nullptr;

    const VkSubpassDependency dependency = vkrender::MakePipelineCompatibleSubpassDependency();

    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    rpInfo.pAttachments = attachments.data();
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 1;
    rpInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(m_vkCore->GetDevice(), &rpInfo, nullptr, &m_renderPass) != VK_SUCCESS) {
        INXLOG_ERROR("GPUMeshPreview: failed to create render pass");
        m_renderPass = VK_NULL_HANDLE;
    }
}

void GPUMeshPreview::CreateFramebuffer(int size)
{
    VkDevice device = m_vkCore->GetDevice();
    VmaAllocator allocator = m_vkCore->GetDeviceContext().GetVmaAllocator();
    uint32_t w = static_cast<uint32_t>(size);
    uint32_t h = static_cast<uint32_t>(size);

    m_msaaColor.Create(allocator, device, w, h, m_colorFormat, VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_sampleCount);
    m_msaaColor.CreateView(m_colorFormat, VK_IMAGE_ASPECT_COLOR_BIT);

    if (m_sampleCount != VK_SAMPLE_COUNT_1_BIT) {
        m_resolveColor.Create(allocator, device, w, h, m_colorFormat, VK_IMAGE_TILING_OPTIMAL,
                              VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_SAMPLE_COUNT_1_BIT);
    }

    if (m_depthFormat != VK_FORMAT_UNDEFINED) {
        m_depth.Create(allocator, device, w, h, m_depthFormat, VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_sampleCount);
        m_depth.CreateView(m_depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    std::vector<VkImageView> fbAttachments;
    fbAttachments.push_back(m_msaaColor.GetView());
    if (m_depth.IsValid())
        fbAttachments.push_back(m_depth.GetView());

    VkFramebufferCreateInfo fbInfo{};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = m_renderPass;
    fbInfo.attachmentCount = static_cast<uint32_t>(fbAttachments.size());
    fbInfo.pAttachments = fbAttachments.data();
    fbInfo.width = w;
    fbInfo.height = h;
    fbInfo.layers = 1;

    if (vkCreateFramebuffer(device, &fbInfo, nullptr, &m_framebuffer) != VK_SUCCESS) {
        INXLOG_ERROR("GPUMeshPreview: failed to create framebuffer");
        m_framebuffer = VK_NULL_HANDLE;
        return;
    }

    VkDeviceSize pixelBytes = (m_colorFormat == VK_FORMAT_R16G16B16A16_SFLOAT) ? 8 : 4;
    VkDeviceSize stagingSize = w * h * pixelBytes;
    m_staging.Create(allocator, device, stagingSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

void GPUMeshPreview::DestroyFramebuffer()
{
    VkDevice device = m_vkCore->GetDevice();
    if (m_framebuffer != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(device, m_framebuffer, nullptr);
        m_framebuffer = VK_NULL_HANDLE;
    }
    m_msaaColor.Destroy();
    m_resolveColor.Destroy();
    m_depth.Destroy();
    m_staging.Destroy();
    m_currentSize = 0;
}

} // namespace infernux
