/**
 * @file VkCoreDraw.cpp
 * @brief InxVkCoreModular — Drawing and per-object buffer management
 *
 * Split from InxVkCoreModular.cpp for maintainability.
 * Contains: DrawFrame, DrawSceneFiltered,
 *           SetDrawCalls, EnsureObjectBuffers, CleanupUnusedBuffers.
 */

#include "InxError.h"
#include "InxVkCoreModular.h"
#include "ProfileConfig.h"
#include "SceneRenderGraph.h"
#include "vk/VkRenderUtils.h"
#include "vk/VkTypes.h"

#include <function/renderer/Frustum.h>
#include <function/renderer/shader/ShaderProgram.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/scene/LightingData.h>

#include <SDL3/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <unordered_set>

namespace infernux
{

// ============================================================================
// Rendering
// ============================================================================

void InxVkCoreModular::DrawFrame(const float *viewPos, const float *viewLookAt, const float *viewUp)
{
#if INFERNUX_FRAME_PROFILE
    using Clock = std::chrono::high_resolution_clock;
    auto _t0 = Clock::now();
    auto _tPrev = _t0;
    auto _tNow = _t0;
#endif

    // Skip rendering when the window is minimized (zero extent).
    // Without this guard, vkAcquireNextImageKHR blocks indefinitely
    // because the swapchain has no presentable images at 0×0.
    {
        VkExtent2D ext = m_swapchain.GetExtent();
        if (ext.width == 0 || ext.height == 0) {
            // Yield a bit so we don't spin-lock the CPU while minimized
            SDL_Delay(16);
            return;
        }
    }

    // Acquire next swapchain image
    uint32_t imageIndex;
    auto result = m_swapchain.AcquireNextImage(imageIndex);

    if (result == vk::SwapchainResult::NeedRecreate) {
        RecreateSwapchain();
        return;
    }

    if (result == vk::SwapchainResult::Error) {
        INXLOG_ERROR("Failed to acquire swapchain image");
        return;
    }
#if INFERNUX_FRAME_PROFILE
    _tNow = Clock::now();
    m_drawSubMs[0] += std::chrono::duration<double, std::milli>(_tNow - _tPrev).count();
    _tPrev = _tNow;
#endif

    // Update uniform buffer
    UpdateUniformBuffer(m_currentFrame, viewPos, viewLookAt, viewUp);

    // Reset fence for this frame before submission
    m_swapchain.ResetCurrentFence();

    // Reset and record command buffer
    vkResetCommandBuffer(m_commandBuffers[m_currentFrame], 0);
    RecordCommandBuffer(imageIndex);
#if INFERNUX_FRAME_PROFILE
    _tNow = Clock::now();
    m_drawSubMs[1] += std::chrono::duration<double, std::milli>(_tNow - _tPrev).count();
    _tPrev = _tNow;
#endif

    // Submit command buffer
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {m_swapchain.GetImageAvailableSemaphore()};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBuffers[m_currentFrame];

    VkSemaphore signalSemaphores[] = {m_swapchain.GetRenderFinishedSemaphore(imageIndex)};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    VkResult submitResult =
        vkQueueSubmit(m_deviceContext.GetGraphicsQueue(), 1, &submitInfo, m_swapchain.GetInFlightFence());
    if (submitResult != VK_SUCCESS) {
        INXLOG_ERROR("Failed to submit draw command buffer: ", vk::VkResultToString(submitResult));
    }
#if INFERNUX_FRAME_PROFILE
    _tNow = Clock::now();
    m_drawSubMs[2] += std::chrono::duration<double, std::milli>(_tNow - _tPrev).count();
    _tPrev = _tNow;
#endif

    // Present
    result = m_swapchain.Present(imageIndex);
    if (result == vk::SwapchainResult::NeedRecreate || m_framebufferResized) {
        m_framebufferResized = false;
        RecreateSwapchain();
    }
#if INFERNUX_FRAME_PROFILE
    _tNow = Clock::now();
    m_drawSubMs[3] += std::chrono::duration<double, std::milli>(_tNow - _tPrev).count();

    ++m_drawSubCount;
#endif

    // Advance frame
    m_swapchain.AdvanceFrame();
    m_currentFrame = (m_currentFrame + 1) % m_maxFramesInFlight;
}

void InxVkCoreModular::SetDrawCalls(const std::vector<DrawCall> *drawCalls)
{
    m_drawCallsPtr = drawCalls;

    // Refresh cached builtin materials (avoids string-hash lookup per DrawSceneFiltered call)
    if (!m_cachedDefaultLit) {
        m_cachedDefaultLit = AssetRegistry::Instance().GetBuiltinMaterial("DefaultLit");
        m_cachedErrorMat = AssetRegistry::Instance().GetBuiltinMaterial("ErrorMaterial");
    }
}

void InxVkCoreModular::SetShadowDrawCalls(const std::vector<DrawCall> *drawCalls)
{
    m_shadowDrawCallsPtr = drawCalls;
}

// ============================================================================
// Multi-camera UBO update via command buffer
// ============================================================================

void InxVkCoreModular::CmdUpdateUniformBuffer(VkCommandBuffer cmdBuf, const glm::mat4 &view, const glm::mat4 &proj)
{
    // All material descriptor sets reference m_uniformBuffers[0] (hardcoded
    // in MaterialDescriptorManager). Always target buffer 0 regardless of
    // m_currentFrame so the shaders see the updated VP matrices.
    if (m_uniformBuffers.empty() || !m_uniformBuffers[0]) {
        return;
    }

    VkBuffer buffer = m_uniformBuffers[0]->GetBuffer();

    UniformBufferObject ubo{};
    ubo.model = glm::mat4(1.0f);
    ubo.view = view;
    ubo.proj = proj;

    // Barrier: ensure previous shader reads from the UBO are complete
    vkrender::CmdBarrierUniformReadToTransferWrite(cmdBuf);

    // Update the UBO inline in the command buffer
    vkCmdUpdateBuffer(cmdBuf, buffer, 0, sizeof(ubo), &ubo);

    // Barrier: ensure write is visible before subsequent shader reads
    vkrender::CmdBarrierTransferWriteToUniformRead(cmdBuf);
}

void InxVkCoreModular::CmdUpdateShadowUBO(VkCommandBuffer cmdBuf)
{
    if (m_shadowUboBuffers.empty()) {
        if (!EnsureShadowPipeline(VK_NULL_HANDLE)) {
            return;
        }
    }

    const uint32_t frameIndex = m_currentFrame % m_maxFramesInFlight;
    const uint32_t cascadeCount = m_lightCollector.GetShadowCascadeCount();
    if (cascadeCount == 0)
        return;

    struct ShadowUBO
    {
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 proj;
    };

    vkrender::CmdBarrierUniformReadToTransferWrite(cmdBuf);

    for (uint32_t ci = 0; ci < cascadeCount; ++ci) {
        uint32_t bufIdx = frameIndex * NUM_SHADOW_CASCADES + ci;
        if (bufIdx >= m_shadowUboBuffers.size())
            break;

        ShadowUBO shadowUbo{};
        shadowUbo.model = glm::mat4(1.0f);
        shadowUbo.view = glm::mat4(1.0f);
        shadowUbo.proj = m_lightCollector.GetShadowLightVP(ci);

        VkBuffer shadowBuffer = m_shadowUboBuffers[bufIdx];
        if (shadowBuffer == VK_NULL_HANDLE)
            continue;

        vkCmdUpdateBuffer(cmdBuf, shadowBuffer, 0, sizeof(ShadowUBO), &shadowUbo);
    }

    vkrender::CmdBarrierTransferWriteToUniformRead(cmdBuf);
}

// ============================================================================
// Phase 2: Filtered Draw — renders only draw calls within a queue range
// ============================================================================

void InxVkCoreModular::DrawSceneFiltered(VkCommandBuffer cmdBuf, uint32_t width, uint32_t height, int queueMin,
                                         int queueMax, const std::string &sortMode, const std::string &overrideMaterial,
                                         const std::string &passTag)
{
    // One-shot diagnostic: log queue-range filtering for first N frames
    static int s_filterDiagFrames = 0;

#if INFERNUX_FRAME_PROFILE
    using Clock = std::chrono::high_resolution_clock;
    const auto totalStart = Clock::now();
    auto stageStart = totalStart;
#endif

    // Fast early-out when no draw calls are staged
    if (drawCalls().empty())
        return;

#if INFERNUX_FRAME_PROFILE
    ++m_drawSceneFilteredCalls;
#endif

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(width);
    viewport.height = static_cast<float>(height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmdBuf, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {width, height};
    vkCmdSetScissor(cmdBuf, 0, 1, &scissor);

    bool hasAnyBuffers = !m_perObjectBuffers.empty();
    if (!hasAnyBuffers) {
        return;
    }

    const auto &defaultMaterial = m_cachedDefaultLit;
    const auto &errorMaterial = m_cachedErrorMat;
    if (!defaultMaterial) {
        return;
    }

    // Resolve override material (if specified)
    InxMaterial *overrideMatRaw = nullptr;
    std::shared_ptr<InxMaterial> overrideMatOwner; // keeps alive during this scope
    if (!overrideMaterial.empty()) {
        overrideMatOwner = AssetRegistry::Instance().GetBuiltinMaterial(overrideMaterial);
        overrideMatRaw = overrideMatOwner.get();
    }

    InxMaterial *defaultMatRaw = defaultMaterial.get();

    // ---- Collect eligible draw calls (queue filter + frustum cull) ----
    m_eligibleScratch.clear();

    for (const DrawCall &dc : drawCalls()) {
        if (!dc.frustumVisible)
            continue;

        InxMaterial *material = overrideMatRaw ? overrideMatRaw : (dc.material ? dc.material : defaultMatRaw);
        if (!material)
            continue;

        int queue = material->GetRenderQueue();
        if (queue < queueMin || queue > queueMax)
            continue;

        // Pass tag filter: if a pass tag is specified, only draw materials whose
        // passTag matches. Empty passTag on either side means "match all".
        if (!passTag.empty()) {
            const std::string &matTag = material->GetPassTag();
            if (!matTag.empty() && matTag != passTag)
                continue;
        }

        // Compute view-space depth for sorting.
        glm::vec4 viewPos = m_stagedUBO.view * glm::vec4(glm::vec3(dc.worldMatrix[3]), 1.0f);
        float sortKey = viewPos.z;

        // Material + mesh hash for grouping optimization
        size_t matHash = std::hash<void *>{}(static_cast<void *>(material));
        auto bufIt = m_perObjectBuffers.find(dc.objectId);
        VkBuffer vb = VK_NULL_HANDLE;
        if (bufIt != m_perObjectBuffers.end() && bufIt->second.vertexBuffer)
            vb = bufIt->second.vertexBuffer->GetBuffer();

        m_eligibleScratch.push_back({&dc, sortKey, matHash, vb, material, bufIt});
    }

    // Diagnostic: log per-call eligible count with queue range
    if (s_filterDiagFrames < 3) {
        INXLOG_DEBUG("[DrawSceneFiltered] queue=[", queueMin, ",", queueMax, "] totalDC=", drawCalls().size(),
                     " eligible=", m_eligibleScratch.size());
        if (!m_eligibleScratch.empty()) {
            for (const auto &entry : m_eligibleScratch) {
                INXLOG_DEBUG("  -> objId=", entry.dc->objectId, " mat='", entry.material->GetName(),
                             "' queue=", entry.material->GetRenderQueue());
            }
        }
        ++s_filterDiagFrames;
    }

#if INFERNUX_FRAME_PROFILE
    auto stageNow = Clock::now();
    m_drawSubMs[9] += std::chrono::duration<double, std::milli>(stageNow - stageStart).count();
    stageStart = stageNow;
    m_drawSceneFilteredEligible += static_cast<uint64_t>(m_eligibleScratch.size());
#endif

    // One-shot diagnostic: log icon draw calls that pass filtering
    {
        static int s_iconDiagCount = 0;
        if (s_iconDiagCount < 5) {
            for (const auto &entry : m_eligibleScratch) {
                const std::string &matName = entry.material->GetName();
                if (matName.find("GizmoIcon") != std::string::npos || matName.find("Gizmo") != std::string::npos) {
                    const DrawCall &dc = *entry.dc;
                    auto bufIt = m_perObjectBuffers.find(dc.objectId);
                    bool hasBuf =
                        (bufIt != m_perObjectBuffers.end() && bufIt->second.vertexBuffer && bufIt->second.indexBuffer);
                    VkPipeline pip = entry.material->GetPassPipeline(ShaderCompileTarget::Forward);
                    VkDescriptorSet ds = entry.material->GetPassDescriptorSet(ShaderCompileTarget::Forward);
                    ++s_iconDiagCount;
                }
            }
        }
    }

    if (m_eligibleScratch.empty()) {
#if INFERNUX_FRAME_PROFILE
        m_drawSubMs[8] += std::chrono::duration<double, std::milli>(Clock::now() - totalStart).count();
#endif
        return;
    }

    // ---- Sort if requested (skip for 0-1 elements) ----
    if (m_eligibleScratch.size() > 1) {
        // In left-handed view space: near objects have small positive Z, far
        // objects have larger positive Z.
        if (sortMode == "front_to_back") {
            // Front-to-back: group by material first (minimizes state changes),
            // then sort by depth within each material group.
            std::sort(m_eligibleScratch.begin(), m_eligibleScratch.end(),
                      [](const SortableDrawCall &a, const SortableDrawCall &b) {
                          if (a.materialHash != b.materialHash)
                              return a.materialHash < b.materialHash;
                          if (a.vertexBuf != b.vertexBuf)
                              return a.vertexBuf < b.vertexBuf;
                          return a.sortKey < b.sortKey;
                      });
        } else if (sortMode == "back_to_front") {
            // Far first → descending Z.
            // Depth order is critical for correct alpha blending.
            std::sort(m_eligibleScratch.begin(), m_eligibleScratch.end(),
                      [](const SortableDrawCall &a, const SortableDrawCall &b) { return a.sortKey > b.sortKey; });
        } else {
            // No explicit sort requested — group by material + mesh for max state reuse
            std::sort(m_eligibleScratch.begin(), m_eligibleScratch.end(),
                      [](const SortableDrawCall &a, const SortableDrawCall &b) {
                          if (a.materialHash != b.materialHash)
                              return a.materialHash < b.materialHash;
                          return a.vertexBuf < b.vertexBuf;
                      });
        }
    } // size() > 1

#if INFERNUX_FRAME_PROFILE
    stageNow = Clock::now();
    m_drawSubMs[10] += std::chrono::duration<double, std::milli>(stageNow - stageStart).count();
    stageStart = stageNow;
#endif

    // ---- Upload instance model matrices to SSBO (set 2, binding 1) ----
    // Reset per-frame write offset on new frame
    if (m_lastInstanceFrame != m_currentFrame) {
        m_instanceWriteOffset = 0;
        m_lastInstanceFrame = m_currentFrame;
    }

    const uint32_t frameIndex = m_currentFrame % m_maxFramesInFlight;
    const size_t totalEligible = m_eligibleScratch.size();
    const uint32_t writeBase = m_instanceWriteOffset;

    if (totalEligible > 0 && frameIndex < m_instanceBuffers.size()) {
        EnsureInstanceBufferCapacity(frameIndex, writeBase + totalEligible);
        auto &instFrame = m_instanceBuffers[frameIndex];
        if (instFrame.buffer) {
            void *mapped = instFrame.mapped;
            if (!mapped) {
                mapped = instFrame.buffer->Map();
                instFrame.mapped = mapped;
            }
            if (mapped) {
                glm::mat4 *matrices = static_cast<glm::mat4 *>(mapped);
                for (size_t i = 0; i < totalEligible; ++i) {
                    matrices[writeBase + i] = m_eligibleScratch[i].dc->worldMatrix;
                }
            }
        }
        m_instanceWriteOffset += static_cast<uint32_t>(totalEligible);
    }

    // ---- Draw loop with instanced batching ----
    VkPipeline currentPipeline = VK_NULL_HANDLE;
    VkPipelineLayout currentLayout = VK_NULL_HANDLE;
    VkDescriptorSet currentDescriptorSet = VK_NULL_HANDLE;
    InxMaterial *currentMaterialRaw = nullptr;
    VkBuffer currentVertexBuffer = VK_NULL_HANDLE;
    uint64_t issuedDraws = 0;

    // Batch accumulation: consecutive entries sharing (pipeline, descriptorSet, VB, submesh) are
    // emitted as a single vkCmdDrawIndexed with instanceCount > 1.
    const bool allowBatching = (sortMode != "back_to_front");

    size_t batchFirstInstance = 0;
    uint32_t batchInstanceCount = 0;
    uint32_t batchIndexStart = 0;
    uint32_t batchIndexCount = 0;
    int32_t batchVertexStart = 0;
    VkPipelineLayout batchPipelineLayout = VK_NULL_HANDLE;

    auto emitBatch = [&]() {
        if (batchInstanceCount == 0)
            return;
        // Push constants: model matrix for vertex shader (normalMat computed in shader from SSBO model)
        struct PushConstants
        {
            glm::mat4 model;
            glm::mat4 normalMat;
        };
        PushConstants pushData;
        pushData.model = m_eligibleScratch[batchFirstInstance].dc->worldMatrix;
        pushData.normalMat = glm::mat4(1.0f); // normalMat computed in shader from SSBO model
        vkCmdPushConstants(cmdBuf, batchPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants),
                           &pushData);
        vkCmdDrawIndexed(cmdBuf, batchIndexCount, batchInstanceCount, batchIndexStart, batchVertexStart,
                         writeBase + static_cast<uint32_t>(batchFirstInstance));
        issuedDraws += batchInstanceCount;
#if INFERNUX_FRAME_PROFILE
        ++m_drawFilteredActualDraws;
#endif
        batchInstanceCount = 0;
    };

    for (size_t idx = 0; idx < totalEligible; ++idx) {
        const auto &entry = m_eligibleScratch[idx];
        const DrawCall &dc = *entry.dc;

        // Material already resolved in filter loop — use directly
        InxMaterial *matRaw = entry.material;

        VkPipeline pipeline = matRaw->GetPassPipeline(ShaderCompileTarget::Forward);
        VkPipelineLayout pipelineLayout = matRaw->GetPassPipelineLayout(ShaderCompileTarget::Forward);

        // Force re-evaluation when material's shader config changed
        if (pipeline != VK_NULL_HANDLE && matRaw->IsPipelineDirty()) {
            pipeline = VK_NULL_HANDLE;
            pipelineLayout = VK_NULL_HANDLE;
        }

        if (pipeline == VK_NULL_HANDLE) {
            const std::string &vertName = matRaw->GetVertShaderName();
            const std::string &fragName = matRaw->GetFragShaderName();
            // Non-owning shared_ptr for legacy RefreshMaterialPipeline API (rare path)
            auto matShared = std::shared_ptr<InxMaterial>(matRaw, [](InxMaterial *) {});
            if (!fragName.empty()) {
                RefreshMaterialPipeline(matShared, vertName, fragName);
                pipeline = matRaw->GetPassPipeline(ShaderCompileTarget::Forward);
                pipelineLayout = matRaw->GetPassPipelineLayout(ShaderCompileTarget::Forward);
            }
            if (pipeline == VK_NULL_HANDLE && errorMaterial) {
                if (errorMaterial->GetPassPipeline(ShaderCompileTarget::Forward) == VK_NULL_HANDLE) {
                    const std::string &errVert = errorMaterial->GetVertShaderName();
                    const std::string &errFrag = errorMaterial->GetFragShaderName();
                    if (!errFrag.empty()) {
                        RefreshMaterialPipeline(errorMaterial, errVert, errFrag);
                    }
                }
                if (errorMaterial->GetPassPipeline(ShaderCompileTarget::Forward) != VK_NULL_HANDLE) {
                    pipeline = errorMaterial->GetPassPipeline(ShaderCompileTarget::Forward);
                    pipelineLayout = errorMaterial->GetPassPipelineLayout(ShaderCompileTarget::Forward);
                    matRaw = errorMaterial.get();
                }
            }
            if (pipeline == VK_NULL_HANDLE && defaultMaterial) {
                pipeline = defaultMaterial->GetPassPipeline(ShaderCompileTarget::Forward);
                pipelineLayout = defaultMaterial->GetPassPipelineLayout(ShaderCompileTarget::Forward);
                matRaw = defaultMaterial.get();
            }
            if (pipeline == VK_NULL_HANDLE) {
                emitBatch();
                continue;
            }
        }

        VkDescriptorSet descriptorSet = matRaw->GetPassDescriptorSet(ShaderCompileTarget::Forward);

        if (descriptorSet == VK_NULL_HANDLE) {
            static int warnCount = 0;
            if (warnCount++ < 10) {
                INXLOG_WARN("[DrawSceneFiltered] descriptorSet=NULL for material '", matRaw->GetName(),
                            "' queue=", matRaw->GetRenderQueue(),
                            " pipeline=", (pipeline != VK_NULL_HANDLE ? "OK" : "NULL"), " vert='",
                            matRaw->GetVertShaderName(), "' frag='", matRaw->GetFragShaderName(), "'");
            }
            emitBatch();
            continue;
        }

        // Check GPU buffers for this entry
        const auto &bufIt = entry.bufIt;
        if (bufIt == m_perObjectBuffers.end() || !bufIt->second.vertexBuffer || !bufIt->second.indexBuffer) {
            static int bufWarnCount = 0;
            if (bufWarnCount++ < 10) {
                INXLOG_WARN("[DrawSceneFiltered] no GPU buffers for objectId=", dc.objectId, " material='",
                            matRaw->GetName(), "' queue=", matRaw->GetRenderQueue());
            }
            emitBatch();
            continue;
        }

        VkBuffer vb = bufIt->second.vertexBuffer->GetBuffer();

        // Check if this entry can extend the current batch
        bool canExtendBatch = allowBatching && batchInstanceCount > 0 && pipeline == currentPipeline &&
                              descriptorSet == currentDescriptorSet && matRaw == currentMaterialRaw &&
                              vb == currentVertexBuffer && dc.indexStart == batchIndexStart &&
                              dc.indexCount == batchIndexCount && dc.vertexStart == batchVertexStart;

        if (canExtendBatch) {
            ++batchInstanceCount;
            continue;
        }

        // Emit previous batch before changing state
        emitBatch();

        // ---- Bind new state ----
        if (pipeline != currentPipeline) {
            vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            currentPipeline = pipeline;
        }

        if (matRaw != currentMaterialRaw) {
            UpdateMaterialUBO(*matRaw);
            currentMaterialRaw = matRaw;
            currentLayout = VK_NULL_HANDLE;
            currentDescriptorSet = VK_NULL_HANDLE;
        }

        if (descriptorSet != currentDescriptorSet || pipelineLayout != currentLayout) {
            vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0,
                                    nullptr);
            currentDescriptorSet = descriptorSet;
            currentLayout = pipelineLayout;

            ShaderProgram *program = matRaw->GetPassShaderProgram(ShaderCompileTarget::Forward);

            if (m_activeShadowDescSet != VK_NULL_HANDLE) {
                if (program && program->HasDeclaredDescriptorSet(1)) {
                    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 1, 1,
                                            &m_activeShadowDescSet, 0, nullptr);
                }
            }

            if (program && program->HasDeclaredDescriptorSet(2)) {
                if (frameIndex < m_globalsDescSets.size()) {
                    VkDescriptorSet globalsDescSet = m_globalsDescSets[frameIndex];
                    if (globalsDescSet != VK_NULL_HANDLE) {
                        vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 2, 1,
                                                &globalsDescSet, 0, nullptr);
                    }
                }
            }
        }

        if (vb != currentVertexBuffer) {
            VkBuffer vertBuffers[] = {vb};
            VkDeviceSize vbOffsets[] = {0};
            vkCmdBindVertexBuffers(cmdBuf, 0, 1, vertBuffers, vbOffsets);
            vkCmdBindIndexBuffer(cmdBuf, bufIt->second.indexBuffer->GetBuffer(), 0, VK_INDEX_TYPE_UINT32);
            currentVertexBuffer = vb;
        }

        // Start new batch
        batchFirstInstance = idx;
        batchInstanceCount = 1;
        batchIndexStart = dc.indexStart;
        batchIndexCount = dc.indexCount;
        batchVertexStart = dc.vertexStart;
        batchPipelineLayout = pipelineLayout;
    }

    // Flush final batch
    emitBatch();

#if INFERNUX_FRAME_PROFILE
    stageNow = Clock::now();
    m_drawSubMs[11] += std::chrono::duration<double, std::milli>(stageNow - stageStart).count();
    m_drawSubMs[8] += std::chrono::duration<double, std::milli>(stageNow - totalStart).count();
    m_drawSceneFilteredIssued += issuedDraws;
#endif
}

// ============================================================================
// Shadow Caster Draw — renders shadow-casting objects with shadow pipeline
// ============================================================================

void InxVkCoreModular::DrawShadowCasters(VkCommandBuffer cmdBuf, uint32_t width, uint32_t height, int queueMin,
                                         int queueMax, int lightIndex, const std::string &shadowType)
{
#if INFERNUX_FRAME_PROFILE
    using Clock = std::chrono::high_resolution_clock;
    const auto totalStart = Clock::now();
    auto stageStart = totalStart;
#endif

    // Currently only light index 0 (first directional light) is supported.
    // Log a warning if callers request a different index so the mismatch is visible.
    if (lightIndex != 0) {
        INXLOG_WARN("DrawShadowCasters: lightIndex=", lightIndex,
                    " requested but only index 0 is supported; using light 0");
    }
    // shadowType is stored for future soft-shadow filtering but not yet consumed.
    (void)shadowType;

    // Skip if shadow pipeline infrastructure not ready (lazy init)
    if (!m_shadowPipelineReady) {
        if (!EnsureShadowPipeline(VK_NULL_HANDLE)) {
            return;
        }
    }

    const uint32_t cascadeCount = m_lightCollector.GetShadowCascadeCount();
    if (cascadeCount == 0)
        return;

#if INFERNUX_FRAME_PROFILE
    ++m_drawShadowCalls;
#endif

    const uint32_t frameIndex = m_currentFrame % m_maxFramesInFlight;

    // Bind shadow pipeline once
    // NOTE: Per-material shadow pipelines override this in the inner loop
    VkPipeline lastBoundPipeline = VK_NULL_HANDLE;

    // Atlas layout: 2x2 tiles inside the full shadow map
    uint32_t tileW = width / 2;
    uint32_t tileH = height / 2;

    // Pre-build draw list (filter once, reuse for all cascades)
    m_shadowDrawScratch.clear();
    m_shadowDrawScratch.reserve(shadowDrawCalls().size());
    for (const DrawCall &dc : shadowDrawCalls()) {
        if (!dc.material)
            continue;
        int renderQueue = dc.material->GetRenderQueue();
        if (renderQueue < queueMin || renderQueue > queueMax)
            continue;
        auto bufIt = m_perObjectBuffers.find(dc.objectId);
        if (bufIt == m_perObjectBuffers.end() || !bufIt->second.vertexBuffer || !bufIt->second.indexBuffer)
            continue;

        VkPipeline pip = dc.material->GetPassPipeline(ShaderCompileTarget::Shadow);
        if (pip == VK_NULL_HANDLE) {
            // Lazy creation: shadow shared resources are ready, create per-material pipeline now
            auto matShared = std::shared_ptr<InxMaterial>(dc.material, [](InxMaterial *) {});
            CreateMaterialShadowPipeline(matShared, dc.material->GetVertShaderName(),
                                         dc.material->GetFragShaderName());
            pip = dc.material->GetPassPipeline(ShaderCompileTarget::Shadow);
        }
        if (pip == VK_NULL_HANDLE)
            continue; // no per-material shadow pipeline available, skip
        m_shadowDrawScratch.push_back(
            {&dc, bufIt, pip, dc.material->GetPassDescriptorSet(ShaderCompileTarget::Shadow), dc.worldBounds});
    }

#if INFERNUX_FRAME_PROFILE
    auto stageNow = Clock::now();
    m_drawSubMs[13] += std::chrono::duration<double, std::milli>(stageNow - stageStart).count();
    stageStart = stageNow;
    m_drawShadowEligible += static_cast<uint64_t>(m_shadowDrawScratch.size());
#endif

    if (m_shadowDrawScratch.empty()) {
        static int s_noShadowDrawsWarnCount = 0;

#if INFERNUX_FRAME_PROFILE
        m_drawSubMs[12] += std::chrono::duration<double, std::milli>(Clock::now() - totalStart).count();
#endif
        return;
    }

    // Sort shadow draw scratch by (pipeline, VB, submesh) for instanced batching
    std::sort(m_shadowDrawScratch.begin(), m_shadowDrawScratch.end(), [](const ShadowDraw &a, const ShadowDraw &b) {
        if (a.shadowPipeline != b.shadowPipeline)
            return a.shadowPipeline < b.shadowPipeline;
        VkBuffer va = a.bufIt->second.vertexBuffer->GetBuffer();
        VkBuffer vb_b = b.bufIt->second.vertexBuffer->GetBuffer();
        if (va != vb_b)
            return va < vb_b;
        if (a.dc->indexStart != b.dc->indexStart)
            return a.dc->indexStart < b.dc->indexStart;
        return a.dc->indexCount < b.dc->indexCount;
    });

#if INFERNUX_FRAME_PROFILE
    stageNow = Clock::now();
    m_drawSubMs[15] += std::chrono::duration<double, std::milli>(stageNow - stageStart).count();
    stageStart = stageNow;
#endif

    uint64_t issuedDraws = 0;

    // Pre-extract frustums for all cascades (for per-cascade AABB culling)
    // Use override VPs when set (game camera), otherwise fall back to m_lightCollector (editor camera).
    Frustum cascadeFrustums[NUM_SHADOW_CASCADES];
    for (uint32_t ci = 0; ci < cascadeCount && ci < NUM_SHADOW_CASCADES; ++ci) {
        if (m_shadowCascadeVPOverride && ci < m_shadowCascadeVPOverrideCount) {
            cascadeFrustums[ci].ExtractFromMatrix(m_shadowCascadeVPOverride[ci]);
        } else {
            cascadeFrustums[ci].ExtractFromMatrix(m_lightCollector.GetShadowLightVP(ci));
        }
    }

    // Bind globals descriptor set (set 1) with instance SSBO — once for all cascades
    if (frameIndex < m_globalsDescSets.size()) {
        VkDescriptorSet globalsDescSet = m_globalsDescSets[frameIndex];
        if (globalsDescSet != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipelineLayout, 1, 1,
                                    &globalsDescSet, 0, nullptr);
        }
    }

    for (uint32_t ci = 0; ci < cascadeCount; ++ci) {
        uint32_t descIdx = frameIndex * NUM_SHADOW_CASCADES + ci;
        if (descIdx >= m_shadowDescSets.size())
            break;

        // Set viewport/scissor to cascade's atlas tile
        uint32_t tileX = (ci % 2) * tileW;
        uint32_t tileY = (ci / 2) * tileH;

        VkViewport viewport{};
        viewport.x = static_cast<float>(tileX);
        viewport.y = static_cast<float>(tileY);
        viewport.width = static_cast<float>(tileW);
        viewport.height = static_cast<float>(tileH);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmdBuf, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {static_cast<int32_t>(tileX), static_cast<int32_t>(tileY)};
        scissor.extent = {tileW, tileH};
        vkCmdSetScissor(cmdBuf, 0, 1, &scissor);

        // Bind per-cascade descriptor set (set 0)
        VkDescriptorSet cascadeDescSet = m_shadowDescSets[descIdx];
        vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipelineLayout, 0, 1, &cascadeDescSet,
                                0, nullptr);

        const Frustum &cascadeFrustum = cascadeFrustums[ci];
        VkBuffer currentVertexBuffer = VK_NULL_HANDLE;
        VkDescriptorSet lastBoundShadowMaterialDescSet = VK_NULL_HANDLE;

        // Per-cascade frustum cull into a compact index list, then upload
        // model matrices for visible objects and batch by (pipeline, VB, submesh).
        m_shadowCascadeVisible.clear();
        m_shadowCascadeVisible.reserve(m_shadowDrawScratch.size());
        for (size_t si = 0; si < m_shadowDrawScratch.size(); ++si) {
            const auto &sd = m_shadowDrawScratch[si];
            if (sd.worldBounds.IsValid() && !cascadeFrustum.IntersectsAABB(sd.worldBounds))
                continue;
            m_shadowCascadeVisible.push_back(static_cast<uint32_t>(si));
        }

#if INFERNUX_FRAME_PROFILE
        stageNow = Clock::now();
        m_drawSubMs[16] += std::chrono::duration<double, std::milli>(stageNow - stageStart).count();
        stageStart = stageNow;
#endif

        const uint32_t visibleCount = static_cast<uint32_t>(m_shadowCascadeVisible.size());
        if (visibleCount == 0)
            continue;

        // Upload model matrices for this cascade's visible objects into the shared SSBO
        const uint32_t writeBase = m_instanceWriteOffset;
        EnsureInstanceBufferCapacity(frameIndex, writeBase + visibleCount);

        auto &instFrame = m_instanceBuffers[frameIndex];
        void *mapped = instFrame.mapped;
        if (!mapped) {
            mapped = instFrame.buffer->Map();
            instFrame.mapped = mapped;
        }
        if (!mapped)
            continue;
        glm::mat4 *matrices = static_cast<glm::mat4 *>(mapped);
        for (uint32_t vi = 0; vi < visibleCount; ++vi) {
            matrices[writeBase + vi] = m_shadowDrawScratch[m_shadowCascadeVisible[vi]].dc->worldMatrix;
        }
        m_instanceWriteOffset += visibleCount;

#if INFERNUX_FRAME_PROFILE
        stageNow = Clock::now();
        m_drawSubMs[17] += std::chrono::duration<double, std::milli>(stageNow - stageStart).count();
        stageStart = stageNow;
#endif

        // Batch accumulation: consecutive visible entries sharing (pipeline, VB, submesh)
        // are emitted as a single instanced draw call.
        size_t batchStart = 0;
        uint32_t batchCount = 0;
        VkPipeline batchPipeline = VK_NULL_HANDLE;
        VkDescriptorSet batchShadowMaterialDescSet = VK_NULL_HANDLE;
        uint32_t batchIdxStart = 0;
        uint32_t batchIdxCount = 0;
        int32_t batchVtxStart = 0;

        auto emitShadowBatch = [&]() {
            if (batchCount == 0)
                return;
            struct PushData
            {
                glm::mat4 model;
                glm::mat4 normalMat;
            } pushData;
            pushData.model = m_shadowDrawScratch[m_shadowCascadeVisible[batchStart]].dc->worldMatrix;
            pushData.normalMat = glm::mat4(1.0f);
            vkCmdPushConstants(cmdBuf, m_shadowPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushData),
                               &pushData);
            vkCmdDrawIndexed(cmdBuf, batchIdxCount, batchCount, batchIdxStart, batchVtxStart,
                             writeBase + static_cast<uint32_t>(batchStart));
            issuedDraws += batchCount;
#if INFERNUX_FRAME_PROFILE
            ++m_drawShadowActualDraws;
#endif
            batchCount = 0;
        };

        for (uint32_t vi = 0; vi < visibleCount; ++vi) {
            const auto &sd = m_shadowDrawScratch[m_shadowCascadeVisible[vi]];

            VkBuffer vb = sd.bufIt->second.vertexBuffer->GetBuffer();

            bool canExtend = batchCount > 0 && sd.shadowPipeline == batchPipeline && vb == currentVertexBuffer &&
                             sd.shadowMaterialDescSet == batchShadowMaterialDescSet &&
                             sd.dc->indexStart == batchIdxStart && sd.dc->indexCount == batchIdxCount &&
                             sd.dc->vertexStart == batchVtxStart;

            if (canExtend) {
                ++batchCount;
                continue;
            }

            emitShadowBatch();

            // Bind per-material shadow pipeline if changed
            if (sd.shadowPipeline != lastBoundPipeline) {
                vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, sd.shadowPipeline);
                lastBoundPipeline = sd.shadowPipeline;
            }

            if (sd.shadowMaterialDescSet != lastBoundShadowMaterialDescSet &&
                sd.shadowMaterialDescSet != VK_NULL_HANDLE) {
                vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipelineLayout, 2, 1,
                                        &sd.shadowMaterialDescSet, 0, nullptr);
                lastBoundShadowMaterialDescSet = sd.shadowMaterialDescSet;
            }

            if (vb != currentVertexBuffer) {
                VkDeviceSize offsets[] = {0};
                vkCmdBindVertexBuffers(cmdBuf, 0, 1, &vb, offsets);
                vkCmdBindIndexBuffer(cmdBuf, sd.bufIt->second.indexBuffer->GetBuffer(), 0, VK_INDEX_TYPE_UINT32);
                currentVertexBuffer = vb;
            }

            batchStart = vi;
            batchCount = 1;
            batchPipeline = sd.shadowPipeline;
            batchShadowMaterialDescSet = sd.shadowMaterialDescSet;
            batchIdxStart = sd.dc->indexStart;
            batchIdxCount = sd.dc->indexCount;
            batchVtxStart = sd.dc->vertexStart;
        }

        emitShadowBatch();

#if INFERNUX_FRAME_PROFILE
        stageNow = Clock::now();
        m_drawSubMs[18] += std::chrono::duration<double, std::milli>(stageNow - stageStart).count();
        stageStart = stageNow;
#endif
    }

#if INFERNUX_FRAME_PROFILE
    stageNow = Clock::now();
    m_drawSubMs[14] += std::chrono::duration<double, std::milli>(stageNow - totalStart).count();
    m_drawSubMs[12] += std::chrono::duration<double, std::milli>(stageNow - totalStart).count();
    m_drawShadowIssued += issuedDraws;
#endif
}

// ============================================================================
// Shadow Pipeline Management
// ============================================================================

bool InxVkCoreModular::EnsureShadowPipeline(VkRenderPass /*compatibleRenderPass*/)
{
    if (m_shadowPipelineReady)
        return true;

    VkDevice device = GetDevice();

    // --- Create a compatible depth-only render pass ---
    if (m_shadowCompatRenderPass == VK_NULL_HANDLE) {
        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = VK_FORMAT_D32_SFLOAT;
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkAttachmentReference depthRef{};
        depthRef.attachment = 0;
        depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 0;
        subpass.pDepthStencilAttachment = &depthRef;

        const VkSubpassDependency dependency = vkrender::MakePipelineCompatibleSubpassDependency();

        VkRenderPassCreateInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpInfo.attachmentCount = 1;
        rpInfo.pAttachments = &depthAttachment;
        rpInfo.subpassCount = 1;
        rpInfo.pSubpasses = &subpass;
        rpInfo.dependencyCount = 1;
        rpInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &rpInfo, nullptr, &m_shadowCompatRenderPass) != VK_SUCCESS) {
            INXLOG_ERROR("Failed to create shadow-compatible render pass");
            return false;
        }
    }

    // --- Create descriptor set layout (binding 0 = UBO) ---
    if (m_shadowDescSetLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding uboBinding{};
        uboBinding.binding = 0;
        uboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboBinding.descriptorCount = 1;
        uboBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &uboBinding;

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_shadowDescSetLayout) != VK_SUCCESS) {
            INXLOG_ERROR("Failed to create shadow descriptor set layout");
            return false;
        }
    }

    if (m_shadowMaterialDescSetLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding materialBinding{};
        materialBinding.binding = 14;
        materialBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        materialBinding.descriptorCount = 1;
        materialBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &materialBinding;

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_shadowMaterialDescSetLayout) != VK_SUCCESS) {
            INXLOG_ERROR("Failed to create shadow material descriptor set layout");
            return false;
        }
    }

    // --- Create descriptor pool (frames × cascades) ---
    const uint32_t totalSets = m_maxFramesInFlight * NUM_SHADOW_CASCADES;
    if (m_shadowDescPool == VK_NULL_HANDLE) {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount = totalSets;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = totalSets;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_shadowDescPool) != VK_SUCCESS) {
            INXLOG_ERROR("Failed to create shadow descriptor pool");
            return false;
        }
    }

    if (m_shadowMaterialDescPool == VK_NULL_HANDLE) {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount = 1024;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.maxSets = 1024;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_shadowMaterialDescPool) != VK_SUCCESS) {
            INXLOG_ERROR("Failed to create shadow material descriptor pool");
            return false;
        }
    }

    // --- Allocate descriptor sets (frames × cascades) ---
    if (m_shadowDescSets.empty()) {
        m_shadowDescSets.resize(totalSets, VK_NULL_HANDLE);
        std::vector<VkDescriptorSetLayout> setLayouts(totalSets, m_shadowDescSetLayout);

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_shadowDescPool;
        allocInfo.descriptorSetCount = totalSets;
        allocInfo.pSetLayouts = setLayouts.data();

        if (vkAllocateDescriptorSets(device, &allocInfo, m_shadowDescSets.data()) != VK_SUCCESS) {
            INXLOG_ERROR("Failed to allocate shadow descriptor sets");
            return false;
        }
    }

    // --- Create per-frame per-cascade UBO buffers ---
    if (m_shadowUboBuffers.empty()) {
        VkDeviceSize uboSize = sizeof(glm::mat4) * 3;

        m_shadowUboBuffers.resize(totalSets, VK_NULL_HANDLE);
        m_shadowUboAllocations.resize(totalSets, VK_NULL_HANDLE);
        m_shadowUboMappedPtrs.resize(totalSets, nullptr);

        VmaAllocator allocator = m_deviceContext.GetVmaAllocator();
        for (uint32_t i = 0; i < totalSets; ++i) {
            VkBufferCreateInfo bufInfo{};
            bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufInfo.size = uboSize;
            bufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo allocCreateInfo{};
            allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
            allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            allocCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

            VmaAllocationInfo vmaAllocInfo{};
            VkResult result = vmaCreateBuffer(allocator, &bufInfo, &allocCreateInfo, &m_shadowUboBuffers[i],
                                              &m_shadowUboAllocations[i], &vmaAllocInfo);
            if (result != VK_SUCCESS) {
                INXLOG_ERROR("Failed to create shadow UBO buffer via VMA");
                return false;
            }
            m_shadowUboMappedPtrs[i] = vmaAllocInfo.pMappedData;

            VkDescriptorBufferInfo bufDesc{};
            bufDesc.buffer = m_shadowUboBuffers[i];
            bufDesc.offset = 0;
            bufDesc.range = uboSize;

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = m_shadowDescSets[i];
            write.dstBinding = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &bufDesc;

            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }
    }

    // --- Create shadow depth sampler ---
    if (m_shadowDepthSampler == VK_NULL_HANDLE) {
        if (!CreateShadowDepthSampler()) {
            return false;
        }
    }

    // --- Create shadow pipeline layout (shared by all per-material shadow pipelines) ---
    // Set 0 = shadow UBO (per-cascade), set 1 = globals (UBO + instance SSBO),
    // set 2 = vertex material UBO (binding 14, when needed).
    if (m_globalsDescSetLayout == VK_NULL_HANDLE) {
        INXLOG_ERROR("EnsureShadowPipeline: m_globalsDescSetLayout is null — globals not yet initialized");
        return false;
    }
    if (m_shadowPipelineLayout == VK_NULL_HANDLE) {
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(glm::mat4) * 2; // model + normalMat

        VkDescriptorSetLayout setLayouts[3] = {m_shadowDescSetLayout, m_globalsDescSetLayout,
                                               m_shadowMaterialDescSetLayout};

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 3;
        pipelineLayoutInfo.pSetLayouts = setLayouts;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &m_shadowPipelineLayout) != VK_SUCCESS) {
            INXLOG_ERROR("Failed to create shadow pipeline layout");
            return false;
        }
    }

    m_shadowPipelineReady = true;
    // INXLOG_INFO("Shadow pipeline infrastructure created successfully");
    return true;
}

bool InxVkCoreModular::CreateShadowDepthSampler()
{
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_NEVER;
    samplerInfo.maxLod = 1.0f;

    if (vkCreateSampler(GetDevice(), &samplerInfo, nullptr, &m_shadowDepthSampler) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to create shadow depth sampler");
        return false;
    }
    return true;
}

void InxVkCoreModular::CleanupShadowPipeline()
{
    VkDevice device = GetDevice();
    if (device == VK_NULL_HANDLE)
        return;

    if (m_shadowPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, m_shadowPipelineLayout, nullptr);
        m_shadowPipelineLayout = VK_NULL_HANDLE;
    }
    if (m_shadowDescPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, m_shadowDescPool, nullptr);
        m_shadowDescPool = VK_NULL_HANDLE;
        m_shadowDescSets.clear();
    }
    if (m_shadowDescSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, m_shadowDescSetLayout, nullptr);
        m_shadowDescSetLayout = VK_NULL_HANDLE;
    }
    if (m_shadowMaterialDescPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, m_shadowMaterialDescPool, nullptr);
        m_shadowMaterialDescPool = VK_NULL_HANDLE;
    }
    if (m_shadowMaterialDescSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, m_shadowMaterialDescSetLayout, nullptr);
        m_shadowMaterialDescSetLayout = VK_NULL_HANDLE;
    }
    if (!m_shadowUboBuffers.empty()) {
        VmaAllocator allocator = m_deviceContext.GetVmaAllocator();
        for (size_t i = 0; i < m_shadowUboBuffers.size(); ++i) {
            if (m_shadowUboBuffers[i] != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, m_shadowUboBuffers[i], m_shadowUboAllocations[i]);
            }
        }
        m_shadowUboBuffers.clear();
        m_shadowUboAllocations.clear();
        m_shadowUboMappedPtrs.clear();
    }
    if (m_shadowDepthSampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, m_shadowDepthSampler, nullptr);
        m_shadowDepthSampler = VK_NULL_HANDLE;
    }
    if (m_shadowCompatRenderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device, m_shadowCompatRenderPass, nullptr);
        m_shadowCompatRenderPass = VK_NULL_HANDLE;
    }
    m_shadowPipelineReady = false;
}

// ============================================================================
// Per-Object Buffer Management (Phase 2.3.4)
// ============================================================================

void InxVkCoreModular::EnsureObjectBuffers(uint64_t objectId, const std::vector<Vertex> &vertices,
                                           const std::vector<uint32_t> &indices, bool forceUpdate)
{
    if (vertices.empty() || indices.empty())
        return;

    auto objectIt = m_perObjectBuffers.find(objectId);
    if (objectIt != m_perObjectBuffers.end() && !forceUpdate) {
        // Frame-stamp dedup: if already ensured this frame, skip entirely
        if (objectIt->second.ensuredOnFrame == m_ensureFrameCounter) {
            return;
        }
        // Fast path: if data pointers AND sizes match, content hasn't changed
        if (objectIt->second.lastVertexPtr == vertices.data() && objectIt->second.lastIndexPtr == indices.data() &&
            objectIt->second.vertexCount == vertices.size() && objectIt->second.indexCount == indices.size()) {
            objectIt->second.ensuredOnFrame = m_ensureFrameCounter;
            return;
        }
    }

    // Slow path: compute content hash for deduplication
    const size_t vtxBytes = vertices.size() * sizeof(Vertex);
    const size_t idxBytes = indices.size() * sizeof(uint32_t);
    const size_t contentHash = HashMeshContent(vertices.data(), vtxBytes, indices.data(), idxBytes);
    const SharedMeshKey sharedKey{contentHash, vertices.size(), indices.size()};

    // Check if object already maps to this exact content (pointer changed but content same)
    if (objectIt != m_perObjectBuffers.end() && !forceUpdate) {
        if (objectIt->second.sharedKey == sharedKey) {
            objectIt->second.lastVertexPtr = vertices.data();
            objectIt->second.lastIndexPtr = indices.data();
            return;
        }
    }

    auto sharedIt = m_sharedMeshBuffers.find(sharedKey);
    // NOTE: forceUpdate is intentionally NOT included in needsCreate.
    // The content hash already guarantees correctness — if the hash matches
    // an existing shared buffer, the GPU data is identical regardless of
    // forceUpdate.  Including forceUpdate here caused a catastrophic bug:
    // when the Scene view opened and ConsumeMeshBufferDirty() returned true
    // for all objects, each object would create its own VkBuffer (replacing
    // the shared entry), permanently destroying instancing (4 → 6000+ draws).
    const bool needsCreate =
        (sharedIt == m_sharedMeshBuffers.end() || sharedIt->second.vertexCount != vertices.size() ||
         sharedIt->second.indexCount != indices.size() || !sharedIt->second.vertexBuffer ||
         !sharedIt->second.indexBuffer);

    if (needsCreate) {
        SharedMeshBuffers oldSharedBuffers;
        const bool replacingExistingSharedBuffers = (sharedIt != m_sharedMeshBuffers.end());
        if (replacingExistingSharedBuffers) {
            oldSharedBuffers = sharedIt->second;
        }

        SharedMeshBuffers sharedBuffers;
        sharedBuffers.vertexBuffer = std::shared_ptr<vk::VkBufferHandle>(
            m_resourceManager.CreateVertexBuffer(vertices.data(), vertices.size() * sizeof(Vertex)).release());
        sharedBuffers.indexBuffer = std::shared_ptr<vk::VkBufferHandle>(
            m_resourceManager.CreateIndexBuffer(indices.data(), indices.size() * sizeof(uint32_t)).release());
        sharedBuffers.vertexCount = vertices.size();
        sharedBuffers.indexCount = indices.size();

        if (replacingExistingSharedBuffers) {
            sharedIt->second = std::move(sharedBuffers);

            // The old shared VB/IB may still be referenced by command buffers
            // from earlier in-flight frames. Defer the final shared_ptr release
            // until the frame deletion queue says it is safe.
            if (oldSharedBuffers.vertexBuffer || oldSharedBuffers.indexBuffer) {
                auto retiredBuffers = std::make_shared<SharedMeshBuffers>(std::move(oldSharedBuffers));
                m_deletionQueue.Push([retiredBuffers]() mutable {
                    retiredBuffers->vertexBuffer.reset();
                    retiredBuffers->indexBuffer.reset();
                });
            }
        } else {
            sharedIt = m_sharedMeshBuffers.insert_or_assign(sharedKey, std::move(sharedBuffers)).first;
        }
    }

    PerObjectBuffers objectBuffers;
    objectBuffers.vertexBuffer = sharedIt->second.vertexBuffer;
    objectBuffers.indexBuffer = sharedIt->second.indexBuffer;
    objectBuffers.vertexCount = sharedIt->second.vertexCount;
    objectBuffers.indexCount = sharedIt->second.indexCount;
    objectBuffers.sharedKey = sharedKey;
    objectBuffers.lastVertexPtr = vertices.data();
    objectBuffers.lastIndexPtr = indices.data();
    objectBuffers.ensuredOnFrame = m_ensureFrameCounter;

    m_perObjectBuffers[objectId] = std::move(objectBuffers);
}

void InxVkCoreModular::CleanupUnusedBuffers(const std::vector<DrawCall> &activeDrawCalls)
{
    // Build set of active objectIds
    std::unordered_set<uint64_t> activeIds;
    for (const auto &dc : activeDrawCalls) {
        activeIds.insert(dc.objectId);
    }
    CleanupUnusedBuffersByIds(activeIds);
}

void InxVkCoreModular::CleanupUnusedBuffersByIds(const std::unordered_set<uint64_t> &activeIds)
{    // Remove buffers for objects no longer in the scene.
    // Actual GPU resource destruction is deferred via FrameDeletionQueue
    // so that in-flight command buffers are never invalidated.
    for (auto it = m_perObjectBuffers.begin(); it != m_perObjectBuffers.end();) {
        if (activeIds.find(it->first) == activeIds.end()) {
            it = m_perObjectBuffers.erase(it);
        } else {
            ++it;
        }
    }

    // Collect active shared keys from surviving per-object entries
    std::unordered_set<SharedMeshKey, SharedMeshKeyHash> activeSharedKeysFromObjects;
    for (const auto &kv : m_perObjectBuffers) {
        activeSharedKeysFromObjects.insert(kv.second.sharedKey);
    }

    for (auto it = m_sharedMeshBuffers.begin(); it != m_sharedMeshBuffers.end();) {
        if (activeSharedKeysFromObjects.find(it->first) == activeSharedKeysFromObjects.end()) {
            auto buffers = std::make_shared<SharedMeshBuffers>(std::move(it->second));
            m_deletionQueue.Push([buffers]() mutable {
                buffers->vertexBuffer.reset();
                buffers->indexBuffer.reset();
            });
            it = m_sharedMeshBuffers.erase(it);
        } else {
            ++it;
        }
    }
}

size_t InxVkCoreModular::CleanupUnusedBuffersByFrameStamp()
{
    // Remove objects that were not referenced by EnsureObjectBuffers on this frame.
    bool anyRemoved = false;
    for (auto it = m_perObjectBuffers.begin(); it != m_perObjectBuffers.end();) {
        if (it->second.ensuredOnFrame != m_ensureFrameCounter) {
            it = m_perObjectBuffers.erase(it);
            anyRemoved = true;
        } else {
            ++it;
        }
    }

    // Only rebuild the active-shared-keys set and prune shared buffers when
    // at least one per-object entry was removed.  In a stable scene (no adds/removes)
    // this skips building a 10k-element unordered_set every frame.
    if (anyRemoved) {
        std::unordered_set<SharedMeshKey, SharedMeshKeyHash> activeSharedKeysFromObjects;
        activeSharedKeysFromObjects.reserve(m_perObjectBuffers.size());
        for (const auto &kv : m_perObjectBuffers) {
            activeSharedKeysFromObjects.insert(kv.second.sharedKey);
        }

        for (auto it = m_sharedMeshBuffers.begin(); it != m_sharedMeshBuffers.end();) {
            if (activeSharedKeysFromObjects.find(it->first) == activeSharedKeysFromObjects.end()) {
                auto buffers = std::make_shared<SharedMeshBuffers>(std::move(it->second));
                m_deletionQueue.Push([buffers]() mutable {
                    buffers->vertexBuffer.reset();
                    buffers->indexBuffer.reset();
                });
                it = m_sharedMeshBuffers.erase(it);
            } else {
                ++it;
            }
        }
    }

    return m_perObjectBuffers.size();
}

// ============================================================================
// Per-View Descriptor Set (set 1) — multi-camera shadow isolation
// ============================================================================

bool InxVkCoreModular::CreatePerViewDescriptorResources()
{
    VkDevice device = GetDevice();
    if (device == VK_NULL_HANDLE)
        return false;

    // Layout: set 1, binding 0 = combined image sampler (shadow map), fragment stage
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    binding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_perViewDescSetLayout) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to create per-view descriptor set layout");
        return false;
    }

    // Pool: enough for multiple render graphs (scene + game + future cameras)
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 16; // Up to 16 render graphs

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 16;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_perViewDescPool) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to create per-view descriptor pool");
        vkDestroyDescriptorSetLayout(device, m_perViewDescSetLayout, nullptr);
        m_perViewDescSetLayout = VK_NULL_HANDLE;
        return false;
    }

    INXLOG_INFO("Created per-view descriptor set layout and pool (multi-camera shadow)");
    return true;
}

void InxVkCoreModular::DestroyPerViewDescriptorResources()
{
    VkDevice device = GetDevice();
    if (device == VK_NULL_HANDLE)
        return;

    m_activeShadowDescSet = VK_NULL_HANDLE;

    if (m_perViewDescPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, m_perViewDescPool, nullptr);
        m_perViewDescPool = VK_NULL_HANDLE;
    }
    if (m_perViewDescSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, m_perViewDescSetLayout, nullptr);
        m_perViewDescSetLayout = VK_NULL_HANDLE;
    }
}

VkDescriptorSet InxVkCoreModular::AllocatePerViewDescriptorSet()
{
    if (m_perViewDescSetLayout == VK_NULL_HANDLE || m_perViewDescPool == VK_NULL_HANDLE) {
        INXLOG_ERROR("Per-view descriptor resources not initialized");
        return VK_NULL_HANDLE;
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_perViewDescPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_perViewDescSetLayout;

    VkDescriptorSet descSet = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(GetDevice(), &allocInfo, &descSet) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to allocate per-view descriptor set");
        return VK_NULL_HANDLE;
    }

    // Initialize with default (white) texture so shaders don't sample garbage
    ClearPerViewShadowMap(descSet);

    return descSet;
}

void InxVkCoreModular::UpdatePerViewShadowMap(VkDescriptorSet perViewDescSet, VkImageView shadowView,
                                              VkSampler shadowSampler, VkImageLayout imageLayout)
{
    if (perViewDescSet == VK_NULL_HANDLE || shadowView == VK_NULL_HANDLE || shadowSampler == VK_NULL_HANDLE)
        return;

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = imageLayout;
    imageInfo.imageView = shadowView;
    imageInfo.sampler = shadowSampler;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = perViewDescSet;
    write.dstBinding = 0;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(GetDevice(), 1, &write, 0, nullptr);
}

void InxVkCoreModular::ClearPerViewShadowMap(VkDescriptorSet perViewDescSet)
{
    if (perViewDescSet == VK_NULL_HANDLE)
        return;

    // Use default white texture so depth comparison = 1.0 → fully lit (no shadow)
    auto &descMgr = m_materialPipelineManager.GetDescriptorManager();
    VkImageView defaultView = descMgr.GetDefaultImageView();
    VkSampler defaultSampler = descMgr.GetDefaultSampler();

    if (defaultView == VK_NULL_HANDLE || defaultSampler == VK_NULL_HANDLE) {
        INXLOG_WARN("ClearPerViewShadowMap: default texture not available");
        return;
    }

    UpdatePerViewShadowMap(perViewDescSet, defaultView, defaultSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

} // namespace infernux
