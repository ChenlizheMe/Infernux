/**
 * @file InxVkCoreModular.cpp
 * @brief Implementation of the modular Vulkan core — init, lifecycle, texture/shader, internal
 *
 * Drawing methods → VkCoreDraw.cpp
 * Material/lighting/accessors → VkCoreMaterial.cpp
 */

#include "InxVkCoreModular.h"
#include "InxError.h"
#include "ProfileConfig.h"
#include "SceneRenderTarget.h"
#include "gui/GPUMaterialPreview.h"

#include <function/renderer/shader/ShaderProgram.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/scene/SceneRenderer.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace infernux
{

namespace
{

void DestroyLingeringMaterialPassPipelines(VkDevice device)
{
    if (device == VK_NULL_HANDLE) {
        return;
    }

    auto &assetRegistry = AssetRegistry::Instance();
    if (!assetRegistry.IsInitialized()) {
        return;
    }

    std::unordered_set<VkPipeline> destroyedPipelines;
    size_t destroyedCount = 0;

    for (const auto &material : assetRegistry.GetAllMaterials()) {
        if (!material) {
            continue;
        }

        for (int passIndex = 0; passIndex < static_cast<int>(ShaderCompileTarget::Count); ++passIndex) {
            const auto pass = static_cast<ShaderCompileTarget>(passIndex);
            // Shadow pipelines are owned by m_shadowPipelineCache and
            // destroyed in CleanupShadowPipeline() — skip here.
            if (pass == ShaderCompileTarget::Shadow) {
                material->ClearPassPipeline(pass);
                continue;
            }
            const VkPipeline pipeline = material->GetPassPipeline(pass);
            if (pipeline != VK_NULL_HANDLE) {
                if (destroyedPipelines.insert(pipeline).second) {
                    vkDestroyPipeline(device, pipeline, nullptr);
                    ++destroyedCount;
                }
            }

            material->ClearPassPipeline(pass);
        }
    }

    if (destroyedCount > 0) {
        INXLOG_WARN("InxVkCoreModular shutdown: destroyed ", destroyedCount,
                    " lingering material pass pipeline(s) outside MaterialPipelineManager ownership");
    }
}

} // namespace

// ============================================================================
// Constructor / Destructor
// ============================================================================

InxVkCoreModular::InxVkCoreModular(int maxFrameInFlight) : m_maxFramesInFlight(static_cast<uint32_t>(maxFrameInFlight))
{
    m_deletionQueue.Initialize(m_maxFramesInFlight);
}

InxVkCoreModular::~InxVkCoreModular()
{
    if (m_deviceContext.IsValid() && !m_shuttingDown) {
        m_deviceContext.WaitIdle();
    }

    // Flush all deferred deletions before tearing down subsystems
    m_deletionQueue.FlushAll();

    if (m_materialPipelineManagerInitialized) {
        m_materialPipelineManager.Shutdown(m_shuttingDown);
        m_materialPipelineManagerInitialized = false;
    }

    // Some material-owned auxiliary pass pipelines (typically shadow variants)
    // can outlive MaterialPipelineManager bookkeeping after invalidation or
    // hot-reload paths. Retire any pass handles still attached to live
    // materials before the device goes away.
    DestroyLingeringMaterialPassPipelines(GetDevice());

    // Cleanup shaders via VkShaderCache → VkPipelineManager.DestroyShaderModule
    // which also removes handles from tracking, preventing double-free.
    m_shaderCache.DestroyModules(m_pipelineManager);

    // During shutdown the device is already idle (drained once by ~InxRenderer).
    // Tell RAII members to skip their own vkDeviceWaitIdle calls.
    if (m_shuttingDown) {
        m_resourceManager.SetSkipWaitIdle(true);
        m_pipelineManager.SetSkipWaitIdle(true);
        m_swapchain.SetSkipWaitIdle(true);
    }

    // Cleanup shadow pipeline resources before general cleanup
    CleanupShadowPipeline();

    // Cleanup per-view descriptor resources (multi-camera shadow)
    DestroyPerViewDescriptorResources();

    // Cleanup engine globals descriptor resources
    DestroyGlobalsDescriptorResources();

    // Explicit destruction in controlled order (avoids double-free from
    // RAII reverse-declaration order when handles are shared across systems).
    m_perObjectBuffers.clear();
    m_sharedMeshBuffers.clear();
    m_textureCache.Clear();
    m_shaderCache.Clear();

    m_lightingUboBuffers.clear();
    m_lightingUboMapped.clear();
    m_materialUboBuffers.clear();
    m_materialUboMapped.clear();
    m_uniformBuffers.clear();
    m_globalsBuffers.clear();
    m_gpuMaterialPreview.reset();

    m_commandBuffers.clear();
    m_depthImage.reset();

    m_renderGraph.Destroy();
    m_resourceManager.Destroy();

    // RenderGraph::Destroy() and MaterialPipelineManager::Shutdown()
    // already destroyed render passes, layouts, and descriptor set
    // layouts that VkPipelineManager may have tracked. Keep pipeline
    // tracking alive so any leftovers still get reclaimed here.
    m_pipelineManager.ClearTrackedNonPipelineResources();
    m_pipelineManager.Destroy();

    m_swapchain.Destroy();

    // Some shutdown paths defer frees into the frame deletion queue. Flush one
    // final time after all subsystems have torn down but before destroying the
    // allocator/device.
    m_deletionQueue.FlushAll();

    m_deviceContext.Destroy();

    // All RAII destructors that fire after this point will find VK_NULL_HANDLE
    // in their stored device and skip Vulkan calls.
}

// ============================================================================
// Initialization
// ============================================================================

void InxVkCoreModular::Init(InxAppMetadata appMetaData, InxAppMetadata rendererMetaData, uint32_t vkWindowExtCount,
                            const char **vkWindowExts)
{
    INXLOG_INFO("Initializing InxVkCoreModular...");

    // Configure device (store for use in PrepareSurface)
    m_deviceConfig.appName = appMetaData.appName ? appMetaData.appName : "Infernux App";
    m_deviceConfig.engineName = rendererMetaData.appName ? rendererMetaData.appName : "Infernux";

    // Initialize instance only (device will be created in PrepareSurface after surface is available)
    if (!m_deviceContext.InitializeInstance(m_deviceConfig)) {
        INXLOG_ERROR("Failed to initialize Vulkan instance");
        return;
    }

    // Store instance for InxRenderer access
    m_instance = m_deviceContext.GetInstance();

    INXLOG_INFO("InxVkCoreModular instance initialized successfully");
}

void InxVkCoreModular::PrepareSurface()
{
    // Surface should be set by InxRenderer before calling this
    if (m_surface == VK_NULL_HANDLE) {
        INXLOG_ERROR("Surface not set. Call CreateSurface first.");
        return;
    }

    // Complete device initialization with the surface
    if (!m_deviceContext.InitializeDevice(m_surface, m_deviceConfig)) {
        INXLOG_ERROR("Failed to initialize Vulkan device");
        return;
    }

    // Now that device is ready, initialize resource manager
    if (!m_resourceManager.Initialize(m_deviceContext)) {
        INXLOG_ERROR("Failed to initialize resource manager");
        return;
    }

    // Initialize pipeline manager
    m_pipelineManager.Initialize(m_deviceContext.GetDevice());

    // Initialize render graph
    m_renderGraph.Initialize(&m_deviceContext, &m_pipelineManager);

    // Get extent from surface capabilities
    auto swapchainSupport = m_deviceContext.QuerySwapchainSupport();
    uint32_t width = swapchainSupport.capabilities.currentExtent.width;
    uint32_t height = swapchainSupport.capabilities.currentExtent.height;

    if (width == std::numeric_limits<uint32_t>::max() || height == std::numeric_limits<uint32_t>::max() || width == 0 ||
        height == 0) {
        width = (m_windowWidth > 0) ? m_windowWidth : swapchainSupport.capabilities.minImageExtent.width;
        height = (m_windowHeight > 0) ? m_windowHeight : swapchainSupport.capabilities.minImageExtent.height;

        width = std::clamp(width, swapchainSupport.capabilities.minImageExtent.width,
                           swapchainSupport.capabilities.maxImageExtent.width);
        height = std::clamp(height, swapchainSupport.capabilities.minImageExtent.height,
                            swapchainSupport.capabilities.maxImageExtent.height);
    }

    // Create swapchain
    if (!m_swapchain.Create(m_deviceContext, width, height)) {
        INXLOG_ERROR("Failed to create swapchain");
        return;
    }

    // Create depth resources
    CreateDepthResources();

    // Create uniform buffers
    CreateUniformBuffers();

    // Allocate command buffers
    m_commandBuffers.resize(m_maxFramesInFlight);
    for (uint32_t i = 0; i < m_maxFramesInFlight; ++i) {
        auto alloc = m_resourceManager.AllocatePrimaryCommandBuffer();
        m_commandBuffers[i] = alloc.cmdBuffer;
    }

    INXLOG_INFO("Surface prepared successfully");
}

void InxVkCoreModular::PreparePipeline()
{
    // Create default white texture
    m_textureCache.CreateDefaultWhiteTexture("white", m_resourceManager);

    // Create default flat normal texture (0.5, 0.5, 1.0 = tangent-space (0,0,1))
    m_textureCache.CreateSolidColorTexture("_default_normal", 128, 128, 255, 255, VK_FORMAT_R8G8B8A8_UNORM,
                                           m_resourceManager);
    INXLOG_INFO("Created default flat normal texture: _default_normal");

    // Initialize material system (default material + pipelines)
    InitializeMaterialSystem();

    // Create per-view descriptor set layout and pool (multi-camera shadow isolation).
    // Must be after InitializeMaterialSystem so default textures are available.
    CreatePerViewDescriptorResources();

    // Create shadow depth sampler eagerly so that it is available when
    // RefreshPerViewShadowDescriptor runs (before the first DrawShadowCasters).
    if (m_shadowDepthSampler == VK_NULL_HANDLE) {
        CreateShadowDepthSampler();
    }

    INXLOG_INFO("Pipeline prepared successfully");
}

// ============================================================================
// Texture Management (delegates to VkTextureCache)
// ============================================================================

void InxVkCoreModular::CreateTextureImage(std::string name, std::string path)
{
    m_textureCache.CreateTextureImage(name, path, m_resourceManager);
}

void InxVkCoreModular::CreateDefaultWhiteTexture(std::string name)
{
    m_textureCache.CreateDefaultWhiteTexture(name, m_resourceManager);
}

void InxVkCoreModular::LoadTexture(const std::string &name, const std::string &path)
{
    m_textureCache.CreateTextureImage(name, path, m_resourceManager);
}

// ============================================================================
// Shader and Pipeline Management (delegates to VkShaderCache)
// ============================================================================

void InxVkCoreModular::LoadShader(const char *name, const std::vector<char> &spirvCode, const char *type)
{
    m_shaderCache.LoadShader(name, spirvCode, type, m_pipelineManager);
}

void InxVkCoreModular::StoreShaderRenderMeta(const std::string &shaderId, const std::string &cullMode,
                                             const std::string &depthWrite, const std::string &depthTest,
                                             const std::string &blend, int queue, const std::string &passTag,
                                             const std::string &stencil, const std::string &alphaClip)
{
    m_shaderCache.StoreRenderMeta(shaderId, cullMode, depthWrite, depthTest, blend, queue, passTag, stencil, alphaClip);
}

void InxVkCoreModular::UnloadShader(const char *name)
{
    m_shaderCache.UnloadShader(name, GetDevice());
}

bool InxVkCoreModular::HasShader(const std::string &name, const std::string &type) const
{
    return m_shaderCache.HasShader(name, type);
}

void InxVkCoreModular::InvalidateShaderCache(const std::string &shaderId)
{
    INXLOG_INFO("Invalidating shader cache for: ", shaderId);

    // Wait for GPU to finish any pending work using this shader
    m_deviceContext.WaitIdle();

    // Remove shader programs from cache that contain this shader
    m_shaderCache.GetProgramCache().RemoveProgramsContainingShader(shaderId);

    // Invalidate all materials using this shader in MaterialPipelineManager
    if (m_materialPipelineManagerInitialized) {
        m_materialPipelineManager.InvalidateMaterialsUsingShader(shaderId);
    }

    // Invalidate cached shadow pipelines that reference this shader
    {
        VkDevice dev = GetDevice();
        for (auto it = m_shadowPipelineCache.begin(); it != m_shadowPipelineCache.end();) {
            if (it->first.find(shaderId) != std::string::npos) {
                if (it->second != VK_NULL_HANDLE)
                    vkDestroyPipeline(dev, it->second, nullptr);
                it = m_shadowPipelineCache.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Also unload the shader module so it gets recreated
    m_shaderCache.UnloadShader(shaderId.c_str(), GetDevice());

    INXLOG_INFO("Shader cache invalidated for: ", shaderId);
}

void InxVkCoreModular::InvalidateTextureCache(const std::string &textureIdentifier)
{
    // The identifier may be a GUID or a file path.
    // Cache keys are GUID-based ("GUID::srgb" or "GUID::unorm").
    // Resolve the identifier to a GUID for matching against cache keys.
    std::string matchKey = textureIdentifier;
    std::replace(matchKey.begin(), matchKey.end(), '\\', '/');

    auto *adb = AssetRegistry::Instance().GetAssetDatabase();
    if (adb) {
        // If the identifier looks like a path, resolve it to a GUID
        std::string guid = adb->GetGuidFromPath(textureIdentifier);
        if (!guid.empty())
            matchKey = guid;
        // If it was already a GUID, GetGuidFromPath returns empty → keep as-is
    }

    INXLOG_INFO("Invalidating texture cache for: ", matchKey);

    // Wait for GPU to finish using the texture
    m_deviceContext.WaitIdle();

    // Evict all cached variants for this GUID/path
    size_t evicted = m_textureCache.EvictByPrefix(matchKey);
    (void)evicted;

    INXLOG_INFO("Texture cache invalidated for: ", matchKey);
}

void InxVkCoreModular::RemoveMaterialPipeline(const std::string &materialName)
{
    if (m_materialPipelineManagerInitialized) {
        m_deviceContext.WaitIdle();
        m_materialPipelineManager.RemoveRenderData(materialName);
        INXLOG_INFO("Removed material pipeline render data for: ", materialName);
    }
}

// ============================================================================
// Command Buffer Utilities
// ============================================================================

VkCommandBuffer InxVkCoreModular::BeginSingleTimeCommands()
{
    return m_resourceManager.BeginSingleTimeCommands();
}

void InxVkCoreModular::EndSingleTimeCommands(VkCommandBuffer commandBuffer)
{
    m_resourceManager.EndSingleTimeCommands(commandBuffer);
}

// ============================================================================
// Render Callbacks (RenderGraph-based)
// ============================================================================

void InxVkCoreModular::SetRenderGraphExecutor(std::function<void(VkCommandBuffer cmdBuf)> executor)
{
    m_renderGraphExecutor = std::move(executor);
}

void InxVkCoreModular::SetGuiRenderCallback(std::function<void(vk::RenderContext &ctx)> callback)
{
    m_guiRenderCallback = std::move(callback);
}

// ============================================================================
// Internal Methods
// ============================================================================

void InxVkCoreModular::RecreateSwapchain()
{
    // Wait for device idle
    m_deviceContext.WaitIdle();

    // Cleanup old depth resources
    m_depthImage.reset();

    // Get new extent from surface capabilities
    auto swapchainSupport = m_deviceContext.QuerySwapchainSupport();
    uint32_t width = swapchainSupport.capabilities.currentExtent.width;
    uint32_t height = swapchainSupport.capabilities.currentExtent.height;

    if (width == std::numeric_limits<uint32_t>::max() || height == std::numeric_limits<uint32_t>::max() || width == 0 ||
        height == 0) {
        width = (m_windowWidth > 0) ? m_windowWidth : swapchainSupport.capabilities.minImageExtent.width;
        height = (m_windowHeight > 0) ? m_windowHeight : swapchainSupport.capabilities.minImageExtent.height;

        width = std::clamp(width, swapchainSupport.capabilities.minImageExtent.width,
                           swapchainSupport.capabilities.maxImageExtent.width);
        height = std::clamp(height, swapchainSupport.capabilities.minImageExtent.height,
                            swapchainSupport.capabilities.maxImageExtent.height);
    }

    // Handle special case (some window managers report invalid extents)
    if (width == 0 || height == 0) {
        // Cannot recreate, wait for valid extent
        return;
    }

    // Recreate swapchain
    m_swapchain.Recreate(m_deviceContext, width, height);

    // Recreate depth resources
    CreateDepthResources();
}

void InxVkCoreModular::CreateDepthResources()
{
    VkExtent2D extent = m_swapchain.GetExtent();
    VkFormat depthFormat = m_deviceContext.FindDepthFormat();

    m_depthImage = m_resourceManager.CreateDepthBuffer(extent.width, extent.height, depthFormat);
}

void InxVkCoreModular::CreateUniformBuffers()
{
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    m_uniformBuffers.resize(m_maxFramesInFlight);
    for (size_t i = 0; i < m_maxFramesInFlight; ++i) {
        m_uniformBuffers[i] = m_resourceManager.CreateUniformBuffer(bufferSize);
    }

    // Create lighting UBO buffers (binding 1)
    VkDeviceSize lightingUboSize = sizeof(ShaderLightingUBO);
    m_lightingUboBuffers.resize(m_maxFramesInFlight);
    m_lightingUboMapped.resize(m_maxFramesInFlight, nullptr);

    for (size_t i = 0; i < m_maxFramesInFlight; ++i) {
        m_lightingUboBuffers[i] = m_resourceManager.CreateUniformBuffer(lightingUboSize);
        if (m_lightingUboBuffers[i]) {
            m_lightingUboMapped[i] = m_lightingUboBuffers[i]->Map(0, lightingUboSize);
            if (m_lightingUboMapped[i]) {
                // Initialize with default ambient lighting (neutral white)
                ShaderLightingUBO defaultLighting{};
                defaultLighting.lightCounts = glm::ivec4(0, 0, 0, 0);
                defaultLighting.ambientColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.3f); // White ambient, low intensity
                defaultLighting.cameraPos = glm::vec4(0.0f, 0.0f, 5.0f, 1.0f);
                std::memcpy(m_lightingUboMapped[i], &defaultLighting, lightingUboSize);
            }
        }
    }
    INXLOG_INFO("Created lighting UBO buffers: ", lightingUboSize, " bytes x ", m_maxFramesInFlight, " frames");

    // Create default material UBO buffers (binding 2)
    // 256 bytes is a safe default for the fallback per-frame material UBO.
    // Per-material UBOs use reflection-derived sizes via MaterialDescriptorManager.
    constexpr size_t materialUboSize = 256;
    m_materialUboBuffers.resize(m_maxFramesInFlight);
    m_materialUboMapped.resize(m_maxFramesInFlight, nullptr);

    for (size_t i = 0; i < m_maxFramesInFlight; ++i) {
        m_materialUboBuffers[i] = m_resourceManager.CreateUniformBuffer(materialUboSize);
        if (m_materialUboBuffers[i]) {
            m_materialUboMapped[i] = m_materialUboBuffers[i]->Map(0, materialUboSize);
            if (m_materialUboMapped[i]) {
                std::memset(m_materialUboMapped[i], 0, materialUboSize);
            }
        }
    }

    // Create engine globals UBO buffers (set 2, binding 0)
    CreateGlobalsBuffers();
    CreateGlobalsDescriptorResources();
}

void InxVkCoreModular::RecordCommandBuffer(uint32_t imageIndex)
{
#if INFERNUX_FRAME_PROFILE
    using Clock = std::chrono::high_resolution_clock;
    auto _tPrev = Clock::now();
    auto _tNow = _tPrev;
#endif

    VkCommandBuffer cmdBuf = m_commandBuffers[m_currentFrame];

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(cmdBuf, &beginInfo) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to begin recording command buffer");
        return;
    }

    // ========================================================================
    // Inline UBO Updates (Fix 1: replaces CPU-side memcpy)
    // ========================================================================
    if (m_uboDirty) {
        CmdUpdateUniformBuffer(cmdBuf, m_stagedUBO.view, m_stagedUBO.proj);
        m_uboDirty = false;
    }
    CmdUpdateLightingUBO(cmdBuf);
    CmdUpdateShadowUBO(cmdBuf);
    CmdUpdateGlobals(cmdBuf);
#if INFERNUX_FRAME_PROFILE
    _tNow = Clock::now();
    m_drawSubMs[4] += std::chrono::duration<double, std::milli>(_tNow - _tPrev).count();
    _tPrev = _tNow;
#endif

    // ========================================================================
    // Execute scene render graph (offscreen scene rendering)
    // ========================================================================
    if (m_renderGraphExecutor) {
        m_renderGraphExecutor(cmdBuf);
    }
#if INFERNUX_FRAME_PROFILE
    _tNow = Clock::now();
    m_drawSubMs[5] += std::chrono::duration<double, std::milli>(_tNow - _tPrev).count();
    _tPrev = _tNow;
#endif

    // ========================================================================
    // Post-Scene-Render Callback (OutlineRenderer injection point)
    // ========================================================================
    if (m_postSceneRenderCallback) {
        m_postSceneRenderCallback(cmdBuf, drawCalls());
    }
#if INFERNUX_FRAME_PROFILE
    _tNow = Clock::now();
    m_drawSubMs[6] += std::chrono::duration<double, std::milli>(_tNow - _tPrev).count();
    _tPrev = _tNow;
#endif

    // ========================================================================
    // Swapchain GUI Pass via RenderGraph
    // ========================================================================
    m_renderGraph.Reset();

    VkImage swapchainImage = m_swapchain.GetImage(imageIndex);
    VkImageView swapchainView = m_swapchain.GetImageView(imageIndex);
    VkExtent2D extent = m_swapchain.GetExtent();
    VkFormat format = m_swapchain.GetImageFormat();

    vk::ResourceHandle backbuffer =
        m_renderGraph.SetBackbuffer(swapchainImage, swapchainView, format, extent.width, extent.height,
                                    VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_LAYOUT_UNDEFINED);

    m_renderGraph.SetBackbufferFinalLayout(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    auto guiCallback = m_guiRenderCallback;

    m_renderGraph.AddPass("GUI", [backbuffer, extent, guiCallback](vk::PassBuilder &builder) {
        builder.WriteColor(backbuffer, 0);
        builder.SetRenderArea(extent.width, extent.height);
        builder.SetClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        return [guiCallback, extent](vk::RenderContext &ctx) {
            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = static_cast<float>(extent.width);
            viewport.height = static_cast<float>(extent.height);
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            ctx.SetViewport(viewport);

            VkRect2D scissor{};
            scissor.offset = {0, 0};
            scissor.extent = extent;
            ctx.SetScissor(scissor);

            if (guiCallback) {
                guiCallback(ctx);
            }
        };
    });

    m_renderGraph.SetOutput(backbuffer);

    if (!m_renderGraph.Compile()) {
        INXLOG_ERROR("Failed to compile swapchain render graph");
        vkEndCommandBuffer(cmdBuf);
        return;
    }

    m_renderGraph.Execute(cmdBuf);
#if INFERNUX_FRAME_PROFILE
    _tNow = Clock::now();
    m_drawSubMs[7] += std::chrono::duration<double, std::milli>(_tNow - _tPrev).count();
#endif

    if (vkEndCommandBuffer(cmdBuf) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to record command buffer");
    }
}

void InxVkCoreModular::UpdateUniformBuffer(uint32_t currentImage, const float *viewPos, const float *viewLookAt,
                                           const float *viewUp)
{
    UniformBufferObject ubo{};
    ubo.model = glm::mat4(1.0f); // Identity - actual model matrices are per-object

    // Use camera's actual matrices from SceneRenderBridge for consistency with picking
    SceneRenderBridge &bridge = SceneRenderBridge::Instance();
    Camera *activeCamera = bridge.GetEditorCamera();

    if (activeCamera) {
        // Use camera's actual view and projection matrices
        ubo.view = activeCamera->GetViewMatrix();
        ubo.proj = activeCamera->GetProjectionMatrix(); // Already has Y-flip for Vulkan
    } else {
        // Fallback to legacy calculation
        glm::vec3 eye(viewPos[0], viewPos[1], viewPos[2]);
        glm::vec3 center(viewLookAt[0], viewLookAt[1], viewLookAt[2]);
        glm::vec3 up(viewUp[0], viewUp[1], viewUp[2]);
        ubo.view = glm::lookAt(eye, center, up);

        VkExtent2D extent = m_swapchain.GetExtent();
        float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
        if (m_sceneRenderTargetWidth > 0 && m_sceneRenderTargetHeight > 0) {
            aspect = static_cast<float>(m_sceneRenderTargetWidth) / static_cast<float>(m_sceneRenderTargetHeight);
        }
        ubo.proj = glm::perspective(glm::radians(60.0f), aspect, 0.01f, 1000.0f);
        ubo.proj[1][1] *= -1; // Flip Y for Vulkan
    }

    // Stage the UBO data — the actual GPU write happens inline in the
    // command buffer via CmdUpdateUniformBuffer() during RecordCommandBuffer().
    // This eliminates the CPU→GPU race on m_uniformBuffers[0] that previously
    // required vkDeviceWaitIdle() for correctness.
    m_stagedUBO = ubo;
    m_uboDirty = true;
}

// ============================================================================
// Frame Synchronization & Deferred Deletion
// ============================================================================

void InxVkCoreModular::WaitForCurrentFrame()
{
    m_swapchain.WaitForFrame();
}

void InxVkCoreModular::TickDeletionQueue()
{
    m_deletionQueue.Tick();
}

void InxVkCoreModular::FlushDeletionQueue()
{
    m_deletionQueue.FlushAll();
}

void InxVkCoreModular::DeferDeletion(std::function<void()> deleter)
{
    m_deletionQueue.Push(std::move(deleter));
}

} // namespace infernux
