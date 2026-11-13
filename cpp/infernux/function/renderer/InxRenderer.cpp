#include "InxRenderer.h"
#include "CaptureService.h"
#include "gui/PreviewImage.h"
#include <core/math/DepthConvention.h>
// Explicit includes for all subsystem types that were forward-declared in
// InxRenderer.h. These are required here so that unique_ptr destructors,
// make_unique calls, and member-function calls all have complete types.
#include "EditorGizmos.h"
#include "EditorTools.h"
#include "GizmosDrawCallBuffer.h"
#include "InxVkCoreModular.h"
#include "MsaaPolicy.h"
#include "OutlineRenderer.h"
#include "ScenePickingService.h"
#include "SceneRenderGraph.h"
#include "SceneRenderTarget.h"
#include "ScriptableRenderContext.h"
#include "gui/InxGUI.h"
#include "gui/InxGUIContext.h"
#include "gui/InxGUISemantics.h"
#include "gui/InxScreenUIRenderer.h"
#include "gui/UIMaterialTextureResolver.h"
#include "particle/ParticleGpuBounds.h"
#include "particle/ParticleGpuCollisionScene.h"
#include "particle/ParticleGpuCuller.h"
#include "particle/ParticleGpuDrawRegistry.h"
#include "particle/ParticleGpuMigrator.h"
#include "particle/ParticleGpuRibbonRenderer.h"
#include "particle/ParticleGpuRibbonTopology.h"
#include "particle/ParticleGpuSorter.h"
#include "particle/ParticleGpuSystemManager.h"
#include "particle/ParticleSceneSources.h"
#include "rhi/RhiComputeHost.h"
#include "rhi/RhiRenderTexture.h"
#include "vk/RenderGraph.h"
#include "vk/RhiVulkanTypes.h"
#include "vk/VkHandle.h"
#include "vk/VmaContext.h"
#include <SDL3/SDL.h>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <core/config/EngineConfig.h>
#include <core/config/MathConstants.h>
#include <core/threading/FrameWorkOwner.h>
#include <core/threading/JobSystem.h>
#include <cstdlib>
#include <cstring>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxFileLoader/InxShaderLoader.hpp>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxMesh/InxMesh.h>
#include <function/resources/ShaderAsset/SlangNativeModule.h>
#include <function/scene/Camera.h>
#include <function/scene/Collider.h>
#include <function/scene/GameObject.h>
#include <function/scene/Light.h>
#include <function/scene/LightingData.h>
#include <function/scene/MeshRenderer.h>
#include <function/scene/PrimitiveMeshes.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <function/scene/SceneRenderBridge.h>
#include <function/scene/SkinnedMeshRenderer.h>
#include <function/scene/TransformECSStore.h>
#include <function/scene/physics/PhysicsECSStore.h>
#include <function/scene/physics/PhysicsWorld.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <platform/filesystem/InxPath.h>
#include <platform/window/InxView.h>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace infernux
{
namespace
{

uint64_t ParticleCollisionStableHash(std::string_view value) noexcept
{
    uint64_t hash = 1469598103934665603ull;
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash;
}

void ParticleCollisionHashAppend(uint64_t &hash, uint64_t value) noexcept
{
    for (uint32_t byte = 0; byte < 8; ++byte) {
        hash ^= static_cast<uint8_t>(value >> (byte * 8u));
        hash *= 1099511628211ull;
    }
}

void StoreParticleCollisionAffine(particle::GpuParticleAffineTransform &destination, const glm::mat4 &matrix)
{
    destination.row0 = {matrix[0][0], matrix[1][0], matrix[2][0], matrix[3][0]};
    destination.row1 = {matrix[0][1], matrix[1][1], matrix[2][1], matrix[3][1]};
    destination.row2 = {matrix[0][2], matrix[1][2], matrix[2][2], matrix[3][2]};
}

void CopyParticleCollisionAffine(particle::GpuParticleAffineTransform &destination, const std::array<float, 12> &source)
{
    std::copy_n(source.begin(), 4, destination.row0.begin());
    std::copy_n(source.begin() + 4, 4, destination.row1.begin());
    std::copy_n(source.begin() + 8, 4, destination.row2.begin());
}

void CopyParticleCollisionAffine(std::array<float, 12> &destination, const particle::GpuParticleAffineTransform &source)
{
    std::copy(source.row0.begin(), source.row0.end(), destination.begin());
    std::copy(source.row1.begin(), source.row1.end(), destination.begin() + 4);
    std::copy(source.row2.begin(), source.row2.end(), destination.begin() + 8);
}

void StoreParticleCollisionAabb(particle::GpuParticleColliderRecord &record, const glm::mat4 &localToWorld,
                                const glm::vec3 &localMin, const glm::vec3 &localMax)
{
    const glm::vec3 localCenter = (localMin + localMax) * 0.5f;
    const glm::vec3 localExtent = (localMax - localMin) * 0.5f;
    const glm::vec3 worldCenter = glm::vec3(localToWorld * glm::vec4(localCenter, 1.0f));
    const glm::mat3 linear(localToWorld);
    const glm::vec3 worldExtent =
        glm::abs(linear[0]) * localExtent.x + glm::abs(linear[1]) * localExtent.y + glm::abs(linear[2]) * localExtent.z;
    const glm::vec3 worldMin = worldCenter - worldExtent;
    const glm::vec3 worldMax = worldCenter + worldExtent;
    record.worldAabbMin = {worldMin.x, worldMin.y, worldMin.z, 0.0f};
    record.worldAabbMax = {worldMax.x, worldMax.y, worldMax.z, 0.0f};
}

template <typename Program, size_t Count> struct CompiledParticleProgram
{
    std::array<particle::CompiledParticleShader, Count> shaders;

    [[nodiscard]] Program View() const noexcept
    {
        return View(std::make_index_sequence<Count>{});
    }

  private:
    template <size_t... Index> Program View(std::index_sequence<Index...>) const noexcept
    {
        return {shaders[Index]...};
    }
};

using CompiledParticleSortProgram = CompiledParticleProgram<particle::GpuParticleSortProgram, 5>;
using CompiledParticleCullProgram = CompiledParticleProgram<particle::GpuParticleCullProgram, 3>;
using CompiledParticleBoundsProgram = CompiledParticleProgram<particle::GpuParticleBoundsProgram, 3>;
using CompiledParticleMigrationProgram = CompiledParticleProgram<particle::GpuParticleMigrationProgram, 2>;
using CompiledParticleSpawnProgram = CompiledParticleProgram<particle::GpuParticleSpawnProgram, 2>;
using CompiledParticleRibbonTopologyProgram = CompiledParticleProgram<particle::GpuParticleRibbonProgram, 5>;
struct CompiledParticleRibbonRenderProgram
{
    std::array<particle::ShaderBytecodeStorage, 4> shaders;
    particle::GpuParticleRibbonRenderProgram View() const noexcept
    {
        return {shaders[0].View(), shaders[1].View(), shaders[2].View(), shaders[3].View()};
    }
};

particle::CompiledParticleShader CompileParticleShader(std::string_view source, const char *name,
                                                       ShaderCodeTarget target = ShaderCodeTarget::SpirV)
{
    return std::make_shared<const ShaderTargetCode>(
        CompileSlangNativeModule({name, name, std::string(source)}, target));
}

struct CompiledParticleProgramBundle
{
    CompiledParticleSortProgram sort;
    CompiledParticleCullProgram cull;
    CompiledParticleBoundsProgram bounds;
    CompiledParticleMigrationProgram migration;
    CompiledParticleSpawnProgram spawn;
    CompiledParticleRibbonTopologyProgram ribbonTopology;
    CompiledParticleRibbonRenderProgram ribbonRender;
    particle::CompiledParticleShader stateGather;

    [[nodiscard]] bool IsValid() const noexcept
    {
        return sort.View().IsValid() && cull.View().IsValid() && bounds.View().IsValid() &&
               migration.View().IsValid() && spawn.View().IsValid() && ribbonTopology.View().IsValid() &&
               ribbonRender.View().IsValid() && particle::IsCompiledParticleShaderValid(stateGather);
    }
};

CompiledParticleSortProgram CompileParticleSortProgram()
{
    const std::array<std::pair<std::string_view, const char *>, 5> sources = {{
        {particle::GpuParticleSortShaderSources::Small(), "Infernux/ParticleSortSmall.slang"},
        {particle::GpuParticleSortShaderSources::Generate(), "Infernux/ParticleSortGenerate.slang"},
        {particle::GpuParticleSortShaderSources::Histogram(), "Infernux/ParticleSortHistogram.slang"},
        {particle::GpuParticleSortShaderSources::Scan(), "Infernux/ParticleSortScan.slang"},
        {particle::GpuParticleSortShaderSources::Scatter(), "Infernux/ParticleSortScatter.slang"},
    }};
    CompiledParticleSortProgram result;
    for (size_t index = 0; index < sources.size(); ++index) {
        result.shaders[index] = CompileParticleShader(sources[index].first, sources[index].second);
    }
    return result;
}

CompiledParticleCullProgram CompileParticleCullProgram()
{
    const std::array<std::pair<std::string_view, const char *>, 3> sources = {{
        {particle::GpuParticleCullShaderSources::Reset(), "Infernux/ParticleCullReset.slang"},
        {particle::GpuParticleCullShaderSources::Cull(), "Infernux/ParticleCull.slang"},
        {particle::GpuParticleCullShaderSources::Finalize(), "Infernux/ParticleCullFinalize.slang"},
    }};
    CompiledParticleCullProgram result;
    for (size_t index = 0; index < sources.size(); ++index) {
        result.shaders[index] = CompileParticleShader(sources[index].first, sources[index].second);
    }
    return result;
}

CompiledParticleBoundsProgram CompileParticleBoundsProgram()
{
    const std::array<std::pair<std::string_view, const char *>, 3> sources = {{
        {particle::GpuParticleBoundsShaderSources::Prepare(), "Infernux/ParticleVisibilityPrepare.slang"},
        {particle::GpuParticleBoundsShaderSources::Reset(), "Infernux/ParticleBoundsReset.slang"},
        {particle::GpuParticleBoundsShaderSources::Reduce(), "Infernux/ParticleBoundsReduce.slang"},
    }};
    CompiledParticleBoundsProgram result;
    for (size_t index = 0; index < sources.size(); ++index) {
        result.shaders[index] = CompileParticleShader(sources[index].first, sources[index].second);
    }
    return result;
}

CompiledParticleMigrationProgram CompileParticleMigrationProgram()
{
    const std::array<std::pair<std::string_view, const char *>, 2> sources = {{
        {particle::GpuParticleMigrationShaderSources::Reset(), "Infernux/ParticleMigrationReset.slang"},
        {particle::GpuParticleMigrationShaderSources::Migrate(), "Infernux/ParticleMigration.slang"},
    }};
    CompiledParticleMigrationProgram result;
    for (size_t index = 0; index < sources.size(); ++index) {
        result.shaders[index] = CompileParticleShader(sources[index].first, sources[index].second);
    }
    return result;
}

CompiledParticleSpawnProgram CompileParticleSpawnProgram()
{
    const std::array<std::pair<std::string_view, const char *>, 2> sources = {{
        {particle::GpuParticleSpawnShaderSources::Advance(), "Infernux/ParticleSpawnAdvance.slang"},
        {particle::GpuParticleSpawnShaderSources::Prepare(), "Infernux/ParticleSpawnPrepare.slang"},
    }};
    CompiledParticleSpawnProgram result;
    for (size_t index = 0; index < sources.size(); ++index) {
        result.shaders[index] = CompileParticleShader(sources[index].first, sources[index].second);
    }
    return result;
}

CompiledParticleRibbonTopologyProgram CompileParticleRibbonTopologyProgram()
{
    const std::array<std::pair<std::string_view, const char *>, 5> sources = {{
        {particle::GpuParticleRibbonShaderSources::Reset(), "Infernux/ParticleRibbonReset.slang"},
        {particle::GpuParticleRibbonShaderSources::Initialize(), "Infernux/ParticleRibbonInitialize.slang"},
        {particle::GpuParticleRibbonShaderSources::Histogram(), "Infernux/ParticleRibbonHistogram.slang"},
        {particle::GpuParticleRibbonShaderSources::Scan(), "Infernux/ParticleRibbonScan.slang"},
        {particle::GpuParticleRibbonShaderSources::Scatter(), "Infernux/ParticleRibbonScatter.slang"},
    }};
    CompiledParticleRibbonTopologyProgram result;
    for (size_t index = 0; index < sources.size(); ++index) {
        result.shaders[index] = CompileParticleShader(sources[index].first, sources[index].second);
    }
    return result;
}

CompiledParticleRibbonRenderProgram CompileParticleRibbonRenderProgram()
{
    const std::array<std::pair<std::string_view, const char *>, 4> sources = {{
        {particle::GpuParticleRibbonRenderShaderSources::Vertex(), "Infernux/ParticleRibbon.slang"},
        {particle::GpuParticleRibbonRenderShaderSources::PickingFragment(), "Infernux/ParticleRibbonPicking.slang"},
        {particle::GpuParticleRibbonRenderShaderSources::MotionVertex(), "Infernux/ParticleRibbonMotionVertex.slang"},
        {particle::GpuParticleRibbonRenderShaderSources::MotionFragment(),
         "Infernux/ParticleRibbonMotionFragment.slang"},
    }};
    CompiledParticleRibbonRenderProgram result;
    for (size_t index = 0; index < sources.size(); ++index) {
        const auto compiled = CompileParticleShader(sources[index].first, sources[index].second);
        const auto &bytes = compiled->entries.front().code;
        result.shaders[index].Assign(particle::ShaderBytecode::FromSpirV(
            reinterpret_cast<const uint32_t *>(bytes.data()), bytes.size() / sizeof(uint32_t)));
    }
    return result;
}

CompiledParticleProgramBundle CompileParticleProgramBundle()
{
    CompiledParticleProgramBundle result;
    result.sort = CompileParticleSortProgram();
    result.cull = CompileParticleCullProgram();
    result.bounds = CompileParticleBoundsProgram();
    result.migration = CompileParticleMigrationProgram();
    result.spawn = CompileParticleSpawnProgram();
    result.ribbonTopology = CompileParticleRibbonTopologyProgram();
    result.ribbonRender = CompileParticleRibbonRenderProgram();
    result.stateGather =
        CompileParticleShader(particle::GpuParticleStateGatherShaderSource(), "Infernux/ParticleStateGather.slang");
    return result;
}

uint32_t ResolveMaxFramesInFlight() noexcept
{
    auto &config = EngineConfig::Get();
    uint32_t frames = std::clamp(config.maxFramesInFlight, 1u, EngineConfig::kFrameSlotCapacity);
    if (const char *raw = std::getenv("INFERNUX_MAX_FRAMES_IN_FLIGHT")) {
        const char *end = raw + std::strlen(raw);
        uint32_t requested = 0;
        const auto parsed = std::from_chars(raw, end, requested);
        if (parsed.ec == std::errc{} && parsed.ptr == end && requested >= 1u &&
            requested <= EngineConfig::kFrameSlotCapacity)
            frames = requested;
    }
    config.maxFramesInFlight = frames;
    return frames;
}

} // namespace

struct ParticleProgramBootstrap
{
    CompiledParticleProgramBundle programs;
    JobHandle job;
};

std::shared_ptr<rhi::RenderTexture> InxRenderer::CreateRenderTexture(const rhi::RenderTextureDesc &description,
                                                                     const std::string &assetGuid)
{
    if (!m_vkCore || !m_vkCore->GetDeviceContext().IsValid())
        throw std::runtime_error("RenderTexture requires an initialized graphical engine");
    const bool relative = description.sizeMode == rhi::RenderTextureSizeMode::Relative;
    if (relative && (!m_gameRenderTarget || !m_gameRenderTarget->IsReady()))
        throw std::runtime_error("Relative RenderTexture requires the Game output resolution to be initialized");
    static std::atomic<uint64_t> nextIdentity{1};
    auto texture = std::make_shared<rhi::RenderTexture>(
        m_vkCore->GetDeviceContext().GetRhiDevice(),
        "runtime/render-texture/" + std::to_string(nextIdentity.fetch_add(1, std::memory_order_relaxed)), description,
        relative ? m_gameRenderTarget->GetWidth() : 0, relative ? m_gameRenderTarget->GetHeight() : 0, assetGuid);
    if (relative) {
        m_relativeRenderTextures.erase(std::remove_if(m_relativeRenderTextures.begin(), m_relativeRenderTextures.end(),
                                                      [](const auto &entry) { return entry.expired(); }),
                                       m_relativeRenderTextures.end());
        m_relativeRenderTextures.push_back(texture);
    }
    return texture;
}

struct InxRenderer::ImportedRenderTexture
{
    std::shared_ptr<rhi::RenderTextureDesc> description;
    std::shared_ptr<rhi::RenderTexture> texture;
};

std::shared_ptr<rhi::RenderTexture> InxRenderer::LoadRenderTexture(const std::string &guid)
{
    auto description = AssetRegistry::Instance().LoadAsset<rhi::RenderTextureDesc>(guid, ResourceType::RenderTexture);
    if (!description)
        throw std::invalid_argument("RenderTexture requires an imported asset GUID: " + guid);
    auto existing = m_assetRenderTextures.find(guid);
    if (existing != m_assetRenderTextures.end()) {
        if (auto owner = existing->second.lock()) {
            if (owner->description != description) {
                // Explicit unload or delete/Undo can replace the CPU registry
                // entry while a graphics lease still holds this same identity.
                ReconfigureImportedRenderTexture(guid, *description);
                owner->description = std::move(description);
            }
            return {owner, owner->texture.get()};
        }
    }
    auto texture = CreateRenderTexture(*description, guid);
    // One CPU identity in AssetRegistry and one GPU owner for all consumers.
    // The alias retains the registry payload while any graphics consumer lives,
    // so residency eviction cannot disconnect it from subsequent reimports.
    auto owner = std::make_shared<ImportedRenderTexture>(ImportedRenderTexture{std::move(description), texture});
    for (auto it = m_assetRenderTextures.begin(); it != m_assetRenderTextures.end();) {
        if (it->second.expired())
            it = m_assetRenderTextures.erase(it);
        else
            ++it;
    }
    m_assetRenderTextures[guid] = owner;
    return {owner, texture.get()};
}

void InxRenderer::ReconfigureImportedRenderTexture(const std::string &guid, const rhi::RenderTextureDesc &description)
{
    auto entry = m_assetRenderTextures.find(guid);
    if (entry == m_assetRenderTextures.end())
        return; // Importing an unused description must not allocate a GPU target.
    auto owner = entry->second.lock();
    if (!owner) {
        m_assetRenderTextures.erase(entry);
        return;
    }
    const auto &target = owner->texture;
    const bool relative = description.sizeMode == rhi::RenderTextureSizeMode::Relative;
    if (relative && (!m_gameRenderTarget || !m_gameRenderTarget->IsReady()))
        throw std::runtime_error("Relative RenderTexture requires the Game output resolution");
    if (relative)
        m_relativeRenderTextures.reserve(m_relativeRenderTextures.size() + 1);
    target->Reconfigure(description, relative ? m_gameRenderTarget->GetWidth() : 0,
                        relative ? m_gameRenderTarget->GetHeight() : 0);
    // Publish tracking only after the whole allocation succeeded. No allocation
    // or per-frame polling is needed to switch between absolute/relative sizing.
    m_relativeRenderTextures.erase(std::remove_if(m_relativeRenderTextures.begin(), m_relativeRenderTextures.end(),
                                                  [&target](const auto &weak) {
                                                      auto live = weak.lock();
                                                      return !live || live == target;
                                                  }),
                                   m_relativeRenderTextures.end());
    if (relative)
        m_relativeRenderTextures.push_back(target);
}

void InxRenderer::ResizeRelativeRenderTextures(uint32_t width, uint32_t height)
{
    std::vector<std::shared_ptr<rhi::RenderTexture>> live;
    live.reserve(m_relativeRenderTextures.size());
    m_relativeRenderTextures.erase(std::remove_if(m_relativeRenderTextures.begin(), m_relativeRenderTextures.end(),
                                                  [&live](const auto &entry) {
                                                      if (auto target = entry.lock()) {
                                                          live.push_back(std::move(target));
                                                          return false;
                                                      }
                                                      return true;
                                                  }),
                                   m_relativeRenderTextures.end());
    rhi::RenderTexture::ReconfigureReferenceSize(live, width, height);
}

InxRenderer::InxRenderer(vk::VulkanBackendContext &backend)
{
    m_vkCore = std::make_unique<InxVkCoreModular>(backend, static_cast<int>(ResolveMaxFramesInFlight()));
    m_vkCore->SetRenderTextureAssetLoader([this](const std::string &guid) { return LoadRenderTexture(guid); });
    m_view = std::make_unique<InxView>();
    m_view->SetPresentationSuspendHandler([this]() {
        if (m_vkCore)
            m_vkCore->SuspendPresentationSurface();
    });
    m_scenePickingService = std::make_unique<ScenePickingService>();
}

void InxRenderer::PrepareMaterialTextureAssets(const std::shared_ptr<InxMaterial> &material)
{
    m_vkCore->PrepareMaterialTextureAssets(material);
}

bool InxRenderer::InvalidateMaterialTextureAssets(const std::string &owner, const std::string &guid, bool deleted)
{
    return m_vkCore->InvalidateMaterialTextureAssets(owner, guid, deleted);
}

InxRenderer::~InxRenderer()
{
    // ========================================================================
    // Shutdown order matters!  Every subsystem holds a raw VkCore/device
    // pointer, so they MUST be destroyed BEFORE m_vkCore.  We do ONE
    // vkDeviceWaitIdle up front, then tear down subsystems without further
    // GPU drains (each subsystem's individual WaitIdle is a no-op when the
    // device is already idle and the flag below is checked).
    // ========================================================================

    // 1. Single GPU drain — after this the device is idle, all subsequent
    //    vkDeviceWaitIdle calls inside subsystem destructors are redundant.
    const bool hasVulkanDevice = m_vkCore && m_vkCore->GetDeviceContext().IsValid();
    if (hasVulkanDevice) {
        m_vkCore->GetDeviceContext().WaitIdleForShutdown();
        m_vkCore->SetShuttingDown(true);
        m_vkCore->ReleaseActiveDrawLists();
    }

    // 2. Destroy all subsystems that hold raw VkCore pointers — order:
    //    render graphs → screen UI → render targets → auxiliary renderers → GUI.
    m_graphEvents.clear();
    m_gameRenderGraph = nullptr;
    m_gameRenderGraphs.clear();
    m_sceneCameraPreviewGraph.reset();
    m_sceneCameraPreviewTarget.reset();
    m_sceneRenderGraph.reset();

    m_particleGpuSystemManager.reset();

    m_screenUIRenderer.reset();

    m_captureService.reset();
    m_scenePickingService.reset();

    m_gameRenderTarget.reset();
    m_sceneRenderTarget.reset();
    m_replacedViewTargets.clear();

    m_outlineRenderer.reset();
    m_editorGizmos.reset();
    m_editorTools.reset();
    m_componentGizmos.reset();
    m_particleGpuDrawRegistry.reset();

    if (hasVulkanDevice) {
        m_vkCore->GetDeviceContext().GetRhiDevice().SetManagedHeapLimits({});
        m_vkCore->ReleaseGpuPreviews();
    }
    m_gui.reset();

    // 3. Stop the platform callback before destroying the Vulkan owner. On
    // Android this also releases any SurfaceView teardown waiting at the
    // terminal runtime boundary.
    if (m_view)
        m_view->SetPresentationSuspendHandler({});

    // 4. Release renderer consumers, swapchain and surface. The Engine keeps
    //    the shared device and compute queue alive until their final retirement.
    m_vkCore.reset();

    // 5. Tear down the platform window last.
    if (m_view) {
        m_view->Quit();
    }
    m_view.reset();

    // The Infernux runtime owner retires its worker pool after all graphics
    // consumers. Standalone renderers only borrow the caller's JobSystem.
}

void InxRenderer::SetCameraPos(float x, float y, float z)
{
    m_cameraPos[0] = x;
    m_cameraPos[1] = y;
    m_cameraPos[2] = z;
}

void InxRenderer::SetCameraLookAt(float x, float y, float z)
{
    m_cameraLookAt[0] = x;
    m_cameraLookAt[1] = y;
    m_cameraLookAt[2] = z;
}

void InxRenderer::SetCameraUp(float x, float y, float z)
{
    m_cameraUp[0] = x;
    m_cameraUp[1] = y;
    m_cameraUp[2] = z;
}

float *InxRenderer::GetCameraPos()
{
    return &m_cameraPos[0];
}
float *InxRenderer::GetCameraLookAt()
{
    return &m_cameraLookAt[0];
}
float *InxRenderer::GetCameraUp()
{
    return &m_cameraUp[0];
}

void InxRenderer::TranslateCamera(float x, float y, float z)
{
    m_cameraPos[0] += x;
    m_cameraPos[1] += y;
    m_cameraPos[2] += z;
}

void InxRenderer::SetAppMetadata(InxAppMetadata appMetaData)
{
    m_appMetadata = appMetaData;
    m_view->SetAppMetadata(appMetaData);
}

InxAppMetadata InxRenderer::GetAppMetadata()
{
    return m_appMetadata;
}

InxAppMetadata InxRenderer::GetRendererMetadata()
{
    return m_rendererMetadata;
}

void InxRenderer::Init(int width, int height, InxAppMetadata appMetaData)
{
    SetAppMetadata(appMetaData);
    if (!m_vkCore) {
        INXLOG_ERROR("Failed to create InxVkCoreModular.");
        return;
    }

    if (!JobSystem::IsAvailable())
        throw std::logic_error("Renderer initialization requires the runtime's worker JobSystem");

    INXLOG_DEBUG("Init View.");
    m_view->Init(width, height);
    m_vkCore->SetWindowSize(static_cast<uint32_t>(m_view->m_framebufferWidth),
                            static_cast<uint32_t>(m_view->m_framebufferHeight));

    // Keep the native window hidden until the Player bootstrap has finished.
    // A game may provide its own splash sequence, but the engine must not
    // impose a visible loading window while Vulkan and runtime assets start.
    if (!PumpStartupEvents())
        throw std::runtime_error("Startup cancelled");

    uint32_t extCount = 0;
    auto ext = m_view->GetVkExtensions(&extCount);
    if (!ext) {
        throw std::runtime_error("Failed to get Vulkan extensions");
    }

    INXLOG_DEBUG("Vulkan extensions count: ", extCount);
    for (uint32_t i = 0; i < extCount; ++i) {
        INXLOG_DEBUG("Vulkan extension: ", ext[i]);
    }

    if (!m_vkCore->Init(m_appMetadata, m_rendererMetadata, extCount, const_cast<const char **>(ext))) {
        throw std::runtime_error("Failed to initialize Vulkan instance");
    }
    if (!PumpStartupEvents())
        throw std::runtime_error("Startup cancelled");

    m_view->CreateSurface(&m_vkCore->m_instance, &m_vkCore->m_surface);
    // Publish ownership before pumping cancellation: a close request here
    // must release the surface even though PrepareSurface has not run yet.
    m_vkCore->GetDeviceContext().SetExternalSurface(m_vkCore->m_surface);
    if (!PumpStartupEvents())
        throw std::runtime_error("Startup cancelled");

    INXLOG_DEBUG("Prepare surface.");
    if (!m_vkCore->PrepareSurface()) {
        throw std::runtime_error("Failed to initialize Vulkan device or swapchain");
    }
    if (!PumpStartupEvents())
        throw std::runtime_error("Startup cancelled");
}

void InxRenderer::PreparePipeline()
{
    if (!m_frameWorkOwner)
        throw std::logic_error("Renderer pipeline requires registered frame work");
    auto pumpStartup = [this]() {
        if (!PumpStartupEvents())
            throw std::runtime_error("Startup cancelled");
    };

    if (m_vkCore) {
        m_vkCore->PreparePipeline();
        pumpStartup();
        m_scenePickingService->Initialize(m_vkCore.get());

        INXLOG_DEBUG("Init GUI.");
        m_gui = std::make_unique<InxGUI>(m_vkCore.get());
        m_gui->Init(m_view->m_window);
        m_gui->RegisterTextureUploadWork(*m_frameWorkOwner);
        pumpStartup();

        // Initialize scene render target with default size
        m_sceneRenderTarget = std::make_unique<SceneRenderTarget>(m_vkCore.get());
        m_sceneRenderTarget->SetMsaaSampleCount(m_vkCore->GetMaterialPipelineManager().GetSampleCount());
        m_sceneRenderTarget->Initialize(800, 600);
        ++m_sceneRenderTargetGeneration;
        pumpStartup();

        // Set initial scene render target size for aspect ratio calculation
        m_vkCore->SetSceneRenderTargetSize(800, 600);

        const char *playerModeFlag = std::getenv("_INFERNUX_PLAYER_MODE");
        const bool playerMode = playerModeFlag != nullptr && playerModeFlag[0] == '1' && playerModeFlag[1] == '\0';
        if (playerMode)
            InitializeRuntimeScene();
        else
            InitializeDefaultScene();
        pumpStartup();

        // These support shaders are CPU-compiled. Overlap that work with the
        // remaining renderer setup and Player catalog/scene publication. The
        // first real ParticleSystem consumes the finished bundle, while a
        // project without particles never blocks startup on it.
        if (JobSystem::IsAvailable() && !m_particleProgramBootstrap) {
            m_particleProgramBootstrap = std::make_shared<ParticleProgramBootstrap>();
            auto bootstrap = m_particleProgramBootstrap;
            bootstrap->job =
                JobSystem::Get().Schedule([bootstrap] { bootstrap->programs = CompileParticleProgramBundle(); },
                                          JobDomain::Render, JobPriority::Low);
        }

        // Initialize RenderGraph pipeline (scene rendering in pre-render pass)
        if (!m_sceneRenderGraph) {
            m_sceneRenderGraph = std::make_unique<SceneRenderGraph>();
        }
        if (m_sceneRenderGraph) {
            if (!m_sceneRenderGraph->Initialize(m_vkCore.get(), m_sceneRenderTarget.get(),
                                                rhi::RenderViewKind::Scene)) {
                INXLOG_ERROR("Failed to initialize the Scene View render graph");
                m_sceneRenderGraph.reset();
            } else {
                m_sceneRenderGraph->SetParticleGpuDrawRegistry(m_particleGpuDrawRegistry.get());
            }
        }
        EnsureScreenUIRenderer();
        pumpStartup();

        m_vkCore->SetFrameComputeExecutor([this](VkCommandBuffer cmdBuf) {
            if (!m_particleGpuSystemManager)
                return;
            m_particleGpuSystemManager->Execute(cmdBuf);
        });
        m_vkCore->SetFrameComputeSubmissionCallback([this](bool submitted) {
            if (m_particleGpuSystemManager)
                m_particleGpuSystemManager->NotifySubmission(submitted);
        });
        m_vkCore->SetFrameComputeWorkPredicate(
            [this] { return m_particleGpuSystemManager && m_particleGpuSystemManager->HasPendingGpuWork(); });
        m_vkCore->SetFrameAsyncComputeExecutors(
            [this](VkCommandBuffer cmdBuf) {
                if (!m_particleGpuSystemManager)
                    return false;
                return m_particleGpuSystemManager->RecordAsyncSimulation(cmdBuf);
            },
            [this](VkCommandBuffer cmdBuf) {
                return m_particleGpuSystemManager && m_particleGpuSystemManager->RecordAsyncExport(cmdBuf);
            },
            [this] {
                // Independent-compute submission has a real cross-queue and
                // cross-frame synchronization cost. Small systems are faster
                // and just as safe when their compute graph is recorded on
                // Graphics, where Vulkan queue order provides the dependency
                // without a CPU fence or a terminal timeline join.
                constexpr uint32_t kAsyncParticleCapacityThreshold = 4096;
                return m_particleGpuSystemManager && m_particleGpuSystemManager->CanExecuteAsync() &&
                       m_particleGpuSystemManager->TelemetrySnapshot().totalCapacity >= kAsyncParticleCapacityThreshold;
            },
            [this] { return m_particleGpuSystemManager ? m_particleGpuSystemManager->AsyncExecutionGeneration() : 0; },
            [this] {
                constexpr uint32_t kAsyncParticleCapacityThreshold = 4096;
                return m_particleGpuSystemManager && m_particleGpuSystemManager->CanRecordPartitioned() &&
                       m_particleGpuSystemManager->TelemetrySnapshot().totalCapacity >= kAsyncParticleCapacityThreshold;
            });

        m_vkCore->SetFramePreSetupBuilder(
            [this](vk::VulkanFrameSubmission &submission, std::vector<uint32_t> &extraSetupDependencies) {
                if (m_particleGpuSystemManager && m_frameWorkOwner) {
                    rhi::ComputeHost host(m_vkCore->GetDeviceContext().GetRhiDevice(), m_vkCore->PrepareComputeQueue());
                    m_particleGpuSystemManager->PrepareStateCaptures(submission, host, *m_frameWorkOwner);
                }
                uint32_t initializeImages = 0;
                auto pendingImages = m_gui->TakeRenderTextureInitializations();
                for (const auto index : m_viewSchedule.order)
                    m_scheduledViews[index].graph->GetCompiledRenderGraph()->CollectPersistentImageInitializations(
                        pendingImages);
                if (!pendingImages.empty()) {
                    auto images =
                        std::make_shared<vk::RenderGraph::PersistentImageInitializations>(std::move(pendingImages));
                    auto &device = m_vkCore->GetDeviceContext();
                    initializeImages = submission.AddWork(
                        device.GetDeviceId(), rhi::QueueRole::Graphics, rhi::SubmissionDomain::Frame, 0,
                        rhi::PipelineStage::AllCommands, {},
                        [&device, images](VkCommandBuffer commandBuffer) {
                            vk::RenderGraph::RecordPersistentImageInitializations(commandBuffer, device.GetRhiDevice(),
                                                                                  *images);
                            return true;
                        },
                        "Frame/InitializeRenderTextures");
                    submission.Retain(images);
                    submission.OnAccepted([images](rhi::SubmissionTicket) {
                        vk::RenderGraph::PublishPersistentImageInitializations(*images);
                    });
                    extraSetupDependencies.push_back(initializeImages);
                }
                const auto collectOwnershipReleases = [&](SceneRenderGraph *graph) {
                    if (!graph || !graph->GetCompiledRenderGraph())
                        return;
                    vk::RenderGraph *compiled = graph->GetCompiledRenderGraph();
                    const auto &viewContext = graph->GetRenderViewContext();
                    for (const rhi::QueueRole sourceQueue :
                         {rhi::QueueRole::Graphics, rhi::QueueRole::Compute, rhi::QueueRole::Transfer}) {
                        if (!compiled->HasExternalQueueOwnershipReleases(sourceQueue))
                            continue;
                        extraSetupDependencies.push_back(submission.AddWork(
                            viewContext.device, sourceQueue, rhi::SubmissionDomain::Frame, viewContext.id,
                            rhi::PipelineStage::AllCommands,
                            initializeImages ? std::vector<uint32_t>{initializeImages} : std::vector<uint32_t>{},
                            [compiled, sourceQueue](VkCommandBuffer commandBuffer) {
                                return compiled->RecordExternalQueueOwnershipReleases(sourceQueue, commandBuffer);
                            },
                            "SceneGraph/OwnershipRelease"));
                    }
                };
                // AlwaysSimulate particles keep submitting compute after the user
                // hides Scene/Game. Those compiled graphs still owe last frame's
                // Graphics→Compute releases; skipping them hangs PrimeSimulation.
                collectOwnershipReleases(m_sceneRenderGraph.get());
                collectOwnershipReleases(m_sceneCameraPreviewGraph.get());
                for (auto &[cameraId, graph] : m_gameRenderGraphs) {
                    (void)cameraId;
                    collectOwnershipReleases(graph.get());
                }
                return true;
            });
        m_vkCore->SetFrameSubmissionBuilder([this](vk::VulkanFrameSubmission &submission, uint32_t frameSetupWorkItem,
                                                   uint32_t particleComputeWorkItem,
                                                   std::vector<uint32_t> &presentationDependencies) {
            const bool sceneViewActive = ((m_sceneViewVisible || HasPendingCapture(CaptureSource::Scene) ||
                                           (m_scenePickingService && m_scenePickingService->HasPendingRecord())) &&
                                          m_sceneRenderTarget && m_sceneRenderTarget->IsReady() &&
                                          m_sceneRenderTarget->GetWidth() > 1 && m_sceneRenderTarget->GetHeight() > 1);
            const auto activateView = [this](SceneRenderGraph *graph) {
                if (!graph)
                    return;
                m_vkCore->SetDrawBufferBindings(&graph->GetComputeBufferBindings());
                m_vkCore->SetDrawCalls(graph->HasCachedDrawCalls() ? &graph->GetCachedDrawCalls() : nullptr);
                m_vkCore->SetShadowDrawCalls(graph->HasCachedShadowDrawCalls() ? &graph->GetCachedShadowDrawCalls()
                                                                               : nullptr);
            };

            const auto appendView = [&](SceneRenderGraph *graph, bool sceneView, const glm::mat4 &view,
                                        const std::vector<uint32_t> &predecessors, uint32_t *finalWorkItem) {
                if (!graph || !graph->GetCompiledRenderGraph())
                    return true;
                graph->SetDrawViewMatrix(view);
                vk::RenderGraph *compiled = graph->GetCompiledRenderGraph();
                for (const auto &read : graph->GetBufferReadDependencies())
                    m_vkCore->RegisterFrameBufferReadDependency(read.producer, read.stages);
                m_vkCore->CaptureResidentGeometryReads(graph->GetCachedDrawCalls(), graph->GetCachedShadowDrawCalls());
                const auto &viewContext = graph->GetRenderViewContext();
                submission.Retain(graph->AcquireComputeReadValues());
                std::vector<uint32_t> graphDependencies = predecessors;
                if (particleComputeWorkItem != 0)
                    graphDependencies.push_back(particleComputeWorkItem);
                const auto range = submission.AppendRenderGraph(
                    *compiled, graphDependencies, [graph, activateView](uint32_t batchIndex, VkCommandBuffer) {
                        activateView(graph);
                        return batchIndex != 0 || graph->PrepareSubmissionExecution();
                    });
                if (range.Empty())
                    return false;
                const uint64_t previewId =
                    viewContext.kind == rhi::RenderViewKind::Preview ? m_sceneCameraPreviewId : 0;
                submission.OnAccepted(
                    [this, graph, previewId,
                     publication = graph->PreparePublication(m_graphEventInterests)](rhi::SubmissionTicket ticket) {
                        graph->CommitSubmissionExecution();
                        if (previewId)
                            m_sceneCameraPreviewRenderedId = previewId;
                        if (publication)
                            publication->Accept(ticket, m_graphEvents);
                    });

                const uint32_t finalize = submission.AddWork(
                    viewContext.device, rhi::QueueRole::Graphics, rhi::SubmissionDomain::Frame, viewContext.id,
                    rhi::PipelineStage::AllGraphics, range.terminals,
                    [this, graph, sceneView, activateView](VkCommandBuffer commandBuffer) {
                        activateView(graph);
                        if (!graph->CompleteSubmissionExecution(commandBuffer))
                            return false;
                        if (sceneView && m_scenePickingService && m_scenePickingService->HasPendingRecord() &&
                            m_sceneRenderTarget) {
                            m_scenePickingService->Record(
                                commandBuffer, m_sceneRenderTarget->GetWidth(), m_sceneRenderTarget->GetHeight(),
                                graph->GetPerViewBindGroup(), graph->GetViewGlobalsBinding(), graph->GetCachedView());
                        }
                        return true;
                    },
                    "SceneGraph/Finalize");
                presentationDependencies.push_back(finalize);
                if (finalWorkItem)
                    *finalWorkItem = finalize;
                return true;
            };

            std::vector<uint32_t> viewFinals(m_scheduledViews.size());
            const auto gameImage = CurrentGameImageGeneration();
            const bool observeGameImage = m_gameRenderTarget && (!(m_acceptedGameImage.generation == gameImage) ||
                                                                 HasPendingCapture(CaptureSource::Game));
            bool producesGameImage = false;
            for (const auto index : m_viewSchedule.order) {
                const auto &view = m_scheduledViews[index];
                std::vector<uint32_t> dependencies{frameSetupWorkItem};
                for (const auto producer : m_viewSchedule.predecessors[index])
                    dependencies.push_back(viewFinals[producer]);
                if (!appendView(view.graph, view.scene, view.graph->GetCachedView(), dependencies, &viewFinals[index]))
                    return false;
                if (observeGameImage && !producesGameImage && viewFinals[index] != 0 &&
                    view.graph->GetOutputTarget() == m_gameRenderTarget.get())
                    producesGameImage =
                        view.graph->GetCompiledRenderGraph()->HasLiveWriteToImage(m_gameRenderTarget->GetColorImage());
            }
            // The view schedule keeps requested layers and their producers
            // together. Publish the screen image only if its work was accepted.
            if (producesGameImage)
                submission.OnAccepted([this, gameImage](rhi::SubmissionTicket) {
                    m_acceptedGameImage = {gameImage, m_vkCore->GetLastSubmittedCompletionEpoch()};
                });

            if (sceneViewActive && m_sceneRenderGraph)
                activateView(m_sceneRenderGraph.get());

            return true;
        });

        // GUI is rendered via RenderGraph pass (no more imperative BeginRenderPass/EndRenderPass)
        m_vkCore->SetGuiRenderCallback(
            [this](vk::RenderContext &ctx) { m_gui->RecordCommand(ctx.GetCommandBuffer()); });

        // The Scene RenderGraph owns the outline passes and target layouts.
        m_outlineRenderer = std::make_unique<OutlineRenderer>();
    }
}

RenderFrameResult InxRenderer::DrawFrame(bool pace)
{
    if (m_drawingFrame)
        throw std::logic_error("DrawFrame cannot reenter an active frame");
    struct FrameScope
    {
        bool &active;
        uint64_t &cameraRequestSerial;
        uint64_t viewRequestSerial;
        void CloseCameraRequests()
        {
            cameraRequestSerial = viewRequestSerial + 1;
        }
        ~FrameScope()
        {
            active = false;
            CloseCameraRequests();
        }
    } frameScope{m_drawingFrame, m_cameraRequestSerial, ++m_viewRequestSerial};
    m_drawingFrame = true;
    RetireReplacedViewTargets();
    const bool samplePerformanceFrame = m_framePerformanceWindowActive;
    RendererFramePerformanceSample performanceSample;
    const auto performanceWindowFrameStart =
        samplePerformanceFrame ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    // Calculate delta time
    auto currentTime = std::chrono::steady_clock::now();
    m_deltaTime = std::chrono::duration<float>(currentTime - m_lastFrameTime).count();
    m_lastFrameTime = currentTime;

    // Deterministic stepping (Infernux::Tick in graphical mode) supplies the
    // exact simulation delta; the wall clock only reset m_lastFrameTime above
    // so a later free-running frame does not observe a stale timestamp.
    if (m_nextFrameDeltaTimeOverride >= 0.0f) {
        m_deltaTime = m_nextFrameDeltaTimeOverride;
        m_nextFrameDeltaTimeOverride = -1.0f;
    }
    m_totalTime += m_deltaTime;
    m_smoothDeltaTime += (m_deltaTime - m_smoothDeltaTime) * 0.1f;

    // ========================================================================
    // Frame Profiler - aggregates per-phase timings and reports at a bounded cadence.
    // Controlled by INFERNUX_FRAME_PROFILE in ProfileConfig.h (0 = off, 1 = on).
    // ========================================================================

#if INFERNUX_FRAME_PROFILE
    struct FrameProfiler
    {
        using Clock = std::chrono::high_resolution_clock;
        using TimePoint = Clock::time_point;
        TimePoint stamps[16];
        int idx = 0;
        void stamp()
        {
            if (idx < 16)
                stamps[idx++] = Clock::now();
        }
        float ms(int a, int b) const
        {
            return std::chrono::duration<float, std::milli>(stamps[b] - stamps[a]).count();
        }
    };
    FrameProfiler _fp;
    static int _fpCounter = 0;
    static auto _fpLastReport = std::chrono::steady_clock::now();
    static double _fpAccum[12] = {};
    static double _deltaAccumMs = 0.0;
    static double _srpSceneViewMs = 0;
    static double _srpGameViewMs = 0;
    static std::vector<double> _frameTimeSamples;
    static std::vector<double> _fixedStepFrameSamples;
    static std::vector<double> _renderOnlyFrameSamples;
    static InxRenderer::FrameDetailTiming _detailAccum;
    static SceneManager::FrameProfile _sceneAccum;
    static std::unordered_map<std::string, double> _guiAccum;
    static std::unordered_map<std::string, double> _inspSubAccum;
    static std::unordered_map<std::string, double> _hierSubAccum;
    static std::unordered_map<std::string, double> _projectSubAccum;
    struct FramePacingAccum
    {
        double targetFps = 0.0;
        double elapsedBeforeSleepMs = 0.0;
        double frameBudgetMs = 0.0;
        double requestedSleepMs = 0.0;
        double actualSleepMs = 0.0;
        int sleptFrames = 0;
        int idleFrames = 0;
        int wokeByEventFrames = 0;
        int wokeByInputFrames = 0;
        int wokeByWindowFrames = 0;
        int wokeByOtherFrames = 0;
        int inputFrames = 0;
        int playBypassFrames = 0;
        uint64_t queuedEvents = 0;
        uint64_t mouseMotionEvents = 0;
        uint64_t dispatchedMouseMotions = 0;
        double inputBeginFrameMs = 0.0;
        double inputPumpEventsMs = 0.0;
        double inputPeepEventsMs = 0.0;
        double inputImGuiDispatchMs = 0.0;
        double inputManagerDispatchMs = 0.0;
        double inputSyntheticDispatchMs = 0.0;
        double inputWindowQueryMs = 0.0;
    };
    static FramePacingAccum _pacingAccum;
    _fp.stamp(); // [0] frame start
#endif

#if INFERNUX_FRAME_PROFILE
    m_frameDetailTiming = {};
#endif

    // NOTE: do NOT reset m_lastGameRenderMs here — Python panels read it
    // during BuildFrame() which runs BEFORE the game camera render pass.
    // The value from the previous frame is the correct one for display.

    // Window events
    m_view->ProcessEvent(pace);
    if (samplePerformanceFrame) {
        performanceSample.inputMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - performanceWindowFrameStart)
                .count();
        const auto &pacing = m_view->GetLastPacingSample();
        performanceSample.inputPhases = {pacing.inputBeginFrameMs,        pacing.inputPeepEventsMs,
                                         pacing.inputImGuiDispatchMs,     pacing.inputManagerDispatchMs,
                                         pacing.inputSyntheticDispatchMs, pacing.inputWindowQueryMs};
        performanceSample.inputEventCount = pacing.queuedEventCount;
        performanceSample.dispatchedMouseMotionCount = pacing.dispatchedMouseMotionCount;
    }
    // Surface capabilities may leave the extent to the application (Wayland,
    // Retina). Publish the current pixel extent, never the requested initial
    // window size, before surface/swapchain recreation consumes it.
    m_vkCore->SetWindowSize(static_cast<uint32_t>(m_view->m_framebufferWidth),
                            static_cast<uint32_t>(m_view->m_framebufferHeight));
    const uint64_t syntheticSequence = m_view->GetLastProcessedSyntheticInputSequence();
    if (syntheticSequence != 0 && syntheticSequence != m_lastSemanticSyntheticInputSequence) {
        m_lastSemanticSyntheticInputSequence = syntheticSequence;
        // Synthetic SDL events are consumed before the GUI build below.
        // They must also wake the scheduler explicitly: semantic capture can
        // remain enabled across frames, so treating the capture request itself
        // as a level-triggered force would allow the scheduler's force edge to
        // be consumed only once. A new input sequence is the authoritative
        // wake boundary for both editor and Player GUI input.
        if (m_gui)
            m_gui->RequestSyntheticInputFrame();
        // Player mode has no semantic editor snapshot, but still needs the
        // same GUI wake/consumption boundary for its Python UI callbacks.
        if (!m_guiPlayerMode)
            InxGUISemantics::RequestSnapshot(syntheticSequence);
    } else if (m_view->NeedsImmediateGuiRefresh()) {
        // Continuous semantic capture can hold the force level high. Wake each
        // physical input batch before the next input frame clears its edges.
        m_gui->RequestFrame();
    }
#if INFERNUX_FRAME_PROFILE
    _fp.stamp(); // [1] after input/event processing
    _deltaAccumMs += static_cast<double>(m_deltaTime) * 1000.0;
    if (m_view) {
        const auto &pacing = m_view->GetLastPacingSample();
        _pacingAccum.targetFps += pacing.targetFps;
        _pacingAccum.elapsedBeforeSleepMs += pacing.elapsedBeforeSleepMs;
        _pacingAccum.frameBudgetMs += pacing.frameBudgetMs;
        _pacingAccum.requestedSleepMs += pacing.requestedSleepMs;
        _pacingAccum.actualSleepMs += pacing.actualSleepMs;
        _pacingAccum.sleptFrames += pacing.slept ? 1 : 0;
        _pacingAccum.idleFrames += pacing.idleMode ? 1 : 0;
        _pacingAccum.wokeByEventFrames += pacing.wokeByEvent ? 1 : 0;
        _pacingAccum.wokeByInputFrames += pacing.wokeByInputEvent ? 1 : 0;
        _pacingAccum.wokeByWindowFrames += pacing.wokeByWindowEvent ? 1 : 0;
        _pacingAccum.wokeByOtherFrames += pacing.wokeByOtherEvent ? 1 : 0;
        _pacingAccum.inputFrames += pacing.hadInputEvent ? 1 : 0;
        _pacingAccum.playBypassFrames += pacing.playModeBypass ? 1 : 0;
        _pacingAccum.queuedEvents += pacing.queuedEventCount;
        _pacingAccum.mouseMotionEvents += pacing.mouseMotionEventCount;
        _pacingAccum.dispatchedMouseMotions += pacing.dispatchedMouseMotionCount;
        _pacingAccum.inputBeginFrameMs += pacing.inputBeginFrameMs;
        _pacingAccum.inputPumpEventsMs += pacing.inputPumpEventsMs;
        _pacingAccum.inputPeepEventsMs += pacing.inputPeepEventsMs;
        _pacingAccum.inputImGuiDispatchMs += pacing.inputImGuiDispatchMs;
        _pacingAccum.inputManagerDispatchMs += pacing.inputManagerDispatchMs;
        _pacingAccum.inputSyntheticDispatchMs += pacing.inputSyntheticDispatchMs;
        _pacingAccum.inputWindowQueryMs += pacing.inputWindowQueryMs;
    }
#endif

    // Run the pre-GUI callback after input processing, even while the window
    // is minimized. Authoring/reload runs before UI work and before the native
    // scene frame, independently of whether a swapchain can present.
    // Actual ImGui/Vulkan work remains below the minimized early return.
    const auto callbacksStart =
        samplePerformanceFrame ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (auto callback = m_frameBeginCallback)
        callback(m_deltaTime);
    if (auto callback = m_preGuiCallback)
        callback();
    if (samplePerformanceFrame)
        performanceSample.callbacksMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - callbacksStart).count();

    // Deferred owner work must also advance when no image can be presented.
    // Call only outside SceneManager's active frame: either before Update,
    // or after EndFrame on the draw/submission paths below.
    const auto runDeferredTasks = [this]() {
        if (auto callback = m_postDrawCallback)
            callback();
    };

    if (m_view->NeedsSurfaceRecreation() && !m_view->IsApplicationInBackground()) {
        const bool recreated =
            m_vkCore && m_vkCore->RecreatePresentationSurface([this](VkInstance instance, VkSurfaceKHR *surface) {
                return m_view->TryCreateSurface(instance, surface);
            });
        if (recreated)
            m_view->AcknowledgeSurfaceRecreation();
    }

    const bool windowSuspended = m_view->IsMinimized();
    bool allowPresentation = !windowSuspended;
    // A semantic capture or synthetic input can need a GUI build even while
    // the window has no output. Simulation and camera rendering follow their
    // normal paths below.
    const bool forceGuiBuild = m_view->NeedsImmediateGuiRefresh() ||
                               ((!m_guiPlayerMode || windowSuspended) && InxGUISemantics::HasPendingCaptureRequest());
    const bool guiScheduled = (!windowSuspended || forceGuiBuild) && m_gui->ScheduleFrameBuild(forceGuiBuild);
#if INFERNUX_FRAME_PROFILE
    double gpuPrepareMs = 0.0;
#endif
    const auto prepareGpuFrame = [&](bool collectRetired) {
        const auto waitStart =
            samplePerformanceFrame ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
#if INFERNUX_FRAME_PROFILE
        const auto start = std::chrono::high_resolution_clock::now();
#endif
        if (m_vkCore) {
            if (!m_vkCore->TryPrepareCurrentFrame())
                return false;
            if (collectRetired)
                m_vkCore->CollectRetiredGpuResources();
            // Commit SDL pixel-size changes before recording the frame.
            if (allowPresentation)
                allowPresentation = m_vkCore->RefreshPresentationSize();
        }
        if (samplePerformanceFrame)
            performanceSample.gpuWaitMs +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - waitStart).count();
#if INFERNUX_FRAME_PROFILE
        gpuPrepareMs =
            std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
#endif
        return true;
    };

    // Input capture belongs to the host frame. Its draw data can be rebuilt
    // even while the GPU still owns the presentation slot.
    m_guiBuildMs = 0.0;
    bool slotReady = guiScheduled && prepareGpuFrame(true);
    if (guiScheduled) {
        const auto start = std::chrono::high_resolution_clock::now();
        m_gui->BuildScheduledFrame(slotReady && allowPresentation);
        m_view->NotifyGuiFrameBuilt();
        m_guiBuildMs =
            std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
        RecordUIPerformanceFrame();
    }
    // Update scene system
    auto _sceneUpdateStart = std::chrono::high_resolution_clock::now();
    auto _scenePhaseT0 = std::chrono::high_resolution_clock::now();
    SceneManager &sceneManager = SceneManager::Instance();
    const auto renderWorld = [&sceneManager]() {
        const Scene *scene = sceneManager.GetActiveScene();
        return scene ? scene->GetWorldId() : uint64_t{0};
    };
    if (m_waitingRenderResults &&
        (m_waitingRenderEpoch != sceneManager.GetSimulationEpoch() || m_waitingRenderWorld != renderWorld()))
        m_waitingRenderResults = false;
    if (!m_waitingRenderResults) {
        const float sceneDeltaTime = sceneManager.ConsumeFrameDeltaTime(m_deltaTime);
        TransformECSStore::Instance().BeginFrameCache();
#if INFERNUX_FRAME_PROFILE
        m_frameDetailTiming.frameCacheBeginMs =
            std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - _scenePhaseT0)
                .count();
#endif

        _scenePhaseT0 = std::chrono::high_resolution_clock::now();
        sceneManager.Update(sceneDeltaTime);
#if INFERNUX_FRAME_PROFILE
        m_frameDetailTiming.sceneUpdateCallMs =
            std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - _scenePhaseT0)
                .count();
#endif

        // LateUpdate runs immediately after Update — before rendering — so that
        // camera-follow scripts see interpolated physics transforms and their
        // results are picked up by the same frame's render pass.  This matches
        // Unity's execution order: FixedUpdate → Update → LateUpdate → Render.
        _scenePhaseT0 = std::chrono::high_resolution_clock::now();
        sceneManager.LateUpdate(sceneDeltaTime);
        // GUI capture is prepared before gameplay; unconsumed Editor Escape is
        // resolved afterwards, including frames that reuse the previous GUI.
        m_gui->CompleteInputFrame();
#if INFERNUX_FRAME_PROFILE
        m_frameDetailTiming.lateUpdateCallMs =
            std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - _scenePhaseT0)
                .count();
#endif

        _scenePhaseT0 = std::chrono::high_resolution_clock::now();
        const bool publishedPhysicsPose = TransformECSStore::Instance().EndFrameCache();
        if (publishedPhysicsPose)
            sceneManager.PublishPhysicsTransformsToRenderer();
        sceneManager.PublishAuthoredTransformsToPhysics();
        if (!sceneManager.HasPendingFixedStep()) {
            sceneManager.EmitRuntimeFrameBarrier(SceneManager::RuntimeFrameBarrier::FinalTransformResolve);
            // Animation and Timeline currently execute as ordinary component updates.
            // This boundary publishes their finalized transform/pose output to all
            // downstream render and snapshot consumers without running them twice.
            sceneManager.EmitRuntimeFrameBarrier(SceneManager::RuntimeFrameBarrier::AnimationTimeline);
        }
        ConsumeSceneTemporalDiscontinuity();
    } else {
        m_gui->CompleteInputFrame();
    }
#if INFERNUX_FRAME_PROFILE
    m_frameDetailTiming.frameCacheEndMs =
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - _scenePhaseT0).count();
#endif
    auto _sceneUpdateEnd = std::chrono::high_resolution_clock::now();
    m_sceneUpdateMs = std::chrono::duration<double, std::milli>(_sceneUpdateEnd - _sceneUpdateStart).count();
    // The slot may have retired while the GUI and scene advanced.
    if (!slotReady)
        slotReady = prepareGpuFrame(!guiScheduled);
    // Calls after view selection belong to the next opportunity. A skipped
    // opportunity leaves the camera's resident texture available as-is.
    frameScope.CloseCameraRequests();
    if (!slotReady) {
        sceneManager.EndFrame(false);
        runDeferredTasks();
        RequestFullSpeedFrame();
        if (pace)
            SDL_Delay(1);
        return {};
    }
#if INFERNUX_FRAME_PROFILE
    _fp.stamp(); // [2] after scene update and GPU frame preparation
#endif

    // An explicit Step can still have required view consumers while the
    // window is suspended. Keep them in this logical step's view schedule.
    const bool allowViewportViews = !windowSuspended || sceneManager.IsStepping();
    const bool sceneViewActive =
        (((allowViewportViews && m_sceneViewVisible) || HasPendingCapture(CaptureSource::Scene) ||
          (m_scenePickingService && m_scenePickingService->HasPendingRecord())) &&
         m_sceneRenderTarget && m_sceneRenderTarget->IsReady() && m_sceneRenderTarget->GetWidth() > 1 &&
         m_sceneRenderTarget->GetHeight() > 1);
    const bool gameViewActive = HasActiveGameViews(allowViewportViews);

    // Prepare scene rendering data (collect + cull + sort) AFTER GUI processing
    // so we always operate on the current scene state.
    SceneRenderBridge &bridge = SceneRenderBridge::Instance();
    auto _prepareStart = std::chrono::high_resolution_clock::now();
    const uint64_t resultEpoch = sceneManager.GetSimulationEpoch();
    const uint64_t resultWorld = renderWorld();
    if (!sceneManager.RuntimeResultsReady(SceneManager::RuntimeFrameBarrier::RenderExtraction)) {
        m_waitingRenderResults = true;
        m_waitingRenderEpoch = resultEpoch;
        m_waitingRenderWorld = resultWorld;
        sceneManager.EndFrame(false);
        runDeferredTasks();
        RequestFullSpeedFrame();
        // No command recording or acquire has begun. The same GPU services
        // must remain available to the request this rendering is waiting for.
        return {true};
    }
    m_waitingRenderResults = false;
    sceneManager.EmitRuntimeFrameBarrier(SceneManager::RuntimeFrameBarrier::RenderExtraction);
    bridge.PrepareFrame(sceneViewActive);
    auto _prepareEnd = std::chrono::high_resolution_clock::now();
    m_prepareFrameMs = std::chrono::duration<double, std::milli>(_prepareEnd - _prepareStart).count();
#if INFERNUX_FRAME_PROFILE
    _fp.stamp(); // [3] after PrepareFrame (CollectRenderables)
#endif

    // Begin a new mesh preparation and retention cycle for these views.
    if (m_vkCore) {
        m_vkCore->AdvanceEnsureFrame();
    }

    // Update camera from scene system (uses PrepareFrame results)
    bridge.UpdateCameraData(m_cameraPos, m_cameraLookAt, m_cameraUp);

    // A compiled view is reusable, but each draw must request it again.
    if (m_renderPipeline) {
        EditorGizmosContext gizmoCtx;
        if (m_editorGizmos) {
            gizmoCtx.gizmos = m_editorGizmos.get();
            gizmoCtx.editorTools = m_editorTools.get();
            gizmoCtx.componentGizmos = m_componentGizmos.get();
            auto &registry = AssetRegistry::Instance();
            gizmoCtx.gizmoMaterial = registry.GetBuiltinMaterial("GizmoMaterial");
            gizmoCtx.gridMaterial = registry.GetBuiltinMaterial("GridMaterial");
            gizmoCtx.editorToolsMaterial = registry.GetBuiltinMaterial("EditorToolsMaterial");
            gizmoCtx.componentGizmosMaterial = registry.GetBuiltinMaterial("ComponentGizmosMaterial");
            gizmoCtx.componentGizmoIconMaterial = registry.GetBuiltinMaterial("ComponentGizmoIconMaterial");
            gizmoCtx.cameraGizmoIconMaterial = registry.GetBuiltinMaterial("ComponentGizmoCameraIconMaterial");
            gizmoCtx.lightGizmoIconMaterial = registry.GetBuiltinMaterial("ComponentGizmoLightIconMaterial");
            gizmoCtx.particleGizmoIconMaterial = registry.GetBuiltinMaterial("ComponentGizmoParticleIconMaterial");
            gizmoCtx.selectedObjectId = m_selectedObjectId;
            gizmoCtx.activeScene = SceneManager::Instance().GetActiveScene();
            gizmoCtx.iconDpiScale = m_gui ? m_gui->GetDisplayScale() : 1.0f;
        }

        // ---- Scene View: always use editor camera ----
        if (sceneViewActive) {
            ScriptableRenderContext ctx(m_vkCore.get(), m_sceneRenderGraph.get(), m_viewRequestSerial, gizmoCtx);
            Camera *editorCam = SceneManager::Instance().GetEditorCameraController().GetCamera();

            // World-space Billboard UI is gameplay-facing UI.  The Scene view
            // still renders meshes and receives editor rays through editorCam,
            // while billboard expansion explicitly follows the active Game
            // Camera used by scripts and the Game panel.
            if (m_sceneRenderGraph) {
                Camera *gameBillboardCam = FindGameCameraCached();
                m_sceneRenderGraph->SetBillboardViewMatrix(
                    gameBillboardCam ? gameBillboardCam->GetViewMatrix()
                                     : (editorCam ? editorCam->GetViewMatrix() : glm::mat4(1.0f)));
            }

#if INFERNUX_FRAME_PROFILE
            auto _srpT0 = std::chrono::high_resolution_clock::now();
#endif
            if (editorCam)
                m_renderPipeline->Render(ctx, editorCam);
#if INFERNUX_FRAME_PROFILE
            auto _srpT1 = std::chrono::high_resolution_clock::now();
            _srpSceneViewMs += std::chrono::duration<double, std::milli>(_srpT1 - _srpT0).count();
#endif
        }

        // ---- Game View: one context and one RenderView per active camera ----
        if (gameViewActive) {
            const auto _srpT2 = std::chrono::high_resolution_clock::now();
            for (Camera *gameCam : FindGameCamerasCached()) {
                if (!ShouldRenderGameCamera(gameCam, allowViewportViews))
                    continue;
                SceneRenderGraph *graph = EnsureGameRenderGraph(gameCam);
                if (!graph)
                    continue;
                // Set aspect ratio BEFORE SetupCameraProperties() so the
                // projection matrix snapshot matches the render target size.
                const auto &viewTarget = graph->GetRenderViewContext();
                if (viewTarget.height > 0) {
                    const auto *output = graph->GetOutputTarget();
                    gameCam->UpdateAutomaticAspectRatio(static_cast<float>(output->GetWidth()) / output->GetHeight());
                    gameCam->SetScreenDimensions(output->GetWidth(), output->GetHeight());
                }

                // Game camera excludes editor-only gizmos/grid/outline.
                EditorGizmosContext gameContentCtx;
                if (GameObject *cameraObject = gameCam->GetGameObject())
                    gameContentCtx.activeScene = cameraObject->GetScene();
                ScriptableRenderContext gameCtx(m_vkCore.get(), graph, m_viewRequestSerial, gameContentCtx);

                graph->SetBillboardViewMatrix(gameCam->GetViewMatrix());

                m_renderPipeline->Render(gameCtx, gameCam);
            }
            const auto _srpT3 = std::chrono::high_resolution_clock::now();
            m_lastGameRenderMs = std::chrono::duration<double, std::milli>(_srpT3 - _srpT2).count();
#if INFERNUX_FRAME_PROFILE
            _srpGameViewMs += m_lastGameRenderMs;
#endif
        }
    } else {
        INXLOG_ERROR("No render pipeline set — scene will not be rendered. "
                     "Call engine.set_render_pipeline(DefaultRenderPipelineAsset()) to activate rendering.");
    }

    if (sceneViewActive && m_sceneViewVisible && !windowSuspended && m_sceneCameraPreviewId)
        PrepareSceneCameraPreview();

    // All runtime UI packets now exist. Their dynamic font images must be
    // resident before Scene/Game command recording, including the first draw.
    if (m_gui)
        m_gui->PrepareRuntimeFontTextures();

    // The current pipeline submissions decide the shared screen attachments.
    CollectRequestedViews(sceneViewActive, gameViewActive);
    if (CheckAndApplyMsaaRequest()) {
        sceneManager.EndFrame(false);
        runDeferredTasks();
        return {};
    }
#if INFERNUX_FRAME_PROFILE
    _fp.stamp(); // [4] after RenderPipeline::Render (Python SRP)
#endif

    // Hidden views are allowed to go unused, but a frame with no cameras must
    // not treat every mesh as dead. Game/Scene reopen rebuilds from cache miss
    // instead of drawing an empty buffer table.
    if (sceneViewActive || gameViewActive)
        CleanupDrawCallBuffers();
#if INFERNUX_FRAME_PROFILE
    _fp.stamp(); // [5] after CleanupUnusedBuffers
#endif

    UpdateSceneLighting();
#if INFERNUX_FRAME_PROFILE
    _fp.stamp(); // [6] after UpdateSceneLighting
#endif

    // Lazy-init and update outline renderer
    if (sceneViewActive && m_outlineRenderer && m_sceneRenderTarget && m_vkCore) {
        m_outlineRenderer->Initialize(m_vkCore.get(), m_sceneRenderTarget.get());
        if (ShouldDrawSelectionOutline())
            m_outlineRenderer->SetOutlineObjectIds(m_selectedOutlineObjectIds);
        else
            m_outlineRenderer->SetOutlineObjectIds({});
    } else if (m_outlineRenderer) {
        m_outlineRenderer->SetOutlineObjectIds({});
    }

    if (m_sceneRenderGraph)
        m_sceneRenderGraph->SetOutlineRenderer(sceneViewActive ? m_outlineRenderer.get() : nullptr);

    UpdateViewSchedule();
    const bool pendingStepView = sceneManager.IsStepping() &&
                                 std::any_of(m_scheduledViews.begin(), m_scheduledViews.end(), [](const auto &view) {
                                     return view.waitingForResult && view.graph->HasPendingBufferInputs(true);
                                 });
    StageEngineGlobalsUBO();

    // Stage view-local constants and grow shared instance streams before
    // recording so later passes cannot rewrite already-bound descriptors.
    size_t requiredInstances = 0;
    for (const auto index : m_viewSchedule.order) {
        auto *graph = m_scheduledViews[index].graph;
        if (graph->HasCachedCameraVP()) {
            const glm::mat4 previous = graph->GetPreviousViewProj();
            (void)graph->StageCameraMatrices(graph->GetCachedView(), graph->GetCachedProj(), &previous);
        }
        if (graph->HasCachedDrawCalls())
            requiredInstances += graph->GetCachedDrawCalls().size();
        if (graph->HasCachedShadowDrawCalls())
            requiredInstances += graph->GetCachedShadowDrawCalls().size() * graph->GetCameraShadowViewCount();
    }
#if INFERNUX_FRAME_PROFILE
    _fp.stamp(); // [7] after Outline+GraphBuild+UBO staging
#endif
    m_vkCore->PreallocateInstances(requiredInstances);

    // Render frame with scene camera
    sceneManager.EmitRuntimeFrameBarrier(SceneManager::RuntimeFrameBarrier::RenderGraph);
    // Publish physical scene data once, before either GPU recording path.
    // This also retires history on frames with no particle compute work.
    UpdateParticleCollisionScene();
    using WindowOutput = InxVkCoreModular::WindowOutput;
    const auto windowOutput = !allowPresentation                         ? WindowOutput::Disabled
                              : HasPendingCapture(CaptureSource::Editor) ? WindowOutput::DisplayAndCapture
                                                                         : WindowOutput::Display;
    const auto drawStart =
        samplePerformanceFrame ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto previousCompletionEpoch = m_vkCore->GetLastSubmittedCompletionEpoch();
    const auto presentationAcceptedNs =
        m_vkCore->DrawFrame(windowOutput, samplePerformanceFrame ? &performanceSample.drawPhases : nullptr);
    if (samplePerformanceFrame)
        performanceSample.drawMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - drawStart).count();
    const bool renderSubmitted = m_vkCore->GetLastSubmittedCompletionEpoch() != previousCompletionEpoch;
    if (renderSubmitted) {
        ++m_frameCount;
        sceneManager.CommitSkinPoseHistories();
    }
    if (m_vkCore->ConsumeFirstVisiblePresentation())
        m_view->RevealAfterFirstPresentation();
    if (m_vkCore->ConsumePresentationSurfaceLost()) {
        // VK_ERROR_SURFACE_LOST_KHR cannot be repaired by recreating a
        // swapchain against the old VkSurfaceKHR. Route it through the same
        // platform-surface rebind used by Android lifecycle transitions.
        m_view->RequestSurfaceRecreation();
    }
    SubmitPendingCaptureReadbacks(renderSubmitted);
    sceneManager.EmitRuntimeFrameBarrier(SceneManager::RuntimeFrameBarrier::SnapshotPublication);
#if INFERNUX_FRAME_PROFILE
    _fp.stamp(); // [8] after VkCore::DrawFrame (GPU submit + present)
#endif

    sceneManager.EndFrame((renderSubmitted || !allowPresentation) && !pendingStepView);

    // Budget accounting is diagnostic work, not a per-second render task. At
    // uncapped frame rates a 60-frame cadence runs every ~86 ms and the VMA
    // statistics query dominates the 99th percentile. Five seconds still
    // bounds residency well before a scene can exhaust the device budget.
    m_gpuResidencyCheckAccumulatorMs += static_cast<double>(m_deltaTime) * 1000.0;
    if (m_gpuResidencyCheckAccumulatorMs >= 5000.0) {
        m_gpuResidencyCheckAccumulatorMs = 0.0;
        (void)TrimGpuResidencyBudget();
    }

    // Compute game-only frame cost: sum of game phases, excluding editor UI.
    m_gameOnlyFrameMs = m_sceneUpdateMs + m_prepareFrameMs + m_lastGameRenderMs;

    // ========================================================================
    // Post-draw callback: scene loading / deferred tasks that must run
    // AFTER GPU submission.  The Python callback pumps OS events itself
    // (via engine.pump_events()) when performing heavy scene loads, so
    // we no longer sandwich every frame with SDL_PumpEvents here.
    // ========================================================================
    runDeferredTasks();

    if (samplePerformanceFrame) {
        performanceSample.frameMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - performanceWindowFrameStart)
                .count();
        RecordFramePerformanceSample(performanceSample);
    }

    // ========================================================================
    // Unified Frame Profiler output. Require enough samples, then throttle by
    // wall time so an uncapped Debug Player cannot flood the engine log.
    // ========================================================================
#if INFERNUX_FRAME_PROFILE
    {
        _fp.stamp();                    // [9] end of frame
        _fpAccum[0] += _fp.ms(0, 9);    // total frame
        _fpAccum[1] += _fp.ms(0, 1);    // Input
        _fpAccum[2] += m_sceneUpdateMs; // Scene+Late
        _fpAccum[3] += gpuPrepareMs;    // GPUFence
        _fpAccum[4] += m_guiBuildMs;    // UI
        _fpAccum[5] += _fp.ms(2, 3);    // Prepare
        _fpAccum[6] += _fp.ms(3, 4);    // Render (Python SRP)
        _fpAccum[7] += _fp.ms(4, 5);    // CleanupBuffers
        _fpAccum[8] += _fp.ms(5, 6);    // Lighting
        _fpAccum[9] += _fp.ms(6, 7);    // Outline+Graph+UBO
        _fpAccum[10] += _fp.ms(7, 8);   // DrawFrame (submit+present)
        _fpAccum[11] += _fp.ms(8, 9);   // End

        _detailAccum.frameCacheBeginMs += m_frameDetailTiming.frameCacheBeginMs;
        _detailAccum.sceneUpdateCallMs += m_frameDetailTiming.sceneUpdateCallMs;
        _detailAccum.lateUpdateCallMs += m_frameDetailTiming.lateUpdateCallMs;
        _detailAccum.frameCacheEndMs += m_frameDetailTiming.frameCacheEndMs;
        _detailAccum.cleanupCollectIdsMs += m_frameDetailTiming.cleanupCollectIdsMs;
        _detailAccum.cleanupReleaseMs += m_frameDetailTiming.cleanupReleaseMs;
        _detailAccum.cleanupActiveIds += m_frameDetailTiming.cleanupActiveIds;
        _detailAccum.lightingCollectMs += m_frameDetailTiming.lightingCollectMs;
        _detailAccum.lightingShadowEditorMs += m_frameDetailTiming.lightingShadowEditorMs;
        _detailAccum.lightingShadowGameMs += m_frameDetailTiming.lightingShadowGameMs;
        _detailAccum.lightingUploadMs += m_frameDetailTiming.lightingUploadMs;

        const auto &sceneProfile = SceneManager::Instance().GetLastFrameProfile();
        const double totalFrameMs = _fp.ms(0, 9);
        _frameTimeSamples.push_back(totalFrameMs);
        if (sceneProfile.fixedSteps > 0.0)
            _fixedStepFrameSamples.push_back(totalFrameMs);
        else
            _renderOnlyFrameSamples.push_back(totalFrameMs);
        _sceneAccum.editorCameraMs += sceneProfile.editorCameraMs;
        _sceneAccum.editorUpdateMs += sceneProfile.editorUpdateMs;
        _sceneAccum.pendingStartsMs += sceneProfile.pendingStartsMs;
        _sceneAccum.syncExternalMovesMs += sceneProfile.syncExternalMovesMs;
        _sceneAccum.syncCollidersMs += sceneProfile.syncCollidersMs;
        _sceneAccum.fixedUpdateMs += sceneProfile.fixedUpdateMs;
        _sceneAccum.physicsStepMs += sceneProfile.physicsStepMs;
        _sceneAccum.physicsEventsMs += sceneProfile.physicsEventsMs;
        _sceneAccum.syncRigidbodiesMs += sceneProfile.syncRigidbodiesMs;
        _sceneAccum.interpolationMs += sceneProfile.interpolationMs;
        _sceneAccum.gameplayUpdateMs += sceneProfile.gameplayUpdateMs;
        _sceneAccum.lateUpdateMs += sceneProfile.lateUpdateMs;
        _sceneAccum.audioMs += sceneProfile.audioMs;
        _sceneAccum.endFrameMs += sceneProfile.endFrameMs;
        _sceneAccum.fixedSteps += sceneProfile.fixedSteps;

        const auto &panelTimes = m_gui->GetLastPanelTimesMs();
        for (const auto &name : m_gui->GetRenderableOrder()) {
            auto it = panelTimes.find(name);
            if (it != panelTimes.end()) {
                _guiAccum[name] += it->second;
            }
        }

        const auto &panelSubTimes = m_gui->GetLastPanelSubTimesMs();
        const auto accumulatePanelSubTimes = [&panelSubTimes](const char *panelName, auto &destination) {
            const auto panel = panelSubTimes.find(panelName);
            if (panel == panelSubTimes.end())
                return;
            for (const auto &kv : panel->second)
                destination[kv.first] += kv.second;
        };
        accumulatePanelSubTimes("inspector", _inspSubAccum);
        accumulatePanelSubTimes("hierarchy", _hierSubAccum);
        accumulatePanelSubTimes("project", _projectSubAccum);

        ++_fpCounter;
        const auto _fpReportNow = std::chrono::steady_clock::now();
        if (_fpCounter >= INFERNUX_FRAME_PROFILE_WINDOW &&
            _fpReportNow - _fpLastReport >= std::chrono::seconds(INFERNUX_FRAME_PROFILE_REPORT_INTERVAL_SECONDS)) {
            const double kWindow = static_cast<double>(_fpCounter);
            const auto percentile = [](const std::vector<double> &samples, double fraction) {
                if (samples.empty())
                    return 0.0;
                std::vector<double> sorted = samples;
                std::sort(sorted.begin(), sorted.end());
                const size_t index = static_cast<size_t>(std::round(fraction * static_cast<double>(sorted.size() - 1)));
                return sorted[index];
            };
            const auto sampleMean = [](const std::vector<double> &samples) {
                return samples.empty()
                           ? 0.0
                           : std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
            };
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2);
            oss << "[Profile] avg" << _fpCounter << " frame=" << (_fpAccum[0] / kWindow) << "ms"
                << " p50=" << percentile(_frameTimeSamples, 0.50) << "ms"
                << " p95=" << percentile(_frameTimeSamples, 0.95) << "ms"
                << " p99=" << percentile(_frameTimeSamples, 0.99) << "ms"
                << " min=" << percentile(_frameTimeSamples, 0.00) << "ms"
                << " max=" << percentile(_frameTimeSamples, 1.00) << "ms"
                << " | Delta=" << (_deltaAccumMs / kWindow) << "ms" << " | Input=" << (_fpAccum[1] / kWindow) << "ms"
                << " | Scene+Late=" << (_fpAccum[2] / kWindow) << "ms" << " | GPUFence=" << (_fpAccum[3] / kWindow)
                << "ms" << " | UI=" << (_fpAccum[4] / kWindow) << "ms" << " | Prepare=" << (_fpAccum[5] / kWindow)
                << "ms" << " | Render=" << (_fpAccum[6] / kWindow) << "ms" << "(SV=" << (_srpSceneViewMs / kWindow)
                << "ms GV=" << (_srpGameViewMs / kWindow) << "ms)" << " | Cleanup=" << (_fpAccum[7] / kWindow) << "ms"
                << " | Lighting=" << (_fpAccum[8] / kWindow) << "ms" << " | Graph+UBO=" << (_fpAccum[9] / kWindow)
                << "ms" << " | DrawFrame=" << (_fpAccum[10] / kWindow) << "ms" << " | End=" << (_fpAccum[11] / kWindow)
                << "ms";

            {
                const auto bridgeProfile = SceneRenderBridge::Instance().GetProfileSnapshot();
                const auto srcProfile = ScriptableRenderContext::GetProfileSnapshot();
                const auto deletionStats =
                    m_vkCore ? m_vkCore->GetRetirementQueue().GetStats() : GpuRetirementQueue::Stats{};

                oss << "\n  FrameCache: begin=" << (_detailAccum.frameCacheBeginMs / kWindow)
                    << "ms updateCall=" << (_detailAccum.sceneUpdateCallMs / kWindow)
                    << "ms lateCall=" << (_detailAccum.lateUpdateCallMs / kWindow)
                    << "ms end=" << (_detailAccum.frameCacheEndMs / kWindow) << "ms" << "\n  PrepareDetail: total="
                    << (bridgeProfile.prepareCalls ? bridgeProfile.prepareMs / bridgeProfile.prepareCalls : 0.0)
                    << "ms collect="
                    << (bridgeProfile.prepareCalls ? bridgeProfile.collectMs / bridgeProfile.prepareCalls : 0.0)
                    << "ms update="
                    << (bridgeProfile.prepareCalls ? bridgeProfile.updateMs / bridgeProfile.prepareCalls : 0.0)
                    << "ms cull="
                    << (bridgeProfile.prepareCalls ? bridgeProfile.cullMs / bridgeProfile.prepareCalls : 0.0)
                    << "ms sort="
                    << (bridgeProfile.prepareCalls ? bridgeProfile.sortMs / bridgeProfile.prepareCalls : 0.0)
                    << "ms fastCalls=" << bridgeProfile.prepareFastCalls
                    << " slowCalls=" << bridgeProfile.prepareSlowCalls << " renderables/frame="
                    << (bridgeProfile.prepareCalls ? bridgeProfile.renderables / bridgeProfile.prepareCalls : 0.0)
                    << " visible/frame="
                    << (bridgeProfile.prepareCalls ? bridgeProfile.visible / bridgeProfile.prepareCalls : 0.0)
                    << " cameraCache=" << bridgeProfile.cameraCacheHits << "/" << bridgeProfile.cameraCacheMisses
                    << " miss(world/structure/transform/mask/frustum/vp)=" << bridgeProfile.cameraMissWorld << "/"
                    << bridgeProfile.cameraMissStructural << "/" << bridgeProfile.cameraMissTransform << "/"
                    << bridgeProfile.cameraMissMask << "/" << bridgeProfile.cameraMissFrustum << "/"
                    << bridgeProfile.cameraMissViewProjection
                    << " conservative=" << bridgeProfile.conservativeFullListUses
                    << "\n  SRC: cull=" << (srcProfile.cullCalls ? srcProfile.cullMs / srcProfile.cullCalls : 0.0)
                    << "ms editorCull="
                    << (srcProfile.cullEditorCalls ? srcProfile.cullEditorMs / srcProfile.cullEditorCalls : 0.0)
                    << "ms gameCull="
                    << (srcProfile.cullGameCalls ? srcProfile.cullGameMs / srcProfile.cullGameCalls : 0.0)
                    << "ms applyGraph="
                    << (srcProfile.submitCalls ? srcProfile.applyGraphMs / srcProfile.submitCalls : 0.0)
                    << "ms submit=" << (srcProfile.submitCalls ? srcProfile.submitMs / srcProfile.submitCalls : 0.0)
                    << "ms base=" << (srcProfile.submitCalls ? srcProfile.submitBaseMs / srcProfile.submitCalls : 0.0)
                    << "ms editorAppend="
                    << (srcProfile.submitCalls ? srcProfile.submitEditorAppendMs / srcProfile.submitCalls : 0.0)
                    << "ms ensure="
                    << (srcProfile.submitCalls ? srcProfile.ensureBuffersMs / srcProfile.submitCalls : 0.0)
                    << "ms cache=" << (srcProfile.submitCalls ? srcProfile.cacheGraphMs / srcProfile.submitCalls : 0.0)
                    << "ms finalDraws/submit="
                    << (srcProfile.submitCalls ? srcProfile.finalDrawCalls / srcProfile.submitCalls : 0.0)
                    << " listBorrowed=" << srcProfile.borrowedRendererListSubmits
                    << " listOwned=" << srcProfile.ownedRendererListSubmits << " materialized/frame="
                    << (srcProfile.submitCalls ? srcProfile.materializedDrawCalls / srcProfile.submitCalls : 0.0)
                    << " reuse=" << srcProfile.cachedSubmissionReuses
                    << " reject(editor/visible/shadow/graph/signature)=" << srcProfile.submissionRejectEditorAppenders
                    << "/" << srcProfile.submissionRejectOwnedVisibleList << "/"
                    << srcProfile.submissionRejectOwnedShadowList << "/" << srcProfile.submissionRejectMissingGraph
                    << "/" << srcProfile.submissionRejectSignature
                    << "\n  CleanupDetail: collectIds=" << (_detailAccum.cleanupCollectIdsMs / kWindow)
                    << "ms release=" << (_detailAccum.cleanupReleaseMs / kWindow)
                    << "ms activeIds/frame=" << (_detailAccum.cleanupActiveIds / kWindow)
                    << " retirePending=" << deletionStats.pending << " retireHighWater=" << deletionStats.highWatermark
                    << " retired=" << deletionStats.retired << " retirePushed=" << deletionStats.pushed
                    << "\n  LightingDetail: collect=" << (_detailAccum.lightingCollectMs / kWindow)
                    << "ms shadowEditor=" << (_detailAccum.lightingShadowEditorMs / kWindow)
                    << "ms shadowGame=" << (_detailAccum.lightingShadowGameMs / kWindow)
                    << "ms upload=" << (_detailAccum.lightingUploadMs / kWindow) << "ms";
            }

            // DrawFrame sub-timing breakdown
            if (m_vkCore) {
                double drawSub[infernux::InxVkCoreModular::kDrawSubSlots];
                int drawN = m_vkCore->GetDrawSubTimings(drawSub);
                if (drawN > 0) {
                    double n = static_cast<double>(drawN);
                    const auto rgProfile = infernux::vk::RenderGraph::GetExecuteProfileSnapshot();
                    const auto topPassProfiles = infernux::vk::RenderGraph::GetTopCallbackProfiles(6);
                    const auto &gpuTimestamps = m_vkCore->GetLatestGpuTimestampFrame();
                    uint64_t filteredCalls = 0, filteredEligible = 0, filteredIssued = 0, filteredActualDraws = 0;
                    uint64_t shadowCalls = 0, shadowEligible = 0, shadowIssued = 0, shadowActualDraws = 0;
                    m_vkCore->GetDrawSubCounters(filteredCalls, filteredEligible, filteredIssued, filteredActualDraws,
                                                 shadowCalls, shadowEligible, shadowIssued, shadowActualDraws);
                    oss << "\n  DrawFrame: Acquire=" << (drawSub[0] / n) << "ms RecordAndSubmit=" << (drawSub[1] / n)
                        << "ms Publish=" << (drawSub[2] / n) << "ms Present=" << (drawSub[3] / n) << "ms"
                        << " GPUFrameWait=" << (drawSub[4] / n) << "ms"
                        << "\n    Filtered: total=" << (drawSub[5] / n) << "ms" << " filter=" << (drawSub[6] / n)
                        << "ms" << " sort=" << (drawSub[7] / n) << "ms" << " draw=" << (drawSub[8] / n) << "ms"
                        << " calls/frame=" << (static_cast<double>(filteredCalls) / n) << " eligible/call="
                        << (filteredCalls ? static_cast<double>(filteredEligible) / static_cast<double>(filteredCalls)
                                          : 0.0)
                        << " issued/call="
                        << (filteredCalls ? static_cast<double>(filteredIssued) / static_cast<double>(filteredCalls)
                                          : 0.0)
                        << " vkDraws/frame=" << (static_cast<double>(filteredActualDraws) / n)
                        << "\n    Shadow: total=" << (drawSub[9] / n) << "ms" << " filter=" << (drawSub[10] / n) << "ms"
                        << " sort=" << (drawSub[12] / n) << "ms" << " cull=" << (drawSub[13] / n) << "ms"
                        << " upload=" << (drawSub[14] / n) << "ms" << " batch=" << (drawSub[15] / n) << "ms"
                        << " calls/frame=" << (static_cast<double>(shadowCalls) / n) << " eligible/call="
                        << (shadowCalls ? static_cast<double>(shadowEligible) / static_cast<double>(shadowCalls) : 0.0)
                        << " issued/call="
                        << (shadowCalls ? static_cast<double>(shadowIssued) / static_cast<double>(shadowCalls) : 0.0)
                        << " vkDraws/frame=" << (static_cast<double>(shadowActualDraws) / n)
                        << "\n    RenderGraph: barrier="
                        << (rgProfile.executeCalls ? rgProfile.barrierMs / static_cast<double>(rgProfile.executeCalls)
                                                   : 0.0)
                        << "ms begin="
                        << (rgProfile.executeCalls ? rgProfile.beginPassMs / static_cast<double>(rgProfile.executeCalls)
                                                   : 0.0)
                        << "ms callback="
                        << (rgProfile.executeCalls ? rgProfile.callbackMs / static_cast<double>(rgProfile.executeCalls)
                                                   : 0.0)
                        << "ms end="
                        << (rgProfile.executeCalls ? rgProfile.endPassMs / static_cast<double>(rgProfile.executeCalls)
                                                   : 0.0)
                        << "ms passes/exec="
                        << (rgProfile.executeCalls
                                ? static_cast<double>(rgProfile.passCount) / static_cast<double>(rgProfile.executeCalls)
                                : 0.0)
                        << " gfx/exec="
                        << (rgProfile.executeCalls ? static_cast<double>(rgProfile.graphicsPassCount) /
                                                         static_cast<double>(rgProfile.executeCalls)
                                                   : 0.0)
                        << " barrierCalls/exec="
                        << (rgProfile.executeCalls ? static_cast<double>(rgProfile.barrierCallCount) /
                                                         static_cast<double>(rgProfile.executeCalls)
                                                   : 0.0)
                        << " rhiPipeline/exec="
                        << (rgProfile.executeCalls ? static_cast<double>(rgProfile.rhiPipelineBinds) /
                                                         static_cast<double>(rgProfile.executeCalls)
                                                   : 0.0)
                        << " skipped="
                        << (rgProfile.executeCalls ? static_cast<double>(rgProfile.rhiPipelineBindSkips) /
                                                         static_cast<double>(rgProfile.executeCalls)
                                                   : 0.0)
                        << " rhiGroups/exec="
                        << (rgProfile.executeCalls ? static_cast<double>(rgProfile.rhiGroupBinds) /
                                                         static_cast<double>(rgProfile.executeCalls)
                                                   : 0.0)
                        << " skipped="
                        << (rgProfile.executeCalls ? static_cast<double>(rgProfile.rhiGroupBindSkips) /
                                                         static_cast<double>(rgProfile.executeCalls)
                                                   : 0.0);

                    if (gpuTimestamps.available) {
                        oss << "\n    GPU#" << gpuTimestamps.serial << ':';
                        for (uint32_t i = 0; i < gpuTimestamps.sampleCount; ++i) {
                            const auto &sample = gpuTimestamps.samples[i];
                            oss << ' ' << sample.Name() << '=' << sample.milliseconds << "ms";
                        }
                    }

                    const auto gizmoPass = infernux::vk::RenderGraph::GetCallbackProfile("_ComponentGizmos");
                    if (gizmoPass.calls != 0) {
                        oss << "\n    ComponentGizmos: callback=" << (gizmoPass.totalMs / gizmoPass.calls)
                            << "ms/call calls=" << gizmoPass.calls;
                        double gpuGizmoMs = 0.0;
                        uint32_t gpuGizmoRegions = 0;
                        for (uint32_t i = 0; i < gpuTimestamps.sampleCount; ++i) {
                            if (gpuTimestamps.samples[i].Name() == "_ComponentGizmos") {
                                gpuGizmoMs += gpuTimestamps.samples[i].milliseconds;
                                ++gpuGizmoRegions;
                            }
                        }
                        if (gpuGizmoRegions != 0)
                            oss << " gpu#" << gpuTimestamps.serial << '=' << gpuGizmoMs
                                << "ms regions=" << gpuGizmoRegions;
                    }

                    if (!topPassProfiles.empty()) {
                        oss << "\n    CallbackTop:";
                        for (const auto &entry : topPassProfiles) {
                            oss << ' ' << entry.name << '=' << (drawN ? entry.totalMs / n : 0.0) << "ms" << '('
                                << entry.calls << ')';
                        }
                    }

                    for (const auto &entry : infernux::vk::RenderGraph::GetViewPassProfiles()) {
                        oss << "\n    ViewPass V" << entry.view << "/P" << entry.pass << ' ' << entry.name
                            << ": record=" << entry.recordMs / n << "ms barrier=" << entry.barrierMs / n
                            << "ms callback=" << entry.callbackMs / n << "ms calls=" << entry.calls
                            << " gpuRegions=" << entry.gpuRegions << " gpuNotSampled=" << entry.gpuNotSampled;
                    }

                    const auto particlePassProfiles = infernux::vk::RenderGraph::GetParticlePassProfiles(32);
                    if (!particlePassProfiles.empty()) {
                        oss << "\n    ParticleGPU:";
                        for (const auto &entry : particlePassProfiles) {
                            oss << ' ' << entry.name << "{passes=" << entry.passCalls
                                << ",dispatch=" << entry.dispatchCalls << ",direct=" << entry.directDispatchCalls
                                << ",indirect=" << entry.indirectDispatchCalls << ",workgroups=" << entry.workgroups
                                << ",inputs=" << entry.inputCount << ",barrierPhases=" << entry.barrierPhases
                                << ",draws=" << entry.drawCalls << ",indirectDraws=" << entry.indirectDrawCalls << '}';
                        }
                    }
                }
            }

            oss << "\n  Pacing: target=" << (_pacingAccum.targetFps / kWindow)
                << "fps budget=" << (_pacingAccum.frameBudgetMs / kWindow)
                << "ms elapsed=" << (_pacingAccum.elapsedBeforeSleepMs / kWindow)
                << "ms reqSleep=" << (_pacingAccum.requestedSleepMs / kWindow)
                << "ms actualSleep=" << (_pacingAccum.actualSleepMs / kWindow)
                << "ms slept=" << _pacingAccum.sleptFrames << '/' << static_cast<int>(kWindow)
                << " idle=" << _pacingAccum.idleFrames << '/' << static_cast<int>(kWindow)
                << " wakeByEvent=" << _pacingAccum.wokeByEventFrames << " wakeInput=" << _pacingAccum.wokeByInputFrames
                << " wakeWindow=" << _pacingAccum.wokeByWindowFrames << " wakeOther=" << _pacingAccum.wokeByOtherFrames
                << " input=" << _pacingAccum.inputFrames << " playBypass=" << _pacingAccum.playBypassFrames
                << " events/frame=" << (static_cast<double>(_pacingAccum.queuedEvents) / kWindow)
                << " motion/frame=" << (static_cast<double>(_pacingAccum.mouseMotionEvents) / kWindow)
                << " motionDispatch/frame=" << (static_cast<double>(_pacingAccum.dispatchedMouseMotions) / kWindow);
            oss << "\n    InputDetail: begin=" << (_pacingAccum.inputBeginFrameMs / kWindow)
                << "ms pump=" << (_pacingAccum.inputPumpEventsMs / kWindow)
                << "ms peep+dispatch=" << (_pacingAccum.inputPeepEventsMs / kWindow)
                << "ms imgui=" << (_pacingAccum.inputImGuiDispatchMs / kWindow)
                << "ms manager=" << (_pacingAccum.inputManagerDispatchMs / kWindow)
                << "ms synthetic=" << (_pacingAccum.inputSyntheticDispatchMs / kWindow)
                << "ms windowQuery=" << (_pacingAccum.inputWindowQueryMs / kWindow) << "ms";

            oss << "\n  Scene: editorCam=" << (_sceneAccum.editorCameraMs / kWindow)
                << "ms editor=" << (_sceneAccum.editorUpdateMs / kWindow)
                << "ms pending=" << (_sceneAccum.pendingStartsMs / kWindow)
                << "ms syncExt=" << (_sceneAccum.syncExternalMovesMs / kWindow)
                << "ms syncCol=" << (_sceneAccum.syncCollidersMs / kWindow)
                << "ms fixed=" << (_sceneAccum.fixedUpdateMs / kWindow)
                << "ms physics=" << (_sceneAccum.physicsStepMs / kWindow)
                << "ms events=" << (_sceneAccum.physicsEventsMs / kWindow)
                << "ms syncRb=" << (_sceneAccum.syncRigidbodiesMs / kWindow)
                << "ms interp=" << (_sceneAccum.interpolationMs / kWindow)
                << "ms gameplay=" << (_sceneAccum.gameplayUpdateMs / kWindow)
                << "ms late=" << (_sceneAccum.lateUpdateMs / kWindow) << "ms audio=" << (_sceneAccum.audioMs / kWindow)
                << "ms end=" << (_sceneAccum.endFrameMs / kWindow)
                << "ms fixedSteps=" << (_sceneAccum.fixedSteps / kWindow)
                << " fixedFrame=" << sampleMean(_fixedStepFrameSamples) << "ms(" << _fixedStepFrameSamples.size()
                << ") renderFrame=" << sampleMean(_renderOnlyFrameSamples) << "ms(" << _renderOnlyFrameSamples.size()
                << ')';

            double guiTotal = 0.0;
            oss << "\n  UI:";
            for (const auto &name : m_gui->GetRenderableOrder()) {
                auto it = _guiAccum.find(name);
                if (it == _guiAccum.end()) {
                    continue;
                }
                const double avgMs = it->second / kWindow;
                guiTotal += avgMs;
                oss << ' ' << name << '=' << avgMs << "ms";
            }
            oss << " total=" << guiTotal << "ms";

            // Inspector sub-timing breakdown
            if (!_inspSubAccum.empty()) {
                std::vector<std::pair<std::string, double>> inspItems(_inspSubAccum.begin(), _inspSubAccum.end());
                std::sort(inspItems.begin(), inspItems.end(),
                          [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
                auto isCountMetric = [](const std::string &key) {
                    constexpr char kSuffix[] = "_count";
                    return key.size() >= (sizeof(kSuffix) - 1) &&
                           key.compare(key.size() - (sizeof(kSuffix) - 1), sizeof(kSuffix) - 1, kSuffix) == 0;
                };
                oss << "\n    Inspector:";
                for (const auto &kv : inspItems) {
                    oss << ' ' << kv.first << '=' << (kv.second / kWindow);
                    if (!isCountMetric(kv.first))
                        oss << "ms";
                }
            }
            if (!_hierSubAccum.empty()) {
                std::vector<std::pair<std::string, double>> hierItems(_hierSubAccum.begin(), _hierSubAccum.end());
                std::sort(hierItems.begin(), hierItems.end(),
                          [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
                oss << "\n    Hierarchy:";
                for (const auto &kv : hierItems)
                    oss << ' ' << kv.first << '=' << (kv.second / kWindow) << "ms";
            }
            if (!_projectSubAccum.empty()) {
                std::vector<std::pair<std::string, double>> projectItems(_projectSubAccum.begin(),
                                                                         _projectSubAccum.end());
                std::sort(projectItems.begin(), projectItems.end(),
                          [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
                oss << "\n    Project:";
                for (const auto &kv : projectItems)
                    oss << ' ' << kv.first << '=' << (kv.second / kWindow) << "ms";
            }
            // Periodic profiling is diagnostic telemetry, not an engine fault.
            // Keep it visible without poisoning Console warning/error gates.
            INXLOG_INFO(oss.str());
#if INFERNUX_FRAME_PROFILE_TERMINAL
            std::cerr << oss.str() << std::endl;
#endif

            for (double &value : _fpAccum) {
                value = 0.0;
            }
            _deltaAccumMs = 0.0;
            _srpSceneViewMs = 0;
            _srpGameViewMs = 0;
            _frameTimeSamples.clear();
            _fixedStepFrameSamples.clear();
            _renderOnlyFrameSamples.clear();
            _sceneAccum = {};
            _guiAccum.clear();
            _inspSubAccum.clear();
            _hierSubAccum.clear();
            _projectSubAccum.clear();
            _pacingAccum = {};
            _detailAccum = {};
            if (m_vkCore) {
                m_vkCore->ResetDrawSubTimings();
            }
            infernux::vk::RenderGraph::ResetExecuteProfileSnapshot();
            SceneRenderBridge::Instance().ResetProfileSnapshot();
            ScriptableRenderContext::ResetProfileSnapshot();
            _fpCounter = 0;
            _fpLastReport = _fpReportNow;
        }
    }
#endif
    if (pace && (windowSuspended || !renderSubmitted))
        SDL_Delay(windowSuspended ? 16 : 1);
    return {true, renderSubmitted, presentationAcceptedNs};
}

// ============================================================================
// DrawFrame sub-methods
// ============================================================================

bool InxRenderer::CheckAndApplyMsaaRequest()
{
    m_sceneRequestedMsaaSamples = 0;
    m_gameRequestedMsaaSamples = 0;
    for (const auto &view : m_nextScheduledViews) {
        if (view.scene) {
            m_sceneRequestedMsaaSamples = view.graph->GetRequestedMsaaSamples();
        } else if (!view.graph->HasOutputTexture()) {
            const int request = view.graph->GetRequestedMsaaSamples();
            if (m_gameRequestedMsaaSamples == 0)
                m_gameRequestedMsaaSamples = request;
            else if (request != 0 && request != m_gameRequestedMsaaSamples)
                INXLOG_ERROR("Active game cameras requested conflicting MSAA sample counts");
        }
    }

    const MsaaRequestResolution resolution =
        ResolveMsaaRequests(m_sceneRequestedMsaaSamples, m_gameRequestedMsaaSamples);
    const int currentSamples = GetMsaaSamples();

    auto rejectRequest = [this, resolution](MsaaRequestStatus status, uint32_t supportedMask) {
        const uint64_t signature = (static_cast<uint64_t>(status) << 56u) |
                                   (static_cast<uint64_t>(m_sceneRequestedMsaaSamples & 0xff) << 40u) |
                                   (static_cast<uint64_t>(m_gameRequestedMsaaSamples & 0xff) << 32u) | supportedMask;
        if (signature == m_lastMsaaRejectionSignature)
            return;
        m_lastMsaaRejectionSignature = signature;
        ++m_msaaRejectedRequestCount;

        if (status == MsaaRequestStatus::ConflictingRequests) {
            INXLOG_ERROR(
                "Scene and Game render graphs request incompatible global MSAA settings (Scene=",
                m_sceneRequestedMsaaSamples, "x, Game=", m_gameRequestedMsaaSamples,
                "x). Keeping the last-known-good setting; both graphs must agree while material pipelines are shared.");
        } else if (status == MsaaRequestStatus::InvalidSceneRequest ||
                   status == MsaaRequestStatus::InvalidGameRequest) {
            INXLOG_ERROR("Invalid render graph MSAA request (Scene=", m_sceneRequestedMsaaSamples,
                         "x, Game=", m_gameRequestedMsaaSamples, "x). Valid values are 0, 1, 2, 4, and 8.");
        } else {
            INXLOG_ERROR("Requested ", resolution.samples,
                         "x MSAA is unsupported by the HDR color/depth target pair (supported mask=", supportedMask,
                         "). Keeping the last-known-good setting.");
        }
    };

    if (resolution.status == MsaaRequestStatus::NoRequest) {
        m_msaaRequestConflict = false;
        m_lastMsaaRejectionSignature = 0;
        SetEffectiveGraphMsaaSamples(currentSamples);
        return false;
    }

    if (!resolution.IsAccepted()) {
        m_msaaRequestConflict = resolution.status == MsaaRequestStatus::ConflictingRequests;
        rejectRequest(resolution.status, GetSupportedMsaaSampleMask());
        SetEffectiveGraphMsaaSamples(currentSamples);
        return true;
    }

    const uint32_t supportedMask = GetSupportedMsaaSampleMask();
    if (!SupportsMsaaSampleCount(static_cast<rhi::SampleCountMask>(supportedMask), resolution.samples)) {
        m_msaaRequestConflict = false;
        rejectRequest(MsaaRequestStatus::Accepted, supportedMask);
        SetEffectiveGraphMsaaSamples(currentSamples);
        return false;
    }

    m_msaaRequestConflict = false;
    m_lastMsaaRejectionSignature = 0;
    SetEffectiveGraphMsaaSamples(resolution.samples);
    const bool reconfigured = ApplyMsaaSamples(resolution.samples, "render graph");
    if (!reconfigured)
        SetEffectiveGraphMsaaSamples(GetMsaaSamples());
    return reconfigured;
}

void InxRenderer::StageEngineGlobalsUBO()
{
    EngineGlobalsUBO globals{};

    // Time
    float t = m_totalTime;
    globals.time = glm::vec4(t, std::sin(t), std::cos(t), m_deltaTime);
    globals.sinTime = glm::vec4(std::sin(t / 20.0f), std::sin(t / 4.0f), std::sin(t * 2.0f), std::sin(t * 3.0f));
    globals.cosTime = glm::vec4(std::cos(t / 20.0f), std::cos(t / 4.0f), std::cos(t * 2.0f), std::cos(t * 3.0f));

    // Camera and screen fields are populated by each RenderView.

    // Frame
    float dt = m_deltaTime > kEpsilon ? m_deltaTime : kEpsilon;
    globals.frameParams = glm::vec4(static_cast<float>(GetNextFrameIndex()), m_smoothDeltaTime, 1.0f / dt, 0.0f);

    m_vkCore->StageGlobals(globals);
}

void InxRenderer::CleanupDrawCallBuffers()
{
    if (!m_vkCore)
        return;

#if INFERNUX_FRAME_PROFILE
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();
#endif
    const auto key = [](const ScheduledView &view) {
        return std::make_pair(view.viewId, view.graph->GetObjectBindingIdentity());
    };
    const bool sameBindings =
        std::equal(m_nextScheduledViews.begin(), m_nextScheduledViews.end(), m_objectBindingViews.begin(),
                   m_objectBindingViews.end(),
                   [&](const ScheduledView &view, const auto &previous) { return key(view) == previous; });
    if (sameBindings && m_sweptObjectBufferRevision == m_vkCore->GetObjectBufferRevision())
        return;

    if (m_nextScheduledViews.empty())
        m_vkCore->ReleaseActiveDrawLists();

    // A view that reused its bindings has not stamped them during preparation.
    // The sweep must preserve the union of both reused and prepared bindings.
    m_objectBindingViews.clear();
    for (const auto &view : m_nextScheduledViews) {
        m_objectBindingViews.push_back(key(view));
        if (view.graph->ReusesObjectBindings()) {
            m_vkCore->KeepObjectBuffers(view.graph->GetCachedDrawCalls());
            m_vkCore->KeepObjectBuffers(view.graph->GetCachedShadowDrawCalls());
        }
    }
#if INFERNUX_FRAME_PROFILE
    m_frameDetailTiming.cleanupCollectIdsMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    t0 = Clock::now();
#endif
    [[maybe_unused]] const size_t activeObjectCount = m_vkCore->CleanupUnusedBuffersByFrameStamp();
    m_sweptObjectBufferRevision = m_vkCore->GetObjectBufferRevision();
#if INFERNUX_FRAME_PROFILE
    m_frameDetailTiming.cleanupActiveIds = static_cast<double>(activeObjectCount);
    m_frameDetailTiming.cleanupReleaseMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif
}

void InxRenderer::WaitForGpuIdle()
{
    if (!m_vkCore) {
        return;
    }

    if (m_sceneRenderGraph) {
        m_sceneRenderGraph->ClearCachedFrameState();
    }
    for (auto &[cameraId, graph] : m_gameRenderGraphs) {
        (void)cameraId;
        graph->ClearCachedFrameState();
    }

    // Drop the currently staged draw list as well. If the upcoming frame skips
    // re-submitting one of the graphs (for example because no valid game camera
    // exists yet), VkCore must not keep drawing the previous scene's objects.
    m_vkCore->SetDrawCalls(nullptr);

    // Scene replacement destroys meshes/materials/textures immediately.
    // A full device drain avoids probabilistic crashes from old-scene resources
    // still being referenced by previously submitted command buffers.
    m_vkCore->GetDeviceContext().WaitIdle();
    m_vkCore->FlushRetiredGpuResources();

    // Pump the OS message queue so Windows does not flag the application
    // as "Not Responding" during long GPU drains (scene switch, MSAA change).
    SDL_PumpEvents();
}

void InxRenderer::PrepareSceneAssets()
{
    auto &bridge = SceneRenderBridge::Instance();
    bridge.PrepareFrame();
    m_vkCore->PrepareSceneAssets(bridge.BuildDrawCalls().drawCalls);
    SDL_PumpEvents();
}

size_t InxRenderer::GetPendingMeshUploadCount() const
{
    return m_vkCore ? m_vkCore->GetPendingMeshUploadCount() : 0;
}

uint64_t InxRenderer::GetSubmittedMeshUploadCount() const
{
    return m_vkCore ? m_vkCore->GetSubmittedMeshUploadCount() : 0;
}

size_t InxRenderer::GetResidentMeshVertexBufferCount() const
{
    return m_vkCore ? m_vkCore->GetResidentMeshVertexBufferCount() : 0;
}

uint64_t InxRenderer::GetCompletedMeshUploadCount() const
{
    return m_vkCore ? m_vkCore->GetCompletedMeshUploadCount() : 0;
}

uint64_t InxRenderer::GetAsyncMeshUploadCount() const
{
    return m_vkCore ? m_vkCore->GetAsyncMeshUploadCount() : 0;
}

size_t InxRenderer::GetPendingTextureCpuLoadCount() const
{
    return m_vkCore ? m_vkCore->GetPendingTextureCpuLoadCount() : 0;
}

size_t InxRenderer::GetPendingTextureUploadCount() const
{
    return m_vkCore ? m_vkCore->GetPendingTextureUploadCount() : 0;
}

uint64_t InxRenderer::GetSubmittedTextureUploadCount() const
{
    return m_vkCore ? m_vkCore->GetSubmittedTextureUploadCount() : 0;
}

uint64_t InxRenderer::GetCompletedTextureUploadCount() const
{
    return m_vkCore ? m_vkCore->GetCompletedTextureUploadCount() : 0;
}

uint64_t InxRenderer::GetAsyncTextureUploadCount() const
{
    return m_vkCore ? m_vkCore->GetAsyncTextureUploadCount() : 0;
}

uint64_t InxRenderer::GetStagingPoolBytes() const
{
    return m_vkCore ? m_vkCore->GetStagingPoolBytes() : 0;
}

size_t InxRenderer::GetStagingPoolBufferCount() const
{
    return m_vkCore ? m_vkCore->GetStagingPoolBufferCount() : 0;
}

uint64_t InxRenderer::GetStagingAllocationCount() const
{
    return m_vkCore ? m_vkCore->GetStagingAllocationCount() : 0;
}

uint64_t InxRenderer::GetStagingReuseCount() const
{
    return m_vkCore ? m_vkCore->GetStagingReuseCount() : 0;
}

uint64_t InxRenderer::GetStagingDiscardCount() const
{
    return m_vkCore ? m_vkCore->GetStagingDiscardCount() : 0;
}

uint64_t InxRenderer::GetTextureGpuResidentBytes() const
{
    return m_vkCore ? m_vkCore->GetTextureGpuResidentBytes() : 0;
}

uint64_t InxRenderer::GetTextureGpuBudgetBytes() const
{
    return m_vkCore ? m_vkCore->GetTextureGpuBudgetBytes() : 0;
}

size_t InxRenderer::GetTextureGpuCacheEntryCount() const
{
    return m_vkCore ? m_vkCore->GetTextureGpuCacheEntryCount() : 0;
}

size_t InxRenderer::GetRetiredTextureGpuLeaseCount() const
{
    return m_vkCore ? m_vkCore->GetRetiredTextureGpuLeaseCount() : 0;
}

uint64_t InxRenderer::GetTextureGpuEvictionCount() const
{
    return m_vkCore ? m_vkCore->GetTextureGpuEvictionCount() : 0;
}

void InxRenderer::SetTextureGpuBudgetBytes(uint64_t bytes)
{
    if (!m_vkCore)
        throw std::logic_error("Cannot set the GPU texture budget before renderer initialization");
    m_vkCore->SetTextureGpuBudgetBytes(bytes);
}

size_t InxRenderer::TrimTextureGpuBudget()
{
    if (!m_vkCore)
        throw std::logic_error("Cannot trim the GPU texture budget before renderer initialization");
    return m_vkCore->TrimTextureGpuBudget();
}

uint64_t InxRenderer::GetMeshGpuResidentBytes() const
{
    return m_vkCore ? m_vkCore->GetMeshGpuResidentBytes() : 0;
}

uint64_t InxRenderer::GetMeshGpuBudgetBytes() const
{
    return m_vkCore ? m_vkCore->GetMeshGpuBudgetBytes() : 0;
}

size_t InxRenderer::GetMeshGpuCacheEntryCount() const
{
    return m_vkCore ? m_vkCore->GetMeshGpuCacheEntryCount() : 0;
}

size_t InxRenderer::GetRetiredMeshGpuLeaseCount() const
{
    return m_vkCore ? m_vkCore->GetRetiredMeshGpuLeaseCount() : 0;
}

uint64_t InxRenderer::GetMeshGpuEvictionCount() const
{
    return m_vkCore ? m_vkCore->GetMeshGpuEvictionCount() : 0;
}

MeshIndexFormat InxRenderer::GetObjectMeshIndexFormat(uint64_t objectId) const
{
    if (!m_vkCore)
        throw std::logic_error("Cannot inspect a GPU mesh before renderer initialization");
    return m_vkCore->GetObjectIndexFormat(objectId);
}

uint64_t InxRenderer::GetObjectMeshIndexBufferBytes(uint64_t objectId) const
{
    return m_vkCore ? m_vkCore->GetObjectIndexBufferBytes(objectId) : 0;
}

void InxRenderer::SetMeshGpuBudgetBytes(uint64_t bytes)
{
    if (!m_vkCore)
        throw std::logic_error("Cannot set the GPU mesh budget before renderer initialization");
    m_vkCore->SetMeshGpuBudgetBytes(bytes);
}

size_t InxRenderer::TrimMeshGpuBudget()
{
    if (!m_vkCore)
        throw std::logic_error("Cannot trim the GPU mesh budget before renderer initialization");
    return m_vkCore->TrimMeshGpuBudget();
}

uint64_t InxRenderer::GetGpuResidencyBudgetBytes() const
{
    if (m_gpuResidencyBudgetBytes != 0)
        return m_gpuResidencyBudgetBytes;
    if (!m_vkCore)
        return 0;
    const auto statistics = vk::QueryVmaRuntimeStatistics(m_vkCore->GetDeviceContext().GetVmaAllocator(),
                                                          m_vkCore->GetDeviceContext().GetPhysicalDevice());
    const uint64_t totalManagedBudget = (statistics.deviceLocalBudgetBytes / 10ULL) * 7ULL;
    const uint64_t editorTextureBudget = m_gui && !m_guiPlayerMode ? m_gui->GetImGuiTextureBudgetBytes() : 0;
    return totalManagedBudget > editorTextureBudget ? totalManagedBudget - editorTextureBudget : totalManagedBudget;
}

bool InxRenderer::HasPendingBackgroundGpuWork() const
{
    if (!m_vkCore)
        return false;
    const auto &resources = m_vkCore->GetResourceManager();
    return resources.GetPendingAsyncGraphicsSubmissionCount() != 0 || resources.GetPendingGpuTransferCount() != 0;
}

GpuResidencySnapshot InxRenderer::GetGpuResidencySnapshot() const
{
    GpuResidencySnapshot snapshot;
    if (!m_vkCore)
        return snapshot;

    const auto statistics = vk::QueryVmaRuntimeStatistics(m_vkCore->GetDeviceContext().GetVmaAllocator(),
                                                          m_vkCore->GetDeviceContext().GetPhysicalDevice());
    snapshot.runtimeBudgetBytes = GetGpuResidencyBudgetBytes();
    snapshot.editorTextureBudgetBytes = m_gui && !m_guiPlayerMode ? m_gui->GetImGuiTextureBudgetBytes() : 0;
    snapshot.budgetBytes = snapshot.runtimeBudgetBytes + snapshot.editorTextureBudgetBytes;
    snapshot.allocatorAllocationBytes = statistics.allocationBytes;
    snapshot.allocatorBlockBytes = statistics.blockBytes;
    snapshot.deviceLocalAllocationBytes = statistics.deviceLocalAllocationBytes;
    snapshot.deviceLocalUsageBytes = statistics.deviceLocalUsageBytes;
    snapshot.deviceLocalBudgetBytes = statistics.deviceLocalBudgetBytes;
    snapshot.allocatorAllocationCount = statistics.allocationCount;
    snapshot.meshBytes = m_vkCore->GetMeshGpuResidentBytes();
    snapshot.textureBytes = m_vkCore->GetTextureGpuResidentBytes();
    snapshot.imguiTextureBytes = m_gui ? m_gui->GetImGuiTextureResidentBytes() : 0;
    snapshot.pendingImguiTextureBytes = m_gui ? m_gui->GetAllocatedImGuiTextureUploadBytes() : 0;
    snapshot.stagingPoolBytes = m_vkCore->GetStagingPoolBytes();
    snapshot.pendingReadbackBytes = m_vkCore->GetResourceManager().GetPendingImageReadbackBytes();
    snapshot.pendingReadbackCount = m_vkCore->GetResourceManager().GetPendingImageReadbackCount();
    snapshot.pendingGpuTransferCount = m_vkCore->GetResourceManager().GetPendingGpuTransferCount();
    snapshot.uploadTimelineEnabled = m_vkCore->GetResourceManager().IsUploadTimelineEnabled();
    snapshot.timelineUploadPublicationCount = m_vkCore->GetResourceManager().GetTimelineUploadPublicationCount();
    snapshot.deviceWaitIdleCount = m_vkCore->GetDeviceContext().GetWaitIdleCount();
    snapshot.shaderHotReloadRetirementCount = m_vkCore->GetShaderHotReloadRetirementCount();
    snapshot.pendingAsyncGraphicsSubmissionCount =
        m_vkCore->GetResourceManager().GetPendingAsyncGraphicsSubmissionCount();
    snapshot.asyncGraphicsSubmissionCount = m_vkCore->GetResourceManager().GetAsyncGraphicsSubmissionCount();
    snapshot.renderTargetBytes = (m_sceneRenderTarget ? m_sceneRenderTarget->GetResidentBytes() : 0) +
                                 (m_gameRenderTarget ? m_gameRenderTarget->GetResidentBytes() : 0);
    snapshot.renderGraphBytes = m_sceneRenderGraph ? m_sceneRenderGraph->GetTransientResidentBytes() : 0;
    for (const auto &[cameraId, graph] : m_gameRenderGraphs) {
        (void)cameraId;
        snapshot.renderGraphBytes += graph->GetTransientResidentBytes();
    }
    // Legacy telemetry field; temporary image ownership is in each RenderGraph.
    snapshot.transientPoolBytes = 0;
    const MaterialGpuResidencySnapshot materialResidency = m_vkCore->GetMaterialGpuResidency();
    snapshot.materialUboBytes = materialResidency.uboBytes;
    snapshot.materialRenderDataCount = materialResidency.renderDataCount;
    snapshot.runtimeMaterialCount = materialResidency.runtimeMaterialCount;
    snapshot.assetMaterialCount = materialResidency.assetMaterialCount;
    snapshot.materialDescriptorSetCount = materialResidency.descriptorSetCount;
    snapshot.pendingMaterialTextureDescriptorSetCount = materialResidency.pendingTextureDescriptorSetCount;
    snapshot.retiredMaterialDescriptorSetCount = materialResidency.retiredDescriptorSetCount;
    snapshot.materialDescriptorPoolCount = materialResidency.descriptorPoolCount;
    snapshot.materialPipelineCount = materialResidency.pipelineCount;
    snapshot.runtimeMeshEntryCount = m_vkCore->GetRuntimeMeshGpuEntryCount();
    snapshot.runtimeMeshBytes = m_vkCore->GetRuntimeMeshGpuResidentBytes();
    snapshot.scheduledReleaseBytes = m_vkCore->GetRetiredMeshGpuLeaseBytes() +
                                     m_vkCore->GetRetiredTextureGpuLeaseBytes() +
                                     (m_gui ? m_gui->GetScheduledTextureReleaseBytes() : 0);
    snapshot.trackedBytes = snapshot.meshBytes + snapshot.textureBytes + snapshot.imguiTextureBytes +
                            snapshot.pendingImguiTextureBytes + snapshot.stagingPoolBytes +
                            snapshot.pendingReadbackBytes + snapshot.renderTargetBytes + snapshot.renderGraphBytes +
                            snapshot.transientPoolBytes + snapshot.materialUboBytes;
    snapshot.unclassifiedBytes = snapshot.allocatorAllocationBytes > snapshot.trackedBytes
                                     ? snapshot.allocatorAllocationBytes - snapshot.trackedBytes
                                     : 0;
    // These effective values are eviction projections, not available space
    // for a new allocation. Admission must use the actual VMA allocation above.
    snapshot.effectiveAllocationBytes = snapshot.deviceLocalAllocationBytes > snapshot.scheduledReleaseBytes
                                            ? snapshot.deviceLocalAllocationBytes - snapshot.scheduledReleaseBytes
                                            : 0;
    snapshot.editorTextureEffectiveAllocationBytes =
        m_guiPlayerMode ? uint64_t{0} : snapshot.imguiTextureBytes + snapshot.pendingImguiTextureBytes;
    const uint64_t scheduledGuiBytes = m_gui ? m_gui->GetScheduledTextureReleaseBytes() : 0;
    snapshot.editorTextureEffectiveAllocationBytes -=
        std::min(snapshot.editorTextureEffectiveAllocationBytes, scheduledGuiBytes);
    snapshot.runtimeEffectiveAllocationBytes =
        snapshot.effectiveAllocationBytes > snapshot.editorTextureEffectiveAllocationBytes
            ? snapshot.effectiveAllocationBytes - snapshot.editorTextureEffectiveAllocationBytes
            : 0;
    snapshot.runtimeOverBudgetBytes = snapshot.runtimeEffectiveAllocationBytes > snapshot.runtimeBudgetBytes
                                          ? snapshot.runtimeEffectiveAllocationBytes - snapshot.runtimeBudgetBytes
                                          : 0;
    snapshot.editorTextureOverBudgetBytes =
        snapshot.editorTextureEffectiveAllocationBytes > snapshot.editorTextureBudgetBytes
            ? snapshot.editorTextureEffectiveAllocationBytes - snapshot.editorTextureBudgetBytes
            : 0;
    snapshot.overBudgetBytes = snapshot.runtimeOverBudgetBytes + snapshot.editorTextureOverBudgetBytes;
    return snapshot;
}

void InxRenderer::PollGpuCompletions()
{
    if (m_vkCore) {
        m_vkCore->CollectRetiredGpuResources();
        if (m_vkCore->GetDeviceContext().GetRhiDevice().HasDeviceLocalAllocationPressure())
            (void)TrimGpuResidencyBudget();
    }
    if (m_gui)
        m_gui->PublishCompletedTextureUploads();
}

void InxRenderer::RetireReplacedViewTargets()
{
    if (m_replacedViewTargets.empty())
        return;
    // Cached GUI data can submit the old descriptor again. After it is
    // replaced, retire against the last epoch that could have used it.
    auto &retirement = m_vkCore->GetRetirementQueue();
    const auto epoch = m_vkCore->GetBackendContext().Queues().GetLastReservedCompletionEpoch();
    const auto retired =
        std::remove_if(m_replacedViewTargets.begin(), m_replacedViewTargets.end(), [&](const auto &target) {
            if (m_gui->IsTextureReferencedByCurrentDrawData(target->GetImGuiTextureId()))
                return false;
            target->RetireResourcesAfter(retirement, epoch);
            return true;
        });
    m_replacedViewTargets.erase(retired, m_replacedViewTargets.end());
}

uint64_t InxRenderer::RegisterFrameWork(FrameWorkOwner &owner)
{
    m_frameWorkOwner = &owner;
    m_vkCore->GetDeviceContext().GetRhiDevice().SetManagedHeapLimits([this] {
        if (!m_gpuResidencyBudgetBytes)
            return uint64_t{0};
        const uint64_t editor = m_gui && !m_guiPlayerMode ? m_gui->GetImGuiTextureBudgetBytes() : 0;
        return (std::min)(m_gpuResidencyBudgetBytes, (std::numeric_limits<uint64_t>::max)() - editor) + editor;
    });
    const auto textureSource = owner.AddSource(m_vkCore->GetTextureWorkSource(owner));
    m_vkCore->RegisterMeshUploadWork(owner);
    if (m_particleGpuSystemManager)
        m_particleGpuSystemManager->RegisterMeshUploadWork(owner);
    m_captureService = std::make_unique<CaptureService>(owner);
    return textureSource;
}

void InxRenderer::SetGpuResidencyBudgetBytes(uint64_t bytes)
{
    if (bytes == 0)
        throw std::invalid_argument("GPU residency budget must be greater than zero");
    m_gpuResidencyBudgetBytes = bytes;
    (void)TrimGpuResidencyBudget();
}

size_t InxRenderer::TrimGpuResidencyBudget()
{
    if (!m_vkCore)
        throw std::logic_error("Cannot trim GPU residency before renderer initialization");

    const auto snapshot = GetGpuResidencySnapshot();
    const auto pressure = m_vkCore->GetDeviceContext().GetRhiDevice().TakeDeviceLocalAllocationPressure();
    const auto uncoveredPressure = pressure - (std::min)(pressure, snapshot.scheduledReleaseBytes);
    auto bytesToRetire = (std::max)(snapshot.runtimeOverBudgetBytes, uncoveredPressure);
    size_t evicted = m_gui ? m_gui->TrimImGuiTextureBudget() : 0;
    while (bytesToRetire) {
        enum class Domain
        {
            None,
            Mesh,
            Texture,
        };
        Domain domain = Domain::None;
        GpuEvictionCandidate selected;
        const auto consider = [&](Domain candidateDomain, GpuEvictionCandidate candidate) {
            if (!candidate.valid)
                return;
            if (!selected.valid || candidate.lastUsedFrame < selected.lastUsedFrame ||
                (candidate.lastUsedFrame == selected.lastUsedFrame &&
                 candidate.residentBytes > selected.residentBytes)) {
                selected = candidate;
                domain = candidateDomain;
            }
        };
        consider(Domain::Mesh, m_vkCore->PeekOldestMeshGpuEvictable());
        consider(Domain::Texture, m_vkCore->PeekOldestTextureGpuEvictable());
        if (!selected.valid)
            break;

        uint64_t releasedBytes = 0;
        switch (domain) {
        case Domain::Mesh:
            releasedBytes = m_vkCore->EvictOldestMeshGpu();
            break;
        case Domain::Texture:
            releasedBytes = m_vkCore->EvictOldestTextureGpu();
            break;
        case Domain::None:
            break;
        }
        if (releasedBytes == 0)
            break;
        bytesToRetire -= (std::min)(bytesToRetire, releasedBytes);
        ++evicted;
    }
    return evicted;
}

std::vector<GpuAssetResidencyRecord> InxRenderer::GetAssetGpuResidency() const
{
    return m_vkCore ? m_vkCore->GetAssetGpuResidency() : std::vector<GpuAssetResidencyRecord>{};
}

void InxRenderer::LoadShader(const char *name, const std::vector<char> &code, const char *type)
{
    m_vkCore->LoadShader(name, code, type);
}

void InxRenderer::SetShaderAssetResolver(std::function<bool(const std::string &, const std::string &)> resolver)
{
    if (m_vkCore)
        m_vkCore->SetShaderAssetResolver(std::move(resolver));
}

bool InxRenderer::PublishShaderProgramArtifact(const ShaderProgramArtifact &artifact)
{
    if (!m_vkCore)
        return false;
    const bool unchanged = m_vkCore->HasShaderProgramArtifact(artifact.key);
    if (!m_vkCore->PublishShaderProgramArtifact(artifact))
        return false;
    if (!unchanged && m_screenUIRenderer)
        m_screenUIRenderer->InvalidateMaterialProgram(artifact.key.stages);
    return true;
}

bool InxRenderer::PublishShaderProgramArtifacts(const std::vector<ShaderProgramArtifact> &artifacts)
{
    if (!m_vkCore)
        return false;
    std::vector<ShaderStagePair> changed;
    for (const auto &artifact : artifacts)
        if (!m_vkCore->HasShaderProgramArtifact(artifact.key))
            changed.push_back(artifact.key.stages);
    if (!m_vkCore->PublishShaderProgramArtifacts(artifacts))
        return false;
    if (m_screenUIRenderer)
        for (const auto &stages : changed)
            m_screenUIRenderer->InvalidateMaterialProgram(stages);
    return true;
}

void InxRenderer::InvalidateUIMaterialProgram(const ShaderStagePair &stages)
{
    // Reload can touch a UI pair that was never drawn. Such a pair has no UI
    // renderer owner, so invalidating draw caches alone would retain its
    // artifact and Forward modules until engine shutdown.
    const auto oldProgram = m_vkCore ? m_vkCore->ShareShaderProgramArtifact(stages) : nullptr;
    if (m_screenUIRenderer)
        m_screenUIRenderer->InvalidateMaterialProgram(stages);
    if (oldProgram)
        m_vkCore->ReleaseUIShaderProgramArtifact(oldProgram->key);
}

void InxRenderer::RetireShaderProgramArtifact(const ShaderProgramKey &key)
{
    if (m_screenUIRenderer)
        m_screenUIRenderer->InvalidateMaterialProgram(key.stages);
    if (m_vkCore)
        m_vkCore->RetireShaderProgramArtifact(key);
}

bool InxRenderer::HasShaderProgramArtifact(const ShaderProgramKey &programKey) const
{
    return m_vkCore && m_vkCore->HasShaderProgramArtifact(programKey);
}

std::shared_ptr<const ShaderProgramArtifact>
InxRenderer::ResolveShaderProgramArtifact(const std::shared_ptr<InxMaterial> &material,
                                          std::optional<ShaderProgramDomain> expectedDomain)
{
    if (!m_vkCore || !material)
        return nullptr;
    if (m_shaderProgramArtifactResolver)
        m_shaderProgramArtifactResolver(material, expectedDomain);
    auto artifact = m_vkCore->ShareShaderProgramArtifact(material->GetShaderStages());
    if (artifact &&
        (artifact->domain == ShaderProgramDomain::ScreenUI || artifact->domain == ShaderProgramDomain::WorldUI ||
         (expectedDomain && artifact->domain != *expectedDomain))) {
        throw std::runtime_error("Material shader domain mismatch before non-UI resolution");
    }
    return artifact;
}

void InxRenderer::SetShaderProgramArtifactResolver(
    std::function<void(const std::shared_ptr<InxMaterial> &, std::optional<ShaderProgramDomain>)> resolver)
{
    m_shaderProgramArtifactResolver = std::move(resolver);
    if (m_vkCore)
        m_vkCore->SetShaderProgramArtifactResolver(m_shaderProgramArtifactResolver);
}

void InxRenderer::SetMaterialShaderDomainInspector(
    std::function<std::optional<ShaderProgramDomain>(const std::shared_ptr<InxMaterial> &)> inspector)
{
    m_materialShaderDomainInspector = std::move(inspector);
}

bool InxRenderer::CanPreviewMaterialOnMesh(const std::shared_ptr<InxMaterial> &material) const
{
    if (!material)
        return false;
    try {
        const auto domain = m_materialShaderDomainInspector ? m_materialShaderDomainInspector(material) : std::nullopt;
        return !domain || *domain == ShaderProgramDomain::Mesh;
    } catch (const ShaderProgramSelectionError &) {
        return false;
    }
}

void InxRenderer::SetUIMaterialShaderValidator(
    std::function<bool(const std::shared_ptr<InxMaterial> &, ShaderProgramDomain)> validator)
{
    m_uiMaterialShaderValidator = std::move(validator);
}

void InxRenderer::StoreShaderRenderMeta(const std::string &shaderId, const std::string &cullMode,
                                        const std::string &depthWrite, const std::string &depthTest,
                                        const std::string &blend, int queue, const std::string &passTag,
                                        const std::string &stencil, const std::string &alphaClip)
{
    m_vkCore->StoreShaderRenderMeta(shaderId, cullMode, depthWrite, depthTest, blend, queue, passTag, stencil,
                                    alphaClip);
}

bool InxRenderer::HasShader(const std::string &name, const std::string &type) const
{
    if (!m_vkCore)
        return false;
    return m_vkCore->HasShader(name, type);
}

bool InxRenderer::GetUserEvent()
{
    return m_view->GetUserEvent();
}

uint64_t InxRenderer::QueueSyntheticKeyInput(int scancode, bool pressed, bool repeat)
{
    return m_view ? m_view->QueueSyntheticKeyInput(scancode, pressed, repeat) : 0;
}

uint64_t InxRenderer::QueueSyntheticMouseButtonInput(int button, bool pressed, float x, float y)
{
    return m_view ? m_view->QueueSyntheticMouseButtonInput(button, pressed, x, y) : 0;
}

uint64_t InxRenderer::QueueSyntheticMouseMotionInput(float x, float y, float deltaX, float deltaY)
{
    return m_view ? m_view->QueueSyntheticMouseMotionInput(x, y, deltaX, deltaY) : 0;
}

uint64_t InxRenderer::QueueSyntheticMouseWheelInput(float horizontal, float vertical)
{
    return m_view ? m_view->QueueSyntheticMouseWheelInput(horizontal, vertical) : 0;
}

uint64_t InxRenderer::QueueSyntheticTextInput(const std::string &text)
{
    return m_view ? m_view->QueueSyntheticTextInput(text) : 0;
}

uint64_t InxRenderer::QueueSyntheticCloseRequest()
{
    return m_view ? m_view->QueueSyntheticCloseRequest() : 0;
}

uint64_t InxRenderer::GetLastProcessedSyntheticInputSequence() const
{
    return m_view ? m_view->GetLastProcessedSyntheticInputSequence() : 0;
}

size_t InxRenderer::GetPendingSyntheticInputCount() const
{
    return m_view ? m_view->GetPendingSyntheticInputCount() : 0;
}

void InxRenderer::ShowWindow()
{
    if (m_view && m_view->ShouldRevealAfterFirstPresentation()) {
        if (!m_vkCore) {
            INXLOG_ERROR("Cannot defer Player window reveal without an initialized presentation backend");
            return;
        }
        m_vkCore->ObserveNextPresentationAsFirstVisible();
        SDL_Log("INFERNUX_WINDOW_REVEAL_ARMED");
        return;
    }
    m_view->Show();
    SDL_Log("INFERNUX_WINDOW_SHOWN");
}

bool InxRenderer::PumpStartupEvents()
{
    return !m_view || m_view->PumpStartupEvents();
}

void InxRenderer::HideWindow()
{
    m_view->Hide();
}

bool InxRenderer::IsWindowMinimized() const
{
    return m_view && m_view->IsMinimized();
}

void InxRenderer::SetWindowIcon(const std::string &iconPath)
{
    if (m_view) {
        m_view->SetWindowIcon(iconPath);
    }
}

void InxRenderer::SetWindowFullscreen(bool fullscreen)
{
    if (m_view) {
        m_view->SetWindowFullscreen(fullscreen);
    }
}

void InxRenderer::SetWindowTitle(const std::string &title)
{
    if (m_view) {
        m_view->SetWindowTitle(title);
    }
}

void InxRenderer::SetWindowMaximized(bool maximized)
{
    if (m_view) {
        m_view->SetWindowMaximized(maximized);
    }
}

void InxRenderer::SetWindowResizable(bool resizable)
{
    if (m_view) {
        m_view->SetWindowResizable(resizable);
    }
}

bool InxRenderer::IsCloseRequested() const
{
    return m_view && m_view->IsCloseRequested();
}

void InxRenderer::ConfirmClose()
{
    if (m_view) {
        m_view->ConfirmClose();
    }
}

void InxRenderer::CancelClose()
{
    if (m_view) {
        m_view->CancelClose();
    }
}

void InxRenderer::SetGUIFont(const char *fontPath, float fontSize)
{
    if (m_gui) {
        m_gui->SetGUIFont(fontPath, fontSize);
    } else {
        INXLOG_ERROR("InxGUI is not initialized.");
    }
}

void InxRenderer::InvalidateGUIFontAsset(const std::string &path)
{
    if (m_gui)
        m_gui->InvalidateFontAsset(path);
}

float InxRenderer::GetDisplayScale() const
{
    if (!m_gui)
        throw std::logic_error("Cannot query display scale before the GUI is initialized");
    return m_gui->GetDisplayScale();
}

void InxRenderer::RegisterGUIRenderable(const char *name, std::shared_ptr<InxGUIRenderable> renderable, int priority)
{
    if (m_gui) {
        m_gui->Register(name, std::move(renderable), priority);
    } else {
        INXLOG_ERROR("InxGUI is not initialized.");
    }
}

void InxRenderer::UnregisterGUIRenderable(const char *name)
{
    if (m_gui) {
        m_gui->Unregister(name);
    } else {
        INXLOG_ERROR("InxGUI is not initialized.");
    }
}

void InxRenderer::SetGUIPlayerMode(bool enabled)
{
    m_guiPlayerMode = enabled;
    if (m_gui) {
        m_gui->SetPlayerMode(enabled);
    }
}

void InxRenderer::QueueDockTabSelection(const char *windowId, bool allowDuringModal)
{
    if (m_gui) {
        m_gui->QueueDockTabSelection(windowId != nullptr ? windowId : "", allowDuringModal);
    } else {
        INXLOG_ERROR("InxGUI is not initialized.");
    }
}

uint64_t InxRenderer::SubmitTextureForImGui(const std::string &name, const unsigned char *pixels, size_t byteCount,
                                            int width, int height, rhi::FilterMode filter, bool pinned,
                                            std::function<bool()> canPublish)
{
    if (!m_gui)
        throw std::logic_error("ImGui texture submission requires an initialized GUI");
    const VkFilter vkFilter = filter == rhi::FilterMode::Nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
    return m_gui->SubmitTextureForImGui(name, pixels, byteCount, width, height, vkFilter, pinned,
                                        std::move(canPublish));
}

uint64_t InxRenderer::SubmitTextureForImGui(const std::string &name, std::shared_ptr<const TextureCpuData> pixels,
                                            rhi::FilterMode filter, bool pinned, std::function<bool()> canPublish,
                                            std::shared_ptr<FrameCpuStorageClaim> storage)
{
    if (!m_gui)
        throw std::logic_error("ImGui texture submission requires an initialized GUI");
    const VkFilter vkFilter = filter == rhi::FilterMode::Nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
    return m_gui->SubmitTextureForImGui(name, std::move(pixels), vkFilter, pinned, std::move(canPublish),
                                        std::move(storage));
}

uint64_t InxRenderer::QueryImportedTextureForImGui(const std::string &name, const std::string &textureGuid,
                                                   bool &progressed)
{
    if (!m_gui || !m_vkCore || textureGuid.empty())
        return 0;
    auto texture = m_vkCore->ResolveTextureForEditorPreview(textureGuid, progressed);
    return texture ? m_gui->PublishTextureViewForImGui(name, std::move(texture)) : 0;
}

uint64_t InxRenderer::SubmitDocumentTextureForImGui(const std::string &name,
                                                    std::shared_ptr<const TextureCpuData> pixels,
                                                    std::function<bool()> canPublish,
                                                    std::shared_ptr<FrameCpuStorageClaim> storage)
{
    if (!m_gui)
        throw std::logic_error("Document texture submission requires an initialized GUI");
    return m_gui->SubmitDocumentTextureForImGui(name, std::move(pixels), std::move(canPublish), std::move(storage));
}

uint64_t InxRenderer::GetRenderTextureUITextureId(const std::shared_ptr<rhi::RenderTexture> &texture)
{
    if (!m_gui || !texture)
        throw std::invalid_argument("RenderTexture UI binding requires an initialized GUI and a resource");
    return m_gui->PublishRenderTextureForImGui(texture);
}

std::array<uint64_t, 3> InxRenderer::GetRenderTexturePreview(const std::shared_ptr<rhi::RenderTexture> &texture)
{
    if (!texture)
        return {};
    const auto generation = texture->Acquire();
    return {GetRenderTextureUITextureId(texture), generation->width, generation->height};
}

std::array<uint64_t, 3> InxRenderer::GetImportedRenderTexturePreview(const std::string &guid)
{
    const auto found = m_assetRenderTextures.find(guid);
    if (found != m_assetRenderTextures.end())
        if (const auto owner = found->second.lock())
            return GetRenderTexturePreview(owner->texture);
    return {};
}

std::array<uint64_t, 3> InxRenderer::RequestSceneCameraPreview(uint64_t cameraId,
                                                               const RenderGraphDescription *description)
{
    if (cameraId != m_sceneCameraPreviewId) {
        m_sceneCameraPreviewId = cameraId;
        m_sceneCameraPreviewRenderedId = 0;
        m_sceneCameraPreviewDue = {};
    }
    if (description && !m_sceneCameraPreviewDescription)
        m_sceneCameraPreviewDescription = std::make_unique<RenderGraphDescription>(*description);
    auto *camera = cameraId ? dynamic_cast<Camera *>(Component::FindByComponentId(cameraId)) : nullptr;
    if (!camera || !camera->GetGameObject() || !m_sceneViewVisible)
        return {};
    if (camera->GetRenderToTexture())
        return GetRenderTexturePreview(camera->GetOutputTexture());
    return m_sceneCameraPreviewRenderedId == cameraId ? GetRenderTexturePreview(m_sceneCameraPreviewTarget)
                                                      : std::array<uint64_t, 3>{};
}

void InxRenderer::PrepareSceneCameraPreview()
{
    if (!m_sceneCameraPreviewDescription)
        return;
    const auto now = std::chrono::steady_clock::now();
    if (now < m_sceneCameraPreviewDue)
        return;
    auto *camera = dynamic_cast<Camera *>(Component::FindByComponentId(m_sceneCameraPreviewId));
    if (!camera || !camera->GetGameObject())
        return;
    if (camera->GetRenderToTexture())
        return;

    const auto &pacing = m_view->GetLastPacingSample();
    if (pacing.frameBudgetMs > 0.0 && pacing.elapsedBeforeSleepMs + 1.0 >= pacing.frameBudgetMs)
        return; // A thumbnail may stay stale while the foreground uses its budget.
    m_sceneCameraPreviewDue = now + std::chrono::milliseconds(83);

    const float aspect = std::max(0.01f, camera->GetAspectRatio());
    rhi::RenderTextureDesc target;
    target.width = std::max(1u, static_cast<uint32_t>(std::round(std::min(240.0f, 135.0f * aspect))));
    target.height = std::max(1u, static_cast<uint32_t>(std::round(std::min(135.0f, 240.0f / aspect))));
    target.depthFormat = rhi::PixelFormat::D32SFloat;
    try {
        if (!m_sceneCameraPreviewGraph) {
            auto texture = CreateRenderTexture(target);
            auto graph = std::make_unique<SceneRenderGraph>();
            if (!graph->Initialize(m_vkCore.get(), nullptr, rhi::RenderViewKind::Preview, texture))
                return;
            m_sceneCameraPreviewTarget = std::move(texture);
            m_sceneCameraPreviewGraph = std::move(graph);
        } else if (m_sceneCameraPreviewTarget->Reconfigure(target)) {
            m_sceneCameraPreviewGraph->SetOutputTexture(m_sceneCameraPreviewTarget);
        }
    } catch (const std::exception &error) {
        INXLOG_WARN("Scene camera preview could not allocate its target: ", error.what());
        m_sceneCameraPreviewDue = now + std::chrono::seconds(1);
        return;
    }
    auto *graph = m_sceneCameraPreviewGraph.get();
    EditorGizmosContext content;
    content.activeScene = camera->GetGameObject()->GetScene();
    ScriptableRenderContext context(m_vkCore.get(), graph, m_viewRequestSerial, content);
    const auto &description = *m_sceneCameraPreviewDescription;
    context.RenderWithGraph(camera, description);
}

void InxRenderer::SupersedePendingImGuiTextureUploads(const std::string &name)
{
    if (m_gui)
        m_gui->SupersedePendingImGuiTextureUploads(name);
}

void InxRenderer::RemoveImGuiTexture(const std::string &name)
{
    if (m_gui) {
        m_gui->RemoveImGuiTexture(name);
    } else {
        INXLOG_ERROR("InxGUI is not initialized.");
    }
}

bool InxRenderer::HasImGuiTexture(const std::string &name) const
{
    if (m_gui) {
        return m_gui->HasImGuiTexture(name);
    }
    return false;
}

uint64_t InxRenderer::GetImGuiTextureId(const std::string &name) const
{
    if (m_gui) {
        return m_gui->GetImGuiTextureId(name);
    }
    return 0;
}

ImGuiTexturePublication InxRenderer::GetImGuiTexturePublication(const std::string &name) const
{
    return m_gui ? m_gui->GetImGuiTexturePublication(name) : ImGuiTexturePublication{};
}

uint64_t InxRenderer::GetImGuiTextureVersion(const std::string &name) const
{
    return m_gui ? m_gui->GetImGuiTextureVersion(name) : 0;
}

uint64_t InxRenderer::GetFailedImGuiTextureVersion(const std::string &name) const
{
    return m_gui ? m_gui->GetFailedImGuiTextureVersion(name) : 0;
}

void InxRenderer::SetImGuiTextureBudgetBytes(uint64_t bytes)
{
    if (!m_gui)
        throw std::logic_error("ImGui texture budgeting requires an initialized GUI");
    m_gui->SetImGuiTextureBudgetBytes(bytes);
}

size_t InxRenderer::TrimImGuiTextureBudget()
{
    if (!m_gui)
        throw std::logic_error("ImGui texture budgeting requires an initialized GUI");
    return m_gui->TrimImGuiTextureBudget();
}

uint64_t InxRenderer::GetImGuiTextureBudgetBytes() const
{
    return m_gui ? m_gui->GetImGuiTextureBudgetBytes() : 0;
}

uint64_t InxRenderer::GetImGuiTextureResidentBytes() const
{
    return m_gui ? m_gui->GetImGuiTextureResidentBytes() : 0;
}

size_t InxRenderer::GetImGuiTextureEntryCount() const
{
    return m_gui ? m_gui->GetImGuiTextureEntryCount() : 0;
}

size_t InxRenderer::GetPendingImGuiTextureUploadCount() const
{
    return m_gui ? m_gui->GetPendingImGuiTextureUploadCount() : 0;
}

uint64_t InxRenderer::GetPendingImGuiTextureUploadBytes() const
{
    return m_gui ? m_gui->GetPendingImGuiTextureUploadBytes() : 0;
}

uint64_t InxRenderer::GetSubmittedImGuiTextureUploadCount() const
{
    return m_gui ? m_gui->GetSubmittedImGuiTextureUploadCount() : 0;
}

uint64_t InxRenderer::GetCompletedImGuiTextureUploadCount() const
{
    return m_gui ? m_gui->GetCompletedImGuiTextureUploadCount() : 0;
}

uint64_t InxRenderer::GetAsyncImGuiTextureUploadCount() const
{
    return m_gui ? m_gui->GetAsyncImGuiTextureUploadCount() : 0;
}

uint64_t InxRenderer::GetImGuiTextureEvictionCount() const
{
    return m_gui ? m_gui->GetImGuiTextureEvictionCount() : 0;
}

ResourcePreviewManager *InxRenderer::GetResourcePreviewManager()
{
    if (m_gui) {
        return &m_gui->GetResourcePreviewManager();
    }
    return nullptr;
}

void InxRenderer::SetLogLevel(LogLevel level)
{
    INXLOG_SET_LEVEL(level);
}

void InxRenderer::InitializeDefaultScene()
{
    // Initialize editor gizmos
    m_editorGizmos = std::make_unique<EditorGizmos>();

    // Initialize editor tools (translate/rotate/scale handles)
    m_editorTools = std::make_unique<EditorTools>();

    // Initialize component gizmos buffer used by the scripting layer
    m_componentGizmos = std::make_unique<GizmosDrawCallBuffer>();
    m_particleGpuDrawRegistry = std::make_unique<particle::ParticleGpuDrawRegistry>();
    if (m_scenePickingService)
        m_scenePickingService->SetParticleGpuDrawRegistry(m_particleGpuDrawRegistry.get());

    // Pass gizmos reference to VkCore for rendering
    if (m_vkCore) {
        m_vkCore->SetEditorGizmos(m_editorGizmos.get());
    }

    // Create a default scene
    Scene *scene = SceneManager::Instance().CreateScene("Default Scene");

    // Create a directional light (sun)
    GameObject *lightObj = scene->CreateGameObject("Directional Light");
    lightObj->GetTransform()->SetEulerAngles(glm::vec3(-50.0f, -30.0f, 0.0f)); // Angled downward (degrees)
    Light *light = lightObj->AddComponent<Light>();
    light->SetLightType(LightType::Directional);
    light->SetColor(glm::vec3(1.0f, 0.95f, 0.9f)); // Warm white sunlight
    light->SetIntensity(1.0f);
    light->SetShadows(LightShadows::Hard); // Enable hard shadows by default

    // Create a cube game object with MeshRenderer
    GameObject *cube = scene->CreateGameObject("Cube");
    cube->GetTransform()->SetPosition(glm::vec3(0.0f, 0.5f, 0.0f)); // Raise above ground plane

    // Add MeshRenderer with cube mesh data
    MeshRenderer *renderer = cube->AddComponent<MeshRenderer>();
    renderer->SetMesh(
        std::vector<Vertex>(PrimitiveMeshes::GetCubeVertices().begin(), PrimitiveMeshes::GetCubeVertices().end()),
        std::vector<uint32_t>(PrimitiveMeshes::GetCubeIndices().begin(), PrimitiveMeshes::GetCubeIndices().end()));

    // Create a ground plane to receive shadows
    GameObject *ground = scene->CreateGameObject("Ground");
    ground->GetTransform()->SetPosition(glm::vec3(0.0f, -0.5f, 0.0f));
    ground->GetTransform()->SetScale(glm::vec3(20.0f, 0.01f, 20.0f)); // Thin, wide plane
    MeshRenderer *groundRenderer = ground->AddComponent<MeshRenderer>();
    groundRenderer->SetMesh(
        std::vector<Vertex>(PrimitiveMeshes::GetCubeVertices().begin(), PrimitiveMeshes::GetCubeVertices().end()),
        std::vector<uint32_t>(PrimitiveMeshes::GetCubeIndices().begin(), PrimitiveMeshes::GetCubeIndices().end()));

    // Setup editor camera
    EditorCameraController &editorCam = SceneManager::Instance().GetEditorCameraController();
    editorCam.Reset();

    // Position camera to see the cube - use FocusOn to keep yaw/pitch in sync
    editorCam.FocusOn(glm::vec3(0.0f, 0.0f, 0.0f), 5.0f);

    // Initialize last frame time
    m_lastFrameTime = std::chrono::steady_clock::now();

    INXLOG_INFO("Default scene initialized with cube, directional light, and gizmos");
}

void InxRenderer::InitializeRuntimeScene()
{
    // The Player replaces this scene before its first rendered frame. Keep
    // only the state required by scene publication and GPU particle binding;
    // editor gizmos, tools, preview geometry, and their materials are not
    // runtime dependencies.
    m_particleGpuDrawRegistry = std::make_unique<particle::ParticleGpuDrawRegistry>();
    if (m_scenePickingService)
        m_scenePickingService->SetParticleGpuDrawRegistry(m_particleGpuDrawRegistry.get());

    SceneManager::Instance().CreateScene("Runtime Bootstrap");
    m_lastFrameTime = std::chrono::steady_clock::now();
}

void InxRenderer::UpdateSceneLighting()
{
    if (!m_vkCore)
        return;

    Scene *activeScene = SceneManager::Instance().GetActiveScene();
    if (!activeScene)
        return;

    // Get camera position for light sorting
    glm::vec3 cameraPos(m_cameraPos[0], m_cameraPos[1], m_cameraPos[2]);

    // Collect lights from scene into the light collector
    SceneLightCollector &collector = m_vkCore->GetLightCollector();
#if INFERNUX_FRAME_PROFILE
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();
#endif
    collector.CollectLights(activeScene, cameraPos);
#if INFERNUX_FRAME_PROFILE
    m_frameDetailTiming.lightingCollectMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif

    Camera *editorCam = SceneRenderBridge::Instance().GetEditorCamera();
#if INFERNUX_FRAME_PROFILE
    t0 = Clock::now();
#endif
    m_vkCore->UpdateLightingState();
    const ShaderLightingUBO environmentLighting = collector.GetShaderLightingUBO();
    if (m_sceneRenderGraph && editorCam && m_sceneRenderGraph->IsViewRequested(m_viewRequestSerial))
        m_sceneRenderGraph->StageCameraLighting(activeScene, editorCam, cameraPos, environmentLighting);
    if (m_sceneCameraPreviewGraph && m_sceneCameraPreviewGraph->IsViewRequested(m_viewRequestSerial)) {
        auto *camera = dynamic_cast<Camera *>(Component::FindByComponentId(m_sceneCameraPreviewId));
        m_sceneCameraPreviewGraph->StageCameraLighting(camera->GetGameObject()->GetScene(), camera,
                                                       glm::vec3(camera->GetCameraToWorldMatrix()[3]),
                                                       environmentLighting);
    }
#if INFERNUX_FRAME_PROFILE
    m_frameDetailTiming.lightingShadowEditorMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif

#if INFERNUX_FRAME_PROFILE
    m_frameDetailTiming.lightingShadowGameMs = 0.0;
#endif
    if (HasActiveGameViews()) {
#if INFERNUX_FRAME_PROFILE
        t0 = Clock::now();
#endif
        for (Camera *gameCam : FindGameCamerasCached()) {
            if (!ShouldRenderGameCamera(gameCam))
                continue;
            const glm::vec3 gameCameraPosition(gameCam->GetCameraToWorldMatrix()[3]);
            if (SceneRenderGraph *graph = EnsureGameRenderGraph(gameCam);
                graph && graph->IsViewRequested(m_viewRequestSerial))
                graph->StageCameraLighting(activeScene, gameCam, gameCameraPosition, environmentLighting);
        }
#if INFERNUX_FRAME_PROFILE
        m_frameDetailTiming.lightingShadowGameMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif
    }

#if INFERNUX_FRAME_PROFILE
    t0 = Clock::now();
#endif
#if INFERNUX_FRAME_PROFILE
    m_frameDetailTiming.lightingUploadMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
#endif
}

uint64_t InxRenderer::GetSceneTextureId() const
{
    if (m_sceneRenderTarget && m_sceneRenderTarget->IsReady()) {
        return m_sceneRenderTarget->GetImGuiTextureId();
    }
    return 0;
}

static std::shared_ptr<vk::ImageReadbackTicket>
BeginTargetReadback(vk::VkResourceManager &resources, const SceneRenderTarget &target, vk::ImageReadbackOutput output)
{
    return resources.BeginImageReadback(target.GetColorImage(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                        VK_IMAGE_ASPECT_COLOR_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                        VK_ACCESS_SHADER_READ_BIT, target.GetWidth(), target.GetHeight(),
                                        target.GetColorFormat(), {}, {}, output);
}

std::shared_ptr<vk::ImageReadbackTicket> InxRenderer::RequestRenderTargetReadback(bool gameView)
{
    SceneRenderTarget *target = gameView ? m_gameRenderTarget.get() : m_sceneRenderTarget.get();
    if (!m_vkCore || !target || !target->IsReady())
        throw std::logic_error(gameView ? "Game render target is not initialized"
                                        : "Scene render target is not initialized");

    return BeginTargetReadback(m_vkCore->GetResourceManager(), *target, vk::ImageReadbackOutput::Pixels);
}

std::shared_ptr<vk::ImageReadbackTicket> InxRenderer::RequestImGuiTextureReadback(const std::string &name)
{
    if (!m_gui)
        throw std::logic_error("GUI texture readback requires graphical renderer initialization");
    return m_gui->RequestTextureReadback(name);
}

RendererFrameTelemetrySnapshot InxRenderer::GetFrameTelemetrySnapshot()
{
    RendererFrameTelemetrySnapshot snapshot;
    const auto countShadowCasters = [](const std::vector<DrawCall> &drawCalls) {
        return static_cast<size_t>(std::count_if(drawCalls.begin(), drawCalls.end(),
                                                 [](const DrawCall &drawCall) { return drawCall.castsShadows; }));
    };
    snapshot.frame = m_frameCount;
    snapshot.sceneTemporalDiscontinuityRevision = m_observedSceneTemporalDiscontinuityRevision;
    snapshot.temporalHistoryInvalidationCount = m_temporalHistoryInvalidationCount;
    snapshot.sceneViewVisible = m_sceneViewVisible;
    snapshot.gameCameraEnabled = m_gameCameraEnabled;
    const auto &gameCameras = FindGameCamerasCached();
    snapshot.gameCameraAvailable = !gameCameras.empty();
    snapshot.gameCameraCount = gameCameras.size();
    snapshot.gameCameraIds.reserve(gameCameras.size());
    snapshot.gameRenderViewIds.reserve(gameCameras.size());
    for (Camera *camera : gameCameras) {
        snapshot.gameCameraIds.push_back(camera->GetComponentID());
        if (SceneRenderGraph *graph = EnsureGameRenderGraph(camera))
            snapshot.gameRenderViewIds.push_back(graph->GetRenderViewContext().id);
    }
    snapshot.sceneTargetReady = m_sceneRenderTarget && m_sceneRenderTarget->IsReady();
    snapshot.gameTargetReady = m_gameRenderTarget && m_gameRenderTarget->IsReady();
    if (m_sceneRenderTarget) {
        snapshot.sceneTargetWidth = m_sceneRenderTarget->GetWidth();
        snapshot.sceneTargetHeight = m_sceneRenderTarget->GetHeight();
    }
    if (m_gameRenderTarget) {
        snapshot.gameTargetWidth = m_gameRenderTarget->GetWidth();
        snapshot.gameTargetHeight = m_gameRenderTarget->GetHeight();
    }
    if (m_sceneRenderGraph) {
        snapshot.sceneRenderViewId = m_sceneRenderGraph->GetRenderViewContext().id;
        snapshot.sceneTemporalHistoryCount = m_sceneRenderGraph->GetTemporalHistoryCount();
        snapshot.sceneValidTemporalHistoryCount = m_sceneRenderGraph->GetValidTemporalHistoryCount();
        snapshot.sceneTemporalSampleIndex = m_sceneRenderGraph->GetTemporalSampleIndex();
        snapshot.sceneRenderGraphName = m_sceneRenderGraph->GetGraphName();
        snapshot.sceneRenderGraphExecutionCount = m_sceneRenderGraph->GetExecutionCount();
        snapshot.sceneRenderGraphCurrentExecuted = m_sceneRenderGraph->HasExecutedCurrentGraph();
        snapshot.sceneRenderGraphPassNames = m_sceneRenderGraph->GetExecutedPassNames();
        snapshot.sceneRenderGraphDebug = m_sceneRenderGraph->GetDebugString();
        snapshot.sceneShadowViewCount = m_sceneRenderGraph->GetCameraShadowViewCount();
        snapshot.sceneShadowAssignmentCount = m_sceneRenderGraph->GetCameraShadowAssignmentCount();
        snapshot.sceneShadowResourceIdentity = m_sceneRenderGraph->GetShadowResourceIdentity();
        if (m_sceneRenderGraph->HasCachedDrawCalls())
            snapshot.sceneDrawCallCount = m_sceneRenderGraph->GetCachedDrawCalls().size();
        if (m_sceneRenderGraph->HasCachedShadowDrawCalls())
            snapshot.sceneShadowDrawCallCount = countShadowCasters(m_sceneRenderGraph->GetCachedShadowDrawCalls());
    }
    if (m_gameRenderGraph) {
        snapshot.gameRenderViewId = m_gameRenderGraph->GetRenderViewContext().id;
        snapshot.gameTemporalHistoryCount = m_gameRenderGraph->GetTemporalHistoryCount();
        snapshot.gameValidTemporalHistoryCount = m_gameRenderGraph->GetValidTemporalHistoryCount();
        snapshot.gameTemporalSampleIndex = m_gameRenderGraph->GetTemporalSampleIndex();
        snapshot.gameRenderGraphName = m_gameRenderGraph->GetGraphName();
        snapshot.gameRenderGraphExecutionCount = m_gameRenderGraph->GetExecutionCount();
        snapshot.gameRenderGraphCurrentExecuted = m_gameRenderGraph->HasExecutedCurrentGraph();
        snapshot.gameRenderGraphPassNames = m_gameRenderGraph->GetExecutedPassNames();
        snapshot.gameRenderGraphDebug = m_gameRenderGraph->GetDebugString();
        snapshot.gameShadowViewCount = m_gameRenderGraph->GetCameraShadowViewCount();
        snapshot.gameShadowAssignmentCount = m_gameRenderGraph->GetCameraShadowAssignmentCount();
        snapshot.gameShadowResourceIdentity = m_gameRenderGraph->GetShadowResourceIdentity();
        if (m_gameRenderGraph->HasCachedDrawCalls())
            snapshot.gameDrawCallCount = m_gameRenderGraph->GetCachedDrawCalls().size();
        if (m_gameRenderGraph->HasCachedShadowDrawCalls())
            snapshot.gameShadowDrawCallCount = countShadowCasters(m_gameRenderGraph->GetCachedShadowDrawCalls());
    }
    if (gameCameras.size() > 1) {
        snapshot.gameTemporalHistoryCount = 0;
        snapshot.gameValidTemporalHistoryCount = 0;
        snapshot.gameDrawCallCount = 0;
        snapshot.gameShadowDrawCallCount = 0;
        snapshot.gameShadowViewCount = 0;
        snapshot.gameShadowAssignmentCount = 0;
        for (Camera *camera : gameCameras) {
            SceneRenderGraph *graph = EnsureGameRenderGraph(camera);
            if (!graph)
                continue;
            snapshot.gameTemporalHistoryCount += graph->GetTemporalHistoryCount();
            snapshot.gameValidTemporalHistoryCount += graph->GetValidTemporalHistoryCount();
            snapshot.gameShadowViewCount += graph->GetCameraShadowViewCount();
            snapshot.gameShadowAssignmentCount += graph->GetCameraShadowAssignmentCount();
            if (graph->HasCachedDrawCalls())
                snapshot.gameDrawCallCount += graph->GetCachedDrawCalls().size();
            if (graph->HasCachedShadowDrawCalls())
                snapshot.gameShadowDrawCallCount += countShadowCasters(graph->GetCachedShadowDrawCalls());
        }
    }
    if (m_vkCore) {
        snapshot.lightCount = m_vkCore->GetLightCollector().GetTotalLightCount();
        const FrameSubmissionTelemetry &submission = m_vkCore->GetFrameSubmissionTelemetry();
        snapshot.submissionGeneration = submission.generation;
        snapshot.submissionComputeQueueIndependent = submission.computeQueueIndependent;
        snapshot.submissionTransferQueueIndependent = submission.transferQueueIndependent;
        snapshot.submissionAsyncComputeActive = submission.asyncComputeActive;
        snapshot.submissionParallelComputeGraphics = submission.parallelComputeGraphics;
        snapshot.submissionBatchCount = submission.batchCount;
        snapshot.submissionGraphicsBatchCount = submission.graphicsBatchCount;
        snapshot.submissionComputeBatchCount = submission.computeBatchCount;
        snapshot.submissionTransferBatchCount = submission.transferBatchCount;
        snapshot.submissionCrossQueueDependencyCount = submission.crossQueueDependencyCount;
        snapshot.submissionUnorderedComputeGraphicsPairCount = submission.unorderedComputeGraphicsPairCount;
        snapshot.submissionResidentComputeWriteSerial = submission.residentComputeWriteSerial;
        snapshot.submissionLatestBackgroundComputeSerial = submission.latestBackgroundComputeSerial;
        snapshot.submissionResidentComputeWaitPending = submission.residentComputeWaitPending;
        if (const auto *lights = m_vkCore->GetCanonicalLightGpuFrame()) {
            snapshot.canonicalLightGpuBufferReady = lights->buffer.IsValid();
            snapshot.canonicalLightGpuBytes = lights->dataBytes;
            snapshot.canonicalLightGeneration = lights->generation;
            snapshot.canonicalDirectionalLightCount = lights->directionalCount;
            snapshot.canonicalLocalLightCount = lights->localCount;
        }
    }
    if (m_particleGpuSystemManager) {
        const auto gpuParticles = m_particleGpuSystemManager->TelemetrySnapshot();
        snapshot.gpuParticleSystemCount = gpuParticles.systemCount;
        snapshot.gpuParticleOutputCount = gpuParticles.outputCount;
        snapshot.gpuParticleCapacity = gpuParticles.totalCapacity;
        snapshot.gpuParticleLastScheduledFrame = gpuParticles.lastScheduledFrame;
        snapshot.gpuParticleScheduledSystemCount = gpuParticles.scheduledSystemCount;
        snapshot.gpuParticleSimulatingSystemCount = gpuParticles.simulatingSystemCount;
        snapshot.gpuParticleRenderingSystemCount = gpuParticles.renderingSystemCount;
        snapshot.gpuParticleRequestedSpawnCount = gpuParticles.requestedSpawnCount;
        snapshot.gpuParticleContinuationSystemCount = gpuParticles.continuationSystemCount;
        snapshot.gpuParticleContinuationCapacity = gpuParticles.totalContinuationCapacity;
        snapshot.gpuParticleContinuationProgramGeneration = gpuParticles.maximumContinuationProgramGeneration;
        snapshot.gpuParticleContinuationPrepareRecordCalls = gpuParticles.continuationPrepareRecordCalls;
        snapshot.gpuParticleContinuationClassifyRecordCalls = gpuParticles.continuationClassifyRecordCalls;
        snapshot.gpuParticleContinuationDispatchRecordCalls = gpuParticles.continuationDispatchRecordCalls;
        snapshot.gpuParticleContinuationResetPendingCount = gpuParticles.continuationResetPendingCount;
        snapshot.gpuParticleContactRuntimeSystemCount = gpuParticles.contactRuntimeSystemCount;
        snapshot.gpuParticleContactRecordCapacity = gpuParticles.totalContactRecordCapacity;
        snapshot.gpuParticleContactWorkItemCapacity = gpuParticles.totalContactWorkItemCapacity;
        snapshot.gpuParticleContactResidentBytes = gpuParticles.totalContactResidentBytes;
        snapshot.gpuParticleContactPrepareRecordCalls = gpuParticles.contactPrepareRecordCalls;
        snapshot.gpuParticleContactSolveRecordCalls = gpuParticles.contactSolveRecordCalls;
        snapshot.gpuParticleCollisionSceneRevision = gpuParticles.collisionSceneRevision;
        snapshot.gpuParticleCollisionSceneColliderCount = gpuParticles.collisionSceneColliderCount;
        snapshot.gpuParticleCollisionSceneTopologyRevision = gpuParticles.collisionSceneTopologyRevision;
        snapshot.gpuParticleCollisionSceneMeshVertexCount = gpuParticles.collisionSceneMeshVertexCount;
        snapshot.gpuParticleCollisionSceneMeshIndexCount = gpuParticles.collisionSceneMeshIndexCount;
        snapshot.gpuParticleCollisionSceneMeshBvhNodeCount = gpuParticles.collisionSceneMeshBvhNodeCount;
    }
    snapshot.gameRenderMs = m_lastGameRenderMs;
    snapshot.gameOnlyFrameMs = m_gameOnlyFrameMs;
    snapshot.sceneUpdateMs = m_sceneUpdateMs;
    snapshot.guiBuildMs = m_guiBuildMs;
    snapshot.prepareFrameMs = m_prepareFrameMs;
    if (m_gui) {
        const auto scheduler = m_gui->GetFrameSchedulerSnapshot();
        snapshot.guiFrame = m_gui->GetFrameCounter();
        snapshot.guiFrameRequested = scheduler.requested;
        snapshot.guiFrameIntervalMs = scheduler.intervalMs;
        snapshot.guiFrameUntilDueMs = scheduler.untilDueMs;
        snapshot.guiFrameConsumeCount = scheduler.consumeCount;
        snapshot.guiFrameApprovedCount = scheduler.approvedCount;
        snapshot.guiFrameForcedCount = scheduler.forcedCount;
        snapshot.guiFrameRequestCount = scheduler.requestCount;
        snapshot.guiPanelTimesMs = m_gui->GetLastPanelTimesMs();
        snapshot.guiPanelSubTimesMs = m_gui->GetLastPanelSubTimesMs();
    }
    return snapshot;
}

uint64_t InxRenderer::BeginFramePerformanceWindow(size_t sampleCount)
{
    if (sampleCount == 0 || sampleCount > 65536)
        throw std::invalid_argument("Frame performance sample count must be between 1 and 65536");
    m_framePerformanceHistory.reserve(sampleCount);
    m_framePerformanceHistory.clear();
    m_framePerformanceSampleLimit = sampleCount;
    m_framePerformanceWindowActive = true;
    return m_frameCount + 1;
}

void InxRenderer::RecordFramePerformanceSample(RendererFramePerformanceSample sample)
{
    sample.frame = m_frameCount;
    sample.gameOnlyMs = m_gameOnlyFrameMs;
    sample.renderMs = m_lastGameRenderMs;
    sample.sceneMs = m_sceneUpdateMs;
    sample.guiMs = m_guiBuildMs;
    sample.prepareMs = m_prepareFrameMs;
    m_framePerformanceHistory.push_back(sample);

    if (m_framePerformanceHistory.size() == m_framePerformanceSampleLimit)
        m_framePerformanceWindowActive = false;
}

RendererFramePerformanceSnapshot InxRenderer::GetFramePerformanceWindow() const
{
    RendererFramePerformanceSnapshot snapshot;
    const size_t count = m_framePerformanceHistory.size();
    snapshot.sampleCount = count;
    snapshot.targetSampleCount = m_framePerformanceSampleLimit;
    snapshot.active = m_framePerformanceWindowActive;
    if (count == 0)
        return snapshot;

    snapshot.firstFrame = m_framePerformanceHistory.front().frame;
    snapshot.lastFrame = m_framePerformanceHistory.back().frame;
    snapshot.slowestFrames.resize((std::min)(count, size_t{16}));
    std::partial_sort_copy(
        m_framePerformanceHistory.begin(), m_framePerformanceHistory.end(), snapshot.slowestFrames.begin(),
        snapshot.slowestFrames.end(), [](const auto &left, const auto &right) {
            return left.frameMs > right.frameMs || (left.frameMs == right.frameMs && left.frame < right.frame);
        });

    const auto summarize = [](std::vector<double> values) {
        UIPerformanceMetricStats stats;
        stats.sampleCount = values.size();
        if (values.empty())
            return stats;

        stats.meanMs = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
        std::sort(values.begin(), values.end());
        const auto percentile = [&values](double fraction) {
            if (values.size() == 1)
                return values.front();
            const double position = fraction * static_cast<double>(values.size() - 1);
            const size_t lower = static_cast<size_t>(std::floor(position));
            const size_t upper = static_cast<size_t>(std::ceil(position));
            if (lower == upper)
                return values[lower];
            return values[lower] + (values[upper] - values[lower]) * (position - static_cast<double>(lower));
        };
        stats.medianMs = percentile(0.50);
        stats.p95Ms = percentile(0.95);
        stats.p99Ms = percentile(0.99);
        stats.maxMs = values.back();
        return stats;
    };

    std::vector<double> frame;
    std::vector<double> gameOnly;
    std::vector<double> render;
    std::vector<double> scene;
    std::vector<double> gui;
    std::vector<double> prepare;
    frame.reserve(count);
    gameOnly.reserve(count);
    render.reserve(count);
    scene.reserve(count);
    gui.reserve(count);
    prepare.reserve(count);
    for (const auto &sample : m_framePerformanceHistory) {
        frame.push_back(sample.frameMs);
        gameOnly.push_back(sample.gameOnlyMs);
        render.push_back(sample.renderMs);
        scene.push_back(sample.sceneMs);
        gui.push_back(sample.guiMs);
        prepare.push_back(sample.prepareMs);
    }
    snapshot.frame = summarize(std::move(frame));
    snapshot.gameOnly = summarize(std::move(gameOnly));
    snapshot.render = summarize(std::move(render));
    snapshot.scene = summarize(std::move(scene));
    snapshot.gui = summarize(std::move(gui));
    snapshot.prepare = summarize(std::move(prepare));
    return snapshot;
}

void InxRenderer::RecordUIPerformanceFrame()
{
    if (!m_gui)
        return;

    const size_t slot = m_uiPerformanceWriteIndex;
    // Store a non-zero stamp so default-initialized history slots never look valid.
    const uint64_t frame = m_frameCount + 1;
    m_uiBuildHistory.values[slot] = m_guiBuildMs;
    m_uiBuildHistory.frames[slot] = frame;

    for (const auto &[name, value] : m_gui->GetLastPanelTimesMs()) {
        auto &history = m_uiPanelHistory[name];
        history.values[slot] = value;
        history.frames[slot] = frame;
    }
    for (const auto &[panelName, stages] : m_gui->GetLastPanelSubTimesMs()) {
        auto &panelHistory = m_uiPanelSubHistory[panelName];
        for (const auto &[stageName, value] : stages) {
            auto &history = panelHistory[stageName];
            history.values[slot] = value;
            history.frames[slot] = frame;
        }
    }

    m_uiPerformanceWriteIndex = (slot + 1) % UI_PERFORMANCE_HISTORY_SIZE;
    m_uiPerformanceSampleCount = (std::min)(m_uiPerformanceSampleCount + 1, UI_PERFORMANCE_HISTORY_SIZE);
}

RendererUIPerformanceSnapshot InxRenderer::GetUIPerformanceSnapshot(size_t maxSamples) const
{
    RendererUIPerformanceSnapshot snapshot;
    const size_t count = (std::min)({maxSamples, m_uiPerformanceSampleCount, UI_PERFORMANCE_HISTORY_SIZE});
    if (count == 0)
        return snapshot;

    const size_t firstSlot =
        (m_uiPerformanceWriteIndex + UI_PERFORMANCE_HISTORY_SIZE - count) % UI_PERFORMANCE_HISTORY_SIZE;
    const size_t lastSlot = (m_uiPerformanceWriteIndex + UI_PERFORMANCE_HISTORY_SIZE - 1) % UI_PERFORMANCE_HISTORY_SIZE;
    snapshot.firstFrame = m_uiBuildHistory.frames[firstSlot] - 1;
    snapshot.lastFrame = m_uiBuildHistory.frames[lastSlot] - 1;
    snapshot.sampleCount = count;

    const auto summarize = [&](const UIMetricHistory &history) {
        std::vector<double> values;
        values.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            const size_t slot = (firstSlot + index) % UI_PERFORMANCE_HISTORY_SIZE;
            if (history.frames[slot] == m_uiBuildHistory.frames[slot])
                values.push_back(history.values[slot]);
        }

        UIPerformanceMetricStats stats;
        stats.sampleCount = values.size();
        if (values.empty())
            return stats;
        stats.meanMs = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
        std::sort(values.begin(), values.end());
        const size_t medianIndex = (values.size() - 1) / 2;
        stats.medianMs =
            values.size() % 2 == 0 ? (values[medianIndex] + values[medianIndex + 1]) * 0.5 : values[medianIndex];
        const size_t p95Index = static_cast<size_t>(std::ceil(static_cast<double>(values.size()) * 0.95)) - 1;
        stats.p95Ms = values[(std::min)(p95Index, values.size() - 1)];
        const size_t p99Index = static_cast<size_t>(std::ceil(static_cast<double>(values.size()) * 0.99)) - 1;
        stats.p99Ms = values[(std::min)(p99Index, values.size() - 1)];
        stats.maxMs = values.back();
        return stats;
    };

    snapshot.guiBuild = summarize(m_uiBuildHistory);
    for (const auto &[name, history] : m_uiPanelHistory) {
        auto stats = summarize(history);
        if (stats.sampleCount > 0)
            snapshot.panelTimes.emplace(name, stats);
    }
    for (const auto &[panelName, stages] : m_uiPanelSubHistory) {
        auto &outputStages = snapshot.panelSubTimes[panelName];
        for (const auto &[stageName, history] : stages) {
            auto stats = summarize(history);
            if (stats.sampleCount > 0)
                outputStages.emplace(stageName, stats);
        }
        if (outputStages.empty())
            snapshot.panelSubTimes.erase(panelName);
    }
    return snapshot;
}

uint64_t InxRenderer::RequestCapture(CaptureSource source, const std::string &outputPath, uint64_t cameraComponentId)
{
    if (!m_captureService)
        throw std::logic_error("Capture service is not initialized");
    if (source == CaptureSource::Camera) {
        const auto &cameras = FindGameCamerasCached();
        const auto found = std::find_if(cameras.begin(), cameras.end(), [cameraComponentId](const Camera *camera) {
            return camera->GetComponentID() == cameraComponentId;
        });
        auto *graph = found == cameras.end() ? nullptr : EnsureGameRenderGraph(*found);
        if (!graph || !graph->HasOutputTexture())
            throw std::invalid_argument("Camera capture requires an active offscreen Camera");
        auto captureView = graph->GetRenderViewContext();
        if (const auto *output = graph->GetOutputTarget()) {
            captureView.width = output->GetWidth();
            captureView.height = output->GetHeight();
        }
        const auto generation = captureView.revision;
        captureView.source = captureView.id;
        captureView.id = rhi::AllocateRenderViewId();
        captureView.kind = rhi::RenderViewKind::Capture;
        captureView.output = rhi::RenderOutputKind::Readback;
        captureView.color = captureView.depth = captureView.motion = {};
        const auto id = m_captureService->Request(source, captureView, generation, m_frameCount, outputPath);
        m_pendingCaptures.push_back({id, source, generation, cameraComponentId});
        RequestFullSpeedFrame();
        return id;
    }
    if (cameraComponentId != 0)
        throw std::invalid_argument("camera_component_id is only valid for Camera capture");
    if (source == CaptureSource::Editor) {
        rhi::RenderViewContext captureView = m_vkCore->GetPresentationViewContext();
        if (!captureView.IsValid())
            throw std::logic_error("Editor presentation view is not initialized");
        const uint64_t generation = captureView.revision;
        captureView.source = captureView.id;
        captureView.id = rhi::AllocateRenderViewId();
        captureView.kind = rhi::RenderViewKind::Capture;
        captureView.output = rhi::RenderOutputKind::Readback;
        captureView.color = {};
        captureView.depth = {};
        captureView.motion = {};
        const uint64_t captureId = m_captureService->Request(source, captureView, generation, m_frameCount, outputPath);
        m_pendingCaptures.push_back({captureId, source, generation});
        RequestFullSpeedFrame();
        return captureId;
    }
    const bool gameView = source == CaptureSource::Game;
    if (gameView && (!m_gameRenderTarget || !m_gameRenderTarget->IsReady())) {
        if (!FindGameCameraCached())
            throw std::logic_error("Game camera is not available");
        constexpr uint32_t kBackgroundCaptureWidth = 1280;
        constexpr uint32_t kBackgroundCaptureHeight = 720;
        ResizeGameRenderTarget(kBackgroundCaptureWidth, kBackgroundCaptureHeight);
    }
    const uint64_t generation = gameView ? m_gameRenderTargetGeneration : m_sceneRenderTargetGeneration;
    SceneRenderTarget *target = gameView ? m_gameRenderTarget.get() : m_sceneRenderTarget.get();
    if (!target || !target->IsReady())
        throw std::logic_error(gameView ? "Game render target is not initialized"
                                        : "Scene render target is not initialized");

    rhi::RenderViewContext captureView;
    if (gameView) {
        captureView.id = m_gameScreenViewId;
        captureView.device = m_vkCore->GetDeviceContext().GetDeviceId();
        captureView.colorFormat = rhi::FromVkFormat(target->GetColorFormat());
        captureView.revision = generation;
    } else if (m_sceneRenderGraph) {
        captureView = m_sceneRenderGraph->GetRenderViewContext();
    }
    captureView.width = target->GetWidth();
    captureView.height = target->GetHeight();
    if (!captureView.IsValid())
        throw std::logic_error("Capture source render view is not initialized");
    captureView.source = captureView.id;
    captureView.id = rhi::AllocateRenderViewId();
    captureView.kind = rhi::RenderViewKind::Capture;
    captureView.output = rhi::RenderOutputKind::Readback;
    // Capture metadata identifies the source but never owns its live handles.
    captureView.color = {};
    captureView.depth = {};
    captureView.motion = {};
    const uint64_t captureId = m_captureService->Request(source, captureView, generation, m_frameCount, outputPath);
    m_pendingCaptures.push_back(
        {captureId, source, generation, 0, gameView ? CurrentGameImageGeneration() : GameImageGeneration{}});
    RequestFullSpeedFrame();
    return captureId;
}

bool InxRenderer::HasPendingCapture(CaptureSource source, uint64_t cameraId) const
{
    return std::any_of(m_pendingCaptures.begin(), m_pendingCaptures.end(),
                       [source, cameraId](const PendingCapture &capture) {
                           return capture.source == source && capture.cameraComponentId == cameraId;
                       });
}

void InxRenderer::SubmitPendingCaptureReadbacks(bool frameSubmitted)
{
    // DrawFrame may return before submission while acquiring/resizing the surface.
    // Keep requests pending until a real frame submission produces their source.
    if (!m_captureService || m_pendingCaptures.empty() || !frameSubmitted)
        return;

    const auto sourceWasRendered = [this](const SceneRenderGraph *graph) {
        return graph && graph->HasExecutedCurrentGraph() &&
               std::any_of(m_viewSchedule.order.begin(), m_viewSchedule.order.end(), [this, graph](uint32_t index) {
                   const auto &view = m_scheduledViews[index];
                   return view.graph == graph && view.viewId == graph->GetRenderViewContext().id;
               });
    };

    std::vector<PendingCapture> pending;
    pending.swap(m_pendingCaptures);
    const auto presentationReadback = m_vkCore->ConsumePresentationReadback();
    const std::string presentationError = m_vkCore->ConsumePresentationReadbackError();
    const auto attach = [this](uint64_t id, const SceneRenderTarget &target) {
        (void)m_captureService->AttachReadback(
            id, BeginTargetReadback(m_vkCore->GetResourceManager(), target, vk::ImageReadbackOutput::Mapped),
            m_frameCount);
    };
    for (const PendingCapture &capture : pending) {
        try {
            const auto snapshot = m_captureService->Query(capture.id);
            if (snapshot.status != CaptureStatus::PendingGpu)
                continue;
            if (capture.source == CaptureSource::Camera) {
                const auto found = m_gameRenderGraphs.find(capture.cameraComponentId);
                if (found == m_gameRenderGraphs.end() || !found->second->HasOutputTexture() ||
                    found->second->GetRenderViewContext().id != snapshot.view.source ||
                    found->second->GetRenderViewContext().revision != capture.sourceGeneration) {
                    m_captureService->Fail(capture.id, "Camera output changed before capture submission");
                    continue;
                }
                if (!sourceWasRendered(found->second.get())) {
                    m_captureService->Fail(capture.id, "Camera capture source was not rendered in the submitted frame");
                    continue;
                }
                attach(capture.id, *found->second->GetOutputTarget());
                continue;
            }
            if (capture.source == CaptureSource::Editor) {
                const uint64_t currentGeneration = m_vkCore->GetPresentationViewContext().revision;
                if (capture.sourceGeneration != currentGeneration) {
                    m_captureService->InvalidateSource(capture.source, currentGeneration);
                    continue;
                }
                if (!presentationError.empty()) {
                    m_captureService->Fail(capture.id, presentationError);
                    continue;
                }
                if (!presentationReadback) {
                    m_pendingCaptures.push_back(capture);
                    RequestFullSpeedFrame();
                    continue;
                }
                (void)m_captureService->AttachReadback(capture.id, presentationReadback, m_frameCount);
                continue;
            }
            if (capture.source == CaptureSource::Game) {
                if (capture.sourceGeneration != m_gameRenderTargetGeneration) {
                    m_captureService->InvalidateSource(capture.source, m_gameRenderTargetGeneration);
                } else if (!(capture.gameImage == CurrentGameImageGeneration())) {
                    m_captureService->Fail(capture.id, "Game capture world changed before submission");
                } else if (m_acceptedGameImage.completionEpoch == m_vkCore->GetLastSubmittedCompletionEpoch() &&
                           m_acceptedGameImage.generation == capture.gameImage) {
                    attach(capture.id, *m_gameRenderTarget);
                } else {
                    m_pendingCaptures.push_back(capture);
                }
                continue;
            }
            if (capture.sourceGeneration != m_sceneRenderTargetGeneration) {
                m_captureService->InvalidateSource(capture.source, m_sceneRenderTargetGeneration);
                continue;
            }
            const auto *graph = m_sceneRenderGraph.get();
            if (!graph || !m_sceneRenderTarget || !m_sceneRenderTarget->IsReady() ||
                graph->GetRenderViewContext().id != snapshot.view.source) {
                m_captureService->Fail(capture.id, "Capture source render view changed before submission");
                continue;
            }
            if (!sourceWasRendered(graph)) {
                m_captureService->Fail(capture.id, "Capture source was not rendered in the submitted frame");
                continue;
            }
            attach(capture.id, *m_sceneRenderTarget);
        } catch (const std::exception &exc) {
            const std::string error = std::string("Capture readback submission failed: ") + exc.what();
            INXLOG_ERROR(error);
            m_captureService->Fail(capture.id, error);
        }
    }
}

CaptureSnapshot InxRenderer::QueryCapture(uint64_t captureId) const
{
    if (!m_captureService)
        throw std::logic_error("Capture service is not initialized");
    return m_captureService->Query(captureId);
}

bool InxRenderer::CancelCapture(uint64_t captureId)
{
    if (!m_captureService || !m_captureService->Cancel(captureId))
        return false;
    m_pendingCaptures.erase(
        std::remove_if(m_pendingCaptures.begin(), m_pendingCaptures.end(),
                       [captureId](const PendingCapture &capture) { return capture.id == captureId; }),
        m_pendingCaptures.end());
    return true;
}

uint64_t InxRenderer::RequestScenePick(float x, float y, float viewportWidth, float viewportHeight)
{
    if (!m_scenePickingService)
        return 0;
    const uint64_t requestId = m_scenePickingService->Request(x, y, viewportWidth, viewportHeight);
    if (requestId != 0)
        RequestFullSpeedFrame();
    return requestId;
}

ScenePickSnapshot InxRenderer::QueryScenePick(uint64_t requestId) const
{
    if (!m_scenePickingService)
        return {requestId, ScenePickStatus::Unknown, 0, "Scene picking service is unavailable"};
    return m_scenePickingService->Query(requestId);
}

void InxRenderer::ResizeSceneRenderTarget(uint32_t width, uint32_t height)
{
    if (m_sceneRenderTarget && width > 0 && height > 0 &&
        (width != m_sceneRenderTarget->GetWidth() || height != m_sceneRenderTarget->GetHeight())) {
        auto replacementTarget = std::make_unique<SceneRenderTarget>(m_vkCore.get());
        replacementTarget->SetMsaaSampleCount(m_sceneRenderTarget->GetMsaaSampleCount());
        if (!replacementTarget->Initialize(width, height)) {
            INXLOG_ERROR("Failed to prepare replacement Scene render target for resize to ", width, "x", height,
                         "; keeping the current generation.");
            return;
        }

        std::unique_ptr<OutlineRenderer> replacementOutline;
        if (m_outlineRenderer) {
            replacementOutline = std::make_unique<OutlineRenderer>();
            if (m_outlineRenderer->IsReady() &&
                !replacementOutline->Initialize(m_vkCore.get(), replacementTarget.get())) {
                const rhi::SubmissionSerial failedCandidateEpoch =
                    m_vkCore->GetBackendContext().Queues().GetLastReservedCompletionEpoch();
                replacementTarget->RetireResourcesAfter(m_vkCore->GetRetirementQueue(), failedCandidateEpoch);
                INXLOG_ERROR("Failed to prepare replacement Outline generation for Scene resize to ", width, "x",
                             height, "; keeping the current generation.");
                return;
            }
        }

        const rhi::SubmissionSerial cutoverEpoch =
            m_vkCore->GetBackendContext().Queues().GetLastReservedCompletionEpoch();
        if (m_sceneRenderGraph)
            m_sceneRenderGraph->InvalidateBeforeTargetReplacement();

        std::unique_ptr<SceneRenderTarget> retiredTarget = std::move(m_sceneRenderTarget);
        std::unique_ptr<OutlineRenderer> retiredOutline = std::move(m_outlineRenderer);
        m_sceneRenderTarget = std::move(replacementTarget);
        m_outlineRenderer = std::move(replacementOutline);
        if (m_sceneRenderGraph)
            m_sceneRenderGraph->ReplaceSceneTarget(m_sceneRenderTarget.get());

        auto &retirementQueue = m_vkCore->GetRetirementQueue();
        if (retiredOutline) {
            std::shared_ptr<OutlineRenderer> retired(retiredOutline.release());
            retirementQueue.RetireAfter(cutoverEpoch, [retired = std::move(retired)] { retired->Cleanup(false); });
        }
        m_replacedViewTargets.push_back(std::move(retiredTarget));

        ++m_sceneRenderTargetGeneration;
        if (m_captureService)
            m_captureService->InvalidateSource(CaptureSource::Scene, m_sceneRenderTargetGeneration);
        // Update aspect ratio for projection matrix calculation
        if (m_vkCore) {
            m_vkCore->SetSceneRenderTargetSize(width, height);
        }

        // Update scene camera aspect ratio
        SceneRenderBridge::Instance().OnWindowResize(width, height);
        if (m_gui)
            m_gui->RequestFrame();
    }
}

void InxRenderer::SetShowGrid(bool show)
{
    if (m_editorGizmos) {
        m_editorGizmos->SetShowGrid(show);
        // Grid visibility change takes effect on next frame via SRP Submit()
    }
}

bool InxRenderer::IsShowGrid() const
{
    return m_editorGizmos ? m_editorGizmos->IsShowGrid() : false;
}

EditorGizmos &InxRenderer::GetEditorGizmos()
{
    if (!m_editorGizmos) {
        m_editorGizmos = std::make_unique<EditorGizmos>();
    }
    return *m_editorGizmos;
}

EditorTools *InxRenderer::GetEditorTools()
{
    return m_editorTools.get();
}

GizmosDrawCallBuffer *InxRenderer::GetGizmosDrawCallBuffer()
{
    return m_componentGizmos.get();
}

particle::ParticleGpuDrawRegistry *InxRenderer::GetParticleGpuDrawRegistry()
{
    return m_particleGpuDrawRegistry.get();
}

particle::ParticleGpuSystemManager *InxRenderer::GetParticleGpuSystemManager()
{
    if (m_particleGpuSystemManager)
        return m_particleGpuSystemManager.get();
    if (m_particleGpuSystemManagerInitializationAttempted || !m_vkCore || !m_particleGpuDrawRegistry)
        return nullptr;

    m_particleGpuSystemManagerInitializationAttempted = true;
    auto manager = std::make_unique<particle::ParticleGpuSystemManager>();
    auto particleTextureResolver = [core = m_vkCore.get()](const std::string &textureGuid,
                                                           const std::string &bindingName,
                                                           particle::GpuParticleTextureRequest) {
        particle::GpuBillboardTextureLease lease;
        TextureResolveResult resolved;
        if (textureGuid.empty() || textureGuid == "white" || textureGuid == "black" || textureGuid == "normal") {
            // SurfaceBinding already resolves ShaderInfo defaults. A property's
            // spelling must never change the meaning of its texture value.
            const char *textureKey = textureGuid == "normal"  ? "_default_normal"
                                     : textureGuid == "black" ? "_default_black"
                                                              : "white";
            auto residentSlot = core->GetTextureCache().Find(textureKey);
            auto resident = residentSlot ? residentSlot->Acquire() : nullptr;
            if (!resident || !resident->IsValid()) {
                lease.status = particle::GpuBillboardTextureStatus::Pending;
                return lease;
            }
            resolved.status = TextureResolveStatus::Ready;
            resolved.binding.gpuSlot = std::move(residentSlot);
            resolved.binding.gpuView = std::move(resident);
        } else {
            resolved = core->ResolveTextureForMaterial(textureGuid, bindingName, nullptr, false);
        }

        if (resolved.status == TextureResolveStatus::Pending) {
            lease.status = particle::GpuBillboardTextureStatus::Pending;
            return lease;
        }
        if (resolved.status != TextureResolveStatus::Ready || !resolved.binding.gpuView ||
            !resolved.binding.gpuView->IsValid()) {
            lease.status = particle::GpuBillboardTextureStatus::Failed;
            return lease;
        }
        lease.texture = resolved.binding.gpuView->GetView();
        lease.sampler = resolved.binding.gpuView->GetSampler();
        lease.gpuSlot = std::move(resolved.binding.gpuSlot);
        lease.gpuView = std::move(resolved.binding.gpuView);
        lease.status = particle::GpuBillboardTextureStatus::Ready;
        return lease;
    };
    auto particleVectorFieldTextureResolver = [core = m_vkCore.get()](const std::string &textureGuid,
                                                                      bool linearFiltering, bool repeat,
                                                                      particle::GpuParticleTextureRequest) {
        particle::GpuBillboardTextureLease lease;
        auto resolved = core->ResolveTextureForVectorField(textureGuid, linearFiltering, repeat, false);
        if (resolved.status == TextureResolveStatus::Pending) {
            lease.status = particle::GpuBillboardTextureStatus::Pending;
            return lease;
        }
        if (resolved.status != TextureResolveStatus::Ready || !resolved.binding.gpuView ||
            !resolved.binding.gpuView->IsValid()) {
            lease.status = particle::GpuBillboardTextureStatus::Failed;
            return lease;
        }
        lease.texture = resolved.binding.gpuView->GetView();
        lease.sampler = resolved.binding.gpuView->GetSampler();
        lease.gpuSlot = std::move(resolved.binding.gpuSlot);
        lease.gpuView = std::move(resolved.binding.gpuView);
        lease.status = particle::GpuBillboardTextureStatus::Ready;
        return lease;
    };
    CompiledParticleProgramBundle programs;
    if (m_particleProgramBootstrap) {
        JobSystem::Get().WaitPassive(m_particleProgramBootstrap->job);
        programs = std::move(m_particleProgramBootstrap->programs);
        m_particleProgramBootstrap.reset();
    } else {
        programs = CompileParticleProgramBundle();
    }
    if (!programs.IsValid()) {
        INXLOG_ERROR("Failed to compile one or more GPU particle support programs");
        return nullptr;
    }
    if (!manager->Initialize(m_vkCore->GetDeviceContext(), m_vkCore->GetPipelineManager(),
                             m_vkCore->GetResourceManager(), m_vkCore->GetRetirementQueue(), *m_particleGpuDrawRegistry,
                             std::move(particleTextureResolver), std::move(particleVectorFieldTextureResolver),
                             particle::ResolveSceneSkinnedMeshSource, programs.sort.View(), programs.cull.View(),
                             programs.bounds.View(), programs.migration.View(), programs.spawn.View(),
                             programs.ribbonTopology.View(), programs.ribbonRender.View(), programs.stateGather)) {
        INXLOG_ERROR("Failed to initialize the GPU particle system manager");
        return nullptr;
    }
    if (m_frameWorkOwner)
        manager->RegisterMeshUploadWork(*m_frameWorkOwner);
    m_particleGpuSystemManager = std::move(manager);
    return m_particleGpuSystemManager.get();
}

void InxRenderer::UpdateParticleCollisionScene()
{
    if (!m_particleGpuSystemManager || !m_particleGpuSystemManager->RequiresCollisionScene()) {
        // A collision-free interval has no immediately preceding simulation
        // pose. Resuming starts a new history, retaining installed topology.
        m_particleCollisionPublishedState.clear();
        m_particleCollisionHistoryPending = true;
        return;
    }

    auto &manager = SceneManager::Instance();
    manager.EnsurePhysicsQueriesCurrent();
    auto &store = PhysicsECSStore::Instance();
    Scene *scene = manager.GetActiveScene();
    const uint64_t sourceRevision = store.GetCollisionSceneRevision();
    if (scene == m_particleCollisionSourceScene && sourceRevision == m_particleCollisionSourceRevision &&
        !m_particleCollisionHistoryPending)
        return;

    const bool sceneChanged = scene != m_particleCollisionSourceScene;
    if (sceneChanged)
        m_particleCollisionPublishedState.clear();

    std::vector<PhysicsShapeState> shapes;
    shapes.reserve(store.GetAliveColliderCount());
    store.ForEachAliveActor([&](const PhysicsActorData &actor) {
        if (actor.owner->GetScene() != scene || !actor.bodyInBroadphase || actor.sceneSuspended)
            return;
        auto bodyShapes = PhysicsWorld::Instance().GetBodyShapeStates(actor.bodyId);
        shapes.insert(shapes.end(), std::make_move_iterator(bodyShapes.begin()),
                      std::make_move_iterator(bodyShapes.end()));
    });
    uint64_t topologyFingerprint = 1469598103934665603ull;
    for (const auto &shape : shapes) {
        if (shape.geometry->type != PhysicsShapeType::Mesh)
            continue;
        ParticleCollisionHashAppend(topologyFingerprint,
                                    ParticleCollisionStableHash(shape.collider->GetInstanceGuid()));
        ParticleCollisionHashAppend(topologyFingerprint, shape.geometry->identity);
    }
    const bool replaceTopology = sceneChanged || topologyFingerprint != m_particleCollisionTopologyFingerprint;
    const uint64_t topologyRevision =
        replaceTopology ? m_particleCollisionTopologyRevision + 1u : m_particleCollisionTopologyRevision;

    particle::GpuParticleCollisionSceneSnapshot snapshot;
    snapshot.revision = ++m_particleCollisionPublishedRevision;
    snapshot.topologyRevision = topologyRevision;
    snapshot.replaceMeshTopology = replaceTopology;
    snapshot.staticColliders.reserve(shapes.size());
    snapshot.dynamicColliders.reserve(shapes.size());
    std::unordered_map<uint64_t, ParticleCollisionPublishedState> nextPublishedState;
    nextPublishedState.reserve(shapes.size());
    bool moved = false;

    for (const auto &shape : shapes) {
        Collider *collider = shape.collider;
        GameObject *gameObject = collider->GetGameObject();
        const auto &geometry = *shape.geometry;
        const uint64_t componentIdentity = ParticleCollisionStableHash(collider->GetInstanceGuid());
        if (replaceTopology && geometry.type == PhysicsShapeType::Mesh) {
            particle::GpuParticleCollisionMeshGeometry mesh;
            mesh.identity = componentIdentity;
            const auto vertices = geometry.GetTriangles();
            mesh.positions.reserve(vertices.size());
            mesh.indices.reserve(vertices.size());
            for (const auto &vertex : vertices) {
                mesh.indices.push_back(static_cast<uint32_t>(mesh.positions.size()));
                mesh.positions.push_back({vertex.x, vertex.y, vertex.z, 0.0f});
            }
            snapshot.meshGeometries.push_back(std::move(mesh));
        }

        particle::GpuParticleColliderRecord record;
        record.shape = {geometry.dimensions.x, geometry.dimensions.y, geometry.dimensions.z, geometry.dimensions.w};
        const glm::mat4 colliderToWorld =
            glm::translate(glm::mat4(1.0f), shape.center) * glm::mat4_cast(shape.rotation);
        const glm::mat4 worldToCollider = glm::inverse(colliderToWorld);
        StoreParticleCollisionAffine(record.colliderToWorld, colliderToWorld);
        StoreParticleCollisionAffine(record.worldToCollider, worldToCollider);
        StoreParticleCollisionAabb(record, colliderToWorld, geometry.boundsMin, geometry.boundsMax);

        uint32_t flags = shape.trigger ? particle::GpuParticleColliderTrigger : 0u;
        if (geometry.convex)
            flags |= particle::GpuParticleColliderConvex;
        if (shape.moving)
            flags |= particle::GpuParticleColliderDynamic;
        record.linearVelocity = {shape.linearVelocity.x, shape.linearVelocity.y, shape.linearVelocity.z, 0.0f};
        record.angularVelocity = {shape.angularVelocity.x, shape.angularVelocity.y, shape.angularVelocity.z, 0.0f};
        record.centerOfMass = {shape.bodyCenterOfMass.x, shape.bodyCenterOfMass.y, shape.bodyCenterOfMass.z, 0.0f};
        record.material = {collider->GetFriction(), collider->GetBounciness(),
                           static_cast<float>(collider->GetFrictionCombine()),
                           static_cast<float>(collider->GetBounceCombine())};
        record.metadata = {static_cast<uint32_t>(geometry.type), shape.layerBit, flags, 0u};
        const uint64_t objectIdentity = gameObject->GetID();
        record.identity = {static_cast<uint32_t>(componentIdentity), static_cast<uint32_t>(componentIdentity >> 32u),
                           static_cast<uint32_t>(objectIdentity), static_cast<uint32_t>(objectIdentity >> 32u)};
        const auto previous = m_particleCollisionPublishedState.find(componentIdentity);
        const bool sameGeometry = previous != m_particleCollisionPublishedState.end() &&
                                  previous->second.geometryIdentity == geometry.identity;
        if (sameGeometry) {
            CopyParticleCollisionAffine(record.previousWorldToCollider, previous->second.worldToCollider);
            record.previousWorldAabbMin = previous->second.worldAabbMin;
            record.previousWorldAabbMax = previous->second.worldAabbMax;
        } else {
            record.previousWorldToCollider = record.worldToCollider;
            record.previousWorldAabbMin = record.worldAabbMin;
            record.previousWorldAabbMax = record.worldAabbMax;
        }
        auto &nextState = nextPublishedState[componentIdentity];
        nextState.geometryIdentity = geometry.identity;
        CopyParticleCollisionAffine(nextState.worldToCollider, record.worldToCollider);
        moved |= sameGeometry && nextState.worldToCollider != previous->second.worldToCollider;
        nextState.worldAabbMin = record.worldAabbMin;
        nextState.worldAabbMax = record.worldAabbMax;
        if (shape.moving)
            snapshot.dynamicColliders.push_back(record);
        else
            snapshot.staticColliders.push_back(record);
    }

    std::string error;
    const bool published = m_particleGpuSystemManager->PublishCollisionScene(snapshot, &error);
    if (published) {
        m_particleCollisionSourceScene = scene;
        m_particleCollisionSourceRevision = sourceRevision;
        // Retire the one-frame sweep even when the next frame has no pose edits.
        m_particleCollisionHistoryPending = moved;
        m_particleCollisionPublishedState = std::move(nextPublishedState);
        if (replaceTopology) {
            m_particleCollisionTopologyFingerprint = topologyFingerprint;
            m_particleCollisionTopologyRevision = topologyRevision;
        }
    } else {
        INXLOG_ERROR("Failed to publish GPU particle collision scene: ", error);
    }
}

uint32_t InxRenderer::GetMaxFramesInFlight() const noexcept
{
    return m_vkCore ? m_vkCore->GetMaxFramesInFlight() : 0;
}

bool InxRenderer::RefreshMaterialPipeline(std::shared_ptr<InxMaterial> material)
{
    INXLOG_DEBUG("RefreshMaterialPipeline called");

    if (!material) {
        INXLOG_ERROR("RefreshMaterialPipeline: material is null");
        return false;
    }

    INXLOG_DEBUG("RefreshMaterialPipeline: material name = ", material->GetName());

    if (!m_vkCore) {
        INXLOG_ERROR("RefreshMaterialPipeline: VkCore is null");
        return false;
    }

    std::shared_ptr<const ShaderProgramArtifact> shaderProgram;
    try {
        shaderProgram = ResolveShaderProgramArtifact(material, std::nullopt);
    } catch (const ShaderProgramSelectionError &) {
        return m_vkCore->RefreshMaterialPipeline(material);
    }

    // Get shader names from material
    const std::string &vertName = material->GetVertShaderName();
    const std::string &fragName = material->GetFragShaderName();

    INXLOG_DEBUG("RefreshMaterialPipeline: vert='", vertName, "' frag='", fragName, "'");

    if (fragName.empty()) {
        INXLOG_ERROR("RefreshMaterialPipeline: fragment shader name is empty");
        return false;
    }

    bool refreshed = true;
    if (!shaderProgram || shaderProgram->domain != ShaderProgramDomain::ParticleSprite)
        refreshed = m_vkCore->RefreshMaterialPipeline(material);

    if (m_particleGpuSystemManager) {
        std::string particleError;
        if (!m_particleGpuSystemManager->RefreshMaterialProgram(material, shaderProgram, &particleError)) {
            INXLOG_ERROR("RefreshMaterialPipeline: ", particleError);
            refreshed = false;
        }
    }
    return refreshed;
}

std::shared_ptr<PreviewImage> InxRenderer::BeginMaterialPreviewGPU(const std::shared_ptr<InxMaterial> &material,
                                                                   int size, bool &progressed, bool *texturePending)
{
    if (texturePending)
        *texturePending = false;
    if (!m_vkCore || !material)
        return nullptr;
    // A material sphere is mesh geometry. UI and particle programs have their
    // own vertex/descriptor contract and are represented by the static CPU
    // thumbnail instead of being published into a mesh preview pipeline.
    if (!CanPreviewMaterialOnMesh(material))
        return nullptr;
    return m_vkCore->BeginMaterialPreviewGPU(material, size, progressed, texturePending);
}

bool InxRenderer::PrepareMeshPreviewMaterials(const std::vector<std::shared_ptr<InxMaterial>> &sources,
                                              std::vector<std::shared_ptr<InxMaterial>> &prepared, bool &progressed)
{
    if (!m_vkCore)
        return false;
    return m_vkCore->PrepareMeshPreviewMaterials(sources, prepared, progressed);
}

std::shared_ptr<PreviewImage>
InxRenderer::BeginMeshPreviewGPU(const InxMesh &mesh, const std::vector<std::shared_ptr<InxMaterial>> &materials,
                                 int size, std::shared_ptr<PreviewGeometryUpload> &geometry, bool &progressed,
                                 bool &materialsChanged)
{
    return m_vkCore ? m_vkCore->BeginMeshPreviewGPU(mesh, materials, size, geometry, progressed, materialsChanged)
                    : nullptr;
}

uint64_t InxRenderer::PublishPreviewImageForImGui(const std::string &name, const std::shared_ptr<PreviewImage> &image)
{
    if (!m_gui || !image || !image->IsComplete())
        return 0;
    return m_gui->PublishTextureViewForImGui(name, image->texture, false, image->texture->GetResidentBytes());
}

uint64_t InxRenderer::RenderMeshPreviewGPUImGuiCamera(const InxMesh &mesh,
                                                      const std::vector<std::shared_ptr<InxMaterial>> &materials,
                                                      int size, const glm::mat4 &view, const glm::mat4 &proj,
                                                      const glm::vec3 &cameraPos,
                                                      std::shared_ptr<PreviewGeometryUpload> &geometry,
                                                      bool &progressed, bool cloneMaterials)
{
    if (!m_vkCore)
        return 0;
    return m_vkCore->RenderMeshPreviewGPUImGuiCamera(mesh, materials, size, view, proj, cameraPos, geometry, progressed,
                                                     cloneMaterials);
}

uint64_t InxRenderer::GetMeshPreviewDisplayTextureId() const
{
    return m_vkCore ? m_vkCore->GetMeshPreviewDisplayTextureId() : 0;
}

uint64_t InxRenderer::RenderModelAnimationPreview(const std::shared_ptr<InxMesh> &mesh, const std::string &take,
                                                  float seconds, int size, uint64_t dependencyRevision)
{
    return m_vkCore ? m_vkCore->RenderModelAnimationPreview(mesh, take, seconds, size, dependencyRevision) : 0;
}

void InxRenderer::InvalidateShaderCache(const std::string &shaderId, const std::string &shaderType)
{
    INXLOG_DEBUG("InvalidateShaderCache called: ", shaderId);

    if (!m_vkCore) {
        INXLOG_ERROR("InvalidateShaderCache: VkCore is null");
        return;
    }

    // Fullscreen RenderEffect pipelines are not part of the material cache.
    // Retire their old shader revision before publishing the replacement raw
    // module so the next graph record cannot silently keep the old effect.
    if (m_sceneRenderGraph)
        m_sceneRenderGraph->InvalidateFullscreenShader(shaderId);
    for (auto &[cameraId, graph] : m_gameRenderGraphs) {
        (void)cameraId;
        graph->InvalidateFullscreenShader(shaderId);
    }

    m_vkCore->InvalidateShaderCache(shaderId, shaderType);
}

void InxRenderer::InvalidateTextureCache(const std::string &texturePath)
{
    INXLOG_DEBUG("InvalidateTextureCache called: ", texturePath);

    if (!m_vkCore) {
        INXLOG_ERROR("InvalidateTextureCache: VkCore is null");
        return;
    }

    m_vkCore->InvalidateTextureCache(texturePath);
    if (m_particleGpuSystemManager)
        m_particleGpuSystemManager->InvalidateTextureAssets(texturePath);
}

void InxRenderer::InvalidateMeshCache(const std::string &meshGuid)
{
    if (m_vkCore)
        m_vkCore->InvalidateMeshCache(meshGuid);
}

void InxRenderer::RemoveMaterialPipeline(const std::string &materialName)
{
    if (!m_vkCore) {
        INXLOG_ERROR("RemoveMaterialPipeline: VkCore is null");
        return;
    }
    m_vkCore->RemoveMaterialPipeline(materialName);
}

SceneRenderGraph *InxRenderer::GetSceneRenderGraph()
{
    return m_sceneRenderGraph.get();
}

std::shared_ptr<GraphEventInterest> InxRenderer::WatchGraphEvents(const std::string &label)
{
    if (const auto existing = m_graphEventInterests.find(label); existing != m_graphEventInterests.end())
        if (auto interest = existing->second.lock())
            return interest;
    for (auto entry = m_graphEventInterests.begin(); entry != m_graphEventInterests.end();) {
        if (entry->second.expired())
            entry = m_graphEventInterests.erase(entry);
        else
            ++entry;
    }
    auto interest = std::make_shared<GraphEventInterest>();
    m_graphEventInterests.emplace(label, interest);
    return interest;
}

GraphEventQueue InxRenderer::TakeGraphEvents()
{
    GraphEventQueue events;
    events.splice(events.end(), m_graphEvents);
    return events;
}

SceneRenderGraph *InxRenderer::GetCameraRenderGraph(uint64_t cameraComponentId)
{
    if (cameraComponentId == 0)
        return m_gameRenderGraph;
    const auto found = m_gameRenderGraphs.find(cameraComponentId);
    return found == m_gameRenderGraphs.end() ? nullptr : found->second.get();
}

uint64_t InxRenderer::RequestGpuParticleViewDiagnostics(bool gameView, uint64_t graphInstanceId,
                                                        uint64_t cameraComponentId)
{
    SceneRenderGraph *graph = m_sceneRenderGraph.get();
    if (gameView) {
        (void)FindGameCamerasCached();
        if (cameraComponentId == 0) {
            graph = m_gameRenderGraph;
        } else if (const auto found = m_gameRenderGraphs.find(cameraComponentId); found != m_gameRenderGraphs.end()) {
            graph = found->second.get();
        } else {
            graph = nullptr;
        }
    } else if (cameraComponentId != 0) {
        return 0;
    }
    return graph ? graph->RequestParticleViewDiagnostics(graphInstanceId) : 0;
}

particle::GpuParticleViewDiagnosticSnapshot
InxRenderer::QueryGpuParticleViewDiagnostics(bool gameView, uint64_t requestId, uint64_t cameraComponentId) const
{
    const SceneRenderGraph *graph = m_sceneRenderGraph.get();
    uint64_t resolvedCameraComponentId = 0;
    if (gameView) {
        if (cameraComponentId == 0) {
            graph = m_gameRenderGraph;
            resolvedCameraComponentId = m_cachedGameCamera ? m_cachedGameCamera->GetComponentID() : 0;
        } else if (const auto found = m_gameRenderGraphs.find(cameraComponentId); found != m_gameRenderGraphs.end()) {
            graph = found->second.get();
            resolvedCameraComponentId = cameraComponentId;
        } else {
            graph = nullptr;
        }
    } else if (cameraComponentId != 0) {
        graph = nullptr;
    }
    if (!graph)
        return {};
    auto snapshot = graph->QueryParticleViewDiagnostics(requestId);
    snapshot.cameraComponentId = resolvedCameraComponentId;
    snapshot.renderViewId = graph->GetRenderViewContext().id;
    return snapshot;
}

std::shared_ptr<InxMaterial> InxRenderer::GetFirstMeshRendererMaterial()
{
    Scene *activeScene = SceneManager::Instance().GetActiveScene();
    if (!activeScene) {
        return nullptr;
    }

    std::vector<GameObject *> allObjects = activeScene->GetAllObjects();
    for (GameObject *obj : allObjects) {
        if (!obj->IsActiveInHierarchy()) {
            continue;
        }

        MeshRenderer *renderer = obj->GetComponent<MeshRenderer>();
        if (!renderer || !renderer->IsEnabled()) {
            continue;
        }

        // Return the first valid material found
        auto material = renderer->GetEffectiveMaterial();
        if (material) {
            return material;
        }
    }

    return nullptr;
}

void InxRenderer::SetRenderPipeline(std::shared_ptr<RenderPipelineCallback> pipeline)
{
    // Dispose the old pipeline if replacing
    if (m_renderPipeline) {
        m_renderPipeline->Dispose();
    }
    m_renderPipeline = std::move(pipeline);

    if (m_renderPipeline) {
        INXLOG_INFO("Render pipeline set (Python SRP path active)");
    } else {
        INXLOG_INFO("Render pipeline cleared (C++ default path active)");
    }
}

// ============================================================================
// Game Camera Render Target
// ============================================================================

Camera *InxRenderer::FindGameCamera()
{
    Scene *activeScene = SceneManager::Instance().GetActiveScene();
    if (!activeScene)
        return nullptr;

    Camera *editorCam = SceneManager::Instance().GetEditorCameraController().GetCamera();
    return activeScene->FindGameCamera(editorCam);
}

std::vector<Camera *> InxRenderer::FindGameCameras()
{
    Scene *activeScene = SceneManager::Instance().GetActiveScene();
    if (!activeScene)
        return {};
    Camera *editorCam = SceneManager::Instance().GetEditorCameraController().GetCamera();
    return activeScene->GetActiveGameCameras(editorCam);
}

std::shared_ptr<rhi::RenderTexture> InxRenderer::GetCameraOutputTexture(Camera &camera)
{
    if (!camera.GetRenderToTexture())
        return nullptr;
    if (!camera.WantsMemoryOutput()) {
        const auto &guid = camera.GetTargetTextureGuid();
        const auto *database = AssetRegistry::Instance().GetAssetDatabase();
        if (!camera.GetTargetTexture() && !guid.empty() && database && database->GetMetaByGuid(guid))
            camera.SetTargetTexture(LoadRenderTexture(guid));
        return camera.GetOutputTexture();
    }
    auto &texture = camera.m_memoryOutputTexture;
    const auto size = camera.GetMemoryOutputSize();
    if (!texture) {
        rhi::RenderTextureDesc description;
        description.width = size.x;
        description.height = size.y;
        description.colorFormat = rhi::PixelFormat::RGBA32SFloat;
        description.depthFormat = rhi::PixelFormat::D32SFloat;
        texture = CreateRenderTexture(description);
    } else {
        const auto generation = texture->Acquire();
        if (generation->width != size.x || generation->height != size.y) {
            auto description = generation->description;
            description.width = size.x;
            description.height = size.y;
            texture->Reconfigure(description);
        }
    }
    return texture;
}

void InxRenderer::RequestCameraRender(Camera &camera)
{
    if (camera.GetRenderToTexture() && camera.GetRenderOnDemand()) {
        camera.m_requestedRenderSerials[m_cameraRequestSerial % 2] = m_cameraRequestSerial;
        RequestFullSpeedFrame();
    }
}

SceneRenderGraph *InxRenderer::EnsureGameRenderGraph(Camera *camera)
{
    if (!camera || !m_vkCore)
        return nullptr;
    const auto &texture = camera->GetOutputTexture();
    if (!texture && (camera->GetRenderToTexture() || !m_gameRenderTarget || !m_gameRenderTarget->IsReady()))
        return nullptr;

    const uint64_t cameraId = camera->GetComponentID();
    if (texture) {
        const auto allocation = texture->Acquire();
        camera->SetScreenDimensions(allocation->width, allocation->height);
    } else {
        camera->SetScreenDimensions(m_gameRenderTarget->GetWidth(), m_gameRenderTarget->GetHeight());
    }
    auto found = m_gameRenderGraphs.find(cameraId);
    if (found != m_gameRenderGraphs.end()) {
        found->second->SetOutputTexture(texture);
        return found->second->SetCameraViewport(*camera) ? found->second.get() : nullptr;
    }

    auto graph = std::make_unique<SceneRenderGraph>();
    if (!graph->Initialize(m_vkCore.get(), m_gameRenderTarget.get(), rhi::RenderViewKind::Game, texture)) {
        INXLOG_ERROR("Failed to initialize Game RenderGraph for Camera component ", cameraId);
        return nullptr;
    }
    graph->SetParticleGpuDrawRegistry(m_particleGpuDrawRegistry.get());
    graph->SetEffectiveMsaaSamples(GetMsaaSamples());
    const bool visibleViewport = graph->SetCameraViewport(*camera);
    SceneRenderGraph *result = graph.get();
    m_gameRenderGraphs.emplace(cameraId, std::move(graph));
    return visibleViewport ? result : nullptr;
}

bool InxRenderer::ShouldRenderGameCamera(const Camera *camera, bool allowScreen) const
{
    if (camera->GetRenderToTexture())
        return camera->GetOutputTexture() &&
               (!camera->GetRenderOnDemand() ||
                (m_viewRequestSerial != 0 &&
                 camera->m_requestedRenderSerials[m_viewRequestSerial % 2] == m_viewRequestSerial) ||
                HasPendingCapture(CaptureSource::Camera, camera->GetComponentID()));
    return (allowScreen && m_gameCameraEnabled) || HasPendingCapture(CaptureSource::Game);
}

bool InxRenderer::HasActiveGameViews(bool allowScreen)
{
    if ((allowScreen && m_gameCameraEnabled) || HasPendingCapture(CaptureSource::Game))
        return true;
    const auto &cameras = FindGameCamerasCached();
    return std::any_of(cameras.begin(), cameras.end(), [this, allowScreen](const Camera *camera) {
        return ShouldRenderGameCamera(camera, allowScreen);
    });
}

void InxRenderer::CollectRequestedViews(bool sceneActive, bool gameActive)
{
    m_nextScheduledViews.clear();
    const auto add = [&](SceneRenderGraph *graph, bool scene) {
        if (graph && graph->IsViewRequested(m_viewRequestSerial))
            m_nextScheduledViews.push_back({graph, graph->GetRenderViewContext().id, 0, scene});
    };
    if (sceneActive)
        add(m_sceneRenderGraph.get(), true);
    if (gameActive) {
        for (const auto *camera : FindGameCamerasCached()) {
            const auto view = m_gameRenderGraphs.find(camera->GetComponentID());
            if (view != m_gameRenderGraphs.end())
                add(view->second.get(), false);
        }
    }
    add(m_sceneCameraPreviewGraph.get(), false);
}

void InxRenderer::UpdateViewSchedule()
{
    // The last admitted screen view composites the screen-space lists.
    // Prepare from the end to assign that owner before building each view.
    bool hasScreenUIOwner = false;
    bool waitingForResults = false;
    for (auto entry = m_nextScheduledViews.rbegin(); entry != m_nextScheduledViews.rend(); ++entry) {
        auto &view = *entry;
        const bool game = view.graph->GetRenderViewContext().kind == rhi::RenderViewKind::Game;
        const bool ownsScreenUI = game && !hasScreenUIOwner && !view.graph->HasOutputTexture();
        if (game)
            view.graph->SetScreenUIRenderer(m_screenUIRenderer.get(), ownsScreenUI);
        view.graph->EnsureGraphBuilt();
        view.waitingForResult = view.graph->HasPendingBufferInputs();
        view.admitted = !view.waitingForResult && view.graph->AdmitGraphEvents(m_graphEventInterests);
        view.accessRevision = view.graph->GetResourceAccessRevision();
        hasScreenUIOwner |= view.admitted && ownsScreenUI;
        waitingForResults |= view.waitingForResult;
    }
    const bool changed = m_scheduledViews != m_nextScheduledViews;
    if (!waitingForResults && !changed)
        return;
    m_scheduledViews.swap(m_nextScheduledViews);
    std::vector<RenderViewAccess> accesses;
    accesses.reserve(m_scheduledViews.size());
    try {
        for (const auto &view : m_scheduledViews) {
            accesses.push_back(view.graph->GetResourceAccess());
            accesses.back().admitted = view.admitted;
        }
        m_viewSchedule = RenderViewSchedule::Build(accesses);
        for (auto *input : m_viewSchedule.cyclicInputs)
            input->MarkResultCycle();
        if (changed) {
            for (const auto index : m_viewSchedule.missingProducers)
                INXLOG_WARN("Render view ", m_scheduledViews[index].viewId, " is waiting for a RenderTexture producer");
            for (const auto index : m_viewSchedule.feedbackViews)
                INXLOG_ERROR("Render view ", m_scheduledViews[index].viewId, " samples its own RenderTexture output");
        }
    } catch (const std::invalid_argument &error) {
        m_viewSchedule = {};
        if (changed)
            INXLOG_ERROR("Render view publication failed: ", error.what());
    }
}

const std::vector<Camera *> &InxRenderer::FindGameCamerasCached()
{
    const auto &manager = SceneManager::Instance();
    const auto &scenes = manager.GetAllScenes();
    const auto state = [](const Scene *scene) {
        return std::make_pair(scene ? scene->GetWorldId() : uint64_t{0},
                              scene ? scene->GetStructureVersion() : uint64_t{0});
    };
    const auto activeWorld = state(manager.GetActiveScene()).first;
    const auto persistent = state(manager.GetRuntimePersistentScene());
    bool unchanged = m_gameCameraCacheValid && m_cachedCameraActiveWorld == activeWorld &&
                     m_cachedCameraSceneVersions.size() == scenes.size() + 1 &&
                     m_cachedCameraSceneVersions.back() == persistent;
    for (size_t i = 0; unchanged && i < scenes.size(); ++i)
        unchanged = m_cachedCameraSceneVersions[i] == state(scenes[i].get());
    if (unchanged)
        return m_cachedGameCameras;

    // Camera pointers are borrowed from scene ownership. Same-frame edits,
    // additive unload and queries between frames must observe the existing
    // structure versions before dereferencing them. No entity scan/hash on a
    // cache hit, and no attempt to validate or recover a dangling pointer.
    m_cachedGameCameras = FindGameCameras();
    for (Camera *camera : m_cachedGameCameras) {
        if (!camera->GetRenderToTexture())
            continue;
        try {
            (void)GetCameraOutputTexture(*camera);
        } catch (const std::exception &error) {
            // User-authored device/attachment errors keep their GUID visible.
            // Never route a failed offscreen output into the screen camera.
            camera->m_memoryOutputTexture.reset();
            INXLOG_ERROR("Camera output ", camera->GetComponentID(), " could not be prepared: ", error.what());
        }
    }
    // Binding an output changes the structure version; publish the final state.
    m_cachedCameraActiveWorld = activeWorld;
    m_cachedCameraSceneVersions.resize(scenes.size() + 1);
    for (size_t i = 0; i < scenes.size(); ++i)
        m_cachedCameraSceneVersions[i] = state(scenes[i].get());
    m_cachedCameraSceneVersions.back() = state(manager.GetRuntimePersistentScene());
    m_cachedGameCamera = FindGameCamera();
    if (m_cachedGameCamera && m_cachedGameCamera->GetRenderToTexture()) {
        const auto screen =
            std::find_if(m_cachedGameCameras.begin(), m_cachedGameCameras.end(), [](const Camera *camera) {
                return !camera->GetRenderToTexture() && camera->GetPixelWidth() && camera->GetPixelHeight();
            });
        m_cachedGameCamera = screen == m_cachedGameCameras.end() ? nullptr : *screen;
    }
    m_gameCameraCacheValid = true;

    for (auto it = m_gameRenderGraphs.begin(); it != m_gameRenderGraphs.end();) {
        const bool active = std::any_of(
            m_cachedGameCameras.begin(), m_cachedGameCameras.end(),
            [cameraId = it->first](const Camera *camera) { return camera && camera->GetComponentID() == cameraId; });
        if (active) {
            ++it;
            continue;
        }
        if (m_gameRenderGraph == it->second.get())
            m_gameRenderGraph = nullptr;
        it = m_gameRenderGraphs.erase(it);
    }

    const auto lastScreen =
        std::find_if(m_cachedGameCameras.rbegin(), m_cachedGameCameras.rend(), [](const Camera *camera) {
            return !camera->GetRenderToTexture() && camera->GetPixelWidth() && camera->GetPixelHeight();
        });
    m_gameRenderGraph = m_cachedGameCamera ? EnsureGameRenderGraph(m_cachedGameCamera) : nullptr;
    if (!m_gameRenderGraph && lastScreen != m_cachedGameCameras.rend())
        m_gameRenderGraph = EnsureGameRenderGraph(*lastScreen);

    return m_cachedGameCameras;
}

Camera *InxRenderer::FindGameCameraCached()
{
    (void)FindGameCamerasCached();
    if (m_cachedGameCamera && m_cachedGameCamera->GetPixelWidth() && m_cachedGameCamera->GetPixelHeight())
        return m_cachedGameCamera;
    const auto visible = std::find_if(m_cachedGameCameras.begin(), m_cachedGameCameras.end(), [](const Camera *camera) {
        return !camera->GetRenderToTexture() && camera->GetPixelWidth() && camera->GetPixelHeight();
    });
    return visible == m_cachedGameCameras.end() ? nullptr : *visible;
}

InxRenderer::GameImageGeneration InxRenderer::CurrentGameImageGeneration() const
{
    const auto &manager = SceneManager::Instance();
    const Scene *scene = manager.GetActiveScene();
    return {m_gameRenderTargetGeneration, scene ? scene->GetWorldId() : 0, manager.GetSimulationEpoch(),
            scene ? scene->GetTemporalDiscontinuityRevision() : 0};
}

bool InxRenderer::IsGamePresentationReady() const
{
    return m_gameRenderTarget && m_gameRenderTarget->IsReady() &&
           m_acceptedGameImage.generation == CurrentGameImageGeneration();
}

uint64_t InxRenderer::GetGameTextureId()
{
    if (m_gameRenderTarget && m_gameRenderTarget->IsReady()) {
        // Resolving here makes the GUI's no-camera state change in the same
        // frame instead of briefly presenting the previous camera image.
        if (FindGameCameraCached())
            return m_gameRenderTarget->GetImGuiTextureId();
        return 0;
    }
    return 0;
}

void InxRenderer::ResizeGameRenderTarget(uint32_t width, uint32_t height, rhi::FilterMode filter)
{
    if (width == 0 || height == 0) {
        return;
    }

    // Lazy-initialize game render target on first resize request
    if (!m_gameRenderTarget) {
        // INXLOG_INFO("Game render target: lazy-initializing (", width, "x", height, ")");
        m_gameRenderTarget = std::make_unique<SceneRenderTarget>(m_vkCore.get());
        // Match the current MSAA setting from the scene render target.
        // Without this, the game render target defaults to 4x MSAA even
        // when the engine has already switched to 1x, causing a layout
        // mismatch (COLOR_ATTACHMENT_OPTIMAL vs SHADER_READ_ONLY_OPTIMAL)
        // that triggers VK_ERROR_DEVICE_LOST.
        if (m_sceneRenderTarget) {
            m_gameRenderTarget->SetMsaaSampleCount(m_sceneRenderTarget->GetMsaaSampleCount());
        }
        m_gameRenderTarget->Initialize(width, height, filter);
        ++m_gameRenderTargetGeneration;

        EnsureScreenUIRenderer();
        // Pre-discover the ordered camera stack and create one graph per
        // camera so the first Game frame cannot reuse Scene-view state.
        m_gameCameraCacheValid = false;
        (void)FindGameCamerasCached();

        return;
    }

    if (width != m_gameRenderTarget->GetWidth() || height != m_gameRenderTarget->GetHeight() ||
        filter != m_gameRenderTarget->GetFilterMode()) {
        auto replacementTarget = std::make_unique<SceneRenderTarget>(m_vkCore.get());
        replacementTarget->SetMsaaSampleCount(m_gameRenderTarget->GetMsaaSampleCount());
        if (!replacementTarget->Initialize(width, height, filter)) {
            INXLOG_ERROR("Failed to prepare replacement Game render target for resize to ", width, "x", height,
                         "; keeping the current generation.");
            return;
        }

        // Allocate every dependent generation before replacing the Game target.
        // No graph can observe a partially published reference-size transition.
        ResizeRelativeRenderTextures(width, height);
        for (auto &[cameraId, graph] : m_gameRenderGraphs) {
            (void)cameraId;
            graph->InvalidateBeforeTargetReplacement();
        }

        std::unique_ptr<SceneRenderTarget> retiredTarget = std::move(m_gameRenderTarget);
        m_gameRenderTarget = std::move(replacementTarget);
        for (auto &[cameraId, graph] : m_gameRenderGraphs) {
            (void)cameraId;
            graph->ReplaceSceneTarget(m_gameRenderTarget.get());
        }
        m_replacedViewTargets.push_back(std::move(retiredTarget));

        ++m_gameRenderTargetGeneration;
        if (m_captureService)
            m_captureService->InvalidateSource(CaptureSource::Game, m_gameRenderTargetGeneration);
        if (m_gui)
            m_gui->RequestFrame();
    }
}

void InxRenderer::InvalidateTemporalHistory(bool sceneView, bool gameView)
{
    bool invalidated = false;
    if (sceneView && m_sceneRenderGraph) {
        m_sceneRenderGraph->InvalidateTemporalHistory();
        invalidated = true;
    }
    if (gameView) {
        for (auto &[cameraId, graph] : m_gameRenderGraphs) {
            (void)cameraId;
            graph->InvalidateTemporalHistory();
            invalidated = true;
        }
    }
    if (invalidated)
        ++m_temporalHistoryInvalidationCount;
}

void InxRenderer::ConsumeSceneTemporalDiscontinuity()
{
    Scene *scene = SceneManager::Instance().GetActiveScene();
    const uint64_t worldId = scene ? scene->GetWorldId() : 0;
    if (worldId != m_temporalHistoryWorldId) {
        m_temporalHistoryWorldId = worldId;
        m_observedSceneTemporalDiscontinuityRevision = scene ? scene->GetTemporalDiscontinuityRevision() : 0;
        InvalidateGpuViewStateForSceneBoundary();
        m_gameRenderGraph = nullptr;
        m_gameRenderGraphs.clear();
        m_gameCameraCacheValid = false;
        m_cachedGameCamera = nullptr;
        m_cachedGameCameras.clear();
        return;
    }
    if (!scene)
        return;

    const uint64_t revision = scene->GetTemporalDiscontinuityRevision();
    if (revision == m_observedSceneTemporalDiscontinuityRevision)
        return;
    m_observedSceneTemporalDiscontinuityRevision = revision;
    // Play/Stop keeps the same world id. Treat that edge like opening Scene:
    // drop compiled Scene/Game graphs and rebuild from the current generation.
    InvalidateGpuViewStateForSceneBoundary();
    InvalidateTemporalHistory();
}

void InxRenderer::InvalidateGpuViewStateForSceneBoundary()
{
    if (m_particleGpuSystemManager) {
        m_particleGpuSystemManager->AbortAsyncRecording();
        // View invalidation runs after Scene Update. Particle owners have
        // already rebuilt and scheduled the new scene's first frame here.
        // Resetting simulation would discard that frame's initial bursts;
        // scene/component lifecycle owns simulation resets, not view caches.
    }
    if (m_sceneRenderGraph)
        m_sceneRenderGraph->InvalidateParticleViews();
    for (auto &[cameraId, graph] : m_gameRenderGraphs) {
        (void)cameraId;
        graph->InvalidateParticleViews();
    }
    WaitForGpuIdle();
    // Game cameras keep the same IDs across Play/Stop. Destroy the compiled
    // graphs only after the device is idle: SceneRenderGraph::Destroy releases
    // RHI objects immediately. The next Game frame must import the current
    // particle generation instead of recycled handles from two Play cycles ago.
    m_gameRenderGraph = nullptr;
    m_gameRenderGraphs.clear();
    m_gameCameraCacheValid = false;
    m_cachedGameCamera = nullptr;
    m_cachedGameCameras.clear();
    m_sceneCameraPreviewGraph.reset();
    m_sceneCameraPreviewTarget.reset();
    m_sceneCameraPreviewRenderedId = 0;
    m_sceneCameraPreviewDue = {};
    RecreateSceneRenderGraph();
    if (m_vkCore)
        m_vkCore->FlushRetiredGpuResources();
}

void InxRenderer::RecreateSceneRenderGraph()
{
    if (!m_sceneRenderGraph)
        return;
    const int samples = GetMsaaSamples();
    auto previous = std::move(m_sceneRenderGraph);
    if (!m_vkCore || !m_sceneRenderTarget)
        return;
    auto replacement = std::make_unique<SceneRenderGraph>();
    if (!replacement->Initialize(m_vkCore.get(), m_sceneRenderTarget.get(), rhi::RenderViewKind::Scene)) {
        INXLOG_ERROR("Failed to reinitialize the Scene View render graph after a scene boundary");
        return;
    }
    // The caller has drained old GPU work. Initialize first, then transfer only
    // graph-owned allocations; fresh view identity/history and bindings remain.
    replacement->TakeReusableImagesFrom(*previous);
    previous.reset();
    m_sceneRenderGraph = std::move(replacement);
    if (m_particleGpuDrawRegistry)
        m_sceneRenderGraph->SetParticleGpuDrawRegistry(m_particleGpuDrawRegistry.get());
    if (m_screenUIRenderer)
        m_sceneRenderGraph->SetScreenUIRenderer(m_screenUIRenderer.get());
    if (samples > 0)
        m_sceneRenderGraph->SetEffectiveMsaaSamples(samples);
    if (m_sceneViewVisible && m_outlineRenderer)
        m_sceneRenderGraph->SetOutlineRenderer(m_outlineRenderer.get());
}

void InxRenderer::SetSceneViewVisible(bool visible)
{
    if (m_sceneViewVisible == visible)
        return;
    m_sceneViewVisible = visible;
    if (!visible)
        m_sceneCameraPreviewId = 0;

    // Hiding a panel only invalidates its camera-dependent submissions.  The
    // compiled graph description remains authoritative for shared pipeline
    // settings such as the shadow atlas resolution.
    if (!visible && m_sceneRenderGraph) {
        m_sceneRenderGraph->ClearCachedViewSubmission();
    }
}

void InxRenderer::ConfigureScreenUIMaterialResolver(InxScreenUIRenderer &renderer)
{
    renderer.SetFontTexturePublisher([this] { m_gui->PrepareRuntimeFontTextures(); });
    renderer.SetMaterialAssetResolver([](const std::string &guid, uint64_t generation) {
        auto material = AssetRegistry::Instance().GetAsset<InxMaterial>(guid);
        if (!material || material->IsDeleted() || material->GetVersion() != generation)
            throw std::runtime_error("UI material GUID is missing or its retained generation is stale: " + guid);
        return std::shared_ptr<const InxMaterial>(std::move(material));
    });
    renderer.SetMaterialTextureResolver(
        [this](const std::string &textureGuid, const std::string &bindingName, const MaterialTextureSampler *sampler) {
            if (!m_vkCore)
                throw std::logic_error("UI material texture resolver requires a live Vulkan core");
            if (textureGuid == "white" || textureGuid == "black" || textureGuid == "normal") {
                const bool normal = textureGuid == "normal" || bindingName.find("normal") != std::string::npos ||
                                    bindingName.find("Normal") != std::string::npos;
                auto slot = m_vkCore->GetTextureCache().Find(normal ? "_default_normal" : "white");
                auto &device = m_vkCore->GetDeviceContext().GetRhiDevice();
                return ResolveBuiltinUIMaterialTexture(slot, device, sampler);
            }
            return m_vkCore->ResolveTextureForMaterial(textureGuid, bindingName, sampler);
        });
    renderer.SetMaterialTextureGenerationResolver([](const std::string &textureGuid) -> uint64_t {
        if (textureGuid == "white" || textureGuid == "black" || textureGuid == "normal")
            return 1;
        return AssetRegistry::Instance().GetAssetVersion(textureGuid);
    });
    renderer.SetMaterialProgramAcquire([this](const ShaderProgramKey &key) {
        if (m_vkCore)
            m_vkCore->AcquireUIShaderProgramOwner(key);
    });
    renderer.SetMaterialProgramRelease([this](const ShaderProgramKey &key) {
        if (m_vkCore)
            m_vkCore->ReleaseUIShaderProgramOwner(key);
    });
    renderer.SetMaterialProgramReleaseSweep([this] {
        if (m_vkCore)
            m_vkCore->SweepReleasedUIShaderProgramArtifacts();
    });
    renderer.SetMaterialIdentityValidator([](const std::string &guid, uint64_t generation) {
        const auto material = AssetRegistry::Instance().GetAsset<InxMaterial>(guid);
        return material && !material->IsDeleted() && material->GetVersion() == generation;
    });
    renderer.SetMaterialProgramResolver([this](const std::string &guid, uint64_t generation,
                                               ShaderProgramDomain expectedDomain) {
        auto material = AssetRegistry::Instance().GetAsset<InxMaterial>(guid);
        if (!material || material->IsDeleted() || material->GetVersion() != generation)
            throw UIShaderMaterialError("UI material GUID is missing or its retained generation is stale: " + guid);
        if (!m_uiMaterialShaderValidator)
            throw std::runtime_error("UI material shader publication validator is unavailable for GUID " + guid);
        if (!m_uiMaterialShaderValidator(material, expectedDomain))
            return std::shared_ptr<const ShaderProgramArtifact>{}; // Ordinary UI material uses the built-in pipeline.
        return m_vkCore->ShareShaderProgramArtifact(material->GetShaderStages());
    });
}

void InxRenderer::EnsureScreenUIRenderer()
{
    if (m_screenUIRenderer)
        return;
    if (!m_vkCore || !m_sceneRenderTarget || !m_sceneRenderTarget->IsReady())
        throw std::runtime_error("Screen UI renderer requires the initialized Scene render target");

    auto renderer = std::make_unique<InxScreenUIRenderer>();
    if (!renderer->Initialize(m_vkCore->GetDevice(), m_vkCore->GetDeviceContext().GetVmaAllocator(),
                              m_vkCore->GetDeviceContext().GetRhiDevice().GetDescriptorManager(),
                              m_sceneRenderTarget->GetColorFormat(), m_sceneRenderTarget->GetDepthFormat(),
                              m_sceneRenderTarget->GetMsaaSampleCount(), m_vkCore->GetMaxFramesInFlight())) {
        throw std::runtime_error("Failed to initialize the shared Screen UI renderer");
    }
    renderer->SetRetirementQueue(&m_vkCore->GetRetirementQueue());
    renderer->SetTextureUsageValidator(
        [this](uint64_t textureId) { return m_gui && m_gui->TouchImGuiTextureId(textureId); });
    renderer->SetRenderTextureResolver(
        [this](uint64_t textureId) { return m_gui->ResolveImGuiRenderTexture(textureId); });
    renderer->SetTextureColorSpaceQuery(
        [this](uint64_t textureId) { return m_gui->ImGuiTextureNeedsDisplayEncoding(textureId); });
    ConfigureScreenUIMaterialResolver(*renderer);
    m_screenUIRenderer = std::move(renderer);
    if (m_sceneRenderGraph)
        m_sceneRenderGraph->SetScreenUIRenderer(m_screenUIRenderer.get());
}

void InxRenderer::SetGameCameraEnabled(bool enabled)
{
    if (m_gameCameraEnabled == enabled)
        return;
    m_gameCameraEnabled = enabled;
    // Panel visibility invalidates camera submissions, not graphics resources.
    // Scene/target replacement and particle registry revisions own those
    // invalidations. Rebuilding here reallocated every transient attachment
    // and particle view whenever the user returned to the Game tab.
    for (auto &[cameraId, graph] : m_gameRenderGraphs) {
        (void)cameraId;
        graph->ClearCachedViewSubmission();
    }
    INXLOG_DEBUG("Game camera rendering {}", enabled ? "enabled" : "disabled");
}

InxScreenUIRenderer *InxRenderer::GetScreenUIRenderer()
{
    return m_screenUIRenderer.get();
}

// ============================================================================
// MSAA Configuration
// ============================================================================

uint32_t InxRenderer::GetSupportedMsaaSampleMask() const
{
    if (!m_vkCore)
        return 0;
    const VkFormat colorFormat =
        m_sceneRenderTarget ? m_sceneRenderTarget->GetColorFormat() : VK_FORMAT_R16G16B16A16_SFLOAT;
    const VkFormat depthFormat =
        m_sceneRenderTarget ? m_sceneRenderTarget->GetDepthFormat() : m_vkCore->GetDeviceContext().FindDepthFormat();
    const auto &deviceContext = m_vkCore->GetDeviceContext();
    const auto colorSamples = deviceContext.GetImageSampleCountMask(colorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                                                                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    const auto resolveSamples = deviceContext.GetImageSampleCountMask(
        colorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    const auto depthSamples =
        deviceContext.GetImageSampleCountMask(depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    return GetSceneTargetSampleCountMask(colorSamples, resolveSamples, depthSamples);
}

void InxRenderer::SetMsaaSamples(int samples)
{
    if (!ApplyMsaaSamples(samples, "engine API"))
        SetEffectiveGraphMsaaSamples(GetMsaaSamples());
}

bool InxRenderer::ApplyMsaaSamples(int samples, const char *source)
{
    if (!IsValidMsaaSampleCount(samples)) {
        ++m_msaaRejectedRequestCount;
        INXLOG_ERROR("Rejected invalid ", source, " MSAA request ", samples, "x; valid values are 1, 2, 4, and 8.");
        return false;
    }

    const uint32_t supportedMask = GetSupportedMsaaSampleMask();
    if (!SupportsMsaaSampleCount(static_cast<rhi::SampleCountMask>(supportedMask), samples)) {
        ++m_msaaRejectedRequestCount;
        INXLOG_ERROR("Rejected unsupported ", source, " MSAA request ", samples,
                     "x for the HDR color/depth target pair (supported mask=", supportedMask,
                     "). The active setting remains unchanged.");
        return false;
    }

    const VkSampleCountFlagBits vkSamples = rhi::ToVkSampleCount(ToRhiSampleCount(samples));
    const bool sceneAligned = !m_sceneRenderTarget || m_sceneRenderTarget->GetMsaaSampleCount() == vkSamples;
    const bool gameAligned = !m_gameRenderTarget || m_gameRenderTarget->GetMsaaSampleCount() == vkSamples;
    const bool materialsAligned = !m_vkCore || m_vkCore->GetMaterialPipelineManager().GetSampleCount() == vkSamples;

    if (sceneAligned && gameAligned && materialsAligned)
        return false;

    std::unique_ptr<SceneRenderTarget> replacementSceneTarget;
    std::unique_ptr<SceneRenderTarget> replacementGameTarget;
    std::unique_ptr<InxScreenUIRenderer> replacementScreenUI;
    std::unique_ptr<OutlineRenderer> replacementOutline;

    auto createReplacement = [this, vkSamples](const SceneRenderTarget &source) {
        auto replacement = std::make_unique<SceneRenderTarget>(m_vkCore.get());
        replacement->SetMsaaSampleCount(vkSamples);
        if (!replacement->Initialize(source.GetWidth(), source.GetHeight(), source.GetFilterMode()))
            replacement.reset();
        return replacement;
    };

    if (!sceneAligned) {
        replacementSceneTarget = createReplacement(*m_sceneRenderTarget);
        if (!replacementSceneTarget) {
            INXLOG_ERROR("Failed to create the replacement Scene target for ", samples,
                         "x MSAA; keeping all current resources.");
            return false;
        }
    }

    if (!gameAligned) {
        replacementGameTarget = createReplacement(*m_gameRenderTarget);
        if (!replacementGameTarget) {
            INXLOG_ERROR("Failed to create the replacement Game target for ", samples,
                         "x MSAA; keeping all current resources.");
            return false;
        }
    }

    if (m_screenUIRenderer && (!sceneAligned || !gameAligned || !materialsAligned)) {
        replacementScreenUI = std::make_unique<InxScreenUIRenderer>();
        const SceneRenderTarget *screenUITarget =
            replacementSceneTarget ? replacementSceneTarget.get() : m_sceneRenderTarget.get();
        if (!replacementScreenUI->Initialize(m_vkCore->GetDevice(), m_vkCore->GetDeviceContext().GetVmaAllocator(),
                                             m_vkCore->GetDeviceContext().GetRhiDevice().GetDescriptorManager(),
                                             screenUITarget->GetColorFormat(), screenUITarget->GetDepthFormat(),
                                             vkSamples, m_vkCore->GetMaxFramesInFlight())) {
            INXLOG_ERROR("Failed to create the replacement Screen UI pipeline for ", samples,
                         "x MSAA; keeping all current resources.");
            return false;
        }
        replacementScreenUI->SetRetirementQueue(&m_vkCore->GetRetirementQueue());
        replacementScreenUI->SetTextureUsageValidator(
            [this](uint64_t textureId) { return m_gui && m_gui->TouchImGuiTextureId(textureId); });
        replacementScreenUI->SetRenderTextureResolver(
            [this](uint64_t textureId) { return m_gui->ResolveImGuiRenderTexture(textureId); });
        replacementScreenUI->SetTextureColorSpaceQuery(
            [this](uint64_t textureId) { return m_gui->ImGuiTextureNeedsDisplayEncoding(textureId); });
        ConfigureScreenUIMaterialResolver(*replacementScreenUI);
    }

    if (replacementSceneTarget && m_outlineRenderer) {
        replacementOutline = std::make_unique<OutlineRenderer>();
        if (m_outlineRenderer->IsReady() &&
            !replacementOutline->Initialize(m_vkCore.get(), replacementSceneTarget.get())) {
            INXLOG_ERROR("Failed to prepare the replacement Outline generation for ", samples,
                         "x MSAA; keeping all current resources.");
            return false;
        }
    }

    // No command buffer is submitted after this snapshot and before the
    // generation commit. Every old object therefore shares one cross-queue
    // retirement boundary, including asynchronous particle compute/export.
    const rhi::SubmissionSerial cutoverEpoch = m_vkCore->GetBackendContext().Queues().GetLastReservedCompletionEpoch();

    // Material pipelines are the only fallible part of the publication step.
    // Prepare and commit their complete generation before replacing any live
    // render target so a pipeline build failure leaves the old frame coherent.
    if (m_vkCore && !m_vkCore->CommitMaterialPipelineGeneration(vkSamples)) {
        ++m_msaaRejectedRequestCount;
        INXLOG_ERROR("Failed to prepare the material pipeline generation for ", samples,
                     "x MSAA; keeping the previous setting.");
        return false;
    }

    std::unique_ptr<SceneRenderTarget> retiredSceneTarget;
    std::unique_ptr<SceneRenderTarget> retiredGameTarget;
    std::unique_ptr<InxScreenUIRenderer> retiredScreenUI;
    std::unique_ptr<OutlineRenderer> retiredOutline;

    // Invalidate graphs before publishing replacement target views.
    if (replacementSceneTarget && m_sceneRenderGraph)
        m_sceneRenderGraph->InvalidateBeforeTargetReplacement();
    if (replacementGameTarget) {
        for (auto &[cameraId, graph] : m_gameRenderGraphs) {
            (void)cameraId;
            graph->InvalidateBeforeTargetReplacement();
        }
    }

    if (replacementSceneTarget) {
        retiredSceneTarget = std::move(m_sceneRenderTarget);
        m_sceneRenderTarget = std::move(replacementSceneTarget);
        if (m_sceneRenderGraph)
            m_sceneRenderGraph->ReplaceSceneTarget(m_sceneRenderTarget.get());
        ++m_sceneRenderTargetGeneration;
        if (m_captureService)
            m_captureService->InvalidateSource(CaptureSource::Scene, m_sceneRenderTargetGeneration);

        retiredOutline = std::move(m_outlineRenderer);
        m_outlineRenderer = std::move(replacementOutline);
    }
    if (replacementGameTarget) {
        retiredGameTarget = std::move(m_gameRenderTarget);
        m_gameRenderTarget = std::move(replacementGameTarget);
        for (auto &[cameraId, graph] : m_gameRenderGraphs) {
            (void)cameraId;
            graph->ReplaceSceneTarget(m_gameRenderTarget.get());
        }
        ++m_gameRenderTargetGeneration;
        if (m_captureService)
            m_captureService->InvalidateSource(CaptureSource::Game, m_gameRenderTargetGeneration);
    }
    if (replacementScreenUI) {
        retiredScreenUI = std::move(m_screenUIRenderer);
        m_screenUIRenderer = std::move(replacementScreenUI);
        if (m_sceneRenderGraph)
            m_sceneRenderGraph->SetScreenUIRenderer(m_screenUIRenderer.get());
        m_gameCameraCacheValid = false;
        (void)FindGameCamerasCached();
    }

    auto &retirementQueue = m_vkCore->GetRetirementQueue();
    if (retiredOutline) {
        std::shared_ptr<OutlineRenderer> retired(retiredOutline.release());
        retirementQueue.RetireAfter(cutoverEpoch, [retired = std::move(retired)] { retired->Cleanup(false); });
    }
    if (retiredScreenUI) {
        std::shared_ptr<InxScreenUIRenderer> retired(retiredScreenUI.release());
        retirementQueue.RetireAfter(cutoverEpoch, [retired = std::move(retired)] {});
    }
    if (retiredSceneTarget)
        m_replacedViewTargets.push_back(std::move(retiredSceneTarget));
    if (retiredGameTarget)
        m_replacedViewTargets.push_back(std::move(retiredGameTarget));

    // BuildFrame() ran before the MSAA request was observed. Replacing render
    // targets and preview renderers invalidates texture descriptor sets already
    // baked into that ImDrawData. Make the next editor frame rebuild the GUI
    // before it can submit those stale handles.
    if (m_gui)
        m_gui->RequestFrame();

    SetEffectiveGraphMsaaSamples(samples);
    ++m_msaaReconfigurationCount;
    INXLOG_INFO("Applied ", samples, "x MSAA from ", source,
                " to Scene, Game, material, preview, and Screen UI paths.");
    return true;
}

int InxRenderer::GetMsaaSamples() const
{
    if (m_sceneRenderTarget)
        return static_cast<int>(m_sceneRenderTarget->GetMsaaSampleCount());
    if (m_gameRenderTarget)
        return static_cast<int>(m_gameRenderTarget->GetMsaaSampleCount());
    if (m_vkCore)
        return static_cast<int>(m_vkCore->GetMaterialPipelineManager().GetSampleCount());
    return 4; // default
}

void InxRenderer::SetEffectiveGraphMsaaSamples(int samples)
{
    if (m_sceneRenderGraph)
        m_sceneRenderGraph->SetEffectiveMsaaSamples(samples);
    for (auto &[cameraId, graph] : m_gameRenderGraphs) {
        (void)cameraId;
        graph->SetEffectiveMsaaSamples(samples);
    }
}

MsaaStateSnapshot InxRenderer::GetMsaaStateSnapshot() const
{
    MsaaStateSnapshot snapshot;
    snapshot.activeSamples = GetMsaaSamples();
    snapshot.sceneRequestedSamples = m_sceneRequestedMsaaSamples;
    snapshot.gameRequestedSamples = m_gameRequestedMsaaSamples;
    snapshot.supportedSampleMask = GetSupportedMsaaSampleMask();
    snapshot.requestConflict = m_msaaRequestConflict;
    snapshot.sceneTargetAligned =
        !m_sceneRenderTarget || static_cast<int>(m_sceneRenderTarget->GetMsaaSampleCount()) == snapshot.activeSamples;
    snapshot.gameTargetAligned =
        !m_gameRenderTarget || static_cast<int>(m_gameRenderTarget->GetMsaaSampleCount()) == snapshot.activeSamples;
    snapshot.materialPipelinesAligned =
        !m_vkCore ||
        static_cast<int>(m_vkCore->GetMaterialPipelineManager().GetSampleCount()) == snapshot.activeSamples;
    snapshot.sceneMsaaColorBytes = m_sceneRenderTarget ? m_sceneRenderTarget->GetMsaaColorResidentBytes() : 0;
    snapshot.gameMsaaColorBytes = m_gameRenderTarget ? m_gameRenderTarget->GetMsaaColorResidentBytes() : 0;
    snapshot.reconfigurationCount = m_msaaReconfigurationCount;
    snapshot.rejectedRequestCount = m_msaaRejectedRequestCount;
    return snapshot;
}

void InxRenderer::SetPresentMode(int mode)
{
    if (m_vkCore)
        m_vkCore->SetPresentMode(mode);
}

int InxRenderer::GetPresentMode() const
{
    if (m_vkCore)
        return m_vkCore->GetPresentMode();
    return 1; // MAILBOX default
}

// ========================================================================
// Editor Power-Save / Idle Mode
// ========================================================================

void InxRenderer::SetEditorIdleEnabled(bool enabled)
{
    if (m_view)
        m_view->GetIdling().enableIdling = enabled;
}

bool InxRenderer::IsEditorIdleEnabled() const
{
    return m_view ? m_view->GetIdling().enableIdling : false;
}

void InxRenderer::SetEditorIdleFps(float fps)
{
    if (m_view)
        m_view->GetIdling().fpsIdle = fps;
}

float InxRenderer::GetEditorIdleFps() const
{
    return m_view ? m_view->GetIdling().fpsIdle : 0.0f;
}

bool InxRenderer::IsEditorIdling() const
{
    return m_view ? m_view->GetIdling().isIdling : false;
}

void InxRenderer::SetEditorFpsCap(float fps)
{
    if (m_view)
        m_view->GetIdling().editorFpsCap = fps;
}

float InxRenderer::GetEditorFpsCap() const
{
    return m_view ? m_view->GetIdling().editorFpsCap : 0.0f;
}

void InxRenderer::SetPlayFpsCap(float fps)
{
    if (m_view)
        m_view->GetIdling().playFpsCap = std::max(0.0f, fps);
}

float InxRenderer::GetPlayFpsCap() const
{
    return m_view ? m_view->GetIdling().playFpsCap : 0.0f;
}

void InxRenderer::SetPlayModeRendering(bool play)
{
    const bool changed = m_view && m_view->IsPlayMode() != play;
    if (m_view)
        m_view->SetPlayMode(play);
    if (changed) {
        InvalidateGpuViewStateForSceneBoundary();
        // Play can start inside this frame's GUI callback. Its loading work
        // must not become simulation time on the following frame.
        OverrideNextFrameDeltaTime(0.0f);
    }
}

bool InxRenderer::IsPlayModeRendering() const
{
    return m_view ? m_view->IsPlayMode() : false;
}

void InxRenderer::SetSelectedObjectId(uint64_t objectId)
{
    SetSelectionState(objectId, objectId == 0 ? std::vector<uint64_t>{} : std::vector<uint64_t>{objectId});
}

bool InxRenderer::ShouldDrawSelectionOutline() const
{
    // Rect already describes the selected object's bounds. Rendering the
    // ordinary mesh-selection outline at the same time produces a second,
    // strongly coloured border and obscures the Rect handles.
    return !m_editorTools || m_editorTools->GetToolMode() != EditorTools::ToolMode::Rect;
}

void InxRenderer::SetSelectionState(uint64_t primaryObjectId, const std::vector<uint64_t> &outlineObjectIds)
{
    if (m_selectedObjectId == primaryObjectId && m_selectedOutlineObjectIds == outlineObjectIds)
        return;

    m_selectedObjectId = primaryObjectId;
    m_selectedOutlineObjectIds = outlineObjectIds;

    // Selection changes are editor state publications, not ordinary scene
    // mutations. Publish them to the outline owner immediately so an idle or
    // throttled Scene view cannot keep a stale graph topology until another
    // operation (such as capture) happens to wake rendering.
    if (m_outlineRenderer) {
        if (ShouldDrawSelectionOutline())
            m_outlineRenderer->SetOutlineObjectIds(m_selectedOutlineObjectIds);
        else
            m_outlineRenderer->SetOutlineObjectIds({});
    }
    if (m_sceneRenderGraph)
        m_sceneRenderGraph->SetOutlineRenderer(m_sceneViewVisible ? m_outlineRenderer.get() : nullptr);

    RequestFullSpeedFrame();
}

void InxRenderer::RequestFullSpeedFrame()
{
    if (m_view)
        m_view->RequestFullSpeedFrame();
}

void InxRenderer::RequestExternalWake()
{
    if (m_view)
        m_view->RequestExternalWake();
}

} // namespace infernux
