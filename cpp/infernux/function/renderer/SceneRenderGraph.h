/**
 * @file SceneRenderGraph.h
 * @brief RenderGraph-based scene rendering system
 *
 * This class fully integrates with the low-level vk::RenderGraph API for
 * declarative, frame-graph-driven rendering. All rendering is now handled
 * via RenderGraph passes - no more imperative BeginRenderPass/EndRenderPass.
 *
 * Architecture:
 * - Uses vk::RenderGraph for automatic resource management and barrier handling
 * - All passes are defined via RenderGraph's AddPass API
 * - Transient resources managed by RenderGraph
 * - External resources (scene target) imported into RenderGraph
 * - Supports GPU->CPU readback for Python/ML integration
 */

#pragma once
#include "RenderViewSchedule.h"
#include <core/config/EngineConfig.h>
#include <function/scene/NativeLifetime.h>

#include "FullscreenRenderer.h"
#include "InxRenderStruct.h"
#include "MaterialPassPipeline.h"
#include "RenderGraphDescription.h"
#include "RenderInstanceHistory.h"
#include "RendererList.h"
#include "SceneDepthResolver.h"
#include "lighting/CanonicalLightGpuBuffer.h"
#include "lighting/ForwardPlusLightGrid.h"
#include "particle/ParticleGpuViewDiagnostics.h"
#include "rhi/CameraViewportTarget.h"
#include "rhi/RhiComputeBuffer.h"
#include "rhi/RhiRenderTexture.h"
#include "vk/RenderGraph.h"
#include "vk/ViewGlobalsBinding.h"
#include "vk/VkDescriptorManager.h"
#include "vk/VkDeviceContext.h"
#include "vk/VkPipelineManager.h"
#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace infernux
{
/// A capture of one submitted graph execution, not a mutable latest-value slot.
struct GraphBufferCapture
{
    enum class Status
    {
        Pending,
        Accepted,
        Cancelled
    };
    Status status = Status::Pending;
    std::string error;
    std::vector<uint64_t> shape;
    rhi::ComputeScalarType scalarType = rhi::ComputeScalarType::Float32;
    uint64_t byteOffset = 0;
    uint64_t storageByteSize = 0;
    std::string storageClaim;
    std::shared_ptr<rhi::ComputeBuffer> storageBuffer;
    rhi::RenderViewId view = rhi::InvalidRenderViewId;
    uint64_t generation = 0, occurrence = 0;
    std::string name;
    std::shared_ptr<rhi::ComputeBuffer> value;
    bool selected = false;
    void Cancel(const std::string &reason)
    {
        if (status == Status::Pending) {
            status = Status::Cancelled;
            error = reason;
            storageBuffer.reset();
        }
    }
};

class GraphFramePublication
{
  public:
    struct Write
    {
        std::shared_ptr<rhi::ComputeBufferAllocation> allocation;
        intervals::Set initialized, written;
    };
    struct Capture
    {
        std::shared_ptr<GraphBufferCapture> request;
        std::shared_ptr<rhi::ComputeBuffer> value;
    };
    std::vector<Write> writes;
    std::vector<Capture> captures;
    GraphEventQueue events;
    ~GraphFramePublication();
    void Accept(rhi::SubmissionTicket completion, GraphEventQueue &acceptedEvents) noexcept;
};

class InxVkCoreModular;
class InxMaterial;
class InxScreenUIRenderer;
class OutlineRenderer;
class SceneRenderTarget;
class Camera;
class Scene;
namespace rhi
{
class TextureGpuView;
class ComputeWorkInstance;
} // namespace rhi
namespace particle
{
class ParticleGpuDrawRegistry;
class ParticleGpuCuller;
class ParticleGpuSorter;
class ParticleGpuOutputRenderer;
} // namespace particle

// Forward-declare from Camera.h
enum class CameraClearFlags;

/**
 * @brief Pass render callback signature using RenderGraph context
 * @param ctx RenderGraph context for drawing commands
 * @param width Render target width
 * @param height Render target height
 */
using ScenePassRenderCallback = std::function<void(vk::RenderContext &ctx, uint32_t width, uint32_t height)>;

/**
 * @brief RenderGraph-based scene rendering system
 *
 * Provides a fully declarative rendering pipeline using vk::RenderGraph.
 * All rendering is handled through RenderGraph passes with automatic
 * resource management and barrier handling.
 */
class SceneRenderGraph : public NativeLifetimeOwner
{
  public:
    SceneRenderGraph();
    ~SceneRenderGraph();

    // Non-copyable
    SceneRenderGraph(const SceneRenderGraph &) = delete;
    SceneRenderGraph &operator=(const SceneRenderGraph &) = delete;

    /**
     * @brief Initialize logical view identity; GPU view resources are prepared on first use
     * @param vkCore Vulkan core for resource access
     * @param sceneTarget Scene render target for external resources
     * @return true if successful
     */
    bool Initialize(InxVkCoreModular *vkCore, SceneRenderTarget *sceneTarget,
                    rhi::RenderViewKind viewKind = rhi::RenderViewKind::Scene,
                    std::shared_ptr<rhi::RenderTexture> output = {});

    /// Called after Initialize at the renderer's idle scene boundary. Only
    /// physical transient images move; all previous view/scene state is retired.
    void TakeReusableImagesFrom(SceneRenderGraph &previous)
    {
        m_renderGraph->TakeReusableImagesFrom(*previous.m_renderGraph);
    }

    [[nodiscard]] uint64_t GetResourceAccessRevision() const
    {
        return m_resourceAccessRevision;
    }
    [[nodiscard]] RenderViewAccess GetResourceAccess() const;
    [[nodiscard]] bool HasPendingBufferInputs(bool forStep = false) const;

    /**
     * @brief Cleanup resources
     */
    void Destroy();

    [[nodiscard]] uint64_t GetTransientResidentBytes() const;

    /// Invalidate the compiled graph before replacing its imported target views.
    void InvalidateBeforeTargetReplacement();

    [[nodiscard]] const rhi::RenderViewContext &GetRenderViewContext() const noexcept
    {
        return m_renderView;
    }

    [[nodiscard]] size_t GetTemporalHistoryCount() const noexcept
    {
        return m_temporalHistories.size();
    }

    [[nodiscard]] size_t GetValidTemporalHistoryCount() const noexcept
    {
        return static_cast<size_t>(std::count_if(m_temporalHistories.begin(), m_temporalHistories.end(),
                                                 [](const auto &entry) { return entry.second.valid; }));
    }

    [[nodiscard]] uint32_t GetTemporalSampleIndex() const noexcept
    {
        return m_temporalSampleIndex;
    }

    // ========================================================================
    // RenderGraph topology defined from Python
    // ========================================================================

    /**
     * @brief Apply a render graph topology defined from Python
     *
     * Receives a RenderGraphDescription from Python and translates it into
     * SceneRenderGraph passes with appropriate callbacks. C++ retains
     * compilation authority (DAG compilation, barrier insertion, resource
     * allocation) while Python has definition authority (topology, pass
     * order, resource connections).
     *
     * @param desc The graph topology description from Python
     * @return Whether the description was accepted, including an unchanged replay.
     */
    bool ApplyPythonGraph(const RenderGraphDescription &desc);

    /// Upload changed runtime parameter blocks without rebuilding graph topology.
    void UpdateParameterBlocks(const std::vector<GraphParameterBlockUpdate> &updates);

    /// Publish the current pass-local values for the pending view submission.
    /// Graph execution consumes this immutable-by-convention snapshot, so an
    /// update recorded after SubmitCulling cannot rewrite already-submitted
    /// draw payloads. The map is copied only when its publication changes.
    void CaptureParameterBlocksForSubmission();

    /// True when this graph already owns the requested Python artifact and its
    /// callback sample contract still matches the active render targets.
    [[nodiscard]] bool IsPythonGraphCurrent(uint64_t sourceRevision) const;

    /**
     * @brief Set the screen UI renderer for DrawScreenUI passes
     * @param renderer Pointer to the screen UI renderer (may be nullptr)
     */
    void SetScreenUIRenderer(InxScreenUIRenderer *renderer, bool screenOutput = true)
    {
        if (m_screenUIOutput != screenOutput && m_viewportGeneration)
            m_needsRebuild = true;
        m_screenUIRenderer = renderer;
        m_screenUIOutput = screenOutput;
    }

    /// Attach the Scene-only editor outline renderer. The graph topology only
    /// changes when outline rendering transitions between inactive and active.
    void SetOutlineRenderer(OutlineRenderer *renderer);

    void SetParticleGpuDrawRegistry(particle::ParticleGpuDrawRegistry *registry)
    {
        if (m_particleDrawRegistry == registry)
            return;
        m_particleDrawRegistry = registry;
        m_particleDrawRegistryRevision = 0;
        m_needsRebuild = true;
    }

    [[nodiscard]] uint64_t RequestParticleViewDiagnostics(uint64_t graphInstanceId);
    [[nodiscard]] particle::GpuParticleViewDiagnosticSnapshot QueryParticleViewDiagnostics(uint64_t requestId) const;

    /**
     * @brief Check if a custom graph topology has been applied
     */
    [[nodiscard]] bool HasPythonGraph() const
    {
        return m_hasPythonGraph;
    }

    /**
     * @brief Get the MSAA sample count requested by the current graph (0 = no preference).
     */
    [[nodiscard]] int GetRequestedMsaaSamples() const
    {
        return m_pythonGraphValid ? m_pythonGraphDesc.msaaSamples : 0;
    }

    /// Return the square shadow atlas extent declared by this graph, or zero
    /// when the graph has no shadow-caster pass.
    [[nodiscard]] uint32_t GetShadowMapResolution() const;

    /// Set the globally validated sample count used by this view. Scene and
    /// game graphs currently share material pipelines and therefore must use
    /// one compatible sample count.
    void SetEffectiveMsaaSamples(int samples)
    {
        m_effectiveMsaaSamples = HasOutputTexture() ? static_cast<int>(m_renderView.samples) : samples;
    }

    void SetOutputTexture(std::shared_ptr<rhi::RenderTexture> output);
    bool SetCameraViewport(const Camera &camera);
    [[nodiscard]] bool HasOutputTexture() const noexcept
    {
        return m_outputTexture != nullptr;
    }
    [[nodiscard]] SceneRenderTarget *GetOutputTarget() const noexcept
    {
        return m_outputTarget && m_outputTexture ? m_outputTarget.get() : m_screenTarget;
    }

    /// Replace the external target without reinitializing graph-owned pools.
    /// The next EnsureGraphBuilt call imports the replacement resources.
    void ReplaceSceneTarget(SceneRenderTarget *sceneTarget);

    // ========================================================================
    // Resource management
    // ========================================================================

    /**
     * @brief Get the scene color target resource handle
     * @return Handle to the imported scene color target
     */
    [[nodiscard]] vk::ResourceHandle GetSceneColorTarget() const
    {
        return m_importedColorTarget;
    }

    /**
     * @brief Get the scene depth target resource handle
     * @return Handle to the imported scene depth target
     */
    [[nodiscard]] vk::ResourceHandle GetSceneDepthTarget() const
    {
        return m_importedDepthTarget;
    }

    // ========================================================================
    // Execution (Pure RenderGraph)
    // ========================================================================

    /// Prepare mutable per-frame state before the backend records the
    /// compiler-produced submission batches on their native queues.
    [[nodiscard]] bool PrepareSubmissionExecution();

    /// Record diagnostics after the view's batches.
    [[nodiscard]] bool CompleteSubmissionExecution(VkCommandBuffer commandBuffer);
    /// Advance the view's history after the frame submission is accepted.
    void CommitSubmissionExecution();

    [[nodiscard]] vk::RenderGraph *GetCompiledRenderGraph() noexcept
    {
        return m_graphBuilt ? m_renderGraph.get() : nullptr;
    }

    [[nodiscard]] const vk::RenderGraph *GetCompiledRenderGraph() const noexcept
    {
        return m_graphBuilt ? m_renderGraph.get() : nullptr;
    }

    /**
     * @brief Rebuild and compile the render graph if needed (pre-record phase).
     *
     * Must be called BEFORE command buffer recording starts.  Moves
     * BuildRenderGraph / Compile out of the recording path so that graph-owned
     * descriptors and transient resources are never recreated while their old
     * generation is bound to an in-recording command buffer.
     */
    void EnsureGraphBuilt();

    /// @brief Diagnostic: whether the graph is currently built and ready to execute
    [[nodiscard]] bool IsGraphBuilt() const
    {
        return m_graphBuilt && !m_graphStorageDeferred;
    }

    [[nodiscard]] bool IsGraphStorageDeferred() const noexcept
    {
        return m_graphStorageDeferred;
    }
    [[nodiscard]] bool AdmitGraphEvents(const GraphEventInterests &interests);

    /// @brief Diagnostic: whether the graph needs a rebuild before next execute
    [[nodiscard]] bool NeedsRebuild() const
    {
        return m_needsRebuild;
    }

    /**
     * @brief Called when scene render target is resized
     */
    /**
     * @brief Force rebuild of the render graph on next frame
     */
    void MarkDirty()
    {
        m_needsRebuild = true;
    }

    /// Drop cached fullscreen pipelines compiled from a replaced shader
    /// module. Graph commands keep their stable shader identifier and resolve
    /// the newly published pipeline on the next record.
    void InvalidateFullscreenShader(const std::string &shaderName)
    {
        m_fullscreenRenderer.InvalidateShader(shaderName);
        // Resource types (Texture2D vs Texture2DMS) determine resolve topology.
        // Re-publish that contract together with a hot-edited shader, not just
        // its pipeline against the old descriptor/image graph.
        for (const auto &pass : m_pythonGraphDesc.passes) {
            for (const auto &command : pass.commands) {
                if (command.type == GraphCommandType::FullscreenQuad && command.shaderName == shaderName) {
                    InvalidateTemporalHistory();
                    m_needsRebuild = true;
                    return;
                }
            }
        }
    }

    void UpdateMainPassClearSettings(CameraClearFlags clearFlags, const glm::vec4 &bgColor, bool dithering,
                                     bool stopNaNs);

    // ========================================================================
    // ========================================================================
    // Debug
    // ========================================================================

    /**
     * @brief Get debug visualization of the render graph
     */
    [[nodiscard]] std::string GetDebugString() const;

    /**
     * @brief Get pass count
     */
    [[nodiscard]] size_t GetPassCount() const
    {
        return m_hasPythonGraph ? m_pythonGraphDesc.passes.size() : 0;
    }

    [[nodiscard]] const std::string &GetGraphName() const
    {
        return m_pythonGraphDesc.name;
    }

    [[nodiscard]] uint64_t GetExecutionCount() const
    {
        return m_executionCount;
    }

    [[nodiscard]] bool HasExecutedCurrentGraph() const
    {
        return m_graphBuilt && m_executionCount > 0 && m_lastExecutedBuildRevision == m_graphBuildRevision;
    }

    [[nodiscard]] std::vector<std::string> GetExecutedPassNames() const
    {
        return HasExecutedCurrentGraph() && m_renderGraph ? m_renderGraph->GetExecutionPassNames()
                                                          : std::vector<std::string>{};
    }

    // ========================================================================
    // Per-Graph Draw Call Cache (for multi-camera rendering)
    // ========================================================================

    /// @brief Cache the submitted renderer list for this graph.
    void SetCachedRendererList(RendererList rendererList)
    {
        m_cachedRenderers = std::move(rendererList);
        m_hasCachedDrawCalls = true;
    }

    struct MaterialTextureRead
    {
        std::string passName;
        std::shared_ptr<rhi::RenderTexture> texture;
        std::shared_ptr<const rhi::RenderTextureGeneration> generation;
        bool operator==(const MaterialTextureRead &other) const
        {
            return passName == other.passName && texture == other.texture && generation == other.generation;
        }
    };

    struct MaterialBufferRead
    {
        std::string passName;
        std::shared_ptr<rhi::ComputeBuffer> buffer;
        bool operator==(const MaterialBufferRead &other) const
        {
            return passName == other.passName && buffer == other.buffer;
        }
    };

    /// Actual producers consumed by live passes; serials from distinct queues
    /// are separate dependencies, never a device-wide ordering counter.
    [[nodiscard]] std::vector<rhi::SubmissionDependency> GetBufferReadDependencies() const;
    [[nodiscard]] const rhi::ComputeBufferBindings &GetComputeBufferBindings() const noexcept
    {
        return m_computeBufferReads;
    }
    [[nodiscard]] std::shared_ptr<void> AcquireComputeReadValues() const;
    [[nodiscard]] std::shared_ptr<GraphBufferCapture> CaptureBuffer(const std::string &name);
    [[nodiscard]] std::shared_ptr<GraphFramePublication> PreparePublication(const GraphEventInterests &interests);

    /// Classify an implicit material/UI sample against explicit same-graph
    /// writers. Sampling before a local producer or in its writing pass is
    /// invalid; false means the frame schedule must supply an external producer.
    static bool HasLocalTextureProducer(const RenderGraphDescription &description, const MaterialTextureRead &read);

    [[nodiscard]] bool CanReuseCachedSubmission(uint64_t signature, uint64_t objectBufferRevision) const noexcept
    {
        return m_hasCachedDrawCalls && signature != 0 && HasReadyObjectBindings(objectBufferRevision) &&
               m_cachedSubmissionSignature == signature &&
               m_cachedParticleDrawRegistryRevision == CurrentParticleDrawRegistryRevision();
    }

    [[nodiscard]] bool HasReadyObjectBindings(uint64_t revision) const noexcept
    {
        return revision != 0 && m_cachedObjectBufferRevision == revision;
    }

    void SetCachedSubmissionSignature(uint64_t signature, std::shared_ptr<const void> owner,
                                      uint64_t readyObjectBufferRevision)
    {
        m_cachedSubmissionSignature = signature;
        m_cachedRenderWorldOwner = std::move(owner);
        m_cachedObjectBufferRevision = readyObjectBufferRevision;
        m_cachedParticleDrawRegistryRevision = CurrentParticleDrawRegistryRevision();
    }

    /// @brief Cache owned or borrowed shadow-caster candidates for this graph.
    void SetCachedShadowRendererList(RendererList rendererList)
    {
        m_hasCachedShadowDrawCalls = !rendererList.Empty();
        m_cachedShadowRenderers = std::move(rendererList);
    }

    /// @brief Clear cached shadow-caster candidates.
    void ClearCachedShadowDrawCalls()
    {
        m_cachedShadowRenderers.Clear();
        m_hasCachedShadowDrawCalls = false;
        m_cachedSubmissionSignature = 0;
        m_cachedObjectBufferRevision = 0;
        m_cachedParticleDrawRegistryRevision = 0;
        m_cachedRenderWorldOwner.reset();
    }

    /// @brief Get cached draw calls
    [[nodiscard]] const std::vector<DrawCall> &GetCachedDrawCalls() const
    {
        return m_cachedRenderers.DrawCalls();
    }

    [[nodiscard]] const RendererList &GetCachedRendererList() const
    {
        return m_cachedRenderers;
    }

    /// @brief Check if this graph has cached draw calls
    [[nodiscard]] bool HasCachedDrawCalls() const
    {
        return m_hasCachedDrawCalls;
    }

    /// @brief Get cached shadow draw calls.
    [[nodiscard]] const std::vector<DrawCall> &GetCachedShadowDrawCalls() const
    {
        return m_cachedShadowRenderers.DrawCalls();
    }

    [[nodiscard]] const RendererList &GetCachedShadowRendererList() const
    {
        return m_cachedShadowRenderers;
    }

    /// @brief Check if this graph has cached shadow draw calls.
    [[nodiscard]] bool HasCachedShadowDrawCalls() const
    {
        return m_hasCachedShadowDrawCalls;
    }

    /// @brief True when the current Python graph contains a shadow-caster pass.
    [[nodiscard]] bool HasShadowCasterPass() const
    {
        return m_hasShadowCasterPass;
    }

    // ========================================================================
    // Per-Graph Camera VP Cache (for multi-camera UBO updates)
    // ========================================================================

    /// @brief Cache camera VP matrices (called by SubmitCulling)
    void SetCachedCameraVP(const Camera *camera, const glm::mat4 &view, const glm::mat4 &proj);
    /// World UI billboards follow the active Game Camera even in Scene View.
    /// Scene geometry and editor picking continue to use the cached view.
    void SetBillboardViewMatrix(const glm::mat4 &view)
    {
        m_billboardView = view;
        m_hasBillboardView = true;
    }
    /// Published with SetupCameraProperties, before graph revision checks.
    void SetCameraInvertCulling(bool invert);

    /// Return the centered sub-pixel projection offset for one temporal sample.
    /// The result is expressed in NDC and is independent for each RenderView.
    [[nodiscard]] static glm::vec2 ComputeTemporalJitterNdc(uint32_t sampleIndex, uint32_t width, uint32_t height);

    /// Apply an NDC offset to a projection without assuming perspective or
    /// orthographic matrix layout. Culling continues to use the source matrix.
    [[nodiscard]] static glm::mat4 ApplyTemporalJitter(const glm::mat4 &projection, const glm::vec2 &jitterNdc);
    [[nodiscard]] static glm::mat4 ProjectionForPass(const glm::mat4 &projection, const glm::vec2 &jitterNdc,
                                                     bool editorOverlay);

    /// A caster may be frozen in the atlas only when its authoring contract
    /// explicitly declares it static and it has no frame-varying skin pose.
    [[nodiscard]] static bool ShadowCasterRequiresContinuousUpdate(const DrawCall &drawCall) noexcept
    {
        return !drawCall.isStatic || drawCall.skinBoneMatrices != nullptr;
    }

    /// Upload this view's camera constants into its current frame-local buffer.
    /// This is host-side preparation and must happen before graph recording.
    bool StageCameraMatrices(const glm::mat4 &view, const glm::mat4 &proj, const glm::mat4 *previousViewProj = nullptr);

    /// @brief Check if this graph has cached camera VP matrices
    [[nodiscard]] bool HasCachedCameraVP() const
    {
        return m_hasCachedCameraVP;
    }

    /// @brief Drop only camera-dependent submissions for an inactive view.
    ///
    /// Hiding an editor panel must not invalidate the compiled pipeline or its
    /// resource description.  The Game view still relies on that shared
    /// pipeline description when it builds the per-camera shadow layout.
    void ClearCachedViewSubmission()
    {
        m_cachedRenderers.Clear();
        m_hasCachedDrawCalls = false;
        m_cachedShadowRenderers.Clear();
        m_hasCachedShadowDrawCalls = false;
        m_cachedSubmissionSignature = 0;
        m_cachedObjectBufferRevision = 0;
        m_cachedParticleDrawRegistryRevision = 0;
        m_cachedRenderWorldOwner.reset();
        m_cachedView = glm::mat4(1.0f);
        m_cachedProj = glm::mat4(1.0f);
        m_cachedUnjitteredProj = glm::mat4(1.0f);
        m_hasCachedCameraVP = false;
        m_cachedCamera = nullptr;
        m_requestedDraw = 0;
        InvalidateTemporalHistory();
    }

    void RequestView(uint64_t draw, uint64_t objectBindingIdentity, bool reusesObjectBindings) noexcept
    {
        m_requestedDraw = draw;
        m_objectBindingIdentity = objectBindingIdentity;
        m_reusesObjectBindings = reusesObjectBindings;
    }
    [[nodiscard]] bool IsViewRequested(uint64_t draw) const noexcept
    {
        return m_requestedDraw == draw;
    }
    [[nodiscard]] uint64_t GetObjectBindingIdentity() const noexcept
    {
        return m_objectBindingIdentity;
    }
    [[nodiscard]] bool ReusesObjectBindings() const noexcept
    {
        return m_reusesObjectBindings;
    }

    /// Retire per-view particle cullers/sorters and forbid executing a compiled
    /// graph that still imports those retired buffers. Vulkan may recycle the
    /// same raw handle values, so handle equality cannot keep these views.
    void InvalidateParticleViews();

    /// @brief Clear cached draw calls and camera state.
    ///
    /// Scene switching destroys the source meshes/components immediately.
    /// Any cross-frame cache that still references those draw calls becomes
    /// invalid and must be dropped before the next render submission.
    /// Also forces a full render graph rebuild so that stale transient
    /// VkImage handles are not used in barrier insertion.
    void ClearCachedFrameState()
    {
        CancelBufferCaptures("render view scene was reset");
        ++m_bufferGeneration;
        ClearCachedViewSubmission();
        InvalidateParticleViews();
        RetireWorkInstances();
        InvalidatePerViewShadowBindings();
        m_needsRebuild = true;
        // Prevent Execute() from running the old compiled graph before
        // EnsureGraphBuilt() has a chance to rebuild it.  Without this,
        // an early-return path in EnsureGraphBuilt (e.g. MSAA mismatch
        // guard) could leave m_graphBuilt = true while the graph still
        // references stale VkImage handles from the previous scene.
        m_graphBuilt = false;
        // Clear the stale Python graph descriptor so that
        // GetRequestedMsaaSamples() returns 0 until the new scene's
        // pipeline calls ApplyPythonGraph().  This avoids the MSAA
        // mismatch guard firing on an outdated descriptor.
        m_hasPythonGraph = false;
        m_pythonGraphValid = false;
        m_hasShadowCasterPass = false;
        m_parameterBlocks.clear();
        m_submittedParameterBlocks.clear();
        ++m_parameterBlockGeneration;
        m_submittedParameterBlockGeneration = m_parameterBlockGeneration;
        m_pythonMaterialPasses.clear();
        m_materialTextureReads.clear();
        m_materialBufferReads.clear();
        m_liveBufferInputs.clear();
        m_pythonGraphDesc = {};
        ++m_resourceAccessRevision;
    }

    /// @brief Get cached view matrix
    [[nodiscard]] const glm::mat4 &GetCachedView() const
    {
        return m_cachedView;
    }

    void SetDrawViewMatrix(const glm::mat4 &view)
    {
        m_drawView = view;
    }

    /// @brief Get cached projection matrix
    [[nodiscard]] const glm::mat4 &GetCachedProj() const
    {
        return m_cachedProj;
    }

    [[nodiscard]] glm::mat4 GetPreviousViewProj() const
    {
        return m_cameraHistoryValid ? m_previousViewProj : m_cachedProj * m_cachedView;
    }

    void CommitCameraHistory()
    {
        if (!m_hasCachedCameraVP)
            return;
        m_previousViewProj = m_cachedProj * m_cachedView;
        m_cameraHistoryValid = true;
        if (m_pythonGraphValid && m_pythonGraphDesc.temporalJitter)
            m_temporalSampleIndex = (m_temporalSampleIndex + 1u) % kTemporalJitterSampleCount;
    }

    /// Drop accumulated temporal color history for this view.
    void InvalidateTemporalHistory();

    /// @brief Get per-graph shadow descriptor set (set 1) for the current frame-in-flight
    [[nodiscard]] VkDescriptorSet GetPerViewDescriptorSet() const;
    [[nodiscard]] rhi::BindGroupHandle GetPerViewBindGroup() const;
    [[nodiscard]] vk::ViewGlobalsBinding &GetViewGlobalsBinding();
    [[nodiscard]] rhi::BindGroupHandle GetEditorOverlayBindGroup() const;

    /// Build immutable lighting and shadow state owned by this camera graph.
    /// Stage the existing CPU collector; EnsureGraphBuilt publishes its GPU data.
    void StageCameraLighting(Scene *scene, Camera *camera, const glm::vec3 &cameraPosition,
                             const ShaderLightingUBO &environmentLighting);
    [[nodiscard]] bool HasCameraShadows() const noexcept
    {
        return !m_cameraLightCollector.GetShadowFrame().views.empty();
    }
    [[nodiscard]] uint32_t GetCameraLightCount() const noexcept
    {
        return m_cameraLightCollector.GetTotalLightCount();
    }
    [[nodiscard]] uint32_t GetCameraShadowViewCount() const noexcept
    {
        return static_cast<uint32_t>(m_cameraLightCollector.GetShadowFrame().views.size());
    }
    [[nodiscard]] uint32_t GetCameraShadowAssignmentCount() const noexcept
    {
        return static_cast<uint32_t>(m_cameraLightCollector.GetShadowFrame().assignments.size());
    }
    [[nodiscard]] uint64_t GetShadowResourceIdentity() const noexcept
    {
        return m_shadowCameraResourceId;
    }

  private:
    struct PendingParticleViewDiagnostic
    {
        uint64_t requestId = 0;
        uint64_t graphInstanceId = 0;
    };

    struct ParticleViewDiagnosticState
    {
        mutable std::mutex mutex;
        std::unordered_map<uint64_t, particle::GpuParticleViewDiagnosticSnapshot> snapshots;
    };

    struct GraphAssetTexture
    {
        const GraphTextureDesc *descriptor;
        std::shared_ptr<const rhi::TextureGpuView> publication;
        uint64_t contentGeneration;
    };
    void BuildRenderGraph(bool buffersPrepared, const std::vector<GraphAssetTexture> &assetTextures);
    void CacheLiveAuthorPasses();
    [[nodiscard]] bool EnsureViewGpuResources();
    void ReleaseViewGpuResources() noexcept;
    [[nodiscard]] bool UploadCameraLighting();
    void RefreshMaterialTextureReads();

    /// Register graph texture allocations and publish the imported bindings before adding the author passes.
    bool RegisterGraphTextures(uint32_t width, uint32_t height,
                               const std::vector<GraphAssetTexture> &assetTextures,
                               std::unordered_map<std::string, vk::ResourceHandle> &customRTHandles);
    void RetireImportedTextureAssets();
    void RetireWorkInstances();
    bool ProveGraphBufferContents();
    bool PrepareGraphBuffers();
    [[nodiscard]] bool CanRetainEventBuffers(const GraphEventInterests &interests) const;
    void PublishGraphBuffers();
    void CancelBufferCaptures(const std::string &reason);
    [[nodiscard]] const GraphBufferDesc &FindBufferExport(const std::string &name) const;
    [[nodiscard]] std::shared_ptr<GraphBufferCapture> MakeBufferCapture(const GraphBufferDesc &buffer) const;
    [[nodiscard]] rhi::ComputeBufferBindings CaptureComputeBufferReads() const;
    bool RefreshComputeBufferBindings(bool &prepared);
    void RetireComputeBufferBindings();
    void DeclareMaterialBufferReads(vk::PassBuilder &builder, const std::string &passName);
    bool ValidateOutputTextureAliases(const RenderGraphDescription &description) const;
    std::optional<GraphPassDesc> ResolveCameraClearOverride() const;
    std::shared_ptr<rhi::ComputeBuffer> ResolveGraphBufferOwner(const GraphBufferDesc &buffer) const;

    [[nodiscard]] MaterialPassPipelineDescriptor GetEditorOverlayMaterialPass() const;

    /**
     * @brief Append a system auto-pass (gizmos / editor tools) that draws
     * into the backbuffer with read-only depth testing.
     */
    vk::ResourceHandle AppendAutoPass(const std::string &name, vk::ResourceHandle colorTarget,
                                      vk::ResourceHandle depthTarget, uint32_t width, uint32_t height);

    /**
     * @brief Set the graph output handle for dead-pass culling.
     */
    void FinalizeGraphOutput(const std::unordered_map<std::string, vk::ResourceHandle> &customRTHandles);
    vk::ResourceHandle CopyViewportRegion(const std::string &name, vk::ResourceHandle source,
                                          vk::ResourceHandle destination, rhi::TextureAspect aspect, bool restore);
    void ImportViewportDestination();
    vk::ResourceHandle AppendEditorOutline(vk::ResourceHandle displayTarget);

    /**
     * @brief Import scene target resources into RenderGraph
     */
    void ImportSceneTargetResources();
    void ImportTemporalHistoryResources(std::unordered_map<std::string, vk::ResourceHandle> &handles);
    void BindTemporalHistoryResources();
    void CommitTemporalHistory();
    void UpdateTemporalProjection();

    /// @brief Update this frame's per-view shadow descriptor before recording.
    void RefreshPerViewShadowDescriptor();
    void InvalidatePerViewShadowBindings() noexcept;
    [[nodiscard]] bool UsesForwardPlus() const;
    void RefreshForwardPlusParticleRequirement();
    [[nodiscard]] bool PrepareForwardPlusFrame();
    void RetireForwardPlusResources();
    [[nodiscard]] uint64_t ComputeShadowContentSignature(bool &hasDynamicCaster) const;
    void RefreshShadowAtlasUpdateState();
    void CommitShadowAtlasUpdate();
    void RecordParticleViewDiagnostics(VkCommandBuffer commandBuffer);
    [[nodiscard]] uint64_t CurrentParticleDrawRegistryRevision() const noexcept;

    InxVkCoreModular *m_vkCore = nullptr;
    SceneRenderTarget *m_sceneTarget = nullptr;
    SceneRenderTarget *m_screenTarget = nullptr;
    std::shared_ptr<rhi::RenderTexture> m_outputTexture;
    std::shared_ptr<const rhi::RenderTextureGeneration> m_outputGeneration;
    std::unique_ptr<SceneRenderTarget> m_outputTarget;
    rhi::CameraViewportTarget m_viewport;
    std::shared_ptr<const rhi::RenderTextureGeneration> m_viewportGeneration;
    std::unique_ptr<SceneRenderTarget> m_viewportWorkTarget;
    vk::ResourceHandle m_viewportOutputColor, m_viewportOutputRaster, m_viewportOutputDepth;
    bool m_viewportColorPublished = false;
    rhi::RenderViewContext m_renderView;
    InxScreenUIRenderer *m_screenUIRenderer = nullptr;
    bool m_screenUIOutput = true;
    OutlineRenderer *m_outlineRenderer = nullptr;
    bool m_outlinePassesEnabled = false;
    bool m_outlinePipelineFailureReported = false;

    // Build state
    bool m_needsRebuild = true;
    uint64_t m_worldUIDepthRunSignature = 0;
    bool m_needsCompile = true;
    bool m_graphBuilt = false;
    bool m_graphStorageDeferred = false;
    bool m_graphBuffersPublished = false;
    bool m_viewGpuResourcesReady = false;
    int m_effectiveMsaaSamples = 0;
    bool m_hasPythonGraph = false;
    bool m_pythonGraphValid = false;
    uint64_t m_pythonGraphSourceRevision = 0;
    rhi::GraphicsRenderingSignature m_pythonCallbackTarget;
    bool m_cameraInvertCulling = false;
    bool m_pythonCallbackInvertCulling = false;
    uint64_t m_graphBuildRevision = 0;
    uint64_t m_lastExecutedBuildRevision = 0;
    uint64_t m_executionCount = 0;

    // Python graph description (stored for BuildRenderGraph)
    RenderGraphDescription m_pythonGraphDesc;
    std::unordered_map<std::string, std::shared_ptr<rhi::ComputeWorkInstance>> m_workInstances;
    std::unordered_map<std::string, std::shared_ptr<rhi::ComputeBuffer>> m_graphBufferOwners;
    std::vector<std::pair<vk::ResourceHandle, size_t>> m_graphBufferImports;
    std::vector<std::pair<vk::ResourceHandle, const rhi::ComputeBuffer *>> m_materialBufferImports;
    GraphBufferContents m_graphBufferContents;
    std::unordered_set<std::string> m_graphBufferWrites;
    std::vector<std::weak_ptr<GraphBufferCapture>> m_bufferCaptures;
    struct AuthorPass
    {
        size_t index;
        std::optional<size_t> dispatch;
    };
    std::unordered_map<uint32_t, AuthorPass> m_authorPassIndices;
    std::vector<size_t> m_executedEventPasses;
    // Validated readers of the current description survive a deferred rebuild.
    std::vector<GraphBufferInput *> m_liveBufferInputs;
    uint64_t m_bufferOccurrence = 0;
    uint64_t m_bufferGeneration = 0;
    uint64_t m_historyEpoch = 0;
    bool m_historyResetPending = false;
    uint64_t m_requestedDraw = 0;
    uint64_t m_objectBindingIdentity = 0;
    bool m_reusesObjectBindings = false;
    CameraClearFlags m_bufferProofClearFlags = {};
    uint64_t m_resourceAccessRevision = 1;

    struct RuntimeParameterBlock
    {
        uint64_t revision = 0;
        std::vector<std::string> names;
        FullscreenPushConstants values{};
        uint32_t byteSize = 0;
    };

    // Dynamic values keyed by GraphCommandDesc::parameterBlock. Layout is
    // compiled with the graph; revisions update independently at runtime.
    // Execution reads the submission snapshot rather than this live authoring
    // state.
    std::unordered_map<std::string, RuntimeParameterBlock> m_parameterBlocks;
    std::unordered_map<std::string, RuntimeParameterBlock> m_submittedParameterBlocks;
    uint64_t m_parameterBlockGeneration = 0;
    uint64_t m_submittedParameterBlockGeneration = 0;

    // Render callbacks keyed by pass name.
    // Published by ApplyPythonGraph and consumed when building the runtime graph.
    std::unordered_map<std::string, ScenePassRenderCallback> m_pythonCallbacks;
    std::unordered_map<std::string, MaterialPassPipelineDescriptor> m_pythonMaterialPasses;

    particle::ParticleGpuDrawRegistry *m_particleDrawRegistry = nullptr;
    uint64_t m_particleDrawRegistryRevision = 0;
    std::unordered_map<uint64_t, std::shared_ptr<particle::ParticleGpuCuller>> m_particleCullers;
    std::unordered_map<uint64_t, std::shared_ptr<particle::ParticleGpuSorter>> m_particleSorters;
    struct ParticleViewBinding
    {
        std::shared_ptr<particle::ParticleGpuOutputRenderer> renderer;
        rhi::BufferHandle indices;
    };
    std::unordered_map<uint64_t, ParticleViewBinding> m_particleViewBindings;
    void ReleaseParticleViewBindings() noexcept;
    std::shared_ptr<ParticleViewDiagnosticState> m_particleViewDiagnosticState;
    std::vector<PendingParticleViewDiagnostic> m_pendingParticleViewDiagnostics;
    uint64_t m_nextParticleViewDiagnosticRequestId = 1;

    // The underlying render graph (now fully utilized)
    std::unique_ptr<vk::RenderGraph> m_renderGraph;

    // Imported resource handles from scene target
    vk::ResourceHandle m_importedColorTarget;
    vk::ResourceHandle m_importedResolveTarget; // 1x resolve target for MSAA
    vk::ResourceHandle m_importedDepthTarget;
    vk::ResourceHandle m_visibleRendererList;
    vk::ResourceHandle m_shadowRendererList;

    struct TemporalHistoryResource
    {
        std::array<std::shared_ptr<rhi::RenderTexture>, 2> targets{};
        vk::ResourceHandle readHandle;
        vk::ResourceHandle writeHandle;
        std::string readName;
        std::string writeName;
        rhi::PixelFormat format = rhi::PixelFormat::Undefined;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t readIndex = 0;
        bool valid = false;
    };
    std::unordered_map<std::string, TemporalHistoryResource> m_temporalHistories;
    struct ImportedTextureBinding
    {
        rhi::TextureHandle texture;
        rhi::SamplerHandle sampler;
        rhi::PixelFormat format;
        uint64_t revision;
        uint64_t contentGeneration = 0;
        /// Retained by the compiled graph's external allocation owner.
        const rhi::TextureResource *persistentAttachment = nullptr;
        bool operator!=(const ImportedTextureBinding &other) const noexcept
        {
            return texture != other.texture || sampler != other.sampler || format != other.format ||
                   revision != other.revision || contentGeneration != other.contentGeneration;
        }
    };
    std::unordered_map<std::string, ImportedTextureBinding> m_importedTextureBindings;
    std::vector<std::shared_ptr<const rhi::TextureGpuView>> m_graphTexturePublications;
    std::vector<MaterialTextureRead> m_materialTextureReads;
    std::vector<MaterialBufferRead> m_materialBufferReads;
    rhi::ComputeBufferBindings m_computeBufferReads;
    std::unordered_map<std::string, std::vector<std::shared_ptr<const rhi::RenderTextureGeneration>>>
        m_drawTextureInputs;

    // Camera-driven clear overrides (set per-frame by UpdateMainPassClearSettings)
    bool m_hasCameraClearOverride = false;
    CameraClearFlags m_cameraClearFlags = {};
    glm::vec4 m_cameraBgColor{0.1f, 0.1f, 0.1f, 1.0f};
    bool m_cameraDithering = false;
    bool m_cameraStopNaNs = false;

    // Previous frame's camera clear state — used to detect changes that
    // actually require a graph rebuild (= loadOp change) vs. changes that
    // only require updating clear *values* (no rebuild needed).
    bool m_prevClearStateValid = false;
    CameraClearFlags m_prevCameraClearFlags = {};
    glm::vec4 m_prevCameraBgColor{0.1f, 0.1f, 0.1f, 1.0f};

    // Name of the first graph pass that clears color (set during BuildRenderGraph).
    // Used to apply per-frame clear value updates without rebuilding the graph.
    std::string m_mainClearPassName;

    // Dimensions
    uint32_t m_width = 0;
    uint32_t m_height = 0;

    // Per-graph draw call cache for multi-camera rendering
    RendererList m_cachedRenderers;
    bool m_hasCachedDrawCalls = false;
    RendererList m_cachedShadowRenderers;
    bool m_hasCachedShadowDrawCalls = false;
    uint64_t m_cachedSubmissionSignature = 0;
    uint64_t m_cachedObjectBufferRevision = 0;
    uint64_t m_cachedParticleDrawRegistryRevision = 0;
    std::shared_ptr<const void> m_cachedRenderWorldOwner;
    bool m_hasShadowCasterPass = false;
    // Shadow attachments belong to one camera graph and survive across graph
    // executions. Static scenes can therefore preserve the atlas until an
    // input generation changes instead of clearing and redrawing every view.
    uint64_t m_committedShadowContentSignature = 0;
    uint64_t m_pendingShadowContentSignature = 0;
    bool m_shadowAtlasValid = false;
    bool m_shadowAtlasUpdateRequired = true;

    // Per-graph camera VP cache — set by SubmitCulling so the executor
    // uses the exact same matrices that were active during SetupCameraProperties.
    glm::mat4 m_cachedView{1.0f};
    glm::mat4 m_cachedProj{1.0f};
    glm::mat4 m_cachedUnjitteredProj{1.0f};
    std::array<float, 24> m_particleFrustumPlanes{};
    bool m_hasCachedCameraVP = false;
    bool m_cameraMatricesStaged = false;
    glm::mat4 m_drawView{1.0f};
    glm::mat4 m_billboardView{1.0f};
    bool m_hasBillboardView = false;
    glm::mat4 m_previousViewProj{1.0f};
    RenderInstanceHistory m_motionHistory;
    bool m_cameraHistoryValid = false;
    const Camera *m_cachedCamera = nullptr;
    uint64_t m_cachedCameraHistoryRevision = 0;
    static constexpr uint32_t kTemporalJitterSampleCount = 8;
    uint32_t m_temporalSampleIndex = 0;
    glm::vec2 m_temporalJitterNdc{0.0f};

    // Per-graph shadow descriptor sets (set 1) — multi-camera shadow isolation.
    // One set per frame-in-flight to prevent host-side vkUpdateDescriptorSets
    // from stomping a set the GPU is still sampling in the previous frame.
    struct PerViewBufferBindingState
    {
        VkBuffer canonicalLights = VK_NULL_HANDLE;
        VkBuffer tileHeaders = VK_NULL_HANDLE;
        VkBuffer tileLightMasks = VK_NULL_HANDLE;
        VkBuffer lighting = VK_NULL_HANDLE;
        uint64_t canonicalBytes = 0;
        uint64_t tileHeaderBytes = 0;
        uint64_t tileLightMaskBytes = 0;
        bool initialized = false;
    };
    struct PerViewShadowBindingState
    {
        VkImageView imageView = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        // A reset invalidates the descriptor's publication proof. Each frame
        // slot must bind its default texture after its own fence has passed.
        bool usesDefaultTexture = false;
    };

    struct PerViewFrameState
    {
        vk::ViewGlobalsBinding globals;
        vk::DescriptorLease geometryDescriptor;
        vk::DescriptorLease particleDescriptor;
        vk::DescriptorLease editorOverlayDescriptor;
        rhi::BindGroupHandle geometryGroup;
        rhi::BindGroupHandle particleGroup;
        rhi::BindGroupHandle editorOverlayGroup;
        rhi::BufferHandle cameraMatrix;
        rhi::BufferHandle editorOverlayCameraMatrix;
        rhi::BufferHandle lighting;
        PerViewBufferBindingState geometryBindings;
        PerViewBufferBindingState particleBindings;
        PerViewShadowBindingState shadowBinding;

        [[nodiscard]] VkDescriptorSet GeometrySet() const noexcept
        {
            return geometryDescriptor.set;
        }

        [[nodiscard]] VkDescriptorSet ParticleSet() const noexcept
        {
            return particleDescriptor.set;
        }

        [[nodiscard]] VkDescriptorSet EditorOverlaySet() const noexcept
        {
            return editorOverlayDescriptor.set;
        }
    };

    std::array<PerViewFrameState, EngineConfig::kFrameSlotCapacity> m_perViewFrames{};
    rhi::BindingLayoutHandle m_perViewLayout;
    lighting::CanonicalLightGpuBuffer m_cameraCanonicalLights;
    SceneLightCollector m_cameraLightCollector;
    uint64_t m_shadowCameraResourceId = 0;

    // Resource handle bound to the graph's "shadowMap" sampler input.
    // Resolved after Compile() so the per-view descriptor can be updated
    // BEFORE command buffer recording starts.
    vk::ResourceHandle m_shadowMapInputHandle;
    bool m_shadowMapInputIsDepth = false;

    // Fullscreen effect renderer — manages pipeline cache, descriptor pool,
    // and linear sampler for FullscreenQuad graph passes.
    FullscreenRenderer m_fullscreenRenderer;
    SceneDepthResolver m_sceneDepthResolver;
    lighting::ForwardPlusLightGrid m_forwardPlusGeometryGrid;
    lighting::ForwardPlusLightGrid m_forwardPlusParticleGrid;
    bool m_forwardPlusParticlesRequired = false;
};

} // namespace infernux
