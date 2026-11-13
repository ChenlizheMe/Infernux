#include <core/math/DepthConvention.h>
/**
 * @file SceneRenderGraph.cpp
 * @brief Implementation of RenderGraph-based scene rendering
 *
 * This implementation fully utilizes vk::RenderGraph for all rendering.
 * No more imperative BeginRenderPass/EndRenderPass calls.
 */

#include "Frustum.h"
#include "FullscreenRenderer.h"
#include "InxVkCoreModular.h"
#include "MsaaPolicy.h"
#include "OutlineRenderer.h"
#include "RendererSelection.h"
#include "SceneRenderGraph.h"
#include "SceneRenderTarget.h"
#include "gui/InxScreenUIRenderer.h"
#include "gui/WorldUIDepthReplay.h"
#include "particle/ParticleGpuBounds.h"
#include "particle/ParticleGpuCuller.h"
#include "particle/ParticleGpuDrawRegistry.h"
#include "particle/ParticleGpuSorter.h"
#include "rhi/RhiComputeBuffer.h"
#include "rhi/RhiComputeKernel.h"
#include "shader/ShaderReflection.h"
#include "vk/RhiVulkanTypes.h"
#include "vk/VkDeviceContext.h"
#include "vk/VkPipelineManager.h"
#include "vk/VkRenderUtils.h"
#include "vk/VulkanRhiDevice.h"
#include <algorithm>
#include <cmath>
#include <core/config/EngineConfig.h>
#include <core/error/InxError.h>
#include <core/types/ColorSpace.h>
#include <cstring>
#include <function/renderer/rhi/RhiBuffer.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxFileLoader/InxShaderLoader.hpp>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxTexture/InxTexture.h>
#include <function/scene/Camera.h>
#include <function/scene/LightingData.h>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_set>

namespace infernux
{

GraphFramePublication::~GraphFramePublication()
{
    for (const auto &capture : captures)
        capture.request->Cancel("graph submission did not succeed");
}

void GraphFramePublication::Accept(rhi::SubmissionTicket completion, GraphEventQueue &acceptedEvents) noexcept
{
    for (auto &write : writes)
        write.allocation->PublishGraphValue(completion, rhi::QueueRole::Graphics, std::move(write.initialized),
                                            std::move(write.written));
    for (auto &capture : captures)
        if (capture.request->status == GraphBufferCapture::Status::Pending) {
            capture.request->value = std::move(capture.value);
            capture.request->status = GraphBufferCapture::Status::Accepted;
        }
    captures.clear();
    for (auto &event : events)
        event.submission = completion;
    acceptedEvents.splice(acceptedEvents.end(), events);
}

void SceneRenderGraph::CancelBufferCaptures(const std::string &reason)
{
    for (const auto &weak : m_bufferCaptures)
        if (const auto capture = weak.lock())
            capture->Cancel(reason);
    m_bufferCaptures.clear();
}

const GraphBufferDesc &SceneRenderGraph::FindBufferExport(const std::string &name) const
{
    const auto buffer =
        std::find_if(m_pythonGraphDesc.buffers.begin(), m_pythonGraphDesc.buffers.end(),
                     [&](const auto &item) { return item.name == name && item.exportLayout.has_value(); });
    if (!m_hasPythonGraph || !m_pythonGraphValid || buffer == m_pythonGraphDesc.buffers.end())
        throw std::invalid_argument("No exported buffer in this view: " + name);
    return *buffer;
}

std::shared_ptr<GraphBufferCapture> SceneRenderGraph::MakeBufferCapture(const GraphBufferDesc &buffer) const
{
    auto capture = std::make_shared<GraphBufferCapture>();
    capture->name = buffer.name;
    capture->shape = buffer.exportShape;
    capture->scalarType = buffer.exportLayout->scalarType;
    capture->byteOffset = buffer.byteOffset;
    capture->storageByteSize = buffer.exportLayout->GetByteSize();
    capture->view = m_renderView.id;
    capture->generation = m_bufferGeneration;
    capture->occurrence = m_bufferOccurrence + 1;
    return capture;
}

std::shared_ptr<GraphBufferCapture> SceneRenderGraph::CaptureBuffer(const std::string &name)
{
    const auto &buffer = FindBufferExport(name);
    auto capture = MakeBufferCapture(buffer);
    const auto &storage = ResolveGraphBufferStorage(m_pythonGraphDesc, buffer);
    auto imported = storage.computeBuffer;
    if (!storage.workBinding.empty()) {
        const auto *work = FindGraphWorkBinding(m_pythonGraphDesc, storage.workBinding);
        imported = work->recipe->GetSlots()[storage.workSlot].external;
        capture->storageClaim = "work:" + work->name + "/" + std::to_string(storage.workSlot);
    }
    if (storage.input) {
        capture->storageClaim = "input:" + std::to_string(reinterpret_cast<uintptr_t>(storage.input.get()));
    } else if (imported && imported->IsFrozen()) {
        capture->storageBuffer = imported;
        const auto handle = imported->GetBuffer();
        capture->storageClaim = "frozen:" + std::to_string(handle.index) + ":" + std::to_string(handle.generation);
    } else if (imported)
        capture->storageClaim = "import:" + std::to_string(reinterpret_cast<uintptr_t>(imported.get()));
    else if (capture->storageClaim.empty())
        capture->storageClaim = "buffer:" + storage.name;
    m_bufferCaptures.push_back(capture);
    return capture;
}

static std::shared_ptr<GraphEventInterest> FindActiveEventInterest(const GraphEventInterests &interests,
                                                                   const std::string &label)
{
    const auto found = interests.find(label);
    if (found == interests.end())
        return {};
    auto interest = found->second.lock();
    return interest && interest->active ? interest : nullptr;
}

std::shared_ptr<GraphFramePublication> SceneRenderGraph::PreparePublication(const GraphEventInterests &interests)
{
    if (m_graphBufferOwners.empty() && m_executedEventPasses.empty())
        return {};
    std::shared_ptr<GraphFramePublication> publication;
    const auto prepare = [&]() -> GraphFramePublication & {
        if (!publication)
            publication = std::make_shared<GraphFramePublication>();
        return *publication;
    };
    std::unordered_set<const rhi::ComputeBufferAllocation *> graphWriters;
    for (const auto &name : m_graphBufferWrites) {
        const auto owner = m_graphBufferOwners.find(name);
        if (owner != m_graphBufferOwners.end())
            graphWriters.insert(owner->second->GetAllocation().get());
    }
    std::unordered_set<const rhi::ComputeBufferAllocation *> written;
    for (const auto &buffer : m_pythonGraphDesc.buffers) {
        if (!buffer.exportLayout)
            continue;
        const auto &allocation = m_computeBufferReads.at(m_graphBufferOwners.at(buffer.name).get());
        if (graphWriters.count(allocation.get()) && written.insert(allocation.get()).second) {
            const auto &known = m_graphBufferContents.at(buffer.name);
            prepare().writes.push_back({allocation, known, known});
        }
    }
    ++m_bufferOccurrence;
    std::unordered_map<std::string, std::shared_ptr<rhi::ComputeBuffer>> frozen;
    const auto prepareCapture = [&](const std::shared_ptr<GraphBufferCapture> &capture) {
        auto [entry, inserted] = frozen.try_emplace(capture->name);
        if (inserted) {
            const auto &owner = m_graphBufferOwners.at(capture->name);
            // Keep the allocation used by this execution's descriptors and commands.
            if (owner->GetAllocation() != m_computeBufferReads.at(owner.get()))
                throw std::logic_error("Graph output changed after actual resource capture");
            entry->second = owner->FreezeScalars();
        }
        prepare().captures.push_back({capture, entry->second});
        capture->occurrence = m_bufferOccurrence;
        capture->selected = true;
    };
    auto output = m_bufferCaptures.begin();
    for (auto it = m_bufferCaptures.begin(); it != m_bufferCaptures.end(); ++it) {
        const auto capture = it->lock();
        if (!capture || capture->status != GraphBufferCapture::Status::Pending)
            continue;
        *output++ = *it;
        if (!capture->selected)
            prepareCapture(capture);
    }
    m_bufferCaptures.erase(output, m_bufferCaptures.end());
    std::shared_ptr<GraphEventViewSample> viewSample;
    for (size_t index : m_executedEventPasses) {
        const auto &pass = m_pythonGraphDesc.passes[index];
        for (size_t ordinal = 0; ordinal < pass.events.size(); ++ordinal) {
            const auto &declaration = pass.events[ordinal];
            const auto interest = FindActiveEventInterest(interests, declaration.label);
            if (!interest)
                continue;
            if (!viewSample && m_hasCachedCameraVP && m_cameraMatricesStaged) {
                viewSample = std::make_shared<GraphEventViewSample>(GraphEventViewSample{
                    m_width, m_height, m_temporalJitterNdc, m_cachedView, m_cachedProj, m_cachedUnjitteredProj,
                    m_cameraHistoryValid ? std::optional{m_previousViewProj} : std::nullopt});
            }
            decltype(GraphEventRecord::sources) sources;
            sources.reserve(declaration.buffers.size());
            for (const auto &[name, resource] : declaration.buffers) {
                auto capture = MakeBufferCapture(FindBufferExport(resource));
                prepareCapture(capture);
                sources.emplace_back(name, std::move(capture));
            }
            prepare().events.push_back({declaration.label,
                                        declaration.metadata,
                                        interest,
                                        pass.name,
                                        ordinal,
                                        m_renderView.id,
                                        m_bufferGeneration,
                                        m_bufferOccurrence,
                                        m_historyEpoch,
                                        {},
                                        viewSample,
                                        std::move(sources)});
        }
    }
    return publication;
}

namespace
{

constexpr uint64_t kShadowHashOffset = 1469598103934665603ull;
constexpr uint64_t kShadowHashPrime = 1099511628211ull;

void HashShadowBytes(uint64_t &hash, const void *data, size_t size)
{
    const auto *bytes = static_cast<const uint8_t *>(data);
    for (size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= kShadowHashPrime;
    }
}

template <typename T> void HashShadowValue(uint64_t &hash, const T &value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    HashShadowBytes(hash, &value, sizeof(T));
}

lighting::ShadowDepthRange VisibleShadowDepthRange(const Camera *camera, const std::vector<DrawCall> &drawCalls)
{
    lighting::ShadowDepthRange result{};
    if (!camera)
        return result;

    // Cascade selection in the shader uses view-space z, including any
    // affine view override. A normalized world forward changes those units.
    const auto view = camera->GetViewMatrix();
    const glm::vec3 depthAxis(view[0][2], view[1][2], view[2][2]);
    float nearest = std::numeric_limits<float>::max();
    float farthest = 0.0f;
    for (const DrawCall &drawCall : drawCalls) {
        // Device-authored receivers cover the camera projection. Their final
        // GPU bounds cannot participate in a CPU-only tightening of cascades.
        if (drawCall.devicePose.value)
            return {};
        if (!drawCall.frustumVisible || !drawCall.worldBounds.IsValid())
            continue;
        const glm::vec3 center = (drawCall.worldBounds.min + drawCall.worldBounds.max) * 0.5f;
        const glm::vec3 extent = (drawCall.worldBounds.max - drawCall.worldBounds.min) * 0.5f;
        const float centerDepth = glm::dot(center, depthAxis) + view[3][2];
        const float depthRadius = glm::dot(glm::abs(depthAxis), extent);
        const float objectFar = centerDepth + depthRadius;
        if (objectFar <= 0.0f)
            continue;

        // A huge receiver crossing the camera should not collapse the entire
        // logarithmic distribution onto the near clip plane.
        const float boundedRadius = std::min(depthRadius, std::max(centerDepth * 0.5f, 0.0f));
        nearest = std::min(nearest, std::max(centerDepth - boundedRadius, 0.001f));
        farthest = std::max(farthest, objectFar);
    }
    if (farthest <= 0.0f || nearest == std::numeric_limits<float>::max())
        return result;

    // These are geometry bounds. ComputeShadowVP intersects them with the
    // effective projection; dormant authored clipping planes do not apply.
    const float span = std::max(farthest - nearest, 0.001f);
    result.nearDepth = nearest;
    result.farDepth = farthest + span * 0.05f;
    return result;
}

} // namespace

// ============================================================================
// Constructor / Destructor
// ============================================================================

SceneRenderGraph::SceneRenderGraph()
    : m_particleViewDiagnosticState(std::make_shared<ParticleViewDiagnosticState>()),
      m_renderGraph(std::make_unique<vk::RenderGraph>())
{
}

SceneRenderGraph::~SceneRenderGraph()
{
    Destroy();
}

uint64_t SceneRenderGraph::GetTransientResidentBytes() const
{
    return m_renderGraph ? m_renderGraph->GetTransientResidentBytes() : 0;
}

// ============================================================================
// Initialization
// ============================================================================

bool SceneRenderGraph::Initialize(InxVkCoreModular *vkCore, SceneRenderTarget *sceneTarget,
                                  rhi::RenderViewKind viewKind, std::shared_ptr<rhi::RenderTexture> output)
{
    if (!vkCore || (!sceneTarget && !output)) {
        INXLOG_ERROR("SceneRenderGraph::Initialize: Invalid parameters");
        return false;
    }

    m_vkCore = vkCore;
    m_screenTarget = sceneTarget;
    if (output) {
        m_outputTexture = std::move(output);
        m_outputGeneration = m_outputTexture->Acquire();
        m_outputTarget = std::make_unique<SceneRenderTarget>(vkCore);
        m_outputTarget->BindAttachments(m_outputGeneration);
        sceneTarget = m_outputTarget.get();
    }
    m_sceneTarget = sceneTarget;
    m_width = sceneTarget->GetWidth();
    m_height = sceneTarget->GetHeight();
    ++m_resourceAccessRevision;

    m_renderView.id = rhi::AllocateRenderViewId();
    m_renderView.device = vkCore->GetDeviceContext().GetDeviceId();
    m_renderView.kind = viewKind;
    m_renderView.output = rhi::RenderOutputKind::OffscreenTexture;
    m_renderView.width = m_width;
    m_renderView.height = m_height;
    m_renderView.colorFormat = rhi::FromVkFormat(sceneTarget->GetColorFormat());
    m_renderView.depthFormat = rhi::FromVkFormat(sceneTarget->GetDepthFormat());
    m_renderView.samples = rhi::FromVkSampleCount(sceneTarget->GetMsaaSampleCount());
    m_renderView.revision = 1;

    m_renderGraph->Initialize(&vkCore->GetDeviceContext(), &vkCore->GetRetirementQueue(),
                              &vkCore->GetBackendContext().Queues());
    m_renderGraph->SetRenderView(m_renderView);

    // Logical identity is immediate; an unused successor must not allocate a
    // complete set of view buffers and descriptors before it is replaced.
    m_shadowCameraResourceId = vkCore->CreateShadowCameraResources();
    if (m_shadowCameraResourceId == 0) {
        INXLOG_ERROR("SceneRenderGraph: failed to allocate camera-local shadow resource identity");
        return false;
    }
    return true;
}

bool SceneRenderGraph::EnsureViewGpuResources()
{
    if (m_viewGpuResourcesReady)
        return true;
    if (!m_vkCore || !m_sceneTarget || !m_sceneTarget->IsReady())
        return false;

    // Every early return and exception rolls back this same owned subset.
    // Logical view identity and RenderGraph image inventory remain untouched.
    struct Rollback
    {
        SceneRenderGraph &graph;
        ~Rollback()
        {
            if (!graph.m_viewGpuResourcesReady)
                graph.ReleaseViewGpuResources();
        }
    } rollback{*this};
    auto *vkCore = m_vkCore;
    // Allocate a complete camera-local lighting domain. Geometry and particles
    // share this camera's atlas and LightingUBO, but Scene, Game and future
    // camera graphs never overwrite one another's shadow state.
    auto &rhiDevice = vkCore->GetDeviceContext().GetRhiDevice();
    rhi::BufferDesc lightingBufferDesc{};
    lightingBufferDesc.byteSize = sizeof(ShaderLightingUBO);
    lightingBufferDesc.usage = rhi::BufferUsageFlags::Uniform;
    lightingBufferDesc.memory = rhi::BufferMemory::Upload;
    rhi::BufferDesc cameraBufferDesc{};
    cameraBufferDesc.byteSize = sizeof(UniformBufferObject);
    cameraBufferDesc.usage = rhi::BufferUsageFlags::Uniform;
    cameraBufferDesc.memory = rhi::BufferMemory::Upload;
    rhi::BufferDesc globalsBufferDesc = cameraBufferDesc;
    globalsBufferDesc.byteSize = sizeof(EngineGlobalsUBO);
    for (auto &frame : m_perViewFrames) {
        frame.globals.buffer = rhiDevice.CreateBuffer(globalsBufferDesc);
        frame.cameraMatrix = rhiDevice.CreateBuffer(cameraBufferDesc);
        frame.editorOverlayCameraMatrix = rhiDevice.CreateBuffer(cameraBufferDesc);
        if (!frame.globals.buffer.IsValid() || !frame.cameraMatrix.IsValid() ||
            !frame.editorOverlayCameraMatrix.IsValid()) {
            INXLOG_ERROR("SceneRenderGraph: failed to allocate camera-local matrix UBO");
            return false;
        }
    }
    for (auto &frame : m_perViewFrames) {
        frame.lighting = rhiDevice.CreateBuffer(lightingBufferDesc);
        if (!frame.lighting.IsValid()) {
            INXLOG_ERROR("SceneRenderGraph: failed to allocate camera-local LightingUBO");
            return false;
        }
    }
    if (!m_cameraCanonicalLights.Initialize(rhiDevice, EngineConfig::kFrameSlotCapacity)) {
        INXLOG_ERROR("SceneRenderGraph: failed to allocate camera-local canonical light buffers");
        return false;
    }
    m_perViewLayout = rhiDevice.RegisterBindingLayout(vkCore->GetPerViewDescSetLayout());
    if (!m_perViewLayout.IsValid()) {
        INXLOG_ERROR("SceneRenderGraph: failed to register the canonical per-view descriptor layout");
        return false;
    }
    for (uint32_t i = 0; i < EngineConfig::kFrameSlotCapacity; ++i) {
        auto &frame = m_perViewFrames[i];
        frame.geometryDescriptor = vkCore->AllocatePerViewDescriptorLease();
        frame.particleDescriptor = vkCore->AllocatePerViewDescriptorLease();
        frame.editorOverlayDescriptor = vkCore->AllocatePerViewDescriptorLease();
        frame.geometryGroup = rhiDevice.RegisterBindGroup(frame.GeometrySet());
        frame.particleGroup = rhiDevice.RegisterBindGroup(frame.ParticleSet());
        frame.editorOverlayGroup = rhiDevice.RegisterBindGroup(frame.EditorOverlaySet());
        if (!frame.geometryDescriptor.IsValid() || !frame.particleDescriptor.IsValid() ||
            !frame.editorOverlayDescriptor.IsValid() || !frame.geometryGroup.IsValid() ||
            !frame.particleGroup.IsValid() || !frame.editorOverlayGroup.IsValid()) {
            INXLOG_ERROR("SceneRenderGraph: failed to allocate the canonical per-view descriptor resources [", i, "]");
            return false;
        }
        const VkBuffer lightingBuffer = rhiDevice.Resolve(frame.lighting);
        const VkBuffer cameraBuffer = rhiDevice.Resolve(frame.cameraMatrix);
        vkCore->UpdatePerViewLightingBuffer(frame.GeometrySet(), lightingBuffer, sizeof(ShaderLightingUBO));
        vkCore->UpdatePerViewLightingBuffer(frame.ParticleSet(), lightingBuffer, sizeof(ShaderLightingUBO));
        vkCore->UpdatePerViewLightingBuffer(frame.EditorOverlaySet(), lightingBuffer, sizeof(ShaderLightingUBO));
        vkCore->UpdatePerViewCameraBuffer(frame.GeometrySet(), cameraBuffer, sizeof(UniformBufferObject));
        vkCore->UpdatePerViewCameraBuffer(frame.ParticleSet(), cameraBuffer, sizeof(UniformBufferObject));
        vkCore->UpdatePerViewCameraBuffer(frame.EditorOverlaySet(), rhiDevice.Resolve(frame.editorOverlayCameraMatrix),
                                          sizeof(UniformBufferObject));
    }

    // Initialize fullscreen effect renderer for FullscreenQuad passes
    m_fullscreenRenderer.Initialize(vkCore);
    if (!m_fullscreenRenderer.GetLinearSampler().IsValid()) {
        INXLOG_ERROR("SceneRenderGraph: failed to prepare fullscreen view resources");
        return false;
    }

    const auto depthResolveSupport = vkCore->GetDeviceContext().GetCapabilities().CheckFormat(
        rhi::PixelFormat::R32SFloat, rhi::FormatFeature::Sampled | rhi::FormatFeature::Storage);
    if (depthResolveSupport.IsSupported()) {
        m_sceneDepthResolver.Initialize(vkCore->GetSceneDepthResolveProgram());
    } else {
        INXLOG_WARN("SceneRenderGraph: R32SFloat sampled-storage textures are unavailable; MSAA soft particles are "
                    "disabled on this adapter");
    }

    // These two fixed built-in programs belong to the device, not a camera.
    // Recreating a Scene/Game view must not compile the same shaders again.
    const auto lightGridProgram = vkCore->GetForwardPlusGridProgram();
    if (!m_forwardPlusGeometryGrid.Initialize(lightGridProgram, EngineConfig::kFrameSlotCapacity) ||
        !m_forwardPlusParticleGrid.Initialize(lightGridProgram, EngineConfig::kFrameSlotCapacity))
        return false;
    for (uint32_t frameIndex = 0; frameIndex < EngineConfig::kFrameSlotCapacity; ++frameIndex) {
        const auto &lights = m_cameraCanonicalLights.Frame(frameIndex);
        if (lights.buffer.IsValid()) {
            if (!m_forwardPlusGeometryGrid.PrepareFrame(frameIndex, m_width, m_height, lights.localCount,
                                                        CanonicalLightAffectsGeometry, lights.buffer) ||
                !m_forwardPlusParticleGrid.PrepareFrame(frameIndex, m_width, m_height, lights.localCount,
                                                        CanonicalLightAffectsParticles, lights.buffer))
                return false;
        }
    }

    m_viewGpuResourcesReady = true;
    return true;
}

void SceneRenderGraph::InvalidateBeforeTargetReplacement()
{
    if (HasOutputTexture())
        return; // A screen resize does not replace this camera's allocation.
    m_graphBuilt = false;
    m_needsCompile = true;
    if (!m_workInstances.empty()) {
        RetireWorkInstances();
        m_needsRebuild = true;
    }
}

void SceneRenderGraph::SetOutputTexture(std::shared_ptr<rhi::RenderTexture> output)
{
    const auto generation = output ? output->Acquire() : nullptr;
    if (m_outputTexture == output && m_outputGeneration == generation)
        return;
    ++m_resourceAccessRevision;
    m_outputTexture = std::move(output);
    m_outputGeneration = generation;
    if (generation) {
        if (!m_outputTarget)
            m_outputTarget = std::make_unique<SceneRenderTarget>(m_vkCore);
        m_outputTarget->BindAttachments(generation);
    }
    m_sceneTarget = nullptr;
    ReplaceSceneTarget(m_screenTarget);
    if (!m_outputTexture)
        m_outputTarget.reset();
    m_pythonGraphSourceRevision = 0; // Output domain can change without changing the Python artifact.
    SetEffectiveMsaaSamples(m_effectiveMsaaSamples);
}

bool SceneRenderGraph::SetCameraViewport(const Camera &camera)
{
    auto *output = m_outputTarget && m_outputTexture ? m_outputTarget.get() : m_screenTarget;
    if (!output || !output->GetAttachments())
        return false;
    const bool changed = m_viewport.Prepare(m_vkCore->GetDeviceContext().GetRhiDevice(),
                                            "camera-viewport/" + std::to_string(m_renderView.id),
                                            output->GetAttachments(), camera.GetRect());
    const auto generation = m_viewport.Work() ? m_viewport.Work()->Acquire() : nullptr;
    if (changed || generation != m_viewportGeneration) {
        m_sceneTarget = nullptr;
        m_viewportGeneration = generation;
        if (generation) {
            if (!m_viewportWorkTarget)
                m_viewportWorkTarget = std::make_unique<SceneRenderTarget>(m_vkCore);
            m_viewportWorkTarget->BindAttachments(generation);
        } else {
            m_viewportWorkTarget.reset();
        }
        ReplaceSceneTarget(m_screenTarget);
        SetEffectiveMsaaSamples(m_effectiveMsaaSamples);
    }
    return !m_viewport.IsEmpty();
}

bool SceneRenderGraph::HasLocalTextureProducer(const RenderGraphDescription &description,
                                               const MaterialTextureRead &read)
{
    bool reachedReader = false, hasWriter = false, priorWriter = false;
    for (const auto &pass : description.passes) {
        const bool writes =
            std::any_of(description.textures.begin(), description.textures.end(), [&](const GraphTextureDesc &texture) {
                return texture.role == GraphTextureRole::Persistent && texture.renderTexture == read.texture &&
                       ResolveGraphTextureAttachment(*read.generation, texture.attachment) ==
                           read.generation->color.get() &&
                       pass.WritesTexture(texture.name);
            });
        if (pass.name == read.passName) {
            if (writes)
                throw std::invalid_argument("RenderTexture sampled and written by the same pass: " + read.passName);
            reachedReader = true;
        }
        if (writes) {
            hasWriter = true;
            priorWriter |= !reachedReader;
        }
    }
    if (!reachedReader)
        throw std::invalid_argument("RenderTexture reader pass is not in this graph: " + read.passName);
    if (hasWriter && !priorWriter)
        throw std::invalid_argument("RenderTexture sampled before its local producer completes: " + read.passName);
    return hasWriter;
}

RenderViewAccess SceneRenderGraph::GetResourceAccess() const
{
    RenderViewAccess access;
    access.viewId = m_renderView.id;
    if (m_hasPythonGraph && !m_pythonGraphValid)
        return access;
    const auto addRead = [&access](const rhi::TextureResource *texture, bool requiresProducer) {
        const auto found = std::find_if(access.reads.begin(), access.reads.end(),
                                        [texture](const auto &read) { return read.texture == texture; });
        if (found == access.reads.end())
            access.reads.emplace_back(texture, requiresProducer);
        else
            found->requiresProducer |= requiresProducer;
    };
    for (auto *input : m_liveBufferInputs)
        if (input->IsWaitingForResult())
            for (const auto &source : input->GetResultSources())
                if (source->status == GraphBufferCapture::Status::Pending)
                    access.pendingResults.push_back({input, source->view});
    if (!m_outputGeneration)
        access.writes.push_back(m_screenTarget);
    const auto &device = m_vkCore->GetDeviceContext().GetRhiDevice();
    const auto addWrite = [&](const rhi::TextureResource *attachment, bool expected) {
        if (!attachment)
            return false;
        const bool writes =
            m_graphBuilt ? m_renderGraph->HasLiveWriteToImage(device.Resolve(attachment->GetTexture())) : expected;
        if (!writes)
            return false;
        if (std::find(access.writes.begin(), access.writes.end(), attachment) == access.writes.end())
            access.writes.push_back(attachment);
        return true;
    };
    if (m_outputGeneration) {
        addWrite(m_outputGeneration->color.get(), true);
        addWrite(m_outputGeneration->multisampleColor.get(), true);
        addWrite(m_outputGeneration->depth.get(), true);
    }
    const auto &executionOrder = m_renderGraph->GetExecutionOrder();
    for (const auto &texture : m_pythonGraphDesc.textures) {
        if (texture.role != GraphTextureRole::Persistent)
            continue;
        const auto bound = m_importedTextureBindings.find(texture.name);
        const auto generation = m_graphBuilt ? nullptr : texture.renderTexture->Acquire();
        const auto *attachment = generation ? ResolveGraphTextureAttachment(*generation, texture.attachment)
                                 : bound == m_importedTextureBindings.end() ? nullptr
                                                                            : bound->second.persistentAttachment;
        if (!attachment)
            continue;
        const bool expected =
            !m_graphBuilt && std::any_of(m_pythonGraphDesc.passes.begin(), m_pythonGraphDesc.passes.end(),
                                         [&](const auto &pass) { return pass.WritesTexture(texture.name); });
        if (addWrite(attachment, expected) || !m_graphBuilt)
            continue;
        const bool reads = m_pythonGraphDesc.outputTexture == texture.name ||
                           m_pythonGraphDesc.linearOutputTexture == texture.name ||
                           std::any_of(executionOrder.begin(), executionOrder.end(), [&](uint32_t passId) {
                               const auto author = m_authorPassIndices.find(passId);
                               return author != m_authorPassIndices.end() &&
                                      m_pythonGraphDesc.passes[author->second.index].ReadsTexture(texture.name);
                           });
        if (reads)
            addRead(attachment, texture.requireCurrentWrite);
    }
    if (m_graphBuilt)
        for (const auto &read : m_materialTextureReads)
            if (!HasLocalTextureProducer(m_pythonGraphDesc, read))
                addRead(read.generation->color.get(), false);
    return access;
}

bool SceneRenderGraph::HasPendingBufferInputs(bool forStep) const
{
    return std::any_of(m_liveBufferInputs.begin(), m_liveBufferInputs.end(), [forStep](const auto *input) {
        return input->IsWaitingForResult() && (!forStep || input->HoldsStep());
    });
}

void SceneRenderGraph::CacheLiveAuthorPasses()
{
    m_executedEventPasses.clear();
    m_liveBufferInputs.clear();
    std::unordered_map<std::string_view, GraphBufferInput *> inputs;
    for (const auto &buffer : m_pythonGraphDesc.buffers) {
        const auto &storage = ResolveGraphBufferStorage(m_pythonGraphDesc, buffer);
        if (storage.input)
            inputs.emplace(buffer.name, storage.input.get());
    }
    const auto read = [&](std::string_view name) {
        const auto input = inputs.find(name);
        if (input != inputs.end() &&
            std::find(m_liveBufferInputs.begin(), m_liveBufferInputs.end(), input->second) == m_liveBufferInputs.end())
            m_liveBufferInputs.push_back(input->second);
    };
    // Native export passes are retained reads even when the author only
    // exports an imported slot and no other pass consumes it.
    for (const auto &buffer : m_pythonGraphDesc.buffers)
        if (buffer.exportLayout)
            read(buffer.name);
    if (m_authorPassIndices.empty())
        return;
    const auto &order = m_renderGraph->GetExecutionOrder();
    std::unordered_set<size_t> seen;
    // A split author pass publishes at its last live piece. Identify slots
    // by their authored binding; several slots can alias one GPU allocation.
    for (auto nativePass = order.rbegin(); nativePass != order.rend(); ++nativePass) {
        const auto author = m_authorPassIndices.find(*nativePass);
        if (author == m_authorPassIndices.end())
            continue;
        const auto &record = author->second;
        const auto &pass = m_pythonGraphDesc.passes[record.index];
        const bool firstLivePart = seen.insert(record.index).second;
        if (firstLivePart && !pass.events.empty())
            m_executedEventPasses.push_back(record.index);
        if (inputs.empty())
            continue;
        if (record.dispatch) {
            const auto &work = *FindGraphWorkBinding(m_pythonGraphDesc, pass.PrimaryCommand()->workBinding);
            const auto &ports = work.recipe->GetPorts();
            const auto &operation = work.recipe->GetDispatches()[*record.dispatch];
            const auto &bindings = operation.bindings;
            for (const auto &inputBinding : work.inputBindings) {
                if (inputs.find(inputBinding.second) == inputs.end())
                    continue;
                const auto port = std::find_if(ports.begin(), ports.end(), [&inputBinding](const auto &entry) {
                    return entry.input && entry.name == inputBinding.first;
                });
                if (std::any_of(bindings.begin(), bindings.end(),
                                [&](const auto &binding) {
                                    return binding.range.slot == port->range.slot &&
                                           (static_cast<uint8_t>(binding.access) &
                                            static_cast<uint8_t>(rhi::ComputeBufferAccess::Read));
                                }) ||
                    (operation.indirectArguments && operation.indirectArguments->slot == port->range.slot))
                    read(inputBinding.second);
            }
            continue;
        }
        if (!firstLivePart)
            continue;
        for (const auto &access : pass.bufferAccesses)
            if (access.type != GraphBufferAccessType::StorageWrite &&
                access.type != GraphBufferAccessType::TransferWrite)
                read(access.resource);
        for (const auto &command : pass.commands)
            if (command.type == GraphCommandType::CopyBuffer || command.type == GraphCommandType::CopyBufferToTexture)
                read(command.sourceResource);
    }
    std::reverse(m_executedEventPasses.begin(), m_executedEventPasses.end());
}

void SceneRenderGraph::ReplaceSceneTarget(SceneRenderTarget *sceneTarget)
{
    m_screenTarget = sceneTarget;
    sceneTarget = m_viewportGeneration                ? m_viewportWorkTarget.get()
                  : m_outputTarget && m_outputTexture ? m_outputTarget.get()
                                                      : sceneTarget;
    if (!sceneTarget)
        return;
    if (m_sceneTarget == sceneTarget && m_width == sceneTarget->GetWidth() && m_height == sceneTarget->GetHeight() &&
        m_renderView.colorFormat == rhi::FromVkFormat(sceneTarget->GetColorFormat()) &&
        m_renderView.samples == rhi::FromVkSampleCount(sceneTarget->GetMsaaSampleCount()) && HasOutputTexture())
        return;

    m_sceneTarget = sceneTarget;
    ++m_resourceAccessRevision;
    m_width = sceneTarget->GetWidth();
    m_height = sceneTarget->GetHeight();
    m_renderView.width = m_width;
    m_renderView.height = m_height;
    m_renderView.colorFormat = rhi::FromVkFormat(sceneTarget->GetColorFormat());
    m_renderView.depthFormat = rhi::FromVkFormat(sceneTarget->GetDepthFormat());
    m_renderView.samples = rhi::FromVkSampleCount(sceneTarget->GetMsaaSampleCount());
    ++m_renderView.revision;
    if (m_renderGraph)
        m_renderGraph->SetRenderView(m_renderView);
    m_graphBuilt = false;
    m_needsRebuild = true;
    m_needsCompile = true;
    m_importedColorTarget = {};
    m_importedResolveTarget = {};
    m_importedDepthTarget = {};
    m_temporalHistories.clear();
    InvalidateTemporalHistory();
}

void SceneRenderGraph::InvalidateTemporalHistory()
{
    // Several discontinuities before the next accepted view begin one new history.
    if (!m_historyResetPending) {
        ++m_historyEpoch;
        m_historyResetPending = true;
    }
    m_cameraHistoryValid = false;
    m_motionHistory.Clear();
    m_temporalSampleIndex = 0;
    UpdateTemporalProjection();
    for (auto &[key, history] : m_temporalHistories) {
        (void)key;
        history.valid = false;
        history.readIndex = 0;
    }
}

uint64_t SceneRenderGraph::CurrentParticleDrawRegistryRevision() const noexcept
{
    return m_particleDrawRegistry ? m_particleDrawRegistry->Revision() : 0;
}

void SceneRenderGraph::ReleaseParticleViewBindings() noexcept
{
    for (auto &[id, binding] : m_particleViewBindings)
        binding.renderer->ReleaseViewBinding(binding.indices);
    m_particleViewBindings.clear();
}

void SceneRenderGraph::InvalidateParticleViews()
{
    ReleaseParticleViewBindings();
    const auto retireCuller = [this](std::shared_ptr<particle::ParticleGpuCuller> culler) {
        if (!culler)
            return;
        if (m_vkCore) {
            m_vkCore->GetRetirementQueue().Retire([culler = std::move(culler)]() mutable { culler.reset(); });
            return;
        }
        culler.reset();
    };
    const auto retireSorter = [this](std::shared_ptr<particle::ParticleGpuSorter> sorter) {
        if (!sorter)
            return;
        if (m_vkCore) {
            m_vkCore->GetRetirementQueue().Retire([sorter = std::move(sorter)]() mutable { sorter.reset(); });
            return;
        }
        sorter.reset();
    };
    for (auto &[id, culler] : m_particleCullers) {
        (void)id;
        retireCuller(std::move(culler));
    }
    m_particleCullers.clear();
    for (auto &[id, sorter] : m_particleSorters) {
        (void)id;
        retireSorter(std::move(sorter));
    }
    m_particleSorters.clear();
    m_pendingParticleViewDiagnostics.clear();
    m_particleDrawRegistryRevision = 0;
    m_needsRebuild = true;
    m_graphBuilt = false;
}

void SceneRenderGraph::RetireImportedTextureAssets()
{
    if (m_graphTexturePublications.empty())
        return;
    if (!m_vkCore) {
        m_graphTexturePublications.clear();
        return;
    }
    auto retired = std::move(m_graphTexturePublications);
    m_vkCore->GetRetirementQueue().Retire([retired = std::move(retired)]() mutable { retired.clear(); });
}

void SceneRenderGraph::RetireWorkInstances()
{
    const bool published = m_graphBuffersPublished;
    m_graphBuffersPublished = false;
    if (published)
        RetireComputeBufferBindings();
    else
        m_computeBufferReads.clear();
    auto instances = std::move(m_workInstances);
    auto owners = std::move(m_graphBufferOwners);
    if (published && m_vkCore && (!instances.empty() || !owners.empty()))
        m_vkCore->GetRetirementQueue().Retire([instances = std::move(instances), owners = std::move(owners)]() mutable {
            instances.clear();
            owners.clear();
        });
}

std::shared_ptr<rhi::ComputeBuffer> SceneRenderGraph::ResolveGraphBufferOwner(const GraphBufferDesc &buffer) const
{
    if (buffer.computeBuffer)
        return buffer.computeBuffer;
    const auto found = m_graphBufferOwners.find(buffer.name);
    return found == m_graphBufferOwners.end() ? nullptr : found->second;
}

bool SceneRenderGraph::ProveGraphBufferContents()
{
    GraphBufferContents contents;
    if (RequiresGraphBufferProof(m_pythonGraphDesc)) {
        auto effective = m_pythonGraphDesc;
        effective.msaaSamples = 0;
        if (const auto overridden = ResolveCameraClearOverride()) {
            const auto pass = std::find_if(effective.passes.begin(), effective.passes.end(),
                                           [&](const auto &item) { return item.name == overridden->name; });
            pass->clearColor = overridden->clearColor;
            pass->clearDepth = overridden->clearDepth;
        }
        if (!ValidateOutputTextureAliases(effective) ||
            !ValidateRenderGraphDescription(effective, m_renderView, &contents))
            return false;
    }
    m_graphBufferContents = std::move(contents);
    m_graphBufferWrites = RequiresGraphBufferOwners(m_pythonGraphDesc) ? FindGraphBufferWrites(m_pythonGraphDesc)
                                                                       : std::unordered_set<std::string>{};
    m_bufferProofClearFlags = m_cameraClearFlags;
    return true;
}

bool SceneRenderGraph::PrepareGraphBuffers()
{
    m_graphStorageDeferred = false;
    if (!RequiresGraphBufferOwners(m_pythonGraphDesc)) {
        RetireWorkInstances();
        m_computeBufferReads = CaptureComputeBufferReads();
        return true;
    }
    try {
        rhi::ComputeHost current(m_vkCore->GetDeviceContext().GetRhiDevice(), m_vkCore->PrepareComputeQueue());
        // A cached graph description may outlive its original Engine. Compare
        // captured service facts before dereferencing any old device or queue.
        for (const auto &binding : m_pythonGraphDesc.workBindings) {
            if (!binding.recipe->GetHost().SharesServicesWith(current))
                throw std::invalid_argument("Graph work belongs to an expired or different compute host");
        }
        decltype(m_graphBufferOwners) owners;
        for (const auto &buffer : m_pythonGraphDesc.buffers) {
            if (buffer.computeBuffer) {
                owners.emplace(buffer.name, buffer.computeBuffer);
            } else if (buffer.workBinding.empty() && buffer.exportLayout) {
                const auto old = m_graphBufferOwners.find(buffer.name);
                auto layout = *buffer.exportLayout;
                layout.usage = static_cast<rhi::BufferUsageFlags>(buffer.usage);
                std::shared_ptr<rhi::ComputeBuffer> owner;
                if (old != m_graphBufferOwners.end() && old->second->GetHost().SharesServicesWith(current)) {
                    const auto &previous = old->second->GetDesc();
                    if (previous.elementCount == layout.elementCount && previous.scalarType == layout.scalarType &&
                        previous.lanes == layout.lanes && rhi::HasBufferUsage(previous.usage, layout.usage) &&
                        !old->second->GetAllocation()->IsReadProtected())
                        owner = old->second;
                }
                if (!owner)
                    owner = rhi::ComputeBuffer::TryCreateGraphBuffer(current, layout);
                if (!owner) {
                    m_graphStorageDeferred = true;
                    return false;
                }
                owners.emplace(buffer.name, std::move(owner));
            }
        }
        std::unordered_map<std::string, std::shared_ptr<rhi::ComputeWorkInstance>> instances;
        for (const auto &binding : m_pythonGraphDesc.workBindings) {
            std::vector<rhi::ComputeWorkInput> inputs;
            for (const auto &[port, resource] : binding.inputBindings) {
                const auto owner = owners.find(resource);
                if (owner == owners.end())
                    throw std::invalid_argument("Work input has no earlier typed graph allocation owner");
                inputs.push_back({port, owner->second});
            }
            const auto previous = m_workInstances.find(binding.name);
            auto instance = previous != m_workInstances.end() && &previous->second->GetRecipe() == binding.recipe.get()
                                ? previous->second->Rebind(std::move(inputs))
                                : binding.recipe->Instantiate(std::move(inputs));
            if (!instance) {
                m_graphStorageDeferred = true;
                return false;
            }
            for (const auto &buffer : m_pythonGraphDesc.buffers)
                if (buffer.workBinding == binding.name)
                    owners[buffer.name] = instance->GetBuffers().at(buffer.workSlot);
            // A resource-only rebind can keep the compiled passes. Prepare its
            // immutable graphics bindings here, before any recording callback.
            (void)instance->GetDispatches();
            instances.emplace(binding.name, std::move(instance));
        }
        RetireWorkInstances();
        m_workInstances = std::move(instances);
        m_graphBufferOwners = std::move(owners);
        m_computeBufferReads = CaptureComputeBufferReads();
        return true;
    } catch (const std::exception &error) {
        INXLOG_ERROR("SceneRenderGraph: cannot prepare graph work: ", error.what());
        return false;
    }
}

bool SceneRenderGraph::CanRetainEventBuffers(const GraphEventInterests &interests) const
{
    if (interests.empty() || m_graphBufferWrites.empty())
        return true;
    std::unordered_set<rhi::BufferHandle, rhi::HandleHash<rhi::BufferTag>> captured;
    for (size_t index : m_executedEventPasses) {
        for (const auto &event : m_pythonGraphDesc.passes[index].events) {
            if (!FindActiveEventInterest(interests, event.label))
                continue;
            for (const auto &[name, resource] : event.buffers)
                captured.insert(m_graphBufferOwners.at(resource)->GetBuffer());
        }
    }
    if (captured.empty())
        return true;

    // Check replacement headroom once for every actual captured writer.
    // Read-only imports do not acquire another allocation on the next draw.
    std::vector<rhi::BufferHandle> retained;
    retained.reserve(captured.size());
    for (const auto &name : m_graphBufferWrites) {
        const auto owner = m_graphBufferOwners.find(name);
        if (owner == m_graphBufferOwners.end())
            continue;
        const auto handle = owner->second->GetBuffer();
        if (captured.erase(handle))
            retained.push_back(handle);
    }
    return retained.empty() || m_vkCore->GetDeviceContext().GetRhiDevice().CanRetainGraphBuffers(retained);
}

bool SceneRenderGraph::AdmitGraphEvents(const GraphEventInterests &interests)
{
    if (!m_graphBuilt)
        return false;
    m_graphStorageDeferred = !CanRetainEventBuffers(interests);
    if (m_graphStorageDeferred && !m_graphBuffersPublished) {
        // Unrecorded buffers do not need to wait for GPU retirement.
        m_renderGraph->Reset();
        RetireWorkInstances();
        m_graphBuilt = false;
        m_needsRebuild = true;
    }
    return !m_graphStorageDeferred;
}

void SceneRenderGraph::PublishGraphBuffers()
{
    const auto publish = [](const auto &owner) {
        if (owner->GetAllocation()->IsGraphWritable())
            owner->PublishStorage();
    };
    for (const auto &[name, owner] : m_graphBufferOwners)
        publish(owner);
    for (const auto &[name, instance] : m_workInstances)
        for (const auto &owner : instance->GetBuffers())
            publish(owner);
    m_graphBuffersPublished = true;
}

bool SceneRenderGraph::ValidateOutputTextureAliases(const RenderGraphDescription &description) const
{
    if (!m_outputGeneration)
        return true;
    if (const auto *alias = FindGraphOutputTextureAlias(description, *m_outputGeneration)) {
        INXLOG_ERROR("A camera target attachment cannot also be imported as a separate persistent graph resource: ",
                     alias->name);
        return false;
    }
    return true;
}

void SceneRenderGraph::ReleaseViewGpuResources() noexcept
{
    RetireWorkInstances();
    m_viewGpuResourcesReady = false;
    m_fullscreenRenderer.Destroy();
    m_sceneDepthResolver.Destroy();
    m_forwardPlusGeometryGrid.Shutdown();
    m_forwardPlusParticleGrid.Shutdown();
    if (m_vkCore) {
        auto &rhiDevice = m_vkCore->GetDeviceContext().GetRhiDevice();
        auto &descriptorManager = rhiDevice.GetDescriptorManager();
        for (auto &frame : m_perViewFrames) {
            rhiDevice.Release(frame.globals.group);
            descriptorManager.Retire(frame.globals.descriptor);
            rhiDevice.Release(frame.globals.buffer);
            rhiDevice.Release(frame.geometryGroup);
            rhiDevice.Release(frame.particleGroup);
            rhiDevice.Release(frame.editorOverlayGroup);
            descriptorManager.Retire(frame.geometryDescriptor);
            descriptorManager.Retire(frame.particleDescriptor);
            descriptorManager.Retire(frame.editorOverlayDescriptor);
            rhiDevice.Release(frame.lighting);
            rhiDevice.Release(frame.cameraMatrix);
            rhiDevice.Release(frame.editorOverlayCameraMatrix);
            frame = {};
        }
        rhiDevice.Release(m_perViewLayout);
    }
    m_perViewLayout = {};
    m_cameraCanonicalLights.Shutdown();
}

void SceneRenderGraph::Destroy()
{
    CancelBufferCaptures("render view was destroyed");
    ReleaseParticleViewBindings();
    if (m_particleViewDiagnosticState) {
        std::scoped_lock lock(m_particleViewDiagnosticState->mutex);
        for (const auto &request : m_pendingParticleViewDiagnostics) {
            auto &snapshot = m_particleViewDiagnosticState->snapshots[request.requestId];
            if (snapshot.status == particle::GpuParticleViewDiagnosticStatus::Pending) {
                snapshot.status = particle::GpuParticleViewDiagnosticStatus::Failed;
                snapshot.error = "Particle view render graph was destroyed before diagnostics were recorded";
            }
        }
    }
    m_pendingParticleViewDiagnostics.clear();
    m_particleCullers.clear();
    m_particleSorters.clear();
    RetireImportedTextureAssets();
    m_importedTextureBindings.clear();
    ReleaseViewGpuResources();
    m_temporalHistories.clear();
    if (m_vkCore)
        m_vkCore->DestroyShadowCameraResources(m_shadowCameraResourceId);
    m_shadowCameraResourceId = 0;
    m_parameterBlocks.clear();
    m_submittedParameterBlocks.clear();
    ++m_parameterBlockGeneration;
    m_submittedParameterBlockGeneration = m_parameterBlockGeneration;

    if (m_renderGraph) {
        m_renderGraph->Destroy();
    }
    m_importedColorTarget = {};
    m_importedDepthTarget = {};
    m_outlineRenderer = nullptr;
    m_outlinePassesEnabled = false;
    m_outlinePipelineFailureReported = false;
    m_graphBuilt = false;
    m_vkCore = nullptr;
    m_sceneTarget = nullptr;
    m_screenTarget = nullptr;
    m_viewportWorkTarget.reset();
    m_viewportGeneration.reset();
    m_viewport.Reset();
    m_outputTarget.reset();
    m_outputGeneration.reset();
    m_outputTexture.reset();
}

uint64_t SceneRenderGraph::RequestParticleViewDiagnostics(uint64_t graphInstanceId)
{
    if (!m_particleViewDiagnosticState)
        return 0;
    uint64_t requestId = m_nextParticleViewDiagnosticRequestId++;
    if (requestId == 0)
        requestId = m_nextParticleViewDiagnosticRequestId++;

    particle::GpuParticleViewDiagnosticSnapshot snapshot;
    snapshot.requestId = requestId;
    snapshot.graphInstanceId = graphInstanceId;
    if (!m_vkCore || !m_particleDrawRegistry || graphInstanceId == 0) {
        snapshot.status = particle::GpuParticleViewDiagnosticStatus::Failed;
        snapshot.error = "Particle view diagnostics require an initialized render graph and a resident particle graph";
    } else {
        snapshot.status = particle::GpuParticleViewDiagnosticStatus::Pending;
        m_pendingParticleViewDiagnostics.push_back({requestId, graphInstanceId});
    }

    std::scoped_lock lock(m_particleViewDiagnosticState->mutex);
    if (m_particleViewDiagnosticState->snapshots.size() >= 128) {
        auto oldest = m_particleViewDiagnosticState->snapshots.end();
        for (auto it = m_particleViewDiagnosticState->snapshots.begin();
             it != m_particleViewDiagnosticState->snapshots.end(); ++it) {
            if (it->second.status == particle::GpuParticleViewDiagnosticStatus::Pending)
                continue;
            if (oldest == m_particleViewDiagnosticState->snapshots.end() || it->first < oldest->first)
                oldest = it;
        }
        if (oldest != m_particleViewDiagnosticState->snapshots.end())
            m_particleViewDiagnosticState->snapshots.erase(oldest);
    }
    m_particleViewDiagnosticState->snapshots[requestId] = std::move(snapshot);
    return requestId;
}

particle::GpuParticleViewDiagnosticSnapshot SceneRenderGraph::QueryParticleViewDiagnostics(uint64_t requestId) const
{
    if (!m_particleViewDiagnosticState)
        return {};
    std::scoped_lock lock(m_particleViewDiagnosticState->mutex);
    const auto found = m_particleViewDiagnosticState->snapshots.find(requestId);
    return found != m_particleViewDiagnosticState->snapshots.end() ? found->second
                                                                   : particle::GpuParticleViewDiagnosticSnapshot{};
}

void SceneRenderGraph::RecordParticleViewDiagnostics(VkCommandBuffer commandBuffer)
{
    if (m_pendingParticleViewDiagnostics.empty() || !m_vkCore || !m_particleDrawRegistry ||
        commandBuffer == VK_NULL_HANDLE) {
        return;
    }

    auto requests = std::move(m_pendingParticleViewDiagnostics);
    m_pendingParticleViewDiagnostics.clear();
    auto &device = m_vkCore->GetDeviceContext().GetRhiDevice();
    vk::VulkanTransferCommandContext transferContext;
    const auto transfer = device.MakeTransferCommandEncoder(transferContext, commandBuffer);
    const auto entries = m_particleDrawRegistry->SnapshotShared(std::numeric_limits<int32_t>::min(),
                                                                std::numeric_limits<int32_t>::max());

    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                         &barrier, 0, nullptr, 0, nullptr);

    for (const auto &request : requests) {
        struct OutputCapture
        {
            particle::GpuParticleViewOutputDiagnostic diagnostic;
            uint64_t dispatchOffset = 0;
            uint64_t drawOffset = 0;
            std::shared_ptr<particle::ParticleGpuCuller> culler;
        };

        std::vector<OutputCapture> captures;
        for (const auto &entry : *entries) {
            if (entry.graphInstanceId != request.graphInstanceId)
                continue;
            const auto culler = m_particleCullers.find(entry.id);
            if (culler == m_particleCullers.end() || !culler->second || !culler->second->IsValid())
                continue;
            OutputCapture capture;
            capture.diagnostic.outputId = entry.id;
            capture.diagnostic.emitterId = entry.emitterId;
            capture.diagnostic.emitterIndex = entry.emitterIndex;
            capture.diagnostic.outputStableId = entry.outputStableId;
            capture.diagnostic.capacity = entry.capacity;
            capture.diagnostic.residentViewBindingCount = static_cast<uint32_t>(entry.renderer->ViewBindingCount());
            capture.diagnostic.cullMode = entry.cullMode;
            capture.diagnostic.sortMode = entry.semantics.sortMode;
            const auto sorter = m_particleSorters.find(entry.id);
            capture.diagnostic.sorterAllocated =
                sorter != m_particleSorters.end() && sorter->second && sorter->second->IsValid();
            capture.culler = culler->second;
            captures.push_back(std::move(capture));
        }
        std::sort(captures.begin(), captures.end(), [](const auto &lhs, const auto &rhs) {
            if (lhs.diagnostic.emitterIndex != rhs.diagnostic.emitterIndex)
                return lhs.diagnostic.emitterIndex < rhs.diagnostic.emitterIndex;
            return lhs.diagnostic.outputStableId < rhs.diagnostic.outputStableId;
        });

        if (captures.empty()) {
            std::scoped_lock lock(m_particleViewDiagnosticState->mutex);
            auto &snapshot = m_particleViewDiagnosticState->snapshots[request.requestId];
            snapshot.status = particle::GpuParticleViewDiagnosticStatus::Failed;
            snapshot.error = "Particle view contains no resident outputs for the requested graph";
            continue;
        }

        constexpr uint64_t BytesPerOutput = sizeof(particle::GpuParticleCullDispatchState) + 16u;
        const uint64_t totalBytes = captures.size() * BytesPerOutput;
        const auto readbackHandle =
            device.CreateBuffer({totalBytes, rhi::BufferUsageFlags::TransferDestination, rhi::BufferMemory::Readback});
        if (!readbackHandle.IsValid()) {
            std::scoped_lock lock(m_particleViewDiagnosticState->mutex);
            auto &snapshot = m_particleViewDiagnosticState->snapshots[request.requestId];
            snapshot.status = particle::GpuParticleViewDiagnosticStatus::Failed;
            snapshot.error = "Failed to allocate particle view diagnostic readback buffer";
            continue;
        }
        auto readback = std::make_shared<rhi::BufferResource>(device, readbackHandle, totalBytes);
        uint64_t offset = 0;
        for (auto &capture : captures) {
            capture.dispatchOffset = offset;
            transfer.CopyBuffer(capture.culler->SortDispatchBuffer(), readbackHandle,
                                {0, offset, sizeof(particle::GpuParticleCullDispatchState)});
            offset += sizeof(particle::GpuParticleCullDispatchState);
            capture.drawOffset = offset;
            transfer.CopyBuffer(capture.culler->DrawIndirectBuffer(), readbackHandle, {0, offset, 16});
            offset += 16;
        }

        const auto state = m_particleViewDiagnosticState;
        auto *readbackDevice = static_cast<rhi::Device *>(&device);
        m_vkCore->GetRetirementQueue().Retire([state, readback, captures = std::move(captures), request,
                                               readbackDevice]() mutable {
            std::vector<uint8_t> bytes(static_cast<size_t>(readback->GetByteSize()));
            particle::GpuParticleViewDiagnosticSnapshot result;
            result.requestId = request.requestId;
            result.graphInstanceId = request.graphInstanceId;
            if (!readbackDevice->ReadBuffer(readback->GetBuffer(), 0, bytes.data(), bytes.size())) {
                result.status = particle::GpuParticleViewDiagnosticStatus::Failed;
                result.error = "Failed to read completed particle view diagnostic buffer";
            } else {
                result.status = particle::GpuParticleViewDiagnosticStatus::Completed;
                for (auto &capture : captures) {
                    particle::GpuParticleCullDispatchState dispatch{};
                    std::array<uint32_t, 4> draw{};
                    std::memcpy(&dispatch, bytes.data() + capture.dispatchOffset, sizeof(dispatch));
                    std::memcpy(draw.data(), bytes.data() + capture.drawOffset, sizeof(draw));
                    capture.diagnostic.sourceCount = dispatch.sourceCount;
                    capture.diagnostic.visibleCount = draw[1];
                    capture.diagnostic.sortGroupCountX = dispatch.groupCountX;
                    capture.diagnostic.boundsValid = (dispatch.flags & 1u) != 0u;
                    capture.diagnostic.coarseRejected = capture.diagnostic.boundsValid && (dispatch.flags & 2u) == 0u;
                    capture.diagnostic.cullMode = (dispatch.flags & 4u) != 0u
                                                      ? particle::GpuParticleCullMode::RibbonSegments
                                                      : particle::GpuParticleCullMode::Instances;
                    capture.diagnostic.drawVertexCount = draw[0];
                    capture.diagnostic.drawInstanceCount = draw[1];
                    result.outputs.push_back(std::move(capture.diagnostic));
                }
            }
            std::scoped_lock lock(state->mutex);
            const auto found = state->snapshots.find(request.requestId);
            if (found != state->snapshots.end() &&
                found->second.status == particle::GpuParticleViewDiagnosticStatus::Pending) {
                found->second = std::move(result);
            }
        });
    }
}

VkDescriptorSet SceneRenderGraph::GetPerViewDescriptorSet() const
{
    if (!m_vkCore)
        return VK_NULL_HANDLE;
    uint32_t frameIdx = m_vkCore->GetCurrentFrameSlot() % EngineConfig::kFrameSlotCapacity;
    return m_perViewFrames[frameIdx].GeometrySet();
}

rhi::BindGroupHandle SceneRenderGraph::GetPerViewBindGroup() const
{
    if (!m_vkCore)
        return {};
    const uint32_t frameIndex = m_vkCore->GetCurrentFrameSlot() % EngineConfig::kFrameSlotCapacity;
    return m_perViewFrames[frameIndex].geometryGroup;
}

rhi::BindGroupHandle SceneRenderGraph::GetEditorOverlayBindGroup() const
{
    if (!m_vkCore)
        return {};
    const uint32_t frameIndex = m_vkCore->GetCurrentFrameSlot() % EngineConfig::kFrameSlotCapacity;
    return m_perViewFrames[frameIndex].editorOverlayGroup;
}

vk::ViewGlobalsBinding &SceneRenderGraph::GetViewGlobalsBinding()
{
    return m_perViewFrames[m_vkCore->GetCurrentFrameSlot()].globals;
}

void SceneRenderGraph::SetCameraInvertCulling(bool invert)
{
    if (m_cameraInvertCulling == invert)
        return;
    m_cameraInvertCulling = invert;
    InvalidateTemporalHistory();
}

void SceneRenderGraph::SetCachedCameraVP(const Camera *camera, const glm::mat4 &view, const glm::mat4 &proj)
{
    const uint64_t historyRevision = camera ? camera->GetTemporalHistoryRevision() : 0;
    bool cameraCut =
        m_hasCachedCameraVP && (camera != m_cachedCamera || historyRevision != m_cachedCameraHistoryRevision);
    if (m_hasCachedCameraVP && !cameraCut) {
        const glm::mat4 previousInverseView = glm::inverse(m_cachedView);
        const glm::mat4 currentInverseView = glm::inverse(view);
        const glm::vec3 previousPosition = glm::vec3(previousInverseView[3]);
        const glm::vec3 currentPosition = glm::vec3(currentInverseView[3]);
        const glm::vec3 previousForward = -glm::normalize(glm::vec3(previousInverseView[2]));
        const glm::vec3 currentForward = -glm::normalize(glm::vec3(currentInverseView[2]));

        float projectionDelta = 0.0f;
        for (glm::length_t column = 0; column < 4; ++column) {
            for (glm::length_t row = 0; row < 4; ++row) {
                projectionDelta =
                    std::max(projectionDelta, std::abs(proj[column][row] - m_cachedUnjitteredProj[column][row]));
            }
        }

        constexpr float kCameraCutDistance = 10.0f;
        constexpr float kCameraCutDirectionCosine = 0.70710678f;
        const glm::vec3 cameraDelta = currentPosition - previousPosition;
        cameraCut = glm::dot(cameraDelta, cameraDelta) > kCameraCutDistance * kCameraCutDistance ||
                    glm::dot(previousForward, currentForward) < kCameraCutDirectionCosine || projectionDelta > 1e-3f;
    }
    if (cameraCut)
        InvalidateTemporalHistory();

    m_cachedCamera = camera;
    m_cachedCameraHistoryRevision = historyRevision;
    m_cachedView = view;
    m_drawView = view;
    m_cachedUnjitteredProj = proj;
    m_hasCachedCameraVP = true;
    UpdateTemporalProjection();
}

void SceneRenderGraph::UpdateTemporalProjection()
{
    m_temporalJitterNdc = m_hasCachedCameraVP && m_pythonGraphValid && m_pythonGraphDesc.temporalJitter
                              ? ComputeTemporalJitterNdc(m_temporalSampleIndex, m_width, m_height)
                              : glm::vec2(0.0f);
    m_cachedProj = ProjectionForPass(m_cachedUnjitteredProj, m_temporalJitterNdc, false);
}

glm::vec2 SceneRenderGraph::ComputeTemporalJitterNdc(uint32_t sampleIndex, uint32_t width, uint32_t height)
{
    if (width == 0 || height == 0)
        return glm::vec2(0.0f);

    const auto halton = [](uint32_t index, uint32_t base) {
        float value = 0.0f;
        float fraction = 1.0f;
        while (index > 0) {
            fraction /= static_cast<float>(base);
            value += fraction * static_cast<float>(index % base);
            index /= base;
        }
        return value;
    };

    const uint32_t sequenceIndex = sampleIndex % kTemporalJitterSampleCount + 1u;
    const glm::vec2 pixelOffset{halton(sequenceIndex, 2u) - 0.5f, halton(sequenceIndex, 3u) - 0.5f};
    return {2.0f * pixelOffset.x / static_cast<float>(width), 2.0f * pixelOffset.y / static_cast<float>(height)};
}

glm::mat4 SceneRenderGraph::ApplyTemporalJitter(const glm::mat4 &projection, const glm::vec2 &jitterNdc)
{
    glm::mat4 result = projection;
    for (glm::length_t column = 0; column < 4; ++column) {
        result[column][0] += jitterNdc.x * projection[column][3];
        result[column][1] += jitterNdc.y * projection[column][3];
    }
    return result;
}

glm::mat4 SceneRenderGraph::ProjectionForPass(const glm::mat4 &projection, const glm::vec2 &jitterNdc,
                                              bool editorOverlay)
{
    return editorOverlay ? projection : ApplyTemporalJitter(projection, jitterNdc);
}

bool SceneRenderGraph::StageCameraMatrices(const glm::mat4 &view, const glm::mat4 &proj,
                                           const glm::mat4 *previousViewProj)
{
    m_cameraMatricesStaged = false;
    if (!m_vkCore || !m_cachedCamera || !EnsureViewGpuResources())
        return false;
    const uint32_t frameIndex = m_vkCore->GetCurrentFrameSlot() % EngineConfig::kFrameSlotCapacity;
    auto &frame = m_perViewFrames[frameIndex];
    if (!frame.cameraMatrix.IsValid())
        return false;

    UniformBufferObject camera{};
    camera.model = glm::mat4(1.0f);
    camera.view = view;
    camera.proj = proj;
    camera.previousViewProj =
        previousViewProj ? *previousViewProj : (m_cameraHistoryValid ? m_previousViewProj : proj * view);
    camera.inverseViewProj = glm::inverse(proj * view);
    const float nearClip = m_cachedCamera->GetNearClip();
    const float farClip = m_cachedCamera->GetFarClip();
    camera.projectionParams = glm::vec4(nearClip, farClip, 1.0f / farClip, nearClip / farClip);
    camera.zBufferParams = depth::LinearizationParams(nearClip, farClip);
    auto &rhiDevice = m_vkCore->GetDeviceContext().GetRhiDevice();
    EngineGlobalsUBO globals = m_vkCore->GetStagedGlobals();
    m_stagedShaderTime = globals.time.x;
    globals.frameParams.w = m_cameraHistoryValid ? m_previousShaderTime : m_stagedShaderTime;
    const float width = static_cast<float>(m_width);
    const float height = static_cast<float>(m_height);
    globals.screenParams = glm::vec4(width, height, 1.0f / width, 1.0f / height);
    globals.worldSpaceCameraPos = glm::vec4(glm::vec3(glm::inverse(view)[3]), 1.0f);
    globals.projectionParams = camera.projectionParams;
    globals.zBufferParams = camera.zBufferParams;
    globals.viewOptions = glm::uvec4(static_cast<uint32_t>(m_cameraDithering), static_cast<uint32_t>(m_cameraStopNaNs),
                                     static_cast<uint32_t>(m_renderView.kind == rhi::RenderViewKind::Scene), 0u);
    if (!rhiDevice.WriteBuffer(frame.globals.buffer, 0, &globals, sizeof(globals))) {
        INXLOG_ERROR("SceneRenderGraph: failed to upload camera-local globals UBO");
        return false;
    }
    if (!rhiDevice.WriteBuffer(frame.cameraMatrix, 0, &camera, sizeof(camera))) {
        INXLOG_ERROR("SceneRenderGraph: failed to upload camera-local matrix UBO");
        return false;
    }
    UniformBufferObject overlayCamera = camera;
    if (m_renderView.kind == rhi::RenderViewKind::Scene) {
        // Editor overlays run after the temporal resolve. Their own subpixel
        // jitter would otherwise reappear as icon/grid shimmer every frame.
        overlayCamera.proj = GetEditorOverlayProjection();
        overlayCamera.previousViewProj = overlayCamera.proj * view;
        overlayCamera.inverseViewProj = glm::inverse(overlayCamera.proj * view);
    }
    if (!rhiDevice.WriteBuffer(frame.editorOverlayCameraMatrix, 0, &overlayCamera, sizeof(overlayCamera))) {
        INXLOG_ERROR("SceneRenderGraph: failed to upload editor overlay camera UBO");
        return false;
    }
    m_cameraMatricesStaged = true;
    return true;
}

void SceneRenderGraph::StageCameraLighting(Scene *scene, Camera *camera, const glm::vec3 &cameraPosition,
                                           const ShaderLightingUBO &environmentLighting)
{
    if (!m_vkCore || !scene)
        return;

    m_cameraLightCollector.CollectLights(scene, cameraPosition);
    // environmentLighting values are already linear (converted once by the
    // scene collector's SetAmbient* setters) — copy them verbatim to avoid a
    // second sRGB -> linear conversion.
    m_cameraLightCollector.SetAmbientLinear(environmentLighting.ambientSkyColor,
                                            environmentLighting.ambientEquatorColor,
                                            environmentLighting.ambientGroundColor);
    if (m_renderView.kind != rhi::RenderViewKind::Preview) {
        const lighting::ShadowDepthRange visibleDepthRange =
            m_hasCachedDrawCalls ? VisibleShadowDepthRange(camera, m_cachedRenderers.DrawCalls())
                                 : lighting::ShadowDepthRange{};
        m_cameraLightCollector.ComputeShadowVP(scene, cameraPosition, static_cast<float>(GetShadowMapResolution()),
                                               camera, visibleDepthRange);
    }
    m_cameraLightCollector.BuildShaderLightingUBO();
}

bool SceneRenderGraph::UploadCameraLighting()
{
    const uint32_t frameIndex = m_vkCore->GetCurrentFrameSlot() % EngineConfig::kFrameSlotCapacity;
    auto &frame = m_perViewFrames[frameIndex];
    auto &rhiDevice = m_vkCore->GetDeviceContext().GetRhiDevice();
    const ShaderLightingUBO &lighting = m_cameraLightCollector.GetShaderLightingUBO();
    if (!frame.lighting.IsValid() || !rhiDevice.WriteBuffer(frame.lighting, 0, &lighting, sizeof(lighting))) {
        INXLOG_ERROR("SceneRenderGraph: failed to upload camera-local LightingUBO");
        return false;
    }
    const auto previous = m_cameraCanonicalLights.Frame(frameIndex).buffer;
    if (!m_cameraCanonicalLights.Update(frameIndex, m_cameraLightCollector.GetCanonicalLightSnapshot())) {
        INXLOG_ERROR("SceneRenderGraph: failed to upload camera-local canonical lights");
        return false;
    }
    if (m_cameraCanonicalLights.Frame(frameIndex).buffer != previous) {
        // Both Forward+ and authored light-list passes can import this slot.
        m_needsRebuild = true;
        m_graphBuilt = false;
    }
    return true;
}

// ============================================================================
// Resource management
// ============================================================================

// ============================================================================
// RenderGraph topology defined from Python
// ============================================================================

void SceneRenderGraph::SetOutlineRenderer(OutlineRenderer *renderer)
{
    const bool enabled = m_renderView.kind == rhi::RenderViewKind::Scene && renderer && renderer->IsReady() &&
                         renderer->HasActiveOutline();
    if (m_outlineRenderer != renderer)
        m_outlinePipelineFailureReported = false;
    m_outlineRenderer = renderer;
    if (m_outlinePassesEnabled == enabled)
        return;

    m_outlinePassesEnabled = enabled;
    m_outlinePipelineFailureReported = false;
    m_needsRebuild = true;
}

MaterialPassPipelineDescriptor SceneRenderGraph::GetEditorOverlayMaterialPass() const
{
    MaterialPassPipelineDescriptor descriptor;
    descriptor.colorFormats = {m_renderView.colorFormat};
    descriptor.depthFormat = m_renderView.depthFormat;
    descriptor.samples = m_renderView.samples;
    descriptor.depthReadOnly = descriptor.depthFormat != rhi::PixelFormat::Undefined;
    descriptor.invertCulling = m_cameraInvertCulling;
    return descriptor;
}

namespace
{
const GraphTextureDesc &FullscreenTexture(const RenderGraphDescription &graph, const std::string &name)
{
    const auto texture = std::find_if(graph.textures.begin(), graph.textures.end(),
                                      [&](const auto &value) { return value.name == name; });
    if (texture == graph.textures.end())
        throw std::invalid_argument("Fullscreen texture is not declared: " + name);
    return *texture;
}

std::vector<std::string> ImplicitFullscreenInputs(const GraphPassDesc &pass, const RenderGraphDescription &graph)
{
    std::vector<std::string> inputs;
    for (bool viewColor : {false, true})
        for (const auto &name : pass.readTextures) {
            const auto &texture = FullscreenTexture(graph, name);
            if (!texture.IsDepth() && texture.IsViewColor() == viewColor)
                inputs.push_back(name);
        }
    return inputs;
}

void ValidateFullscreenHistories(const FullscreenProgram &program, const FullscreenParameters &parameters,
                                 const GraphCommandDesc &command, const GraphPassDesc &pass,
                                 const RenderGraphDescription &graph)
{
    if (parameters.histories.empty())
        return;
    const auto schema = nlohmann::json::parse(program.schema);
    const auto implicit =
        command.inputBindings.empty() ? ImplicitFullscreenInputs(pass, graph) : std::vector<std::string>{};
    for (const auto &history : parameters.histories) {
        const auto name = schema.at("inputs").at(history.input).at("name").get<std::string>();
        std::string source;
        if (command.inputBindings.empty()) {
            if (history.input >= implicit.size())
                throw std::invalid_argument("History2D input is not bound: " + command.shaderName + "." + name);
            source = implicit[history.input];
        } else {
            const auto binding = std::find_if(command.inputBindings.begin(), command.inputBindings.end(),
                                              [&](const auto &value) { return value.first == name; });
            if (binding == command.inputBindings.end())
                throw std::invalid_argument("History2D input is not bound: " + command.shaderName + "." + name);
            source = binding->second;
        }
        if (FullscreenTexture(graph, source).role != GraphTextureRole::TemporalRead)
            throw std::invalid_argument("History2D input requires a temporal history read resource: " +
                                        command.shaderName + "." + name + " <- " + source);
    }
}
} // namespace

bool SceneRenderGraph::ApplyPythonGraph(const RenderGraphDescription &desc)
{
    if (desc.sourceRevision != 0 && m_rejectedFullscreenSourceRevision == desc.sourceRevision)
        return false;
    if (!m_vkCore || !m_sceneTarget) {
        INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: Not initialized");
        return false;
    }

    const auto rejectGraph = [&](RenderGraphDescription attempted) {
        CancelBufferCaptures("render graph was rejected");
        RetireWorkInstances();
        if (m_renderGraph)
            m_renderGraph->Reset();
        m_liveBufferInputs.clear();
        m_pythonGraphDesc = std::move(attempted);
        m_hasPythonGraph = true;
        m_pythonGraphValid = false;
        m_graphBuilt = false;
        m_needsRebuild = true;
        m_pythonGraphSourceRevision = 0;
        m_pythonCallbacks.clear();
        m_pythonMaterialPasses.clear();
        m_materialTextureReads.clear();
        m_materialBufferReads.clear();
        m_hasShadowCasterPass = false;
        m_parameterBlocks.clear();
        m_submittedParameterBlocks.clear();
        ++m_resourceAccessRevision;
        return false;
    };
    RenderGraphDescription normalizedDesc;
    try {
        normalizedDesc = NormalizeRenderGraphForOutput(desc, HasOutputTexture(), m_renderView.samples);
        for (auto &pass : normalizedDesc.passes) {
            for (auto &command : pass.commands) {
                if (command.type != GraphCommandType::FullscreenQuad)
                    continue;
                if (const auto program = m_vkCore->AcquireFullscreenProgram(command.shaderName)) {
                    m_authorShaderGuids.insert(program->assetGuid);
                    const auto prepared = FullscreenRenderer::PrepareParameters(*program, command.parameters);
                    ValidateFullscreenHistories(*program, prepared, command, pass, normalizedDesc);
                    for (size_t index = 0; index < command.parameters.size(); ++index)
                        command.parameters[index].second = prepared.values[index];
                } else {
                    if (command.parameters.size() > 32)
                        throw std::invalid_argument("Native fullscreen push constants exceed 128 bytes");
                    for (auto &[name, value] : command.parameters)
                        value = value.ScalarFloat();
                }
            }
        }
    } catch (const std::invalid_argument &error) {
        INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: ", error.what());
        m_rejectedFullscreenSourceRevision = desc.sourceRevision;
        return rejectGraph(desc);
    }
    m_needsRebuild = m_needsRebuild || m_rejectedFullscreenSourceRevision.has_value();
    m_rejectedFullscreenSourceRevision.reset();
    for (auto &buffer : normalizedDesc.buffers)
        if (buffer.input)
            buffer.computeBuffer = buffer.input->GetValue();
    const auto callbackTarget = GetEditorOverlayMaterialPass().RenderingSignature();
    const bool topologyChanged = !m_pythonGraphValid || !RenderGraphTopologyEquals(normalizedDesc, m_pythonGraphDesc);
    const bool callbackContractChanged =
        m_pythonCallbackTarget != callbackTarget || m_pythonCallbackInvertCulling != m_cameraInvertCulling;
    const auto publishParameterValues = [&] {
        std::vector<GraphParameterBlockUpdate> updates;
        for (const auto &pass : normalizedDesc.passes) {
            const auto *command = pass.PrimaryCommand();
            if (command && !command->parameterBlock.empty())
                updates.push_back({command->parameterBlock, 0, command->parameters});
        }
        UpdateParameterBlocks(updates);
    };
    if (!normalizedDesc.RenderTextureLayoutsCurrent()) {
        INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: RenderTexture attachment layout changed");
        return rejectGraph(std::move(normalizedDesc));
    }
    if (!ValidateOutputTextureAliases(normalizedDesc))
        return rejectGraph(std::move(normalizedDesc));
    if (topologyChanged || callbackContractChanged ||
        (desc.sourceRevision != 0 && desc.sourceRevision != m_pythonGraphSourceRevision)) {
        CancelBufferCaptures("render graph definition changed");
        ++m_bufferGeneration;
    }
    if (!topologyChanged && !callbackContractChanged) {
        // Replaying an already-applied description must not clobber live
        // parameter blocks: effect edits arrive through UpdateParameterBlocks
        // *after* the description was built, so its parameter values
        // are only authoritative when a genuinely new Python build
        // (fresh source revision) is being applied.
        if (desc.sourceRevision == 0 || desc.sourceRevision != m_pythonGraphSourceRevision) {
            publishParameterValues();
        }
        if (desc.sourceRevision != 0) {
            m_pythonGraphSourceRevision = desc.sourceRevision;
            m_pythonGraphDesc.sourceRevision = desc.sourceRevision;
        }
        return true;
    }

    if (!ValidateRenderGraphDescription(normalizedDesc, m_renderView)) {
        return rejectGraph(std::move(normalizedDesc));
    }

    // A cold asset can postpone the replacement graph's build.
    m_graphBuilt = false;
    RetireWorkInstances();
    m_renderGraph->Reset();

    if (topologyChanged) {
        ++m_resourceAccessRevision;
        std::unordered_map<std::string, RuntimeParameterBlock> blocks;
        for (const auto &pass : normalizedDesc.passes) {
            const GraphCommandDesc *command = pass.PrimaryCommand();
            if (!command || command->parameterBlock.empty())
                continue;
            RuntimeParameterBlock block;
            auto values = std::make_shared<std::vector<ShaderParameterValue>>();
            const bool nativePushConstants = !m_vkCore->AcquireFullscreenProgram(command->shaderName);
            if (nativePushConstants)
                block.scalarPushConstants.emplace();
            block.names.reserve(command->parameters.size());
            values->reserve(command->parameters.size());
            for (size_t i = 0; i < command->parameters.size(); ++i) {
                block.names.push_back(command->parameters[i].first);
                values->push_back(command->parameters[i].second);
                if (nativePushConstants)
                    block.scalarPushConstants->values[i] = values->back().Read<float>();
            }
            block.values = std::move(values);
            blocks.emplace(command->parameterBlock, std::move(block));
        }
        m_parameterBlocks = std::move(blocks);
        ++m_parameterBlockGeneration;
    } else if (desc.sourceRevision != m_pythonGraphSourceRevision) {
        publishParameterValues();
    }

    m_pythonCallbacks.clear();
    m_pythonMaterialPasses.clear();
    m_hasShadowCasterPass = false;

    InxVkCoreModular *vkCore = m_vkCore;
    for (const auto &passDesc : normalizedDesc.passes) {
        const GraphCommandDesc *command = passDesc.PrimaryCommand();
        if (!command) {
            m_pythonCallbacks[passDesc.name] = [](vk::RenderContext &, uint32_t, uint32_t) {};
            continue;
        }
        if (passDesc.type != GraphPassType::Raster) {
            m_pythonCallbacks[passDesc.name] = [](vk::RenderContext &, uint32_t, uint32_t) {};
            continue;
        }
        const auto commandType = command->type;
        if (commandType == GraphCommandType::DrawShadowCasters) {
            m_hasShadowCasterPass = true;
        }
        const int queueMin = command->queueMin;
        const int queueMax = command->queueMax;
        const int screenUIListIndex = command->screenUIList;
        const auto worldUILayerMask = command->worldUILayerMask;
        const int lightIndex = command->lightIndex;
        const std::string sortMode = command->sortMode;
        const std::string overrideMaterial = command->overrideMaterial;
        const auto rendererSelection = command->rendererSelection;
        const std::string passTag = command->passTag;
        const GraphMaterialFilter materialFilter = command->materialFilter;
        MaterialPassPipelineDescriptor materialPass;
        if (commandType == GraphCommandType::DrawRenderers || commandType == GraphCommandType::DrawShadowCasters ||
            commandType == GraphCommandType::DrawSkybox || commandType == GraphCommandType::DrawWorldUI) {
            materialPass = ResolveMaterialPass(normalizedDesc, passDesc, m_renderView, m_cameraInvertCulling);
            m_pythonMaterialPasses[passDesc.name] = materialPass;
        }

        m_pythonCallbacks[passDesc.name] = [this, vkCore, commandType, queueMin, queueMax, screenUIListIndex,
                                            worldUILayerMask, lightIndex, sortMode, overrideMaterial, rendererSelection,
                                            passTag, materialFilter,
                                            materialPass](vk::RenderContext &ctx, uint32_t w, uint32_t h) {
            switch (commandType) {
            case GraphCommandType::DrawRenderers:
                vkCore->DrawSceneFiltered(ctx.GetCommandBuffer(), w, h, GetPerViewBindGroup(), GetViewGlobalsBinding(),
                                          m_drawView, queueMin, queueMax, sortMode, overrideMaterial, passTag,
                                          &materialPass, materialFilter, rendererSelection.get(), 0, false,
                                          materialPass.target == ShaderCompileTarget::Motion ? &m_motionHistory
                                                                                             : nullptr);
                break;
            case GraphCommandType::DrawSkybox: {
                const int32_t skyboxQueue = EngineConfig::Get().skyboxQueue;
                vkCore->DrawSceneFiltered(ctx.GetCommandBuffer(), w, h, GetPerViewBindGroup(), GetViewGlobalsBinding(),
                                          m_drawView, skyboxQueue, skyboxQueue, "", "", "__infernux_internal_skybox",
                                          &materialPass);
                break;
            }
            case GraphCommandType::DrawShadowCasters:
                // Shadow caster pass: draw filtered objects using shadow pipeline
                // with lightVP from SceneLightCollector. The shadow pipeline is
                // lazily created inside DrawShadowCasters().
                vkCore->DrawShadowCasters(
                    ctx.GetCommandBuffer(), w, h, queueMin, queueMax, m_shadowCameraResourceId,
                    m_vkCore->GetDeviceContext().GetRhiDevice().Resolve(GetViewGlobalsBinding().buffer),
                    m_cameraLightCollector.GetShadowFrame(), lightIndex);
                break;
            case GraphCommandType::DrawWorldUI:
                if (m_screenUIRenderer)
                    m_screenUIRenderer->RenderWorld(
                        ctx.GetCommandBuffer(), w, h, m_cachedProj * m_drawView, materialPass.RenderingSignature(),
                        vkCore->GetCurrentFrameSlot(),
                        worldUILayerMask & (m_cachedCamera ? m_cachedCamera->GetCullingMask() : 0xffffffffu),
                        m_drawView, m_cachedProj, m_hasBillboardView ? m_billboardView : m_drawView);
                break;
            case GraphCommandType::DrawScreenUI:
                if (m_screenUIRenderer && m_screenUIOutput && m_renderView.kind != rhi::RenderViewKind::Scene) {
                    auto list = (screenUIListIndex == 0) ? ScreenUIList::Camera : ScreenUIList::Overlay;
                    m_screenUIRenderer->Render(ctx.GetCommandBuffer(), list, w, h, vkCore->GetCurrentFrameSlot());
                }
                break;
            case GraphCommandType::FullscreenQuad:
                // FullscreenQuad passes are handled entirely inside
                // BuildRenderGraph's execute lambda — the callback is a
                // no-op placeholder so the pass entry exists in m_pythonCallbacks.
                break;
            default:
                break;
            }
        };
    }

    // ========================================================================
    // Auto-append _ComponentGizmos pass (queue 10000-20000).
    // Python-defined per-component gizmos, rendered with depth testing
    // against existing scene geometry. Runs before editor gizmos.
    // ========================================================================
    static constexpr int COMP_GIZMO_QUEUE_MIN = 10000;
    static constexpr int COMP_GIZMO_QUEUE_MAX = 20000;
    static const std::string kComponentGizmosPassName = "_ComponentGizmos";
    const auto editorOverlayPass = GetEditorOverlayMaterialPass();
    m_pythonCallbacks[kComponentGizmosPassName] = [this, vkCore, editorOverlayPass](vk::RenderContext &ctx, uint32_t w,
                                                                                    uint32_t h) {
#if INFERNUX_FRAME_PROFILE
        const auto region = vkCore->BeginGpuProfileRegion(ctx.GetCommandBuffer(), "_ComponentGizmos");
        try {
            vkCore->DrawSceneFiltered(ctx.GetCommandBuffer(), w, h, GetEditorOverlayBindGroup(),
                                      GetViewGlobalsBinding(), m_drawView, COMP_GIZMO_QUEUE_MIN, COMP_GIZMO_QUEUE_MAX,
                                      "", "", "", &editorOverlayPass);
        } catch (...) {
            vkCore->EndGpuProfileRegion(ctx.GetCommandBuffer(), region);
            throw;
        }
        vkCore->EndGpuProfileRegion(ctx.GetCommandBuffer(), region);
#else
        vkCore->DrawSceneFiltered(ctx.GetCommandBuffer(), w, h, GetEditorOverlayBindGroup(), GetViewGlobalsBinding(),
                                  m_drawView, COMP_GIZMO_QUEUE_MIN, COMP_GIZMO_QUEUE_MAX, "", "", "",
                                  &editorOverlayPass);
#endif
    };

    // ========================================================================
    // Auto-append editor gizmos pass (queue 20001-25000).
    // This ensures grid/gizmos always render after all user-defined passes,
    // regardless of what queue ranges the user pipeline declares.
    // In game view (no gizmo draw calls), DrawSceneFiltered finds nothing
    // in this range and the pass is effectively a no-op.
    // ========================================================================
    static constexpr int GIZMO_QUEUE_MIN = 20001;
    static constexpr int GIZMO_QUEUE_MAX = 25000;
    static const std::string kEditorGizmosPassName = "_EditorGizmos";
    m_pythonCallbacks[kEditorGizmosPassName] = [this, vkCore, editorOverlayPass](vk::RenderContext &ctx, uint32_t w,
                                                                                 uint32_t h) {
        vkCore->DrawSceneFiltered(ctx.GetCommandBuffer(), w, h, GetEditorOverlayBindGroup(), GetViewGlobalsBinding(),
                                  m_drawView, GIZMO_QUEUE_MIN, GIZMO_QUEUE_MAX, "", "", "", &editorOverlayPass);
    };

    // ========================================================================
    // Auto-append editor tools pass (queue 25001-30000).
    // Translation/rotation/scale handles rendered on top of everything
    // (no depth test). In game view, no draw calls exist in this range.
    // ========================================================================
    static constexpr int TOOLS_QUEUE_MIN = 25001;
    static constexpr int TOOLS_QUEUE_MAX = 30000;
    static const std::string kEditorToolsPassName = "_EditorTools";
    m_pythonCallbacks[kEditorToolsPassName] = [this, vkCore, editorOverlayPass](vk::RenderContext &ctx, uint32_t w,
                                                                                uint32_t h) {
        vkCore->DrawSceneFiltered(ctx.GetCommandBuffer(), w, h, GetEditorOverlayBindGroup(), GetViewGlobalsBinding(),
                                  m_drawView, TOOLS_QUEUE_MIN, TOOLS_QUEUE_MAX, "preserve", "", "", &editorOverlayPass);
    };

    // Store description for BuildRenderGraph()'s topology traversal. Exact
    // repeats return before validation/callback construction above.
    if (topologyChanged) {
        m_liveBufferInputs.clear();
        m_importedTextureBindings.clear();
        m_pythonGraphDesc = std::move(normalizedDesc);
    }
    m_hasPythonGraph = true;
    m_pythonGraphValid = true;
    m_pythonGraphSourceRevision = desc.sourceRevision;
    m_pythonCallbackTarget = callbackTarget;
    m_pythonCallbackInvertCulling = m_cameraInvertCulling;
    if (topologyChanged || callbackContractChanged) {
        InvalidateTemporalHistory();
        m_needsRebuild = true;
    }
    return true;
}

void SceneRenderGraph::UpdateParameterBlocks(const std::vector<GraphParameterBlockUpdate> &updates)
{
    for (const auto &update : updates) {
        auto blockIt = m_parameterBlocks.find(update.id);
        if (blockIt == m_parameterBlocks.end())
            throw std::invalid_argument("Parameter block is not declared by this graph: " + update.id);

        RuntimeParameterBlock &block = blockIt->second;
        if (update.revision != 0 && update.revision == block.revision)
            continue;
        if (update.values.size() != block.names.size())
            throw std::invalid_argument("Parameter block '" + update.id + "' has a different compiled value count");

        auto values = std::make_shared<std::vector<ShaderParameterValue>>();
        values->reserve(update.values.size());
        for (size_t i = 0; i < update.values.size(); ++i) {
            if (update.values[i].first != block.names[i])
                throw std::invalid_argument("Parameter block '" + update.id + "' has different compiled names");
            const auto &value = update.values[i].second;
            values->push_back(block.scalarPushConstants ? ShaderParameterValue(value.ScalarFloat())
                                                        : value.ConvertTo((*block.values)[i].Kind()));
        }
        if (block.scalarPushConstants)
            for (size_t index = 0; index < values->size(); ++index)
                block.scalarPushConstants->values[index] = (*values)[index].Read<float>();
        block.values = std::move(values);
        block.revision = update.revision;
        ++m_parameterBlockGeneration;
    }
}

void SceneRenderGraph::CaptureParameterBlocksForSubmission()
{
    if (m_submittedParameterBlockGeneration == m_parameterBlockGeneration)
        return;
    m_submittedParameterBlocks = m_parameterBlocks;
    m_submittedParameterBlockGeneration = m_parameterBlockGeneration;
}

bool SceneRenderGraph::IsPythonGraphCurrent(uint64_t sourceRevision) const
{
    return sourceRevision != 0 && m_pythonGraphValid && m_pythonGraphSourceRevision == sourceRevision && m_vkCore &&
           m_pythonCallbackTarget.colorFormats[0] == m_renderView.colorFormat &&
           m_pythonCallbackTarget.depthFormat == m_renderView.depthFormat &&
           m_pythonCallbackTarget.samples == m_renderView.samples &&
           m_pythonCallbackInvertCulling == m_cameraInvertCulling && m_pythonGraphDesc.RenderTextureLayoutsCurrent();
}

uint32_t SceneRenderGraph::GetShadowMapResolution() const
{
    if (m_hasPythonGraph && !m_pythonGraphValid)
        return 0;
    for (const auto &pass : m_pythonGraphDesc.passes) {
        const bool drawsShadows =
            std::any_of(pass.commands.begin(), pass.commands.end(), [](const GraphCommandDesc &command) {
                return command.type == GraphCommandType::DrawShadowCasters;
            });
        if (!drawsShadows || pass.writeDepth.empty())
            continue;

        const auto texture =
            std::find_if(m_pythonGraphDesc.textures.begin(), m_pythonGraphDesc.textures.end(),
                         [&](const GraphTextureDesc &candidate) { return candidate.name == pass.writeDepth; });
        if (texture == m_pythonGraphDesc.textures.end())
            return 0;

        const auto [width, height] = texture->ResolveExtent(m_width, m_height);
        return width == height ? width : 0u;
    }
    return 0;
}

// ============================================================================
// Execution (Pure RenderGraph)
// ============================================================================

void SceneRenderGraph::RefreshMaterialTextureReads()
{
    if (m_hasPythonGraph && !m_pythonGraphValid)
        return;
    std::vector<MaterialTextureRead> reads;
    std::vector<MaterialBufferRead> bufferReads;
    std::vector<const DrawCall *> resourceDraws;
    for (const auto &draw : GetCachedDrawCalls()) {
        if (draw.frustumVisible)
            m_vkCore->PrepareMaterialTextureAssets(draw.material);
        const bool hasMaterialResources =
            draw.material && (!draw.material->GetRenderTextures().empty() || !draw.material->GetBuffers().empty());
        const bool hasRendererBuffers = draw.parameterBlock && !draw.parameterBlock->buffers.empty();
        if (draw.frustumVisible && (hasMaterialResources || hasRendererBuffers))
            resourceDraws.push_back(&draw);
    }
    const auto readBuffers = [&](const std::string &pass, const DrawCall &draw, bool parameters) {
        if (draw.material)
            for (const auto &[name, buffer] : draw.material->GetBuffers())
                if (buffer)
                    bufferReads.push_back({pass, buffer});
        if (parameters && draw.parameterBlock)
            for (const auto &[name, buffer] : draw.parameterBlock->buffers)
                if (buffer)
                    bufferReads.push_back({pass, buffer});
    };
    if (m_outlinePassesEnabled && m_outlineRenderer)
        for (const auto &draw : GetCachedDrawCalls())
            if (m_outlineRenderer->IsOutlinedObject(draw.objectId))
                readBuffers("__EditorOutlineMask", draw, true);
    for (const auto &pass : m_pythonGraphDesc.passes) {
        const auto *command = pass.PrimaryCommand();
        if (command && command->type == GraphCommandType::DrawShadowCasters) {
            const auto &shadowViews = m_cameraLightCollector.GetShadowFrame().views;
            if (shadowViews.empty())
                continue;
            uint32_t shadowLayers = 0;
            for (const auto &view : shadowViews)
                shadowLayers |= view.cullingMask;
            for (const auto &draw : GetCachedShadowDrawCalls()) {
                if (draw.castsShadows && (draw.layerMask & shadowLayers) && draw.material &&
                    draw.material->GetRenderQueue() >= command->queueMin &&
                    draw.material->GetRenderQueue() <= command->queueMax) {
                    const auto &textures = draw.material->GetRenderTextures();
                    if (!textures.empty()) {
                        const auto *material = m_vkCore->ResolveMeshMaterial(draw.material);
                        const auto program =
                            material ? m_vkCore->ShareShaderProgramArtifact(material->programKey.stages) : nullptr;
                        if (program && program->FindVariant(ShaderCompileTarget::Shadow))
                            for (const auto &binding : program->properties) {
                                if (!binding.IsTexture())
                                    continue;
                                if (const auto texture = textures.find(binding.name); texture != textures.end())
                                    reads.push_back({pass.name, texture->second, texture->second->Acquire()});
                            }
                    }
                    readBuffers(pass.name, draw, false);
                }
            }
            continue;
        }
        if (command && m_screenUIRenderer &&
            (command->type == GraphCommandType::DrawWorldUI ||
             (command->type == GraphCommandType::DrawScreenUI && m_screenUIOutput &&
              m_renderView.kind != rhi::RenderViewKind::Scene))) {
            const auto list = command->type == GraphCommandType::DrawWorldUI
                                  ? ScreenUIList::World
                                  : (command->screenUIList == 0 ? ScreenUIList::Camera : ScreenUIList::Overlay);
            auto mask = m_cachedCamera ? m_cachedCamera->GetCullingMask() : 0xffffffffu;
            if (command->type == GraphCommandType::DrawWorldUI)
                mask &= command->worldUILayerMask;
            for (const auto &texture : m_screenUIRenderer->GetRenderTextureReads(list, mask))
                reads.push_back({pass.name, texture, texture->Acquire()});
            continue;
        }
        if (!command ||
            (command->type != GraphCommandType::DrawRenderers && command->type != GraphCommandType::DrawSkybox))
            continue;
        const bool skybox = command->type == GraphCommandType::DrawSkybox;
        std::shared_ptr<InxMaterial> overrideMaterial;
        if (command->rendererSelection) {
            overrideMaterial = command->rendererSelection->Material();
            m_vkCore->PrepareMaterialTextureAssets(overrideMaterial);
        } else if (!skybox && !command->overrideMaterial.empty()) {
            auto &registry = AssetRegistry::Instance();
            overrideMaterial = registry.GetBuiltinMaterial(command->overrideMaterial);
            if (!overrideMaterial)
                overrideMaterial = registry.LoadAsset<InxMaterial>(command->overrideMaterial, ResourceType::Material);
            m_vkCore->PrepareMaterialTextureAssets(overrideMaterial);
        }
        const int minimum = skybox ? EngineConfig::Get().skyboxQueue : command->queueMin;
        const int maximum = skybox ? minimum : command->queueMax;
        const auto visit = [&](const DrawCall &draw) {
            if (!draw.frustumVisible || skybox != (draw.identity.domain == RenderDomain::Skybox))
                return;
            if (command->rendererSelection && !command->rendererSelection->Find(draw.identity))
                return;
            const auto &material = overrideMaterial ? overrideMaterial : draw.material;
            const std::shared_ptr<const RendererParameterBlock> *parameters =
                command->rendererSelection ? command->rendererSelection->Find(draw.identity)
                                           : (overrideMaterial ? nullptr : &draw.parameterBlock);
            const bool hasParameterBuffers = parameters && *parameters && !(*parameters)->buffers.empty();
            if (!material ||
                (material->GetRenderTextures().empty() && material->GetBuffers().empty() && !hasParameterBuffers))
                return;
            const auto &source = draw.material ? draw.material : material;
            if (!skybox && (source->GetRenderQueue() < minimum || source->GetRenderQueue() > maximum))
                return;
            const auto &tag = source->GetPassTag();
            if (!skybox && !command->passTag.empty() &&
                ((overrideMaterial && tag != command->passTag) ||
                 (!overrideMaterial && !tag.empty() && tag != command->passTag)))
                return;
            if (command->materialFilter != GraphMaterialFilter::All) {
                const auto *committed = m_vkCore->ResolveMeshMaterial(material);
                const auto artifact =
                    committed ? m_vkCore->ShareShaderProgramArtifact(committed->programKey.stages) : nullptr;
                const bool deferred = artifact && artifact->FindVariant(ShaderCompileTarget::GBuffer);
                if ((command->materialFilter == GraphMaterialFilter::DeferredCompatible && !deferred) ||
                    (command->materialFilter == GraphMaterialFilter::DeferredUnsupported && deferred))
                    return;
            }
            for (const auto &[name, texture] : material->GetRenderTextures())
                reads.push_back({pass.name, texture, texture->Acquire()});
            for (const auto &[name, buffer] : material->GetBuffers()) {
                if (buffer)
                    bufferReads.push_back({pass.name, buffer});
            }
            if (parameters && *parameters) {
                for (const auto &[name, buffer] : (*parameters)->buffers) {
                    if (buffer)
                        bufferReads.push_back({pass.name, buffer});
                }
            }
        };
        if (overrideMaterial) {
            if (!overrideMaterial->GetRenderTextures().empty() || !overrideMaterial->GetBuffers().empty() ||
                command->rendererSelection) {
                for (const auto &draw : GetCachedDrawCalls())
                    visit(draw);
            }
        } else {
            for (const auto *draw : resourceDraws)
                visit(*draw);
        }
    }
    std::sort(reads.begin(), reads.end(), [](const auto &a, const auto &b) {
        if (a.passName != b.passName)
            return a.passName < b.passName;
        return std::less<const rhi::RenderTexture *>{}(a.texture.get(), b.texture.get());
    });
    reads.erase(std::unique(reads.begin(), reads.end()), reads.end());
    std::sort(bufferReads.begin(), bufferReads.end(), [](const auto &a, const auto &b) {
        if (a.passName != b.passName)
            return a.passName < b.passName;
        return std::less<const rhi::ComputeBuffer *>{}(a.buffer.get(), b.buffer.get());
    });
    bufferReads.erase(std::unique(bufferReads.begin(), bufferReads.end()), bufferReads.end());
    const bool bufferSourcesChanged = bufferReads != m_materialBufferReads;
    if (reads != m_materialTextureReads || bufferSourcesChanged) {
        // A pending asset can leave the old compiled graph in service. It can
        // keep running only while it covers every current sampled allocation.
        if (bufferSourcesChanged || (m_graphBuilt && std::any_of(reads.begin(), reads.end(), [this](const auto &read) {
                                         const auto bound = m_drawTextureInputs.find(read.passName);
                                         return bound == m_drawTextureInputs.end() ||
                                                std::find(bound->second.begin(), bound->second.end(),
                                                          read.generation) == bound->second.end();
                                     })))
            m_graphBuilt = false;
        m_materialTextureReads = std::move(reads);
        m_materialBufferReads = std::move(bufferReads);
        m_needsRebuild = true;
        ++m_resourceAccessRevision;
    }
}

void SceneRenderGraph::RetireComputeBufferBindings()
{
    if (m_computeBufferReads.empty())
        return;
    auto retired = std::move(m_computeBufferReads);
    if (m_vkCore)
        m_vkCore->GetRetirementQueue().Retire([retired = std::move(retired)]() mutable { retired.clear(); });
}

static rhi::QueueRole GraphBufferWriter(const rhi::ComputeBufferAllocation &allocation)
{
    return allocation.GetLastWriteSubmission().IsValid() ? allocation.GetLastWriterQueue() : rhi::QueueRole::Graphics;
}

void SceneRenderGraph::DeclareMaterialBufferReads(vk::PassBuilder &builder, const std::string &passName)
{
    for (const auto &read : m_materialBufferReads) {
        if (read.passName != passName)
            continue;
        const auto &allocation = m_computeBufferReads.at(read.buffer.get());
        const auto handle =
            builder.ImportBuffer("__MaterialBuffer/" + passName, allocation->GetBuffer(), allocation->GetByteSize());
        if (!handle.IsValid())
            throw std::runtime_error("Material storage buffer could not be imported into the render graph");
        m_materialBufferImports.emplace_back(handle, read.buffer.get());
        m_renderGraph->SetResourceInitialState(handle, rhi::TextureLayout::Undefined, rhi::Access::MemoryWrite,
                                               rhi::PipelineStage::AllCommands, GraphBufferWriter(*allocation));
        builder.ReadStorageBuffer(handle, rhi::PipelineStage::AllGraphics);
    }
}

rhi::ComputeBufferBindings SceneRenderGraph::CaptureComputeBufferReads() const
{
    rhi::ComputeBufferBindings reads;
    const auto include = [&](const std::shared_ptr<rhi::ComputeBuffer> &buffer) {
        if (buffer)
            reads.emplace(buffer.get(), buffer->GetAllocation());
    };
    for (const auto &read : m_materialBufferReads)
        include(read.buffer);
    for (const auto &buffer : m_pythonGraphDesc.buffers)
        include(buffer.computeBuffer);
    for (const auto &[name, owner] : m_graphBufferOwners)
        include(owner);
    return reads;
}

bool SceneRenderGraph::RefreshComputeBufferBindings(bool &prepared)
{
    bool inputChanged = false;
    for (auto &buffer : m_pythonGraphDesc.buffers) {
        if (buffer.input && buffer.computeBuffer != buffer.input->GetValue()) {
            buffer.computeBuffer = buffer.input->GetValue();
            inputChanged = true;
        }
    }
    // Static graph validation checks layout. Every actual frame checks its
    // captured external value, including unchanged graph descriptions.
    for (const auto &buffer : m_pythonGraphDesc.buffers) {
        if (!buffer.computeBuffer)
            continue;
        const auto allocation = buffer.computeBuffer->GetAllocation();
        const auto producer = allocation->GetLastWriteSubmission();
        // Typed compute allocations support ordinary graph reads. Indirect
        // commands additionally require that capability on the actual value.
        const bool indirect = (buffer.usage & static_cast<uint32_t>(GraphBufferUsage::Indirect)) != 0;
        if (!producer.IsValid() || producer.device != allocation->GetBuffer().Device() ||
            !allocation->HasInitializedRange(buffer.byteOffset, buffer.byteSize) ||
            (indirect && !rhi::HasBufferUsage(allocation->GetStorageDesc().usage, rhi::BufferUsageFlags::Indirect))) {
            INXLOG_ERROR("Graph buffer '", buffer.name, "' lacks an accepted range or the required indirect usage");
            m_graphBuilt = false;
            m_needsRebuild = true;
            return false;
        }
    }

    const bool writesPinned =
        std::any_of(m_graphBufferWrites.begin(), m_graphBufferWrites.end(), [this](const auto &name) {
            const auto owner = m_graphBufferOwners.find(name);
            return owner != m_graphBufferOwners.end() && owner->second->GetAllocation()->IsReadProtected();
        });
    if (m_graphBuilt && !inputChanged && !writesPinned && CaptureComputeBufferReads() == m_computeBufferReads)
        return true;

    m_graphBuilt = false;
    if (m_needsRebuild)
        return true;
    if (!PrepareGraphBuffers())
        return false;
    prepared = true;

    std::vector<vk::RenderGraph::ImportedBufferBinding> bindings;
    bindings.reserve(m_graphBufferImports.size() + m_materialBufferImports.size());
    const auto bind = [&](vk::ResourceHandle handle, const rhi::ComputeBuffer &owner, bool graphLocal) {
        const auto &allocation = m_computeBufferReads.at(&owner);
        bindings.push_back({handle, allocation->GetBuffer(), GraphBufferWriter(*allocation), graphLocal});
    };
    for (const auto &[handle, index] : m_graphBufferImports) {
        const auto owner = ResolveGraphBufferOwner(m_pythonGraphDesc.buffers[index]);
        bind(handle, *owner, owner->GetAllocation()->IsGraphWritable() && !owner->IsFrozen());
    }
    for (const auto &[handle, owner] : m_materialBufferImports)
        bind(handle, *owner, false);
    m_graphBuilt = m_renderGraph->RebindImportedBuffers(bindings);
    m_needsRebuild = !m_graphBuilt;
    return true;
}

std::shared_ptr<void> SceneRenderGraph::AcquireComputeReadValues() const
{
    if (m_computeBufferReads.empty())
        return {};
    auto values = std::make_shared<std::vector<std::shared_ptr<rhi::ComputeBufferState>>>();
    std::unordered_set<const rhi::ComputeBufferAllocation *> included;
    const auto include = [&](const std::shared_ptr<rhi::ComputeBuffer> &buffer) {
        const auto &allocation = m_computeBufferReads.at(buffer.get());
        if (m_renderGraph->GetImportedBufferAccessStages(allocation->GetBuffer()) != rhi::PipelineStage::None &&
            included.insert(allocation.get()).second)
            values->push_back(std::make_shared<rhi::ComputeBufferState>(buffer->GetDesc(), allocation, true));
    };
    for (const auto &read : m_materialBufferReads)
        include(read.buffer);
    for (const auto &buffer : m_pythonGraphDesc.buffers)
        if (buffer.computeBuffer)
            include(buffer.computeBuffer);
    for (const auto &[name, instance] : m_workInstances) {
        const auto &slots = instance->GetRecipe().GetSlots();
        for (size_t index = 0; index < slots.size(); ++index)
            if (slots[index].external)
                include(instance->GetBuffers()[index]);
    }
    return values->empty() ? nullptr : values;
}

std::vector<rhi::SubmissionDependency> SceneRenderGraph::GetBufferReadDependencies() const
{
    std::vector<rhi::SubmissionDependency> dependencies;
    if (!m_graphBuilt || !m_renderGraph || (m_hasPythonGraph && !m_pythonGraphValid))
        return dependencies;
    for (const auto &[owner, allocation] : m_computeBufferReads) {
        const auto stages = m_renderGraph->GetImportedBufferAccessStages(allocation->GetBuffer());
        const auto ticket = allocation->GetLastWriteSubmission();
        if (ticket.IsValid() && stages != rhi::PipelineStage::None)
            dependencies.push_back({ticket, stages});
    }
    return dependencies;
}

void SceneRenderGraph::EnsureGraphBuilt()
{
    if (m_hasPythonGraph && (!m_pythonGraphValid || m_rejectedFullscreenSourceRevision.has_value())) {
        m_graphBuilt = false;
        return;
    }
    if (!m_sceneTarget || !m_sceneTarget->IsReady() || !m_renderGraph) {
        return;
    }

    // Only views entering actual preparation allocate GPU resources. Keep the
    // staged lighting current even when a later rebuild waits on an asset.
    if (!EnsureViewGpuResources() || !UploadCameraLighting()) {
        m_graphBuilt = false;
        m_needsRebuild = true;
        return;
    }

    // Culling is submitted after the Python graph. Discover runtime material
    // inputs here, before graph compilation and cross-camera scheduling.
    RefreshMaterialTextureReads();
    if (m_bufferProofClearFlags != m_cameraClearFlags && RequiresGraphBufferProof(m_pythonGraphDesc))
        m_needsRebuild = true;

    uint64_t worldUIDepthSignature = 0;
    if (m_screenUIRenderer) {
        const uint32_t cameraMask = m_cachedCamera ? m_cachedCamera->GetCullingMask() : 0xffffffffu;
        for (size_t passIndex = 0; passIndex < m_pythonGraphDesc.passes.size(); ++passIndex) {
            const auto &pass = m_pythonGraphDesc.passes[passIndex];
            const auto *command = pass.PrimaryCommand();
            if (!command || command->type != GraphCommandType::DrawWorldUI)
                continue;
            const uint32_t mask = cameraMask & command->worldUILayerMask;
            if (!m_screenUIRenderer->HasSelectiveWorldOcclusion(mask))
                continue;
            const auto runs = m_screenUIRenderer->GetWorldDepthRuns(m_cachedProj * m_drawView, mask);
            if (!worldUIDepthSignature)
                worldUIDepthSignature = 1469598103934665603ull;
            worldUIDepthSignature = (worldUIDepthSignature ^ passIndex) * 1099511628211ull;
            worldUIDepthSignature = (worldUIDepthSignature ^ mask) * 1099511628211ull;
            for (const auto &run : runs) {
                for (const uint64_t value : {uint64_t(run.firstOrdinal), uint64_t(run.endOrdinal),
                                             run.ignoredOccluderId, uint64_t(run.alwaysOnTop)}) {
                    worldUIDepthSignature = (worldUIDepthSignature ^ value) * 1099511628211ull;
                }
            }
        }
    }
    if (worldUIDepthSignature != m_worldUIDepthRunSignature) {
        m_worldUIDepthRunSignature = worldUIDepthSignature;
        m_needsRebuild = true;
    }
    if (worldUIDepthSignature && m_vkCore) {
        std::optional<WorldUIDepthReplay> replay;
        const uint32_t cameraMask = m_cachedCamera ? m_cachedCamera->GetCullingMask() : 0xffffffffu;
        for (const auto &pass : m_pythonGraphDesc.passes) {
            const auto *command = pass.PrimaryCommand();
            if (!command || command->type != GraphCommandType::DrawWorldUI ||
                !m_screenUIRenderer->HasSelectiveWorldOcclusion(cameraMask & command->worldUILayerMask))
                continue;
            auto candidate = BuildWorldUIDepthReplay(m_pythonGraphDesc,
                                                     m_cameraClearFlags == CameraClearFlags::DontClear, pass.name);
            if (!candidate) {
                INXLOG_ERROR("Selective World UI occlusion requires one cleared Forward scene-depth writer; "
                             "pass '",
                             pass.name, "' or camera depth-preservation policy cannot be replayed exactly");
                m_graphBuilt = false;
                m_needsRebuild = true;
                return;
            }
            // Every accepted stage shares the one scene-depth producer. Keep
            // its owned command for material checks, independent of its name.
            if (!replay)
                replay = std::move(candidate);
        }
        const GraphCommandDesc *opaqueCommand = replay ? &replay->draw : nullptr;
        if (opaqueCommand && opaqueCommand->type == GraphCommandType::DrawRenderers) {
            const auto defaultMaterial = AssetRegistry::Instance().GetBuiltinMaterial("DefaultLit");
            for (const auto &draw : GetCachedDrawCalls()) {
                if (!draw.frustumVisible)
                    continue;
                const auto &material = draw.material ? draw.material : defaultMaterial;
                if (!material || material->GetRenderQueue() < opaqueCommand->queueMin ||
                    material->GetRenderQueue() > opaqueCommand->queueMax)
                    continue;
                const auto &tag = material->GetPassTag();
                if (!opaqueCommand->passTag.empty() && !tag.empty() && tag != opaqueCommand->passTag)
                    continue;
                const auto &state = material->GetRenderState();
                if (!state.depthWriteEnable)
                    continue;
                const auto *committed = m_vkCore->ResolveMeshMaterial(material);
                const auto artifact =
                    committed ? m_vkCore->ShareShaderProgramArtifact(committed->programKey.stages) : nullptr;
                if (!state.depthTestEnable || state.stencilTestEnable || !artifact ||
                    !artifact->FindVariant(ShaderCompileTarget::Depth)) {
                    INXLOG_ERROR("Selective World UI occlusion cannot replay depth-writing material '",
                                 material->GetName(),
                                 "': it requires ordinary depth testing, no stencil test, and a Depth shader variant");
                    m_graphBuilt = false;
                    m_needsRebuild = true;
                    return;
                }
            }
        }
    }

    const auto &assetRegistry = AssetRegistry::Instance();
    bool persistentImportsChanged = false;
    for (const auto &texture : m_pythonGraphDesc.textures) {
        if (texture.role != GraphTextureRole::Asset && texture.role != GraphTextureRole::Persistent)
            continue;
        const bool asset = texture.role == GraphTextureRole::Asset;
        const uint64_t revision =
            asset ? assetRegistry.GetAssetVersion(texture.assetGuid) : texture.renderTexture->Acquire()->revision;
        const auto found = m_importedTextureBindings.find(texture.name);
        // The content change can precede the next CPU/GPU publication by several frames.
        if (found == m_importedTextureBindings.end() || found->second.revision != revision ||
            (asset && found->second.contentGeneration != assetRegistry.GetAssetContentGeneration(texture.assetGuid))) {
            m_needsRebuild = true;
            persistentImportsChanged |= !asset;
        }
    }

    if (m_particleDrawRegistry) {
        const uint64_t revision = m_particleDrawRegistry->Revision();
        if (revision != m_particleDrawRegistryRevision)
            m_needsRebuild = true;
    }

    // ========================================================================
    // MSAA mismatch guard: if the Python pipeline requested a different MSAA
    // sample count than the scene target currently has, skip this frame.
    // InxRenderer::DrawFrame() will detect the mismatch on the NEXT frame
    // (via GetRequestedMsaaSamples()) and call SetMsaaSamples() to recreate
    if (m_hasPythonGraph && m_pythonGraphDesc.msaaSamples > 0) {
        auto currentMsaa = static_cast<int>(m_sceneTarget->GetMsaaSampleCount());
        const int effectiveMsaa = m_effectiveMsaaSamples > 0 ? m_effectiveMsaaSamples : m_pythonGraphDesc.msaaSamples;
        if (effectiveMsaa != currentMsaa) {
            m_needsRebuild = true;
            // Prevent Execute() from running the stale compiled graph
            // whose render passes reference images with the old sample
            // count.  Without this, a single frame of execution with
            // mismatched MSAA resources can cause a Vulkan error or hang.
            m_graphBuilt = false;
            return;
        }
    }

    // Update dimensions if changed
    if (m_width != m_sceneTarget->GetWidth() || m_height != m_sceneTarget->GetHeight()) {
        m_width = m_sceneTarget->GetWidth();
        m_height = m_sceneTarget->GetHeight();
        m_renderView.width = m_width;
        m_renderView.height = m_height;
        m_renderGraph->SetRenderView(m_renderView);
        m_needsRebuild = true;
    }

    bool buffersPrepared = false;
    if (!RefreshComputeBufferBindings(buffersPrepared))
        return;

    // Request all asset imports together and build from those exact publications.
    // A pending upload can leave a compatible compiled graph running.
    std::vector<GraphAssetTexture> assetTextures;
    const auto resolveAssetTextures = [&] {
        bool allResolved = true;
        for (const auto &texture : m_pythonGraphDesc.textures) {
            if (texture.role != GraphTextureRole::Asset)
                continue;
            auto resolved = m_vkCore->ResolveTextureForGraph(texture.assetGuid, texture.isVolume, false);
            if (resolved.status == TextureResolveStatus::Pending)
                allResolved = false;
            else
                assetTextures.push_back({&texture, std::move(resolved.binding.gpuView), resolved.contentGeneration});
        }
        return allResolved;
    };
    const bool rebuildBeforeLighting = m_needsRebuild;
    if (rebuildBeforeLighting) {
        if (!resolveAssetTextures())
            return;
        if (persistentImportsChanged && !ValidateRenderGraphDescription(m_pythonGraphDesc, m_renderView)) {
            m_graphBuilt = false;
            return;
        }
        RefreshForwardPlusParticleRequirement();
    }

    if (UsesForwardPlus() && !PrepareForwardPlusFrame()) {
        INXLOG_ERROR("SceneRenderGraph: Forward+ resources are unavailable for the current view");
        m_graphBuilt = false;
        return;
    }

    if (m_needsRebuild) {
        if (!rebuildBeforeLighting && !resolveAssetTextures()) {
            // Forward+ replaced a buffer used by the previous graph.
            m_graphBuilt = false;
            return;
        }
        // RenderGraph rebuilds can retire and recreate the transient shadow
        // image view. Vulkan may reuse the same raw handle value, so cached
        // handle equality cannot prove an existing descriptor still refers
        // to the new object. Force every frame-local descriptor generation
        // to be republished after the rebuild.
        InvalidatePerViewShadowBindings();
        BuildRenderGraph(buffersPrepared, assetTextures);
        if (!m_graphBuilt)
            return;
        m_needsRebuild = false;
        m_needsCompile = true; // Need to compile after rebuild
    }

    if (m_graphBuilt && m_needsCompile) {
        if (!m_renderGraph->Compile()) {
            INXLOG_ERROR("SceneRenderGraph: Failed to compile render graph — disabling graph until next rebuild");
            m_graphBuilt = false;
            return;
        }
        CacheLiveAuthorPasses();
        ++m_resourceAccessRevision;
        const vk::ResourceHandle displayTarget =
            m_importedResolveTarget.IsValid() ? m_importedResolveTarget : m_importedColorTarget;
        m_renderView.color = m_renderGraph->ResolveRhiTextureView(displayTarget);
        m_renderView.depth = m_renderGraph->ResolveRhiTextureView(m_importedDepthTarget);
        m_needsCompile = false;
    }

    if (m_graphBuilt && m_outlinePassesEnabled) {
        const auto maskContract = m_renderGraph->GetPassRenderingContract("__EditorOutlineMask");
        const auto compositeContract = m_renderGraph->GetPassRenderingContract("__EditorOutlineComposite");
        const bool outlineReady = m_outlineRenderer && m_outlineRenderer->EnsureGraphPipelines(
                                                           maskContract.attachments, compositeContract.attachments);
        if (outlineReady) {
            m_outlinePipelineFailureReported = false;
        } else if (!m_outlinePipelineFailureReported) {
            INXLOG_ERROR("SceneRenderGraph: failed to prepare graph-compatible editor outline pipelines");
            m_outlinePipelineFailureReported = true;
        }
        // Outline is an editor aid. Keep the otherwise valid scene graph
        // executable and retry pipeline publication on a later frame instead
        // of blacking out the Scene view after a transient creation failure.
    }

    if (m_graphBuilt) {
        RefreshPerViewShadowDescriptor();
    }
}

bool SceneRenderGraph::UsesForwardPlus() const
{
    if (!m_pythonGraphValid)
        return false;
    for (const auto &pass : m_pythonGraphDesc.passes) {
        for (const auto &command : pass.commands) {
            if (command.type == GraphCommandType::DrawRenderers &&
                command.shaderTarget == ShaderCompileTarget::ForwardPlus) {
                return true;
            }
            if (command.type == GraphCommandType::FullscreenQuad) {
                const auto program = m_fullscreenRenderer.AcquireProgram(command.shaderName);
                if (program && (program->UsesViewBinding(2) || program->UsesViewBinding(3)))
                    return true;
            }
        }
    }
    return false;
}

void SceneRenderGraph::RefreshForwardPlusParticleRequirement()
{
    m_forwardPlusParticlesRequired = false;
    if (!m_pythonGraphValid || !m_particleDrawRegistry)
        return;
    for (const auto &pass : m_pythonGraphDesc.passes) {
        for (const auto &command : pass.commands) {
            if (command.type != GraphCommandType::DrawRenderers || command.rendererSelection ||
                command.shaderTarget != ShaderCompileTarget::ForwardPlus) {
                continue;
            }
            const auto entries = m_particleDrawRegistry->SnapshotShared(command.queueMin, command.queueMax);
            if (std::any_of(entries->begin(), entries->end(),
                            [](const auto &entry) { return entry.semantics.receiveSceneLighting; })) {
                m_forwardPlusParticlesRequired = true;
                return;
            }
        }
    }
}

bool SceneRenderGraph::PrepareForwardPlusFrame()
{
    if (!m_vkCore || !m_forwardPlusGeometryGrid.IsValid() ||
        (m_forwardPlusParticlesRequired && !m_forwardPlusParticleGrid.IsValid()))
        return false;
    const uint32_t frameIndex = m_vkCore->GetCurrentFrameSlot() % EngineConfig::kFrameSlotCapacity;
    const auto &lights = m_cameraCanonicalLights.Frame(frameIndex);
    if (!lights.buffer.IsValid())
        return false;

    const auto previousHeaders = m_forwardPlusGeometryGrid.Frame(frameIndex).headers;
    const auto previousMasks = m_forwardPlusGeometryGrid.Frame(frameIndex).lightMasks;
    if (!m_forwardPlusGeometryGrid.PrepareFrame(frameIndex, m_width, m_height, lights.localCount,
                                                CanonicalLightAffectsGeometry, lights.buffer)) {
        return false;
    }
    const auto &frame = m_forwardPlusGeometryGrid.Frame(frameIndex);
    if (frame.headers != previousHeaders || frame.lightMasks != previousMasks)
        m_needsRebuild = true;

    if (m_forwardPlusParticlesRequired) {
        const auto previousParticleHeaders = m_forwardPlusParticleGrid.Frame(frameIndex).headers;
        const auto previousParticleMasks = m_forwardPlusParticleGrid.Frame(frameIndex).lightMasks;
        if (!m_forwardPlusParticleGrid.PrepareFrame(frameIndex, m_width, m_height, lights.localCount,
                                                    CanonicalLightAffectsParticles, lights.buffer)) {
            return false;
        }
        const auto &particleFrame = m_forwardPlusParticleGrid.Frame(frameIndex);
        if (particleFrame.headers != previousParticleHeaders || particleFrame.lightMasks != previousParticleMasks) {
            m_needsRebuild = true;
        }
    }

    RetireForwardPlusResources();
    auto &viewFrame = m_perViewFrames[frameIndex];
    const VkDescriptorSet descriptorSet = viewFrame.GeometrySet();
    if (descriptorSet == VK_NULL_HANDLE)
        return false;
    auto &rhiDevice = m_vkCore->GetDeviceContext().GetRhiDevice();
    const PerViewBufferBindingState geometryBindings{rhiDevice.Resolve(lights.buffer),
                                                     rhiDevice.Resolve(frame.headers),
                                                     rhiDevice.Resolve(frame.lightMasks),
                                                     rhiDevice.Resolve(viewFrame.lighting),
                                                     lights.dataBytes,
                                                     frame.config.headerBytes,
                                                     frame.config.maskBytes,
                                                     true};
    auto bindingsDiffer = [](const PerViewBufferBindingState &left, const PerViewBufferBindingState &right) {
        return !left.initialized || left.canonicalLights != right.canonicalLights ||
               left.tileHeaders != right.tileHeaders || left.tileLightMasks != right.tileLightMasks ||
               left.lighting != right.lighting || left.canonicalBytes != right.canonicalBytes ||
               left.tileHeaderBytes != right.tileHeaderBytes || left.tileLightMaskBytes != right.tileLightMaskBytes;
    };
    const bool geometryBindingsDirty = bindingsDiffer(viewFrame.geometryBindings, geometryBindings);
    bool particleBindingsDirty = false;
    PerViewBufferBindingState particleBindings{};
    if (m_forwardPlusParticlesRequired) {
        const auto &particleFrame = m_forwardPlusParticleGrid.Frame(frameIndex);
        particleBindings = {rhiDevice.Resolve(lights.buffer),
                            rhiDevice.Resolve(particleFrame.headers),
                            rhiDevice.Resolve(particleFrame.lightMasks),
                            rhiDevice.Resolve(viewFrame.lighting),
                            lights.dataBytes,
                            particleFrame.config.headerBytes,
                            particleFrame.config.maskBytes,
                            true};
        particleBindingsDirty = bindingsDiffer(viewFrame.particleBindings, particleBindings);
    }
    // InxRenderer waits for this frame slot's graphics fence before any
    // SceneRenderGraph update. Each slot owns distinct per-view descriptor
    // sets, so publishing new buffer bindings here cannot race a command
    // buffer that still consumes the same sets. A device-wide idle would
    // unnecessarily drain the other frame slot and all independent queues.
    if (geometryBindingsDirty) {
        m_vkCore->UpdatePerViewForwardPlusBuffers(descriptorSet, lights.buffer, lights.dataBytes, frame.headers,
                                                  frame.config.headerBytes, frame.lightMasks, frame.config.maskBytes,
                                                  viewFrame.lighting, sizeof(ShaderLightingUBO));
        m_vkCore->UpdatePerViewForwardPlusBuffers(
            viewFrame.EditorOverlaySet(), lights.buffer, lights.dataBytes, frame.headers, frame.config.headerBytes,
            frame.lightMasks, frame.config.maskBytes, viewFrame.lighting, sizeof(ShaderLightingUBO));
        viewFrame.geometryBindings = geometryBindings;
    }
    if (m_forwardPlusParticlesRequired) {
        if (viewFrame.ParticleSet() == VK_NULL_HANDLE || !m_perViewLayout.IsValid() ||
            !viewFrame.particleGroup.IsValid() || !viewFrame.lighting.IsValid()) {
            return false;
        }
        if (particleBindingsDirty) {
            const auto &particleFrame = m_forwardPlusParticleGrid.Frame(frameIndex);
            m_vkCore->UpdatePerViewForwardPlusBuffers(viewFrame.ParticleSet(), lights.buffer, lights.dataBytes,
                                                      particleFrame.headers, particleFrame.config.headerBytes,
                                                      particleFrame.lightMasks, particleFrame.config.maskBytes,
                                                      viewFrame.lighting, sizeof(ShaderLightingUBO));
            viewFrame.particleBindings = particleBindings;
        }
    }
    return true;
}

void SceneRenderGraph::RetireForwardPlusResources()
{
    auto retired = m_forwardPlusGeometryGrid.TakeRetiredResources();
    auto retiredParticles = m_forwardPlusParticleGrid.TakeRetiredResources();
    retired.insert(retired.end(), std::make_move_iterator(retiredParticles.begin()),
                   std::make_move_iterator(retiredParticles.end()));
    if (retired.empty() || !m_vkCore)
        return;
    rhi::Device *device = &m_vkCore->GetDeviceContext().GetRhiDevice();
    m_vkCore->GetRetirementQueue().Retire([device, retired = std::move(retired)]() mutable {
        for (const auto &resource : retired) {
            device->Release(resource.bindGroup);
            device->Release(resource.consumerBindGroup);
            device->Release(resource.lightMasks);
            device->Release(resource.headers);
        }
    });
}

bool SceneRenderGraph::PrepareSubmissionExecution()
{
    if (!IsGraphBuilt() || !m_sceneTarget || !m_sceneTarget->IsReady() || !m_renderGraph) {
        return false;
    }
    if (!m_graphBuffersPublished)
        PublishGraphBuffers();

    m_motionHistory.BeginFrame();
    if (m_importedColorTarget.IsValid()) {
        if (m_sceneTarget->IsMsaaEnabled()) {
            m_renderGraph->SetResourceInitialState(m_importedColorTarget, rhi::TextureLayout::ColorAttachment,
                                                   rhi::Access::ColorWrite, rhi::PipelineStage::ColorOutput);
        } else {
            m_renderGraph->SetResourceInitialState(m_importedColorTarget, rhi::TextureLayout::ShaderReadOnly,
                                                   rhi::Access::ShaderRead, rhi::RenderTextureSampleStages);
        }
    }

    if (m_importedResolveTarget.IsValid()) {
        m_renderGraph->SetResourceInitialState(m_importedResolveTarget, rhi::TextureLayout::ShaderReadOnly,
                                               rhi::Access::ShaderRead, rhi::RenderTextureSampleStages);
    }

    if (m_importedDepthTarget.IsValid()) {
        const bool sampled = m_outputGeneration && m_outputGeneration->description.sampledDepth &&
                             m_outputGeneration->description.samples == rhi::SampleCount::One;
        m_renderGraph->SetResourceInitialState(
            m_importedDepthTarget,
            sampled ? rhi::TextureLayout::DepthStencilReadOnly : rhi::TextureLayout::DepthStencilAttachment,
            sampled ? rhi::Access::ShaderRead : rhi::Access::DepthRead | rhi::Access::DepthWrite,
            sampled ? rhi::RenderTextureSampleStages : rhi::PipelineStage::EarlyDepth | rhi::PipelineStage::LateDepth);
    }

    RefreshShadowAtlasUpdateState();

    BindTemporalHistoryResources();

    if (m_hasCameraClearOverride && !m_mainClearPassName.empty()) {
        if (m_cameraClearFlags == CameraClearFlags::SolidColor) {
            m_renderGraph->UpdatePassClearColor(m_mainClearPassName, m_cameraBgColor.r, m_cameraBgColor.g,
                                                m_cameraBgColor.b, m_cameraBgColor.a);
        }
    }

    m_prevClearStateValid = true;
    m_prevCameraClearFlags = m_cameraClearFlags;
    m_prevCameraBgColor = m_cameraBgColor;

    m_fullscreenRenderer.ResetPool();

    if (!m_particleCullers.empty()) {
        Frustum frustum;
        frustum.ExtractFromMatrix(m_cachedProj * m_cachedView);
        for (uint32_t index = 0; index < particle::ParticleGpuCuller::PlaneCount; ++index) {
            const auto &plane = frustum.GetPlane(static_cast<Frustum::PlaneIndex>(index));
            m_particleFrustumPlanes[index * 4 + 0] = plane.normal.x;
            m_particleFrustumPlanes[index * 4 + 1] = plane.normal.y;
            m_particleFrustumPlanes[index * 4 + 2] = plane.normal.z;
            m_particleFrustumPlanes[index * 4 + 3] = plane.distance;
        }
    }
    return true;
}

uint64_t SceneRenderGraph::ComputeShadowContentSignature(bool &hasDynamicCaster) const
{
    hasDynamicCaster = false;
    uint64_t hash = kShadowHashOffset;
    HashShadowValue(hash, m_cachedSubmissionSignature);
    HashShadowValue(hash, m_cachedObjectBufferRevision);
    if (m_vkCore) {
        const uint64_t materialGeneration = m_vkCore->GetMaterialPipelineManager().GetPublicationGeneration();
        HashShadowValue(hash, materialGeneration);
    }

    const auto &shadowFrame = m_cameraLightCollector.GetShadowFrame();
    HashShadowValue(hash, shadowFrame.atlasSize);
    const uint64_t viewCount = shadowFrame.views.size();
    const uint64_t assignmentCount = shadowFrame.assignments.size();
    HashShadowValue(hash, viewCount);
    HashShadowValue(hash, assignmentCount);
    for (const auto &view : shadowFrame.views) {
        HashShadowValue(hash, view.lightId);
        HashShadowValue(hash, view.type);
        HashShadowValue(hash, view.subView);
        HashShadowValue(hash, view.viewProjection);
        HashShadowValue(hash, view.atlas);
        HashShadowValue(hash, view.nearPlane);
        HashShadowValue(hash, view.farPlane);
        HashShadowValue(hash, view.worldUnitsPerTexel);
        HashShadowValue(hash, view.filterRadiusTexels);
        HashShadowValue(hash, view.splitNear);
        HashShadowValue(hash, view.splitFar);
        HashShadowValue(hash, view.lightVector);
        HashShadowValue(hash, view.lightVectorIsPosition);
        HashShadowValue(hash, view.depthBiasTexels);
        HashShadowValue(hash, view.normalBiasTexels);
        HashShadowValue(hash, view.viewRight);
        HashShadowValue(hash, view.viewUp);
        HashShadowValue(hash, view.cullingMask);
    }
    for (const auto &assignment : shadowFrame.assignments) {
        HashShadowValue(hash, assignment.lightId);
        HashShadowValue(hash, assignment.firstView);
        HashShadowValue(hash, assignment.viewCount);
    }

    // A renderer that is not explicitly marked static remains a dynamic
    // shadow caster even while its transform happens to be unchanged.  The
    // first atlas publication after a view/graph boundary can precede the
    // next frame-local transform publication; freezing that one result made
    // Edit mode keep a corrupted low-resolution cascade while Play mode
    // repaired it immediately as gameplay advanced transform revisions.
    // Explicitly-static renderers still use the signature cache. GPU skinning
    // and particles mutate device buffers without necessarily changing the
    // ordinary transform revision, so they remain dynamic as well.
    for (const DrawCall &drawCall : m_cachedShadowRenderers.DrawCalls()) {
        if (ShadowCasterRequiresContinuousUpdate(drawCall)) {
            hasDynamicCaster = true;
            break;
        }
    }
    if (!hasDynamicCaster && m_particleDrawRegistry) {
        const auto particles = m_particleDrawRegistry->SnapshotShared(std::numeric_limits<int32_t>::min(),
                                                                      std::numeric_limits<int32_t>::max());
        if (particles) {
            hasDynamicCaster = std::any_of(particles->begin(), particles->end(), [](const auto &entry) {
                return entry.semantics.castShadows && entry.renderer && entry.renderer->CanCastShadows();
            });
        }
    }
    return hash == 0 ? 1 : hash;
}

void SceneRenderGraph::RefreshShadowAtlasUpdateState()
{
    if (!m_hasShadowCasterPass || m_cameraLightCollector.GetShadowFrame().views.empty()) {
        m_pendingShadowContentSignature = 0;
        m_shadowAtlasUpdateRequired = false;
        return;
    }
    bool hasDynamicCaster = false;
    m_pendingShadowContentSignature = ComputeShadowContentSignature(hasDynamicCaster);
    m_shadowAtlasUpdateRequired =
        hasDynamicCaster || !m_shadowAtlasValid || m_pendingShadowContentSignature != m_committedShadowContentSignature;
}

void SceneRenderGraph::CommitShadowAtlasUpdate()
{
    m_committedShadowContentSignature = m_pendingShadowContentSignature;
    m_shadowAtlasValid = true;
    m_shadowAtlasUpdateRequired = false;
}

bool SceneRenderGraph::CompleteSubmissionExecution(VkCommandBuffer commandBuffer)
{
    if (!m_graphBuilt || !m_renderGraph || commandBuffer == VK_NULL_HANDLE)
        return false;

    RecordParticleViewDiagnostics(commandBuffer);
    return true;
}

void SceneRenderGraph::CommitSubmissionExecution()
{
    if (m_hasCachedCameraVP && !m_cameraMatricesStaged) {
        InvalidateTemporalHistory();
    } else {
        CommitTemporalHistory();
        CommitCameraHistory();
        m_motionHistory.CommitFrame();
        m_historyResetPending = false;
    }
    ++m_executionCount;
    m_lastExecutedBuildRevision = m_graphBuildRevision;
}

void SceneRenderGraph::RefreshPerViewShadowDescriptor()
{
    if (!m_vkCore) {
        return;
    }

    VkDescriptorSet graphShadowDesc = GetPerViewDescriptorSet();
    const uint32_t frameIndex = m_vkCore->GetCurrentFrameSlot() % EngineConfig::kFrameSlotCapacity;
    auto &viewFrame = m_perViewFrames[frameIndex];
    VkDescriptorSet particleShadowDesc = viewFrame.ParticleSet();
    if (graphShadowDesc == VK_NULL_HANDLE || particleShadowDesc == VK_NULL_HANDLE) {
        return;
    }

    if (!m_hasShadowCasterPass) {
        if (!viewFrame.shadowBinding.usesDefaultTexture) {
            m_vkCore->ClearPerViewShadowMap(graphShadowDesc);
            m_vkCore->ClearPerViewShadowMap(particleShadowDesc);
            m_vkCore->ClearPerViewShadowMap(viewFrame.EditorOverlaySet());
            viewFrame.shadowBinding = {};
            viewFrame.shadowBinding.usesDefaultTexture = true;
        }
        return;
    }
    if (!m_renderGraph || !m_shadowMapInputHandle.IsValid()) {
        throw std::logic_error("Shadow-caster graph did not publish its shadowMap input");
    }

    VkImageView view = m_renderGraph->ResolveTextureView(m_shadowMapInputHandle);
    VkSampler shadowSampler = m_vkCore->GetShadowDepthSampler();
    if (view == VK_NULL_HANDLE || shadowSampler == VK_NULL_HANDLE) {
        throw std::runtime_error("Shadow-caster graph could not resolve its shadow image view and sampler");
    }

    const VkImageLayout imageLayout = m_shadowMapInputIsDepth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                                              : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    const auto &bound = viewFrame.shadowBinding;
    if (!bound.usesDefaultTexture && bound.imageView == view && bound.sampler == shadowSampler &&
        bound.layout == imageLayout)
        return;
    // The renderer has already waited for this frame slot. Geometry and
    // particles publish the camera-local shadow binding together so neither
    // domain can observe a half-updated per-view generation.
    m_vkCore->UpdatePerViewShadowMap(graphShadowDesc, view, shadowSampler, imageLayout);
    m_vkCore->UpdatePerViewShadowMap(particleShadowDesc, view, shadowSampler, imageLayout);
    m_vkCore->UpdatePerViewShadowMap(viewFrame.EditorOverlaySet(), view, shadowSampler, imageLayout);
    viewFrame.shadowBinding = {view, shadowSampler, imageLayout, false};
}

void SceneRenderGraph::InvalidatePerViewShadowBindings() noexcept
{
    for (auto &frame : m_perViewFrames)
        frame.shadowBinding = {};
}

// ============================================================================
// Debug
// ============================================================================

std::string SceneRenderGraph::GetDebugString() const
{
    std::string result =
        "SceneRenderGraph [RenderGraph Mode] (" + std::to_string(m_width) + "x" + std::to_string(m_height) + ")\n";
    result += "Graph Built: " + std::string(m_graphBuilt ? "Yes" : "No") + "\n";
    result += "Python Graph: " + std::string(m_hasPythonGraph ? "Yes" : "No") + "\n";
    result += "Python Graph Valid: " + std::string(m_pythonGraphValid ? "Yes" : "No") + "\n";
    if (m_hasPythonGraph) {
        result += "Passes (" + std::to_string(m_pythonGraphDesc.passes.size()) + "):\n";
        for (const auto &pass : m_pythonGraphDesc.passes) {
            result += "  " + pass.name + "\n";
        }
        if (m_vkCore && m_pythonGraphValid) {
            std::vector<std::string> fullscreenShaders;
            for (const auto &pass : m_pythonGraphDesc.passes) {
                for (const auto &command : pass.commands) {
                    if (command.type == GraphCommandType::FullscreenQuad && !command.shaderName.empty() &&
                        std::find(fullscreenShaders.begin(), fullscreenShaders.end(), command.shaderName) ==
                            fullscreenShaders.end()) {
                        fullscreenShaders.push_back(command.shaderName);
                    }
                }
            }
            if (!fullscreenShaders.empty()) {
                std::sort(fullscreenShaders.begin(), fullscreenShaders.end());
                result += "Fullscreen shader fingerprints (resident SPIR-V):\n";
                for (const auto &shader : fullscreenShaders) {
                    result += "  " + shader + "=" +
                              std::to_string(m_vkCore->GetShaderCodeFingerprint(shader, "fragment")) + "\n";
                }
            }
        }
        if (!m_parameterBlocks.empty()) {
            result += "Parameter blocks (authored defaults, before native runtime overrides) (" +
                      std::to_string(m_parameterBlocks.size()) + "):\n";
            for (const auto &[id, block] : m_parameterBlocks) {
                result += "  " + id + " [";
                for (size_t index = 0; index < block.names.size(); ++index) {
                    if (index != 0)
                        result += ", ";
                    const auto &value = (*block.values)[index];
                    result += block.names[index] + "=";
                    if (value.Kind() <= ShaderParameterKind::Bool)
                        result += std::to_string(value.ScalarFloat());
                    else
                        result += "(" + std::to_string(value.ByteSize() / sizeof(float)) + " components)";
                }
                result += "]\n";
            }
        }
    }

    if (!m_temporalHistories.empty()) {
        std::vector<std::string> temporalKeys;
        temporalKeys.reserve(m_temporalHistories.size());
        for (const auto &[key, history] : m_temporalHistories) {
            (void)history;
            temporalKeys.push_back(key);
        }
        std::sort(temporalKeys.begin(), temporalKeys.end());

        result += "Temporal histories (" + std::to_string(temporalKeys.size()) + "):\n";
        for (const auto &key : temporalKeys) {
            const auto &history = m_temporalHistories.at(key);
            result += "  " + key + " [" + std::to_string(history.width) + "x" + std::to_string(history.height) +
                      ", valid=" + (history.valid ? "Yes" : "No") + ", read=" + std::to_string(history.readIndex) +
                      ", write=" + std::to_string(history.readIndex ^ 1u) + "]\n";
        }
        result += "Temporal jitter: sample=" + std::to_string(m_temporalSampleIndex) + "/" +
                  std::to_string(kTemporalJitterSampleCount) + ", ndc=(" + std::to_string(m_temporalJitterNdc.x) +
                  ", " + std::to_string(m_temporalJitterNdc.y) + ")\n";
    }

    // Add underlying RenderGraph debug info
    if (m_renderGraph) {
        result += "\nUnderlying RenderGraph:\n";
        result += m_renderGraph->GetDebugString();
    }

    return result;
}

// ============================================================================
// Pass Output Access
// ============================================================================

// ============================================================================
// Private Methods
// ============================================================================

void SceneRenderGraph::ImportSceneTargetResources()
{
    m_viewportColorPublished = false;
    if (!m_sceneTarget || !m_renderGraph) {
        return;
    }

    const auto generation = m_viewportGeneration ? m_viewportGeneration : m_outputGeneration;
    if (generation) {
        const auto attachments = m_renderGraph->ImportRenderTexture("CameraOutput", generation);
        m_importedColorTarget = attachments.color;
        m_importedDepthTarget = attachments.depth;
        m_importedResolveTarget = attachments.resolve;
        m_renderGraph->SetBackbuffer(m_importedColorTarget);
        if (m_viewportGeneration)
            ImportViewportDestination();
        return;
    }

    m_importedColorTarget = m_renderGraph->SetBackbuffer(
        m_sceneTarget->GetMsaaColorImage(), m_sceneTarget->GetMsaaColorImageView(), m_sceneTarget->GetColorFormat(),
        m_width, m_height, m_sceneTarget->GetMsaaSampleCount(), rhi::TextureLayout::Undefined);

    if (m_sceneTarget->IsMsaaEnabled()) {
        m_importedResolveTarget =
            m_renderGraph->ImportResolveTarget(m_sceneTarget->GetColorImage(), m_sceneTarget->GetColorImageView(),
                                               m_sceneTarget->GetColorFormat(), m_width, m_height);
    } else {
        m_importedResolveTarget = {}; // Clear — no separate resolve target needed
    }

    // Color and depth belong to the same persistent output. Importing the
    // target-owned depth image lets ordered camera graphs preserve or clear a
    // shared depth domain according to CameraClearFlags instead of allocating
    // a private depth image per graph.
    m_importedDepthTarget = m_renderGraph->ImportTexture(
        "SceneDepth", m_sceneTarget->GetDepthImage(), m_sceneTarget->GetDepthImageView(),
        m_sceneTarget->GetDepthFormat(), m_width, m_height, m_sceneTarget->GetMsaaSampleCount());
}

vk::ResourceHandle SceneRenderGraph::CopyViewportRegion(const std::string &name, vk::ResourceHandle source,
                                                        vk::ResourceHandle destination, rhi::TextureAspect aspect,
                                                        bool restore)
{
    vk::ResourceHandle written;
    m_renderGraph->AddTransferPass(name, [=, &written](vk::PassBuilder &builder) {
        builder.SetQueueRole(rhi::QueueRole::Graphics);
        builder.TransferRead(source);
        written = builder.TransferWrite(destination);
        builder.SetSideEffect();
        const auto target = written;
        return [=](vk::RenderContext &context) {
            rhi::TextureCopyRegion region;
            region.aspect = aspect == rhi::TextureAspect::Depth &&
                                    rhi::IsStencilFormat(m_viewport.Output()->description.depthFormat)
                                ? rhi::TextureAspect::DepthStencil
                                : aspect;
            region.width = m_viewport.Width();
            region.height = m_viewport.Height();
            if (restore) {
                region.sourceX = m_viewport.X();
                region.sourceY = m_viewport.Y();
            } else {
                region.destinationX = m_viewport.X();
                region.destinationY = m_viewport.Y();
            }
            context.GetTransferCommandEncoder().CopyTexture(context.GetTextureHandle(source),
                                                            context.GetTextureHandle(target), region);
        };
    });
    return written;
}

void SceneRenderGraph::ImportViewportDestination()
{
    const auto output = m_renderGraph->ImportRenderTexture("ViewportDestination", m_viewport.Output());
    m_viewportOutputRaster = output.color;
    m_viewportOutputColor = output.resolve.IsValid() ? output.resolve : output.color;
    m_viewportOutputDepth = output.depth;
    if (m_cameraClearFlags == CameraClearFlags::DepthOnly || m_cameraClearFlags == CameraClearFlags::DontClear) {
        m_importedColorTarget = CopyViewportRegion("__ViewportRestoreColor", m_viewportOutputRaster,
                                                   m_importedColorTarget, rhi::TextureAspect::Color, true);
        m_renderGraph->SetBackbuffer(m_importedColorTarget);
    }
    if (m_cameraClearFlags == CameraClearFlags::DontClear && m_importedDepthTarget.IsValid())
        m_importedDepthTarget = CopyViewportRegion("__ViewportRestoreDepth", m_viewportOutputDepth,
                                                   m_importedDepthTarget, rhi::TextureAspect::Depth, true);
}

void SceneRenderGraph::ImportTemporalHistoryResources(std::unordered_map<std::string, vk::ResourceHandle> &handles)
{
    struct RequestedHistory
    {
        const GraphTextureDesc *read = nullptr;
        const GraphTextureDesc *write = nullptr;
    };
    std::unordered_map<std::string, RequestedHistory> requested;
    for (const auto &texture : m_pythonGraphDesc.textures) {
        if (texture.role == GraphTextureRole::TemporalRead)
            requested[texture.temporalKey].read = &texture;
        else if (texture.role == GraphTextureRole::TemporalWrite)
            requested[texture.temporalKey].write = &texture;
    }

    // Owners use the same allocation and in-flight retirement as public
    // RenderTextures. No second raw-image allocation/teardown implementation.
    for (auto it = m_temporalHistories.begin(); it != m_temporalHistories.end();) {
        if (requested.find(it->first) == requested.end())
            it = m_temporalHistories.erase(it);
        else
            ++it;
    }
    auto &device = m_vkCore->GetDeviceContext().GetRhiDevice();
    for (const auto &requestEntry : requested) {
        const auto &key = requestEntry.first;
        const auto &request = requestEntry.second;
        const auto &desc = *request.read; // Pair validated when the graph is published.
        const auto [width, height] = desc.ResolveExtent(m_width, m_height);
        auto &history = m_temporalHistories[key];
        if (!history.targets[0] || history.width != width || history.height != height ||
            history.format != desc.format) {
            rhi::RenderTextureDesc allocation;
            allocation.width = width;
            allocation.height = height;
            allocation.colorFormat = desc.format;
            std::array<std::shared_ptr<rhi::RenderTexture>, 2> targets;
            for (uint32_t index = 0; index < 2; ++index)
                targets[index] = std::make_shared<rhi::RenderTexture>(
                    device, "History/" + key + "/" + std::to_string(index), allocation);
            history.targets = std::move(targets);
            history.width = width;
            history.height = height;
            history.format = desc.format;
            history.valid = false;
            history.readIndex = 0;
        }
        history.readName = request.read->name;
        history.writeName = request.write->name;
        const auto read = history.targets[history.readIndex]->Acquire();
        const auto write = history.targets[history.readIndex ^ 1u]->Acquire();
        history.readHandle = m_renderGraph->ImportRenderTexture(history.readName, read).color;
        history.writeHandle = m_renderGraph->ImportRenderTexture(history.writeName, write).color;

        // The first read after creation/cut is defined zero, not uninitialized
        // device memory. The declared write carries ordering/layout hazards on
        // every execution; clearing itself occurs only when history is invalid.
        vk::ResourceHandle initializedRead;
        const auto readHandle = history.readHandle;
        m_renderGraph->AddTransferPass(
            "__HistoryInit/" + key, [this, key, readHandle, &initializedRead](vk::PassBuilder &builder) {
                initializedRead = builder.TransferWrite(readHandle);
                builder.SetQueueRole(rhi::QueueRole::Graphics);
                return [this, key, readHandle](vk::RenderContext &ctx) {
                    if (m_temporalHistories.at(key).valid)
                        return;
                    const VkClearColorValue zero{};
                    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    const auto image =
                        m_vkCore->GetDeviceContext().GetRhiDevice().Resolve(ctx.GetTextureHandle(readHandle));
                    vkCmdClearColorImage(ctx.GetCommandBuffer(), image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1,
                                         &range);
                };
            });
        handles[history.readName] = initializedRead;
        handles[history.writeName] = history.writeHandle;
    }
}

void SceneRenderGraph::BindTemporalHistoryResources()
{
    for (auto &[key, history] : m_temporalHistories) {
        const auto read = history.targets[history.readIndex]->Acquire();
        const auto write = history.targets[history.readIndex ^ 1u]->Acquire();
        m_renderGraph->UpdateImportedRenderTextureColor(history.readHandle, read);
        m_renderGraph->UpdateImportedRenderTextureColor(history.writeHandle, write);
    }
}

void SceneRenderGraph::CommitTemporalHistory()
{
    for (auto &[key, history] : m_temporalHistories) {
        (void)key;
        history.valid = true;
        history.readIndex ^= 1u;
    }
}

void SceneRenderGraph::UpdateMainPassClearSettings(CameraClearFlags clearFlags, const glm::vec4 &bgColor,
                                                   bool dithering, bool stopNaNs)
{
    m_hasCameraClearOverride = true;
    m_cameraClearFlags = clearFlags;
    // Authored camera background color is sRGB; the scene color target is
    // linear (encoded for display by the display encode pass).
    m_cameraBgColor = inx::color::SrgbToLinear(bgColor);
    m_cameraDithering = dithering;
    m_cameraStopNaNs = stopNaNs;

    if (m_prevClearStateValid && m_prevCameraClearFlags != clearFlags) {
        InvalidateTemporalHistory();
        m_needsRebuild = true;
    }
}

std::optional<GraphPassDesc> SceneRenderGraph::ResolveCameraClearOverride() const
{
    if (!m_hasCameraClearOverride)
        return std::nullopt;
    const auto persistent = [&](const std::string &name) {
        return std::any_of(
            m_pythonGraphDesc.textures.begin(), m_pythonGraphDesc.textures.end(),
            [&](const auto &texture) { return texture.name == name && texture.role == GraphTextureRole::Persistent; });
    };
    for (const auto &pass : m_pythonGraphDesc.passes) {
        if (!pass.clearColor || persistent(pass.writeDepth) || persistent(pass.resolveColor) ||
            std::any_of(pass.writeColors.begin(), pass.writeColors.end(),
                        [&](const auto &output) { return persistent(output.second); }))
            continue;
        auto effective = pass;
        const bool writesDepth = !pass.writeDepth.empty();
        const auto clearFlags =
            m_renderView.kind == rhi::RenderViewKind::Preview && (m_cameraClearFlags == CameraClearFlags::DepthOnly ||
                                                                  m_cameraClearFlags == CameraClearFlags::DontClear)
                ? CameraClearFlags::SolidColor
                : m_cameraClearFlags;
        switch (clearFlags) {
        case CameraClearFlags::Skybox:
            effective.clearColor = true;
            effective.clearDepth = writesDepth;
            break;
        case CameraClearFlags::SolidColor:
            effective.clearColor = true;
            effective.clearDepth = writesDepth;
            effective.clearColorR = m_cameraBgColor.r;
            effective.clearColorG = m_cameraBgColor.g;
            effective.clearColorB = m_cameraBgColor.b;
            effective.clearColorA = m_cameraBgColor.a;
            break;
        case CameraClearFlags::DepthOnly:
            effective.clearColor = false;
            effective.clearDepth = writesDepth;
            break;
        case CameraClearFlags::DontClear:
            effective.clearColor = false;
            effective.clearDepth = false;
            break;
        }
        return effective;
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// BuildRenderGraph helpers
// ---------------------------------------------------------------------------

bool SceneRenderGraph::RegisterGraphTextures(uint32_t width, uint32_t height,
                                             const std::vector<GraphAssetTexture> &assetTextures,
                                             std::unordered_map<std::string, vk::ResourceHandle> &customRTHandles)
{
    // Camera post processing uses viewport-local images. Overlay UI and its
    // following display-space effects operate on the composited full output.
    // Allocate a tail-only intermediate in that domain from its first writer,
    // rather than stretching a small camera texture over the screen later.
    std::unordered_set<std::string> screenTextures;
    if (m_viewportGeneration && m_screenUIOutput) {
        std::unordered_set<std::string> writtenTextures;
        bool screenDomain = false;
        for (const auto &pass : m_pythonGraphDesc.passes) {
            const auto *command = pass.PrimaryCommand();
            if (command && command->type == GraphCommandType::DrawScreenUI && command->screenUIList != 0)
                screenDomain = true;
            const auto rememberWrite = [&](const std::string &name) {
                if (!name.empty() && writtenTextures.insert(name).second && screenDomain)
                    screenTextures.insert(name);
            };
            for (const auto &[slot, name] : pass.writeColors)
                rememberWrite(name);
            rememberWrite(pass.writeDepth);
            rememberWrite(pass.resolveColor);
            if (command && command->CopiesToTexture())
                rememberWrite(command->destinationResource);
        }
    }
    std::unordered_map<std::string, ImportedTextureBinding> bindings;
    m_drawTextureInputs.clear();
    for (const auto &tex : m_pythonGraphDesc.textures) {
        if (tex.role != GraphTextureRole::Persistent)
            continue;
        const auto generation = tex.renderTexture->Acquire();
        const auto attachments = m_renderGraph->ImportRenderTexture(tex.name, generation);
        const auto *image = ResolveGraphTextureAttachment(*generation, tex.attachment);
        const auto handle = tex.attachment == GraphTextureAttachment::Depth ? attachments.depth
                            : tex.attachment == GraphTextureAttachment::Resolve && attachments.resolve.IsValid()
                                ? attachments.resolve
                                : attachments.color;
        customRTHandles[tex.name] = handle;
        bindings.emplace(tex.name, ImportedTextureBinding{image->GetTexture(), image->GetSampler(), image->GetFormat(),
                                                          generation->revision, 0, image});
    }
    const auto &device = m_vkCore->GetDeviceContext().GetRhiDevice();
    m_graphTexturePublications.reserve(assetTextures.size());
    for (const auto &[descriptor, publication, contentGeneration] : assetTextures) {
        const auto &tex = *descriptor;
        if (!publication || !publication->IsValid()) {
            INXLOG_ERROR("SceneRenderGraph: Texture asset '", tex.name, "' (GUID ", tex.assetGuid,
                         ") could not be resolved");
            return false;
        }
        const bool publicationIsVolume = publication->GetViewDesc().dimension == rhi::TextureViewDimension::Texture3D;
        if (publicationIsVolume != tex.isVolume) {
            INXLOG_ERROR("SceneRenderGraph: Texture asset '", tex.name,
                         "' changed dimension after the graph was authored");
            return false;
        }
        rhi::TextureDesc description;
        if (!device.GetTextureDesc(publication->GetTexture(), description)) {
            INXLOG_ERROR("SceneRenderGraph: Texture asset '", tex.name, "' has no resident description");
            return false;
        }
        const auto handle = m_renderGraph->ImportTexture(tex.name, publication->GetTexture(), publication->GetView(),
                                                         rhi::ToVkFormat(description.format), description.width,
                                                         description.height, rhi::ToVkSampleCount(description.samples),
                                                         description.depthOrLayers, tex.isVolume);
        if (!handle.IsValid()) {
            INXLOG_ERROR("SceneRenderGraph: Texture asset '", tex.name, "' could not be imported");
            return false;
        }
        m_graphTexturePublications.push_back(publication);
        m_renderGraph->SetResourceInitialState(handle, rhi::TextureLayout::ShaderReadOnly, rhi::Access::ShaderRead,
                                               rhi::PipelineStage::FragmentShader);
        customRTHandles[tex.name] = handle;
        bindings.emplace(tex.name, ImportedTextureBinding{publication->GetTexture(), publication->GetSampler(),
                                                          publication->GetFormat(), publication->GetRevision(),
                                                          contentGeneration});
    }
    const bool sourceReplaced = std::any_of(bindings.begin(), bindings.end(), [&](const auto &entry) {
        const auto previous = m_importedTextureBindings.find(entry.first);
        return previous != m_importedTextureBindings.end() && previous->second != entry.second;
    });
    // A pending file or upload leaves the previously bound history untouched.
    if (sourceReplaced)
        InvalidateTemporalHistory();
    m_importedTextureBindings = std::move(bindings);
    for (const auto &read : m_materialTextureReads) {
        m_drawTextureInputs[read.passName].push_back(read.generation);
    }
    // Only the root Camera depth is shared. All other transient declarations
    // own a distinct image, regardless of matching size, format or sample count.
    for (const auto &tex : m_pythonGraphDesc.textures) {
        if (tex.IsViewDepth()) {
            customRTHandles[tex.name] = m_importedDepthTarget;
        } else if (tex.role == GraphTextureRole::Transient) {
            const bool screenDomain = screenTextures.find(tex.name) != screenTextures.end();
            const auto [texW, texH] =
                tex.ResolveExtent(screenDomain ? m_viewport.Output()->description.width : width,
                                  screenDomain ? m_viewport.Output()->description.height : height);
            const uint32_t requestedSamples =
                tex.samples == 0 ? static_cast<uint32_t>(m_sceneTarget->GetMsaaSampleCount()) : tex.samples;
            vk::ResourceHandle handle = m_renderGraph->RegisterTransientTexture(
                tex.name, texW, texH, rhi::ToVkFormat(tex.format),
                rhi::ToVkSampleCount(ToRhiSampleCount(static_cast<int>(requestedSamples))), true);
            customRTHandles[tex.name] = handle;
        }
    }
    return true;
}

vk::ResourceHandle SceneRenderGraph::AppendAutoPass(const std::string &name, vk::ResourceHandle colorTarget,
                                                    vk::ResourceHandle depthTarget, uint32_t width, uint32_t height)
{
    auto callbackIt = m_pythonCallbacks.find(name);
    if (callbackIt == m_pythonCallbacks.end())
        return colorTarget;

    auto callback = callbackIt->second;
    const vk::ResourceHandle rendererListHandle = m_visibleRendererList;
    InxVkCoreModular *vkCore = m_vkCore;
    vk::ResourceHandle writtenColor;
    m_renderGraph->AddPass(name, [=, &writtenColor](vk::PassBuilder &builder) {
        writtenColor = builder.WriteColor(colorTarget, 0);
        if (depthTarget.IsValid()) {
            builder.ReadDepth(depthTarget);
        }
        if (rendererListHandle.IsValid()) {
            builder.ReadRendererList(rendererListHandle);
            builder.SkipCallbackWhenRendererListsEmpty();
        }
        builder.SetRenderArea(width, height);
        return [callback, width, height, rendererListHandle, vkCore](vk::RenderContext &ctx) {
            if (rendererListHandle.IsValid()) {
                const RendererList *rendererList = ctx.GetRendererList(rendererListHandle);
                const auto *drawCalls = rendererList ? &rendererList->DrawCalls() : nullptr;
                if (!vkCore->UsesDrawCalls(drawCalls))
                    vkCore->SetDrawCalls(drawCalls);
            }
            if (callback) {
                callback(ctx, width, height);
            }
        };
    });
    return writtenColor.IsValid() ? writtenColor : colorTarget;
}

vk::ResourceHandle SceneRenderGraph::AppendEditorOutline(vk::ResourceHandle displayTarget)
{
    if (!m_outlinePassesEnabled || !m_outlineRenderer || !displayTarget.IsValid() || !m_visibleRendererList.IsValid() ||
        !m_sceneTarget || !m_sceneTarget->IsReady()) {
        return displayTarget;
    }

    const uint32_t width = m_width;
    const uint32_t height = m_height;
    const vk::ResourceHandle rendererList = m_visibleRendererList;
    vk::ResourceHandle mask;
    vk::ResourceHandle writtenMask;
    m_renderGraph->AddPass("__EditorOutlineMask", [this, width, height, rendererList, &mask,
                                                   &writtenMask](vk::PassBuilder &builder) {
        DeclareMaterialBufferReads(builder, "__EditorOutlineMask");
        mask = builder.ImportTexture("__EditorOutlineMaskTexture", m_sceneTarget->GetOutlineMaskImage(),
                                     m_sceneTarget->GetOutlineMaskImageView(), VK_FORMAT_R8G8B8A8_UNORM, width, height);
        if (mask.IsValid()) {
            m_renderGraph->SetResourceInitialState(mask, rhi::TextureLayout::ShaderReadOnly, rhi::Access::ShaderRead,
                                                   rhi::PipelineStage::FragmentShader);
            writtenMask = builder.WriteColor(mask, 0);
            builder.SetClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        }
        builder.ReadRendererList(rendererList);
        builder.SetDepthTest(false);
        builder.SetRenderArea(width, height);
        return [this, rendererList](vk::RenderContext &context) {
            if (!m_outlineRenderer)
                return;
            const RendererList *list = context.GetRendererList(rendererList);
            if (list)
                m_outlineRenderer->RecordMaskDraws(
                    context.GetCommandBuffer(), list->DrawCalls(), GetPerViewBindGroup(),
                    m_vkCore->GetDeviceContext().GetRhiDevice().Resolve(GetViewGlobalsBinding().buffer));
        };
    });

    if (!writtenMask.IsValid())
        return displayTarget;

    vk::ResourceHandle composited;
    m_renderGraph->AddPass("__EditorOutlineComposite",
                           [this, width, height, displayTarget, writtenMask, &composited](vk::PassBuilder &builder) {
                               builder.Read(writtenMask, rhi::PipelineStage::FragmentShader);
                               composited = builder.WriteColor(displayTarget, 0);
                               builder.SetDepthTest(false);
                               builder.SetRenderArea(width, height);
                               return [this](vk::RenderContext &context) {
                                   if (m_outlineRenderer)
                                       m_outlineRenderer->RecordCompositeDraw(context.GetCommandBuffer());
                               };
                           });
    return composited.IsValid() ? composited : displayTarget;
}

void SceneRenderGraph::FinalizeGraphOutput(const std::unordered_map<std::string, vk::ResourceHandle> &customRTHandles)
{
    vk::ResourceHandle graphOutput;
    if (m_hasPythonGraph && !m_pythonGraphDesc.outputTexture.empty()) {
        auto texIt = std::find_if(m_pythonGraphDesc.textures.begin(), m_pythonGraphDesc.textures.end(),
                                  [&](const GraphTextureDesc &t) { return t.name == m_pythonGraphDesc.outputTexture; });
        if (texIt != m_pythonGraphDesc.textures.end()) {
            if (!texIt->IsViewColor() && !texIt->IsDepth()) {
                auto rtIt = customRTHandles.find(m_pythonGraphDesc.outputTexture);
                if (rtIt != customRTHandles.end())
                    graphOutput = rtIt->second;
            }
        }
    }

    vk::ResourceHandle displayTarget = m_importedColorTarget;
    vk::ResourceHandle msaaSource = m_importedColorTarget;
    vk::ResourceHandle exportDepth = m_importedDepthTarget;
    if (m_sceneTarget->IsMsaaEnabled() && msaaSource.IsValid() && m_importedResolveTarget.IsValid()) {
        const auto resolveDestination = m_importedResolveTarget;
        const uint32_t width = m_viewportColorPublished ? m_viewport.Output()->description.width : m_width;
        const uint32_t height = m_viewportColorPublished ? m_viewport.Output()->description.height : m_height;
        vk::ResourceHandle resolvedVersion;
        m_renderGraph->AddTransferPass("__SceneFinalMsaaResolve", [=, &resolvedVersion](vk::PassBuilder &builder) {
            // Vulkan image resolve requires a graphics-capable command pool,
            // while its resource accesses remain transfer operations.
            builder.SetQueueRole(rhi::QueueRole::Graphics);
            builder.TransferRead(msaaSource);
            resolvedVersion = builder.TransferWrite(resolveDestination);
            builder.SetRenderArea(width, height);
            return [=](vk::RenderContext &context) {
                context.GetTransferCommandEncoder().ResolveTexture(
                    context.GetTextureHandle(msaaSource), context.GetTextureHandle(resolvedVersion),
                    {rhi::TextureAspect::Color, 0, 0, 0, 0, width, height, 1});
            };
        });
        if (resolvedVersion.IsValid()) {
            m_importedResolveTarget = resolvedVersion;
            displayTarget = resolvedVersion;
        }
    }

    if (!graphOutput.IsValid())
        graphOutput = displayTarget;

    if (m_viewportGeneration && !m_viewportColorPublished) {
        displayTarget = CopyViewportRegion("__ViewportPublishColor", graphOutput, m_viewportOutputColor,
                                           rhi::TextureAspect::Color, false);
        graphOutput = displayTarget;
        if (m_sceneTarget->IsMsaaEnabled())
            msaaSource = CopyViewportRegion("__ViewportPublishMsaa", msaaSource, m_viewportOutputRaster,
                                            rhi::TextureAspect::Color, false);
        if (exportDepth.IsValid())
            exportDepth = CopyViewportRegion("__ViewportPublishDepth", exportDepth, m_viewportOutputDepth,
                                             rhi::TextureAspect::Depth, false);
    }

    if (displayTarget.IsValid()) {
        const bool preserveMsaaAttachment = m_sceneTarget->IsMsaaEnabled() && msaaSource.IsValid();
        m_renderGraph->AddPresentPass("__SceneOutputExport", [=](vk::PassBuilder &builder) {
            // Scene/Game targets are sampled by ImGui, capture and later view
            // consumers. The graph, not an out-of-band Vulkan barrier, owns
            // their published layout.
            builder.Read(displayTarget,
                         m_outputTexture ? rhi::RenderTextureSampleStages : rhi::PipelineStage::FragmentShader);
            if (graphOutput.IsValid() && graphOutput != displayTarget)
                builder.Read(graphOutput, rhi::PipelineStage::FragmentShader);
            if (preserveMsaaAttachment)
                builder.PrepareColorAttachment(msaaSource);
            if (exportDepth.IsValid()) {
                if (m_outputGeneration && m_outputGeneration->description.sampledDepth &&
                    m_outputGeneration->description.samples == rhi::SampleCount::One)
                    builder.ReadSampledDepth(exportDepth, rhi::RenderTextureSampleStages);
                else
                    builder.PrepareDepthStencilAttachment(exportDepth);
            }
            builder.SetSideEffect();
            return [](vk::RenderContext &) {};
        });
    }

    if (graphOutput.IsValid())
        m_renderGraph->SetOutput(graphOutput);
}

void SceneRenderGraph::BuildRenderGraph(bool buffersPrepared, const std::vector<GraphAssetTexture> &assetTextures)
{
    if (m_hasPythonGraph && !m_pythonGraphValid) {
        m_graphBuilt = false;
        return;
    }
    if (!m_renderGraph || !m_sceneTarget || !m_vkCore) {
        INXLOG_WARN("SceneRenderGraph::BuildRenderGraph - Missing required components");
        return;
    }

    RetireImportedTextureAssets();
    m_sceneDepthResolver.RetireBindings(m_vkCore->GetRetirementQueue());
    m_renderGraph->Reset();
    m_graphBuilt = false;
    m_authorPassIndices.clear();
    m_authorShaderGuids.clear();
    m_executedEventPasses.clear();
    m_graphBufferImports.clear();
    m_materialBufferImports.clear();
    if (!ProveGraphBufferContents())
        return;
    if (!buffersPrepared && !PrepareGraphBuffers()) {
        if (m_graphStorageDeferred)
            RetireWorkInstances();
        return;
    }
    m_committedShadowContentSignature = 0;
    m_pendingShadowContentSignature = 0;
    m_shadowAtlasValid = false;
    m_shadowAtlasUpdateRequired = true;
    m_renderView.color = {};
    m_renderView.depth = {};
    m_renderView.motion = {};
    // The renderer has waited only for the current frame slot. Clear both
    // geometry and particle views for that slot before retiring graph-owned
    // shadow images; the other slot is republished after its own fence.
    const uint32_t currentFrame = m_vkCore->GetCurrentFrameSlot() % EngineConfig::kFrameSlotCapacity;
    auto &currentViewFrame = m_perViewFrames[currentFrame];
    if (currentViewFrame.GeometrySet() != VK_NULL_HANDLE)
        m_vkCore->ClearPerViewShadowMap(currentViewFrame.GeometrySet());
    if (currentViewFrame.ParticleSet() != VK_NULL_HANDLE)
        m_vkCore->ClearPerViewShadowMap(currentViewFrame.ParticleSet());
    if (currentViewFrame.EditorOverlaySet() != VK_NULL_HANDLE)
        m_vkCore->ClearPerViewShadowMap(currentViewFrame.EditorOverlaySet());
    currentViewFrame.shadowBinding = {};
    currentViewFrame.shadowBinding.usesDefaultTexture = true;

    // Graph topology and material pipeline compatibility are independent.
    // Effect-stack edits commonly rebuild transient graph resources while the
    // attachment formats and sample count remain unchanged. Material pass
    // descriptors already key their cache by compile target, formats, samples
    // and render state; invalidating every material here discarded valid GPU
    // pipelines and made a runtime effect/pipeline switch compile the whole
    // scene again. True compatibility changes (currently MSAA publication)
    // are handled transactionally by MaterialPipelineManager itself.

    m_graphBuilt = false;
    m_shadowMapInputHandle = {};
    m_shadowMapInputIsDepth = false;
    m_visibleRendererList = {};
    m_shadowRendererList = {};

    const auto retireCuller = [this](std::shared_ptr<particle::ParticleGpuCuller> culler) {
        if (!culler)
            return;
        m_vkCore->GetRetirementQueue().Retire([culler = std::move(culler)]() mutable { culler.reset(); });
    };
    const auto retireSorter = [this](std::shared_ptr<particle::ParticleGpuSorter> sorter) {
        if (sorter)
            m_vkCore->GetRetirementQueue().Retire([sorter = std::move(sorter)]() mutable { sorter.reset(); });
    };
    particle::ParticleGpuDrawRegistry::SnapshotHandle particleSnapshot;
    if (m_particleDrawRegistry) {
        particleSnapshot = m_particleDrawRegistry->SnapshotShared(std::numeric_limits<int32_t>::min(),
                                                                  std::numeric_limits<int32_t>::max());
    }
    const particle::ParticleGpuDrawRegistry::SnapshotEntries emptyParticleEntries;
    const auto &particleEntries = particleSnapshot ? *particleSnapshot : emptyParticleEntries;
    std::unordered_set<uint64_t> activeCullers;
    activeCullers.reserve(particleEntries.size());
    for (const auto &entry : particleEntries) {
        if (!entry.cullProgram || !entry.cullProgram->IsValid())
            continue;
        activeCullers.insert(entry.id);
        auto existing = m_particleCullers.find(entry.id);
        if (existing != m_particleCullers.end() && existing->second && existing->second->Capacity() == entry.capacity &&
            existing->second->VertexCount() == entry.renderer->VertexCount() &&
            existing->second->VisibilityBuffer() == entry.visibility &&
            existing->second->SourceIndirectBuffer() == entry.indirectArguments &&
            existing->second->SourceIndexBuffer() == entry.renderIndices &&
            existing->second->BoundsBuffer() == entry.bounds &&
            existing->second->SimulationControlBuffer() == entry.simulationControl &&
            existing->second->Mode() == entry.cullMode) {
            continue;
        }

        auto culler = std::make_shared<particle::ParticleGpuCuller>();
        particle::GpuParticleCullerDesc desc;
        desc.capacity = entry.capacity;
        desc.vertexCount = entry.renderer->VertexCount();
        desc.visibility = entry.visibility;
        desc.ribbonInstances = entry.instances;
        desc.sourceIndirectArguments = entry.indirectArguments;
        desc.sourceIndices = entry.renderIndices;
        desc.bounds = entry.bounds;
        desc.simulationControl = entry.simulationControl;
        desc.mode = entry.cullMode;
        desc.program = entry.cullProgram->View();
        if (!culler->Create(m_vkCore->GetDeviceContext().GetRhiDevice(), desc)) {
            INXLOG_ERROR("SceneRenderGraph: failed to create per-view culler for GPU particle output ", entry.id);
            activeCullers.erase(entry.id);
            continue;
        }
        if (existing != m_particleCullers.end()) {
            retireCuller(std::move(existing->second));
            existing->second = std::move(culler);
        } else {
            m_particleCullers.emplace(entry.id, std::move(culler));
        }
    }
    for (auto it = m_particleCullers.begin(); it != m_particleCullers.end();) {
        if (activeCullers.find(it->first) != activeCullers.end()) {
            ++it;
            continue;
        }
        retireCuller(std::move(it->second));
        it = m_particleCullers.erase(it);
    }

    std::unordered_set<uint64_t> activeSorters;
    activeSorters.reserve(particleEntries.size());
    for (const auto &entry : particleEntries) {
        if (entry.semantics.sortMode == particle::ParticleSortMode::None)
            continue;
        const auto cullerIt = m_particleCullers.find(entry.id);
        if (cullerIt == m_particleCullers.end() || !cullerIt->second || !cullerIt->second->IsValid()) {
            INXLOG_ERROR("SceneRenderGraph: sorted GPU particle output ", entry.id, " has no valid per-view culler");
            continue;
        }
        if (!entry.sortProgram || !entry.sortProgram->IsValid()) {
            INXLOG_ERROR("SceneRenderGraph: sorted GPU particle output ", entry.id, " has no valid sorting program");
            continue;
        }
        const auto &culler = cullerIt->second;
        activeSorters.insert(entry.id);
        auto existing = m_particleSorters.find(entry.id);
        if (existing != m_particleSorters.end() && existing->second && existing->second->Capacity() == entry.capacity &&
            existing->second->VisibilityBuffer() == entry.visibility &&
            existing->second->IndirectBuffer() == culler->DrawIndirectBuffer() &&
            existing->second->SourceIndexBuffer() == culler->VisibleIndexBuffer() &&
            existing->second->DispatchBuffer() == culler->SortDispatchBuffer()) {
            continue;
        }

        auto sorter = std::make_shared<particle::ParticleGpuSorter>();
        particle::GpuParticleSorterDesc desc;
        desc.capacity = entry.capacity;
        desc.visibility = entry.visibility;
        desc.indirectArguments = culler->DrawIndirectBuffer();
        desc.sourceIndices = culler->VisibleIndexBuffer();
        desc.dispatchArguments = culler->SortDispatchBuffer();
        desc.program = entry.sortProgram->View();
        if (!sorter->Create(m_vkCore->GetDeviceContext().GetRhiDevice(), desc)) {
            INXLOG_ERROR("SceneRenderGraph: failed to create per-view sorter for GPU particle output ", entry.id);
            activeSorters.erase(entry.id);
            continue;
        }
        if (existing != m_particleSorters.end()) {
            retireSorter(std::move(existing->second));
            existing->second = std::move(sorter);
        } else {
            m_particleSorters.emplace(entry.id, std::move(sorter));
        }
    }
    for (auto it = m_particleSorters.begin(); it != m_particleSorters.end();) {
        if (activeSorters.find(it->first) != activeSorters.end()) {
            ++it;
            continue;
        }
        retireSorter(std::move(it->second));
        it = m_particleSorters.erase(it);
    }

    // Reconcile ownership only when the graph topology changes. An output can
    // replace its renderer while retaining its culler, or replace cull/sort
    // buffers while retaining its renderer; both invalidate the old binding.
    for (const auto &entry : particleEntries) {
        const auto culler = m_particleCullers.find(entry.id);
        if (culler == m_particleCullers.end())
            continue;
        const auto sorter = m_particleSorters.find(entry.id);
        const auto indices =
            sorter == m_particleSorters.end() ? culler->second->VisibleIndexBuffer() : sorter->second->SortedIndices();
        auto [binding, inserted] =
            m_particleViewBindings.try_emplace(entry.id, ParticleViewBinding{entry.renderer, indices});
        if (!inserted && (binding->second.renderer != entry.renderer || binding->second.indices != indices)) {
            binding->second.renderer->ReleaseViewBinding(binding->second.indices);
            binding->second = {entry.renderer, indices};
        }
    }
    for (auto binding = m_particleViewBindings.begin(); binding != m_particleViewBindings.end();) {
        if (activeCullers.find(binding->first) != activeCullers.end()) {
            ++binding;
        } else {
            binding->second.renderer->ReleaseViewBinding(binding->second.indices);
            binding = m_particleViewBindings.erase(binding);
        }
    }

    if (!m_hasPythonGraph) {
        return;
    }

    ImportSceneTargetResources();
    m_visibleRendererList = m_renderGraph->ImportRendererList("VisibleRenderers", &m_cachedRenderers);
    m_shadowRendererList = m_renderGraph->ImportRendererList("ShadowRenderers", &m_cachedShadowRenderers);

    std::unordered_map<std::string, vk::ResourceHandle> customRTHandles;
    std::unordered_map<std::string, vk::ResourceHandle> bufferHandles;
    std::array<vk::ResourceHandle, EngineConfig::kFrameSlotCapacity> viewLightListHandles{};
    uint64_t viewLightListByteSize = 0;
    const bool hasViewLightList = std::any_of(m_pythonGraphDesc.buffers.begin(), m_pythonGraphDesc.buffers.end(),
                                              [](const GraphBufferDesc &buffer) { return buffer.viewLightList; });
    if (hasViewLightList) {
        for (uint32_t frameIndex = 0; frameIndex < EngineConfig::kFrameSlotCapacity; ++frameIndex) {
            const auto lights = m_cameraCanonicalLights.Frame(frameIndex);
            if (!lights.buffer.IsValid() || lights.capacityBytes < sizeof(uint32_t) * 4) {
                INXLOG_ERROR("SceneRenderGraph: view light list is unavailable for frame slot ", frameIndex);
                return;
            }
            viewLightListByteSize = std::max(viewLightListByteSize, lights.capacityBytes);
        }
    }
    if (!m_pythonGraphDesc.passes.empty() && !RegisterGraphTextures(m_width, m_height, assetTextures, customRTHandles))
        return;
    ImportTemporalHistoryResources(customRTHandles);
    if (!m_pythonGraphDesc.passes.empty()) {
        std::unordered_map<std::string, const GraphTextureDesc *> texDescMap;
        for (const auto &tex : m_pythonGraphDesc.textures) {
            texDescMap[tex.name] = &tex;
        }

        // Import the canonical camera light upload once per frame slot. A
        // fullscreen consumer selects the active slot at execution time,
        // so descriptor bindings never retain another View's or frame's
        // light data.
        if (hasViewLightList) {
            for (uint32_t frameIndex = 0; frameIndex < EngineConfig::kFrameSlotCapacity; ++frameIndex) {
                const auto lights = m_cameraCanonicalLights.Frame(frameIndex);
                const std::string prefix = "View/LightList/Frame" + std::to_string(frameIndex);
                m_renderGraph->AddComputePass(prefix, [&, frameIndex, prefix, lights](vk::PassBuilder &builder) {
                    viewLightListHandles[frameIndex] =
                        builder.ImportBuffer(prefix, lights.buffer, lights.capacityBytes);
                    m_renderGraph->SetResourceInitialState(viewLightListHandles[frameIndex],
                                                           rhi::TextureLayout::Undefined, rhi::Access::HostWrite,
                                                           rhi::PipelineStage::Host);
                    builder.ReadStorageBuffer(viewLightListHandles[frameIndex], rhi::PipelineStage::FragmentShader);
                    return [](vk::RenderContext &) {};
                });
            }
        }

        const auto &sortedPasses = m_pythonGraphDesc.passes;

        const uint32_t cameraWorldUIMask = m_cachedCamera ? m_cachedCamera->GetCullingMask() : 0xffffffffu;
        std::unordered_map<std::string, WorldUIDepthReplay> worldUIOpaqueSources;
        for (const auto &candidate : sortedPasses) {
            const auto *candidateCommand = candidate.PrimaryCommand();
            if (!candidateCommand || candidateCommand->type != GraphCommandType::DrawWorldUI || !m_screenUIRenderer ||
                !m_screenUIRenderer->HasSelectiveWorldOcclusion(cameraWorldUIMask & candidateCommand->worldUILayerMask))
                continue;
            auto source = BuildWorldUIDepthReplay(m_pythonGraphDesc, m_cameraClearFlags == CameraClearFlags::DontClear,
                                                  candidate.name);
            if (!source) {
                INXLOG_ERROR("Selective World UI occlusion requires one cleared Forward scene-depth writer; "
                             "pass '",
                             candidate.name, "' or camera depth-preservation policy cannot be replayed exactly");
                return;
            }
            worldUIOpaqueSources.emplace(candidate.name, std::move(*source));
        }

        uint32_t width = m_width;
        uint32_t height = m_height;
        VkFormat depthFormat = m_sceneTarget->GetDepthFormat();
        VkSampleCountFlagBits msaaSamples = m_sceneTarget->GetMsaaSampleCount();

        // Capture vkCore for pass lambdas (avoids capturing 'this')
        InxVkCoreModular *vkCore = m_vkCore;

        // All camera graphs targeting this output import the same persistent
        // depth attachment. The first camera normally clears it; later
        // cameras may clear or preserve it through their clear flags.
        vk::ResourceHandle sharedDepth = m_importedDepthTarget;

        for (size_t index = 0; index < m_pythonGraphDesc.buffers.size(); ++index) {
            const auto &buffer = m_pythonGraphDesc.buffers[index];
            if (buffer.viewLightList) {
                bufferHandles[buffer.name] = viewLightListHandles[0];
                continue;
            }
            if (const auto owner = ResolveGraphBufferOwner(buffer)) {
                const auto &allocation = m_computeBufferReads.at(owner.get());
                if (allocation->GetBuffer().Device() != m_renderView.device) {
                    INXLOG_ERROR("SceneRenderGraph: imported buffer '", buffer.name, "' belongs to another device");
                    return;
                }
                const bool graphLocal = allocation->IsGraphWritable() && !owner->IsFrozen();
                const auto handle = m_renderGraph->ImportBuffer(buffer.name, allocation->GetBuffer(),
                                                                allocation->GetByteSize(), graphLocal);
                if (!handle.IsValid()) {
                    INXLOG_ERROR("SceneRenderGraph: imported buffer '", buffer.name, "' is unavailable");
                    return;
                }
                bufferHandles[buffer.name] = handle;
                m_graphBufferImports.emplace_back(handle, index);
                m_renderGraph->SetResourceInitialState(handle, rhi::TextureLayout::Undefined,
                                                       rhi::Access::MemoryRead | rhi::Access::MemoryWrite,
                                                       rhi::PipelineStage::AllCommands, GraphBufferWriter(*allocation));
            } else {
                bufferHandles[buffer.name] = m_renderGraph->RegisterTransientBuffer(
                    buffer.name, buffer.byteSize, static_cast<GraphBufferUsage>(buffer.usage));
            }
        }

        struct ParticleGraphResources
        {
            vk::ResourceHandle instances;
            vk::ResourceHandle visibility;
            vk::ResourceHandle sourceIndirectArguments;
            vk::ResourceHandle sourceRenderIndices;
            vk::ResourceHandle bounds;
            vk::ResourceHandle simulationControl;
            vk::ResourceHandle renderIndices;
            vk::ResourceHandle indirectArguments;
            vk::ResourceHandle sortDispatchArguments;
            std::array<vk::ResourceHandle, 2> keys;
            std::array<vk::ResourceHandle, 2> indices;
            vk::ResourceHandle histogram;
            vk::ResourceHandle blockOffsets;
            vk::ResourceHandle globalOffsets;
            rhi::BufferHandle drawRenderIndices;
        };
        std::unordered_map<uint64_t, ParticleGraphResources> particleGraphResources;
        particleGraphResources.reserve(m_particleCullers.size());
        for (const auto &entry : particleEntries) {
            const auto cullerIt = m_particleCullers.find(entry.id);
            if (cullerIt == m_particleCullers.end() || !cullerIt->second || !cullerIt->second->IsValid())
                continue;

            auto culler = cullerIt->second;
            ParticleGraphResources resources;
            resources.drawRenderIndices = culler->VisibleIndexBuffer();
            const std::string prefix = "GpuParticleCull/" + std::to_string(entry.id);
            m_renderGraph->AddComputePass(prefix + "/Reset", [&, culler, entry, prefix](vk::PassBuilder &builder) {
                const uint64_t elementBytes = static_cast<uint64_t>(entry.capacity) * sizeof(uint32_t);
                resources.instances = builder.ImportBuffer(prefix + "/Instances", entry.instances,
                                                           static_cast<uint64_t>(entry.capacity) *
                                                               particle::ParticleGpuRuntime::RenderInstanceStride);
                resources.visibility = builder.ImportBuffer(prefix + "/Visibility", entry.visibility,
                                                            static_cast<uint64_t>(entry.capacity) *
                                                                particle::ParticleGpuRuntime::VisibilityInstanceStride);
                resources.sourceIndirectArguments =
                    builder.ImportBuffer(prefix + "/SourceIndirect", entry.indirectArguments, 16);
                resources.sourceRenderIndices =
                    builder.ImportBuffer(prefix + "/SourceIndices", entry.renderIndices, elementBytes);
                resources.bounds = builder.ImportBuffer(prefix + "/Bounds", entry.bounds,
                                                        particle::ParticleGpuBounds::BoundsBufferBytes);
                resources.simulationControl =
                    builder.ImportBuffer(prefix + "/SimulationControl", entry.simulationControl,
                                         sizeof(particle::GpuParticleSimulationControl));
                resources.renderIndices =
                    builder.ImportBuffer(prefix + "/VisibleIndices", culler->VisibleIndexBuffer(), elementBytes);
                resources.indirectArguments =
                    builder.ImportBuffer(prefix + "/DrawIndirect", culler->DrawIndirectBuffer(), 16);
                resources.sortDispatchArguments =
                    builder.ImportBuffer(prefix + "/SortDispatch", culler->SortDispatchBuffer(),
                                         sizeof(particle::GpuParticleCullDispatchState));

                m_renderGraph->SetResourceInitialState(resources.instances, rhi::TextureLayout::Undefined,
                                                       rhi::Access::ShaderWrite, rhi::PipelineStage::ComputeShader,
                                                       rhi::QueueRole::Compute);
                m_renderGraph->SetResourceInitialState(resources.visibility, rhi::TextureLayout::Undefined,
                                                       rhi::Access::ShaderRead, rhi::PipelineStage::ComputeShader,
                                                       rhi::QueueRole::Compute);
                m_renderGraph->SetResourceInitialState(resources.sourceIndirectArguments, rhi::TextureLayout::Undefined,
                                                       rhi::Access::ShaderWrite, rhi::PipelineStage::ComputeShader,
                                                       rhi::QueueRole::Compute);
                m_renderGraph->SetResourceInitialState(resources.sourceRenderIndices, rhi::TextureLayout::Undefined,
                                                       rhi::Access::ShaderWrite, rhi::PipelineStage::ComputeShader,
                                                       rhi::QueueRole::Compute);
                m_renderGraph->SetResourceInitialState(resources.bounds, rhi::TextureLayout::Undefined,
                                                       rhi::Access::ShaderWrite, rhi::PipelineStage::ComputeShader,
                                                       rhi::QueueRole::Compute);
                m_renderGraph->SetResourceInitialState(resources.simulationControl, rhi::TextureLayout::Undefined,
                                                       rhi::Access::ShaderWrite, rhi::PipelineStage::ComputeShader,
                                                       rhi::QueueRole::Compute);
                const auto visibleIndexLastStage = entry.semantics.sortMode == particle::ParticleSortMode::None
                                                       ? rhi::PipelineStage::VertexShader
                                                       : rhi::PipelineStage::ComputeShader;
                const auto visibleIndexLastQueue = entry.semantics.sortMode == particle::ParticleSortMode::None
                                                       ? rhi::QueueRole::Graphics
                                                       : rhi::QueueRole::Compute;
                m_renderGraph->SetResourceInitialState(resources.renderIndices, rhi::TextureLayout::Undefined,
                                                       rhi::Access::ShaderRead, visibleIndexLastStage,
                                                       visibleIndexLastQueue);
                m_renderGraph->SetResourceInitialState(resources.indirectArguments, rhi::TextureLayout::Undefined,
                                                       rhi::Access::IndirectRead, rhi::PipelineStage::DrawIndirect,
                                                       rhi::QueueRole::Graphics);
                const auto sortDispatchLastAccess = entry.semantics.sortMode == particle::ParticleSortMode::None
                                                        ? rhi::Access::ShaderWrite
                                                        : rhi::Access::IndirectRead;
                const auto sortDispatchLastStage = entry.semantics.sortMode == particle::ParticleSortMode::None
                                                       ? rhi::PipelineStage::ComputeShader
                                                       : rhi::PipelineStage::DrawIndirect;
                m_renderGraph->SetResourceInitialState(resources.sortDispatchArguments, rhi::TextureLayout::Undefined,
                                                       sortDispatchLastAccess, sortDispatchLastStage,
                                                       rhi::QueueRole::Compute);
                builder.ReadStorageBuffer(resources.sourceIndirectArguments);
                builder.ReadStorageBuffer(resources.bounds);
                resources.simulationControl =
                    builder.ReadWrite(resources.simulationControl, rhi::PipelineStage::ComputeShader);
                resources.indirectArguments = builder.WriteStorageBuffer(resources.indirectArguments);
                resources.sortDispatchArguments = builder.WriteStorageBuffer(resources.sortDispatchArguments);
                return [this, culler, ownerLayerMask = entry.ownerLayerMask](vk::RenderContext &ctx) {
                    const uint32_t cameraMask = m_cachedCamera ? m_cachedCamera->GetCullingMask() : 0xffffffffu;
                    // Keep each view's indirect output and visibility feedback
                    // consistent. A masked view must not keep offscreen
                    // simulation awake for other cameras.
                    culler->RecordReset(ctx.GetComputeCommandEncoder(), m_particleFrustumPlanes,
                                        (ownerLayerMask & cameraMask) != 0);
                    ctx.RecordComputeDispatch(1, 1, 1, culler->Capacity(), false);
                };
            });
            m_renderGraph->AddComputePass(prefix + "/Cull", [&, culler](vk::PassBuilder &builder) {
                builder.ReadStorageBuffer(resources.visibility);
                builder.ReadStorageBuffer(resources.instances);
                builder.ReadStorageBuffer(resources.sourceIndirectArguments);
                builder.ReadStorageBuffer(resources.sourceRenderIndices);
                builder.ReadIndirectBuffer(resources.sortDispatchArguments);
                builder.ReadStorageBuffer(resources.sortDispatchArguments);
                resources.renderIndices = builder.WriteStorageBuffer(resources.renderIndices);
                resources.indirectArguments =
                    builder.ReadWrite(resources.indirectArguments, rhi::PipelineStage::ComputeShader);
                return [this, culler](vk::RenderContext &ctx) {
                    culler->RecordCull(ctx.GetComputeCommandEncoder(), m_particleFrustumPlanes);
                    ctx.RecordComputeDispatch(0, 0, 0, culler->Capacity(), true);
                };
            });
            m_renderGraph->AddComputePass(prefix + "/Finalize", [&, culler](vk::PassBuilder &builder) {
                resources.indirectArguments =
                    builder.ReadWrite(resources.indirectArguments, rhi::PipelineStage::ComputeShader);
                resources.sortDispatchArguments = builder.WriteStorageBuffer(resources.sortDispatchArguments);
                return [culler](vk::RenderContext &ctx) {
                    culler->RecordFinalize(ctx.GetComputeCommandEncoder());
                    ctx.RecordComputeDispatch(1, 1, 1, culler->Capacity(), false);
                };
            });
            if (resources.instances.IsValid() && resources.visibility.IsValid() && resources.bounds.IsValid() &&
                resources.simulationControl.IsValid() && resources.renderIndices.IsValid() &&
                resources.indirectArguments.IsValid() && resources.sortDispatchArguments.IsValid()) {
                particleGraphResources.emplace(entry.id, resources);
            }
        }

        for (const auto &entry : particleEntries) {
            if (entry.semantics.sortMode == particle::ParticleSortMode::None)
                continue;
            auto resourcesIt = particleGraphResources.find(entry.id);
            if (resourcesIt == particleGraphResources.end())
                continue;
            const auto sorterIt = m_particleSorters.find(entry.id);
            if (sorterIt == m_particleSorters.end() || !sorterIt->second || !sorterIt->second->IsValid())
                continue;

            auto sorter = sorterIt->second;
            auto &resources = resourcesIt->second;
            const std::string prefix = "GpuParticleSort/" + std::to_string(entry.id);
            if (sorter->UsesSmallSort()) {
                m_renderGraph->AddComputePass(prefix + "/Small", [&, sorter, entry, prefix](vk::PassBuilder &builder) {
                    const uint64_t elementBytes = static_cast<uint64_t>(entry.capacity) * sizeof(uint32_t);
                    resources.indices[0] =
                        builder.ImportBuffer(prefix + "/Indices0", sorter->IndexBuffer(0), elementBytes);
                    m_renderGraph->SetResourceInitialState(resources.indices[0], rhi::TextureLayout::Undefined,
                                                           rhi::Access::ShaderRead, rhi::PipelineStage::VertexShader);

                    builder.ReadStorageBuffer(resources.visibility);
                    builder.ReadStorageBuffer(resources.indirectArguments);
                    builder.ReadStorageBuffer(resources.renderIndices);
                    // Finalize publishes the current visible count and this
                    // indirect token. Keep the direct small-sort dispatch
                    // ordered after it just like the radix path.
                    builder.ReadIndirectBuffer(resources.sortDispatchArguments);
                    resources.indices[0] = builder.WriteStorageBuffer(resources.indices[0]);

                    const auto mode = entry.semantics.sortMode;
                    return [this, sorter, mode](vk::RenderContext &ctx) {
                        std::array<float, 16> view{};
                        std::memcpy(view.data(), &m_cachedView[0][0], sizeof(m_cachedView));
                        sorter->RecordSmall(ctx.GetComputeCommandEncoder(), view, mode);
                        ctx.RecordComputeDispatch(1, 1, 1, sorter->Capacity(), false);
                    };
                });
                resources.renderIndices = resources.indices[0];
                resources.drawRenderIndices = sorter->SortedIndices();
                continue;
            }
            m_renderGraph->AddComputePass(prefix + "/Generate", [&, sorter, entry, prefix](vk::PassBuilder &builder) {
                const uint64_t elementBytes = static_cast<uint64_t>(entry.capacity) * sizeof(uint32_t);
                const uint64_t keyBytes =
                    static_cast<uint64_t>(entry.capacity) * particle::ParticleGpuSorter::PackedKeyStride;
                const uint64_t blockBytes =
                    static_cast<uint64_t>(sorter->BlockCount()) * particle::ParticleGpuSorter::Radix * sizeof(uint32_t);
                resources.keys = {
                    builder.ImportBuffer(prefix + "/Keys0", sorter->KeyBuffer(0), keyBytes),
                    builder.ImportBuffer(prefix + "/Keys1", sorter->KeyBuffer(1), keyBytes),
                };
                resources.indices = {
                    builder.ImportBuffer(prefix + "/Indices0", sorter->IndexBuffer(0), elementBytes),
                    builder.ImportBuffer(prefix + "/Indices1", sorter->IndexBuffer(1), elementBytes),
                };
                resources.histogram =
                    builder.ImportBuffer(prefix + "/Histogram", sorter->HistogramBuffer(), blockBytes);
                resources.blockOffsets =
                    builder.ImportBuffer(prefix + "/BlockOffsets", sorter->BlockOffsetBuffer(), blockBytes);
                resources.globalOffsets = builder.ImportBuffer(prefix + "/GlobalOffsets", sorter->GlobalOffsetBuffer(),
                                                               particle::ParticleGpuSorter::Radix * sizeof(uint32_t));

                m_renderGraph->SetResourceInitialState(resources.keys[0], rhi::TextureLayout::Undefined,
                                                       rhi::Access::ShaderWrite, rhi::PipelineStage::ComputeShader);
                m_renderGraph->SetResourceInitialState(resources.keys[1], rhi::TextureLayout::Undefined,
                                                       rhi::Access::ShaderRead, rhi::PipelineStage::ComputeShader);
                m_renderGraph->SetResourceInitialState(resources.indices[0], rhi::TextureLayout::Undefined,
                                                       rhi::Access::ShaderRead, rhi::PipelineStage::VertexShader);
                m_renderGraph->SetResourceInitialState(resources.indices[1], rhi::TextureLayout::Undefined,
                                                       rhi::Access::ShaderRead, rhi::PipelineStage::ComputeShader);
                m_renderGraph->SetResourceInitialState(resources.histogram, rhi::TextureLayout::Undefined,
                                                       rhi::Access::ShaderRead, rhi::PipelineStage::ComputeShader);
                m_renderGraph->SetResourceInitialState(resources.blockOffsets, rhi::TextureLayout::Undefined,
                                                       rhi::Access::ShaderRead, rhi::PipelineStage::ComputeShader);
                m_renderGraph->SetResourceInitialState(resources.globalOffsets, rhi::TextureLayout::Undefined,
                                                       rhi::Access::ShaderRead, rhi::PipelineStage::ComputeShader);

                builder.ReadStorageBuffer(resources.visibility);
                builder.ReadStorageBuffer(resources.indirectArguments);
                builder.ReadStorageBuffer(resources.renderIndices);
                builder.ReadIndirectBuffer(resources.sortDispatchArguments);
                resources.keys[0] = builder.WriteStorageBuffer(resources.keys[0]);
                resources.indices[0] = builder.WriteStorageBuffer(resources.indices[0]);

                const auto mode = entry.semantics.sortMode;
                return [this, sorter, mode](vk::RenderContext &ctx) {
                    std::array<float, 16> view{};
                    std::memcpy(view.data(), &m_cachedView[0][0], sizeof(m_cachedView));
                    sorter->RecordGenerate(ctx.GetComputeCommandEncoder(), view, mode);
                    ctx.RecordComputeDispatch(0, 0, 0, sorter->Capacity(), true);
                };
            });
            for (uint32_t passIndex = 0; passIndex < particle::ParticleGpuSorter::PassCount; ++passIndex) {
                const uint32_t input = passIndex % 2u;
                const uint32_t output = 1u - input;
                const std::string radixPrefix = prefix + "/Radix" + std::to_string(passIndex);
                m_renderGraph->AddComputePass(
                    radixPrefix + "/Histogram", [&, sorter, passIndex, input](vk::PassBuilder &builder) {
                        builder.ReadStorageBuffer(resources.indirectArguments);
                        builder.ReadStorageBuffer(resources.keys[input]);
                        builder.ReadIndirectBuffer(resources.sortDispatchArguments);
                        resources.histogram = builder.WriteStorageBuffer(resources.histogram);
                        return [sorter, passIndex](vk::RenderContext &ctx) {
                            sorter->RecordHistogram(ctx.GetComputeCommandEncoder(), passIndex);
                            ctx.RecordComputeDispatch(0, 0, 0, sorter->Capacity(), true);
                        };
                    });
                m_renderGraph->AddComputePass(radixPrefix + "/Scan", [&, sorter, passIndex](vk::PassBuilder &builder) {
                    builder.ReadStorageBuffer(resources.histogram);
                    builder.ReadStorageBuffer(resources.sortDispatchArguments);
                    resources.blockOffsets = builder.WriteStorageBuffer(resources.blockOffsets);
                    resources.globalOffsets = builder.WriteStorageBuffer(resources.globalOffsets);
                    return [sorter, passIndex](vk::RenderContext &ctx) {
                        sorter->RecordScan(ctx.GetComputeCommandEncoder(), passIndex);
                        ctx.RecordComputeDispatch(1, 1, 1, sorter->BlockCount(), false);
                    };
                });
                m_renderGraph->AddComputePass(
                    radixPrefix + "/Scatter", [&, sorter, passIndex, input, output](vk::PassBuilder &builder) {
                        builder.ReadStorageBuffer(resources.indirectArguments);
                        builder.ReadStorageBuffer(resources.keys[input]);
                        builder.ReadStorageBuffer(resources.indices[input]);
                        builder.ReadStorageBuffer(resources.blockOffsets);
                        builder.ReadStorageBuffer(resources.globalOffsets);
                        builder.ReadIndirectBuffer(resources.sortDispatchArguments);
                        resources.keys[output] = builder.WriteStorageBuffer(resources.keys[output]);
                        resources.indices[output] = builder.WriteStorageBuffer(resources.indices[output]);
                        return [sorter, passIndex](vk::RenderContext &ctx) {
                            sorter->RecordScatter(ctx.GetComputeCommandEncoder(), passIndex);
                            ctx.RecordComputeDispatch(0, 0, 0, sorter->Capacity(), true);
                        };
                    });
            }
            resources.renderIndices = resources.indices[0];
            resources.drawRenderIndices = sorter->SortedIndices();
        }

        auto publishResourceVersion = [&](vk::ResourceHandle handle) {
            if (!handle.IsValid())
                return;
            if (m_importedColorTarget.IsValid() && handle.id == m_importedColorTarget.id) {
                m_importedColorTarget = handle;
            }
            if (m_importedResolveTarget.IsValid() && handle.id == m_importedResolveTarget.id) {
                m_importedResolveTarget = handle;
            }
            for (auto &[name, current] : customRTHandles) {
                if (current.id == handle.id) {
                    current = handle;
                }
            }
            for (auto &[name, current] : bufferHandles) {
                if (current.id == handle.id) {
                    current = handle;
                }
            }
        };

        // Track whether the multisampled backbuffer has been written since the
        // last explicit resolve. FullscreenQuad passes that sample the backbuffer
        // must resolve it again whenever a preceding pass wrote new MSAA data.
        bool backbufferDirtySinceResolve = false;
        uint32_t msaaResolvePassCounter = 0;
        uint32_t depthResolvePassCounter = 0;
        vk::ResourceHandle resolvedParticleSceneDepth;
        vk::ResourceHandle resolvedParticleDepthSource;

        struct ForwardPlusGraphResources
        {
            vk::ResourceHandle canonicalLights;
            vk::ResourceHandle headers;
            vk::ResourceHandle lightMasks;
        };
        std::array<ForwardPlusGraphResources, EngineConfig::kFrameSlotCapacity> forwardPlusResources{};
        std::array<ForwardPlusGraphResources, EngineConfig::kFrameSlotCapacity> particleForwardPlusResources{};
        if (UsesForwardPlus()) {
            for (uint32_t frameIndex = 0; frameIndex < EngineConfig::kFrameSlotCapacity; ++frameIndex) {
                const auto &lights = m_cameraCanonicalLights.Frame(frameIndex);
                const auto &gridFrame = m_forwardPlusGeometryGrid.Frame(frameIndex);
                if (!lights.buffer.IsValid() || !gridFrame.config.IsValid())
                    continue;
                const std::string prefix = "ForwardPlus/Geometry/Frame" + std::to_string(frameIndex);
                m_renderGraph->AddComputePass(prefix + "/Build", [&, frameIndex, prefix](vk::PassBuilder &builder) {
                    auto &resources = forwardPlusResources[frameIndex];
                    resources.canonicalLights =
                        builder.ImportBuffer(prefix + "/CanonicalLights", lights.buffer, lights.capacityBytes);
                    resources.headers =
                        builder.ImportBuffer(prefix + "/TileHeaders", gridFrame.headers, gridFrame.headerCapacityBytes);
                    resources.lightMasks = builder.ImportBuffer(prefix + "/TileLightMasks", gridFrame.lightMasks,
                                                                gridFrame.maskCapacityBytes);
                    m_renderGraph->SetResourceInitialState(resources.canonicalLights, rhi::TextureLayout::Undefined,
                                                           rhi::Access::HostWrite, rhi::PipelineStage::Host);
                    m_renderGraph->SetResourceInitialState(resources.headers, rhi::TextureLayout::Undefined,
                                                           rhi::Access::ShaderRead, rhi::PipelineStage::FragmentShader);
                    m_renderGraph->SetResourceInitialState(resources.lightMasks, rhi::TextureLayout::Undefined,
                                                           rhi::Access::ShaderRead, rhi::PipelineStage::FragmentShader);
                    builder.ReadStorageBuffer(resources.canonicalLights);
                    resources.headers = builder.WriteStorageBuffer(resources.headers);
                    resources.lightMasks = builder.WriteStorageBuffer(resources.lightMasks);
                    return [this, frameIndex](vk::RenderContext &ctx) {
                        const uint32_t activeFrame = m_vkCore->GetCurrentFrameSlot() % EngineConfig::kFrameSlotCapacity;
                        if (activeFrame != frameIndex)
                            return;
                        lighting::ForwardPlusGridConstants constants{};
                        const glm::mat4 viewProjection = m_cachedProj * m_cachedView;
                        std::memcpy(constants.viewProjection, &viewProjection[0][0], sizeof(viewProjection));
                        constants.viewportAndProjectionScale[2] = std::abs(m_cachedProj[0][0]);
                        constants.viewportAndProjectionScale[3] = std::abs(m_cachedProj[1][1]);
                        m_forwardPlusGeometryGrid.Record(frameIndex, ctx.GetComputeCommandEncoder(), constants);
                    };
                });
            }
            if (m_forwardPlusParticlesRequired) {
                for (uint32_t frameIndex = 0; frameIndex < EngineConfig::kFrameSlotCapacity; ++frameIndex) {
                    const auto &lights = m_cameraCanonicalLights.Frame(frameIndex);
                    const auto &gridFrame = m_forwardPlusParticleGrid.Frame(frameIndex);
                    if (!lights.buffer.IsValid() || !gridFrame.config.IsValid())
                        continue;
                    const std::string prefix = "ForwardPlus/Particles/Frame" + std::to_string(frameIndex);
                    m_renderGraph->AddComputePass(prefix + "/Build", [&, frameIndex, prefix](vk::PassBuilder &builder) {
                        auto &resources = particleForwardPlusResources[frameIndex];
                        resources.canonicalLights =
                            builder.ImportBuffer(prefix + "/CanonicalLights", lights.buffer, lights.capacityBytes);
                        resources.headers = builder.ImportBuffer(prefix + "/TileHeaders", gridFrame.headers,
                                                                 gridFrame.headerCapacityBytes);
                        resources.lightMasks = builder.ImportBuffer(prefix + "/TileLightMasks", gridFrame.lightMasks,
                                                                    gridFrame.maskCapacityBytes);
                        m_renderGraph->SetResourceInitialState(resources.canonicalLights, rhi::TextureLayout::Undefined,
                                                               rhi::Access::HostWrite, rhi::PipelineStage::Host);
                        m_renderGraph->SetResourceInitialState(resources.headers, rhi::TextureLayout::Undefined,
                                                               rhi::Access::ShaderRead,
                                                               rhi::PipelineStage::FragmentShader);
                        m_renderGraph->SetResourceInitialState(resources.lightMasks, rhi::TextureLayout::Undefined,
                                                               rhi::Access::ShaderRead,
                                                               rhi::PipelineStage::FragmentShader);
                        builder.ReadStorageBuffer(resources.canonicalLights);
                        resources.headers = builder.WriteStorageBuffer(resources.headers);
                        resources.lightMasks = builder.WriteStorageBuffer(resources.lightMasks);
                        return [this, frameIndex](vk::RenderContext &ctx) {
                            const uint32_t activeFrame =
                                m_vkCore->GetCurrentFrameSlot() % EngineConfig::kFrameSlotCapacity;
                            if (activeFrame != frameIndex)
                                return;
                            lighting::ForwardPlusGridConstants constants{};
                            const glm::mat4 viewProjection = m_cachedProj * m_cachedView;
                            std::memcpy(constants.viewProjection, &viewProjection[0][0], sizeof(viewProjection));
                            constants.viewportAndProjectionScale[2] = std::abs(m_cachedProj[0][0]);
                            constants.viewportAndProjectionScale[3] = std::abs(m_cachedProj[1][1]);
                            m_forwardPlusParticleGrid.Record(frameIndex, ctx.GetComputeCommandEncoder(), constants);
                        };
                    });
                }
            }
        }

        // Fullscreen lighting shaders consume the camera-local shadow map
        // through the per-view descriptor set rather than their explicit
        // set-0 texture list. Discover that graph resource up front so those
        // passes still participate in dependency and layout tracking.
        std::string perViewShadowTextureName;
        for (const auto &candidatePass : sortedPasses) {
            const GraphCommandDesc *candidateCommand = candidatePass.PrimaryCommand();
            if (!candidateCommand)
                continue;
            const auto shadowBinding =
                std::find_if(candidateCommand->inputBindings.begin(), candidateCommand->inputBindings.end(),
                             [](const auto &binding) { return binding.first == "shadowMap"; });
            if (shadowBinding != candidateCommand->inputBindings.end()) {
                perViewShadowTextureName = shadowBinding->second;
                break;
            }
        }
        bool fullscreenShadowDependencyDeclared = false;
        // Camera clear settings remain active across frames. Consumption is
        // local to this graph build so only the first color-clearing pass is
        // overridden without disabling later per-frame color updates.
        const auto cameraClearOverride = ResolveCameraClearOverride();
        m_mainClearPassName.clear();

        // Scene overlays are authored in linear space. Composite them before
        // the final display encode so icons, handles, outlines and the grid all
        // share the same color contract as scene geometry. Player/Game graphs
        // never receive these editor-only passes.
        bool sceneOverlaysAppended = false;
        const auto appendSceneOverlays = [&]() {
            if (sceneOverlaysAppended || m_renderView.kind != rhi::RenderViewKind::Scene)
                return;
            m_importedColorTarget =
                AppendAutoPass("_ComponentGizmos", m_importedColorTarget, sharedDepth, width, height);
            m_importedColorTarget = AppendAutoPass("_EditorGizmos", m_importedColorTarget, sharedDepth, width, height);
            m_importedColorTarget = AppendEditorOutline(m_importedColorTarget);
            m_importedColorTarget = AppendAutoPass("_EditorTools", m_importedColorTarget, sharedDepth, width, height);
            sceneOverlaysAppended = true;
        };

        const bool trackInputs = std::any_of(m_pythonGraphDesc.buffers.begin(), m_pythonGraphDesc.buffers.end(),
                                             [](const auto &buffer) { return buffer.input != nullptr; });
        const bool trackPersistentTextures =
            std::any_of(m_pythonGraphDesc.textures.begin(), m_pythonGraphDesc.textures.end(),
                        [](const auto &texture) { return texture.role == GraphTextureRole::Persistent; });
        for (size_t passIndex = 0; passIndex < sortedPasses.size(); ++passIndex) {
            const auto &passDesc = sortedPasses[passIndex];
            const auto trackAuthorPass = [&](vk::PassHandle pass, std::optional<size_t> dispatch = std::nullopt) {
                if (trackInputs || trackPersistentTextures || !passDesc.events.empty())
                    m_authorPassIndices.emplace(pass.id, AuthorPass{passIndex, dispatch});
            };
            const GraphCommandDesc *command = passDesc.PrimaryCommand();
            if (m_viewportGeneration && m_screenUIOutput && !m_viewportColorPublished && command &&
                command->type == GraphCommandType::DrawScreenUI && command->screenUIList != 0) {
                // Finish the camera's display-encoded rectangle first. Overlay
                // coordinates and subsequent display-space effects belong to
                // the full output, including pixels outside this camera.
                m_importedColorTarget = CopyViewportRegion("__ViewportComposeBeforeOverlay", m_importedColorTarget,
                                                           m_viewportOutputRaster, rhi::TextureAspect::Color, false);
                m_importedDepthTarget = CopyViewportRegion("__ViewportDepthBeforeOverlay", m_importedDepthTarget,
                                                           m_viewportOutputDepth, rhi::TextureAspect::Depth, false);
                sharedDepth = m_importedDepthTarget;
                for (const auto &texture : m_pythonGraphDesc.textures) {
                    if (texture.IsViewDepth())
                        customRTHandles[texture.name] = sharedDepth;
                }
                m_importedResolveTarget = m_sceneTarget->IsMsaaEnabled() ? m_viewportOutputColor : vk::ResourceHandle{};
                m_renderGraph->SetBackbuffer(m_importedColorTarget);
                m_viewportColorPublished = true;
                width = m_viewport.Output()->description.width;
                height = m_viewport.Output()->description.height;
                backbufferDirtySinceResolve = true;
            }
            if (command && command->type == GraphCommandType::FullscreenQuad &&
                command->shaderName == "infernux/internal/display_encode") {
                appendSceneOverlays();
            }
            const auto isPersistent = [&](const std::string &name) {
                const auto texture = texDescMap.find(name);
                return texture != texDescMap.end() && texture->second->role == GraphTextureRole::Persistent;
            };
            const bool writesPersistent =
                isPersistent(passDesc.writeDepth) || isPersistent(passDesc.resolveColor) ||
                std::any_of(passDesc.writeColors.begin(), passDesc.writeColors.end(),
                            [&](const auto &output) { return isPersistent(output.second); }) ||
                (command && command->CopiesToTexture() && isPersistent(command->destinationResource));
            static const std::vector<std::pair<std::string, std::string>> kNoInputBindings;
            const auto &commandInputBindings = command ? command->inputBindings : kNoInputBindings;
            // Look up render callback from the Python callbacks map
            auto callbackIt = m_pythonCallbacks.find(passDesc.name);
            if (callbackIt == m_pythonCallbacks.end()) {
                INXLOG_WARN("SceneRenderGraph: Pass '", passDesc.name,
                            "' has no render callback — skipping. "
                            "This usually means ApplyPythonGraph() was not called or validation failed.");
                continue;
            }
            auto callback = callbackIt->second;
            std::vector<particle::GpuParticleDrawEntry> particleEntries;
            MaterialPassPipelineDescriptor particlePass;
            const bool particleSurfacePass = command && !command->rendererSelection &&
                                             command->type == GraphCommandType::DrawRenderers &&
                                             (command->shaderTarget == ShaderCompileTarget::Forward ||
                                              command->shaderTarget == ShaderCompileTarget::ForwardPlus ||
                                              command->shaderTarget == ShaderCompileTarget::Motion);
            const bool particleShadowPass = command && command->type == GraphCommandType::DrawShadowCasters &&
                                            command->shaderTarget == ShaderCompileTarget::Shadow;
            if (m_particleDrawRegistry && (particleSurfacePass || particleShadowPass)) {
                const auto particlePassIt = m_pythonMaterialPasses.find(passDesc.name);
                if (particlePassIt != m_pythonMaterialPasses.end()) {
                    particlePass = particlePassIt->second;
                    const auto snapshot = m_particleDrawRegistry->SnapshotShared(command->queueMin, command->queueMax);
                    particleEntries.reserve(snapshot->size());
                    std::copy_if(snapshot->begin(), snapshot->end(), std::back_inserter(particleEntries),
                                 [&](const auto &entry) {
                                     if (particleShadowPass)
                                         return entry.semantics.castShadows && entry.renderer->CanCastShadows();
                                     return entry.semantics.sortMode == particle::ParticleSortMode::None ||
                                            particleGraphResources.find(entry.id) != particleGraphResources.end();
                                 });
                }
            }

            auto resolveTextureHandle = [&](const std::string &name) -> vk::ResourceHandle {
                const auto texture = texDescMap.find(name);
                if (texture == texDescMap.end())
                    return {};
                if (texture->second->IsViewColor())
                    return m_importedColorTarget;
                const auto custom = customRTHandles.find(name);
                if (custom != customRTHandles.end())
                    return custom->second;
                return {};
            };
            auto textureExtent = [&](const std::string &name) -> VkExtent3D {
                const auto texture = texDescMap.find(name);
                if (texture == texDescMap.end())
                    return {0, 0, 1};
                const auto [resourceWidth, resourceHeight] = texture->second->ResolveExtent(width, height);
                return {resourceWidth, resourceHeight, 1};
            };

            if (passDesc.type == GraphPassType::Copy && command) {
                if (command->type == GraphCommandType::CopyBuffer) {
                    auto source = bufferHandles.find(command->sourceResource);
                    auto destination = bufferHandles.find(command->destinationResource);
                    if (source == bufferHandles.end() || destination == bufferHandles.end())
                        continue;
                    const auto sourceDesc = std::find_if(
                        m_pythonGraphDesc.buffers.begin(), m_pythonGraphDesc.buffers.end(),
                        [&](const GraphBufferDesc &buffer) { return buffer.name == command->sourceResource; });
                    const auto destinationDesc = std::find_if(
                        m_pythonGraphDesc.buffers.begin(), m_pythonGraphDesc.buffers.end(),
                        [&](const GraphBufferDesc &buffer) { return buffer.name == command->destinationResource; });
                    const VkDeviceSize copyBytes = command->copyBytes > 0
                                                       ? command->copyBytes
                                                       : std::min(sourceDesc->byteSize, destinationDesc->byteSize);
                    vk::ResourceHandle written;
                    const auto sourceHandle = source->second;
                    const auto destinationHandle = destination->second;
                    const auto sourceOffset = sourceDesc->byteOffset;
                    const auto destinationOffset = destinationDesc->byteOffset;
                    const bool partialCopy = destinationOffset != 0 || copyBytes != destinationDesc->byteSize;
                    const auto pass = m_renderGraph->AddTransferPass(passDesc.name, [&](vk::PassBuilder &builder) {
                        // Public scene-graph copies sit between dependent raster
                        // passes, like MSAA resolves; keep them on Graphics instead
                        // of adding two ownership handoffs with no async work.
                        builder.SetQueueRole(rhi::QueueRole::Graphics);
                        builder.TransferRead(sourceHandle);
                        written = builder.TransferWrite(destinationHandle, partialCopy);
                        builder.SetSideEffect(passDesc.sideEffect);
                        return [sourceHandle, written, copyBytes, sourceOffset,
                                destinationOffset](vk::RenderContext &ctx) {
                            ctx.GetTransferCommandEncoder().CopyBuffer(ctx.GetBufferHandle(sourceHandle),
                                                                       ctx.GetBufferHandle(written),
                                                                       {sourceOffset, destinationOffset, copyBytes});
                        };
                    });
                    trackAuthorPass(pass);
                    publishResourceVersion(written);
                    continue;
                }

                if (command->type == GraphCommandType::CopyTextureToBuffer ||
                    command->type == GraphCommandType::CopyBufferToTexture) {
                    const bool toBuffer = command->type == GraphCommandType::CopyTextureToBuffer;
                    const auto &textureName = toBuffer ? command->sourceResource : command->destinationResource;
                    const auto &bufferName = toBuffer ? command->destinationResource : command->sourceResource;
                    const auto texture = resolveTextureHandle(textureName);
                    const auto buffer = bufferHandles.at(bufferName);
                    const auto bufferDesc =
                        std::find_if(m_pythonGraphDesc.buffers.begin(), m_pythonGraphDesc.buffers.end(),
                                     [&](const auto &desc) { return desc.name == bufferName; });
                    const auto &image = *texDescMap.at(textureName);
                    const auto format = image.EffectiveFormat(m_renderView);
                    const auto extent = textureExtent(textureName);
                    const rhi::TextureBufferCopyRegion region{rhi::TextureBufferCopyRegion::AspectFor(format),
                                                              0,
                                                              0,
                                                              toBuffer ? command->copyWidth : extent.width,
                                                              toBuffer ? command->copyHeight : extent.height,
                                                              bufferDesc->byteOffset + command->copyBufferOffset};
                    const bool partialBufferWrite =
                        toBuffer && (command->copyBufferOffset != 0 ||
                                     rhi::TextureBufferCopyRegion::ByteSize(format, region.width, region.height) !=
                                         bufferDesc->byteSize);
                    const auto source = toBuffer ? texture : buffer;
                    const auto destination = toBuffer ? buffer : texture;
                    vk::ResourceHandle written;
                    const auto pass = m_renderGraph->AddTransferPass(passDesc.name, [&](vk::PassBuilder &builder) {
                        builder.SetQueueRole(rhi::QueueRole::Graphics);
                        builder.TransferRead(source);
                        written = builder.TransferWrite(destination, partialBufferWrite);
                        builder.SetSideEffect(passDesc.sideEffect || writesPersistent);
                        return [source, written, region, toBuffer](vk::RenderContext &ctx) {
                            auto encoder = ctx.GetTransferCommandEncoder();
                            const bool copied =
                                toBuffer ? encoder.CopyTextureToBuffer(ctx.GetTextureHandle(source),
                                                                       ctx.GetBufferHandle(written), region)
                                         : encoder.CopyBufferToTexture(ctx.GetBufferHandle(source),
                                                                       ctx.GetTextureHandle(written), region);
                            if (!copied)
                                throw std::runtime_error("Prepared graph texture-buffer transfer could not record");
                        };
                    });
                    trackAuthorPass(pass);
                    publishResourceVersion(written);
                    continue;
                }

                if (command->type == GraphCommandType::CopyTexture) {
                    const auto sourceHandle = resolveTextureHandle(command->sourceResource);
                    const auto destinationHandle = resolveTextureHandle(command->destinationResource);
                    if (!sourceHandle.IsValid() || !destinationHandle.IsValid())
                        continue;
                    const auto sourceDesc = texDescMap.at(command->sourceResource);
                    const VkExtent3D sourceExtent = textureExtent(command->sourceResource);
                    const VkExtent3D destinationExtent = textureExtent(command->destinationResource);
                    const VkExtent3D copyExtent{std::min(sourceExtent.width, destinationExtent.width),
                                                std::min(sourceExtent.height, destinationExtent.height), 1};
                    const bool partialCopy =
                        copyExtent.width < destinationExtent.width || copyExtent.height < destinationExtent.height;
                    const rhi::TextureAspect aspect =
                        sourceDesc->IsDepth() ? rhi::TextureAspect::Depth : rhi::TextureAspect::Color;
                    vk::ResourceHandle written;
                    const auto pass = m_renderGraph->AddTransferPass(passDesc.name, [&](vk::PassBuilder &builder) {
                        builder.SetQueueRole(rhi::QueueRole::Graphics);
                        builder.TransferRead(sourceHandle);
                        written = builder.TransferWrite(destinationHandle, partialCopy);
                        builder.SetSideEffect(passDesc.sideEffect || writesPersistent);
                        return [sourceHandle, written, copyExtent, aspect](vk::RenderContext &ctx) {
                            ctx.GetTransferCommandEncoder().CopyTexture(
                                ctx.GetTextureHandle(sourceHandle), ctx.GetTextureHandle(written),
                                {aspect, 0, 0, 0, 0, copyExtent.width, copyExtent.height, copyExtent.depth});
                        };
                    });
                    trackAuthorPass(pass);
                    publishResourceVersion(written);
                    continue;
                }
            }

            if (passDesc.type == GraphPassType::Compute && command) {
                const auto instance = m_workInstances.at(command->workBinding);
                const auto &recipe = instance->GetRecipe();
                std::vector<std::string> slotNames(recipe.GetSlots().size());
                for (const auto &buffer : m_pythonGraphDesc.buffers) {
                    if (buffer.workBinding == command->workBinding && buffer.byteOffset == 0 &&
                        buffer.byteSize == recipe.GetSlots()[buffer.workSlot].allocation.GetByteSize())
                        slotNames[buffer.workSlot] = buffer.name;
                }
                for (size_t index = 0; index < instance->GetDispatches().size(); ++index) {
                    const auto &work = instance->GetDispatches()[index];
                    const auto &operation = recipe.GetDispatches()[index];
                    struct Access
                    {
                        vk::ResourceHandle handle;
                        bool uniformRead = false;
                        bool storageRead = false;
                        bool write = false;
                    };
                    std::vector<Access> accesses;
                    for (size_t binding = 0; binding < operation.bindings.size(); ++binding) {
                        const auto &item = operation.bindings[binding];
                        const auto handle = bufferHandles.at(slotNames[item.range.slot]);
                        auto found = std::find_if(accesses.begin(), accesses.end(),
                                                  [&](const auto &access) { return access.handle.id == handle.id; });
                        if (found == accesses.end()) {
                            accesses.push_back({handle});
                            found = std::prev(accesses.end());
                        }
                        const auto bits = static_cast<uint8_t>(item.access);
                        if (bits & static_cast<uint8_t>(rhi::ComputeBufferAccess::Read)) {
                            if (work.GetBufferBindingTypes()[binding] == rhi::BindingType::UniformBuffer)
                                found->uniformRead = true;
                            else
                                found->storageRead = true;
                        }
                        found->write |= (bits & static_cast<uint8_t>(rhi::ComputeBufferAccess::Write)) != 0;
                    }
                    std::vector<vk::ResourceHandle> written;
                    const auto pass = m_renderGraph->AddComputePass(
                        passDesc.name + "/" + std::to_string(index), [&](vk::PassBuilder &builder) {
                            builder.SetQueueRole(rhi::QueueRole::Graphics);
                            for (const auto &access : accesses) {
                                if (access.uniformRead)
                                    builder.ReadUniformBuffer(access.handle);
                                if (access.storageRead)
                                    builder.ReadStorageBuffer(access.handle);
                                if (access.write)
                                    written.push_back(builder.WriteStorageBuffer(access.handle));
                            }
                            if (operation.indirectArguments)
                                builder.ReadIndirectBuffer(
                                    bufferHandles.at(slotNames[operation.indirectArguments->slot]));
                            builder.SetSideEffect(passDesc.sideEffect);
                            return [this, workName = command->workBinding, index](vk::RenderContext &ctx) {
                                const auto &current = *m_workInstances.at(workName);
                                if (!current.GetDispatches()[index].RecordOne(ctx.GetComputeCommandEncoder()))
                                    throw std::runtime_error("Prepared common graph work could not record");
                                const auto &dispatch = current.GetRecipe().GetDispatches()[index];
                                if (dispatch.indirectArguments)
                                    ctx.RecordComputeDispatch(0, 0, 0, 0, true);
                                else
                                    ctx.RecordComputeDispatch(dispatch.groupCountX, dispatch.groupCountY,
                                                              dispatch.groupCountZ, 0, false);
                            };
                        });
                    trackAuthorPass(pass, index);
                    for (const auto handle : written)
                        publishResourceVersion(handle);
                }
                continue;
            }

            if (passDesc.type == GraphPassType::Present && command) {
                const auto source = resolveTextureHandle(command->sourceResource);
                if (!source.IsValid())
                    continue;
                const auto pass = m_renderGraph->AddPresentPass(passDesc.name, [&](vk::PassBuilder &builder) {
                    builder.Read(source, rhi::PipelineStage::FragmentShader);
                    builder.SetSideEffect();
                    return [](vk::RenderContext &) {};
                });
                trackAuthorPass(pass);
                continue;
            }

            // Determine color targets (MRT support).
            // Build a map of slot → ResourceHandle for all declared color outputs.
            // Slot 0 defaults to the MSAA backbuffer if not specified.
            std::map<int, vk::ResourceHandle> colorTargets;
            for (const auto &[slot, texName] : passDesc.writeColors) {
                if (texName.empty()) {
                    continue;
                }
                auto texIt = texDescMap.find(texName);
                if (texIt != texDescMap.end() && texIt->second->IsViewColor()) {
                    colorTargets[slot] = m_importedColorTarget;
                } else {
                    // Non-backbuffer texture: look up pre-registered transient handle
                    auto rtIt = customRTHandles.find(texName);
                    if (rtIt != customRTHandles.end()) {
                        colorTargets[slot] = rtIt->second;
                    }
                }
            }
            // Default: if no color outputs declared and not a depth-only pass,
            // write to MSAA backbuffer at slot 0.
            // Shadow and semantic Depth passes have no color attachments.
            bool isShadowPassAction = command && command->type == GraphCommandType::DrawShadowCasters;
            const bool isDepthOnlyMaterialPass = command && command->type == GraphCommandType::DrawRenderers &&
                                                 (command->shaderTarget == ShaderCompileTarget::Depth ||
                                                  command->shaderTarget == ShaderCompileTarget::Shadow);
            if (colorTargets.empty() && passDesc.writeDepth.empty() && !isShadowPassAction &&
                !isDepthOnlyMaterialPass) {
                colorTargets[0] = m_importedColorTarget;
            }
            // Primary color target (slot 0) — used for MSAA resolve and compute fallback
            vk::ResourceHandle primaryColorTarget = colorTargets.count(0) ? colorTargets[0] : vk::ResourceHandle{};
            const bool writesBackbuffer = primaryColorTarget.IsValid() && m_importedColorTarget.IsValid() &&
                                          primaryColorTarget.id == m_importedColorTarget.id;

            // Collect non-depth read texture handles for builder.Read()
            // This creates proper DAG edges and Vulkan barriers for
            // color texture dependencies between passes.
            std::vector<vk::ResourceHandle> colorReadHandles;
            bool readsDepth = false;
            vk::ResourceHandle declaredReadOnlyDepth;
            for (const auto &readTex : passDesc.readTextures) {
                auto texIt = texDescMap.find(readTex);
                if (texIt != texDescMap.end()) {
                    if (texIt->second->IsDepth()) {
                        const bool sampledInput = command && command->BindsResource(readTex);
                        if (!sampledInput) {
                            readsDepth = true;
                            const auto depthIt = customRTHandles.find(readTex);
                            if (depthIt != customRTHandles.end())
                                declaredReadOnlyDepth = depthIt->second;
                        }
                    } else if (!texIt->second->IsViewColor()) {
                        // Non-depth, non-backbuffer read: look up custom RT handle
                        auto rtIt = customRTHandles.find(readTex);
                        if (rtIt != customRTHandles.end()) {
                            colorReadHandles.push_back(rtIt->second);
                        }
                    }
                }
            }

            // Build input binding handles: map sampler name → ResourceHandle
            // so the execute lambda can resolve VkImageViews at runtime.
            struct InputBindingHandle
            {
                std::string samplerName;
                vk::ResourceHandle handle;
                bool isDepth = false;
            };
            std::vector<InputBindingHandle> inputBindingHandles;
            for (const auto &[samplerName, textureName] : commandInputBindings) {
                if (command && command->type == GraphCommandType::FullscreenQuad &&
                    bufferHandles.find(textureName) != bufferHandles.end())
                    continue;
                auto texIt = texDescMap.find(textureName);
                if (texIt != texDescMap.end() && texIt->second->IsViewColor()) {
                    // Backbuffer texture — use the imported color target
                    inputBindingHandles.push_back({samplerName, m_importedColorTarget, false});
                } else {
                    auto rtIt = customRTHandles.find(textureName);
                    if (rtIt != customRTHandles.end()) {
                        bool isDepthInput = texIt != texDescMap.end() && texIt->second->IsDepth();
                        inputBindingHandles.push_back({samplerName, rtIt->second, isDepthInput});
                    } else {
                        INXLOG_WARN("SceneRenderGraph: Input binding '", samplerName, "' references unknown texture '",
                                    textureName, "'");
                    }
                }
            }

            // Determine depth relationship
            bool writesDepth = !passDesc.writeDepth.empty();

            // Read clear values from the Python graph description, but
            // allow camera ClearFlags to override the first color-clearing pass.
            bool clearColor = passDesc.clearColor;
            bool clearDepth = passDesc.clearDepth;
            float clearColorR = passDesc.clearColorR;
            float clearColorG = passDesc.clearColorG;
            float clearColorB = passDesc.clearColorB;
            float clearColorA = passDesc.clearColorA;
            float clearDepthVal = passDesc.clearDepthValue;

            // Persistent outputs have their own producer clear contract. Camera
            // background/accumulation settings belong to its view, not those outputs.
            if (cameraClearOverride && cameraClearOverride->name == passDesc.name) {
                clearColor = cameraClearOverride->clearColor;
                clearDepth = cameraClearOverride->clearDepth;
                clearColorR = cameraClearOverride->clearColorR;
                clearColorG = cameraClearOverride->clearColorG;
                clearColorB = cameraClearOverride->clearColorB;
                clearColorA = cameraClearOverride->clearColorA;
                // Record the pass name for per-frame clear-value updates.
                // Execute() uses this to update clear values without
                // rebuilding the graph.
                m_mainClearPassName = passDesc.name;
            }

            // Capture depth state for the lambda (by value — sharedDepth
            // is updated between iterations so we capture the CURRENT value)
            vk::ResourceHandle depthForThisPass = declaredReadOnlyDepth.IsValid() ? declaredReadOnlyDepth : sharedDepth;
            bool needsCreateDepth = writesDepth && !sharedDepth.IsValid();
            bool passReadsDepth = readsDepth && !writesDepth;
            vk::ResourceHandle particleSceneDepth;
            bool particleSceneDepthIsDepth = true;
            const bool hasSoftParticleOutputs =
                std::any_of(particleEntries.begin(), particleEntries.end(),
                            [](const auto &entry) { return entry.semantics.softParticles; });
            if (hasSoftParticleOutputs) {
                const bool hasDepthContract = passReadsDepth && depthForThisPass.IsValid() &&
                                              particlePass.depthFormat != rhi::PixelFormat::Undefined;
                if (hasDepthContract && msaaSamples == VK_SAMPLE_COUNT_1_BIT) {
                    particleSceneDepth = depthForThisPass;
                } else if (hasDepthContract && m_sceneDepthResolver.IsValid()) {
                    if (!resolvedParticleSceneDepth.IsValid() || resolvedParticleDepthSource != depthForThisPass) {
                        const auto sourceDepth = depthForThisPass;
                        const std::string resolvePassName =
                            "_SceneDepthResolve/" + std::to_string(depthResolvePassCounter++);
                        m_renderGraph->AddComputePass(resolvePassName, [&, sourceDepth](vk::PassBuilder &builder) {
                            // Immediate raster dependency; no asynchronous overlap to gain.
                            builder.SetQueueRole(rhi::QueueRole::Graphics);
                            builder.ReadSampledDepth(sourceDepth, rhi::PipelineStage::ComputeShader);
                            auto target = builder.CreateTexture("SceneDepthResolved", width, height,
                                                                VK_FORMAT_R32_SFLOAT, VK_SAMPLE_COUNT_1_BIT);
                            resolvedParticleSceneDepth = builder.WriteStorageTexture(target);
                            const auto outputDepth = resolvedParticleSceneDepth;
                            return [this, sourceDepth, outputDepth, width, height,
                                    sampleCount = static_cast<uint32_t>(msaaSamples)](vk::RenderContext &ctx) {
                                m_sceneDepthResolver.Record(
                                    ctx.GetComputeCommandEncoder(), ctx.GetTextureView(sourceDepth),
                                    ctx.GetTextureView(outputDepth), width, height, sampleCount);
                            };
                        });
                        resolvedParticleDepthSource = sourceDepth;
                    }
                    particleSceneDepth = resolvedParticleSceneDepth;
                    particleSceneDepthIsDepth = false;
                } else {
                    if (hasDepthContract && msaaSamples != VK_SAMPLE_COUNT_1_BIT && !m_sceneDepthResolver.IsValid()) {
                        INXLOG_ERROR("SceneRenderGraph: soft particle depth resolve is unavailable for pass '",
                                     passDesc.name, "'");
                    }
                    // A queue range may deliberately overlap multiple surface
                    // passes. Soft particles can only run after opaque depth is
                    // complete, so an earlier pass is an ineligible route, not
                    // a graph failure. Leave the entries for a later pass whose
                    // contract reads the scene depth.
                    particleEntries.erase(
                        std::remove_if(particleEntries.begin(), particleEntries.end(),
                                       [](const auto &entry) { return entry.semantics.softParticles; }),
                        particleEntries.end());
                }
            }

            vk::ResourceHandle resolveTarget;
            if (!passDesc.resolveColor.empty()) {
                const auto resolveIt = customRTHandles.find(passDesc.resolveColor);
                if (resolveIt != customRTHandles.end())
                    resolveTarget = resolveIt->second;
            }

            // =================================================================
            // FullscreenQuad passes: fullscreen triangle with named shader,
            // push constants, and input texture sampling.
            // Uses FullscreenRenderer to manage pipeline cache + draw.
            //
            // MSAA handling:
            //   - Reading the MSAA backbuffer: a multisample image cannot be
            //     sampled by a regular sampler2D.  When MSAA is active, an
            //     automatic transfer pass resolves the backbuffer to the 1x
            //     resolve target before the first FullscreenQuad that reads
            //     it.  Subsequent reads reference the 1x resolve target.
            //   - Writing to the MSAA backbuffer: the pipeline sample count
            //     must match the render pass attachment. We propagate the
            //     actual MSAA sample count into the FullscreenPipelineKey.
            // =================================================================
            if (command && command->type == GraphCommandType::FullscreenQuad) {

                // Reflect once when building this graph, never in the frame loop.
                // Texture2DMS retains sample identity; Texture2D consumes resolved data.
                const auto authorProgram = m_fullscreenRenderer.AcquireProgram(command->shaderName);
                if (authorProgram)
                    m_authorShaderGuids.insert(authorProgram->assetGuid);
                FullscreenParameters authorParameters;
                if (authorProgram) {
                    try {
                        authorParameters = FullscreenRenderer::PrepareParameters(*authorProgram, command->parameters);
                        ValidateFullscreenHistories(*authorProgram, authorParameters, *command, passDesc,
                                                    m_pythonGraphDesc);
                    } catch (const std::invalid_argument &error) {
                        INXLOG_ERROR("SceneRenderGraph: fullscreen program rejected: ", error.what());
                        m_rejectedFullscreenSourceRevision = m_pythonGraphSourceRevision;
                        m_needsRebuild = false;
                        m_needsCompile = false;
                        return;
                    }
                }
                struct FullscreenInput
                {
                    std::string name;
                    uint32_t binding = 0;
                    bool storage = false;
                    bool readOnly = true;
                    bool multisampled = false;
                    bool history = false;
                    uint32_t arraySize = 1;
                    ReflectedImageDimension dimension = ReflectedImageDimension::D2;
                    VkFormat scalarArrayFormat = VK_FORMAT_UNDEFINED;
                    uint32_t arrayMemberOffset = 0;
                    uint32_t arrayStride = 0;
                    uint32_t arrayElementCount = 0;
                };
                std::vector<FullscreenInput> declaredInputs;
                if (authorProgram) {
                    const auto schema = nlohmann::json::parse(authorProgram->schema);
                    for (const auto &input : schema.at("inputs")) {
                        const auto name = input.at("name").get<std::string>();
                        const auto &resources = authorProgram->code->resources;
                        const auto resource = std::find_if(resources.begin(), resources.end(), [&](const auto &item) {
                            return item.name == "inx_" + name + ".texture";
                        });
                        if (resource == resources.end())
                            throw std::logic_error("Fullscreen author input has no target resource: " + name);
                        FullscreenInput item;
                        item.name = name;
                        item.history = input.at("kind") == "History2D";
                        item.binding = static_cast<uint32_t>(declaredInputs.size());
                        item.multisampled = resource->binding.multisampled;
                        declaredInputs.push_back(std::move(item));
                    }
                } else {
                    if (!m_vkCore->EnsureShaderAvailable(command->shaderName, "fragment"))
                        return;
                    const auto *fragmentCode = m_vkCore->GetShaderCache().FindFragCode(command->shaderName);
                    ShaderReflection reflection;
                    if (!fragmentCode || !reflection.Reflect(*fragmentCode, VK_SHADER_STAGE_FRAGMENT_BIT))
                        return;
                    for (const auto &image : reflection.GetSampledImages()) {
                        if (image.set == 0 && image.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                            FullscreenInput item;
                            item.name = image.name;
                            item.binding = image.binding;
                            item.multisampled = image.multisampled;
                            item.arraySize = image.arraySize;
                            item.dimension = image.dimension;
                            declaredInputs.push_back(std::move(item));
                        }
                    }
                    for (const auto &buffer : reflection.GetStorageBuffers()) {
                        if (buffer.set == 0) {
                            FullscreenInput item;
                            item.name = buffer.name;
                            item.binding = buffer.binding;
                            item.storage = true;
                            item.readOnly = buffer.readOnly;
                            item.arraySize = buffer.arraySize;
                            item.scalarArrayFormat = buffer.scalarArrayFormat;
                            item.arrayMemberOffset = buffer.arrayMemberOffset;
                            item.arrayStride = buffer.arrayStride;
                            item.arrayElementCount = buffer.arrayElementCount;
                            declaredInputs.push_back(std::move(item));
                        }
                    }
                }
                std::unordered_set<uint32_t> multisampledBindings;
                std::unordered_map<uint32_t, ReflectedImageDimension> sampledDimensions;
                for (const auto &image : declaredInputs) {
                    if (image.storage)
                        continue;
                    sampledDimensions[image.binding] = image.dimension;
                    if (image.multisampled)
                        multisampledBindings.insert(image.binding);
                }
                std::vector<std::string> inputNames(commandInputBindings.size());
                const auto declaredInputCount = declaredInputs.size();
                if (!commandInputBindings.empty() && commandInputBindings.size() != declaredInputCount) {
                    INXLOG_ERROR("Fullscreen pass '", passDesc.name,
                                 "' must bind every declared set-0 input exactly once");
                    return;
                }
                for (const auto &[resourceName, sourceName] : commandInputBindings) {
                    std::optional<uint32_t> binding;
                    const bool bufferSource = bufferHandles.find(sourceName) != bufferHandles.end();
                    for (const auto &input : declaredInputs)
                        if (bufferSource == input.storage && input.name == resourceName && input.arraySize == 1)
                            binding = input.binding;
                    if (!binding) {
                        INXLOG_ERROR("Fullscreen pass '", passDesc.name, "' has no matching scalar shader input '",
                                     resourceName, "' for resource '", sourceName, "' in set 0");
                        return;
                    }
                    if (*binding >= inputNames.size() || !inputNames[*binding].empty()) {
                        INXLOG_ERROR("Fullscreen pass '", passDesc.name,
                                     "' must bind each declared input from binding 0 without gaps");
                        return;
                    }
                    inputNames[*binding] = sourceName;
                }
                if (inputNames.empty())
                    inputNames = ImplicitFullscreenInputs(passDesc, m_pythonGraphDesc);
                struct FullscreenReadResource
                {
                    vk::ResourceHandle handle;
                    std::array<vk::ResourceHandle, EngineConfig::kFrameSlotCapacity> frameHandles{};
                    rhi::PixelFormat format = rhi::PixelFormat::Undefined;
                    bool depthRead = false;
                    bool storageBuffer = false;
                    bool frameLocal = false;
                    // Map nodes remain owned by this graph across frame rotation and camera cuts.
                    const bool *historyValidity = nullptr;
                    uint64_t byteOffset = 0;
                    uint64_t byteSize = 0;
                    rhi::SamplerHandle sampler;
                };
                std::vector<FullscreenReadResource> fsReadInputs;
                for (uint32_t binding = 0; binding < inputNames.size(); ++binding) {
                    const auto &name = inputNames[binding];
                    if (const auto buffer = bufferHandles.find(name); buffer != bufferHandles.end()) {
                        const auto reflected =
                            std::find_if(declaredInputs.begin(), declaredInputs.end(), [binding](const auto &item) {
                                return item.storage && item.binding == binding;
                            });
                        if (reflected == declaredInputs.end() || !reflected->readOnly || reflected->arraySize != 1) {
                            INXLOG_ERROR("Fullscreen pass '", passDesc.name, "' input '", name,
                                         "' requires one read-only storage-buffer descriptor at binding ", binding);
                            return;
                        }
                        const auto description =
                            std::find_if(m_pythonGraphDesc.buffers.begin(), m_pythonGraphDesc.buffers.end(),
                                         [&name](const GraphBufferDesc &item) { return item.name == name; });
                        if (description == m_pythonGraphDesc.buffers.end())
                            return;
                        const uint64_t inputByteSize =
                            description->viewLightList ? viewLightListByteSize : description->byteSize;
                        const auto &capabilities = m_vkCore->GetDeviceContext().GetRhiDevice().GetCapabilities();
                        if (!capabilities.IsStorageBufferRangeValid(description->byteOffset, inputByteSize)) {
                            INXLOG_ERROR("Fullscreen pass '", passDesc.name, "' buffer '", name,
                                         "' has an unsupported storage-buffer offset or range");
                            return;
                        }
                        if (const auto *type = ResolveGraphComputeBufferDesc(m_pythonGraphDesc, *description)) {
                            const auto scalar = type->scalarType;
                            const auto format =
                                scalar == rhi::ComputeScalarType::Float32 ? VK_FORMAT_R32_SFLOAT : VK_FORMAT_R32_UINT;
                            if (type->lanes != 1 ||
                                (scalar != rhi::ComputeScalarType::Float32 &&
                                 scalar != rhi::ComputeScalarType::UInt32) ||
                                reflected->scalarArrayFormat != format || reflected->arrayMemberOffset != 0 ||
                                reflected->arrayStride != sizeof(uint32_t) ||
                                reflected->arrayElementCount > inputByteSize / sizeof(uint32_t)) {
                                INXLOG_ERROR("Fullscreen pass '", passDesc.name, "' buffer '", name,
                                             "' requires a matching flat scalar array with offset 0 and stride 4");
                                return;
                            }
                        }
                        FullscreenReadResource input{};
                        input.handle = buffer->second;
                        input.format = rhi::PixelFormat::Undefined;
                        input.storageBuffer = true;
                        input.byteOffset = description->byteOffset;
                        input.byteSize = inputByteSize;
                        input.frameLocal = description->viewLightList;
                        if (input.frameLocal)
                            input.frameHandles = viewLightListHandles;
                        fsReadInputs.push_back(input);
                        continue;
                    }
                    const auto &texture = *texDescMap.at(name);
                    const bool historyInput = authorProgram && declaredInputs[binding].history;
                    const uint32_t samples = texture.EffectiveSamples(static_cast<uint32_t>(msaaSamples));
                    auto source = texture.IsViewColor() ? m_importedColorTarget : customRTHandles.at(name);
                    auto format = texture.EffectiveFormat(m_renderView);
                    rhi::SamplerHandle sampler;
                    if (texture.role == GraphTextureRole::Asset) {
                        const auto dimension = sampledDimensions.find(binding);
                        const auto expected =
                            texture.isVolume ? ReflectedImageDimension::D3 : ReflectedImageDimension::D2;
                        if (dimension == sampledDimensions.end() || dimension->second != expected) {
                            INXLOG_ERROR("Fullscreen pass '", passDesc.name, "' binds Texture asset '", name,
                                         "' to an incompatible shader resource dimension at binding ", binding);
                            return;
                        }
                    }
                    if (texture.role == GraphTextureRole::Asset || texture.role == GraphTextureRole::Persistent) {
                        const auto &imported = m_importedTextureBindings.at(name);
                        format = imported.format;
                        sampler = imported.sampler;
                    }
                    const bool rawSamples = multisampledBindings.count(binding) != 0;
                    if (rawSamples && samples == 1) {
                        INXLOG_ERROR("Fullscreen pass '", passDesc.name, "' binds single-sample texture '", name,
                                     "' to a Texture2DMS input");
                        return;
                    }
                    if (rawSamples || samples == 1) {
                        FullscreenReadResource input{};
                        if (historyInput)
                            input.historyValidity = &m_temporalHistories.at(texture.temporalKey).valid;
                        input.handle = source;
                        input.format = format;
                        input.depthRead = texture.IsDepth();
                        input.sampler = sampler;
                        fsReadInputs.push_back(input);
                        continue;
                    }
                    const auto inputExtent = texture.ResolveExtent(width, height);
                    const auto inputWidth = inputExtent.first;
                    const auto inputHeight = inputExtent.second;
                    if (texture.IsDepth()) {
                        if (!m_sceneDepthResolver.IsValid()) {
                            INXLOG_ERROR("SceneRenderGraph: sampled MSAA depth resolve is unavailable for pass '",
                                         passDesc.name, "'");
                            return;
                        }
                        if (!resolvedParticleSceneDepth.IsValid() || resolvedParticleDepthSource != source) {
                            const auto resolveName = "_SceneDepthResolve/" + std::to_string(depthResolvePassCounter++);
                            m_renderGraph->AddComputePass(resolveName, [&](vk::PassBuilder &builder) {
                                builder.SetQueueRole(rhi::QueueRole::Graphics);
                                builder.ReadSampledDepth(source, rhi::PipelineStage::ComputeShader);
                                const auto target = builder.CreateTexture("SceneDepthResolved", inputWidth, inputHeight,
                                                                          VK_FORMAT_R32_SFLOAT, VK_SAMPLE_COUNT_1_BIT);
                                resolvedParticleSceneDepth = builder.WriteStorageTexture(target);
                                const auto outputDepth = resolvedParticleSceneDepth;
                                return [this, source, outputDepth, inputWidth, inputHeight,
                                        samples](vk::RenderContext &ctx) {
                                    m_sceneDepthResolver.Record(
                                        ctx.GetComputeCommandEncoder(), ctx.GetTextureView(source),
                                        ctx.GetTextureView(outputDepth), inputWidth, inputHeight, samples);
                                };
                            });
                            resolvedParticleDepthSource = source;
                        }
                        FullscreenReadResource input{};
                        input.handle = resolvedParticleSceneDepth;
                        input.format = rhi::PixelFormat::R32SFloat;
                        fsReadInputs.push_back(input);
                        continue;
                    }
                    // Camera color shares its existing resolve; transient MSAA color
                    // resolves at the consuming version, never by averaging owner IDs
                    // explicitly declared Texture2DMS in the shader.
                    vk::ResourceHandle resolved =
                        texture.IsViewColor() ? m_importedResolveTarget : vk::ResourceHandle{};
                    if (!texture.IsViewColor() || backbufferDirtySinceResolve) {
                        const auto resolveName = "__MSAA_resolve_pre_fs_" + std::to_string(msaaResolvePassCounter++);
                        m_renderGraph->AddTransferPass(resolveName, [&](vk::PassBuilder &builder) {
                            builder.SetQueueRole(rhi::QueueRole::Graphics);
                            builder.TransferRead(source);
                            if (!resolved.IsValid())
                                resolved = builder.CreateTexture("FullscreenColorResolved", inputWidth, inputHeight,
                                                                 rhi::ToVkFormat(format), VK_SAMPLE_COUNT_1_BIT);
                            resolved = builder.TransferWrite(resolved);
                            builder.SetRenderArea(inputWidth, inputHeight);
                            return [source, resolved, inputWidth, inputHeight](vk::RenderContext &ctx) {
                                ctx.GetTransferCommandEncoder().ResolveTexture(
                                    ctx.GetTextureHandle(source), ctx.GetTextureHandle(resolved),
                                    {rhi::TextureAspect::Color, 0, 0, 0, 0, inputWidth, inputHeight, 1});
                            };
                        });
                        if (texture.IsViewColor()) {
                            publishResourceVersion(resolved);
                            backbufferDirtySinceResolve = false;
                        }
                    }
                    FullscreenReadResource input{};
                    input.handle = resolved;
                    input.format = format;
                    fsReadInputs.push_back(input);
                }

                // Capture references for the execute lambda
                FullscreenRenderer *fsRenderer = &m_fullscreenRenderer;
                std::string shaderName = command->shaderName;
                std::string parameterBlock = command->parameterBlock;
                const rhi::DepthState fsDepthState{command->depthTest, command->depthWrite,
                                                   rhi::ReverseDepthCompare(command->depthCompare)};
                const bool fsAlphaBlend = command->alphaBlend;
                FullscreenPushConstants packedPushConstants{};
                uint32_t packedPushConstantSize = 0;
                if (!authorProgram) {
                    for (const auto &[name, value] : command->parameters) {
                        packedPushConstants.values[packedPushConstantSize / sizeof(float)] = value.Read<float>();
                        packedPushConstantSize += sizeof(float);
                    }
                }

                // Determine output target (primary color)
                vk::ResourceHandle fsOutputTarget = primaryColorTarget;
                vk::ResourceHandle fsWrittenVersion;
                vk::ResourceHandle fsDepthTarget;
                vk::ResourceHandle fsWrittenDepthVersion;
                rhi::PixelFormat fsDepthFormat = rhi::PixelFormat::Undefined;
                if (!passDesc.writeDepth.empty()) {
                    fsDepthTarget = customRTHandles.at(passDesc.writeDepth);
                    fsDepthFormat = ResolveMaterialPass(m_pythonGraphDesc, passDesc, m_renderView).depthFormat;
                }

                // Determine MSAA sample count and output format.
                // When writing to the MSAA backbuffer the pipeline sample
                // count must match the render pass attachment.
                rhi::SampleCount fsSamples = rhi::SampleCount::One;
                rhi::PixelFormat fsColorFormat = rhi::FromVkFormat(m_sceneTarget->GetColorFormat());
                for (const auto &[slot, texName] : passDesc.writeColors) {
                    if (slot == 0 && !texName.empty()) {
                        auto texIt = texDescMap.find(texName);
                        if (texIt != texDescMap.end()) {
                            const uint32_t outputSamples =
                                texIt->second->EffectiveSamples(static_cast<uint32_t>(msaaSamples));
                            fsSamples = ToRhiSampleCount(static_cast<int>(outputSamples));
                            if (!texIt->second->IsViewColor() && texIt->second->format != rhi::PixelFormat::Undefined) {
                                fsColorFormat = texIt->second->format;
                            }
                        }
                    }
                }

                const auto fsPassExtent = ResolveGraphPassExtent(m_pythonGraphDesc, passDesc, width, height);
                const auto fsPassWidth = fsPassExtent.first;
                const auto fsPassHeight = fsPassExtent.second;

                vk::ResourceHandle fullscreenShadowInput;
                if (!fullscreenShadowDependencyDeclared && !perViewShadowTextureName.empty()) {
                    const auto shadow = customRTHandles.find(perViewShadowTextureName);
                    if (shadow != customRTHandles.end())
                        fullscreenShadowInput = shadow->second;
                }
                vk::ResourceHandle fsResolvedVersion;
                FullscreenPipelineKey fullscreenPipelineKey;
                fullscreenPipelineKey.depthFormat = fsDepthFormat;
                fullscreenPipelineKey.depth = fsDepthState;
                fullscreenPipelineKey.alphaBlend = fsAlphaBlend;
                fullscreenPipelineKey.shaderName = shaderName;
                fullscreenPipelineKey.samples = fsSamples;
                fullscreenPipelineKey.colorFormat = fsColorFormat;
                fullscreenPipelineKey.inputResourceCount = static_cast<uint32_t>(fsReadInputs.size());
                fullscreenPipelineKey.useDynamicRendering = true;
                fullscreenPipelineKey.program = authorProgram;
                for (uint32_t index = 0; index < fsReadInputs.size() && index < 32; ++index) {
                    if (fsReadInputs[index].storageBuffer)
                        fullscreenPipelineKey.inputBufferMask |= 1u << index;
                    if (fsReadInputs[index].depthRead)
                        fullscreenPipelineKey.depthInputMask |= 1u << index;
                }
                const auto pass = m_renderGraph->AddPass(passDesc.name, [=, &fsWrittenVersion, &fsResolvedVersion,
                                                                         &fsWrittenDepthVersion](
                                                                            vk::PassBuilder &builder) {
                    builder.SetSideEffect(passDesc.sideEffect || writesPersistent);
                    // Declare read dependencies for DAG edges + barriers
                    for (const auto &input : fsReadInputs) {
                        if (input.storageBuffer) {
                            if (input.frameLocal) {
                                for (const auto handle : input.frameHandles) {
                                    if (handle.IsValid())
                                        builder.ReadStorageBuffer(handle, rhi::PipelineStage::FragmentShader);
                                }
                            } else {
                                builder.ReadStorageBuffer(input.handle, rhi::PipelineStage::FragmentShader);
                            }
                        } else if (input.depthRead) {
                            builder.ReadSampledDepth(input.handle);
                        } else {
                            builder.Read(input.handle);
                        }
                    }
                    if (fullscreenShadowInput.IsValid() &&
                        std::none_of(fsReadInputs.begin(), fsReadInputs.end(),
                                     [&](const auto &input) { return input.handle == fullscreenShadowInput; })) {
                        builder.ReadSampledDepth(fullscreenShadowInput);
                    }
                    if (authorProgram && (authorProgram->UsesViewBinding(2) || authorProgram->UsesViewBinding(3))) {
                        for (const auto &resources : forwardPlusResources) {
                            if (!resources.canonicalLights.IsValid() || !resources.headers.IsValid() ||
                                !resources.lightMasks.IsValid()) {
                                continue;
                            }
                            builder.ReadStorageBuffer(resources.canonicalLights, rhi::PipelineStage::FragmentShader);
                            builder.ReadStorageBuffer(resources.headers, rhi::PipelineStage::FragmentShader);
                            builder.ReadStorageBuffer(resources.lightMasks, rhi::PipelineStage::FragmentShader);
                        }
                    }
                    // Declare color output
                    fsWrittenVersion = builder.WriteColor(fsOutputTarget, 0);
                    if (resolveTarget.IsValid())
                        fsResolvedVersion = builder.WriteResolve(resolveTarget);
                    if (fsDepthTarget.IsValid())
                        fsWrittenDepthVersion = builder.WriteDepth(fsDepthTarget);
                    builder.SetDepthTest(fsDepthState.testEnabled);
                    // Replacement passes fully define color. Partial coverage
                    // and blending load the preceding graph version unless the
                    // author explicitly clears it; never load a new transient.
                    if (clearColor)
                        builder.SetClearColor(clearColorR, clearColorG, clearColorB, clearColorA);
                    else if (!fsAlphaBlend && !fsDepthState.testEnabled)
                        builder.SetClearColor(0.0F, 0.0F, 0.0F, 0.0F);
                    if (clearDepth && fsDepthTarget.IsValid())
                        builder.SetClearDepth(clearDepthVal, 0);
                    builder.SetRenderArea(fsPassWidth, fsPassHeight);
                    return [=](vk::RenderContext &ctx) {
                        // Resolve input texture views using a stack path for the common case.
                        FullscreenResourceInput inputsStack[8] = {};
                        std::vector<FullscreenResourceInput> inputsHeap;
                        FullscreenResourceInput *inputs = inputsStack;
                        if (fsReadInputs.size() > 8) {
                            inputsHeap.resize(fsReadInputs.size());
                            inputs = inputsHeap.data();
                        }
                        const uint32_t inputCount = static_cast<uint32_t>(fsReadInputs.size());
                        for (uint32_t i = 0; i < inputCount; ++i) {
                            const auto &input = fsReadInputs[i];
                            if (input.storageBuffer) {
                                auto resource = input.handle;
                                auto byteSize = input.byteSize;
                                if (input.frameLocal) {
                                    const auto slot =
                                        m_vkCore->GetCurrentFrameSlot() % EngineConfig::kFrameSlotCapacity;
                                    resource = input.frameHandles[slot];
                                    byteSize = m_cameraCanonicalLights.Frame(slot).capacityBytes;
                                }
                                inputs[i].buffer = ctx.GetBufferHandle(resource);
                                inputs[i].byteOffset = input.byteOffset;
                                inputs[i].byteSize = byteSize;
                                if (!inputs[i].buffer.IsValid()) {
                                    INXLOG_ERROR("FullscreenQuad '", shaderName,
                                                 "': input buffer is unavailable at binding ", i);
                                    return;
                                }
                                continue;
                            }
                            inputs[i].view = ctx.GetTextureView(input.handle);
                            inputs[i].sampler = input.sampler;
                            inputs[i].format = input.format;
                            inputs[i].depthRead = input.depthRead;
                            inputs[i].historyValid = input.historyValidity && *input.historyValidity;
                            if (!inputs[i].view.IsValid()) {
                                INXLOG_ERROR("FullscreenQuad '", shaderName,
                                             "': input texture view is unavailable at binding ", i);
                                return;
                            }
                        }

                        const auto &entry = fsRenderer->EnsurePipeline(fullscreenPipelineKey);
                        if (!entry.pipeline.IsValid())
                            return;

                        if (authorProgram) {
                            const auto *values = parameterBlock.empty()
                                                     ? nullptr
                                                     : m_submittedParameterBlocks.at(parameterBlock).values->data();
                            const auto &frame = m_perViewFrames[m_vkCore->GetCurrentFrameSlot()];
                            const FullscreenViewResources viewResources{
                                {0, rhi::BindingType::UniformBuffer, frame.globals.buffer, 0, sizeof(EngineGlobalsUBO)},
                                frame.geometryGroup,
                            };
                            fsRenderer->DrawProgram(ctx.GetGraphicsCommandEncoder(), entry, inputs, authorParameters,
                                                    values, viewResources);
                            return;
                        }

                        // Allocate descriptor set for input textures
                        const auto bindGroup = fsRenderer->AllocateBindGroup(entry.inputLayout, inputs, inputCount,
                                                                             fsRenderer->GetLinearSampler());
                        if (!bindGroup.IsValid()) {
                            INXLOG_ERROR("FullscreenQuad '", shaderName, "': descriptor pool exhausted, skipping pass");
                            return;
                        }

                        // Dynamic blocks replace only values. Shader, parameter
                        // order, and byte size remain part of compiled topology.
                        FullscreenPushConstants drawPushConstants = packedPushConstants;
                        uint32_t drawPushConstantSize = packedPushConstantSize;
                        if (!parameterBlock.empty()) {
                            const auto &block = m_submittedParameterBlocks.at(parameterBlock);
                            drawPushConstants = *block.scalarPushConstants;
                            drawPushConstantSize = static_cast<uint32_t>(block.values->size() * sizeof(float));
                        }
                        fsRenderer->Draw(ctx.GetGraphicsCommandEncoder(), entry, bindGroup, GetPerViewBindGroup(),
                                         vkCore->ResolveViewGlobalsBinding(GetViewGlobalsBinding()), drawPushConstants,
                                         drawPushConstantSize);
                    };
                });
                trackAuthorPass(pass);
                if (fullscreenShadowInput.IsValid()) {
                    m_shadowMapInputHandle = fullscreenShadowInput;
                    m_shadowMapInputIsDepth = true;
                    fullscreenShadowDependencyDeclared = true;
                }
                publishResourceVersion(fsWrittenVersion);
                publishResourceVersion(fsResolvedVersion);
                publishResourceVersion(fsWrittenDepthVersion);
                if (sharedDepth.IsValid() && sharedDepth.id == fsWrittenDepthVersion.id) {
                    sharedDepth = fsWrittenDepthVersion;
                    m_importedDepthTarget = sharedDepth;
                }

                if (writesBackbuffer) {
                    backbufferDirtySinceResolve = true;
                }
                continue;
            }

            // Resolve the sample at this pass's position, after earlier writers
            // published their SSA versions. An upfront handle would read v0.
            std::vector<vk::ResourceHandle> drawTextureReads;
            if (const auto inputs = m_drawTextureInputs.find(passDesc.name); inputs != m_drawTextureInputs.end()) {
                for (const auto &generation : inputs->second) {
                    const auto resources = m_renderGraph->ImportRenderTexture(
                        "__DrawTexture/" + generation->sampledColor->GetSourceId(), generation);
                    drawTextureReads.push_back(generation->multisampleColor ? resources.resolve : resources.color);
                }
            }
            std::vector<vk::ResourceHandle> writtenColorVersions;
            vk::ResourceHandle writtenDepthVersion;
            vk::ResourceHandle writtenResolveVersion;
            const bool usesShadowRendererList = command && command->type == GraphCommandType::DrawShadowCasters;
            const bool usesVisibleRendererList = command && (command->type == GraphCommandType::DrawRenderers ||
                                                             command->type == GraphCommandType::DrawSkybox);
            VkFormat shadowDepthFormat = VK_FORMAT_D32_SFLOAT;
            if (usesShadowRendererList && !passDesc.writeDepth.empty()) {
                const auto shadowDepthIt = texDescMap.find(passDesc.writeDepth);
                if (shadowDepthIt != texDescMap.end())
                    shadowDepthFormat = rhi::ToVkFormat(shadowDepthIt->second->format);
            }
            const int shadowQueueMin = command ? command->queueMin : 0;
            const int shadowQueueMax = command ? command->queueMax : 2999;
            const int shadowLightIndex = command ? command->lightIndex : 0;
            const vk::ResourceHandle rendererListHandle =
                usesShadowRendererList ? m_shadowRendererList
                                       : (usesVisibleRendererList ? m_visibleRendererList : vk::ResourceHandle{});
            const auto selectiveSource = worldUIOpaqueSources.find(passDesc.name);
            if (selectiveSource != worldUIOpaqueSources.end()) {
                if (!primaryColorTarget.IsValid() || !sharedDepth.IsValid() || colorTargets.size() != 1 ||
                    colorTargets.begin()->first != 0) {
                    INXLOG_ERROR("Selective World UI requires one existing color target and scene depth");
                    return;
                }
                const auto &worldUIOpaqueSource = selectiveSource->second;
                const auto worldUIDepthRuns = m_screenUIRenderer->GetWorldDepthRuns(
                    m_cachedProj * m_drawView, cameraWorldUIMask & command->worldUILayerMask);
                const auto opaqueCommand = worldUIOpaqueSource.draw;
                auto replayPipeline = m_pythonMaterialPasses.at(worldUIOpaqueSource.passName);
                const auto uiMaterialPass = m_pythonMaterialPasses.at(passDesc.name);
                const uint32_t worldUILayerMask = command->worldUILayerMask;
                replayPipeline.target = ShaderCompileTarget::Depth;
                replayPipeline.colorFormats.clear();
                replayPipeline.depthReadOnly = false;
                std::vector<vk::ResourceHandle> replayTextureReads;
                if (const auto inputs = m_drawTextureInputs.find(worldUIOpaqueSource.passName);
                    inputs != m_drawTextureInputs.end()) {
                    for (const auto &generation : inputs->second) {
                        const auto resources = m_renderGraph->ImportRenderTexture(
                            "__DrawTexture/" + generation->sampledColor->GetSourceId(), generation);
                        replayTextureReads.push_back(generation->multisampleColor ? resources.resolve
                                                                                  : resources.color);
                    }
                }
                std::unordered_map<uint64_t, vk::ResourceHandle> alternateDepths;
                for (const auto &run : worldUIDepthRuns) {
                    const uint64_t excludedId = run.ignoredOccluderId;
                    if (!excludedId || alternateDepths.count(excludedId))
                        continue;
                    const std::string replayName = passDesc.name + "/Exclude/" + std::to_string(excludedId);
                    auto alternateDepth = m_renderGraph->RegisterTransientTexture(replayName + "/Depth", width, height,
                                                                                  depthFormat, msaaSamples);
                    vk::ResourceHandle writtenAlternateDepth;
                    m_renderGraph->AddPass(replayName, [=, &writtenAlternateDepth](vk::PassBuilder &builder) {
                        builder.ReadRendererList(m_visibleRendererList);
                        DeclareMaterialBufferReads(builder, worldUIOpaqueSource.passName);
                        for (const auto texture : replayTextureReads)
                            builder.Read(texture,
                                         rhi::PipelineStage::VertexShader | rhi::PipelineStage::FragmentShader);
                        writtenAlternateDepth = builder.WriteDepth(alternateDepth);
                        builder.SetRenderArea(width, height);
                        builder.SetClearDepth(0.0f, 0);
                        return [this, vkCore, opaqueCommand, replayPipeline, excludedId, width,
                                height](vk::RenderContext &ctx) {
                            const auto *list = ctx.GetRendererList(m_visibleRendererList);
                            const auto *draws = list ? &list->DrawCalls() : nullptr;
                            if (!vkCore->UsesDrawCalls(draws))
                                vkCore->SetDrawCalls(draws);
                            vkCore->DrawSceneFiltered(ctx.GetCommandBuffer(), width, height, GetPerViewBindGroup(),
                                                      GetViewGlobalsBinding(), m_drawView, opaqueCommand.queueMin,
                                                      opaqueCommand.queueMax, opaqueCommand.sortMode, "",
                                                      opaqueCommand.passTag, &replayPipeline,
                                                      opaqueCommand.materialFilter, nullptr, excludedId, true);
                        };
                    });
                    alternateDepths.emplace(excludedId, writtenAlternateDepth);
                }

                vk::ResourceHandle currentColor = primaryColorTarget;
                for (size_t runIndex = 0; runIndex < worldUIDepthRuns.size(); ++runIndex) {
                    const auto run = worldUIDepthRuns[runIndex];
                    const auto runDepth =
                        run.ignoredOccluderId ? alternateDepths.at(run.ignoredOccluderId) : sharedDepth;
                    const std::string runName = passDesc.name + "/Run/" + std::to_string(runIndex);
                    const bool lastRun = runIndex + 1 == worldUIDepthRuns.size();
                    vk::ResourceHandle writtenColor;
                    vk::ResourceHandle writtenResolve;
                    const auto pass = m_renderGraph->AddPass(runName, [=, &writtenColor,
                                                                       &writtenResolve](vk::PassBuilder &builder) {
                        builder.SetSideEffect(passDesc.sideEffect || writesPersistent);
                        builder.ReadDepth(runDepth);
                        for (const auto texture : drawTextureReads)
                            builder.Read(texture,
                                         rhi::PipelineStage::VertexShader | rhi::PipelineStage::FragmentShader);
                        writtenColor = builder.WriteColor(currentColor, 0);
                        if (lastRun && resolveTarget.IsValid())
                            writtenResolve = builder.WriteResolve(resolveTarget);
                        builder.SetRenderArea(width, height);
                        return [this, vkCore, uiMaterialPass, worldUILayerMask, run, width,
                                height](vk::RenderContext &ctx) {
                            m_screenUIRenderer->RenderWorld(
                                ctx.GetCommandBuffer(), width, height, m_cachedProj * m_drawView,
                                uiMaterialPass.RenderingSignature(), vkCore->GetCurrentFrameSlot(),
                                worldUILayerMask & (m_cachedCamera ? m_cachedCamera->GetCullingMask() : 0xffffffffu),
                                m_drawView, m_cachedProj, m_hasBillboardView ? m_billboardView : m_drawView,
                                run.firstOrdinal, run.endOrdinal);
                        };
                    });
                    trackAuthorPass(pass);
                    currentColor = writtenColor;
                    publishResourceVersion(writtenColor);
                    publishResourceVersion(writtenResolve);
                }
                if (writesBackbuffer)
                    backbufferDirtySinceResolve = true;
                continue;
            }
            const auto pass = m_renderGraph->AddPass(passDesc.name, [=, &sharedDepth, &writtenColorVersions,
                                                                     &writtenDepthVersion,
                                                                     &writtenResolveVersion](vk::PassBuilder &builder) {
                builder.SetSideEffect(passDesc.sideEffect || writesPersistent);
                // Local alias to make vkCore capturable by nested lambdas (MSVC C3481)
                InxVkCoreModular *localVkCore = vkCore;
                DeclareMaterialBufferReads(builder, passDesc.name);

                struct ParticlePacket
                {
                    std::shared_ptr<particle::ParticleGpuOutputRenderer> renderer;
                    vk::ResourceHandle instances;
                    vk::ResourceHandle renderIndices;
                    vk::ResourceHandle indirectArguments;
                    rhi::BufferHandle drawRenderIndices;
                    uint32_t ownerLayerMask = 1u;
                };
                std::vector<ParticlePacket> particlePackets;
                particlePackets.reserve(particleEntries.size());

                if (command && command->type == GraphCommandType::DrawRenderers &&
                    command->shaderTarget == ShaderCompileTarget::ForwardPlus) {
                    for (const auto &resources : forwardPlusResources) {
                        if (!resources.canonicalLights.IsValid() || !resources.headers.IsValid() ||
                            !resources.lightMasks.IsValid()) {
                            continue;
                        }
                        builder.ReadStorageBuffer(resources.canonicalLights, rhi::PipelineStage::FragmentShader);
                        builder.ReadStorageBuffer(resources.headers, rhi::PipelineStage::FragmentShader);
                        builder.ReadStorageBuffer(resources.lightMasks, rhi::PipelineStage::FragmentShader);
                    }
                    const bool hasLitParticles =
                        std::any_of(particleEntries.begin(), particleEntries.end(),
                                    [](const auto &entry) { return entry.semantics.receiveSceneLighting; });
                    if (hasLitParticles) {
                        for (const auto &resources : particleForwardPlusResources) {
                            if (!resources.canonicalLights.IsValid() || !resources.headers.IsValid() ||
                                !resources.lightMasks.IsValid()) {
                                continue;
                            }
                            builder.ReadStorageBuffer(resources.canonicalLights, rhi::PipelineStage::FragmentShader);
                            builder.ReadStorageBuffer(resources.headers, rhi::PipelineStage::FragmentShader);
                            builder.ReadStorageBuffer(resources.lightMasks, rhi::PipelineStage::FragmentShader);
                        }
                    }
                }
                for (const auto &entry : particleEntries) {
                    vk::ResourceHandle instances;
                    vk::ResourceHandle renderIndices;
                    vk::ResourceHandle indirectArguments;
                    rhi::BufferHandle drawRenderIndices = entry.renderIndices;
                    const auto sorted = particleGraphResources.find(entry.id);
                    if (!usesShadowRendererList && sorted != particleGraphResources.end()) {
                        instances = sorted->second.instances;
                        renderIndices = sorted->second.renderIndices;
                        indirectArguments = sorted->second.indirectArguments;
                        drawRenderIndices = sorted->second.drawRenderIndices;
                    } else {
                        const std::string prefix = "GpuParticle/" + std::to_string(entry.id);
                        instances = builder.ImportBuffer(prefix + "/Instances", entry.instances,
                                                         static_cast<uint64_t>(entry.capacity) *
                                                             particle::ParticleGpuRuntime::RenderInstanceStride);
                        renderIndices = builder.ImportBuffer(prefix + "/RenderIndices", entry.renderIndices,
                                                             static_cast<uint64_t>(entry.capacity) * sizeof(uint32_t));
                        indirectArguments = builder.ImportBuffer(prefix + "/Indirect", entry.indirectArguments, 16);
                        m_renderGraph->SetResourceInitialState(
                            instances, rhi::TextureLayout::Undefined, rhi::Access::ShaderWrite,
                            rhi::PipelineStage::ComputeShader, rhi::QueueRole::Compute);
                        m_renderGraph->SetResourceInitialState(
                            indirectArguments, rhi::TextureLayout::Undefined, rhi::Access::ShaderWrite,
                            rhi::PipelineStage::ComputeShader, rhi::QueueRole::Compute);
                        m_renderGraph->SetResourceInitialState(
                            renderIndices, rhi::TextureLayout::Undefined, rhi::Access::ShaderWrite,
                            rhi::PipelineStage::ComputeShader, rhi::QueueRole::Compute);
                    }
                    if (!instances.IsValid() || !renderIndices.IsValid() || !indirectArguments.IsValid())
                        continue;
                    builder.ReadStorageBuffer(instances, rhi::PipelineStage::VertexShader);
                    builder.ReadStorageBuffer(renderIndices, rhi::PipelineStage::VertexShader);
                    if (entry.cullMode == particle::GpuParticleCullMode::RibbonSegments &&
                        sorted != particleGraphResources.end() && sorted->second.sourceRenderIndices.IsValid() &&
                        sorted->second.sourceRenderIndices.id != renderIndices.id) {
                        builder.ReadStorageBuffer(sorted->second.sourceRenderIndices, rhi::PipelineStage::VertexShader);
                    }
                    uint32_t staticBufferIndex = 0;
                    for (const auto &staticBuffer : entry.renderer->StaticVertexStorageBuffers()) {
                        const std::string name = "GpuParticle/" + std::to_string(entry.id) + "/StaticVertex" +
                                                 std::to_string(staticBufferIndex++);
                        const auto handle = builder.ImportBuffer(name, staticBuffer.buffer, staticBuffer.byteSize);
                        if (!handle.IsValid())
                            continue;
                        m_renderGraph->SetResourceInitialState(handle, rhi::TextureLayout::Undefined,
                                                               rhi::Access::TransferWrite,
                                                               rhi::PipelineStage::Transfer);
                        builder.ReadStorageBuffer(handle, rhi::PipelineStage::VertexShader);
                    }
                    builder.ReadIndirectBuffer(indirectArguments);
                    particlePackets.push_back({entry.renderer, instances, renderIndices, indirectArguments,
                                               drawRenderIndices, entry.ownerLayerMask});
                }

                if (rendererListHandle.IsValid()) {
                    builder.ReadRendererList(rendererListHandle);
                    // A shadow pass with no casters must still clear a
                    // previously populated persistent atlas.
                    builder.SkipCallbackWhenRendererListsEmpty(!usesShadowRendererList && particlePackets.empty());
                }

                const auto passExtent = ResolveGraphPassExtent(m_pythonGraphDesc, passDesc, width, height);
                const auto passWidth = passExtent.first;
                const auto passHeight = passExtent.second;
                bool isShadowPass = command && command->type == GraphCommandType::DrawShadowCasters;

                // ----- Depth -----
                vk::ResourceHandle depth;
                if (!passDesc.writeDepth.empty()) {
                    // Every declared attachment has already been registered.
                    auto rtIt = customRTHandles.find(passDesc.writeDepth);
                    if (rtIt != customRTHandles.end()) {
                        depth = rtIt->second;
                        writtenDepthVersion = builder.WriteDepth(depth);
                        depth = writtenDepthVersion;
                    }
                }
                if (!depth.IsValid() && needsCreateDepth) {
                    // First pass that writes depth: create the shared resource
                    depth = builder.CreateDepthStencil("SceneDepth", width, height, depthFormat, msaaSamples);
                    writtenDepthVersion = builder.WriteDepth(depth);
                    depth = writtenDepthVersion;
                    // Store for subsequent passes (captured by ref)
                    sharedDepth = depth;
                } else if (!depth.IsValid() && writesDepth && depthForThisPass.IsValid()) {
                    // Later pass that also writes depth (rare)
                    writtenDepthVersion = builder.WriteDepth(depthForThisPass);
                } else if (!depth.IsValid() && passReadsDepth && depthForThisPass.IsValid()) {
                    // Pass reads depth (e.g., skybox, transparent) — attach as read-only
                    builder.ReadDepth(depthForThisPass);
                }
                if (particleSceneDepth.IsValid()) {
                    if (particleSceneDepthIsDepth)
                        builder.ReadSampledDepth(particleSceneDepth, rhi::PipelineStage::FragmentShader);
                    else
                        builder.Read(particleSceneDepth, rhi::PipelineStage::FragmentShader);
                }

                // Scene-sized depth textures are pre-registered in customRTHandles,
                // so later writers commonly take the first branch above. Keep the
                // shared read-only alias on the same SSA version; otherwise skybox
                // and editor passes keep reading the initial depth after it has been
                // overwritten, which creates a real physical-resource scheduling
                // cycle in the versioned RenderGraph.
                if (sharedDepth.IsValid() && writtenDepthVersion.IsValid() &&
                    sharedDepth.id == writtenDepthVersion.id) {
                    sharedDepth = writtenDepthVersion;
                }

                // ----- Color reads (non-depth textures) -----
                // Declare Read() for each color texture this pass reads.
                // This creates proper DAG edges and Vulkan barriers.
                for (const auto &readHandle : colorReadHandles) {
                    builder.Read(readHandle);
                }
                for (const auto handle : drawTextureReads)
                    builder.Read(handle, rhi::RenderTextureSampleStages);

                // ----- Input binding reads (sampled textures, e.g. shadow map) -----
                // Input bindings reference textures by name for descriptor
                // binding at draw time. We also need DAG edges here so that:
                //   1. The writer pass is not dead-pass-culled.
                //   2. Vulkan barriers transition the texture for shader read.
                for (const auto &binding : inputBindingHandles) {
                    if (binding.isDepth) {
                        builder.ReadSampledDepth(binding.handle);
                    } else {
                        builder.Read(binding.handle);
                    }
                }

                // ----- Color outputs (MRT) -----
                // Write all declared color targets at their respective slots.
                for (const auto &[slot, handle] : colorTargets) {
                    auto written = builder.WriteColor(handle, slot);
                    if (written.IsValid())
                        writtenColorVersions.push_back(written);
                }

                // ----- MSAA Resolve (only on the last backbuffer pass) -----
                if (resolveTarget.IsValid()) {
                    writtenResolveVersion = builder.WriteResolve(resolveTarget);
                }

                // ----- Render area -----
                builder.SetRenderArea(passWidth, passHeight);
                // ----- Clear values -----
                if (clearColor) {
                    builder.SetClearColor(clearColorR, clearColorG, clearColorB, clearColorA);
                }
                if (clearDepth && !isShadowPass) {
                    builder.SetClearDepth(clearDepthVal, 0);
                }

                for (const auto &binding : inputBindingHandles) {
                    if (binding.samplerName == "shadowMap" && !m_shadowMapInputHandle.IsValid()) {
                        m_shadowMapInputHandle = binding.handle;
                        m_shadowMapInputIsDepth = binding.isDepth;
                    }
                }

                return [this, callback, passWidth, passHeight, inputBindingHandles, isShadowPass, rendererListHandle,
                        usesShadowRendererList, shadowDepthFormat, localVkCore, particlePackets, particlePass,
                        particleSceneDepth, particleSceneDepthIsDepth, shadowQueueMin, shadowQueueMax,
                        shadowLightIndex](vk::RenderContext &ctx) {
                    if (rendererListHandle.IsValid()) {
                        const RendererList *rendererList = ctx.GetRendererList(rendererListHandle);
                        if (usesShadowRendererList) {
                            localVkCore->SetShadowDrawCalls(rendererList ? &rendererList->DrawCalls() : nullptr);
                        } else {
                            const auto *drawCalls = rendererList ? &rendererList->DrawCalls() : nullptr;
                            if (!localVkCore->UsesDrawCalls(drawCalls))
                                localVkCore->SetDrawCalls(drawCalls);
                        }
                    }
                    if (isShadowPass) {
                        if (!m_shadowAtlasUpdateRequired)
                            return;
                        VkClearAttachment clearAttachment{};
                        clearAttachment.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
                        clearAttachment.clearValue.depthStencil = {0.0f, 0};
                        VkClearRect clearRect{};
                        clearRect.rect.offset = {0, 0};
                        clearRect.rect.extent = {passWidth, passHeight};
                        clearRect.baseArrayLayer = 0;
                        clearRect.layerCount = 1;
                        vkCmdClearAttachments(ctx.GetCommandBuffer(), 1, &clearAttachment, 1, &clearRect);
                        auto &encoder = ctx.GetGraphicsCommandEncoder();
                        localVkCore->DrawShadowCasters(
                            ctx.GetCommandBuffer(), passWidth, passHeight, shadowQueueMin, shadowQueueMax,
                            m_shadowCameraResourceId,
                            m_vkCore->GetDeviceContext().GetRhiDevice().Resolve(GetViewGlobalsBinding().buffer),
                            m_cameraLightCollector.GetShadowFrame(), shadowLightIndex, shadowDepthFormat,
                            [&](uint32_t, const lighting::ShadowView &activeShadowView) {
                                particle::GpuParticleViewConstants shadowView;
                                std::memcpy(shadowView.viewProjection.data(), &activeShadowView.viewProjection[0][0],
                                            sizeof(activeShadowView.viewProjection));
                                shadowView.previousViewProjection = shadowView.viewProjection;
                                // Particle billboards expand along these axes in
                                // the vertex shader; the caster pass must face
                                // the light, not the camera, or the quads
                                // collapse into degenerate shadow silhouettes.
                                shadowView.cameraRight = {
                                    activeShadowView.viewRight.x,
                                    activeShadowView.viewRight.y,
                                    activeShadowView.viewRight.z,
                                    0.0f,
                                };
                                shadowView.cameraUp = {
                                    activeShadowView.viewUp.x,
                                    activeShadowView.viewUp.y,
                                    activeShadowView.viewUp.z,
                                    0.0f,
                                };
                                for (const auto &packet : particlePackets) {
                                    [[maybe_unused]] const bool recorded = packet.renderer->RecordDraw(
                                        encoder, particlePass, ctx.GetBufferHandle(packet.indirectArguments),
                                        shadowView, packet.drawRenderIndices);
                                    ctx.RecordParticleDraw(true);
                                }
                            });
                        CommitShadowAtlasUpdate();
                        return;
                    }
                    if (callback)
                        callback(ctx, passWidth, passHeight);
                    if (!particlePackets.empty()) {
                        particle::GpuParticleViewConstants view;
                        const glm::mat4 viewProjection = m_cachedProj * m_cachedView;
                        const glm::mat4 previousViewProjection = GetPreviousViewProj();
                        const glm::mat4 inverseView = glm::inverse(m_cachedView);
                        std::memcpy(view.viewProjection.data(), &viewProjection[0][0], sizeof(viewProjection));
                        std::memcpy(view.previousViewProjection.data(), &previousViewProjection[0][0],
                                    sizeof(previousViewProjection));
                        std::memcpy(view.cameraRight.data(), &inverseView[0][0], sizeof(glm::vec4));
                        std::memcpy(view.cameraUp.data(), &inverseView[1][0], sizeof(glm::vec4));
                        std::memcpy(view.alignmentReference.data(), &inverseView[3][0], sizeof(glm::vec4));
                        view.depthReconstruct = {m_cachedProj[2][2], m_cachedProj[3][2], m_cachedProj[2][3],
                                                 m_cachedProj[3][3]};
                        auto &encoder = ctx.GetGraphicsCommandEncoder();
                        const auto sceneDepth = particleSceneDepth.IsValid() ? ctx.GetTextureView(particleSceneDepth)
                                                                             : rhi::TextureViewHandle{};
                        particle::GpuParticlePerViewBindings particlePerView;
                        const uint32_t frameIndex = m_vkCore->GetCurrentFrameSlot() % EngineConfig::kFrameSlotCapacity;
                        const auto &viewFrame = m_perViewFrames[frameIndex];
                        if (m_perViewLayout.IsValid() && viewFrame.particleGroup.IsValid()) {
                            particlePerView.layout = m_perViewLayout;
                            particlePerView.group = viewFrame.particleGroup;
                        }
                        for (const auto &packet : particlePackets) {
                            const uint32_t cameraMask = m_cachedCamera ? m_cachedCamera->GetCullingMask() : 0xffffffffu;
                            if ((packet.ownerLayerMask & cameraMask) == 0)
                                continue;
                            auto packetView = view;
                            std::memcpy(&packetView.lightingControl[3], &packet.ownerLayerMask,
                                        sizeof(packet.ownerLayerMask));
                            [[maybe_unused]] const bool recorded = packet.renderer->RecordDraw(
                                encoder, particlePass, ctx.GetBufferHandle(packet.indirectArguments), packetView,
                                packet.drawRenderIndices,
                                packet.renderer->RequiresSceneDepth() ? sceneDepth : rhi::TextureViewHandle{},
                                particleSceneDepthIsDepth, particlePerView);
                            ctx.RecordParticleDraw(true);
                        }
                    }
                };
            });

            trackAuthorPass(pass);
            for (const auto &written : writtenColorVersions)
                publishResourceVersion(written);
            publishResourceVersion(writtenDepthVersion);
            publishResourceVersion(writtenResolveVersion);

            if (sharedDepth.IsValid()) {
                m_importedDepthTarget = sharedDepth;
            }

            if (writesBackbuffer) {
                backbufferDirtySinceResolve = true;
            }
        }

        // ====================================================================
        // Persistent outputs remain observable even when not used by the screen.
        for (const auto &texture : m_pythonGraphDesc.textures) {
            if (texture.role != GraphTextureRole::Persistent && texture.role != GraphTextureRole::TemporalRead &&
                texture.role != GraphTextureRole::TemporalWrite)
                continue;
            // Every imported attachment leaves the graph in its declared
            // handoff layout, including unsampleable MSAA/depth attachments.
            const bool sampleable = texture.samples == 1 &&
                                    (!texture.IsDepth() || texture.renderTexture->Acquire()->description.sampledDepth);
            const auto output = customRTHandles.at(texture.name);
            m_renderGraph->AddPass("__Export/" + texture.name,
                                   [output, isDepth = texture.IsDepth(), sampleable](vk::PassBuilder &builder) {
                                       if (!sampleable && isDepth)
                                           builder.PrepareDepthStencilAttachment(output);
                                       else if (!sampleable)
                                           builder.PrepareColorAttachment(output);
                                       else if (isDepth)
                                           builder.ReadSampledDepth(output, rhi::RenderTextureSampleStages);
                                       else
                                           builder.Read(output, rhi::RenderTextureSampleStages);
                                       builder.SetSideEffect();
                                       return [](vk::RenderContext &) {};
                                   });
        }

        for (const auto &buffer : m_pythonGraphDesc.buffers)
            if (buffer.exportLayout) {
                const auto value = bufferHandles.at(buffer.name);
                m_renderGraph->AddPass("__ExportBuffer/" + buffer.name, [value](vk::PassBuilder &builder) {
                    builder.ReadStorageBuffer(value, rhi::PipelineStage::AllCommands);
                    builder.SetSideEffect();
                    return [](vk::RenderContext &) {};
                });
            }

        // Custom author pipelines may omit the built-in display encode. Keep
        // their editor overlays observable at the graph tail without adding
        // anything to Game/Player graphs.
        appendSceneOverlays();
    }

    // Set output for proper resource tracking and dead-pass culling.
    FinalizeGraphOutput(customRTHandles);

    ++m_graphBuildRevision;
    m_particleDrawRegistryRevision = m_particleDrawRegistry ? m_particleDrawRegistry->Revision() : 0;
    m_graphBuilt = true;
}

} // namespace infernux
