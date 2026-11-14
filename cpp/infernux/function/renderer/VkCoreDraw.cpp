/**
 * @file VkCoreDraw.cpp
 * @brief InxVkCoreModular — Drawing and per-object buffer management
 *
 * Split from InxVkCoreModular.cpp for maintainability.
 * Contains: DrawFrame, DrawSceneFiltered,
 *           SetDrawCalls, mesh binding, and unused buffer collection.
 */

#include "InxError.h"
#include "InxVkCoreModular.h"
#include "ProfileConfig.h"
#include "RendererSelection.h"
#include "SceneRenderGraph.h"
#include "vk/DescriptorBindTrace.h"
#include "vk/RhiVulkanTypes.h"
#include "vk/VkRenderUtils.h"
#include "vk/VkTypes.h"

#include <core/threading/FrameWorkOwner.h>
#include <function/renderer/Frustum.h>
#include <function/renderer/shader/ShaderProgram.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxMesh/InxMesh.h>
#include <function/scene/LightingData.h>

#include <SDL3/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace infernux
{

namespace
{

struct alignas(16) ShadowPassParameters
{
    glm::vec4 lightVector{};
    glm::vec4 bias{};
};

static_assert(sizeof(ShadowPassParameters) == 32);

VkIndexType ToVkIndexType(MeshIndexFormat format)
{
    if (format == MeshIndexFormat::UInt16)
        return VK_INDEX_TYPE_UINT16;
    if (format == MeshIndexFormat::UInt32)
        return VK_INDEX_TYPE_UINT32;
    throw std::logic_error("Vulkan draw received an unresolved mesh index format");
}

bool IsPackagedPlayerRuntime()
{
    const char *playerMode = std::getenv("_INFERNUX_PLAYER_MODE");
    return playerMode != nullptr && std::strcmp(playerMode, "1") == 0;
}

} // namespace

// ============================================================================
// Rendering
// ============================================================================

uint64_t InxVkCoreModular::DrawFrame(WindowOutput output, std::array<double, 6> *phaseTimings)
try {
    struct DrawBufferBindingsScope
    {
        InxVkCoreModular &core;
        ~DrawBufferBindingsScope()
        {
            core.SetDrawBufferBindings(nullptr);
        }
    } drawBufferBindingsScope{*this};
    if (m_frameFailure)
        std::rethrow_exception(m_frameFailure);
    auto phaseStart = phaseTimings ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (phaseTimings)
        phaseTimings->fill(0.0);
#if INFERNUX_FRAME_PROFILE
    using Clock = std::chrono::high_resolution_clock;
    auto _t0 = Clock::now();
    auto _tPrev = _t0;
    auto _tNow = _t0;
#endif

    const uint32_t frameSlot = GetCurrentFrameSlot();
    auto &presentation = m_backend.Presentation();
    const VkExtent2D extent = presentation.GetExtent();
    uint32_t imageIndex = 0;
    bool presenting = false;
    if (output != WindowOutput::Disabled && extent.width && extent.height) {
        const auto result = presentation.AcquireNextImage(frameSlot, imageIndex);
        switch (result) {
        case vk::SwapchainResult::Success:
            presenting = true;
            break;
        case vk::SwapchainResult::SkipFrame:
            break;
        case vk::SwapchainResult::NeedRecreate:
            m_framebufferResized = true;
            break;
        case vk::SwapchainResult::SurfaceLost:
            m_presentationSurfaceLost = true;
            INXLOG_WARN("Platform presentation surface was lost while acquiring an image; scheduling a surface rebind");
            break;
        case vk::SwapchainResult::Error:
            throw std::runtime_error("Failed to acquire swapchain image");
        }
    }
    if (phaseTimings) {
        const auto now = std::chrono::steady_clock::now();
        (*phaseTimings)[0] = std::chrono::duration<double, std::milli>(now - phaseStart).count();
        phaseStart = now;
    }
#if INFERNUX_FRAME_PROFILE
    _tNow = Clock::now();
    m_drawSubMs[0] += std::chrono::duration<double, std::milli>(_tNow - _tPrev).count();
    _tPrev = _tNow;
#endif

    rhi::SubmissionPlan submissionPlan;
    std::string submissionPlanError;
    const auto graphicsQueue = m_backend.Queues().GetSnapshot(rhi::QueueRole::Graphics);
    const auto computeQueue = m_backend.Queues().GetSnapshot(rhi::QueueRole::Compute);
    const bool independentCompute = graphicsQueue.queue != VK_NULL_HANDLE && computeQueue.queue != VK_NULL_HANDLE &&
                                    graphicsQueue.nativeLane != UINT32_MAX && computeQueue.nativeLane != UINT32_MAX &&
                                    graphicsQueue.nativeLane != computeQueue.nativeLane;
    const bool partitionedCompute = independentCompute && m_frameAsyncSimulationExecutor &&
                                    m_frameAsyncExportExecutor && m_framePartitionedComputeReady &&
                                    m_framePartitionedComputeReady();
    const bool asyncCompute =
        partitionedCompute && m_frameAsyncComputeReady && m_frameAsyncComputeGeneration && m_frameAsyncComputeReady();
    if (!asyncCompute) {
        // Losing async readiness is a lifecycle boundary (Play/Stop, reset,
        // graph removal, or a temporarily empty scheduler). The next async
        // frame must prime exported state again. Keep the last timeline: it is
        // cheap once signaled and is the only exact dependency on old resident
        // particle resources when readiness returns.
        m_frameAsyncComputePrimed = false;
        m_frameAsyncComputePrimedGeneration = 0;
    }
    const uint64_t asyncComputeGeneration = asyncCompute ? m_frameAsyncComputeGeneration() : 0;
    const bool primeAsyncCompute =
        asyncCompute && (!m_frameAsyncComputePrimed || asyncComputeGeneration != m_frameAsyncComputePrimedGeneration);
    const bool frameComputeHasWork =
        m_frameComputeExecutor && (!m_frameComputeWorkPredicate || m_frameComputeWorkPredicate());
    const bool separateComputeBatch = !asyncCompute && frameComputeHasWork && partitionedCompute;

    m_frameBufferReadDependencies.clear();
    m_frameResidentVertexReads.clear();
    RetirePoseBindings();
    m_frameSubmission.Reset();
    if (presenting && !EnsureGuiRenderGraph(imageIndex))
        throw std::runtime_error("Failed to prepare the presentation render graph");

    const rhi::DeviceId device = m_backend.Device().GetDeviceId();
    std::vector<uint32_t> setupDependencies;
    if (m_framePreSetupBuilder && !m_framePreSetupBuilder(m_frameSubmission, setupDependencies)) {
        throw std::runtime_error("Failed to publish pre-setup queue ownership releases");
    }

    uint32_t particleComputeWork = 0;
    if (primeAsyncCompute || separateComputeBatch) {
        // OwnershipRelease must be submitted before PrimeSimulation waits
        // on it. The previous order queued compute first; when Scene/Game
        // views were hidden that frame, the release never existed and the
        // compute acquire sat on exclusive Graphics-owned buffers.
        const uint32_t primeSimulation = m_frameSubmission.AddWork(
            device, rhi::QueueRole::Compute, rhi::SubmissionDomain::Frame, rhi::InvalidRenderViewId,
            rhi::PipelineStage::ComputeShader, setupDependencies,
            [this]([[maybe_unused]] VkCommandBuffer commandBuffer) {
                return m_frameAsyncSimulationExecutor(commandBuffer);
            },
            "GpuParticle/PrimeSimulation");
        particleComputeWork = m_frameSubmission.AddWork(
            device, rhi::QueueRole::Compute, rhi::SubmissionDomain::Frame, rhi::InvalidRenderViewId,
            rhi::PipelineStage::ComputeShader, {primeSimulation},
            [this](VkCommandBuffer commandBuffer) { return m_frameAsyncExportExecutor(commandBuffer); },
            "GpuParticle/PrimeExport");
        // Play -> Stop -> Play often has preroll, so CanExecuteAsync is
        // false and the old path recorded the entire particle graph into
        // one GpuParticle/Simulation buffer. That erased the compiled
        // sim/export boundary and tripped the 1000 ms frame watchdog.
        // Keep the same two-submission split as async prime.
    } else if (!asyncCompute && frameComputeHasWork) {
        // Small particle workloads stay on Graphics. This is not a CPU
        // fallback: the same GPU compute graph is recorded before the
        // camera graphs, but queue order replaces the expensive
        // previous-frame fence/timeline dependency. It also keeps all
        // particle buffers on one queue family, preserving the validation
        // guarantees that motivated the cross-frame guard.
        particleComputeWork = m_frameSubmission.AddWork(
            device, rhi::QueueRole::Graphics, rhi::SubmissionDomain::Frame, rhi::InvalidRenderViewId,
            rhi::PipelineStage::ComputeShader, setupDependencies,
            [this](VkCommandBuffer commandBuffer) {
                m_frameComputeExecutor(commandBuffer);
                return true;
            },
            "GpuParticle/InlineSimulation");
    }

    // Frame/Setup starts frame accounting. Waiting here for Simulation plus
    // queuing OwnershipRelease after Setup deadlocks the independent
    // compute family when a newly selected emitter first touches exclusive
    // Graphics buffers (Game-only six-way preview).
    const uint32_t setupWork = m_frameSubmission.AddWork(
        device, rhi::QueueRole::Graphics, rhi::SubmissionDomain::Frame, m_presentationView.id,
        rhi::PipelineStage::AllGraphics, std::move(setupDependencies),
        [this](VkCommandBuffer commandBuffer) {
#if INFERNUX_FRAME_PROFILE
            m_gpuTimestampQueries.BeginFrame(commandBuffer, m_currentFrame);
            m_frameTimestampRegion =
                m_gpuTimestampQueries.BeginRegion(commandBuffer, "Frame", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
#endif
            return true;
        },
        "Frame/Setup");

    std::vector<uint32_t> presentationDependencies{setupWork};
    if (m_frameSubmissionBuilder &&
        !m_frameSubmissionBuilder(m_frameSubmission, setupWork, particleComputeWork, presentationDependencies)) {
        throw std::runtime_error("Failed to compose frame RenderGraph submissions");
    }

    uint32_t simulationWork = 0;
    if (asyncCompute && !primeAsyncCompute) {
        // RenderGraphs consume the previous exported generation. Queueing
        // next-frame simulation after their view-local compute batches
        // lets simulation overlap the current Graphics lane.
        simulationWork = m_frameSubmission.AddWork(
            device, rhi::QueueRole::Compute, rhi::SubmissionDomain::Frame, rhi::InvalidRenderViewId,
            rhi::PipelineStage::ComputeShader, {},
            [this](VkCommandBuffer commandBuffer) { return m_frameAsyncSimulationExecutor(commandBuffer); },
            "GpuParticle/AsyncSimulation");
    }

    // Without a window image, skip empty plans while keeping any admitted
    // view, particle, initialization or capture work on the same frame slot.
    if (!presenting && m_frameSubmission.WorkCount() == 1)
        return 0;

#if INFERNUX_FRAME_PROFILE
    const auto finishTimings = [this](VkCommandBuffer commandBuffer) {
        m_gpuTimestampQueries.EndRegion(commandBuffer, m_frameTimestampRegion, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        m_frameTimestampRegion = {};
        m_gpuTimestampQueries.FinishFrame(commandBuffer, m_currentFrame);
        return true;
    };
#else
    const auto finishTimings = [](VkCommandBuffer) { return true; };
#endif
    uint32_t presentationWork = 0;
    const bool captureWindow = output == WindowOutput::DisplayAndCapture;
    if (presenting) {
        const auto guiRange = m_frameSubmission.AppendRenderGraph(
            GetGuiRenderGraph(imageIndex), presentationDependencies, {},
            [this, imageIndex, finishTimings, captureWindow](VkCommandBuffer commandBuffer) {
                return finishTimings(commandBuffer) &&
                       (!captureWindow || RecordPresentationReadback(commandBuffer, imageIndex));
            });
        if (guiRange.Empty())
            throw std::runtime_error("Swapchain GUI RenderGraph produced no submission work");
        presentationWork = guiRange.workItems.back();
    }
#if INFERNUX_FRAME_PROFILE
    else {
        (void)m_frameSubmission.AddWork(device, rhi::QueueRole::Graphics, rhi::SubmissionDomain::Frame,
                                        rhi::InvalidRenderViewId, rhi::PipelineStage::AllGraphics,
                                        presentationDependencies, finishTimings, "Frame/Finalize");
    }
#endif

    if (asyncCompute && !primeAsyncCompute) {
        std::vector<uint32_t> exportDependencies{simulationWork};
        const uint32_t finalGraphics = m_frameSubmission.LastWork(rhi::QueueRole::Graphics);
        if (finalGraphics != 0)
            exportDependencies.push_back(finalGraphics);
        (void)m_frameSubmission.AddWork(
            device, rhi::QueueRole::Compute, rhi::SubmissionDomain::Frame, rhi::InvalidRenderViewId,
            rhi::PipelineStage::ComputeShader, std::move(exportDependencies),
            [this](VkCommandBuffer commandBuffer) { return m_frameAsyncExportExecutor(commandBuffer); },
            "GpuParticle/AsyncExport");
    }

    if (!m_frameSubmission.Build(submissionPlan, submissionPlanError)) {
        throw std::runtime_error("Failed to build frame submission plan: " + submissionPlanError);
    }
    {
        const rhi::SubmissionPlanStatistics statistics = rhi::AnalyzeSubmissionPlan(submissionPlan);
        auto &telemetry = m_frameSubmissionTelemetry;
        ++telemetry.generation;
        telemetry.computeQueueIndependent = independentCompute;
        const auto transferQueue = m_backend.Queues().GetSnapshot(rhi::QueueRole::Transfer);
        telemetry.transferQueueIndependent =
            graphicsQueue.queue != VK_NULL_HANDLE && transferQueue.queue != VK_NULL_HANDLE &&
            graphicsQueue.nativeLane != UINT32_MAX && transferQueue.nativeLane != UINT32_MAX &&
            graphicsQueue.nativeLane != transferQueue.nativeLane;
        telemetry.asyncComputeActive = asyncCompute;
        telemetry.batchCount = statistics.batchCount;
        telemetry.graphicsBatchCount = statistics.graphicsBatchCount;
        telemetry.computeBatchCount = statistics.computeBatchCount;
        telemetry.transferBatchCount = statistics.transferBatchCount;
        telemetry.crossQueueDependencyCount = statistics.crossQueueDependencyCount;
        telemetry.unorderedComputeGraphicsPairCount = statistics.unorderedComputeGraphicsPairCount;
        telemetry.parallelComputeGraphics = independentCompute && statistics.unorderedComputeGraphicsPairCount != 0;
    }
    if (!m_backend.Queues().ResetGraphicsFrameFence(frameSlot)) {
        throw std::runtime_error("Failed to reset graphics frame fence for slot " + std::to_string(frameSlot));
    }

    vk::VulkanSubmissionExecutor::ExternalSync externalSync{};
    std::array<double, 3> executionTimings{};
    externalSync.cpuTimings = phaseTimings ? &executionTimings : nullptr;
    if (presenting) {
        externalSync.imageAvailable = presentation.GetImageAvailableSemaphore(frameSlot);
        externalSync.renderFinished = presentation.GetRenderFinishedSemaphore(imageIndex);
    }
    m_frameSubmissionTelemetry.latestBackgroundComputeSerial =
        m_backend.Compute().LastSubmission().completionTicket.serial;
    m_frameSubmissionTelemetry.residentComputeWriteSerial = 0;
    m_frameSubmissionTelemetry.residentComputeWaitPending = false;
    VkPipelineStageFlags residentReadStages = 0;
    VkAccessFlags residentReadAccess = 0;
    for (const auto &read : m_frameBufferReadDependencies) {
        const auto point = m_backend.Queues().DependencyFor(read.producer, rhi::QueueRole::Graphics);
        if (point.semaphore != VK_NULL_HANDLE)
            externalSync.producers.push_back(
                {point.semaphore, point.value, rhi::ToVkPipelineStages(read.stages), rhi::QueueRole::Graphics});
        else {
            residentReadStages |= rhi::ToVkPipelineStages(
                read.stages & (rhi::PipelineStage::VertexInput | rhi::PipelineStage::VertexShader));
            if ((read.stages & rhi::PipelineStage::VertexInput) != rhi::PipelineStage::None)
                residentReadAccess |= VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
            if ((read.stages & rhi::PipelineStage::VertexShader) != rhi::PipelineStage::None)
                residentReadAccess |= VK_ACCESS_SHADER_READ_BIT;
        }
        // Compute-only diagnostics are not a device-wide resource version.
        if (read.producer.queue == rhi::QueueRole::Compute) {
            m_frameSubmissionTelemetry.residentComputeWriteSerial =
                (std::max)(m_frameSubmissionTelemetry.residentComputeWriteSerial, read.producer.serial);
            m_frameSubmissionTelemetry.residentComputeWaitPending |= point.pending;
        }
    }
    externalSync.completionFence = m_backend.Queues().GetGraphicsFrameFence(frameSlot);
    externalSync.completionEpoch = m_backend.Queues().GetFrameCompletionEpoch(frameSlot);
    if (m_previousFrameCompletion.IsValid() &&
        (asyncCompute || separateComputeBatch || m_previousFrameCompletion.queue != rhi::QueueRole::Graphics)) {
        // The old export can have ended on Compute even when this frame only
        // reads resident particles from Graphics. Carry the actual terminal
        // dependency across changes in simulation and capture readiness.
        externalSync.previousFrameTimeline = m_backend.Queues().GetDependencyTimeline(m_previousFrameCompletion.queue);
        externalSync.previousFrameTimelineValue = m_previousFrameCompletion.serial;
        externalSync.previousFrameStages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        externalSync.previousFrameWaitAtFirstBatch = primeAsyncCompute || separateComputeBatch;
    }

    vk::VulkanSubmissionExecutor::ExecuteResult executeResult{};
    try {
        m_backend.Queues().ReserveWorkCompletions(m_frameSubmission.WorkPublicationCount());
        executeResult = m_submissionExecutor.Execute(
            frameSlot, submissionPlan,
            [this, &submissionPlan, &residentReadStages, &residentReadAccess](uint32_t batchIndex,
                                                                              VkCommandBuffer commandBuffer) {
                if (batchIndex >= submissionPlan.batches.size())
                    return false;
                const auto queue = submissionPlan.batches[batchIndex].queue;
                if (residentReadStages && queue == rhi::QueueRole::Graphics) {
                    // The native lane can alias Compute. Make prior writes visible
                    // to direct vertex reads outside RenderGraph's buffer barriers.
                    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
                    barrier.dstAccessMask = residentReadAccess;
                    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, residentReadStages, 0, 1,
                                         &barrier, 0, nullptr, 0, nullptr);
                    residentReadStages = 0;
                }
                return m_frameSubmission.RecordBatch(submissionPlan, batchIndex, commandBuffer);
            },
            externalSync);
        if (phaseTimings)
            std::copy(executionTimings.begin(), executionTimings.end(), phaseTimings->begin() + 3);
    } catch (...) {
        m_resourceManager.CancelFrameImageReadbacks(externalSync.completionEpoch);
        if (m_frameComputeSubmissionCallback)
            m_frameComputeSubmissionCallback(false);
        if (!m_backend.Queues().AbandonGraphicsFrameSlot(frameSlot))
            INXLOG_ERROR("Failed to restore graphics frame slot after command recording exception");
        throw;
    }
    if (presentationWork && captureWindow && m_presentationReadback) {
        const auto accepted = submissionPlan.batches.begin() + executeResult.submittedBatchCount;
        if (std::any_of(submissionPlan.batches.begin(), accepted, [presentationWork](const auto &batch) {
                return std::find(batch.workItems.begin(), batch.workItems.end(), presentationWork) !=
                       batch.workItems.end();
            })) {
            m_resourceManager.AcceptFrameImageReadback(*m_presentationReadback, externalSync.completionEpoch);
        }
    }
    if (m_frameComputeSubmissionCallback && (executeResult.Succeeded() || !executeResult.submittedBatchCount ||
                                             executeResult.failureDrainResult == VK_SUCCESS))
        m_frameComputeSubmissionCallback(executeResult.Succeeded());
#if INFERNUX_FRAME_PROFILE
    _tNow = Clock::now();
    m_drawSubMs[1] += std::chrono::duration<double, std::milli>(_tNow - _tPrev).count();
    _tPrev = _tNow;
#endif

    if (executeResult.submittedBatchCount)
        m_lastSubmittedCompletionEpoch = externalSync.completionEpoch;
    m_frameSubmission.PublishWorkAccepted(submissionPlan, executeResult.batchTickets,
                                          executeResult.submittedBatchCount);
    if (executeResult.result == VK_ERROR_DEVICE_LOST)
        m_backend.Queues().FailWorkCompletions();
    const VkResult submitResult = executeResult.result;
    if (!executeResult.Succeeded()) {
        m_resourceManager.CancelFrameImageReadbacks(externalSync.completionEpoch);
        if (executeResult.submittedBatchCount && executeResult.failureDrainResult != VK_SUCCESS) {
            auto owners = m_frameSubmission.TakeResourceOwners();
            if (!owners.empty())
                m_deletionQueue.RetireAfter(externalSync.completionEpoch,
                                            [owners = std::move(owners)]() mutable { owners.clear(); });
            // The permanent frame-failure latch prevents Reset/reuse. This
            // epoch stays unsignaled until the strict shutdown owner drains.
        } else if (!m_backend.Queues().AbandonGraphicsFrameSlot(frameSlot)) {
            INXLOG_ERROR("Failed to restore graphics frame fence after submission failure");
        }
        // Earlier queue batches may already have executed. A failed terminal
        // submit is not permission to reset or replay their simulation state.
        throw std::runtime_error(
            std::string("Frame GPU submission failed: ") + vk::VkResultToString(submitResult) +
            (executeResult.submittedBatchCount ? " (work was already submitted)" : " (no work was submitted)"));
    } else {
        // The frame slot epoch was reserved immediately before recording and
        // is completed only by the terminal frame fence. Publish it now so
        // transient resource pools can retire exactly against this submission.
        m_lastSubmittedCompletionEpoch = externalSync.completionEpoch;
        m_frameSubmission.PublishAccepted(executeResult.completionTicket);
        auto owners = m_frameSubmission.TakeResourceOwners();
        if (!owners.empty())
            m_deletionQueue.RetireAfter(externalSync.completionEpoch,
                                        [owners = std::move(owners)]() mutable { owners.clear(); });
        (void)m_backend.Queues().AssociateFrameSlot(frameSlot, executeResult.completionTicket);
        if (asyncCompute)
            m_frameAsyncComputePrimed = true;
        if (asyncCompute)
            m_frameAsyncComputePrimedGeneration = asyncComputeGeneration;
        // Always retain the previous frame's terminal completion point. The
        // next frame may enter async particle execution after an edit/play or
        // graph-readiness transition, while this frame's Graphics lane still
        // consumes the same exported indirect buffers.
        m_previousFrameCompletion = executeResult.completionTicket;
    }
#if INFERNUX_FRAME_PROFILE
    if (submitResult == VK_SUCCESS) {
        m_gpuTimestampQueries.MarkSubmitted(m_currentFrame);
    }
#endif
#if INFERNUX_FRAME_PROFILE
    _tNow = Clock::now();
    m_drawSubMs[2] += std::chrono::duration<double, std::milli>(_tNow - _tPrev).count();
    _tPrev = _tNow;
#endif

    if (phaseTimings) {
        const auto now = std::chrono::steady_clock::now();
        (*phaseTimings)[1] = std::chrono::duration<double, std::milli>(now - phaseStart).count();
        phaseStart = now;
    }
    uint64_t presentationAcceptedNs = 0;
    if (presenting) {
        const auto result = presentation.Present(m_backend.Queues(), imageIndex);
        if (result == vk::SwapchainResult::Success) {
            const auto acceptedAt = std::chrono::steady_clock::now().time_since_epoch();
            presentationAcceptedNs =
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(acceptedAt).count());
            if (m_firstVisiblePresentationPending) {
                m_firstVisiblePresentationPending = false;
                m_firstVisiblePresentationCompleted = true;
                SDL_Log("INFERNUX_FIRST_REVEAL_PRESENT");
            }
        }
        if (result == vk::SwapchainResult::SurfaceLost) {
            m_presentationSurfaceLost = true;
            INXLOG_WARN("Platform presentation surface was lost while presenting; scheduling a surface rebind");
        } else if (result == vk::SwapchainResult::NeedRecreate) {
            m_framebufferResized = true;
        } else if (result == vk::SwapchainResult::Error) {
            throw std::runtime_error("Failed to present the submitted frame");
        }
    }
#if INFERNUX_FRAME_PROFILE
    _tNow = Clock::now();
    m_drawSubMs[3] += std::chrono::duration<double, std::milli>(_tNow - _tPrev).count();

    ++m_drawSubCount;
#endif

    // Advance frame
    m_currentFrame = (m_currentFrame + 1) % m_maxFramesInFlight;
    if (phaseTimings && presenting)
        (*phaseTimings)[2] =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - phaseStart).count();
    return presentationAcceptedNs;
} catch (...) {
    m_frameSubmission.DiscardPublications();
    if (!m_frameFailure)
        m_frameFailure = std::current_exception();
    throw;
}

void InxVkCoreModular::InvalidateDrawBufferCaches() noexcept
{
    ++m_drawListActivation;
    m_staticInstanceRanges.clear();
    m_staticFilteredListCaches.clear();
    m_shadowScratchValid = false;
}

void InxVkCoreModular::SetDrawCalls(const std::vector<DrawCall> *drawCalls, bool forceRefresh)
{
    if (!forceRefresh && m_drawCallsPtr == drawCalls && m_drawListMetadataSource == drawCalls &&
        m_drawListBufferRevision == m_objectBufferRevision &&
        m_drawListRenderMetaRevision == m_shaderCache.GetRenderMetaRevision() &&
        m_drawListMaterialRoutingRevision == InxMaterial::GetRoutingPublicationRevision() &&
        (!drawCalls || m_drawListMetadata.size() == drawCalls->size()))
        return;

    InvalidateDrawBufferCaches();
    m_drawCallsPtr = drawCalls;
    m_drawListMetadataSource = drawCalls;
    m_drawListBufferRevision = 0;
    m_drawListRenderMetaRevision = m_shaderCache.GetRenderMetaRevision();
    m_drawListMetadata.clear();
    m_drawListMaterialRoutingRevision = InxMaterial::GetRoutingPublicationRevision();
    m_skyboxDrawListSource = drawCalls;
    m_skyboxDrawCallIndices.clear();

    // Refresh cached builtin materials (avoids string-hash lookup per DrawSceneFiltered call)
    if (!m_cachedDefaultLit) {
        m_cachedDefaultLit = AssetRegistry::Instance().GetBuiltinMaterial("DefaultLit");
    }

    // Track only the small unique queue set. Sorting one queue value per
    // DrawCall costs more than the empty render-pass scans this replaces.
    m_drawQueueValues.clear();
    m_drawQueueValuesOverflow = false;
    if (!drawCalls)
        return;
    constexpr size_t kTrackedQueueLimit = 16;
    m_drawListMetadata.reserve(drawCalls->size());
    m_skyboxDrawCallIndices.reserve(1);
    m_drawQueueValues.reserve(kTrackedQueueLimit);
    std::unordered_set<InxMaterial *> preparedMaterials;
    for (size_t drawCallIndex = 0; drawCallIndex < drawCalls->size(); ++drawCallIndex) {
        const DrawCall &drawCall = (*drawCalls)[drawCallIndex];
        const auto &owner = drawCall.material ? drawCall.material : m_cachedDefaultLit;
        InxMaterial *material = owner.get();
        // Queue and pass-tag filtering must see the shader's complete defaults
        // before any pass can initialize its GPU pipeline lazily.
        if (material && preparedMaterials.insert(material).second)
            PrepareMaterialRenderState(owner);
        const int queue = material ? material->GetRenderQueue() : 2000;
        if (drawCall.identity.domain == RenderDomain::Skybox)
            m_skyboxDrawCallIndices.push_back(drawCallIndex);
        const auto bufferIt = m_perObjectBuffers.find(drawCall.objectId);
        if (bufferIt != m_perObjectBuffers.end()) {
            m_drawListMetadata.push_back({drawCall.objectId, material, queue, bufferIt->second.vertexBuffer,
                                          bufferIt->second.indexBuffer, bufferIt->second.indexCount,
                                          bufferIt->second.residentVertexHandle, bufferIt->second.indexFormat});
        } else {
            m_drawListMetadata.push_back({drawCall.objectId, material, queue, {}, {}, 0});
        }
        if (!material)
            continue;
        if (m_drawQueueValuesOverflow)
            continue;
        if (std::find(m_drawQueueValues.begin(), m_drawQueueValues.end(), queue) != m_drawQueueValues.end())
            continue;
        if (m_drawQueueValues.size() == kTrackedQueueLimit) {
            m_drawQueueValuesOverflow = true;
            m_drawQueueValues.clear();
            // Keep building per-draw metadata after the small queue-summary
            // optimization overflows; DrawSceneFiltered still benefits from
            // the cached buffer entries on large/mixed queue lists.
            continue;
        }
        m_drawQueueValues.push_back(queue);
    }
    m_drawListBufferRevision = m_objectBufferRevision;
    m_drawListRenderMetaRevision = m_shaderCache.GetRenderMetaRevision();
    m_drawListMaterialRoutingRevision = InxMaterial::GetRoutingPublicationRevision();
}

void InxVkCoreModular::SetShadowDrawCalls(const std::vector<DrawCall> *drawCalls, bool forceRefresh)
{
    if (!forceRefresh && m_shadowDrawCallsPtr == drawCalls && m_shadowListMetadataSource == drawCalls &&
        m_shadowListBufferRevision == m_objectBufferRevision &&
        m_shadowListRenderMetaRevision == m_shaderCache.GetRenderMetaRevision() &&
        m_shadowListMaterialRoutingRevision == InxMaterial::GetRoutingPublicationRevision() &&
        (!drawCalls || m_shadowListMetadata.size() == drawCalls->size()))
        return;

    m_shadowDrawCallsPtr = drawCalls;
    m_shadowScratchValid = false;
    m_shadowListMetadataSource = drawCalls;
    m_shadowListBufferRevision = 0;
    m_shadowListRenderMetaRevision = m_shaderCache.GetRenderMetaRevision();
    m_shadowListMetadata.clear();
    m_shadowListMaterialRoutingRevision = InxMaterial::GetRoutingPublicationRevision();
    if (!drawCalls)
        return;

    m_shadowListMetadata.reserve(drawCalls->size());
    std::unordered_set<InxMaterial *> preparedMaterials;
    for (const DrawCall &drawCall : *drawCalls) {
        const auto &owner = drawCall.material ? drawCall.material : m_cachedDefaultLit;
        InxMaterial *material = owner.get();
        if (material && preparedMaterials.insert(material).second)
            PrepareMaterialRenderState(owner);
        const auto bufferIt = m_perObjectBuffers.find(drawCall.objectId);
        if (bufferIt != m_perObjectBuffers.end()) {
            m_shadowListMetadata.push_back({drawCall.objectId, material, material ? material->GetRenderQueue() : 2000,
                                            bufferIt->second.vertexBuffer, bufferIt->second.indexBuffer,
                                            bufferIt->second.indexCount, bufferIt->second.residentVertexHandle,
                                            bufferIt->second.indexFormat});
        } else {
            m_shadowListMetadata.push_back(
                {drawCall.objectId, material, material ? material->GetRenderQueue() : 2000, {}, {}, 0});
        }
    }
    m_shadowListBufferRevision = m_objectBufferRevision;
    m_shadowListRenderMetaRevision = m_shaderCache.GetRenderMetaRevision();
    m_shadowListMaterialRoutingRevision = InxMaterial::GetRoutingPublicationRevision();
}

void InxVkCoreModular::ReleaseActiveDrawLists() noexcept
{
    m_drawCallsPtr = nullptr;
    m_shadowDrawCallsPtr = nullptr;
    m_drawListMetadataSource = nullptr;
    m_shadowListMetadataSource = nullptr;
    m_skyboxDrawListSource = nullptr;
    m_drawListBufferRevision = 0;
    m_shadowListBufferRevision = 0;

    m_drawListMetadata.clear();
    m_shadowListMetadata.clear();
    m_skyboxDrawCallIndices.clear();
    m_drawQueueValues.clear();
    m_drawQueueValuesOverflow = false;
    m_staticInstanceRanges.clear();
    m_staticFilteredListCaches.clear();
    m_staticInstanceRangeFrame = UINT64_MAX;
    m_drawListActivation = 0;

    m_eligibleScratch.clear();
    m_shadowDrawScratch.clear();
    m_shadowCullGroups.clear();
    m_shadowAllVisible.clear();
    m_shadowScratchValid = false;
    m_shadowScratchStatic = false;
    m_shadowScratchUniformBatch = false;
    m_shadowViewVisible.clear();
    m_resolvedShadowMaterialsScratch.clear();
    m_cachedDefaultLit.reset();
}

// ============================================================================
// Filtered draw — renders only draw calls within a queue range
// ============================================================================

void InxVkCoreModular::DrawSceneFiltered(
    VkCommandBuffer cmdBuf, uint32_t width, uint32_t height, rhi::BindGroupHandle perViewGroup,
    vk::ViewGlobalsBinding &viewGlobals, const glm::mat4 &viewMatrix, int queueMin, int queueMax,
    const std::string &sortMode, const std::string &overrideMaterial, const std::string &passTag,
    const MaterialPassPipelineDescriptor *pipelineDescriptor, GraphMaterialFilter materialFilter,
    const RendererSelection *selection, uint64_t excludedObjectId, bool requireSourceDepthWrite,
    RenderInstanceHistory *motionHistory, const std::shared_ptr<InxMaterial> &overridePublication)
{
    const MaterialPassPipelineDescriptor activePass =
        pipelineDescriptor ? *pipelineDescriptor : m_materialPipelineManager.GetDefaultPassPipelineDescriptor();
    if (!activePass.IsValid()) {
        INXLOG_ERROR("DrawSceneFiltered received an invalid ", ShaderCompileTargetName(activePass.target),
                     " pass pipeline descriptor");
        return;
    }
    const VkDescriptorSet perViewDescriptorSet = m_backend.Device().GetRhiDevice().Resolve(perViewGroup);
    if (perViewDescriptorSet == VK_NULL_HANDLE) {
        INXLOG_ERROR("DrawSceneFiltered received an invalid per-view bind group");
        return;
    }
    // One-shot diagnostic: log queue-range filtering for first N frames
    static int s_filterDiagFrames = 0;

#if INFERNUX_FRAME_PROFILE
    using Clock = std::chrono::high_resolution_clock;
    const auto totalStart = Clock::now();
    auto stageStart = totalStart;
#endif

    // Fast early-out when no draw calls are staged
    if (drawCalls().empty() || (selection && selection->Size() == 0))
        return;

    if (!m_drawQueueValuesOverflow) {
        const bool queuePresent =
            std::any_of(m_drawQueueValues.begin(), m_drawQueueValues.end(),
                        [queueMin, queueMax](int queue) { return queue >= queueMin && queue <= queueMax; });
        if (!queuePresent)
            return;
    }
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

    const auto &defaultMaterial = m_cachedDefaultLit;
    if (!defaultMaterial) {
        return;
    }

    // Resolve override material (if specified)
    InxMaterial *overrideMatRaw = nullptr;
    std::shared_ptr<InxMaterial> overrideMatOwner; // keeps alive during this scope
    if (selection) {
        overrideMatOwner = selection->Material();
        overrideMatRaw = overrideMatOwner.get();
    } else if (overridePublication) {
        overrideMatOwner = overridePublication;
        overrideMatRaw = overrideMatOwner.get();
    } else if (!overrideMaterial.empty()) {
        auto &registry = AssetRegistry::Instance();
        overrideMatOwner = registry.GetBuiltinMaterial(overrideMaterial);
        if (!overrideMatOwner)
            overrideMatOwner = registry.LoadAsset<InxMaterial>(overrideMaterial, ResourceType::Material);
        overrideMatRaw = overrideMatOwner.get();
    }

    InxMaterial *defaultMatRaw = defaultMaterial.get();

    // ---- Collect eligible draw calls (queue filter + frustum cull) ----
    m_eligibleScratch.clear();

    const auto &activeDrawCalls = drawCalls();
    // SceneRenderGraph marks its skybox callback explicitly. Do not infer this
    // from a queue number: user-authored pipelines are allowed to reuse any
    // queue value.
    const bool skyboxPass = passTag == "__infernux_internal_skybox";
    const uint64_t materialPublicationGeneration = m_materialPipelineManager.GetPublicationGeneration();
    auto staticFilterCache = m_staticFilteredListCaches.end();
    if (!selection && excludedObjectId == 0 && !requireSourceDepthWrite && overrideMaterial.empty() &&
        sortMode != "back_to_front") {
        staticFilterCache =
            std::find_if(m_staticFilteredListCaches.begin(), m_staticFilteredListCaches.end(),
                         [&](const StaticFilteredListCache &cache) {
                             return cache.drawListActivation == m_drawListActivation &&
                                    cache.materialPublicationGeneration == materialPublicationGeneration &&
                                    cache.queueMin == queueMin && cache.queueMax == queueMax &&
                                    cache.target == activePass.target && cache.materialFilter == materialFilter &&
                                    cache.sortMode == sortMode && cache.passTag == passTag;
                         });
    }
    const bool reusedStaticFilter = staticFilterCache != m_staticFilteredListCaches.end();
    uint64_t staticFilteredSequenceKey = reusedStaticFilter ? staticFilterCache->sequenceKey : 0;
    const bool hasListMetadata = m_drawListMetadataSource == m_drawCallsPtr &&
                                 m_drawListMetadata.size() == activeDrawCalls.size() &&
                                 m_drawListBufferRevision == m_objectBufferRevision;
    // This is only populated when an unusual caller stages draw calls before
    // their object buffers become available. It preserves ownership for the
    // remainder of this recording without adding shared_ptr traffic to the
    // normal per-pass loop.
    std::vector<DrawListMetadata> fallbackBufferLeases;
    if (!hasListMetadata)
        fallbackBufferLeases.reserve(activeDrawCalls.size());
    const bool hasCachedSkyboxIndices =
        skyboxPass && m_skyboxDrawListSource == m_drawCallsPtr && !m_skyboxDrawCallIndices.empty();
    const size_t candidateCount = hasCachedSkyboxIndices ? m_skyboxDrawCallIndices.size() : activeDrawCalls.size();
    size_t filterCandidateCount = 0;
    for (size_t candidateIndex = 0; !reusedStaticFilter && candidateIndex < candidateCount; ++candidateIndex) {
        const size_t drawCallIndex = hasCachedSkyboxIndices ? m_skyboxDrawCallIndices[candidateIndex] : candidateIndex;
        ++filterCandidateCount;
        const DrawCall &dc = activeDrawCalls[drawCallIndex];
        if (excludedObjectId != 0 && dc.objectId == excludedObjectId)
            continue;
        const auto *parameters =
            selection ? selection->Find(dc.identity) : (overrideMatRaw ? nullptr : &dc.parameterBlock);
        if (selection && !parameters)
            continue;
        const DrawListMetadata *metadata = hasListMetadata ? &m_drawListMetadata[drawCallIndex] : nullptr;
        const InxMaterial *expectedMaterial = dc.material ? dc.material.get() : m_cachedDefaultLit.get();
        const uint64_t requiredIndexEnd = static_cast<uint64_t>(dc.indexStart) + dc.indexCount;
        if (metadata && (metadata->objectId != dc.objectId || metadata->material != expectedMaterial))
            metadata = nullptr;
        // The lease can still hold the previous published geometry of an
        // every-frame dynamic mesh whose fresh upload has not been published
        // yet (async/fence transfer). A whole-buffer draw (inline meshes use
        // indexStart==0, vertexStart==0) can safely present the complete
        // previous geometry for that frame; dropping the object instead makes
        // moving LineRenderer trails flicker on every growth frame.
        uint32_t indexCountClamp = 0;
        if (metadata && requiredIndexEnd > metadata->indexCapacity) {
            if (dc.indexStart == 0 && dc.vertexStart == 0 && metadata->indexCapacity > 0 &&
                metadata->HasVertexBuffer() && metadata->indexBuffer) {
                indexCountClamp = static_cast<uint32_t>(metadata->indexCapacity);
            } else {
                metadata = nullptr;
            }
        }
        if (!dc.frustumVisible)
            continue;
        if (skyboxPass != (dc.identity.domain == RenderDomain::Skybox))
            continue;

        const std::shared_ptr<InxMaterial> *materialOwner =
            overrideMatOwner ? &overrideMatOwner : (dc.material ? &dc.material : &defaultMaterial);
        InxMaterial *material =
            overrideMatRaw ? overrideMatRaw : (metadata ? metadata->material : materialOwner->get());
        if (!material)
            continue;
        // The semantic Depth target forces depth-write on. During alternate
        // scene-depth replay, exclude sources whose original Forward pipeline
        // only wrote color, or they would become new (incorrect) occluders.
        if (requireSourceDepthWrite && !material->GetRenderState().depthWriteEnable)
            continue;

        // Filtering describes which source renderers participate in the pass.
        // An override material only changes how an accepted renderer is drawn;
        // it must not replace the source queue or pass tag used for selection.
        const InxMaterial *sourceMaterial = metadata ? metadata->material : expectedMaterial;
        const int queue = metadata ? metadata->renderQueue
                                   : (sourceMaterial ? sourceMaterial->GetRenderQueue() : material->GetRenderQueue());
        if (!skyboxPass && (queue < queueMin || queue > queueMax))
            continue;

        if (materialFilter != GraphMaterialFilter::All) {
            const auto *committed = ResolveMeshMaterial(*materialOwner);
            const auto *artifact =
                committed ? m_shaderCache.FindProgramArtifact(committed->programKey.stages) : nullptr;
            const bool deferredCompatible = artifact && artifact->FindVariant(ShaderCompileTarget::GBuffer);
            if ((materialFilter == GraphMaterialFilter::DeferredCompatible && !deferredCompatible) ||
                (materialFilter == GraphMaterialFilter::DeferredUnsupported && deferredCompatible))
                continue;
        }

        // Pass tag filter. Override passes are selectors, so an untagged source
        // must not leak into a tagged outline/depth/etc. pass. Legacy regular
        // passes retain the empty-source wildcard behavior.
        if (!passTag.empty() && !skyboxPass) {
            const std::string &matTag = sourceMaterial ? sourceMaterial->GetPassTag() : material->GetPassTag();
            const bool strictSourceTag = overrideMatRaw != nullptr;
            if ((strictSourceTag && matTag != passTag) || (!strictSourceTag && !matTag.empty() && matTag != passTag))
                continue;
        }

        // Compute view-space depth for transparent sort only.
        // Opaque front_to_back groups by material hash + vertex buffer
        // (stable order), so depth sort is unnecessary and its O(N log N)
        // cost every frame is avoided via the is_sorted() early-out.
        float sortKey = 0.0f;
        if (sortMode == "back_to_front") {
            if (dc.devicePose.value) {
                if (m_reportedPoseTransparency.insert(material->GetMaterialKey()).second)
                    INXLOG_ERROR("Material '", material->GetName(),
                                 "' cannot draw a device pose with CPU transparency sorting; this draw is suspended");
                continue;
            }
            glm::vec4 viewPos = viewMatrix * glm::vec4(glm::vec3(dc.worldMatrix[3]), 1.0f);
            sortKey = viewPos.z;
        }

        // Material + mesh hash for grouping optimization
        size_t matHash = std::hash<void *>{}(static_cast<void *>(material));
        if (dc.devicePose.value) {
            const auto poseHash = std::hash<const void *>{}(dc.devicePose.value.get());
            matHash ^= poseHash + 0x9e3779b9u + (matHash << 6u) + (matHash >> 2u);
        }
        if (parameters && *parameters) {
            const size_t parameterHash = std::hash<const void *>{}(parameters->get());
            matHash ^= parameterHash + 0x9e3779b9u + (matHash << 6u) + (matHash >> 2u);
        }
        const DrawListMetadata *bufferLease =
            metadata && metadata->HasVertexBuffer() && metadata->indexBuffer ? metadata : nullptr;
        if (!bufferLease) {
            const auto bufferIt = m_perObjectBuffers.find(dc.objectId);
            if (bufferIt != m_perObjectBuffers.end() && bufferIt->second.HasVertexBuffer() &&
                bufferIt->second.indexBuffer) {
                if (requiredIndexEnd <= bufferIt->second.indexCount) {
                    fallbackBufferLeases.push_back({dc.objectId, material, queue, bufferIt->second.vertexBuffer,
                                                    bufferIt->second.indexBuffer, bufferIt->second.indexCount,
                                                    bufferIt->second.residentVertexHandle,
                                                    bufferIt->second.indexFormat});
                    bufferLease = &fallbackBufferLeases.back();
                } else if (dc.indexStart == 0 && dc.vertexStart == 0 && bufferIt->second.indexCount > 0) {
                    // Same stale-lease fallback as the metadata path above.
                    fallbackBufferLeases.push_back({dc.objectId, material, queue, bufferIt->second.vertexBuffer,
                                                    bufferIt->second.indexBuffer, bufferIt->second.indexCount,
                                                    bufferIt->second.residentVertexHandle,
                                                    bufferIt->second.indexFormat});
                    bufferLease = &fallbackBufferLeases.back();
                    indexCountClamp = static_cast<uint32_t>(bufferIt->second.indexCount);
                }
            }
        }
        const VkBuffer vb = bufferLease ? bufferLease->GetVertexHandle() : VK_NULL_HANDLE;
        const VkBuffer ib = bufferLease ? bufferLease->indexBuffer->GetBuffer() : VK_NULL_HANDLE;

        m_eligibleScratch.push_back(
            {&dc, sortKey, matHash, vb, ib, materialOwner, material, indexCountClamp, parameters,
             bufferLease ? bufferLease->indexFormat : MeshIndexFormat::UInt32,
             bufferLease ? bufferLease->GetVertexUpload() : rhi::SubmissionTicket{},
             bufferLease ? bufferLease->indexBuffer->GetUploadSubmission() : rhi::SubmissionTicket{}});
    }

    // A stable static publication already owns a sorted, fully resolved list.
    // Read it in place instead of copying tens of thousands of entries back
    // into the frame scratch on every pass. The scratch remains the mutable
    // source while a cache is built for the first time.
    const std::vector<SortableDrawCall> &eligibleDraws =
        reusedStaticFilter ? staticFilterCache->draws : m_eligibleScratch;

    // Diagnostic: log per-call eligible count with queue range
    if (s_filterDiagFrames < 3) {
        INXLOG_DEBUG("[DrawSceneFiltered] queue=[", queueMin, ",", queueMax, "] totalDC=", drawCalls().size(),
                     " eligible=", eligibleDraws.size());
        ++s_filterDiagFrames;
    }

#if INFERNUX_FRAME_PROFILE
    auto stageNow = Clock::now();
    m_drawSubMs[6] += std::chrono::duration<double, std::milli>(stageNow - stageStart).count();
    stageStart = stageNow;
    m_drawSceneFilteredEligible += static_cast<uint64_t>(eligibleDraws.size());
#endif

    if (eligibleDraws.empty()) {
        if (skyboxPass) {
            static int s_emptySkyboxDiagCount = 0;
            if (s_emptySkyboxDiagCount < 8) {
                INXLOG_DEBUG("[DrawSceneFiltered] SkyboxPass candidates=", filterCandidateCount,
                             " eligible=0 materialResolve=0 materialCacheHits=0 materialUpdates=0 issued=0");
                ++s_emptySkyboxDiagCount;
            }
        }
#if INFERNUX_FRAME_PROFILE
        m_drawSubMs[5] += std::chrono::duration<double, std::milli>(Clock::now() - totalStart).count();
#endif
        return;
    }

    // ---- Sort if requested (skip for 0-1 elements) ----
    // Uniform-batch fast path: when every eligible entry shares the same
    // material hash and vertex/index buffer pair, all entries will be emitted as a
    // single instanced draw regardless of ordering.  Sorting would only
    // permute elements within that single batch, so we skip it entirely.
    bool uniformBatch = reusedStaticFilter && staticFilterCache->uniformBatch;
    if (!reusedStaticFilter && m_eligibleScratch.size() > 1) {
        const size_t firstMatHash = m_eligibleScratch[0].materialHash;
        const InxMaterial *firstMaterial = m_eligibleScratch[0].material;
        const VkBuffer firstVB = m_eligibleScratch[0].vertexBuf;
        const VkBuffer firstIB = m_eligibleScratch[0].indexBuf;
        const DrawCall &firstDraw = *m_eligibleScratch[0].dc;
        uniformBatch = true;
        const uint32_t firstIndexCount =
            m_eligibleScratch[0].indexCountClamp ? m_eligibleScratch[0].indexCountClamp : firstDraw.indexCount;
        for (size_t i = 1; i < m_eligibleScratch.size(); ++i) {
            const DrawCall &draw = *m_eligibleScratch[i].dc;
            const uint32_t drawIndexCount =
                m_eligibleScratch[i].indexCountClamp ? m_eligibleScratch[i].indexCountClamp : draw.indexCount;
            if (m_eligibleScratch[i].materialHash != firstMatHash || m_eligibleScratch[i].material != firstMaterial ||
                m_eligibleScratch[i].ParameterIdentity() != m_eligibleScratch[0].ParameterIdentity() ||
                m_eligibleScratch[i].vertexBuf != firstVB || m_eligibleScratch[i].indexBuf != firstIB ||
                m_eligibleScratch[i].indexFormat != m_eligibleScratch[0].indexFormat ||
                draw.indexStart != firstDraw.indexStart || drawIndexCount != firstIndexCount ||
                draw.vertexStart != firstDraw.vertexStart) {
                uniformBatch = false;
                break;
            }
        }
    }

    // is_sorted() early-out: O(N) comparison-only scan avoids the O(N log N)
    // std::sort when the eligible scratch is already in correct order from
    // a previous frame (common in stable scenes with static camera).
    const bool preserveOrder = sortMode.empty() || sortMode == "none" || sortMode == "preserve";
    // Transparent entries still need depth sorting even when they form one
    // instance batch. The sorted matrix stream determines blend order.
    const bool skipUniformBatchSort = uniformBatch && sortMode != "back_to_front";
    if (m_eligibleScratch.size() > 1 && !skipUniformBatchSort && !preserveOrder) {
        // In left-handed view space: near objects have small positive Z, far
        // objects have larger positive Z.
        if (sortMode == "front_to_back") {
            // Group by material + vertex/index buffer pair only (no depth).
            // This order is stable across frames for static material assignments,
            // so is_sorted() returns true and std::sort is skipped entirely.
            auto cmp = [](const SortableDrawCall &a, const SortableDrawCall &b) {
                if (a.materialHash != b.materialHash)
                    return a.materialHash < b.materialHash;
                if (a.vertexBuf != b.vertexBuf)
                    return a.vertexBuf < b.vertexBuf;
                if (a.indexBuf != b.indexBuf)
                    return a.indexBuf < b.indexBuf;
                return static_cast<uint32_t>(a.indexFormat) < static_cast<uint32_t>(b.indexFormat);
            };
            if (!std::is_sorted(m_eligibleScratch.begin(), m_eligibleScratch.end(), cmp)) {
                std::sort(m_eligibleScratch.begin(), m_eligibleScratch.end(), cmp);
            }
        } else if (sortMode == "back_to_front") {
            auto cmp = [](const SortableDrawCall &a, const SortableDrawCall &b) { return a.sortKey > b.sortKey; };
            if (!std::is_sorted(m_eligibleScratch.begin(), m_eligibleScratch.end(), cmp)) {
                std::sort(m_eligibleScratch.begin(), m_eligibleScratch.end(), cmp);
            }
        } else {
            auto cmp = [](const SortableDrawCall &a, const SortableDrawCall &b) {
                if (a.materialHash != b.materialHash)
                    return a.materialHash < b.materialHash;
                if (a.vertexBuf != b.vertexBuf)
                    return a.vertexBuf < b.vertexBuf;
                if (a.indexBuf != b.indexBuf)
                    return a.indexBuf < b.indexBuf;
                return static_cast<uint32_t>(a.indexFormat) < static_cast<uint32_t>(b.indexFormat);
            };
            if (!std::is_sorted(m_eligibleScratch.begin(), m_eligibleScratch.end(), cmp)) {
                std::sort(m_eligibleScratch.begin(), m_eligibleScratch.end(), cmp);
            }
        }
    } // size() > 1

    if (!reusedStaticFilter && !selection && overrideMaterial.empty() && sortMode != "back_to_front" &&
        m_eligibleScratch.size() >= 256 &&
        std::all_of(m_eligibleScratch.begin(), m_eligibleScratch.end(), [](const SortableDrawCall &entry) {
            return entry.dc && entry.dc->isStatic && !entry.dc->skinBoneMatrices;
        })) {
        StaticFilteredListCache cache;
        cache.drawListActivation = m_drawListActivation;
        cache.materialPublicationGeneration = materialPublicationGeneration;
        cache.queueMin = queueMin;
        cache.queueMax = queueMax;
        cache.target = activePass.target;
        cache.materialFilter = materialFilter;
        cache.sortMode = sortMode;
        cache.passTag = passTag;
        cache.draws = m_eligibleScratch;
        cache.sequenceKey = m_nextStaticFilteredSequenceKey++;
        if (m_nextStaticFilteredSequenceKey == 0)
            m_nextStaticFilteredSequenceKey = 1;
        staticFilteredSequenceKey = cache.sequenceKey;
        cache.uniformBatch = uniformBatch;
        constexpr size_t kMaximumStaticFilterCaches = 8;
        if (m_staticFilteredListCaches.size() == kMaximumStaticFilterCaches)
            m_staticFilteredListCaches.erase(m_staticFilteredListCaches.begin());
        m_staticFilteredListCaches.push_back(std::move(cache));
    }

#if INFERNUX_FRAME_PROFILE
    stageNow = Clock::now();
    m_drawSubMs[7] += std::chrono::duration<double, std::milli>(stageNow - stageStart).count();
    stageStart = stageNow;
#endif

    // ---- Upload instance model matrices to SSBO (set 2, binding 1) ----
    ResetPerFrameGpuStreamOffsets();

    const uint32_t frameIndex = m_currentFrame % m_maxFramesInFlight;
    const size_t totalEligible = eligibleDraws.size();
    uint32_t writeBase = m_instanceWriteOffset;
    const bool needsInstanceAuxiliary = ShaderCompileTargetUsesInstanceAuxiliary(activePass.target);
    // InstanceAuxBuffer contents are just as immutable as model matrices for a
    // static, unskinned publication. Keep one uploaded copy in every
    // frame-in-flight buffer instead of excluding Forward/Forward+/GBuffer
    // from the static range cache merely because they consume object identity,
    // layer mask, or transform history.
    const bool staticUnskinnedSequence = staticFilteredSequenceKey != 0 && !motionHistory;
    bool reusedStaticRange = false;
    if (staticUnskinnedSequence) {
        for (const StaticInstanceRange &range : m_staticInstanceRanges) {
            if (range.drawListActivation != m_drawListActivation || range.sequenceKey != staticFilteredSequenceKey ||
                range.frameIndex != frameIndex || range.instanceCount != totalEligible)
                continue;
            const VkBuffer activeInstanceBuffer =
                frameIndex < m_instanceBuffers.size() && m_instanceBuffers[frameIndex].buffer
                    ? m_instanceBuffers[frameIndex].buffer->GetBuffer()
                    : VK_NULL_HANDLE;
            const VkBuffer activeSkinInstanceBuffer =
                frameIndex < m_skinInstanceBuffers.size() && m_skinInstanceBuffers[frameIndex].buffer
                    ? m_skinInstanceBuffers[frameIndex].buffer->GetBuffer()
                    : VK_NULL_HANDLE;
            const VkBuffer activeInstanceAuxBuffer =
                frameIndex < m_instanceAuxBuffers.size() && m_instanceAuxBuffers[frameIndex].buffer
                    ? m_instanceAuxBuffers[frameIndex].buffer->GetBuffer()
                    : VK_NULL_HANDLE;
            if (range.instanceBuffer != activeInstanceBuffer || range.skinInstanceBuffer != activeSkinInstanceBuffer)
                continue;
            if (needsInstanceAuxiliary && range.instanceAuxBuffer != activeInstanceAuxBuffer)
                continue;
            writeBase = range.firstInstance;
            reusedStaticRange = true;
            break;
        }
    }

    if (needsInstanceAuxiliary)
        PrepareInstanceAuxiliary(writeBase + totalEligible);

    if (!reusedStaticRange && totalEligible > 0 && frameIndex < m_instanceBuffers.size()) {
        size_t requiredBoneMatrices = m_skinPaletteWriteOffset;
        if (motionHistory) {
            m_previousMotionScratch.clear();
            m_previousMotionScratch.reserve(totalEligible);
        }
        for (const auto &entry : eligibleDraws) {
            const auto *palette = entry.dc->skinBoneMatrices.get();
            const auto boneCount = palette ? palette->size() : 0;
            requiredBoneMatrices += boneCount;
            if (motionHistory) {
                const auto *previous =
                    motionHistory->Previous(entry.dc->identity, entry.dc->skinContinuity, entry.dc->devicePose);
                m_previousMotionScratch.push_back(previous);
                if (previous && previous->skin.get() != palette)
                    requiredBoneMatrices += boneCount;
            }
        }
        const VkBuffer previousInstanceBuffer =
            m_instanceBuffers[frameIndex].buffer ? m_instanceBuffers[frameIndex].buffer->GetBuffer() : VK_NULL_HANDLE;
        const VkBuffer previousSkinInstanceBuffer =
            frameIndex < m_skinInstanceBuffers.size() && m_skinInstanceBuffers[frameIndex].buffer
                ? m_skinInstanceBuffers[frameIndex].buffer->GetBuffer()
                : VK_NULL_HANDLE;
        const VkBuffer previousSkinPaletteBuffer =
            frameIndex < m_skinPaletteBuffers.size() && m_skinPaletteBuffers[frameIndex].buffer
                ? m_skinPaletteBuffers[frameIndex].buffer->GetBuffer()
                : VK_NULL_HANDLE;

        EnsureInstanceBufferCapacity(frameIndex, writeBase + totalEligible);
        EnsureSkinBuffersCapacity(frameIndex, writeBase + totalEligible, requiredBoneMatrices);
        const bool instanceBufferChanged = m_instanceBuffers[frameIndex].buffer &&
                                           previousInstanceBuffer != m_instanceBuffers[frameIndex].buffer->GetBuffer();
        const bool skinBufferChanged =
            frameIndex < m_skinInstanceBuffers.size() && frameIndex < m_skinPaletteBuffers.size() &&
            ((m_skinInstanceBuffers[frameIndex].buffer &&
              previousSkinInstanceBuffer != m_skinInstanceBuffers[frameIndex].buffer->GetBuffer()) ||
             (m_skinPaletteBuffers[frameIndex].buffer &&
              previousSkinPaletteBuffer != m_skinPaletteBuffers[frameIndex].buffer->GetBuffer()));
        if (instanceBufferChanged || skinBufferChanged)
            (void)PublishGlobalsDescriptorRevision(frameIndex);

        auto &instFrame = m_instanceBuffers[frameIndex];
        auto &skinInstFrame = m_skinInstanceBuffers[frameIndex];
        auto &skinPaletteFrame = m_skinPaletteBuffers[frameIndex];
        bool instanceMatricesWritten = false;
        if (instFrame.buffer) {
            void *mapped = instFrame.mapped;
            if (!mapped) {
                mapped = instFrame.buffer->Map();
                instFrame.mapped = mapped;
            }
            if (mapped) {
                glm::mat4 *matrices = static_cast<glm::mat4 *>(mapped);
                for (size_t i = 0; i < totalEligible; ++i) {
                    if (!eligibleDraws[i].dc->devicePose.value)
                        matrices[writeBase + i] = eligibleDraws[i].dc->worldMatrix;
                }
                instanceMatricesWritten = true;
            }
        }

        bool instanceAuxiliaryWritten = !needsInstanceAuxiliary;
        if (needsInstanceAuxiliary) {
            instanceAuxiliaryWritten = true;
            for (size_t i = 0; i < totalEligible; ++i) {
                const DrawCall &draw = *eligibleDraws[i].dc;
                const uint64_t pickingId = draw.pickingObjectId != 0 ? draw.pickingObjectId : draw.objectId;
                const auto *previous = motionHistory ? m_previousMotionScratch[i] : nullptr;
                GPUInstanceAuxData auxiliary{previous ? previous->model : draw.worldMatrix, PackGPUObjectId(pickingId),
                                             (previous ? kGPUInstanceAuxFlagValidHistory : 0u) |
                                                 (draw.receivesShadows ? 0u : kGPUInstanceAuxFlagNoReceiveShadows),
                                             draw.layerMask};
                ApplyDevicePose(auxiliary, draw.devicePose, previous ? &previous->devicePose : nullptr);
                instanceAuxiliaryWritten &=
                    WriteInstanceAuxiliary(frameIndex, writeBase + static_cast<uint32_t>(i), auxiliary);
            }
        }

        bool skinInstancesWritten = false;
        if (skinInstFrame.buffer && skinPaletteFrame.buffer) {
            auto *skinInstances = static_cast<GPUSkinInstanceData *>(skinInstFrame.mapped);
            if (!skinInstances) {
                skinInstances = static_cast<GPUSkinInstanceData *>(skinInstFrame.buffer->Map());
                skinInstFrame.mapped = skinInstances;
            }
            auto *skinBones = static_cast<glm::mat4 *>(skinPaletteFrame.mapped);
            if (!skinBones) {
                skinBones = static_cast<glm::mat4 *>(skinPaletteFrame.buffer->Map());
                skinPaletteFrame.mapped = skinBones;
            }
            if (skinInstances && skinBones) {
                auto appendPalette = [&](const std::vector<glm::mat4> *palette) {
                    const uint32_t offset = m_skinPaletteWriteOffset;
                    if (!palette || palette->empty())
                        return offset;
                    std::memcpy(&skinBones[offset], palette->data(), palette->size() * sizeof(glm::mat4));
                    m_skinPaletteWriteOffset += static_cast<uint32_t>(palette->size());
                    return offset;
                };
                auto resolveSkinData = [&](size_t index) {
                    GPUSkinInstanceData skinData{};
                    const auto *palette = eligibleDraws[index].dc->skinBoneMatrices.get();
                    if (!palette || palette->empty())
                        return skinData;

                    skinData.boneCount = static_cast<uint32_t>(palette->size());
                    skinData.flags = kGPUSkinFlagEnabled;
                    if (!motionHistory) {
                        const void *key = static_cast<const void *>(palette);
                        auto cached = m_skinPaletteFrameCache.find(key);
                        if (cached != m_skinPaletteFrameCache.end())
                            return cached->second;
                        skinData.boneOffset = appendPalette(palette);
                        skinData.previousBoneOffset = skinData.boneOffset;
                        m_skinPaletteFrameCache[key] = skinData;
                        return skinData;
                    }

                    skinData.boneOffset = appendPalette(palette);
                    const auto *previous = m_previousMotionScratch[index];
                    skinData.previousBoneOffset = previous && previous->skin.get() != palette
                                                      ? appendPalette(previous->skin.get())
                                                      : skinData.boneOffset;
                    return skinData;
                };

                for (size_t i = 0; i < totalEligible; ++i) {
                    skinInstances[writeBase + i] = resolveSkinData(i);
                }
                skinInstancesWritten = true;
            }
        }
        m_instanceWriteOffset += static_cast<uint32_t>(totalEligible);
        if (staticUnskinnedSequence && instanceMatricesWritten && skinInstancesWritten && instanceAuxiliaryWritten) {
            const auto staleRange = std::remove_if(
                m_staticInstanceRanges.begin(), m_staticInstanceRanges.end(), [&](const StaticInstanceRange &range) {
                    return range.drawListActivation == m_drawListActivation &&
                           range.sequenceKey == staticFilteredSequenceKey && range.frameIndex == frameIndex;
                });
            m_staticInstanceRanges.erase(staleRange, m_staticInstanceRanges.end());
            StaticInstanceRange range;
            range.drawListActivation = m_drawListActivation;
            range.sequenceKey = staticFilteredSequenceKey;
            range.frameIndex = frameIndex;
            range.firstInstance = writeBase;
            range.instanceCount = static_cast<uint32_t>(totalEligible);
            range.instanceBuffer = m_instanceBuffers[frameIndex].buffer
                                       ? m_instanceBuffers[frameIndex].buffer->GetBuffer()
                                       : VK_NULL_HANDLE;
            range.skinInstanceBuffer = m_skinInstanceBuffers[frameIndex].buffer
                                           ? m_skinInstanceBuffers[frameIndex].buffer->GetBuffer()
                                           : VK_NULL_HANDLE;
            range.instanceAuxBuffer = m_instanceAuxBuffers[frameIndex].buffer
                                          ? m_instanceAuxBuffers[frameIndex].buffer->GetBuffer()
                                          : VK_NULL_HANDLE;
            range.lastUsedFrame = m_ensureFrameCounter;
            m_staticInstanceRanges.push_back(std::move(range));
        }
    }
    if (reusedStaticRange)
        m_instanceWriteOffset =
            std::max<uint32_t>(m_instanceWriteOffset, writeBase + static_cast<uint32_t>(totalEligible));

    // ---- Draw loop with instanced batching ----
    VkPipeline currentPipeline = VK_NULL_HANDLE;
    VkPipelineLayout currentLayout = VK_NULL_HANDLE;
    VkDescriptorSet currentDescriptorSet = VK_NULL_HANDLE;
    VkDescriptorSet currentGlobalsSet = VK_NULL_HANDLE;
    VkPipelineLayout globalsPipelineLayout = VK_NULL_HANDLE;
    InxMaterial *currentMaterialRaw = nullptr;
    VkBuffer currentVertexBuffer = VK_NULL_HANDLE;
    VkBuffer currentIndexBuffer = VK_NULL_HANDLE;
    MeshIndexFormat currentIndexFormat = MeshIndexFormat::UInt32;
    uint64_t issuedDraws = 0;
    auto *uploadReads = vk::SubmissionReadScope::Current(cmdBuf);

    struct ResolvedMaterialPass
    {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        ShaderProgramPublication program;

        [[nodiscard]] bool IsValid() const noexcept
        {
            return pipeline != VK_NULL_HANDLE && layout != VK_NULL_HANDLE && descriptorSet != VK_NULL_HANDLE &&
                   program != nullptr;
        }
    };

    std::unordered_map<const InxMaterial *, ResolvedMaterialPass> resolvedMaterialCache;
    // Draw count is not a useful estimate of unique materials. Reserving one
    // hash bucket per instance wastes memory when renderers share a material.
    const size_t expectedUniqueMaterials = std::min<size_t>(totalEligible, 256);
    resolvedMaterialCache.reserve(expectedUniqueMaterials);
    size_t materialResolveCalls = 0;
    size_t materialCacheHits = 0;
    size_t materialUpdateCalls = 0;

    const auto syncPersistentMaterialPassCache = [&]() {
        const uint64_t publicationGeneration = m_materialPipelineManager.GetPublicationGeneration();
        if (m_materialPassResolutionCacheGeneration == publicationGeneration)
            return;
        ReleaseMaterialPassResolutionCache();
        m_materialPassResolutionCacheGeneration = publicationGeneration;
    };

    auto resolveMaterialPass = [&](const std::shared_ptr<InxMaterial> &owner,
                                   MaterialRenderData *forward = nullptr) -> ResolvedMaterialPass {
        if (!owner)
            return {};

        if (!forward)
            forward = ResolveMeshMaterial(owner);
        if (!forward)
            return {};
        const MaterialPassPipelineDescriptor defaultForward =
            m_materialPipelineManager.GetDefaultPassPipelineDescriptor(ShaderCompileTarget::Forward);
        if (activePass == defaultForward) {
            return {forward->pipeline, forward->pipelineLayout, forward->descriptorSet, forward->shaderProgram};
        }

        ShaderProgramPublication program = forward->shaderProgram;
        if (activePass.target != ShaderCompileTarget::Forward) {
            // Every semantic pass belongs to the same committed generation as
            // Forward. The material may currently hold a rejected shader pair,
            // so resolving from the mutable asset fields would mix two ABIs.
            const ShaderStagePair &stages = forward->programKey.stages;
            const auto &artifact = forward->shaderArtifact;
            if (!artifact || artifact->key != forward->programKey || !artifact->FindVariant(activePass.target)) {
                if (activePass.target == ShaderCompileTarget::Normal ||
                    activePass.target == ShaderCompileTarget::BaseColor) {
                    static std::unordered_set<std::string> reportedMissingGeometryVariants;
                    const std::string diagnosticKey =
                        stages.ToString() + "|" + ShaderCompileTargetName(activePass.target);
                    if (reportedMissingGeometryVariants.insert(diagnosticKey).second) {
                        INXLOG_ERROR(ShaderCompileTargetName(activePass.target),
                                     " buffer draw skipped: material shader program '", stages.ToString(),
                                     "' has no compiled ", ShaderCompileTargetName(activePass.target), " variant");
                    }
                }
                return {};
            }
            program = m_shaderCache.MaterializeProgramVariant(artifact, activePass.target, forward->shaderProgram);
            if (!program)
                return {};
        }
        MaterialPassRenderData *pass = m_materialPipelineManager.GetOrCreatePassRenderData(owner, program, activePass);
        if (!pass || !pass->isValid) {
            if (activePass.target == ShaderCompileTarget::Normal ||
                activePass.target == ShaderCompileTarget::BaseColor) {
                static std::unordered_set<std::string> reportedInvalidGeometryPipelines;
                const std::string diagnosticKey =
                    forward->programKey.stages.ToString() + "|" + ShaderCompileTargetName(activePass.target);
                if (reportedInvalidGeometryPipelines.insert(diagnosticKey).second) {
                    INXLOG_ERROR(ShaderCompileTargetName(activePass.target),
                                 " buffer draw skipped: failed to create the ",
                                 ShaderCompileTargetName(activePass.target), " material pipeline for '",
                                 forward->programKey.stages.ToString(), "'");
                }
            }
            return {};
        }
        return {pass->pipeline, pass->pipelineLayout, pass->descriptorSet, pass->shaderProgram};
    };

    const auto resolveCachedMaterialPass = [&](const std::shared_ptr<InxMaterial> &owner) {
        if (!owner)
            return ResolvedMaterialPass{};

        syncPersistentMaterialPassCache();
        const auto cacheIt = resolvedMaterialCache.find(owner.get());
        if (cacheIt != resolvedMaterialCache.end()) {
            ++materialCacheHits;
            return cacheIt->second;
        }

        // Both sampled textures and storage buffers can publish another
        // allocation without changing material properties. Resolve them before
        // any draw in this list borrows the cached descriptor.
        const bool propertiesChanged = owner->IsPropertiesDirty() ||
                                       m_materialPipelineManager.HasPendingTextureProperties(owner->GetMaterialKey());
        MaterialRenderData *forward = nullptr;
        if (propertiesChanged || !owner->GetBuffers().empty()) {
            forward = ResolveMeshMaterial(owner);
            if (!forward)
                return ResolvedMaterialPass{};
            materialUpdateCalls += propertiesChanged;
            syncPersistentMaterialPassCache();
        }

        const MaterialPassResolutionCacheKey persistentKey{owner.get(),
                                                           MaterialPassPipelineDescriptorHash{}(activePass)};
        const auto persistentIt = m_materialPassResolutionCache.find(persistentKey);
        if (persistentIt != m_materialPassResolutionCache.end()) {
            auto &cached = persistentIt->second;
            const auto cachedOwner = cached.owner.lock();
            const bool cacheUsable = cachedOwner && cachedOwner.get() == owner.get() && cached.pipeline == activePass &&
                                     !owner->IsPipelineDirty() && cached.IsValid() &&
                                     m_materialPipelineManager.IsDescriptorSetLive(cached.descriptorSet);
            if (cacheUsable) {
                ++materialCacheHits;
                const ResolvedMaterialPass resolved{cached.pipelineHandle, cached.pipelineLayout, cached.descriptorSet,
                                                    cached.shaderProgram};
                resolvedMaterialCache.emplace(owner.get(), resolved);
                return resolved;
            }
            m_materialPassResolutionCache.erase(persistentIt);
        }

        ++materialResolveCalls;
        ResolvedMaterialPass resolved = resolveMaterialPass(owner, forward);
        if (resolved.IsValid()) {
            resolvedMaterialCache.emplace(owner.get(), resolved);
            MaterialPassResolutionCacheEntry cached;
            cached.owner = owner;
            cached.pipeline = activePass;
            cached.pipelineHandle = resolved.pipeline;
            cached.pipelineLayout = resolved.layout;
            cached.descriptorSet = resolved.descriptorSet;
            cached.shaderProgram = resolved.program;
            m_materialPassResolutionCache[persistentKey] = std::move(cached);
        }
        return resolved;
    };

    // Instance preparation can replace the shared streams. Resolve their
    // revision now while keeping this view's camera UBO immutable.
    const VkDescriptorSet globalsDescSet =
        m_backend.Device().GetRhiDevice().Resolve(ResolveViewGlobalsBinding(viewGlobals));

    const auto previousPose = [&](size_t index) -> const RenderPoseBinding & {
        const auto *previous = motionHistory ? m_previousMotionScratch[index] : nullptr;
        return previous ? previous->devicePose : eligibleDraws[index].dc->devicePose;
    };
    const auto samePoseSources = [&](size_t first, size_t second) {
        return eligibleDraws[first].dc->devicePose.value == eligibleDraws[second].dc->devicePose.value &&
               previousPose(first).value == previousPose(second).value;
    };

    // Batch accumulation: consecutive entries sharing (pipeline, descriptorSet, VB, IB, submesh) are
    // emitted as a single vkCmdDrawIndexed with instanceCount > 1.
    const bool allowBatching = (sortMode != "back_to_front" && sortMode != "preserve");

    size_t batchFirstInstance = 0;
    uint32_t batchInstanceCount = 0;
    uint32_t batchIndexStart = 0;
    uint32_t batchIndexCount = 0;
    int32_t batchVertexStart = 0;
    VkPipelineLayout batchPipelineLayout = VK_NULL_HANDLE;
    bool batchUsesPushConstants = false;

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
        pushData.model = eligibleDraws[batchFirstInstance].dc->worldMatrix;
        pushData.normalMat = glm::mat4(1.0f); // normalMat computed in shader from SSBO model
        if (batchUsesPushConstants)
            vkCmdPushConstants(cmdBuf, batchPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants),
                               &pushData);
        vkCmdDrawIndexed(cmdBuf, batchIndexCount, batchInstanceCount, batchIndexStart, batchVertexStart,
                         writeBase + static_cast<uint32_t>(batchFirstInstance));
        if (motionHistory) {
            for (size_t index = batchFirstInstance; index < batchFirstInstance + batchInstanceCount; ++index) {
                const auto &draw = *eligibleDraws[index].dc;
                motionHistory->Record(draw.identity, draw.worldMatrix, draw.skinBoneMatrices, draw.skinContinuity,
                                      draw.devicePose);
            }
        }
        issuedDraws += batchInstanceCount;
#if INFERNUX_FRAME_PROFILE
        ++m_drawFilteredActualDraws;
#endif
        batchInstanceCount = 0;
    };

    for (size_t idx = 0; idx < totalEligible; ++idx) {
        const auto &entry = eligibleDraws[idx];
        const DrawCall &dc = *entry.dc;
        // A stale-lease clamp draws the leased buffer's complete previous
        // geometry instead of this frame's not-yet-published range.
        const uint32_t effectiveIndexCount = entry.indexCountClamp ? entry.indexCountClamp : dc.indexCount;

        // Once a batch has established valid Vulkan state, subsequent
        // consecutive instances with the same material/mesh can extend it
        // without repeating material-pipeline and descriptor validation.
        if (batchInstanceCount > 0) {
            const DrawCall &batchFirst = *eligibleDraws[batchFirstInstance].dc;
            const bool batchingAllowed =
                allowBatching || (batchFirst.allowTransparentInstancing && dc.allowTransparentInstancing);
            if (batchingAllowed && samePoseSources(batchFirstInstance, idx) && entry.material == currentMaterialRaw &&
                entry.vertexBuf == currentVertexBuffer &&
                entry.ParameterIdentity() == eligibleDraws[batchFirstInstance].ParameterIdentity() &&
                entry.indexBuf == currentIndexBuffer && dc.indexStart == batchIndexStart &&
                entry.indexFormat == currentIndexFormat && effectiveIndexCount == batchIndexCount &&
                dc.vertexStart == batchVertexStart) {
                ++batchInstanceCount;
                continue;
            }
        }

        // Material already resolved in filter loop — use directly without
        // incrementing a shared_ptr reference count for every instance.
        const std::shared_ptr<InxMaterial> *matOwner = entry.materialOwner;
        InxMaterial *matRaw = matOwner->get();
        ResolvedMaterialPass resolved = resolveCachedMaterialPass(*matOwner);
        // Missing authored materials use the default at extraction. A rejected
        // explicit program retains its own accepted publication or suspends;
        // substituting another shader here breaks all-pass coverage and ABI.
        const bool dedicatedMaterial = skyboxPass || RenderDomainRequiresDedicatedMaterial(dc.identity.domain);
        if (!resolved.IsValid()) {
            emitBatch();
            continue;
        }

        // The cached pass has already resolved pending textures. A default
        // white descriptor is Vulkan-valid but is not the requested icon alpha
        // mask. Dedicated draws stay unpublished on both pending and failed
        // first publication; an existing complete descriptor remains visible
        // during hot replacement.
        if (dedicatedMaterial &&
            m_materialPipelineManager.HasUnresolvedExplicitTextureProperties(matRaw->GetMaterialKey())) {
            emitBatch();
            continue;
        }

        if (entry.ParameterIdentity() && matRaw == entry.material) {
            MaterialDescriptorSet *rendererDescriptor =
                m_materialPipelineManager.GetDescriptorManager().GetOrCreateRendererDescriptorSet(
                    *matRaw, *resolved.program, *entry.parameters);
            if (!rendererDescriptor || !rendererDescriptor->isValid ||
                rendererDescriptor->descriptorSet == VK_NULL_HANDLE) {
                emitBatch();
                continue;
            }
            resolved.descriptorSet = rendererDescriptor->descriptorSet;
        }

        VkPipeline pipeline = resolved.pipeline;
        VkPipelineLayout pipelineLayout = resolved.layout;
        VkDescriptorSet descriptorSet = resolved.descriptorSet;
        if (dc.devicePose.value && !resolved.program->UsesDevicePoseABI()) {
            if (m_reportedPosePrograms.insert(resolved.program->GetShaderId()).second)
                INXLOG_ERROR("Material '", matRaw->GetName(),
                             "' has no resident model/bounds vertex contract; this draw is suspended");
            emitBatch();
            continue;
        }
        const VkDescriptorSet poseGlobals = ResolvePoseGlobalsBinding(globalsDescSet, dc.devicePose, previousPose(idx));

        const bool requiresMaterialBuffer =
            std::any_of(resolved.program->GetDescriptorBindings().begin(),
                        resolved.program->GetDescriptorBindings().end(), [](const MergedDescriptorBinding &binding) {
                            return binding.set == 0 && binding.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                        });
        if (requiresMaterialBuffer &&
            !m_materialPipelineManager.GetDescriptorManager().IsDescriptorSetComplete(descriptorSet)) {
            emitBatch();
            continue;
        }

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

        // The raw handles are backed by the active draw-list lease (or the
        // function-local fallback lease) for this entire command recording.
        if (entry.vertexBuf == VK_NULL_HANDLE || entry.indexBuf == VK_NULL_HANDLE) {
            emitBatch();
            continue;
        }

        const VkBuffer vb = entry.vertexBuf;

        // Check if this entry can extend the current batch. Transparent
        // instancing is opt-in so normal alpha surfaces retain per-object draws.
        bool canExtendBatch = false;
        if (batchInstanceCount > 0) {
            const DrawCall &batchFirst = *eligibleDraws[batchFirstInstance].dc;
            const bool batchingAllowed =
                allowBatching || (batchFirst.allowTransparentInstancing && dc.allowTransparentInstancing);
            canExtendBatch = batchingAllowed && samePoseSources(batchFirstInstance, idx) &&
                             pipeline == currentPipeline && descriptorSet == currentDescriptorSet &&
                             matRaw == currentMaterialRaw && vb == currentVertexBuffer &&
                             entry.ParameterIdentity() == eligibleDraws[batchFirstInstance].ParameterIdentity() &&
                             entry.indexBuf == currentIndexBuffer && dc.indexStart == batchIndexStart &&
                             entry.indexFormat == currentIndexFormat && effectiveIndexCount == batchIndexCount &&
                             dc.vertexStart == batchVertexStart;
        }

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
            currentMaterialRaw = matRaw;
            currentLayout = VK_NULL_HANDLE;
            currentDescriptorSet = VK_NULL_HANDLE;
        }

        if (descriptorSet != currentDescriptorSet || pipelineLayout != currentLayout) {
            if (!m_materialPipelineManager.IsDescriptorSetLive(descriptorSet)) {
                static int staleSet0WarnCount = 0;
                if (staleSet0WarnCount++ < 32) {
                    const uint64_t rawHandle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(descriptorSet));
                    INXLOG_WARN("[DrawSceneFiltered] stale set0 descriptor before bind: 0x", rawHandle, " mat='",
                                matRaw->GetMaterialKey(), "' name='", matRaw->GetName(),
                                "' -- forcing pipeline refresh");
                }

                resolved = resolveMaterialPass(*matOwner);
                if (resolved.IsValid()) {
                    pipeline = resolved.pipeline;
                    pipelineLayout = resolved.layout;
                    descriptorSet = resolved.descriptorSet;
                    if (pipeline != currentPipeline) {
                        vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                        currentPipeline = pipeline;
                    }
                }

                if (descriptorSet == VK_NULL_HANDLE || !m_materialPipelineManager.IsDescriptorSetLive(descriptorSet)) {
                    emitBatch();
                    continue;
                }
            }

            vkdebug::CmdBindDescriptorSetsTracked("VkCoreDraw.DrawSceneFiltered.Set0", cmdBuf,
                                                  VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet,
                                                  0, nullptr);
            currentDescriptorSet = descriptorSet;
            currentLayout = pipelineLayout;

            const ShaderProgram *program = resolved.program.get();

            if (program && program->HasDeclaredDescriptorSet(1)) {
                vkdebug::CmdBindDescriptorSetsTracked("VkCoreDraw.DrawSceneFiltered.Set1", cmdBuf,
                                                      VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 1, 1,
                                                      &perViewDescriptorSet, 0, nullptr);
            }

            // Bindless material shaders opt into the canonical device-global
            // texture table at set 3. Existing bounded and preview/special
            // shaders do not declare this set and therefore keep their old
            // descriptor path unchanged.
            if (program && program->UsesBindlessTextureABI()) {
                auto &rhiDevice = m_backend.Device().GetRhiDevice();
                const auto bindlessBinding = rhiDevice.GetBindlessTextureTableBinding();
                const VkDescriptorSet bindlessSet = rhiDevice.Resolve(bindlessBinding.group);
                if (bindlessSet == VK_NULL_HANDLE) {
                    static int missingBindlessSetErrorCount = 0;
                    if (missingBindlessSetErrorCount++ < 8) {
                        INXLOG_ERROR("Bindless material draw skipped because the device-global texture table is "
                                     "unavailable (material='",
                                     matRaw->GetMaterialKey(), "')");
                    }
                    currentDescriptorSet = VK_NULL_HANDLE;
                    currentLayout = VK_NULL_HANDLE;
                    continue;
                } else {
                    vkdebug::CmdBindDescriptorSetsTracked(
                        "VkCoreDraw.DrawSceneFiltered.BindlessTextures", cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipelineLayout, ShaderProgram::BindlessTextureSet, 1, &bindlessSet, 0, nullptr);

                    const auto *textureIndices =
                        m_materialPipelineManager.GetDescriptorManager().GetBindlessTextureIndices(descriptorSet);
                    rhiDevice.MarkBindlessTexturesUsed(
                        textureIndices && !textureIndices->empty() ? textureIndices->data() : nullptr,
                        textureIndices ? textureIndices->size() : 0);
                }
            }
        }

        if (resolved.program->HasDeclaredDescriptorSet(2) &&
            (currentGlobalsSet != poseGlobals || globalsPipelineLayout != pipelineLayout)) {
            vkdebug::CmdBindDescriptorSetsTracked("VkCoreDraw.DrawSceneFiltered.Set2", cmdBuf,
                                                  VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 2, 1, &poseGlobals,
                                                  0, nullptr);
            currentGlobalsSet = poseGlobals;
            globalsPipelineLayout = pipelineLayout;
        }

        if (vb != currentVertexBuffer) {
            if (uploadReads)
                uploadReads->Add(entry.vertexUpload, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
            VkBuffer vertBuffers[] = {vb};
            VkDeviceSize vbOffsets[] = {0};
            vkCmdBindVertexBuffers(cmdBuf, 0, 1, vertBuffers, vbOffsets);
            currentVertexBuffer = vb;
        }
        if (entry.indexBuf != currentIndexBuffer || entry.indexFormat != currentIndexFormat) {
            if (uploadReads)
                uploadReads->Add(entry.indexUpload, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
            vkCmdBindIndexBuffer(cmdBuf, entry.indexBuf, 0, ToVkIndexType(entry.indexFormat));
            currentIndexBuffer = entry.indexBuf;
            currentIndexFormat = entry.indexFormat;
        }

        // Start new batch
        batchFirstInstance = idx;
        batchInstanceCount = 1;
        batchIndexStart = dc.indexStart;
        batchIndexCount = effectiveIndexCount;
        batchVertexStart = dc.vertexStart;
        batchPipelineLayout = pipelineLayout;
        batchUsesPushConstants = resolved.program->HasVertexPushConstants();
        if (uniformBatch && allowBatching) {
            batchInstanceCount = static_cast<uint32_t>(totalEligible);
            break;
        }
    }

    // Flush final batch
    emitBatch();

    if (skyboxPass) {
        static int s_skyboxDiagCount = 0;
        if (s_skyboxDiagCount < 8) {
            INXLOG_DEBUG("[DrawSceneFiltered] SkyboxPass candidates=", filterCandidateCount,
                         " eligible=", totalEligible, " materialResolve=", materialResolveCalls,
                         " materialCacheHits=", materialCacheHits, " materialUpdates=", materialUpdateCalls,
                         " issued=", issuedDraws);
            ++s_skyboxDiagCount;
        }
    }

#if INFERNUX_FRAME_PROFILE
    stageNow = Clock::now();
    m_drawSubMs[8] += std::chrono::duration<double, std::milli>(stageNow - stageStart).count();
    m_drawSubMs[5] += std::chrono::duration<double, std::milli>(stageNow - totalStart).count();
    m_drawSceneFilteredIssued += issuedDraws;
#endif
}

// ============================================================================
// Shadow Caster Draw — renders shadow-casting objects with shadow pipeline
// ============================================================================

void InxVkCoreModular::DrawShadowCasters(VkCommandBuffer cmdBuf, uint32_t width, uint32_t height, int queueMin,
                                         int queueMax, ShadowCameraResourceId resourceId, VkBuffer viewGlobalsBuffer,
                                         const lighting::ShadowFrame &shadowFrame, int lightIndex, VkFormat depthFormat,
                                         const ShadowViewDrawCallback &additionalDraws)
{
#if INFERNUX_FRAME_PROFILE
    using Clock = std::chrono::high_resolution_clock;
    const auto totalStart = Clock::now();
    auto stageStart = totalStart;
#endif

    // NOTE: hard/soft shadow selection is NOT a property of this pass.
    // The shadow map only stores depth; stable PCF filtering happens in the
    // lit pass via shadowParams.w, driven
    // by the Light component. A former "shadowType" string parameter here was
    // a dead end and has been removed.

    (void)lightIndex;

    // Skip if shadow pipeline infrastructure not ready (lazy init)
    if (!EnsureShadowPipeline(depthFormat) || !EnsureShadowCameraResources(resourceId))
        return;
    auto resourcesIt = m_shadowCameraResources.find(resourceId);
    if (resourcesIt == m_shadowCameraResources.end())
        return;
    ShadowCameraResources &cameraResources = resourcesIt->second;
    const uint32_t viewCount =
        std::min<uint32_t>(static_cast<uint32_t>(shadowFrame.views.size()), lighting::MaxShadowViews);
    if (viewCount == 0)
        return;

#if INFERNUX_FRAME_PROFILE
    ++m_drawShadowCalls;
#endif

    const uint32_t frameIndex = m_currentFrame % m_maxFramesInFlight;

    auto &stream = cameraResources.streamFrames[frameIndex];
    if (stream.globalsBuffer != viewGlobalsBuffer) {
        stream.globalsBuffer = viewGlobalsBuffer;
        UpdateShadowCameraStreamDescriptor(stream);
    }

    for (uint32_t viewIndex = 0; viewIndex < viewCount; ++viewIndex) {
        const uint32_t bufferIndex = frameIndex * lighting::MaxShadowViews + viewIndex;
        if (bufferIndex >= cameraResources.mappedPointers.size() || !cameraResources.mappedPointers[bufferIndex])
            return;
        const lighting::ShadowView &shadowView = shadowFrame.views[viewIndex];
        UniformBufferObject view{};
        view.model = view.view = glm::mat4(1.0f);
        view.proj = view.previousViewProj = shadowView.viewProjection;
        view.inverseViewProj = glm::inverse(shadowView.viewProjection);
        ShadowPassParameters parameters;
        parameters.lightVector = glm::vec4(shadowView.lightVector, shadowView.lightVectorIsPosition ? 1.0f : 0.0f);
        parameters.bias = glm::vec4(shadowView.depthBiasTexels, shadowView.normalBiasTexels,
                                    shadowView.worldUnitsPerTexel, shadowView.farPlane);
        auto *mapped = static_cast<char *>(cameraResources.mappedPointers[bufferIndex]);
        std::memcpy(mapped, &view, sizeof(view));
        std::memcpy(mapped + m_shadowUniformParametersOffset, &parameters, sizeof(parameters));
    }

    // Bind shadow pipeline once
    // NOTE: Per-material shadow pipelines override this in the inner loop
    VkPipeline lastBoundPipeline = VK_NULL_HANDLE;

    // Pre-build draw list (filter once, reuse for all cascades and, for an
    // immutable static publication, across frames).
    const uint64_t shadowMaterialPublicationGeneration = m_materialPipelineManager.GetPublicationGeneration();
    const bool reusedShadowScratch =
        m_shadowScratchValid && m_shadowScratchDrawListActivation == m_drawListActivation &&
        m_shadowScratchMaterialPublicationGeneration == shadowMaterialPublicationGeneration &&
        m_shadowScratchDepthFormat == depthFormat && m_shadowScratchQueueMin == queueMin &&
        m_shadowScratchQueueMax == queueMax;
#if INFERNUX_FRAME_PROFILE
    auto stageNow = Clock::now();
#endif
    const auto &activeShadowDrawCalls = shadowDrawCalls();
    const bool hasShadowListMetadata = m_shadowListMetadataSource == m_shadowDrawCallsPtr &&
                                       m_shadowListMetadata.size() == activeShadowDrawCalls.size() &&
                                       m_shadowListBufferRevision == m_objectBufferRevision;
    const bool usesMainDrawListMetadata = !m_shadowDrawCallsPtr && m_drawListMetadataSource == m_drawCallsPtr &&
                                          m_drawListMetadata.size() == activeShadowDrawCalls.size() &&
                                          m_drawListBufferRevision == m_objectBufferRevision;
    std::vector<DrawListMetadata> fallbackBufferLeases;
    if (!reusedShadowScratch) {
        m_shadowDrawScratch.clear();
        m_shadowDrawScratch.reserve(activeShadowDrawCalls.size());
        m_resolvedShadowMaterialsScratch.clear();
        m_resolvedShadowMaterialsScratch.reserve(std::min<size_t>(activeShadowDrawCalls.size(), 256));
        if (!hasShadowListMetadata && !usesMainDrawListMetadata)
            fallbackBufferLeases.reserve(activeShadowDrawCalls.size());
        const InxMaterial *lastResolvedShadowMaterial = nullptr;
        ResolvedShadowMaterial lastResolvedShadowResources{};
        for (size_t drawCallIndex = 0; drawCallIndex < activeShadowDrawCalls.size(); ++drawCallIndex) {
            const DrawCall &dc = activeShadowDrawCalls[drawCallIndex];
            // Empty renderer slots use DefaultLit in both color and shadow
            // draws, including the first primitive in a fresh project.
            const auto &material = dc.material ? dc.material : m_cachedDefaultLit;
            const DrawListMetadata *metadata =
                hasShadowListMetadata ? &m_shadowListMetadata[drawCallIndex]
                                      : (usesMainDrawListMetadata ? &m_drawListMetadata[drawCallIndex] : nullptr);
            const uint64_t requiredIndexEnd = static_cast<uint64_t>(dc.indexStart) + dc.indexCount;
            if (metadata && (metadata->objectId != dc.objectId || metadata->material != material.get() ||
                             requiredIndexEnd > metadata->indexCapacity))
                metadata = nullptr;
            if (!dc.castsShadows || !material)
                continue;
            const int renderQueue = metadata ? metadata->renderQueue : material->GetRenderQueue();
            if (renderQueue < queueMin || renderQueue > queueMax)
                continue;
            const DrawListMetadata *bufferLease =
                metadata && metadata->HasVertexBuffer() && metadata->indexBuffer ? metadata : nullptr;
            if (!bufferLease) {
                const auto bufferIt = m_perObjectBuffers.find(dc.objectId);
                if (bufferIt != m_perObjectBuffers.end() && bufferIt->second.HasVertexBuffer() &&
                    bufferIt->second.indexBuffer && requiredIndexEnd <= bufferIt->second.indexCount) {
                    fallbackBufferLeases.push_back({dc.objectId, material.get(), renderQueue,
                                                    bufferIt->second.vertexBuffer, bufferIt->second.indexBuffer,
                                                    bufferIt->second.indexCount, bufferIt->second.residentVertexHandle,
                                                    bufferIt->second.indexFormat});
                    bufferLease = &fallbackBufferLeases.back();
                }
            }
            if (!bufferLease)
                continue;

            ResolvedShadowMaterial resources{};
            if (material.get() == lastResolvedShadowMaterial) {
                resources = lastResolvedShadowResources;
            } else {
                auto resolved = m_resolvedShadowMaterialsScratch.find(material.get());
                if (resolved == m_resolvedShadowMaterialsScratch.end()) {
                    const VkDescriptorSet descriptorSet = EnsureMaterialShadowPipeline(material, depthFormat);
                    resources.program = material->GetPassShaderProgram(ShaderCompileTarget::Shadow);
                    resources.pipeline = material->GetPassPipeline(ShaderCompileTarget::Shadow);
                    resources.descriptorSet = descriptorSet;
                    resolved = m_resolvedShadowMaterialsScratch.emplace(material.get(), resources).first;
                } else {
                    resources = resolved->second;
                }
                lastResolvedShadowMaterial = material.get();
                lastResolvedShadowResources = resources;
            }
            const VkPipeline pip = resources.pipeline;
            VkDescriptorSet shadowMatDesc = resources.descriptorSet;
            if (pip == VK_NULL_HANDLE || shadowMatDesc == VK_NULL_HANDLE)
                continue;
            if (dc.parameterBlock) {
                const auto *descriptor =
                    m_materialPipelineManager.GetDescriptorManager().GetOrCreateRendererDescriptorSet(
                        *material, *resources.program, dc.parameterBlock);
                if (!descriptor || !descriptor->isValid)
                    continue;
                shadowMatDesc = descriptor->descriptorSet;
                if (descriptor->usesBindlessTextureABI)
                    m_backend.Device().GetRhiDevice().MarkBindlessTexturesUsed(
                        descriptor->bindlessTextureIndices.data(), descriptor->bindlessTextureIndices.size());
            }
            if (dc.devicePose.value && !resources.program->UsesDevicePoseABI())
                continue;
            m_shadowDrawScratch.push_back({&dc, bufferLease->GetVertexHandle(), bufferLease->indexBuffer->GetBuffer(),
                                           pip, resources.program, shadowMatDesc, dc.worldBounds,
                                           bufferLease->indexFormat, bufferLease->GetVertexUpload(),
                                           bufferLease->indexBuffer->GetUploadSubmission()});
        }

#if INFERNUX_FRAME_PROFILE
        stageNow = Clock::now();
        m_drawSubMs[10] += std::chrono::duration<double, std::milli>(stageNow - stageStart).count();
        stageStart = stageNow;
        m_drawShadowEligible += static_cast<uint64_t>(m_shadowDrawScratch.size());
#endif

        if (m_shadowDrawScratch.empty() && !additionalDraws) {
#if INFERNUX_FRAME_PROFILE
            m_drawSubMs[9] += std::chrono::duration<double, std::milli>(Clock::now() - totalStart).count();
#endif
            return;
        }

        // Sort shadow draw scratch by (pipeline, VB, submesh) for instanced batching.
        // Stable scene extraction commonly already produces this order (and one
        // uniform batch compares equivalent throughout), so avoid an unconditional
        // O(N log N) sort for large standard-GameObject scenes.
        const auto shadowDrawLess = [](const ShadowDraw &a, const ShadowDraw &b) {
            if (a.shadowPipeline != b.shadowPipeline)
                return a.shadowPipeline < b.shadowPipeline;
            if (a.shadowMaterialDescSet != b.shadowMaterialDescSet)
                return a.shadowMaterialDescSet < b.shadowMaterialDescSet;
            const VkBuffer va = a.vertexBuf;
            const VkBuffer vb_b = b.vertexBuf;
            if (va != vb_b)
                return va < vb_b;
            if (a.indexBuf != b.indexBuf)
                return a.indexBuf < b.indexBuf;
            if (a.indexFormat != b.indexFormat)
                return static_cast<uint32_t>(a.indexFormat) < static_cast<uint32_t>(b.indexFormat);
            if (a.dc->indexStart != b.dc->indexStart)
                return a.dc->indexStart < b.dc->indexStart;
            return a.dc->indexCount < b.dc->indexCount;
        };
        if (!std::is_sorted(m_shadowDrawScratch.begin(), m_shadowDrawScratch.end(), shadowDrawLess))
            std::sort(m_shadowDrawScratch.begin(), m_shadowDrawScratch.end(), shadowDrawLess);

        // Build a lightweight hierarchy over the stable, batch-sorted caster
        // list. It is deliberately independent of scene type: any large static
        // renderer publication benefits, while intersecting boundary groups
        // still fall back to exact per-object tests.
        constexpr uint32_t kShadowCullGroupSize = 64;
        m_shadowCullGroups.clear();
        m_shadowCullGroups.reserve((m_shadowDrawScratch.size() + kShadowCullGroupSize - 1) / kShadowCullGroupSize);
        for (uint32_t first = 0; first < m_shadowDrawScratch.size(); first += kShadowCullGroupSize) {
            ShadowCullGroup group;
            group.first = first;
            group.count =
                std::min<uint32_t>(kShadowCullGroupSize, static_cast<uint32_t>(m_shadowDrawScratch.size()) - first);
            group.layerMaskIntersection = UINT32_MAX;
            group.allBoundsValid = true;
            bool haveBounds = false;
            for (uint32_t offset = 0; offset < group.count; ++offset) {
                const ShadowDraw &draw = m_shadowDrawScratch[first + offset];
                group.layerMaskUnion |= draw.dc->layerMask;
                group.layerMaskIntersection &= draw.dc->layerMask;
                if (draw.dc->devicePose.value || !draw.worldBounds.IsValid()) {
                    group.allBoundsValid = false;
                    continue;
                }
                if (!haveBounds) {
                    group.worldBounds = draw.worldBounds;
                    haveBounds = true;
                } else {
                    group.worldBounds.min = glm::min(group.worldBounds.min, draw.worldBounds.min);
                    group.worldBounds.max = glm::max(group.worldBounds.max, draw.worldBounds.max);
                }
            }
            group.allBoundsValid &= haveBounds;
            m_shadowCullGroups.push_back(group);
        }
        m_shadowAllVisible.resize(m_shadowDrawScratch.size());
        for (uint32_t index = 0; index < m_shadowAllVisible.size(); ++index)
            m_shadowAllVisible[index] = index;

        m_shadowScratchStatic =
            !m_shadowDrawScratch.empty() &&
            std::all_of(m_shadowDrawScratch.begin(), m_shadowDrawScratch.end(), [](const ShadowDraw &draw) {
                return draw.dc && draw.dc->isStatic && !draw.dc->skinBoneMatrices;
            });
        m_shadowScratchUniformBatch = m_shadowDrawScratch.size() > 1;
        if (m_shadowScratchUniformBatch) {
            const ShadowDraw &first = m_shadowDrawScratch.front();
            for (size_t index = 1; index < m_shadowDrawScratch.size(); ++index) {
                const ShadowDraw &draw = m_shadowDrawScratch[index];
                if (draw.shadowPipeline != first.shadowPipeline ||
                    draw.shadowMaterialDescSet != first.shadowMaterialDescSet || draw.vertexBuf != first.vertexBuf ||
                    draw.indexBuf != first.indexBuf || draw.indexFormat != first.indexFormat ||
                    draw.dc->indexStart != first.dc->indexStart || draw.dc->indexCount != first.dc->indexCount ||
                    draw.dc->vertexStart != first.dc->vertexStart) {
                    m_shadowScratchUniformBatch = false;
                    break;
                }
            }
        }
        m_shadowScratchDrawListActivation = m_drawListActivation;
        m_shadowScratchMaterialPublicationGeneration = shadowMaterialPublicationGeneration;
        m_shadowScratchDepthFormat = depthFormat;
        m_shadowScratchQueueMin = queueMin;
        m_shadowScratchQueueMax = queueMax;
        m_shadowScratchValid = m_shadowScratchStatic && (hasShadowListMetadata || usesMainDrawListMetadata);
    }

#if INFERNUX_FRAME_PROFILE
    stageNow = Clock::now();
    m_drawSubMs[12] += std::chrono::duration<double, std::milli>(stageNow - stageStart).count();
    stageStart = stageNow;
#endif

    uint64_t issuedDraws = 0;

    // Static shadow cache eligibility is intentionally strict. A single
    // dynamic or skinned entry keeps the original per-view cull/upload path,
    // while a fully static submission is validated field-for-field before any
    // cached visibility is reused. This also catches edits made to objects
    // that are still marked static.
    const bool staticShadowSubmission = m_shadowScratchStatic;
    auto &staticShadowCache = cameraResources.staticSubmissionCache;
    auto stateMatches = [](const ShadowCameraResources::StaticShadowDrawState &cached, const ShadowDraw &current) {
        const DrawCall &draw = *current.dc;
        return cached.objectId == draw.objectId && cached.layerMask == draw.layerMask &&
               cached.receivesShadows == draw.receivesShadows &&
               std::memcmp(&cached.worldMatrix, &draw.worldMatrix, sizeof(glm::mat4)) == 0 &&
               cached.worldBounds.min.x == draw.worldBounds.min.x &&
               cached.worldBounds.min.y == draw.worldBounds.min.y &&
               cached.worldBounds.min.z == draw.worldBounds.min.z &&
               cached.worldBounds.max.x == draw.worldBounds.max.x &&
               cached.worldBounds.max.y == draw.worldBounds.max.y &&
               cached.worldBounds.max.z == draw.worldBounds.max.z && cached.vertexBuffer == current.vertexBuf &&
               cached.indexBuffer == current.indexBuf && cached.indexFormat == current.indexFormat &&
               cached.pipeline == current.shadowPipeline &&
               cached.materialDescriptor == current.shadowMaterialDescSet && cached.indexStart == draw.indexStart &&
               cached.indexCount == draw.indexCount && cached.vertexStart == draw.vertexStart;
    };
    auto captureState = [](const ShadowDraw &current) {
        const DrawCall &draw = *current.dc;
        ShadowCameraResources::StaticShadowDrawState state;
        state.objectId = draw.objectId;
        state.layerMask = draw.layerMask;
        state.receivesShadows = draw.receivesShadows;
        state.worldMatrix = draw.worldMatrix;
        state.worldBounds = draw.worldBounds;
        state.vertexBuffer = current.vertexBuf;
        state.indexBuffer = current.indexBuf;
        state.indexFormat = current.indexFormat;
        state.pipeline = current.shadowPipeline;
        state.materialDescriptor = current.shadowMaterialDescSet;
        state.indexStart = draw.indexStart;
        state.indexCount = draw.indexCount;
        state.vertexStart = draw.vertexStart;
        return state;
    };

    bool staticShadowStateUnchanged = reusedShadowScratch && staticShadowSubmission && staticShadowCache.valid;
    if (!reusedShadowScratch)
        staticShadowStateUnchanged = staticShadowSubmission && staticShadowCache.valid &&
                                     staticShadowCache.draws.size() == m_shadowDrawScratch.size();
    if (staticShadowStateUnchanged && !reusedShadowScratch) {
        for (size_t index = 0; index < m_shadowDrawScratch.size(); ++index) {
            if (!stateMatches(staticShadowCache.draws[index], m_shadowDrawScratch[index])) {
                staticShadowStateUnchanged = false;
                break;
            }
        }
    }
    if (staticShadowSubmission && !staticShadowStateUnchanged) {
        staticShadowCache.draws.clear();
        staticShadowCache.draws.reserve(m_shadowDrawScratch.size());
        for (const ShadowDraw &draw : m_shadowDrawScratch)
            staticShadowCache.draws.push_back(captureState(draw));
        for (auto &view : staticShadowCache.views) {
            view.valid = false;
            view.fullSequence = false;
            view.visibleIndices.clear();
        }
        staticShadowCache.valid = true;
    } else if (!staticShadowSubmission) {
        staticShadowCache.valid = false;
    }

    std::array<Frustum, lighting::MaxShadowViews> shadowFrustums{};
    for (uint32_t viewIndex = 0; viewIndex < viewCount; ++viewIndex)
        shadowFrustums[viewIndex].ExtractFromMatrix(shadowFrame.views[viewIndex].viewProjection);

    if (frameIndex >= cameraResources.streamFrames.size())
        return;
    auto &shadowStream = cameraResources.streamFrames[frameIndex];
    if (shadowStream.frameSerial != m_ensureFrameCounter) {
        shadowStream.frameSerial = m_ensureFrameCounter;
        shadowStream.instanceWriteOffset = 0;
        shadowStream.skinPaletteWriteOffset = 0;
        shadowStream.skinPaletteCache.clear();
    }

    size_t maxShadowBoneMatrices = shadowStream.skinPaletteWriteOffset;
    for (const auto &sd : m_shadowDrawScratch) {
        if (sd.dc->skinBoneMatrices)
            maxShadowBoneMatrices += sd.dc->skinBoneMatrices->size() * viewCount;
    }
    const size_t maxShadowInstances =
        static_cast<size_t>(shadowStream.instanceWriteOffset) + m_shadowDrawScratch.size() * viewCount;
    if (!EnsureShadowCameraStreamCapacity(cameraResources, frameIndex, maxShadowInstances, maxShadowBoneMatrices))
        return;

    const VkDescriptorSet shadowStreamDescSet = shadowStream.descriptorSet;
    if (shadowStreamDescSet == VK_NULL_HANDLE)
        return;

    for (uint32_t viewIndex = 0; viewIndex < viewCount; ++viewIndex) {
        const uint32_t descIdx = frameIndex * lighting::MaxShadowViews + viewIndex;
        if (descIdx >= cameraResources.descriptorSets.size())
            break;

        const lighting::ShadowView &shadowView = shadowFrame.views[viewIndex];
        if (!shadowView.atlas.IsValid())
            continue;
        const uint32_t tileX = shadowView.atlas.x + shadowView.atlas.guard;
        const uint32_t tileY = shadowView.atlas.y + shadowView.atlas.guard;
        const uint32_t tileW = std::min(shadowView.atlas.InnerSize(), width - std::min(tileX, width));
        const uint32_t tileH = std::min(shadowView.atlas.InnerSize(), height - std::min(tileY, height));
        if (tileW == 0 || tileH == 0)
            continue;

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

        // Unified caster bias for every view type: the slope-scaled raster
        // bias tracks each polygon's own depth gradient (which no
        // receiver-side term can express), while the receiver-side
        // normal/light offsets in lighting.glsl absorb quantization. Casters
        // themselves are never moved in world space.
        vkCmdSetDepthBias(cmdBuf, -1.0f, 0.0f, -2.0f);

        // The complete shadow descriptor state is bound per batch below. Set 0
        // belongs to this camera and this shadow view.
        VkDescriptorSet cascadeDescSet = cameraResources.descriptorSets[descIdx];
        if (cascadeDescSet == VK_NULL_HANDLE)
            continue;

        const Frustum &shadowFrustum = shadowFrustums[viewIndex];
        VkBuffer currentVertexBuffer = VK_NULL_HANDLE;
        VkBuffer currentIndexBuffer = VK_NULL_HANDLE;
        MeshIndexFormat currentIndexFormat = MeshIndexFormat::UInt32;
        // Per-cascade frustum cull into a compact index list, then upload
        // model matrices for visible objects and batch by (pipeline, VB, IB, submesh).
        // Directional cascades must not cull against the light-space near
        // plane: casters between the light and the cascade volume are pancaked
        // onto the near plane by the shadow vertex shader, so rejecting them
        // here would punch holes into shadows cast by tall or distant objects.
        const bool ignoreNearPlane = shadowView.type == lighting::ShadowViewType::DirectionalCascade;
        const std::vector<uint32_t> *visibleIndices = nullptr;
        auto &cachedView = staticShadowCache.views[viewIndex];
        const bool cachedVisibility =
            staticShadowSubmission && cachedView.valid && cachedView.cullingMask == shadowView.cullingMask &&
            cachedView.type == shadowView.type &&
            (cachedView.fullSequence ||
             std::memcmp(&cachedView.viewProjection, &shadowView.viewProjection, sizeof(glm::mat4)) == 0);
        if (cachedVisibility) {
            visibleIndices = cachedView.fullSequence ? &m_shadowAllVisible : &cachedView.visibleIndices;
        } else {
            m_shadowViewVisible.clear();
            m_shadowViewVisible.reserve(m_shadowDrawScratch.size());
            for (const ShadowCullGroup &group : m_shadowCullGroups) {
                if ((group.layerMaskUnion & shadowView.cullingMask) == 0u)
                    continue;

                const Frustum::AABBRelation relation =
                    group.allBoundsValid ? shadowFrustum.ClassifyAABB(group.worldBounds, ignoreNearPlane)
                                         : Frustum::AABBRelation::Intersecting;
                if (relation == Frustum::AABBRelation::Outside)
                    continue;

                const bool everyLayerAccepted = (group.layerMaskIntersection & shadowView.cullingMask) != 0u;
                if (relation == Frustum::AABBRelation::Inside && everyLayerAccepted) {
                    for (uint32_t offset = 0; offset < group.count; ++offset)
                        m_shadowViewVisible.push_back(group.first + offset);
                    continue;
                }

                for (uint32_t offset = 0; offset < group.count; ++offset) {
                    const uint32_t drawIndex = group.first + offset;
                    const ShadowDraw &draw = m_shadowDrawScratch[drawIndex];
                    if ((draw.dc->layerMask & shadowView.cullingMask) == 0u)
                        continue;
                    if (!draw.dc->devicePose.value && relation == Frustum::AABBRelation::Intersecting &&
                        draw.worldBounds.IsValid() && !shadowFrustum.IntersectsAABB(draw.worldBounds, ignoreNearPlane))
                        continue;
                    m_shadowViewVisible.push_back(drawIndex);
                }
            }
            const bool everyCasterLayerAccepted =
                std::all_of(m_shadowCullGroups.begin(), m_shadowCullGroups.end(), [&](const ShadowCullGroup &group) {
                    return (group.layerMaskIntersection & shadowView.cullingMask) != 0u;
                });
            // Compacting and uploading a list that already contains nearly all
            // static casters costs more CPU than letting fixed-function clipping
            // reject the small outside tail. Canonicalizing dense views to the
            // full sequence also makes their per-frame-in-flight uploads stable.
            // A large static submission that resolves to one cheap instanced
            // batch is usually faster when its immutable matrix stream is
            // reused verbatim. Re-compacting even one cascade copies several
            // megabytes every frame and prevents every frame-in-flight stream
            // from becoming resident. Keep the conservative dense-view rule
            // for arbitrary geometry, and extend it only to simple uniform
            // batches whose full index workload remains bounded.
            constexpr uint64_t kSimpleStaticShadowIndexBudget = 4ull * 1024ull * 1024ull;
            const uint64_t fullSequenceIndexWork =
                m_shadowScratchUniformBatch && !m_shadowDrawScratch.empty()
                    ? static_cast<uint64_t>(m_shadowAllVisible.size()) *
                          static_cast<uint64_t>(m_shadowDrawScratch.front().dc->indexCount)
                    : std::numeric_limits<uint64_t>::max();
            const bool cheapUniformFullSequence = m_shadowScratchUniformBatch && m_shadowAllVisible.size() >= 4096 &&
                                                  fullSequenceIndexWork <= kSimpleStaticShadowIndexBudget &&
                                                  m_shadowViewVisible.size() * 4 >= m_shadowAllVisible.size();
            const bool useFullStaticSequence =
                staticShadowSubmission && everyCasterLayerAccepted && !m_shadowAllVisible.empty() &&
                (m_shadowViewVisible.size() * 5 >= m_shadowAllVisible.size() * 4 || cheapUniformFullSequence);
            visibleIndices = useFullStaticSequence ? &m_shadowAllVisible : &m_shadowViewVisible;
            if (staticShadowSubmission) {
                // Adaptive CSM may move the light-space matrix by tiny amounts
                // while selecting the exact same static casters. Preserve the
                // visibility generation in that case so every frame-in-flight
                // stream can reuse its already uploaded matrix range.
                const bool sameVisibility = cachedView.valid && cachedView.cullingMask == shadowView.cullingMask &&
                                            cachedView.type == shadowView.type &&
                                            cachedView.fullSequence == useFullStaticSequence &&
                                            (useFullStaticSequence || cachedView.visibleIndices == m_shadowViewVisible);
                cachedView.viewProjection = shadowView.viewProjection;
                cachedView.cullingMask = shadowView.cullingMask;
                cachedView.type = shadowView.type;
                if (!sameVisibility) {
                    cachedView.fullSequence = useFullStaticSequence;
                    if (useFullStaticSequence)
                        cachedView.visibleIndices.clear();
                    else
                        cachedView.visibleIndices = m_shadowViewVisible;
                    cachedView.generation = staticShadowCache.nextGeneration++;
                    if (staticShadowCache.nextGeneration == 0)
                        staticShadowCache.nextGeneration = 1;
                }
                cachedView.valid = true;
                visibleIndices = cachedView.fullSequence ? &m_shadowAllVisible : &cachedView.visibleIndices;
            }
        }

#if INFERNUX_FRAME_PROFILE
        stageNow = Clock::now();
        m_drawSubMs[13] += std::chrono::duration<double, std::milli>(stageNow - stageStart).count();
        stageStart = stageNow;
#endif

        const uint32_t visibleCount = static_cast<uint32_t>(visibleIndices->size());
        if (visibleCount == 0) {
            if (additionalDraws) {
                additionalDraws(viewIndex, shadowView);
                lastBoundPipeline = VK_NULL_HANDLE;
            }
            continue;
        }

        // Each camera appends into its own frame-local shadow instance stream.
        const uint32_t writeBase = shadowStream.instanceWriteOffset;
        if (!shadowStream.instanceMapped || !shadowStream.skinInstanceMapped || !shadowStream.skinPaletteMapped)
            continue;
        const VkBuffer instanceBuffer =
            shadowStream.instanceBuffer ? shadowStream.instanceBuffer->GetBuffer() : VK_NULL_HANDLE;
        const VkBuffer skinInstanceBuffer =
            shadowStream.skinInstanceBuffer ? shadowStream.skinInstanceBuffer->GetBuffer() : VK_NULL_HANDLE;
        auto &staticUpload = shadowStream.staticViewUploads[viewIndex];
        const bool reuseStaticUpload =
            staticShadowSubmission && cachedView.valid && staticUpload.valid &&
            staticUpload.visibilityGeneration == cachedView.generation && staticUpload.firstInstance == writeBase &&
            staticUpload.instanceCount == visibleCount && staticUpload.instanceBuffer == instanceBuffer &&
            staticUpload.skinInstanceBuffer == skinInstanceBuffer;
        if (!reuseStaticUpload) {
            auto *matrices = static_cast<glm::mat4 *>(shadowStream.instanceMapped);
            auto *auxiliary = static_cast<GPUInstanceAuxData *>(shadowStream.auxiliaryMapped);
            for (uint32_t vi = 0; vi < visibleCount; ++vi) {
                const auto &draw = *m_shadowDrawScratch[(*visibleIndices)[vi]].dc;
                if (!draw.devicePose.value)
                    matrices[writeBase + vi] = draw.worldMatrix;
                auxiliary[writeBase + vi] = {
                    draw.worldMatrix,
                    glm::uvec2(static_cast<uint32_t>(draw.objectId), static_cast<uint32_t>(draw.objectId >> 32u)),
                    draw.receivesShadows ? 0u : kGPUInstanceAuxFlagNoReceiveShadows, draw.layerMask};
                ApplyDevicePose(auxiliary[writeBase + vi], draw.devicePose);
            }

            auto *skinInstances = static_cast<GPUSkinInstanceData *>(shadowStream.skinInstanceMapped);
            auto *skinBones = static_cast<glm::mat4 *>(shadowStream.skinPaletteMapped);
            auto resolveSkinData = [&](const std::vector<glm::mat4> *palette) {
                GPUSkinInstanceData skinData{};
                if (!palette || palette->empty())
                    return skinData;
                const void *key = static_cast<const void *>(palette);
                auto cached = shadowStream.skinPaletteCache.find(key);
                if (cached != shadowStream.skinPaletteCache.end())
                    return cached->second;

                skinData.boneOffset = shadowStream.skinPaletteWriteOffset;
                skinData.boneCount = static_cast<uint32_t>(palette->size());
                skinData.flags = kGPUSkinFlagEnabled;
                std::memcpy(&skinBones[shadowStream.skinPaletteWriteOffset], palette->data(),
                            palette->size() * sizeof(glm::mat4));
                shadowStream.skinPaletteWriteOffset += static_cast<uint32_t>(palette->size());
                shadowStream.skinPaletteCache[key] = skinData;
                return skinData;
            };

            for (uint32_t vi = 0; vi < visibleCount; ++vi) {
                const DrawCall *dc = m_shadowDrawScratch[(*visibleIndices)[vi]].dc;
                skinInstances[writeBase + vi] = resolveSkinData(dc ? dc->skinBoneMatrices.get() : nullptr);
            }
            if (staticShadowSubmission) {
                staticUpload.visibilityGeneration = cachedView.generation;
                staticUpload.firstInstance = writeBase;
                staticUpload.instanceCount = visibleCount;
                staticUpload.instanceBuffer = instanceBuffer;
                staticUpload.skinInstanceBuffer = skinInstanceBuffer;
                staticUpload.valid = true;
            } else {
                staticUpload.valid = false;
            }
        }
        shadowStream.instanceWriteOffset += visibleCount;

#if INFERNUX_FRAME_PROFILE
        stageNow = Clock::now();
        m_drawSubMs[14] += std::chrono::duration<double, std::milli>(stageNow - stageStart).count();
        stageStart = stageNow;
#endif

        // Batch accumulation: consecutive visible entries sharing (pipeline, VB, IB, submesh)
        // are emitted as a single instanced draw call.
        size_t batchStart = 0;
        uint32_t batchCount = 0;
        VkPipeline batchPipeline = VK_NULL_HANDLE;
        VkDescriptorSet batchShadowMaterialDescSet = VK_NULL_HANDLE;
        uint32_t batchIdxStart = 0;
        uint32_t batchIdxCount = 0;
        int32_t batchVtxStart = 0;
        auto &rhiDevice = m_backend.Device().GetRhiDevice();
        const VkDescriptorSet bindlessSet = rhiDevice.Resolve(rhiDevice.GetBindlessTextureTableBinding().group);

        auto *uploadReads = vk::SubmissionReadScope::Current(cmdBuf);
        auto emitShadowBatch = [&]() {
            if (batchCount == 0)
                return;
            const auto &first = m_shadowDrawScratch[(*visibleIndices)[batchStart]];
            const VkPipelineLayout layout = first.program->GetPipelineLayout();
            const auto poseGlobals = ResolvePoseGlobalsBinding(shadowStreamDescSet, first.dc->devicePose);
            const std::array<VkDescriptorSet, 4> descriptorSets = {batchShadowMaterialDescSet, cascadeDescSet,
                                                                   poseGlobals, bindlessSet};
            vkdebug::CmdBindDescriptorSetsTracked(
                "VkCoreDraw.DrawShadowCasters.AllSets", cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0,
                first.program->UsesBindlessTextureABI() ? 4u : 3u, descriptorSets.data(), 0, nullptr);
            struct PushData
            {
                glm::mat4 model;
                glm::mat4 normalMat;
            } pushData;
            pushData.model = m_shadowDrawScratch[(*visibleIndices)[batchStart]].dc->worldMatrix;
            pushData.normalMat = glm::mat4(1.0f);
            if (!first.program->GetVertexReflection().GetPushConstants().empty())
                vkCmdPushConstants(cmdBuf, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushData), &pushData);
            vkCmdDrawIndexed(cmdBuf, batchIdxCount, batchCount, batchIdxStart, batchVtxStart,
                             writeBase + static_cast<uint32_t>(batchStart));
            issuedDraws += batchCount;
#if INFERNUX_FRAME_PROFILE
            ++m_drawShadowActualDraws;
#endif
            batchCount = 0;
        };

        for (uint32_t vi = 0; vi < visibleCount; ++vi) {
            const auto &sd = m_shadowDrawScratch[(*visibleIndices)[vi]];

            const VkBuffer vb = sd.vertexBuf;

            bool canExtend =
                batchCount > 0 &&
                sd.dc->devicePose.value == m_shadowDrawScratch[(*visibleIndices)[batchStart]].dc->devicePose.value &&
                sd.shadowPipeline == batchPipeline && vb == currentVertexBuffer && sd.indexBuf == currentIndexBuffer &&
                sd.indexFormat == currentIndexFormat && sd.shadowMaterialDescSet == batchShadowMaterialDescSet &&
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

            if (vb != currentVertexBuffer) {
                if (uploadReads)
                    uploadReads->Add(sd.vertexUpload, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
                VkDeviceSize offsets[] = {0};
                vkCmdBindVertexBuffers(cmdBuf, 0, 1, &vb, offsets);
                currentVertexBuffer = vb;
            }
            if (sd.indexBuf != currentIndexBuffer || sd.indexFormat != currentIndexFormat) {
                if (uploadReads)
                    uploadReads->Add(sd.indexUpload, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
                vkCmdBindIndexBuffer(cmdBuf, sd.indexBuf, 0, ToVkIndexType(sd.indexFormat));
                currentIndexBuffer = sd.indexBuf;
                currentIndexFormat = sd.indexFormat;
            }

            batchStart = vi;
            batchCount = 1;
            batchPipeline = sd.shadowPipeline;
            batchShadowMaterialDescSet = sd.shadowMaterialDescSet;
            batchIdxStart = sd.dc->indexStart;
            batchIdxCount = sd.dc->indexCount;
            batchVtxStart = sd.dc->vertexStart;
            if (m_shadowScratchUniformBatch) {
                batchCount = visibleCount;
                break;
            }
        }

        emitShadowBatch();

        if (additionalDraws) {
            additionalDraws(viewIndex, shadowView);
            // The callback owns its RHI pipeline state. Geometry must bind its
            // pipeline again before recording the next cascade.
            lastBoundPipeline = VK_NULL_HANDLE;
        }

#if INFERNUX_FRAME_PROFILE
        stageNow = Clock::now();
        m_drawSubMs[15] += std::chrono::duration<double, std::milli>(stageNow - stageStart).count();
        stageStart = stageNow;
#endif
    }

#if INFERNUX_FRAME_PROFILE
    stageNow = Clock::now();
    m_drawSubMs[11] += std::chrono::duration<double, std::milli>(stageNow - totalStart).count();
    m_drawSubMs[9] += std::chrono::duration<double, std::milli>(stageNow - totalStart).count();
    m_drawShadowIssued += issuedDraws;
#endif
}

// ============================================================================
// Shadow Pipeline Management
// ============================================================================

InxVkCoreModular::ShadowCameraResourceId InxVkCoreModular::CreateShadowCameraResources()
{
    const ShadowCameraResourceId resourceId = m_nextShadowCameraResourceId++;
    m_shadowCameraResources.try_emplace(resourceId);
    return resourceId;
}

void InxVkCoreModular::DestroyShadowCameraResources(ShadowCameraResources &resources) noexcept
{
    const VkDevice device = GetDevice();
    const VmaAllocator allocator = m_backend.Device().GetVmaAllocator();
    auto uniformBuffers = std::make_shared<std::vector<VkBuffer>>(std::move(resources.uniformBuffers));
    auto allocations = std::make_shared<std::vector<VmaAllocation>>(std::move(resources.allocations));
    auto streamFrames =
        std::make_shared<std::vector<ShadowCameraResources::StreamFrame>>(std::move(resources.streamFrames));
    resources.mappedPointers.clear();
    resources.descriptorSets.clear();
    if (device != VK_NULL_HANDLE) {
        auto &descriptorManager = m_backend.Device().GetRhiDevice().GetDescriptorManager();
        for (const auto &lease : resources.descriptorLeases)
            descriptorManager.Retire(lease);
        for (const auto &lease : resources.streamDescriptorLeases)
            descriptorManager.Retire(lease);
    }
    resources.descriptorLeases.clear();
    resources.streamDescriptorLeases.clear();
    resources.staticSubmissionCache = {};
    if (allocator != VK_NULL_HANDLE) {
        m_deletionQueue.Retire([allocator, uniformBuffers, allocations, streamFrames]() mutable {
            for (size_t index = 0; index < uniformBuffers->size(); ++index) {
                if ((*uniformBuffers)[index] != VK_NULL_HANDLE)
                    vmaDestroyBuffer(allocator, (*uniformBuffers)[index], (*allocations)[index]);
            }
            streamFrames->clear();
        });
    }
}

void InxVkCoreModular::DestroyShadowCameraResources(ShadowCameraResourceId resourceId) noexcept
{
    auto found = m_shadowCameraResources.find(resourceId);
    if (found == m_shadowCameraResources.end())
        return;
    DestroyShadowCameraResources(found->second);
    m_shadowCameraResources.erase(found);
}

bool InxVkCoreModular::EnsureShadowCameraResources(ShadowCameraResourceId resourceId)
{
    if (resourceId == 0 || m_perViewDescSetLayout == VK_NULL_HANDLE || m_globalsDescSetLayout == VK_NULL_HANDLE)
        return false;
    auto found = m_shadowCameraResources.find(resourceId);
    if (found == m_shadowCameraResources.end())
        return false;
    ShadowCameraResources &resources = found->second;
    if (!resources.descriptorSets.empty() && resources.streamFrames.size() == m_maxFramesInFlight)
        return true;

    const uint32_t totalSets = m_maxFramesInFlight * lighting::MaxShadowViews;
    auto &descriptorManager = m_backend.Device().GetRhiDevice().GetDescriptorManager();
    resources.descriptorSets.resize(totalSets, VK_NULL_HANDLE);
    resources.descriptorLeases.reserve(totalSets);
    for (uint32_t index = 0; index < totalSets; ++index) {
        auto lease = descriptorManager.Allocate(m_perViewDescSetLayout, vk::DescriptorArena::ViewPersistent);
        if (!lease.IsValid()) {
            INXLOG_ERROR("Failed to allocate camera-local shadow descriptor set ", index);
            DestroyShadowCameraResources(resources);
            return false;
        }
        resources.descriptorSets[index] = lease.set;
        resources.descriptorLeases.push_back(lease);
    }

    const VkDeviceSize uboSize = m_shadowUniformParametersOffset + sizeof(ShadowPassParameters);
    resources.uniformBuffers.resize(totalSets, VK_NULL_HANDLE);
    resources.allocations.resize(totalSets, VK_NULL_HANDLE);
    resources.mappedPointers.resize(totalSets, nullptr);
    const VmaAllocator allocator = m_backend.Device().GetVmaAllocator();
    for (uint32_t index = 0; index < totalSets; ++index) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = uboSize;
        bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo memoryInfo{};
        memoryInfo.usage = VMA_MEMORY_USAGE_AUTO;
        memoryInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        memoryInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VmaAllocationInfo allocation{};
        if (vmaCreateBuffer(allocator, &bufferInfo, &memoryInfo, &resources.uniformBuffers[index],
                            &resources.allocations[index], &allocation) != VK_SUCCESS) {
            INXLOG_ERROR("Failed to create camera-local shadow UBO");
            DestroyShadowCameraResources(resources);
            return false;
        }
        resources.mappedPointers[index] = allocation.pMappedData;
        const std::array<VkDescriptorBufferInfo, 2> infos{
            {{resources.uniformBuffers[index], 0, sizeof(UniformBufferObject)},
             {resources.uniformBuffers[index], m_shadowUniformParametersOffset, sizeof(ShadowPassParameters)}}};
        std::array<VkWriteDescriptorSet, 2> writes{};
        for (uint32_t slot = 0; slot < writes.size(); ++slot) {
            writes[slot].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[slot].dstSet = resources.descriptorSets[index];
            writes[slot].dstBinding = slot == 0 ? 5u : 7u;
            writes[slot].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[slot].descriptorCount = 1;
            writes[slot].pBufferInfo = &infos[slot];
        }
        vkUpdateDescriptorSets(GetDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    std::vector<VkDescriptorSet> streamDescriptorSets(m_maxFramesInFlight, VK_NULL_HANDLE);
    resources.streamDescriptorLeases.reserve(m_maxFramesInFlight);
    for (uint32_t index = 0; index < m_maxFramesInFlight; ++index) {
        auto lease = descriptorManager.Allocate(m_globalsDescSetLayout, vk::DescriptorArena::ViewPersistent);
        if (!lease.IsValid()) {
            INXLOG_ERROR("Failed to allocate camera-local shadow stream descriptor set ", index);
            DestroyShadowCameraResources(resources);
            return false;
        }
        streamDescriptorSets[index] = lease.set;
        resources.streamDescriptorLeases.push_back(lease);
    }

    resources.streamFrames.resize(m_maxFramesInFlight);
    for (uint32_t frameIndex = 0; frameIndex < m_maxFramesInFlight; ++frameIndex) {
        resources.streamFrames[frameIndex].descriptorSet = streamDescriptorSets[frameIndex];
        if (!EnsureShadowCameraStreamCapacity(resources, frameIndex, INSTANCE_BUFFER_INITIAL_CAPACITY,
                                              SKIN_PALETTE_BUFFER_INITIAL_CAPACITY)) {
            INXLOG_ERROR("Failed to create camera-local shadow instance streams for frame ", frameIndex);
            DestroyShadowCameraResources(resources);
            return false;
        }
    }
    return true;
}

void InxVkCoreModular::UpdateShadowCameraStreamDescriptor(ShadowCameraResources::StreamFrame &frame)
{
    if (frame.descriptorSet == VK_NULL_HANDLE || frame.globalsBuffer == VK_NULL_HANDLE || !frame.instanceBuffer ||
        !frame.skinInstanceBuffer || !frame.skinPaletteBuffer)
        return;

    std::array<VkDescriptorBufferInfo, 7> infos{};
    infos[0] = {frame.globalsBuffer, 0, sizeof(EngineGlobalsUBO)};
    infos[1] = {frame.instanceBuffer->GetBuffer(), 0, VK_WHOLE_SIZE};
    infos[2] = {frame.skinInstanceBuffer->GetBuffer(), 0, VK_WHOLE_SIZE};
    infos[3] = {frame.skinPaletteBuffer->GetBuffer(), 0, VK_WHOLE_SIZE};
    infos[4] = {frame.auxiliaryBuffer->GetBuffer(), 0, VK_WHOLE_SIZE};

    infos[5] = infos[1];
    infos[6] = infos[1];
    std::array<VkWriteDescriptorSet, 7> writes{};
    for (uint32_t binding = 0; binding < writes.size(); ++binding) {
        writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[binding].dstSet = frame.descriptorSet;
        writes[binding].dstBinding = binding;
        writes[binding].descriptorType =
            binding == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[binding].descriptorCount = 1;
        writes[binding].pBufferInfo = &infos[binding];
    }
    vkUpdateDescriptorSets(GetDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

bool InxVkCoreModular::EnsureShadowCameraStreamCapacity(ShadowCameraResources &resources, uint32_t frameIndex,
                                                        size_t instanceCount, size_t skinPaletteCount)
{
    if (frameIndex >= resources.streamFrames.size())
        return false;
    auto &frame = resources.streamFrames[frameIndex];
    const bool growInstances = !frame.instanceBuffer || !frame.skinInstanceBuffer ||
                               frame.instanceCapacity < std::max<size_t>(instanceCount, 1);
    const bool growPalette =
        !frame.skinPaletteBuffer || frame.skinPaletteCapacity < std::max<size_t>(skinPaletteCount, 1);
    if (!growInstances && !growPalette)
        return true;

    size_t newInstanceCapacity = std::max<size_t>(frame.instanceCapacity, INSTANCE_BUFFER_INITIAL_CAPACITY);
    while (newInstanceCapacity < std::max<size_t>(instanceCount, 1))
        newInstanceCapacity *= 2;
    size_t newPaletteCapacity = std::max<size_t>(frame.skinPaletteCapacity, SKIN_PALETTE_BUFFER_INITIAL_CAPACITY);
    while (newPaletteCapacity < std::max<size_t>(skinPaletteCount, 1))
        newPaletteCapacity *= 2;

    auto newInstances = m_resourceManager.CreateStorageBuffer(newInstanceCapacity * sizeof(glm::mat4), false);
    auto newSkinInstances =
        m_resourceManager.CreateStorageBuffer(newInstanceCapacity * sizeof(GPUSkinInstanceData), false);
    auto newSkinPalette = m_resourceManager.CreateStorageBuffer(newPaletteCapacity * sizeof(glm::mat4), false);
    auto newAuxiliary = m_resourceManager.CreateStorageBuffer(newInstanceCapacity * sizeof(GPUInstanceAuxData), false);
    if (!newInstances || !newSkinInstances || !newSkinPalette || !newAuxiliary)
        return false;

    void *newInstanceMapped = newInstances->Map();
    void *newSkinInstanceMapped = newSkinInstances->Map();
    void *newSkinPaletteMapped = newSkinPalette->Map();
    void *newAuxiliaryMapped = newAuxiliary->Map();
    if (!newInstanceMapped || !newSkinInstanceMapped || !newSkinPaletteMapped || !newAuxiliaryMapped)
        return false;

    if (frame.instanceBuffer && frame.instanceMapped && frame.instanceWriteOffset > 0) {
        std::memcpy(newInstanceMapped, frame.instanceMapped,
                    static_cast<size_t>(frame.instanceWriteOffset) * sizeof(glm::mat4));
    }
    if (frame.skinInstanceBuffer && frame.skinInstanceMapped && frame.instanceWriteOffset > 0) {
        std::memcpy(newSkinInstanceMapped, frame.skinInstanceMapped,
                    static_cast<size_t>(frame.instanceWriteOffset) * sizeof(GPUSkinInstanceData));
    }
    if (frame.skinPaletteBuffer && frame.skinPaletteMapped && frame.skinPaletteWriteOffset > 0) {
        std::memcpy(newSkinPaletteMapped, frame.skinPaletteMapped,
                    static_cast<size_t>(frame.skinPaletteWriteOffset) * sizeof(glm::mat4));
    }

    if (frame.auxiliaryBuffer && frame.instanceWriteOffset > 0)
        std::memcpy(newAuxiliaryMapped, frame.auxiliaryMapped,
                    static_cast<size_t>(frame.instanceWriteOffset) * sizeof(GPUInstanceAuxData));
    frame.auxiliaryBuffer = std::move(newAuxiliary);
    frame.auxiliaryMapped = newAuxiliaryMapped;
    frame.instanceBuffer = std::move(newInstances);
    frame.skinInstanceBuffer = std::move(newSkinInstances);
    frame.skinPaletteBuffer = std::move(newSkinPalette);
    frame.instanceMapped = newInstanceMapped;
    frame.skinInstanceMapped = newSkinInstanceMapped;
    frame.skinPaletteMapped = newSkinPaletteMapped;
    frame.instanceCapacity = newInstanceCapacity;
    frame.skinPaletteCapacity = newPaletteCapacity;
    UpdateShadowCameraStreamDescriptor(frame);
    return true;
}

bool InxVkCoreModular::EnsureShadowPipeline(VkFormat depthFormat)
{
    if (m_shadowPipelineReady)
        return true;

    VkDevice device = GetDevice();
    if (device == VK_NULL_HANDLE || depthFormat == VK_FORMAT_UNDEFINED)
        return false;

    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(m_backend.Device().GetPhysicalDevice(), &properties);
    const VkDeviceSize alignment = properties.limits.minUniformBufferOffsetAlignment;
    m_shadowUniformParametersOffset = (sizeof(UniformBufferObject) + alignment - 1) & ~(alignment - 1);

    // --- Create shadow depth sampler ---
    if (m_shadowDepthSampler == VK_NULL_HANDLE) {
        if (!CreateShadowDepthSampler()) {
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
    // Shaders gather four raw depth texels and compare before interpolation.
    // Keeping this a regular depth sampler also permits the engine's fully-lit
    // fallback descriptor when a graph has no shadow pass.
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
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

    for (auto &[resourceId, resources] : m_shadowCameraResources) {
        (void)resourceId;
        DestroyShadowCameraResources(resources);
    }
    m_shadowCameraResources.clear();
    const VkSampler retiredDepthSampler = std::exchange(m_shadowDepthSampler, VK_NULL_HANDLE);

    auto retiredPipelines = std::move(m_shadowPipelineCache);
    m_shadowPipelineCache.clear();
    m_deletionQueue.Retire([device, retiredDepthSampler, retiredPipelines = std::move(retiredPipelines)]() mutable {
        for (const auto &[key, pipeline] : retiredPipelines) {
            (void)key;
            vkDestroyPipeline(device, pipeline, nullptr);
        }
        retiredPipelines.clear();
        if (retiredDepthSampler != VK_NULL_HANDLE)
            vkDestroySampler(device, retiredDepthSampler, nullptr);
    });
    m_shadowPipelineReady = false;
}

// ============================================================================
// Per-object buffer management
// ============================================================================

void InxVkCoreModular::RegisterMeshUploadWork(FrameWorkOwner &owner)
{
    FrameWorkSource source;
    source.observe = [this] { PumpPendingMeshUploads(); };
    source.refresh = source.observe;
    source.describe = [this, &owner] {
        FrameWorkFacts facts;
        facts.pending = !m_pendingSharedMeshBuffers.empty();
        for (const auto &[key, upload] : m_pendingSharedMeshBuffers) {
            if (m_requiredMeshOnly && upload.requiredFrame != m_ensureFrameCounter)
                continue;
            const bool vertex = upload.vertexUpload->GetSubmittedBytes() != upload.vertexUpload->GetSize();
            const auto &ticket = vertex ? upload.vertexUpload : upload.indexUpload;
            const auto unit = vertex ? sizeof(float) : key.IndexBytes();
            const auto order = [this](uint64_t token) { return std::pair(token <= m_lastMeshGpuToken, token); };
            if (m_resourceManager.PlanBufferUploadBlock(*ticket, owner, unit) &&
                (!facts.gpu.ready || order(upload.token) < order(facts.gpu.token)))
                facts.gpu = {upload.token, 0, true, false};
        }
        return facts;
    };
    source.advance = [this, &owner](bool, uint64_t token) {
        for (auto &[key, upload] : m_pendingSharedMeshBuffers) {
            if (upload.token != token)
                continue;
            m_lastMeshGpuToken = token;
            const bool vertex = upload.vertexUpload->GetSubmittedBytes() != upload.vertexUpload->GetSize();
            const auto &ticket = vertex ? upload.vertexUpload : upload.indexUpload;
            const auto unit = vertex ? sizeof(float) : key.IndexBytes();
            const auto bytes = m_resourceManager.PlanBufferUploadBlock(*ticket, owner, unit);
            return bytes && AppendMeshUploadBlock(key, upload, vertex, bytes, &owner);
        }
        return false;
    };
    m_meshWorkSource = owner.AddSource(std::move(source));
}

bool InxVkCoreModular::AdvanceRequiredMeshUploads(const std::function<FrameGpuAvailability()> &gpuAvailability)
{
    const auto required = [this](const auto &entry) { return entry.second.requiredFrame == m_ensureFrameCounter; };
    if (!std::any_of(m_pendingSharedMeshBuffers.begin(), m_pendingSharedMeshBuffers.end(), required))
        return false;
    struct RequiredScope
    {
        bool &only;
        ~RequiredScope()
        {
            only = false;
        }
    } scope{m_requiredMeshOnly};
    m_requiredMeshOnly = true;
    // The source retains all pending ownership. Only this admission selects
    // the previous opportunity's actual view inputs; companions stay background.
    (void)m_frameWorkOwner->AdvanceRequired(m_meshWorkSource, gpuAvailability);
    return std::any_of(m_pendingSharedMeshBuffers.begin(), m_pendingSharedMeshBuffers.end(), required);
}

bool InxVkCoreModular::AppendMeshUploadBlock(const SharedMeshKey &key, PendingSharedMeshBuffers &upload, bool vertex,
                                             VkDeviceSize bytes, FrameWorkOwner *owner)
{
    const auto &ticket = vertex ? upload.vertexUpload : upload.indexUpload;
    const auto offset = ticket->GetSubmittedBytes();
    const auto unit = vertex ? sizeof(float) : key.IndexBytes();
    if (!vertex && key.indexFormat == MeshIndexFormat::UInt16)
        return m_resourceManager.TryAppendIndex16Upload(ticket, upload.indices->data() + offset / unit, bytes, owner);

    const auto *data = vertex ? reinterpret_cast<const std::byte *>(upload.vertices->data())
                              : reinterpret_cast<const std::byte *>(upload.indices->data());
    if (!m_resourceManager.TryAppendBufferUpload(ticket, data + offset, bytes, owner, bytes == unit, unit,
                                                 {vertex ? upload.indexUpload.get() : nullptr}))
        return false;
    if (vertex && offset == 0) {
        ++m_submittedMeshUploadCount;
        m_asyncMeshUploadCount += ticket->IsAsync();
    }
    return true;
}

void InxVkCoreModular::PrepareSceneAssets(const std::vector<DrawCall> &draws)
{
    AdvanceEnsureFrame();
    std::unordered_set<uint64_t> geometry;
    std::unordered_set<InxMaterial *> materials;
    auto *database = AssetRegistry::Instance().GetAssetDatabase();
    for (const auto &draw : draws) {
        if (draw.meshVertices && draw.meshIndices && geometry.insert(draw.objectId).second)
            (void)PrepareObjectBuffers(draw);
        const auto &material = draw.material;
        if (!material || !materials.insert(material.get()).second)
            continue;
        PrepareMaterialTextureAssets(material);
        for (const auto &[name, property] : material->GetAllProperties()) {
            if (property.type != MaterialPropertyType::Texture2D || material->HasRuntimeTextureOverride(name))
                continue;
            const auto &guid = std::get<std::string>(property.value);
            const auto metadata = database->GetMetaByGuid(guid);
            if (metadata && metadata->GetResourceType() == ResourceType::Texture)
                (void)ResolveTextureForMaterial(guid, name, material->GetTextureSampler(name), true);
        }
    }

    // The explicit scene-open operation can wait for transfers. Reuse the
    // incremental uploader and bound each batch by its staging footprint.
    for (;;) {
        PumpPendingMeshUploads();
        VkDeviceSize remaining = rhi::UploadStagingPoolBytes;
        bool hasWork = false;
        for (auto &[key, upload] : m_pendingSharedMeshBuffers) {
            if (upload.requestedFrame != m_ensureFrameCounter)
                continue;
            hasWork = true;
            for (const bool vertex : {true, false}) {
                const auto &ticket = vertex ? upload.vertexUpload : upload.indexUpload;
                if (ticket->GetSubmittedBytes() == ticket->GetSize())
                    continue;
                if (ticket->IsPending())
                    break;
                const auto bytes = (std::min)(ticket->GetSize() - ticket->GetSubmittedBytes(),
                                              VkDeviceSize{rhi::MaxPooledUploadStagingClass});
                const auto staging = vk::VkResourceManager::GetUploadStagingBytes(bytes);
                if (staging > remaining)
                    break;
                if (AppendMeshUploadBlock(key, upload, vertex, bytes, nullptr))
                    remaining -= staging;
                if (ticket->GetSubmittedBytes() != ticket->GetSize())
                    break;
            }
        }
        if (!hasWork)
            return;
        m_resourceManager.DrainBufferUploads();
        if (m_resourceManager.GetPendingGpuTransferCount())
            throw std::runtime_error("Scene uploads did not finish");
    }
}

void InxVkCoreModular::PumpPendingMeshUploads()
{
    if (m_pendingSharedMeshBuffers.empty())
        return;
    // Pending uploads own CPU source storage until their accepted copies
    // finish. Count those internal owners separately from actual draw/world
    // publications; two asset views can share one immutable geometry owner.
    std::unordered_map<const void *, long> pendingOwners;
    for (const auto &[key, upload] : m_pendingSharedMeshBuffers)
        ++pendingOwners[upload.source.get()];
    auto &registry = AssetRegistry::Instance();
    for (auto pending = m_pendingSharedMeshBuffers.begin(); pending != m_pendingSharedMeshBuffers.end();) {
        const auto &key = pending->first;
        const auto &source = pending->second.source;
        auto &internalOwners = pendingOwners.at(source.get());
        if (key.IsRegistryAsset() && key.runtimeVersion != registry.GetAssetVersion(key.assetGuid) &&
            source.use_count() == internalOwners) {
            --internalOwners;
            m_assetMeshViewKeys.erase({key.assetGuid, key.runtimeVersion, key.geometryView});
            // Accepted copies remain owned by VkResourceManager until their
            // fences retire. Unsubmitted blocks no longer have a consumer.
            pending = m_pendingSharedMeshBuffers.erase(pending);
        } else {
            ++pending;
        }
    }
    m_resourceManager.PollGpuUploads();
    bool publishedRuntimeMesh = false;
    for (auto pending = m_pendingSharedMeshBuffers.begin(); pending != m_pendingSharedMeshBuffers.end();) {
        const auto &key = pending->first;
        const bool vertexReady = m_resourceManager.TryPublishBufferUpload(pending->second.vertexUpload);
        const bool indexReady = m_resourceManager.TryPublishBufferUpload(pending->second.indexUpload);
        if (!vertexReady || !indexReady) {
            ++pending;
            continue;
        }

        SharedMeshBuffers buffers;
        buffers.vertexBuffer = pending->second.vertexUpload->GetBuffer();
        buffers.indexBuffer = pending->second.indexUpload->GetBuffer();
        buffers.vertexCount = key.vertexCount;
        buffers.indexCount = key.indexCount;
        buffers.indexFormat = key.indexFormat;
        PublishSharedMeshBuffers(key, std::move(buffers));
        publishedRuntimeMesh |= key.dynamicObjectId != 0;
        pending = m_pendingSharedMeshBuffers.erase(pending);
        ++m_completedMeshUploadCount;
    }
    // A runtime upload can arrive after its last view stopped using it.
    // Recheck the bindings and retire unclaimed generations after preparation.
    if (publishedRuntimeMesh)
        ++m_objectBufferRevision;
}

bool InxVkCoreModular::QueueSharedMeshUpload(const SharedMeshKey &key, const std::vector<Vertex> &vertices,
                                             const std::vector<uint32_t> &indices, std::shared_ptr<const void> source)
{
    if (const auto pending = m_pendingSharedMeshBuffers.find(key); pending != m_pendingSharedMeshBuffers.end()) {
        pending->second.requestedFrame = m_ensureFrameCounter;
        return true;
    }
    if (key.IsRegistryAsset() && !source)
        throw std::logic_error("Asset mesh upload requires its immutable geometry generation");
    PendingSharedMeshBuffers upload;
    upload.vertices = &vertices;
    upload.indices = &indices;
    upload.source = std::move(source);
    if (key.assetGuid.empty()) {
        const auto bytes = (upload.source ? vertices.capacity() : vertices.size()) * sizeof(Vertex) +
                           (upload.source ? indices.capacity() : indices.size()) * sizeof(uint32_t);
        upload.sourceStorage = m_frameWorkOwner->TryReserveCpuStorage(bytes);
        if (!upload.sourceStorage)
            return false;
    }
    if (!upload.source && key.assetGuid.empty()) {
        // Editor tools borrow mutable arrays for the current render only.
        auto snapshot = std::make_shared<MeshGeometry>();
        snapshot->vertices = vertices;
        snapshot->indices = indices;
        upload.vertices = &snapshot->vertices;
        upload.indices = &snapshot->indices;
        upload.source = std::move(snapshot);
    }
    upload.vertexUpload =
        m_resourceManager.CreateBufferUpload(vertices.size() * sizeof(Vertex), rhi::BufferUsageFlags::Vertex);
    upload.indexUpload =
        m_resourceManager.CreateBufferUpload(indices.size() * key.IndexBytes(), rhi::BufferUsageFlags::Index);
    if (!key.assetGuid.empty()) {
        const AssetMeshViewIdentity identity{key.assetGuid, key.runtimeVersion, key.geometryView};
        const auto [published, inserted] = m_assetMeshViewKeys.emplace(identity, key);
        if (!inserted && !(published->second == key))
            throw std::logic_error("Asset mesh view identity changed inside one runtime generation");
    }
    upload.token = ++m_meshWorkToken;
    upload.requestedFrame = m_ensureFrameCounter;
    m_pendingSharedMeshBuffers.emplace(key, std::move(upload));
    m_frameWorkOwner->Wake(m_meshWorkSource);
    return true;
}

void InxVkCoreModular::QueueHierarchyCompanionView(const std::string &assetGuid, uint64_t runtimeVersion,
                                                   MeshGeometryView currentView)
{
    if (assetGuid.empty())
        return;
    const MeshGeometryView companionView = currentView == MeshGeometryView::MergedModelSpace
                                               ? MeshGeometryView::NodeLocal
                                               : MeshGeometryView::MergedModelSpace;
    const AssetMeshViewIdentity identity{assetGuid, runtimeVersion, companionView};
    if (const auto known = m_assetMeshViewKeys.find(identity); known != m_assetMeshViewKeys.end()) {
        if (m_sharedMeshBuffers.find(known->second) != m_sharedMeshBuffers.end() ||
            m_pendingSharedMeshBuffers.find(known->second) != m_pendingSharedMeshBuffers.end())
            return;
        m_assetMeshViewKeys.erase(known);
    }
    auto &registry = AssetRegistry::Instance();
    if (!registry.IsLoaded(assetGuid) || registry.GetAssetVersion(assetGuid) != runtimeVersion)
        return;
    const uint8_t companionBit = companionView == MeshGeometryView::MergedModelSpace ? uint8_t{1} : uint8_t{2};
    if ((registry.GetMeshGpuViewResidencyMask(assetGuid, runtimeVersion) & companionBit) != 0)
        return;
    const auto mesh = registry.GetAsset<InxMesh>(assetGuid);
    if (!mesh)
        return;
    const auto source = mesh->GetModelSourceGeometry();
    if (!source)
        return;
    const auto geometry = companionView == MeshGeometryView::NodeLocal ? source : mesh->GetGeometrySnapshot();
    if (!geometry || geometry->vertices.empty() || geometry->indices.empty())
        throw std::logic_error("Hierarchical model CPU geometry was released before both GPU views became resident");
    const MeshIndexFormat format =
        ResolveMeshIndexFormat(mesh->GetIndexFormat(), geometry->vertices.size(), geometry->indices);
    const SharedMeshKey key{
        assetGuid, 0, runtimeVersion, geometry->vertices.size(), geometry->indices.size(), format, companionView};
    (void)QueueSharedMeshUpload(key, geometry->vertices, geometry->indices, geometry);
}

InxVkCoreModular::PerObjectBuffers *InxVkCoreModular::PrepareObjectGeometry(const DrawCall &draw)
{
    const auto &vertices = *draw.meshVertices;
    const auto &indices = *draw.meshIndices;
    const auto &assetGuid = draw.meshAssetGuid;
    const auto geometryView = draw.meshGeometryView;
    const auto objectId = draw.objectId;
    const auto runtimeVersion = IsBuiltinMeshGuid(assetGuid) ? uint64_t{1} : draw.meshRuntimeVersion;
    // Small engine-owned overlays need to draw in the same frame as their
    // input. Each geometry revision gets an immutable, host-visible pair;
    // the regular mesh cache keeps prior pairs alive through GPU completion.
    const auto domain = draw.identity.domain;
    const bool smallComponentGizmo =
        domain == RenderDomain::ComponentGizmo &&
        vertices.size() * sizeof(Vertex) + indices.size() * sizeof(uint32_t) <= rhi::InlineBufferUpdateBytes;
    const bool editorOverlay =
        domain == RenderDomain::EditorTool || domain == RenderDomain::EditorGizmo || smallComponentGizmo;
    const auto requestedIndexFormat = editorOverlay ? MeshIndexFormat::UInt32 : draw.meshIndexFormat;
    const bool cpuGeometryReleased = vertices.empty() || indices.empty();
    if (!assetGuid.empty() && runtimeVersion == 0)
        throw std::invalid_argument("Mesh GPU identity requires GUID and runtime version together");
    auto objectIt = m_perObjectBuffers.find(objectId);
    if (objectIt != m_perObjectBuffers.end()) {
        auto &binding = objectIt->second;
        // All views of this object use the same captured draw source. A stamp
        // can represent a deferred replacement that kept the old binding alive.
        if (binding.ensuredOnFrame == m_ensureFrameCounter)
            return binding.requestedGeometryReady ? &binding : nullptr;
        const bool generationMatches = (runtimeVersion == 0 || binding.sharedKey.runtimeVersion == runtimeVersion) &&
                                       binding.sharedKey.assetGuid == assetGuid &&
                                       binding.sharedKey.geometryView == geometryView;
        const bool sourceMatches =
            !draw.forceBufferUpdate && binding.lastVertexPtr == vertices.data() &&
            binding.lastIndexPtr == indices.data() && binding.vertexCount == vertices.size() &&
            binding.indexCount == indices.size() &&
            (requestedIndexFormat == MeshIndexFormat::Auto || binding.indexFormat == requestedIndexFormat);
        if (generationMatches && (cpuGeometryReleased || sourceMatches)) {
            binding.ensuredOnFrame = m_ensureFrameCounter;
            binding.requestedGeometryReady = true;
            if (cpuGeometryReleased) {
                if (const auto shared = m_sharedMeshBuffers.find(binding.sharedKey);
                    shared != m_sharedMeshBuffers.end())
                    shared->second.lastUsedFrame = m_ensureFrameCounter;
            }
            return &binding;
        }
    }

    SharedMeshKey sharedKey;
    if (!assetGuid.empty()) {
        const AssetMeshViewIdentity identity{assetGuid, runtimeVersion, geometryView};
        const auto known = m_assetMeshViewKeys.find(identity);
        if (known != m_assetMeshViewKeys.end()) {
            sharedKey = known->second;
            if (!cpuGeometryReleased &&
                (sharedKey.vertexCount != vertices.size() || sharedKey.indexCount != indices.size()))
                throw std::logic_error("Asset mesh geometry changed without publishing a new runtime generation");
            if (requestedIndexFormat != MeshIndexFormat::Auto && requestedIndexFormat != sharedKey.indexFormat)
                throw std::logic_error("Asset mesh index format changed without publishing a new runtime generation");
        } else {
            if (cpuGeometryReleased)
                throw std::runtime_error("Mesh GPU generation has no published geometry view identity");
            sharedKey = {assetGuid,      0,
                         runtimeVersion, vertices.size(),
                         indices.size(), ResolveMeshIndexFormat(requestedIndexFormat, vertices.size(), indices),
                         geometryView};
        }
    } else {
        // Mutable draw producers supply a revision. Unversioned runtime
        // geometry has fixed topology and storage for this draw identity.
        sharedKey = {{},
                     draw.meshUsesDetachedStorage ? 0 : objectId,
                     runtimeVersion,
                     vertices.size(),
                     indices.size(),
                     ResolveMeshIndexFormat(requestedIndexFormat, vertices.size(), indices),
                     geometryView};
    }

    auto sharedIt = m_sharedMeshBuffers.find(sharedKey);
    if (editorOverlay) {
        if (sharedIt == m_sharedMeshBuffers.end()) {
            SharedMeshBuffers buffers;
            buffers.vertexBuffer = m_resourceManager.CreateHostDrawBuffer(
                vertices.data(), vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            buffers.indexBuffer = m_resourceManager.CreateHostDrawBuffer(
                indices.data(), indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            buffers.vertexCount = vertices.size();
            buffers.indexCount = indices.size();
            PublishSharedMeshBuffers(sharedKey, std::move(buffers));
            sharedIt = m_sharedMeshBuffers.find(sharedKey);
        }
    } else if (sharedKey.dynamicObjectId) {
        // Finish one captured revision per object. Continually changing
        // procedural geometry can publish intermediate complete revisions.
        auto active = m_runtimeMeshUploads.find(objectId);
        if (active != m_runtimeMeshUploads.end() && !(active->second.key == sharedKey)) {
            const auto pending = m_pendingSharedMeshBuffers.find(active->second.key);
            if (pending != m_pendingSharedMeshBuffers.end() && pending->second.vertexUpload->GetSubmittedBytes() == 0) {
                m_pendingSharedMeshBuffers.erase(pending);
                m_runtimeMeshUploads.erase(active);
                active = m_runtimeMeshUploads.end();
            }
        }
        if (active != m_runtimeMeshUploads.end()) {
            active->second.requestedFrame = m_ensureFrameCounter;
            if (auto pending = m_pendingSharedMeshBuffers.find(active->second.key);
                pending != m_pendingSharedMeshBuffers.end())
                pending->second.requiredFrame = m_ensureFrameCounter;
            if (auto completed = m_sharedMeshBuffers.find(active->second.key); completed != m_sharedMeshBuffers.end()) {
                if (sharedIt == m_sharedMeshBuffers.end())
                    sharedIt = completed;
                m_runtimeMeshUploads.erase(active);
                active = m_runtimeMeshUploads.end();
            }
        }
        const bool requestedReady = sharedIt != m_sharedMeshBuffers.end() && sharedIt->first == sharedKey;
        if (!requestedReady && active == m_runtimeMeshUploads.end() &&
            QueueSharedMeshUpload(sharedKey, vertices, indices, draw.meshDataOwner))
            m_runtimeMeshUploads.emplace(objectId, RuntimeMeshUpload{sharedKey, m_ensureFrameCounter});
    } else if (!cpuGeometryReleased) {
        if (sharedIt == m_sharedMeshBuffers.end())
            (void)QueueSharedMeshUpload(sharedKey, vertices, indices, draw.meshDataOwner);
        if (sharedKey.IsRegistryAsset())
            QueueHierarchyCompanionView(assetGuid, runtimeVersion, geometryView);
    }
    if (auto pending = m_pendingSharedMeshBuffers.find(sharedKey); pending != m_pendingSharedMeshBuffers.end())
        pending->second.requiredFrame = m_ensureFrameCounter;
    if (sharedIt == m_sharedMeshBuffers.end()) {
        // Read/Write-disabled Player assets rebind the resident generation.
        if (cpuGeometryReleased)
            throw std::runtime_error("Mesh CPU geometry was released before its GPU generation became resident");
        // Render with the last complete pair while the requested revision is
        // private. A view with deferred geometry cannot reuse its bindings.
        if (objectIt != m_perObjectBuffers.end()) {
            objectIt->second.ensuredOnFrame = m_ensureFrameCounter;
            objectIt->second.requestedGeometryReady = false;
        }
        return nullptr;
    }

    const bool requestedReady = sharedIt->first == sharedKey;
    PerObjectBuffers objectBuffers;
    objectBuffers.vertexBuffer = sharedIt->second.vertexBuffer;
    objectBuffers.indexBuffer = sharedIt->second.indexBuffer;
    objectBuffers.vertexCount = sharedIt->second.vertexCount;
    objectBuffers.indexCount = sharedIt->second.indexCount;
    objectBuffers.indexFormat = sharedIt->second.indexFormat;
    objectBuffers.sharedKey = sharedIt->first;
    objectBuffers.lastVertexPtr = requestedReady ? vertices.data() : nullptr;
    objectBuffers.lastIndexPtr = requestedReady ? indices.data() : nullptr;
    objectBuffers.ensuredOnFrame = m_ensureFrameCounter;
    objectBuffers.requestedGeometryReady = requestedReady;

    m_residentVertexObjects.erase(objectId);
    auto binding = m_perObjectBuffers.insert_or_assign(objectId, std::move(objectBuffers)).first;
    ++m_objectBufferRevision;
    sharedIt->second.lastUsedFrame = m_ensureFrameCounter;
    (void)TrimMeshGpuBudget();
    return requestedReady ? &binding->second : nullptr;
}

bool InxVkCoreModular::PrepareObjectBuffers(const DrawCall &draw)
{
    if (draw.meshAssetGuid.empty() && (draw.meshVertices->empty() || draw.meshIndices->empty()))
        return true;
    auto *binding = PrepareObjectGeometry(draw);
    if (!binding)
        return false;
    const auto &buffer = draw.meshVertexBuffer;
    if (binding->residentVertexBuffer == buffer)
        return true;
    if (buffer) {
        if (buffer->GetHost().device.GetDeviceId() != m_backend.Device().GetDeviceId())
            throw std::invalid_argument("Resident mesh vertex buffer belongs to another rendering device");
        const uint64_t requiredBytes = static_cast<uint64_t>(binding->vertexCount) * sizeof(Vertex);
        const auto &desc = buffer->GetDesc();
        if (desc.scalarType != rhi::ComputeScalarType::Float32 || desc.lanes != 1 ||
            buffer->GetByteSize() < requiredBytes || buffer->GetByteSize() % sizeof(Vertex) != 0)
            throw std::invalid_argument(
                "Resident mesh vertex buffer must use canonical Vertex storage with capacity >= vertex_count");
        m_residentVertexObjects.insert(draw.objectId);
    } else {
        m_residentVertexObjects.erase(draw.objectId);
    }
    binding->residentVertexBuffer = buffer;
    binding->residentVertexHandle = VK_NULL_HANDLE;
    ++m_objectBufferRevision;
    InvalidateDrawBufferCaches();
    return true;
}

void InxVkCoreModular::CaptureResidentGeometryReads(const std::vector<DrawCall> &draws,
                                                    const std::vector<DrawCall> &shadowDraws)
{
    auto &device = m_backend.Device().GetRhiDevice();
    bool changed = false;
    const auto capture = [&](const auto &list) {
        uint64_t previous = 0;
        for (const auto &draw : list) {
            if (draw.devicePose.value)
                CapturePoseRead(draw.devicePose.value);
            if (draw.objectId == previous)
                continue;
            previous = draw.objectId;
            if (m_residentVertexObjects.find(draw.objectId) == m_residentVertexObjects.end())
                continue;
            auto &buffers = m_perObjectBuffers.at(draw.objectId);
            const auto &buffer = buffers.residentVertexBuffer;
            const auto allocation = buffer->GetAllocation();
            const auto native = device.Resolve(allocation->GetBuffer());
            if (native == VK_NULL_HANDLE)
                throw std::runtime_error("Resident mesh vertex buffer has no Vulkan allocation");
            changed |= buffers.residentVertexHandle != native;
            buffers.residentVertexHandle = native;
            if (m_frameResidentVertexReads.insert(allocation.get()).second) {
                RegisterFrameBufferReadDependency(allocation->GetLastWriteSubmission(),
                                                  rhi::PipelineStage::VertexInput);
                m_frameSubmission.Retain(
                    std::make_shared<rhi::ComputeBufferState>(buffer->GetDesc(), allocation, true));
            }
        }
    };
    capture(draws);
    if (&shadowDraws != &draws)
        capture(shadowDraws);
    if (changed) {
        // The authored draw lists still refer to the same object bindings.
        // Only their resolved vertex handles and derived draw caches changed.
        m_drawListBufferRevision = 0;
        m_shadowListBufferRevision = 0;
        InvalidateDrawBufferCaches();
    }
}

void InxVkCoreModular::KeepObjectBuffers(const std::vector<DrawCall> &drawCalls)
{
    uint64_t previous = 0;
    for (const auto &draw : drawCalls) {
        if (draw.objectId == previous)
            continue;
        previous = draw.objectId;
        const auto binding = m_perObjectBuffers.find(draw.objectId);
        if (binding != m_perObjectBuffers.end())
            binding->second.ensuredOnFrame = m_ensureFrameCounter;
    }
}

size_t InxVkCoreModular::CleanupUnusedBuffersByFrameStamp()
{
    bool anyRemoved = false;
    for (auto it = m_perObjectBuffers.begin(); it != m_perObjectBuffers.end();) {
        if (it->second.ensuredOnFrame != m_ensureFrameCounter) {
            m_residentVertexObjects.erase(it->first);
            auto shared = m_sharedMeshBuffers.find(it->second.sharedKey);
            if (shared != m_sharedMeshBuffers.end())
                shared->second.lastUsedFrame = m_ensureFrameCounter;
            it = m_perObjectBuffers.erase(it);
            anyRemoved = true;
        } else {
            ++it;
        }
    }

    if (anyRemoved)
        ++m_objectBufferRevision;

    RetireUnusedRuntimeMeshBuffers();
    if (anyRemoved)
        (void)TrimMeshGpuBudget();

    return m_perObjectBuffers.size();
}

void InxVkCoreModular::RetireUnusedRuntimeMeshBuffers()
{
    for (auto active = m_runtimeMeshUploads.begin(); active != m_runtimeMeshUploads.end();) {
        if (active->second.requestedFrame == m_ensureFrameCounter) {
            ++active;
            continue;
        }
        m_pendingSharedMeshBuffers.erase(active->second.key);
        active = m_runtimeMeshUploads.erase(active);
    }
    for (auto pending = m_pendingSharedMeshBuffers.begin(); pending != m_pendingSharedMeshBuffers.end();) {
        const auto &key = pending->first;
        if (key.assetGuid.empty() && !key.dynamicObjectId && pending->second.requestedFrame != m_ensureFrameCounter &&
            pending->second.source.use_count() == 1)
            pending = m_pendingSharedMeshBuffers.erase(pending);
        else
            ++pending;
    }
    // Runtime geometry belongs to exactly one object binding. Accepted GPU
    // work keeps its own lifetime after the logical cache entry is retired.
    for (auto entry = m_sharedMeshBuffers.begin(); entry != m_sharedMeshBuffers.end();) {
        const auto &key = entry->first;
        if (key.dynamicObjectId == 0) {
            ++entry;
            continue;
        }
        const auto object = m_perObjectBuffers.find(key.dynamicObjectId);
        if (object != m_perObjectBuffers.end() && object->second.sharedKey == key) {
            ++entry;
            continue;
        }
        const auto pending = m_runtimeMeshUploads.find(key.dynamicObjectId);
        if (pending != m_runtimeMeshUploads.end() && pending->second.key == key) {
            ++entry;
            continue;
        }
        SharedMeshBuffers retired = std::move(entry->second);
        entry = m_sharedMeshBuffers.erase(entry);
        RetireSharedMeshBuffers(std::move(retired), false);
    }
}

void InxVkCoreModular::InvalidateMeshCache(const std::string &meshGuid)
{
    if (meshGuid.empty())
        throw std::invalid_argument("Mesh GPU invalidation requires an asset identity");
    for (auto entry = m_sharedMeshBuffers.begin(); entry != m_sharedMeshBuffers.end();) {
        if (entry->first.assetGuid != meshGuid) {
            ++entry;
            continue;
        }
        SharedMeshBuffers retired = std::move(entry->second);
        entry = m_sharedMeshBuffers.erase(entry);
        RetireSharedMeshBuffers(std::move(retired), false);
    }
    for (auto entry = m_pendingSharedMeshBuffers.begin(); entry != m_pendingSharedMeshBuffers.end();) {
        if (entry->first.assetGuid == meshGuid)
            entry = m_pendingSharedMeshBuffers.erase(entry);
        else
            ++entry;
    }
    for (auto entry = m_assetMeshViewKeys.begin(); entry != m_assetMeshViewKeys.end();) {
        if (entry->first.assetGuid == meshGuid)
            entry = m_assetMeshViewKeys.erase(entry);
        else
            ++entry;
    }
    ++m_objectBufferRevision;
}

void InxVkCoreModular::PublishSharedMeshBuffers(const SharedMeshKey &key, SharedMeshBuffers buffers)
{
    if (!buffers.vertexBuffer || !buffers.indexBuffer || !buffers.vertexBuffer->IsValid() ||
        !buffers.indexBuffer->IsValid())
        throw std::invalid_argument("Cannot publish invalid shared mesh buffers");
    const uint64_t vertexBytes = buffers.vertexBuffer->GetSize();
    const uint64_t indexBytes = buffers.indexBuffer->GetSize();
    if (vertexBytes == 0 || indexBytes == 0 || indexBytes > std::numeric_limits<uint64_t>::max() - vertexBytes)
        throw std::overflow_error("Invalid shared mesh GPU byte size");
    buffers.residentBytes = vertexBytes + indexBytes;
    buffers.lastUsedFrame = m_ensureFrameCounter;

    auto existing = m_sharedMeshBuffers.find(key);
    if (existing != m_sharedMeshBuffers.end()) {
        SharedMeshBuffers retired = std::move(existing->second);
        m_sharedMeshBuffers.erase(existing);
        RetireSharedMeshBuffers(std::move(retired), false);
    }
    if (buffers.residentBytes > std::numeric_limits<uint64_t>::max() - m_meshGpuResidentBytes)
        throw std::overflow_error("Mesh GPU residency byte counter overflow");
    m_meshGpuResidentBytes += buffers.residentBytes;
    const auto [inserted, didInsert] = m_sharedMeshBuffers.emplace(key, std::move(buffers));
    (void)inserted;
    if (!didInsert)
        throw std::logic_error("Shared mesh cache rejected a unique key");
    if (key.IsRegistryAsset()) {
        auto &registry = AssetRegistry::Instance();
        registry.MarkMeshGpuViewResident(key.assetGuid, key.runtimeVersion, key.geometryView);
        // A packaged Player retains immutable GPU storage as the authoritative
        // render copy. Hierarchical assets release only after both merged and
        // node-local view identities have been published for this generation.
        if (IsPackagedPlayerRuntime())
            (void)registry.ReleaseMeshCpuGeometry(key.assetGuid, key.runtimeVersion);
    }
    // Published asset generations are not reusable content-cache entries.
    // Existing draws retain their leases; retire superseded cache ownership
    // through the same GPU completion queue used by ordinary eviction.
    if (!key.assetGuid.empty()) {
        for (auto entry = m_sharedMeshBuffers.begin(); entry != m_sharedMeshBuffers.end();) {
            if (entry->first.assetGuid == key.assetGuid && entry->first.runtimeVersion < key.runtimeVersion) {
                SharedMeshBuffers retired = std::move(entry->second);
                entry = m_sharedMeshBuffers.erase(entry);
                RetireSharedMeshBuffers(std::move(retired), false);
            } else {
                ++entry;
            }
        }
        for (auto entry = m_assetMeshViewKeys.begin(); entry != m_assetMeshViewKeys.end();) {
            if (entry->first.assetGuid == key.assetGuid && entry->first.runtimeVersion < key.runtimeVersion)
                entry = m_assetMeshViewKeys.erase(entry);
            else
                ++entry;
        }
    }
}

void InxVkCoreModular::RetireSharedMeshBuffers(SharedMeshBuffers buffers, bool eviction)
{
    if (!buffers.vertexBuffer || !buffers.indexBuffer || buffers.residentBytes == 0)
        throw std::logic_error("Cannot retire invalid shared mesh residency");
    m_retiredMeshLeases.push_back({buffers.vertexBuffer, buffers.indexBuffer, buffers.residentBytes});
    auto retired = std::make_shared<SharedMeshBuffers>(std::move(buffers));
    m_deletionQueue.Retire([retired = std::move(retired)]() mutable {
        retired->vertexBuffer.reset();
        retired->indexBuffer.reset();
    });
    if (eviction)
        ++m_meshGpuEvictionCount;
}

void InxVkCoreModular::SweepRetiredMeshLeases() const
{
    size_t writeIndex = 0;
    for (size_t index = 0; index < m_retiredMeshLeases.size(); ++index) {
        const auto &lease = m_retiredMeshLeases[index];
        if (lease.vertexBuffer.expired() && lease.indexBuffer.expired()) {
            if (lease.residentBytes > m_meshGpuResidentBytes)
                throw std::logic_error("Mesh GPU residency byte counter underflow");
            m_meshGpuResidentBytes -= lease.residentBytes;
            continue;
        }
        if (writeIndex != index)
            m_retiredMeshLeases[writeIndex] = std::move(m_retiredMeshLeases[index]);
        ++writeIndex;
    }
    m_retiredMeshLeases.resize(writeIndex);
}

uint64_t InxVkCoreModular::GetMeshGpuResidentBytes() const
{
    SweepRetiredMeshLeases();
    return m_meshGpuResidentBytes;
}

size_t InxVkCoreModular::GetRetiredMeshGpuLeaseCount() const
{
    SweepRetiredMeshLeases();
    return m_retiredMeshLeases.size();
}

std::vector<GpuAssetResidencyRecord> InxVkCoreModular::GetAssetMeshGpuResidency() const
{
    std::vector<GpuAssetResidencyRecord> records;
    records.reserve(m_sharedMeshBuffers.size() + m_pendingSharedMeshBuffers.size());
    for (const auto &[key, buffers] : m_sharedMeshBuffers) {
        if (key.assetGuid.empty())
            continue;
        const bool pinned = (buffers.vertexBuffer && buffers.vertexBuffer.use_count() != 1) ||
                            (buffers.indexBuffer && buffers.indexBuffer.use_count() != 1);
        records.push_back({key.assetGuid, key.runtimeVersion, GpuAssetDomain::Mesh, buffers.residentBytes,
                           buffers.lastUsedFrame, false, pinned});
    }
    for (const auto &[key, uploads] : m_pendingSharedMeshBuffers) {
        if (key.assetGuid.empty())
            continue;
        const uint64_t bytes = uploads.vertexUpload->GetDestinationBytes() + uploads.indexUpload->GetDestinationBytes();
        records.push_back(
            {key.assetGuid, key.runtimeVersion, GpuAssetDomain::Mesh, bytes, m_ensureFrameCounter, true, true});
    }
    return records;
}

std::vector<GpuAssetResidencyRecord> InxVkCoreModular::GetAssetGpuResidency() const
{
    auto records = GetAssetMeshGpuResidency();
    auto textures = GetAssetTextureGpuResidency();
    records.insert(records.end(), textures.begin(), textures.end());
    return records;
}

size_t InxVkCoreModular::GetRuntimeMeshGpuEntryCount() const
{
    return static_cast<size_t>(std::count_if(m_sharedMeshBuffers.begin(), m_sharedMeshBuffers.end(),
                                             [](const auto &entry) { return entry.first.assetGuid.empty(); }));
}

uint64_t InxVkCoreModular::GetRuntimeMeshGpuResidentBytes() const
{
    uint64_t bytes = 0;
    for (const auto &[key, buffers] : m_sharedMeshBuffers) {
        if (key.assetGuid.empty())
            bytes += buffers.residentBytes;
    }
    return bytes;
}

uint64_t InxVkCoreModular::GetRetiredMeshGpuLeaseBytes() const
{
    SweepRetiredMeshLeases();
    uint64_t bytes = 0;
    for (const auto &lease : m_retiredMeshLeases)
        bytes += lease.residentBytes;
    return bytes;
}

GpuEvictionCandidate InxVkCoreModular::PeekOldestMeshGpuEvictable() const
{
    auto candidate = m_sharedMeshBuffers.end();
    for (auto entry = m_sharedMeshBuffers.begin(); entry != m_sharedMeshBuffers.end(); ++entry) {
        if (!entry->first.assetGuid.empty() &&
            AssetRegistry::Instance().IsMeshGpuResidencyRequired(entry->first.assetGuid, entry->first.runtimeVersion))
            continue;
        if (!entry->second.vertexBuffer || !entry->second.indexBuffer || entry->second.vertexBuffer.use_count() != 1 ||
            entry->second.indexBuffer.use_count() != 1 || entry->second.lastUsedFrame >= m_ensureFrameCounter)
            continue;
        if (candidate == m_sharedMeshBuffers.end() || entry->second.lastUsedFrame < candidate->second.lastUsedFrame)
            candidate = entry;
    }
    if (candidate == m_sharedMeshBuffers.end())
        return {};
    return {candidate->second.lastUsedFrame, candidate->second.residentBytes, true};
}

uint64_t InxVkCoreModular::EvictOldestMeshGpu()
{
    auto candidate = m_sharedMeshBuffers.end();
    for (auto entry = m_sharedMeshBuffers.begin(); entry != m_sharedMeshBuffers.end(); ++entry) {
        if (!entry->first.assetGuid.empty() &&
            AssetRegistry::Instance().IsMeshGpuResidencyRequired(entry->first.assetGuid, entry->first.runtimeVersion))
            continue;
        if (!entry->second.vertexBuffer || !entry->second.indexBuffer || entry->second.vertexBuffer.use_count() != 1 ||
            entry->second.indexBuffer.use_count() != 1 || entry->second.lastUsedFrame >= m_ensureFrameCounter)
            continue;
        if (candidate == m_sharedMeshBuffers.end() || entry->second.lastUsedFrame < candidate->second.lastUsedFrame)
            candidate = entry;
    }
    if (candidate == m_sharedMeshBuffers.end())
        return 0;
    SharedMeshBuffers retired = std::move(candidate->second);
    const uint64_t bytes = retired.residentBytes;
    m_sharedMeshBuffers.erase(candidate);
    RetireSharedMeshBuffers(std::move(retired), true);
    return bytes;
}

void InxVkCoreModular::SetMeshGpuBudgetBytes(uint64_t bytes)
{
    if (bytes == 0)
        throw std::invalid_argument("GPU mesh budget must be greater than zero");
    m_meshGpuBudgetBytes = bytes;
    (void)TrimMeshGpuBudget();
}

size_t InxVkCoreModular::TrimMeshGpuBudget()
{
    SweepRetiredMeshLeases();
    size_t evicted = 0;
    while (m_meshGpuResidentBytes > m_meshGpuBudgetBytes) {
        auto candidate = m_sharedMeshBuffers.end();
        for (auto entry = m_sharedMeshBuffers.begin(); entry != m_sharedMeshBuffers.end(); ++entry) {
            if (!entry->first.assetGuid.empty() && AssetRegistry::Instance().IsMeshGpuResidencyRequired(
                                                       entry->first.assetGuid, entry->first.runtimeVersion))
                continue;
            if (!entry->second.vertexBuffer || !entry->second.indexBuffer ||
                entry->second.vertexBuffer.use_count() != 1 || entry->second.indexBuffer.use_count() != 1 ||
                entry->second.lastUsedFrame >= m_ensureFrameCounter)
                continue;
            if (candidate == m_sharedMeshBuffers.end() || entry->second.lastUsedFrame < candidate->second.lastUsedFrame)
                candidate = entry;
        }
        if (candidate == m_sharedMeshBuffers.end())
            break;
        SharedMeshBuffers retired = std::move(candidate->second);
        m_sharedMeshBuffers.erase(candidate);
        RetireSharedMeshBuffers(std::move(retired), true);
        ++evicted;
    }
    return evicted;
}

// ============================================================================
// Per-View Descriptor Set (set 1) — multi-camera shadow isolation
// ============================================================================

bool InxVkCoreModular::CreatePerViewDescriptorResources()
{
    VkDevice device = GetDevice();
    if (device == VK_NULL_HANDLE)
        return false;

    // Canonical per-view ABI. Geometry uses binding 0 plus the tiled buffers;
    // particles also consume binding 4 because their set 0 remains dedicated
    // to simulation instances and material resources.
    std::array<VkDescriptorSetLayoutBinding, 8> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    for (uint32_t binding = 1; binding <= 3; ++binding) {
        bindings[binding].binding = binding;
        bindings[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[binding].descriptorCount = 1;
        bindings[binding].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[6].descriptorCount = 1;
    bindings[6].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[7] = {7, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_perViewDescSetLayout) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to create per-view descriptor set layout");
        return false;
    }
    ShaderProgram::SetPerViewDescSetLayout(m_perViewDescSetLayout);

    INXLOG_INFO("Created per-view descriptor set layout (multi-camera shadow)");
    return true;
}

void InxVkCoreModular::DestroyPerViewDescriptorResources()
{
    VkDevice device = GetDevice();
    if (device == VK_NULL_HANDLE)
        return;

    if (m_perViewDescSetLayout != VK_NULL_HANDLE) {
        ShaderProgram::SetPerViewDescSetLayout(VK_NULL_HANDLE);
        vkDestroyDescriptorSetLayout(device, m_perViewDescSetLayout, nullptr);
        m_perViewDescSetLayout = VK_NULL_HANDLE;
    }
}

vk::DescriptorLease InxVkCoreModular::AllocatePerViewDescriptorLease()
{
    if (m_perViewDescSetLayout == VK_NULL_HANDLE) {
        INXLOG_ERROR("Per-view descriptor resources not initialized");
        return {};
    }

    auto lease = m_backend.Device().GetRhiDevice().GetDescriptorManager().Allocate(m_perViewDescSetLayout,
                                                                                   vk::DescriptorArena::ViewPersistent);
    if (!lease.IsValid()) {
        INXLOG_ERROR("Failed to allocate per-view descriptor set");
        return {};
    }

    // Initialize with default (white) texture so shaders don't sample garbage
    ClearPerViewShadowMap(lease.set);

    return lease;
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

    std::array<VkWriteDescriptorSet, 2> writes{};
    for (uint32_t index = 0; index < writes.size(); ++index) {
        auto &write = writes[index];
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = perViewDescSet;
        write.dstBinding = index == 0 ? 0 : 6;
        write.descriptorCount = 1;
        write.descriptorType = index == 0 ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLER;
        write.pImageInfo = &imageInfo;
    }
    vkUpdateDescriptorSets(GetDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
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

void InxVkCoreModular::UpdatePerViewForwardPlusBuffers(VkDescriptorSet perViewDescSet,
                                                       rhi::BufferHandle canonicalLights, uint64_t canonicalBytes,
                                                       rhi::BufferHandle tileHeaders, uint64_t tileHeaderBytes,
                                                       rhi::BufferHandle tileLightMasks, uint64_t tileLightMaskBytes,
                                                       rhi::BufferHandle lightingUbo, uint64_t lightingUboBytes)
{
    if (perViewDescSet == VK_NULL_HANDLE || canonicalBytes == 0 || tileHeaderBytes == 0 || tileLightMaskBytes == 0)
        return;

    auto &rhiDevice = m_backend.Device().GetRhiDevice();
    const std::array<VkDescriptorBufferInfo, 3> infos = {{
        {rhiDevice.Resolve(canonicalLights), 0, canonicalBytes},
        {rhiDevice.Resolve(tileHeaders), 0, tileHeaderBytes},
        {rhiDevice.Resolve(tileLightMasks), 0, tileLightMaskBytes},
    }};
    if (std::any_of(infos.begin(), infos.end(), [](const auto &info) { return info.buffer == VK_NULL_HANDLE; }))
        return;

    std::array<VkWriteDescriptorSet, 4> writes{};
    uint32_t writeCount = 0;
    for (uint32_t index = 0; index < infos.size(); ++index) {
        auto &write = writes[writeCount++];
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = perViewDescSet;
        write.dstBinding = index + 1u;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &infos[index];
    }
    VkDescriptorBufferInfo lightingInfo{};
    if (lightingUbo.IsValid() && lightingUboBytes > 0) {
        lightingInfo = {rhiDevice.Resolve(lightingUbo), 0, lightingUboBytes};
        if (lightingInfo.buffer != VK_NULL_HANDLE) {
            auto &write = writes[writeCount++];
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = perViewDescSet;
            write.dstBinding = 4;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            write.pBufferInfo = &lightingInfo;
        }
    }
    vkUpdateDescriptorSets(GetDevice(), writeCount, writes.data(), 0, nullptr);
}

void InxVkCoreModular::UpdatePerViewLightingBuffer(VkDescriptorSet perViewDescSet, VkBuffer lightingUbo,
                                                   uint64_t lightingUboBytes)
{
    if (perViewDescSet == VK_NULL_HANDLE || lightingUbo == VK_NULL_HANDLE || lightingUboBytes == 0)
        return;
    VkDescriptorBufferInfo info{lightingUbo, 0, lightingUboBytes};
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = perViewDescSet;
    write.dstBinding = 4;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.pBufferInfo = &info;
    vkUpdateDescriptorSets(GetDevice(), 1, &write, 0, nullptr);
}

void InxVkCoreModular::UpdatePerViewCameraBuffer(VkDescriptorSet perViewDescSet, VkBuffer cameraUbo,
                                                 uint64_t cameraUboBytes)
{
    if (perViewDescSet == VK_NULL_HANDLE || cameraUbo == VK_NULL_HANDLE || cameraUboBytes == 0)
        return;
    VkDescriptorBufferInfo info{cameraUbo, 0, cameraUboBytes};
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = perViewDescSet;
    write.dstBinding = 5;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.pBufferInfo = &info;
    vkUpdateDescriptorSets(GetDevice(), 1, &write, 0, nullptr);
}

} // namespace infernux
