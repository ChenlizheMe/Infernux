#pragma once

// Internal owner implementation, included only by InxRenderer.cpp.
#include "GameScreenOutput.h"
#include "InxVkCoreModular.h"
#include "RenderWorld.h"
#include "SceneRenderTarget.h"
#include "VulkanFrameSubmission.h"
#include "core/threading/FrameWorkOwner.h"
#include "function/scene/SceneManager.h"
#include "function/scene/SceneRenderBridge.h"
#include "gui/InxGUI.h"
#include "rhi/RhiRenderTexture.h"
#include "vk/RenderGraph.h"
#include "vk/RhiVulkanTypes.h"
#include "vk/VulkanQueueManager.h"
#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace infernux
{
/// Device-owner-thread registry. It allocates images through existing RHI,
/// contributes to the existing frame plan, and never owns a device/queue/renderer.
class GameScreenOutputs final
{
  public:
    struct PixelStorage
    {
        std::shared_ptr<rhi::RenderTexture> texture;
        mutable uint64_t textureId = 0;
        mutable bool registered = false;
        mutable std::vector<uint64_t> guiDescriptors;
    };
    struct Slot
    {
        std::shared_ptr<PixelStorage> pixels = std::make_shared<PixelStorage>();
        std::shared_ptr<SceneRenderTarget> target;
    };
    struct Output
    {
        uint64_t worldId = 0, generation = 1, nextSequence = 1, nextSlot = 1;
        const rhi::RenderViewId id = rhi::AllocateRenderViewId();
        GameScreenOutputConfig config;
        rhi::SampleCount samples = rhi::SampleCount::One;
        bool retired = false;
        bool workWithheld = false;
        bool waitingForPublication = false;
        bool forceSample = false, requiredBufferWork = false;
        std::string error;
        std::string publicationWaitReason;
        std::vector<std::shared_ptr<Slot>> slots;
        std::shared_ptr<Slot> work;
        std::shared_ptr<const GameScreenAdmission> admission;
        std::shared_ptr<const GameScreenSample> front, lastAccepted;
        std::vector<std::shared_ptr<const GameScreenSample>> pending;
        std::shared_ptr<const RenderWorldFrame> acceptedSource;
        struct Consumer
        {
            GameScreenOutputConfig config;
            GameScreenConsumerKind kind = GameScreenConsumerKind::Presentation;
        };
        std::map<std::string, Consumer> consumers;
    };

    GameScreenOutputs(InxVkCoreModular &core, InxGUI &gui) : m_core(core), m_gui(gui)
    {
    }
    std::shared_ptr<Output> Touch(uint64_t worldId)
    {
        return Ensure(worldId);
    }
    rhi::SampleCount Samples() const noexcept
    {
        return m_samples;
    }

    void Request(const std::string &consumer, uint64_t worldId, const GameScreenOutputConfig &config,
                 GameScreenConsumerKind kind = GameScreenConsumerKind::Presentation)
    {
        if (consumer.empty() || !worldId || !config.width || !config.height)
            throw std::invalid_argument(
                "Game screen request needs a stable consumer, actual World and positive extent");
        auto output = Ensure(worldId);
        // One logical screen output has one raster domain. Multiple consumers
        // may choose independent display zoom; conflicting raster sizes are a
        // configuration error, not implicit duplicate camera executions.
        for (const auto &[other, requested] : output->consumers)
            if (kind == GameScreenConsumerKind::Presentation && other != consumer &&
                requested.kind == GameScreenConsumerKind::Presentation && !(requested.config == config))
                throw std::invalid_argument("Consumers of one World screen output requested different raster sizes");
        const bool first = output->consumers.empty() && !output->requiredBufferWork;
        output->consumers[consumer] = {config, kind};
        if (kind == GameScreenConsumerKind::Required && !first)
            return;
        if (!(output->config == config)) {
            output->config = config;
            Reconfigure(*output);
        }
    }

    std::vector<uint64_t> Release(const std::string &consumer, uint64_t keepWorld = 0)
    {
        std::vector<uint64_t> emptied;
        for (auto &[world, output] : m_outputs) {
            if (world == keepWorld || !output->consumers.erase(consumer) || !output->consumers.empty() ||
                output->requiredBufferWork)
                continue;
            Reconfigure(*output, false);
            output->front.reset();
            output->lastAccepted.reset();
            output->pending.clear();
            output->acceptedSource.reset();
            emptied.push_back(world);
        }
        return emptied;
    }

    void SetSamples(rhi::SampleCount samples)
    {
        m_samples = samples;
        for (auto &[world, output] : m_outputs)
            if (output->samples != samples) {
                output->samples = samples;
                // No commands from this admission have been submitted. The
                // new pipeline generation cannot use its older raster images.
                output->work.reset();
                output->admission.reset();
                Reconfigure(*output);
                output->forceSample = true;
            }
    }

    void InvalidatePresentation()
    {
        for (auto &[world, output] : m_outputs) {
            output->work.reset();
            output->admission.reset();
            output->forceSample = true;
        }
    }

    std::shared_ptr<Output> Find(uint64_t worldId) const
    {
        const auto found = m_outputs.find(worldId);
        return found == m_outputs.end() ? nullptr : found->second;
    }

    GameScreenOutputConfig Config(uint64_t worldId) const
    {
        const auto output = Find(worldId);
        return output ? output->config : m_defaultConfig;
    }

    void SetDefaultConfig(const GameScreenOutputConfig &config)
    {
        m_defaultConfig = config;
    }

    bool Requested(uint64_t worldId) const
    {
        const auto output = Find(worldId);
        return output && !output->retired && (!output->consumers.empty() || output->requiredBufferWork);
    }

    bool Required(uint64_t worldId) const
    {
        const auto output = Find(worldId);
        return output && (output->requiredBufferWork ||
                          std::any_of(output->consumers.begin(), output->consumers.end(), [](const auto &entry) {
                              return entry.second.kind == GameScreenConsumerKind::Required;
                          }));
    }

    void RequestSample(const std::string &consumer, uint64_t worldId)
    {
        const auto output = Find(worldId);
        if (!output || !output->consumers.count(consumer) ||
            output->consumers.at(consumer).kind != GameScreenConsumerKind::Required)
            throw std::invalid_argument("Explicit screen sampling needs an active Required consumer");
        output->forceSample = true;
    }

    std::vector<uint64_t> SyncRequiredBufferWork(const std::unordered_set<uint64_t> &required)
    {
        for (const auto world : required)
            Ensure(world)->requiredBufferWork = true;
        std::vector<uint64_t> emptied;
        for (auto &[world, output] : m_outputs)
            if (output->requiredBufferWork && !required.count(world)) {
                output->requiredBufferWork = false;
                if (output->consumers.empty()) {
                    Reconfigure(*output, false);
                    output->work.reset();
                    output->admission.reset();
                    output->front.reset();
                    output->lastAccepted.reset();
                    output->pending.clear();
                    emptied.push_back(world);
                }
            }
        return emptied;
    }

    bool Due(uint64_t worldId, const std::shared_ptr<const RenderWorldFrame> &source) const
    {
        const auto output = Find(worldId);
        return Requested(worldId) && source &&
               (output->work || output->acceptedSource != source || output->forceSample || output->requiredBufferWork);
    }

    /// At most one work generation per output in a Host opportunity. No wait:
    /// allocation/lease pressure withholds only this output's new work.
    bool Prepare(uint64_t worldId, const std::shared_ptr<const RenderWorldFrame> &source,
                 std::shared_ptr<const RenderWorldPublication> publication = {})
    {
        const auto output = Find(worldId);
        if (!Due(worldId, source))
            return false;
        if (output->work)
            return true;
        if (publication && !publication->screenConfig) {
            output->waitingForPublication = true;
            output->publicationWaitReason = "Completed World frame has no sealed screen admission";
            return false;
        }
        if (!publication && SceneManager::Instance().IsRuntimeWorldManagementActive())
            throw std::logic_error("Runtime screen source requires its completed World publication");
        const auto raster =
            publication ? *publication->screenConfig
                        : GameScreenSourceConfig{output->id, output->generation, output->config, output->samples};
        if (raster.outputId != output->id)
            throw std::logic_error("Sealed screen admission belongs to a different logical output");
        for (const auto &camera : source->Views())
            if (!camera.editorView && !camera.renderToTexture &&
                (camera.outputWidth != raster.config.width || camera.outputHeight != raster.config.height)) {
                output->waitingForPublication = true;
                output->workWithheld = true;
                return false;
            }
        output->waitingForPublication = raster.generation != output->generation;
        output->publicationWaitReason =
            output->waitingForPublication ? "Waiting for a new completed World frame for the requested raster" : "";
        const auto admit = [&output, &source, &publication, &raster](const std::shared_ptr<Slot> &slot) {
            auto admission = std::make_shared<GameScreenAdmission>();
            admission->worldId = output->worldId;
            admission->generation = raster.generation;
            admission->outputId = raster.outputId;
            admission->config = raster.config;
            admission->samples = raster.samples;
            admission->source = source;
            admission->publication = publication;
            output->work = slot;
            output->admission = std::move(admission);
        };
        for (const auto &slot : output->slots) {
            const auto image = slot->pixels->texture->Acquire();
            if (image->width != raster.config.width || image->height != raster.config.height ||
                image->description.samples != raster.samples || image->description.filter != raster.config.filter)
                continue;
            const bool guiUsesPixels =
                std::any_of(slot->pixels->guiDescriptors.begin(), slot->pixels->guiDescriptors.end(),
                            [this](uint64_t id) { return m_gui.IsTextureReferencedByCurrentDrawData(id); });
            if (slot->pixels.use_count() == 1 && !guiUsesPixels) {
                Unregister(*slot->pixels);
                slot->pixels->guiDescriptors.clear();
                admit(slot);
                output->workWithheld = false;
                output->error.clear();
                return true;
            }
        }
        // Bounded outstanding samples; pinned training/capture consumers apply
        // local backpressure instead of unbounded image allocation.
        const size_t limit = static_cast<size_t>(m_core.GetMaxFramesInFlight()) + 2;
        if (output->slots.size() >= limit) {
            output->workWithheld = true;
            return false;
        }
        try {
            auto slot = std::make_shared<Slot>();
            rhi::RenderTextureDesc desc;
            desc.width = raster.config.width;
            desc.height = raster.config.height;
            desc.filter = raster.config.filter;
            desc.colorFormat = rhi::PixelFormat::RGBA16SFloat;
            desc.depthFormat = rhi::FromVkFormat(m_core.GetDeviceContext().FindSampledDepthFormat());
            desc.sampledDepth = true;
            desc.samples = raster.samples;
            slot->pixels->texture = std::make_shared<rhi::RenderTexture>(
                m_core.GetDeviceContext().GetRhiDevice(),
                "WorldScreen/" + std::to_string(output->id) + "/storage/" + std::to_string(output->nextSlot++), desc);
            // No synchronous SceneRenderTarget::Initialize or outline attachment.
            // The normal frame pre-setup initializes this RHI generation once.
            slot->target = std::make_shared<SceneRenderTarget>(&m_core);
            slot->target->BindAttachments(slot->pixels->texture->Acquire());
            const auto generation = slot->pixels->texture->Acquire();
            m_images.push_back(generation->color);
            if (generation->multisampleColor)
                m_images.push_back(generation->multisampleColor);
            if (generation->depth)
                m_images.push_back(generation->depth);
            output->slots.push_back(slot);
            admit(slot);
            output->workWithheld = false;
            output->error.clear();
            return true;
        } catch (const std::exception &error) {
            output->error = error.what();
            output->workWithheld = true;
            return false;
        }
    }

    SceneRenderTarget *Work(uint64_t worldId) const
    {
        const auto output = Find(worldId);
        return output && output->work ? output->work->target.get() : nullptr;
    }

    std::shared_ptr<const GameScreenAdmission> Admitted(uint64_t worldId) const
    {
        const auto output = Find(worldId);
        return output ? output->admission : nullptr;
    }

    std::shared_ptr<const void> WorkOwner(uint64_t worldId) const
    {
        const auto output = Find(worldId);
        return output ? output->work : nullptr;
    }

    SceneRenderTarget *LatestTarget(uint64_t worldId) const
    {
        const auto output = Find(worldId);
        if (!output)
            return nullptr;
        if (output->work)
            return output->work->target.get();
        if (output->lastAccepted)
            for (const auto &slot : output->slots)
                if (slot->pixels == output->lastAccepted->storageOwner)
                    return slot->target.get();
        return nullptr;
    }

    /// Only authored preservation requires a copy. Ordinary full-clear Camera
    /// stacks promote owning allocations directly, without a front resolve.
    /// The previous accepted sample, not the older completed front, is the
    /// source, preserving consecutive sample semantics on the Graphics queue.
    uint32_t AppendPreserve(vk::VulkanFrameSubmission &submission, uint64_t worldId, uint32_t setup, bool color,
                            bool depth)
    {
        const auto output = Find(worldId);
        const auto previous = output ? output->lastAccepted : nullptr;
        if (!output || !output->work || !output->admission || !previous ||
            previous->generation != output->admission->generation || (!color && !depth))
            return setup;
        const auto source = previous->image;
        const auto destination = output->work->pixels->texture->Acquire();
        if (source == destination)
            throw std::logic_error("A pinned screen sample was selected for writable work");
        if (color && source->multisampleColor &&
            !m_core.GetDeviceContext().GetRhiDevice().GetCapabilities().features.multisampleTextureCopies)
            throw std::runtime_error("Preserving a multisampled screen requires multisample texture copies");
        const auto work = submission.AddWork(
            m_core.GetDeviceContext().GetDeviceId(), rhi::QueueRole::Graphics, rhi::SubmissionDomain::Frame, output->id,
            rhi::PipelineStage::Transfer, {setup},
            [this, source, destination, color, depth](VkCommandBuffer commands) {
                const auto copy = [&](const rhi::TextureResource &from, const rhi::TextureResource &to,
                                      VkImageLayout layout, VkAccessFlags access, VkImageAspectFlags aspect) {
                    auto &device = m_core.GetDeviceContext().GetRhiDevice();
                    VkImageMemoryBarrier barriers[2]{};
                    for (auto &barrier : barriers) {
                        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                        barrier.oldLayout = layout;
                        barrier.srcAccessMask = access;
                        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        barrier.subresourceRange.aspectMask = aspect;
                        barrier.subresourceRange.levelCount = barrier.subresourceRange.layerCount = 1;
                    }
                    barriers[0].image = device.Resolve(from.GetTexture());
                    barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                    barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    barriers[1].image = device.Resolve(to.GetTexture());
                    barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         0, 0, nullptr, 0, nullptr, 2, barriers);
                    // Combined depth/stencil formats use separate aspect regions.
                    VkImageCopy regions[2]{};
                    uint32_t count = 0;
                    for (const auto bit :
                         {VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_ASPECT_STENCIL_BIT})
                        if (aspect & bit) {
                            auto &region = regions[count++];
                            region.srcSubresource.aspectMask = region.dstSubresource.aspectMask = bit;
                            region.srcSubresource.layerCount = region.dstSubresource.layerCount = 1;
                            region.extent = {source->width, source->height, 1};
                        }
                    vkCmdCopyImage(commands, barriers[0].image, barriers[0].newLayout, barriers[1].image,
                                   barriers[1].newLayout, count, regions);
                    for (auto &barrier : barriers) {
                        barrier.oldLayout = barrier.newLayout;
                        barrier.newLayout = layout;
                        barrier.srcAccessMask = barrier.dstAccessMask;
                        barrier.dstAccessMask = access;
                    }
                    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                         0, 0, nullptr, 0, nullptr, 2, barriers);
                };
                if (color) {
                    copy(*source->color, *destination->color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
                    if (source->multisampleColor)
                        copy(*source->multisampleColor, *destination->multisampleColor,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                             VK_IMAGE_ASPECT_COLOR_BIT);
                }
                if (depth && source->depth) {
                    const bool sampled =
                        source->description.sampledDepth && source->description.samples == rhi::SampleCount::One;
                    copy(*source->depth, *destination->depth,
                         sampled ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                 : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                         sampled ? VK_ACCESS_SHADER_READ_BIT
                                 : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                         rhi::ToVkImageAspectMask(rhi::ToVkFormat(source->description.depthFormat)));
                }
                return true;
            },
            "WorldScreen/PreserveAuthoredContents");
        submission.Retain(std::const_pointer_cast<GameScreenSample>(previous));
        submission.Retain(output->work);
        return work;
    }

    std::shared_ptr<const GameScreenSample> Acquire(uint64_t worldId) const
    {
        const auto output = Find(worldId);
        return output && !output->retired ? output->front : nullptr;
    }

    GameScreenOutputStatus Status(uint64_t worldId) const
    {
        GameScreenOutputStatus result;
        result.worldId = worldId;
        const auto output = Find(worldId);
        if (!output)
            return result;
        result.outputId = output->id;
        result.generation = output->generation;
        result.requested = output->config;
        result.consumers = output->consumers.size();
        result.pendingSamples = output->pending.size();
        result.allocatedSlots = output->slots.size();
        result.hasCompletedSample = static_cast<bool>(output->front);
        result.hasPresentationConsumer =
            std::any_of(output->consumers.begin(), output->consumers.end(),
                        [](const auto &entry) { return entry.second.kind == GameScreenConsumerKind::Presentation; });
        result.hasRequiredConsumer = Required(worldId);
        result.admitted = output->admission;
        result.accepted = output->lastAccepted;
        result.completed = output->front;
        result.workWithheld = output->workWithheld;
        result.waitingForPublication = output->waitingForPublication;
        result.error = output->error;
        result.publicationWaitReason = output->publicationWaitReason;
        return result;
    }

    void WaitForPublication(uint64_t worldId, std::string reason)
    {
        if (const auto output = Find(worldId)) {
            output->waitingForPublication = !reason.empty();
            output->workWithheld = !reason.empty();
            output->publicationWaitReason = std::move(reason);
        }
    }

    /// Output completion gets a distinct public-plan work item and batch ticket.
    /// No per-World submitter, frame fence, image resolve or full-image copy.
    uint32_t AppendCompletion(vk::VulkanFrameSubmission &submission, uint64_t worldId,
                              const std::shared_ptr<const RenderWorldFrame> &source,
                              const std::vector<uint32_t> &stackTerminals)
    {
        const auto output = Find(worldId);
        const auto admission = output ? output->admission : nullptr;
        if (!output || !output->work || !admission || stackTerminals.empty())
            throw std::logic_error("Game screen completion has no real stack work");
        if (admission->source != source || admission->worldId != worldId)
            throw std::logic_error("Screen work source changed after admission");
        output->pending.reserve(output->pending.size() + 1);
        auto sample = std::make_shared<GameScreenSample>();
        sample->worldId = worldId;
        sample->generation = admission->generation;
        sample->sequence = output->nextSequence++;
        sample->outputId = output->id;
        sample->completedWorldFrame = source->CompletedWorldFrame();
        sample->simulationEpoch = source->SimulationEpoch();
        sample->source = source;
        sample->admission = admission;
        sample->image = output->work->pixels->texture->Acquire();
        sample->storageOwner = output->work->pixels;
        const auto workOwner = output->work;
        // AddWork forces a real boundary in SubmissionPlanComposer; executor
        // retains each VkSubmitInfo's timeline signal even when batching the
        // host vkQueueSubmit call. A later World cannot share this batch ticket.
        const auto completionWork = submission.AddWork(
            m_core.GetDeviceContext().GetDeviceId(), rhi::QueueRole::Graphics, rhi::SubmissionDomain::Frame, output->id,
            rhi::PipelineStage::AllCommands, stackTerminals, [](VkCommandBuffer) { return true; },
            "WorldScreen/Complete");
        submission.OnWorkAccepted(
            completionWork, [this, output, sample, source, admission, workOwner](rhi::SubmissionTicket ticket) {
                sample->completion = ticket;
                m_core.GetBackendContext().Queues().WatchWorkCompletion(ticket, sample->receipt);
                sample->retirementEpoch = m_core.GetLastSubmittedCompletionEpoch();
                if (!output->retired) {
                    output->acceptedSource = source;
                    output->forceSample = false;
                    output->pending.push_back(sample);
                    output->lastAccepted = sample;
                    if (output->admission == admission) {
                        output->work.reset();
                        output->admission.reset();
                    }
                    if (admission->generation != output->generation) {
                        output->slots.erase(std::remove(output->slots.begin(), output->slots.end(), workOwner),
                                            output->slots.end());
                        m_core.GetRetirementQueue().RetireAfter(sample->retirementEpoch, [workOwner] {});
                    }
                }
            });
        // Retain pixels through the real frame owner even if retirement/source
        // switch removes the registry entry before the GPU completes.
        submission.Retain(sample);
        return completionWork;
    }

    void Poll()
    {
        m_core.GetBackendContext().Queues().ObserveWorkCompletions(m_core.GetFrameWorkOwner()->Opportunity());
        for (auto &[world, output] : m_outputs) {
            auto &pending = output->pending;
            for (auto it = pending.begin(); it != pending.end();) {
                const auto &sample = *it;
                if (sample->receipt->failed) {
                    output->error = "Graphics work completion failed or its device owner stopped";
                    INXLOG_ERROR(output->error, ", World=", world);
                    it = pending.erase(it);
                    continue;
                }
                if (!sample->receipt->complete) {
                    ++it;
                    continue;
                }
                if (!output->retired && (!output->front || sample->sequence > output->front->sequence)) {
                    if (output->front)
                        Unregister(*std::static_pointer_cast<const PixelStorage>(output->front->storageOwner));
                    output->front = sample;
                }
                it = pending.erase(it);
            }
        }
        m_images.erase(
            std::remove_if(m_images.begin(), m_images.end(), [](const auto &image) { return image.expired(); }),
            m_images.end());
    }

    bool HasPendingCompletion() const
    {
        return std::any_of(m_outputs.begin(), m_outputs.end(),
                           [](const auto &entry) { return !entry.second->pending.empty(); });
    }

    void Retire(uint64_t worldId)
    {
        const auto found = m_outputs.find(worldId);
        if (found == m_outputs.end())
            return;
        auto output = found->second;
        output->retired = true;
        output->consumers.clear();
        Reconfigure(*output);
        output->front.reset();
        output->lastAccepted.reset();
        output->pending.clear();
        output->acceptedSource.reset();
        m_outputs.erase(found);
        // RHI generations and GUI external views use existing deferred
        // retirement. Public samples keep their own pixel/storage lease.
        const auto epoch = m_core.GetBackendContext().Queues().GetLastReservedCompletionEpoch();
        m_core.GetRetirementQueue().RetireAfter(epoch, [output = std::move(output)] {});
    }

    uint64_t ResidentBytes() const
    {
        uint64_t bytes = 0;
        // Each actual attachment is registered once at allocation, not once
        // per output/GUI/sample alias. Weak observation includes old fronts,
        // in-flight and retired generations while any owner retains them.
        for (const auto &entry : m_images)
            if (const auto image = entry.lock())
                bytes += image->GetResidentBytes();
        return bytes;
    }

  private:
    void Unregister(const PixelStorage &pixels)
    {
        if (!pixels.registered)
            return;
        const auto &sampled = pixels.texture->Acquire()->sampledColor;
        m_gui.RemoveImGuiTexture("RenderTextureUI/" + sampled->GetSourceId());
        pixels.registered = false;
    }

    std::shared_ptr<Output> Ensure(uint64_t worldId)
    {
        auto &output = m_outputs[worldId];
        if (!output) {
            output = std::make_shared<Output>();
            output->worldId = worldId;
            output->config = m_defaultConfig;
            output->samples = m_samples;
        }
        return output;
    }
    void Reconfigure(Output &output, bool rasterChanged = true)
    {
        if (rasterChanged)
            ++output.generation;
        output.error.clear();
        output.workWithheld = false;
        output.forceSample = false;
        for (const auto &slot : output.slots)
            Unregister(*slot->pixels);
        if (output.front)
            Unregister(*std::static_pointer_cast<const PixelStorage>(output.front->storageOwner));
        output.waitingForPublication = true;
        output.publicationWaitReason = "Waiting for a new completed World frame for the requested raster";
        // Already admitted work continues with its immutable source/config.
        // Desired generation changes only which allocation may be admitted next.
        auto slots = std::move(output.slots);
        output.slots.clear();
        if (output.work && !output.retired && (!output.consumers.empty() || output.requiredBufferWork)) {
            slots.erase(std::remove(slots.begin(), slots.end(), output.work), slots.end());
            output.slots.push_back(output.work);
        } else {
            output.work.reset();
            output.admission.reset();
        }
        const auto epoch = m_core.GetBackendContext().Queues().GetLastReservedCompletionEpoch();
        m_core.GetRetirementQueue().RetireAfter(epoch, [slots = std::move(slots)] {});
        // Existing front stays valid until a new correctly sized sample
        // completes. Its metadata reports its actual old dimensions/generation.
    }
    InxVkCoreModular &m_core;
    InxGUI &m_gui;
    GameScreenOutputConfig m_defaultConfig;
    rhi::SampleCount m_samples = rhi::SampleCount::One;
    std::unordered_map<uint64_t, std::shared_ptr<Output>> m_outputs;
    std::vector<std::weak_ptr<const rhi::TextureResource>> m_images;
};
} // namespace infernux
