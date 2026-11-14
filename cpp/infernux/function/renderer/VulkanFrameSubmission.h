#pragma once

#include <function/renderer/rhi/RenderSubmissionPlan.h>
#include <vulkan/vulkan_core.h>

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace infernux::vk
{

class RenderGraph;

/// Vulkan recording companion for the backend-neutral SubmissionPlanComposer.
/// Every task owns one recorder, while imported RenderGraphs retain their
/// compiler-produced batch boundaries and queue-ownership transitions.
class VulkanFrameSubmission
{
  public:
    using Recorder = std::function<bool(VkCommandBuffer)>;
    using BatchHook = std::function<bool(uint32_t, VkCommandBuffer)>;
    using GraphHook = std::function<bool(VkCommandBuffer)>;

    void Reset(uint32_t firstWorkItemId = 1);
    void Retain(std::shared_ptr<void> owner);
    [[nodiscard]] std::vector<std::shared_ptr<void>> TakeResourceOwners();
    void OnAccepted(std::function<void(rhi::SubmissionTicket)> publish);
    void PublishAccepted(rhi::SubmissionTicket ticket);
    void OnWorkAccepted(uint32_t workItem, std::function<void(rhi::SubmissionTicket)> publish);
    void PublishWorkAccepted(const rhi::SubmissionPlan &plan, const std::vector<rhi::SubmissionTicket> &batchTickets,
                             size_t acceptedCount);
    [[nodiscard]] size_t WorkPublicationCount() const noexcept
    {
        return m_workPublications.size();
    }
    void DiscardPublications() noexcept;

    [[nodiscard]] uint32_t AddWork(rhi::DeviceId device, rhi::QueueRole queue, rhi::SubmissionDomain domain,
                                   rhi::RenderViewId view, rhi::PipelineStage waitStages,
                                   std::vector<uint32_t> dependencies, Recorder recorder,
                                   std::string diagnosticName = {});

    [[nodiscard]] rhi::ComposedSubmissionRange AppendRenderGraph(RenderGraph &graph,
                                                                 const std::vector<uint32_t> &externalDependencies = {},
                                                                 BatchHook beforeBatch = {}, GraphHook afterGraph = {});

    [[nodiscard]] bool Build(rhi::SubmissionPlan &output, std::string &error) const;
    [[nodiscard]] bool RecordBatch(const rhi::SubmissionPlan &plan, uint32_t batchIndex,
                                   VkCommandBuffer commandBuffer) const;

    [[nodiscard]] uint32_t LastWork(rhi::QueueRole queue) const noexcept;
    [[nodiscard]] size_t WorkCount() const noexcept
    {
        return m_recorders.size();
    }

  private:
    rhi::SubmissionPlanComposer m_composer;
    std::unordered_map<uint32_t, Recorder> m_recorders;
    std::vector<std::shared_ptr<void>> m_resourceOwners;
    std::vector<std::function<void(rhi::SubmissionTicket)>> m_publications;
    std::unordered_map<uint32_t, std::vector<std::function<void(rhi::SubmissionTicket)>>> m_workPublications;
    std::array<uint32_t, static_cast<size_t>(rhi::QueueRole::Count)> m_lastWork{};
};

} // namespace infernux::vk
