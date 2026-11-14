#include "VulkanFrameSubmission.h"

#include "vk/RenderGraph.h"

#include <core/error/InxError.h>
#include <stdexcept>

namespace infernux::vk
{

void VulkanFrameSubmission::Reset(uint32_t firstWorkItemId)
{
    m_composer.Reset(firstWorkItemId);
    m_recorders.clear();
    m_resourceOwners.clear();
    m_publications.clear();
    m_workPublications.clear();
    m_lastWork.fill(0);
}

void VulkanFrameSubmission::Retain(std::shared_ptr<void> owner)
{
    if (owner)
        m_resourceOwners.push_back(std::move(owner));
}

std::vector<std::shared_ptr<void>> VulkanFrameSubmission::TakeResourceOwners()
{
    auto owners = std::move(m_resourceOwners);
    m_resourceOwners.clear();
    return owners;
}

void VulkanFrameSubmission::OnAccepted(std::function<void(rhi::SubmissionTicket)> publish)
{
    m_publications.push_back(std::move(publish));
}

void VulkanFrameSubmission::PublishAccepted(rhi::SubmissionTicket ticket)
{
    for (auto &publish : m_publications)
        publish(ticket);
    m_publications.clear();
}

void VulkanFrameSubmission::OnWorkAccepted(uint32_t workItem, std::function<void(rhi::SubmissionTicket)> publish)
{
    if (!workItem || !publish)
        throw std::invalid_argument("Work publication needs a real work item and callback");
    m_workPublications[workItem].push_back(std::move(publish));
}

void VulkanFrameSubmission::PublishWorkAccepted(const rhi::SubmissionPlan &plan,
                                                const std::vector<rhi::SubmissionTicket> &tickets, size_t acceptedCount)
{
    if (acceptedCount == 0) {
        m_workPublications.clear();
        return;
    }
    if (tickets.size() != plan.batches.size())
        throw std::logic_error("Accepted batch tickets do not match the submitted plan");
    if (acceptedCount > plan.batches.size())
        throw std::logic_error("Accepted work prefix exceeds the actual plan");
    for (size_t index = 0; index < acceptedCount; ++index)
        for (const auto work : plan.batches[index].workItems)
            if (const auto found = m_workPublications.find(work); found != m_workPublications.end()) {
                for (auto &publish : found->second)
                    publish(tickets[index]);
                m_workPublications.erase(found);
            }
    if (acceptedCount == plan.batches.size() && !m_workPublications.empty())
        throw std::logic_error("Accepted output publication work was omitted from the submitted plan");
    m_workPublications.clear(); // Unaccepted reservations publish nothing.
}

void VulkanFrameSubmission::DiscardPublications() noexcept
{
    m_publications.clear();
    m_workPublications.clear();
}

uint32_t VulkanFrameSubmission::AddWork(rhi::DeviceId device, rhi::QueueRole queue, rhi::SubmissionDomain domain,
                                        rhi::RenderViewId view, rhi::PipelineStage waitStages,
                                        std::vector<uint32_t> dependencies, Recorder recorder,
                                        std::string diagnosticName)
{
    const uint32_t workItem = m_composer.AddWork(device, queue, domain, view, waitStages, std::move(dependencies), true,
                                                 std::move(diagnosticName));
    if (recorder)
        m_recorders.emplace(workItem, std::move(recorder));
    if (queue != rhi::QueueRole::Count)
        m_lastWork[static_cast<size_t>(queue)] = workItem;
    return workItem;
}

rhi::ComposedSubmissionRange VulkanFrameSubmission::AppendRenderGraph(RenderGraph &graph,
                                                                      const std::vector<uint32_t> &externalDependencies,
                                                                      BatchHook beforeBatch, GraphHook afterGraph)
{
    const auto &source = graph.GetSubmissionPlan();
    rhi::ComposedSubmissionRange range = m_composer.Append(source, externalDependencies);
    for (size_t batchIndex = 0; batchIndex < range.workItems.size(); ++batchIndex) {
        const uint32_t workItem = range.workItems[batchIndex];
        m_recorders.emplace(workItem, [&graph, batchIndex, batchCount = range.workItems.size(), beforeBatch,
                                       afterGraph](VkCommandBuffer cmd) {
            if (beforeBatch && !beforeBatch(static_cast<uint32_t>(batchIndex), cmd))
                return false;
            if (batchIndex == 0)
                graph.BeginExecution();
            if (!graph.RecordSubmissionBatch(static_cast<uint32_t>(batchIndex), cmd))
                return false;
            return batchIndex + 1 != batchCount || !afterGraph || afterGraph(cmd);
        });
        const auto queue = source.batches[batchIndex].queue;
        if (queue != rhi::QueueRole::Count)
            m_lastWork[static_cast<size_t>(queue)] = workItem;
    }
    return range;
}

bool VulkanFrameSubmission::Build(rhi::SubmissionPlan &output, std::string &error) const
{
    return m_composer.Build(output, error);
}

bool VulkanFrameSubmission::RecordBatch(const rhi::SubmissionPlan &plan, uint32_t batchIndex,
                                        VkCommandBuffer commandBuffer) const
{
    if (batchIndex >= plan.batches.size() || commandBuffer == VK_NULL_HANDLE)
        return false;
    for (const uint32_t workItem : plan.batches[batchIndex].workItems) {
        const auto recorder = m_recorders.find(workItem);
        if (recorder == m_recorders.end()) {
            INXLOG_ERROR("VulkanFrameSubmission has no recorder for work item ", workItem);
            return false;
        }
        if (!recorder->second(commandBuffer))
            return false;
    }
    return true;
}

uint32_t VulkanFrameSubmission::LastWork(rhi::QueueRole queue) const noexcept
{
    if (queue == rhi::QueueRole::Count)
        return 0;
    return m_lastWork[static_cast<size_t>(queue)];
}

} // namespace infernux::vk
