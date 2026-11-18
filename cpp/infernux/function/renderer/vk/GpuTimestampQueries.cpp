#include "GpuTimestampQueries.h"

#include "VkDeviceContext.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace infernux::vk
{

BudgetGpuTiming::BudgetGpuTiming(const VkDeviceContext &context)
{
    if (!m_queries.Initialize(context, 1, 4))
        throw std::runtime_error("Could not create budgeted GPU timestamp queries");
}

void BudgetGpuTiming::Reset() noexcept
{
    m_region = {};
    m_serial = 0;
    m_started = m_finished = m_submitted = false;
}

void BudgetGpuTiming::Destroy() noexcept
{
    m_queries.Destroy();
}

void BudgetGpuTiming::BeginPass(VkCommandBuffer commands, std::string_view name)
{
    if (!m_started) {
        m_queries.BeginFrame(commands, 0, false);
        m_serial = m_queries.FrameSerial(0);
        m_started = true;
    }
    m_region = m_queries.BeginRegion(commands, name, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
}

void BudgetGpuTiming::EndPass(VkCommandBuffer commands, bool last)
{
    m_queries.EndRegion(commands, m_region, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    if (last) {
        m_queries.FinishFrame(commands, 0);
        m_finished = true;
    }
}

std::optional<rhi::GpuTimestampFrame> BudgetGpuTiming::Collect(const rhi::WorkCompletionReceipt &receipt)
{
    if (!receipt.complete || !m_finished)
        return {};
    if (!m_submitted) {
        m_queries.MarkSubmitted(0);
        m_submitted = true;
    }
    auto frame = m_queries.CollectCompletedFrame(0);
    if (frame && frame->serial != m_serial)
        throw std::logic_error("Budgeted GPU timing did not belong to its leased query frame");
    return frame;
}

#if INFERNUX_FRAME_PROFILE
namespace
{
thread_local GpuTimestampQueries *s_recordingQueries = nullptr;
}

GpuTimestampQueries *GpuTimestampQueries::CurrentRecording() noexcept
{
    return s_recordingQueries;
}
#endif

GpuTimestampQueries::~GpuTimestampQueries()
{
    Destroy();
}

bool GpuTimestampQueries::Initialize(const VkDeviceContext &context, uint32_t framesInFlight,
                                     uint32_t maxRegionsPerFrame, uint32_t queueFamily)
{
    Destroy();

    m_device = context.GetDevice();
    if (m_device == VK_NULL_HANDLE || framesInFlight == 0) {
        return false;
    }

    auto capabilities = context.GetCapabilities().timestampQueries;
    if (queueFamily != VK_QUEUE_FAMILY_IGNORED) {
        const auto support = context.QueryTimestampSupport(queueFamily);
        capabilities.validBits = support.validBits;
        capabilities.supported = support.validBits > 0;
        capabilities.nanosecondsPerTick = context.GetDeviceProperties().limits.timestampPeriod;
        m_resetOnHost = support.resetOnHost;
    }
    if (!capabilities.supported) {
        m_device = VK_NULL_HANDLE;
        return false;
    }

    const uint32_t regionCapacity = std::clamp(maxRegionsPerFrame, 1u, rhi::kGpuTimestampMaxRegions);
    m_maxQueriesPerFrame = regionCapacity * 2;
    m_frames.resize(framesInFlight);
    m_queryScratch.resize(m_maxQueriesPerFrame);

    VkQueryPoolCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    createInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    createInfo.queryCount = m_maxQueriesPerFrame;

    for (auto &frame : m_frames) {
        frame.regions.resize(regionCapacity);
        if (vkCreateQueryPool(m_device, &createInfo, nullptr, &frame.pool) != VK_SUCCESS) {
            Destroy();
            return false;
        }
    }

    m_capabilities = capabilities;
    m_capabilities.maxRegionsPerFrame = regionCapacity;
    return true;
}

void GpuTimestampQueries::Destroy() noexcept
{
#if INFERNUX_FRAME_PROFILE
    if (s_recordingQueries == this)
        s_recordingQueries = nullptr;
#endif
    if (m_device != VK_NULL_HANDLE) {
        for (auto &frame : m_frames) {
            if (frame.pool != VK_NULL_HANDLE) {
                vkDestroyQueryPool(m_device, frame.pool, nullptr);
                frame.pool = VK_NULL_HANDLE;
            }
        }
    }
    m_frames.clear();
    m_queryScratch.clear();
    m_capabilities = {};
    m_latestFrame.Reset();
    m_recordingFrame = std::numeric_limits<uint32_t>::max();
    m_maxQueriesPerFrame = 0;
    m_resetOnHost = false;
    m_device = VK_NULL_HANDLE;
}

void GpuTimestampQueries::BeginFrame(VkCommandBuffer commandBuffer, uint32_t frameIndex,
                                     [[maybe_unused]] bool publishRecording)
{
#if INFERNUX_FRAME_PROFILE
    if (publishRecording)
        s_recordingQueries = nullptr;
#endif
    m_recordingFrame = std::numeric_limits<uint32_t>::max();
    if (!m_capabilities.supported || commandBuffer == VK_NULL_HANDLE || frameIndex >= m_frames.size()) {
        return;
    }

    auto &frame = m_frames[frameIndex];
    frame.queryCount = 0;
    frame.regionCount = 0;
    frame.serial = m_nextSerial++;
    frame.recorded = false;
    frame.pending = false;
    // Managed upload pools are recycled only after the submission fence.
    if (m_resetOnHost)
        vkResetQueryPool(m_device, frame.pool, 0, m_maxQueriesPerFrame);
    else
        vkCmdResetQueryPool(commandBuffer, frame.pool, 0, m_maxQueriesPerFrame);
    m_recordingFrame = frameIndex;
#if INFERNUX_FRAME_PROFILE
    if (publishRecording)
        s_recordingQueries = this;
#endif
}

rhi::TimestampRegionHandle GpuTimestampQueries::BeginRegion(VkCommandBuffer commandBuffer, std::string_view name,
                                                            VkPipelineStageFlagBits stage)
{
    auto *frame = CurrentRecordingFrame();
    if (!frame || commandBuffer == VK_NULL_HANDLE || frame->queryCount + 2 > m_maxQueriesPerFrame ||
        frame->regionCount >= frame->regions.size()) {
        return {};
    }

    const uint32_t regionIndex = frame->regionCount++;
    auto &region = frame->regions[regionIndex];
    region.name.fill('\0');
    const size_t copyCount = std::min(name.size(), region.name.size() - 1);
    std::copy_n(name.data(), copyCount, region.name.data());
    region.beginQuery = frame->queryCount++;
    region.endQuery = 0;
    region.ended = false;
    vkCmdWriteTimestamp(commandBuffer, stage, frame->pool, region.beginQuery);
    return {static_cast<uint16_t>(regionIndex)};
}

void GpuTimestampQueries::EndRegion(VkCommandBuffer commandBuffer, rhi::TimestampRegionHandle regionHandle,
                                    VkPipelineStageFlagBits stage)
{
    auto *frame = CurrentRecordingFrame();
    if (!frame || commandBuffer == VK_NULL_HANDLE || !regionHandle.IsValid() ||
        regionHandle.index >= frame->regionCount || frame->queryCount >= m_maxQueriesPerFrame) {
        return;
    }

    auto &region = frame->regions[regionHandle.index];
    if (region.ended) {
        return;
    }
    region.endQuery = frame->queryCount++;
    region.ended = true;
    vkCmdWriteTimestamp(commandBuffer, stage, frame->pool, region.endQuery);
}

void GpuTimestampQueries::FinishFrame([[maybe_unused]] VkCommandBuffer commandBuffer, uint32_t frameIndex) noexcept
{
    if (frameIndex < m_frames.size() && m_recordingFrame == frameIndex) {
        auto &frame = m_frames[frameIndex];
        frame.recorded = frame.queryCount > 0;
    }
    m_recordingFrame = std::numeric_limits<uint32_t>::max();
#if INFERNUX_FRAME_PROFILE
    if (s_recordingQueries == this)
        s_recordingQueries = nullptr;
#endif
}

void GpuTimestampQueries::MarkSubmitted(uint32_t frameIndex) noexcept
{
    if (frameIndex < m_frames.size()) {
        m_frames[frameIndex].pending = m_frames[frameIndex].recorded;
    }
}

std::optional<rhi::GpuTimestampFrame> GpuTimestampQueries::CollectCompletedFrame(uint32_t frameIndex)
{
    if (!m_capabilities.supported || frameIndex >= m_frames.size()) {
        return std::nullopt;
    }

    auto &frame = m_frames[frameIndex];
    if (!frame.pending || frame.queryCount == 0) {
        return std::nullopt;
    }

    // Every caller has observed this slot's terminal fence/ticket. Read its
    // query pool before reuse; no device-side result copy or extra wait is needed.
    std::fill_n(m_queryScratch.begin(), frame.queryCount, QueryValue{});
    const VkResult result = vkGetQueryPoolResults(
        m_device, frame.pool, 0, frame.queryCount, sizeof(QueryValue) * frame.queryCount, m_queryScratch.data(),
        sizeof(QueryValue), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (result != VK_SUCCESS)
        return std::nullopt;
    const auto *values = reinterpret_cast<const char *>(m_queryScratch.data());
    frame.pending = false;

    rhi::GpuTimestampFrame completed;
    completed.Reset(frame.serial);
    const uint32_t regionCount = std::min(frame.regionCount, static_cast<uint32_t>(frame.regions.size()));
    for (uint32_t i = 0; i < regionCount && completed.sampleCount < completed.samples.size(); ++i) {
        const auto &region = frame.regions[i];
        if (!region.ended || region.beginQuery >= frame.queryCount || region.endQuery >= frame.queryCount) {
            continue;
        }
        QueryValue begin, end;
        std::memcpy(&begin, values + region.beginQuery * sizeof(QueryValue), sizeof(QueryValue));
        std::memcpy(&end, values + region.endQuery * sizeof(QueryValue), sizeof(QueryValue));
        if (begin.available == 0 || end.available == 0) {
            continue;
        }

        auto &sample = completed.samples[completed.sampleCount++];
        sample.name = region.name;
        sample.milliseconds =
            static_cast<double>(rhi::TimestampTickDelta(begin.value, end.value, m_capabilities.validBits)) *
            m_capabilities.nanosecondsPerTick / 1.0e6;
    }
    completed.available = completed.sampleCount > 0;
    if (completed.serial >= m_latestFrame.serial)
        m_latestFrame = completed;
    return completed;
}

GpuTimestampQueries::FrameState *GpuTimestampQueries::CurrentRecordingFrame() noexcept
{
    if (m_recordingFrame >= m_frames.size()) {
        return nullptr;
    }
    return &m_frames[m_recordingFrame];
}

} // namespace infernux::vk
