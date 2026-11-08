#include "GpuTimestampQueries.h"

#include "VkDeviceContext.h"
#include "VkHandle.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace infernux::vk
{

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
    if (m_resetOnHost)
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
        // The same queue capability permits query reset and result copy.
        // Copy to mapped memory on the queue so completion collection never
        // needs a graphics/compute driver to read an active query pool.
        if (!m_resetOnHost) {
            frame.readback = std::make_shared<VkBufferHandle>();
            if (!frame.readback->Create(context.GetVmaAllocator(), m_device, sizeof(QueryValue) * m_maxQueriesPerFrame,
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ||
                !frame.readback->Map()) {
                Destroy();
                return false;
            }
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

void GpuTimestampQueries::FinishFrame(VkCommandBuffer commandBuffer, uint32_t frameIndex) noexcept
{
    if (frameIndex < m_frames.size() && m_recordingFrame == frameIndex) {
        auto &frame = m_frames[frameIndex];
        frame.recorded = frame.queryCount > 0;
        if (frame.recorded && frame.readback) {
            const auto buffer = frame.readback->GetBuffer();
            vkCmdCopyQueryPoolResults(commandBuffer, frame.pool, 0, frame.queryCount, buffer, 0, sizeof(QueryValue),
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT |
                                          VK_QUERY_RESULT_WAIT_BIT);
            VkBufferMemoryBarrier hostRead{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            hostRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            hostRead.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            hostRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            hostRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            hostRead.buffer = buffer;
            hostRead.size = sizeof(QueryValue) * frame.queryCount;
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0,
                                 nullptr, 1, &hostRead, 0, nullptr);
        }
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

    const char *values;
    if (frame.readback) {
        values = static_cast<const char *>(frame.readback->GetMappedPtr());
    } else {
        // Transfer-only command pools cannot record vkCmdCopyQueryPoolResults.
        std::fill_n(m_queryScratch.begin(), frame.queryCount, QueryValue{});
        const VkResult result = vkGetQueryPoolResults(
            m_device, frame.pool, 0, frame.queryCount, sizeof(QueryValue) * frame.queryCount, m_queryScratch.data(),
            sizeof(QueryValue), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (result != VK_SUCCESS)
            return std::nullopt;
        values = reinterpret_cast<const char *>(m_queryScratch.data());
    }
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
