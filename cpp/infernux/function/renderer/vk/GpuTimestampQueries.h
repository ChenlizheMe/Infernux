#pragma once

#include "VulkanBackendApi.h"

#include <function/renderer/ProfileConfig.h>
#include <function/renderer/rhi/RhiQuery.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>
#include <vulkan/vulkan.h>

namespace infernux::vk
{

class VkDeviceContext;
class VkBufferHandle;

class GpuTimestampQueries
{
  public:
    GpuTimestampQueries() = default;
    INFERNUX_VULKAN_BACKEND_API ~GpuTimestampQueries();

    GpuTimestampQueries(const GpuTimestampQueries &) = delete;
    GpuTimestampQueries &operator=(const GpuTimestampQueries &) = delete;
    GpuTimestampQueries(GpuTimestampQueries &&) = delete;
    GpuTimestampQueries &operator=(GpuTimestampQueries &&) = delete;

    INFERNUX_VULKAN_BACKEND_API bool Initialize(const VkDeviceContext &context, uint32_t framesInFlight,
                                                uint32_t maxRegionsPerFrame = rhi::kGpuTimestampMaxRegions,
                                                uint32_t queueFamily = VK_QUEUE_FAMILY_IGNORED);
    INFERNUX_VULKAN_BACKEND_API void Destroy() noexcept;

    INFERNUX_VULKAN_BACKEND_API void BeginFrame(VkCommandBuffer commandBuffer, uint32_t frameIndex,
                                                bool publishRecording = true);
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API rhi::TimestampRegionHandle
    BeginRegion(VkCommandBuffer commandBuffer, std::string_view name, VkPipelineStageFlagBits stage);
    INFERNUX_VULKAN_BACKEND_API void EndRegion(VkCommandBuffer commandBuffer, rhi::TimestampRegionHandle region,
                                               VkPipelineStageFlagBits stage);
    INFERNUX_VULKAN_BACKEND_API void FinishFrame(VkCommandBuffer commandBuffer, uint32_t frameIndex) noexcept;
    INFERNUX_VULKAN_BACKEND_API void MarkSubmitted(uint32_t frameIndex) noexcept;
    /// The terminal fence for this slot must have retired before reading its results.
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API std::optional<rhi::GpuTimestampFrame>
    CollectCompletedFrame(uint32_t frameIndex);
#if INFERNUX_FRAME_PROFILE
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API static GpuTimestampQueries *CurrentRecording() noexcept;
#endif

    [[nodiscard]] const rhi::TimestampQueryCapabilities &Capabilities() const noexcept
    {
        return m_capabilities;
    }

    [[nodiscard]] const rhi::GpuTimestampFrame &LatestFrame() const noexcept
    {
        return m_latestFrame;
    }

  private:
    struct RegionRecord
    {
        std::array<char, rhi::kGpuTimestampNameCapacity> name{};
        uint32_t beginQuery = 0;
        uint32_t endQuery = 0;
        bool ended = false;
    };

    struct FrameState
    {
        VkQueryPool pool = VK_NULL_HANDLE;
        std::shared_ptr<VkBufferHandle> readback;
        std::vector<RegionRecord> regions;
        uint32_t queryCount = 0;
        uint32_t regionCount = 0;
        uint64_t serial = 0;
        bool recorded = false;
        bool pending = false;
    };

    struct QueryValue
    {
        uint64_t value = 0;
        uint64_t available = 0;
    };

    [[nodiscard]] FrameState *CurrentRecordingFrame() noexcept;
    VkDevice m_device = VK_NULL_HANDLE;
    std::vector<FrameState> m_frames;
    std::vector<QueryValue> m_queryScratch;
    rhi::TimestampQueryCapabilities m_capabilities;
    rhi::GpuTimestampFrame m_latestFrame;
    uint32_t m_recordingFrame = std::numeric_limits<uint32_t>::max();
    uint32_t m_maxQueriesPerFrame = 0;
    uint64_t m_nextSerial = 1;
    bool m_resetOnHost = false;
};

} // namespace infernux::vk
