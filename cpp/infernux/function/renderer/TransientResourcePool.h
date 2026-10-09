/**
 * @file TransientResourcePool.h
 * @brief Pool-based allocator for temporary render targets used by CommandBuffer.
 *
 * Used by the deferred command-buffer path.
 *
 * Manages the lifecycle of transient GPU images:
 * - Acquire(): allocate or reuse a render target matching (w, h, format, samples)
 * - Release(): mark a render target as reclaimable
 * - RetireReleased(): associate released targets with the exact GPU
 *   completion epoch that contains their last use

 * * - Collect(): move only completed targets back to the free pool
 * Hash-based pooling avoids recreating
 * VkImage/VkImageView every frame when the same (dimensions, format) are requested across frames.
 *
 * NOTE: Uses VMA (Vulkan Memory Allocator) for memory management.
 */

#pragma once

#include "rhi/RhiSubmission.h"

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace infernux
{

namespace vk
{
class VkDeviceContext;
class VkResourceManager;
} // namespace vk

/**
 * @brief Manages temporary render target allocation and pooling.
 */
class TransientResourcePool
{
  public:
    TransientResourcePool() = default;
    ~TransientResourcePool();

    // Non-copyable
    TransientResourcePool(const TransientResourcePool &) = delete;
    TransientResourcePool &operator=(const TransientResourcePool &) = delete;

    // ====================================================================
    // Lifecycle
    // ====================================================================

    /// @brief Initialize with Vulkan device context.
    void Initialize(vk::VkDeviceContext *deviceContext, vk::VkResourceManager *resourceManager);

    /// @brief Destroy all pooled resources.
    void Shutdown();

    // ====================================================================
    // Frame-Level Operations
    // ====================================================================

    /// @brief Allocate or reuse a temporary render target.
    /// @return Internal pool slot ID (not the same as RenderTargetHandle::id)
    uint32_t Acquire(int width, int height, VkFormat format, VkSampleCountFlagBits samples);

    /// @brief Mark a render target slot for recycling at end of frame.
    void Release(uint32_t slotId);

    /// @brief Associate all releases from the current recording phase with an
    /// exact GPU completion epoch. The slots remain unavailable until Collect.
    /// An invalid epoch is rejected; frame-age guesses are deliberately not
    /// supported because they can recycle an image still referenced by GPU work.
    void RetireReleased(rhi::SubmissionSerial completionEpoch);

    /// @brief Reclaim only slots whose GPU completion epoch has completed.
    void Collect(rhi::SubmissionSerial completedEpoch);

    /// @brief Return releases from a recording phase that was never submitted.
    /// The caller must use this only after proving that no command buffer from
    /// the phase reached a queue.
    void AbandonReleased();

    // ====================================================================
    // Accessors
    // ====================================================================

    /// @brief Get the VkImage for a pool slot.
    [[nodiscard]] VkImage GetImage(uint32_t slotId) const;

    /// @brief Get the VkImageView for a pool slot.
    [[nodiscard]] VkImageView GetImageView(uint32_t slotId) const;

    /// @brief Get dimensions of a pooled resource.
    void GetDimensions(uint32_t slotId, int &outWidth, int &outHeight) const;

    /// @brief Get the VkFormat of a pooled resource.
    [[nodiscard]] VkFormat GetFormat(uint32_t slotId) const;

    /// @brief Get total number of pool entries (allocated + free).
    [[nodiscard]] size_t GetTotalEntryCount() const noexcept
    {
        return m_entries.size();
    }

    /// @brief Get number of currently in-use entries.
    [[nodiscard]] size_t GetActiveEntryCount() const noexcept;
    [[nodiscard]] uint64_t GetResidentBytes() const noexcept
    {
        return m_residentBytes;
    }

  private:
    /// @brief A single pooled render target (image + view + memory).
    struct RTEntry
    {
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        uint64_t allocationBytes = 0;
        int width = 0;
        int height = 0;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
        bool inUse = false;
        bool pendingRelease = false; // released, awaiting RetireReleased()
    };

    /// @brief Hash key for matching compatible render targets.
    struct RTKey
    {
        int width;
        int height;
        VkFormat format;
        VkSampleCountFlagBits samples;

        bool operator==(const RTKey &o) const noexcept
        {
            return width == o.width && height == o.height && format == o.format && samples == o.samples;
        }
    };

    struct RTKeyHash
    {
        size_t operator()(const RTKey &k) const noexcept
        {
            size_t h = std::hash<int>()(k.width);
            h ^= std::hash<int>()(k.height) << 1;
            h ^= std::hash<uint32_t>()(static_cast<uint32_t>(k.format)) << 2;
            h ^= std::hash<uint32_t>()(static_cast<uint32_t>(k.samples)) << 3;
            return h;
        }
    };

    struct RetiringEntry
    {
        uint32_t slotId = UINT32_MAX;
        rhi::SubmissionSerial completionEpoch = rhi::InvalidSubmissionSerial;
    };

    /// @brief Create a new VkImage + VkImageView for a render target.
    uint32_t CreateEntry(int width, int height, VkFormat format, VkSampleCountFlagBits samples);

    /// @brief Destroy a single pool entry's Vulkan resources.
    void DestroyEntry(RTEntry &entry);

    void RecycleEntry(uint32_t slotId);

    // ----- State -----
    vk::VkDeviceContext *m_deviceContext = nullptr;
    vk::VkResourceManager *m_resourceManager = nullptr;

    std::vector<RTEntry> m_entries;

    // Free-list indexed by (w, h, fmt, samples): maps to vector of available slot IDs
    std::unordered_map<RTKey, std::vector<uint32_t>, RTKeyHash> m_freeList;
    std::vector<RetiringEntry> m_retiring;
    uint64_t m_residentBytes = 0;
};

} // namespace infernux
