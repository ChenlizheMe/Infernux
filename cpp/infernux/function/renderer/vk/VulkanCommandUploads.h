#pragma once

#include "VulkanBackendApi.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace infernux::vk
{

class VulkanRhiDevice;
class VkBufferHandle;

// Owned by one executor frame slot and queue role. Every recording appends
// immutable slices; only completion permits return to the device staging pool.
class VulkanCommandUploads
{
  public:
    INFERNUX_VULKAN_BACKEND_API explicit VulkanCommandUploads(VulkanRhiDevice &device) noexcept;
    INFERNUX_VULKAN_BACKEND_API ~VulkanCommandUploads();
    VulkanCommandUploads(const VulkanCommandUploads &) = delete;
    VulkanCommandUploads &operator=(const VulkanCommandUploads &) = delete;

    /// Pin one page for all known large writes before recording starts.
    [[nodiscard]] bool Prepare(uint64_t bytes, bool managed);
    INFERNUX_VULKAN_BACKEND_API void Recycle() noexcept;
    void Discard() noexcept;
    [[nodiscard]] uint64_t RetainedBytes() const noexcept;
    [[nodiscard]] bool Update(VkCommandBuffer commands, VkBuffer destination, uint64_t offset, const void *data,
                              uint64_t byteSize);
    [[nodiscard]] size_t PageCount() const noexcept
    {
        return m_pages.size();
    }

    // Native callers outside the submission executor can own the same explicit
    // scope. The owner must retain this storage until its submissions complete.
    class Scope
    {
      public:
        INFERNUX_VULKAN_BACKEND_API Scope(VulkanCommandUploads &storage, VkCommandBuffer commands) noexcept;
        INFERNUX_VULKAN_BACKEND_API ~Scope();
        Scope(const Scope &) = delete;
        Scope &operator=(const Scope &) = delete;

      private:
        friend class VulkanCommandUploads;
        VulkanCommandUploads &m_storage;
        VkCommandBuffer m_commands;
        const Scope *m_previous;
    };

    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API static VulkanCommandUploads *Current(const VulkanRhiDevice &device,
                                                                                   VkCommandBuffer commands) noexcept;

  private:
    struct Page
    {
        std::shared_ptr<VkBufferHandle> buffer;
        std::byte *mapped;
        uint64_t used = 0;
    };
    [[nodiscard]] bool AddPage(uint64_t bytes, bool managed);
    VulkanRhiDevice &m_device;
    size_t m_cursor = 0;
    bool m_prepared = false;
    std::vector<Page> m_pages;
    static thread_local const Scope *s_scope;
};

} // namespace infernux::vk
