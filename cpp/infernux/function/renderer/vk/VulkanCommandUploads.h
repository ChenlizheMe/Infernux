#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace infernux::vk
{

// Owned by one executor frame slot and queue role. Every recording appends
// immutable slices; only completion of that slot permits Reset and reuse.
class VulkanCommandUploads
{
  public:
    explicit VulkanCommandUploads(VmaAllocator allocator);
    ~VulkanCommandUploads();
    VulkanCommandUploads(const VulkanCommandUploads &) = delete;
    VulkanCommandUploads &operator=(const VulkanCommandUploads &) = delete;

    void Reset() noexcept;
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
        Scope(VulkanCommandUploads &storage, VkCommandBuffer commands) noexcept;
        ~Scope();
        Scope(const Scope &) = delete;
        Scope &operator=(const Scope &) = delete;

      private:
        friend class VulkanCommandUploads;
        VulkanCommandUploads &m_storage;
        VkCommandBuffer m_commands;
        const Scope *m_previous;
    };

    [[nodiscard]] static VulkanCommandUploads *Current(VmaAllocator allocator, VkCommandBuffer commands) noexcept;

  private:
    struct Page;
    VmaAllocator m_allocator;
    uint64_t m_alignment = 4;
    size_t m_cursor = 0;
    std::vector<std::unique_ptr<Page>> m_pages;
    static thread_local const Scope *s_scope;
};

} // namespace infernux::vk
