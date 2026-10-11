#include "VulkanCommandUploads.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace infernux::vk
{

struct VulkanCommandUploads::Page
{
    VmaAllocator allocator;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void *mapped = nullptr;
    uint64_t capacity = 0;
    uint64_t used = 0;

    ~Page()
    {
        if (buffer)
            vmaDestroyBuffer(allocator, buffer, allocation);
    }
};

thread_local const VulkanCommandUploads::Scope *VulkanCommandUploads::s_scope = nullptr;

VulkanCommandUploads::VulkanCommandUploads(VmaAllocator allocator) : m_allocator(allocator)
{
    if (allocator) {
        const VkPhysicalDeviceProperties *properties = nullptr;
        vmaGetPhysicalDeviceProperties(allocator, &properties);
        m_alignment = (std::max)(uint64_t{4}, properties->limits.nonCoherentAtomSize);
    }
}

VulkanCommandUploads::~VulkanCommandUploads() = default;

void VulkanCommandUploads::Reset() noexcept
{
    // Drop pages skipped by the completed recording. Retain useful capacity,
    // rather than accumulating every historical allocation size forever.
    m_pages.erase(std::remove_if(m_pages.begin(), m_pages.end(), [](const auto &page) { return page->used == 0; }),
                  m_pages.end());
    for (auto &page : m_pages)
        page->used = 0;
    m_cursor = 0;
}

bool VulkanCommandUploads::Update(VkCommandBuffer commands, VkBuffer destination, uint64_t offset, const void *data,
                                  uint64_t byteSize)
{
    if (!m_allocator || !commands || !destination || !data || byteSize == 0 || byteSize % 4 != 0 || offset % 4 != 0 ||
        byteSize > std::numeric_limits<size_t>::max() ||
        byteSize > std::numeric_limits<uint64_t>::max() - (m_alignment - 1))
        return false;
    const uint64_t allocationBytes = (byteSize + m_alignment - 1) & ~(m_alignment - 1);
    while (m_cursor < m_pages.size() && allocationBytes > m_pages[m_cursor]->capacity - m_pages[m_cursor]->used)
        ++m_cursor;
    if (m_cursor == m_pages.size()) {
        auto page = std::make_unique<Page>();
        page->allocator = m_allocator;
        page->capacity = 1024 * 1024;
        while (page->capacity < allocationBytes) {
            if (page->capacity > std::numeric_limits<uint64_t>::max() / 2)
                return false;
            page->capacity *= 2;
        }
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = page->capacity;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo allocationInfo{};
        allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        allocationInfo.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo created{};
        if (vmaCreateBuffer(m_allocator, &bufferInfo, &allocationInfo, &page->buffer, &page->allocation, &created) !=
            VK_SUCCESS)
            return false;
        page->mapped = created.pMappedData;
        if (!page->mapped)
            return false;
        m_pages.push_back(std::move(page));
    }
    auto &page = *m_pages[m_cursor];
    const uint64_t sourceOffset = page.used;
    std::memcpy(static_cast<std::byte *>(page.mapped) + sourceOffset, data, static_cast<size_t>(byteSize));
    if (vmaFlushAllocation(m_allocator, page.allocation, sourceOffset, byteSize) != VK_SUCCESS)
        return false;
    // Reserve full non-coherent atoms: a later host flush must never overlap
    // a slice that an earlier submission could already be reading.
    page.used += allocationBytes;
    const VkBufferCopy copy{sourceOffset, offset, byteSize};
    vkCmdCopyBuffer(commands, page.buffer, destination, 1, &copy);
    return true;
}

VulkanCommandUploads::Scope::Scope(VulkanCommandUploads &storage, VkCommandBuffer commands) noexcept
    : m_storage(storage), m_commands(commands), m_previous(s_scope)
{
    s_scope = this;
}

VulkanCommandUploads::Scope::~Scope()
{
    s_scope = m_previous;
}

VulkanCommandUploads *VulkanCommandUploads::Current(VmaAllocator allocator, VkCommandBuffer commands) noexcept
{
    return s_scope && s_scope->m_commands == commands && s_scope->m_storage.m_allocator == allocator
               ? &s_scope->m_storage
               : nullptr;
}

} // namespace infernux::vk
