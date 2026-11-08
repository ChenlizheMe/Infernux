#include "VulkanCommandUploads.h"
#include "VulkanRhiDevice.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace infernux::vk
{

thread_local const VulkanCommandUploads::Scope *VulkanCommandUploads::s_scope = nullptr;

VulkanCommandUploads::VulkanCommandUploads(VulkanRhiDevice &device) noexcept : m_device(device)
{
}

VulkanCommandUploads::~VulkanCommandUploads() = default;

bool VulkanCommandUploads::AddPage(uint64_t bytes, bool managed)
{
    auto buffer = m_device.AcquireStagingBuffer(bytes, managed);
    if (!buffer)
        return false;
    auto *mapped = static_cast<std::byte *>(buffer->Map());
    if (!mapped)
        return false;
    m_pages.push_back({std::move(buffer), mapped});
    return true;
}

bool VulkanCommandUploads::Prepare(uint64_t bytes, bool managed)
{
    if (bytes && !AddPage(bytes, managed))
        return false;
    m_prepared = true;
    return true;
}

uint64_t VulkanCommandUploads::RetainedBytes() const noexcept
{
    uint64_t bytes = 0;
    for (const auto &page : m_pages)
        bytes += page.buffer->GetSize();
    return bytes;
}

void VulkanCommandUploads::Recycle() noexcept
{
    for (auto &page : m_pages)
        m_device.RecycleStagingBuffer(std::move(page.buffer));
    Discard();
}

void VulkanCommandUploads::Discard() noexcept
{
    m_pages.clear();
    m_cursor = 0;
    m_prepared = false;
}

bool VulkanCommandUploads::Update(VkCommandBuffer commands, VkBuffer destination, uint64_t offset, const void *data,
                                  uint64_t byteSize)
{
    if (!commands || !destination || !data || byteSize == 0 || byteSize % 4 != 0 || offset % 4 != 0 ||
        byteSize > std::numeric_limits<size_t>::max())
        return false;
    while (m_cursor < m_pages.size() && byteSize > m_pages[m_cursor].buffer->GetSize() - m_pages[m_cursor].used)
        ++m_cursor;
    if (m_cursor == m_pages.size() && (m_prepared || !AddPage((std::max)(uint64_t{1024 * 1024}, byteSize), false)))
        return false;
    auto &page = m_pages[m_cursor];
    const uint64_t sourceOffset = page.used;
    // The shared staging pool uses coherent host memory for both IO paths.
    std::memcpy(page.mapped + sourceOffset, data, static_cast<size_t>(byteSize));
    page.used += byteSize;
    const VkBufferCopy copy{sourceOffset, offset, byteSize};
    vkCmdCopyBuffer(commands, page.buffer->GetBuffer(), destination, 1, &copy);
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

VulkanCommandUploads *VulkanCommandUploads::Current(const VulkanRhiDevice &device, VkCommandBuffer commands) noexcept
{
    return s_scope && s_scope->m_commands == commands && &s_scope->m_storage.m_device == &device ? &s_scope->m_storage
                                                                                                 : nullptr;
}

} // namespace infernux::vk
