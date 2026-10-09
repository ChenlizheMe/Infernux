/**
 * @file TransientResourcePool.cpp
 * @brief Implementation of the temporary render target pool.
 */

#include "TransientResourcePool.h"
#include "VkDeviceContext.h"
#include "VkResourceManager.h"
#include <cassert>
#include <core/log/InxLog.h>
#include <stdexcept>

namespace infernux
{

// ============================================================================
// Destructor
// ============================================================================

TransientResourcePool::~TransientResourcePool()
{
    Shutdown();
}

// ============================================================================
// Lifecycle
// ============================================================================

void TransientResourcePool::Initialize(vk::VkDeviceContext *deviceContext, vk::VkResourceManager *resourceManager)
{
    m_deviceContext = deviceContext;
    m_resourceManager = resourceManager;
}

void TransientResourcePool::Shutdown()
{
    if (!m_deviceContext)
        return;

    VkDevice device = m_deviceContext->GetDevice();
    if (device != VK_NULL_HANDLE) {
        // Skip WaitIdle during engine shutdown — caller already drained the GPU.
        if (!m_deviceContext->IsShuttingDown()) {
            vkDeviceWaitIdle(device);
        }
        for (auto &entry : m_entries) {
            DestroyEntry(entry);
        }
    }

    m_entries.clear();
    m_freeList.clear();
    m_retiring.clear();
    m_residentBytes = 0;
    m_deviceContext = nullptr;
    m_resourceManager = nullptr;
}

// ============================================================================
// Frame-Level Operations
// ============================================================================

uint32_t TransientResourcePool::Acquire(int width, int height, VkFormat format, VkSampleCountFlagBits samples)
{
    RTKey key{width, height, format, samples};

    // Try to reclaim from free list
    auto it = m_freeList.find(key);
    if (it != m_freeList.end() && !it->second.empty()) {
        uint32_t slotId = it->second.back();
        it->second.pop_back();

        auto &entry = m_entries[slotId];
        entry.inUse = true;
        entry.pendingRelease = false;
        return slotId;
    }

    // No match — create a new entry
    return CreateEntry(width, height, format, samples);
}

void TransientResourcePool::Release(uint32_t slotId)
{
    if (slotId >= m_entries.size()) {
        INXLOG_WARN("TransientResourcePool::Release — invalid slotId ", slotId);
        return;
    }

    auto &entry = m_entries[slotId];
    if (!entry.inUse) {
        INXLOG_WARN("TransientResourcePool::Release — slot ", slotId, " is not in use");
        return;
    }

    // Do not return to the free list here. The command buffer recorded for
    // this frame may still reference the image; RetireReleased() associates it
    // with the exact completion epoch after submission.
    entry.pendingRelease = true;
}

void TransientResourcePool::RetireReleased(rhi::SubmissionSerial completionEpoch)
{
    if (completionEpoch == rhi::InvalidSubmissionSerial)
        throw std::invalid_argument("TransientResourcePool requires a valid GPU completion epoch");

    for (uint32_t i = 0; i < static_cast<uint32_t>(m_entries.size()); ++i) {
        auto &entry = m_entries[i];
        if (entry.pendingRelease) {
            entry.inUse = false;
            entry.pendingRelease = false;
            m_retiring.push_back({i, completionEpoch});
        }
    }
}

void TransientResourcePool::Collect(rhi::SubmissionSerial completedEpoch)
{
    if (completedEpoch == rhi::InvalidSubmissionSerial && !m_retiring.empty())
        throw std::invalid_argument("TransientResourcePool requires a valid completed GPU epoch");

    size_t writeIndex = 0;
    for (size_t index = 0; index < m_retiring.size(); ++index) {
        const RetiringEntry retiring = m_retiring[index];
        if (retiring.completionEpoch <= completedEpoch) {
            RecycleEntry(retiring.slotId);
        } else {
            if (writeIndex != index)
                m_retiring[writeIndex] = retiring;
            ++writeIndex;
        }
    }
    m_retiring.resize(writeIndex);
}

void TransientResourcePool::AbandonReleased()
{
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_entries.size()); ++i) {
        auto &entry = m_entries[i];
        if (!entry.pendingRelease)
            continue;
        entry.inUse = false;
        entry.pendingRelease = false;
        RecycleEntry(i);
    }
}

// ============================================================================
// Accessors
// ============================================================================

VkImage TransientResourcePool::GetImage(uint32_t slotId) const
{
    if (slotId >= m_entries.size())
        return VK_NULL_HANDLE;
    return m_entries[slotId].image;
}

VkImageView TransientResourcePool::GetImageView(uint32_t slotId) const
{
    if (slotId >= m_entries.size())
        return VK_NULL_HANDLE;
    return m_entries[slotId].view;
}

void TransientResourcePool::GetDimensions(uint32_t slotId, int &outWidth, int &outHeight) const
{
    if (slotId >= m_entries.size()) {
        outWidth = outHeight = 0;
        return;
    }
    outWidth = m_entries[slotId].width;
    outHeight = m_entries[slotId].height;
}

VkFormat TransientResourcePool::GetFormat(uint32_t slotId) const
{
    if (slotId >= m_entries.size())
        return VK_FORMAT_UNDEFINED;
    return m_entries[slotId].format;
}

size_t TransientResourcePool::GetActiveEntryCount() const noexcept
{
    size_t count = 0;
    for (const auto &e : m_entries) {
        if (e.inUse)
            ++count;
    }
    return count;
}

void TransientResourcePool::RecycleEntry(uint32_t slotId)
{
    if (slotId >= m_entries.size())
        throw std::logic_error("TransientResourcePool retirement references an invalid slot");

    const auto &entry = m_entries[slotId];
    RTKey key{entry.width, entry.height, entry.format, entry.samples};
    m_freeList[key].push_back(slotId);
}

// ============================================================================
// Internal Helpers
// ============================================================================

uint32_t TransientResourcePool::CreateEntry(int width, int height, VkFormat format, VkSampleCountFlagBits samples)
{
    assert(m_deviceContext && "TransientResourcePool not initialized");

    VkDevice device = m_deviceContext->GetDevice();

    // --- Create VkImage ---
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.samples = samples;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    // --- Create image + allocate memory via VMA ---
    VmaAllocator allocator = m_deviceContext->GetVmaAllocator();

    VmaAllocationCreateInfo vmaAllocCreateInfo{};
    vmaAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo allocationInfo{};
    VkResult allocResult =
        vmaCreateImage(allocator, &imageInfo, &vmaAllocCreateInfo, &image, &allocation, &allocationInfo);
    if (allocResult != VK_SUCCESS) {
        INXLOG_ERROR("TransientResourcePool: Failed to create VkImage (", width, "x", height, ")");
        return UINT32_MAX;
    }

    // --- Create VkImageView ---
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView view = VK_NULL_HANDLE;
    if (vkCreateImageView(device, &viewInfo, nullptr, &view) != VK_SUCCESS) {
        INXLOG_ERROR("TransientResourcePool: Failed to create VkImageView");
        vmaDestroyImage(m_deviceContext->GetVmaAllocator(), image, allocation);
        return UINT32_MAX;
    }

    // --- Store entry ---
    uint32_t slotId = static_cast<uint32_t>(m_entries.size());
    RTEntry entry;
    entry.image = image;
    entry.view = view;
    entry.allocation = allocation;
    entry.allocationBytes = allocationInfo.size;
    entry.width = width;
    entry.height = height;
    entry.format = format;
    entry.samples = samples;
    entry.inUse = true;
    entry.pendingRelease = false;

    m_entries.push_back(entry);
    m_residentBytes += entry.allocationBytes;
    return slotId;
}

void TransientResourcePool::DestroyEntry(RTEntry &entry)
{
    if (!m_deviceContext)
        return;

    VkDevice device = m_deviceContext->GetDevice();
    if (device == VK_NULL_HANDLE)
        return;

    if (entry.view != VK_NULL_HANDLE) {
        vkDestroyImageView(device, entry.view, nullptr);
        entry.view = VK_NULL_HANDLE;
    }
    if (entry.image != VK_NULL_HANDLE) {
        if (entry.allocationBytes > m_residentBytes)
            throw std::logic_error("Transient render-target residency byte counter underflow");
        m_residentBytes -= entry.allocationBytes;
        vmaDestroyImage(m_deviceContext->GetVmaAllocator(), entry.image, entry.allocation);
        entry.image = VK_NULL_HANDLE;
        entry.allocation = VK_NULL_HANDLE;
        entry.allocationBytes = 0;
    }
}

} // namespace infernux
