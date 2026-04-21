/**
 * @file RenderPassOutput.cpp
 * @brief Implementation of RenderPassOutput for GPU->CPU readback
 */

#include "RenderPassOutput.h"
#include "InxVkCoreModular.h"
#include "vk/VkDeviceContext.h"
#include "vk/VkRenderUtils.h"
#include <array>
#include <core/error/InxError.h>
#include <imgui.h>
#include <imgui_impl_vulkan.h>

namespace infernux
{

RenderPassOutput::RenderPassOutput(InxVkCoreModular *vkCore) : m_vkCore(vkCore)
{
}

RenderPassOutput::~RenderPassOutput()
{
    Cleanup();
}

bool RenderPassOutput::Initialize(const std::string &name, uint32_t width, uint32_t height, VkFormat format,
                                  bool includeDepth)
{
    if (!m_vkCore) {
        INXLOG_ERROR("RenderPassOutput::Initialize: vkCore is null");
        return false;
    }

    m_name = name;
    m_width = width;
    m_height = height;
    m_format = format;
    m_includeDepth = includeDepth;

    try {
        CreateColorAttachment();
        if (m_includeDepth) {
            CreateDepthAttachment();
        }
        CreateStagingBuffer();
        CreateImGuiDescriptor();

        m_isInitialized = true;
        INXLOG_INFO("RenderPassOutput '", name, "' initialized: ", width, "x", height);
        return true;
    } catch (const std::exception &e) {
        INXLOG_ERROR("RenderPassOutput::Initialize failed: ", e.what());
        CleanupResources();
        return false;
    }
}

void RenderPassOutput::Resize(uint32_t width, uint32_t height)
{
    if (width == m_width && height == m_height) {
        return;
    }

    CleanupResources();
    m_width = width;
    m_height = height;

    try {
        CreateColorAttachment();
        if (m_includeDepth) {
            CreateDepthAttachment();
        }
        CreateStagingBuffer();
        CreateImGuiDescriptor();

        m_isInitialized = true;
        INXLOG_DEBUG("RenderPassOutput '", m_name, "' resized: ", width, "x", height);
    } catch (const std::exception &e) {
        INXLOG_ERROR("RenderPassOutput::Resize failed: ", e.what());
        m_isInitialized = false;
    }
}

void RenderPassOutput::Cleanup()
{
    if (m_vkCore) {
        if (!m_vkCore->IsShuttingDown()) {
            m_vkCore->GetDeviceContext().WaitIdle();
        }
    }
    CleanupResources();
    m_isInitialized = false;
}

bool RenderPassOutput::ReadbackColorPixels(std::vector<uint8_t> &outData)
{
    if (!m_isInitialized || !m_colorStagingBuffer) {
        INXLOG_ERROR("RenderPassOutput::ReadbackColorPixels: not initialized");
        return false;
    }

    VkDevice device = m_vkCore->GetDevice();

    // Copy image to staging buffer
    VkCommandBuffer cmdBuf = m_vkCore->BeginSingleTimeCommands();

    // Transition image to transfer src
    VkImageMemoryBarrier barrier = vkrender::MakeImageBarrier(
        m_colorImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT);

    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &barrier);

    // Copy image to buffer
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {m_width, m_height, 1};

    vkCmdCopyImageToBuffer(cmdBuf, m_colorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_colorStagingBuffer, 1,
                           &region);

    // Transition back to shader read
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &barrier);

    m_vkCore->EndSingleTimeCommands(cmdBuf);

    // Map staging buffer and copy data
    VmaAllocator allocator = m_vkCore->GetDeviceContext().GetVmaAllocator();
    void *data;
    if (vmaMapMemory(allocator, m_colorStagingAllocation, &data) != VK_SUCCESS) {
        INXLOG_ERROR("RenderPassOutput::ReadbackColorPixels: failed to map staging memory");
        return false;
    }

    outData.resize(static_cast<size_t>(m_colorStagingSize));
    memcpy(outData.data(), data, static_cast<size_t>(m_colorStagingSize));

    vmaUnmapMemory(allocator, m_colorStagingAllocation);

    return true;
}

bool RenderPassOutput::ReadbackDepthPixels(std::vector<float> &outData)
{
    if (!m_isInitialized || !m_depthStagingBuffer || !m_includeDepth) {
        INXLOG_ERROR("RenderPassOutput::ReadbackDepthPixels: not initialized or no depth");
        return false;
    }

    VkCommandBuffer cmdBuf = m_vkCore->BeginSingleTimeCommands();

    // Transition depth image to transfer src
    VkImageMemoryBarrier barrier = vkrender::MakeImageBarrier(
        m_depthImage, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_ASPECT_DEPTH_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT);

    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &barrier);

    // Copy depth image to staging buffer
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {m_width, m_height, 1};

    vkCmdCopyImageToBuffer(cmdBuf, m_depthImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_depthStagingBuffer, 1,
                           &region);

    // Transition back to depth read-only
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;

    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &barrier);

    m_vkCore->EndSingleTimeCommands(cmdBuf);

    // Map staging buffer and copy depth data
    VmaAllocator allocator = m_vkCore->GetDeviceContext().GetVmaAllocator();
    void *data;
    if (vmaMapMemory(allocator, m_depthStagingAllocation, &data) != VK_SUCCESS) {
        INXLOG_ERROR("RenderPassOutput::ReadbackDepthPixels: failed to map staging memory");
        return false;
    }

    size_t pixelCount = static_cast<size_t>(m_width) * static_cast<size_t>(m_height);
    outData.resize(pixelCount);
    memcpy(outData.data(), data, pixelCount * sizeof(float));

    vmaUnmapMemory(allocator, m_depthStagingAllocation);

    return true;
}

// ============================================================================
// Private Methods
// ============================================================================

void RenderPassOutput::CreateColorAttachment()
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = m_width;
    imageInfo.extent.height = m_height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = m_format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // Add TRANSFER_SRC for readback support
    imageInfo.usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    // Create image + allocate memory via VMA (combined create+alloc+bind)
    VmaAllocator allocator = m_vkCore->GetDeviceContext().GetVmaAllocator();
    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VkResult result =
        vmaCreateImage(allocator, &imageInfo, &allocCreateInfo, &m_colorImage, &m_colorAllocation, nullptr);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create color image via VMA");
    }

    // Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = m_colorImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = m_format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(m_vkCore->GetDevice(), &viewInfo, nullptr, &m_colorImageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create color image view");
    }

    // Transition to shader read optimal
    VkCommandBuffer cmdBuf = m_vkCore->BeginSingleTimeCommands();

    VkImageMemoryBarrier barrier =
        vkrender::MakeImageBarrier(m_colorImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_ACCESS_SHADER_READ_BIT);

    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &barrier);

    m_vkCore->EndSingleTimeCommands(cmdBuf);
}

void RenderPassOutput::CreateDepthAttachment()
{
    VkFormat depthFormat = m_vkCore->GetDeviceContext().FindDepthFormat();

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = m_width;
    imageInfo.extent.height = m_height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = depthFormat;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    VmaAllocator allocator = m_vkCore->GetDeviceContext().GetVmaAllocator();
    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VkResult result =
        vmaCreateImage(allocator, &imageInfo, &allocCreateInfo, &m_depthImage, &m_depthAllocation, nullptr);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create depth image via VMA");
    }

    // Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = m_depthImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = depthFormat;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(m_vkCore->GetDevice(), &viewInfo, nullptr, &m_depthImageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create depth image view");
    }
}

void RenderPassOutput::CreateStagingBuffer()
{
    // Calculate buffer size for RGBA8
    m_colorStagingSize = static_cast<VkDeviceSize>(m_width) * m_height * 4;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = m_colorStagingSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocator allocator = m_vkCore->GetDeviceContext().GetVmaAllocator();
    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    allocCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VkResult result = vmaCreateBuffer(allocator, &bufferInfo, &allocCreateInfo, &m_colorStagingBuffer,
                                      &m_colorStagingAllocation, nullptr);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create staging buffer via VMA");
    }
}

void RenderPassOutput::CreateImGuiDescriptor()
{
    // Create sampler
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(m_vkCore->GetDevice(), &samplerInfo, nullptr, &m_sampler) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create sampler");
    }

    // Create ImGui descriptor set for display
    m_imguiDescriptorSet =
        ImGui_ImplVulkan_AddTexture(m_sampler, m_colorImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void RenderPassOutput::CleanupResources()
{
    if (!m_vkCore) {
        return;
    }

    VkDevice device = m_vkCore->GetDevice();
    if (device == VK_NULL_HANDLE) {
        return;
    }

    VmaAllocator allocator = m_vkCore->GetDeviceContext().GetVmaAllocator();

    if (m_imguiDescriptorSet != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(m_imguiDescriptorSet);
        m_imguiDescriptorSet = VK_NULL_HANDLE;
    }

    if (m_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, m_sampler, nullptr);
        m_sampler = VK_NULL_HANDLE;
    }

    if (m_colorStagingBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator, m_colorStagingBuffer, m_colorStagingAllocation);
        m_colorStagingBuffer = VK_NULL_HANDLE;
        m_colorStagingAllocation = VK_NULL_HANDLE;
    }

    if (m_depthStagingBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator, m_depthStagingBuffer, m_depthStagingAllocation);
        m_depthStagingBuffer = VK_NULL_HANDLE;
        m_depthStagingAllocation = VK_NULL_HANDLE;
    }

    if (m_colorImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, m_colorImageView, nullptr);
        m_colorImageView = VK_NULL_HANDLE;
    }
    if (m_colorImage != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator, m_colorImage, m_colorAllocation);
        m_colorImage = VK_NULL_HANDLE;
        m_colorAllocation = VK_NULL_HANDLE;
    }

    if (m_depthImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, m_depthImageView, nullptr);
        m_depthImageView = VK_NULL_HANDLE;
    }
    if (m_depthImage != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator, m_depthImage, m_depthAllocation);
        m_depthImage = VK_NULL_HANDLE;
        m_depthAllocation = VK_NULL_HANDLE;
    }
}

} // namespace infernux
