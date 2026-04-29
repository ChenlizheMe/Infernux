/**
 * @file VkResourceManager.cpp
 * @brief Implementation of Vulkan resource management
 */

#include "VkResourceManager.h"
#include "AsyncTransferContext.h"
#include "VkDeviceContext.h"
#include <SDL3/SDL.h>
#include <core/error/InxError.h>
#include <function/resources/InxFileLoader/InxTextureLoader.hpp>
#include <platform/filesystem/InxPath.h>

// STB_IMAGE_IMPLEMENTATION moved here after InxVkCore removal
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace infernux
{
namespace vk
{

namespace
{

void WaitForFencePumpingEvents(VkDevice device, VkFence fence)
{
    constexpr uint64_t kPollTimeoutNs = 50'000'000; // 50 ms
    while (true) {
        VkResult result = vkWaitForFences(device, 1, &fence, VK_TRUE, kPollTimeoutNs);
        if (result == VK_SUCCESS) {
            return;
        }
        if (result != VK_TIMEOUT) {
            INXLOG_ERROR("VkResourceManager::EndSingleTimeCommands fence wait failed: ", result);
            return;
        }
        SDL_PumpEvents();
    }
}

} // namespace

// ============================================================================
// Constructor / Destructor / Move
// ============================================================================

VkResourceManager::~VkResourceManager()
{
    Destroy();
}

VkResourceManager::VkResourceManager(VkResourceManager &&other) noexcept
    : m_device(other.m_device), m_physicalDevice(other.m_physicalDevice), m_vmaAllocator(other.m_vmaAllocator),
      m_graphicsQueue(other.m_graphicsQueue), m_commandPool(other.m_commandPool),
      m_linearSampler(other.m_linearSampler), m_nearestSampler(other.m_nearestSampler),
      m_descriptorPools(std::move(other.m_descriptorPools))
{
    other.m_device = VK_NULL_HANDLE;
    other.m_physicalDevice = VK_NULL_HANDLE;
    other.m_vmaAllocator = VK_NULL_HANDLE;
    other.m_graphicsQueue = VK_NULL_HANDLE;
    other.m_commandPool = VK_NULL_HANDLE;
    other.m_linearSampler = VK_NULL_HANDLE;
    other.m_nearestSampler = VK_NULL_HANDLE;
}

VkResourceManager &VkResourceManager::operator=(VkResourceManager &&other) noexcept
{
    if (this != &other) {
        Destroy();

        m_device = other.m_device;
        m_physicalDevice = other.m_physicalDevice;
        m_vmaAllocator = other.m_vmaAllocator;
        m_graphicsQueue = other.m_graphicsQueue;
        m_commandPool = other.m_commandPool;
        m_linearSampler = other.m_linearSampler;
        m_nearestSampler = other.m_nearestSampler;
        m_descriptorPools = std::move(other.m_descriptorPools);

        other.m_device = VK_NULL_HANDLE;
        other.m_physicalDevice = VK_NULL_HANDLE;
        other.m_vmaAllocator = VK_NULL_HANDLE;
        other.m_graphicsQueue = VK_NULL_HANDLE;
        other.m_commandPool = VK_NULL_HANDLE;
        other.m_linearSampler = VK_NULL_HANDLE;
        other.m_nearestSampler = VK_NULL_HANDLE;
    }
    return *this;
}

// ============================================================================
// Initialization
// ============================================================================

bool VkResourceManager::Initialize(const VkDeviceContext &context)
{
    m_device = context.GetDevice();
    m_physicalDevice = context.GetPhysicalDevice();
    m_vmaAllocator = context.GetVmaAllocator();
    m_graphicsQueue = context.GetGraphicsQueue();

    // Create command pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = context.GetQueueIndices().graphicsFamily.value();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to create command pool");
        return false;
    }

    INXLOG_INFO("VkResourceManager initialized");
    return true;
}

void VkResourceManager::Destroy() noexcept
{
    if (m_device == VK_NULL_HANDLE) {
        return;
    }

    if (!m_skipWaitIdle) {
        vkDeviceWaitIdle(m_device);
    }

    // Destroy samplers
    if (m_linearSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_linearSampler, nullptr);
        m_linearSampler = VK_NULL_HANDLE;
    }

    if (m_nearestSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_nearestSampler, nullptr);
        m_nearestSampler = VK_NULL_HANDLE;
    }

    // Destroy descriptor pools
    for (auto pool : m_descriptorPools) {
        vkDestroyDescriptorPool(m_device, pool, nullptr);
    }
    m_descriptorPools.clear();

    // Tear down the single-time-command pools.
    // m_allSingleTimeCmdBuffers is owned by m_commandPool — destroying the
    // pool below frees them implicitly, so we only need to clear the lists.
    {
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        for (VkFence fence : m_allSingleTimeFences) {
            if (fence != VK_NULL_HANDLE) {
                vkDestroyFence(m_device, fence, nullptr);
            }
        }
        m_allSingleTimeFences.clear();
        m_freeSingleTimeFences.clear();
        m_allSingleTimeCmdBuffers.clear();
        m_freeSingleTimeCmdBuffers.clear();
    }

    // Destroy command pool
    if (m_commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device, m_commandPool, nullptr);
        m_commandPool = VK_NULL_HANDLE;
    }

    m_device = VK_NULL_HANDLE;
    m_physicalDevice = VK_NULL_HANDLE;
    m_graphicsQueue = VK_NULL_HANDLE;
}

// ============================================================================
// Buffer Management
// ============================================================================

std::unique_ptr<VkBufferHandle> VkResourceManager::CreateVertexBuffer(const void *data, VkDeviceSize size)
{
    // Create staging buffer
    auto stagingBuffer = CreateStagingBuffer(size);
    if (!stagingBuffer) {
        return nullptr;
    }

    // Copy data to staging buffer
    stagingBuffer->CopyFrom(data, size, 0);

    // Create device-local vertex buffer
    auto vertexBuffer = CreateBufferInternal(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (!vertexBuffer) {
        return nullptr;
    }

    // Copy from staging to vertex buffer
    CopyBuffer(stagingBuffer->GetBuffer(), vertexBuffer->GetBuffer(), size);

    return vertexBuffer;
}

std::unique_ptr<VkBufferHandle> VkResourceManager::CreateIndexBuffer(const void *data, VkDeviceSize size)
{
    // Create staging buffer
    auto stagingBuffer = CreateStagingBuffer(size);
    if (!stagingBuffer) {
        return nullptr;
    }

    // Copy data to staging buffer
    stagingBuffer->CopyFrom(data, size, 0);

    // Create device-local index buffer
    auto indexBuffer = CreateBufferInternal(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (!indexBuffer) {
        return nullptr;
    }

    // Copy from staging to index buffer
    CopyBuffer(stagingBuffer->GetBuffer(), indexBuffer->GetBuffer(), size);

    return indexBuffer;
}

std::unique_ptr<VkBufferHandle> VkResourceManager::CreateUniformBuffer(VkDeviceSize size)
{
    // TRANSFER_DST_BIT is required for vkCmdUpdateBuffer (multi-camera UBO updates)
    return CreateBufferInternal(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

std::unique_ptr<VkBufferHandle> VkResourceManager::CreateStorageBuffer(VkDeviceSize size, bool deviceLocal)
{
    VkMemoryPropertyFlags properties =
        deviceLocal ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                    : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (deviceLocal) {
        usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }

    return CreateBufferInternal(size, usage, properties);
}

std::unique_ptr<VkBufferHandle> VkResourceManager::CreateStagingBuffer(VkDeviceSize size)
{
    return CreateBufferInternal(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

void VkResourceManager::CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
{
    VkCommandBuffer cmdBuffer = BeginSingleTimeCommands();

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(cmdBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    EndSingleTimeCommands(cmdBuffer);
}

std::unique_ptr<VkBufferHandle> VkResourceManager::CreateBufferInternal(VkDeviceSize size, VkBufferUsageFlags usage,
                                                                        VkMemoryPropertyFlags properties)
{
    auto buffer = std::make_unique<VkBufferHandle>();
    if (!buffer->Create(m_vmaAllocator, m_device, size, usage, properties)) {
        return nullptr;
    }
    return buffer;
}

// ============================================================================
// Image and Texture Management
// ============================================================================

std::unique_ptr<VkImageHandle> VkResourceManager::CreateImage(uint32_t width, uint32_t height, VkFormat format,
                                                              VkImageUsageFlags usage, VkMemoryPropertyFlags properties)
{
    auto image = std::make_unique<VkImageHandle>();
    if (!image->Create(m_vmaAllocator, m_device, width, height, format, VK_IMAGE_TILING_OPTIMAL, usage, properties)) {
        return nullptr;
    }
    return image;
}

std::unique_ptr<VkImageHandle> VkResourceManager::CreateDepthBuffer(uint32_t width, uint32_t height, VkFormat format)
{
    if (format == VK_FORMAT_UNDEFINED) {
        format = FindDepthFormat();
    }

    auto depthImage = CreateImage(width, height, format, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (!depthImage) {
        return nullptr;
    }

    // Create image view
    if (!depthImage->CreateView(format, VK_IMAGE_ASPECT_DEPTH_BIT, 1)) {
        return nullptr;
    }

    return depthImage;
}

std::unique_ptr<VkTexture> VkResourceManager::LoadTexture(const std::string &filePath, bool generateMipmaps,
                                                          VkFormat format, int maxSize, bool normalMapMode,
                                                          VkFilter filter, VkSamplerAddressMode addressMode, int aniso)
{
    int texWidth, texHeight, texChannels;
    // Read file bytes first to support Unicode paths on Windows
    std::vector<unsigned char> fileBytes;
    if (!ReadFileBytes(filePath, fileBytes) || fileBytes.empty()) {
        INXLOG_ERROR("Failed to read texture file: ", filePath);
        return nullptr;
    }

    // Detect HDR images (stbi_is_hdr_from_memory checks file header)
    bool isHdr = stbi_is_hdr_from_memory(fileBytes.data(), static_cast<int>(fileBytes.size())) != 0;

    if (isHdr) {
        // HDR path: load as float data → VK_FORMAT_R32G32B32A32_SFLOAT
        float *floatPixels = stbi_loadf_from_memory(fileBytes.data(), static_cast<int>(fileBytes.size()), &texWidth,
                                                    &texHeight, &texChannels, STBI_rgb_alpha);
        if (!floatPixels) {
            INXLOG_ERROR("Failed to load HDR texture: ", filePath);
            return nullptr;
        }

        VkFormat hdrFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
        auto texture = std::make_unique<VkTexture>();
        if (!texture->CreateFromPixels(m_vmaAllocator, m_device, m_physicalDevice, m_commandPool, m_graphicsQueue,
                                       reinterpret_cast<const unsigned char *>(floatPixels), texWidth, texHeight,
                                       hdrFormat, generateMipmaps, filter, addressMode, aniso)) {
            stbi_image_free(floatPixels);
            return nullptr;
        }
        stbi_image_free(floatPixels);
        return texture;
    }

    // LDR path: load as 8-bit RGBA
    stbi_uc *pixels = stbi_load_from_memory(fileBytes.data(), static_cast<int>(fileBytes.size()), &texWidth, &texHeight,
                                            &texChannels, STBI_rgb_alpha);
    InxTextureData pnmFallback;

    if (!pixels) {
        pnmFallback = InxTextureLoader::LoadFromMemory(fileBytes.data(), fileBytes.size(), filePath);
        if (!pnmFallback.IsValid()) {
            INXLOG_ERROR("Failed to load texture: ", filePath);
            return nullptr;
        }
        texWidth = pnmFallback.width;
        texHeight = pnmFallback.height;
        texChannels = pnmFallback.channels;
    }

    // Apply max_size clamping: downscale if either dimension exceeds maxSize
    uint32_t finalW = static_cast<uint32_t>(texWidth);
    uint32_t finalH = static_cast<uint32_t>(texHeight);
    std::vector<unsigned char> resizedBuf;
    const unsigned char *basePixels = pixels ? pixels : pnmFallback.pixels.data();

    if (normalMapMode) {
        INXLOG_INFO("LoadTexture: preserving authored tangent-space normal map '", filePath, "'");
    }

    if (maxSize > 0 && (texWidth > maxSize || texHeight > maxSize)) {
        float scale = static_cast<float>(maxSize) / static_cast<float>((std::max)(texWidth, texHeight));
        finalW = (std::max)(1u, static_cast<uint32_t>(texWidth * scale));
        finalH = (std::max)(1u, static_cast<uint32_t>(texHeight * scale));

        resizedBuf.resize(finalW * finalH * 4);
        stbir_resize_uint8_linear(basePixels, texWidth, texHeight, texWidth * 4, resizedBuf.data(),
                                  static_cast<int>(finalW), static_cast<int>(finalH), static_cast<int>(finalW * 4),
                                  STBIR_RGBA);

        INXLOG_INFO("LoadTexture: resized '", filePath, "' from ", texWidth, "x", texHeight, " to ", finalW, "x",
                    finalH, " (maxSize=", maxSize, ")");
    }

    const unsigned char *srcPixels = resizedBuf.empty() ? basePixels : resizedBuf.data();
    auto texture =
        CreateTextureFromPixels(srcPixels, finalW, finalH, format, generateMipmaps, filter, addressMode, aniso);

    if (pixels) {
        stbi_image_free(pixels);
    }

    return texture;
}

std::unique_ptr<VkTexture> VkResourceManager::CreateTextureFromPixels(const unsigned char *pixels, uint32_t width,
                                                                      uint32_t height, VkFormat format,
                                                                      bool generateMipmaps, VkFilter filter,
                                                                      VkSamplerAddressMode addressMode, int aniso)
{
    auto texture = std::make_unique<VkTexture>();

    // Phase 5d-1: prefer the async transfer queue when the device exposes
    // a dedicated DMA family AND we don't need graphics-queue features
    // (e.g. mipmap blits). CreateFromPixelsAsync returns false when it
    // can't satisfy the request, in which case we transparently fall
    // through to the legacy synchronous graphics-queue path so callers
    // never observe the difference beyond perf.
    if (m_asyncTransfer != nullptr && m_asyncTransfer->IsAsyncCapable()) {
        if (texture->CreateFromPixelsAsync(m_vmaAllocator, m_device, m_physicalDevice, *m_asyncTransfer,
                                           m_graphicsQueueFamily, pixels, width, height, format, generateMipmaps,
                                           filter, addressMode, aniso)) {
            return texture;
        }
        // Reset the in-progress texture so the synchronous fallback below
        // gets a clean handle to write into.
        texture = std::make_unique<VkTexture>();
    }

    if (!texture->CreateFromPixels(m_vmaAllocator, m_device, m_physicalDevice, m_commandPool, m_graphicsQueue, pixels,
                                   width, height, format, generateMipmaps, filter, addressMode, aniso)) {
        return nullptr;
    }

    return texture;
}

std::unique_ptr<VkTexture> VkResourceManager::CreateSolidColorTexture(uint32_t width, uint32_t height, uint8_t r,
                                                                      uint8_t g, uint8_t b, uint8_t a, VkFormat format)
{
    auto texture = std::make_unique<VkTexture>();

    if (!texture->CreateSolidColor(m_vmaAllocator, m_device, m_physicalDevice, m_commandPool, m_graphicsQueue, width,
                                   height, r, g, b, a, format)) {
        return nullptr;
    }

    return texture;
}

void VkResourceManager::TransitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout,
                                              VkImageLayout newLayout)
{
    VkCommandBuffer cmdBuffer = BeginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (HasStencilComponent(format)) {
            barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
    } else {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }

    VkPipelineStageFlags srcStage;
    VkPipelineStageFlags dstStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
               newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask =
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    } else {
        INXLOG_WARN("Unsupported layout transition");
        srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }

    vkCmdPipelineBarrier(cmdBuffer, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    EndSingleTimeCommands(cmdBuffer);
}

void VkResourceManager::CopyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
{
    VkCommandBuffer cmdBuffer = BeginSingleTimeCommands();

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};

    vkCmdCopyBufferToImage(cmdBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    EndSingleTimeCommands(cmdBuffer);
}

VkFormat VkResourceManager::FindDepthFormat() const
{
    std::vector<VkFormat> candidates = {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
                                        VK_FORMAT_D24_UNORM_S8_UINT};

    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(m_physicalDevice, format, &props);

        if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            return format;
        }
    }

    INXLOG_ERROR("Failed to find supported depth format");
    return VK_FORMAT_D32_SFLOAT;
}

bool VkResourceManager::HasStencilComponent(VkFormat format)
{
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

// ============================================================================
// Command Buffer Management
// ============================================================================

CommandBufferAllocation VkResourceManager::AllocatePrimaryCommandBuffer()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    CommandBufferAllocation allocation;
    allocation.pool = m_commandPool;

    if (vkAllocateCommandBuffers(m_device, &allocInfo, &allocation.cmdBuffer) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to allocate primary command buffer");
        return {};
    }

    return allocation;
}

CommandBufferAllocation VkResourceManager::AllocateSecondaryCommandBuffer()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
    allocInfo.commandBufferCount = 1;

    CommandBufferAllocation allocation;
    allocation.pool = m_commandPool;

    if (vkAllocateCommandBuffers(m_device, &allocInfo, &allocation.cmdBuffer) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to allocate secondary command buffer");
        return {};
    }

    return allocation;
}

void VkResourceManager::FreeCommandBuffer(const CommandBufferAllocation &allocation)
{
    if (allocation.cmdBuffer != VK_NULL_HANDLE && allocation.pool != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(m_device, allocation.pool, 1, &allocation.cmdBuffer);
    }
}

VkCommandBuffer VkResourceManager::BeginSingleTimeCommands()
{
    // ────────────────────────────────────────────────────────────────────
    // Phase 5b: pool command buffers and fences instead of churning them
    // per upload. Hot path now hits zero kernel allocations after the
    // first few warm-up uploads — see VkResourceManager.h for rationale.
    // ────────────────────────────────────────────────────────────────────
    VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        if (!m_freeSingleTimeCmdBuffers.empty()) {
            cmdBuffer = m_freeSingleTimeCmdBuffers.back();
            m_freeSingleTimeCmdBuffers.pop_back();
        }
    }

    if (cmdBuffer == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = m_commandPool;
        allocInfo.commandBufferCount = 1;
        vkAllocateCommandBuffers(m_device, &allocInfo, &cmdBuffer);

        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        m_allSingleTimeCmdBuffers.push_back(cmdBuffer);
    } else {
        // Recycled buffer carries left-over recording state — reset before reuse.
        vkResetCommandBuffer(cmdBuffer, 0);
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(cmdBuffer, &beginInfo);

    return cmdBuffer;
}

void VkResourceManager::EndSingleTimeCommands(VkCommandBuffer cmdBuffer)
{
    vkEndCommandBuffer(cmdBuffer);

    // Acquire (or lazily create) a recycled fence.
    VkFence submitFence = VK_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        if (!m_freeSingleTimeFences.empty()) {
            submitFence = m_freeSingleTimeFences.back();
            m_freeSingleTimeFences.pop_back();
        }
    }
    if (submitFence == VK_NULL_HANDLE) {
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(m_device, &fenceInfo, nullptr, &submitFence) != VK_SUCCESS) {
            INXLOG_ERROR("VkResourceManager::EndSingleTimeCommands: vkCreateFence failed");
            return;
        }
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        m_allSingleTimeFences.push_back(submitFence);
    } else {
        vkResetFences(m_device, 1, &submitFence);
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuffer;

    vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, submitFence);
    WaitForFencePumpingEvents(m_device, submitFence);

    // Return both objects to the free list — the fence is reusable after
    // vkResetFences (above) and the command buffer is reusable after the
    // GPU has signalled (which is what WaitForFencePumpingEvents proved).
    {
        std::lock_guard<std::mutex> guard(m_singleTimeMutex);
        m_freeSingleTimeFences.push_back(submitFence);
        m_freeSingleTimeCmdBuffers.push_back(cmdBuffer);
    }
}

// ============================================================================
// Descriptor Management
// ============================================================================

VkDescriptorPool VkResourceManager::CreateDescriptorPool(const std::vector<VkDescriptorPoolSize> &poolSizes,
                                                         uint32_t maxSets)
{
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = maxSets;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    VkDescriptorPool pool;
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &pool) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to create descriptor pool");
        return VK_NULL_HANDLE;
    }

    m_descriptorPools.push_back(pool);
    return pool;
}

std::vector<VkDescriptorSet>
VkResourceManager::AllocateDescriptorSets(VkDescriptorPool pool, const std::vector<VkDescriptorSetLayout> &layouts)
{
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = pool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    allocInfo.pSetLayouts = layouts.data();

    std::vector<VkDescriptorSet> sets(layouts.size());
    if (vkAllocateDescriptorSets(m_device, &allocInfo, sets.data()) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to allocate descriptor sets");
        return {};
    }

    return sets;
}

void VkResourceManager::UpdateDescriptorSet(VkDescriptorSet set, uint32_t binding, VkBuffer buffer, VkDeviceSize offset,
                                            VkDeviceSize range)
{
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer;
    bufferInfo.offset = offset;
    bufferInfo.range = range;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
}

void VkResourceManager::UpdateDescriptorSet(VkDescriptorSet set, uint32_t binding, VkImageView imageView,
                                            VkSampler sampler, VkImageLayout layout)
{
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = layout;
    imageInfo.imageView = imageView;
    imageInfo.sampler = sampler;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = 1;
    write.pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
}

void VkResourceManager::DestroyDescriptorPool(VkDescriptorPool pool)
{
    if (pool == VK_NULL_HANDLE) {
        return;
    }

    auto it = std::find(m_descriptorPools.begin(), m_descriptorPools.end(), pool);
    if (it != m_descriptorPools.end()) {
        m_descriptorPools.erase(it);
    }

    vkDestroyDescriptorPool(m_device, pool, nullptr);
}

// ============================================================================
// Sampler Management
// ============================================================================

VkSampler VkResourceManager::GetLinearSampler()
{
    if (m_linearSampler != VK_NULL_HANDLE) {
        return m_linearSampler;
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(m_physicalDevice, &properties);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(m_device, &samplerInfo, nullptr, &m_linearSampler) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to create linear sampler");
        return VK_NULL_HANDLE;
    }

    return m_linearSampler;
}

VkSampler VkResourceManager::GetNearestSampler()
{
    if (m_nearestSampler != VK_NULL_HANDLE) {
        return m_nearestSampler;
    }

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

    if (vkCreateSampler(m_device, &samplerInfo, nullptr, &m_nearestSampler) != VK_SUCCESS) {
        INXLOG_ERROR("Failed to create nearest sampler");
        return VK_NULL_HANDLE;
    }

    return m_nearestSampler;
}

VkCommandPool VkResourceManager::GetCommandPool() const
{
    return m_commandPool;
}

std::unique_ptr<VkSamplerHandle> VkResourceManager::CreateSampler(VkFilter filter, VkSamplerAddressMode addressMode)
{
    auto sampler = std::make_unique<VkSamplerHandle>();
    if (!sampler->Create(m_device, m_physicalDevice, filter, addressMode)) {
        return nullptr;
    }
    return sampler;
}

} // namespace vk
} // namespace infernux
