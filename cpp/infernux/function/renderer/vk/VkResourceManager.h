/**
 * @file VkResourceManager.h
 * @brief Vulkan resource management - buffers, images, textures, and command pools
 *
 * This class manages GPU resource allocation and provides:
 * - Staging buffer pooling for efficient uploads
 * - Texture loading and caching
 * - Depth buffer management
 * - Command pool and command buffer management
 * - Descriptor pool and set management
 *
 * Architecture Notes:
 * - Resources are reference counted or managed via unique ownership
 * - Staging uploads use double buffering for async transfers (future)
 * - Designed for easy integration with RenderGraph
 *
 * Usage:
 *   VkResourceManager resources;
 *   resources.Initialize(deviceContext);
 *
 *   // Create a vertex buffer
 *   auto vertexBuffer = resources.CreateVertexBuffer(vertices.data(), vertices.size() * sizeof(Vertex));
 *
 *   // Load a texture
 *   auto texture = resources.LoadTexture("textures/diffuse.png");
 */

#pragma once

#include "VkHandle.h"
#include "VkTypes.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace infernux
{
namespace vk
{

// Forward declarations
class VkDeviceContext;

/**
 * @brief Descriptor binding information
 */
struct DescriptorBindingInfo
{
    uint32_t binding = 0;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    VkShaderStageFlags stageFlags = VK_SHADER_STAGE_ALL;
    uint32_t count = 1;
};

/**
 * @brief Command buffer allocation result
 */
struct CommandBufferAllocation
{
    VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
};

/**
 * @brief Manages Vulkan resources - buffers, images, descriptors, etc.
 */
class VkResourceManager
{
  public:
    VkResourceManager() = default;
    ~VkResourceManager();

    // Non-copyable, movable
    VkResourceManager(const VkResourceManager &) = delete;
    VkResourceManager &operator=(const VkResourceManager &) = delete;
    VkResourceManager(VkResourceManager &&other) noexcept;
    VkResourceManager &operator=(VkResourceManager &&other) noexcept;

    // ========================================================================
    // Initialization
    // ========================================================================

    /**
     * @brief Initialize the resource manager
     *
     * @param context Device context for Vulkan access
     * @return true if initialization succeeded
     */
    bool Initialize(const VkDeviceContext &context);

    /**
     * @brief Cleanup all managed resources
     */
    void Destroy() noexcept;

    /// @brief Check if initialized
    [[nodiscard]] bool IsValid() const
    {
        return m_device != VK_NULL_HANDLE;
    }

    /// @brief Get the internal command pool
    [[nodiscard]] VkCommandPool GetCommandPool() const;

    // ========================================================================
    // Buffer Management
    // ========================================================================

    /**
     * @brief Create a vertex buffer with data
     *
     * @param data Pointer to vertex data
     * @param size Size in bytes
     * @return Unique pointer to buffer handle
     */
    [[nodiscard]] std::unique_ptr<VkBufferHandle> CreateVertexBuffer(const void *data, VkDeviceSize size);

    /**
     * @brief Create an index buffer with data
     *
     * @param data Pointer to index data
     * @param size Size in bytes
     * @return Unique pointer to buffer handle
     */
    [[nodiscard]] std::unique_ptr<VkBufferHandle> CreateIndexBuffer(const void *data, VkDeviceSize size);

    /**
     * @brief Create a uniform buffer
     *
     * @param size Size in bytes
     * @return Unique pointer to buffer handle (host-visible for frequent updates)
     */
    [[nodiscard]] std::unique_ptr<VkBufferHandle> CreateUniformBuffer(VkDeviceSize size);

    /**
     * @brief Create a storage buffer
     *
     * @param size Size in bytes
     * @param deviceLocal If true, create device-local buffer
     * @return Unique pointer to buffer handle
     */
    [[nodiscard]] std::unique_ptr<VkBufferHandle> CreateStorageBuffer(VkDeviceSize size, bool deviceLocal = true);

    /**
     * @brief Create a staging buffer for data uploads
     *
     * @param size Size in bytes
     * @return Unique pointer to buffer handle (host-visible)
     */
    [[nodiscard]] std::unique_ptr<VkBufferHandle> CreateStagingBuffer(VkDeviceSize size);

    /**
     * @brief Copy buffer data using a one-shot command buffer
     *
     * @param srcBuffer Source buffer
     * @param dstBuffer Destination buffer
     * @param size Size to copy (0 = entire buffer)
     */
    void CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

    // ========================================================================
    // Image and Texture Management
    // ========================================================================

    /**
     * @brief Create an image handle
     *
     * @param width Image width
     * @param height Image height
     * @param format Image format
     * @param usage Usage flags
     * @param properties Memory properties
     * @return Unique pointer to image handle
     */
    [[nodiscard]] std::unique_ptr<VkImageHandle>
    CreateImage(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage,
                VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    /**
     * @brief Create a depth buffer
     *
     * @param width Width
     * @param height Height
     * @param format Depth format (VK_FORMAT_UNDEFINED = auto-select)
     * @return Unique pointer to image handle with view
     */
    [[nodiscard]] std::unique_ptr<VkImageHandle> CreateDepthBuffer(uint32_t width, uint32_t height,
                                                                   VkFormat format = VK_FORMAT_UNDEFINED);

    /**
     * @brief Load a texture from file
     *
     * @param filePath Path to image file
     * @param generateMipmaps Whether to generate mipmaps
     * @param format GPU texture format (SRGB for color, UNORM for linear data)
     * @param maxSize Optional max dimension clamp (0 = no clamp)
     * @param normalMapMode True when the texture is an authored tangent-space normal map.
     *        This does not regenerate normals from height; it preserves the source pixels
     *        and only lets higher-level code select linear sampling / normal-map handling.
     * @return Unique pointer to texture handle
     */
    [[nodiscard]] std::unique_ptr<VkTexture> LoadTexture(const std::string &filePath, bool generateMipmaps = true,
                                                         VkFormat format = VK_FORMAT_R8G8B8A8_SRGB, int maxSize = 0,
                                                         bool normalMapMode = false,
                                                         VkFilter filter = VK_FILTER_LINEAR,
                                                         VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                                         int aniso = -1);

    /**
     * @brief Create a texture from raw pixel data
     *
     * @param pixels Pixel data (RGBA)
     * @param width Width
     * @param height Height
     * @param format Format
     * @return Unique pointer to texture handle
     */
    [[nodiscard]] std::unique_ptr<VkTexture> CreateTextureFromPixels(const unsigned char *pixels, uint32_t width,
                                                                     uint32_t height,
                                                                     VkFormat format = VK_FORMAT_R8G8B8A8_SRGB,
                                                                     bool generateMipmaps = false,
                                                                     VkFilter filter = VK_FILTER_LINEAR,
                                                                     VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                                                     int aniso = -1);

    /**
     * @brief Create a solid color texture
     *
     * @param width Width
     * @param height Height
     * @param r Red component (0-255)
     * @param g Green component (0-255)
     * @param b Blue component (0-255)
     * @param a Alpha component (0-255)
     * @return Unique pointer to texture handle
     */
    [[nodiscard]] std::unique_ptr<VkTexture> CreateSolidColorTexture(uint32_t width, uint32_t height, uint8_t r,
                                                                     uint8_t g, uint8_t b, uint8_t a = 255,
                                                                     VkFormat format = VK_FORMAT_R8G8B8A8_SRGB);

    /**
     * @brief Transition image layout
     *
     * @param image Image to transition
     * @param format Image format
     * @param oldLayout Current layout
     * @param newLayout Target layout
     */
    void TransitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);

    /**
     * @brief Copy buffer to image
     *
     * @param buffer Source buffer
     * @param image Destination image
     * @param width Image width
     * @param height Image height
     */
    void CopyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

    // ========================================================================
    // Command Buffer Management
    // ========================================================================

    /**
     * @brief Allocate a primary command buffer
     * @return Allocated command buffer
     */
    [[nodiscard]] CommandBufferAllocation AllocatePrimaryCommandBuffer();

    /**
     * @brief Allocate a secondary command buffer
     * @return Allocated command buffer
     */
    [[nodiscard]] CommandBufferAllocation AllocateSecondaryCommandBuffer();

    /**
     * @brief Free a command buffer
     * @param allocation Command buffer allocation to free
     */
    void FreeCommandBuffer(const CommandBufferAllocation &allocation);

    /**
     * @brief Begin a one-shot command buffer for immediate submission
     * @return Command buffer ready for recording
     */
    [[nodiscard]] VkCommandBuffer BeginSingleTimeCommands();

    /**
     * @brief End and submit a one-shot command buffer
     * @param cmdBuffer Command buffer to submit
     */
    void EndSingleTimeCommands(VkCommandBuffer cmdBuffer);

    // ========================================================================
    // Descriptor Management
    // ========================================================================

    /**
     * @brief Create a descriptor pool
     *
     * @param poolSizes Pool sizes for each descriptor type
     * @param maxSets Maximum number of sets
     * @return Descriptor pool handle
     */
    [[nodiscard]] VkDescriptorPool CreateDescriptorPool(const std::vector<VkDescriptorPoolSize> &poolSizes,
                                                        uint32_t maxSets);

    /**
     * @brief Allocate descriptor sets
     *
     * @param pool Descriptor pool
     * @param layouts Layouts to allocate
     * @return Vector of allocated descriptor sets
     */
    [[nodiscard]] std::vector<VkDescriptorSet>
    AllocateDescriptorSets(VkDescriptorPool pool, const std::vector<VkDescriptorSetLayout> &layouts);

    /**
     * @brief Update a descriptor set with a uniform buffer
     *
     * @param set Descriptor set
     * @param binding Binding index
     * @param buffer Buffer handle
     * @param offset Offset in buffer
     * @param range Range in buffer (VK_WHOLE_SIZE for entire buffer)
     */
    void UpdateDescriptorSet(VkDescriptorSet set, uint32_t binding, VkBuffer buffer, VkDeviceSize offset = 0,
                             VkDeviceSize range = VK_WHOLE_SIZE);

    /**
     * @brief Update a descriptor set with a texture
     *
     * @param set Descriptor set
     * @param binding Binding index
     * @param imageView Image view
     * @param sampler Sampler
     * @param layout Image layout
     */
    void UpdateDescriptorSet(VkDescriptorSet set, uint32_t binding, VkImageView imageView, VkSampler sampler,
                             VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    /**
     * @brief Destroy a descriptor pool
     */
    void DestroyDescriptorPool(VkDescriptorPool pool);

    // ========================================================================
    // Sampler Management
    // ========================================================================

    /**
     * @brief Get or create a standard linear sampler
     */
    [[nodiscard]] VkSampler GetLinearSampler();

    /**
     * @brief Get or create a standard nearest sampler
     */
    [[nodiscard]] VkSampler GetNearestSampler();

    /**
     * @brief Create a custom sampler
     */
    [[nodiscard]] std::unique_ptr<VkSamplerHandle>
    CreateSampler(VkFilter filter = VK_FILTER_LINEAR,
                  VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT);

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] VkDevice GetDevice() const
    {
        return m_device;
    }
    [[nodiscard]] VkPhysicalDevice GetPhysicalDevice() const
    {
        return m_physicalDevice;
    }
    [[nodiscard]] VkQueue GetGraphicsQueue() const
    {
        return m_graphicsQueue;
    }

  private:
    // ========================================================================
    // Internal Methods
    // ========================================================================

    /// @brief Create a buffer with given usage and properties
    std::unique_ptr<VkBufferHandle> CreateBufferInternal(VkDeviceSize size, VkBufferUsageFlags usage,
                                                         VkMemoryPropertyFlags properties);

    /// @brief Get best depth format for the device
    VkFormat FindDepthFormat() const;

    /// @brief Check if format has stencil component
    static bool HasStencilComponent(VkFormat format);

  public:
    void SetSkipWaitIdle(bool v)
    {
        m_skipWaitIdle = v;
    }

  private:
    bool m_skipWaitIdle = false;
    VmaAllocator m_vmaAllocator = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;

    // Cached samplers
    VkSampler m_linearSampler = VK_NULL_HANDLE;
    VkSampler m_nearestSampler = VK_NULL_HANDLE;

    // Tracked descriptor pools for cleanup
    std::vector<VkDescriptorPool> m_descriptorPools;
};

} // namespace vk
} // namespace infernux
