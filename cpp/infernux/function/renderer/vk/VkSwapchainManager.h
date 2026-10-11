/**
 * @file VkSwapchainManager.h
 * @brief Vulkan presentation management - surface swapchain and semaphores
 *
 * This class manages the swapchain lifecycle including:
 * - Initial creation with optimal settings
 * - Recreation on window resize
 * - Image-acquire and present semaphores
 * - Image acquisition and presentation
 *
 * Architecture Notes:
 * - Single responsibility: presentation surface, swapchain images, and
 *   acquire/present
 * semaphores only
 * - Queue handles, submission ordering, and frame-completion fences are owned
 *   by
 * VulkanQueueManager
 * - RAII: All Vulkan objects are automatically cleaned up on destruction
 * - Uses VkDeviceContext for device access
 *
 * Usage:
 *   VkSwapchainManager swapchain;
 *   if (!swapchain.Create(deviceContext, width, height)) { // error }
 *   // In render loop:
 *   uint32_t imageIndex;
 *   if (swapchain.AcquireNextImage(frameSlot, imageIndex) == SwapchainResult::NeedRecreate) {
 *
 * swapchain.Recreate(newWidth, newHeight);
 *   }
 *   // Render to imageIndex...
 *   swapchain.Present(queueManager, imageIndex);
 */

#pragma once

#include "VkTypes.h"
#include <function/renderer/rhi/RhiHandles.h>
#include <functional>
#include <vector>

namespace infernux
{
namespace vk
{

// Forward declarations
class VkDeviceContext;
class VulkanQueueManager;

/**
 * @brief Result codes for swapchain operations
 */
enum class SwapchainResult
{
    Success,      ///< Operation completed successfully
    NeedRecreate, ///< Swapchain is out of date and must be replaced
    SurfaceLost,  ///< The platform surface was replaced and must be rebound
    SkipFrame,    ///< No image is available yet; keep the current generation
    Error         ///< Fatal error occurred
};

/**
 * @brief Manages Vulkan swapchain presentation lifecycle
 */
class VkSwapchainManager
{
  public:
    using BeforeGenerationRetire = std::function<void()>;

    /// Device-call boundary owned by this presentation instance. Supplying it
    /// explicitly also permits controlled driver failures without submitting
    /// invalid work to Vulkan.
    struct Dispatch
    {
        PFN_vkCreateSwapchainKHR createSwapchain;
        PFN_vkGetSwapchainImagesKHR getSwapchainImages;
        PFN_vkCreateImageView createImageView;
        PFN_vkCreateSemaphore createSemaphore;
        PFN_vkAcquireNextImageKHR acquireNextImage;
        PFN_vkDestroySwapchainKHR destroySwapchain;
        PFN_vkDestroyImageView destroyImageView;
        PFN_vkDestroySemaphore destroySemaphore;
    };

    /// @brief Maximum number of frames that can be processed concurrently
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    VkSwapchainManager() = default;
    ~VkSwapchainManager();

    // Non-copyable, movable
    VkSwapchainManager(const VkSwapchainManager &) = delete;
    VkSwapchainManager &operator=(const VkSwapchainManager &) = delete;
    VkSwapchainManager(VkSwapchainManager &&other) noexcept;
    VkSwapchainManager &operator=(VkSwapchainManager &&other) noexcept;

    // ========================================================================
    // Lifecycle Management
    // ========================================================================

    /**
     * @brief Create the swapchain
     *
     * @param context Device context for Vulkan access
     * @param width Initial width (0 = use surface capabilities)
     * @param height Initial height (0 = use surface capabilities)
     * @return true if creation succeeded
     */
    bool Create(const VkDeviceContext &context, uint32_t width = 0, uint32_t height = 0);
    bool Create(const VkDeviceContext &context, uint32_t width, uint32_t height, const Dispatch &dispatch);

    /**
     * @brief Recreate the swapchain (e.g., after window resize)
     *
     * @param context Device context for Vulkan access
     * @param width New width (0 = use surface capabilities)
     * @param height New height (0 = use surface capabilities)
     * @return true if recreation succeeded
     */
    bool Recreate(const VkDeviceContext &context, VulkanQueueManager &queues, uint32_t width, uint32_t height,
                  const BeforeGenerationRetire &beforeRetire);

    /**
     * @brief Cleanup all swapchain resources
     */
    void Destroy() noexcept;

    /// @brief Skip vkDeviceWaitIdle in Destroy (device already drained at shutdown)
    void SetSkipWaitIdle(bool v)
    {
        m_skipWaitIdle = v;
    }

    /// @brief Set the preferred present mode. Returns true when it changed.
    [[nodiscard]] bool SetPreferredPresentMode(VkPresentModeKHR mode)
    {
        if (m_preferredPresentMode == mode) {
            return false;
        }
        m_preferredPresentMode = mode;
        return true;
    }

    /// @brief Get the preferred present mode.
    [[nodiscard]] VkPresentModeKHR GetPreferredPresentMode() const
    {
        return m_preferredPresentMode;
    }

    // ========================================================================
    // Frame Operations
    // ========================================================================

    /**
     * @brief Acquire the next swapchain image
     *
     * @param[out] imageIndex Index of the acquired image
     * @return SwapchainResult indicating success or need for recreation
     */
    SwapchainResult AcquireNextImage(uint32_t frameSlot, uint32_t &imageIndex);

    /**
     * @brief Present the rendered image
     *
     * @param imageIndex Index of the image to present
     * @return SwapchainResult indicating success or need for recreation
     */
    SwapchainResult Present(VulkanQueueManager &queues, uint32_t imageIndex);

    // ========================================================================
    // Accessors
    // ========================================================================

    /// @brief Whether the published generation can acquire new images.
    [[nodiscard]] bool IsValid() const
    {
        return m_generation.swapchain != VK_NULL_HANDLE && !m_generation.retired;
    }

    [[nodiscard]] rhi::DeviceId GetDeviceId() const noexcept
    {
        return m_deviceId;
    }

    /// @brief Get swapchain handle
    [[nodiscard]] VkSwapchainKHR GetSwapchain() const
    {
        return m_generation.swapchain;
    }

    /// @brief Get swapchain image format
    [[nodiscard]] VkFormat GetImageFormat() const
    {
        return m_generation.imageFormat;
    }

    /// @brief Get swapchain extent
    [[nodiscard]] VkExtent2D GetExtent() const
    {
        return m_generation.extent;
    }

    /// @brief Get number of swapchain images
    [[nodiscard]] uint32_t GetImageCount() const
    {
        return static_cast<uint32_t>(m_generation.images.size());
    }

    /// @brief Get swapchain images
    [[nodiscard]] const std::vector<VkImage> &GetImages() const
    {
        return m_generation.images;
    }

    /// @brief Get swapchain image views
    [[nodiscard]] const std::vector<VkImageView> &GetImageViews() const
    {
        return m_generation.imageViews;
    }

    /// @brief Get image at index
    [[nodiscard]] VkImage GetImage(size_t index) const
    {
        return index < m_generation.images.size() ? m_generation.images[index] : VK_NULL_HANDLE;
    }

    /// @brief Get image view at index
    [[nodiscard]] VkImageView GetImageView(size_t index) const;

    [[nodiscard]] bool SupportsTransferSource() const noexcept
    {
        return (m_generation.imageUsage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;
    }

    /// @brief Get the image-available semaphore for a renderer-owned frame slot.
    [[nodiscard]] VkSemaphore GetImageAvailableSemaphore(uint32_t frameSlot) const;

    /// @brief Get render finished semaphore for a specific swapchain image
    [[nodiscard]] VkSemaphore GetRenderFinishedSemaphore(uint32_t imageIndex) const;

  private:
    struct SwapchainGeneration
    {
        VkSwapchainKHR swapchain = VK_NULL_HANDLE;
        std::vector<VkImage> images;
        std::vector<VkImageView> imageViews;
        std::vector<VkSemaphore> renderFinishedSemaphores;
        VkFormat imageFormat = VK_FORMAT_UNDEFINED;
        VkExtent2D extent{};
        VkImageUsageFlags imageUsage = 0;
        bool retired = false;
    };

    // ========================================================================
    // Internal Methods
    // ========================================================================

    /// @brief Choose the best surface format
    VkSurfaceFormatKHR ChooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &formats) const;

    /// @brief Choose the best present mode
    VkPresentModeKHR ChoosePresentMode(const std::vector<VkPresentModeKHR> &modes) const;

    /// @brief Choose the optimal extent
    VkExtent2D ChooseExtent(const VkSurfaceCapabilitiesKHR &capabilities, uint32_t requestedWidth,
                            uint32_t requestedHeight) const;

    /// @brief Create image views for swapchain images
    bool CreateImageViews(SwapchainGeneration &generation);

    /// @brief Create frame synchronization objects
    bool CreateSyncObjects();

    /// @brief Create per-image render-finished semaphores
    bool CreateRenderFinishedSemaphores(SwapchainGeneration &generation);

    /// @brief Destroy per-image render-finished semaphores
    void DestroyGeneration(SwapchainGeneration &generation) noexcept;

    /// @brief Build an unpublished generation. Passing the published handle to
    /// vkCreateSwapchainKHR retires it, even when that call or a later step fails.
    bool BuildGeneration(const VkDeviceContext &context, uint32_t width, uint32_t height, VkSwapchainKHR oldSwapchain,
                         SwapchainGeneration &generation);

  private:
    // ========================================================================
    // Vulkan Objects
    // ========================================================================

    bool m_skipWaitIdle = false;
    rhi::DeviceId m_deviceId = rhi::InvalidDeviceId;
#if defined(__ANDROID__)
    VkPresentModeKHR m_preferredPresentMode = VK_PRESENT_MODE_FIFO_KHR;
#else
    VkPresentModeKHR m_preferredPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
#endif
    VkDevice m_device = VK_NULL_HANDLE;
    Dispatch m_dispatch{};
    SwapchainGeneration m_generation;

    // Presentation synchronization. GPU completion fences are owned by
    // VulkanQueueManager and are intentionally absent from this class.
    std::vector<VkSemaphore> m_imageAvailableSemaphores;
};

} // namespace vk
} // namespace infernux
