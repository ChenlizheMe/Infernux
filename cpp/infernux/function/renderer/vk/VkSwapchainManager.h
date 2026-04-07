/**
 * @file VkSwapchainManager.h
 * @brief Vulkan swapchain management - creation, recreation, frame synchronization
 *
 * This class manages the swapchain lifecycle including:
 * - Initial creation with optimal settings
 * - Recreation on window resize
 * - Frame synchronization objects (semaphores, fences)
 * - Image acquisition and presentation
 *
 * Architecture Notes:
 * - Single responsibility: Swapchain and frame management only
 * - RAII: All Vulkan objects are automatically cleaned up on destruction
 * - Uses VkDeviceContext for device access
 *
 * Usage:
 *   VkSwapchainManager swapchain;
 *   if (!swapchain.Create(deviceContext, width, height)) { // error }
 *   // In render loop:
 *   uint32_t imageIndex;
 *   if (swapchain.AcquireNextImage(imageIndex) == SwapchainResult::NeedRecreate) {
 *       swapchain.Recreate(newWidth, newHeight);
 *   }
 *   // Render to imageIndex...
 *   swapchain.Present(imageIndex);
 */

#pragma once

#include "VkTypes.h"
#include <vector>

namespace infernux
{
namespace vk
{

// Forward declarations
class VkDeviceContext;

/**
 * @brief Result codes for swapchain operations
 */
enum class SwapchainResult
{
    Success,      ///< Operation completed successfully
    NeedRecreate, ///< Swapchain needs recreation (resize, suboptimal)
    Error         ///< Fatal error occurred
};

/**
 * @brief Frame synchronization objects
 *
 * Each frame in flight has its own set of sync objects to allow
 * CPU-GPU parallelism.
 */
struct FrameSyncObjects
{
    VkSemaphore imageAvailableSemaphore = VK_NULL_HANDLE;
    VkFence inFlightFence = VK_NULL_HANDLE;
};

/**
 * @brief Manages Vulkan swapchain lifecycle and frame synchronization
 */
class VkSwapchainManager
{
  public:
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

    /**
     * @brief Recreate the swapchain (e.g., after window resize)
     *
     * @param context Device context for Vulkan access
     * @param width New width (0 = use surface capabilities)
     * @param height New height (0 = use surface capabilities)
     * @return true if recreation succeeded
     */
    bool Recreate(const VkDeviceContext &context, uint32_t width, uint32_t height);

    /**
     * @brief Cleanup all swapchain resources
     */
    void Destroy() noexcept;

    /// @brief Skip vkDeviceWaitIdle in Destroy (device already drained at shutdown)
    void SetSkipWaitIdle(bool v)
    {
        m_skipWaitIdle = v;
    }

    /// @brief Set the preferred present mode.  Takes effect on next Recreate().
    void SetPreferredPresentMode(VkPresentModeKHR mode)
    {
        m_preferredPresentMode = mode;
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
    SwapchainResult AcquireNextImage(uint32_t &imageIndex);
    void ResetCurrentFence();

    /**
     * @brief Mark that the current frame fence has already been waited on.
     *
     * When set, the next AcquireNextImage() call skips the internal
     * WaitForFrame().  The flag is automatically cleared after acquire.
     * This allows the caller to wait for the fence earlier in the frame
     * (e.g., before GUI processing) to enable CPU/GPU overlap.
     */
    void MarkFenceAlreadyWaited()
    {
        m_fenceAlreadyWaited = true;
    }

    /**
     * @brief Present the rendered image
     *
     * @param imageIndex Index of the image to present
     * @return SwapchainResult indicating success or need for recreation
     */
    SwapchainResult Present(uint32_t imageIndex);

    /**
     * @brief Wait for the current frame's fence
     */
    void WaitForFrame();

    /**
     * @brief Advance to the next frame
     */
    void AdvanceFrame();

    // ========================================================================
    // Accessors
    // ========================================================================

    /// @brief Check if swapchain is valid
    [[nodiscard]] bool IsValid() const
    {
        return m_swapchain != VK_NULL_HANDLE;
    }

    /// @brief Get swapchain handle
    [[nodiscard]] VkSwapchainKHR GetSwapchain() const
    {
        return m_swapchain;
    }

    /// @brief Get swapchain image format
    [[nodiscard]] VkFormat GetImageFormat() const
    {
        return m_imageFormat;
    }

    /// @brief Get swapchain extent
    [[nodiscard]] VkExtent2D GetExtent() const
    {
        return m_extent;
    }

    /// @brief Get number of swapchain images
    [[nodiscard]] uint32_t GetImageCount() const
    {
        return static_cast<uint32_t>(m_images.size());
    }

    /// @brief Get swapchain images
    [[nodiscard]] const std::vector<VkImage> &GetImages() const
    {
        return m_images;
    }

    /// @brief Get swapchain image views
    [[nodiscard]] const std::vector<VkImageView> &GetImageViews() const
    {
        return m_imageViews;
    }

    /// @brief Get image at index
    [[nodiscard]] VkImage GetImage(size_t index) const
    {
        return index < m_images.size() ? m_images[index] : VK_NULL_HANDLE;
    }

    /// @brief Get image view at index
    [[nodiscard]] VkImageView GetImageView(size_t index) const;

    /// @brief Get current frame index (for double/triple buffering)
    [[nodiscard]] uint32_t GetCurrentFrame() const
    {
        return m_currentFrame;
    }

    /// @brief Get sync objects for current frame
    [[nodiscard]] const FrameSyncObjects &GetCurrentSyncObjects() const;

    /// @brief Get image available semaphore for current frame
    [[nodiscard]] VkSemaphore GetImageAvailableSemaphore() const;

    /// @brief Get render finished semaphore for a specific swapchain image
    [[nodiscard]] VkSemaphore GetRenderFinishedSemaphore(uint32_t imageIndex) const;

    /// @brief Get in-flight fence for current frame
    [[nodiscard]] VkFence GetInFlightFence() const;

  private:
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
    bool CreateImageViews();

    /// @brief Create frame synchronization objects
    bool CreateSyncObjects();

    /// @brief Create per-image render-finished semaphores
    bool CreateRenderFinishedSemaphores();

    /// @brief Destroy per-image render-finished semaphores
    void DestroyRenderFinishedSemaphores() noexcept;

    /// @brief Cleanup swapchain (but not sync objects)
    void CleanupSwapchain();

    /// @brief Shared implementation for Create/Recreate — populates m_swapchain,
    /// m_images, m_imageViews from the given context and dimensions.
    bool CreateSwapchainCore(const VkDeviceContext &context, uint32_t width, uint32_t height,
                             VkSwapchainKHR oldSwapchain);

  private:
    // ========================================================================
    // Vulkan Objects
    // ========================================================================

    bool m_skipWaitIdle = false;
    bool m_fenceAlreadyWaited = false;
    VkPresentModeKHR m_preferredPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    VkDevice m_device = VK_NULL_HANDLE;
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    VkQueue m_presentQueue = VK_NULL_HANDLE;

    // Swapchain images and views (images owned by swapchain, views owned by us)
    std::vector<VkImage> m_images;
    std::vector<VkImageView> m_imageViews;

    // Swapchain properties
    VkFormat m_imageFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D m_extent{};

    // Frame synchronization
    std::vector<FrameSyncObjects> m_syncObjects;
    std::vector<VkSemaphore> m_renderFinishedSemaphores;
    uint32_t m_currentFrame = 0;
};

} // namespace vk
} // namespace infernux
