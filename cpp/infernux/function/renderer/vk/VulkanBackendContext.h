#pragma once

#include "VulkanBackendApi.h"

#include "VkSwapchainManager.h"
#include "VulkanComputeQueue.h"
#include "VulkanDeviceManager.h"
#include "VulkanQueueManager.h"
#include <function/renderer/rhi/RhiComputeHost.h>

namespace infernux::vk
{

/// Engine-owned device services. Rendering borrows this same device/compute
/// queue; window presentation is optional and closes before the platform window.
class VulkanBackendContext
{
  public:
    VulkanBackendContext() = default;
    INFERNUX_VULKAN_BACKEND_API ~VulkanBackendContext();

    VulkanBackendContext(const VulkanBackendContext &) = delete;
    VulkanBackendContext &operator=(const VulkanBackendContext &) = delete;
    VulkanBackendContext(VulkanBackendContext &&) = delete;
    VulkanBackendContext &operator=(VulkanBackendContext &&) = delete;

    INFERNUX_VULKAN_BACKEND_API bool InitializeInstance(const DeviceConfig &config);
    INFERNUX_VULKAN_BACKEND_API bool InitializeDevice(VkSurfaceKHR surface, uint32_t graphicsFrameSlots);
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API bool IsInitialized() const noexcept;
    [[nodiscard]] INFERNUX_VULKAN_BACKEND_API std::unique_ptr<rhi::ComputeHost> AcquireComputeHost();
    [[nodiscard]] size_t GetComputeHostLeaseCount() const noexcept
    {
        return m_services->leases;
    }
    [[nodiscard]] VulkanComputeQueue &Compute() noexcept
    {
        return m_compute;
    }
    INFERNUX_VULKAN_BACKEND_API void CollectComputeResources();
    INFERNUX_VULKAN_BACKEND_API void BeginShutdown() noexcept;
    /// Device queue timelines outlive presentation and its command pools.
    INFERNUX_VULKAN_BACKEND_API void ReleasePresentation() noexcept;
    void Destroy() noexcept;

    [[nodiscard]] VkDeviceContext &Device() noexcept
    {
        return m_devices.Primary();
    }
    [[nodiscard]] const VkDeviceContext &Device() const noexcept
    {
        return m_devices.Primary();
    }
    [[nodiscard]] VulkanDeviceManager &Devices() noexcept
    {
        return m_devices;
    }
    [[nodiscard]] const VulkanDeviceManager &Devices() const noexcept
    {
        return m_devices;
    }
    [[nodiscard]] VkSwapchainManager &Presentation() noexcept
    {
        return m_presentation;
    }
    [[nodiscard]] VulkanQueueManager &Queues() noexcept
    {
        return m_queues;
    }
    [[nodiscard]] const VulkanQueueManager &Queues() const noexcept
    {
        return m_queues;
    }
    [[nodiscard]] const VkSwapchainManager &Presentation() const noexcept
    {
        return m_presentation;
    }

    [[nodiscard]] bool OwnsConsistentPresentation() const noexcept
    {
        const auto &device = m_devices.Primary();
        return !m_presentation.IsValid() || (device.IsValid() && m_presentation.GetDeviceId() == device.GetDeviceId());
    }

  private:
    VulkanDeviceManager m_devices;
    VulkanQueueManager m_queues;
    VkSwapchainManager m_presentation;
    VulkanComputeQueue m_compute;
    DeviceConfig m_config;
    std::shared_ptr<rhi::ComputeHostServices> m_services = std::make_shared<rhi::ComputeHostServices>();
};

} // namespace infernux::vk
