#include "VulkanBackendContext.h"
#include "VulkanRhiDevice.h"
#include <core/log/InxLog.h>
#include <limits>

namespace infernux::vk
{
namespace
{
class ComputeLease final : public rhi::ComputeHost
{
  public:
    ComputeLease(rhi::Device &device, rhi::ComputeQueue &queue, std::shared_ptr<rhi::ComputeHostServices> services)
        : ComputeHost(device, queue, services), m_services(std::move(services))
    {
        ++m_services->leases;
    }
    ~ComputeLease() override
    {
        ReleaseLease();
    }
    void ReleaseLease() noexcept override
    {
        if (m_counted) {
            --m_services->leases;
            m_counted = false;
        }
    }

  private:
    std::shared_ptr<rhi::ComputeHostServices> m_services;
    bool m_counted = true;
};
} // namespace

VulkanBackendContext::~VulkanBackendContext()
{
    Destroy();
}

bool VulkanBackendContext::InitializeInstance(const DeviceConfig &config)
{
    m_config = config;
    return m_devices.InitializePrimaryInstance(m_config);
}

bool VulkanBackendContext::InitializeDevice(VkSurfaceKHR surface, uint32_t graphicsFrameSlots)
{
    if (!m_devices.InitializePrimaryDevice(surface, m_config))
        return false;
    auto &device = Device().GetRhiDevice();
    const auto contract = rhi::CheckDeviceContract(device.GetApiVersion(), device.GetCapabilities());
    if (!contract.IsValid()) {
        INXLOG_ERROR("RHI device contract rejected: ", contract.Message());
        return false;
    }
    if (!m_queues.Initialize(Device(), graphicsFrameSlots) || !m_compute.Initialize(Device(), m_queues))
        return false;
    device.UseSubmissionSerials([this] { return m_queues.GetLastReservedCompletionEpoch(); });
    m_services->collect = [this] { CollectComputeResources(); };
    m_services->framePressure = [this] { return m_queues.GetFramePressure(); };
    m_services->alive = true;
    return true;
}

bool VulkanBackendContext::IsInitialized() const noexcept
{
    return m_services->alive;
}

std::unique_ptr<rhi::ComputeHost> VulkanBackendContext::AcquireComputeHost()
{
    if (!IsInitialized())
        throw std::runtime_error("GPU compute requires initialized engine device services");
    return std::make_unique<ComputeLease>(Device().GetRhiDevice(), m_compute, m_services);
}

void VulkanBackendContext::CollectComputeResources()
{
    m_compute.Collect();
    const auto completed = m_queues.GetCompletedCompletionEpoch();
    auto &device = Device().GetRhiDevice();
    (void)device.CollectDescriptorRetirements(completed);
    (void)device.CollectResourceRetirements(completed);
}

void VulkanBackendContext::BeginShutdown() noexcept
{
    m_services->alive = false;
    m_services->collect = {};
    m_services->framePressure = {};
    Device().WaitIdleForShutdown();
    Device().SetShuttingDown(true);
    m_presentation.SetSkipWaitIdle(true);
}

void VulkanBackendContext::ReleasePresentation() noexcept
{
    // The caller has drained graphics and compute before destroying presentation
    // resources. Device timelines survive. Accepted packet owners stay in their queue
    // slots until normal collection or final compute-queue teardown.
    m_presentation.Destroy();
    Device().DestroySurface();
}

void VulkanBackendContext::Destroy() noexcept
{
    BeginShutdown();
    ReleasePresentation();
    m_compute.Destroy();
    if (Device().HasRhiDevice()) {
        auto &device = Device().GetRhiDevice();
        const auto completed = (std::numeric_limits<rhi::SubmissionSerial>::max)();
        (void)device.CollectDescriptorRetirements(completed);
        (void)device.CollectResourceRetirements(completed);
    }
    m_queues.Destroy();
    m_devices.Destroy();
}
} // namespace infernux::vk
