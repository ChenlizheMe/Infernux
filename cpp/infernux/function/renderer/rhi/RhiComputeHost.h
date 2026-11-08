#pragma once

#include "RhiComputeQueue.h"
#include "RhiDevice.h"
#include <functional>
#include <stdexcept>

namespace infernux::rhi
{
/// Owner-thread service lifetime, shared by leases and resident resource views.
/// It ends before the queue is destroyed; DeviceLifetime separately guards frees.
struct ComputeHostServices
{
    bool alive = false;
    size_t leases = 0;
    std::function<void()> collect;
    std::function<GpuQueuePressure()> framePressure;
};

/// Borrowed engine services. Resources copy this view, not an acquired lease.
class ComputeHost
{
  public:
    ComputeHost(Device &device, ComputeQueue &queue, std::shared_ptr<ComputeHostServices> services = {})
        : device(device), queue(queue), m_lifetime(device.GetLifetime()), m_deviceId(device.GetDeviceId()),
          m_services(std::move(services))
    {
    }
    virtual ~ComputeHost() = default;
    // Copy only the borrowed service view, including its original device
    // identity. A native resource does not copy an acquired wrapper's lease.
    ComputeHost(const ComputeHost &) = default;
    ComputeHost &operator=(const ComputeHost &) = delete;

    // Engine-acquired host wrappers override this to surrender their teardown
    // lease explicitly. Native buffers and kernels store ordinary borrowed
    // service views, so releasing the wrapper does not invalidate resident
    // resources that are already being retired by the compute queue.
    virtual void ReleaseLease() noexcept
    {
    }

    [[nodiscard]] bool SharesServicesWith(const ComputeHost &other) const noexcept
    {
        return IsAlive() && other.IsAlive() && m_lifetime == other.m_lifetime && m_deviceId == other.m_deviceId &&
               &device == &other.device && &queue == &other.queue;
    }

    [[nodiscard]] bool IsAlive() const noexcept
    {
        return (!m_services || m_services->alive) && (!m_lifetime || m_lifetime->alive.load(std::memory_order_acquire));
    }
    void RequireAlive() const
    {
        if (!IsAlive())
            throw std::runtime_error("Compute host has retired");
    }
    [[nodiscard]] ComputeQueue &Queue() const
    {
        RequireAlive();
        return queue;
    }
    void Collect() const
    {
        RequireAlive();
        if (m_services && m_services->collect)
            m_services->collect();
        else
            queue.Collect();
    }
    /// Non-waiting collection followed by actual pool/fence observation. This
    /// accepts no work and does not consume or retire a graphics frame slot.
    [[nodiscard]] GpuQueuePressure GetQueuePressure() const
    {
        RequireAlive();
        queue.Collect();
        auto pressure = m_services && m_services->framePressure ? m_services->framePressure() : GpuQueuePressure{};
        pressure.computeSubmissions = queue.GetPendingSubmissionCount();
        return pressure;
    }
    [[nodiscard]] DeviceId GetDeviceId() const noexcept
    {
        return m_deviceId;
    }
    [[nodiscard]] const std::shared_ptr<DeviceLifetime> &GetLifetime() const noexcept
    {
        return m_lifetime;
    }

    Device &device;
    ComputeQueue &queue;

  private:
    std::shared_ptr<DeviceLifetime> m_lifetime;
    DeviceId m_deviceId;
    std::shared_ptr<ComputeHostServices> m_services;
};
} // namespace infernux::rhi
