#pragma once

#include "RhiComputeQueue.h"
#include "RhiDevice.h"

namespace infernux::rhi
{
/// Borrowed engine services. Python acquires a counted lease derived from this
/// view; native resources store their own view, never a pointer to that wrapper.
/// The renderer releases retained resources before destroying these services.
class ComputeHost
{
  public:
    ComputeHost(Device &device, ComputeQueue &queue) : device(device), queue(queue)
    {
    }
    virtual ~ComputeHost() = default;
    ComputeHost(const ComputeHost &) = delete;
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
        return &device == &other.device && &queue == &other.queue;
    }

    Device &device;
    ComputeQueue &queue;
};
} // namespace infernux::rhi
