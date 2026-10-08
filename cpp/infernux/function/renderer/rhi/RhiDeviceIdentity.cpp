#include "RenderViewContext.h"
#include "RhiHandles.h"

#include <atomic>
#include <stdexcept>

namespace infernux::rhi
{

DeviceId AllocateDeviceId()
{
    static std::atomic<uint32_t> next{1};
    auto candidate = next.load(std::memory_order_relaxed);
    while (candidate <= std::numeric_limits<DeviceId>::max()) {
        if (next.compare_exchange_weak(candidate, candidate + 1, std::memory_order_relaxed))
            return static_cast<DeviceId>(candidate);
    }
    throw std::overflow_error("RHI device identity namespace exhausted; identities cannot be recycled");
}

RenderViewId AllocateRenderViewId() noexcept
{
    static std::atomic<RenderViewId> next{1};
    while (true) {
        const RenderViewId candidate = next.fetch_add(1, std::memory_order_relaxed);
        if (candidate != InvalidRenderViewId)
            return candidate;
    }
}

} // namespace infernux::rhi
