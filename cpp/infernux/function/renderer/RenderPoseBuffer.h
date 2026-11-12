#pragma once

#include <function/renderer/rhi/RenderCoreApi.h>
#include <function/renderer/rhi/RhiComputeBuffer.h>

#include <atomic>
#include <cstdint>
#include <memory>

namespace infernux
{

/// One accepted device value. Models are 64-byte column-major matrices;
/// bounds are 32-byte world-space min/max float4 pairs of the rendered mesh.
/// Frozen buffers retain the actual allocations, including their producer
/// tickets, independently of subsequent writes to the logical endpoints.
struct RenderPoseValue final
{
    std::shared_ptr<rhi::ComputeBuffer> models;
    std::shared_ptr<rhi::ComputeBuffer> bounds;
    std::shared_ptr<const void> sourceIdentity;
    uint64_t generation = 0;
    uint32_t count = 0;
    uint32_t firstModel = 0;
    uint32_t firstBounds = 0;
};

/// A scene binding selects an element without tying draw order to source order.
/// Continuity changes on a rebind or teleport, not on ordinary value publication.
struct RenderPoseBinding
{
    std::shared_ptr<const RenderPoseValue> value;
    uint32_t element = 0;
    uint64_t continuity = 0;

    [[nodiscard]] bool HasValue() const noexcept
    {
        return static_cast<bool>(value);
    }

    [[nodiscard]] bool SharesHistoryWith(const RenderPoseBinding &other) const noexcept
    {
        if (!value || !other.value)
            return !value && !other.value;
        return value->sourceIdentity == other.value->sourceIdentity && element == other.element &&
               continuity == other.continuity;
    }
};

/// Shared publication endpoint for physics, neural animation and GPU scene work.
/// The producer publishes a complete model/bounds pair; extraction captures one
/// immutable value, while each view owns its own accepted motion history.
class RenderPoseBuffer final
{
  public:
    INFERNUX_RENDER_CORE_API RenderPoseBuffer(const std::shared_ptr<rhi::ComputeBuffer> &models,
                                              const std::shared_ptr<rhi::ComputeBuffer> &bounds, uint32_t count,
                                              uint32_t firstModel = 0, uint32_t firstBounds = 0);

    INFERNUX_RENDER_CORE_API void Publish(const std::shared_ptr<rhi::ComputeBuffer> &models,
                                          const std::shared_ptr<rhi::ComputeBuffer> &bounds, uint32_t firstModel = 0,
                                          uint32_t firstBounds = 0);

    [[nodiscard]] std::shared_ptr<const RenderPoseValue> Acquire() const noexcept
    {
        return std::atomic_load_explicit(&m_value, std::memory_order_acquire);
    }

    [[nodiscard]] uint32_t GetCount() const noexcept
    {
        return m_count;
    }

  private:
    const std::shared_ptr<const void> m_identity = std::make_shared<const uint8_t>(0);
    const uint32_t m_count;
    uint64_t m_generation = 0;
    std::shared_ptr<const RenderPoseValue> m_value;
};

} // namespace infernux
