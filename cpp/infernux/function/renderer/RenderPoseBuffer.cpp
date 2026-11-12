#include "RenderPoseBuffer.h"

#include <stdexcept>

namespace infernux
{
namespace
{
void RequirePoseRange(const std::shared_ptr<rhi::ComputeBuffer> &buffer, uint32_t first, uint32_t count,
                      uint64_t stride)
{
    if (!buffer || buffer->GetDesc().scalarType != rhi::ComputeScalarType::Float32)
        throw std::invalid_argument("Pose storage must be an engine float32 buffer");
    const uint64_t offset = uint64_t{first} * stride;
    const uint64_t bytes = uint64_t{count} * stride;
    if (offset > buffer->GetByteSize() || bytes > buffer->GetByteSize() - offset)
        throw std::out_of_range("Pose range exceeds its buffer allocation");
    if (!buffer->GetLastWriteSubmission().IsValid() || !buffer->HasInitializedRange(offset, bytes))
        throw std::invalid_argument(
            "Pose publication requires accepted values for the complete model and bounds ranges");
}
} // namespace

RenderPoseBuffer::RenderPoseBuffer(const std::shared_ptr<rhi::ComputeBuffer> &models,
                                   const std::shared_ptr<rhi::ComputeBuffer> &bounds, uint32_t count,
                                   uint32_t firstModel, uint32_t firstBounds)
    : m_count(count)
{
    if (count == 0)
        throw std::invalid_argument("A pose source must contain at least one render instance");
    Publish(models, bounds, firstModel, firstBounds);
}

void RenderPoseBuffer::Publish(const std::shared_ptr<rhi::ComputeBuffer> &models,
                               const std::shared_ptr<rhi::ComputeBuffer> &bounds, uint32_t firstModel,
                               uint32_t firstBounds)
{
    RequirePoseRange(models, firstModel, m_count, 64);
    RequirePoseRange(bounds, firstBounds, m_count, 32);
    if (!models->GetHost().SharesServicesWith(bounds->GetHost()) ||
        (m_value && !models->GetHost().SharesServicesWith(m_value->models->GetHost())))
        throw std::invalid_argument("Pose publication must use one rendering device and compute host");

    std::shared_ptr<const RenderPoseValue> next = std::make_shared<RenderPoseValue>(RenderPoseValue{
        models->Freeze(), bounds->Freeze(), m_identity, ++m_generation, m_count, firstModel, firstBounds});
    std::atomic_store_explicit(&m_value, std::move(next), std::memory_order_release);
}
} // namespace infernux
