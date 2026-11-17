#pragma once

#include "rhi/RhiBuffer.h"

#include <glm/glm.hpp>

#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace infernux
{

/// Indexed by the world-local surface ID. Record zero represents background.
struct alignas(16) RenderSurfaceRecord
{
    glm::uvec4 lighting{0u};  // Layer mask, deferred model, receive shadows, cast shadows.
    glm::uvec4 identity{0u};  // Object ID low/high, draw primitive, material slot.
    glm::uvec4 revisions{0u}; // Mesh revision low/high, material revision low/high.
};

static_assert(sizeof(RenderSurfaceRecord) == 48);

class RenderSurfaceRecords final
{
  public:
    explicit RenderSurfaceRecords(std::vector<RenderSurfaceRecord> records) : m_records(std::move(records))
    {
    }

    [[nodiscard]] const glm::uvec4 &Lighting(uint32_t surfaceId) const noexcept
    {
        return m_records[surfaceId].lighting;
    }

    [[nodiscard]] std::shared_ptr<const rhi::BufferResource> Acquire(rhi::Device &device) const
    {
        std::lock_guard lock(m_mutex);
        const auto id = device.GetDeviceId();
        if (const auto found = m_uploads.find(id); found != m_uploads.end())
            return found->second;

        const uint64_t bytes = m_records.size() * sizeof(RenderSurfaceRecord);
        if (!device.GetCapabilities().IsStorageBufferRangeValid(0, bytes))
            throw std::length_error("Render world surface records exceed the device storage range");
        rhi::BufferDesc desc;
        desc.byteSize = bytes;
        desc.usage = rhi::BufferUsageFlags::Storage;
        desc.memory = rhi::BufferMemory::Upload;
        const auto allocation = device.CreateBuffer(desc);
        const auto buffer = std::make_shared<rhi::BufferResource>(device, allocation, bytes);
        if (!allocation.IsValid() || !device.WriteBuffer(allocation, 0, m_records.data(), bytes))
            throw std::runtime_error("Could not publish the render world's surface records");
        m_uploads.emplace(id, buffer);
        return buffer;
    }

  private:
    const std::vector<RenderSurfaceRecord> m_records;
    mutable std::mutex m_mutex;
    mutable std::unordered_map<rhi::DeviceId, std::shared_ptr<const rhi::BufferResource>> m_uploads;
};

} // namespace infernux
