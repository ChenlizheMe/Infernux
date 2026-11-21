#pragma once

#include <cstdint>
#include <memory>

namespace infernux
{
namespace rhi
{
class ComputeBuffer;
}

/// A scalar-aligned range in a caller-owned device buffer.
struct PhysicsDeviceBufferRange
{
    std::shared_ptr<rhi::ComputeBuffer> buffer;
    uint64_t byteOffset = 0;
    uint64_t byteEnd = UINT64_MAX;
};
} // namespace infernux
