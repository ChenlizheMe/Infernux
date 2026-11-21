#pragma once

#include <cstdint>
#include <memory>

namespace infernux
{
namespace rhi
{
class ComputeBuffer;
}

/// A packed float3 range. The start only needs scalar alignment.
struct PhysicsDeviceVectorInput
{
    std::shared_ptr<rhi::ComputeBuffer> buffer;
    uint64_t byteOffset = 0;
    uint64_t byteEnd = UINT64_MAX;
};
} // namespace infernux
