#pragma once

#include <functional>
#include <memory>
#include <utility>

namespace infernux
{
namespace rhi
{
class PreparedComputeBatch;
}

struct PhysicsGpuWork
{
    std::shared_ptr<rhi::PreparedComputeBatch> packet;
    std::function<PhysicsGpuWork()> continueWork;

    PhysicsGpuWork() = default;
    PhysicsGpuWork(std::shared_ptr<rhi::PreparedComputeBatch> value, std::function<PhysicsGpuWork()> continuation = {})
        : packet(std::move(value)), continueWork(std::move(continuation))
    {
    }
};

/// The frame work source runs the next operation after the packet completes.
inline PhysicsGpuWork ContinuePhysicsGpuWork(PhysicsGpuWork work, std::function<PhysicsGpuWork()> next)
{
    if (!work.packet && !work.continueWork)
        return next();
    auto continuation = std::move(work.continueWork);
    work.continueWork = [continuation = std::move(continuation), next = std::move(next)]() mutable {
        return continuation ? ContinuePhysicsGpuWork(continuation(), std::move(next)) : next();
    };
    return work;
}
} // namespace infernux
