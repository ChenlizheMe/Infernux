#pragma once

#include "TransformECSStore.h"
#include <vector>

namespace infernux
{
class Transform;

/// Internal CPU skeleton demand boundary: first batch/read the actual physics
/// dependencies, then keep animation writes from causing per-bone demand reads.
/// It retains no numerical state or scope across this synchronous operation.
class SkeletonPhysicsPoseRead final
{
  public:
    enum class Space
    {
        Local,
        World
    };
    SkeletonPhysicsPoseRead(const std::vector<Transform *> &transforms, Space space)
        : m_reading(Prepare(transforms, space))
    {
    }
    SkeletonPhysicsPoseRead(const SkeletonPhysicsPoseRead &) = delete;
    SkeletonPhysicsPoseRead &operator=(const SkeletonPhysicsPoseRead &) = delete;

  private:
    static TransformECSStore &Prepare(const std::vector<Transform *> &transforms, Space space);
    TransformECSStore::PhysicsPoseReadScope m_reading;
};
} // namespace infernux
