#pragma once

#include "../ObjectHandle.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace infernux
{
class HingeJoint;

/// A strict ordered control binding for one live Play scene. No owning joint
/// pointers survive a call; rebindings require a new batch because zero poses change.
class HingeJointBatch
{
  public:
    explicit HingeJointBatch(const std::vector<HingeJoint *> &joints);
    [[nodiscard]] size_t Size() const noexcept
    {
        return m_joints.size();
    }
    [[nodiscard]] uint64_t GetWorldId() const noexcept
    {
        return m_worldId;
    }
    [[nodiscard]] uint64_t GetSimulationEpoch() const noexcept
    {
        return m_epoch;
    }
    void Validate() const;
    /// Write interleaved (angle rad, angular velocity rad/s), returning the sampled fixed step.
    uint64_t ReadMotion(float *output, size_t count) const;
    /// Targets in radians. Validate the full batch before updating any target.
    void SetMotorTargets(const float *targets, size_t count, std::optional<uint64_t> expectedStep) const;

  private:
    struct Joint
    {
        ObjectHandle handle;
        uint64_t constraintId;
    };
    [[nodiscard]] std::vector<HingeJoint *> Resolve() const;
    uint64_t m_worldId = 0;
    uint64_t m_epoch = 0;
    std::vector<Joint> m_joints;
};
} // namespace infernux
