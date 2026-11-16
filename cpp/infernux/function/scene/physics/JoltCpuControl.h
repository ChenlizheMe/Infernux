#pragma once

#include <Jolt/Jolt.h>
#include "JoltCpuScene.h"

#include <functional>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace infernux
{
/// One RuntimeWorld's ordered CPU authoring. Calls run on the world owner.
class JoltCpuControl final
{
  public:
    struct Pose
    {
        std::optional<JPH::RVec3> position;
        std::optional<JPH::Quat> rotation;
        bool cancelDrive = true;
    };
    struct Velocity
    {
        std::optional<JPH::Vec3> linear, angular;
    };
    struct Force
    {
        JoltCpuScene::ForceKind kind;
        ForceMode mode;
        JPH::Vec3 value;
        JPH::RVec3 point = JPH::RVec3::sZero();
    };
    struct Awake
    {
        bool awake;
    };
    struct ResetMotion
    {
    };
    struct Remove
    {
        bool preserveVelocity = false;
    };
    struct KinematicMove
    {
        std::optional<JPH::RVec3> position;
        std::optional<JPH::Quat> rotation;
        float duration;
        float maxSpeed;
    };
    using Command = std::variant<Pose, Velocity, Force, Awake, ResetMotion, Remove>;
    using RetiredJoints = std::function<void(uint32_t bodyId, const std::vector<uint64_t> &jointIds)>;

    explicit JoltCpuControl(JoltCpuScene &scene);
    ~JoltCpuControl();
    JoltCpuControl(const JoltCpuControl &) = delete;
    JoltCpuControl &operator=(const JoltCpuControl &) = delete;

    /// A new public identity can be staged before it owns a native allocation.
    /// An invalid actor handle denotes a static replica; the owner commits the shared author metadata.
    void StageActor(uint32_t bodyId, PhysicsActorHandle actor, PreparedCpuActor prepared);
    [[nodiscard]] bool HasStagedRigidbody(uint32_t bodyId, PhysicsECSStore::RigidbodyHandle rigidbody) const;
    void AddBodies(std::vector<uint32_t> bodyIds);
    void RetireActor(uint32_t bodyId);
    void Queue(uint32_t bodyId, Command command);
    PhysicsTargetSource QueueMove(uint32_t bodyId, KinematicMove command, const PhysicsTargetAuthor *author = nullptr);
    [[nodiscard]] bool HasPending() const noexcept;

    /// The shared Fixed admission boundary takes the old FIFO. Later writes stay pending.
    void SealFixed();
    /// A control-only boundary applies numeric commands but does not start or finish a drive.
    bool ApplyControl(const RetiredJoints &);
    /// Applies the one sealed prefix and starts that Fixed's resident drives.
    void ApplyFixed(const RetiredJoints &);
    /// Called after the shared result is available. Resolves only matching owner/channel sources.
    void PublishApplied();
    /// The RuntimeWorld retires the native scene separately.
    void Close() noexcept;

  private:
    struct State;
    std::unique_ptr<State> m_state;
};
} // namespace infernux
