#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Constraints/TwoBodyConstraint.h>
#include "JointSettings.h"
#include <cstdint>
#include <unordered_map>
#include <variant>
#include <vector>

namespace infernux
{
class JoltCpuScene;

/// Mutations and reads run on the world owner between native updates.
class JoltCpuConstraints final
{
  public:
    static constexpr uint32_t WorldBody = UINT32_MAX;
    struct Binding
    {
        uint32_t ownerBody = WorldBody;
        uint32_t connectedBody = WorldBody;
        glm::vec3 ownerAnchor{0};          // Baked relative to the body origin.
        glm::vec3 connectedAnchor{0};      // Distance only. World position when connectedBody == WorldBody.
        glm::vec3 ownerLocalAnchor{0};     // Original author value for collapsed scale axes.
        glm::vec3 connectedLocalAnchor{0}; // Distance with a body endpoint, before baking.
    };
    struct Hinge
    {
        glm::vec3 axis{0, 0, 1};
        bool useLimits = false;
        float minimum = 0, maximum = 0; // radians
        bool motorEnabled = false;
        float targetAngle = 0, stiffness = 0, damping = 0, maxTorque = 0;
        float jointDamping = 0, frictionTorque = 0, armature = 0;
    };
    struct Slider
    {
        glm::vec3 axis{1, 0, 0};
        bool useLimits = false;
        float minimum = 0, maximum = 0; // metres
    };
    struct Fixed
    {
    };
    struct Distance
    {
        bool autoDistance = true;
        float minimum = 0, maximum = 0, stiffness = 0, damping = 0;
    };
    using Settings = std::variant<Hinge, Slider, Fixed, Distance, SwingTwistJointSettings, SixDOFJointSettings>;

    explicit JoltCpuConstraints(JoltCpuScene &scene) : m_scene(scene)
    {
    }
    ~JoltCpuConstraints();
    JoltCpuConstraints(const JoltCpuConstraints &) = delete;
    JoltCpuConstraints &operator=(const JoltCpuConstraints &) = delete;

    /// Both bodies must exist. The native joint only solves while both endpoints are resident.
    /// ID allocation and component lifetime stay with the owner.
    void Create(uint64_t id, const Binding &binding, const Settings &settings, bool enableCollision);
    void Update(uint64_t id, const Settings &settings, bool enableCollision);
    void SetCollisionEnabled(uint64_t id, bool enabled);
    bool Retire(uint64_t id);
    /// The owner completes retirement for these public IDs before destroying the body.
    std::vector<uint64_t> RetireBody(uint32_t bodyId);
    [[nodiscard]] bool HasBodyConstraints(uint32_t bodyId) const;
    void OnBodyResidenceChanged(uint32_t bodyId);
    void ResetBodyWarmStart(uint32_t bodyId);
    void NotifyBodyShapeChanged(uint32_t bodyId, JPH::Vec3Arg deltaCOM, bool scaleChanged);

    [[nodiscard]] glm::vec2 ReadHinge(uint64_t id) const; // Angle and angular speed in radians.
    [[nodiscard]] float ReadSliderPosition(uint64_t id) const;
    [[nodiscard]] float ReadDistance(uint64_t id) const;
    [[nodiscard]] glm::quat ReadOrientation(uint64_t id) const; // SwingTwist or SixDOF.
    [[nodiscard]] glm::vec3 ReadSixDOFPosition(uint64_t id) const;

    /// SimShapeFilter reads this during Update. The owner mutates it only between updates.
    [[nodiscard]] bool SuppressesCollision(uint32_t bodyA, uint32_t bodyB) const;

  private:
    struct CapturedAnchor
    {
        JPH::Vec3 unscaled = JPH::Vec3::sZero();
        JPH::Vec3 collapsed = JPH::Vec3::sZero();
    };
    static CapturedAnchor CaptureAnchor(JPH::Vec3Arg baked, JPH::Vec3Arg scale, JPH::Vec3Arg authored);

    struct Record
    {
        JPH::Ref<JPH::TwoBodyConstraint> native;
        uint32_t ownerBody;
        uint32_t connectedBody;
        CapturedAnchor ownerAnchor;
        CapturedAnchor connectedAnchor;
        JPH::Vec3 primary;
        JPH::Vec3 secondary;
        bool collisionEnabled;
        bool autoDistance;
        JointSwingType swingType;
        Settings acceptedSettings;
    };

    void EraseBodyIndex(uint32_t bodyId, uint64_t id);
    bool ChangeCollision(Record &joint, bool enabled);
    void WakeEndpoints(const Record &joint, bool invalidateContacts);

    JoltCpuScene &m_scene;
    std::unordered_map<uint64_t, Record> m_joints;
    std::unordered_multimap<uint32_t, uint64_t> m_byBody;
    std::unordered_map<uint64_t, size_t> m_suppressedPairs;
};
} // namespace infernux
