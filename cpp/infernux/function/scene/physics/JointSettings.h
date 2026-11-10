#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace infernux
{
enum class JointMotorMode
{
    Off,
    Velocity,
    Position,
    PositionAndVelocity
};
enum class JointSwingType
{
    Cone,
    Pyramid
};
enum class JointMotion
{
    Locked,
    Limited,
    Free
};

/// Angular values exposed to authors use degrees; drive coefficients use SI.
struct SwingTwistJointSettings
{
    glm::vec3 twistAxis{1, 0, 0};
    glm::vec3 planeAxis{0, 0, 1}; // The Jolt constraint's Z axis, not its Y axis.
    JointSwingType swingType = JointSwingType::Cone;
    float normalHalfConeAngle = 45;
    float planeHalfConeAngle = 45;
    float minimumTwistAngle = -45;
    float maximumTwistAngle = 45;
    float frictionTorque = 0;
    JointMotorMode swingMotorMode = JointMotorMode::Off;
    JointMotorMode twistMotorMode = JointMotorMode::Off;
    float swingMotorStiffness = 0;
    float swingMotorDamping = 0;
    float swingMotorMaxTorque = 10;
    float twistMotorStiffness = 0;
    float twistMotorDamping = 0;
    float twistMotorMaxTorque = 10;
    glm::vec4 targetOrientation{0, 0, 0, 1}; // XYZW, in constraint space.
    glm::vec3 targetAngularVelocity{0};      // Degrees/s, owner constraint space.
};

/// Six axes in the bound constraint frame. Linear units are metres, angular units degrees.
struct SixDOFJointSettings
{
    glm::vec3 axis{1, 0, 0};
    glm::vec3 secondaryAxis{0, 1, 0};
    JointMotion xMotion = JointMotion::Locked;
    JointMotion yMotion = JointMotion::Locked;
    JointMotion zMotion = JointMotion::Locked;
    JointMotion angularXMotion = JointMotion::Locked;
    JointMotion angularYMotion = JointMotion::Locked;
    JointMotion angularZMotion = JointMotion::Locked;
    glm::vec3 linearLimitMinimum{-1};
    glm::vec3 linearLimitMaximum{1};
    glm::vec3 angularLimitMinimum{-45};
    glm::vec3 angularLimitMaximum{45};
    glm::vec3 linearLimitStiffness{0};
    glm::vec3 linearLimitDamping{0};
    glm::vec3 linearFrictionForce{0};
    glm::vec3 angularFrictionTorque{0};
    glm::vec3 linearMotorStiffness{0};
    glm::vec3 linearMotorDamping{0};
    glm::vec3 linearMotorMaxForce{10};
    glm::vec3 angularMotorStiffness{0};
    glm::vec3 angularMotorDamping{0};
    glm::vec3 angularMotorMaxTorque{10};
    JointMotorMode xMotorMode = JointMotorMode::Off;
    JointMotorMode yMotorMode = JointMotorMode::Off;
    JointMotorMode zMotorMode = JointMotorMode::Off;
    JointMotorMode angularXMotorMode = JointMotorMode::Off;
    JointMotorMode angularYMotorMode = JointMotorMode::Off;
    JointMotorMode angularZMotorMode = JointMotorMode::Off;
    glm::vec3 targetPosition{0};
    glm::vec4 targetOrientation{0, 0, 0, 1};
    glm::vec3 targetVelocity{0};
    glm::vec3 targetAngularVelocity{0};
};
} // namespace infernux
