#pragma once

#include "Joint.h"
#include <glm/glm.hpp>

namespace infernux
{

class Rigidbody;

/// A single-axis Jolt hinge. The joint lives on the same GameObject as its
/// required Rigidbody and may connect to another Rigidbody or the fixed world.
class HingeJoint final : public Joint
{
  public:
    [[nodiscard]] const char *GetTypeName() const override
    {
        return "HingeJoint";
    }

    [[nodiscard]] glm::vec3 GetAxis() const
    {
        return m_axis;
    }
    void SetAxis(const glm::vec3 &axis);

    [[nodiscard]] bool GetUseLimits() const
    {
        return m_useLimits;
    }
    void SetUseLimits(bool value);
    [[nodiscard]] float GetMinimumAngle() const
    {
        return m_minimumAngle;
    }
    void SetMinimumAngle(float degrees);
    [[nodiscard]] float GetMaximumAngle() const
    {
        return m_maximumAngle;
    }
    void SetMaximumAngle(float degrees);
    [[nodiscard]] bool GetMotorEnabled() const
    {
        return m_motorEnabled;
    }
    void SetMotorEnabled(bool value);
    [[nodiscard]] float GetMotorTargetAngle() const
    {
        return m_motorTargetAngle;
    }
    void SetMotorTargetAngle(float degrees);
    [[nodiscard]] float GetMotorStiffness() const
    {
        return m_motorStiffness;
    }
    void SetMotorStiffness(float stiffness);
    [[nodiscard]] float GetMotorDamping() const
    {
        return m_motorDamping;
    }
    void SetMotorDamping(float damping);
    [[nodiscard]] float GetMotorMaxTorque() const
    {
        return m_motorMaxTorque;
    }
    void SetMotorMaxTorque(float torque);
    [[nodiscard]] float GetJointDamping() const
    {
        return m_jointDamping;
    }
    void SetJointDamping(float damping);
    [[nodiscard]] float GetFrictionTorque() const
    {
        return m_frictionTorque;
    }
    void SetFrictionTorque(float torque);
    /// Reflected motor inertia about the joint axis, kg m^2.
    [[nodiscard]] float GetArmature() const
    {
        return m_armature;
    }
    void SetArmature(float inertia);
    [[nodiscard]] float GetCurrentAngle() const;
    [[nodiscard]] float GetCurrentAngularVelocity() const;
    /// Strict solver state in radians and radians/second; an unbound joint is an error.
    [[nodiscard]] glm::vec2 GetMotionState() const;
    [[nodiscard]] nlohmann::json SerializeDocument() const override;
    static void ValidateSerializedDocument(const nlohmann::json &document);
    bool DeserializeDocument(const nlohmann::json &document) override;
    [[nodiscard]] std::unique_ptr<Component> Clone() const override;

  private:
    friend class PhysicsWorld;
    void RecordPoseDrive(float angle)
    {
        m_motorEnabled = true;
        m_motorTargetAngle = angle;
    }
    uint64_t CreateConstraint(PhysicsWorld &world, uint32_t bodyId, uint32_t connectedBodyId) override;
    void UpdateConstraintSettings() override;
    void UpdateConstraintMotor();
    [[nodiscard]] float GetClampedMotorTargetRadians() const;
    void OnConstraintCreated() override
    {
        UpdateConstraintMotor();
    }

    glm::vec3 m_axis{1.0f, 0.0f, 0.0f};
    bool m_useLimits = false;
    float m_minimumAngle = -90.0f;
    float m_maximumAngle = 90.0f;
    bool m_motorEnabled = false;
    float m_motorTargetAngle = 0.0f;
    float m_motorStiffness = 10.0f;
    float m_motorDamping = 1.0f;
    float m_motorMaxTorque = 1.0f;
    float m_jointDamping = 0.0f;
    float m_frictionTorque = 0.0f;
    float m_armature = 0.0f;
};

} // namespace infernux
