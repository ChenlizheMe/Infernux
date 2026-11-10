#pragma once
#include "Joint.h"
#include "physics/JointSettings.h"

namespace infernux
{
/// Six-axis Jolt constraint. Rotation limits use a pyramid, allowing asymmetric bounds.
class SixDOFJoint final : public Joint
{
  public:
    const char *GetTypeName() const override
    {
        return "SixDOFJoint";
    }
    static SemanticTypeDescriptor DescribeSemanticType();
    glm::vec3 GetAxis() const
    {
        return m_settings.axis;
    }
    void SetAxis(glm::vec3 value);
    glm::vec3 GetSecondaryAxis() const
    {
        return m_settings.secondaryAxis;
    }
    void SetSecondaryAxis(glm::vec3 value);
    JointMotion GetXMotion() const
    {
        return m_settings.xMotion;
    }
    void SetXMotion(JointMotion value);
    JointMotion GetYMotion() const
    {
        return m_settings.yMotion;
    }
    void SetYMotion(JointMotion value);
    JointMotion GetZMotion() const
    {
        return m_settings.zMotion;
    }
    void SetZMotion(JointMotion value);
    JointMotion GetAngularXMotion() const
    {
        return m_settings.angularXMotion;
    }
    void SetAngularXMotion(JointMotion value);
    JointMotion GetAngularYMotion() const
    {
        return m_settings.angularYMotion;
    }
    void SetAngularYMotion(JointMotion value);
    JointMotion GetAngularZMotion() const
    {
        return m_settings.angularZMotion;
    }
    void SetAngularZMotion(JointMotion value);
    glm::vec3 GetLinearLimitMinimum() const
    {
        return m_settings.linearLimitMinimum;
    }
    void SetLinearLimitMinimum(glm::vec3 value);
    glm::vec3 GetLinearLimitMaximum() const
    {
        return m_settings.linearLimitMaximum;
    }
    void SetLinearLimitMaximum(glm::vec3 value);
    glm::vec3 GetAngularLimitMinimum() const
    {
        return m_settings.angularLimitMinimum;
    }
    void SetAngularLimitMinimum(glm::vec3 value);
    glm::vec3 GetAngularLimitMaximum() const
    {
        return m_settings.angularLimitMaximum;
    }
    void SetAngularLimitMaximum(glm::vec3 value);
    glm::vec3 GetLinearLimitStiffness() const
    {
        return m_settings.linearLimitStiffness;
    }
    void SetLinearLimitStiffness(glm::vec3 value);
    glm::vec3 GetLinearLimitDamping() const
    {
        return m_settings.linearLimitDamping;
    }
    void SetLinearLimitDamping(glm::vec3 value);
    glm::vec3 GetLinearFrictionForce() const
    {
        return m_settings.linearFrictionForce;
    }
    void SetLinearFrictionForce(glm::vec3 value);
    glm::vec3 GetAngularFrictionTorque() const
    {
        return m_settings.angularFrictionTorque;
    }
    void SetAngularFrictionTorque(glm::vec3 value);
    glm::vec3 GetLinearMotorStiffness() const
    {
        return m_settings.linearMotorStiffness;
    }
    void SetLinearMotorStiffness(glm::vec3 value);
    glm::vec3 GetLinearMotorDamping() const
    {
        return m_settings.linearMotorDamping;
    }
    void SetLinearMotorDamping(glm::vec3 value);
    glm::vec3 GetLinearMotorMaxForce() const
    {
        return m_settings.linearMotorMaxForce;
    }
    void SetLinearMotorMaxForce(glm::vec3 value);
    glm::vec3 GetAngularMotorStiffness() const
    {
        return m_settings.angularMotorStiffness;
    }
    void SetAngularMotorStiffness(glm::vec3 value);
    glm::vec3 GetAngularMotorDamping() const
    {
        return m_settings.angularMotorDamping;
    }
    void SetAngularMotorDamping(glm::vec3 value);
    glm::vec3 GetAngularMotorMaxTorque() const
    {
        return m_settings.angularMotorMaxTorque;
    }
    void SetAngularMotorMaxTorque(glm::vec3 value);
    JointMotorMode GetXMotorMode() const
    {
        return m_settings.xMotorMode;
    }
    void SetXMotorMode(JointMotorMode value);
    JointMotorMode GetYMotorMode() const
    {
        return m_settings.yMotorMode;
    }
    void SetYMotorMode(JointMotorMode value);
    JointMotorMode GetZMotorMode() const
    {
        return m_settings.zMotorMode;
    }
    void SetZMotorMode(JointMotorMode value);
    JointMotorMode GetAngularXMotorMode() const
    {
        return m_settings.angularXMotorMode;
    }
    void SetAngularXMotorMode(JointMotorMode value);
    JointMotorMode GetAngularYMotorMode() const
    {
        return m_settings.angularYMotorMode;
    }
    void SetAngularYMotorMode(JointMotorMode value);
    JointMotorMode GetAngularZMotorMode() const
    {
        return m_settings.angularZMotorMode;
    }
    void SetAngularZMotorMode(JointMotorMode value);
    glm::vec3 GetTargetPosition() const
    {
        return m_settings.targetPosition;
    }
    void SetTargetPosition(glm::vec3 value);
    glm::vec4 GetTargetOrientation() const
    {
        return m_settings.targetOrientation;
    }
    void SetTargetOrientation(glm::vec4 value);
    glm::vec3 GetTargetVelocity() const
    {
        return m_settings.targetVelocity;
    }
    void SetTargetVelocity(glm::vec3 value);
    glm::vec3 GetTargetAngularVelocity() const
    {
        return m_settings.targetAngularVelocity;
    }
    void SetTargetAngularVelocity(glm::vec3 value);
    glm::quat GetCurrentOrientation() const;
    glm::vec3 GetCurrentPosition() const;
    void SetFrame(const glm::vec3 &axis, const glm::vec3 &secondaryAxis);
    void SetLinearLimits(const glm::vec3 &minimum, const glm::vec3 &maximum);
    void SetAngularLimits(const glm::vec3 &minimum, const glm::vec3 &maximum);
    nlohmann::json SerializeDocument() const override;
    static void ValidateSerializedDocument(const nlohmann::json &document);
    bool DeserializeDocument(const nlohmann::json &document) override;
    std::unique_ptr<Component> Clone() const override;

  private:
    uint64_t CreateConstraint(PhysicsWorld &world, uint32_t bodyId, uint32_t connectedBodyId) override;
    void UpdateConstraintSettings() override;
    void OnConstraintCreated() override
    {
        UpdateConstraintSettings();
    }
    void ApplySettings(SixDOFJointSettings settings, bool frameChanged = false);
    SixDOFJointSettings m_settings;
};
} // namespace infernux
