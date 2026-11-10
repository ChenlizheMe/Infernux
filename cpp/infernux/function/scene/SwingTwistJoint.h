#pragma once
#include "Joint.h"
#include "physics/JointSettings.h"

namespace infernux
{
/// Three angular freedoms with cone/pyramid swing limits and a twist range.
class SwingTwistJoint final : public Joint
{
  public:
    const char *GetTypeName() const override
    {
        return "SwingTwistJoint";
    }
    static SemanticTypeDescriptor DescribeSemanticType();
    glm::vec3 GetTwistAxis() const
    {
        return m_settings.twistAxis;
    }
    void SetTwistAxis(glm::vec3 value);
    glm::vec3 GetPlaneAxis() const
    {
        return m_settings.planeAxis;
    }
    void SetPlaneAxis(glm::vec3 value);
    JointSwingType GetSwingType() const
    {
        return m_settings.swingType;
    }
    void SetSwingType(JointSwingType value);
    float GetNormalHalfConeAngle() const
    {
        return m_settings.normalHalfConeAngle;
    }
    void SetNormalHalfConeAngle(float value);
    float GetPlaneHalfConeAngle() const
    {
        return m_settings.planeHalfConeAngle;
    }
    void SetPlaneHalfConeAngle(float value);
    float GetMinimumTwistAngle() const
    {
        return m_settings.minimumTwistAngle;
    }
    void SetMinimumTwistAngle(float value);
    float GetMaximumTwistAngle() const
    {
        return m_settings.maximumTwistAngle;
    }
    void SetMaximumTwistAngle(float value);
    float GetFrictionTorque() const
    {
        return m_settings.frictionTorque;
    }
    void SetFrictionTorque(float value);
    JointMotorMode GetSwingMotorMode() const
    {
        return m_settings.swingMotorMode;
    }
    void SetSwingMotorMode(JointMotorMode value);
    JointMotorMode GetTwistMotorMode() const
    {
        return m_settings.twistMotorMode;
    }
    void SetTwistMotorMode(JointMotorMode value);
    float GetSwingMotorStiffness() const
    {
        return m_settings.swingMotorStiffness;
    }
    void SetSwingMotorStiffness(float value);
    float GetSwingMotorDamping() const
    {
        return m_settings.swingMotorDamping;
    }
    void SetSwingMotorDamping(float value);
    float GetSwingMotorMaxTorque() const
    {
        return m_settings.swingMotorMaxTorque;
    }
    void SetSwingMotorMaxTorque(float value);
    float GetTwistMotorStiffness() const
    {
        return m_settings.twistMotorStiffness;
    }
    void SetTwistMotorStiffness(float value);
    float GetTwistMotorDamping() const
    {
        return m_settings.twistMotorDamping;
    }
    void SetTwistMotorDamping(float value);
    float GetTwistMotorMaxTorque() const
    {
        return m_settings.twistMotorMaxTorque;
    }
    void SetTwistMotorMaxTorque(float value);
    glm::vec4 GetTargetOrientation() const
    {
        return m_settings.targetOrientation;
    }
    void SetTargetOrientation(glm::vec4 value);
    glm::vec3 GetTargetAngularVelocity() const
    {
        return m_settings.targetAngularVelocity;
    }
    void SetTargetAngularVelocity(glm::vec3 value);
    glm::quat GetCurrentOrientation() const;
    void SetFrame(const glm::vec3 &twistAxis, const glm::vec3 &planeAxis);
    void SetTwistLimits(float minimum, float maximum);
    nlohmann::json SerializeDocument() const override;
    static void ValidateSerializedDocument(const nlohmann::json &document);
    bool DeserializeDocument(const nlohmann::json &document) override;
    std::unique_ptr<Component> Clone() const override;

  private:
    friend class PhysicsWorld;
    void RecordPoseDrive(glm::vec4 orientation, bool enableMotors = true)
    {
        if (enableMotors)
            m_settings.swingMotorMode = m_settings.twistMotorMode = JointMotorMode::Position;
        m_settings.targetOrientation = orientation;
    }
    uint64_t CreateConstraint(PhysicsWorld &world, uint32_t bodyId, uint32_t connectedBodyId) override;
    void UpdateConstraintSettings() override;
    void OnConstraintCreated() override
    {
        UpdateConstraintSettings();
    }
    void ApplySettings(SwingTwistJointSettings settings, bool rebuild = false);
    SwingTwistJointSettings m_settings;
};
} // namespace infernux
