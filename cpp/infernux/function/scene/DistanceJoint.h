#pragma once
#include "Joint.h"

namespace infernux
{
/// A Jolt distance range or spring between two attachment points, in metres.
class DistanceJoint final : public Joint
{
  public:
    const char *GetTypeName() const override
    {
        return "DistanceJoint";
    }
    static SemanticTypeDescriptor DescribeSemanticType();
    glm::vec3 GetConnectedAnchor() const
    {
        return m_connectedAnchor;
    }
    void SetConnectedAnchor(const glm::vec3 &anchor);
    bool GetAutoDistance() const
    {
        return m_autoDistance;
    }
    void SetAutoDistance(bool enabled);
    float GetMinimumDistance() const
    {
        return m_minimumDistance;
    }
    float GetMaximumDistance() const
    {
        return m_maximumDistance;
    }
    void SetMinimumDistance(float value)
    {
        SetLimits(value, m_maximumDistance);
    }
    void SetMaximumDistance(float value)
    {
        SetLimits(m_minimumDistance, value);
    }
    void SetLimits(float minimum, float maximum);
    float GetStiffness() const
    {
        return m_stiffness;
    }
    float GetDamping() const
    {
        return m_damping;
    }
    void SetStiffness(float value);
    void SetDamping(float value);
    float GetCurrentDistance() const;
    nlohmann::json SerializeDocument() const override;
    static void ValidateSerializedDocument(const nlohmann::json &document);
    bool DeserializeDocument(const nlohmann::json &document) override;
    std::unique_ptr<Component> Clone() const override;

  private:
    uint64_t CreateConstraint(PhysicsWorld &world, uint32_t bodyId, uint32_t connectedBodyId) override;
    void UpdateConstraintSettings() override;
    glm::vec3 m_connectedAnchor{0};
    bool m_autoDistance = true;
    float m_minimumDistance = 0;
    float m_maximumDistance = 1;
    float m_stiffness = 0;
    float m_damping = 0;
};
} // namespace infernux
