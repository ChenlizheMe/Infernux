#pragma once

#include "Joint.h"
#include <glm/glm.hpp>

namespace infernux
{

class Rigidbody;

/// A Jolt prismatic joint that permits translation along one local axis and
/// locks every other translational and rotational degree of freedom.
class SliderJoint final : public Joint
{
  public:
    [[nodiscard]] const char *GetTypeName() const override
    {
        return "SliderJoint";
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
    [[nodiscard]] float GetMinimumDistance() const
    {
        return m_minimumDistance;
    }
    void SetMinimumDistance(float metres);
    [[nodiscard]] float GetMaximumDistance() const
    {
        return m_maximumDistance;
    }
    void SetMaximumDistance(float metres);
    [[nodiscard]] float GetCurrentPosition() const;

    [[nodiscard]] nlohmann::json SerializeDocument() const override;
    static void ValidateSerializedDocument(const nlohmann::json &document);
    bool DeserializeDocument(const nlohmann::json &document) override;
    [[nodiscard]] std::unique_ptr<Component> Clone() const override;

  private:
    uint64_t CreateConstraint(PhysicsWorld &world, uint32_t bodyId, uint32_t connectedBodyId) override;
    void UpdateConstraintSettings() override;

    glm::vec3 m_axis{0.0f, 1.0f, 0.0f};
    bool m_useLimits = true;
    float m_minimumDistance = -1.0f;
    float m_maximumDistance = 1.0f;
};

} // namespace infernux
