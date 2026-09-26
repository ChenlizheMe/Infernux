#pragma once

#include "Component.h"
#include <glm/glm.hpp>

namespace infernux
{

class Rigidbody;

/// A Jolt prismatic joint that permits translation along one local axis and
/// locks every other translational and rotational degree of freedom.
class SliderJoint final : public Component
{
  public:
    [[nodiscard]] static ComponentTypeConstraints GetTypeConstraints()
    {
        ComponentTypeConstraints constraints;
        constraints.allowMultiple = false;
        constraints.requiredTypes = {"Rigidbody"};
        return constraints;
    }

    ~SliderJoint() override;
    void Awake() override;
    void FixedUpdate(float fixedDeltaTime) override;
    void OnEnable() override;
    void OnDisable() override;
    void OnGameObjectDeactivated() override;
    void OnDestroy() override;
    [[nodiscard]] bool WantsRuntimeFixedUpdate() const override
    {
        return true;
    }
    [[nodiscard]] const char *GetTypeName() const override
    {
        return "SliderJoint";
    }

    [[nodiscard]] glm::vec3 GetAnchor() const
    {
        return m_anchor;
    }
    void SetAnchor(const glm::vec3 &anchor);
    [[nodiscard]] glm::vec3 GetAxis() const
    {
        return m_axis;
    }
    void SetAxis(const glm::vec3 &axis);
    [[nodiscard]] Rigidbody *GetConnectedBody() const;
    void SetConnectedBody(Rigidbody *body);
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
    [[nodiscard]] bool GetEnableCollision() const
    {
        return m_enableCollision;
    }
    void SetEnableCollision(bool value);
    [[nodiscard]] float GetCurrentPosition() const;

    [[nodiscard]] nlohmann::json SerializeDocument() const override;
    static void ValidateSerializedDocument(const nlohmann::json &document);
    bool DeserializeDocument(const nlohmann::json &document) override;
    [[nodiscard]] std::unique_ptr<Component> Clone() const override;
    void RemapComponentReferences(const std::unordered_map<uint64_t, uint64_t> &componentIdRemap) override;

  private:
    void RebuildConstraint();
    void ReleaseConstraint();
    [[nodiscard]] uint32_t ResolveBodyId(const Rigidbody *body) const;

    glm::vec3 m_anchor{0.0f};
    glm::vec3 m_axis{0.0f, 1.0f, 0.0f};
    uint64_t m_connectedBodyComponentId = 0;
    bool m_useLimits = true;
    float m_minimumDistance = -1.0f;
    float m_maximumDistance = 1.0f;
    bool m_enableCollision = false;
    uint64_t m_constraintId = 0;
    uint32_t m_boundBodyId = 0xFFFFFFFF;
    uint32_t m_boundConnectedBodyId = 0xFFFFFFFF;
};

} // namespace infernux
