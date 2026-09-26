#pragma once

#include "Component.h"
#include <glm/glm.hpp>

namespace infernux
{

class Rigidbody;

/// A single-axis Jolt hinge. The joint lives on the same GameObject as its
/// required Rigidbody and may connect to another Rigidbody or the fixed world.
class HingeJoint final : public Component
{
  public:
    [[nodiscard]] static ComponentTypeConstraints GetTypeConstraints()
    {
        ComponentTypeConstraints constraints;
        constraints.allowMultiple = false;
        constraints.requiredTypes = {"Rigidbody"};
        return constraints;
    }

    ~HingeJoint() override;

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
        return "HingeJoint";
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
    [[nodiscard]] uint64_t GetConnectedBodyComponentId() const
    {
        return m_connectedBodyComponentId;
    }

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
    [[nodiscard]] bool GetEnableCollision() const
    {
        return m_enableCollision;
    }
    void SetEnableCollision(bool value);
    [[nodiscard]] float GetCurrentAngle() const;

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
    glm::vec3 m_axis{1.0f, 0.0f, 0.0f};
    uint64_t m_connectedBodyComponentId = 0;
    bool m_useLimits = false;
    float m_minimumAngle = -90.0f;
    float m_maximumAngle = 90.0f;
    bool m_enableCollision = false;

    uint64_t m_constraintId = 0;
    uint32_t m_boundBodyId = 0xFFFFFFFF;
    uint32_t m_boundConnectedBodyId = 0xFFFFFFFF;
};

} // namespace infernux
