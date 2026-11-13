#pragma once

#include "Component.h"
#include "physics/PhysicsConstraintOwner.h"
#include <glm/glm.hpp>
#include <initializer_list>

namespace infernux
{
class PhysicsECSStore;
class PhysicsWorld;
class Rigidbody;
struct SemanticTypeDescriptor;

/// Shared body participation and lifetime for Jolt two-body constraints.
/// Limits, axes and motors belong to the concrete constraint type.
class Joint : public Component, protected PhysicsConstraintOwner
{
  public:
    static ComponentTypeConstraints GetTypeConstraints()
    {
        ComponentTypeConstraints constraints;
        constraints.allowMultiple = false;
        constraints.requiredTypes = {"Rigidbody"};
        constraints.satisfiedTypes = {"Joint"};
        constraints.documentComponentReferences = {"connected_body_component_id"};
        return constraints;
    }
    ~Joint() override;
    void Awake() override;
    void OnHierarchyPublished() override;
    void OnEnable() override;
    void OnDisable() override;
    void OnGameObjectDeactivated() override;
    void OnDestroy() override;

    glm::vec3 GetAnchor() const
    {
        return m_anchor;
    }
    void SetAnchor(const glm::vec3 &anchor);
    Rigidbody *GetConnectedBody() const;
    void SetConnectedBody(Rigidbody *body);
    uint64_t GetConnectedBodyComponentId() const
    {
        return m_connectedBodyComponentId;
    }
    bool GetEnableCollision() const
    {
        return m_enableCollision;
    }
    void SetEnableCollision(bool value);
    uint64_t GetPhysicsConstraintId() const
    {
        return m_constraintId;
    }
    nlohmann::json SerializeDocument() const override;
    void RemapComponentReferences(const std::unordered_map<uint64_t, uint64_t> &remap) override;

  protected:
    void RebuildConstraint();
    void ReleaseConstraint();
    virtual uint64_t CreateConstraint(PhysicsWorld &world, uint32_t bodyId, uint32_t connectedBodyId) = 0;
    virtual void UpdateConstraintSettings() = 0;
    virtual void OnConstraintCreated()
    {
    }
    void ReadJointDocumentFields(const nlohmann::json &document);
    void CopyJointFieldsTo(Joint &destination) const;
    static void ValidateJointDocumentFields(const nlohmann::json &document, const char *typeName);
    static SemanticTypeDescriptor DescribeJointType(const char *name, const char *readableId, const char *displayName);
    static nlohmann::json &AddJointField(SemanticTypeDescriptor &type, const char *name, const char *kind,
                                         nlohmann::json initial);
    static void AddJointEnum(SemanticTypeDescriptor &type, const char *field, const char *enumName,
                             std::initializer_list<const char *> members, const char *initial);

    glm::vec3 m_anchor{0};
    uint64_t m_connectedBodyComponentId = 0;
    bool m_enableCollision = false;
    uint64_t m_constraintId = 0;

  private:
    friend class PhysicsECSStore;
    // Intrusive membership is removed before this Joint leaves its live lifetime.
    // Retirement callbacks may enqueue without allocating or invoking scripts.
    Joint *m_pendingPrevious = nullptr;
    Joint *m_pendingNext = nullptr;
    uint64_t m_pendingEpoch = 0;
    bool m_pendingConstraint = false;
    bool m_retryOnRetire = false;
    void OnPhysicsConstraintDestroyed() noexcept final;
    static uint32_t ResolveBodyId(const Rigidbody *body);
    uint32_t m_boundBodyId = 0xFFFFFFFF;
    uint32_t m_boundConnectedBodyId = 0xFFFFFFFF;
};
} // namespace infernux
