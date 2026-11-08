/**
 * @file Collider.h
 * @brief Base class for all physics collider components.
 *
 * Mirrors Unity's Collider API. Derived classes (BoxCollider, SphereCollider,
 * CapsuleCollider) override CreateJoltShape() to provide their specific geometry.
 *
 * A Collider without an attached Rigidbody acts as a static collider (for raycasts,
 * overlap tests, etc.). With a Rigidbody it becomes dynamic.
 */

#pragma once

#include "Component.h"
#include "physics/PhysicsECSStore.h"
#include <function/resources/AssetRef.h>
#include <function/resources/PhysicMaterial/PhysicMaterial.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace JPH
{
class Shape;
template <class T> class RefConst;
} // namespace JPH

namespace infernux
{

enum class AssetEvent;

// Forward declaration — Collider caches a Rigidbody* pointer but does not
// dereference it in the header (only in Collider.cpp which includes Rigidbody.h).
class Rigidbody;
struct PhysicsBodyPoseUpdate;
struct SemanticTypeDescriptor;

/// Abstract base class for Collider components.
class Collider : public Component
{
  public:
    [[nodiscard]] static ComponentTypeConstraints GetTypeConstraints()
    {
        ComponentTypeConstraints constraints;
        constraints.satisfiedTypes = {"Collider"};
        return constraints;
    }

    /// Build the authoritative semantic declaration shared by each concrete
    /// collider type. Concrete geometry fields are appended by the derived type.
    [[nodiscard]] static SemanticTypeDescriptor DescribeSemanticType(std::string typeName, std::string readableId,
                                                                     std::string displayName);
    static nlohmann::json &AddSemanticField(SemanticTypeDescriptor &type, std::string fieldId, std::string valueType,
                                            nlohmann::json defaultValue, std::string displayNameKey,
                                            std::string tooltip);

    using ECSHandle = PhysicsECSStore::ColliderHandle;

    Collider();
    ~Collider() override;

    // ====================================================================
    // Lifecycle
    // ====================================================================

    void Awake() override;
    void OnHierarchyPublished() override;
    void OnEnable() override;
    void OnDisable() override;
    void OnDestroy() override;

    // ====================================================================
    // Properties (Unity-style)
    // ====================================================================

    /// @brief Is this collider a trigger? (Unity: Collider.isTrigger)
    [[nodiscard]] bool IsTrigger() const
    {
        return Data().isTrigger;
    }
    void SetIsTrigger(bool trigger);

    /// @brief Center offset in local space (Unity: Collider.center for Box/Capsule)
    [[nodiscard]] glm::vec3 GetCenter() const
    {
        return Data().center;
    }
    void SetCenter(const glm::vec3 &center);

    [[nodiscard]] std::shared_ptr<PhysicMaterial> GetPhysicMaterial() const;
    /// First access detaches a private surface instance from the shared material.
    [[nodiscard]] std::shared_ptr<PhysicMaterial> GetMaterial();
    [[nodiscard]] const std::string &GetPhysicMaterialGuid() const;
    void SetPhysicMaterial(std::shared_ptr<PhysicMaterial> material, bool instance = false);
    [[nodiscard]] bool IsMaterialInstance() const
    {
        return m_materialIsInstance;
    }
    void SetPhysicMaterialGuid(const std::string &guid);
    void ClearPhysicMaterial();
    void OnPhysicMaterialAssetEvent(AssetEvent event);

    [[nodiscard]] float GetFriction() const;
    [[nodiscard]] float GetBounciness() const;
    [[nodiscard]] PhysicsMaterialCombine GetFrictionCombine() const;
    [[nodiscard]] PhysicsMaterialCombine GetBounceCombine() const;

    // ====================================================================
    // Jolt integration
    // ====================================================================

    /// Construct a candidate collision shape with shared Jolt ownership.
    [[nodiscard]] virtual JPH::RefConst<JPH::Shape> CreateJoltShape() const = 0;

    /// @brief Get the absolute world scale of the owning GameObject.
    ///        Returns (1,1,1) if no GameObject/Transform is available.
    [[nodiscard]] glm::vec3 GetWorldScale() const;
    /// Local center transformed by the full hierarchy, expressed in the body's rigid frame.
    [[nodiscard]] glm::vec3 GetCenterInRigidSpace() const;

    /// @brief Get the Jolt body ID (0xFFFFFFFF = not registered).
    [[nodiscard]] uint32_t GetBodyId() const;
    [[nodiscard]] uint64_t GetShapeRevision() const;
    [[nodiscard]] static uint32_t GetSharedBodyId(const GameObject *gameObject);

    // ====================================================================
    // Type info
    // ====================================================================

    [[nodiscard]] const char *GetTypeName() const override
    {
        return "Collider";
    }

    // ====================================================================
    // Serialization
    // ====================================================================

    [[nodiscard]] nlohmann::json SerializeDocument() const override;
    bool DeserializeDocument(const nlohmann::json &document) override;

    /// @brief Auto-fit collider shape to sibling MeshRenderer bounds.
    ///        Called in Awake() for freshly-added colliders (not deserialized).
    ///        Override in derived classes to set size/radius/height from mesh AABB.
    virtual void AutoFitToMesh();

    [[nodiscard]] Rigidbody *GetAttachedRigidbody() const;

    /// @brief Get the ECS pool handle.
    [[nodiscard]] ECSHandle GetECSHandle() const
    {
        return m_ecsHandle;
    }

  protected:
    /// @brief Copy base Collider ECS properties to a clone.
    /// Called by derived Clone() implementations.
    void CloneBaseColliderData(Collider &target) const;
    void SetMaterialBinding(AssetRef<PhysicMaterial> material, bool instance);

    /// Called after shape parameters change to update the Jolt body.
    void RebuildShape();

    /// Pool-backed data — read access
    [[nodiscard]] const ColliderECSData &Data() const
    {
        return PhysicsECSStore::Instance().GetCollider(m_ecsHandle);
    }
    /// Pool-backed data — write access
    [[nodiscard]] ColliderECSData &DataMut() const
    {
        return PhysicsECSStore::Instance().GetCollider(m_ecsHandle);
    }

    [[nodiscard]] PhysicsActorData &ActorMut();
    [[nodiscard]] const PhysicsActorData &Actor() const;
    void EnsureActor();
    void DetachActor();

    ECSHandle m_ecsHandle;
    AssetRef<PhysicMaterial> m_physicMaterial;
    bool m_materialIsInstance = false;
};

} // namespace infernux
