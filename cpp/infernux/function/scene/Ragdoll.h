#pragma once

#include "Component.h"
#include "physics/PhysicsTargetReference.h"
#include <glm/gtc/quaternion.hpp>
#include <vector>

namespace infernux
{
class Rigidbody;
class Joint;
class SkeletonInstance;
class InxSkinnedMesh;
struct SemanticTypeDescriptor;

/// Root controller over real child Rigidbody/Collider/Joint components.
/// Body and constraint lifetime stays with the normal physics component path.
class Ragdoll final : public Component
{
  public:
    static ComponentTypeConstraints GetTypeConstraints()
    {
        ComponentTypeConstraints constraints;
        constraints.allowMultiple = false;
        constraints.documentComponentReferences = {"skeleton_component_id", "generated_component_ids"};
        return constraints;
    }
    const char *GetTypeName() const override
    {
        return "Ragdoll";
    }
    ~Ragdoll() override;
    void OnHierarchyPublished() override;
    void Awake() override;
    void OnEnable() override;
    void OnDisable() override;
    void OnDestroy() override;
    void OnGameObjectDeactivated() override;
    void FixedUpdate(float) override;
    bool WantsRuntimeFixedUpdate() const override
    {
        return true;
    }

    bool Rebuild();
    bool IsBound() const;
    std::vector<Rigidbody *> GetBodies() const;
    size_t GetBodyCount() const
    {
        return GetBodies().size();
    }
    size_t GetConstraintCount() const;
    void DrivePose(const std::vector<glm::quat> &worldRotations);
    /// Convenience over actual member Rigidbody modes; not a second mode store.
    bool GetActive() const;
    void SetActive(bool active);
    bool GetDriveAnimation() const
    {
        return m_driveAnimation;
    }
    void SetDriveAnimation(bool enabled);
    void SetPassive();
    size_t GeneratePhysics(float totalMass = 70.0f);
    float GetTotalMass() const;
    void SetTotalMass(float totalMass);
    SkeletonInstance *GetSkeleton() const;
    void OnSkeletonHierarchyReady(SkeletonInstance &skeleton);
    void OnSkeletonDefinitionChanging(SkeletonInstance &skeleton, const InxSkinnedMesh *definition);
    void RemapComponentReferences(const std::unordered_map<uint64_t, uint64_t> &remap) override;

    static SemanticTypeDescriptor DescribeSemanticType();
    static void ValidateSerializedDocument(const nlohmann::json &document);
    nlohmann::json SerializeDocument() const override;
    bool DeserializeDocument(const nlohmann::json &document) override;
    nlohmann::json PrepareAuthoringDocument(const nlohmann::json &document) const override;
    std::unique_ptr<Component> Clone() const override;

  private:
    void Release();
    void UpdateAnimationDrive();
    std::vector<Rigidbody *> GetConfiguredBodies() const;
    Component *ResolveOwnedComponent(uint64_t id) const;
    bool OwnsObject(const GameObject *object) const;
    uint64_t m_skeletonComponentId = 0;
    std::vector<uint64_t> m_generatedComponentIds;
    uint64_t m_assemblyId = 0;
    uint64_t m_structureVersion = 0;
    std::vector<PhysicsTargetReference> m_bodies;
    std::vector<PhysicsTargetReference> m_joints;
    std::vector<int> m_parents;
    size_t m_poseInputCount = 0;
    std::vector<size_t> m_activePoseInputs;
    bool m_driveAnimation = false;
};
} // namespace infernux
