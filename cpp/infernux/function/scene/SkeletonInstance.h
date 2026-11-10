#pragma once

#include "Component.h"
#include "ObjectHandle.h"
#include <function/resources/InxSkinnedMesh/InxSkinnedMesh.h>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace infernux
{
class Transform;
class SkinnedMeshRenderer;
enum class AssetEvent;

/// One character's editable hierarchy, independent of animation playback.
class SkeletonInstance final : public Component
{
  public:
    static ComponentTypeConstraints GetTypeConstraints()
    {
        ComponentTypeConstraints constraints;
        constraints.allowMultiple = false;
        return constraints;
    }
    const char *GetTypeName() const override
    {
        return "SkeletonInstance";
    }
    ~SkeletonInstance() override;
    void ValidateHierarchy() const override;
    void OnHierarchyPublished() override;
    void Awake() override;
    void OnModelAssetEvent(const std::string &guid, AssetEvent event);
    void SetSourceModelGuid(const std::string &guid);
    const std::string &GetSourceModelGuid() const
    {
        return m_sourceModelGuid;
    }
    std::shared_ptr<const InxSkinnedMesh> GetModelSnapshot() const
    {
        return m_model;
    }
    bool HasBoundHierarchy() const
    {
        return m_model && !m_nodes.empty() && m_bindingError.empty();
    }

    /// Create missing source nodes and bind existing saved/cloned nodes by path.
    void RebuildHierarchy();
    void ResetToBindPose();
    std::optional<uint64_t> GetResultTargetEpoch() const;
    Transform *GetRootBone() const;
    Transform *GetBoneTransform(const std::string &name) const;
    Transform *GetHumanBoneTransform(const std::string &slot) const;
    std::vector<Transform *> GetBones() const;
    std::vector<std::string> GetBoneNames() const;
    std::vector<int> GetParentIndices() const;
    std::vector<glm::mat4> GetBindLocalMatrices() const;
    std::vector<glm::mat4> GetLocalMatrices() const;
    std::vector<glm::mat4> GetRootSpaceMatrices() const;
    std::vector<glm::mat4> GetWorldMatrices() const;
    bool HasAnimationPose() const
    {
        return !m_animationPose.empty();
    }
    bool PhysicsControlsPose() const;
    void CaptureAnimationPose();
    void ApplyAnimationPose();
    std::vector<glm::mat4> GetAnimationWorldMatrices() const;
    std::vector<glm::quat> GetAnimationWorldRotations() const;
    /// Rigidly align the applied animation to a previously captured physical root.
    void AlignAnimationRoot(Transform *physicalRoot, const glm::vec3 &position, const glm::quat &rotation);
    /// Apply this model's sampled locals. Imported translation units convert once here.
    void ApplySampledPose(const EvaluatedSkinPose &pose);
    void AttachRenderer(SkinnedMeshRenderer *renderer);
    void DetachRenderer(SkinnedMeshRenderer *renderer);
    void MarkPoseDirty();
    void PublishRenderPoses();

    nlohmann::json SerializeDocument() const override;
    static void ValidateSerializedDocument(const nlohmann::json &document);
    bool DeserializeDocument(const nlohmann::json &document) override;
    std::unique_ptr<Component> Clone() const override;

  private:
    void LoadDefinition(const std::string &guid);
    void BindHierarchy(bool createMissing);
    void RefreshBindings() const;
    void EnsureBindings() const;
    void RetireHierarchy(const std::string &guid);
    Transform *ResolveBone(size_t index) const;
    std::vector<Transform *> GetPhysicsControlledTransforms() const;
    std::string m_sourceModelGuid;
    std::shared_ptr<const InxSkinnedMesh> m_model;
    std::vector<ObjectHandle> m_nodes;
    std::vector<SkinnedNodePose> m_bindPose;
    std::vector<SkinnedNodePose> m_animationPose;
    uint64_t m_boundStructureVersion = 0;
    uint64_t m_poseEpoch = 1;
    std::string m_bindingError;
    std::vector<Transform *> m_registeredBones;
    std::unordered_set<SkinnedMeshRenderer *> m_renderers;
    bool m_poseDirty = false;
};
} // namespace infernux
