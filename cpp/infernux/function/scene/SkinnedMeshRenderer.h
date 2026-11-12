#pragma once

#include "MeshRenderer.h"
#include "SkinPoseHistory.h"

#include <function/resources/InxSkinnedMesh/InxSkinnedMesh.h>

#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace infernux
{

class InxSkinnedMesh;
class SkeletonInstance;
class Transform;

/**
 * @brief Renderer component for animated skeletal model instances.
 *
 * Geometry and inverse binds come from an immutable InxMesh asset. An independent
 * SkeletonInstance owns editable bone Transforms; each skin derives its final
 * palette from that instance and retains the previous rendered palette.
 */
class SkinnedMeshRenderer : public MeshRenderer
{
  public:
    static ComponentTypeConstraints GetTypeConstraints()
    {
        auto constraints = MeshRenderer::GetTypeConstraints();
        constraints.documentComponentReferences = {"skeleton_component_id"};
        return constraints;
    }
    SkinnedMeshRenderer() = default;
    ~SkinnedMeshRenderer() override;
    void OnHierarchyPublished() override;
    void Awake() override;
    void OnEnable() override;
    SkeletonInstance *GetSkeletonInstance() const
    {
        return m_skeleton;
    }
    void SetSkeletonInstance(SkeletonInstance *skeleton);
    void RemapComponentReferences(const std::unordered_map<uint64_t, uint64_t> &componentIdRemap) override;
    std::vector<Transform *> GetBones() const;
    void ForgetSkeleton(SkeletonInstance *skeleton);
    void PublishSkeletonPalette(const std::vector<glm::mat4> &rootSpaceMatrices);
    void InvalidateSkeletonPose();
    glm::mat4 ResolveRenderWorldMatrix(const glm::mat4 &objectWorldMatrix) const override;
    glm::mat4 ResolveBoundsWorldMatrix(const glm::mat4 &objectWorldMatrix) const override;
    [[nodiscard]] bool RequiresCpuPoseForGeometry() const noexcept override
    {
        return true;
    }
    glm::mat4 ResolveSubmeshWorldMatrix(const glm::mat4 &rendererWorldMatrix) const override;

    [[nodiscard]] const char *GetTypeName() const override
    {
        return "SkinnedMeshRenderer";
    }

    void SetSourceModelGuid(const std::string &guid);
    [[nodiscard]] const std::string &GetSourceModelGuid() const
    {
        return GetMeshAssetGuid();
    }

    /// Model asset that owns the currently sampled animation take. Empty uses
    /// the render model itself. This is runtime state driven by SkeletalAnimator.
    void SetAnimationSourceGuid(const std::string &guid);
    [[nodiscard]] const std::string &GetAnimationSourceGuid() const
    {
        return m_animationSourceGuid;
    }

    [[nodiscard]] std::vector<std::string> GetAnimationTakeNames() const;
    [[nodiscard]] size_t GetAnimationTakeCount() const;

    void SetActiveTakeName(const std::string &name);
    [[nodiscard]] const std::string &GetActiveTakeName() const
    {
        return m_activeTakeName;
    }

    /// Elapsed time in the current clip (seconds). Runtime-only, not serialized — fed by SkeletalAnimator.
    void SetRuntimeAnimationTime(float t);
    [[nodiscard]] float GetRuntimeAnimationTime() const
    {
        return m_runtimeAnimationTime;
    }

    /// Normalized clip progress [0,1] when a duration is known. Runtime-only.
    void SetRuntimeAnimationNormalizedTime(float n);
    [[nodiscard]] float GetRuntimeAnimationNormalizedTime() const
    {
        return m_runtimeAnimationNormalized;
    }

    /// Optional second clip for cross-fade / pose blending. Runtime-only.
    /// An empty takeName renders the bind pose (mesh stays visible).
    /// loop=false clamps sampling time so a finished clip holds its end pose.
    void SubmitAnimationPose(const std::string &takeName, float timeSeconds, float normalizedTime,
                             const std::string &blendTakeName, float blendTimeSeconds, float blendWeight,
                             bool loop = true, const std::string &animationSourceGuid = {},
                             const std::string &blendAnimationSourceGuid = {});
    void SetBlendTakeName(const std::string &name);
    [[nodiscard]] const std::string &GetBlendTakeName() const
    {
        return m_blendTakeName;
    }
    void SetBlendAnimationTime(float t);
    [[nodiscard]] float GetBlendAnimationTime() const
    {
        return m_blendAnimationTime;
    }
    void SetBlendWeight(float w);
    [[nodiscard]] float GetBlendWeight() const
    {
        return m_blendWeight;
    }
    void ClearAnimationBlend();

    /// Submit a multi-layer pose stack (AnimationTree output: N-way weighted +
    /// additive blend with optional per-layer bone masks). This overrides the
    /// single-clip / crossfade path until ClearPoseStack() is called.
    void SubmitPoseStack(const std::vector<PoseStackLayer> &layers);
    void ClearPoseStack();
    [[nodiscard]] bool HasPoseStack() const
    {
        return m_usePoseStack;
    }

    [[nodiscard]] bool HasAnimationTakes() const
    {
        return GetAnimationTakeCount() != 0;
    }
    [[nodiscard]] float GetAnimationDurationSeconds(const std::string &takeName,
                                                    const std::string &animationSourceGuid = {}) const;
    [[nodiscard]] RootMotionDelta GetRootMotionDelta(const std::string &takeName, float fromSeconds, float toSeconds,
                                                     bool loop, const std::string &animationSourceGuid = {}) const;

    void ReloadSourceModel();
    void OnMeshAssetEvent(const std::string &guid, AssetEvent event) override;
    [[nodiscard]] bool ReferencesModelGuid(const std::string &guid) const;

    [[nodiscard]] bool HasRuntimeSkinnedMesh() const;
    [[nodiscard]] const std::vector<Vertex> &GetRuntimeSkinnedVertices() const;
    [[nodiscard]] const std::vector<uint32_t> &GetRuntimeSkinnedIndices() const;
    [[nodiscard]] const std::vector<SubMesh> &GetRuntimeSkinnedSubMeshes() const;
    [[nodiscard]] std::shared_ptr<const InxSkinnedMesh> GetRuntimeModelSnapshot() const
    {
        return m_runtimeModel;
    }
    [[nodiscard]] const std::vector<glm::mat4> &GetRuntimeSkinBoneMatrices() const;
    [[nodiscard]] std::shared_ptr<const std::vector<glm::mat4>> GetRuntimeSkinBonePalette() const
    {
        EnsureFinalPose();
        return m_skinPoseHistory.Current();
    }
    [[nodiscard]] std::shared_ptr<const std::vector<glm::mat4>> GetPreviousRuntimeSkinBonePalette() const
    {
        EnsureFinalPose();
        return m_skinPoseHistory.Previous();
    }
    [[nodiscard]] SkinPoseHistory::SnapshotPtr GetRuntimeSkinPoseSnapshot() const
    {
        EnsureFinalPose();
        return m_skinPoseHistory.Acquire();
    }
    [[nodiscard]] std::vector<std::string> GetSkeletonNodeNames() const;
    /// Latest submitted pose, with bone-local coordinates expressed in engine units.
    [[nodiscard]] std::vector<glm::mat4> GetSkeletonNodeWorldMatrices(const std::vector<std::string> &nodeNames) const;
    void CommitRuntimeSkinPoseHistory();

    /// Animated geometry can leave the imported bind-pose AABB even when the
    /// GameObject Transform is unchanged. Culling therefore uses the current
    /// skin palette rather than MeshRenderer's static mesh bounds.
    void ComputeWorldBounds(const glm::mat4 &worldMatrix, glm::vec3 &outMin, glm::vec3 &outMax) const override;

    [[nodiscard]] nlohmann::json SerializeDocument() const override;
    static void ValidateSerializedDocument(const nlohmann::json &document);
    [[nodiscard]] std::unique_ptr<Component> Clone() const override;

  private:
    void OnMeshSourceChanged() override;
    void ApplyRendererDocumentFields(const nlohmann::json &document) override;
    std::shared_ptr<const InxSkinnedMesh> GetAnimationSourceModel() const;
    void RefreshRuntimeSkinnedMesh(bool evaluateAnimation = true);
    void BindSkeleton();
    void EnsureFinalPose() const;
    void RebuildBoneMapping() const;
    void UpdateAnimationDependencies();
    void ClearRuntimeSkinnedMesh();
    [[nodiscard]] std::shared_ptr<const InxSkinnedMesh> GetOrLoadRuntimeModel() const;
    [[nodiscard]] std::shared_ptr<const InxSkinnedMesh>
    LoadCompatibleAnimationSource(const std::string &guid, const std::string &takeName = {}) const;

    std::string m_activeTakeName;
    std::string m_animationSourceGuid;
    std::unordered_set<std::string> m_animationDependencies;
    std::string m_dependencyGeometryGuid;
    std::string m_blendAnimationSourceGuid;
    float m_runtimeAnimationTime = 0.0f;
    float m_runtimeAnimationNormalized = 0.0f;
    std::string m_blendTakeName;
    float m_blendAnimationTime = 0.0f;
    float m_blendWeight = 0.0f;
    bool m_runtimeAnimationLoop = true;
    std::vector<PoseStackLayer> m_poseStack;
    std::vector<std::shared_ptr<const InxSkinnedMesh>> m_poseStackAnimationSources;
    bool m_usePoseStack = false;
    SkinPoseHistory m_skinPoseHistory;
    SkeletonInstance *m_skeleton = nullptr;
    uint64_t m_skeletonComponentId = 0;
    mutable std::vector<size_t> m_skinBoneNodes;
    mutable bool m_boneMappingChanged = true;
    mutable std::string m_skinBindingError;
    mutable std::shared_ptr<const InxSkinnedMesh> m_boundSkeletonModel;
    mutable std::shared_ptr<const InxSkinnedMesh> m_boundSkinModel;
    mutable std::shared_ptr<const InxSkinnedMesh> m_boundsModel;
    mutable SkinPoseHistory::PalettePtr m_boundsPalette;
    mutable int32_t m_boundsNodeGroup = -1;
    mutable int32_t m_boundsSubmeshIndex = -1;
    mutable glm::vec3 m_skinnedBoundsMin{0.0f};
    mutable glm::vec3 m_skinnedBoundsMax{0.0f};
    mutable bool m_skinnedBoundsValid = false;
    mutable std::shared_ptr<const InxSkinnedMesh> m_runtimeModel;
    mutable std::shared_ptr<const InxSkinnedMesh> m_animationSourceModel;
    mutable std::shared_ptr<const InxSkinnedMesh> m_blendAnimationSourceModel;
    mutable std::unordered_set<std::string> m_reportedRetargetWarnings;
};

} // namespace infernux
