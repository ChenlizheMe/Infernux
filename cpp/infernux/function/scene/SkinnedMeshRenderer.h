#pragma once

#include "MeshRenderer.h"

#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

namespace infernux
{

class InxSkinnedMesh;

/**
 * @brief Renderer component for animated skeletal model instances.
 *
 * Runtime FBX skeleton/animation data lives in SkinnedModelCache/InxSkinnedMesh.
 * This component owns playback-facing state and exposes bind-pose vertices plus
 * a per-frame GPU bone palette to the renderer.
 */
class SkinnedMeshRenderer : public MeshRenderer
{
  public:
    SkinnedMeshRenderer() = default;

    [[nodiscard]] const char *GetTypeName() const override
    {
        return "SkinnedMeshRenderer";
    }

    void SetSourceModelGuid(const std::string &guid);
    [[nodiscard]] const std::string &GetSourceModelGuid() const
    {
        return m_sourceModelGuid;
    }

    void SetSourceModelPath(const std::string &path);
    [[nodiscard]] const std::string &GetSourceModelPath() const
    {
        return m_sourceModelPath;
    }

    void SetAnimationTakeNames(std::vector<std::string> names)
    {
        m_animationTakeNames = std::move(names);
    }
    [[nodiscard]] const std::vector<std::string> &GetAnimationTakeNames() const
    {
        return m_animationTakeNames;
    }

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
    void SetRuntimeAnimationNormalizedTime(float n)
    {
        m_runtimeAnimationNormalized = n;
    }
    [[nodiscard]] float GetRuntimeAnimationNormalizedTime() const
    {
        return m_runtimeAnimationNormalized;
    }

    /// Optional second clip for cross-fade / pose blending. Runtime-only.
    void SubmitAnimationPose(const std::string &takeName, float timeSeconds, float normalizedTime,
                             const std::string &blendTakeName, float blendTimeSeconds, float blendWeight);
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

    [[nodiscard]] bool HasAnimationTakes() const
    {
        return !m_animationTakeNames.empty();
    }
    [[nodiscard]] float GetAnimationDurationSeconds(const std::string &takeName) const;

    [[nodiscard]] bool HasRuntimeSkinnedMesh() const
    {
        return !m_runtimeSkinnedVertices.empty() && !m_runtimeSkinnedIndices.empty() &&
               !m_runtimeSkinBoneMatrices.empty();
    }
    [[nodiscard]] const std::vector<Vertex> &GetRuntimeSkinnedVertices() const
    {
        return m_runtimeSkinnedVertices;
    }
    [[nodiscard]] const std::vector<uint32_t> &GetRuntimeSkinnedIndices() const
    {
        return m_runtimeSkinnedIndices;
    }
    [[nodiscard]] const std::vector<SubMesh> &GetRuntimeSkinnedSubMeshes() const
    {
        return m_runtimeSkinnedSubMeshes;
    }
    [[nodiscard]] const std::vector<glm::mat4> &GetRuntimeSkinBoneMatrices() const
    {
        return m_runtimeSkinBoneMatrices;
    }

    [[nodiscard]] std::string Serialize() const override;
    bool Deserialize(const std::string &jsonStr) override;
    [[nodiscard]] std::unique_ptr<Component> Clone() const override;

  private:
    void RefreshRuntimeSkinnedMesh();
    void ClearRuntimeSkinnedMesh();
    [[nodiscard]] std::shared_ptr<InxSkinnedMesh> GetOrLoadRuntimeModel() const;

    std::string m_sourceModelGuid;
    std::string m_sourceModelPath;
    std::vector<std::string> m_animationTakeNames;
    std::string m_activeTakeName;
    float m_runtimeAnimationTime = 0.0f;
    float m_runtimeAnimationNormalized = 0.0f;
    std::string m_blendTakeName;
    float m_blendAnimationTime = 0.0f;
    float m_blendWeight = 0.0f;
    std::vector<Vertex> m_runtimeSkinnedVertices;
    std::vector<uint32_t> m_runtimeSkinnedIndices;
    std::vector<SubMesh> m_runtimeSkinnedSubMeshes;
    std::vector<glm::mat4> m_runtimeSkinBoneMatrices;
    mutable std::shared_ptr<InxSkinnedMesh> m_runtimeModel;
};

} // namespace infernux
