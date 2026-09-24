#pragma once

#include <function/resources/InxMesh/InxMesh.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace infernux
{

constexpr uint32_t kMaxSkinInfluences = 4;

struct SkinInfluence
{
    std::array<uint32_t, kMaxSkinInfluences> boneIndex{0, 0, 0, 0};
    std::array<float, kMaxSkinInfluences> weight{0.0f, 0.0f, 0.0f, 0.0f};
};

struct SkinnedRuntimeBone
{
    std::string name;
    int nodeIndex = -1;
    glm::mat4 inverseBind{1.0f};
};

struct SkinnedRuntimeNode
{
    std::string name;
    int parent = -1;
    glm::mat4 bindLocal{1.0f};
    glm::mat4 bindGlobal{1.0f};
};

struct SkinnedRuntimeTrack
{
    /// Source-skeleton node targeted by this channel. FBX channel names are
    /// resolved once during import; skeleton node names establish retarget identity.
    int nodeIndex = -1;
    std::vector<std::pair<double, glm::vec3>> positions;
    std::vector<std::pair<double, glm::quat>> rotations;
    std::vector<std::pair<double, glm::vec3>> scales;
};

struct SkinnedRuntimeFloatCurve
{
    std::string name;
    std::vector<std::pair<double, float>> keys;
};

struct SkinnedRuntimeEvent
{
    double normalizedTime = 0.0;
    std::string function;
    std::string stringArgument;
    double numberArgument = 0.0;
};

struct SkinnedRuntimeAnimation
{
    std::string name;
    std::string id; ///< Stable imported clip identity, independent of label/order.
    double durationTicks = 0.0;
    double ticksPerSecond = 25.0;
    std::vector<SkinnedRuntimeTrack> tracks;
    /// Indexed by source-skeleton node; -1 means that node has no channel.
    std::vector<int> trackByNodeIndex;
    /// Authoritative import defaults consumed by SkeletalAnimator.  An FSM
    /// state may disable looping, but cannot force a non-looping imported clip
    /// to wrap past its authored end.
    bool defaultLoop = true;
    /// Root motion is stored outside the skeletal pose.  The root track in
    /// `tracks` is pinned to the selected reference pose so motion is applied
    /// exactly once to the owning Transform.
    int rootMotionNodeIndex = -1;
    std::string rootMotionReferencePose = "bind_pose";
    std::vector<std::pair<double, glm::vec3>> rootMotionPositions;
    std::vector<std::pair<double, glm::quat>> rootMotionRotations;
    std::vector<SkinnedRuntimeFloatCurve> curves;
    std::vector<SkinnedRuntimeEvent> events;
    std::vector<std::string> boneMask;

    [[nodiscard]] float DurationSeconds() const;
};

struct RootMotionDelta
{
    glm::vec3 translation{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
};

struct SkeletonRetargetMap
{
    /// Target node index -> source node index; -1 leaves the target at bind pose.
    std::vector<int> targetToSourceNode;
    size_t mappedNodes = 0;
    /// Mappings established by exact skeleton node identity.
    size_t exactNameMatches = 0;
    size_t mappedAnimatedNodes = 0;
    /// Source animation tracks that affect at least one mapped target node,
    /// including FBX pivot/helper tracks above an exactly matched joint.
    size_t mappedAnimationTracks = 0;
    size_t missingTargetDeformJoints = 0;
    size_t topologyDifferences = 0;
    bool identicalTopology = false;
    /// True when cross-skeleton identity came exclusively from the published
    /// humanoid slot maps. Node names and source paths do not participate.
    bool humanoidSlots = false;
};

struct HumanoidValidationIssue
{
    std::string code;
    std::string bone;
    std::string detail;
};

/// Resolved Unity-style humanoid description. Slot names are stable Pythonic
/// snake_case identities; node indices are local to this artifact's skeleton.
/// This is the runtime identity consumed by humanoid retargeting.
struct HumanoidRigDefinition
{
    bool enabled = false;
    bool requiredBonesValid = false;
    bool hierarchyValid = false;
    bool referencePoseValid = false;
    std::vector<std::pair<std::string, int>> bones;
    std::vector<HumanoidValidationIssue> issues;

    [[nodiscard]] bool IsValid() const noexcept
    {
        return enabled && requiredBonesValid && hierarchyValid && referencePoseValid;
    }
};

/**
 * Runtime skeleton shared by a renderable skin and independently imported
 * animation sources. Geometry, inverse-bind
 * data, and animation tracks no
 * longer define one asset identity. Automatic retargeting requires exact joint
 *
 * names and the same mapped deformation hierarchy, while allowing unmatched
 * importer helper nodes between those
 * joints.
 */
class Skeleton
{
  public:
    std::vector<SkinnedRuntimeNode> nodes;
    std::unordered_map<std::string, int> nodeByName;
    std::vector<SkinnedRuntimeBone> bones;
    std::unordered_map<std::string, uint32_t> boneByName;

    [[nodiscard]] bool IsValid() const noexcept;
    [[nodiscard]] SkeletonRetargetMap BuildRetargetMap(const Skeleton &source,
                                                       const SkinnedRuntimeAnimation &animation) const;
    [[nodiscard]] bool IsAnimationCompatible(const Skeleton &source, const SkinnedRuntimeAnimation &animation,
                                             std::string *reason = nullptr) const;
    [[nodiscard]] size_t GetRuntimeMemoryBytes() const noexcept;
};

struct SkinnedNodePose
{
    glm::vec3 translation{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 scale{1.0f};
};

struct SkinnedSampleRequest
{
    std::string takeName; ///< Empty = bind pose (no animation applied)
    float timeSeconds = 0.0f;
    bool loop = true; ///< Loop wraps time (fmod); non-loop clamps so the end pose holds
    std::string blendTakeName;
    float blendTimeSeconds = 0.0f;
    float blendWeight = 0.0f;
};

/// One weighted contribution to a multi-layer pose blend (AnimationTree output).
/// Non-additive layers are combined as a coverage-normalized weighted average
/// toward bind pose; additive layers add their (sample − bind) delta on top.
/// An empty boneMask affects all nodes; otherwise only exactly named local
/// node channels are sampled. Children still inherit a masked parent's global
/// transform, but their own channels require explicit mask entries.
struct PoseStackLayer
{
    std::string takeName;
    /// GUID of the model that owns this take. Empty means the render model.
    std::string sourceModelGuid;
    float timeSeconds = 0.0f;
    float weight = 1.0f;
    bool additive = false;
    bool loop = true;
    std::vector<std::string> boneMask;
};

class InxSkinnedMesh
{
  public:
    std::string sourcePath;
    std::string guid;
    float scaleFactor = 0.01f;

    std::vector<Vertex> baseVertices;
    std::vector<SkinInfluence> influences;
    std::vector<uint32_t> indices;
    std::vector<SubMesh> subMeshes;
    std::vector<MeshMorphTarget> morphTargets;
    MeshIndexFormat indexFormat = MeshIndexFormat::Auto;
    MeshCompression compression = MeshCompression::Off;
    // A rig is a real model subresource.  References use this owner GUID plus
    // the stable local id; source filenames never participate in runtime
    // identity.  A definition may point at a compatible published definition
    // owned by another model, while this payload still carries its own bind
    // data for rendering.
    std::string skeletonDefinitionGuid;
    std::string skeletonDefinitionId = "skeleton";
    int skeletonRootNodeIndex = -1;
    std::vector<int> exposedSkeletonNodeIndices;
    HumanoidRigDefinition humanoid;
    Skeleton skeleton;
    std::vector<SkinnedRuntimeAnimation> animations;

    [[nodiscard]] bool IsValid() const
    {
        return !baseVertices.empty() && !indices.empty();
    }
    [[nodiscard]] bool IsAnimationSource() const
    {
        return skeleton.IsValid() && !animations.empty();
    }
    /// Validate imported blend-shape identity and every vertex-aligned stream.
    /// Kept separate from IsValid() because render hot paths only need base
    /// geometry, while publication/Cook boundaries validate the whole asset.
    [[nodiscard]] bool HasValidMorphTargets() const noexcept;
    [[nodiscard]] bool IsAssetPayloadValid() const
    {
        return (IsValid() || IsAnimationSource()) && HasValidMorphTargets();
    }

    [[nodiscard]] const SkinnedRuntimeAnimation *FindAnimation(const std::string &takeName) const;
    [[nodiscard]] float GetAnimationDurationSeconds(const std::string &takeName) const;
    [[nodiscard]] RootMotionDelta SampleRootMotionDelta(const std::string &takeName, float fromSeconds, float toSeconds,
                                                        bool loop) const;
    /// Build the authoritative target-node -> source-node map for one take.
    /// Different valid humanoid assets use their published slot maps; generic
    /// rigs retain the strict same-identity skeleton path.
    [[nodiscard]] SkeletonRetargetMap BuildRetargetMap(const InxSkinnedMesh &source,
                                                       const SkinnedRuntimeAnimation &animation) const;
    [[nodiscard]] bool IsAnimationCompatible(const InxSkinnedMesh &source, const SkinnedRuntimeAnimation &animation,
                                             std::string *reason = nullptr) const;
    [[nodiscard]] std::vector<glm::mat4>
    BuildGpuBonePalette(const SkinnedSampleRequest &request, const InxSkinnedMesh *animationSource = nullptr,
                        const InxSkinnedMesh *blendAnimationSource = nullptr) const;
    [[nodiscard]] std::shared_ptr<const std::vector<glm::mat4>>
    GetOrBuildGpuBonePalette(const SkinnedSampleRequest &request, const InxSkinnedMesh *animationSource = nullptr,
                             const InxSkinnedMesh *blendAnimationSource = nullptr) const;
    [[nodiscard]] std::vector<Vertex> SampleVertices(const SkinnedSampleRequest &request) const;

    /// Compute a conservative local-space AABB for a GPU-skinned pose.  The
    /// palette already contains the model import scale, so the result is in
    /// the same space as the rendered vertices rather than the unscaled FBX
    /// bind-pose stream.
    [[nodiscard]] bool ComputeSkinnedBounds(const std::vector<glm::mat4> &palette, glm::vec3 &outMin,
                                            glm::vec3 &outMax) const;

    /// Build bone matrices from a multi-layer pose stack (N-way weighted +
    /// additive blending with optional per-layer bone masks). Used by the
    /// Python AnimationTree runtime. Not cached (the stack is dynamic).
    [[nodiscard]] std::vector<glm::mat4> BuildBoneMatricesFromPoseStack(
        const std::vector<PoseStackLayer> &layers,
        const std::vector<std::shared_ptr<const InxSkinnedMesh>> &animationSources = {}) const;
    [[nodiscard]] std::vector<glm::mat4> BuildGpuBonePaletteFromPoseStack(
        const std::vector<PoseStackLayer> &layers,
        const std::vector<std::shared_ptr<const InxSkinnedMesh>> &animationSources = {}) const;

    void NormalizeInfluences();
    [[nodiscard]] size_t GetRuntimeMemoryBytes() const noexcept;

  private:
    struct PaletteCacheKey
    {
        std::string takeName;
        int64_t timeMicros = 0;
        bool loop = true;
        std::string blendTakeName;
        std::string animationSourceKey;
        std::string blendAnimationSourceKey;
        int64_t blendTimeMicros = 0;
        int32_t blendWeightMicros = 0;

        bool operator==(const PaletteCacheKey &rhs) const
        {
            return takeName == rhs.takeName && timeMicros == rhs.timeMicros && loop == rhs.loop &&
                   blendTakeName == rhs.blendTakeName && animationSourceKey == rhs.animationSourceKey &&
                   blendAnimationSourceKey == rhs.blendAnimationSourceKey && blendTimeMicros == rhs.blendTimeMicros &&
                   blendWeightMicros == rhs.blendWeightMicros;
        }
    };

    struct PaletteCacheKeyHash
    {
        size_t operator()(const PaletteCacheKey &key) const;
    };

    [[nodiscard]] SkinnedNodePose SampleRetargetedNodePose(const SkinnedRuntimeAnimation *anim,
                                                           const Skeleton &sourceSkeleton, size_t targetNodeIndex,
                                                           int sourceNodeIndex, double tTicks) const;
    [[nodiscard]] std::vector<SkinnedNodePose> BuildRetargetedLocalPoses(const Skeleton &sourceSkeleton,
                                                                         const SkinnedRuntimeAnimation *animation,
                                                                         const SkeletonRetargetMap &mapping,
                                                                         double timeTicks) const;
    [[nodiscard]] std::vector<glm::mat4> BuildBoneMatrices(const SkinnedSampleRequest &request,
                                                           const InxSkinnedMesh *animationSource,
                                                           const InxSkinnedMesh *blendAnimationSource) const;
    [[nodiscard]] static PaletteCacheKey MakePaletteCacheKey(const SkinnedSampleRequest &request,
                                                             const InxSkinnedMesh *animationSource,
                                                             const InxSkinnedMesh *blendAnimationSource);

    mutable std::unordered_map<PaletteCacheKey, std::shared_ptr<const std::vector<glm::mat4>>, PaletteCacheKeyHash>
        m_gpuPaletteCache;
    mutable std::vector<PaletteCacheKey> m_gpuPaletteCacheOrder;
};

} // namespace infernux
