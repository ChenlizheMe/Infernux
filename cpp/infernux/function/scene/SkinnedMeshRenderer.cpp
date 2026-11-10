#include "SkinnedMeshRenderer.h"
#include "ComponentFactory.h"
#include "GameObject.h"
#include "SceneManager.h"
#include "SkeletonInstance.h"
#include "Transform.h"

#include <core/config/MathConstants.h>
#include <core/log/InxLog.h>
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMesh/InxMesh.h>

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::json;

namespace infernux
{

namespace
{
SemanticTypeDescriptor DescribeSkinnedMeshRenderer()
{
    auto type = MeshRenderer::DescribeSemanticType();
    type.typeGuid = "native:infernux.SkinnedMeshRenderer";
    type.readableId = "infernux.component.skinned_mesh_renderer";
    type.displayName = "SkinnedMeshRenderer";
    type.baseTypeGuid = "native:infernux.MeshRenderer";
    type.fields.push_back({"SkinnedMeshRenderer.active_take_name",
                           "FieldType.STRING",
                           false,
                           {{"field_id", "active_take_name"},
                            {"serialized_name", "activeTakeName"},
                            {"serialized", true},
                            {"document_required", false},
                            {"hidden", false},
                            {"nullable", false},
                            {"storage_kind", "native_property"},
                            {"default", ""},
                            {"tooltip", "Currently selected animation take name"}}});
    type.fields.push_back(
        {"SkinnedMeshRenderer.skeleton",
         "FieldType.COMPONENT",
         false,
         {{"field_id", "skeleton"},
          {"serialized_name", "skeleton_component_id"},
          {"serialized", true},
          {"document_required", false},
          {"hidden", false},
          {"nullable", true},
          {"storage_kind", "native_property"},
          {"component_type", "SkeletonInstance"},
          {"default", nullptr},
          {"tooltip", "Owning skeleton; unassigned renderers bind to their character hierarchy automatically"}}});
    return type;
}

const bool registeredSkinnedMeshRenderer = ComponentFactory::Register(
    "SkinnedMeshRenderer", [] { return std::make_unique<SkinnedMeshRenderer>(); },
    SkinnedMeshRenderer::ValidateSerializedDocument, SkinnedMeshRenderer::GetTypeConstraints(),
    DescribeSkinnedMeshRenderer);

const std::vector<glm::mat4> &EmptySkinPalette()
{
    static const std::vector<glm::mat4> empty;
    return empty;
}
const std::vector<Vertex> &EmptyVertices()
{
    static const std::vector<Vertex> empty;
    return empty;
}
const std::vector<uint32_t> &EmptyIndices()
{
    static const std::vector<uint32_t> empty;
    return empty;
}
const std::vector<SubMesh> &EmptySubMeshes()
{
    static const std::vector<SubMesh> empty;
    return empty;
}

std::string MapSkinNodes(const InxSkinnedMesh &skin, const InxSkinnedMesh &skeleton, std::vector<size_t> &nodes)
{
    nodes.reserve(skin.skeleton.bones.size());
    for (const auto &bone : skin.skeleton.bones) {
        const auto &name = skin.skeleton.nodes.at(bone.nodeIndex).name;
        const auto found = skeleton.skeleton.nodeByName.find(name);
        if (found == skeleton.skeleton.nodeByName.end())
            return "Skin node is absent from the owning skeleton: " + name;
        nodes.push_back(found->second);
    }
    return {};
}

void TransformBounds(const glm::mat4 &matrix, const glm::vec3 &localMin, const glm::vec3 &localMax, glm::vec3 &outMin,
                     glm::vec3 &outMax)
{
    const glm::vec3 center = (localMin + localMax) * 0.5f;
    const glm::vec3 extent = (localMax - localMin) * 0.5f;
    glm::vec3 worldCenter(0.0f);
    glm::vec3 worldExtent(0.0f);
    for (int axis = 0; axis < 3; ++axis) {
        worldCenter[axis] = matrix[3][axis];
        for (int component = 0; component < 3; ++component) {
            const float value = matrix[component][axis];
            worldCenter[axis] += value * center[component];
            worldExtent[axis] += std::abs(value) * extent[component];
        }
    }
    outMin = worldCenter - worldExtent;
    outMax = worldCenter + worldExtent;
}
} // namespace

SkinnedMeshRenderer::~SkinnedMeshRenderer()
{
    if (m_skeleton)
        m_skeleton->DetachRenderer(this);
}

void SkinnedMeshRenderer::OnHierarchyPublished()
{
    RefreshRuntimeSkinnedMesh(false);
}

void SkinnedMeshRenderer::Awake()
{
    RefreshRuntimeSkinnedMesh(false);
}

void SkinnedMeshRenderer::OnEnable()
{
    MeshRenderer::OnEnable();
    RefreshRuntimeSkinnedMesh(false);
}

void SkinnedMeshRenderer::SetSkeletonInstance(SkeletonInstance *skeleton)
{
    if (skeleton && skeleton->GetModelSnapshot() && m_runtimeModel) {
        std::vector<size_t> nodes;
        const auto error = MapSkinNodes(*m_runtimeModel, *skeleton->GetModelSnapshot(), nodes);
        if (!error.empty())
            throw std::invalid_argument(error);
    }
    if (m_skeleton)
        m_skeleton->DetachRenderer(this);
    ForgetSkeleton(m_skeleton);
    m_skeletonComponentId = skeleton ? skeleton->GetComponentID() : 0;
    BindSkeleton();
}

void SkinnedMeshRenderer::ForgetSkeleton(SkeletonInstance *skeleton)
{
    if (m_skeleton != skeleton)
        return;
    m_skeleton = nullptr;
    m_boundSkeletonModel.reset();
    m_boundSkinModel.reset();
    m_skinBoneNodes.clear();
    m_skinPoseHistory.Reset();
    SceneManager::Instance().NotifyMeshRendererChanged(this);
}

void SkinnedMeshRenderer::BindSkeleton()
{
    auto *owner = GetGameObject();
    auto *scene = owner ? owner->GetScene() : nullptr;
    if (!m_runtimeModel || !scene || scene->ResolveGameObject(owner->GetHandle()) != owner)
        return;
    SkeletonInstance *skeleton = nullptr;
    if (m_skeletonComponentId) {
        skeleton = dynamic_cast<SkeletonInstance *>(Component::FindByComponentId(m_skeletonComponentId));
    } else {
        for (auto *node = owner; node && !skeleton; node = node->GetParent())
            skeleton = node->GetComponent<SkeletonInstance>();
        if (!skeleton)
            skeleton = owner->AddComponent<SkeletonInstance>();
        if (skeleton->GetSourceModelGuid().empty())
            skeleton->SetSourceModelGuid(GetSourceModelGuid());
    }
    if (skeleton == m_skeleton)
        return;
    if (m_skeleton) {
        m_skeleton->DetachRenderer(this);
        ForgetSkeleton(m_skeleton);
    }
    m_skeleton = skeleton;
    if (skeleton) {
        m_skeletonComponentId = skeleton->GetComponentID();
        skeleton->AttachRenderer(this);
    }
}

void SkinnedMeshRenderer::RemapComponentReferences(const std::unordered_map<uint64_t, uint64_t> &remap)
{
    if (const auto found = remap.find(m_skeletonComponentId); found != remap.end())
        m_skeletonComponentId = found->second;
}

void SkinnedMeshRenderer::EnsureFinalPose() const
{
    if (m_skeleton)
        m_skeleton->PublishRenderPoses();
}

void SkinnedMeshRenderer::RebuildBoneMapping() const
{
    const auto skeleton = m_skeleton ? m_skeleton->GetModelSnapshot() : nullptr;
    if (skeleton == m_boundSkeletonModel && m_runtimeModel == m_boundSkinModel)
        return;
    std::vector<size_t> nodes;
    m_skinBindingError = skeleton && m_runtimeModel ? MapSkinNodes(*m_runtimeModel, *skeleton, nodes) : std::string{};
    m_skinBoneNodes = std::move(nodes);
    m_boundSkeletonModel = skeleton;
    m_boundSkinModel = m_runtimeModel;
    m_boneMappingChanged = true;
}

std::vector<Transform *> SkinnedMeshRenderer::GetBones() const
{
    RebuildBoneMapping();
    if (!m_skinBindingError.empty())
        throw std::invalid_argument(m_skinBindingError);
    std::vector<Transform *> result;
    if (m_skeleton) {
        const auto bones = m_skeleton->GetBones();
        for (auto node : m_skinBoneNodes)
            result.push_back(bones.at(node));
    }
    return result;
}

glm::mat4 SkinnedMeshRenderer::ResolveRenderWorldMatrix(const glm::mat4 &) const
{
    return m_skeleton ? m_skeleton->GetTransform()->GetWorldMatrix() : glm::mat4(1);
}

glm::mat4 SkinnedMeshRenderer::ResolveBoundsWorldMatrix(const glm::mat4 &objectWorldMatrix) const
{
    return ResolveRenderWorldMatrix(objectWorldMatrix);
}

glm::mat4 SkinnedMeshRenderer::ResolveSubmeshWorldMatrix(const glm::mat4 &rendererWorldMatrix) const
{
    return rendererWorldMatrix;
}

void SkinnedMeshRenderer::PublishSkeletonPalette(const std::vector<glm::mat4> &matrices)
{
    RebuildBoneMapping();
    const bool resetHistory = m_boneMappingChanged || !m_skinPoseHistory.Acquire()->IsValid();
    if (!m_skinBindingError.empty() || !m_boundSkeletonModel || !m_runtimeModel) {
        m_skinPoseHistory.Reset();
        SceneManager::Instance().NotifyMeshRendererChanged(this);
        return;
    }
    auto palette = std::make_shared<std::vector<glm::mat4>>();
    palette->reserve(m_skinBoneNodes.size());
    const auto scale = glm::scale(glm::mat4(1), glm::vec3(m_runtimeModel->scaleFactor));
    for (size_t index = 0; index < m_skinBoneNodes.size(); ++index)
        palette->push_back(matrices.at(m_skinBoneNodes[index]) * scale *
                           m_runtimeModel->skeleton.bones[index].inverseBind);
    m_skinPoseHistory.Publish(std::move(palette), resetHistory);
    m_boneMappingChanged = false;
    auto &manager = SceneManager::Instance();
    manager.QueueSkinPoseHistoryCommit(this);
    manager.NotifyMeshRendererGeometryChanged(this);
    if (resetHistory)
        manager.NotifyMeshRendererChanged(this);
}

void SkinnedMeshRenderer::InvalidateSkeletonPose()
{
    m_skinPoseHistory.Reset();
    SceneManager::Instance().NotifyMeshRendererChanged(this);
}

bool SkinnedMeshRenderer::HasRuntimeSkinnedMesh() const
{
    EnsureFinalPose();
    const auto pose = m_skinPoseHistory.Acquire();
    return m_runtimeModel && m_runtimeModel->IsValid() && pose && pose->IsValid();
}

const std::vector<Vertex> &SkinnedMeshRenderer::GetRuntimeSkinnedVertices() const
{
    return m_runtimeModel ? m_runtimeModel->baseVertices : EmptyVertices();
}

const std::vector<uint32_t> &SkinnedMeshRenderer::GetRuntimeSkinnedIndices() const
{
    return m_runtimeModel ? m_runtimeModel->indices : EmptyIndices();
}

const std::vector<SubMesh> &SkinnedMeshRenderer::GetRuntimeSkinnedSubMeshes() const
{
    return m_runtimeModel ? m_runtimeModel->subMeshes : EmptySubMeshes();
}

const std::vector<glm::mat4> &SkinnedMeshRenderer::GetRuntimeSkinBoneMatrices() const
{
    EnsureFinalPose();
    const auto palette = m_skinPoseHistory.Current();
    return palette ? *palette : EmptySkinPalette();
}

std::vector<std::string> SkinnedMeshRenderer::GetSkeletonNodeNames() const
{
    return m_skeleton ? m_skeleton->GetBoneNames() : std::vector<std::string>{};
}

std::vector<glm::mat4>
SkinnedMeshRenderer::GetSkeletonNodeWorldMatrices(const std::vector<std::string> &nodeNames) const
{
    if (!m_skeleton)
        throw std::logic_error("SkinnedMeshRenderer has no owning skeleton");
    std::vector<glm::mat4> matrices;
    for (const auto &name : nodeNames) {
        auto *bone = m_skeleton->GetBoneTransform(name);
        if (!bone)
            throw std::invalid_argument("Skeleton node was not found: " + name);
        matrices.push_back(bone->GetWorldMatrix());
    }
    return matrices;
}

void SkinnedMeshRenderer::ComputeWorldBounds(const glm::mat4 &worldMatrix, glm::vec3 &outMin, glm::vec3 &outMax) const
{
    const auto model = GetOrLoadRuntimeModel();
    EnsureFinalPose();
    const auto palette = m_skinPoseHistory.Current();
    // Camera and world-transform changes reuse the same local deformation bounds.
    if (model != m_boundsModel || palette != m_boundsPalette || GetNodeGroup() != m_boundsNodeGroup ||
        GetSubmeshIndex() != m_boundsSubmeshIndex) {
        m_boundsModel = model;
        m_boundsPalette = palette;
        m_boundsNodeGroup = GetNodeGroup();
        m_boundsSubmeshIndex = GetSubmeshIndex();
        m_skinnedBoundsValid = model && palette &&
                               model->ComputeSkinnedBounds(*palette, m_skinnedBoundsMin, m_skinnedBoundsMax,
                                                           m_boundsNodeGroup, m_boundsSubmeshIndex);
    }
    if (m_skinnedBoundsValid) {
        TransformBounds(worldMatrix, m_skinnedBoundsMin, m_skinnedBoundsMax, outMin, outMax);
        return;
    }
    outMin = outMax = glm::vec3(worldMatrix[3]);
}

void SkinnedMeshRenderer::SetSourceModelGuid(const std::string &guid)
{
    if (GetMeshAssetGuid() == guid && m_runtimeModel)
        return;
    if (guid.empty()) {
        ClearMeshAsset();
        return;
    }
    auto mesh = AssetRegistry::Instance().LoadAsset<InxMesh>(guid, ResourceType::Mesh);
    if (!mesh)
        throw std::invalid_argument("SkinnedMeshRenderer could not load Mesh asset GUID: " + guid);
    if (!mesh->HasSkinnedData() || !mesh->GetSkinnedData()->IsValid())
        throw std::invalid_argument("SkinnedMeshRenderer requires a renderable Mesh asset with skin data: " + guid);
    SetMeshAsset(guid, mesh);
}

std::shared_ptr<const InxSkinnedMesh>
SkinnedMeshRenderer::LoadCompatibleAnimationSource(const std::string &guid, const std::string &takeName) const
{
    auto target = m_skeleton ? m_skeleton->GetModelSnapshot() : GetOrLoadRuntimeModel();
    if (!target || !target->skeleton.IsValid())
        throw std::invalid_argument("SkinnedMeshRenderer has no target skeleton");
    if (guid.empty() || guid == (m_skeleton ? m_skeleton->GetSourceModelGuid() : GetSourceModelGuid()))
        return target;

    auto mesh = AssetRegistry::Instance().LoadAsset<InxMesh>(guid, ResourceType::Mesh);
    const auto source = mesh ? mesh->GetSkinnedData() : nullptr;
    if (!source || !source->IsAnimationSource())
        throw std::invalid_argument("animation source has no imported skeleton animation data: " + guid);

    const SkinnedRuntimeAnimation *animation = nullptr;
    if (!takeName.empty()) {
        animation = source->FindAnimation(takeName);
        if (!animation)
            throw std::invalid_argument("animation take '" + takeName + "' was not found in source: " + guid);
    } else if (!source->animations.empty()) {
        animation = &source->animations.front();
    }
    if (!animation)
        throw std::invalid_argument("animation source has no takes: " + guid);

    std::string reason;
    if (!target->IsAnimationCompatible(*source, *animation, &reason))
        throw std::invalid_argument("animation source skeleton is incompatible with render model: " + reason);
    if (!reason.empty()) {
        const std::string warningKey = GetSourceModelGuid() + "\n" + guid + "\n" + reason;
        if (m_reportedRetargetWarnings.insert(warningKey).second)
            INXLOG_WARN("SkinnedMeshRenderer: ", reason, " (animation source ", guid, ", target ", GetSourceModelGuid(),
                        ")");
    }
    return source;
}

void SkinnedMeshRenderer::SetAnimationSourceGuid(const std::string &guid)
{
    const std::string canonical = guid == GetSourceModelGuid() ? std::string() : guid;
    if (canonical == m_animationSourceGuid && (canonical.empty() || m_animationSourceModel))
        return;
    auto source = LoadCompatibleAnimationSource(canonical, m_activeTakeName);
    m_animationSourceGuid = canonical;
    m_animationSourceModel = canonical.empty() ? nullptr : std::move(source);
    RefreshRuntimeSkinnedMesh();
}

void SkinnedMeshRenderer::SetActiveTakeName(const std::string &name)
{
    if (m_activeTakeName == name)
        return;
    m_activeTakeName = name;
    RefreshRuntimeSkinnedMesh();
}

void SkinnedMeshRenderer::SetRuntimeAnimationTime(float t)
{
    if (std::abs(m_runtimeAnimationTime - t) <= kEpsilon)
        return;
    m_runtimeAnimationTime = t;
    RefreshRuntimeSkinnedMesh();
}

void SkinnedMeshRenderer::SetRuntimeAnimationNormalizedTime(float n)
{
    m_runtimeAnimationNormalized = n;
}

void SkinnedMeshRenderer::SubmitAnimationPose(const std::string &takeName, float timeSeconds, float normalizedTime,
                                              const std::string &blendTakeName, float blendTimeSeconds,
                                              float blendWeight, bool loop, const std::string &animationSourceGuid,
                                              const std::string &blendAnimationSourceGuid)
{
    const std::string activeGuid = animationSourceGuid == GetSourceModelGuid() ? std::string() : animationSourceGuid;
    const std::string blendGuid =
        blendAnimationSourceGuid == GetSourceModelGuid() ? std::string() : blendAnimationSourceGuid;
    auto activeSource = activeGuid.empty() ? GetOrLoadRuntimeModel()
                                           : (activeGuid == m_animationSourceGuid && m_animationSourceModel
                                                  ? m_animationSourceModel
                                                  : LoadCompatibleAnimationSource(activeGuid, takeName));
    auto blendSource =
        blendTakeName.empty()
            ? nullptr
            : (blendGuid.empty() ? GetOrLoadRuntimeModel()
                                 : (blendGuid == m_blendAnimationSourceGuid && m_blendAnimationSourceModel
                                        ? m_blendAnimationSourceModel
                                        : LoadCompatibleAnimationSource(blendGuid, blendTakeName)));
    // Switching back to the single-clip / crossfade path clears any active
    // pose stack so a stale AnimationTree/blend pose doesn't keep overriding.
    if (m_usePoseStack) {
        m_usePoseStack = false;
        m_poseStack.clear();
        m_poseStackAnimationSources.clear();
    }
    m_activeTakeName = takeName;
    m_animationSourceGuid = activeGuid;
    m_animationSourceModel = activeGuid.empty() ? nullptr : std::move(activeSource);
    m_runtimeAnimationTime = timeSeconds;
    m_runtimeAnimationNormalized = normalizedTime;
    m_blendTakeName = blendTakeName;
    m_blendAnimationSourceGuid = blendGuid;
    m_blendAnimationSourceModel = blendGuid.empty() ? nullptr : std::move(blendSource);
    m_blendAnimationTime = blendTimeSeconds;
    m_blendWeight = std::clamp(blendWeight, 0.0f, 1.0f);
    m_runtimeAnimationLoop = loop;
    RefreshRuntimeSkinnedMesh();
}

void SkinnedMeshRenderer::SetBlendTakeName(const std::string &name)
{
    if (m_blendTakeName == name)
        return;
    m_blendTakeName = name;
    RefreshRuntimeSkinnedMesh();
}

void SkinnedMeshRenderer::SetBlendAnimationTime(float t)
{
    m_blendAnimationTime = t;
    RefreshRuntimeSkinnedMesh();
}

void SkinnedMeshRenderer::SetBlendWeight(float w)
{
    const float clamped = std::clamp(w, 0.0f, 1.0f);
    if (std::abs(m_blendWeight - clamped) <= kEpsilon)
        return;
    m_blendWeight = clamped;
    RefreshRuntimeSkinnedMesh();
}

void SkinnedMeshRenderer::ClearAnimationBlend()
{
    if (m_blendTakeName.empty() && m_blendAnimationTime == 0.0f && m_blendWeight == 0.0f)
        return;
    m_blendTakeName.clear();
    m_blendAnimationSourceGuid.clear();
    m_blendAnimationSourceModel.reset();
    m_blendAnimationTime = 0.0f;
    m_blendWeight = 0.0f;
    RefreshRuntimeSkinnedMesh();
}

void SkinnedMeshRenderer::SubmitPoseStack(const std::vector<PoseStackLayer> &layers)
{
    std::vector<std::shared_ptr<const InxSkinnedMesh>> sources;
    sources.reserve(layers.size());
    for (size_t index = 0; index < layers.size(); ++index) {
        const auto &layer = layers[index];
        if (index < m_poseStack.size() && index < m_poseStackAnimationSources.size() &&
            layer.sourceModelGuid == m_poseStack[index].sourceModelGuid && m_poseStackAnimationSources[index]) {
            sources.push_back(m_poseStackAnimationSources[index]);
        } else {
            sources.push_back(LoadCompatibleAnimationSource(layer.sourceModelGuid, layer.takeName));
        }
    }
    m_poseStack = layers;
    m_poseStackAnimationSources = std::move(sources);
    m_usePoseStack = true;
    // Keep a representative active take name so inspector/UI reflect playback.
    if (!layers.empty()) {
        const PoseStackLayer *dominant = &layers.front();
        for (const auto &ly : layers)
            if (!ly.additive && ly.weight * ly.groupWeight > dominant->weight * dominant->groupWeight)
                dominant = &ly;
        m_activeTakeName = dominant->takeName;
        m_runtimeAnimationTime = dominant->timeSeconds;
    }
    RefreshRuntimeSkinnedMesh();
}

void SkinnedMeshRenderer::ClearPoseStack()
{
    if (!m_usePoseStack)
        return;
    m_usePoseStack = false;
    m_poseStack.clear();
    m_poseStackAnimationSources.clear();
    RefreshRuntimeSkinnedMesh();
}

float SkinnedMeshRenderer::GetAnimationDurationSeconds(const std::string &takeName,
                                                       const std::string &animationSourceGuid) const
{
    auto model = LoadCompatibleAnimationSource(animationSourceGuid, takeName);
    return model ? model->GetAnimationDurationSeconds(takeName) : 0.0f;
}

RootMotionDelta SkinnedMeshRenderer::GetRootMotionDelta(const std::string &takeName, float fromSeconds, float toSeconds,
                                                        bool loop, const std::string &animationSourceGuid) const
{
    auto model = LoadCompatibleAnimationSource(animationSourceGuid, takeName);
    return model ? model->SampleRootMotionDelta(takeName, fromSeconds, toSeconds, loop) : RootMotionDelta{};
}

void SkinnedMeshRenderer::ReloadSourceModel()
{
    m_runtimeModel.reset();
    m_boundsModel.reset();
    m_boundsPalette.reset();
    m_skinnedBoundsValid = false;
    m_animationSourceModel.reset();
    m_blendAnimationSourceModel.reset();
    m_poseStackAnimationSources.clear();
    m_skinPoseHistory.Reset();
    RefreshRuntimeSkinnedMesh(false);
}

void SkinnedMeshRenderer::OnMeshSourceChanged()
{
    // GUID-only assignments have no resolved AssetRef yet. Resolve through the
    // same base publication path before binding this skin's immutable model.
    if (HasMeshAsset())
        MeshRenderer::OnMeshAssetEvent(GetSourceModelGuid(), AssetEvent::RuntimeModified);
    ReloadSourceModel();
}

void SkinnedMeshRenderer::OnMeshAssetEvent(const std::string &guid, AssetEvent event)
{
    if (const auto target = m_skeleton ? m_skeleton->GetModelSnapshot() : nullptr)
        target->InvalidatePoseCache();
    if (guid == GetSourceModelGuid()) {
        MeshRenderer::OnMeshAssetEvent(guid, event);
        if (event == AssetEvent::Deleted) {
            ClearRuntimeSkinnedMesh();
            if (m_skeleton)
                m_skeleton->MarkPoseDirty();
        } else {
            ReloadSourceModel();
        }
    } else if (ReferencesModelGuid(guid)) {
        // Source publication changes future samples, not the current authored pose.
        if (guid == m_animationSourceGuid)
            m_animationSourceModel.reset();
        if (guid == m_blendAnimationSourceGuid)
            m_blendAnimationSourceModel.reset();
        m_poseStackAnimationSources.clear();
    }
}

void SkinnedMeshRenderer::UpdateAnimationDependencies()
{
    std::unordered_set<std::string> sources;
    const auto add = [&](const std::string &guid) {
        if (!guid.empty())
            sources.insert(guid);
    };
    add(m_animationSourceGuid);
    add(m_blendAnimationSourceGuid);
    for (const auto &layer : m_poseStack)
        add(layer.sourceModelGuid);
    const auto &geometry = GetSourceModelGuid();
    if (sources == m_animationDependencies && geometry == m_dependencyGeometryGuid)
        return;
    auto &graph = AssetDependencyGraph::Instance();
    for (const auto &guid : m_animationDependencies)
        if (!sources.count(guid) && guid != geometry)
            graph.RemoveRuntimeDependency(GetInstanceGuid(), guid);
    for (const auto &guid : sources)
        if (guid != geometry)
            graph.AddRuntimeDependency(GetInstanceGuid(), guid);
    m_animationDependencies = std::move(sources);
    m_dependencyGeometryGuid = geometry;
}

bool SkinnedMeshRenderer::ReferencesModelGuid(const std::string &guid) const
{
    if (guid.empty())
        return false;
    if (GetSourceModelGuid() == guid || m_animationSourceGuid == guid || m_blendAnimationSourceGuid == guid)
        return true;
    return std::any_of(m_poseStack.begin(), m_poseStack.end(),
                       [&guid](const PoseStackLayer &layer) { return layer.sourceModelGuid == guid; });
}

void SkinnedMeshRenderer::ClearRuntimeSkinnedMesh()
{
    const auto pose = m_skinPoseHistory.Acquire();
    if (!m_runtimeModel && !m_animationSourceModel && !m_blendAnimationSourceModel &&
        m_poseStackAnimationSources.empty() && (!pose || (!pose->current && !pose->previous)))
        return;
    m_runtimeModel.reset();
    m_boundsModel.reset();
    m_boundsPalette.reset();
    m_skinnedBoundsValid = false;
    m_animationSourceModel.reset();
    m_blendAnimationSourceModel.reset();
    m_poseStackAnimationSources.clear();
    m_skinPoseHistory.Reset();
    MarkMeshBufferDirty();
}

std::shared_ptr<const InxSkinnedMesh> SkinnedMeshRenderer::GetOrLoadRuntimeModel() const
{
    if (m_runtimeModel && m_runtimeModel->IsValid())
        return m_runtimeModel;
    const auto mesh = GetMeshAssetRef().Get();
    if (!mesh)
        return nullptr;
    m_runtimeModel = mesh->GetSkinnedData();
    return m_runtimeModel;
}

std::shared_ptr<const InxSkinnedMesh> SkinnedMeshRenderer::GetAnimationSourceModel() const
{
    if (!m_runtimeModel)
        return nullptr;
    if (!m_animationSourceGuid.empty()) {
        if (!m_animationSourceModel)
            m_animationSourceModel = LoadCompatibleAnimationSource(m_animationSourceGuid);
        return m_animationSourceModel;
    }
    return m_skeleton ? m_skeleton->GetModelSnapshot() : GetOrLoadRuntimeModel();
}

size_t SkinnedMeshRenderer::GetAnimationTakeCount() const
{
    const auto source = GetAnimationSourceModel();
    return source ? source->animations.size() : 0;
}

std::vector<std::string> SkinnedMeshRenderer::GetAnimationTakeNames() const
{
    const auto source = GetAnimationSourceModel();
    std::vector<std::string> names;
    if (source) {
        names.reserve(source->animations.size());
        for (const auto &animation : source->animations)
            names.push_back(animation.name);
    }
    return names;
}

void SkinnedMeshRenderer::RefreshRuntimeSkinnedMesh(bool evaluateAnimation)
{
    UpdateAnimationDependencies();
    if (!HasMeshAsset()) {
        ClearRuntimeSkinnedMesh();
        return;
    }
    // NOTE: an empty m_activeTakeName is a valid state — it renders the bind
    // pose (FindAnimation("") → nullptr → SampleNodePose falls back to the
    // node's bind-local TRS). The old behavior of clearing the mesh made
    // characters invisible before the Animator's first play() and after
    // stop(), which was a P0 correctness bug.

    const InxSkinnedMesh *previousModel = m_runtimeModel.get();
    auto model = GetOrLoadRuntimeModel();
    if (!model || !model->IsValid()) {
        ClearRuntimeSkinnedMesh();
        return;
    }

    const bool modelChanged = previousModel != model.get();
    m_runtimeModel = model;
    if (modelChanged)
        MarkMeshBufferDirty();
    BindSkeleton();
    if (!m_skeleton)
        return; // Document staging only loads immutable geometry; publication binds the instance.
    if (!evaluateAnimation) {
        m_skeleton->MarkPoseDirty();
        return;
    }
    const auto target = m_skeleton->GetModelSnapshot();
    if (!target)
        return;
    if (!m_animationSourceGuid.empty() && !m_animationSourceModel)
        m_animationSourceModel = LoadCompatibleAnimationSource(m_animationSourceGuid, m_activeTakeName);
    if (!m_blendAnimationSourceGuid.empty() && !m_blendAnimationSourceModel && !m_blendTakeName.empty())
        m_blendAnimationSourceModel = LoadCompatibleAnimationSource(m_blendAnimationSourceGuid, m_blendTakeName);
    if (m_usePoseStack && m_poseStackAnimationSources.size() != m_poseStack.size()) {
        m_poseStackAnimationSources.clear();
        m_poseStackAnimationSources.reserve(m_poseStack.size());
        for (const auto &layer : m_poseStack)
            m_poseStackAnimationSources.push_back(LoadCompatibleAnimationSource(layer.sourceModelGuid, layer.takeName));
    }
    const auto animationSource = m_animationSourceModel ? m_animationSourceModel : target;

    std::shared_ptr<const EvaluatedSkinPose> nextPose;
    if (m_usePoseStack) {
        // AnimationTree path: N-way weighted + additive + masked blend.
        // Not cached (the stack changes most frames); built fresh each refresh.
        nextPose = std::make_shared<const EvaluatedSkinPose>(
            target->EvaluatePoseStack(m_poseStack, m_poseStackAnimationSources));
    } else {
        SkinnedSampleRequest request;
        request.takeName = m_activeTakeName;
        request.timeSeconds = m_runtimeAnimationTime;
        request.loop = m_runtimeAnimationLoop;
        request.blendTakeName = m_blendTakeName;
        request.blendTimeSeconds = m_blendAnimationTime;
        request.blendWeight = m_blendWeight;
        nextPose = target->GetOrBuildEvaluatedPose(request, animationSource.get(), m_blendAnimationSourceModel.get());
    }
    m_skeleton->ApplySampledPose(*nextPose);
    if (modelChanged)
        SceneManager::Instance().NotifyMeshRendererChanged(this);
}

void SkinnedMeshRenderer::CommitRuntimeSkinPoseHistory()
{
    if (m_skinPoseHistory.CommitFrame())
        SceneManager::Instance().NotifyMeshRendererContentChanged(this);
}

nlohmann::json SkinnedMeshRenderer::SerializeDocument() const
{
    json j = MeshRenderer::SerializeDocument();
    if (!m_activeTakeName.empty())
        j["activeTakeName"] = m_activeTakeName;
    if (m_skeletonComponentId)
        j["skeleton_component_id"] = m_skeletonComponentId;
    return j;
}

void SkinnedMeshRenderer::ValidateSerializedDocument(const nlohmann::json &document)
{
    ValidateSerializedDocumentForType(document, "SkinnedMeshRenderer");
}

void SkinnedMeshRenderer::ApplyRendererDocumentFields(const nlohmann::json &j)
{
    m_activeTakeName = j.value("activeTakeName", std::string());
    m_skeletonComponentId = j.value("skeleton_component_id", uint64_t{0});
}

std::unique_ptr<Component> SkinnedMeshRenderer::Clone() const
{
    auto clone = std::make_unique<SkinnedMeshRenderer>();
    auto document = SerializeDocument();
    document.erase("component_id");
    if (!clone->DeserializeDocument(document))
        throw std::runtime_error("SkinnedMeshRenderer clone could not restore its authored state");
    CopyMaterialBindingsTo(*clone);
    return clone;
}

} // namespace infernux
