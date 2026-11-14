#include "SkeletonInstance.h"
#include "ComponentDocumentValidation.h"
#include "ComponentFactory.h"
#include "GameObject.h"
#include "Ragdoll.h"
#include "Scene.h"
#include "SceneManager.h"
#include "SkeletonPhysicsPoseRead.h"
#include "SkinnedMeshRenderer.h"
#include "Transform.h"
#include <core/math/TransformPose.h>
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace infernux
{
namespace
{
SemanticTypeDescriptor DescribeSkeletonInstance()
{
    SemanticTypeDescriptor type;
    type.typeGuid = "native:infernux.SkeletonInstance";
    type.readableId = "infernux.component.skeleton_instance";
    type.owner = "engine:native";
    type.origin = "native";
    type.displayName = "Skeleton Instance";
    type.runtimeProfiles = {"editor", "player", "headless"};
    type.fields.push_back({"SkeletonInstance.model",
                           "FieldType.ASSET",
                           false,
                           {{"field_id", "model"},
                            {"serialized_name", "source_model_guid"},
                            {"serialized", true},
                            {"hidden", false},
                            {"nullable", true},
                            {"storage_kind", "native_property"},
                            {"asset_type", "Mesh"},
                            {"setter_owns_document_shape", true},
                            {"display_name", "Skeleton model"},
                            {"default", {{"$type", "asset_ref"}, {"asset_type", "Mesh"}, {"guid", ""}}}}});
    return type;
}
const bool registered = ComponentFactory::Register(
    "SkeletonInstance", [] { return std::make_unique<SkeletonInstance>(); },
    SkeletonInstance::ValidateSerializedDocument, SkeletonInstance::GetTypeConstraints(), DescribeSkeletonInstance);

using Path = std::vector<std::string>;
using NodesByPath = std::map<Path, GameObject *>;

NodesByPath CollectNodes(GameObject &root, const std::string &guid)
{
    NodesByPath nodes;
    const auto collect = [&](const auto &self, GameObject &object) -> void {
        if (object.IsDestroying() || (object.GetScene() && object.GetScene()->IsPendingDestroy(&object)))
            return;
        if (&object != &root) {
            if (object.GetComponent<SkeletonInstance>())
                return; // A nested character owns its own skeleton, even for the same model.
            if (object.GetModelSourceGuid() == guid && !object.GetModelSourcePath().empty()) {
                const auto [previous, inserted] = nodes.emplace(object.GetModelSourcePath(), &object);
                if (!inserted)
                    throw std::invalid_argument("Duplicate skeleton source path on '" + root.GetName() +
                                                "': " + object.GetName() + " (objects " +
                                                std::to_string(previous->second->GetID()) + ", " +
                                                std::to_string(object.GetID()) + ")");
            }
        }
        for (const auto &child : object.GetChildren())
            self(self, *child);
    };
    collect(collect, root);
    return nodes;
}

void ApplyPose(Transform &transform, const SkinnedNodePose &pose)
{
    transform.SetLocalPosition(pose.translation);
    transform.SetLocalRotation(pose.rotation);
    transform.SetLocalScale(pose.scale);
}

Ragdoll *FindRagdoll(GameObject *object)
{
    for (; object; object = object->GetParent())
        if (auto *ragdoll = object->GetComponent<Ragdoll>())
            return ragdoll;
    return nullptr;
}
} // namespace

void SkeletonInstance::Awake()
{
    if (m_model && m_nodes.empty())
        BindHierarchy(true);
}

void SkeletonInstance::ValidateHierarchy() const
{
    if (m_model)
        CollectNodes(*GetGameObject(), m_sourceModelGuid);
}

void SkeletonInstance::OnHierarchyPublished()
{
    RebuildHierarchy();
}

SkeletonInstance::~SkeletonInstance()
{
    SceneManager::Instance().ReleaseSkeletonBones(this, m_registeredBones);
    for (auto *renderer : m_renderers)
        renderer->ForgetSkeleton(this);
    AssetDependencyGraph::Instance().ClearRuntimeDependenciesOf(GetInstanceGuid());
}

void SkeletonInstance::OnModelAssetEvent(const std::string &guid, AssetEvent event)
{
    if (guid != m_sourceModelGuid)
        return;
    if (event == AssetEvent::Modified)
        RebuildHierarchy();
    else if (event == AssetEvent::Deleted) {
        if (auto *ragdoll = FindRagdoll(GetGameObject()))
            ragdoll->OnSkeletonDefinitionChanging(*this, nullptr);
        SceneManager::Instance().ReleaseSkeletonBones(this, m_registeredBones);
        m_registeredBones.clear();
        m_poseDirty = false;
        m_model.reset();
        m_nodes.clear();
        m_bindPose.clear();
        m_animationPose.clear();
        m_bindingError.clear();
        ++m_poseEpoch;
        MarkPoseDirty();
    }
}

void SkeletonInstance::SetSourceModelGuid(const std::string &guid)
{
    const auto previousGuid = m_sourceModelGuid;
    auto *root = GetGameObject();
    const auto *scene = root ? root->GetScene() : nullptr;
    if (scene && scene->IsPendingDestroy(root))
        throw std::logic_error("Cannot change a skeleton pending destruction");
    const auto previousNodes =
        root && scene && !previousGuid.empty() ? CollectNodes(*root, previousGuid) : NodesByPath{};
    if (root && scene && !guid.empty() && guid != previousGuid)
        CollectNodes(*root, guid);
    const auto previousPaths = m_model ? m_model->skeleton.GetNodePaths() : std::vector<Path>{};
    const auto previousBind = m_bindPose;
    LoadDefinition(guid); // Validate the new definition before touching existing scene objects.
    RetireHierarchy(previousGuid);
    BindHierarchy(true);
    // SkeletonInstance owns this hierarchy. Do not also classify its owner as
    // a static imported-model container (that path would scale/rebuild it twice).
    if (root && scene && root->GetModelSourcePath().empty())
        root->SetModelSource({}, {});
    // Reimport updates unedited bind transforms. Authored/animated local poses stay intact.
    if (m_model && guid == previousGuid) {
        std::map<Path, size_t> oldIndices;
        for (size_t index = 0; index < previousPaths.size(); ++index)
            oldIndices.emplace(previousPaths[index], index);
        const auto paths = m_model->skeleton.GetNodePaths();
        for (size_t index = 0; index < paths.size(); ++index) {
            const auto old = oldIndices.find(paths[index]);
            const auto object = previousNodes.find(paths[index]);
            if (old == oldIndices.end() || object == previousNodes.end())
                continue;
            auto &transform = *object->second->GetTransform();
            const auto &pose = previousBind[old->second];
            if (glm::length(transform.GetLocalPosition() - pose.translation) <= 1e-6f &&
                glm::length(transform.GetLocalScale() - pose.scale) <= 1e-6f &&
                std::abs(glm::dot(transform.GetLocalRotation(), pose.rotation)) >= 1.0f - 1e-6f)
                ApplyPose(transform, m_bindPose[index]);
        }
    }
}

void SkeletonInstance::LoadDefinition(const std::string &guid)
{
    if (guid.empty()) {
        if (auto *ragdoll = FindRagdoll(GetGameObject()))
            ragdoll->OnSkeletonDefinitionChanging(*this, nullptr);
        SceneManager::Instance().ReleaseSkeletonBones(this, m_registeredBones);
        m_registeredBones.clear();
        m_poseDirty = false;
        MarkPoseDirty();
        AssetDependencyGraph::Instance().ClearRuntimeDependenciesOf(GetInstanceGuid());
        m_sourceModelGuid.clear();
        m_bindingError.clear();
        m_model.reset();
        m_nodes.clear();
        m_bindPose.clear();
        m_animationPose.clear();
        ++m_poseEpoch;
        return;
    }
    const auto mesh = AssetRegistry::Instance().LoadAsset<InxMesh>(guid, ResourceType::Mesh);
    const auto model = mesh ? mesh->GetSkinnedData() : nullptr;
    if (!model || !model->skeleton.IsValid())
        throw std::invalid_argument("SkeletonInstance requires a model with an imported skeleton");
    std::vector<SkinnedNodePose> bindPose;
    bindPose.reserve(model->skeleton.nodes.size());
    model->skeleton.GetNodePaths();
    for (const auto &node : model->skeleton.nodes) {
        const auto pose = DecomposeTransformPose(node.bindLocal, node.name);
        bindPose.push_back({pose.position * model->scaleFactor, pose.rotation, pose.scale});
    }
    if (auto *ragdoll = FindRagdoll(GetGameObject()))
        ragdoll->OnSkeletonDefinitionChanging(*this, model.get());
    auto &graph = AssetDependencyGraph::Instance();
    SceneManager::Instance().ReleaseSkeletonBones(this, m_registeredBones);
    m_registeredBones.clear();
    m_poseDirty = false;
    graph.ClearRuntimeDependenciesOf(GetInstanceGuid());
    graph.AddRuntimeDependency(GetInstanceGuid(), guid);
    m_sourceModelGuid = guid;
    m_model = model;
    m_bindPose = std::move(bindPose);
    m_animationPose.clear();
    m_nodes.clear();
    m_boundStructureVersion = 0;
    m_bindingError.clear();
    ++m_poseEpoch;
}

void SkeletonInstance::RetireHierarchy(const std::string &guid)
{
    auto *root = GetGameObject();
    auto *scene = root ? root->GetScene() : nullptr;
    if (!scene || guid.empty())
        return;
    const auto existing = CollectNodes(*root, guid);
    std::set<Path> retained;
    if (m_model && guid == m_sourceModelGuid) {
        const auto paths = m_model->skeleton.GetNodePaths();
        retained.insert(paths.begin(), paths.end());
    }
    std::unordered_set<GameObject *> retiring;
    for (const auto &[path, object] : existing)
        if (!retained.count(path))
            retiring.insert(object);
    std::unordered_set<GameObject *> preserved;
    std::unordered_set<uint64_t> scriptedObjects;
    for (const auto &pending : scene->GetPendingPyComponents())
        scriptedObjects.insert(pending.gameObjectId);
    const auto preserveSubtree = [&](const auto &self, GameObject *object) -> void {
        if (retiring.count(object))
            preserved.insert(object);
        for (const auto &child : object->GetChildren())
            self(self, child.get());
    };
    for (auto *object : retiring)
        if (!object->GetAllComponents().empty() || scriptedObjects.count(object->GetID()))
            preserveSubtree(preserveSubtree, object);
    // Deepest first: attachments survive; plain generated Transform nodes retire.
    for (auto item = existing.rbegin(); item != existing.rend(); ++item) {
        auto *object = item->second;
        if (!retiring.count(object))
            continue;
        object->SetModelSource({}, {});
        if (!preserved.count(object)) {
            std::vector<GameObject *> attachments;
            for (const auto &child : object->GetChildren())
                if (!retiring.count(child.get()))
                    attachments.push_back(child.get());
            for (auto *child : attachments)
                child->GetTransform()->SetParent(root->GetTransform(), true);
            scene->DestroyGameObject(object);
        } else if (retiring.count(object->GetParent()) && !preserved.count(object->GetParent()))
            object->GetTransform()->SetParent(root->GetTransform(), true);
    }
}

void SkeletonInstance::RebuildHierarchy()
{
    SetSourceModelGuid(m_sourceModelGuid);
}

void SkeletonInstance::RefreshBindings() const
{
    const auto *root = GetGameObject();
    const auto *scene = root ? root->GetScene() : nullptr;
    if (m_model && scene &&
        (m_nodes.empty() || m_nodes.front().worldId != scene->GetWorldId() ||
         m_boundStructureVersion != scene->GetStructureVersion()))
        const_cast<SkeletonInstance *>(this)->BindHierarchy(false);
}

void SkeletonInstance::EnsureBindings() const
{
    RefreshBindings();
    if (!m_bindingError.empty())
        throw std::runtime_error(m_bindingError);
}

std::optional<uint64_t> SkeletonInstance::GetResultTargetEpoch() const
{
    RefreshBindings();
    return HasBoundHierarchy() ? std::optional{m_poseEpoch} : std::nullopt;
}

void SkeletonInstance::BindHierarchy(bool createMissing)
{
    auto *root = GetGameObject();
    auto *scene = root ? root->GetScene() : nullptr;
    if (!scene || !m_model)
        return; // Staged documents bind after their complete hierarchy is published.
    if (scene->IsPendingDestroy(root))
        throw std::logic_error("Cannot bind a skeleton pending destruction");

    const auto paths = m_model->skeleton.GetNodePaths();
    m_boundStructureVersion = scene->GetStructureVersion();
    m_bindingError.clear();
    NodesByPath existing;
    try {
        existing = CollectNodes(*root, m_sourceModelGuid);
    } catch (const std::invalid_argument &error) {
        if (createMissing)
            throw;
        m_bindingError = error.what();
        ++m_poseEpoch;
        return;
    }
    // Check the complete saved hierarchy before creating or reparenting anything.
    if (!createMissing) {
        for (size_t index = 0; index < paths.size(); ++index) {
            const auto found = existing.find(paths[index]);
            if (found == existing.end()) {
                m_bindingError = "Skeleton node is missing; call rebuild_hierarchy: " + paths[index].back();
                ++m_poseEpoch;
                return;
            }
            const int parent = m_model->skeleton.nodes[index].parent;
            const auto *expected = parent < 0 ? root : existing.at(paths[parent]);
            if (found->second->GetParent() != expected) {
                m_bindingError = "Skeleton topology changed; call rebuild_hierarchy: " + paths[index].back();
                ++m_poseEpoch;
                return;
            }
        }
    }
    std::vector<ObjectHandle> handles;
    std::vector<Transform *> transforms;
    handles.reserve(paths.size());
    transforms.reserve(paths.size());
    for (size_t index = 0; index < paths.size(); ++index) {
        const int parent = m_model->skeleton.nodes[index].parent;
        const auto found = existing.find(paths[index]);
        auto *object = found != existing.end() ? found->second : scene->CreateGameObject(paths[index].back());
        auto *transform = object->GetTransform();
        if (createMissing) {
            object->SetModelSource(m_sourceModelGuid, paths[index]);
            transform->SetParent(parent < 0 ? root->GetTransform() : transforms[parent], false);
            if (found == existing.end())
                ApplyPose(*transform, m_bindPose[index]);
        }
        transforms.push_back(transform);
        handles.push_back(transform->GetHandle());
    }
    if (m_nodes != handles)
        ++m_poseEpoch;
    m_nodes = std::move(handles);
    m_boundStructureVersion = scene->GetStructureVersion();
    SceneManager::Instance().ReleaseSkeletonBones(this, m_registeredBones);
    m_poseDirty = false;
    m_registeredBones = std::move(transforms);
    for (auto *bone : m_registeredBones)
        SceneManager::Instance().RegisterSkeletonBone(bone, this);
    MarkPoseDirty();
    if (createMissing)
        if (auto *ragdoll = FindRagdoll(root))
            ragdoll->OnSkeletonHierarchyReady(*this);
}

void SkeletonInstance::AttachRenderer(SkinnedMeshRenderer *renderer)
{
    m_renderers.insert(renderer);
    MarkPoseDirty();
}

void SkeletonInstance::DetachRenderer(SkinnedMeshRenderer *renderer)
{
    m_renderers.erase(renderer);
}

void SkeletonInstance::MarkPoseDirty()
{
    if (!m_poseDirty) {
        m_poseDirty = true;
        SceneManager::Instance().QueueSkeletonPose(this);
    }
}

void SkeletonInstance::PublishRenderPoses()
{
    if (!m_poseDirty)
        return;
    const auto *scene = GetGameObject()->GetScene();
    if (scene && scene->IsPendingDestroy(GetGameObject())) {
        m_poseDirty = false;
        return;
    }
    if (m_model && scene &&
        (m_nodes.empty() || m_nodes.front().worldId != scene->GetWorldId() ||
         m_boundStructureVersion != scene->GetStructureVersion()))
        BindHierarchy(false);
    if (!m_bindingError.empty()) {
        for (auto *renderer : m_renderers)
            renderer->InvalidateSkeletonPose();
        m_poseDirty = false;
        return;
    }
    if (m_renderers.empty()) {
        m_poseDirty = false;
        return;
    }
    SkeletonPhysicsPoseRead reading(GetBones(), SkeletonPhysicsPoseRead::Space::Local);
    const auto matrices = GetRootSpaceMatrices();
    for (auto *renderer : m_renderers)
        renderer->PublishSkeletonPalette(matrices);
    m_poseDirty = false;
}

Transform *SkeletonInstance::ResolveBone(size_t index) const
{
    EnsureBindings();
    const auto *scene = GetGameObject()->GetScene();
    auto *bone = scene ? dynamic_cast<Transform *>(scene->ResolveComponent(m_nodes.at(index))) : nullptr;
    if (!bone)
        throw std::runtime_error("Skeleton bone was removed; rebuild the hierarchy before using this skeleton");
    return bone;
}

Transform *SkeletonInstance::GetRootBone() const
{
    return m_model ? ResolveBone(
                         static_cast<size_t>(m_model->skeletonRootNodeIndex < 0 ? 0 : m_model->skeletonRootNodeIndex))
                   : nullptr;
}

Transform *SkeletonInstance::GetBoneTransform(const std::string &name) const
{
    if (!m_model)
        return nullptr;
    const auto found = m_model->skeleton.nodeByName.find(name);
    return found == m_model->skeleton.nodeByName.end() ? nullptr : ResolveBone(found->second);
}

Transform *SkeletonInstance::GetHumanBoneTransform(const std::string &slot) const
{
    if (!m_model || !m_model->humanoid.IsValid())
        throw std::logic_error("Humanoid bone queries require a valid humanoid definition");
    for (const auto &[name, index] : m_model->humanoid.bones)
        if (name == slot)
            return ResolveBone(index);
    return nullptr;
}

std::vector<Transform *> SkeletonInstance::GetBones() const
{
    EnsureBindings();
    std::vector<Transform *> bones;
    bones.reserve(m_nodes.size());
    for (size_t index = 0; index < m_nodes.size(); ++index)
        bones.push_back(ResolveBone(index));
    return bones;
}

std::vector<std::string> SkeletonInstance::GetBoneNames() const
{
    std::vector<std::string> names;
    if (m_model)
        for (const auto &node : m_model->skeleton.nodes)
            names.push_back(node.name);
    return names;
}

std::vector<int> SkeletonInstance::GetParentIndices() const
{
    EnsureBindings();
    std::vector<int> parents;
    if (m_model)
        for (const auto &node : m_model->skeleton.nodes)
            parents.push_back(node.parent);
    return parents;
}

std::vector<glm::mat4> SkeletonInstance::GetBindLocalMatrices() const
{
    std::vector<glm::mat4> matrices;
    for (const auto &pose : m_bindPose)
        matrices.push_back(glm::translate(glm::mat4(1), pose.translation) * glm::mat4_cast(pose.rotation) *
                           glm::scale(glm::mat4(1), pose.scale));
    return matrices;
}

std::vector<glm::mat4> SkeletonInstance::GetWorldMatrices() const
{
    EnsureBindings();
    std::vector<glm::mat4> matrices;
    for (size_t index = 0; index < m_nodes.size(); ++index)
        matrices.push_back(ResolveBone(index)->GetWorldMatrix());
    return matrices;
}

std::vector<glm::mat4> SkeletonInstance::GetLocalMatrices() const
{
    EnsureBindings();
    std::vector<glm::mat4> matrices;
    matrices.reserve(m_nodes.size());
    for (size_t index = 0; index < m_nodes.size(); ++index)
        matrices.push_back(ResolveBone(index)->GetLocalMatrix());
    return matrices;
}

std::vector<glm::mat4> SkeletonInstance::GetRootSpaceMatrices() const
{
    auto matrices = GetLocalMatrices();
    for (size_t index = 0; index < matrices.size(); ++index) {
        const auto parent = m_model->skeleton.nodes[index].parent;
        if (parent >= 0)
            matrices[index] = matrices[parent] * matrices[index];
    }
    return matrices;
}

void SkeletonInstance::ResetToBindPose()
{
    EnsureBindings();
    m_animationPose = m_bindPose;
    for (size_t index = 0; index < m_nodes.size(); ++index)
        ApplyPose(*ResolveBone(index), m_bindPose[index]);
    ++m_poseEpoch;
}

nlohmann::json SkeletonInstance::SerializeDocument() const
{
    auto document = Component::SerializeDocument();
    document["source_model_guid"] = m_sourceModelGuid;
    return document;
}

void SkeletonInstance::ValidateSerializedDocument(const nlohmann::json &document)
{
    component_document_validation::ValidateComponentDocument(document, "SkeletonInstance", {"source_model_guid"});
    component_document_validation::RequireString(document, "source_model_guid", "SkeletonInstance");
}

bool SkeletonInstance::DeserializeDocument(const nlohmann::json &document)
{
    ValidateSerializedDocument(document);
    const auto *owner = GetGameObject();
    const auto *scene = owner ? owner->GetScene() : nullptr;
    const bool published = scene && scene->ResolveGameObject(owner->GetHandle()) == owner;
    if (!Component::DeserializeDocument(document))
        return false;
    const auto guid = document.at("source_model_guid").get<std::string>();
    if (published)
        SetSourceModelGuid(guid);
    else
        LoadDefinition(guid);
    return true;
}

std::unique_ptr<Component> SkeletonInstance::Clone() const
{
    auto copy = std::make_unique<SkeletonInstance>();
    copy->SetEnabled(IsEnabled());
    copy->SetExecutionOrder(GetExecutionOrder());
    copy->LoadDefinition(m_sourceModelGuid);
    return copy;
}
} // namespace infernux
