#include "Ragdoll.h"
#include "ComponentDocumentValidation.h"
#include "ComponentFactory.h"
#include "GameObject.h"
#include "Joint.h"
#include "Rigidbody.h"
#include "Scene.h"
#include "SkeletonInstance.h"
#include "Transform.h"
#include "physics/PhysicsWorld.h"
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace infernux
{
namespace
{
const bool registered = ComponentFactory::Register(
    "Ragdoll", [] { return std::make_unique<Ragdoll>(); }, Ragdoll::ValidateSerializedDocument,
    Ragdoll::GetTypeConstraints(), Ragdoll::DescribeSemanticType);
} // namespace

SemanticTypeDescriptor Ragdoll::DescribeSemanticType()
{
    SemanticTypeDescriptor type;
    type.typeGuid = "native:infernux.Ragdoll";
    type.readableId = "infernux.component.ragdoll";
    type.owner = "engine:native";
    type.origin = "native";
    type.displayName = "Ragdoll";
    type.runtimeProfiles = {"editor", "player", "headless"};
    type.fields.push_back({"Ragdoll.drive_animation",
                           "FieldType.BOOL",
                           false,
                           {{"field_id", "drive_animation"},
                            {"serialized_name", "drive_animation"},
                            {"serialized", true},
                            {"hidden", false},
                            {"nullable", false},
                            {"storage_kind", "native_property"},
                            {"default", false}}});
    return type;
}

Ragdoll::~Ragdoll()
{
    Release();
}
void Ragdoll::OnHierarchyPublished()
{
    if (!ResolveOwnedComponent(m_skeletonComponentId))
        m_skeletonComponentId = 0;
    m_generatedComponentIds.erase(std::remove_if(m_generatedComponentIds.begin(), m_generatedComponentIds.end(),
                                                 [&](auto id) { return !ResolveOwnedComponent(id); }),
                                  m_generatedComponentIds.end());
    if (auto *skeleton = GetSkeleton(); skeleton && skeleton->HasBoundHierarchy())
        OnSkeletonHierarchyReady(*skeleton);
}
void Ragdoll::Awake()
{
    // A newly inserted nested root takes ownership before its first fixed step.
    for (auto *parent = GetGameObject()->GetParent(); parent; parent = parent->GetParent())
        if (auto *outer = parent->GetComponent<Ragdoll>()) {
            outer->Rebuild();
            break;
        }
    Rebuild();
}
void Ragdoll::OnEnable()
{
    Rebuild();
}
void Ragdoll::OnDisable()
{
    Release();
}
void Ragdoll::OnDestroy()
{
    Release();
}
void Ragdoll::OnGameObjectDeactivated()
{
    Release();
}

bool Ragdoll::IsBound() const
{
    return PhysicsWorld::Instance().HasRagdollAssembly(m_assemblyId);
}

std::vector<Rigidbody *> Ragdoll::GetBodies() const
{
    return GetConfiguredBodies();
}

size_t Ragdoll::GetConstraintCount() const
{
    size_t count = 0;
    for (const auto &reference : m_joints)
        if (reference.GetComponent())
            ++count;
    return count;
}

void Ragdoll::Release()
{
    PhysicsWorld::Instance().DestroyRagdollAssembly(m_assemblyId);
    m_assemblyId = 0;
}

std::vector<Rigidbody *> Ragdoll::GetConfiguredBodies() const
{
    auto *root = GetGameObject();
    std::vector<Rigidbody *> bodies;
    if (!root)
        return bodies;
    const auto collect = [&](auto &&self, GameObject *object) -> void {
        if (object != root && object->GetComponent<Ragdoll>())
            return;
        if (auto *body = object->GetComponent<Rigidbody>())
            bodies.push_back(body);
        for (size_t index = 0; index < object->GetChildCount(); ++index)
            self(self, object->GetChild(index));
    };
    if (auto *skeleton = GetSkeleton()) {
        for (auto *bone : skeleton->GetBones())
            if (OwnsObject(bone->GetGameObject()))
                if (auto *body = bone->GetGameObject()->GetComponent<Rigidbody>())
                    bodies.push_back(body);
    } else if (!m_generatedComponentIds.empty()) {
        for (auto id : m_generatedComponentIds)
            if (auto *body = dynamic_cast<Rigidbody *>(ResolveOwnedComponent(id)))
                bodies.push_back(body);
    } else
        collect(collect, root);
    const std::unordered_set<Rigidbody *> members(bodies.begin(), bodies.end());
    std::unordered_map<Rigidbody *, bool> visited;
    std::vector<Rigidbody *> ordered;
    const auto append = [&](auto &&self, Rigidbody *body) -> void {
        if (const auto found = visited.find(body); found != visited.end()) {
            if (!found->second)
                throw std::invalid_argument("ragdoll parent joints must form an acyclic skeleton");
            return;
        }
        visited.emplace(body, false);
        if (auto *joint = body->GetGameObject()->GetComponent<Joint>(); joint && joint->IsEnabled()) {
            auto *parent = joint->GetConnectedBody();
            if (!parent || !members.count(parent))
                throw std::invalid_argument("ragdoll parent joint must connect a Rigidbody inside this root");
            self(self, parent);
        }
        visited.at(body) = true;
        ordered.push_back(body);
    };
    for (auto *body : bodies)
        append(append, body);
    return ordered;
}

bool Ragdoll::Rebuild()
{
    auto *root = GetGameObject();
    if (!root || !root->GetScene() || !root->IsActiveInHierarchy() || !IsEnabled()) {
        Release();
        return false;
    }
    const auto configured = GetConfiguredBodies();
    std::vector<Rigidbody *> bodies;
    m_poseInputCount = configured.size();
    m_activePoseInputs.clear();
    for (size_t index = 0; index < configured.size(); ++index) {
        auto *body = configured[index];
        if (body->IsEnabled() && body->GetGameObject()->IsActiveInHierarchy()) {
            bodies.push_back(body);
            m_activePoseInputs.push_back(index);
        }
    }
    std::unordered_map<Rigidbody *, int> indices;
    for (auto *body : bodies)
        indices.emplace(body, -1);
    std::vector<Rigidbody *> ordered;
    std::vector<Joint *> joints;
    std::vector<int> parents;
    bool ready = true;
    for (auto *body : bodies) {
        auto *joint = body->GetGameObject()->GetComponent<Joint>();
        if (joint && !joint->IsEnabled())
            joint = nullptr;
        auto *parent = joint ? joint->GetConnectedBody() : nullptr;
        if (parent && !indices.count(parent)) {
            ready = false; // An internal endpoint is temporarily disabled.
            break;
        }
        indices.at(body) = static_cast<int>(ordered.size());
        ordered.push_back(body);
        joints.push_back(joint);
        parents.push_back(parent ? indices.at(parent) : -1);
    }
    m_structureVersion = root->GetScene()->GetStructureVersion();
    if (!ready) {
        Release();
        m_bodies.clear();
        m_joints.clear();
        m_parents.clear();
        return false;
    }

    bool unchanged = IsBound() && m_bodies.size() == ordered.size() && m_parents == parents;
    for (size_t index = 0; unchanged && index < ordered.size(); ++index)
        unchanged = m_bodies[index].GetComponent() == ordered[index] && m_joints[index].GetComponent() == joints[index];
    if (unchanged)
        return true;
    Release();
    m_bodies.clear();
    m_joints.clear();
    m_parents = std::move(parents);
    for (size_t index = 0; index < ordered.size(); ++index) {
        m_bodies.emplace_back(ordered[index]);
        m_joints.emplace_back(joints[index]);
    }
    m_assemblyId = PhysicsWorld::Instance().CreateRagdollAssembly(ordered, joints, m_parents);
    return IsBound();
}

void Ragdoll::FixedUpdate(float)
{
    auto *root = GetGameObject();
    if (!IsBound() || m_structureVersion != root->GetScene()->GetStructureVersion())
        Rebuild();
    if (m_driveAnimation)
        UpdateAnimationDrive();
}

void Ragdoll::DrivePose(const std::vector<glm::quat> &worldRotations)
{
    if ((!IsBound() || m_structureVersion != GetGameObject()->GetScene()->GetStructureVersion()) && !Rebuild())
        throw std::runtime_error("ragdoll bodies or parent joints have not been published");
    if (worldRotations.size() != m_poseInputCount)
        throw std::invalid_argument("ragdoll pose must match its public bodies array");
    std::vector<glm::quat> active;
    active.reserve(m_activePoseInputs.size());
    for (auto index : m_activePoseInputs)
        active.push_back(worldRotations[index]);
    PhysicsWorld::Instance().DriveRagdollPose(m_assemblyId, active);
}

nlohmann::json Ragdoll::SerializeDocument() const
{
    auto document = Component::SerializeDocument();
    document["skeleton_component_id"] = m_skeletonComponentId;
    document["drive_animation"] = m_driveAnimation;
    document["generated_component_ids"] = nlohmann::json::array();
    for (auto id : m_generatedComponentIds)
        if (ResolveOwnedComponent(id))
            document["generated_component_ids"].push_back(id);
    return document;
}
void Ragdoll::ValidateSerializedDocument(const nlohmann::json &document)
{
    using namespace component_document_validation;
    ValidateComponentDocument(document, "Ragdoll", {},
                              {"skeleton_component_id", "generated_component_ids", "drive_animation"});
    if (document.contains("drive_animation"))
        RequireBoolean(document, "drive_animation", "Ragdoll");
    if (document.contains("skeleton_component_id"))
        RequireUnsignedInteger(document, "skeleton_component_id", "Ragdoll");
    if (const auto values = document.find("generated_component_ids"); values != document.end()) {
        if (!values->is_array())
            throw std::invalid_argument("Ragdoll generated component references must be an array");
        std::unordered_set<uint64_t> unique;
        for (const auto &value : *values) {
            const auto id = RequireUnsignedInteger({{"id", value}}, "id", "Ragdoll");
            if (id == 0 || !unique.insert(id).second)
                throw std::invalid_argument("Ragdoll generated component references must be nonzero and unique");
        }
    }
}
bool Ragdoll::DeserializeDocument(const nlohmann::json &document)
{
    ValidateSerializedDocument(document);
    Release();
    if (!Component::DeserializeDocument(document))
        return false;
    m_skeletonComponentId = document.value("skeleton_component_id", uint64_t{0});
    m_driveAnimation = document.value("drive_animation", false);
    m_generatedComponentIds = document.value("generated_component_ids", std::vector<uint64_t>{});
    return true;
}
std::unique_ptr<Component> Ragdoll::Clone() const
{
    auto copy = std::make_unique<Ragdoll>();
    copy->m_enabled = m_enabled;
    copy->m_executionOrder = m_executionOrder;
    copy->m_skeletonComponentId = m_skeletonComponentId;
    copy->m_driveAnimation = m_driveAnimation;
    copy->m_generatedComponentIds = m_generatedComponentIds;
    return copy;
}

nlohmann::json Ragdoll::PrepareAuthoringDocument(const nlohmann::json &document) const
{
    auto candidate = document;
    candidate["skeleton_component_id"] = m_skeletonComponentId;
    candidate["generated_component_ids"] = m_generatedComponentIds;
    return Component::PrepareAuthoringDocument(candidate);
}

void Ragdoll::RemapComponentReferences(const std::unordered_map<uint64_t, uint64_t> &remap)
{
    if (const auto skeleton = remap.find(m_skeletonComponentId); skeleton != remap.end())
        m_skeletonComponentId = skeleton->second;
    for (auto &id : m_generatedComponentIds)
        if (const auto found = remap.find(id); found != remap.end())
            id = found->second;
    Release();
}
} // namespace infernux
