#include "Joint.h"
#include "ComponentDocumentValidation.h"
#include "ComponentFactory.h"
#include "GameObject.h"
#include "Rigidbody.h"
#include "physics/PhysicsECSStore.h"
#include "physics/PhysicsWorld.h"
#include <cmath>
#include <stdexcept>

namespace infernux
{
nlohmann::json &Joint::AddJointField(SemanticTypeDescriptor &type, const char *name, const char *kind,
                                     nlohmann::json initial)
{
    const auto shortName = type.typeGuid.substr(type.typeGuid.rfind('.') + 1);
    type.fields.push_back({shortName + "." + name,
                           std::string("FieldType.") + kind,
                           false,
                           {{"field_id", name},
                            {"serialized_name", name},
                            {"serialized", true},
                            {"hidden", false},
                            {"nullable", false},
                            {"storage_kind", "native_property"},
                            {"default", std::move(initial)}}});
    return type.fields.back().attributes;
}

void Joint::AddJointEnum(SemanticTypeDescriptor &type, const char *field, const char *enumName,
                         std::initializer_list<const char *> names, const char *initial)
{
    auto &attributes =
        AddJointField(type, field, "ENUM", {{"$type", "enum"}, {"enum_type", enumName}, {"name", initial}});
    auto members = nlohmann::json::array();
    auto labels = nlohmann::json::array();
    int value = 0;
    for (const auto name : names) {
        members.push_back({{"name", name}, {"value", value++}});
        labels.push_back(name);
    }
    attributes["enum"] = {
        {"type_id", std::string("native:infernux.") + enumName}, {"members", members}, {"labels", labels}};
}

SemanticTypeDescriptor Joint::DescribeJointType(const char *name, const char *readableId, const char *displayName)
{
    SemanticTypeDescriptor type;
    type.typeGuid = std::string("native:infernux.") + name;
    type.readableId = readableId;
    type.owner = "engine:native";
    type.origin = "native";
    type.displayName = displayName;
    type.runtimeProfiles = {"editor", "player", "headless"};
    AddJointField(type, "anchor", "VEC3", {0.0, 0.0, 0.0});
    auto &connected = AddJointField(type, "connected_body", "COMPONENT", nullptr);
    connected["nullable"] = true;
    connected["component_type"] = "Rigidbody";
    connected["serialized_name"] = "connected_body_component_id";
    connected["document_type"] = "FieldType.UINT";
    AddJointField(type, "enable_collision", "BOOL", false);
    return type;
}

Joint::~Joint()
{
    ReleaseConstraint();
}
void Joint::Awake()
{
    RebuildConstraint();
}
void Joint::OnHierarchyPublished()
{
    RebuildConstraint();
}
void Joint::OnEnable()
{
    RebuildConstraint();
}
void Joint::OnDisable()
{
    ReleaseConstraint();
}
void Joint::OnGameObjectDeactivated()
{
    ReleaseConstraint();
}
void Joint::OnDestroy()
{
    ReleaseConstraint();
}
uint32_t Joint::ResolveBodyId(const Rigidbody *body)
{
    return body && body->IsEnabled() && body->GetGameObject() ? body->GetParticipatingBodyId() : 0xFFFFFFFF;
}

Rigidbody *Joint::GetConnectedBody() const
{
    return m_connectedBodyComponentId
               ? dynamic_cast<Rigidbody *>(Component::FindByComponentId(m_connectedBodyComponentId))
               : nullptr;
}

void Joint::RebuildConstraint()
{
    auto &world = PhysicsWorld::Instance();
    auto *object = GetGameObject();
    if (!IsEnabled() || IsBeingDestroyed() || IsDestroyed() || !object || !object->IsActiveInHierarchy() ||
        !world.IsInitialized()) {
        ReleaseConstraint();
        return;
    }
    const auto bodyId = ResolveBodyId(object->GetComponent<Rigidbody>());
    const auto connectedBodyId = ResolveBodyId(GetConnectedBody());
    auto &store = PhysicsECSStore::Instance();
    if (bodyId == 0xFFFFFFFF || (m_connectedBodyComponentId && connectedBodyId == 0xFFFFFFFF)) {
        // Leave an existing pending node in place: a failed attempt must not
        // schedule itself again in the absence of another readiness event.
        if (m_constraintId)
            ReleaseConstraint();
        m_retryOnRetire = true;
        store.QueuePendingJoint(*this);
        return;
    }
    if (m_constraintId && bodyId == m_boundBodyId && connectedBodyId == m_boundConnectedBodyId) {
        store.CancelPendingJoint(*this);
        return;
    }
    ReleaseConstraint();
    m_retryOnRetire = true;
    m_constraintId = CreateConstraint(world, bodyId, connectedBodyId);
    m_boundBodyId = bodyId;
    m_boundConnectedBodyId = connectedBodyId;
    OnConstraintCreated();
}

void Joint::ReleaseConstraint()
{
    m_retryOnRetire = false;
    PhysicsECSStore::Instance().CancelPendingJoint(*this);
    if (m_constraintId && PhysicsWorld::Instance().IsInitialized())
        PhysicsWorld::Instance().DestroyConstraint(m_constraintId);
    OnPhysicsConstraintDestroyed();
}

void Joint::OnPhysicsConstraintDestroyed() noexcept
{
    m_constraintId = 0;
    m_boundBodyId = m_boundConnectedBodyId = 0xFFFFFFFF;
    if (m_retryOnRetire)
        PhysicsECSStore::Instance().QueuePendingJoint(*this);
}

void Joint::SetAnchor(const glm::vec3 &anchor)
{
    if (!std::isfinite(anchor.x) || !std::isfinite(anchor.y) || !std::isfinite(anchor.z))
        throw std::invalid_argument("Joint.anchor must be finite");
    if (anchor == m_anchor && m_constraintId)
        return;
    m_anchor = anchor;
    ReleaseConstraint();
    RebuildConstraint();
}

void Joint::SetConnectedBody(Rigidbody *body)
{
    if (body && GetGameObject() && body->GetGameObject() == GetGameObject())
        throw std::invalid_argument("Joint.connected_body must differ from its own Rigidbody");
    const auto id = body ? body->GetComponentID() : 0;
    if (id == m_connectedBodyComponentId && m_constraintId)
        return;
    m_connectedBodyComponentId = id;
    ReleaseConstraint();
    RebuildConstraint();
}

void Joint::SetEnableCollision(bool enabled)
{
    if (enabled == m_enableCollision && m_constraintId)
        return;
    m_enableCollision = enabled;
    UpdateConstraintSettings();
}

nlohmann::json Joint::SerializeDocument() const
{
    auto document = Component::SerializeDocument();
    document["anchor"] = {m_anchor.x, m_anchor.y, m_anchor.z};
    document["connected_body_component_id"] = m_connectedBodyComponentId;
    document["enable_collision"] = m_enableCollision;
    return document;
}

void Joint::ValidateJointDocumentFields(const nlohmann::json &document, const char *typeName)
{
    using namespace component_document_validation;
    RequireFiniteVector(document, "anchor", 3, typeName);
    RequireUnsignedInteger(document, "connected_body_component_id", typeName);
    RequireBoolean(document, "enable_collision", typeName);
}

void Joint::ReadJointDocumentFields(const nlohmann::json &document)
{
    const auto &anchor = document.at("anchor");
    m_anchor = {anchor[0].get<float>(), anchor[1].get<float>(), anchor[2].get<float>()};
    m_connectedBodyComponentId = document.at("connected_body_component_id").get<uint64_t>();
    m_enableCollision = document.at("enable_collision").get<bool>();
}

void Joint::CopyJointFieldsTo(Joint &destination) const
{
    destination.m_enabled = m_enabled;
    destination.m_executionOrder = m_executionOrder;
    destination.m_anchor = m_anchor;
    destination.m_connectedBodyComponentId = m_connectedBodyComponentId;
    destination.m_enableCollision = m_enableCollision;
}

void Joint::RemapComponentReferences(const std::unordered_map<uint64_t, uint64_t> &remap)
{
    if (!m_connectedBodyComponentId)
        return;
    const auto found = remap.find(m_connectedBodyComponentId);
    if (found == remap.end())
        return;
    m_connectedBodyComponentId = found->second;
    ReleaseConstraint();
    RebuildConstraint();
}
} // namespace infernux
