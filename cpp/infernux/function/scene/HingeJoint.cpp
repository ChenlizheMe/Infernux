#include "HingeJoint.h"

#include "Collider.h"
#include "ComponentDocumentValidation.h"
#include "ComponentFactory.h"
#include "GameObject.h"
#include "Rigidbody.h"
#include "Transform.h"
#include "physics/PhysicsWorld.h"
#include <core/log/InxLog.h>

#include <cmath>
#include <glm/gtc/constants.hpp>
#include <stdexcept>

namespace infernux
{
namespace
{
bool IsFinite(const glm::vec3 &value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

SemanticTypeDescriptor DescribeHingeJoint()
{
    SemanticTypeDescriptor type;
    type.typeGuid = "native:infernux.HingeJoint";
    type.readableId = "infernux.component.hinge_joint";
    type.owner = "engine:native";
    type.origin = "native";
    type.displayName = "Hinge Joint";
    type.runtimeProfiles = {"editor", "player", "headless"};
    const auto add = [&](const char *name, const char *kind, nlohmann::json initial) -> nlohmann::json & {
        type.fields.push_back({std::string("HingeJoint.") + name,
                               std::string("FieldType.") + kind,
                               false,
                               {{"field_id", name},
                                {"serialized_name", name},
                                {"serialized", true},
                                {"hidden", false},
                                {"nullable", false},
                                {"storage_kind", "native_property"},
                                {"display_name_key", std::string("hinge_joint.") + name},
                                {"tooltip", std::string("hinge_joint.tooltip.") + name},
                                {"default", std::move(initial)}}});
        return type.fields.back().attributes;
    };
    add("anchor", "VEC3", {0.0, 0.0, 0.0});
    add("axis", "VEC3", {1.0, 0.0, 0.0});
    auto &connected = add("connected_body", "COMPONENT", nullptr);
    connected["nullable"] = true;
    connected["component_type"] = "Rigidbody";
    connected["serialized_name"] = "connected_body_component_id";
    add("use_limits", "BOOL", false);
    add("minimum_angle", "FLOAT", -90.0)["range"] = {-180.0, 0.0};
    add("maximum_angle", "FLOAT", 90.0)["range"] = {0.0, 180.0};
    add("enable_collision", "BOOL", false);
    return type;
}
} // namespace

namespace
{
const bool registeredHingeJoint = ComponentFactory::Register(
    "HingeJoint", [] { return std::make_unique<HingeJoint>(); }, HingeJoint::ValidateSerializedDocument,
    HingeJoint::GetTypeConstraints(), DescribeHingeJoint);
} // namespace

HingeJoint::~HingeJoint()
{
    ReleaseConstraint();
}

void HingeJoint::Awake()
{
    RebuildConstraint();
}
void HingeJoint::OnEnable()
{
    RebuildConstraint();
}
void HingeJoint::OnDisable()
{
    ReleaseConstraint();
}
void HingeJoint::OnGameObjectDeactivated()
{
    ReleaseConstraint();
}
void HingeJoint::OnDestroy()
{
    ReleaseConstraint();
}

void HingeJoint::FixedUpdate(float)
{
    RebuildConstraint();
}

uint32_t HingeJoint::ResolveBodyId(const Rigidbody *body) const
{
    if (!body || !body->IsEnabled() || !body->GetGameObject())
        return 0xFFFFFFFF;
    return Collider::GetSharedBodyId(body->GetGameObject());
}

Rigidbody *HingeJoint::GetConnectedBody() const
{
    if (m_connectedBodyComponentId == 0)
        return nullptr;
    return dynamic_cast<Rigidbody *>(Component::FindByComponentId(m_connectedBodyComponentId));
}

void HingeJoint::RebuildConstraint()
{
    auto &world = PhysicsWorld::Instance();
    auto *gameObject = GetGameObject();
    if (!IsEnabled() || !gameObject || !gameObject->IsActiveInHierarchy() || !world.IsInitialized()) {
        ReleaseConstraint();
        return;
    }

    auto *body = gameObject->GetComponent<Rigidbody>();
    const uint32_t bodyId = ResolveBodyId(body);
    Rigidbody *connectedBody = GetConnectedBody();
    const uint32_t connectedBodyId = connectedBody ? ResolveBodyId(connectedBody) : 0xFFFFFFFF;
    if (bodyId == 0xFFFFFFFF || (m_connectedBodyComponentId != 0 && connectedBodyId == 0xFFFFFFFF)) {
        ReleaseConstraint();
        return;
    }
    if (m_constraintId != 0 && bodyId == m_boundBodyId && connectedBodyId == m_boundConnectedBodyId)
        return;

    ReleaseConstraint();
    auto *transform = gameObject->GetTransform();
    const glm::vec3 worldAnchor = transform->TransformPoint(m_anchor);
    const glm::vec3 worldAxis = transform->TransformDirection(m_axis);
    m_constraintId =
        world.CreateHingeConstraint(bodyId, connectedBodyId, worldAnchor, worldAxis, m_useLimits,
                                    glm::radians(m_minimumAngle), glm::radians(m_maximumAngle), m_enableCollision);
    m_boundBodyId = bodyId;
    m_boundConnectedBodyId = connectedBodyId;
}

void HingeJoint::ReleaseConstraint()
{
    if (m_constraintId != 0 && PhysicsWorld::Instance().IsInitialized())
        PhysicsWorld::Instance().DestroyConstraint(m_constraintId);
    m_constraintId = 0;
    m_boundBodyId = 0xFFFFFFFF;
    m_boundConnectedBodyId = 0xFFFFFFFF;
}

void HingeJoint::SetAnchor(const glm::vec3 &anchor)
{
    if (!IsFinite(anchor))
        throw std::invalid_argument("hinge anchor must be finite");
    m_anchor = anchor;
    ReleaseConstraint();
    RebuildConstraint();
}

void HingeJoint::SetAxis(const glm::vec3 &axis)
{
    if (!IsFinite(axis) || glm::dot(axis, axis) <= 1e-12f)
        throw std::invalid_argument("hinge axis must be finite and non-zero");
    m_axis = glm::normalize(axis);
    ReleaseConstraint();
    RebuildConstraint();
}

void HingeJoint::SetConnectedBody(Rigidbody *body)
{
    if (body && body->GetGameObject() == GetGameObject())
        throw std::invalid_argument("hinge connected body must differ from its own Rigidbody");
    m_connectedBodyComponentId = body ? body->GetComponentID() : 0;
    ReleaseConstraint();
    RebuildConstraint();
}

void HingeJoint::SetUseLimits(bool value)
{
    m_useLimits = value;
    ReleaseConstraint();
    RebuildConstraint();
}

void HingeJoint::SetMinimumAngle(float degrees)
{
    if (!std::isfinite(degrees) || degrees < -180.0f || degrees > 0.0f || degrees > m_maximumAngle)
        throw std::invalid_argument("hinge minimum angle must be within [-180, 0] and not exceed maximum");
    m_minimumAngle = degrees;
    ReleaseConstraint();
    RebuildConstraint();
}

void HingeJoint::SetMaximumAngle(float degrees)
{
    if (!std::isfinite(degrees) || degrees < 0.0f || degrees > 180.0f || degrees < m_minimumAngle)
        throw std::invalid_argument("hinge maximum angle must be within [0, 180] and not precede minimum");
    m_maximumAngle = degrees;
    ReleaseConstraint();
    RebuildConstraint();
}

void HingeJoint::SetEnableCollision(bool value)
{
    m_enableCollision = value;
    ReleaseConstraint();
    RebuildConstraint();
}

float HingeJoint::GetCurrentAngle() const
{
    return m_constraintId == 0 ? 0.0f : glm::degrees(PhysicsWorld::Instance().GetHingeConstraintAngle(m_constraintId));
}

nlohmann::json HingeJoint::SerializeDocument() const
{
    auto document = Component::SerializeDocument();
    document["anchor"] = {m_anchor.x, m_anchor.y, m_anchor.z};
    document["axis"] = {m_axis.x, m_axis.y, m_axis.z};
    document["connected_body_component_id"] = m_connectedBodyComponentId;
    document["use_limits"] = m_useLimits;
    document["minimum_angle"] = m_minimumAngle;
    document["maximum_angle"] = m_maximumAngle;
    document["enable_collision"] = m_enableCollision;
    return document;
}

void HingeJoint::ValidateSerializedDocument(const nlohmann::json &document)
{
    using namespace component_document_validation;
    ValidateComponentDocument(document, "HingeJoint",
                              {"anchor", "axis", "connected_body_component_id", "use_limits", "minimum_angle",
                               "maximum_angle", "enable_collision"});
    RequireFiniteVector(document, "anchor", 3, "HingeJoint");
    RequireFiniteVector(document, "axis", 3, "HingeJoint");
    const auto &axis = document["axis"];
    const glm::vec3 axisValue(axis[0].get<float>(), axis[1].get<float>(), axis[2].get<float>());
    if (glm::dot(axisValue, axisValue) <= 1e-12f)
        throw std::invalid_argument("HingeJoint.axis must be non-zero");
    RequireUnsignedInteger(document, "connected_body_component_id", "HingeJoint");
    RequireBoolean(document, "use_limits", "HingeJoint");
    const float minimum = RequireFiniteFloat(document, "minimum_angle", "HingeJoint");
    const float maximum = RequireFiniteFloat(document, "maximum_angle", "HingeJoint");
    if (minimum < -180.0f || minimum > 0.0f || maximum < 0.0f || maximum > 180.0f || minimum > maximum)
        throw std::invalid_argument("HingeJoint angle limits must be ordered, span zero and stay within [-180, 180]");
    RequireBoolean(document, "enable_collision", "HingeJoint");
}

bool HingeJoint::DeserializeDocument(const nlohmann::json &document)
{
    try {
        ValidateSerializedDocument(document);
        const auto &anchor = document["anchor"];
        const auto &axis = document["axis"];
        if (!Component::DeserializeDocument(document))
            return false;
        m_anchor = {anchor[0].get<float>(), anchor[1].get<float>(), anchor[2].get<float>()};
        m_axis = glm::normalize(glm::vec3(axis[0].get<float>(), axis[1].get<float>(), axis[2].get<float>()));
        m_connectedBodyComponentId = document["connected_body_component_id"].get<uint64_t>();
        m_useLimits = document["use_limits"].get<bool>();
        m_minimumAngle = document["minimum_angle"].get<float>();
        m_maximumAngle = document["maximum_angle"].get<float>();
        m_enableCollision = document["enable_collision"].get<bool>();
        ReleaseConstraint();
        RebuildConstraint();
        return true;
    } catch (const std::exception &error) {
        INXLOG_ERROR("HingeJoint::Deserialize failed: ", error.what());
        return false;
    }
}

std::unique_ptr<Component> HingeJoint::Clone() const
{
    auto clone = std::make_unique<HingeJoint>();
    clone->m_enabled = m_enabled;
    clone->m_executionOrder = m_executionOrder;
    clone->m_anchor = m_anchor;
    clone->m_axis = m_axis;
    clone->m_connectedBodyComponentId = m_connectedBodyComponentId;
    clone->m_useLimits = m_useLimits;
    clone->m_minimumAngle = m_minimumAngle;
    clone->m_maximumAngle = m_maximumAngle;
    clone->m_enableCollision = m_enableCollision;
    return clone;
}

void HingeJoint::RemapComponentReferences(const std::unordered_map<uint64_t, uint64_t> &componentIdRemap)
{
    if (m_connectedBodyComponentId == 0)
        return;
    const auto remapped = componentIdRemap.find(m_connectedBodyComponentId);
    if (remapped == componentIdRemap.end())
        return;
    m_connectedBodyComponentId = remapped->second;
    ReleaseConstraint();
    RebuildConstraint();
}

} // namespace infernux
