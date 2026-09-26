#include "SliderJoint.h"

#include "Collider.h"
#include "ComponentDocumentValidation.h"
#include "ComponentFactory.h"
#include "GameObject.h"
#include "Rigidbody.h"
#include "Transform.h"
#include "physics/PhysicsWorld.h"
#include <core/log/InxLog.h>

#include <cmath>
#include <stdexcept>

namespace infernux
{
namespace
{
bool IsFinite(const glm::vec3 &value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

SemanticTypeDescriptor DescribeSliderJoint()
{
    SemanticTypeDescriptor type;
    type.typeGuid = "native:infernux.SliderJoint";
    type.readableId = "infernux.component.slider_joint";
    type.owner = "engine:native";
    type.origin = "native";
    type.displayName = "Slider Joint";
    type.runtimeProfiles = {"editor", "player", "headless"};
    const auto add = [&](const char *name, const char *kind, nlohmann::json initial) -> nlohmann::json & {
        type.fields.push_back({std::string("SliderJoint.") + name,
                               std::string("FieldType.") + kind,
                               false,
                               {{"field_id", name},
                                {"serialized_name", name},
                                {"serialized", true},
                                {"hidden", false},
                                {"nullable", false},
                                {"storage_kind", "native_property"},
                                {"display_name_key", std::string("slider_joint.") + name},
                                {"tooltip", std::string("slider_joint.tooltip.") + name},
                                {"default", std::move(initial)}}});
        return type.fields.back().attributes;
    };
    add("anchor", "VEC3", {0.0, 0.0, 0.0});
    add("axis", "VEC3", {0.0, 1.0, 0.0});
    auto &connected = add("connected_body", "COMPONENT", nullptr);
    connected["nullable"] = true;
    connected["component_type"] = "Rigidbody";
    connected["serialized_name"] = "connected_body_component_id";
    add("use_limits", "BOOL", true);
    add("minimum_distance", "FLOAT", -1.0);
    add("maximum_distance", "FLOAT", 1.0);
    add("enable_collision", "BOOL", false);
    return type;
}

const bool registeredSliderJoint = ComponentFactory::Register(
    "SliderJoint", [] { return std::make_unique<SliderJoint>(); }, SliderJoint::ValidateSerializedDocument,
    SliderJoint::GetTypeConstraints(), DescribeSliderJoint);
} // namespace

SliderJoint::~SliderJoint()
{
    ReleaseConstraint();
}
void SliderJoint::Awake()
{
    RebuildConstraint();
}
void SliderJoint::OnEnable()
{
    RebuildConstraint();
}
void SliderJoint::OnDisable()
{
    ReleaseConstraint();
}
void SliderJoint::OnGameObjectDeactivated()
{
    ReleaseConstraint();
}
void SliderJoint::OnDestroy()
{
    ReleaseConstraint();
}
void SliderJoint::FixedUpdate(float)
{
    RebuildConstraint();
}

uint32_t SliderJoint::ResolveBodyId(const Rigidbody *body) const
{
    if (!body || !body->IsEnabled() || !body->GetGameObject())
        return 0xFFFFFFFF;
    return Collider::GetSharedBodyId(body->GetGameObject());
}

Rigidbody *SliderJoint::GetConnectedBody() const
{
    if (m_connectedBodyComponentId == 0)
        return nullptr;
    return dynamic_cast<Rigidbody *>(Component::FindByComponentId(m_connectedBodyComponentId));
}

void SliderJoint::RebuildConstraint()
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
    m_constraintId = world.CreateSliderConstraint(bodyId, connectedBodyId, worldAnchor, worldAxis, m_useLimits,
                                                  m_minimumDistance, m_maximumDistance, m_enableCollision);
    m_boundBodyId = bodyId;
    m_boundConnectedBodyId = connectedBodyId;
}

void SliderJoint::ReleaseConstraint()
{
    if (m_constraintId != 0 && PhysicsWorld::Instance().IsInitialized())
        PhysicsWorld::Instance().DestroyConstraint(m_constraintId);
    m_constraintId = 0;
    m_boundBodyId = 0xFFFFFFFF;
    m_boundConnectedBodyId = 0xFFFFFFFF;
}

void SliderJoint::SetAnchor(const glm::vec3 &anchor)
{
    if (!IsFinite(anchor))
        throw std::invalid_argument("slider anchor must be finite");
    m_anchor = anchor;
    ReleaseConstraint();
    RebuildConstraint();
}

void SliderJoint::SetAxis(const glm::vec3 &axis)
{
    if (!IsFinite(axis) || glm::dot(axis, axis) <= 1e-12f)
        throw std::invalid_argument("slider axis must be finite and non-zero");
    m_axis = glm::normalize(axis);
    ReleaseConstraint();
    RebuildConstraint();
}

void SliderJoint::SetConnectedBody(Rigidbody *body)
{
    if (body && body->GetGameObject() == GetGameObject())
        throw std::invalid_argument("slider connected body must differ from its own Rigidbody");
    m_connectedBodyComponentId = body ? body->GetComponentID() : 0;
    ReleaseConstraint();
    RebuildConstraint();
}

void SliderJoint::SetUseLimits(bool value)
{
    m_useLimits = value;
    ReleaseConstraint();
    RebuildConstraint();
}

void SliderJoint::SetMinimumDistance(float metres)
{
    if (!std::isfinite(metres) || metres > 0.0f || metres > m_maximumDistance)
        throw std::invalid_argument("slider minimum distance must not exceed zero or maximum distance");
    m_minimumDistance = metres;
    ReleaseConstraint();
    RebuildConstraint();
}

void SliderJoint::SetMaximumDistance(float metres)
{
    if (!std::isfinite(metres) || metres < 0.0f || metres < m_minimumDistance)
        throw std::invalid_argument("slider maximum distance must not precede zero or minimum distance");
    m_maximumDistance = metres;
    ReleaseConstraint();
    RebuildConstraint();
}

void SliderJoint::SetEnableCollision(bool value)
{
    m_enableCollision = value;
    ReleaseConstraint();
    RebuildConstraint();
}

float SliderJoint::GetCurrentPosition() const
{
    return m_constraintId == 0 ? 0.0f : PhysicsWorld::Instance().GetSliderConstraintPosition(m_constraintId);
}

nlohmann::json SliderJoint::SerializeDocument() const
{
    auto document = Component::SerializeDocument();
    document["anchor"] = {m_anchor.x, m_anchor.y, m_anchor.z};
    document["axis"] = {m_axis.x, m_axis.y, m_axis.z};
    document["connected_body_component_id"] = m_connectedBodyComponentId;
    document["use_limits"] = m_useLimits;
    document["minimum_distance"] = m_minimumDistance;
    document["maximum_distance"] = m_maximumDistance;
    document["enable_collision"] = m_enableCollision;
    return document;
}

void SliderJoint::ValidateSerializedDocument(const nlohmann::json &document)
{
    using namespace component_document_validation;
    ValidateComponentDocument(document, "SliderJoint",
                              {"anchor", "axis", "connected_body_component_id", "use_limits", "minimum_distance",
                               "maximum_distance", "enable_collision"});
    RequireFiniteVector(document, "anchor", 3, "SliderJoint");
    RequireFiniteVector(document, "axis", 3, "SliderJoint");
    const auto &axis = document["axis"];
    const glm::vec3 axisValue(axis[0].get<float>(), axis[1].get<float>(), axis[2].get<float>());
    if (glm::dot(axisValue, axisValue) <= 1e-12f)
        throw std::invalid_argument("SliderJoint.axis must be non-zero");
    RequireUnsignedInteger(document, "connected_body_component_id", "SliderJoint");
    RequireBoolean(document, "use_limits", "SliderJoint");
    const float minimum = RequireFiniteFloat(document, "minimum_distance", "SliderJoint");
    const float maximum = RequireFiniteFloat(document, "maximum_distance", "SliderJoint");
    if (minimum > 0.0f || maximum < 0.0f || minimum > maximum)
        throw std::invalid_argument("SliderJoint distance limits must be ordered and span zero");
    RequireBoolean(document, "enable_collision", "SliderJoint");
}

bool SliderJoint::DeserializeDocument(const nlohmann::json &document)
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
        m_minimumDistance = document["minimum_distance"].get<float>();
        m_maximumDistance = document["maximum_distance"].get<float>();
        m_enableCollision = document["enable_collision"].get<bool>();
        ReleaseConstraint();
        RebuildConstraint();
        return true;
    } catch (const std::exception &error) {
        INXLOG_ERROR("SliderJoint::Deserialize failed: ", error.what());
        return false;
    }
}

std::unique_ptr<Component> SliderJoint::Clone() const
{
    auto clone = std::make_unique<SliderJoint>();
    clone->m_enabled = m_enabled;
    clone->m_executionOrder = m_executionOrder;
    clone->m_anchor = m_anchor;
    clone->m_axis = m_axis;
    clone->m_connectedBodyComponentId = m_connectedBodyComponentId;
    clone->m_useLimits = m_useLimits;
    clone->m_minimumDistance = m_minimumDistance;
    clone->m_maximumDistance = m_maximumDistance;
    clone->m_enableCollision = m_enableCollision;
    return clone;
}

void SliderJoint::RemapComponentReferences(const std::unordered_map<uint64_t, uint64_t> &componentIdRemap)
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
