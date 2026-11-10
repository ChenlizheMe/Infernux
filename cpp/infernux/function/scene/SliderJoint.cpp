#include "SliderJoint.h"

#include "ComponentDocumentValidation.h"
#include "ComponentFactory.h"
#include "GameObject.h"
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

uint64_t SliderJoint::CreateConstraint(PhysicsWorld &world, uint32_t bodyId, uint32_t connectedBodyId)
{
    const auto *transform = GetTransform();
    const auto worldAnchor = transform->TransformPoint(m_anchor);
    const auto worldAxis = transform->TransformDirection(m_axis);
    return world.CreateSliderConstraint(*this, bodyId, connectedBodyId, worldAnchor, worldAxis, m_anchor, m_useLimits,
                                        m_minimumDistance, m_maximumDistance, m_enableCollision);
}

void SliderJoint::SetAxis(const glm::vec3 &axis)
{
    if (!IsFinite(axis) || glm::dot(axis, axis) <= 1e-12f)
        throw std::invalid_argument("slider axis must be finite and non-zero");
    m_axis = glm::normalize(axis);
    ReleaseConstraint();
    RebuildConstraint();
}

void SliderJoint::SetUseLimits(bool value)
{
    m_useLimits = value;
    UpdateConstraintSettings();
}

void SliderJoint::SetMinimumDistance(float metres)
{
    if (!std::isfinite(metres) || metres > 0.0f || metres > m_maximumDistance)
        throw std::invalid_argument("slider minimum distance must not exceed zero or maximum distance");
    m_minimumDistance = metres;
    UpdateConstraintSettings();
}

void SliderJoint::SetMaximumDistance(float metres)
{
    if (!std::isfinite(metres) || metres < 0.0f || metres < m_minimumDistance)
        throw std::invalid_argument("slider maximum distance must not precede zero or minimum distance");
    m_maximumDistance = metres;
    UpdateConstraintSettings();
}

void SliderJoint::UpdateConstraintSettings()
{
    if (m_constraintId == 0) {
        RebuildConstraint();
        return;
    }
    PhysicsWorld::Instance().UpdateSliderConstraintSettings(m_constraintId, m_useLimits, m_minimumDistance,
                                                            m_maximumDistance, m_enableCollision);
}

float SliderJoint::GetCurrentPosition() const
{
    return m_constraintId == 0 ? 0.0f : PhysicsWorld::Instance().GetSliderConstraintPosition(m_constraintId);
}

nlohmann::json SliderJoint::SerializeDocument() const
{
    auto document = Joint::SerializeDocument();
    document["axis"] = {m_axis.x, m_axis.y, m_axis.z};
    document["use_limits"] = m_useLimits;
    document["minimum_distance"] = m_minimumDistance;
    document["maximum_distance"] = m_maximumDistance;
    return document;
}

void SliderJoint::ValidateSerializedDocument(const nlohmann::json &document)
{
    using namespace component_document_validation;
    ValidateComponentDocument(document, "SliderJoint",
                              {"anchor", "axis", "connected_body_component_id", "use_limits", "minimum_distance",
                               "maximum_distance", "enable_collision"});
    ValidateJointDocumentFields(document, "SliderJoint");
    RequireFiniteVector(document, "axis", 3, "SliderJoint");
    const auto &axis = document["axis"];
    const glm::vec3 axisValue(axis[0].get<float>(), axis[1].get<float>(), axis[2].get<float>());
    if (glm::dot(axisValue, axisValue) <= 1e-12f)
        throw std::invalid_argument("SliderJoint.axis must be non-zero");
    RequireBoolean(document, "use_limits", "SliderJoint");
    const float minimum = RequireFiniteFloat(document, "minimum_distance", "SliderJoint");
    const float maximum = RequireFiniteFloat(document, "maximum_distance", "SliderJoint");
    if (minimum > 0.0f || maximum < 0.0f || minimum > maximum)
        throw std::invalid_argument("SliderJoint distance limits must be ordered and span zero");
}

bool SliderJoint::DeserializeDocument(const nlohmann::json &document)
{
    try {
        ValidateSerializedDocument(document);
        const auto &axis = document["axis"];
        if (!Component::DeserializeDocument(document))
            return false;
        ReadJointDocumentFields(document);
        m_axis = glm::normalize(glm::vec3(axis[0].get<float>(), axis[1].get<float>(), axis[2].get<float>()));
        m_useLimits = document["use_limits"].get<bool>();
        m_minimumDistance = document["minimum_distance"].get<float>();
        m_maximumDistance = document["maximum_distance"].get<float>();
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
    CopyJointFieldsTo(*clone);
    clone->m_axis = m_axis;
    clone->m_useLimits = m_useLimits;
    clone->m_minimumDistance = m_minimumDistance;
    clone->m_maximumDistance = m_maximumDistance;
    return clone;
}

} // namespace infernux
