#include "DistanceJoint.h"
#include "ComponentDocumentValidation.h"
#include "ComponentFactory.h"
#include "Rigidbody.h"
#include "Transform.h"
#include "physics/PhysicsWorld.h"
#include <cmath>
#include <stdexcept>

namespace infernux
{
namespace
{
const bool registered = ComponentFactory::Register(
    "DistanceJoint", [] { return std::make_unique<DistanceJoint>(); }, DistanceJoint::ValidateSerializedDocument,
    DistanceJoint::GetTypeConstraints(), DistanceJoint::DescribeSemanticType);

void RequireRange(float minimum, float maximum)
{
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum < 0 || maximum < minimum)
        throw std::invalid_argument("DistanceJoint limits must be finite ordered non-negative metres");
}

void RequireSpringValue(float value)
{
    if (!std::isfinite(value) || value < 0)
        throw std::invalid_argument("DistanceJoint spring coefficients must be finite and non-negative");
}
} // namespace

SemanticTypeDescriptor DistanceJoint::DescribeSemanticType()
{
    auto type = DescribeJointType("DistanceJoint", "infernux.component.distance_joint", "Distance Joint");
    AddJointField(type, "connected_anchor", "VEC3", {0.0, 0.0, 0.0})["tooltip"] =
        "Point in connected body local space, or world space when connected_body is empty";
    AddJointField(type, "auto_distance", "BOOL", true)["tooltip"] =
        "Capture the distance on constraint creation; changing spring coefficients preserves that length";
    AddJointField(type, "minimum_distance", "FLOAT", 0.0)["tooltip"] = "Minimum metres when auto_distance is false";
    AddJointField(type, "maximum_distance", "FLOAT", 1.0)["tooltip"] = "Maximum metres when auto_distance is false";
    AddJointField(type, "stiffness", "FLOAT", 0.0)["tooltip"] =
        "Spring stiffness in N/m; zero stiffness gives a hard constraint";
    AddJointField(type, "damping", "FLOAT", 0.0)["tooltip"] =
        "Spring damping in N s/m; used when stiffness is positive";
    return type;
}

uint64_t DistanceJoint::CreateConstraint(PhysicsWorld &world, uint32_t bodyId, uint32_t connectedBodyId)
{
    return world.CreateDistanceConstraint(*this, bodyId, connectedBodyId, m_anchor, m_connectedAnchor, m_autoDistance,
                                          m_minimumDistance, m_maximumDistance, m_stiffness, m_damping,
                                          m_enableCollision);
}

void DistanceJoint::UpdateConstraintSettings()
{
    if (!m_constraintId)
        RebuildConstraint();
    else
        PhysicsWorld::Instance().UpdateDistanceConstraintSettings(m_constraintId, !m_autoDistance, m_minimumDistance,
                                                                  m_maximumDistance, m_stiffness, m_damping,
                                                                  m_enableCollision);
}

void DistanceJoint::SetConnectedAnchor(const glm::vec3 &anchor)
{
    if (!std::isfinite(anchor.x) || !std::isfinite(anchor.y) || !std::isfinite(anchor.z))
        throw std::invalid_argument("DistanceJoint.connected_anchor must be finite");
    if (anchor == m_connectedAnchor && m_constraintId)
        return;
    m_connectedAnchor = anchor;
    ReleaseConstraint();
    RebuildConstraint();
}

void DistanceJoint::SetAutoDistance(bool enabled)
{
    if (enabled == m_autoDistance && m_constraintId)
        return;
    m_autoDistance = enabled;
    ReleaseConstraint();
    RebuildConstraint();
}

void DistanceJoint::SetLimits(float minimum, float maximum)
{
    RequireRange(minimum, maximum);
    m_minimumDistance = minimum;
    m_maximumDistance = maximum;
    UpdateConstraintSettings();
}

void DistanceJoint::SetStiffness(float value)
{
    RequireSpringValue(value);
    m_stiffness = value;
    UpdateConstraintSettings();
}

void DistanceJoint::SetDamping(float value)
{
    RequireSpringValue(value);
    m_damping = value;
    UpdateConstraintSettings();
}

float DistanceJoint::GetCurrentDistance() const
{
    return m_constraintId ? PhysicsWorld::Instance().GetDistanceConstraintLength(m_constraintId) : 0;
}

nlohmann::json DistanceJoint::SerializeDocument() const
{
    auto document = Joint::SerializeDocument();
    document["connected_anchor"] = {m_connectedAnchor.x, m_connectedAnchor.y, m_connectedAnchor.z};
    document["auto_distance"] = m_autoDistance;
    document["minimum_distance"] = m_minimumDistance;
    document["maximum_distance"] = m_maximumDistance;
    document["stiffness"] = m_stiffness;
    document["damping"] = m_damping;
    return document;
}

void DistanceJoint::ValidateSerializedDocument(const nlohmann::json &document)
{
    using namespace component_document_validation;
    ValidateComponentDocument(document, "DistanceJoint",
                              {"anchor", "connected_body_component_id", "enable_collision", "connected_anchor",
                               "auto_distance", "minimum_distance", "maximum_distance", "stiffness", "damping"});
    ValidateJointDocumentFields(document, "DistanceJoint");
    RequireFiniteVector(document, "connected_anchor", 3, "DistanceJoint");
    RequireBoolean(document, "auto_distance", "DistanceJoint");
    RequireRange(RequireFiniteFloat(document, "minimum_distance", "DistanceJoint"),
                 RequireFiniteFloat(document, "maximum_distance", "DistanceJoint"));
    RequireSpringValue(RequireFiniteFloat(document, "stiffness", "DistanceJoint"));
    RequireSpringValue(RequireFiniteFloat(document, "damping", "DistanceJoint"));
}

bool DistanceJoint::DeserializeDocument(const nlohmann::json &document)
{
    ValidateSerializedDocument(document);
    if (!Component::DeserializeDocument(document))
        return false;
    ReadJointDocumentFields(document);
    const auto &anchor = document.at("connected_anchor");
    m_connectedAnchor = {anchor[0].get<float>(), anchor[1].get<float>(), anchor[2].get<float>()};
    m_autoDistance = document.at("auto_distance").get<bool>();
    m_minimumDistance = document.at("minimum_distance").get<float>();
    m_maximumDistance = document.at("maximum_distance").get<float>();
    m_stiffness = document.at("stiffness").get<float>();
    m_damping = document.at("damping").get<float>();
    ReleaseConstraint();
    RebuildConstraint();
    return true;
}

std::unique_ptr<Component> DistanceJoint::Clone() const
{
    auto clone = std::make_unique<DistanceJoint>();
    CopyJointFieldsTo(*clone);
    clone->m_connectedAnchor = m_connectedAnchor;
    clone->m_autoDistance = m_autoDistance;
    clone->m_minimumDistance = m_minimumDistance;
    clone->m_maximumDistance = m_maximumDistance;
    clone->m_stiffness = m_stiffness;
    clone->m_damping = m_damping;
    return clone;
}
} // namespace infernux
