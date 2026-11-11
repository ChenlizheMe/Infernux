#include "HingeJoint.h"

#include "ComponentDocumentValidation.h"
#include "ComponentFactory.h"
#include "GameObject.h"
#include "Transform.h"
#include "physics/PhysicsWorld.h"
#include <core/log/InxLog.h>

#include <algorithm>
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
    add("motor_enabled", "BOOL", false);
    add("motor_target_angle", "FLOAT", 0.0);
    add("motor_stiffness", "FLOAT", 10.0);
    add("motor_damping", "FLOAT", 1.0);
    add("motor_max_torque", "FLOAT", 1.0);
    add("joint_damping", "FLOAT", 0.0);
    add("friction_torque", "FLOAT", 0.0);
    add("armature", "FLOAT", 0.0);
    return type;
}
} // namespace

namespace
{
const bool registeredHingeJoint = ComponentFactory::Register(
    "HingeJoint", [] { return std::make_unique<HingeJoint>(); }, HingeJoint::ValidateSerializedDocument,
    HingeJoint::GetTypeConstraints(), DescribeHingeJoint);
} // namespace

uint64_t HingeJoint::CreateConstraint(PhysicsWorld &world, uint32_t bodyId, uint32_t connectedBodyId)
{
    const auto *transform = GetTransform();
    const auto worldAnchor = transform->TransformPoint(m_anchor);
    const auto worldAxis = transform->TransformDirection(m_axis);
    return world.CreateHingeConstraint(*this, bodyId, connectedBodyId, worldAnchor, worldAxis, m_anchor, m_useLimits,
                                       glm::radians(m_minimumAngle), glm::radians(m_maximumAngle), m_enableCollision);
}

void HingeJoint::SetAxis(const glm::vec3 &axis)
{
    if (!IsFinite(axis) || glm::dot(axis, axis) <= 1e-12f)
        throw std::invalid_argument("hinge axis must be finite and non-zero");
    m_axis = glm::normalize(axis);
    ReleaseConstraint();
    RebuildConstraint();
}

void HingeJoint::SetUseLimits(bool value)
{
    m_useLimits = value;
    UpdateConstraintSettings();
}

void HingeJoint::SetMinimumAngle(float degrees)
{
    if (!std::isfinite(degrees) || degrees < -180.0f || degrees > 0.0f || degrees > m_maximumAngle)
        throw std::invalid_argument("hinge minimum angle must be within [-180, 0] and not exceed maximum");
    m_minimumAngle = degrees;
    UpdateConstraintSettings();
}

void HingeJoint::SetMaximumAngle(float degrees)
{
    if (!std::isfinite(degrees) || degrees < 0.0f || degrees > 180.0f || degrees < m_minimumAngle)
        throw std::invalid_argument("hinge maximum angle must be within [0, 180] and not precede minimum");
    m_maximumAngle = degrees;
    UpdateConstraintSettings();
}

void HingeJoint::UpdateConstraintSettings()
{
    if (m_constraintId == 0) {
        RebuildConstraint();
        return;
    }
    PhysicsWorld::Instance().UpdateHingeConstraintSettings(m_constraintId, m_useLimits, glm::radians(m_minimumAngle),
                                                           glm::radians(m_maximumAngle), m_enableCollision);
    // Re-clamp the authored target when limits change, without changing zero.
    UpdateConstraintMotor();
}

void HingeJoint::SetMotorEnabled(bool value)
{
    m_motorEnabled = value;
    UpdateConstraintMotor();
}

void HingeJoint::SetMotorTargetAngle(float degrees)
{
    if (!std::isfinite(degrees))
        throw std::invalid_argument("hinge motor target angle must be finite");
    m_motorTargetAngle = degrees;
    if (m_constraintId != 0)
        PhysicsWorld::Instance().UpdateHingeConstraintTarget(m_constraintId, GetClampedMotorTargetRadians());
}

void HingeJoint::SetMotorStiffness(float stiffness)
{
    if (!std::isfinite(stiffness) || stiffness < 0.0f)
        throw std::invalid_argument("hinge motor stiffness must be non-negative and finite (N m / rad)");
    m_motorStiffness = stiffness;
    UpdateConstraintMotor();
}

void HingeJoint::SetMotorDamping(float damping)
{
    if (!std::isfinite(damping) || damping < 0.0f)
        throw std::invalid_argument("hinge motor damping must be non-negative and finite (N m s / rad)");
    m_motorDamping = damping;
    UpdateConstraintMotor();
}

void HingeJoint::SetMotorMaxTorque(float torque)
{
    if (!std::isfinite(torque) || torque < 0.0f)
        throw std::invalid_argument("hinge motor torque limit must be non-negative and finite");
    m_motorMaxTorque = torque;
    UpdateConstraintMotor();
}

void HingeJoint::SetJointDamping(float damping)
{
    if (!std::isfinite(damping) || damping < 0.0f)
        throw std::invalid_argument("hinge joint damping must be non-negative and finite (N m s / rad)");
    m_jointDamping = damping;
    UpdateConstraintMotor();
}

void HingeJoint::SetFrictionTorque(float torque)
{
    if (!std::isfinite(torque) || torque < 0.0f)
        throw std::invalid_argument("hinge friction torque must be non-negative and finite (N m)");
    m_frictionTorque = torque;
    UpdateConstraintMotor();
}

void HingeJoint::SetArmature(float inertia)
{
    if (!std::isfinite(inertia) || inertia < 0.0f)
        throw std::invalid_argument("hinge armature must be non-negative and finite (kg m^2)");
    m_armature = inertia;
    UpdateConstraintMotor();
}

float HingeJoint::GetClampedMotorTargetRadians() const
{
    // The authored full-range limit still clamps, even when Jolt treats it as unlimited.
    return glm::radians(m_useLimits ? std::clamp(m_motorTargetAngle, m_minimumAngle, m_maximumAngle)
                                    : m_motorTargetAngle);
}

void HingeJoint::UpdateConstraintMotor()
{
    if (m_constraintId != 0) {
        PhysicsWorld::Instance().UpdateHingeConstraintMotor(
            m_constraintId, m_motorEnabled, GetClampedMotorTargetRadians(), m_motorStiffness, m_motorDamping,
            m_motorMaxTorque, m_jointDamping, m_frictionTorque, m_armature);
    }
}

float HingeJoint::GetCurrentAngle() const
{
    return m_constraintId == 0 ? 0.0f : glm::degrees(PhysicsWorld::Instance().GetHingeConstraintAngle(m_constraintId));
}

float HingeJoint::GetCurrentAngularVelocity() const
{
    return m_constraintId == 0
               ? 0.0f
               : glm::degrees(PhysicsWorld::Instance().GetHingeConstraintAngularVelocity(m_constraintId));
}

glm::vec2 HingeJoint::GetMotionState() const
{
    if (m_constraintId == 0)
        throw std::logic_error("hinge motion state requires an active constraint");
    return PhysicsWorld::Instance().GetHingeConstraintMotionState(m_constraintId);
}

nlohmann::json HingeJoint::SerializeDocument() const
{
    auto document = Joint::SerializeDocument();
    document["axis"] = {m_axis.x, m_axis.y, m_axis.z};
    document["use_limits"] = m_useLimits;
    document["minimum_angle"] = m_minimumAngle;
    document["maximum_angle"] = m_maximumAngle;
    document["motor_enabled"] = m_motorEnabled;
    document["motor_target_angle"] = m_motorTargetAngle;
    document["motor_stiffness"] = m_motorStiffness;
    document["motor_damping"] = m_motorDamping;
    document["motor_max_torque"] = m_motorMaxTorque;
    document["joint_damping"] = m_jointDamping;
    document["friction_torque"] = m_frictionTorque;
    document["armature"] = m_armature;
    return document;
}

void HingeJoint::ValidateSerializedDocument(const nlohmann::json &document)
{
    using namespace component_document_validation;
    ValidateComponentDocument(document, "HingeJoint",
                              {"anchor", "axis", "connected_body_component_id", "use_limits", "minimum_angle",
                               "maximum_angle", "enable_collision", "motor_enabled", "motor_target_angle",
                               "motor_stiffness", "motor_damping", "motor_max_torque", "joint_damping",
                               "friction_torque", "armature"});
    ValidateJointDocumentFields(document, "HingeJoint");
    RequireFiniteVector(document, "axis", 3, "HingeJoint");
    const auto &axis = document["axis"];
    const glm::vec3 axisValue(axis[0].get<float>(), axis[1].get<float>(), axis[2].get<float>());
    if (glm::dot(axisValue, axisValue) <= 1e-12f)
        throw std::invalid_argument("HingeJoint.axis must be non-zero");
    RequireBoolean(document, "use_limits", "HingeJoint");
    const float minimum = RequireFiniteFloat(document, "minimum_angle", "HingeJoint");
    const float maximum = RequireFiniteFloat(document, "maximum_angle", "HingeJoint");
    if (minimum < -180.0f || minimum > 0.0f || maximum < 0.0f || maximum > 180.0f || minimum > maximum)
        throw std::invalid_argument("HingeJoint angle limits must be ordered, span zero and stay within [-180, 180]");
    RequireBoolean(document, "motor_enabled", "HingeJoint");
    RequireFiniteFloat(document, "motor_target_angle", "HingeJoint");
    const float stiffness = RequireFiniteFloat(document, "motor_stiffness", "HingeJoint");
    const float damping = RequireFiniteFloat(document, "motor_damping", "HingeJoint");
    const float torque = RequireFiniteFloat(document, "motor_max_torque", "HingeJoint");
    const float passive = RequireFiniteFloat(document, "joint_damping", "HingeJoint");
    const float friction = RequireFiniteFloat(document, "friction_torque", "HingeJoint");
    const float armature = RequireFiniteFloat(document, "armature", "HingeJoint");
    if (armature < 0.0f || stiffness < 0.0f || damping < 0.0f || torque < 0.0f || passive < 0.0f || friction < 0.0f)
        throw std::invalid_argument("HingeJoint stiffness, damping and torque coefficients must be non-negative");
}

bool HingeJoint::DeserializeDocument(const nlohmann::json &document)
{
    try {
        ValidateSerializedDocument(document);
        const auto &axis = document["axis"];
        if (!Component::DeserializeDocument(document))
            return false;
        ReadJointDocumentFields(document);
        m_axis = glm::normalize(glm::vec3(axis[0].get<float>(), axis[1].get<float>(), axis[2].get<float>()));
        m_useLimits = document["use_limits"].get<bool>();
        m_minimumAngle = document["minimum_angle"].get<float>();
        m_maximumAngle = document["maximum_angle"].get<float>();
        m_motorEnabled = document["motor_enabled"].get<bool>();
        m_motorTargetAngle = document["motor_target_angle"].get<float>();
        m_motorStiffness = document["motor_stiffness"].get<float>();
        m_motorDamping = document["motor_damping"].get<float>();
        m_motorMaxTorque = document["motor_max_torque"].get<float>();
        m_jointDamping = document["joint_damping"].get<float>();
        m_frictionTorque = document["friction_torque"].get<float>();
        m_armature = document["armature"].get<float>();
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
    CopyJointFieldsTo(*clone);
    clone->m_axis = m_axis;
    clone->m_useLimits = m_useLimits;
    clone->m_minimumAngle = m_minimumAngle;
    clone->m_maximumAngle = m_maximumAngle;
    clone->m_motorEnabled = m_motorEnabled;
    clone->m_motorTargetAngle = m_motorTargetAngle;
    clone->m_motorStiffness = m_motorStiffness;
    clone->m_motorDamping = m_motorDamping;
    clone->m_motorMaxTorque = m_motorMaxTorque;
    clone->m_jointDamping = m_jointDamping;
    clone->m_frictionTorque = m_frictionTorque;
    clone->m_armature = m_armature;
    return clone;
}

} // namespace infernux
