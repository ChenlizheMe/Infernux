#include "SwingTwistJoint.h"
#include "ComponentDocumentValidation.h"
#include "ComponentFactory.h"
#include "Transform.h"
#include "TransformECSStore.h"
#include "physics/PhysicsWorld.h"
#include <cmath>
#include <stdexcept>

namespace infernux
{
namespace
{
const bool registered = ComponentFactory::Register(
    "SwingTwistJoint", [] { return std::make_unique<SwingTwistJoint>(); }, SwingTwistJoint::ValidateSerializedDocument,
    SwingTwistJoint::GetTypeConstraints(), SwingTwistJoint::DescribeSemanticType);

void Validate(const SwingTwistJointSettings &s)
{
    const auto finite = [](const auto &v) {
        for (int i = 0; i < v.length(); ++i)
            if (!std::isfinite(v[i]))
                return false;
        return true;
    };
    if (!finite(s.twistAxis) || !finite(s.planeAxis) || glm::dot(s.twistAxis, s.twistAxis) < 1e-12f ||
        glm::dot(s.planeAxis, s.planeAxis) < 1e-12f ||
        glm::dot(glm::cross(glm::normalize(s.twistAxis), glm::normalize(s.planeAxis)),
                 glm::cross(glm::normalize(s.twistAxis), glm::normalize(s.planeAxis))) < 1e-8f)
        throw std::invalid_argument("SwingTwistJoint frame axes must be finite, nonzero and nonparallel");
    if ((int)s.swingType < 0 || (int)s.swingType > 1 || (int)s.swingMotorMode < 0 || (int)s.swingMotorMode > 3 ||
        (int)s.twistMotorMode < 0 || (int)s.twistMotorMode > 3)
        throw std::invalid_argument("SwingTwistJoint has an invalid swing type or motor mode");
    if (!std::isfinite(s.normalHalfConeAngle) || !std::isfinite(s.planeHalfConeAngle) || s.normalHalfConeAngle < 0 ||
        s.normalHalfConeAngle > 180 || s.planeHalfConeAngle < 0 || s.planeHalfConeAngle > 180 ||
        !std::isfinite(s.minimumTwistAngle) || !std::isfinite(s.maximumTwistAngle) || s.minimumTwistAngle < -180 ||
        s.maximumTwistAngle > 180 || s.minimumTwistAngle > s.maximumTwistAngle)
        throw std::invalid_argument(
            "SwingTwistJoint requires ordered twist degrees in [-180,180] and swing half-angles in [0,180]");
    for (float v : {s.frictionTorque, s.swingMotorStiffness, s.swingMotorDamping, s.swingMotorMaxTorque,
                    s.twistMotorStiffness, s.twistMotorDamping, s.twistMotorMaxTorque})
        if (!std::isfinite(v) || v < 0)
            throw std::invalid_argument(
                "SwingTwistJoint drive and friction coefficients must be finite and non-negative SI values");
    if (!finite(s.targetOrientation) || glm::dot(s.targetOrientation, s.targetOrientation) < 1e-12f ||
        !finite(s.targetAngularVelocity))
        throw std::invalid_argument(
            "SwingTwistJoint requires a nonzero finite XYZW orientation and finite angular velocity");
}

SwingTwistJointSettings ReadSettings(const nlohmann::json &d)
{
    SwingTwistJointSettings s;
    s.twistAxis = {d.at("twist_axis").at(0).get<float>(), d.at("twist_axis").at(1).get<float>(),
                   d.at("twist_axis").at(2).get<float>()};
    s.planeAxis = {d.at("plane_axis").at(0).get<float>(), d.at("plane_axis").at(1).get<float>(),
                   d.at("plane_axis").at(2).get<float>()};
    s.swingType = static_cast<JointSwingType>(d.at("swing_type").get<int>());
    s.normalHalfConeAngle = d.at("normal_half_cone_angle").get<float>();
    s.planeHalfConeAngle = d.at("plane_half_cone_angle").get<float>();
    s.minimumTwistAngle = d.at("minimum_twist_angle").get<float>();
    s.maximumTwistAngle = d.at("maximum_twist_angle").get<float>();
    s.frictionTorque = d.at("friction_torque").get<float>();
    s.swingMotorMode = static_cast<JointMotorMode>(d.at("swing_motor_mode").get<int>());
    s.twistMotorMode = static_cast<JointMotorMode>(d.at("twist_motor_mode").get<int>());
    s.swingMotorStiffness = d.at("swing_motor_stiffness").get<float>();
    s.swingMotorDamping = d.at("swing_motor_damping").get<float>();
    s.swingMotorMaxTorque = d.at("swing_motor_max_torque").get<float>();
    s.twistMotorStiffness = d.at("twist_motor_stiffness").get<float>();
    s.twistMotorDamping = d.at("twist_motor_damping").get<float>();
    s.twistMotorMaxTorque = d.at("twist_motor_max_torque").get<float>();
    s.targetOrientation = {d.at("target_orientation").at(0).get<float>(), d.at("target_orientation").at(1).get<float>(),
                           d.at("target_orientation").at(2).get<float>(),
                           d.at("target_orientation").at(3).get<float>()};
    s.targetAngularVelocity = {d.at("target_angular_velocity").at(0).get<float>(),
                               d.at("target_angular_velocity").at(1).get<float>(),
                               d.at("target_angular_velocity").at(2).get<float>()};
    return s;
}
} // namespace

SemanticTypeDescriptor SwingTwistJoint::DescribeSemanticType()
{
    auto type = DescribeJointType("SwingTwistJoint", "infernux.component.swing_twist_joint", "Swing Twist Joint");
    AddJointField(type, "twist_axis", "VEC3", {1.0, 0.0, 0.0})["tooltip"] = "Local frame axis";
    AddJointField(type, "plane_axis", "VEC3", {0.0, 0.0, 1.0})["tooltip"] = "Local frame axis";
    AddJointEnum(type, "swing_type", "JointSwingType", {"Cone", "Pyramid"}, "Cone");
    AddJointField(type, "normal_half_cone_angle", "FLOAT", 45.0)["tooltip"] = "Degrees";
    AddJointField(type, "plane_half_cone_angle", "FLOAT", 45.0)["tooltip"] = "Degrees";
    AddJointField(type, "minimum_twist_angle", "FLOAT", -45.0)["tooltip"] = "Degrees";
    AddJointField(type, "maximum_twist_angle", "FLOAT", 45.0)["tooltip"] = "Degrees";
    AddJointField(type, "friction_torque", "FLOAT", 0.0)["tooltip"] = "N m";
    AddJointEnum(type, "swing_motor_mode", "JointMotorMode", {"Off", "Velocity", "Position", "PositionAndVelocity"},
                 "Off");
    AddJointEnum(type, "twist_motor_mode", "JointMotorMode", {"Off", "Velocity", "Position", "PositionAndVelocity"},
                 "Off");
    AddJointField(type, "swing_motor_stiffness", "FLOAT", 0.0)["tooltip"] = "N m/rad";
    AddJointField(type, "swing_motor_damping", "FLOAT", 0.0)["tooltip"] = "N m s/rad";
    AddJointField(type, "swing_motor_max_torque", "FLOAT", 10.0)["tooltip"] = "N m";
    AddJointField(type, "twist_motor_stiffness", "FLOAT", 0.0)["tooltip"] = "N m/rad";
    AddJointField(type, "twist_motor_damping", "FLOAT", 0.0)["tooltip"] = "N m s/rad";
    AddJointField(type, "twist_motor_max_torque", "FLOAT", 10.0)["tooltip"] = "N m";
    AddJointField(type, "target_orientation", "VEC4", {0.0, 0.0, 0.0, 1.0})["tooltip"] =
        "Unit XYZW quaternion in constraint space";
    AddJointField(type, "target_angular_velocity", "VEC3", {0.0, 0.0, 0.0})["tooltip"] =
        "Degrees/s in owner constraint space";
    return type;
}

void SwingTwistJoint::ApplySettings(SwingTwistJointSettings settings, bool rebuild)
{
    Validate(settings);
    if (settings.twistAxis != m_settings.twistAxis)
        settings.twistAxis = glm::normalize(settings.twistAxis);
    if (settings.twistAxis != m_settings.twistAxis || settings.planeAxis != m_settings.planeAxis)
        settings.planeAxis =
            glm::normalize(settings.planeAxis - settings.twistAxis * glm::dot(settings.planeAxis, settings.twistAxis));
    if (settings.targetOrientation != m_settings.targetOrientation)
        settings.targetOrientation = glm::normalize(settings.targetOrientation);
    rebuild = rebuild && (settings.twistAxis != m_settings.twistAxis || settings.planeAxis != m_settings.planeAxis);
    m_settings = settings;
    if (rebuild)
        ReleaseConstraint();
    UpdateConstraintSettings();
}

void SwingTwistJoint::SetFrame(const glm::vec3 &twistAxis, const glm::vec3 &planeAxis)
{
    if (twistAxis == m_settings.twistAxis && planeAxis == m_settings.planeAxis && m_constraintId)
        return;
    auto settings = m_settings;
    settings.twistAxis = twistAxis;
    settings.planeAxis = planeAxis;
    ApplySettings(settings, true);
}

void SwingTwistJoint::SetTwistLimits(float minimum, float maximum)
{
    auto settings = m_settings;
    settings.minimumTwistAngle = minimum;
    settings.maximumTwistAngle = maximum;
    ApplySettings(settings);
}

void SwingTwistJoint::SetTwistAxis(glm::vec3 value)
{
    if (value == m_settings.twistAxis && m_constraintId)
        return;
    auto settings = m_settings;
    settings.twistAxis = value;
    ApplySettings(settings, true);
}

void SwingTwistJoint::SetPlaneAxis(glm::vec3 value)
{
    if (value == m_settings.planeAxis && m_constraintId)
        return;
    auto settings = m_settings;
    settings.planeAxis = value;
    ApplySettings(settings, true);
}

void SwingTwistJoint::SetSwingType(JointSwingType value)
{
    if (value == m_settings.swingType && m_constraintId)
        return;
    auto settings = m_settings;
    settings.swingType = value;
    ApplySettings(settings);
}

void SwingTwistJoint::SetNormalHalfConeAngle(float value)
{
    auto settings = m_settings;
    settings.normalHalfConeAngle = value;
    ApplySettings(settings);
}

void SwingTwistJoint::SetPlaneHalfConeAngle(float value)
{
    auto settings = m_settings;
    settings.planeHalfConeAngle = value;
    ApplySettings(settings);
}

void SwingTwistJoint::SetMinimumTwistAngle(float value)
{
    auto settings = m_settings;
    settings.minimumTwistAngle = value;
    ApplySettings(settings);
}

void SwingTwistJoint::SetMaximumTwistAngle(float value)
{
    auto settings = m_settings;
    settings.maximumTwistAngle = value;
    ApplySettings(settings);
}

void SwingTwistJoint::SetFrictionTorque(float value)
{
    auto settings = m_settings;
    settings.frictionTorque = value;
    ApplySettings(settings);
}

void SwingTwistJoint::SetSwingMotorMode(JointMotorMode value)
{
    auto settings = m_settings;
    settings.swingMotorMode = value;
    ApplySettings(settings);
}

void SwingTwistJoint::SetTwistMotorMode(JointMotorMode value)
{
    auto settings = m_settings;
    settings.twistMotorMode = value;
    ApplySettings(settings);
}

void SwingTwistJoint::SetSwingMotorStiffness(float value)
{
    auto settings = m_settings;
    settings.swingMotorStiffness = value;
    ApplySettings(settings);
}

void SwingTwistJoint::SetSwingMotorDamping(float value)
{
    auto settings = m_settings;
    settings.swingMotorDamping = value;
    ApplySettings(settings);
}

void SwingTwistJoint::SetSwingMotorMaxTorque(float value)
{
    auto settings = m_settings;
    settings.swingMotorMaxTorque = value;
    ApplySettings(settings);
}

void SwingTwistJoint::SetTwistMotorStiffness(float value)
{
    auto settings = m_settings;
    settings.twistMotorStiffness = value;
    ApplySettings(settings);
}

void SwingTwistJoint::SetTwistMotorDamping(float value)
{
    auto settings = m_settings;
    settings.twistMotorDamping = value;
    ApplySettings(settings);
}

void SwingTwistJoint::SetTwistMotorMaxTorque(float value)
{
    auto settings = m_settings;
    settings.twistMotorMaxTorque = value;
    ApplySettings(settings);
}

void SwingTwistJoint::SetTargetOrientation(glm::vec4 value)
{
    if (!std::isfinite(value[0]) || !std::isfinite(value[1]) || !std::isfinite(value[2]) || !std::isfinite(value[3]) ||
        glm::dot(value, value) < 1e-12f)
        throw std::invalid_argument("SwingTwistJoint.target_orientation must be finite and nonzero");
    m_settings.targetOrientation = glm::normalize(value);
    UpdateConstraintSettings();
}

void SwingTwistJoint::SetTargetAngularVelocity(glm::vec3 value)
{
    if (!std::isfinite(value[0]) || !std::isfinite(value[1]) || !std::isfinite(value[2]))
        throw std::invalid_argument("SwingTwistJoint.target_angular_velocity must be finite");
    m_settings.targetAngularVelocity = value;
    UpdateConstraintSettings();
}

uint64_t SwingTwistJoint::CreateConstraint(PhysicsWorld &world, uint32_t bodyId, uint32_t connectedBodyId)
{
    TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    const auto *transform = GetTransform();
    const auto baked = glm::conjugate(transform->GetWorldRotation()) * transform->TransformVector(m_anchor);
    return world.CreateSwingTwistConstraint(*this, bodyId, connectedBodyId, baked, m_anchor, m_settings,
                                            m_enableCollision);
}

void SwingTwistJoint::UpdateConstraintSettings()
{
    if (!m_constraintId)
        RebuildConstraint();
    else
        PhysicsWorld::Instance().UpdateSwingTwistConstraint(m_constraintId, m_settings, m_enableCollision);
}

glm::quat SwingTwistJoint::GetCurrentOrientation() const
{
    return m_constraintId ? PhysicsWorld::Instance().GetAngularConstraintOrientation(m_constraintId)
                          : glm::quat(1, 0, 0, 0);
}

nlohmann::json SwingTwistJoint::SerializeDocument() const
{
    auto d = Joint::SerializeDocument();
    d["twist_axis"] = {m_settings.twistAxis[0], m_settings.twistAxis[1], m_settings.twistAxis[2]};
    d["plane_axis"] = {m_settings.planeAxis[0], m_settings.planeAxis[1], m_settings.planeAxis[2]};
    d["swing_type"] = (int)m_settings.swingType;
    d["normal_half_cone_angle"] = m_settings.normalHalfConeAngle;
    d["plane_half_cone_angle"] = m_settings.planeHalfConeAngle;
    d["minimum_twist_angle"] = m_settings.minimumTwistAngle;
    d["maximum_twist_angle"] = m_settings.maximumTwistAngle;
    d["friction_torque"] = m_settings.frictionTorque;
    d["swing_motor_mode"] = (int)m_settings.swingMotorMode;
    d["twist_motor_mode"] = (int)m_settings.twistMotorMode;
    d["swing_motor_stiffness"] = m_settings.swingMotorStiffness;
    d["swing_motor_damping"] = m_settings.swingMotorDamping;
    d["swing_motor_max_torque"] = m_settings.swingMotorMaxTorque;
    d["twist_motor_stiffness"] = m_settings.twistMotorStiffness;
    d["twist_motor_damping"] = m_settings.twistMotorDamping;
    d["twist_motor_max_torque"] = m_settings.twistMotorMaxTorque;
    d["target_orientation"] = {m_settings.targetOrientation[0], m_settings.targetOrientation[1],
                               m_settings.targetOrientation[2], m_settings.targetOrientation[3]};
    d["target_angular_velocity"] = {m_settings.targetAngularVelocity[0], m_settings.targetAngularVelocity[1],
                                    m_settings.targetAngularVelocity[2]};
    return d;
}

void SwingTwistJoint::ValidateSerializedDocument(const nlohmann::json &document)
{
    component_document_validation::ValidateComponentDocument(document, "SwingTwistJoint",
                                                             {"anchor",
                                                              "connected_body_component_id",
                                                              "enable_collision",
                                                              "twist_axis",
                                                              "plane_axis",
                                                              "swing_type",
                                                              "normal_half_cone_angle",
                                                              "plane_half_cone_angle",
                                                              "minimum_twist_angle",
                                                              "maximum_twist_angle",
                                                              "friction_torque",
                                                              "swing_motor_mode",
                                                              "twist_motor_mode",
                                                              "swing_motor_stiffness",
                                                              "swing_motor_damping",
                                                              "swing_motor_max_torque",
                                                              "twist_motor_stiffness",
                                                              "twist_motor_damping",
                                                              "twist_motor_max_torque",
                                                              "target_orientation",
                                                              "target_angular_velocity"});
    ValidateJointDocumentFields(document, "SwingTwistJoint");
    component_document_validation::ValidateDeclaredFields(document, "SwingTwistJoint", DescribeSemanticType());
    Validate(ReadSettings(document));
}

bool SwingTwistJoint::DeserializeDocument(const nlohmann::json &document)
{
    ValidateSerializedDocument(document);
    if (!Component::DeserializeDocument(document))
        return false;
    ReadJointDocumentFields(document);
    ReleaseConstraint();
    ApplySettings(ReadSettings(document), true);
    return true;
}

std::unique_ptr<Component> SwingTwistJoint::Clone() const
{
    auto clone = std::make_unique<SwingTwistJoint>();
    CopyJointFieldsTo(*clone);
    clone->m_settings = m_settings;
    return clone;
}
} // namespace infernux
