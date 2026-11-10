#include "SixDOFJoint.h"
#include "ComponentDocumentValidation.h"
#include "ComponentFactory.h"
#include "physics/PhysicsWorld.h"
#include <cmath>
#include <stdexcept>

namespace infernux
{
namespace
{
const bool registered = ComponentFactory::Register(
    "SixDOFJoint", [] { return std::make_unique<SixDOFJoint>(); }, SixDOFJoint::ValidateSerializedDocument,
    SixDOFJoint::GetTypeConstraints(), SixDOFJoint::DescribeSemanticType);

void Validate(const SixDOFJointSettings &s)
{
    const auto finite = [](const auto &v) {
        for (int i = 0; i < v.length(); ++i)
            if (!std::isfinite(v[i]))
                return false;
        return true;
    };
    if (!finite(s.axis) || !finite(s.secondaryAxis) || glm::dot(s.axis, s.axis) < 1e-12f ||
        glm::dot(s.secondaryAxis, s.secondaryAxis) < 1e-12f ||
        glm::dot(glm::cross(glm::normalize(s.axis), glm::normalize(s.secondaryAxis)),
                 glm::cross(glm::normalize(s.axis), glm::normalize(s.secondaryAxis))) < 1e-8f)
        throw std::invalid_argument("SixDOFJoint frame axes must be finite, nonzero and nonparallel");
    const JointMotion modes[] = {s.xMotion, s.yMotion, s.zMotion, s.angularXMotion, s.angularYMotion, s.angularZMotion};
    const JointMotorMode motors[] = {s.xMotorMode,        s.yMotorMode,        s.zMotorMode,
                                     s.angularXMotorMode, s.angularYMotorMode, s.angularZMotorMode};
    for (int i = 0; i < 6; ++i) {
        if ((int)modes[i] < 0 || (int)modes[i] > 2 || (int)motors[i] < 0 || (int)motors[i] > 3)
            throw std::invalid_argument("SixDOFJoint has an invalid motion or motor mode");
        const float minimum = i < 3 ? s.linearLimitMinimum[i] : s.angularLimitMinimum[i - 3];
        const float maximum = i < 3 ? s.linearLimitMaximum[i] : s.angularLimitMaximum[i - 3];
        if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum > maximum ||
            (modes[i] == JointMotion::Limited && minimum == maximum) || (i >= 3 && (minimum < -180 || maximum > 180)))
            throw std::invalid_argument(
                "SixDOFJoint requires ordered limits, nonempty Limited ranges, and angular degrees in [-180,180]");
    }
    if (!finite(s.linearLimitStiffness) || glm::any(glm::lessThan(s.linearLimitStiffness, glm::vec3(0))))
        throw std::invalid_argument("SixDOFJoint.linear_limit_stiffness requires finite non-negative SI coefficients");
    if (!finite(s.linearLimitDamping) || glm::any(glm::lessThan(s.linearLimitDamping, glm::vec3(0))))
        throw std::invalid_argument("SixDOFJoint.linear_limit_damping requires finite non-negative SI coefficients");
    if (!finite(s.linearFrictionForce) || glm::any(glm::lessThan(s.linearFrictionForce, glm::vec3(0))))
        throw std::invalid_argument("SixDOFJoint.linear_friction_force requires finite non-negative SI coefficients");
    if (!finite(s.angularFrictionTorque) || glm::any(glm::lessThan(s.angularFrictionTorque, glm::vec3(0))))
        throw std::invalid_argument("SixDOFJoint.angular_friction_torque requires finite non-negative SI coefficients");
    if (!finite(s.linearMotorStiffness) || glm::any(glm::lessThan(s.linearMotorStiffness, glm::vec3(0))))
        throw std::invalid_argument("SixDOFJoint.linear_motor_stiffness requires finite non-negative SI coefficients");
    if (!finite(s.linearMotorDamping) || glm::any(glm::lessThan(s.linearMotorDamping, glm::vec3(0))))
        throw std::invalid_argument("SixDOFJoint.linear_motor_damping requires finite non-negative SI coefficients");
    if (!finite(s.linearMotorMaxForce) || glm::any(glm::lessThan(s.linearMotorMaxForce, glm::vec3(0))))
        throw std::invalid_argument("SixDOFJoint.linear_motor_max_force requires finite non-negative SI coefficients");
    if (!finite(s.angularMotorStiffness) || glm::any(glm::lessThan(s.angularMotorStiffness, glm::vec3(0))))
        throw std::invalid_argument("SixDOFJoint.angular_motor_stiffness requires finite non-negative SI coefficients");
    if (!finite(s.angularMotorDamping) || glm::any(glm::lessThan(s.angularMotorDamping, glm::vec3(0))))
        throw std::invalid_argument("SixDOFJoint.angular_motor_damping requires finite non-negative SI coefficients");
    if (!finite(s.angularMotorMaxTorque) || glm::any(glm::lessThan(s.angularMotorMaxTorque, glm::vec3(0))))
        throw std::invalid_argument(
            "SixDOFJoint.angular_motor_max_torque requires finite non-negative SI coefficients");
    if (!finite(s.targetPosition) || !finite(s.targetVelocity) || !finite(s.targetAngularVelocity) ||
        !finite(s.targetOrientation) || glm::dot(s.targetOrientation, s.targetOrientation) < 1e-12f)
        throw std::invalid_argument("SixDOFJoint targets must be finite and its XYZW orientation nonzero");
}

SixDOFJointSettings ReadSettings(const nlohmann::json &d)
{
    SixDOFJointSettings s;
    s.axis = {d.at("axis").at(0).get<float>(), d.at("axis").at(1).get<float>(), d.at("axis").at(2).get<float>()};
    s.secondaryAxis = {d.at("secondary_axis").at(0).get<float>(), d.at("secondary_axis").at(1).get<float>(),
                       d.at("secondary_axis").at(2).get<float>()};
    s.xMotion = static_cast<JointMotion>(d.at("x_motion").get<int>());
    s.yMotion = static_cast<JointMotion>(d.at("y_motion").get<int>());
    s.zMotion = static_cast<JointMotion>(d.at("z_motion").get<int>());
    s.angularXMotion = static_cast<JointMotion>(d.at("angular_x_motion").get<int>());
    s.angularYMotion = static_cast<JointMotion>(d.at("angular_y_motion").get<int>());
    s.angularZMotion = static_cast<JointMotion>(d.at("angular_z_motion").get<int>());
    s.linearLimitMinimum = {d.at("linear_limit_minimum").at(0).get<float>(),
                            d.at("linear_limit_minimum").at(1).get<float>(),
                            d.at("linear_limit_minimum").at(2).get<float>()};
    s.linearLimitMaximum = {d.at("linear_limit_maximum").at(0).get<float>(),
                            d.at("linear_limit_maximum").at(1).get<float>(),
                            d.at("linear_limit_maximum").at(2).get<float>()};
    s.angularLimitMinimum = {d.at("angular_limit_minimum").at(0).get<float>(),
                             d.at("angular_limit_minimum").at(1).get<float>(),
                             d.at("angular_limit_minimum").at(2).get<float>()};
    s.angularLimitMaximum = {d.at("angular_limit_maximum").at(0).get<float>(),
                             d.at("angular_limit_maximum").at(1).get<float>(),
                             d.at("angular_limit_maximum").at(2).get<float>()};
    s.linearLimitStiffness = {d.at("linear_limit_stiffness").at(0).get<float>(),
                              d.at("linear_limit_stiffness").at(1).get<float>(),
                              d.at("linear_limit_stiffness").at(2).get<float>()};
    s.linearLimitDamping = {d.at("linear_limit_damping").at(0).get<float>(),
                            d.at("linear_limit_damping").at(1).get<float>(),
                            d.at("linear_limit_damping").at(2).get<float>()};
    s.linearFrictionForce = {d.at("linear_friction_force").at(0).get<float>(),
                             d.at("linear_friction_force").at(1).get<float>(),
                             d.at("linear_friction_force").at(2).get<float>()};
    s.angularFrictionTorque = {d.at("angular_friction_torque").at(0).get<float>(),
                               d.at("angular_friction_torque").at(1).get<float>(),
                               d.at("angular_friction_torque").at(2).get<float>()};
    s.linearMotorStiffness = {d.at("linear_motor_stiffness").at(0).get<float>(),
                              d.at("linear_motor_stiffness").at(1).get<float>(),
                              d.at("linear_motor_stiffness").at(2).get<float>()};
    s.linearMotorDamping = {d.at("linear_motor_damping").at(0).get<float>(),
                            d.at("linear_motor_damping").at(1).get<float>(),
                            d.at("linear_motor_damping").at(2).get<float>()};
    s.linearMotorMaxForce = {d.at("linear_motor_max_force").at(0).get<float>(),
                             d.at("linear_motor_max_force").at(1).get<float>(),
                             d.at("linear_motor_max_force").at(2).get<float>()};
    s.angularMotorStiffness = {d.at("angular_motor_stiffness").at(0).get<float>(),
                               d.at("angular_motor_stiffness").at(1).get<float>(),
                               d.at("angular_motor_stiffness").at(2).get<float>()};
    s.angularMotorDamping = {d.at("angular_motor_damping").at(0).get<float>(),
                             d.at("angular_motor_damping").at(1).get<float>(),
                             d.at("angular_motor_damping").at(2).get<float>()};
    s.angularMotorMaxTorque = {d.at("angular_motor_max_torque").at(0).get<float>(),
                               d.at("angular_motor_max_torque").at(1).get<float>(),
                               d.at("angular_motor_max_torque").at(2).get<float>()};
    s.xMotorMode = static_cast<JointMotorMode>(d.at("x_motor_mode").get<int>());
    s.yMotorMode = static_cast<JointMotorMode>(d.at("y_motor_mode").get<int>());
    s.zMotorMode = static_cast<JointMotorMode>(d.at("z_motor_mode").get<int>());
    s.angularXMotorMode = static_cast<JointMotorMode>(d.at("angular_x_motor_mode").get<int>());
    s.angularYMotorMode = static_cast<JointMotorMode>(d.at("angular_y_motor_mode").get<int>());
    s.angularZMotorMode = static_cast<JointMotorMode>(d.at("angular_z_motor_mode").get<int>());
    s.targetPosition = {d.at("target_position").at(0).get<float>(), d.at("target_position").at(1).get<float>(),
                        d.at("target_position").at(2).get<float>()};
    s.targetOrientation = {d.at("target_orientation").at(0).get<float>(), d.at("target_orientation").at(1).get<float>(),
                           d.at("target_orientation").at(2).get<float>(),
                           d.at("target_orientation").at(3).get<float>()};
    s.targetVelocity = {d.at("target_velocity").at(0).get<float>(), d.at("target_velocity").at(1).get<float>(),
                        d.at("target_velocity").at(2).get<float>()};
    s.targetAngularVelocity = {d.at("target_angular_velocity").at(0).get<float>(),
                               d.at("target_angular_velocity").at(1).get<float>(),
                               d.at("target_angular_velocity").at(2).get<float>()};
    return s;
}
} // namespace

SemanticTypeDescriptor SixDOFJoint::DescribeSemanticType()
{
    auto type = DescribeJointType("SixDOFJoint", "infernux.component.six_dof_joint", "Six DOF Joint");
    AddJointField(type, "axis", "VEC3", {1.0, 0.0, 0.0})["tooltip"] = "Owner-local frame X axis";
    AddJointField(type, "secondary_axis", "VEC3", {0.0, 1.0, 0.0})["tooltip"] = "Owner-local frame Y axis";
    AddJointEnum(type, "x_motion", "JointMotion", {"Locked", "Limited", "Free"}, "Locked");
    AddJointEnum(type, "y_motion", "JointMotion", {"Locked", "Limited", "Free"}, "Locked");
    AddJointEnum(type, "z_motion", "JointMotion", {"Locked", "Limited", "Free"}, "Locked");
    AddJointEnum(type, "angular_x_motion", "JointMotion", {"Locked", "Limited", "Free"}, "Locked");
    AddJointEnum(type, "angular_y_motion", "JointMotion", {"Locked", "Limited", "Free"}, "Locked");
    AddJointEnum(type, "angular_z_motion", "JointMotion", {"Locked", "Limited", "Free"}, "Locked");
    AddJointField(type, "linear_limit_minimum", "VEC3", {-1.0, -1.0, -1.0})["tooltip"] =
        "Metres in the constraint frame";
    AddJointField(type, "linear_limit_maximum", "VEC3", {1.0, 1.0, 1.0})["tooltip"] = "Metres in the constraint frame";
    AddJointField(type, "angular_limit_minimum", "VEC3", {-45.0, -45.0, -45.0})["tooltip"] =
        "Degrees in the constraint frame";
    AddJointField(type, "angular_limit_maximum", "VEC3", {45.0, 45.0, 45.0})["tooltip"] =
        "Degrees in the constraint frame";
    AddJointField(type, "linear_limit_stiffness", "VEC3", {0.0, 0.0, 0.0})["tooltip"] = "N/m";
    AddJointField(type, "linear_limit_damping", "VEC3", {0.0, 0.0, 0.0})["tooltip"] =
        "N s/m; limit damping requires positive limit stiffness";
    AddJointField(type, "linear_friction_force", "VEC3", {0.0, 0.0, 0.0})["tooltip"] = "N";
    AddJointField(type, "angular_friction_torque", "VEC3", {0.0, 0.0, 0.0})["tooltip"] = "N m";
    AddJointField(type, "linear_motor_stiffness", "VEC3", {0.0, 0.0, 0.0})["tooltip"] = "N/m";
    AddJointField(type, "linear_motor_damping", "VEC3", {0.0, 0.0, 0.0})["tooltip"] = "N s/m";
    AddJointField(type, "linear_motor_max_force", "VEC3", {10.0, 10.0, 10.0})["tooltip"] = "N";
    AddJointField(type, "angular_motor_stiffness", "VEC3", {0.0, 0.0, 0.0})["tooltip"] = "N m/rad";
    AddJointField(type, "angular_motor_damping", "VEC3", {0.0, 0.0, 0.0})["tooltip"] = "N m s/rad";
    AddJointField(type, "angular_motor_max_torque", "VEC3", {10.0, 10.0, 10.0})["tooltip"] = "N m";
    AddJointEnum(type, "x_motor_mode", "JointMotorMode", {"Off", "Velocity", "Position", "PositionAndVelocity"}, "Off");
    AddJointEnum(type, "y_motor_mode", "JointMotorMode", {"Off", "Velocity", "Position", "PositionAndVelocity"}, "Off");
    AddJointEnum(type, "z_motor_mode", "JointMotorMode", {"Off", "Velocity", "Position", "PositionAndVelocity"}, "Off");
    AddJointEnum(type, "angular_x_motor_mode", "JointMotorMode", {"Off", "Velocity", "Position", "PositionAndVelocity"},
                 "Off");
    AddJointEnum(type, "angular_y_motor_mode", "JointMotorMode", {"Off", "Velocity", "Position", "PositionAndVelocity"},
                 "Off");
    AddJointEnum(type, "angular_z_motor_mode", "JointMotorMode", {"Off", "Velocity", "Position", "PositionAndVelocity"},
                 "Off");
    AddJointField(type, "target_position", "VEC3", {0.0, 0.0, 0.0})["tooltip"] =
        "Metres in the connected constraint frame";
    AddJointField(type, "target_orientation", "VEC4", {0.0, 0.0, 0.0, 1.0})["tooltip"] =
        "Unit XYZW quaternion in constraint space";
    AddJointField(type, "target_velocity", "VEC3", {0.0, 0.0, 0.0})["tooltip"] =
        "Metres/s in the connected constraint frame";
    AddJointField(type, "target_angular_velocity", "VEC3", {0.0, 0.0, 0.0})["tooltip"] =
        "Degrees/s in the owner constraint frame";
    return type;
}

void SixDOFJoint::ApplySettings(SixDOFJointSettings settings, bool frameChanged)
{
    Validate(settings);
    if (settings.axis != m_settings.axis)
        settings.axis = glm::normalize(settings.axis);
    if (settings.axis != m_settings.axis || settings.secondaryAxis != m_settings.secondaryAxis)
        settings.secondaryAxis =
            glm::normalize(settings.secondaryAxis - settings.axis * glm::dot(settings.secondaryAxis, settings.axis));
    if (settings.targetOrientation != m_settings.targetOrientation)
        settings.targetOrientation = glm::normalize(settings.targetOrientation);
    frameChanged =
        frameChanged && (settings.axis != m_settings.axis || settings.secondaryAxis != m_settings.secondaryAxis);
    m_settings = settings;
    if (frameChanged)
        ReleaseConstraint();
    UpdateConstraintSettings();
}

void SixDOFJoint::SetFrame(const glm::vec3 &axis, const glm::vec3 &secondaryAxis)
{
    if (axis == m_settings.axis && secondaryAxis == m_settings.secondaryAxis && m_constraintId)
        return;
    auto settings = m_settings;
    settings.axis = axis;
    settings.secondaryAxis = secondaryAxis;
    ApplySettings(settings, true);
}

void SixDOFJoint::SetLinearLimits(const glm::vec3 &minimum, const glm::vec3 &maximum)
{
    auto settings = m_settings;
    settings.linearLimitMinimum = minimum;
    settings.linearLimitMaximum = maximum;
    ApplySettings(settings);
}

void SixDOFJoint::SetAngularLimits(const glm::vec3 &minimum, const glm::vec3 &maximum)
{
    auto settings = m_settings;
    settings.angularLimitMinimum = minimum;
    settings.angularLimitMaximum = maximum;
    ApplySettings(settings);
}

void SixDOFJoint::SetAxis(glm::vec3 value)
{
    if (value == m_settings.axis && m_constraintId)
        return;
    auto settings = m_settings;
    settings.axis = value;
    ApplySettings(settings, true);
}

void SixDOFJoint::SetSecondaryAxis(glm::vec3 value)
{
    if (value == m_settings.secondaryAxis && m_constraintId)
        return;
    auto settings = m_settings;
    settings.secondaryAxis = value;
    ApplySettings(settings, true);
}

void SixDOFJoint::SetXMotion(JointMotion value)
{
    auto settings = m_settings;
    settings.xMotion = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetYMotion(JointMotion value)
{
    auto settings = m_settings;
    settings.yMotion = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetZMotion(JointMotion value)
{
    auto settings = m_settings;
    settings.zMotion = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetAngularXMotion(JointMotion value)
{
    auto settings = m_settings;
    settings.angularXMotion = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetAngularYMotion(JointMotion value)
{
    auto settings = m_settings;
    settings.angularYMotion = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetAngularZMotion(JointMotion value)
{
    auto settings = m_settings;
    settings.angularZMotion = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetLinearLimitMinimum(glm::vec3 value)
{
    auto settings = m_settings;
    settings.linearLimitMinimum = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetLinearLimitMaximum(glm::vec3 value)
{
    auto settings = m_settings;
    settings.linearLimitMaximum = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetAngularLimitMinimum(glm::vec3 value)
{
    auto settings = m_settings;
    settings.angularLimitMinimum = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetAngularLimitMaximum(glm::vec3 value)
{
    auto settings = m_settings;
    settings.angularLimitMaximum = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetLinearLimitStiffness(glm::vec3 value)
{
    auto settings = m_settings;
    settings.linearLimitStiffness = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetLinearLimitDamping(glm::vec3 value)
{
    auto settings = m_settings;
    settings.linearLimitDamping = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetLinearFrictionForce(glm::vec3 value)
{
    auto settings = m_settings;
    settings.linearFrictionForce = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetAngularFrictionTorque(glm::vec3 value)
{
    auto settings = m_settings;
    settings.angularFrictionTorque = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetLinearMotorStiffness(glm::vec3 value)
{
    auto settings = m_settings;
    settings.linearMotorStiffness = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetLinearMotorDamping(glm::vec3 value)
{
    auto settings = m_settings;
    settings.linearMotorDamping = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetLinearMotorMaxForce(glm::vec3 value)
{
    auto settings = m_settings;
    settings.linearMotorMaxForce = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetAngularMotorStiffness(glm::vec3 value)
{
    auto settings = m_settings;
    settings.angularMotorStiffness = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetAngularMotorDamping(glm::vec3 value)
{
    auto settings = m_settings;
    settings.angularMotorDamping = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetAngularMotorMaxTorque(glm::vec3 value)
{
    auto settings = m_settings;
    settings.angularMotorMaxTorque = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetXMotorMode(JointMotorMode value)
{
    auto settings = m_settings;
    settings.xMotorMode = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetYMotorMode(JointMotorMode value)
{
    auto settings = m_settings;
    settings.yMotorMode = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetZMotorMode(JointMotorMode value)
{
    auto settings = m_settings;
    settings.zMotorMode = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetAngularXMotorMode(JointMotorMode value)
{
    auto settings = m_settings;
    settings.angularXMotorMode = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetAngularYMotorMode(JointMotorMode value)
{
    auto settings = m_settings;
    settings.angularYMotorMode = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetAngularZMotorMode(JointMotorMode value)
{
    auto settings = m_settings;
    settings.angularZMotorMode = value;
    ApplySettings(settings);
}

void SixDOFJoint::SetTargetPosition(glm::vec3 value)
{
    if (!std::isfinite(value[0]) || !std::isfinite(value[1]) || !std::isfinite(value[2]))
        throw std::invalid_argument("SixDOFJoint.target_position must be finite");
    m_settings.targetPosition = value;
    UpdateConstraintSettings();
}

void SixDOFJoint::SetTargetOrientation(glm::vec4 value)
{
    if (!std::isfinite(value[0]) || !std::isfinite(value[1]) || !std::isfinite(value[2]) || !std::isfinite(value[3]) ||
        glm::dot(value, value) < 1e-12f)
        throw std::invalid_argument("SixDOFJoint.target_orientation must be finite and nonzero");
    m_settings.targetOrientation = glm::normalize(value);
    UpdateConstraintSettings();
}

void SixDOFJoint::SetTargetVelocity(glm::vec3 value)
{
    if (!std::isfinite(value[0]) || !std::isfinite(value[1]) || !std::isfinite(value[2]))
        throw std::invalid_argument("SixDOFJoint.target_velocity must be finite");
    m_settings.targetVelocity = value;
    UpdateConstraintSettings();
}

void SixDOFJoint::SetTargetAngularVelocity(glm::vec3 value)
{
    if (!std::isfinite(value[0]) || !std::isfinite(value[1]) || !std::isfinite(value[2]))
        throw std::invalid_argument("SixDOFJoint.target_angular_velocity must be finite");
    m_settings.targetAngularVelocity = value;
    UpdateConstraintSettings();
}

uint64_t SixDOFJoint::CreateConstraint(PhysicsWorld &world, uint32_t bodyId, uint32_t connectedBodyId)
{
    return world.CreateSixDOFConstraint(*this, bodyId, connectedBodyId, m_anchor, m_settings, m_enableCollision);
}

void SixDOFJoint::UpdateConstraintSettings()
{
    if (!m_constraintId)
        RebuildConstraint();
    else
        PhysicsWorld::Instance().UpdateSixDOFConstraint(m_constraintId, m_settings, m_enableCollision);
}

glm::quat SixDOFJoint::GetCurrentOrientation() const
{
    return m_constraintId ? PhysicsWorld::Instance().GetAngularConstraintOrientation(m_constraintId)
                          : glm::quat(1, 0, 0, 0);
}

glm::vec3 SixDOFJoint::GetCurrentPosition() const
{
    return m_constraintId ? PhysicsWorld::Instance().GetSixDOFConstraintPosition(m_constraintId) : glm::vec3(0);
}

nlohmann::json SixDOFJoint::SerializeDocument() const
{
    auto d = Joint::SerializeDocument();
    d["axis"] = {m_settings.axis[0], m_settings.axis[1], m_settings.axis[2]};
    d["secondary_axis"] = {m_settings.secondaryAxis[0], m_settings.secondaryAxis[1], m_settings.secondaryAxis[2]};
    d["x_motion"] = (int)m_settings.xMotion;
    d["y_motion"] = (int)m_settings.yMotion;
    d["z_motion"] = (int)m_settings.zMotion;
    d["angular_x_motion"] = (int)m_settings.angularXMotion;
    d["angular_y_motion"] = (int)m_settings.angularYMotion;
    d["angular_z_motion"] = (int)m_settings.angularZMotion;
    d["linear_limit_minimum"] = {m_settings.linearLimitMinimum[0], m_settings.linearLimitMinimum[1],
                                 m_settings.linearLimitMinimum[2]};
    d["linear_limit_maximum"] = {m_settings.linearLimitMaximum[0], m_settings.linearLimitMaximum[1],
                                 m_settings.linearLimitMaximum[2]};
    d["angular_limit_minimum"] = {m_settings.angularLimitMinimum[0], m_settings.angularLimitMinimum[1],
                                  m_settings.angularLimitMinimum[2]};
    d["angular_limit_maximum"] = {m_settings.angularLimitMaximum[0], m_settings.angularLimitMaximum[1],
                                  m_settings.angularLimitMaximum[2]};
    d["linear_limit_stiffness"] = {m_settings.linearLimitStiffness[0], m_settings.linearLimitStiffness[1],
                                   m_settings.linearLimitStiffness[2]};
    d["linear_limit_damping"] = {m_settings.linearLimitDamping[0], m_settings.linearLimitDamping[1],
                                 m_settings.linearLimitDamping[2]};
    d["linear_friction_force"] = {m_settings.linearFrictionForce[0], m_settings.linearFrictionForce[1],
                                  m_settings.linearFrictionForce[2]};
    d["angular_friction_torque"] = {m_settings.angularFrictionTorque[0], m_settings.angularFrictionTorque[1],
                                    m_settings.angularFrictionTorque[2]};
    d["linear_motor_stiffness"] = {m_settings.linearMotorStiffness[0], m_settings.linearMotorStiffness[1],
                                   m_settings.linearMotorStiffness[2]};
    d["linear_motor_damping"] = {m_settings.linearMotorDamping[0], m_settings.linearMotorDamping[1],
                                 m_settings.linearMotorDamping[2]};
    d["linear_motor_max_force"] = {m_settings.linearMotorMaxForce[0], m_settings.linearMotorMaxForce[1],
                                   m_settings.linearMotorMaxForce[2]};
    d["angular_motor_stiffness"] = {m_settings.angularMotorStiffness[0], m_settings.angularMotorStiffness[1],
                                    m_settings.angularMotorStiffness[2]};
    d["angular_motor_damping"] = {m_settings.angularMotorDamping[0], m_settings.angularMotorDamping[1],
                                  m_settings.angularMotorDamping[2]};
    d["angular_motor_max_torque"] = {m_settings.angularMotorMaxTorque[0], m_settings.angularMotorMaxTorque[1],
                                     m_settings.angularMotorMaxTorque[2]};
    d["x_motor_mode"] = (int)m_settings.xMotorMode;
    d["y_motor_mode"] = (int)m_settings.yMotorMode;
    d["z_motor_mode"] = (int)m_settings.zMotorMode;
    d["angular_x_motor_mode"] = (int)m_settings.angularXMotorMode;
    d["angular_y_motor_mode"] = (int)m_settings.angularYMotorMode;
    d["angular_z_motor_mode"] = (int)m_settings.angularZMotorMode;
    d["target_position"] = {m_settings.targetPosition[0], m_settings.targetPosition[1], m_settings.targetPosition[2]};
    d["target_orientation"] = {m_settings.targetOrientation[0], m_settings.targetOrientation[1],
                               m_settings.targetOrientation[2], m_settings.targetOrientation[3]};
    d["target_velocity"] = {m_settings.targetVelocity[0], m_settings.targetVelocity[1], m_settings.targetVelocity[2]};
    d["target_angular_velocity"] = {m_settings.targetAngularVelocity[0], m_settings.targetAngularVelocity[1],
                                    m_settings.targetAngularVelocity[2]};
    return d;
}

void SixDOFJoint::ValidateSerializedDocument(const nlohmann::json &document)
{
    component_document_validation::ValidateComponentDocument(document, "SixDOFJoint",
                                                             {"anchor",
                                                              "connected_body_component_id",
                                                              "enable_collision",
                                                              "axis",
                                                              "secondary_axis",
                                                              "x_motion",
                                                              "y_motion",
                                                              "z_motion",
                                                              "angular_x_motion",
                                                              "angular_y_motion",
                                                              "angular_z_motion",
                                                              "linear_limit_minimum",
                                                              "linear_limit_maximum",
                                                              "angular_limit_minimum",
                                                              "angular_limit_maximum",
                                                              "linear_limit_stiffness",
                                                              "linear_limit_damping",
                                                              "linear_friction_force",
                                                              "angular_friction_torque",
                                                              "linear_motor_stiffness",
                                                              "linear_motor_damping",
                                                              "linear_motor_max_force",
                                                              "angular_motor_stiffness",
                                                              "angular_motor_damping",
                                                              "angular_motor_max_torque",
                                                              "x_motor_mode",
                                                              "y_motor_mode",
                                                              "z_motor_mode",
                                                              "angular_x_motor_mode",
                                                              "angular_y_motor_mode",
                                                              "angular_z_motor_mode",
                                                              "target_position",
                                                              "target_orientation",
                                                              "target_velocity",
                                                              "target_angular_velocity"});
    ValidateJointDocumentFields(document, "SixDOFJoint");
    component_document_validation::ValidateDeclaredFields(document, "SixDOFJoint", DescribeSemanticType());
    Validate(ReadSettings(document));
}

bool SixDOFJoint::DeserializeDocument(const nlohmann::json &document)
{
    ValidateSerializedDocument(document);
    if (!Component::DeserializeDocument(document))
        return false;
    ReadJointDocumentFields(document);
    ReleaseConstraint();
    ApplySettings(ReadSettings(document), true);
    return true;
}

std::unique_ptr<Component> SixDOFJoint::Clone() const
{
    auto clone = std::make_unique<SixDOFJoint>();
    CopyJointFieldsTo(*clone);
    clone->m_settings = m_settings;
    return clone;
}
} // namespace infernux
