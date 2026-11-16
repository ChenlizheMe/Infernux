#include "JoltCpuConstraints.h"
#include "JoltCpuScene.h"

#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Body/BodyLockMulti.h>
#include <Jolt/Physics/Constraints/DistanceConstraint.h>
#include <Jolt/Physics/Constraints/FixedConstraint.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/SixDOFConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>
#include <Jolt/Physics/Constraints/SwingTwistConstraint.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace infernux
{
namespace
{
using Constraints = JoltCpuConstraints;
using SubType = JPH::EConstraintSubType;

struct Axes
{
    JPH::Vec3 primary = JPH::Vec3::sAxisX();
    JPH::Vec3 secondary = JPH::Vec3::sAxisY();
};
struct Frame
{
    JPH::RVec3 point;
    Axes axes;
};
struct Frames
{
    Frame connected, owner;
    JPH::EConstraintSpace space = JPH::EConstraintSpace::WorldSpace;
};

JPH::Vec3 Vector(const glm::vec3 &v)
{
    if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z))
        throw std::invalid_argument("Joint vectors must be finite");
    return {v.x, v.y, v.z};
}
JPH::Quat Orientation(const glm::vec4 &v)
{
    const float lengthSq = glm::dot(v, v);
    if (!std::isfinite(lengthSq) || lengthSq < 1e-12f)
        throw std::invalid_argument("Joint target orientation must be finite and nonzero");
    return JPH::Quat(v.x, v.y, v.z, v.w).Normalized();
}
JPH::Vec3 UnitAxis(JPH::Vec3Arg axis)
{
    const float lengthSq = axis.LengthSq();
    if (!std::isfinite(lengthSq) || lengthSq < 1e-12f)
        throw std::invalid_argument("Joint axis must be finite and nonzero");
    return axis / std::sqrt(lengthSq);
}
Axes MakeAxes(const glm::vec3 &primary, const glm::vec3 &secondary)
{
    const auto x = UnitAxis(Vector(primary));
    const auto other = UnitAxis(Vector(secondary));
    const auto y = other - x * x.Dot(other);
    if (y.LengthSq() < 1e-8f)
        throw std::invalid_argument("Joint frame axes must be nonparallel");
    return {x, y.Normalized()};
}
Axes MakeAxes(const glm::vec3 &axis)
{
    const auto primary = UnitAxis(Vector(axis));
    return {primary, primary.GetNormalizedPerpendicular()};
}
void Range(float low, float high, float lowerBound, float upperBound)
{
    if (!std::isfinite(low) || !std::isfinite(high) || low > high || low < lowerBound || high > upperBound)
        throw std::invalid_argument("Joint limits must be finite and ordered within the supported range");
}
void Coefficients(std::initializer_list<float> values)
{
    for (const float value : values)
        if (!std::isfinite(value) || value < 0.0f)
            throw std::invalid_argument("Joint coefficients must be finite and nonnegative");
}
JPH::EMotorState MotorState(JointMotorMode mode)
{
    switch (mode) {
    case JointMotorMode::Off:
        return JPH::EMotorState::Off;
    case JointMotorMode::Velocity:
        return JPH::EMotorState::Velocity;
    case JointMotorMode::Position:
        return JPH::EMotorState::Position;
    case JointMotorMode::PositionAndVelocity:
        return JPH::EMotorState::PositionAndVelocity;
    }
    throw std::invalid_argument("Invalid joint motor mode");
}
JPH::ESwingType SwingType(JointSwingType type)
{
    switch (type) {
    case JointSwingType::Cone:
        return JPH::ESwingType::Cone;
    case JointSwingType::Pyramid:
        return JPH::ESwingType::Pyramid;
    }
    throw std::invalid_argument("Invalid joint swing type");
}

Axes Validate(const Constraints::Hinge &s)
{
    Range(s.minimum, s.maximum, -JPH::JPH_PI, JPH::JPH_PI);
    if (s.minimum > 0.0f || s.maximum < 0.0f || !std::isfinite(s.targetAngle))
        throw std::invalid_argument("Hinge limits must span zero and its target must be finite");
    Coefficients({s.stiffness, s.damping, s.maxTorque, s.jointDamping, s.frictionTorque, s.armature});
    return MakeAxes(s.axis);
}
Axes Validate(const Constraints::Slider &s)
{
    Range(s.minimum, s.maximum, -FLT_MAX, FLT_MAX);
    if (s.minimum > 0.0f || s.maximum < 0.0f)
        throw std::invalid_argument("Slider limits must span zero");
    return MakeAxes(s.axis);
}
Axes Validate(const Constraints::Fixed &)
{
    return {};
}
Axes Validate(const Constraints::Distance &s)
{
    Range(s.minimum, s.maximum, 0.0f, FLT_MAX);
    Coefficients({s.stiffness, s.damping});
    return {};
}
Axes Validate(const SwingTwistJointSettings &s)
{
    Range(s.minimumTwistAngle, s.maximumTwistAngle, -180.0f, 180.0f);
    Range(0.0f, s.planeHalfConeAngle, 0.0f, 180.0f);
    Range(0.0f, s.normalHalfConeAngle, 0.0f, 180.0f);
    Coefficients({s.frictionTorque, s.swingMotorStiffness, s.swingMotorDamping, s.swingMotorMaxTorque,
                  s.twistMotorStiffness, s.twistMotorDamping, s.twistMotorMaxTorque});
    (void)SwingType(s.swingType);
    (void)MotorState(s.swingMotorMode);
    (void)MotorState(s.twistMotorMode);
    (void)Orientation(s.targetOrientation);
    (void)Vector(s.targetAngularVelocity);
    return MakeAxes(s.twistAxis, s.planeAxis);
}

auto SixMotions(const SixDOFJointSettings &s)
{
    return std::array{s.xMotion, s.yMotion, s.zMotion, s.angularXMotion, s.angularYMotion, s.angularZMotion};
}
auto SixMotors(const SixDOFJointSettings &s)
{
    return std::array{s.xMotorMode,        s.yMotorMode,        s.zMotorMode,
                      s.angularXMotorMode, s.angularYMotorMode, s.angularZMotorMode};
}
Axes Validate(const SixDOFJointSettings &s)
{
    const auto motions = SixMotions(s);
    const auto motors = SixMotors(s);
    for (int axis = 0; axis < 3; ++axis) {
        Range(s.linearLimitMinimum[axis], s.linearLimitMaximum[axis], -FLT_MAX, FLT_MAX);
        Range(s.angularLimitMinimum[axis], s.angularLimitMaximum[axis], -180.0f, 180.0f);
        if ((motions[axis] == JointMotion::Limited && s.linearLimitMinimum[axis] == s.linearLimitMaximum[axis]) ||
            (motions[axis + 3] == JointMotion::Limited && s.angularLimitMinimum[axis] == s.angularLimitMaximum[axis]))
            throw std::invalid_argument("SixDOF limited axes need a nonempty range");
        Coefficients({s.linearLimitStiffness[axis], s.linearLimitDamping[axis], s.linearFrictionForce[axis],
                      s.angularFrictionTorque[axis], s.linearMotorStiffness[axis], s.linearMotorDamping[axis],
                      s.linearMotorMaxForce[axis], s.angularMotorStiffness[axis], s.angularMotorDamping[axis],
                      s.angularMotorMaxTorque[axis]});
    }
    for (size_t i = 0; i < motions.size(); ++i) {
        if (motions[i] != JointMotion::Locked && motions[i] != JointMotion::Limited && motions[i] != JointMotion::Free)
            throw std::invalid_argument("Invalid SixDOF motion mode");
        (void)MotorState(motors[i]);
    }
    (void)Vector(s.targetPosition);
    (void)Vector(s.targetVelocity);
    (void)Vector(s.targetAngularVelocity);
    (void)Orientation(s.targetOrientation);
    return MakeAxes(s.axis, s.secondaryAxis);
}

JPH::SpringSettings Spring(float stiffness, float damping)
{
    return {JPH::ESpringMode::StiffnessAndDamping, stiffness, damping};
}

using Native = JPH::TwoBodyConstraint;
JPH::Ref<Native> CreateNative(const Constraints::Hinge &, const Frames &f, JPH::Body &connected, JPH::Body &owner)
{
    JPH::HingeConstraintSettings settings;
    settings.mPoint1 = f.connected.point;
    settings.mPoint2 = f.owner.point;
    settings.mHingeAxis1 = f.connected.axes.primary;
    settings.mHingeAxis2 = f.owner.axes.primary;
    settings.mNormalAxis1 = f.connected.axes.secondary;
    settings.mNormalAxis2 = f.owner.axes.secondary;
    return settings.Create(connected, owner);
}
JPH::Ref<Native> CreateNative(const Constraints::Slider &, const Frames &f, JPH::Body &connected, JPH::Body &owner)
{
    JPH::SliderConstraintSettings settings;
    settings.mPoint1 = f.connected.point;
    settings.mPoint2 = f.owner.point;
    settings.mSliderAxis1 = f.connected.axes.primary;
    settings.mSliderAxis2 = f.owner.axes.primary;
    settings.mNormalAxis1 = f.connected.axes.secondary;
    settings.mNormalAxis2 = f.owner.axes.secondary;
    return settings.Create(connected, owner);
}
JPH::Ref<Native> CreateNative(const Constraints::Fixed &, const Frames &f, JPH::Body &connected, JPH::Body &owner)
{
    JPH::FixedConstraintSettings settings;
    settings.mPoint1 = f.connected.point;
    settings.mPoint2 = f.owner.point;
    return settings.Create(connected, owner);
}
JPH::Ref<Native> CreateNative(const Constraints::Distance &s, const Frames &f, JPH::Body &connected, JPH::Body &owner)
{
    JPH::DistanceConstraintSettings settings;
    settings.mPoint1 = f.connected.point;
    settings.mPoint2 = f.owner.point;
    settings.mMinDistance = s.autoDistance ? -1.0f : s.minimum;
    settings.mMaxDistance = s.autoDistance ? -1.0f : s.maximum;
    return settings.Create(connected, owner);
}
JPH::Ref<Native> CreateNative(const SwingTwistJointSettings &s, const Frames &f, JPH::Body &connected, JPH::Body &owner)
{
    JPH::SwingTwistConstraintSettings settings;
    settings.mSpace = f.space;
    settings.mPosition1 = f.connected.point;
    settings.mPosition2 = f.owner.point;
    settings.mTwistAxis1 = f.connected.axes.primary;
    settings.mTwistAxis2 = f.owner.axes.primary;
    settings.mPlaneAxis1 = f.connected.axes.secondary;
    settings.mPlaneAxis2 = f.owner.axes.secondary;
    settings.mSwingType = SwingType(s.swingType);
    return settings.Create(connected, owner);
}
JPH::Ref<Native> CreateNative(const SixDOFJointSettings &, const Frames &f, JPH::Body &connected, JPH::Body &owner)
{
    JPH::SixDOFConstraintSettings settings;
    settings.mPosition1 = f.connected.point;
    settings.mPosition2 = f.owner.point;
    settings.mAxisX1 = f.connected.axes.primary;
    settings.mAxisX2 = f.owner.axes.primary;
    settings.mAxisY1 = f.connected.axes.secondary;
    settings.mAxisY2 = f.owner.axes.secondary;
    settings.mSwingType = JPH::ESwingType::Pyramid; // Supports asymmetric author limits on Y and Z.
    return settings.Create(connected, owner);
}

float HingeTarget(const Constraints::Hinge &s)
{
    return s.useLimits ? std::clamp(s.targetAngle, s.minimum, s.maximum)
                       : std::remainder(s.targetAngle, 2.0f * JPH::JPH_PI);
}

void Apply(Native &native, const Constraints::Hinge &s)
{
    auto &joint = static_cast<JPH::HingeConstraint &>(native);
    // Apply after construction so the supported [0, 0] range is also a native hinge.
    joint.SetLimits(s.useLimits ? s.minimum : -JPH::JPH_PI, s.useLimits ? s.maximum : JPH::JPH_PI);
    auto &motor = joint.GetMotorSettings();
    motor.mSpringSettings = Spring(s.stiffness, s.damping);
    motor.SetTorqueLimit(s.maxTorque);
    joint.SetTargetAngle(HingeTarget(s));
    const auto state = s.motorEnabled ? JPH::EMotorState::Position : JPH::EMotorState::Off;
    if (joint.GetMotorState() != state) {
        joint.SetMotorState(state);
        joint.ResetWarmStart();
    }
    joint.SetViscousDamping(s.jointDamping);
    joint.SetMaxFrictionTorque(s.frictionTorque);
    if (joint.GetArmature() != s.armature)
        joint.SetArmature(s.armature);
}
void Apply(Native &native, const Constraints::Slider &s)
{
    static_cast<JPH::SliderConstraint &>(native).SetLimits(s.useLimits ? s.minimum : -FLT_MAX,
                                                           s.useLimits ? s.maximum : FLT_MAX);
}
void Apply(Native &, const Constraints::Fixed &)
{
}
void Apply(Native &native, const Constraints::Distance &s)
{
    auto &joint = static_cast<JPH::DistanceConstraint &>(native);
    if (!s.autoDistance)
        joint.SetDistance(s.minimum, s.maximum);
    joint.SetLimitsSpringSettings(Spring(s.stiffness, s.stiffness > 0.0f ? s.damping : 0.0f));
}
void Apply(Native &native, const SwingTwistJointSettings &s)
{
    auto &joint = static_cast<JPH::SwingTwistConstraint &>(native);
    joint.SetPlaneHalfConeAngle(glm::radians(s.planeHalfConeAngle));
    joint.SetNormalHalfConeAngle(glm::radians(s.normalHalfConeAngle));
    const float lower = glm::radians(s.minimumTwistAngle), upper = glm::radians(s.maximumTwistAngle);
    // Each setter validates the interval, including when moving to a disjoint range.
    if (lower > joint.GetTwistMaxAngle()) {
        joint.SetTwistMaxAngle(upper);
        joint.SetTwistMinAngle(lower);
    } else {
        joint.SetTwistMinAngle(lower);
        joint.SetTwistMaxAngle(upper);
    }
    joint.SetMaxFrictionTorque(s.frictionTorque);
    auto &swing = joint.GetSwingMotorSettings(), &twist = joint.GetTwistMotorSettings();
    swing.mSpringSettings = Spring(s.swingMotorStiffness, s.swingMotorDamping);
    swing.SetTorqueLimit(s.swingMotorMaxTorque);
    twist.mSpringSettings = Spring(s.twistMotorStiffness, s.twistMotorDamping);
    twist.SetTorqueLimit(s.twistMotorMaxTorque);
    joint.SetTargetOrientationCS(Orientation(s.targetOrientation));
    joint.SetTargetAngularVelocityCS(Vector(glm::radians(s.targetAngularVelocity)));
    joint.SetSwingMotorState(MotorState(s.swingMotorMode));
    joint.SetTwistMotorState(MotorState(s.twistMotorMode));
}

void Apply(Native &native, const SixDOFJointSettings &s)
{
    auto &joint = static_cast<JPH::SixDOFConstraint &>(native);
    using Axis = JPH::SixDOFConstraintSettings::EAxis;
    const auto motions = SixMotions(s);
    const auto motors = SixMotors(s);
    auto linearMin = s.linearLimitMinimum, linearMax = s.linearLimitMaximum;
    auto angularMin = glm::radians(s.angularLimitMinimum), angularMax = glm::radians(s.angularLimitMaximum);
    for (int i = 0; i < 3; ++i) {
        const auto applyMotion = [](JointMotion mode, float extent, float &lo, float &hi) {
            if (mode == JointMotion::Locked)
                lo = hi = 0.0f;
            else if (mode == JointMotion::Free) {
                lo = -extent;
                hi = extent;
            }
        };
        applyMotion(motions[i], FLT_MAX, linearMin[i], linearMax[i]);
        applyMotion(motions[i + 3], JPH::JPH_PI, angularMin[i], angularMax[i]);
    }
    joint.SetTranslationLimits(Vector(linearMin), Vector(linearMax));
    joint.SetRotationLimits(Vector(angularMin), Vector(angularMax));
    for (int i = 0; i < 3; ++i) {
        const auto linear = static_cast<Axis>(i), angular = static_cast<Axis>(i + 3);
        joint.SetLimitsSpringSettings(
            linear,
            Spring(s.linearLimitStiffness[i], s.linearLimitStiffness[i] > 0.0f ? s.linearLimitDamping[i] : 0.0f));
        joint.SetMaxFriction(linear, s.linearFrictionForce[i]);
        joint.SetMaxFriction(angular, s.angularFrictionTorque[i]);
        auto &lm = joint.GetMotorSettings(linear), &am = joint.GetMotorSettings(angular);
        lm.mSpringSettings = Spring(s.linearMotorStiffness[i], s.linearMotorDamping[i]);
        lm.SetForceLimit(s.linearMotorMaxForce[i]);
        am.mSpringSettings = Spring(s.angularMotorStiffness[i], s.angularMotorDamping[i]);
        am.SetTorqueLimit(s.angularMotorMaxTorque[i]);
        joint.SetMotorState(linear, MotorState(motors[i]));
        joint.SetMotorState(angular, MotorState(motors[i + 3]));
    }
    joint.SetTargetPositionCS(Vector(s.targetPosition));
    joint.SetTargetVelocityCS(Vector(s.targetVelocity));
    joint.SetTargetOrientationCS(Orientation(s.targetOrientation));
    joint.SetTargetAngularVelocityCS(Vector(glm::radians(s.targetAngularVelocity)));
}

bool SameSettings(const Constraints::Settings &left, const Constraints::Settings &right)
{
    return std::visit(
        [&](const auto &current) {
            using T = std::decay_t<decltype(current)>;
            const auto *next = std::get_if<T>(&right);
            if (!next)
                return false;
            const auto fields = [](const T &s) {
                if constexpr (std::is_same_v<T, Constraints::Hinge>)
                    return std::tie(s.axis, s.useLimits, s.minimum, s.maximum, s.motorEnabled, s.targetAngle,
                                    s.stiffness, s.damping, s.maxTorque, s.jointDamping, s.frictionTorque, s.armature);
                else if constexpr (std::is_same_v<T, Constraints::Slider>)
                    return std::tie(s.axis, s.useLimits, s.minimum, s.maximum);
                else if constexpr (std::is_same_v<T, Constraints::Distance>)
                    return std::tie(s.autoDistance, s.minimum, s.maximum, s.stiffness, s.damping);
                else if constexpr (std::is_same_v<T, SwingTwistJointSettings>)
                    return std::tie(s.twistAxis, s.planeAxis, s.swingType, s.normalHalfConeAngle, s.planeHalfConeAngle,
                                    s.minimumTwistAngle, s.maximumTwistAngle, s.frictionTorque, s.swingMotorMode,
                                    s.twistMotorMode, s.swingMotorStiffness, s.swingMotorDamping, s.swingMotorMaxTorque,
                                    s.twistMotorStiffness, s.twistMotorDamping, s.twistMotorMaxTorque,
                                    s.targetOrientation, s.targetAngularVelocity);
                else if constexpr (std::is_same_v<T, SixDOFJointSettings>)
                    return std::tie(
                        s.axis, s.secondaryAxis, s.xMotion, s.yMotion, s.zMotion, s.angularXMotion, s.angularYMotion,
                        s.angularZMotion, s.linearLimitMinimum, s.linearLimitMaximum, s.angularLimitMinimum,
                        s.angularLimitMaximum, s.linearLimitStiffness, s.linearLimitDamping, s.linearFrictionForce,
                        s.angularFrictionTorque, s.linearMotorStiffness, s.linearMotorDamping, s.linearMotorMaxForce,
                        s.angularMotorStiffness, s.angularMotorDamping, s.angularMotorMaxTorque, s.xMotorMode,
                        s.yMotorMode, s.zMotorMode, s.angularXMotorMode, s.angularYMotorMode, s.angularZMotorMode,
                        s.targetPosition, s.targetOrientation, s.targetVelocity, s.targetAngularVelocity);
                else
                    return std::tie(); // Fixed has no settings fields.
            };
            return fields(current) == fields(*next);
        },
        left);
}

SubType Type(const Constraints::Settings &settings)
{
    constexpr std::array kinds{SubType::Hinge,    SubType::Slider,     SubType::Fixed,
                               SubType::Distance, SubType::SwingTwist, SubType::SixDOF};
    return kinds[settings.index()];
}

uint64_t PairKey(uint32_t a, uint32_t b)
{
    return (uint64_t(std::min(a, b)) << 32) | std::max(a, b);
}

bool EndpointsResident(const Native &joint)
{
    const auto *connected = joint.GetBody1();
    return joint.GetBody2()->IsInBroadPhase() &&
           (connected == &JPH::Body::sFixedToWorld || connected->IsInBroadPhase());
}

JPH::RVec3 Anchor(const JPH::Body &body, JPH::Vec3Arg localPoint)
{
    return body.GetPosition() + body.GetRotation() * localPoint;
}
JPH::RVec3 Anchor(const JPH::Body &body, const JPH::Mat44 &localFrame)
{
    return body.GetCenterOfMassPosition() + body.GetRotation() * localFrame.GetTranslation();
}
glm::quat ToGlm(JPH::QuatArg rotation)
{
    return {rotation.GetW(), rotation.GetX(), rotation.GetY(), rotation.GetZ()};
}

template <class T> const T &As(const Native &constraint, SubType type)
{
    if (constraint.GetSubType() != type)
        throw std::logic_error("CPU joint does not have the requested motion output");
    return static_cast<const T &>(constraint);
}
} // namespace

void JoltCpuConstraints::ValidateAuthor(const Binding &binding, const Settings &settings)
{
    if (binding.ownerBody == WorldBody || binding.ownerBody == binding.connectedBody)
        throw std::invalid_argument("CPU joint requires a real owner and distinct endpoints");
    (void)Vector(binding.ownerAnchor);
    (void)Vector(binding.ownerLocalAnchor);
    (void)Vector(binding.connectedAnchor);
    (void)Vector(binding.connectedLocalAnchor);
    (void)std::visit([](const auto &s) { return Validate(s); }, settings);
}

void JoltCpuConstraints::ValidateAuthorUpdate(const Settings &from, const Settings &to)
{
    if (from.index() != to.index())
        throw std::logic_error("CPU joint changes kind only when rebound");
    const auto axes = [](const auto &value) { return std::visit([](const auto &s) { return Validate(s); }, value); };
    const auto before = axes(from), after = axes(to);
    if (!before.primary.IsClose(after.primary) || !before.secondary.IsClose(after.secondary))
        throw std::logic_error("CPU joint frame changes require rebinding");
    if (const auto *distance = std::get_if<Distance>(&to);
        distance && distance->autoDistance != std::get<Distance>(from).autoDistance)
        throw std::logic_error("Changing Distance auto mode requires rebinding");
}

JoltCpuConstraints::~JoltCpuConstraints()
{
    for (const auto &[id, joint] : m_joints)
        m_scene.System().RemoveConstraint(joint.native.GetPtr());
}

JoltCpuConstraints::CapturedAnchor JoltCpuConstraints::CaptureAnchor(JPH::Vec3Arg baked, JPH::Vec3Arg scale,
                                                                     JPH::Vec3Arg authored)
{
    // Match jph_joint_unscale. For a collapsed axis keep the accepted baked
    // offset and recover the raw author coordinate when it becomes nonzero.
    CapturedAnchor anchor{authored, JPH::Vec3::sZero()};
    for (JPH::uint axis = 0; axis < 3; ++axis) {
        if (scale[axis] == 0.0f) {
            anchor.collapsed.SetComponent(axis, baked[axis]);
        } else {
            const float coordinate = baked[axis] / scale[axis];
            if (!std::isfinite(coordinate))
                throw std::invalid_argument("Joint anchor is not representable at the accepted scale");
            anchor.unscaled.SetComponent(axis, coordinate);
        }
    }
    return anchor;
}

void JoltCpuConstraints::Create(uint64_t id, const Binding &binding, const Settings &settings, bool enableCollision)
{
    if (id == 0 || m_joints.find(id) != m_joints.end())
        throw std::logic_error("CPU joint identity is invalid or already owned");
    if (binding.ownerBody == WorldBody || binding.ownerBody == binding.connectedBody)
        throw std::invalid_argument("CPU joint requires a real owner and distinct endpoints");
    const auto axes = std::visit([](const auto &s) { return Validate(s); }, settings);
    const auto ownerAnchor = Vector(binding.ownerAnchor);
    const bool world = binding.connectedBody == WorldBody;
    const auto *distance = std::get_if<Distance>(&settings);
    const auto connectedAnchor = distance ? Vector(binding.connectedAnchor) : JPH::Vec3::sZero();
    const auto ownerAnchorCapture =
        CaptureAnchor(ownerAnchor, m_scene.m_bodies.at(binding.ownerBody).rootScale, Vector(binding.ownerLocalAnchor));
    CapturedAnchor connectedAnchorCapture;
    const JPH::BodyID bodies[] = {world ? JPH::BodyID() : m_scene.NativeBody(binding.connectedBody),
                                  m_scene.NativeBody(binding.ownerBody)};
    auto &system = m_scene.System();
    JPH::Ref<Native> native;
    {
        JPH::BodyLockMultiWrite lock(system.GetBodyLockInterface(), bodies, 2);
        auto *owner = lock.GetBody(1);
        auto *connected = world ? &JPH::Body::sFixedToWorld : lock.GetBody(0);
        if (!owner || !connected)
            throw std::logic_error("CPU joint requires live native endpoints");
        const auto rotation = owner->GetRotation();
        const Axes worldAxes{rotation * axes.primary, rotation * axes.secondary};
        const auto point = Anchor(*owner, ownerAnchor);
        const auto connectedPoint =
            distance ? (world ? JPH::RVec3(connectedAnchor) : Anchor(*connected, connectedAnchor)) : point;
        if (!world) {
            const auto &record = m_scene.m_bodies.at(binding.connectedBody);
            const auto offset =
                distance
                    ? connectedAnchor
                    : connected->GetRotation().InverseRotate(JPH::Vec3(point - connected->GetCenterOfMassPosition())) +
                          record.shape->GetCenterOfMass();
            connectedAnchorCapture = CaptureAnchor(
                offset, record.rootScale, distance ? Vector(binding.connectedLocalAnchor) : JPH::Vec3::sZero());
        }
        const Frames frames{{connectedPoint, worldAxes}, {point, worldAxes}};
        // The connected endpoint is reference frame 1; positive motion is the owner relative to it.
        native = std::visit([&](const auto &s) { return CreateNative(s, frames, *connected, *owner); }, settings);
        native->SetUserData(id);
        native->SetEnabled(EndpointsResident(*native.GetPtr()));
        std::visit([&](const auto &s) { Apply(*native.GetPtr(), s); }, settings);
    }
    const auto *swing = std::get_if<SwingTwistJointSettings>(&settings);
    const auto entry = m_joints
                           .try_emplace(id, Record{native,
                                                   binding.ownerBody,
                                                   binding.connectedBody,
                                                   ownerAnchorCapture,
                                                   connectedAnchorCapture,
                                                   axes.primary,
                                                   axes.secondary,
                                                   true,
                                                   distance && distance->autoDistance,
                                                   swing ? swing->swingType : JointSwingType::Cone,
                                                   settings,
                                                   {}})
                           .first;
    bool changed = false;
    try {
        m_byBody.emplace(binding.ownerBody, id);
        if (!world)
            m_byBody.emplace(binding.connectedBody, id);
        changed = ChangeCollision(entry->second, enableCollision);
        system.AddConstraint(native.GetPtr());
    } catch (...) {
        ChangeCollision(entry->second, true);
        EraseBodyIndex(binding.ownerBody, id);
        EraseBodyIndex(binding.connectedBody, id);
        m_joints.erase(entry);
        throw;
    }
    WakeEndpoints(entry->second, changed);
}

bool JoltCpuConstraints::SupportsPoseDrive(const Settings &settings) noexcept
{
    return std::holds_alternative<Hinge>(settings) || std::holds_alternative<SwingTwistJointSettings>(settings) ||
           std::holds_alternative<SixDOFJointSettings>(settings);
}

void JoltCpuConstraints::ApplyPoseDrive(Record &joint)
{
    const auto &relative = joint.poseDriveRelative;
    // Native body 1 is the connected parent and 2 is the owner child. The
    // native conversion includes the original constraint frames on both bodies.
    if (const auto *hinge = std::get_if<Hinge>(&joint.acceptedSettings)) {
        auto &native = static_cast<JPH::HingeConstraint &>(*joint.native.GetPtr());
        if (relative)
            native.SetTargetOrientationBS(*relative);
        else
            native.SetTargetAngle(HingeTarget(*hinge));
    } else if (const auto *swing = std::get_if<SwingTwistJointSettings>(&joint.acceptedSettings)) {
        auto &native = static_cast<JPH::SwingTwistConstraint &>(*joint.native.GetPtr());
        if (relative)
            native.SetTargetOrientationBS(*relative);
        else
            native.SetTargetOrientationCS(Orientation(swing->targetOrientation));
    } else if (const auto *six = std::get_if<SixDOFJointSettings>(&joint.acceptedSettings)) {
        auto &native = static_cast<JPH::SixDOFConstraint &>(*joint.native.GetPtr());
        if (relative)
            native.SetTargetOrientationBS(*relative);
        else
            native.SetTargetOrientationCS(Orientation(six->targetOrientation));
    }
}

void JoltCpuConstraints::SetPoseDrive(uint64_t id, const std::optional<JPH::Quat> &relative)
{
    auto &joint = m_joints.at(id);
    if (!SupportsPoseDrive(joint.acceptedSettings) || (relative && !relative->IsNormalized()))
        throw std::invalid_argument("CPU joint pose drive requires an angular joint and normalized target");
    const auto &previous = joint.poseDriveRelative;
    if ((!previous && !relative) || (previous && relative && (*previous == *relative || *previous == -*relative)))
        return;
    joint.poseDriveRelative = relative;
    ApplyPoseDrive(joint);
    WakeEndpoints(joint, false);
}

void JoltCpuConstraints::Update(uint64_t id, const Settings &settings, bool enableCollision, bool clearPoseDrive)
{
    auto &joint = m_joints.at(id);
    if (joint.collisionEnabled == enableCollision && SameSettings(joint.acceptedSettings, settings)) {
        if (clearPoseDrive && joint.poseDriveRelative)
            SetPoseDrive(id, std::nullopt);
        return;
    }
    if (joint.native->GetSubType() != Type(settings))
        throw std::logic_error("CPU joint changes kind only when rebound");
    const auto axes = std::visit([](const auto &s) { return Validate(s); }, settings);
    if (!axes.primary.IsClose(joint.primary) || !axes.secondary.IsClose(joint.secondary))
        throw std::logic_error("CPU joint frame changes require rebinding");
    if (const auto *distance = std::get_if<Distance>(&settings);
        distance && distance->autoDistance != joint.autoDistance)
        throw std::logic_error("Changing Distance auto mode requires rebinding");

    JPH::Ref<Native> replacement;
    auto &system = m_scene.System();
    const auto *swing = std::get_if<SwingTwistJointSettings>(&settings);
    if (swing && joint.swingType != swing->swingType) {
        const auto frame = [](const JPH::Mat44 &local) {
            return Frame{JPH::RVec3(local.GetTranslation()), {local.GetAxisX(), local.GetAxisZ()}};
        };
        // Native Jolt binds the cone/pyramid choice at construction. The current
        // COM-local frames already contain the original binding and any COM rebase.
        const Frames frames{frame(joint.native->GetConstraintToBody1Matrix()),
                            frame(joint.native->GetConstraintToBody2Matrix()), JPH::EConstraintSpace::LocalToBodyCOM};
        replacement = CreateNative(*swing, frames, *joint.native->GetBody1(), *joint.native->GetBody2());
        replacement->SetUserData(id);
        Apply(*replacement.GetPtr(), *swing);
        replacement->SetEnabled(joint.native->GetEnabled());
        system.AddConstraint(replacement.GetPtr());
    }
    bool changed = false;
    try {
        changed = ChangeCollision(joint, enableCollision);
    } catch (...) {
        if (replacement.GetPtr())
            system.RemoveConstraint(replacement.GetPtr());
        throw;
    }
    if (replacement.GetPtr()) {
        system.RemoveConstraint(joint.native.GetPtr());
        joint.native = std::move(replacement);
        joint.swingType = swing->swingType;
    } else {
        std::visit([&](const auto &s) { Apply(*joint.native.GetPtr(), s); }, settings);
    }
    joint.acceptedSettings = settings;
    if (clearPoseDrive)
        joint.poseDriveRelative.reset();
    if (joint.poseDriveRelative)
        ApplyPoseDrive(joint);
    WakeEndpoints(joint, changed);
}

void JoltCpuConstraints::SetCollisionEnabled(uint64_t id, bool enabled)
{
    auto &joint = m_joints.at(id);
    if (ChangeCollision(joint, enabled))
        WakeEndpoints(joint, true);
}

bool JoltCpuConstraints::Retire(uint64_t id)
{
    const auto found = m_joints.find(id);
    if (found == m_joints.end())
        return false;
    auto &joint = found->second;
    const bool changed = ChangeCollision(joint, true);
    m_scene.System().RemoveConstraint(joint.native.GetPtr());
    WakeEndpoints(joint, changed);
    EraseBodyIndex(joint.ownerBody, id);
    EraseBodyIndex(joint.connectedBody, id);
    m_joints.erase(found);
    return true;
}

std::vector<uint64_t> JoltCpuConstraints::RetireBody(uint32_t bodyId)
{
    const auto [begin, end] = m_byBody.equal_range(bodyId);
    std::vector<uint64_t> ids;
    for (auto it = begin; it != end; ++it)
        ids.push_back(it->second);
    std::sort(ids.begin(), ids.end());
    for (const auto id : ids)
        Retire(id);
    return ids;
}

void JoltCpuConstraints::OnBodyResidenceChanged(uint32_t bodyId)
{
    const auto [begin, end] = m_byBody.equal_range(bodyId);
    for (auto it = begin; it != end; ++it) {
        auto &joint = *m_joints.at(it->second).native.GetPtr();
        joint.SetEnabled(EndpointsResident(joint));
    }
}

void JoltCpuConstraints::ResetBodyWarmStart(uint32_t bodyId)
{
    const auto [begin, end] = m_byBody.equal_range(bodyId);
    for (auto it = begin; it != end; ++it)
        m_joints.at(it->second).native->ResetWarmStart();
}

bool JoltCpuConstraints::HasBodyConstraints(uint32_t bodyId) const
{
    return m_byBody.find(bodyId) != m_byBody.end();
}

void JoltCpuConstraints::NotifyBodyShapeChanged(uint32_t bodyId, JPH::Vec3Arg deltaCOM, bool scaleChanged)
{
    if (!scaleChanged && deltaCOM == JPH::Vec3::sZero())
        return;
    const auto [begin, end] = m_byBody.equal_range(bodyId);
    if (begin == end)
        return;
    const auto &record = m_scene.m_bodies.at(bodyId);
    for (auto it = begin; it != end; ++it) {
        const auto &joint = m_joints.at(it->second);
        auto adjustment = deltaCOM;
        if (scaleChanged) {
            const bool owner = joint.ownerBody == bodyId;
            const auto &anchor = owner ? joint.ownerAnchor : joint.connectedAnchor;
            const auto frame =
                owner ? joint.native->GetConstraintToBody2Matrix() : joint.native->GetConstraintToBody1Matrix();
            const auto local = anchor.unscaled * record.rootScale + anchor.collapsed - record.shape->GetCenterOfMass();
            // Native NotifyShapeChanged subtracts its argument from the current
            // COM-local anchor. Keep the angular frame and auto-distance state.
            adjustment = frame.GetTranslation() - local;
        }
        joint.native->NotifyShapeChanged(record.native, adjustment);
    }
}

void JoltCpuConstraints::EraseBodyIndex(uint32_t bodyId, uint64_t id)
{
    const auto [begin, end] = m_byBody.equal_range(bodyId);
    for (auto it = begin; it != end; ++it)
        if (it->second == id) {
            m_byBody.erase(it);
            return;
        }
}

bool JoltCpuConstraints::ChangeCollision(Record &joint, bool enabled)
{
    if (joint.collisionEnabled == enabled)
        return false;
    bool changed = false;
    if (joint.connectedBody != WorldBody) {
        const auto key = PairKey(joint.ownerBody, joint.connectedBody);
        if (enabled) {
            const auto pair = m_suppressedPairs.find(key);
            if (--pair->second == 0) {
                m_suppressedPairs.erase(pair);
                changed = true;
            }
        } else {
            auto [pair, inserted] = m_suppressedPairs.try_emplace(key, 0);
            ++pair->second;
            changed = inserted;
        }
    }
    joint.collisionEnabled = enabled;
    return changed;
}

bool JoltCpuConstraints::SuppressesCollision(uint32_t bodyA, uint32_t bodyB) const
{
    return !m_suppressedPairs.empty() && m_suppressedPairs.find(PairKey(bodyA, bodyB)) != m_suppressedPairs.end();
}

void JoltCpuConstraints::WakeEndpoints(const Record &joint, bool invalidateContacts)
{
    auto &bodies = m_scene.System().GetBodyInterface();
    for (const auto id : {joint.connectedBody, joint.ownerBody}) {
        if (id == WorldBody)
            continue;
        if (invalidateContacts)
            bodies.InvalidateContactCache(m_scene.NativeBody(id));
        m_scene.SetBodyAwake(id, true);
    }
}

glm::vec2 JoltCpuConstraints::ReadHinge(uint64_t id) const
{
    const auto &joint = As<JPH::HingeConstraint>(*m_joints.at(id).native.GetPtr(), SubType::Hinge);
    const auto *connected = joint.GetBody1(), *owner = joint.GetBody2();
    const auto axis = owner->GetRotation() * joint.GetConstraintToBody2Matrix().GetAxisX();
    return {joint.GetCurrentAngle(), axis.Dot(owner->GetAngularVelocity() - connected->GetAngularVelocity())};
}

float JoltCpuConstraints::ReadSliderPosition(uint64_t id) const
{
    return As<JPH::SliderConstraint>(*m_joints.at(id).native.GetPtr(), SubType::Slider).GetCurrentPosition();
}

float JoltCpuConstraints::ReadDistance(uint64_t id) const
{
    const auto &joint = As<JPH::DistanceConstraint>(*m_joints.at(id).native.GetPtr(), SubType::Distance);
    return JPH::Vec3(Anchor(*joint.GetBody2(), joint.GetConstraintToBody2Matrix()) -
                     Anchor(*joint.GetBody1(), joint.GetConstraintToBody1Matrix()))
        .Length();
}

glm::quat JoltCpuConstraints::ReadOrientation(uint64_t id) const
{
    const auto &native = *m_joints.at(id).native.GetPtr();
    if (native.GetSubType() == SubType::SwingTwist)
        return ToGlm(static_cast<const JPH::SwingTwistConstraint &>(native).GetRotationInConstraintSpace());
    return ToGlm(As<JPH::SixDOFConstraint>(native, SubType::SixDOF).GetRotationInConstraintSpace());
}

glm::vec3 JoltCpuConstraints::ReadSixDOFPosition(uint64_t id) const
{
    const auto &joint = As<JPH::SixDOFConstraint>(*m_joints.at(id).native.GetPtr(), SubType::SixDOF);
    const auto &connected = *joint.GetBody1();
    const auto frame = joint.GetConstraintToBody1Matrix();
    const JPH::Vec3 delta(Anchor(*joint.GetBody2(), joint.GetConstraintToBody2Matrix()) - Anchor(connected, frame));
    const auto position = (connected.GetRotation() * frame.GetQuaternion()).InverseRotate(delta);
    return {position.GetX(), position.GetY(), position.GetZ()};
}
} // namespace infernux
