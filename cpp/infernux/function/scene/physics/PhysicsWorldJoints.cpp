// Native Jolt constraint creation, mutation and endpoint lifetime.
#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyLockMulti.h>
#include <Jolt/Physics/Constraints/DistanceConstraint.h>
#include <Jolt/Physics/Constraints/FixedConstraint.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/SixDOFConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>
#include <Jolt/Physics/Constraints/SwingTwistConstraint.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include "../Collider.h"
#include "../GameObject.h"
#include "../HingeJoint.h"
#include "../Transform.h"
#include "JointSettings.h"
#include "PhysicsContactListener.h"
#include "PhysicsWorld.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <glm/gtc/constants.hpp>
#include <stdexcept>

namespace infernux
{
namespace
{
bool IsFinite(const glm::vec3 &v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

JPH::Vec3 ToJolt(const glm::vec3 &v)
{
    return {v.x, v.y, v.z};
}
JPH::Quat ToJolt(const glm::vec4 &v)
{
    return {v.x, v.y, v.z, v.w};
}

bool UpdateDrive(JPH::MotorSettings &drive, float stiffness, float damping, float maximum, bool linear = false)
{
    const float lower = linear ? drive.mMinForceLimit : drive.mMinTorqueLimit;
    const float upper = linear ? drive.mMaxForceLimit : drive.mMaxTorqueLimit;
    const bool changed = drive.mSpringSettings.mMode != JPH::ESpringMode::StiffnessAndDamping ||
                         drive.mSpringSettings.mStiffness != stiffness || drive.mSpringSettings.mDamping != damping ||
                         lower != -maximum || upper != maximum;
    if (changed) {
        drive.mSpringSettings = {JPH::ESpringMode::StiffnessAndDamping, stiffness, damping};
        if (linear)
            drive.SetForceLimit(maximum);
        else
            drive.SetTorqueLimit(maximum);
    }
    return changed;
}

void CopySwingTwistMotorState(const JPH::SwingTwistConstraint &previous, JPH::SwingTwistConstraint &next)
{
    next.SetTargetOrientationCS(previous.GetTargetOrientationCS());
    next.SetTargetAngularVelocityCS(previous.GetTargetAngularVelocityCS());
    next.SetSwingMotorState(previous.GetSwingMotorState());
    next.SetTwistMotorState(previous.GetTwistMotorState());
}

void CopySixDOFMotorState(const JPH::SixDOFConstraint &previous, JPH::SixDOFConstraint &next)
{
    next.SetTargetPositionCS(previous.GetTargetPositionCS());
    next.SetTargetOrientationCS(previous.GetTargetOrientationCS());
    next.SetTargetVelocityCS(previous.GetTargetVelocityCS());
    next.SetTargetAngularVelocityCS(previous.GetTargetAngularVelocityCS());
    for (int i = 0; i < JPH::SixDOFConstraintSettings::EAxis::Num; ++i) {
        const auto axis = static_cast<JPH::SixDOFConstraintSettings::EAxis>(i);
        next.SetMotorState(axis, previous.GetMotorState(axis));
    }
}

void SixDOFLimits(const SixDOFJointSettings &s, std::array<float, 6> &minimum, std::array<float, 6> &maximum)
{
    const JointMotion modes[] = {s.xMotion, s.yMotion, s.zMotion, s.angularXMotion, s.angularYMotion, s.angularZMotion};
    for (int i = 0; i < 6; ++i) {
        switch (modes[i]) {
        case JointMotion::Free:
            minimum[i] = i < 3 ? -FLT_MAX : -JPH::JPH_PI;
            maximum[i] = i < 3 ? FLT_MAX : JPH::JPH_PI;
            break;
        // Compare and publish Jolt's actual normalized representation. Its
        // constructor converts fixed-axis sentinels to zero and angular free
        // sentinels to +/-pi; resending sentinels would clear warm-start every update.
        case JointMotion::Locked:
            minimum[i] = maximum[i] = 0.0f;
            break;
        case JointMotion::Limited:
            minimum[i] = i < 3 ? s.linearLimitMinimum[i] : glm::radians(s.angularLimitMinimum[i - 3]);
            maximum[i] = i < 3 ? s.linearLimitMaximum[i] : glm::radians(s.angularLimitMaximum[i - 3]);
            break;
        }
    }
}
} // namespace

uint64_t PhysicsWorld::CreateHingeConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                             const glm::vec3 &worldAnchor, const glm::vec3 &worldAxis,
                                             const glm::vec3 &localAnchor, bool useLimits, float minimumAngle,
                                             float maximumAngle, bool enableCollision)
{
    if (!IsFinite(worldAnchor) || !IsFinite(worldAxis) || glm::dot(worldAxis, worldAxis) <= 1e-12f)
        throw std::invalid_argument("hinge anchor and axis must be finite and the axis must be non-zero");
    if (!std::isfinite(minimumAngle) || !std::isfinite(maximumAngle) || minimumAngle > 0.0f || maximumAngle < 0.0f ||
        minimumAngle > maximumAngle || minimumAngle < -glm::pi<float>() || maximumAngle > glm::pi<float>())
        throw std::invalid_argument("hinge limits must be ordered radians within [-pi, pi] and span zero");

    const glm::vec3 axis = glm::normalize(worldAxis);
    const glm::vec3 reference = std::abs(axis.y) < 0.9f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    const glm::vec3 normal = glm::normalize(glm::cross(axis, reference));

    JPH::HingeConstraintSettings settings;
    settings.mSpace = JPH::EConstraintSpace::WorldSpace;
    settings.mPoint1 = settings.mPoint2 = JPH::RVec3(worldAnchor.x, worldAnchor.y, worldAnchor.z);
    settings.mHingeAxis1 = settings.mHingeAxis2 = JPH::Vec3(axis.x, axis.y, axis.z);
    settings.mNormalAxis1 = settings.mNormalAxis2 = JPH::Vec3(normal.x, normal.y, normal.z);
    settings.mLimitsMin = useLimits ? minimumAngle : -glm::pi<float>();
    settings.mLimitsMax = useLimits ? maximumAngle : glm::pi<float>();

    return CreateTwoBodyConstraint(owner, bodyIdA, bodyIdB, settings, localAnchor, enableCollision);
}

uint64_t PhysicsWorld::CreateSliderConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                              const glm::vec3 &worldAnchor, const glm::vec3 &worldAxis,
                                              const glm::vec3 &localAnchor, bool useLimits, float minimumDistance,
                                              float maximumDistance, bool enableCollision)
{
    if (!IsFinite(worldAnchor) || !IsFinite(worldAxis) || glm::dot(worldAxis, worldAxis) <= 1e-12f)
        throw std::invalid_argument("slider anchor and axis must be finite and the axis must be non-zero");
    if (!std::isfinite(minimumDistance) || !std::isfinite(maximumDistance) || minimumDistance > 0.0f ||
        maximumDistance < 0.0f || minimumDistance > maximumDistance)
        throw std::invalid_argument("slider limits must be ordered metres and span zero");

    const glm::vec3 axis = glm::normalize(worldAxis);
    JPH::SliderConstraintSettings settings;
    settings.mSpace = JPH::EConstraintSpace::WorldSpace;
    settings.mPoint1 = settings.mPoint2 = JPH::RVec3(worldAnchor.x, worldAnchor.y, worldAnchor.z);
    settings.SetSliderAxis(JPH::Vec3(axis.x, axis.y, axis.z));
    settings.mLimitsMin = useLimits ? minimumDistance : -FLT_MAX;
    settings.mLimitsMax = useLimits ? maximumDistance : FLT_MAX;

    return CreateTwoBodyConstraint(owner, bodyIdA, bodyIdB, settings, localAnchor, enableCollision);
}

uint64_t PhysicsWorld::CreateTwoBodyConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                               const JPH::TwoBodyConstraintSettings &settings,
                                               const glm::vec3 &localAnchor, bool enableCollision,
                                               const glm::vec3 &connectedLocalAnchor)
{
    if (!m_initialized || !m_physicsSystem)
        throw std::logic_error("constraint creation requires an initialized physics world");
    if (bodyIdA == 0xFFFFFFFF || bodyIdA == bodyIdB)
        throw std::invalid_argument("constraint requires distinct participating bodies");
    JPH::BodyID ids[2] = {JPH::BodyID(bodyIdA), JPH::BodyID(bodyIdB)};
    JPH::TwoBodyConstraint *constraint;
    if (bodyIdB == 0xFFFFFFFF) {
        JPH::BodyLockWrite lock(m_physicsSystem->GetBodyLockInterface(), ids[0]);
        if (!lock.Succeeded())
            throw std::logic_error("constraint body no longer exists");
        constraint = settings.Create(JPH::Body::sFixedToWorld, lock.GetBody());
    } else {
        JPH::BodyLockMultiWrite lock(m_physicsSystem->GetBodyLockInterface(), ids, 2);
        auto *bodyA = lock.GetBody(0);
        auto *bodyB = lock.GetBody(1);
        if (!bodyA || !bodyB)
            throw std::logic_error("constraint endpoint no longer exists");
        constraint = settings.Create(*bodyB, *bodyA);
    }
    InvalidateRagdollsForBody(bodyIdA);
    InvalidateRagdollsForBody(bodyIdB);
    m_physicsSystem->AddConstraint(constraint);
    const auto id = m_nextConstraintId++;
    const bool suppress = bodyIdB != 0xFFFFFFFF && !enableCollision;
    ConstraintRecord record{owner, constraint, bodyIdA, bodyIdB, suppress};
    CaptureConstraintAnchors(record, localAnchor, connectedLocalAnchor);
    m_constraints.emplace(id, record);
    if (suppress)
        SetConstraintPairSuppressed(bodyIdA, bodyIdB, true);
    ActivateConstraintBodies(record);
    return id;
}

uint64_t PhysicsWorld::CreateFixedConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                             const glm::vec3 &worldAnchor, const glm::vec3 &localAnchor,
                                             bool enableCollision)
{
    JPH::FixedConstraintSettings settings;
    settings.mSpace = JPH::EConstraintSpace::WorldSpace;
    settings.mPoint1 = settings.mPoint2 = JPH::RVec3(worldAnchor.x, worldAnchor.y, worldAnchor.z);
    return CreateTwoBodyConstraint(owner, bodyIdA, bodyIdB, settings, localAnchor, enableCollision);
}

void PhysicsWorld::SetConstraintCollisionEnabled(uint64_t constraintId, bool enabled)
{
    auto &record = m_constraints.at(constraintId);
    UpdateConstraintCollision(record, enabled);
}

uint64_t PhysicsWorld::CreateDistanceConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                                const glm::vec3 &localAnchor, const glm::vec3 &connectedLocalAnchor,
                                                bool autoDistance, float minimum, float maximum, float stiffness,
                                                float damping, bool enableCollision)
{
    JPH::DistanceConstraintSettings settings;
    // Explicit local anchors belong to the bound body frame, even when its
    // Transform already contains a kinematic target for the next step.
    settings.mSpace = JPH::EConstraintSpace::LocalToBodyCOM;
    const auto localPoint = [&](uint32_t bodyId, const glm::vec3 &anchor) -> JPH::RVec3 {
        if (bodyId == 0xFFFFFFFF)
            return JPH::RVec3(anchor.x, anchor.y, anchor.z);
        const auto scale = m_bodies.at(bodyId).gameObject->GetTransform()->GetWorldScale();
        const auto scaled = anchor * scale;
        return JPH::RVec3(JPH::Vec3(scaled.x, scaled.y, scaled.z) -
                          m_physicsSystem->GetBodyInterface().GetShape(JPH::BodyID(bodyId))->GetCenterOfMass());
    };
    settings.mPoint1 = localPoint(bodyIdB, connectedLocalAnchor);
    settings.mPoint2 = localPoint(bodyIdA, localAnchor);
    // Jolt captures the length from these same actual body endpoints at creation.
    settings.mMinDistance = autoDistance ? -1.0f : minimum;
    settings.mMaxDistance = autoDistance ? -1.0f : maximum;
    settings.mLimitsSpringSettings = {JPH::ESpringMode::StiffnessAndDamping, stiffness, damping};
    return CreateTwoBodyConstraint(owner, bodyIdA, bodyIdB, settings, localAnchor, enableCollision,
                                   connectedLocalAnchor);
}

void PhysicsWorld::UpdateDistanceConstraintSettings(uint64_t id, bool updateLimits, float minimum, float maximum,
                                                    float stiffness, float damping, bool enableCollision)
{
    auto &record = m_constraints.at(id);
    if (record.constraint->GetSubType() != JPH::EConstraintSubType::Distance)
        throw std::logic_error("constraint is not a distance joint");
    auto *distance = static_cast<JPH::DistanceConstraint *>(record.constraint);
    bool changed = false;
    if (updateLimits && (distance->GetMinDistance() != minimum || distance->GetMaxDistance() != maximum)) {
        distance->SetDistance(minimum, maximum);
        changed = true;
    }
    const auto &spring = distance->GetLimitsSpringSettings();
    if (spring.mMode != JPH::ESpringMode::StiffnessAndDamping || spring.mStiffness != stiffness ||
        spring.mDamping != damping) {
        distance->SetLimitsSpringSettings({JPH::ESpringMode::StiffnessAndDamping, stiffness, damping});
        changed = true;
    }
    if (changed) {
        distance->ResetWarmStart();
        ActivateConstraintBodies(record);
    }
    UpdateConstraintCollision(record, enableCollision);
}

float PhysicsWorld::GetDistanceConstraintLength(uint64_t id) const
{
    const auto &record = m_constraints.at(id);
    if (record.constraint->GetSubType() != JPH::EConstraintSubType::Distance)
        throw std::logic_error("constraint is not a distance joint");
    const auto *joint = static_cast<const JPH::TwoBodyConstraint *>(record.constraint);
    const auto first =
        joint->GetBody1()->GetCenterOfMassTransform() * joint->GetConstraintToBody1Matrix().GetTranslation();
    const auto second =
        joint->GetBody2()->GetCenterOfMassTransform() * joint->GetConstraintToBody2Matrix().GetTranslation();
    return static_cast<float>((second - first).Length());
}

void PhysicsWorld::CaptureConstraintAnchors(ConstraintRecord &record, const glm::vec3 &localAnchor,
                                            const glm::vec3 &connectedLocalAnchor)
{
    const auto capture = [&](uint32_t bodyId, JPH::Vec3Arg point, glm::vec3 &anchor, glm::vec3 &scale,
                             glm::vec3 &collapsedOffset) {
        if (bodyId == 0xFFFFFFFF)
            return;
        const auto *transform = m_bodies.at(bodyId).gameObject->GetTransform();
        scale = transform->GetWorldScale();
        const auto offset =
            point + m_physicsSystem->GetBodyInterface().GetShape(JPH::BodyID(bodyId))->GetCenterOfMass();
        // Capture the actual bound physics frame: a kinematic body's visual
        // Transform may already contain its next target pose. A collapsed
        // scale axis has no inverse: retain that endpoint's authored coordinate
        // and the actual bound offset that cannot be represented by scaling.
        const auto &collapsed = bodyId == record.bodyIdA ? localAnchor : connectedLocalAnchor;
        for (int axis = 0; axis < 3; ++axis) {
            anchor[axis] = scale[axis] == 0.0f ? collapsed[axis] : offset[axis] / scale[axis];
            collapsedOffset[axis] = scale[axis] == 0.0f ? offset[axis] : 0.0f;
        }
    };
    const auto *joint = static_cast<JPH::TwoBodyConstraint *>(record.constraint);
    capture(record.bodyIdA, joint->GetConstraintToBody2Matrix().GetTranslation(), record.anchorA, record.scaleA,
            record.collapsedOffsetA);
    capture(record.bodyIdB, joint->GetConstraintToBody1Matrix().GetTranslation(), record.anchorB, record.scaleB,
            record.collapsedOffsetB);
}

void PhysicsWorld::UpdateConstraintScale(ConstraintRecord &record, uint32_t bodyId, const glm::vec3 &scale)
{
    const bool ownerChanged = bodyId == record.bodyIdA;
    auto &boundScale = ownerChanged ? record.scaleA : record.scaleB;
    if (scale == boundScale)
        return;

    // Retain both local attachment points and the original joint frames.
    // Recreating from a single current world point would move the other end
    // and reset the hinge's zero angle / slider's zero displacement.
    const glm::vec3 anchor = ownerChanged ? record.anchorA : record.anchorB;
    const glm::vec3 offset = anchor * scale + (ownerChanged ? record.collapsedOffsetA : record.collapsedOffsetB);
    auto settings = record.constraint->GetConstraintSettings();
    JPH::BodyID ids[2] = {JPH::BodyID(record.bodyIdA), JPH::BodyID(record.bodyIdB)};
    JPH::Constraint *replacement = nullptr;
    {
        JPH::BodyLockMultiWrite lock(m_physicsSystem->GetBodyLockInterface(), ids, 2);
        auto *bodyA = lock.GetBody(0);
        auto *bodyB = record.bodyIdB == 0xFFFFFFFF ? &JPH::Body::sFixedToWorld : lock.GetBody(1);
        if (!bodyA || !bodyB)
            throw std::logic_error("constraint scale publication requires both bound bodies");
        const auto *changedBody = ownerChanged ? bodyA : bodyB;
        const JPH::RVec3 point(JPH::Vec3(offset.x, offset.y, offset.z) - changedBody->GetShape()->GetCenterOfMass());
        if (record.constraint->GetSubType() == JPH::EConstraintSubType::Hinge) {
            auto *hinge = static_cast<JPH::HingeConstraintSettings *>(settings.GetPtr());
            (ownerChanged ? hinge->mPoint2 : hinge->mPoint1) = point;
            replacement = hinge->Create(*bodyB, *bodyA);
            // Settings contain the spring/torque limits, but runtime motor
            // state and targets are not part of GetConstraintSettings().
            const auto *previous = static_cast<JPH::HingeConstraint *>(record.constraint);
            auto *motor = static_cast<JPH::HingeConstraint *>(replacement);
            motor->SetTargetAngle(previous->GetTargetAngle());
            motor->SetTargetAngularVelocity(previous->GetTargetAngularVelocity());
            motor->SetMotorState(previous->GetMotorState());
            motor->SetRotorSpeed(previous->GetRotorSpeed());
        } else if (record.constraint->GetSubType() == JPH::EConstraintSubType::Slider) {
            auto *slider = static_cast<JPH::SliderConstraintSettings *>(settings.GetPtr());
            (ownerChanged ? slider->mPoint2 : slider->mPoint1) = point;
            replacement = slider->Create(*bodyB, *bodyA);
        } else if (record.constraint->GetSubType() == JPH::EConstraintSubType::Fixed) {
            auto *fixed = static_cast<JPH::FixedConstraintSettings *>(settings.GetPtr());
            (ownerChanged ? fixed->mPoint2 : fixed->mPoint1) = point;
            replacement = fixed->Create(*bodyB, *bodyA);
        } else if (record.constraint->GetSubType() == JPH::EConstraintSubType::Distance) {
            auto *distance = static_cast<JPH::DistanceConstraintSettings *>(settings.GetPtr());
            (ownerChanged ? distance->mPoint2 : distance->mPoint1) = point;
            replacement = distance->Create(*bodyB, *bodyA);
        } else if (record.constraint->GetSubType() == JPH::EConstraintSubType::SwingTwist) {
            auto *swing = static_cast<JPH::SwingTwistConstraintSettings *>(settings.GetPtr());
            (ownerChanged ? swing->mPosition2 : swing->mPosition1) = point;
            auto *next = static_cast<JPH::SwingTwistConstraint *>(swing->Create(*bodyB, *bodyA));
            const auto *previous = static_cast<const JPH::SwingTwistConstraint *>(record.constraint);
            CopySwingTwistMotorState(*previous, *next);
            replacement = next;
        } else if (record.constraint->GetSubType() == JPH::EConstraintSubType::SixDOF) {
            auto *six = static_cast<JPH::SixDOFConstraintSettings *>(settings.GetPtr());
            (ownerChanged ? six->mPosition2 : six->mPosition1) = point;
            auto *next = static_cast<JPH::SixDOFConstraint *>(six->Create(*bodyB, *bodyA));
            CopySixDOFMotorState(*static_cast<const JPH::SixDOFConstraint *>(record.constraint), *next);
            replacement = next;
        } else {
            throw std::logic_error("constraint subtype has no scale reconstruction");
        }
    }
    boundScale = scale;
    ReplaceConstraint(record, replacement);
}

void PhysicsWorld::ReplaceConstraint(ConstraintRecord &record, JPH::Constraint *replacement)
{
    InvalidateRagdollsForConstraint(record.constraint);
    m_physicsSystem->AddConstraint(replacement);
    m_physicsSystem->RemoveConstraint(record.constraint);
    record.constraint = replacement;
    ActivateConstraintBodies(record);
}

void PhysicsWorld::ActivateConstraintBodies(const ConstraintRecord &record)
{
    auto &bodies = m_physicsSystem->GetBodyInterface();
    for (const auto id : {record.bodyIdA, record.bodyIdB}) {
        if (id != 0xFFFFFFFF && bodies.IsAdded(JPH::BodyID(id)))
            bodies.ActivateBody(JPH::BodyID(id));
    }
}

void PhysicsWorld::UpdateConstraintCollision(ConstraintRecord &record, bool enableCollision)
{
    const bool ignores = record.bodyIdB != 0xFFFFFFFF && !enableCollision;
    if (ignores != record.ignoresCollision) {
        SetConstraintPairSuppressed(record.bodyIdA, record.bodyIdB, ignores);
        record.ignoresCollision = ignores;
        UpdateRagdollCollision(record.constraint, enableCollision);
    }
}

void PhysicsWorld::UpdateHingeConstraintSettings(uint64_t constraintId, bool useLimits, float minimum, float maximum,
                                                 bool enableCollision)
{
    auto &record = m_constraints.at(constraintId);
    if (record.constraint->GetSubType() != JPH::EConstraintSubType::Hinge)
        throw std::logic_error("constraint is not a hinge");
    auto *hinge = static_cast<JPH::HingeConstraint *>(record.constraint);
    const float lower = useLimits ? minimum : -glm::pi<float>();
    const float upper = useLimits ? maximum : glm::pi<float>();
    if (hinge->GetLimitsMin() != lower || hinge->GetLimitsMax() != upper) {
        hinge->SetLimits(lower, upper);
        hinge->ResetWarmStart();
        ActivateConstraintBodies(record);
    }
    UpdateConstraintCollision(record, enableCollision);
}

void PhysicsWorld::UpdateHingeConstraintMotor(uint64_t constraintId, bool enabled, float targetAngle, float stiffness,
                                              float damping, float maxTorque, float jointDamping, float frictionTorque,
                                              float armature)
{
    auto &record = m_constraints.at(constraintId);
    if (record.constraint->GetSubType() != JPH::EConstraintSubType::Hinge)
        throw std::logic_error("constraint is not a hinge");
    auto *hinge = static_cast<JPH::HingeConstraint *>(record.constraint);
    auto &motor = hinge->GetMotorSettings();
    const bool coefficientsChanged = motor.mSpringSettings.mMode != JPH::ESpringMode::StiffnessAndDamping ||
                                     motor.mSpringSettings.mStiffness != stiffness ||
                                     motor.mSpringSettings.mDamping != damping || motor.mMinTorqueLimit != -maxTorque ||
                                     motor.mMaxTorqueLimit != maxTorque || hinge->GetViscousDamping() != jointDamping ||
                                     hinge->GetMaxFrictionTorque() != frictionTorque ||
                                     (hinge->GetMotorState() != JPH::EMotorState::Off) != enabled;
    const bool armatureChanged = hinge->GetArmature() != armature;
    const float previousTarget = hinge->GetTargetAngle();
    hinge->SetTargetAngle(targetAngle); // Applies the native limit convention.
    const bool targetChanged = hinge->GetTargetAngle() != previousTarget;
    if (!coefficientsChanged && !armatureChanged && !(enabled && targetChanged))
        return;
    if (coefficientsChanged) {
        motor.mSpringSettings = JPH::SpringSettings(JPH::ESpringMode::StiffnessAndDamping, stiffness, damping);
        motor.SetTorqueLimit(maxTorque);
        hinge->SetViscousDamping(jointDamping);
        hinge->SetMaxFrictionTorque(frictionTorque);
        hinge->ResetWarmStart();
    }
    if (armatureChanged)
        hinge->SetArmature(armature);
    hinge->SetMotorState(enabled ? JPH::EMotorState::Position : JPH::EMotorState::Off);
    ActivateConstraintBodies(record);
}

void PhysicsWorld::UpdateHingeConstraintTarget(uint64_t constraintId, float targetAngle)
{
    const auto &record = m_constraints.at(constraintId);
    if (record.constraint->GetSubType() != JPH::EConstraintSubType::Hinge)
        throw std::logic_error("constraint is not a hinge");
    auto &hinge = *static_cast<JPH::HingeConstraint *>(record.constraint);
    const float previous = hinge.GetTargetAngle();
    hinge.SetTargetAngle(targetAngle);
    if (hinge.GetMotorState() != JPH::EMotorState::Off && hinge.GetTargetAngle() != previous)
        ActivateConstraintBodies(record);
}

void PhysicsWorld::SetHingeMotorTargets(const std::vector<HingeJoint *> &joints, const float *targets)
{
    std::vector<JPH::BodyID> bodies;
    bodies.reserve(2 * joints.size());
    for (size_t row = 0; row < joints.size(); ++row) {
        auto &joint = *joints[row];
        joint.m_motorTargetAngle = glm::degrees(targets[row]);
        const auto &record = m_constraints.at(joint.GetPhysicsConstraintId());
        auto &hinge = *static_cast<JPH::HingeConstraint *>(record.constraint);
        const float previous = hinge.GetTargetAngle();
        hinge.SetTargetAngle(joint.GetClampedMotorTargetRadians());
        if (hinge.GetMotorState() == JPH::EMotorState::Off || hinge.GetTargetAngle() == previous)
            continue;
        bodies.emplace_back(record.bodyIdA);
        if (record.bodyIdB != 0xFFFFFFFF)
            bodies.emplace_back(record.bodyIdB);
    }
    if (bodies.empty())
        return;
    std::sort(bodies.begin(), bodies.end());
    bodies.erase(std::unique(bodies.begin(), bodies.end()), bodies.end());
    JPH::BodyLockMultiWrite lock(m_physicsSystem->GetBodyLockInterface(), bodies.data(), int(bodies.size()));
    std::vector<JPH::BodyID> sleeping;
    for (size_t i = 0; i < bodies.size(); ++i) {
        auto *body = lock.GetBody(int(i));
        if (!body || !body->IsInBroadPhase() || body->IsStatic())
            continue;
        if (body->IsActive())
            body->ResetSleepTimer();
        else
            sleeping.push_back(body->GetID());
    }
    // Active bodies need no active-list lock; all body locks are already held.
    if (!sleeping.empty())
        m_physicsSystem->GetBodyInterfaceNoLock().ActivateBodies(sleeping.data(), int(sleeping.size()));
}

void PhysicsWorld::ReadHingeMotion(const std::vector<HingeJoint *> &joints, float *output) const
{
    std::vector<JPH::BodyID> bodies;
    std::vector<const JPH::HingeConstraint *> hinges;
    bodies.reserve(2 * joints.size());
    hinges.reserve(joints.size());
    for (auto *joint : joints) {
        const auto &record = m_constraints.at(joint->GetPhysicsConstraintId());
        bodies.emplace_back(record.bodyIdA);
        bodies.emplace_back(record.bodyIdB);
        hinges.push_back(static_cast<const JPH::HingeConstraint *>(record.constraint));
    }
    // MultiRead merges mutex masks, including shared endpoints between edges.
    JPH::BodyLockMultiRead lock(m_physicsSystem->GetBodyLockInterface(), bodies.data(), int(bodies.size()));
    for (size_t row = 0; row < joints.size(); ++row) {
        const auto &hinge = *hinges[row];
        const auto &owner = *lock.GetBody(int(2 * row));
        const auto *connected =
            bodies[2 * row + 1].IsInvalid() ? &JPH::Body::sFixedToWorld : lock.GetBody(int(2 * row + 1));
        const auto axis = owner.GetRotation() * hinge.GetLocalSpaceHingeAxis2();
        output[2 * row] = hinge.GetCurrentAngle();
        output[2 * row + 1] = (owner.GetAngularVelocity() - connected->GetAngularVelocity()).Dot(axis);
    }
}

void PhysicsWorld::UpdateSliderConstraintSettings(uint64_t constraintId, bool useLimits, float minimum, float maximum,
                                                  bool enableCollision)
{
    auto &record = m_constraints.at(constraintId);
    if (record.constraint->GetSubType() != JPH::EConstraintSubType::Slider)
        throw std::logic_error("constraint is not a slider");
    auto *slider = static_cast<JPH::SliderConstraint *>(record.constraint);
    const float lower = useLimits ? minimum : -FLT_MAX;
    const float upper = useLimits ? maximum : FLT_MAX;
    if (slider->GetLimitsMin() != lower || slider->GetLimitsMax() != upper) {
        slider->SetLimits(lower, upper);
        slider->ResetWarmStart();
        ActivateConstraintBodies(record);
    }
    UpdateConstraintCollision(record, enableCollision);
}

void PhysicsWorld::DestroyConstraintsForBody(uint32_t bodyId)
{
    for (auto it = m_constraints.begin(); it != m_constraints.end();) {
        const auto current = it++;
        if (current->second.bodyIdA == bodyId || current->second.bodyIdB == bodyId)
            DestroyConstraint(current->first);
    }
}

void PhysicsWorld::DestroyConstraint(uint64_t constraintId)
{
    auto it = m_constraints.find(constraintId);
    if (it == m_constraints.end())
        return;
    InvalidateRagdollsForConstraint(it->second.constraint);
    ConstraintRecord record = it->second;
    m_constraints.erase(it);
    if (m_physicsSystem && record.constraint)
        m_physicsSystem->RemoveConstraint(record.constraint);
    if (record.ignoresCollision)
        SetConstraintPairSuppressed(record.bodyIdA, record.bodyIdB, false);
    record.owner.OnPhysicsConstraintDestroyed();
}

float PhysicsWorld::GetHingeConstraintAngle(uint64_t constraintId) const
{
    auto it = m_constraints.find(constraintId);
    if (it == m_constraints.end())
        throw std::logic_error("hinge constraint is not active");
    if (it->second.constraint->GetSubType() != JPH::EConstraintSubType::Hinge)
        throw std::logic_error("constraint is not a hinge");
    // Avoid cross-DLL RTTI: the record already carries the authoritative kind.
    return static_cast<JPH::HingeConstraint *>(it->second.constraint)->GetCurrentAngle();
}

float PhysicsWorld::GetHingeConstraintAngularVelocity(uint64_t constraintId) const
{
    return GetHingeConstraintMotionState(constraintId).y;
}

glm::vec2 PhysicsWorld::GetHingeConstraintMotionState(uint64_t constraintId) const
{
    const auto &record = m_constraints.at(constraintId);
    if (record.constraint->GetSubType() != JPH::EConstraintSubType::Hinge)
        throw std::logic_error("constraint is not a hinge");
    const auto *hinge = static_cast<JPH::HingeConstraint *>(record.constraint);
    JPH::BodyID ids[2] = {JPH::BodyID(record.bodyIdA), JPH::BodyID(record.bodyIdB)};
    JPH::BodyLockMultiRead lock(m_physicsSystem->GetBodyLockInterface(), ids, 2);
    const auto *owner = lock.GetBody(0);
    const auto *connected = record.bodyIdB == 0xFFFFFFFF ? &JPH::Body::sFixedToWorld : lock.GetBody(1);
    const auto axis = owner->GetRotation() * hinge->GetLocalSpaceHingeAxis2();
    return {hinge->GetCurrentAngle(), (owner->GetAngularVelocity() - connected->GetAngularVelocity()).Dot(axis)};
}

float PhysicsWorld::GetSliderConstraintPosition(uint64_t constraintId) const
{
    auto it = m_constraints.find(constraintId);
    if (it == m_constraints.end())
        throw std::logic_error("slider constraint is not active");
    if (it->second.constraint->GetSubType() != JPH::EConstraintSubType::Slider)
        throw std::logic_error("constraint is not a slider");
    return static_cast<JPH::SliderConstraint *>(it->second.constraint)->GetCurrentPosition();
}

void PhysicsWorld::SetConstraintPairSuppressed(uint32_t bodyIdA, uint32_t bodyIdB, bool suppressed)
{
    if (!m_contactListener || !m_physicsSystem || bodyIdA == 0xFFFFFFFF || bodyIdB == 0xFFFFFFFF)
        return;
    m_contactListener->SetBodyPairIgnored(bodyIdA, bodyIdB, suppressed);
    m_contactListener->InvalidatePairsForBody(bodyIdA);
    m_contactListener->InvalidatePairsForBody(bodyIdB);
    JPH::BodyInterface &bodyInterface = m_physicsSystem->GetBodyInterface();
    bodyInterface.InvalidateContactCache(JPH::BodyID(bodyIdA));
    bodyInterface.InvalidateContactCache(JPH::BodyID(bodyIdB));
    // Constraint retirement also runs after broadphase removal, immediately
    // before body destruction. Jolt activation requires a resident body;
    // reactivating a removed body would leave its freed ID in the active set.
    if (bodyInterface.IsAdded(JPH::BodyID(bodyIdA)))
        bodyInterface.ActivateBody(JPH::BodyID(bodyIdA));
    if (bodyInterface.IsAdded(JPH::BodyID(bodyIdB)))
        bodyInterface.ActivateBody(JPH::BodyID(bodyIdB));
}

uint64_t PhysicsWorld::CreateSwingTwistConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                                  const glm::vec3 &localAnchor, const SwingTwistJointSettings &s,
                                                  bool enableCollision)
{
    const auto rotation = GetBodyRotation(bodyIdA);
    const auto scale = m_bodies.at(bodyIdA).gameObject->GetTransform()->GetWorldScale();
    const auto point = GetBodyPosition(bodyIdA) + rotation * (localAnchor * scale);
    JPH::SwingTwistConstraintSettings settings;
    settings.mSpace = JPH::EConstraintSpace::WorldSpace;
    settings.mPosition1 = settings.mPosition2 = JPH::RVec3(point.x, point.y, point.z);
    settings.mTwistAxis1 = settings.mTwistAxis2 = ToJolt(rotation * s.twistAxis);
    settings.mPlaneAxis1 = settings.mPlaneAxis2 = ToJolt(rotation * s.planeAxis);
    settings.mSwingType = static_cast<JPH::ESwingType>(s.swingType);
    settings.mNormalHalfConeAngle = glm::radians(s.normalHalfConeAngle);
    settings.mPlaneHalfConeAngle = glm::radians(s.planeHalfConeAngle);
    settings.mTwistMinAngle = glm::radians(s.minimumTwistAngle);
    settings.mTwistMaxAngle = glm::radians(s.maximumTwistAngle);
    return CreateTwoBodyConstraint(owner, bodyIdA, bodyIdB, settings, localAnchor, enableCollision);
}

void PhysicsWorld::SetSwingTwistConstraintType(uint64_t id, JointSwingType type)
{
    auto &record = m_constraints.at(id);
    auto settings = record.constraint->GetConstraintSettings();
    auto *swing = static_cast<JPH::SwingTwistConstraintSettings *>(settings.GetPtr());
    swing->mSwingType = static_cast<JPH::ESwingType>(type);
    JPH::SwingTwistConstraint *replacement;
    {
        const JPH::BodyID ids[] = {JPH::BodyID(record.bodyIdA), JPH::BodyID(record.bodyIdB)};
        JPH::BodyLockMultiWrite lock(m_physicsSystem->GetBodyLockInterface(), ids, 2);
        auto *connected = record.bodyIdB == 0xFFFFFFFF ? &JPH::Body::sFixedToWorld : lock.GetBody(1);
        replacement = static_cast<JPH::SwingTwistConstraint *>(swing->Create(*connected, *lock.GetBody(0)));
        CopySwingTwistMotorState(*static_cast<JPH::SwingTwistConstraint *>(record.constraint), *replacement);
    }
    ReplaceConstraint(record, replacement);
}

void PhysicsWorld::UpdateSwingTwistConstraint(uint64_t id, const SwingTwistJointSettings &s, bool enableCollision)
{
    auto &record = m_constraints.at(id);
    auto *joint = static_cast<JPH::SwingTwistConstraint *>(record.constraint);
    bool changed = false;
    const auto setAngle = [&](float value, float current, auto setter) {
        const float radians = glm::radians(value);
        if (radians != current) {
            (joint->*setter)(radians);
            changed = true;
        }
    };
    setAngle(s.normalHalfConeAngle, joint->GetNormalHalfConeAngle(),
             &JPH::SwingTwistConstraint::SetNormalHalfConeAngle);
    setAngle(s.planeHalfConeAngle, joint->GetPlaneHalfConeAngle(), &JPH::SwingTwistConstraint::SetPlaneHalfConeAngle);
    // Keep every intermediate native interval ordered while changing both bounds.
    if (glm::radians(s.minimumTwistAngle) > joint->GetTwistMaxAngle()) {
        setAngle(s.maximumTwistAngle, joint->GetTwistMaxAngle(), &JPH::SwingTwistConstraint::SetTwistMaxAngle);
        setAngle(s.minimumTwistAngle, joint->GetTwistMinAngle(), &JPH::SwingTwistConstraint::SetTwistMinAngle);
    } else {
        setAngle(s.minimumTwistAngle, joint->GetTwistMinAngle(), &JPH::SwingTwistConstraint::SetTwistMinAngle);
        setAngle(s.maximumTwistAngle, joint->GetTwistMaxAngle(), &JPH::SwingTwistConstraint::SetTwistMaxAngle);
    }
    if (joint->GetMaxFrictionTorque() != s.frictionTorque) {
        joint->SetMaxFrictionTorque(s.frictionTorque);
        changed = true;
    }
    changed |=
        UpdateDrive(joint->GetSwingMotorSettings(), s.swingMotorStiffness, s.swingMotorDamping, s.swingMotorMaxTorque);
    changed |=
        UpdateDrive(joint->GetTwistMotorSettings(), s.twistMotorStiffness, s.twistMotorDamping, s.twistMotorMaxTorque);
    const auto swingMode = static_cast<JPH::EMotorState>(s.swingMotorMode);
    const auto twistMode = static_cast<JPH::EMotorState>(s.twistMotorMode);
    if (joint->GetSwingMotorState() != swingMode) {
        joint->SetSwingMotorState(swingMode);
        changed = true;
    }
    if (joint->GetTwistMotorState() != twistMode) {
        joint->SetTwistMotorState(twistMode);
        changed = true;
    }
    if (changed)
        joint->ResetWarmStart();
    bool wake = changed;
    const auto before = joint->GetTargetOrientationCS();
    joint->SetTargetOrientationCS(ToJolt(s.targetOrientation));
    // Jolt clamps to the current limits; repeating an out-of-range request must not wake it every tick.
    const auto after = joint->GetTargetOrientationCS();
    wake |= before != after && before != -after;
    const auto velocity = ToJolt(glm::radians(s.targetAngularVelocity));
    if (joint->GetTargetAngularVelocityCS() != velocity) {
        joint->SetTargetAngularVelocityCS(velocity);
        wake = true;
    }
    if (wake) {
        ActivateConstraintBodies(record);
    }
    UpdateConstraintCollision(record, enableCollision);
}

glm::quat PhysicsWorld::GetAngularConstraintOrientation(uint64_t id) const
{
    const auto &record = m_constraints.at(id);
    JPH::Quat value;
    if (record.constraint->GetSubType() == JPH::EConstraintSubType::SwingTwist)
        value = static_cast<const JPH::SwingTwistConstraint *>(record.constraint)->GetRotationInConstraintSpace();
    else if (record.constraint->GetSubType() == JPH::EConstraintSubType::SixDOF)
        value = static_cast<const JPH::SixDOFConstraint *>(record.constraint)->GetRotationInConstraintSpace();
    else
        throw std::logic_error("constraint is not an angular joint");
    return {value.GetW(), value.GetX(), value.GetY(), value.GetZ()};
}

uint64_t PhysicsWorld::CreateSixDOFConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                              const glm::vec3 &localAnchor, const SixDOFJointSettings &s,
                                              bool enableCollision)
{
    const auto rotation = GetBodyRotation(bodyIdA);
    const auto scale = m_bodies.at(bodyIdA).gameObject->GetTransform()->GetWorldScale();
    const auto point = GetBodyPosition(bodyIdA) + rotation * (localAnchor * scale);
    JPH::SixDOFConstraintSettings settings;
    settings.mSpace = JPH::EConstraintSpace::WorldSpace;
    settings.mPosition1 = settings.mPosition2 = JPH::RVec3(point.x, point.y, point.z);
    settings.mAxisX1 = settings.mAxisX2 = ToJolt(rotation * s.axis);
    settings.mAxisY1 = settings.mAxisY2 = ToJolt(rotation * s.secondaryAxis);
    // Pyramid supports independent asymmetric rotation limits on all axes.
    settings.mSwingType = JPH::ESwingType::Pyramid;
    std::array<float, 6> minimum, maximum;
    SixDOFLimits(s, minimum, maximum);
    std::copy(minimum.begin(), minimum.end(), settings.mLimitMin);
    std::copy(maximum.begin(), maximum.end(), settings.mLimitMax);
    for (int i = 0; i < 3; ++i)
        settings.mLimitsSpringSettings[i] = {JPH::ESpringMode::StiffnessAndDamping, s.linearLimitStiffness[i],
                                             s.linearLimitDamping[i]};
    return CreateTwoBodyConstraint(owner, bodyIdA, bodyIdB, settings, localAnchor, enableCollision);
}

void PhysicsWorld::UpdateSixDOFConstraint(uint64_t id, const SixDOFJointSettings &s, bool enableCollision)
{
    auto &record = m_constraints.at(id);
    auto *joint = static_cast<JPH::SixDOFConstraint *>(record.constraint);
    std::array<float, 6> minimum, maximum;
    SixDOFLimits(s, minimum, maximum);
    bool linearChanged = false, angularChanged = false;
    for (int i = 0; i < 6; ++i) {
        const auto axis = static_cast<JPH::SixDOFConstraintSettings::EAxis>(i);
        const bool changed = joint->GetLimitsMin(axis) != minimum[i] || joint->GetLimitsMax(axis) != maximum[i];
        (i < 3 ? linearChanged : angularChanged) |= changed;
    }
    if (linearChanged)
        joint->SetTranslationLimits({minimum[0], minimum[1], minimum[2]}, {maximum[0], maximum[1], maximum[2]});
    if (angularChanged)
        joint->SetRotationLimits({minimum[3], minimum[4], minimum[5]}, {maximum[3], maximum[4], maximum[5]});
    bool changed = linearChanged || angularChanged;
    const JointMotorMode modes[] = {s.xMotorMode,        s.yMotorMode,        s.zMotorMode,
                                    s.angularXMotorMode, s.angularYMotorMode, s.angularZMotorMode};
    for (int i = 0; i < 6; ++i) {
        const auto axis = static_cast<JPH::SixDOFConstraintSettings::EAxis>(i);
        const int component = i % 3;
        const float friction = i < 3 ? s.linearFrictionForce[i] : s.angularFrictionTorque[component];
        // Jolt's friction activation cache also depends on whether this axis
        // is fixed. Refresh it after a motion-mode change, even at the same force.
        if (joint->GetMaxFriction(axis) != friction || (i < 3 ? linearChanged : angularChanged)) {
            joint->SetMaxFriction(axis, friction);
            changed = true;
        }
        changed |= UpdateDrive(joint->GetMotorSettings(axis),
                               (i < 3 ? s.linearMotorStiffness : s.angularMotorStiffness)[component],
                               (i < 3 ? s.linearMotorDamping : s.angularMotorDamping)[component],
                               (i < 3 ? s.linearMotorMaxForce : s.angularMotorMaxTorque)[component], i < 3);
        const auto mode = static_cast<JPH::EMotorState>(modes[i]);
        if (joint->GetMotorState(axis) != mode) {
            joint->SetMotorState(axis, mode);
            changed = true;
        }
        if (i < 3) {
            const auto &spring = joint->GetLimitsSpringSettings(axis);
            if (spring.mMode != JPH::ESpringMode::StiffnessAndDamping ||
                spring.mStiffness != s.linearLimitStiffness[i] || spring.mDamping != s.linearLimitDamping[i]) {
                joint->SetLimitsSpringSettings(
                    axis, {JPH::ESpringMode::StiffnessAndDamping, s.linearLimitStiffness[i], s.linearLimitDamping[i]});
                changed = true;
            }
        }
    }
    if (changed)
        joint->ResetWarmStart();
    bool wake = changed;
    const auto position = ToJolt(s.targetPosition);
    if (joint->GetTargetPositionCS() != position) {
        joint->SetTargetPositionCS(position);
        wake = true;
    }
    const auto velocity = ToJolt(s.targetVelocity);
    if (joint->GetTargetVelocityCS() != velocity) {
        joint->SetTargetVelocityCS(velocity);
        wake = true;
    }
    const auto angularVelocity = ToJolt(glm::radians(s.targetAngularVelocity));
    if (joint->GetTargetAngularVelocityCS() != angularVelocity) {
        joint->SetTargetAngularVelocityCS(angularVelocity);
        wake = true;
    }
    const auto before = joint->GetTargetOrientationCS();
    joint->SetTargetOrientationCS(ToJolt(s.targetOrientation));
    const auto after = joint->GetTargetOrientationCS();
    wake |= before != after && before != -after;
    if (wake)
        ActivateConstraintBodies(record);
    UpdateConstraintCollision(record, enableCollision);
}

glm::vec3 PhysicsWorld::GetSixDOFConstraintPosition(uint64_t id) const
{
    const auto *joint = static_cast<const JPH::SixDOFConstraint *>(m_constraints.at(id).constraint);
    const auto frame = joint->GetBody1()->GetCenterOfMassTransform() * joint->GetConstraintToBody1Matrix();
    const auto point =
        joint->GetBody2()->GetCenterOfMassTransform() * joint->GetConstraintToBody2Matrix().GetTranslation();
    const auto local = frame.InversedRotationTranslation() * point;
    return {local.GetX(), local.GetY(), local.GetZ()};
}
} // namespace infernux
