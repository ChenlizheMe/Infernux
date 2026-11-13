#include "../HingeJoint.h"
#include "JoltGpuJoints.h"
#include "PhysicsWorld.h"

#include <algorithm>
#include <cmath>
#include <glm/gtc/constants.hpp>
#include <stdexcept>
#include <unordered_map>

namespace infernux
{
namespace
{
using JPH::GPU::JointAuthor;

JPH::GPU::Float4 JointVector(const glm::vec3 &value)
{
    if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z))
        throw std::invalid_argument("Joint anchor and axis must be finite");
    return {value.x, value.y, value.z, 0.0f};
}

void ValidateHingeLimits(float minimum, float maximum)
{
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum > 0.0f || maximum < 0.0f || minimum > maximum ||
        minimum < -glm::pi<float>() || maximum > glm::pi<float>())
        throw std::invalid_argument("Hinge limits must be ordered radians within [-pi, pi] and span zero");
}

void RequireHinge(const JoltGpuScene::PublishedJoint &joint)
{
    if (joint.kind != JPH::GPU::JointHinge)
        throw std::logic_error("Constraint is not a hinge");
}

void SetJointFlag(JointAuthor &author, uint32_t flag, bool enabled)
{
    author.bodies.w = enabled ? author.bodies.w | flag : author.bodies.w & ~flag;
}

float ClampHingeTarget(const JointAuthor &author, float target)
{
    return author.bodies.w & JPH::GPU::JointLimitsEnabled ? std::clamp(target, author.limits.x, author.limits.y)
                                                          : target;
}

JoltGpuControlWork PrepareJointReadWork(std::function<JoltGpuControlWork()> prepare, std::exception_ptr &error)
{
    try {
        auto work = prepare();
        if (auto continuation = std::move(work.continueAfterCompletion))
            work.continueAfterCompletion = [continuation = std::move(continuation), &error]() mutable {
                return PrepareJointReadWork(std::move(continuation), error);
            };
        return work;
    } catch (const JoltGpuQueryRequestError &) {
        error = std::current_exception();
        return {};
    }
}
} // namespace

uint64_t PhysicsWorld::CreateHingeConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                             const glm::vec3 &localBakedAnchor, const glm::vec3 &ownerAxis,
                                             const glm::vec3 &localAnchor, bool useLimits, float minimumAngle,
                                             float maximumAngle, bool enableCollision)
{
    RequireOwnerThread();
    ValidateHingeLimits(minimumAngle, maximumAngle);
    (void)JointVector(ownerAxis);
    if (glm::dot(ownerAxis, ownerAxis) <= 1e-12f)
        throw std::invalid_argument("Hinge axis must be non-zero");
    JointAuthor author{};
    author.bodies.z = JPH::GPU::JointHinge;
    author.bodies.w = JPH::GPU::JointEnabled | (enableCollision ? JPH::GPU::JointCollisionEnabled : 0) |
                      (useLimits ? JPH::GPU::JointLimitsEnabled : 0);
    author.ownerAnchor = JointVector(localBakedAnchor);
    author.ownerAxis = JointVector(glm::normalize(ownerAxis));
    author.ownerLocalAnchor = JointVector(localAnchor);
    author.limits = {minimumAngle, maximumAngle, 0.0f, 0.0f};
    RequireGpu();
    return m_gpuScene->CreateJoint(owner, bodyIdA, bodyIdB, author);
}

uint64_t PhysicsWorld::CreateSliderConstraint(PhysicsConstraintOwner &, uint32_t, uint32_t, const glm::vec3 &,
                                              const glm::vec3 &, const glm::vec3 &, bool, float, float, bool)
{
    throw std::logic_error("GPU CreateSliderConstraint has not been implemented");
}

uint64_t PhysicsWorld::CreateFixedConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                             const glm::vec3 &localBakedAnchor, const glm::vec3 &localAnchor,
                                             bool enableCollision)
{
    RequireOwnerThread();
    JointAuthor author{};
    author.bodies.z = JPH::GPU::JointFixed;
    author.bodies.w = JPH::GPU::JointEnabled | (enableCollision ? JPH::GPU::JointCollisionEnabled : 0);
    author.ownerAnchor = JointVector(localBakedAnchor);
    author.ownerLocalAnchor = JointVector(localAnchor);
    RequireGpu();
    return m_gpuScene->CreateJoint(owner, bodyIdA, bodyIdB, author);
}

void PhysicsWorld::SetConstraintCollisionEnabled(uint64_t constraintId, bool enabled)
{
    RequireOwnerThread();
    RequireGpu();
    m_gpuScene->UpdateJoint(
        constraintId, [=](JointAuthor &author) { SetJointFlag(author, JPH::GPU::JointCollisionEnabled, enabled); });
}

uint64_t PhysicsWorld::CreateDistanceConstraint(PhysicsConstraintOwner &, uint32_t, uint32_t, const glm::vec3 &,
                                                const glm::vec3 &, bool, float, float, float, float, bool)
{
    throw std::logic_error("GPU CreateDistanceConstraint has not been implemented");
}

void PhysicsWorld::UpdateDistanceConstraintSettings(uint64_t, bool, float, float, float, float, bool)
{
    throw std::logic_error("GPU UpdateDistanceConstraintSettings has not been implemented");
}

float PhysicsWorld::GetDistanceConstraintLength(uint64_t) const
{
    throw std::logic_error("GPU GetDistanceConstraintLength has not been implemented");
}

void PhysicsWorld::UpdateHingeConstraintSettings(uint64_t constraintId, bool useLimits, float minimum, float maximum,
                                                 bool enableCollision)
{
    RequireOwnerThread();
    ValidateHingeLimits(minimum, maximum);
    RequireGpu();
    RequireHinge(m_gpuScene->GetAuthoredJoint(constraintId));
    m_gpuScene->UpdateJoint(constraintId, [=](JointAuthor &author) {
        SetJointFlag(author, JPH::GPU::JointLimitsEnabled, useLimits);
        SetJointFlag(author, JPH::GPU::JointCollisionEnabled, enableCollision);
        author.limits = {minimum, maximum, 0.0f, 0.0f};
    });
}

void PhysicsWorld::UpdateHingeConstraintMotor(uint64_t constraintId, bool enabled, float targetAngle, float stiffness,
                                              float damping, float maxTorque, float jointDamping, float frictionTorque,
                                              float armature)
{
    RequireOwnerThread();
    if (!std::isfinite(targetAngle))
        throw std::invalid_argument("Hinge motor target must be finite radians");
    for (const float value : {stiffness, damping, maxTorque, jointDamping, frictionTorque, armature})
        if (!std::isfinite(value) || value < 0.0f)
            throw std::invalid_argument("Hinge motor and passive coefficients must be finite and non-negative");
    RequireGpu();
    RequireHinge(m_gpuScene->GetAuthoredJoint(constraintId));
    m_gpuScene->UpdateJoint(constraintId, [=](JointAuthor &author) {
        SetJointFlag(author, JPH::GPU::JointMotorEnabled, enabled);
        author.motor = {ClampHingeTarget(author, targetAngle), stiffness, damping, maxTorque};
        author.passive = {jointDamping, frictionTorque, armature, 0.0f};
    });
}

void PhysicsWorld::UpdateHingeConstraintTarget(uint64_t constraintId, float targetAngle)
{
    RequireOwnerThread();
    if (!std::isfinite(targetAngle))
        throw std::invalid_argument("Hinge motor target must be finite radians");
    RequireGpu();
    RequireHinge(m_gpuScene->GetAuthoredJoint(constraintId));
    m_gpuScene->UpdateJoint(constraintId,
                            [=](JointAuthor &author) { author.motor.x = ClampHingeTarget(author, targetAngle); });
}

void PhysicsWorld::SetHingeMotorTargets(const std::vector<HingeJoint *> &joints, const float *targets)
{
    RequireOwnerThread();
    if (!targets && !joints.empty())
        throw std::invalid_argument("Hinge targets require input storage");
    RequireGpu();
    struct Target
    {
        HingeJoint *joint;
        float degrees;
    };
    std::vector<Target> prepared;
    prepared.reserve(joints.size());
    std::vector<JoltGpuScene::JointUpdate> updates;
    updates.reserve(joints.size());
    // Validate every identity and coefficient before queuing any author update.
    for (size_t row = 0; row < joints.size(); ++row) {
        if (!joints[row] || !joints[row]->GetPhysicsConstraintId())
            throw std::logic_error("Hinge targets require bound joints");
        const auto id = joints[row]->GetPhysicsConstraintId();
        const auto &identity = m_gpuScene->GetAuthoredJoint(id);
        RequireHinge(identity);
        if (!std::isfinite(targets[row]) || !std::isfinite(glm::degrees(targets[row])))
            throw std::invalid_argument("Hinge targets must be finite representable radians");
        auto &joint = *joints[row];
        const float degrees = glm::degrees(targets[row]);
        auto author = identity.author;
        author.motor.x =
            glm::radians(joint.m_useLimits ? std::clamp(degrees, joint.m_minimumAngle, joint.m_maximumAngle) : degrees);
        prepared.push_back({&joint, degrees});
        updates.push_back({id, author});
    }
    m_gpuScene->UpdateJoints(std::move(updates));
    for (const auto &target : prepared)
        target.joint->m_motorTargetAngle = target.degrees;
}

void PhysicsWorld::ReadHingeMotion(const std::vector<HingeJoint *> &joints, float *output) const
{
    RequireOwnerThread();
    std::vector<uint64_t> ids;
    ids.reserve(joints.size());
    for (const auto *joint : joints) {
        if (!joint || !joint->GetPhysicsConstraintId())
            throw std::logic_error("Hinge motion requires bound joints");
        ids.push_back(joint->GetPhysicsConstraintId());
    }
    ReadJointMotion(ids, output);
}

void PhysicsWorld::ReadJointMotion(const std::vector<uint64_t> &ids, float *output) const
{
    RequireOwnerThread();
    if (!output && !ids.empty())
        throw std::invalid_argument("Hinge motion requires output storage");
    if (ids.empty())
        return;
    JoltGpuPreparedJointMotion prepared;
    std::exception_ptr requestError;
    RunControl([&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        return PrepareJointReadWork(
            [&]() -> JoltGpuControlWork {
                std::unordered_map<uint64_t, const JoltGpuScene::PublishedJoint *> identities;
                identities.reserve(publication->joints.size());
                for (const auto &joint : publication->joints)
                    identities.emplace(joint.id, &joint);
                std::vector<uint32_t> rows;
                rows.reserve(ids.size());
                for (const auto id : ids) {
                    const auto found = identities.find(id);
                    if (found == identities.end() || found->second->kind != JPH::GPU::JointHinge)
                        throw JoltGpuQueryRequestError("Hinge motion requires an accepted hinge identity");
                    rows.push_back(found->second->row);
                }
                prepared = m_gpuJoints->PrepareReadMotion(publication, rows);
                return std::move(prepared.work);
            },
            requestError);
    });
    if (requestError)
        std::rethrow_exception(requestError);
    if (!prepared.result)
        throw std::runtime_error("GPU hinge motion did not produce a result");
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    prepared.result->CopyMotion(output, ids.size() * 2);
}

void PhysicsWorld::UpdateSliderConstraintSettings(uint64_t, bool, float, float, bool)
{
    throw std::logic_error("GPU UpdateSliderConstraintSettings has not been implemented");
}

void PhysicsWorld::DestroyConstraintsForBody(uint32_t bodyId)
{
    RequireOwnerThread();
    if (m_gpuScene && bodyId != UINT32_MAX)
        m_gpuScene->DestroyJointsForBody(bodyId);
}

void PhysicsWorld::DestroyConstraint(uint64_t constraintId)
{
    if (constraintId == 0)
        return;
    RequireOwnerThread();
    if (m_gpuScene)
        m_gpuScene->DestroyJoint(constraintId);
}

float PhysicsWorld::GetHingeConstraintAngle(uint64_t constraintId) const
{
    return GetHingeConstraintMotionState(constraintId).x;
}

float PhysicsWorld::GetHingeConstraintAngularVelocity(uint64_t constraintId) const
{
    return GetHingeConstraintMotionState(constraintId).y;
}

glm::vec2 PhysicsWorld::GetHingeConstraintMotionState(uint64_t constraintId) const
{
    float motion[2];
    ReadJointMotion({constraintId}, motion);
    return {motion[0], motion[1]};
}

float PhysicsWorld::GetSliderConstraintPosition(uint64_t) const
{
    throw std::logic_error("GPU GetSliderConstraintPosition has not been implemented");
}

uint64_t PhysicsWorld::CreateSwingTwistConstraint(PhysicsConstraintOwner &, uint32_t, uint32_t, const glm::vec3 &,
                                                  const SwingTwistJointSettings &, bool)
{
    throw std::logic_error("GPU CreateSwingTwistConstraint has not been implemented");
}

void PhysicsWorld::SetSwingTwistConstraintType(uint64_t, JointSwingType)
{
    throw std::logic_error("GPU SetSwingTwistConstraintType has not been implemented");
}

void PhysicsWorld::UpdateSwingTwistConstraint(uint64_t, const SwingTwistJointSettings &, bool)
{
    throw std::logic_error("GPU UpdateSwingTwistConstraint has not been implemented");
}

glm::quat PhysicsWorld::GetAngularConstraintOrientation(uint64_t) const
{
    throw std::logic_error("GPU GetAngularConstraintOrientation has not been implemented");
}

uint64_t PhysicsWorld::CreateSixDOFConstraint(PhysicsConstraintOwner &, uint32_t, uint32_t, const glm::vec3 &,
                                              const SixDOFJointSettings &, bool)
{
    throw std::logic_error("GPU CreateSixDOFConstraint has not been implemented");
}

void PhysicsWorld::UpdateSixDOFConstraint(uint64_t, const SixDOFJointSettings &, bool)
{
    throw std::logic_error("GPU UpdateSixDOFConstraint has not been implemented");
}

glm::vec3 PhysicsWorld::GetSixDOFConstraintPosition(uint64_t) const
{
    throw std::logic_error("GPU GetSixDOFConstraintPosition has not been implemented");
}

} // namespace infernux
