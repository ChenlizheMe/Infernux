#include "../HingeJoint.h"
#include "JointSettings.h"
#include "JoltCpuControl.h"
#include "JoltGpuJoints.h"
#include "PhysicsDomain.h"
#include "PhysicsWorld.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <glm/gtc/constants.hpp>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace infernux
{
namespace
{
using JPH::GPU::JointAuthor;

using Cpu = JoltCpuConstraints;
using CpuControl = JoltCpuControl;

const JoltGpuScene::PublishedJoint &RequireGpuJoint(const JoltGpuScene *scene, uint64_t id)
{
    if (!scene)
        throw std::invalid_argument("Physics joint identity is invalid or retired");
    return scene->GetAuthoredJoint(id);
}

uint64_t StageCpuJoint(PhysicsDomain &domain, std::unordered_map<uint64_t, PhysicsDomain *> &routes,
                       PhysicsConstraintOwner &owner, Cpu::Binding binding, Cpu::Settings settings, bool collision)
{
    const auto id = PhysicsECSStore::Instance().AllocateJointIdentity();
    routes.emplace(id, &domain);
    try {
        domain.cpuControl->CreateJoint(id, owner, {std::move(binding), std::move(settings), collision});
    } catch (...) {
        routes.erase(id);
        throw;
    }
    return id;
}

template <class Settings> const Settings &RequireCpuSettings(const CpuControl::AuthoredJoint &joint)
{
    const auto *settings = std::get_if<Settings>(&joint.settings);
    if (!settings)
        throw std::logic_error("CPU joint does not have the requested constraint kind");
    return *settings;
}

template <class Settings, class Edit>
void EditCpuJoint(PhysicsDomain &domain, uint64_t id, Edit edit, bool clearPoseDrive = false)
{
    const auto &joint = *domain.cpuControl->FindJoint(id);
    auto value = RequireCpuSettings<Settings>(joint);
    auto collision = joint.enableCollision;
    edit(value, collision);
    domain.cpuControl->UpdateJoints({{id, std::move(value), collision, clearPoseDrive}});
}

JPH::GPU::Float4 JointVector(const glm::vec3 &value)
{
    if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z))
        throw std::invalid_argument("Joint anchor and axis must be finite");
    return {value.x, value.y, value.z, 0.0f};
}

void ApplySwingSettings(JointAuthor &author, const SwingTwistJointSettings &s)
{
    const auto finite = [](const auto &v) {
        for (int i = 0; i < v.length(); ++i)
            if (!std::isfinite(v[i]))
                return false;
        return true;
    };
    if (!finite(s.twistAxis) || !finite(s.planeAxis) || glm::dot(s.twistAxis, s.twistAxis) < 1e-12f ||
        glm::dot(s.planeAxis, s.planeAxis) < 1e-12f || !finite(s.targetOrientation) ||
        glm::dot(s.targetOrientation, s.targetOrientation) < 1e-12f || !finite(s.targetAngularVelocity))
        throw std::invalid_argument("SwingTwist requires finite nonzero frame and target quaternions");
    glm::vec3 x = glm::normalize(s.twistAxis);
    glm::vec3 z = s.planeAxis - x * glm::dot(x, s.planeAxis);
    if (glm::dot(z, z) < 1e-8f * glm::dot(s.planeAxis, s.planeAxis))
        throw std::invalid_argument("SwingTwist frame axes must be nonparallel");
    z = glm::normalize(z);
    const glm::quat frame = glm::normalize(glm::quat_cast(glm::mat3(x, glm::cross(z, x), z)));
    const glm::vec4 target = glm::normalize(s.targetOrientation);
    author.angularFrame() = {frame.x, frame.y, frame.z, frame.w};
    author.angularLimits() = {glm::radians(s.minimumTwistAngle), glm::radians(s.maximumTwistAngle),
                              glm::radians(s.planeHalfConeAngle), glm::radians(s.normalHalfConeAngle)};
    author.angularModes() = {uint32_t(s.swingType), uint32_t(s.swingMotorMode), uint32_t(s.twistMotorMode), 0};
    author.angularSpring() = {s.swingMotorStiffness, s.swingMotorDamping, s.twistMotorStiffness, s.twistMotorDamping};
    author.angularTorque() = {s.swingMotorMaxTorque, s.twistMotorMaxTorque, s.frictionTorque, 0.0f};
    author.angularTarget() = {target.x, target.y, target.z, target.w};
    author.angularVelocity() = JointVector(glm::radians(s.targetAngularVelocity));
}

void ApplySixSettings(JointAuthor &author, const SixDOFJointSettings &s)
{
    const auto finite = [](const auto &v) {
        for (int i = 0; i < v.length(); ++i)
            if (!std::isfinite(v[i]))
                return false;
        return true;
    };
    if (!finite(s.axis) || !finite(s.secondaryAxis) || glm::dot(s.axis, s.axis) < 1e-12f ||
        glm::dot(s.secondaryAxis, s.secondaryAxis) < 1e-12f || !finite(s.targetOrientation) ||
        glm::dot(s.targetOrientation, s.targetOrientation) < 1e-12f || !finite(s.targetPosition) ||
        !finite(s.targetVelocity) || !finite(s.targetAngularVelocity))
        throw std::invalid_argument("SixDOF requires finite nonzero frame/orientation and finite targets");
    glm::vec3 x = glm::normalize(s.axis), y = s.secondaryAxis - x * glm::dot(x, s.secondaryAxis);
    if (glm::dot(y, y) < 1e-8f * glm::dot(s.secondaryAxis, s.secondaryAxis))
        throw std::invalid_argument("SixDOF frame axes must be nonparallel");
    y = glm::normalize(y);
    const auto frame = glm::normalize(glm::quat_cast(glm::mat3(x, y, glm::cross(x, y))));
    const auto target = glm::normalize(s.targetOrientation);
    const uint32_t motions[] = {uint32_t(s.xMotion),        uint32_t(s.yMotion),        uint32_t(s.zMotion),
                                uint32_t(s.angularXMotion), uint32_t(s.angularYMotion), uint32_t(s.angularZMotion)};
    const uint32_t motors[] = {uint32_t(s.xMotorMode),        uint32_t(s.yMotorMode),
                               uint32_t(s.zMotorMode),        uint32_t(s.angularXMotorMode),
                               uint32_t(s.angularYMotorMode), uint32_t(s.angularZMotorMode)};
    uint32_t motionBits = 0, motorBits = 0;
    for (uint32_t i = 0; i < 6; ++i) {
        const float lo = i < 3 ? s.linearLimitMinimum[i] : s.angularLimitMinimum[i - 3];
        const float hi = i < 3 ? s.linearLimitMaximum[i] : s.angularLimitMaximum[i - 3];
        if (motions[i] > 2 || motors[i] > 3 || !std::isfinite(lo) || !std::isfinite(hi) || lo > hi ||
            (motions[i] == 1 && lo == hi) || (i >= 3 && (lo < -180 || hi > 180)))
            throw std::invalid_argument("SixDOF motion/motor modes or limits are invalid");
        motionBits |= motions[i] << (2 * i);
        motorBits |= motors[i] << (2 * i);
    }
    for (const auto &v : {s.linearLimitStiffness, s.linearLimitDamping, s.linearFrictionForce, s.angularFrictionTorque,
                          s.linearMotorStiffness, s.linearMotorDamping, s.linearMotorMaxForce, s.angularMotorStiffness,
                          s.angularMotorDamping, s.angularMotorMaxTorque})
        if (!finite(v) || glm::any(glm::lessThan(v, glm::vec3(0))))
            throw std::invalid_argument("SixDOF coefficients must be finite and nonnegative");
    author.angularFrame() = {frame.x, frame.y, frame.z, frame.w};
    author.angularTarget() = {target.x, target.y, target.z, target.w};
    author.angularVelocity() = JointVector(glm::radians(s.targetAngularVelocity));
    author.sixModes() = {motionBits, motorBits, 0, 0};
    author.sixLinearMin() = JointVector(s.linearLimitMinimum);
    author.sixLinearMax() = JointVector(s.linearLimitMaximum);
    author.sixAngularMin() = JointVector(glm::radians(s.angularLimitMinimum));
    author.sixAngularMax() = JointVector(glm::radians(s.angularLimitMaximum));
    author.sixLimitStiffness() = JointVector(s.linearLimitStiffness);
    author.sixLimitDamping() = JointVector(s.linearLimitDamping);
    author.sixLinearFriction() = JointVector(s.linearFrictionForce);
    author.sixAngularFriction() = JointVector(s.angularFrictionTorque);
    author.sixLinearMotorK() = JointVector(s.linearMotorStiffness);
    author.sixLinearMotorC() = JointVector(s.linearMotorDamping);
    author.sixLinearMotorForce() = JointVector(s.linearMotorMaxForce);
    author.sixAngularMotorK() = JointVector(s.angularMotorStiffness);
    author.sixAngularMotorC() = JointVector(s.angularMotorDamping);
    author.sixAngularMotorTorque() = JointVector(s.angularMotorMaxTorque);
    author.sixTargetPosition() = JointVector(s.targetPosition);
    author.sixTargetVelocity() = JointVector(s.targetVelocity);
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

void ValidateSliderLimits(float minimum, float maximum)
{
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum > 0.0f || maximum < 0.0f || minimum > maximum)
        throw std::invalid_argument("Slider limits must be ordered finite metres and span zero");
}

void RequireSlider(const JoltGpuScene::PublishedJoint &joint)
{
    if (joint.kind != JPH::GPU::JointSlider)
        throw std::logic_error("Constraint is not a slider");
}

void SetJointFlag(JointAuthor &author, uint32_t flag, bool enabled)
{
    author.bodies.w = enabled ? author.bodies.w | flag : author.bodies.w & ~flag;
}

float ClampHingeTarget(const JointAuthor &author, float target)
{
    return author.bodies.w & JPH::GPU::JointLimitsEnabled ? std::clamp(target, author.limits().x, author.limits().y)
                                                          : target;
}

JoltGpuControlWork PrepareJointReadWork(std::function<JoltGpuControlWork()> prepare, std::exception_ptr &error)
{
    try {
        auto work = prepare();
        if (auto continuation = std::move(work.continueWork))
            work.continueWork = [continuation = std::move(continuation), &error]() mutable {
                return PrepareJointReadWork(std::move(continuation), error);
            };
        return work;
    } catch (const JoltGpuQueryRequestError &) {
        error = std::current_exception();
        return {};
    }
}
} // namespace

PhysicsWorld::Backend PhysicsWorld::PrepareJointEndpoints(PhysicsDomain &domain, uint32_t ownerBody,
                                                          uint32_t connectedBody)
{
    const auto &owner = *FindBodyRegistration(ownerBody);
    const auto a = owner.backend;
    if (owner.transferTarget)
        throw std::invalid_argument("A joint endpoint is transferring its physics backend");
    auto b = Backend::StaticBoundary;
    if (connectedBody != Cpu::WorldBody) {
        if (ownerBody == connectedBody || &DomainForBody(connectedBody) != &domain)
            throw std::invalid_argument("Joint endpoints must be distinct bodies belonging to the same RuntimeWorld");
        const auto &connected = *FindBodyRegistration(connectedBody);
        if (connected.transferTarget)
            throw std::invalid_argument("A joint endpoint is transferring its physics backend");
        b = connected.backend;
    }
    const auto consumer = a == Backend::StaticBoundary ? b : a;
    if (consumer == Backend::StaticBoundary)
        throw std::invalid_argument("Joint requires a Rigidbody endpoint to select its physics backend");
    if (a != Backend::StaticBoundary && b != Backend::StaticBoundary && a != b)
        throw std::invalid_argument("Joint endpoints on different physics backends are unsupported");
    if (a == Backend::StaticBoundary)
        EnsureStaticBoundary(domain, ownerBody, consumer);
    if (b == Backend::StaticBoundary && connectedBody != Cpu::WorldBody)
        EnsureStaticBoundary(domain, connectedBody, consumer);
    return consumer;
}

PhysicsDomain *PhysicsWorld::FindCpuJointDomain(uint64_t id) const noexcept
{
    const auto found = m_cpuJointDomains.find(id);
    return found != m_cpuJointDomains.end() ? found->second : nullptr;
}

void PhysicsWorld::RetireCpuJointRoutes(const std::vector<uint64_t> &ids) noexcept
{
    for (const auto id : ids)
        m_cpuJointDomains.erase(id);
}

void PhysicsWorld::RetireCpuJointRoutes(PhysicsDomain &domain) noexcept
{
    for (auto it = m_cpuJointDomains.begin(); it != m_cpuJointDomains.end();) {
        if (it->second == &domain)
            it = m_cpuJointDomains.erase(it);
        else
            ++it;
    }
}

uint64_t PhysicsWorld::CreateHingeConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                             const glm::vec3 &localBakedAnchor, const glm::vec3 &ownerAxis,
                                             const glm::vec3 &localAnchor, bool useLimits, float minimumAngle,
                                             float maximumAngle, bool enableCollision)
{
    auto &domain = DomainForBody(bodyIdA);
    RequireOwnerThread();
    ValidateHingeLimits(minimumAngle, maximumAngle);
    (void)JointVector(ownerAxis);
    if (glm::dot(ownerAxis, ownerAxis) <= 1e-12f)
        throw std::invalid_argument("Hinge axis must be non-zero");
    if (PrepareJointEndpoints(domain, bodyIdA, bodyIdB) == Backend::Cpu)
        return StageCpuJoint(domain, m_cpuJointDomains, owner,
                             {bodyIdA, bodyIdB, localBakedAnchor, {}, localAnchor, {}},
                             Cpu::Hinge{ownerAxis, useLimits, minimumAngle, maximumAngle}, enableCollision);
    JointAuthor author{};
    author.SetKind(JPH::GPU::JointHinge);
    author.bodies.w = JPH::GPU::JointEnabled | (enableCollision ? JPH::GPU::JointCollisionEnabled : 0) |
                      (useLimits ? JPH::GPU::JointLimitsEnabled : 0);
    author.ownerAnchor() = JointVector(localBakedAnchor);
    author.ownerAxis() = JointVector(glm::normalize(ownerAxis));
    author.ownerLocalAnchor() = JointVector(localAnchor);
    author.limits() = {minimumAngle, maximumAngle, 0.0f, 0.0f};
    RequireGpu(domain);
    return m_gpuScene->CreateJoint(owner, bodyIdA, bodyIdB, author);
}

uint64_t PhysicsWorld::CreateSliderConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                              const glm::vec3 &localBakedAnchor, const glm::vec3 &ownerAxis,
                                              const glm::vec3 &localAnchor, bool useLimits, float minimumDistance,
                                              float maximumDistance, bool enableCollision)
{
    auto &domain = DomainForBody(bodyIdA);
    RequireOwnerThread();
    ValidateSliderLimits(minimumDistance, maximumDistance);
    (void)JointVector(ownerAxis);
    if (glm::dot(ownerAxis, ownerAxis) <= 1e-12f)
        throw std::invalid_argument("Slider axis must be non-zero");
    if (PrepareJointEndpoints(domain, bodyIdA, bodyIdB) == Backend::Cpu)
        return StageCpuJoint(domain, m_cpuJointDomains, owner,
                             {bodyIdA, bodyIdB, localBakedAnchor, {}, localAnchor, {}},
                             Cpu::Slider{ownerAxis, useLimits, minimumDistance, maximumDistance}, enableCollision);
    JointAuthor author{};
    author.SetKind(JPH::GPU::JointSlider);
    author.bodies.w = JPH::GPU::JointEnabled | (enableCollision ? JPH::GPU::JointCollisionEnabled : 0) |
                      (useLimits ? JPH::GPU::JointLimitsEnabled : 0);
    author.ownerAnchor() = JointVector(localBakedAnchor);
    author.ownerAxis() = JointVector(glm::normalize(ownerAxis));
    author.ownerLocalAnchor() = JointVector(localAnchor);
    author.limits() = {minimumDistance, maximumDistance, 0.0f, 0.0f};
    RequireGpu(domain);
    return m_gpuScene->CreateJoint(owner, bodyIdA, bodyIdB, author);
}

uint64_t PhysicsWorld::CreateFixedConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                             const glm::vec3 &localBakedAnchor, const glm::vec3 &localAnchor,
                                             bool enableCollision)
{
    auto &domain = DomainForBody(bodyIdA);
    RequireOwnerThread();
    if (PrepareJointEndpoints(domain, bodyIdA, bodyIdB) == Backend::Cpu)
        return StageCpuJoint(domain, m_cpuJointDomains, owner,
                             {bodyIdA, bodyIdB, localBakedAnchor, {}, localAnchor, {}}, Cpu::Fixed{}, enableCollision);
    JointAuthor author{};
    author.SetKind(JPH::GPU::JointFixed);
    author.bodies.w = JPH::GPU::JointEnabled | (enableCollision ? JPH::GPU::JointCollisionEnabled : 0);
    author.ownerAnchor() = JointVector(localBakedAnchor);
    author.ownerLocalAnchor() = JointVector(localAnchor);
    RequireGpu(domain);
    return m_gpuScene->CreateJoint(owner, bodyIdA, bodyIdB, author);
}

void PhysicsWorld::SetConstraintCollisionEnabled(uint64_t constraintId, bool enabled)
{
    RequireOwnerThread();
    if (auto *cpu = FindCpuJointDomain(constraintId)) {
        const auto &joint = *cpu->cpuControl->FindJoint(constraintId);
        cpu->cpuControl->UpdateJoints({{constraintId, joint.settings, enabled}});
        return;
    }
    auto &domain = DomainForBody(RequireGpuJoint(m_gpuScene.get(), constraintId).bodyIdA);
    RequireGpu(domain);
    m_gpuScene->UpdateJoint(
        constraintId, [=](JointAuthor &author) { SetJointFlag(author, JPH::GPU::JointCollisionEnabled, enabled); });
}

uint64_t PhysicsWorld::CreateDistanceConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                                const glm::vec3 &localBakedAnchor, const glm::vec3 &localAnchor,
                                                const glm::vec3 &connectedBakedAnchor,
                                                const glm::vec3 &connectedLocalAnchor, bool autoDistance, float minimum,
                                                float maximum, float stiffness, float damping, bool enableCollision)
{
    auto &domain = DomainForBody(bodyIdA);
    RequireOwnerThread();
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum < 0.0f || maximum < minimum ||
        !std::isfinite(stiffness) || stiffness < 0.0f || !std::isfinite(damping) || damping < 0.0f)
        throw std::invalid_argument(
            "Distance limits and spring coefficients must be finite ordered non-negative values");
    if (PrepareJointEndpoints(domain, bodyIdA, bodyIdB) == Backend::Cpu)
        return StageCpuJoint(
            domain, m_cpuJointDomains, owner,
            {bodyIdA, bodyIdB, localBakedAnchor, connectedBakedAnchor, localAnchor, connectedLocalAnchor},
            Cpu::Distance{autoDistance, minimum, maximum, stiffness, damping}, enableCollision);
    JointAuthor author{};
    author.SetKind(JPH::GPU::JointDistance);
    author.bodies.w = JPH::GPU::JointEnabled | (enableCollision ? JPH::GPU::JointCollisionEnabled : 0) |
                      (autoDistance ? JPH::GPU::JointAutoDistance : 0);
    author.ownerAnchor() = JointVector(localBakedAnchor);
    author.ownerLocalAnchor() = JointVector(localAnchor);
    author.connectedAnchor() = JointVector(connectedBakedAnchor);
    author.connectedLocalAnchor() = JointVector(connectedLocalAnchor);
    author.limits() = {minimum, maximum, 0.0f, 0.0f};
    author.linearSpring() = {stiffness, damping, 0.0f, 0.0f};
    RequireGpu(domain);
    return m_gpuScene->CreateJoint(owner, bodyIdA, bodyIdB, author);
}

void PhysicsWorld::UpdateDistanceConstraintSettings(uint64_t constraintId, bool updateLimits, float minimum,
                                                    float maximum, float stiffness, float damping, bool enableCollision)
{
    RequireOwnerThread();
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum < 0.0f || maximum < minimum ||
        !std::isfinite(stiffness) || stiffness < 0.0f || !std::isfinite(damping) || damping < 0.0f)
        throw std::invalid_argument(
            "Distance limits and spring coefficients must be finite ordered non-negative values");
    if (auto *cpu = FindCpuJointDomain(constraintId)) {
        EditCpuJoint<Cpu::Distance>(*cpu, constraintId, [=](auto &value, bool &collision) {
            if (value.autoDistance == updateLimits)
                throw std::logic_error("Changing Distance auto mode requires rebuilding its captured reference");
            value.minimum = minimum;
            value.maximum = maximum;
            value.stiffness = stiffness;
            value.damping = damping;
            collision = enableCollision;
        });
        return;
    }
    auto &domain = DomainForBody(RequireGpuJoint(m_gpuScene.get(), constraintId).bodyIdA);
    RequireGpu(domain);
    const auto &joint = RequireGpuJoint(m_gpuScene.get(), constraintId);
    if (joint.kind != JPH::GPU::JointDistance)
        throw std::logic_error("Constraint is not a distance joint");
    if (bool(joint.author.bodies.w & JPH::GPU::JointAutoDistance) == updateLimits)
        throw std::logic_error("Changing Distance auto mode requires rebuilding its captured reference");
    m_gpuScene->UpdateJoint(constraintId, [=](JointAuthor &author) {
        SetJointFlag(author, JPH::GPU::JointCollisionEnabled, enableCollision);
        author.limits() = {minimum, maximum, 0.0f, 0.0f};
        author.linearSpring() = {stiffness, damping, 0.0f, 0.0f};
    });
}

float PhysicsWorld::GetDistanceConstraintLength(uint64_t constraintId) const
{
    RequireOwnerThread();
    if (auto *cpu = FindCpuJointDomain(constraintId)) {
        RunCpuControl(*cpu);
        return cpu->cpuScene->Constraints().ReadDistance(constraintId);
    }
    auto &domain = DomainForBody(RequireGpuJoint(m_gpuScene.get(), constraintId).bodyIdA);
    float motion[2];
    ReadJointMotion(domain, {constraintId}, motion, JPH::GPU::JointDistance);
    return motion[0];
}

void PhysicsWorld::UpdateHingeConstraintSettings(uint64_t constraintId, bool useLimits, float minimum, float maximum,
                                                 bool enableCollision)
{
    RequireOwnerThread();
    ValidateHingeLimits(minimum, maximum);
    if (auto *cpu = FindCpuJointDomain(constraintId)) {
        EditCpuJoint<Cpu::Hinge>(*cpu, constraintId, [=](auto &value, bool &collision) {
            value.useLimits = useLimits;
            value.minimum = minimum;
            value.maximum = maximum;
            collision = enableCollision;
        });
        return;
    }
    auto &domain = DomainForBody(RequireGpuJoint(m_gpuScene.get(), constraintId).bodyIdA);
    RequireGpu(domain);
    RequireHinge(RequireGpuJoint(m_gpuScene.get(), constraintId));
    m_gpuScene->UpdateJoint(constraintId, [=](JointAuthor &author) {
        SetJointFlag(author, JPH::GPU::JointLimitsEnabled, useLimits);
        SetJointFlag(author, JPH::GPU::JointCollisionEnabled, enableCollision);
        author.limits() = {minimum, maximum, 0.0f, 0.0f};
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
    if (auto *cpu = FindCpuJointDomain(constraintId)) {
        EditCpuJoint<Cpu::Hinge>(*cpu, constraintId, [=](auto &value, bool &) {
            value.motorEnabled = enabled;
            value.targetAngle = targetAngle;
            value.stiffness = stiffness;
            value.damping = damping;
            value.maxTorque = maxTorque;
            value.jointDamping = jointDamping;
            value.frictionTorque = frictionTorque;
            value.armature = armature;
        });
        return;
    }
    auto &domain = DomainForBody(RequireGpuJoint(m_gpuScene.get(), constraintId).bodyIdA);
    RequireGpu(domain);
    RequireHinge(RequireGpuJoint(m_gpuScene.get(), constraintId));
    m_gpuScene->UpdateJoint(constraintId, [=](JointAuthor &author) {
        SetJointFlag(author, JPH::GPU::JointMotorEnabled, enabled);
        author.motor() = {ClampHingeTarget(author, targetAngle), stiffness, damping, maxTorque};
        author.passive() = {jointDamping, frictionTorque, armature, 0.0f};
    });
}

void PhysicsWorld::UpdateHingeConstraintTarget(uint64_t constraintId, float targetAngle)
{
    RequireOwnerThread();
    if (!std::isfinite(targetAngle))
        throw std::invalid_argument("Hinge motor target must be finite radians");
    if (auto *cpu = FindCpuJointDomain(constraintId)) {
        EditCpuJoint<Cpu::Hinge>(
            *cpu, constraintId, [=](auto &value, bool &) { value.targetAngle = targetAngle; }, true);
        return;
    }
    auto &domain = DomainForBody(RequireGpuJoint(m_gpuScene.get(), constraintId).bodyIdA);
    RequireGpu(domain);
    const auto &joint = RequireGpuJoint(m_gpuScene.get(), constraintId);
    RequireHinge(joint);
    m_gpuScene->UpdateJoint(
        constraintId, [=](JointAuthor &author) { author.motor().x = ClampHingeTarget(author, targetAngle); },
        RagdollTargetReset(joint.bodyIdA));
}

void PhysicsWorld::SetHingeMotorTargets(const std::vector<HingeJoint *> &joints, const float *targets)
{
    RequireOwnerThread();
    if (!targets && !joints.empty())
        throw std::invalid_argument("Hinge targets require input storage");
    struct Target
    {
        HingeJoint *joint;
        float degrees;
    };
    struct Group
    {
        PhysicsDomain *domain;
        std::vector<CpuControl::JointUpdate> cpu;
        std::vector<JoltGpuScene::JointUpdate> gpu;
    };
    std::vector<Target> prepared;
    prepared.reserve(joints.size());
    std::vector<Group> groups;
    std::unordered_map<PhysicsDomain *, size_t> owners;
    // Validate every identity and coefficient before queuing any author update.
    for (size_t row = 0; row < joints.size(); ++row) {
        if (!joints[row] || !joints[row]->GetPhysicsConstraintId())
            throw std::logic_error("Hinge targets require bound joints");
        const auto id = joints[row]->GetPhysicsConstraintId();
        auto &joint = *joints[row];
        auto &domain = DomainForComponent(joint);
        const float degrees = glm::degrees(targets[row]);
        if (!std::isfinite(targets[row]) || !std::isfinite(degrees))
            throw std::invalid_argument("Hinge targets must be finite representable radians");
        const auto [entry, inserted] = owners.emplace(&domain, groups.size());
        if (inserted)
            groups.push_back({&domain, {}, {}});
        auto &group = groups[entry->second];
        const auto *cpu = FindCpuJointDomain(id);
        const auto target =
            glm::radians(joint.m_useLimits ? std::clamp(degrees, joint.m_minimumAngle, joint.m_maximumAngle) : degrees);
        if (cpu) {
            if (cpu != &domain)
                throw std::logic_error("CPU hinge no longer belongs to its author RuntimeWorld");
            const auto &authored = *domain.cpuControl->FindJoint(id);
            auto next = RequireCpuSettings<Cpu::Hinge>(authored);
            next.targetAngle = target;
            group.cpu.push_back({id, next, authored.enableCollision, true});
        } else {
            const auto &authored = RequireGpuJoint(m_gpuScene.get(), id);
            RequireHinge(authored);
            if (&DomainForBody(authored.bodyIdA) != &domain)
                throw std::logic_error("GPU hinge no longer belongs to its author RuntimeWorld");
            RequireGpu(domain);
            auto author = authored.author;
            author.motor().x = target;
            group.gpu.push_back({id, author, RagdollTargetReset(authored.bodyIdA)});
        }
        prepared.push_back({&joint, degrees});
    }
    for (auto &group : groups) {
        if (!group.cpu.empty())
            group.domain->cpuControl->UpdateJoints(std::move(group.cpu));
        if (!group.gpu.empty())
            m_gpuScene->UpdateJoints(std::move(group.gpu));
    }
    for (const auto &target : prepared) {
        target.joint->m_motorTargetAngle = target.degrees;
        ClaimRagdollTarget(*target.joint);
    }
}

void PhysicsWorld::ReadHingeMotion(const std::vector<HingeJoint *> &joints, float *output) const
{
    RequireOwnerThread();
    if (!output && !joints.empty())
        throw std::invalid_argument("Hinge motion requires output storage");
    struct Row
    {
        uint64_t id;
        size_t row;
    };
    struct Group
    {
        PhysicsDomain *domain;
        std::vector<Row> cpu;
        std::vector<uint64_t> gpuIds;
        std::vector<size_t> gpuRows;
    };
    std::vector<Group> groups;
    std::unordered_map<PhysicsDomain *, size_t> owners;
    for (size_t row = 0; row < joints.size(); ++row) {
        const auto *joint = joints[row];
        if (!joint || !joint->GetPhysicsConstraintId())
            throw std::logic_error("Hinge motion requires bound joints");
        const auto id = joint->GetPhysicsConstraintId();
        auto &domain = DomainForComponent(*joint);
        const auto [entry, inserted] = owners.emplace(&domain, groups.size());
        if (inserted)
            groups.push_back({&domain, {}, {}, {}});
        auto &group = groups[entry->second];
        if (const auto *cpu = FindCpuJointDomain(id)) {
            if (cpu != &domain)
                throw std::logic_error("CPU hinge no longer belongs to its author RuntimeWorld");
            (void)RequireCpuSettings<Cpu::Hinge>(*domain.cpuControl->FindJoint(id));
            group.cpu.push_back({id, row});
        } else {
            const auto &authored = RequireGpuJoint(m_gpuScene.get(), id);
            RequireHinge(authored);
            if (&DomainForBody(authored.bodyIdA) != &domain)
                throw std::logic_error("GPU hinge no longer belongs to its author RuntimeWorld");
            group.gpuIds.push_back(id);
            group.gpuRows.push_back(row);
        }
    }
    std::vector<float> aggregate(joints.size() * 2);
    for (auto &group : groups) {
        if (!group.cpu.empty()) {
            RunCpuControl(*group.domain);
            for (const auto &row : group.cpu) {
                const auto state = group.domain->cpuScene->Constraints().ReadHinge(row.id);
                aggregate[2 * row.row] = state.x;
                aggregate[2 * row.row + 1] = state.y;
            }
        }
        if (!group.gpuIds.empty()) {
            std::vector<float> values(group.gpuIds.size() * 2);
            ReadJointMotion(*group.domain, group.gpuIds, values.data(), JPH::GPU::JointHinge);
            for (size_t local = 0; local < group.gpuRows.size(); ++local) {
                aggregate[2 * group.gpuRows[local]] = values[2 * local];
                aggregate[2 * group.gpuRows[local] + 1] = values[2 * local + 1];
            }
        }
    }
    std::copy(aggregate.begin(), aggregate.end(), output);
}

void PhysicsWorld::ReadJointMotion(PhysicsDomain &domain, const std::vector<uint64_t> &ids, float *output,
                                   uint32_t kind) const
{
    RequireOwnerThread();
    if (!output && !ids.empty())
        throw std::invalid_argument("Joint motion requires output storage");
    if (ids.empty())
        return;
    JoltGpuPreparedJointMotion prepared;
    std::exception_ptr requestError;
    RunControl(domain, [&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        return PrepareJointReadWork(
            [&]() -> JoltGpuControlWork {
                std::unordered_map<uint64_t, const JoltGpuScene::PublishedJoint *> identities;
                identities.reserve(publication->GetJoints().size());
                for (const auto &joint : publication->GetJoints())
                    identities.emplace(joint.id, &joint);
                std::vector<uint32_t> rows;
                rows.reserve(ids.size());
                for (const auto id : ids) {
                    const auto found = identities.find(id);
                    if (found == identities.end() || found->second->kind != kind)
                        throw JoltGpuQueryRequestError(
                            "Joint motion requires an accepted identity of the requested kind");
                    rows.push_back(found->second->row);
                }
                prepared = m_gpuJoints->PrepareReadMotion(publication, rows, kind);
                return std::move(prepared.work);
            },
            requestError);
    });
    if (requestError)
        std::rethrow_exception(requestError);
    if (!prepared.result)
        throw std::runtime_error("GPU joint motion did not produce a result");
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    prepared.result->CopyMotion(output, ids.size() * (kind == JPH::GPU::JointSixDOF       ? 7
                                                      : kind == JPH::GPU::JointSwingTwist ? 4
                                                                                          : 2));
}

void PhysicsWorld::UpdateSliderConstraintSettings(uint64_t constraintId, bool useLimits, float minimum, float maximum,
                                                  bool enableCollision)
{
    RequireOwnerThread();
    ValidateSliderLimits(minimum, maximum);
    if (auto *cpu = FindCpuJointDomain(constraintId)) {
        EditCpuJoint<Cpu::Slider>(*cpu, constraintId, [=](auto &value, bool &collision) {
            value.useLimits = useLimits;
            value.minimum = minimum;
            value.maximum = maximum;
            collision = enableCollision;
        });
        return;
    }
    auto &domain = DomainForBody(RequireGpuJoint(m_gpuScene.get(), constraintId).bodyIdA);
    RequireGpu(domain);
    RequireSlider(RequireGpuJoint(m_gpuScene.get(), constraintId));
    m_gpuScene->UpdateJoint(constraintId, [=](JointAuthor &author) {
        SetJointFlag(author, JPH::GPU::JointLimitsEnabled, useLimits);
        SetJointFlag(author, JPH::GPU::JointCollisionEnabled, enableCollision);
        author.limits() = {minimum, maximum, 0.0f, 0.0f};
    });
}

void PhysicsWorld::DestroyConstraintsForBody(uint32_t bodyId)
{
    RequireOwnerThread();
    if (FindBodyRegistration(bodyId)) {
        auto &domain = DomainForBody(bodyId);
        if (domain.cpuControl)
            RetireCpuJointRoutes(domain.cpuControl->DestroyJointsForBody(bodyId));
    }
    if (m_gpuScene && m_gpuScene->FindBody(bodyId).bodyId != UINT32_MAX)
        m_gpuScene->DestroyJointsForBody(bodyId);
}

void PhysicsWorld::DestroyConstraint(uint64_t constraintId)
{
    if (constraintId == 0)
        return;
    RequireOwnerThread();
    if (auto *cpu = FindCpuJointDomain(constraintId)) {
        cpu->cpuControl->DestroyJoint(constraintId);
        m_cpuJointDomains.erase(constraintId);
        return;
    }
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
    RequireOwnerThread();
    if (auto *cpu = FindCpuJointDomain(constraintId)) {
        RunCpuControl(*cpu);
        return cpu->cpuScene->Constraints().ReadHinge(constraintId);
    }
    auto &domain = DomainForBody(RequireGpuJoint(m_gpuScene.get(), constraintId).bodyIdA);
    float motion[2];
    ReadJointMotion(domain, {constraintId}, motion, JPH::GPU::JointHinge);
    return {motion[0], motion[1]};
}

float PhysicsWorld::GetSliderConstraintPosition(uint64_t constraintId) const
{
    RequireOwnerThread();
    if (auto *cpu = FindCpuJointDomain(constraintId)) {
        RunCpuControl(*cpu);
        return cpu->cpuScene->Constraints().ReadSliderPosition(constraintId);
    }
    auto &domain = DomainForBody(RequireGpuJoint(m_gpuScene.get(), constraintId).bodyIdA);
    float motion[2];
    ReadJointMotion(domain, {constraintId}, motion, JPH::GPU::JointSlider);
    return motion[0];
}

uint64_t PhysicsWorld::CreateSwingTwistConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                                  const glm::vec3 &localBakedAnchor, const glm::vec3 &localAnchor,
                                                  const SwingTwistJointSettings &settings, bool enableCollision)
{
    auto &domain = DomainForBody(bodyIdA);
    RequireOwnerThread();
    if (PrepareJointEndpoints(domain, bodyIdA, bodyIdB) == Backend::Cpu)
        return StageCpuJoint(domain, m_cpuJointDomains, owner,
                             {bodyIdA, bodyIdB, localBakedAnchor, {}, localAnchor, {}}, settings, enableCollision);
    JointAuthor author{};
    author.SetKind(JPH::GPU::JointSwingTwist);
    author.bodies.w = JPH::GPU::JointEnabled | (enableCollision ? JPH::GPU::JointCollisionEnabled : 0);
    author.ownerAnchor() = JointVector(localBakedAnchor);
    author.ownerLocalAnchor() = JointVector(localAnchor);
    ApplySwingSettings(author, settings);
    RequireGpu(domain);
    return m_gpuScene->CreateJoint(owner, bodyIdA, bodyIdB, author);
}

void PhysicsWorld::UpdateSwingTwistConstraint(uint64_t constraintId, const SwingTwistJointSettings &settings,
                                              bool enableCollision, bool claimTarget)
{
    RequireOwnerThread();
    if (auto *cpu = FindCpuJointDomain(constraintId)) {
        EditCpuJoint<SwingTwistJointSettings>(
            *cpu, constraintId,
            [&](auto &value, bool &enabled) {
                value = settings;
                enabled = enableCollision;
            },
            claimTarget);
        return;
    }
    const auto &joint = RequireGpuJoint(m_gpuScene.get(), constraintId);
    RequireGpu(DomainForBody(joint.bodyIdA));
    if (joint.kind != JPH::GPU::JointSwingTwist)
        throw std::logic_error("Constraint is not SwingTwist");
    auto updated = joint.author;
    ApplySwingSettings(updated, settings);
    const auto &before = joint.author.angularFrame();
    const auto &after = updated.angularFrame();
    if (before.x != after.x || before.y != after.y || before.z != after.z || before.w != after.w)
        throw std::logic_error("SwingTwist frame changes require rebuilding the constraint");
    SetJointFlag(updated, JPH::GPU::JointCollisionEnabled, enableCollision);
    m_gpuScene->UpdateJoint(
        constraintId,
        [=](JointAuthor &author) {
            author.angularLimits() = updated.angularLimits();
            author.angularModes() = updated.angularModes();
            author.angularSpring() = updated.angularSpring();
            author.angularTorque() = updated.angularTorque();
            author.angularTarget() = updated.angularTarget();
            author.angularVelocity() = updated.angularVelocity();
            SetJointFlag(author, JPH::GPU::JointCollisionEnabled, enableCollision);
        },
        claimTarget ? RagdollTargetReset(joint.bodyIdA) : 0u);
}

glm::quat PhysicsWorld::GetAngularConstraintOrientation(uint64_t id) const
{
    RequireOwnerThread();
    if (auto *cpu = FindCpuJointDomain(id)) {
        RunCpuControl(*cpu);
        return cpu->cpuScene->Constraints().ReadOrientation(id);
    }
    const auto &joint = RequireGpuJoint(m_gpuScene.get(), id);
    auto &domain = DomainForBody(joint.bodyIdA);
    if (joint.kind == JPH::GPU::JointSixDOF) {
        float values[7];
        ReadJointMotion(domain, {id}, values, JPH::GPU::JointSixDOF);
        return {values[6], values[3], values[4], values[5]};
    }
    if (joint.kind != JPH::GPU::JointSwingTwist)
        throw std::logic_error("Constraint has no angular orientation output");
    float values[4];
    ReadJointMotion(domain, {id}, values, JPH::GPU::JointSwingTwist);
    return {values[3], values[0], values[1], values[2]};
}

uint64_t PhysicsWorld::CreateSixDOFConstraint(PhysicsConstraintOwner &owner, uint32_t ownerBody, uint32_t connectedBody,
                                              const glm::vec3 &baked, const glm::vec3 &raw,
                                              const SixDOFJointSettings &settings, bool collision)
{
    RequireOwnerThread();
    auto &domain = DomainForBody(ownerBody);
    if (PrepareJointEndpoints(domain, ownerBody, connectedBody) == Backend::Cpu)
        return StageCpuJoint(domain, m_cpuJointDomains, owner, {ownerBody, connectedBody, baked, {}, raw, {}}, settings,
                             collision);
    JointAuthor author(JPH::GPU::JointSixDOF);
    author.bodies.w = JPH::GPU::JointEnabled | (collision ? JPH::GPU::JointCollisionEnabled : 0u);
    author.ownerAnchor() = JointVector(baked);
    author.ownerLocalAnchor() = JointVector(raw);
    ApplySixSettings(author, settings);
    RequireGpu(domain);
    return m_gpuScene->CreateJoint(owner, ownerBody, connectedBody, std::move(author));
}
void PhysicsWorld::UpdateSixDOFConstraint(uint64_t id, const SixDOFJointSettings &settings, bool collision,
                                          bool claimTarget)
{
    RequireOwnerThread();
    if (auto *cpu = FindCpuJointDomain(id)) {
        EditCpuJoint<SixDOFJointSettings>(
            *cpu, id,
            [&](auto &value, bool &enabled) {
                value = settings;
                enabled = collision;
            },
            claimTarget);
        return;
    }
    const auto &joint = RequireGpuJoint(m_gpuScene.get(), id);
    if (joint.kind != JPH::GPU::JointSixDOF)
        throw std::logic_error("Constraint is not SixDOF");
    auto updated = joint.author;
    ApplySixSettings(updated, settings);
    if (std::memcmp(&updated.angularFrame(), &joint.author.angularFrame(), sizeof(JPH::GPU::Float4)))
        throw std::logic_error("SixDOF frame changes require rebuilding the constraint");
    SetJointFlag(updated, JPH::GPU::JointCollisionEnabled, collision);
    RequireGpu(DomainForBody(joint.bodyIdA));
    m_gpuScene->UpdateJoint(
        id, [updated = std::move(updated)](JointAuthor &author) { author = updated; },
        claimTarget ? RagdollTargetReset(joint.bodyIdA) : 0u);
}
glm::vec3 PhysicsWorld::GetSixDOFConstraintPosition(uint64_t id) const
{
    RequireOwnerThread();
    if (auto *cpu = FindCpuJointDomain(id)) {
        RunCpuControl(*cpu);
        return cpu->cpuScene->Constraints().ReadSixDOFPosition(id);
    }
    const auto &joint = RequireGpuJoint(m_gpuScene.get(), id);
    if (joint.kind != JPH::GPU::JointSixDOF)
        throw std::logic_error("Constraint is not SixDOF");
    float values[7];
    ReadJointMotion(DomainForBody(joint.bodyIdA), {id}, values, JPH::GPU::JointSixDOF);
    return {values[0], values[1], values[2]};
}

} // namespace infernux
