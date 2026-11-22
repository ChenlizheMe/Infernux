#include "../HingeJoint.h"
#include "../Rigidbody.h"
#include "../RuntimeWorld.h"
#include "JointSettings.h"
#include "JoltCpuControl.h"
#include "JoltGpuJoints.h"
#include "PhysicsDomain.h"
#include "PhysicsWorld.h"
#include "inxphygpu/InxPhyGpuSceneRuntime.h"

#include <algorithm>
#include <cmath>
#include <glm/gtc/constants.hpp>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace infernux
{
namespace
{
using JPH::GPU::JointAuthor;
using namespace physics::inxphygpu;

using Cpu = JoltCpuConstraints;
using CpuControl = JoltCpuControl;

uint64_t StageCpuJoint(PhysicsDomain &domain, std::unordered_map<uint64_t, PhysicsDomain *> &routes,
                       PhysicsConstraintOwner &owner, Cpu::Binding binding, Cpu::Settings settings, bool collision)
{
    const auto id = PhysicsECSStore::Instance().AllocateJointIdentity();
    if (!routes.emplace(id, &domain).second)
        throw std::logic_error("CPU joint route already owns its public identity");
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

glm::quat ValidateSwingSettings(const SwingTwistJointSettings &s)
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
    return glm::normalize(glm::quat_cast(glm::mat3(x, glm::cross(z, x), z)));
}

void ApplySwingSettings(JointAuthor &author, const SwingTwistJointSettings &s, const glm::quat &frame)
{
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

struct SixDOFFrame
{
    glm::quat orientation;
    uint32_t motion, motor;
};

SixDOFFrame ValidateSixSettings(const SixDOFJointSettings &s)
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
    return {frame, motionBits, motorBits};
}

void ApplySixSettings(JointAuthor &author, const SixDOFJointSettings &s, const SixDOFFrame &frame)
{
    const auto target = glm::normalize(s.targetOrientation);
    author.angularFrame() = {frame.orientation.x, frame.orientation.y, frame.orientation.z, frame.orientation.w};
    author.angularTarget() = {target.x, target.y, target.z, target.w};
    author.angularVelocity() = JointVector(glm::radians(s.targetAngularVelocity));
    author.sixModes() = {frame.motion, frame.motor, 0, 0};
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

void ValidateSliderLimits(float minimum, float maximum)
{
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum > 0.0f || maximum < 0.0f || minimum > maximum)
        throw std::invalid_argument("Slider limits must be ordered finite metres and span zero");
}

template <class Settings> Settings &RequireSceneJointSettings(PhysicsSceneJointAuthor &joint)
{
    auto *settings = std::get_if<Settings>(&joint.settings);
    if (!settings)
        throw std::logic_error("GPU joint does not have the requested constraint kind");
    return *settings;
}

void SetJointFlag(JointAuthor &author, uint32_t flag, bool enabled)
{
    author.bodies.w = enabled ? author.bodies.w | flag : author.bodies.w & ~flag;
}

float ClampHingeTarget(const InxPhyGpuSceneHinge &hinge, float target)
{
    return hinge.useLimits ? std::clamp(target, hinge.minimum, hinge.maximum) : target;
}

} // namespace

PhysicsWorld::JointBackend PhysicsWorld::PrepareJointEndpoints(PhysicsDomain &domain, uint32_t ownerBody,
                                                               uint32_t connectedBody)
{
    const auto *owner = FindBodyRegistration(ownerBody);
    if (!owner || &DomainForBody(ownerBody) != &domain)
        throw std::invalid_argument("Joint owner must belong to its Scene physics domain");
    const auto target = domain.UsesGpu() ? Backend::Gpu : Backend::Cpu;
    const auto requireEndpoint = [&](uint32_t bodyId) {
        const auto *body = FindBodyRegistration(bodyId);
        if (!body || &DomainForBody(bodyId) != &domain)
            throw std::invalid_argument("Joint endpoints must belong to the same Scene physics domain");
        if (body->backend == Backend::StaticBoundary)
            EnsureStaticBoundary(domain, bodyId);
        else if (body->backend != target)
            throw std::logic_error("Joint endpoint differs from its Scene physics backend");
    };
    requireEndpoint(ownerBody);
    if (connectedBody != Cpu::WorldBody) {
        if (ownerBody == connectedBody)
            throw std::invalid_argument("Joint endpoints must be distinct bodies");
        requireEndpoint(connectedBody);
        if (owner->backend == Backend::StaticBoundary &&
            FindBodyRegistration(connectedBody)->backend == Backend::StaticBoundary)
            throw std::invalid_argument("Joint requires a Rigidbody endpoint");
    } else if (owner->backend == Backend::StaticBoundary)
        throw std::invalid_argument("Joint requires a Rigidbody endpoint");
    return domain.UsesGpu() ? JointBackend::Gpu : JointBackend::Cpu;
}

PhysicsDomain &PhysicsWorld::RequireJointDomain(uint64_t id) const
{
    const auto found = m_jointDomains.find(id);
    if (found == m_jointDomains.end())
        throw std::invalid_argument("Physics joint identity is invalid or retired");
    return *found->second;
}

PhysicsDomain *PhysicsWorld::FindCpuJointDomain(uint64_t id) const noexcept
{
    const auto found = m_jointDomains.find(id);
    return found != m_jointDomains.end() && !found->second->UsesGpu() ? found->second : nullptr;
}

void PhysicsWorld::RetireCpuJointRoutes(const std::vector<uint64_t> &ids) noexcept
{
    for (const auto id : ids)
        m_jointDomains.erase(id);
}

void PhysicsWorld::RetireCpuJointRoutes(PhysicsDomain &domain) noexcept
{
    if (domain.UsesGpu())
        return;
    for (auto it = m_jointDomains.begin(); it != m_jointDomains.end();) {
        if (it->second == &domain)
            it = m_jointDomains.erase(it);
        else
            ++it;
    }
}

uint64_t PhysicsWorld::StageGpuJoint(PhysicsDomain &domain, PhysicsConstraintOwner &owner,
                                     PhysicsSceneJointAuthor author, JointAuthor packed)
{
    if (!domain.UsesGpu())
        RequireGpu(domain);
    const auto id = PhysicsECSStore::Instance().AllocateJointIdentity();
    author.owner = &owner;
    m_jointDomains.emplace(id, &domain);
    try {
        const auto &binding = domain.gpuJointAuthors.emplace(id, std::move(author)).first->second;
        if (!domain.UsesGpu())
            m_gpuScene->CreateJoint(id, binding.ownerBody, binding.connectedBody, std::move(packed));
    } catch (...) {
        m_jointDomains.erase(id);
        domain.gpuJointAuthors.erase(id);
        throw;
    }
    domain.gpuJointLayoutDirty = true;
    domain.RequestInxPhyGpuAuthorPresentation();
    return id;
}

void PhysicsDomain::RetireGpuJoint(uint64_t id) noexcept
{
    auto retired = gpuJointAuthors.extract(id);
    if (retired) {
        PhysicsWorld::Instance().m_jointDomains.erase(id);
        gpuJointLayoutDirty = true;
        RequestInxPhyGpuAuthorPresentation();
        retired.mapped().owner->OnPhysicsConstraintDestroyed();
    }
}

void PhysicsDomain::MarkGpuJointEdit(uint64_t id, PhysicsSceneJointEdit edit)
{
    auto &mask = gpuJointAuthors.at(id).pendingEdits;
    if (!mask)
        gpuJointEdits.push_back(id);
    mask |= static_cast<uint8_t>(edit);
    RequestInxPhyGpuAuthorPresentation();
}

void PhysicsDomain::SetGpuJointPose(uint64_t id, std::optional<glm::quat> relative)
{
    auto &previous = gpuJointAuthors.at(id).poseTarget;
    if (previous == relative || (previous && relative && *previous == -*relative))
        return;
    MarkGpuJointEdit(id, PhysicsSceneJointEdit::Pose);
    previous = std::move(relative);
}

void PhysicsDomain::SetGpuJointCollision(PhysicsSceneJointAuthor &joint, bool enabled) noexcept
{
    if (joint.enableCollision != enabled) {
        gpuJointLayoutDirty = true;
        joint.enableCollision = enabled;
        RequestInxPhyGpuAuthorPresentation();
    }
}

std::vector<InxPhyGpuSceneJointUpdate> PhysicsDomain::CaptureGpuJointEdits() const
{
    std::vector<InxPhyGpuSceneJointUpdate> edits;
    edits.reserve(gpuJointEdits.size());
    for (const auto id : gpuJointEdits) {
        const auto found = gpuJointAuthors.find(id);
        if (found == gpuJointAuthors.end())
            continue; // A destroyed identity drops any unsubmitted parameter writes.
        const auto &joint = found->second;
        const auto mask = joint.pendingEdits;
        const auto has = [mask](PhysicsSceneJointEdit edit) { return (mask & static_cast<uint8_t>(edit)) != 0; };
        const auto append = [&](const auto &value) { edits.push_back({owner.id, id, value}); };
        std::visit(
            [&](const auto &value) {
                using Setting = std::decay_t<decltype(value)>;
                const auto limits = has(PhysicsSceneJointEdit::Limits);
                const auto drive = has(PhysicsSceneJointEdit::Drive);
                if constexpr (std::is_same_v<Setting, InxPhyGpuSceneHinge>) {
                    if (limits)
                        append(InxPhyGpuJointLimits{value.useLimits, value.minimum, value.maximum});
                    if (drive)
                        append(value.drive);
                } else if constexpr (std::is_same_v<Setting, InxPhyGpuSceneSlider>) {
                    if (limits)
                        append(InxPhyGpuJointLimits{value.useLimits, value.minimum, value.maximum});
                } else if constexpr (std::is_same_v<Setting, InxPhyGpuSceneDistance>) {
                    if (limits && !value.autoDistance)
                        append(InxPhyGpuJointLimits{true, value.minimum, value.maximum});
                    if (drive)
                        append(InxPhyGpuDistanceSpring{value.stiffness, value.damping});
                } else if constexpr (std::is_same_v<Setting, SwingTwistJointSettings>) {
                    if (drive)
                        append(InxPhyGpuSwingTwistParameters(value));
                } else if constexpr (std::is_same_v<Setting, SixDOFJointSettings>) {
                    if (drive)
                        append(PackSixDOFControls(value));
                }
            },
            joint.settings);
        if (has(PhysicsSceneJointEdit::Pose))
            append(InxPhyGpuJointPoseTarget{joint.poseTarget});
    }
    return edits;
}

void PhysicsDomain::AcceptGpuJointEdits() noexcept
{
    for (const auto id : gpuJointEdits)
        if (const auto found = gpuJointAuthors.find(id); found != gpuJointAuthors.end())
            found->second.pendingEdits = 0;
    gpuJointEdits.clear();
    gpuJointLayoutDirty = false;
}

uint64_t PhysicsWorld::CreateHingeConstraint(HingeJoint &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                             const glm::vec3 &localBakedAnchor, const glm::vec3 &localAnchor)
{
    auto &domain = DomainForBody(bodyIdA);
    RequireOwnerThread();
    const Cpu::Hinge settings{owner.m_axis,
                              owner.m_useLimits,
                              glm::radians(owner.m_minimumAngle),
                              glm::radians(owner.m_maximumAngle),
                              owner.m_motorEnabled,
                              owner.GetClampedMotorTargetRadians(),
                              owner.m_motorStiffness,
                              owner.m_motorDamping,
                              owner.m_motorMaxTorque,
                              owner.m_jointDamping,
                              owner.m_frictionTorque,
                              owner.m_armature};
    if (PrepareJointEndpoints(domain, bodyIdA, bodyIdB) == JointBackend::Cpu)
        return StageCpuJoint(domain, m_jointDomains, owner, {bodyIdA, bodyIdB, localBakedAnchor, {}, localAnchor, {}},
                             settings, owner.GetEnableCollision());
    JointAuthor author{};
    author.SetKind(JPH::GPU::JointHinge);
    author.bodies.w = JPH::GPU::JointEnabled | (owner.GetEnableCollision() ? JPH::GPU::JointCollisionEnabled : 0) |
                      (settings.useLimits ? JPH::GPU::JointLimitsEnabled : 0) |
                      (settings.motorEnabled ? JPH::GPU::JointMotorEnabled : 0);
    author.ownerAnchor() = JointVector(localBakedAnchor);
    author.ownerAxis() = JointVector(settings.axis);
    author.ownerLocalAnchor() = JointVector(localAnchor);
    author.limits() = {settings.minimum, settings.maximum, 0.0f, 0.0f};
    author.motor() = {settings.targetAngle, settings.stiffness, settings.damping, settings.maxTorque};
    author.passive() = {settings.jointDamping, settings.frictionTorque, settings.armature, 0.0f};
    return StageGpuJoint(
        domain, owner,
        {bodyIdA, bodyIdB, localAnchor, owner.GetEnableCollision(),
         InxPhyGpuSceneHinge{settings.axis,
                             settings.useLimits,
                             settings.minimum,
                             settings.maximum,
                             {settings.motorEnabled, settings.targetAngle, settings.stiffness, settings.damping,
                              settings.maxTorque, settings.jointDamping, settings.frictionTorque, settings.armature}}},
        std::move(author));
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
    if (PrepareJointEndpoints(domain, bodyIdA, bodyIdB) == JointBackend::Cpu)
        return StageCpuJoint(domain, m_jointDomains, owner, {bodyIdA, bodyIdB, localBakedAnchor, {}, localAnchor, {}},
                             Cpu::Slider{ownerAxis, useLimits, minimumDistance, maximumDistance}, enableCollision);
    JointAuthor author{};
    author.SetKind(JPH::GPU::JointSlider);
    author.bodies.w = JPH::GPU::JointEnabled | (enableCollision ? JPH::GPU::JointCollisionEnabled : 0) |
                      (useLimits ? JPH::GPU::JointLimitsEnabled : 0);
    author.ownerAnchor() = JointVector(localBakedAnchor);
    author.ownerAxis() = JointVector(glm::normalize(ownerAxis));
    author.ownerLocalAnchor() = JointVector(localAnchor);
    author.limits() = {minimumDistance, maximumDistance, 0.0f, 0.0f};
    return StageGpuJoint(domain, owner,
                         {bodyIdA, bodyIdB, localAnchor, enableCollision,
                          InxPhyGpuSceneSlider{ownerAxis, useLimits, minimumDistance, maximumDistance}},
                         std::move(author));
}

uint64_t PhysicsWorld::CreateFixedConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                             const glm::vec3 &localBakedAnchor, const glm::vec3 &localAnchor,
                                             bool enableCollision)
{
    auto &domain = DomainForBody(bodyIdA);
    RequireOwnerThread();
    const auto route = PrepareJointEndpoints(domain, bodyIdA, bodyIdB);
    if (route == JointBackend::Cpu)
        return StageCpuJoint(domain, m_jointDomains, owner, {bodyIdA, bodyIdB, localBakedAnchor, {}, localAnchor, {}},
                             Cpu::Fixed{}, enableCollision);
    JointAuthor author{};
    author.SetKind(JPH::GPU::JointFixed);
    author.bodies.w = JPH::GPU::JointEnabled | (enableCollision ? JPH::GPU::JointCollisionEnabled : 0);
    author.ownerAnchor() = JointVector(localBakedAnchor);
    author.ownerLocalAnchor() = JointVector(localAnchor);
    return StageGpuJoint(domain, owner, {bodyIdA, bodyIdB, localAnchor, enableCollision, {}}, std::move(author));
}

void PhysicsWorld::SetConstraintCollisionEnabled(uint64_t constraintId, bool enabled)
{
    RequireOwnerThread();
    if (auto *cpu = FindCpuJointDomain(constraintId)) {
        const auto &joint = *cpu->cpuControl->FindJoint(constraintId);
        cpu->cpuControl->UpdateJoints({{constraintId, joint.settings, enabled}});
        return;
    }
    auto &domain = RequireJointDomain(constraintId);
    if (!domain.UsesGpu()) {
        RequireGpu(domain);
        m_gpuScene->UpdateJoint(
            constraintId, [=](JointAuthor &author) { SetJointFlag(author, JPH::GPU::JointCollisionEnabled, enabled); });
    }
    domain.SetGpuJointCollision(domain.gpuJointAuthors.at(constraintId), enabled);
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
    if (PrepareJointEndpoints(domain, bodyIdA, bodyIdB) == JointBackend::Cpu)
        return StageCpuJoint(
            domain, m_jointDomains, owner,
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
    return StageGpuJoint(domain, owner,
                         {bodyIdA, bodyIdB, localAnchor, enableCollision,
                          InxPhyGpuSceneDistance{connectedBakedAnchor, autoDistance, minimum, maximum, stiffness,
                                                 damping, connectedLocalAnchor}},
                         std::move(author));
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
    auto &domain = RequireJointDomain(constraintId);
    auto &binding = domain.gpuJointAuthors.at(constraintId);
    auto &settings = RequireSceneJointSettings<InxPhyGpuSceneDistance>(binding);
    if (settings.autoDistance == updateLimits)
        throw std::logic_error("Changing Distance auto mode requires rebuilding its captured reference");
    if (!domain.UsesGpu()) {
        RequireGpu(domain);
        m_gpuScene->UpdateJoint(constraintId, [=](JointAuthor &author) {
            SetJointFlag(author, JPH::GPU::JointCollisionEnabled, enableCollision);
            author.limits() = {minimum, maximum, 0.0f, 0.0f};
            author.linearSpring() = {stiffness, damping, 0.0f, 0.0f};
        });
    }
    domain.SetGpuJointCollision(binding, enableCollision);
    settings.minimum = minimum;
    settings.maximum = maximum;
    settings.stiffness = stiffness;
    settings.damping = damping;
    domain.MarkGpuJointEdit(constraintId, PhysicsSceneJointEdit::Limits);
    domain.MarkGpuJointEdit(constraintId, PhysicsSceneJointEdit::Drive);
}

float PhysicsWorld::GetDistanceConstraintLength(uint64_t constraintId) const
{
    RequireOwnerThread();
    if (auto *cpu = FindCpuJointDomain(constraintId)) {
        RunCpuControl(*cpu);
        return cpu->cpuScene->Constraints().ReadDistance(constraintId);
    }
    auto &domain = RequireJointDomain(constraintId);
    float motion[2];
    ReadJointMotion(domain, {constraintId}, motion, InxPhyGpuSceneJointKind::Distance);
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
    auto &domain = RequireJointDomain(constraintId);
    auto &binding = domain.gpuJointAuthors.at(constraintId);
    auto &settings = RequireSceneJointSettings<InxPhyGpuSceneHinge>(binding);
    if (!domain.UsesGpu()) {
        RequireGpu(domain);
        m_gpuScene->UpdateJoint(constraintId, [=](JointAuthor &author) {
            SetJointFlag(author, JPH::GPU::JointLimitsEnabled, useLimits);
            SetJointFlag(author, JPH::GPU::JointCollisionEnabled, enableCollision);
            author.limits() = {minimum, maximum, 0.0f, 0.0f};
        });
    }
    domain.SetGpuJointCollision(binding, enableCollision);
    settings.useLimits = useLimits;
    settings.minimum = minimum;
    settings.maximum = maximum;
    domain.MarkGpuJointEdit(constraintId, PhysicsSceneJointEdit::Limits);
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
    auto &domain = RequireJointDomain(constraintId);
    auto &settings = RequireSceneJointSettings<InxPhyGpuSceneHinge>(domain.gpuJointAuthors.at(constraintId));
    const auto target = ClampHingeTarget(settings, targetAngle);
    if (!domain.UsesGpu()) {
        RequireGpu(domain);
        m_gpuScene->UpdateJoint(constraintId, [=](JointAuthor &author) {
            SetJointFlag(author, JPH::GPU::JointMotorEnabled, enabled);
            author.motor() = {target, stiffness, damping, maxTorque};
            author.passive() = {jointDamping, frictionTorque, armature, 0.0f};
        });
    }
    settings.drive = {enabled, target, stiffness, damping, maxTorque, jointDamping, frictionTorque, armature};
    domain.MarkGpuJointEdit(constraintId, PhysicsSceneJointEdit::Drive);
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
    auto &domain = RequireJointDomain(constraintId);
    auto &binding = domain.gpuJointAuthors.at(constraintId);
    auto &settings = RequireSceneJointSettings<InxPhyGpuSceneHinge>(binding);
    const auto target = ClampHingeTarget(settings, targetAngle);
    const auto reset = RagdollTargetReset(binding.ownerBody);
    if (!domain.UsesGpu()) {
        RequireGpu(domain);
        m_gpuScene->UpdateJoint(constraintId, [=](JointAuthor &author) { author.motor().x = target; }, reset);
    }
    settings.drive.target = target;
    domain.MarkGpuJointEdit(constraintId, PhysicsSceneJointEdit::Drive);
    if (reset)
        domain.SetGpuJointPose(constraintId, std::nullopt);
}

void PhysicsWorld::SetHingeMotorTargets(const std::vector<HingeJoint *> &joints, const float *targets)
{
    RequireOwnerThread();
    if (!targets && !joints.empty())
        throw std::invalid_argument("Hinge targets require input storage");
    struct Target
    {
        HingeJoint *joint;
        float degrees, radians;
        PhysicsDomain *gpuDomain;
        bool resetPose;
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
        uint32_t reset = 0;
        if (cpu) {
            if (cpu != &domain)
                throw std::logic_error("CPU hinge no longer belongs to its author RuntimeWorld");
            const auto &authored = *domain.cpuControl->FindJoint(id);
            auto next = RequireCpuSettings<Cpu::Hinge>(authored);
            next.targetAngle = target;
            group.cpu.push_back({id, next, authored.enableCollision, true});
        } else {
            if (&RequireJointDomain(id) != &domain)
                throw std::logic_error("GPU hinge no longer belongs to its author RuntimeWorld");
            auto &binding = domain.gpuJointAuthors.at(id);
            (void)RequireSceneJointSettings<InxPhyGpuSceneHinge>(binding);
            reset = RagdollTargetReset(binding.ownerBody);
            if (!domain.UsesGpu()) {
                RequireGpu(domain);
                auto author = m_gpuScene->GetAuthoredJoint(id).author;
                author.motor().x = target;
                group.gpu.push_back({id, author, reset});
            }
        }
        prepared.push_back({&joint, degrees, target, cpu ? nullptr : &domain, reset != 0});
    }
    for (auto &group : groups) {
        if (!group.cpu.empty())
            group.domain->cpuControl->UpdateJoints(std::move(group.cpu));
        if (!group.gpu.empty())
            m_gpuScene->UpdateJoints(std::move(group.gpu));
    }
    for (const auto &target : prepared) {
        if (auto *domain = target.gpuDomain) {
            const auto id = target.joint->GetPhysicsConstraintId();
            auto &binding = domain->gpuJointAuthors.at(id);
            std::get<InxPhyGpuSceneHinge>(binding.settings).drive.target = target.radians;
            domain->MarkGpuJointEdit(id, PhysicsSceneJointEdit::Drive);
            if (target.resetPose)
                domain->SetGpuJointPose(id, std::nullopt);
        }
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
            if (&RequireJointDomain(id) != &domain)
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
            ReadJointMotion(*group.domain, group.gpuIds, values.data(), InxPhyGpuSceneJointKind::Hinge);
            for (size_t local = 0; local < group.gpuRows.size(); ++local) {
                aggregate[2 * group.gpuRows[local]] = values[2 * local];
                aggregate[2 * group.gpuRows[local] + 1] = values[2 * local + 1];
            }
        }
    }
    std::copy(aggregate.begin(), aggregate.end(), output);
}

void PhysicsWorld::ReadJointMotion(PhysicsDomain &domain, const std::vector<uint64_t> &ids, float *output,
                                   InxPhyGpuSceneJointKind kind) const
{
    RequireOwnerThread();
    if (!output && !ids.empty())
        throw std::invalid_argument("Joint motion requires output storage");
    if (ids.empty())
        return;
    PhysicsGpuJointMotionRead prepared;
    if (domain.UsesGpu()) {
        const auto &publication = ReadInxPhyGpuQueryPublication(domain);
        prepared = domain.nativeRuntime->PrepareJointMotion(publication, 0, ids, true, kind);
        RunPreparedWork(domain, prepared.packet);
    } else {
        const auto legacyKind = static_cast<uint32_t>(kind);
        static_assert(static_cast<uint32_t>(InxPhyGpuSceneJointKind::Hinge) == JPH::GPU::JointHinge &&
                      static_cast<uint32_t>(InxPhyGpuSceneJointKind::Slider) == JPH::GPU::JointSlider &&
                      static_cast<uint32_t>(InxPhyGpuSceneJointKind::Distance) == JPH::GPU::JointDistance &&
                      static_cast<uint32_t>(InxPhyGpuSceneJointKind::SwingTwist) == JPH::GPU::JointSwingTwist &&
                      static_cast<uint32_t>(InxPhyGpuSceneJointKind::SixDOF) == JPH::GPU::JointSixDOF);
        std::exception_ptr requestError;
        RunGpuPublicationRead(domain, [&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
            return PrepareJoltGpuQueryWork(
                [&]() -> JoltGpuControlWork {
                    std::unordered_map<uint64_t, const JoltGpuScene::PublishedJoint *> identities;
                    identities.reserve(publication->GetJoints().size());
                    for (const auto &joint : publication->GetJoints())
                        identities.emplace(joint.id, &joint);
                    std::vector<uint32_t> rows;
                    rows.reserve(ids.size());
                    for (const auto id : ids) {
                        const auto found = identities.find(id);
                        if (found == identities.end() || found->second->kind != legacyKind)
                            throw JoltGpuQueryRequestError(
                                "Joint motion requires an accepted identity of the requested kind");
                        rows.push_back(found->second->row);
                    }
                    prepared = m_gpuJoints->PrepareReadMotion(*publication, rows, legacyKind);
                    return {prepared.packet};
                },
                requestError);
        });
        if (requestError)
            std::rethrow_exception(requestError);
    }
    if (!prepared.packet)
        throw std::runtime_error("GPU joint motion did not produce a result");
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    const auto motion = prepared.ReadHostMotion();
    const bool sixDOF = kind == InxPhyGpuSceneJointKind::SixDOF;
    const bool orientation = sixDOF || kind == InxPhyGpuSceneJointKind::SwingTwist;
    for (const auto &row : motion) {
        if (sixDOF || !orientation) {
            const auto count = sixDOF ? 3 : 2;
            output = std::copy_n(row.coordinate.begin(), count, output);
        }
        if (orientation) {
            output = std::copy(row.orientation.begin() + 1, row.orientation.end(), output);
            *output++ = row.orientation.front();
        }
    }
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
    auto &domain = RequireJointDomain(constraintId);
    auto &binding = domain.gpuJointAuthors.at(constraintId);
    auto &settings = RequireSceneJointSettings<InxPhyGpuSceneSlider>(binding);
    if (!domain.UsesGpu()) {
        RequireGpu(domain);
        m_gpuScene->UpdateJoint(constraintId, [=](JointAuthor &author) {
            SetJointFlag(author, JPH::GPU::JointLimitsEnabled, useLimits);
            SetJointFlag(author, JPH::GPU::JointCollisionEnabled, enableCollision);
            author.limits() = {minimum, maximum, 0.0f, 0.0f};
        });
    }
    domain.SetGpuJointCollision(binding, enableCollision);
    settings.useLimits = useLimits;
    settings.minimum = minimum;
    settings.maximum = maximum;
    domain.MarkGpuJointEdit(constraintId, PhysicsSceneJointEdit::Limits);
}

void PhysicsWorld::DestroyConstraintsForBody(uint32_t bodyId)
{
    RequireOwnerThread();
    if (FindBodyRegistration(bodyId)) {
        auto &domain = DomainForBody(bodyId);
        if (domain.cpuControl)
            RetireCpuJointRoutes(domain.cpuControl->DestroyJointsForBody(bodyId));
        if (domain.UsesGpu()) {
            for (auto it = domain.gpuJointAuthors.begin(); it != domain.gpuJointAuthors.end();) {
                const auto current = it++;
                if (current->second.ownerBody == bodyId || current->second.connectedBody == bodyId)
                    domain.RetireGpuJoint(current->first);
            }
            if (!domain.gpuRegistered)
                return;
        }
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
        m_jointDomains.erase(constraintId);
        return;
    }
    if (const auto route = m_jointDomains.find(constraintId);
        route != m_jointDomains.end() && route->second->UsesGpu()) {
        auto &domain = *route->second;
        domain.RetireGpuJoint(constraintId);
        if (!domain.gpuRegistered)
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
    auto &domain = RequireJointDomain(constraintId);
    float motion[2];
    ReadJointMotion(domain, {constraintId}, motion, InxPhyGpuSceneJointKind::Hinge);
    return {motion[0], motion[1]};
}

float PhysicsWorld::GetSliderConstraintPosition(uint64_t constraintId) const
{
    RequireOwnerThread();
    if (auto *cpu = FindCpuJointDomain(constraintId)) {
        RunCpuControl(*cpu);
        return cpu->cpuScene->Constraints().ReadSliderPosition(constraintId);
    }
    auto &domain = RequireJointDomain(constraintId);
    float motion[2];
    ReadJointMotion(domain, {constraintId}, motion, InxPhyGpuSceneJointKind::Slider);
    return motion[0];
}

uint64_t PhysicsWorld::CreateSwingTwistConstraint(PhysicsConstraintOwner &owner, uint32_t bodyIdA, uint32_t bodyIdB,
                                                  const glm::vec3 &localBakedAnchor, const glm::vec3 &localAnchor,
                                                  const SwingTwistJointSettings &settings, bool enableCollision)
{
    auto &domain = DomainForBody(bodyIdA);
    RequireOwnerThread();
    if (PrepareJointEndpoints(domain, bodyIdA, bodyIdB) == JointBackend::Cpu)
        return StageCpuJoint(domain, m_jointDomains, owner, {bodyIdA, bodyIdB, localBakedAnchor, {}, localAnchor, {}},
                             settings, enableCollision);
    JointAuthor author{};
    author.SetKind(JPH::GPU::JointSwingTwist);
    author.bodies.w = JPH::GPU::JointEnabled | (enableCollision ? JPH::GPU::JointCollisionEnabled : 0);
    author.ownerAnchor() = JointVector(localBakedAnchor);
    author.ownerLocalAnchor() = JointVector(localAnchor);
    ApplySwingSettings(author, settings, ValidateSwingSettings(settings));
    return StageGpuJoint(domain, owner, {bodyIdA, bodyIdB, localAnchor, enableCollision, settings}, std::move(author));
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
    auto &domain = RequireJointDomain(constraintId);
    auto &binding = domain.gpuJointAuthors.at(constraintId);
    const auto &previous = RequireSceneJointSettings<SwingTwistJointSettings>(binding);
    const auto frame = ValidateSwingSettings(settings);
    if (frame != ValidateSwingSettings(previous))
        throw std::logic_error("SwingTwist frame changes require rebuilding the constraint");
    const auto reset = claimTarget ? RagdollTargetReset(binding.ownerBody) : 0u;
    if (!domain.UsesGpu()) {
        RequireGpu(domain);
        m_gpuScene->UpdateJoint(
            constraintId,
            [=](JointAuthor &author) {
                ApplySwingSettings(author, settings, frame);
                SetJointFlag(author, JPH::GPU::JointCollisionEnabled, enableCollision);
            },
            reset);
    }
    domain.SetGpuJointCollision(binding, enableCollision);
    binding.settings = settings;
    domain.MarkGpuJointEdit(constraintId, PhysicsSceneJointEdit::Drive);
    if (reset)
        domain.SetGpuJointPose(constraintId, std::nullopt);
}

glm::quat PhysicsWorld::GetAngularConstraintOrientation(uint64_t id) const
{
    RequireOwnerThread();
    if (auto *cpu = FindCpuJointDomain(id)) {
        RunCpuControl(*cpu);
        return cpu->cpuScene->Constraints().ReadOrientation(id);
    }
    auto &domain = RequireJointDomain(id);
    const auto &settings = domain.gpuJointAuthors.at(id).settings;
    if (std::holds_alternative<SixDOFJointSettings>(settings)) {
        float values[7];
        ReadJointMotion(domain, {id}, values, InxPhyGpuSceneJointKind::SixDOF);
        return {values[6], values[3], values[4], values[5]};
    }
    if (!std::holds_alternative<SwingTwistJointSettings>(settings))
        throw std::logic_error("Constraint has no angular orientation output");
    float values[4];
    ReadJointMotion(domain, {id}, values, InxPhyGpuSceneJointKind::SwingTwist);
    return {values[3], values[0], values[1], values[2]};
}

uint64_t PhysicsWorld::CreateSixDOFConstraint(PhysicsConstraintOwner &owner, uint32_t ownerBody, uint32_t connectedBody,
                                              const glm::vec3 &baked, const glm::vec3 &raw,
                                              const SixDOFJointSettings &settings, bool collision)
{
    RequireOwnerThread();
    auto &domain = DomainForBody(ownerBody);
    if (PrepareJointEndpoints(domain, ownerBody, connectedBody) == JointBackend::Cpu)
        return StageCpuJoint(domain, m_jointDomains, owner, {ownerBody, connectedBody, baked, {}, raw, {}}, settings,
                             collision);
    JointAuthor author(JPH::GPU::JointSixDOF);
    author.bodies.w = JPH::GPU::JointEnabled | (collision ? JPH::GPU::JointCollisionEnabled : 0u);
    author.ownerAnchor() = JointVector(baked);
    author.ownerLocalAnchor() = JointVector(raw);
    ApplySixSettings(author, settings, ValidateSixSettings(settings));
    return StageGpuJoint(domain, owner, {ownerBody, connectedBody, raw, collision, settings}, std::move(author));
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
    auto &domain = RequireJointDomain(id);
    auto &binding = domain.gpuJointAuthors.at(id);
    const auto &previous = RequireSceneJointSettings<SixDOFJointSettings>(binding);
    const auto frame = ValidateSixSettings(settings);
    if (frame.orientation != ValidateSixSettings(previous).orientation)
        throw std::logic_error("SixDOF frame changes require rebuilding the constraint");
    const auto reset = claimTarget ? RagdollTargetReset(binding.ownerBody) : 0u;
    if (!domain.UsesGpu()) {
        RequireGpu(domain);
        m_gpuScene->UpdateJoint(
            id,
            [=](JointAuthor &author) {
                ApplySixSettings(author, settings, frame);
                SetJointFlag(author, JPH::GPU::JointCollisionEnabled, collision);
            },
            reset);
    }
    domain.SetGpuJointCollision(binding, collision);
    binding.settings = settings;
    domain.MarkGpuJointEdit(id, PhysicsSceneJointEdit::Drive);
    if (reset)
        domain.SetGpuJointPose(id, std::nullopt);
}
glm::vec3 PhysicsWorld::GetSixDOFConstraintPosition(uint64_t id) const
{
    RequireOwnerThread();
    if (auto *cpu = FindCpuJointDomain(id)) {
        RunCpuControl(*cpu);
        return cpu->cpuScene->Constraints().ReadSixDOFPosition(id);
    }
    float values[7];
    ReadJointMotion(RequireJointDomain(id), {id}, values, InxPhyGpuSceneJointKind::SixDOF);
    return {values[0], values[1], values[2]};
}

} // namespace infernux
