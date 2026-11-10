#include "HingeJointBatch.h"

#include "../HingeJoint.h"
#include "../Scene.h"
#include "../SceneManager.h"
#include "PhysicsWorld.h"
#include <cmath>
#include <glm/glm.hpp>
#include <stdexcept>
#include <unordered_set>

namespace infernux
{
HingeJointBatch::HingeJointBatch(const std::vector<HingeJoint *> &joints)
{
    PhysicsWorld::Instance().RequireOwnerThread();
    const auto &manager = SceneManager::Instance();
    if (!manager.IsPlaying() || joints.empty())
        throw std::invalid_argument("hinge batches require a nonempty set of bound joints in Play");
    m_epoch = manager.GetSimulationEpoch();
    std::unordered_set<uint64_t> identities;
    m_joints.reserve(joints.size());
    for (const auto *joint : joints) {
        if (!joint)
            throw std::invalid_argument("hinge batches cannot contain null joints");
        const auto handle = joint->GetHandle();
        if (!handle.IsValid() || joint->GetPhysicsConstraintId() == 0)
            throw std::invalid_argument("hinge batch joints must have live components and active constraints");
        if (m_worldId == 0)
            m_worldId = handle.worldId;
        if (handle.worldId != m_worldId || !identities.insert(handle.id).second)
            throw std::invalid_argument("hinge batches require unique ordered joints in one scene");
        m_joints.push_back({handle, joint->GetPhysicsConstraintId()});
    }
}

std::vector<HingeJoint *> HingeJointBatch::Resolve() const
{
    PhysicsWorld::Instance().RequireOwnerThread();
    auto &manager = SceneManager::Instance();
    if (!manager.IsPlaying() || manager.GetSimulationEpoch() != m_epoch)
        throw std::logic_error("hinge batch Play session has retired");
    auto *scene = manager.GetSceneByWorldId(m_worldId);
    if (!scene)
        throw std::logic_error("hinge batch scene has retired");
    std::vector<HingeJoint *> resolved;
    resolved.reserve(m_joints.size());
    for (const auto &entry : m_joints) {
        auto *component = scene->ResolveComponent(entry.handle);
        if (!component)
            throw std::logic_error("hinge batch component has retired");
        // A generation-exact handle resolves only the original typed component.
        auto *joint = static_cast<HingeJoint *>(component);
        if (joint->GetPhysicsConstraintId() != entry.constraintId)
            throw std::logic_error("hinge batch constraint was disabled or rebound; create a new binding");
        resolved.push_back(joint);
    }
    return resolved;
}

void HingeJointBatch::Validate() const
{
    (void)Resolve();
}

uint64_t HingeJointBatch::ReadMotion(float *output, size_t count) const
{
    if (count != 2 * m_joints.size())
        throw std::invalid_argument("hinge motion output requires two floats per joint");
    const auto joints = Resolve();
    for (size_t row = 0; row < joints.size(); ++row) {
        const auto state = joints[row]->GetMotionState();
        output[2 * row] = state.x;
        output[2 * row + 1] = state.y;
    }
    return SceneManager::Instance().GetCompletedFixedStepIndex();
}

void HingeJointBatch::SetMotorTargets(const float *targets, size_t count, std::optional<uint64_t> expectedStep) const
{
    if (count != m_joints.size())
        throw std::invalid_argument("hinge motor targets require one float per joint");
    const auto joints = Resolve();
    if (expectedStep && *expectedStep != SceneManager::Instance().GetCompletedFixedStepIndex())
        throw std::logic_error("hinge action belongs to a different fixed step");
    for (size_t row = 0; row < count; ++row)
        if (!std::isfinite(targets[row]) || !std::isfinite(glm::degrees(targets[row])))
            throw std::invalid_argument("hinge motor targets must be finite representable radians");
    for (size_t row = 0; row < joints.size(); ++row)
        joints[row]->SetMotorTargetAngle(glm::degrees(targets[row]));
}
} // namespace infernux
