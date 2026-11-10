#include "GameObject.h"
#include "HingeJoint.h"
#include "Ragdoll.h"
#include "Rigidbody.h"
#include "SixDOFJoint.h"
#include "SkeletonInstance.h"
#include "SwingTwistJoint.h"
#include "Transform.h"
#include "physics/PhysicsWorld.h"
#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace infernux
{
bool Ragdoll::GetActive() const
{
    const auto bodies = GetConfiguredBodies();
    return std::any_of(bodies.begin(), bodies.end(), [](auto *body) { return !body->IsKinematic(); });
}

void Ragdoll::SetActive(bool active)
{
    auto &world = PhysicsWorld::Instance();
    world.RequireOwnerThread();
    const auto bodies = GetConfiguredBodies();
    if (bodies.empty())
        throw std::logic_error("Configure ragdoll bodies before changing their mode");
    std::vector<Rigidbody *> changed;
    for (auto *body : bodies) {
        if (body->IsKinematic() != active)
            continue;
        if (active && body->m_massProperties)
            world.ValidateBodyMassProperties(*body, body->Data().mass, &*body->m_massProperties);
        changed.push_back(body);
    }
    if (changed.empty())
        return;

    auto *skeleton = GetSkeleton();
    const bool handBack = !active && skeleton && skeleton->HasAnimationPose();
    if (handBack)
        skeleton->GetBones(); // Resolve the complete target before changing any member mode.
    auto *anchor = bodies.front()->GetGameObject()->GetTransform();
    const auto anchorPosition = anchor->GetPosition();
    const auto anchorRotation = anchor->GetWorldRotation();
    // Snapshot the displayed poses before any parent changes mode. These may
    // be newer than Jolt on a render frame without a fixed step.
    std::vector<std::pair<glm::vec3, glm::quat>> poses;
    for (auto *body : changed) {
        auto *transform = body->GetGameObject()->GetTransform();
        poses.emplace_back(transform->GetPosition(), transform->GetWorldRotation());
    }
    for (auto *body : changed)
        body->SetIsKinematic(!active);
    if (handBack) {
        skeleton->ApplyAnimationPose();
        skeleton->AlignAnimationRoot(anchor, anchorPosition, anchorRotation);
    }
    const auto &published = handBack ? bodies : changed;
    for (size_t index = 0; index < published.size(); ++index) {
        auto *body = published[index];
        auto *object = body->GetGameObject();
        auto *transform = object->GetTransform();
        const auto position = handBack ? transform->GetPosition() : poses[index].first;
        const auto rotation = handBack ? transform->GetWorldRotation() : poses[index].second;
        if (body->GetParticipatingBodyId() != 0xFFFFFFFF)
            body->TeleportBodies(world, object, position, rotation);
    }
    // No episode reset or extra clearing of velocities / joint history.
    // Jolt's normal dynamic -> kinematic transition cancels accumulated forces.
}

void Ragdoll::SetDriveAnimation(bool enabled)
{
    PhysicsWorld::Instance().RequireOwnerThread();
    if (enabled == m_driveAnimation)
        return;
    if (enabled) {
        auto *skeleton = GetSkeleton();
        if (!skeleton || !skeleton->HasBoundHierarchy())
            throw std::logic_error("Animation drive requires a bound SkeletonInstance");
        if (!skeleton->HasAnimationPose())
            skeleton->CaptureAnimationPose();
        for (auto *body : GetConfiguredBodies()) {
            auto *object = body->GetGameObject();
            if (auto *hinge = object->GetComponent<HingeJoint>())
                hinge->SetMotorEnabled(true);
            else if (auto *swing = object->GetComponent<SwingTwistJoint>()) {
                swing->SetSwingMotorMode(JointMotorMode::Position);
                swing->SetTwistMotorMode(JointMotorMode::Position);
            }
        }
    }
    m_driveAnimation = enabled;
    if (enabled && IsBound())
        UpdateAnimationDrive();
}

void Ragdoll::SetPassive()
{
    PhysicsWorld::Instance().RequireOwnerThread();
    m_driveAnimation = false;
    for (auto *body : GetConfiguredBodies()) {
        auto *object = body->GetGameObject();
        if (auto *hinge = object->GetComponent<HingeJoint>())
            hinge->SetMotorEnabled(false);
        else if (auto *swing = object->GetComponent<SwingTwistJoint>()) {
            swing->SetSwingMotorMode(JointMotorMode::Off);
            swing->SetTwistMotorMode(JointMotorMode::Off);
        } else if (auto *six = object->GetComponent<SixDOFJoint>()) {
            six->SetXMotorMode(JointMotorMode::Off);
            six->SetYMotorMode(JointMotorMode::Off);
            six->SetZMotorMode(JointMotorMode::Off);
            six->SetAngularXMotorMode(JointMotorMode::Off);
            six->SetAngularYMotorMode(JointMotorMode::Off);
            six->SetAngularZMotorMode(JointMotorMode::Off);
        }
    }
}

void Ragdoll::UpdateAnimationDrive()
{
    auto *skeleton = GetSkeleton();
    if (!IsBound() || !skeleton || !skeleton->HasAnimationPose())
        return;
    const auto bones = skeleton->GetBones();
    const auto rotations = skeleton->GetAnimationWorldRotations();
    std::unordered_map<Transform *, size_t> indices;
    for (size_t index = 0; index < bones.size(); ++index)
        indices.emplace(bones[index], index);
    std::vector<glm::quat> targets;
    targets.reserve(m_bodies.size());
    for (const auto &reference : m_bodies) {
        auto *transform = reference.GetComponent()->GetGameObject()->GetTransform();
        targets.push_back(rotations.at(indices.at(transform)));
    }
    PhysicsWorld::Instance().DriveRagdollPose(m_assemblyId, targets, false);
}
} // namespace infernux
