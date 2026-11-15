#include "GameObject.h"
#include "HingeJoint.h"
#include "Ragdoll.h"
#include "Rigidbody.h"
#include "SixDOFJoint.h"
#include "SkeletonInstance.h"
#include "SkeletonPhysicsPoseRead.h"
#include "SwingTwistJoint.h"
#include "Transform.h"
#include "physics/PhysicsWorld.h"
#include <algorithm>
#include <stdexcept>
#include <unordered_set>

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
    std::vector<Transform *> dependencies{anchor};
    for (auto *body : changed)
        dependencies.push_back(body->GetGameObject()->GetTransform());
    glm::vec3 anchorPosition;
    glm::quat anchorRotation;
    // Snapshot the displayed poses before any parent changes mode. These may
    // be newer than Jolt on a render frame without a fixed step.
    std::vector<std::pair<glm::vec3, glm::quat>> poses;
    {
        SkeletonPhysicsPoseRead reading(dependencies, SkeletonPhysicsPoseRead::Space::World);
        anchorPosition = anchor->GetPosition();
        anchorRotation = anchor->GetWorldRotation();
        for (auto *body : changed) {
            auto *transform = body->GetGameObject()->GetTransform();
            poses.emplace_back(transform->GetPosition(), transform->GetWorldRotation());
        }
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
    }
    m_driveAnimation = enabled;
    m_animationTargetPending = enabled;
    m_initializedAnimationJoints.clear();
    if (enabled) {
        InitializeAnimationDrive();
        Rebuild();
    }
}

void Ragdoll::InitializeAnimationDrive()
{
    if (!m_driveAnimation || m_initializingAnimationDrive)
        return;
    struct Initializing
    {
        bool &active;
        explicit Initializing(bool &value) : active(value)
        {
            active = true;
        }
        ~Initializing()
        {
            active = false;
        }
    } initializing(m_initializingAnimationDrive);
    auto *skeleton = GetSkeleton();
    if (!skeleton)
        throw std::logic_error("Animation drive requires a SkeletonInstance");
    if (!skeleton->HasBoundHierarchy())
        return;
    if (!skeleton->HasAnimationPose()) {
        // A saved controller with no sampled target starts from authored locals.
        // Settling an author setter must never demand numerical bone poses.
        TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
        skeleton->CaptureAnimationPose();
    }
    std::unordered_set<Component *> initialized;
    for (const auto &reference : m_initializedAnimationJoints)
        if (auto *joint = reference.GetComponent())
            initialized.insert(joint);
    const auto configured = GetConfiguredBodies();
    const std::unordered_set<Rigidbody *> members(configured.begin(), configured.end());
    for (auto *body : configured) {
        auto *joint = body->GetGameObject()->GetComponent<Joint>();
        if (!joint || !members.count(joint->GetConnectedBody()) || joint->IsDestroyed() || joint->IsBeingDestroyed() ||
            !initialized.insert(joint).second)
            continue;
        m_initializedAnimationJoints.emplace_back(joint);
        if (auto *hinge = dynamic_cast<HingeJoint *>(joint))
            hinge->SetMotorEnabled(true);
        else if (auto *swing = dynamic_cast<SwingTwistJoint *>(joint)) {
            swing->SetSwingMotorMode(JointMotorMode::Position);
            swing->SetTwistMotorMode(JointMotorMode::Position);
        } else if (auto *six = dynamic_cast<SixDOFJoint *>(joint)) {
            six->SetAngularXMotorMode(JointMotorMode::Position);
            six->SetAngularYMotorMode(JointMotorMode::Position);
            six->SetAngularZMotorMode(JointMotorMode::Position);
        }
    }
}

void Ragdoll::SetPassive()
{
    PhysicsWorld::Instance().RequireOwnerThread();
    m_driveAnimation = false;
    m_animationTargetPending = false;
    m_initializedAnimationJoints.clear();
    PhysicsWorld::Instance().ClearRagdollDrive(m_assemblyId);
    const auto configured = GetConfiguredBodies();
    const std::unordered_set<Rigidbody *> members(configured.begin(), configured.end());
    for (auto *body : configured) {
        auto *object = body->GetGameObject();
        auto *joint = object->GetComponent<Joint>();
        if (!joint || !members.count(joint->GetConnectedBody()))
            continue;
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

bool Ragdoll::UpdateAnimationDrive(bool deferred)
{
    auto *skeleton = GetSkeleton();
    if (!IsBound() || !skeleton || !skeleton->HasAnimationPose())
        return false;
    // Only the saved animation target is needed. Its common world-root factor
    // cancels in parent-relative D, so no numerical pose demand belongs here.
    TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    const auto rotations = skeleton->GetAnimationWorldRotations();
    std::vector<glm::quat> targets;
    targets.reserve(m_animationPoseInputs.size());
    for (const auto index : m_animationPoseInputs)
        targets.push_back(rotations.at(index));
    PhysicsWorld::Instance().DriveRagdollPoseImpl(m_assemblyId, targets, false, true, deferred);
    return true;
}

} // namespace infernux
