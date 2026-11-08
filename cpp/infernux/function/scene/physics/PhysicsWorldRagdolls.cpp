#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Collision/GroupFilterTable.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/SwingTwistConstraint.h>
#include <Jolt/Physics/Constraints/TwoBodyConstraint.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Ragdoll/Ragdoll.h>

#include "../GameObject.h"
#include "../HingeJoint.h"
#include "../Joint.h"
#include "../Rigidbody.h"
#include "../SwingTwistJoint.h"
#include "PhysicsWorld.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace infernux
{
struct PhysicsWorld::RagdollAssembly
{
    JPH::Ref<JPH::Ragdoll> view;
    JPH::Ref<JPH::GroupFilterTable> collisions;
    std::vector<uint32_t> bodies;
    std::vector<Joint *> joints;
};

uint64_t PhysicsWorld::CreateRagdollAssembly(const std::vector<Rigidbody *> &bodies,
                                             const std::vector<Joint *> &parentJoints, const std::vector<int> &parents)
{
    RequireOwnerThread();
    if (bodies.empty())
        return 0;
    if (parentJoints.size() != bodies.size() || parents.size() != bodies.size())
        throw std::invalid_argument("ragdoll body, parent and joint arrays must match");

    JPH::Array<JPH::BodyID> ids;
    JPH::Array<JPH::Ref<JPH::TwoBodyConstraint>> constraints;
    JPH::Ref<JPH::RagdollSettings> settings = new JPH::RagdollSettings;
    settings->mSkeleton = new JPH::Skeleton;
    settings->mParts.resize(bodies.size());
    auto assembly = std::make_shared<RagdollAssembly>();
    assembly->joints = parentJoints;
    for (size_t index = 0; index < bodies.size(); ++index) {
        auto *body = bodies[index];
        const uint32_t id = body->GetParticipatingBodyId();
        if (id == 0xFFFFFFFF || !m_physicsSystem->GetBodyInterface().IsAdded(JPH::BodyID(id)))
            return 0;
        if (std::find(assembly->bodies.begin(), assembly->bodies.end(), id) != assembly->bodies.end())
            throw std::invalid_argument("a Rigidbody can occur only once in a ragdoll");
        for (const auto &[otherId, other] : m_ragdolls)
            if (std::find(other->bodies.begin(), other->bodies.end(), id) != other->bodies.end())
                throw std::invalid_argument("a Rigidbody cannot belong to two ragdolls");
        {
            JPH::BodyLockRead lock(m_physicsSystem->GetBodyLockInterface(), JPH::BodyID(id));
            static_cast<JPH::BodyCreationSettings &>(settings->mParts[index]) =
                lock.GetBody().GetBodyCreationSettings();
        }
        const int parent = parents[index];
        if (parent < -1 || parent >= static_cast<int>(index))
            throw std::invalid_argument("ragdoll parents must precede their children");
        settings->mSkeleton->AddJoint(body->GetGameObject()->GetName(), parent);
        if (parent >= 0) {
            auto *joint = parentJoints[index];
            if (!joint || joint->GetGameObject() != body->GetGameObject() ||
                joint->GetConnectedBody() != bodies[parent])
                throw std::invalid_argument("ragdoll Joint must connect its actual child and parent Rigidbody");
            auto found = m_constraints.find(joint->GetPhysicsConstraintId());
            if (found == m_constraints.end())
                return 0;
            const auto &record = found->second;
            auto *constraint = static_cast<JPH::TwoBodyConstraint *>(record.constraint);
            settings->mParts[index].mToParent =
                static_cast<JPH::TwoBodyConstraintSettings *>(constraint->GetConstraintSettings().GetPtr());
            constraints.push_back(constraint);
        } else if (parentJoints[index]) {
            throw std::invalid_argument("ragdoll root cannot have a parent Joint");
        }
        ids.push_back(JPH::BodyID(id));
        assembly->bodies.push_back(id);
    }
    settings->CalculateBodyIndexToConstraintIndex();
    settings->CalculateConstraintIndexToBodyIdxPair();
    settings->DisableParentChildCollisions();
    // The helper supplies the defaults. Real Joint.enable_collision remains
    // the editing authority, including after this table has been installed.
    assembly->collisions = new JPH::GroupFilterTable(
        *static_cast<const JPH::GroupFilterTable *>(settings->mParts.front().mCollisionGroup.GetGroupFilter()));
    for (size_t index = 0; index < bodies.size(); ++index)
        if (parents[index] >= 0 && parentJoints[index]->GetEnableCollision())
            assembly->collisions->EnableCollision(static_cast<uint32_t>(index), parents[index]);
    assembly->view = settings->CreateRagdollView(ids, constraints, m_physicsSystem.get());

    const uint64_t id = m_nextRagdollId++;
    auto &interface = m_physicsSystem->GetBodyInterface();
    for (uint32_t index = 0; index < ids.size(); ++index) {
        interface.SetCollisionGroup(ids[index],
                                    JPH::CollisionGroup(assembly->collisions, static_cast<uint32_t>(id), index));
        interface.InvalidateContactCache(ids[index]);
    }
    m_ragdolls.emplace(id, std::move(assembly));
    return id;
}

void PhysicsWorld::DestroyRagdollAssembly(uint64_t id)
{
    auto found = m_ragdolls.find(id);
    if (found == m_ragdolls.end())
        return;
    auto &interface = m_physicsSystem->GetBodyInterface();
    for (uint32_t body : found->second->bodies) {
        interface.SetCollisionGroup(JPH::BodyID(body), {});
        interface.InvalidateContactCache(JPH::BodyID(body));
    }
    // This releases constraint references, never component-owned bodies.
    m_ragdolls.erase(found);
}

void PhysicsWorld::InvalidateRagdollsForBody(uint32_t bodyId)
{
    for (auto it = m_ragdolls.begin(); it != m_ragdolls.end();) {
        const auto current = it++;
        if (std::find(current->second->bodies.begin(), current->second->bodies.end(), bodyId) !=
            current->second->bodies.end())
            DestroyRagdollAssembly(current->first);
    }
}

void PhysicsWorld::InvalidateRagdollsForConstraint(const JPH::Constraint *constraint)
{
    for (auto it = m_ragdolls.begin(); it != m_ragdolls.end();) {
        const auto current = it++;
        const auto &view = current->second->view;
        for (int index = 0; index < static_cast<int>(view->GetConstraintCount()); ++index)
            if (view->GetConstraint(index) == constraint) {
                DestroyRagdollAssembly(current->first);
                break;
            }
    }
}

void PhysicsWorld::UpdateRagdollCollision(const JPH::Constraint *constraint, bool enabled)
{
    for (auto &[id, assembly] : m_ragdolls) {
        const auto &view = assembly->view;
        for (int index = 0; index < static_cast<int>(view->GetConstraintCount()); ++index) {
            if (view->GetConstraint(index) != constraint)
                continue;
            const auto pair = view->GetRagdollSettings()->GetBodyIndicesForConstraintIndex(index);
            if (enabled)
                assembly->collisions->EnableCollision(pair.first, pair.second);
            else
                assembly->collisions->DisableCollision(pair.first, pair.second);
            auto &interface = m_physicsSystem->GetBodyInterface();
            interface.InvalidateContactCache(JPH::BodyID(assembly->bodies[pair.first]));
            interface.InvalidateContactCache(JPH::BodyID(assembly->bodies[pair.second]));
        }
    }
}

void PhysicsWorld::DriveRagdollPose(uint64_t id, const std::vector<glm::quat> &worldRotations, bool enableMotors)
{
    RequireOwnerThread();
    const auto &assembly = *m_ragdolls.at(id);
    auto &view = assembly.view;
    const auto *settings = view->GetRagdollSettings();
    if (worldRotations.size() != view->GetBodyCount())
        throw std::invalid_argument("ragdoll pose must contain one rotation per body");
    std::vector<glm::quat> rotations;
    rotations.reserve(worldRotations.size());
    for (const auto &rotation : worldRotations) {
        const float lengthSquared = glm::dot(rotation, rotation);
        if (!std::isfinite(rotation.x) || !std::isfinite(rotation.y) || !std::isfinite(rotation.z) ||
            !std::isfinite(rotation.w) || !std::isfinite(lengthSquared) || lengthSquared <= 0.0f)
            throw std::invalid_argument("ragdoll rotations must be finite and nonzero");
        rotations.push_back(glm::normalize(rotation));
    }
    for (int index = 0; enableMotors && index < static_cast<int>(view->GetConstraintCount()); ++index) {
        const auto type = view->GetConstraint(index)->GetSubType();
        if (type != JPH::EConstraintSubType::Hinge && type != JPH::EConstraintSubType::SwingTwist)
            throw std::invalid_argument("Jolt ragdoll pose motors require Hinge or SwingTwist joints");
        const auto pair = settings->GetBodyIndicesForConstraintIndex(index);
        if (!dynamic_cast<HingeJoint *>(assembly.joints[pair.second]) &&
            !dynamic_cast<SwingTwistJoint *>(assembly.joints[pair.second]))
            throw std::invalid_argument("ragdoll pose drive requires HingeJoint or SwingTwistJoint components");
    }
    JPH::SkeletonPose pose;
    pose.SetSkeleton(settings->mSkeleton);
    for (int index = 0; index < static_cast<int>(worldRotations.size()); ++index) {
        const int parent = settings->mSkeleton->GetJoint(index).mParentJointIndex;
        const auto relative = parent < 0 ? rotations[index] : glm::conjugate(rotations[parent]) * rotations[index];
        pose.GetJoint(index).mRotation = JPH::Quat(relative.x, relative.y, relative.z, relative.w);
        pose.GetJoint(index).mTranslation = JPH::Vec3::sZero();
    }
    pose.CalculateJointMatrices();
    if (enableMotors)
        view->DriveToPoseUsingMotors(pose);
    // The command changes the actual Joint's motor state, including values
    // used by the inspector, cloning and subsequent constraint reconstruction.
    for (int index = 0; index < static_cast<int>(view->GetConstraintCount()); ++index) {
        const auto pair = settings->GetBodyIndicesForConstraintIndex(index);
        bool changed;
        if (auto *hinge = dynamic_cast<HingeJoint *>(assembly.joints[pair.second])) {
            if (!enableMotors && !hinge->GetMotorEnabled())
                continue;
            auto *motor = static_cast<JPH::HingeConstraint *>(view->GetConstraint(index));
            if (!enableMotors)
                motor->SetTargetOrientationBS(pose.GetJoint(pair.second).mRotation);
            const float angle = glm::degrees(motor->GetTargetAngle());
            changed = !hinge->GetMotorEnabled() || hinge->GetMotorTargetAngle() != angle;
            hinge->RecordPoseDrive(angle);
        } else if (auto *joint = dynamic_cast<SwingTwistJoint *>(assembly.joints[pair.second])) {
            if (!enableMotors && joint->GetSwingMotorMode() != JointMotorMode::Position &&
                joint->GetTwistMotorMode() != JointMotorMode::Position)
                continue;
            auto *motor = static_cast<JPH::SwingTwistConstraint *>(view->GetConstraint(index));
            if (!enableMotors)
                motor->SetTargetOrientationBS(pose.GetJoint(pair.second).mRotation);
            const auto rotation = motor->GetTargetOrientationCS();
            const glm::vec4 target(rotation.GetX(), rotation.GetY(), rotation.GetZ(), rotation.GetW());
            const auto previous = joint->GetTargetOrientation();
            changed = (enableMotors && (joint->GetSwingMotorMode() != JointMotorMode::Position ||
                                        joint->GetTwistMotorMode() != JointMotorMode::Position)) ||
                      (previous != target && previous != -target);
            joint->RecordPoseDrive(target, enableMotors);
        } else
            continue;
        if (changed)
            ActivateConstraintBodies(m_constraints.at(assembly.joints[pair.second]->GetPhysicsConstraintId()));
    }
}
} // namespace infernux
