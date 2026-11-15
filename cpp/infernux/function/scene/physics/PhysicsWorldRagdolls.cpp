#include "../Collider.h"
#include "../GameObject.h"
#include "../HingeJoint.h"
#include "../Joint.h"
#include "../Ragdoll.h"
#include "../Rigidbody.h"
#include "../RuntimeWorld.h"
#include "../Scene.h"
#include "../SixDOFJoint.h"
#include "../SkeletonInstance.h"
#include "../SwingTwistJoint.h"
#include "../Transform.h"
#include "PhysicsDomain.h"
#include "PhysicsWorld.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace infernux
{
namespace
{
std::array<uint64_t, 2> Identity(const Component &component)
{
    return {component.GetComponentID(), component.GetLifetimeGeneration()};
}

PhysicsColliderPair PairIdentity(const Collider &a, const Collider &b)
{
    auto first = Identity(a), second = Identity(b);
    if (second < first)
        std::swap(first, second);
    return {first[0], first[1], second[0], second[1]};
}

bool AngularDriveKind(uint32_t kind)
{
    return kind == JPH::GPU::JointHinge || kind == JPH::GPU::JointSwingTwist || kind == JPH::GPU::JointSixDOF;
}

glm::quat TargetRotation(glm::quat q)
{
    const float scale = std::max({std::abs(q.x), std::abs(q.y), std::abs(q.z), std::abs(q.w)});
    if (!std::isfinite(scale) || !(scale > 0) || !std::isfinite(q.x) || !std::isfinite(q.y) || !std::isfinite(q.z) ||
        !std::isfinite(q.w))
        throw std::invalid_argument("Ragdoll pose rotations must be finite and nonzero");
    q = glm::normalize(q / scale);
    if (q.w < 0)
        return -q;
    if (q.w == 0) {
        const float component[3]{q.x, q.y, q.z};
        size_t dominant = 0;
        for (size_t index = 1; index < 3; ++index)
            if (std::abs(component[index]) > std::abs(component[dominant]))
                dominant = index;
        if (component[dominant] < 0)
            q = -q;
        q.w = 0.0f;
    }
    return q;
}
} // namespace

void PhysicsWorld::DriveRagdollPose(uint64_t id, const std::vector<glm::quat> &worldRotations, bool enableMotors)
{
    DriveRagdollPoseImpl(id, worldRotations, enableMotors, false, false);
}

void PhysicsWorld::QueueRagdollRebuild(Ragdoll &owner) noexcept
{
    if (owner.IsDestroyed() || owner.IsBeingDestroyed() ||
        Component::FindByComponentId(owner.GetComponentID()) != &owner)
        return;
    PhysicsDomain *domain = nullptr;
    const auto old = m_ragdolls.find(owner.m_assemblyId);
    if (old != m_ragdolls.end())
        domain = old->second.domain;
    else if (auto *object = owner.GetGameObject(); object && object->GetScene())
        if (auto *world = object->GetScene()->GetRuntimeWorld())
            domain = world->physics.get();
    if (!domain || domain->retiring)
        return;
    owner.m_membershipDirty = true;
    if (owner.m_pendingDomain)
        return;
    owner.m_pendingDomain = domain;
    owner.m_pendingPrevious = domain->pendingRagdollTail;
    owner.m_pendingNext = nullptr;
    if (domain->pendingRagdollTail)
        domain->pendingRagdollTail->m_pendingNext = &owner;
    else
        domain->pendingRagdollHead = &owner;
    domain->pendingRagdollTail = &owner;
}

void PhysicsWorld::CancelRagdollRebuild(Ragdoll &owner) noexcept
{
    auto *domain = owner.m_pendingDomain;
    if (domain) {
        if (owner.m_pendingPrevious)
            owner.m_pendingPrevious->m_pendingNext = owner.m_pendingNext;
        else
            domain->pendingRagdollHead = owner.m_pendingNext;
        if (owner.m_pendingNext)
            owner.m_pendingNext->m_pendingPrevious = owner.m_pendingPrevious;
        else
            domain->pendingRagdollTail = owner.m_pendingPrevious;
    }
    owner.m_pendingDomain = nullptr;
    owner.m_pendingPrevious = owner.m_pendingNext = nullptr;
    owner.m_membershipDirty = false;
}

bool PhysicsWorld::HasRagdollAssembly(uint64_t id) const
{
    const auto found = m_ragdolls.find(id);
    if (found == m_ragdolls.end())
        return false;
    const auto &view = found->second;
    auto *owner = dynamic_cast<Ragdoll *>(view.owner.GetComponent());
    if (!view.ready || !owner || owner->IsDestroyed() || owner->IsBeingDestroyed() || !owner->IsEnabled() ||
        !owner->GetGameObject() || owner->GetGameObject()->IsDestroying() ||
        !owner->GetGameObject()->IsActiveInHierarchy() || view.domain->retiring || view.domain->gpuCancelled ||
        view.epoch != view.domain->owner.simulation.simulationEpoch)
        return false;
    for (const auto &member : view.members) {
        auto *body = dynamic_cast<Rigidbody *>(member.body.GetComponent());
        if (!body || body->IsDestroyed() || body->IsBeingDestroyed() || !body->IsEnabled() ||
            body->GetParticipatingBodyId() != member.bodyId)
            return false;
        if (member.jointId) {
            auto *joint = dynamic_cast<Joint *>(member.joint.GetComponent());
            if (!joint || !joint->IsEnabled() || joint->IsDestroyed() || joint->IsBeingDestroyed() ||
                joint->GetPhysicsConstraintId() != member.jointId)
                return false;
        }
    }
    return true;
}

void PhysicsWorld::ReconcileRagdolls(PhysicsDomain &domain)
{
    if (domain.retiring || domain.reconcilingRagdolls || !domain.pendingRagdollHead)
        return;
    struct Guard
    {
        bool &active;
        explicit Guard(bool &value) : active(value)
        {
            active = true;
        }
        ~Guard()
        {
            active = false;
        }
    } guard(domain.reconcilingRagdolls);
    struct Draft
    {
        Ragdoll *owner = nullptr;
        uint64_t id = 0;
        RagdollAssembly value;
        size_t poseCount = 0;
        std::vector<size_t> activeInputs, animationInputs;
        std::vector<std::array<Collider *, 2>> pairs;
        bool active = false, newDrive = false;
    };
    std::vector<Draft> drafts;
    std::unordered_set<uint64_t> affected;
    auto &store = PhysicsECSStore::Instance();
    // Capture the entire affected batch before changing any ownership claims.
    while (domain.pendingRagdollHead) {
        auto *owner = domain.pendingRagdollHead;
        CancelRagdollRebuild(*owner);
        if (Component::FindByComponentId(owner->GetComponentID()) != owner) {
            const auto retained = m_ragdolls.find(owner->m_assemblyId);
            if (retained != m_ragdolls.end())
                retained->second.ready = false;
            continue; // Retained Scene registry nodes are unpublished until rollback.
        }
        Draft draft;
        draft.owner = owner;
        draft.id = owner->m_assemblyId;
        if (draft.id)
            affected.insert(draft.id);
        auto *root = owner->GetGameObject();
        draft.active = root && root->GetScene() && root->GetScene()->GetRuntimeWorld() == &domain.owner &&
                       !root->IsDestroying() && root->IsActiveInHierarchy() && owner->IsEnabled() &&
                       !owner->IsDestroyed() && !owner->IsBeingDestroyed();
        auto &view = draft.value;
        view.domain = &domain;
        view.owner = PhysicsTargetReference(owner);
        view.ownerIdentity = Identity(*owner);
        view.epoch = domain.owner.simulation.simulationEpoch;
        if (draft.active) {
            const auto configured = owner->GetConfiguredBodies();
            const std::unordered_set<Rigidbody *> configuredMembers(configured.begin(), configured.end());
            draft.poseCount = configured.size();
            std::unordered_map<Rigidbody *, int> indices;
            for (size_t index = 0; index < configured.size(); ++index) {
                auto *body = configured[index];
                if (!body->IsEnabled() || body->IsDestroyed() || body->IsBeingDestroyed() ||
                    !body->GetGameObject()->IsActiveInHierarchy())
                    continue;
                if (Ragdoll::FindOwner(body->GetGameObject()) != owner ||
                    body->GetGameObject()->GetScene()->GetRuntimeWorld() != &domain.owner)
                    throw std::invalid_argument("Ragdoll members must belong to their unique root and RuntimeWorld");
                if (!indices.emplace(body, static_cast<int>(view.members.size())).second)
                    throw std::invalid_argument("Ragdoll bodies must be unique");
                draft.activeInputs.push_back(index);
                RagdollMember member;
                member.body = PhysicsTargetReference(body);
                member.bodyIdentity = Identity(*body);
                member.bodyId = body->GetParticipatingBodyId();
                view.members.push_back(std::move(member));
            }
            view.ready = !view.members.empty() && !domain.gpuCancelled;
            std::vector<std::vector<Collider *>> surfaces(view.members.size());
            for (size_t index = 0; index < view.members.size(); ++index) {
                auto &member = view.members[index];
                auto *body = static_cast<Rigidbody *>(member.body.GetComponent());
                const auto actorHandle = store.GetRigidbody(body->GetECSHandle()).actorHandle;
                if (!store.IsValid(actorHandle))
                    view.ready = false;
                else {
                    const auto &actor = store.GetActor(actorHandle);
                    if (actor.sceneSuspended)
                        view.ready = false;
                    if (actor.runtimeWorld && actor.runtimeWorld != &domain.owner)
                        throw std::invalid_argument("Ragdoll actor belongs to another RuntimeWorld");
                    for (auto *collider : actor.colliders)
                        if (collider && !collider->IsDestroyed() && !collider->IsBeingDestroyed())
                            surfaces[index].push_back(collider);
                }
                if (member.bodyId == UINT32_MAX)
                    view.ready = false;
                auto *joint = body->GetGameObject()->GetComponent<Joint>();
                if (!joint || !joint->IsEnabled() || joint->IsDestroyed() || joint->IsBeingDestroyed())
                    continue;
                auto *parent = joint->GetConnectedBody();
                const auto parentIndex = indices.find(parent);
                if (!parent || !configuredMembers.count(parent))
                    continue; // External constraints do not belong to this assembly's drive graph.
                if (parentIndex == indices.end()) {
                    view.ready = false; // A configured internal endpoint is temporarily unavailable.
                    continue;
                }
                member.parent = parentIndex->second;
                member.joint = PhysicsTargetReference(joint);
                member.jointId = joint->GetPhysicsConstraintId();
                if (!member.jointId || !m_gpuScene)
                    view.ready = false;
                else {
                    const auto &identity = m_gpuScene->GetAuthoredJoint(member.jointId);
                    if (identity.bodyIdA != member.bodyId || identity.bodyIdB != view.members[member.parent].bodyId)
                        throw std::logic_error("Ragdoll joint endpoints differ from their component membership");
                    member.kind = identity.kind;
                }
            }
            for (size_t index = 0; index < view.members.size(); ++index) {
                const auto parent = view.members[index].parent;
                if (parent < 0)
                    continue;
                for (auto *a : surfaces[parent])
                    for (auto *b : surfaces[index]) {
                        if (a == b)
                            throw std::logic_error("A Collider cannot belong to both Ragdoll member actors");
                        draft.pairs.push_back({a, b});
                        view.pairs.push_back(PairIdentity(*a, *b));
                    }
            }
            std::sort(view.pairs.begin(), view.pairs.end());
            view.pairs.erase(std::unique(view.pairs.begin(), view.pairs.end()), view.pairs.end());
            if (auto *skeleton = owner->GetSkeleton()) {
                const auto bones = skeleton->GetBones();
                std::unordered_map<Transform *, size_t> boneIndices;
                for (size_t index = 0; index < bones.size(); ++index)
                    boneIndices.emplace(bones[index], index);
                for (const auto &member : view.members) {
                    auto *transform = member.body.GetGameObject()->GetTransform();
                    const auto found = boneIndices.find(transform);
                    if (found == boneIndices.end())
                        throw std::invalid_argument("Ragdoll animation member is not a bound skeleton node");
                    draft.animationInputs.push_back(found->second);
                }
            }
        }
        const auto old = m_ragdolls.find(draft.id);
        if (old == m_ragdolls.end())
            draft.newDrive = view.ready;
        else {
            draft.newDrive = view.ready && !old->second.ready;
            for (auto &member : view.members) {
                const auto previous =
                    std::find_if(old->second.members.begin(), old->second.members.end(), [&](const auto &entry) {
                        return entry.body.GetComponent() == member.body.GetComponent() &&
                               entry.joint.GetComponent() == member.joint.GetComponent();
                    });
                if (previous == old->second.members.end()) {
                    member.animationTarget = owner->m_driveAnimation;
                    draft.newDrive |= view.ready;
                } else {
                    member.animationTarget = previous->animationTarget;
                    if (view.ready && member.animationTarget && member.jointId != previous->jointId)
                        draft.newDrive = true;
                }
            }
        }
        if (old == m_ragdolls.end())
            for (auto &member : view.members)
                member.animationTarget = owner->m_driveAnimation;
        drafts.push_back(std::move(draft));
    }
    // Validate final claims jointly, allowing old/new roots in this same batch to transfer members.
    std::map<RagdollToken, size_t> finalClaims;
    std::unordered_set<uint64_t> retainedJoints;
    for (size_t index = 0; index < drafts.size(); ++index)
        for (const auto &member : drafts[index].value.members) {
            if (!finalClaims.emplace(member.bodyIdentity, index).second)
                throw std::invalid_argument("Two active Ragdoll assemblies claim the same Rigidbody");
            const auto old = m_ragdollBodyOwners.find(member.bodyIdentity);
            if (old != m_ragdollBodyOwners.end() && !affected.count(old->second))
                throw std::invalid_argument("Rigidbody is already owned by another active Ragdoll assembly");
            if (member.jointId)
                retainedJoints.insert(member.jointId);
        }
    std::vector<JoltGpuScene::PoseDriveIntent> clears;
    // First withdraw every old claim/filter before any new owner can bind.
    for (auto &draft : drafts) {
        const auto old = m_ragdolls.find(draft.id);
        if (old == m_ragdolls.end())
            continue;
        for (const auto &member : old->second.members) {
            m_ragdollBodyOwners.erase(member.bodyIdentity);
            auto *joint = dynamic_cast<Joint *>(member.joint.GetComponent());
            if (member.jointId && AngularDriveKind(member.kind) && !retainedJoints.count(member.jointId) && joint &&
                joint->GetPhysicsConstraintId() == member.jointId)
                clears.push_back({member.jointId, member.kind, {0, 0, 0, 1}, true});
        }
        if (old->second.pairs != draft.value.pairs)
            ReleaseColliderPairOwner(domain, old->second.ownerIdentity);
    }
    if (!clears.empty() && m_gpuScene && !domain.gpuCancelled && !domain.retiring)
        m_gpuScene->QueuePoseDrive(domain.owner, std::move(clears));
    // Then install every final membership and owner contribution as one author prefix.
    for (auto &draft : drafts) {
        auto &owner = *draft.owner;
        const auto old = m_ragdolls.find(draft.id);
        const bool pairsChanged = old == m_ragdolls.end() || old->second.pairs != draft.value.pairs;
        if (!draft.active || draft.value.members.empty()) {
            if (draft.id)
                m_ragdolls.erase(draft.id);
            owner.m_assemblyId = 0;
            owner.m_joints.clear();
            continue;
        }
        if (!draft.id) {
            if (!m_nextRagdollId)
                throw std::overflow_error("Ragdoll assembly identity space exhausted");
            draft.id = m_nextRagdollId++;
        }
        owner.m_assemblyId = draft.id;
        owner.m_poseInputCount = draft.poseCount;
        owner.m_activePoseInputs = std::move(draft.activeInputs);
        owner.m_animationPoseInputs = std::move(draft.animationInputs);
        owner.m_joints.clear();
        for (const auto &member : draft.value.members) {
            m_ragdollBodyOwners[member.bodyIdentity] = draft.id;
            owner.m_joints.push_back(member.joint);
        }
        owner.m_initializedAnimationJoints.erase(
            std::remove_if(owner.m_initializedAnimationJoints.begin(), owner.m_initializedAnimationJoints.end(),
                           [&](const auto &reference) {
                               auto *component = reference.GetComponent();
                               return !component || !owner.OwnsObject(component->GetGameObject());
                           }),
            owner.m_initializedAnimationJoints.end());
        m_ragdolls[draft.id] = std::move(draft.value);
        if (pairsChanged)
            for (const auto &pair : draft.pairs)
                AcquireColliderPairOwner(pair[0], pair[1], owner);
    }
    // Only after the complete commit may normal setters/drive queue further groups.
    for (auto &draft : drafts)
        if (draft.owner->m_driveAnimation && m_ragdolls.count(draft.owner->m_assemblyId)) {
            // Author motor initialization does not require a numerical pose or a bound GPU ID.
            draft.owner->InitializeAnimationDrive();
            if (HasRagdollAssembly(draft.owner->m_assemblyId) &&
                (draft.newDrive || draft.owner->m_animationTargetPending))
                if (draft.owner->UpdateAnimationDrive(!draft.owner->m_animationTargetPending))
                    draft.owner->m_animationTargetPending = false;
        }
}

uint32_t PhysicsWorld::RagdollTargetReset(uint32_t bodyId) const
{
    auto *body = FindRigidbodyByBodyId(bodyId);
    if (!body)
        return 0;
    const auto owner = m_ragdollBodyOwners.find(Identity(*body));
    auto *joint = body->GetGameObject()->GetComponent<Joint>();
    auto *parent = joint ? joint->GetConnectedBody() : nullptr;
    if (owner == m_ragdollBodyOwners.end() || !parent)
        return 0;
    const auto parentOwner = m_ragdollBodyOwners.find(Identity(*parent));
    return parentOwner != m_ragdollBodyOwners.end() && parentOwner->second == owner->second
               ? JPH::GPU::JointResetPoseDrive
               : 0u;
}

void PhysicsWorld::PrepareRagdollControl(const Joint &joint)
{
    auto *owner = Ragdoll::FindOwner(joint.GetGameObject());
    if (!owner || !owner->m_driveAnimation || owner->m_initializingAnimationDrive)
        return;
    const auto existing = m_ragdolls.find(owner->m_assemblyId);
    auto *domain = existing != m_ragdolls.end() ? existing->second.domain : nullptr;
    if (!domain)
        if (auto *object = owner->GetGameObject(); object && object->GetScene())
            if (auto *world = object->GetScene()->GetRuntimeWorld())
                domain = world->physics.get();
    if (!domain || domain->retiring || domain->reconcilingRagdolls)
        return;
    if (owner->m_membershipDirty || !owner->m_assemblyId || owner->m_animationTargetPending)
        owner->Rebuild();
}

void PhysicsWorld::ClaimRagdollTarget(const Joint &joint) noexcept
{
    auto *owner = Ragdoll::FindOwner(joint.GetGameObject());
    if (!owner)
        return;
    const auto found = m_ragdolls.find(owner->m_assemblyId);
    if (found == m_ragdolls.end())
        return;
    for (auto &member : found->second.members)
        if (member.joint.GetComponent() == &joint)
            member.animationTarget = false;
}

void PhysicsWorld::ClearRagdollDrive(uint64_t id)
{
    const auto found = m_ragdolls.find(id);
    if (found == m_ragdolls.end())
        return;
    auto &view = found->second;
    std::vector<JoltGpuScene::PoseDriveIntent> clears;
    for (auto &member : view.members) {
        member.animationTarget = false;
        auto *joint = dynamic_cast<Joint *>(member.joint.GetComponent());
        if (joint && member.jointId && joint->GetPhysicsConstraintId() == member.jointId &&
            AngularDriveKind(member.kind))
            clears.push_back({member.jointId, member.kind, {0, 0, 0, 1}, true});
    }
    if (!clears.empty() && m_gpuScene && !view.domain->gpuCancelled && !view.domain->retiring)
        m_gpuScene->QueuePoseDrive(view.domain->owner, std::move(clears));
}

void PhysicsWorld::DestroyRagdollAssembly(uint64_t id)
{
    const auto found = m_ragdolls.find(id);
    if (found == m_ragdolls.end())
        return;
    auto &view = found->second;
    ClearRagdollDrive(id);
    ReleaseColliderPairOwner(*view.domain, view.ownerIdentity);
    for (const auto &member : view.members) {
        const auto claim = m_ragdollBodyOwners.find(member.bodyIdentity);
        if (claim != m_ragdollBodyOwners.end() && claim->second == id)
            m_ragdollBodyOwners.erase(claim);
    }
    m_ragdolls.erase(found);
}

void PhysicsWorld::SuspendRagdolls(PhysicsDomain &domain) noexcept
{
    for (auto &[id, view] : m_ragdolls)
        if (view.domain == &domain) {
            view.ready = false;
            if (auto *owner = dynamic_cast<Ragdoll *>(view.owner.GetComponent()))
                QueueRagdollRebuild(*owner);
        }
}

void PhysicsWorld::RetireRagdolls(PhysicsDomain &domain) noexcept
{
    while (domain.pendingRagdollHead)
        CancelRagdollRebuild(*domain.pendingRagdollHead);
    for (auto it = m_ragdolls.begin(); it != m_ragdolls.end();) {
        auto &view = it->second;
        if (view.domain != &domain) {
            ++it;
            continue;
        }
        ReleaseColliderPairOwner(domain, view.ownerIdentity);
        for (const auto &member : view.members) {
            const auto claim = m_ragdollBodyOwners.find(member.bodyIdentity);
            if (claim != m_ragdollBodyOwners.end() && claim->second == it->first)
                m_ragdollBodyOwners.erase(claim);
        }
        it = m_ragdolls.erase(it);
    }
}

void PhysicsWorld::DriveRagdollPoseImpl(uint64_t id, const std::vector<glm::quat> &worldRotations, bool enableMotors,
                                        bool animationController, bool deferred)
{
    RequireOwnerThread();
    if (!HasRagdollAssembly(id))
        throw std::runtime_error("Ragdoll members have not been published");
    RequireGpu(*m_ragdolls.at(id).domain);
    auto &view = m_ragdolls.at(id);
    if (worldRotations.size() != view.members.size())
        throw std::invalid_argument("Ragdoll target must match its active member layout");
    std::vector<glm::quat> targets;
    targets.reserve(worldRotations.size());
    for (auto rotation : worldRotations)
        targets.push_back(TargetRotation(rotation));
    std::vector<JoltGpuScene::PoseDriveIntent> intents;
    for (size_t index = 0; index < view.members.size(); ++index) {
        auto &member = view.members[index];
        if (member.parent < 0 || !AngularDriveKind(member.kind) || (deferred && !member.animationTarget))
            continue;
        auto *joint = dynamic_cast<Joint *>(member.joint.GetComponent());
        if (enableMotors) {
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
        const auto relative = TargetRotation(glm::conjugate(targets[member.parent]) * targets[index]);
        member.animationTarget = animationController;
        intents.push_back({member.jointId, member.kind, {relative.x, relative.y, relative.z, relative.w}, false});
    }
    m_gpuScene->QueuePoseDrive(view.domain->owner, std::move(intents));
}

} // namespace infernux
