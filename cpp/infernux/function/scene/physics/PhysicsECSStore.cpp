/**
 * @file PhysicsECSStore.cpp
 * @brief Contiguous memory pools for Collider and Rigidbody data.
 *
 * Lifecycle invariants enforced by this file:
 *
 *   * Allocate*() always zero-initialises the entry before stamping the owner,
 *     so reused pool slots never inherit stale state from a previous owner.
 *   * Release*() always nulls the owner pointer first, then frees the slot;
 *     callers may safely observe a half-released entry (owner == nullptr,
 *     IsAlive == false).
 *   * Deferred collider/actor handles include their pool generation; retiring
 *     one World must not discard another World's pending physics work.
 *     Body destruction cancels its own queued broadphase commands.
 *     ClearPendingQueues() is reserved for complete SceneManager shutdown.
 */

#include "PhysicsECSStore.h"
#include "../Collider.h"
#include "../GameObject.h"
#include "../Joint.h"
#include "../Rigidbody.h"
#include "../SceneManager.h"
#include "../Transform.h"
#include "../TransformECSStore.h"
#include "PhysicsWorld.h"
#include <core/config/EngineConfig.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace infernux
{

namespace
{
uint64_t ActorHandleKey(PhysicsECSStore::ActorHandle handle)
{
    return (static_cast<uint64_t>(handle.generation) << 32U) | handle.index;
}

} // namespace

PhysicsECSStore &PhysicsECSStore::Instance()
{
    // Intentionally leaked (see SceneManager::Instance) — Collider
    // destructors must always find a live store.
    static PhysicsECSStore *instance = new PhysicsECSStore();
    return *instance;
}

PhysicsECSStore::ActorHandle PhysicsECSStore::AcquireActor(GameObject *owner)
{
    if (!owner)
        throw std::invalid_argument("Physics actor requires an owner");
    auto existing = m_actorByOwner.find(owner);
    if (existing != m_actorByOwner.end()) {
        if (!m_actorPool.IsAlive(existing->second))
            throw std::logic_error("Physics actor owner map contains a stale handle");
        return existing->second;
    }

    ActorHandle handle = m_actorPool.Allocate();
    auto &actor = m_actorPool.Get(handle);
    actor = PhysicsActorData{};
    actor.owner = owner;
    m_actorByOwner.emplace(owner, handle);
    return handle;
}

void PhysicsECSStore::AdoptActorOwner(GameObject *stagedOwner, GameObject *owner)
{
    auto actor = m_actorByOwner.extract(stagedOwner);
    if (actor.empty())
        return;
    m_actorPool.Get(actor.mapped()).owner = owner;
    actor.key() = owner;
    m_actorByOwner.insert(std::move(actor));
}

void PhysicsECSStore::ReleaseActor(ActorHandle handle, Collider *collider)
{
    if (!m_actorPool.IsAlive(handle))
        return;
    auto &actor = m_actorPool.Get(handle);
    actor.colliders.erase(std::remove(actor.colliders.begin(), actor.colliders.end(), collider), actor.colliders.end());
    if (actor.rigidbody || !actor.colliders.empty()) {
        if (!actor.owner->IsDestroying())
            RefreshActor(handle);
        return;
    }
    RetireActor(handle);
}

void PhysicsECSStore::RetireActor(ActorHandle handle)
{
    auto &actor = m_actorPool.Get(handle);
    m_pendingBodyCreationSet.erase(ActorHandleKey(handle));
    if (actor.bodyId != 0xFFFFFFFF) {
        const bool pendingAdd = CancelBroadphaseAdd(actor.bodyId);
        const bool pendingRemove = CancelBroadphaseRemove(actor.bodyId);
        auto &world = PhysicsWorld::Instance();
        if (!pendingAdd && (actor.bodyInBroadphase || pendingRemove))
            world.RemoveBodyFromBroadphase(actor.bodyId);
        world.DestroyBody(actor.bodyId);
    }
    m_actorByOwner.erase(actor.owner);
    actor = PhysicsActorData{};
    m_actorPool.Free(handle);
}

PhysicsECSStore::ActorHandle PhysicsECSStore::FindActor(const GameObject *owner) const
{
    const auto found = m_actorByOwner.find(const_cast<GameObject *>(owner));
    return found == m_actorByOwner.end() ? ActorHandle{} : found->second;
}

void PhysicsECSStore::AttachRigidbody(Rigidbody *body)
{
    BindRigidbodyActor(body);
    RefreshHierarchy(body->GetGameObject());
}

PhysicsECSStore::ActorHandle PhysicsECSStore::BindRigidbodyActor(Rigidbody *body)
{
    auto &data = GetRigidbody(body->GetECSHandle());
    if (!IsValid(data.actorHandle))
        data.actorHandle = AcquireActor(body->GetGameObject());
    auto &actor = GetActor(data.actorHandle);
    actor.rigidbody = body;
    QueueBodyCreation(data.actorHandle);
    MarkActorDirty(data.actorHandle);
    return data.actorHandle;
}

Rigidbody *PhysicsECSStore::FindAttachedRigidbody(GameObject *object, const Rigidbody *excludedBody)
{
    for (auto *ancestor = object; ancestor; ancestor = ancestor->GetParent()) {
        auto *body = ancestor->GetComponent<Rigidbody>();
        if (body && body != excludedBody && !body->IsBeingDestroyed() && !body->IsDestroyed())
            return body;
    }
    return nullptr;
}

void PhysicsECSStore::MigrateCollider(Collider *collider, const Rigidbody *excludedBody,
                                      std::vector<ActorHandle> &changed)
{
    const auto colliderHandle = collider->GetECSHandle();
    const auto previous = GetCollider(colliderHandle).actorHandle;
    auto *body = FindAttachedRigidbody(collider->GetGameObject(), excludedBody);
    const auto target = body ? BindRigidbodyActor(body) : AcquireActor(collider->GetGameObject());
    if (target == previous)
        return;
    // Acquiring the target can grow the actor pool: re-read by handle below.
    GetCollider(colliderHandle).actorHandle = target;
    GetActor(target).colliders.push_back(collider);
    changed.push_back(target);
    if (IsValid(previous)) {
        auto &old = GetActor(previous);
        old.colliders.erase(std::remove(old.colliders.begin(), old.colliders.end(), collider), old.colliders.end());
        if (!old.rigidbody && old.colliders.empty())
            RetireActor(previous);
        else
            changed.push_back(previous);
    }
}

void PhysicsECSStore::PublishChangedActors(std::vector<ActorHandle> &changed)
{
    std::unordered_set<uint64_t> published;
    for (const auto handle : changed)
        if (published.insert(ActorHandleKey(handle)).second)
            RefreshActor(handle);
}

void PhysicsECSStore::AttachCollider(Collider *collider)
{
    std::vector<ActorHandle> changed;
    MigrateCollider(collider, nullptr, changed);
    PublishChangedActors(changed);
}

void PhysicsECSStore::RefreshHierarchy(GameObject *root, const Rigidbody *excludedBody)
{
    std::vector<ActorHandle> changed;
    const auto visit = [&](auto &&self, GameObject *object) -> void {
        if (object->IsDestroying())
            return;
        if (object != root) {
            auto *body = object->GetComponent<Rigidbody>();
            if (body && body != excludedBody && !body->IsBeingDestroyed())
                return; // A nested Rigidbody owns its own subtree.
        }
        for (const auto &component : object->GetAllComponents()) {
            auto *collider = dynamic_cast<Collider *>(component.get());
            if (collider && !collider->IsBeingDestroyed())
                MigrateCollider(collider, excludedBody, changed);
        }
        for (size_t index = 0; index < object->GetChildCount(); ++index)
            self(self, object->GetChild(index));
    };
    visit(visit, root);
    PublishChangedActors(changed);
}

void PhysicsECSStore::DetachRigidbody(Rigidbody *body)
{
    const auto bodyHandle = body->GetECSHandle();
    const auto handle = GetRigidbody(bodyHandle).actorHandle;
    if (!IsValid(handle)) {
        GetRigidbody(bodyHandle).physicsTargetSources = {};
        return;
    }
    // A live owner keeps its current device pose when this component leaves.
    // Complete this demand before exposing a half-detached actor to authoring
    // collection. Whole-object retirement has no CPU pose consumer.
    if (!GetActor(handle).owner->IsDestroying() && GetActor(handle).bodyId != UINT32_MAX)
        PhysicsWorld::Instance().MaterializeCpuPoseForRelease(*GetActor(handle).owner->GetTransform());
    // Demand collection can grow either pool; retain identities, not references.
    if (!IsValid(bodyHandle))
        return;
    GetRigidbody(bodyHandle).physicsTargetSources = {};
    if (!IsValid(handle))
        return;
    GetRigidbody(bodyHandle).actorHandle = {};
    auto &actor = GetActor(handle);
    actor.rigidbody = nullptr;
    if (actor.bodyId != 0xFFFFFFFF)
        PhysicsWorld::Instance().DestroyConstraintsForBody(actor.bodyId);
    if (actor.owner->IsDestroying()) {
        for (auto *collider : actor.colliders)
            GetCollider(collider->GetECSHandle()).actorHandle = {};
        RetireActor(handle);
    } else {
        RefreshHierarchy(body->GetGameObject(), body);
        if (IsValid(handle)) {
            if (GetActor(handle).colliders.empty())
                RetireActor(handle);
            else
                RefreshActor(handle);
        }
    }
}

void PhysicsECSStore::RegisterActor(ActorHandle handle)
{
    if (!IsValid(handle))
        return;
    auto &actor = GetActor(handle);
    auto &world = PhysicsWorld::Instance();
    if (!world.IsInitialized() || actor.sceneSuspended ||
        !SceneManager::Instance().IsRuntimeScene(actor.owner->GetScene()))
        return;
    actor.primaryCollider = nullptr;
    for (auto *collider : actor.colliders) {
        if (collider->IsEnabled() && !collider->IsBeingDestroyed() &&
            collider->GetGameObject()->IsActiveInHierarchy()) {
            actor.primaryCollider = collider;
            break;
        }
    }
    const bool dynamicOwner = actor.rigidbody && actor.rigidbody->IsEnabled();
    const bool active = actor.owner->IsActiveInHierarchy() && (dynamicOwner || actor.primaryCollider);
    if (dynamicOwner)
        actor.rigidbody->CaptureAuthorPoseCache();
    if (actor.bodyId == 0xFFFFFFFF) {
        if (!active)
            return;
        auto *transform = actor.owner->GetTransform();
        glm::vec3 position, scale;
        glm::quat rotation;
        {
            TransformECSStore::PhysicsPoseReadScope authoring(TransformECSStore::Instance());
            position = transform->GetPosition();
            rotation = transform->GetWorldRotation();
            scale = transform->GetWorldScale();
        }
        actor.bodyId = world.CreateBody(actor);
        if (actor.bodyId == 0xFFFFFFFF)
            return; // A mesh member is still being cooked.
        actor.lastSyncedPos = position;
        actor.lastSyncedRot = rotation;
        actor.lastScale = scale;
    } else {
        world.UpdateBodyShape(actor);
    }
    if (active) {
        if (CancelBroadphaseRemove(actor.bodyId))
            actor.bodyInBroadphase = true;
        if (!actor.bodyInBroadphase) {
            QueueBroadphaseAdd(actor.bodyId);
            actor.bodyInBroadphase = true;
        }
    } else if (actor.bodyInBroadphase) {
        world.DestroyConstraintsForBody(actor.bodyId);
        if (!CancelBroadphaseAdd(actor.bodyId))
            QueueBroadphaseRemove(actor.bodyId);
        actor.bodyInBroadphase = false;
    }
    UpdateJointEndpointReadiness(actor);
    NotifyCollisionSceneChanged();
}

void PhysicsECSStore::RefreshActor(ActorHandle handle)
{
    if (!IsValid(handle))
        return;
    if (GetActor(handle).bodyId == 0xFFFFFFFF)
        QueueBodyCreation(handle);
    else
        RegisterActor(handle);
}

void PhysicsECSStore::SuspendActor(ActorHandle handle)
{
    auto &actor = GetActor(handle);
    actor.sceneSuspended = true;
    if (actor.bodyInBroadphase) {
        PhysicsWorld::Instance().RemoveBodyFromBroadphase(actor.bodyId, true);
        actor.bodyInBroadphase = false;
    }
    UpdateJointEndpointReadiness(actor);
}

void PhysicsECSStore::RestoreActor(ActorHandle handle)
{
    auto &actor = GetActor(handle);
    actor.sceneSuspended = false;
    if (actor.bodyId != 0xFFFFFFFF && actor.owner->IsActiveInHierarchy() &&
        ((actor.rigidbody && actor.rigidbody->IsEnabled()) || actor.primaryCollider)) {
        QueueBroadphaseAdd(actor.bodyId);
        actor.bodyInBroadphase = true;
    }
    UpdateJointEndpointReadiness(actor);
}

void PhysicsECSStore::SyncActorTransform(ActorHandle handle, float fixedDeltaTime,
                                         std::vector<PhysicsBodyPoseUpdate> *staticPoseBatch)
{
    auto &actor = GetActor(handle);
    if (actor.bodyId == 0xFFFFFFFF || actor.sceneSuspended)
        return;
    auto *transform = actor.owner->GetTransform();
    const auto scale = transform->GetWorldScale();
    const uint8_t ancestorPoseChannels = actor.ancestorPoseChannels;
    actor.ancestorPoseChannels = 0;
    bool centerChanged = false;
    if (actor.centerTransformDirty) {
        actor.centerTransformDirty = false;
        for (auto *collider : actor.colliders) {
            if (collider->GetGameObject() != actor.owner || !collider->IsEnabled() ||
                collider->GetCenter() == glm::vec3(0.0f))
                continue;
            const auto center = collider->GetCenterInRigidSpace();
            const auto published = GetCollider(collider->GetECSHandle()).publishedRigidCenter;
            // Only suppress roundoff from the world-matrix/quaternion conversion;
            // this scales with the offset rather than imposing a physical distance.
            const float tolerance =
                32 * std::numeric_limits<float>::epsilon() * std::max(glm::length(center), glm::length(published));
            if (glm::length(center - published) > tolerance) {
                centerChanged = true;
                break;
            }
        }
    }
    if (actor.shapeTransformDirty || scale != actor.lastScale || centerChanged) {
        actor.shapeTransformDirty = false;
        actor.lastScale = scale;
        PhysicsWorld::Instance().UpdateBodyShape(actor);
    }
    if (actor.rigidbody && actor.rigidbody->IsEnabled()) {
        actor.rigidbody->SyncExternalMovesToPhysics(fixedDeltaTime, ancestorPoseChannels);
        return;
    }
    const auto position = transform->GetPosition();
    const auto rotation = transform->GetWorldRotation();
    if (position == actor.lastSyncedPos && rotation == actor.lastSyncedRot)
        return;
    actor.lastSyncedPos = position;
    actor.lastSyncedRot = rotation;
    ++actor.transformRevision;
    auto &world = PhysicsWorld::Instance();
    const bool hasRigidbodies = GetAliveRigidbodyCount() > 0;
    if (fixedDeltaTime > 0.0f && hasRigidbodies) {
        world.MoveStaticBodyWithVelocity(actor.bodyId, position, rotation, fixedDeltaTime);
    } else {
        if (staticPoseBatch)
            staticPoseBatch->push_back({actor.bodyId, position, rotation});
        else
            world.SetBodyPosition(actor.bodyId, position, rotation);
    }
}

bool PhysicsECSStore::IsValid(ActorHandle handle) const
{
    return m_actorPool.IsAlive(handle);
}

PhysicsActorData &PhysicsECSStore::GetActor(ActorHandle handle)
{
    return m_actorPool.Get(handle);
}

const PhysicsActorData &PhysicsECSStore::GetActor(ActorHandle handle) const
{
    return m_actorPool.Get(handle);
}

// ============================================================================
// Collider pool
// ============================================================================

PhysicsECSStore::ColliderHandle PhysicsECSStore::AllocateCollider(Collider *owner)
{
    ColliderHandle handle = m_colliderPool.Allocate();
    ColliderECSData &data = m_colliderPool.Get(handle);
    data = ColliderECSData{};
    data.owner = owner;
    NotifyCollisionSceneChanged();
    return handle;
}

void PhysicsECSStore::ReleaseCollider(ColliderHandle handle)
{
    if (!m_colliderPool.IsAlive(handle))
        return;
    m_colliderPool.Get(handle).owner = nullptr;
    m_colliderPool.Free(handle);
    NotifyCollisionSceneChanged();
}

bool PhysicsECSStore::IsValid(ColliderHandle handle) const
{
    return m_colliderPool.IsAlive(handle);
}

ColliderECSData &PhysicsECSStore::GetCollider(ColliderHandle handle)
{
    return m_colliderPool.Get(handle);
}

const ColliderECSData &PhysicsECSStore::GetCollider(ColliderHandle handle) const
{
    return m_colliderPool.Get(handle);
}

// ============================================================================
// Rigidbody pool
// ============================================================================

PhysicsECSStore::RigidbodyHandle PhysicsECSStore::AllocateRigidbody(Rigidbody *owner)
{
    RigidbodyHandle handle = m_rigidbodyPool.Allocate();
    RigidbodyECSData &data = m_rigidbodyPool.Get(handle);
    data = RigidbodyECSData{};
    const auto &config = EngineConfig::Get();
    data.mass = config.defaultRigidbodyMass;
    data.drag = config.defaultRigidbodyDrag;
    data.angularDrag = config.defaultRigidbodyAngularDrag;
    data.maxAngularVelocity = config.defaultMaxAngularVelocity;
    data.maxLinearVelocity = config.defaultMaxLinearVelocity;
    data.owner = owner;
    return handle;
}

void PhysicsECSStore::ReleaseRigidbody(RigidbodyHandle handle)
{
    if (!m_rigidbodyPool.IsAlive(handle))
        return;

    DetachRigidbody(m_rigidbodyPool.Get(handle).owner);

    m_rigidbodyPool.Get(handle).owner = nullptr;
    m_rigidbodyPool.Free(handle);
}

bool PhysicsECSStore::IsValid(RigidbodyHandle handle) const
{
    return m_rigidbodyPool.IsAlive(handle);
}

RigidbodyECSData &PhysicsECSStore::GetRigidbody(RigidbodyHandle handle)
{
    return m_rigidbodyPool.Get(handle);
}

const RigidbodyECSData &PhysicsECSStore::GetRigidbody(RigidbodyHandle handle) const
{
    return m_rigidbodyPool.Get(handle);
}

// ============================================================================
// Dirty collider tracking
// ============================================================================

void PhysicsECSStore::MarkColliderDirty(ColliderHandle handle)
{
    if (!m_colliderPool.IsAlive(handle))
        return;
    NotifyCollisionSceneChanged();
    MarkActorDirty(m_colliderPool.Get(handle).actorHandle);
}

void PhysicsECSStore::MarkActorDirty(ActorHandle actorHandle)
{
    if (!IsValid(actorHandle))
        return;
    auto &actor = m_actorPool.Get(actorHandle);
    if (actor.transformDirtyQueued)
        return;
    actor.transformDirtyQueued = true;
    m_dirtyActorList.push_back(actorHandle);
}

void PhysicsECSStore::MarkGameObjectDirty(GameObject *owner, const Transform *physicsSource,
                                          const Transform *authoredSource, bool rotationAuthored)
{
    if (!owner)
        return;
    if (physicsSource) {
        const auto *sourceOwner = physicsSource->GetGameObject();
        for (auto *current = owner; current && current != sourceOwner; current = current->GetParent()) {
            const auto *body = current->GetComponent<Rigidbody>();
            if (body && body->IsEnabled() && body->HasBody())
                return;
        }
    }
    const auto found = m_actorByOwner.find(owner);
    if (found != m_actorByOwner.end()) {
        if (!physicsSource || physicsSource->GetGameObject() != owner) {
            if (!physicsSource) {
                GetActor(found->second).centerTransformDirty = true;
                if (authoredSource && authoredSource->GetGameObject() != owner)
                    GetActor(found->second).ancestorPoseChannels |= rotationAuthored ? 3u : 1u;
            }
            NotifyCollisionSceneChanged();
            MarkActorDirty(found->second);
        }
        return;
    }
    // All colliders on this object have the same actor. Walk the existing
    // component storage once; never allocate a child search in a frame loop.
    for (const auto &component : owner->GetAllComponents()) {
        auto *collider = dynamic_cast<Collider *>(component.get());
        if (!collider)
            continue;
        const auto handle = GetCollider(collider->GetECSHandle()).actorHandle;
        if (IsValid(handle) && !physicsSource) {
            GetActor(handle).shapeTransformDirty = true;
            NotifyCollisionSceneChanged();
            MarkActorDirty(handle);
        }
        break;
    }
}

void PhysicsECSStore::MarkAuthoredHierarchyDirty(GameObject *root, bool rotationAuthored)
{
    const auto visit = [&](auto &&self, GameObject *object) -> void {
        MarkGameObjectDirty(object, nullptr, root->GetTransform(), rotationAuthored);
        for (size_t index = 0; index < object->GetChildCount(); ++index)
            self(self, object->GetChild(index));
    };
    visit(visit, root);
}

const std::vector<PhysicsECSStore::ActorHandle> &PhysicsECSStore::ConsumeDirtyActors()
{
    m_dirtyActorScratch.clear();
    m_dirtyActorScratch.reserve(m_dirtyActorList.size());
    for (const ActorHandle actorHandle : m_dirtyActorList) {
        if (!m_actorPool.IsAlive(actorHandle))
            continue;
        auto &actor = m_actorPool.Get(actorHandle);
        actor.transformDirtyQueued = false;
        m_dirtyActorScratch.push_back(actorHandle);
    }
    m_dirtyActorList.clear();
    return m_dirtyActorScratch;
}

void PhysicsECSStore::NotifyCollisionSceneChanged() noexcept
{
    ++m_collisionSceneRevision;
    if (m_collisionSceneRevision == 0)
        ++m_collisionSceneRevision;
}

uint64_t PhysicsECSStore::GetCollisionSceneRevision() const noexcept
{
    return m_collisionSceneRevision + PhysicMaterial::GetSurfaceRevision();
}

void PhysicsECSStore::MarkAllCollidersDirty()
{
    NotifyCollisionSceneChanged();
    for (const ActorHandle actorHandle : m_dirtyActorList) {
        if (m_actorPool.IsAlive(actorHandle))
            m_actorPool.Get(actorHandle).transformDirtyQueued = false;
    }
    m_dirtyActorList.clear();

    const auto actors = m_actorPool.GetAliveHandles();
    m_dirtyActorList.reserve(actors.size());
    for (const ActorHandle actorHandle : actors) {
        auto &actor = m_actorPool.Get(actorHandle);
        actor.transformDirtyQueued = true;
        m_dirtyActorList.push_back(actorHandle);
    }
}

// ============================================================================
// Pending body creation queue
// ============================================================================

void PhysicsECSStore::QueueBodyCreation(ActorHandle handle)
{
    if (IsValid(handle)) {
        if (m_pendingBodyCreationSet.insert(ActorHandleKey(handle)).second)
            m_pendingBodyCreationList.push_back(handle);
    }
}

std::vector<PhysicsECSStore::ActorHandle> PhysicsECSStore::ConsumePendingBodyCreations()
{
    std::vector<ActorHandle> pending;
    pending.swap(m_pendingBodyCreationList);
    m_pendingBodyCreationSet.clear();
    pending.erase(
        std::remove_if(pending.begin(), pending.end(), [this](ActorHandle handle) { return !IsValid(handle); }),
        pending.end());
    return pending;
}

// ============================================================================
// Pending broadphase queue
// ============================================================================

void PhysicsECSStore::QueueBroadphaseAdd(uint32_t bodyId)
{
    if (bodyId == 0xFFFFFFFF)
        return;
    if (m_pendingBroadphaseSet.insert(bodyId).second)
        m_pendingBroadphaseAdds.push_back(bodyId);
}

std::vector<uint32_t> PhysicsECSStore::ConsumePendingBroadphaseAdds()
{
    std::vector<uint32_t> result;
    result.reserve(m_pendingBroadphaseAdds.size());
    // A later request must not revive an older canceled queue entry. Claim
    // each body's latest request once, then restore final request order.
    for (auto it = m_pendingBroadphaseAdds.rbegin(); it != m_pendingBroadphaseAdds.rend(); ++it) {
        if (m_pendingBroadphaseSet.erase(*it) != 0)
            result.push_back(*it);
    }
    std::reverse(result.begin(), result.end());
    m_pendingBroadphaseAdds.clear();
    m_pendingBroadphaseSet.clear();
    return result;
}

bool PhysicsECSStore::CancelBroadphaseAdd(uint32_t bodyId)
{
    return bodyId != 0xFFFFFFFF && m_pendingBroadphaseSet.erase(bodyId) != 0;
}

void PhysicsECSStore::QueueBroadphaseRemove(uint32_t bodyId)
{
    if (bodyId == 0xFFFFFFFF)
        return;
    if (m_pendingBroadphaseRemoveSet.insert(bodyId).second)
        m_pendingBroadphaseRemoves.push_back(bodyId);
}

bool PhysicsECSStore::CancelBroadphaseRemove(uint32_t bodyId)
{
    return bodyId != 0xFFFFFFFF && m_pendingBroadphaseRemoveSet.erase(bodyId) != 0;
}

std::vector<uint32_t> PhysicsECSStore::ConsumePendingBroadphaseRemoves()
{
    std::vector<uint32_t> result;
    result.reserve(m_pendingBroadphaseRemoves.size());
    for (auto it = m_pendingBroadphaseRemoves.rbegin(); it != m_pendingBroadphaseRemoves.rend(); ++it) {
        if (m_pendingBroadphaseRemoveSet.erase(*it) != 0)
            result.push_back(*it);
    }
    std::reverse(result.begin(), result.end());
    m_pendingBroadphaseRemoves.clear();
    m_pendingBroadphaseRemoveSet.clear();
    return result;
}

void PhysicsECSStore::UpdateJointEndpointReadiness(PhysicsActorData &actor) noexcept
{
    const uint32_t endpoint = actor.bodyInBroadphase && !actor.sceneSuspended && actor.rigidbody &&
                                      actor.rigidbody->IsEnabled() && actor.owner->IsActiveInHierarchy()
                                  ? actor.bodyId
                                  : UINT32_MAX;
    if (endpoint != actor.jointEndpointBodyId) {
        actor.jointEndpointBodyId = endpoint;
        m_pendingJointsDirty = true;
    }
}

void PhysicsECSStore::QueuePendingJoint(Joint &joint) noexcept
{
    joint.m_pendingEpoch = SceneManager::Instance().GetSimulationEpoch();
    if (joint.m_pendingConstraint)
        return;
    joint.m_pendingConstraint = true;
    joint.m_pendingPrevious = m_pendingJointTail;
    joint.m_pendingNext = nullptr;
    if (m_pendingJointTail)
        m_pendingJointTail->m_pendingNext = &joint;
    else
        m_pendingJointHead = &joint;
    m_pendingJointTail = &joint;
    m_pendingJointsDirty = true;
}

void PhysicsECSStore::CancelPendingJoint(Joint &joint) noexcept
{
    if (!joint.m_pendingConstraint)
        return;
    if (joint.m_pendingPrevious)
        joint.m_pendingPrevious->m_pendingNext = joint.m_pendingNext;
    else
        m_pendingJointHead = joint.m_pendingNext;
    if (joint.m_pendingNext)
        joint.m_pendingNext->m_pendingPrevious = joint.m_pendingPrevious;
    else
        m_pendingJointTail = joint.m_pendingPrevious;
    joint.m_pendingPrevious = joint.m_pendingNext = nullptr;
    joint.m_pendingConstraint = false;
}

void PhysicsECSStore::DrainPendingJoints()
{
    if (!m_pendingJointsDirty || m_drainingPendingJoints)
        return;
    m_pendingJointsDirty = false;
    m_drainingPendingJoints = true;
    try {
        auto &manager = SceneManager::Instance();
        // Rebuild/OnConstraintCreated only append physics author commands; they
        // do not dispatch gameplay callbacks or destroy other component nodes.
        for (auto *joint = m_pendingJointHead; joint;) {
            auto *next = joint->m_pendingNext;
            if (joint->m_pendingEpoch != manager.GetSimulationEpoch()) {
                joint->m_retryOnRetire = false;
                CancelPendingJoint(*joint);
            } else {
                joint->RebuildConstraint();
            }
            joint = next;
        }
    } catch (...) {
        m_drainingPendingJoints = false;
        m_pendingJointsDirty = true;
        throw;
    }
    m_drainingPendingJoints = false;
}

void PhysicsECSStore::ClearPendingJoints() noexcept
{
    while (m_pendingJointHead) {
        m_pendingJointHead->m_retryOnRetire = false;
        CancelPendingJoint(*m_pendingJointHead);
    }
    m_pendingJointsDirty = false;
}

void PhysicsECSStore::ClearPendingQueues()
{
    ClearPendingJoints();
    m_pendingBodyCreationList.clear();
    m_pendingBodyCreationSet.clear();
    m_pendingBroadphaseAdds.clear();
    m_pendingBroadphaseSet.clear();
    m_pendingBroadphaseRemoves.clear();
    m_pendingBroadphaseRemoveSet.clear();
    for (const ActorHandle actorHandle : m_dirtyActorList) {
        if (m_actorPool.IsAlive(actorHandle))
            m_actorPool.Get(actorHandle).transformDirtyQueued = false;
    }
    m_dirtyActorList.clear();
}

} // namespace infernux
