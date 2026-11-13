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
#include "../RuntimeWorld.h"
#include "../Scene.h"
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

RuntimeWorld *PublishedWorld(const GameObject *owner) noexcept
{
    auto *scene = owner ? owner->GetScene() : nullptr;
    auto *world = scene ? scene->GetRuntimeWorld() : nullptr;
    if (!world || scene->IsPreview() || scene->ResolveGameObject(owner->GetHandle()) != owner)
        return nullptr;
    if (scene != world->scenes.persistent.get() && world->scenes.loadedSet.find(scene) == world->scenes.loadedSet.end())
        return nullptr;
    return world;
}

bool CanAuthor(const PhysicsActorData &actor) noexcept
{
    // Publication fixed the owner. Subsequent commands only test admission;
    // they do not revalidate Scene membership or object lookup identities.
    const auto *scene = actor.owner->GetScene();
    return actor.runtimeWorld && scene && scene->GetRuntimeWorld() == actor.runtimeWorld && !actor.sceneSuspended &&
           !actor.owner->IsDestroying();
}

std::vector<uint32_t> ConsumeBroadphase(std::vector<uint32_t> &requests, std::unordered_set<uint32_t> &accepted)
{
    std::vector<uint32_t> result;
    result.reserve(requests.size());
    // Claim each final request once; cancelled older entries cannot resurface.
    for (auto it = requests.rbegin(); it != requests.rend(); ++it)
        if (accepted.erase(*it))
            result.push_back(*it);
    std::reverse(result.begin(), result.end());
    requests.clear();
    accepted.clear();
    return result;
}

} // namespace

PhysicsECSStore &PhysicsECSStore::Instance()
{
    // Intentionally leaked (see SceneManager::Instance) — Collider
    // destructors must always find a live store.
    static PhysicsECSStore *instance = new PhysicsECSStore();
    return *instance;
}

void PhysicsECSStore::RegisterRuntimeWorld(RuntimeWorld &world)
{
    auto [it, inserted] = m_worldQueues.try_emplace(world.id);
    if (!inserted && it->second.owner != &world)
        throw std::logic_error("Physics author World identity is already registered");
    it->second.owner = &world;
}

PhysicsECSStore::DomainAuthorQueues *PhysicsECSStore::FindQueues(const RuntimeWorld &world) noexcept
{
    const auto it = m_worldQueues.find(world.id);
    return it != m_worldQueues.end() && it->second.owner == &world ? &it->second : nullptr;
}

const PhysicsECSStore::DomainAuthorQueues *PhysicsECSStore::FindQueues(const RuntimeWorld &world) const noexcept
{
    const auto it = m_worldQueues.find(world.id);
    return it != m_worldQueues.end() && it->second.owner == &world ? &it->second : nullptr;
}

PhysicsECSStore::DomainAuthorQueues &PhysicsECSStore::RequireQueues(const RuntimeWorld &world)
{
    auto *queues = FindQueues(world);
    if (!queues)
        throw std::logic_error("Register the RuntimeWorld before publishing physics authors");
    return *queues;
}

void PhysicsECSStore::RetireRuntimeWorld(RuntimeWorld &world)
{
    m_actorPool.ForEachAlive([&](const PhysicsActorData &actor) {
        if (actor.runtimeWorld == &world)
            throw std::logic_error("Destroy a RuntimeWorld's actors before retiring its author queues");
    });
    ClearWorldPendingQueues(world);
    m_worldQueues.erase(world.id);
}

void PhysicsECSStore::BindActorWorld(ActorHandle handle)
{
    auto &actor = GetActor(handle);
    if (actor.runtimeWorld)
        return;
    auto *world = PublishedWorld(actor.owner);
    if (!world)
        return; // Author-only staging and detached retirement never bind a default World.
    auto &queues = RequireQueues(*world);
    actor.runtimeWorld = world;
    if (actor.rigidbody)
        ++queues.aliveRigidbodyCount;
    QueueBodyCreation(handle);
    MarkActorDirty(handle);
}

void PhysicsECSStore::ValidateHierarchyWorld(const GameObject *root, const RuntimeWorld *destination) const
{
    if (!root)
        return;
    const auto handle = FindActor(root);
    if (destination && IsValid(handle) && GetActor(handle).runtimeWorld && GetActor(handle).runtimeWorld != destination)
        throw std::invalid_argument("Moving a live physics actor between RuntimeWorlds is not implemented");
    for (const auto &child : root->GetChildren())
        ValidateHierarchyWorld(child.get(), destination);
}

void PhysicsECSStore::PublishHierarchyOwnership(GameObject *root)
{
    if (!root)
        return;
    const auto handle = FindActor(root);
    if (IsValid(handle))
        BindActorWorld(handle);
    // A staged Joint may already be waiting for an endpoint. Add/Attach root
    // publication has no Awake pass, so restore only that existing request.
    for (const auto &component : root->GetAllComponents())
        if (auto *joint = dynamic_cast<Joint *>(component.get()); joint && joint->m_retryOnRetire)
            QueuePendingJoint(*joint);
    for (const auto &child : root->GetChildren())
        PublishHierarchyOwnership(child.get());
}

PhysicsECSStore::ActorHandle PhysicsECSStore::AcquireActor(GameObject *owner)
{
    if (!owner)
        throw std::invalid_argument("Physics actor requires an owner");
    auto existing = m_actorByOwner.find(owner);
    if (existing != m_actorByOwner.end()) {
        if (!m_actorPool.IsAlive(existing->second))
            throw std::logic_error("Physics actor owner map contains a stale handle");
        BindActorWorld(existing->second);
        return existing->second;
    }

    ActorHandle handle = m_actorPool.Allocate();
    auto &actor = m_actorPool.Get(handle);
    actor = PhysicsActorData{};
    actor.owner = owner;
    m_actorByOwner.emplace(owner, handle);
    BindActorWorld(handle);
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
    const WorldExecutionScope worldScope(actor.runtimeWorld);
    if (actor.runtimeWorld)
        RequireQueues(*actor.runtimeWorld).pendingCreationSet.erase(ActorHandleKey(handle));
    if (actor.bodyId != 0xFFFFFFFF) {
        const bool pendingAdd = CancelBroadphaseAdd(*actor.runtimeWorld, actor.bodyId);
        const bool pendingRemove = CancelBroadphaseRemove(*actor.runtimeWorld, actor.bodyId);
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
    BindActorWorld(data.actorHandle);
    auto &actor = GetActor(data.actorHandle);
    if (!actor.rigidbody && actor.runtimeWorld)
        ++RequireQueues(*actor.runtimeWorld).aliveRigidbodyCount;
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
    if (actor.runtimeWorld)
        --RequireQueues(*actor.runtimeWorld).aliveRigidbodyCount;
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
    BindActorWorld(handle);
    auto &actor = GetActor(handle);
    if (!CanAuthor(actor))
        return;
    const WorldExecutionScope worldScope(actor.runtimeWorld);
    auto &world = PhysicsWorld::Instance();
    if (!world.IsInitialized())
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
        if (CancelBroadphaseRemove(*actor.runtimeWorld, actor.bodyId))
            actor.bodyInBroadphase = true;
        if (!actor.bodyInBroadphase) {
            QueueBroadphaseAdd(*actor.runtimeWorld, actor.bodyId);
            actor.bodyInBroadphase = true;
        }
    } else if (actor.bodyInBroadphase) {
        world.DestroyConstraintsForBody(actor.bodyId);
        if (!CancelBroadphaseAdd(*actor.runtimeWorld, actor.bodyId))
            QueueBroadphaseRemove(*actor.runtimeWorld, actor.bodyId);
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
        QueueBroadphaseAdd(*actor.runtimeWorld, actor.bodyId);
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
    const bool hasRigidbodies = GetAliveRigidbodyCount(*actor.runtimeWorld) > 0;
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

size_t PhysicsECSStore::GetAliveRigidbodyCount(const RuntimeWorld &world) const noexcept
{
    const auto *queues = FindQueues(world);
    return queues ? queues->aliveRigidbodyCount : 0;
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
    auto &actor = GetActor(actorHandle);
    if (!CanAuthor(actor) || actor.transformDirtyQueued)
        return;
    RequireQueues(*actor.runtimeWorld).dirtyActors.push_back(actorHandle);
    actor.transformDirtyQueued = true;
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

const std::vector<PhysicsECSStore::ActorHandle> &PhysicsECSStore::ConsumeDirtyActors(const RuntimeWorld &world)
{
    auto &queues = RequireQueues(world);
    queues.dirtyScratch.clear();
    queues.dirtyScratch.reserve(queues.dirtyActors.size());
    for (const auto handle : queues.dirtyActors) {
        if (!IsValid(handle) || GetActor(handle).runtimeWorld != &world)
            continue;
        GetActor(handle).transformDirtyQueued = false;
        queues.dirtyScratch.push_back(handle);
    }
    queues.dirtyActors.clear();
    return queues.dirtyScratch;
}

bool PhysicsECSStore::HasDirtyColliders(const RuntimeWorld &world) const noexcept
{
    const auto *queues = FindQueues(world);
    return queues && !queues->dirtyActors.empty();
}

void PhysicsECSStore::NotifyCollisionSceneChanged() noexcept
{
    if (++m_collisionSceneRevision == 0)
        ++m_collisionSceneRevision;
}

uint64_t PhysicsECSStore::GetCollisionSceneRevision() const noexcept
{
    return m_collisionSceneRevision + PhysicMaterial::GetSurfaceRevision();
}

void PhysicsECSStore::MarkAllCollidersDirty(const RuntimeWorld &world)
{
    NotifyCollisionSceneChanged();
    (void)RequireQueues(world);
    for (const auto handle : m_actorPool.GetAliveHandles())
        if (GetActor(handle).runtimeWorld == &world)
            MarkActorDirty(handle);
}

void PhysicsECSStore::QueueBodyCreation(ActorHandle handle)
{
    if (!IsValid(handle))
        return;
    const auto &actor = GetActor(handle);
    if (!CanAuthor(actor))
        return;
    auto &queues = RequireQueues(*actor.runtimeWorld);
    if (queues.pendingCreationSet.insert(ActorHandleKey(handle)).second)
        queues.pendingCreations.push_back(handle);
}

std::vector<PhysicsECSStore::ActorHandle> PhysicsECSStore::ConsumePendingBodyCreations(const RuntimeWorld &world)
{
    auto &queues = RequireQueues(world);
    std::vector<ActorHandle> pending;
    pending.swap(queues.pendingCreations);
    queues.pendingCreationSet.clear();
    pending.erase(
        std::remove_if(pending.begin(), pending.end(),
                       [&](ActorHandle handle) { return !IsValid(handle) || GetActor(handle).runtimeWorld != &world; }),
        pending.end());
    return pending;
}

bool PhysicsECSStore::HasPendingBodyCreations(const RuntimeWorld &world) const noexcept
{
    const auto *queues = FindQueues(world);
    return queues && !queues->pendingCreationSet.empty();
}

void PhysicsECSStore::QueueBroadphaseAdd(RuntimeWorld &world, uint32_t bodyId)
{
    if (bodyId == UINT32_MAX)
        return;
    auto &queues = RequireQueues(world);
    if (queues.broadphaseAddSet.insert(bodyId).second)
        queues.broadphaseAdds.push_back(bodyId);
}

bool PhysicsECSStore::CancelBroadphaseAdd(const RuntimeWorld &world, uint32_t bodyId)
{
    auto *queues = FindQueues(world);
    return bodyId != UINT32_MAX && queues && queues->broadphaseAddSet.erase(bodyId);
}

std::vector<uint32_t> PhysicsECSStore::ConsumePendingBroadphaseAdds(const RuntimeWorld &world)
{
    auto &queues = RequireQueues(world);
    return ConsumeBroadphase(queues.broadphaseAdds, queues.broadphaseAddSet);
}

bool PhysicsECSStore::HasPendingBroadphaseAdds(const RuntimeWorld &world) const noexcept
{
    const auto *queues = FindQueues(world);
    return queues && !queues->broadphaseAdds.empty();
}

void PhysicsECSStore::QueueBroadphaseRemove(RuntimeWorld &world, uint32_t bodyId)
{
    if (bodyId == UINT32_MAX)
        return;
    auto &queues = RequireQueues(world);
    if (queues.broadphaseRemoveSet.insert(bodyId).second)
        queues.broadphaseRemoves.push_back(bodyId);
}

bool PhysicsECSStore::CancelBroadphaseRemove(const RuntimeWorld &world, uint32_t bodyId)
{
    auto *queues = FindQueues(world);
    return bodyId != UINT32_MAX && queues && queues->broadphaseRemoveSet.erase(bodyId);
}

std::vector<uint32_t> PhysicsECSStore::ConsumePendingBroadphaseRemoves(const RuntimeWorld &world)
{
    auto &queues = RequireQueues(world);
    return ConsumeBroadphase(queues.broadphaseRemoves, queues.broadphaseRemoveSet);
}

bool PhysicsECSStore::HasPendingBroadphaseRemoves(const RuntimeWorld &world) const noexcept
{
    const auto *queues = FindQueues(world);
    return queues && !queues->broadphaseRemoves.empty();
}

void PhysicsECSStore::UpdateJointEndpointReadiness(PhysicsActorData &actor) noexcept
{
    const uint32_t endpoint = actor.bodyInBroadphase && !actor.sceneSuspended && actor.rigidbody &&
                                      actor.rigidbody->IsEnabled() && actor.owner->IsActiveInHierarchy()
                                  ? actor.bodyId
                                  : UINT32_MAX;
    if (endpoint != actor.jointEndpointBodyId) {
        actor.jointEndpointBodyId = endpoint;
        if (actor.runtimeWorld)
            if (auto *queues = FindQueues(*actor.runtimeWorld))
                queues->pendingJointsDirty = true;
    }
}

void PhysicsECSStore::QueuePendingJoint(Joint &joint) noexcept
{
    const auto actor = FindActor(joint.GetGameObject());
    auto *world = IsValid(actor) && CanAuthor(GetActor(actor)) ? GetActor(actor).runtimeWorld : nullptr;
    auto *queues = world ? FindQueues(*world) : nullptr;
    if (!queues)
        return; // Staging is author-only, with no default-world epoch or node.
    if (joint.m_pendingConstraint && joint.m_pendingWorld != world)
        return; // A live cross-World transfer must be rejected before mutation.
    joint.m_pendingEpoch = world->simulation.simulationEpoch;
    if (joint.m_pendingConstraint)
        return;
    joint.m_pendingWorld = world;
    joint.m_pendingConstraint = true;
    joint.m_pendingPrevious = queues->pendingJointTail;
    joint.m_pendingNext = nullptr;
    if (queues->pendingJointTail)
        queues->pendingJointTail->m_pendingNext = &joint;
    else
        queues->pendingJointHead = &joint;
    queues->pendingJointTail = &joint;
    queues->pendingJointsDirty = true;
}

void PhysicsECSStore::CancelPendingJoint(Joint &joint) noexcept
{
    if (!joint.m_pendingConstraint)
        return;
    auto *queues = joint.m_pendingWorld ? FindQueues(*joint.m_pendingWorld) : nullptr;
    if (joint.m_pendingPrevious)
        joint.m_pendingPrevious->m_pendingNext = joint.m_pendingNext;
    else if (queues)
        queues->pendingJointHead = joint.m_pendingNext;
    if (joint.m_pendingNext)
        joint.m_pendingNext->m_pendingPrevious = joint.m_pendingPrevious;
    else if (queues)
        queues->pendingJointTail = joint.m_pendingPrevious;
    joint.m_pendingPrevious = joint.m_pendingNext = nullptr;
    joint.m_pendingWorld = nullptr;
    joint.m_pendingConstraint = false;
}

void PhysicsECSStore::DrainPendingJoints(RuntimeWorld &world)
{
    auto &queues = RequireQueues(world);
    if (!queues.pendingJointsDirty || queues.drainingPendingJoints)
        return;
    const WorldExecutionScope worldScope(&world);
    queues.pendingJointsDirty = false;
    queues.drainingPendingJoints = true;
    try {
        for (auto *joint = queues.pendingJointHead; joint;) {
            auto *next = joint->m_pendingNext;
            if (joint->m_pendingEpoch != world.simulation.simulationEpoch) {
                joint->m_retryOnRetire = false;
                CancelPendingJoint(*joint);
            } else {
                joint->RebuildConstraint();
            }
            joint = next;
        }
    } catch (...) {
        queues.drainingPendingJoints = false;
        queues.pendingJointsDirty = true;
        throw;
    }
    queues.drainingPendingJoints = false;
}

void PhysicsECSStore::ClearPendingJoints(const RuntimeWorld &world) noexcept
{
    auto *queues = FindQueues(world);
    if (!queues)
        return;
    while (queues->pendingJointHead) {
        queues->pendingJointHead->m_retryOnRetire = false;
        CancelPendingJoint(*queues->pendingJointHead);
    }
    queues->pendingJointsDirty = false;
}

bool PhysicsECSStore::HasPendingJointDependencies(const RuntimeWorld &world) const noexcept
{
    const auto *queues = FindQueues(world);
    return queues && queues->pendingJointsDirty && queues->pendingJointHead;
}

void PhysicsECSStore::ClearWorldPendingQueues(const RuntimeWorld &world)
{
    auto *queues = FindQueues(world);
    if (!queues)
        return;
    ClearPendingJoints(world);
    queues->pendingCreations.clear();
    queues->pendingCreationSet.clear();
    queues->broadphaseAdds.clear();
    queues->broadphaseAddSet.clear();
    queues->broadphaseRemoves.clear();
    queues->broadphaseRemoveSet.clear();
    for (const auto handle : queues->dirtyActors)
        if (IsValid(handle) && GetActor(handle).runtimeWorld == &world)
            GetActor(handle).transformDirtyQueued = false;
    queues->dirtyActors.clear();
    queues->dirtyScratch.clear();
}

void PhysicsECSStore::ClearPendingQueues()
{
    for (const auto &[id, queues] : m_worldQueues) {
        (void)id;
        ClearWorldPendingQueues(*queues.owner);
    }
}

} // namespace infernux
