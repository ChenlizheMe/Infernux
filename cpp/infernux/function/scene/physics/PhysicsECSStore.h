#pragma once

/**
 * @file PhysicsECSStore.h
 * @brief Contiguous memory pools for Collider and Rigidbody hot data.
 *
 * Mirrors the TransformECSStore pattern: Component classes hold a Handle into
 * the pool; every-frame sync loops iterate contiguous data for cache-friendly
 * access.
 */

#include "PhysicsTargetReference.h"
#include "core/types/InxContiguousPool.h"
#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace infernux
{

class Collider;
class GameObject;
class Joint;
class Rigidbody;
class RuntimeWorld;
class Transform;

struct PhysicsActorData
{
    GameObject *owner = nullptr;
    // Bound only after the author hierarchy is published. Retained through
    // Scene detachment until this actor is destroyed; never a default World.
    RuntimeWorld *runtimeWorld = nullptr;
    Collider *primaryCollider = nullptr;
    Rigidbody *rigidbody = nullptr;
    uint32_t bodyId = 0xFFFFFFFF;
    // Last authored Joint endpoint readiness; no numerical body state is cached.
    uint32_t jointEndpointBodyId = 0xFFFFFFFF;
    std::vector<Collider *> colliders;
    bool bodyInBroadphase = false;
    bool sceneSuspended = false;
    bool shapeTransformDirty = false;
    bool centerTransformDirty = false;
    uint8_t ancestorPoseChannels = 0;
    glm::vec3 lastSyncedPos{0.0f};
    glm::quat lastSyncedRot{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 lastScale{1.0f};
    uint64_t shapeRevision = 0;
    uint64_t transformRevision = 0;
    bool transformDirtyQueued = false;
};

using PhysicsActorPool = InxContiguousPool<PhysicsActorData>;
using PhysicsActorHandle = PhysicsActorPool::Handle;

// ============================================================================
// Pooled Collider data (hot path — touched every SyncCollidersToPhysics)
// ============================================================================

struct ColliderECSData
{
    // ---- Identity ----
    Collider *owner = nullptr;

    PhysicsActorHandle actorHandle;

    // ---- Properties ----
    bool isTrigger = false;
    glm::vec3 center{0.0f};
    glm::vec3 publishedRigidCenter{0.0f};

    // ---- Misc ----
    bool deserialized = false;
};

// ============================================================================
// Pooled Rigidbody data (hot path — touched every SyncPhysicsToTransform)
// ============================================================================

struct RigidbodyECSData
{
    // ---- Identity ----
    Rigidbody *owner = nullptr;
    PhysicsActorHandle actorHandle;

    // ---- Serialized properties ----
    float mass = 1.0f;
    float drag = 0.0f;
    float angularDrag = 0.05f;
    bool useGravity = true;
    bool isKinematic = false;
    int constraints = 0;
    int collisionDetectionMode = 0;
    int interpolation = 0;
    float maxAngularVelocity = 7.0f;
    float maxLinearVelocity = 500.0f;

    // ---- Runtime velocity state (also survives deferred body creation) ----
    glm::vec3 linearVelocity{0.0f};
    glm::vec3 angularVelocity{0.0f};
    bool hasLinearVelocity = false;
    bool hasAngularVelocity = false;

    // ---- Sync cache (external-move detection) ----
    glm::vec3 lastSyncedPosition{0.0f};
    glm::quat lastSyncedRotation{1.0f, 0.0f, 0.0f, 0.0f};
    bool hasSyncedOnce = false;
    std::array<uint64_t, 2> lastAuthoredPoseRevision{0, 0};
    // Sources of authored future targets, shared by their existing control batch.
    // Acknowledging author revisions prevents replay; these guards persist until
    // the numerical publication that actually consumes or supersedes the target.
    std::array<PhysicsTargetSource, 2> physicsTargetSources;

    // ---- Physics interpolation cache ----
    glm::vec3 previousPhysicsPosition{0.0f};
    glm::quat previousPhysicsRotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 currentPhysicsPosition{0.0f};
    glm::quat currentPhysicsRotation{1.0f, 0.0f, 0.0f, 0.0f};
    bool hasPhysicsPose = false;
    bool wasSleeping = false;
};

// ============================================================================
// PhysicsECSStore — singleton managing both pools
// ============================================================================

class PhysicsECSStore
{
  public:
    using ColliderHandle = InxContiguousPool<ColliderECSData>::Handle;
    using RigidbodyHandle = InxContiguousPool<RigidbodyECSData>::Handle;
    using ActorHandle = PhysicsActorHandle;

    static PhysicsECSStore &Instance();

    // Register before publishing Scenes; retire only after their actors die.
    void RegisterRuntimeWorld(RuntimeWorld &world);
    void RetireRuntimeWorld(RuntimeWorld &world);
    /// Process-wide public body identity; never reused by a later World or Play session.
    [[nodiscard]] uint32_t AllocateBodyIdentity();
    void ValidateHierarchyWorld(const GameObject *root, const RuntimeWorld *destination) const;
    void PublishHierarchyOwnership(GameObject *root);

    // A Rigidbody owns its actor even when it has no collision shapes.
    // Collider-only objects own static actors.
    ActorHandle AcquireActor(GameObject *owner);
    /// Transfer a staged root's actor when its components are committed to an existing object.
    void AdoptActorOwner(GameObject *stagedOwner, GameObject *owner);
    void ReleaseActor(ActorHandle handle, Collider *collider);
    void AttachCollider(Collider *collider);
    void RefreshHierarchy(GameObject *root, const Rigidbody *excludedBody = nullptr);
    [[nodiscard]] static Rigidbody *FindAttachedRigidbody(GameObject *object, const Rigidbody *excludedBody = nullptr);
    [[nodiscard]] ActorHandle FindActor(const GameObject *owner) const;
    void AttachRigidbody(Rigidbody *body);
    void DetachRigidbody(Rigidbody *body);
    void RegisterActor(ActorHandle handle);
    void RefreshActor(ActorHandle handle);
    void SuspendActor(ActorHandle handle);
    void RestoreActor(ActorHandle handle);
    void SyncActorTransform(ActorHandle handle, float fixedDeltaTime,
                            std::vector<struct PhysicsBodyPoseUpdate> *staticPoseBatch);
    [[nodiscard]] bool IsValid(ActorHandle handle) const;
    PhysicsActorData &GetActor(ActorHandle handle);
    const PhysicsActorData &GetActor(ActorHandle handle) const;
    [[nodiscard]] size_t GetAliveActorCount() const
    {
        return m_actorPool.AliveCount();
    }
    template <typename Func> void ForEachAliveActor(Func &&func)
    {
        m_actorPool.ForEachAlive(std::forward<Func>(func));
    }

    // ---- Collider pool ----
    ColliderHandle AllocateCollider(Collider *owner);
    void ReleaseCollider(ColliderHandle handle);
    [[nodiscard]] bool IsValid(ColliderHandle handle) const;
    ColliderECSData &GetCollider(ColliderHandle handle);
    const ColliderECSData &GetCollider(ColliderHandle handle) const;
    [[nodiscard]] std::vector<ColliderHandle> GetAliveColliderHandles() const
    {
        return m_colliderPool.GetAliveHandles();
    }
    [[nodiscard]] size_t GetAliveColliderCount() const
    {
        return m_colliderPool.AliveCount();
    }

    /// Zero-allocation iteration over all alive colliders.
    /// @p func receives (ColliderECSData &data).
    template <typename Func> void ForEachAliveCollider(Func &&func)
    {
        m_colliderPool.ForEachAlive(std::forward<Func>(func));
    }

    template <typename Func> void ForEachAliveCollider(Func &&func) const
    {
        m_colliderPool.ForEachAlive(std::forward<Func>(func));
    }

    // ---- Dirty collider tracking (Unity-style deferred sync) ----

    /// Mark a collider as needing transform→physics sync before the next physics step.
    void MarkColliderDirty(ColliderHandle handle);

    /// Mark the single shared physics actor owned by a GameObject dirty.
    /// Avoids allocating/scanning a temporary component list from Transform observers.
    void MarkGameObjectDirty(GameObject *owner, const Transform *physicsSource = nullptr,
                             const Transform *authoredSource = nullptr, bool rotationAuthored = false);
    void MarkAuthoredHierarchyDirty(GameObject *root, bool rotationAuthored);

    /// Consume only this World's dirty prefix; other World scratch stays valid.
    const std::vector<ActorHandle> &ConsumeDirtyActors(const RuntimeWorld &world);
    [[nodiscard]] bool HasDirtyColliders(const RuntimeWorld &world) const noexcept;
    void MarkAllCollidersDirty(const RuntimeWorld &world);

    /// Shared author revision, independent of per-World queue consumption.
    void NotifyCollisionSceneChanged() noexcept;
    [[nodiscard]] uint64_t GetCollisionSceneRevision() const noexcept;

    // Pending intrusive Joint membership belongs to its actual published owner.
    void QueuePendingJoint(Joint &joint) noexcept;
    void CancelPendingJoint(Joint &joint) noexcept;
    void DrainPendingJoints(RuntimeWorld &world);
    void ClearPendingJoints(const RuntimeWorld &world) noexcept;
    [[nodiscard]] bool HasPendingJointDependencies(const RuntimeWorld &world) const noexcept;

    void QueueBodyCreation(ActorHandle handle);
    std::vector<ActorHandle> ConsumePendingBodyCreations(const RuntimeWorld &world);
    [[nodiscard]] bool HasPendingBodyCreations(const RuntimeWorld &world) const noexcept;

    // Explicit owner survives Scene=null retirement. No ambient-world fallback.
    void QueueBroadphaseAdd(RuntimeWorld &world, uint32_t bodyId);
    bool CancelBroadphaseAdd(const RuntimeWorld &world, uint32_t bodyId);
    std::vector<uint32_t> ConsumePendingBroadphaseAdds(const RuntimeWorld &world);
    [[nodiscard]] bool HasPendingBroadphaseAdds(const RuntimeWorld &world) const noexcept;
    void QueueBroadphaseRemove(RuntimeWorld &world, uint32_t bodyId);
    bool CancelBroadphaseRemove(const RuntimeWorld &world, uint32_t bodyId);
    std::vector<uint32_t> ConsumePendingBroadphaseRemoves(const RuntimeWorld &world);
    [[nodiscard]] bool HasPendingBroadphaseRemoves(const RuntimeWorld &world) const noexcept;

    /// Reserve author pools. Domain queues grow only for their real owner.
    void ReserveForBulkCreation(size_t count)
    {
        m_colliderPool.Reserve(m_colliderPool.Capacity() + count);
        m_actorPool.Reserve(m_actorPool.Capacity() + count);
    }

    void ClearWorldPendingQueues(const RuntimeWorld &world);
    /// Entire engine shutdown only, after all actors and callbacks retire.
    void ClearPendingQueues();

    // ---- Rigidbody pool ----
    RigidbodyHandle AllocateRigidbody(Rigidbody *owner);
    void ReleaseRigidbody(RigidbodyHandle handle);
    [[nodiscard]] bool IsValid(RigidbodyHandle handle) const;
    RigidbodyECSData &GetRigidbody(RigidbodyHandle handle);
    const RigidbodyECSData &GetRigidbody(RigidbodyHandle handle) const;
    [[nodiscard]] std::vector<RigidbodyHandle> GetAliveRigidbodyHandles() const
    {
        return m_rigidbodyPool.GetAliveHandles();
    }
    [[nodiscard]] size_t GetAliveRigidbodyCount(const RuntimeWorld &world) const noexcept;

    /// Zero-allocation iteration over all alive rigidbodies.
    template <typename Func> void ForEachAliveRigidbody(Func &&func)
    {
        m_rigidbodyPool.ForEachAlive(std::forward<Func>(func));
    }

    template <typename Func> void ForEachAliveRigidbody(Func &&func) const
    {
        m_rigidbodyPool.ForEachAlive(std::forward<Func>(func));
    }

  private:
    PhysicsECSStore() = default;
    void RetireActor(ActorHandle handle);
    void BindActorWorld(ActorHandle handle);
    void UpdateJointEndpointReadiness(PhysicsActorData &actor) noexcept;
    void MarkActorDirty(ActorHandle handle);
    ActorHandle BindRigidbodyActor(Rigidbody *body);
    void MigrateCollider(Collider *collider, const Rigidbody *excludedBody, std::vector<ActorHandle> &changed);
    void PublishChangedActors(std::vector<ActorHandle> &changed);

    InxContiguousPool<ColliderECSData> m_colliderPool;
    PhysicsActorPool m_actorPool;
    InxContiguousPool<RigidbodyECSData> m_rigidbodyPool;
    std::unordered_map<GameObject *, ActorHandle> m_actorByOwner;

    struct DomainAuthorQueues
    {
        RuntimeWorld *owner = nullptr;
        size_t aliveRigidbodyCount = 0;
        std::vector<ActorHandle> dirtyActors;
        std::vector<ActorHandle> dirtyScratch;
        std::vector<ActorHandle> pendingCreations;
        std::unordered_set<uint64_t> pendingCreationSet;
        std::vector<uint32_t> broadphaseAdds;
        std::unordered_set<uint32_t> broadphaseAddSet;
        std::vector<uint32_t> broadphaseRemoves;
        std::unordered_set<uint32_t> broadphaseRemoveSet;
        Joint *pendingJointHead = nullptr;
        Joint *pendingJointTail = nullptr;
        bool pendingJointsDirty = false;
        bool drainingPendingJoints = false;
    };
    DomainAuthorQueues &RequireQueues(const RuntimeWorld &world);
    DomainAuthorQueues *FindQueues(const RuntimeWorld &world) noexcept;
    const DomainAuthorQueues *FindQueues(const RuntimeWorld &world) const noexcept;
    std::unordered_map<uint64_t, DomainAuthorQueues> m_worldQueues;
    uint32_t m_nextBodyIdentity = 0;
    uint64_t m_collisionSceneRevision = 1;
};

} // namespace infernux
