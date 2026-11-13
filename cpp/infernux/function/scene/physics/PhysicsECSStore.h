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
class Rigidbody;
class Transform;

struct PhysicsActorData
{
    GameObject *owner = nullptr;
    Collider *primaryCollider = nullptr;
    Rigidbody *rigidbody = nullptr;
    uint32_t bodyId = 0xFFFFFFFF;
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

    // A Rigidbody owns its actor even when it has no collision shapes.
    // Collider-only objects own static actors.
    ActorHandle AcquireActor(GameObject *owner);
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

    /// Consume changed actors once, including actors without collision shapes.
    /// The returned reference remains valid until the next call.
    const std::vector<ActorHandle> &ConsumeDirtyActors();

    /// True if any collider has been marked dirty since last consume.
    [[nodiscard]] bool HasDirtyColliders() const
    {
        return !m_dirtyActorList.empty();
    }

    /// Mark all alive colliders dirty (used for force-sync scenarios).
    void MarkAllCollidersDirty();

    /// Monotonic source revision for renderer-owned collision mirrors. This is
    /// independent of ConsumeDirtyActors(), which remains owned by PhysX/Jolt
    /// synchronization and may only be consumed once.
    void NotifyCollisionSceneChanged() noexcept;
    [[nodiscard]] uint64_t GetCollisionSceneRevision() const noexcept;

    // ---- Pending body creation queue (deferred from Collider::Awake) ----

    /// Queue one actor for deferred Jolt body creation at the physics boundary.
    void QueueBodyCreation(ActorHandle handle);

    /// Consume pending body creation queue.  Returns handles (dead entries filtered).
    std::vector<ActorHandle> ConsumePendingBodyCreations();

    /// True if any colliders are waiting for body creation.
    [[nodiscard]] bool HasPendingBodyCreations() const
    {
        return !m_pendingBodyCreationSet.empty();
    }

    // ---- Pending broadphase queue (deferred body activation) ----

    /// Queue a body ID for batch broadphase addition at next flush.
    void QueueBroadphaseAdd(uint32_t bodyId);

    /// Consume each final add request once, in request order. Motion type is
    /// read from the body at publication, after all authored changes.
    std::vector<uint32_t> ConsumePendingBroadphaseAdds();

    /// Cancel an add that has not reached Jolt yet.
    bool CancelBroadphaseAdd(uint32_t bodyId);

    /// True if any bodies are waiting to be added to the broadphase.
    [[nodiscard]] bool HasPendingBroadphaseAdds() const
    {
        return !m_pendingBroadphaseAdds.empty();
    }

    /// Queue a body for removal at the next pre-physics flush. Gameplay
    /// callbacks must not mutate Jolt's broadphase directly.
    void QueueBroadphaseRemove(uint32_t bodyId);

    /// Cancel a removal when the Collider is re-enabled before the flush.
    bool CancelBroadphaseRemove(uint32_t bodyId);

    /// Consume pending broadphase removals.
    std::vector<uint32_t> ConsumePendingBroadphaseRemoves();

    [[nodiscard]] bool HasPendingBroadphaseRemoves() const
    {
        return !m_pendingBroadphaseRemoves.empty();
    }

    /// Pre-allocate internal pools and queues for @p count new colliders.
    void ReserveForBulkCreation(size_t count)
    {
        m_colliderPool.Reserve(m_colliderPool.Capacity() + count);
        m_actorPool.Reserve(m_actorPool.Capacity() + count);
        m_pendingBodyCreationList.reserve(m_pendingBodyCreationList.size() + count);
        m_pendingBodyCreationSet.reserve(m_pendingBodyCreationSet.size() + count);
        m_pendingBroadphaseAdds.reserve(m_pendingBroadphaseAdds.size() + count);
        m_pendingBroadphaseSet.reserve(m_pendingBroadphaseSet.size() + count);
        m_pendingBroadphaseRemoves.reserve(m_pendingBroadphaseRemoves.size() + count);
        m_pendingBroadphaseRemoveSet.reserve(m_pendingBroadphaseRemoveSet.size() + count);
    }

    /// Clear all pending work after all Worlds have retired at engine shutdown.
    /// Scene replacement must preserve live Worlds' commands. Generation-aware
    /// handles already distinguish newly allocated slots from retired owners.
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
    [[nodiscard]] size_t GetAliveRigidbodyCount() const
    {
        return m_rigidbodyPool.AliveCount();
    }

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
    void MarkActorDirty(ActorHandle handle);
    ActorHandle BindRigidbodyActor(Rigidbody *body);
    void MigrateCollider(Collider *collider, const Rigidbody *excludedBody, std::vector<ActorHandle> &changed);
    void PublishChangedActors(std::vector<ActorHandle> &changed);

    InxContiguousPool<ColliderECSData> m_colliderPool;
    PhysicsActorPool m_actorPool;
    InxContiguousPool<RigidbodyECSData> m_rigidbodyPool;
    std::unordered_map<GameObject *, ActorHandle> m_actorByOwner;

    // Dirty collider tracking — colliders whose Transform changed and need physics sync.
    std::vector<ActorHandle> m_dirtyActorList;
    std::vector<ActorHandle> m_dirtyActorScratch;
    uint64_t m_collisionSceneRevision = 1;

    // Pending actor creation, deduplicated by generation-aware handle.
    std::vector<ActorHandle> m_pendingBodyCreationList;
    std::unordered_set<uint64_t> m_pendingBodyCreationSet; // generation-aware handle dedup

    // Pending broadphase add queue — body IDs in authored request order.
    std::vector<uint32_t> m_pendingBroadphaseAdds;
    std::unordered_set<uint32_t> m_pendingBroadphaseSet; // deduplicate

    // Pending broadphase removals. Entries remain in the vector after a
    // cancellation and are filtered through the set during consume.
    std::vector<uint32_t> m_pendingBroadphaseRemoves;
    std::unordered_set<uint32_t> m_pendingBroadphaseRemoveSet;
};

} // namespace infernux
