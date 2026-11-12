#pragma once

/**
 * @file PhysicsContactListener.h
 * @brief Jolt ContactListener implementation that collects collision/trigger
 *        events for dispatch to Unity-style component callbacks.
 *
 * Events are buffered during PhysicsWorld::Step() and flushed by
 * PhysicsWorld::DispatchContactEvents() immediately after.
 */

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Collision/ContactListener.h>

#include "PhysicsContactTypes.h"

#include <glm/glm.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward-declare Jolt BodyInterface for ResolveEvents parameter
namespace JPH
{
class BodyInterface;
}

namespace infernux
{

// ============================================================================
// InxContactListener — Jolt callback implementation
// ============================================================================

class InxContactListener : public JPH::ContactListener
{
  public:
    /// Publish only event classes requested by live components. Contact
    /// material resolution remains active regardless of this mask.
    void SetEventInterestMask(uint8_t mask);

    /// Reset per-step buffers. Call before PhysicsWorld::Step().
    void PreStep();

    /// Process raw contact events through pair tracking to produce final
    /// events that suppress sleep-related spurious Enter/Exit.
    /// Call after Step(), before DispatchContactEvents().
    void ResolveEvents(JPH::BodyInterface &bodyInterface);

    /// Full reset — clears all cached pair state (call on scene change / shutdown).
    void ClearAll();

    /// Remove all tracked contact pairs that involve @p bodyId.
    /// Call when a body's sensor flag changes at runtime so that the next
    /// step produces fresh Enter events instead of suppressing them as
    /// wake-from-sleep duplicates.
    void InvalidatePairsForBody(uint32_t bodyId);

    /// Drop episode contact history without changing collision policy.
    /// Called on the owner outside Step/dispatch, after Jolt cache invalidation.
    void ResetBodies(const std::unordered_set<uint32_t> &bodyIds);

    /// Reference-count a body pair that must not produce contacts. Multiple
    /// joints may independently own the same pair.
    void SetBodyPairIgnored(uint32_t bodyIdA, uint32_t bodyIdB, bool ignored);
    [[nodiscard]] bool GetBodyPairIgnored(uint32_t bodyIdA, uint32_t bodyIdB) const;
    void RemoveIgnoredPairsForBody(uint32_t bodyId);

    /// Set the idempotent game-level policy for one exact Collider pair.
    /// Unlike body-pair ownership used by joints, this preserves compound
    /// subshape identity and is not reference counted.
    void SetColliderPairIgnored(uint64_t componentIdA, uint64_t componentIdB, bool ignored);
    [[nodiscard]] bool GetColliderPairIgnored(uint64_t componentIdA, uint64_t componentIdB) const;
    void RemoveIgnoredPairsForCollider(uint64_t componentId);

    /// Get resolved events for dispatch.
    const std::vector<ContactEvent> &GetEvents() const
    {
        return m_events;
    }

    // -- JPH::ContactListener overrides --

    JPH::ValidateResult OnContactValidate(const JPH::Body &inBody1, const JPH::Body &inBody2,
                                          JPH::RVec3Arg inBaseOffset,
                                          const JPH::CollideShapeResult &inCollisionResult) override;

    void OnContactAdded(const JPH::Body &inBody1, const JPH::Body &inBody2, const JPH::ContactManifold &inManifold,
                        JPH::ContactSettings &ioSettings) override;

    void OnContactPersisted(const JPH::Body &inBody1, const JPH::Body &inBody2, const JPH::ContactManifold &inManifold,
                            JPH::ContactSettings &ioSettings) override;

    void OnContactRemoved(const JPH::SubShapeIDPair &inSubShapePair) override;

  private:
    static constexpr size_t kEventShardCount = 16;

    void PushEvent(ContactEventType type, const JPH::Body &bodyA, const JPH::Body &bodyB,
                   const JPH::ContactManifold *manifold);
    void PushRawEvent(ContactEvent event);
    void MergeRawEvents();
    [[nodiscard]] bool Wants(ContactEventType type) const;

    struct ContactPairKey
    {
        uint32_t bodyA;
        uint32_t subShapeA;
        uint32_t bodyB;
        uint32_t subShapeB;

        bool operator==(const ContactPairKey &other) const
        {
            return bodyA == other.bodyA && subShapeA == other.subShapeA && bodyB == other.bodyB &&
                   subShapeB == other.subShapeB;
        }
    };

    struct ContactPairKeyHash
    {
        size_t operator()(const ContactPairKey &key) const;
    };

    static ContactPairKey MakePairKey(const ContactEvent &event);

    struct ColliderPairKey
    {
        uint64_t first;
        uint64_t second;

        bool operator==(const ColliderPairKey &other) const
        {
            return first == other.first && second == other.second;
        }
    };

    struct ColliderPairKeyHash
    {
        size_t operator()(const ColliderPairKey &key) const;
    };

    static ColliderPairKey MakeColliderPairKey(uint64_t componentIdA, uint64_t componentIdB);

    struct alignas(64) EventShard
    {
        std::mutex mutex;
        std::vector<ContactEvent> events;
    };

    std::array<EventShard, kEventShardCount> m_eventShards;
    std::vector<ContactEvent> m_rawEvents; ///< Main-thread aggregate built after Step
    std::vector<ContactEvent> m_events;    ///< Resolved events for dispatch

    /// Persistent pair tracking across physics steps.
    struct PairState
    {
        bool touchedThisStep = false;
        bool sleeping = false; ///< Exit was suppressed because a body went to sleep
        ContactEvent lastEvent{};
    };
    std::unordered_map<ContactPairKey, PairState, ContactPairKeyHash> m_contactPairs;
    uint8_t m_eventInterestMask = 0;

    mutable std::shared_mutex m_ignoredPairMutex;
    std::unordered_map<uint64_t, uint32_t> m_ignoredBodyPairs;
    std::atomic_bool m_hasIgnoredBodyPairs{false};
    std::unordered_set<ColliderPairKey, ColliderPairKeyHash> m_ignoredColliderPairs;
    std::atomic_bool m_hasIgnoredColliderPairs{false};

    // m_contactPairs and m_events are main-thread only. Jolt worker callbacks
    // write exclusively to m_eventShards.
};

} // namespace infernux
