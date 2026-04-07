#pragma once

/**
 * @file PhysicsWorld.h
 * @brief Singleton physics world backed by Jolt Physics.
 *
 * Manages Jolt initialisation/shutdown, stepping, body management,
 * and raycast queries. Integrated with SceneManager::FixedUpdate.
 */

#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <unordered_map>
#include <vector>

// Forward declaration
namespace infernux
{
class InxContactListener;
}

// Forward-declare Jolt types to avoid Jolt.h leak into every TU
namespace JPH
{
class PhysicsSystem;
class TempAllocatorImpl;
class JobSystemThreadPool;
class Body;
class BodyID;
class Shape;
} // namespace JPH

namespace infernux
{

class GameObject;
class Collider;
class Rigidbody;

/**
 * @brief Result of a physics raycast (Unity: RaycastHit).
 */
struct RaycastHit
{
    glm::vec3 point{0.0f};            ///< World-space hit point
    glm::vec3 normal{0.0f};           ///< Surface normal at hit
    float distance = 0.0f;            ///< Distance from ray origin
    uint32_t bodyId = 0xFFFFFFFF;     ///< Hit Jolt body id (index+sequence)
    GameObject *gameObject = nullptr; ///< Hit GameObject
    Collider *collider = nullptr;     ///< Hit Collider component
};

/**
 * @brief Singleton physics world wrapping Jolt Physics.
 */
class PhysicsWorld
{
  public:
    static PhysicsWorld &Instance();

    /// Initialise Jolt (call once after engine starts)
    void Initialize();

    /// Shut down Jolt (call on engine cleanup)
    void Shutdown();

    /// @return true if initialised
    [[nodiscard]] bool IsInitialized() const
    {
        return m_initialized;
    }

    /// Advance the simulation by one fixed step.
    void Step(float deltaTime);

    // ========================================================================
    // Body management (called by Collider components)
    // ========================================================================

    /// Register a collider's Jolt body. Returns Jolt body ID as uint32.
    uint32_t CreateBody(Collider *collider, bool isStatic, bool isTrigger);

    /// Remove a body by collider pointer.
    void DestroyBody(Collider *collider);

    /// Inform the physics world that a body has moved (kinematic / editor move).
    void SetBodyPosition(uint32_t bodyId, const glm::vec3 &pos, const glm::quat &rot);

    /// Notify that a body's shape or properties changed.
    void UpdateBodyShape(Collider *collider, const Collider *exclude = nullptr);

    /// Update a body's sensor (trigger) flag at runtime without recreating the body.
    void SetBodyIsSensor(uint32_t bodyId, bool isSensor);

    /// Clear cached contact-pair tracking for a body so the next physics step
    /// produces fresh Enter events.  Call after changing sensor flag at runtime.
    void InvalidateContactPairsForBody(uint32_t bodyId);

    /// Add an existing body to the broadphase (visible to raycasts/queries).
    void AddBodyToBroadphase(uint32_t bodyId, bool isStatic);

    /// Remove a body from the broadphase (body stays alive for re-adding later).
    void RemoveBodyFromBroadphase(uint32_t bodyId);

    // ========================================================================
    // Body dynamics — used by Rigidbody component
    // ========================================================================

    /// Switch a body between Static, Dynamic, Kinematic.
    /// motionType: 0 = Static, 1 = Kinematic, 2 = Dynamic.
    void SetBodyMotionType(uint32_t bodyId, int motionType);

    /// Update a body's user layer while preserving whether it is moving/static.
    void SetBodyGameLayer(uint32_t bodyId, int gameLayer);

    /// Set mass override via the body's MassProperties.
    void SetBodyMassProperties(uint32_t bodyId, float mass);

    /// Set linear / angular damping (drag).
    void SetBodyDamping(uint32_t bodyId, float linearDamping, float angularDamping);

    /// Override per-body gravity factor (0 = no gravity, 1 = normal).
    void SetBodyGravityFactor(uint32_t bodyId, float factor);

    /// Set per-body friction coefficient [0..1].
    void SetBodyFriction(uint32_t bodyId, float friction);

    /// Set per-body restitution (bounciness) [0..1].
    void SetBodyRestitution(uint32_t bodyId, float restitution);

    // ---- Velocity ----

    [[nodiscard]] glm::vec3 GetBodyLinearVelocity(uint32_t bodyId) const;
    void SetBodyLinearVelocity(uint32_t bodyId, const glm::vec3 &vel);

    [[nodiscard]] glm::vec3 GetBodyAngularVelocity(uint32_t bodyId) const;
    void SetBodyAngularVelocity(uint32_t bodyId, const glm::vec3 &vel);

    // ---- Forces ----

    void AddBodyForce(uint32_t bodyId, const glm::vec3 &force);
    void AddBodyImpulse(uint32_t bodyId, const glm::vec3 &impulse);
    void AddBodyTorque(uint32_t bodyId, const glm::vec3 &torque);
    void AddBodyAngularImpulse(uint32_t bodyId, const glm::vec3 &impulse);

    // ---- Forces at position ----

    void AddBodyForceAtPosition(uint32_t bodyId, const glm::vec3 &force, const glm::vec3 &point);
    void AddBodyImpulseAtPosition(uint32_t bodyId, const glm::vec3 &impulse, const glm::vec3 &point);

    // ---- Constraints / Motion quality ----

    /// Set allowed degrees-of-freedom (Jolt EAllowedDOFs bitmask). Recalculates mass for the new DOFs.
    void SetBodyAllowedDOFs(uint32_t bodyId, int allowedDOFs, float mass);

    /// Set motion quality: 0 = Discrete, 1 = LinearCast (continuous).
    void SetBodyMotionQuality(uint32_t bodyId, int quality);

    /// Set max angular velocity (rad/s).
    void SetBodyMaxAngularVelocity(uint32_t bodyId, float maxVel);

    /// Set max linear velocity (m/s).
    void SetBodyMaxLinearVelocity(uint32_t bodyId, float maxVel);

    // ---- Kinematic move ----

    /// Move a kinematic body towards target position/rotation over deltaTime.
    void MoveBodyKinematic(uint32_t bodyId, const glm::vec3 &targetPos, const glm::quat &targetRot, float deltaTime);

    // ---- Sleep ----

    [[nodiscard]] bool IsBodySleeping(uint32_t bodyId) const;
    [[nodiscard]] bool IsBodySensor(uint32_t bodyId) const;
    void ActivateBody(uint32_t bodyId);
    void DeactivateBody(uint32_t bodyId);

    /// Wake all dynamic bodies whose AABBs overlap the given world-space box.
    /// Used after moving a static collider to wake sleeping bodies that were resting on it.
    void ActivateBodiesInAABB(const glm::vec3 &min, const glm::vec3 &max);

    /// Wake dynamic bodies overlapping a specific body's world-space AABB.
    /// Convenience wrapper for use after moving a static collider.
    void WakeBodiesTouchingStatic(uint32_t bodyId);

    // ---- Read-back (physics → transform) ----

    [[nodiscard]] glm::vec3 GetBodyPosition(uint32_t bodyId) const;
    [[nodiscard]] glm::quat GetBodyRotation(uint32_t bodyId) const;
    [[nodiscard]] glm::vec3 GetBodyCenterOfMassPosition(uint32_t bodyId) const;

    /// Get world-space inertia tensor (3x3) for dynamic bodies.
    /// Returns identity for invalid/static bodies.
    [[nodiscard]] glm::mat3 GetBodyWorldSpaceInertiaTensor(uint32_t bodyId) const;

    // ========================================================================
    // Raycast API (Unity: Physics.Raycast)
    // ========================================================================

    /// Cast a ray and return the closest hit.  Returns true if hit.
    bool Raycast(const glm::vec3 &origin, const glm::vec3 &direction, float maxDistance, RaycastHit &outHit,
                 uint32_t layerMask = (0xFFFFFFFFu & ~(1u << 2)), bool queryTriggers = true) const;

    /// Cast a ray and return all hits.
    std::vector<RaycastHit> RaycastAll(const glm::vec3 &origin, const glm::vec3 &direction, float maxDistance,
                                       uint32_t layerMask = (0xFFFFFFFFu & ~(1u << 2)),
                                       bool queryTriggers = true) const;

    // ========================================================================
    // Overlap queries (Unity: Physics.OverlapSphere / OverlapBox)
    // ========================================================================

    /// Find all Colliders within a sphere. Returns list of Collider*.
    std::vector<Collider *> OverlapSphere(const glm::vec3 &center, float radius,
                                          uint32_t layerMask = (0xFFFFFFFFu & ~(1u << 2)),
                                          bool queryTriggers = true) const;

    /// Find all Colliders within an axis-aligned box.
    std::vector<Collider *> OverlapBox(const glm::vec3 &center, const glm::vec3 &halfExtents,
                                       uint32_t layerMask = (0xFFFFFFFFu & ~(1u << 2)),
                                       bool queryTriggers = true) const;

    // ========================================================================
    // Shape cast queries (Unity: Physics.SphereCast / BoxCast)
    // ========================================================================

    /// Cast a sphere along a direction. Returns closest RaycastHit or empty.
    bool SphereCast(const glm::vec3 &origin, float radius, const glm::vec3 &direction, float maxDistance,
                    RaycastHit &outHit, uint32_t layerMask = (0xFFFFFFFFu & ~(1u << 2)),
                    bool queryTriggers = true) const;

    /// Cast a box along a direction. Returns closest RaycastHit or empty.
    bool BoxCast(const glm::vec3 &center, const glm::vec3 &halfExtents, const glm::vec3 &direction, float maxDistance,
                 RaycastHit &outHit, uint32_t layerMask = (0xFFFFFFFFu & ~(1u << 2)), bool queryTriggers = true) const;

    // ========================================================================
    // Lookup
    // ========================================================================

    /// Find the Collider* that owns a given body ID.
    Collider *FindColliderByBodyId(uint32_t bodyId) const;

    /// Resolve a specific subshape hit/contact back to the owning Collider.
    Collider *ResolveColliderForSubShape(uint32_t bodyId, uint32_t subShapeIdValue) const;

    /// Rebind a body lookup entry to another collider on the same body.
    void RebindBodyCollider(uint32_t bodyId, Collider *collider);

    /// Ensure all Colliders in the given scene have registered bodies
    /// and their transforms are up to date. Call before editor-mode raycasts.
    void EnsureSceneBodiesRegistered(class Scene *scene);

    /// Rebuild the broad-phase tree so raycasts find newly added static bodies.
    void OptimizeBroadPhase();

    /// @brief Dispatch buffered contact events to Component callbacks.
    ///        Call once per fixed step, immediately after Step().
    void DispatchContactEvents();

    /// Get the Jolt PhysicsSystem (for advanced usage). May be nullptr.
    JPH::PhysicsSystem *GetJoltSystem() const
    {
        return m_physicsSystem.get();
    }

  private:
    PhysicsWorld() = default;
    ~PhysicsWorld();
    PhysicsWorld(const PhysicsWorld &) = delete;
    PhysicsWorld &operator=(const PhysicsWorld &) = delete;

    /// @brief Shared overlap implementation for OverlapSphere/OverlapBox.
    std::vector<Collider *> OverlapShapeImpl(const JPH::Shape &shape, const glm::vec3 &center, uint32_t layerMask,
                                             bool queryTriggers) const;

    /// @brief Shared shape cast implementation for SphereCast/BoxCast.
    bool ShapeCastImpl(const JPH::Shape &shape, const glm::vec3 &origin, const glm::vec3 &direction, float maxDistance,
                       RaycastHit &outHit, uint32_t layerMask, bool queryTriggers) const;

    bool m_initialized = false;

    std::unique_ptr<JPH::TempAllocatorImpl> m_tempAllocator;
    std::unique_ptr<JPH::JobSystemThreadPool> m_jobSystem;
    std::unique_ptr<JPH::PhysicsSystem> m_physicsSystem;

    // Layer interfaces (must outlive PhysicsSystem)
    struct LayerInterfaces;
    std::unique_ptr<LayerInterfaces> m_layers;

    // Mapping: Jolt body index → Collider*
    std::unordered_map<uint32_t, Collider *> m_bodyToCollider;

    // Contact listener for collision/trigger callbacks
    std::unique_ptr<InxContactListener> m_contactListener;
};

} // namespace infernux
