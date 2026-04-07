/**
 * @file Rigidbody.cpp
 * @brief Rigidbody component — drives dynamic/kinematic physics simulation.
 *
 * Lifecycle:
 *   Awake  → (Collider has already registered its body as static)
 *   OnEnable → Tell sibling Colliders to switch body to Dynamic/Kinematic,
 *              apply mass/drag/gravity settings.
 *   OnDisable → Tell sibling Colliders to switch body back to Static.
 *   OnDestroy → Same as OnDisable.
 *
 * The Jolt body is owned by the Collider, not the Rigidbody. Rigidbody
 * merely configures the body's motion type and dynamics properties.
 */

#include "Rigidbody.h"

#include "Collider.h"
#include "ComponentFactory.h"
#include "GameObject.h"
#include "SceneManager.h"
#include "Transform.h"
#include "physics/PhysicsECSStore.h"
#include "physics/PhysicsWorld.h"

#include <core/log/InxLog.h>

#include <algorithm>
#include <nlohmann/json.hpp>
#include <unordered_set>

namespace infernux
{

namespace
{

static std::vector<uint32_t> CollectUniqueBodyIds(GameObject *go)
{
    std::vector<uint32_t> result;
    if (!go) {
        return result;
    }

    std::unordered_set<uint32_t> seen;
    auto colliders = go->GetComponents<Collider>();
    result.reserve(colliders.size());
    for (auto *col : colliders) {
        if (!col) {
            continue;
        }

        const uint32_t bodyId = col->GetBodyId();
        if (bodyId == 0xFFFFFFFF || !seen.insert(bodyId).second) {
            continue;
        }
        result.push_back(bodyId);
    }
    return result;
}

static Collider *GetPrimaryBodyCollider(GameObject *go)
{
    if (!go) {
        return nullptr;
    }

    std::unordered_set<uint32_t> seen;
    auto colliders = go->GetComponents<Collider>();
    for (auto *col : colliders) {
        if (!col) {
            continue;
        }

        const uint32_t bodyId = col->GetBodyId();
        if (bodyId != 0xFFFFFFFF && seen.insert(bodyId).second) {
            return col;
        }
    }
    return nullptr;
}

/// Returns the first valid body ID for this game object, or 0xFFFFFFFF.
static uint32_t GetPrimaryBodyId(GameObject *go)
{
    auto *col = GetPrimaryBodyCollider(go);
    return (col && col->GetBodyId() != 0xFFFFFFFF) ? col->GetBodyId() : 0xFFFFFFFF;
}

static int MapCollisionDetectionModeToMotionQuality(int mode, bool isKinematic)
{
    switch (mode) {
    case static_cast<int>(CollisionDetectionMode::Continuous):
        // Unity-style expectation: basic Continuous is primarily aimed at
        // simulation-driven rigidbodies. For kinematic bodies the closer
        // default is speculative contacts rather than sweep CCD.
        return isKinematic ? 0 : 1;
    case static_cast<int>(CollisionDetectionMode::ContinuousDynamic):
        // Explicit request for the strongest sweep-based CCD mode.
        return 1;
    case static_cast<int>(CollisionDetectionMode::Discrete):
        return 0;
    case static_cast<int>(CollisionDetectionMode::ContinuousSpeculative):
        INXLOG_WARN("Rigidbody: ContinuousSpeculative is not fully implemented — "
                    "falling back to Discrete motion quality. "
                    "Use Continuous or ContinuousDynamic for sweep-based CCD.");
        return 0;
    default:
        return 0;
    }
}

} // namespace

INFERNUX_REGISTER_COMPONENT("Rigidbody", Rigidbody)

// ============================================================================
// Shared helpers
// ============================================================================

template <typename Fn> void Rigidbody::ForEachBody(Fn &&fn)
{
    GameObject *go = nullptr;
    auto *pw = GetActivePhysicsWorld(go);
    if (!pw)
        return;
    for (uint32_t bodyId : CollectUniqueBodyIds(go))
        fn(*pw, bodyId);
}

void Rigidbody::TeleportBodies(PhysicsWorld &pw, GameObject *go, const glm::vec3 &pos, const glm::quat &rot)
{
    auto colliders = go->GetComponents<Collider>();
    for (auto *col : colliders) {
        if (col && col->GetBodyId() != 0xFFFFFFFF)
            col->SetLastSyncedTransform(pos, rot);
    }
    for (uint32_t bodyId : CollectUniqueBodyIds(go)) {
        pw.SetBodyPosition(bodyId, pos, rot);
        pw.ActivateBody(bodyId);
        pw.SetBodyLinearVelocity(bodyId, glm::vec3(0.0f));
        pw.SetBodyAngularVelocity(bodyId, glm::vec3(0.0f));
    }
    auto &d = DataMut();
    d.previousPhysicsPosition = pos;
    d.previousPhysicsRotation = rot;
    d.currentPhysicsPosition = pos;
    d.currentPhysicsRotation = rot;
    d.hasPhysicsPose = true;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

Rigidbody::Rigidbody()
{
    m_ecsHandle = PhysicsECSStore::Instance().AllocateRigidbody(this);
}

Rigidbody::~Rigidbody()
{
    // Safety-net unregister (idempotent — no-op if already done by OnDisable).
    // NOTE: Do NOT call GetComponents<Collider>() here — during
    // ~GameObject → m_components.clear(), sibling unique_ptrs may already be
    // destroyed, so iterating the vector and calling dynamic_cast on dangling
    // pointers is undefined behaviour.  OnDisable() already clears
    // cachedRigidbody on every sibling Collider before we reach this point.

    // Release pool slot
    PhysicsECSStore::Instance().ReleaseRigidbody(m_ecsHandle);
}

// ============================================================================
// Lifecycle
// ============================================================================

void Rigidbody::Awake()
{
    // Nothing here — Collider::Awake() creates the body as static.
    // OnEnable (called right after Awake) will switch it to dynamic.
}

void Rigidbody::OnEnable()
{

    // Cache this Rigidbody pointer on all sibling Colliders
    if (auto *go = GetGameObject()) {
        auto colliders = go->GetComponents<Collider>();
        for (auto *col : colliders)
            if (col)
                col->SetCachedRigidbody(this);
    }

    NotifyCollidersBodyTypeChanged();
}

void Rigidbody::OnDisable()
{

    // Clear cached Rigidbody pointer on sibling Colliders
    auto *go = GetGameObject();
    if (!go)
        return;

    auto colliders = go->GetComponents<Collider>();
    for (auto *col : colliders)
        if (col)
            col->SetCachedRigidbody(nullptr);

    auto *pw = &PhysicsWorld::Instance();
    if (!pw->IsInitialized())
        return;

    // Rebuild shapes — MeshCollider may switch back to MeshShape now
    // that there is no dynamic Rigidbody.  Must happen before setting
    // motionType to Static so the shape matches the new body type.
    for (auto *col : colliders) {
        if (col && col->IsEnabled() && col->GetBodyId() != 0xFFFFFFFF) {
            pw->UpdateBodyShape(col);
            break; // shared body — one rebuild is enough
        }
    }

    for (uint32_t bodyId : CollectUniqueBodyIds(go)) {
        pw->SetBodyMotionType(bodyId, 0); // 0 = Static
    }
}

void Rigidbody::OnDestroy()
{
    // Ensure bodies revert to static before Rigidbody goes away.
    // Use CallOnDisable() (not OnDisable()) so the m_wasEnabled guard
    // prevents double-execution when ~GameObject calls OnDisable + OnDestroy.
    CallOnDisable();
}

// ============================================================================
// Property setters
// ============================================================================

void Rigidbody::SetMass(float mass)
{
    auto &d = DataMut();
    d.mass = (mass < 0.001f) ? 0.001f : mass;
    ForEachBody([&](PhysicsWorld &pw, uint32_t id) { pw.SetBodyMassProperties(id, d.mass); });
}

void Rigidbody::SetDrag(float drag)
{
    DataMut().drag = (drag < 0.001f) ? 0.001f : drag;
    ApplyDragSettings();
}

void Rigidbody::SetAngularDrag(float drag)
{
    DataMut().angularDrag = (drag < 0.001f) ? 0.001f : drag;
    ApplyDragSettings();
}

void Rigidbody::SetUseGravity(bool use)
{
    auto &d = DataMut();
    if (d.useGravity == use)
        return;
    d.useGravity = use;
    float factor = d.useGravity ? 1.0f : 0.0f;
    ForEachBody([&](PhysicsWorld &pw, uint32_t id) { pw.SetBodyGravityFactor(id, factor); });
}

void Rigidbody::SetIsKinematic(bool kinematic)
{
    auto &d = DataMut();
    if (d.isKinematic == kinematic)
        return;
    d.isKinematic = kinematic;
    NotifyCollidersBodyTypeChanged();
}

void Rigidbody::SetConstraints(int constraints)
{
    auto &d = DataMut();
    if (d.constraints == constraints)
        return;
    d.constraints = constraints;
    ApplyConstraints();
}

void Rigidbody::SetFreezeRotation(bool freeze)
{
    int rotBits = static_cast<int>(RigidbodyConstraints::FreezeRotation);
    int newConstraints = freeze ? (Data().constraints | rotBits) : (Data().constraints & ~rotBits);
    SetConstraints(newConstraints);
}

void Rigidbody::SetCollisionDetectionMode(int mode)
{
    auto &d = DataMut();
    if (d.collisionDetectionMode == mode)
        return;
    d.collisionDetectionMode = std::clamp(mode, 0, 3);
    ApplyMotionQuality();
}

void Rigidbody::SetInterpolation(int mode)
{
    auto &d = DataMut();
    d.interpolation = std::clamp(mode, 0, 1);
}

void Rigidbody::SetMaxAngularVelocity(float vel)
{
    DataMut().maxAngularVelocity = (vel < 0.0f) ? 0.0f : vel;
    ApplyVelocityLimits();
}

void Rigidbody::SetMaxLinearVelocity(float vel)
{
    DataMut().maxLinearVelocity = (vel < 0.0f) ? 0.0f : vel;
    ApplyVelocityLimits();
}

// ============================================================================
// Velocity
// ============================================================================

glm::vec3 Rigidbody::GetVelocity() const
{
    uint32_t bid = GetPrimaryBodyId(GetGameObject());
    return (bid != 0xFFFFFFFF) ? PhysicsWorld::Instance().GetBodyLinearVelocity(bid) : glm::vec3(0.0f);
}

void Rigidbody::SetVelocity(const glm::vec3 &vel)
{
    auto *go = GetGameObject();
    if (!go)
        return;

    auto &pw = PhysicsWorld::Instance();
    for (uint32_t bodyId : CollectUniqueBodyIds(go)) {
        pw.SetBodyLinearVelocity(bodyId, vel);
    }
}

glm::vec3 Rigidbody::GetAngularVelocity() const
{
    uint32_t bid = GetPrimaryBodyId(GetGameObject());
    return (bid != 0xFFFFFFFF) ? PhysicsWorld::Instance().GetBodyAngularVelocity(bid) : glm::vec3(0.0f);
}

void Rigidbody::SetAngularVelocity(const glm::vec3 &vel)
{
    auto *go = GetGameObject();
    if (!go)
        return;

    auto &pw = PhysicsWorld::Instance();
    for (uint32_t bodyId : CollectUniqueBodyIds(go)) {
        pw.SetBodyAngularVelocity(bodyId, vel);
    }
}

// ============================================================================
// Forces
// ============================================================================

void Rigidbody::AddForce(const glm::vec3 &force, ForceMode mode)
{
    auto *go = GetGameObject();
    if (!go)
        return;

    auto &pw = PhysicsWorld::Instance();
    for (uint32_t bid : CollectUniqueBodyIds(go)) {
        switch (mode) {
        case ForceMode::Force:
            pw.AddBodyForce(bid, force);
            break;
        case ForceMode::Acceleration:
            // F = m * a → convert acceleration to force
            pw.AddBodyForce(bid, force * Data().mass);
            break;
        case ForceMode::Impulse:
            pw.AddBodyImpulse(bid, force);
            break;
        case ForceMode::VelocityChange:
            // impulse = m * dv → convert velocity change to impulse
            pw.AddBodyImpulse(bid, force * Data().mass);
            break;
        }
    }
}

void Rigidbody::AddTorque(const glm::vec3 &torque, ForceMode mode)
{
    auto *go = GetGameObject();
    if (!go)
        return;

    auto &pw = PhysicsWorld::Instance();
    for (uint32_t bid : CollectUniqueBodyIds(go)) {
        switch (mode) {
        case ForceMode::Force:
            pw.AddBodyTorque(bid, torque);
            break;
        case ForceMode::Acceleration: {
            // Unity semantics: angular acceleration is mass/inertia independent.
            // Jolt AddTorque expects torque, so convert alpha -> tau via inertia tensor.
            const glm::mat3 inertia = pw.GetBodyWorldSpaceInertiaTensor(bid);
            pw.AddBodyTorque(bid, inertia * torque);
            break;
        }
        case ForceMode::Impulse:
            pw.AddBodyAngularImpulse(bid, torque);
            break;
        case ForceMode::VelocityChange: {
            // Unity semantics: delta angular velocity is mass/inertia independent.
            // Jolt AddAngularImpulse expects angular impulse, so convert dω -> L.
            const glm::mat3 inertia = pw.GetBodyWorldSpaceInertiaTensor(bid);
            pw.AddBodyAngularImpulse(bid, inertia * torque);
            break;
        }
        }
    }
}

// ============================================================================
// AddForceAtPosition
// ============================================================================

void Rigidbody::AddForceAtPosition(const glm::vec3 &force, const glm::vec3 &position, ForceMode mode)
{
    auto *go = GetGameObject();
    if (!go)
        return;

    auto &pw = PhysicsWorld::Instance();
    for (uint32_t bid : CollectUniqueBodyIds(go)) {
        switch (mode) {
        case ForceMode::Force:
            pw.AddBodyForceAtPosition(bid, force, position);
            break;
        case ForceMode::Acceleration:
            pw.AddBodyForceAtPosition(bid, force * Data().mass, position);
            break;
        case ForceMode::Impulse:
            pw.AddBodyImpulseAtPosition(bid, force, position);
            break;
        case ForceMode::VelocityChange:
            pw.AddBodyImpulseAtPosition(bid, force * Data().mass, position);
            break;
        }
    }
}

// ============================================================================
// Kinematic movement
// ============================================================================

void Rigidbody::MovePosition(const glm::vec3 &position)
{
    if (!Data().isKinematic)
        return;

    GameObject *go = nullptr;
    auto *pw = GetActivePhysicsWorld(go);
    if (!pw)
        return;

    // Use current fixed timestep from SceneManager.
    float dt = SceneManager::Instance().GetFixedTimeStep();

    for (uint32_t bodyId : CollectUniqueBodyIds(go)) {
        glm::quat rot = pw->GetBodyRotation(bodyId);
        pw->MoveBodyKinematic(bodyId, position, rot, dt);
    }

    // Also update Transform for consistency
    if (auto *tf = go->GetTransform())
        tf->SetPosition(position);
}

void Rigidbody::MoveRotation(const glm::quat &rotation)
{
    if (!Data().isKinematic)
        return;

    GameObject *go = nullptr;
    auto *pw = GetActivePhysicsWorld(go);
    if (!pw)
        return;

    float dt = SceneManager::Instance().GetFixedTimeStep();

    Transform *tf = go->GetTransform();
    glm::vec3 pos = tf ? tf->GetPosition() : glm::vec3(0.0f);

    for (uint32_t bodyId : CollectUniqueBodyIds(go)) {
        pw->MoveBodyKinematic(bodyId, pos, rotation, dt);
    }

    if (tf)
        tf->SetWorldRotation(rotation);
}

// ============================================================================
// Read-only world info
// ============================================================================

glm::vec3 Rigidbody::GetWorldCenterOfMass() const
{
    uint32_t bid = GetPrimaryBodyId(GetGameObject());
    return (bid != 0xFFFFFFFF) ? PhysicsWorld::Instance().GetBodyCenterOfMassPosition(bid) : glm::vec3(0.0f);
}

glm::vec3 Rigidbody::GetPosition() const
{
    auto *go = GetGameObject();
    if (!go)
        return glm::vec3(0.0f);

    uint32_t bid = GetPrimaryBodyId(go);
    if (bid == 0xFFFFFFFF) {
        if (auto *tf = go->GetTransform())
            return tf->GetPosition();
        return glm::vec3(0.0f);
    }

    return PhysicsWorld::Instance().GetBodyPosition(bid);
}

glm::quat Rigidbody::GetRotation() const
{
    auto *go = GetGameObject();
    if (!go)
        return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

    uint32_t bid = GetPrimaryBodyId(go);
    if (bid == 0xFFFFFFFF) {
        if (auto *tf = go->GetTransform())
            return tf->GetWorldRotation();
        return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    }

    return PhysicsWorld::Instance().GetBodyRotation(bid);
}

// ============================================================================
// Sleep
// ============================================================================

bool Rigidbody::IsSleeping() const
{
    uint32_t bid = GetPrimaryBodyId(GetGameObject());
    return (bid != 0xFFFFFFFF) ? PhysicsWorld::Instance().IsBodySleeping(bid) : true;
}

void Rigidbody::WakeUp()
{
    auto *go = GetGameObject();
    if (!go)
        return;

    auto &pw = PhysicsWorld::Instance();
    for (uint32_t bodyId : CollectUniqueBodyIds(go)) {
        pw.ActivateBody(bodyId);
    }
}

void Rigidbody::Sleep()
{
    auto *go = GetGameObject();
    if (!go)
        return;

    auto &pw = PhysicsWorld::Instance();
    for (uint32_t bodyId : CollectUniqueBodyIds(go)) {
        pw.DeactivateBody(bodyId);
    }
}

// ============================================================================
// Physics → Transform writeback (called after Jolt step)
// ============================================================================

void Rigidbody::SyncPhysicsToTransform()
{
    auto &d = DataMut();
    if (d.isKinematic)
        return; // Kinematic bodies are driven by Transform, not physics

    auto *go = GetGameObject();
    if (!go)
        return;

    uint32_t bid = GetPrimaryBodyId(go);
    if (bid == 0xFFFFFFFF)
        return;

    auto &pw = PhysicsWorld::Instance();

    glm::vec3 bodyPos = pw.GetBodyPosition(bid);
    glm::quat bodyRot = glm::normalize(pw.GetBodyRotation(bid));

    Transform *tf = go->GetTransform();
    if (!tf)
        return;

    const bool firstPose = !d.hasPhysicsPose;
    if (firstPose) {
        d.previousPhysicsPosition = bodyPos;
        d.previousPhysicsRotation = bodyRot;
    } else {
        d.previousPhysicsPosition = d.currentPhysicsPosition;
        d.previousPhysicsRotation = d.currentPhysicsRotation;
    }
    d.currentPhysicsPosition = bodyPos;
    d.currentPhysicsRotation = bodyRot;
    d.hasPhysicsPose = true;

    const bool rotFrozen = (d.constraints & static_cast<int>(RigidbodyConstraints::FreezeRotation)) ==
                           static_cast<int>(RigidbodyConstraints::FreezeRotation);

    if (d.interpolation == static_cast<int>(RigidbodyInterpolation::None) || firstPose) {
        tf->SetPosition(bodyPos);
        if (!rotFrozen) {
            tf->SetWorldRotation(bodyRot);
        }
        d.lastSyncedPosition = bodyPos;
        // Always read back the reconstructed rotation from the Transform so
        // that the cached value matches what SyncExternalMovesToPhysics() will
        // later read via GetWorldRotation() (avoids float round-trip mismatch).
        d.lastSyncedRotation = tf->GetWorldRotation();
        d.hasSyncedOnce = true;
    }

    // Also update the cached transform on ALL sibling colliders so that
    // SyncTransformToPhysics() won't see a spurious delta next step.
    auto colliders = go->GetComponents<Collider>();
    for (auto *c : colliders) {
        if (c && c->GetBodyId() != 0xFFFFFFFF) {
            const glm::vec3 cachePos = (d.interpolation == static_cast<int>(RigidbodyInterpolation::None) || firstPose)
                                           ? bodyPos
                                           : d.lastSyncedPosition;
            // Use the same reconstructed rotation we cached above.
            const glm::quat cacheRot = d.lastSyncedRotation;
            c->SetLastSyncedTransform(cachePos, cacheRot);
        }
    }
}

void Rigidbody::ApplyInterpolatedTransform(float alpha)
{
    auto &d = DataMut();
    if (d.isKinematic || !d.hasPhysicsPose)
        return;

    auto *go = GetGameObject();
    if (!go)
        return;

    Transform *tf = go->GetTransform();
    if (!tf)
        return;

    glm::vec3 presentedPos = d.currentPhysicsPosition;
    glm::quat presentedRot = glm::normalize(d.currentPhysicsRotation);

    if (d.interpolation == static_cast<int>(RigidbodyInterpolation::Interpolate)) {
        float t = std::clamp(alpha, 0.0f, 1.0f);
        presentedPos = glm::mix(d.previousPhysicsPosition, d.currentPhysicsPosition, t);
        presentedRot = glm::normalize(
            glm::slerp(glm::normalize(d.previousPhysicsRotation), glm::normalize(d.currentPhysicsRotation), t));
    }

    const bool rotFrozen = (d.constraints & static_cast<int>(RigidbodyConstraints::FreezeRotation)) ==
                           static_cast<int>(RigidbodyConstraints::FreezeRotation);

    tf->SetPosition(presentedPos);
    if (!rotFrozen) {
        tf->SetWorldRotation(presentedRot);
    }

    d.lastSyncedPosition = presentedPos;
    // Read back reconstructed rotation so the cache matches what
    // SyncExternalMovesToPhysics() will see via GetWorldRotation().
    d.lastSyncedRotation = tf->GetWorldRotation();
    d.hasSyncedOnce = true;

    auto colliders = go->GetComponents<Collider>();
    for (auto *c : colliders) {
        if (c && c->GetBodyId() != 0xFFFFFFFF) {
            c->SetLastSyncedTransform(presentedPos, d.lastSyncedRotation);
        }
    }
}

void Rigidbody::SyncExternalMovesToPhysics()
{
    auto &d = DataMut();
    if (d.isKinematic)
        return; // Kinematic is user-driven via SyncCollidersToPhysics already

    auto *go = GetGameObject();
    if (!go)
        return;

    Transform *tf = go->GetTransform();
    if (!tf)
        return;

    glm::vec3 currentPos = tf->GetPosition();
    glm::quat currentRot = tf->GetWorldRotation();

    const float posEps = 1e-4f;
    const float rotEps = 1e-4f;

    // First frame: initialise cache from current Transform.
    // Also check whether the script already moved the Transform away from
    // the body position (e.g. instantiate then set position).  If so,
    // teleport the body immediately instead of waiting one more tick.
    if (!d.hasSyncedOnce) {
        d.lastSyncedPosition = currentPos;
        d.lastSyncedRotation = currentRot;
        d.hasSyncedOnce = true;

        auto &pw = PhysicsWorld::Instance();
        if (!pw.IsInitialized())
            return;

        auto bodyIds = CollectUniqueBodyIds(go);
        if (bodyIds.empty())
            return;

        glm::vec3 bodyPos = pw.GetBodyPosition(bodyIds[0]);
        bool firstFrameDiff = glm::length(currentPos - bodyPos) > posEps;
        if (!firstFrameDiff)
            return;

        // Transform was modified after body creation — teleport now.
        TeleportBodies(pw, go, currentPos, currentRot);
        return;
    }

    bool posDiff = glm::length(currentPos - d.lastSyncedPosition) > posEps;
    bool rotDiff = (1.0f - std::abs(glm::dot(currentRot, d.lastSyncedRotation))) > rotEps;

    if (!posDiff && !rotDiff)
        return; // Transform unchanged since last physics write — nothing to do

    INXLOG_WARN("Rigidbody::SyncExternalMovesToPhysics TELEPORT — posDiff=", posDiff, " rotDiff=", rotDiff,
                " posDelta=", glm::length(currentPos - d.lastSyncedPosition),
                " rotDelta=", (1.0f - std::abs(glm::dot(currentRot, d.lastSyncedRotation))));

    // The user (gizmo / inspector) moved the object externally.
    // Teleport ALL sibling collider bodies to the new Transform position.
    auto &pw = PhysicsWorld::Instance();
    if (!pw.IsInitialized())
        return;

    TeleportBodies(pw, go, currentPos, currentRot);

    // Update cache
    d.lastSyncedPosition = currentPos;
    d.lastSyncedRotation = currentRot;
}

bool Rigidbody::HasLinkedColliders() const
{
    auto *go = GetGameObject();
    if (!go)
        return false;

    return !CollectUniqueBodyIds(go).empty();
}

// ============================================================================
// Internal helpers
// ============================================================================

PhysicsWorld *Rigidbody::GetActivePhysicsWorld(GameObject *&outGo) const
{
    outGo = GetGameObject();
    if (!outGo)
        return nullptr;
    auto &pw = PhysicsWorld::Instance();
    return pw.IsInitialized() ? &pw : nullptr;
}

void Rigidbody::NotifyCollidersBodyTypeChanged()
{
    GameObject *go = nullptr;
    auto *pw = GetActivePhysicsWorld(go);
    if (!pw)
        return;

    const auto &d = Data();

    // Rebuild shapes first — some colliders (e.g. MeshCollider) produce
    // different shape types depending on whether a dynamic Rigidbody exists.
    // This must happen BEFORE SetBodyMotionType so Jolt never sees a
    // MeshShape on a dynamic body.
    auto colliders = go->GetComponents<Collider>();
    for (auto *col : colliders) {
        if (col && col->IsEnabled() && col->GetBodyId() != 0xFFFFFFFF) {
            pw->UpdateBodyShape(col);
            break; // shared body — one rebuild is enough
        }
    }

    // motionType: 0=Static, 1=Kinematic, 2=Dynamic
    int motionType = d.isKinematic ? 1 : 2;

    for (uint32_t bodyId : CollectUniqueBodyIds(go)) {
        pw->SetBodyMotionType(bodyId, motionType);

        // Apply mass, drag, gravity settings
        pw->SetBodyMassProperties(bodyId, d.mass);
        pw->SetBodyDamping(bodyId, d.drag, d.angularDrag);
        pw->SetBodyGravityFactor(bodyId, d.useGravity ? 1.0f : 0.0f);
    }

    // Also apply the new extended settings
    ApplyConstraints();
    ApplyMotionQuality();
    ApplyVelocityLimits();
}

void Rigidbody::ApplyDragSettings()
{
    const auto &d = Data();
    ForEachBody([&](PhysicsWorld &pw, uint32_t id) { pw.SetBodyDamping(id, d.drag, d.angularDrag); });
}

void Rigidbody::ApplyConstraints()
{
    const auto &d = Data();
    // Convert Unity constraints bitmask to Jolt EAllowedDOFs.
    // Unity constraints bits start at bit 1 (FreezePositionX=2), Jolt at bit 0.
    int joltAllowed = 0x3F & ~(d.constraints >> 1);
    ForEachBody([&](PhysicsWorld &pw, uint32_t id) { pw.SetBodyAllowedDOFs(id, joltAllowed, d.mass); });
}

void Rigidbody::ApplyMotionQuality()
{
    const auto &d = Data();
    int joltQuality = MapCollisionDetectionModeToMotionQuality(d.collisionDetectionMode, d.isKinematic);
    ForEachBody([&](PhysicsWorld &pw, uint32_t id) { pw.SetBodyMotionQuality(id, joltQuality); });
}

void Rigidbody::ApplyVelocityLimits()
{
    const auto &d = Data();
    ForEachBody([&](PhysicsWorld &pw, uint32_t id) {
        pw.SetBodyMaxAngularVelocity(id, d.maxAngularVelocity);
        pw.SetBodyMaxLinearVelocity(id, d.maxLinearVelocity);
    });
}

// ============================================================================
// Serialization
// ============================================================================

std::string Rigidbody::Serialize() const
{
    const auto &d = Data();
    auto j = nlohmann::json::parse(Component::Serialize());
    j["mass"] = d.mass;
    j["drag"] = d.drag;
    j["angular_drag"] = d.angularDrag;
    j["use_gravity"] = d.useGravity;
    j["is_kinematic"] = d.isKinematic;
    j["constraints"] = d.constraints;
    j["collision_detection_mode"] = d.collisionDetectionMode;
    j["interpolation"] = d.interpolation;
    j["max_angular_velocity"] = d.maxAngularVelocity;
    j["max_linear_velocity"] = d.maxLinearVelocity;
    j["max_depenetration_velocity"] = d.maxDepenetrationVelocity;
    return j.dump();
}

bool Rigidbody::Deserialize(const std::string &jsonStr)
{
    if (!Component::Deserialize(jsonStr))
        return false;

    auto &d = DataMut();

    try {
        auto j = nlohmann::json::parse(jsonStr);
        if (j.contains("mass"))
            d.mass = j["mass"].get<float>();
        if (j.contains("drag"))
            d.drag = j["drag"].get<float>();
        if (j.contains("angular_drag"))
            d.angularDrag = j["angular_drag"].get<float>();
        if (j.contains("use_gravity"))
            d.useGravity = j["use_gravity"].get<bool>();
        if (j.contains("is_kinematic"))
            d.isKinematic = j["is_kinematic"].get<bool>();
        if (j.contains("constraints"))
            d.constraints = j["constraints"].get<int>();
        if (j.contains("collision_detection_mode"))
            d.collisionDetectionMode = j["collision_detection_mode"].get<int>();
        if (j.contains("interpolation"))
            d.interpolation = std::clamp(j["interpolation"].get<int>(), 0, 1);
        if (j.contains("max_angular_velocity"))
            d.maxAngularVelocity = j["max_angular_velocity"].get<float>();
        if (j.contains("max_linear_velocity"))
            d.maxLinearVelocity = j["max_linear_velocity"].get<float>();
        if (j.contains("max_depenetration_velocity"))
            d.maxDepenetrationVelocity = j["max_depenetration_velocity"].get<float>();

        // Propagate all settings to Jolt bodies (e.g. when edited in Inspector during play)
        NotifyCollidersBodyTypeChanged();

        return true;
    } catch (...) {
        return false;
    }
}

std::unique_ptr<Component> Rigidbody::Clone() const
{
    auto clone = std::make_unique<Rigidbody>();
    clone->m_enabled = m_enabled;
    clone->m_executionOrder = m_executionOrder;
    const auto &src = Data();
    auto &dst = clone->DataMut();
    dst.mass = src.mass;
    dst.drag = src.drag;
    dst.angularDrag = src.angularDrag;
    dst.useGravity = src.useGravity;
    dst.isKinematic = src.isKinematic;
    dst.constraints = src.constraints;
    dst.collisionDetectionMode = src.collisionDetectionMode;
    dst.interpolation = src.interpolation;
    dst.maxAngularVelocity = src.maxAngularVelocity;
    dst.maxLinearVelocity = src.maxLinearVelocity;
    dst.maxDepenetrationVelocity = src.maxDepenetrationVelocity;
    return clone;
}

} // namespace infernux
