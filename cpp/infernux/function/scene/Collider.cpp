/**
 * @file Collider.cpp
 * @brief Base Collider implementation — body registration, transform sync.
 */

#include "Collider.h"

#include "GameObject.h"
#include "MeshRenderer.h"
#include "Rigidbody.h"
#include "SceneManager.h"
#include "Transform.h"
#include "physics/PhysicsECSStore.h"
#include "physics/PhysicsWorld.h"
#include <InxLog.h>

#include <algorithm>
#include <nlohmann/json.hpp>

namespace infernux
{

// ============================================================================
// Constructor / Destructor
// ============================================================================

Collider::Collider()
{
    m_ecsHandle = PhysicsECSStore::Instance().AllocateCollider(this);
}

Collider::~Collider()
{
    // Safe: UnregisterBody checks bodyId == 0xFFFFFFFF and returns early
    // if already cleaned up via OnDestroy().
    UnregisterBody();

    // Release pool slot
    PhysicsECSStore::Instance().ReleaseCollider(m_ecsHandle);
}

// ============================================================================
// Lifecycle
// ============================================================================

void Collider::Awake()
{
    // If a sibling Rigidbody already exists and is enabled, cache it before
    // body creation so the body is created as dynamic/kinematic instead of
    // static.  This handles the case where the Collider is added *after* the
    // Rigidbody (whose OnEnable already ran and won't re-fire).
    if (auto *go = GetGameObject()) {
        auto *rb = go->GetComponent<Rigidbody>();
        if (rb && rb->IsEnabled())
            DataMut().cachedRigidbody = rb;
    }

    if (!Data().deserialized) {
        AutoFitToMesh();
    }

    // Defer body creation to the next pre-physics flush (Unity-style).
    // This is the key batch-creation optimization: add_component("BoxCollider")
    // in a Python loop becomes near-zero physics cost; the actual Jolt bodies
    // are created in one batch inside SceneManager::FlushPendingBroadphase().
    PhysicsECSStore::Instance().QueueBodyCreation(m_ecsHandle);
}

void Collider::OnEnable()
{
    if (Data().bodyId == 0xFFFFFFFF) {
        // Body not yet created (deferred from Awake) — ensure it's queued.
        PhysicsECSStore::Instance().QueueBodyCreation(m_ecsHandle);
        return;
    }
    // Re-enable after disable — body already exists, normal path.
    PhysicsWorld::Instance().UpdateBodyShape(this);
    AddToBroadphase();
}

void Collider::OnDisable()
{
    if (IsBeingDestroyed()) {
        return;
    }

    auto *go = GetGameObject();
    if (go && Data().bodyId != 0xFFFFFFFF) {
        bool hasOtherEnabledSibling = false;
        Collider *replacement = nullptr;
        auto colliders = go->GetComponents<Collider>();
        for (auto *col : colliders) {
            if (!col || col == this || !col->IsEnabled())
                continue;
            if (col->GetBodyId() == Data().bodyId) {
                hasOtherEnabledSibling = true;
                replacement = col;
                break;
            }
        }

        if (hasOtherEnabledSibling) {
            PhysicsWorld::Instance().RebindBodyCollider(Data().bodyId, replacement);
            PhysicsWorld::Instance().UpdateBodyShape(this, this);
        }
    }

    RemoveFromBroadphase();
}

void Collider::OnDestroy()
{
    UnregisterBody();
}

// ============================================================================
// Property setters (rebuild body when changed)
// ============================================================================

void Collider::SetIsTrigger(bool trigger)
{
    auto &d = DataMut();
    if (d.isTrigger == trigger)
        return;
    d.isTrigger = trigger;

    if (d.bodyId != 0xFFFFFFFF) {
        bool groupIsTrigger = false;
        if (auto *go = GetGameObject()) {
            auto colliders = go->GetComponents<Collider>();
            for (auto *col : colliders) {
                if (!col || !col->IsEnabled())
                    continue;
                if (col->GetBodyId() == d.bodyId && col->IsTrigger()) {
                    groupIsTrigger = true;
                    break;
                }
            }
        } else {
            groupIsTrigger = trigger;
        }

        auto &physics = PhysicsWorld::Instance();
        physics.SetBodyIsSensor(d.bodyId, groupIsTrigger);

        // Wake dynamic bodies that may be resting on (or overlapping) this
        // collider so they re-evaluate the changed sensor state.
        physics.WakeBodiesTouchingStatic(d.bodyId);

        // Clear stale contact-pair tracking so the listener produces fresh
        // Enter events (Trigger or Collision) on the next physics step
        // instead of suppressing them as wake-from-sleep duplicates.
        physics.InvalidateContactPairsForBody(d.bodyId);
    }
}

void Collider::SetCenter(const glm::vec3 &center)
{
    auto &d = DataMut();
    if (d.center == center)
        return;
    d.center = center;
    RebuildShape();
}

// ---- Friction / Bounciness helpers ----

/// @brief Recompute max friction & bounciness across all enabled colliders on the same body
///        and push the values to Jolt.
static void ApplyMaterialToBody(Collider *self)
{
    auto &d = PhysicsECSStore::Instance().GetCollider(self->GetECSHandle());
    if (d.bodyId == 0xFFFFFFFF)
        return;

    auto *go = self->GetGameObject();
    if (!go)
        return;

    float maxFriction = 0.0f;
    float maxBounciness = 0.0f;
    auto colliders = go->GetComponents<Collider>();
    for (auto *col : colliders) {
        if (!col || !col->IsEnabled())
            continue;
        const auto &cd = PhysicsECSStore::Instance().GetCollider(col->GetECSHandle());
        maxFriction = std::max(maxFriction, cd.friction);
        maxBounciness = std::max(maxBounciness, cd.bounciness);
    }

    auto &pw = PhysicsWorld::Instance();
    pw.SetBodyFriction(d.bodyId, maxFriction);
    pw.SetBodyRestitution(d.bodyId, maxBounciness);
}

void Collider::SetFriction(float friction)
{
    auto &d = DataMut();
    friction = std::max(friction, 0.0f);
    if (d.friction == friction)
        return;
    d.friction = friction;
    ApplyMaterialToBody(this);
}

void Collider::SetBounciness(float bounciness)
{
    auto &d = DataMut();
    bounciness = std::clamp(bounciness, 0.0f, 1.0f);
    if (d.bounciness == bounciness)
        return;
    d.bounciness = bounciness;
    ApplyMaterialToBody(this);
}

// ============================================================================
// Helpers
// ============================================================================

glm::vec3 Collider::GetWorldScale() const
{
    if (auto *go = GetGameObject()) {
        if (auto *tf = go->GetTransform()) {
            return glm::abs(tf->GetWorldScale());
        }
    }
    return glm::vec3(1.0f);
}

void Collider::AutoFitToMesh()
{
    // Default no-op. Derived classes override to set size/radius/height
    // from sibling MeshRenderer bounds.
}

// ============================================================================
// Jolt body management
// ============================================================================

void Collider::RegisterBody()
{
    auto &pw = PhysicsWorld::Instance();
    if (!pw.IsInitialized())
        return;

    auto &d = DataMut();

    if (d.bodyId != 0xFFFFFFFF)
        return; // already registered

    auto *go = GetGameObject();
    if (!go)
        return;

    // Skip physics body creation for objects in non-active scenes
    // (e.g. prefab template cache) to avoid phantom colliders.
    if (go->GetScene() != SceneManager::Instance().GetActiveScene())
        return;

    auto colliders = go->GetComponents<Collider>();
    for (auto *col : colliders) {
        if (!col || col == this)
            continue;

        const auto &otherData = col->Data();
        if (otherData.bodyId == 0xFFFFFFFF)
            continue;

        d.bodyId = otherData.bodyId;
        d.bodyInBroadphase = otherData.bodyInBroadphase;

        if (auto *tf = go->GetTransform()) {
            const glm::quat rot = tf->GetWorldRotation();
            d.lastSyncedRot = rot;
            d.lastSyncedPos = tf->GetPosition();
            d.lastScale = tf->GetWorldScale();
        }

        pw.UpdateBodyShape(this);
        return;
    }

    // Use cached Rigidbody pointer
    bool isStatic = (d.cachedRigidbody == nullptr || !d.cachedRigidbody->IsEnabled());

    bool groupIsTrigger = false;
    for (auto *col : colliders) {
        if (!col || !col->IsEnabled())
            continue;
        if (col->IsTrigger()) {
            groupIsTrigger = true;
            break;
        }
    }

    d.bodyId = pw.CreateBody(this, isStatic, groupIsTrigger);

    // Initialize cached transform so first SyncTransformToPhysics doesn't
    // see a spurious move from the (0,0,0) default.
    if (auto *tf = go->GetTransform()) {
        glm::quat rot = tf->GetWorldRotation();
        d.lastSyncedRot = rot;
        d.lastSyncedPos = tf->GetPosition();
        d.lastScale = tf->GetWorldScale();
    }

    for (auto *col : colliders) {
        if (!col)
            continue;
        auto &other = col->DataMut();
        other.bodyId = d.bodyId;
        other.bodyInBroadphase = false;
        if (auto *tf = go->GetTransform()) {
            other.lastSyncedPos = tf->GetPosition();
            other.lastSyncedRot = tf->GetWorldRotation();
            other.lastScale = tf->GetWorldScale();
        }
    }

    // ========================================================================
    // Fix component-ordering dependency: If a sibling Rigidbody has already
    // been OnEnable'd (setting cachedRigidbody) but this Collider's body
    // didn't exist yet at that time, NotifyCollidersBodyTypeChanged() would
    // have skipped this body.  Apply the Rigidbody's full configuration now
    // that the body exists.
    // ========================================================================
    if (d.cachedRigidbody && d.cachedRigidbody->IsEnabled() && d.bodyId != 0xFFFFFFFF) {
        int motionType = d.cachedRigidbody->IsKinematic() ? 1 : 2;
        pw.SetBodyMotionType(d.bodyId, motionType);
        pw.SetBodyMassProperties(d.bodyId, d.cachedRigidbody->GetMass());
        pw.SetBodyDamping(d.bodyId, d.cachedRigidbody->GetDrag(), d.cachedRigidbody->GetAngularDrag());
        pw.SetBodyGravityFactor(d.bodyId, d.cachedRigidbody->GetUseGravity() ? 1.0f : 0.0f);

        // Unity mapping:
        // - Dynamic + Continuous / ContinuousDynamic -> Jolt LinearCast sweep CCD
        // - Kinematic + Continuous -> speculative/discrete by default
        // - Any + ContinuousDynamic -> full sweep CCD when explicitly requested
        // - ContinuousSpeculative -> Jolt Discrete + speculative contacts
        const int collisionMode = d.cachedRigidbody->GetCollisionDetectionMode();
        const bool isKinematicBody = d.cachedRigidbody->IsKinematic();
        int joltQuality = 0;
        if (collisionMode == 2) {
            joltQuality = 1;
        } else if (collisionMode == 1) {
            joltQuality = isKinematicBody ? 0 : 1;
        }
        pw.SetBodyMotionQuality(d.bodyId, joltQuality);

        // Constraints (allowed DOFs)
        int constraints = d.cachedRigidbody->GetConstraints();
        if (constraints != 0) {
            int joltAllowed = 0x3F & ~(constraints >> 1);
            pw.SetBodyAllowedDOFs(d.bodyId, joltAllowed, d.cachedRigidbody->GetMass());
        }

        // Max angular velocity
        pw.SetBodyMaxAngularVelocity(d.bodyId, d.cachedRigidbody->GetMaxAngularVelocity());
        pw.SetBodyMaxLinearVelocity(d.bodyId, d.cachedRigidbody->GetMaxLinearVelocity());
    }
}

void Collider::UnregisterBody()
{
    auto &d = DataMut();
    if (d.bodyId == 0xFFFFFFFF)
        return;

    auto *go = GetGameObject();
    Collider *replacement = nullptr;
    if (go) {
        auto colliders = go->GetComponents<Collider>();
        for (auto *col : colliders) {
            if (!col || col == this)
                continue;
            if (col->GetBodyId() == d.bodyId) {
                replacement = col;
                break;
            }
        }
    }

    if (replacement) {
        PhysicsWorld::Instance().RebindBodyCollider(d.bodyId, replacement);

        bool hasOtherEnabledSibling = false;
        if (go) {
            auto colliders = go->GetComponents<Collider>();
            for (auto *col : colliders) {
                if (!col || col == this || !col->IsEnabled())
                    continue;
                if (col->GetBodyId() == d.bodyId) {
                    hasOtherEnabledSibling = true;
                    break;
                }
            }
        }

        if (hasOtherEnabledSibling) {
            PhysicsWorld::Instance().UpdateBodyShape(replacement, this);
        }
    } else {
        RemoveFromBroadphase();
        PhysicsWorld::Instance().DestroyBody(this);
    }

    d.bodyId = 0xFFFFFFFF;
    d.bodyInBroadphase = false;
}

void Collider::AddToBroadphase()
{
    auto &d = DataMut();
    if (d.bodyId == 0xFFFFFFFF)
        return;

    auto *go = GetGameObject();
    if (go) {
        auto colliders = go->GetComponents<Collider>();
        for (auto *col : colliders) {
            if (!col)
                continue;
            if (col != this && col->GetBodyId() == d.bodyId && col->Data().bodyInBroadphase) {
                d.bodyInBroadphase = true;
                return;
            }
        }
    }

    if (d.bodyInBroadphase)
        return;

    bool isStatic = (d.cachedRigidbody == nullptr || !d.cachedRigidbody->IsEnabled());

    // Defer broadphase addition to the next pre-physics flush (Unity-style).
    // The body exists in Jolt but won't participate in queries/simulation
    // until SceneManager flushes the pending queue.
    PhysicsECSStore::Instance().QueueBroadphaseAdd(d.bodyId, isStatic);
    d.bodyInBroadphase = true;

    if (go) {
        auto colliders = go->GetComponents<Collider>();
        for (auto *col : colliders) {
            if (col && col->GetBodyId() == d.bodyId) {
                col->DataMut().bodyInBroadphase = true;
            }
        }
    }
}

void Collider::RemoveFromBroadphase()
{
    auto &d = DataMut();
    if (d.bodyId == 0xFFFFFFFF || !d.bodyInBroadphase)
        return;

    auto *go = GetGameObject();
    if (go) {
        auto colliders = go->GetComponents<Collider>();
        for (auto *col : colliders) {
            if (!col || col == this || !col->IsEnabled())
                continue;
            if (col->GetBodyId() == d.bodyId) {
                d.bodyInBroadphase = true;
                return;
            }
        }
    }

    PhysicsWorld::Instance().RemoveBodyFromBroadphase(d.bodyId);
    d.bodyInBroadphase = false;

    if (go) {
        auto colliders = go->GetComponents<Collider>();
        for (auto *col : colliders) {
            if (col && col->GetBodyId() == d.bodyId) {
                col->DataMut().bodyInBroadphase = false;
            }
        }
    }
}

void Collider::RebuildShape()
{
    auto &d = DataMut();
    if (d.bodyId == 0xFFFFFFFF)
        return;

    auto *go = GetGameObject();
    if (go) {
        auto colliders = go->GetComponents<Collider>();
        for (auto *col : colliders) {
            if (!col || !col->IsEnabled() || col->GetBodyId() != d.bodyId)
                continue;
            if (col != this) {
                return;
            }
            break;
        }
    }

    PhysicsWorld::Instance().UpdateBodyShape(this);

    // If this is a static body (no Rigidbody), wake nearby dynamic
    // bodies so they react to the shape change immediately.
    bool isStatic = (d.cachedRigidbody == nullptr || !d.cachedRigidbody->IsEnabled());
    if (isStatic) {
        PhysicsWorld::Instance().WakeBodiesTouchingStatic(d.bodyId);
    }
}

void Collider::SyncTransformToPhysics()
{
    auto &d = DataMut();
    if (d.bodyId == 0xFFFFFFFF)
        return;

    if (auto *go = GetGameObject()) {
        auto colliders = go->GetComponents<Collider>();
        for (auto *col : colliders) {
            if (!col || !col->IsEnabled() || col->GetBodyId() != d.bodyId)
                continue;
            if (col != this) {
                return;
            }
            break;
        }
    }

    // Use cached Rigidbody pointer — no dynamic_cast needed.
    Rigidbody *rb = d.cachedRigidbody;

    // If a Rigidbody sibling exists and is enabled, the body is dynamic.
    // Dynamic bodies are driven by physics — don't overwrite their position.
    if (rb && rb->IsEnabled() && !rb->IsKinematic())
        return;

    auto *go = GetGameObject();
    if (!go)
        return;

    Transform *tf = go->GetTransform();
    if (!tf)
        return;

    // Rebuild the Jolt shape when effective world scale changes.
    // Collider geometry is baked using world-space scale, so parent scaling
    // must also invalidate the body shape.
    glm::vec3 currentScale = tf->GetWorldScale();
    if (currentScale != d.lastScale) {
        d.lastScale = currentScale;
        RebuildShape();
    }

    glm::quat rot = tf->GetWorldRotation();
    glm::vec3 pos = tf->GetPosition();

    // Compare against cached last-synced values (pure C++ — no Jolt lock)
    bool moved = (pos != d.lastSyncedPos || rot != d.lastSyncedRot);

    if (moved) {
        d.lastSyncedPos = pos;
        d.lastSyncedRot = rot;
        PhysicsWorld::Instance().SetBodyPosition(d.bodyId, pos, rot);

        // After moving a static/kinematic body, wake nearby dynamic bodies.
        bool isStaticBody = (rb == nullptr || !rb->IsEnabled());
        if (isStaticBody) {
            PhysicsWorld::Instance().WakeBodiesTouchingStatic(d.bodyId);
        }
    }
}

// ============================================================================
// Serialization
// ============================================================================

std::string Collider::Serialize() const
{
    const auto &d = Data();
    // Start from Component base (provides type, component_id, enabled, schema_version)
    auto j = nlohmann::json::parse(Component::Serialize());
    j["is_trigger"] = d.isTrigger;
    j["center"] = {d.center.x, d.center.y, d.center.z};
    j["friction"] = d.friction;
    j["bounciness"] = d.bounciness;
    return j.dump();
}

bool Collider::Deserialize(const std::string &jsonStr)
{
    // Deserialize Component base fields first (enabled, component_id)
    if (!Component::Deserialize(jsonStr))
        return false;

    auto &d = DataMut();
    d.deserialized = true;

    try {
        auto j = nlohmann::json::parse(jsonStr);
        if (j.contains("is_trigger"))
            d.isTrigger = j["is_trigger"].get<bool>();
        if (j.contains("center")) {
            auto &c = j["center"];
            d.center = glm::vec3(c[0].get<float>(), c[1].get<float>(), c[2].get<float>());
        }
        if (j.contains("friction"))
            d.friction = j["friction"].get<float>();
        if (j.contains("bounciness"))
            d.bounciness = j["bounciness"].get<float>();
        // NOTE: RebuildShape() is called by derived classes after their own
        // fields are deserialized (so both base + derived changes are applied
        // in a single shape rebuild).
        return true;
    } catch (const std::exception &e) {
        INXLOG_ERROR("Collider::Deserialize failed: ", e.what());
        return false;
    }
}

void Collider::CloneBaseColliderData(Collider &target) const
{
    target.m_enabled = m_enabled;
    target.m_executionOrder = m_executionOrder;
    const auto &src = Data();
    auto &dst = target.DataMut();
    dst.isTrigger = src.isTrigger;
    dst.center = src.center;
    dst.friction = src.friction;
    dst.bounciness = src.bounciness;
}

} // namespace infernux
