/**
 * @file Rigidbody.cpp
 * @brief Rigidbody component — drives dynamic/kinematic physics simulation.
 *
 * Rigidbody owns the actor and native body. Colliders contribute shapes;
 * removing the last shape does not remove motion state or joint endpoints.
 */

#include "Rigidbody.h"
#include "ComponentDocumentValidation.h"

#include "Collider.h"
#include "ComponentFactory.h"
#include "GameObject.h"
#include "MeshCollider.h"
#include "SceneManager.h"
#include "Transform.h"
#include "TransformECSStore.h"
#include "physics/PhysicsECSStore.h"
#include "physics/PhysicsWorld.h"

#include <core/log/InxLog.h>

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_set>

namespace infernux
{

namespace
{

static bool RotationChanged(const glm::quat &current, const glm::quat &previous)
{
    // q and -q represent the same orientation. Compare aligned components:
    // 1 - abs(dot(q, previous)) loses small angles to float cancellation, and
    // a 1e-4 dot tolerance silently discards rotations below about 1.6 degrees.
    const glm::quat aligned = glm::dot(current, previous) < 0.0f ? -previous : previous;
    const glm::quat delta = current - aligned;
    constexpr float componentToleranceSquared = 1e-12f;
    return glm::dot(delta, delta) > componentToleranceSquared;
}

static int MapCollisionDetectionModeToMotionQuality(int mode, bool isKinematic)
{
    switch (mode) {
    case static_cast<int>(CollisionDetectionMode::ContinuousDynamic):
        // ContinuousDynamic keeps Jolt's static sweep and additionally opts
        // into PhysicsWorld's dynamic-pair TOI subdivision.
        return isKinematic ? 0 : 2;
    case static_cast<int>(CollisionDetectionMode::Continuous):
        // Jolt's LinearCast path only runs for dynamic bodies. Kinematic CCD
        // would require a separate speculative-contact implementation.
        return isKinematic ? 0 : 1;
    case static_cast<int>(CollisionDetectionMode::Discrete):
        return 0;
    default:
        throw std::invalid_argument("Unsupported collision detection mode");
    }
}

static bool IsFinite(const glm::vec3 &value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

static bool IsFiniteRotation(const glm::quat &value)
{
    return std::isfinite(value.w) && std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) &&
           glm::dot(value, value) > 1e-12f;
}

static RigidbodyMassProperties ValidateMassProperties(const glm::vec3 &center, const glm::vec3 &moments,
                                                      const glm::quat &rotation)
{
    if (!IsFinite(center) || !IsFinite(moments) || glm::any(glm::lessThanEqual(moments, glm::vec3(0.0f))))
        throw std::invalid_argument("mass properties require a finite center and positive finite principal moments");
    for (int i = 0; i < 3; ++i)
        if (!std::isfinite(1.0f / moments[i]))
            throw std::invalid_argument("principal inverse inertia must be representable");
    const double largest = std::max({double(moments.x), double(moments.y), double(moments.z)});
    if (2.0 * largest > (double(moments.x) + double(moments.y) + double(moments.z)) + 1e-6 * largest)
        throw std::invalid_argument("principal inertia moments must satisfy the physical triangle inequality");
    if (!IsFiniteRotation(rotation))
        throw std::invalid_argument("principal inertia rotation must be a finite nonzero quaternion");
    return {center, moments, glm::quat(glm::normalize(glm::dquat(rotation)))};
}

static std::optional<RigidbodyMassProperties> ReadMassProperties(const nlohmann::json &document)
{
    const auto entry = document.find("mass_properties");
    if (entry == document.end())
        return std::nullopt; // Absence means geometry-derived COM and inertia.
    using namespace component_document_validation;
    if (!entry->is_object() || entry->size() != 3)
        throw std::invalid_argument("mass_properties requires center, moments and rotation");
    RequireFiniteVector(*entry, "center", 3, "Rigidbody.mass_properties");
    RequireFiniteVector(*entry, "moments", 3, "Rigidbody.mass_properties");
    RequireFiniteVector(*entry, "rotation", 4, "Rigidbody.mass_properties");
    const auto &c = entry->at("center"), &i = entry->at("moments"), &q = entry->at("rotation");
    return ValidateMassProperties({c[0].get<float>(), c[1].get<float>(), c[2].get<float>()},
                                  {i[0].get<float>(), i[1].get<float>(), i[2].get<float>()},
                                  {q[3].get<float>(), q[0].get<float>(), q[1].get<float>(), q[2].get<float>()});
}

static void ValidateForceMode(ForceMode mode)
{
    switch (mode) {
    case ForceMode::Force:
    case ForceMode::Acceleration:
    case ForceMode::Impulse:
    case ForceMode::VelocityChange:
        return;
    }
    throw std::invalid_argument("unsupported force mode");
}

SemanticTypeDescriptor DescribeRigidbody()
{
    SemanticTypeDescriptor type;
    type.typeGuid = "native:infernux.Rigidbody";
    type.readableId = "infernux.component.rigidbody";
    type.owner = "engine:native";
    type.origin = "native";
    type.displayName = "Rigidbody";
    type.runtimeProfiles = {"editor", "player", "headless"};
    const auto add = [&](const char *name, const char *kind, nlohmann::json initial) -> nlohmann::json & {
        type.fields.push_back({std::string("Rigidbody.") + name,
                               std::string("FieldType.") + kind,
                               false,
                               {{"field_id", name},
                                {"serialized_name", name},
                                {"serialized", true},
                                {"hidden", false},
                                {"nullable", false},
                                {"storage_kind", "native_property"},
                                {"display_name_key", std::string("rigidbody.") + name},
                                {"tooltip", std::string("rigidbody.tooltip.") + name},
                                {"default", std::move(initial)}}});
        return type.fields.back().attributes;
    };
    const auto enumeration = [&](const char *name, const char *enumName, std::initializer_list<const char *> names,
                                 std::initializer_list<const char *> labels) {
        nlohmann::json members = nlohmann::json::array();
        size_t value = 0;
        for (const char *member : names)
            members.push_back({{"name", member}, {"value", value++}});
        auto &attributes = add(name, "ENUM", {{"$type", "enum"}, {"enum_type", enumName}, {"name", *names.begin()}});
        attributes["enum"] = {{"type_id", std::string("native:infernux.") + enumName},
                              {"members", std::move(members)},
                              {"labels", labels}};
    };

    add("mass", "FLOAT", 1.0)["range"] = {0.001, 1000.0};
    add("drag", "FLOAT", 0.0)["range"] = {0.0, 100.0};
    add("angular_drag", "FLOAT", 0.05)["range"] = {0.0, 100.0};
    add("use_gravity", "BOOL", true);
    add("is_kinematic", "BOOL", false);
    add("constraints", "INT", 0)["flags_type"] = "native:infernux.RigidbodyConstraints";
    enumeration(
        "collision_detection_mode", "CollisionDetectionMode", {"Discrete", "Continuous", "ContinuousDynamic"},
        {"rigidbody.collision.discrete", "rigidbody.collision.continuous", "rigidbody.collision.continuous_dynamic"});
    enumeration("interpolation", "RigidbodyInterpolation", {"None", "Interpolate"},
                {"rigidbody.interpolation.none", "rigidbody.interpolation.interpolate"});
    add("max_angular_velocity", "FLOAT", 7.0)["range"] = {0.0, 1000.0};
    add("max_linear_velocity", "FLOAT", 500.0)["range"] = {0.0, 10000.0};
    return type;
}

} // namespace

namespace
{
const bool registeredRigidbody = ComponentFactory::Register(
    "Rigidbody", [] { return std::make_unique<Rigidbody>(); }, Rigidbody::ValidateSerializedDocument,
    Rigidbody::GetTypeConstraints(), DescribeRigidbody);
} // namespace

// ============================================================================
// Shared helpers
// ============================================================================

template <typename Fn> void Rigidbody::ForEachBody(Fn &&fn)
{
    GameObject *go = nullptr;
    auto *pw = GetActivePhysicsWorld(go);
    if (!pw)
        return;
    const uint32_t bodyId = GetBodyId();
    if (bodyId != 0xFFFFFFFF)
        fn(*pw, bodyId);
}

void Rigidbody::TeleportBodies(PhysicsWorld &pw, GameObject *go, const glm::vec3 &pos, const glm::quat &rot,
                               bool publishTransform)
{
    auto &d = DataMut();
    if (publishTransform) {
        auto *transform = go->GetTransform();
        transform->ApplyWorldPoseFromPhysics(pos, rot, true);
        if (!TransformECSStore::Instance().IsFrameCacheActive())
            SceneManager::Instance().PublishPhysicsTransformsToRenderer();
        d.lastAuthoredPoseRevision = TransformECSStore::Instance().GetAuthoredPoseRevision(transform->GetECSHandle());
        CacheActorPose(pos, rot);
        d.lastSyncedPosition = pos;
        d.lastSyncedRotation = rot;
        d.hasSyncedOnce = true;
    } else if (!d.hasSyncedOnce) {
        auto *transform = go->GetTransform();
        d.lastSyncedPosition = transform->GetPosition();
        d.lastSyncedRotation = transform->GetWorldRotation();
        d.lastAuthoredPoseRevision = TransformECSStore::Instance().GetAuthoredPoseRevision(transform->GetECSHandle());
        d.hasSyncedOnce = true;
    }
    const uint32_t bodyId = GetBodyId();
    if (bodyId != 0xFFFFFFFF) {
        pw.SetBodyPosition(bodyId, pos, rot, publishTransform);
        pw.ActivateBody(bodyId);
    }
    d.previousPhysicsPosition = pos;
    d.previousPhysicsRotation = rot;
    d.currentPhysicsPosition = pos;
    d.currentPhysicsRotation = rot;
    d.hasPhysicsPose = true;
    d.wasSleeping = false;
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
    PhysicsECSStore::Instance().ReleaseRigidbody(m_ecsHandle);
}

uint32_t Rigidbody::GetBodyId() const
{
    auto &store = PhysicsECSStore::Instance();
    const auto handle = Data().actorHandle;
    return store.IsValid(handle) ? store.GetActor(handle).bodyId : 0xFFFFFFFF;
}

uint32_t Rigidbody::GetParticipatingBodyId() const
{
    auto &store = PhysicsECSStore::Instance();
    const auto handle = Data().actorHandle;
    return store.IsValid(handle) && store.GetActor(handle).bodyInBroadphase ? store.GetActor(handle).bodyId
                                                                            : 0xFFFFFFFF;
}

std::optional<std::tuple<uint32_t, uint32_t, uint64_t, uint64_t>> Rigidbody::GetResultTargetEpoch() const
{
    const auto &store = PhysicsECSStore::Instance();
    const auto handle = Data().actorHandle;
    if (!IsEnabled() || !GetGameObject() || !GetGameObject()->IsActiveInHierarchy() || !store.IsValid(handle))
        return std::nullopt;
    // A Jolt body is created at most once for an actor. The actor handle is
    // already assigned when Start or FixedUpdate first registers a result.
    return std::tuple{handle.index, handle.generation, store.GetActor(handle).shapeRevision, m_resultEpoch};
}

void Rigidbody::Awake()
{
    PhysicsECSStore::Instance().AttachRigidbody(this);
}

void Rigidbody::OnHierarchyPublished()
{
    PhysicsECSStore::Instance().AttachRigidbody(this);
}

void Rigidbody::OnEnable()
{
    auto &store = PhysicsECSStore::Instance();
    if (!store.IsValid(Data().actorHandle))
        store.AttachRigidbody(this);
    NotifyCollidersBodyTypeChanged();
}

void Rigidbody::OnDisable()
{
    ++m_resultEpoch;
    auto &store = PhysicsECSStore::Instance();
    const auto handle = Data().actorHandle;
    if (!store.IsValid(handle))
        return;
    const auto bodyId = GetBodyId();
    if (bodyId != 0xFFFFFFFF)
        PhysicsWorld::Instance().DestroyConstraintsForBody(bodyId);
    if (!IsBeingDestroyed())
        store.RefreshActor(handle);
}

void Rigidbody::OnDestroy()
{
    PhysicsECSStore::Instance().DetachRigidbody(this);
}

// ============================================================================
// Property setters
// ============================================================================

std::optional<RigidbodyMassProperties> Rigidbody::PrepareMass(float mass) const
{
    if (!std::isfinite(mass) || mass < 0.001f)
        throw std::invalid_argument("mass must be finite and at least 0.001");
    const auto &d = Data();
    auto properties = m_massProperties;
    if (properties) {
        const glm::vec3 moments = glm::vec3(glm::dvec3(properties->moments) * (double(mass) / d.mass));
        if (!IsFinite(moments) || glm::any(glm::lessThanEqual(moments, glm::vec3(0.0f))))
            throw std::invalid_argument("mass change produces unrepresentable inertia");
        properties->moments = moments;
    }
    PhysicsWorld::Instance().ValidateBodyMassProperties(*this, mass, properties ? &*properties : nullptr);
    return properties;
}

void Rigidbody::ValidateMass(float mass) const
{
    (void)PrepareMass(mass);
}

void Rigidbody::SetMass(float mass)
{
    const auto properties = PrepareMass(mass);
    auto &d = DataMut();
    d.mass = mass;
    m_massProperties = properties;
    ApplyConstraints();
}

void Rigidbody::SetMassProperties(float mass, const glm::vec3 &center, const glm::vec3 &moments,
                                  const glm::quat &rotation)
{
    if (!std::isfinite(mass) || mass < 0.001f)
        throw std::invalid_argument("mass must be finite and at least 0.001");
    const auto properties = ValidateMassProperties(center, moments, rotation);
    PhysicsWorld::Instance().ValidateBodyMassProperties(*this, mass, &properties);
    auto &data = DataMut();
    data.mass = mass;
    m_massProperties = properties;
    auto &store = PhysicsECSStore::Instance();
    if (store.IsValid(Data().actorHandle))
        store.RefreshActor(Data().actorHandle);
}

void Rigidbody::ResetMassProperties()
{
    PhysicsWorld::Instance().ValidateBodyMassProperties(*this, Data().mass, nullptr);
    m_massProperties.reset();
    auto &store = PhysicsECSStore::Instance();
    if (store.IsValid(Data().actorHandle))
        store.RefreshActor(Data().actorHandle);
}

void Rigidbody::SetDrag(float drag)
{
    if (!std::isfinite(drag) || drag < 0.0f)
        throw std::invalid_argument("drag must be finite and non-negative");
    DataMut().drag = drag;
    ApplyDragSettings();
}

void Rigidbody::SetAngularDrag(float drag)
{
    if (!std::isfinite(drag) || drag < 0.0f)
        throw std::invalid_argument("angular drag must be finite and non-negative");
    DataMut().angularDrag = drag;
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
    bool wasKinematic = d.isKinematic;
    if (!kinematic && m_massProperties)
        PhysicsWorld::Instance().ValidateBodyMassProperties(*this, d.mass, &*m_massProperties);
    d.isKinematic = kinematic;
    NotifyCollidersBodyTypeChanged(false);
    PhysicsECSStore::Instance().NotifyCollisionSceneChanged();

    // When switching kinematic → dynamic, synchronise interpolation caches
    // from the current Jolt body position.  During kinematic mode neither
    // SyncPhysicsToTransform nor ApplyInterpolatedTransform update
    // previousPhysicsPosition / currentPhysicsPosition, so they remain at
    // the pre-drag value.  Without this reset, the first
    // ApplyInterpolatedRigidbodies call after the switch overwrites the
    // Transform with the stale position, causing a one-frame flash-back.
    if (wasKinematic && !kinematic) {
        auto *go = GetGameObject();
        if (go) {
            uint32_t bid = GetBodyId();
            if (bid != 0xFFFFFFFF) {
                auto &pw = PhysicsWorld::Instance();
                glm::vec3 bodyPos = pw.GetBodyPosition(bid);
                glm::quat bodyRot = glm::normalize(pw.GetBodyRotation(bid));
                d.previousPhysicsPosition = bodyPos;
                d.previousPhysicsRotation = bodyRot;
                d.currentPhysicsPosition = bodyPos;
                d.currentPhysicsRotation = bodyRot;
                d.hasPhysicsPose = true;
            }
            Transform *tf = go->GetTransform();
            if (tf) {
                d.lastSyncedPosition = tf->GetPosition();
                d.lastSyncedRotation = tf->GetWorldRotation();
                d.lastAuthoredPoseRevision = TransformECSStore::Instance().GetAuthoredPoseRevision(tf->GetECSHandle());
                d.hasSyncedOnce = true;
            }
        }
    }
}

void Rigidbody::SetConstraints(int constraints)
{
    constexpr int kValidConstraintBits = static_cast<int>(RigidbodyConstraints::FreezeAll);
    if (constraints < 0 || (constraints & ~kValidConstraintBits) != 0)
        throw std::invalid_argument("constraints contains unsupported bits");
    auto &d = DataMut();
    if (d.constraints == constraints)
        return;
    PhysicsWorld::Instance().ValidateBodyMassProperties(*this, d.mass, m_massProperties ? &*m_massProperties : nullptr);
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
    if (mode < static_cast<int>(CollisionDetectionMode::Discrete) ||
        mode > static_cast<int>(CollisionDetectionMode::ContinuousDynamic)) {
        throw std::invalid_argument("collision detection mode must be Discrete, Continuous, or ContinuousDynamic");
    }
    auto &d = DataMut();
    if (d.collisionDetectionMode == mode)
        return;
    d.collisionDetectionMode = mode;
    ApplyMotionQuality();
}

void Rigidbody::SetInterpolation(int mode)
{
    if (mode < static_cast<int>(RigidbodyInterpolation::None) ||
        mode > static_cast<int>(RigidbodyInterpolation::Interpolate))
        throw std::invalid_argument("interpolation must be None or Interpolate");
    auto &d = DataMut();
    if (d.interpolation == mode)
        return;
    d.interpolation = mode;
}

void Rigidbody::SetMaxAngularVelocity(float vel)
{
    if (!std::isfinite(vel) || vel < 0.0f)
        throw std::invalid_argument("max angular velocity must be finite and non-negative");
    DataMut().maxAngularVelocity = vel;
    ApplyVelocityLimits();
}

void Rigidbody::SetMaxLinearVelocity(float vel)
{
    if (!std::isfinite(vel) || vel < 0.0f)
        throw std::invalid_argument("max linear velocity must be finite and non-negative");
    DataMut().maxLinearVelocity = vel;
    ApplyVelocityLimits();
}

// ============================================================================
// Velocity
// ============================================================================

glm::vec3 Rigidbody::GetVelocity() const
{
    uint32_t bid = GetBodyId();
    return (bid != 0xFFFFFFFF) ? PhysicsWorld::Instance().GetBodyLinearVelocity(bid) : Data().linearVelocity;
}

void Rigidbody::SetVelocity(const glm::vec3 &vel)
{
    if (!IsFinite(vel))
        throw std::invalid_argument("velocity must contain finite values");
    auto &data = DataMut();
    data.linearVelocity = vel;
    data.hasLinearVelocity = true;
    auto *go = GetGameObject();
    if (go) {
        auto &pw = PhysicsWorld::Instance();
        const uint32_t bodyId = GetBodyId();
        if (bodyId != 0xFFFFFFFF) {
            pw.SetBodyLinearVelocity(bodyId, vel);
            data.hasLinearVelocity = false;
        }
    }
    PhysicsECSStore::Instance().NotifyCollisionSceneChanged();
}

glm::vec3 Rigidbody::GetAngularVelocity() const
{
    uint32_t bid = GetBodyId();
    return (bid != 0xFFFFFFFF) ? PhysicsWorld::Instance().GetBodyAngularVelocity(bid) : Data().angularVelocity;
}

void Rigidbody::SetAngularVelocity(const glm::vec3 &vel)
{
    if (!IsFinite(vel))
        throw std::invalid_argument("angular velocity must contain finite values");
    auto &data = DataMut();
    data.angularVelocity = vel;
    data.hasAngularVelocity = true;
    auto *go = GetGameObject();
    if (go) {
        auto &pw = PhysicsWorld::Instance();
        const uint32_t bodyId = GetBodyId();
        if (bodyId != 0xFFFFFFFF) {
            pw.SetBodyAngularVelocity(bodyId, vel);
            data.hasAngularVelocity = false;
        }
    }
    PhysicsECSStore::Instance().NotifyCollisionSceneChanged();
}

// ============================================================================
// Forces
// ============================================================================

void Rigidbody::AddForce(const glm::vec3 &force, ForceMode mode)
{
    if (!IsFinite(force))
        throw std::invalid_argument("force must contain finite values");
    ValidateForceMode(mode);
    SubmitForceCommand(ForceCommand{ForceCommandKind::Force, force, glm::vec3(0.0f), mode});
}

void Rigidbody::AddTorque(const glm::vec3 &torque, ForceMode mode)
{
    if (!IsFinite(torque))
        throw std::invalid_argument("torque must contain finite values");
    ValidateForceMode(mode);
    SubmitForceCommand(ForceCommand{ForceCommandKind::Torque, torque, glm::vec3(0.0f), mode});
}

// ============================================================================
// AddForceAtPosition
// ============================================================================

void Rigidbody::AddForceAtPosition(const glm::vec3 &force, const glm::vec3 &position, ForceMode mode)
{
    if (!IsFinite(force) || !IsFinite(position))
        throw std::invalid_argument("force and position must contain finite values");
    ValidateForceMode(mode);
    SubmitForceCommand(ForceCommand{ForceCommandKind::ForceAtPosition, force, position, mode});
}

void Rigidbody::SubmitForceCommand(ForceCommand command)
{
    if (command.value == glm::vec3(0.0f))
        return;
    const uint32_t bodyId = GetBodyId();
    auto &world = PhysicsWorld::Instance();
    if (!world.IsBodyInBroadphase(bodyId)) {
        m_pendingForceCommands.push_back(std::move(command));
        return;
    }
    ApplyForceCommand(world, bodyId, command);
}

void Rigidbody::ApplyForceCommand(PhysicsWorld &world, uint32_t bodyId, const ForceCommand &command)
{
    const float mass = Data().mass;
    if (command.kind == ForceCommandKind::Force) {
        switch (command.mode) {
        case ForceMode::Force:
            world.AddBodyForce(bodyId, command.value);
            return;
        case ForceMode::Acceleration:
            world.AddBodyForce(bodyId, command.value * mass);
            return;
        case ForceMode::Impulse:
            world.AddBodyImpulse(bodyId, command.value);
            return;
        case ForceMode::VelocityChange:
            world.AddBodyImpulse(bodyId, command.value * mass);
            return;
        }
    }

    if (command.kind == ForceCommandKind::Torque) {
        switch (command.mode) {
        case ForceMode::Force:
            world.AddBodyTorque(bodyId, command.value);
            return;
        case ForceMode::Acceleration:
            world.AddBodyTorque(bodyId, world.GetBodyWorldSpaceInertiaTensor(bodyId) * command.value);
            return;
        case ForceMode::Impulse:
            world.AddBodyAngularImpulse(bodyId, command.value);
            return;
        case ForceMode::VelocityChange:
            world.AddBodyAngularImpulse(bodyId, world.GetBodyWorldSpaceInertiaTensor(bodyId) * command.value);
            return;
        }
    }

    switch (command.mode) {
    case ForceMode::Force:
        world.AddBodyForceAtPosition(bodyId, command.value, command.position);
        return;
    case ForceMode::Acceleration:
        world.AddBodyForceAtPosition(bodyId, command.value * mass, command.position);
        return;
    case ForceMode::Impulse:
        world.AddBodyImpulseAtPosition(bodyId, command.value, command.position);
        return;
    case ForceMode::VelocityChange:
        world.AddBodyImpulseAtPosition(bodyId, command.value * mass, command.position);
        return;
    }
    throw std::logic_error("validated force command has an invalid mode");
}

void Rigidbody::FlushPendingForceCommands()
{
    if (m_pendingForceCommands.empty())
        return;
    const uint32_t bodyId = GetBodyId();
    if (bodyId == 0xFFFFFFFF)
        throw std::logic_error("cannot flush force commands without a physics body");

    auto &world = PhysicsWorld::Instance();
    if (!world.IsBodyInBroadphase(bodyId))
        return;

    auto commands = std::move(m_pendingForceCommands);
    m_pendingForceCommands.clear();
    for (const auto &command : commands)
        ApplyForceCommand(world, bodyId, command);
}

// ============================================================================
// Kinematic movement
// ============================================================================

void Rigidbody::MovePosition(const glm::vec3 &position)
{
    if (!IsFinite(position))
        throw std::invalid_argument("position must contain finite values");
    if (!Data().isKinematic)
        throw std::logic_error("MovePosition requires a kinematic Rigidbody");

    GameObject *go = nullptr;
    auto *pw = GetActivePhysicsWorld(go);
    if (!pw)
        throw std::logic_error("MovePosition requires an active physics world");

    // Use current fixed timestep from SceneManager.
    float dt = SceneManager::Instance().GetFixedTimeStep();

    const uint32_t bodyId = GetBodyId();
    if (bodyId == 0xFFFFFFFF)
        throw std::logic_error("MovePosition requires a resident Rigidbody");

    pw->MoveBodyKinematicPosition(bodyId, position, dt);
}

void Rigidbody::MoveRotation(const glm::quat &rotation)
{
    if (!IsFiniteRotation(rotation))
        throw std::invalid_argument("rotation must be a finite non-zero quaternion");
    if (!Data().isKinematic)
        throw std::logic_error("MoveRotation requires a kinematic Rigidbody");

    GameObject *go = nullptr;
    auto *pw = GetActivePhysicsWorld(go);
    if (!pw)
        throw std::logic_error("MoveRotation requires an active physics world");

    float dt = SceneManager::Instance().GetFixedTimeStep();
    const uint32_t bodyId = GetBodyId();
    if (bodyId == 0xFFFFFFFF)
        throw std::logic_error("MoveRotation requires a resident Rigidbody");

    pw->MoveBodyKinematicRotation(bodyId, glm::normalize(rotation), dt);
}

// ============================================================================
// Read-only world info
// ============================================================================

glm::vec3 Rigidbody::GetWorldCenterOfMass() const
{
    uint32_t bid = GetBodyId();
    return (bid != 0xFFFFFFFF) ? PhysicsWorld::Instance().GetBodyCenterOfMassPosition(bid) : glm::vec3(0.0f);
}

PhysicsBodyMotionState Rigidbody::GetMotionState() const
{
    return GetMotionState(PhysicsStateFields::All);
}

PhysicsBodyMotionState Rigidbody::GetMotionState(PhysicsStateFields fields) const
{
    const uint32_t bodyId = GetBodyId();
    auto state = PhysicsWorld::Instance().GetBodyMotionState(bodyId, fields);
    // ECS actor metadata is owner-thread state, not protected by Jolt's body
    // lock. The physics boundary serializes shape publication with this read.
    if (HasPhysicsStateField(fields, PhysicsStateFields::ShapeRevision))
        state.shapeRevision = PhysicsECSStore::Instance().GetActor(Data().actorHandle).shapeRevision;
    return state;
}

void Rigidbody::ResetMotionStates(const std::vector<Rigidbody *> &bodies, std::vector<PhysicsBodyMotionState> states)
{
    auto &world = PhysicsWorld::Instance();
    world.RequireResetPhase(); // Before accessing any scene/component state.
    if (bodies.size() != states.size())
        throw std::invalid_argument("reset requires one motion state per Rigidbody");
    std::unordered_set<uint32_t> identities;
    for (size_t i = 0; i < bodies.size(); ++i) {
        auto *body = bodies[i];
        auto &state = states[i];
        if (!body || !body->IsEnabled() || body->IsDestroyed() || !body->GetGameObject())
            throw std::invalid_argument("reset requires live enabled Rigidbodies");
        if (!IsFinite(state.position) || !IsFiniteRotation(state.rotation) || !IsFinite(state.linearVelocity) ||
            !IsFinite(state.angularVelocity))
            throw std::invalid_argument("reset requires finite poses/velocities and nonzero rotations");
        state.bodyId = body->GetMotionState(PhysicsStateFields::None).bodyId;
        if (!identities.insert(state.bodyId).second)
            throw std::invalid_argument("reset cannot contain duplicate bodies");
        state.rotation = glm::quat(glm::normalize(glm::dquat(state.rotation)));
    }
    world.ResetBodyMotionStates(states);
    // Initialize all physical caches before publishing any parent Transform.
    for (size_t i = 0; i < bodies.size(); ++i) {
        auto &data = bodies[i]->DataMut();
        const auto &state = states[i];
        ++bodies[i]->m_resultEpoch;
        bodies[i]->m_pendingForceCommands.clear();
        data.linearVelocity = state.linearVelocity;
        data.angularVelocity = state.angularVelocity;
        data.hasLinearVelocity = data.hasAngularVelocity = false;
        data.previousPhysicsPosition = data.currentPhysicsPosition = state.position;
        data.previousPhysicsRotation = data.currentPhysicsRotation = state.rotation;
        data.hasPhysicsPose = true;
        data.wasSleeping = false;
    }
    std::vector<std::pair<size_t, size_t>> publicationOrder;
    publicationOrder.reserve(bodies.size());
    for (size_t i = 0; i < bodies.size(); ++i) {
        size_t depth = 0;
        for (auto *parent = bodies[i]->GetGameObject()->GetParent(); parent; parent = parent->GetParent())
            ++depth;
        publicationOrder.emplace_back(depth, i);
    }
    std::sort(publicationOrder.begin(), publicationOrder.end());
    for (const auto &[depth, i] : publicationOrder) {
        auto *body = bodies[i];
        const auto &state = states[i];
        auto *go = body->GetGameObject();
        auto *transform = go->GetTransform();
        transform->ApplyWorldPoseFromPhysics(state.position, state.rotation, true);
        auto &data = body->DataMut();
        data.lastSyncedPosition = state.position;
        data.lastSyncedRotation = transform->GetWorldRotation();
        data.hasSyncedOnce = true;
        body->CacheActorPose(data.lastSyncedPosition, data.lastSyncedRotation);
        body->PreserveDescendantPhysicsPoses();
    }
    if (!bodies.empty())
        SceneManager::Instance().PublishPhysicsTransformsToRenderer();
}

glm::vec3 Rigidbody::GetPosition() const
{
    auto *go = GetGameObject();
    if (!go)
        return glm::vec3(0.0f);

    uint32_t bid = GetBodyId();
    if (bid == 0xFFFFFFFF) {
        if (auto *tf = go->GetTransform())
            return tf->GetPosition();
        return glm::vec3(0.0f);
    }

    return PhysicsWorld::Instance().GetBodyPosition(bid);
}

void Rigidbody::SetPosition(const glm::vec3 &position)
{
    if (!IsFinite(position))
        throw std::invalid_argument("position must contain finite values");
    auto *go = GetGameObject();
    if (!go)
        throw std::runtime_error("cannot set position on a detached Rigidbody");
    auto *transform = go->GetTransform();
    if (!transform)
        throw std::logic_error("Rigidbody owner has no Transform");

    auto &world = PhysicsWorld::Instance();
    const uint32_t bodyId = GetBodyId();
    const bool hasBody = world.IsInitialized() && bodyId != 0xFFFFFFFF;
    const glm::quat rotation =
        hasBody ? glm::normalize(world.GetBodyRotation(bodyId)) : glm::normalize(transform->GetWorldRotation());
    if (hasBody) {
        TeleportBodies(world, go, position, rotation, !SceneManager::Instance().IsPlaying());
    } else {
        transform->SetPosition(position);
        auto &d = DataMut();
        d.previousPhysicsPosition = d.currentPhysicsPosition = position;
        d.previousPhysicsRotation = d.currentPhysicsRotation = rotation;
        d.hasPhysicsPose = true;
        d.wasSleeping = false;
        d.lastSyncedPosition = position;
        d.lastSyncedRotation = rotation;
        d.lastAuthoredPoseRevision = TransformECSStore::Instance().GetAuthoredPoseRevision(transform->GetECSHandle());
        d.hasSyncedOnce = true;
    }
}

glm::quat Rigidbody::GetRotation() const
{
    auto *go = GetGameObject();
    if (!go)
        return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

    uint32_t bid = GetBodyId();
    if (bid == 0xFFFFFFFF) {
        if (auto *tf = go->GetTransform())
            return tf->GetWorldRotation();
        return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    }

    return PhysicsWorld::Instance().GetBodyRotation(bid);
}

void Rigidbody::SetRotation(const glm::quat &rotation)
{
    if (!IsFiniteRotation(rotation))
        throw std::invalid_argument("rotation must be a finite non-zero quaternion");
    auto *go = GetGameObject();
    if (!go)
        throw std::runtime_error("cannot set rotation on a detached Rigidbody");
    auto *transform = go->GetTransform();
    if (!transform)
        throw std::logic_error("Rigidbody owner has no Transform");

    const glm::quat normalized = glm::normalize(rotation);
    auto &world = PhysicsWorld::Instance();
    const uint32_t bodyId = GetBodyId();
    const bool hasBody = world.IsInitialized() && bodyId != 0xFFFFFFFF;
    const glm::vec3 position = hasBody ? world.GetBodyPosition(bodyId) : transform->GetPosition();
    if (hasBody) {
        TeleportBodies(world, go, position, normalized, !SceneManager::Instance().IsPlaying());
    } else {
        transform->SetWorldRotation(normalized);
        auto &d = DataMut();
        d.previousPhysicsPosition = d.currentPhysicsPosition = position;
        d.previousPhysicsRotation = d.currentPhysicsRotation = normalized;
        d.hasPhysicsPose = true;
        d.wasSleeping = false;
        d.lastSyncedPosition = position;
        d.lastSyncedRotation = normalized;
        d.lastAuthoredPoseRevision = TransformECSStore::Instance().GetAuthoredPoseRevision(transform->GetECSHandle());
        d.hasSyncedOnce = true;
    }
}

// ============================================================================
// Sleep
// ============================================================================

bool Rigidbody::IsSleeping() const
{
    uint32_t bid = GetBodyId();
    return (bid != 0xFFFFFFFF) ? PhysicsWorld::Instance().IsBodySleeping(bid) : true;
}

void Rigidbody::WakeUp()
{
    auto *go = GetGameObject();
    if (!go)
        return;

    auto &pw = PhysicsWorld::Instance();
    const uint32_t bodyId = GetBodyId();
    if (bodyId != 0xFFFFFFFF)
        pw.ActivateBody(bodyId);
}

void Rigidbody::Sleep()
{
    auto *go = GetGameObject();
    if (!go)
        return;

    auto &pw = PhysicsWorld::Instance();
    const uint32_t bodyId = GetBodyId();
    if (bodyId != 0xFFFFFFFF)
        pw.DeactivateBody(bodyId);
}

// ============================================================================
// Physics → Transform writeback (called after Jolt step)
// ============================================================================

void Rigidbody::SyncPhysicsToTransform()
{
    auto &d = DataMut();
    auto *go = GetGameObject();
    if (!go)
        return;

    uint32_t bid = GetBodyId();
    if (bid == 0xFFFFFFFF)
        return;

    auto &pw = PhysicsWorld::Instance();

    // Read once on the active -> sleeping edge so the final solver pose is not
    // lost. Only bodies that were already sleeping can skip the Jolt reads.
    const bool sleeping = pw.IsBodySleeping(bid);
    if (sleeping && d.wasSleeping && d.hasPhysicsPose)
        return;
    d.wasSleeping = sleeping;

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

    if (d.interpolation == static_cast<int>(RigidbodyInterpolation::None) || firstPose)
        PublishPhysicsPose(bodyPos, bodyRot);
    else
        CacheActorPose(d.lastSyncedPosition, d.lastSyncedRotation);
}

void Rigidbody::ApplyInterpolatedTransform(float alpha)
{
    auto &d = DataMut();
    if (!d.hasPhysicsPose)
        return;

    if (d.interpolation == static_cast<int>(RigidbodyInterpolation::None)) {
        // A normal fixed-step publication has already written this pose. A
        // mode handoff may still need the complete sample, but must not consume
        // a newer authored Transform before its normal physics boundary.
        const bool rotationFrozen = (d.constraints & static_cast<int>(RigidbodyConstraints::FreezeRotation)) ==
                                    static_cast<int>(RigidbodyConstraints::FreezeRotation);
        if (d.currentPhysicsPosition == d.lastSyncedPosition &&
            (rotationFrozen || !RotationChanged(d.currentPhysicsRotation, d.lastSyncedRotation)))
            return;
        auto *go = GetGameObject();
        auto *transform = go ? go->GetTransform() : nullptr;
        if (!transform || TransformECSStore::Instance().GetAuthoredPoseRevision(transform->GetECSHandle()) !=
                              d.lastAuthoredPoseRevision)
            return;
        // Actor authoring dirtiness includes ancestors. Do not infer it from
        // world/local floating-point reconstruction differences.
        auto &physics = PhysicsECSStore::Instance();
        if (!physics.IsValid(d.actorHandle) || physics.GetActor(d.actorHandle).centerTransformDirty)
            return;
        PublishPhysicsPose(d.currentPhysicsPosition, d.currentPhysicsRotation);
        return;
    }

    const float t = std::clamp(alpha, 0.0f, 1.0f);
    const glm::vec3 presentedPos = glm::mix(d.previousPhysicsPosition, d.currentPhysicsPosition, t);
    const glm::quat presentedRot = glm::normalize(
        glm::slerp(glm::normalize(d.previousPhysicsRotation), glm::normalize(d.currentPhysicsRotation), t));
    PublishPhysicsPose(presentedPos, presentedRot);
}

void Rigidbody::PublishPhysicsPose(const glm::vec3 &presentedPos, const glm::quat &presentedRot)
{
    auto &d = DataMut();
    auto *go = GetGameObject();
    if (!go)
        return;

    Transform *tf = go->GetTransform();
    if (!tf)
        return;

    const bool rotFrozen = (d.constraints & static_cast<int>(RigidbodyConstraints::FreezeRotation)) ==
                           static_cast<int>(RigidbodyConstraints::FreezeRotation);

    tf->ApplyWorldPoseFromPhysics(presentedPos, presentedRot, !rotFrozen);

    d.lastSyncedPosition = presentedPos;
    // Read back reconstructed rotation so the cache matches what
    // SyncExternalMovesToPhysics() will see via GetWorldRotation().
    d.lastSyncedRotation = tf->GetWorldRotation();
    d.lastAuthoredPoseRevision = TransformECSStore::Instance().GetAuthoredPoseRevision(tf->GetECSHandle());
    d.hasSyncedOnce = true;

    CacheActorPose(presentedPos, d.lastSyncedRotation);
    PreserveDescendantPhysicsPoses();
}

void Rigidbody::PreserveDescendantPhysicsPoses()
{
    auto *go = GetGameObject();
    if (!go)
        return;
    auto &transforms = TransformECSStore::Instance();
    const auto preserve = [&](auto &&self, GameObject *child) -> void {
        if (!child->IsActive())
            return;
        auto *body = child->GetComponent<Rigidbody>();
        if (body && body->IsEnabled() && !body->IsKinematic() && body->Data().hasPhysicsPose && body->HasBody()) {
            const auto handle = child->GetTransform()->GetECSHandle();
            // An already published solver/interpolated or authored target is
            // authoritative. Do not replace it with an older interpolation
            // sample. Its own publication already handled its descendants.
            if (!transforms.IsFrameCacheActiveFor(handle) || !transforms.HasFrameCacheWorldPoseOverride(handle))
                body->PublishPhysicsPose(body->Data().currentPhysicsPosition, body->Data().currentPhysicsRotation);
            return;
        }
        for (size_t i = 0; i < child->GetChildCount(); ++i)
            self(self, child->GetChild(i));
    };
    // Only traverse this moving actor's descendants, never the whole physics
    // pool. Cached solver poses need no Jolt read, wake-up or teleport.
    for (size_t i = 0; i < go->GetChildCount(); ++i)
        preserve(preserve, go->GetChild(i));
}

void Rigidbody::SyncExternalMovesToPhysics(float fixedDeltaTime)
{
    auto &d = DataMut();
    auto *go = GetGameObject();
    if (!go)
        return;

    Transform *tf = go->GetTransform();
    if (!tf)
        return;

    glm::vec3 currentPos = tf->GetPosition();
    glm::quat currentRot = tf->GetWorldRotation();
    const auto authoredRevision = TransformECSStore::Instance().GetAuthoredPoseRevision(tf->GetECSHandle());

    const float posEps = 1e-4f;

    // First frame: initialise cache from current Transform.
    // Also check whether the script already moved the Transform away from
    // the body position (e.g. instantiate then set position).  If so,
    // teleport the body immediately instead of waiting one more tick.
    if (!d.hasSyncedOnce) {
        d.lastAuthoredPoseRevision = authoredRevision;
        d.lastSyncedPosition = currentPos;
        d.lastSyncedRotation = currentRot;
        d.hasSyncedOnce = true;

        auto &pw = PhysicsWorld::Instance();
        if (!pw.IsInitialized())
            return;

        const uint32_t bodyId = GetBodyId();
        if (bodyId == 0xFFFFFFFF)
            return;

        glm::vec3 bodyPos = pw.GetBodyPosition(bodyId);
        const glm::quat bodyRot = glm::normalize(pw.GetBodyRotation(bodyId));
        bool firstFrameDiff = glm::length(currentPos - bodyPos) > posEps || RotationChanged(currentRot, bodyRot);
        if (!firstFrameDiff)
            return;

        if (d.isKinematic && fixedDeltaTime > 0.0f) {
            pw.MoveBodyKinematic(bodyId, currentPos, currentRot, fixedDeltaTime, PhysicsWorld::kMaxTransformDriveSpeed);
        } else {
            // A direct Transform assignment outside the fixed simulation
            // phase is an authored teleport. Resetting both interpolation
            // samples prevents the previous physics pose from snapping the
            // object back before the next fixed step.
            TeleportBodies(pw, go, currentPos, currentRot);
        }
        return;
    }

    const bool positionAuthored = authoredRevision[0] != d.lastAuthoredPoseRevision[0];
    const bool rotationAuthored = authoredRevision[1] != d.lastAuthoredPoseRevision[1];
    bool posDiff =
        positionAuthored ? currentPos != d.lastSyncedPosition : glm::length(currentPos - d.lastSyncedPosition) > posEps;
    bool rotDiff =
        rotationAuthored ? currentRot != d.lastSyncedRotation : RotationChanged(currentRot, d.lastSyncedRotation);

    if (!posDiff && !rotDiff) {
        d.lastAuthoredPoseRevision = authoredRevision;
        return;
    }

    auto &pw = PhysicsWorld::Instance();
    if (!pw.IsInitialized())
        return;

    const uint32_t bodyId = GetBodyId();
    if (bodyId == 0xFFFFFFFF)
        return;

    if (d.isKinematic && fixedDeltaTime > 0.0f) {
        // A kinematic Transform animated into a fixed step should carry
        // contact velocity, matching Unity's script-driven kinematic bodies.
        pw.MoveBodyKinematic(bodyId, currentPos, currentRot, fixedDeltaTime, PhysicsWorld::kMaxTransformDriveSpeed);
        d.lastSyncedPosition = currentPos;
        d.lastSyncedRotation = currentRot;
        CacheActorPose(currentPos, currentRot);
    } else {
        // Dynamic bodies and frame-authored kinematic poses use teleport
        // semantics. TeleportBodies also resets interpolation history.
        TeleportBodies(pw, go, currentPos, currentRot);
    }

    d.lastAuthoredPoseRevision = authoredRevision;
    // Update cache
    d.lastSyncedPosition = currentPos;
    d.lastSyncedRotation = currentRot;
}

bool Rigidbody::HasBody() const
{
    auto *go = GetGameObject();
    if (!go)
        return false;

    return GetBodyId() != 0xFFFFFFFF;
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

void Rigidbody::NotifyCollidersBodyTypeChanged(bool refreshShape)
{
    auto &store = PhysicsECSStore::Instance();
    const auto handle = Data().actorHandle;
    if (!store.IsValid(handle))
        return;
    if (!Data().isKinematic) {
        for (auto *collider : store.GetActor(handle).colliders) {
            auto *mesh = dynamic_cast<MeshCollider *>(collider);
            if (mesh && mesh->IsEnabled() && !mesh->IsConvex())
                mesh->SetConvex(true);
        }
    }
    const auto bodyId = GetBodyId();
    if (refreshShape || bodyId == 0xFFFFFFFF)
        store.RefreshActor(handle);
    else {
        // A motion-mode edit leaves the installed shapes and their identities
        // intact. Actual mesh convexification above publishes its own geometry.
        auto &world = PhysicsWorld::Instance();
        world.SetBodyMotionType(bodyId, IsEnabled() ? (Data().isKinematic ? 1 : 2) : 0);
        if (IsEnabled())
            ApplyConfigurationToBody(bodyId);
    }
    if (GetBodyId() != 0xFFFFFFFF)
        FlushPendingForceCommands();
}

void Rigidbody::CacheActorPose(const glm::vec3 &position, const glm::quat &rotation)
{
    auto &store = PhysicsECSStore::Instance();
    if (store.IsValid(Data().actorHandle)) {
        auto &actor = store.GetActor(Data().actorHandle);
        actor.lastSyncedPos = position;
        actor.lastSyncedRot = rotation;
    }
}

void Rigidbody::ApplyConfigurationToBody(uint32_t bodyId)
{
    if (bodyId == 0xFFFFFFFF)
        throw std::invalid_argument("cannot configure an invalid physics body");
    auto &world = PhysicsWorld::Instance();
    if (!world.IsInitialized())
        throw std::runtime_error("cannot configure a body before PhysicsWorld initialization");

    auto &data = DataMut();
    if (!data.hasSyncedOnce) {
        auto *transform = GetGameObject()->GetTransform();
        data.lastSyncedPosition = transform->GetPosition();
        data.lastSyncedRotation = transform->GetWorldRotation();
        data.lastAuthoredPoseRevision =
            TransformECSStore::Instance().GetAuthoredPoseRevision(transform->GetECSHandle());
        data.hasSyncedOnce = true;
    }
    world.SetBodyDamping(bodyId, data.drag, data.angularDrag);
    world.SetBodyGravityFactor(bodyId, data.useGravity ? 1.0f : 0.0f);
    const int allowedDofs = 0x3F & ~(data.constraints >> 1);
    world.SetBodyAllowedDOFs(bodyId, allowedDofs, data.mass);
    world.SetBodyMotionQuality(bodyId,
                               MapCollisionDetectionModeToMotionQuality(data.collisionDetectionMode, data.isKinematic));
    world.SetBodyMaxAngularVelocity(bodyId, data.maxAngularVelocity);
    world.SetBodyMaxLinearVelocity(bodyId, data.maxLinearVelocity);
    if (data.hasLinearVelocity) {
        world.SetBodyLinearVelocity(bodyId, data.linearVelocity);
        data.hasLinearVelocity = false;
    }
    if (data.hasAngularVelocity) {
        world.SetBodyAngularVelocity(bodyId, data.angularVelocity);
        data.hasAngularVelocity = false;
    }
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

nlohmann::json Rigidbody::SerializeDocument() const
{
    const auto &d = Data();
    auto j = Component::SerializeDocument();
    j["mass"] = d.mass;
    if (m_massProperties) {
        const auto &p = *m_massProperties;
        j["mass_properties"] = {{"center", {p.center.x, p.center.y, p.center.z}},
                                {"moments", {p.moments.x, p.moments.y, p.moments.z}},
                                {"rotation", {p.rotation.x, p.rotation.y, p.rotation.z, p.rotation.w}}};
    }
    j["drag"] = d.drag;
    j["angular_drag"] = d.angularDrag;
    j["use_gravity"] = d.useGravity;
    j["is_kinematic"] = d.isKinematic;
    j["constraints"] = d.constraints;
    j["collision_detection_mode"] = d.collisionDetectionMode;
    j["interpolation"] = d.interpolation;
    j["max_angular_velocity"] = d.maxAngularVelocity;
    j["max_linear_velocity"] = d.maxLinearVelocity;
    return j;
}

void Rigidbody::ValidateSerializedDocument(const nlohmann::json &j)
{
    using namespace component_document_validation;
    ValidateComponentDocument(j, "Rigidbody",
                              {"mass", "drag", "angular_drag", "use_gravity", "is_kinematic", "constraints",
                               "collision_detection_mode", "interpolation", "max_angular_velocity",
                               "max_linear_velocity"},
                              {"mass_properties"});
    (void)ReadMassProperties(j);

    const float mass = RequireFiniteFloat(j, "mass", "Rigidbody");
    const float drag = RequireFiniteFloat(j, "drag", "Rigidbody");
    const float angularDrag = RequireFiniteFloat(j, "angular_drag", "Rigidbody");
    RequireBoolean(j, "use_gravity", "Rigidbody");
    RequireBoolean(j, "is_kinematic", "Rigidbody");
    const int constraints = RequireInteger(j, "constraints", "Rigidbody");
    const int collisionDetectionMode = RequireInteger(j, "collision_detection_mode", "Rigidbody");
    const int interpolation = RequireInteger(j, "interpolation", "Rigidbody");
    const float maxAngularVelocity = RequireFiniteFloat(j, "max_angular_velocity", "Rigidbody");
    const float maxLinearVelocity = RequireFiniteFloat(j, "max_linear_velocity", "Rigidbody");

    if (mass < 0.001f)
        throw std::invalid_argument("Rigidbody.mass must be at least 0.001");
    if (drag < 0.0f || angularDrag < 0.0f)
        throw std::invalid_argument("Rigidbody drag values must be non-negative");
    constexpr int kValidConstraintBits = static_cast<int>(RigidbodyConstraints::FreezeAll);
    if (constraints < 0 || (constraints & ~kValidConstraintBits) != 0)
        throw std::invalid_argument("Rigidbody.constraints contains unsupported bits");
    if (collisionDetectionMode < static_cast<int>(CollisionDetectionMode::Discrete) ||
        collisionDetectionMode > static_cast<int>(CollisionDetectionMode::ContinuousDynamic))
        throw std::invalid_argument("Rigidbody.collision_detection_mode is unsupported");
    if (interpolation < static_cast<int>(RigidbodyInterpolation::None) ||
        interpolation > static_cast<int>(RigidbodyInterpolation::Interpolate))
        throw std::invalid_argument("Rigidbody.interpolation is unsupported");
    if (maxAngularVelocity < 0.0f || maxLinearVelocity < 0.0f)
        throw std::invalid_argument("Rigidbody velocity limits must be non-negative");
}

bool Rigidbody::DeserializeDocument(const nlohmann::json &j)
{
    try {
        ValidateSerializedDocument(j);

        RigidbodyECSData staged = Data();
        staged.mass = j.at("mass").get<float>();
        const auto properties = ReadMassProperties(j);
        staged.drag = j.at("drag").get<float>();
        staged.angularDrag = j.at("angular_drag").get<float>();
        staged.useGravity = j.at("use_gravity").get<bool>();
        staged.isKinematic = j.at("is_kinematic").get<bool>();
        staged.constraints = j.at("constraints").get<int>();
        staged.collisionDetectionMode = j.at("collision_detection_mode").get<int>();
        staged.interpolation = j.at("interpolation").get<int>();
        staged.maxAngularVelocity = j.at("max_angular_velocity").get<float>();
        staged.maxLinearVelocity = j.at("max_linear_velocity").get<float>();

        PhysicsWorld::Instance().ValidateBodyMassProperties(*this, staged.mass, properties ? &*properties : nullptr);

        if (!Component::DeserializeDocument(j))
            return false;
        DataMut() = staged;
        m_massProperties = properties;

        // Propagate all settings to Jolt bodies (e.g. when edited in Inspector during play)
        NotifyCollidersBodyTypeChanged();

        return true;
    } catch (const std::exception &e) {
        INXLOG_ERROR("Rigidbody::Deserialize failed: ", e.what());
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
    clone->m_massProperties = m_massProperties;
    dst.drag = src.drag;
    dst.angularDrag = src.angularDrag;
    dst.useGravity = src.useGravity;
    dst.isKinematic = src.isKinematic;
    dst.constraints = src.constraints;
    dst.collisionDetectionMode = src.collisionDetectionMode;
    dst.interpolation = src.interpolation;
    dst.maxAngularVelocity = src.maxAngularVelocity;
    dst.maxLinearVelocity = src.maxLinearVelocity;
    return clone;
}

} // namespace infernux
