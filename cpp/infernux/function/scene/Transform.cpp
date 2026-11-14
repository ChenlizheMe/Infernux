#include "Transform.h"
#include "ComponentDocumentValidation.h"
#include "ComponentFactory.h"
#include "GameObject.h"
#include "RectTransformLayout.h"
#include "Rigidbody.h"
#include "RuntimeWorld.h"
#include "Scene.h"
#include "SceneManager.h"
#include "TransformECSStore.h"
#include "core/math/TransformMatrix.h"
#include "core/math/TransformVectorMath.h"
#include "physics/PhysicsECSStore.h"
#include "physics/PhysicsWorld.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <core/log/InxLog.h>
#include <glm/gtc/type_ptr.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace infernux
{

namespace
{
glm::mat4 ComposeAffineMatrices(const float *left, const float *right)
{
    const auto result = transform_math::ComposeAffine(left, right);
    return glm::make_mat4(result.data());
}

glm::vec3 RotateLocalVector(const glm::quat &rotation, const glm::vec3 &vector)
{
    const auto value =
        transform_math::RotateLocal({rotation.x, rotation.y, rotation.z, rotation.w}, {vector.x, vector.y, vector.z});
    return {value[0], value[1], value[2]};
}

glm::vec3 InverseLocalScale(const glm::vec3 &scale)
{
    const auto inverse = transform_math::InverseLocalScale({scale.x, scale.y, scale.z});
    return {inverse[0], inverse[1], inverse[2]};
}

glm::quat NormalizeStoredRotation(const glm::quat &value)
{
    const auto normalized = rotation::NormalizeTransform(value.x, value.y, value.z, value.w);
    return glm::quat(normalized[3], normalized[0], normalized[1], normalized[2]);
}

glm::quat MultiplyStoredRotation(const glm::quat &left, const glm::quat &right)
{
    const auto result =
        rotation::MultiplyTransform({left.x, left.y, left.z, left.w}, {right.x, right.y, right.z, right.w});
    return glm::quat(result[3], result[0], result[1], result[2]);
}

SemanticTypeDescriptor DescribeTransform()
{
    SemanticTypeDescriptor type;
    // Use the existing native identity namespace; Transform stays in the
    // intrinsic scene document slot, not in the attachable component array.
    type.typeGuid = "native:infernux.Transform";
    type.readableId = "infernux.component.transform";
    type.owner = "engine:native";
    type.origin = "native";
    type.displayName = "Transform";
    type.runtimeProfiles = {"editor", "player", "headless"};
    const auto vector = [](const char *name, const char *serializedName, const char *label, double initial) {
        return SemanticFieldDescriptor{std::string("Transform.") + name,
                                       "FieldType.VEC3",
                                       false,
                                       {{"field_id", name},
                                        {"serialized_name", serializedName},
                                        {"serialized", true},
                                        {"hidden", false},
                                        {"nullable", false},
                                        {"storage_kind", "native_property"},
                                        {"display_name", label},
                                        {"default", {initial, initial, initial}}}};
    };
    type.fields = {vector("local_position", "position", "Position", 0.0),
                   vector("local_euler_angles", "rotation", "Rotation", 0.0),
                   vector("local_scale", "scale", "Scale", 1.0)};
    type.fields[1].attributes["unit"] = "degrees";
    return type;
}

const bool registeredTransform = ComponentFactory::Register(
    "Transform", [] { return std::make_unique<Transform>(); }, Transform::ValidateSerializedDocument,
    Transform::GetTypeConstraints(), DescribeTransform);
} // namespace

Transform::Transform() : m_ecsHandle(TransformECSStore::Instance().Allocate(this))
{
}

Transform::~Transform()
{
    SceneManager::Instance().ForgetSkeletonBone(this);
    TransformECSStore::Instance().Release(m_ecsHandle);
}

void Transform::EnsureCpuPhysicsPose() const
{
    if (!TransformECSStore::Instance().IsPhysicsPoseReadSuppressed())
        PhysicsWorld::Instance().EnsureCpuPhysicsPose(*const_cast<Transform *>(this));
}

bool Transform::HasSimulatedLocalPose() const
{
    const auto *body = m_gameObject ? m_gameObject->GetComponent<Rigidbody>() : nullptr;
    if (!body || !body->IsEnabled() || body->GetBodyId() == UINT32_MAX)
        return false;
    const auto &store = PhysicsECSStore::Instance();
    const auto actor = store.GetRigidbody(body->GetECSHandle()).actorHandle;
    return store.GetActor(actor).runtimeWorld->simulation.isPlaying;
}

void PhysicsWorld::EnsureCpuPhysicsPose(Transform &requested)
{
    requested.ResolveCpuPhysicsPose(*this, false);
}

void PhysicsWorld::MaterializeCpuPoseForRelease(Transform &requested)
{
    requested.ResolveCpuPhysicsPose(*this, true);
}

void Transform::ResolveCpuPhysicsPose(PhysicsWorld &world, bool release)
{
    auto &transforms = TransformECSStore::Instance();
    if (!world.IsInitialized() || (!release && transforms.IsPhysicsPoseReadSuppressed()))
        return;

    auto &physics = PhysicsECSStore::Instance();
    auto *object = GetGameObject();
    auto actorHandle = physics.FindActor(object);
    while (object && !physics.IsValid(actorHandle)) {
        object = object->GetParent();
        actorHandle = physics.FindActor(object);
    }
    if (!object || !physics.IsValid(actorHandle))
        return;
    const WorldExecutionScope ownerScope(physics.GetActor(actorHandle).runtimeWorld);
    auto *body = physics.GetActor(actorHandle).rigidbody;
    if (!body || object->IsDestroying() || physics.GetActor(actorHandle).bodyId == UINT32_MAX ||
        (!release && (!body->IsEnabled() || physics.GetActor(actorHandle).sceneSuspended)))
        return;
    // Ordinary and editor-only transforms have no simulated pose. Resolve that
    // dependency before consulting lifecycle state: the editor camera itself is
    // initialized while SceneManager's singleton is still being constructed.
    if (!SceneManager::Instance().IsPlaying())
        return;
    const auto bodyHandle = body->GetECSHandle();
    const auto bodyId = physics.GetActor(actorHandle).bodyId;
    auto *target = object->GetTransform();
    // Independent physical children retain their world pose. Resolve only
    // their physical ancestors needed for world-to-local conversion.
    if (auto *parent = target->GetParentTransformSafe()) {
        if (release)
            parent->ResolveCpuPhysicsPose(world, true);
        else
            parent->EnsureCpuPhysicsPose();
    }

    const auto generation = world.GetQueryGeneration();
    const auto epoch = SceneManager::Instance().GetSimulationEpoch();
    if (!release && !world.HasPendingStep() && !world.HasPendingControl() &&
        target->m_cpuPhysicsPoseGeneration == generation && target->m_cpuPhysicsPoseBodyId == bodyId &&
        target->m_cpuPhysicsPoseEpoch == epoch)
        return;

    const auto transformHandle = target->GetECSHandle();
    const auto authoredBefore = transforms.GetAuthoredPoseRevision(transformHandle);
    const auto &before = physics.GetRigidbody(bodyHandle);
    const bool positionAuthored = authoredBefore[0] != before.lastAuthoredPoseRevision[0];
    const bool rotationAuthored = authoredBefore[1] != before.lastAuthoredPoseRevision[1];
    // Ancestor/scale/reparent writes need the same protection as local pose
    // writes; their world interpretation is not encoded by two local counters.
    const auto &currentBefore = physics.GetActor(actorHandle);
    const bool hierarchyAuthored = (currentBefore.centerTransformDirty || currentBefore.shapeTransformDirty) &&
                                   !positionAuthored && !rotationAuthored;

    constexpr auto fields = static_cast<PhysicsStateFields>(static_cast<uint32_t>(PhysicsStateFields::Position) |
                                                            static_cast<uint32_t>(PhysicsStateFields::Rotation));
    const auto sample = [&] {
        TransformECSStore::PhysicsPoseReadScope reading(transforms);
        return world.GetBodyMotionState(bodyId, fields);
    }();
    // Completing the query may also publish an authored parent boundary.
    // Rebase against that parent before caching this independent body's local
    // pose, otherwise the parent's subsequent lazy read would move it twice.
    if (auto *parent = target->GetParentTransformSafe()) {
        if (release)
            parent->ResolveCpuPhysicsPose(world, true);
        else
            parent->EnsureCpuPhysicsPose();
    }
    if (!physics.IsValid(actorHandle) || !physics.IsValid(bodyHandle) || !transforms.IsValid(transformHandle) ||
        physics.GetActor(actorHandle).bodyId != bodyId)
        return;
    TransformECSStore::PhysicsPoseReadScope applying(transforms);
    const auto authoredAfter = transforms.GetAuthoredPoseRevision(transformHandle);
    const auto &after = physics.GetRigidbody(bodyHandle);
    const auto &currentActor = physics.GetActor(actorHandle);
    // The query can complete an old target or accept a newer one. Read live
    // guards only after the query and the second parent-pose resolution.
    const bool pendingPosition = body->HasPendingPhysicsTarget(static_cast<uint8_t>(PhysicsTargetChannels::Position));
    const bool pendingRotation = body->HasPendingPhysicsTarget(static_cast<uint8_t>(PhysicsTargetChannels::Rotation));
    const bool hierarchyPending =
        hierarchyAuthored && (currentActor.centerTransformDirty || currentActor.shapeTransformDirty);
    const bool applyPosition = !pendingPosition && !hierarchyPending && authoredAfter[0] == authoredBefore[0] &&
                               (!positionAuthored || after.lastAuthoredPoseRevision[0] == authoredAfter[0]);
    const bool rotationFrozen = (after.constraints & static_cast<int>(RigidbodyConstraints::FreezeRotation)) ==
                                static_cast<int>(RigidbodyConstraints::FreezeRotation);
    const bool applyRotation = !pendingRotation && !hierarchyPending && !rotationFrozen &&
                               authoredAfter[1] == authoredBefore[1] &&
                               (!rotationAuthored || after.lastAuthoredPoseRevision[1] == authoredAfter[1]);
    const auto position = applyPosition ? sample.position : target->GetWorldPosition();
    const auto rotation = applyRotation ? sample.rotation : target->GetWorldRotation();
    if (applyPosition || applyRotation) {
        target->ApplyWorldPoseFromPhysics(position, rotation, applyRotation);
        body->CacheCpuPhysicsPose(target->GetWorldPosition(), target->GetWorldRotation(), applyPosition, applyRotation);
    }
    if (release && (hierarchyPending || pendingPosition || pendingRotation || (positionAuthored && !applyPosition) ||
                    (rotationAuthored && !applyRotation))) {
        // A disabled body can leave a collider-only actor behind. Preserve its
        // unconsumed CPU author channels, but compare their next static update
        // against actual device state rather than the older CPU sample.
        auto &handoff = physics.GetActor(actorHandle);
        if (hierarchyPending || pendingPosition || (positionAuthored && !applyPosition))
            handoff.lastSyncedPos = sample.position;
        if (hierarchyPending || pendingRotation || (rotationAuthored && !applyRotation))
            handoff.lastSyncedRot = sample.rotation;
        physics.MarkAuthoredHierarchyDirty(object,
                                           pendingRotation || (rotationAuthored && !applyRotation) ||
                                               (hierarchyPending && (currentActor.ancestorPoseChannels & 2u) != 0));
    }
    target->m_cpuPhysicsPoseGeneration = world.GetQueryGeneration();
    target->m_cpuPhysicsPoseBodyId = bodyId;
    target->m_cpuPhysicsPoseEpoch = epoch;
}

void Transform::SetLocalEulerAngles(const glm::vec3 &euler)
{
    auto &store = TransformECSStore::Instance();
    store.SetLocalEulerAngles(m_ecsHandle, euler);
    store.SetLocalRotation(m_ecsHandle, NormalizeStoredRotation(EulerYXZToQuat(euler)));
    store.RecordLocalPoseWrite(m_ecsHandle, false, true);
    store.SetHasCachedWorldEulerAngles(m_ecsHandle, false);
    store.SetDirty(m_ecsHandle, true);
    InvalidateWorldMatrix(true);
}

void Transform::SetLocalRotation(const glm::quat &rotation)
{
    auto &store = TransformECSStore::Instance();
    const auto normalized = NormalizeStoredRotation(rotation);
    if (!store.HasFrameCacheWorldRotationOverride(m_ecsHandle) && store.GetLocalRotation(m_ecsHandle) == normalized &&
        !HasSimulatedLocalPose())
        return;
    store.SetLocalRotation(m_ecsHandle, normalized);
    store.RecordLocalPoseWrite(m_ecsHandle, false, true);
    store.SetLocalEulerAngles(m_ecsHandle, ExtractEulerAnglesNear(normalized, store.GetLocalEulerAngles(m_ecsHandle)));
    store.SetHasCachedWorldEulerAngles(m_ecsHandle, false);
    store.SetDirty(m_ecsHandle, true);
    InvalidateWorldMatrix(true);
}

Transform *Transform::GetParentTransformSafe() const
{
    if (!m_gameObject || !m_gameObject->GetParent()) {
        return nullptr;
    }
    return m_gameObject->GetParent()->GetTransform();
}

glm::vec3 Transform::GetWorldDirection(const glm::vec3 &localAxis) const
{
    const glm::quat q = GetWorldRotation();
    const auto value = rotation::RotatePublic({q.x, q.y, q.z, q.w}, {localAxis.x, localAxis.y, localAxis.z});
    return glm::vec3(value[0], value[1], value[2]);
}

// ============================================================================
// World-space Position
// ============================================================================

glm::vec3 Transform::GetWorldPosition() const
{
    const auto local = GetLocalPosition();
    const Transform *parent = GetParentTransformSafe();
    return parent ? parent->TransformPoint(local) : local;
}

void Transform::SetWorldPosition(const glm::vec3 &worldPos)
{
    if (!std::isfinite(worldPos.x) || !std::isfinite(worldPos.y) || !std::isfinite(worldPos.z))
        return;
    auto &store = TransformECSStore::Instance();
    if (store.GetRectLayout(m_ecsHandle)) {
        const auto *parent = GetParentTransformSafe();
        SetLocalPosition(parent ? parent->InverseTransformPoint(worldPos) : worldPos);
        return;
    }
    if (store.IsFrameCacheActiveFor(m_ecsHandle)) {
        // Preserve the local pose resolved against the parent at this call.
        store.SetCachedAuthoredWorldPosition(m_ecsHandle.index, worldPos);
        return;
    }

    Transform *parentTransform = GetParentTransformSafe();
    const auto local = parentTransform ? parentTransform->InverseTransformPoint(worldPos) : worldPos;
    store.RecordLocalPoseWrite(m_ecsHandle, true, false);
    store.SetLocalPosition(m_ecsHandle, local);
    store.SetDirty(m_ecsHandle, true);
    InvalidateWorldMatrix(false);
}

void Transform::ApplyWorldPoseFromPhysics(const glm::vec3 &worldPos, const glm::quat &worldRot, bool applyRotation)
{
    auto &store = TransformECSStore::Instance();
    TransformECSStore::PhysicsPoseReadScope applying(store);
    if (store.IsFrameCacheActiveFor(m_ecsHandle)) {
        store.SetCachedWorldPoseFromPhysics(m_ecsHandle.index, worldPos, glm::normalize(worldRot), applyRotation);
        SceneManager::Instance().NotifySkeletonBoneChanged(this);
        return;
    }

    Transform *parentTransform = GetParentTransformSafe();
    if (!parentTransform) {
        store.SetLocalPosition(m_ecsHandle, worldPos);
    } else {
        store.SetLocalPosition(m_ecsHandle, parentTransform->InverseTransformPoint(worldPos));
    }

    if (applyRotation) {
        const glm::quat safeWorldRot = glm::normalize(worldRot);
        const glm::quat localRot = WorldRotationToLocal(safeWorldRot);
        store.SetLocalRotation(m_ecsHandle, localRot);
        store.SetLocalEulerAngles(m_ecsHandle,
                                  ExtractEulerAnglesNear(localRot, store.GetLocalEulerAngles(m_ecsHandle)));
        if (store.GetHasCachedWorldEulerAngles(m_ecsHandle)) {
            store.SetCachedWorldEulerAngles(
                m_ecsHandle, ExtractEulerAnglesNear(safeWorldRot, store.GetCachedWorldEulerAngles(m_ecsHandle)));
        } else {
            store.SetCachedWorldEulerAngles(m_ecsHandle, ExtractEulerAngles(safeWorldRot));
            store.SetHasCachedWorldEulerAngles(m_ecsHandle, true);
        }
    }

    store.SetDirty(m_ecsHandle, true);
    store.InvalidateSubtree(this, applyRotation, this);
    SceneManager::Instance().NotifySkeletonBoneChanged(this);
}

// ============================================================================
// World Matrix
// ============================================================================

const glm::mat4 &Transform::GetWorldMatrix() const
{
    EnsureCpuPhysicsPose();
    auto &store = TransformECSStore::Instance();
    // Staged world writes invalidate this matrix and its descendants. The
    // local getters retain the existing affine interpretation of those writes.
    if (!store.GetWorldMatrixDirty(m_ecsHandle)) {
        return store.GetCachedWorldMatrix(m_ecsHandle);
    }

    glm::mat4 localMatrix = GetLocalMatrix();

    Transform *parentTransform = GetParentTransformSafe();
    if (!parentTransform) {
        store.SetCachedWorldMatrix(m_ecsHandle, localMatrix);
    } else {
        store.SetCachedWorldMatrix(m_ecsHandle, ComposeAffineMatrices(glm::value_ptr(parentTransform->GetWorldMatrix()),
                                                                      glm::value_ptr(localMatrix)));
    }
    store.SetWorldMatrixDirty(m_ecsHandle, false);
    return store.GetCachedWorldMatrix(m_ecsHandle);
}

// ============================================================================
// World Matrix Cache Invalidation
// ============================================================================

void Transform::InvalidateWorldMatrix(bool clearWorldEulerExact) const
{
    TransformECSStore::Instance().InvalidateSubtree(const_cast<Transform *>(this), clearWorldEulerExact);
}

// ============================================================================
// World-space Rotation
// ============================================================================

glm::quat Transform::GetWorldRotation() const
{
    EnsureCpuPhysicsPose();
    auto &store = TransformECSStore::Instance();
    if (store.IsFrameCacheActiveFor(m_ecsHandle) && store.HasFrameCacheWorldRotationOverride(m_ecsHandle)) {
        return store.GetCachedWorldRotation(m_ecsHandle.index);
    }

    return LocalRotationToWorld(GetLocalRotation());
}

glm::quat Transform::GetLocalRotation() const
{
    EnsureCpuPhysicsPose();
    auto &store = TransformECSStore::Instance();
    if (store.IsFrameCacheActiveFor(m_ecsHandle) && store.HasFrameCacheWorldRotationOverride(m_ecsHandle))
        return store.GetFrameDerivedLocalRotation(m_ecsHandle.index);
    return store.GetLocalRotation(m_ecsHandle);
}

glm::quat Transform::LocalRotationToWorld(const glm::quat &local) const
{
    glm::quat rotation = local;
    for (const Transform *parent = GetParentTransformSafe(); parent; parent = parent->GetParentTransformSafe())
        rotation = MultiplyStoredRotation(parent->GetLocalRotation(), rotation);
    return rotation;
}

glm::quat Transform::WorldRotationToLocal(const glm::quat &world) const
{
    const Transform *parent = GetParentTransformSafe();
    if (!parent)
        return world;
    // Invert the hierarchy from the root down. Collapsing its quaternions
    // first or dividing their conjugates by squared length changes storage.
    return MultiplyStoredRotation(glm::conjugate(parent->GetLocalRotation()), parent->WorldRotationToLocal(world));
}

void Transform::SetWorldRotation(const glm::quat &worldRot)
{
    auto &store = TransformECSStore::Instance();
    const glm::quat newLocalRot = NormalizeStoredRotation(WorldRotationToLocal(worldRot));
    if (!store.HasFrameCacheWorldRotationOverride(m_ecsHandle) && store.GetLocalRotation(m_ecsHandle) == newLocalRot &&
        !HasSimulatedLocalPose())
        return;

    store.RecordLocalPoseWrite(m_ecsHandle, false, true);

    store.SetLocalRotation(m_ecsHandle, newLocalRot);
    store.SetLocalEulerAngles(m_ecsHandle, ExtractEulerAnglesNear(newLocalRot, store.GetLocalEulerAngles(m_ecsHandle)));
    if (store.GetHasCachedWorldEulerAngles(m_ecsHandle)) {
        store.SetCachedWorldEulerAngles(m_ecsHandle,
                                        ExtractEulerAnglesNear(worldRot, store.GetCachedWorldEulerAngles(m_ecsHandle)));
    } else {
        store.SetCachedWorldEulerAngles(m_ecsHandle, ExtractEulerAngles(worldRot));
        store.SetHasCachedWorldEulerAngles(m_ecsHandle, true);
    }
    store.SetDirty(m_ecsHandle, true);
    InvalidateWorldMatrix(true);
}

glm::vec3 Transform::GetWorldEulerAngles() const
{
    EnsureCpuPhysicsPose();
    auto &store = TransformECSStore::Instance();

    // If world euler was set directly (not via quaternion), return the exact value
    if (store.GetWorldEulerExact(m_ecsHandle)) {
        return ToPublicEulerAngles(store.GetCachedWorldEulerAngles(m_ecsHandle));
    }

    glm::quat worldRotation = GetWorldRotation();
    if (store.GetHasCachedWorldEulerAngles(m_ecsHandle)) {
        store.SetCachedWorldEulerAngles(
            m_ecsHandle, ExtractEulerAnglesNear(worldRotation, store.GetCachedWorldEulerAngles(m_ecsHandle)));
    } else {
        store.SetCachedWorldEulerAngles(m_ecsHandle, ExtractEulerAngles(worldRotation));
        store.SetHasCachedWorldEulerAngles(m_ecsHandle, true);
    }
    return ToPublicEulerAngles(store.GetCachedWorldEulerAngles(m_ecsHandle));
}

void Transform::SetWorldEulerAngles(const glm::vec3 &euler)
{
    const glm::quat localRot = NormalizeStoredRotation(WorldRotationToLocal(EulerYXZToQuat(euler)));

    auto &store = TransformECSStore::Instance();

    Transform *parentTransform = GetParentTransformSafe();

    // Stored equality is sufficient only for a CPU-authored pose. A simulated
    // body may already have rotated away from that stored value on the device.
    if (store.GetLocalRotation(m_ecsHandle) != localRot || store.HasFrameCacheWorldRotationOverride(m_ecsHandle) ||
        HasSimulatedLocalPose()) {
        store.RecordLocalPoseWrite(m_ecsHandle, false, true);
        store.SetLocalRotation(m_ecsHandle, localRot);
        store.SetDirty(m_ecsHandle, true);
        InvalidateWorldMatrix(true);
    }
    if (!parentTransform) {
        store.SetLocalEulerAngles(m_ecsHandle, euler); // root: local == world, store exact
    } else {
        store.SetLocalEulerAngles(m_ecsHandle,
                                  ExtractEulerAnglesNear(localRot, store.GetLocalEulerAngles(m_ecsHandle)));
    }

    // Set exact world euler AFTER InvalidateWorldMatrix so the cascade doesn't clear it
    store.SetCachedWorldEulerAngles(m_ecsHandle, euler);
    store.SetHasCachedWorldEulerAngles(m_ecsHandle, true);
    store.SetWorldEulerExact(m_ecsHandle, true);
}

// ============================================================================
// World-space Scale (lossyScale). Particles use GetWorldMatrix() instead of
// this vector: direction conversion must not inherit authoring scale.
// ============================================================================

glm::vec3 Transform::GetWorldScale() const
{
    Transform *parentTransform = GetParentTransformSafe();
    if (!parentTransform) {
        return GetLocalScale();
    }

    return parentTransform->GetWorldScale() * GetLocalScale();
}

void Transform::SetWorldScale(const glm::vec3 &worldScale)
{
    auto &store = TransformECSStore::Instance();
    Transform *parentTransform = GetParentTransformSafe();
    if (!parentTransform) {
        store.SetLocalScale(m_ecsHandle, worldScale);
    } else {
        glm::vec3 parentScale = parentTransform->GetWorldScale();
        glm::vec3 local;
        local.x = (std::abs(parentScale.x) > 1e-6f) ? worldScale.x / parentScale.x : worldScale.x;
        local.y = (std::abs(parentScale.y) > 1e-6f) ? worldScale.y / parentScale.y : worldScale.y;
        local.z = (std::abs(parentScale.z) > 1e-6f) ? worldScale.z / parentScale.z : worldScale.z;
        store.SetLocalScale(m_ecsHandle, local);
    }
    store.SetDirty(m_ecsHandle, true);
    InvalidateWorldMatrix(false);
}

// ============================================================================
// World-space Direction Vectors
// ============================================================================

glm::vec3 Transform::GetWorldForward() const
{
    return GetWorldDirection(glm::vec3(0.0f, 0.0f, 1.0f));
}

glm::vec3 Transform::GetWorldRight() const
{
    return GetWorldDirection(glm::vec3(1.0f, 0.0f, 0.0f));
}

glm::vec3 Transform::GetWorldUp() const
{
    return GetWorldDirection(glm::vec3(0.0f, 1.0f, 0.0f));
}

// ============================================================================
// Hierarchy (delegates to owning GameObject)
// ============================================================================

Transform *Transform::GetParent() const
{
    return GetParentTransformSafe();
}

void Transform::SetParent(Transform *parent, bool worldPositionStays)
{
    if (!m_gameObject)
        return;
    GameObject *newParentGO = parent ? parent->GetGameObject() : nullptr;
    m_gameObject->SetParent(newParentGO, worldPositionStays);
}

Transform *Transform::GetRoot()
{
    Transform *current = this;
    while (current->GetParent()) {
        current = current->GetParent();
    }
    return current;
}

size_t Transform::GetChildCount() const
{
    if (!m_gameObject)
        return 0;
    return m_gameObject->GetChildCount();
}

Transform *Transform::GetChild(size_t index) const
{
    if (!m_gameObject)
        return nullptr;
    GameObject *child = m_gameObject->GetChild(index);
    return child ? child->GetTransform() : nullptr;
}

Transform *Transform::Find(const std::string &name) const
{
    if (!m_gameObject)
        return nullptr;
    GameObject *child = m_gameObject->FindChild(name);
    return child ? child->GetTransform() : nullptr;
}

void Transform::DetachChildren()
{
    if (!m_gameObject)
        return;
    // Collect children first to avoid modifying container during iteration
    std::vector<GameObject *> children;
    for (size_t i = 0; i < m_gameObject->GetChildCount(); ++i) {
        children.push_back(m_gameObject->GetChild(i));
    }
    for (auto *child : children) {
        child->SetParent(nullptr, true);
    }
}

bool Transform::IsChildOf(const Transform *parent) const
{
    if (!parent || !m_gameObject)
        return false;
    const Transform *current = GetParent();
    while (current) {
        if (current == parent)
            return true;
        current = current->GetParent();
    }
    return false;
}

int Transform::GetSiblingIndex() const
{
    if (!m_gameObject)
        return 0;
    GameObject *parentGO = m_gameObject->GetParent();
    if (!parentGO) {
        // Root object - check in scene
        if (m_gameObject->GetScene()) {
            auto &roots = m_gameObject->GetScene()->GetRootObjects();
            for (size_t i = 0; i < roots.size(); ++i) {
                if (roots[i].get() == m_gameObject)
                    return static_cast<int>(i);
            }
        }
        return 0;
    }
    auto &siblings = parentGO->GetChildren();
    for (size_t i = 0; i < siblings.size(); ++i) {
        if (siblings[i].get() == m_gameObject)
            return static_cast<int>(i);
    }
    return 0;
}

void Transform::SetSiblingIndex(int index)
{
    if (!m_gameObject)
        return;
    GameObject *parentGO = m_gameObject->GetParent();

    if (parentGO) {
        parentGO->SetChildSiblingIndex(m_gameObject, index);
    } else if (m_gameObject->GetScene()) {
        m_gameObject->GetScene()->SetRootObjectSiblingIndex(m_gameObject, index);
    }
}

void Transform::SetAsFirstSibling()
{
    SetSiblingIndex(0);
}

void Transform::SetAsLastSibling()
{
    SetSiblingIndex(INT_MAX);
}

// ============================================================================
// Space Conversion Methods
// ============================================================================

glm::vec3 Transform::TransformPoint(const glm::vec3 &point) const
{
    auto result = point;
    for (const Transform *node = this; node; node = node->GetParentTransformSafe())
        result = RotateLocalVector(node->GetLocalRotation(), result * node->GetLocalScale()) + node->GetLocalPosition();
    return result;
}

glm::vec3 Transform::InverseTransformPoint(const glm::vec3 &point) const
{
    const Transform *parent = GetParentTransformSafe();
    const auto inParent = parent ? parent->InverseTransformPoint(point) : point;
    return RotateLocalVector(glm::conjugate(GetLocalRotation()), inParent - GetLocalPosition()) *
           InverseLocalScale(GetLocalScale());
}

glm::vec3 Transform::TransformDirection(const glm::vec3 &direction) const
{
    auto result = direction;
    for (const Transform *node = this; node; node = node->GetParentTransformSafe())
        result = RotateLocalVector(node->GetLocalRotation(), result);
    return result;
}

glm::vec3 Transform::InverseTransformDirection(const glm::vec3 &direction) const
{
    const Transform *parent = GetParentTransformSafe();
    const auto inParent = parent ? parent->InverseTransformDirection(direction) : direction;
    return RotateLocalVector(glm::conjugate(GetLocalRotation()), inParent);
}

glm::vec3 Transform::TransformVector(const glm::vec3 &vector) const
{
    auto result = vector;
    for (const Transform *node = this; node; node = node->GetParentTransformSafe())
        result = RotateLocalVector(node->GetLocalRotation(), result * node->GetLocalScale());
    return result;
}

glm::vec3 Transform::InverseTransformVector(const glm::vec3 &vector) const
{
    const Transform *parent = GetParentTransformSafe();
    const auto inParent = parent ? parent->InverseTransformVector(vector) : vector;
    return RotateLocalVector(glm::conjugate(GetLocalRotation()), inParent) * InverseLocalScale(GetLocalScale());
}

// ============================================================================
// World-to-Local Matrix
// ============================================================================

glm::mat4 Transform::GetWorldToLocalMatrix() const
{
    const glm::vec3 position = GetLocalPosition();
    const glm::quat rotation = GetLocalRotation();
    const glm::vec3 scale = GetLocalScale();
    const auto inverse =
        transform_math::InverseTRS({position.x, position.y, position.z},
                                   {rotation.x, rotation.y, rotation.z, rotation.w}, {scale.x, scale.y, scale.z});
    const Transform *parent = GetParentTransformSafe();
    if (parent) {
        const glm::mat4 parentInverse = parent->GetWorldToLocalMatrix();
        return ComposeAffineMatrices(inverse.data(), glm::value_ptr(parentInverse));
    }
    return glm::make_mat4(inverse.data());
}

// ============================================================================
// Translate / TranslateLocal (correct for child objects)
// ============================================================================

void Transform::Translate(const glm::vec3 &delta)
{
    // Unity convention: delta is in world space.
    // SetWorldPosition handles the local ← world conversion internally.
    SetWorldPosition(GetWorldPosition() + delta);
}

void Transform::TranslateLocal(const glm::vec3 &delta)
{
    // Move along the object's own axes (world-space rotation * local delta).
    glm::vec3 worldDelta = GetWorldRotation() * delta;
    Translate(worldDelta);
}

// ============================================================================
// RotateAround (world-space point, Unity convention)
// ============================================================================

void Transform::RotateAround(const glm::vec3 &point, const glm::vec3 &axis, float angle)
{
    // 1. Rotate our world position around the given point
    glm::vec3 worldPos = GetWorldPosition();
    glm::quat rot = glm::angleAxis(glm::radians(angle), glm::normalize(axis));
    glm::vec3 offset = worldPos - point;
    glm::vec3 newPos = point + rot * offset;
    SetWorldPosition(newPos);

    // 2. Rotate our orientation
    SetWorldRotation(rot * GetWorldRotation());
}

// ============================================================================
// LookAt (world-space target, Unity convention)
// ============================================================================

void Transform::LookAt(const glm::vec3 &target, const glm::vec3 &up)
{
    glm::vec3 worldPos = GetWorldPosition();
    glm::vec3 direction = glm::normalize(target - worldPos);

    glm::quat worldRot;
    if (glm::abs(glm::dot(direction, up)) > 0.999f) {
        worldRot = glm::quatLookAt(direction, glm::vec3(0.0f, 0.0f, 1.0f));
    } else {
        worldRot = glm::quatLookAt(direction, up);
    }
    SetWorldRotation(worldRot);
}

// ============================================================================
// Serialization (stores LOCAL transform)
// ============================================================================

nlohmann::json Transform::SerializeDocument() const
{
    json j = Component::SerializeDocument();

    // Position
    glm::vec3 pos = GetLocalPosition();
    const auto *rectState = TransformECSStore::Instance().GetRectLayout(m_ecsHandle);
    if (rectState && (rectState->drive || rectState->drivenParent)) {
        const auto authored = rectState->AuthoringLocalPosition();
        pos.x = authored[0];
        pos.y = authored[1];
    }
    j["position"] = {pos.x, pos.y, pos.z};

    // Rotation (as Euler angles)
    glm::vec3 euler = GetLocalEulerAngles();
    j["rotation"] = {euler.x, euler.y, euler.z};

    // Scale
    glm::vec3 scale = GetLocalScale();
    j["scale"] = {scale.x, scale.y, scale.z};

    if (const auto *layout = TransformECSStore::Instance().GetRectLayout(m_ecsHandle)) {
        const auto spec = layout->AuthoringSpec();
        j["rect_layout"] = {{"anchor_min", spec.anchorMin},
                            {"anchor_max", spec.anchorMax},
                            {"pivot", spec.pivot},
                            {"size_delta", spec.sizeDelta},
                            {"anchored_position", spec.anchoredPosition},
                            {"position_scale", layout->positionScale}};
        if (layout->unitMode != ui::RectUnitMode::Explicit)
            j["rect_layout"]["unit_mode"] = static_cast<int>(layout->unitMode);
    }

    return j;
}

void Transform::ValidateSerializedDocument(const nlohmann::json &document)
{
    using namespace component_document_validation;
    ValidateComponentDocument(document, "Transform", {"position", "rotation", "scale"}, {"rect_layout"});
    RequireFiniteVector(document, "position", 3, "Transform");
    RequireFiniteVector(document, "rotation", 3, "Transform");
    RequireFiniteVector(document, "scale", 3, "Transform");
    if (document.contains("rect_layout")) {
        const auto &layout = document["rect_layout"];
        if (!layout.is_object())
            throw std::invalid_argument("Transform.rect_layout must be an object");
        for (const auto *field : {"anchor_min", "anchor_max", "pivot", "size_delta", "anchored_position"})
            RequireFiniteVector(layout, field, 2, "Transform.rect_layout");
        if (layout.contains("position_scale") &&
            (!layout["position_scale"].is_number() || !std::isfinite(layout["position_scale"].get<float>()) ||
             layout["position_scale"].get<float>() <= 0))
            throw std::invalid_argument("Transform.rect_layout.position_scale must be positive and finite");
        if (layout.contains("unit_mode") && (!layout["unit_mode"].is_number_integer() ||
                                             layout["unit_mode"].get<int>() < 0 || layout["unit_mode"].get<int>() > 2))
            throw std::invalid_argument("Transform.rect_layout.unit_mode is invalid");
    }
}

bool Transform::DeserializeDocument(const nlohmann::json &j)
{
    try {
        ValidateSerializedDocument(j);
        if (!Component::DeserializeDocument(j)) {
            return false;
        }

        if (!j.contains("rect_layout"))
            TransformECSStore::Instance().DisableRectLayout(m_ecsHandle);
        SetLocalPosition(j["position"][0].get<float>(), j["position"][1].get<float>(), j["position"][2].get<float>());
        SetLocalEulerAngles(j["rotation"][0].get<float>(), j["rotation"][1].get<float>(),
                            j["rotation"][2].get<float>());
        SetLocalScale(j["scale"][0].get<float>(), j["scale"][1].get<float>(), j["scale"][2].get<float>());

        auto &store = TransformECSStore::Instance();
        if (j.contains("rect_layout")) {
            auto &state = store.EnableRectLayout(m_ecsHandle);
            state.drive.reset();
            auto &spec = state.spec;
            const auto &layout = j["rect_layout"];
            state.positionScale = layout.value("position_scale", 1.f);
            state.unitMode = static_cast<ui::RectUnitMode>(layout.value("unit_mode", 0));
            spec.anchorMin = layout["anchor_min"].get<ui::RectPoint>();
            spec.anchorMax = layout["anchor_max"].get<ui::RectPoint>();
            spec.pivot = layout["pivot"].get<ui::RectPoint>();
            spec.sizeDelta = layout["size_delta"].get<ui::RectPoint>();
            spec.anchoredPosition = layout["anchored_position"].get<ui::RectPoint>();
            store.MarkRectLayoutInputChanged();
            RectTransformLayout::RefreshSubtree(this);
        } else {
            // Removing a rectangle also removes its direct children's parent
            // frame; their anchored coordinates remain authoritative.
            for (size_t index = 0; index < GetChildCount(); ++index)
                RectTransformLayout::RefreshSubtree(GetChild(index));
        }
        store.SetDirty(m_ecsHandle, true);
        store.SetHasCachedWorldEulerAngles(m_ecsHandle, false);
        store.SetWorldEulerExact(m_ecsHandle, false);
        InvalidateWorldMatrix(false);
        return true;
    } catch (const std::exception &e) {
        INXLOG_ERROR("Transform::Deserialize failed: ", e.what());
        return false;
    }
}

void Transform::CloneDataTo(Transform &target) const
{
    EnsureCpuPhysicsPose();
    auto &store = TransformECSStore::Instance();
    const auto src = store.GetSnapshot(m_ecsHandle);

    TransformECSData dst{};
    dst.localPosition = src.localPosition;
    dst.localEulerAngles = src.localEulerAngles;
    dst.localRotation = src.localRotation;
    dst.localScale = src.localScale;
    dst.dirty = true;
    dst.hasCachedWorldEulerAngles = false;
    dst.worldEulerExact = false;
    dst.worldMatrixDirty = true;
    dst.owner = store.GetOwner(target.m_ecsHandle);
    store.SetSnapshot(target.m_ecsHandle, dst);
    if (const auto *layout = store.GetRectLayout(m_ecsHandle)) {
        auto &copied = store.EnableRectLayout(target.m_ecsHandle);
        copied = *layout;
        copied.spec = layout->AuthoringSpec();
        copied.drive.reset();
        RectTransformLayout::RefreshSubtree(&target);
    } else
        store.DisableRectLayout(target.m_ecsHandle);

    target.SetEnabled(IsEnabled());
}

std::unique_ptr<Component> Transform::Clone() const
{
    // Transform is embedded in GameObject; this should not be called directly.
    // Return nullptr — GameObject::Clone handles transform cloning via CloneDataTo.
    return nullptr;
}

} // namespace infernux
