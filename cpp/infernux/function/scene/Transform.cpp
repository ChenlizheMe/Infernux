#include "Transform.h"
#include "ComponentFactory.h"
#include "GameObject.h"
#include "Scene.h"
#include "TransformECSStore.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace infernux
{

INFERNUX_REGISTER_COMPONENT("Transform", Transform)

Transform::Transform() : m_ecsHandle(TransformECSStore::Instance().Allocate(this))
{
}

Transform::~Transform()
{
    TransformECSStore::Instance().Release(m_ecsHandle);
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
    return glm::normalize(GetWorldRotation() * localAxis);
}

// ============================================================================
// World-space Position
// ============================================================================

glm::vec3 Transform::GetWorldPosition() const
{
    Transform *parentTransform = GetParentTransformSafe();
    if (!parentTransform) {
        return GetLocalPosition();
    }

    glm::mat4 parentWorld = parentTransform->GetWorldMatrix();
    return glm::vec3(parentWorld * glm::vec4(GetLocalPosition(), 1.0f));
}

void Transform::SetWorldPosition(const glm::vec3 &worldPos)
{
    auto &data = TransformECSStore::Instance().Get(m_ecsHandle);
    Transform *parentTransform = GetParentTransformSafe();
    if (!parentTransform) {
        data.localPosition = worldPos;
    } else {
        glm::mat4 invParentWorld = glm::inverse(parentTransform->GetWorldMatrix());
        data.localPosition = glm::vec3(invParentWorld * glm::vec4(worldPos, 1.0f));
    }
    data.dirty = true;
    InvalidateWorldMatrix(false);
}

// ============================================================================
// World Matrix
// ============================================================================

glm::mat4 Transform::GetWorldMatrix() const
{
    auto &store = TransformECSStore::Instance();
    auto &data = store.Get(m_ecsHandle);
    if (!data.worldMatrixDirty) {
        return data.cachedWorldMatrix;
    }

    glm::mat4 localMatrix = GetLocalMatrix();

    Transform *parentTransform = GetParentTransformSafe();
    if (!parentTransform) {
        data.cachedWorldMatrix = localMatrix;
    } else {
        data.cachedWorldMatrix = parentTransform->GetWorldMatrix() * localMatrix;
    }

    data.worldMatrixDirty = false;
    return data.cachedWorldMatrix;
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
    Transform *parentTransform = GetParentTransformSafe();
    if (!parentTransform) {
        return GetLocalRotation();
    }

    return parentTransform->GetWorldRotation() * GetLocalRotation();
}

void Transform::SetWorldRotation(const glm::quat &worldRot)
{
    glm::quat safeRot = glm::normalize(worldRot);
    glm::quat newLocalRot;
    Transform *parentTransform = GetParentTransformSafe();
    if (!parentTransform) {
        newLocalRot = safeRot;
    } else {
        newLocalRot = glm::inverse(parentTransform->GetWorldRotation()) * safeRot;
    }

    auto &data = TransformECSStore::Instance().Get(m_ecsHandle);

    // Skip only when the quaternion is *exactly* the same (bit-equal).
    // A floating-point dot-product threshold (e.g. 0.999999) drops tiny
    // rotations at high frame-rates, causing visible jitter when an
    // object orbits slowly.
    if (data.localRotation == newLocalRot) {
        return;
    }

    data.localRotation = newLocalRot;
    data.localEulerAngles = ExtractEulerAnglesNear(newLocalRot, data.localEulerAngles);
    if (data.hasCachedWorldEulerAngles) {
        data.cachedWorldEulerAngles = ExtractEulerAnglesNear(worldRot, data.cachedWorldEulerAngles);
    } else {
        data.cachedWorldEulerAngles = ExtractEulerAngles(worldRot);
        data.hasCachedWorldEulerAngles = true;
    }
    data.dirty = true;
    InvalidateWorldMatrix(true);
}

glm::vec3 Transform::GetWorldEulerAngles() const
{
    auto &data = TransformECSStore::Instance().Get(m_ecsHandle);

    // If world euler was set directly (not via quaternion), return the exact value
    if (data.worldEulerExact) {
        return ToPublicEulerAngles(data.cachedWorldEulerAngles);
    }

    glm::quat worldRotation = GetWorldRotation();
    if (data.hasCachedWorldEulerAngles) {
        data.cachedWorldEulerAngles = ExtractEulerAnglesNear(worldRotation, data.cachedWorldEulerAngles);
    } else {
        data.cachedWorldEulerAngles = ExtractEulerAngles(worldRotation);
        data.hasCachedWorldEulerAngles = true;
    }
    return ToPublicEulerAngles(data.cachedWorldEulerAngles);
}

void Transform::SetWorldEulerAngles(const glm::vec3 &euler)
{
    glm::quat worldRot = EulerYXZToQuat(euler);

    auto &data = TransformECSStore::Instance().Get(m_ecsHandle);
    Transform *parentTransform = GetParentTransformSafe();

    // Compute local rotation from world rotation (inlined, not via SetWorldRotation,
    // to avoid intermediate euler extraction that could corrupt the exact values).
    if (!parentTransform) {
        data.localRotation = worldRot;
        data.localEulerAngles = euler; // root: local == world, store exact
    } else {
        data.localRotation = glm::inverse(parentTransform->GetWorldRotation()) * worldRot;
        data.localEulerAngles = ExtractEulerAnglesNear(data.localRotation, data.localEulerAngles);
    }

    data.dirty = true;
    InvalidateWorldMatrix(true);

    // Set exact world euler AFTER InvalidateWorldMatrix so the cascade doesn't clear it
    data.cachedWorldEulerAngles = euler;
    data.hasCachedWorldEulerAngles = true;
    data.worldEulerExact = true;
}

// ============================================================================
// World-space Scale (approximate lossyScale)
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
    auto &data = TransformECSStore::Instance().Get(m_ecsHandle);
    Transform *parentTransform = GetParentTransformSafe();
    if (!parentTransform) {
        data.localScale = worldScale;
    } else {
        glm::vec3 parentScale = parentTransform->GetWorldScale();
        data.localScale.x = (std::abs(parentScale.x) > 1e-6f) ? worldScale.x / parentScale.x : worldScale.x;
        data.localScale.y = (std::abs(parentScale.y) > 1e-6f) ? worldScale.y / parentScale.y : worldScale.y;
        data.localScale.z = (std::abs(parentScale.z) > 1e-6f) ? worldScale.z / parentScale.z : worldScale.z;
    }
    data.dirty = true;
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
    return glm::vec3(GetWorldMatrix() * glm::vec4(point, 1.0f));
}

glm::vec3 Transform::InverseTransformPoint(const glm::vec3 &point) const
{
    return glm::vec3(GetWorldToLocalMatrix() * glm::vec4(point, 1.0f));
}

glm::vec3 Transform::TransformDirection(const glm::vec3 &direction) const
{
    // Rotation only, no scale
    return glm::normalize(GetWorldRotation() * direction);
}

glm::vec3 Transform::InverseTransformDirection(const glm::vec3 &direction) const
{
    // Rotation only, no scale
    return glm::normalize(glm::inverse(GetWorldRotation()) * direction);
}

glm::vec3 Transform::TransformVector(const glm::vec3 &vector) const
{
    // Rotation + scale, no translation (w=0)
    glm::mat4 worldMatrix = GetWorldMatrix();
    return glm::vec3(worldMatrix * glm::vec4(vector, 0.0f));
}

glm::vec3 Transform::InverseTransformVector(const glm::vec3 &vector) const
{
    // Rotation + scale, no translation (w=0)
    return glm::vec3(GetWorldToLocalMatrix() * glm::vec4(vector, 0.0f));
}

// ============================================================================
// World-to-Local Matrix
// ============================================================================

glm::mat4 Transform::GetWorldToLocalMatrix() const
{
    return glm::inverse(GetWorldMatrix());
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

std::string Transform::Serialize() const
{
    json j;
    j["schema_version"] = 1;
    j["type"] = GetTypeName();
    j["enabled"] = IsEnabled();
    j["component_id"] = GetComponentID();

    // Position
    glm::vec3 pos = GetLocalPosition();
    j["position"] = {pos.x, pos.y, pos.z};

    // Rotation (as Euler angles)
    glm::vec3 euler = GetLocalEulerAngles();
    j["rotation"] = {euler.x, euler.y, euler.z};

    // Scale
    glm::vec3 scale = GetLocalScale();
    j["scale"] = {scale.x, scale.y, scale.z};

    return j.dump(2);
}

bool Transform::Deserialize(const std::string &jsonStr)
{
    try {
        json j = json::parse(jsonStr);

        // Call base class deserialize
        Component::Deserialize(jsonStr);

        // Position
        if (j.contains("position") && j["position"].is_array() && j["position"].size() == 3) {
            SetLocalPosition(j["position"][0].get<float>(), j["position"][1].get<float>(),
                             j["position"][2].get<float>());
        }

        // Rotation
        if (j.contains("rotation") && j["rotation"].is_array() && j["rotation"].size() == 3) {
            float x = j["rotation"][0].get<float>();
            float y = j["rotation"][1].get<float>();
            float z = j["rotation"][2].get<float>();
            SetLocalEulerAngles(x, y, z);
        }

        // Scale
        if (j.contains("scale") && j["scale"].is_array() && j["scale"].size() == 3) {
            SetLocalScale(j["scale"][0].get<float>(), j["scale"][1].get<float>(), j["scale"][2].get<float>());
        }

        auto &data = TransformECSStore::Instance().Get(m_ecsHandle);
        data.dirty = true;
        data.hasCachedWorldEulerAngles = false;
        data.worldEulerExact = false;
        InvalidateWorldMatrix(false);
        return true;
    } catch (const std::exception &e) {
        return false;
    }
}

void Transform::CloneDataTo(Transform &target) const
{
    auto &store = TransformECSStore::Instance();
    const auto &src = store.Get(m_ecsHandle);
    auto &dst = store.Get(target.m_ecsHandle);

    dst.localPosition = src.localPosition;
    dst.localEulerAngles = src.localEulerAngles;
    dst.localRotation = src.localRotation;
    dst.localScale = src.localScale;
    dst.dirty = true;
    dst.hasCachedWorldEulerAngles = false;
    dst.worldEulerExact = false;
    dst.worldMatrixDirty = true;

    target.SetEnabled(IsEnabled());
}

std::unique_ptr<Component> Transform::Clone() const
{
    // Transform is embedded in GameObject; this should not be called directly.
    // Return nullptr — GameObject::Clone handles transform cloning via CloneDataTo.
    return nullptr;
}

} // namespace infernux
