#include "TransformECSStore.h"
#include "GameObject.h"
#include "Scene.h"
#include "Transform.h"
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace infernux
{

TransformECSStore &TransformECSStore::Instance()
{
    static TransformECSStore instance;
    return instance;
}

TransformECSStore::Handle TransformECSStore::Allocate(Transform *owner)
{
    uint32_t index;

    if (m_freeListHead != UINT32_MAX) {
        index = m_freeListHead;
        m_freeListHead = m_nextFree[index];
        m_alive[index] = 1;
    } else {
        index = static_cast<uint32_t>(m_generations.size());
        m_localPositions.emplace_back(0.0f, 0.0f, 0.0f);
        m_localEulerAngles.emplace_back(0.0f, 0.0f, 0.0f);
        m_localRotations.emplace_back(1.0f, 0.0f, 0.0f, 0.0f);
        m_cachedWorldEulerAngles.emplace_back(0.0f, 0.0f, 0.0f);
        m_hasCachedWorldEulerAngles.push_back(0);
        m_worldEulerExact.push_back(0);
        m_localScales.emplace_back(1.0f, 1.0f, 1.0f);
        m_dirty.push_back(1);
        m_cachedWorldMatrices.emplace_back(1.0f);
        m_worldMatrixDirty.push_back(1);
        m_anyWorldMatrixDirty = true;
        m_owners.push_back(nullptr);
        m_generations.push_back(1);
        m_alive.push_back(1);
        m_nextFree.push_back(UINT32_MAX);

        // Keep frame cache arrays in sync with capacity.
        m_fcWorldPositions.emplace_back(0.0f, 0.0f, 0.0f);
        m_fcWorldRotations.emplace_back(1.0f, 0.0f, 0.0f, 0.0f);
        m_fcDirty.push_back(0);
    }

    // Reset fields to defaults for recycled slots.
    m_localPositions[index] = glm::vec3(0.0f);
    m_localEulerAngles[index] = glm::vec3(0.0f);
    m_localRotations[index] = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    m_cachedWorldEulerAngles[index] = glm::vec3(0.0f);
    m_hasCachedWorldEulerAngles[index] = 0;
    m_worldEulerExact[index] = 0;
    m_localScales[index] = glm::vec3(1.0f);
    m_dirty[index] = 1;
    m_cachedWorldMatrices[index] = glm::mat4(1.0f);
    m_worldMatrixDirty[index] = 1;
    m_anyWorldMatrixDirty = true;
    m_owners[index] = owner;

    ++m_aliveCount;
    return Handle{index, m_generations[index]};
}

void TransformECSStore::Release(Handle handle)
{
    if (!IsValid(handle)) {
        return;
    }
    uint32_t idx = handle.index;
    m_owners[idx] = nullptr;
    m_alive[idx] = 0;
    ++m_generations[idx];
    m_nextFree[idx] = m_freeListHead;
    m_freeListHead = idx;
    --m_aliveCount;
}

bool TransformECSStore::IsValid(Handle handle) const
{
    if (!handle.IsValid() || handle.index >= m_generations.size()) {
        return false;
    }
    return m_alive[handle.index] && m_generations[handle.index] == handle.generation;
}

void TransformECSStore::RebindOwner(Handle handle, Transform *owner)
{
    if (!IsValid(handle)) {
        return;
    }
    m_owners[handle.index] = owner;
}

void TransformECSStore::Reserve(size_t capacity)
{
    m_localPositions.reserve(capacity);
    m_localEulerAngles.reserve(capacity);
    m_localRotations.reserve(capacity);
    m_cachedWorldEulerAngles.reserve(capacity);
    m_hasCachedWorldEulerAngles.reserve(capacity);
    m_worldEulerExact.reserve(capacity);
    m_localScales.reserve(capacity);
    m_dirty.reserve(capacity);
    m_cachedWorldMatrices.reserve(capacity);
    m_worldMatrixDirty.reserve(capacity);
    m_owners.reserve(capacity);
    m_generations.reserve(capacity);
    m_alive.reserve(capacity);
    m_nextFree.reserve(capacity);
    m_fcWorldPositions.reserve(capacity);
    m_fcWorldRotations.reserve(capacity);
    m_fcDirty.reserve(capacity);
}

TransformECSData TransformECSStore::GetSnapshot(Handle h) const
{
    uint32_t i = h.index;
    TransformECSData d;
    d.localPosition = m_localPositions[i];
    d.localEulerAngles = m_localEulerAngles[i];
    d.localRotation = m_localRotations[i];
    d.cachedWorldEulerAngles = m_cachedWorldEulerAngles[i];
    d.hasCachedWorldEulerAngles = m_hasCachedWorldEulerAngles[i] != 0;
    d.worldEulerExact = m_worldEulerExact[i] != 0;
    d.localScale = m_localScales[i];
    d.dirty = m_dirty[i] != 0;
    d.cachedWorldMatrix = m_cachedWorldMatrices[i];
    d.worldMatrixDirty = m_worldMatrixDirty[i] != 0;
    d.owner = m_owners[i];
    return d;
}

void TransformECSStore::SetSnapshot(Handle h, const TransformECSData &d)
{
    uint32_t i = h.index;
    m_localPositions[i] = d.localPosition;
    m_localEulerAngles[i] = d.localEulerAngles;
    m_localRotations[i] = d.localRotation;
    m_cachedWorldEulerAngles[i] = d.cachedWorldEulerAngles;
    m_hasCachedWorldEulerAngles[i] = d.hasCachedWorldEulerAngles ? 1 : 0;
    m_worldEulerExact[i] = d.worldEulerExact ? 1 : 0;
    m_localScales[i] = d.localScale;
    m_dirty[i] = d.dirty ? 1 : 0;
    m_cachedWorldMatrices[i] = d.cachedWorldMatrix;
    m_worldMatrixDirty[i] = d.worldMatrixDirty ? 1 : 0;
    if (d.worldMatrixDirty) m_anyWorldMatrixDirty = true;
    m_owners[i] = d.owner;
}

void TransformECSStore::InvalidateSubtree(Transform *root, bool clearWorldEulerExact) const
{
    if (!root) {
        return;
    }

    auto handle = root->GetECSHandle();
    if (!IsValid(handle)) {
        return;
    }

    uint32_t idx = handle.index;
    // const_cast: cache invalidation is logically const — it only marks
    // cached data as stale.
    auto &self = const_cast<TransformECSStore &>(*this);
    if (!self.m_worldMatrixDirty[idx]) {
        self.m_worldMatrixDirty[idx] = 1;
        self.m_anyWorldMatrixDirty = true;
    }
    if (clearWorldEulerExact) {
        self.m_worldEulerExact[idx] = 0;
        self.m_anyRotationDirtied = true;
    }

    GameObject *go = root->GetGameObject();
    if (!go) {
        return;
    }

    for (size_t i = 0; i < go->GetChildCount(); ++i) {
        GameObject *child = go->GetChild(i);
        if (child) {
            InvalidateSubtree(child->GetTransform(), clearWorldEulerExact);
        }
    }
}

void TransformECSStore::SyncSceneWorldMatrices(Scene *scene)
{
    if (!scene) {
        return;
    }

    // Skip redundant syncs during frame cache phase — all world-space
    // reads/writes go through the cache arrays, so recomputing
    // m_cachedWorldMatrices from live SoA is wasted work.
    if (m_frameCacheActive) {
        return;
    }

    // Fast skip when no transform was dirtied since the last sync.
    if (!m_anyWorldMatrixDirty) {
        return;
    }

    const auto &roots = scene->GetRootObjects();
    for (const auto &root : roots) {
        SyncObjectWorldMatrices(root.get());
    }

    m_anyWorldMatrixDirty = false;
}

void TransformECSStore::SyncObjectWorldMatrices(GameObject *obj)
{
    if (!obj) {
        return;
    }

    Transform *t = obj->GetTransform();
    if (t) {
        auto handle = t->GetECSHandle();
        if (IsValid(handle)) {
            uint32_t idx = handle.index;
            if (m_worldMatrixDirty[idx]) {
                glm::mat4 local = glm::translate(glm::mat4(1.0f), m_localPositions[idx]) *
                                  glm::mat4_cast(m_localRotations[idx]) *
                                  glm::scale(glm::mat4(1.0f), m_localScales[idx]);

                GameObject *parent = obj->GetParent();
                if (!parent) {
                    m_cachedWorldMatrices[idx] = local;
                } else {
                    Transform *pt = parent->GetTransform();
                    auto parentHandle = pt->GetECSHandle();
                    if (IsValid(parentHandle)) {
                        m_cachedWorldMatrices[idx] = m_cachedWorldMatrices[parentHandle.index] * local;
                    } else {
                        m_cachedWorldMatrices[idx] = local;
                    }
                }
                m_worldMatrixDirty[idx] = 0;
            }
        }
    }

    for (size_t i = 0; i < obj->GetChildCount(); ++i) {
        SyncObjectWorldMatrices(obj->GetChild(i));
    }
}

// ── batch gather/scatter ─────────────────────────────────────────────

void TransformECSStore::GatherLocalPositions(Transform *const *transforms, float *out, size_t count) const
{
    for (size_t i = 0; i < count; ++i) {
        uint32_t idx = transforms[i]->GetECSHandle().index;
        const auto &v = m_localPositions[idx];
        out[i * 3 + 0] = v.x;
        out[i * 3 + 1] = v.y;
        out[i * 3 + 2] = v.z;
    }
}

void TransformECSStore::ScatterLocalPositions(Transform *const *transforms, const float *in, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        auto h = transforms[i]->GetECSHandle();
        uint32_t idx = h.index;
        m_localPositions[idx] = glm::vec3(in[i * 3], in[i * 3 + 1], in[i * 3 + 2]);
        m_dirty[idx] = 1;
        m_worldMatrixDirty[idx] = 1;
    }
    m_anyWorldMatrixDirty = true;
    // Invalidate subtrees only for objects that actually have children.
    for (size_t i = 0; i < count; ++i) {
        GameObject *go = transforms[i]->GetGameObject();
        if (go && go->GetChildCount() > 0) {
            InvalidateSubtree(transforms[i], false);
        }
    }
}

void TransformECSStore::GatherLocalScales(Transform *const *transforms, float *out, size_t count) const
{
    for (size_t i = 0; i < count; ++i) {
        uint32_t idx = transforms[i]->GetECSHandle().index;
        const auto &v = m_localScales[idx];
        out[i * 3 + 0] = v.x;
        out[i * 3 + 1] = v.y;
        out[i * 3 + 2] = v.z;
    }
}

void TransformECSStore::ScatterLocalScales(Transform *const *transforms, const float *in, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        auto h = transforms[i]->GetECSHandle();
        uint32_t idx = h.index;
        m_localScales[idx] = glm::vec3(in[i * 3], in[i * 3 + 1], in[i * 3 + 2]);
        m_dirty[idx] = 1;
        m_worldMatrixDirty[idx] = 1;
    }
    m_anyWorldMatrixDirty = true;
    for (size_t i = 0; i < count; ++i) {
        GameObject *go = transforms[i]->GetGameObject();
        if (go && go->GetChildCount() > 0) {
            InvalidateSubtree(transforms[i], false);
        }
    }
}

void TransformECSStore::GatherLocalRotations(Transform *const *transforms, float *out, size_t count) const
{
    for (size_t i = 0; i < count; ++i) {
        uint32_t idx = transforms[i]->GetECSHandle().index;
        const auto &q = m_localRotations[idx];
        out[i * 4 + 0] = q.x;
        out[i * 4 + 1] = q.y;
        out[i * 4 + 2] = q.z;
        out[i * 4 + 3] = q.w;
    }
}

void TransformECSStore::ScatterLocalRotations(Transform *const *transforms, const float *in, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        auto h = transforms[i]->GetECSHandle();
        uint32_t idx = h.index;
        glm::quat q(in[i * 4 + 3], in[i * 4], in[i * 4 + 1], in[i * 4 + 2]); // glm: (w,x,y,z)
        m_localRotations[idx] = q;
        m_localEulerAngles[idx] = glm::degrees(glm::eulerAngles(q));
        m_hasCachedWorldEulerAngles[idx] = 0;
        m_dirty[idx] = 1;
        m_worldMatrixDirty[idx] = 1;
    }
    m_anyWorldMatrixDirty = true;
    m_anyRotationDirtied = true;
    for (size_t i = 0; i < count; ++i) {
        GameObject *go = transforms[i]->GetGameObject();
        if (go && go->GetChildCount() > 0) {
            InvalidateSubtree(transforms[i], true);
        }
    }
}

void TransformECSStore::GatherLocalEulerAngles(Transform *const *transforms, float *out, size_t count) const
{
    for (size_t i = 0; i < count; ++i) {
        uint32_t idx = transforms[i]->GetECSHandle().index;
        const auto &v = m_localEulerAngles[idx];
        out[i * 3 + 0] = v.x;
        out[i * 3 + 1] = v.y;
        out[i * 3 + 2] = v.z;
    }
}

void TransformECSStore::ScatterLocalEulerAngles(Transform *const *transforms, const float *in, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        auto h = transforms[i]->GetECSHandle();
        uint32_t idx = h.index;
        glm::vec3 euler(in[i * 3], in[i * 3 + 1], in[i * 3 + 2]);
        m_localEulerAngles[idx] = euler;
        // Recompute quaternion from euler (YXZ convention).
        glm::vec3 rad = glm::radians(euler);
        m_localRotations[idx] = glm::quat(rad);
        m_hasCachedWorldEulerAngles[idx] = 0;
        m_dirty[idx] = 1;
        m_worldMatrixDirty[idx] = 1;
    }
    m_anyWorldMatrixDirty = true;
    m_anyRotationDirtied = true;
    for (size_t i = 0; i < count; ++i) {
        GameObject *go = transforms[i]->GetGameObject();
        if (go && go->GetChildCount() > 0) {
            InvalidateSubtree(transforms[i], true);
        }
    }
}

void TransformECSStore::GatherWorldPositions(Transform *const *transforms, float *out, size_t count) const
{
    for (size_t i = 0; i < count; ++i) {
        glm::vec3 wp = transforms[i]->GetWorldPosition();
        out[i * 3 + 0] = wp.x;
        out[i * 3 + 1] = wp.y;
        out[i * 3 + 2] = wp.z;
    }
}

void TransformECSStore::ScatterWorldPositions(Transform *const *transforms, const float *in, size_t count)
{
    // Fast path: batch-set local positions and dirty flags, then
    // batch-invalidate subtrees.  SetWorldPosition per-element is
    // expensive due to GetParentTransformSafe + inverse matrix per call.
    for (size_t i = 0; i < count; ++i) {
        Transform *t = transforms[i];
        Transform *parent = t->GetParent();
        glm::vec3 wp(in[i * 3], in[i * 3 + 1], in[i * 3 + 2]);
        uint32_t idx = t->GetECSHandle().index;
        if (!parent) {
            m_localPositions[idx] = wp;
        } else {
            glm::mat4 invParent = glm::inverse(parent->GetWorldMatrix());
            m_localPositions[idx] = glm::vec3(invParent * glm::vec4(wp, 1.0f));
        }
        m_dirty[idx] = 1;
        m_worldMatrixDirty[idx] = 1;
    }
    m_anyWorldMatrixDirty = true;
    // Batch invalidate children (skip leaf nodes quickly).
    for (size_t i = 0; i < count; ++i) {
        GameObject *go = transforms[i]->GetGameObject();
        if (go && go->GetChildCount() > 0) {
            InvalidateSubtree(transforms[i], false);
        }
    }
}

void TransformECSStore::GatherWorldEulerAngles(Transform *const *transforms, float *out, size_t count) const
{
    for (size_t i = 0; i < count; ++i) {
        glm::vec3 we = transforms[i]->GetWorldEulerAngles();
        out[i * 3 + 0] = we.x;
        out[i * 3 + 1] = we.y;
        out[i * 3 + 2] = we.z;
    }
}

void TransformECSStore::ScatterWorldEulerAngles(Transform *const *transforms, const float *in, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        transforms[i]->SetWorldEulerAngles(glm::vec3(in[i * 3], in[i * 3 + 1], in[i * 3 + 2]));
    }
}

void TransformECSStore::GatherWorldRotations(Transform *const *transforms, float *out, size_t count) const
{
    for (size_t i = 0; i < count; ++i) {
        glm::quat wr = transforms[i]->GetWorldRotation();
        out[i * 4 + 0] = wr.x;
        out[i * 4 + 1] = wr.y;
        out[i * 4 + 2] = wr.z;
        out[i * 4 + 3] = wr.w;
    }
}

void TransformECSStore::ScatterWorldRotations(Transform *const *transforms, const float *in, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        transforms[i]->SetWorldRotation(glm::quat(in[i * 4 + 3], in[i * 4], in[i * 4 + 1], in[i * 4 + 2]));
    }
}

// ── Frame Cache ──────────────────────────────────────────────────────

void TransformECSStore::BeginFrameCache(Scene *scene)
{
    if (!scene) {
        return;
    }

    // Ensure world matrices are up-to-date before snapshotting.
    SyncSceneWorldMatrices(scene);

    const size_t cap = m_generations.size();

    // Resize cache arrays to match current capacity.
    if (m_fcWorldPositions.size() < cap) {
        m_fcWorldPositions.resize(cap);
        m_fcWorldRotations.resize(cap);
        m_fcDirty.resize(cap, 0);
    }

    // Extract world position (column 3) and rotation from cached world matrices.
    if (m_anyRotationDirtied) {
        // Full extraction: position + rotation (expensive quat_cast).
        for (size_t i = 0; i < cap; ++i) {
            if (!m_alive[i]) {
                continue;
            }
            const glm::mat4 &wm = m_cachedWorldMatrices[i];
            m_fcWorldPositions[i] = glm::vec3(wm[3]);
            m_fcWorldRotations[i] = glm::quat_cast(glm::mat3(wm));
        }
        m_anyRotationDirtied = false;
    } else {
        // Fast path: only extract positions (cheap vec3 copy).
        // Rotations haven't changed since last snapshot.
        for (size_t i = 0; i < cap; ++i) {
            if (!m_alive[i]) {
                continue;
            }
            m_fcWorldPositions[i] = glm::vec3(m_cachedWorldMatrices[i][3]);
        }
    }
    std::memset(m_fcDirty.data(), 0, cap);

    m_frameCacheActive = true;
    m_fcScene = scene;
}

void TransformECSStore::EndFrameCache()
{
    if (!m_frameCacheActive) {
        return;
    }

    m_frameCacheActive = false;

    const size_t cap = m_fcDirty.size();
    for (size_t i = 0; i < cap; ++i) {
        const uint8_t d = m_fcDirty[i];
        if (d == 0 || !m_alive[i]) {
            continue;
        }

        Transform *owner = m_owners[i];
        if (!owner) {
            continue;
        }

        // World position dirty → compute local position from inverse parent.
        if (d & 0x01) {
            Transform *parent = owner->GetParent();
            if (!parent) {
                m_localPositions[i] = m_fcWorldPositions[i];
            } else {
                glm::mat4 invParent = glm::inverse(parent->GetWorldMatrix());
                m_localPositions[i] = glm::vec3(invParent * glm::vec4(m_fcWorldPositions[i], 1.0f));
            }
            m_dirty[i] = 1;
            m_worldMatrixDirty[i] = 1;
        }

        // World rotation dirty → compute local rotation from inverse parent.
        if (d & 0x02) {
            Transform *parent = owner->GetParent();
            if (!parent) {
                m_localRotations[i] = m_fcWorldRotations[i];
            } else {
                m_localRotations[i] = glm::inverse(parent->GetWorldRotation()) * m_fcWorldRotations[i];
            }
            m_localEulerAngles[i] = glm::degrees(glm::eulerAngles(m_localRotations[i]));
            m_hasCachedWorldEulerAngles[i] = 0;
            m_dirty[i] = 1;
            m_worldMatrixDirty[i] = 1;
        }

        // Local property dirty bits (2-5) already wrote to live SoA in SetCachedLocal*.
        // Just ensure subtree invalidation.
        if (d & 0x3C) { // bits 2-5
            m_dirty[i] = 1;
            m_worldMatrixDirty[i] = 1;
        }

        // Invalidate subtrees for any dirty slot.
        GameObject *go = owner->GetGameObject();
        if (go && go->GetChildCount() > 0) {
            InvalidateSubtree(owner, (d & 0x02) != 0);
        }
    }

    // Pre-sync world matrices now so CollectRenderables hits clean
    // caches (avoids 14,400 lazy recomputes with poor cache locality).
    if (m_fcScene) {
        m_anyWorldMatrixDirty = true;   // dirty from the loop above
        SyncSceneWorldMatrices(m_fcScene);
        m_fcScene = nullptr;
    }
}

void TransformECSStore::SetCachedWorldPosition(uint32_t slotIndex, const glm::vec3 &v)
{
    m_fcWorldPositions[slotIndex] = v;
    m_fcDirty[slotIndex] |= 0x01;
}

void TransformECSStore::SetCachedWorldRotation(uint32_t slotIndex, const glm::quat &q)
{
    m_fcWorldRotations[slotIndex] = q;
    m_fcDirty[slotIndex] |= 0x02;
    m_anyRotationDirtied = true;
}

void TransformECSStore::SetCachedLocalPosition(uint32_t slotIndex, const glm::vec3 &v)
{
    m_localPositions[slotIndex] = v;
    m_fcDirty[slotIndex] |= 0x04;
    m_worldMatrixDirty[slotIndex] = 1;
    m_anyWorldMatrixDirty = true;
}

void TransformECSStore::SetCachedLocalScale(uint32_t slotIndex, const glm::vec3 &v)
{
    m_localScales[slotIndex] = v;
    m_fcDirty[slotIndex] |= 0x08;
    m_worldMatrixDirty[slotIndex] = 1;
    m_anyWorldMatrixDirty = true;
}

void TransformECSStore::SetCachedLocalRotation(uint32_t slotIndex, const glm::quat &q)
{
    m_localRotations[slotIndex] = q;
    m_localEulerAngles[slotIndex] = glm::degrees(glm::eulerAngles(q));
    m_hasCachedWorldEulerAngles[slotIndex] = 0;
    m_fcDirty[slotIndex] |= 0x10;
    m_worldMatrixDirty[slotIndex] = 1;
    m_anyWorldMatrixDirty = true;
    m_anyRotationDirtied = true;
}

void TransformECSStore::SetCachedLocalEulerAngles(uint32_t slotIndex, const glm::vec3 &v)
{
    m_localEulerAngles[slotIndex] = v;
    glm::vec3 rad = glm::radians(v);
    m_localRotations[slotIndex] = glm::quat(rad);
    m_hasCachedWorldEulerAngles[slotIndex] = 0;
    m_fcDirty[slotIndex] |= 0x20;
    m_worldMatrixDirty[slotIndex] = 1;
    m_anyWorldMatrixDirty = true;
    m_anyRotationDirtied = true;
}

} // namespace infernux
