#include "TransformECSStore.h"
#include "GameObject.h"
#include "Scene.h"
#include "Transform.h"
#include <glm/gtc/matrix_transform.hpp>

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
        m_owners.push_back(nullptr);
        m_generations.push_back(1);
        m_alive.push_back(1);
        m_nextFree.push_back(UINT32_MAX);
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
    }
    if (clearWorldEulerExact) {
        self.m_worldEulerExact[idx] = 0;
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

    const auto &roots = scene->GetRootObjects();
    for (const auto &root : roots) {
        SyncObjectWorldMatrices(root.get());
    }
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
    // Invalidate subtrees for hierarchy consistency.
    for (size_t i = 0; i < count; ++i) {
        InvalidateSubtree(transforms[i], false);
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
    for (size_t i = 0; i < count; ++i) {
        InvalidateSubtree(transforms[i], false);
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
    for (size_t i = 0; i < count; ++i) {
        InvalidateSubtree(transforms[i], true);
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
    for (size_t i = 0; i < count; ++i) {
        InvalidateSubtree(transforms[i], true);
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
    for (size_t i = 0; i < count; ++i) {
        transforms[i]->SetWorldPosition(glm::vec3(in[i * 3], in[i * 3 + 1], in[i * 3 + 2]));
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

} // namespace infernux
