#include "JoltGpuRenderBindings.h"

#include "../GameObject.h"
#include "../MeshRenderer.h"
#include "../Scene.h"
#include "../SceneManager.h"
#include "../Transform.h"
#include "../TransformECSStore.h"
#include "core/math/QuaternionEuler.h"
#include "core/math/TransformMatrix.h"
#include <function/renderer/RenderPoseBuffer.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace infernux
{
namespace
{
constexpr uint32_t NoNode = UINT32_MAX;

JPH::GPU::Float4 Float4(const glm::vec4 &v)
{
    return {v.x, v.y, v.z, v.w};
}

glm::quat Multiply(const glm::quat &left, const glm::quat &right)
{
    const auto q = rotation::MultiplyTransform({left.x, left.y, left.z, left.w}, {right.x, right.y, right.z, right.w});
    return {q[3], q[0], q[1], q[2]};
}
} // namespace

struct JoltGpuRenderBindings::Snapshot
{
    struct TransformValue
    {
        Transform *identity = nullptr;
        PhysicsTargetReference reference;
        PhysicsActorHandle actor;
        uint32_t bodyId = UINT32_MAX;
        uint32_t parent = NoNode;
        uint64_t world = 0;
        std::array<uint64_t, 2> authored{};
        glm::vec3 position{};
        glm::quat rotation{1, 0, 0, 0};
        glm::vec3 scale{1};

        [[nodiscard]] glm::mat4 LocalMatrix() const
        {
            return glm::translate(glm::mat4(1), position) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1), scale);
        }

        [[nodiscard]] glm::mat4 LocalInverse() const
        {
            const auto value = transform_math::InverseTRS({position.x, position.y, position.z},
                                                          {rotation.x, rotation.y, rotation.z, rotation.w},
                                                          {scale.x, scale.y, scale.z});
            return glm::make_mat4(value.data());
        }
    };
    struct RendererValue
    {
        MeshRenderer *identity = nullptr;
        PhysicsTargetReference reference;
        uint32_t transform = NoNode;
        glm::vec3 pivot{};
        glm::vec3 boundsMin{};
        glm::vec3 boundsMax{};
    };
    std::vector<TransformValue> transforms;
    std::vector<RendererValue> renderers;
    std::unordered_map<uint32_t, uint64_t> continuity;
    uint64_t lifetime = 0;
    bool worldOverrides = false;
};

namespace
{
using Snapshot = JoltGpuRenderBindings::Snapshot;

bool SameSnapshot(const Snapshot &a, const Snapshot &b)
{
    if (a.transforms.size() != b.transforms.size() || a.renderers.size() != b.renderers.size() ||
        a.continuity != b.continuity)
        return false;
    for (size_t i = 0; i < a.transforms.size(); ++i) {
        const auto &x = a.transforms[i];
        const auto &y = b.transforms[i];
        if (x.identity != y.identity || x.reference.GetComponent() != y.reference.GetComponent() ||
            x.actor != y.actor || x.bodyId != y.bodyId || x.parent != y.parent || x.world != y.world ||
            x.scale != y.scale || x.authored != y.authored)
            return false;
        // An independent actor gets its pose from accepted device state, not
        // from the last lazily materialized CPU local position/rotation.
        if (!x.actor.IsValid() && (x.position != y.position || x.rotation != y.rotation))
            return false;
    }
    for (size_t i = 0; i < a.renderers.size(); ++i) {
        const auto &x = a.renderers[i];
        const auto &y = b.renderers[i];
        if (x.identity != y.identity || x.reference.GetComponent() != y.reference.GetComponent() ||
            x.transform != y.transform || x.pivot != y.pivot || x.boundsMin != y.boundsMin ||
            x.boundsMax != y.boundsMax)
            return false;
    }
    return true;
}

bool SameHierarchy(const Snapshot &a, uint32_t ai, const Snapshot &b, uint32_t bi)
{
    while (ai != NoNode && bi != NoNode) {
        const auto &x = a.transforms[ai];
        const auto &y = b.transforms[bi];
        if (x.identity != y.identity || x.reference.GetComponent() != y.reference.GetComponent() || x.world != y.world)
            return false;
        ai = x.parent;
        bi = y.parent;
    }
    return ai == bi;
}

bool CurrentHierarchy(const Snapshot &snapshot, uint32_t index)
{
    auto &store = TransformECSStore::Instance();
    while (index != NoNode) {
        const auto &value = snapshot.transforms[index];
        if (value.reference.GetComponent() != value.identity)
            return false;
        auto *object = value.identity->GetGameObject();
        if (!object || !object->GetScene() || object->GetScene()->GetWorldId() != value.world ||
            store.GetAuthoredPoseRevision(value.identity->GetECSHandle()) != value.authored ||
            store.GetLocalScale(value.identity->GetECSHandle()) != value.scale)
            return false;
        const auto *parent = value.identity->GetParent();
        if (parent != (value.parent == NoNode ? nullptr : snapshot.transforms[value.parent].identity))
            return false;
        index = value.parent;
    }
    return true;
}

uint64_t Continuity(const Snapshot &snapshot, uint32_t bodyId)
{
    const auto found = snapshot.continuity.find(bodyId);
    return found == snapshot.continuity.end() ? 0 : found->second;
}
} // namespace

struct JoltGpuRenderBindings::Impl
{
    struct Entry
    {
        PhysicsTargetReference reference;
        std::shared_ptr<RenderPoseBuffer> source;
        PhysicsActorHandle actor;
        uint32_t bodyId = UINT32_MAX;
        uint64_t bodyContinuity = 0;
        uint64_t continuity = 0;
        SnapshotPtr snapshot;
        uint32_t transform = NoNode;
    };
    std::unordered_map<MeshRenderer *, Entry> entries;
    std::unordered_map<uint32_t, uint64_t> continuity;
    SnapshotPtr published;
    uint64_t numericalGeneration = 0;
    uint64_t nextContinuity = 0;
    uint64_t lifetime = 1;

    static void ClearOwned(MeshRenderer *identity, const Entry &entry)
    {
        if (entry.reference.GetComponent() == identity && identity->GetPoseBuffer() == entry.source)
            identity->ClearPoseBuffer();
    }
};

JoltGpuRenderBindings::JoltGpuRenderBindings() : m_impl(std::make_unique<Impl>())
{
}

JoltGpuRenderBindings::~JoltGpuRenderBindings()
{
    Clear();
}

JoltGpuRenderBindings::SnapshotPtr JoltGpuRenderBindings::Capture() const
{
    auto snapshot = std::make_shared<Snapshot>();
    snapshot->continuity = m_impl->continuity;
    snapshot->lifetime = m_impl->lifetime;
    auto &transforms = TransformECSStore::Instance();
    auto &physics = PhysicsECSStore::Instance();
    std::unordered_map<const Transform *, uint32_t> indices;
    const auto captureTransform = [&](auto &&self, Transform *transform) -> uint32_t {
        if (const auto found = indices.find(transform); found != indices.end())
            return found->second;
        const auto parent = transform->GetParent();
        const auto parentIndex = parent ? self(self, parent) : NoNode;
        Snapshot::TransformValue value;
        value.identity = transform;
        value.reference = PhysicsTargetReference(transform);
        value.parent = parentIndex;
        auto *object = transform->GetGameObject();
        value.world = object->GetScene()->GetWorldId();
        value.actor = physics.FindActor(object);
        if (physics.IsValid(value.actor))
            value.bodyId = physics.GetActor(value.actor).bodyId;
        const auto handle = transform->GetECSHandle();
        value.position = transforms.GetLocalPosition(handle);
        value.rotation = transforms.GetLocalRotation(handle);
        value.scale = transforms.GetLocalScale(handle);
        if (transforms.IsFrameCacheActiveFor(handle)) {
            TransformECSStore::PhysicsPoseReadScope cachedAuthor(transforms);
            if (transforms.HasFrameCacheWorldPositionOverride(handle)) {
                value.position = transform->GetLocalPosition();
                snapshot->worldOverrides = true;
            }
            if (transforms.HasFrameCacheWorldRotationOverride(handle)) {
                value.rotation = transform->GetLocalRotation();
                snapshot->worldOverrides = true;
            }
        }
        value.authored = transforms.GetAuthoredPoseRevision(handle);
        const auto index = static_cast<uint32_t>(snapshot->transforms.size());
        snapshot->transforms.push_back(std::move(value));
        indices.emplace(transform, index);
        return index;
    };
    for (auto *renderer : SceneManager::Instance().GetActiveMeshRenderers()) {
        if (!renderer || !renderer->IsEnabled() || renderer->RequiresCpuPoseForGeometry())
            continue;
        auto *object = renderer->GetGameObject();
        if (!object || !object->IsActiveInHierarchy() || !object->GetScene())
            continue;
        const auto existing = m_impl->entries.find(renderer);
        const auto &source = renderer->GetPoseBuffer();
        if (source && (existing == m_impl->entries.end() || existing->second.reference.GetComponent() != renderer ||
                       source != existing->second.source))
            continue; // Explicit user/NN pose bindings have priority.
        bool hasActor = false;
        for (auto *ancestor = object; ancestor; ancestor = ancestor->GetParent())
            hasActor |= physics.IsValid(physics.FindActor(ancestor));
        if (!hasActor)
            continue;
        const auto transform = captureTransform(captureTransform, object->GetTransform());
        snapshot->renderers.push_back({renderer, PhysicsTargetReference(renderer), transform,
                                       renderer->GetSubmeshIndex() >= 0 ? renderer->GetMeshPivotOffset() : glm::vec3(0),
                                       renderer->GetLocalBoundsMin(), renderer->GetLocalBoundsMax()});
    }
    return snapshot;
}

bool JoltGpuRenderBindings::HasWorldOverrides(const Snapshot &snapshot) noexcept
{
    return snapshot.worldOverrides;
}

bool JoltGpuRenderBindings::NeedsExport(const Snapshot &snapshot, uint64_t numericalGeneration) const
{
    if (!m_impl->published || m_impl->numericalGeneration != numericalGeneration ||
        !SameSnapshot(snapshot, *m_impl->published))
        return true;
    for (const auto &renderer : snapshot.renderers) {
        const auto entry = m_impl->entries.find(renderer.identity);
        if (renderer.reference.GetComponent() == renderer.identity && entry != m_impl->entries.end() &&
            renderer.identity->GetPoseBuffer() != entry->second.source)
            return true;
    }
    return false;
}

JoltGpuRenderBindings::Prepared JoltGpuRenderBindings::Prepare(SnapshotPtr snapshot,
                                                               const JoltGpuScene::Publication &publication) const
{
    Prepared prepared;
    prepared.snapshot = std::move(snapshot);
    const auto &captured = *prepared.snapshot;
    std::vector<std::shared_ptr<const JoltGpuScene::PublishedActor>> actors(captured.transforms.size());
    for (size_t i = 0; i < captured.transforms.size(); ++i)
        for (const auto &actor : publication.actors)
            if (actor && actor->actor == captured.transforms[i].actor) {
                actors[i] = actor;
                break;
            }
    std::vector<uint32_t> nodeIndices(captured.transforms.size(), NoNode);
    for (size_t i = 0; i < actors.size(); ++i)
        if (actors[i]) {
            nodeIndices[i] = static_cast<uint32_t>(prepared.nodes.size());
            prepared.nodes.emplace_back();
        }
    for (size_t i = 0; i < actors.size(); ++i) {
        if (!actors[i])
            continue;
        auto parent = captured.transforms[i].parent;
        glm::mat4 parentChain(1);
        glm::mat4 parentInverse(1);
        glm::quat parentRotation(1, 0, 0, 0);
        while (parent != NoNode && !actors[parent]) {
            parentChain = captured.transforms[parent].LocalMatrix() * parentChain;
            parentInverse = parentInverse * captured.transforms[parent].LocalInverse();
            parentRotation = Multiply(captured.transforms[parent].rotation, parentRotation);
            parent = captured.transforms[parent].parent;
        }
        auto &node = prepared.nodes[nodeIndices[i]];
        node.bodyParent = {actors[i]->slot, parent == NoNode ? NoNode : nodeIndices[parent], 0, 0};
        node.rootCOM = actors[i]->rootCenterOfMass;
        node.localScale = Float4(glm::vec4(captured.transforms[i].scale, 0));
        node.parentRotation = {parentRotation.x, parentRotation.y, parentRotation.z, parentRotation.w};
        node.parentLocalToAncestor0 = Float4(parentChain[0]);
        node.parentLocalToAncestor1 = Float4(parentChain[1]);
        node.parentLocalToAncestor2 = Float4(parentChain[2]);
        node.parentLocalToAncestor3 = Float4(parentChain[3]);
        node.parentAncestorToLocal0 = Float4(parentInverse[0]);
        node.parentAncestorToLocal1 = Float4(parentInverse[1]);
        node.parentAncestorToLocal2 = Float4(parentInverse[2]);
        node.parentAncestorToLocal3 = Float4(parentInverse[3]);
    }
    for (size_t i = 0; i < captured.renderers.size(); ++i) {
        const auto &renderer = captured.renderers[i];
        auto root = renderer.transform;
        glm::mat4 localChain(1);
        while (root != NoNode && !actors[root]) {
            localChain = captured.transforms[root].LocalMatrix() * localChain;
            root = captured.transforms[root].parent;
        }
        if (root == NoNode)
            continue;
        JPH::GPU::PoseBinding binding{};
        binding.nodeIndex = {nodeIndices[root], 0, 0, 0};
        binding.objectLocal0 = Float4(localChain[0]);
        binding.objectLocal1 = Float4(localChain[1]);
        binding.objectLocal2 = Float4(localChain[2]);
        binding.objectLocal3 = Float4(localChain[3]);
        binding.drawPivot = Float4(glm::vec4(renderer.pivot, 0));
        binding.boundsCenter = Float4(glm::vec4((renderer.boundsMin + renderer.boundsMax) * 0.5f, 0));
        binding.boundsExtent = Float4(glm::vec4((renderer.boundsMax - renderer.boundsMin) * 0.5f, 0));
        prepared.bindings.push_back(binding);
        prepared.targets.push_back({i, actors[root], Continuity(captured, actors[root]->bodyId)});
    }
    return prepared;
}

void JoltGpuRenderBindings::Publish(const Prepared &prepared, const std::shared_ptr<rhi::ComputeBuffer> &models,
                                    const std::shared_ptr<rhi::ComputeBuffer> &bounds, uint64_t numericalGeneration)
{
    const auto &snapshot = *prepared.snapshot;
    if (snapshot.lifetime != m_impl->lifetime)
        return; // A completed old job cannot reacquire bindings after Stop.
    auto &physics = PhysicsECSStore::Instance();
    std::unordered_set<MeshRenderer *> retained;
    bool allPublished = true;
    for (size_t output = 0; output < prepared.targets.size(); ++output) {
        const auto &target = prepared.targets[output];
        const auto &captured = snapshot.renderers[target.renderer];
        auto *renderer = captured.identity;
        if (captured.reference.GetComponent() != renderer || !renderer->IsEnabled() ||
            !renderer->GetGameObject()->IsActiveInHierarchy() || renderer->RequiresCpuPoseForGeometry() ||
            !physics.IsValid(target.actor->actor) ||
            physics.GetActor(target.actor->actor).bodyId != target.actor->bodyId) {
            allPublished = false;
            continue;
        }
        auto found = m_impl->entries.find(renderer);
        if (found != m_impl->entries.end() && found->second.reference.GetComponent() != renderer) {
            m_impl->entries.erase(found);
            found = m_impl->entries.end();
        }
        const auto &current = renderer->GetPoseBuffer();
        if (current && (found == m_impl->entries.end() || current != found->second.source)) {
            allPublished = false;
            continue;
        }
        retained.insert(renderer);
        const auto pivot = renderer->GetSubmeshIndex() >= 0 ? renderer->GetMeshPivotOffset() : glm::vec3(0);
        const auto currentContinuity = m_impl->continuity.find(target.actor->bodyId);
        if (pivot != captured.pivot || renderer->GetLocalBoundsMin() != captured.boundsMin ||
            renderer->GetLocalBoundsMax() != captured.boundsMax || !CurrentHierarchy(snapshot, captured.transform) ||
            Continuity(snapshot, target.actor->bodyId) !=
                (currentContinuity != m_impl->continuity.end() ? currentContinuity->second : 0)) {
            allPublished = false;
            continue;
        }
        if (found == m_impl->entries.end())
            found = m_impl->entries.emplace(renderer, Impl::Entry{}).first;
        auto &entry = found->second;
        const bool rebind = !entry.source || entry.actor != target.actor->actor ||
                            entry.bodyId != target.actor->bodyId || entry.bodyContinuity != target.continuity ||
                            current != entry.source || !entry.snapshot ||
                            !SameHierarchy(*entry.snapshot, entry.transform, snapshot, captured.transform);
        if (!entry.source)
            entry.source = std::make_shared<RenderPoseBuffer>(models, bounds, 1, static_cast<uint32_t>(output),
                                                              static_cast<uint32_t>(output));
        else
            entry.source->Publish(models, bounds, static_cast<uint32_t>(output), static_cast<uint32_t>(output));
        entry.reference = captured.reference;
        entry.actor = target.actor->actor;
        entry.bodyId = target.actor->bodyId;
        entry.bodyContinuity = target.continuity;
        entry.snapshot = prepared.snapshot;
        entry.transform = captured.transform;
        if (rebind) {
            entry.continuity = ++m_impl->nextContinuity;
            renderer->SetPoseBuffer(entry.source, 0, entry.continuity);
        }
    }
    for (auto it = m_impl->entries.begin(); it != m_impl->entries.end();) {
        if (retained.find(it->first) == retained.end()) {
            Impl::ClearOwned(it->first, it->second);
            it = m_impl->entries.erase(it);
        } else
            ++it;
    }
    if (allPublished) {
        m_impl->published = prepared.snapshot;
        m_impl->numericalGeneration = numericalGeneration;
    } else {
        // Completion belongs to the old captured job. A changed component or
        // author hierarchy must be recaptured, not remembered as exported.
        m_impl->published.reset();
    }
}

void JoltGpuRenderBindings::MarkDiscontinuity(uint32_t bodyId)
{
    if (bodyId != UINT32_MAX)
        ++m_impl->continuity[bodyId];
}

void JoltGpuRenderBindings::RetireBody(uint32_t bodyId) noexcept
{
    for (auto it = m_impl->entries.begin(); it != m_impl->entries.end();) {
        if (it->second.bodyId == bodyId) {
            Impl::ClearOwned(it->first, it->second);
            it = m_impl->entries.erase(it);
        } else
            ++it;
    }
    m_impl->continuity.erase(bodyId);
    m_impl->published.reset();
}

void JoltGpuRenderBindings::Clear() noexcept
{
    ++m_impl->lifetime;
    for (const auto &[renderer, entry] : m_impl->entries)
        Impl::ClearOwned(renderer, entry);
    m_impl->entries.clear();
    m_impl->continuity.clear();
    m_impl->published.reset();
    m_impl->numericalGeneration = 0;
}

} // namespace infernux
