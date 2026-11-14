#include "JoltGpuRenderBindings.h"

#include "../GameObject.h"
#include "../MeshRenderer.h"
#include "../RuntimeWorld.h"
#include "../Scene.h"
#include "../SceneManager.h"
#include "../Transform.h"
#include "../TransformECSStore.h"
#include "core/math/QuaternionEuler.h"
#include "core/math/TransformMatrix.h"
#include <function/renderer/RenderPoseBuffer.h>

#include <functional>
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

struct ActorHandleHash
{
    size_t operator()(PhysicsActorHandle handle) const noexcept
    {
        return std::hash<uint64_t>{}((uint64_t(handle.generation) << 32) | handle.index);
    }
};

// Active state follows the live GameObject graph, which can differ from a
// pending snapshot after reparenting. Share ancestors without recursive calls.
struct ActiveHierarchy
{
    std::unordered_map<const GameObject *, bool> values;
    std::vector<const GameObject *> path;

    bool IsActive(const GameObject *object)
    {
        if (!object)
            return false;
        path.clear();
        const auto *ancestor = object;
        while (ancestor && values.find(ancestor) == values.end()) {
            path.push_back(ancestor);
            ancestor = ancestor->GetParent();
        }
        bool active = !ancestor || values.at(ancestor);
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            active = active && (*it)->IsActive();
            values.emplace(*it, active);
        }
        return values.at(object);
    }
};
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

struct HierarchyComparison
{
    JoltGpuRenderBindings::SnapshotPtr previous;
    std::unordered_map<uint64_t, bool> values;
    std::vector<uint64_t> path;

    bool Matches(const Snapshot &current, uint32_t oldIndex, uint32_t currentIndex)
    {
        path.clear();
        bool same;
        for (;;) {
            if (oldIndex == NoNode || currentIndex == NoNode) {
                same = oldIndex == currentIndex;
                break;
            }
            const uint64_t pair = (uint64_t(oldIndex) << 32) | currentIndex;
            if (const auto found = values.find(pair); found != values.end()) {
                same = found->second;
                break;
            }
            path.push_back(pair);
            const auto &old = previous->transforms[oldIndex];
            const auto &next = current.transforms[currentIndex];
            // Continuity compares native identities, lifetime references and
            // scene/parent membership only, independently of authored poses.
            if (old.identity != next.identity || old.reference.GetComponent() != next.reference.GetComponent() ||
                old.world != next.world) {
                same = false;
                break;
            }
            oldIndex = old.parent;
            currentIndex = next.parent;
        }
        for (const auto pair : path)
            values.emplace(pair, same);
        return same;
    }
};

std::vector<uint8_t> CurrentHierarchy(const Snapshot &snapshot)
{
    std::vector<uint8_t> current(snapshot.transforms.size(), 0);
    auto &store = TransformECSStore::Instance();
    // Capture stores parents first. A changed ancestor invalidates every
    // descendant in this one pass instead of being checked for each mesh.
    for (size_t index = 0; index < snapshot.transforms.size(); ++index) {
        const auto &value = snapshot.transforms[index];
        if ((value.parent != NoNode && !current[value.parent]) || value.reference.GetComponent() != value.identity)
            continue;
        auto *object = value.identity->GetGameObject();
        if (!object || !object->GetScene() || object->GetScene()->GetWorldId() != value.world ||
            store.GetAuthoredPoseRevision(value.identity->GetECSHandle()) != value.authored ||
            store.GetLocalScale(value.identity->GetECSHandle()) != value.scale)
            continue;
        const auto *parent = value.identity->GetParent();
        if (parent != (value.parent == NoNode ? nullptr : snapshot.transforms[value.parent].identity))
            continue;
        current[index] = 1;
    }
    return current;
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

JoltGpuRenderBindings::SnapshotPtr JoltGpuRenderBindings::Capture(RuntimeWorld &world) const
{
    const WorldExecutionScope ownerScope(&world);
    auto snapshot = std::make_shared<Snapshot>();
    snapshot->continuity = m_impl->continuity;
    snapshot->lifetime = m_impl->lifetime;
    auto &transforms = TransformECSStore::Instance();
    auto &physics = PhysicsECSStore::Instance();
    struct Ancestry
    {
        Transform *parent = nullptr;
        PhysicsActorHandle actor;
        uint32_t bodyId = UINT32_MAX;
        bool hasActor = false;
    };
    std::unordered_map<const Transform *, Ancestry> ancestry;
    std::unordered_map<const Transform *, uint32_t> indices;
    ActiveHierarchy active;
    std::vector<Transform *> ancestryPath, capturePath;
    const auto &renderers = SceneManager::Instance().GetActiveMeshRenderers();
    ancestry.reserve(renderers.size());
    indices.reserve(renderers.size());
    snapshot->renderers.reserve(renderers.size());
    const auto hasActor = [&](Transform *transform) {
        ancestryPath.clear();
        auto *ancestor = transform;
        while (ancestor && ancestry.find(ancestor) == ancestry.end()) {
            ancestryPath.push_back(ancestor);
            ancestor = ancestor->GetParent();
        }
        for (auto it = ancestryPath.rbegin(); it != ancestryPath.rend(); ++it) {
            auto *current = *it;
            Ancestry value;
            value.parent = current->GetParent();
            value.actor = physics.FindActor(current->GetGameObject());
            const bool ownActor = physics.IsValid(value.actor);
            if (ownActor)
                value.bodyId = physics.GetActor(value.actor).bodyId;
            value.hasActor = ownActor || (value.parent && ancestry.at(value.parent).hasActor);
            ancestry.emplace(current, value);
        }
        return ancestry.at(transform).hasActor;
    };
    const auto captureTransform = [&](Transform *transform) -> uint32_t {
        if (const auto found = indices.find(transform); found != indices.end())
            return found->second;
        capturePath.clear();
        auto *ancestor = transform;
        while (ancestor && indices.find(ancestor) == indices.end()) {
            capturePath.push_back(ancestor);
            ancestor = ancestry.at(ancestor).parent;
        }
        uint32_t parentIndex = ancestor ? indices.at(ancestor) : NoNode;
        for (auto it = capturePath.rbegin(); it != capturePath.rend(); ++it) {
            auto *current = *it;
            const auto &capturedAncestry = ancestry.at(current);
            Snapshot::TransformValue value;
            value.identity = current;
            value.reference = PhysicsTargetReference(current);
            value.parent = parentIndex;
            value.world = current->GetGameObject()->GetScene()->GetWorldId();
            value.actor = capturedAncestry.actor;
            value.bodyId = capturedAncestry.bodyId;
            const auto handle = current->GetECSHandle();
            value.position = transforms.GetLocalPosition(handle);
            value.rotation = transforms.GetLocalRotation(handle);
            value.scale = transforms.GetLocalScale(handle);
            if (transforms.IsFrameCacheActiveFor(handle)) {
                // Preserve Transform's per-ancestor inverse/quaternion order.
                // This exceptional getter path can still cost depth; folding
                // its ancestors would change the existing storage contract.
                TransformECSStore::PhysicsPoseReadScope cachedAuthor(transforms);
                if (transforms.HasFrameCacheWorldPositionOverride(handle))
                    value.position = current->GetLocalPosition();
                if (transforms.HasFrameCacheWorldRotationOverride(handle))
                    value.rotation = current->GetLocalRotation();
            }
            value.authored = transforms.GetAuthoredPoseRevision(handle);
            parentIndex = static_cast<uint32_t>(snapshot->transforms.size());
            snapshot->transforms.push_back(std::move(value));
            indices.emplace(current, parentIndex);
        }
        return parentIndex;
    };
    for (auto *renderer : renderers) {
        if (!renderer || !renderer->IsEnabled() || renderer->RequiresCpuPoseForGeometry())
            continue;
        auto *object = renderer->GetGameObject();
        if (!object || !object->GetScene() || object->GetScene()->GetRuntimeWorld() != &world ||
            !active.IsActive(object))
            continue;
        const auto existing = m_impl->entries.find(renderer);
        const auto &source = renderer->GetPoseBuffer();
        if (source && (existing == m_impl->entries.end() || existing->second.reference.GetComponent() != renderer ||
                       source != existing->second.source))
            continue; // Explicit user/NN pose bindings have priority.
        if (!hasActor(object->GetTransform()))
            continue;
        const auto transform = captureTransform(object->GetTransform());
        snapshot->renderers.push_back({renderer, PhysicsTargetReference(renderer), transform,
                                       renderer->GetSubmeshIndex() >= 0 ? renderer->GetMeshPivotOffset() : glm::vec3(0),
                                       renderer->GetLocalBoundsMin(), renderer->GetLocalBoundsMax()});
    }
    return snapshot;
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
    std::unordered_map<PhysicsActorHandle, std::shared_ptr<const JoltGpuScene::PublishedActor>, ActorHandleHash>
        actorByHandle;
    actorByHandle.reserve(publication.actors.size());
    for (uint32_t local = 0; local < publication.bodyCount; ++local) {
        const auto &actor = publication.actors[publication.bodyBase + local];
        if (actor)
            actorByHandle.emplace(actor->actor, actor);
    }
    std::vector<std::shared_ptr<const JoltGpuScene::PublishedActor>> actors(captured.transforms.size());
    for (size_t i = 0; i < captured.transforms.size(); ++i)
        if (const auto found = actorByHandle.find(captured.transforms[i].actor); found != actorByHandle.end())
            actors[i] = found->second;
    std::vector<uint32_t> nodeIndices(captured.transforms.size(), NoNode);
    for (size_t i = 0; i < actors.size(); ++i)
        if (actors[i]) {
            nodeIndices[i] = static_cast<uint32_t>(prepared.nodes.size());
            prepared.nodes.emplace_back();
        }
    struct OrdinaryChain
    {
        uint32_t root = NoNode;
        glm::mat4 affine{1};
        glm::mat4 inverse{1};
        glm::quat rotation{1, 0, 0, 0};
    };
    const OrdinaryChain noParent;
    std::vector<OrdinaryChain> chains(captured.transforms.size());
    // Capture is parent-first. Each ordinary path is composed once and ends
    // at the nearest accepted physical root; independent child actors restart
    // the path instead of inheriting the parent's numerical pose a second time.
    for (size_t i = 0; i < actors.size(); ++i) {
        const auto &transform = captured.transforms[i];
        const auto &parent = transform.parent == NoNode ? noParent : chains[transform.parent];
        auto &chain = chains[i];
        if (actors[i]) {
            auto &node = prepared.nodes[nodeIndices[i]];
            node.bodyParent = {actors[i]->slot, parent.root == NoNode ? NoNode : nodeIndices[parent.root], 0, 0};
            node.rootCOM = actors[i]->rootCenterOfMass;
            node.localScale = Float4(glm::vec4(transform.scale, 0));
            node.parentRotation = {parent.rotation.x, parent.rotation.y, parent.rotation.z, parent.rotation.w};
            node.parentLocalToAncestor0 = Float4(parent.affine[0]);
            node.parentLocalToAncestor1 = Float4(parent.affine[1]);
            node.parentLocalToAncestor2 = Float4(parent.affine[2]);
            node.parentLocalToAncestor3 = Float4(parent.affine[3]);
            node.parentAncestorToLocal0 = Float4(parent.inverse[0]);
            node.parentAncestorToLocal1 = Float4(parent.inverse[1]);
            node.parentAncestorToLocal2 = Float4(parent.inverse[2]);
            node.parentAncestorToLocal3 = Float4(parent.inverse[3]);
            chain.root = static_cast<uint32_t>(i);
        } else {
            chain.root = parent.root;
            chain.affine = parent.affine * transform.LocalMatrix();
            chain.inverse = transform.LocalInverse() * parent.inverse;
            chain.rotation = Multiply(parent.rotation, transform.rotation);
        }
    }
    for (size_t i = 0; i < captured.renderers.size(); ++i) {
        const auto &renderer = captured.renderers[i];
        const auto &chain = chains[renderer.transform];
        const auto root = chain.root;
        if (root == NoNode)
            continue;
        JPH::GPU::PoseBinding binding{};
        binding.nodeIndex = {nodeIndices[root], 0, 0, 0};
        binding.objectLocal0 = Float4(chain.affine[0]);
        binding.objectLocal1 = Float4(chain.affine[1]);
        binding.objectLocal2 = Float4(chain.affine[2]);
        binding.objectLocal3 = Float4(chain.affine[3]);
        binding.drawPivot = Float4(glm::vec4(renderer.pivot, 0));
        binding.boundsCenter = Float4(glm::vec4((renderer.boundsMin + renderer.boundsMax) * 0.5f, 0));
        binding.boundsExtent = Float4(glm::vec4((renderer.boundsMax - renderer.boundsMin) * 0.5f, 0));
        prepared.bindings.push_back(binding);
        prepared.targets.push_back({i, actors[root], Continuity(captured, actors[root]->bodyId)});
    }
    return prepared;
}

bool JoltGpuRenderBindings::Publish(const Prepared &prepared, const std::shared_ptr<rhi::ComputeBuffer> &models,
                                    const std::shared_ptr<rhi::ComputeBuffer> &bounds, uint64_t numericalGeneration)
{
    const auto &snapshot = *prepared.snapshot;
    if (snapshot.lifetime != m_impl->lifetime)
        return false; // A completed old job cannot reacquire bindings after Stop.
    auto &physics = PhysicsECSStore::Instance();
    const auto currentHierarchy = CurrentHierarchy(snapshot);
    ActiveHierarchy active;
    std::unordered_map<const Snapshot *, HierarchyComparison> hierarchyComparisons;
    std::unordered_set<MeshRenderer *> retained;
    std::vector<size_t> outputs;
    outputs.reserve(prepared.targets.size());
    bool allPublished = true;
    // Inspect the complete automatic set before changing any endpoint. Scene
    // extraction then sees either the previous or this completed generation,
    // even when an author edited one child while its export was pending.
    for (size_t output = 0; output < prepared.targets.size(); ++output) {
        const auto &target = prepared.targets[output];
        const auto &captured = snapshot.renderers[target.renderer];
        auto *renderer = captured.identity;
        if (captured.reference.GetComponent() != renderer || !renderer->IsEnabled() ||
            !active.IsActive(renderer->GetGameObject()) || renderer->RequiresCpuPoseForGeometry() ||
            !physics.IsValid(target.actor->actor) ||
            physics.GetActor(target.actor->actor).bodyId != target.actor->bodyId) {
            continue;
        }
        auto found = m_impl->entries.find(renderer);
        if (found != m_impl->entries.end() && found->second.reference.GetComponent() != renderer)
            found = m_impl->entries.end();
        const auto &current = renderer->GetPoseBuffer();
        if (current && (found == m_impl->entries.end() || current != found->second.source))
            continue;
        retained.insert(renderer);
        const auto pivot = renderer->GetSubmeshIndex() >= 0 ? renderer->GetMeshPivotOffset() : glm::vec3(0);
        const auto currentContinuity = m_impl->continuity.find(target.actor->bodyId);
        if (pivot != captured.pivot || renderer->GetLocalBoundsMin() != captured.boundsMin ||
            renderer->GetLocalBoundsMax() != captured.boundsMax || !currentHierarchy[captured.transform] ||
            Continuity(snapshot, target.actor->bodyId) !=
                (currentContinuity != m_impl->continuity.end() ? currentContinuity->second : 0)) {
            allPublished = false;
            continue;
        }
        outputs.push_back(output);
    }
    // Removed components and explicit replacement sources no longer belong
    // to the automatic publication. Clearing them cannot release a user source.
    for (auto it = m_impl->entries.begin(); it != m_impl->entries.end();) {
        if (retained.find(it->first) == retained.end()) {
            Impl::ClearOwned(it->first, it->second);
            it = m_impl->entries.erase(it);
        } else
            ++it;
    }
    if (!allPublished) {
        m_impl->published.reset();
        return false;
    }
    for (const auto output : outputs) {
        const auto &target = prepared.targets[output];
        const auto &captured = snapshot.renderers[target.renderer];
        auto *renderer = captured.identity;
        auto found = m_impl->entries.find(renderer);
        if (found != m_impl->entries.end() && found->second.reference.GetComponent() != renderer) {
            m_impl->entries.erase(found);
            found = m_impl->entries.end();
        }
        const auto &current = renderer->GetPoseBuffer();
        if (found == m_impl->entries.end())
            found = m_impl->entries.emplace(renderer, Impl::Entry{}).first;
        auto &entry = found->second;
        bool rebind = !entry.source || entry.actor != target.actor->actor || entry.bodyId != target.actor->bodyId ||
                      entry.bodyContinuity != target.continuity || current != entry.source || !entry.snapshot;
        if (!rebind) {
            auto &comparison = hierarchyComparisons[entry.snapshot.get()];
            if (!comparison.previous)
                comparison.previous = entry.snapshot;
            rebind = !comparison.Matches(snapshot, entry.transform, captured.transform);
        }
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
    m_impl->published = prepared.snapshot;
    m_impl->numericalGeneration = numericalGeneration;
    return true;
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
