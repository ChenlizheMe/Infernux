#include "JoltGpuRenderBindings.h"
#include "PhysicsWorld.h"

#include "../GameObject.h"
#include "../MeshRenderer.h"
#include "../RuntimeWorld.h"
#include "../Scene.h"
#include "../SceneManager.h"
#include "../Transform.h"
#include "../TransformECSStore.h"
#include "core/math/QuaternionEuler.h"
#include "core/math/TransformMatrix.h"
#include "core/math/TransformVectorMath.h"
#include <function/renderer/RenderPoseBuffer.h>

#include <algorithm>
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
        std::optional<glm::vec3> worldPosition;
        std::optional<glm::quat> worldRotation;
        bool gpuBody = false;

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
    struct CpuAnchor
    {
        glm::mat4 world, inverse;
        glm::quat rotation;
    };
    std::unordered_map<uint32_t, CpuAnchor> cpuAnchors;
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
        a.continuity != b.continuity || a.cpuAnchors.size() != b.cpuAnchors.size())
        return false;
    for (const auto &[index, anchor] : a.cpuAnchors) {
        const auto next = b.cpuAnchors.find(index);
        if (next == b.cpuAnchors.end() || next->second.world != anchor.world ||
            next->second.inverse != anchor.inverse || next->second.rotation != anchor.rotation)
            return false;
    }
    for (size_t i = 0; i < a.transforms.size(); ++i) {
        const auto &x = a.transforms[i];
        const auto &y = b.transforms[i];
        if (x.identity != y.identity || x.reference.GetComponent() != y.reference.GetComponent() ||
            x.actor != y.actor || x.bodyId != y.bodyId || x.parent != y.parent || x.world != y.world ||
            x.scale != y.scale || x.authored != y.authored || x.gpuBody != y.gpuBody)
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

JoltGpuRenderBindings::SnapshotPtr JoltGpuRenderBindings::Capture(RuntimeWorld &world, bool captureCpuConsumers) const
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
        bool hasActor = false, cpuAnchor = false, gpuBody = false, gpuPath = false, mixedCpuPath = false;
    };
    std::unordered_map<const Transform *, Ancestry> ancestry;
    std::unordered_map<const Transform *, uint32_t> indices;
    ActiveHierarchy active;
    std::vector<Transform *> ancestryPath, capturePath;
    const auto &renderers = SceneManager::Instance().GetActiveMeshRenderers();
    ancestry.reserve(renderers.size());
    indices.reserve(renderers.size());
    snapshot->renderers.reserve(renderers.size());
    const auto captureAncestry = [&](Transform *transform) -> const Ancestry & {
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
            if (ownActor) {
                value.bodyId = physics.GetActor(value.actor).bodyId;
                const auto *entry = PhysicsWorld::Instance().FindBodyRegistration(value.bodyId);
                value.cpuAnchor = entry && entry->backend == PhysicsWorld::Backend::Cpu;
                value.gpuBody = entry && entry->backend == PhysicsWorld::Backend::Gpu;
            }
            const auto *parent = value.parent ? &ancestry.at(value.parent) : nullptr;
            value.hasActor = !value.cpuAnchor && (ownActor || (parent && parent->hasActor));
            value.gpuPath = value.gpuBody || (parent && parent->gpuPath);
            value.mixedCpuPath = parent && (parent->mixedCpuPath || (value.cpuAnchor && parent->gpuPath));
            ancestry.emplace(current, value);
        }
        return ancestry.at(transform);
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
            value.gpuBody = capturedAncestry.gpuBody;
            const auto handle = current->GetECSHandle();
            value.position = transforms.GetLocalPosition(handle);
            value.rotation = transforms.GetLocalRotation(handle);
            value.scale = transforms.GetLocalScale(handle);
            if (transforms.IsFrameCacheActiveFor(handle)) {
                // Preserve Transform's per-ancestor inverse/quaternion order.
                // This exceptional getter path can still cost depth; folding
                // its ancestors would change the existing storage contract.
                TransformECSStore::PhysicsPoseReadScope cachedAuthor(transforms);
                if (transforms.HasFrameCacheWorldPositionOverride(handle)) {
                    value.worldPosition = transforms.GetCachedWorldPosition(handle.index);
                    value.position = current->GetLocalPosition();
                }
                if (transforms.HasFrameCacheWorldRotationOverride(handle)) {
                    value.worldRotation = transforms.GetCachedWorldRotation(handle.index);
                    value.rotation = current->GetLocalRotation();
                }
            }
            value.authored = transforms.GetAuthoredPoseRevision(handle);
            parentIndex = static_cast<uint32_t>(snapshot->transforms.size());
            if (capturedAncestry.cpuAnchor) {
                TransformECSStore::PhysicsPoseReadScope capture(transforms);
                snapshot->cpuAnchors.emplace(parentIndex, Snapshot::CpuAnchor{current->GetWorldMatrix(),
                                                                              current->GetWorldToLocalMatrix(),
                                                                              current->GetWorldRotation()});
            }
            snapshot->transforms.push_back(std::move(value));
            indices.emplace(current, parentIndex);
        }
        return parentIndex;
    };
    for (auto *renderer : renderers) {
        if (!renderer || !renderer->IsEnabled())
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
        const auto &path = captureAncestry(object->GetTransform());
        if (!path.hasActor || renderer->RequiresCpuPoseForGeometry()) {
            if (path.mixedCpuPath)
                captureTransform(object->GetTransform());
            continue;
        }
        const auto transform = captureTransform(object->GetTransform());
        snapshot->renderers.push_back({renderer, PhysicsTargetReference(renderer), transform,
                                       renderer->GetSubmeshIndex() >= 0 ? renderer->GetMeshPivotOffset() : glm::vec3(0),
                                       renderer->GetLocalBoundsMin(), renderer->GetLocalBoundsMax()});
    }
    if (captureCpuConsumers) {
        std::vector<Transform *> consumers;
        SceneManager::Instance().CollectPresentationWorldTransforms(consumers);
        for (auto *transform : consumers)
            if (captureAncestry(transform).mixedCpuPath)
                captureTransform(transform);
        // A new physics generation can dirty a palette whose previous work
        // has already left the pending queue. Use its registered hierarchy.
        for (const auto &[bone, skeleton] : world.render.skeletonBoneOwners)
            if (captureAncestry(bone).mixedCpuPath)
                captureTransform(bone);
    }
    return snapshot;
}

bool JoltGpuRenderBindings::HasMixedHierarchy(const Snapshot &snapshot) const
{
    return !snapshot.cpuAnchors.empty();
}

std::vector<Rigidbody *> JoltGpuRenderBindings::MixedGpuAncestors(const Snapshot &snapshot) const
{
    std::vector<Rigidbody *> result;
    std::vector<uint8_t> needed(snapshot.transforms.size());
    for (const auto &[index, anchor] : snapshot.cpuAnchors)
        if (const auto parent = snapshot.transforms[index].parent; parent != NoNode)
            needed[parent] = 1;
    for (size_t i = snapshot.transforms.size(); i > 0; --i) {
        if (needed[i - 1]) {
            const auto parent = snapshot.transforms[i - 1].parent;
            if (parent != NoNode)
                needed[parent] = 1;
        }
    }
    for (size_t i = 0; i < snapshot.transforms.size(); ++i) {
        const auto &value = snapshot.transforms[i];
        if (needed[i] && value.gpuBody && value.reference.GetComponent() == value.identity)
            if (auto *body = PhysicsWorld::Instance().FindRigidbodyByBodyId(value.bodyId))
                result.push_back(body);
    }
    return result;
}

JoltGpuRenderBindings::SnapshotPtr JoltGpuRenderBindings::ResolveMixedAnchors(SnapshotPtr snapshot,
                                                                              const MixedPose &pose) const
{
    auto resolved = std::make_shared<Snapshot>(*snapshot);
    const Snapshot::CpuAnchor root{glm::mat4(1), glm::mat4(1), glm::quat(1, 0, 0, 0)};
    std::vector<Snapshot::CpuAnchor> hierarchy(resolved->transforms.size());
    std::vector<uint32_t> path;
    const auto compose = [](const glm::mat4 &left, const glm::mat4 &right) {
        const auto result = transform_math::ComposeAffine(glm::value_ptr(left), glm::value_ptr(right));
        return glm::make_mat4(result.data());
    };
    constexpr auto position = static_cast<uint8_t>(PhysicsTargetChannels::Position);
    constexpr auto rotation = static_cast<uint8_t>(PhysicsTargetChannels::Rotation);
    for (size_t i = 0; i < resolved->transforms.size(); ++i) {
        auto &value = resolved->transforms[i];
        const auto &parent = value.parent == NoNode ? root : hierarchy[value.parent];
        auto worldPosition = value.worldPosition;
        auto worldRotation = value.worldRotation;
        const auto anchor = resolved->cpuAnchors.find(static_cast<uint32_t>(i));
        if (anchor != resolved->cpuAnchors.end() || value.gpuBody) {
            if (const auto sample = pose(value.bodyId, anchor != resolved->cpuAnchors.end())) {
                if (sample->channels & position)
                    worldPosition = sample->position;
                if (sample->channels & rotation)
                    worldRotation = sample->rotation;
            }
        }
        if (worldPosition || worldRotation || anchor != resolved->cpuAnchors.end()) {
            path.clear();
            for (auto index = value.parent; index != NoNode; index = resolved->transforms[index].parent)
                path.push_back(index);
        }
        if (worldPosition) {
            value.position = *worldPosition;
            // Match Transform::InverseTransformPoint, including collapsed scales.
            for (auto it = path.rbegin(); it != path.rend(); ++it) {
                const auto &ancestor = resolved->transforms[*it];
                const auto &q = ancestor.rotation;
                const auto delta = value.position - ancestor.position;
                const auto rotated = transform_math::RotateLocal({-q.x, -q.y, -q.z, q.w}, {delta.x, delta.y, delta.z});
                const auto scale =
                    transform_math::InverseLocalScale({ancestor.scale.x, ancestor.scale.y, ancestor.scale.z});
                value.position = {rotated[0] * scale[0], rotated[1] * scale[1], rotated[2] * scale[2]};
            }
        }
        if (worldRotation) {
            value.rotation = *worldRotation;
            for (auto it = path.rbegin(); it != path.rend(); ++it)
                value.rotation = Multiply(glm::conjugate(resolved->transforms[*it].rotation), value.rotation);
        }
        auto &result = hierarchy[i];
        result.world = compose(parent.world, value.LocalMatrix());
        result.inverse = compose(value.LocalInverse(), parent.inverse);
        if (anchor != resolved->cpuAnchors.end()) {
            result.rotation = value.rotation;
            for (const auto index : path)
                result.rotation = Multiply(resolved->transforms[index].rotation, result.rotation);
            anchor->second = result;
        }
    }
    return resolved;
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
    actorByHandle.reserve(publication.bodyCount);
    for (uint32_t local = 0; local < publication.bodyCount; ++local) {
        const auto &actor = publication.directory->actors[publication.bodyBase + local];
        if (actor)
            actorByHandle.emplace(actor->actor, actor);
    }
    std::vector<std::shared_ptr<const JoltGpuScene::PublishedActor>> actors(captured.transforms.size());
    for (size_t i = 0; i < captured.transforms.size(); ++i) {
        // A CPU anchor may retain a parked GPU allocation under the same ID.
        if (captured.cpuAnchors.count(static_cast<uint32_t>(i)))
            continue;
        if (const auto found = actorByHandle.find(captured.transforms[i].actor); found != actorByHandle.end())
            actors[i] = found->second;
    }
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
        if (const auto anchor = captured.cpuAnchors.find(static_cast<uint32_t>(i));
            anchor != captured.cpuAnchors.end()) {
            chain.affine = anchor->second.world;
            chain.inverse = anchor->second.inverse;
            chain.rotation = anchor->second.rotation;
        } else if (actors[i]) {
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
                                    const std::shared_ptr<rhi::ComputeBuffer> &bounds, uint64_t numericalGeneration,
                                    const std::function<void()> &commitTransforms)
{
    const auto &snapshot = *prepared.snapshot;
    if (snapshot.lifetime != m_impl->lifetime)
        return false; // A completed old job cannot reacquire bindings after Stop.
    auto &physics = PhysicsECSStore::Instance();
    const auto currentHierarchy = CurrentHierarchy(snapshot);
    if (commitTransforms &&
        std::find(currentHierarchy.begin(), currentHierarchy.end(), uint8_t(0)) != currentHierarchy.end())
        return false;
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
    if (commitTransforms) {
        if (!allPublished)
            return false;
        commitTransforms();
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
