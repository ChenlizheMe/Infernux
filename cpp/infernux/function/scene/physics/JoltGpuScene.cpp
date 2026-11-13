#include "JoltGpuScene.h"

#include "JoltGpuBodyGrowth.h"
#include "JoltGpuBodyMutations.h"
#include "JoltGpuBroadphase.h"
#include "JoltGpuConstraints.h"
#include "JoltGpuJoints.h"
#include <function/scene/Collider.h>
#include <function/scene/GameObject.h>
#include <function/scene/Rigidbody.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace infernux
{
namespace
{
using namespace JPH::GPU;

BodyModel Tombstone()
{
    BodyModel body{};
    body.meta = {0, 0, MotionStatic | MotionNotResident, 0};
    body.inertiaRotation.w = 1.0f;
    return body;
}

BodyState EmptyState()
{
    BodyState state{};
    state.rotation.w = 1.0f;
    return state;
}

JoltGpuModel EmptyModel(uint32_t capacity)
{
    JoltGpuModel model;
    model.bodies.reserve(capacity);
    for (uint32_t slot = 0; slot < capacity; ++slot)
        model.bodies.push_back(Tombstone());
    model.worlds.push_back({{0, capacity, 0, 0}, {0, 0, 0, 0}});
    return model;
}

bool FiniteVector(const Float4 &value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) && value.w == 0;
}

JoltGpuControlWork Then(JoltGpuControlWork work, std::function<JoltGpuControlWork()> next)
{
    if (!work.packet && !work.continueWork)
        return next();
    auto continuation = std::move(work.continueWork);
    work.continueWork = [continuation = std::move(continuation), next = std::move(next)]() mutable {
        return continuation ? Then(continuation(), std::move(next)) : next();
    };
    return work;
}

} // namespace

class JoltGpuScene::Impl
{
  public:
    struct TargetState
    {
        bool pending = false;
        bool temporaryStatic = false;
        PhysicsTargetSource position;
        PhysicsTargetSource rotation;
    };

    struct TargetCompletion
    {
        PhysicsTargetSource source;
        uint8_t channels = 0;
    };

    struct PrefixTarget
    {
        uint32_t bodyId = InvalidBodyId;
        TargetState target;
    };

    struct Slot
    {
        uint32_t bodyId = InvalidBodyId;
        PhysicsActorHandle actor;
        OwnerIdentity owner;
        bool desiredResident = false;
        // Initial author inputs are claimed by the first nonstatic group for a
        // component generation. Static groups must not release that claim.
        PhysicsECSStore::RigidbodyHandle initialRigidbody;
        std::shared_ptr<const PublishedActor> accepted;
        // Shares the actual pending transaction record. After acceptance these
        // pointers normally name the same object; no second model is cached.
        std::shared_ptr<const PublishedActor> authored;
        // Ordered command provenance, without a numerical pose/velocity copy.
        // A candidate captures its own prefix before future author calls.
        TargetState queuedTarget;
    };

    struct ActorChange
    {
        uint32_t slot;
        uint32_t bodyId;
        BodyModel model;
        std::vector<ShapeModel> leaves;
        BodyState initialState;
        PhysicsECSStore::RigidbodyHandle initialRigidbody;
        bool hasInitialLinearVelocity = false;
        bool hasInitialAngularVelocity = false;
        std::vector<PreparedGpuActor::ColliderCenter> colliderCenters;
        std::shared_ptr<PublishedActor> identity;
        bool creation = false;
        bool initializesRigidbody = false;
    };

    struct JointRecord
    {
        PublishedJoint value;
        PhysicsConstraintOwner *owner = nullptr;
    };

    static uint8_t PositionChannel() noexcept
    {
        return static_cast<uint8_t>(PhysicsTargetChannels::Position);
    }

    static uint8_t RotationChannel() noexcept
    {
        return static_cast<uint8_t>(PhysicsTargetChannels::Rotation);
    }

    static bool HasAuthor(const PhysicsTargetSource &source, uint8_t channel) noexcept
    {
        const auto *author = source.Get();
        return author && (author->channels & channel);
    }

    static void AddCompletion(std::vector<TargetCompletion> &completions, const PhysicsTargetSource &source,
                              uint8_t channel)
    {
        if (HasAuthor(source, channel))
            completions.push_back({source, channel});
    }

    static void CancelTarget(TargetState &target, std::vector<TargetCompletion> &completions)
    {
        AddCompletion(completions, target.position, PositionChannel());
        AddCompletion(completions, target.rotation, RotationChannel());
        target = {};
    }

    PhysicsTargetSource AppendSource(const PhysicsTargetAuthor &author)
    {
        if (!pendingSources) {
            pendingSources = std::make_shared<PhysicsTargetBatch>();
            pendingSources->records.reserve(slots.size() * 2);
        }
        if (pendingSources->records.size() == UINT32_MAX)
            throw std::length_error("Jolt GPU author command source batch exceeds its index range");
        const auto index = static_cast<uint32_t>(pendingSources->records.size());
        pendingSources->records.push_back(author);
        return {pendingSources, index};
    }

    /// Model groups are prepared lazily, so carry flags must come from this
    /// already ordered source shadow, never from the later Rigidbody guard.
    static uint32_t CarryFlags(const TargetState &target) noexcept
    {
        return (HasAuthor(target.position, PositionChannel()) ? MotionBoundaryCarryAuthoredPosition : 0) |
               (HasAuthor(target.rotation, RotationChannel()) ? MotionBoundaryCarryAuthoredRotation : 0);
    }

    Impl(rhi::ComputeHost &host, FrameWorkOwner &work, uint32_t capacity, const ShaderTargetCode &stepTarget,
         const ShaderTargetCode &mutationTarget, const ShaderTargetCode &commandTarget,
         const ShaderTargetCode &broadphaseTarget, const ShaderTargetCode &constraintTarget,
         JoltGpuJoints &jointProgram)
        : actors(PhysicsECSStore::Instance()), storage(host, EmptyModel(capacity)), programs(host, stepTarget),
          broadphase(host, broadphaseTarget), constraints(host, constraintTarget), mutations(host, mutationTarget),
          commands(host, commandTarget), joints(jointProgram), slots(capacity),
          tick(work, storage, programs, broadphase, constraints)
    {
        byId.reserve(capacity);
        freeSlots.reserve(capacity);
        for (uint32_t index = capacity; index > 0; --index)
            freeSlots.push_back(index - 1);
        initialization = storage.PrepareInitialization(std::vector<BodyState>(capacity, EmptyState()),
                                                       std::vector<BodyForces>(capacity));
    }

    ~Impl()
    {
        Cancel();
    }

    void RequireUsable() const
    {
        if (cancelled || tick.GetStatus() == JoltGpuTick::Status::Cancelled ||
            tick.GetStatus() == JoltGpuTick::Status::Failed)
            throw std::logic_error("A cancelled or failed Jolt GPU scene must retire before replacement");
    }

    uint32_t RequireSlot(uint32_t id) const
    {
        RequireUsable();
        const auto found = byId.find(id);
        if (found == byId.end())
            throw std::invalid_argument("Jolt GPU body identity is invalid or retired");
        return found->second;
    }

    void RequireOwnership(uint32_t slot, uint32_t id) const
    {
        if (slots[slot].bodyId != id)
            throw std::logic_error("Jolt GPU control references a reused body slot");
    }

    const PublishedJoint &GetAuthoredJoint(uint64_t id) const
    {
        RequireUsable();
        const auto found = authoredJoints.find(id);
        if (found == authoredJoints.end())
            throw std::invalid_argument("GPU joint identity is invalid or retired");
        return found->second.value;
    }

    PublishedJoint &AcceptedJoint(uint64_t id)
    {
        const auto found = std::find_if(acceptedJoints.begin(), acceptedJoints.end(),
                                        [id](const auto &joint) { return joint.id == id; });
        if (found == acceptedJoints.end())
            throw std::logic_error("GPU joint update precedes its accepted creation");
        return *found;
    }

    void QueueJointTopology()
    {
        auto catalog = std::make_shared<std::vector<PublishedJoint>>();
        catalog->reserve(authoredJoints.size());
        for (auto &[id, joint] : authoredJoints) {
            joint.value.row = static_cast<uint32_t>(catalog->size());
            catalog->push_back(joint.value);
        }
        JoltGpuControlGroup group;
        group.prepareModel = [this, catalog] {
            std::unordered_map<uint64_t, uint32_t> previous;
            for (const auto &joint : acceptedJoints)
                previous.emplace(joint.id, joint.row);
            std::vector<JointAuthor> authors;
            std::vector<uint32_t> remap;
            authors.reserve(catalog->size());
            remap.reserve(catalog->size());
            for (const auto &joint : *catalog) {
                RequireOwnership(joint.author.bodies.y, joint.bodyIdA);
                if (joint.bodyIdB != InvalidBodyId)
                    RequireOwnership(joint.author.bodies.x, joint.bodyIdB);
                authors.push_back(joint.author);
                const auto old = previous.find(joint.id);
                remap.push_back(old == previous.end() ? InvalidQueryBodySlot : old->second);
            }
            auto prepared =
                joints.PrepareReplacement(authors, remap, storage.Get(Resource::Joints)->Freeze(),
                                          storage.GetJointCount(), storage.Get(Resource::BodyStates)->Freeze());
            return storage.PrepareJointReplacement(std::move(authors), std::move(prepared.stream),
                                                   std::move(prepared.updates), std::move(prepared.dispatches));
        };
        group.modelAccepted = [this, catalog]() noexcept { acceptedJoints.swap(*catalog); };
        pending.push_back(std::move(group));
    }

    uint64_t CreateJoint(PhysicsConstraintOwner &owner, uint32_t ownerBody, uint32_t connectedBody, JointAuthor author)
    {
        const auto a = RequireSlot(ownerBody);
        const auto b = connectedBody == InvalidBodyId ? InvalidQueryBodySlot : RequireSlot(connectedBody);
        if (ownerBody == connectedBody || !slots[a].authored || (b != InvalidQueryBodySlot && !slots[b].authored))
            throw std::invalid_argument("A GPU joint requires distinct authored body endpoints");
        if (!nextJointId)
            throw std::overflow_error("GPU joint identity space is exhausted");
        const auto id = nextJointId++;
        author.bodies.x = b;
        author.bodies.y = a;
        author.identity = {static_cast<uint32_t>(id), static_cast<uint32_t>(id >> 32), 0, 0};
        author.rootCOM1 = b == InvalidQueryBodySlot ? Float4{} : slots[b].authored->rootCenterOfMass;
        author.rootCOM2 = slots[a].authored->rootCenterOfMass;
        author.scale1 = b == InvalidQueryBodySlot ? Float4{1, 1, 1, 0} : slots[b].authored->rootScale;
        author.scale2 = slots[a].authored->rootScale;
        PublishedJoint value{id, 0, ownerBody, connectedBody, author.bodies.z, author};
        authoredJoints.emplace(id, JointRecord{value, &owner});
        QueueJointTopology();
        return id;
    }

    void UpdateJoint(uint64_t id, const std::function<void(JointAuthor &)> &update)
    {
        auto author = GetAuthoredJoint(id).author;
        update(author);
        UpdateJoints({{id, author}});
    }

    void UpdateJoints(std::vector<JointUpdate> updates)
    {
        if (updates.empty())
            return;
        auto values = std::make_shared<std::vector<PublishedJoint>>();
        values->reserve(updates.size());
        for (const auto &update : updates) {
            auto value = GetAuthoredJoint(update.id);
            value.author = update.author;
            values->push_back(value);
        }
        JoltGpuControlGroup group;
        group.prepareModel = [this, values] {
            std::vector<JoltGpuJointModelUpdate> changes;
            changes.reserve(values->size());
            for (auto &value : *values) {
                value.row = AcceptedJoint(value.id).row;
                changes.push_back({value.row, value.author});
            }
            return joints.PrepareUpdate(storage, std::move(changes));
        };
        group.modelAccepted = [this, values]() noexcept {
            for (const auto &value : *values)
                acceptedJoints[value.row].author = value.author;
        };
        pending.push_back(std::move(group));
        for (const auto &value : *values)
            authoredJoints.at(value.id).value.author = value.author;
    }

    void RefreshJointBody(uint32_t bodyId, const PublishedActor &actor)
    {
        std::vector<JointUpdate> changes;
        for (const auto &[id, joint] : authoredJoints)
            if (joint.value.bodyIdA == bodyId || joint.value.bodyIdB == bodyId) {
                auto author = joint.value.author;
                if (joint.value.bodyIdA == bodyId) {
                    author.rootCOM2 = actor.rootCenterOfMass;
                    author.scale2 = actor.rootScale;
                } else {
                    author.rootCOM1 = actor.rootCenterOfMass;
                    author.scale1 = actor.rootScale;
                }
                changes.push_back({id, author});
            }
        UpdateJoints(std::move(changes));
    }

    void DestroyJoint(uint64_t id)
    {
        const auto found = authoredJoints.find(id);
        if (found == authoredJoints.end())
            return;
        auto *owner = found->second.owner;
        authoredJoints.erase(found);
        QueueJointTopology();
        owner->OnPhysicsConstraintDestroyed();
    }

    void DestroyJointsForBody(uint32_t bodyId)
    {
        std::vector<PhysicsConstraintOwner *> owners;
        for (auto it = authoredJoints.begin(); it != authoredJoints.end();) {
            if (it->second.value.bodyIdA == bodyId || it->second.value.bodyIdB == bodyId) {
                owners.push_back(it->second.owner);
                it = authoredJoints.erase(it);
            } else {
                ++it;
            }
        }
        if (!owners.empty())
            QueueJointTopology();
        for (auto *owner : owners)
            owner->OnPhysicsConstraintDestroyed();
    }

    void ReserveCapacity(uint32_t capacity)
    {
        const auto previous = static_cast<uint32_t>(slots.size());
        if (capacity <= previous)
            return;
        if (!storage.GetHost().device.GetCapabilities().IsStorageBufferRangeValid(0, uint64_t(capacity) *
                                                                                         sizeof(BodyModel)))
            throw std::length_error("Jolt GPU body capacity exceeds the device storage range");
        auto growth = std::make_shared<JoltGpuBodyGrowth>(storage, mutations, previous, capacity);
        JoltGpuControlGroup preparation, acceptance;
        preparation.prepareWork = [growth] { return growth->Prepare(); };
        acceptance.prepareModel = [growth] { return growth->Commit(); };
        pending.reserve(pending.size() + 2);
        freeSlots.reserve(capacity);
        slots.resize(capacity);
        for (uint32_t index = capacity; index > previous; --index)
            freeSlots.push_back(index - 1);
        pending.push_back(std::move(preparation));
        pending.push_back(std::move(acceptance));
    }

    BodyAllocation Reserve(PhysicsActorHandle actor, OwnerIdentity owner)
    {
        RequireUsable();
        if (!actors.IsValid(actor))
            throw std::invalid_argument("Jolt GPU body requires a generation-aware actor handle");
        const auto *object = actors.GetActor(actor).owner;
        if (!object || owner.gameObjectId != object->GetID() || owner.target.GetGameObject() != object)
            throw std::invalid_argument("Jolt GPU body owner identity does not resolve to its actor");
        if (nextBodyId == InvalidBodyId)
            throw std::overflow_error("Jolt GPU public body identity space is exhausted");
        for (const auto &[id, index] : byId)
            if (slots[index].actor == actor)
                throw std::logic_error("The physics actor already owns a Jolt GPU body");
        if (freeSlots.empty())
            throw std::length_error("Jolt GPU scene has no retired capacity slot available");
        const auto index = freeSlots.back();
        const auto id = nextBodyId;
        byId.emplace(id, index);
        freeSlots.pop_back();
        auto &slot = slots[index];
        slot.bodyId = id;
        slot.actor = actor;
        slot.owner = std::move(owner);
        slot.desiredResident = false;
        slot.initialRigidbody = {};
        ++nextBodyId;
        return {id, index};
    }

    void CommitActor(const std::shared_ptr<ActorChange> &change) noexcept
    {
        auto &slot = slots[change->slot];
        slot.accepted = change->identity;
        if (!actors.IsValid(change->identity->actor))
            return;
        auto &actor = actors.GetActor(change->identity->actor);
        if (actor.bodyId != change->bodyId)
            return;
        actor.shapeRevision = change->identity->shapeRevision;
        for (const auto &center : change->colliderCenters)
            if (actors.IsValid(center.collider)) {
                auto &collider = actors.GetCollider(center.collider);
                if (collider.actorHandle == change->identity->actor)
                    collider.publishedRigidCenter = center.center;
            }
        if (change->initializesRigidbody && actors.IsValid(change->initialRigidbody)) {
            auto &body = actors.GetRigidbody(change->initialRigidbody);
            if (body.actorHandle != change->identity->actor)
                return;
            const auto &linear = change->initialState.linearVelocity;
            const auto &angular = change->initialState.angularVelocity;
            if (change->hasInitialLinearVelocity && body.hasLinearVelocity && body.linearVelocity.x == linear.x &&
                body.linearVelocity.y == linear.y && body.linearVelocity.z == linear.z)
                body.hasLinearVelocity = false;
            if (change->hasInitialAngularVelocity && body.hasAngularVelocity && body.angularVelocity.x == angular.x &&
                body.angularVelocity.y == angular.y && body.angularVelocity.z == angular.z)
                body.hasAngularVelocity = false;
        }
    }

    JoltGpuModel RepackModel(uint32_t changedSlot, BodyModel changedBody,
                             const std::vector<ShapeModel> &changedLeaves) const
    {
        // Only this transaction's actor is new. Everything else comes from the
        // actual accepted model, including earlier groups in this same prefix.
        const auto &accepted = storage.GetModel();
        const uint64_t leafCount =
            uint64_t(accepted.shapes.size()) - accepted.bodies[changedSlot].meta.y + changedLeaves.size();
        if (leafCount > std::numeric_limits<uint32_t>::max())
            throw std::length_error("Jolt GPU scene exceeds its leaf index range");
        JoltGpuModel replacement;
        replacement.bodies = accepted.bodies;
        replacement.shapes.reserve(static_cast<size_t>(leafCount));
        for (uint32_t slot = 0; slot < replacement.bodies.size(); ++slot) {
            const auto &old = accepted.bodies[slot];
            auto body = slot == changedSlot ? changedBody : old;
            body.meta.x = static_cast<uint32_t>(replacement.shapes.size());
            body.meta.y = slot == changedSlot ? static_cast<uint32_t>(changedLeaves.size()) : old.meta.y;
            for (uint32_t leaf = 0; leaf < body.meta.y; ++leaf) {
                auto model = slot == changedSlot ? changedLeaves[leaf] : accepted.shapes[old.meta.x + leaf];
                model.meta.y = slot;
                replacement.shapes.push_back(model);
            }
            replacement.bodies[slot] = body;
        }
        const auto count = static_cast<uint32_t>(leafCount);
        replacement.worlds.push_back({{0, static_cast<uint32_t>(replacement.bodies.size()), 0, 0}, {0, count, 0, 0}});
        return replacement;
    }

    void StageActor(uint32_t id, PreparedGpuActor actor)
    {
        const uint32_t index = RequireSlot(id);
        auto &slot = slots[index];
        const bool firstOwnership = !slot.authored;
        const auto center = actor.geometry.centerOfMass;
        const Float4 rootCOM{center.GetX(), center.GetY(), center.GetZ(), 0.0f};
        if (!FiniteVector(rootCOM) || actor.body.meta.y != actor.geometry.leaves.size())
            throw std::invalid_argument("Jolt GPU actor has inconsistent geometry or an invalid root COM");
        auto identity = std::make_shared<PublishedActor>();
        identity->bodyId = id;
        identity->slot = index;
        identity->acceptedMotionType = actor.body.meta.z & ~MotionNotResident;
        identity->actor = slot.actor;
        identity->rigidbody = actor.initialRigidbody;
        identity->rigidbodyComponentId = actor.rigidbodyComponentId;
        identity->owner = slot.owner;
        identity->rootCenterOfMass = rootCOM;
        identity->rootScale = actor.rootScale;
        identity->authoredShape = std::move(actor.authoredShape);
        identity->leaves.reserve(actor.geometry.leaves.size());
        for (const auto &leaf : actor.geometry.leaves) {
            if (leaf.memberIndex >= actor.members.size() || !actor.members[leaf.memberIndex])
                throw std::invalid_argument("Jolt GPU leaf has no contributing Collider identity");
            const auto *surface = actor.members[leaf.memberIndex];
            PublishedLeaf publishedLeaf;
            publishedLeaf.surfaceTarget = PhysicsTargetReference(surface);
            publishedLeaf.colliderId = surface->GetComponentID();
            if (auto *object = surface->GetGameObject())
                publishedLeaf.surfaceGameObjectId = object->GetID();
            publishedLeaf.subShapeId = leaf.subShapeID.GetValue();
            publishedLeaf.subShapeBits = leaf.subShapeBits;
            publishedLeaf.memberIndex = leaf.memberIndex;
            publishedLeaf.shape = leaf.shape;
            publishedLeaf.sensor = (leaf.model.meta.w & ShapeSensor) != 0;
            identity->leaves.push_back(std::move(publishedLeaf));
        }
        // No component pointer escapes this capture boundary.
        actor.members.clear();
        auto change = std::make_shared<ActorChange>();
        change->slot = index;
        change->bodyId = id;
        change->model = actor.body;
        change->model.meta.x = 0;
        change->model.meta.w = 0;
        change->model.meta.z =
            (change->model.meta.z & ~MotionNotResident) | (slot.desiredResident ? 0 : MotionNotResident);
        change->leaves.reserve(actor.geometry.leaves.size());
        for (size_t ordinal = 0; ordinal < actor.geometry.leaves.size(); ++ordinal) {
            auto &leaf = actor.geometry.leaves[ordinal];
            leaf.model.meta.y = index;
            const auto colliderId = identity->leaves[ordinal].colliderId;
            leaf.model.source.x = static_cast<uint32_t>(colliderId);
            leaf.model.source.y = static_cast<uint32_t>(colliderId >> 32);
            change->leaves.push_back(std::move(leaf.model));
        }
        change->initialState = actor.initialState;
        change->initialRigidbody = actor.initialRigidbody;
        change->hasInitialLinearVelocity = actor.hasInitialLinearVelocity;
        change->hasInitialAngularVelocity = actor.hasInitialAngularVelocity;
        change->colliderCenters = std::move(actor.colliderCenters);
        change->identity = std::move(identity);
        const auto motion = change->identity->acceptedMotionType;
        const bool motionChanged = !firstOwnership && slot.authored->acceptedMotionType != motion;
        change->initializesRigidbody =
            motion != MotionStatic && (firstOwnership || slot.initialRigidbody != change->initialRigidbody);
        // Reserve before transferring the command group, so subsequent source
        // cancellation cannot fail after this group has acquired ownership.
        pendingCompletions.reserve(pendingCompletions.size() + 2);

        JoltGpuControlGroup group;
        group.prepareModel = [this, change] {
            RequireOwnership(change->slot, change->bodyId);
            const auto &previous = slots[change->slot].accepted;
            change->creation = !previous;
            change->identity->shapeRevision = previous ? previous->shapeRevision + 1 : 0;
            std::vector<BodyInitialization> initial;
            std::vector<CenterOfMassUpdate> rebases;
            if (change->creation) {
                initial.push_back({{change->slot, 0, 0, 0}, change->initialState, {}});
            } else {
                const auto oldCOM = previous->rootCenterOfMass;
                const auto newCOM = change->identity->rootCenterOfMass;
                rebases.push_back(
                    {{change->slot, 0, 0, 0}, {newCOM.x - oldCOM.x, newCOM.y - oldCOM.y, newCOM.z - oldCOM.z, 0.0f}});
            }
            const auto &acceptedBody = storage.GetModel().bodies[change->slot];
            if (acceptedBody.meta.y == change->leaves.size()) {
                auto body = change->model;
                body.meta.x = acceptedBody.meta.x;
                std::vector<JoltGpuShapeModelUpdate> leaves;
                leaves.reserve(change->leaves.size());
                for (uint32_t leaf = 0; leaf < change->leaves.size(); ++leaf)
                    leaves.push_back({acceptedBody.meta.x + leaf, change->leaves[leaf]});
                return mutations.Prepare(storage, {{change->slot, body}}, std::move(leaves), initial, rebases);
            }
            return mutations.PrepareReplacement(storage, RepackModel(change->slot, change->model, change->leaves),
                                                initial, rebases);
        };
        group.modelAccepted = [this, change]() noexcept { CommitActor(change); };
        if (motionChanged) {
            BodyCommand boundary{};
            boundary.meta = {index, CommandMotionBoundary, MotionBoundaryTypeChange | CarryFlags(slot.queuedTarget), 0};
            group.commands.push_back(boundary);
        }
        if (motion != MotionStatic) {
            // A new slot clamps its initial values. A new Rigidbody attached
            // to an existing actor applies only that component's captured
            // pending inputs, preserving the resident numerical pose/state.
            if (firstOwnership || (change->initializesRigidbody && change->hasInitialLinearVelocity)) {
                BodyCommand linear{};
                linear.meta = {index, CommandSetLinearVelocity, 0, 0};
                linear.value = change->initialState.linearVelocity;
                group.commands.push_back(linear);
            }
            if (firstOwnership || (change->initializesRigidbody && change->hasInitialAngularVelocity)) {
                BodyCommand angular{};
                angular.meta = {index, CommandSetAngularVelocity, 0, 0};
                angular.value = change->initialState.angularVelocity;
                group.commands.push_back(angular);
            }
        }
        pending.push_back(std::move(group));
        if (motionChanged || (motion != MotionStatic && (firstOwnership || (change->initializesRigidbody &&
                                                                            (change->hasInitialLinearVelocity ||
                                                                             change->hasInitialAngularVelocity)))))
            CancelTarget(slot.queuedTarget, pendingCompletions);
        if (motion != MotionStatic)
            slot.initialRigidbody = change->initialRigidbody;
        slot.authored = change->identity;
        RefreshJointBody(id, *change->identity);
    }

    void SetResident(uint32_t id, bool resident, bool preserveVelocity)
    {
        const uint32_t index = RequireSlot(id);
        auto &slot = slots[index];
        if (slot.desiredResident == resident && (resident || preserveVelocity))
            return;
        JoltGpuControlGroup group;
        group.prepareModel = [this, index, id, resident] {
            RequireOwnership(index, id);
            auto model = storage.GetModel().bodies[index];
            model.meta.z = (model.meta.z & ~MotionNotResident) | (resident ? 0 : MotionNotResident);
            return storage.PrepareModelUpdate({{index, model}}, {});
        };
        if (!resident) {
            pendingCompletions.reserve(pendingCompletions.size() + 2);
            BodyCommand boundary{};
            boundary.meta = {index, CommandMotionBoundary,
                             preserveVelocity ? MotionBoundarySuspendPreserve : MotionBoundarySuspendReset, 0};
            group.commands.push_back(boundary);
        }
        pending.push_back(std::move(group));
        if (!resident)
            CancelTarget(slot.queuedTarget, pendingCompletions);
        slot.desiredResident = resident;
    }

    PhysicsTargetSource QueueCommand(uint32_t id, BodyCommand command, const PhysicsTargetAuthor *author)
    {
        const uint32_t index = RequireSlot(id);
        auto &slot = slots[index];
        if (!slot.authored)
            throw std::logic_error("Jolt GPU actor must be staged before authoring a body command");
        command.meta.x = index;
        const auto kind = command.meta.y;
        const bool move = kind >= CommandMovePosition && kind <= CommandMovePose;
        const uint8_t channels = (kind == CommandMovePosition   ? PositionChannel()
                                  : kind == CommandMoveRotation ? RotationChannel()
                                  : kind == CommandMovePose     ? PositionChannel() | RotationChannel()
                                                                : 0);
        const auto motion = slot.authored->acceptedMotionType;
        const bool acceptsMove = move && (motion == MotionKinematic ||
                                          (motion == MotionStatic && (command.meta.z & CommandTemporaryStatic)));
        if (author && (!acceptsMove || author->bodyId != id || author->channels != channels ||
                       !actors.IsValid(slot.authored->rigidbody) ||
                       author->rigidbody.GetComponent() != actors.GetRigidbody(slot.authored->rigidbody).owner ||
                       author->ownerTransform.GetComponent() != slot.owner.target.GetComponent()))
            throw std::invalid_argument("Jolt GPU author target does not match its body or command channels");
        if (kind <= CommandSetPose || move) {
            const float speed = command.rootCOM.w;
            command.rootCOM = slot.authored->rootCenterOfMass;
            if (move)
                command.rootCOM.w = speed;
        }
        pendingCompletions.reserve(pendingCompletions.size() + 2);
        const auto source = author ? AppendSource(*author) : PhysicsTargetSource{};
        if (!pending.empty() && !pending.back().prepareModel && !pending.back().modelAccepted &&
            !pending.back().prepareWork) {
            pending.back().commands.push_back(command);
        } else {
            JoltGpuControlGroup group;
            group.commands.push_back(command);
            pending.push_back(std::move(group));
        }
        auto &target = slot.queuedTarget;
        if (acceptsMove) {
            if (!target.pending) {
                // The device samples the other channel from its current pose
                // for the first Pending command, rather than an old drive.
                CancelTarget(target, pendingCompletions);
            }
            if (channels & PositionChannel()) {
                AddCompletion(pendingCompletions, target.position, PositionChannel());
                target.position = source;
            }
            if (channels & RotationChannel()) {
                AddCompletion(pendingCompletions, target.rotation, RotationChannel());
                target.rotation = source;
            }
            target.pending = true;
            target.temporaryStatic = motion == MotionStatic;
        } else if ((kind <= CommandSetPose && !(command.meta.z & CommandPreserveKinematicMove)) ||
                   kind == CommandResetMotion || kind == CommandMotionBoundary ||
                   ((kind == CommandSetLinearVelocity || kind == CommandSetAngularVelocity) &&
                    motion != MotionStatic)) {
            CancelTarget(target, pendingCompletions);
        }
        return source;
    }

    void StageRetire(uint32_t id)
    {
        const uint32_t index = RequireSlot(id);
        DestroyJointsForBody(id);
        JoltGpuControlGroup group;
        group.prepareModel = [this, index, id] {
            RequireOwnership(index, id);
            const BodyInitialization initial{{index, 0, 0, 0}, EmptyState(), {}};
            const auto &body = storage.GetModel().bodies[index];
            if (body.meta.y == 0) {
                auto tombstone = Tombstone();
                tombstone.meta.x = body.meta.x;
                return mutations.Prepare(storage, {{index, tombstone}}, {}, {initial});
            }
            return mutations.PrepareReplacement(storage, RepackModel(index, Tombstone(), {}), {initial});
        };
        group.modelAccepted = [this, index]() noexcept {
            slots[index] = Slot{};
            // Every reservation pops once and every retirement pushes once;
            // Capacity planning reserves the complete slot range up front.
            freeSlots.push_back(index);
        };
        pendingCompletions.reserve(pendingCompletions.size() + 2);
        pending.push_back(std::move(group));
        CancelTarget(slots[index].queuedTarget, pendingCompletions);
        byId.erase(id);
    }

    JoltGpuTickControl PrepareControl(uint64_t epoch, uint64_t step, bool controlOnly, Consumer consumer = {},
                                      Consumer finalConsumer = {})
    {
        RequireUsable();
        if (tick.GetStatus() != JoltGpuTick::Status::Idle && tick.GetStatus() != JoltGpuTick::Status::Complete)
            throw std::logic_error("Jolt GPU scene cannot admit a second logical tick");
        if (candidate)
            throw std::logic_error("Publish the previous completed Jolt GPU tick before admitting its successor");
        const auto prefixCapacity = slots.size();
        auto publication = std::make_shared<Publication>();
        publication->simulationEpoch = epoch;
        publication->fixedStepIndex = step;
        publication->actors.resize(prefixCapacity);
        publication->residentBodies.resize(prefixCapacity);
        auto prefix = std::make_shared<std::vector<PrefixTarget>>();
        prefix->reserve(prefixCapacity);
        for (const auto &slot : slots)
            prefix->push_back({slot.bodyId, slot.queuedTarget});
        auto completions = std::make_shared<std::vector<TargetCompletion>>(pendingCompletions);
        completions->reserve(completions->size() + prefixCapacity * 2);
        JoltGpuControlGroup capture;
        capture.prepareModel = [this, publication, prefix, completions, prefixCapacity,
                                controlOnly]() -> std::unique_ptr<JoltGpuModelUpdate> {
            if (storage.GetBodyCount() != prefixCapacity)
                throw std::logic_error("Jolt GPU publication must match its accepted body-capacity prefix");
            if (storage.GetBodyCount()) {
                publication->bodyModels = storage.Get(Resource::BodyModels)->Freeze();
                if (storage.GetShapeCount())
                    publication->shapeModels = storage.Get(Resource::ShapeModels)->Freeze();
                publication->worlds = storage.Get(Resource::Worlds)->Freeze();
                if (controlOnly)
                    publication->bodyStates = storage.Get(Resource::BodyStates)->Freeze();
            }
            publication->joints = acceptedJoints;
            if (controlOnly && storage.GetJointCount())
                publication->jointStream = storage.Get(Resource::Joints)->Freeze();
            publication->leaves.resize(storage.GetShapeCount());
            for (size_t index = 0; index < prefixCapacity; ++index) {
                publication->actors[index] = slots[index].accepted;
                publication->residentBodies[index] = (storage.GetModel().bodies[index].meta.z & MotionNotResident) == 0;
                const auto &acceptedBody = storage.GetModel().bodies[index];
                if (acceptedBody.meta.y != 0) {
                    const auto &identity = publication->actors[index];
                    if (!identity || identity->leaves.size() != acceptedBody.meta.y)
                        throw std::logic_error("Jolt GPU publication leaf catalog differs from its accepted range");
                    for (uint32_t leaf = 0; leaf < acceptedBody.meta.y; ++leaf)
                        publication->leaves[acceptedBody.meta.x + leaf] = {static_cast<uint32_t>(index), leaf};
                }
                auto &slot = slots[index];
                const auto &input = (*prefix)[index];
                if (input.bodyId == InvalidBodyId || input.bodyId != slot.bodyId || !slot.accepted)
                    continue;
                // Every preceding model/command packet is now accepted. Only
                // this captured prefix contributes target provenance; author
                // calls made while it was pending belong to the next prefix.
                const auto motion = storage.GetModel().bodies[index].meta.z;
                const bool drives = (motion & ~MotionNotResident) == MotionKinematic ||
                                    ((motion & ~MotionNotResident) == MotionStatic && input.target.temporaryStatic);
                if (!controlOnly && !(motion & MotionNotResident) && drives && input.target.pending) {
                    AddCompletion(*completions, input.target.position, PositionChannel());
                    AddCompletion(*completions, input.target.rotation, RotationChannel());
                }
            }
            return {};
        };
        const std::shared_ptr<const Publication> retainedPublication = publication;
        if (consumer)
            capture.prepareWork = [retainedPublication, consumer = std::move(consumer)] {
                return consumer(retainedPublication);
            };
        JoltGpuTickControl control;
        control.prepareFinalWork = [this, publication, retainedPublication, consumer = std::move(finalConsumer)](
                                       const std::shared_ptr<rhi::ComputeBuffer> &states) {
            publication->bodyStates = states;
            if (storage.GetJointCount())
                publication->jointStream = storage.Get(Resource::Joints)->Freeze();
            // Numerical failure rejects the whole unpublished step. The last
            // complete world stays available; a local guard is not a fallback.
            return Then(joints.PrepareStatus(retainedPublication), [retainedPublication, consumer] {
                return consumer ? consumer(retainedPublication) : JoltGpuControlWork{};
            });
        };
        // Reserve before moving any pending work so allocation failure leaves
        // the authored queue intact and no initialization packet is orphaned.
        pending.reserve(pending.size() + 1);
        pending.push_back(std::move(capture));
        control.initialization = std::move(initialization);
        control.commandProgram = &commands;
        control.groups = std::move(pending);
        pendingSources.reset();
        pendingCompletions.clear();
        candidateCompletions = std::move(completions);
        candidate = std::move(publication);
        candidateControlOnly = controlOnly;
        if (!controlOnly) {
            for (auto &slot : slots) {
                const bool drives =
                    slot.authored &&
                    (slot.authored->acceptedMotionType == MotionKinematic ||
                     (slot.authored->acceptedMotionType == MotionStatic && slot.queuedTarget.temporaryStatic));
                if (!slot.desiredResident || !drives || !slot.queuedTarget.pending)
                    continue;
                // Hand ownership to this logical tick, without releasing the
                // Rigidbody guard. Later type changes must not carry/snap an
                // endpoint already consumed by this tick's requested duration.
                slot.queuedTarget.pending = false;
                slot.queuedTarget.position = {};
                slot.queuedTarget.rotation = {};
            }
        }
        return control;
    }

    void BeginTick(uint64_t epoch, uint64_t step, float dt, const JoltGpuStepParameters &parameters,
                   Consumer finalConsumer)
    {
        auto control = PrepareControl(epoch, step, false, {}, std::move(finalConsumer));
        try {
            tick.Begin(dt, parameters, std::move(control));
        } catch (...) {
            // Begin may have retained a partial plan. This storage cannot be
            // resumed by rebuilding author input or replaying commands.
            Cancel();
            throw;
        }
    }

    void BeginControl(uint64_t epoch, Consumer consumer, Consumer finalConsumer)
    {
        const auto step = published && published->simulationEpoch == epoch ? published->fixedStepIndex : 0;
        auto control = PrepareControl(epoch, step, true, std::move(consumer), std::move(finalConsumer));
        try {
            tick.BeginControl(std::move(control));
        } catch (...) {
            Cancel();
            throw;
        }
    }

    void BeginWork(JoltGpuControlWork work)
    {
        RequireUsable();
        if (tick.GetStatus() != JoltGpuTick::Status::Idle && tick.GetStatus() != JoltGpuTick::Status::Complete)
            throw std::logic_error("Jolt GPU scene work requires an idle or fully completed source");
        if (candidate)
            throw std::logic_error("Publish the completed Jolt GPU scene generation before borrowing its source");
        JoltGpuTickControl control;
        JoltGpuControlGroup group;
        group.prepareWork = [work = std::move(work)]() mutable { return std::move(work); };
        control.groups.push_back(std::move(group));
        try {
            tick.BeginControl(std::move(control));
        } catch (...) {
            Cancel();
            throw;
        }
    }

    bool PublishCompleted(bool controlOnly)
    {
        if (!candidate || candidateControlOnly != controlOnly || cancelled || !tick.IsComplete())
            return false;
        candidate->bodyStates = tick.GetCompletedBodyStates();
        published = std::move(candidate);
        if (candidateCompletions) {
            for (const auto &completion : *candidateCompletions) {
                const auto *author = completion.source.Get();
                auto *body = author ? dynamic_cast<Rigidbody *>(author->rigidbody.GetComponent()) : nullptr;
                if (body)
                    body->CompletePhysicsTarget(completion.source, completion.channels);
            }
        }
        candidateCompletions.reset();
        return true;
    }

    void Cancel() noexcept
    {
        cancelled = true;
        tick.Cancel();
        if (initialization)
            (void)initialization->Cancel();
        initialization.reset();
        pending.clear();
        pendingSources.reset();
        pendingCompletions.clear();
        candidateCompletions.reset();
        for (auto &slot : slots) {
            slot.queuedTarget = {};
        }
        candidate.reset();
        byId.clear();
        for (auto &[id, joint] : authoredJoints)
            joint.owner->OnPhysicsConstraintDestroyed();
        authoredJoints.clear();
        acceptedJoints.clear();
    }

    PhysicsECSStore &actors;
    JoltGpuStorage storage;
    JoltGpuPrograms programs;
    JoltGpuBroadphase broadphase;
    JoltGpuConstraints constraints;
    JoltGpuBodyMutations mutations;
    JoltGpuBodyCommands commands;
    JoltGpuJoints &joints;
    std::map<uint64_t, JointRecord> authoredJoints;
    std::vector<PublishedJoint> acceptedJoints;
    uint64_t nextJointId = 1;
    std::vector<Slot> slots;
    std::vector<uint32_t> freeSlots;
    std::unordered_map<uint32_t, uint32_t> byId;
    uint32_t nextBodyId = 0;
    bool cancelled = false;
    bool candidateControlOnly = false;
    std::vector<JoltGpuControlGroup> pending;
    std::shared_ptr<PhysicsTargetBatch> pendingSources;
    std::vector<TargetCompletion> pendingCompletions;
    std::shared_ptr<std::vector<TargetCompletion>> candidateCompletions;
    std::shared_ptr<rhi::PreparedComputeBatch> initialization;
    std::shared_ptr<Publication> candidate;
    std::shared_ptr<const Publication> published;
    // Destroy first: accepted receipts drain while storage/programs and all
    // callback owners still exist. Ordinary Stop uses Cancel/CanRetire first.
    JoltGpuTick tick;
};

JoltGpuScene::JoltGpuScene(rhi::ComputeHost &host, FrameWorkOwner &work, uint32_t capacity,
                           const ShaderTargetCode &stepTarget, const ShaderTargetCode &mutationTarget,
                           const ShaderTargetCode &commandTarget, const ShaderTargetCode &broadphaseTarget,
                           const ShaderTargetCode &constraintTarget, JoltGpuJoints &joints)
    : m_impl(std::make_unique<Impl>(host, work, capacity, stepTarget, mutationTarget, commandTarget, broadphaseTarget,
                                    constraintTarget, joints))
{
}

JoltGpuScene::~JoltGpuScene() = default;

JoltGpuScene::BodyAllocation JoltGpuScene::Reserve(PhysicsActorHandle actor, OwnerIdentity owner)
{
    return m_impl->Reserve(actor, std::move(owner));
}

void JoltGpuScene::StageActor(uint32_t id, PreparedGpuActor actor)
{
    m_impl->StageActor(id, std::move(actor));
}

void JoltGpuScene::ReserveCapacity(uint32_t capacity)
{
    m_impl->RequireUsable();
    m_impl->ReserveCapacity(capacity);
}

void JoltGpuScene::SetResident(uint32_t id, bool resident, bool preserveVelocity)
{
    m_impl->SetResident(id, resident, preserveVelocity);
}

PhysicsTargetSource JoltGpuScene::QueueCommand(uint32_t id, BodyCommand command, const PhysicsTargetAuthor *author)
{
    return m_impl->QueueCommand(id, command, author);
}

void JoltGpuScene::StageRetire(uint32_t id)
{
    m_impl->StageRetire(id);
}

uint64_t JoltGpuScene::CreateJoint(PhysicsConstraintOwner &owner, uint32_t ownerBody, uint32_t connectedBody,
                                   JointAuthor author)
{
    return m_impl->CreateJoint(owner, ownerBody, connectedBody, author);
}

void JoltGpuScene::UpdateJoint(uint64_t id, const std::function<void(JointAuthor &)> &update)
{
    m_impl->UpdateJoint(id, update);
}

void JoltGpuScene::UpdateJoints(std::vector<JointUpdate> updates)
{
    m_impl->UpdateJoints(std::move(updates));
}

void JoltGpuScene::DestroyJoint(uint64_t id)
{
    m_impl->DestroyJoint(id);
}

void JoltGpuScene::DestroyJointsForBody(uint32_t bodyId)
{
    m_impl->DestroyJointsForBody(bodyId);
}

const JoltGpuScene::PublishedJoint &JoltGpuScene::GetAuthoredJoint(uint64_t id) const
{
    return m_impl->GetAuthoredJoint(id);
}

JoltGpuScene::BodyAllocation JoltGpuScene::FindBody(uint32_t id) const noexcept
{
    const auto found = m_impl->byId.find(id);
    return found == m_impl->byId.end() ? BodyAllocation{} : BodyAllocation{id, found->second};
}

PhysicsActorHandle JoltGpuScene::FindActor(uint32_t id) const noexcept
{
    const auto body = FindBody(id);
    return body.bodyId == InvalidBodyId ? PhysicsActorHandle{} : m_impl->slots[body.slot].actor;
}

size_t JoltGpuScene::GetBodyCount() const noexcept
{
    return m_impl->byId.size();
}

uint32_t JoltGpuScene::GetCapacity() const noexcept
{
    return static_cast<uint32_t>(m_impl->slots.size());
}
bool JoltGpuScene::CanReserve() const noexcept
{
    return !m_impl->cancelled && !m_impl->freeSlots.empty();
}

bool JoltGpuScene::HasPendingControl() const noexcept
{
    return m_impl->initialization || !m_impl->pending.empty() ||
           (m_impl->tick.IsPending() && !m_impl->tick.AreControlInputsAccepted());
}

void JoltGpuScene::BeginTick(uint64_t epoch, uint64_t step, float dt, const JoltGpuStepParameters &parameters)
{
    BeginTick(epoch, step, dt, parameters, {});
}

void JoltGpuScene::BeginTick(uint64_t epoch, uint64_t step, float dt, const JoltGpuStepParameters &parameters,
                             Consumer finalConsumer)
{
    m_impl->BeginTick(epoch, step, dt, parameters, std::move(finalConsumer));
}

bool JoltGpuScene::PublishCompletedTick()
{
    return m_impl->PublishCompleted(false);
}

void JoltGpuScene::BeginControl(uint64_t epoch)
{
    BeginControl(epoch, {});
}

void JoltGpuScene::BeginControl(uint64_t epoch, Consumer consumer)
{
    BeginControl(epoch, std::move(consumer), {});
}

void JoltGpuScene::BeginControl(uint64_t epoch, Consumer consumer, Consumer finalConsumer)
{
    m_impl->BeginControl(epoch, std::move(consumer), std::move(finalConsumer));
}

void JoltGpuScene::BeginWork(JoltGpuControlWork work)
{
    m_impl->BeginWork(std::move(work));
}

bool JoltGpuScene::PublishCompletedControl()
{
    return m_impl->PublishCompleted(true);
}

const std::shared_ptr<const JoltGpuScene::Publication> &JoltGpuScene::GetPublished() const noexcept
{
    return m_impl->published;
}

JoltGpuTick::Status JoltGpuScene::GetTickStatus() const noexcept
{
    return m_impl->tick.GetStatus();
}

FrameWorkOwner::SourceId JoltGpuScene::GetSourceId() const noexcept
{
    return m_impl->tick.GetSourceId();
}

bool JoltGpuScene::AreControlInputsAccepted() const noexcept
{
    return m_impl->tick.AreControlInputsAccepted();
}
FrameWorkWaitResult JoltGpuScene::WaitForProgress(uint64_t maxWaitNanoseconds)
{
    return m_impl->tick.WaitForProgress(maxWaitNanoseconds);
}

void JoltGpuScene::SetAdmissionEnabled(bool enabled) noexcept
{
    m_impl->tick.SetAdmissionEnabled(enabled);
}

bool JoltGpuScene::IsAdmissionEnabled() const noexcept
{
    return m_impl->tick.IsAdmissionEnabled();
}

void JoltGpuScene::Cancel() noexcept
{
    m_impl->Cancel();
}

bool JoltGpuScene::CanRetire() const noexcept
{
    return m_impl->tick.CanRetire();
}

} // namespace infernux
