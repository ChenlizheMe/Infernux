#include "JoltGpuScene.h"

#include "JoltGpuBodyMutations.h"
#include <function/scene/Collider.h>
#include <function/scene/GameObject.h>
#include <function/scene/Rigidbody.h>

#include <cmath>
#include <limits>
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
    if (capacity > MaxBodiesPerWorld)
        throw std::invalid_argument("Jolt GPU scene capacity exceeds the fork's single-world bound");
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
         const ShaderTargetCode &mutationTarget, const ShaderTargetCode &commandTarget)
        : actors(PhysicsECSStore::Instance()), storage(host, EmptyModel(capacity)), programs(host, stepTarget),
          mutations(host, mutationTarget), commands(host, commandTarget), slots(capacity), tick(work, storage, programs)
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
        if (leafCount > std::numeric_limits<uint32_t>::max() ||
            leafCount * leafCount > std::numeric_limits<uint32_t>::max() / MaxContactPointsPerPair)
            throw std::length_error("Jolt GPU scene exceeds its leaf-pair contact index range");
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
        replacement.worlds.push_back(
            {{0, static_cast<uint32_t>(replacement.bodies.size()), 0, count * count}, {0, count, 0, 0}});
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
        identity->owner = slot.owner;
        identity->rootCenterOfMass = rootCOM;
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
        if (!pending.empty() && !pending.back().prepareModel && !pending.back().modelAccepted) {
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
            // construction reserves the complete fixed capacity up front.
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
        auto publication = std::make_shared<Publication>();
        publication->simulationEpoch = epoch;
        publication->fixedStepIndex = step;
        publication->actors.resize(slots.size());
        auto prefix = std::make_shared<std::vector<PrefixTarget>>();
        prefix->reserve(slots.size());
        for (const auto &slot : slots)
            prefix->push_back({slot.bodyId, slot.queuedTarget});
        auto completions = std::make_shared<std::vector<TargetCompletion>>(pendingCompletions);
        completions->reserve(completions->size() + slots.size() * 2);
        JoltGpuControlGroup capture;
        capture.prepareModel = [this, publication, prefix, completions,
                                controlOnly]() -> std::unique_ptr<JoltGpuModelUpdate> {
            if (storage.GetBodyCount()) {
                publication->bodyModels = storage.Get(Resource::BodyModels)->Freeze();
                if (const auto &shapes = storage.Get(Resource::ShapeModels))
                    publication->shapeModels = shapes->Freeze();
                publication->worlds = storage.Get(Resource::Worlds)->Freeze();
                if (controlOnly)
                    publication->bodyStates = storage.Get(Resource::BodyStates)->Freeze();
            }
            publication->leaves.resize(storage.GetShapeCount());
            for (size_t index = 0; index < slots.size(); ++index) {
                publication->actors[index] = slots[index].accepted;
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
        if (consumer)
            capture.prepareWork = [publication, consumer = std::move(consumer)] { return consumer(*publication); };
        JoltGpuTickControl control;
        if (finalConsumer)
            control.prepareFinalWork =
                [publication, consumer = std::move(finalConsumer)](const std::shared_ptr<rhi::ComputeBuffer> &states) {
                    publication->bodyStates = states;
                    return consumer(*publication);
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
    }

    PhysicsECSStore &actors;
    JoltGpuStorage storage;
    JoltGpuPrograms programs;
    JoltGpuBodyMutations mutations;
    JoltGpuBodyCommands commands;
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
                           const ShaderTargetCode &commandTarget)
    : m_impl(std::make_unique<Impl>(host, work, capacity, stepTarget, mutationTarget, commandTarget))
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
