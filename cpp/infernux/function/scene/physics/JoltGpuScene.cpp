#include "JoltGpuScene.h"

#include "JoltGpuBodyMutations.h"
#include <function/scene/Collider.h>
#include <function/scene/GameObject.h>
#include <function/scene/Rigidbody.h>

#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace infernux
{
namespace
{
using namespace JPH::GPU;

BodyModel Tombstone(uint32_t slot)
{
    BodyModel body{};
    body.meta = {slot, ShapeEmpty, MotionStatic | MotionNotResident, 0};
    body.inertiaRotation.w = 1.0f;
    return body;
}

ShapeModel EmptyShape()
{
    ShapeModel shape{};
    shape.localRotation.w = 1.0f;
    return shape;
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
    model.shapes.assign(capacity, EmptyShape());
    for (uint32_t slot = 0; slot < capacity; ++slot)
        model.bodies.push_back(Tombstone(slot));
    model.worlds.push_back({{0, capacity, 0, capacity * capacity}});
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
    struct Slot
    {
        uint32_t bodyId = InvalidBodyId;
        PhysicsActorHandle actor;
        OwnerIdentity owner;
        bool desiredResident = false;
        std::shared_ptr<const PublishedActor> accepted;
        // Shares the actual pending transaction record. After acceptance these
        // pointers normally name the same object; no second model is cached.
        std::shared_ptr<const PublishedActor> authored;
    };

    struct ActorChange
    {
        uint32_t slot;
        uint32_t bodyId;
        BodyModel model;
        ShapeModel shape;
        BodyState initialState;
        std::shared_ptr<PublishedActor> identity;
        bool creation = false;
    };

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
        if (change->creation && actor.rigidbody) {
            const auto handle = actor.rigidbody->GetECSHandle();
            if (actors.IsValid(handle)) {
                auto &body = actors.GetRigidbody(handle);
                // A setter can author another deferred value while admission
                // waits. Consume only the exact initial value captured by this
                // creation, leaving a newer authored velocity pending.
                const auto &linear = change->initialState.linearVelocity;
                const auto &angular = change->initialState.angularVelocity;
                if (body.hasLinearVelocity && body.linearVelocity.x == linear.x && body.linearVelocity.y == linear.y &&
                    body.linearVelocity.z == linear.z)
                    body.hasLinearVelocity = false;
                if (body.hasAngularVelocity && body.angularVelocity.x == angular.x &&
                    body.angularVelocity.y == angular.y && body.angularVelocity.z == angular.z)
                    body.hasAngularVelocity = false;
            }
        }
    }

    void StageActor(uint32_t id, PreparedGpuActor actor)
    {
        const uint32_t index = RequireSlot(id);
        auto &slot = slots[index];
        const bool firstOwnership = !slot.authored;
        const auto center = actor.geometry.centerOfMass;
        const Float4 rootCOM{center.GetX(), center.GetY(), center.GetZ(), 0.0f};
        if (!FiniteVector(rootCOM) || actor.body.meta.y != actor.geometry.type)
            throw std::invalid_argument("Jolt GPU actor has inconsistent geometry or an invalid root COM");
        auto identity = std::make_shared<PublishedActor>();
        identity->bodyId = id;
        identity->slot = index;
        identity->actor = slot.actor;
        identity->owner = slot.owner;
        identity->rootCenterOfMass = rootCOM;
        identity->subShapeId = actor.geometry.subShapeID.GetValue();
        identity->authoredShape = std::move(actor.authoredShape);
        if (auto *surface = actor.surfaceCollider) {
            identity->surfaceTarget = PhysicsTargetReference(surface);
            identity->colliderId = surface->GetComponentID();
            if (auto *object = surface->GetGameObject())
                identity->surfaceGameObjectId = object->GetID();
        }
        // No component pointer escapes this capture boundary.
        actor.surfaceCollider = nullptr;
        auto change = std::make_shared<ActorChange>();
        change->slot = index;
        change->bodyId = id;
        change->model = actor.body;
        change->model.meta.x = index;
        change->model.meta.w = 0;
        change->model.meta.z =
            (change->model.meta.z & ~MotionNotResident) | (slot.desiredResident ? 0 : MotionNotResident);
        change->shape = actor.geometry.model;
        change->initialState = actor.initialState;
        change->identity = std::move(identity);

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
            return mutations.Prepare(storage, {{change->slot, change->model}}, {{change->slot, change->shape}}, initial,
                                     rebases);
        };
        group.modelAccepted = [this, change]() noexcept { CommitActor(change); };
        if ((change->model.meta.z & ~MotionNotResident) == MotionStatic) {
            // Static authoring discards motion and accumulated force so a
            // later dynamic transition cannot revive the previous motion.
            BodyCommand reset{};
            reset.meta = {index, CommandResetMotion, 0, 0};
            group.commands.push_back(reset);
        } else if (firstOwnership && (change->model.meta.z & ~MotionNotResident) == MotionDynamic) {
            // Initialization retains the captured raw input. Apply the normal
            // GPU velocity setters once so control-only reads also observe the
            // authored speed limits before the first solver step.
            BodyCommand linear{};
            linear.meta = {index, CommandSetLinearVelocity, 0, 0};
            linear.value = change->initialState.linearVelocity;
            BodyCommand angular{};
            angular.meta = {index, CommandSetAngularVelocity, 0, 0};
            angular.value = change->initialState.angularVelocity;
            group.commands = {linear, angular};
        }
        pending.push_back(std::move(group));
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
        if (!resident && !preserveVelocity) {
            BodyCommand linear{};
            linear.meta = {index, CommandSetLinearVelocity, 0, 0};
            BodyCommand angular{};
            angular.meta = {index, CommandSetAngularVelocity, 0, 0};
            group.commands = {linear, angular};
        }
        pending.push_back(std::move(group));
        slot.desiredResident = resident;
    }

    void QueueCommand(uint32_t id, BodyCommand command)
    {
        const uint32_t index = RequireSlot(id);
        const auto &slot = slots[index];
        if (!slot.authored)
            throw std::logic_error("Jolt GPU actor must be staged before authoring a body command");
        command.meta.x = index;
        if (command.meta.y <= CommandSetPose)
            command.rootCOM = slot.authored->rootCenterOfMass;
        if (!pending.empty() && !pending.back().prepareModel && !pending.back().modelAccepted) {
            pending.back().commands.push_back(command);
        } else {
            JoltGpuControlGroup group;
            group.commands.push_back(command);
            pending.push_back(std::move(group));
        }
    }

    void StageRetire(uint32_t id)
    {
        const uint32_t index = RequireSlot(id);
        JoltGpuControlGroup group;
        group.prepareModel = [this, index, id] {
            RequireOwnership(index, id);
            const BodyInitialization initial{{index, 0, 0, 0}, EmptyState(), {}};
            return mutations.Prepare(storage, {{index, Tombstone(index)}}, {{index, EmptyShape()}}, {initial});
        };
        group.modelAccepted = [this, index]() noexcept {
            slots[index] = Slot{};
            // Every reservation pops once and every retirement pushes once;
            // construction reserves the complete fixed capacity up front.
            freeSlots.push_back(index);
        };
        pending.push_back(std::move(group));
        byId.erase(id);
    }

    JoltGpuTickControl
    PrepareControl(uint64_t epoch, uint64_t step, bool controlOnly,
                   std::function<std::shared_ptr<rhi::PreparedComputeBatch>(const Publication &)> consumer = {})
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
        JoltGpuControlGroup capture;
        capture.prepareModel = [this, publication, controlOnly]() -> std::unique_ptr<JoltGpuModelUpdate> {
            if (storage.GetBodyCount()) {
                publication->bodyModels = storage.Get(Resource::BodyModels)->Freeze();
                publication->shapeModels = storage.Get(Resource::ShapeModels)->Freeze();
                publication->worlds = storage.Get(Resource::Worlds)->Freeze();
                if (controlOnly)
                    publication->bodyStates = storage.Get(Resource::BodyStates)->Freeze();
            }
            for (size_t index = 0; index < slots.size(); ++index)
                publication->actors[index] = slots[index].accepted;
            return {};
        };
        if (consumer)
            capture.prepareWork = [publication, consumer = std::move(consumer)] { return consumer(*publication); };
        // Reserve before moving any pending work so allocation failure leaves
        // the authored queue intact and no initialization packet is orphaned.
        pending.reserve(pending.size() + 1);
        pending.push_back(std::move(capture));
        JoltGpuTickControl control;
        control.initialization = std::move(initialization);
        control.commandProgram = &commands;
        control.groups = std::move(pending);
        candidate = std::move(publication);
        candidateControlOnly = controlOnly;
        return control;
    }

    void BeginTick(uint64_t epoch, uint64_t step, float dt, const JoltGpuStepParameters &parameters)
    {
        auto control = PrepareControl(epoch, step, false);
        try {
            tick.Begin(dt, parameters, std::move(control));
        } catch (...) {
            // Begin may have retained a partial plan. This storage cannot be
            // resumed by rebuilding author input or replaying commands.
            Cancel();
            throw;
        }
    }

    void BeginControl(uint64_t epoch,
                      std::function<std::shared_ptr<rhi::PreparedComputeBatch>(const Publication &)> consumer)
    {
        const auto step = published && published->simulationEpoch == epoch ? published->fixedStepIndex : 0;
        auto control = PrepareControl(epoch, step, true, std::move(consumer));
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

void JoltGpuScene::QueueCommand(uint32_t id, BodyCommand command)
{
    m_impl->QueueCommand(id, command);
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

bool JoltGpuScene::HasPendingControl() const noexcept
{
    return m_impl->initialization || !m_impl->pending.empty() ||
           (m_impl->tick.IsPending() && !m_impl->tick.AreControlInputsAccepted());
}

void JoltGpuScene::BeginTick(uint64_t epoch, uint64_t step, float dt, const JoltGpuStepParameters &parameters)
{
    m_impl->BeginTick(epoch, step, dt, parameters);
}

bool JoltGpuScene::PublishCompletedTick()
{
    return m_impl->PublishCompleted(false);
}

void JoltGpuScene::BeginControl(uint64_t epoch)
{
    BeginControl(epoch, {});
}

void JoltGpuScene::BeginControl(uint64_t epoch,
                                std::function<std::shared_ptr<rhi::PreparedComputeBatch>(const Publication &)> consumer)
{
    m_impl->BeginControl(epoch, std::move(consumer));
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
