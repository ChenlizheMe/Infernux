#include "JoltGpuScene.h"

#include "JoltGpuBodyMutations.h"
#include "JoltGpuBroadphase.h"
#include "JoltGpuConstraints.h"
#include "JoltGpuJoints.h"
#include "JoltGpuPackedLayout.h"
#include <cstring>
#include <function/scene/Collider.h>
#include <function/scene/GameObject.h>
#include <function/scene/Rigidbody.h>
#include <function/scene/RuntimeWorld.h>

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

    struct Partition
    {
        RuntimeWorld *world = nullptr;
        uint64_t id = 0;
        uint32_t row = 0, capacity = 0, liveCount = 0;
        std::vector<uint32_t> records;
        std::vector<uint32_t> freeLocals;
        std::vector<JoltGpuControlGroup> pending;
        std::shared_ptr<PhysicsTargetBatch> pendingSources;
        std::vector<TargetCompletion> pendingCompletions;
        std::shared_ptr<const Publication> published;
        JoltGpuTick::Status status = JoltGpuTick::Status::Idle;
        uint64_t tickSerial = 0, controlSerial = 0, observedTick = 0, observedControl = 0;
        bool admission = true, closed = false, cleanup = false, retiring = false, retired = false;
    };

    struct Slot
    {
        Partition *partition = nullptr;
        uint32_t local = 0;
        bool retiring = false;
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

    PhysicsTargetSource AppendSource(Partition &domain, const PhysicsTargetAuthor &author)
    {
        auto &pendingSources = domain.pendingSources;
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

    Impl(rhi::ComputeHost &host, FrameWorkOwner &budget, const ShaderTargetCode &stepTarget,
         const ShaderTargetCode &mutationTarget, const ShaderTargetCode &commandTarget,
         const ShaderTargetCode &broadphaseTarget, const ShaderTargetCode &constraintTarget,
         JoltGpuJoints &jointProgram)
        : actors(PhysicsECSStore::Instance()), work(budget), storage(host, JoltGpuModel{}), programs(host, stepTarget),
          broadphase(host, broadphaseTarget), constraints(host, constraintTarget), mutations(host, mutationTarget),
          commands(host, commandTarget), joints(jointProgram), tick(budget, storage, programs, broadphase, constraints)
    {
    }

    Partition &Domain(const RuntimeWorld &world) const
    {
        const auto found = domains.find(world.id);
        if (found == domains.end() || found->second->world != &world)
            throw std::invalid_argument("GPU partition is not registered to this RuntimeWorld");
        return *found->second;
    }
    Partition &BodyDomain(uint32_t id) const
    {
        return *slots[RequireSlot(id)].partition;
    }
    uint32_t Packed(uint32_t record) const
    {
        const auto &slot = slots[record];
        const auto row = slot.partition->row;
        if (row >= storage.GetModel().worlds.size() || slot.local >= storage.GetModel().worlds[row].bodies.y)
            throw std::logic_error("GPU body address precedes its accepted layout boundary");
        return storage.GetModel().worlds[row].bodies.x + slot.local;
    }
    void RegisterPartition(RuntimeWorld &world, uint32_t capacity)
    {
        if (domains.count(world.id))
            throw std::logic_error("GPU RuntimeWorld partition is already registered");
        if (freeRows.empty()) {
            if (ordered.size() >= UINT32_MAX)
                throw std::length_error("GPU world row address space is exhausted");
            // Retirement runs inside CanRetireDomain's nonthrowing boundary.
            // Reserve its freelist entry while registering, never while releasing.
            if (freeRows.capacity() < ordered.size() + 1u)
                freeRows.reserve(std::max(ordered.size() + 1u, freeRows.capacity() * 2u));
            if (ordered.capacity() < ordered.size() + 1u)
                ordered.reserve(std::max(ordered.size() + 1u, ordered.capacity() * 2u));
        }
        auto domain = std::make_unique<Partition>();
        domain->world = &world;
        domain->id = world.id;
        domain->row = freeRows.empty() ? static_cast<uint32_t>(ordered.size()) : freeRows.back();
        auto *address = domain.get();
        domains.emplace(world.id, std::move(domain));
        if (freeRows.empty())
            ordered.push_back(address);
        else {
            ordered[address->row] = address;
            freeRows.pop_back();
        }
        ReserveCapacity(world, capacity);
        // Even a zero-capacity registration needs its real empty GPU world row.
        if (!capacity)
            QueueLayout(*address, 0u);
    }

    uint32_t RequireSlot(uint32_t id) const
    {
        const auto found = byId.find(id);
        if (found == byId.end())
            throw std::invalid_argument("Jolt GPU body identity is invalid or retired");
        if (slots[found->second].retiring || slots[found->second].partition->closed)
            throw std::invalid_argument("GPU body owner partition is closed or retiring");
        return found->second;
    }

    void RequireOwnership(uint32_t slot, uint32_t id) const
    {
        if (slots[slot].bodyId != id)
            throw std::logic_error("Jolt GPU control references a reused body slot");
    }

    const PublishedJoint &GetAuthoredJoint(uint64_t id) const
    {
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

    void QueueJointTopology(Partition &domain)
    {
        auto own = std::make_shared<std::vector<PublishedJoint>>();
        for (const auto &[id, record] : authoredJoints)
            if (slots[byId.at(record.value.bodyIdA)].partition == &domain)
                own->push_back(record.value);
        auto catalog = std::make_shared<std::vector<PublishedJoint>>();
        JoltGpuControlGroup group;
        group.prepareModel = [this, own, catalog, &domain] {
            for (const auto &joint : acceptedJoints)
                if (joint.author.identity.z != domain.row)
                    catalog->push_back(joint);
            catalog->insert(catalog->end(), own->begin(), own->end());
            std::sort(catalog->begin(), catalog->end(), [this](const auto &a, const auto &b) {
                const auto ar = slots[byId.at(a.bodyIdA)].partition->row;
                const auto br = slots[byId.at(b.bodyIdA)].partition->row;
                return ar != br ? ar < br : a.id < b.id;
            });
            std::unordered_map<uint64_t, uint32_t> previous;
            for (const auto &joint : acceptedJoints)
                previous.emplace(joint.id, joint.row);
            std::vector<JointAuthor> authors;
            std::vector<uint32_t> remap;
            for (auto &joint : *catalog) {
                const auto owner = byId.at(joint.bodyIdA);
                joint.row = static_cast<uint32_t>(authors.size());
                joint.author.bodies.y = Packed(owner);
                joint.author.bodies.x =
                    joint.bodyIdB == InvalidBodyId ? InvalidQueryBodySlot : Packed(byId.at(joint.bodyIdB));
                joint.author.identity.z = slots[owner].partition->row;
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
        domain.pending.push_back(std::move(group));
    }

    uint64_t CreateJoint(PhysicsConstraintOwner &owner, uint32_t ownerBody, uint32_t connectedBody, JointAuthor author)
    {
        const auto a = RequireSlot(ownerBody);
        const auto b = connectedBody == InvalidBodyId ? InvalidQueryBodySlot : RequireSlot(connectedBody);
        if (b != InvalidQueryBodySlot && slots[a].partition != slots[b].partition)
            throw std::invalid_argument("GPU Joint endpoints belong to different RuntimeWorld owners");
        if (ownerBody == connectedBody || !slots[a].authored || (b != InvalidQueryBodySlot && !slots[b].authored))
            throw std::invalid_argument("A GPU joint requires distinct authored body endpoints");
        if (!nextJointId)
            throw std::overflow_error("GPU joint identity space is exhausted");
        const auto id = nextJointId++;
        author.bodies.x = b;
        author.bodies.y = a;
        author.identity = {static_cast<uint32_t>(id), static_cast<uint32_t>(id >> 32), slots[a].partition->row, 0};
        author.rootCOM1 = b == InvalidQueryBodySlot ? Float4{} : slots[b].authored->rootCenterOfMass;
        author.rootCOM2 = slots[a].authored->rootCenterOfMass;
        author.scale1 = b == InvalidQueryBodySlot ? Float4{1, 1, 1, 0} : slots[b].authored->rootScale;
        author.scale2 = slots[a].authored->rootScale;
        PublishedJoint value{id, 0, ownerBody, connectedBody, author.bodies.z, author};
        authoredJoints.emplace(id, JointRecord{value, &owner});
        QueueJointTopology(*slots[a].partition);
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
        auto &domain = BodyDomain(GetAuthoredJoint(updates.front().id).bodyIdA);
        auto &pending = domain.pending;
        auto values = std::make_shared<std::vector<PublishedJoint>>();
        values->reserve(updates.size());
        for (const auto &update : updates) {
            auto value = GetAuthoredJoint(update.id);
            if (&BodyDomain(value.bodyIdA) != &domain)
                throw std::invalid_argument("GPU joint batch crosses RuntimeWorld domains");
            value.author = update.author;
            value.author.identity = GetAuthoredJoint(update.id).author.identity;
            values->push_back(value);
        }
        JoltGpuControlGroup group;
        group.prepareModel = [this, values] {
            std::vector<JoltGpuJointModelUpdate> changes;
            changes.reserve(values->size());
            for (auto &value : *values) {
                value.row = AcceptedJoint(value.id).row;
                value.author.bodies.x =
                    value.bodyIdB == InvalidBodyId ? InvalidQueryBodySlot : Packed(byId.at(value.bodyIdB));
                value.author.bodies.y = Packed(byId.at(value.bodyIdA));
                value.author.identity.z = slots[byId.at(value.bodyIdA)].partition->row;
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
        auto &domain = BodyDomain(found->second.value.bodyIdA);
        authoredJoints.erase(found);
        QueueJointTopology(domain);
        owner->OnPhysicsConstraintDestroyed();
    }

    void DestroyJointsForBody(uint32_t bodyId)
    {
        auto &domain = BodyDomain(bodyId);
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
            QueueJointTopology(domain);
        for (auto *owner : owners)
            owner->OnPhysicsConstraintDestroyed();
    }

    struct LayoutPlan
    {
        JoltGpuModel model;
        std::vector<uint32_t> bodies, joints;
        std::vector<PublishedJoint> catalog;
        std::vector<std::pair<uint32_t, std::shared_ptr<const PublishedActor>>> identities;
    };

    std::shared_ptr<LayoutPlan> BuildLayout(Partition &changed, uint32_t capacity)
    {
        auto plan = std::make_shared<LayoutPlan>();
        const auto &old = storage.GetModel();
        std::vector<uint32_t> capacities;
        for (auto *domain : ordered) {
            if (!domain) {
                // Rows are addresses, not public identities. Preserve holes so
                // survivors retain their rows; released rows have no GPU range.
                capacities.push_back(0u);
                continue;
            }
            const uint32_t oldCount = domain->row < old.worlds.size() ? old.worlds[domain->row].bodies.y : 0u;
            const uint32_t count = domain == &changed ? capacity : oldCount;
            capacities.push_back(count);
            for (uint32_t local = 0; local < count; ++local) {
                const auto source = local < oldCount ? old.worlds[domain->row].bodies.x + local : UINT32_MAX;
                auto body = source == UINT32_MAX ? Tombstone() : old.bodies[source];
                const auto oldLeaf = body.meta.x;
                body.meta.x = static_cast<uint32_t>(plan->model.shapes.size());
                const auto target = static_cast<uint32_t>(plan->model.bodies.size());
                for (uint32_t leaf = 0; leaf < body.meta.y; ++leaf) {
                    auto shape = old.shapes[oldLeaf + leaf];
                    shape.meta.y = target;
                    plan->model.shapes.push_back(shape);
                }
                body.meta.w = domain->row;
                plan->model.bodies.push_back(body);
                plan->bodies.push_back(source);
                if (local < domain->records.size() && domain->records[local] != UINT32_MAX) {
                    const auto record = domain->records[local];
                    if (slots[record].accepted) {
                        auto identity = std::make_shared<PublishedActor>(*slots[record].accepted);
                        identity->slot = target;
                        plan->identities.emplace_back(record, std::move(identity));
                    }
                }
            }
        }
        std::vector<uint32_t> bases;
        uint32_t base = 0;
        for (auto count : capacities) {
            bases.push_back(base);
            base += count;
        }
        for (const auto &joint : acceptedJoints) {
            if (joint.author.identity.z == changed.row && !capacity)
                continue;
            auto value = joint;
            const auto &a = slots[byId.at(value.bodyIdA)];
            value.author.bodies.y = bases[a.partition->row] + a.local;
            if (value.bodyIdB != InvalidBodyId) {
                const auto &b = slots[byId.at(value.bodyIdB)];
                value.author.bodies.x = bases[b.partition->row] + b.local;
            }
            value.row = static_cast<uint32_t>(plan->catalog.size());
            plan->model.joints.push_back(value.author);
            plan->joints.push_back(joint.row);
            plan->catalog.push_back(value);
        }
        JoltGpuBuildWorldRanges(plan->model, capacities);
        return plan;
    }

    void QueueLayout(Partition &domain, uint32_t capacity)
    {
        auto transaction = std::make_shared<std::shared_ptr<JoltGpuPackedLayout>>();
        auto plan = std::make_shared<std::shared_ptr<LayoutPlan>>();
        JoltGpuControlGroup prepare, commit;
        prepare.prepareWork = [this, &domain, capacity, transaction, plan] {
            *plan = BuildLayout(domain, capacity);
            *transaction = std::make_shared<JoltGpuPackedLayout>(storage, mutations, work, (*plan)->model,
                                                                 (*plan)->bodies, (*plan)->joints);
            return (*transaction)->Prepare();
        };
        commit.prepareModel = [transaction] { return (*transaction)->Commit(); };
        commit.modelAccepted = [this, plan]() noexcept {
            for (const auto &[record, identity] : (*plan)->identities)
                slots[record].accepted = identity;
            acceptedJoints.swap((*plan)->catalog);
            ++layoutGeneration;
        };
        domain.pending.push_back(std::move(prepare));
        domain.pending.push_back(std::move(commit));
    }
    void ReserveCapacity(RuntimeWorld &world, uint32_t capacity)
    {
        auto &domain = Domain(world);
        if (domain.closed)
            throw std::logic_error("Cannot grow a closed GPU world partition");
        if (capacity <= domain.capacity)
            return;
        const auto previous = domain.capacity;
        domain.records.resize(capacity, UINT32_MAX);
        domain.freeLocals.reserve(capacity);
        for (uint32_t local = capacity; local > previous; --local)
            domain.freeLocals.push_back(local - 1u);
        domain.capacity = capacity;
        QueueLayout(domain, capacity);
    }

    BodyAllocation Reserve(RuntimeWorld &world, uint32_t id, PhysicsActorHandle actor, OwnerIdentity owner)
    {
        auto &domain = Domain(world);
        if (domain.closed || !actors.IsValid(actor) || id == InvalidBodyId || byId.count(id))
            throw std::invalid_argument("GPU body reservation requires a new live author identity");
        const auto &author = actors.GetActor(actor);
        if (author.runtimeWorld != &world || author.bodyId != InvalidBodyId || !author.owner ||
            owner.gameObjectId != author.owner->GetID() || owner.target.GetGameObject() != author.owner)
            throw std::invalid_argument("GPU body ID / RuntimeWorld / owner do not match the actor registry");
        if (domain.freeLocals.empty())
            throw std::length_error("GPU world partition needs a capacity reservation");
        const auto local = domain.freeLocals.back();
        uint32_t record;
        if (freeRecords.empty()) {
            if (freeRecords.capacity() < slots.size() + 1u)
                freeRecords.reserve(std::max(slots.size() + 1u, freeRecords.capacity() * 2u));
            record = static_cast<uint32_t>(slots.size());
            slots.emplace_back();
        } else {
            record = freeRecords.back();
            freeRecords.pop_back();
            slots[record] = Slot{};
        }
        slots[record].partition = &domain;
        slots[record].local = local;
        slots[record].bodyId = id;
        slots[record].actor = actor;
        slots[record].owner = std::move(owner);
        domain.freeLocals.pop_back();
        domain.records[local] = record;
        ++domain.liveCount;
        byId.emplace(id, record);
        // A new capacity slot has no accepted GPU address until its layout.
        const auto &worlds = storage.GetModel().worlds;
        const uint32_t packed = domain.row < worlds.size() && local < worlds[domain.row].bodies.y
                                    ? worlds[domain.row].bodies.x + local
                                    : UINT32_MAX;
        return {id, packed};
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
        (void)count;
        replacement.joints = accepted.joints;
        std::vector<uint32_t> capacities;
        for (const auto &world : accepted.worlds)
            capacities.push_back(world.bodies.y);
        JoltGpuBuildWorldRanges(replacement, capacities);
        return replacement;
    }

    void StageActor(uint32_t id, PreparedGpuActor actor)
    {
        auto &domain = BodyDomain(id);
        auto &pending = domain.pending;
        auto &pendingCompletions = domain.pendingCompletions;
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
        change->model.meta.w = slot.partition->row;
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
            const uint32_t packed = Packed(change->slot);
            change->identity->slot = packed;
            change->model.meta.w = slots[change->slot].partition->row;
            for (auto &leaf : change->leaves)
                leaf.meta.y = packed;
            const auto &previous = slots[change->slot].accepted;
            change->creation = !previous;
            change->identity->shapeRevision = previous ? previous->shapeRevision + 1 : 0;
            std::vector<BodyInitialization> initial;
            std::vector<CenterOfMassUpdate> rebases;
            if (change->creation) {
                initial.push_back({{packed, 0, 0, 0}, change->initialState, {}});
            } else {
                const auto oldCOM = previous->rootCenterOfMass;
                const auto newCOM = change->identity->rootCenterOfMass;
                rebases.push_back(
                    {{packed, 0, 0, 0}, {newCOM.x - oldCOM.x, newCOM.y - oldCOM.y, newCOM.z - oldCOM.z, 0.0f}});
            }
            const auto &acceptedBody = storage.GetModel().bodies[packed];
            if (acceptedBody.meta.y == change->leaves.size()) {
                auto body = change->model;
                body.meta.x = acceptedBody.meta.x;
                std::vector<JoltGpuShapeModelUpdate> leaves;
                leaves.reserve(change->leaves.size());
                for (uint32_t leaf = 0; leaf < change->leaves.size(); ++leaf)
                    leaves.push_back({acceptedBody.meta.x + leaf, change->leaves[leaf]});
                return mutations.Prepare(storage, {{packed, body}}, std::move(leaves), initial, rebases);
            }
            return mutations.PrepareReplacement(storage, RepackModel(packed, change->model, change->leaves), initial,
                                                rebases);
        };
        group.modelAccepted = [this, change]() noexcept { CommitActor(change); };
        if (motionChanged) {
            BodyCommand boundary{};
            boundary.meta = {id, CommandMotionBoundary, MotionBoundaryTypeChange | CarryFlags(slot.queuedTarget), 0};
            group.commands.push_back(boundary);
        }
        if (motion != MotionStatic) {
            // A new slot clamps its initial values. A new Rigidbody attached
            // to an existing actor applies only that component's captured
            // pending inputs, preserving the resident numerical pose/state.
            if (firstOwnership || (change->initializesRigidbody && change->hasInitialLinearVelocity)) {
                BodyCommand linear{};
                linear.meta = {id, CommandSetLinearVelocity, 0, 0};
                linear.value = change->initialState.linearVelocity;
                group.commands.push_back(linear);
            }
            if (firstOwnership || (change->initializesRigidbody && change->hasInitialAngularVelocity)) {
                BodyCommand angular{};
                angular.meta = {id, CommandSetAngularVelocity, 0, 0};
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
        auto &domain = BodyDomain(id);
        auto &pending = domain.pending;
        auto &pendingCompletions = domain.pendingCompletions;
        const uint32_t index = RequireSlot(id);
        auto &slot = slots[index];
        if (slot.desiredResident == resident && (resident || preserveVelocity))
            return;
        JoltGpuControlGroup group;
        group.prepareModel = [this, index, id, resident] {
            RequireOwnership(index, id);
            const auto packed = Packed(index);
            auto model = storage.GetModel().bodies[packed];
            model.meta.z = (model.meta.z & ~MotionNotResident) | (resident ? 0 : MotionNotResident);
            return storage.PrepareModelUpdate({{packed, model}}, {});
        };
        if (!resident) {
            pendingCompletions.reserve(pendingCompletions.size() + 2);
            BodyCommand boundary{};
            boundary.meta = {id, CommandMotionBoundary,
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
        auto &domain = BodyDomain(id);
        auto &pending = domain.pending;
        auto &pendingCompletions = domain.pendingCompletions;
        const uint32_t index = RequireSlot(id);
        auto &slot = slots[index];
        if (!slot.authored)
            throw std::logic_error("Jolt GPU actor must be staged before authoring a body command");
        command.meta.x = id;
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
        const auto source = author ? AppendSource(domain, *author) : PhysicsTargetSource{};
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
        auto &domain = BodyDomain(id);
        auto &pending = domain.pending;
        auto &pendingCompletions = domain.pendingCompletions;
        const uint32_t index = RequireSlot(id);
        DestroyJointsForBody(id);
        JoltGpuControlGroup group;
        group.prepareModel = [this, index, id] {
            RequireOwnership(index, id);
            const auto packed = Packed(index);
            const BodyInitialization initial{{packed, 0, 0, 0}, EmptyState(), {}};
            const auto &body = storage.GetModel().bodies[packed];
            if (body.meta.y == 0) {
                auto tombstone = Tombstone();
                tombstone.meta.x = body.meta.x;
                tombstone.meta.w = slots[index].partition->row;
                return mutations.Prepare(storage, {{packed, tombstone}}, {}, {initial});
            }
            return mutations.PrepareReplacement(storage, RepackModel(packed, Tombstone(), {}), {initial});
        };
        group.modelAccepted = [this, index, id]() noexcept {
            auto &slot = slots[index];
            slot.partition->records[slot.local] = UINT32_MAX;
            slot.partition->freeLocals.push_back(slot.local);
            byId.erase(id);
            // Free records must not retain a retiring Partition*, target owner,
            // initial Rigidbody generation, actor model or command provenance.
            slot = Slot{};
            freeRecords.push_back(index);
        };
        pendingCompletions.reserve(pendingCompletions.size() + 2);
        pending.push_back(std::move(group));
        CancelTarget(slots[index].queuedTarget, pendingCompletions);
        slots[index].retiring = true;
        --domain.liveCount;
    }

    enum class WriterKind
    {
        Tick,
        Control,
        Work,
        Cancelled,
        Cleanup,
        Retirement
    };

    JoltGpuTickControl PrepareControl(Partition &domain, uint64_t epoch, uint64_t step, bool controlOnly,
                                      Consumer consumer, Consumer finalConsumer)
    {
        auto publication = std::make_shared<Publication>();
        publication->worldId = domain.id;
        publication->simulationEpoch = epoch;
        publication->fixedStepIndex = step;
        auto prefix = std::make_shared<std::vector<PrefixTarget>>();
        for (const auto index : domain.records)
            if (index != UINT32_MAX)
                prefix->push_back({slots[index].bodyId, slots[index].queuedTarget});
        auto completions = std::make_shared<std::vector<TargetCompletion>>(domain.pendingCompletions);
        completions->reserve(completions->size() + 2u * prefix->size());
        JoltGpuControlGroup capture;
        capture.prepareModel = [this, &domain, publication, prefix, completions,
                                controlOnly]() -> std::unique_ptr<JoltGpuModelUpdate> {
            publication->layoutGeneration = layoutGeneration;
            publication->bodyModels = storage.Get(Resource::BodyModels)->Freeze();
            publication->shapeModels = storage.Get(Resource::ShapeModels)->Freeze();
            publication->worlds = storage.Get(Resource::Worlds)->Freeze();
            publication->actors.resize(storage.GetBodyCount());
            publication->residentBodies.resize(storage.GetBodyCount());
            publication->leaves.resize(storage.GetShapeCount());
            for (const auto &[id, index] : byId) {
                const auto &slot = slots[index];
                if (!slot.accepted)
                    continue;
                const auto packed = Packed(index);
                publication->actors[packed] = slot.accepted;
                const auto &model = storage.GetModel().bodies[packed];
                publication->residentBodies[packed] = (model.meta.z & MotionNotResident) == 0u;
                if (slot.accepted->leaves.size() != model.meta.y)
                    throw std::logic_error("GPU publication actor directory differs from accepted leaf layout");
                for (uint32_t leaf = 0; leaf < model.meta.y; ++leaf)
                    publication->leaves[model.meta.x + leaf] = {packed, leaf};
            }
            for (const auto &joint : acceptedJoints)
                if (joint.author.identity.z == domain.row) {
                    auto local = joint;
                    local.row = static_cast<uint32_t>(publication->joints.size());
                    publication->joints.push_back(local);
                }
            for (const auto &input : *prefix) {
                const auto found = byId.find(input.bodyId);
                if (found == byId.end() || !slots[found->second].accepted)
                    continue;
                const auto motion = storage.GetModel().bodies[Packed(found->second)].meta.z;
                const bool drives = (motion & ~MotionNotResident) == MotionKinematic ||
                                    ((motion & ~MotionNotResident) == MotionStatic && input.target.temporaryStatic);
                if (!controlOnly && !(motion & MotionNotResident) && drives && input.target.pending) {
                    AddCompletion(*completions, input.target.position, PositionChannel());
                    AddCompletion(*completions, input.target.rotation, RotationChannel());
                }
            }
            return {};
        };
        domain.pending.push_back(std::move(capture));
        for (auto &group : domain.pending)
            if (!group.commands.empty())
                group.materializeCommands = [this, &domain](std::vector<BodyCommand> &values) {
                    for (auto &command : values) {
                        const auto found = byId.find(command.meta.x);
                        if (found == byId.end() || slots[found->second].partition != &domain)
                            throw std::logic_error("GPU command identity is not owned by the accepted partition");
                        const auto record = found->second;
                        command.meta.x = Packed(record);
                        const auto kind = command.meta.y;
                        const bool move = kind >= CommandMovePosition && kind <= CommandMovePose;
                        if ((kind <= CommandSetPose || move) && slots[record].accepted) {
                            const auto speed = command.rootCOM.w;
                            command.rootCOM = slots[record].accepted->rootCenterOfMass;
                            if (move)
                                command.rootCOM.w = speed;
                        }
                    }
                };
        JoltGpuTickControl control;
        control.commandProgram = &commands;
        control.groups = std::move(domain.pending);
        domain.pendingSources.reset();
        domain.pendingCompletions.clear();
        const std::shared_ptr<const Publication> retained = publication;
        control.prepareFinalWork = [this, publication, retained, consumer = std::move(consumer),
                                    finalConsumer = std::move(finalConsumer)](const JoltGpuTickOutput &output) {
            publication->worldRow = output.worldRow;
            publication->bodyBase = output.bodyBase;
            publication->bodyCount = output.bodyCount;
            publication->leafBase = output.leafBase;
            publication->leafCount = output.leafCount;
            publication->jointBase = output.jointBase;
            publication->jointCount = output.jointCount;
            publication->bodyStates = output.states;
            publication->jointStream = output.joints;
            return Then(joints.PrepareStatus(retained), [retained, consumer, finalConsumer] {
                return Then(consumer ? consumer(retained) : JoltGpuControlWork{}, [retained, finalConsumer] {
                    return finalConsumer ? finalConsumer(retained) : JoltGpuControlWork{};
                });
            });
        };
        candidate = std::move(publication);
        candidateCompletions = std::move(completions);
        if (!controlOnly)
            for (const auto index : domain.records) {
                if (index == UINT32_MAX)
                    continue;
                auto &slot = slots[index];
                const bool drives =
                    slot.authored &&
                    (slot.authored->acceptedMotionType == MotionKinematic ||
                     (slot.authored->acceptedMotionType == MotionStatic && slot.queuedTarget.temporaryStatic));
                if (slot.desiredResident && drives && slot.queuedTarget.pending) {
                    slot.queuedTarget.pending = false;
                    slot.queuedTarget.position = {};
                    slot.queuedTarget.rotation = {};
                }
            }
        return control;
    }

    bool TryBegin(Partition &domain, uint64_t epoch, uint64_t step, float dt, const JoltGpuStepParameters *parameters,
                  Consumer consumer, Consumer finalConsumer)
    {
        Service();
        if (writer || domain.closed || !domain.admission)
            return false;
        auto control =
            PrepareControl(domain, epoch, step, parameters == nullptr, std::move(consumer), std::move(finalConsumer));
        writer = &domain;
        writerKind = parameters ? WriterKind::Tick : WriterKind::Control;
        domain.status = JoltGpuTick::Status::Running;
        tick.SetAdmissionEnabled(true);
        try {
            if (parameters)
                tick.Begin(domain.row, dt, *parameters, std::move(control));
            else
                tick.BeginControl(domain.row, std::move(control));
        } catch (...) {
            Cancel(domain);
            throw;
        }
        return true;
    }

    void PublishWriter()
    {
        auto &domain = *writer;
        if (candidate) {
            domain.published = std::move(candidate);
            if (writerKind == WriterKind::Control)
                ++domain.controlSerial;
            else
                ++domain.tickSerial;
            if (candidateCompletions)
                for (const auto &completion : *candidateCompletions) {
                    const auto *author = completion.source.Get();
                    auto *body = author ? dynamic_cast<Rigidbody *>(author->rigidbody.GetComponent()) : nullptr;
                    if (body)
                        body->CompletePhysicsTarget(completion.source, completion.channels);
                }
        }
        candidateCompletions.reset();
        domain.status = JoltGpuTick::Status::Complete;
        writer = nullptr;
    }

    void Cancel(Partition &domain)
    {
        if (domain.retired)
            return;
        domain.closed = true;
        domain.cleanup = true;
        domain.observedTick = domain.tickSerial;
        domain.observedControl = domain.controlSerial;
        // Close and retire author owners while the RuntimeWorld/component
        // lifetime is still held. GPU cleanup may run much later behind a
        // neighboring writer and must not call a raw retired Joint owner.
        for (auto it = authoredJoints.begin(); it != authoredJoints.end();) {
            const auto found = byId.find(it->second.value.bodyIdA);
            if (found != byId.end() && slots[found->second].partition == &domain) {
                auto *owner = it->second.owner;
                it = authoredJoints.erase(it);
                owner->OnPhysicsConstraintDestroyed();
            } else
                ++it;
        }
        std::vector<TargetCompletion> targets = domain.pendingCompletions;
        targets.reserve(targets.size() + domain.records.size() * 2u +
                        (writer == &domain && candidateCompletions ? candidateCompletions->size() : 0u));
        if (writer == &domain && candidateCompletions)
            targets.insert(targets.end(), candidateCompletions->begin(), candidateCompletions->end());
        domain.pending.clear();
        domain.pendingSources.reset();
        domain.pendingCompletions.clear();
        for (const auto index : domain.records)
            if (index != UINT32_MAX)
                CancelTarget(slots[index].queuedTarget, targets);
        for (const auto &target : targets) {
            const auto *author = target.source.Get();
            auto *body = author ? dynamic_cast<Rigidbody *>(author->rigidbody.GetComponent()) : nullptr;
            if (body)
                body->CompletePhysicsTarget(target.source, target.channels);
        }
        if (writer == &domain) {
            tick.Cancel();
            candidate.reset();
            candidateCompletions.reset();
            writerKind = WriterKind::Cancelled;
        }
        domain.status = JoltGpuTick::Status::Cancelled;
    }

    void QueueCleanup(Partition &domain)
    {
        // Only this domain's accepted joints are removed. Neighboring stable
        // IDs/runtime survive the normal joint topology remap.
        if (domain.row >= storage.GetWorldCount())
            QueueLayout(domain, 0u);
        QueueJointTopology(domain);
        JoltGpuControlGroup cleanup;
        cleanup.prepareModel = [this, &domain] {
            const auto &old = storage.GetModel();
            const auto range = old.worlds[domain.row].bodies;
            JoltGpuModel model;
            model.bodies = old.bodies;
            model.joints = old.joints;
            std::vector<uint32_t> capacities;
            for (const auto &world : old.worlds)
                capacities.push_back(world.bodies.y);
            for (uint32_t slot = 0; slot < model.bodies.size(); ++slot) {
                const bool selected = slot >= range.x && slot - range.x < range.y;
                auto body = selected ? Tombstone() : old.bodies[slot];
                body.meta.x = static_cast<uint32_t>(model.shapes.size());
                if (!selected)
                    for (uint32_t leaf = 0; leaf < body.meta.y; ++leaf)
                        model.shapes.push_back(old.shapes[old.bodies[slot].meta.x + leaf]);
                model.bodies[slot] = body;
            }
            JoltGpuBuildWorldRanges(model, capacities);
            return mutations.PrepareReplacement(storage, std::move(model));
        };
        cleanup.prepareWork = [this, &domain] { return PrepareLocalReset(domain); };
        cleanup.modelAccepted = [this, &domain]() noexcept { ReleaseRecords(domain); };
        domain.pending.push_back(std::move(cleanup));
    }

    void ReleaseRecords(Partition &domain) noexcept
    {
        for (auto &index : domain.records) {
            if (index == UINT32_MAX)
                continue;
            auto &slot = slots[index];
            byId.erase(slot.bodyId);
            domain.freeLocals.push_back(slot.local);
            slot = Slot{};
            freeRecords.push_back(index);
            index = UINT32_MAX;
        }
        domain.liveCount = 0;
    }

    bool ReleaseRetiredPartition(Partition &domain) noexcept
    {
        // Ordinary Cancel is reopenable for this same RuntimeWorld. Only its
        // final retirement may return a row to a different public World identity.
        if (!domain.retiring || !domain.retired || domain.cleanup || writer == &domain || !domain.pending.empty() ||
            !domain.records.empty() || domain.liveCount)
            return false;
        const auto &worlds = storage.GetModel().worlds;
        if (domain.row < worlds.size()) {
            const auto &range = worlds[domain.row];
            if (range.bodies.y || range.leaves.y || range.leaves.w)
                return false;
        }
        // Service marks retired only after cancellation/cleanup and the zero-range
        // layout writer have completed all receipts. Retained Publications own
        // frozen resources/immutable identities, never this Partition or its row.
        const auto id = domain.id;
        ordered[domain.row] = nullptr;
        freeRows.push_back(domain.row); // Capacity was reserved at registration.
        domains.erase(id);
        return true;
    }

    JoltGpuControlWork PrepareLocalReset(Partition &domain)
    {
        const auto range = storage.GetModel().worlds[domain.row].bodies;
        struct Reset : std::enable_shared_from_this<Reset>
        {
            Impl &scene;
            uint32_t first, end;
            Reset(Impl &owner, uint32_t base, uint32_t count) : scene(owner), first(base), end(base + count)
            {
            }
            JoltGpuControlWork Next()
            {
                if (first == end)
                    return {};
                auto host = scene.storage.GetHost();
                auto uniform = rhi::ComputeBuffer::TryCreatePrivate(
                    host, {sizeof(StageWork) / sizeof(uint32_t), rhi::ComputeScalarType::UInt32, 1});
                if (!uniform)
                    return {nullptr, [self = shared_from_this()] { return self->Next(); }};
                const auto count = std::min(uint32_t{2048}, end - first);
                const StageWork range{{first, count, 0u, 0u}};
                std::vector<uint8_t> bytes(sizeof(range));
                std::memcpy(bytes.data(), &range, sizeof(range));
                auto dispatch = scene.mutations.PrepareCapacityInitialization(
                    {scene.storage.GetBodyCount(), scene.storage.Get(Resource::BodyStates),
                     scene.storage.Get(Resource::BodyForces), scene.storage.Get(Resource::KinematicTargets)},
                    first, count, uniform);
                first += count;
                auto packet = rhi::PrepareComputeBatch(host, {{uniform, 0u, std::move(bytes)}}, {std::move(dispatch)});
                return {packet, [self = shared_from_this()] { return self->Next(); }};
            }
        };
        return std::make_shared<Reset>(*this, range.x, range.y)->Next();
    }

    void Service()
    {
        if (writer &&
            (tick.GetStatus() == JoltGpuTick::Status::Failed || tick.GetStatus() == JoltGpuTick::Status::Cancelled) &&
            writerKind != WriterKind::Cancelled)
            Cancel(*writer);
        if (writer && writerKind == WriterKind::Cancelled) {
            if (!tick.CanRetire())
                return;
            if (tick.GetStatus() == JoltGpuTick::Status::Cancelled || tick.GetStatus() == JoltGpuTick::Status::Failed)
                tick.ReopenForPartitionCleanup();
            writer = nullptr;
        }
        if (writer && tick.IsComplete()) {
            if (writerKind == WriterKind::Cleanup) {
                writer->cleanup = false;
                writer->status = JoltGpuTick::Status::Cancelled;
                writer = nullptr;
            } else if (writerKind == WriterKind::Retirement) {
                writer->retired = true;
                writer->capacity = 0;
                writer->records.clear();
                writer->freeLocals.clear();
                writer->published.reset();
                writer = nullptr;
            } else
                PublishWriter();
        }
        if (writer)
            return;
        for (auto *domain : ordered) {
            if (!domain)
                continue;
            if (!domain->cleanup && (!domain->retiring || domain->retired))
                continue;
            const auto &worlds = storage.GetModel().worlds;
            const bool noNumericalRange =
                domain->row >= worlds.size() || (!worlds[domain->row].bodies.y && !worlds[domain->row].leaves.w);
            if (noNumericalRange) {
                ReleaseRecords(*domain);
                domain->cleanup = false;
                if (domain->retiring) {
                    domain->retired = true;
                    domain->capacity = 0;
                    domain->records.clear();
                    domain->freeLocals.clear();
                    domain->published.reset();
                }
                continue;
            }
            if (domain->cleanup)
                QueueCleanup(*domain);
            else
                QueueLayout(*domain, 0u);
            JoltGpuTickControl control;
            control.commandProgram = &commands;
            control.groups = std::move(domain->pending);
            writer = domain;
            writerKind = domain->cleanup ? WriterKind::Cleanup : WriterKind::Retirement;
            tick.SetAdmissionEnabled(true);
            tick.BeginControl(domain->row, std::move(control));
            break;
        }
    }

    PhysicsECSStore &actors;
    FrameWorkOwner &work;
    JoltGpuStorage storage;
    JoltGpuPrograms programs;
    JoltGpuBroadphase broadphase;
    JoltGpuConstraints constraints;
    JoltGpuBodyMutations mutations;
    JoltGpuBodyCommands commands;
    JoltGpuJoints &joints;
    std::map<uint64_t, JointRecord> authoredJoints;
    std::vector<PublishedJoint> acceptedJoints;
    uint64_t nextJointId = 1, layoutGeneration = 0;
    std::vector<Slot> slots;
    std::vector<uint32_t> freeRecords;
    std::unordered_map<uint32_t, uint32_t> byId;
    std::map<uint64_t, std::unique_ptr<Partition>> domains;
    // Stable row-address table, bounded by the concurrently registered high-water
    // mark. Null entries are retired zero-range rows available for later Worlds.
    std::vector<Partition *> ordered;
    std::vector<uint32_t> freeRows;
    Partition *writer = nullptr;
    WriterKind writerKind = WriterKind::Control;
    std::shared_ptr<Publication> candidate;
    std::shared_ptr<std::vector<TargetCompletion>> candidateCompletions;
    // Entire backend shutdown drains this one source; domain retirement never
    // destroys it. RuntimeWorld owners stay alive until their CanRetire result.
    JoltGpuTick tick;
};

JoltGpuScene::JoltGpuScene(rhi::ComputeHost &host, FrameWorkOwner &work, const ShaderTargetCode &stepTarget,
                           const ShaderTargetCode &mutationTarget, const ShaderTargetCode &commandTarget,
                           const ShaderTargetCode &broadphaseTarget, const ShaderTargetCode &constraintTarget,
                           JoltGpuJoints &joints)
    : m_impl(std::make_unique<Impl>(host, work, stepTarget, mutationTarget, commandTarget, broadphaseTarget,
                                    constraintTarget, joints))
{
}
JoltGpuScene::~JoltGpuScene() = default;
bool JoltGpuScene::RegisterPartition(RuntimeWorld &world, uint32_t capacity)
{
    m_impl->Service();
    const auto found = m_impl->domains.find(world.id);
    if (found != m_impl->domains.end()) {
        auto &domain = m_impl->Domain(world);
        if (domain.retiring || domain.retired)
            throw std::logic_error("A retired RuntimeWorld partition cannot reopen");
        if (domain.cleanup || m_impl->writer == &domain)
            return !domain.closed;
        if (domain.closed) {
            domain.closed = false;
            domain.status = JoltGpuTick::Status::Idle;
        }
        m_impl->ReserveCapacity(world, capacity);
        const auto &ranges = m_impl->storage.GetModel().worlds;
        if (domain.pending.empty() && (domain.row >= ranges.size() || ranges[domain.row].bodies.y < domain.capacity))
            m_impl->QueueLayout(domain, domain.capacity);
        return true;
    }
    m_impl->RegisterPartition(world, capacity);
    return true;
}
JoltGpuScene::BodyAllocation JoltGpuScene::Reserve(RuntimeWorld &world, uint32_t id, PhysicsActorHandle actor,
                                                   OwnerIdentity owner)
{
    return m_impl->Reserve(world, id, actor, std::move(owner));
}
void JoltGpuScene::ReserveCapacity(RuntimeWorld &world, uint32_t count)
{
    m_impl->ReserveCapacity(world, count);
}
void JoltGpuScene::StageActor(uint32_t id, PreparedGpuActor actor)
{
    m_impl->StageActor(id, std::move(actor));
}
void JoltGpuScene::SetResident(uint32_t id, bool value, bool preserve)
{
    m_impl->SetResident(id, value, preserve);
}
PhysicsTargetSource JoltGpuScene::QueueCommand(uint32_t id, BodyCommand value, const PhysicsTargetAuthor *author)
{
    return m_impl->QueueCommand(id, value, author);
}
void JoltGpuScene::StageRetire(uint32_t id)
{
    m_impl->StageRetire(id);
}
uint64_t JoltGpuScene::CreateJoint(PhysicsConstraintOwner &owner, uint32_t a, uint32_t b, JointAuthor author)
{
    return m_impl->CreateJoint(owner, a, b, author);
}
void JoltGpuScene::UpdateJoint(uint64_t id, const std::function<void(JointAuthor &)> &update)
{
    m_impl->UpdateJoint(id, update);
}
void JoltGpuScene::UpdateJoints(std::vector<JointUpdate> values)
{
    m_impl->UpdateJoints(std::move(values));
}
void JoltGpuScene::DestroyJoint(uint64_t id)
{
    m_impl->DestroyJoint(id);
}
void JoltGpuScene::DestroyJointsForBody(uint32_t id)
{
    m_impl->DestroyJointsForBody(id);
}
const JoltGpuScene::PublishedJoint &JoltGpuScene::GetAuthoredJoint(uint64_t id) const
{
    return m_impl->GetAuthoredJoint(id);
}
JoltGpuScene::BodyAllocation JoltGpuScene::FindBody(uint32_t id) const noexcept
{
    const auto found = m_impl->byId.find(id);
    if (found == m_impl->byId.end())
        return {};
    const auto &slot = m_impl->slots[found->second];
    if (slot.retiring || slot.partition->closed)
        return {};
    const auto &worlds = m_impl->storage.GetModel().worlds;
    const auto row = slot.partition->row;
    const uint32_t packed =
        row < worlds.size() && slot.local < worlds[row].bodies.y ? worlds[row].bodies.x + slot.local : UINT32_MAX;
    return {id, packed};
}
PhysicsActorHandle JoltGpuScene::FindActor(uint32_t id) const noexcept
{
    const auto found = m_impl->byId.find(id);
    return found == m_impl->byId.end() || FindBody(id).bodyId == InvalidBodyId ? PhysicsActorHandle{}
                                                                               : m_impl->slots[found->second].actor;
}
size_t JoltGpuScene::GetBodyCount(const RuntimeWorld &world) const
{
    if (!m_impl->domains.count(world.id))
        return 0;
    const auto &domain = m_impl->Domain(world);
    return domain.liveCount;
}
uint32_t JoltGpuScene::GetCapacity(const RuntimeWorld &world) const
{
    return m_impl->domains.count(world.id) ? m_impl->Domain(world).capacity : 0u;
}
bool JoltGpuScene::CanReserve(const RuntimeWorld &world) const
{
    if (!m_impl->domains.count(world.id))
        return false;
    const auto &domain = m_impl->Domain(world);
    return !domain.closed && !domain.freeLocals.empty();
}
bool JoltGpuScene::HasPendingControl(const RuntimeWorld &world) const
{
    if (!m_impl->domains.count(world.id))
        return false;
    const auto &domain = m_impl->Domain(world);
    return !domain.pending.empty() || (m_impl->writer == &domain && !m_impl->tick.AreControlInputsAccepted());
}
bool JoltGpuScene::TryBeginTick(RuntimeWorld &world, uint64_t epoch, uint64_t step, float dt,
                                const JoltGpuStepParameters &parameters, Consumer finalConsumer)
{
    return m_impl->TryBegin(m_impl->Domain(world), epoch, step, dt, &parameters, {}, std::move(finalConsumer));
}
bool JoltGpuScene::TryBeginControl(RuntimeWorld &world, uint64_t epoch, Consumer consumer, Consumer finalConsumer)
{
    m_impl->Service();
    auto &domain = m_impl->Domain(world);
    const auto step =
        domain.published && domain.published->simulationEpoch == epoch ? domain.published->fixedStepIndex : 0u;
    return m_impl->TryBegin(domain, epoch, step, 0.0f, nullptr, std::move(consumer), std::move(finalConsumer));
}
bool JoltGpuScene::TryBeginWork(RuntimeWorld &world, JoltGpuControlWork &value)
{
    m_impl->Service();
    auto &domain = m_impl->Domain(world);
    if (m_impl->writer || domain.closed || !domain.admission || domain.row >= m_impl->storage.GetWorldCount())
        return false;
    JoltGpuTickControl control;
    JoltGpuControlGroup group;
    group.prepareWork = [value = std::move(value)]() mutable { return std::move(value); };
    control.groups.push_back(std::move(group));
    m_impl->writer = &domain;
    m_impl->writerKind = Impl::WriterKind::Work;
    domain.status = JoltGpuTick::Status::Running;
    m_impl->tick.SetAdmissionEnabled(true);
    m_impl->tick.BeginControl(domain.row, std::move(control));
    return true;
}
bool JoltGpuScene::PublishCompletedTick(RuntimeWorld &world)
{
    m_impl->Service();
    auto &domain = m_impl->Domain(world);
    if (domain.observedTick == domain.tickSerial)
        return false;
    domain.observedTick = domain.tickSerial;
    return true;
}
bool JoltGpuScene::PublishCompletedControl(RuntimeWorld &world)
{
    m_impl->Service();
    auto &domain = m_impl->Domain(world);
    if (domain.observedControl == domain.controlSerial)
        return false;
    domain.observedControl = domain.controlSerial;
    return true;
}
const std::shared_ptr<const JoltGpuScene::Publication> &JoltGpuScene::GetPublished(const RuntimeWorld &world) const
{
    static const std::shared_ptr<const Publication> empty;
    return m_impl->domains.count(world.id) ? m_impl->Domain(world).published : empty;
}
bool JoltGpuScene::IsWriterFor(const RuntimeWorld &world) const noexcept
{
    return m_impl->writer && m_impl->writer->world == &world;
}
JoltGpuTick::Status JoltGpuScene::GetTickStatus(const RuntimeWorld &world)
{
    m_impl->Service();
    return m_impl->Domain(world).status;
}
FrameWorkOwner::SourceId JoltGpuScene::GetSourceId() const noexcept
{
    return m_impl->tick.GetSourceId();
}
bool JoltGpuScene::AreControlInputsAccepted(const RuntimeWorld &world) const
{
    return !IsWriterFor(world) || m_impl->tick.AreControlInputsAccepted();
}
FrameWorkWaitResult JoltGpuScene::WaitForProgress(RuntimeWorld &world, uint64_t maxWait)
{
    m_impl->Service();
    return IsWriterFor(world) ? m_impl->tick.WaitForProgress(maxWait) : FrameWorkWaitResult::Blocked;
}
FrameWorkWaitResult JoltGpuScene::WaitForSourceProgress(uint64_t maxWait)
{
    return m_impl->tick.WaitForProgress(maxWait);
}
void JoltGpuScene::SetAdmissionEnabled(RuntimeWorld &world, bool enabled)
{
    auto &domain = m_impl->Domain(world);
    domain.admission = enabled;
    // Admission controls future logical work. A previously admitted shared
    // writer must still retire when another world's synchronous read needs it.
    if (m_impl->writer)
        m_impl->tick.SetAdmissionEnabled(true);
}
bool JoltGpuScene::IsAdmissionEnabled(const RuntimeWorld &world) const
{
    return m_impl->Domain(world).admission;
}
void JoltGpuScene::Cancel(RuntimeWorld &world)
{
    if (!m_impl->domains.count(world.id))
        return;
    m_impl->Cancel(m_impl->Domain(world));
    m_impl->Service();
}
void JoltGpuScene::RetirePartition(RuntimeWorld &world)
{
    if (!m_impl->domains.count(world.id))
        return;
    auto &domain = m_impl->Domain(world);
    domain.retiring = true;
    m_impl->Cancel(domain);
    m_impl->Service();
}
bool JoltGpuScene::CanRetire(const RuntimeWorld &world)
{
    if (!m_impl->domains.count(world.id))
        return true;
    m_impl->Service();
    auto &domain = m_impl->Domain(world);
    if (domain.cleanup || m_impl->writer == &domain)
        return false;
    if (!domain.retiring)
        return true; // Cancel/reopen retains the same partition, capacity and row.
    return m_impl->ReleaseRetiredPartition(domain);
}
} // namespace infernux
