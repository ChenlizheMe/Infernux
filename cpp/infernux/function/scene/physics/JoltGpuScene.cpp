#include "JoltGpuScene.h"
#include "JoltBodyTransfer.h"
#include "JoltGpuArticulations.h"
#include "JoltGpuSleep.h"
#include "PhysicsDomain.h"

#include "JoltGpuBodyMutations.h"
#include "JoltGpuBroadphase.h"
#include "JoltGpuConstraints.h"
#include "JoltGpuGeometryStore.h"
#include "JoltGpuJoints.h"
#include "JoltGpuNarrowphase.h"
#include "JoltGpuPackedLayout.h"
#include <cstring>
#include <function/scene/Collider.h>
#include <function/scene/GameObject.h>
#include <function/scene/Rigidbody.h>
#include <function/scene/RuntimeWorld.h>
#include <function/scene/SceneManager.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
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

Float4 TransferVector(JPH::Vec3Arg value, float extra = 0.0f)
{
    return {value.GetX(), value.GetY(), value.GetZ(), extra};
}

JPH::Vec3 TransferVector(const Float4 &value)
{
    return {value.x, value.y, value.z};
}

Float4 TransferRotation(JPH::QuatArg value)
{
    return {value.GetX(), value.GetY(), value.GetZ(), value.GetW()};
}

JPH::Quat TransferRotation(const Float4 &value)
{
    return {value.x, value.y, value.z, value.w};
}

template <class Visit> auto WithJointState(uint32_t kind, const Visit &visit)
{
    switch (kind) {
    case JointFixed:
        return visit(JointFixedState{});
    case JointHinge:
        return visit(JointHingeState{});
    case JointSlider:
        return visit(JointSliderState{});
    case JointDistance:
        return visit(JointDistanceState{});
    case JointSwingTwist:
        return visit(JointSwingState{});
    case JointSixDOF:
        return visit(JointSixState{});
    }
    throw std::logic_error("Cannot transfer an unknown GPU joint kind");
}

// Native Jolt represents a hinge by (axis, normal, axis cross normal).
// The GPU hinge keeps the same frame in (normal, axis cross normal, axis).
const JPH::Quat HingeAxisOrder(0.5f, 0.5f, 0.5f, 0.5f);

template <class State>
constexpr bool JointHasPoseDrive = std::is_same_v<State, JointHingeState> || std::is_same_v<State, JointSwingState> ||
                                   std::is_same_v<State, JointSixState>;

template <class State> JoltJointTransfer DecodeJointTransfer(const State &state, const JointAuthor &author)
{
    JoltJointTransfer value;
    value.connected = {TransferVector(state.anchor1), TransferVector(state.collapsed1)};
    value.owner = {TransferVector(state.anchor2), TransferVector(state.collapsed2)};
    if constexpr (std::is_same_v<State, JointDistanceState>) {
        value.automaticDistance = state.distanceReference.w;
    } else {
        value.connected.frame = TransferRotation(state.frame1);
        value.owner.frame = TransferRotation(state.frame2);
        if constexpr (std::is_same_v<State, JointHingeState>) {
            value.connected.frame = value.connected.frame * HingeAxisOrder.Conjugated();
            value.owner.frame = value.owner.frame * HingeAxisOrder.Conjugated();
            const auto armature = author.passive().z;
            value.rotorSpeed = armature > 0.0f ? state.rotor.x - state.rotor.y / armature : state.rotor.x;
        }
    }
    if constexpr (JointHasPoseDrive<State>)
        if (state.flags.w & JointPoseDriveActive)
            value.poseDrive = TransferRotation(state.poseDriveRelative);
    return value;
}

template <class State> State EncodeJointTransfer(State state, const JoltJointTransfer &value, const JointAuthor &author)
{
    state.flags = {1, JointStatusSuccess, 0, 0};
    state.anchor1 = TransferVector(value.connected.unscaled);
    state.anchor2 = TransferVector(value.owner.unscaled);
    state.collapsed1 = TransferVector(value.connected.collapsed);
    state.collapsed2 = TransferVector(value.owner.collapsed);
    if constexpr (std::is_same_v<State, JointDistanceState>) {
        state.distanceReference = {0, 1, 0, value.automaticDistance};
    } else if constexpr (std::is_same_v<State, JointHingeState>) {
        state.frame1 = TransferRotation(value.connected.frame * HingeAxisOrder);
        state.frame2 = TransferRotation(value.owner.frame * HingeAxisOrder);
        state.rotor = {float(value.rotorSpeed), 0, 0, author.passive().z};
    } else {
        state.frame1 = TransferRotation(value.connected.frame);
        state.frame2 = TransferRotation(value.owner.frame);
    }
    if constexpr (JointHasPoseDrive<State>) {
        state.poseDriveRelative = TransferRotation(value.poseDrive.value_or(JPH::Quat::sIdentity()));
        if (value.poseDrive)
            state.flags.w |= JointPoseDriveActive;
    }
    return state;
}

template <class T>
rhi::ComputeBufferUpdate TransferUpload(JoltGpuStorage &storage, Resource resource, uint32_t slot, const T &value)
{
    std::vector<uint8_t> bytes(sizeof(T));
    std::memcpy(bytes.data(), &value, sizeof(T));
    return {storage.Get(resource), uint64_t(slot) * sizeof(T), std::move(bytes)};
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

    struct ActorChanges;

    struct Partition
    {
        RuntimeWorld *world = nullptr;
        uint64_t id = 0;
        uint32_t row = 0, capacity = 0, liveCount = 0;
        std::vector<uint32_t> records;
        std::vector<uint32_t> freeLocals;
        std::vector<JoltGpuControlGroup> pending;
        std::shared_ptr<ActorChanges> trailingCreations;
        size_t trailingCreationGroup = 0;
        std::shared_ptr<JoltGpuSleep::BoundsValidation> pendingBoundsValidation;
        std::shared_ptr<PhysicsTargetBatch> pendingSources;
        std::vector<TargetCompletion> pendingCompletions;
        std::shared_ptr<const Publication> published;
        std::shared_ptr<const PhysicsColliderPairs> acceptedPairs = std::make_shared<const PhysicsColliderPairs>();
        std::shared_ptr<const PhysicsColliderPairs> queuedPairs;
        std::shared_ptr<const JoltGpuColliderPairs> pairCache;
        std::shared_ptr<JoltGpuArticulationState> articulationCache;
        std::shared_ptr<Publication> candidate;
        std::shared_ptr<std::vector<TargetCompletion>> candidateCompletions;
        JoltGpuTick *cursor = nullptr;
        bool shareable = false;
        uint64_t memberEpoch = 0, memberStep = 0;
        JoltGpuTick::Status status = JoltGpuTick::Status::Idle;
        uint64_t tickSerial = 0, controlSerial = 0, observedTick = 0, observedControl = 0;
        bool admission = true, closed = false, cleanup = false, retiring = false;
    };

    struct PairUpload : std::enable_shared_from_this<PairUpload>
    {
        Impl *scene;
        Partition *domain;
        std::shared_ptr<Publication> publication;
        std::shared_ptr<JoltGpuColliderPairCapture> capture;
        std::shared_ptr<JoltGpuColliderPairs> pairs;
        uint32_t first = 0;
        JoltGpuControlWork Next()
        {
            auto host = scene->storage.GetHost();
            const uint64_t total = std::max<size_t>(1, pairs->keys.size());
            if (!pairs->buffer) {
                const uint64_t bytes = total * sizeof(ColliderPairKey);
                if (total > UINT32_MAX || !host.device.GetCapabilities().IsStorageBufferRangeValid(0, bytes))
                    throw std::length_error("Collider pair table exceeds device range");
                pairs->buffer =
                    rhi::ComputeBuffer::TryCreatePrivate(host, {bytes / 4, rhi::ComputeScalarType::UInt32, 1});
                if (!pairs->buffer)
                    return {nullptr, [self = shared_from_this()] { return self->Next(); }};
            }
            if (first == total) {
                pairs->buffer = pairs->buffer->Freeze();
                domain->pairCache = pairs;
                publication->colliderPairs = pairs;
                capture->value = pairs;
                return {};
            }
            const uint32_t count = static_cast<uint32_t>(std::min<uint64_t>(1024, total - first));
            std::vector<uint8_t> bytes(uint64_t(count) * sizeof(ColliderPairKey));
            if (!pairs->keys.empty())
                std::memcpy(bytes.data(), pairs->keys.data() + first, bytes.size());
            auto packet = rhi::PrepareComputeBatch(
                host, {{pairs->buffer, uint64_t(first) * sizeof(ColliderPairKey), std::move(bytes)}}, {}, {});
            return {packet,
                    [self = shared_from_this(), count] {
                        self->first += count;
                        return self->Next();
                    },
                    JoltGpuControlWork::ResumeAfter::Acceptance};
        }
    };

    JoltGpuControlWork PrepareColliderPairs(Partition &domain, std::shared_ptr<Publication> publication,
                                            std::shared_ptr<JoltGpuColliderPairCapture> capture)
    {
        if (domain.pairCache && domain.pairCache->identities == domain.acceptedPairs) {
            publication->colliderPairs = domain.pairCache;
            capture->value = domain.pairCache;
            return {};
        }
        auto pairs = std::make_shared<JoltGpuColliderPairs>();
        pairs->identities = domain.acceptedPairs;
        pairs->keys.reserve(pairs->identities->size());
        for (const auto &pair : *pairs->identities) {
            pairs->keys.push_back({{static_cast<uint32_t>(pair[0]), static_cast<uint32_t>(pair[0] >> 32),
                                    static_cast<uint32_t>(pair[1]), static_cast<uint32_t>(pair[1] >> 32)},
                                   {static_cast<uint32_t>(pair[2]), static_cast<uint32_t>(pair[2] >> 32),
                                    static_cast<uint32_t>(pair[3]), static_cast<uint32_t>(pair[3] >> 32)}});
        }
        std::sort(pairs->keys.begin(), pairs->keys.end(), [](const auto &a, const auto &b) {
            return std::tie(a.a.y, a.a.x, a.a.w, a.a.z, a.b.y, a.b.x, a.b.w, a.b.z) <
                   std::tie(b.a.y, b.a.x, b.a.w, b.a.z, b.b.y, b.b.x, b.b.w, b.b.z);
        });
        auto upload = std::make_shared<PairUpload>();
        upload->scene = this;
        upload->domain = &domain;
        upload->publication = std::move(publication);
        upload->capture = std::move(capture);
        upload->pairs = std::move(pairs);
        return upload->Next();
    }

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
        BodySleepAuthor sleep;
        std::vector<ShapeModel> leaves;
        BodyState initialState;
        PhysicsActorPublication publication;
        std::shared_ptr<PublishedActor> identity;
        bool initializesRigidbody = false;
    };

    struct ActorChanges
    {
        std::vector<std::shared_ptr<ActorChange>> actors;
        bool deferred = false;
        bool wakeExisting = false;
    };

    struct JointRecord
    {
        PublishedJoint value;
        PhysicsConstraintOwner *owner = nullptr;
        // Last authored drive intent, not a mirror of device dynamic state.
        std::optional<PoseDriveIntent> poseDrive;
    };

    StepParams SleepSelection(const Partition &domain) const
    {
        const auto &range = storage.GetModel().worlds.at(domain.row);
        StepParams result{};
        result.selection = {domain.row, range.bodies.x, range.leaves.x, range.leaves.z};
        result.selectedCounts = {range.bodies.y, range.leaves.y, range.leaves.w, 0u};
        return result;
    }
    static std::shared_ptr<JoltGpuSleep::BoundsValidation> PendingBoundsValidation(Partition &domain)
    {
        if (!domain.pendingBoundsValidation)
            domain.pendingBoundsValidation = std::make_shared<JoltGpuSleep::BoundsValidation>();
        return domain.pendingBoundsValidation;
    }
    std::vector<JoltGpuControlGroup> TakeControlGroups(Partition &domain)
    {
        if (auto validation = std::exchange(domain.pendingBoundsValidation, {})) {
            // Bounds edits retain their command order and report device errors before publishing the batch.
            JoltGpuControlGroup check;
            check.prepareWork = [this, validation] { return programs.GetSleep().ValidateBoundsEdits(*validation); };
            domain.pending.push_back(std::move(check));
        }
        domain.trailingCreations.reset();
        return std::exchange(domain.pending, {});
    }

    using OldBounds = std::shared_ptr<std::shared_ptr<rhi::ComputeBuffer>>;
    OldBounds QueueOldBounds(Partition &domain, uint32_t record, uint32_t id)
    {
        auto old = std::make_shared<std::shared_ptr<rhi::ComputeBuffer>>();
        JoltGpuControlGroup capture;
        capture.prepareWork = [this, &domain, record, id, old, validation = PendingBoundsValidation(domain)] {
            RequireOwnership(record, id);
            return programs.GetSleep().PrepareOldBounds(storage, SleepSelection(domain), Packed(record), validation,
                                                        [old](auto value) { *old = std::move(value); });
        };
        domain.pending.push_back(std::move(capture));
        return old;
    }
    void QueueBoundsAfter(Partition &domain, OldBounds old, uint32_t record, uint32_t id, bool newBody)
    {
        JoltGpuControlGroup wake;
        wake.prepareWork = [this, &domain, old, record, id, newBody, validation = PendingBoundsValidation(domain)] {
            uint32_t source = UINT32_MAX;
            if (newBody) {
                RequireOwnership(record, id);
                source = Packed(record);
            }
            return programs.GetSleep().PrepareBoundsWake(storage, SleepSelection(domain), *old, source, validation);
        };
        domain.pending.push_back(std::move(wake));
    }
    void AddJointWake(JoltGpuControlGroup &group, const PublishedJoint &joint)
    {
        for (auto id : {joint.bodyIdA, joint.bodyIdB})
            if (id != InvalidBodyId) {
                BodyCommand wake{};
                wake.meta = {id, CommandWake, 0u, 0u};
                group.commands.push_back(wake);
            }
    }

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
         const ShaderTargetCode &broadphaseTarget, const ShaderTargetCode &narrowphaseTarget,
         const ShaderTargetCode &constraintTarget, JoltGpuJoints &jointProgram, FailureObserver onFailure,
         const ShaderTargetCode *captureTarget, const ShaderTargetCode *dynamicsTarget)
        : actors(PhysicsECSStore::Instance()), work(budget), geometry(std::make_shared<JoltGpuGeometryStore>(host)),
          storage(host, JoltGpuModel{}), programs(host, stepTarget, captureTarget, dynamicsTarget),
          broadphase(host, broadphaseTarget), narrowphase(host, narrowphaseTarget), constraints(host, constraintTarget),
          mutations(host, mutationTarget), commands(host, commandTarget), joints(jointProgram),
          failureObserver(std::move(onFailure)),
          tick(budget, storage, programs, broadphase, narrowphase, constraints, [this] {
              const auto *owner = writer ? writer : pressureDomain;
              if (owner && !owner->retiring && failureObserver)
                  failureObserver(*owner->world);
          })
    {
        storage.SetGeometryStore(geometry);
        tick.SetPressureOwner([this] { return ObservePressure(); }, [this] { return pressurePending; },
                              [this](uint32_t witnessWorld, auto completed) {
                                  return PreparePressureRetirement(witnessWorld, std::move(completed));
                              });
    }

    bool ObservePressure()
    {
        const bool hasCache = std::any_of(ordered.begin(), ordered.end(), [](const auto *domain) {
            const auto state = domain ? domain->articulationCache : nullptr;
            return state && !state->resourceDeclined && (!state->trees.empty() || state->numeric);
        });
        const auto fact = storage.GetHost().device.ObserveManagedDeviceLocalPressure();
        if (fact.serial != observedPressure) {
            // Release optional tile tables at the actual pressure fact, even
            // when there is no P3 cache that requires a representation sweep.
            tick.ReleaseBodyBatchInputs();
            observedPressure = fact.serial;
            if (hasCache || pressurePending) {
                if (!pressurePending)
                    pressureCursor = 0;
                pressurePending = true;
            }
        }
        return pressurePending || hasCache;
    }

    JoltGpuControlWork PreparePressureRetirement(uint32_t witnessWorld, JoltGpuTick::PressureCompletion completed)
    {
        while (pressureCursor < ordered.size()) {
            auto *domain = ordered[pressureCursor];
            const auto state = domain ? domain->articulationCache : nullptr;
            if (!state || state->resourceDeclined || (state->trees.empty() && !state->numeric)) {
                ++pressureCursor;
                continue;
            }
            // Advance this cursor only at completed conversion. Cancellation
            // retains the same origin so a later sweep can resume idempotently.
            pressureDomain = domain;
            return ContinueJoltGpuControlWork(
                programs.GetArticulations().PreparePressureDeclined(storage, state, domain->row == witnessWorld,
                                                                    [this, domain, completed](auto empty) {
                                                                        domain->articulationCache = empty;
                                                                        completed(domain->row, std::move(empty));
                                                                        ++pressureCursor;
                                                                        pressureDomain = nullptr;
                                                                    }),
                [this, witnessWorld, completed] { return PreparePressureRetirement(witnessWorld, completed); });
        }
        pressurePending = false;
        pressureDomain = nullptr;
        return {};
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
        auto deferred = std::make_shared<bool>(false);
        JoltGpuControlGroup group;
        group.modelDeferred = [deferred] { return *deferred; };
        group.prepareModel = [this, own, catalog, deferred, &domain]() -> std::unique_ptr<JoltGpuModelUpdate> {
            catalog->clear();
            *deferred = false;
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
                                          storage.GetJointLayout(), storage.Get(Resource::BodyStates)->Freeze());
            if (!prepared.stream) {
                *deferred = true;
                return {};
            }
            return storage.PrepareJointReplacement(std::move(authors), std::move(prepared.stream),
                                                   std::move(prepared.updates), std::move(prepared.dispatches));
        };
        group.modelAccepted = [this, catalog]() noexcept {
            acceptedJoints.swap(*catalog);
            publicationDirectory.reset();
        };
        domain.pending.push_back(std::move(group));
    }

    uint64_t CreateJoint(PhysicsConstraintOwner &owner, uint32_t ownerBody, uint32_t connectedBody, JointAuthor author,
                         uint64_t importedId)
    {
        const auto a = RequireSlot(ownerBody);
        const auto b = connectedBody == InvalidBodyId ? InvalidQueryBodySlot : RequireSlot(connectedBody);
        if (b != InvalidQueryBodySlot && slots[a].partition != slots[b].partition)
            throw std::invalid_argument("GPU Joint endpoints belong to different RuntimeWorld owners");
        if (ownerBody == connectedBody || !slots[a].authored || (b != InvalidQueryBodySlot && !slots[b].authored))
            throw std::invalid_argument("A GPU joint requires distinct authored body endpoints");
        const auto id = importedId ? importedId : actors.AllocateJointIdentity();
        author.bodies.x = b;
        author.bodies.y = a;
        author.identity = {static_cast<uint32_t>(id), static_cast<uint32_t>(id >> 32), slots[a].partition->row, 0};
        author.rootCOM1() = b == InvalidQueryBodySlot ? Float4{} : slots[b].authored->rootCenterOfMass;
        author.rootCOM2() = slots[a].authored->rootCenterOfMass;
        author.scale1() = b == InvalidQueryBodySlot ? Float4{1, 1, 1, 0} : slots[b].authored->rootScale;
        author.scale2() = slots[a].authored->rootScale;
        PublishedJoint value{id, 0, ownerBody, connectedBody, author.bodies.z, author};
        if (!authoredJoints.emplace(id, JointRecord{value, &owner}).second)
            throw std::logic_error("GPU joint already owns its public identity");
        if (!importedId) {
            QueueJointTopology(*slots[a].partition);
            JoltGpuControlGroup wake;
            AddJointWake(wake, value);
            slots[a].partition->pending.push_back(std::move(wake));
        }
        return id;
    }

    void QueueJointTransferRestore(Partition &domain, std::vector<JoltJointImport> transfers)
    {
        if (transfers.empty())
            return;
        JoltGpuControlGroup wake;
        for (const auto &transfer : transfers) {
            auto &record = authoredJoints.at(transfer.id);
            if (!transfer.state)
                AddJointWake(wake, record.value);
            else if (const auto &pose = transfer.state->value().poseDrive)
                record.poseDrive = PoseDriveIntent{transfer.id, record.value.kind, TransferRotation(*pose)};
        }
        QueueJointTopology(domain);
        if (!wake.commands.empty())
            domain.pending.push_back(std::move(wake));
        transfers.erase(
            std::remove_if(transfers.begin(), transfers.end(), [](const auto &transfer) { return !transfer.state; }),
            transfers.end());
        if (transfers.empty())
            return;
        JoltGpuControlGroup group;
        group.prepareWork = [this, transfers = std::move(transfers)] {
            std::vector<rhi::ComputeBufferUpdate> writes;
            writes.reserve(transfers.size());
            const auto &layout = *storage.GetJointLayout();
            const auto &stream = storage.Get(Resource::Joints);
            for (const auto &transfer : transfers) {
                const auto &joint = AcceptedJoint(transfer.id);
                auto bytes = WithJointState(joint.kind, [&](auto empty) {
                    auto state = EncodeJointTransfer(empty, transfer.state->value(), joint.author);
                    std::vector<uint8_t> data(sizeof(state));
                    std::memcpy(data.data(), &state, sizeof(state));
                    return data;
                });
                writes.push_back({stream, uint64_t(layout.Row(joint.row).layout.y) * 16, std::move(bytes)});
            }
            auto host = storage.GetHost();
            return JoltGpuControlWork{rhi::PrepareComputeBatch(host, std::move(writes), {})};
        };
        domain.pending.push_back(std::move(group));
    }

    void TransferJointsOut(Partition &domain, const std::vector<uint64_t> &ids)
    {
        if (ids.empty())
            return;
        for (const auto id : ids)
            authoredJoints.erase(id);
        QueueJointTopology(domain);
    }

    void UpdateJoint(uint64_t id, const std::function<void(JointAuthor &)> &update, uint32_t runtimeIntent)
    {
        auto author = GetAuthoredJoint(id).author;
        update(author);
        UpdateJoints({{id, author, runtimeIntent}});
    }

    void UpdateJoints(std::vector<JointUpdate> updates)
    {
        if (updates.empty())
            return;
        auto &domain = BodyDomain(GetAuthoredJoint(updates.front().id).bodyIdA);
        auto &pending = domain.pending;
        struct CapturedUpdate
        {
            PublishedJoint value;
            uint32_t runtimeIntent;
        };
        auto values = std::make_shared<std::vector<CapturedUpdate>>();
        values->reserve(updates.size());
        for (const auto &update : updates) {
            auto value = GetAuthoredJoint(update.id);
            if (&BodyDomain(value.bodyIdA) != &domain)
                throw std::invalid_argument("GPU joint batch crosses RuntimeWorld domains");
            value.author = update.author;
            value.author.identity = GetAuthoredJoint(update.id).author.identity;
            values->push_back({value, update.runtimeIntent});
        }
        JoltGpuControlGroup group;
        for (const auto &captured : *values) {
            const auto &old = GetAuthoredJoint(captured.value.id).author;
            const auto &author = captured.value.author;
            const bool samePayload = std::visit(
                [&](const auto &payload) {
                    const auto *previous = std::get_if<std::decay_t<decltype(payload)>>(&old.payload);
                    return previous && std::memcmp(previous, &payload, sizeof(payload)) == 0;
                },
                author.payload);
            if (captured.runtimeIntent || !samePayload || std::memcmp(&old.bodies, &author.bodies, sizeof(UInt4)) ||
                std::memcmp(&old.identity, &author.identity, sizeof(UInt4)))
                AddJointWake(group, captured.value);
            if (captured.runtimeIntent & JointResetPoseDrive)
                authoredJoints.at(captured.value.id).poseDrive.reset();
        }
        auto deferred = std::make_shared<bool>(false);
        group.modelDeferred = [deferred] { return *deferred; };
        group.prepareModel = [this, values, deferred] {
            std::vector<JoltGpuJointModelUpdate> changes;
            changes.reserve(values->size());
            for (auto &captured : *values) {
                auto &value = captured.value;
                value.row = AcceptedJoint(value.id).row;
                value.author.bodies.x =
                    value.bodyIdB == InvalidBodyId ? InvalidQueryBodySlot : Packed(byId.at(value.bodyIdB));
                value.author.bodies.y = Packed(byId.at(value.bodyIdA));
                value.author.identity.z = slots[byId.at(value.bodyIdA)].partition->row;
                changes.push_back({value.row, value.author, captured.runtimeIntent});
            }
            return joints.PrepareUpdate(storage, std::move(changes), *deferred);
        };
        group.modelAccepted = [this, values]() noexcept {
            for (const auto &captured : *values)
                acceptedJoints[captured.value.row].author = captured.value.author;
            publicationDirectory.reset();
        };
        pending.push_back(std::move(group));
        for (const auto &captured : *values)
            authoredJoints.at(captured.value.id).value.author = captured.value.author;
    }

    void RefreshJointBody(uint32_t bodyId, const PublishedActor &actor)
    {
        std::vector<JointUpdate> changes;
        for (const auto &[id, joint] : authoredJoints)
            if (joint.value.bodyIdA == bodyId || joint.value.bodyIdB == bodyId) {
                auto author = joint.value.author;
                if (joint.value.bodyIdA == bodyId) {
                    author.rootCOM2() = actor.rootCenterOfMass;
                    author.scale2() = actor.rootScale;
                } else {
                    author.rootCOM1() = actor.rootCenterOfMass;
                    author.scale1() = actor.rootScale;
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
        JoltGpuControlGroup wake;
        AddJointWake(wake, found->second.value);
        authoredJoints.erase(found);
        QueueJointTopology(domain);
        domain.pending.push_back(std::move(wake));
        owner->OnPhysicsConstraintDestroyed();
    }

    void DestroyJointsForBodies(Partition &domain, const std::vector<uint32_t> &ids)
    {
        std::vector<PhysicsConstraintOwner *> owners;
        JoltGpuControlGroup wake;
        for (auto it = authoredJoints.begin(); it != authoredJoints.end();) {
            if (std::binary_search(ids.begin(), ids.end(), it->second.value.bodyIdA) ||
                std::binary_search(ids.begin(), ids.end(), it->second.value.bodyIdB)) {
                AddJointWake(wake, it->second.value);
                owners.push_back(it->second.owner);
                it = authoredJoints.erase(it);
            } else {
                ++it;
            }
        }
        if (!owners.empty())
            QueueJointTopology(domain);
        if (!wake.commands.empty())
            domain.pending.push_back(std::move(wake));
        for (auto *owner : owners)
            owner->OnPhysicsConstraintDestroyed();
    }

    void DestroyJointsForBody(uint32_t id)
    {
        DestroyJointsForBodies(BodyDomain(id), {id});
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
        prepare.retireJointRuntime = [transaction] {
            if (*transaction)
                (*transaction)->InvalidateJointRuntimeCopies();
        };
        commit.reprepareModel = prepare.prepareWork;
        commit.prepareModel = [transaction] { return (*transaction)->Commit(); };
        commit.modelAccepted = [this, plan]() noexcept {
            for (const auto &[record, identity] : (*plan)->identities)
                slots[record].accepted = identity;
            acceptedJoints.swap((*plan)->catalog);
            ++layoutGeneration;
            publicationDirectory.reset();
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

    BodyAllocation Reserve(RuntimeWorld &world, uint32_t id, PhysicsActorHandle actor, OwnerIdentity owner,
                           bool initialResident, uint32_t expectedAuthorBodyId)
    {
        auto &domain = Domain(world);
        if (domain.closed || !actors.IsValid(actor) || id == InvalidBodyId || byId.count(id))
            throw std::invalid_argument("GPU body reservation requires a new live author identity");
        const auto &author = actors.GetActor(actor);
        if (author.runtimeWorld != &world || author.bodyId != expectedAuthorBodyId || !author.owner ||
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
        slots[record].desiredResident = initialResident;
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
        change->publication.AcceptGeometry(change->identity->actor, change->bodyId, change->identity->shapeRevision);
        if (change->initializesRigidbody)
            change->publication.ConsumeInitialVelocities(change->identity->actor, change->bodyId);
    }

    struct ModelEdit
    {
        uint32_t bodyIndex;
        BodyModel body;
        const std::vector<ShapeModel> *leaves;
    };

    JoltGpuModel RepackModel(const std::vector<ModelEdit> &edits) const
    {
        // Copy each accepted leaf range once, applying the ordered slot edits in
        // this transaction. Earlier control groups are already part of the input.
        const auto &accepted = storage.GetModel();
        uint64_t leafCount = accepted.shapes.size();
        for (const auto &edit : edits)
            leafCount = leafCount - accepted.bodies[edit.bodyIndex].meta.y + edit.leaves->size();
        if (leafCount > std::numeric_limits<uint32_t>::max())
            throw std::length_error("Jolt GPU scene exceeds its leaf index range");
        JoltGpuModel replacement;
        replacement.bodies = accepted.bodies;
        replacement.shapes.reserve(static_cast<size_t>(leafCount));
        auto edit = edits.begin();
        for (uint32_t slot = 0; slot < replacement.bodies.size(); ++slot) {
            auto &body = replacement.bodies[slot];
            auto first = accepted.shapes.begin() + body.meta.x;
            auto last = first + body.meta.y;
            if (edit != edits.end() && edit->bodyIndex == slot) {
                body = edit->body;
                body.meta.y = static_cast<uint32_t>(edit->leaves->size());
                first = edit->leaves->begin();
                last = edit->leaves->end();
                ++edit;
            }
            body.meta.x = static_cast<uint32_t>(replacement.shapes.size());
            replacement.shapes.insert(replacement.shapes.end(), first, last);
        }
        replacement.joints = accepted.joints;
        std::vector<uint32_t> capacities;
        for (const auto &world : accepted.worlds)
            capacities.push_back(world.bodies.y);
        JoltGpuBuildWorldRanges(replacement, capacities);
        return replacement;
    }

    std::unique_ptr<JoltGpuModelUpdate> PrepareActorChanges(ActorChanges &changes)
    {
        changes.deferred = false;
        const auto &accepted = storage.GetModel();
        std::vector<ModelEdit> edits;
        std::vector<BodyInitialization> initial;
        std::vector<CenterOfMassUpdate> rebases;
        std::vector<rhi::ComputeBufferUpdate> uploads;
        std::vector<rhi::ComputeDispatchDesc> dispatches;
        edits.reserve(changes.actors.size());
        initial.reserve(changes.actors.size());
        uploads.reserve(changes.actors.size());
        bool repack = false;
        for (const auto &change : changes.actors) {
            RequireOwnership(change->slot, change->bodyId);
            const uint32_t packed = Packed(change->slot);
            const auto &slot = slots[change->slot];
            change->identity->slot = packed;
            change->model.meta.w = slot.partition->row;
            for (auto &leaf : change->leaves)
                leaf.meta.y = packed;
            const auto &previous = slot.accepted;
            change->identity->shapeRevision =
                change->publication.sharedGeometryRevision.value_or(previous ? previous->shapeRevision + 1 : 0);
            const auto &old = accepted.bodies[packed];
            if (previous) {
                const auto a = previous->rootCenterOfMass;
                const auto b = change->identity->rootCenterOfMass;
                rebases.push_back({{packed, 0, 0, 0}, {b.x - a.x, b.y - a.y, b.z - a.z, 0.0f}});
                const auto different = [](const Float4 &lhs, const Float4 &rhs) {
                    return lhs.x != rhs.x || lhs.y != rhs.y || lhs.z != rhs.z || lhs.w != rhs.w;
                };
                const auto &next = change->model;
                if (old.mass.x != next.mass.x || old.mass.y != next.mass.y || old.mass.z != next.mass.z ||
                    different(old.invInertiaDiagonal, next.invInertiaDiagonal) ||
                    different(old.inertiaRotation, next.inertiaRotation)) {
                    auto warm = joints.PrepareMobilityReset(storage, packed, changes.deferred);
                    if (changes.deferred)
                        return {};
                    for (auto &upload : warm.updates)
                        uploads.push_back(std::move(upload));
                    for (auto &dispatch : warm.dispatches)
                        dispatches.push_back(std::move(dispatch));
                }
            } else
                initial.push_back({{packed, 0, 0, 0}, change->initialState, {}});
            std::vector<uint8_t> sleepAuthor(sizeof(change->sleep));
            std::memcpy(sleepAuthor.data(), &change->sleep, sizeof(change->sleep));
            uploads.push_back({storage.Get(Resource::SleepStates), uint64_t(packed) * sizeof(BodySleepRecord),
                               std::move(sleepAuthor)});
            repack |= old.meta.y != change->leaves.size();
            edits.push_back({packed, change->model, &change->leaves});
        }
        if (repack) {
            std::sort(edits.begin(), edits.end(),
                      [](const auto &a, const auto &b) { return a.bodyIndex < b.bodyIndex; });
            return mutations.PrepareReplacement(storage, RepackModel(edits), initial, rebases, std::move(uploads),
                                                std::move(dispatches));
        }

        std::vector<JoltGpuBodyModelUpdate> bodies;
        std::vector<JoltGpuShapeModelUpdate> leaves;
        bodies.reserve(edits.size());
        for (const auto &edit : edits) {
            auto body = edit.body;
            body.meta.x = accepted.bodies[edit.bodyIndex].meta.x;
            bodies.push_back({edit.bodyIndex, body});
            for (uint32_t leaf = 0; leaf < edit.leaves->size(); ++leaf)
                leaves.push_back({body.meta.x + leaf, (*edit.leaves)[leaf]});
        }
        return mutations.Prepare(storage, std::move(bodies), std::move(leaves), initial, rebases, std::move(uploads),
                                 std::move(dispatches));
    }

    void BindActorChanges(JoltGpuControlGroup &group, const std::shared_ptr<ActorChanges> &changes,
                          Partition *creationWorld = nullptr)
    {
        group.modelDeferred = [changes] { return changes->deferred; };
        group.prepareModel = [this, changes, creationWorld] {
            if (creationWorld) {
                // Initialization starts each new body awake with a fresh sleep
                // history. Only earlier dynamics can need a creation wake.
                const auto &accepted = storage.GetModel();
                const auto &range = accepted.worlds[creationWorld->row].bodies;
                const auto first = accepted.bodies.begin() + range.x;
                changes->wakeExisting =
                    std::any_of(first, first + range.y, [](const auto &body) { return body.meta.z == MotionDynamic; });
            }
            return PrepareActorChanges(*changes);
        };
        group.modelAccepted = [this, changes]() noexcept {
            publicationDirectory.reset();
            for (const auto &change : changes->actors)
                CommitActor(change);
        };
    }

    void QueueActorCreation(Partition &domain, std::shared_ptr<ActorChange> change, JoltGpuControlGroup group)
    {
        auto batch = domain.trailingCreations;
        // A layout, body command, joint or edit between creations closes the
        // batch. Only independent new slots share a model acceptance boundary.
        if (batch && domain.pending.size() == domain.trailingCreationGroup &&
            batch->actors.size() < JoltGpuPoseWake::BodiesPerDispatch) {
            if (batch->actors.size() == batch->actors.capacity())
                batch->actors.reserve(batch->actors.size() * 2u);
            auto &batchedCommands = domain.pending.back().commands;
            batchedCommands.insert(batchedCommands.end(), group.commands.begin(), group.commands.end());
            batch->actors.push_back(std::move(change));
            return;
        }
        batch = std::make_shared<ActorChanges>();
        batch->actors.push_back(std::move(change));
        BindActorChanges(group, batch, &domain);
        group.prepareWork = [this, &domain, batch, validation = PendingBoundsValidation(domain)] {
            if (!batch->wakeExisting)
                return JoltGpuControlWork{};
            std::vector<uint32_t> sources;
            sources.reserve(batch->actors.size());
            for (const auto &actor : batch->actors)
                sources.push_back(actor->identity->slot);
            return programs.GetSleep().PrepareSourceBoundsWake(storage, SleepSelection(domain), std::move(sources),
                                                               validation);
        };
        domain.pending.push_back(std::move(group));
        domain.trailingCreations = std::move(batch);
        domain.trailingCreationGroup = domain.pending.size();
    }

    void StageActor(uint32_t id, PreparedGpuActor actor, bool resetMotion)
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
        identity->physicalModel = actor.body;
        identity->actor = slot.actor;
        identity->rigidbody = actor.publication.rigidbody;
        identity->rigidbodyComponentId = actor.publication.rigidbodyComponentId;
        if (actors.IsValid(identity->rigidbody)) {
            const auto *body = actors.GetRigidbody(identity->rigidbody).owner;
            if (body && body->GetComponentID() == identity->rigidbodyComponentId)
                identity->bodyTarget = PhysicsTargetReference(body);
        }
        identity->owner = slot.owner;
        identity->rootCenterOfMass = rootCOM;
        identity->rootScale = actor.rootScale;
        identity->authoredShape = std::move(actor.authoredShape);
        identity->leaves.reserve(actor.geometry.leaves.size());
        for (const auto &leaf : actor.geometry.leaves) {
            if (leaf.memberIndex >= actor.members.size())
                throw std::invalid_argument("Jolt GPU leaf has no contributing Collider identity");
            const auto &surface = actor.members[leaf.memberIndex];
            PublishedLeaf publishedLeaf;
            publishedLeaf.surfaceTarget = surface.target;
            publishedLeaf.colliderId = surface.colliderId;
            publishedLeaf.colliderGeneration = surface.generation;
            publishedLeaf.surfaceGameObjectId = surface.gameObjectId;
            publishedLeaf.subShapeId = leaf.subShapeID.GetValue();
            publishedLeaf.subShapeBits = leaf.subShapeBits;
            publishedLeaf.memberIndex = leaf.memberIndex;
            publishedLeaf.shape = leaf.shape;
            publishedLeaf.geometry = geometry->Intern(leaf.geometryRecord);
            publishedLeaf.sensor = (leaf.model.meta.w & ShapeSensor) != 0;
            identity->leaves.push_back(std::move(publishedLeaf));
        }
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
            const auto &cold = identity->leaves[ordinal].geometry;
            leaf.model.geometryReference = cold ? cold->reference : UInt4{};
            const auto colliderId = identity->leaves[ordinal].colliderId;
            leaf.model.source.x = static_cast<uint32_t>(colliderId);
            leaf.model.source.y = static_cast<uint32_t>(colliderId >> 32);
            const auto colliderGeneration = identity->leaves[ordinal].colliderGeneration;
            leaf.model.source.z = static_cast<uint32_t>(colliderGeneration);
            leaf.model.source.w = static_cast<uint32_t>(colliderGeneration >> 32);
            change->leaves.push_back(std::move(leaf.model));
        }
        change->initialState = actor.initialState;
        change->sleep = actor.sleep;
        change->publication = std::move(actor.publication);
        change->identity = std::move(identity);
        const auto motion = change->identity->acceptedMotionType;
        const bool motionChanged = !firstOwnership && slot.authored->acceptedMotionType != motion;
        change->initializesRigidbody =
            motion != MotionStatic && (firstOwnership || slot.initialRigidbody != change->publication.rigidbody);
        // Reserve before transferring the command group, so subsequent source
        // cancellation cannot fail after this group has acquired ownership.
        pendingCompletions.reserve(pendingCompletions.size() + 2);

        JoltGpuControlGroup group;
        if (resetMotion && !firstOwnership) {
            BodyCommand reset{};
            reset.meta = {id, CommandResetMotion, 0, 0};
            group.commands.push_back(reset);
        }
        if (motionChanged) {
            BodyCommand boundary{};
            boundary.meta = {id, CommandMotionBoundary, MotionBoundaryTypeChange | CarryFlags(slot.queuedTarget), 0};
            group.commands.push_back(boundary);
        }
        if (!firstOwnership) {
            BodyCommand project{};
            project.meta = {id, CommandMotionBoundary, MotionBoundaryProjectMobility, 0};
            group.commands.push_back(project);
        }
        if (motion != MotionStatic) {
            // A new slot clamps its initial values. A new Rigidbody attached
            // to an existing actor applies only that component's captured
            // pending inputs, preserving the resident numerical pose/state.
            if (firstOwnership || (change->initializesRigidbody && change->publication.linearVelocity)) {
                BodyCommand linear{};
                linear.meta = {id, CommandSetLinearVelocity, 0, 0};
                linear.value = change->initialState.linearVelocity;
                group.commands.push_back(linear);
            }
            if (firstOwnership || (change->initializesRigidbody && change->publication.angularVelocity)) {
                BodyCommand angular{};
                angular.meta = {id, CommandSetAngularVelocity, 0, 0};
                angular.value = change->initialState.angularVelocity;
                group.commands.push_back(angular);
            }
        }
        if (motion != MotionStatic && !firstOwnership) {
            BodyCommand wake{};
            wake.meta = {id, CommandWake, 0u, 0u};
            group.commands.push_back(wake);
        }
        if (firstOwnership) {
            QueueActorCreation(domain, change, std::move(group));
        } else {
            auto oldBounds = QueueOldBounds(domain, index, id);
            auto changes = std::make_shared<ActorChanges>();
            changes->actors.push_back(change);
            BindActorChanges(group, changes);
            pending.push_back(std::move(group));
            QueueBoundsAfter(domain, std::move(oldBounds), index, id, true);
        }
        const bool setsInitialVelocity =
            change->initializesRigidbody && (change->publication.linearVelocity || change->publication.angularVelocity);
        if (resetMotion || motionChanged || (motion != MotionStatic && firstOwnership) || setsInitialVelocity)
            CancelTarget(slot.queuedTarget, pendingCompletions);
        if (motion != MotionStatic)
            slot.initialRigidbody = change->publication.rigidbody;
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
        group.modelAccepted = [this]() noexcept { publicationDirectory.reset(); };
        if (resident && !preserveVelocity) {
            BodyCommand wake{};
            wake.meta = {id, CommandWake, 0u, 0u};
            group.commands.push_back(wake);
        }
        if (!resident) {
            pendingCompletions.reserve(pendingCompletions.size() + 2);
            BodyCommand boundary{};
            boundary.meta = {id, CommandMotionBoundary,
                             preserveVelocity ? MotionBoundarySuspendPreserve : MotionBoundarySuspendReset, 0};
            group.commands.push_back(boundary);
        }
        auto oldBounds = QueueOldBounds(domain, index, id);
        pending.push_back(std::move(group));
        QueueBoundsAfter(domain, std::move(oldBounds), index, id, resident);
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
                   kind == CommandResetMotion || kind == CommandSleep || kind == CommandMotionBoundary ||
                   ((kind == CommandSetLinearVelocity || kind == CommandSetAngularVelocity) &&
                    motion != MotionStatic)) {
            CancelTarget(target, pendingCompletions);
        }
        return source;
    }

    struct BodyTransferCapture
    {
        JoltBodyImport target;
        uint32_t record;
        TargetState drive;
    };

    struct TransferReadback
    {
        std::vector<BodyTransferCapture> bodies;
        std::vector<JoltJointImport> joints;
    };

    JoltGpuControlWork PrepareTransferReads(std::shared_ptr<const TransferReadback> transfer, size_t firstBody,
                                            size_t firstJoint)
    {
        if (firstBody == transfer->bodies.size() && firstJoint == transfer->joints.size())
            return {};
        constexpr uint64_t bytesPerBody =
            sizeof(BodyState) + sizeof(BodyForces) + sizeof(KinematicTarget) + sizeof(UInt4);
        const auto available = work.ResourceRemaining(FrameWorkResource::Transfer);
        const auto budget = available ? available : work.ResourceState(FrameWorkResource::Transfer).limit;
        const auto bodyCount = static_cast<size_t>(
            std::min<uint64_t>(transfer->bodies.size() - firstBody, std::max(uint64_t{1}, budget / bytesPerBody)));
        struct Source
        {
            Float4 root;
            uint64_t revision;
            bool resident;
        };
        std::vector<Source> sources;
        std::vector<PublishedJoint> capturedJoints;
        std::vector<rhi::ComputeBufferRead> reads;
        sources.reserve(bodyCount);
        reads.reserve(4 * bodyCount);
        for (size_t index = firstBody; index < firstBody + bodyCount; ++index) {
            const auto &body = transfer->bodies[index];
            RequireOwnership(body.record, body.target.id);
            const auto slot = Packed(body.record);
            const auto &accepted = *slots[body.record].accepted;
            sources.push_back({accepted.rootCenterOfMass, accepted.shapeRevision,
                               (storage.GetModel().bodies[slot].meta.z & MotionNotResident) == 0});
            reads.push_back({storage.Get(Resource::BodyStates), uint64_t(slot) * sizeof(BodyState), sizeof(BodyState)});
            reads.push_back(
                {storage.Get(Resource::BodyForces), uint64_t(slot) * sizeof(BodyForces), sizeof(BodyForces)});
            reads.push_back({storage.Get(Resource::KinematicTargets), uint64_t(slot) * sizeof(KinematicTarget),
                             sizeof(KinematicTarget)});
            reads.push_back({storage.Get(Resource::SleepStates),
                             uint64_t(slot) * sizeof(BodySleepRecord) + offsetof(BodySleepRecord, runtime) +
                                 offsetof(BodySleepRuntime, state),
                             sizeof(UInt4)});
        }
        auto packetBytes = bodyCount * bytesPerBody;
        for (size_t index = firstJoint; index < transfer->joints.size(); ++index) {
            const auto &joint = AcceptedJoint(transfer->joints[index].id);
            const auto bytes = uint64_t(JoltGpuJointLayout::StateWords(joint.kind)) * 16;
            if (!reads.empty() && packetBytes + bytes > budget)
                break;
            const auto offset = uint64_t(storage.GetJointLayout()->Row(joint.row).layout.y) * 16;
            reads.push_back({storage.Get(Resource::Joints), offset, bytes});
            capturedJoints.push_back(joint);
            packetBytes += bytes;
        }
        auto host = storage.GetHost();
        auto packet = rhi::PrepareComputeBatch(host, {}, {}, std::move(reads));
        return {packet, [this, packet, transfer = std::move(transfer), firstBody, firstJoint,
                         sources = std::move(sources), capturedJoints = std::move(capturedJoints)] {
                    for (size_t index = 0; index < sources.size(); ++index) {
                        const auto &body = transfer->bodies[firstBody + index];
                        const auto &source = sources[index];
                        BodyState state;
                        BodyForces forces;
                        KinematicTarget move;
                        UInt4 sleep;
                        packet->Readback(4 * index)->ReadInto(&state, sizeof(state));
                        packet->Readback(4 * index + 1)->ReadInto(&forces, sizeof(forces));
                        packet->Readback(4 * index + 2)->ReadInto(&move, sizeof(move));
                        packet->Readback(4 * index + 3)->ReadInto(&sleep, sizeof(sleep));
                        const auto rotation = TransferRotation(state.rotation);
                        JoltBodyTransfer value{JPH::RVec3(TransferVector(state.positionCOM)) -
                                                   JPH::RVec3(rotation * TransferVector(source.root)),
                                               rotation,
                                               TransferVector(state.linearVelocity),
                                               TransferVector(state.angularVelocity),
                                               TransferVector(forces.force),
                                               TransferVector(forces.torque),
                                               {},
                                               source.revision,
                                               source.resident,
                                               sleep.x != 0};
                        const bool pending = (move.state.x & TargetPending) != 0;
                        const bool ownsVelocity = (move.state.x & (TargetActive | TargetDrivenLastTick)) != 0;
                        if (pending || ownsVelocity)
                            value.drive = JoltBodyTransfer::Drive{JPH::RVec3(TransferVector(move.positionSpeed)),
                                                                  TransferRotation(move.rotation),
                                                                  move.rootCOMDt.w,
                                                                  move.positionSpeed.w,
                                                                  body.drive.position,
                                                                  body.drive.rotation,
                                                                  pending,
                                                                  ownsVelocity,
                                                                  (move.state.x & TargetTemporaryStatic) != 0};
                        *body.target.state = std::move(value);
                    }
                    for (size_t index = 0; index < capturedJoints.size(); ++index) {
                        const auto &joint = capturedJoints[index];
                        *transfer->joints[firstJoint + index].state = WithJointState(joint.kind, [&](auto state) {
                            packet->Readback(4 * sources.size() + index)->ReadInto(&state, sizeof(state));
                            return DecodeJointTransfer(state, joint.author);
                        });
                    }
                    return PrepareTransferReads(transfer, firstBody + sources.size(),
                                                firstJoint + capturedJoints.size());
                }};
    }

    void QueueTransferReads(Partition &domain, const std::vector<JoltBodyImport> &bodies,
                            std::vector<JoltJointImport> jointImports)
    {
        auto transfer = std::make_shared<TransferReadback>();
        transfer->joints = std::move(jointImports);
        transfer->bodies.reserve(bodies.size());
        for (const auto &body : bodies) {
            const auto record = RequireSlot(body.id);
            transfer->bodies.push_back({body, record, slots[record].queuedTarget});
        }
        JoltGpuControlGroup group;
        group.prepareWork = [this, transfer] { return PrepareTransferReads(transfer, 0, 0); };
        domain.pending.push_back(std::move(group));
    }

    void QueueTransferRestore(uint32_t id, const JoltBodyTransfer &value)
    {
        auto &domain = BodyDomain(id);
        const auto record = RequireSlot(id);
        auto &slot = slots[record];
        JoltGpuControlGroup group;
        if (!value.awake) {
            BodyCommand signaledSleep{};
            signaledSleep.meta = {id, CommandSleep, 0, 0};
            group.commands.push_back(signaledSleep);
        }
        const BodyForces forces{TransferVector(value.force), TransferVector(value.torque)};
        KinematicTarget target{};
        if (value.drive) {
            const auto &drive = *value.drive;
            const auto root = slot.authored->rootCenterOfMass;
            target.positionSpeed = TransferVector(JPH::Vec3(drive.position), drive.maxSpeed);
            target.rotation = {drive.rotation.GetX(), drive.rotation.GetY(), drive.rotation.GetZ(),
                               drive.rotation.GetW()};
            target.rootCOMDt = {root.x, root.y, root.z, drive.duration};
            target.state.x = (drive.pending        ? TargetPending
                              : drive.ownsVelocity ? TargetDrivenLastTick
                                                   : 0u) |
                             (drive.temporaryStatic ? TargetTemporaryStatic : 0u);
        }
        const bool hasForces = value.force != JPH::Vec3::sZero() || value.torque != JPH::Vec3::sZero();
        if (hasForces || value.drive) {
            group.prepareWork = [this, id, record, forces, target, hasForces, hasDrive = bool(value.drive)] {
                RequireOwnership(record, id);
                const auto packed = Packed(record);
                std::vector<rhi::ComputeBufferUpdate> writes;
                if (hasForces)
                    writes.push_back(TransferUpload(storage, Resource::BodyForces, packed, forces));
                if (hasDrive)
                    writes.push_back(TransferUpload(storage, Resource::KinematicTargets, packed, target));
                auto host = storage.GetHost();
                return JoltGpuControlWork{rhi::PrepareComputeBatch(host, std::move(writes), {}, {})};
            };
        }
        if (!group.commands.empty() || group.prepareWork)
            domain.pending.push_back(std::move(group));
        if (value.drive) {
            const auto &drive = *value.drive;
            slot.queuedTarget = {drive.pending, drive.temporaryStatic, drive.positionSource, drive.rotationSource};
        }
    }

    void StageRetireBodies(Partition &domain, const std::vector<uint32_t> &ids, bool transferDrive,
                           std::vector<std::pair<uint32_t, uint32_t>> wakeExclusions = {})
    {
        auto records = std::make_shared<std::vector<std::pair<uint32_t, uint32_t>>>();
        records->reserve(ids.size());
        for (const auto id : ids)
            records->emplace_back(id, RequireSlot(id));
        // Resolve IDs while the source slots still accept owner lookups.
        for (auto &[source, target] : wakeExclusions) {
            source = RequireSlot(source);
            target = RequireSlot(target);
        }
        DestroyJointsForBodies(domain, ids);

        // The source bounds are still available locally. Wake from those exact
        // shapes before withdrawing the group, after any state export.
        JoltGpuControlGroup wake;
        wake.prepareWork = [this, &domain, records, wakeExclusions = std::move(wakeExclusions),
                            validation = PendingBoundsValidation(domain)] {
            std::vector<uint32_t> sources;
            sources.reserve(records->size());
            for (const auto &[id, record] : *records) {
                RequireOwnership(record, id);
                sources.push_back(Packed(record));
            }
            std::sort(sources.begin(), sources.end());
            const auto selection = SleepSelection(domain);
            const auto &models = storage.GetModel().bodies;
            for (uint32_t index = selection.selection.y; index < selection.selection.y + selection.selectedCounts.x;
                 ++index)
                if (models[index].meta.z == MotionDynamic &&
                    !std::binary_search(sources.begin(), sources.end(), index)) {
                    std::vector<std::pair<uint32_t, uint32_t>> packedExclusions;
                    packedExclusions.reserve(wakeExclusions.size());
                    for (const auto &[source, target] : wakeExclusions)
                        packedExclusions.emplace_back(Packed(source), Packed(target));
                    return programs.GetSleep().PrepareSourceBoundsWake(storage, selection, std::move(sources),
                                                                       validation, std::move(packedExclusions));
                }
            return JoltGpuControlWork{};
        };
        domain.pending.push_back(std::move(wake));
        JoltGpuControlGroup group;
        group.prepareModel = [this, &domain, records] {
            const std::vector<ShapeModel> empty;
            std::vector<ModelEdit> replacements;
            std::vector<BodyInitialization> initial;
            replacements.reserve(records->size());
            initial.reserve(records->size());
            bool hasLeaves = false;
            for (const auto &[id, record] : *records) {
                RequireOwnership(record, id);
                const auto packed = Packed(record);
                const auto &body = storage.GetModel().bodies[packed];
                auto tombstone = Tombstone();
                tombstone.meta.x = body.meta.x;
                tombstone.meta.w = domain.row;
                hasLeaves |= body.meta.y != 0;
                replacements.push_back({packed, tombstone, &empty});
                initial.push_back({{packed, 0, 0, 0}, EmptyState(), {}});
            }
            if (hasLeaves) {
                std::sort(replacements.begin(), replacements.end(),
                          [](const auto &a, const auto &b) { return a.bodyIndex < b.bodyIndex; });
                return mutations.PrepareReplacement(storage, RepackModel(replacements), initial);
            }
            std::vector<JoltGpuBodyModelUpdate> bodies;
            bodies.reserve(replacements.size());
            for (const auto &value : replacements)
                bodies.push_back({value.bodyIndex, value.body});
            return mutations.Prepare(storage, std::move(bodies), {}, initial);
        };
        group.modelAccepted = [this, records]() noexcept {
            for (const auto &[id, record] : *records) {
                auto &slot = slots[record];
                slot.partition->records[slot.local] = UINT32_MAX;
                slot.partition->freeLocals.push_back(slot.local);
                byId.erase(id);
                slot = Slot{};
                freeRecords.push_back(record);
            }
            publicationDirectory.reset();
        };
        auto &completions = domain.pendingCompletions;
        completions.reserve(completions.size() + records->size() * 2);
        domain.pending.push_back(std::move(group));
        for (const auto &[id, record] : *records) {
            auto &slot = slots[record];
            if (transferDrive)
                slot.queuedTarget = {};
            else
                CancelTarget(slot.queuedTarget, completions);
            slot.retiring = true;
        }
        domain.liveCount -= static_cast<uint32_t>(records->size());
    }

    void StageRetire(uint32_t id, bool transferDrive = false)
    {
        StageRetireBodies(BodyDomain(id), {id}, transferDrive);
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

    std::shared_ptr<const PublicationDirectory> CapturePublicationDirectory()
    {
        if (publicationDirectory)
            return publicationDirectory;
        auto directory = std::make_shared<PublicationDirectory>();
        directory->actors.resize(storage.GetBodyCount());
        directory->bodySlots.reserve(byId.size());
        directory->residentBodies.resize(storage.GetBodyCount());
        directory->leaves.resize(storage.GetShapeCount());
        directory->worldJoints.resize(storage.GetWorldCount());
        for (const auto &[id, index] : byId) {
            const auto &slot = slots[index];
            if (!slot.accepted)
                continue;
            const auto packed = Packed(index);
            directory->actors[packed] = slot.accepted;
            directory->bodySlots.emplace(id, packed);
            const auto &model = storage.GetModel().bodies[packed];
            directory->residentBodies[packed] = (model.meta.z & MotionNotResident) == 0u;
            if (slot.accepted->leaves.size() != model.meta.y)
                throw std::logic_error("GPU publication actor directory differs from accepted leaf layout");
            for (uint32_t leaf = 0; leaf < model.meta.y; ++leaf)
                directory->leaves[model.meta.x + leaf] = {packed, leaf};
        }
        for (const auto &joint : acceptedJoints) {
            auto &local = directory->worldJoints.at(joint.author.identity.z);
            auto value = joint;
            value.row = static_cast<uint32_t>(local.size());
            local.push_back(std::move(value));
        }
        publicationDirectory = std::move(directory);
        return publicationDirectory;
    }

    JoltGpuTickControl PrepareControl(Partition &domain, uint64_t epoch, uint64_t step, bool controlOnly,
                                      Consumer consumer, Consumer finalConsumer, ContactObserverFactory prepareContacts,
                                      bool cleanMember = false)
    {
        auto publication = std::make_shared<Publication>();
        auto pairCapture = std::make_shared<JoltGpuColliderPairCapture>();
        auto articulationCapture = std::make_shared<JoltGpuArticulationCapture>();
        publication->worldId = domain.id;
        publication->worldRow = domain.row;
        publication->simulationEpoch = epoch;
        publication->fixedStepIndex = step;
        publication->isFixedStep = !controlOnly;
        publication->geometryStore = geometry;
        auto reports = prepareContacts ? prepareContacts(publication) : ContactObserver{};
        auto prefix = std::make_shared<std::vector<PrefixTarget>>();
        for (const auto index : domain.records)
            if (index != UINT32_MAX)
                prefix->push_back({slots[index].bodyId, slots[index].queuedTarget});
        auto completions = std::make_shared<std::vector<TargetCompletion>>(domain.pendingCompletions);
        completions->reserve(completions->size() + 2u * prefix->size());
        const bool acceptedStateIntent = std::any_of(domain.pending.begin(), domain.pending.end(), [](const auto &g) {
            return bool(g.prepareModel) || !g.commands.empty();
        });
        JoltGpuControlGroup capture;
        capture.prepareModel = [this, &domain, publication, prefix, completions,
                                controlOnly]() -> std::unique_ptr<JoltGpuModelUpdate> {
            publication->layoutGeneration = layoutGeneration;
            publication->bodyModels = storage.Get(Resource::BodyModels)->Freeze();
            publication->shapeModels = storage.Get(Resource::ShapeModels)->Freeze();
            publication->worlds = storage.Get(Resource::Worlds)->Freeze();
            publication->directory = CapturePublicationDirectory();
            const auto &selected = storage.GetModel().worlds.at(domain.row);
            publication->bodyBase = selected.bodies.x;
            publication->bodyCount = selected.bodies.y;
            publication->leafBase = selected.leaves.x;
            publication->leafCount = selected.leaves.y;
            publication->jointBase = selected.leaves.z;
            publication->jointCount = selected.leaves.w;
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
        capture.prepareWork = [this, &domain, publication, pairCapture, articulationCapture, acceptedStateIntent] {
            JoltGpuGeometryHandles records;
            for (uint32_t slot = publication->bodyBase; slot < uint64_t(publication->bodyBase) + publication->bodyCount;
                 ++slot)
                if (const auto &actor = publication->directory->actors.at(slot))
                    for (const auto &leaf : actor->leaves) {
                        if (leaf.geometry)
                            records.push_back(leaf.geometry);
                        publication->hasGenericGeometry |=
                            leaf.shape && (leaf.shape->GetSubType() == JPH::EShapeSubType::Cylinder ||
                                           leaf.shape->GetSubType() == JPH::EShapeSubType::ConvexHull);
                    }
            return ContinueJoltGpuControlWork(
                geometry->PrepareCapture(std::move(records),
                                         [this, publication](auto lease) {
                                             publication->geometry = lease;
                                             storage.SetGeometry(std::move(lease), publication->hasGenericGeometry);
                                         }),
                [this, &domain, publication, pairCapture, articulationCapture, acceptedStateIntent] {
                    return ContinueJoltGpuControlWork(
                        PrepareColliderPairs(domain, publication, pairCapture),
                        [this, &domain, publication, articulationCapture, acceptedStateIntent] {
                            articulationCapture->preparing = true;
                            return programs.GetArticulations().PrepareAccepted(
                                storage, domain.row, publication->directory, domain.articulationCache,
                                [&domain, articulationCapture](auto state) {
                                    articulationCapture->preparing = false;
                                    domain.articulationCache = state;
                                    articulationCapture->value = std::move(state);
                                },
                                acceptedStateIntent);
                        });
                });
        };
        if (cleanMember) {
            // Eligibility proved the directory, pair table and geometry are already accepted.
            publication->geometry = domain.published->geometry;
            publication->hasGenericGeometry = domain.published->hasGenericGeometry;
            publication->colliderPairs = domain.pairCache;
            pairCapture->value = domain.pairCache;
            articulationCapture->value = domain.articulationCache;
            capture.prepareWork = {};
        }
        domain.pending.push_back(std::move(capture));
        for (auto &group : domain.pending) {
            if (group.commands.empty())
                continue;
            const bool poses = std::any_of(group.commands.begin(), group.commands.end(),
                                           [](const BodyCommand &command) { return command.meta.y <= CommandSetPose; });
            group.prepareCommands = [this, &domain, validation = poses ? PendingBoundsValidation(domain) : nullptr](
                                        std::vector<BodyCommand> values) {
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
                auto poseWake = validation
                                    ? programs.GetSleep().PreparePoseWake(storage, SleepSelection(domain), *validation)
                                    : JoltGpuPoseWake{};
                return commands.Prepare(storage, std::move(values), std::move(poseWake));
            };
        }
        JoltGpuTickControl control;
        std::vector<std::pair<uint32_t, uint32_t>> prescribedSources;
        for (const auto record : domain.records) {
            if (record == UINT32_MAX)
                continue;
            const auto &slot = slots[record];
            if (slot.retiring || !slot.authored || !slot.desiredResident)
                continue;
            if (slot.authored->acceptedMotionType == MotionKinematic ||
                (slot.authored->acceptedMotionType == MotionStatic && slot.queuedTarget.temporaryStatic))
                prescribedSources.emplace_back(record, slot.bodyId);
        }
        control.prepareSleepSources = [this, prescribedSources = std::move(prescribedSources)] {
            std::vector<uint32_t> sources;
            sources.reserve(prescribedSources.size());
            for (const auto &[record, id] : prescribedSources) {
                RequireOwnership(record, id);
                sources.push_back(Packed(record));
            }
            return sources;
        };
        control.colliderPairs = pairCapture;
        control.articulations = articulationCapture;
        control.commandProgram = &commands;
        if (cleanMember) {
            control.acceptedGeometry = publication->geometry;
            control.acceptedGenericGeometry = publication->hasGenericGeometry;
        }
        // Retained reduced representation/history must still take its original
        // conversion path even after the last Rigidbody migrated away.
        const auto &rigidCache = domain.articulationCache;
        control.simulateRigidBodies =
            reports.simulateRigidBodies || (rigidCache && (rigidCache->numeric || !rigidCache->trees.empty()));
        control.prepareContactReport = std::move(reports.phase);
        control.prepareCoupledContacts = std::move(reports.couple);
        control.groups = TakeControlGroups(domain);
        domain.pendingSources.reset();
        domain.pendingCompletions.clear();
        const std::shared_ptr<const Publication> retained = publication;
        control.prepareFinalWork = [this, publication, retained, consumer = std::move(consumer),
                                    finalConsumer = std::move(finalConsumer),
                                    reportFinish = std::move(reports.finish)](const JoltGpuTickOutput &output) {
            publication->worldRow = output.worldRow;
            publication->bodyBase = output.bodyBase;
            publication->bodyCount = output.bodyCount;
            publication->leafBase = output.leafBase;
            publication->leafCount = output.leafCount;
            publication->jointBase = output.jointBase;
            publication->jointCount = output.jointCount;
            publication->bodyStates = output.states;
            publication->jointStream = output.joints;
            publication->jointLayout = output.jointLayout;
            publication->geometryStore = output.geometryStore;
            publication->geometry = output.geometry;
            publication->hasGenericGeometry = output.hasGenericGeometry;
            JPH::GPU::StepParams selected{};
            selected.selection = {output.worldRow, output.bodyBase, output.leafBase, output.jointBase};
            selected.selectedCounts = {output.bodyCount, output.leafCount, output.jointCount, 0u};
            return ContinueJoltGpuControlWork(
                programs.GetSleep().PrepareSnapshot(
                    storage, selected,
                    [publication](auto snapshot) { publication->sleepStatus = std::move(snapshot); }),
                [this, retained, consumer, finalConsumer, reportFinish] {
                    return ContinueJoltGpuControlWork(joints.PrepareStatus(retained), [retained, consumer,
                                                                                       finalConsumer, reportFinish] {
                        return ContinueJoltGpuControlWork(
                            reportFinish ? reportFinish(retained) : JoltGpuControlWork{},
                            [retained, consumer, finalConsumer] {
                                return ContinueJoltGpuControlWork(
                                    consumer ? consumer(retained) : JoltGpuControlWork{}, [retained, finalConsumer] {
                                        return finalConsumer ? finalConsumer(retained) : JoltGpuControlWork{};
                                    });
                            });
                    });
                });
        };
        domain.candidate = std::move(publication);
        domain.candidateCompletions = std::move(completions);
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

    bool BlocksNumericalWindow(const Partition &domain) const
    {
        // Inactive authoring has no Play consumer. Its queued FIFO is retained, not GPU-ready maintenance.
        // Preparing/playing runtime Worlds, and every cleanup/retirement, still close clean refill.
        return domain.cleanup || domain.retiring ||
               (!domain.pending.empty() &&
                (domain.world != &SceneManager::Instance().AuthoringWorld() || domain.world->simulation.isPlaying));
    }

    bool MaintenancePending() const
    {
        return pressurePending || std::any_of(ordered.begin(), ordered.end(), [this](const auto *domain) {
                   return domain && BlocksNumericalWindow(*domain);
               });
    }

    bool CleanMember(const Partition &domain) const
    {
        const auto &published = domain.published;
        const auto &cache = domain.articulationCache;
        return domain.world->physics && !domain.world->physics->admittedParticipants && domain.pending.empty() &&
               domain.row < storage.GetModel().worlds.size() && storage.GetModel().worlds[domain.row].leaves.w == 0 &&
               (!cache || (!cache->numeric && cache->trees.empty())) && published &&
               published->layoutGeneration == layoutGeneration && published->geometry &&
               published->directory == publicationDirectory && domain.pairCache &&
               domain.pairCache->identities == domain.acceptedPairs;
    }

    template <typename FinalConsumerFactory>
    bool TryBegin(Partition &domain, uint64_t epoch, uint64_t step, float dt, const JoltGpuStepParameters *parameters,
                  Consumer consumer, FinalConsumerFactory prepareFinalConsumer,
                  ContactObserverFactory prepareContacts = {}, ParticipantFactory prepareParticipants = {})
    {
        Service();
        if (domain.cursor && domain.status == JoltGpuTick::Status::Running) {
            if (writer == &domain && writerKind == WriterKind::Work)
                return false;
            if (domain.memberEpoch != epoch || domain.memberStep != step)
                throw std::logic_error("GPU member retry changed its fixed identity");
            return true;
        }
        if (domain.closed || !domain.admission || tick.IsPressureActive())
            return false;
        const bool clean = parameters && CleanMember(domain) && !MaintenancePending();
        if ((writer || peerWriter) &&
            (!clean || (writer && !writer->shareable) || (peerWriter && !peerWriter->shareable)))
            return false;
        if (writer && peerWriter)
            return false;
        JoltGpuTick *selected = &tick;
        if (writer)
            selected = &tick.GetSharedMember([this] {
                if (peerWriter && failureObserver)
                    failureObserver(*peerWriter->world);
            });
        // Existing admission has excluded both numerical writers here for an
        // unclean/exclusive window. Surrender optional tables before factories
        // can prepare the next author/model/layout allocations.
        if (!clean)
            tick.ReleaseBodyBatchInputs();
        auto finalConsumer = prepareFinalConsumer();
        auto control = PrepareControl(domain, epoch, step, parameters == nullptr, std::move(consumer),
                                      std::move(finalConsumer), std::move(prepareContacts), clean);
        if (prepareParticipants)
            control.participants = prepareParticipants();
        if (selected == &tick)
            writer = &domain;
        else
            peerWriter = &domain;
        domain.cursor = selected;
        domain.shareable = clean;
        domain.memberEpoch = epoch;
        domain.memberStep = step;
        if (selected == &tick)
            writerKind = parameters ? WriterKind::Tick : WriterKind::Control;
        domain.status = JoltGpuTick::Status::Running;
        selected->SetAdmissionEnabled(true);
        try {
            if (parameters)
                selected->Begin(domain.row, dt, *parameters, std::move(control));
            else
                selected->BeginControl(domain.row, std::move(control));
        } catch (...) {
            Cancel(domain);
            throw;
        }
        return true;
    }

    void PublishWriter(Partition &domain)
    {
        if (domain.candidate) {
            domain.published = std::move(domain.candidate);
            if (&domain == writer && writerKind == WriterKind::Control)
                ++domain.controlSerial;
            else
                ++domain.tickSerial;
            if (domain.candidateCompletions)
                for (const auto &completion : *domain.candidateCompletions) {
                    const auto *author = completion.source.Get();
                    auto *body = author ? dynamic_cast<Rigidbody *>(author->rigidbody.GetComponent()) : nullptr;
                    if (body)
                        body->CompletePhysicsTarget(completion.source, completion.channels);
                }
        }
        domain.candidateCompletions.reset();
        domain.status = JoltGpuTick::Status::Complete;
        domain.cursor = nullptr;
        if (writer == &domain)
            writer = nullptr;
        else
            peerWriter = nullptr;
    }

    void Cancel(Partition &domain)
    {
        domain.closed = true;
        storage.ClearContactHistory(domain.row);
        if (observationOwner == &domain)
            observationCursor->Cancel();
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
                        (domain.candidateCompletions ? domain.candidateCompletions->size() : 0u));
        if (domain.candidateCompletions)
            targets.insert(targets.end(), domain.candidateCompletions->begin(), domain.candidateCompletions->end());
        domain.pending.clear();
        domain.trailingCreations.reset();
        domain.pendingBoundsValidation.reset();
        domain.queuedPairs.reset();
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
        if (domain.cursor) {
            domain.cursor->Cancel();
            if (writer == &domain)
                writerKind = WriterKind::Cancelled;
        }
        domain.candidate.reset();
        domain.candidateCompletions.reset();
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
        publicationDirectory.reset();
    }

    void ReleasePartition(Partition &domain) noexcept
    {
        // The selected range has been removed and all its receipts settled.
        // Frozen publications own their resources without this row or its slots.
        ReleaseRecords(domain);
        const auto id = domain.id;
        ordered[domain.row] = nullptr;
        freeRows.push_back(domain.row); // Capacity was reserved at registration.
        domains.erase(id);
        --retiringPartitions;
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
                     scene.storage.Get(Resource::BodyForces), scene.storage.Get(Resource::KinematicTargets),
                     scene.storage.Get(Resource::SleepStates)},
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
        if (observationOwner && !observationCallActive && observationCursor->CanRetire()) {
            observationOwner = nullptr;
            observationPublication.reset();
        }
        if (peerWriter) {
            auto &domain = *peerWriter;
            auto &cursor = *domain.cursor;
            if ((cursor.GetStatus() == JoltGpuTick::Status::Failed ||
                 cursor.GetStatus() == JoltGpuTick::Status::Cancelled) &&
                domain.status != JoltGpuTick::Status::Cancelled)
                Cancel(domain);
            if (cursor.IsComplete() && !domain.closed && domain.status != JoltGpuTick::Status::Cancelled)
                PublishWriter(domain);
            else if (cursor.CanRetire() && domain.status == JoltGpuTick::Status::Cancelled) {
                if (cursor.GetStatus() == JoltGpuTick::Status::Cancelled ||
                    cursor.GetStatus() == JoltGpuTick::Status::Failed)
                    cursor.ReopenForPartitionCleanup();
                domain.cursor = nullptr;
                peerWriter = nullptr;
            }
        }
        // An idle pressure sweep also owns this shared Tick source. It is not
        // a second Scene writer and cannot overlap a new layout/cleanup writer.
        if (tick.IsPressureActive())
            return;
        if (!writer && pressureDomain &&
            (tick.GetStatus() == JoltGpuTick::Status::Failed || tick.GetStatus() == JoltGpuTick::Status::Cancelled)) {
            // Idle retirement has the same fatal-owner/receipt cleanup path as
            // an ordinary writer; a failed source must not become unreopenable.
            writer = pressureDomain;
            writer->cursor = &tick;
            writerKind = WriterKind::Cancelled;
            Cancel(*writer);
        }
        if (writer &&
            (tick.GetStatus() == JoltGpuTick::Status::Failed || tick.GetStatus() == JoltGpuTick::Status::Cancelled) &&
            writerKind != WriterKind::Cancelled)
            Cancel(*writer);
        if (writer && writerKind == WriterKind::Cancelled) {
            if (!tick.CanRetire())
                return;
            if (tick.GetStatus() == JoltGpuTick::Status::Cancelled || tick.GetStatus() == JoltGpuTick::Status::Failed)
                tick.ReopenForPartitionCleanup();
            writer->cursor = nullptr;
            writer = nullptr;
            pressureDomain = nullptr;
        }
        if (writer && tick.IsComplete()) {
            if (writerKind == WriterKind::Cleanup) {
                // Complete follows every accepted cleanup receipt. Do not drop
                // the old numeric tail at the earlier modelAccepted callback.
                writer->articulationCache.reset();
                writer->cleanup = false;
                writer->status = JoltGpuTick::Status::Cancelled;
                writer->cursor = nullptr;
                writer = nullptr;
            } else if (writerKind == WriterKind::Retirement) {
                auto &retired = *writer;
                writer = nullptr;
                ReleasePartition(retired);
            } else
                PublishWriter(*writer);
        }
        if (writer || peerWriter || observationOwner)
            return;
        for (auto *domain : ordered) {
            if (!domain)
                continue;
            if (!domain->cleanup && !domain->retiring)
                continue;
            // No shared writer remains, and Cancel's accepted receipts passed
            // CanRetire above. Cleanup prevents reopen and removes every native
            // row of this World, so its old representation has no future reader.
            // Release before topology/reset allocations, not after they succeed.
            if (domain->cleanup)
                domain->articulationCache.reset();
            const auto &worlds = storage.GetModel().worlds;
            const bool noNumericalRange =
                domain->row >= worlds.size() || (!worlds[domain->row].bodies.y && !worlds[domain->row].leaves.w);
            if (noNumericalRange) {
                if (domain->retiring)
                    ReleasePartition(*domain);
                else {
                    ReleaseRecords(*domain);
                    domain->cleanup = false;
                }
                continue;
            }
            // Reopening a cancelled partition reuses its slots. Retirement
            // drops the range entirely, so a scrub of that range is redundant.
            if (domain->retiring)
                QueueLayout(*domain, 0u);
            else
                QueueCleanup(*domain);
            JoltGpuTickControl control;
            control.needsSnapshot = false;
            control.partitionCleanup = true;
            control.commandProgram = &commands;
            control.groups = TakeControlGroups(*domain);
            writer = domain;
            domain->cursor = &tick;
            domain->shareable = false;
            writerKind = domain->retiring ? WriterKind::Retirement : WriterKind::Cleanup;
            tick.SetAdmissionEnabled(true);
            tick.BeginControl(domain->row, std::move(control));
            break;
        }
    }

    PhysicsECSStore &actors;
    FrameWorkOwner &work;
    std::shared_ptr<JoltGpuGeometryStore> geometry;
    JoltGpuStorage storage;
    JoltGpuPrograms programs;
    JoltGpuBroadphase broadphase;
    JoltGpuNarrowphase narrowphase;
    JoltGpuConstraints constraints;
    JoltGpuBodyMutations mutations;
    JoltGpuBodyCommands commands;
    JoltGpuJoints &joints;
    std::map<uint64_t, JointRecord> authoredJoints;
    std::vector<PublishedJoint> acceptedJoints;
    uint64_t layoutGeneration = 0;
    std::shared_ptr<const PublicationDirectory> publicationDirectory;
    std::vector<Slot> slots;
    std::vector<uint32_t> freeRecords;
    std::unordered_map<uint32_t, uint32_t> byId;
    std::map<uint64_t, std::unique_ptr<Partition>> domains;
    // Stable row-address table, bounded by the concurrently registered high-water
    // mark. Null entries are retired zero-range rows available for later Worlds.
    std::vector<Partition *> ordered;
    std::vector<uint32_t> freeRows;
    Partition *writer = nullptr, *peerWriter = nullptr;
    RuntimeWorld *requiredWorld = nullptr;
    Partition *observationOwner = nullptr;
    JoltGpuTick *observationCursor = nullptr;
    std::shared_ptr<const Publication> observationPublication;
    bool observationCallActive = false;
    Partition *pressureDomain = nullptr;
    size_t retiringPartitions = 0;
    WriterKind writerKind = WriterKind::Control;
    uint64_t observedPressure = 0;
    size_t pressureCursor = 0;
    bool pressurePending = false;
    // Entire backend shutdown drains this one source; domain retirement never
    // destroys it. RuntimeWorld owners stay alive until their CanRetire result.
    FailureObserver failureObserver;
    JoltGpuTick tick;
};

JoltGpuScene::JoltGpuScene(rhi::ComputeHost &host, FrameWorkOwner &work, const ShaderTargetCode &stepTarget,
                           const ShaderTargetCode &mutationTarget, const ShaderTargetCode &commandTarget,
                           const ShaderTargetCode &broadphaseTarget, const ShaderTargetCode &narrowphaseTarget,
                           const ShaderTargetCode &constraintTarget, JoltGpuJoints &joints,
                           FailureObserver failureObserver, const ShaderTargetCode *captureTarget,
                           const ShaderTargetCode *dynamicsTarget)
    : m_impl(std::make_unique<Impl>(host, work, stepTarget, mutationTarget, commandTarget, broadphaseTarget,
                                    narrowphaseTarget, constraintTarget, joints, std::move(failureObserver),
                                    captureTarget, dynamicsTarget))
{
}
JoltGpuScene::~JoltGpuScene() = default;

bool JoltGpuScene::TryPrepareStartupPrograms()
{
    const bool forces = m_impl->programs.TryPrepareStartupPrograms();
    const bool pairs = m_impl->broadphase.TryPrepareSmallPairPipeline();
    return forces && pairs;
}

void JoltGpuScene::RequireParticipantBodies(RuntimeWorld &world,
                                            const std::shared_ptr<JoltGpuParticipantBodies> &bodies) const
{
    const auto &domain = m_impl->Domain(world);
    if (!bodies || bodies->m_worldId != world.id || !domain.cursor || domain.closed ||
        domain.status != JoltGpuTick::Status::Running || !domain.candidate || !domain.candidate->isFixedStep ||
        bodies->m_candidate.get() != domain.candidate.get())
        throw std::logic_error("GPU participant body map is outside its admitted Fixed");
}

JoltGpuControlWork JoltGpuScene::PrepareParticipantBodies(RuntimeWorld &world, const StepParams &selection,
                                                          std::vector<JoltGpuParticipantBodyTarget> targets,
                                                          uint32_t portRows,
                                                          std::shared_ptr<JoltGpuParticipantBodies> &out)
{
    auto &domain = m_impl->Domain(world);
    if (targets.empty() || !portRows || portRows > UINT32_MAX / 16u || targets.size() > portRows || !domain.cursor ||
        domain.closed || domain.status != JoltGpuTick::Status::Running || !domain.candidate ||
        !domain.candidate->isFixedStep || domain.row != selection.selection.x)
        throw std::invalid_argument("GPU participant bodies require a nonempty admitted Fixed selection");
    const auto &range = m_impl->storage.GetModel().worlds.at(domain.row).bodies;
    if (range.x != selection.selection.y || range.y != selection.selectedCounts.x ||
        targets.size() * 8ull > m_impl->storage.GetHost().device.GetCapabilities().portable.maxStorageBufferBinding)
        throw std::invalid_argument("GPU participant body map exceeds its accepted world or device range");
    auto bodies = std::shared_ptr<JoltGpuParticipantBodies>(new JoltGpuParticipantBodies());
    bodies->m_candidate = domain.candidate;
    bodies->m_worldId = world.id;
    bodies->m_worldRow = domain.row;
    bodies->m_portRows = portRows;
    out = bodies;
    struct Preparation : std::enable_shared_from_this<Preparation>
    {
        std::vector<JoltGpuParticipantBodyTarget> targets;
        std::unordered_set<uint32_t> slots, rows;
        size_t next = 0;
        std::function<uint32_t(const JoltGpuParticipantBodyTarget &)> resolve;
        std::function<JoltGpuControlWork()> finish;
        JoltGpuControlWork Next()
        {
            const size_t end = std::min(targets.size(), next + 256u);
            for (; next < end; ++next) {
                const auto &target = targets[next];
                // Resolve and reject duplicate destinations before device work is admitted.
                const uint32_t slot = resolve(target);
                if (!slots.insert(slot).second || !rows.insert(target.portRow).second)
                    throw std::invalid_argument("GPU participant targets and project rows must be distinct");
            }
            if (next == targets.size())
                return finish();
            return {nullptr, [self = shared_from_this()] { return self->Next(); },
                    JoltGpuControlWork::ResumeAfter::HostWork};
        }
    };
    auto prepare = std::make_shared<Preparation>();
    prepare->targets = std::move(targets);
    prepare->resolve = [this, &world, &domain, bodies](const JoltGpuParticipantBodyTarget &target) {
        RequireParticipantBodies(world, bodies);
        if (target.portRow >= bodies->m_portRows)
            throw std::invalid_argument("GPU participant project row exceeds its scalar port");
        const uint32_t recordIndex = m_impl->RequireSlot(target.bodyId);
        m_impl->RequireOwnership(recordIndex, target.bodyId);
        const auto &record = m_impl->slots[recordIndex];
        if (record.partition != &domain || !record.accepted || record.accepted->bodyId != target.bodyId ||
            record.accepted->rigidbodyComponentId == 0)
            throw std::invalid_argument("GPU participant target has no matching accepted Rigidbody identity");
        const uint32_t slot = m_impl->Packed(recordIndex);
        const auto &model = m_impl->storage.GetModel().bodies.at(slot);
        if (model.meta.w != domain.row || model.meta.z != MotionDynamic)
            throw std::invalid_argument("GPU participant target must be an accepted resident Dynamic GPU body");
        bodies->m_rows.push_back({slot, target.portRow});
        return slot;
    };
    prepare->finish = [this, &world, bodies] {
        RequireParticipantBodies(world, bodies);
        return m_impl->commands.PrepareParticipantBodies(m_impl->storage, bodies);
    };
    return prepare->Next();
}

JoltGpuControlWork JoltGpuScene::PrepareParticipantMotion(RuntimeWorld &world,
                                                          const std::shared_ptr<JoltGpuParticipantBodies> &bodies,
                                                          std::shared_ptr<rhi::ComputeBuffer> motion)
{
    RequireParticipantBodies(world, bodies);
    return m_impl->commands.PrepareParticipantMotion(bodies, std::move(motion));
}

JoltGpuControlWork JoltGpuScene::PrepareParticipantImpulses(RuntimeWorld &world,
                                                            const std::shared_ptr<JoltGpuParticipantBodies> &bodies,
                                                            std::shared_ptr<rhi::ComputeBuffer> impulses)
{
    RequireParticipantBodies(world, bodies);
    return m_impl->commands.PrepareParticipantImpulses(bodies, std::move(impulses));
}

bool JoltGpuScene::TryPrepareParticipantCpuResponse()
{
    return m_impl->commands.TryPrepareParticipantCpuResponse();
}

rhi::ComputeDispatchDesc JoltGpuScene::ParticipantCpuResponse(
    std::shared_ptr<rhi::ComputeBuffer> parameters, std::shared_ptr<rhi::ComputeBuffer> models,
    std::shared_ptr<rhi::ComputeBuffer> motion, std::shared_ptr<rhi::ComputeBuffer> impulses,
    std::shared_ptr<rhi::ComputeBuffer> journal, uint32_t count, uint32_t round)
{
    return m_impl->commands.ParticipantCpuResponse(std::move(parameters), std::move(models), std::move(motion),
                                                  std::move(impulses), std::move(journal), count, round);
}

const std::shared_ptr<JoltGpuGeometryStore> &JoltGpuScene::GetGeometryStore() const noexcept
{
    return m_impl->geometry;
}
bool JoltGpuScene::RegisterPartition(RuntimeWorld &world, uint32_t capacity)
{
    m_impl->Service();
    const auto found = m_impl->domains.find(world.id);
    if (found != m_impl->domains.end()) {
        auto &domain = m_impl->Domain(world);
        if (domain.retiring)
            return false;
        if (domain.cleanup || m_impl->writer == &domain || m_impl->peerWriter == &domain ||
            m_impl->observationOwner == &domain)
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
                                                   OwnerIdentity owner, bool initialResident,
                                                   uint32_t expectedAuthorBodyId)
{
    return m_impl->Reserve(world, id, actor, std::move(owner), initialResident, expectedAuthorBodyId);
}
void JoltGpuScene::ReserveCapacity(RuntimeWorld &world, uint32_t count)
{
    m_impl->ReserveCapacity(world, count);
}
void JoltGpuScene::StageActor(uint32_t id, PreparedGpuActor actor, bool resetMotion)
{
    m_impl->StageActor(id, std::move(actor), resetMotion);
}
void JoltGpuScene::SetResident(uint32_t id, bool value, bool preserve)
{
    m_impl->SetResident(id, value, preserve);
}
PhysicsTargetSource JoltGpuScene::QueueCommand(uint32_t id, BodyCommand value, const PhysicsTargetAuthor *author)
{
    return m_impl->QueueCommand(id, value, author);
}

void JoltGpuScene::QueuePoseDrive(RuntimeWorld &world, std::vector<PoseDriveIntent> intents)
{
    if (intents.empty())
        return;
    auto &domain = m_impl->Domain(world);
    if (domain.closed || domain.retiring)
        throw std::logic_error("Cannot submit ragdoll drive to a closed World");
    JoltGpuControlGroup wake;
    for (const auto &intent : intents) {
        auto &record = m_impl->authoredJoints.at(intent.id);
        if (record.value.kind != intent.kind || record.value.author.identity.z != domain.row)
            throw std::invalid_argument("Ragdoll drive author crosses joint kind or World identity");
        const auto sameRotation = [](const Float4 &a, const Float4 &b) {
            return (a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w) ||
                   (a.x == -b.x && a.y == -b.y && a.z == -b.z && a.w == -b.w);
        };
        const bool changed = record.poseDrive
                                 ? record.poseDrive->clear != intent.clear ||
                                       (!intent.clear && !sameRotation(record.poseDrive->relative, intent.relative))
                                 : !intent.clear;
        if (changed)
            m_impl->AddJointWake(wake, record.value);
        record.poseDrive = intent;
    }
    auto captured = std::make_shared<const std::vector<PoseDriveIntent>>(std::move(intents));
    JoltGpuControlGroup group;
    group.prepareWork = [impl = m_impl.get(), &domain, captured] {
        std::vector<JointPoseDrive> requests;
        requests.reserve(captured->size());
        for (const auto &intent : *captured) {
            // The live author may already be retired; only this ordered accepted
            // prefix can resolve the previously captured joint lifetime.
            const auto &joint = impl->AcceptedJoint(intent.id);
            if (joint.kind != intent.kind || joint.author.identity.z != domain.row)
                throw std::logic_error("Captured ragdoll drive crosses joint kind or World identity");
            requests.push_back({{joint.row, static_cast<uint32_t>(intent.id), static_cast<uint32_t>(intent.id >> 32),
                                 intent.clear ? JointPoseDriveClear : JointPoseDriveSet},
                                intent.relative});
        }
        return impl->joints.PreparePoseDrive(impl->storage, std::move(requests));
    };
    domain.pending.push_back(std::move(group));
    // Identical static targets do not restart quiet history every render frame.
    // Changed intents wake after the actual pose-drive writer in this FIFO.
    if (!wake.commands.empty())
        domain.pending.push_back(std::move(wake));
}

void JoltGpuScene::SetColliderPairs(RuntimeWorld &world, std::shared_ptr<const PhysicsColliderPairs> pairs)
{
    auto &domain = m_impl->Domain(world);
    if (!pairs || domain.closed || domain.retiring)
        throw std::invalid_argument("Collider policy requires its live registered World");
    if (domain.queuedPairs == pairs)
        return;
    JoltGpuControlGroup group;
    group.modelAccepted = [&domain, pairs]() noexcept { domain.acceptedPairs = pairs; };
    domain.pending.push_back(std::move(group));
    domain.queuedPairs = std::move(pairs);
}
void JoltGpuScene::QueueTransferReads(RuntimeWorld &world, const std::vector<JoltBodyImport> &bodies,
                                      std::vector<JoltJointImport> joints)
{
    m_impl->QueueTransferReads(m_impl->Domain(world), bodies, std::move(joints));
}
void JoltGpuScene::QueueTransferRestore(uint32_t id, const JoltBodyTransfer &value)
{
    m_impl->QueueTransferRestore(id, value);
}
void JoltGpuScene::QueueJointTransferRestore(RuntimeWorld &world, std::vector<JoltJointImport> joints)
{
    m_impl->QueueJointTransferRestore(m_impl->Domain(world), std::move(joints));
}
void JoltGpuScene::StageRetire(uint32_t id, bool transferDrive)
{
    m_impl->StageRetire(id, transferDrive);
}
void JoltGpuScene::StageRetireBodies(RuntimeWorld &world, const std::vector<uint32_t> &ids, bool transferDrive,
                                     std::vector<std::pair<uint32_t, uint32_t>> wakeExclusions)
{
    m_impl->StageRetireBodies(m_impl->Domain(world), ids, transferDrive, std::move(wakeExclusions));
}
void JoltGpuScene::QueueBoundsWake(RuntimeWorld &world, Float4 low, Float4 high)
{
    auto &domain = m_impl->Domain(world);
    if (domain.closed || domain.retiring)
        throw std::logic_error("Cannot wake a closed World");
    JoltGpuControlGroup group;
    group.prepareWork = [impl = m_impl.get(), &domain, low, high, validation = Impl::PendingBoundsValidation(domain)] {
        return impl->programs.GetSleep().PrepareAuthoredBoundsWake(impl->storage, impl->SleepSelection(domain), low,
                                                                   high, validation);
    };
    domain.pending.push_back(std::move(group));
}
void JoltGpuScene::QueueStaticWake(uint32_t id)
{
    auto &domain = m_impl->BodyDomain(id);
    const auto record = m_impl->RequireSlot(id);
    auto old = m_impl->QueueOldBounds(domain, record, id);
    m_impl->QueueBoundsAfter(domain, std::move(old), record, id, true);
}
uint64_t JoltGpuScene::CreateJoint(PhysicsConstraintOwner &owner, uint32_t a, uint32_t b, JointAuthor author,
                                   uint64_t importedId)
{
    return m_impl->CreateJoint(owner, a, b, std::move(author), importedId);
}
JPH::GPU::BodyModel JoltGpuScene::GetAcceptedBodyModel(uint32_t bodyId) const
{
    const auto &slot = m_impl->slots[m_impl->RequireSlot(bodyId)];
    if (!slot.accepted)
        throw std::logic_error("Rigidbody has no accepted physical model");
    return slot.accepted->physicalModel;
}

void JoltGpuScene::UpdateJoint(uint64_t id, const std::function<void(JointAuthor &)> &update, uint32_t runtimeIntent)
{
    m_impl->UpdateJoint(id, update, runtimeIntent);
}
void JoltGpuScene::UpdateJoints(std::vector<JointUpdate> values)
{
    m_impl->UpdateJoints(std::move(values));
}
void JoltGpuScene::DestroyJoint(uint64_t id)
{
    m_impl->DestroyJoint(id);
}
void JoltGpuScene::TransferJointsOut(RuntimeWorld &world, const std::vector<uint64_t> &ids)
{
    m_impl->TransferJointsOut(m_impl->Domain(world), ids);
}
void JoltGpuScene::DestroyJointsForBody(uint32_t id)
{
    m_impl->DestroyJointsForBody(id);
}
bool JoltGpuScene::HasJointForBody(uint32_t bodyId) const noexcept
{
    const auto &joints = m_impl->authoredJoints;
    return std::any_of(joints.begin(), joints.end(), [bodyId](const auto &entry) {
        return entry.second.value.bodyIdA == bodyId || entry.second.value.bodyIdB == bodyId;
    });
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
bool JoltGpuScene::HasStagedRigidbody(uint32_t id, PhysicsECSStore::RigidbodyHandle rigidbody) const noexcept
{
    const auto found = m_impl->byId.find(id);
    if (found == m_impl->byId.end() || !rigidbody.IsValid())
        return false;
    const auto &slot = m_impl->slots[found->second];
    return !slot.retiring && !slot.partition->closed && slot.authored &&
           slot.authored->acceptedMotionType != JPH::GPU::MotionStatic && slot.authored->rigidbody == rigidbody;
}
void JoltGpuScene::VisitActorHandles(const RuntimeWorld &world,
                                     const std::function<void(PhysicsActorHandle)> &visitor) const
{
    const auto &domain = m_impl->Domain(world);
    for (const auto index : domain.records)
        if (index != UINT32_MAX)
            visitor(m_impl->slots[index].actor);
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
bool JoltGpuScene::CanReserve(const RuntimeWorld &world, size_t count) const
{
    if (!m_impl->domains.count(world.id))
        return false;
    const auto &domain = m_impl->Domain(world);
    return !domain.closed && domain.freeLocals.size() >= count;
}
bool JoltGpuScene::HasPendingControl(const RuntimeWorld &world) const
{
    if (!m_impl->domains.count(world.id))
        return false;
    const auto &domain = m_impl->Domain(world);
    const bool observing = m_impl->writer == &domain && m_impl->writerKind == Impl::WriterKind::Work;
    return !domain.pending.empty() || (domain.cursor && !observing && !domain.cursor->AreControlInputsAccepted());
}
bool JoltGpuScene::TryBeginTick(RuntimeWorld &world, uint64_t epoch, uint64_t step, float dt,
                                const JoltGpuStepParameters &parameters, ConsumerFactory prepareFinalConsumer,
                                ContactObserverFactory prepareContacts, ParticipantFactory prepareParticipants)
{
    return m_impl->TryBegin(
        m_impl->Domain(world), epoch, step, dt, &parameters, {},
        [&prepareFinalConsumer] { return prepareFinalConsumer ? prepareFinalConsumer() : Consumer{}; },
        std::move(prepareContacts), std::move(prepareParticipants));
}
bool JoltGpuScene::TryBeginControl(RuntimeWorld &world, uint64_t epoch, Consumer consumer, Consumer finalConsumer)
{
    m_impl->Service();
    auto &domain = m_impl->Domain(world);
    const auto step =
        domain.published && domain.published->simulationEpoch == epoch ? domain.published->fixedStepIndex : 0u;
    return m_impl->TryBegin(domain, epoch, step, 0.0f, nullptr, std::move(consumer),
                            [&finalConsumer] { return std::move(finalConsumer); });
}
bool JoltGpuScene::TryBeginWork(RuntimeWorld &world, JoltGpuControlWork &value, bool needsSnapshot)
{
    m_impl->Service();
    auto &domain = m_impl->Domain(world);
    if (m_impl->writer || m_impl->peerWriter || domain.closed || !domain.admission || m_impl->tick.IsPressureActive() ||
        domain.row >= m_impl->storage.GetWorldCount())
        return false;
    JoltGpuTickControl control;
    control.needsSnapshot = needsSnapshot;
    JoltGpuControlGroup group;
    group.prepareWork = [value = std::move(value)]() mutable { return std::move(value); };
    control.groups.push_back(std::move(group));
    m_impl->writer = &domain;
    domain.cursor = &m_impl->tick;
    domain.shareable = false;
    m_impl->writerKind = Impl::WriterKind::Work;
    domain.status = JoltGpuTick::Status::Running;
    m_impl->tick.SetAdmissionEnabled(true);
    m_impl->tick.BeginControl(domain.row, std::move(control));
    return true;
}
bool JoltGpuScene::IsCurrentPublication(const RuntimeWorld &world,
                                        const std::shared_ptr<const Publication> &publication) const
{
    return publication && publication->worldId == world.id &&
           publication->simulationEpoch == world.simulation.simulationEpoch &&
           publication->layoutGeneration == m_impl->layoutGeneration &&
           publication->directory == m_impl->publicationDirectory;
}

void JoltGpuScene::RunPublicationObservation(RuntimeWorld &world, const std::shared_ptr<const Publication> &publication,
                                             JoltGpuControlWork prepared)
{
    if (!publication || publication->worldId != world.id ||
        publication->simulationEpoch != world.simulation.simulationEpoch || !publication->geometry ||
        (publication->bodyCount && (!publication->bodyStates || !publication->bodyStates->IsFrozen())) ||
        !publication->jointStream || !publication->jointStream->IsFrozen())
        throw std::logic_error("Publication read requires its complete immutable World/epoch source");
    if (!prepared.packet && !prepared.continueWork)
        return;
    auto &domain = m_impl->Domain(world);
    if (domain.closed || !domain.admission)
        throw std::logic_error("Publication read owner is closed");
    if (m_impl->observationCallActive)
        throw std::logic_error("Publication read cannot reenter its shared cursor");
    auto &read = m_impl->tick.GetObservationMember();
    auto *previousWorld = m_impl->requiredWorld;
    m_impl->tick.SetSourceFocus(&read);
    const auto availability = [this] {
        return m_impl->storage.GetHost().GetQueuePressure().recordingFrames ? FrameGpuAvailability::ForegroundRecording
                                                                            : FrameGpuAvailability::Ready;
    };
    const auto drain = [&] {
        m_impl->work.DrainRequired(m_impl->tick.GetSourceId(), availability, [this](auto, uint64_t maximum) {
            return m_impl->tick.WaitForSourceProgress(maximum);
        });
    };
    const auto finish = [&] {
        m_impl->observationCallActive = false;
        if (read.CanRetire()) {
            read.ReleaseObservation();
            m_impl->observationOwner = nullptr;
            m_impl->observationPublication.reset();
        }
        SetRequiredWorld(previousWorld);
    };
    try {
        // A previous failed read may still own accepted tickets. Retire only those tickets.
        if (m_impl->observationOwner) {
            drain();
            if (!read.CanRetire())
                throw std::runtime_error("Previous publication read has not retired");
            read.ReleaseObservation();
            m_impl->observationOwner = nullptr;
            m_impl->observationPublication.reset();
        }
        // Private friend entry: only Host gather/pose factories, with no external exports,
        // supply this work. Their kernels read Publication and write private query results.
        m_impl->observationOwner = &domain;
        m_impl->observationCursor = &read;
        m_impl->observationPublication = publication;
        m_impl->observationCallActive = true;
        JoltGpuTickControl control;
        control.needsSnapshot = false;
        JoltGpuControlGroup group;
        group.prepareWork = [prepared = std::move(prepared)]() mutable { return std::move(prepared); };
        control.groups.push_back(std::move(group));
        read.SetAdmissionEnabled(true);
        read.BeginControl(domain.row, std::move(control));
        m_impl->tick.TracePublicationRead("before", domain.row);
        drain();
        if (!read.IsComplete())
            throw std::runtime_error("Publication read did not complete its query work");
        m_impl->tick.TracePublicationRead("after", domain.row);
        finish();
    } catch (...) {
        read.Cancel();
        finish();
        throw;
    }
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
    return (m_impl->writer && m_impl->writer->world == &world) ||
           (m_impl->peerWriter && m_impl->peerWriter->world == &world);
}
JoltGpuControlWork JoltGpuScene::PrepareSleepQuery(const std::shared_ptr<const Publication> &publication,
                                                   std::vector<uint32_t> relativeBodies,
                                                   std::function<void(std::vector<JPH::GPU::UInt4>)> completed)
{
    if (!publication || !publication->sleepStatus)
        throw std::invalid_argument("Sleep query requires a completed nonempty publication");
    return m_impl->programs.GetSleep().PrepareQuery(publication->sleepStatus, publication->bodyCount,
                                                    std::move(relativeBodies), std::move(completed));
}
bool JoltGpuScene::HasRunningFixedTick(const RuntimeWorld &world) const noexcept
{
    // Unlike GetTickStatus, this does not Service/publish any Scene owner.
    const auto found = m_impl->domains.find(world.id);
    if (found == m_impl->domains.end())
        return false;
    const auto &domain = *found->second;
    return domain.cursor && domain.candidate && domain.candidate->isFixedStep &&
           domain.candidate->simulationEpoch == world.simulation.simulationEpoch &&
           domain.candidate->fixedStepIndex == world.simulation.fixedStepIndex &&
           domain.cursor->GetStatus() == JoltGpuTick::Status::Running;
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
    const auto &domain = m_impl->Domain(world);
    return !domain.cursor || domain.cursor->AreControlInputsAccepted();
}
FrameWorkWaitResult JoltGpuScene::WaitForProgress(RuntimeWorld &world, uint64_t maxWait)
{
    m_impl->Service();
    const auto &domain = m_impl->Domain(world);
    return domain.cursor ? domain.cursor->WaitForProgress(maxWait) : FrameWorkWaitResult::Blocked;
}
FrameWorkWaitResult JoltGpuScene::WaitForSourceProgress(uint64_t maxWait)
{
    return m_impl->tick.WaitForSourceProgress(maxWait);
}
RuntimeWorld *JoltGpuScene::SetRequiredWorld(RuntimeWorld *world)
{
    auto *previous = m_impl->requiredWorld;
    m_impl->requiredWorld = world;
    m_impl->tick.SetSourceFocus(world ? m_impl->Domain(*world).cursor : nullptr);
    return previous;
}
void JoltGpuScene::SetAdmissionEnabled(RuntimeWorld &world, bool enabled)
{
    auto &domain = m_impl->Domain(world);
    domain.admission = enabled;
    // Admission controls future logical work. A previously admitted shared
    // writer must still retire when another world's synchronous read needs it.
    if (m_impl->writer)
        m_impl->tick.SetAdmissionEnabled(true);
    if (m_impl->peerWriter && m_impl->peerWriter->cursor)
        m_impl->peerWriter->cursor->SetAdmissionEnabled(true);
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
    if (domain.retiring)
        return;
    domain.retiring = true;
    ++m_impl->retiringPartitions;
    if (!domain.closed)
        m_impl->Cancel(domain);
    m_impl->Service();
}
void JoltGpuScene::ServiceRetirements()
{
    if (m_impl->retiringPartitions)
        m_impl->Service();
}
bool JoltGpuScene::CanRetire(const RuntimeWorld &world)
{
    ServiceRetirements();
    return !m_impl->domains.count(world.id);
}
} // namespace infernux
