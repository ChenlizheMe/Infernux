#include "JoltCpuControl.h"

#include "../Rigidbody.h"
#include "PhysicsBodyState.h"
#include "PhysicsConstraintOwner.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace infernux
{
namespace
{
constexpr uint8_t Position = static_cast<uint8_t>(PhysicsTargetChannels::Position);
constexpr uint8_t Rotation = static_cast<uint8_t>(PhysicsTargetChannels::Rotation);
} // namespace

struct JoltCpuControl::State
{
    struct Binding
    {
        PhysicsECSStore::RigidbodyHandle staged, initialized;
        bool kinematic = false;
        bool staticBody = false;
        bool retiring = false;
    };
    struct Model
    {
        PhysicsActorHandle actor;
        std::unique_ptr<PreparedCpuActor> prepared;
        bool create;
        bool resetMotion;
    };
    struct Add
    {
        std::vector<uint32_t> bodies;
    };
    struct Retire
    {
        bool transferDrive;
    };
    struct Import
    {
        std::shared_ptr<std::optional<JoltBodyTransfer>> state;
    };
    struct Move
    {
        KinematicMove pose;
        PhysicsTargetSource source;
        bool temporaryStatic = false;
    };
    struct JointRecord
    {
        AuthoredJoint value;
        PhysicsConstraintOwner *owner;
    };
    struct JointCreate
    {
        uint64_t id;
        std::unique_ptr<AuthoredJoint> value;
    };
    struct JointUpdates
    {
        std::vector<JointUpdate> updates;
    };
    struct JointPoseDrives
    {
        std::vector<JointPoseDrive> drives;
    };
    struct JointRetire
    {
        std::vector<uint64_t> ids;
    };

    struct Entry
    {
        uint32_t body;
        std::variant<Model, Add, Retire, Import, Command, Move, JointCreate, JointUpdates, JointPoseDrives, JointRetire>
            data;
    };
    struct Fixed
    {
        std::vector<Entry> commands;
        bool applied = false;
    };
    struct Completion
    {
        PhysicsTargetSource source;
        uint8_t channel;
    };

    explicit State(JoltCpuScene &native) : scene(native)
    {
    }

    void RequireStaged(uint32_t body) const
    {
        const auto found = bindings.find(body);
        if (found == bindings.end() || found->second.retiring)
            throw std::invalid_argument("CPU physics command requires a staged body identity");
    }

    void Complete(PhysicsTargetSource &source, uint8_t channel)
    {
        if (source)
            completions.push_back({std::exchange(source, {}), channel});
    }

    bool HasTemporaryStaticDrive(uint32_t body) const
    {
        const auto found = drives.find(body);
        return found != drives.end() && found->second.temporaryStatic;
    }

    void CancelDrive(uint32_t body, bool stopVelocity)
    {
        const auto found = drives.find(body);
        if (found == drives.end())
            return;
        auto &drive = found->second;
        Complete(drive.positionSource, Position);
        Complete(drive.rotationSource, Rotation);
        if (drive.temporaryStatic && drive.ownsVelocity)
            scene.StopTemporaryStaticDrive(body);
        else if (stopVelocity && drive.ownsVelocity)
            scene.SetBodyVelocity(body, JPH::Vec3::sZero(), JPH::Vec3::sZero());
        drives.erase(found);
    }

    void CarryAuthorPoseAndCancel(uint32_t body)
    {
        const auto found = drives.find(body);
        if (found == drives.end())
            return;
        const auto &drive = found->second;
        Pose carry;
        if (drive.pending && drive.positionSource)
            carry.position = drive.position;
        if (drive.pending && drive.rotationSource)
            carry.rotation = drive.rotation;
        CancelDrive(body, true);
        if (carry.position || carry.rotation)
            scene.SetBodyOriginPose(body, carry.position, carry.rotation);
    }

    void RetireBodyConstraints(uint32_t body, const RetiredJoints &retired)
    {
        const auto joints = scene.Constraints().RetireBody(body);
        if (!joints.empty()) {
            retired(body, joints);
            for (const auto id : joints)
                NotifyJointDestroyed(id);
        }
    }

    void Apply(uint32_t body, const Model &model, const RetiredJoints &retired)
    {
        const auto &prepared = *model.prepared;
        const auto &author = prepared.publication;
        auto &binding = bindings.at(body);
        const bool initialize = author.rigidbody.IsValid() && binding.initialized != author.rigidbody;
        const bool replace = !model.create && scene.NeedsBodyReplacement(body, prepared);
        if (model.create) {
            scene.CreateBody(body, prepared);
        } else {
            if (model.resetMotion) {
                CancelDrive(body, false);
                scene.ResetBodyMotion(body);
            }
            const bool kinematic =
                prepared.settings.mMotionType == JPH::EMotionType::Kinematic && !prepared.authoredDynamic;
            const bool preserveStaticDrive =
                prepared.settings.mMotionType == JPH::EMotionType::Static && HasTemporaryStaticDrive(body);
            if (!kinematic && !preserveStaticDrive)
                CarryAuthorPoseAndCancel(body);
            else if (initialize)
                CancelDrive(body, true);
            if (replace) {
                RetireBodyConstraints(body, retired);
                scene.ReplaceBody(body, prepared);
            } else {
                scene.UpdateBody(body, prepared);
                if (initialize && (author.linearVelocity || author.angularVelocity)) {
                    Velocity initial;
                    if (author.linearVelocity)
                        initial.linear = prepared.settings.mLinearVelocity;
                    if (author.angularVelocity)
                        initial.angular = prepared.settings.mAngularVelocity;
                    scene.SetBodyVelocity(body, initial.linear, initial.angular);
                }
            }
        }
        if (initialize)
            binding.initialized = author.rigidbody;
        if (model.actor.IsValid()) {
            const auto revision = scene.ReadMotionState(body, PhysicsStateFields::None).shapeRevision;
            author.AcceptGeometry(model.actor, body, revision);
            if (initialize)
                author.ConsumeInitialVelocities(model.actor, body);
        }
    }

    void Apply(uint32_t, const Add &add, const RetiredJoints &)
    {
        scene.AddBodiesBatch(add.bodies);
    }

    void Apply(uint32_t body, Retire retire, const RetiredJoints &retired)
    {
        if (retire.transferDrive)
            drives.erase(body);
        else
            CancelDrive(body, false);
        RetireBodyConstraints(body, retired);
        scene.DestroyBody(body);
        bindings.erase(body);
    }

    void Apply(uint32_t body, const Import &import, const RetiredJoints &)
    {
        const auto &value = import.state->value();
        scene.RestoreBodyTransfer(body, value);
        if (value.drive)
            drives.insert_or_assign(body, *value.drive);
    }

    void Apply(uint32_t body, const Command &command, const RetiredJoints &)
    {
        std::visit([&](const auto &value) { Apply(body, value); }, command);
    }

    void Apply(uint32_t body, const Pose &pose)
    {
        if (pose.cancelDrive)
            CancelDrive(body, true);
        scene.SetBodyOriginPose(body, pose.position, pose.rotation);
    }

    void Apply(uint32_t body, const Velocity &velocity)
    {
        if (HasTemporaryStaticDrive(body))
            return; // A velocity command has no effect on an authored static body.
        CancelDrive(body, false);
        scene.SetBodyVelocity(body, velocity.linear, velocity.angular);
    }

    void Apply(uint32_t body, const Force &force)
    {
        scene.ApplyBodyForce(body, force.kind, force.mode, force.value, force.point);
    }

    void Apply(uint32_t body, Awake awake)
    {
        if (!awake.awake)
            CancelDrive(body, false);
        if (!HasTemporaryStaticDrive(body))
            scene.SetBodyAwake(body, awake.awake);
    }

    void Apply(uint32_t body, ResetMotion)
    {
        CancelDrive(body, false);
        scene.ResetBodyMotion(body);
    }

    void Apply(uint32_t body, Remove remove)
    {
        CancelDrive(body, true);
        scene.RemoveBodyFromBroadphase(body, remove.preserveVelocity);
    }

    void Apply(uint32_t body, const Move &move, const RetiredJoints &)
    {
        auto &drive = drives[body];
        if (!drive.pending)
            scene.System().GetBodyInterface().GetPositionAndRotation(scene.NativeBody(body), drive.position,
                                                                     drive.rotation);
        if (move.pose.position) {
            Complete(drive.positionSource, Position);
            drive.position = *move.pose.position;
            drive.positionSource = move.source;
        }
        if (move.pose.rotation) {
            Complete(drive.rotationSource, Rotation);
            drive.rotation = *move.pose.rotation;
            drive.rotationSource = move.source;
        }
        drive.duration = move.pose.duration;
        drive.maxSpeed = move.pose.maxSpeed;
        drive.temporaryStatic = move.temporaryStatic;
        drive.pending = true;
    }

    void NotifyJointDestroyed(uint64_t id) noexcept
    {
        const auto found = joints.find(id);
        if (found == joints.end())
            return;
        auto *owner = found->second.owner;
        joints.erase(found);
        owner->OnPhysicsConstraintDestroyed();
    }

    void QueueJointRetirement(const std::vector<uint64_t> &ids)
    {
        if (ids.empty())
            return;
        pending.push_back({UINT32_MAX, JointRetire{ids}});
        for (const auto id : ids)
            NotifyJointDestroyed(id);
    }

    void Apply(uint32_t, const JointCreate &create, const RetiredJoints &)
    {
        const auto &joint = *create.value;
        scene.Constraints().Create(create.id, joint.binding, joint.settings, joint.enableCollision);
    }

    void Apply(uint32_t, const JointUpdates &update, const RetiredJoints &)
    {
        for (const auto &joint : update.updates)
            // A native body replacement earlier in the FIFO can retire this joint.
            if (scene.Constraints().HasJoint(joint.id))
                scene.Constraints().Update(joint.id, joint.settings, joint.enableCollision, joint.clearPoseDrive);
    }

    void Apply(uint32_t, const JointPoseDrives &group, const RetiredJoints &)
    {
        for (const auto &drive : group.drives)
            if (scene.Constraints().HasJoint(drive.id))
                scene.Constraints().SetPoseDrive(drive.id, drive.relative);
    }

    void Apply(uint32_t, const JointRetire &retire, const RetiredJoints &)
    {
        for (const auto id : retire.ids)
            scene.Constraints().Retire(id);
    }

    void Apply(const std::vector<Entry> &entries, const RetiredJoints &retired)
    {
        for (const auto &entry : entries)
            std::visit([&](const auto &value) { Apply(entry.body, value, retired); }, entry.data);
    }

    void StartDrives()
    {
        for (auto it = drives.begin(); it != drives.end();) {
            auto &[body, drive] = *it;
            if (!drive.pending) {
                if (drive.temporaryStatic)
                    scene.StopTemporaryStaticDrive(body);
                else
                    scene.SetBodyVelocity(body, JPH::Vec3::sZero(), JPH::Vec3::sZero());
                it = drives.erase(it);
                continue;
            }
            if (scene.IsBodyInBroadphase(body)) {
                if (drive.temporaryStatic)
                    scene.StartTemporaryStaticDrive(body, drive.position, drive.rotation, drive.duration,
                                                    drive.maxSpeed);
                else
                    scene.StartKinematicDrive(body, drive.position, drive.rotation, drive.duration, drive.maxSpeed);
                Complete(drive.positionSource, Position);
                Complete(drive.rotationSource, Rotation);
                drive.pending = false;
                drive.ownsVelocity = true;
            }
            ++it;
        }
    }

    JoltCpuScene &scene;
    std::unordered_map<uint32_t, Binding> bindings;
    std::unordered_map<uint64_t, JointRecord> joints;
    std::vector<Entry> pending;
    std::optional<Fixed> fixed;
    std::map<uint32_t, JoltBodyTransfer::Drive> drives;
    std::vector<Completion> completions;
};

JoltCpuControl::JoltCpuControl(JoltCpuScene &scene) : m_state(std::make_unique<State>(scene))
{
}

JoltCpuControl::~JoltCpuControl()
{
    Close();
}

void JoltCpuControl::StageActor(uint32_t bodyId, PhysicsActorHandle actor, PreparedCpuActor prepared, bool resetMotion)
{
    auto &state = *m_state;
    const auto rigidbody = prepared.publication.rigidbody;
    const bool kinematic = prepared.settings.mMotionType == JPH::EMotionType::Kinematic && !prepared.authoredDynamic;
    const bool staticBody = prepared.settings.mMotionType == JPH::EMotionType::Static;
    auto model = std::make_unique<PreparedCpuActor>(std::move(prepared));
    const auto [binding, create] = state.bindings.try_emplace(bodyId);
    if (binding->second.retiring)
        throw std::logic_error("A retiring CPU body identity cannot admit a new model");
    try {
        state.pending.push_back({bodyId, State::Model{actor, std::move(model), create, resetMotion}});
    } catch (...) {
        if (create)
            state.bindings.erase(binding);
        throw;
    }
    binding->second.staged = rigidbody;
    binding->second.kinematic = kinematic;
    binding->second.staticBody = staticBody;
}

bool JoltCpuControl::HasStagedRigidbody(uint32_t bodyId, PhysicsECSStore::RigidbodyHandle rigidbody) const
{
    const auto &bindings = m_state->bindings;
    const auto found = bindings.find(bodyId);
    return rigidbody.IsValid() && found != bindings.end() && !found->second.retiring &&
           found->second.staged == rigidbody;
}

void JoltCpuControl::StageTransfer(uint32_t bodyId, const PreparedCpuActor &prepared,
                                   std::shared_ptr<std::optional<JoltBodyTransfer>> value)
{
    auto &state = *m_state;
    const auto rigidbody = prepared.publication.rigidbody;
    const bool kinematic = prepared.settings.mMotionType == JPH::EMotionType::Kinematic && !prepared.authoredDynamic;
    const bool staticBody = prepared.settings.mMotionType == JPH::EMotionType::Static;
    const auto [binding, inserted] =
        state.bindings.try_emplace(bodyId, State::Binding{rigidbody, rigidbody, kinematic, staticBody});
    if (!inserted)
        throw std::logic_error("CPU transfer requires an unused body binding");
    try {
        state.pending.reserve(state.pending.size() + 1);
        state.scene.CreateBody(bodyId, prepared);
    } catch (...) {
        state.bindings.erase(binding);
        throw;
    }
    state.pending.push_back({bodyId, State::Import{std::move(value)}});
}

JoltBodyTransfer JoltCpuControl::ReadBodyTransfer(uint32_t bodyId) const
{
    const auto &state = *m_state;
    auto value = state.scene.ReadBodyTransfer(bodyId);
    if (const auto drive = state.drives.find(bodyId); drive != state.drives.end())
        value.drive = drive->second;
    return value;
}

void JoltCpuControl::AddBodies(std::vector<uint32_t> bodyIds)
{
    if (bodyIds.empty())
        return;
    auto &state = *m_state;
    for (const auto id : bodyIds)
        state.RequireStaged(id);
    state.pending.push_back({UINT32_MAX, State::Add{std::move(bodyIds)}});
}

void JoltCpuControl::RetireActor(uint32_t bodyId, bool transferDrive)
{
    auto &state = *m_state;
    state.RequireStaged(bodyId);
    state.pending.push_back({bodyId, State::Retire{transferDrive}});
    state.bindings.at(bodyId).retiring = true;
}

void JoltCpuControl::Queue(uint32_t bodyId, Command command)
{
    auto &state = *m_state;
    state.RequireStaged(bodyId);
    state.pending.push_back({bodyId, std::move(command)});
}

PhysicsTargetSource JoltCpuControl::QueueMove(uint32_t bodyId, KinematicMove command, const PhysicsTargetAuthor *author)
{
    auto &state = *m_state;
    state.RequireStaged(bodyId);
    if (!state.bindings.at(bodyId).kinematic)
        throw std::invalid_argument("CPU Move requires an authored kinematic body");
    const uint8_t channels = (command.position ? Position : 0) | (command.rotation ? Rotation : 0);
    if (!channels || (author && (author->bodyId != bodyId || author->channels != channels)))
        throw std::invalid_argument("CPU Move pose does not match its authored channels");
    PhysicsTargetSource source;
    if (author) {
        auto batch = std::make_shared<PhysicsTargetBatch>();
        batch->records.push_back(*author);
        source = {std::move(batch), 0};
    }
    state.pending.push_back({bodyId, State::Move{std::move(command), source}});
    return source;
}

void JoltCpuControl::QueueTemporaryStaticMove(uint32_t bodyId, JPH::RVec3Arg position, JPH::QuatArg rotation,
                                              float duration, float maxSpeed)
{
    auto &state = *m_state;
    state.RequireStaged(bodyId);
    if (!state.bindings.at(bodyId).staticBody)
        throw std::invalid_argument("CPU temporary movement requires an authored static body");
    if (!std::isfinite(position.GetX()) || !std::isfinite(position.GetY()) || !std::isfinite(position.GetZ()) ||
        !rotation.IsNormalized() || !std::isfinite(duration) || duration <= 0.0f || !std::isfinite(maxSpeed) ||
        maxSpeed < 0.0f)
        throw std::invalid_argument("Invalid CPU temporary static pose, duration or speed limit");
    state.pending.push_back({bodyId, State::Move{{position, rotation, duration, maxSpeed}, {}, true}});
}

void JoltCpuControl::CreateJoint(uint64_t id, PhysicsConstraintOwner &owner, AuthoredJoint joint)
{
    auto &state = *m_state;
    state.RequireStaged(joint.binding.ownerBody);
    if (joint.binding.connectedBody != JoltCpuConstraints::WorldBody)
        state.RequireStaged(joint.binding.connectedBody);
    JoltCpuConstraints::ValidateAuthor(joint.binding, joint.settings);
    auto captured = std::make_unique<AuthoredJoint>(joint);
    const auto [entry, inserted] = state.joints.emplace(id, State::JointRecord{std::move(joint), &owner});
    if (!inserted)
        throw std::logic_error("CPU joint already owns its public identity");
    try {
        state.pending.push_back({UINT32_MAX, State::JointCreate{id, std::move(captured)}});
    } catch (...) {
        state.joints.erase(entry);
        throw;
    }
}

const JoltCpuControl::AuthoredJoint *JoltCpuControl::FindJoint(uint64_t id) const noexcept
{
    const auto &joints = m_state->joints;
    const auto entry = joints.find(id);
    return entry != joints.end() ? &entry->second.value : nullptr;
}

void JoltCpuControl::UpdateJoints(std::vector<JointUpdate> updates)
{
    if (updates.empty())
        return;
    auto &state = *m_state;
    for (const auto &update : updates) {
        const auto &joint = state.joints.at(update.id).value;
        JoltCpuConstraints::ValidateAuthorUpdate(joint.settings, update.settings);
    }
    state.pending.push_back({UINT32_MAX, State::JointUpdates{std::move(updates)}});
    for (const auto &update : std::get<State::JointUpdates>(state.pending.back().data).updates) {
        auto &joint = state.joints.at(update.id).value;
        joint.settings = update.settings;
        joint.enableCollision = update.enableCollision;
    }
}

void JoltCpuControl::QueueJointPoseDrives(std::vector<JointPoseDrive> drives)
{
    if (drives.empty())
        return;
    auto &state = *m_state;
    for (const auto &drive : drives) {
        const auto *joint = FindJoint(drive.id);
        if (!joint || !JoltCpuConstraints::SupportsPoseDrive(joint->settings) ||
            (drive.relative && !drive.relative->IsNormalized()))
            throw std::invalid_argument("CPU pose drive does not match an authored angular joint");
    }
    state.pending.push_back({UINT32_MAX, State::JointPoseDrives{std::move(drives)}});
}

void JoltCpuControl::DestroyJoint(uint64_t id)
{
    if (FindJoint(id))
        m_state->QueueJointRetirement({id});
}

bool JoltCpuControl::HasJointForBody(uint32_t bodyId) const noexcept
{
    const auto &joints = m_state->joints;
    return std::any_of(joints.begin(), joints.end(), [bodyId](const auto &entry) {
        const auto &binding = entry.second.value.binding;
        return binding.ownerBody == bodyId || binding.connectedBody == bodyId;
    });
}

std::vector<uint64_t> JoltCpuControl::DestroyJointsForBody(uint32_t bodyId)
{
    auto &state = *m_state;
    std::vector<uint64_t> ids;
    for (const auto &[id, entry] : state.joints) {
        const auto &binding = entry.value.binding;
        if (binding.ownerBody == bodyId || binding.connectedBody == bodyId)
            ids.push_back(id);
    }
    state.QueueJointRetirement(ids);
    return ids;
}

bool JoltCpuControl::HasPending() const noexcept
{
    return !m_state->pending.empty();
}

void JoltCpuControl::SealFixed()
{
    auto &state = *m_state;
    if (state.fixed)
        throw std::logic_error("CPU Fixed already owns an author prefix");
    state.fixed.emplace(State::Fixed{std::exchange(state.pending, {})});
}

bool JoltCpuControl::ApplyControl(const RetiredJoints &retired)
{
    auto &state = *m_state;
    if (state.fixed)
        throw std::logic_error("CPU control crossed a retained Fixed prefix");
    const auto entries = std::exchange(state.pending, {});
    state.Apply(entries, retired);
    return !entries.empty();
}

void JoltCpuControl::ApplyFixed(const RetiredJoints &retired)
{
    auto &state = *m_state;
    if (!state.fixed || std::exchange(state.fixed->applied, true))
        throw std::logic_error("CPU Fixed must consume its sealed author prefix once");
    const auto entries = std::move(state.fixed->commands);
    state.Apply(entries, retired);
    state.StartDrives();
}

void JoltCpuControl::PublishApplied()
{
    auto &state = *m_state;
    if (state.fixed && !state.fixed->applied)
        throw std::logic_error("CPU Fixed publication requires an applied prefix");
    const auto completed = std::exchange(state.completions, {});
    for (const auto &item : completed)
        if (const auto *author = item.source.Get())
            if (auto *body = dynamic_cast<Rigidbody *>(author->rigidbody.GetComponent()))
                body->CompletePhysicsTarget(item.source, item.channel);
    state.fixed.reset();
}

void JoltCpuControl::Close() noexcept
{
    auto &state = *m_state;
    state.pending.clear();
    state.fixed.reset();
    state.drives.clear();
    state.completions.clear();
    state.bindings.clear();
    while (!state.joints.empty())
        state.NotifyJointDestroyed(state.joints.begin()->first);
}
} // namespace infernux
