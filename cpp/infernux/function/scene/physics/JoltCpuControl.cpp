#include "JoltCpuControl.h"

#include "../Rigidbody.h"
#include "PhysicsBodyState.h"
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
        bool retiring = false;
    };
    struct Model
    {
        PhysicsActorHandle actor;
        std::unique_ptr<PreparedCpuActor> prepared;
        bool create;
    };
    struct Add
    {
        std::vector<uint32_t> bodies;
    };
    struct Retire
    {
    };
    struct Move
    {
        KinematicMove pose;
        PhysicsTargetSource source;
    };
    struct Entry
    {
        uint32_t body;
        std::variant<Model, Add, Retire, Command, Move> data;
    };
    struct Fixed
    {
        std::vector<Entry> commands;
        bool applied = false;
    };
    struct Drive
    {
        JPH::RVec3 position;
        JPH::Quat rotation;
        float duration = 0.0f;
        float maxSpeed = 0.0f;
        PhysicsTargetSource positionSource, rotationSource;
        bool pending = false;
        bool ownsVelocity = false;
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

    void CancelDrive(uint32_t body, bool stopVelocity)
    {
        const auto found = drives.find(body);
        if (found == drives.end())
            return;
        auto &drive = found->second;
        Complete(drive.positionSource, Position);
        Complete(drive.rotationSource, Rotation);
        if (stopVelocity && drive.ownsVelocity)
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
        if (!joints.empty())
            retired(body, joints);
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
            const bool kinematic =
                prepared.settings.mMotionType == JPH::EMotionType::Kinematic && !prepared.authoredDynamic;
            if (!kinematic)
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

    void Apply(uint32_t body, Retire, const RetiredJoints &retired)
    {
        CancelDrive(body, false);
        RetireBodyConstraints(body, retired);
        scene.DestroyBody(body);
        bindings.erase(body);
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
        drive.pending = true;
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
                scene.SetBodyVelocity(body, JPH::Vec3::sZero(), JPH::Vec3::sZero());
                it = drives.erase(it);
                continue;
            }
            if (scene.IsBodyInBroadphase(body)) {
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
    std::vector<Entry> pending;
    std::optional<Fixed> fixed;
    std::map<uint32_t, Drive> drives;
    std::vector<Completion> completions;
};

JoltCpuControl::JoltCpuControl(JoltCpuScene &scene) : m_state(std::make_unique<State>(scene))
{
}

JoltCpuControl::~JoltCpuControl() = default;

void JoltCpuControl::StageActor(uint32_t bodyId, PhysicsActorHandle actor, PreparedCpuActor prepared)
{
    auto &state = *m_state;
    const auto rigidbody = prepared.publication.rigidbody;
    const bool kinematic = prepared.settings.mMotionType == JPH::EMotionType::Kinematic && !prepared.authoredDynamic;
    auto model = std::make_unique<PreparedCpuActor>(std::move(prepared));
    const auto [binding, create] = state.bindings.try_emplace(bodyId);
    if (binding->second.retiring)
        throw std::logic_error("A retiring CPU body identity cannot admit a new model");
    try {
        state.pending.push_back({bodyId, State::Model{actor, std::move(model), create}});
    } catch (...) {
        if (create)
            state.bindings.erase(binding);
        throw;
    }
    binding->second.staged = rigidbody;
    binding->second.kinematic = kinematic;
}

bool JoltCpuControl::HasStagedRigidbody(uint32_t bodyId, PhysicsECSStore::RigidbodyHandle rigidbody) const
{
    const auto &bindings = m_state->bindings;
    const auto found = bindings.find(bodyId);
    return rigidbody.IsValid() && found != bindings.end() && !found->second.retiring &&
           found->second.staged == rigidbody;
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

void JoltCpuControl::RetireActor(uint32_t bodyId)
{
    auto &state = *m_state;
    state.RequireStaged(bodyId);
    state.pending.push_back({bodyId, State::Retire{}});
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
}
} // namespace infernux
