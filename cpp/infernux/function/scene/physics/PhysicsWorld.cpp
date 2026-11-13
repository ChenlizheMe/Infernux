#include "PhysicsWorld.h"
#include <Jolt/Jolt.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <Jolt/RegisterTypes.h>
#include "../Collider.h"
#include "../Component.h"
#include "../GameObject.h"
#include "../Rigidbody.h"
#include "../SceneManager.h"
#include "../Transform.h"
#include "../TransformECSStore.h"
#include "JoltGpuProgramSources.h"
#include "JoltGpuQueries.h"
#include "JoltGpuRenderBindings.h"
#include "JoltGpuRenderPoses.h"
#include "PhysicsContactTypes.h"
#include "PhysicsMass.h"
#include "PhysicsShapeAuthoring.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <core/config/EngineConfig.h>
#include <core/threading/FrameWorkOwner.h>
#include <function/renderer/rhi/RhiComputeHost.h>
#include <function/renderer/rhi/RhiComputeKernel.h>
#include <function/resources/PhysicMaterial/PhysicMaterial.h>
#include <limits>
#include <stdexcept>
#include <utility>

namespace infernux
{
namespace
{
using namespace JPH::GPU;
[[noreturn]] void Unsupported(const char *operation)
{
    throw std::runtime_error(std::string("GPU physics has not implemented ") + operation);
}
bool Finite(const glm::vec3 &v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}
Float4 Vector(const glm::vec3 &v)
{
    if (!Finite(v))
        throw std::invalid_argument("Physics vector must be finite");
    return {v.x, v.y, v.z, 0};
}
glm::vec3 Vector(const Float4 &v)
{
    return {v.x, v.y, v.z};
}
Float4 Rotation(const glm::quat &v)
{
    const float norm = glm::dot(v, v);
    if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z) || !std::isfinite(v.w) ||
        !std::isfinite(norm) || norm <= 1e-12f)
        throw std::invalid_argument("Physics rotation must be finite and nonzero");
    const auto q = glm::normalize(v);
    return {q.x, q.y, q.z, q.w};
}
QueryInput Query(uint32_t operation, uint32_t collector, uint32_t layers = UINT32_MAX, bool triggers = true)
{
    QueryInput request{};
    request.meta = {operation, collector, ShapeSphere, 0};
    request.filter = {layers, triggers ? 1u : 0u, 0, 0};
    request.source.x = InvalidQueryBodySlot;
    request.rotation.w = 1;
    return request;
}
void TargetCollider(QueryInput &request, const Collider &collider)
{
    const auto id = collider.GetComponentID();
    request.meta.w |= QueryTargetCollider;
    request.filter.z = static_cast<uint32_t>(id);
    request.filter.w = static_cast<uint32_t>(id >> 32);
}
void QueryRayInput(QueryInput &request, const glm::vec3 &origin, const glm::vec3 &direction, float maximum)
{
    request.positionMaximum = Vector(origin);
    request.direction = Vector(direction);
    if (!std::isfinite(maximum) || maximum <= 0 || (direction.x == 0 && direction.y == 0 && direction.z == 0))
        throw std::invalid_argument("Physics ray requires positive finite distance and nonzero direction");
    request.positionMaximum.w = maximum;
}
void QuerySphereInput(QueryInput &request, const glm::vec3 &center, float radius)
{
    request.positionMaximum = Vector(center);
    if (!std::isfinite(radius) || radius <= 0)
        throw std::invalid_argument("Physics query sphere requires a positive finite radius");
    request.meta.z = ShapeSphere;
    request.geometry = {0, 0, 0, radius};
}
void QueryBoxInput(QueryInput &request, const glm::vec3 &center, const glm::vec3 &halfExtents,
                   const glm::quat &rotation)
{
    request.positionMaximum = Vector(center);
    const auto half = Vector(halfExtents);
    if (half.x <= 0 || half.y <= 0 || half.z <= 0)
        throw std::invalid_argument("Physics query box requires positive finite half extents");
    request.meta.z = ShapeBox;
    request.geometry = {half.x, half.y, half.z, std::min(0.05f, std::min({half.x, half.y, half.z}))};
    request.rotation = Rotation(rotation);
}
void QueryCapsuleInput(QueryInput &request, const glm::vec3 &a, const glm::vec3 &b, float radius)
{
    (void)Vector(a);
    (void)Vector(b);
    const glm::dvec3 delta = glm::dvec3(b) - glm::dvec3(a);
    const double radial = std::hypot(delta.x, delta.z);
    const double length = std::hypot(radial, delta.y);
    const auto center = glm::vec3((glm::dvec3(a) + glm::dvec3(b)) * 0.5);
    QuerySphereInput(request, center, radius);
    if (length == 0)
        return;
    if (length * 0.5 > std::numeric_limits<float>::max())
        throw std::invalid_argument("Physics query capsule length exceeds device coordinates");
    request.meta.z = ShapeCapsule;
    request.geometry.y = static_cast<float>(length * 0.5);
    const double halfAngle = std::atan2(radial, delta.y) * 0.5;
    const double sine = std::sin(halfAngle);
    request.rotation =
        radial > 0    ? Float4{static_cast<float>(delta.z / radial * sine), 0,
                               static_cast<float>(-delta.x / radial * sine), static_cast<float>(std::cos(halfAngle))}
        : delta.y > 0 ? Float4{0, 0, 0, 1}
                      : Float4{1, 0, 0, 0};
}
struct AdmissionScope
{
    explicit AdmissionScope(JoltGpuScene &scene) : scene(scene)
    {
        scene.SetAdmissionEnabled(true);
    }
    ~AdmissionScope()
    {
        scene.SetAdmissionEnabled(false);
    }
    JoltGpuScene &scene;
};
struct FlagScope
{
    explicit FlagScope(bool &flag) : flag(flag)
    {
        flag = true;
    }
    ~FlagScope()
    {
        flag = false;
    }
    bool &flag;
};
FrameGpuAvailability Availability(const rhi::ComputeHost &host)
{
    return host.GetQueuePressure().recordingFrames ? FrameGpuAvailability::ForegroundRecording
                                                   : FrameGpuAvailability::Ready;
}
ShaderTargetCode CompileProgram(rhi::ComputeHost &host, JoltGpuProgram program)
{
    ShaderCodeTarget target;
    switch (host.device.GetShaderSourceLanguage()) {
    case rhi::ShaderSourceLanguage::SpirV:
        target = ShaderCodeTarget::SpirV;
        break;
    case rhi::ShaderSourceLanguage::Wgsl:
        target = ShaderCodeTarget::Wgsl;
        break;
    default:
        throw std::runtime_error("Unsupported physics shader target");
    }
    auto compiled =
        SlangCompiler::Get().Compile(GetJoltGpuProgramSource(program), GetJoltGpuProgramLibraries(), {target}, true);
    if (compiled.targets.size() != 1 || compiled.targets.front().target != target)
        throw std::runtime_error("Common compiler did not provide the requested GPU physics target");
    return std::move(compiled.targets.front());
}
PhysicsActorData &ActorForBody(JoltGpuScene &scene, uint32_t bodyId)
{
    auto &store = PhysicsECSStore::Instance();
    const auto handle = scene.FindActor(bodyId);
    if (!store.IsValid(handle) || store.GetActor(handle).bodyId != bodyId)
        throw std::invalid_argument("Physics body identity is invalid or retired");
    return store.GetActor(handle);
}
Rigidbody &KinematicBody(JoltGpuScene &scene, uint32_t bodyId)
{
    auto *body = ActorForBody(scene, bodyId).rigidbody;
    if (!body || !body->IsEnabled() || !body->IsKinematic())
        throw std::logic_error("Kinematic movement requires an enabled kinematic Rigidbody");
    return *body;
}
BodyCommand MoveCommand(uint32_t kind, const glm::vec3 &position, const glm::quat &rotation, float dt,
                        float maxSpeed = 0.0f)
{
    if (!std::isfinite(dt) || dt <= 0 || !std::isfinite(maxSpeed) || maxSpeed < 0)
        throw std::invalid_argument("Physics movement requires a positive duration and nonnegative speed limit");
    BodyCommand command{};
    command.meta.y = kind;
    if (kind != CommandMoveRotation)
        command.value = Vector(position);
    command.value.w = dt;
    if (kind != CommandMovePosition)
        command.auxiliary = Rotation(rotation);
    command.rootCOM.w = maxSpeed;
    return command;
}
JoltGpuStepParameters StepParameters(float dt)
{
    const auto &c = EngineConfig::Get();
    if (c.physicsVelocitySteps < 0 || c.physicsPositionSteps < 0 || c.physicsCollisionSteps <= 0)
        throw std::invalid_argument("Physics iterations and substeps are invalid");
    StepParams p{};
    p.counts.w = static_cast<uint32_t>(c.physicsVelocitySteps);
    p.solve.x = static_cast<uint32_t>(c.physicsPositionSteps);
    p.solve.y = static_cast<uint32_t>(c.physicsCollisionSteps);
    p.gravityDt = {c.physicsGravity.x, c.physicsGravity.y, c.physicsGravity.z, dt};
    p.contact = {c.physicsSpeculativeContactDistance, c.physicsPenetrationSlop, c.physicsMaxPenetrationDistance,
                 c.physicsBaumgarte};
    p.restitution.x = c.physicsMinVelocityForRestitution;
    return JoltGpuStepParameters(p);
}
} // namespace

struct PhysicsWorld::CompiledGpuPrograms
{
    ShaderTargetCode step, mutations, commands;
};
struct PhysicsWorld::GpuRenderState
{
    using Keys = std::array<uint64_t, 3>;
    static Keys CaptureKeys()
    {
        const auto &manager = SceneManager::Instance();
        return {manager.GetGlobalTransformSerial(), manager.GetMeshRendererVersion(),
                manager.GetRenderContentRevision()};
    }
    struct Job
    {
        JoltGpuRenderBindings::SnapshotPtr snapshot;
        Keys keys{};
        uint64_t numericalGeneration = 0;
        std::optional<JoltGpuRenderBindings::Prepared> bindings;
        JoltGpuRenderPoseBatch output;
        const JoltGpuScene::Publication *publication = nullptr;
    };
    GpuRenderState(rhi::ComputeHost &host, const ShaderTargetCode &target) : program(host, target)
    {
    }
    JoltGpuRenderBindings bindings;
    JoltGpuRenderPoses program;
    std::shared_ptr<Job> pending;
    std::shared_ptr<Job> ready;
    std::optional<Keys> checkedKeys;
};
PhysicsWorld &PhysicsWorld::Instance()
{
    static PhysicsWorld *world = new PhysicsWorld();
    return *world;
}
PhysicsWorld::~PhysicsWorld()
{
    Shutdown();
}
void PhysicsWorld::RequireOwnerThread() const
{
    if (m_ownerThreadId != std::thread::id{} && m_ownerThreadId != std::this_thread::get_id())
        throw std::logic_error("GPU physics requires its engine owner thread");
}
void PhysicsWorld::BindGpuServices(std::unique_ptr<rhi::ComputeHost> host, FrameWorkOwner *work)
{
    if (m_initialized)
        throw std::logic_error("Bind physics compute services before initializing the world");
    m_gpuHost = std::move(host);
    m_gpuWork = work;
}
void PhysicsWorld::Initialize()
{
    if (m_initialized)
        return;
    m_ownerThreadId = std::this_thread::get_id();
    JPH::RegisterDefaultAllocator();
    if (!JPH::Factory::sInstance) {
        JPH::Factory::sInstance = new JPH::Factory();
        m_ownsJoltGlobals = true;
        JPH::RegisterTypes();
    }
    m_initialized = true;
    if (m_gpuHost && m_gpuWork) {
        try {
            (void)EnsureGpuReady();
        } catch (...) {
            Shutdown();
            throw;
        }
    }
}
bool PhysicsWorld::EnsureGpuReady()
{
    RequireOwnerThread();
    RequireHealthyStep();
    if (!m_initialized)
        throw std::logic_error("PhysicsWorld is not initialized");
    if (!m_gpuHost || !m_gpuWork)
        throw std::runtime_error("Physics execution requires engine GPU compute services; CPU physics is unavailable");
    m_gpuHost->RequireAlive();
    if (m_gpuCancelled) {
        if (m_gpuScene && !m_gpuScene->CanRetire())
            return false;
        m_gpuScene.reset();
        m_lastPublication.reset();
        m_requestedMotion.clear();
        m_gpuCancelled = false;
    }
    if (!m_gpuPrograms) {
        auto p = std::make_unique<CompiledGpuPrograms>();
        p->step = CompileProgram(*m_gpuHost, JoltGpuProgram::Step);
        p->mutations = CompileProgram(*m_gpuHost, JoltGpuProgram::BodyMutations);
        p->commands = CompileProgram(*m_gpuHost, JoltGpuProgram::BodyCommands);
        const auto target = CompileProgram(*m_gpuHost, JoltGpuProgram::Queries);
        auto queries = std::make_unique<JoltGpuQueries>(*m_gpuHost, target);
        auto rendering =
            std::make_unique<GpuRenderState>(*m_gpuHost, CompileProgram(*m_gpuHost, JoltGpuProgram::RenderPoses));
        m_gpuPrograms = std::move(p);
        m_gpuQueries = std::move(queries);
        m_gpuRenderState = std::move(rendering);
    }
    if (!m_gpuScene) {
        m_recreateActors.reserve(MaxBodiesPerWorld);
        m_gpuScene = std::make_unique<JoltGpuScene>(*m_gpuHost, *m_gpuWork, MaxBodiesPerWorld, m_gpuPrograms->step,
                                                    m_gpuPrograms->mutations, m_gpuPrograms->commands);
        m_gpuScene->SetAdmissionEnabled(false);
        auto &store = PhysicsECSStore::Instance();
        for (const auto handle : m_recreateActors)
            if (store.IsValid(handle) && store.GetActor(handle).bodyId == UINT32_MAX)
                store.QueueBodyCreation(handle);
        m_recreateActors.clear();
    }
    return true;
}
void PhysicsWorld::RequireGpu() const
{
    if (!const_cast<PhysicsWorld *>(this)->EnsureGpuReady())
        throw std::runtime_error("Stopped GPU physics is still retiring accepted receipts");
}
void PhysicsWorld::Shutdown()
{
    CancelPendingStep();
    m_gpuScene.reset(); // Explicit shutdown drain. Normal Stop remains nonblocking.
    m_gpuRenderState.reset();
    m_gpuQueries.reset();
    m_gpuPrograms.reset();
    m_lastPublication.reset();
    m_requestedMotion.clear();
    m_recreateActors.clear();
    m_gpuHost.reset();
    m_gpuWork = nullptr;
    if (m_ownsJoltGlobals) {
        JPH::UnregisterTypes();
        delete JPH::Factory::sInstance;
        JPH::Factory::sInstance = nullptr;
        m_ownsJoltGlobals = false;
    }
    m_initialized = false;
    m_gpuCancelled = false;
    m_ownerThreadId = {};
}
size_t PhysicsWorld::GetBodyCount() const
{
    return m_gpuScene ? m_gpuScene->GetBodyCount() : 0;
}
glm::vec3 PhysicsWorld::GetGravity() const
{
    return EngineConfig::Get().physicsGravity;
}
void PhysicsWorld::SetGravity(const glm::vec3 &v)
{
    if (!Finite(v))
        throw std::invalid_argument("Gravity must be finite");
    EngineConfig::Get().physicsGravity = v;
}
uint32_t PhysicsWorld::GetDefaultQueryLayerMask()
{
    return EngineConfig::Get().defaultQueryLayerMask;
}
void PhysicsWorld::CheckContactSupport() const
{
    bool requested = m_contactEventStreamEnabled || m_contactImpulseStreamEnabled;
    PhysicsECSStore::Instance().ForEachAliveActor([&](const PhysicsActorData &actor) {
        if (!actor.owner || actor.sceneSuspended || !actor.owner->IsActiveInHierarchy())
            return;
        for (const auto &c : actor.owner->GetAllComponents())
            requested = requested || (c && c->WantsPhysicsCallbacks());
        for (const auto *col : actor.colliders)
            if (col && col->GetGameObject() != actor.owner)
                for (const auto &c : col->GetGameObject()->GetAllComponents())
                    requested = requested || (c && c->WantsPhysicsCallbacks());
    });
    if (requested)
        Unsupported("collision/trigger events and impulse streams");
}
void PhysicsWorld::ObserveGpuPublication(bool numericalChange) const
{
    const auto &p = m_gpuScene->GetPublished();
    if (p && p.get() != m_lastPublication.get()) {
        m_lastPublication = p;
        if (numericalChange) {
            m_queryGeneration.fetch_add(1, std::memory_order_release);
            m_requestedMotion.clear();
        }
        if (m_gpuRenderState) {
            FrameWorkOwner::Scope publication(*m_gpuWork);
            auto &rendering = *m_gpuRenderState;
            const auto &job = rendering.pending;
            if (job && job->bindings && job->publication == p.get()) {
                // Preserve this complete value before another Prepare reuses
                // the live output endpoints, even while the fixed gate waits.
                if (job->output.models)
                    job->output.models = job->output.models->Freeze();
                if (job->output.bounds)
                    job->output.bounds = job->output.bounds->Freeze();
                job->output.batch.reset();
                job->numericalGeneration = GetQueryGeneration();
                rendering.ready = job;
            }
            rendering.pending.reset();
        }
        auto &manager = SceneManager::Instance();
        if (!manager.HasPendingFixedStep())
            manager.PublishPhysicsStateToRenderer();
    }
}
bool PhysicsWorld::PublishRenderPoses() const
{
    RequireOwnerThread();
    if (!m_initialized || !m_gpuRenderState || m_gpuCancelled || m_fixedGpuPhase == FixedGpuPhase::Failed)
        return false;
    auto &rendering = *m_gpuRenderState;
    const auto &job = rendering.ready;
    if (!job)
        return false;
    FrameWorkOwner::Scope publication(*m_gpuWork);
    try {
        const bool published = rendering.bindings.Publish(*job->bindings, job->output.models, job->output.bounds,
                                                          job->numericalGeneration);
        if (published)
            rendering.checkedKeys = job->keys;
        else
            rendering.checkedKeys.reset();
        rendering.ready.reset();
        return published;
    } catch (...) {
        const_cast<PhysicsWorld *>(this)->FailStep();
        throw;
    }
}
JoltGpuScene::Consumer PhysicsWorld::PrepareRenderConsumer(bool numericalChange) const
{
    FrameWorkOwner::Scope preparation(*m_gpuWork);
    auto &rendering = *m_gpuRenderState;
    const auto keys = GpuRenderState::CaptureKeys();
    auto snapshot = rendering.bindings.Capture();
    const auto generation = GetQueryGeneration() + static_cast<uint64_t>(numericalChange);
    if (rendering.ready && rendering.ready->numericalGeneration == generation && rendering.ready->keys == keys) {
        rendering.pending.reset();
        return {};
    }
    if (!rendering.bindings.NeedsExport(*snapshot, generation)) {
        rendering.pending.reset();
        rendering.checkedKeys = keys;
        return {};
    }
    auto job = std::make_shared<GpuRenderState::Job>();
    job->snapshot = std::move(snapshot);
    job->keys = keys;
    rendering.pending = job;
    return [this, job = std::move(job)](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        auto &state = *m_gpuRenderState;
        job->bindings = state.bindings.Prepare(job->snapshot, *publication);
        job->output = state.program.Prepare(publication->bodyStates, job->bindings->nodes, job->bindings->bindings);
        job->publication = publication.get();
        return job->output.batch;
    };
}
void PhysicsWorld::RequireHealthyStep() const
{
    if (m_fixedGpuPhase == FixedGpuPhase::Failed)
        throw std::runtime_error("GPU physics step failed; stop the runtime before restarting");
}
void PhysicsWorld::FailStep() noexcept
{
    m_fixedGpuPhase = FixedGpuPhase::Failed;
    m_stepCompleted = false;
    if (m_gpuScene)
        m_gpuScene->SetAdmissionEnabled(false);
}
void PhysicsWorld::RefreshGpuSurfaces()
{
    const auto revision = PhysicMaterial::GetSurfaceRevision();
    if (revision == m_surfaceRevision)
        return;
    auto &store = PhysicsECSStore::Instance();
    store.ForEachAliveActor([&](const PhysicsActorData &actor) {
        if (actor.bodyId == UINT32_MAX)
            return;
        const auto handle = m_gpuScene->FindActor(actor.bodyId);
        if (store.IsValid(handle) && &store.GetActor(handle) == &actor)
            RefreshGpuActor(actor.bodyId);
    });
    m_surfaceRevision = revision;
}
bool PhysicsWorld::Step(float dt, PhysicsStepProfile *profile)
{
    RequireOwnerThread();
    RequireHealthyStep();
    if (profile)
        *profile = {};
    if (!std::isfinite(dt) || dt <= 0)
        throw std::invalid_argument("Fixed physics dt must be positive and finite");
    try {
        const auto &manager = SceneManager::Instance();
        const auto epoch = manager.GetSimulationEpoch(), index = manager.GetFixedStepIndex();
        if (m_stepPending) {
            if (m_stepEpoch != epoch || m_stepIndex != index)
                throw std::logic_error("Physics step must be consumed with its original epoch and index");
            if (m_stepCompleted) {
                m_stepPending = m_stepCompleted = false;
                m_fixedGpuPhase = FixedGpuPhase::Idle;
                m_admittedParameters.reset();
                return true;
            }
        } else {
            if (!EnsureGpuReady())
                return false;
            CheckContactSupport();
            RefreshGpuSurfaces();
            m_stepEpoch = epoch;
            m_stepIndex = index;
            m_admittedDeltaTime = dt;
            m_admittedParameters = StepParameters(dt);
            // Own this fixed identity before waiting for an earlier visual
            // control. Scripts and accounting cannot be replayed meanwhile.
            m_stepPending = true;
            if (m_controlPending) {
                m_fixedGpuPhase = FixedGpuPhase::RetireControl;
            } else if (m_waitingForSlot) {
                (void)BeginGpuControl(true);
                m_fixedGpuPhase = FixedGpuPhase::RetireControl;
            } else {
                m_gpuScene->BeginTick(epoch, index, dt, *m_admittedParameters, PrepareRenderConsumer(true));
                m_fixedGpuPhase = FixedGpuPhase::Solver;
            }
        }
        if (!AdvanceFixedStep(false))
            return false;
        m_stepPending = m_stepCompleted = false;
        m_fixedGpuPhase = FixedGpuPhase::Idle;
        m_admittedParameters.reset();
        return true;
    } catch (...) {
        FailStep();
        throw;
    }
}
bool PhysicsWorld::AdvanceFixedStep(bool synchronous)
{
    RequireHealthyStep();
    try {
        for (;;) {
            if (m_fixedGpuPhase == FixedGpuPhase::RetryCreation) {
                RetryCapacityBodyCreations();
                RefreshGpuSurfaces();
                if (m_waitingForSlot) {
                    (void)BeginGpuControl(true);
                    m_fixedGpuPhase = FixedGpuPhase::RetireControl;
                } else {
                    m_gpuScene->BeginTick(m_stepEpoch, m_stepIndex, m_admittedDeltaTime, *m_admittedParameters,
                                          PrepareRenderConsumer(true));
                    m_fixedGpuPhase = FixedGpuPhase::Solver;
                }
            }
            if (m_fixedGpuPhase == FixedGpuPhase::RetireControl) {
                if (!AdvanceGpuControl(synchronous))
                    return false;
                m_fixedGpuPhase = FixedGpuPhase::RetryCreation;
                continue;
            }
            if (!AdvanceGpuJob(synchronous))
                return false;
            if (m_gpuScene->PublishCompletedTick())
                ObserveGpuPublication(true);
            m_stepCompleted = true;
            return true;
        }
    } catch (...) {
        FailStep();
        throw;
    }
}
void PhysicsWorld::RetryCapacityBodyCreations()
{
    m_waitingForSlot = false;
    auto &store = PhysicsECSStore::Instance();
    // Revisit only the existing actor factory/residency queues. No scripts,
    // clocks, fixed callbacks or unrelated world work are replayed here.
    for (const auto handle : store.ConsumePendingBodyCreations())
        if (store.IsValid(handle) && store.GetActor(handle).bodyId == UINT32_MAX)
            store.RegisterActor(handle);
    for (const auto id : store.ConsumePendingBroadphaseRemoves())
        RemoveBodyFromBroadphase(id);
    AddBodiesBatch(store.ConsumePendingBroadphaseAdds());
}
bool PhysicsWorld::HasPendingStep() const noexcept
{
    return m_stepPending;
}
bool PhysicsWorld::HasPendingControl() const noexcept
{
    const auto &store = PhysicsECSStore::Instance();
    return m_controlPending || (m_gpuScene && m_gpuScene->HasPendingControl()) || store.HasDirtyColliders() ||
           store.HasPendingBodyCreations() || store.HasPendingBroadphaseAdds() || store.HasPendingBroadphaseRemoves() ||
           (m_gpuScene && !m_gpuCancelled && PhysicMaterial::GetSurfaceRevision() != m_surfaceRevision);
}
void PhysicsWorld::DrainGpuSource() const
{
    AdmissionScope admission(*m_gpuScene);
    m_gpuWork->DrainRequired(
        m_gpuScene->GetSourceId(), [this] { return Availability(*m_gpuHost); },
        [this](FrameWorkOwner::SourceId source, uint64_t maximum) {
            return source == m_gpuScene->GetSourceId() ? m_gpuScene->WaitForProgress(maximum)
                                                       : FrameWorkWaitResult::Blocked;
        });
    if (m_gpuScene->GetTickStatus() != JoltGpuTick::Status::Complete)
        throw std::runtime_error("GPU physics did not complete its synchronous dependency prefix");
}
bool PhysicsWorld::AdvanceGpuJob(bool synchronous) const
{
    if (synchronous) {
        DrainGpuSource();
    } else {
        if (!m_gpuWork->IsActive() || m_gpuWork->IsLate())
            return false;
        AdmissionScope admission(*m_gpuScene);
        (void)m_gpuWork->AdvanceRequired(m_gpuScene->GetSourceId(), [this] { return Availability(*m_gpuHost); });
    }
    if (m_gpuScene->GetTickStatus() == JoltGpuTick::Status::Failed)
        throw std::runtime_error("GPU physics step failed; stop the runtime before restarting");
    return m_gpuScene->GetTickStatus() == JoltGpuTick::Status::Complete;
}
bool PhysicsWorld::BeginGpuControl(bool numericalChange, const JoltGpuScene::Consumer &consumer) const
{
    FrameWorkOwner::Scope preparation(*m_gpuWork);
    auto rendering = PrepareRenderConsumer(numericalChange);
    if (!numericalChange && !consumer && !rendering)
        return false;
    m_gpuScene->BeginControl(SceneManager::Instance().GetSimulationEpoch(), consumer, std::move(rendering));
    m_controlNumericalChange = numericalChange;
    m_controlPending = true;
    return true;
}
bool PhysicsWorld::AdvanceGpuControl(bool synchronous) const
{
    if (!m_controlPending)
        return true;
    if (!AdvanceGpuJob(synchronous))
        return false;
    if (!m_gpuScene->PublishCompletedControl())
        throw std::runtime_error("GPU control failed to publish its complete result");
    ObserveGpuPublication(m_controlNumericalChange);
    m_controlPending = false;
    m_controlNumericalChange = false;
    return true;
}
void PhysicsWorld::CompletePendingStep()
{
    RequireOwnerThread();
    RequireHealthyStep();
    if ((!m_stepPending || m_stepCompleted) && !m_controlPending)
        return;
    RequireGpu();
    try {
        if (m_stepPending && !m_stepCompleted)
            (void)AdvanceFixedStep(true);
        else
            (void)AdvanceGpuControl(true);
    } catch (...) {
        FailStep();
        throw;
    }
}
void PhysicsWorld::ServiceGpuWork()
{
    RequireOwnerThread();
    if (!m_initialized || !m_gpuScene || m_gpuCancelled || m_fixedGpuPhase == FixedGpuPhase::Failed)
        return;
    try {
        // A public host opportunity can continue this already admitted prefix
        // even when the renderer is waiting and skips Update/LateUpdate.
        (void)AdvanceGpuControl(false);
        if (!SceneManager::Instance().HasPendingFixedStep())
            SceneManager::Instance().PublishPhysicsStateToRenderer();
    } catch (...) {
        FailStep();
        throw;
    }
}
void PhysicsWorld::RefreshRenderPoses()
{
    RequireOwnerThread();
    if (!m_initialized || !m_gpuScene || !m_gpuRenderState || m_gpuCancelled ||
        m_fixedGpuPhase == FixedGpuPhase::Failed)
        return;
    ServiceGpuWork();
    if (m_controlPending || m_stepPending || SceneManager::Instance().HasPendingFixedStep())
        return;
    // This automatic publication uses the existing host allowance. It never
    // opens an opportunity or turns render extraction into a strict GPU wait.
    if (!m_gpuWork->IsActive() || !m_gpuWork->IsAccepting() || m_gpuWork->IsLate() || m_gpuWork->Remaining() <= 0 ||
        !m_gpuWork->ResourceRemaining(FrameWorkResource::Cpu))
        return;
    try {
        auto &rendering = *m_gpuRenderState;
        if (!HasPendingControl() && rendering.checkedKeys && *rendering.checkedKeys == GpuRenderState::CaptureKeys())
            return;
        {
            FrameWorkOwner::Scope preparation(*m_gpuWork);
            FlushAuthoring();
        }
        while (m_waitingForSlot) {
            (void)BeginGpuControl(true);
            if (!AdvanceGpuControl(false))
                return;
            RetryCapacityBodyCreations();
        }
        if (BeginGpuControl(m_gpuScene->HasPendingControl()))
            (void)AdvanceGpuControl(false);
    } catch (...) {
        FailStep();
        throw;
    }
}
void PhysicsWorld::CancelPendingStep() noexcept
{
    if (m_gpuRenderState) {
        m_gpuRenderState->bindings.Clear();
        m_gpuRenderState->pending.reset();
        m_gpuRenderState->ready.reset();
        m_gpuRenderState->checkedKeys.reset();
    }
    if (m_gpuScene) {
        if (!m_gpuCancelled) {
            m_recreateActors.clear();
            auto &store = PhysicsECSStore::Instance();
            store.ForEachAliveActor([&](PhysicsActorData &actor) {
                if (actor.bodyId == UINT32_MAX)
                    return;
                const auto handle = m_gpuScene->FindActor(actor.bodyId);
                if (!store.IsValid(handle) || &store.GetActor(handle) != &actor)
                    return;
                store.CancelBroadphaseAdd(actor.bodyId);
                store.CancelBroadphaseRemove(actor.bodyId);
                actor.bodyId = UINT32_MAX;
                actor.bodyInBroadphase = false;
                // Capacity was reserved before the GPU owner was constructed.
                m_recreateActors.push_back(handle);
            });
        }
        m_gpuScene->SetAdmissionEnabled(false);
        m_gpuScene->Cancel();
        m_gpuCancelled = true;
    }
    PhysicsECSStore::Instance().ForEachAliveRigidbody([](RigidbodyECSData &body) { body.physicsTargetSources = {}; });
    m_stepPending = m_stepCompleted = false;
    m_fixedGpuPhase = FixedGpuPhase::Idle;
    m_admittedParameters.reset();
    m_waitingForSlot = false;
    m_controlPending = m_controlNumericalChange = false;
    m_requestedMotion.clear();
}
void PhysicsWorld::FlushAuthoring() const
{
    if (m_flushingAuthoring)
        return;
    FlagScope flushing(m_flushingAuthoring);
    TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    SceneManager::Instance().EnsurePhysicsQueriesCurrent();
    if (m_gpuScene && !m_gpuCancelled)
        const_cast<PhysicsWorld *>(this)->RefreshGpuSurfaces();
}
void PhysicsWorld::RunControl(const JoltGpuScene::Consumer &consumer) const
{
    RequireOwnerThread();
    RequireHealthyStep();
    RequireGpu();
    if (m_synchronousRead)
        throw std::logic_error("Physics query cannot reenter its synchronous source");
    FlagScope reading(m_synchronousRead);
    try {
        // This also completes an earlier managed render-control before a new
        // query consumer or later author prefix can acquire the same tick.
        const_cast<PhysicsWorld *>(this)->CompletePendingStep();
        {
            FrameWorkOwner::Scope preparation(*m_gpuWork);
            FlushAuthoring();
        }
        while (m_waitingForSlot) {
            (void)BeginGpuControl(true);
            (void)AdvanceGpuControl(true);
            const_cast<PhysicsWorld *>(this)->RetryCapacityBodyCreations();
        }
        if (BeginGpuControl(m_gpuScene->HasPendingControl(), consumer))
            (void)AdvanceGpuControl(true);
    } catch (...) {
        const_cast<PhysicsWorld *>(this)->FailStep();
        throw;
    }
}

std::shared_ptr<JoltGpuQueryResult> PhysicsWorld::RunQuery(const std::vector<QueryInput> &requests) const
{
    JoltGpuPreparedQuery prepared;
    RunControl([&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        prepared = m_gpuQueries->Prepare(publication, requests, {},
                                         GetQueryGeneration() + static_cast<uint64_t>(m_controlNumericalChange));
        return std::move(prepared.work);
    });
    return std::move(prepared.result);
}

std::shared_ptr<JoltGpuQueryResult> PhysicsWorld::RunGeometryQuery(const QueryInput &request,
                                                                   const QueryPrimitive &other) const
{
    RequireOwnerThread();
    RequireHealthyStep();
    RequireGpu();
    if (m_synchronousRead)
        throw std::logic_error("Physics geometry query cannot reenter its synchronous source");
    FlagScope reading(m_synchronousRead);
    // Finish only a logical operation already owned by this source. Do not
    // flush or admit queued author input, and never start a new fixed step.
    const_cast<PhysicsWorld *>(this)->CompletePendingStep();
    JoltGpuPreparedQuery prepared;
    {
        FrameWorkOwner::Scope preparation(*m_gpuWork);
        prepared = m_gpuQueries->Prepare({}, {request}, {other}, 0);
        m_gpuScene->BeginWork(std::move(prepared.work));
    }
    (void)AdvanceGpuJob(true);
    return std::move(prepared.result);
}

std::vector<Collider *> PhysicsWorld::QueryColliders(const QueryInput &request) const
{
    auto result = RunQuery({request});
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    std::vector<Collider *> colliders;
    const auto &records = result->ReadHits();
    colliders.reserve(records.size());
    for (const auto &record : records)
        if (auto *collider = DecodeQueryHit(*result, record).collider)
            colliders.push_back(collider);
    return colliders;
}

RaycastHit PhysicsWorld::DecodeQueryHit(const JoltGpuQueryResult &result, const QueryHit &record) const
{
    RaycastHit hit;
    if (!record.meta.y)
        return hit;
    const auto &publication = *result.publication;
    if (record.meta.x >= publication.actors.size() || !publication.actors[record.meta.x] ||
        record.meta.z >= publication.leaves.size())
        throw std::runtime_error("GPU query address is absent from its retained publication");
    const auto &identity = *publication.actors[record.meta.x];
    const auto &address = publication.leaves[record.meta.z];
    if (address.bodySlot != record.meta.x || address.actorLeaf >= identity.leaves.size())
        throw std::runtime_error("GPU query leaf belongs to a different published actor");
    const auto &leaf = identity.leaves[address.actorLeaf];
    hit.bodyId = identity.bodyId;
    hit.subShapeId = leaf.subShapeId;
    hit.point = Vector(record.positionDistance);
    hit.normal = Vector(record.normalFraction);
    hit.distance = record.positionDistance.w;
    hit.target = leaf.surfaceTarget;
    hit.collider = leaf.surfaceTarget.GetCollider();
    hit.gameObject = leaf.surfaceTarget.GetGameObject();
    hit.gameObjectId = leaf.surfaceGameObjectId;
    hit.colliderId = leaf.colliderId;
    return hit;
}

uint32_t PhysicsWorld::CreateBody(PhysicsActorData &actor)
{
    RequireOwnerThread();
    auto &store = PhysicsECSStore::Instance();
    const auto handle = store.FindActor(actor.owner);
    if (!store.IsValid(handle))
        throw std::logic_error("GPU body creation requires an ECS actor");
    if (!EnsureGpuReady()) {
        store.QueueBodyCreation(handle);
        return UINT32_MAX;
    }
    TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    auto prepared = PrepareJoltGpuActor(actor, 0, 0);
    if (!prepared)
        return UINT32_MAX;
    if (m_gpuScene->GetBodyCount() >= m_gpuScene->GetCapacity())
        throw std::length_error("GPU physics world exceeds its current 32-body capacity");
    if (!m_gpuScene->CanReserve()) {
        m_waitingForSlot = true;
        store.QueueBodyCreation(handle);
        return UINT32_MAX;
    }
    const auto allocation =
        m_gpuScene->Reserve(handle, {PhysicsTargetReference(actor.owner->GetTransform()), actor.owner->GetID()});
    actor.bodyId = allocation.bodyId;
    try {
        m_gpuScene->StageActor(allocation.bodyId, std::move(*prepared));
    } catch (...) {
        actor.bodyId = UINT32_MAX;
        m_gpuScene->StageRetire(allocation.bodyId);
        throw;
    }
    return allocation.bodyId;
}
void PhysicsWorld::DestroyBody(uint32_t id)
{
    RequireOwnerThread();
    if (!m_gpuScene || m_gpuScene->FindBody(id).bodyId == UINT32_MAX)
        return;
    m_gpuScene->StageRetire(id);
    if (m_gpuRenderState)
        m_gpuRenderState->bindings.RetireBody(id);
    m_requestedMotion.erase(id);
}
void PhysicsWorld::RefreshGpuActor(uint32_t id, const std::function<void(PreparedGpuActor &)> &edit)
{
    RequireGpu();
    auto &actor = ActorForBody(*m_gpuScene, id);
    TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    auto prepared = PrepareJoltGpuActor(actor, m_gpuScene->FindBody(id).slot, 0);
    if (!prepared)
        return;
    if (edit)
        edit(*prepared);
    m_gpuScene->StageActor(id, std::move(*prepared));
}
void PhysicsWorld::UpdateBodyShape(Collider *collider, const Collider *exclude)
{
    if (!collider)
        throw std::invalid_argument("Physics shape update requires a Collider");
    auto &store = PhysicsECSStore::Instance();
    const auto handle = store.GetCollider(collider->GetECSHandle()).actorHandle;
    if (store.IsValid(handle))
        UpdateBodyShape(store.GetActor(handle), exclude);
}
void PhysicsWorld::UpdateBodyShape(PhysicsActorData &actor, const Collider *exclude)
{
    RequireGpu();
    if (!exclude) {
        RefreshGpuActor(actor.bodyId);
        return;
    }
    auto author = actor;
    author.colliders.erase(std::remove(author.colliders.begin(), author.colliders.end(), exclude),
                           author.colliders.end());
    TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    auto prepared = PrepareJoltGpuActor(author, m_gpuScene->FindBody(actor.bodyId).slot, 0);
    if (prepared)
        m_gpuScene->StageActor(actor.bodyId, std::move(*prepared));
}
void PhysicsWorld::SetBodyPosition(uint32_t id, const glm::vec3 &position, const glm::quat &rotation,
                                   bool cancelKinematicMove)
{
    RequireGpu();
    BodyCommand c{};
    c.meta.y = CommandSetPose;
    c.meta.z = cancelKinematicMove ? 0 : CommandPreserveKinematicMove;
    c.value = Vector(position);
    c.auxiliary = Rotation(rotation);
    m_gpuScene->QueueCommand(id, c);
    m_gpuRenderState->bindings.MarkDiscontinuity(id);
}
void PhysicsWorld::SetBodyPositionOnly(uint32_t id, const glm::vec3 &position)
{
    RequireGpu();
    BodyCommand c{};
    c.meta.y = CommandSetPosition;
    c.value = Vector(position);
    m_gpuScene->QueueCommand(id, c);
    m_gpuRenderState->bindings.MarkDiscontinuity(id);
}
void PhysicsWorld::SetBodyRotationOnly(uint32_t id, const glm::quat &rotation)
{
    RequireGpu();
    BodyCommand c{};
    c.meta.y = CommandSetRotation;
    c.auxiliary = Rotation(rotation);
    m_gpuScene->QueueCommand(id, c);
    m_gpuRenderState->bindings.MarkDiscontinuity(id);
}
void PhysicsWorld::SetBodyPositionsBatch(const std::vector<PhysicsBodyPoseUpdate> &updates)
{
    if (updates.empty())
        return;
    RequireGpu();
    for (const auto &u : updates) {
        (void)ActorForBody(*m_gpuScene, u.bodyId);
        (void)Vector(u.position);
        (void)Rotation(u.rotation);
    }
    for (const auto &u : updates)
        SetBodyPosition(u.bodyId, u.position, u.rotation);
}
void PhysicsWorld::AddBodiesBatch(const std::vector<uint32_t> &bodies)
{
    if (bodies.empty())
        return;
    RequireGpu();
    for (const auto id : bodies)
        m_gpuScene->SetResident(id, true);
    for (const auto id : bodies)
        if (auto *body = FindRigidbodyByBodyId(id))
            body->FlushPendingForceCommands();
}
void PhysicsWorld::RemoveBodyFromBroadphase(uint32_t id, bool preserveVelocity)
{
    if (!m_gpuScene || m_gpuScene->FindBody(id).bodyId == UINT32_MAX)
        return;
    m_gpuScene->SetResident(id, false, preserveVelocity);
}
bool PhysicsWorld::IsBodyInBroadphase(uint32_t id) const
{
    if (id == UINT32_MAX)
        return false;
    RunControl({});
    if (m_gpuScene->FindBody(id).bodyId == UINT32_MAX)
        return false;
    const auto &actor = ActorForBody(*m_gpuScene, id);
    return actor.bodyInBroadphase && !actor.sceneSuspended;
}
void PhysicsWorld::SetBodyMotionType(uint32_t id, int type)
{
    if (type < 0 || type > 2)
        throw std::invalid_argument("Physics motion type must be static, kinematic or dynamic");
    RefreshGpuActor(id);
}
void PhysicsWorld::RefreshBodyLayers(uint32_t id)
{
    RefreshGpuActor(id);
}
void PhysicsWorld::SetBodyDamping(uint32_t id, float linear, float angular)
{
    if (!std::isfinite(linear) || !std::isfinite(angular) || linear < 0 || angular < 0)
        throw std::invalid_argument("Damping must be finite and nonnegative");
    RefreshGpuActor(id, [=](PreparedGpuActor &a) {
        a.body.damping.x = linear;
        a.body.damping.y = angular;
    });
}
void PhysicsWorld::SetBodyGravityFactor(uint32_t id, float factor)
{
    if (!std::isfinite(factor))
        throw std::invalid_argument("Gravity factor must be finite");
    RefreshGpuActor(id, [=](PreparedGpuActor &a) { a.body.mass.w = factor; });
}
void PhysicsWorld::SetBodyFriction(uint32_t id, float value)
{
    RefreshGpuActor(id, [=](PreparedGpuActor &a) {
        for (auto &leaf : a.geometry.leaves)
            leaf.model.material.x = value;
    });
}
void PhysicsWorld::SetBodyRestitution(uint32_t id, float value)
{
    RefreshGpuActor(id, [=](PreparedGpuActor &a) {
        for (auto &leaf : a.geometry.leaves)
            leaf.model.material.y = value;
    });
}
void PhysicsWorld::SetBodyMaxAngularVelocity(uint32_t id, float value)
{
    RefreshGpuActor(id, [=](PreparedGpuActor &a) { a.body.damping.w = value; });
}
void PhysicsWorld::SetBodyMaxLinearVelocity(uint32_t id, float value)
{
    RefreshGpuActor(id, [=](PreparedGpuActor &a) { a.body.damping.z = value; });
}
void PhysicsWorld::SetBodyAllowedDOFs(uint32_t id, int allowed, float mass)
{
    if (allowed != 0x3f)
        Unsupported("locked rigidbody degrees of freedom");
    if (!std::isfinite(mass) || mass <= 0)
        throw std::invalid_argument("Body mass must be positive and finite");
    RefreshGpuActor(id);
}
void PhysicsWorld::SetBodyMotionQuality(uint32_t, int quality)
{
    if (quality != 0)
        Unsupported("continuous collision detection");
}
void PhysicsWorld::ValidateBodyMassProperties(const Rigidbody &body, float mass,
                                              const RigidbodyMassProperties *properties) const
{
    RequireOwnerThread();
    TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    auto *owner = body.GetGameObject();
    const auto scale = owner ? owner->GetTransform()->GetWorldScale() : glm::vec3(1);
    if (properties) {
        (void)PrepareBodyMass(nullptr, mass, properties, scale);
        return;
    }
    const auto &store = PhysicsECSStore::Instance();
    const auto handle = store.FindActor(owner);
    if (!store.IsValid(handle))
        return;
    auto shape = BuildActorShape(store.GetActor(handle));
    if (shape)
        (void)PrepareBodyMass(shape.GetPtr(), mass, nullptr, scale);
}
void PhysicsWorld::SetBodyLinearVelocity(uint32_t id, const glm::vec3 &v)
{
    RequireGpu();
    BodyCommand c{};
    c.meta.y = CommandSetLinearVelocity;
    c.value = Vector(v);
    m_gpuScene->QueueCommand(id, c);
}
void PhysicsWorld::SetBodyAngularVelocity(uint32_t id, const glm::vec3 &v)
{
    RequireGpu();
    BodyCommand c{};
    c.meta.y = CommandSetAngularVelocity;
    c.value = Vector(v);
    m_gpuScene->QueueCommand(id, c);
}
void PhysicsWorld::QueueBodyForce(uint32_t id, int kind, int mode, const glm::vec3 &value, const glm::vec3 &point)
{
    RequireGpu();
    if (kind < 0 || kind > 2 || mode < 0 || mode > 3)
        throw std::invalid_argument("Invalid physics force kind or mode");
    BodyCommand c{};
    c.meta.y = kind == 0 ? CommandAddForce : kind == 1 ? CommandAddTorque : CommandAddForceAtPosition;
    c.meta.z = static_cast<uint32_t>(mode);
    c.value = Vector(value);
    if (kind == 2)
        c.auxiliary = Vector(point);
    m_gpuScene->QueueCommand(id, c);
}
void PhysicsWorld::AddBodyForce(uint32_t id, const glm::vec3 &v)
{
    QueueBodyForce(id, 0, 0, v);
}
void PhysicsWorld::AddBodyImpulse(uint32_t id, const glm::vec3 &v)
{
    QueueBodyForce(id, 0, 2, v);
}
void PhysicsWorld::AddBodyTorque(uint32_t id, const glm::vec3 &v)
{
    QueueBodyForce(id, 1, 0, v);
}
void PhysicsWorld::AddBodyAngularImpulse(uint32_t id, const glm::vec3 &v)
{
    QueueBodyForce(id, 1, 2, v);
}
void PhysicsWorld::AddBodyForceAtPosition(uint32_t id, const glm::vec3 &v, const glm::vec3 &p)
{
    QueueBodyForce(id, 2, 0, v, p);
}
void PhysicsWorld::AddBodyImpulseAtPosition(uint32_t id, const glm::vec3 &v, const glm::vec3 &p)
{
    QueueBodyForce(id, 2, 2, v, p);
}
void PhysicsWorld::RequireResetPhase() const
{
    RequireOwnerThread();
    RequireGpu();
    if (HasPendingStep())
        throw std::logic_error("Physics reset requires a boundary between completed fixed steps");
}
void PhysicsWorld::ResetBodyMotionStates(const std::vector<PhysicsBodyMotionState> &states)
{
    if (states.empty())
        return;
    RequireResetPhase();
    for (const auto &s : states) {
        (void)ActorForBody(*m_gpuScene, s.bodyId);
        (void)Vector(s.position);
        (void)Rotation(s.rotation);
        (void)Vector(s.linearVelocity);
        (void)Vector(s.angularVelocity);
    }
    for (const auto &s : states) {
        BodyCommand c{};
        c.meta.y = CommandResetMotion;
        m_gpuScene->QueueCommand(s.bodyId, c);
        SetBodyPosition(s.bodyId, s.position, s.rotation);
        SetBodyLinearVelocity(s.bodyId, s.linearVelocity);
        SetBodyAngularVelocity(s.bodyId, s.angularVelocity);
    }
}

PhysicsBodyMotionState PhysicsWorld::ReadMotionState(uint32_t id, PhysicsStateFields fields) const
{
    RequireHealthyStep();
    RequireGpu();
    if (HasPhysicsStateField(fields, PhysicsStateFields::InverseInertia))
        Unsupported("GPU inertia-tensor state gathering");
    const bool wantMass = HasPhysicsStateField(fields, PhysicsStateFields::InverseMass);
    const auto cached = m_requestedMotion.find(id);
    if ((!m_stepPending || m_stepCompleted) && !HasPendingControl() && cached != m_requestedMotion.end() &&
        cached->second.generation == GetQueryGeneration() && (!wantMass || cached->second.hasInverseMass))
        return cached->second.state;
    BodyState state{};
    BodyModel model{};
    std::shared_ptr<rhi::PreparedComputeBatch> packet;
    std::shared_ptr<const JoltGpuScene::Publication> retainedPublication;
    RunControl([&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        retainedPublication = publication;
        const auto allocation = m_gpuScene->FindBody(id);
        if (allocation.bodyId == UINT32_MAX || allocation.slot >= publication->actors.size() ||
            !publication->actors[allocation.slot] || publication->actors[allocation.slot]->bodyId != id)
            throw std::invalid_argument("Physics state requires an accepted live body");
        std::vector<rhi::ComputeBufferRead> reads{
            {publication->bodyStates, uint64_t(allocation.slot) * sizeof(BodyState), sizeof(BodyState)}};
        if (wantMass)
            reads.push_back(
                {publication->bodyModels, uint64_t(allocation.slot) * sizeof(BodyModel), sizeof(BodyModel)});
        packet = rhi::PrepareComputeBatch(*m_gpuHost, {}, {}, std::move(reads));
        return packet;
    });
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    packet->Readback(0)->ReadInto(&state, sizeof(state));
    if (wantMass)
        packet->Readback(1)->ReadInto(&model, sizeof(model));
    const auto allocation = m_gpuScene->FindBody(id);
    const auto &identity = *retainedPublication->actors[allocation.slot];
    PhysicsBodyMotionState result;
    result.bodyId = id;
    result.shapeRevision = identity.shapeRevision;
    result.rotation = {state.rotation.w, state.rotation.x, state.rotation.y, state.rotation.z};
    result.centerOfMass = Vector(state.positionCOM);
    // CPU Transform keeps body-origin/hierarchy semantics. This converts the
    // actual completed device pose; no CPU integration or collision runs here.
    result.position = result.centerOfMass - result.rotation * Vector(identity.rootCenterOfMass);
    result.linearVelocity = Vector(state.linearVelocity);
    result.angularVelocity = Vector(state.angularVelocity);
    if (wantMass && (model.meta.z & ~MotionNotResident) == MotionDynamic)
        result.inverseMass = glm::vec3(model.mass.x);
    m_requestedMotion[id] = {GetQueryGeneration(), result, wantMass};
    return result;
}
PhysicsBodyMotionState PhysicsWorld::GetBodyMotionState(uint32_t id, PhysicsStateFields fields) const
{
    return ReadMotionState(id, fields);
}
glm::vec3 PhysicsWorld::GetBodyPosition(uint32_t id) const
{
    return ReadMotionState(id, PhysicsStateFields::Position).position;
}
glm::quat PhysicsWorld::GetBodyRotation(uint32_t id) const
{
    return ReadMotionState(id, PhysicsStateFields::Rotation).rotation;
}
glm::vec3 PhysicsWorld::GetBodyCenterOfMassPosition(uint32_t id) const
{
    return ReadMotionState(id, PhysicsStateFields::CenterOfMass).centerOfMass;
}
glm::vec3 PhysicsWorld::GetBodyLinearVelocity(uint32_t id) const
{
    return ReadMotionState(id, PhysicsStateFields::LinearVelocity).linearVelocity;
}
glm::vec3 PhysicsWorld::GetBodyAngularVelocity(uint32_t id) const
{
    return ReadMotionState(id, PhysicsStateFields::AngularVelocity).angularVelocity;
}
glm::mat3 PhysicsWorld::GetBodyWorldSpaceInertiaTensor(uint32_t) const
{
    Unsupported("GPU world inertia-tensor query");
}

bool PhysicsWorld::PrepareRaycastBatchQuery() const
{
    RequireOwnerThread();
    RunControl({});
    return true;
}
void PhysicsWorld::RaycastBatch(const float *origins, const float *directions, size_t count, float maximum,
                                RaycastHit *hits, uint8_t *mask, uint32_t layers, bool triggers, uint64_t *generation,
                                RaycastBatchProfile *profile) const
{
    RaycastBatchPublished(origins, directions, count, maximum, hits, mask, layers, triggers, generation, profile);
}
void PhysicsWorld::RaycastBatchPublished(const float *origins, const float *directions, size_t count, float maximum,
                                         RaycastHit *hits, uint8_t *mask, uint32_t layers, bool triggers,
                                         uint64_t *generation, RaycastBatchProfile *profile, bool) const
{
    if (profile)
        *profile = {};
    if (count == 0) {
        if (generation)
            *generation = GetQueryGeneration();
        return;
    }
    RequireOwnerThread();
    if (!origins || !directions || !hits || !mask || count > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Raycast batch requires valid arrays within the device index range");
    RequireGpu();
    std::vector<QueryInput> requests;
    {
        FrameWorkOwner::Scope preparation(*m_gpuWork);
        requests.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const auto offset = i * 3;
            auto request = Query(QueryRay, QueryClosest, layers, triggers);
            QueryRayInput(request, {origins[offset], origins[offset + 1], origins[offset + 2]},
                          {directions[offset], directions[offset + 1], directions[offset + 2]}, maximum);
            requests.push_back(request);
        }
    }
    const auto result = RunQuery(requests);
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    const auto &records = result->ReadHits();
    if (records.size() != count)
        throw std::runtime_error("GPU closest query did not produce one complete record per request");
    for (size_t i = 0; i < count; ++i) {
        hits[i] = DecodeQueryHit(*result, records[i]);
        mask[i] = static_cast<uint8_t>(records[i].meta.y != 0);
    }
    if (generation)
        *generation = result->numericalGeneration;
}

bool PhysicsWorld::Raycast(const glm::vec3 &origin, const glm::vec3 &direction, float maximum, RaycastHit &hit,
                           uint32_t layers, bool triggers) const
{
    uint8_t mask = 0;
    const float origins[]{origin.x, origin.y, origin.z}, directions[]{direction.x, direction.y, direction.z};
    RaycastBatch(origins, directions, 1, maximum, &hit, &mask, layers, triggers);
    return mask != 0;
}

Rigidbody *PhysicsWorld::FindRigidbodyByBodyId(uint32_t id) const
{
    if (!m_gpuScene || m_gpuScene->FindBody(id).bodyId == UINT32_MAX)
        return nullptr;
    auto &store = PhysicsECSStore::Instance();
    const auto handle = m_gpuScene->FindActor(id);
    return store.IsValid(handle) && store.GetActor(handle).bodyId == id ? store.GetActor(handle).rigidbody : nullptr;
}
Collider *PhysicsWorld::FindColliderByBodyId(uint32_t id) const
{
    if (!m_gpuScene || !m_gpuScene->GetPublished())
        return nullptr;
    for (const auto &actor : m_gpuScene->GetPublished()->actors)
        if (actor && actor->bodyId == id)
            for (const auto &leaf : actor->leaves)
                if (auto *collider = leaf.surfaceTarget.GetCollider())
                    return collider;
    return nullptr;
}
Collider *PhysicsWorld::ResolveColliderForSubShape(uint32_t id, uint32_t subShape) const
{
    if (!m_gpuScene || !m_gpuScene->GetPublished())
        return nullptr;
    for (const auto &actor : m_gpuScene->GetPublished()->actors)
        if (actor && actor->bodyId == id)
            for (const auto &leaf : actor->leaves) {
                const auto prefixMask = leaf.subShapeBits == 32 ? UINT32_MAX : (uint32_t(1) << leaf.subShapeBits) - 1;
                if ((leaf.subShapeId & prefixMask) == (subShape & prefixMask))
                    return leaf.surfaceTarget.GetCollider();
            }
    return nullptr;
}
void PhysicsWorld::EnsureSceneBodiesRegistered(Scene *)
{
    RequireOwnerThread();
    FlushAuthoring();
}
void PhysicsWorld::OptimizeBroadPhase()
{
    RequireOwnerThread();
} // Bounded GPU pairs have no persistent CPU tree.
const std::vector<uint32_t> &PhysicsWorld::GetPoseReadbackBodyIds() const
{
    if (GetBodyCount())
        Unsupported("automatic bulk CPU pose readback");
    static const std::vector<uint32_t> empty;
    return empty;
}
bool PhysicsWorld::HasRagdollAssembly(uint64_t) const
{
    return false;
}
void PhysicsWorld::SetContactEventStreamEnabled(bool enabled, bool)
{
    if (enabled)
        Unsupported("contact event stream");
    m_contactEventStreamEnabled = false;
}
void PhysicsWorld::SetContactImpulseStreamEnabled(bool enabled)
{
    if (enabled)
        Unsupported("contact impulse stream");
    m_contactImpulseStreamEnabled = false;
}
const std::vector<ContactEvent> &PhysicsWorld::GetContactEvents() const
{
    Unsupported("contact events");
}
const std::vector<ContactImpulse> &PhysicsWorld::GetContactImpulses() const
{
    Unsupported("contact impulses");
}
size_t PhysicsWorld::DispatchContactEvents()
{
    CheckContactSupport();
    return 0;
}
void PhysicsWorld::RefreshBodyTriggerState(uint32_t id)
{
    RefreshGpuActor(id);
}
void PhysicsWorld::InvalidateContactPairsForBody(uint32_t)
{
    RequireOwnerThread();
} // No retained GPU pair cache yet.
bool PhysicsWorld::IsBodySensor(uint32_t id) const
{
    if (!m_initialized || id == UINT32_MAX)
        return false;
    RunControl({});
    const auto body = m_gpuScene->FindBody(id);
    if (body.bodyId == UINT32_MAX)
        return false;
    const auto &actor = m_gpuScene->GetPublished()->actors[body.slot];
    return actor && !actor->leaves.empty() &&
           std::all_of(actor->leaves.begin(), actor->leaves.end(), [](const auto &leaf) { return leaf.sensor; });
}
void PhysicsWorld::SetColliderPairIgnored(Collider *, Collider *, bool)
{
    Unsupported("per-Collider collision suppression");
}
bool PhysicsWorld::GetColliderPairIgnored(const Collider *, const Collider *) const
{
    Unsupported("per-Collider collision suppression");
}
void PhysicsWorld::RemoveIgnoredPairsForCollider(const Collider *)
{
} // No suppression resource can be created yet.

void PhysicsWorld::MoveBodyKinematic(uint32_t id, const glm::vec3 &position, const glm::quat &rotation, float dt,
                                     float maxSpeed)
{
    RequireGpu();
    (void)KinematicBody(*m_gpuScene, id);
    m_gpuScene->QueueCommand(id, MoveCommand(CommandMovePose, position, rotation, dt, maxSpeed));
}
void PhysicsWorld::MoveBodyKinematicPosition(uint32_t id, const glm::vec3 &position, float dt)
{
    RequireGpu();
    (void)KinematicBody(*m_gpuScene, id);
    m_gpuScene->QueueCommand(id, MoveCommand(CommandMovePosition, position, glm::quat(1, 0, 0, 0), dt));
}
void PhysicsWorld::MoveBodyKinematicRotation(uint32_t id, const glm::quat &rotation, float dt)
{
    RequireGpu();
    (void)KinematicBody(*m_gpuScene, id);
    m_gpuScene->QueueCommand(id, MoveCommand(CommandMoveRotation, glm::vec3(0), rotation, dt));
}
PhysicsTargetSource PhysicsWorld::MoveBodyKinematicAuthored(uint32_t id, const glm::vec3 &position,
                                                            const glm::quat &rotation, uint32_t channels, float dt,
                                                            float maxSpeed,
                                                            const std::array<uint64_t, 2> &authoredRevision)
{
    RequireGpu();
    auto &body = KinematicBody(*m_gpuScene, id);
    constexpr uint32_t positionChannel = static_cast<uint32_t>(PhysicsTargetChannels::Position);
    constexpr uint32_t rotationChannel = static_cast<uint32_t>(PhysicsTargetChannels::Rotation);
    if (channels == 0 || (channels & ~(positionChannel | rotationChannel)) != 0)
        throw std::invalid_argument("Authored kinematic movement requires position or rotation channels");
    const auto kind = channels == positionChannel   ? CommandMovePosition
                      : channels == rotationChannel ? CommandMoveRotation
                                                    : CommandMovePose;
    PhysicsTargetAuthor author;
    author.rigidbody = PhysicsTargetReference(&body);
    author.ownerTransform = PhysicsTargetReference(body.GetGameObject()->GetTransform());
    author.bodyId = id;
    author.positionRevision = authoredRevision[0];
    author.rotationRevision = authoredRevision[1];
    author.channels = static_cast<uint8_t>(channels);
    return m_gpuScene->QueueCommand(id, MoveCommand(kind, position, rotation, dt, maxSpeed), &author);
}
void PhysicsWorld::MoveStaticBodyWithVelocity(uint32_t id, const glm::vec3 &position, const glm::quat &rotation,
                                              float dt)
{
    RequireGpu();
    const auto &actor = ActorForBody(*m_gpuScene, id);
    if (actor.rigidbody && actor.rigidbody->IsEnabled())
        throw std::logic_error("Static movement requires a collider-only physics actor");
    auto command = MoveCommand(CommandMovePose, position, rotation, dt, kMaxTransformDriveSpeed);
    command.meta.z = CommandTemporaryStatic;
    m_gpuScene->QueueCommand(id, command);
}
bool PhysicsWorld::IsBodySleeping(uint32_t) const
{
    Unsupported("sleep state");
}
void PhysicsWorld::ActivateBody(uint32_t)
{
    Unsupported("body sleep/wake control");
}
void PhysicsWorld::DeactivateBody(uint32_t)
{
    Unsupported("body sleep/wake control");
}
void PhysicsWorld::ActivateBodiesInAABB(const glm::vec3 &, const glm::vec3 &)
{
    Unsupported("bounds sleep/wake control");
}
void PhysicsWorld::WakeBodiesTouchingStatic(uint32_t)
{
    Unsupported("static-neighbour sleep/wake control");
}

std::optional<PhysicsPenetrationResult> PhysicsWorld::ComputePenetration(const Collider &a, const glm::vec3 &positionA,
                                                                         const glm::quat &rotationA, const Collider &b,
                                                                         const glm::vec3 &positionB,
                                                                         const glm::quat &rotationB) const
{
    RequireOwnerThread();
    const auto cook = [](const Collider &collider, const glm::vec3 &position,
                         const glm::quat &rotation) -> std::optional<QueryPrimitive> {
        (void)Vector(position);
        const auto q = Rotation(rotation);
        const auto shape = collider.CreateJoltShape();
        if (!shape)
            throw std::runtime_error("Collider geometry is not ready for an explicit penetration query");
        const auto prepared = PrepareJoltGpuShape(*shape, 1);
        if (prepared.leaves.empty())
            return std::nullopt;
        if (prepared.leaves.size() != 1)
            throw std::runtime_error("GPU penetration for a multi-leaf Collider requires a union separation query");
        const auto &leaf = prepared.leaves.front().model;
        const glm::quat world(q.w, q.x, q.y, q.z);
        const glm::quat local(leaf.localRotation.w, leaf.localRotation.x, leaf.localRotation.y, leaf.localRotation.z);
        const auto center = position + world * (glm::vec3(prepared.centerOfMass.GetX(), prepared.centerOfMass.GetY(),
                                                          prepared.centerOfMass.GetZ()) +
                                                Vector(leaf.localPositionCOM));
        return QueryPrimitive{{leaf.meta.x, 0, 0, 0}, leaf.geometry, Vector(center), Rotation(world * local)};
    };
    std::optional<QueryPrimitive> first, second;
    {
        TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
        first = cook(a, positionA, rotationA);
        second = cook(b, positionB, rotationB);
    }
    if (!first || !second)
        return std::nullopt;
    auto request = Query(QueryPenetration, QueryClosest);
    request.meta.z = first->meta.x;
    request.positionMaximum = first->position;
    request.rotation = first->rotation;
    request.geometry = first->geometry;
    auto result = RunGeometryQuery(request, *second);
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    const auto &records = result->ReadHits();
    if (records.empty() || !records.front().meta.y)
        return std::nullopt;
    const auto &hit = records.front();
    return PhysicsPenetrationResult{Vector(hit.normalFraction), hit.pointADepth.w, Vector(hit.pointADepth),
                                    Vector(hit.positionDistance)};
}
bool PhysicsWorld::RaycastCollider(const Collider &collider, const glm::vec3 &origin, const glm::vec3 &direction,
                                   float maximum, RaycastHit &hit) const
{
    auto request = Query(QueryRay, QueryClosest);
    QueryRayInput(request, origin, direction, maximum);
    TargetCollider(request, collider);
    auto result = RunQuery({request});
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    const auto &record = result->ReadHits().front();
    hit = DecodeQueryHit(*result, record);
    return record.meta.y != 0;
}
glm::vec3 PhysicsWorld::ClosestPointOnCollider(const Collider &collider, const glm::vec3 &point) const
{
    (void)Vector(point);
    if (!collider.IsEnabled() || !collider.GetGameObject() || !collider.GetGameObject()->IsActiveInHierarchy())
        return point;
    auto request = Query(QueryClosestPoint, QueryClosest);
    request.positionMaximum = Vector(point);
    TargetCollider(request, collider);
    auto result = RunQuery({request});
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    const auto &record = result->ReadHits().front();
    if (!record.meta.y)
        throw std::runtime_error("Active Collider has no accepted geometry for a closest-point query");
    return Vector(record.positionDistance);
}
std::vector<RaycastHit> PhysicsWorld::RaycastAll(const glm::vec3 &origin, const glm::vec3 &direction, float maximum,
                                                 uint32_t layers, bool triggers) const
{
    auto request = Query(QueryRay, QueryAll, layers, triggers);
    QueryRayInput(request, origin, direction, maximum);
    auto result = RunQuery({request});
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    std::vector<RaycastHit> hits;
    const auto &records = result->ReadHits();
    hits.reserve(records.size());
    for (const auto &record : records)
        hits.push_back(DecodeQueryHit(*result, record));
    return hits;
}
std::vector<Collider *> PhysicsWorld::OverlapSphere(const glm::vec3 &center, float radius, uint32_t layers,
                                                    bool triggers) const
{
    auto request = Query(QueryOverlap, QueryUniqueCollider, layers, triggers);
    QuerySphereInput(request, center, radius);
    return QueryColliders(request);
}
std::vector<Collider *> PhysicsWorld::OverlapBox(const glm::vec3 &center, const glm::vec3 &halfExtents,
                                                 const glm::quat &rotation, uint32_t layers, bool triggers) const
{
    auto request = Query(QueryOverlap, QueryUniqueCollider, layers, triggers);
    QueryBoxInput(request, center, halfExtents, rotation);
    return QueryColliders(request);
}
std::vector<Collider *> PhysicsWorld::OverlapCapsule(const glm::vec3 &a, const glm::vec3 &b, float radius,
                                                     uint32_t layers, bool triggers) const
{
    auto request = Query(QueryOverlap, QueryUniqueCollider, layers, triggers);
    QueryCapsuleInput(request, a, b, radius);
    return QueryColliders(request);
}
std::vector<Rigidbody *> PhysicsWorld::QueryRigidbodiesInBounds(const glm::vec3 &minimum, const glm::vec3 &maximum,
                                                                uint32_t layers, bool triggers) const
{
    (void)Vector(minimum);
    (void)Vector(maximum);
    if (glm::any(glm::greaterThan(minimum, maximum)))
        throw std::invalid_argument("Physics query bounds require minimum <= maximum");
    auto request = Query(QueryBodyBounds, QueryUniqueBody, layers, triggers);
    request.positionMaximum = Vector(glm::vec3((glm::dvec3(minimum) + glm::dvec3(maximum)) * 0.5));
    request.direction = Vector(glm::vec3((glm::dvec3(maximum) - glm::dvec3(minimum)) * 0.5));
    auto result = RunQuery({request});
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    auto &store = PhysicsECSStore::Instance();
    std::vector<Rigidbody *> bodies;
    for (const auto &record : result->ReadHits()) {
        const auto &identity = *result->publication->actors[record.meta.x];
        if (store.IsValid(identity.rigidbody))
            bodies.push_back(store.GetRigidbody(identity.rigidbody).owner);
    }
    return bodies;
}
bool PhysicsWorld::SphereCast(const glm::vec3 &, float, const glm::vec3 &, float, RaycastHit &, uint32_t, bool) const
{
    Unsupported("sphere cast");
}
bool PhysicsWorld::BoxCast(const glm::vec3 &, const glm::vec3 &, const glm::vec3 &, const glm::quat &, float,
                           RaycastHit &, uint32_t, bool) const
{
    Unsupported("box cast");
}
bool PhysicsWorld::SweepBox(const BoxCollider &, const glm::vec3 &, const glm::quat &, const glm::vec3 &,
                            RaycastHit &) const
{
    Unsupported("BoxCollider sweep");
}
bool PhysicsWorld::CapsuleCast(const glm::vec3 &, const glm::vec3 &, float, const glm::vec3 &, float, RaycastHit &,
                               uint32_t, bool) const
{
    Unsupported("capsule cast");
}
std::vector<PhysicsShapeState> PhysicsWorld::GetBodyShapeStates(uint32_t, bool) const
{
    Unsupported("GPU world shape-state gathering");
}
std::vector<PhysicsBoxShapeState> PhysicsWorld::GetBodyBoxShapeStates(uint32_t, bool) const
{
    Unsupported("GPU box shape-state gathering");
}

std::vector<glm::vec3> PhysicsShapeGeometry::GetTriangles() const
{
    if (!shape)
        throw std::logic_error("Shape geometry has no immutable authored shape");
    JPH::Shape::GetTrianglesContext context;
    const JPH::Vec3 leafScale(scale.x, scale.y, scale.z);
    shape->GetTrianglesStart(context, shape->GetLocalBounds().Scaled(leafScale), JPH::Vec3::sZero(),
                             JPH::Quat::sIdentity(), leafScale);
    constexpr int batchSize = 256;
    JPH::Float3 vertices[3 * batchSize];
    std::vector<glm::vec3> result;
    for (int count; (count = shape->GetTrianglesNext(context, batchSize, vertices)) != 0;)
        for (int i = 0; i < count * 3; ++i)
            result.emplace_back(vertices[i].x, vertices[i].y, vertices[i].z);
    return result;
}

} // namespace infernux
