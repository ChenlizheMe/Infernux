#include "PhysicsWorld.h"
#include <Jolt/Jolt.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <Jolt/RegisterTypes.h>
#include "../BoxCollider.h"
#include "../Collider.h"
#include "../Component.h"
#include "../GameObject.h"
#include "../Rigidbody.h"
#include "../Scene.h"
#include "../SceneManager.h"
#include "../TagLayerManager.h"
#include "../Transform.h"
#include "../TransformECSStore.h"
#include "JoltGpuJoints.h"
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
#include <exception>
#include <function/renderer/particle/ParticleGpuCollisionScene.h>
#include <function/renderer/rhi/RhiComputeHost.h>
#include <function/renderer/rhi/RhiComputeKernel.h>
#include <function/resources/PhysicMaterial/PhysicMaterial.h>
#include <limits>
#include <map>
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
JoltGpuControlWork AfterWork(JoltGpuControlWork work, std::function<JoltGpuControlWork()> next)
{
    if (!work.packet && !work.continueWork)
        return next();
    auto continuation = std::move(work.continueWork);
    work.continueWork = [continuation = std::move(continuation), next = std::move(next)]() mutable {
        return continuation ? AfterWork(continuation(), std::move(next)) : next();
    };
    return work;
}
JoltGpuControlWork PrepareQueryWork(std::function<JoltGpuControlWork()> prepare, std::exception_ptr &error)
{
    try {
        auto work = prepare();
        if (auto continuation = std::move(work.continueWork))
            work.continueWork = [continuation = std::move(continuation), &error]() mutable {
                return PrepareQueryWork(std::move(continuation), error);
            };
        return work;
    } catch (const JoltGpuQueryRequestError &) {
        error = std::current_exception();
        return {};
    }
}
PhysicsBodyMotionState DecodeMotion(const JoltGpuScene::Publication &publication, const QueryMotionState &record)
{
    if (record.meta.x >= publication.actors.size() || !publication.actors[record.meta.x] ||
        publication.actors[record.meta.x]->bodyId != record.meta.z)
        throw std::runtime_error("GPU motion state differs from its retained body identity");
    const auto &actor = *publication.actors[record.meta.x];
    PhysicsBodyMotionState result;
    result.bodyId = actor.bodyId;
    result.shapeRevision = actor.shapeRevision;
    result.position = Vector(record.positionInverseMass);
    result.rotation = {record.rotation.w, record.rotation.x, record.rotation.y, record.rotation.z};
    result.centerOfMass = Vector(record.centerOfMass);
    result.linearVelocity = Vector(record.linearVelocity);
    result.angularVelocity = Vector(record.angularVelocity);
    result.inverseMass = glm::vec3(record.positionInverseMass.w);
    result.inverseInertia = {Vector(record.inverseInertia0), Vector(record.inverseInertia1),
                             Vector(record.inverseInertia2)};
    result.inertia = {Vector(record.inertia0), Vector(record.inertia1), Vector(record.inertia2)};
    return result;
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
std::optional<QueryPrimitive> CookColliderQueryPrimitive(const Collider &collider, const glm::vec3 &position,
                                                         const glm::quat &rotation)
{
    (void)Vector(position);
    const auto q = Rotation(rotation);
    // Collider dimensions and center offsets use authored affine transforms,
    // including scale and mirrors, without demanding a live device pose.
    TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    const auto shape = collider.CreateJoltShape();
    if (!shape)
        throw std::runtime_error("Collider geometry is not ready for an explicit geometry query");
    const auto prepared = PrepareJoltGpuShape(*shape, 1);
    if (prepared.leaves.empty())
        return std::nullopt;
    if (prepared.leaves.size() != 1)
        throw std::runtime_error("Explicit GPU Collider queries require a single cooked leaf");
    const auto &leaf = prepared.leaves.front().model;
    const glm::quat world(q.w, q.x, q.y, q.z);
    const glm::quat local(leaf.localRotation.w, leaf.localRotation.x, leaf.localRotation.y, leaf.localRotation.z);
    const auto center = position + world * (glm::vec3(prepared.centerOfMass.GetX(), prepared.centerOfMass.GetY(),
                                                      prepared.centerOfMass.GetZ()) +
                                            Vector(leaf.localPositionCOM));
    return QueryPrimitive{{leaf.meta.x, 0, 0, 0}, leaf.geometry, Vector(center), Rotation(world * local)};
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
    ShaderTargetCode step, mutations, commands, broadphase;
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
        p->broadphase = CompileProgram(*m_gpuHost, JoltGpuProgram::Broadphase);
        const auto target = CompileProgram(*m_gpuHost, JoltGpuProgram::Queries);
        auto queries = std::make_unique<JoltGpuQueries>(*m_gpuHost, target);
        auto joints = std::make_unique<JoltGpuJoints>(*m_gpuHost, CompileProgram(*m_gpuHost, JoltGpuProgram::Joints));
        auto rendering =
            std::make_unique<GpuRenderState>(*m_gpuHost, CompileProgram(*m_gpuHost, JoltGpuProgram::RenderPoses));
        m_gpuPrograms = std::move(p);
        m_gpuQueries = std::move(queries);
        m_gpuJoints = std::move(joints);
        m_gpuRenderState = std::move(rendering);
    }
    if (!m_gpuScene) {
        m_recreateActors.reserve(InitialBodyCapacity);
        m_gpuScene = std::make_unique<JoltGpuScene>(*m_gpuHost, *m_gpuWork, InitialBodyCapacity, m_gpuPrograms->step,
                                                    m_gpuPrograms->mutations, m_gpuPrograms->commands,
                                                    m_gpuPrograms->broadphase, *m_gpuJoints);
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
    m_gpuJoints.reset();
    m_gpuPrograms.reset();
    m_lastPublication.reset();
    m_requestedMotion.clear();
    m_recreateActors.clear();
    m_shapeGeometry.clear();
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
    store.DrainPendingJoints();
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
           store.HasPendingJointDependencies() ||
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
                actor.jointEndpointBodyId = UINT32_MAX;
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

std::shared_ptr<JoltGpuQueryResult> PhysicsWorld::RunQuery(const std::vector<QueryInput> &requests,
                                                           uint64_t sweepColliderId) const
{
    JoltGpuPreparedQuery prepared;
    std::exception_ptr requestError;
    RunControl([&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) -> JoltGpuControlWork {
        return PrepareQueryWork(
            [&] {
                const auto *inputs = &requests;
                std::vector<QueryInput> sweepInputs;
                if (sweepColliderId) {
                    sweepInputs = requests;
                    // Only this accepted publication can name the source slot.
                    // An unregistered Collider still excludes its GameObject.
                    for (const auto &actor : publication->actors) {
                        if (!actor || !std::any_of(actor->leaves.begin(), actor->leaves.end(), [&](const auto &leaf) {
                                return leaf.colliderId == sweepColliderId;
                            }))
                            continue;
                        for (auto &input : sweepInputs) {
                            input.source.x = actor->slot;
                            input.meta.w |= QueryExcludeBody;
                        }
                        break;
                    }
                    inputs = &sweepInputs;
                }
                prepared = m_gpuQueries->Prepare(
                    publication, *inputs, {}, GetQueryGeneration() + static_cast<uint64_t>(m_controlNumericalChange));
                return std::move(prepared.work);
            },
            requestError);
    });
    if (requestError)
        std::rethrow_exception(requestError);
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
    if (m_gpuScene->GetBodyCount() >= m_gpuScene->GetCapacity()) {
        const uint64_t capacity = std::max(uint64_t{InitialBodyCapacity}, uint64_t{2} * m_gpuScene->GetCapacity());
        if (capacity > UINT32_MAX)
            throw std::length_error("GPU physics world exceeds body indexing");
        // CancelPendingStep is noexcept and must not allocate while rebuilding
        // the complete planned set, including a growth not yet accepted.
        m_recreateActors.reserve(static_cast<size_t>(capacity));
        m_gpuScene->ReserveCapacity(static_cast<uint32_t>(capacity));
    }
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

JoltGpuControlWork
PhysicsWorld::PrepareStateBatch(const std::shared_ptr<const JoltGpuScene::Publication> &publication,
                                const std::vector<Rigidbody *> &bodies, PhysicsStateFields fields, bool includeBoxes,
                                bool triggers, const std::vector<JoltGpuQueryExportBinding> &exports,
                                PhysicsRigidbodyStateBatch &batch, std::shared_ptr<JoltGpuQueryResult> &motion,
                                std::shared_ptr<JoltGpuQueryResult> &shapes, std::exception_ptr &requestError) const
{
    return PrepareQueryWork(
        [&]() -> JoltGpuControlWork {
            for (const auto &binding : exports)
                if (binding.field > JoltGpuQueryExportField::ShapeBounciness ||
                    (!includeBoxes && binding.field >= JoltGpuQueryExportField::ShapeBodyIndex))
                    throw JoltGpuQueryRequestError("Physics export requires its requested state family");
            batch.bodies = bodies;
            batch.states.resize(bodies.size());
            std::vector<QueryInput> motionInputs, shapeInputs;
            const bool numericMotion =
                (static_cast<uint32_t>(fields) & static_cast<uint32_t>(PhysicsStateFields::All)) != 0 ||
                std::any_of(exports.begin(), exports.end(), [](const auto &value) {
                    return value.field <= JoltGpuQueryExportField::MotionInverseInertia;
                });
            motionInputs.reserve(numericMotion ? bodies.size() : 0);
            shapeInputs.reserve(includeBoxes ? bodies.size() : 0);
            for (size_t index = 0; index < bodies.size(); ++index) {
                auto *body = bodies[index];
                if (!body || !body->IsEnabled() || body->IsDestroyed() || !body->GetGameObject())
                    throw JoltGpuQueryRequestError("Physics batch requires live enabled Rigidbody components");
                const auto allocation = m_gpuScene->FindBody(body->GetBodyId());
                if (allocation.bodyId == UINT32_MAX || allocation.slot >= publication->actors.size() ||
                    !publication->actors[allocation.slot])
                    throw JoltGpuQueryRequestError("Physics batch requires accepted body identities");
                const auto &identity = *publication->actors[allocation.slot];
                if (identity.bodyId != allocation.bodyId || identity.rigidbodyComponentId != body->GetComponentID() ||
                    identity.rigidbody != body->GetECSHandle())
                    throw JoltGpuQueryRequestError("Physics batch Rigidbody differs from its retained publication");
                batch.states[index].bodyId = identity.bodyId;
                batch.states[index].shapeRevision = identity.shapeRevision;
                if (numericMotion) {
                    auto input = Query(QueryGatherMotion, QueryClosest, 0, false);
                    input.meta.z = 0;
                    input.source.y = allocation.slot;
                    input.source.z = static_cast<uint32_t>(fields);
                    input.geometry = identity.rootCenterOfMass;
                    motionInputs.push_back(input);
                }
                if (includeBoxes) {
                    auto input = Query(QueryGatherShapes, QueryAll, UINT32_MAX, triggers);
                    input.meta.z = ShapeBox;
                    input.source.y = allocation.slot;
                    shapeInputs.push_back(input);
                }
            }
            const uint64_t generation = GetQueryGeneration() + static_cast<uint64_t>(m_controlNumericalChange);
            const auto readback = exports.empty() ? JoltGpuQueryReadback::Host : JoltGpuQueryReadback::Device;
            JoltGpuPreparedQuery first, second;
            if (!motionInputs.empty()) {
                first = m_gpuQueries->Prepare(publication, motionInputs, {}, generation, readback);
                motion = first.result;
            }
            if (!shapeInputs.empty()) {
                second = m_gpuQueries->Prepare(publication, shapeInputs, {}, generation, readback);
                shapes = second.result;
            }
            return AfterWork(std::move(first.work), [this, motionResult = motion, shapeResult = shapes,
                                                     work = std::move(second.work), &batch, exports]() mutable {
                if (motionResult && motionResult->GetSummary().counts.y != QuerySuccess)
                    return JoltGpuControlWork{};
                return AfterWork(std::move(work), [this, motionResult, shapeResult, &batch, exports] {
                    if (shapeResult) {
                        const auto &summary = shapeResult->GetSummary();
                        if (summary.counts.y != QuerySuccess)
                            return JoltGpuControlWork{};
                        batch.boxCount = summary.counts.x;
                    }
                    if (exports.empty())
                        return JoltGpuControlWork{};
                    std::vector<std::shared_ptr<JoltGpuQueryResult>> results;
                    if (motionResult)
                        results.push_back(motionResult);
                    if (shapeResult)
                        results.push_back(shapeResult);
                    return m_gpuQueries->PrepareExport(results, exports);
                });
            });
        },
        requestError);
}

void PhysicsWorld::MaterializeStateBatch(PhysicsRigidbodyStateBatch &batch,
                                         const std::shared_ptr<JoltGpuQueryResult> &motion,
                                         const std::shared_ptr<JoltGpuQueryResult> &shapes, bool numerical) const
{
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    if (motion) {
        (void)motion->ReadSpans();
        if (numerical) {
            const auto &records = motion->ReadMotion();
            if (records.size() != batch.bodies.size())
                throw std::runtime_error("GPU motion batch did not preserve input body order");
            for (size_t index = 0; index < records.size(); ++index) {
                auto state = DecodeMotion(*motion->publication, records[index]);
                if (state.bodyId != batch.states[index].bodyId)
                    throw std::runtime_error("GPU motion batch differs from its input body order");
                batch.states[index] = state;
                m_requestedMotion[batch.states[index].bodyId] = {motion->numericalGeneration, batch.states[index],
                                                                 static_cast<PhysicsStateFields>(0x1ff)};
            }
        }
    }
    if (shapes) {
        const auto &spans = shapes->ReadSpans();
        if (numerical) {
            const auto &records = shapes->ReadShapes();
            batch.boxes.reserve(records.size());
            for (size_t input = 0; input < spans.size(); ++input)
                for (uint32_t local = 0; local < spans[input].range.y; ++local) {
                    const auto &record = records[spans[input].range.x + local];
                    const auto &address = shapes->publication->leaves.at(record.meta.z);
                    if (address.bodySlot != record.meta.x)
                        throw std::runtime_error("GPU box batch differs from its published leaf owner");
                    const auto &leaf = shapes->publication->actors.at(record.meta.x)->leaves.at(address.actorLeaf);
                    PhysicsBoxShapeState box;
                    box.memberIndex = leaf.memberIndex;
                    box.center = Vector(record.center);
                    box.rotation = {record.rotation.w, record.rotation.x, record.rotation.y, record.rotation.z};
                    box.halfExtents = Vector(record.geometry);
                    box.friction = record.material.x;
                    box.bounciness = record.material.y;
                    batch.boxes.push_back({static_cast<uint32_t>(input), box});
                }
        }
    }
}

PhysicsRigidbodyStateBatch
PhysicsWorld::GatherRigidbodyStates(const std::vector<Rigidbody *> &bodies, PhysicsStateFields fields,
                                    bool includeBoxes, bool triggers,
                                    const std::vector<JoltGpuQueryExportBinding> &exports) const
{
    PhysicsRigidbodyStateBatch batch;
    RequireOwnerThread();
    if (bodies.empty())
        return batch;
    for (auto *body : bodies)
        if (!body || !body->IsEnabled() || body->IsDestroyed() || !body->GetGameObject())
            throw std::invalid_argument("Physics batch requires live enabled Rigidbody components");
    std::shared_ptr<JoltGpuQueryResult> motion, shapes;
    std::exception_ptr error;
    RunControl([&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        return PrepareStateBatch(publication, bodies, fields, includeBoxes, triggers, exports, batch, motion, shapes,
                                 error);
    });
    if (error)
        std::rethrow_exception(error);
    MaterializeStateBatch(batch, motion, shapes, exports.empty());
    return batch;
}

PhysicsRigidbodyStateBatch
PhysicsWorld::GatherRigidbodyStatesInBounds(const glm::vec3 &minimum, const glm::vec3 &maximum, uint32_t layers,
                                            bool triggers, PhysicsStateFields fields, bool includeBoxes,
                                            const std::vector<JoltGpuQueryExportBinding> &exports) const
{
    (void)Vector(minimum);
    (void)Vector(maximum);
    if (glm::any(glm::greaterThan(minimum, maximum)))
        throw std::invalid_argument("Physics query bounds require minimum <= maximum");
    auto input = Query(QueryBodyBounds, QueryUniqueBody, layers, triggers);
    input.positionMaximum = Vector(glm::vec3((glm::dvec3(minimum) + glm::dvec3(maximum)) * .5));
    input.direction = Vector(glm::vec3((glm::dvec3(maximum) - glm::dvec3(minimum)) * .5));
    PhysicsRigidbodyStateBatch batch;
    std::shared_ptr<JoltGpuQueryResult> bounds, motion, shapes;
    std::exception_ptr error;
    RunControl([&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        return PrepareQueryWork(
            [&]() {
                auto query = m_gpuQueries->Prepare(
                    publication, {input}, {}, GetQueryGeneration() + static_cast<uint64_t>(m_controlNumericalChange));
                bounds = query.result;
                return AfterWork(std::move(query.work), [this, publication, &batch, &bounds, &motion, &shapes, fields,
                                                         includeBoxes, triggers, exports, &error] {
                    if (bounds->GetSummary().counts.y != QuerySuccess)
                        return JoltGpuControlWork{};
                    std::vector<Rigidbody *> bodies;
                    auto &store = PhysicsECSStore::Instance();
                    for (const auto &record : bounds->ReadHits()) {
                        const auto &identity = *publication->actors.at(record.meta.x);
                        if (!store.IsValid(identity.rigidbody))
                            throw JoltGpuQueryRequestError("Bounds result Rigidbody no longer exists");
                        bodies.push_back(store.GetRigidbody(identity.rigidbody).owner);
                    }
                    return PrepareStateBatch(publication, bodies, fields, includeBoxes, triggers, exports, batch,
                                             motion, shapes, error);
                });
            },
            error);
    });
    if (error)
        std::rethrow_exception(error);
    (void)bounds->ReadSpans();
    MaterializeStateBatch(batch, motion, shapes, exports.empty());
    return batch;
}

PhysicsBodyMotionState PhysicsWorld::ReadMotionState(uint32_t id, PhysicsStateFields fields) const
{
    RequireOwnerThread();
    RequireHealthyStep();
    RequireGpu();
    const auto cached = m_requestedMotion.find(id);
    if ((!m_stepPending || m_stepCompleted) && !HasPendingControl() && cached != m_requestedMotion.end() &&
        cached->second.generation == GetQueryGeneration() &&
        (static_cast<uint32_t>(fields) & ~static_cast<uint32_t>(cached->second.fields)) == 0)
        return cached->second.state;
    PhysicsBodyMotionState result;
    std::shared_ptr<JoltGpuQueryResult> motion;
    std::exception_ptr error;
    RunControl([&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        return PrepareQueryWork(
            [&]() -> JoltGpuControlWork {
                const auto allocation = m_gpuScene->FindBody(id);
                if (allocation.bodyId == UINT32_MAX || allocation.slot >= publication->actors.size() ||
                    !publication->actors[allocation.slot] || publication->actors[allocation.slot]->bodyId != id)
                    throw JoltGpuQueryRequestError("Physics state requires an accepted live body");
                const auto &identity = *publication->actors[allocation.slot];
                result.bodyId = id;
                result.shapeRevision = identity.shapeRevision;
                if ((static_cast<uint32_t>(fields) & static_cast<uint32_t>(PhysicsStateFields::All)) == 0)
                    return {};
                auto input = Query(QueryGatherMotion, QueryClosest, 0, false);
                input.meta.z = 0;
                input.source.y = allocation.slot;
                input.source.z = static_cast<uint32_t>(fields);
                input.geometry = identity.rootCenterOfMass;
                auto prepared = m_gpuQueries->Prepare(
                    publication, {input}, {}, GetQueryGeneration() + static_cast<uint64_t>(m_controlNumericalChange));
                motion = prepared.result;
                return std::move(prepared.work);
            },
            error);
    });
    if (error)
        std::rethrow_exception(error);
    if (motion) {
        FrameWorkOwner::Scope materialize(*m_gpuWork);
        result = DecodeMotion(*motion->publication, motion->ReadMotion().at(0));
        m_requestedMotion[id] = {motion->numericalGeneration, result, static_cast<PhysicsStateFields>(0x1ff)};
    }
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
glm::mat3 PhysicsWorld::GetBodyWorldSpaceInertiaTensor(uint32_t id) const
{
    return ReadMotionState(id, PhysicsStateFields::InverseInertia).inertia;
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
    const auto first = CookColliderQueryPrimitive(a, positionA, rotationA);
    const auto second = CookColliderQueryPrimitive(b, positionB, rotationB);
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
bool PhysicsWorld::SphereCast(const glm::vec3 &center, float radius, const glm::vec3 &direction, float maximum,
                              RaycastHit &hit, uint32_t layers, bool triggers) const
{
    auto request = Query(QueryShapeCast, QueryClosest, layers, triggers);
    QuerySphereInput(request, center, radius);
    QueryRayInput(request, center, direction, maximum);
    auto result = RunQuery({request});
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    const auto &record = result->ReadHits().front();
    hit = DecodeQueryHit(*result, record);
    return record.meta.y != 0;
}
bool PhysicsWorld::BoxCast(const glm::vec3 &center, const glm::vec3 &halfExtents, const glm::vec3 &direction,
                           const glm::quat &rotation, float maximum, RaycastHit &hit, uint32_t layers,
                           bool triggers) const
{
    auto request = Query(QueryShapeCast, QueryClosest, layers, triggers);
    QueryBoxInput(request, center, halfExtents, rotation);
    QueryRayInput(request, center, direction, maximum);
    auto result = RunQuery({request});
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    const auto &record = result->ReadHits().front();
    hit = DecodeQueryHit(*result, record);
    return record.meta.y != 0;
}
bool PhysicsWorld::SweepBox(const BoxCollider &box, const glm::vec3 &position, const glm::quat &rotation,
                            const glm::vec3 &displacement, RaycastHit &hit) const
{
    RequireOwnerThread();
    (void)Vector(position);
    (void)Rotation(rotation);
    (void)Vector(displacement);
    const double distance =
        std::hypot(std::hypot(double(displacement.x), double(displacement.y)), double(displacement.z));
    auto *object = box.GetGameObject();
    if (!object || distance == 0) {
        hit = {};
        return false;
    }
    if (distance > std::numeric_limits<float>::max())
        throw std::invalid_argument("BoxCollider sweep displacement exceeds device coordinates");
    const auto colliderId = box.GetComponentID();
    const auto objectId = object->GetID();
    const auto layers = TagLayerManager::Instance().GetLayerCollisionMask(object->GetLayer());
    const auto primitive = CookColliderQueryPrimitive(box, position, rotation);
    if (!primitive) {
        hit = {};
        return false;
    }
    auto request = Query(QueryShapeCast, QueryClosest, layers, false);
    request.meta.z = primitive->meta.x;
    request.meta.w = QuerySweepPolicy;
    request.geometry = primitive->geometry;
    request.rotation = primitive->rotation;
    QueryRayInput(request, Vector(primitive->position), displacement, static_cast<float>(distance));
    request.source.z = static_cast<uint32_t>(objectId);
    request.source.w = static_cast<uint32_t>(objectId >> 32);
    request.filter.z = static_cast<uint32_t>(colliderId);
    request.filter.w = static_cast<uint32_t>(colliderId >> 32);
    auto result = RunQuery({request}, colliderId);
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    const auto &record = result->ReadHits().front();
    hit = DecodeQueryHit(*result, record);
    return record.meta.y != 0;
}
bool PhysicsWorld::CapsuleCast(const glm::vec3 &a, const glm::vec3 &b, float radius, const glm::vec3 &direction,
                               float maximum, RaycastHit &hit, uint32_t layers, bool triggers) const
{
    auto request = Query(QueryShapeCast, QueryClosest, layers, triggers);
    QueryCapsuleInput(request, a, b, radius);
    QueryRayInput(request, Vector(request.positionMaximum), direction, maximum);
    auto result = RunQuery({request});
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    const auto &record = result->ReadHits().front();
    hit = DecodeQueryHit(*result, record);
    return record.meta.y != 0;
}
std::shared_ptr<const PhysicsShapeGeometry> PhysicsWorld::ResolveShapeGeometry(const JoltGpuScene::PublishedLeaf &leaf,
                                                                               const QueryShapeState &record) const
{
    const auto *shape = leaf.shape.GetPtr();
    if (!shape)
        throw std::runtime_error("Published shape state has no retained immutable geometry");
    if (const auto found = m_shapeGeometry.find(shape); found != m_shapeGeometry.end())
        if (auto geometry = found->second.lock())
            return geometry;
    auto geometry = std::make_shared<PhysicsShapeGeometry>();
    // Intrusive Jolt ownership is retained by the aliasing owner. The pointer
    // is an in-process identity only and remains unique while this value lives.
    auto owner = std::make_shared<JPH::RefConst<JPH::Shape>>(leaf.shape);
    geometry->shape = std::shared_ptr<const JPH::Shape>(std::move(owner), shape);
    geometry->identity = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(shape));
    geometry->convex = true;
    const auto bounds = shape->GetLocalBounds();
    geometry->boundsMin = {bounds.mMin.GetX(), bounds.mMin.GetY(), bounds.mMin.GetZ()};
    geometry->boundsMax = {bounds.mMax.GetX(), bounds.mMax.GetY(), bounds.mMax.GetZ()};
    if (record.surface.y == ShapeBox) {
        geometry->type = PhysicsShapeType::Box;
        geometry->dimensions = {record.geometry.x * 2, record.geometry.y * 2, record.geometry.z * 2, record.geometry.w};
    } else if (record.surface.y == ShapeSphere) {
        geometry->type = PhysicsShapeType::Sphere;
        geometry->dimensions = {record.geometry.w, 0, 0, 0};
    } else if (record.surface.y == ShapeCapsule) {
        geometry->type = PhysicsShapeType::Capsule;
        geometry->dimensions = {record.geometry.w, 2 * (record.geometry.y + record.geometry.w), 1, 0};
    } else {
        throw std::runtime_error("GPU shape state has no immutable geometry adapter");
    }
    m_shapeGeometry[shape] = geometry;
    return geometry;
}

std::shared_ptr<JoltGpuQueryResult> PhysicsWorld::GatherBodyShapes(uint32_t id, bool triggers) const
{
    JoltGpuPreparedQuery query;
    std::exception_ptr error;
    RunControl([&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        return PrepareQueryWork(
            [&] {
                const auto allocation = m_gpuScene->FindBody(id);
                if (allocation.bodyId == UINT32_MAX || allocation.slot >= publication->actors.size() ||
                    !publication->actors[allocation.slot] || publication->actors[allocation.slot]->bodyId != id)
                    throw JoltGpuQueryRequestError("Shape gathering requires an accepted body identity");
                auto input = Query(QueryGatherShapes, QueryAll, UINT32_MAX, triggers);
                input.meta.z = QueryAllShapeTypes;
                input.source.y = allocation.slot;
                query = m_gpuQueries->Prepare(publication, {input}, {},
                                              GetQueryGeneration() + static_cast<uint64_t>(m_controlNumericalChange));
                return std::move(query.work);
            },
            error);
    });
    if (error)
        std::rethrow_exception(error);
    return std::move(query.result);
}

std::vector<PhysicsShapeState> PhysicsWorld::GetBodyShapeStates(uint32_t id, bool triggers) const
{
    auto query = GatherBodyShapes(id, triggers);
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    std::vector<PhysicsShapeState> result;
    const auto &records = query->ReadShapes();
    result.reserve(records.size());
    for (const auto &record : records) {
        const auto &address = query->publication->leaves.at(record.meta.z);
        const auto &identity = *query->publication->actors.at(record.meta.x);
        if (address.bodySlot != record.meta.x)
            throw std::runtime_error("GPU shape state has a different published leaf owner");
        const auto &leaf = identity.leaves.at(address.actorLeaf);
        PhysicsShapeState state;
        state.collider = leaf.surfaceTarget.GetCollider();
        state.memberIndex = leaf.memberIndex;
        state.layerBit = record.surface.x;
        state.trigger = (record.meta.w & 1u) != 0;
        state.moving = (record.meta.w & 2u) != 0;
        state.geometry = ResolveShapeGeometry(leaf, record);
        state.center = Vector(record.center);
        state.rotation = {record.rotation.w, record.rotation.x, record.rotation.y, record.rotation.z};
        state.bodyCenterOfMass = Vector(record.bodyCenterOfMass);
        state.linearVelocity = Vector(record.linearVelocity);
        state.angularVelocity = Vector(record.angularVelocity);
        state.friction = record.material.x;
        state.bounciness = record.material.y;
        state.frictionCombine = record.surface.z;
        state.bounceCombine = record.surface.w;
        result.push_back(std::move(state));
    }
    return result;
}

std::vector<PhysicsBoxShapeState> PhysicsWorld::GetBodyBoxShapeStates(uint32_t id, bool triggers) const
{
    std::vector<PhysicsBoxShapeState> result;
    for (const auto &state : GetBodyShapeStates(id, triggers))
        if (state.geometry->type == PhysicsShapeType::Box)
            result.push_back({state.memberIndex, state.center, state.rotation,
                              glm::vec3(state.geometry->dimensions) * .5f, state.friction, state.bounciness});
    return result;
}

std::shared_ptr<particle::GpuParticleCollisionDevicePayload> PhysicsWorld::PrepareParticleCollisionScene(
    Scene &scene, const std::shared_ptr<const particle::GpuParticleCollisionDevicePayload> &previous,
    uint64_t revision) const
{
    using RowKey = particle::GpuParticleCollisionRowKey;
    struct Row
    {
        RowKey key;
        JoltGpuParticleExportRow mapping;
    };
    JoltGpuPreparedQuery query;
    std::shared_ptr<JoltGpuParticleExportResult> exported;
    std::vector<RowKey> keys;
    std::exception_ptr error;
    RunControl([&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        return PrepareQueryWork(
            [&]() -> JoltGpuControlWork {
                std::vector<QueryInput> requests;
                std::array<std::vector<Row>, 2> rows;
                uint32_t sourceRow = 0;
                for (uint32_t slot = 0; slot < publication->actors.size(); ++slot) {
                    const auto &actor = publication->actors[slot];
                    auto *owner = actor ? actor->owner.target.GetGameObject() : nullptr;
                    if (!owner || owner->GetScene() != &scene || !publication->residentBodies.at(slot))
                        continue;
                    auto input = Query(QueryGatherShapes, QueryAll);
                    input.meta.z = QueryAllShapeTypes;
                    input.source.y = slot;
                    requests.push_back(input);
                    for (const auto &leaf : actor->leaves) {
                        Row row;
                        row.key = {leaf.colliderId, leaf.subShapeId, leaf.subShapeBits,
                                   reinterpret_cast<uintptr_t>(leaf.shape.GetPtr())};
                        row.mapping.selection = {sourceRow++, InvalidQueryBodySlot, 0, 0};
                        row.mapping.identity = {static_cast<uint32_t>(leaf.colliderId),
                                                static_cast<uint32_t>(leaf.colliderId >> 32),
                                                static_cast<uint32_t>(leaf.surfaceGameObjectId),
                                                static_cast<uint32_t>(leaf.surfaceGameObjectId >> 32)};
                        rows[actor->acceptedMotionType == MotionStatic ? 0 : 1].push_back(std::move(row));
                    }
                }
                // Preserve accepted leaf order within the two particle table ranges.
                // This is identity mapping only; all numerical projection stays GPU-side.
                const auto staticCount = static_cast<uint32_t>(rows[0].size());
                const bool hasHistory = previous && previous->simulationEpoch == publication->simulationEpoch;
                std::map<RowKey, uint32_t> history;
                if (hasHistory)
                    for (uint32_t index = 0; index < previous->rows.size(); ++index)
                        history.emplace(previous->rows[index], index);
                std::vector<JoltGpuParticleExportRow> mappings;
                mappings.reserve(sourceRow);
                keys.reserve(sourceRow);
                for (auto &range : rows)
                    for (auto &row : range) {
                        if (const auto old = history.find(row.key); old != history.end())
                            row.mapping.selection.y = old->second;
                        mappings.push_back(row.mapping);
                        keys.push_back(row.key);
                    }
                const uint64_t topology = !hasHistory ? 1
                                                      : previous->topologyRevision +
                                                            static_cast<uint64_t>(previous->rows != keys ||
                                                                                  previous->staticCount != staticCount);
                query = m_gpuQueries->Prepare(publication, requests, {},
                                              GetQueryGeneration() + static_cast<uint64_t>(m_controlNumericalChange),
                                              JoltGpuQueryReadback::Device);
                return AfterWork(
                    std::move(query.work),
                    [this, source = query.result, mappings = std::move(mappings), previous, hasHistory, staticCount,
                     revision, topology, &exported]() -> JoltGpuControlWork {
                        const auto &summary = source->GetSummary();
                        if (summary.counts.y != QuerySuccess)
                            return {};
                        if (summary.counts.x != mappings.size())
                            throw std::runtime_error("GPU particle gather differs from its retained leaf catalog");
                        auto prepared = m_gpuQueries->PrepareParticleExport(source, mappings,
                                                                            hasHistory ? previous->records : nullptr,
                                                                            staticCount, revision, topology);
                        exported = std::move(prepared.result);
                        return std::move(prepared.work);
                    });
            },
            error);
    });
    if (error)
        std::rethrow_exception(error);
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    (void)query.result->ReadSpans();
    // A recognized read-only numerical failure is reported outside the world's
    // source, so a failed export does not poison subsequent simulation steps.
    exported->Validate();
    exported->payload->simulationEpoch = query.result->publication->simulationEpoch;
    exported->payload->rows = std::move(keys);
    return std::move(exported->payload);
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
