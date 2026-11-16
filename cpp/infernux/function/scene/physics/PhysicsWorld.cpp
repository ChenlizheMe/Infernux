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
#include "../RuntimeWorld.h"
#include "../Scene.h"
#include "../SceneManager.h"
#include "../TagLayerManager.h"
#include "../Transform.h"
#include "../TransformECSStore.h"
#include "JoltCpuControl.h"
#include "JoltGpuContactReports.h"
#include "JoltGpuJoints.h"
#include "JoltGpuProgramSources.h"
#include "JoltGpuQueries.h"
#include "JoltGpuRenderBindings.h"
#include "JoltGpuRenderPoses.h"
#include "PhysicsContactTypes.h"
#include "PhysicsDomain.h"
#include "PhysicsLayers.h"
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
#include <unordered_set>
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
    if (record.meta.x >= publication.directory->actors.size() || !publication.directory->actors[record.meta.x] ||
        publication.directory->actors[record.meta.x]->bodyId != record.meta.z)
        throw std::runtime_error("GPU motion state differs from its retained body identity");
    const auto &actor = *publication.directory->actors[record.meta.x];
    PhysicsBodyMotionState result;
    result.bodyId = actor.bodyId;
    result.shapeRevision = actor.shapeRevision;
    result.position = Vector(record.positionInverseMass);
    result.rotation = {record.rotation.w, record.rotation.x, record.rotation.y, record.rotation.z};
    result.centerOfMass = Vector(record.centerOfMass);
    result.linearVelocity = Vector(record.linearVelocity);
    result.angularVelocity = Vector(record.angularVelocity);
    const auto frozenTranslation = record.meta.w;
    const float inverseMass = record.positionInverseMass.w;
    result.inverseMass = {frozenTranslation & 1u ? 0.0f : inverseMass, frozenTranslation & 2u ? 0.0f : inverseMass,
                          frozenTranslation & 4u ? 0.0f : inverseMass};
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
struct CookedQueryOperand
{
    QueryPrimitive primitive;
    std::shared_ptr<const JoltGpuGeometryRecord> geometry;
    JPH::RefConst<JPH::Shape> shape;
};
std::optional<CookedQueryOperand> CookColliderQueryPrimitive(const Collider &collider, const glm::vec3 &position,
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
    return CookedQueryOperand{
        QueryPrimitive{{leaf.meta.x, leaf.meta.w, 0, 0}, leaf.geometry, Vector(center), Rotation(world * local), {}},
        prepared.leaves.front().geometryRecord, shape};
}
struct AdmissionScope
{
    explicit AdmissionScope(JoltGpuScene &scene, RuntimeWorld &world)
        : scene(scene), world(world), previous(scene.IsAdmissionEnabled(world))
    {
        scene.SetAdmissionEnabled(world, true);
    }
    ~AdmissionScope()
    {
        scene.SetAdmissionEnabled(world, previous);
    }
    JoltGpuScene &scene;
    RuntimeWorld &world;
    bool previous;
};
struct FlagScope
{
    explicit FlagScope(bool &flag) : flag(flag), previous(flag)
    {
        flag = true;
    }
    ~FlagScope()
    {
        flag = previous;
    }
    bool &flag;
    bool previous;
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
Rigidbody &KinematicBody(const PhysicsActorData &actor)
{
    auto *body = actor.rigidbody;
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
JoltGpuStepParameters StepParameters(float dt, const glm::vec3 &gravity)
{
    const auto &c = EngineConfig::Get();
    if (c.physicsVelocitySteps < 0 || c.physicsPositionSteps < 0 || c.physicsCollisionSteps <= 0)
        throw std::invalid_argument("Physics iterations and substeps are invalid");
    StepParams p{};
    p.counts.w = static_cast<uint32_t>(c.physicsVelocitySteps);
    p.solve.x = static_cast<uint32_t>(c.physicsPositionSteps);
    p.solve.y = static_cast<uint32_t>(c.physicsCollisionSteps);
    p.gravityDt = {gravity.x, gravity.y, gravity.z, dt};
    p.contact = {c.physicsSpeculativeContactDistance, c.physicsPenetrationSlop, c.physicsMaxPenetrationDistance,
                 c.physicsBaumgarte};
    p.restitution.x = c.physicsMinVelocityForRestitution;
    return JoltGpuStepParameters(p);
}
} // namespace

struct PhysicsWorld::CompiledGpuPrograms
{
    ShaderTargetCode step, mutations, commands, broadphase, narrowphase, constraints, articulationCapture,
        articulationDynamics;
};
struct PhysicsWorld::CpuServices
{
    explicit CpuServices(const EngineConfig &config) : execution(config)
    {
    }
    BPLayerInterface layers;
    ObjectVsBPLayerFilter broadphaseFilter;
    ObjectLayerPairFilter pairFilter;
    JoltCpuExecution execution;
};
PhysicsDomain::PhysicsDomain(RuntimeWorld &world) : owner(world), gravity(EngineConfig::Get().physicsGravity)
{
}
PhysicsDomain::~PhysicsDomain() = default;
namespace
{
// Geometry cooking and mass authoring also run before Play and between engine
// instances. Their Jolt allocation/type lifetime is independent of GPU physics.
class JoltAuthoringRuntime final
{
  public:
    JoltAuthoringRuntime()
    {
#ifndef JPH_DISABLE_CUSTOM_ALLOCATOR
        // An existing allocator family belongs to its upstream owner.
        if (!JPH::Allocate)
            JPH::RegisterDefaultAllocator();
#endif
        if (!JPH::Factory::sInstance) {
            m_ownedFactory = new JPH::Factory();
            JPH::Factory::sInstance = m_ownedFactory;
            JPH::RegisterTypes();
        }
    }
    ~JoltAuthoringRuntime()
    {
        if (m_ownedFactory) {
            if (JPH::Factory::sInstance == m_ownedFactory) {
                JPH::UnregisterTypes();
                JPH::Factory::sInstance = nullptr;
            }
            delete m_ownedFactory;
        }
    }

  private:
    JPH::Factory *m_ownedFactory = nullptr;
};

PhysicsDomain::RenderState::Keys CaptureRenderKeys()
{
    const auto &manager = SceneManager::Instance();
    return {manager.GetRenderTransformRevision(), manager.GetMeshRendererVersion(), manager.GetRenderContentRevision()};
}
JoltGpuTickOutput CompactOutput(const JoltGpuScene::Publication &p)
{
    return {p.worldRow,   p.bodyBase,    p.bodyCount,   p.leafBase,      p.leafCount, p.jointBase,         p.jointCount,
            p.bodyStates, p.jointStream, p.jointLayout, p.geometryStore, p.geometry,  p.hasGenericGeometry};
}
} // namespace
PhysicsWorld &PhysicsWorld::Instance()
{
    static PhysicsWorld *world = new PhysicsWorld();
    return *world;
}
PhysicsWorld::PhysicsWorld()
{
    static JoltAuthoringRuntime authoring;
}
PhysicsWorld::~PhysicsWorld()
{
    Shutdown();
}
void PhysicsWorld::RequireOwnerThread() const
{
    if (m_ownerThreadId != std::thread::id{} && m_ownerThreadId != std::this_thread::get_id())
        throw std::logic_error("Physics requires its engine owner thread");
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
    m_initialized = true;
}
void PhysicsWorld::RegisterDomain(PhysicsDomain &domain)
{
    m_domains.insert(&domain);
    auto &store = PhysicsECSStore::Instance();
    for (const auto handle : domain.recreateActors)
        if (store.IsValid(handle) && store.GetActor(handle).bodyId == UINT32_MAX)
            store.QueueBodyCreation(handle);
    domain.recreateActors.clear();
}
void PhysicsWorld::EnsureCpuReady(PhysicsDomain &domain)
{
    RequireOwnerThread();
    RequireHealthyStep(domain);
    if (!m_initialized || domain.retiring)
        throw std::logic_error("CPU physics requires a live initialized world");
    if (!m_cpuServices)
        m_cpuServices = std::make_unique<CpuServices>(EngineConfig::Get());
    if (!domain.cpuScene) {
        auto scene = std::make_unique<JoltCpuScene>(EngineConfig::Get(), m_cpuServices->layers,
                                                    m_cpuServices->broadphaseFilter, m_cpuServices->pairFilter);
        auto control = std::make_unique<JoltCpuControl>(*scene);
        scene->SetIgnoredPairs(CaptureColliderPairs(domain));
        domain.cpuScene = std::move(scene);
        domain.cpuControl = std::move(control);
    }
    RegisterDomain(domain);
}
bool PhysicsWorld::EnsureGpuReady(PhysicsDomain &domain)
{
    RequireOwnerThread();
    RequireHealthyStep(domain);
    if (!m_initialized)
        throw std::logic_error("PhysicsWorld is not initialized");
    if (!m_gpuHost || !m_gpuWork)
        throw std::runtime_error("GPU physics requires engine compute services");
    m_gpuHost->RequireAlive();
    if (domain.retiring)
        return false;
    if (domain.gpuCancelled) {
        if (!m_gpuScene->CanRetire(domain.owner))
            return false;
        domain.lastPublication.reset();
        domain.requestedMotion.clear();
        domain.gpuCancelled = false;
        domain.queuedColliderPairs.reset();
    }
    if (!m_gpuPrograms) {
        auto p = std::make_unique<CompiledGpuPrograms>();
        p->step = CompileProgram(*m_gpuHost, JoltGpuProgram::Step);
        p->articulationCapture = CompileProgram(*m_gpuHost, JoltGpuProgram::ArticulationCapture);
        p->articulationDynamics = CompileProgram(*m_gpuHost, JoltGpuProgram::ArticulationDynamics);
        p->mutations = CompileProgram(*m_gpuHost, JoltGpuProgram::BodyMutations);
        p->commands = CompileProgram(*m_gpuHost, JoltGpuProgram::BodyCommands);
        p->broadphase = CompileProgram(*m_gpuHost, JoltGpuProgram::Broadphase);
        p->narrowphase = CompileProgram(*m_gpuHost, JoltGpuProgram::Narrowphase);
        p->constraints = CompileProgram(*m_gpuHost, JoltGpuProgram::Constraints);
        const auto target = CompileProgram(*m_gpuHost, JoltGpuProgram::Queries);
        auto queries = std::make_unique<JoltGpuQueries>(*m_gpuHost, target);
        auto joints = std::make_unique<JoltGpuJoints>(*m_gpuHost, CompileProgram(*m_gpuHost, JoltGpuProgram::Joints));
        auto rendering =
            std::make_unique<JoltGpuRenderPoses>(*m_gpuHost, CompileProgram(*m_gpuHost, JoltGpuProgram::RenderPoses));
        m_gpuPrograms = std::move(p);
        m_gpuQueries = std::move(queries);
        m_gpuJoints = std::move(joints);
        m_gpuRenderProgram = std::move(rendering);
        m_gpuContactReports = std::make_unique<JoltGpuContactReports>(
            *m_gpuHost, CompileProgram(*m_gpuHost, JoltGpuProgram::ContactReports));
    }
    if (!m_gpuScene) {
        m_gpuScene = std::make_unique<JoltGpuScene>(
            *m_gpuHost, *m_gpuWork, m_gpuPrograms->step, m_gpuPrograms->mutations, m_gpuPrograms->commands,
            m_gpuPrograms->broadphase, m_gpuPrograms->narrowphase, m_gpuPrograms->constraints, *m_gpuJoints,
            [this](RuntimeWorld &world) noexcept {
                // A shared source can fail outside Step and outside this World's
                // execution scope. Mark only the actual retained writer.
                FailStep(*world.physics);
                world.simulation.fixedPhase = RuntimeWorld::FixedPhase::Failed;
                world.simulation.pausedStepPending = false;
                world.simulation.isPaused = true;
            },
            &m_gpuPrograms->articulationCapture, &m_gpuPrograms->articulationDynamics);
    }
    if (!m_gpuScene->RegisterPartition(domain.owner, InitialBodyCapacity))
        return false;
    domain.gpuRegistered = true;
    RegisterDomain(domain);
    ReconcileRagdolls(domain);
    QueueColliderPairs(domain);
    return true;
}
void PhysicsWorld::RequireGpu(PhysicsDomain &domain) const
{
    if (!const_cast<PhysicsWorld *>(this)->EnsureGpuReady(domain))
        throw std::runtime_error("Stopped GPU physics is still retiring accepted receipts");
}
void PhysicsWorld::Shutdown()
{
    for (auto *domain : m_domains) {
        RetireRagdolls(*domain);
        if (domain->gpuRegistered)
            m_gpuScene->Cancel(domain->owner);
        domain->rendering.bindings.Clear();
        domain->rendering.pending.reset();
        domain->rendering.ready.reset();
        domain->contacts.Close();
        domain->gpuContacts.Close();
        domain->cpuContacts.Close();
        domain->cpuControl.reset();
        domain->cpuScene.reset();
        domain->lastPublication.reset();
        domain->queuedColliderPairs.reset();
        domain->requestedMotion.clear();
    }
    m_gpuScene.reset(); // Whole-engine teardown after all logical worlds stop.
    m_bodyActors.clear();
    for (auto *domain : m_domains) {
        // The logical World can survive a complete engine shutdown/reinitialization.
        // Its cancelled writer belonged to the arena which has now been drained.
        domain->gpuCancelled = false;
        domain->gpuRegistered = false;
        domain->registeredBodyCount = 0;
        domain->recreateActors.clear();
        domain->shapeGeometry.clear();
        domain->rendering.checkedKeys.reset();
    }
    m_gpuContactReports.reset();
    m_gpuRenderProgram.reset();
    m_gpuQueries.reset();
    m_gpuJoints.reset();
    m_gpuPrograms.reset();
    m_cpuServices.reset();
    m_domains.clear();
    m_gpuHost.reset();
    m_gpuWork = nullptr;
    m_initialized = false;
    m_ownerThreadId = {};
}
size_t PhysicsWorld::GetBodyCount() const
{
    auto &domain = CurrentDomain();
    return domain.registeredBodyCount;
}
glm::vec3 PhysicsWorld::GetGravity() const
{
    auto &domain = CurrentDomain();
    return domain.gravity;
}
void PhysicsWorld::SetGravity(const glm::vec3 &v)
{
    auto &domain = CurrentDomain();
    if (!Finite(v))
        throw std::invalid_argument("Gravity must be finite");
    domain.gravity = v;
}
uint32_t PhysicsWorld::GetDefaultQueryLayerMask()
{
    return EngineConfig::Get().defaultQueryLayerMask;
}
JoltGpuScene::ContactObserver
PhysicsWorld::PrepareContactObserver(PhysicsDomain &domain,
                                     const std::shared_ptr<const JoltGpuScene::Publication> &publication) const
{
    const WorldExecutionScope ownerScope(&domain.owner);
    return domain.gpuContacts.Prepare(
        *m_gpuContactReports, *m_gpuScene, domain.owner, publication, domain.admittedParameters->Get().solve.y,
        {domain.contactEventStreamEnabled, domain.contactIncludeTriggers, domain.contactImpulseStreamEnabled, {}});
}
void PhysicsWorld::ObserveGpuPublication(PhysicsDomain &domain, bool numericalChange) const
{
    const WorldExecutionScope ownerScope(&domain.owner);
    const auto &p = m_gpuScene->GetPublished(domain.owner);
    if (p && p.get() != domain.lastPublication.get()) {
        domain.lastPublication = p;
        if (p->isFixedStep)
            domain.gpuContacts.Publish(domain.contacts, p);
        if (numericalChange) {
            domain.queryGeneration.fetch_add(1, std::memory_order_release);
            domain.requestedMotion.clear();
        }
        if (m_gpuRenderProgram) {
            FrameWorkOwner::Scope publication(*m_gpuWork);
            auto &rendering = domain.rendering;
            const auto &job = rendering.pending;
            if (job && job->bindings && job->publication == p.get()) {
                // Preserve this complete value before another Prepare reuses
                // the live output endpoints, even while the fixed gate waits.
                if (job->output.models)
                    job->output.models = job->output.models->Freeze();
                if (job->output.bounds)
                    job->output.bounds = job->output.bounds->Freeze();
                job->output.batch.reset();
                job->numericalGeneration = domain.queryGeneration.load(std::memory_order_acquire);
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
    auto &domain = CurrentDomain();
    RequireOwnerThread();
    if (!m_initialized || !domain.gpuRegistered || !m_gpuRenderProgram || domain.gpuCancelled ||
        domain.fixedGpuPhase == PhysicsDomain::FixedGpuPhase::Failed)
        return false;
    auto &rendering = domain.rendering;
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
        const_cast<PhysicsWorld *>(this)->FailStep(domain);
        throw;
    }
}
JoltGpuScene::Consumer PhysicsWorld::PrepareRenderConsumer(PhysicsDomain &domain, bool numericalChange) const
{
    const WorldExecutionScope ownerScope(&domain.owner);
    FrameWorkOwner::Scope preparation(*m_gpuWork);
    auto &rendering = domain.rendering;
    const auto keys = CaptureRenderKeys();
    const auto generation =
        domain.queryGeneration.load(std::memory_order_acquire) + static_cast<uint64_t>(numericalChange);
    if (rendering.ready && rendering.ready->numericalGeneration == generation && rendering.ready->keys == keys) {
        rendering.pending.reset();
        return {};
    }
    auto snapshot = rendering.bindings.Capture(domain.owner);
    if (!rendering.bindings.NeedsExport(*snapshot, generation)) {
        rendering.pending.reset();
        rendering.checkedKeys = keys;
        return {};
    }
    auto job = std::make_shared<PhysicsDomain::RenderState::Job>();
    job->snapshot = std::move(snapshot);
    job->keys = keys;
    rendering.pending = job;
    return [this, &domain, job = std::move(job)](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        auto &state = domain.rendering;
        job->bindings = state.bindings.Prepare(job->snapshot, *publication);
        job->output =
            m_gpuRenderProgram->Prepare(CompactOutput(*publication), job->bindings->nodes, job->bindings->bindings);
        job->publication = publication.get();
        return AfterWork(job->output.batch, [job] {
            if (job->output.models)
                job->output.models = job->output.models->Freeze();
            if (job->output.bounds)
                job->output.bounds = job->output.bounds->Freeze();
            job->output.batch.reset();
            return JoltGpuControlWork{};
        });
    };
}
void PhysicsWorld::RequireHealthyStep(PhysicsDomain &domain) const
{
    if (domain.fixedGpuPhase == PhysicsDomain::FixedGpuPhase::Failed)
        throw std::runtime_error("GPU physics step failed; stop the runtime before restarting");
}
void PhysicsWorld::FailStep(PhysicsDomain &domain) noexcept
{
    domain.gpuContacts.CancelPending();
    domain.fixedGpuPhase = PhysicsDomain::FixedGpuPhase::Failed;
    domain.stepCompleted = false;
    if (m_gpuScene)
        m_gpuScene->SetAdmissionEnabled(domain.owner, false);
}
void PhysicsWorld::RefreshGpuSurfaces(PhysicsDomain &domain)
{
    const auto revision = PhysicMaterial::GetSurfaceRevision();
    if (revision == domain.surfaceRevision)
        return;
    auto &store = PhysicsECSStore::Instance();
    store.ForEachAliveActor([&](const PhysicsActorData &actor) {
        if (actor.runtimeWorld != &domain.owner || actor.bodyId == UINT32_MAX)
            return;
        const auto handle = m_gpuScene->FindActor(actor.bodyId);
        if (store.IsValid(handle) && &store.GetActor(handle) == &actor)
            RefreshGpuActor(domain, actor.bodyId);
    });
    domain.surfaceRevision = revision;
}
std::shared_ptr<PhysicsParticipant> PhysicsWorld::RegisterParticipant(const Component &component,
                                                                      PhysicsParticipant::Recipes recipes,
                                                                      PhysicsParticipant::Controls controls,
                                                                      uint32_t count)
{
    RequireOwnerThread();
    auto &domain = DomainForComponent(component);
    RequireHealthyStep(domain);
    if (domain.retiring || recipes.empty() || !m_gpuHost ||
        !recipes.begin()->second->GetHost().SharesServicesWith(*m_gpuHost))
        throw std::invalid_argument("Participant requires its live World's engine compute host");
    auto participant = std::make_shared<PhysicsParticipant>(component, std::move(recipes), std::move(controls), count);
    domain.participants.push_back(participant);
    return participant;
}
bool PhysicsWorld::HasActiveParticipants() const
{
    const auto &values = CurrentDomain().participants;
    return std::any_of(values.begin(), values.end(), [](const auto &value) {
        const auto owner = value.lock();
        return owner && owner->IsActive();
    });
}
bool PhysicsWorld::ServiceParticipantPreparation()
{
    RequireOwnerThread();
    auto &domain = CurrentDomain();
    RequireHealthyStep(domain);
    auto &values = domain.participants;
    values.erase(std::remove_if(values.begin(), values.end(),
                                [](const auto &value) {
                                    const auto owner = value.lock();
                                    return !owner || owner->IsRetired();
                                }),
                 values.end());
    const auto preparing = domain.participantPreparing.lock();
    if (values.empty() && !preparing)
        return true;
    if (domain.stepPending || domain.controlPending || !EnsureGpuReady(domain))
        return false;
    AdmissionScope admission(*m_gpuScene, domain.owner);
    FrameWorkOwner::Scope preparation(*m_gpuWork);
    if (preparing) {
        const auto status = m_gpuScene->GetTickStatus(domain.owner);
        if (status == JoltGpuTick::Status::Failed)
            throw std::runtime_error("Participant initialization failed with its retained GPU source");
        if (status != JoltGpuTick::Status::Complete)
            return false;
        domain.participantPreparing.reset();
    }
    for (const auto &entry : values) {
        const auto value = entry.lock();
        if (!value || value->IsRetired() || value->IsReady())
            continue;
        // The same control prefix handles both a first pending partition layout
        // and an existing row. Pure-query TryBeginWork keeps its original meaning.
        JoltGpuScene::Consumer initialize = [value](const auto &) { return value->PrepareInitialize(); };
        if (!m_gpuScene->TryBeginControl(domain.owner, domain.owner.simulation.simulationEpoch, std::move(initialize)))
            return false;
        domain.participantPreparing = value;
        // Wake/admit only. The ordinary FrameWorkOwner drives the same Tick
        // source; a session preparation callback never recursively drives it.
        return false;
    }
    return true;
}
bool PhysicsWorld::ServiceParticipantOutputs()
{
    RequireOwnerThread();
    auto &domain = CurrentDomain();
    if (domain.retiring || domain.owner.handle->destroyRequested)
        return true; // The existing retirement chain synchronously cancels leases.
    RequireHealthyStep(domain);
    std::vector<std::shared_ptr<PhysicsParticipant>> observers;
    for (const auto &entry : domain.participants)
        if (const auto owner = entry.lock(); owner && owner->HasPendingOutputs())
            observers.push_back(owner);
    if (observers.empty() && !domain.participantOutputPending)
        return true;
    if (!EnsureGpuReady(domain))
        return false;
    // Observe only. The Host continuation never drives/drains the GPU source.
    const auto status = m_gpuScene->GetTickStatus(domain.owner);
    if (status == JoltGpuTick::Status::Failed)
        throw std::runtime_error("Participant output source failed");
    if (domain.participantOutputPending) {
        if (status != JoltGpuTick::Status::Complete)
            return false;
        domain.participantOutputPending = false;
    }
    if (domain.stepPending && !domain.stepCompleted) {
        if (!domain.solverStarted || status != JoltGpuTick::Status::Complete)
            return false;
        if (m_gpuScene->PublishCompletedTick(domain.owner))
            ObserveGpuPublication(domain, true);
        domain.stepCompleted = true;
    }
    if (domain.controlPending) {
        if (status != JoltGpuTick::Status::Complete)
            return false;
        if (!m_gpuScene->PublishCompletedControl(domain.owner))
            throw std::runtime_error("Participant output could not observe completed control");
        ObserveGpuPublication(domain, domain.controlNumericalChange);
        domain.controlPending = domain.controlNumericalChange = false;
    }
    if (!domain.participantPreparing.expired()) {
        if (status != JoltGpuTick::Status::Complete)
            return false;
        domain.participantPreparing.reset();
    }
    if (std::any_of(observers.begin(), observers.end(), [](const auto &owner) { return !owner->IsReady(); })) {
        (void)ServiceParticipantPreparation();
        return false;
    }
    if (observers.empty())
        return true;
    const auto publication = m_gpuScene->GetPublished(domain.owner);
    if (!publication)
        return false;
    // Capture the actual completed boundary. Source-write tickets are read
    // later, inside the exclusive Work writer, before preparing each copy.
    const auto epoch = publication->simulationEpoch, step = publication->fixedStepIndex;
    const bool fixed = publication->isFixedStep;
    JoltGpuControlWork copies;
    for (auto it = observers.rbegin(); it != observers.rend(); ++it) {
        auto tail = std::move(copies);
        copies = {nullptr, [owner = *it, epoch, step, fixed, tail = std::move(tail)]() mutable {
                      return AfterWork(owner->PrepareOutputs(epoch, step, fixed),
                                       [tail = std::move(tail)]() mutable { return std::move(tail); });
                  }};
    }
    AdmissionScope admission(*m_gpuScene, domain.owner);
    FrameWorkOwner::Scope preparation(*m_gpuWork);
    if (!m_gpuScene->TryBeginWork(domain.owner, copies, false))
        return false;
    domain.participantOutputPending = true;
    return false;
}
bool PhysicsWorld::Step(float dt, PhysicsStepProfile *profile)
{
    auto &domain = CurrentDomain();
    RequireOwnerThread();
    RequireHealthyStep(domain);
    if (profile)
        *profile = {};
    if (!std::isfinite(dt) || dt <= 0)
        throw std::invalid_argument("Fixed physics dt must be positive and finite");
    try {
        const auto &manager = SceneManager::Instance();
        const auto epoch = domain.owner.simulation.simulationEpoch, index = domain.owner.simulation.fixedStepIndex;
        if (domain.stepPending) {
            if (domain.stepEpoch != epoch || domain.stepIndex != index)
                throw std::logic_error("Physics step must be consumed with its original epoch and index");
            if (domain.stepCompleted) {
                domain.stepPending = domain.stepCompleted = false;
                domain.fixedGpuPhase = PhysicsDomain::FixedGpuPhase::Idle;
                domain.admittedParameters.reset();
                domain.admittedParticipants = {};
                return true;
            }
        } else {
            // A registered observation owns the earliest safe state before any
            // later numerical writer, including the next fixed integration.
            if (!ServiceParticipantOutputs() || !EnsureGpuReady(domain))
                return false;
            RefreshGpuSurfaces(domain);
            domain.stepEpoch = epoch;
            domain.stepIndex = index;
            domain.admittedDeltaTime = dt;
            domain.admittedParameters = StepParameters(dt, domain.gravity);
            domain.admittedParticipants = CapturePhysicsParticipants(domain.participants);
            // Own this fixed identity before waiting for an earlier visual
            // control. Scripts and accounting cannot be replayed meanwhile.
            domain.stepPending = true;
            domain.solverStarted = false;
            if (domain.controlPending) {
                domain.fixedGpuPhase = PhysicsDomain::FixedGpuPhase::RetireControl;
            } else if (domain.waitingForSlot) {
                (void)BeginGpuControl(domain, true);
                domain.fixedGpuPhase = PhysicsDomain::FixedGpuPhase::RetireControl;
            } else {
                AdmissionScope admission(*m_gpuScene, domain.owner);
                domain.solverStarted = m_gpuScene->TryBeginTick(
                    domain.owner, epoch, index, dt, *domain.admittedParameters,
                    [this, &domain] { return PrepareRenderConsumer(domain, true); },
                    [this, &domain](const std::shared_ptr<const JoltGpuScene::Publication> &p) {
                        return PrepareContactObserver(domain, p);
                    },
                    [&domain] { return domain.admittedParticipants; });
                domain.fixedGpuPhase = PhysicsDomain::FixedGpuPhase::Solver;
            }
        }
        if (!AdvanceFixedStep(domain, false))
            return false;
        domain.stepPending = domain.stepCompleted = false;
        domain.fixedGpuPhase = PhysicsDomain::FixedGpuPhase::Idle;
        domain.admittedParameters.reset();
        domain.admittedParticipants = {};
        return true;
    } catch (...) {
        FailStep(domain);
        throw;
    }
}
bool PhysicsWorld::AdvanceFixedStep(PhysicsDomain &domain, bool synchronous)
{
    RequireHealthyStep(domain);
    AdmissionScope admission(*m_gpuScene, domain.owner);
    try {
        for (;;) {
            if (domain.fixedGpuPhase == PhysicsDomain::FixedGpuPhase::RetryCreation) {
                RetryCapacityBodyCreations(domain);
                RefreshGpuSurfaces(domain);
                if (domain.waitingForSlot) {
                    (void)BeginGpuControl(domain, true);
                    domain.fixedGpuPhase = PhysicsDomain::FixedGpuPhase::RetireControl;
                } else {
                    domain.solverStarted = m_gpuScene->TryBeginTick(
                        domain.owner, domain.stepEpoch, domain.stepIndex, domain.admittedDeltaTime,
                        *domain.admittedParameters, [this, &domain] { return PrepareRenderConsumer(domain, true); },
                        [this, &domain](const std::shared_ptr<const JoltGpuScene::Publication> &p) {
                            return PrepareContactObserver(domain, p);
                        },
                        [&domain] { return domain.admittedParticipants; });
                    domain.fixedGpuPhase = PhysicsDomain::FixedGpuPhase::Solver;
                }
            }
            if (domain.fixedGpuPhase == PhysicsDomain::FixedGpuPhase::RetireControl) {
                if (!domain.controlPending && !BeginGpuControl(domain, true)) {
                    if (synchronous) {
                        DrainGpuSource(domain);
                        continue;
                    }
                    return false;
                }
                if (!AdvanceGpuControl(domain, synchronous))
                    return false;
                domain.fixedGpuPhase = PhysicsDomain::FixedGpuPhase::RetryCreation;
                continue;
            }
            if (!domain.solverStarted) {
                if (!(domain.solverStarted = m_gpuScene->TryBeginTick(
                          domain.owner, domain.stepEpoch, domain.stepIndex, domain.admittedDeltaTime,
                          *domain.admittedParameters, [this, &domain] { return PrepareRenderConsumer(domain, true); },
                          [this, &domain](const std::shared_ptr<const JoltGpuScene::Publication> &p) {
                              return PrepareContactObserver(domain, p);
                          },
                          [&domain] { return domain.admittedParticipants; }))) {
                    if (synchronous) {
                        DrainGpuSource(domain);
                        continue;
                    }
                    return false;
                }
            }
            if (!AdvanceGpuJob(domain, synchronous))
                return false;
            if (m_gpuScene->PublishCompletedTick(domain.owner))
                ObserveGpuPublication(domain, true);
            domain.stepCompleted = true;
            return true;
        }
    } catch (...) {
        FailStep(domain);
        throw;
    }
}
void PhysicsWorld::RetryCapacityBodyCreations(PhysicsDomain &domain)
{
    const WorldExecutionScope ownerScope(&domain.owner);
    auto &world = domain.owner;
    domain.waitingForSlot = false;
    auto &store = PhysicsECSStore::Instance();
    // Revisit only the existing actor factory/residency queues. No scripts,
    // clocks, fixed callbacks or unrelated world work are replayed here.
    for (const auto handle : store.ConsumePendingBodyCreations(world))
        if (store.IsValid(handle) && store.GetActor(handle).bodyId == UINT32_MAX)
            store.RegisterActor(handle);
    for (const auto id : store.ConsumePendingBroadphaseRemoves(world))
        RemoveBodyFromBroadphase(id);
    AddBodiesBatch(store.ConsumePendingBroadphaseAdds(world));
    store.DrainPendingJoints(world);
}
bool PhysicsWorld::HasPendingStep() const noexcept
{
    auto &domain = CurrentDomain();
    return domain.stepPending;
}
bool PhysicsWorld::HasDomainControl(PhysicsDomain &domain) const
{
    const auto &world = domain.owner;
    const auto &store = PhysicsECSStore::Instance();
    return domain.pendingRagdollHead || domain.controlPending ||
           (domain.cpuControl && domain.cpuControl->HasPending()) ||
           (domain.gpuRegistered && m_gpuScene->HasPendingControl(domain.owner)) || store.HasDirtyColliders(world) ||
           store.HasPendingBodyCreations(world) || store.HasPendingBroadphaseAdds(world) ||
           store.HasPendingBroadphaseRemoves(world) || store.HasPendingJointDependencies(world) ||
           (domain.gpuRegistered && !domain.gpuCancelled &&
            PhysicMaterial::GetSurfaceRevision() != domain.surfaceRevision);
}
void PhysicsWorld::DrainGpuSource(PhysicsDomain &domain) const
{
    AdmissionScope admission(*m_gpuScene, domain.owner);
    struct RequiredMemberScope
    {
        JoltGpuScene &scene;
        RuntimeWorld *previous;
        ~RequiredMemberScope()
        {
            scene.SetRequiredWorld(previous);
        }
    } required{*m_gpuScene, m_gpuScene->SetRequiredWorld(&domain.owner)};
    m_gpuWork->DrainRequired(
        m_gpuScene->GetSourceId(), [this, &domain] { return Availability(*m_gpuHost); },
        [this, &domain](FrameWorkOwner::SourceId source, uint64_t maximum) {
            return source == m_gpuScene->GetSourceId() ? m_gpuScene->WaitForSourceProgress(maximum)
                                                       : FrameWorkWaitResult::Blocked;
        });
}
bool PhysicsWorld::AdvanceGpuJob(PhysicsDomain &domain, bool synchronous) const
{
    if (!m_gpuScene->IsWriterFor(domain.owner) &&
        m_gpuScene->GetTickStatus(domain.owner) != JoltGpuTick::Status::Complete)
        return false;
    if (synchronous) {
        DrainGpuSource(domain);
    } else {
        if (!m_gpuWork->IsActive())
            return false;
        AdmissionScope admission(*m_gpuScene, domain.owner);
        (void)m_gpuWork->AdvanceDependency(m_gpuScene->GetSourceId(),
                                           [this, &domain] { return Availability(*m_gpuHost); });
    }
    if (m_gpuScene->GetTickStatus(domain.owner) == JoltGpuTick::Status::Failed)
        throw std::runtime_error("GPU physics step failed; stop the runtime before restarting");
    return m_gpuScene->GetTickStatus(domain.owner) == JoltGpuTick::Status::Complete;
}
bool PhysicsWorld::BeginGpuControl(PhysicsDomain &domain, bool numericalChange, const JoltGpuScene::Consumer &consumer,
                                   bool requirePublication) const
{
    RequireGpu(domain);
    numericalChange = numericalChange || m_gpuScene->HasPendingControl(domain.owner);
    AdmissionScope admission(*m_gpuScene, domain.owner);
    FrameWorkOwner::Scope preparation(*m_gpuWork);
    auto rendering = PrepareRenderConsumer(domain, numericalChange);
    if (!numericalChange && !consumer && !rendering && !requirePublication)
        return true;
    if (!m_gpuScene->TryBeginControl(domain.owner, domain.owner.simulation.simulationEpoch, consumer,
                                     std::move(rendering)))
        return false;
    domain.controlNumericalChange = numericalChange;
    domain.controlPending = true;
    return true;
}
bool PhysicsWorld::AdvanceGpuControl(PhysicsDomain &domain, bool synchronous) const
{
    if (!domain.controlPending)
        return true;
    if (!AdvanceGpuJob(domain, synchronous))
        return false;
    if (!m_gpuScene->PublishCompletedControl(domain.owner))
        throw std::runtime_error("GPU control failed to publish its complete result");
    ObserveGpuPublication(domain, domain.controlNumericalChange);
    domain.controlPending = false;
    domain.controlNumericalChange = false;
    return true;
}
void PhysicsWorld::CompleteDomainStep(PhysicsDomain &domain)
{
    RequireOwnerThread();
    RequireHealthyStep(domain);
    if ((!domain.stepPending || domain.stepCompleted) && !domain.controlPending)
        return;
    RequireGpu(domain);
    try {
        if (domain.stepPending && !domain.stepCompleted)
            (void)AdvanceFixedStep(domain, true);
        else
            (void)AdvanceGpuControl(domain, true);
    } catch (...) {
        FailStep(domain);
        throw;
    }
}
void PhysicsWorld::ServiceGpuWork()
{
    auto &domain = CurrentDomain();
    RequireOwnerThread();
    if (!m_initialized || !domain.gpuRegistered || domain.gpuCancelled ||
        domain.fixedGpuPhase == PhysicsDomain::FixedGpuPhase::Failed)
        return;
    try {
        // A public host opportunity can continue this already admitted prefix
        // even when the renderer is waiting and skips Update/LateUpdate.
        (void)AdvanceGpuControl(domain, false);
        if (!SceneManager::Instance().HasPendingFixedStep())
            SceneManager::Instance().PublishPhysicsStateToRenderer();
    } catch (...) {
        FailStep(domain);
        throw;
    }
}
void PhysicsWorld::RefreshRenderPoses()
{
    auto &domain = CurrentDomain();
    RequireOwnerThread();
    if (!m_initialized || !domain.gpuRegistered || !m_gpuRenderProgram || domain.gpuCancelled ||
        domain.fixedGpuPhase == PhysicsDomain::FixedGpuPhase::Failed)
        return;
    ServiceGpuWork();
    if (domain.controlPending || domain.stepPending || SceneManager::Instance().HasPendingFixedStep())
        return;
    // This automatic publication uses the existing host allowance. It never
    // opens an opportunity or turns render extraction into a strict GPU wait.
    if (!m_gpuWork->IsActive() || !m_gpuWork->IsAccepting() || m_gpuWork->IsLate() || m_gpuWork->Remaining() <= 0 ||
        !m_gpuWork->ResourceRemaining(FrameWorkResource::Cpu))
        return;
    try {
        auto &rendering = domain.rendering;
        if (!HasDomainControl(domain) && rendering.checkedKeys && *rendering.checkedKeys == CaptureRenderKeys())
            return;
        {
            FrameWorkOwner::Scope preparation(*m_gpuWork);
            FlushAuthoring(domain);
        }
        while (domain.waitingForSlot) {
            if (!BeginGpuControl(domain, true))
                return;
            if (!AdvanceGpuControl(domain, false))
                return;
            RetryCapacityBodyCreations(domain);
        }
        if (BeginGpuControl(domain, m_gpuScene->HasPendingControl(domain.owner)))
            (void)AdvanceGpuControl(domain, false);
    } catch (...) {
        FailStep(domain);
        throw;
    }
}
bool PhysicsWorld::IsFixedPublicationHeld(const PhysicsDomain &domain) const noexcept
{
    const auto &simulation = domain.owner.simulation;
    return domain.stepEpoch == simulation.simulationEpoch && domain.stepIndex == simulation.fixedStepIndex &&
           (domain.contacts.dispatching || (domain.stepPending && domain.stepCompleted) ||
            (simulation.fixedHasPhysics && simulation.fixedPhase == RuntimeWorld::FixedPhase::Publication));
}
bool PhysicsWorld::ServicePausedAuthorPresentation()
{
    auto &domain = CurrentDomain();
    RequireOwnerThread();
    RequireHealthyStep(domain);
    if (IsFixedPublicationHeld(domain) || !m_initialized)
        return true;
    if (!domain.gpuRegistered || !m_gpuRenderProgram || domain.gpuCancelled || domain.retiring ||
        !m_gpuWork->IsActive() || !m_gpuWork->IsAccepting() || m_gpuWork->ServiceRemaining() <= 0 ||
        !m_gpuWork->ResourceRemaining(FrameWorkResource::Cpu))
        return false;
    try {
        // Retain the original numerical identity, dt, parameters and contact batch.
        // In particular do not call Step: its completed shortcut consumes the receipt.
        if (domain.stepPending && !domain.stepCompleted && !AdvanceFixedStep(domain, false))
            return false;
        if (IsFixedPublicationHeld(domain))
            return true;
        if (!AdvanceGpuControl(domain, false))
            return false;
        SceneManager::Instance().PublishPhysicsStateToRenderer();
        {
            FrameWorkOwner::Scope preparation(*m_gpuWork);
            TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
            // This dirty sync deliberately bypasses query's synchronous fixed drain.
            SceneManager::Instance().SyncDraggedTransforms(0.0f, false);
            RefreshGpuSurfaces(domain);
        }
        if (domain.waitingForSlot) {
            if (!BeginGpuControl(domain, true) || !AdvanceGpuControl(domain, false))
                return false;
            RetryCapacityBodyCreations(domain);
            return false;
        }
        auto &rendering = domain.rendering;
        if (!HasDomainControl(domain) && rendering.checkedKeys && *rendering.checkedKeys == CaptureRenderKeys())
            return true;
        if (!BeginGpuControl(domain, m_gpuScene->HasPendingControl(domain.owner)) || !AdvanceGpuControl(domain, false))
            return false;
        SceneManager::Instance().PublishPhysicsStateToRenderer();
        return !HasDomainControl(domain) && !domain.waitingForSlot && rendering.checkedKeys &&
               *rendering.checkedKeys == CaptureRenderKeys();
    } catch (...) {
        FailStep(domain);
        throw;
    }
}

void PhysicsWorld::CancelPendingStep() noexcept
{
    auto &domain = CurrentDomain();
    SuspendRagdolls(domain);
    domain.contacts.Close();
    domain.gpuContacts.Close();
    domain.cpuContacts.Close();
    domain.cpuControl.reset();
    domain.cpuScene.reset();
    domain.presentationPoseReads.reset();
    auto &world = domain.owner;
    if (m_gpuRenderProgram) {
        domain.rendering.bindings.Clear();
        domain.rendering.pending.reset();
        domain.rendering.ready.reset();
        domain.rendering.checkedKeys.reset();
    }
    auto &store = PhysicsECSStore::Instance();
    store.ForEachAliveActor([&](PhysicsActorData &actor) {
        if (actor.runtimeWorld != &world || actor.bodyId == UINT32_MAX)
            return;
        // A backend may already be closed. The public directory still owns
        // these identities until this boundary.
        const auto *registration = FindBodyRegistration(actor.bodyId);
        if (!registration || &store.GetActor(registration->actor) != &actor)
            return;
        store.CancelBroadphaseAdd(world, actor.bodyId);
        store.CancelBroadphaseRemove(world, actor.bodyId);
        domain.recreateActors.push_back(registration->actor);
        UnregisterBodyIdentity(domain, actor.bodyId);
        actor.bodyId = UINT32_MAX;
        actor.bodyInBroadphase = false;
        actor.jointEndpointBodyId = UINT32_MAX;
    });
    if (domain.gpuRegistered) {
        m_gpuScene->Cancel(domain.owner);
        domain.gpuCancelled = true;
    }
    store.ForEachAliveRigidbody([&](RigidbodyECSData &body) {
        if (store.IsValid(body.actorHandle) && store.GetActor(body.actorHandle).runtimeWorld == &world)
            body.physicsTargetSources = {};
    });
    domain.stepPending = domain.stepCompleted = false;
    domain.fixedGpuPhase = PhysicsDomain::FixedGpuPhase::Idle;
    domain.admittedParameters.reset();
    domain.admittedParticipants = {};
    domain.waitingForSlot = false;
    domain.controlPending = domain.controlNumericalChange = false;
    domain.participantOutputPending = false;
    domain.requestedMotion.clear();
}
void PhysicsWorld::FlushAuthoring(PhysicsDomain &domain) const
{
    const WorldExecutionScope ownerScope(&domain.owner);
    if (domain.flushingAuthoring)
        return;
    FlagScope flushing(domain.flushingAuthoring);
    TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    SceneManager::Instance().EnsurePhysicsQueriesCurrent();
    if (domain.gpuRegistered && !domain.gpuCancelled)
        const_cast<PhysicsWorld *>(this)->RefreshGpuSurfaces(domain);
}
void PhysicsWorld::RunControl(PhysicsDomain &domain, const JoltGpuScene::Consumer &consumer, bool publicationRead) const
{
    RequireOwnerThread();
    const WorldExecutionScope ownerScope(&domain.owner);
    RequireHealthyStep(domain);
    RequireGpu(domain);
    if (domain.synchronousRead)
        throw std::logic_error("Physics query cannot reenter its synchronous source");
    FlagScope reading(domain.synchronousRead);
    const auto observe = [&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        AdmissionScope admission(*m_gpuScene, domain.owner);
        JoltGpuControlWork prepared;
        {
            FrameWorkOwner::Scope preparation(*m_gpuWork);
            if (consumer)
                prepared = consumer(publication);
        }
        m_gpuScene->RunPublicationObservation(domain.owner, publication, std::move(prepared));
    };
    // Completing an admitted solver can establish the Fixed publication hold.
    if (!IsFixedPublicationHeld(domain))
        const_cast<PhysicsWorld *>(this)->CompleteDomainStep(domain);
    if (IsFixedPublicationHeld(domain)) {
        if (!consumer)
            return;
        const auto publication = domain.gpuContacts.GetPublication();
        if (!publication || !publication->isFixedStep || publication->worldId != domain.owner.id ||
            publication->simulationEpoch != domain.stepEpoch || publication->fixedStepIndex != domain.stepIndex)
            throw std::logic_error("Physics query has no matching completed Fixed publication");
        if (publicationRead)
            observe(publication);
        else {
            FrameWorkOwner::Scope preparation(*m_gpuWork);
            RunPreparedWork(domain, consumer(publication));
        }
        return;
    }
    try {
        {
            FrameWorkOwner::Scope preparation(*m_gpuWork);
            FlushAuthoring(domain);
        }
        while (domain.waitingForSlot) {
            while (!BeginGpuControl(domain, true))
                DrainGpuSource(domain);
            (void)AdvanceGpuControl(domain, true);
            const_cast<PhysicsWorld *>(this)->RetryCapacityBodyCreations(domain);
        }
        if (publicationRead) {
            auto publication = m_gpuScene->GetPublished(domain.owner);
            if (HasDomainControl(domain) || !m_gpuScene->IsCurrentPublication(domain.owner, publication)) {
                // Complete a real author/layout source dependency, never an alternate query execution path.
                while (!BeginGpuControl(domain, m_gpuScene->HasPendingControl(domain.owner), {}, true))
                    DrainGpuSource(domain);
                (void)AdvanceGpuControl(domain, true);
                publication = m_gpuScene->GetPublished(domain.owner);
            }
            observe(publication);
            return;
        }
        while (!BeginGpuControl(domain, m_gpuScene->HasPendingControl(domain.owner), consumer))
            DrainGpuSource(domain);
        (void)AdvanceGpuControl(domain, true);
    } catch (...) {
        const_cast<PhysicsWorld *>(this)->FailStep(domain);
        throw;
    }
}

std::shared_ptr<JoltGpuQueryResult> PhysicsWorld::RunQuery(PhysicsDomain &domain,
                                                           const std::vector<QueryInput> &requests,
                                                           uint64_t sweepColliderId,
                                                           uint64_t sweepColliderGeneration) const
{
    JoltGpuPreparedQuery prepared;
    std::exception_ptr requestError;
    RunControl(domain, [&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) -> JoltGpuControlWork {
        return PrepareQueryWork(
            [&] {
                const auto *inputs = &requests;
                std::vector<QueryInput> sweepInputs;
                if (sweepColliderId) {
                    sweepInputs = requests;
                    for (auto &input : sweepInputs) {
                        input.meta.w |= QuerySweepPairPolicy;
                        input.lifetime = {static_cast<uint32_t>(sweepColliderGeneration),
                                          static_cast<uint32_t>(sweepColliderGeneration >> 32), 0, 0};
                    }
                    // Only this accepted publication can name the source slot.
                    // An unregistered Collider still excludes its GameObject.
                    for (const auto &actor : publication->directory->actors) {
                        if (!actor || !std::any_of(actor->leaves.begin(), actor->leaves.end(), [&](const auto &leaf) {
                                return leaf.colliderId == sweepColliderId &&
                                       leaf.colliderGeneration == sweepColliderGeneration;
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
                prepared = m_gpuQueries->Prepare(publication, *inputs, {},
                                                 domain.queryGeneration.load(std::memory_order_acquire) +
                                                     static_cast<uint64_t>(domain.controlNumericalChange));
                return std::move(prepared.work);
            },
            requestError);
    });
    if (requestError)
        std::rethrow_exception(requestError);
    return std::move(prepared.result);
}

std::shared_ptr<JoltGpuQueryResult> PhysicsWorld::RunGeometryQuery(PhysicsDomain &domain, const QueryInput &request,
                                                                   const QueryPrimitive &other,
                                                                   JoltGpuQueryGeometry geometry) const
{
    RequireOwnerThread();
    RequireHealthyStep(domain);
    RequireGpu(domain);
    AdmissionScope admission(*m_gpuScene, domain.owner);
    if (domain.synchronousRead)
        throw std::logic_error("Physics geometry query cannot reenter its synchronous source");
    FlagScope reading(domain.synchronousRead);
    // Finish only a logical operation already owned by this source. Do not
    // flush or admit queued author input, and never start a new fixed step.
    const_cast<PhysicsWorld *>(this)->CompleteDomainStep(domain);
    JoltGpuPreparedQuery prepared;
    std::exception_ptr requestError;
    {
        FrameWorkOwner::Scope preparation(*m_gpuWork);
        auto work = PrepareQueryWork(
            [&]() {
                if (!geometry.store)
                    geometry.store = m_gpuScene->GetGeometryStore();
                prepared = m_gpuQueries->Prepare({}, {request}, {other}, 0, JoltGpuQueryReadback::Host, 32,
                                                 std::move(geometry));
                return std::move(prepared.work);
            },
            requestError);
        while (!m_gpuScene->TryBeginWork(domain.owner, work))
            DrainGpuSource(domain);
    }
    (void)AdvanceGpuJob(domain, true);
    if (requestError)
        std::rethrow_exception(requestError);
    return std::move(prepared.result);
}

std::vector<Collider *> PhysicsWorld::QueryColliders(PhysicsDomain &domain, const QueryInput &request) const
{
    auto result = RunQuery(domain, {request});
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
    if (record.meta.x >= publication.directory->actors.size() || !publication.directory->actors[record.meta.x] ||
        record.meta.z >= publication.directory->leaves.size())
        throw std::runtime_error("GPU query address is absent from its retained publication");
    const auto &identity = *publication.directory->actors[record.meta.x];
    const auto &address = publication.directory->leaves[record.meta.z];
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
    auto &domain = DomainForActor(actor);
    RequireOwnerThread();
    auto &store = PhysicsECSStore::Instance();
    const auto handle = store.FindActor(actor.owner);
    if (!store.IsValid(handle))
        throw std::logic_error("Physics body creation requires an ECS actor");
    if (!EnsureGpuReady(domain)) {
        store.QueueBodyCreation(handle);
        return UINT32_MAX;
    }
    TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    auto prepared = PrepareJoltGpuActor(actor, 0, 0);
    if (!prepared)
        return UINT32_MAX;
    if (m_gpuScene->GetBodyCount(domain.owner) >= m_gpuScene->GetCapacity(domain.owner)) {
        const uint64_t capacity =
            std::max(uint64_t{InitialBodyCapacity}, uint64_t{2} * m_gpuScene->GetCapacity(domain.owner));
        if (capacity > UINT32_MAX)
            throw std::length_error("GPU physics world exceeds body indexing");
        m_gpuScene->ReserveCapacity(domain.owner, static_cast<uint32_t>(capacity));
    }
    if (!m_gpuScene->CanReserve(domain.owner)) {
        domain.waitingForSlot = true;
        store.QueueBodyCreation(handle);
        return UINT32_MAX;
    }
    // RegisterActor creates only an active actor. Install its first model as resident;
    // the later broadphase add still flushes pending forces through the same source.
    const auto allocation =
        m_gpuScene->Reserve(domain.owner, store.AllocateBodyIdentity(), handle,
                            {PhysicsTargetReference(actor.owner->GetTransform()), actor.owner->GetID()}, true);
    try {
        RegisterBodyIdentity(domain, allocation.bodyId, handle, Backend::Gpu);
        actor.bodyId = allocation.bodyId;
        m_gpuScene->StageActor(allocation.bodyId, std::move(*prepared));
    } catch (...) {
        if (actor.bodyId == allocation.bodyId) {
            UnregisterBodyIdentity(domain, allocation.bodyId);
            actor.bodyId = UINT32_MAX;
        }
        m_gpuScene->StageRetire(allocation.bodyId);
        throw;
    }
    return allocation.bodyId;
}
void PhysicsWorld::DestroyBody(uint32_t id)
{
    RequireOwnerThread();
    const auto *registration = FindBodyRegistration(id);
    if (!registration)
        return;
    auto &domain = DomainForActor(PhysicsECSStore::Instance().GetActor(registration->actor));
    switch (registration->backend) {
    case Backend::Gpu:
        // Closed partitions retire their numerical records as a batch.
        if (m_gpuScene && m_gpuScene->FindBody(id).bodyId != UINT32_MAX)
            m_gpuScene->StageRetire(id);
        if (m_gpuRenderProgram)
            domain.rendering.bindings.RetireBody(id);
        break;
    case Backend::Cpu:
    case Backend::StaticBoundary:
        throw std::logic_error("Physics backend body retirement is unavailable");
    }
    UnregisterBodyIdentity(domain, id);
    domain.requestedMotion.erase(id);
}
bool PhysicsWorld::CanQueueRigidbodyCommand(uint32_t id, const Rigidbody &body) const
{
    RequireOwnerThread();
    return m_gpuScene && m_gpuScene->HasStagedRigidbody(id, body.GetECSHandle());
}
void PhysicsWorld::StageUpdatedGpuActor(const PhysicsActorData &actor, PreparedGpuActor prepared)
{
    m_gpuScene->StageActor(actor.bodyId, std::move(prepared));
    if (auto *body = actor.rigidbody)
        body->FlushPendingBodyCommands();
}
void PhysicsWorld::RefreshGpuActor(PhysicsDomain &domain, uint32_t id,
                                   const std::function<void(PreparedGpuActor &)> &edit)
{
    RequireGpu(domain);
    auto &actor = ActorForBody(id);
    TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    auto prepared = PrepareJoltGpuActor(actor, m_gpuScene->FindBody(id).slot, 0);
    if (!prepared)
        return;
    if (edit)
        edit(*prepared);
    StageUpdatedGpuActor(actor, std::move(*prepared));
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
    auto &domain = DomainForActor(actor);
    RequireGpu(domain);
    if (!exclude) {
        RefreshGpuActor(domain, actor.bodyId);
        return;
    }
    auto author = actor;
    author.colliders.erase(std::remove(author.colliders.begin(), author.colliders.end(), exclude),
                           author.colliders.end());
    TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    auto prepared = PrepareJoltGpuActor(author, m_gpuScene->FindBody(actor.bodyId).slot, 0);
    if (prepared)
        StageUpdatedGpuActor(actor, std::move(*prepared));
}
void PhysicsWorld::SetBodyPosition(uint32_t id, const glm::vec3 &position, const glm::quat &rotation,
                                   bool cancelKinematicMove)
{
    auto &domain = DomainForBody(id);
    RequireGpu(domain);
    BodyCommand c{};
    c.meta.y = CommandSetPose;
    c.meta.z = cancelKinematicMove ? 0 : CommandPreserveKinematicMove;
    c.value = Vector(position);
    c.auxiliary = Rotation(rotation);
    m_gpuScene->QueueCommand(id, c);
    domain.rendering.bindings.MarkDiscontinuity(id);
}
void PhysicsWorld::SetBodyPositionOnly(uint32_t id, const glm::vec3 &position)
{
    auto &domain = DomainForBody(id);
    RequireGpu(domain);
    BodyCommand c{};
    c.meta.y = CommandSetPosition;
    c.value = Vector(position);
    m_gpuScene->QueueCommand(id, c);
    domain.rendering.bindings.MarkDiscontinuity(id);
}
void PhysicsWorld::SetBodyRotationOnly(uint32_t id, const glm::quat &rotation)
{
    auto &domain = DomainForBody(id);
    RequireGpu(domain);
    BodyCommand c{};
    c.meta.y = CommandSetRotation;
    c.auxiliary = Rotation(rotation);
    m_gpuScene->QueueCommand(id, c);
    domain.rendering.bindings.MarkDiscontinuity(id);
}
void PhysicsWorld::SetBodyPositionsBatch(const std::vector<PhysicsBodyPoseUpdate> &updates)
{
    if (updates.empty())
        return;
    for (const auto &u : updates) {
        RequireGpu(DomainForBody(u.bodyId));
        (void)ActorForBody(u.bodyId);
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
    for (const auto id : bodies)
        RequireGpu(DomainForBody(id));
    for (const auto id : bodies)
        m_gpuScene->SetResident(id, true);
    for (const auto id : bodies)
        if (auto *body = FindRigidbodyByBodyId(id))
            body->FlushPendingBodyCommands();
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
    auto &domain = DomainForBody(id);
    RunControl(domain, {});
    if (m_gpuScene->FindBody(id).bodyId == UINT32_MAX)
        return false;
    const auto &actor = ActorForBody(id);
    return actor.bodyInBroadphase && !actor.sceneSuspended;
}
void PhysicsWorld::SetBodyMotionType(uint32_t id, int type)
{
    auto &domain = DomainForBody(id);
    if (type < 0 || type > 2)
        throw std::invalid_argument("Physics motion type must be static, kinematic or dynamic");
    RefreshGpuActor(domain, id);
}
void PhysicsWorld::RefreshBodyLayers(uint32_t id)
{
    auto &domain = DomainForBody(id);
    RefreshGpuActor(domain, id);
}
void PhysicsWorld::SetBodyDamping(uint32_t id, float linear, float angular)
{
    auto &domain = DomainForBody(id);
    if (!std::isfinite(linear) || !std::isfinite(angular) || linear < 0 || angular < 0)
        throw std::invalid_argument("Damping must be finite and nonnegative");
    RefreshGpuActor(domain, id, [=](PreparedGpuActor &a) {
        a.body.damping.x = linear;
        a.body.damping.y = angular;
    });
}
void PhysicsWorld::SetBodyGravityFactor(uint32_t id, float factor)
{
    auto &domain = DomainForBody(id);
    if (!std::isfinite(factor))
        throw std::invalid_argument("Gravity factor must be finite");
    RefreshGpuActor(domain, id, [=](PreparedGpuActor &a) { a.body.mass.w = factor; });
}
void PhysicsWorld::SetBodyFriction(uint32_t id, float value)
{
    auto &domain = DomainForBody(id);
    RefreshGpuActor(domain, id, [=](PreparedGpuActor &a) {
        for (auto &leaf : a.geometry.leaves)
            leaf.model.material.x = value;
    });
}
void PhysicsWorld::SetBodyRestitution(uint32_t id, float value)
{
    auto &domain = DomainForBody(id);
    RefreshGpuActor(domain, id, [=](PreparedGpuActor &a) {
        for (auto &leaf : a.geometry.leaves)
            leaf.model.material.y = value;
    });
}
void PhysicsWorld::SetBodyMaxAngularVelocity(uint32_t id, float value)
{
    auto &domain = DomainForBody(id);
    RefreshGpuActor(domain, id, [=](PreparedGpuActor &a) { a.body.damping.w = value; });
}
void PhysicsWorld::SetBodyMaxLinearVelocity(uint32_t id, float value)
{
    auto &domain = DomainForBody(id);
    RefreshGpuActor(domain, id, [=](PreparedGpuActor &a) { a.body.damping.z = value; });
}
void PhysicsWorld::SetBodyAllowedDOFs(uint32_t id, int allowed, float mass)
{
    auto &domain = DomainForBody(id);
    if (allowed < 0 || (allowed & ~0x3f) != 0)
        throw std::invalid_argument("Allowed degrees of freedom contain unknown bits");
    if (!std::isfinite(mass) || mass <= 0)
        throw std::invalid_argument("Body mass must be positive and finite");
    RefreshGpuActor(domain, id, [=](PreparedGpuActor &a) {
        a.body.mass.y = float((~allowed) & 7);
        a.body.mass.z = float(((~allowed) >> 3) & 7);
    });
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
    if (shape && !(body.IsKinematic() && shape->MustBeStatic()))
        (void)PrepareBodyMass(shape.GetPtr(), mass, nullptr, scale);
}
void PhysicsWorld::SetBodyLinearVelocity(uint32_t id, const glm::vec3 &v)
{
    auto &domain = DomainForBody(id);
    RequireGpu(domain);
    BodyCommand c{};
    c.meta.y = CommandSetLinearVelocity;
    c.value = Vector(v);
    m_gpuScene->QueueCommand(id, c);
}
void PhysicsWorld::SetBodyAngularVelocity(uint32_t id, const glm::vec3 &v)
{
    auto &domain = DomainForBody(id);
    RequireGpu(domain);
    BodyCommand c{};
    c.meta.y = CommandSetAngularVelocity;
    c.value = Vector(v);
    m_gpuScene->QueueCommand(id, c);
}
void PhysicsWorld::QueueBodyForce(uint32_t id, int kind, int mode, const glm::vec3 &value, const glm::vec3 &point)
{
    auto &domain = DomainForBody(id);
    RequireGpu(domain);
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
    auto &domain = CurrentDomain();
    RequireOwnerThread();
    RequireGpu(domain);
    if (HasPendingStep() || domain.contacts.dispatching)
        throw std::logic_error("Physics reset requires a boundary outside contact delivery");
}
void PhysicsWorld::ResetBodyMotionStates(const std::vector<PhysicsBodyMotionState> &states)
{
    if (states.empty())
        return;
    for (const auto &s : states) {
        auto &domain = DomainForBody(s.bodyId);
        RequireGpu(domain);
        if (domain.stepPending)
            throw std::logic_error("Physics reset requires completed fixed steps");
        (void)ActorForBody(s.bodyId);
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

JoltGpuControlWork PhysicsWorld::PrepareStateBatch(
    PhysicsDomain &domain, const std::shared_ptr<const JoltGpuScene::Publication> &publication,
    const std::vector<Rigidbody *> &bodies, PhysicsStateFields fields, bool includeBoxes, bool triggers,
    const std::vector<JoltGpuQueryExportBinding> &exports, PhysicsRigidbodyStateBatch &batch,
    std::shared_ptr<JoltGpuQueryResult> &motion, std::shared_ptr<JoltGpuQueryResult> &shapes,
    std::exception_ptr &requestError, bool deferExport) const
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
                if (allocation.bodyId == UINT32_MAX || allocation.slot >= publication->directory->actors.size() ||
                    !publication->directory->actors[allocation.slot])
                    throw JoltGpuQueryRequestError("Physics batch requires accepted body identities");
                const auto &identity = *publication->directory->actors[allocation.slot];
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
            const uint64_t generation = domain.queryGeneration.load(std::memory_order_acquire) +
                                        static_cast<uint64_t>(domain.controlNumericalChange);
            batch.sources = {
                {publication->worldId, publication->simulationEpoch, publication->fixedStepIndex, generation}};
            batch.sourceForBody.assign(bodies.size(), 0u);
            const auto readback =
                (exports.empty() && !deferExport) ? JoltGpuQueryReadback::Host : JoltGpuQueryReadback::Device;
            JoltGpuPreparedQuery first, second;
            if (!motionInputs.empty()) {
                first = m_gpuQueries->Prepare(publication, motionInputs, {}, generation, readback);
                motion = first.result;
            }
            if (!shapeInputs.empty()) {
                second = m_gpuQueries->Prepare(publication, shapeInputs, {}, generation, readback);
                shapes = second.result;
            }
            return AfterWork(std::move(first.work), [this, &domain, motionResult = motion, shapeResult = shapes,
                                                     work = std::move(second.work), &batch, exports,
                                                     deferExport]() mutable {
                if (motionResult && motionResult->GetSummary().counts.y != QuerySuccess)
                    return JoltGpuControlWork{};
                return AfterWork(std::move(work),
                                 [this, &domain, motionResult, shapeResult, &batch, exports, deferExport] {
                                     if (shapeResult) {
                                         const auto &summary = shapeResult->GetSummary();
                                         if (summary.counts.y != QuerySuccess)
                                             return JoltGpuControlWork{};
                                         batch.boxCount = summary.counts.x;
                                     }
                                     if (exports.empty() || deferExport)
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

void PhysicsWorld::MaterializeStateBatch(PhysicsDomain &domain, PhysicsRigidbodyStateBatch &batch,
                                         const std::shared_ptr<JoltGpuQueryResult> &motion,
                                         const std::shared_ptr<JoltGpuQueryResult> &shapes, bool numerical,
                                         PhysicsStateFields cachedFields) const
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
                domain.requestedMotion[batch.states[index].bodyId] = {motion->numericalGeneration, batch.states[index],
                                                                      cachedFields};
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
                    const auto &address = shapes->publication->directory->leaves.at(record.meta.z);
                    if (address.bodySlot != record.meta.x)
                        throw std::runtime_error("GPU box batch differs from its published leaf owner");
                    const auto &leaf =
                        shapes->publication->directory->actors.at(record.meta.x)->leaves.at(address.actorLeaf);
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
    RequireOwnerThread();
    PhysicsRigidbodyStateBatch batch;
    batch.bodies = bodies;
    batch.states.resize(bodies.size());
    batch.sourceForBody.resize(bodies.size());
    if (bodies.empty())
        return batch;
    struct Group
    {
        PhysicsDomain *domain = nullptr;
        std::vector<Rigidbody *> bodies;
        std::vector<size_t> indices;
        PhysicsRigidbodyStateBatch batch;
        std::shared_ptr<JoltGpuQueryResult> motion, shapes;
        std::shared_ptr<const JoltGpuScene::Publication> publication;
    };
    std::vector<Group> groups;
    std::unordered_map<PhysicsDomain *, size_t> byOwner;
    for (size_t row = 0; row < bodies.size(); ++row) {
        auto *body = bodies[row];
        if (!body || !body->IsEnabled() || body->IsDestroyed() || !body->GetGameObject())
            throw std::invalid_argument("Physics batch requires live enabled Rigidbody components");
        auto &domain = DomainForComponent(*body);
        const auto [entry, inserted] = byOwner.emplace(&domain, groups.size());
        if (inserted)
            groups.push_back(Group{&domain});
        auto &group = groups[entry->second];
        group.bodies.push_back(body);
        group.indices.push_back(row);
    }
    std::vector<uint32_t> shapeCounts(bodies.size(), 0u);
    std::vector<std::vector<PhysicsRigidbodyBoxState>> boxes(bodies.size());
    for (auto &group : groups) {
        auto &domain = *group.domain;
        std::exception_ptr error;
        RunControl(
            domain,
            [&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
                group.publication = publication;
                return PrepareStateBatch(domain, publication, group.bodies, fields, includeBoxes, triggers, exports,
                                         group.batch, group.motion, group.shapes, error, !exports.empty());
            },
            exports.empty());
        if (error)
            std::rethrow_exception(error);
        MaterializeStateBatch(domain, group.batch, group.motion, group.shapes, exports.empty());
        const uint32_t source = static_cast<uint32_t>(batch.sources.size());
        const auto &p = *group.publication;
        const uint64_t generation = group.motion   ? group.motion->numericalGeneration
                                    : group.shapes ? group.shapes->numericalGeneration
                                                   : domain.queryGeneration.load(std::memory_order_acquire);
        batch.sources.push_back({p.worldId, p.simulationEpoch, p.fixedStepIndex, generation});
        const auto *spans = group.shapes ? &group.shapes->ReadSpans() : nullptr;
        for (size_t local = 0; local < group.indices.size(); ++local) {
            const auto row = group.indices[local];
            batch.states[row] = group.batch.states[local];
            batch.sourceForBody[row] = source;
            if (spans)
                shapeCounts[row] = (*spans)[local].range.y;
        }
        for (auto box : group.batch.boxes) {
            box.bodyIndex = static_cast<uint32_t>(group.indices.at(box.bodyIndex));
            boxes[box.bodyIndex].push_back(std::move(box));
        }
    }
    std::vector<uint64_t> shapeFirst(bodies.size());
    uint64_t boxCount = 0;
    for (size_t row = 0; row < bodies.size(); ++row) {
        shapeFirst[row] = boxCount;
        boxCount += shapeCounts[row];
        batch.boxes.insert(batch.boxes.end(), boxes[row].begin(), boxes[row].end());
    }
    if (boxCount > UINT32_MAX)
        throw std::length_error("Physics shape aggregate exceeds device indexing");
    batch.boxCount = static_cast<size_t>(boxCount);
    constexpr std::array<uint32_t, 13> widths{3, 4, 3, 3, 3, 3, 9, 1, 3, 3, 4, 1, 1};
    std::vector<std::pair<PhysicsDomain *, JoltGpuControlWork>> plannedExports;
    for (auto &group : groups) {
        JoltGpuControlWork work;
        std::vector<std::shared_ptr<JoltGpuQueryResult>> sources;
        if (group.motion)
            sources.push_back(group.motion);
        if (group.shapes)
            sources.push_back(group.shapes);
        for (size_t local = 0; local < group.indices.size(); ++local) {
            std::vector<JoltGpuQueryExportBinding> targets;
            const auto row = group.indices[local];
            for (auto target : exports) {
                const auto field = static_cast<uint32_t>(target.field);
                if (field >= widths.size())
                    throw JoltGpuQueryRequestError("Physics aggregate export has an unknown field");
                const bool motion = field <= static_cast<uint32_t>(JoltGpuQueryExportField::MotionInverseInertia);
                const uint64_t outputRow = motion ? row : shapeFirst[row];
                const uint64_t byteAdvance = outputRow * widths[field] * sizeof(uint32_t);
                if (byteAdvance > UINT64_MAX - target.byteOffset)
                    throw JoltGpuQueryRequestError("Physics aggregate export destination offset overflow");
                target.byteOffset += byteAdvance;
                target.recordFirst = motion ? static_cast<uint32_t>(local) : group.shapes->ReadSpans()[local].range.x;
                target.recordCount = motion ? 1u : shapeCounts[row];
                target.bodyIndex = static_cast<uint32_t>(row);
                targets.push_back(std::move(target));
            }
            if (!targets.empty()) {
                auto projected = m_gpuQueries->PrepareExport(sources, targets);
                work = AfterWork(std::move(work),
                                 [projected = std::move(projected)]() mutable { return std::move(projected); });
            }
        }
        if (work.packet || work.continueWork)
            plannedExports.emplace_back(group.domain, std::move(work));
    }
    for (auto &planned : plannedExports)
        RunPreparedWork(*planned.first, std::move(planned.second));
    return batch;
}

PhysicsRigidbodyStateBatch
PhysicsWorld::GatherRigidbodyStatesInBounds(const glm::vec3 &minimum, const glm::vec3 &maximum, uint32_t layers,
                                            bool triggers, PhysicsStateFields fields, bool includeBoxes,
                                            const std::vector<JoltGpuQueryExportBinding> &exports) const
{
    auto &domain = CurrentDomain();
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
    RunControl(domain, [&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        return PrepareQueryWork(
            [&]() {
                auto query = m_gpuQueries->Prepare(publication, {input}, {},
                                                   domain.queryGeneration.load(std::memory_order_acquire) +
                                                       static_cast<uint64_t>(domain.controlNumericalChange));
                bounds = query.result;
                return AfterWork(std::move(query.work), [this, &domain, publication, &batch, &bounds, &motion, &shapes,
                                                         fields, includeBoxes, triggers, exports, &error] {
                    if (bounds->GetSummary().counts.y != QuerySuccess)
                        return JoltGpuControlWork{};
                    std::vector<Rigidbody *> bodies;
                    auto &store = PhysicsECSStore::Instance();
                    for (const auto &record : bounds->ReadHits()) {
                        const auto &identity = *publication->directory->actors.at(record.meta.x);
                        if (!store.IsValid(identity.rigidbody))
                            throw JoltGpuQueryRequestError("Bounds result Rigidbody no longer exists");
                        bodies.push_back(store.GetRigidbody(identity.rigidbody).owner);
                    }
                    return PrepareStateBatch(domain, publication, bodies, fields, includeBoxes, triggers, exports,
                                             batch, motion, shapes, error);
                });
            },
            error);
    });
    if (error)
        std::rethrow_exception(error);
    (void)bounds->ReadSpans();
    MaterializeStateBatch(domain, batch, motion, shapes, exports.empty());
    return batch;
}

const PhysicsBodyMotionState *PhysicsWorld::FindCachedMotionState(PhysicsDomain &domain, uint32_t id,
                                                                  PhysicsStateFields fields) const
{
    const auto cached = domain.requestedMotion.find(id);
    const bool preparedPresentation =
        domain.presentationPoseReadDepth &&
        domain.presentationPoseReadGeneration == domain.queryGeneration.load(std::memory_order_acquire);
    if ((!domain.stepPending || domain.stepCompleted) && (preparedPresentation || !HasDomainControl(domain)) &&
        cached != domain.requestedMotion.end() &&
        cached->second.generation == domain.queryGeneration.load(std::memory_order_acquire) &&
        (static_cast<uint32_t>(fields) & ~static_cast<uint32_t>(cached->second.fields)) == 0)
        return &cached->second.state;
    return nullptr;
}

std::vector<std::pair<Transform *, Rigidbody *>>
PhysicsWorld::CollectCpuPoseReads(const std::vector<Transform *> &requested, bool worldSpace) const
{
    auto &transforms = TransformECSStore::Instance();
    // Inherited author/cache maintenance deliberately reads stored CPU values.
    // A new demand boundary must not promote those existing suppressed reads.
    if (!IsInitialized() || transforms.IsPhysicsPoseReadSuppressed())
        return {};
    std::vector<std::pair<Transform *, Rigidbody *>> physical;
    std::unordered_map<Transform *, uint8_t> visited;
    std::unordered_set<Rigidbody *> selectedBodies;
    auto &physics = PhysicsECSStore::Instance();
    const auto collect = [&](auto &&self, Transform *transform, bool world) -> void {
        if (!transform)
            return;
        const uint8_t bit = world ? 2u : 1u;
        if (visited[transform] & bit)
            return;
        visited[transform] |= bit;
        auto *object = transform->GetGameObject();
        if (!object || object->IsDestroying())
            return;
        // Local getters need parent affine data only for a stored world
        // override; ordinary stored local TRS must respect disabled/static
        // nearest-actor barriers instead of querying every higher ancestor.
        const auto handle = transform->GetECSHandle();
        if (world || (transforms.IsFrameCacheActiveFor(handle) && transforms.HasFrameCacheWorldPoseOverride(handle)))
            if (auto *parent = object->GetParent())
                self(self, parent->GetTransform(), true);
        auto actorHandle = physics.FindActor(object);
        while (object && !physics.IsValid(actorHandle)) {
            object = object->GetParent();
            actorHandle = physics.FindActor(object);
        }
        if (!physics.IsValid(actorHandle))
            return;
        const auto &actor = physics.GetActor(actorHandle);
        auto *body = actor.rigidbody;
        // This is the real lazy Transform dependency, including kinematic
        // ancestors/bones that normal getters also query. No world body scan.
        if (body && !object->IsDestroying() && body->IsEnabled() && !body->IsDestroyed() &&
            actor.bodyId != UINT32_MAX && !actor.sceneSuspended && actor.runtimeWorld &&
            actor.runtimeWorld->simulation.isPlaying && selectedBodies.insert(body).second) {
            // Applying this actual body's sampled world pose requires its
            // parent's affine transform, even for an eventual local consumer.
            if (auto *parent = object->GetParent())
                self(self, parent->GetTransform(), true);
            physical.push_back({object->GetTransform(), body});
        }
    };
    for (auto *transform : requested)
        collect(collect, transform, worldSpace);
    return physical;
}

PhysicsPresentationPoseReadScope::PhysicsPresentationPoseReadScope()
{
    auto &world = PhysicsWorld::Instance();
    world.RequireOwnerThread();
    m_domain = &world.CurrentDomain();
    if (m_domain->presentationPoseReadDepth++ == 0)
        m_domain->presentationPoseReadGeneration = m_domain->queryGeneration.load(std::memory_order_acquire);
}

PhysicsPresentationPoseReadScope::~PhysicsPresentationPoseReadScope()
{
    --m_domain->presentationPoseReadDepth;
}

bool PhysicsWorld::ServicePresentationPoseReads(const std::vector<Transform *> &world,
                                                const std::vector<Transform *> &local)
{
    RequireOwnerThread();
    auto &domain = CurrentDomain();
    RequireHealthyStep(domain);
    if (!m_initialized)
        return true;
    if (!domain.gpuRegistered || domain.gpuCancelled || domain.retiring || !m_gpuWork->IsActive() ||
        !m_gpuWork->IsAccepting() || m_gpuWork->ServiceRemaining() <= 0 ||
        !m_gpuWork->ResourceRemaining(FrameWorkResource::Cpu))
        return false;
    constexpr auto fields = static_cast<PhysicsStateFields>(static_cast<uint32_t>(PhysicsStateFields::Position) |
                                                            static_cast<uint32_t>(PhysicsStateFields::Rotation));
    // Pure Work never changes the published numerical owner, contacts or Fixed receipt.
    if (const auto job = domain.presentationPoseReads) {
        if (!AdvanceGpuJob(domain, false))
            return false;
        domain.presentationPoseReads.reset();
        if (job->error)
            std::rethrow_exception(job->error);
        if (job->publication != m_gpuScene->GetPublished(domain.owner) ||
            job->generation != domain.queryGeneration.load(std::memory_order_acquire))
            return false;
        MaterializeStateBatch(domain, job->batch, job->motion, job->shapes, true, fields);
    }
    if ((domain.stepPending && !domain.stepCompleted) || HasDomainControl(domain))
        return false;
    std::vector<Rigidbody *> missing;
    auto dependencies = CollectCpuPoseReads(world, true);
    auto localDependencies = CollectCpuPoseReads(local, false);
    dependencies.insert(dependencies.end(), localDependencies.begin(), localDependencies.end());
    std::unordered_set<Rigidbody *> selected;
    for (const auto &entry : dependencies) {
        if (&DomainForComponent(*entry.second) != &domain)
            throw std::logic_error("Presentation pose dependencies must belong to the selected World");
        if (selected.insert(entry.second).second && !FindCachedMotionState(domain, entry.second->GetBodyId(), fields))
            missing.push_back(entry.second);
    }
    if (missing.empty())
        return true;
    const auto publication = m_gpuScene->GetPublished(domain.owner);
    if (!publication)
        return false;
    auto job = std::make_shared<PhysicsDomain::PresentationPoseReads>();
    job->publication = publication;
    job->generation = domain.queryGeneration.load(std::memory_order_acquire);
    AdmissionScope admission(*m_gpuScene, domain.owner);
    FrameWorkOwner::Scope preparation(*m_gpuWork);
    auto work = PrepareStateBatch(domain, publication, missing, fields, false, false, {}, job->batch, job->motion,
                                  job->shapes, job->error);
    if (job->error)
        std::rethrow_exception(job->error);
    // The accepted Work retains every referenced result/batch even if Stop drops the domain slot.
    work = AfterWork(std::move(work), [job] { return JoltGpuControlWork{}; });
    if (!m_gpuScene->TryBeginWork(domain.owner, work, false))
        return false;
    domain.presentationPoseReads = std::move(job);
    return false;
}

void PhysicsWorld::PrepareCpuPoseReads(const std::vector<Transform *> &requested, bool worldSpace)
{
    const auto physical = CollectCpuPoseReads(requested, worldSpace);
    if (physical.empty())
        return;
    RequireOwnerThread();

    constexpr auto fields = static_cast<PhysicsStateFields>(static_cast<uint32_t>(PhysicsStateFields::Position) |
                                                            static_cast<uint32_t>(PhysicsStateFields::Rotation));
    std::vector<Rigidbody *> missing;
    missing.reserve(physical.size());
    for (const auto &entry : physical) {
        auto *body = entry.second;
        auto &domain = DomainForComponent(*body);
        RequireHealthyStep(domain);
        RequireGpu(domain);
        if (!FindCachedMotionState(domain, body->GetBodyId(), fields))
            missing.push_back(body);
    }
    if (!missing.empty()) {
        if (CurrentDomain().presentationPoseReadDepth)
            throw PhysicsPresentationPosePending{};
        (void)GatherRigidbodyStates(missing, fields, false, false);
    }

    // Batch queries fill the existing per-domain requestedMotion cache. Apply
    // it through the normal parent-first path, retaining all author/pending/
    // FreezeRotation and health guards. This is still outside the read scope:
    // a real intervening geometry control must be observed, never suppressed.
    for (const auto &entry : physical)
        EnsureCpuPhysicsPose(*entry.first);
}

PhysicsBodyMotionState PhysicsWorld::ReadMotionState(PhysicsDomain &domain, uint32_t id,
                                                     PhysicsStateFields fields) const
{
    RequireOwnerThread();
    RequireHealthyStep(domain);
    RequireGpu(domain);
    if (const auto *cached = FindCachedMotionState(domain, id, fields))
        return *cached;
    if (domain.presentationPoseReadDepth)
        throw PhysicsPresentationPosePending{};
    PhysicsBodyMotionState result;
    std::shared_ptr<JoltGpuQueryResult> motion;
    std::exception_ptr error;
    RunControl(
        domain,
        [&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
            return PrepareQueryWork(
                [&]() -> JoltGpuControlWork {
                    const auto allocation = m_gpuScene->FindBody(id);
                    if (allocation.bodyId == UINT32_MAX || allocation.slot >= publication->directory->actors.size() ||
                        !publication->directory->actors[allocation.slot] ||
                        publication->directory->actors[allocation.slot]->bodyId != id)
                        throw JoltGpuQueryRequestError("Physics state requires an accepted live body");
                    const auto &identity = *publication->directory->actors[allocation.slot];
                    result.bodyId = id;
                    result.shapeRevision = identity.shapeRevision;
                    if ((static_cast<uint32_t>(fields) & static_cast<uint32_t>(PhysicsStateFields::All)) == 0)
                        return {};
                    auto input = Query(QueryGatherMotion, QueryClosest, 0, false);
                    input.meta.z = 0;
                    input.source.y = allocation.slot;
                    input.source.z = static_cast<uint32_t>(fields);
                    input.geometry = identity.rootCenterOfMass;
                    auto prepared = m_gpuQueries->Prepare(publication, {input}, {},
                                                          domain.queryGeneration.load(std::memory_order_acquire) +
                                                              static_cast<uint64_t>(domain.controlNumericalChange));
                    motion = prepared.result;
                    return std::move(prepared.work);
                },
                error);
        },
        true);
    if (error)
        std::rethrow_exception(error);
    if (motion) {
        FrameWorkOwner::Scope materialize(*m_gpuWork);
        result = DecodeMotion(*motion->publication, motion->ReadMotion().at(0));
        domain.requestedMotion[id] = {motion->numericalGeneration, result, static_cast<PhysicsStateFields>(0x1ff)};
    }
    return result;
}
PhysicsBodyMotionState PhysicsWorld::GetBodyMotionState(uint32_t id, PhysicsStateFields fields) const
{
    auto &domain = DomainForBody(id);
    return ReadMotionState(domain, id, fields);
}
glm::vec3 PhysicsWorld::GetBodyPosition(uint32_t id) const
{
    auto &domain = DomainForBody(id);
    return ReadMotionState(domain, id, PhysicsStateFields::Position).position;
}
glm::quat PhysicsWorld::GetBodyRotation(uint32_t id) const
{
    auto &domain = DomainForBody(id);
    return ReadMotionState(domain, id, PhysicsStateFields::Rotation).rotation;
}
glm::vec3 PhysicsWorld::GetBodyCenterOfMassPosition(uint32_t id) const
{
    auto &domain = DomainForBody(id);
    return ReadMotionState(domain, id, PhysicsStateFields::CenterOfMass).centerOfMass;
}
glm::vec3 PhysicsWorld::GetBodyLinearVelocity(uint32_t id) const
{
    auto &domain = DomainForBody(id);
    return ReadMotionState(domain, id, PhysicsStateFields::LinearVelocity).linearVelocity;
}
glm::vec3 PhysicsWorld::GetBodyAngularVelocity(uint32_t id) const
{
    auto &domain = DomainForBody(id);
    return ReadMotionState(domain, id, PhysicsStateFields::AngularVelocity).angularVelocity;
}
glm::vec3 PhysicsWorld::GetBodyPrincipalInertia(uint32_t id) const
{
    (void)DomainForBody(id);
    const auto model = m_gpuScene->GetAcceptedBodyModel(id);
    const auto d = model.invInertiaDiagonal;
    if (!(d.x > 0 && d.y > 0 && d.z > 0))
        throw std::logic_error("Rigidbody has no accepted positive principal inertia");
    return glm::vec3(1.0f / d.x, 1.0f / d.y, 1.0f / d.z);
}

glm::quat PhysicsWorld::GetBodyPrincipalInertiaRotation(uint32_t id) const
{
    (void)DomainForBody(id);
    const auto q = m_gpuScene->GetAcceptedBodyModel(id).inertiaRotation;
    return glm::quat(q.w, q.x, q.y, q.z);
}

glm::mat3 PhysicsWorld::GetBodyWorldSpaceInertiaTensor(uint32_t id) const
{
    auto &domain = DomainForBody(id);
    return ReadMotionState(domain, id, PhysicsStateFields::InverseInertia).inertia;
}

bool PhysicsWorld::PrepareRaycastBatchQuery() const
{
    auto &domain = CurrentDomain();
    RequireOwnerThread();
    RunControl(domain, {});
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
    auto &domain = CurrentDomain();
    if (profile)
        *profile = {};
    if (count == 0) {
        if (generation)
            *generation = domain.queryGeneration.load(std::memory_order_acquire);
        return;
    }
    RequireOwnerThread();
    if (!origins || !directions || !hits || !mask || count > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Raycast batch requires valid arrays within the device index range");
    RequireGpu(domain);
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
    const auto result = RunQuery(domain, requests);
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
    auto &store = PhysicsECSStore::Instance();
    const auto handle = FindBodyActor(id);
    return store.IsValid(handle) ? store.GetActor(handle).rigidbody : nullptr;
}
Collider *PhysicsWorld::FindColliderByBodyId(uint32_t id) const
{
    if (!m_gpuScene || m_gpuScene->FindBody(id).bodyId == UINT32_MAX)
        return nullptr;
    auto &domain = DomainForBody(id);
    if (!m_gpuScene->GetPublished(domain.owner))
        return nullptr;
    for (const auto &actor : m_gpuScene->GetPublished(domain.owner)->directory->actors)
        if (actor && actor->bodyId == id)
            for (const auto &leaf : actor->leaves)
                if (auto *collider = leaf.surfaceTarget.GetCollider())
                    return collider;
    return nullptr;
}
Collider *PhysicsWorld::ResolveColliderForSubShape(uint32_t id, uint32_t subShape) const
{
    if (!m_gpuScene || m_gpuScene->FindBody(id).bodyId == UINT32_MAX)
        return nullptr;
    auto &domain = DomainForBody(id);
    if (!m_gpuScene->GetPublished(domain.owner))
        return nullptr;
    for (const auto &actor : m_gpuScene->GetPublished(domain.owner)->directory->actors)
        if (actor && actor->bodyId == id)
            for (const auto &leaf : actor->leaves) {
                const auto prefixMask = leaf.subShapeBits == 32 ? UINT32_MAX : (uint32_t(1) << leaf.subShapeBits) - 1;
                if ((leaf.subShapeId & prefixMask) == (subShape & prefixMask))
                    return leaf.surfaceTarget.GetCollider();
            }
    return nullptr;
}
void PhysicsWorld::EnsureSceneBodiesRegistered(Scene *scene)
{
    auto &domain = *scene->GetRuntimeWorld()->physics;
    RequireOwnerThread();
    FlushAuthoring(domain);
}
const std::vector<uint32_t> &PhysicsWorld::GetPoseReadbackBodyIds() const
{
    if (GetBodyCount())
        Unsupported("automatic bulk CPU pose readback");
    static const std::vector<uint32_t> empty;
    return empty;
}
void PhysicsWorld::SetContactEventStreamEnabled(bool enabled, bool includeTriggers)
{
    RequireOwnerThread();
    auto &domain = CurrentDomain();
    domain.contactEventStreamEnabled = enabled;
    domain.contactIncludeTriggers = includeTriggers;
}
void PhysicsWorld::SetContactImpulseStreamEnabled(bool enabled)
{
    RequireOwnerThread();
    CurrentDomain().contactImpulseStreamEnabled = enabled;
}
const std::vector<ContactEvent> &PhysicsWorld::GetContactEvents() const
{
    RequireOwnerThread();
    return CurrentDomain().contacts.Events();
}
const std::vector<ContactImpulse> &PhysicsWorld::GetContactImpulses() const
{
    RequireOwnerThread();
    return CurrentDomain().contacts.Impulses();
}
ContactStreamIdentity PhysicsWorld::GetContactStreamIdentity() const
{
    RequireOwnerThread();
    const auto &batch = CurrentDomain().contacts.batch;
    return batch ? batch->identity : ContactStreamIdentity{};
}
size_t PhysicsWorld::DispatchContactEvents()
{
    RequireOwnerThread();
    auto &domain = CurrentDomain();
    const auto &batch = domain.contacts.batch;
    const bool stale = batch && (batch->identity.worldId != domain.owner.id ||
                                 batch->identity.simulationEpoch != domain.owner.simulation.simulationEpoch);
    const auto delivered = domain.contacts.Dispatch(domain.owner);
    if (stale)
        domain.gpuContacts.Close();
    return delivered;
}
void PhysicsWorld::RefreshBodyTriggerState(uint32_t id)
{
    auto &domain = DomainForBody(id);
    RefreshGpuActor(domain, id);
}
void PhysicsWorld::InvalidateContactPairsForBody(uint32_t id)
{
    RequireOwnerThread();
    DomainForBody(id).contacts.Invalidate(id);
}
bool PhysicsWorld::IsBodySensor(uint32_t id) const
{
    auto &domain = DomainForBody(id);
    if (!m_initialized || id == UINT32_MAX)
        return false;
    RunControl(domain, {});
    const auto body = m_gpuScene->FindBody(id);
    if (body.bodyId == UINT32_MAX)
        return false;
    const auto &actor = m_gpuScene->GetPublished(domain.owner)->directory->actors[body.slot];
    return actor && !actor->leaves.empty() &&
           std::all_of(actor->leaves.begin(), actor->leaves.end(), [](const auto &leaf) { return leaf.sensor; });
}
std::shared_ptr<const PhysicsColliderPairs> PhysicsWorld::CaptureColliderPairs(PhysicsDomain &domain,
                                                                               bool userOnly) const
{
    if (!userOnly && domain.colliderPairSnapshot)
        return domain.colliderPairSnapshot;
    auto pairs = std::make_shared<PhysicsColliderPairs>();
    pairs->reserve(domain.colliderPairs.size());
    for (const auto &[key, value] : domain.colliderPairs)
        if (userOnly ? value.user : (value.user || !value.owners.empty()))
            pairs->push_back(key);
    if (!userOnly)
        domain.colliderPairSnapshot = pairs;
    return pairs;
}
void PhysicsWorld::QueueColliderPairs(PhysicsDomain &domain) const
{
    if (!domain.gpuRegistered || domain.gpuCancelled || domain.retiring)
        return;
    const auto pairs = CaptureColliderPairs(domain);
    if (domain.queuedColliderPairs == pairs)
        return;
    m_gpuScene->SetColliderPairs(domain.owner, pairs);
    domain.queuedColliderPairs = pairs;
}
void PhysicsWorld::SetColliderPairIgnored(Collider *a, Collider *b, bool ignored)
{
    RequireOwnerThread();
    if (!a || !b || a == b || a->IsDestroyed() || b->IsDestroyed() || a->IsBeingDestroyed() || b->IsBeingDestroyed())
        throw std::invalid_argument("Collision policy requires two distinct live Colliders");
    auto &domain = DomainForComponent(*a);
    if (&domain != &DomainForComponent(*b) || domain.retiring)
        throw std::invalid_argument("Collision policy endpoints must belong to one live RuntimeWorld");
    auto first = std::make_pair(a->GetComponentID(), a->GetLifetimeGeneration());
    auto second = std::make_pair(b->GetComponentID(), b->GetLifetimeGeneration());
    if (second < first)
        std::swap(first, second);
    const PhysicsColliderPair key{first.first, first.second, second.first, second.second};
    const auto found = domain.colliderPairs.find(key);
    if ((found != domain.colliderPairs.end() && found->second.user) == ignored)
        return;
    auto &policy = domain.colliderPairs[key];
    const bool wasIgnored = policy.user || !policy.owners.empty();
    policy.user = ignored;
    if (ignored) {
        auto &store = PhysicsECSStore::Instance();
        store.GetCollider(a->GetECSHandle()).collisionPolicyWorld = &domain.owner;
        store.GetCollider(b->GetECSHandle()).collisionPolicyWorld = &domain.owner;
    }
    const bool isIgnored = policy.user || !policy.owners.empty();
    if (!isIgnored)
        domain.colliderPairs.erase(key);
    if (wasIgnored != isIgnored)
        domain.colliderPairSnapshot.reset();
}
void PhysicsWorld::AcquireColliderPairOwner(Collider *a, Collider *b, const Component &owner)
{
    RequireOwnerThread();
    if (!a || !b || a == b || a->IsDestroyed() || b->IsDestroyed() || a->IsBeingDestroyed() || b->IsBeingDestroyed())
        throw std::invalid_argument("Collision policy requires two distinct live Colliders");
    auto &domain = DomainForComponent(*a);
    if (&domain != &DomainForComponent(*b) || domain.retiring)
        throw std::invalid_argument("Collision policy endpoints must belong to one live RuntimeWorld");
    if (owner.IsDestroyed() || owner.IsBeingDestroyed() || &DomainForComponent(owner) != &domain)
        throw std::invalid_argument("Collision policy owner must be live in the endpoint RuntimeWorld");
    auto first = std::make_pair(a->GetComponentID(), a->GetLifetimeGeneration());
    auto second = std::make_pair(b->GetComponentID(), b->GetLifetimeGeneration());
    if (second < first)
        std::swap(first, second);
    const PhysicsColliderPair key{first.first, first.second, second.first, second.second};
    const std::array<uint64_t, 2> token{owner.GetComponentID(), owner.GetLifetimeGeneration()};
    auto &policy = domain.colliderPairs[key];
    const bool wasIgnored = policy.user || !policy.owners.empty();
    if (!policy.owners.insert(token).second)
        return;
    auto &store = PhysicsECSStore::Instance();
    store.GetCollider(a->GetECSHandle()).collisionPolicyWorld = &domain.owner;
    store.GetCollider(b->GetECSHandle()).collisionPolicyWorld = &domain.owner;
    if (!wasIgnored)
        domain.colliderPairSnapshot.reset();
}
void PhysicsWorld::ReleaseColliderPairOwner(PhysicsDomain &domain, const std::array<uint64_t, 2> &ownerIdentity)
{
    RequireOwnerThread();
    // The assembly owner retains this domain and lifetime token from acquisition.
    // Teardown need not retain a Component, Scene link, or either endpoint.
    bool changed = false;
    for (auto it = domain.colliderPairs.begin(); it != domain.colliderPairs.end();) {
        auto &policy = it->second;
        if (!policy.owners.erase(ownerIdentity)) {
            ++it;
            continue;
        }
        if (!policy.user && policy.owners.empty()) {
            it = domain.colliderPairs.erase(it);
            changed = true;
        } else
            ++it;
    }
    if (changed)
        domain.colliderPairSnapshot.reset();
}
bool PhysicsWorld::GetColliderPairIgnored(const Collider *a, const Collider *b) const
{
    RequireOwnerThread();
    if (!a || !b || a == b || a->IsDestroyed() || b->IsDestroyed() || a->IsBeingDestroyed() || b->IsBeingDestroyed())
        throw std::invalid_argument("Collision policy requires two distinct live Colliders");
    auto &domain = DomainForComponent(*a);
    if (&domain != &DomainForComponent(*b))
        throw std::invalid_argument("Collision policy endpoints must belong to one RuntimeWorld");
    auto first = std::make_pair(a->GetComponentID(), a->GetLifetimeGeneration());
    auto second = std::make_pair(b->GetComponentID(), b->GetLifetimeGeneration());
    if (second < first)
        std::swap(first, second);
    const auto found = domain.colliderPairs.find({first.first, first.second, second.first, second.second});
    return found != domain.colliderPairs.end() && (found->second.user || !found->second.owners.empty());
}
void PhysicsWorld::RemoveIgnoredPairsForCollider(const Collider *collider)
{
    RequireOwnerThread();
    if (!collider)
        return;
    auto &store = PhysicsECSStore::Instance();
    const auto handle = collider->GetECSHandle();
    if (!store.IsValid(handle))
        return;
    auto &data = store.GetCollider(handle);
    auto *world = data.collisionPolicyWorld;
    if (!world)
        return; // Unregistered staging never acquired a policy domain.
    auto &domain = *world->physics;
    data.collisionPolicyWorld = nullptr;
    bool changed = false;
    const auto id = collider->GetComponentID(), generation = collider->GetLifetimeGeneration();
    for (auto it = domain.colliderPairs.begin(); it != domain.colliderPairs.end();) {
        const auto &k = it->first;
        if ((k[0] == id && k[1] == generation) || (k[2] == id && k[3] == generation)) {
            it = domain.colliderPairs.erase(it);
            changed = true;
        } else
            ++it;
    }
    if (changed)
        domain.colliderPairSnapshot.reset();
}
void PhysicsWorld::RestoreColliderPairs(RuntimeWorld &world, const PhysicsColliderPairs &source,
                                        const std::unordered_map<uint64_t, uint64_t> &remap)
{
    RequireOwnerThread();
    auto &domain = *world.physics;
    for (const auto &pair : source) {
        const auto a = remap.find(pair[0]), b = remap.find(pair[2]);
        if (a == remap.end() || b == remap.end())
            continue;
        auto *first = dynamic_cast<Collider *>(Component::FindByComponentId(a->second));
        auto *second = dynamic_cast<Collider *>(Component::FindByComponentId(b->second));
        if (!first || !second || &DomainForComponent(*first) != &domain || &DomainForComponent(*second) != &domain)
            throw std::logic_error("Cloned collision policy endpoint differs from its preparing World");
        SetColliderPairIgnored(first, second, true);
    }
}

void PhysicsWorld::MoveBodyKinematic(uint32_t id, const glm::vec3 &position, const glm::quat &rotation, float dt,
                                     float maxSpeed)
{
    auto &domain = DomainForBody(id);
    RequireGpu(domain);
    (void)KinematicBody(ActorForBody(id));
    m_gpuScene->QueueCommand(id, MoveCommand(CommandMovePose, position, rotation, dt, maxSpeed));
}
void PhysicsWorld::MoveBodyKinematicPosition(uint32_t id, const glm::vec3 &position, float dt)
{
    auto &domain = DomainForBody(id);
    RequireGpu(domain);
    (void)KinematicBody(ActorForBody(id));
    m_gpuScene->QueueCommand(id, MoveCommand(CommandMovePosition, position, glm::quat(1, 0, 0, 0), dt));
}
void PhysicsWorld::MoveBodyKinematicRotation(uint32_t id, const glm::quat &rotation, float dt)
{
    auto &domain = DomainForBody(id);
    RequireGpu(domain);
    (void)KinematicBody(ActorForBody(id));
    m_gpuScene->QueueCommand(id, MoveCommand(CommandMoveRotation, glm::vec3(0), rotation, dt));
}
PhysicsTargetSource PhysicsWorld::MoveBodyKinematicAuthored(uint32_t id, const glm::vec3 &position,
                                                            const glm::quat &rotation, uint32_t channels, float dt,
                                                            float maxSpeed,
                                                            const std::array<uint64_t, 2> &authoredRevision)
{
    auto &domain = DomainForBody(id);
    RequireGpu(domain);
    auto &body = KinematicBody(ActorForBody(id));
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
    auto &domain = DomainForBody(id);
    RequireGpu(domain);
    const auto &actor = ActorForBody(id);
    if (actor.rigidbody && actor.rigidbody->IsEnabled())
        throw std::logic_error("Static movement requires a collider-only physics actor");
    auto command = MoveCommand(CommandMovePose, position, rotation, dt, kMaxTransformDriveSpeed);
    command.meta.z = CommandTemporaryStatic;
    m_gpuScene->QueueCommand(id, command);
}
bool PhysicsWorld::IsBodySleeping(uint32_t id) const
{
    auto &domain = DomainForBody(id);
    bool sleeping = false;
    RunControl(domain, [&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        const auto allocation = m_gpuScene->FindBody(id);
        if (allocation.bodyId == UINT32_MAX || allocation.slot < publication->bodyBase ||
            allocation.slot - publication->bodyBase >= publication->bodyCount ||
            allocation.slot >= publication->directory->actors.size() ||
            !publication->directory->actors[allocation.slot] ||
            publication->directory->actors[allocation.slot]->bodyId != id)
            throw std::logic_error("Sleep state requires this publication's live body identity");
        return m_gpuScene->PrepareSleepQuery(publication, {allocation.slot - publication->bodyBase}, [&](auto values) {
            if (values.size() != 1u || values[0].x > 1u || values[0].z != 1u || values[0].w)
                throw std::runtime_error("GPU sleep publication failed its initialized activity contract");
            sleeping = values[0].x == 0u;
        });
    });
    return sleeping;
}
void PhysicsWorld::ActivateBody(uint32_t id)
{
    auto &domain = DomainForBody(id);
    RequireGpu(domain);
    BodyCommand command{};
    command.meta.y = CommandWake;
    m_gpuScene->QueueCommand(id, command);
}
void PhysicsWorld::DeactivateBody(uint32_t id)
{
    auto &domain = DomainForBody(id);
    RequireGpu(domain);
    BodyCommand command{};
    command.meta.y = CommandSleep;
    m_gpuScene->QueueCommand(id, command);
}
void PhysicsWorld::ActivateBodiesInAABB(const glm::vec3 &low, const glm::vec3 &high)
{
    RequireOwnerThread();
    auto &domain = CurrentDomain();
    RequireGpu(domain);
    for (uint32_t axis = 0; axis < 3; ++axis)
        if (!std::isfinite(low[axis]) || !std::isfinite(high[axis]) || low[axis] > high[axis])
            throw std::invalid_argument("Wake bounds require finite ordered world coordinates");
    m_gpuScene->QueueBoundsWake(domain.owner, {low.x, low.y, low.z, 0}, {high.x, high.y, high.z, 0});
}
void PhysicsWorld::WakeBodiesTouchingStatic(uint32_t id)
{
    auto &domain = DomainForBody(id);
    RequireGpu(domain);
    const auto model = m_gpuScene->GetAcceptedBodyModel(id);
    if ((model.meta.z & ~MotionNotResident) != MotionStatic)
        throw std::invalid_argument("Static-neighbour wake requires a static source body");
    m_gpuScene->QueueStaticWake(id);
}

std::optional<PhysicsPenetrationResult> PhysicsWorld::ComputePenetration(const Collider &a, const glm::vec3 &positionA,
                                                                         const glm::quat &rotationA, const Collider &b,
                                                                         const glm::vec3 &positionB,
                                                                         const glm::quat &rotationB) const
{
    auto &domain = CurrentDomain();
    RequireOwnerThread();
    auto first = CookColliderQueryPrimitive(a, positionA, rotationA);
    auto second = CookColliderQueryPrimitive(b, positionB, rotationB);
    if (!first || !second)
        return std::nullopt;
    // The mesh surface policy belongs to the target operand. Preserve it when
    // the caller supplies the triangle first, then restore the caller's order.
    const bool reverseOperands = (first->primitive.meta.y & ShapeOneSidedTriangle) != 0u;
    if (reverseOperands)
        std::swap(first, second);
    RequireGpu(domain);
    JoltGpuQueryGeometry geometry;
    geometry.store = m_gpuScene->GetGeometryStore();
    for (auto *operand : {&*first, &*second}) {
        geometry.shapes.push_back(operand->shape);
        if (operand->geometry) {
            auto handle = geometry.store->Intern(operand->geometry);
            operand->primitive.geometryReference = handle->reference;
            geometry.operands.push_back(std::move(handle));
        }
    }
    auto request = Query(QueryPenetration, QueryClosest);
    request.meta.z = first->primitive.meta.x;
    request.positionMaximum = first->primitive.position;
    request.rotation = first->primitive.rotation;
    request.geometry = first->primitive.geometry;
    if (first->primitive.meta.y & ShapeOneSidedTriangle)
        request.geometry = {request.geometry.x, 0, 0, 0}; // Request support has no target-side surface policy.
    request.geometryReference = first->primitive.geometryReference;
    auto result = RunGeometryQuery(domain, request, second->primitive, std::move(geometry));
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    const auto &records = result->ReadHits();
    if (records.empty() || !records.front().meta.y)
        return std::nullopt;
    const auto &hit = records.front();
    if (reverseOperands)
        return PhysicsPenetrationResult{-Vector(hit.normalFraction), hit.pointADepth.w, Vector(hit.positionDistance),
                                        Vector(hit.pointADepth)};
    return PhysicsPenetrationResult{Vector(hit.normalFraction), hit.pointADepth.w, Vector(hit.pointADepth),
                                    Vector(hit.positionDistance)};
}
bool PhysicsWorld::RaycastCollider(const Collider &collider, const glm::vec3 &origin, const glm::vec3 &direction,
                                   float maximum, RaycastHit &hit) const
{
    auto &domain = DomainForComponent(collider);
    auto request = Query(QueryRay, QueryClosest);
    QueryRayInput(request, origin, direction, maximum);
    TargetCollider(request, collider);
    auto result = RunQuery(domain, {request});
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    const auto &record = result->ReadHits().front();
    hit = DecodeQueryHit(*result, record);
    return record.meta.y != 0;
}
glm::vec3 PhysicsWorld::ClosestPointOnCollider(const Collider &collider, const glm::vec3 &point) const
{
    auto &domain = DomainForComponent(collider);
    (void)Vector(point);
    if (!collider.IsEnabled() || !collider.GetGameObject() || !collider.GetGameObject()->IsActiveInHierarchy())
        return point;
    auto request = Query(QueryClosestPoint, QueryClosest);
    request.positionMaximum = Vector(point);
    TargetCollider(request, collider);
    auto result = RunQuery(domain, {request});
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    const auto &record = result->ReadHits().front();
    if (!record.meta.y)
        throw std::runtime_error("Active Collider has no accepted geometry for a closest-point query");
    return Vector(record.positionDistance);
}
std::vector<RaycastHit> PhysicsWorld::RaycastAll(const glm::vec3 &origin, const glm::vec3 &direction, float maximum,
                                                 uint32_t layers, bool triggers) const
{
    auto &domain = CurrentDomain();
    auto request = Query(QueryRay, QueryAll, layers, triggers);
    QueryRayInput(request, origin, direction, maximum);
    auto result = RunQuery(domain, {request});
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
    auto &domain = CurrentDomain();
    auto request = Query(QueryOverlap, QueryUniqueCollider, layers, triggers);
    QuerySphereInput(request, center, radius);
    return QueryColliders(domain, request);
}
std::vector<Collider *> PhysicsWorld::OverlapBox(const glm::vec3 &center, const glm::vec3 &halfExtents,
                                                 const glm::quat &rotation, uint32_t layers, bool triggers) const
{
    auto &domain = CurrentDomain();
    auto request = Query(QueryOverlap, QueryUniqueCollider, layers, triggers);
    QueryBoxInput(request, center, halfExtents, rotation);
    return QueryColliders(domain, request);
}
std::vector<Collider *> PhysicsWorld::OverlapCapsule(const glm::vec3 &a, const glm::vec3 &b, float radius,
                                                     uint32_t layers, bool triggers) const
{
    auto &domain = CurrentDomain();
    auto request = Query(QueryOverlap, QueryUniqueCollider, layers, triggers);
    QueryCapsuleInput(request, a, b, radius);
    return QueryColliders(domain, request);
}
std::vector<Rigidbody *> PhysicsWorld::QueryRigidbodiesInBounds(const glm::vec3 &minimum, const glm::vec3 &maximum,
                                                                uint32_t layers, bool triggers) const
{
    auto &domain = CurrentDomain();
    (void)Vector(minimum);
    (void)Vector(maximum);
    if (glm::any(glm::greaterThan(minimum, maximum)))
        throw std::invalid_argument("Physics query bounds require minimum <= maximum");
    auto request = Query(QueryBodyBounds, QueryUniqueBody, layers, triggers);
    request.positionMaximum = Vector(glm::vec3((glm::dvec3(minimum) + glm::dvec3(maximum)) * 0.5));
    request.direction = Vector(glm::vec3((glm::dvec3(maximum) - glm::dvec3(minimum)) * 0.5));
    auto result = RunQuery(domain, {request});
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    auto &store = PhysicsECSStore::Instance();
    std::vector<Rigidbody *> bodies;
    for (const auto &record : result->ReadHits()) {
        const auto &identity = *result->publication->directory->actors[record.meta.x];
        if (store.IsValid(identity.rigidbody))
            bodies.push_back(store.GetRigidbody(identity.rigidbody).owner);
    }
    return bodies;
}
bool PhysicsWorld::SphereCast(const glm::vec3 &center, float radius, const glm::vec3 &direction, float maximum,
                              RaycastHit &hit, uint32_t layers, bool triggers) const
{
    auto &domain = CurrentDomain();
    auto request = Query(QueryShapeCast, QueryClosest, layers, triggers);
    QuerySphereInput(request, center, radius);
    QueryRayInput(request, center, direction, maximum);
    auto result = RunQuery(domain, {request});
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    const auto &record = result->ReadHits().front();
    hit = DecodeQueryHit(*result, record);
    return record.meta.y != 0;
}
bool PhysicsWorld::BoxCast(const glm::vec3 &center, const glm::vec3 &halfExtents, const glm::vec3 &direction,
                           const glm::quat &rotation, float maximum, RaycastHit &hit, uint32_t layers,
                           bool triggers) const
{
    auto &domain = CurrentDomain();
    auto request = Query(QueryShapeCast, QueryClosest, layers, triggers);
    QueryBoxInput(request, center, halfExtents, rotation);
    QueryRayInput(request, center, direction, maximum);
    auto result = RunQuery(domain, {request});
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    const auto &record = result->ReadHits().front();
    hit = DecodeQueryHit(*result, record);
    return record.meta.y != 0;
}
bool PhysicsWorld::SweepBox(const BoxCollider &box, const glm::vec3 &position, const glm::quat &rotation,
                            const glm::vec3 &displacement, RaycastHit &hit) const
{
    auto &domain = DomainForComponent(box);
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
    request.meta.z = primitive->primitive.meta.x;
    request.meta.w = QuerySweepPolicy;
    request.geometry = primitive->primitive.geometry;
    if (primitive->primitive.meta.y & ShapeOneSidedTriangle)
        request.geometry = {request.geometry.x, 0, 0, 0};
    request.rotation = primitive->primitive.rotation;
    QueryRayInput(request, Vector(primitive->primitive.position), displacement, static_cast<float>(distance));
    request.source.z = static_cast<uint32_t>(objectId);
    request.source.w = static_cast<uint32_t>(objectId >> 32);
    request.filter.z = static_cast<uint32_t>(colliderId);
    request.filter.w = static_cast<uint32_t>(colliderId >> 32);
    auto result = RunQuery(domain, {request}, colliderId, box.GetLifetimeGeneration());
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    const auto &record = result->ReadHits().front();
    hit = DecodeQueryHit(*result, record);
    return record.meta.y != 0;
}
bool PhysicsWorld::CapsuleCast(const glm::vec3 &a, const glm::vec3 &b, float radius, const glm::vec3 &direction,
                               float maximum, RaycastHit &hit, uint32_t layers, bool triggers) const
{
    auto &domain = CurrentDomain();
    auto request = Query(QueryShapeCast, QueryClosest, layers, triggers);
    QueryCapsuleInput(request, a, b, radius);
    QueryRayInput(request, Vector(request.positionMaximum), direction, maximum);
    auto result = RunQuery(domain, {request});
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    const auto &record = result->ReadHits().front();
    hit = DecodeQueryHit(*result, record);
    return record.meta.y != 0;
}
std::shared_ptr<const PhysicsShapeGeometry> PhysicsWorld::ResolveShapeGeometry(PhysicsDomain &domain,
                                                                               const JoltGpuScene::PublishedLeaf &leaf,
                                                                               const QueryShapeState &record) const
{
    const auto *shape = leaf.shape.GetPtr();
    if (!shape)
        throw std::runtime_error("Published shape state has no retained immutable geometry");
    if (const auto found = domain.shapeGeometry.find(shape); found != domain.shapeGeometry.end())
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
    } else if (record.surface.y == ShapeCylinder || record.surface.y == ShapeConvexHull) {
        // The existing immutable convex mesh view exposes the cooked native surface.
        // Dynamic collision/query calculations remain exclusively on the GPU.
        geometry->type = PhysicsShapeType::Mesh;
    } else {
        throw std::runtime_error("GPU shape state has no immutable geometry adapter");
    }
    domain.shapeGeometry[shape] = geometry;
    return geometry;
}

std::shared_ptr<JoltGpuQueryResult> PhysicsWorld::GatherBodyShapes(PhysicsDomain &domain, uint32_t id,
                                                                   bool triggers) const
{
    JoltGpuPreparedQuery query;
    std::exception_ptr error;
    RunControl(domain, [&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        return PrepareQueryWork(
            [&] {
                const auto allocation = m_gpuScene->FindBody(id);
                if (allocation.bodyId == UINT32_MAX || allocation.slot >= publication->directory->actors.size() ||
                    !publication->directory->actors[allocation.slot] ||
                    publication->directory->actors[allocation.slot]->bodyId != id)
                    throw JoltGpuQueryRequestError("Shape gathering requires an accepted body identity");
                auto input = Query(QueryGatherShapes, QueryAll, UINT32_MAX, triggers);
                input.meta.z = QueryAllShapeTypes;
                input.source.y = allocation.slot;
                query = m_gpuQueries->Prepare(publication, {input}, {},
                                              domain.queryGeneration.load(std::memory_order_acquire) +
                                                  static_cast<uint64_t>(domain.controlNumericalChange));
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
    auto &domain = DomainForBody(id);
    auto query = GatherBodyShapes(domain, id, triggers);
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    std::vector<PhysicsShapeState> result;
    const auto &records = query->ReadShapes();
    result.reserve(records.size());
    for (const auto &record : records) {
        const auto &address = query->publication->directory->leaves.at(record.meta.z);
        const auto &identity = *query->publication->directory->actors.at(record.meta.x);
        if (address.bodySlot != record.meta.x)
            throw std::runtime_error("GPU shape state has a different published leaf owner");
        const auto &leaf = identity.leaves.at(address.actorLeaf);
        PhysicsShapeState state;
        state.collider = leaf.surfaceTarget.GetCollider();
        state.memberIndex = leaf.memberIndex;
        state.layerBit = record.surface.x;
        state.trigger = (record.meta.w & 1u) != 0;
        state.moving = (record.meta.w & 2u) != 0;
        state.geometry = ResolveShapeGeometry(domain, leaf, record);
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
    auto &domain = *scene.GetRuntimeWorld()->physics;
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
    RunControl(domain, [&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        return PrepareQueryWork(
            [&]() -> JoltGpuControlWork {
                std::vector<QueryInput> requests;
                std::array<std::vector<Row>, 2> rows;
                uint32_t sourceRow = 0;
                for (uint32_t slot = 0; slot < publication->directory->actors.size(); ++slot) {
                    const auto &actor = publication->directory->actors[slot];
                    auto *owner = actor ? actor->owner.target.GetGameObject() : nullptr;
                    if (!owner || owner->GetScene() != &scene || !publication->directory->residentBodies.at(slot))
                        continue;
                    auto input = Query(QueryGatherShapes, QueryAll);
                    input.meta.z = QueryAllShapeTypes;
                    input.source.y = slot;
                    requests.push_back(input);
                    for (const auto &leaf : actor->leaves) {
                        Row row;
                        row.key = {leaf.colliderId, leaf.colliderGeneration, leaf.subShapeId, leaf.subShapeBits,
                                   reinterpret_cast<uintptr_t>(leaf.shape.GetPtr())};
                        row.mapping.selection = {sourceRow++, InvalidQueryBodySlot,
                                                 static_cast<uint32_t>(leaf.colliderGeneration),
                                                 static_cast<uint32_t>(leaf.colliderGeneration >> 32)};
                        const auto bounds = leaf.shape->GetLocalBounds();
                        row.mapping.localBoundsMin = {bounds.mMin.GetX(), bounds.mMin.GetY(), bounds.mMin.GetZ(), 0};
                        row.mapping.localBoundsMax = {bounds.mMax.GetX(), bounds.mMax.GetY(), bounds.mMax.GetZ(), 0};
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
                const bool hasHistory = previous && previous->worldId == publication->worldId &&
                                        previous->simulationEpoch == publication->simulationEpoch;
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
                                              domain.queryGeneration.load(std::memory_order_acquire) +
                                                  static_cast<uint64_t>(domain.controlNumericalChange),
                                              JoltGpuQueryReadback::Device);
                return AfterWork(
                    std::move(query.work),
                    [this, &domain, source = query.result, mappings = std::move(mappings), previous, hasHistory,
                     staticCount, revision, topology, &exported]() -> JoltGpuControlWork {
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

namespace infernux
{
bool PhysicsWorld::HasPendingControl() const noexcept
{
    return HasDomainControl(CurrentDomain());
}
void PhysicsWorld::CompletePendingStep()
{
    CompleteDomainStep(CurrentDomain());
}
} // namespace infernux

namespace infernux
{
void PhysicsWorld::RunPreparedWork(PhysicsDomain &domain, JoltGpuControlWork prepared) const
{
    RequireGpu(domain);
    AdmissionScope admission(*m_gpuScene, domain.owner);
    while (!m_gpuScene->TryBeginWork(domain.owner, prepared))
        DrainGpuSource(domain);
    if (!AdvanceGpuJob(domain, true))
        throw std::runtime_error("GPU result consumer did not complete");
}
} // namespace infernux

namespace infernux
{
PhysicsDomain &PhysicsWorld::CurrentDomain() const
{
    return *SceneManager::Instance().CurrentWorld().physics;
}
PhysicsDomain &PhysicsWorld::DomainForActor(const PhysicsActorData &actor) const
{
    if (!actor.runtimeWorld)
        throw std::logic_error("Physics actor has no published RuntimeWorld owner");
    return *actor.runtimeWorld->physics;
}
void PhysicsWorld::RegisterBodyIdentity(PhysicsDomain &domain, uint32_t bodyId, PhysicsActorHandle actor,
                                        Backend backend)
{
    // Stop retains the handles for recreation and cannot allocate. Reserve
    // for all registered bodies, including any earlier withdrawals still queued.
    const size_t required = domain.recreateActors.size() + domain.registeredBodyCount + 1;
    if (required > domain.recreateActors.capacity())
        domain.recreateActors.reserve(std::max(required, 2 * domain.recreateActors.capacity()));
    if (!m_bodyActors.emplace(bodyId, BodyRegistration{actor, backend}).second)
        throw std::logic_error("Duplicate physics body identity");
    ++domain.registeredBodyCount;
}
void PhysicsWorld::UnregisterBodyIdentity(PhysicsDomain &domain, uint32_t bodyId) noexcept
{
    domain.registeredBodyCount -= m_bodyActors.erase(bodyId);
}
const PhysicsWorld::BodyRegistration *PhysicsWorld::FindBodyRegistration(uint32_t bodyId) const noexcept
{
    const auto entry = m_bodyActors.find(bodyId);
    if (entry == m_bodyActors.end())
        return nullptr;
    const auto &store = PhysicsECSStore::Instance();
    const auto handle = entry->second.actor;
    if (!store.IsValid(handle) || store.GetActor(handle).bodyId != bodyId)
        return nullptr;
    return &entry->second;
}
PhysicsActorHandle PhysicsWorld::FindBodyActor(uint32_t bodyId) const noexcept
{
    const auto *registration = FindBodyRegistration(bodyId);
    return registration ? registration->actor : PhysicsActorHandle{};
}
PhysicsActorData &PhysicsWorld::ActorForBody(uint32_t bodyId) const
{
    auto &store = PhysicsECSStore::Instance();
    const auto handle = FindBodyActor(bodyId);
    if (!store.IsValid(handle))
        throw std::invalid_argument("Physics body identity is invalid or retired");
    return store.GetActor(handle);
}
PhysicsDomain &PhysicsWorld::DomainForBody(uint32_t bodyId) const
{
    return DomainForActor(ActorForBody(bodyId));
}
PhysicsDomain &PhysicsWorld::DomainForComponent(const Component &component) const
{
    auto *object = component.GetGameObject();
    const auto &store = PhysicsECSStore::Instance();
    const auto actor = store.FindActor(object);
    if (store.IsValid(actor))
        return DomainForActor(store.GetActor(actor));
    if (!object || !object->GetScene() || !object->GetScene()->GetRuntimeWorld())
        throw std::logic_error("Physics component has no published RuntimeWorld owner");
    return *object->GetScene()->GetRuntimeWorld()->physics;
}
uint64_t PhysicsWorld::GetQueryGeneration() const noexcept
{
    return CurrentDomain().queryGeneration.load(std::memory_order_acquire);
}
bool PhysicsWorld::IsContactEventStreamEnabled() const noexcept
{
    return CurrentDomain().contactEventStreamEnabled;
}
bool PhysicsWorld::IsContactImpulseStreamEnabled() const noexcept
{
    return CurrentDomain().contactImpulseStreamEnabled;
}
void PhysicsWorld::BeginRetireDomain(RuntimeWorld &world) noexcept
{
    const WorldExecutionScope ownerScope(&world);
    auto &domain = *world.physics;
    // Components still run OnDisable/OnDestroy after the numerical owner closes.
    // Retire their accepted body/target identities through the same cancellation
    // boundary as Stop, while the shared directory can still resolve them.
    CancelPendingStep();
    domain.recreateActors.clear();
    domain.retiring = true;
    for (const auto &entry : domain.participants)
        if (const auto participant = entry.lock())
            participant->Retire();
    domain.participants.clear();
    domain.participantPreparing.reset();
    RetireRagdolls(domain);
    if (domain.gpuRegistered)
        m_gpuScene->RetirePartition(world);
    domain.rendering.bindings.Clear();
    // Accepted continuations may still own pending jobs; WorldManager retains world.
}
bool PhysicsWorld::CanRetireDomain(const RuntimeWorld &world) const noexcept
{
    return !world.physics->gpuRegistered || m_gpuScene->CanRetire(world);
}
void PhysicsWorld::RetireDomain(PhysicsDomain &domain) noexcept
{
    RetireRagdolls(domain);
    m_domains.erase(&domain);
}
} // namespace infernux
