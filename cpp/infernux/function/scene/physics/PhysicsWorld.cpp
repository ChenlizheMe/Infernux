#include "PhysicsWorld.h"
#include <Jolt/Jolt.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <Jolt/RegisterTypes.h>
#include "../BoxCollider.h"
#include "../Collider.h"
#include "../Component.h"
#include "../DistanceJoint.h"
#include "../FixedJoint.h"
#include "../GameObject.h"
#include "../HingeJoint.h"
#include "../Joint.h"
#include "../Ragdoll.h"
#include "../Rigidbody.h"
#include "../RuntimeWorld.h"
#include "../Scene.h"
#include "../SceneManager.h"
#include "../SliderJoint.h"
#include "../SwingTwistJoint.h"
#include "../TagLayerManager.h"
#include "../Transform.h"
#include "../TransformECSStore.h"
#include "JoltCpuControl.h"
#include "JoltCpuQueries.h"
#include "JoltGpuContactReports.h"
#include "JoltGpuJoints.h"
#include "JoltGpuProgramSources.h"
#include "JoltGpuQueries.h"
#include "JoltGpuRenderBindings.h"
#include "JoltGpuRenderPoses.h"
#include "PhysicsBodyCommand.h"
#include "PhysicsContactTypes.h"
#include "PhysicsDomain.h"
#include "PhysicsLayers.h"
#include "PhysicsMass.h"
#include "PhysicsParticipantBodies.h"
#include "PhysicsShapeAuthoring.h"
#include "warp/WarpSceneRuntime.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <core/config/EngineConfig.h>
#include <core/log/InxLog.h>
#include <core/threading/FrameWorkOwner.h>
#include <exception>
#include <function/renderer/particle/ParticleGpuCollisionScene.h>
#include <function/renderer/rhi/RhiComputeHost.h>
#include <function/renderer/rhi/RhiComputeKernel.h>
#include <function/resources/PhysicMaterial/PhysicMaterial.h>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <variant>

namespace infernux
{
namespace
{
using namespace JPH::GPU;
JPH::Vec3 CpuVector(const Float4 &v)
{
    return JPH::Vec3(v.x, v.y, v.z);
}
JPH::RVec3 CpuPosition(const Float4 &v)
{
    return JPH::RVec3(v.x, v.y, v.z);
}
JPH::Quat CpuRotation(const Float4 &q)
{
    return JPH::Quat(q.x, q.y, q.z, q.w);
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
constexpr std::array<uint32_t, 13> kStateExportWidths{3, 4, 3, 3, 3, 3, 9, 1, 3, 3, 4, 1, 1};

std::array<float, 9> StateExportFloats(JoltGpuQueryExportField field, const PhysicsBodyMotionState &state,
                                       const PhysicsBoxShapeState &shape)
{
    using Field = JoltGpuQueryExportField;
    const auto vector = [](glm::vec3 v) { return std::array<float, 9>{v.x, v.y, v.z}; };
    const auto quaternion = [](glm::quat q) { return std::array<float, 9>{q.x, q.y, q.z, q.w}; };
    switch (field) {
    case Field::MotionPosition:
        return vector(state.position);
    case Field::MotionRotation:
        return quaternion(state.rotation);
    case Field::MotionCenterOfMass:
        return vector(state.centerOfMass);
    case Field::MotionLinearVelocity:
        return vector(state.linearVelocity);
    case Field::MotionAngularVelocity:
        return vector(state.angularVelocity);
    case Field::MotionInverseMass:
        return vector(state.inverseMass);
    case Field::MotionInverseInertia: {
        const auto &m = state.inverseInertia;
        return {m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1], m[0][2], m[1][2], m[2][2]};
    }
    case Field::ShapeCenter:
        return vector(shape.center);
    case Field::ShapeHalfExtents:
        return vector(shape.halfExtents);
    case Field::ShapeRotation:
        return quaternion(shape.rotation);
    case Field::ShapeFriction:
        return {shape.friction};
    case Field::ShapeBounciness:
        return {shape.bounciness};
    case Field::ShapeBodyIndex:
        break; // This field is packed as an integer.
    }
    throw JoltGpuQueryRequestError("Physics aggregate export requires a floating point field");
}

std::shared_ptr<rhi::PreparedComputeBatch>
PrepareCpuStateExports(const PhysicsRigidbodyStateBatch &batch, const std::vector<uint8_t> &cpuRows,
                       const std::vector<std::vector<PhysicsRigidbodyBoxState>> &boxes,
                       const std::vector<uint64_t> &shapeFirst, const std::vector<JoltGpuQueryExportBinding> &bindings,
                       bool includeBoxes)
{
    using Field = JoltGpuQueryExportField;
    if (!bindings.front().buffer)
        throw JoltGpuQueryRequestError("Physics aggregate export requires a compute destination");
    auto &host = bindings.front().buffer->GetHost();
    std::vector<uint64_t> targetSizes;
    targetSizes.reserve(bindings.size());
    std::vector<rhi::ComputeBufferUpdate> updates;
    for (const auto &binding : bindings) {
        const auto field = static_cast<uint32_t>(binding.field);
        const auto &buffer = binding.buffer;
        if (field >= kStateExportWidths.size() || !buffer || buffer->IsFrozen() ||
            !host.SharesServicesWith(buffer->GetHost()) || binding.byteOffset % sizeof(uint32_t))
            throw JoltGpuQueryRequestError(
                "Physics aggregate export field or destination differs from its compute owner");
        const bool motion = binding.field <= Field::MotionInverseInertia;
        if (!motion && !includeBoxes)
            throw JoltGpuQueryRequestError("Physics aggregate box export requires Box rows");
        const auto &desc = buffer->GetDesc();
        const auto scalar =
            binding.field == Field::ShapeBodyIndex ? rhi::ComputeScalarType::Int32 : rhi::ComputeScalarType::Float32;
        const auto width = kStateExportWidths[field];
        const auto lanes = width == 9 ? 1u : width;
        const uint64_t count = motion ? batch.states.size() : batch.boxCount;
        const uint64_t bytes = count * width * sizeof(uint32_t);
        const auto offset = binding.byteOffset;
        const auto end = std::min(binding.byteEnd, buffer->GetByteSize());
        if (desc.scalarType != scalar || desc.lanes != lanes || offset > end || bytes > end - offset ||
            offset / sizeof(uint32_t) > UINT32_MAX || count * width > UINT32_MAX - offset / sizeof(uint32_t))
            throw JoltGpuQueryRequestError("Physics aggregate export destination has invalid type or prefix capacity");
        for (size_t i = 0; i < targetSizes.size(); ++i) {
            const auto &previous = bindings[i];
            if ((previous.buffer->GetWriterStateIdentity() == buffer->GetWriterStateIdentity() ||
                 previous.buffer->GetAllocation() == buffer->GetAllocation()) &&
                bytes && targetSizes[i] && offset < previous.byteOffset + targetSizes[i] &&
                previous.byteOffset < offset + bytes)
                throw JoltGpuQueryRequestError("Physics aggregate export destination prefixes must not overlap");
        }
        targetSizes.push_back(bytes);

        // Consecutive CPU rows share an upload. GPU rows remain owned by the
        // existing device projection and leave gaps in these byte ranges.
        const size_t firstUpdate = updates.size();
        const size_t rowBytes = width * sizeof(uint32_t);
        const auto append = [&](uint64_t row, const void *value) {
            const auto destination = offset + row * rowBytes;
            if (updates.size() == firstUpdate || updates.back().offset + updates.back().data.size() != destination)
                updates.push_back({buffer, destination, {}});
            const auto *data = static_cast<const uint8_t *>(value);
            auto &bytes = updates.back().data;
            bytes.insert(bytes.end(), data, data + rowBytes);
        };
        for (size_t row = 0; row < cpuRows.size(); ++row) {
            if (!cpuRows[row])
                continue;
            if (motion) {
                const auto values = StateExportFloats(binding.field, batch.states[row], {});
                append(row, values.data());
            } else {
                for (size_t box = 0; box < boxes[row].size(); ++box) {
                    const auto &entry = boxes[row][box];
                    if (binding.field == Field::ShapeBodyIndex)
                        append(shapeFirst[row] + box, &entry.bodyIndex);
                    else {
                        const auto values = StateExportFloats(binding.field, {}, entry.shape);
                        append(shapeFirst[row] + box, values.data());
                    }
                }
            }
        }
    }
    if (updates.empty())
        return {};
    return rhi::PrepareComputeBatch(host, std::move(updates), {});
}

uint32_t ColliderBodyIdentity(const Collider &collider)
{
    const auto &store = PhysicsECSStore::Instance();
    const auto handle = collider.GetECSHandle();
    if (!store.IsValid(handle))
        return UINT32_MAX;
    const auto actor = store.GetCollider(handle).actorHandle;
    return store.IsValid(actor) ? store.GetActor(actor).bodyId : UINT32_MAX;
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
ShaderCodeTarget PhysicsShaderTarget(const rhi::ComputeHost &host)
{
    switch (host.device.GetShaderSourceLanguage()) {
    case rhi::ShaderSourceLanguage::SpirV:
        return ShaderCodeTarget::SpirV;
    case rhi::ShaderSourceLanguage::Wgsl:
        return ShaderCodeTarget::Wgsl;
    default:
        throw std::runtime_error("Unsupported physics shader target");
    }
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
    StepParams p{};
    p.counts.w = static_cast<uint32_t>(c.physicsVelocitySteps);
    p.solve.x = static_cast<uint32_t>(c.physicsPositionSteps);
    p.solve.y = static_cast<uint32_t>(c.physicsCollisionSteps);
    p.gravityDt = {gravity.x, gravity.y, gravity.z, dt};
    p.contact = {c.physicsSpeculativeContactDistance, c.physicsPenetrationSlop, c.physicsMaxPenetrationDistance,
                 c.physicsBaumgarte};
    p.restitution.x = c.physicsMinVelocityForRestitution;
    p.restitution.y = c.physicsLinearCastThreshold;
    p.restitution.z = c.physicsLinearCastMaxPenetration;
    return JoltGpuStepParameters(p);
}
} // namespace

struct PhysicsStaticBoundaryModel
{
    using Candidate = std::variant<PreparedCpuActor, PreparedGpuActor>;

    explicit PhysicsStaticBoundaryModel(const Transform &transform)
    {
        TransformECSStore::PhysicsPoseReadScope authoring(TransformECSStore::Instance());
        position = transform.GetPosition();
        rotation = transform.GetWorldRotation();
    }

    template <class Actor> void Hold(Actor &&actor)
    {
        pending = std::make_unique<Candidate>(std::move(actor));
    }
    void ApplyPose(PreparedCpuActor &actor) const
    {
        actor.settings.mPosition = CpuPosition(Vector(position));
        actor.settings.mRotation = CpuRotation(Rotation(rotation));
    }
    void ApplyPose(PreparedGpuActor &actor) const
    {
        actor.SetOriginPose(position, rotation);
    }

    glm::vec3 position;
    glm::quat rotation;
    uint64_t positionRevision = 0, rotationRevision = 0;
    // Only a boundary that has not reached its scene backend owns a prepared model here.
    std::unique_ptr<Candidate> pending;
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

void PhysicsDomain::MarkGpuPoseDiscontinuity(uint32_t bodyId)
{
    rendering.bindings.MarkDiscontinuity(bodyId);
}
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
constexpr auto PresentationPoseFields = static_cast<PhysicsStateFields>(
    static_cast<uint32_t>(PhysicsStateFields::Position) | static_cast<uint32_t>(PhysicsStateFields::Rotation));

float CpuPresentationAlpha(const PhysicsDomain &domain)
{
    const auto &simulation = domain.owner.simulation;
    return !simulation.isPaused && domain.admittedDeltaTime > 0.0f
               ? std::clamp(simulation.fixedTimeAccumulator / domain.admittedDeltaTime, 0.0f, 1.0f)
               : 1.0f;
}
} // namespace

struct PhysicsCpuRaycastEpoch
{
    std::shared_ptr<JoltCpuScene> scene;
    std::shared_lock<std::shared_mutex> read;
    uint64_t generation;

    PhysicsCpuRaycastEpoch(std::shared_ptr<JoltCpuScene> accepted, const std::atomic<uint64_t> &publishedGeneration)
        : scene(std::move(accepted)),
          read(scene ? std::shared_lock(scene->QueryMutex()) : std::shared_lock<std::shared_mutex>{}),
          generation(publishedGeneration.load(std::memory_order_acquire))
    {
    }
};

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
void PhysicsWorld::ConfigureSceneBackend(RuntimeWorld &world, PhysicsBackend backend)
{
    auto &domain = *world.physics;
    if (domain.backend == backend)
        return;
    const WorldExecutionScope scope(&world);
    if (domain.cpuScene || domain.gpuRegistered || !domain.bodyIds.empty())
        CancelPendingStep();
    domain.backend = backend;
    domain.startupPrograms.reset();
    if (m_initialized && !domain.recreateActors.empty())
        RegisterDomain(domain);
}
bool PhysicsDomain::UsesGpu() const noexcept
{
    return backend == PhysicsBackend::GPU;
}
void PhysicsWorld::RegisterDomain(PhysicsDomain &domain)
{
    m_domains.insert(&domain);
    auto &store = PhysicsECSStore::Instance();
    for (const auto handle : domain.recreateActors) {
        if (!store.IsValid(handle))
            continue;
        const auto &actor = store.GetActor(handle);
        if (!actor.sceneSuspended && actor.bodyId == UINT32_MAX)
            store.QueueBodyCreation(handle);
    }
    domain.recreateActors.clear();
}
void PhysicsWorld::EnsureCpuReady(PhysicsDomain &domain)
{
    RequireOwnerThread();
    RequireHealthyStep(domain);
    if (!m_initialized || domain.retiring || domain.UsesGpu())
        throw std::logic_error("CPU physics requires a live CPU physics scene");
    if (!m_cpuServices)
        m_cpuServices = std::make_unique<CpuServices>(EngineConfig::Get());
    if (!domain.cpuScene) {
        auto scene = std::make_shared<JoltCpuScene>(EngineConfig::Get(), m_cpuServices->layers,
                                                    m_cpuServices->broadphaseFilter, m_cpuServices->pairFilter);
        auto control = std::make_unique<JoltCpuControl>(*scene);
        scene->SetIgnoredPairs(CaptureColliderPairs(domain));
        domain.cpuScene = std::move(scene);
        domain.cpuControl = std::move(control);
    }
    RegisterDomain(domain);
}
JoltGpuScene &PhysicsWorld::EnsureSharedGpuScene()
{
    if (!m_initialized)
        throw std::logic_error("PhysicsWorld is not initialized");
    if (!m_gpuHost || !m_gpuWork)
        throw std::runtime_error("GPU physics requires engine compute services");
    m_gpuHost->RequireAlive();
    if (!m_gpuScene) {
        auto &host = *m_gpuHost;
        const auto target = PhysicsShaderTarget(host);
        const auto load = [target](JoltGpuProgram program) { return LoadJoltGpuProgram(program, target); };
        auto queries = std::make_unique<JoltGpuQueries>(host, load(JoltGpuProgram::Queries));
        auto joints = std::make_unique<JoltGpuJoints>(host, load(JoltGpuProgram::Joints));
        auto rendering = std::make_unique<JoltGpuRenderPoses>(host, load(JoltGpuProgram::RenderPoses));
        auto reports = std::make_unique<JoltGpuContactReports>(host, load(JoltGpuProgram::ContactReports));
        const auto articulationCapture = load(JoltGpuProgram::ArticulationCapture);
        const auto articulationDynamics = load(JoltGpuProgram::ArticulationDynamics);
        auto scene = std::make_unique<JoltGpuScene>(
            host, *m_gpuWork, load(JoltGpuProgram::Step), load(JoltGpuProgram::BodyMutations),
            load(JoltGpuProgram::BodyCommands), load(JoltGpuProgram::Broadphase), load(JoltGpuProgram::Narrowphase),
            load(JoltGpuProgram::Constraints), *joints,
            [this](RuntimeWorld &world) noexcept {
                // A shared source can fail outside Step and outside this World's
                // execution scope. Mark only the actual retained writer.
                FailStep(*world.physics);
                world.simulation.fixedPhase = RuntimeWorld::FixedPhase::Failed;
                world.simulation.pausedStepPending = false;
                world.simulation.isPaused = true;
            },
            &articulationCapture, &articulationDynamics);
        m_gpuJoints = std::move(joints);
        m_gpuQueries = std::move(queries);
        m_gpuRenderProgram = std::move(rendering);
        m_gpuContactReports = std::move(reports);
        m_gpuScene = std::move(scene);
    }
    return *m_gpuScene;
}
bool PhysicsWorld::EnsureGpuReady(PhysicsDomain &domain)
{
    RequireOwnerThread();
    RequireHealthyStep(domain);
    if (!domain.UsesGpu())
        throw std::logic_error("GPU physics requires a GPU physics scene");
    if (domain.retiring)
        return false;
    auto &scene = EnsureSharedGpuScene();
    if (!scene.RegisterPartition(domain.owner, InitialBodyCapacity))
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
        throw std::runtime_error("The GPU physics partition is still retiring accepted receipts");
}
void PhysicsWorld::RetireIdleGpuPartition(PhysicsDomain &domain)
{
    if (!domain.gpuRegistered || !domain.bodyIds.empty() || domain.controlPending || domain.waitingForSlot ||
        domain.participantOutputPending || !domain.participantPreparing.expired())
        return;
    if (!domain.particleCollisionConsumer.expired())
        return;
    // A participant can read its last completed state while it is disabled.
    for (const auto &entry : domain.participants)
        if (const auto owner = entry.lock(); owner && !owner->IsRetired())
            return;

    RetireGpuPartition(domain);
}
void PhysicsWorld::RetireGpuPartition(PhysicsDomain &domain)
{
    if (!std::exchange(domain.gpuRegistered, false))
        return;
    m_gpuScene->RetirePartition(domain.owner);
    domain.lastPublication.reset();
    domain.gpuContacts.Close();
    domain.queuedColliderPairs.reset();
    domain.presentationPoseReads.reset();
    domain.requestedMotion.clear();
    domain.queryGeneration.fetch_add(1, std::memory_order_release);
    auto &rendering = domain.rendering;
    rendering.bindings.Clear();
    rendering.pending.reset();
    rendering.ready.reset();
    rendering.checkedKeys.reset();
}
void PhysicsWorld::ServiceGpuRetirements()
{
    if (m_gpuScene)
        m_gpuScene->ServiceRetirements();
}
void PhysicsWorld::Shutdown()
{
    for (auto *domain : m_domains) {
        if (domain->gpuRegistered)
            m_gpuScene->Cancel(domain->owner);
        domain->rendering.bindings.Clear();
        domain->rendering.pending.reset();
        domain->rendering.ready.reset();
        domain->presentationPoseReads.reset();
        domain->contacts.Close();
        domain->gpuContacts.Close();
        domain->cpuContacts.Close();
        RetireRagdolls(*domain);
        RetireCpuJointRoutes(*domain);
        domain->cpuFixedQueryWrite.reset();
        domain->cpuControl.reset();
        domain->cpuScene.reset();
        domain->cpuPublication = {};
        domain->cpuFixedAdmitted = false;
        domain->lastPublication.reset();
        domain->queuedColliderPairs.reset();
        domain->requestedMotion.clear();
    }
    m_gpuScene.reset(); // Whole-engine teardown after all logical worlds stop.
    m_bodyActors.clear();
    for (auto *domain : m_domains) {
        domain->gpuRegistered = false;
        domain->gpuBodyOwners = 0;
        domain->bodyIds.clear();
        domain->pendingStatic.clear();
        domain->pendingStaticMoves.clear();
        domain->admittedStaticMoves.clear();
        domain->recreateActors.clear();
        domain->shapeGeometry.clear();
        domain->rendering.checkedKeys.reset();
    }
    m_gpuContactReports.reset();
    m_gpuRenderProgram.reset();
    m_gpuQueries.reset();
    m_gpuJoints.reset();
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
    return domain.bodyIds.size();
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
PhysicsContactPublication::Policy PhysicsWorld::CaptureContactPolicy(const PhysicsDomain &domain) const
{
    PhysicsContactPublication::Policy policy{
        domain.contactEventStreamEnabled, domain.contactIncludeTriggers, domain.contactImpulseStreamEnabled, {}};
    const auto &store = PhysicsECSStore::Instance();
    for (const auto id : domain.bodyIds)
        if (const auto *entry = FindBodyRegistration(id))
            policy.CaptureActor(domain.owner, store.GetActor(entry->actor));
    return policy;
}

bool PhysicsWorld::TryBeginFixed(PhysicsDomain &domain)
{
    if (!EnsureStaticBoundaries(domain))
        return false;
    QueueStaticFixedMoves(domain);
    auto policy = CaptureContactPolicy(domain);
    if (domain.gpuRegistered) {
        QueueColliderPairs(domain);
        AdmissionScope admission(*m_gpuScene, domain.owner);
        const bool simulateBodies = domain.gpuBodyOwners || !domain.pendingStaticMoves.empty();
        if (!m_gpuScene->TryBeginTick(
                domain.owner, domain.stepEpoch, domain.stepIndex, domain.admittedDeltaTime, *domain.admittedParameters,
                [this, &domain] { return PrepareRenderConsumer(domain, true); },
                [this, &domain, policy, simulateBodies](const auto &publication) {
                    auto observer = simulateBodies ? domain.gpuContacts.Prepare(*m_gpuContactReports, publication,
                                                                                domain.admittedSubsteps, policy)
                                                   : JoltGpuScene::ContactObserver{};
                    observer.simulateRigidBodies = simulateBodies;
                    return observer;
                },
                [&domain] { return domain.admittedParticipants; }))
            return false;
    }
    domain.admittedContactPolicy = std::move(policy);
    SealStaticFixedMoves(domain);
    domain.cpuFixedAdmitted = bool(domain.cpuControl);
    if (domain.cpuFixedAdmitted) {
        domain.cpuFixedQueryWrite.emplace(domain.cpuScene->QueryMutex());
        domain.cpuControl->SealFixed();
        domain.cpuControl->ApplyFixed([this](uint32_t, const auto &ids) { RetireCpuJointRoutes(ids); });
        auto &system = domain.cpuScene->System();
        system.SetGravity(domain.admittedCpuGravity);
        system.SetPhysicsSettings(domain.admittedCpuSettings);
        domain.cpuScene->SetIgnoredPairs(CaptureColliderPairs(domain));
        const ContactStreamIdentity identity{domain.owner.id, domain.stepEpoch, domain.stepIndex};
        const auto error = domain.cpuContacts.Step(m_cpuServices->execution, *domain.cpuScene, domain.admittedDeltaTime,
                                                   domain.admittedSubsteps, identity, domain.admittedContactPolicy, {});
        if (error != JPH::EPhysicsUpdateError::None)
            throw std::runtime_error("CPU physics exceeded its configured native capacity");
    }
    return true;
}

void PhysicsWorld::FinishFixed(PhysicsDomain &domain)
{
    const ContactStreamIdentity identity{domain.owner.id, domain.stepEpoch, domain.stepIndex};
    const auto substeps = domain.admittedSubsteps;
    if (domain.cpuFixedAdmitted)
        domain.cpuContacts.ValidateCompleted(identity, substeps);
    std::shared_ptr<const JoltGpuScene::Publication> gpu;
    if (domain.gpuRegistered) {
        if (!m_gpuScene->ConsumeCompletedTick(domain.owner))
            throw std::logic_error("Completed Fixed did not publish its GPU state");
        gpu = m_gpuScene->GetPublished(domain.owner);
        domain.gpuContacts.ValidateCompleted(gpu, substeps);
    }
    AcceptStaticFixedMoves(domain);
    if (domain.admittedContactPolicy.Requested()) {
        domain.contacts.Publish(identity, domain.admittedContactPolicy, substeps, [&](uint32_t substep) {
            auto frame =
                domain.cpuFixedAdmitted ? domain.cpuContacts.TakeFrame(substep) : PhysicsContactPublication::Frame{};
            if (gpu)
                domain.gpuContacts.AppendFrame(substep, frame);
            return frame;
        });
    } else {
        domain.contacts.PublishEmpty(identity);
    }
    if (domain.cpuFixedAdmitted) {
        domain.cpuContacts.AcceptCompleted();
        domain.cpuControl->PublishApplied();
        domain.cpuPublication = identity;
        CaptureCpuPoses(domain, true);
    }
    if (gpu) {
        domain.gpuContacts.AcceptCompleted(gpu);
        ObserveGpuPublication(domain, true);
    } else {
        domain.queryGeneration.fetch_add(1, std::memory_order_release);
        domain.requestedMotion.clear();
    }
    domain.stepCompleted = true;
    domain.cpuFixedQueryWrite.reset();
}
void PhysicsWorld::ObserveGpuPublication(PhysicsDomain &domain, bool numericalChange) const
{
    const WorldExecutionScope ownerScope(&domain.owner);
    const auto &p = m_gpuScene->GetPublished(domain.owner);
    if (p && p.get() != domain.lastPublication.get()) {
        domain.lastPublication = p;
        if (numericalChange) {
            domain.queryGeneration.fetch_add(1, std::memory_order_release);
            domain.requestedMotion.clear();
        }
        if (m_gpuRenderProgram) {
            FrameWorkOwner::Scope publication(*m_gpuWork);
            auto &rendering = domain.rendering;
            const auto &job = rendering.pending;
            if (job && job->output && job->publication == p) {
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
void PhysicsWorld::CaptureCpuPoses(PhysicsDomain &domain, bool fixed) const
{
    for (const auto &update : domain.cpuScene->TakePoseUpdates()) {
        const auto *entry = FindBodyRegistration(update.pose.bodyId);
        if (!entry || entry->backend != Backend::Cpu)
            continue;
        const PhysicsDomain::CpuPose pose{update.pose, update.sleeping, fixed};
        auto &index = entry->cpuPoseIndex;
        if (index < domain.cpuPoses.size() && domain.cpuPoses[index].pose.bodyId == update.pose.bodyId) {
            domain.cpuPoses[index] = pose;
        } else {
            index = domain.cpuPoses.size();
            domain.cpuPoses.push_back(pose);
        }
    }
}

Rigidbody *PhysicsWorld::FindCpuPoseBody(const PhysicsDomain &domain, uint32_t id) const
{
    const auto *entry = FindBodyRegistration(id);
    if (!entry || entry->backend != Backend::Cpu)
        return nullptr;
    const auto &actor = PhysicsECSStore::Instance().GetActor(entry->actor);
    const auto *owner = actor.owner;
    auto *body = actor.rigidbody;
    if (actor.runtimeWorld != &domain.owner || actor.sceneSuspended || !owner || owner->IsDestroying() ||
        !owner->IsActiveInHierarchy() || !body || !body->IsEnabled() || body->IsBeingDestroyed() ||
        body->IsDestroyed() || body->GetECSHandle() != entry->rigidbody)
        return nullptr;
    return body;
}

void PhysicsWorld::RebaseCpuPoseDescendants(const GameObject &parent) const
{
    const auto &domain = CurrentDomain();
    if (!domain.cpuScene)
        return;
    auto &store = PhysicsECSStore::Instance();
    const auto rebase = [&](auto &&self, const GameObject &object) -> void {
        const auto handle = store.FindActor(&object);
        if (store.IsValid(handle))
            if (auto *body = FindCpuPoseBody(domain, store.GetActor(handle).bodyId))
                body->RebaseCpuPhysicsPose();
        for (const auto &child : object.GetChildren())
            self(self, *child);
    };
    for (const auto &child : parent.GetChildren())
        rebase(rebase, *child);
}

void PhysicsWorld::InvalidateBodyInterpolation(uint32_t bodyId) const
{
    DomainForBody(bodyId).publishedCpuInterpolationAlpha.reset();
}

PhysicsPosePublication PhysicsWorld::PublishCpuPoses(PhysicsDomain &domain, float alpha) const
{
    const bool advanceInterpolation = domain.publishedCpuInterpolationAlpha != alpha;
    if (domain.cpuPoses.empty() && (domain.cpuInterpolated.empty() || !advanceInterpolation))
        return {};

    auto &store = PhysicsECSStore::Instance();
    TransformECSStore::PhysicsPoseReadScope publishing(TransformECSStore::Instance());
    std::vector<std::vector<GameObject *>> byDepth;
    const auto collect = [&](const Rigidbody &body) {
        auto *object = body.GetGameObject();
        size_t depth = 0;
        for (auto *parent = object->GetParent(); parent; parent = parent->GetParent())
            ++depth;
        if (byDepth.size() <= depth)
            byDepth.resize(depth + 1);
        byDepth[depth].push_back(object);
    };
    const auto pending = [&](uint32_t id) -> const PhysicsDomain::CpuPose * {
        const auto index = m_bodyActors.at(id).cpuPoseIndex;
        return index < domain.cpuPoses.size() && domain.cpuPoses[index].pose.bodyId == id ? &domain.cpuPoses[index]
                                                                                          : nullptr;
    };
    for (const auto &update : domain.cpuPoses)
        if (const auto *body = FindCpuPoseBody(domain, update.pose.bodyId))
            collect(*body);
    if (advanceInterpolation) {
        for (auto it = domain.cpuInterpolated.begin(); it != domain.cpuInterpolated.end();) {
            if (const auto *body = FindCpuPoseBody(domain, *it)) {
                if (!pending(*it))
                    collect(*body);
                ++it;
            } else {
                it = domain.cpuInterpolated.erase(it);
            }
        }
    }
    PhysicsPosePublication publication;
    std::unordered_set<GameObject *> visited;
    visited.reserve(domain.cpuPoses.size() + domain.cpuInterpolated.size());
    const auto apply = [&](auto &&self, GameObject *object, bool ancestorMoved) -> void {
        if (!visited.insert(object).second)
            return;
        bool moved = false;
        const auto handle = store.FindActor(object);
        if (store.IsValid(handle)) {
            const auto id = store.GetActor(handle).bodyId;
            if (auto *body = FindCpuPoseBody(domain, id)) {
                const bool interpolate =
                    body->GetInterpolation() == static_cast<int>(RigidbodyInterpolation::Interpolate);
                const auto *sample = pending(id);
                if (sample) {
                    ++publication.syncedBodies;
                    const auto &[pose, sleeping, fixed] = *sample;
                    PhysicsBodyMotionState state;
                    state.bodyId = id;
                    state.position = pose.position;
                    state.rotation = pose.rotation;
                    moved = body->SyncPhysicsToTransform(state, sleeping, !fixed);
                    if (fixed && !sleeping && interpolate)
                        domain.cpuInterpolated.insert(id);
                    else
                        domain.cpuInterpolated.erase(id);
                }
                if ((sample || advanceInterpolation) && domain.cpuInterpolated.count(id)) {
                    ++publication.interpolatedBodies;
                    moved |= body->ApplyInterpolatedTransform(alpha);
                    if (!interpolate)
                        domain.cpuInterpolated.erase(id);
                }
                if (ancestorMoved)
                    moved |= body->RebaseCpuPhysicsPose();
            }
        }
        publication.changed |= moved;
        if (ancestorMoved || moved)
            for (const auto &child : object->GetChildren())
                self(self, child.get(), true);
    };
    for (const auto &level : byDepth)
        for (auto *object : level)
            apply(apply, object, false);
    domain.cpuPoses.clear();
    domain.publishedCpuInterpolationAlpha = alpha;
    return publication;
}

PhysicsPosePublication PhysicsWorld::PublishRenderPoses() const
{
    auto &domain = CurrentDomain();
    RequireOwnerThread();
    if (!m_initialized || domain.retiring || domain.fixedGpuPhase == PhysicsDomain::FixedGpuPhase::Failed)
        return {};
    try {
        auto &rendering = domain.rendering;
        if (!domain.UsesGpu())
            return PublishCpuPoses(domain, CpuPresentationAlpha(domain));
        if (!domain.gpuRegistered || !m_gpuRenderProgram)
            return {};
        const auto &job = rendering.ready;
        if (!job)
            return {};
        FrameWorkOwner::Scope publication(*m_gpuWork);
        const bool gpuPublished = rendering.bindings.Publish(*job->output, job->numericalGeneration);
        if (gpuPublished)
            rendering.checkedKeys = job->keys;
        else
            rendering.checkedKeys.reset();
        rendering.ready.reset();
        return {gpuPublished};
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
        job->output = state.bindings.PrepareExport(
            job->snapshot,
            [&](PhysicsActorHandle handle, uint32_t bodyId) -> std::optional<PhysicsGpuBodyPoseBinding> {
                const auto slot = publication->directory->FindBodySlot(bodyId);
                if (slot < publication->bodyBase || slot - publication->bodyBase >= publication->bodyCount)
                    return std::nullopt;
                const auto &actor = publication->directory->actors[slot];
                if (!actor || actor->actor != handle)
                    return std::nullopt;
                const auto &center = actor->rootCenterOfMass;
                return PhysicsGpuBodyPoseBinding{bodyId, slot - publication->bodyBase, {center.x, center.y, center.z}};
            },
            {publication->bodyStates, 0, publication->bodyCount}, *m_gpuRenderProgram);
        job->publication = publication;
        return ContinueJoltGpuControlWork(job->output->Packet(), [job] {
            job->output->Seal();
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
    domain.cpuContacts.CancelPending();
    domain.fixedGpuPhase = PhysicsDomain::FixedGpuPhase::Failed;
    domain.stepCompleted = false;
    if (domain.gpuRegistered)
        m_gpuScene->SetAdmissionEnabled(domain.owner, false);
}
void PhysicsWorld::RefreshBodySurfaces(PhysicsDomain &domain)
{
    const auto revision = PhysicMaterial::GetSurfaceRevision();
    if (revision == domain.surfaceRevision)
        return;
    auto &store = PhysicsECSStore::Instance();
    TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    for (const auto id : domain.bodyIds) {
        auto &actor = store.GetActor(m_bodyActors.at(id).actor);
        StageActorModel(actor, actor);
    }
    domain.surfaceRevision = revision;
}
JoltGpuParticipantStages PhysicsWorld::CaptureParticipants(PhysicsDomain &domain)
{
    std::shared_ptr<std::vector<uint32_t>> slots;
    auto stages = CapturePhysicsParticipants(domain.participants, [&](auto bodyIds) {
        if (!slots)
            slots = std::make_shared<std::vector<uint32_t>>();
        return BindPhysicsParticipantBodies(domain, EnsureSharedGpuScene(), *m_gpuWork, std::move(bodyIds), slots);
    });
    stages.gpuBodySlots = std::move(slots);
    return stages;
}

std::shared_ptr<PhysicsParticipant>
PhysicsWorld::RegisterParticipant(const Component &component, PhysicsParticipant::Recipes recipes,
                                  PhysicsParticipant::Controls controls, uint32_t count,
                                  const std::vector<Rigidbody *> &rigidbodies,
                                  const PhysicsParticipant::ExchangeStates &exchangeStates)
{
    RequireOwnerThread();
    auto &domain = DomainForComponent(component);
    RequireHealthyStep(domain);
    if (!domain.UsesGpu())
        throw std::invalid_argument("GPU physics participants require a GPU physics scene");
    if (domain.retiring || recipes.empty() || !m_gpuHost ||
        !recipes.begin()->second->GetHost().SharesServicesWith(*m_gpuHost))
        throw std::invalid_argument("Participant requires its live World's engine compute host");
    auto participant = std::make_shared<PhysicsParticipant>(component, std::move(recipes), std::move(controls), count,
                                                            rigidbodies, exchangeStates);
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
bool PhysicsWorld::TryPrepareStartupPrograms()
{
    RequireOwnerThread();
    auto &domain = CurrentDomain();
    if (!domain.UsesGpu())
        return true;
    if (!domain.startupPrograms) {
        PhysicsDomain::StartupPrograms programs;
        auto &store = PhysicsECSStore::Instance();
        size_t localColliderOwners = 0;
        bool gpuCollider = false;
        const auto visitObject = [&](auto &&visit, const GameObject &object) -> void {
            if (!object.IsActive())
                return;
            const auto *body = object.GetComponent<Rigidbody>();
            programs.gpuBodies |= body && body->IsEnabled();
            const auto actor = store.FindActor(&object);
            if (store.IsValid(actor)) {
                const auto &shapes = store.GetActor(actor).colliders;
                if (std::any_of(shapes.begin(), shapes.end(), [](const auto *shape) { return shape->IsEnabled(); })) {
                    ++localColliderOwners;
                    gpuCollider |= body && body->IsEnabled();
                }
            }
            for (const auto &child : object.GetChildren())
                visit(visit, *child);
        };
        const auto visitScene = [&](const std::unique_ptr<Scene> &scene) {
            if (scene)
                for (const auto &root : scene->GetRootObjects())
                    visitObject(visitObject, *root);
        };
        const auto &scenes = domain.owner.scenes;
        for (const auto &scene : scenes.loaded)
            visitScene(scene);
        visitScene(scenes.persistent);
        programs.gpuContacts = gpuCollider && localColliderOwners > 1;
        domain.startupPrograms = programs;
    }
    const auto &programs = *domain.startupPrograms;
    if (!programs.gpuBodies)
        return true;
    return EnsureSharedGpuScene().TryPrepareStartupPrograms(programs.gpuContacts);
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
        FinishFixed(domain);
    }
    if (domain.controlPending) {
        if (status != JoltGpuTick::Status::Complete)
            return false;
        if (!m_gpuScene->ConsumeCompletedControl(domain.owner))
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
    if (!publication || m_gpuScene->IsWriterFor(domain.owner))
        return false;
    const auto epoch = publication->simulationEpoch, step = publication->fixedStepIndex;
    const bool fixed = publication->isFixedStep;
    JoltGpuControlWork copies;
    for (auto it = observers.rbegin(); it != observers.rend(); ++it) {
        if (!(*it)->PublishResidentOutputs(epoch, step, fixed))
            continue;
        auto tail = std::move(copies);
        copies = {nullptr,
                  [owner = *it, epoch, step, fixed, tail = std::move(tail)]() mutable {
                      return ContinueJoltGpuControlWork(owner->PrepareOutputs(epoch, step, fixed),
                                                        [tail = std::move(tail)]() mutable { return std::move(tail); });
                  },
                  JoltGpuControlWork::ResumeAfter::HostWork};
    }
    if (!copies.continueWork)
        return true;
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
            if (!ServiceParticipantOutputs())
                return false;
            auto participants = domain.UsesGpu() ? CaptureParticipants(domain) : JoltGpuParticipantStages{};
            if ((domain.gpuRegistered || participants) && !EnsureGpuReady(domain))
                return false;
            RefreshBodySurfaces(domain);
            domain.stepEpoch = epoch;
            domain.stepIndex = index;
            domain.admittedDeltaTime = dt;
            const auto &config = EngineConfig::Get();
            if (config.physicsVelocitySteps < 0 || config.physicsPositionSteps < 0 || config.physicsCollisionSteps <= 0)
                throw std::invalid_argument("Physics iterations and substeps are invalid");
            if (domain.UsesGpu()) {
                domain.admittedParameters = StepParameters(dt, domain.gravity);
            } else {
                domain.admittedCpuGravity = JPH::Vec3(domain.gravity.x, domain.gravity.y, domain.gravity.z);
                domain.admittedCpuSettings = JoltCpuScene::CaptureSettings(config);
            }
            domain.admittedSubsteps = static_cast<uint32_t>(config.physicsCollisionSteps);
            domain.admittedParticipants = std::move(participants);
            // Own this fixed identity before waiting for an earlier visual
            // control. Scripts and accounting cannot be replayed meanwhile.
            domain.stepPending = true;
            domain.solverStarted = false;
            if (domain.controlPending || domain.waitingForSlot)
                domain.fixedGpuPhase = PhysicsDomain::FixedGpuPhase::RetireControl;
            else
                domain.fixedGpuPhase = PhysicsDomain::FixedGpuPhase::Solver;
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
    using Phase = PhysicsDomain::FixedGpuPhase;
    RequireHealthyStep(domain);
    try {
        for (;;) {
            if (domain.fixedGpuPhase == Phase::RetryCreation) {
                RetryCapacityBodyCreations(domain);
                RefreshBodySurfaces(domain);
                domain.fixedGpuPhase = domain.waitingForSlot ? Phase::RetireControl : Phase::Solver;
            }
            if (domain.fixedGpuPhase == Phase::RetireControl) {
                if (!domain.controlPending && !BeginGpuControl(domain, true)) {
                    if (synchronous) {
                        DrainGpuSource(domain);
                        continue;
                    }
                    return false;
                }
                if (!AdvanceGpuControl(domain, synchronous))
                    return false;
                domain.fixedGpuPhase = Phase::RetryCreation;
                continue;
            }
            if (!domain.solverStarted) {
                RetireIdleGpuPartition(domain);
                if (!(domain.solverStarted = TryBeginFixed(domain))) {
                    if (domain.waitingForSlot) {
                        domain.fixedGpuPhase = Phase::RetireControl;
                        continue;
                    }
                    if (synchronous && domain.gpuRegistered) {
                        DrainGpuSource(domain);
                        continue;
                    }
                    return false;
                }
            }
            if (domain.gpuRegistered && !AdvanceGpuJob(domain, synchronous))
                return false;
            FinishFixed(domain);
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
        if (store.IsValid(handle) &&
            (store.GetActor(handle).bodyId == UINT32_MAX || NeedsBodyBindingChange(store.GetActor(handle))))
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
bool PhysicsWorld::HasCpuControl(const PhysicsDomain &domain) const noexcept
{
    return domain.cpuControl &&
           (domain.cpuControl->HasPending() || !domain.cpuScene->UsesIgnoredPairs(domain.colliderPairSnapshot));
}

bool PhysicsWorld::HasDomainControl(PhysicsDomain &domain) const
{
    const auto &world = domain.owner;
    const auto &store = PhysicsECSStore::Instance();
    return domain.pendingRagdollHead || domain.controlPending || HasCpuControl(domain) ||
           (domain.gpuRegistered && m_gpuScene->HasPendingControl(domain.owner)) || store.HasDirtyColliders(world) ||
           store.HasPendingBodyCreations(world) || store.HasPendingBroadphaseAdds(world) ||
           store.HasPendingBroadphaseRemoves(world) || store.HasPendingJointDependencies(world) ||
           (PhysicMaterial::GetSurfaceRevision() != domain.surfaceRevision);
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
    // A published World has already released its writer. Its consumer must not
    // drive the shared GPU source again while another World owns the next job.
    if (m_gpuScene->IsWriterFor(domain.owner)) {
        if (synchronous) {
            DrainGpuSource(domain);
        } else {
            if (!m_gpuWork->IsActive())
                return false;
            AdmissionScope admission(*m_gpuScene, domain.owner);
            (void)m_gpuWork->AdvanceDependency(m_gpuScene->GetSourceId(), [this] { return Availability(*m_gpuHost); });
        }
    }
    const auto status = m_gpuScene->GetTickStatus(domain.owner);
    if (status == JoltGpuTick::Status::Failed)
        throw std::runtime_error("GPU physics step failed; stop the runtime before restarting");
    return status == JoltGpuTick::Status::Complete;
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
    if (!m_gpuScene->ConsumeCompletedControl(domain.owner))
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
    try {
        const bool completed = domain.stepPending && !domain.stepCompleted ? AdvanceFixedStep(domain, true)
                                                                           : AdvanceGpuControl(domain, true);
        if (!completed)
            throw std::runtime_error("Synchronous physics read could not complete its admitted step");
    } catch (...) {
        FailStep(domain);
        throw;
    }
}
void PhysicsWorld::ServiceGpuWork()
{
    RequireOwnerThread();
    if (!m_initialized)
        return;
    ServiceGpuRetirements();
    if (m_gpuScene && m_gpuWork->IsActive() && !m_gpuWork->IsLate() && m_gpuWork->ServiceRemaining() > 0) {
        auto *world = m_gpuScene->GetRunningObservationWorld();
        if (world && world->simulation.isPlaying && !world->handle->destroyRequested &&
            world->simulation.fixedPhase == RuntimeWorld::FixedPhase::Integration) {
            auto &dependency = *world->physics;
            if (!dependency.retiring && dependency.participantOutputPending && !dependency.stepPending) {
                // Step must preserve an already accepted observation before
                // admitting its next writer. This is a dependency of that
                // admitted Fixed even though its solver has not started yet.
                // Focus the shared source on this exact Work writer and use
                // only the current pre-render allowance; never drain or Step.
                const WorldExecutionScope ownerScope(world);
                struct RequiredWorldScope
                {
                    JoltGpuScene &scene;
                    RuntimeWorld *previous;
                    ~RequiredWorldScope()
                    {
                        scene.SetRequiredWorld(previous);
                    }
                } required{*m_gpuScene, m_gpuScene->SetRequiredWorld(world)};
                try {
                    (void)AdvanceGpuJob(dependency, false);
                } catch (...) {
                    FailStep(dependency);
                    throw;
                }
            }
        }
    }
    auto &domain = CurrentDomain();
    if (!domain.gpuRegistered || domain.fixedGpuPhase == PhysicsDomain::FixedGpuPhase::Failed)
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
    if (!m_initialized || domain.retiring || domain.fixedGpuPhase == PhysicsDomain::FixedGpuPhase::Failed)
        return;
    auto &manager = SceneManager::Instance();
    if (!domain.stepPending && !manager.HasPendingFixedStep() && !IsFixedPublicationHeld(domain)) {
        try {
            if (domain.cpuControl) {
                RefreshBodySurfaces(domain);
                if (HasCpuControl(domain))
                    ApplyCpuControl(domain);
            }
            RetireIdleGpuPartition(domain);
            if (!domain.cpuPoses.empty() || !domain.cpuInterpolated.empty())
                manager.PublishPhysicsStateToRenderer();
        } catch (...) {
            FailStep(domain);
            throw;
        }
    }
    if (!domain.gpuRegistered || !m_gpuRenderProgram)
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
    if (domain.retiring)
        return false;
    const auto applyAuthors = [&] {
        TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
        SceneManager::Instance().SyncDraggedTransforms(0.0f, false);
        RefreshBodySurfaces(domain);
        ApplyCpuControl(domain);
    };
    if (!domain.gpuRegistered) {
        try {
            if (domain.stepPending && !domain.stepCompleted && !AdvanceFixedStep(domain, false))
                return false;
            if (!IsFixedPublicationHeld(domain)) {
                applyAuthors();
                SceneManager::Instance().PublishPhysicsStateToRenderer();
            }
            return true;
        } catch (...) {
            FailStep(domain);
            throw;
        }
    }
    if (!m_gpuRenderProgram || !m_gpuWork->IsActive() || !m_gpuWork->IsAccepting() ||
        m_gpuWork->ServiceRemaining() <= 0 || !m_gpuWork->ResourceRemaining(FrameWorkResource::Cpu))
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
            applyAuthors();
        }
        RetireIdleGpuPartition(domain);
        if (!domain.cpuPoses.empty())
            SceneManager::Instance().PublishPhysicsStateToRenderer();
        if (!domain.gpuRegistered)
            return true;
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
    RetireGpuPartition(domain);
    domain.cpuContacts.Close();
    SuspendRagdolls(domain);
    domain.contacts.Close();
    RetireCpuJointRoutes(domain);
    domain.cpuFixedQueryWrite.reset();
    domain.cpuControl.reset();
    domain.cpuScene.reset();
    domain.cpuPublication = {};
    domain.cpuFixedAdmitted = false;
    domain.pendingStaticMoves.clear();
    domain.admittedStaticMoves.clear();
    domain.presentationPoseReads.reset();
    domain.cpuPoses.clear();
    domain.cpuInterpolated.clear();
    auto &world = domain.owner;
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
        if (!actor.sceneSuspended)
            domain.recreateActors.push_back(registration->actor);
        UnregisterBodyIdentity(domain, actor.bodyId);
        actor.bodyId = UINT32_MAX;
        actor.bodyInBroadphase = false;
        actor.jointEndpointBodyId = UINT32_MAX;
    });
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
    const_cast<PhysicsWorld *>(this)->RefreshBodySurfaces(domain);
}
void PhysicsWorld::RunCpuControl(PhysicsDomain &domain) const
{
    RequireOwnerThread();
    const WorldExecutionScope ownerScope(&domain.owner);
    RequireHealthyStep(domain);
    if (domain.synchronousRead)
        throw std::logic_error("Physics query cannot reenter its synchronous source");
    FlagScope reading(domain.synchronousRead);
    if (!IsFixedPublicationHeld(domain))
        const_cast<PhysicsWorld *>(this)->CompleteDomainStep(domain);
    if (IsFixedPublicationHeld(domain))
        return;
    FlushAuthoring(domain);
    ApplyCpuControl(domain);
}

void PhysicsWorld::ApplyCpuControl(PhysicsDomain &domain) const
{
    if (domain.cpuControl) {
        const std::unique_lock queryWrite(domain.cpuScene->QueryMutex());
        const bool pairsChanged = domain.cpuScene->SetIgnoredPairs(CaptureColliderPairs(domain));
        const bool commandsChanged = domain.cpuControl->ApplyControl(
            [this](uint32_t, const auto &ids) { const_cast<PhysicsWorld *>(this)->RetireCpuJointRoutes(ids); });
        if (pairsChanged || commandsChanged) {
            domain.cpuControl->PublishApplied();
            CaptureCpuPoses(domain, false);
            if (domain.cpuPublication.worldId != domain.owner.id ||
                domain.cpuPublication.simulationEpoch != domain.owner.simulation.simulationEpoch)
                domain.cpuPublication = {domain.owner.id, domain.owner.simulation.simulationEpoch, 0};
            domain.queryGeneration.fetch_add(1, std::memory_order_release);
            domain.requestedMotion.clear();
        }
    }
}
void PhysicsWorld::DrainQueryGpuCapacity(PhysicsDomain &domain) const
{
    while (domain.waitingForSlot) {
        while (!BeginGpuControl(domain, true))
            DrainGpuSource(domain);
        if (!AdvanceGpuControl(domain, true))
            throw std::runtime_error("GPU query control did not complete");
        const_cast<PhysicsWorld *>(this)->RetryCapacityBodyCreations(domain);
    }
}

void PhysicsWorld::AcceptQueryGpuPublication(PhysicsDomain &domain, bool refresh) const
{
    QueueColliderPairs(domain);
    const bool pending = m_gpuScene->HasPendingControl(domain.owner);
    if (refresh || pending || !m_gpuScene->IsCurrentPublication(domain.owner, m_gpuScene->GetPublished(domain.owner))) {
        while (!BeginGpuControl(domain, pending, {}, true))
            DrainGpuSource(domain);
        if (!AdvanceGpuControl(domain, true))
            throw std::runtime_error("GPU query control did not complete");
    }
}

void PhysicsWorld::RunControl(PhysicsDomain &domain, const JoltGpuScene::Consumer &consumer, bool publicationRead,
                              bool requireStaticGpuBoundaries) const
{
    RequireOwnerThread();
    const WorldExecutionScope ownerScope(&domain.owner);
    RequireHealthyStep(domain);
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
    RequireGpu(domain);
    try {
        {
            FrameWorkOwner::Scope preparation(*m_gpuWork);
            FlushAuthoring(domain);
        }
        // Device collision readers consume collider-only actors even when no
        // Rigidbody step runs. An empty pending set stages nothing; capacity
        // deferral retains the existing control cursor and accepted receipts.
        while (requireStaticGpuBoundaries) {
            {
                FrameWorkOwner::Scope preparation(*m_gpuWork);
                if (const_cast<PhysicsWorld *>(this)->EnsureStaticBoundaries(domain))
                    break;
            }
            if (!domain.waitingForSlot)
                throw std::runtime_error("GPU collision query is awaiting static collider acceptance");
            DrainQueryGpuCapacity(domain);
        }
        DrainQueryGpuCapacity(domain);
        if (publicationRead) {
            AcceptQueryGpuPublication(domain, HasDomainControl(domain));
            observe(m_gpuScene->GetPublished(domain.owner));
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

void PhysicsWorld::SealSpatialQuery(PhysicsDomain &domain) const
{
    RequireOwnerThread();
    RequireHealthyStep(domain);
    const WorldExecutionScope ownerScope(&domain.owner);
    if (domain.synchronousRead)
        throw std::logic_error("Physics query cannot reenter its synchronous source");
    FlagScope reading(domain.synchronousRead);
    if (!IsFixedPublicationHeld(domain))
        const_cast<PhysicsWorld *>(this)->CompleteDomainStep(domain);
    if (IsFixedPublicationHeld(domain))
        return;

    {
        std::optional<FrameWorkOwner::Scope> preparation;
        if (domain.gpuRegistered)
            preparation.emplace(*m_gpuWork);
        FlushAuthoring(domain);
    }
    while (!const_cast<PhysicsWorld *>(this)->EnsureStaticBoundaries(domain)) {
        if (!domain.waitingForSlot)
            throw std::runtime_error("Physics query has static collider geometry awaiting acceptance");
        DrainQueryGpuCapacity(domain);
    }
    if (domain.gpuRegistered) {
        DrainQueryGpuCapacity(domain);
        AcceptQueryGpuPublication(domain, false);
    }

    if (HasCpuControl(domain))
        ApplyCpuControl(domain);
}

std::shared_ptr<const JoltGpuScene::Publication> PhysicsWorld::ReadGpuQueryPublication(PhysicsDomain &domain) const
{
    if (!domain.gpuRegistered)
        return {};
    const bool held = IsFixedPublicationHeld(domain);
    auto publication = held ? domain.gpuContacts.GetPublication() : m_gpuScene->GetPublished(domain.owner);
    if (!publication || publication->worldId != domain.owner.id ||
        publication->simulationEpoch != domain.owner.simulation.simulationEpoch ||
        (held && (!publication->isFixedStep || publication->fixedStepIndex != domain.stepIndex)))
        throw std::logic_error("Physics query has no matching completed GPU publication");
    return publication;
}

std::shared_ptr<const JoltGpuScene::Publication>
PhysicsWorld::ReadGpuSpatialQueryPublication(PhysicsDomain &domain) const
{
    auto publication = ReadGpuQueryPublication(domain);
    // Test the accepted per-world layout only after sealing query sources and
    // validating their World/epoch. GPU participants and shapeless bodies can
    // publish numerical state without any collision leaves. Keep their source
    // available to state gathers, but do not run an empty spatial query on it.
    return publication && publication->leafCount ? publication : nullptr;
}

void PhysicsWorld::ObserveSpatialGpuQuery(PhysicsDomain &domain,
                                          const std::shared_ptr<const JoltGpuScene::Publication> &publication,
                                          const JoltGpuScene::Consumer &consumer) const
{
    RequireOwnerThread();
    RequireHealthyStep(domain);
    if (!publication)
        throw std::logic_error("Physics observation requires an accepted GPU publication");
    if (domain.synchronousRead)
        throw std::logic_error("Physics query cannot reenter its synchronous source");
    FlagScope reading(domain.synchronousRead);
    const WorldExecutionScope ownerScope(&domain.owner);
    AdmissionScope admission(*m_gpuScene, domain.owner);
    JoltGpuControlWork prepared;
    {
        FrameWorkOwner::Scope preparation(*m_gpuWork);
        prepared = consumer(publication);
    }
    m_gpuScene->RunPublicationObservation(domain.owner, publication, std::move(prepared));
}

std::shared_ptr<JoltGpuQueryResult>
PhysicsWorld::RunQuery(PhysicsDomain &domain, const std::shared_ptr<const JoltGpuScene::Publication> &publication,
                       const std::vector<QueryInput> &requests, uint64_t sweepColliderId,
                       uint64_t sweepColliderGeneration, uint32_t sweepBodyId) const
{
    JoltGpuPreparedQuery prepared;
    std::exception_ptr requestError;
    ObserveSpatialGpuQuery(domain, publication, [&](const auto &accepted) {
        return PrepareJoltGpuQueryWork(
            [&] {
                const auto *inputs = &requests;
                std::vector<QueryInput> sweepInputs;
                if (sweepColliderId) {
                    sweepInputs = requests;
                    const auto &directory = *accepted->directory;
                    const auto slot = directory.FindBodySlot(sweepBodyId);
                    const auto &actors = directory.actors;
                    const bool acceptedSelf = slot >= accepted->bodyBase &&
                                              slot - accepted->bodyBase < accepted->bodyCount &&
                                              std::any_of(
                                                  actors[slot]->leaves.begin(), actors[slot]->leaves.end(),
                                                  [&](const auto &leaf) {
                                                      return leaf.colliderId == sweepColliderId &&
                                                             leaf.colliderGeneration == sweepColliderGeneration;
                                                  });
                    for (auto &input : sweepInputs) {
                        input.meta.w |= QuerySweepPairPolicy;
                        input.lifetime = {static_cast<uint32_t>(sweepColliderGeneration),
                                          static_cast<uint32_t>(sweepColliderGeneration >> 32), 0, 0};
                        if (acceptedSelf) {
                            input.source.x = slot;
                            input.meta.w |= QueryExcludeBody;
                        }
                    }
                    inputs = &sweepInputs;
                }
                prepared = m_gpuQueries->Prepare(accepted, *inputs, {},
                                                 domain.queryGeneration.load(std::memory_order_acquire));
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
        auto work = PrepareJoltGpuQueryWork(
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

std::vector<Collider *> PhysicsWorld::QueryColliders(PhysicsDomain &domain, const QueryInput &request,
                                                     const JoltCpuQueries::Shape &shape) const
{
    SealSpatialQuery(domain);
    if (domain.cpuScene)
        return JoltCpuQueries(*domain.cpuScene).Overlap(shape, {request.filter.x, request.filter.y != 0});
    const auto gpu = ReadGpuSpatialQueryPublication(domain);
    std::vector<Collider *> colliders;
    if (!gpu)
        return colliders;
    const auto result = RunQuery(domain, gpu, {request});
    {
        FrameWorkOwner::Scope materialize(*m_gpuWork);
        const auto &records = result->ReadHits();
        colliders.reserve(records.size());
        for (const auto &record : records)
            if (auto *collider = DecodeQueryHit(*result, record).collider)
                colliders.push_back(collider);
    }
    return colliders;
}

bool PhysicsWorld::QueryClosestCast(PhysicsDomain &domain, const QueryInput &request,
                                    const JoltCpuQueries::Shape &shape, RaycastHit &hit) const
{
    SealSpatialQuery(domain);
    hit = {};
    if (domain.cpuScene)
        return JoltCpuQueries(*domain.cpuScene)
            .Cast(shape, Vector(request.direction), request.positionMaximum.w, hit,
                  {request.filter.x, request.filter.y != 0});
    if (const auto gpu = ReadGpuSpatialQueryPublication(domain)) {
        const auto result = RunQuery(domain, gpu, {request});
        FrameWorkOwner::Scope materialize(*m_gpuWork);
        hit = DecodeQueryHit(*result, result->ReadHits().front());
    }
    return hit.bodyId != UINT32_MAX;
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

bool PhysicsWorld::CanStageNewGpuBody(PhysicsDomain &domain, size_t count)
{
    const uint64_t required = m_gpuScene->GetBodyCount(domain.owner) + count;
    const uint64_t reserved = m_gpuScene->GetCapacity(domain.owner);
    if (required > reserved) {
        const uint64_t capacity = std::max({uint64_t{InitialBodyCapacity}, uint64_t{2} * reserved, required});
        if (capacity > UINT32_MAX)
            throw std::length_error("GPU physics world exceeds body indexing");
        m_gpuScene->ReserveCapacity(domain.owner, static_cast<uint32_t>(capacity));
    }
    if (m_gpuScene->CanReserve(domain.owner, count))
        return true;
    domain.waitingForSlot = true;
    return false;
}
void PhysicsWorld::StageNewGpuBody(PhysicsDomain &domain, uint32_t id, PhysicsActorHandle handle,
                                   const PhysicsActorData &actor, bool resident, PreparedGpuActor prepared)
{
    (void)m_gpuScene->Reserve(domain.owner, id, handle,
                              {PhysicsTargetReference(actor.owner->GetTransform()), actor.owner->GetID()}, resident,
                              actor.bodyId);
    try {
        m_gpuScene->StageActor(id, std::move(prepared));
    } catch (...) {
        m_gpuScene->StageRetire(id);
        throw;
    }
}
PhysicsWorld::Backend PhysicsWorld::SelectBodyBackend(const PhysicsActorData &actor,
                                                      const BodyRegistration *existing) const
{
    if (!actor.rigidbody)
        return Backend::StaticBoundary;
    if (existing && existing->rigidbody == actor.rigidbody->GetECSHandle() &&
        (!actor.rigidbody->IsEnabled() || actor.runtimeWorld->simulation.isPlaying))
        return existing->backend;
    if (!actor.rigidbody->IsEnabled())
        return Backend::StaticBoundary;
    return DomainForActor(actor).UsesGpu() ? Backend::Gpu : Backend::Cpu;
}
bool PhysicsWorld::NeedsBodyBindingChange(const PhysicsActorData &actor) const
{
    const auto *existing = FindBodyRegistration(actor.bodyId);
    if (!existing)
        return false;
    const auto backend = SelectBodyBackend(actor, existing);
    return existing->backend != backend ||
           (backend != Backend::StaticBoundary && existing->rigidbody != actor.rigidbody->GetECSHandle());
}

bool PhysicsWorld::EnsureStaticBoundary(PhysicsDomain &domain, uint32_t id)
{
    auto &entry = m_bodyActors.at(id);
    if (entry.backend != Backend::StaticBoundary)
        throw std::logic_error("Only a collider-only actor can supply a static boundary");
    if (entry.stagedOn) {
        domain.pendingStatic.erase(id);
        return true;
    }
    auto &boundary = *entry.staticModel;
    if (domain.UsesGpu()) {
        if (!EnsureGpuReady(domain) || !CanStageNewGpuBody(domain))
            return false;
        auto &prepared = std::get<PreparedGpuActor>(*boundary.pending);
        boundary.ApplyPose(prepared);
        StageNewGpuBody(domain, id, entry.actor, ActorForBody(id), entry.staticResident, prepared);
        entry.stagedOn = Backend::Gpu;
    } else {
        EnsureCpuReady(domain);
        auto &prepared = std::get<PreparedCpuActor>(*boundary.pending);
        boundary.ApplyPose(prepared);
        domain.cpuControl->StageActor(id, entry.actor, prepared);
        if (entry.staticResident)
            domain.cpuControl->AddBodies({id});
        entry.stagedOn = Backend::Cpu;
    }
    boundary.pending.reset();
    domain.pendingStatic.erase(id);
    return true;
}

bool PhysicsWorld::EnsureStaticBoundaries(PhysicsDomain &domain)
{
    while (!domain.pendingStatic.empty())
        if (!EnsureStaticBoundary(domain, *domain.pendingStatic.begin()))
            return false;
    return true;
}
uint32_t PhysicsWorld::StageNewBody(const PhysicsActorData &actor, PhysicsActorHandle handle, Backend backend,
                                    bool resident)
{
    auto &domain = DomainForActor(actor);
    auto &store = PhysicsECSStore::Instance();
    const bool boundary = backend == Backend::StaticBoundary;
    auto authoring = PhysicsActorAuthoring::Capture(actor);
    if (!authoring)
        return UINT32_MAX;
    std::optional<PreparedGpuActor> gpu;
    std::optional<PreparedCpuActor> cpu;
    if (domain.UsesGpu()) {
        gpu = PrepareJoltGpuActor(std::move(*authoring), 0, 0);
        if (!boundary && (!EnsureGpuReady(domain) || !CanStageNewGpuBody(domain))) {
            store.QueueBodyCreation(handle);
            return UINT32_MAX;
        }
    } else {
        cpu = PrepareJoltCpuActor(std::move(*authoring));
        if (!boundary)
            EnsureCpuReady(domain);
    }
    if (boundary)
        RegisterDomain(domain);

    const auto id = store.AllocateBodyIdentity();
    RegisterBodyIdentity(domain, id, handle, backend);
    auto &entry = m_bodyActors.at(id);
    try {
        if (boundary) {
            entry.staticResident = resident;
            entry.staticModel = std::make_shared<PhysicsStaticBoundaryModel>(*actor.owner->GetTransform());
            if (gpu) {
                gpu->publication.sharedGeometryRevision = entry.stagedShapeRevision;
                entry.staticModel->Hold(std::move(*gpu));
            } else {
                cpu->publication.sharedGeometryRevision = entry.stagedShapeRevision;
                entry.staticModel->Hold(std::move(*cpu));
            }
        } else {
            entry.rigidbody = actor.rigidbody->GetECSHandle();
            if (gpu) {
                StageNewGpuBody(domain, id, handle, actor, resident, std::move(*gpu));
                entry.stagedOn = Backend::Gpu;
            } else {
                domain.cpuControl->StageActor(id, handle, std::move(*cpu));
                entry.stagedOn = Backend::Cpu;
            }
        }
    } catch (...) {
        UnregisterBodyIdentity(domain, id);
        throw;
    }
    return id;
}
uint32_t PhysicsWorld::CreateBody(PhysicsActorData &actor)
{
    RequireOwnerThread();
    auto &store = PhysicsECSStore::Instance();
    const auto handle = store.FindActor(actor.owner);
    if (!store.IsValid(handle) || actor.bodyId != UINT32_MAX)
        throw std::logic_error("Physics body creation requires an ECS actor without a body");
    const auto id = StageNewBody(actor, handle, SelectBodyBackend(actor, nullptr), true);
    if (id != UINT32_MAX)
        actor.bodyId = id;
    return id;
}
void PhysicsWorld::RetireStagedBody(PhysicsDomain &domain, uint32_t id, const BodyRegistration &registration)
{
    if (registration.HasCpuBody() && domain.cpuControl)
        domain.cpuControl->RetireActor(id);
    else if (registration.HasGpuBody()) {
        // Closed partitions retire their numerical records as a batch.
        if (m_gpuScene && m_gpuScene->FindBody(id).bodyId != UINT32_MAX)
            m_gpuScene->StageRetire(id);
        if (m_gpuRenderProgram)
            domain.rendering.bindings.RetireBody(id);
    }
}
void PhysicsWorld::DestroyBody(uint32_t id)
{
    RequireOwnerThread();
    const auto *registration = FindBodyRegistration(id);
    if (!registration)
        return;
    auto &domain = DomainForActor(PhysicsECSStore::Instance().GetActor(registration->actor));
    RetireStagedBody(domain, id, *registration);
    UnregisterBodyIdentity(domain, id);
    domain.requestedMotion.erase(id);
    // Close an emptied partition at the removal boundary. A new scene can
    // otherwise acquire this World before the next render opportunity and
    // admit stale create/remove work that no longer has a consumer.
    if (!domain.stepPending && !IsFixedPublicationHeld(domain))
        RetireIdleGpuPartition(domain);
}
bool PhysicsWorld::RebindBody(PhysicsActorData &actor)
{
    auto &store = PhysicsECSStore::Instance();
    auto &domain = DomainForActor(actor);
    const auto id = actor.bodyId;
    auto &entry = m_bodyActors.at(id);
    // A sealed Fixed owns the old binding until publication finishes. A
    // capacity retry before solver admission may still stage the new binding.
    if ((domain.stepPending && domain.solverStarted) || IsFixedPublicationHeld(domain)) {
        store.QueueBodyCreation(entry.actor);
        return false;
    }
    const auto backend = SelectBodyBackend(actor, &entry);
    const bool boundary = backend == Backend::StaticBoundary;
    const bool stage = !boundary || entry.stagedOn.has_value();
    const bool resident = actor.owner->IsActiveInHierarchy() && !actor.sceneSuspended &&
                          ((actor.rigidbody && actor.rigidbody->IsEnabled()) || actor.primaryCollider);
    const auto rigidbody = boundary ? PhysicsECSStore::RigidbodyHandle{} : actor.rigidbody->GetECSHandle();

    TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    const auto revision = entry.stagedShapeRevision + 1;
    const bool gpuWorld = domain.UsesGpu();
    if (gpuWorld && stage && (!EnsureGpuReady(domain) || (!entry.stagedOn && !CanStageNewGpuBody(domain)))) {
        store.QueueBodyCreation(entry.actor);
        return false;
    }
    auto authoring = PhysicsActorAuthoring::Capture(actor);
    if (!authoring)
        return false;
    std::optional<PreparedCpuActor> cpu;
    std::optional<PreparedGpuActor> gpu;
    if (gpuWorld) {
        const auto slot = entry.stagedOn ? m_gpuScene->FindBody(id).slot : 0;
        gpu = PrepareJoltGpuActor(std::move(*authoring), slot, 0);
        gpu->publication.sharedGeometryRevision = revision;
    } else {
        cpu = PrepareJoltCpuActor(std::move(*authoring));
        cpu->publication.sharedGeometryRevision = revision;
        if (stage)
            EnsureCpuReady(domain);
    }
    auto staticModel = boundary ? std::make_shared<PhysicsStaticBoundaryModel>(*actor.owner->GetTransform()) : nullptr;
    const bool resetMotion = entry.rigidbody != rigidbody || entry.backend != backend;
    // Joint endpoint ownership belongs to the old component generation even
    // though the body's public ID remains the same.
    DestroyConstraintsForBody(id);
    if (!stage) {
        if (gpu)
            staticModel->Hold(std::move(*gpu));
        else
            staticModel->Hold(std::move(*cpu));
    } else if (gpu) {
        if (entry.stagedOn) {
            m_gpuScene->StageActor(id, std::move(*gpu), resetMotion);
            if (resetMotion) {
                const auto *transform = actor.owner->GetTransform();
                const auto position = Vector(transform->GetPosition());
                const auto rotation = Rotation(transform->GetWorldRotation());
                m_gpuScene->QueueBodyCommand(
                    {id, PhysicsBodyCommand::Pose{Vector(position),
                                                  glm::quat(rotation.w, rotation.x, rotation.y, rotation.z)}});
            }
            if (!resident)
                m_gpuScene->SetResident(id, false);
        } else {
            StageNewGpuBody(domain, id, entry.actor, actor, resident, std::move(*gpu));
            entry.stagedOn = Backend::Gpu;
        }
    } else {
        const auto position = cpu->settings.mPosition;
        const auto rotation = cpu->settings.mRotation;
        domain.cpuControl->StageActor(id, entry.actor, std::move(*cpu), resetMotion);
        if (entry.stagedOn && resetMotion)
            domain.cpuControl->Queue(id, JoltCpuControl::Pose{position, rotation});
        if (!resident && entry.stagedOn)
            domain.cpuControl->Queue(id, JoltCpuControl::Remove{});
        entry.stagedOn = Backend::Cpu;
    }

    domain.pendingStatic.erase(id);
    if (boundary && !entry.stagedOn)
        domain.pendingStatic.insert(id);
    const auto poseIndex = entry.cpuPoseIndex;
    if (poseIndex < domain.cpuPoses.size() && domain.cpuPoses[poseIndex].pose.bodyId == id)
        domain.cpuPoses[poseIndex].pose.bodyId = UINT32_MAX;
    entry.cpuPoseIndex = SIZE_MAX;
    domain.cpuInterpolated.erase(id);
    domain.requestedMotion.erase(id);
    store.CancelBroadphaseAdd(domain.owner, id);
    store.CancelBroadphaseRemove(domain.owner, id);
    ChangeBodyBackend(domain, entry, backend);
    entry.rigidbody = rigidbody;
    entry.staticModel = std::move(staticModel);
    entry.staticResident = resident;
    entry.stagedShapeRevision = revision;
    domain.pendingStaticMoves.erase(id);
    actor.bodyInBroadphase = resident;
    if (resident)
        store.QueueBroadphaseAdd(domain.owner, id);
    return true;
}
bool PhysicsWorld::CanQueueRigidbodyCommand(uint32_t id, const Rigidbody &body) const
{
    RequireOwnerThread();
    const auto *registration = FindBodyRegistration(id);
    if (!registration || registration->rigidbody != body.GetECSHandle())
        return false;
    const auto &domain = DomainForBody(id);
    if (registration->backend == Backend::Cpu)
        return domain.cpuControl && domain.cpuControl->HasStagedRigidbody(id, body.GetECSHandle());
    return registration->backend == Backend::Gpu && m_gpuScene &&
           m_gpuScene->HasStagedRigidbody(id, body.GetECSHandle());
}
void PhysicsWorld::RefreshGpuActor(PhysicsDomain &, uint32_t id, const std::function<void(PreparedGpuActor &)> &edit)
{
    auto &actor = ActorForBody(id);
    StageActorModel(actor, actor, {}, edit);
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
void PhysicsWorld::StageActorModel(PhysicsActorData &actor, const PhysicsActorData &author,
                                   const std::function<void(PreparedCpuActor &)> &editCpu,
                                   const std::function<void(PreparedGpuActor &)> &editGpu)
{
    RequireOwnerThread();
    if (NeedsBodyBindingChange(actor)) {
        PhysicsECSStore::Instance().QueueBodyCreation(m_bodyActors.at(actor.bodyId).actor);
        return;
    }
    auto &domain = DomainForActor(actor);
    RequireHealthyStep(domain);
    auto &entry = m_bodyActors.at(actor.bodyId);
    const auto revision = entry.stagedShapeRevision + 1;
    auto authoring = PhysicsActorAuthoring::Capture(author);
    if (!authoring)
        return;
    if (domain.UsesGpu()) {
        const auto slot = entry.HasGpuBody() ? m_gpuScene->FindBody(actor.bodyId).slot : 0;
        auto prepared = PrepareJoltGpuActor(std::move(*authoring), slot, 0);
        if (editGpu)
            editGpu(prepared);
        prepared.publication.sharedGeometryRevision = revision;
        if (entry.staticModel)
            entry.staticModel->ApplyPose(prepared);
        if (entry.HasGpuBody()) {
            RequireGpu(domain);
            m_gpuScene->StageActor(actor.bodyId, std::move(prepared));
        } else {
            entry.staticModel->Hold(std::move(prepared));
        }
    } else {
        auto prepared = PrepareJoltCpuActor(std::move(*authoring));
        if (editCpu)
            editCpu(prepared);
        prepared.publication.sharedGeometryRevision = revision;
        if (entry.staticModel)
            entry.staticModel->ApplyPose(prepared);
        if (entry.HasCpuBody())
            domain.cpuControl->StageActor(actor.bodyId, entry.actor, std::move(prepared));
        else
            entry.staticModel->Hold(std::move(prepared));
    }
    entry.stagedShapeRevision = revision;
    if (!entry.staticModel && actor.rigidbody)
        actor.rigidbody->FlushPendingBodyCommands();
}
void PhysicsWorld::UpdateBodyShape(PhysicsActorData &actor, const Collider *exclude)
{
    if (!exclude) {
        StageActorModel(actor, actor);
        return;
    }
    auto author = actor;
    author.colliders.erase(std::remove(author.colliders.begin(), author.colliders.end(), exclude),
                           author.colliders.end());
    StageActorModel(actor, author);
}
void PhysicsWorld::SetBodyPosition(uint32_t id, const glm::vec3 &position, const glm::quat &rotation,
                                   bool cancelKinematicMove)
{
    QueueBodyPose(id, &position, &rotation, cancelKinematicMove);
}
void PhysicsWorld::SetBodyPositionOnly(uint32_t id, const glm::vec3 &position)
{
    QueueBodyPose(id, &position, nullptr, true);
}
void PhysicsWorld::SetBodyRotationOnly(uint32_t id, const glm::quat &rotation)
{
    QueueBodyPose(id, nullptr, &rotation, true);
}
void PhysicsWorld::QueueBodyPose(uint32_t id, const glm::vec3 *position, const glm::quat *rotation, bool cancelMove)
{
    RequireOwnerThread();
    auto &domain = DomainForBody(id);
    const auto &entry = m_bodyActors.at(id);
    const auto p = position ? Vector(*position) : Float4{};
    const auto q = rotation ? Rotation(*rotation) : Float4{};
    RecordStaticPose(domain, id, position, rotation, cancelMove);
    if (entry.HasCpuBody())
        domain.cpuControl->Queue(id, JoltCpuControl::Pose{position ? std::make_optional(CpuPosition(p)) : std::nullopt,
                                                          rotation ? std::make_optional(CpuRotation(q)) : std::nullopt,
                                                          cancelMove});
    if (entry.HasGpuBody()) {
        RequireGpu(domain);
        m_gpuScene->QueueBodyCommand(
            {id, PhysicsBodyCommand::Pose{position ? std::make_optional(*position) : std::nullopt,
                                          rotation ? std::make_optional(glm::quat(q.w, q.x, q.y, q.z)) : std::nullopt,
                                          cancelMove}});
        domain.MarkGpuPoseDiscontinuity(id);
    }
}
void PhysicsWorld::SetBodyPositionsBatch(const std::vector<PhysicsBodyPoseUpdate> &updates)
{
    RequireOwnerThread();
    for (const auto &u : updates) {
        auto &domain = DomainForBody(u.bodyId);
        if (m_bodyActors.at(u.bodyId).HasGpuBody())
            RequireGpu(domain);
        (void)Vector(u.position);
        (void)Rotation(u.rotation);
    }
    for (const auto &u : updates)
        SetBodyPosition(u.bodyId, u.position, u.rotation);
}
void PhysicsWorld::AddBodiesBatch(const std::vector<uint32_t> &bodies)
{
    RequireOwnerThread();
    std::map<PhysicsDomain *, std::vector<uint32_t>> cpuBodies;
    for (const auto id : bodies) {
        auto &domain = DomainForBody(id);
        const auto &entry = m_bodyActors.at(id);
        if (entry.HasGpuBody())
            RequireGpu(domain);
        if (entry.HasCpuBody())
            cpuBodies[&domain].push_back(id);
    }
    for (auto &[domain, ids] : cpuBodies)
        domain->cpuControl->AddBodies(std::move(ids));
    for (const auto id : bodies) {
        auto &entry = m_bodyActors.at(id);
        if (entry.backend == Backend::StaticBoundary)
            entry.staticResident = true;
        if (entry.HasGpuBody())
            m_gpuScene->SetResident(id, true);
    }
    for (const auto id : bodies)
        if (auto *body = FindRigidbodyByBodyId(id))
            body->FlushPendingBodyCommands();
}
void PhysicsWorld::RemoveBodyFromBroadphase(uint32_t id, bool preserveVelocity)
{
    RequireOwnerThread();
    const auto *entry = FindBodyRegistration(id);
    if (!entry)
        return;
    auto &domain = DomainForBody(id);
    if (entry->HasCpuBody() && domain.cpuControl)
        domain.cpuControl->Queue(id, JoltCpuControl::Remove{preserveVelocity});
    if (entry->HasGpuBody() && m_gpuScene && m_gpuScene->FindBody(id).bodyId != UINT32_MAX)
        m_gpuScene->SetResident(id, false, preserveVelocity);
    if (entry->backend == Backend::StaticBoundary) {
        m_bodyActors.at(id).staticResident = false;
        domain.pendingStaticMoves.erase(id);
    }
}
bool PhysicsWorld::IsBodyInBroadphase(uint32_t id) const
{
    RequireOwnerThread();
    if (id == UINT32_MAX)
        return false;
    auto &domain = DomainForBody(id);
    const auto &entry = m_bodyActors.at(id);
    if (entry.HasCpuBody()) {
        RunCpuControl(domain);
        return domain.cpuScene->IsBodyInBroadphase(id);
    }
    if (!entry.HasGpuBody())
        return false;
    RunControl(domain, {});
    const auto publication =
        IsFixedPublicationHeld(domain) ? domain.gpuContacts.GetPublication() : m_gpuScene->GetPublished(domain.owner);
    if (!publication)
        return false;
    const auto &directory = *publication->directory;
    const auto found = directory.bodySlots.find(id);
    return found != directory.bodySlots.end() && directory.residentBodies[found->second];
}
void PhysicsWorld::SetBodyMotionType(uint32_t id, int type)
{
    if (type < 0 || type > 2)
        throw std::invalid_argument("Physics motion type must be static, kinematic or dynamic");
    RefreshBodyLayers(id);
}
void PhysicsWorld::RefreshBodyLayers(uint32_t id)
{
    auto &actor = ActorForBody(id);
    StageActorModel(actor, actor);
}
void PhysicsWorld::SetBodyDamping(uint32_t id, float linear, float angular)
{
    if (!std::isfinite(linear) || !std::isfinite(angular) || linear < 0 || angular < 0)
        throw std::invalid_argument("Damping must be finite and nonnegative");
    auto &actor = ActorForBody(id);
    StageActorModel(
        actor, actor,
        [=](PreparedCpuActor &a) {
            a.settings.mLinearDamping = linear;
            a.settings.mAngularDamping = angular;
        },
        [=](PreparedGpuActor &a) {
            a.body.damping.x = linear;
            a.body.damping.y = angular;
        });
}
void PhysicsWorld::SetBodyGravityFactor(uint32_t id, float factor)
{
    if (!std::isfinite(factor))
        throw std::invalid_argument("Gravity factor must be finite");
    auto &actor = ActorForBody(id);
    StageActorModel(
        actor, actor, [=](PreparedCpuActor &a) { a.settings.mGravityFactor = factor; },
        [=](PreparedGpuActor &a) { a.body.mass.w = factor; });
}
void PhysicsWorld::SetBodyFriction(uint32_t id, float value)
{
    auto &actor = ActorForBody(id);
    StageActorModel(
        actor, actor,
        [=](PreparedCpuActor &a) {
            a.settings.mFriction = value;
            for (auto &surface : a.surfaces)
                surface.friction = value;
        },
        [=](PreparedGpuActor &a) {
            for (auto &leaf : a.geometry.leaves)
                leaf.model.material.x = value;
        });
}
void PhysicsWorld::SetBodyRestitution(uint32_t id, float value)
{
    auto &actor = ActorForBody(id);
    StageActorModel(
        actor, actor,
        [=](PreparedCpuActor &a) {
            a.settings.mRestitution = value;
            for (auto &surface : a.surfaces)
                surface.restitution = value;
        },
        [=](PreparedGpuActor &a) {
            for (auto &leaf : a.geometry.leaves)
                leaf.model.material.y = value;
        });
}
void PhysicsWorld::SetBodyMaxAngularVelocity(uint32_t id, float value)
{
    auto &actor = ActorForBody(id);
    StageActorModel(
        actor, actor, [=](PreparedCpuActor &a) { a.settings.mMaxAngularVelocity = value; },
        [=](PreparedGpuActor &a) { a.body.damping.w = value; });
}
void PhysicsWorld::SetBodyMaxLinearVelocity(uint32_t id, float value)
{
    auto &actor = ActorForBody(id);
    StageActorModel(
        actor, actor, [=](PreparedCpuActor &a) { a.settings.mMaxLinearVelocity = value; },
        [=](PreparedGpuActor &a) { a.body.damping.z = value; });
}
void PhysicsWorld::SetBodyAllowedDOFs(uint32_t id, int allowed, float mass)
{
    if (allowed < 0 || (allowed & ~0x3f) != 0)
        throw std::invalid_argument("Allowed degrees of freedom contain unknown bits");
    if (!std::isfinite(mass) || mass <= 0)
        throw std::invalid_argument("Body mass must be positive and finite");
    auto &actor = ActorForBody(id);
    StageActorModel(
        actor, actor, [=](PreparedCpuActor &a) { a.settings.mAllowedDOFs = static_cast<JPH::EAllowedDOFs>(allowed); },
        [=](PreparedGpuActor &a) {
            a.body.mass.y = float((~allowed) & 7);
            a.body.mass.z = float(((~allowed) >> 3) & 7);
        });
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
    QueueBodyVelocity(id, v, false);
}
void PhysicsWorld::SetBodyAngularVelocity(uint32_t id, const glm::vec3 &v)
{
    QueueBodyVelocity(id, v, true);
}
void PhysicsWorld::QueueBodyVelocity(uint32_t id, const glm::vec3 &v, bool angular)
{
    RequireOwnerThread();
    auto &domain = DomainForBody(id);
    const auto backend = m_bodyActors.at(id).backend;
    if (backend == Backend::StaticBoundary)
        throw std::invalid_argument("Physics motion command requires a Rigidbody");
    const auto value = Vector(v);
    if (backend == Backend::Cpu) {
        const auto velocity = std::make_optional(CpuVector(value));
        domain.cpuControl->Queue(
            id, JoltCpuControl::Velocity{angular ? std::nullopt : velocity, angular ? velocity : std::nullopt});
        return;
    }
    RequireGpu(domain);
    const auto velocity = std::make_optional(v);
    m_gpuScene->QueueBodyCommand(
        {id, PhysicsBodyCommand::Velocity{angular ? std::nullopt : velocity, angular ? velocity : std::nullopt}});
}
void PhysicsWorld::QueueBodyForce(uint32_t id, int kind, int mode, const glm::vec3 &value, const glm::vec3 &point)
{
    RequireOwnerThread();
    auto &domain = DomainForBody(id);
    const auto backend = m_bodyActors.at(id).backend;
    if (backend == Backend::StaticBoundary)
        throw std::invalid_argument("Physics motion command requires a Rigidbody");
    if (kind < 0 || kind > 2 || mode < 0 || mode > 3)
        throw std::invalid_argument("Invalid physics force kind or mode");
    const auto force = Vector(value);
    const auto origin = kind == 2 ? Vector(point) : Float4{};
    if (backend == Backend::Cpu) {
        domain.cpuControl->Queue(id, JoltCpuControl::Force{static_cast<JoltCpuScene::ForceKind>(kind),
                                                           static_cast<ForceMode>(mode), CpuVector(force),
                                                           CpuPosition(origin)});
        return;
    }
    RequireGpu(domain);
    m_gpuScene->QueueBodyCommand({id, PhysicsBodyCommand::Force{static_cast<PhysicsForceKind>(kind),
                                                                static_cast<ForceMode>(mode), value, Vector(origin)}});
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
    RequireHealthyStep(domain);
    if (domain.stepPending || IsFixedPublicationHeld(domain))
        throw std::logic_error("Physics reset requires a boundary outside contact delivery");
}
void PhysicsWorld::ResetBodyMotionStates(const std::vector<PhysicsBodyMotionState> &states)
{
    RequireOwnerThread();
    for (const auto &s : states) {
        auto &domain = DomainForBody(s.bodyId);
        const auto backend = m_bodyActors.at(s.bodyId).backend;
        if (backend == Backend::StaticBoundary)
            throw std::invalid_argument("Physics reset requires a Rigidbody");
        if (backend == Backend::Gpu)
            RequireGpu(domain);
        if (domain.stepPending || IsFixedPublicationHeld(domain))
            throw std::logic_error("Physics reset requires completed fixed steps");
        (void)Vector(s.position);
        (void)Rotation(s.rotation);
        (void)Vector(s.linearVelocity);
        (void)Vector(s.angularVelocity);
    }
    for (const auto &s : states) {
        if (m_bodyActors.at(s.bodyId).backend == Backend::Cpu) {
            auto &control = *DomainForBody(s.bodyId).cpuControl;
            control.QueueMotionReset({s.bodyId, CpuPosition(Vector(s.position)), CpuRotation(Rotation(s.rotation)),
                                      CpuVector(Vector(s.linearVelocity)), CpuVector(Vector(s.angularVelocity))});
            continue;
        }
        const auto q = Rotation(s.rotation);
        m_gpuScene->QueueBodyCommand(
            {s.bodyId, PhysicsBodyCommand::ResetMotion{s.position, glm::quat(q.w, q.x, q.y, q.z), s.linearVelocity,
                                                       s.angularVelocity}});
        DomainForBody(s.bodyId).MarkGpuPoseDiscontinuity(s.bodyId);
    }
}

JoltGpuControlWork PhysicsWorld::PrepareStateBatch(
    PhysicsDomain &domain, const std::shared_ptr<const JoltGpuScene::Publication> &publication,
    const std::vector<Rigidbody *> &bodies, PhysicsStateFields fields, bool includeBoxes, bool triggers,
    const std::vector<JoltGpuQueryExportBinding> &exports, PhysicsRigidbodyStateBatch &batch,
    std::shared_ptr<JoltGpuQueryResult> &motion, std::shared_ptr<JoltGpuQueryResult> &shapes,
    std::exception_ptr &requestError, bool deferExport) const
{
    return PrepareJoltGpuQueryWork(
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
                const auto slot = publication->directory->FindBodySlot(body->GetBodyId());
                if (slot < publication->bodyBase || slot - publication->bodyBase >= publication->bodyCount)
                    throw JoltGpuQueryRequestError("Physics batch requires accepted body identities");
                const auto &identity = *publication->directory->actors[slot];
                if (identity.rigidbodyComponentId != body->GetComponentID() ||
                    identity.rigidbody != body->GetECSHandle())
                    throw JoltGpuQueryRequestError("Physics batch Rigidbody differs from its retained publication");
                batch.states[index].bodyId = identity.bodyId;
                batch.states[index].shapeRevision = identity.shapeRevision;
                if (numericMotion) {
                    auto input = Query(QueryGatherMotion, QueryClosest, 0, false);
                    input.meta.z = 0;
                    input.source.y = slot;
                    input.source.z = static_cast<uint32_t>(fields);
                    input.geometry = identity.rootCenterOfMass;
                    motionInputs.push_back(input);
                }
                if (includeBoxes) {
                    auto input = Query(QueryGatherShapes, QueryAll, UINT32_MAX, triggers);
                    input.meta.z = ShapeBox;
                    input.source.y = slot;
                    shapeInputs.push_back(input);
                }
            }
            const uint64_t generation = domain.queryGeneration.load(std::memory_order_acquire) +
                                        static_cast<uint64_t>(domain.controlNumericalChange);
            batch.sources = {
                {publication->worldId, publication->simulationEpoch, publication->fixedStepIndex, generation}};
            batch.sourceForBody.assign(bodies.size(), 0u);
            const auto readback = deferExport       ? JoltGpuQueryReadback::DeviceWithSpans
                                  : exports.empty() ? JoltGpuQueryReadback::HostProjected
                                                    : JoltGpuQueryReadback::Device;
            JoltGpuPreparedQuery first, second;
            if (!motionInputs.empty()) {
                first = m_gpuQueries->Prepare(publication, motionInputs, {}, generation, readback);
                motion = first.result;
            }
            if (!shapeInputs.empty()) {
                second = m_gpuQueries->Prepare(publication, shapeInputs, {}, generation, readback);
                shapes = second.result;
            }
            return ContinueJoltGpuControlWork(
                std::move(first.work), [this, &domain, motionResult = motion, shapeResult = shapes,
                                        work = std::move(second.work), &batch, exports, deferExport]() mutable {
                    if (motionResult && motionResult->GetSummary().counts.y != QuerySuccess)
                        return JoltGpuControlWork{};
                    return ContinueJoltGpuControlWork(
                        std::move(work), [this, &domain, motionResult, shapeResult, &batch, exports, deferExport] {
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
        } else {
            (void)motion->GetRecordCount();
        }
    }
    if (shapes) {
        if (numerical) {
            const auto &spans = shapes->ReadSpans();
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
        } else {
            batch.boxCount = shapes->GetRecordCount();
        }
    }
}

PhysicsRigidbodyStateBatch
PhysicsWorld::GatherRigidbodyStates(const std::vector<Rigidbody *> &bodies, PhysicsStateFields fields,
                                    bool includeBoxes, bool triggers,
                                    const std::vector<JoltGpuQueryExportBinding> &exports) const
{
    return GatherRigidbodyStateRows(bodies, fields, includeBoxes, triggers, exports, false, {});
}

PhysicsRigidbodyStateBatch
PhysicsWorld::GatherRigidbodyStateRows(const std::vector<Rigidbody *> &bodies, PhysicsStateFields fields,
                                       bool includeBoxes, bool triggers,
                                       const std::vector<JoltGpuQueryExportBinding> &exports, bool sealed,
                                       const std::shared_ptr<const JoltGpuScene::Publication> &gpu) const
{
    RequireOwnerThread();
    if (bodies.empty())
        return {};
    struct Group
    {
        PhysicsDomain *domain;
        Backend backend;
        std::vector<Rigidbody *> bodies;
        std::vector<JoltCpuQueries::BodyRequest> cpuRequests;
        std::vector<size_t> indices;
        PhysicsRigidbodyStateBatch batch;
        std::shared_ptr<JoltGpuQueryResult> motion, shapes;
        std::shared_ptr<const JoltGpuScene::Publication> publication;
    };
    std::vector<Group> groups;
    std::unordered_map<PhysicsDomain *, size_t> byOwner;
    std::vector<uint8_t> cpuRows;
    const bool exportMotion = std::any_of(exports.begin(), exports.end(), [](const auto &value) {
        return value.field <= JoltGpuQueryExportField::MotionInverseInertia;
    });
    const auto cpuFields = exportMotion ? PhysicsStateFields::All : fields;
    for (size_t row = 0; row < bodies.size(); ++row) {
        auto *body = bodies[row];
        if (!body || !body->IsEnabled() || body->IsDestroyed() || !body->GetGameObject())
            throw std::invalid_argument("Physics batch requires live enabled Rigidbody components");
        const auto bodyId = body->GetBodyId();
        const auto *registration = FindBodyRegistration(bodyId);
        if (!registration || registration->backend == Backend::StaticBoundary)
            throw std::invalid_argument("Physics batch requires registered Rigidbody identities");
        auto &domain = DomainForComponent(*body);
        const auto [index, inserted] = byOwner.try_emplace(&domain, groups.size());
        if (inserted)
            groups.push_back(Group{&domain, registration->backend});
        auto &group = groups[index->second];
        if (group.backend == Backend::Cpu) {
            group.cpuRequests.push_back({bodyId, body});
            if (!exports.empty()) {
                if (cpuRows.empty())
                    cpuRows.resize(bodies.size());
                cpuRows[row] = 1;
            }
        } else {
            group.bodies.push_back(body);
        }
        group.indices.push_back(row);
    }
    const bool direct = groups.size() == 1 && (groups.front().backend == Backend::Gpu || exports.empty());
    const auto aggregateRows = direct ? size_t{0} : bodies.size();
    PhysicsRigidbodyStateBatch batch;
    if (!direct)
        batch.bodies = bodies;
    batch.states.resize(aggregateRows);
    batch.sourceForBody.resize(aggregateRows);
    std::vector<uint32_t> shapeCounts(aggregateRows, 0u);
    std::vector<std::vector<PhysicsRigidbodyBoxState>> boxes(aggregateRows);
    for (auto &group : groups) {
        auto &domain = *group.domain;
        const uint32_t source = static_cast<uint32_t>(batch.sources.size());
        if (group.backend == Backend::Cpu) {
            if (!sealed)
                RunCpuControl(domain);
            group.batch = JoltCpuQueries(*domain.cpuScene)
                              .GatherRigidbodyStates(group.cpuRequests, cpuFields, includeBoxes, triggers);
            const auto generation = domain.queryGeneration.load(std::memory_order_acquire);
            const auto &p = domain.cpuPublication;
            batch.sources.push_back({p.worldId, p.simulationEpoch, p.fixedStepIndex, generation});
            if (exports.empty() && (static_cast<uint32_t>(fields) & static_cast<uint32_t>(PhysicsStateFields::All)))
                for (const auto &state : group.batch.states)
                    domain.requestedMotion[state.bodyId] = {generation, state, fields};
            if (direct) {
                // The native query already wrote body and box rows in caller order.
                if (group.batch.boxCount > UINT32_MAX)
                    throw std::length_error("Physics shape aggregate exceeds device indexing");
                group.batch.sources = std::move(batch.sources);
                group.batch.sourceForBody.resize(bodies.size());
                return std::move(group.batch);
            }
        } else {
            std::exception_ptr error;
            const auto prepare = [&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
                group.publication = publication;
                return PrepareStateBatch(domain, publication, group.bodies, fields, includeBoxes, triggers, exports,
                                         group.batch, group.motion, group.shapes, error, !direct && !exports.empty());
            };
            if (sealed) {
                ObserveSpatialGpuQuery(domain, gpu, prepare);
            } else {
                RunControl(domain, prepare, exports.empty());
            }
            if (error)
                std::rethrow_exception(error);
            MaterializeStateBatch(domain, group.batch, group.motion, group.shapes, exports.empty(), fields);
            // One world's query rows already follow caller order. The device can export
            // the entire batch, including the owning body of each compound shape.
            if (direct)
                return std::move(group.batch);
            const auto &p = *group.publication;
            const uint64_t generation = group.motion   ? group.motion->numericalGeneration
                                        : group.shapes ? group.shapes->numericalGeneration
                                                       : domain.queryGeneration.load(std::memory_order_acquire);
            batch.sources.push_back({p.worldId, p.simulationEpoch, p.fixedStepIndex, generation});
        }
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
            if (!spans)
                ++shapeCounts[box.bodyIndex];
            boxes[box.bodyIndex].push_back(std::move(box));
        }
    }
    std::vector<uint64_t> shapeFirst(bodies.size());
    uint64_t boxCount = 0;
    for (size_t row = 0; row < bodies.size(); ++row) {
        shapeFirst[row] = boxCount;
        boxCount += shapeCounts[row];
        if (exports.empty())
            batch.boxes.insert(batch.boxes.end(), boxes[row].begin(), boxes[row].end());
    }
    if (boxCount > UINT32_MAX)
        throw std::length_error("Physics shape aggregate exceeds device indexing");
    batch.boxCount = static_cast<size_t>(boxCount);
    if (exports.empty())
        return batch;
    auto cpuTransfers = PrepareCpuStateExports(batch, cpuRows, boxes, shapeFirst, exports, includeBoxes);
    std::vector<std::pair<PhysicsDomain *, JoltGpuControlWork>> plannedExports;
    for (auto &group : groups) {
        if (group.backend == Backend::Cpu)
            continue;
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
                if (field >= kStateExportWidths.size())
                    throw JoltGpuQueryRequestError("Physics aggregate export has an unknown field");
                const bool motion = field <= static_cast<uint32_t>(JoltGpuQueryExportField::MotionInverseInertia);
                const uint64_t outputRow = motion ? row : shapeFirst[row];
                const uint64_t byteAdvance = outputRow * kStateExportWidths[field] * sizeof(uint32_t);
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
                work = ContinueJoltGpuControlWork(
                    std::move(work), [projected = std::move(projected)]() mutable { return std::move(projected); });
            }
        }
        if (work.packet || work.continueWork)
            plannedExports.emplace_back(group.domain, std::move(work));
    }
    for (auto &planned : plannedExports)
        RunPreparedWork(*planned.first, std::move(planned.second));
    if (cpuTransfers)
        (void)cpuTransfers->Submit();
    for (size_t row = 0; row < cpuRows.size(); ++row) {
        if (cpuRows[row]) {
            const auto &state = batch.states[row];
            batch.states[row] = PhysicsBodyMotionState{state.bodyId, state.shapeRevision};
        }
    }
    return batch;
}

PhysicsRigidbodyStateBatch
PhysicsWorld::GatherRigidbodyStatesInBounds(const glm::vec3 &minimum, const glm::vec3 &maximum, uint32_t layers,
                                            bool triggers, PhysicsStateFields fields, bool includeBoxes,
                                            const std::vector<JoltGpuQueryExportBinding> &exports) const
{
    auto &domain = CurrentDomain();
    SealSpatialQuery(domain);
    const auto gpu = ReadGpuSpatialQueryPublication(domain);

    const auto bodies = QueryRigidbodyCandidatesInBounds(domain, gpu, minimum, maximum, layers, triggers);
    return GatherRigidbodyStateRows(bodies, fields, includeBoxes, triggers, exports, true, gpu);
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
        const auto *entry = FindBodyRegistration(actor.bodyId);
        if (entry && entry->backend == Backend::Cpu) {
            if (auto *parent = object->GetParent())
                self(self, parent->GetTransform(), true);
            return;
        }
        if (!entry || entry->backend != Backend::Gpu || !body || entry->rigidbody != body->GetECSHandle())
            return;
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
    if (!m_initialized || !domain.gpuRegistered)
        return true;
    auto dependencies = CollectCpuPoseReads(world, true);
    auto localDependencies = CollectCpuPoseReads(local, false);
    dependencies.insert(dependencies.end(), localDependencies.begin(), localDependencies.end());
    if (dependencies.empty())
        return true;
    if ((domain.stepPending && !domain.stepCompleted) || HasDomainControl(domain))
        return false;
    std::vector<Rigidbody *> bodies;
    bodies.reserve(dependencies.size());
    for (const auto &entry : dependencies)
        bodies.push_back(entry.second);
    return ServicePresentationPoseReads(domain, bodies, m_gpuScene->GetPublished(domain.owner));
}

bool PhysicsWorld::ServicePresentationPoseReads(PhysicsDomain &domain, const std::vector<Rigidbody *> &bodies,
                                                const std::shared_ptr<const JoltGpuScene::Publication> &publication)
{
    if (bodies.empty())
        return true;
    if (!domain.gpuRegistered || domain.retiring || !m_gpuWork->IsActive() || !m_gpuWork->IsAccepting() ||
        m_gpuWork->ServiceRemaining() <= 0 || !m_gpuWork->ResourceRemaining(FrameWorkResource::Cpu))
        return false;
    constexpr auto fields = PresentationPoseFields;
    // Pure Work never changes the published numerical owner, contacts or Fixed receipt.
    if (const auto job = domain.presentationPoseReads) {
        if (!AdvanceGpuJob(domain, false))
            return false;
        domain.presentationPoseReads.reset();
        if (job->error)
            std::rethrow_exception(job->error);
        if (job->publication != publication ||
            job->generation != domain.queryGeneration.load(std::memory_order_acquire))
            return false;
        MaterializeStateBatch(domain, job->batch, job->motion, job->shapes, true, fields);
    }
    std::vector<Rigidbody *> missing;
    std::unordered_set<Rigidbody *> selected;
    for (auto *body : bodies) {
        if (&DomainForComponent(*body) != &domain)
            throw std::logic_error("Presentation pose dependencies must belong to the selected World");
        if (selected.insert(body).second && !FindCachedMotionState(domain, body->GetBodyId(), fields))
            missing.push_back(body);
    }
    if (missing.empty())
        return true;
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
    work = ContinueJoltGpuControlWork(std::move(work), [job] { return JoltGpuControlWork{}; });
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
    if (const auto *cached = FindCachedMotionState(domain, id, fields))
        return *cached;
    if (domain.presentationPoseReadDepth)
        throw PhysicsPresentationPosePending{};
    if (ReadBodyBackend(domain, id) == Backend::Cpu) {
        RunCpuControl(domain);
        const auto state = domain.cpuScene->ReadMotionState(id, fields);
        if (static_cast<uint32_t>(fields) & static_cast<uint32_t>(PhysicsStateFields::All))
            domain.requestedMotion[id] = {domain.queryGeneration.load(std::memory_order_acquire), state, fields};
        return state;
    }
    PhysicsBodyMotionState result;
    std::shared_ptr<JoltGpuQueryResult> motion;
    std::exception_ptr error;
    RunControl(
        domain,
        [&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
            return PrepareJoltGpuQueryWork(
                [&]() -> JoltGpuControlWork {
                    const auto slot = publication->directory->FindBodySlot(id);
                    if (slot < publication->bodyBase || slot - publication->bodyBase >= publication->bodyCount)
                        throw JoltGpuQueryRequestError("Physics state requires an accepted live body");
                    const auto &identity = *publication->directory->actors[slot];
                    result.bodyId = id;
                    result.shapeRevision = identity.shapeRevision;
                    if ((static_cast<uint32_t>(fields) & static_cast<uint32_t>(PhysicsStateFields::All)) == 0)
                        return {};
                    auto input = Query(QueryGatherMotion, QueryClosest, 0, false);
                    input.meta.z = 0;
                    input.source.y = slot;
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
    RequireOwnerThread();
    auto &domain = DomainForBody(id);
    if (ReadBodyBackend(domain, id) == Backend::Cpu) {
        RunCpuControl(domain);
        const auto d = domain.cpuScene->GetAcceptedMass(id).inverse.Reciprocal();
        return {d.GetX(), d.GetY(), d.GetZ()};
    }
    const auto model = m_gpuScene->GetAcceptedBodyModel(id);
    const auto d = model.invInertiaDiagonal;
    if (!(d.x > 0 && d.y > 0 && d.z > 0))
        throw std::logic_error("Rigidbody has no accepted positive principal inertia");
    return glm::vec3(1.0f / d.x, 1.0f / d.y, 1.0f / d.z);
}

glm::quat PhysicsWorld::GetBodyPrincipalInertiaRotation(uint32_t id) const
{
    RequireOwnerThread();
    auto &domain = DomainForBody(id);
    if (ReadBodyBackend(domain, id) == Backend::Cpu) {
        RunCpuControl(domain);
        const auto q = domain.cpuScene->GetAcceptedMass(id).axes;
        return {q.GetW(), q.GetX(), q.GetY(), q.GetZ()};
    }
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
    if (m_ownerThreadId != std::thread::id{} && m_ownerThreadId != std::this_thread::get_id())
        return false;
    SealSpatialQuery(CurrentDomain());
    return true;
}
std::shared_ptr<PhysicsCpuRaycastEpoch> PhysicsWorld::CaptureCpuRaycastBatchQuery() const
{
    auto &domain = CurrentDomain();
    RequireHealthyStep(domain);
    if (domain.UsesGpu())
        return {};
    return std::make_shared<PhysicsCpuRaycastEpoch>(domain.cpuScene, domain.queryGeneration);
}
void PhysicsWorld::RaycastBatch(const float *origins, const float *directions, size_t count, float maximum,
                                RaycastHit *hits, uint8_t *mask, uint32_t layers, bool triggers, uint64_t *generation,
                                RaycastBatchProfile *profile) const
{
    if (count)
        SealSpatialQuery(CurrentDomain());
    const auto cpuEpoch = count != 0 ? CaptureCpuRaycastBatchQuery() : nullptr;
    RaycastBatchPublished(origins, directions, count, maximum, hits, mask, layers, triggers, generation, profile, false,
                          cpuEpoch);
}
void PhysicsWorld::RaycastBatchPublished(const float *origins, const float *directions, size_t count, float maximum,
                                         RaycastHit *hits, uint8_t *mask, uint32_t layers, bool triggers,
                                         uint64_t *generation, RaycastBatchProfile *profile, bool,
                                         const std::shared_ptr<PhysicsCpuRaycastEpoch> &cpuEpoch) const
{
    if (profile)
        *profile = {};
    if (cpuEpoch) {
        if (cpuEpoch->scene) {
            JoltCpuQueries(*cpuEpoch->scene)
                .RaycastBatch(origins, directions, count, maximum, hits, mask, {layers, triggers}, profile);
        } else {
            if (count && (!origins || !directions || !hits || !mask || count > UINT32_MAX))
                throw std::invalid_argument("Raycast batch requires valid arrays within the query index range");
            for (size_t i = 0; i < count; ++i) {
                auto input = Query(QueryRay, QueryClosest, layers, triggers);
                const auto offset = i * 3;
                QueryRayInput(input, {origins[offset], origins[offset + 1], origins[offset + 2]},
                              {directions[offset], directions[offset + 1], directions[offset + 2]}, maximum);
            }
            std::fill_n(hits, count, RaycastHit{});
            std::fill_n(mask, count, uint8_t{0});
        }
        if (generation)
            *generation = cpuEpoch->generation;
        return;
    }
    auto &domain = CurrentDomain();
    if (count == 0) {
        if (generation)
            *generation = domain.queryGeneration.load(std::memory_order_acquire);
        return;
    }
    RequireOwnerThread();
    RequireHealthyStep(domain);
    if (!origins || !directions || !hits || !mask || count > UINT32_MAX)
        throw std::invalid_argument("Raycast batch requires valid arrays within the query index range");

    const auto request = Query(QueryRay, QueryClosest, layers, triggers);
    if (domain.cpuScene) {
        const std::shared_lock queryRead(domain.cpuScene->QueryMutex());
        JoltCpuQueries(*domain.cpuScene)
            .RaycastBatch(origins, directions, count, maximum, hits, mask, {layers, triggers}, profile);
    } else if (const auto gpu = ReadGpuSpatialQueryPublication(domain)) {
        std::vector<QueryInput> requests;
        requests.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const auto offset = i * 3;
            auto input = request;
            QueryRayInput(input, {origins[offset], origins[offset + 1], origins[offset + 2]},
                          {directions[offset], directions[offset + 1], directions[offset + 2]}, maximum);
            requests.push_back(input);
        }
        const auto result = RunQuery(domain, gpu, requests);
        FrameWorkOwner::Scope materialize(*m_gpuWork);
        const auto &records = result->ReadHits();
        if (records.size() != count)
            throw std::runtime_error("GPU closest query did not produce one complete record per request");
        for (size_t i = 0; i < count; ++i) {
            hits[i] = DecodeQueryHit(*result, records[i]);
            mask[i] = static_cast<uint8_t>(records[i].meta.y != 0);
            if (profile)
                profile->publishedHits += mask[i];
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            auto input = request;
            const auto offset = i * 3;
            QueryRayInput(input, {origins[offset], origins[offset + 1], origins[offset + 2]},
                          {directions[offset], directions[offset + 1], directions[offset + 2]}, maximum);
        }
        std::fill_n(hits, count, RaycastHit{});
        std::fill_n(mask, count, uint8_t{0});
    }
    if (generation)
        *generation = domain.queryGeneration.load(std::memory_order_acquire);
}

PhysicsRigidbodyBatchSource PhysicsWorld::RaycastInto(const float *origins, const float *directions, size_t count,
                                                      float maximum,
                                                      const std::vector<JoltGpuQueryExportBinding> &exports,
                                                      uint32_t layers, bool triggers) const
{
    if ((count && (!origins || !directions)) || count > UINT32_MAX)
        throw std::invalid_argument("Raycast batch requires valid arrays within the query index range");
    std::vector<QueryInput> requests;
    requests.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        auto input = Query(QueryRay, QueryClosest, layers, triggers);
        const auto offset = i * 3;
        QueryRayInput(input, {origins[offset], origins[offset + 1], origins[offset + 2]},
                      {directions[offset], directions[offset + 1], directions[offset + 2]}, maximum);
        requests.push_back(input);
    }
    return ExportRaycastBatch(requests, exports);
}

PhysicsRigidbodyBatchSource PhysicsWorld::RaycastInto(const JoltGpuQueryVector3Input &origins,
                                                      const JoltGpuQueryVector3Input &directions, size_t count,
                                                      float maximum,
                                                      const std::vector<JoltGpuQueryExportBinding> &exports,
                                                      uint32_t layers, bool triggers) const
{
    if (count > UINT32_MAX || !std::isfinite(maximum) || maximum <= 0)
        throw std::invalid_argument("GPU raycast requires a valid count and positive finite maximum distance");
    // A single validated template describes the family. Numerical rays are
    // packed on the GPU, never uploaded as N author-side QueryInput records.
    auto input = Query(QueryRay, QueryClosest, layers, triggers);
    QueryRayInput(input, {}, {1, 0, 0}, maximum);
    return ExportRaycastBatch({input}, exports,
                              JoltGpuQueryRayInputs{origins, directions, static_cast<uint32_t>(count)});
}

PhysicsRigidbodyBatchSource PhysicsWorld::ExportRaycastBatch(const std::vector<QueryInput> &requests,
                                                             const std::vector<JoltGpuQueryExportBinding> &exports,
                                                             std::optional<JoltGpuQueryRayInputs> rays) const
{
    RequireOwnerThread();
    auto &domain = CurrentDomain();
    RequireHealthyStep(domain);
    if (!domain.UsesGpu())
        throw std::invalid_argument("Device raycast output requires a GPU physics scene");
    for (const auto &binding : exports)
        if (binding.field < JoltGpuQueryExportField::HitPoint || binding.field > JoltGpuQueryExportField::HitValid)
            throw std::invalid_argument("Device raycast output requires hit projection fields");
    // The same publication represents an empty scene and produces one miss per ray.
    RequireGpu(domain);
    SealSpatialQuery(domain);
    const auto publication = ReadGpuQueryPublication(domain);
    const auto generation = domain.queryGeneration.load(std::memory_order_acquire);
    std::exception_ptr error;
    std::shared_ptr<JoltGpuQueryResult> result;
    ObserveSpatialGpuQuery(domain, publication, [&](const auto &accepted) {
        return PrepareJoltGpuQueryWork(
            [&] {
                auto query = m_gpuQueries->Prepare(accepted, requests, {}, generation, JoltGpuQueryReadback::Device, 32,
                                                   {}, rays);
                result = query.result;
                return ContinueJoltGpuControlWork(std::move(query.work), [this, result = query.result, &exports] {
                    if (result->GetSummary().counts.y != QuerySuccess)
                        return JoltGpuControlWork{};
                    return m_gpuQueries->PrepareExport({result}, exports);
                });
            },
            error);
    });
    if (error)
        std::rethrow_exception(error);
    (void)result->GetRecordCount(); // Deliver a numerical error to the caller without failing the world.
    return {publication->worldId, publication->simulationEpoch, publication->fixedStepIndex, generation};
}

bool PhysicsWorld::Raycast(const glm::vec3 &origin, const glm::vec3 &direction, float maximum, RaycastHit &hit,
                           uint32_t layers, bool triggers) const
{
    uint8_t mask = 0;
    const float origins[]{origin.x, origin.y, origin.z}, directions[]{direction.x, direction.y, direction.z};
    RaycastBatch(origins, directions, 1, maximum, &hit, &mask, layers, triggers);
    if (mask) {
        hit.collider = hit.target.GetCollider();
        hit.gameObject = hit.target.GetGameObject();
    }
    return mask != 0;
}

Rigidbody *PhysicsWorld::FindRigidbodyByBodyId(uint32_t id) const
{
    const auto *registration = FindBodyRegistration(id);
    const auto &store = PhysicsECSStore::Instance();
    if (!registration || !store.IsValid(registration->rigidbody))
        return nullptr;
    const auto &body = store.GetRigidbody(registration->rigidbody);
    return body.actorHandle == registration->actor ? body.owner : nullptr;
}
void PhysicsWorld::EnsureSceneBodiesRegistered(Scene *scene)
{
    auto &domain = *scene->GetRuntimeWorld()->physics;
    RequireOwnerThread();
    FlushAuthoring(domain);
}

std::unique_ptr<physics::warp::WarpSceneSetup> PhysicsWorld::CaptureWarpSceneSetup(RuntimeWorld &world) const
{
    using namespace physics::warp;
    RequireOwnerThread();
    auto &domain = *world.physics;
    if (world.simulation.fixedStepIndex != 0 || domain.stepPending)
        throw std::logic_error("Warp scene setup must be captured before fixed simulation");

    FlushAuthoring(domain);
    const auto &store = PhysicsECSStore::Instance();
    if (store.HasPendingBodyCreations(world))
        return nullptr;

    auto setup = std::make_unique<WarpSceneSetup>();
    auto &registration = setup->world;
    registration.worldId = world.id;
    registration.simulationEpoch = world.simulation.simulationEpoch;
    std::vector<uint32_t> ids(domain.bodyIds.begin(), domain.bodyIds.end());
    std::sort(ids.begin(), ids.end());

    TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    std::vector<WarpSceneActor> actors;
    std::vector<const Joint *> activeJoints;
    std::unordered_map<uint32_t, uint32_t> rowById;
    actors.reserve(ids.size());
    registration.bodies.reserve(ids.size());
    rowById.reserve(ids.size());
    for (const auto id : ids) {
        const auto &binding = m_bodyActors.at(id);
        const auto &actor = store.GetActor(binding.actor);
        if (!actor.bodyInBroadphase)
            continue;
        auto author = PhysicsActorAuthoring::Capture(actor);
        if (!author)
            return nullptr;
        const auto bodyTarget = author->body ? author->body->target : PhysicsTargetReference{};
        auto prepared = PrepareWarpSceneActor(std::move(*author));
        rowById.emplace(id, static_cast<uint32_t>(actors.size()));
        registration.bodies.push_back(
            {id, binding.stagedShapeRevision, binding.actor, std::move(prepared.surfaces), bodyTarget});
        actors.push_back(std::move(prepared.actor));
        for (const auto *joint : actor.owner->GetComponents<Joint>())
            if (joint->GetPhysicsConstraintId() != 0 && joint->m_boundBodyId == id)
                activeJoints.push_back(joint);
    }

    std::vector<WarpSceneJoint> joints;
    joints.reserve(activeJoints.size());
    for (const auto *joint : activeJoints) {
        const auto *transform = joint->GetGameObject()->GetTransform();
        const auto anchor =
            glm::conjugate(transform->GetWorldRotation()) * transform->TransformVector(joint->GetAnchor());
        const auto connected = joint->m_boundConnectedBodyId;
        auto &model = joints.emplace_back(WarpSceneJoint{rowById.at(joint->m_boundBodyId),
                                                         connected == UINT32_MAX ? NoModelIndex : rowById.at(connected),
                                                         anchor,
                                                         joint->GetEnableCollision(),
                                                         {}});
        if (const auto *hinge = dynamic_cast<const HingeJoint *>(joint)) {
            model.settings =
                WarpSceneHinge{hinge->GetAxis(),
                               hinge->GetUseLimits(),
                               glm::radians(hinge->GetMinimumAngle()),
                               glm::radians(hinge->GetMaximumAngle()),
                               {hinge->GetMotorEnabled(), hinge->GetClampedMotorTargetRadians(),
                                hinge->GetMotorStiffness(), hinge->GetMotorDamping(), hinge->GetMotorMaxTorque(),
                                hinge->GetJointDamping(), hinge->GetFrictionTorque(), hinge->GetArmature()}};
        } else if (const auto *slider = dynamic_cast<const SliderJoint *>(joint)) {
            model.settings = WarpSceneSlider{slider->GetAxis(), slider->GetUseLimits(), slider->GetMinimumDistance(),
                                             slider->GetMaximumDistance()};
        } else if (const auto *distance = dynamic_cast<const DistanceJoint *>(joint)) {
            auto connectedAnchor = distance->GetConnectedAnchor();
            if (connected != UINT32_MAX) {
                const auto *connectedTransform = store.GetActor(m_bodyActors.at(connected).actor).owner->GetTransform();
                connectedAnchor = glm::conjugate(connectedTransform->GetWorldRotation()) *
                                  connectedTransform->TransformVector(connectedAnchor);
            }
            model.settings = WarpSceneDistance{connectedAnchor,
                                               distance->GetAutoDistance(),
                                               distance->GetMinimumDistance(),
                                               distance->GetMaximumDistance(),
                                               distance->GetStiffness(),
                                               distance->GetDamping()};
        } else if (const auto *swing = dynamic_cast<const SwingTwistJoint *>(joint)) {
            model.settings = swing->m_settings;
        } else if (!dynamic_cast<const FixedJoint *>(joint)) {
            throw std::invalid_argument(std::string("Warp scene setup does not support ") + joint->GetTypeName());
        }
    }

    setup->model = BuildWarpSceneModel(actors, joints, *CaptureColliderPairs(domain, true));
    for (uint32_t index = 0; index < activeJoints.size(); ++index)
        if (setup->model.jointBindings[index].row != NoModelIndex)
            registration.joints.push_back({activeJoints[index]->GetPhysicsConstraintId(), index});
    return setup;
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
    RefreshBodyLayers(id);
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
    if (!domain.gpuRegistered || domain.retiring)
        return;
    const auto pairs = CaptureColliderPairs(domain);
    if (domain.queuedColliderPairs == pairs)
        return;
    m_gpuScene->SetColliderPairs(domain.owner, pairs);
    domain.queuedColliderPairs = pairs;
}
void PhysicsWorld::InvalidateColliderPairSnapshot(PhysicsDomain &domain) const
{
    domain.colliderPairSnapshot.reset();
    if (domain.cpuControl) {
        const WorldExecutionScope ownerScope(&domain.owner);
        SceneManager::Instance().RequestPausedAuthorPresentation();
    }
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
        InvalidateColliderPairSnapshot(domain);
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
        InvalidateColliderPairSnapshot(domain);
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
        InvalidateColliderPairSnapshot(domain);
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
        InvalidateColliderPairSnapshot(domain);
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

PhysicsTargetSource PhysicsWorld::QueueKinematicBodyCommand(uint32_t id, const BodyCommand &command,
                                                            const PhysicsTargetAuthor *author)
{
    RequireOwnerThread();
    auto &domain = DomainForBody(id);
    const auto backend = m_bodyActors.at(id).backend;
    if (backend == Backend::StaticBoundary)
        throw std::logic_error("Kinematic movement requires an admitted Rigidbody");
    (void)KinematicBody(ActorForBody(id));
    if (backend == Backend::Cpu) {
        JoltCpuControl::KinematicMove move{std::nullopt, std::nullopt, command.value.w, command.rootCOM.w};
        if (command.meta.y != CommandMoveRotation)
            move.position = CpuPosition(command.value);
        if (command.meta.y != CommandMovePosition)
            move.rotation = CpuRotation(command.auxiliary);
        return domain.cpuControl->QueueMove(id, std::move(move), author);
    }
    RequireGpu(domain);
    return m_gpuScene->QueueCommand(id, command, author);
}
void PhysicsWorld::MoveBodyKinematic(uint32_t id, const glm::vec3 &position, const glm::quat &rotation, float dt,
                                     float maxSpeed)
{
    (void)QueueKinematicBodyCommand(id, MoveCommand(CommandMovePose, position, rotation, dt, maxSpeed));
}
void PhysicsWorld::MoveBodyKinematicPosition(uint32_t id, const glm::vec3 &position, float dt)
{
    (void)QueueKinematicBodyCommand(id, MoveCommand(CommandMovePosition, position, glm::quat(1, 0, 0, 0), dt));
}
void PhysicsWorld::MoveBodyKinematicRotation(uint32_t id, const glm::quat &rotation, float dt)
{
    (void)QueueKinematicBodyCommand(id, MoveCommand(CommandMoveRotation, glm::vec3(0), rotation, dt));
}
PhysicsTargetSource PhysicsWorld::MoveBodyKinematicAuthored(uint32_t id, const glm::vec3 &position,
                                                            const glm::quat &rotation, uint32_t channels, float dt,
                                                            float maxSpeed,
                                                            const std::array<uint64_t, 2> &authoredRevision)
{
    RequireOwnerThread();
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
    return QueueKinematicBodyCommand(id, MoveCommand(kind, position, rotation, dt, maxSpeed), &author);
}
void PhysicsWorld::RecordStaticPose(PhysicsDomain &domain, uint32_t id, const glm::vec3 *position,
                                    const glm::quat *rotation, bool cancelMove)
{
    const auto &entry = m_bodyActors.at(id);
    if (!entry.staticModel)
        return;
    auto &model = *entry.staticModel;
    if (position) {
        model.position = *position;
        ++model.positionRevision;
    }
    if (rotation) {
        model.rotation = *rotation;
        ++model.rotationRevision;
    }
    if (cancelMove)
        domain.pendingStaticMoves.erase(id);
}

void PhysicsWorld::QueueStaticFixedMoves(PhysicsDomain &domain)
{
    // The authored pose and temporary drive belong to the same fixed step.
    for (auto &[id, move] : domain.pendingStaticMoves) {
        const auto &entry = m_bodyActors.at(id);
        if (move.queued || !entry.stagedOn || !entry.staticResident || !entry.staticModel ||
            entry.staticModel != move.model.lock())
            continue;
        if (entry.HasCpuBody()) {
            domain.cpuControl->QueueTemporaryStaticMove(id, CpuPosition(Vector(move.position)),
                                                        CpuRotation(Rotation(move.rotation)), domain.admittedDeltaTime,
                                                        kMaxTransformDriveSpeed);
        } else {
            auto command = MoveCommand(CommandMovePose, move.position, move.rotation, domain.admittedDeltaTime,
                                       kMaxTransformDriveSpeed);
            command.meta.z = CommandTemporaryStatic;
            m_gpuScene->QueueCommand(id, command);
        }
        move.queued = true;
    }
}

void PhysicsWorld::SealStaticFixedMoves(PhysicsDomain &domain)
{
    domain.admittedStaticMoves.reserve(domain.pendingStaticMoves.size());
    for (auto it = domain.pendingStaticMoves.begin(); it != domain.pendingStaticMoves.end();) {
        const auto id = it->first;
        auto &move = it->second;
        const auto model = move.model.lock();
        if (!model || m_bodyActors.at(id).staticModel != model) {
            it = domain.pendingStaticMoves.erase(it);
            continue;
        }
        if (!move.queued) {
            ++it;
            continue;
        }
        move.positionRevision = model->positionRevision;
        move.rotationRevision = model->rotationRevision;
        domain.admittedStaticMoves.emplace_back(id, move);
        it = domain.pendingStaticMoves.erase(it);
    }
}

void PhysicsWorld::AcceptStaticFixedMoves(PhysicsDomain &domain)
{
    for (const auto &[id, move] : domain.admittedStaticMoves) {
        const auto found = m_bodyActors.find(id);
        const auto model = move.model.lock();
        if (found == m_bodyActors.end() || !model || found->second.staticModel != model)
            continue;
        // Full-duration, unconstrained static drives reach this ordered
        // endpoint. A newer queued teleport wins only the channels it wrote.
        if (model->positionRevision == move.positionRevision)
            model->position = move.position;
        if (model->rotationRevision == move.rotationRevision)
            model->rotation = move.rotation;
    }
    domain.admittedStaticMoves.clear();
}

void PhysicsWorld::MoveStaticBodyWithVelocity(uint32_t id, const glm::vec3 &position, const glm::quat &rotation,
                                              float dt)
{
    RequireOwnerThread();
    auto &domain = DomainForBody(id);
    const auto &actor = ActorForBody(id);
    if (actor.rigidbody && actor.rigidbody->IsEnabled() && !NeedsBodyBindingChange(actor))
        throw std::logic_error("Static movement requires a collider-only physics actor");
    auto command = MoveCommand(CommandMovePose, position, rotation, dt, kMaxTransformDriveSpeed);
    const auto &entry = m_bodyActors.at(id);
    if (entry.staticModel) {
        domain.pendingStaticMoves.insert_or_assign(id,
                                                   PhysicsDomain::StaticMove{position, rotation, entry.staticModel});
        return;
    }
    if (entry.HasCpuBody())
        domain.cpuControl->QueueTemporaryStaticMove(id, CpuPosition(command.value), CpuRotation(command.auxiliary),
                                                    command.value.w, command.rootCOM.w);
    if (!entry.HasGpuBody())
        return;
    RequireGpu(domain);
    command.meta.z = CommandTemporaryStatic;
    m_gpuScene->QueueCommand(id, command);
}
bool PhysicsWorld::IsBodySleeping(uint32_t id) const
{
    RequireOwnerThread();
    auto &domain = DomainForBody(id);
    const auto backend = m_bodyActors.at(id).backend;
    if (backend == Backend::StaticBoundary)
        return true;
    bool sleeping = false;
    if (backend == Backend::Cpu) {
        RunCpuControl(domain);
        (void)domain.cpuScene->ReadMotionState(id, PhysicsStateFields::None, &sleeping);
        return sleeping;
    }
    RunControl(domain, [&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        const auto slot = publication->directory->FindBodySlot(id);
        if (slot < publication->bodyBase || slot - publication->bodyBase >= publication->bodyCount)
            throw std::logic_error("Sleep state requires this publication's live body identity");
        return m_gpuScene->PrepareSleepQuery(publication, {slot - publication->bodyBase}, [&](auto values) {
            if (values.size() != 1u || values[0].x > 1u || values[0].z != 1u || values[0].w)
                throw std::runtime_error("GPU sleep publication failed its initialized activity contract");
            sleeping = values[0].x == 0u;
        });
    });
    return sleeping;
}
void PhysicsWorld::ActivateBody(uint32_t id)
{
    RequireOwnerThread();
    auto &domain = DomainForBody(id);
    const auto backend = m_bodyActors.at(id).backend;
    if (backend == Backend::StaticBoundary)
        return;
    if (backend == Backend::Cpu) {
        domain.cpuControl->Queue(id, JoltCpuControl::Awake{true});
        return;
    }
    RequireGpu(domain);
    BodyCommand command{};
    command.meta.y = CommandWake;
    m_gpuScene->QueueCommand(id, command);
}
void PhysicsWorld::DeactivateBody(uint32_t id)
{
    RequireOwnerThread();
    auto &domain = DomainForBody(id);
    const auto backend = m_bodyActors.at(id).backend;
    if (backend == Backend::StaticBoundary)
        return;
    if (backend == Backend::Cpu) {
        domain.cpuControl->Queue(id, JoltCpuControl::Awake{false});
        return;
    }
    RequireGpu(domain);
    BodyCommand command{};
    command.meta.y = CommandSleep;
    m_gpuScene->QueueCommand(id, command);
}
std::optional<PhysicsPenetrationResult> PhysicsWorld::ComputePenetration(const Collider &a, const glm::vec3 &positionA,
                                                                         const glm::quat &rotationA, const Collider &b,
                                                                         const glm::vec3 &positionB,
                                                                         const glm::quat &rotationB) const
{
    auto &domain = CurrentDomain();
    RequireOwnerThread();
    if (&DomainForComponent(a) != &domain || &DomainForComponent(b) != &domain)
        throw std::invalid_argument("Collider penetration requires both colliders in the queried physics world");
    if (!domain.UsesGpu())
        return JoltCpuQueries::ComputePenetration(a, positionA, rotationA, b, positionB, rotationB);
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
    RunCpuControl(domain);
    const auto id = ColliderBodyIdentity(collider);
    hit = {};
    if (!FindBodyRegistration(id))
        return false;
    if (ReadBodyBackend(domain, id) == Backend::Cpu) {
        if (!domain.cpuScene->HasBody(id))
            RunCpuControl(domain);
        const JoltCpuQueries::ColliderIdentity target{collider.GetComponentID(), collider.GetLifetimeGeneration()};
        return JoltCpuQueries(*domain.cpuScene).Raycast(origin, direction, maximum, hit, {UINT32_MAX, true, target});
    }
    RunControl(domain, {}, true);
    TargetCollider(request, collider);
    const auto result = RunQuery(domain, ReadGpuQueryPublication(domain), {request});
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
    RunCpuControl(domain);
    const auto id = ColliderBodyIdentity(collider);
    if (!FindBodyRegistration(id))
        throw std::runtime_error("Active Collider has no accepted geometry for a closest-point query");
    if (ReadBodyBackend(domain, id) == Backend::Cpu) {
        if (!domain.cpuScene->HasBody(id))
            RunCpuControl(domain);
        const auto closest =
            JoltCpuQueries(*domain.cpuScene)
                .ClosestPoint(id, {collider.GetComponentID(), collider.GetLifetimeGeneration()}, point);
        if (!closest)
            throw std::runtime_error("Active Collider has no accepted geometry for a closest-point query");
        return *closest;
    }
    auto request = Query(QueryClosestPoint, QueryClosest);
    request.positionMaximum = Vector(point);
    TargetCollider(request, collider);
    RunControl(domain, {}, true);
    const auto result = RunQuery(domain, ReadGpuQueryPublication(domain), {request});
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
    SealSpatialQuery(domain);
    if (domain.cpuScene)
        return JoltCpuQueries(*domain.cpuScene).RaycastAll(origin, direction, maximum, {layers, triggers});
    const auto gpu = ReadGpuSpatialQueryPublication(domain);
    std::vector<RaycastHit> hits;
    if (!gpu)
        return hits;
    const auto result = RunQuery(domain, gpu, {request});
    {
        FrameWorkOwner::Scope materialize(*m_gpuWork);
        const auto &records = result->ReadHits();
        hits.reserve(records.size());
        for (const auto &record : records)
            hits.push_back(DecodeQueryHit(*result, record));
    }
    return hits;
}

std::vector<Collider *> PhysicsWorld::OverlapSphere(const glm::vec3 &center, float radius, uint32_t layers,
                                                    bool triggers) const
{
    auto &domain = CurrentDomain();
    auto request = Query(QueryOverlap, QueryUniqueCollider, layers, triggers);
    QuerySphereInput(request, center, radius);
    return QueryColliders(domain, request, JoltCpuQueries::Sphere{center, radius});
}
std::vector<Collider *> PhysicsWorld::OverlapBox(const glm::vec3 &center, const glm::vec3 &halfExtents,
                                                 const glm::quat &rotation, uint32_t layers, bool triggers) const
{
    auto &domain = CurrentDomain();
    auto request = Query(QueryOverlap, QueryUniqueCollider, layers, triggers);
    QueryBoxInput(request, center, halfExtents, rotation);
    return QueryColliders(domain, request, JoltCpuQueries::Box{center, halfExtents, rotation});
}
std::vector<Collider *> PhysicsWorld::OverlapCapsule(const glm::vec3 &a, const glm::vec3 &b, float radius,
                                                     uint32_t layers, bool triggers) const
{
    auto &domain = CurrentDomain();
    auto request = Query(QueryOverlap, QueryUniqueCollider, layers, triggers);
    QueryCapsuleInput(request, a, b, radius);
    return QueryColliders(domain, request, JoltCpuQueries::Capsule{a, b, radius});
}
std::vector<Rigidbody *> PhysicsWorld::QueryRigidbodyCandidatesInBounds(
    PhysicsDomain &domain, const std::shared_ptr<const JoltGpuScene::Publication> &gpu, const glm::vec3 &minimum,
    const glm::vec3 &maximum, uint32_t layers, bool triggers) const
{
    (void)Vector(minimum);
    (void)Vector(maximum);
    if (glm::any(glm::greaterThan(minimum, maximum)))
        throw std::invalid_argument("Physics query bounds require minimum <= maximum");
    if (domain.cpuScene) {
        auto bodies = JoltCpuQueries(*domain.cpuScene).QueryRigidbodiesInBounds(minimum, maximum, {layers, triggers});
        std::sort(bodies.begin(), bodies.end(),
                  [](const auto *a, const auto *b) { return a->GetComponentID() < b->GetComponentID(); });
        return bodies;
    }
    std::vector<Rigidbody *> bodies;
    if (gpu) {
        auto request = Query(QueryBodyBounds, QueryUniqueBody, layers, triggers);
        request.positionMaximum = Vector(glm::vec3((glm::dvec3(minimum) + glm::dvec3(maximum)) * 0.5));
        request.direction = Vector(glm::vec3((glm::dvec3(maximum) - glm::dvec3(minimum)) * 0.5));
        const auto result = RunQuery(domain, gpu, {request});
        FrameWorkOwner::Scope materialize(*m_gpuWork);
        auto &store = PhysicsECSStore::Instance();
        for (const auto &record : result->ReadHits()) {
            const auto &identity = *gpu->directory->actors.at(record.meta.x);
            if (store.IsValid(identity.rigidbody))
                bodies.push_back(store.GetRigidbody(identity.rigidbody).owner);
        }
    }
    return bodies;
}

std::vector<Rigidbody *> PhysicsWorld::QueryRigidbodiesInBounds(const glm::vec3 &minimum, const glm::vec3 &maximum,
                                                                uint32_t layers, bool triggers) const
{
    auto &domain = CurrentDomain();
    SealSpatialQuery(domain);
    return QueryRigidbodyCandidatesInBounds(domain, ReadGpuSpatialQueryPublication(domain), minimum, maximum, layers,
                                            triggers);
}
bool PhysicsWorld::SphereCast(const glm::vec3 &center, float radius, const glm::vec3 &direction, float maximum,
                              RaycastHit &hit, uint32_t layers, bool triggers) const
{
    auto &domain = CurrentDomain();
    auto request = Query(QueryShapeCast, QueryClosest, layers, triggers);
    QuerySphereInput(request, center, radius);
    QueryRayInput(request, center, direction, maximum);
    return QueryClosestCast(domain, request, JoltCpuQueries::Sphere{center, radius}, hit);
}
bool PhysicsWorld::BoxCast(const glm::vec3 &center, const glm::vec3 &halfExtents, const glm::vec3 &direction,
                           const glm::quat &rotation, float maximum, RaycastHit &hit, uint32_t layers,
                           bool triggers) const
{
    auto &domain = CurrentDomain();
    auto request = Query(QueryShapeCast, QueryClosest, layers, triggers);
    QueryBoxInput(request, center, halfExtents, rotation);
    QueryRayInput(request, center, direction, maximum);
    return QueryClosestCast(domain, request, JoltCpuQueries::Box{center, halfExtents, rotation}, hit);
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
    hit = {};
    if (!box.GetGameObject() || distance == 0)
        return false;
    if (distance > std::numeric_limits<float>::max())
        throw std::invalid_argument("BoxCollider sweep displacement exceeds device coordinates");
    SealSpatialQuery(domain);
    if (domain.cpuScene)
        return JoltCpuQueries(*domain.cpuScene).SweepBox(box, position, rotation, displacement, hit);
    const auto gpu = ReadGpuSpatialQueryPublication(domain);
    auto *object = box.GetGameObject();
    if (!object)
        return false;
    const auto layers = TagLayerManager::Instance().GetLayerCollisionMask(object->GetLayer());
    auto request = Query(QueryShapeCast, QueryClosest, layers, false);
    if (gpu) {
        const auto primitive = CookColliderQueryPrimitive(box, position, rotation);
        if (!primitive)
            return false;
        const auto colliderId = box.GetComponentID(), objectId = object->GetID();
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
        const auto result =
            RunQuery(domain, gpu, {request}, colliderId, box.GetLifetimeGeneration(), ColliderBodyIdentity(box));
        FrameWorkOwner::Scope materialize(*m_gpuWork);
        hit = DecodeQueryHit(*result, result->ReadHits().front());
    }
    return hit.bodyId != UINT32_MAX;
}

bool PhysicsWorld::CapsuleCast(const glm::vec3 &a, const glm::vec3 &b, float radius, const glm::vec3 &direction,
                               float maximum, RaycastHit &hit, uint32_t layers, bool triggers) const
{
    auto &domain = CurrentDomain();
    auto request = Query(QueryShapeCast, QueryClosest, layers, triggers);
    QueryCapsuleInput(request, a, b, radius);
    QueryRayInput(request, Vector(request.positionMaximum), direction, maximum);
    return QueryClosestCast(domain, request, JoltCpuQueries::Capsule{a, b, radius}, hit);
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
        // This adapter describes the accepted GPU operand without reading its numerical state.
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
        return PrepareJoltGpuQueryWork(
            [&] {
                const auto slot = publication->directory->FindBodySlot(id);
                if (slot < publication->bodyBase || slot - publication->bodyBase >= publication->bodyCount)
                    throw JoltGpuQueryRequestError("Shape gathering requires an accepted body identity");
                auto input = Query(QueryGatherShapes, QueryAll, UINT32_MAX, triggers);
                input.meta.z = QueryAllShapeTypes;
                input.source.y = slot;
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
    RequireOwnerThread();
    auto &domain = DomainForBody(id);
    if (ReadBodyBackend(domain, id) == Backend::Cpu) {
        RunCpuControl(domain);
        return JoltCpuQueries(*domain.cpuScene).GatherBodyShapes(id, triggers);
    }
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
    RequireOwnerThread();
    auto &domain = DomainForBody(id);
    if (ReadBodyBackend(domain, id) == Backend::Cpu) {
        RunCpuControl(domain);
        return JoltCpuQueries(*domain.cpuScene).GatherBodyBoxes(id, triggers);
    }
    std::vector<PhysicsBoxShapeState> result;
    for (const auto &state : GetBodyShapeStates(id, triggers))
        if (state.geometry->type == PhysicsShapeType::Box)
            result.push_back({state.memberIndex, state.center, state.rotation,
                              glm::vec3(state.geometry->dimensions) * .5f, state.friction, state.bounciness});
    return result;
}

std::shared_ptr<particle::GpuParticleCollisionDevicePayload> PhysicsWorld::PrepareParticleCollisionScene(
    Scene &scene, const std::shared_ptr<const particle::GpuParticleCollisionDevicePayload> &previous, uint64_t revision,
    const std::shared_ptr<const void> &consumer) const
{
    RequireOwnerThread();
    auto &domain = *scene.GetRuntimeWorld()->physics;
    if (!consumer)
        throw std::invalid_argument("GPU particle collision export requires a live consumer");
    domain.particleCollisionConsumer = consumer;
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
    const auto prepareCollisionExport = [&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        return PrepareJoltGpuQueryWork(
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
                return ContinueJoltGpuControlWork(
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
    };
    RunControl(domain, prepareCollisionExport, false, true);
    if (error)
        std::rethrow_exception(error);
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    (void)query.result->GetRecordCount();
    // A recognized read-only numerical failure is reported outside the world's
    // source, so a failed export does not poison subsequent simulation steps.
    exported->Validate();
    exported->payload->simulationEpoch = query.result->publication->simulationEpoch;
    exported->payload->rows = std::move(keys);
    return std::move(exported->payload);
}

std::vector<glm::vec3> PhysicsShapeGeometry::GetTriangles() const
{
    if (triangle) {
        // A zero-margin triangle hull exposes both sides of the same face.
        const auto &[a, b, c] = *triangle;
        return {a, b, c, a, c, b};
    }
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
    const WorldExecutionScope ownerScope(&domain.owner);
    AdmissionScope admission(*m_gpuScene, domain.owner);
    // The prepared consumer already owns its immutable numerical inputs.
    while (!m_gpuScene->TryBeginWork(domain.owner, prepared, false))
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
    const size_t required = domain.recreateActors.size() + domain.bodyIds.size() + 1;
    if (required > domain.recreateActors.capacity())
        domain.recreateActors.reserve(std::max(required, 2 * domain.recreateActors.capacity()));
    if (!m_bodyActors.emplace(bodyId, BodyRegistration{actor, backend}).second)
        throw std::logic_error("Duplicate physics body identity");
    try {
        domain.bodyIds.insert(bodyId);
        if (backend == Backend::StaticBoundary) {
            domain.pendingStatic.insert(bodyId);
        }
    } catch (...) {
        m_bodyActors.erase(bodyId);
        domain.bodyIds.erase(bodyId);
        domain.pendingStatic.erase(bodyId);
        throw;
    }
    domain.gpuBodyOwners += backend == Backend::Gpu;
}
void PhysicsWorld::UnregisterBodyIdentity(PhysicsDomain &domain, uint32_t bodyId) noexcept
{
    domain.gpuBodyOwners -= m_bodyActors.at(bodyId).backend == Backend::Gpu;
    m_bodyActors.erase(bodyId);
    domain.bodyIds.erase(bodyId);
    domain.pendingStatic.erase(bodyId);
    domain.pendingStaticMoves.erase(bodyId);
}
void PhysicsWorld::ChangeBodyBackend(PhysicsDomain &domain, BodyRegistration &entry, Backend backend) noexcept
{
    domain.gpuBodyOwners -= entry.backend == Backend::Gpu;
    domain.gpuBodyOwners += backend == Backend::Gpu;
    entry.backend = backend;
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
PhysicsWorld::Backend PhysicsWorld::ReadBodyBackend(PhysicsDomain &domain, uint32_t bodyId) const
{
    const auto *registration = FindBodyRegistration(bodyId);
    if (!registration)
        throw std::invalid_argument("Physics body identity is invalid or retired");

    if (domain.stepPending && !domain.stepCompleted)
        const_cast<PhysicsWorld *>(this)->CompleteDomainStep(domain);
    if (IsFixedPublicationHeld(domain)) {
        if (domain.UsesGpu()) {
            const auto gpu = domain.gpuContacts.GetPublication();
            if (gpu) {
                const auto slot = gpu->directory->FindBodySlot(bodyId);
                if (slot >= gpu->bodyBase && slot - gpu->bodyBase < gpu->bodyCount)
                    return Backend::Gpu;
            }
        } else if (domain.cpuScene && domain.cpuScene->HasBody(bodyId)) {
            return Backend::Cpu;
        }
        throw std::logic_error("Physics body has no accepted representation in this Fixed publication");
    }

    if (registration->backend != Backend::StaticBoundary)
        return registration->backend;
    while (!const_cast<PhysicsWorld *>(this)->EnsureStaticBoundary(domain, bodyId)) {
        if (!domain.waitingForSlot)
            throw std::logic_error("Static body geometry is still being prepared");
        DrainQueryGpuCapacity(domain);
    }
    return domain.UsesGpu() ? Backend::Gpu : Backend::Cpu;
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
    // Accepted continuations may still own pending jobs; WorldManager retains world.
}
bool PhysicsWorld::CanRetireDomain(const RuntimeWorld &world) const noexcept
{
    return !m_gpuScene || m_gpuScene->CanRetire(world);
}
void PhysicsWorld::RetireDomain(PhysicsDomain &domain) noexcept
{
    RetireRagdolls(domain);
    m_domains.erase(&domain);
}
} // namespace infernux
