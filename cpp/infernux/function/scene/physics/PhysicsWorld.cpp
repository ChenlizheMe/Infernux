#include "PhysicsWorld.h"
#include <Jolt/Jolt.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
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
#include "../MeshCollider.h"
#include "../Ragdoll.h"
#include "../Rigidbody.h"
#include "../RuntimeWorld.h"
#include "../Scene.h"
#include "../SceneManager.h"
#include "../SixDOFJoint.h"
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
#include "PhysicsBodyCommand.h"
#include "PhysicsContactTypes.h"
#include "PhysicsDomain.h"
#include "PhysicsGpuRenderBindings.h"
#include "PhysicsGpuRenderPoses.h"
#include "PhysicsLayers.h"
#include "PhysicsMass.h"
#include "PhysicsParticipantBodies.h"
#include "PhysicsShapeAuthoring.h"
#include "PhysicsShapeLeaves.h"
#include "inxphygpu/InxPhyGpuParticleExport.h"
#include "inxphygpu/InxPhyGpuProgramSources.h"
#include "inxphygpu/InxPhyGpuSceneRuntime.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <core/config/EngineConfig.h>
#include <core/log/InxLog.h>
#include <core/threading/FrameWorkOwner.h>
#include <deque>
#include <exception>
#include <function/renderer/particle/ParticleGpuCollisionScene.h>
#include <function/renderer/rhi/RhiComputeHost.h>
#include <function/renderer/rhi/RhiComputeKernel.h>
#include <function/resources/PhysicMaterial/PhysicMaterial.h>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>

namespace infernux
{
struct PhysicsSceneRayPlan
{
    std::shared_ptr<const JoltGpuScene::PublicationDirectory> directory;
    uint32_t bodyBase = 0, bodyCount = 0;
    physics::inxphygpu::InxPhyGpuPublishedRayQuery query;
    std::vector<RaycastHit> surfaces; // Geometry rows, resolved only when a read completes.
    std::map<std::pair<uint64_t, uint64_t>, std::pair<uint32_t, uint32_t>> colliderGeometry;
    std::optional<std::vector<physics::inxphygpu::InxPhyGpuParticleSurface>> particleSurfaces;

    [[nodiscard]] bool Matches(const JoltGpuScene::Publication &publication) const
    {
        return directory == publication.directory && bodyBase == publication.bodyBase &&
               bodyCount == publication.bodyCount;
    }
};

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

PhysicsGpuPoseSource ScenePoseSource(const JoltGpuScene::Publication &p)
{
    return {p.bodyStates, 0, p.bodyCount, sizeof(JPH::GPU::BodyState)};
}

PhysicsGpuBodyPoseBinding ScenePoseBinding(const JoltGpuScene::Publication &p,
                                           const JoltGpuScene::PublishedActor &actor)
{
    const auto &center = actor.rootCenterOfMass;
    return {actor.bodyId, actor.slot - p.bodyBase, {center.x, center.y, center.z}};
}

PhysicsGpuMotionBinding SceneMotionBinding(const JoltGpuScene::Publication &publication,
                                           const JoltGpuScene::PublishedActor &actor)
{
    PhysicsGpuMotionBinding result{ScenePoseBinding(publication, actor), actor.shapeRevision};
    if (actor.acceptedMotionType == JPH::GPU::MotionDynamic) {
        const auto &model = actor.physicalModel;
        result.inverseMass = model.mass.x;
        result.inverseInertia = Vector(model.invInertiaDiagonal);
        result.inertiaFrame = {model.inertiaRotation.w, model.inertiaRotation.x, model.inertiaRotation.y,
                               model.inertiaRotation.z};
        result.frozenTranslation = static_cast<uint32_t>(model.mass.y);
        result.frozenRotation = static_cast<uint32_t>(model.mass.z);
    }
    return result;
}

bool IsMotionExport(const PhysicsDeviceQueryOutput &binding)
{
    return binding.field <= PhysicsDeviceQueryField::MotionInverseInertia;
}

void ValidateStateExportOverlap(const std::vector<PhysicsDeviceQueryOutput> &bindings, size_t bodies, size_t boxes)
{
    std::vector<uint64_t> sizes;
    sizes.reserve(bindings.size());
    for (const auto &binding : bindings) {
        const auto total = IsMotionExport(binding) ? bodies : boxes;
        const uint64_t count = binding.recordCount == UINT32_MAX ? total - std::min(size_t{binding.recordFirst}, total)
                                                                 : binding.recordCount;
        const auto size = count * PhysicsDeviceQueryWidths[static_cast<size_t>(binding.field)] * sizeof(uint32_t);
        if (!binding.buffer || size > UINT64_MAX - binding.byteOffset)
            throw JoltGpuQueryRequestError("Physics state export has an invalid destination");
        for (size_t index = 0; index < sizes.size(); ++index) {
            const auto &prior = bindings[index];
            if ((prior.buffer->GetAllocation() == binding.buffer->GetAllocation() ||
                 prior.buffer->GetWriterStateIdentity() == binding.buffer->GetWriterStateIdentity()) &&
                size && sizes[index] && binding.byteOffset < prior.byteOffset + sizes[index] &&
                prior.byteOffset < binding.byteOffset + size)
                throw JoltGpuQueryRequestError("Physics state export output ranges overlap");
        }
        sizes.push_back(size);
    }
}

template <class Source>
std::shared_ptr<rhi::PreparedComputeBatch>
PrepareGpuStateExports(const PhysicsGpuRenderPoses &program, const Source &source,
                       const std::vector<PhysicsDeviceQueryOutput> &outputs, const std::vector<size_t> &rows = {})
{
    try {
        if constexpr (std::is_same_v<Source, PhysicsGpuMotionExport>)
            return program.PrepareMotion(source, outputs, rows);
        else
            return program.PrepareBoxes(source, outputs, rows);
    } catch (const std::invalid_argument &error) {
        throw JoltGpuQueryRequestError(error.what());
    } catch (const std::length_error &error) {
        throw JoltGpuQueryRequestError(error.what());
    }
}

std::array<float, 9> StateExportFloats(PhysicsDeviceQueryField field, const PhysicsBodyMotionState &state,
                                       const PhysicsBoxShapeState &shape)
{
    using Field = PhysicsDeviceQueryField;
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
                       const std::vector<uint64_t> &shapeFirst, const std::vector<PhysicsDeviceQueryOutput> &bindings,
                       bool includeBoxes)
{
    using Field = PhysicsDeviceQueryField;
    if (!bindings.front().buffer)
        throw JoltGpuQueryRequestError("Physics aggregate export requires a compute destination");
    auto &host = bindings.front().buffer->GetHost();
    std::vector<uint64_t> targetSizes;
    targetSizes.reserve(bindings.size());
    std::vector<rhi::ComputeBufferUpdate> updates;
    for (const auto &binding : bindings) {
        const auto field = static_cast<uint32_t>(binding.field);
        const auto &buffer = binding.buffer;
        if (field >= PhysicsDeviceQueryWidths.size() || !buffer || buffer->IsFrozen() ||
            !host.SharesServicesWith(buffer->GetHost()) || binding.byteOffset % sizeof(uint32_t))
            throw JoltGpuQueryRequestError(
                "Physics aggregate export field or destination differs from its compute owner");
        const bool motion = binding.field <= Field::MotionInverseInertia;
        if (!motion && !includeBoxes)
            throw JoltGpuQueryRequestError("Physics aggregate box export requires Box rows");
        const auto &desc = buffer->GetDesc();
        const auto scalar =
            binding.field == Field::ShapeBodyIndex ? rhi::ComputeScalarType::Int32 : rhi::ComputeScalarType::Float32;
        const auto width = PhysicsDeviceQueryWidths[field];
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
void ValidateRayInput(const glm::vec3 &origin, const glm::vec3 &direction, float maximum)
{
    if (!Finite(origin) || !Finite(direction) || !std::isfinite(maximum) || maximum <= 0 || direction == glm::vec3(0))
        throw std::invalid_argument("Physics ray requires finite inputs, positive distance and nonzero direction");
}

physics::inxphygpu::InxPhyGpuQueryVolume QueryVolume(const JoltCpuQueries::Shape &shape, uint32_t layers, bool triggers)
{
    using namespace physics::inxphygpu;
    return std::visit(
        [&](const auto &primitive) {
            using Type = std::decay_t<decltype(primitive)>;
            InxPhyGpuQueryVolume native{{{0, 0, 0, 0}, {1, 0, 0, 0}}, {}, Sphere, layers, triggers};
            glm::vec3 center;
            if constexpr (std::is_same_v<Type, JoltCpuQueries::Box>) {
                center = primitive.center;
                const auto half = Vector(primitive.halfExtents);
                if (half.x <= 0 || half.y <= 0 || half.z <= 0)
                    throw std::invalid_argument("Physics query box requires positive finite half extents");
                const auto rotation = Rotation(primitive.rotation);
                native.pose.rotation = {rotation.w, rotation.x, rotation.y, rotation.z};
                native.size = {half.x, half.y, half.z, std::min({0.05f, half.x, half.y, half.z})};
                native.type = Box;
            } else {
                if (!std::isfinite(primitive.radius) || primitive.radius <= 0)
                    throw std::invalid_argument("Physics query requires a positive finite radius");
                native.size[0] = primitive.radius;
                if constexpr (std::is_same_v<Type, JoltCpuQueries::Sphere>) {
                    center = primitive.center;
                } else {
                    (void)Vector(primitive.point0);
                    (void)Vector(primitive.point1);
                    const auto a = glm::dvec3(primitive.point0), b = glm::dvec3(primitive.point1);
                    center = glm::vec3((a + b) * 0.5);
                    const auto delta = b - a;
                    const double radial = std::hypot(delta.x, delta.y);
                    const double halfLength = std::hypot(radial, delta.z) * 0.5;
                    if (halfLength > std::numeric_limits<float>::max())
                        throw std::invalid_argument("Physics query capsule length exceeds device coordinates");
                    if (halfLength > 0.0) {
                        const double angle = 0.5 * std::atan2(radial, delta.z);
                        const double scale = radial > 0.0 ? std::sin(angle) / radial : 0.0;
                        // The native capsule runs along +Z.
                        native.pose.rotation =
                            radial == 0.0 && delta.z < 0.0
                                ? ModelVector{0, 1, 0, 0}
                                : ModelVector{static_cast<float>(std::cos(angle)), static_cast<float>(-delta.y * scale),
                                              static_cast<float>(delta.x * scale), 0};
                        native.size[1] = static_cast<float>(halfLength);
                        native.type = Capsule;
                    }
                }
            }
            (void)Vector(center);
            native.pose.position = {center.x, center.y, center.z, 0};
            return native;
        },
        shape);
}

std::optional<physics::inxphygpu::InxPhyGpuQueryVolume>
CookBoxSweep(const BoxCollider &collider, const glm::vec3 &position, const glm::quat &rotation, uint32_t layers)
{
    (void)Vector(position);
    const auto q = Rotation(rotation);
    TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    const auto shape = collider.CreateJoltShape();
    if (!shape)
        throw std::runtime_error("Collider geometry is not ready for an explicit geometry query");
    const auto leaves = CollectPhysicsShapeLeaves(*shape, 1);
    if (leaves.empty())
        return std::nullopt;
    const auto &leaf = leaves.front().shape;
    const auto &box = static_cast<const JPH::BoxShape &>(*leaf.mShape);
    const glm::quat world(q.w, q.x, q.y, q.z);
    const glm::quat local(leaf.mShapeRotation.GetW(), leaf.mShapeRotation.GetX(), leaf.mShapeRotation.GetY(),
                          leaf.mShapeRotation.GetZ());
    const auto offset = leaf.mShapePositionCOM + shape->GetCenterOfMass();
    const auto center = position + world * glm::vec3(offset.GetX(), offset.GetY(), offset.GetZ());
    const auto half = box.GetHalfExtent();
    auto result =
        QueryVolume(JoltCpuQueries::Box{center, {half.GetX(), half.GetY(), half.GetZ()}, world * local}, layers, false);
    result.size[3] = box.GetConvexRadius();
    return result;
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

std::pair<std::unique_ptr<PhysicsSceneRayPlan>, std::vector<rhi::ComputeBufferUpdate>>
BuildSceneRayPlan(rhi::ComputeHost &host, const JoltGpuScene::Publication &publication, uint32_t capacity,
                  bool particles = false)
{
    using namespace physics::inxphygpu;
    auto plan = std::make_unique<PhysicsSceneRayPlan>();
    plan->directory = publication.directory;
    plan->bodyBase = publication.bodyBase;
    plan->bodyCount = publication.bodyCount;
    std::vector<InxPhyGpuSceneActor> actors;
    std::vector<std::vector<RaycastHit>> surfaces;
    std::vector<uint32_t> slots;
    std::vector<uint32_t> boundsBodyRows;
    const auto &directory = *publication.directory;
    for (uint32_t row = 0; row < publication.bodyCount; ++row) {
        const auto slot = publication.bodyBase + row;
        if (slot >= directory.actors.size() || !directory.residentBodies[slot])
            continue;
        const auto &actor = *directory.actors[slot];
        if (actor.leaves.empty())
            continue;
        auto &source = actors.emplace_back();
        source.shape = actor.authoredShape;
        source.kinematic = actor.acceptedMotionType != MotionStatic;
        source.objectId = actor.owner.gameObjectId;
        auto &members = surfaces.emplace_back();
        slots.push_back(row);
        if (actor.rigidbodyComponentId)
            boundsBodyRows.push_back(row);
        for (const auto &leaf : actor.leaves) {
            const auto index = leaf.memberIndex;
            if (members.size() <= index) {
                members.resize(index + 1);
                source.colliders.resize(index + 1);
            }
            source.colliders[index] = {leaf.colliderId,
                                       leaf.colliderGeneration,
                                       leaf.layer,
                                       UINT32_MAX,
                                       0.0f,
                                       leaf.sensor,
                                       leaf.bounciness,
                                       static_cast<PhysicsMaterialCombine>(leaf.frictionCombine),
                                       static_cast<PhysicsMaterialCombine>(leaf.bounceCombine)};
            source.colliders[index].friction = leaf.friction;
            auto &surface = members[index];
            surface.target = leaf.surfaceTarget;
            surface.bodyId = actor.bodyId;
            surface.gameObjectId = leaf.surfaceGameObjectId;
            surface.colliderId = leaf.colliderId;
        }
    }
    std::sort(boundsBodyRows.begin(), boundsBodyRows.end(), [&](uint32_t a, uint32_t b) {
        return directory.actors[publication.bodyBase + a]->rigidbodyComponentId <
               directory.actors[publication.bodyBase + b]->rigidbodyComponentId;
    });
    const auto model = BuildInxPhyGpuSceneModel(actors);
    std::vector<uint32_t> poseRows(model.model.bodies.size()), authorRows(model.model.bodies.size());
    for (size_t i = 0; i < model.model.bodyRows.size(); ++i) {
        const auto row = model.model.bodyRows[i];
        poseRows[row] = slots[i];
        authorRows[row] = static_cast<uint32_t>(i);
    }
    plan->surfaces.reserve(model.geometries.size());
    if (particles)
        plan->particleSurfaces.emplace().reserve(model.geometries.size());
    for (size_t row = 0; row < model.geometries.size(); ++row) {
        const auto body = model.model.geometries[row].indices[0];
        const auto &geometry = model.geometries[row];
        plan->surfaces.push_back(surfaces[authorRows[body]][geometry.surface]);
        if (particles) {
            const auto &surface = plan->surfaces.back();
            const auto &actor = *directory.actors[publication.bodyBase + poseRows[body]];
            const auto &material = geometry.collider;
            const auto &bound = model.collision.bounds[row];
            plan->particleSurfaces->push_back(
                {{material.componentId, material.generation, geometry.subShape, 0,
                  reinterpret_cast<uintptr_t>(actor.authoredShape.GetPtr())},
                 {{static_cast<uint32_t>(row), NoModelIndex, static_cast<uint32_t>(material.generation),
                   static_cast<uint32_t>(material.generation >> 32u)},
                  {static_cast<uint32_t>(material.componentId), static_cast<uint32_t>(material.componentId >> 32u),
                   static_cast<uint32_t>(surface.gameObjectId), static_cast<uint32_t>(surface.gameObjectId >> 32u)},
                  {1u << material.layer, static_cast<uint32_t>(material.trigger), 0u, 0u},
                  bound.halfExtent,
                  bound.center,
                  {material.friction, material.bounciness, static_cast<float>(material.frictionCombine),
                   static_cast<float>(material.bounceCombine)}},
                 surface.target,
                 actor.acceptedMotionType == MotionStatic});
        }
        // CollectPhysicsShapeLeaves orders each body's leaves by collider member.
        const auto key = std::pair{geometry.collider.componentId, geometry.collider.generation};
        const auto first = static_cast<uint32_t>(row);
        auto range = plan->colliderGeometry.try_emplace(key, first, first + 1).first;
        range->second.second = first + 1;
    }
    auto prepared = InxPhyGpuPublishedRayQuery::Build(host, PhysicsShaderTarget(host), model.model, model.collision,
                                                      capacity, publication.bodyCount, poseRows, boundsBodyRows);
    plan->query = std::move(prepared.query);
    return {std::move(plan), std::move(prepared.initialize)};
}

physics::inxphygpu::ModelWords SceneRayFilter(uint32_t layers, bool triggers)
{
    using namespace physics::inxphygpu;
    return {layers, NoModelIndex, InxPhyGpuRayIncludeStatic | (triggers ? InxPhyGpuRayIncludeTriggers : 0u), 0u};
}

std::vector<physics::inxphygpu::InxPhyGpuRay> PackSceneRays(const float *origins, const float *directions, size_t count,
                                                            float maximum, uint32_t layers, bool triggers)
{
    using namespace physics::inxphygpu;
    std::vector<InxPhyGpuRay> rays;
    rays.reserve(count);
    const auto filter = SceneRayFilter(layers, triggers);
    for (size_t row = 0; row < count; ++row) {
        const auto offset = row * 3;
        const glm::vec3 origin{origins[offset], origins[offset + 1], origins[offset + 2]};
        const glm::vec3 direction{directions[offset], directions[offset + 1], directions[offset + 2]};
        ValidateRayInput(origin, direction, maximum);
        const auto axis = glm::abs(direction);
        const auto unit = glm::normalize(direction / std::max({axis.x, axis.y, axis.z}));
        rays.push_back({{origin.x, origin.y, origin.z, maximum}, {unit.x, unit.y, unit.z, 0.0f}, filter});
    }
    return rays;
}

bool DecodeSceneRayHit(const PhysicsSceneRayPlan &plan, const physics::inxphygpu::InxPhyGpuRay &ray,
                       const physics::inxphygpu::InxPhyGpuRayHit &record, RaycastHit &hit)
{
    const auto &[normalDistance, identity] = record;
    if (identity[0] == physics::inxphygpu::NoModelIndex) {
        hit = {};
        return false;
    }
    hit = plan.surfaces.at(identity[0]);
    hit.subShapeId = identity[2];
    hit.triangleIndex = identity[3];
    hit.distance = normalDistance[3];
    hit.normal = {normalDistance[0], normalDistance[1], normalDistance[2]};
    hit.point = glm::vec3(ray.originLimit[0], ray.originLimit[1], ray.originLimit[2]) +
                glm::vec3(ray.direction[0], ray.direction[1], ray.direction[2]) * hit.distance;
    hit.collider = hit.target.GetCollider();
    hit.gameObject = hit.target.GetGameObject();
    return true;
}
Rigidbody &KinematicBody(const PhysicsActorData &actor)
{
    auto *body = actor.rigidbody;
    if (!body || !body->IsEnabled() || !body->IsKinematic())
        throw std::logic_error("Kinematic movement requires an enabled kinematic Rigidbody");
    return *body;
}
JoltGpuStepParameters StepParameters(const PhysicsFixedInputs &inputs)
{
    const auto &c = EngineConfig::Get();
    StepParams p{};
    p.counts.w = static_cast<uint32_t>(c.physicsVelocitySteps);
    p.solve.x = static_cast<uint32_t>(c.physicsPositionSteps);
    p.solve.y = static_cast<uint32_t>(c.physicsCollisionSteps);
    p.gravityDt = {inputs.gravity.x, inputs.gravity.y, inputs.gravity.z, inputs.duration};
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

// Publication work retains its operands and receipt independently of the solver.
struct PhysicsGpuPublicationRequest
{
    struct Receipt
    {
        rhi::ComputeCompletion completion;
        uint64_t frame, gpuGrant, transferGrant, bytes;
    };

    JoltGpuControlWork work;
    std::shared_ptr<rhi::PreparedComputeBatch> preservation;
    std::optional<Receipt> receipt;
    std::exception_ptr error;
    uint64_t id = 0;
    bool cancelled = false;

    [[nodiscard]] bool Pending() const noexcept
    {
        return receipt || (!cancelled && (work.packet || work.continueWork));
    }

    [[nodiscard]] const std::shared_ptr<rhi::PreparedComputeBatch> &Packet() const noexcept
    {
        return preservation ? preservation : work.packet;
    }
};

struct PhysicsWorld::GpuPublicationReader
{
    using Request = PhysicsGpuPublicationRequest;

    GpuPublicationReader(FrameWorkOwner &owner, rhi::ComputeHost &host) : m_owner(owner), m_host(host)
    {
        FrameWorkSource source;
        source.describe = [this] { return Describe(); };
        source.advance = [this](bool gpu, uint64_t token) { return Advance(gpu, token); };
        m_source = owner.AddSource(std::move(source));
    }

    ~GpuPublicationReader()
    {
        m_owner.RemoveSource(m_source);
        for (auto &request : m_pending)
            if (request->receipt)
                request->receipt->completion.Wait();
    }

    [[nodiscard]] std::shared_ptr<Request> Enqueue(JoltGpuControlWork work)
    {
        auto request = std::make_shared<Request>();
        request->work = std::move(work);
        request->id = m_nextRequest++;
        m_pending.push_back(request);
        m_owner.Wake(m_source);
        return request;
    }

    void Run(JoltGpuControlWork work)
    {
        if (!work.packet && !work.continueWork)
            return;
        Request request;
        request.work = std::move(work);
        Focus focus(*this, request);
        try {
            m_owner.DrainRequired(
                m_source, [this] { return Availability(m_host); },
                [this, &request](auto, uint64_t maximum) { return Wait(request, maximum); });
        } catch (...) {
            if (request.receipt)
                request.receipt->completion.Wait();
            throw;
        }
        (void)Ready(request);
    }

    [[nodiscard]] static bool Ready(const Request &request)
    {
        if (request.error)
            std::rethrow_exception(request.error);
        return !request.Pending();
    }

    [[nodiscard]] bool Poll(Request &request)
    {
        if (Ready(request))
            return true;
        Focus focus(*this, request);
        (void)m_owner.AdvanceDependency(m_source, [this] { return Availability(m_host); });
        return Ready(request);
    }

    static void Cancel(std::shared_ptr<Request> &request) noexcept
    {
        if (request) {
            request->cancelled = true;
            if (!request->receipt) {
                request->preservation.reset();
                request->work = {};
            }
        }
        request.reset();
    }

  private:
    struct Focus
    {
        Focus(GpuPublicationReader &reader, Request &request) : reader(reader)
        {
            if (reader.m_focused)
                throw std::logic_error("Physics publication read cannot reenter its source");
            reader.m_focused = &request;
        }
        ~Focus()
        {
            reader.m_focused = nullptr;
            // A focused drain can make the source go to sleep while async work is queued.
            if (!reader.m_pending.empty())
                reader.m_owner.Wake(reader.m_source);
        }
        GpuPublicationReader &reader;
    };

    Request *Current()
    {
        if (m_focused)
            return m_focused;
        while (!m_pending.empty() && !m_pending.front()->Pending())
            m_pending.pop_front();
        return m_pending.empty() ? nullptr : m_pending.front().get();
    }

    FrameWorkFacts Describe()
    {
        FrameWorkFacts facts;
        auto *request = Current();
        if (!request || !request->Pending())
            return facts;
        facts.pending = true;
        if (request->receipt)
            facts.host.ready = request->receipt->completion.IsComplete();
        else if (const auto &packet = request->Packet()) {
            facts.host.ready = packet->RequiresPreparation() || packet->HasPreservationWork();
            facts.gpu.ready = !facts.host.ready;
        } else {
            facts.host.ready = bool(request->work.continueWork);
        }
        // Both consumers are on a live dependency boundary: presentation or the next fixed step.
        facts.host.reserved = facts.gpu.reserved = true;
        facts.host.token = facts.gpu.token = request->id;
        return facts;
    }

    bool Advance(bool gpu, uint64_t token)
    {
        auto *request = Current();
        if (!request || request->id != token || !request->Pending())
            return false;
        try {
            return Advance(*request, gpu);
        } catch (...) {
            request->error = std::current_exception();
            request->cancelled = true;
            return true;
        }
    }

    bool Advance(Request &request, bool gpu)
    {
        if (gpu) {
            const auto &packet = request.Packet();
            const auto bytes = packet->GetTransferBytes();
            const uint64_t prediction = 4'000 * std::max(uint64_t{1}, packet->GetExecutionSignature().front());
            if (!m_owner.CanAdmitResource(FrameWorkResource::Gpu, prediction, true) ||
                (bytes && !m_owner.CanAdmitResource(FrameWorkResource::Transfer, bytes, true)))
                return false;
            const auto accept = [&] {
                request.receipt.emplace(Request::Receipt{
                    packet->GetCompletion(), m_owner.Opportunity(),
                    m_owner.RecordAcceptedResource(FrameWorkResource::Gpu, prediction),
                    bytes ? m_owner.RecordAcceptedResource(FrameWorkResource::Transfer, bytes) : 0, bytes});
            };
            try {
                if (!packet->TrySubmit())
                    return false;
            } catch (...) {
                if (packet->GetSubmission().IsValid())
                    accept();
                throw;
            }
            accept();
        } else if (request.receipt) {
            const auto &receipt = *request.receipt;
            const auto elapsed = receipt.completion.GetDeviceElapsedNanoseconds();
            m_owner.ObserveResource(FrameWorkResource::Gpu,
                                    elapsed ? static_cast<uint64_t>(*elapsed) : receipt.gpuGrant, receipt.frame,
                                    receipt.gpuGrant);
            m_owner.ObserveResource(FrameWorkResource::Transfer, receipt.bytes, receipt.frame, receipt.transferGrant);
            request.receipt.reset();
            if (request.cancelled) {
                request.preservation.reset();
                request.work = {};
            } else if (request.preservation) {
                request.preservation.reset();
            } else {
                Continue(request);
            }
        } else if (request.work.packet) {
            auto prepared = request.work.packet->PreparePreservation(
                std::max(uint64_t{4}, m_owner.ResourceRemaining(FrameWorkResource::Transfer)), true);
            if (!prepared.ready)
                return false;
            request.preservation = std::move(prepared.copy);
        } else {
            Continue(request);
        }
        return true;
    }

    static void Continue(Request &request)
    {
        auto completed = std::exchange(request.work, {});
        if (completed.continueWork)
            request.work = completed.continueWork();
    }

    FrameWorkWaitResult Wait(Request &request, uint64_t maximum)
    {
        if (!request.receipt)
            if (const auto &packet = request.Packet(); packet && packet->RequiresPreparation())
                if (const auto moved = packet->WaitForPreparation(maximum))
                    return *moved ? FrameWorkWaitResult::Progress : FrameWorkWaitResult::Waiting;
        const auto ticket =
            request.receipt ? request.receipt->completion.GetSubmission() : m_host.Queue().GetCapacityPredecessor();
        if (!ticket.IsValid())
            return FrameWorkWaitResult::Blocked;
        return m_host.Queue().WaitFor(ticket, maximum) ? FrameWorkWaitResult::Progress : FrameWorkWaitResult::Waiting;
    }

    FrameWorkOwner &m_owner;
    rhi::ComputeHost &m_host;
    FrameWorkOwner::SourceId m_source = 0;
    uint64_t m_nextRequest = 1;
    std::deque<std::shared_ptr<Request>> m_pending;
    Request *m_focused = nullptr;
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
        auto joints = std::make_unique<JoltGpuJoints>(host, load(JoltGpuProgram::Joints));
        auto rendering = std::make_unique<PhysicsGpuRenderPoses>(
            host, physics::inxphygpu::LoadInxPhyGpuProgram("render_poses", target));
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
        m_gpuRenderProgram = std::move(rendering);
        m_gpuContactReports = std::move(reports);
        EnsureGpuPublicationReader();
        m_gpuScene = std::move(scene);
    }
    return *m_gpuScene;
}
PhysicsWorld::GpuPublicationReader &PhysicsWorld::EnsureGpuPublicationReader()
{
    if (!m_gpuWork || !m_gpuHost)
        throw std::runtime_error("GPU physics requires engine compute services");
    if (!m_gpuPublicationReader)
        m_gpuPublicationReader = std::make_unique<GpuPublicationReader>(*m_gpuWork, *m_gpuHost);
    return *m_gpuPublicationReader;
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
        domain.participantOutput)
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
    domain.gpuTargetAuthors.reset();
    domain.lastPublication.reset();
    domain.sceneRays.reset();
    domain.gpuContacts.Close();
    domain.queuedColliderPairs.reset();
    if (auto pose = std::exchange(domain.presentationPoseReads, {}))
        GpuPublicationReader::Cancel(pose->request);
    GpuPublicationReader::Cancel(domain.participantOutput);
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
        if (auto pose = std::exchange(domain->presentationPoseReads, {}))
            GpuPublicationReader::Cancel(pose->request);
        GpuPublicationReader::Cancel(domain->participantOutput);
        GpuPublicationReader::Cancel(domain->participantInitialization);
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
        domain->sceneRays.reset();
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
    m_gpuPairQuery.reset();
    m_gpuParticleExport.reset();
    m_gpuJoints.reset();
    m_gpuPublicationReader.reset();
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
        domain.gpuTargetAuthors.reset();
    }
    domain.admittedContactPolicy = std::move(policy);
    SealStaticFixedMoves(domain);
    domain.cpuFixedAdmitted = bool(domain.cpuControl);
    if (domain.cpuFixedAdmitted) {
        domain.cpuFixedQueryWrite.emplace(domain.cpuScene->QueryMutex());
        domain.cpuControl->SealFixed();
        domain.cpuControl->ApplyFixed([this](uint32_t, const auto &ids) { RetireCpuJointRoutes(ids); });
        auto &system = domain.cpuScene->System();
        system.SetGravity(domain.admittedGravity);
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
            if (job && job->output && job->receipt == p) {
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
        const PhysicsGpuRenderBindings::Source source{
            ScenePoseSource(*publication),
            [&](PhysicsActorHandle handle, uint32_t bodyId) -> std::optional<PhysicsGpuBodyPoseBinding> {
                const auto slot = publication->directory->FindBodySlot(bodyId);
                if (slot < publication->bodyBase || slot - publication->bodyBase >= publication->bodyCount)
                    return std::nullopt;
                const auto &actor = publication->directory->actors[slot];
                if (!actor || actor->actor != handle)
                    return std::nullopt;
                return ScenePoseBinding(*publication, *actor);
            }};
        job->output = state.bindings.PrepareExport(job->snapshot, source, *m_gpuRenderProgram);
        job->receipt = publication;
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
    GpuPublicationReader::Cancel(domain.participantInitialization);
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
    RegisterDomain(domain);
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
    if (domain.retiring || domain.owner.handle->destroyRequested)
        return true;
    RequireHealthyStep(domain);
    if (domain.participantInitialization) {
        if (!GpuPublicationReader::Ready(*domain.participantInitialization))
            return false;
        domain.participantInitialization.reset();
    }
    auto &values = domain.participants;
    values.erase(std::remove_if(values.begin(), values.end(),
                                [](const auto &entry) {
                                    const auto owner = entry.lock();
                                    return !owner || owner->IsRetired();
                                }),
                 values.end());
    for (const auto &entry : values) {
        auto owner = entry.lock();
        if (!owner || owner->IsReady())
            continue;
        // The next step captures ready participants once, before its solver starts.
        if (domain.stepPending)
            return false;
        domain.participantInitialization = EnsureGpuPublicationReader().Enqueue(
            {nullptr, [owner = std::move(owner)] { return owner->PrepareInitialize(); },
             JoltGpuControlWork::ResumeAfter::HostWork});
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
    if (observers.empty() && !domain.participantOutput)
        return true;
    if (domain.participantOutput) {
        if (!GpuPublicationReader::Ready(*domain.participantOutput))
            return false;
        domain.participantOutput.reset();
    }
    // Observe a previously admitted solver without advancing it from an output reader.
    const auto status = domain.gpuRegistered ? m_gpuScene->GetTickStatus(domain.owner) : JoltGpuTick::Status::Idle;
    if (status == JoltGpuTick::Status::Failed)
        throw std::runtime_error("Participant output source failed");
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
    if (std::any_of(observers.begin(), observers.end(), [](const auto &owner) { return !owner->IsReady(); })) {
        (void)ServiceParticipantPreparation();
        return false;
    }
    if (observers.empty())
        return true;
    const auto &clock = domain.owner.simulation;
    auto epoch = clock.simulationEpoch, step = clock.publishedFixedStepIndex;
    bool fixed = false;
    if (domain.gpuRegistered) {
        if (m_gpuScene->IsWriterFor(domain.owner))
            return false;
        const auto &publication = m_gpuScene->GetPublished(domain.owner);
        if (publication && publication->simulationEpoch == epoch) {
            step = publication->fixedStepIndex;
            fixed = publication->isFixedStep;
        }
    }
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
    domain.participantOutput = EnsureGpuPublicationReader().Enqueue(std::move(copies));
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
            if ((domain.UsesGpu() && !ServiceParticipantPreparation()) || !ServiceParticipantOutputs())
                return false;
            auto participants = domain.UsesGpu() ? CaptureParticipants(domain) : JoltGpuParticipantStages{};
            if ((domain.gpuRegistered || participants) && !EnsureGpuReady(domain))
                return false;
            RefreshBodySurfaces(domain);
            domain.stepEpoch = epoch;
            domain.stepIndex = index;
            domain.admittedDeltaTime = dt;
            domain.admittedGravity = JPH::Vec3(domain.gravity.x, domain.gravity.y, domain.gravity.z);
            const auto &config = EngineConfig::Get();
            if (config.physicsVelocitySteps < 0 || config.physicsPositionSteps < 0 || config.physicsCollisionSteps <= 0)
                throw std::invalid_argument("Physics iterations and substeps are invalid");
            if (domain.UsesGpu()) {
                domain.admittedParameters = StepParameters(domain.AdmittedFixedInputs());
            } else {
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

bool PhysicsWorld::HasRunningGpuFixed(const PhysicsDomain &domain) const noexcept
{
    return domain.solverStarted && domain.gpuRegistered && m_gpuScene->HasRunningFixedTick(domain.owner);
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
bool PhysicsWorld::BeginGpuControl(PhysicsDomain &domain, bool numericalChange, bool requirePublication) const
{
    RequireGpu(domain);
    numericalChange = numericalChange || m_gpuScene->HasPendingControl(domain.owner);
    AdmissionScope admission(*m_gpuScene, domain.owner);
    FrameWorkOwner::Scope preparation(*m_gpuWork);
    auto rendering = PrepareRenderConsumer(domain, numericalChange);
    if (!numericalChange && !rendering && !requirePublication)
        return true;
    if (!m_gpuScene->TryBeginControl(domain.owner, domain.owner.simulation.simulationEpoch, std::move(rendering)))
        return false;
    domain.gpuTargetAuthors.reset();
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
    GpuPublicationReader::Cancel(domain.participantInitialization);
    GpuPublicationReader::Cancel(domain.participantOutput);
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
        while (!BeginGpuControl(domain, pending, true))
            DrainGpuSource(domain);
        if (!AdvanceGpuControl(domain, true))
            throw std::runtime_error("GPU query control did not complete");
    }
}

void PhysicsWorld::RunGpuPublicationRead(PhysicsDomain &domain, const JoltGpuScene::Consumer &consumer,
                                         bool requireStaticGpuBoundaries) const
{
    RequireOwnerThread();
    const WorldExecutionScope ownerScope(&domain.owner);
    RequireHealthyStep(domain);
    if (domain.synchronousRead)
        throw std::logic_error("Physics query cannot reenter its synchronous source");
    FlagScope reading(domain.synchronousRead);
    const auto observe = [&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        JoltGpuControlWork prepared;
        {
            FrameWorkOwner::Scope preparation(*m_gpuWork);
            prepared = consumer(publication);
        }
        RunPreparedWork(domain, std::move(prepared));
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
        observe(publication);
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
        AcceptQueryGpuPublication(domain, HasDomainControl(domain));
        if (consumer)
            observe(ReadGpuQueryPublication(domain));
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
    JoltGpuControlWork prepared;
    {
        FrameWorkOwner::Scope preparation(*m_gpuWork);
        prepared = consumer(publication);
    }
    RunPreparedWork(domain, std::move(prepared));
}

std::vector<Collider *> PhysicsWorld::QueryColliders(PhysicsDomain &domain, const JoltCpuQueries::Shape &shape,
                                                     uint32_t layers, bool triggers) const
{
    const auto request = QueryVolume(shape, layers, triggers);
    SealSpatialQuery(domain);
    if (domain.cpuScene)
        return JoltCpuQueries(*domain.cpuScene).Overlap(shape, {layers, triggers});
    const auto gpu = ReadGpuSpatialQueryPublication(domain);
    std::vector<Collider *> colliders;
    if (!gpu || !layers)
        return colliders;
    std::unique_ptr<PhysicsSceneRayPlan> replacement;
    auto *plan = domain.sceneRays.get();
    std::shared_ptr<rhi::PreparedComputeBatch> result;
    ObserveSpatialGpuQuery(domain, gpu, [&](const auto &accepted) {
        std::vector<rhi::ComputeBufferUpdate> uploads;
        if (!plan || !plan->Matches(*accepted)) {
            auto prepared = BuildSceneRayPlan(*m_gpuHost, *accepted, plan ? plan->query.Capacity() : 64);
            replacement = std::move(prepared.first);
            uploads = std::move(prepared.second);
            plan = replacement.get();
        }
        result = plan->query.PrepareOverlap(*m_gpuHost, ScenePoseSource(*accepted), request, std::move(uploads));
        return JoltGpuControlWork{result};
    });
    if (replacement)
        domain.sceneRays = std::move(replacement);
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    for (const auto row :
         physics::inxphygpu::ReadInxPhyGpuSelection(*result, static_cast<uint32_t>(plan->surfaces.size())))
        if (auto *collider = plan->surfaces[row].target.GetCollider())
            colliders.push_back(collider);
    std::sort(colliders.begin(), colliders.end(),
              [](const auto *a, const auto *b) { return a->GetComponentID() < b->GetComponentID(); });
    colliders.erase(std::unique(colliders.begin(), colliders.end()), colliders.end());
    return colliders;
}

bool PhysicsWorld::QueryClosestCast(PhysicsDomain &domain, const JoltCpuQueries::Shape &shape,
                                    const glm::vec3 &direction, float maximum, RaycastHit &hit, uint32_t layers,
                                    bool triggers) const
{
    const auto volume = QueryVolume(shape, layers, triggers);
    ValidateRayInput({volume.pose.position[0], volume.pose.position[1], volume.pose.position[2]}, direction, maximum);
    SealSpatialQuery(domain);
    if (domain.cpuScene)
        return JoltCpuQueries(*domain.cpuScene).Cast(shape, direction, maximum, hit, {layers, triggers});
    return RunGpuShapeCast(domain, volume, direction, maximum, hit);
}

bool PhysicsWorld::RunGpuShapeCast(PhysicsDomain &domain, const physics::inxphygpu::InxPhyGpuQueryVolume &volume,
                                   const glm::vec3 &direction, float maximum, RaycastHit &hit,
                                   const BoxCollider *swept) const
{
    using namespace physics::inxphygpu;
    hit = {};
    const auto gpu = ReadGpuSpatialQueryPublication(domain);
    if (!gpu || !volume.layers)
        return false;
    const glm::dvec3 delta(direction);
    const auto unit = glm::vec3(delta / std::hypot(delta.x, delta.y, delta.z));
    const ModelVector motion{unit.x, unit.y, unit.z, swept ? 1.0e-3f : 0.0f};
    std::unique_ptr<PhysicsSceneRayPlan> replacement;
    auto *plan = domain.sceneRays.get();
    std::shared_ptr<rhi::PreparedComputeBatch> result;
    std::exception_ptr requestError;
    ObserveSpatialGpuQuery(domain, gpu, [&](const auto &accepted) {
        return PrepareJoltGpuQueryWork(
            [&]() -> JoltGpuControlWork {
                std::vector<rhi::ComputeBufferUpdate> uploads;
                const auto target = PhysicsShaderTarget(*m_gpuHost);
                if (!plan || !plan->Matches(*accepted)) {
                    auto prepared = BuildSceneRayPlan(*m_gpuHost, *accepted, plan ? plan->query.Capacity() : 64);
                    replacement = std::move(prepared.first);
                    uploads = std::move(prepared.second);
                    plan = replacement.get();
                }
                std::vector<ModelWords> exclusions;
                if (swept) {
                    const auto self = std::pair{swept->GetComponentID(), swept->GetLifetimeGeneration()};
                    const auto ownRange = plan->colliderGeometry.find(self);
                    const auto ownBody = ownRange == plan->colliderGeometry.end()
                                             ? UINT32_MAX
                                             : plan->surfaces[ownRange->second.first].bodyId;
                    const auto object = swept->GetGameObject()->GetID();
                    const auto pairs = CaptureColliderPairs(domain);
                    exclusions.resize((plan->surfaces.size() + 127) / 128);
                    for (const auto &[key, range] : plan->colliderGeometry) {
                        const auto &surface = plan->surfaces[range.first];
                        const bool ownsBody = ownBody != UINT32_MAX && surface.bodyId == ownBody;
                        const bool sameObject = surface.gameObjectId == object;
                        const auto [first, second] = std::minmax(self, key);
                        const PhysicsColliderPair pair{first.first, first.second, second.first, second.second};
                        const bool ignored = std::binary_search(pairs->begin(), pairs->end(), pair);
                        if (ownsBody || sameObject || ignored)
                            for (uint32_t row = range.first; row < range.second; ++row)
                                exclusions[row / 128][(row / 32) % 4] |= uint32_t{1} << (row % 32);
                    }
                }
                result = plan->query.PrepareShapeCast(
                    *m_gpuHost, target, ScenePoseSource(*accepted), volume, motion, maximum, exclusions,
                    [plan](uint32_t row) { return plan->surfaces[row].bodyId; }, std::move(uploads));
                return {result};
            },
            requestError);
    });
    if (requestError)
        std::rethrow_exception(requestError);
    if (replacement)
        domain.sceneRays = std::move(replacement);
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    InxPhyGpuShapeCastHit record;
    result->Readback(0)->ReadInto(&record, sizeof(record));
    if (record.point[3] != 0.0f)
        throw std::runtime_error("inxphygpu shape cast did not converge");
    const InxPhyGpuRay path{volume.pose.position, motion, {}};
    if (!DecodeSceneRayHit(*plan, path, record.surface, hit))
        return false;
    hit.point = {record.point[0], record.point[1], record.point[2]};
    return true;
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
                QueueGpuBodyCommand(
                    domain, {id, PhysicsBodyCommand::Pose{Vector(position),
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
        QueueGpuBodyCommand(
            domain, {id, PhysicsBodyCommand::Pose{
                             position ? std::make_optional(*position) : std::nullopt,
                             rotation ? std::make_optional(glm::quat(q.w, q.x, q.y, q.z)) : std::nullopt, cancelMove}});
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
    RunGpuPublicationRead(domain, {});
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
    QueueGpuBodyCommand(domain, {id, PhysicsBodyCommand::Velocity{angular ? std::nullopt : velocity,
                                                                  angular ? velocity : std::nullopt}});
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
    QueueGpuBodyCommand(domain, {id, PhysicsBodyCommand::Force{static_cast<PhysicsForceKind>(kind),
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
        auto &domain = DomainForBody(s.bodyId);
        QueueGpuBodyCommand(domain,
                            {s.bodyId, PhysicsBodyCommand::ResetMotion{s.position, glm::quat(q.w, q.x, q.y, q.z),
                                                                       s.linearVelocity, s.angularVelocity}});
        domain.MarkGpuPoseDiscontinuity(s.bodyId);
    }
}

JoltGpuControlWork PhysicsWorld::PrepareStateBatch(PhysicsDomain &domain,
                                                   const std::shared_ptr<const JoltGpuScene::Publication> &publication,
                                                   const std::vector<Rigidbody *> &bodies, PhysicsStateFields fields,
                                                   bool includeBoxes, bool triggers,
                                                   const std::vector<PhysicsDeviceQueryOutput> &exports,
                                                   PhysicsRigidbodyStateBatch &batch, PhysicsGpuMotionRead &hostMotion,
                                                   PhysicsGpuMotionExport &motion, PhysicsGpuBoxExport &boxes,
                                                   std::exception_ptr &requestError, bool deferExport) const
{
    return PrepareJoltGpuQueryWork(
        [&]() -> JoltGpuControlWork {
            for (const auto &binding : exports)
                if (binding.field > PhysicsDeviceQueryField::ShapeBounciness ||
                    (!includeBoxes && binding.field >= PhysicsDeviceQueryField::ShapeBodyIndex))
                    throw JoltGpuQueryRequestError("Physics export requires its requested state family");
            batch.bodies = bodies;
            batch.states.resize(bodies.size());
            const bool host = exports.empty();
            const bool numericMotion =
                host ? (static_cast<uint32_t>(fields) & static_cast<uint32_t>(PhysicsStateFields::All)) != 0
                     : std::any_of(exports.begin(), exports.end(), IsMotionExport);
            std::vector<PhysicsGpuMotionBinding> selected;
            selected.reserve(numericMotion || (host && includeBoxes) ? bodies.size() : 0);
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
                if (numericMotion || (host && includeBoxes))
                    selected.push_back(SceneMotionBinding(*publication, identity));
                if (includeBoxes && publication->directory->residentBodies[slot])
                    for (const auto &leaf : identity.leaves)
                        if (leaf.shapeType == ShapeBox && (triggers || !leaf.sensor))
                            boxes.boxes.push_back(
                                {slot - publication->bodyBase,
                                 static_cast<uint32_t>(index),
                                 {leaf.centerOfMassOffset, leaf.localRotation, Vector(leaf.dimensions), leaf.friction,
                                  leaf.bounciness, leaf.memberIndex, leaf.sensor}});
            }
            const uint64_t generation = domain.queryGeneration.load(std::memory_order_acquire) +
                                        static_cast<uint64_t>(domain.controlNumericalChange);
            batch.sources = {
                {publication->worldId, publication->simulationEpoch, publication->fixedStepIndex, generation}};
            batch.sourceForBody.assign(bodies.size(), 0u);
            boxes.poses = ScenePoseSource(*publication);
            batch.boxCount = boxes.boxes.size();
            if (host) {
                if (!boxes.boxes.empty())
                    fields = static_cast<PhysicsStateFields>(static_cast<uint32_t>(fields) |
                                                             static_cast<uint32_t>(PhysicsStateFields::CenterOfMass) |
                                                             static_cast<uint32_t>(PhysicsStateFields::Rotation));
                if (numericMotion || !boxes.boxes.empty())
                    hostMotion = PhysicsGpuMotionRead::Prepare(*m_gpuHost, boxes.poses, std::move(selected), fields);
                return hostMotion.packet;
            }
            motion = {boxes.poses, std::move(selected)};
            if (deferExport)
                return {};
            ValidateStateExportOverlap(exports, batch.states.size(), batch.boxCount);
            std::vector<PhysicsDeviceQueryOutput> motionTargets, shapeTargets;
            for (const auto &binding : exports)
                (IsMotionExport(binding) ? motionTargets : shapeTargets).push_back(binding);
            auto motionWork = PrepareGpuStateExports(*m_gpuRenderProgram, motion, motionTargets);
            auto shapeWork = PrepareGpuStateExports(*m_gpuRenderProgram, boxes, shapeTargets);
            return ContinueJoltGpuControlWork(std::move(motionWork),
                                              [work = std::move(shapeWork)]() mutable { return std::move(work); });
        },
        requestError);
}

void PhysicsWorld::MaterializeStateBatch(PhysicsDomain &domain, PhysicsRigidbodyStateBatch &batch,
                                         const PhysicsGpuMotionRead &hostMotion, const PhysicsGpuBoxExport &boxes,
                                         bool numerical, PhysicsStateFields cachedFields) const
{
    if (!numerical || hostMotion.Empty())
        return;
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    batch.states = hostMotion.Read();
    batch.boxes.reserve(boxes.boxes.size());
    for (const auto &binding : boxes.boxes) {
        const auto &body = batch.states[binding.bodyIndex];
        const auto &box = binding.author;
        batch.boxes.push_back(
            {binding.bodyIndex,
             {box.memberIndex, body.centerOfMass + body.rotation * box.centerOfMassOffset,
              glm::normalize(body.rotation * box.localRotation), box.halfExtents, box.friction, box.bounciness}});
    }
    const auto generation = batch.sources.front().numericalGeneration;
    const bool cache = HasPhysicsStateField(cachedFields, PhysicsStateFields::All);
    for (auto &state : batch.states) {
        if (!HasPhysicsStateField(cachedFields, PhysicsStateFields::Rotation))
            state.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        if (!HasPhysicsStateField(cachedFields, PhysicsStateFields::CenterOfMass))
            state.centerOfMass = glm::vec3(0.0f);
        if (cache)
            domain.requestedMotion[state.bodyId] = {generation, state, cachedFields};
    }
}

PhysicsRigidbodyStateBatch
PhysicsWorld::GatherRigidbodyStates(const std::vector<Rigidbody *> &bodies, PhysicsStateFields fields,
                                    bool includeBoxes, bool triggers,
                                    const std::vector<PhysicsDeviceQueryOutput> &exports) const
{
    return GatherRigidbodyStateRows(bodies, fields, includeBoxes, triggers, exports, false, {});
}

PhysicsRigidbodyStateBatch
PhysicsWorld::GatherRigidbodyStateRows(const std::vector<Rigidbody *> &bodies, PhysicsStateFields fields,
                                       bool includeBoxes, bool triggers,
                                       const std::vector<PhysicsDeviceQueryOutput> &exports, bool sealed,
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
        PhysicsGpuMotionRead hostMotion;
        PhysicsGpuMotionExport motion;
        PhysicsGpuBoxExport gpuBoxes;
    };
    std::vector<Group> groups;
    std::unordered_map<PhysicsDomain *, size_t> byOwner;
    std::vector<uint8_t> cpuRows;
    const bool exportMotion = std::any_of(exports.begin(), exports.end(), [](const auto &value) {
        return value.field <= PhysicsDeviceQueryField::MotionInverseInertia;
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
                return PrepareStateBatch(domain, publication, group.bodies, fields, includeBoxes, triggers, exports,
                                         group.batch, group.hostMotion, group.motion, group.gpuBoxes, error,
                                         !direct && !exports.empty());
            };
            if (sealed) {
                ObserveSpatialGpuQuery(domain, gpu, prepare);
            } else {
                RunGpuPublicationRead(domain, prepare);
            }
            if (error)
                std::rethrow_exception(error);
            MaterializeStateBatch(domain, group.batch, group.hostMotion, group.gpuBoxes, exports.empty(), fields);
            // One world's query rows already follow caller order. The device can export
            // the entire batch, including the owning body of each compound shape.
            if (direct)
                return std::move(group.batch);
            batch.sources.push_back(group.batch.sources.front());
        }
        for (size_t local = 0; local < group.indices.size(); ++local) {
            const auto row = group.indices[local];
            batch.states[row] = group.batch.states[local];
            batch.sourceForBody[row] = source;
        }
        if (!exports.empty())
            for (auto &box : group.gpuBoxes.boxes) {
                box.bodyIndex = static_cast<uint32_t>(group.indices[box.bodyIndex]);
                ++shapeCounts[box.bodyIndex];
            }
        for (auto box : group.batch.boxes) {
            box.bodyIndex = static_cast<uint32_t>(group.indices[box.bodyIndex]);
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
    std::vector<PhysicsDeviceQueryOutput> motionTargets, shapeTargets;
    for (const auto &binding : exports)
        (IsMotionExport(binding) ? motionTargets : shapeTargets).push_back(binding);
    for (auto &target : motionTargets) {
        target.recordFirst = 0;
        target.recordCount = UINT32_MAX;
    }
    for (auto &target : shapeTargets) {
        target.recordFirst = 0;
        target.recordCount = UINT32_MAX;
    }
    std::vector<std::pair<PhysicsDomain *, JoltGpuControlWork>> plannedExports;
    auto nextShape = shapeFirst;
    for (auto &group : groups) {
        if (group.backend == Backend::Cpu)
            continue;
        auto motionWork = PrepareGpuStateExports(*m_gpuRenderProgram, group.motion, motionTargets, group.indices);
        std::vector<size_t> shapeRows;
        shapeRows.reserve(group.gpuBoxes.boxes.size());
        for (const auto &box : group.gpuBoxes.boxes)
            shapeRows.push_back(static_cast<size_t>(nextShape[box.bodyIndex]++));
        auto boxWork = PrepareGpuStateExports(*m_gpuRenderProgram, group.gpuBoxes, shapeTargets, shapeRows);
        auto work = ContinueJoltGpuControlWork(std::move(motionWork),
                                               [work = std::move(boxWork)]() mutable { return std::move(work); });
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
                                            const std::vector<PhysicsDeviceQueryOutput> &exports) const
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
    if (const auto job = domain.presentationPoseReads) {
        if (!m_gpuPublicationReader->Poll(*job->request))
            return false;
        domain.presentationPoseReads.reset();
        if (job->publication != publication ||
            job->generation != domain.queryGeneration.load(std::memory_order_acquire))
            return false;
        FrameWorkOwner::Scope materialize(*m_gpuWork);
        for (const auto &state : job->read.Read())
            domain.requestedMotion[state.bodyId] = {job->generation, state, fields};
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
    FrameWorkOwner::Scope preparation(*m_gpuWork);
    std::vector<PhysicsGpuMotionBinding> bindings;
    bindings.reserve(missing.size());
    for (const auto *body : missing) {
        const auto slot = publication->directory->FindBodySlot(body->GetBodyId());
        if (slot < publication->bodyBase || slot - publication->bodyBase >= publication->bodyCount)
            throw JoltGpuQueryRequestError("Presentation pose requires an accepted body identity");
        const auto &actor = *publication->directory->actors[slot];
        if (actor.rigidbodyComponentId != body->GetComponentID() || actor.rigidbody != body->GetECSHandle())
            throw JoltGpuQueryRequestError("Presentation pose differs from its retained Rigidbody");
        bindings.push_back(SceneMotionBinding(*publication, actor));
    }
    job->read = PhysicsGpuMotionRead::Prepare(*m_gpuHost, ScenePoseSource(*publication), std::move(bindings), fields);
    job->request = m_gpuPublicationReader->Enqueue(job->read.packet);
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

PhysicsBodyMotionState PhysicsWorld::ReadMotionState(PhysicsDomain &domain, uint32_t id, PhysicsStateFields fields,
                                                     bool *sleeping) const
{
    RequireOwnerThread();
    RequireHealthyStep(domain);
    if (!sleeping)
        if (const auto *cached = FindCachedMotionState(domain, id, fields))
            return *cached;
    if (!sleeping && domain.presentationPoseReadDepth)
        throw PhysicsPresentationPosePending{};
    if (ReadBodyBackend(domain, id) == Backend::Cpu) {
        RunCpuControl(domain);
        const auto state = domain.cpuScene->ReadMotionState(id, fields, sleeping);
        if (static_cast<uint32_t>(fields) & static_cast<uint32_t>(PhysicsStateFields::All))
            domain.requestedMotion[id] = {domain.queryGeneration.load(std::memory_order_acquire), state, fields};
        return state;
    }
    PhysicsGpuMotionRead read;
    uint64_t generation = 0;
    auto readFields = static_cast<uint32_t>(fields);
    if (readFields & static_cast<uint32_t>(PresentationPoseFields))
        readFields |= static_cast<uint32_t>(PresentationPoseFields);
    const auto requested = static_cast<PhysicsStateFields>(readFields);
    std::exception_ptr error;
    RunGpuPublicationRead(domain, [&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        return PrepareJoltGpuQueryWork(
            [&]() -> JoltGpuControlWork {
                const auto slot = publication->directory->FindBodySlot(id);
                if (slot < publication->bodyBase || slot - publication->bodyBase >= publication->bodyCount)
                    throw JoltGpuQueryRequestError("Physics state requires an accepted live body");
                const auto &identity = *publication->directory->actors[slot];
                generation = domain.queryGeneration.load(std::memory_order_acquire) +
                             static_cast<uint64_t>(domain.controlNumericalChange);
                read = PhysicsGpuMotionRead::Prepare(*m_gpuHost, ScenePoseSource(*publication),
                                                     {SceneMotionBinding(*publication, identity)}, requested,
                                                     sleeping != nullptr);
                return read.packet;
            },
            error);
    });
    if (error)
        std::rethrow_exception(error);
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    std::vector<bool> activity;
    const auto result = read.Read(sleeping ? &activity : nullptr).front();
    if (sleeping)
        *sleeping = activity.front();
    if (readFields & static_cast<uint32_t>(PhysicsStateFields::All))
        domain.requestedMotion[id] = {generation, result, requested};
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
physics::inxphygpu::InxPhyGpuPreparedRayBatch
PhysicsWorld::RunGpuRays(PhysicsDomain &domain, const std::shared_ptr<const JoltGpuScene::Publication> &publication,
                         const std::vector<physics::inxphygpu::InxPhyGpuRay> &rays,
                         const physics::inxphygpu::InxPhyGpuDeviceRays *device, const PhysicsDeviceRayOutputs *outputs,
                         const Collider *target, uint32_t allHitCapacity) const
{
    using namespace physics::inxphygpu;
    const size_t count = device ? device->count : rays.size();
    std::unique_ptr<PhysicsSceneRayPlan> replacement;
    auto *plan = domain.sceneRays.get();
    InxPhyGpuPreparedRayBatch batch;
    ObserveSpatialGpuQuery(domain, publication, [&](const auto &accepted) {
        std::vector<rhi::ComputeBufferUpdate> uploads;
        if (!plan || !plan->Matches(*accepted) || plan->query.Capacity() < count) {
            const auto capacity = std::max(count, plan ? size_t{plan->query.Capacity()} : size_t{64});
            auto prepared = BuildSceneRayPlan(*m_gpuHost, *accepted, static_cast<uint32_t>(capacity));
            replacement = std::move(prepared.first);
            uploads = std::move(prepared.second);
            plan = replacement.get();
        }
        if (!device) {
            auto &input = uploads.emplace_back(rhi::ComputeBufferUpdate{
                plan->query.Inputs(), 0, std::vector<uint8_t>(rays.size() * sizeof(InxPhyGpuRay))});
            if (target) {
                const auto range =
                    plan->colliderGeometry.find({target->GetComponentID(), target->GetLifetimeGeneration()});
                if (range == plan->colliderGeometry.end())
                    return JoltGpuControlWork{};
                auto ray = rays.front();
                ray.filter = {range->second.first, range->second.second, InxPhyGpuRayTargetGeometry, 0};
                std::memcpy(input.data.data(), &ray, sizeof(ray));
            } else {
                std::memcpy(input.data.data(), rays.data(), input.data.size());
            }
        }
        batch = plan->query.Prepare(*m_gpuHost, ScenePoseSource(*accepted), static_cast<uint32_t>(count), !outputs,
                                    std::move(uploads), device, outputs, allHitCapacity);
        return JoltGpuControlWork{batch.packet};
    });
    if (replacement && batch.packet)
        domain.sceneRays = std::move(replacement);
    return batch;
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
                const auto offset = i * 3;
                ValidateRayInput({origins[offset], origins[offset + 1], origins[offset + 2]},
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

    if (domain.cpuScene) {
        const std::shared_lock queryRead(domain.cpuScene->QueryMutex());
        JoltCpuQueries(*domain.cpuScene)
            .RaycastBatch(origins, directions, count, maximum, hits, mask, {layers, triggers}, profile);
    } else if (const auto gpu = ReadGpuSpatialQueryPublication(domain)) {
        using namespace physics::inxphygpu;
        const auto rays = PackSceneRays(origins, directions, count, maximum, layers, triggers);
        const auto batch = RunGpuRays(domain, gpu, rays, nullptr, nullptr);
        FrameWorkOwner::Scope materialize(*m_gpuWork);
        std::vector<InxPhyGpuRayHit> records(count);
        batch.packet->Readback(0)->ReadInto(records.data(), records.size() * sizeof(InxPhyGpuRayHit));
        for (size_t i = 0; i < count; ++i) {
            mask[i] = DecodeSceneRayHit(*domain.sceneRays, rays[i], records[i], hits[i]);
            if (profile && mask[i])
                ++profile->publishedHits;
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            const auto offset = i * 3;
            ValidateRayInput({origins[offset], origins[offset + 1], origins[offset + 2]},
                             {directions[offset], directions[offset + 1], directions[offset + 2]}, maximum);
        }
        std::fill_n(hits, count, RaycastHit{});
        std::fill_n(mask, count, uint8_t{0});
    }
    if (generation)
        *generation = domain.queryGeneration.load(std::memory_order_acquire);
}

PhysicsRigidbodyBatchSource PhysicsWorld::RaycastInto(const float *origins, const float *directions, size_t count,
                                                      float maximum, const PhysicsDeviceRayOutputs &outputs,
                                                      uint32_t layers, bool triggers) const
{
    if ((count && (!origins || !directions)) || count > UINT32_MAX)
        throw std::invalid_argument("Raycast batch requires valid arrays within the query index range");
    return ExportRaycastBatch(PackSceneRays(origins, directions, count, maximum, layers, triggers), nullptr, outputs);
}

PhysicsRigidbodyBatchSource PhysicsWorld::RaycastInto(const PhysicsDeviceBufferRange &origins,
                                                      const PhysicsDeviceBufferRange &directions, size_t count,
                                                      float maximum, const PhysicsDeviceRayOutputs &outputs,
                                                      uint32_t layers, bool triggers) const
{
    if (count > UINT32_MAX || !std::isfinite(maximum) || maximum <= 0)
        throw std::invalid_argument("GPU raycast requires a valid count and positive finite maximum distance");
    const physics::inxphygpu::InxPhyGpuDeviceRays rays{origins, directions, static_cast<uint32_t>(count), maximum,
                                                       SceneRayFilter(layers, triggers)};
    return ExportRaycastBatch({}, &rays, outputs);
}

PhysicsRigidbodyBatchSource PhysicsWorld::ExportRaycastBatch(const std::vector<physics::inxphygpu::InxPhyGpuRay> &host,
                                                             const physics::inxphygpu::InxPhyGpuDeviceRays *device,
                                                             const PhysicsDeviceRayOutputs &outputs) const
{
    RequireOwnerThread();
    auto &domain = CurrentDomain();
    RequireHealthyStep(domain);
    if (!domain.UsesGpu())
        throw std::invalid_argument("Device raycast output requires a GPU physics scene");
    // The same publication represents an empty scene and produces one miss per ray.
    RequireGpu(domain);
    SealSpatialQuery(domain);
    const auto publication = ReadGpuQueryPublication(domain);
    const auto generation = domain.queryGeneration.load(std::memory_order_acquire);
    if (device ? device->count != 0 : !host.empty())
        (void)RunGpuRays(domain, publication, host, device, &outputs);
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

std::unique_ptr<physics::inxphygpu::InxPhyGpuSceneSetup>
PhysicsWorld::CaptureInxPhyGpuSceneSetup(RuntimeWorld &world,
                                         const physics::inxphygpu::InxPhyGpuFixedPublication *previous) const
{
    using namespace physics::inxphygpu;
    RequireOwnerThread();
    auto &domain = *world.physics;
    RequireHealthyStep(domain);
    const auto &clock = world.simulation;
    const ContactStreamIdentity identity{world.id, clock.simulationEpoch, clock.publishedFixedStepIndex};
    const auto accepted = previous ? previous->FindAcceptedWorld(identity) : std::nullopt;
    if ((domain.stepPending &&
         (domain.solverStarted || domain.fixedGpuPhase != PhysicsDomain::FixedGpuPhase::Solver)) ||
        (previous ? !accepted : identity.fixedStepIndex != 0))
        throw std::logic_error("inxphygpu topology requires the pre-solver boundary and its last native publication");

    const auto &store = PhysicsECSStore::Instance();
    if (store.HasDirtyColliders(world) || store.HasPendingBodyCreations(world) ||
        store.HasPendingBroadphaseAdds(world) || store.HasPendingBroadphaseRemoves(world) ||
        store.HasPendingJointDependencies(world))
        return nullptr;

    auto setup = std::make_unique<InxPhyGpuSceneSetup>();
    auto &registration = setup->world;
    registration.worldId = world.id;
    registration.simulationEpoch = world.simulation.simulationEpoch;
    std::vector<uint32_t> ids(domain.bodyIds.begin(), domain.bodyIds.end());
    std::sort(ids.begin(), ids.end());

    TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    std::vector<InxPhyGpuSceneActor> actors;
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
        auto prepared = PrepareInxPhyGpuSceneActor(std::move(*author));
        rowById.emplace(id, static_cast<uint32_t>(actors.size()));
        registration.bodies.push_back(
            {id, binding.stagedShapeRevision, binding.actor, std::move(prepared.surfaces), bodyTarget});
        actors.push_back(std::move(prepared.actor));
        for (const auto *joint : actor.owner->GetComponents<Joint>())
            if (joint->GetPhysicsConstraintId() != 0 && joint->m_boundBodyId == id)
                activeJoints.push_back(joint);
    }

    std::vector<InxPhyGpuSceneJoint> joints;
    std::vector<uint64_t> jointIds;
    joints.reserve(activeJoints.size());
    jointIds.reserve(activeJoints.size());
    for (const auto *joint : activeJoints) {
        const auto *transform = joint->GetGameObject()->GetTransform();
        const auto anchor =
            glm::conjugate(transform->GetWorldRotation()) * transform->TransformVector(joint->GetAnchor());
        const auto connected = joint->m_boundConnectedBodyId;
        const auto connectedRow = rowById.find(connected);
        if (connected != UINT32_MAX && connectedRow == rowById.end())
            continue; // A suspended endpoint owns no solver row in this topology.
        auto &model =
            joints.emplace_back(InxPhyGpuSceneJoint{rowById.at(joint->m_boundBodyId),
                                                    connected == UINT32_MAX ? NoModelIndex : connectedRow->second,
                                                    anchor,
                                                    joint->GetEnableCollision(),
                                                    {}});
        const auto jointId = joint->GetPhysicsConstraintId();
        jointIds.push_back(jointId);
        model.ownerLocalAnchor = joint->GetAnchor();
        if (accepted)
            model.reference = previous->FindJointReference(*accepted, jointId);
        if (const auto *hinge = dynamic_cast<const HingeJoint *>(joint)) {
            model.settings =
                InxPhyGpuSceneHinge{hinge->GetAxis(),
                                    hinge->GetUseLimits(),
                                    glm::radians(hinge->GetMinimumAngle()),
                                    glm::radians(hinge->GetMaximumAngle()),
                                    {hinge->GetMotorEnabled(), hinge->GetClampedMotorTargetRadians(),
                                     hinge->GetMotorStiffness(), hinge->GetMotorDamping(), hinge->GetMotorMaxTorque(),
                                     hinge->GetJointDamping(), hinge->GetFrictionTorque(), hinge->GetArmature()}};
        } else if (const auto *slider = dynamic_cast<const SliderJoint *>(joint)) {
            model.settings = InxPhyGpuSceneSlider{slider->GetAxis(), slider->GetUseLimits(),
                                                  slider->GetMinimumDistance(), slider->GetMaximumDistance()};
        } else if (const auto *distance = dynamic_cast<const DistanceJoint *>(joint)) {
            auto connectedAnchor = distance->GetConnectedAnchor();
            if (connected != UINT32_MAX) {
                const auto *connectedTransform = store.GetActor(m_bodyActors.at(connected).actor).owner->GetTransform();
                connectedAnchor = glm::conjugate(connectedTransform->GetWorldRotation()) *
                                  connectedTransform->TransformVector(connectedAnchor);
            }
            model.settings = InxPhyGpuSceneDistance{connectedAnchor,
                                                    distance->GetAutoDistance(),
                                                    distance->GetMinimumDistance(),
                                                    distance->GetMaximumDistance(),
                                                    distance->GetStiffness(),
                                                    distance->GetDamping(),
                                                    distance->GetConnectedAnchor()};
        } else if (const auto *swing = dynamic_cast<const SwingTwistJoint *>(joint)) {
            model.settings = swing->m_settings;
        } else if (const auto *six = dynamic_cast<const SixDOFJoint *>(joint)) {
            model.settings = six->m_settings;
        } else if (!dynamic_cast<const FixedJoint *>(joint)) {
            throw std::invalid_argument(std::string("inxphygpu scene setup does not support ") + joint->GetTypeName());
        }
    }

    setup->model = BuildInxPhyGpuSceneModel(actors, joints, *CaptureColliderPairs(domain, true));
    for (uint32_t index = 0; index < jointIds.size(); ++index)
        if (setup->model.jointBindings[index].row != NoModelIndex)
            registration.joints.push_back({jointIds[index], index});
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

PhysicsTargetSource PhysicsWorld::QueueGpuBodyCommand(PhysicsDomain &domain, const PhysicsBodyCommand &command,
                                                      const PhysicsTargetAuthor *author)
{
    PhysicsTargetSource source;
    if (author) {
        auto &batch = domain.gpuTargetAuthors;
        if (!batch)
            batch = std::make_shared<PhysicsTargetBatch>();
        if (batch->records.size() == UINT32_MAX)
            throw std::length_error("GPU physics author command source batch exceeds its index range");
        const auto index = static_cast<uint32_t>(batch->records.size());
        batch->records.push_back(*author);
        source = {batch, index};
    }
    m_gpuScene->QueueBodyCommand(command, source);
    return source;
}

PhysicsTargetSource PhysicsWorld::QueueKinematicBodyCommand(uint32_t id, const PhysicsBodyCommand::KinematicMove &move,
                                                            const PhysicsTargetAuthor *author)
{
    RequireOwnerThread();
    if (!std::isfinite(move.duration) || move.duration <= 0 || !std::isfinite(move.maxSpeed) || move.maxSpeed < 0)
        throw std::invalid_argument("Physics movement requires a positive duration and nonnegative speed limit");
    auto &domain = DomainForBody(id);
    const auto backend = m_bodyActors.at(id).backend;
    if (backend == Backend::StaticBoundary)
        throw std::logic_error("Kinematic movement requires an admitted Rigidbody");
    (void)KinematicBody(ActorForBody(id));
    if (backend == Backend::Cpu) {
        JoltCpuControl::KinematicMove native{std::nullopt, std::nullopt, move.duration, move.maxSpeed};
        if (move.position)
            native.position = CpuPosition(Vector(*move.position));
        if (move.rotation)
            native.rotation = CpuRotation(Rotation(*move.rotation));
        return domain.cpuControl->QueueMove(id, std::move(native), author);
    }
    RequireGpu(domain);
    return QueueGpuBodyCommand(domain, {id, move}, author);
}
void PhysicsWorld::MoveBodyKinematic(uint32_t id, const glm::vec3 &position, const glm::quat &rotation, float dt,
                                     float maxSpeed)
{
    (void)QueueKinematicBodyCommand(id, {position, rotation, dt, maxSpeed});
}
void PhysicsWorld::MoveBodyKinematicPosition(uint32_t id, const glm::vec3 &position, float dt)
{
    (void)QueueKinematicBodyCommand(id, {position, std::nullopt, dt});
}
void PhysicsWorld::MoveBodyKinematicRotation(uint32_t id, const glm::quat &rotation, float dt)
{
    (void)QueueKinematicBodyCommand(id, {std::nullopt, rotation, dt});
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
    PhysicsTargetAuthor author;
    author.rigidbody = PhysicsTargetReference(&body);
    author.ownerTransform = PhysicsTargetReference(body.GetGameObject()->GetTransform());
    author.bodyId = id;
    author.positionRevision = authoredRevision[0];
    author.rotationRevision = authoredRevision[1];
    author.channels = static_cast<uint8_t>(channels);
    return QueueKinematicBodyCommand(id,
                                     {channels & positionChannel ? std::make_optional(position) : std::nullopt,
                                      channels & rotationChannel ? std::make_optional(rotation) : std::nullopt, dt,
                                      maxSpeed},
                                     &author);
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
            QueueGpuBodyCommand(
                domain, {id, PhysicsBodyCommand::KinematicMove{move.position, move.rotation, domain.admittedDeltaTime,
                                                               kMaxTransformDriveSpeed, true}});
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
    if (!std::isfinite(dt) || dt <= 0)
        throw std::invalid_argument("Physics movement requires a positive duration");
    const auto p = Vector(position);
    const auto q = Rotation(rotation);
    const glm::quat targetRotation{q.w, q.x, q.y, q.z};
    const auto &entry = m_bodyActors.at(id);
    if (entry.staticModel) {
        domain.pendingStaticMoves.insert_or_assign(
            id, PhysicsDomain::StaticMove{position, targetRotation, entry.staticModel});
        return;
    }
    if (entry.HasCpuBody())
        domain.cpuControl->QueueTemporaryStaticMove(id, CpuPosition(p), CpuRotation(q), dt, kMaxTransformDriveSpeed);
    if (!entry.HasGpuBody())
        return;
    RequireGpu(domain);
    QueueGpuBodyCommand(
        domain, {id, PhysicsBodyCommand::KinematicMove{position, targetRotation, dt, kMaxTransformDriveSpeed, true}});
}
bool PhysicsWorld::IsBodySleeping(uint32_t id) const
{
    RequireOwnerThread();
    auto &domain = DomainForBody(id);
    if (m_bodyActors.at(id).backend == Backend::StaticBoundary)
        return true;
    bool sleeping = false;
    (void)ReadMotionState(domain, id, PhysicsStateFields::None, &sleeping);
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
    QueueGpuBodyCommand(domain, {id, PhysicsBodyCommand::Activity{true}});
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
    QueueGpuBodyCommand(domain, {id, PhysicsBodyCommand::Activity{false}});
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
    RequireHealthyStep(domain);
    (void)Vector(positionA);
    (void)Vector(positionB);
    const auto qA = Rotation(rotationA), qB = Rotation(rotationB);
    const TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    const auto first = a.CreateJoltShape(), second = b.CreateJoltShape();
    if (!first || !second)
        throw std::runtime_error("Collider geometry is not ready for an explicit geometry query");
    using namespace physics::inxphygpu;
    const auto words = CookInxPhyGpuPair({*first, positionA, {qA.w, qA.x, qA.y, qA.z}},
                                         {*second, positionB, {qB.w, qB.x, qB.y, qB.z}});
    if (words.empty())
        return std::nullopt;
    if (!m_gpuHost || !m_gpuWork)
        throw std::runtime_error("GPU physics requires engine compute services");
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    auto &host = *m_gpuHost;
    if (!m_gpuPairQuery)
        m_gpuPairQuery = std::make_unique<InxPhyGpuPairQuery>(host, PhysicsShaderTarget(host));
    auto query = m_gpuPairQuery->Prepare(host, words);
    (void)query->Submit();
    std::array<ModelVector, 3> hit;
    query->Readback(0)->ReadInto(hit.data(), sizeof(hit));
    if (hit[0][3] < 0.0f)
        throw std::runtime_error("inxphygpu could not resolve collider penetration");
    if (hit[0][3] == 0.0f)
        return std::nullopt;
    const auto vector = [](const auto &row) { return glm::vec3(row[0], row[1], row[2]); };
    return PhysicsPenetrationResult{vector(hit[0]), hit[0][3], positionA + vector(hit[1]), positionA + vector(hit[2])};
}
bool PhysicsWorld::RaycastCollider(const Collider &collider, const glm::vec3 &origin, const glm::vec3 &direction,
                                   float maximum, RaycastHit &hit) const
{
    auto &domain = DomainForComponent(collider);
    ValidateRayInput(origin, direction, maximum);
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
    RunGpuPublicationRead(domain, {});
    const auto publication = ReadGpuSpatialQueryPublication(domain);
    if (!publication)
        return false;
    const float coordinates[3] = {origin.x, origin.y, origin.z};
    const float axes[3] = {direction.x, direction.y, direction.z};
    const auto rays = PackSceneRays(coordinates, axes, 1, maximum, UINT32_MAX, true);
    const auto batch = RunGpuRays(domain, publication, rays, nullptr, nullptr, &collider);
    if (!batch.packet)
        return false;
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    physics::inxphygpu::InxPhyGpuRayHit record;
    batch.packet->Readback(0)->ReadInto(&record, sizeof(record));
    return DecodeSceneRayHit(*domain.sceneRays, rays.front(), record, hit);
}

glm::vec3 PhysicsWorld::ClosestPointOnCollider(const Collider &collider, const glm::vec3 &point) const
{
    auto &domain = DomainForComponent(collider);
    (void)Vector(point);
    if (!collider.IsEnabled() || !collider.GetGameObject() || !collider.GetGameObject()->IsActiveInHierarchy())
        return point;
    SealSpatialQuery(domain);
    const auto id = ColliderBodyIdentity(collider);
    if (!FindBodyRegistration(id))
        throw std::runtime_error("Active Collider has no accepted geometry for a closest-point query");
    if (domain.cpuScene) {
        const auto closest =
            JoltCpuQueries(*domain.cpuScene)
                .ClosestPoint(id, {collider.GetComponentID(), collider.GetLifetimeGeneration()}, point);
        if (!closest)
            throw std::runtime_error("Active Collider has no accepted geometry for a closest-point query");
        return *closest;
    }
    const auto publication = ReadGpuSpatialQueryPublication(domain);
    if (!publication)
        throw std::runtime_error("Active Collider has no accepted geometry for a closest-point query");
    const auto *mesh = dynamic_cast<const MeshCollider *>(&collider);
    const auto key = std::make_pair(collider.GetComponentID(), collider.GetLifetimeGeneration());
    std::unique_ptr<PhysicsSceneRayPlan> replacement;
    auto *plan = domain.sceneRays.get();
    std::shared_ptr<rhi::PreparedComputeBatch> result;
    ObserveSpatialGpuQuery(domain, publication, [&](const auto &accepted) {
        std::vector<rhi::ComputeBufferUpdate> uploads;
        if (!plan || !plan->Matches(*accepted)) {
            auto prepared = BuildSceneRayPlan(*m_gpuHost, *accepted, plan ? plan->query.Capacity() : 64);
            replacement = std::move(prepared.first);
            uploads = std::move(prepared.second);
            plan = replacement.get();
        }
        const auto range = plan->colliderGeometry.find(key);
        if (range == plan->colliderGeometry.end())
            return JoltGpuControlWork{};
        result = plan->query.PrepareClosestPoint(
            *m_gpuHost, ScenePoseSource(*accepted), {point.x, point.y, point.z, 0.0f}, range->second.first,
            range->second.second, mesh && mesh->IsPublishedMeshClosed(), std::move(uploads));
        return JoltGpuControlWork{result};
    });
    if (!result)
        throw std::runtime_error("Active Collider has no accepted geometry for a closest-point query");
    if (replacement)
        domain.sceneRays = std::move(replacement);
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    physics::inxphygpu::ModelVector record;
    result->Readback(0)->ReadInto(record.data(), sizeof(record));
    if (record[3] < 0.0f)
        throw std::runtime_error("Active Collider has no accepted geometry for a closest-point query");
    return {record[0], record[1], record[2]};
}

std::vector<RaycastHit> PhysicsWorld::RaycastAll(const glm::vec3 &origin, const glm::vec3 &direction, float maximum,
                                                 uint32_t layers, bool triggers) const
{
    auto &domain = CurrentDomain();
    ValidateRayInput(origin, direction, maximum);
    SealSpatialQuery(domain);
    if (domain.cpuScene)
        return JoltCpuQueries(*domain.cpuScene).RaycastAll(origin, direction, maximum, {layers, triggers});
    const auto gpu = ReadGpuSpatialQueryPublication(domain);
    std::vector<RaycastHit> hits;
    if (!gpu)
        return hits;
    const float coordinates[3] = {origin.x, origin.y, origin.z};
    const float axes[3] = {direction.x, direction.y, direction.z};
    const auto rays = PackSceneRays(coordinates, axes, 1, maximum, layers, triggers);
    using physics::inxphygpu::InxPhyGpuRayHit;
    const auto read = [&](uint32_t capacity) {
        const auto batch = RunGpuRays(domain, gpu, rays, nullptr, nullptr, nullptr, capacity);
        std::vector<InxPhyGpuRayHit> rows(size_t{capacity} + 1);
        FrameWorkOwner::Scope materialize(*m_gpuWork);
        batch.packet->Readback(0)->ReadInto(rows.data(), rows.size() * sizeof(InxPhyGpuRayHit));
        return rows;
    };
    // Read a short result prefix. Dense intersections expand to the exact size
    // against the same retained poses, regardless of the scene's triangle count.
    constexpr uint32_t capacity = 32;
    auto rows = read(capacity);
    const uint32_t count = rows.front().identity[0];
    if (count > capacity)
        rows = read(count);
    hits.resize(count);
    for (size_t i = 0; i < hits.size(); ++i)
        (void)DecodeSceneRayHit(*domain.sceneRays, rays.front(), rows[i + 1], hits[i]);
    std::sort(hits.begin(), hits.end(), [](const RaycastHit &a, const RaycastHit &b) {
        return std::tie(a.distance, a.bodyId, a.subShapeId) < std::tie(b.distance, b.bodyId, b.subShapeId);
    });
    return hits;
}

std::vector<Collider *> PhysicsWorld::OverlapSphere(const glm::vec3 &center, float radius, uint32_t layers,
                                                    bool triggers) const
{
    auto &domain = CurrentDomain();
    return QueryColliders(domain, JoltCpuQueries::Sphere{center, radius}, layers, triggers);
}
std::vector<Collider *> PhysicsWorld::OverlapBox(const glm::vec3 &center, const glm::vec3 &halfExtents,
                                                 const glm::quat &rotation, uint32_t layers, bool triggers) const
{
    auto &domain = CurrentDomain();
    return QueryColliders(domain, JoltCpuQueries::Box{center, halfExtents, rotation}, layers, triggers);
}
std::vector<Collider *> PhysicsWorld::OverlapCapsule(const glm::vec3 &a, const glm::vec3 &b, float radius,
                                                     uint32_t layers, bool triggers) const
{
    auto &domain = CurrentDomain();
    return QueryColliders(domain, JoltCpuQueries::Capsule{a, b, radius}, layers, triggers);
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
    if (!gpu || !layers)
        return {};
    std::unique_ptr<PhysicsSceneRayPlan> replacement;
    auto *plan = domain.sceneRays.get();
    physics::inxphygpu::InxPhyGpuPreparedBodyBounds query;
    ObserveSpatialGpuQuery(domain, gpu, [&](const auto &accepted) {
        std::vector<rhi::ComputeBufferUpdate> uploads;
        if (!plan || !plan->Matches(*accepted)) {
            auto prepared = BuildSceneRayPlan(*m_gpuHost, *accepted, plan ? plan->query.Capacity() : 64);
            replacement = std::move(prepared.first);
            uploads = std::move(prepared.second);
            plan = replacement.get();
        }
        query = plan->query.PrepareBodyBounds(
            *m_gpuHost, ScenePoseSource(*accepted), {minimum.x, minimum.y, minimum.z, 0.0f},
            {maximum.x, maximum.y, maximum.z, 0.0f}, layers, triggers, std::move(uploads));
        return JoltGpuControlWork{query.packet};
    });
    if (!query.packet)
        return {};
    if (replacement)
        domain.sceneRays = std::move(replacement);
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    auto &store = PhysicsECSStore::Instance();
    std::vector<Rigidbody *> bodies;
    for (const auto row : query.ReadMatchingRows()) {
        const auto &identity = *gpu->directory->actors[gpu->bodyBase + row];
        if (store.IsValid(identity.rigidbody))
            bodies.push_back(store.GetRigidbody(identity.rigidbody).owner);
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
    return QueryClosestCast(CurrentDomain(), JoltCpuQueries::Sphere{center, radius}, direction, maximum, hit, layers,
                            triggers);
}
bool PhysicsWorld::BoxCast(const glm::vec3 &center, const glm::vec3 &halfExtents, const glm::vec3 &direction,
                           const glm::quat &rotation, float maximum, RaycastHit &hit, uint32_t layers,
                           bool triggers) const
{
    return QueryClosestCast(CurrentDomain(), JoltCpuQueries::Box{center, halfExtents, rotation}, direction, maximum,
                            hit, layers, triggers);
}
bool PhysicsWorld::SweepBox(const BoxCollider &box, const glm::vec3 &position, const glm::quat &rotation,
                            const glm::vec3 &displacement, RaycastHit &hit) const
{
    auto &domain = DomainForComponent(box);
    RequireOwnerThread();
    (void)Vector(position);
    (void)Rotation(rotation);
    (void)Vector(displacement);
    const double distance = std::hypot(double(displacement.x), double(displacement.y), double(displacement.z));
    hit = {};
    const auto *object = box.GetGameObject();
    if (!object || distance == 0)
        return false;
    if (distance > std::numeric_limits<float>::max())
        throw std::invalid_argument("BoxCollider sweep displacement exceeds device coordinates");
    SealSpatialQuery(domain);
    if (domain.cpuScene)
        return JoltCpuQueries(*domain.cpuScene).SweepBox(box, position, rotation, displacement, hit);
    const auto layers = TagLayerManager::Instance().GetLayerCollisionMask(object->GetLayer());
    const auto volume = CookBoxSweep(box, position, rotation, layers);
    return volume && RunGpuShapeCast(domain, *volume, displacement, static_cast<float>(distance), hit, &box);
}

bool PhysicsWorld::CapsuleCast(const glm::vec3 &a, const glm::vec3 &b, float radius, const glm::vec3 &direction,
                               float maximum, RaycastHit &hit, uint32_t layers, bool triggers) const
{
    return QueryClosestCast(CurrentDomain(), JoltCpuQueries::Capsule{a, b, radius}, direction, maximum, hit, layers,
                            triggers);
}
std::shared_ptr<const PhysicsShapeGeometry>
PhysicsWorld::ResolveShapeGeometry(PhysicsDomain &domain, const JoltGpuScene::PublishedLeaf &leaf) const
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
    if (leaf.shapeType == ShapeBox) {
        geometry->type = PhysicsShapeType::Box;
        geometry->dimensions = {leaf.dimensions.x * 2, leaf.dimensions.y * 2, leaf.dimensions.z * 2, leaf.dimensions.w};
    } else if (leaf.shapeType == ShapeSphere) {
        geometry->type = PhysicsShapeType::Sphere;
        geometry->dimensions = {leaf.dimensions.w, 0, 0, 0};
    } else if (leaf.shapeType == ShapeCapsule) {
        geometry->type = PhysicsShapeType::Capsule;
        geometry->dimensions = {leaf.dimensions.w, 2 * (leaf.dimensions.y + leaf.dimensions.w), 1, 0};
    } else if (leaf.shapeType == ShapeCylinder || leaf.shapeType == ShapeConvexHull) {
        // The existing immutable convex mesh view exposes the cooked native surface.
        // This adapter describes the accepted GPU operand without reading its numerical state.
        geometry->type = PhysicsShapeType::Mesh;
    } else {
        throw std::runtime_error("GPU shape state has no immutable geometry adapter");
    }
    domain.shapeGeometry[shape] = geometry;
    return geometry;
}

std::vector<PhysicsShapeState> PhysicsWorld::GetBodyShapeStates(uint32_t id, bool triggers) const
{
    RequireOwnerThread();
    auto &domain = DomainForBody(id);
    if (ReadBodyBackend(domain, id) == Backend::Cpu) {
        RunCpuControl(domain);
        return JoltCpuQueries(*domain.cpuScene).GatherBodyShapes(id, triggers);
    }

    PhysicsGpuMotionRead motion;
    std::shared_ptr<const JoltGpuScene::PublishedActor> actor;
    std::vector<const JoltGpuScene::PublishedLeaf *> leaves;
    std::exception_ptr error;
    RunGpuPublicationRead(domain, [&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        return PrepareJoltGpuQueryWork(
            [&]() -> JoltGpuControlWork {
                const auto slot = publication->directory->FindBodySlot(id);
                if (slot < publication->bodyBase || slot - publication->bodyBase >= publication->bodyCount)
                    throw JoltGpuQueryRequestError("Shape gathering requires an accepted body identity");
                if (!publication->directory->residentBodies[slot])
                    return {};
                actor = publication->directory->actors[slot];
                for (const auto &leaf : actor->leaves)
                    if (triggers || !leaf.sensor)
                        leaves.push_back(&leaf);
                if (leaves.empty())
                    return {};
                constexpr auto fields =
                    static_cast<PhysicsStateFields>(static_cast<uint32_t>(PhysicsStateFields::CenterOfMass) |
                                                    static_cast<uint32_t>(PhysicsStateFields::Rotation) |
                                                    static_cast<uint32_t>(PhysicsStateFields::LinearVelocity) |
                                                    static_cast<uint32_t>(PhysicsStateFields::AngularVelocity));
                motion = PhysicsGpuMotionRead::Prepare(*m_gpuHost, ScenePoseSource(*publication),
                                                       {SceneMotionBinding(*publication, *actor)}, fields);
                return motion.packet;
            },
            error);
    });
    if (error)
        std::rethrow_exception(error);
    if (leaves.empty())
        return {};

    FrameWorkOwner::Scope materialize(*m_gpuWork);
    const auto accepted = motion.Read().front();
    std::vector<PhysicsShapeState> result;
    result.reserve(leaves.size());
    for (const auto *leaf : leaves) {
        PhysicsShapeState state;
        state.collider = leaf->surfaceTarget.GetCollider();
        state.memberIndex = leaf->memberIndex;
        state.layerBit = 1u << leaf->layer;
        state.trigger = leaf->sensor;
        state.moving = actor->acceptedMotionType != MotionStatic;
        state.geometry = ResolveShapeGeometry(domain, *leaf);
        state.center = accepted.centerOfMass + accepted.rotation * leaf->centerOfMassOffset;
        state.rotation = glm::normalize(accepted.rotation * leaf->localRotation);
        state.bodyCenterOfMass = accepted.centerOfMass;
        state.linearVelocity = accepted.linearVelocity;
        state.angularVelocity = accepted.angularVelocity;
        state.friction = leaf->friction;
        state.bounciness = leaf->bounciness;
        state.frictionCombine = leaf->frictionCombine;
        state.bounceCombine = leaf->bounceCombine;
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
    using namespace physics::inxphygpu;
    auto *plan = domain.sceneRays.get();
    std::unique_ptr<PhysicsSceneRayPlan> replacement;
    std::shared_ptr<InxPhyGpuParticleExport::Work> work;
    const auto prepareCollisionExport = [&](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        std::vector<rhi::ComputeBufferUpdate> uploads;
        if (!plan || !plan->Matches(*publication) || !plan->particleSurfaces) {
            auto prepared = BuildSceneRayPlan(*m_gpuHost, *publication, plan ? plan->query.Capacity() : 64, true);
            replacement = std::move(prepared.first);
            uploads = std::move(prepared.second);
            plan = replacement.get();
        }
        if (!m_gpuParticleExport)
            m_gpuParticleExport =
                std::make_unique<InxPhyGpuParticleExport>(*m_gpuHost, PhysicsShaderTarget(*m_gpuHost));
        const InxPhyGpuParticleExport::Source source{
            plan->query.ParticleGeometry(),
            ScenePoseSource(*publication),
            *plan->particleSurfaces,
            {publication->worldId, publication->simulationEpoch, publication->fixedStepIndex},
            publication->directory};
        work = std::make_shared<InxPhyGpuParticleExport::Work>(m_gpuParticleExport->Prepare(
            *m_gpuHost, source, &scene, previous, revision, std::move(uploads), plan->query.TakeInitialization()));
        return JoltGpuControlWork{work->packet,
                                  [this, work] { return JoltGpuControlWork{work->PrepareGrid(*m_gpuHost)}; }};
    };
    RunGpuPublicationRead(domain, prepareCollisionExport, true);
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    auto result = work->Finish();
    if (replacement)
        domain.sceneRays = std::move(replacement);
    return result;
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
    m_gpuPublicationReader->Run(std::move(prepared));
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
