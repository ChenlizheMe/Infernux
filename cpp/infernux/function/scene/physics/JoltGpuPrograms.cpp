#include "JoltGpuPrograms.h"
#include <Jolt/Jolt.h>
#include <Jolt/Geometry/AABox.h>
#include "JoltGpuArticulations.h"
#include "JoltGpuGeometryStore.h"
#include "JoltGpuProgramSources.h"
#include "JoltGpuSleep.h"
#include "JoltGpuSleepLayout.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace infernux
{
namespace
{
using namespace JPH::GPU;

// The low-level native Model path has no Collider/root Shape owner. Its author
// geometry is exactly the supplied cooked leaf stream. Build that stream's
// root-local bounds here, without reading/integrating any device motion.
// The Scene path later supplies the original final native root bounds instead.
std::vector<BodySleepRecord> InitialSleepAuthors(const JoltGpuModel &model,
                                                 const std::shared_ptr<const JoltGpuGeometryLease> &geometry)
{
    std::vector<BodySleepRecord> result(model.bodies.size());
    for (size_t body = 0; body < model.bodies.size(); ++body) {
        const auto &owner = model.bodies[body];
        JPH::AABox bounds;
        bool sensor = owner.meta.y != 0u;
        for (uint32_t index = 0; index < owner.meta.y; ++index) {
            const auto &leaf = model.shapes.at(owner.meta.x + index);
            const auto &g = leaf.geometry;
            const JPH::Quat rotation(leaf.localRotation.x, leaf.localRotation.y, leaf.localRotation.z,
                                     leaf.localRotation.w);
            JPH::Vec3 center(leaf.localPositionCOM.x, leaf.localPositionCOM.y, leaf.localPositionCOM.z), extent;
            sensor &= (leaf.meta.w & ShapeSensor) != 0u;
            if (leaf.meta.x == ShapeSphere)
                extent = JPH::Vec3::sReplicate(g.w);
            else if (leaf.meta.x == ShapeCapsule || leaf.meta.x == ShapeCylinder) {
                const auto axis = rotation * JPH::Vec3::sAxisY();
                const auto absolute = axis.Abs();
                if (leaf.meta.x == ShapeCapsule)
                    extent = absolute * g.y + JPH::Vec3::sReplicate(g.w);
                else
                    extent = absolute * g.y +
                             JPH::Vec3::sMax(JPH::Vec3::sZero(), JPH::Vec3::sReplicate(1) - axis * axis).Sqrt() * g.x;
            } else {
                JPH::AABox local;
                if (leaf.meta.x == ShapeBox)
                    local = JPH::AABox(-JPH::Vec3(g.x, g.y, g.z), JPH::Vec3(g.x, g.y, g.z));
                else if (leaf.meta.x == ShapeConvexHull) {
                    std::shared_ptr<const JoltGpuGeometryHandle> record;
                    if (geometry)
                        for (const auto &candidate : geometry->records)
                            if (std::memcmp(&candidate->reference, &leaf.geometryReference, sizeof(UInt4)) == 0) {
                                record = candidate;
                                break;
                            }
                    if (!record || !record->record || !record->record->shape)
                        throw std::invalid_argument(
                            "Native Hull model initialization requires its accepted retained geometry record");
                    local = record->record->shape->GetLocalBounds();
                } else
                    throw std::invalid_argument("Native sleep author bounds require a supported cooked leaf type");
                const auto transformed = local.Transformed(JPH::Mat44::sRotationTranslation(rotation, center));
                center = transformed.GetCenter();
                extent = transformed.GetExtent();
            }
            bounds.Encapsulate(center - extent);
            bounds.Encapsulate(center + extent);
        }
        if (!owner.meta.y)
            bounds = JPH::AABox(JPH::Vec3::sZero(), JPH::Vec3::sZero());
        const auto extent = bounds.GetExtent(), center = bounds.GetCenter();
        auto &author = result[body].author;
        if (!bounds.IsValid() || !std::isfinite(extent.GetX()) || !std::isfinite(extent.GetY()) ||
            !std::isfinite(extent.GetZ()) || !std::isfinite(center.GetX()) || !std::isfinite(center.GetY()) ||
            !std::isfinite(center.GetZ()))
            throw std::invalid_argument("Native cooked model sleep bounds are not finite");
        author.rootHalfExtent = {extent.GetX(), extent.GetY(), extent.GetZ(), 0};
        author.rootBoundsCenter = {center.GetX(), center.GetY(), center.GetZ(), 0};
        author.policy = {1u | (sensor ? 2u : 0u), 0u, 0u, 0u};
    }
    return result;
}

struct Port
{
    Resource resource;
    std::string_view name;
    uint64_t stride;
    std::array<std::string_view, 8> fields;
    uint32_t integerFields;
};

constexpr Port Ports[] = {
    {Resource::StepParams,
     "step_params",
     sizeof(StepParams),
     {"counts", "solve", "gravityDt", "contact", "restitution", "selection", "selectedCounts"},
     0x63},
    {Resource::BodyModels,
     "body_models",
     sizeof(BodyModel),
     {"meta", "mass", "invInertiaDiagonal", "inertiaRotation", "damping"},
     1},
    {Resource::ShapeModels,
     "shape_models",
     sizeof(ShapeModel),
     {"meta", "geometry", "localPositionCOM", "localRotation", "material", "surface", "source", "geometryReference"},
     (1u << 0) | (1u << 5) | (1u << 6) | (1u << 7)},
    {Resource::Worlds, "worlds", sizeof(WorldRange), {"bodies", "leaves"}, 3},
    {Resource::BodyForces, "body_forces", sizeof(BodyForces), {"force", "torque"}, 0},
    {Resource::BodyStates,
     "body_states",
     sizeof(BodyState),
     {"positionCOM", "rotation", "linearVelocity", "angularVelocity"},
     0},
    {Resource::Bounds, "bounds", sizeof(Aabb), {"minimum", "maximum"}, 0},
    {Resource::Contacts,
     "contacts",
     sizeof(Contact),
     {"meta", "normalDepth", "localAnchorA", "localAnchorB", "impulse"},
     1},
    {Resource::KinematicTargets,
     "kinematic_targets",
     sizeof(KinematicTarget),
     {"positionSpeed", "rotation", "rootCOMDt", "state"},
     1u << 3},
    {Resource::ContactPatches, "contact_patches", sizeof(ContactPatch), {"leaves"}, 1},
    {Resource::Joints, "joint_stream", sizeof(QueryRecordWord), {"value"}, 1},
    {Resource::SleepStates, "body_sleep", sizeof(BodySleepRecord), {"author", "runtime"}, 1},
    {Resource::StageWork, "stage_work", sizeof(StageWork), {"range"}, 1},
    {Resource::ConstraintSchedule, "constraint_schedule", sizeof(ConstraintScheduleWord), {"value"}, 1},
    {Resource::ConvexGeometry, "convex_geometry", sizeof(UInt4), {}, 1},
    {Resource::ArticulationControl, "articulation_control", sizeof(QueryRecordWord), {"value"}, 1},
    {Resource::ContactHistory, "contact_history", sizeof(QueryRecordWord), {"value"}, 1},
    {Resource::SweepBodies,
     "sweep_bodies",
     sizeof(SweepBody),
     {"positionCOM", "rotation", "linearVelocity", "angularVelocity", "displacement", "linearResponse",
      "angularResponse"},
     0},
};

constexpr size_t PortCount = sizeof(Ports) / sizeof(Ports[0]);
constexpr size_t StoragePortCount = 12;
constexpr size_t WorkUniformCacheCapacity = 64;
struct BodyBatchDefinition
{
    Stage stage;
    const char *entry;
};
constexpr BodyBatchDefinition BodyBatchDefinitions[] = {
    {Stage::PrepareKinematics, "body_batch_prepare_kinematics"},
    {Stage::ApplyForces, "body_batch_apply_forces"},
    {Stage::Integrate, "body_batch_integrate"},
};
constexpr std::array<std::string_view, 3> BodyBatchInputNames = {"body_batch_header", "body_batch_selections",
                                                                 "body_batch_tiles"};
constexpr size_t StageCount = sizeof(Stages) / sizeof(Stages[0]);

constexpr size_t PortIndex(Resource resource)
{
    switch (resource) {
    case Resource::StepParams:
        return 0;
    case Resource::BodyModels:
        return 1;
    case Resource::ShapeModels:
        return 2;
    case Resource::Worlds:
        return 3;
    case Resource::BodyForces:
        return 4;
    case Resource::BodyStates:
        return 5;
    case Resource::Bounds:
        return 6;
    case Resource::Contacts:
        return 7;
    case Resource::KinematicTargets:
        return 8;
    case Resource::ContactPatches:
        return 9;
    case Resource::Joints:
        return 10;
    case Resource::StageWork:
        return 12;
    case Resource::ConstraintSchedule:
        return 13;
    case Resource::ConvexGeometry:
        return 14;
    case Resource::ArticulationControl:
        return 15;
    case Resource::ContactHistory:
        return 16;
    case Resource::SweepBodies:
        return 17;
    case Resource::SleepStates:
        return 11;
    }
    return PortCount;
}

constexpr size_t InputIndex(Resource resource)
{
    switch (resource) {
    case Resource::StepParams:
        return 0;
    case Resource::BodyModels:
        return 1;
    case Resource::ShapeModels:
        return 2;
    case Resource::Worlds:
        return 3;
    case Resource::BodyForces:
        return 4;
    default:
        return 5;
    }
}

std::shared_ptr<rhi::ComputeBuffer> Allocate(rhi::ComputeHost &host, size_t port, uint64_t bytes)
{
    if (!bytes)
        return {};
    const bool mixed = Ports[port].integerFields != 0;
    return std::make_shared<rhi::ComputeBuffer>(
        host, rhi::ComputeBufferDesc{bytes / (mixed ? sizeof(uint32_t) : sizeof(Float4)),
                                     mixed ? rhi::ComputeScalarType::UInt32 : rhi::ComputeScalarType::Float32,
                                     static_cast<uint8_t>(mixed ? 1 : 4)});
}

// These stage bodies initialize every slot. Access qualifiers alone do not
// establish complete stores: all read/write solvers retain their input proof.
constexpr uint32_t CompleteWrites(Stage stage)
{
    switch (stage) {
    case Stage::BuildBounds:
    case Stage::BuildSweptBounds:
        return Mask(Resource::Bounds);
    case Stage::CaptureSweepMotion:
        return Mask(Resource::SweepBodies);
    case Stage::CollidePairs:
        return Mask(Resource::Contacts);
    case Stage::StoreContactHistory:
        return Mask(Resource::ContactHistory);
    default:
        return 0;
    }
}

constexpr bool IndexedDefinitions()
{
    for (size_t slot = 0; slot < PortCount; ++slot)
        if (Mask(Ports[slot].resource) != (1u << slot) || PortIndex(Ports[slot].resource) != slot)
            return false;
    for (size_t slot = 0; slot < StageCount; ++slot) {
        const auto &stage = Stages[slot];
        if (static_cast<size_t>(stage.stage) != slot || stage.groupSizeX == 0 || stage.groupSizeY == 0 ||
            stage.groupSizeZ == 0 || (CompleteWrites(stage.stage) & ~stage.writes) != 0 ||
            (CompleteWrites(stage.stage) & stage.reads) != 0 || !(stage.reads & Mask(Resource::StageWork)) ||
            (stage.writes & Mask(Resource::StageWork)))
            return false;
    }
    return true;
}

static_assert(PortCount == StoragePortCount + 6);
static_assert(IndexedDefinitions(), "Jolt GPU resources/stages must retain their compact index contract");

bool Finite3(const Float4 &value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

bool Finite4(const Float4 &value)
{
    return Finite3(value) && std::isfinite(value.w);
}

bool Normalized(const Float4 &value)
{
    return Finite4(value) &&
           std::abs(value.x * value.x + value.y * value.y + value.z * value.z + value.w * value.w - 1.0f) <= 1e-4f;
}

bool Zero3(const Float4 &value)
{
    return value.x == 0.0f && value.y == 0.0f && value.z == 0.0f;
}

struct ModelCounts
{
    uint32_t bodies, worlds, shapes, maxBodiesPerWorld = 0, maxLeavesPerWorld = 0;
};

bool Sweeps(const BodyModel &body)
{
    return body.meta.z == MotionDynamic && body.meta.y && body.invInertiaDiagonal.w != 0.0f;
}

bool SweepsDynamicPairs(const BodyModel &body)
{
    return Sweeps(body) && body.invInertiaDiagonal.w < 0.0f;
}

template <class Change, class Record>
const Record &ModelRecord(const std::vector<Record> &records, const std::vector<Change> &changes, uint32_t index)
{
    const auto change = std::lower_bound(changes.begin(), changes.end(), index,
                                         [](const auto &value, uint32_t key) { return value.index < key; });
    return change != changes.end() && change->index == index ? change->model : records[index];
}

ModelCounts ValidateModel(const JoltGpuModel &model, const std::vector<JoltGpuBodyModelUpdate> &bodyChanges = {},
                          const std::vector<JoltGpuShapeModelUpdate> &shapeChanges = {},
                          const std::vector<JoltGpuJointModelUpdate> &jointChanges = {},
                          const JoltGpuGeometryStore *geometryStore = nullptr)
{
    constexpr auto limit = std::numeric_limits<uint32_t>::max();
    if (model.bodies.size() > limit || model.worlds.size() > limit || model.shapes.size() > limit ||
        model.joints.size() > (limit - 1u) / JointDirectoryWords)
        throw std::invalid_argument("Jolt GPU model exceeds the device index range");
    ModelCounts counts{static_cast<uint32_t>(model.bodies.size()), static_cast<uint32_t>(model.worlds.size()),
                       static_cast<uint32_t>(model.shapes.size())};
    uint64_t bodyEnd = 0, leafEnd = 0, jointEnd = 0;
    for (size_t world = 0; world < model.worlds.size(); ++world) {
        const auto range = model.worlds[world].bodies;
        const auto leaves = model.worlds[world].leaves;
        if (range.x != bodyEnd || leaves.x != leafEnd || range.z || range.w || bodyEnd + range.y > counts.bodies ||
            leafEnd + leaves.y > counts.shapes || leaves.z != jointEnd || jointEnd + leaves.w > model.joints.size())
            throw std::invalid_argument("Jolt GPU model requires contiguous author ranges and zero contact fields");
        uint64_t bodyLeafEnd = leafEnd;
        for (uint32_t body = range.x; body < bodyEnd + range.y; ++body) {
            const auto &record = ModelRecord(model.bodies, bodyChanges, body);
            const auto motionType = record.meta.z & ~MotionNotResident;
            if (record.meta.w != world || record.meta.x != bodyLeafEnd ||
                bodyLeafEnd + record.meta.y > leafEnd + leaves.y ||
                (motionType != MotionStatic && motionType != MotionKinematic && motionType != MotionDynamic))
                throw std::invalid_argument("Invalid body or leaf range in the Jolt GPU model");
            const bool nonstatic = motionType != MotionStatic;
            const bool finiteMass = record.mass.x > 0 && Finite3(record.invInertiaDiagonal) &&
                                    record.invInertiaDiagonal.x > 0 && record.invInertiaDiagonal.y > 0 &&
                                    record.invInertiaDiagonal.z > 0;
            const bool prescribed = motionType == MotionKinematic && record.mass.x == 0 &&
                                    record.invInertiaDiagonal.x == 0 && record.invInertiaDiagonal.y == 0 &&
                                    record.invInertiaDiagonal.z == 0;
            if (!Finite4(record.mass) || record.mass.y < 0 || record.mass.y > 7 ||
                std::floor(record.mass.y) != record.mass.y || record.mass.z < 0 || record.mass.z > 7 ||
                std::floor(record.mass.z) != record.mass.z ||
                (nonstatic ? (!finiteMass && !prescribed) : record.mass.x != 0) || !Finite4(record.damping) ||
                record.damping.x < 0 || record.damping.y < 0 || record.damping.z < 0 || record.damping.w < 0 ||
                !std::isfinite(record.invInertiaDiagonal.w) || (nonstatic && !Normalized(record.inertiaRotation)))
                throw std::invalid_argument("Jolt GPU body has invalid mass, inertia or damping");
            for (uint32_t leaf = record.meta.x; leaf < bodyLeafEnd + record.meta.y; ++leaf) {
                const auto &shape = ModelRecord(model.shapes, shapeChanges, leaf);
                const bool triangle = (shape.meta.w & ShapeOneSidedTriangle) != 0u;
                const uint32_t triangleFlags = ShapeOneSidedTriangle | (7u << ShapeTriangleActiveEdgeShift);
                if (shape.meta.y != body || (shape.meta.w & ~(ShapeSensor | triangleFlags)) ||
                    (!triangle && (shape.meta.w & triangleFlags)) || (triangle && shape.meta.x != ShapeConvexHull) ||
                    shape.surface.x >= 32 || shape.surface.z > MaterialCombineMaximum ||
                    shape.surface.w > MaterialCombineMaximum)
                    throw std::invalid_argument("Jolt GPU leaf identity, flags or material mode is invalid");
                if (!Finite4(shape.localPositionCOM) || !Normalized(shape.localRotation) || !Finite4(shape.material) ||
                    shape.material.x < 0 || shape.material.y < 0 || shape.material.y > 1 ||
                    (!triangle && (shape.localPositionCOM.w != 0 || shape.material.z != 0 || shape.material.w != 0)))
                    throw std::invalid_argument("Jolt GPU leaf pose or material is invalid");
                const auto geometry = shape.geometry;
                const auto reference = shape.geometryReference;
                const auto cold =
                    shape.meta.x == ShapeConvexHull && geometryStore ? geometryStore->Find(reference) : nullptr;
                if (reference.z || reference.w || (shape.meta.x != ShapeConvexHull && (reference.x || reference.y)) ||
                    (shape.meta.x == ShapeConvexHull && !cold))
                    throw std::invalid_argument("Jolt GPU leaf has no exact registered immutable geometry range");
                switch (shape.meta.x) {
                case ShapeSphere:
                    if (!std::isfinite(geometry.w) || geometry.w <= 0)
                        throw std::invalid_argument("Jolt GPU sphere radius must be positive and finite");
                    break;
                case ShapeBox:
                    if (!Finite4(geometry) || geometry.x <= 0 || geometry.y <= 0 || geometry.z <= 0 || geometry.w < 0 ||
                        geometry.w > std::min({geometry.x, geometry.y, geometry.z}))
                        throw std::invalid_argument("Jolt GPU box dimensions or convex radius are invalid");
                    break;
                case ShapeCapsule:
                    if (!Finite4(geometry) || geometry.x != 0 || geometry.z != 0 || geometry.y <= 0 || geometry.w <= 0)
                        throw std::invalid_argument("Jolt GPU capsule half cylinder and radius must be positive and "
                                                    "finite with zero unused dimensions");
                    break;
                case ShapeCylinder:
                    if (!Finite4(geometry) || geometry.x <= 0 || geometry.y <= 0 || geometry.z < 0 || geometry.w ||
                        geometry.z > std::min(geometry.x, geometry.y))
                        throw std::invalid_argument("Jolt GPU cylinder final dimensions/convex radius are invalid");
                    break;
                case ShapeConvexHull:
                    if (!Finite4(geometry) || geometry.x < 0 || geometry.x != cold->record->convexRadius ||
                        triangle != (cold->record->words[2].z != 0) ||
                        (triangle ? geometry.x != 0 || cold->record->words[0].z != 3u ||
                                        std::abs(geometry.y * geometry.y + geometry.z * geometry.z +
                                                 geometry.w * geometry.w - 1.0f) > 1e-4f
                                  : geometry.y || geometry.z || geometry.w))
                        throw std::invalid_argument("Jolt GPU hull convex radius or padding is invalid");
                    break;
                default:
                    throw std::invalid_argument("Jolt GPU leaf has an unsupported primitive type");
                }
            }
            bodyLeafEnd += record.meta.y;
        }
        if (bodyLeafEnd != leafEnd + leaves.y)
            throw std::invalid_argument("Jolt GPU body leaf ranges do not cover their world");
        for (uint32_t index = leaves.z; index < jointEnd + leaves.w; ++index) {
            const auto change = std::lower_bound(jointChanges.begin(), jointChanges.end(), index,
                                                 [](const auto &value, uint32_t key) { return value.index < key; });
            const auto &joint =
                change != jointChanges.end() && change->index == index ? change->author : model.joints[index];
            if (!joint.HasMatchingKind())
                throw std::invalid_argument("Joint kind differs from typed payload");
            const auto inWorld = [&](uint32_t slot) { return slot >= range.x && uint64_t(slot) < bodyEnd + range.y; };
            if (joint.identity.z != world || joint.identity.w || (!joint.identity.x && !joint.identity.y) ||
                !inWorld(joint.bodies.y) ||
                (joint.bodies.x != UINT32_MAX && (!inWorld(joint.bodies.x) || joint.bodies.x == joint.bodies.y)) ||
                joint.bodies.z > JointSixDOF ||
                (joint.bodies.w &
                 ~(JointEnabled | JointCollisionEnabled | JointLimitsEnabled | JointMotorEnabled | JointAutoDistance)))
                throw std::invalid_argument("Jolt GPU joint identity, endpoints or world range is invalid");
            for (const auto &value :
                 {joint.ownerAnchor(), joint.ownerAxis(), joint.ownerLocalAnchor(), joint.rootCOM1(), joint.rootCOM2(),
                  joint.scale1(), joint.scale2(), joint.connectedAnchor(), joint.connectedLocalAnchor()})
                if (!Finite4(value) || value.w != 0)
                    throw std::invalid_argument("Jolt GPU joint anchor, axis, COM and scale must be finite");
            const auto &axis = joint.ownerAxis();
            if ((joint.bodies.z == JointHinge || joint.bodies.z == JointSlider) &&
                std::abs(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z - 1.0f) > 1e-4f)
                throw std::invalid_argument("Jolt GPU joint axis must be normalized");
            const bool swingTwist = joint.bodies.z == JointSwingTwist;
            const auto zero = [](const Float4 &v) { return v.x == 0 && v.y == 0 && v.z == 0 && v.w == 0; };
            const auto normalized = [](const Float4 &v) {
                return Finite4(v) && std::abs(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w - 1.0f) <= 1e-4f;
            };
            const auto nonnegative = [](const Float4 &v) {
                return Finite4(v) && v.x >= 0 && v.y >= 0 && v.z >= 0 && v.w >= 0;
            };
            if (joint.bodies.z == JointSixDOF) {
                const auto &m = joint.sixModes();
                if (!normalized(joint.angularFrame()) || !normalized(joint.angularTarget()) || m.x & ~4095u ||
                    m.y & ~4095u || m.z || m.w || (joint.bodies.w & ~(JointEnabled | JointCollisionEnabled)))
                    throw std::invalid_argument("SixDOF frame/mode metadata differs from typed contract");
                for (const auto &v :
                     {joint.sixLinearMin(), joint.sixLinearMax(), joint.sixAngularMin(), joint.sixAngularMax(),
                      joint.sixTargetPosition(), joint.sixTargetVelocity(), joint.angularVelocity()})
                    if (!Finite4(v) || v.w)
                        throw std::invalid_argument("SixDOF vector must be finite with zero padding");
                for (const auto &v : {joint.sixLimitStiffness(), joint.sixLimitDamping(), joint.sixLinearFriction(),
                                      joint.sixAngularFriction(), joint.sixLinearMotorK(), joint.sixLinearMotorC(),
                                      joint.sixLinearMotorForce(), joint.sixAngularMotorK(), joint.sixAngularMotorC(),
                                      joint.sixAngularMotorTorque()})
                    if (!nonnegative(v) || v.w)
                        throw std::invalid_argument("SixDOF SI coefficients must be nonnegative and finite");
                for (uint32_t axis = 0; axis < 6; ++axis) {
                    const auto lane = [](const Float4 &v, uint32_t i) { return i == 0 ? v.x : i == 1 ? v.y : v.z; };
                    const auto motion = (m.x >> (2u * axis)) & 3u;
                    const float lo =
                        axis < 3 ? lane(joint.sixLinearMin(), axis) : lane(joint.sixAngularMin(), axis - 3);
                    const float hi =
                        axis < 3 ? lane(joint.sixLinearMax(), axis) : lane(joint.sixAngularMax(), axis - 3);
                    if (motion > 2 || lo > hi || (motion == 1 && lo == hi) ||
                        (axis >= 3 && (lo < -3.14159265358979323846f || hi > 3.14159265358979323846f)))
                        throw std::invalid_argument("SixDOF effective axis limits are invalid");
                }
            } else if (swingTwist) {
                const auto &v = joint.angularLimits();
                if (!normalized(joint.angularFrame()) || !normalized(joint.angularTarget()) || !Finite4(v) ||
                    v.x < -3.14159265358979323846f || v.y > 3.14159265358979323846f || v.x > v.y || v.z < 0 ||
                    v.z > 3.14159265358979323846f || v.w < 0 || v.w > 3.14159265358979323846f ||
                    joint.angularModes().x > 1 || joint.angularModes().y > 3 || joint.angularModes().z > 3 ||
                    joint.angularModes().w || !nonnegative(joint.angularSpring()) ||
                    !nonnegative(joint.angularTorque()) || joint.angularTorque().w ||
                    !Finite4(joint.angularVelocity()) || joint.angularVelocity().w ||
                    (joint.bodies.w & ~(JointEnabled | JointCollisionEnabled)) || !zero(joint.ownerAxis()) ||
                    !zero(joint.limits()) || !zero(joint.motor()) || !zero(joint.passive()) ||
                    !zero(joint.connectedAnchor()) || !zero(joint.connectedLocalAnchor()) ||
                    !zero(joint.linearSpring()))
                    throw std::invalid_argument("Jolt GPU SwingTwist settings differ from their typed contract");
            } else if (!zero(joint.angularFrame()) || !zero(joint.angularLimits()) || joint.angularModes().x ||
                       joint.angularModes().y || joint.angularModes().z || joint.angularModes().w ||
                       !zero(joint.angularSpring()) || !zero(joint.angularTorque()) || !zero(joint.angularTarget()) ||
                       !zero(joint.angularVelocity()))
                throw std::invalid_argument("Jolt GPU scalar joint has an angular suffix");
            const bool distance = joint.bodies.z == JointDistance;
            if (!Finite4(joint.limits()) || joint.limits().x > joint.limits().y ||
                (distance ? joint.limits().x < 0.0f : (joint.limits().x > 0.0f || joint.limits().y < 0.0f)) ||
                (!distance && joint.bodies.z != JointSlider &&
                 (joint.limits().x < -3.14159265358979323846f || joint.limits().y > 3.14159265358979323846f)) ||
                joint.limits().z || joint.limits().w || !Finite4(joint.motor()) || joint.motor().y < 0 ||
                joint.motor().z < 0 || joint.motor().w < 0 || !Finite4(joint.passive()) || joint.passive().x < 0 ||
                joint.passive().y < 0 || joint.passive().z < 0 || joint.passive().w)
                throw std::invalid_argument("Jolt GPU joint limits, motor or passive settings are invalid");
            if (!Finite4(joint.linearSpring()) || joint.linearSpring().x < 0.0f || joint.linearSpring().y < 0.0f ||
                joint.linearSpring().z || joint.linearSpring().w ||
                (!distance &&
                 (joint.linearSpring().x || joint.linearSpring().y || (joint.bodies.w & JointAutoDistance))))
                throw std::invalid_argument("Jolt GPU distance spring settings are invalid for the joint kind");
            if (distance &&
                ((joint.bodies.w & (JointMotorEnabled | JointLimitsEnabled)) || joint.ownerAxis().x ||
                 joint.ownerAxis().y || joint.ownerAxis().z || joint.motor().x || joint.motor().y || joint.motor().z ||
                 joint.motor().w || joint.passive().x || joint.passive().y || joint.passive().z))
                throw std::invalid_argument("Jolt GPU Distance has no angular axis, motor or passive settings");
            if (joint.bodies.z == JointSlider &&
                ((joint.bodies.w & JointMotorEnabled) || joint.motor().x || joint.motor().y || joint.motor().z ||
                 joint.motor().w || joint.passive().x || joint.passive().y || joint.passive().z))
                throw std::invalid_argument("Jolt GPU Slider has no motor or passive drive settings");
        }
        bodyEnd += range.y;
        leafEnd += leaves.y;
        jointEnd += leaves.w;
        counts.maxBodiesPerWorld = std::max(counts.maxBodiesPerWorld, range.y);
        counts.maxLeavesPerWorld = std::max(counts.maxLeavesPerWorld, leaves.y);
    }
    if (bodyEnd != counts.bodies || leafEnd != counts.shapes || jointEnd != model.joints.size())
        throw std::invalid_argument("Jolt GPU ranges do not cover the model");
    return counts;
}

template <class Change> void ValidateModelChangeIndices(std::vector<Change> &changes, size_t capacity)
{
    std::sort(changes.begin(), changes.end(), [](const auto &a, const auto &b) { return a.index < b.index; });
    for (size_t index = 0; index < changes.size(); ++index) {
        const auto &change = changes[index];
        if (change.index >= capacity || (index && changes[index - 1].index == change.index))
            throw std::invalid_argument("Jolt GPU model update requires unique indices within existing capacity");
    }
}

void ValidateForces(const std::vector<BodyForces> &forces, uint32_t bodyCount)
{
    if (forces.size() != bodyCount)
        throw std::invalid_argument("Jolt GPU force input does not match its resident model");
    for (const auto &force : forces)
        if (!Finite3(force.force) || !Finite3(force.torque))
            throw std::invalid_argument("Jolt GPU force and torque must be finite");
}

void ValidateVectorLayout(const Port &port, const ShaderResourceLayout &resource, bool uniform)
{
    const auto &value = resource.value;
    if (resource.binding.type != (uniform ? rhi::BindingType::UniformBuffer : rhi::BindingType::StorageBuffer) ||
        (uniform ? value.byteSize : resource.elementStride) != port.stride || value.kind != ShaderValueKind::Struct ||
        value.fields.size() != port.stride / 16)
        throw std::invalid_argument("Jolt GPU target layout does not match its shared device records");
    for (size_t index = 0; index < value.fields.size(); ++index) {
        const auto &field = value.fields[index];
        const auto scalar = (port.integerFields & (1u << index)) ? ShaderScalarKind::UInt32 : ShaderScalarKind::Float32;
        if (field.name != port.fields[index] || field.offset != index * 16 || field.byteSize != 16 ||
            field.kind != ShaderValueKind::Vector || field.scalar != scalar || field.rows != 1 || field.columns != 4)
            throw std::invalid_argument("Jolt GPU target field differs from the shared device record");
    }
}

void ValidateLayout(const Port &port, const ShaderResourceLayout &resource)
{
    if (port.resource == Resource::SleepStates) {
        if (resource.binding.type != rhi::BindingType::StorageBuffer ||
            resource.elementStride != sizeof(BodySleepRecord))
            throw std::invalid_argument("GPU sleep history must be its shared structured storage record");
        ValidateJoltGpuSleepLayout(resource.value);
        return;
    }
    const bool uniform = port.resource == Resource::StepParams || port.resource == Resource::StageWork;
    const auto &value = resource.value;
    if (port.resource == Resource::ConvexGeometry) {
        if (resource.binding.type != rhi::BindingType::StorageBuffer || !resource.binding.readOnlyStorage ||
            resource.elementStride != sizeof(UInt4) || value.kind != ShaderValueKind::Vector ||
            value.scalar != ShaderScalarKind::UInt32 || value.rows != 1 || value.columns != 4 ||
            value.byteSize != sizeof(UInt4) || !value.fields.empty())
            throw std::invalid_argument("Convex geometry requires its typed read-only UInt4 word binding");
        return;
    }
    ValidateVectorLayout(port, resource, uniform);
}

void ValidateBodyBatchInput(size_t slot, const ShaderResourceLayout &resource)
{
    if (slot == 0) {
        const Port header{Resource::StepParams, BodyBatchInputNames[0], sizeof(BodyBatchHeader), {"counts"}, 1};
        ValidateVectorLayout(header, resource, true);
        return;
    }
    if (resource.binding.type != rhi::BindingType::StorageBuffer || !resource.binding.readOnlyStorage)
        throw std::invalid_argument("Body batch tables require read-only structured storage");
    if (slot == 1) {
        ValidateVectorLayout(Ports[PortIndex(Resource::StepParams)], resource, false);
        return;
    }
    const auto &value = resource.value;
    constexpr std::array<std::string_view, 3> fields{"selectionIndex", "bodyFirst", "bodyCount"};
    if (resource.elementStride != sizeof(BodyBatchTile) || value.kind != ShaderValueKind::Struct ||
        value.byteSize != sizeof(BodyBatchTile) || value.fields.size() != fields.size())
        throw std::invalid_argument("Body batch tile differs from its shared three-scalar layout");
    for (size_t i = 0; i < fields.size(); ++i) {
        const auto &field = value.fields[i];
        if (field.name != fields[i] || field.kind != ShaderValueKind::Scalar ||
            field.scalar != ShaderScalarKind::UInt32 || field.offset != i * sizeof(uint32_t) ||
            field.byteSize != sizeof(uint32_t) || field.rows != 1 || field.columns != 1)
            throw std::invalid_argument("Body batch tile field differs from its shared scalar layout");
    }
}

template <class T> std::vector<uint8_t> Bytes(const T *values, size_t count)
{
    std::vector<uint8_t> bytes(count * sizeof(T));
    if (!bytes.empty())
        std::memcpy(bytes.data(), values, bytes.size());
    return bytes;
}

template <class Change, class Record>
void AppendModelUpload(std::vector<rhi::ComputeBufferUpdate> &updates,
                       const std::shared_ptr<rhi::ComputeBuffer> &buffer, const std::vector<Change> &changes,
                       const std::vector<Record> &records, Record Change::*member, uint64_t baseOffset = 0)
{
    if (changes.empty())
        return;
    const auto first = changes.front().index;
    const size_t count = size_t(changes.back().index) - first + 1;
    auto bytes = Bytes(records.data() + first, count);
    // Allocate only the upload envelope; its holes retain accepted author data.
    for (const auto &change : changes)
        std::memcpy(bytes.data() + size_t(change.index - first) * sizeof(Record), &(change.*member), sizeof(Record));
    updates.push_back({buffer, baseOffset + uint64_t(first) * sizeof(Record), std::move(bytes)});
}

uint32_t Groups(uint32_t count, uint32_t width)
{
    return count / width + static_cast<uint32_t>(count % width != 0);
}

bool IslandDomain(Domain domain)
{
    return domain == Domain::IslandPatches || domain == Domain::IslandJoints;
}

const JoltGpuContactWorkset &RequireContacts(const JoltGpuStepInputs &inputs)
{
    if (!inputs.GetContacts())
        throw std::invalid_argument("Jolt GPU contact work requires its substep workset");
    return *inputs.GetContacts();
}

bool ValidateWork(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs, const StageDefinition &definition,
                  const JoltGpuStageWork &work)
{
    const auto range = work.parameters.range;
    const uint32_t quantum = JoltGpuStageItemLimit(definition.stage, definition.domain, inputs.GetContacts().get());
    if (IslandDomain(definition.domain)) {
        const auto &constraints = inputs.GetConstraints();
        if (!constraints)
            throw std::invalid_argument("Jolt GPU island work requires its validated substep plan");
        const bool joints = definition.domain == Domain::IslandJoints;
        const auto &dispatches = constraints->Dispatches(joints);
        if (work.islandDispatch >= dispatches.size())
            throw std::invalid_argument("Jolt GPU island command is outside its substep plan");
        const auto expected = dispatches[work.islandDispatch].parameters.range;
        const bool wholeRounds = IsContactRoundStage(definition.stage);
        const auto &step = inputs.GetParameters();
        const uint32_t rounds = definition.stage == Stage::SolvePositionRounds ? step.solve.x : step.counts.w;
        if (range.y != expected.y || range.z != expected.z || range.w != expected.w ||
            (wholeRounds ? (expected.z != ConstraintWholeIslandTable || constraints->jointRefCount || !range.x ||
                            range.x > rounds || range.y > IslandPatchRoundVisits / range.x)
                         : range.x != expected.x))
            throw std::invalid_argument("Jolt GPU island dispatch does not match its substep plan");
        return true;
    }
    if (range.y > quantum)
        throw std::invalid_argument("Jolt GPU work exceeds its bounded item quantum");
    if (range.z || range.w)
        throw std::invalid_argument("Jolt GPU global work must leave its unused island range zero");
    uint64_t count = 0, base = 0;
    const auto &selected = inputs.GetParameters();
    switch (definition.domain) {
    case Domain::Bodies:
        count = selected.selectedCounts.x;
        base = selected.selection.y;
        break;
    case Domain::Shapes:
        count = selected.selectedCounts.y;
        base = selected.selection.z;
        break;
    case Domain::Contacts:
        count = uint64_t(RequireContacts(inputs).patchSlots) * MaxContactPointsPerPair;
        break;
    case Domain::Patches:
        count = RequireContacts(inputs).patchSlots;
        break;
    case Domain::Joints:
        count = selected.selectedCounts.z;
        base = selected.selection.w;
        break;
    default:
        throw std::invalid_argument("Jolt GPU work has an unknown global domain");
    }
    if (range.x < base || uint64_t(range.x) - base > count || range.y > count - (uint64_t(range.x) - base))
        throw std::invalid_argument("Jolt GPU work addresses global items outside its accepted model");
    return range.y != 0;
}
} // namespace

void JoltGpuBuildWorldRanges(JoltGpuModel &model, const std::vector<uint32_t> &capacities)
{
    model.worlds.clear();
    model.worlds.reserve(capacities.size());
    uint64_t body = 0, leaf = 0, joint = 0;
    for (uint32_t row = 0; row < capacities.size(); ++row) {
        const uint64_t firstBody = body, firstLeaf = leaf, firstJoint = joint;
        if (body + capacities[row] > model.bodies.size())
            throw std::invalid_argument("Packed WorldRange body capacities exceed target model");
        for (uint32_t local = 0; local < capacities[row]; ++local, ++body) {
            auto &record = model.bodies[static_cast<size_t>(body)];
            if (record.meta.x != leaf || leaf + record.meta.y > model.shapes.size())
                throw std::invalid_argument("Packed WorldRange leaf stream is not contiguous by body");
            record.meta.w = row;
            for (uint32_t ordinal = 0; ordinal < record.meta.y; ++ordinal, ++leaf)
                model.shapes[static_cast<size_t>(leaf)].meta.y = static_cast<uint32_t>(body);
        }
        while (joint < model.joints.size() && model.joints[static_cast<size_t>(joint)].identity.z == row)
            ++joint;
        model.worlds.push_back({{static_cast<uint32_t>(firstBody), capacities[row], 0u, 0u},
                                {static_cast<uint32_t>(firstLeaf), static_cast<uint32_t>(leaf - firstLeaf),
                                 static_cast<uint32_t>(firstJoint), static_cast<uint32_t>(joint - firstJoint)}});
    }
    if (body != model.bodies.size() || leaf != model.shapes.size() || joint != model.joints.size())
        throw std::invalid_argument("Packed WorldRange builder must cover every target author record");
}

JoltGpuStepParameters::JoltGpuStepParameters(StepParams parameters) : m_parameters(parameters)
{
    if (parameters.solve.y == 0 || !Finite4(parameters.contact) || parameters.contact.x < 0 ||
        parameters.contact.y < 0 || parameters.contact.z < 0 || parameters.contact.w < 0 || parameters.contact.w > 1 ||
        !Finite3(parameters.restitution) || parameters.restitution.x < 0 || parameters.restitution.y < 0 ||
        parameters.restitution.z < 0)
        throw std::invalid_argument("Jolt GPU substep count and contact parameters are invalid");
    SetGravityAndDeltaTime(parameters.gravityDt);
}

void JoltGpuStepParameters::SetGravityAndDeltaTime(Float4 gravityDt)
{
    if (!Finite4(gravityDt) || gravityDt.w <= 0.0f)
        throw std::invalid_argument("Jolt GPU gravity and substep duration must be finite with positive duration");
    m_parameters.gravityDt = gravityDt;
}

std::vector<JoltGpuStorage::SweepCounts> JoltGpuStorage::CountSweeps(const JoltGpuModel &model)
{
    std::vector<SweepCounts> counts(model.worlds.size());
    for (const auto &body : model.bodies) {
        auto &count = counts[body.meta.w];
        count.enabled += uint32_t(Sweeps(body));
        count.dynamicPairs += uint32_t(SweepsDynamicPairs(body));
    }
    return counts;
}

const JoltGpuStorage::DiscretePairLimits &JoltGpuStorage::GetDiscretePairLimits(uint32_t world) const
{
    auto &cached = m_sweepCounts[world].discrete;
    if (cached)
        return *cached;

    const auto &range = m_model.worlds[world];
    uint64_t active = 0u, passive = 0u, candidates = 0u;
    uint32_t dynamicBodies = 0u;
    for (uint32_t index = range.bodies.x; index < range.bodies.x + range.bodies.y; ++index) {
        const auto &body = m_model.bodies[index];
        dynamicBodies += uint32_t(body.meta.z == MotionDynamic);
        if (!body.meta.y || (body.meta.z & MotionNotResident))
            continue;
        const auto first = m_model.shapes.begin() + body.meta.x;
        const uint64_t bodyActive = body.meta.z == MotionDynamic
                                        ? body.meta.y
                                        : std::count_if(first, first + body.meta.y, [](const auto &leaf) {
                                              return (leaf.meta.w & ShapeSensor) != 0u;
                                          });
        const uint64_t bodyPassive = body.meta.y - bodyActive;
        candidates += bodyActive * (active + passive) + bodyPassive * active;
        active += bodyActive;
        passive += bodyPassive;
    }
    // Every projected job has an active leaf. The GPU may still have to
    // test same-body or nonresident passive leaves while enumerating it.
    const uint64_t projections = active * (range.leaves.y - active) + active * (active - (active != 0u)) / 2u;
    return cached.emplace(DiscretePairLimits{candidates, projections, static_cast<uint32_t>(active), dynamicBodies});
}

JoltGpuStorage::JoltGpuStorage(rhi::ComputeHost &host, JoltGpuModel modelData)
    : m_host(host), m_model(std::move(modelData))
{
    const auto &model = m_model;
    const auto counts = ValidateModel(model);
    m_sweepCounts = CountSweeps(model);
    m_jointLayout = std::make_shared<JoltGpuJointLayout>(model.joints);
    if (!model.joints.empty())
        throw std::invalid_argument("Jolt GPU initial joint topology requires an ordered GPU initialization");
    m_bodyCount = counts.bodies;
    m_worldCount = counts.worlds;
    m_shapeCount = counts.shapes;
    m_maxBodiesPerWorld = counts.maxBodiesPerWorld;
    m_maxLeavesPerWorld = counts.maxLeavesPerWorld;
    if (m_bodyCount == 0)
        return;

    const uint64_t sizes[] = {sizeof(StepParams),
                              model.bodies.size() * sizeof(BodyModel),
                              std::max<size_t>(1, model.shapes.size()) * sizeof(ShapeModel),
                              model.worlds.size() * sizeof(WorldRange),
                              uint64_t(m_bodyCount) * sizeof(BodyForces),
                              uint64_t(m_bodyCount) * sizeof(BodyState),
                              uint64_t(m_shapeCount) * sizeof(Aabb),
                              sizeof(Contact),
                              uint64_t(m_bodyCount) * sizeof(KinematicTarget),
                              sizeof(ContactPatch),
                              sizeof(QueryRecordWord),
                              uint64_t(m_bodyCount) * sizeof(BodySleepRecord)};
    for (size_t port = 0; port < m_buffers.size(); ++port) {
        // Preserve the scalar type for homogeneous float ports so numerical
        // consumers need no hidden host cast. Mixed records remain raw words;
        // their structured element interpretation comes from target reflection.
        m_buffers[port] = Allocate(m_host, port, sizes[port]);
    }
}

void JoltGpuStorage::SetGeometryStore(std::shared_ptr<JoltGpuGeometryStore> store)
{
    if (!store || m_geometryStore)
        throw std::logic_error("Jolt GPU geometry store must be installed exactly once");
    m_geometryStore = std::move(store);
}

const std::shared_ptr<rhi::ComputeBuffer> &JoltGpuStorage::Get(Resource resource) const
{
    if (resource == Resource::ConvexGeometry) {
        if (!m_geometry || !m_geometry->buffer)
            throw std::logic_error("No accepted convex geometry input");
        return m_geometry->buffer;
    }
    const auto slot = PortIndex(resource);
    if (slot >= StoragePortCount)
        throw std::invalid_argument("Jolt GPU storage requires a persistent resource, not a per-work uniform");
    return m_buffers[slot];
}

JoltGpuContactHistory JoltGpuStorage::GetContactHistory(uint32_t worldRow) const
{
    const auto found = m_contactHistory.find(worldRow);
    return found == m_contactHistory.end() ? JoltGpuContactHistory{} : found->second;
}

void JoltGpuStorage::PublishContactHistory(uint32_t worldRow, const JoltGpuContactHistory &history)
{
    m_contactHistory.insert_or_assign(worldRow, history);
}

void JoltGpuStorage::ClearContactHistory(uint32_t worldRow) noexcept
{
    m_contactHistory.erase(worldRow);
}

std::shared_ptr<rhi::PreparedComputeBatch> JoltGpuStorage::PrepareInitialization(const std::vector<BodyState> &states,
                                                                                 const std::vector<BodyForces> &forces)
{
    if (states.size() != m_bodyCount || forces.size() != m_bodyCount)
        throw std::invalid_argument("Jolt GPU initialization does not match its resident model");
    for (size_t body = 0; body < states.size(); ++body) {
        const auto &state = states[body];
        if (!Finite3(state.positionCOM) || !Normalized(state.rotation) || !Finite3(state.linearVelocity) ||
            !Finite3(state.angularVelocity) || !Finite3(forces[body].force) || !Finite3(forces[body].torque) ||
            ((m_model.bodies[body].meta.z & ~MotionNotResident) == MotionStatic &&
             (!Zero3(state.linearVelocity) || !Zero3(state.angularVelocity))))
            throw std::invalid_argument("Jolt GPU initial pose, velocity or force is invalid");
    }
    if (m_bodyCount == 0)
        return {};
    const std::vector<KinematicTarget> targets(m_bodyCount);
    const auto sleeps = InitialSleepAuthors(m_model, m_geometry);
    std::vector<rhi::ComputeBufferUpdate> updates{
        {Get(Resource::BodyModels), 0, Bytes(m_model.bodies.data(), m_model.bodies.size())},
        {Get(Resource::Worlds), 0, Bytes(m_model.worlds.data(), m_model.worlds.size())},
        {Get(Resource::BodyForces), 0, Bytes(forces.data(), forces.size())},
        {Get(Resource::BodyStates), 0, Bytes(states.data(), states.size())},
        {Get(Resource::KinematicTargets), 0, Bytes(targets.data(), targets.size())},
        {Get(Resource::SleepStates), 0, Bytes(sleeps.data(), sleeps.size())},
    };
    const UInt4 jointHeader{0, 1, 1, 0};
    updates.push_back({Get(Resource::Joints), 0, Bytes(&jointHeader, 1)});
    if (m_shapeCount)
        updates.push_back({Get(Resource::ShapeModels), 0, Bytes(m_model.shapes.data(), m_model.shapes.size())});
    else
        updates.push_back({Get(Resource::ShapeModels), 0, std::vector<uint8_t>(sizeof(ShapeModel))});
    updates.push_back({Get(Resource::Contacts), 0, std::vector<uint8_t>(sizeof(Contact))});
    updates.push_back({Get(Resource::ContactPatches), 0, std::vector<uint8_t>(sizeof(ContactPatch))});
    return rhi::PrepareComputeBatch(m_host, std::move(updates), {});
}

std::shared_ptr<rhi::PreparedComputeBatch> JoltGpuStorage::PrepareParameters(const JoltGpuStepParameters &prepared)
{
    return PrepareInputs(prepared);
}

std::shared_ptr<rhi::PreparedComputeBatch>
JoltGpuStorage::PrepareInputs(const JoltGpuStepParameters &prepared, const std::vector<BodyForces> *cpuForces,
                              const std::shared_ptr<rhi::ComputeBuffer> &parameterBuffer)
{
    if (cpuForces)
        ValidateForces(*cpuForces, prepared.Get().selectedCounts.x);
    if (m_bodyCount == 0)
        return {};
    auto parameters = prepared.Get();
    parameters.counts.x = m_bodyCount;
    parameters.counts.y = m_worldCount;
    parameters.counts.z = m_shapeCount;
    parameters.solve.z = parameters.solve.w = 0;
    const auto &target = parameterBuffer ? parameterBuffer : Get(Resource::StepParams);
    if (!target || target->GetByteSize() != sizeof(StepParams) || !m_host.SharesServicesWith(target->GetHost()))
        throw std::invalid_argument("GPU member parameters require their same-owner typed buffer");
    std::vector<rhi::ComputeBufferUpdate> updates{{target, 0, Bytes(&parameters, 1)}};
    if (cpuForces && !cpuForces->empty())
        updates.push_back({Get(Resource::BodyForces), uint64_t(parameters.selection.y) * sizeof(BodyForces),
                           Bytes(cpuForces->data(), cpuForces->size())});
    return rhi::PrepareComputeBatch(m_host, std::move(updates), {});
}

std::shared_ptr<rhi::PreparedComputeBatch> JoltGpuStorage::PrepareForces(const std::vector<BodyForces> &forces,
                                                                         const StepParams &selected)
{
    ValidateForces(forces, selected.selectedCounts.x);
    if (selected.selectedCounts.x == 0)
        return {};
    return rhi::PrepareComputeBatch(m_host,
                                    {{Get(Resource::BodyForces), uint64_t(selected.selection.y) * sizeof(BodyForces),
                                      Bytes(forces.data(), forces.size())}},
                                    {});
}

std::unique_ptr<JoltGpuModelUpdate> JoltGpuStorage::PrepareModelUpdate(
    std::vector<JoltGpuBodyModelUpdate> bodyChanges, std::vector<JoltGpuShapeModelUpdate> shapeChanges,
    std::vector<rhi::ComputeBufferUpdate> additionalUpdates, std::vector<rhi::ComputeDispatchDesc> mutations,
    std::vector<JoltGpuJointModelUpdate> jointChanges)
{
    if (bodyChanges.empty() && shapeChanges.empty() && jointChanges.empty() && additionalUpdates.empty() &&
        mutations.empty())
        return {};
    if (m_modelUpdatePending)
        throw std::logic_error("Jolt GPU storage already has an unaccepted model transaction");

    const auto &bodyBuffer = Get(Resource::BodyModels);
    const auto &shapeBuffer = Get(Resource::ShapeModels);
    const auto &worldBuffer = Get(Resource::Worlds);
    const auto &jointBuffer = Get(Resource::Joints);
    for (const auto &update : additionalUpdates) {
        if (update.buffer &&
            ((bodyBuffer && update.buffer->GetWriterStateIdentity() == bodyBuffer->GetWriterStateIdentity()) ||
             (shapeBuffer && update.buffer->GetWriterStateIdentity() == shapeBuffer->GetWriterStateIdentity()) ||
             (worldBuffer && update.buffer->GetWriterStateIdentity() == worldBuffer->GetWriterStateIdentity()) ||
             (jointBuffer && update.buffer->GetWriterStateIdentity() == jointBuffer->GetWriterStateIdentity() &&
              update.offset < m_jointLayout->AuthorEndBytes())))
            throw std::invalid_argument("Jolt GPU model ports must be updated through their author-model delta");
    }

    ValidateModelChangeIndices(bodyChanges, m_model.bodies.size());
    ValidateModelChangeIndices(shapeChanges, m_model.shapes.size());
    ValidateModelChangeIndices(jointChanges, m_model.joints.size());
    for (const auto &change : jointChanges) {
        const auto &old = m_model.joints[change.index];
        if (std::memcmp(&old.bodies, &change.author.bodies, 3 * sizeof(uint32_t)) ||
            std::memcmp(&old.identity, &change.author.identity, sizeof(UInt4)) ||
            ((old.bodies.z == JointSwingTwist || old.bodies.z == JointSixDOF) &&
             std::memcmp(&old.angularFrame(), &change.author.angularFrame(), sizeof(Float4))))
            throw std::invalid_argument("Jolt GPU joint endpoint or identity changes require topology replacement");
    }
    if (!bodyChanges.empty() || !shapeChanges.empty() || !jointChanges.empty())
        (void)ValidateModel(m_model, bodyChanges, shapeChanges, jointChanges, m_geometryStore.get());
    // Existing runtime state, force and target ports are retained unless an
    // explicit caller mutation updates them in this same RHI packet.
    AppendModelUpload(additionalUpdates, bodyBuffer, bodyChanges, m_model.bodies, &JoltGpuBodyModelUpdate::model);
    AppendModelUpload(additionalUpdates, shapeBuffer, shapeChanges, m_model.shapes, &JoltGpuShapeModelUpdate::model);
    for (const auto &change : jointChanges) {
        const auto &row = m_jointLayout->Row(change.index);
        auto payload = JoltGpuJointLayout::AuthorBytes(change.author);
        additionalUpdates.push_back({jointBuffer, uint64_t(row.layout.x) * 16u, std::move(payload)});
        // Kind, endpoint and stable identity are immutable in an in-place delta.
        additionalUpdates.push_back(
            {jointBuffer, (1u + uint64_t(change.index) * JointDirectoryWords) * 16u, Bytes(&change.author.bodies, 1)});
    }
    auto batch = rhi::PrepareComputeBatch(m_host, std::move(additionalUpdates), std::move(mutations));
    auto transaction = std::unique_ptr<JoltGpuModelUpdate>(new JoltGpuModelUpdate(
        *this, std::move(batch), std::move(bodyChanges), std::move(shapeChanges), std::move(jointChanges)));
    m_modelUpdatePending = true;
    return transaction;
}

std::unique_ptr<JoltGpuModelUpdate>
JoltGpuStorage::PrepareModelReplacement(JoltGpuModel replacement,
                                        std::vector<rhi::ComputeBufferUpdate> additionalUpdates,
                                        std::vector<rhi::ComputeDispatchDesc> mutations)
{
    if (m_modelUpdatePending)
        throw std::logic_error("Jolt GPU storage already has an unaccepted model transaction");
    // Leaf topology replacement cannot recreate numerical joint state or
    // overwrite accepted joint membership with the caller's geometry model.
    replacement.joints = m_model.joints;
    if (replacement.worlds.size() == m_model.worlds.size())
        for (size_t index = 0; index < replacement.worlds.size(); ++index) {
            replacement.worlds[index].leaves.z = m_model.worlds[index].leaves.z;
            replacement.worlds[index].leaves.w = m_model.worlds[index].leaves.w;
        }
    const auto counts = ValidateModel(replacement, {}, {}, {}, m_geometryStore.get());
    if (counts.bodies != m_bodyCount || counts.worlds != m_worldCount)
        throw std::invalid_argument("Jolt GPU replacement must retain body capacity and world count");
    if (!m_bodyCount)
        return PrepareModelUpdate({}, {}, std::move(additionalUpdates), std::move(mutations));
    for (const auto &update : additionalUpdates)
        for (auto resource : {Resource::BodyModels, Resource::ShapeModels, Resource::Worlds, Resource::Joints}) {
            const auto &author = Get(resource);
            if (update.buffer && author && update.buffer->GetWriterStateIdentity() == author->GetWriterStateIdentity())
                throw std::invalid_argument("Jolt GPU replacement owns all author-model uploads");
        }

    auto buffers = m_buffers;
    for (auto resource : {Resource::BodyModels, Resource::ShapeModels, Resource::Worlds, Resource::Bounds}) {
        const auto port = PortIndex(resource);
        const uint64_t size = resource == Resource::BodyModels ? uint64_t(counts.bodies) * sizeof(BodyModel)
                              : resource == Resource::ShapeModels
                                  ? uint64_t(std::max(1u, counts.shapes)) * sizeof(ShapeModel)
                              : resource == Resource::Worlds ? uint64_t(counts.worlds) * sizeof(WorldRange)
                                                             : uint64_t(counts.shapes) * sizeof(Aabb);
        buffers[port] = Allocate(m_host, port, size);
    }
    additionalUpdates.push_back(
        {buffers[PortIndex(Resource::BodyModels)], 0, Bytes(replacement.bodies.data(), replacement.bodies.size())});
    if (counts.shapes)
        additionalUpdates.push_back({buffers[PortIndex(Resource::ShapeModels)], 0,
                                     Bytes(replacement.shapes.data(), replacement.shapes.size())});
    else
        additionalUpdates.push_back(
            {buffers[PortIndex(Resource::ShapeModels)], 0, std::vector<uint8_t>(sizeof(ShapeModel))});
    additionalUpdates.push_back(
        {buffers[PortIndex(Resource::Worlds)], 0, Bytes(replacement.worlds.data(), replacement.worlds.size())});
    auto batch = rhi::PrepareComputeBatch(m_host, std::move(additionalUpdates), std::move(mutations));
    auto transaction = std::unique_ptr<JoltGpuModelUpdate>(
        new JoltGpuModelUpdate(*this, std::move(batch), std::move(replacement), std::move(buffers),
                               {counts.shapes, counts.maxBodiesPerWorld, counts.maxLeavesPerWorld}));
    m_modelUpdatePending = true;
    return transaction;
}

std::unique_ptr<JoltGpuModelUpdate> JoltGpuStorage::PreparePackedReplacement(JoltGpuModel replacement,
                                                                             JoltGpuBodyStatePorts bodies,
                                                                             std::shared_ptr<rhi::ComputeBuffer> stream)
{
    if (m_modelUpdatePending)
        throw std::logic_error("GPU packed layout already has an unaccepted transaction");
    const auto counts = ValidateModel(replacement, {}, {}, {}, m_geometryStore.get());
    if (bodies.count != counts.bodies)
        throw std::invalid_argument("GPU packed layout numerical body count differs from model");
    auto buffers = m_buffers;
    const auto require = [&](Resource resource, const std::shared_ptr<rhi::ComputeBuffer> &buffer, uint64_t bytes) {
        if (!buffer || buffer->IsFrozen() || !m_host.SharesServicesWith(buffer->GetHost()) ||
            buffer->GetByteSize() != bytes || (counts.bodies && !buffer->HasInitializedRange(0, bytes)))
            throw std::invalid_argument("GPU packed layout requires completed private numerical ports");
        buffers[PortIndex(resource)] = buffer;
    };
    const uint64_t bodyCapacity = std::max(1u, counts.bodies);
    require(Resource::BodyStates, bodies.states, bodyCapacity * sizeof(BodyState));
    require(Resource::BodyForces, bodies.forces, bodyCapacity * sizeof(BodyForces));
    require(Resource::KinematicTargets, bodies.targets, bodyCapacity * sizeof(KinematicTarget));
    require(Resource::SleepStates, bodies.sleeps, bodyCapacity * sizeof(BodySleepRecord));
    const JoltGpuJointLayout jointLayout(replacement.joints);
    const uint64_t jointBytes = jointLayout.ByteSize();
    if (!stream || stream->IsFrozen() || !m_host.SharesServicesWith(stream->GetHost()) ||
        stream->GetByteSize() != jointBytes || !stream->HasInitializedRange(0, jointBytes))
        throw std::invalid_argument("GPU packed layout requires completed remapped joint runtime");
    buffers[PortIndex(Resource::Joints)] = std::move(stream);
    for (const auto resource : {Resource::BodyModels, Resource::ShapeModels, Resource::Worlds, Resource::Bounds}) {
        const auto port = PortIndex(resource);
        const uint64_t count = resource == Resource::BodyModels ? bodyCapacity
                               : resource == Resource::Worlds   ? std::max(1u, counts.worlds)
                                                                : std::max(1u, counts.shapes);
        buffers[port] = Allocate(m_host, port, count * Ports[port].stride);
    }
    for (const auto resource : {Resource::StepParams, Resource::Contacts, Resource::ContactPatches}) {
        const auto port = PortIndex(resource);
        if (!buffers[port])
            buffers[port] = Allocate(m_host, port, Ports[port].stride);
    }
    std::vector<rhi::ComputeBufferUpdate> uploads;
    const auto author = [&](Resource resource, const auto &values) {
        const auto port = PortIndex(resource);
        auto bytes = values.empty() ? std::vector<uint8_t>(Ports[port].stride) : Bytes(values.data(), values.size());
        uploads.push_back({buffers[port], 0u, std::move(bytes)});
    };
    author(Resource::BodyModels, replacement.bodies);
    author(Resource::ShapeModels, replacement.shapes);
    author(Resource::Worlds, replacement.worlds);
    if (!m_buffers[PortIndex(Resource::Contacts)]) {
        uploads.push_back({buffers[PortIndex(Resource::Contacts)], 0u, std::vector<uint8_t>(sizeof(Contact))});
        uploads.push_back(
            {buffers[PortIndex(Resource::ContactPatches)], 0u, std::vector<uint8_t>(sizeof(ContactPatch))});
    }
    auto packet = rhi::PrepareComputeBatch(m_host, std::move(uploads), {});
    auto transaction = std::unique_ptr<JoltGpuModelUpdate>(
        new JoltGpuModelUpdate(*this, std::move(packet), std::move(replacement), std::move(buffers),
                               {counts.shapes, counts.maxBodiesPerWorld, counts.maxLeavesPerWorld}));
    m_modelUpdatePending = true;
    return transaction;
}

std::unique_ptr<JoltGpuModelUpdate>
JoltGpuStorage::PrepareJointReplacement(std::vector<JointAuthor> joints, std::shared_ptr<rhi::ComputeBuffer> newStream,
                                        std::vector<rhi::ComputeBufferUpdate> uploads,
                                        std::vector<rhi::ComputeDispatchDesc> remapInit)
{
    if (m_modelUpdatePending)
        throw std::logic_error("Jolt GPU storage already has an unaccepted model transaction");
    JoltGpuModel replacement = m_model;
    replacement.joints = std::move(joints);
    std::vector<uint32_t> capacities;
    for (const auto &world : replacement.worlds)
        capacities.push_back(world.bodies.y);
    JoltGpuBuildWorldRanges(replacement, capacities);
    const auto counts = ValidateModel(replacement, {}, {}, {}, m_geometryStore.get());
    const JoltGpuJointLayout jointLayout(replacement.joints);
    const auto expectedPrefix = jointLayout.PrefixBytes(replacement.joints);
    const uint64_t authorEnd = jointLayout.AuthorEndBytes();
    const uint64_t streamBytes = jointLayout.ByteSize();
    if (!newStream || newStream->IsFrozen() || !m_host.SharesServicesWith(newStream->GetHost()) ||
        newStream->GetByteSize() != streamBytes || newStream->GetDesc().scalarType != rhi::ComputeScalarType::UInt32)
        throw std::invalid_argument("Jolt GPU replacement stream has invalid ownership, mutability, size or type");
    for (const auto &buffer : m_buffers)
        if (buffer && buffer->GetWriterStateIdentity() == newStream->GetWriterStateIdentity())
            throw std::invalid_argument("Jolt GPU joint topology replacement requires a new stream");

    bool authorUploaded = false;
    const UInt4 expectedHeader = jointLayout.Header();
    for (const auto &upload : uploads) {
        if (!upload.buffer)
            throw std::invalid_argument("Jolt GPU replacement upload requires a buffer");
        for (const auto &buffer : m_buffers)
            if (buffer && buffer->GetWriterStateIdentity() == upload.buffer->GetWriterStateIdentity())
                throw std::invalid_argument("Jolt GPU joint replacement cannot overwrite accepted scene resources");
        if (upload.buffer->GetWriterStateIdentity() != newStream->GetWriterStateIdentity())
            continue;
        if (authorUploaded || upload.offset || upload.data.size() != authorEnd ||
            std::memcmp(upload.data.data(), &expectedHeader, sizeof(expectedHeader)) ||
            std::memcmp(upload.data.data(), expectedPrefix.data(), expectedPrefix.size()))
            throw std::invalid_argument("Jolt GPU replacement must upload exactly its header and author prefix once");
        authorUploaded = true;
    }
    if (!authorUploaded)
        throw std::invalid_argument("Jolt GPU replacement is missing its accepted author prefix");
    std::vector<std::pair<uint64_t, uint64_t>> initialized;
    for (const auto &dispatch : remapInit) {
        for (size_t binding = 0; binding < dispatch.buffers.size(); ++binding) {
            const auto &buffer = dispatch.buffers[binding];
            if (!buffer || buffer->GetWriterStateIdentity() != newStream->GetWriterStateIdentity())
                continue;
            if (binding >= dispatch.bufferAccesses.size() || !dispatch.proof)
                throw std::invalid_argument("Jolt GPU replacement runtime requires explicit range proofs");
            for (const auto &write : dispatch.proof->writes) {
                if (write.binding != binding)
                    continue;
                if (write.offset < authorEnd || write.offset > streamBytes ||
                    write.byteSize > streamBytes - write.offset)
                    throw std::invalid_argument("Jolt GPU initialization cannot overwrite its author prefix");
                initialized.emplace_back(write.offset, write.offset + write.byteSize);
            }
        }
    }
    std::sort(initialized.begin(), initialized.end());
    uint64_t initializedEnd = authorEnd;
    for (const auto &[begin, end] : initialized) {
        if (begin > initializedEnd)
            break;
        initializedEnd = std::max(initializedEnd, end);
    }
    if (initializedEnd != streamBytes)
        throw std::invalid_argument("Jolt GPU replacement must initialize every runtime record on the device");
    auto buffers = m_buffers;
    buffers[PortIndex(Resource::Joints)] = std::move(newStream);
    buffers[PortIndex(Resource::Worlds)] =
        Allocate(m_host, PortIndex(Resource::Worlds), uint64_t(counts.worlds) * sizeof(WorldRange));
    if (counts.worlds)
        uploads.push_back(
            {buffers[PortIndex(Resource::Worlds)], 0, Bytes(replacement.worlds.data(), replacement.worlds.size())});
    auto batch = rhi::PrepareComputeBatch(m_host, std::move(uploads), std::move(remapInit));
    auto transaction = std::unique_ptr<JoltGpuModelUpdate>(
        new JoltGpuModelUpdate(*this, std::move(batch), std::move(replacement), std::move(buffers),
                               {counts.shapes, counts.maxBodiesPerWorld, counts.maxLeavesPerWorld}));
    m_modelUpdatePending = true;
    return transaction;
}

JoltGpuModelUpdate::JoltGpuModelUpdate(JoltGpuStorage &storage, std::shared_ptr<rhi::PreparedComputeBatch> batch,
                                       std::vector<JoltGpuBodyModelUpdate> bodyChanges,
                                       std::vector<JoltGpuShapeModelUpdate> shapeChanges,
                                       std::vector<JoltGpuJointModelUpdate> jointChanges)
    : m_storage(&storage), m_batch(std::move(batch)), m_bodyChanges(std::move(bodyChanges)),
      m_shapeChanges(std::move(shapeChanges)), m_jointChanges(std::move(jointChanges))
{
    if (m_jointChanges.empty())
        m_jointLayout = storage.GetJointLayout();
    else
        m_jointLayout = storage.GetJointLayout()->WithAuthorUpdates(m_jointChanges);
}

JoltGpuModelUpdate::~JoltGpuModelUpdate()
{
    Cancel();
}

JoltGpuModelUpdate::JoltGpuModelUpdate(JoltGpuStorage &storage, std::shared_ptr<rhi::PreparedComputeBatch> batch,
                                       JoltGpuModel replacement,
                                       std::array<std::shared_ptr<rhi::ComputeBuffer>, 12> replacementBuffers,
                                       std::array<uint32_t, 3> replacementCounts)
    : m_storage(&storage), m_batch(std::move(batch)), m_replacement(std::move(replacement)),
      m_replacementBuffers(std::move(replacementBuffers)), m_replacementCounts(replacementCounts)
{
    m_jointLayout = std::make_shared<JoltGpuJointLayout>(m_replacement->joints);
    m_replacementSweepCounts = JoltGpuStorage::CountSweeps(*m_replacement);
}

void JoltGpuModelUpdate::Commit() noexcept
{
    if (m_replacement) {
        const auto slot = PortIndex(Resource::ShapeModels);
        if (m_replacementBuffers[slot] != m_storage->m_buffers[slot])
            m_storage->m_contactHistory.clear();
        m_storage->m_model.bodies.swap(m_replacement->bodies);
        m_storage->m_model.shapes.swap(m_replacement->shapes);
        m_storage->m_model.worlds.swap(m_replacement->worlds);
        m_storage->m_model.joints.swap(m_replacement->joints);
        m_storage->m_sweepCounts.swap(m_replacementSweepCounts);
        m_storage->m_buffers.swap(m_replacementBuffers);
        m_storage->m_bodyCount = static_cast<uint32_t>(m_storage->m_model.bodies.size());
        m_storage->m_worldCount = static_cast<uint32_t>(m_storage->m_model.worlds.size());
        m_storage->m_shapeCount = m_replacementCounts[0];
        m_storage->m_maxBodiesPerWorld = m_replacementCounts[1];
        m_storage->m_maxLeavesPerWorld = m_replacementCounts[2];
    }
    for (const auto &change : m_bodyChanges) {
        m_storage->ClearContactHistory(change.model.meta.w);
        auto &old = m_storage->m_model.bodies[change.index];
        auto &previous = m_storage->m_sweepCounts[old.meta.w];
        previous.enabled -= uint32_t(Sweeps(old));
        previous.dynamicPairs -= uint32_t(SweepsDynamicPairs(old));
        auto &next = m_storage->m_sweepCounts[change.model.meta.w];
        next.enabled += uint32_t(Sweeps(change.model));
        next.dynamicPairs += uint32_t(SweepsDynamicPairs(change.model));
        if (old.meta.x != change.model.meta.x || old.meta.y != change.model.meta.y ||
            old.meta.z != change.model.meta.z || old.meta.w != change.model.meta.w) {
            previous.discrete.reset();
            next.discrete.reset();
        }
        old = change.model;
    }
    for (const auto &change : m_shapeChanges) {
        const auto &bodies = m_storage->m_model.bodies;
        m_storage->ClearContactHistory(bodies[change.model.meta.y].meta.w);
        auto &old = m_storage->m_model.shapes[change.index];
        if (old.meta.y != change.model.meta.y || ((old.meta.w ^ change.model.meta.w) & ShapeSensor) != 0u) {
            m_storage->m_sweepCounts[bodies[old.meta.y].meta.w].discrete.reset();
            m_storage->m_sweepCounts[bodies[change.model.meta.y].meta.w].discrete.reset();
        }
        old = change.model;
    }
    for (const auto &change : m_jointChanges)
        m_storage->m_model.joints[change.index] = change.author;
    m_storage->m_jointLayout = std::move(m_jointLayout);
    m_storage->m_modelUpdatePending = false;
    m_storage = nullptr;
}

std::optional<rhi::SubmissionTicket> JoltGpuModelUpdate::TrySubmit()
{
    if (!m_storage)
        throw std::logic_error("Jolt GPU model transaction is no longer pending");
    std::optional<rhi::SubmissionTicket> ticket;
    try {
        ticket = m_batch->TrySubmit();
    } catch (...) {
        // Acceptance remains final if later RHI bookkeeping throws. No new
        // allocations or validation are needed to install the prepared delta.
        if (m_batch->GetSubmission().IsValid())
            Commit();
        throw;
    }
    if (ticket)
        Commit();
    return ticket;
}

void JoltGpuModelUpdate::Cancel() noexcept
{
    if (!m_storage)
        return;
    (void)m_batch->Cancel();
    m_storage->m_modelUpdatePending = false;
    m_storage = nullptr;
}

void JoltGpuStepInputs::Capture(const JoltGpuStorage &storage, Resource resource)
{
    if (resource == Resource::ConvexGeometry) {
        CaptureGeometry(storage.GetGeometry(), storage.HasGenericGeometry());
        return;
    }
    const auto slot = InputIndex(resource);
    if (slot >= m_buffers.size())
        throw std::invalid_argument("Jolt GPU tick inputs accept only parameters, models, worlds and forces");
    const auto &buffer = storage.Get(resource);
    m_buffers[slot] = !buffer || resource == Resource::BodyForces ? buffer : buffer->Freeze();
}

void JoltGpuStepInputs::CaptureParameters(const JoltGpuStorage &storage,
                                          const std::shared_ptr<rhi::ComputeBuffer> &buffer)
{
    if (!buffer || buffer->GetByteSize() != sizeof(StepParams) ||
        !storage.GetHost().SharesServicesWith(buffer->GetHost()) || !buffer->HasInitializedRange(0, sizeof(StepParams)))
        throw std::logic_error("GPU member parameters require their accepted upload");
    m_buffers[InputIndex(Resource::StepParams)] = buffer->Freeze();
}

void JoltGpuStepInputs::CaptureGeometry(std::shared_ptr<const JoltGpuGeometryLease> geometry, bool generic)
{
    if (!geometry || !geometry->buffer)
        throw std::logic_error("Convex geometry was not accepted before tick input capture");
    m_geometry = std::move(geometry);
    m_hasGenericGeometry = generic;
}

const std::shared_ptr<rhi::ComputeBuffer> &JoltGpuStepInputs::GetCaptured(Resource resource) const
{
    if (resource == Resource::ConvexGeometry) {
        if (!m_geometry)
            throw std::logic_error("No captured convex geometry lease");
        return m_geometry->buffer;
    }
    const auto slot = InputIndex(resource);
    if (slot >= m_buffers.size())
        throw std::invalid_argument("Jolt GPU captured inputs accept only parameters, models, worlds and forces");
    return m_buffers[slot];
}

void JoltGpuStepInputs::SetContacts(std::shared_ptr<const JoltGpuContactWorkset> contacts)
{
    if (!contacts || contacts->patchSlots > std::numeric_limits<uint32_t>::max() / MaxContactPointsPerPair)
        throw std::invalid_argument("Jolt GPU contact workset has an invalid dispatch extent");
    const auto &authorWorlds = GetCaptured(Resource::Worlds);
    if (!authorWorlds)
        throw std::invalid_argument("Jolt GPU contact workset requires its captured author world prefix");
    for (const auto &buffer : {contacts->parameters, contacts->contacts, contacts->patches})
        if (!buffer || !authorWorlds->GetHost().SharesServicesWith(buffer->GetHost()))
            throw std::invalid_argument("Jolt GPU contact workset resources must share their captured compute owner");
    const auto contactCount = uint64_t(contacts->patchSlots) * MaxContactPointsPerPair;
    if (contacts->parameters->GetByteSize() != sizeof(StepParams) ||
        contacts->contacts->GetByteSize() < std::max(uint64_t{1}, contactCount) * sizeof(Contact) ||
        contacts->contacts->GetByteSize() % sizeof(Contact) ||
        contacts->patches->GetByteSize() < sizeof(ContactPatch) ||
        contacts->patches->GetByteSize() % sizeof(ContactPatch))
        throw std::invalid_argument("Jolt GPU contact workset differs from its logical resource layout");
    ClearConstraints();
    m_contacts = std::move(contacts);
}

void JoltGpuStepInputs::ClearContacts() noexcept
{
    m_contacts.reset();
    m_previousContacts = {};
    m_nextContacts = {};
    ClearConstraints();
}

void JoltGpuStepInputs::SetContactHistory(JoltGpuContactHistory previous, JoltGpuContactHistory next) noexcept
{
    m_previousContacts = std::move(previous);
    m_nextContacts = std::move(next);
}

void JoltGpuStepInputs::SetConstraints(std::shared_ptr<const JoltGpuConstraintWorkset> constraints) noexcept
{
    m_constraints = std::move(constraints);
}

void JoltGpuStepInputs::ClearConstraints() noexcept
{
    m_constraints.reset();
}

JoltGpuPrograms::JoltGpuPrograms(rhi::ComputeHost &host, const ShaderTargetCode &target,
                                 const ShaderTargetCode *captureTarget, const ShaderTargetCode *dynamicsTarget)
    : m_host(host)
{
    m_sleep = std::make_shared<JoltGpuSleep>(host, target);
    m_emptyArticulationControl = Allocate(m_host, PortIndex(Resource::ArticulationControl), ArtControlHeaderBytes);
    m_articulations =
        std::make_shared<JoltGpuArticulations>(host, target, m_emptyArticulationControl, captureTarget, dynamicsTarget);
    for (const auto &definition : Stages) {
        const auto code = std::find_if(target.entries.begin(), target.entries.end(),
                                       [&](const auto &entry) { return entry.name == definition.entry; });
        if (code == target.entries.end() || code->stage != rhi::ShaderStage::Compute ||
            code->workgroupSize !=
                std::array<uint32_t, 3>{definition.groupSizeX, definition.groupSizeY, definition.groupSizeZ})
            throw std::invalid_argument("Jolt GPU target is missing a matching stage entry");

        Entry entry{definition, {}, {}, {}};
        std::vector<rhi::ComputeBufferBinding> bindings;
        uint32_t actualResources = 0;
        uint32_t storageBindings = 0, uniformBindings = 0;
        for (const auto &resource : target.resources) {
            const std::array<uint32_t, 2> key{resource.set, resource.binding.binding};
            if (std::find(code->bindings.begin(), code->bindings.end(), key) == code->bindings.end())
                continue;
            const auto port = std::find_if(std::begin(Ports), std::end(Ports),
                                           [&](const Port &value) { return value.name == resource.name; });
            if (port == std::end(Ports) || resource.set != 0)
                throw std::invalid_argument("Jolt GPU stage contains an unknown resource port");
            ValidateLayout(*port, resource);
            if (resource.binding.type == rhi::BindingType::UniformBuffer)
                ++uniformBindings;
            else
                ++storageBindings;
            if ((definition.writes & Mask(port->resource)) && resource.binding.readOnlyStorage)
                throw std::invalid_argument("Jolt GPU writer cannot bind a read-only target port");
            bindings.push_back({resource.binding.binding, resource.binding.type, resource.binding.readOnlyStorage});
            const bool read = (definition.reads & Mask(port->resource)) != 0;
            const bool write = (definition.writes & Mask(port->resource)) != 0;
            const auto access = read && write ? rhi::ComputeBufferAccess::ReadWrite
                                : write       ? rhi::ComputeBufferAccess::Write
                                              : rhi::ComputeBufferAccess::Read;
            entry.bindings.push_back({PortIndex(port->resource), access, read,
                                      (CompleteWrites(definition.stage) & Mask(port->resource)) != 0});
            actualResources |= Mask(port->resource);
        }
        if (actualResources != (definition.reads | definition.writes) || storageBindings > 8 || uniformBindings > 2)
            throw std::invalid_argument("Jolt GPU entry resources differ from the stage recipe");

        if (definition.directEntry) {
            const auto direct = std::find_if(target.entries.begin(), target.entries.end(),
                                             [&](const auto &entry) { return entry.name == definition.directEntry; });
            if (direct == target.entries.end() || direct->stage != code->stage ||
                direct->workgroupSize != code->workgroupSize || direct->bindings != code->bindings)
                throw std::invalid_argument("Direct and tiled Jolt contact stages require the same resource layout");
            entry.directKernel = PrepareJoltGpuKernel(m_host, target.target, *direct, bindings);
        }
        entry.kernel = PrepareJoltGpuKernel(m_host, target.target, *code, std::move(bindings));
        m_entries[static_cast<size_t>(definition.stage)] = std::move(entry);
    }
    for (size_t i = 0; i < std::size(BodyBatchDefinitions); ++i) {
        const auto &definition = BodyBatchDefinitions[i];
        const auto code = std::find_if(target.entries.begin(), target.entries.end(),
                                       [&](const auto &entry) { return entry.name == definition.entry; });
        if (code == target.entries.end() || code->stage != rhi::ShaderStage::Compute ||
            code->workgroupSize != std::array<uint32_t, 3>{64, 1, 1} || code->rootConstants)
            throw std::invalid_argument("Jolt GPU target is missing a matching body batch entry");
        const auto &single = Stages[static_cast<size_t>(definition.stage)];
        const uint32_t reads = single.reads & ~(Mask(Resource::StepParams) | Mask(Resource::StageWork) |
                                                Mask(Resource::ArticulationControl));
        const uint32_t expected = reads | single.writes | (7u << PortCount);
        uint32_t actual = 0, storageBindings = 0, uniformBindings = 0;
        BodyBatchEntry entry;
        std::vector<rhi::ComputeBufferBinding> bindings;
        for (const auto &resource : target.resources) {
            const std::array<uint32_t, 2> key{resource.set, resource.binding.binding};
            if (std::find(code->bindings.begin(), code->bindings.end(), key) == code->bindings.end())
                continue;
            if (resource.set != 0)
                throw std::invalid_argument("Body batch contains a resource outside its one descriptor set");
            const auto table = std::find(BodyBatchInputNames.begin(), BodyBatchInputNames.end(), resource.name);
            size_t slot;
            bool read = true, write = false;
            if (table != BodyBatchInputNames.end()) {
                slot = PortCount + size_t(table - BodyBatchInputNames.begin());
                ValidateBodyBatchInput(slot - PortCount, resource);
            } else {
                const auto port = std::find_if(std::begin(Ports), std::end(Ports),
                                               [&](const auto &value) { return value.name == resource.name; });
                if (port == std::end(Ports))
                    throw std::invalid_argument("Body batch contains an unknown resource");
                slot = PortIndex(port->resource);
                ValidateLayout(*port, resource);
                read = (reads & Mask(port->resource)) != 0;
                write = (single.writes & Mask(port->resource)) != 0;
                if ((!read && !write) || (write && resource.binding.readOnlyStorage))
                    throw std::invalid_argument("Body batch resource access differs from its body stage");
            }
            if (resource.binding.type == rhi::BindingType::UniformBuffer)
                ++uniformBindings;
            else
                ++storageBindings;
            const auto access = read && write ? rhi::ComputeBufferAccess::ReadWrite
                                : write       ? rhi::ComputeBufferAccess::Write
                                              : rhi::ComputeBufferAccess::Read;
            entry.bindings.push_back({slot, access, read, false});
            bindings.push_back({resource.binding.binding, resource.binding.type, resource.binding.readOnlyStorage});
            actual |= 1u << slot;
        }
        if (actual != expected || storageBindings != 6 || uniformBindings != 1)
            throw std::invalid_argument("Body batch resources differ from its exact six-storage recipe");
        entry.kernel = PrepareJoltGpuKernel(m_host, target.target, *code, std::move(bindings));
        m_bodyBatchEntries[i] = std::move(entry);
    }
}

bool JoltGpuPrograms::TryPrepareStartupPrograms(bool hasLocalContacts)
{
    bool ready = m_sleep->TryPrepareStartupPrograms();
    for (const auto stage :
         {Stage::PrepareKinematics, Stage::ApplyForces, Stage::BuildBounds, Stage::CollidePairs, Stage::Integrate})
        ready &= m_entries[static_cast<size_t>(stage)].kernel->TryPreparePipeline();
    if (hasLocalContacts) {
        for (const auto stage : {Stage::SetupContacts, Stage::StoreContactHistory})
            ready &= m_entries[static_cast<size_t>(stage)].kernel->TryPreparePipeline();
        for (const auto stage : {Stage::WarmContacts, Stage::SolveVelocity, Stage::SolvePosition}) {
            const auto &entry = m_entries[static_cast<size_t>(stage)];
            ready &= entry.directKernel->TryPreparePipeline();
            ready &= entry.kernel->TryPreparePipeline();
        }
        ready &= m_entries[static_cast<size_t>(Stage::SolveVelocityRounds)].kernel->TryPreparePipeline();
        ready &= m_entries[static_cast<size_t>(Stage::SolvePositionRounds)].kernel->TryPreparePipeline();
    }
    return ready;
}

rhi::ComputeDispatchDesc
JoltGpuPrograms::BuildDispatch(const Entry &entry, const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                               const JoltGpuStageWork &work, const std::shared_ptr<rhi::ComputeBuffer> &workUniform,
                               const std::shared_ptr<const JoltGpuContactWorkset> &resident) const
{
    rhi::ComputeDispatchDesc dispatch;
    dispatch.kernel = entry.kernel;
    dispatch.buffers.reserve(entry.bindings.size());
    dispatch.bufferAccesses.reserve(entry.bindings.size());
    rhi::ComputeDispatchProof proof;
    proof.reads.reserve(entry.bindings.size());
    proof.writes.reserve(entry.bindings.size());
    const auto range = work.parameters.range;
    const auto &contacts = resident ? resident : inputs.GetContacts();
    const auto &history = inputs.NextContactHistory();
    const bool updateHistoryPrefix = work.stage == Stage::StoreContactHistory && contacts->countedArguments &&
                                     history.buffer->HasInitializedRange(0, history.ByteSize());
    for (size_t index = 0; index < entry.bindings.size(); ++index) {
        const auto &binding = entry.bindings[index];
        const bool writer = binding.access != rhi::ComputeBufferAccess::Read;
        const auto resource = Ports[binding.slot].resource;
        const auto input = InputIndex(resource);
        const auto &constraints = inputs.GetConstraints();
        const auto &buffer =
            resource == Resource::ArticulationControl
                ? (inputs.m_articulations ? inputs.m_articulations->control : m_emptyArticulationControl)
            : resource == Resource::ConvexGeometry     ? inputs.GetCaptured(resource)
            : resource == Resource::StageWork          ? workUniform
            : resource == Resource::SweepBodies        ? inputs.GetSweepBodies()
            : resource == Resource::ConstraintSchedule ? (constraints ? constraints->schedule : nullptr)
            : resource == Resource::ContactHistory
                ? (writer ? inputs.NextContactHistory().buffer : inputs.PreviousContactHistory().buffer)
            : contacts && resource == Resource::StepParams     ? contacts->parameters
            : contacts && resource == Resource::Contacts       ? contacts->contacts
            : contacts && resource == Resource::ContactPatches ? contacts->patches
            : !writer && input < inputs.m_buffers.size()       ? inputs.m_buffers[input]
                                                               : storage.m_buffers[binding.slot];
        if (!buffer || !m_host.SharesServicesWith(buffer->GetHost()))
            throw std::invalid_argument("Jolt GPU stage requires captured input: " +
                                        std::string(Ports[binding.slot].name));
        dispatch.buffers.push_back(buffer);
        const bool partialHistory = resource == Resource::ContactHistory && updateHistoryPrefix;
        dispatch.bufferAccesses.push_back(partialHistory ? rhi::ComputeBufferAccess::ReadWrite : binding.access);
        if (partialHistory)
            proof.reads.push_back({static_cast<uint32_t>(index), 0, history.ByteSize()});
        if (binding.read) {
            if (resource == Resource::ArticulationControl) {
                proof.reads.push_back(
                    {static_cast<uint32_t>(index), 0,
                     inputs.m_articulations ? inputs.m_articulations->header.storage.z : ArtControlHeaderBytes});
            } else if (resource == Resource::ConvexGeometry) {
                for (const auto &[first, end] : inputs.m_geometry->wordRanges)
                    proof.reads.push_back(
                        {static_cast<uint32_t>(index), first * sizeof(UInt4), (end - first) * sizeof(UInt4)});
            } else if (resource == Resource::ContactHistory)
                proof.reads.push_back({static_cast<uint32_t>(index), 0, inputs.PreviousContactHistory().ByteSize()});
            else
                proof.reads.push_back({static_cast<uint32_t>(index), 0, buffer->GetByteSize()});
        }
        if (binding.completeWrite && !partialHistory) {
            const auto complete = [&](uint64_t first, uint64_t count, uint64_t stride) {
                const uint64_t offset = first * stride, bytes = count * stride;
                if (offset > buffer->GetByteSize() || bytes > buffer->GetByteSize() - offset)
                    throw std::invalid_argument("Jolt GPU work complete-write range exceeds its typed resource");
                if (bytes)
                    proof.writes.push_back({static_cast<uint32_t>(index), offset, bytes});
            };
            if (resource == Resource::Bounds) {
                complete(range.x, range.y, sizeof(Aabb));
            } else if (resource == Resource::SweepBodies) {
                complete(range.x - inputs.GetParameters().selection.y, range.y, sizeof(SweepBody));
            } else if (resource == Resource::Contacts) {
                complete(uint64_t(range.x) * MaxContactPointsPerPair, uint64_t(range.y) * MaxContactPointsPerPair,
                         sizeof(Contact));
            } else if (resource == Resource::ContactHistory) {
                complete(range.x ? 1u + uint64_t(range.x) * ContactHistoryWords : 0u,
                         uint64_t(range.y) * ContactHistoryWords + (range.x == 0u), sizeof(QueryRecordWord));
            } else {
                throw std::invalid_argument("Jolt GPU stage lacks a typed bounded complete-write domain");
            }
        }
    }
    dispatch.proof = std::move(proof);
    const auto &definition = entry.definition;
    dispatch.groupCountY = dispatch.groupCountZ = 1;
    if (IslandDomain(definition.domain)) {
        const auto &constraints = *inputs.GetConstraints();
        const bool joints = definition.domain == Domain::IslandJoints;
        const bool direct = joints || constraints.patchRefCount < ContactParallelMinimum;
        if (entry.directKernel && direct)
            dispatch.kernel = entry.directKernel;
        dispatch.groupCountX = direct ? Groups(range.w, definition.groupSizeX) : range.w;
        const auto record = constraints.Dispatches(joints)[work.islandDispatch].argumentRecord;
        dispatch.indirectArguments = rhi::ComputeIndirectArguments{constraints.solveArguments,
                                                                   uint64_t(record) * sizeof(ConstraintScheduleWord)};
    } else {
        dispatch.groupCountX = Groups(range.y, definition.groupSizeX);
    }
    if (definition.stage == Stage::SetupContacts)
        dispatch.indirectArguments =
            contacts->Dispatch(JoltGpuContactWorkset::Range::Setup, range.x / MaxContactPointsPerPair);
    if (updateHistoryPrefix)
        dispatch.indirectArguments = contacts->Dispatch(JoltGpuContactWorkset::Range::History, range.x);
    const auto &maximum = m_host.device.GetCapabilities().limits.maxComputeWorkgroupCount;
    if (!dispatch.groupCountX || !dispatch.groupCountY || !dispatch.groupCountZ || dispatch.groupCountX > maximum[0] ||
        dispatch.groupCountY > maximum[1] || dispatch.groupCountZ > maximum[2])
        throw std::invalid_argument("Jolt GPU bounded work exceeds the device dispatch range");
    return dispatch;
}

void JoltGpuPrograms::AppendResidentCollide(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                                            const std::shared_ptr<const JoltGpuContactWorkset> &contacts,
                                            uint32_t capacity, const StageWork &work,
                                            const std::shared_ptr<rhi::ComputeBuffer> &uniform,
                                            const rhi::ComputeIndirectArguments &arguments,
                                            std::vector<rhi::ComputeBufferUpdate> &uploads,
                                            std::vector<rhi::ComputeDispatchDesc> &dispatches)
{
    const auto range = work.range;
    if (!m_host.SharesServicesWith(storage.GetHost()) || !contacts || range.z || range.w || !range.y ||
        range.y > std::max(WorkItemLimit(Domain::Patches), ConstraintLocalRefs) || range.x > capacity ||
        range.y > capacity - range.x || !contacts->contacts ||
        contacts->contacts->GetByteSize() < uint64_t(capacity) * MaxContactPointsPerPair * sizeof(Contact))
        throw std::invalid_argument("Resident collide requires its bounded admitted capacity");
    if (!uniform || uniform->IsFrozen() || uniform->GetByteSize() != sizeof(StageWork) ||
        !m_host.SharesServicesWith(uniform->GetHost()))
        throw std::invalid_argument("Resident collide requires its admitted StageWork uniform slot");
    uploads.push_back({uniform, 0, Bytes(&work, 1)});
    auto dispatch = BuildDispatch(m_entries[static_cast<size_t>(Stage::CollidePairs)], storage, inputs,
                                  {Stage::CollidePairs, work}, uniform, contacts);
    dispatch.indirectArguments = arguments;
    // Indirect work cannot manufacture initialized coverage. Direct bounded
    // initialization established every Contact capacity slot beforehand.
    dispatch.proof->writes.clear();
    for (size_t i = 0; i < m_entries[static_cast<size_t>(Stage::CollidePairs)].bindings.size(); ++i)
        if (Ports[m_entries[static_cast<size_t>(Stage::CollidePairs)].bindings[i].slot].resource ==
            Resource::Contacts) {
            // This dispatch retains initialized capacity even where indirect
            // groups do not write. Its actual binding must carry the Read bit
            // required by that proof; the shared shader recipe stays Write.
            dispatch.bufferAccesses[i] = rhi::ComputeBufferAccess::ReadWrite;
            dispatch.proof->reads.push_back({static_cast<uint32_t>(i), 0, contacts->contacts->GetByteSize()});
        }
    dispatches.push_back(std::move(dispatch));
}

void JoltGpuPrograms::AppendStages(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                                   const std::vector<JoltGpuStageWork> &work,
                                   std::vector<rhi::ComputeBufferUpdate> &uploads,
                                   std::vector<rhi::ComputeDispatchDesc> &dispatches)
{
    if (!m_host.SharesServicesWith(storage.GetHost()))
        throw std::invalid_argument("Jolt GPU programs and state belong to different compute owners");
    if (storage.GetBodyCount() == 0 || work.empty())
        return;
    dispatches.reserve(dispatches.size() + work.size());
    std::map<std::array<uint32_t, 4>, std::shared_ptr<rhi::ComputeBuffer>> uniforms;
    for (const auto &unit : work) {
        const auto index = static_cast<size_t>(unit.stage);
        if (index >= m_entries.size())
            throw std::invalid_argument("Unknown Jolt GPU stage");
        const auto &entry = m_entries[index];
        if (!ValidateWork(storage, inputs, entry.definition, unit))
            continue;
        const auto range = unit.parameters.range;
        const std::array<uint32_t, 4> key{range.x, range.y, range.z, range.w};
        auto uniform = uniforms.find(key);
        if (uniform == uniforms.end()) {
            auto cached = std::find_if(m_workUniforms.begin(), m_workUniforms.end(),
                                       [&](const auto &value) { return value.range == key; });
            std::shared_ptr<rhi::ComputeBuffer> buffer;
            if (cached != m_workUniforms.end() && cached->buffer->HasInitializedRange(0, sizeof(StageWork))) {
                buffer = cached->buffer;
            } else {
                buffer = Allocate(m_host, PortIndex(Resource::StageWork), sizeof(StageWork));
                uploads.push_back({buffer, 0, Bytes(&unit.parameters, 1)});
                const WorkUniform value{key, buffer};
                if (cached != m_workUniforms.end())
                    *cached = value;
                else if (m_workUniforms.size() < WorkUniformCacheCapacity)
                    m_workUniforms.push_back(value);
                else {
                    m_workUniforms[m_nextWorkUniform] = value;
                    m_nextWorkUniform = (m_nextWorkUniform + 1) % WorkUniformCacheCapacity;
                }
            }
            uniform = uniforms.emplace(key, std::move(buffer)).first;
        }
        if ((entry.definition.reads & Mask(Resource::ArticulationControl)) && !inputs.m_articulations &&
            !m_emptyArticulationControl->HasInitializedRange(0, ArtControlHeaderBytes) &&
            std::none_of(uploads.begin(), uploads.end(),
                         [&](const auto &u) { return u.buffer == m_emptyArticulationControl; })) {
            const ArticulationControlHeader empty{{0, 0, 0, 0}, {0, 48, 48, 48}, {48, 48, 48, 48}};
            uploads.push_back({m_emptyArticulationControl, 0, Bytes(&empty, 1)});
        }
        dispatches.push_back(BuildDispatch(entry, storage, inputs, unit, uniform->second));
    }
}

std::shared_ptr<rhi::PreparedComputeBatch>
JoltGpuPrograms::PrepareBodyBatch(const JoltGpuStorage &storage, const std::vector<JoltGpuBodyBatchMember> &members,
                                  JoltGpuBodyBatchInputs &inputs)
{
    if (!m_host.SharesServicesWith(storage.GetHost()) || members.empty())
        throw std::invalid_argument("Body batch requires ready units on its shared storage owner");
    const auto definition = std::find_if(std::begin(BodyBatchDefinitions), std::end(BodyBatchDefinitions),
                                         [&](const auto &value) { return value.stage == members.front().work.stage; });
    if (definition == std::end(BodyBatchDefinitions))
        throw std::invalid_argument("Body batch requires kinematic setup, forces or integrate");
    const auto &entry = m_bodyBatchEntries[size_t(definition - std::begin(BodyBatchDefinitions))];
    const auto &single = Stages[static_cast<size_t>(definition->stage)];
    const uint32_t maxTiles = WorkItemLimit(Domain::Bodies) / 64u;
    if (members.size() > maxTiles)
        throw std::invalid_argument("Body batch member padding exceeds its bounded body quantum");
    std::vector<StepParams> selections;
    std::vector<BodyBatchTile> tiles;
    std::vector<std::pair<uint32_t, uint32_t>> intervals;
    for (const auto &member : members) {
        if (!member.inputs || member.work.stage != definition->stage ||
            !ValidateWork(storage, *member.inputs, single, member.work))
            throw std::invalid_argument("Body batch requires nonempty units of one ready body stage");
        const auto &selected = member.inputs->GetParameters();
        const auto range = member.work.parameters.range;
        if (selected.counts.x != storage.GetBodyCount() || selected.counts.y != storage.GetWorldCount() ||
            selected.counts.z != storage.GetShapeCount() || selected.selection.x >= storage.GetWorldCount())
            throw std::invalid_argument("Body batch selection differs from its accepted shared extents");
        const auto &world = storage.GetModel().worlds[selected.selection.x];
        if (selected.selection.y != world.bodies.x || selected.selectedCounts.x != world.bodies.y ||
            selected.selection.z != world.leaves.x || selected.selectedCounts.y != world.leaves.y ||
            selected.selection.w != world.leaves.z || selected.selectedCounts.z || world.leaves.w)
            throw std::invalid_argument("Body batch selection differs from its accepted no-joint WorldRange");
        for (const auto resource : {Resource::BodyModels, Resource::Worlds}) {
            const auto &captured = member.inputs->GetCaptured(resource);
            if (!captured || captured->GetAllocation() != storage.Get(resource)->GetAllocation())
                throw std::invalid_argument("Body batch captured author allocation is no longer current");
        }
        if (definition->stage == Stage::ApplyForces) {
            const auto &forces = member.inputs->GetCaptured(Resource::BodyForces);
            if (!forces || forces->GetAllocation() != storage.Get(Resource::BodyForces)->GetAllocation())
                throw std::invalid_argument("Body batch force endpoint is no longer its captured live input");
        }
        const uint32_t count = Groups(range.y, 64);
        if (tiles.size() > maxTiles || count > maxTiles - tiles.size())
            throw std::invalid_argument("Body batch padded tiles exceed its bounded body quantum");
        // Body helpers read gravityDt and selection/counts. The resident contact
        // producer only overrides solve.zw, which this body closure never reads.
        // Do not reconstruct its unused GPU-only header words on the CPU.
        const auto selection = static_cast<uint32_t>(selections.size());
        selections.push_back(selected);
        intervals.emplace_back(range.x, range.y);
        for (uint32_t offset = 0; offset < range.y;) {
            const auto count = std::min(uint32_t{64}, range.y - offset);
            tiles.push_back({selection, range.x + offset, count});
            offset += count;
        }
    }
    std::sort(intervals.begin(), intervals.end());
    for (size_t i = 1; i < intervals.size(); ++i)
        if (uint64_t(intervals[i - 1].first) + intervals[i - 1].second > intervals[i].first)
            throw std::invalid_argument("Body batch units overlap writable body rows");
    const auto &caps = m_host.device.GetCapabilities();
    const std::array<uint64_t, 3> bytes{sizeof(BodyBatchHeader), selections.size() * sizeof(StepParams),
                                        tiles.size() * sizeof(BodyBatchTile)};
    if (tiles.size() > caps.limits.maxComputeWorkgroupCount[0] || caps.limits.maxComputeWorkgroupCount[1] < 1 ||
        caps.limits.maxComputeWorkgroupCount[2] < 1 || !caps.IsStorageBufferRangeValid(0, bytes[1]) ||
        !caps.IsStorageBufferRangeValid(0, bytes[2]))
        throw std::invalid_argument("Body batch exceeds its actual device table or dispatch limit");
    for (size_t i = 0; i < inputs.buffers.size(); ++i) {
        auto &buffer = inputs.buffers[i];
        if (buffer && (!m_host.SharesServicesWith(buffer->GetHost()) || buffer->IsFrozen()))
            throw std::invalid_argument("Body batch tables require their source-owned mutable Compute inputs");
        if (!buffer || buffer->GetByteSize() < bytes[i]) {
            auto grown = rhi::ComputeBuffer::TryCreatePrivate(
                m_host, {bytes[i] / sizeof(uint32_t), rhi::ComputeScalarType::UInt32, 1});
            if (!grown)
                return {}; // No member unit or upload was accepted.
            buffer = std::move(grown);
        }
    }
    const BodyBatchHeader header{{static_cast<uint32_t>(selections.size()), static_cast<uint32_t>(tiles.size()),
                                  storage.GetBodyCount(), storage.GetWorldCount()}};
    std::vector<rhi::ComputeBufferUpdate> uploads{{inputs.buffers[0], 0, Bytes(&header, 1)},
                                                  {inputs.buffers[1], 0, Bytes(selections.data(), selections.size())},
                                                  {inputs.buffers[2], 0, Bytes(tiles.data(), tiles.size())}};
    rhi::ComputeDispatchDesc dispatch;
    dispatch.kernel = entry.kernel;
    dispatch.groupCountX = static_cast<uint32_t>(tiles.size());
    dispatch.groupCountY = dispatch.groupCountZ = 1;
    rhi::ComputeDispatchProof proof;
    for (const auto &binding : entry.bindings) {
        const auto index = static_cast<uint32_t>(dispatch.buffers.size());
        const auto &buffer =
            binding.slot >= PortCount ? inputs.buffers[binding.slot - PortCount] : storage.m_buffers[binding.slot];
        dispatch.buffers.push_back(buffer);
        dispatch.bufferAccesses.push_back(binding.access);
        if (binding.slot >= PortCount) {
            proof.reads.push_back({index, 0, bytes[binding.slot - PortCount]});
        } else if (binding.read) {
            for (const auto &[first, count] : intervals)
                proof.reads.push_back({index, uint64_t(first) * Ports[binding.slot].stride,
                                       uint64_t(count) * Ports[binding.slot].stride});
        }
    }
    // These helpers retain prior state on legal early exits; never claim a
    // complete body write. RHI preservation/proof admission remains unchanged.
    dispatch.proof = std::move(proof);
    return rhi::PrepareComputeBatch(m_host, std::move(uploads), {std::move(dispatch)});
}

std::shared_ptr<rhi::PreparedComputeBatch> JoltGpuPrograms::PrepareStages(const JoltGpuStorage &storage,
                                                                          const JoltGpuStepInputs &inputs,
                                                                          const std::vector<JoltGpuStageWork> &work)
{
    std::vector<rhi::ComputeBufferUpdate> uploads;
    std::vector<rhi::ComputeDispatchDesc> dispatches;
    AppendStages(storage, inputs, work, uploads, dispatches);
    if (dispatches.empty())
        return {};
    return rhi::PrepareComputeBatch(m_host, std::move(uploads), std::move(dispatches));
}

} // namespace infernux
