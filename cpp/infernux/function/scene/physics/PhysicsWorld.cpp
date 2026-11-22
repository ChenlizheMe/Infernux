#include "PhysicsWorld.h"
#include <Jolt/Jolt.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <Jolt/RegisterTypes.h>
#include "../BoxCollider.h"
#include "../Collider.h"
#include "../Component.h"
#include "../GameObject.h"
#include "../MeshCollider.h"
#include "../Ragdoll.h"
#include "../Rigidbody.h"
#include "../RuntimeWorld.h"
#include "../Scene.h"
#include "../SceneManager.h"
#include "../TagLayerManager.h"
#include "../Transform.h"
#include "../TransformECSStore.h"
#include "JoltCpuControl.h"
#include "JoltCpuQueries.h"
#include "PhysicsBodyCommand.h"
#include "PhysicsContactTypes.h"
#include "PhysicsDomain.h"
#include "PhysicsGpuRenderBindings.h"
#include "PhysicsGpuRenderPoses.h"
#include "PhysicsLayers.h"
#include "PhysicsMass.h"
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
namespace
{
JPH::Vec3 CpuVector(const glm::vec4 &v)
{
    return JPH::Vec3(v.x, v.y, v.z);
}
JPH::RVec3 CpuPosition(const glm::vec4 &v)
{
    return JPH::RVec3(v.x, v.y, v.z);
}
JPH::Quat CpuRotation(const glm::vec4 &q)
{
    return JPH::Quat(q.x, q.y, q.z, q.w);
}
bool Finite(const glm::vec3 &v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}
glm::vec4 Vector(const glm::vec3 &v)
{
    if (!Finite(v))
        throw std::invalid_argument("Physics vector must be finite");
    return {v.x, v.y, v.z, 0};
}
glm::vec3 Vector(const glm::vec4 &v)
{
    return {v.x, v.y, v.z};
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
            throw std::invalid_argument("Physics state export has an invalid destination");
        for (size_t index = 0; index < sizes.size(); ++index) {
            const auto &prior = bindings[index];
            if ((prior.buffer->GetAllocation() == binding.buffer->GetAllocation() ||
                 prior.buffer->GetWriterStateIdentity() == binding.buffer->GetWriterStateIdentity()) &&
                size && sizes[index] && binding.byteOffset < prior.byteOffset + sizes[index] &&
                prior.byteOffset < binding.byteOffset + size)
                throw std::invalid_argument("Physics state export output ranges overlap");
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
        throw std::invalid_argument(error.what());
    } catch (const std::length_error &error) {
        throw std::invalid_argument(error.what());
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
    throw std::invalid_argument("Physics aggregate export requires a floating point field");
}

std::shared_ptr<rhi::PreparedComputeBatch>
PrepareCpuStateExports(const PhysicsRigidbodyStateBatch &batch, const std::vector<uint8_t> &cpuRows,
                       const std::vector<std::vector<PhysicsRigidbodyBoxState>> &boxes,
                       const std::vector<uint64_t> &shapeFirst, const std::vector<PhysicsDeviceQueryOutput> &bindings,
                       bool includeBoxes)
{
    using Field = PhysicsDeviceQueryField;
    if (!bindings.front().buffer)
        throw std::invalid_argument("Physics aggregate export requires a compute destination");
    auto &host = bindings.front().buffer->GetHost();
    std::vector<uint64_t> targetSizes;
    targetSizes.reserve(bindings.size());
    std::vector<rhi::ComputeBufferUpdate> updates;
    for (const auto &binding : bindings) {
        const auto field = static_cast<uint32_t>(binding.field);
        const auto &buffer = binding.buffer;
        if (field >= PhysicsDeviceQueryWidths.size() || !buffer || buffer->IsFrozen() ||
            !host.SharesServicesWith(buffer->GetHost()) || binding.byteOffset % sizeof(uint32_t))
            throw std::invalid_argument("Physics aggregate export field or destination differs from its compute owner");
        const bool motion = binding.field <= Field::MotionInverseInertia;
        if (!motion && !includeBoxes)
            throw std::invalid_argument("Physics aggregate box export requires Box rows");
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
            throw std::invalid_argument("Physics aggregate export destination has invalid type or prefix capacity");
        for (size_t i = 0; i < targetSizes.size(); ++i) {
            const auto &previous = bindings[i];
            if ((previous.buffer->GetWriterStateIdentity() == buffer->GetWriterStateIdentity() ||
                 previous.buffer->GetAllocation() == buffer->GetAllocation()) &&
                bytes && targetSizes[i] && offset < previous.byteOffset + targetSizes[i] &&
                previous.byteOffset < offset + bytes)
                throw std::invalid_argument("Physics aggregate export destination prefixes must not overlap");
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
glm::vec4 Rotation(const glm::quat &v)
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

Rigidbody &KinematicBody(const PhysicsActorData &actor)
{
    auto *body = actor.rigidbody;
    if (!body || !body->IsEnabled() || !body->IsKinematic())
        throw std::logic_error("Kinematic movement requires an enabled kinematic Rigidbody");
    return *body;
}
} // namespace

struct PhysicsStaticBoundaryModel
{
    explicit PhysicsStaticBoundaryModel(const Transform &transform)
    {
        TransformECSStore::PhysicsPoseReadScope authoring(TransformECSStore::Instance());
        position = transform.GetPosition();
        rotation = transform.GetWorldRotation();
    }

    void Hold(PreparedCpuActor actor)
    {
        pending = std::make_unique<PreparedCpuActor>(std::move(actor));
    }
    void ApplyPose(PreparedCpuActor &actor) const
    {
        actor.settings.mPosition = CpuPosition(Vector(position));
        actor.settings.mRotation = CpuRotation(Rotation(rotation));
    }

    glm::vec3 position;
    glm::quat rotation;
    uint64_t positionRevision = 0, rotationRevision = 0;
    // Only a boundary that has not reached its scene backend owns a prepared model here.
    std::unique_ptr<PreparedCpuActor> pending;
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

    PhysicsGpuWork work;
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

    [[nodiscard]] std::shared_ptr<Request> Enqueue(PhysicsGpuWork work)
    {
        auto request = std::make_shared<Request>();
        request->work = std::move(work);
        request->id = m_nextRequest++;
        m_pending.push_back(request);
        m_owner.Wake(m_source);
        return request;
    }

    void Run(PhysicsGpuWork work)
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
    if (domain.cpuScene || domain.nativeRuntime || domain.nativeTopology || !domain.bodyIds.empty())
        CancelPendingStep();
    domain.nativeTopology.reset();
    domain.nativeRuntime.reset();
    domain.backend = backend;
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
PhysicsWorld::GpuPublicationReader &PhysicsWorld::EnsureGpuPublicationReader()
{
    if (!m_gpuWork || !m_gpuHost)
        throw std::runtime_error("GPU physics requires engine compute services");
    if (!m_gpuPublicationReader)
        m_gpuPublicationReader = std::make_unique<GpuPublicationReader>(*m_gpuWork, *m_gpuHost);
    return *m_gpuPublicationReader;
}
void PhysicsWorld::Shutdown()
{
    for (auto *domain : m_domains) {
        domain->RetireInxPhyGpuFixed();
        domain->nativeTopology.reset();
        domain->nativeRuntime.reset();
        while (!domain->gpuJointAuthors.empty())
            domain->RetireGpuJoint(domain->gpuJointAuthors.begin()->first);
        domain->rendering.bindings.Clear();
        domain->rendering.ready.reset();
        if (auto pose = std::exchange(domain->presentationPoseReads, {}))
            GpuPublicationReader::Cancel(pose->request);
        GpuPublicationReader::Cancel(domain->participantOutput);
        GpuPublicationReader::Cancel(domain->participantInitialization);
        domain->contacts.Close();
        domain->cpuContacts.Close();
        RetireRagdolls(*domain);
        RetireCpuJointRoutes(*domain);
        domain->cpuFixedQueryWrite.reset();
        domain->cpuControl.reset();
        domain->cpuScene.reset();
        domain->cpuPublication = {};
        domain->cpuFixedAdmitted = false;
        domain->requestedMotion.clear();
    }
    m_bodyActors.clear();
    for (auto *domain : m_domains) {
        domain->bodyIds.clear();
        domain->pendingStatic.clear();
        domain->pendingStaticMoves.clear();
        domain->admittedStaticMoves.clear();
        domain->recreateActors.clear();
        domain->rendering.checkedKeys.reset();
    }
    m_gpuRenderProgram.reset();
    m_gpuPairQuery.reset();
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

bool PhysicsWorld::PrepareInxPhyGpuTopology(PhysicsDomain &domain, PhysicsFixedInputs input, uint32_t substeps,
                                            bool synchronous)
{
    using namespace physics::inxphygpu;
    const auto epoch = domain.owner.simulation.simulationEpoch;
    if (domain.nativeLayoutEpoch != epoch) {
        domain.RetireInxPhyGpuFixed();
        domain.nativeTopology.reset();
        domain.nativeRuntime.reset();
        domain.nativeLayoutEpoch = epoch;
    }
    const auto observe = [&](InxPhyGpuSceneRuntime &runtime) {
        auto initial = runtime.TakeInitial();
        if (!initial) {
            AdvanceInxPhyGpuSource(runtime, synchronous);
            initial = runtime.TakeInitial();
        }
        return initial;
    };
    for (;;) {
        if (domain.nativeTopology) {
            auto ready = observe(*domain.nativeTopology);
            if (!ready)
                return false;
            domain.nativeRuntime = std::move(domain.nativeTopology);
            for (const auto &[id, revision] : domain.nativeTopologyActors) {
                const auto *entry = FindBodyRegistration(id);
                if (entry && entry->nativeAuthor && entry->stagedShapeRevision == revision) {
                    const auto &accepted = entry->nativeAuthor->publication;
                    accepted.AcceptGeometry(entry->actor, id, revision);
                    accepted.ConsumeInitialVelocities(entry->actor, id);
                }
            }
            domain.nativeTopologyActors.clear();
            AcceptInxPhyGpuPublication(domain, std::make_shared<InxPhyGpuFixedPublication>(std::move(*ready)), true,
                                       {});
        }
        if (domain.nativeRuntime && !domain.nativeBodyLayoutDirty && !domain.gpuJointLayoutDirty &&
            domain.nativeRuntime->Substeps() == substeps)
            return true;
        auto setup = CaptureInxPhyGpuSceneSetup(domain.owner, domain.nativeFixedPublication.get());
        if (!setup) {
            RetryCapacityBodyCreations(domain);
            setup = CaptureInxPhyGpuSceneSetup(domain.owner, domain.nativeFixedPublication.get());
        }
        if (!setup)
            return false;
        domain.nativeTopologyActors.reserve(setup->world.bodies.size());
        for (const auto &body : setup->world.bodies)
            domain.nativeTopologyActors.emplace_back(body.bodyId, body.shapeRevision);
        if (domain.nativeRuntime) {
            domain.nativeTopology =
                domain.nativeRuntime->Rebuild(std::move(*setup), *domain.nativeFixedPublication, input, substeps);
        } else {
            if (!m_gpuHost || !m_gpuWork)
                throw std::runtime_error("GPU physics requires engine compute services");
            const auto &config = EngineConfig::Get();
            const uint64_t shapes = setup->model.geometries.size();
            const auto contacts = std::clamp(shapes * 8, uint64_t{64},
                                             std::max(uint64_t{64}, uint64_t{config.physicsMaxContactConstraints}));
            const auto rows = contacts * 4 + setup->world.joints.size() * 6;
            if (rows > UINT32_MAX / 12)
                throw std::length_error("GPU physics scene exceeds constraint index range");
            const InxPhyGpuCollisionCapacity collision{static_cast<uint32_t>(std::max(contacts, shapes * 16)),
                                                       static_cast<uint32_t>(contacts)};
            InxPhyGpuContactSolver solver{static_cast<uint32_t>(rows), static_cast<uint32_t>(rows * 12)};
            solver.minimumRestitutionSpeed = config.physicsMinVelocityForRestitution;
            solver.linearCastThreshold = config.physicsLinearCastThreshold;
            RegisterDomain(domain);
            domain.nativeTopology = std::make_unique<InxPhyGpuSceneRuntime>(
                *m_gpuWork, *m_gpuHost, PhysicsShaderTarget(*m_gpuHost), setup->model,
                std::vector<InxPhyGpuRegisteredWorld>{std::move(setup->world)}, input.duration,
                ModelVector{input.gravity.x, input.gravity.y, input.gravity.z}, collision, solver, 16, true, substeps);
        }
        domain.nativeBodyLayoutDirty = false;
        domain.AcceptGpuJointEdits();
    }
}

bool PhysicsWorld::TryBeginFixed(PhysicsDomain &domain, bool synchronous)
{
    if (!EnsureStaticBoundaries(domain))
        return false;
    if (domain.UsesGpu()) {
        if (!AdvanceInxPhyGpuEdit(domain, synchronous) ||
            !PrepareInxPhyGpuTopology(domain, domain.AdmittedFixedInputs(), domain.admittedSubsteps, synchronous))
            return false;
        QueueStaticFixedMoves(domain);
        domain.BeginInxPhyGpuFixed({domain.owner.id, domain.stepEpoch, domain.stepIndex}, domain.AdmittedFixedInputs());
        SealStaticFixedMoves(domain);
        return true;
    }
    QueueStaticFixedMoves(domain);
    auto policy = CaptureContactPolicy(domain);
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
    AcceptStaticFixedMoves(domain);
    if (domain.admittedContactPolicy.Requested()) {
        domain.contacts.Publish(identity, domain.admittedContactPolicy, substeps, [&](uint32_t substep) {
            return domain.cpuFixedAdmitted ? domain.cpuContacts.TakeFrame(substep) : PhysicsContactPublication::Frame{};
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
    domain.queryGeneration.fetch_add(1, std::memory_order_release);
    domain.requestedMotion.clear();
    domain.stepCompleted = true;
    domain.cpuFixedQueryWrite.reset();
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
std::shared_ptr<PhysicsGpuRenderJob> PhysicsWorld::CaptureGpuRenderJob(PhysicsDomain &domain,
                                                                       bool numericalChange) const
{
    const WorldExecutionScope ownerScope(&domain.owner);
    FrameWorkOwner::Scope preparation(*m_gpuWork);
    auto &rendering = domain.rendering;
    const auto keys = CaptureRenderKeys();
    const auto generation =
        domain.queryGeneration.load(std::memory_order_acquire) + static_cast<uint64_t>(numericalChange);
    if (rendering.ready && rendering.ready->numericalGeneration == generation && rendering.ready->keys == keys) {
        return {};
    }
    auto snapshot = rendering.bindings.Capture(domain.owner);
    if (!rendering.bindings.NeedsExport(*snapshot, generation)) {
        rendering.checkedKeys = keys;
        return {};
    }
    auto job = std::make_shared<PhysicsGpuRenderJob>();
    job->snapshot = std::move(snapshot);
    job->keys = keys;
    return job;
}

PhysicsGpuRenderPoses &PhysicsWorld::EnsureGpuRenderPoses()
{
    if (!m_gpuRenderProgram)
        m_gpuRenderProgram = std::make_unique<PhysicsGpuRenderPoses>(
            *m_gpuHost, physics::inxphygpu::LoadInxPhyGpuProgram("render_poses", PhysicsShaderTarget(*m_gpuHost)));
    return *m_gpuRenderProgram;
}

namespace
{
physics::inxphygpu::InxPhyGpuSceneRuntime::PublicationConsumer
PrepareInxPhyGpuRender(PhysicsDomain &domain, PhysicsGpuRenderPoses &program,
                       const std::shared_ptr<PhysicsGpuRenderJob> &job, ContactStreamIdentity identity)
{
    using namespace physics::inxphygpu;
    return [&domain, &program, job, identity](const InxPhyGpuFixedPublication &publication) {
        const auto row = publication.FindAcceptedWorld(identity);
        if (!row)
            throw std::logic_error("inxphygpu render export requires the matching scene state");
        const PhysicsGpuRenderBindings::Source source{
            publication.PoseSource(*row),
            [&](PhysicsActorHandle actor, uint32_t body) { return publication.FindPoseBody(*row, actor, body); }};
        job->output = domain.rendering.bindings.PrepareExport(job->snapshot, source, program);
        return InxPhyGpuSceneRuntime::PublicationWork{job->output->Packet(), [job] { job->output->Seal(); }};
    };
}
} // namespace

void PhysicsWorld::AcceptInxPhyGpuPublication(
    PhysicsDomain &domain, std::shared_ptr<const physics::inxphygpu::InxPhyGpuFixedPublication> publication,
    bool numericalChange, const std::shared_ptr<PhysicsGpuRenderJob> &job)
{
    if (numericalChange) {
        domain.nativeFixedPublication = std::move(publication);
        domain.queryGeneration.fetch_add(1, std::memory_order_release);
        domain.requestedMotion.clear();
    }
    if (job && job->output) {
        job->numericalGeneration = domain.queryGeneration.load(std::memory_order_acquire);
        domain.rendering.ready = job;
    }
}

void PhysicsDomain::BeginInxPhyGpuFixed(ContactStreamIdentity identity, PhysicsFixedInputs input)
{
    using namespace physics::inxphygpu;
    auto &physics = PhysicsWorld::Instance();
    physics.RequireOwnerThread();
    physics.RequireHealthyStep(*this);
    if (!nativeRuntime || retiring || nativeEdit || nativeFixedAdmission || gpuJointLayoutDirty ||
        identity.worldId != owner.id || identity.simulationEpoch != owner.simulation.simulationEpoch)
        throw std::logic_error("inxphygpu fixed admission requires its live scene layout and fixed identity");
    auto policy = physics.CaptureContactPolicy(*this);
    auto job = physics.CaptureGpuRenderJob(*this, true);
    InxPhyGpuSceneRuntime::PublicationConsumer publish;
    if (job)
        publish = PrepareInxPhyGpuRender(*this, physics.EnsureGpuRenderPoses(), job, identity);
    nativeRuntime->BeginFixed({identity}, gpuBodyCommands, false, policy.Requested(), CaptureGpuJointEdits(),
                              std::move(publish), {input}, {}, {{owner.id, participants}});
    nativeFixedAdmission.emplace(InxPhyGpuFixedAdmission{identity, std::move(policy), std::move(job)});
    gpuBodyCommands.clear();
    gpuTargetAuthors.reset();
    AcceptGpuJointEdits();
}

bool PhysicsDomain::PublishInxPhyGpuFixed()
{
    PhysicsWorld::Instance().RequireOwnerThread();
    const auto &admission = nativeFixedAdmission.value();
    auto completed = nativeRuntime->TakePublication();
    if (!completed)
        return false;
    const auto row = completed->FindAcceptedWorld(admission.identity);
    if (!row || completed->worlds[*row].status != 0)
        throw std::runtime_error("inxphygpu did not complete the admitted scene fixed step");
    auto publication = std::make_shared<physics::inxphygpu::InxPhyGpuFixedPublication>(std::move(*completed));
    publication->PublishContacts(*row, contacts, admission.contacts);
    PhysicsWorld::Instance().AcceptInxPhyGpuPublication(*this, std::move(publication), true, admission.renderJob);
    nativeFixedAdmission.reset();
    return true;
}

void PhysicsDomain::RetireInxPhyGpuFixed() noexcept
{
    if (nativeTopology)
        nativeTopology->Retire();
    if (nativeRuntime)
        nativeRuntime->Retire();
    nativeFixedAdmission.reset();
    nativeEdit.reset();
    nativeFixedPublication.reset();
    nativeTopologyActors.clear();
}

void PhysicsDomain::RequestInxPhyGpuAuthorPresentation() const
{
    if (UsesGpu() && owner.simulation.isPaused && !retiring) {
        const WorldExecutionScope scope(&owner);
        SceneManager::Instance().RequestPausedAuthorPresentation();
    }
}

void PhysicsDomain::MarkInxPhyGpuTopology() noexcept
{
    if (UsesGpu()) {
        nativeBodyLayoutDirty = true;
        RequestInxPhyGpuAuthorPresentation();
    }
}

void PhysicsWorld::RequireHealthyStep(PhysicsDomain &domain) const
{
    if (domain.fixedGpuPhase == PhysicsDomain::FixedGpuPhase::Failed)
        throw std::runtime_error("GPU physics step failed; stop the runtime before restarting");
}
void PhysicsWorld::FailStep(PhysicsDomain &domain) noexcept
{
    domain.RetireInxPhyGpuFixed();
    GpuPublicationReader::Cancel(domain.participantInitialization);
    domain.cpuContacts.CancelPending();
    domain.fixedGpuPhase = PhysicsDomain::FixedGpuPhase::Failed;
    domain.stepCompleted = false;
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
            {nullptr, [owner = std::move(owner)] { return owner->PrepareInitialize(); }});
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
    if (domain.stepPending && !domain.stepCompleted)
        return false;
    if (std::any_of(observers.begin(), observers.end(), [](const auto &owner) { return !owner->IsReady(); })) {
        (void)ServiceParticipantPreparation();
        return false;
    }
    if (observers.empty())
        return true;
    const auto &clock = domain.owner.simulation;
    auto epoch = clock.simulationEpoch, step = clock.publishedFixedStepIndex;
    bool fixed = false;
    if (domain.nativeFixedPublication) {
        const auto &publication = domain.nativeFixedPublication->worlds.front();
        step = publication.identity.fixedStepIndex;
        fixed = domain.nativeFixedPublication->IsFixed();
    }
    PhysicsGpuWork copies;
    for (auto it = observers.rbegin(); it != observers.rend(); ++it) {
        if (!(*it)->PublishResidentOutputs(epoch, step, fixed))
            continue;
        auto tail = std::move(copies);
        copies = {nullptr, [owner = *it, epoch, step, fixed, tail = std::move(tail)]() mutable {
                      return ContinuePhysicsGpuWork(owner->PrepareOutputs(epoch, step, fixed),
                                                    [tail = std::move(tail)]() mutable { return std::move(tail); });
                  }};
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
                return true;
            }
        } else {
            // A registered observation owns the earliest safe state before any
            // later numerical writer, including the next fixed integration.
            if ((domain.UsesGpu() && !ServiceParticipantPreparation()) || !ServiceParticipantOutputs())
                return false;
            RefreshBodySurfaces(domain);
            domain.stepEpoch = epoch;
            domain.stepIndex = index;
            domain.admittedDeltaTime = dt;
            domain.admittedGravity = JPH::Vec3(domain.gravity.x, domain.gravity.y, domain.gravity.z);
            const auto &config = EngineConfig::Get();
            if (config.physicsVelocitySteps < 0 || config.physicsPositionSteps < 0 || config.physicsCollisionSteps <= 0)
                throw std::invalid_argument("Physics iterations and substeps are invalid");
            if (!domain.UsesGpu())
                domain.admittedCpuSettings = JoltCpuScene::CaptureSettings(config);
            domain.admittedSubsteps = static_cast<uint32_t>(config.physicsCollisionSteps);
            // Own this fixed identity before waiting for an earlier visual
            // control. Scripts and accounting cannot be replayed meanwhile.
            domain.stepPending = true;
            domain.solverStarted = false;
            domain.fixedGpuPhase = PhysicsDomain::FixedGpuPhase::Solver;
        }
        if (!AdvanceFixedStep(domain, false))
            return false;
        domain.stepPending = domain.stepCompleted = false;
        domain.fixedGpuPhase = PhysicsDomain::FixedGpuPhase::Idle;
        return true;
    } catch (...) {
        FailStep(domain);
        throw;
    }
}
bool PhysicsWorld::AdvanceFixedStep(PhysicsDomain &domain, bool synchronous)
{
    RequireHealthyStep(domain);
    try {
        if (!domain.solverStarted && !(domain.solverStarted = TryBeginFixed(domain, synchronous)))
            return false;
        if (domain.UsesGpu()) {
            if (!AdvanceGpuJob(domain, synchronous) || !domain.PublishInxPhyGpuFixed())
                return false;
            AcceptStaticFixedMoves(domain);
            domain.stepCompleted = true;
        } else {
            FinishFixed(domain);
        }
        return true;
    } catch (...) {
        FailStep(domain);
        throw;
    }
}

void PhysicsWorld::RetryCapacityBodyCreations(PhysicsDomain &domain)
{
    const WorldExecutionScope ownerScope(&domain.owner);
    auto &world = domain.owner;
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
    return domain.nativeFixedAdmission.has_value() && !domain.nativeRuntime->HasPublication();
}

bool PhysicsWorld::HasDomainControl(PhysicsDomain &domain) const
{
    const auto &world = domain.owner;
    const auto &store = PhysicsECSStore::Instance();
    return domain.pendingRagdollHead || HasCpuControl(domain) || domain.nativeEdit || domain.nativeTopology ||
           domain.nativeBodyLayoutDirty || domain.gpuJointLayoutDirty || !domain.gpuBodyCommands.empty() ||
           !domain.gpuJointEdits.empty() || store.HasDirtyColliders(world) || store.HasPendingBodyCreations(world) ||
           store.HasPendingBroadphaseAdds(world) || store.HasPendingBroadphaseRemoves(world) ||
           store.HasPendingJointDependencies(world) || (PhysicMaterial::GetSurfaceRevision() != domain.surfaceRevision);
}
void PhysicsWorld::AdvanceInxPhyGpuSource(physics::inxphygpu::InxPhyGpuSceneRuntime &runtime, bool synchronous) const
{
    const auto availability = [this] { return Availability(*m_gpuHost); };
    if (synchronous) {
        m_gpuWork->DrainRequired(runtime.SourceId(), availability,
                                 [&](auto, auto timeout) { return runtime.WaitForProgress(timeout); });
    } else if (m_gpuWork->IsActive()) {
        (void)m_gpuWork->AdvanceDependency(runtime.SourceId(), availability);
    }
}

bool PhysicsWorld::AdvanceInxPhyGpuEdit(PhysicsDomain &domain, bool synchronous)
{
    using namespace physics::inxphygpu;
    if (!domain.nativeEdit)
        return true;
    auto &runtime = *domain.nativeRuntime;
    if (!runtime.HasPublication())
        AdvanceInxPhyGpuSource(runtime, synchronous);
    auto published = runtime.TakePublication();
    if (!published)
        return false;
    const auto &edit = *domain.nativeEdit;
    auto publication =
        edit.numericalChange ? std::make_shared<InxPhyGpuFixedPublication>(std::move(*published)) : nullptr;
    AcceptInxPhyGpuPublication(domain, std::move(publication), edit.numericalChange, edit.renderJob);
    domain.nativeEdit.reset();
    return true;
}

bool PhysicsWorld::ServiceInxPhyGpuEdits(PhysicsDomain &domain, bool synchronous)
{
    using namespace physics::inxphygpu;
    if (!AdvanceInxPhyGpuEdit(domain, synchronous))
        return false;
    const auto count = EngineConfig::Get().physicsCollisionSteps;
    if (count <= 0)
        throw std::invalid_argument("Physics substeps must be positive");
    const auto substeps = static_cast<uint32_t>(count);
    const PhysicsFixedInputs input{domain.owner.simulation.fixedTimeStep, domain.gravity};
    if (!EnsureStaticBoundaries(domain) || !PrepareInxPhyGpuTopology(domain, input, substeps, synchronous))
        return false;
    auto edits = domain.CaptureGpuJointEdits();
    const bool numericalChange = !domain.gpuBodyCommands.empty() || !edits.empty();
    auto job = CaptureGpuRenderJob(domain, numericalChange);
    if (!numericalChange && !job)
        return true;
    InxPhyGpuSceneRuntime::PublicationConsumer render;
    if (job)
        render = PrepareInxPhyGpuRender(domain, EnsureGpuRenderPoses(), job,
                                        domain.nativeFixedPublication->worlds.front().identity);
    domain.nativeRuntime->BeginEdits(domain.gpuBodyCommands, edits, std::move(render));
    domain.nativeEdit.emplace(PhysicsDomain::InxPhyGpuEdit{std::move(job), numericalChange});
    domain.gpuBodyCommands.clear();
    domain.gpuTargetAuthors.reset();
    domain.AcceptGpuJointEdits();
    return AdvanceInxPhyGpuEdit(domain, synchronous);
}

void PhysicsWorld::CompleteDomainStep(PhysicsDomain &domain)
{
    RequireOwnerThread();
    RequireHealthyStep(domain);
    if (domain.stepPending && !domain.stepCompleted && !AdvanceFixedStep(domain, true)) {
        FailStep(domain);
        throw std::runtime_error("Synchronous physics read could not complete its admitted step");
    }
}

bool PhysicsWorld::AdvanceGpuJob(PhysicsDomain &domain, bool synchronous) const
{
    auto &runtime = *domain.nativeRuntime;
    if (!runtime.HasPublication())
        AdvanceInxPhyGpuSource(runtime, synchronous);
    return runtime.HasPublication();
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
            if (!domain.cpuPoses.empty() || !domain.cpuInterpolated.empty())
                manager.PublishPhysicsStateToRenderer();
        } catch (...) {
            FailStep(domain);
            throw;
        }
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
    try {
        if (domain.stepPending && !domain.stepCompleted && !AdvanceFixedStep(domain, false))
            return false;
        if (!IsFixedPublicationHeld(domain)) {
            applyAuthors();
            if (domain.UsesGpu() && !ServiceInxPhyGpuEdits(domain, false))
                return false;
            SceneManager::Instance().PublishPhysicsStateToRenderer();
        }
        return true;
    } catch (...) {
        FailStep(domain);
        throw;
    }
}

void PhysicsWorld::CancelPendingStep() noexcept
{
    auto &domain = CurrentDomain();
    domain.RetireInxPhyGpuFixed();
    if (domain.UsesGpu()) {
        auto &rendering = domain.rendering;
        rendering.bindings.Clear();
        rendering.ready.reset();
        rendering.checkedKeys.reset();
    }
    GpuPublicationReader::Cancel(domain.participantInitialization);
    GpuPublicationReader::Cancel(domain.participantOutput);
    while (!domain.gpuJointAuthors.empty())
        domain.RetireGpuJoint(domain.gpuJointAuthors.begin()->first);
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
    if (auto pose = std::exchange(domain.presentationPoseReads, {}))
        GpuPublicationReader::Cancel(pose->request);
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

    FlushAuthoring(domain);
    const_cast<PhysicsWorld *>(this)->EnsureStaticBoundaries(domain);
    if (HasCpuControl(domain))
        ApplyCpuControl(domain);
}

const physics::inxphygpu::InxPhyGpuFixedPublication &
PhysicsWorld::ReadInxPhyGpuQueryPublication(PhysicsDomain &domain) const
{
    RequireOwnerThread();
    RequireHealthyStep(domain);
    auto &physics = *const_cast<PhysicsWorld *>(this);
    if (!IsFixedPublicationHeld(domain)) {
        physics.CompleteDomainStep(domain);
        if (!IsFixedPublicationHeld(domain)) {
            FlushAuthoring(domain);
            if (!physics.ServiceInxPhyGpuEdits(domain, true))
                throw std::runtime_error("Synchronous physics read is waiting for authored scene geometry");
        }
    }
    const auto &publication = *domain.nativeFixedPublication;
    const auto &identity = publication.worlds.front().identity;
    if (identity.worldId != domain.owner.id || identity.simulationEpoch != domain.owner.simulation.simulationEpoch ||
        (IsFixedPublicationHeld(domain) && identity.fixedStepIndex != domain.stepIndex))
        throw std::logic_error("inxphygpu query has no matching scene publication");
    return publication;
}

std::vector<Collider *> PhysicsWorld::QueryColliders(PhysicsDomain &domain, const JoltCpuQueries::Shape &shape,
                                                     uint32_t layers, bool triggers) const
{
    const auto request = QueryVolume(shape, layers, triggers);
    std::vector<Collider *> colliders;
    if (domain.UsesGpu()) {
        const auto &publication = ReadInxPhyGpuQueryPublication(domain);
        const auto query = domain.nativeRuntime->PrepareOverlap(publication, 0, request);
        if (!query.packet)
            return {};
        RunPreparedWork(domain, query.packet);
        FrameWorkOwner::Scope materialize(*m_gpuWork);
        colliders = query.ReadOverlaps();
    } else {
        SealSpatialQuery(domain);
        if (domain.cpuScene)
            return JoltCpuQueries(*domain.cpuScene).Overlap(shape, {layers, triggers});
        return {};
    }
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
    if (!domain.UsesGpu()) {
        SealSpatialQuery(domain);
        hit = {};
        return domain.cpuScene &&
               JoltCpuQueries(*domain.cpuScene).Cast(shape, direction, maximum, hit, {layers, triggers});
    }
    return RunGpuShapeCast(domain, volume, direction, maximum, hit);
}

bool PhysicsWorld::RunGpuShapeCast(PhysicsDomain &domain, const physics::inxphygpu::InxPhyGpuQueryVolume &volume,
                                   const glm::vec3 &direction, float maximum, RaycastHit &hit,
                                   const BoxCollider *swept) const
{
    using namespace physics::inxphygpu;
    hit = {};
    if (!volume.layers)
        return false;
    const glm::dvec3 delta(direction);
    const auto unit = glm::vec3(delta / std::hypot(delta.x, delta.y, delta.z));
    const ModelVector motion{unit.x, unit.y, unit.z, swept ? 1.0e-3f : 0.0f};
    const auto &publication = ReadInxPhyGpuQueryPublication(domain);
    InxPhyGpuSceneVolumeRead query;
    if (swept) {
        const InxPhyGpuSceneSweepSource source{ColliderBodyIdentity(*swept), swept->GetComponentID(),
                                               swept->GetLifetimeGeneration(), swept->GetGameObject()->GetID()};
        query = domain.nativeRuntime->PrepareShapeCast(publication, 0, volume, motion, maximum, &source,
                                                       *CaptureColliderPairs(domain));
    } else {
        query = domain.nativeRuntime->PrepareShapeCast(publication, 0, volume, motion, maximum, nullptr, {});
    }
    if (!query.packet)
        return false;
    RunPreparedWork(domain, query.packet);
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    hit = query.ReadCast();
    return hit.bodyId != UINT32_MAX;
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
    if (domain.UsesGpu()) {
        entry.stagedOn = Backend::Gpu;
        domain.pendingStatic.erase(id);
        domain.MarkInxPhyGpuTopology();
        return true;
    }
    auto &boundary = *entry.staticModel;
    EnsureCpuReady(domain);
    auto &prepared = *boundary.pending;
    boundary.ApplyPose(prepared);
    domain.cpuControl->StageActor(id, entry.actor, prepared);
    if (entry.staticResident)
        domain.cpuControl->AddBodies({id});
    entry.stagedOn = Backend::Cpu;
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
    std::optional<PreparedCpuActor> cpu;
    if (domain.UsesGpu() || boundary)
        RegisterDomain(domain);
    else
        EnsureCpuReady(domain);
    if (!domain.UsesGpu())
        cpu = PrepareJoltCpuActor(std::move(*authoring));

    const auto id = store.AllocateBodyIdentity();
    RegisterBodyIdentity(domain, id, handle, backend);
    auto &entry = m_bodyActors.at(id);
    try {
        if (boundary) {
            entry.staticResident = resident;
            entry.staticModel = std::make_shared<PhysicsStaticBoundaryModel>(*actor.owner->GetTransform());
            if (cpu) {
                cpu->publication.sharedGeometryRevision = entry.stagedShapeRevision;
                entry.staticModel->Hold(std::move(*cpu));
            }
        } else {
            entry.rigidbody = actor.rigidbody->GetECSHandle();
            if (cpu) {
                domain.cpuControl->StageActor(id, handle, std::move(*cpu));
                entry.stagedOn = Backend::Cpu;
            }
        }
        if (domain.UsesGpu()) {
            entry.nativeAuthor = std::make_shared<PhysicsActorAuthoring>(std::move(*authoring));
            entry.stagedOn = Backend::Gpu;
            domain.pendingStatic.erase(id);
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
    if (registration.HasGpuBody())
        domain.rendering.bindings.RetireBody(id);
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
    const bool stage = domain.UsesGpu() || !boundary || entry.stagedOn.has_value();
    const bool resident = actor.owner->IsActiveInHierarchy() && !actor.sceneSuspended &&
                          ((actor.rigidbody && actor.rigidbody->IsEnabled()) || actor.primaryCollider);
    const auto rigidbody = boundary ? PhysicsECSStore::RigidbodyHandle{} : actor.rigidbody->GetECSHandle();

    TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    const auto revision = entry.stagedShapeRevision + 1;
    auto authoring = PhysicsActorAuthoring::Capture(actor);
    if (!authoring)
        return false;
    std::optional<PreparedCpuActor> cpu;
    std::shared_ptr<const PhysicsActorAuthoring> native;
    if (domain.UsesGpu()) {
        native = std::make_shared<PhysicsActorAuthoring>(std::move(*authoring));
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
    if (native) {
        entry.nativeAuthor = std::move(native);
        entry.stagedOn = Backend::Gpu;
    } else if (!stage) {
        staticModel->Hold(std::move(*cpu));
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
    entry.backend = backend;
    entry.rigidbody = rigidbody;
    entry.staticModel = std::move(staticModel);
    entry.staticResident = resident;
    entry.stagedShapeRevision = revision;
    domain.MarkInxPhyGpuTopology();
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
    if (domain.UsesGpu())
        return registration->backend == Backend::Gpu;
    if (registration->backend == Backend::Cpu)
        return domain.cpuControl && domain.cpuControl->HasStagedRigidbody(id, body.GetECSHandle());
    return false;
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
                                   const std::function<void(PhysicsActorAuthoring &)> &edit)
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
    if (edit)
        edit(*authoring);
    if (domain.UsesGpu()) {
        entry.nativeAuthor = std::make_shared<PhysicsActorAuthoring>(std::move(*authoring));
    } else {
        auto prepared = PrepareJoltCpuActor(std::move(*authoring));
        prepared.publication.sharedGeometryRevision = revision;
        if (entry.staticModel)
            entry.staticModel->ApplyPose(prepared);
        if (entry.HasCpuBody())
            domain.cpuControl->StageActor(actor.bodyId, entry.actor, std::move(prepared));
        else
            entry.staticModel->Hold(std::move(prepared));
    }
    entry.stagedShapeRevision = revision;
    domain.MarkInxPhyGpuTopology();
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
    const auto p = position ? Vector(*position) : glm::vec4{};
    const auto q = rotation ? Rotation(*rotation) : glm::vec4{};
    RecordStaticPose(domain, id, position, rotation, cancelMove);
    if (entry.HasCpuBody())
        domain.cpuControl->Queue(id, JoltCpuControl::Pose{position ? std::make_optional(CpuPosition(p)) : std::nullopt,
                                                          rotation ? std::make_optional(CpuRotation(q)) : std::nullopt,
                                                          cancelMove});
    if (entry.HasGpuBody()) {
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
            RequireGpuCommandOwner(domain);
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
        if (entry.HasCpuBody())
            cpuBodies[&domain].push_back(id);
    }
    for (auto &[domain, ids] : cpuBodies)
        domain->cpuControl->AddBodies(std::move(ids));
    for (const auto id : bodies) {
        auto &entry = m_bodyActors.at(id);
        auto &domain = DomainForBody(id);
        domain.MarkInxPhyGpuTopology();
        if (entry.backend == Backend::StaticBoundary)
            entry.staticResident = true;
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
    domain.MarkInxPhyGpuTopology();
    if (entry->HasCpuBody() && domain.cpuControl)
        domain.cpuControl->Queue(id, JoltCpuControl::Remove{preserveVelocity});
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
    if (domain.UsesGpu()) {
        (void)ReadInxPhyGpuQueryPublication(domain);
        return domain.nativeRuntime->BodyRow(0, id).has_value();
    }
    if (entry.HasCpuBody()) {
        RunCpuControl(domain);
        return domain.cpuScene->IsBodyInBroadphase(id);
    }
    return false;
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
    StageActorModel(actor, actor, [=](PhysicsActorAuthoring &a) {
        if (a.body) {
            a.body->drag = linear;
            a.body->angularDrag = angular;
        }
    });
}
void PhysicsWorld::SetBodyGravityFactor(uint32_t id, float factor)
{
    if (!std::isfinite(factor))
        throw std::invalid_argument("Gravity factor must be finite");
    auto &actor = ActorForBody(id);
    StageActorModel(actor, actor, [=](PhysicsActorAuthoring &a) {
        if (a.body)
            a.body->gravityFactor = factor;
    });
}
void PhysicsWorld::SetBodyFriction(uint32_t id, float value)
{
    auto &actor = ActorForBody(id);
    StageActorModel(actor, actor, [=](PhysicsActorAuthoring &a) {
        for (auto &surface : a.surfaces)
            surface.friction = value;
    });
}
void PhysicsWorld::SetBodyRestitution(uint32_t id, float value)
{
    auto &actor = ActorForBody(id);
    StageActorModel(actor, actor, [=](PhysicsActorAuthoring &a) {
        for (auto &surface : a.surfaces)
            surface.restitution = value;
    });
}
void PhysicsWorld::SetBodyMaxAngularVelocity(uint32_t id, float value)
{
    auto &actor = ActorForBody(id);
    StageActorModel(actor, actor, [=](PhysicsActorAuthoring &a) {
        if (a.body)
            a.body->maxAngularVelocity = value;
    });
}
void PhysicsWorld::SetBodyMaxLinearVelocity(uint32_t id, float value)
{
    auto &actor = ActorForBody(id);
    StageActorModel(actor, actor, [=](PhysicsActorAuthoring &a) {
        if (a.body)
            a.body->maxLinearVelocity = value;
    });
}
void PhysicsWorld::SetBodyAllowedDOFs(uint32_t id, int allowed, float mass)
{
    if (allowed < 0 || (allowed & ~0x3f) != 0)
        throw std::invalid_argument("Allowed degrees of freedom contain unknown bits");
    if (!std::isfinite(mass) || mass <= 0)
        throw std::invalid_argument("Body mass must be positive and finite");
    auto &actor = ActorForBody(id);
    StageActorModel(actor, actor, [=](PhysicsActorAuthoring &a) {
        if (a.body)
            a.body->constraints = (~allowed & 0x3f) << 1;
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
    const auto origin = kind == 2 ? Vector(point) : glm::vec4{};
    if (backend == Backend::Cpu) {
        domain.cpuControl->Queue(id, JoltCpuControl::Force{static_cast<JoltCpuScene::ForceKind>(kind),
                                                           static_cast<ForceMode>(mode), CpuVector(force),
                                                           CpuPosition(origin)});
        return;
    }
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
            RequireGpuCommandOwner(domain);
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

namespace
{
PhysicsGpuWork PrepareStateBatchOutput(rhi::ComputeHost &host, const PhysicsGpuRenderPoses *program,
                                       PhysicsStateFields fields, const std::vector<PhysicsDeviceQueryOutput> &exports,
                                       PhysicsRigidbodyStateBatch &batch, PhysicsGpuMotionRead &hostMotion,
                                       PhysicsGpuMotionExport &motion, const PhysicsGpuBoxExport &boxes,
                                       bool deferExport)
{
    batch.boxCount = boxes.boxes.size();
    if (exports.empty()) {
        if (!boxes.boxes.empty())
            fields = static_cast<PhysicsStateFields>(static_cast<uint32_t>(fields) |
                                                     static_cast<uint32_t>(PhysicsStateFields::CenterOfMass) |
                                                     static_cast<uint32_t>(PhysicsStateFields::Rotation));
        if (!motion.bodies.empty() && HasPhysicsStateField(fields, PhysicsStateFields::All))
            hostMotion = PhysicsGpuMotionRead::Prepare(host, motion.poses, std::move(motion.bodies), fields);
        return hostMotion.packet;
    }
    if (deferExport)
        return {};
    ValidateStateExportOverlap(exports, batch.states.size(), batch.boxCount);
    std::vector<PhysicsDeviceQueryOutput> motionTargets, shapeTargets;
    for (const auto &binding : exports)
        (IsMotionExport(binding) ? motionTargets : shapeTargets).push_back(binding);
    auto motionWork = PrepareGpuStateExports(*program, motion, motionTargets);
    auto shapeWork = PrepareGpuStateExports(*program, boxes, shapeTargets);
    return ContinuePhysicsGpuWork(std::move(motionWork),
                                  [work = std::move(shapeWork)]() mutable { return std::move(work); });
}
} // namespace

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
    return GatherRigidbodyStateRows(bodies, fields, includeBoxes, triggers, exports, false);
}

PhysicsRigidbodyStateBatch PhysicsWorld::GatherRigidbodyStateRows(const std::vector<Rigidbody *> &bodies,
                                                                  PhysicsStateFields fields, bool includeBoxes,
                                                                  bool triggers,
                                                                  const std::vector<PhysicsDeviceQueryOutput> &exports,
                                                                  bool sealed) const
{
    RequireOwnerThread();
    if (bodies.empty())
        return {};
    for (const auto &binding : exports)
        if (binding.field > PhysicsDeviceQueryField::ShapeBounciness ||
            (!includeBoxes && binding.field >= PhysicsDeviceQueryField::ShapeBodyIndex))
            throw std::invalid_argument("Physics export requires its requested state family");
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
            const auto &publication = sealed ? *domain.nativeFixedPublication : ReadInxPhyGpuQueryPublication(domain);
            const auto &identity = publication.worlds.front().identity;
            const auto generation = domain.queryGeneration.load(std::memory_order_acquire);
            auto &rows = group.batch;
            rows.bodies = group.bodies;
            rows.sources = {{identity.worldId, identity.simulationEpoch, identity.fixedStepIndex, generation}};
            rows.sourceForBody.resize(group.bodies.size());
            std::vector<uint32_t> ids;
            ids.reserve(group.bodies.size());
            for (const auto *body : group.bodies)
                ids.push_back(body->GetBodyId());
            group.motion = publication.GatherMotion(0, ids);
            group.gpuBoxes =
                includeBoxes ? publication.GatherBoxes(0, ids, triggers) : PhysicsGpuBoxExport{group.motion.poses, {}};
            rows.states.reserve(group.motion.bodies.size());
            for (const auto &body : group.motion.bodies)
                rows.states.push_back({body.pose.id, body.shapeRevision});
            if (!exports.empty())
                const_cast<PhysicsWorld *>(this)->EnsureGpuRenderPoses();
            PhysicsGpuWork work;
            {
                FrameWorkOwner::Scope preparation(*m_gpuWork);
                work = PrepareStateBatchOutput(*m_gpuHost, m_gpuRenderProgram.get(), fields, exports, rows,
                                               group.hostMotion, group.motion, group.gpuBoxes, !direct);
            }
            RunPreparedWork(domain, std::move(work));
            MaterializeStateBatch(domain, rows, group.hostMotion, group.gpuBoxes, exports.empty(), fields);
            if (direct)
                return std::move(rows);
            batch.sources.push_back(rows.sources.front());
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
    std::vector<std::pair<PhysicsDomain *, PhysicsGpuWork>> plannedExports;
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
        auto work = ContinuePhysicsGpuWork(std::move(motionWork),
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
    const auto bodies = QueryRigidbodiesInBounds(minimum, maximum, layers, triggers);
    return GatherRigidbodyStateRows(bodies, fields, includeBoxes, triggers, exports, true);
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
    if (!m_initialized || !domain.UsesGpu())
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
    return ServicePresentationPoseReads(domain, bodies);
}

bool PhysicsWorld::ServicePresentationPoseReads(PhysicsDomain &domain, const std::vector<Rigidbody *> &bodies)
{
    if (bodies.empty())
        return true;
    if (domain.retiring || !m_gpuWork->IsActive() || !m_gpuWork->IsAccepting() || m_gpuWork->ServiceRemaining() <= 0 ||
        !m_gpuWork->ResourceRemaining(FrameWorkResource::Cpu))
        return false;
    constexpr auto fields = PresentationPoseFields;
    if (const auto job = domain.presentationPoseReads) {
        if (!m_gpuPublicationReader->Poll(*job->request))
            return false;
        domain.presentationPoseReads.reset();
        if (job->nativePublication != domain.nativeFixedPublication ||
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
    if (!domain.nativeFixedPublication)
        return false;
    auto job = std::make_shared<PhysicsDomain::PresentationPoseReads>();
    job->nativePublication = domain.nativeFixedPublication;
    job->generation = domain.queryGeneration.load(std::memory_order_acquire);
    FrameWorkOwner::Scope preparation(*m_gpuWork);
    std::vector<uint32_t> ids;
    ids.reserve(missing.size());
    for (const auto *body : missing)
        ids.push_back(body->GetBodyId());
    auto motion = job->nativePublication->GatherMotion(0, ids);
    job->read = PhysicsGpuMotionRead::Prepare(*m_gpuHost, motion.poses, std::move(motion.bodies), fields);
    job->request = EnsureGpuPublicationReader().Enqueue(job->read.packet);
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
    if (!domain.UsesGpu()) {
        (void)ReadBodyBackend(domain, id);
        RunCpuControl(domain);
        const auto state = domain.cpuScene->ReadMotionState(id, fields, sleeping);
        if (static_cast<uint32_t>(fields) & static_cast<uint32_t>(PhysicsStateFields::All))
            domain.requestedMotion[id] = {domain.queryGeneration.load(std::memory_order_acquire), state, fields};
        return state;
    }
    auto readFields = static_cast<uint32_t>(fields);
    if (readFields & static_cast<uint32_t>(PresentationPoseFields))
        readFields |= static_cast<uint32_t>(PresentationPoseFields);
    const auto requested = static_cast<PhysicsStateFields>(readFields);
    const auto &publication = ReadInxPhyGpuQueryPublication(domain);
    auto motion = publication.GatherMotion(0, {id});
    const auto generation = domain.queryGeneration.load(std::memory_order_acquire);
    const auto read = PhysicsGpuMotionRead::Prepare(*m_gpuHost, motion.poses, std::move(motion.bodies), requested,
                                                    sleeping != nullptr);
    if (read.packet)
        RunPreparedWork(domain, read.packet);
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
    if (domain.UsesGpu()) {
        const auto &state = ReadInxPhyGpuQueryPublication(domain);
        const auto d = state.GatherMotion(0, {id}).bodies.front().inverseInertia;
        if (glm::any(glm::lessThanEqual(d, glm::vec3(0))))
            throw std::logic_error("Rigidbody has no accepted positive principal inertia");
        return glm::vec3(1.0f) / d;
    }
    (void)ReadBodyBackend(domain, id);
    RunCpuControl(domain);
    const auto inertia = domain.cpuScene->GetAcceptedMass(id).inverse.Reciprocal();
    return {inertia.GetX(), inertia.GetY(), inertia.GetZ()};
}

glm::quat PhysicsWorld::GetBodyPrincipalInertiaRotation(uint32_t id) const
{
    RequireOwnerThread();
    auto &domain = DomainForBody(id);
    if (domain.UsesGpu())
        return ReadInxPhyGpuQueryPublication(domain).GatherMotion(0, {id}).bodies.front().inertiaFrame;
    (void)ReadBodyBackend(domain, id);
    RunCpuControl(domain);
    const auto q = domain.cpuScene->GetAcceptedMass(id).axes;
    return {q.GetW(), q.GetX(), q.GetY(), q.GetZ()};
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
    auto &domain = CurrentDomain();
    if (domain.UsesGpu())
        (void)ReadInxPhyGpuQueryPublication(domain);
    else
        SealSpatialQuery(domain);
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
    if (count) {
        auto &domain = CurrentDomain();
        if (!domain.UsesGpu())
            SealSpatialQuery(domain);
    }
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

    if (domain.UsesGpu()) {
        const auto &publication = ReadInxPhyGpuQueryPublication(domain);
        const auto rays = PackSceneRays(origins, directions, count, maximum, layers, triggers);
        const auto batch = domain.nativeRuntime->PrepareRaycast(publication, 0, rays, true);
        RunPreparedWork(domain, batch.query.packet);
        FrameWorkOwner::Scope materialize(*m_gpuWork);
        auto nativeHits = batch.ReadHostHits();
        for (size_t i = 0; i < count; ++i) {
            mask[i] = nativeHits[i].bodyId != UINT32_MAX;
            hits[i] = std::move(nativeHits[i]);
            if (profile && mask[i])
                ++profile->publishedHits;
        }
    } else if (domain.cpuScene) {
        const std::shared_lock queryRead(domain.cpuScene->QueryMutex());
        JoltCpuQueries(*domain.cpuScene)
            .RaycastBatch(origins, directions, count, maximum, hits, mask, {layers, triggers}, profile);
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
    const auto &publication = ReadInxPhyGpuQueryPublication(domain);
    if (device ? device->count != 0 : !host.empty()) {
        const auto batch = device ? domain.nativeRuntime->PrepareDeviceRaycast(publication, 0, *device, &outputs)
                                  : domain.nativeRuntime->PrepareRaycast(publication, 0, host, false, &outputs);
        RunPreparedWork(domain, batch.query.packet);
    }
    const auto &identity = publication.worlds.front().identity;
    return {identity.worldId, identity.simulationEpoch, identity.fixedStepIndex,
            domain.queryGeneration.load(std::memory_order_acquire)};
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
    const auto lastPhysicalStep = previous && previous == domain.nativeFixedPublication.get()
                                      ? previous->worlds.front().identity.fixedStepIndex
                                      : clock.publishedFixedStepIndex;
    const ContactStreamIdentity identity{world.id, clock.simulationEpoch, lastPhysicalStep};
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
    std::unordered_map<uint32_t, uint32_t> rowById;
    actors.reserve(ids.size());
    registration.bodies.reserve(ids.size());
    rowById.reserve(ids.size());
    for (const auto id : ids) {
        const auto &binding = m_bodyActors.at(id);
        const auto &actor = store.GetActor(binding.actor);
        if (!actor.bodyInBroadphase)
            continue;
        auto author = binding.nativeAuthor ? std::optional<PhysicsActorAuthoring>(*binding.nativeAuthor)
                                           : PhysicsActorAuthoring::Capture(actor);
        if (!author)
            return nullptr;
        if (binding.staticModel) {
            author->position = binding.staticModel->position;
            author->rotation = binding.staticModel->rotation;
        }
        const auto bodyTarget = author->body ? author->body->target : PhysicsTargetReference{};
        auto prepared = PrepareInxPhyGpuSceneActor(std::move(*author));
        rowById.emplace(id, static_cast<uint32_t>(actors.size()));
        registration.bodies.push_back(
            {id, binding.stagedShapeRevision, binding.actor, std::move(prepared.surfaces), bodyTarget});
        actors.push_back(std::move(prepared.actor));
    }

    std::vector<InxPhyGpuSceneJoint> joints;
    std::vector<uint64_t> jointIds;
    const auto bakeAnchor = [&](uint32_t body, const glm::vec3 &anchor) {
        const auto *transform = store.GetActor(m_bodyActors.at(body).actor).owner->GetTransform();
        return glm::conjugate(transform->GetWorldRotation()) * transform->TransformVector(anchor);
    };
    const auto append = [&](uint64_t id, const PhysicsSceneJointAuthor &joint) {
        const auto ownerRow = rowById.find(joint.ownerBody);
        const auto connectedRow = rowById.find(joint.connectedBody);
        if (ownerRow == rowById.end() || (joint.connectedBody != UINT32_MAX && connectedRow == rowById.end()))
            return;
        auto &model = joints.emplace_back(InxPhyGpuSceneJoint{
            ownerRow->second, joint.connectedBody == UINT32_MAX ? NoModelIndex : connectedRow->second,
            bakeAnchor(joint.ownerBody, joint.ownerLocalAnchor), joint.enableCollision, joint.settings});
        model.ownerLocalAnchor = joint.ownerLocalAnchor;
        if (accepted)
            model.reference = previous->FindJointReference(*accepted, id);
        if (auto *distance = std::get_if<InxPhyGpuSceneDistance>(&model.settings);
            distance && joint.connectedBody != UINT32_MAX)
            distance->connectedAnchor = bakeAnchor(joint.connectedBody, distance->localAnchor);
        jointIds.push_back(id);
    };
    if (domain.UsesGpu()) {
        joints.reserve(domain.gpuJointAuthors.size());
        jointIds.reserve(domain.gpuJointAuthors.size());
        for (const auto &[id, joint] : domain.gpuJointAuthors)
            append(id, joint);
    } else if (domain.cpuControl) {
        std::vector<uint64_t> ordered;
        for (const auto &[id, owner] : m_jointDomains)
            if (owner == &domain)
                ordered.push_back(id);
        std::sort(ordered.begin(), ordered.end());
        joints.reserve(ordered.size());
        jointIds.reserve(ordered.size());
        using Cpu = JoltCpuConstraints;
        for (const auto id : ordered) {
            const auto &joint = *domain.cpuControl->FindJoint(id);
            const auto &binding = joint.binding;
            auto settings = std::visit(
                [&](const auto &value) -> InxPhyGpuSceneJointSettings {
                    using Setting = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<Setting, Cpu::Hinge>)
                        return InxPhyGpuSceneHinge{value.axis,
                                                   value.useLimits,
                                                   value.minimum,
                                                   value.maximum,
                                                   {value.motorEnabled, value.targetAngle, value.stiffness,
                                                    value.damping, value.maxTorque, value.jointDamping,
                                                    value.frictionTorque, value.armature}};
                    else if constexpr (std::is_same_v<Setting, Cpu::Slider>)
                        return InxPhyGpuSceneSlider{value.axis, value.useLimits, value.minimum, value.maximum};
                    else if constexpr (std::is_same_v<Setting, Cpu::Distance>)
                        return InxPhyGpuSceneDistance{binding.connectedAnchor,
                                                      value.autoDistance,
                                                      value.minimum,
                                                      value.maximum,
                                                      value.stiffness,
                                                      value.damping,
                                                      binding.connectedLocalAnchor};
                    else if constexpr (std::is_same_v<Setting, Cpu::Fixed>)
                        return std::monostate{};
                    else
                        return value;
                },
                joint.settings);
            append(id, {binding.ownerBody, binding.connectedBody, binding.ownerLocalAnchor, joint.enableCollision,
                        std::move(settings)});
        }
    }

    setup->model = BuildInxPhyGpuSceneModel(actors, joints, *CaptureColliderPairs(domain, true));
    for (uint32_t index = 0; index < jointIds.size(); ++index) {
        const auto &joint = setup->model.jointBindings[index];
        if (joint.row == NoModelIndex)
            continue;
        const auto id = jointIds[index];
        auto &registered = registration.joints.emplace_back(InxPhyGpuRegisteredJoint{id, index});
        if (domain.UsesGpu() && joint.poseTargetOffset != NoModelIndex)
            registered.authoredPose.emplace(InxPhyGpuJointPoseTarget{domain.gpuJointAuthors.at(id).poseTarget});
    }
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
    return domain.contacts.Dispatch(domain.owner);
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
void PhysicsWorld::InvalidateColliderPairSnapshot(PhysicsDomain &domain) const
{
    domain.colliderPairSnapshot.reset();
    domain.MarkInxPhyGpuTopology();
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

void PhysicsWorld::RequireGpuCommandOwner(PhysicsDomain &domain) const
{
    RequireHealthyStep(domain);
    if (domain.retiring)
        throw std::logic_error("Cannot submit a physics command to a retiring world");
}

PhysicsTargetSource PhysicsWorld::QueueGpuBodyCommand(PhysicsDomain &domain, const PhysicsBodyCommand &command,
                                                      const PhysicsTargetAuthor *author)
{
    RequireGpuCommandOwner(domain);
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
    domain.gpuBodyCommands.push_back({domain.owner.id, command, source});
    domain.RequestInxPhyGpuAuthorPresentation();
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
    if (domain.UsesGpu()) {
        hit = {};
        const auto &publication = ReadInxPhyGpuQueryPublication(domain);
        const float coordinates[]{origin.x, origin.y, origin.z}, axes[]{direction.x, direction.y, direction.z};
        const auto rays = PackSceneRays(coordinates, axes, 1, maximum, UINT32_MAX, true);
        const auto batch = domain.nativeRuntime->PrepareColliderRaycast(publication, 0, ColliderBodyIdentity(collider),
                                                                        collider.GetComponentID(),
                                                                        collider.GetLifetimeGeneration(), rays.front());
        if (!batch.query.packet)
            return false;
        RunPreparedWork(domain, batch.query.packet);
        FrameWorkOwner::Scope materialize(*m_gpuWork);
        hit = batch.ReadHostHits().front();
        return hit.bodyId != UINT32_MAX;
    }
    RunCpuControl(domain);
    const auto id = ColliderBodyIdentity(collider);
    hit = {};
    if (!FindBodyRegistration(id))
        return false;
    (void)ReadBodyBackend(domain, id);
    RunCpuControl(domain);
    const JoltCpuQueries::ColliderIdentity target{collider.GetComponentID(), collider.GetLifetimeGeneration()};
    return JoltCpuQueries(*domain.cpuScene).Raycast(origin, direction, maximum, hit, {UINT32_MAX, true, target});
}

glm::vec3 PhysicsWorld::ClosestPointOnCollider(const Collider &collider, const glm::vec3 &point) const
{
    auto &domain = DomainForComponent(collider);
    (void)Vector(point);
    if (!collider.IsEnabled() || !collider.GetGameObject() || !collider.GetGameObject()->IsActiveInHierarchy())
        return point;
    using namespace physics::inxphygpu;
    const auto materialize = [&](const std::shared_ptr<rhi::PreparedComputeBatch> &result) {
        if (!result)
            throw std::runtime_error("Active Collider has no accepted geometry for a closest-point query");
        FrameWorkOwner::Scope scope(*m_gpuWork);
        ModelVector record;
        result->Readback(0)->ReadInto(record.data(), sizeof(record));
        if (record[3] < 0.0f)
            throw std::runtime_error("Active Collider has no accepted geometry for a closest-point query");
        return glm::vec3(record[0], record[1], record[2]);
    };
    const auto *mesh = dynamic_cast<const MeshCollider *>(&collider);
    if (domain.UsesGpu()) {
        const auto &publication = ReadInxPhyGpuQueryPublication(domain);
        const auto result = domain.nativeRuntime->PrepareClosestPoint(
            publication, 0, ColliderBodyIdentity(collider), collider.GetComponentID(), collider.GetLifetimeGeneration(),
            {point.x, point.y, point.z, 0.0f}, mesh && mesh->IsPublishedMeshClosed());
        if (result)
            RunPreparedWork(domain, result);
        return materialize(result);
    }
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
    throw std::runtime_error("Active Collider has no accepted geometry for a closest-point query");
}

std::vector<RaycastHit> PhysicsWorld::RaycastAll(const glm::vec3 &origin, const glm::vec3 &direction, float maximum,
                                                 uint32_t layers, bool triggers) const
{
    auto &domain = CurrentDomain();
    ValidateRayInput(origin, direction, maximum);
    constexpr uint32_t capacity = 32;
    const auto order = [](const RaycastHit &a, const RaycastHit &b) {
        return std::tie(a.distance, a.bodyId, a.subShapeId) < std::tie(b.distance, b.bodyId, b.subShapeId);
    };
    if (domain.UsesGpu()) {
        const auto &publication = ReadInxPhyGpuQueryPublication(domain);
        const float coordinates[]{origin.x, origin.y, origin.z}, axes[]{direction.x, direction.y, direction.z};
        const auto rays = PackSceneRays(coordinates, axes, 1, maximum, layers, triggers);
        uint32_t count = 0;
        const auto read = [&](uint32_t limit) {
            const auto batch = domain.nativeRuntime->PrepareRaycast(publication, 0, rays, true, nullptr, limit);
            RunPreparedWork(domain, batch.query.packet);
            FrameWorkOwner::Scope materialize(*m_gpuWork);
            return batch.ReadHostHits(&count);
        };
        auto hits = read(capacity);
        if (count > capacity)
            hits = read(count);
        std::sort(hits.begin(), hits.end(), order);
        return hits;
    }
    SealSpatialQuery(domain);
    if (domain.cpuScene)
        return JoltCpuQueries(*domain.cpuScene).RaycastAll(origin, direction, maximum, {layers, triggers});
    return {};
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

std::vector<Rigidbody *> PhysicsWorld::QueryRigidbodiesInBounds(const glm::vec3 &minimum, const glm::vec3 &maximum,
                                                                uint32_t layers, bool triggers) const
{
    auto &domain = CurrentDomain();
    (void)Vector(minimum);
    (void)Vector(maximum);
    if (glm::any(glm::greaterThan(minimum, maximum)))
        throw std::invalid_argument("Physics query bounds require minimum <= maximum");
    if (domain.UsesGpu()) {
        const auto &publication = ReadInxPhyGpuQueryPublication(domain);
        const auto query =
            domain.nativeRuntime->PrepareBodyBounds(publication, 0, {minimum.x, minimum.y, minimum.z, 0},
                                                    {maximum.x, maximum.y, maximum.z, 0}, layers, triggers);
        if (!query.query.packet)
            return {};
        RunPreparedWork(domain, query.query.packet);
        FrameWorkOwner::Scope materialize(*m_gpuWork);
        auto &store = PhysicsECSStore::Instance();
        std::vector<Rigidbody *> bodies;
        for (const auto id : query.ReadHostBodyIds())
            if (const auto *entry = FindBodyRegistration(id))
                if (auto *body = store.GetActor(entry->actor).rigidbody)
                    bodies.push_back(body);
        return bodies;
    }
    SealSpatialQuery(domain);
    if (!domain.cpuScene)
        return {};
    auto bodies = JoltCpuQueries(*domain.cpuScene).QueryRigidbodiesInBounds(minimum, maximum, {layers, triggers});
    std::sort(bodies.begin(), bodies.end(),
              [](const auto *a, const auto *b) { return a->GetComponentID() < b->GetComponentID(); });
    return bodies;
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
    if (!domain.UsesGpu()) {
        SealSpatialQuery(domain);
        return domain.cpuScene && JoltCpuQueries(*domain.cpuScene).SweepBox(box, position, rotation, displacement, hit);
    }
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

std::vector<PhysicsShapeState> PhysicsWorld::GetBodyShapeStates(uint32_t id, bool triggers) const
{
    RequireOwnerThread();
    auto &domain = DomainForBody(id);
    if (ReadBodyBackend(domain, id) == Backend::Cpu) {
        RunCpuControl(domain);
        return JoltCpuQueries(*domain.cpuScene).GatherBodyShapes(id, triggers);
    }

    const auto &publication = ReadInxPhyGpuQueryPublication(domain);
    const auto work = domain.nativeRuntime->PrepareBodyShapes(publication, 0, id, triggers);
    if (work.packet)
        RunPreparedWork(domain, work.packet);
    FrameWorkOwner::Scope materialize(*m_gpuWork);
    return work.ReadHostShapes();
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
    if (!domain.UsesGpu())
        throw std::invalid_argument("Particle collision requires a GPU physics scene");
    using namespace physics::inxphygpu;
    (void)ReadInxPhyGpuQueryPublication(domain);
    const auto publication = domain.nativeFixedPublication;
    std::shared_ptr<particle::GpuParticleCollisionDevicePayload> result;
    InxPhyGpuSceneRuntime::PublicationWork work;
    {
        FrameWorkOwner::Scope preparation(*m_gpuWork);
        work = domain.nativeRuntime->PrepareParticleCollision(*publication, 0, &scene, previous, revision,
                                                              [&](auto payload) { result = std::move(payload); });
    }
    for (;;) {
        RunPreparedWork(domain, std::move(work.packet));
        FrameWorkOwner::Scope materialize(*m_gpuWork);
        if (work.seal)
            work.seal();
        if (!work.continueAfterReceipt)
            break;
        work = work.continueAfterReceipt(*publication);
    }
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
void PhysicsWorld::RunPreparedWork(PhysicsDomain &domain, PhysicsGpuWork prepared) const
{
    const WorldExecutionScope ownerScope(&domain.owner);
    const_cast<PhysicsWorld *>(this)->EnsureGpuPublicationReader().Run(std::move(prepared));
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
    domain.MarkInxPhyGpuTopology();
}
void PhysicsWorld::UnregisterBodyIdentity(PhysicsDomain &domain, uint32_t bodyId) noexcept
{
    m_bodyActors.erase(bodyId);
    domain.bodyIds.erase(bodyId);
    domain.pendingStatic.erase(bodyId);
    domain.pendingStaticMoves.erase(bodyId);
    if (domain.UsesGpu()) {
        auto &commands = domain.gpuBodyCommands;
        commands.erase(std::remove_if(commands.begin(), commands.end(),
                                      [bodyId](const auto &entry) { return entry.command.bodyId == bodyId; }),
                       commands.end());
        domain.MarkInxPhyGpuTopology();
    }
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

    if (domain.UsesGpu())
        return Backend::Gpu;
    if (domain.stepPending && !domain.stepCompleted)
        const_cast<PhysicsWorld *>(this)->CompleteDomainStep(domain);
    if (IsFixedPublicationHeld(domain)) {
        if (!domain.cpuScene || !domain.cpuScene->HasBody(bodyId))
            throw std::logic_error("Physics body has no accepted representation in this Fixed publication");
    } else if (registration->backend == Backend::StaticBoundary) {
        const_cast<PhysicsWorld *>(this)->EnsureStaticBoundary(domain, bodyId);
    }
    return Backend::Cpu;
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
    const auto &native = world.physics->nativeRuntime;
    const auto &topology = world.physics->nativeTopology;
    return (!native || native->CanRetire()) && (!topology || topology->CanRetire());
}
void PhysicsWorld::RetireDomain(PhysicsDomain &domain) noexcept
{
    RetireRagdolls(domain);
    m_domains.erase(&domain);
}
} // namespace infernux
