// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "WarpSceneRuntime.h"
#include "WarpContactPublication.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace infernux::physics::warp
{

struct WarpSceneLayout
{
    struct World
    {
        std::vector<WarpRegisteredBody> bodies; // Device row, including world row zero.
        std::unordered_map<uint32_t, uint32_t> rows;
        std::unordered_map<uint64_t, WarpSceneJointBinding> joints;

        [[nodiscard]] std::optional<uint32_t> BodyRow(uint32_t bodyId) const
        {
            const auto found = rows.find(bodyId);
            return found == rows.end() ? std::nullopt : std::make_optional(found->second);
        }
    };
    struct Geometry
    {
        uint32_t body, surface, subshape;
    };
    std::vector<World> worlds;
    std::unordered_map<uint64_t, uint32_t> worldRows;
    std::vector<Geometry> geometries;

    [[nodiscard]] uint32_t WorldRow(uint64_t id) const
    {
        const auto found = worldRows.find(id);
        if (found == worldRows.end())
            throw std::invalid_argument("Warp scene command requires an enrolled world");
        return found->second;
    }
};

namespace
{
WarpHostBody DecodeHostBody(const WarpBodyState &raw, uint32_t bodyId, uint64_t shapeRevision)
{
    const auto vector = [](const ModelVector &v) { return glm::vec3(v[0], v[1], v[2]); };
    const auto &q = raw.pose.rotation;
    return {{bodyId, vector(raw.pose.position), glm::quat(q[0], q[1], q[2], q[3])},
            shapeRevision,
            vector(raw.centerOfMass),
            vector(raw.linearVelocity),
            vector(raw.angularVelocity)};
}
} // namespace

PhysicsGpuPoseSource WarpFixedPublication::PoseSource(uint32_t world) const
{
    if (worlds.at(world).status != 0)
        throw std::logic_error("Cannot publish the pose of a rejected Warp fixed step");
    return {bodyStates, uint64_t{world} * bodiesPerWorld, bodiesPerWorld, PhysicsGpuPoseSource::Layout::OriginWXYZ};
}

std::optional<PhysicsGpuBodyPoseBinding> WarpFixedPublication::FindPoseBody(uint32_t world, PhysicsActorHandle actor,
                                                                            uint32_t bodyId) const
{
    if (worlds.at(world).status != 0)
        return std::nullopt;
    const auto &layout = m_layout->worlds[world];
    const auto row = layout.BodyRow(bodyId);
    if (!row || layout.bodies[*row].actor != actor)
        return std::nullopt;
    return PhysicsGpuBodyPoseBinding{bodyId, *row};
}

void WarpFixedPublication::PublishContacts(uint32_t world, PhysicsContactPublication &destination,
                                           const PhysicsContactPublication::Policy &policy)
{
    const auto &fixed = worlds.at(world);
    if (!m_integrated || fixed.status != 0)
        throw std::logic_error("Contact publication requires an accepted Warp fixed step");
    if (!policy.Requested()) {
        destination.PublishEmpty(fixed.identity);
        return;
    }
    if (contacts.size() != size_t{substeps} * worlds.size())
        throw std::logic_error("Accepted Warp step does not carry the requested contact reports");
    const auto &layout = m_layout->worlds[world];
    const std::function<PhysicsContactPublication::Endpoint(uint32_t)> resolve = [&](uint32_t geometry) {
        const auto &shape = m_layout->geometries.at(geometry);
        const auto &body = layout.bodies[shape.body];
        const auto &surface = body.surfaces.at(shape.surface);
        return PhysicsContactPublication::Endpoint{body.bodyId,        shape.subshape, surface.colliderId,
                                                   surface.generation, surface.target, body.bodyTarget};
    };
    destination.Publish(fixed.identity, policy, substeps, [&](uint32_t substep) {
        auto &reports = contacts[size_t{substep} * worlds.size() + world];
        return DecodeWarpContactFrame(std::move(reports), resolve, policy);
    });
}

std::vector<WarpHostBody> WarpSceneBodyRead::ReadHostBodies() const
{
    std::vector<WarpBodyState> records(m_uniqueRows);
    size_t offset = 0;
    for (size_t index = 0; index < m_rangeRows.size(); ++index) {
        const auto count = m_rangeRows[index];
        packet->Readback(index)->ReadInto(records.data() + offset, uint64_t{count} * sizeof(WarpBodyState));
        offset += count;
    }
    std::vector<WarpHostBody> bodies;
    bodies.reserve(m_bodies.size());
    for (const auto &body : m_bodies)
        bodies.push_back(DecodeHostBody(records[body.packedRow], body.id, body.shapeRevision));
    return bodies;
}

std::vector<RaycastHit> WarpSceneRayBatch::ReadHostHits() const
{
    if (!m_layout)
        throw std::logic_error("Warp scene ray batch has no host results");
    std::vector<WarpRayHit> records(query.count);
    query.packet->Readback(0)->ReadInto(records.data(), records.size() * sizeof(WarpRayHit));
    const auto &world = m_layout->worlds[m_world];
    std::vector<RaycastHit> hits(records.size());
    for (size_t index = 0; index < records.size(); ++index) {
        const auto &record = records[index];
        const auto &[geometry, body, subShape, triangle] = record.identity;
        if (geometry == NoModelIndex)
            continue;
        const auto &binding = world.bodies[body];
        const auto &line = m_lines[index];
        auto &hit = hits[index];
        hit.distance = record.normalDistance[3];
        hit.point = line.origin + line.direction * hit.distance;
        hit.normal = {record.normalDistance[0], record.normalDistance[1], record.normalDistance[2]};
        hit.bodyId = binding.bodyId;
        hit.subShapeId = subShape;
        hit.triangleIndex = triangle;
        if (!binding.surfaces.empty()) {
            const auto &surface = binding.surfaces[m_layout->geometries[geometry].surface];
            hit.target = surface.target;
            hit.gameObjectId = surface.gameObjectId;
            hit.colliderId = surface.colliderId;
            hit.gameObject = hit.target.GetGameObject();
            hit.collider = hit.target.GetCollider();
        }
    }
    return hits;
}

WarpSceneRuntime::WarpSceneRuntime(FrameWorkOwner &work, rhi::ComputeHost &host, ShaderCodeTarget target,
                                   const WarpSceneModel &scene, std::vector<WarpRegisteredWorld> worlds, float timestep,
                                   const ModelVector &gravity, WarpCollisionCapacity collisionCapacity,
                                   const WarpContactSolver &solver, uint32_t commandCapacity, bool contactReports,
                                   uint32_t substeps, WarpRayBatchLayout rayLayout)
    : m_work(work), m_host(host), m_bodyCount(scene.model.dimensions.state[2]),
      m_contactReportsConfigured(contactReports)
{
    if (worlds.empty() || commandCapacity == 0)
        throw std::invalid_argument("Warp scene requires worlds and a body command capacity");
    auto layout = std::make_shared<WarpSceneLayout>();
    layout->worlds.reserve(worlds.size());
    layout->worldRows.reserve(worlds.size());
    m_acceptedWorlds.reserve(worlds.size());
    layout->geometries.reserve(scene.geometries.size());
    for (size_t row = 0; row < scene.geometries.size(); ++row) {
        const auto &geometry = scene.geometries[row];
        layout->geometries.push_back({scene.model.geometries[row].indices[0], geometry.surface, geometry.subShape});
    }
    for (auto &source : worlds) {
        if (source.bodies.size() != scene.bodies.size() ||
            !layout->worldRows.emplace(source.worldId, static_cast<uint32_t>(layout->worlds.size())).second)
            throw std::invalid_argument("Warp scene world does not match its shared model");
        WarpSceneLayout::World world{std::vector<WarpRegisteredBody>(m_bodyCount), {}, {}};
        world.rows.reserve(source.bodies.size());
        for (size_t index = 0; index < source.bodies.size(); ++index) {
            auto &binding = source.bodies[index];
            const auto &authored = scene.bodies[index];
            const auto row = authored.bodyRow;
            if (binding.bodyId == UINT32_MAX || !world.rows.emplace(binding.bodyId, row).second)
                throw std::invalid_argument("Duplicate or invalid Warp scene body identity");
            if (!binding.surfaces.empty() && binding.surfaces.size() != authored.surfaceCount)
                throw std::invalid_argument("Warp scene collider identities must match the shared model");
            world.bodies[row] = std::move(binding);
        }
        world.joints.reserve(source.joints.size());
        for (const auto &binding : source.joints) {
            const auto &joint = scene.jointBindings.at(binding.authorRow);
            if (joint.row == NoModelIndex || !world.joints.emplace(binding.jointId, joint).second)
                throw std::invalid_argument("Duplicate or invalid Warp scene joint identity");
        }
        m_acceptedWorlds.push_back({{source.worldId, source.simulationEpoch, 0}, 0.0f, 0, 0});
        layout->worlds.push_back(std::move(world));
    }
    m_layout = std::move(layout);
    auto prepared = WarpStep::Build(m_host, target, scene.model, static_cast<uint32_t>(m_layout->worlds.size()),
                                    timestep, gravity, &scene.collision, collisionCapacity, &solver, rayLayout,
                                    commandCapacity, contactReports, substeps);
    if (!prepared.step)
        throw std::runtime_error("Could not allocate the Warp scene world layout");
    m_step = std::move(prepared.step);
    m_packet = std::move(prepared.initialize);
    const auto observe = [this] { Observe(); };
    m_source = m_work.AddSource({observe, observe, [this] { return Describe(); },
                                 [this](bool gpu, uint64_t token) { return Advance(gpu, token); }});
}

WarpSceneRuntime::~WarpSceneRuntime()
{
    Retire();
    m_work.RemoveSource(m_source);
}

std::optional<uint32_t> WarpSceneRuntime::BodyRow(uint32_t world, uint32_t engineBody) const
{
    return m_layout->worlds.at(world).BodyRow(engineBody);
}

std::optional<WarpFixedPublication> WarpSceneRuntime::TakeInitial()
{
    return std::exchange(m_initialPublication, std::nullopt);
}

void WarpSceneRuntime::BeginFixed(std::vector<ContactStreamIdentity> identities,
                                  const std::vector<WarpSceneBodyCommand> &commands, bool host, bool captureContacts,
                                  const std::vector<WarpSceneJointUpdate> &jointUpdates, PublicationConsumer consumer)
{
    if (captureContacts && !m_contactReportsConfigured)
        throw std::logic_error("Warp scene was prepared without contact report storage");
    const auto &worlds = m_layout->worlds;
    if (identities.size() != worlds.size())
        throw std::invalid_argument("Warp fixed step requires an identity per world");
    for (size_t world = 0; world < identities.size(); ++world) {
        const auto &next = identities[world];
        const auto &previous = m_acceptedWorlds[world].identity;
        if (next.worldId != previous.worldId || next.simulationEpoch != previous.simulationEpoch ||
            next.fixedStepIndex <= previous.fixedStepIndex)
            throw std::invalid_argument("Warp fixed step must advance the enrolled world and epoch");
    }
    std::vector<WarpBodyCommand> actions;
    actions.reserve(commands.size());
    for (const auto &entry : commands)
        EncodeBodyCommand(m_layout->WorldRow(entry.worldId), entry.command, actions);
    std::vector<WarpJointParameterUpdate> updates;
    updates.reserve(jointUpdates.size());
    for (const auto &update : jointUpdates) {
        const auto world = m_layout->WorldRow(update.worldId);
        const auto &joints = worlds[world].joints;
        const auto binding = joints.find(update.jointId);
        if (binding == joints.end())
            throw std::invalid_argument("Warp joint update requires an enrolled joint in the selected world");
        const auto &joint = binding->second;
        if (const auto *limits = std::get_if<WarpJointLimits>(&update.value)) {
            const auto minimum = limits->minimum, maximum = limits->maximum;
            if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum > maximum ||
                joint.kind == WarpSceneJointKind::Fixed ||
                (joint.kind == WarpSceneJointKind::Slider && (minimum > 0.0f || maximum < 0.0f)) ||
                (joint.kind == WarpSceneJointKind::Distance &&
                 (joint.autoDistance || !limits->enabled || minimum < 0.0f)))
                throw std::invalid_argument(
                    "Warp joint limits must match the enrolled constraint and its authored mode");
            updates.push_back({world, joint.row, limits->Pack()});
        } else if (const auto *drive = std::get_if<WarpHingeDrive>(&update.value)) {
            if (joint.kind != WarpSceneJointKind::Hinge)
                throw std::invalid_argument("Warp hinge drive requires an enrolled hinge");
            updates.push_back({world, joint.row, drive->Pack()});
        } else {
            const auto &spring = std::get<WarpDistanceSpring>(update.value);
            if (joint.kind != WarpSceneJointKind::Distance || !std::isfinite(spring.stiffness) ||
                !std::isfinite(spring.damping) || spring.stiffness < 0.0f || spring.damping < 0.0f)
                throw std::invalid_argument(
                    "Warp distance spring requires finite nonnegative coefficients on a distance joint");
            updates.push_back({world, joint.row, spring});
        }
    }
    BeginOperation({std::move(identities), std::move(actions), std::move(updates), host, captureContacts,
                    std::move(consumer), true});
}

void WarpSceneRuntime::BeginEdits(uint64_t worldId, const std::vector<PhysicsBodyCommand> &commands,
                                  PublicationConsumer consumer)
{
    const auto world = m_layout->WorldRow(worldId);
    std::vector<WarpBodyCommand> edits;
    edits.reserve(commands.size());
    for (const auto &command : commands)
        EncodeBodyCommand(world, command, edits);
    if (edits.empty())
        throw std::invalid_argument("Warp scene edit requires at least one body command");
    BeginOperation({{}, std::move(edits), {}, false, false, std::move(consumer), false});
}

void WarpSceneRuntime::EncodeBodyCommand(uint32_t world, const PhysicsBodyCommand &command,
                                         std::vector<WarpBodyCommand> &out) const
{
    const auto row = m_layout->worlds[world].BodyRow(command.bodyId);
    if (!row)
        throw std::invalid_argument("Warp scene command requires an accepted body in the selected world");
    const auto vector = [](const glm::vec3 &v) -> ModelVector { return {v.x, v.y, v.z, 0.0f}; };
    const auto rotation = [](const glm::quat &q) -> ModelVector { return {q.w, q.x, q.y, q.z}; };
    using Kind = WarpBodyCommandKind;
    const auto append = [&](Kind kind, ModelVector value = {}, ForceMode mode = {},
                            ModelVector point = {}) -> WarpBodyCommand & {
        return out.emplace_back(WarpBodyCommand{world, *row, kind, mode, value, point});
    };
    if (const auto *pose = std::get_if<PhysicsBodyCommand::Pose>(&command.operation)) {
        const auto flags = pose->cancelDrive ? WarpPoseCommandFlags::None : WarpPoseCommandFlags::PreserveDrive;
        if (pose->position)
            append(Kind::SetPosition, vector(*pose->position)).poseFlags = flags;
        if (pose->rotation)
            append(Kind::SetRotation, rotation(*pose->rotation)).poseFlags = flags;
    } else if (const auto *velocity = std::get_if<PhysicsBodyCommand::Velocity>(&command.operation)) {
        if (velocity->linear)
            append(Kind::SetLinearVelocity, vector(*velocity->linear));
        if (velocity->angular)
            append(Kind::SetAngularVelocity, vector(*velocity->angular));
    } else if (const auto *force = std::get_if<PhysicsBodyCommand::Force>(&command.operation)) {
        constexpr std::array<Kind, 3> kinds{Kind::Force, Kind::Torque, Kind::ForceAtPosition};
        append(kinds[static_cast<size_t>(force->kind)], vector(force->value), force->mode, vector(force->point));
    } else {
        const auto &reset = std::get<PhysicsBodyCommand::ResetMotion>(command.operation);
        append(Kind::ResetMotion);
        append(Kind::SetPosition, vector(reset.position));
        append(Kind::SetRotation, rotation(reset.rotation));
        append(Kind::SetLinearVelocity, vector(reset.linear));
        append(Kind::SetAngularVelocity, vector(reset.angular));
    }
}

void WarpSceneRuntime::BeginOperation(Operation operation)
{
    if (m_retiring || m_operation || m_publication)
        throw std::logic_error("Warp scene still owns a physics operation");
    m_initialPublication.reset();
    m_operation.emplace(std::move(operation));
    ++m_generation;
    m_work.Wake(m_source);
}

std::optional<WarpFixedPublication> WarpSceneRuntime::TakePublication()
{
    return std::exchange(m_publication, std::nullopt);
}

std::optional<WarpHostBody> WarpSceneRuntime::ReadHostBody(const WarpFixedPublication &publication, uint32_t world,
                                                           uint32_t engineBody) const
{
    if (publication.m_layout != m_layout || publication.hostBodies.empty())
        return std::nullopt;
    const auto &layout = m_layout->worlds.at(world);
    const auto row = layout.BodyRow(engineBody);
    if (!row)
        return std::nullopt;
    const auto &raw = publication.hostBodies.at(uint64_t{world} * m_bodyCount + *row);
    return DecodeHostBody(raw, engineBody, layout.bodies[*row].shapeRevision);
}

WarpSceneBodyRead WarpSceneRuntime::PrepareBodyRead(const WarpFixedPublication &publication, uint32_t world,
                                                    const std::vector<uint32_t> &engineBodies) const
{
    if (publication.m_layout != m_layout)
        throw std::invalid_argument("Warp body read does not match its scene publication");
    const auto source = publication.PoseSource(world);
    const auto &layout = m_layout->worlds[world];
    WarpSceneBodyRead result;
    result.identity = publication.worlds[world].identity;
    result.m_bodies.resize(engineBodies.size());
    std::vector<std::pair<uint32_t, size_t>> rows;
    rows.reserve(engineBodies.size());
    for (size_t index = 0; index < engineBodies.size(); ++index) {
        const auto row = layout.BodyRow(engineBodies[index]);
        if (!row)
            throw std::invalid_argument("Warp body read requires an accepted body in the selected world");
        rows.emplace_back(*row, index);
    }
    std::sort(rows.begin(), rows.end());

    constexpr auto bytes = sizeof(WarpBodyState);
    uint32_t previous = NoModelIndex;
    std::vector<rhi::ComputeBufferRead> reads;
    for (const auto &[row, index] : rows) {
        if (row != previous) {
            if (reads.empty() || row != previous + 1) {
                reads.push_back({source.states, (source.firstRow + row) * bytes, 0});
                result.m_rangeRows.push_back(0);
            }
            reads.back().byteSize += bytes;
            ++result.m_rangeRows.back();
            ++result.m_uniqueRows;
            previous = row;
        }
        const auto &body = layout.bodies[row];
        result.m_bodies[index] = {body.bodyId, result.m_uniqueRows - 1, body.shapeRevision};
    }
    if (!reads.empty()) {
        auto host = m_host;
        result.packet = rhi::PrepareComputeBatch(host, {}, {}, std::move(reads));
    }
    return result;
}

WarpSceneRayBatch WarpSceneRuntime::PrepareRaycast(const WarpFixedPublication &publication, uint32_t world,
                                                   const std::vector<WarpRay> &rays, bool readback) const
{
    if (publication.m_layout != m_layout || rays.size() > UINT32_MAX)
        throw std::invalid_argument("Warp ray batch does not match its scene publication");
    const auto source = publication.PoseSource(world);
    std::vector<uint8_t> inputs(rays.size() * sizeof(WarpRay));
    if (!inputs.empty())
        std::memcpy(inputs.data(), rays.data(), inputs.size());
    WarpSceneRayBatch result;
    result.identity = publication.worlds[world].identity;
    result.query = m_step->PreparePublishedRays(source, static_cast<uint32_t>(rays.size()), readback,
                                                {{m_step->Rays().inputs, 0, std::move(inputs)}});
    if (readback) {
        result.m_layout = publication.m_layout;
        result.m_world = world;
        result.m_lines.reserve(rays.size());
        for (const auto &ray : rays)
            result.m_lines.push_back({{ray.originLimit[0], ray.originLimit[1], ray.originLimit[2]},
                                      {ray.direction[0], ray.direction[1], ray.direction[2]}});
    }
    return result;
}

FrameWorkFacts WarpSceneRuntime::Describe() const
{
    FrameWorkFacts facts;
    facts.pending = m_operation || m_packet || m_receipt;
    facts.observe = m_receipt.has_value();
    if (m_receipt) {
        facts.gpuBusy = !m_receipt->ready;
        facts.host = {m_generation, 0, m_receipt->ready, false};
    } else if (m_packet && (m_preservation || !(m_packet->RequiresPreparation() || m_packet->HasPreservationWork()))) {
        facts.gpu = {m_generation, 0, true, static_cast<bool>(m_operation)};
    } else if (facts.pending) {
        facts.host = {m_generation, 0, true, false};
    }
    return facts;
}

bool WarpSceneRuntime::Advance(bool gpu, uint64_t token)
{
    if (token != m_generation)
        return false;
    if (gpu)
        return Submit();
    if (m_receipt) {
        if (!m_receipt->ready)
            return false;
        Finish();
    } else if (m_packet) {
        const auto bytes = std::max(uint64_t{4}, m_work.ResourceRemaining(FrameWorkResource::Transfer));
        auto prepared = m_packet->PreparePreservation(bytes, true);
        if (!prepared.ready)
            return false;
        m_preservation = std::move(prepared.copy);
    } else if (m_operation->integrate) {
        const auto output = m_operation->host ? WarpBodyPublication::Cpu : WarpBodyPublication::Status;
        m_packet = m_step->PrepareControlled(m_operation->commands, output, {}, m_operation->captureContacts,
                                             m_operation->jointUpdates);
        m_packetKind = Packet::Fixed;
        m_operation->commands.clear();
        m_operation->jointUpdates.clear();
    } else {
        m_packet = m_step->PrepareEdits(m_operation->commands);
        m_packetKind = Packet::Edit;
        m_operation->commands.clear();
    }
    ++m_generation;
    return true;
}

bool WarpSceneRuntime::Submit()
{
    const auto &packet = m_preservation ? m_preservation : m_packet;
    const auto kind = m_preservation ? Packet::Preserve : m_packetKind;
    const auto bytes = packet->GetTransferBytes();
    const auto dispatches = packet->GetExecutionSignature().front();
    const auto prediction = kind == Packet::Fixed && m_fixedGpuNanoseconds != 0
                                ? m_fixedGpuNanoseconds
                                : 4'000 * std::max(uint64_t{1}, dispatches);
    if (!m_work.CanAdmitResource(FrameWorkResource::Gpu, prediction, true) ||
        (bytes != 0 && !m_work.CanAdmitResource(FrameWorkResource::Transfer, bytes, true)) || !packet->TrySubmit())
        return false;
    m_receipt.emplace(Receipt{packet->GetCompletion(), kind, m_work.Opportunity(),
                              m_work.RecordAcceptedResource(FrameWorkResource::Gpu, prediction),
                              bytes == 0 ? 0 : m_work.RecordAcceptedResource(FrameWorkResource::Transfer, bytes),
                              bytes});
    ++m_generation;
    return true;
}

void WarpSceneRuntime::Observe()
{
    if (m_receipt && !m_receipt->ready && m_receipt->completion.IsComplete()) {
        m_receipt->ready = true;
        ++m_generation;
        m_work.ReportProgress();
    }
}

void WarpSceneRuntime::Finish()
{
    const auto kind = m_receipt->kind;
    if (const auto measured = m_receipt->completion.GetDeviceElapsedNanoseconds()) {
        const auto nanos = static_cast<uint64_t>(*measured);
        m_work.ObserveResource(FrameWorkResource::Gpu, nanos, m_receipt->opportunity, m_receipt->gpuGrant);
        if (kind == Packet::Fixed)
            m_fixedGpuNanoseconds = nanos;
    }
    if (m_receipt->transferBytes != 0)
        m_work.ObserveResource(FrameWorkResource::Transfer, m_receipt->transferBytes, m_receipt->opportunity,
                               m_receipt->transferGrant);
    m_receipt.reset();
    if (kind == Packet::Preserve) {
        m_preservation.reset();
        return;
    }
    const auto packet = std::exchange(m_packet, {});
    if (m_retiring)
        return;
    if (kind == Packet::Initialize) {
        if (!m_operation) {
            auto result = CapturePublication(*packet);
            result.worlds = m_acceptedWorlds;
            m_initialPublication.emplace(std::move(result));
        }
    } else if (kind == Packet::Fixed) {
        FinishFixed(*packet);
    } else if (kind == Packet::Edit) {
        auto result = CapturePublication(*packet);
        result.worlds = m_acceptedWorlds;
        PrepareConsumer(std::move(result));
    } else if (kind == Packet::Reports) {
        FinishReports(*packet);
    } else {
        if (auto seal = std::exchange(m_sealConsumer, {}))
            seal();
        auto publication = std::exchange(m_pendingPublication, std::nullopt);
        PublishCompleted(std::move(*publication));
    }
}

WarpFixedPublication WarpSceneRuntime::CapturePublication(const rhi::PreparedComputeBatch &packet) const
{
    WarpFixedPublication result;
    result.m_layout = m_layout;
    result.submission = packet.GetSubmission();
    result.bodyStates = m_step->BodyStates()->Freeze();
    result.bodiesPerWorld = m_bodyCount;
    result.substeps = m_step->Substeps();
    result.worlds.reserve(m_layout->worlds.size());
    return result;
}

void WarpSceneRuntime::FinishFixed(const rhi::PreparedComputeBatch &packet)
{
    auto readback = packet.Readback(0);
    (void)readback->IsComplete(); // The enclosing packet receipt is already complete.
    const auto data = readback->GetData();
    auto result = CapturePublication(packet);
    result.m_integrated = true;
    const auto worldCount = m_layout->worlds.size();
    size_t offset = 0;
    if (m_operation->host) {
        result.hostBodies.resize(uint64_t{m_bodyCount} * worldCount);
        offset = result.hostBodies.size() * sizeof(WarpBodyState);
        std::memcpy(result.hostBodies.data(), data.data(), offset);
    }
    for (const auto &identity : m_operation->identities) {
        ModelWords header;
        std::memcpy(header.data(), data.data() + offset, sizeof(header));
        offset += sizeof(header);
        float time;
        std::memcpy(&time, header.data(), sizeof(time));
        result.worlds.push_back({identity, time, header[1], header[2]});
    }
    if (m_operation->captureContacts) {
        result.contacts.resize(worldCount * result.substeps);
        if (m_step->HasContactReports()) {
            auto receipt = packet.Readback(1);
            (void)receipt->IsComplete();
            const auto counts = receipt->GetData();
            std::vector<uint32_t> acceptedCounts(result.contacts.size());
            for (size_t row = 0; row < acceptedCounts.size(); ++row)
                if (result.worlds[row % worldCount].status == 0)
                    std::memcpy(&acceptedCounts[row], counts.data() + row * sizeof(uint32_t), sizeof(uint32_t));
            if (auto next = m_step->PrepareContactReports(acceptedCounts)) {
                for (size_t row = 0; row < acceptedCounts.size(); ++row)
                    result.contacts[row].resize(acceptedCounts[row]);
                m_pendingPublication.emplace(std::move(result));
                m_packet = std::move(next);
                m_packetKind = Packet::Reports;
                return;
            }
        }
    }
    PrepareConsumer(std::move(result));
}

void WarpSceneRuntime::FinishReports(const rhi::PreparedComputeBatch &packet)
{
    auto result = std::exchange(m_pendingPublication, std::nullopt);
    size_t index = 0;
    for (auto &reports : result->contacts) {
        if (reports.empty())
            continue;
        auto readback = packet.Readback(index++);
        (void)readback->IsComplete();
        const auto bytes = readback->GetData();
        std::memcpy(reports.data(), bytes.data(), reports.size() * sizeof(WarpContactReport));
    }
    PrepareConsumer(std::move(*result));
}

void WarpSceneRuntime::PrepareConsumer(WarpFixedPublication publication)
{
    auto work = m_operation->consumer ? m_operation->consumer(publication) : PublicationWork{};
    if (work.packet) {
        m_packet = std::move(work.packet);
        m_sealConsumer = std::move(work.seal);
        m_pendingPublication.emplace(std::move(publication));
        m_packetKind = Packet::Consumer;
    } else {
        if (work.seal)
            work.seal();
        PublishCompleted(std::move(publication));
    }
}

void WarpSceneRuntime::PublishCompleted(WarpFixedPublication publication)
{
    if (m_operation->integrate)
        for (size_t i = 0; i < publication.worlds.size(); ++i)
            if (publication.worlds[i].status == 0)
                m_acceptedWorlds[i] = publication.worlds[i];
    m_operation.reset();
    m_publication.emplace(std::move(publication));
}

FrameWorkWaitResult WarpSceneRuntime::WaitForProgress(uint64_t nanoseconds)
{
    if (m_receipt) {
        if (m_receipt->ready)
            return FrameWorkWaitResult::Progress;
        return m_host.Queue().WaitFor(m_receipt->completion.GetSubmission(), nanoseconds)
                   ? FrameWorkWaitResult::Progress
                   : FrameWorkWaitResult::Waiting;
    }
    if (m_packet && m_packet->RequiresPreparation())
        if (const auto moved = m_packet->WaitForPreparation(nanoseconds))
            return *moved ? FrameWorkWaitResult::Progress : FrameWorkWaitResult::Waiting;
    const auto ticket = m_host.Queue().GetCapacityPredecessor();
    if (ticket.IsValid())
        return m_host.Queue().WaitFor(ticket, nanoseconds) ? FrameWorkWaitResult::Progress
                                                           : FrameWorkWaitResult::Waiting;
    return FrameWorkWaitResult::Blocked;
}

void WarpSceneRuntime::Retire() noexcept
{
    m_retiring = true;
    m_operation.reset();
    m_initialPublication.reset();
    m_publication.reset();
    m_pendingPublication.reset();
    m_sealConsumer = {};
    if (m_packet && !m_packet->GetSubmission().IsValid()) {
        m_packet->Cancel();
        m_packet.reset();
    }
    if (m_preservation && !m_preservation->GetSubmission().IsValid()) {
        m_preservation->Cancel();
        m_preservation.reset();
    }
    ++m_generation;
}

bool WarpSceneRuntime::CanRetire() const noexcept
{
    return !m_packet && !m_preservation && !m_receipt;
}

} // namespace infernux::physics::warp
