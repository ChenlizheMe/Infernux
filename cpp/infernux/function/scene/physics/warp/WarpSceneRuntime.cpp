// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "WarpSceneRuntime.h"
#include "WarpSceneModel.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace infernux::physics::warp
{

struct WarpSceneLayout
{
    struct World
    {
        uint64_t id;
        std::vector<WarpRegisteredBody> bodies; // Device row, including world row zero.
        std::unordered_map<uint32_t, uint32_t> rows;
        std::unordered_map<uint64_t, uint32_t> hinges;

        [[nodiscard]] std::optional<uint32_t> BodyRow(uint32_t bodyId) const
        {
            const auto found = rows.find(bodyId);
            return found == rows.end() ? std::nullopt : std::make_optional(found->second);
        }
    };
    std::vector<World> worlds;
};

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
    std::unordered_set<uint64_t> ids;
    for (const auto &source : worlds) {
        if (source.bodies.size() != scene.bodies.size() || !ids.insert(source.worldId).second)
            throw std::invalid_argument("Warp scene world does not match its shared model");
        WarpSceneLayout::World world{source.worldId, std::vector<WarpRegisteredBody>(m_bodyCount), {}, {}};
        world.rows.reserve(source.bodies.size());
        for (size_t index = 0; index < source.bodies.size(); ++index) {
            const auto &binding = source.bodies[index];
            const auto row = scene.bodies[index].bodyRow;
            if (binding.bodyId == UINT32_MAX || !world.rows.emplace(binding.bodyId, row).second)
                throw std::invalid_argument("Duplicate or invalid Warp scene body identity");
            world.bodies[row] = binding;
        }
        world.hinges.reserve(source.hinges.size());
        for (const auto &binding : source.hinges) {
            const auto row = scene.hingeRows.at(binding.authorRow);
            if (row == NoModelIndex || !world.hinges.emplace(binding.jointId, row).second)
                throw std::invalid_argument("Duplicate or invalid Warp scene hinge identity");
        }
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

void WarpSceneRuntime::BeginFixed(std::vector<ContactStreamIdentity> identities, std::vector<WarpBodyCommand> commands,
                                  bool host, bool captureContacts, std::vector<WarpSceneDriveUpdate> drives,
                                  FixedConsumer consumer)
{
    if (m_retiring || m_fixed || m_publication)
        throw std::logic_error("Warp scene still owns a fixed step");
    if (captureContacts && !m_contactReportsConfigured)
        throw std::logic_error("Warp scene was prepared without contact report storage");
    const auto &worlds = m_layout->worlds;
    if (identities.size() != worlds.size())
        throw std::invalid_argument("Warp fixed step requires an identity per world");
    for (size_t world = 0; world < identities.size(); ++world)
        if (identities[world].worldId != worlds[world].id)
            throw std::invalid_argument("Warp fixed step identities must follow the enrolled world order");
    std::vector<WarpHingeDriveUpdate> updates;
    updates.reserve(drives.size());
    for (const auto &update : drives)
        updates.push_back({update.world, worlds.at(update.world).hinges.at(update.jointId), update.drive.Pack()});
    m_fixed.emplace(Fixed{std::move(identities), std::move(commands), std::move(updates), host, captureContacts,
                          std::move(consumer)});
    ++m_generation;
    m_work.Wake(m_source);
}

std::optional<WarpFixedPublication> WarpSceneRuntime::TakeFixed()
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
    const auto vector = [](const ModelVector &v) { return glm::vec3(v[0], v[1], v[2]); };
    const auto &q = raw.pose.rotation;
    return WarpHostBody{{engineBody, vector(raw.pose.position), glm::quat(q[0], q[1], q[2], q[3])},
                        layout.bodies[*row].shapeRevision,
                        vector(raw.centerOfMass),
                        vector(raw.linearVelocity),
                        vector(raw.angularVelocity)};
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
    return {publication.worlds[world].identity,
            m_step->PreparePublishedRays(source, static_cast<uint32_t>(rays.size()), readback,
                                         {{m_step->Rays().inputs, 0, std::move(inputs)}})};
}

FrameWorkFacts WarpSceneRuntime::Describe() const
{
    FrameWorkFacts facts;
    facts.pending = m_fixed || m_packet || m_receipt;
    facts.observe = m_receipt.has_value();
    if (m_receipt) {
        facts.gpuBusy = !m_receipt->ready;
        facts.host = {m_generation, 0, m_receipt->ready, false};
    } else if (m_packet && (m_preservation || !(m_packet->RequiresPreparation() || m_packet->HasPreservationWork()))) {
        facts.gpu = {m_generation, 0, true, static_cast<bool>(m_fixed)};
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
    } else {
        const auto output = m_fixed->host ? WarpBodyPublication::Cpu : WarpBodyPublication::Status;
        m_packet = m_step->PrepareControlled(m_fixed->commands, output, {}, m_fixed->captureContacts, m_fixed->drives);
        m_packetKind = Packet::Fixed;
        m_fixed->commands.clear();
        m_fixed->drives.clear();
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
    if (m_retiring || kind == Packet::Initialize)
        return;
    if (kind == Packet::Fixed)
        FinishFixed(*packet);
    else if (kind == Packet::Reports)
        FinishReports(*packet);
    else {
        auto publication = std::exchange(m_pendingPublication, std::nullopt);
        PublishFixed(std::move(*publication));
    }
}

void WarpSceneRuntime::FinishFixed(const rhi::PreparedComputeBatch &packet)
{
    auto readback = packet.Readback(0);
    (void)readback->IsComplete(); // The enclosing packet receipt is already complete.
    const auto data = readback->GetData();
    WarpFixedPublication result;
    result.m_layout = m_layout;
    result.submission = packet.GetSubmission();
    result.bodyStates = m_step->BodyStates()->Freeze();
    result.bodiesPerWorld = m_bodyCount;
    result.substeps = m_step->Substeps();
    const auto worldCount = m_layout->worlds.size();
    size_t offset = 0;
    if (m_fixed->host) {
        result.hostBodies.resize(uint64_t{m_bodyCount} * worldCount);
        offset = result.hostBodies.size() * sizeof(WarpBodyState);
        std::memcpy(result.hostBodies.data(), data.data(), offset);
    }
    result.worlds.reserve(worldCount);
    for (const auto &identity : m_fixed->identities) {
        ModelWords header;
        std::memcpy(header.data(), data.data() + offset, sizeof(header));
        offset += sizeof(header);
        float time;
        std::memcpy(&time, header.data(), sizeof(time));
        result.worlds.push_back({identity, time, header[1], header[2]});
    }
    if (m_fixed->captureContacts) {
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
    if (m_fixed->consumer)
        m_packet = m_fixed->consumer(publication);
    if (m_packet) {
        m_pendingPublication.emplace(std::move(publication));
        m_packetKind = Packet::Consumer;
    } else {
        PublishFixed(std::move(publication));
    }
}

void WarpSceneRuntime::PublishFixed(WarpFixedPublication publication)
{
    m_fixed.reset();
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
    m_fixed.reset();
    m_publication.reset();
    m_pendingPublication.reset();
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
