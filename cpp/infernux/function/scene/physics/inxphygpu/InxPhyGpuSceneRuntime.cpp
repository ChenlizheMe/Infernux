// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "InxPhyGpuSceneRuntime.h"
#include "InxPhyGpuContactPublication.h"

#include <algorithm>
#include <cmath>
#include <core/config/EngineConfig.h>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace infernux::physics::inxphygpu
{

struct InxPhyGpuSceneLayout
{
    struct World
    {
        std::vector<InxPhyGpuRegisteredBody> bodies; // Device row, including world row zero.
        std::unordered_map<uint32_t, uint32_t> rows;
        std::unordered_map<uint64_t, InxPhyGpuSceneJointBinding> joints;

        [[nodiscard]] std::optional<uint32_t> BodyRow(uint32_t bodyId) const
        {
            const auto found = rows.find(bodyId);
            return found == rows.end() ? std::nullopt : std::make_optional(found->second);
        }
    };
    struct Geometry
    {
        uint32_t body, surface, subshape;
        uint32_t layers, flags;
        ModelVector halfBounds, boundsCenter, material;
    };
    enum class Drive : uint8_t
    {
        None,
        Kinematic,
        Static,
    };
    std::vector<World> worlds;
    bool hasKinematicBodies = false;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> colliderGeometry; // Body row, collider order.
    std::unordered_map<uint64_t, uint32_t> worldRows;
    std::vector<Geometry> geometries;
    std::vector<std::vector<PhysicsGpuBoxAuthor>> boxes;
    std::vector<PhysicsGpuMotionBinding> motion;
    std::vector<Drive> drive;

    [[nodiscard]] bool CanDrive(uint32_t body, bool temporaryStatic) const
    {
        return drive[body] == (temporaryStatic ? Drive::Static : Drive::Kinematic);
    }

    [[nodiscard]] uint32_t WorldRow(uint64_t id) const
    {
        const auto found = worldRows.find(id);
        if (found == worldRows.end())
            throw std::invalid_argument("inxphygpu scene command requires an enrolled world");
        return found->second;
    }
};

struct InxPhyGpuScenePoseTargets
{
    using World = std::unordered_map<uint64_t, glm::quat>;
    std::vector<std::shared_ptr<const World>> worlds;
};

struct InxPhyGpuSceneKinematicTargets
{
    std::map<std::pair<uint32_t, uint32_t>, PhysicsBodyCommand::KinematicMove> rows;
};

namespace
{
ModelWords PackPoseTarget(const InxPhyGpuSceneJointBinding &joint, const glm::quat &relative)
{
    const auto &frame = joint.reference;
    const auto target =
        glm::normalize(glm::conjugate(frame.connected.orientation) * relative * frame.owner.orientation);
    const ModelVector values{target.w, target.x, target.y, target.z};
    ModelWords words;
    std::memcpy(words.data(), values.data(), sizeof(words));
    return words;
}

InxPhyGpuHostBody DecodeHostBody(const InxPhyGpuBodyState &raw, uint32_t bodyId, uint64_t shapeRevision)
{
    const auto vector = [](const ModelVector &v) { return glm::vec3(v[0], v[1], v[2]); };
    const auto &q = raw.pose.rotation;
    return {{bodyId, vector(raw.pose.position), glm::quat(q[0], q[1], q[2], q[3])},
            shapeRevision,
            vector(raw.centerOfMass),
            vector(raw.linearVelocity),
            vector(raw.angularVelocity),
            raw.centerOfMass[3] != 0.0f};
}
} // namespace

std::optional<uint32_t> InxPhyGpuFixedPublication::FindAcceptedWorld(ContactStreamIdentity identity) const
{
    if (!m_layout)
        return std::nullopt;
    const auto found = m_layout->worldRows.find(identity.worldId);
    if (found == m_layout->worldRows.end())
        return std::nullopt;
    const auto &world = worlds[found->second];
    if (world.identity.simulationEpoch != identity.simulationEpoch ||
        world.identity.fixedStepIndex != identity.fixedStepIndex)
        return std::nullopt;
    return found->second;
}

std::optional<InxPhyGpuSceneJointReference> InxPhyGpuFixedPublication::FindJointReference(uint32_t world,
                                                                                          uint64_t jointId) const
{
    const auto &joints = m_layout->worlds.at(world).joints;
    const auto found = joints.find(jointId);
    return found == joints.end() ? std::nullopt : std::make_optional(found->second.reference);
}

PhysicsGpuPoseSource InxPhyGpuFixedPublication::PoseSource(uint32_t world) const
{
    (void)worlds.at(world);
    return {bodyStates, uint64_t{world} * bodiesPerWorld, bodiesPerWorld, sizeof(InxPhyGpuBodyState),
            PhysicsGpuPoseSource::Layout::OriginWXYZ};
}

std::optional<PhysicsGpuBodyPoseBinding>
InxPhyGpuFixedPublication::FindPoseBody(uint32_t world, PhysicsActorHandle actor, uint32_t bodyId) const
{
    const auto &layout = m_layout->worlds.at(world);
    const auto row = layout.BodyRow(bodyId);
    if (!row || layout.bodies[*row].actor != actor)
        return std::nullopt;
    return PhysicsGpuBodyPoseBinding{bodyId, *row};
}

PhysicsGpuMotionExport InxPhyGpuFixedPublication::GatherMotion(uint32_t world,
                                                               const std::vector<uint32_t> &bodies) const
{
    PhysicsGpuMotionExport result{PoseSource(world), {}};
    const auto &layout = m_layout->worlds[world];
    result.bodies.reserve(bodies.size());
    for (const auto id : bodies) {
        const auto row = layout.BodyRow(id);
        if (!row)
            throw std::invalid_argument("inxphygpu motion selection requires an accepted body in the selected world");
        auto &binding = result.bodies.emplace_back(m_layout->motion[*row]);
        binding.pose.id = id;
        binding.shapeRevision = layout.bodies[*row].shapeRevision;
    }
    return result;
}

PhysicsGpuBoxExport InxPhyGpuFixedPublication::GatherBoxes(uint32_t world, const std::vector<uint32_t> &bodies,
                                                           bool includeTriggers) const
{
    PhysicsGpuBoxExport result{PoseSource(world), {}};
    const auto &layout = m_layout->worlds[world];
    for (size_t index = 0; index < bodies.size(); ++index) {
        const auto row = layout.BodyRow(bodies[index]);
        if (!row)
            throw std::invalid_argument("inxphygpu box selection requires an accepted body in the selected world");
        for (const auto &box : m_layout->boxes[*row])
            if (includeTriggers || !box.trigger)
                result.boxes.push_back({*row, static_cast<uint32_t>(index), box});
    }
    return result;
}

void InxPhyGpuFixedPublication::PublishContacts(uint32_t world, PhysicsContactPublication &destination,
                                                const PhysicsContactPublication::Policy &policy)
{
    const auto &fixed = worlds.at(world);
    if (!m_integrated || fixed.status != 0)
        throw std::logic_error("Contact publication requires an accepted inxphygpu fixed step");
    if (!policy.Requested()) {
        destination.PublishEmpty(fixed.identity);
        return;
    }
    if (contacts.size() != size_t{substeps} * worlds.size())
        throw std::logic_error("Accepted inxphygpu step does not carry the requested contact reports");
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
        return DecodeInxPhyGpuContactFrame(std::move(reports), resolve, policy);
    });
}

std::vector<InxPhyGpuHostBody> InxPhyGpuSceneBodyRead::ReadHostBodies() const
{
    std::vector<bool> sleeping;
    const auto samples = m_motion.Read(&sleeping);
    std::vector<InxPhyGpuHostBody> bodies;
    bodies.reserve(samples.size());
    for (size_t index = 0; index < samples.size(); ++index) {
        const auto &body = samples[index];
        bodies.push_back({{body.bodyId, body.position, body.rotation},
                          body.shapeRevision,
                          body.centerOfMass,
                          body.linearVelocity,
                          body.angularVelocity,
                          sleeping[index]});
    }
    return bodies;
}

std::vector<RaycastHit> InxPhyGpuSceneRayBatch::ReadHostHits() const
{
    if (!m_layout)
        throw std::logic_error("inxphygpu scene ray batch has no host results");
    std::vector<InxPhyGpuRayHit> records(query.count);
    query.packet->Readback(0)->ReadInto(records.data(), records.size() * sizeof(InxPhyGpuRayHit));
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

InxPhyGpuSceneRuntime::InxPhyGpuSceneRuntime(FrameWorkOwner &work, rhi::ComputeHost &host, ShaderCodeTarget target,
                                             const InxPhyGpuSceneModel &scene,
                                             std::vector<InxPhyGpuRegisteredWorld> worlds, float timestep,
                                             const ModelVector &gravity, InxPhyGpuCollisionCapacity collisionCapacity,
                                             const InxPhyGpuContactSolver &solver, uint32_t commandCapacity,
                                             bool contactReports, uint32_t substeps, InxPhyGpuRayBatchLayout rayLayout,
                                             const InxPhyGpuFixedPublication *previous,
                                             const InxPhyGpuSceneMotionSeed *acceptedMotion)
    : m_work(work), m_host(host), m_target(target), m_bodyCount(scene.model.dimensions.state[2]),
      m_contactReportsConfigured(contactReports)
{
    if (worlds.empty() || commandCapacity == 0)
        throw std::invalid_argument("inxphygpu scene requires worlds and a body command capacity");
    auto layout = std::make_shared<InxPhyGpuSceneLayout>();
    layout->hasKinematicBodies = !scene.model.initialKinematicPoses.empty();
    layout->worlds.reserve(worlds.size());
    layout->worldRows.reserve(worlds.size());
    m_acceptedWorlds.reserve(worlds.size());
    layout->geometries.reserve(scene.geometries.size());
    layout->colliderGeometry.resize(m_bodyCount);
    for (const auto &body : scene.bodies)
        layout->colliderGeometry[body.bodyRow].resize(body.surfaceCount, {NoModelIndex, 0});
    layout->boxes.resize(m_bodyCount);
    layout->motion.reserve(m_bodyCount);
    layout->drive.resize(m_bodyCount);
    for (uint32_t row = 0; row < m_bodyCount; ++row) {
        const auto &body = scene.model.bodies[row];
        if (body.dofs[2] != 0)
            layout->drive[row] = InxPhyGpuSceneLayout::Drive::Static;
        else if (body.hierarchy[3] != NoModelIndex)
            layout->drive[row] = InxPhyGpuSceneLayout::Drive::Kinematic;
        auto &motion = layout->motion.emplace_back(PhysicsGpuMotionBinding{{UINT32_MAX, row}});
        const auto &inertia = body.inertiaMass;
        if (inertia[3] > 0) {
            motion.inverseMass = 1.0f / inertia[3];
            motion.inverseInertia = {1.0f / inertia[0], 1.0f / inertia[1], 1.0f / inertia[2]};
            const auto &rotation = body.inertiaPose.rotation;
            motion.inertiaFrame = {rotation[0], rotation[1], rotation[2], rotation[3]};
            motion.frozenTranslation = body.subtree[3] & 7u;
            motion.frozenRotation = (body.subtree[3] >> 3u) & 7u;
        }
    }
    for (uint32_t row = 0; row < scene.geometries.size(); ++row) {
        const auto &geometry = scene.geometries[row];
        const auto &shape = scene.model.geometries[row];
        const auto body = shape.indices[0];
        const auto &bounds = scene.collision.bounds[row];
        const auto &ray = scene.collision.rayGeometry[row];
        const auto &material = geometry.collider;
        layout->geometries.push_back(
            {body,
             geometry.surface,
             geometry.subShape,
             ray[1],
             ray[2],
             bounds.halfExtent,
             bounds.center,
             {material.friction, material.bounciness, static_cast<float>(material.frictionCombine),
              static_cast<float>(material.bounceCombine)}});
        auto &range = layout->colliderGeometry[body][geometry.surface];
        range.first = std::min(range.first, row);
        range.second = row + 1;
        if (shape.indices[1] != Box)
            continue;
        const auto &origin = shape.pose.position;
        const auto &center = scene.model.bodies[body].inertiaPose.position;
        const auto &rotation = shape.pose.rotation;
        const auto &size = scene.collision.shapes[row].size;
        layout->boxes[body].push_back({{origin[0] - center[0], origin[1] - center[1], origin[2] - center[2]},
                                       {rotation[0], rotation[1], rotation[2], rotation[3]},
                                       {size[0], size[1], size[2]},
                                       geometry.collider.friction,
                                       geometry.collider.bounciness,
                                       geometry.surface,
                                       geometry.collider.trigger});
    }
    std::set<std::pair<uint32_t, uint32_t>> kinematicVelocities;
    for (auto &source : worlds) {
        const auto worldRow = static_cast<uint32_t>(layout->worlds.size());
        if (source.bodies.size() != scene.bodies.size() || !layout->worldRows.emplace(source.worldId, worldRow).second)
            throw std::invalid_argument("inxphygpu scene world does not match its shared model");
        InxPhyGpuSceneLayout::World world{std::vector<InxPhyGpuRegisteredBody>(m_bodyCount), {}, {}};
        world.rows.reserve(source.bodies.size());
        for (size_t index = 0; index < source.bodies.size(); ++index) {
            auto &binding = source.bodies[index];
            const auto &authored = scene.bodies[index];
            const auto row = authored.bodyRow;
            if (binding.bodyId == UINT32_MAX || !world.rows.emplace(binding.bodyId, row).second)
                throw std::invalid_argument("Duplicate or invalid inxphygpu scene body identity");
            if (!binding.surfaces.empty() && binding.surfaces.size() != authored.surfaceCount)
                throw std::invalid_argument("inxphygpu scene collider identities must match the shared model");
            world.bodies[row] = std::move(binding);
            if (!scene.model.initialKinematicMotion.empty() && layout->CanDrive(row, false) &&
                scene.model.initialKinematicMotion[scene.model.bodies[row].hierarchy[3]][0][3] != 0.0f)
                kinematicVelocities.emplace(worldRow, row);
        }
        world.joints.reserve(source.joints.size());
        for (const auto &binding : source.joints) {
            const auto &joint = scene.jointBindings.at(binding.authorRow);
            if (joint.row == NoModelIndex || !world.joints.emplace(binding.jointId, joint).second)
                throw std::invalid_argument("Duplicate or invalid inxphygpu scene joint identity");
        }
        m_acceptedWorlds.push_back({{source.worldId, source.simulationEpoch, 0}, 0.0f, 0, 0});
        layout->worlds.push_back(std::move(world));
    }
    InxPhyGpuStateHandoff handoff;
    if (acceptedMotion) {
        if (previous || worlds.size() != 1 || acceptedMotion->bodies.size() != scene.bodies.size() ||
            acceptedMotion->identity.worldId != m_acceptedWorlds[0].identity.worldId ||
            acceptedMotion->identity.simulationEpoch != m_acceptedWorlds[0].identity.simulationEpoch ||
            acceptedMotion->identity.fixedStepIndex > UINT32_MAX || acceptedMotion->poses.firstRow > UINT32_MAX / 4u ||
            acceptedMotion->poses.bodyCount > UINT32_MAX / 4u - acceptedMotion->poses.firstRow)
            throw std::invalid_argument("inxphygpu accepted motion does not match the enrolled scene");
        const auto step = static_cast<uint32_t>(acceptedMotion->identity.fixedStepIndex);
        m_acceptedWorlds[0] = {acceptedMotion->identity, acceptedMotion->fixedTime, 0, step};
        handoff.addresses.resize(m_bodyCount, NoModelIndex);
        handoff.addresses[0] = 0;
        auto &seed = handoff.motionSeed.emplace();
        seed.poses = acceptedMotion->poses;
        seed.rows.resize(m_bodyCount, {0u, 0u, 0u, NoModelIndex});
        std::memcpy(seed.rows[0].data(), &acceptedMotion->fixedTime, sizeof(float));
        seed.rows[0][2] = step;
        seed.rows[0][3] = 0;
        for (size_t author = 0; author < acceptedMotion->bodies.size(); ++author) {
            const auto &binding = acceptedMotion->bodies[author];
            const auto row = scene.bodies[author].bodyRow;
            if (binding.id != layout->worlds[0].bodies[row].bodyId)
                throw std::invalid_argument("inxphygpu motion seed has a different author body order");
            if (binding.row == NoModelIndex)
                continue;
            if (binding.row >= acceptedMotion->poses.bodyCount)
                throw std::invalid_argument("inxphygpu motion seed body exceeds the accepted pose range");
            handoff.addresses[row] = 5 * row;
            std::memcpy(seed.rows[row].data(), binding.centerOffset.data(), 3 * sizeof(float));
            seed.rows[row][3] = static_cast<uint32_t>(acceptedMotion->poses.firstRow) + binding.row;
        }
    } else if (previous) {
        InxPhyGpuScenePoseTargets retained;
        InxPhyGpuSceneKinematicTargets retainedKinematic;
        if (previous->m_poseTargets)
            retained.worlds.resize(worlds.size());
        const uint64_t header = 5 * uint64_t{previous->bodiesPerWorld} * previous->worlds.size();
        for (size_t next = 0; next < worlds.size(); ++next) {
            auto &accepted = m_acceptedWorlds[next];
            const auto old = previous->m_layout->worldRows.find(accepted.identity.worldId);
            if (old == previous->m_layout->worldRows.end())
                continue;
            const auto &prior = previous->worlds[old->second];
            if (prior.identity.simulationEpoch != accepted.identity.simulationEpoch)
                continue;
            if (!handoff.states) {
                handoff.states = previous->bodyStates;
                handoff.addresses.resize(uint64_t{m_bodyCount} * worlds.size(), NoModelIndex);
            }
            accepted = prior;
            accepted.status = 0;
            accepted.attemptedFixedStepIndex = 0;
            const uint64_t first = next * uint64_t{m_bodyCount};
            handoff.addresses[first] = static_cast<uint32_t>(header + old->second);
            const auto &oldWorld = previous->m_layout->worlds[old->second];
            for (const auto &[id, row] : layout->worlds[next].rows) {
                const auto actor = layout->worlds[next].bodies[row].actor;
                const auto oldRow = oldWorld.BodyRow(id);
                if (!oldRow || !actor.IsValid() || actor != oldWorld.bodies[*oldRow].actor)
                    continue;
                handoff.addresses[first + row] = 5 * (old->second * previous->bodiesPerWorld + *oldRow);
                const auto key = std::pair{static_cast<uint32_t>(next), row};
                if (layout->CanDrive(row, false)) {
                    if (std::binary_search(previous->m_kinematicVelocityRows.begin(),
                                           previous->m_kinematicVelocityRows.end(), std::pair{old->second, *oldRow}))
                        kinematicVelocities.insert(key);
                    else
                        kinematicVelocities.erase(key);
                }
                if (layout->motion[row].inverseMass != 0.0f &&
                    std::binary_search(previous->m_pendingWrenchRows.begin(), previous->m_pendingWrenchRows.end(),
                                       std::pair{old->second, *oldRow})) {
                    m_pendingWrenchRows.emplace_back(static_cast<uint32_t>(next), row);
                    handoff.pendingWrenches = previous->m_pendingWrenches;
                }
                if (const auto &priorTargets = previous->m_kinematicTargets) {
                    const auto pendingTarget = priorTargets->rows.find({old->second, *oldRow});
                    if (pendingTarget != priorTargets->rows.end() &&
                        layout->CanDrive(row, pendingTarget->second.temporaryStatic))
                        retainedKinematic.rows.emplace(std::pair{static_cast<uint32_t>(next), row},
                                                       pendingTarget->second);
                }
            }
            if (previous->m_poseTargets && previous->m_poseTargets->worlds[old->second]) {
                const auto &joints = layout->worlds[next].joints;
                auto targets = std::make_shared<InxPhyGpuScenePoseTargets::World>();
                for (const auto &[id, relative] : *previous->m_poseTargets->worlds[old->second]) {
                    const auto found = joints.find(id);
                    if (found == joints.end() || found->second.poseTargetOffset == NoModelIndex)
                        continue;
                    const auto &joint = found->second;
                    handoff.poseTargets.push_back(
                        {static_cast<uint32_t>(next), joint.poseTargetOffset, PackPoseTarget(joint, relative)});
                    targets->emplace(id, relative);
                }
                if (!targets->empty())
                    retained.worlds[next] = std::move(targets);
            }
        }
        if (!handoff.poseTargets.empty())
            m_poseTargets = std::make_shared<InxPhyGpuScenePoseTargets>(std::move(retained));
        if (!retainedKinematic.rows.empty())
            m_kinematicTargets = std::make_shared<InxPhyGpuSceneKinematicTargets>(std::move(retainedKinematic));
        std::sort(m_pendingWrenchRows.begin(), m_pendingWrenchRows.end());
    }
    m_kinematicVelocityRows.assign(kinematicVelocities.begin(), kinematicVelocities.end());
    m_layout = std::move(layout);
    m_activeFixedInputs.assign(m_layout->worlds.size(), {timestep, {gravity[0], gravity[1], gravity[2]}});
    const auto &config = EngineConfig::Get();
    auto prepared = InxPhyGpuStep::Build(m_host, target, scene.model, static_cast<uint32_t>(m_layout->worlds.size()),
                                         timestep, gravity, &scene.collision, collisionCapacity, &solver, rayLayout,
                                         commandCapacity, contactReports, substeps, handoff,
                                         {config.physicsTimeBeforeSleep, config.physicsPointVelocitySleepThreshold});
    if (!prepared.step)
        throw std::runtime_error("Could not allocate the inxphygpu scene world layout");
    m_step = std::move(prepared.step);
    m_packet = std::move(prepared.initialize);
    const auto observe = [this] { Observe(); };
    m_source = m_work.AddSource({observe, observe, [this] { return Describe(); },
                                 [this](bool gpu, uint64_t token) {
                                     try {
                                         return Advance(gpu, token);
                                     } catch (...) {
                                         Retire();
                                         throw;
                                     }
                                 }});
}

InxPhyGpuSceneRuntime::~InxPhyGpuSceneRuntime()
{
    Retire();
    m_work.RemoveSource(m_source);
}

std::optional<uint32_t> InxPhyGpuSceneRuntime::BodyRow(uint32_t world, uint32_t engineBody) const
{
    return m_layout->worlds.at(world).BodyRow(engineBody);
}

std::optional<InxPhyGpuFixedPublication> InxPhyGpuSceneRuntime::TakeInitial()
{
    return std::exchange(m_initialPublication, std::nullopt);
}

void InxPhyGpuSceneRuntime::BeginFixed(std::vector<ContactStreamIdentity> identities,
                                       const std::vector<InxPhyGpuSceneBodyCommand> &commands, bool host,
                                       bool captureContacts, const std::vector<InxPhyGpuSceneJointUpdate> &jointUpdates,
                                       PublicationConsumer consumer, std::vector<PhysicsFixedInputs> fixedInputs,
                                       const std::vector<InxPhyGpuSceneImpulseBatch> &deviceImpulses)
{
    if (captureContacts && !m_contactReportsConfigured)
        throw std::logic_error("inxphygpu scene was prepared without contact report storage");
    const auto &worlds = m_layout->worlds;
    if (identities.size() != worlds.size())
        throw std::invalid_argument("inxphygpu fixed step requires an identity per world");
    if (!fixedInputs.empty() && fixedInputs.size() != worlds.size())
        throw std::invalid_argument("inxphygpu fixed inputs require one row per enrolled world");
    if (fixedInputs == m_activeFixedInputs)
        fixedInputs.clear();
    for (size_t world = 0; world < identities.size(); ++world) {
        const auto &next = identities[world];
        const auto &previous = m_acceptedWorlds[world].identity;
        if (next.worldId != previous.worldId || next.simulationEpoch != previous.simulationEpoch ||
            next.fixedStepIndex <= previous.fixedStepIndex)
            throw std::invalid_argument("inxphygpu fixed step must advance the enrolled world and epoch");
    }
    std::vector<InxPhyGpuBodyCommand> actions;
    actions.reserve(commands.size());
    if (m_kinematicTargets)
        for (const auto &[key, target] : m_kinematicTargets->rows) {
            const auto [world, body] = key;
            EncodeBodyCommand(world, {worlds[world].bodies[body].bodyId, target}, actions);
        }
    for (const auto &entry : commands)
        EncodeBodyCommand(m_layout->WorldRow(entry.worldId), entry.command, actions);
    auto velocities = KinematicVelocityRows(actions);
    if (!m_kinematicVelocityRows.empty()) {
        // Only continuing kinematic bodies without commands need their own spans.
        auto uncovered = std::set(m_kinematicVelocityRows.begin(), m_kinematicVelocityRows.end());
        for (const auto &action : actions)
            uncovered.erase({action.world, action.body});
        for (const auto &[world, body] : uncovered)
            actions.push_back({world, body, InxPhyGpuBodyCommandKind::AdvanceKinematicVelocity});
    }
    if (!m_pendingWrenchRows.empty()) {
        auto uncovered = std::set(m_pendingWrenchRows.begin(), m_pendingWrenchRows.end());
        for (const auto &action : actions)
            uncovered.erase({action.world, action.body});
        for (const auto &[world, body] : uncovered)
            actions.push_back({world, body, InxPhyGpuBodyCommandKind::ApplyPendingWrench});
    }
    std::vector<InxPhyGpuJointParameterUpdate> updates;
    std::shared_ptr<InxPhyGpuScenePoseTargets> poses;
    EncodeJointUpdates(jointUpdates, updates, poses);
    std::vector<InxPhyGpuDeviceImpulseBatch> impulses;
    impulses.reserve(deviceImpulses.size());
    for (const auto &batch : deviceImpulses) {
        if (batch.bodyIds.empty())
            continue;
        const auto world = m_layout->WorldRow(batch.worldId);
        auto input = batch.input;
        if (input.buffer)
            input.buffer = input.buffer->Freeze();
        InxPhyGpuDeviceImpulseBatch rows{world, batch.substep, {}, std::move(input)};
        rows.bodies.reserve(batch.bodyIds.size());
        for (const auto id : batch.bodyIds) {
            const auto row = worlds[world].BodyRow(id);
            if (!row)
                throw std::invalid_argument("inxphygpu impulse batch requires enrolled body identities");
            rows.bodies.push_back(*row);
        }
        impulses.push_back(std::move(rows));
    }
    BeginOperation({std::move(identities),
                    std::move(actions),
                    std::move(updates),
                    host,
                    captureContacts,
                    std::move(consumer),
                    true,
                    std::move(fixedInputs),
                    std::move(poses),
                    {},
                    {},
                    std::move(velocities),
                    std::move(impulses)});
}

void InxPhyGpuSceneRuntime::EncodeJointUpdates(const std::vector<InxPhyGpuSceneJointUpdate> &jointUpdates,
                                               std::vector<InxPhyGpuJointParameterUpdate> &updates,
                                               std::shared_ptr<InxPhyGpuScenePoseTargets> &poses) const
{
    if (jointUpdates.empty())
        return;
    const auto &worlds = m_layout->worlds;
    updates.reserve(jointUpdates.size());
    std::map<std::pair<uint32_t, uint64_t>, std::optional<glm::quat>> poseChanges;
    for (const auto &update : jointUpdates) {
        const auto world = m_layout->WorldRow(update.worldId);
        const auto &joints = worlds[world].joints;
        const auto binding = joints.find(update.jointId);
        if (binding == joints.end())
            throw std::invalid_argument("inxphygpu joint update requires an enrolled joint in the selected world");
        const auto &joint = binding->second;
        const auto append = [&](uint32_t offset, const auto &value) {
            updates.push_back({world, offset, value, joint.dynamicEndpoints});
        };
        if (const auto *limits = std::get_if<InxPhyGpuJointLimits>(&update.value)) {
            const auto minimum = limits->minimum, maximum = limits->maximum;
            const bool axial = joint.kind == InxPhyGpuSceneJointKind::Hinge ||
                               joint.kind == InxPhyGpuSceneJointKind::Slider ||
                               joint.kind == InxPhyGpuSceneJointKind::Distance;
            if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum > maximum || !axial ||
                (joint.kind == InxPhyGpuSceneJointKind::Slider && (minimum > 0.0f || maximum < 0.0f)) ||
                (joint.kind == InxPhyGpuSceneJointKind::Distance &&
                 (joint.autoDistance || !limits->enabled || minimum < 0.0f)))
                throw std::invalid_argument(
                    "inxphygpu joint limits must match the enrolled constraint and its authored mode");
            append(joint.parameterOffset, limits->Pack());
        } else if (const auto *drive = std::get_if<InxPhyGpuHingeDrive>(&update.value)) {
            if (joint.kind != InxPhyGpuSceneJointKind::Hinge)
                throw std::invalid_argument("inxphygpu hinge drive requires an enrolled hinge");
            append(joint.parameterOffset + 1, drive->Pack());
        } else if (const auto *spring = std::get_if<InxPhyGpuDistanceSpring>(&update.value)) {
            if (joint.kind != InxPhyGpuSceneJointKind::Distance || !std::isfinite(spring->stiffness) ||
                !std::isfinite(spring->damping) || spring->stiffness < 0.0f || spring->damping < 0.0f)
                throw std::invalid_argument(
                    "inxphygpu distance spring requires finite nonnegative coefficients on a distance joint");
            append(joint.parameterOffset + 1, *spring);
        } else if (const auto *six = std::get_if<ModelSixDOFControls>(&update.value)) {
            if (joint.kind != InxPhyGpuSceneJointKind::SixDOF)
                throw std::invalid_argument("inxphygpu six-axis parameters require an enrolled SixDOF joint");
            append(joint.parameterOffset, *six);
        } else if (const auto *pose = std::get_if<InxPhyGpuJointPoseTarget>(&update.value)) {
            if (joint.poseTargetOffset == NoModelIndex)
                throw std::invalid_argument("inxphygpu pose drive requires a hinge, swing-twist or SixDOF joint");
            auto relative = pose->relative;
            if (relative) {
                const auto squared = glm::dot(*relative, *relative);
                if (!std::isfinite(squared) || squared <= 0.0f)
                    throw std::invalid_argument("inxphygpu pose target requires a finite nonzero rotation");
                *relative = *relative / std::sqrt(squared);
            }
            poseChanges[{world, update.jointId}] = relative;
        } else {
            if (joint.kind != InxPhyGpuSceneJointKind::SwingTwist)
                throw std::invalid_argument("inxphygpu swing-twist parameters require an enrolled swing-twist joint");
            constexpr uint32_t targetOffset = sizeof(ModelSwingTwistControls) / sizeof(ModelWords);
            if (const auto *parameters = std::get_if<InxPhyGpuSwingTwistParameters>(&update.value)) {
                append(joint.parameterOffset, parameters->controls);
                append(joint.parameterOffset + targetOffset, parameters->targets.Pack());
            } else {
                append(joint.parameterOffset + targetOffset, std::get<InxPhyGpuSwingTwistTargets>(update.value).Pack());
            }
        }
    }
    // Fold repeated animation writes before touching the resident parameters or waking a body.
    std::shared_ptr<InxPhyGpuScenePoseTargets::World> changedWorld;
    uint32_t changedRow = NoModelIndex;
    for (const auto &[key, relative] : poseChanges) {
        const auto &[world, id] = key;
        const auto prior = m_poseTargets ? m_poseTargets->worlds[world].get() : nullptr;
        const auto old = prior ? prior->find(id) : InxPhyGpuScenePoseTargets::World::const_iterator{};
        const bool wasDriven = prior && old != prior->end();
        if (relative ? wasDriven && (*relative == old->second || *relative == -old->second) : !wasDriven)
            continue;
        if (!poses) {
            poses = std::make_shared<InxPhyGpuScenePoseTargets>();
            poses->worlds = m_poseTargets
                                ? m_poseTargets->worlds
                                : std::vector<std::shared_ptr<const InxPhyGpuScenePoseTargets::World>>(worlds.size());
        }
        if (changedRow != world) {
            changedRow = world;
            changedWorld = prior ? std::make_shared<InxPhyGpuScenePoseTargets::World>(*prior)
                                 : std::make_shared<InxPhyGpuScenePoseTargets::World>();
            poses->worlds[world] = changedWorld;
        }
        const auto &joint = worlds[world].joints.at(id);
        if (relative)
            changedWorld->insert_or_assign(id, *relative);
        else
            changedWorld->erase(id);
        updates.push_back({world, joint.poseTargetOffset, relative ? PackPoseTarget(joint, *relative) : ModelWords{},
                           joint.dynamicEndpoints});
    }
}

void InxPhyGpuSceneRuntime::BeginEdits(const std::vector<InxPhyGpuSceneBodyCommand> &commands,
                                       const std::vector<InxPhyGpuSceneJointUpdate> &jointUpdates,
                                       PublicationConsumer consumer)
{
    if (commands.empty() && jointUpdates.empty())
        throw std::invalid_argument("inxphygpu scene edit requires body or joint commands");
    std::vector<InxPhyGpuBodyCommand> edits;
    edits.reserve(commands.size());
    std::shared_ptr<InxPhyGpuSceneKinematicTargets> nextTargets;
    const auto editTargets = [&]() -> InxPhyGpuSceneKinematicTargets & {
        if (!nextTargets)
            nextTargets = m_kinematicTargets ? std::make_shared<InxPhyGpuSceneKinematicTargets>(*m_kinematicTargets)
                                             : std::make_shared<InxPhyGpuSceneKinematicTargets>();
        return *nextTargets;
    };
    for (const auto &entry : commands) {
        const auto world = m_layout->WorldRow(entry.worldId);
        const auto &command = entry.command;
        const auto body = m_layout->worlds[world].BodyRow(command.bodyId);
        if (!body)
            throw std::invalid_argument("inxphygpu scene edit requires an accepted body in the selected world");
        const std::pair key{world, *body};
        if (const auto *target = std::get_if<PhysicsBodyCommand::KinematicMove>(&command.operation)) {
            if (!target->position && !target->rotation)
                continue;
            if (!m_layout->CanDrive(*body, target->temporaryStatic) || !std::isfinite(target->duration) ||
                target->duration <= 0.0f || !std::isfinite(target->maxSpeed) || target->maxSpeed < 0.0f)
                throw std::invalid_argument(
                    "Physics movement requires a matching body, positive duration and speed limit");
            auto &held = editTargets().rows[key];
            if (target->position)
                held.position = target->position;
            if (target->rotation)
                held.rotation = target->rotation;
            held.duration = target->duration;
            held.maxSpeed = target->maxSpeed;
            held.temporaryStatic = target->temporaryStatic;
            continue;
        }
        const auto *pose = std::get_if<PhysicsBodyCommand::Pose>(&command.operation);
        const auto *velocity = std::get_if<PhysicsBodyCommand::Velocity>(&command.operation);
        const auto *activity = std::get_if<PhysicsBodyCommand::Activity>(&command.operation);
        if ((pose && pose->cancelDrive && (pose->position || pose->rotation)) ||
            std::holds_alternative<PhysicsBodyCommand::ResetMotion>(command.operation) ||
            (velocity && (velocity->linear || velocity->angular) && m_layout->CanDrive(*body, false)) ||
            (activity && !activity->awake)) {
            const auto *prior = nextTargets ? nextTargets.get() : m_kinematicTargets.get();
            if (prior && prior->rows.count(key))
                editTargets().rows.erase(key);
        }
        EncodeBodyCommand(world, command, edits);
    }
    using Kind = InxPhyGpuBodyCommandKind;
    std::set pending(m_pendingWrenchRows.begin(), m_pendingWrenchRows.end());
    for (const auto &edit : edits) {
        if (edit.kind == Kind::ResetMotion || edit.kind == Kind::Sleep)
            pending.erase({edit.world, edit.body});
        else if (edit.kind >= Kind::Force && edit.kind <= Kind::ForceAtPosition && static_cast<uint32_t>(edit.mode) < 2)
            pending.emplace(edit.world, edit.body);
    }
    std::shared_ptr<const InxPhyGpuSceneKinematicTargets> targets = m_kinematicTargets;
    if (nextTargets)
        targets = nextTargets->rows.empty() ? nullptr : std::move(nextTargets);
    auto velocities = KinematicVelocityRows(edits);
    std::vector<InxPhyGpuJointParameterUpdate> updates;
    std::shared_ptr<InxPhyGpuScenePoseTargets> poses;
    EncodeJointUpdates(jointUpdates, updates, poses);
    BeginOperation({{},
                    std::move(edits),
                    std::move(updates),
                    false,
                    false,
                    std::move(consumer),
                    false,
                    {},
                    std::move(poses),
                    {pending.begin(), pending.end()},
                    std::move(targets),
                    std::move(velocities)});
}

std::vector<std::pair<uint32_t, uint32_t>>
InxPhyGpuSceneRuntime::KinematicVelocityRows(const std::vector<InxPhyGpuBodyCommand> &commands) const
{
    if (commands.empty() || !m_layout->hasKinematicBodies)
        return m_kinematicVelocityRows;
    auto rows = std::set(m_kinematicVelocityRows.begin(), m_kinematicVelocityRows.end());
    decltype(rows) targets;
    using Kind = InxPhyGpuBodyCommandKind;
    for (const auto &command : commands) {
        if (!m_layout->CanDrive(command.body, false))
            continue;
        const auto key = std::pair{command.world, command.body};
        if (command.kind == Kind::SetLinearVelocity || command.kind == Kind::SetAngularVelocity) {
            rows.insert(key);
            targets.erase(key);
        } else if (command.kind == Kind::MovePosition || command.kind == Kind::MoveRotation) {
            targets.insert(key);
        } else if (command.kind == Kind::Sleep || command.kind == Kind::ResetMotion) {
            rows.erase(key);
            targets.erase(key);
        } else if ((command.kind == Kind::SetPosition || command.kind == Kind::SetRotation) &&
                   command.poseFlags != InxPhyGpuPoseCommandFlags::PreserveDrive) {
            targets.erase(key);
        }
    }
    for (const auto &target : targets)
        rows.erase(target);
    return {rows.begin(), rows.end()};
}

void InxPhyGpuSceneRuntime::EncodeBodyCommand(uint32_t world, const PhysicsBodyCommand &command,
                                              std::vector<InxPhyGpuBodyCommand> &out) const
{
    const auto row = m_layout->worlds[world].BodyRow(command.bodyId);
    if (!row)
        throw std::invalid_argument("inxphygpu scene command requires an accepted body in the selected world");
    const auto vector = [](const glm::vec3 &v) -> ModelVector { return {v.x, v.y, v.z, 0.0f}; };
    const auto rotation = [](const glm::quat &q) -> ModelVector { return {q.w, q.x, q.y, q.z}; };
    using Kind = InxPhyGpuBodyCommandKind;
    const auto append = [&](Kind kind, ModelVector value = {}, ForceMode mode = {},
                            ModelVector point = {}) -> InxPhyGpuBodyCommand & {
        return out.emplace_back(InxPhyGpuBodyCommand{world, *row, kind, mode, value, point});
    };
    if (const auto *pose = std::get_if<PhysicsBodyCommand::Pose>(&command.operation)) {
        const auto flags =
            pose->cancelDrive ? InxPhyGpuPoseCommandFlags::None : InxPhyGpuPoseCommandFlags::PreserveDrive;
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
    } else if (const auto *move = std::get_if<PhysicsBodyCommand::KinematicMove>(&command.operation)) {
        const ModelVector timing{move->duration, move->maxSpeed};
        const auto flags =
            move->temporaryStatic ? InxPhyGpuPoseCommandFlags::TemporaryStatic : InxPhyGpuPoseCommandFlags::None;
        if (move->position)
            append(Kind::MovePosition, vector(*move->position), {}, timing).poseFlags = flags;
        if (move->rotation)
            append(Kind::MoveRotation, rotation(*move->rotation), {}, timing).poseFlags = flags;
    } else if (const auto *activity = std::get_if<PhysicsBodyCommand::Activity>(&command.operation)) {
        append(activity->awake ? Kind::Wake : Kind::Sleep);
    } else {
        const auto &reset = std::get<PhysicsBodyCommand::ResetMotion>(command.operation);
        append(Kind::ResetMotion);
        append(Kind::SetPosition, vector(reset.position));
        append(Kind::SetRotation, rotation(reset.rotation));
        append(Kind::SetLinearVelocity, vector(reset.linear));
        append(Kind::SetAngularVelocity, vector(reset.angular));
    }
}

void InxPhyGpuSceneRuntime::BeginOperation(Operation operation)
{
    if (m_retiring || m_operation || m_publication)
        throw std::logic_error("inxphygpu scene still owns a physics operation");
    m_initialPublication.reset();
    m_operation.emplace(std::move(operation));
    ++m_generation;
    m_work.Wake(m_source);
}

std::optional<InxPhyGpuFixedPublication> InxPhyGpuSceneRuntime::TakePublication()
{
    return std::exchange(m_publication, std::nullopt);
}

std::optional<InxPhyGpuHostBody> InxPhyGpuSceneRuntime::ReadHostBody(const InxPhyGpuFixedPublication &publication,
                                                                     uint32_t world, uint32_t engineBody) const
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

InxPhyGpuSceneBodyRead InxPhyGpuSceneRuntime::PrepareBodyRead(const InxPhyGpuFixedPublication &publication,
                                                              uint32_t world,
                                                              const std::vector<uint32_t> &engineBodies) const
{
    if (publication.m_layout != m_layout)
        throw std::invalid_argument("inxphygpu body read does not match its scene publication");
    auto source = publication.GatherMotion(world, engineBodies);
    constexpr auto fields = static_cast<PhysicsStateFields>(static_cast<uint32_t>(PhysicsStateFields::Position) |
                                                            static_cast<uint32_t>(PhysicsStateFields::Rotation) |
                                                            static_cast<uint32_t>(PhysicsStateFields::CenterOfMass) |
                                                            static_cast<uint32_t>(PhysicsStateFields::LinearVelocity) |
                                                            static_cast<uint32_t>(PhysicsStateFields::AngularVelocity));
    InxPhyGpuSceneBodyRead result;
    result.identity = publication.worlds[world].identity;
    auto host = m_host;
    result.m_motion = PhysicsGpuMotionRead::Prepare(host, source.poses, std::move(source.bodies), fields);
    result.packet = result.m_motion.packet;
    return result;
}

InxPhyGpuPreparedJointMotion InxPhyGpuSceneRuntime::PrepareJointMotion(const InxPhyGpuFixedPublication &publication,
                                                                       uint32_t world,
                                                                       const std::vector<uint64_t> &jointIds,
                                                                       bool readback) const
{
    if (publication.m_layout != m_layout)
        throw std::invalid_argument("inxphygpu joint motion does not match its scene publication");
    const auto source = publication.PoseSource(world);
    const auto &joints = m_layout->worlds[world].joints;
    std::vector<uint32_t> rows;
    rows.reserve(jointIds.size());
    for (const auto id : jointIds) {
        const auto joint = joints.find(id);
        if (joint == joints.end())
            throw std::invalid_argument("inxphygpu joint motion requires an accepted joint in the selected world");
        rows.push_back(joint->second.row);
    }
    return m_step->PrepareJointMotion(source, rows, readback);
}

InxPhyGpuSceneRayBatch InxPhyGpuSceneRuntime::PrepareDeviceRaycast(const InxPhyGpuFixedPublication &publication,
                                                                   uint32_t world, const InxPhyGpuDeviceRays &rays,
                                                                   const PhysicsDeviceRayOutputs *outputs) const
{
    if (publication.m_layout != m_layout)
        throw std::invalid_argument("inxphygpu ray batch does not match its scene publication");
    InxPhyGpuSceneRayBatch result;
    const auto source = publication.PoseSource(world);
    result.identity = publication.worlds[world].identity;
    result.query = m_step->PreparePublishedRays(source, rays.count, false, {}, &rays, outputs);
    return result;
}

InxPhyGpuSceneRayBatch InxPhyGpuSceneRuntime::PrepareRaycast(const InxPhyGpuFixedPublication &publication,
                                                             uint32_t world, const std::vector<InxPhyGpuRay> &rays,
                                                             bool readback) const
{
    if (publication.m_layout != m_layout || rays.size() > UINT32_MAX)
        throw std::invalid_argument("inxphygpu ray batch does not match its scene publication");
    const auto source = publication.PoseSource(world);
    std::vector<uint8_t> inputs(rays.size() * sizeof(InxPhyGpuRay));
    if (!inputs.empty())
        std::memcpy(inputs.data(), rays.data(), inputs.size());
    InxPhyGpuSceneRayBatch result;
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

InxPhyGpuSceneBodyBoundsRead InxPhyGpuSceneRuntime::PrepareBodyBounds(const InxPhyGpuFixedPublication &publication,
                                                                      uint32_t world, const ModelVector &minimum,
                                                                      const ModelVector &maximum, uint32_t layers,
                                                                      bool triggers) const
{
    if (publication.m_layout != m_layout)
        throw std::invalid_argument("inxphygpu body bounds do not match the scene publication");
    InxPhyGpuSceneBodyBoundsRead result;
    result.identity = publication.worlds.at(world).identity;
    result.query =
        m_step->PreparePublishedBodyBounds(publication.PoseSource(world), minimum, maximum, layers, triggers);
    result.m_layout = m_layout;
    result.m_world = world;
    return result;
}

std::shared_ptr<rhi::PreparedComputeBatch>
InxPhyGpuSceneRuntime::PrepareClosestPoint(const InxPhyGpuFixedPublication &publication, uint32_t world,
                                           uint32_t bodyId, uint64_t colliderId, uint64_t generation,
                                           const ModelVector &point, bool closedMesh) const
{
    if (publication.m_layout != m_layout)
        throw std::invalid_argument("inxphygpu closest point does not match the scene publication");
    const auto &layout = m_layout->worlds.at(world);
    const auto row = layout.BodyRow(bodyId);
    if (!row)
        return {};
    const auto &surfaces = layout.bodies[*row].surfaces;
    const auto surface = std::find_if(surfaces.begin(), surfaces.end(), [&](const auto &identity) {
        return identity.colliderId == colliderId && identity.generation == generation;
    });
    if (surface == surfaces.end())
        return {};
    const auto &range = m_layout->colliderGeometry[*row][surface - surfaces.begin()];
    if (range.first == NoModelIndex)
        return {};
    return m_step->PreparePublishedClosestPoint(publication.PoseSource(world), point, range.first, range.second,
                                                closedMesh);
}

std::vector<uint32_t> InxPhyGpuSceneBodyBoundsRead::ReadHostBodyIds() const
{
    std::vector<uint32_t> ids;
    for (const auto row : query.ReadMatchingRows())
        ids.push_back(m_layout->worlds[m_world].bodies[row].bodyId);
    return ids;
}

InxPhyGpuSceneRuntime::PublicationWork InxPhyGpuSceneRuntime::PrepareParticleCollision(
    const InxPhyGpuFixedPublication &publication, uint32_t world, const Scene *scene,
    const std::shared_ptr<const particle::GpuParticleCollisionDevicePayload> &previous, uint64_t revision,
    ParticleConsumer publish)
{
    if (publication.m_layout != m_layout)
        throw std::invalid_argument("inxphygpu particle source does not match the scene publication");
    const auto &bodies = m_layout->worlds.at(world).bodies;
    std::vector<InxPhyGpuParticleSurface> surfaces;
    surfaces.reserve(m_layout->geometries.size());
    for (uint32_t row = 0; row < m_layout->geometries.size(); ++row) {
        const auto &geometry = m_layout->geometries[row];
        const auto &body = bodies[geometry.body];
        if (geometry.surface >= body.surfaces.size())
            continue;
        const auto &surface = body.surfaces[geometry.surface];
        surfaces.push_back(
            {{surface.colliderId, surface.generation, geometry.subshape, 0,
              static_cast<uintptr_t>(body.shapeRevision + 1)},
             {{row, NoModelIndex, static_cast<uint32_t>(surface.generation),
               static_cast<uint32_t>(surface.generation >> 32u)},
              {static_cast<uint32_t>(surface.colliderId), static_cast<uint32_t>(surface.colliderId >> 32u),
               static_cast<uint32_t>(surface.gameObjectId), static_cast<uint32_t>(surface.gameObjectId >> 32u)},
              {geometry.layers, (geometry.flags >> 1u) & 1u, 0u, 0u},
              geometry.halfBounds,
              geometry.boundsCenter,
              geometry.material},
             surface.target,
             (geometry.flags & 1u) != 0u});
    }
    if (!m_particles)
        m_particles = std::make_unique<InxPhyGpuParticleExport>(m_host, m_target);
    const InxPhyGpuParticleExport::Source source{m_step->PublishedParticleGeometry(), publication.PoseSource(world),
                                                 surfaces, publication.worlds.at(world).identity, m_layout};
    auto work = std::make_shared<InxPhyGpuParticleExport::Work>(
        m_particles->Prepare(m_host, source, scene, previous, revision));
    return {work->packet, {}, [this, work, publish = std::move(publish)](const auto &) mutable -> PublicationWork {
                auto packet = work->PrepareGrid(m_host);
                return {std::move(packet), [work, publish = std::move(publish)] { publish(work->Finish()); }};
            }};
}

FrameWorkFacts InxPhyGpuSceneRuntime::Describe() const
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

bool InxPhyGpuSceneRuntime::Advance(bool gpu, uint64_t token)
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
        const auto output = m_operation->host ? InxPhyGpuBodyPublication::Cpu : InxPhyGpuBodyPublication::Status;
        m_packet =
            m_step->PrepareControlled(m_operation->commands, output, {}, m_operation->captureContacts,
                                      m_operation->jointUpdates, m_operation->fixedInputs, m_operation->deviceImpulses);
        m_packetKind = Packet::Fixed;
        m_operation->commands.clear();
        m_operation->jointUpdates.clear();
        m_operation->deviceImpulses.clear();
    } else if (m_operation->commands.empty() && m_operation->jointUpdates.empty()) {
        auto result = CapturePublication(m_step->BodyStates()->GetLastWriteSubmission());
        result.worlds = m_acceptedWorlds;
        PrepareConsumer(std::move(result));
    } else {
        m_packet = m_step->PrepareEdits(std::move(m_operation->commands), std::move(m_operation->jointUpdates));
        m_packetKind = Packet::Edit;
    }
    ++m_generation;
    return true;
}

bool InxPhyGpuSceneRuntime::Submit()
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

void InxPhyGpuSceneRuntime::Observe()
{
    if (m_receipt && !m_receipt->ready && m_receipt->completion.IsComplete()) {
        m_receipt->ready = true;
        ++m_generation;
        m_work.ReportProgress();
    }
}

void InxPhyGpuSceneRuntime::Finish()
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
            auto result = CapturePublication(packet->GetSubmission());
            result.worlds = m_acceptedWorlds;
            result.m_kinematicTargets = m_kinematicTargets;
            result.m_kinematicVelocityRows = m_kinematicVelocityRows;
            RetainWrenches(result, m_pendingWrenchRows);
            m_initialPublication.emplace(std::move(result));
        }
    } else if (kind == Packet::Fixed) {
        FinishFixed(*packet);
    } else if (kind == Packet::Edit) {
        auto result = CapturePublication(packet->GetSubmission());
        result.worlds = m_acceptedWorlds;
        PrepareConsumer(std::move(result));
    } else if (kind == Packet::Reports) {
        FinishReports(*packet);
    } else {
        if (auto seal = std::exchange(m_sealConsumer, {}))
            seal();
        auto next = std::exchange(m_continueConsumer, {});
        ContinueConsumer(next ? next(*m_pendingPublication) : PublicationWork{});
    }
}

InxPhyGpuFixedPublication InxPhyGpuSceneRuntime::CapturePublication(rhi::SubmissionTicket submission) const
{
    InxPhyGpuFixedPublication result;
    result.m_layout = m_layout;
    result.m_poseTargets = m_operation && m_operation->poseTargets ? m_operation->poseTargets : m_poseTargets;
    result.submission = submission;
    result.bodyStates = m_step->BodyStates()->Freeze();
    result.bodiesPerWorld = m_bodyCount;
    result.substeps = m_step->Substeps();
    result.worlds.reserve(m_layout->worlds.size());
    return result;
}

void InxPhyGpuSceneRuntime::RetainWrenches(InxPhyGpuFixedPublication &publication,
                                           std::vector<std::pair<uint32_t, uint32_t>> rows) const
{
    if (rows.empty())
        return;
    publication.m_pendingWrenchRows = std::move(rows);
    publication.m_pendingWrenches = m_step->PendingWrenches()->Freeze();
}

void InxPhyGpuSceneRuntime::FinishFixed(const rhi::PreparedComputeBatch &packet)
{
    auto readback = packet.Readback(0);
    (void)readback->IsComplete(); // The enclosing packet receipt is already complete.
    const auto data = readback->GetData();
    auto result = CapturePublication(packet.GetSubmission());
    result.m_integrated = true;
    const auto worldCount = m_layout->worlds.size();
    size_t offset = 0;
    if (m_operation->host) {
        result.hostBodies.resize(uint64_t{m_bodyCount} * worldCount);
        offset = result.hostBodies.size() * sizeof(InxPhyGpuBodyState);
        std::memcpy(result.hostBodies.data(), data.data(), offset);
    }
    for (size_t world = 0; world < worldCount; ++world) {
        const auto &identity = m_operation->identities[world];
        ModelWords header;
        std::memcpy(header.data(), data.data() + offset, sizeof(header));
        offset += sizeof(header);
        float time;
        std::memcpy(&time, header.data(), sizeof(time));
        // Rejected rows carry the state restored by the device rollback. Give that
        // state its accepted identity and keep the attempted index for diagnostics.
        const auto &accepted = header[1] == 0 ? identity : m_acceptedWorlds[world].identity;
        result.worlds.push_back({accepted, time, header[1], header[2], identity.fixedStepIndex});
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

void InxPhyGpuSceneRuntime::FinishReports(const rhi::PreparedComputeBatch &packet)
{
    auto result = std::exchange(m_pendingPublication, std::nullopt);
    size_t index = 0;
    for (auto &reports : result->contacts) {
        if (reports.empty())
            continue;
        auto readback = packet.Readback(index++);
        (void)readback->IsComplete();
        const auto bytes = readback->GetData();
        std::memcpy(reports.data(), bytes.data(), reports.size() * sizeof(InxPhyGpuContactReport));
    }
    PrepareConsumer(std::move(*result));
}

void InxPhyGpuSceneRuntime::PrepareConsumer(InxPhyGpuFixedPublication publication)
{
    auto velocities = std::move(m_operation->kinematicVelocityRows);
    if (m_operation->integrate) {
        const auto rejectedWorld = [&](const auto &row) { return publication.worlds[row.first].status != 0; };
        velocities.erase(std::remove_if(velocities.begin(), velocities.end(), rejectedWorld), velocities.end());
        for (const auto &row : m_kinematicVelocityRows)
            if (rejectedWorld(row))
                velocities.push_back(row);
        std::sort(velocities.begin(), velocities.end());
        std::vector<std::pair<uint32_t, uint32_t>> rejected;
        for (const auto &row : m_pendingWrenchRows)
            if (publication.worlds[row.first].status != 0)
                rejected.push_back(row);
        RetainWrenches(publication, std::move(rejected));
        if (m_kinematicTargets) {
            std::shared_ptr<InxPhyGpuSceneKinematicTargets> targets;
            for (const auto &[key, target] : m_kinematicTargets->rows)
                if (publication.worlds[key.first].status != 0) {
                    if (!targets)
                        targets = std::make_shared<InxPhyGpuSceneKinematicTargets>();
                    targets->rows.emplace(key, target);
                }
            publication.m_kinematicTargets = std::move(targets);
        }
    } else {
        RetainWrenches(publication, std::move(m_operation->pendingWrenchRows));
        publication.m_kinematicTargets = std::move(m_operation->kinematicTargets);
    }
    publication.m_kinematicVelocityRows = std::move(velocities);
    PublicationWork work;
    if (m_operation->consumer &&
        (!m_operation->integrate || std::any_of(publication.worlds.begin(), publication.worlds.end(),
                                                [](const auto &world) { return world.status == 0; })))
        work = m_operation->consumer(publication);
    m_pendingPublication.emplace(std::move(publication));
    ContinueConsumer(std::move(work));
}

void InxPhyGpuSceneRuntime::ContinueConsumer(PublicationWork work)
{
    while (!work.packet && work.continueAfterReceipt) {
        if (work.seal)
            work.seal();
        work = work.continueAfterReceipt(*m_pendingPublication);
    }
    if (work.packet) {
        m_packet = std::move(work.packet);
        m_sealConsumer = std::move(work.seal);
        m_continueConsumer = std::move(work.continueAfterReceipt);
        m_packetKind = Packet::Consumer;
    } else {
        if (work.seal)
            work.seal();
        auto publication = std::exchange(m_pendingPublication, std::nullopt);
        PublishCompleted(std::move(*publication));
    }
}

void InxPhyGpuSceneRuntime::PublishCompleted(InxPhyGpuFixedPublication publication)
{
    if (m_operation->integrate)
        for (size_t i = 0; i < publication.worlds.size(); ++i)
            if (publication.worlds[i].status == 0)
                m_acceptedWorlds[i] = publication.worlds[i];
    if (!m_operation->fixedInputs.empty())
        m_activeFixedInputs = std::move(m_operation->fixedInputs);
    m_poseTargets = publication.m_poseTargets;
    m_kinematicTargets = publication.m_kinematicTargets;
    m_kinematicVelocityRows = publication.m_kinematicVelocityRows;
    m_pendingWrenchRows = publication.m_pendingWrenchRows;
    m_operation.reset();
    m_publication.emplace(std::move(publication));
}

FrameWorkWaitResult InxPhyGpuSceneRuntime::WaitForProgress(uint64_t nanoseconds)
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

void InxPhyGpuSceneRuntime::Retire() noexcept
{
    m_retiring = true;
    m_operation.reset();
    m_initialPublication.reset();
    m_publication.reset();
    m_pendingPublication.reset();
    m_sealConsumer = {};
    m_continueConsumer = {};
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

bool InxPhyGpuSceneRuntime::CanRetire() const noexcept
{
    return !m_packet && !m_preservation && !m_receipt;
}

} // namespace infernux::physics::inxphygpu
