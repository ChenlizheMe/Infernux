// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "WarpStep.h"
#include "WarpProgramSources.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace infernux::physics::warp
{
namespace
{
using namespace rhi;
using Access = ComputeBufferAccess;
constexpr auto R = Access::Read;
constexpr auto W = Access::Write;
constexpr auto RW = Access::ReadWrite;

struct Use
{
    uint32_t binding;
    uint32_t slot;
    Access access;
    bool complete = false;
};

struct InitialValue
{
    uint32_t slot;
    std::vector<uint8_t> data;
};

class StepBuilder
{
  public:
    StepBuilder(ComputeHost &host, ShaderCodeTarget target, uint32_t worlds)
        : worlds(worlds), host(host), target(target)
    {
    }

    uint32_t Slot(uint64_t words, BufferUsageFlags usage = BufferUsageFlags::None)
    {
        const auto index = static_cast<uint32_t>(slots.size());
        slots.push_back(
            {{std::max(uint64_t{1}, words), ComputeScalarType::UInt32, 1, BufferUsageFlags::Storage | usage},
             {},
             false});
        return index;
    }

    ComputeWorkRange Range(uint32_t slot) const
    {
        return {slot, 0, slots[slot].allocation.GetByteSize()};
    }

    uint32_t Seed(std::string name, uint64_t rowWords, const void *row, uint32_t rows = 1)
    {
        const auto slot = Slot(rowWords * rows, BufferUsageFlags::TransferDestination);
        ports.push_back({std::move(name), Range(slot), true});
        auto &value = initial.emplace_back(InitialValue{slot, std::vector<uint8_t>(Range(slot).byteSize)});
        if (row && rowWords != 0) {
            const auto rowBytes = rowWords * sizeof(uint32_t);
            for (uint32_t index = 0; index < rows; ++index)
                std::memcpy(value.data.data() + index * rowBytes, row, rowBytes);
        }
        return slot;
    }

    template <class T> uint32_t Shared(std::string name, const std::vector<T> &table)
    {
        static_assert(sizeof(T) % sizeof(uint32_t) == 0);
        return Seed(std::move(name), table.size() * (sizeof(T) / sizeof(uint32_t)), table.data());
    }

    ComputeWorkDispatch Dispatch(std::string_view module, std::string_view name, uint32_t x,
                                 std::initializer_list<Use> uses, bool groups = false)
    {
        auto foundProgram = programs.find(module);
        if (foundProgram == programs.end())
            foundProgram = programs.emplace(module, LoadWarpProgram(module, target)).first;
        const auto &program = foundProgram->second;
        const auto entry = std::find_if(program.entries.begin(), program.entries.end(),
                                        [name](const auto &item) { return item.name == name; });
        if (entry == program.entries.end() || entry->bindings.size() != uses.size())
            throw std::logic_error("Warp step resources do not match " + std::string(name));

        ComputeWorkDispatch result;
        const auto width = entry->workgroupSize[0];
        result.groupCountX = groups ? x : (x - 1) / width + 1;
        result.groupCountY = worlds;
        std::vector<ComputeBufferBinding> descriptors;
        descriptors.reserve(uses.size());
        result.bindings.reserve(uses.size());
        for (const auto &use : uses) {
            const std::array<uint32_t, 2> key{0, use.binding};
            const auto resource = std::find_if(program.resources.begin(), program.resources.end(), [&](const auto &r) {
                return r.set == key[0] && r.binding.binding == key[1];
            });
            if (resource == program.resources.end() ||
                std::find(entry->bindings.begin(), entry->bindings.end(), key) == entry->bindings.end())
                throw std::logic_error("Warp step resource is absent from " + std::string(name));
            const auto range = Range(use.slot);
            if (use.complete)
                result.completeWrites.push_back({static_cast<uint32_t>(result.bindings.size()), 0, range.byteSize});
            result.bindings.push_back({range, use.access});
            descriptors.push_back({use.binding, resource->binding.type, resource->binding.readOnlyStorage});
        }
        const ShaderModuleDesc shader{target == ShaderCodeTarget::SpirV ? ShaderSourceLanguage::SpirV
                                                                        : ShaderSourceLanguage::Wgsl,
                                      entry->code.data(), entry->code.size()};
        result.kernel = std::make_shared<ComputeKernel>(host, shader, entry->name, std::move(descriptors), 0,
                                                        ComputeKernel::PipelineCreation::FirstDispatch);
        return result;
    }

    void Add(std::string_view module, std::string_view name, uint32_t x, std::initializer_list<Use> uses,
             bool groups = false)
    {
        if (x != 0)
            dispatches.push_back(Dispatch(module, name, x, uses, groups));
    }

    const uint32_t worlds;
    std::vector<ComputeWorkSlot> slots;
    std::vector<ComputeWorkPort> ports;
    std::vector<ComputeWorkDispatch> dispatches;
    std::vector<InitialValue> initial;

  private:
    ComputeHost &host;
    ShaderCodeTarget target;
    std::unordered_map<std::string_view, ShaderTargetCode> programs;
};

ComputeDispatchDesc BindSnapshot(ComputeWorkDispatch command,
                                 const std::vector<std::shared_ptr<ComputeBuffer>> &buffers)
{
    ComputeDispatchDesc snapshot;
    snapshot.kernel = std::move(command.kernel);
    snapshot.groupCountX = command.groupCountX;
    snapshot.groupCountY = command.groupCountY;
    snapshot.proof.emplace();
    snapshot.proof->writes = std::move(command.completeWrites);
    for (const auto &binding : command.bindings) {
        if (binding.access == R)
            snapshot.proof->reads.push_back(
                {static_cast<uint32_t>(snapshot.buffers.size()), 0, binding.range.byteSize});
        snapshot.buffers.push_back(buffers[binding.range.slot]);
        snapshot.bufferAccesses.push_back(binding.access);
    }
    return snapshot;
}
} // namespace

PreparedWarpStep WarpStep::Build(rhi::ComputeHost &host, ShaderCodeTarget target, const WarpModel &model,
                                 uint32_t worlds, float timestep, const ModelVector &gravity)
{
    if (worlds == 0)
        throw std::invalid_argument("Warp step requires a world");

    const auto [nq, nv, nb, nj] = model.dimensions.state;
    const auto nm = model.dimensions.layout[1];
    const auto branches = model.dimensions.layout[2];
    const auto packedMass = model.dimensions.layout[3];
    const auto worldWords = [worlds](uint64_t words) { return words * worlds; };
    StepBuilder b(host, target, worlds);
    const auto dimensions = b.Seed("model", 8, &model.dimensions);
    const auto stepSizes = b.Seed("timestep", 1, &timestep, worlds);
    const auto clocks = b.Seed("time", 1, nullptr, worlds);
    const auto positions = b.Seed("qpos", nq, model.initialPositions.data(), worlds);
    const auto velocities = b.Seed("qvel", nv, nullptr, worlds);
    const auto applied = b.Seed("appliedForce", nv, nullptr, worlds);
    if (nv != 0) {
        const auto bodies = b.Shared("bodies", model.bodies);
        const auto joints = b.Shared("joints", model.joints);
        const auto dofs = b.Shared("dofs", model.dofs);
        const auto leaves = b.Shared("branches", model.branches);
        const auto mocap = nm == 0
                               ? b.Seed("kinematicPoses", 8, nullptr)
                               : b.Seed("kinematicPoses", uint64_t{8} * nm, model.initialKinematicPoses.data(), worlds);
        std::vector<ModelPose> initialBodies(nb);
        const auto pose = b.Seed("bodyPose", uint64_t{8} * nb, initialBodies.data(), worlds);
        const auto jointPose = b.Slot(worldWords(uint64_t{8} * nj));
        b.Add("position", "kinematics_branch", branches,
              {{0, dimensions, R},
               {1, bodies, R},
               {2, joints, R},
               {3, positions, R},
               {4, leaves, R},
               {5, mocap, R},
               {6, pose, RW},
               {7, jointPose, W, true}});
        const auto gravities = b.Seed("gravity", 4, &gravity, worlds);
        const auto zeroForce = b.Seed("zeroForce", nv, nullptr, worlds);
        const auto inertialFrame = b.Slot(worldWords(uint64_t{16} * nb));
        const auto subtreeCom = b.Seed("subtreeCom", uint64_t{4} * nb, nullptr, worlds);
        const auto inertia = b.Slot(worldWords(uint64_t{12} * nb));
        const auto centeredDof = b.Slot(worldWords(uint64_t{8} * nv));
        const auto composite = b.Slot(worldWords(uint64_t{12} * nb));
        const auto mass = b.Slot(worldWords(packedMass));
        const auto bodyVelocity = b.Slot(worldWords(uint64_t{8} * nb));
        const auto centeredDofDot = b.Slot(worldWords(uint64_t{8} * nv));
        const auto biasAcceleration = b.Slot(worldWords(uint64_t{8} * nb));
        const auto bodyForce = b.Slot(worldWords(uint64_t{8} * nb));
        const auto subtreeForce = b.Seed("subtreeForce", uint64_t{8} * nb, nullptr, worlds);
        const auto bias = b.Slot(worldWords(nv));
        const auto force = b.Slot(worldWords(nv));
        const auto solved = b.Seed("acceleration", nv, nullptr, worlds);
        const auto warmstart = b.Slot(worldWords(nv));

        b.Add("position", "body_inertial_frames", nb,
              {{0, dimensions, R}, {1, bodies, R}, {6, pose, R}, {8, inertialFrame, W, true}});
        b.Add("inertia", "subtree_com", nb,
              {{0, dimensions, R}, {1, bodies, R}, {2, inertialFrame, R}, {3, subtreeCom, W}});
        b.Add("inertia", "subtree_com_world", 1,
              {{0, dimensions, R}, {1, bodies, R}, {2, inertialFrame, R}, {3, subtreeCom, RW}});
        b.Add("inertia", "cinert", nb,
              {{0, dimensions, R}, {1, bodies, R}, {2, inertialFrame, R}, {3, subtreeCom, R}, {4, inertia, W, true}});
        b.Add("inertia", "cdof", nj,
              {{0, dimensions, R},
               {1, bodies, R},
               {3, subtreeCom, R},
               {5, joints, R},
               {6, pose, R},
               {7, jointPose, R},
               {8, centeredDof, W, true}});
        b.Add("inertia", "crb_subtree", nb,
              {{0, dimensions, R}, {1, bodies, R}, {4, inertia, R}, {9, composite, W, true}});
        b.Add("inertia", "assemble_mass", nv,
              {{0, dimensions, R}, {8, centeredDof, R}, {9, composite, R}, {10, dofs, R}, {11, mass, W, true}});
        b.Add("bias", "comvel_branch", branches,
              {{0, dimensions, R},
               {1, bodies, R},
               {2, joints, R},
               {3, leaves, R},
               {4, velocities, R},
               {5, centeredDof, R},
               {6, centeredDofDot, W, true},
               {7, bodyVelocity, W, true}});
        b.Add("bias", "bias_acceleration_branch", branches,
              {{0, dimensions, R},
               {1, bodies, R},
               {3, leaves, R},
               {4, velocities, R},
               {6, centeredDofDot, R},
               {8, gravities, R},
               {9, biasAcceleration, W, true}});
        b.Add("bias", "bias_force_body", nb,
              {{0, dimensions, R},
               {7, bodyVelocity, R},
               {9, biasAcceleration, R},
               {10, inertia, R},
               {11, bodyForce, W, true}});
        b.Add("bias", "bias_force_subtree", nb,
              {{0, dimensions, R}, {1, bodies, R}, {11, bodyForce, R}, {12, subtreeForce, W}});
        b.Add("bias", "bias_force_world", 1, {{0, dimensions, R}, {1, bodies, R}, {12, subtreeForce, RW}});
        b.Add("bias", "project_bias", nv,
              {{0, dimensions, R}, {5, centeredDof, R}, {12, subtreeForce, R}, {13, dofs, R}, {14, bias, W, true}});
        b.Add("acceleration", "assemble_smooth_force", nv,
              {{0, dimensions, R},
               {1, zeroForce, R},
               {2, bias, R},
               {3, zeroForce, R},
               {4, applied, R},
               {5, force, W, true}});

        const auto &factor = model.massFactors;
        const auto layout = b.Seed("factorLayout", 4, &factor.layout);
        const auto factors = b.Seed("factors", factor.layout[0], nullptr, worlds);
        const auto inverse = !factor.smallBlocks.empty() || factor.sparseBlockCount != 0
                                 ? b.Seed("inverseDiagonal", nv, nullptr, worlds)
                                 : 0;
        if (!factor.smallBlocks.empty()) {
            const auto small = b.Shared("smallMassBlocks", factor.smallBlocks);
            const auto count = static_cast<uint32_t>(factor.smallBlocks.size());
            b.Add("mass_solve", "factor_small_mass", count,
                  {{0, dimensions, R},
                   {1, layout, R},
                   {2, small, R},
                   {9, dofs, R},
                   {10, mass, R},
                   {11, factors, W},
                   {12, inverse, W}});
            b.Add("mass_solve", "solve_small_mass", count,
                  {{0, dimensions, R},
                   {1, layout, R},
                   {2, small, R},
                   {11, factors, R},
                   {12, inverse, R},
                   {13, force, R},
                   {14, solved, W}});
        }
        if (!factor.tiledBlocks.empty()) {
            const auto tiled = b.Shared("tiledMassBlocks", factor.tiledBlocks);
            const auto elements = b.Shared("massTileElements", factor.tileElements);
            const auto count = static_cast<uint32_t>(factor.tiledBlocks.size());
            b.Add(
                "mass_solve", "factor_tiled_mass", count,
                {{0, dimensions, R}, {1, layout, R}, {3, tiled, R}, {4, elements, R}, {10, mass, R}, {11, factors, W}},
                true);
            b.Add(
                "mass_solve", "solve_tiled_mass", count,
                {{0, dimensions, R}, {1, layout, R}, {3, tiled, R}, {11, factors, R}, {13, force, R}, {14, solved, W}},
                true);
        }
        if (factor.sparseBlockCount != 0) {
            const auto sparse = b.Shared("sparseMassTopology", factor.sparseTopology);
            b.Add("mass_solve", "factor_sparse_mass", factor.sparseBlockCount,
                  {{0, dimensions, R},
                   {1, layout, R},
                   {5, sparse, R},
                   {9, dofs, R},
                   {10, mass, R},
                   {11, factors, W},
                   {12, inverse, W}},
                  true);
            b.Add("mass_solve", "solve_sparse_mass", factor.sparseBlockCount,
                  {{0, dimensions, R},
                   {1, layout, R},
                   {5, sparse, R},
                   {9, dofs, R},
                   {11, factors, R},
                   {12, inverse, R},
                   {13, force, R},
                   {14, solved, W}},
                  true);
        }
        b.Add("euler", "advance_velocity", nv,
              {{0, dimensions, R},
               {2, stepSizes, R},
               {3, solved, R},
               {4, velocities, RW},
               {6, warmstart, W, true},
               {8, solved, R}});
        b.Add("euler", "advance_position", nj,
              {{0, dimensions, R}, {1, joints, R}, {2, stepSizes, R}, {4, velocities, R}, {5, positions, RW}});
    }
    b.Add("euler", "advance_time", 1, {{2, stepSizes, R}, {7, clocks, RW}});

    const auto snapshotSlot = b.Slot(worldWords(uint64_t{nq} + nv + 1), BufferUsageFlags::TransferSource);
    auto snapshot = b.Dispatch(
        "state_snapshot", "snapshot_generalized_state", std::max({1u, nq, nv}),
        {{0, dimensions, R}, {1, positions, R}, {2, velocities, R}, {3, clocks, R}, {4, snapshotSlot, W, true}});
    auto recipe =
        std::make_shared<rhi::ComputeWorkRecipe>(host, std::move(b.slots), std::move(b.ports), std::move(b.dispatches));
    auto work = recipe->Instantiate();
    if (!work)
        return {};

    std::vector<rhi::ComputeBufferUpdate> initial;
    initial.reserve(b.initial.size());
    const auto &buffers = work->GetBuffers();
    for (auto &upload : b.initial)
        initial.push_back({buffers[upload.slot], 0, std::move(upload.data)});
    auto initialize = rhi::PrepareComputeBatch(host, std::move(initial), {});
    auto step = std::unique_ptr<WarpStep>(
        new WarpStep(work, BindSnapshot(std::move(snapshot), buffers), positions, velocities, applied));
    return {std::move(step), std::move(initialize)};
}

WarpStep::WarpStep(std::shared_ptr<rhi::ComputeWorkInstance> work, rhi::ComputeDispatchDesc snapshot,
                   uint32_t positions, uint32_t velocities, uint32_t appliedForces)
    : m_work(std::move(work)), m_snapshot(std::move(snapshot)), m_positions(positions), m_velocities(velocities),
      m_appliedForces(appliedForces)
{
}

std::shared_ptr<rhi::PreparedComputeBatch> WarpStep::Prepare(bool publish,
                                                             std::vector<rhi::ComputeBufferUpdate> updates) const
{
    if (!publish)
        return m_work->PrepareBatch(std::move(updates));
    std::vector<rhi::ComputeDispatchDesc> dispatches;
    dispatches.reserve(m_work->GetRecipe().GetDispatches().size() + 1);
    m_work->AppendDispatches(dispatches);
    dispatches.push_back(m_snapshot);
    const auto &snapshot = m_snapshot.buffers.back();
    auto host = m_work->GetRecipe().GetHost();
    return rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches),
                                    {{snapshot, 0, snapshot->GetByteSize()}});
}

const std::shared_ptr<rhi::ComputeBuffer> &WarpStep::Positions() const noexcept
{
    return m_work->GetBuffers()[m_positions];
}

const std::shared_ptr<rhi::ComputeBuffer> &WarpStep::Velocities() const noexcept
{
    return m_work->GetBuffers()[m_velocities];
}

const std::shared_ptr<rhi::ComputeBuffer> &WarpStep::AppliedForces() const noexcept
{
    return m_work->GetBuffers()[m_appliedForces];
}

} // namespace infernux::physics::warp
