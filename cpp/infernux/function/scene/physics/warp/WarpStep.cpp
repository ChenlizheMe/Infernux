// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "WarpStep.h"
#include "../../Rigidbody.h"
#include "PhysicsRayIndex.h"
#include "WarpCollisionModel.h"
#include "WarpProgramSources.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <tuple>
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
static_assert(sizeof(WarpRay) == 48);
static_assert(sizeof(WarpRayHit) == 32);
static_assert(sizeof(WarpBodyCommand) == 48);
static_assert(sizeof(ModelHingeDrive) == 32);
static_assert(sizeof(WarpBodyState) == 80);
static_assert(sizeof(WarpContactReport) == 96);
static_assert(static_cast<uint32_t>(ForceMode::Force) == 0 && static_cast<uint32_t>(ForceMode::Acceleration) == 1 &&
              static_cast<uint32_t>(ForceMode::Impulse) == 2 && static_cast<uint32_t>(ForceMode::VelocityChange) == 3);

uint32_t FloatWord(float value)
{
    uint32_t word;
    std::memcpy(&word, &value, sizeof(word));
    return word;
}

enum ControlMotion : uint8_t
{
    Uncontrolled,
    DynamicControl,
    KinematicControl,
};

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

template <class Dispatch> void LinearGrid(Dispatch &dispatch)
{
    constexpr uint32_t groupsPerRow = 65535;
    dispatch.groupCountY = (dispatch.groupCountX - 1) / groupsPerRow + 1;
    dispatch.groupCountX = std::min(dispatch.groupCountX, groupsPerRow);
}

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
        const auto words = std::max(size_t{1}, table.size()) * (sizeof(T) / sizeof(uint32_t));
        return Seed(std::move(name), words, table.empty() ? nullptr : table.data());
    }

    ComputeWorkDispatch Dispatch(std::string_view module, std::string_view name, uint32_t x,
                                 std::initializer_list<Use> uses, bool groups = false)
    {
        return Bind(module, name, x, uses, groups);
    }

    template <class Uses>
    ComputeWorkDispatch Bind(std::string_view module, std::string_view name, uint32_t x, const Uses &uses,
                             bool groups = false)
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
        auto &kernel = kernels[&*entry];
        std::vector<ComputeBufferBinding> descriptors;
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
            if (!kernel)
                descriptors.push_back({use.binding, resource->binding.type, resource->binding.readOnlyStorage});
        }
        if (!kernel) {
            const ShaderModuleDesc shader{target == ShaderCodeTarget::SpirV ? ShaderSourceLanguage::SpirV
                                                                            : ShaderSourceLanguage::Wgsl,
                                          entry->code.data(), entry->code.size()};
            kernel = std::make_shared<ComputeKernel>(host, shader, entry->name, std::move(descriptors), 0,
                                                     ComputeKernel::PipelineCreation::FirstDispatch);
        }
        result.kernel = kernel;
        return result;
    }

    void Add(std::string_view module, std::string_view name, uint32_t x, std::initializer_list<Use> uses,
             bool groups = false)
    {
        if (x != 0)
            dispatches.push_back(Dispatch(module, name, x, uses, groups));
    }

    uint32_t Zero(std::string name, uint64_t words, BufferUsageFlags usage = BufferUsageFlags::None)
    {
        if (words > UINT32_MAX)
            throw std::length_error("Warp scratch exceeds the shader address space");
        const auto slot = Slot(words, usage);
        ports.push_back({std::move(name), Range(slot), true});
        auto clear = Dispatch("state_snapshot", "initialize_step_scratch",
                              static_cast<uint32_t>(slots[slot].allocation.elementCount), {{5, slot, W, true}});
        LinearGrid(clear);
        initialize.push_back(std::move(clear));
        return slot;
    }

    const uint32_t worlds;
    std::vector<ComputeWorkSlot> slots;
    std::vector<ComputeWorkPort> ports;
    std::vector<ComputeWorkDispatch> dispatches;
    std::vector<ComputeWorkDispatch> initialize;
    std::vector<InitialValue> initial;

  private:
    ComputeHost &host;
    ShaderCodeTarget target;
    std::unordered_map<std::string_view, ShaderTargetCode> programs;
    std::unordered_map<const ShaderEntryPointCode *, std::shared_ptr<ComputeKernel>> kernels;
};

struct ContactSlots
{
    uint32_t candidateCount;
    uint32_t counts;
    uint32_t records;
    uint32_t narrowphaseStatus;
};

struct GeometrySlots
{
    uint32_t geometries, bounds, frames, shapes, data;
};

GeometrySlots BindGeometry(StepBuilder &b, const WarpModel &model, const WarpCollisionModel &collision, bool rays)
{
    const auto count = model.dimensions.layout[0];
    return {b.Shared("geometries", model.geometries), b.Shared("localBounds", collision.bounds),
            count != 0 ? b.Slot(16 * uint64_t{count} * b.worlds) : b.Seed("geometryFrames", 16, nullptr),
            b.Shared("collisionShapes", collision.shapes),
            rays || !collision.convexData.empty() ? b.Shared("convexModel", collision.convexData) : NoModelIndex};
}

ContactSlots AddCollision(StepBuilder &b, const WarpModel &model, const WarpCollisionModel &collision,
                          uint32_t dimensions, const GeometrySlots &geometry, WarpCollisionCapacity capacity,
                          float minimumRestitutionSpeed)
{
    const auto ng = model.dimensions.layout[0];
    const auto geometryWords = uint64_t{ng} * b.worlds;
    const auto bounds = geometry.bounds, frames = geometry.frames, shapes = geometry.shapes, data = geometry.data;
    const auto rules = b.Shared("collisionPairRules", collision.pairRules);
    const auto projected = b.Slot(28 * geometryWords);
    const uint32_t sorts[] = {b.Slot(2 * geometryWords), b.Slot(2 * geometryWords)};
    const uint32_t scans[] = {b.Slot(geometryWords), b.Slot(geometryWords)};
    const auto pairCount = b.Slot(1, BufferUsageFlags::TransferSource);
    const auto pairs = b.Zero("collisionCandidates", 5 * uint64_t{capacity.candidates});
    const auto workers = ng;
    constexpr uint32_t boundsFilters = 15;
    constexpr uint32_t parentFilter = 16;
    const auto filters = boundsFilters | (collision.filterConnectedBodies ? parentFilter : 0);
    const ModelWords options{0, capacity.candidates, filters, workers};
    const auto emitSettings = b.Seed("sweepEmission", 4, &options);
    b.Add("broadphase", "sap_project", ng,
          {{0, dimensions, R}, {2, bounds, R}, {3, frames, R}, {4, projected, W, true}, {6, sorts[0], W, true}});
    std::vector<uint32_t> strides;
    uint32_t currentSort = 0;
    for (uint64_t width = 1; width < ng; width *= 2) {
        const auto stride = static_cast<uint32_t>(width);
        const ModelWords values{stride, 0, 0, 0};
        const auto settings = b.Seed("sweepStride" + std::to_string(stride), 4, &values);
        strides.push_back(settings);
        b.Add("broadphase", "sap_merge", ng,
              {{0, dimensions, R}, {1, settings, R}, {5, sorts[currentSort], R}, {6, sorts[1 - currentSort], W, true}});
        currentSort = 1 - currentSort;
    }
    b.Add("broadphase", "sap_find_ranges", ng,
          {{0, dimensions, R},
           {4, projected, R},
           {5, sorts[currentSort], R},
           {8, scans[0], W, true},
           {10, pairCount, W, true}});
    uint32_t currentScan = 0;
    for (const auto settings : strides) {
        b.Add("broadphase", "sap_scan_ranges", ng,
              {{0, dimensions, R}, {1, settings, R}, {7, scans[currentScan], R}, {8, scans[1 - currentScan], W, true}});
        currentScan = 1 - currentScan;
    }
    b.Add("broadphase", "sap_emit_pairs", workers,
          {{0, dimensions, R},
           {1, emitSettings, R},
           {4, projected, R},
           {5, sorts[currentSort], R},
           {7, scans[currentScan], R},
           {9, rules, R},
           {10, pairCount, RW},
           {11, pairs, W}});

    constexpr uint64_t workerLimit = 1024;
    const auto pairUpperBound = std::min(workerLimit, uint64_t{ng} * (ng - 1) / 2) * b.worlds;
    const bool advanced = !collision.convexData.empty();
    const auto convexWorkers =
        advanced ? static_cast<uint32_t>(std::min({uint64_t{capacity.candidates}, workerLimit, pairUpperBound})) : 0u;
    constexpr uint32_t convexManifolds = 3;
    const std::array<ModelWords, 2> narrowOptions{
        {{ng, capacity.candidates, capacity.contactsPerWorld, convexWorkers},
         {collision.convexPairs ? convexManifolds : 0u, FloatWord(minimumRestitutionSpeed), 0, 0}}};
    const auto settings = b.Seed("contactSettings", 8, &narrowOptions);
    const auto materials = b.Shared("collisionMaterials", collision.materials);
    const auto counts = b.Slot(b.worlds, BufferUsageFlags::TransferSource);
    const auto contacts =
        b.Zero("contactRecords", 40 * uint64_t{b.worlds} * capacity.contactsPerWorld, BufferUsageFlags::TransferSource);
    const auto arguments = b.Slot(8, BufferUsageFlags::Indirect);
    b.Add("primitive_contacts", "prepare_narrowphase", 1,
          {{0, settings, R}, {4, pairCount, R}, {7, counts, W, true}, {8, arguments, W, true}});
    auto narrow = b.Dispatch("primitive_contacts", "collide_primitives", capacity.candidates,
                             {{0, settings, R},
                              {1, frames, R},
                              {2, shapes, R},
                              {3, pairs, R},
                              {4, pairCount, R},
                              {5, materials, R},
                              {6, contacts, W},
                              {7, counts, RW}});
    LinearGrid(narrow);
    narrow.indirectArguments = ComputeWorkRange{arguments, 0, ComputeIndirectArguments::ByteSize};
    b.dispatches.push_back(std::move(narrow));

    const auto status = convexWorkers != 0 ? b.Slot(b.worlds) : b.Zero("narrowphaseStatus", b.worlds);
    if (convexWorkers != 0) {
        const auto &layout = collision.convexData[0];
        const auto polygonRecords = 5 * uint64_t{collision.convexData[1][3]};
        const auto candidateRecords = 2 * uint64_t{collision.convexData[3][2]};
        const auto workerRecords = 2 * uint64_t{layout[1]} + layout[2] + std::max(polygonRecords, candidateRecords);
        const auto scratch = b.Slot(4 * convexWorkers * workerRecords);
        const auto horizon = b.Slot(uint64_t{convexWorkers} * layout[3]);
        b.Add("convex_contacts", "clear_convex_status", 1, {{11, status, W, true}});
        const auto narrowphase = [&](std::string_view module, std::string_view name) {
            auto dispatch = b.Dispatch(module, name, convexWorkers,
                                       {{0, settings, R},
                                        {1, frames, R},
                                        {2, shapes, R},
                                        {3, pairs, R},
                                        {4, pairCount, R},
                                        {5, materials, R},
                                        {6, contacts, W},
                                        {7, counts, RW},
                                        {8, data, R},
                                        {9, scratch, W},
                                        {10, horizon, W},
                                        {11, status, RW}});
            LinearGrid(dispatch);
            dispatch.indirectArguments =
                ComputeWorkRange{arguments, sizeof(ModelWords), ComputeIndirectArguments::ByteSize};
            b.dispatches.push_back(std::move(dispatch));
        };
        if (collision.convexPairs)
            narrowphase("convex_contacts", "collide_convex");
        if (collision.heightfields)
            narrowphase("heightfield_contacts", "collide_heightfield");
        if (collision.triangleMeshes)
            narrowphase("triangle_mesh_contacts", "collide_triangle_mesh");
    }
    return {pairCount, counts, contacts, status};
}

struct RaySlots
{
    uint32_t inputs, hits;
};

struct BodyControlSlots
{
    uint32_t spans, commands, force, saved, prescribed, fixedSteps;
};

BodyControlSlots BindBodyControls(StepBuilder &b, const WarpModel &model, uint32_t capacity, bool multiple)
{
    return {b.Seed("bodyCommandSpans", 4 * (uint64_t{capacity} + 1), nullptr),
            b.Seed("bodyCommands", 12 * uint64_t{capacity}, nullptr),
            b.Zero("commandedForce", uint64_t{model.dimensions.state[1]} * b.worlds),
            b.Zero("savedMotions", 44 * uint64_t{capacity}),
            b.Zero("prescribedMotion", 12 * uint64_t{std::max(1u, model.dimensions.layout[1])} * b.worlds),
            b.Zero("acceptedFixedSteps", uint64_t{b.worlds} + uint32_t(multiple))};
}

RaySlots AddRayWork(StepBuilder &b, const WarpModel &model, const WarpCollisionModel &collision,
                    const GeometrySlots &geometry, WarpRayBatchLayout layout, std::vector<ComputeWorkDispatch> &work)
{
    const auto inputRows = layout.sharedRays ? 1u : b.worlds;
    const auto rayTotal = uint64_t{layout.raysPerWorld} * b.worlds;
    if (rayTotal > UINT32_MAX)
        throw std::length_error("Warp ray batch exceeds the shader address space");
    const auto inputs = b.Seed("rays", 12 * uint64_t{layout.raysPerWorld} * inputRows, nullptr);
    const auto hits = b.Slot(8 * rayTotal, BufferUsageFlags::TransferSource);
    const auto geometryCount = model.dimensions.layout[0];
    const ModelWords settings{geometryCount, layout.raysPerWorld, inputRows, 0};
    const auto query = b.Seed("raySettings", 4, &settings);
    const auto index = BuildWarpRayIndex(model, collision);
    const auto nodes = b.Shared("rayIndex", index.records);
    const auto bounds = b.Zero("rayBounds", 8 * uint64_t{std::max(1u, index.nodeCount)} * b.worlds);
    const auto metadata = b.Shared("rayGeometry", collision.rayGeometry);
    const auto refit = [&](uint32_t first, uint32_t count, bool complete) {
        const ModelWords options{geometryCount, first, 0, 0};
        const auto slot = b.Seed(first == 0 ? "rayRefitSubtrees" : "rayRefitRoot", 4, &options);
        work.push_back(b.Dispatch(
            "ray_bvh_refit", "refit_ray_bvh", count,
            {{0, slot, R}, {1, geometry.bounds, R}, {2, nodes, R}, {3, bounds, RW, complete}, {4, geometry.frames, R}},
            true));
    };
    if (index.refitSubtrees != 0) {
        refit(0, index.refitSubtrees, !index.refitRoot);
        if (index.refitRoot)
            refit(index.refitSubtrees, 1, true);
    }
    work.push_back(b.Dispatch("ray_queries", "raycast_worlds", layout.raysPerWorld,
                              {{0, query, R},
                               {1, inputs, R},
                               {2, geometry.frames, R},
                               {3, geometry.shapes, R},
                               {4, metadata, R},
                               {5, geometry.data, R},
                               {7, hits, W, true},
                               {8, nodes, R},
                               {9, bounds, R}}));
    return {inputs, hits};
}

struct ContactResponse
{
    uint32_t acceleration;
    uint32_t status;
    uint32_t topology;
    uint32_t reportRecords = NoModelIndex;
    std::optional<ComputeWorkDispatch> capture;
    uint32_t hingeDrives = NoModelIndex;
};

struct ReferenceWeights
{
    uint32_t dimensions, dofs, mass, centeredDof, inertiaFrames, subtreeCom, topology;
};

void InitializeWeights(StepBuilder &b, const WarpModel &model, const ReferenceWeights &ref)
{
    const auto factorization = BuildWarpReferenceMassFactors(model);
    const auto layout = b.Seed("referenceFactorLayout", 4, &factorization.layout);
    const auto sparse = b.Shared("referenceSparseMassTopology", factorization.sparseTopology);
    const auto packed = model.dimensions.layout[3];
    const auto factors = b.Slot(packed);
    const auto inverseDiagonal = b.Slot(model.dimensions.state[1]);
    const auto inverse = b.Slot(packed);
    const auto add = [&](ComputeWorkDispatch dispatch) {
        dispatch.groupCountY = 1;
        b.initialize.push_back(std::move(dispatch));
    };
    auto factor = b.Dispatch("mass_solve", "factor_sparse_mass", factorization.sparseBlockCount,
                             {{0, ref.dimensions, R},
                              {1, layout, R},
                              {5, sparse, R},
                              {9, ref.dofs, R},
                              {10, ref.mass, R},
                              {11, factors, W, true},
                              {12, inverseDiagonal, W, true}},
                             true);
    factor.bindings[4].range.byteSize /= b.worlds;
    add(std::move(factor));
    add(b.Dispatch(
        "mass_solve", "reference_mass_inverse", factorization.sparseBlockCount,
        {{5, sparse, R}, {9, ref.dofs, R}, {11, factors, R}, {12, inverseDiagonal, R}, {18, inverse, W, true}}, true));
    auto weight = b.Dispatch("mass_solve", "reference_contact_weights", model.dimensions.state[2],
                             {{0, ref.dimensions, R},
                              {9, ref.dofs, R},
                              {18, inverse, R},
                              {19, ref.topology, RW},
                              {20, ref.centeredDof, R},
                              {21, ref.inertiaFrames, R},
                              {22, ref.subtreeCom, R}});
    for (const uint32_t binding : {4u, 5u, 6u})
        weight.bindings[binding].range.byteSize /= b.worlds;
    add(std::move(weight));
}

struct ContactDynamics
{
    uint32_t dofs;
    uint32_t bodyModel;
    uint32_t bodyPose;
    uint32_t centeredDof;
    uint32_t centeredDofDerivative;
    uint32_t subtreeCom;
    uint32_t velocities;
    uint32_t mass;
    uint32_t smoothForce;
    uint32_t smoothAcceleration;
    uint32_t warmstart;
    uint32_t motionModel;
    uint32_t fixedSteps;
    uint32_t prescribedMotion;
    uint32_t bodyVelocity;
};

ContactResponse AddContactResponse(StepBuilder &b, const WarpModel &model, const ContactSlots &stream,
                                   WarpCollisionCapacity capacity, const WarpContactSolver &solver, float timestep,
                                   const ContactDynamics &dyn, std::vector<ComputeWorkDispatch> precondition,
                                   bool contactReports, uint32_t substeps)
{
    const auto nv = model.dimensions.state[1];
    const auto nb = model.dimensions.state[2];
    const auto ng = model.dimensions.layout[0];
    if (solver.rowsPerWorld == 0 || solver.columnsPerWorld == 0 || solver.iterations == 0 || solver.lineIterations == 0)
        throw std::invalid_argument("Warp contact response requires nonzero solver limits");

    const auto rows = solver.rowsPerWorld;
    const auto columns = solver.columnsPerWorld;
    const auto contacts = capacity.contactsPerWorld;
    constexpr uint32_t ellipticCone = 1;
    std::vector<ModelWords> config(b.worlds + 2, {FloatWord(timestep), FloatWord(1.0f), FloatWord(solver.tolerance),
                                                  FloatWord(solver.lineTolerance)});
    config[0] = {ng, contacts, rows, columns};
    config[1] = {nb, nv, contacts, ellipticCone};
    const auto settings = b.Shared("constraintSettings", config);

    std::vector<ModelWords> topology(uint64_t{ng} + nb);
    for (uint32_t geom = 0; geom < ng; ++geom) {
        const auto body = model.geometries[geom].indices[0];
        topology[geom][0] = body;
        topology[geom][1] = model.bodies[model.bodies[body].subtree[0]].hierarchy[3];
    }
    topology[ng] = {NoModelIndex, 0, 0, nb};
    for (uint32_t body = 1; body < nb; ++body) {
        const auto &record = model.bodies[body];
        const auto lastDof =
            record.dofs[1] != 0 ? record.dofs[0] + record.dofs[1] - 1 : topology[ng + record.hierarchy[0]][0];
        topology[ng + body] = {lastDof, record.subtree[0], 0, record.subtree[1]};
    }
    const auto contactTopology = b.Shared("contactTopology", topology);
    const auto across = [count = b.worlds](uint64_t words) { return words * count; };
    const auto counters = b.Slot(across(2));
    const auto reservations = b.Zero("contactReservations", across(4 * uint64_t{contacts}));
    const auto rowStore = b.Zero("constraintRows", across(12 * uint64_t{rows}));
    const auto columnStore = b.Zero("constraintColumns", across(2 * uint64_t{columns}));
    const auto residual = b.Zero("constraintResidual", across(rows));
    const auto evaluation = b.Zero("constraintEvaluation", across(3 * uint64_t{rows}));
    const auto acceleration = b.Zero("contactAcceleration", across(nv));

    b.Add("contact_constraints", "clear_constraint_counts", 1, {{5, counters, W, true}});
    b.Add("contact_constraints", "reserve_contact_rows", contacts,
          {{0, settings, R},
           {1, stream.records, R},
           {2, stream.counts, R},
           {3, contactTopology, R},
           {4, dyn.dofs, R},
           {5, counters, RW},
           {6, reservations, W},
           {8, columnStore, W}});
    b.Add("contact_constraints", "project_contact_rows", contacts,
          {{0, settings, R},
           {1, stream.records, R},
           {2, stream.counts, R},
           {3, contactTopology, R},
           {6, reservations, R},
           {8, columnStore, RW},
           {9, dyn.centeredDof, R},
           {10, dyn.subtreeCom, R}});
    if (dyn.prescribedMotion == NoModelIndex)
        b.Add("contact_constraints", "finish_contact_rows", contacts,
              {{0, settings, R},
               {1, stream.records, R},
               {2, stream.counts, R},
               {3, contactTopology, R},
               {6, reservations, R},
               {7, rowStore, W},
               {8, columnStore, R},
               {11, dyn.velocities, R}});
    else
        b.Add("contact_constraints", "finish_controlled_contact_rows", contacts,
              {{0, settings, R},
               {1, stream.records, R},
               {2, stream.counts, R},
               {3, contactTopology, R},
               {6, reservations, R},
               {7, rowStore, W},
               {8, columnStore, R},
               {11, dyn.velocities, R},
               {17, dyn.motionModel, R},
               {18, dyn.prescribedMotion, R},
               {19, dyn.fixedSteps, R}});
    uint32_t hingeDrives = NoModelIndex;
    if (!model.sceneJoints.empty()) {
        const auto jointCount = model.sceneJoints[0][0];
        const auto modelRows = b.Shared("sceneJoints", model.sceneJoints);
        const auto &drives = model.initialHingeDrives;
        const auto driveWords = sizeof(ModelHingeDrive) / sizeof(uint32_t);
        hingeDrives = drives.empty() ? b.Seed("sceneHingeDrives", driveWords, nullptr)
                                     : b.Seed("sceneHingeDrives", driveWords * drives.size(), drives.data(), b.worlds);
        if (dyn.prescribedMotion != NoModelIndex)
            b.Add("spatial_constraints", "write_scene_kinematic_velocities", nb,
                  {{4, dyn.subtreeCom, R},
                   {12, dyn.bodyModel, R},
                   {13, dyn.prescribedMotion, R},
                   {14, dyn.fixedSteps, R},
                   {15, dyn.bodyVelocity, W},
                   {16, dyn.motionModel, R}});
        b.Add("spatial_constraints", "emit_scene_joint_rows", jointCount,
              {{0, settings, R},
               {1, modelRows, R},
               {3, dyn.bodyPose, R},
               {4, dyn.subtreeCom, R},
               {5, counters, RW},
               {6, dyn.centeredDof, R},
               {7, rowStore, W},
               {8, columnStore, W},
               {9, dyn.centeredDofDerivative, R},
               {10, dyn.bodyVelocity, R},
               {11, dyn.velocities, R},
               {17, hingeDrives, R}});
    }
    b.Add("contact_constraints", "seed_contact_acceleration", nv,
          {{0, settings, R},
           {5, counters, R},
           {12, dyn.smoothAcceleration, R},
           {13, dyn.warmstart, R},
           {14, acceleration, W}});
    auto objective = b.Dispatch("contact_constraints", "evaluate_contact_objective", rows,
                                {{0, settings, R},
                                 {1, stream.records, R},
                                 {5, counters, R},
                                 {7, rowStore, R},
                                 {15, residual, R},
                                 {16, evaluation, W}});

    std::array<ModelWords, 3> options{{{solver.iterations, solver.lineIterations, model.dimensions.layout[3], 0},
                                       {capacity.candidates, substeps, 0, 0},
                                       {}}};
    const auto indirectSlot = b.Slot(7 * 4, BufferUsageFlags::Indirect);
    enum DispatchExtent : uint32_t
    {
        Worlds,
        Dofs,
        Rows,
        RowsOrDofs,
        FirstMassBlock
    };
    const auto grid = [indirectSlot](uint32_t item) {
        return ComputeWorkRange{indirectSlot, uint64_t{item} * sizeof(ModelWords), ComputeIndirectArguments::ByteSize};
    };
    const auto indirectLast = [&](DispatchExtent extent) { b.dispatches.back().indirectArguments = grid(extent); };
    objective.indirectArguments = grid(Rows);
    const auto massTopology = b.Shared("massAdjacency", model.massAdjacency);
    const auto worldState = b.Slot(across(12));
    const auto gradient = b.Zero("contactGradient", across(nv));
    const auto preconditioned = b.Zero("preconditionedContactGradient", across(nv));
    const auto force = b.Zero("contactForce", across(nv));
    const auto history = b.Zero("contactSearchHistory", across(4 * uint64_t{nv}));
    const auto products = b.Zero("contactMassProducts", across(2 * uint64_t{nv}));
    const auto ray = b.Zero("contactSearchRay", across(rows));
    const bool multiple = substeps > 1;
    const auto status = multiple ? b.Zero("fixedStepStatus", across(uint64_t{1} + (contactReports ? substeps : 0)),
                                          contactReports ? BufferUsageFlags::TransferSource : BufferUsageFlags::None)
                                 : b.Slot(b.worlds);
    for (uint32_t index = 0; index < precondition.size(); ++index) {
        auto &solve = precondition[index];
        options[2][index] = solve.groupCountX;
        solve.bindings[solve.bindings.size() - 2].range = b.Range(gradient);
        solve.bindings.back().range = b.Range(preconditioned);
        solve.indirectArguments = grid(FirstMassBlock + index);
    }
    const auto opt = b.Seed("contactSolverOptions", 12, &options);
    b.Add("contact_cg", "begin_contact_cg", 1,
          {{0, settings, R},
           {1, opt, R},
           {2, counters, R},
           {10, dyn.mass, R},
           {11, massTopology, R},
           {16, worldState, W, true}});
    b.Add("contact_cg", "prepare_contact_solve_dispatches", 1,
          {{0, settings, R}, {1, opt, R}, {16, worldState, R}, {28, indirectSlot, W, true}});
    b.dispatches.back().groupCountY = 1;
    b.Add("contact_constraints", "contact_residual", rows,
          {{0, settings, R},
           {5, counters, R},
           {7, rowStore, R},
           {8, columnStore, R},
           {14, acceleration, R},
           {15, residual, W}});
    indirectLast(Rows);
    b.dispatches.push_back(objective);
    b.Add("contact_cg", "cg_initial_mass_product", nv,
          {{0, settings, R},
           {1, opt, R},
           {8, acceleration, R},
           {10, dyn.mass, R},
           {11, massTopology, R},
           {16, worldState, R},
           {17, products, W}});
    indirectLast(Dofs);

    // A fixed recipe records the upper bound. Each world's progress controls its device work.
    for (uint32_t iteration = 0; iteration <= solver.iterations; ++iteration) {
        b.Add("contact_cg", "cg_project_iteration", 1,
              {{0, settings, R},
               {3, rowStore, R},
               {4, columnStore, R},
               {7, evaluation, R},
               {14, force, RW},
               {16, worldState, R}});
        indirectLast(Worlds);
        b.Add("contact_cg", "cg_gradient", nv,
              {{0, settings, R},
               {9, dyn.smoothForce, R},
               {12, gradient, W},
               {14, force, R},
               {16, worldState, R},
               {17, products, R}});
        indirectLast(Dofs);
        b.dispatches.insert(b.dispatches.end(), precondition.begin(), precondition.end());
        b.Add("contact_cg", "cg_search_direction", 1,
              {{0, settings, R},
               {1, opt, R},
               {12, gradient, R},
               {13, preconditioned, R},
               {15, history, RW},
               {16, worldState, RW}});
        indirectLast(Worlds);
        if (iteration == solver.iterations)
            break;
        b.Add("contact_cg", "cg_ray_mass_and_gauss", 1,
              {{0, settings, R},
               {1, opt, R},
               {9, dyn.smoothForce, R},
               {10, dyn.mass, R},
               {11, massTopology, R},
               {15, history, R},
               {16, worldState, RW},
               {17, products, RW}});
        indirectLast(Worlds);
        b.Add("contact_cg", "cg_ray_constraint_product", rows,
              {{0, settings, R},
               {3, rowStore, R},
               {4, columnStore, R},
               {15, history, R},
               {16, worldState, R},
               {18, ray, W}});
        indirectLast(Rows);
        b.Add("contact_cg", "cg_line_search", 1,
              {{0, settings, R},
               {1, opt, R},
               {3, rowStore, R},
               {5, stream.records, R},
               {6, residual, R},
               {16, worldState, RW},
               {18, ray, R}});
        indirectLast(Worlds);
        b.Add("contact_cg", "cg_apply_step", std::max(nv, rows),
              {{0, settings, R},
               {6, residual, RW},
               {8, acceleration, RW},
               {15, history, R},
               {16, worldState, R},
               {17, products, RW},
               {18, ray, R}});
        indirectLast(RowsOrDofs);
        b.dispatches.push_back(objective);
    }
    b.Add("contact_cg", "finish_contact_step", 1,
          {{0, settings, R},
           {1, opt, R},
           {16, worldState, R},
           {25, stream.candidateCount, R},
           {26, stream.counts, R},
           {27, status, multiple ? RW : W, !multiple},
           {30, stream.narrowphaseStatus, R}});
    ContactResponse response{acceleration, status, contactTopology, NoModelIndex, {}, hingeDrives};
    if (contactReports) {
        const auto output =
            b.Zero("contactReports", across(24 * uint64_t{contacts} * substeps), BufferUsageFlags::TransferSource);
        response.reportRecords = output;
        const bool prescribed = dyn.prescribedMotion != NoModelIndex;
        std::vector<Use> uses{{0, settings, R},
                              {1, stream.records, R},
                              {2, stream.counts, R},
                              {3, contactTopology, R},
                              {4, reservations, R},
                              {5, evaluation, R},
                              {6, dyn.bodyVelocity, R},
                              {7, dyn.subtreeCom, R},
                              {8, status, multiple ? RW : R},
                              {9, output, W}};
        if (prescribed)
            uses.push_back({11, dyn.prescribedMotion, R});
        if (prescribed || multiple)
            uses.push_back({12, dyn.fixedSteps, R});
        const auto *entry = multiple ? (prescribed ? "capture_controlled_contact_substep" : "capture_contact_substep")
                                     : (prescribed ? "capture_controlled_contact_reports" : "capture_contact_reports");
        response.capture = b.Bind("contact_reports", entry, contacts, uses);
    }
    return response;
}

ComputeDispatchDesc BindDispatch(ComputeWorkDispatch command,
                                 const std::vector<std::shared_ptr<ComputeBuffer>> &buffers)
{
    ComputeDispatchDesc snapshot;
    snapshot.kernel = std::move(command.kernel);
    snapshot.groupCountX = command.groupCountX;
    snapshot.groupCountY = command.groupCountY;
    snapshot.proof.emplace();
    snapshot.proof->writes = std::move(command.completeWrites);
    for (const auto &binding : command.bindings) {
        if (binding.access != W)
            snapshot.proof->reads.push_back(
                {static_cast<uint32_t>(snapshot.buffers.size()), 0, binding.range.byteSize});
        snapshot.buffers.push_back(buffers[binding.range.slot]);
        snapshot.bufferAccesses.push_back(binding.access);
    }
    return snapshot;
}
} // namespace

PreparedWarpStep WarpStep::Build(rhi::ComputeHost &host, ShaderCodeTarget target, const WarpModel &model,
                                 uint32_t worlds, float timestep, const ModelVector &gravity,
                                 const WarpCollisionModel *collision, WarpCollisionCapacity capacity,
                                 const WarpContactSolver *solver, WarpRayBatchLayout rayLayout,
                                 uint32_t bodyCommandCapacity, bool contactReports, uint32_t substeps)
{
    if (worlds == 0)
        throw std::invalid_argument("Warp step requires a world");
    if (substeps == 0 || (substeps > 1 && bodyCommandCapacity == 0))
        throw std::invalid_argument("Warp fixed substeps require a controlled step layout");

    const auto [nq, nv, nb, nj] = model.dimensions.state;
    if (!model.initialVelocities.empty() && model.initialVelocities.size() != nv)
        throw std::invalid_argument("Warp initial velocities must match the model degrees of freedom");
    const auto nm = model.dimensions.layout[1];
    const auto branches = model.dimensions.layout[2];
    const auto packedMass = model.dimensions.layout[3];
    const auto worldWords = [worlds](uint64_t words) { return words * worlds; };
    StepBuilder b(host, target, worlds);
    const auto dimensions = b.Seed("model", 8, &model.dimensions);
    const bool multiple = substeps > 1;
    const float substepDt = timestep / static_cast<float>(substeps);
    const auto stepSizes = b.Seed("timestep", 1, &substepDt, worlds);
    const auto fixedSizes = multiple ? b.Seed("fixedTimestep", 1, &timestep, worlds) : stepSizes;
    const auto clocks = b.Seed("time", 1, nullptr, worlds);
    const auto positions = b.Seed("qpos", nq, model.initialPositions.data(), worlds);
    const auto velocities =
        b.Seed("qvel", nv, model.initialVelocities.empty() ? nullptr : model.initialVelocities.data(), worlds);
    const auto applied = b.Seed("appliedForce", nv, nullptr, worlds);
    BodyControls controls;
    std::optional<BodyControlSlots> controlSlots;
    if (bodyCommandCapacity != 0) {
        controls.capacity = bodyCommandCapacity;
        controls.worlds = worlds;
        controls.timestep = FloatWord(timestep);
        controls.motion.resize(nb, Uncontrolled);
        for (uint32_t index = 1; index < nb; ++index) {
            const auto &body = model.bodies[index];
            if (body.hierarchy[0] != 0 ||
                (body.dofs[1] != 0 && (body.hierarchy[2] != 1 || model.joints[body.hierarchy[1]].indices[0] !=
                                                                     static_cast<uint32_t>(ModelJointType::Free))))
                throw std::invalid_argument("Warp body commands require independent rigid body rows");
            if (body.dofs[1] != 0)
                controls.motion[index] = DynamicControl;
            if (body.hierarchy[3] != NoModelIndex)
                controls.motion[index] = KinematicControl;
        }
        controlSlots = BindBodyControls(b, model, bodyCommandCapacity, multiple);
    }
    const bool detectContacts = collision && model.dimensions.layout[0] != 0;
    const bool traceRays = rayLayout.raysPerWorld != 0;
    if (traceRays && model.dimensions.layout[0] != 0 && !collision)
        throw std::invalid_argument("Warp rays require the compiled collision geometry");
    std::vector<ComputeWorkDispatch> rayDispatches;
    std::optional<RaySlots> raySlots;
    std::optional<ContactSlots> contactSlots;
    std::optional<ContactResponse> response;
    uint32_t warmstart = NoModelIndex;
    std::optional<ReferenceWeights> reference;
    const auto kinematicsPass = b.dispatches.size();
    struct
    {
        uint32_t bodies, joints, leaves, pose, jointPose, mocap;
    } kinematics{};
    if (nv != 0 || detectContacts || controlSlots) {
        auto &[bodies, joints, leaves, pose, jointPose, mocap] = kinematics;
        bodies = b.Shared("bodies", model.bodies);
        joints = b.Shared("joints", model.joints);
        leaves = b.Shared("branches", model.branches);
        mocap = nm == 0 ? b.Seed("kinematicPoses", 8, nullptr)
                        : b.Seed("kinematicPoses", uint64_t{8} * nm, model.initialKinematicPoses.data(), worlds);
        std::vector<ModelPose> initialBodies(nb);
        pose = b.Seed("bodyPose", uint64_t{8} * nb, initialBodies.data(), worlds);
        jointPose = b.Slot(worldWords(uint64_t{8} * nj));
        b.Add("position", "kinematics_branch", branches,
              {{0, dimensions, R},
               {1, bodies, R},
               {2, joints, R},
               {3, positions, R},
               {4, leaves, R},
               {5, mocap, R},
               {6, pose, RW},
               {7, jointPose, W, nj != 0}});
        if (detectContacts) {
            if (collision->bounds.size() != model.dimensions.layout[0] || capacity.candidates == 0 ||
                capacity.contactsPerWorld == 0)
                throw std::invalid_argument("Warp collision requires geometry rows and nonzero stream capacities");
            const auto geometry = BindGeometry(b, model, *collision, traceRays);
            b.Add("position", "geom_local_to_global", model.dimensions.layout[0],
                  {{0, dimensions, R}, {6, pose, R}, {9, geometry.geometries, R}, {10, geometry.frames, W, true}});
            if (traceRays) {
                if (branches != 0)
                    rayDispatches.push_back(b.dispatches[kinematicsPass]);
                rayDispatches.push_back(b.dispatches.back());
                raySlots = AddRayWork(b, model, *collision, geometry, rayLayout, rayDispatches);
            }
            contactSlots = AddCollision(b, model, *collision, dimensions, geometry, capacity,
                                        solver ? solver->minimumRestitutionSpeed : 0.0f);
        }
    }
    if (traceRays && !detectContacts) {
        const WarpCollisionModel empty;
        const auto geometry = BindGeometry(b, model, empty, true);
        raySlots = AddRayWork(b, model, empty, geometry, rayLayout, rayDispatches);
    }
    if (nv != 0) {
        const auto &[bodies, joints, leaves, pose, jointPose, mocap] = kinematics;
        const auto dofs = b.Shared("dofs", model.dofs);
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
        warmstart = contactSlots && solver ? b.Zero("warmstartAcceleration", worldWords(nv)) : b.Slot(worldWords(nv));

        const auto beginInertia = b.dispatches.size();
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
        const auto endInertia = b.dispatches.size();
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
        auto passiveForce = zeroForce;
        if (std::any_of(model.bodies.begin(), model.bodies.end(),
                        [](const auto &body) { return body.subtreeMass[1] != 0.0f; })) {
            passiveForce = b.Slot(worldWords(nv));
            b.Add("wrench", "project_passive_gravity_compensation", nv,
                  {{0, dimensions, R},
                   {1, bodies, R},
                   {2, inertialFrame, R},
                   {3, subtreeCom, R},
                   {4, gravities, R},
                   {7, dofs, R},
                   {8, centeredDof, R},
                   {12, passiveForce, W, true}});
        }
        b.Add("acceleration", "assemble_smooth_force", nv,
              {{0, dimensions, R},
               {1, passiveForce, R},
               {2, bias, R},
               {3, controlSlots ? controlSlots->force : zeroForce, R},
               {4, applied, R},
               {5, force, W, true}});

        const auto &factor = model.massFactors;
        const auto layout = b.Seed("factorLayout", 4, &factor.layout);
        const auto factors = b.Seed("factors", factor.layout[0], nullptr, worlds);
        const auto inverse = !factor.smallBlocks.empty() || factor.sparseBlockCount != 0
                                 ? b.Seed("inverseDiagonal", nv, nullptr, worlds)
                                 : 0;
        std::vector<ComputeWorkDispatch> massSolves;
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
            massSolves.push_back(b.Dispatch("mass_solve", "solve_small_mass", count,
                                            {{0, dimensions, R},
                                             {1, layout, R},
                                             {2, small, R},
                                             {11, factors, R},
                                             {12, inverse, R},
                                             {13, force, R},
                                             {14, solved, W}}));
        }
        if (!factor.tiledBlocks.empty()) {
            const auto tiled = b.Shared("tiledMassBlocks", factor.tiledBlocks);
            const auto elements = b.Shared("massTileElements", factor.tileElements);
            const auto count = static_cast<uint32_t>(factor.tiledBlocks.size());
            b.Add(
                "mass_solve", "factor_tiled_mass", count,
                {{0, dimensions, R}, {1, layout, R}, {3, tiled, R}, {4, elements, R}, {10, mass, R}, {11, factors, W}},
                true);
            massSolves.push_back(b.Dispatch(
                "mass_solve", "solve_tiled_mass", count,
                {{0, dimensions, R}, {1, layout, R}, {3, tiled, R}, {11, factors, R}, {13, force, R}, {14, solved, W}},
                true));
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
            massSolves.push_back(b.Dispatch("mass_solve", "solve_sparse_mass", factor.sparseBlockCount,
                                            {{0, dimensions, R},
                                             {1, layout, R},
                                             {5, sparse, R},
                                             {9, dofs, R},
                                             {11, factors, R},
                                             {12, inverse, R},
                                             {13, force, R},
                                             {14, solved, W}},
                                            true));
        }
        b.dispatches.insert(b.dispatches.end(), massSolves.begin(), massSolves.end());
        if (contactSlots && solver) {
            const ContactDynamics dynamics{dofs,
                                           bodies,
                                           pose,
                                           centeredDof,
                                           centeredDofDot,
                                           subtreeCom,
                                           velocities,
                                           mass,
                                           force,
                                           solved,
                                           warmstart,
                                           dimensions,
                                           controlSlots ? controlSlots->fixedSteps : NoModelIndex,
                                           controlSlots && nm != 0 ? controlSlots->prescribed : NoModelIndex,
                                           bodyVelocity};
            response = AddContactResponse(b, model, *contactSlots, capacity, *solver, substepDt, dynamics,
                                          std::move(massSolves), contactReports, substeps);
            reference = {dimensions, dofs, mass, centeredDof, inertialFrame, subtreeCom, response->topology};
            const std::array singleWorld{
                positions, pose,        jointPose, inertialFrame, subtreeCom,
                inertia,   centeredDof, composite, mass,          nm == 0 ? NoModelIndex : mocap};
            const auto initializeFrom = [&](size_t pass) {
                auto dispatch = b.dispatches[pass];
                dispatch.groupCountY = 1;
                const auto divide = [&](const auto &binding) {
                    return std::find(singleWorld.begin(), singleWorld.end(), binding.range.slot) != singleWorld.end();
                };
                for (auto &write : dispatch.completeWrites)
                    if (divide(dispatch.bindings[write.binding]))
                        write.byteSize /= worlds;
                for (auto &binding : dispatch.bindings)
                    if (divide(binding))
                        binding.range.byteSize /= worlds;
                b.initialize.push_back(std::move(dispatch));
            };
            initializeFrom(kinematicsPass);
            for (size_t pass = beginInertia; pass < endInertia; ++pass)
                initializeFrom(pass);
            b.Add("euler", "advance_contact_velocity", nv,
                  {{0, dimensions, R},
                   {2, stepSizes, R},
                   {3, response->acceleration, R},
                   {4, velocities, RW},
                   {6, warmstart, W},
                   {8, response->acceleration, R},
                   {15, response->status, R}});
            b.Add("euler", "advance_contact_position", nj,
                  {{0, dimensions, R},
                   {1, joints, R},
                   {2, stepSizes, R},
                   {4, velocities, R},
                   {5, positions, RW},
                   {15, response->status, R}});
        } else {
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
    }
    std::vector<Use> clockUses{{2, fixedSizes, R}, {7, clocks, RW}};
    if (response)
        clockUses.push_back({15, response->status, R});
    if (controlSlots)
        clockUses.push_back({16, controlSlots->fixedSteps, RW});
    const auto *clockEntry = response ? (controlSlots ? "advance_contact_controlled_time" : "advance_contact_time")
                                      : (controlSlots ? "advance_controlled_time" : "advance_time");
    auto clock = b.Bind("euler", clockEntry, 1, clockUses);
    if (!multiple)
        b.dispatches.push_back(std::move(clock));

    std::vector<ComputeWorkDispatch> begin;
    ComputeWorkDispatch advance, restore, drive;
    if (multiple) {
        const auto place = controlSlots->fixedSteps;
        begin.push_back(b.Dispatch("fixed_substeps", "begin_substeps", 1, {{5, place, RW}}));
        begin.back().groupCountY = 1;
        advance = b.Dispatch("fixed_substeps", "advance_substep", 1, {{5, place, RW}});
        advance.groupCountY = 1;
    }
    if (multiple && response) {
        const auto width = uint64_t{nq} + 2 * uint64_t{nv};
        if (worldWords(width) > UINT32_MAX)
            throw std::length_error("Warp fixed step checkpoint exceeds the device address range");
        const auto backup = b.Slot(worldWords(width));
        const auto rows = static_cast<uint32_t>(width);
        begin.push_back(b.Dispatch("fixed_substeps", "checkpoint_fixed_step", rows,
                                   {{0, dimensions, R},
                                    {1, positions, R},
                                    {2, velocities, R},
                                    {3, warmstart, R},
                                    {4, backup, W, true},
                                    {6, response->status, W}}));
        restore = b.Dispatch("fixed_substeps", "restore_rejected_fixed_step", rows,
                             {{0, dimensions, R},
                              {1, positions, W},
                              {2, velocities, W},
                              {3, warmstart, W},
                              {4, backup, R},
                              {6, response->status, R}});
    }

    std::vector<ComputeWorkDispatch> bodyPublication;
    ComputeWorkDispatch applyCommands, finishCommands;
    uint32_t states = NoModelIndex;
    if (controlSlots) {
        const auto &[spans, commands, force, saved, prescribed, fixedSteps] = *controlSlots;
        const auto &[bodies, joints, leaves, pose, jointPose, mocap] = kinematics;
        applyCommands = b.Dispatch("body_controls", "apply_body_commands", bodyCommandCapacity,
                                   {{0, dimensions, R},
                                    {1, bodies, R},
                                    {2, joints, R},
                                    {3, spans, R},
                                    {4, commands, R},
                                    {5, positions, RW},
                                    {6, velocities, RW},
                                    {7, mocap, RW},
                                    {8, force, RW},
                                    {11, saved, RW},
                                    {12, prescribed, RW},
                                    {15, fixedSteps, R}});
        if (multiple)
            drive = b.Dispatch("body_controls", "advance_commanded_substep", bodyCommandCapacity,
                               {{0, dimensions, R},
                                {1, bodies, R},
                                {2, joints, R},
                                {3, spans, R},
                                {5, positions, R},
                                {7, mocap, RW},
                                {8, force, W},
                                {11, saved, R},
                                {12, prescribed, W},
                                {15, fixedSteps, R}});
        finishCommands =
            response ? b.Dispatch("body_controls", "finish_contact_body_commands", bodyCommandCapacity,
                                  {{0, dimensions, R},
                                   {1, bodies, R},
                                   {2, joints, R},
                                   {3, spans, R},
                                   {5, positions, RW},
                                   {6, velocities, RW},
                                   {7, mocap, RW},
                                   {8, force, RW},
                                   {10, response->status, R},
                                   {11, saved, R},
                                   {12, prescribed, RW}})
                     : b.Dispatch("body_controls", "finish_body_commands", bodyCommandCapacity,
                                  {{0, dimensions, R}, {1, bodies, R}, {2, joints, R}, {3, spans, R}, {8, force, RW}});
        states = b.Slot(worldWords(20 * uint64_t{nb} + 4), BufferUsageFlags::TransferSource);
        if (branches != 0)
            bodyPublication.push_back(b.dispatches[kinematicsPass]);
        bodyPublication.push_back(response ? b.Dispatch("body_controls", "publish_contact_body_states", nb,
                                                        {{0, dimensions, R},
                                                         {1, bodies, R},
                                                         {6, velocities, R},
                                                         {9, clocks, R},
                                                         {10, response->status, R},
                                                         {12, prescribed, R},
                                                         {13, pose, R},
                                                         {14, states, W, true},
                                                         {15, fixedSteps, R}})
                                           : b.Dispatch("body_controls", "publish_body_states", nb,
                                                        {{0, dimensions, R},
                                                         {1, bodies, R},
                                                         {6, velocities, R},
                                                         {9, clocks, R},
                                                         {12, prescribed, R},
                                                         {13, pose, R},
                                                         {14, states, W, true},
                                                         {15, fixedSteps, R}}));
    }

    const auto snapshotSlot =
        b.Slot(worldWords(uint64_t{nq} + nv + 1 + uint32_t(response.has_value())), BufferUsageFlags::TransferSource);
    const auto snapshotWidth = std::max({1u, nq, nv});
    auto snapshot = response ? b.Dispatch("state_snapshot", "snapshot_contact_state", snapshotWidth,
                                          {{0, dimensions, R},
                                           {1, positions, R},
                                           {2, velocities, R},
                                           {3, clocks, R},
                                           {4, snapshotSlot, W, true},
                                           {6, response->status, R}})
                             : b.Dispatch("state_snapshot", "snapshot_generalized_state", snapshotWidth,
                                          {{0, dimensions, R},
                                           {1, positions, R},
                                           {2, velocities, R},
                                           {3, clocks, R},
                                           {4, snapshotSlot, W, true}});
    const auto residentSlots = b.slots.size();
    const auto residentPorts = b.ports.size();
    if (reference)
        InitializeWeights(b, model, *reference);
    std::vector<rhi::ComputeBufferDesc> initializationStorage;
    for (size_t slot = residentSlots; slot < b.slots.size(); ++slot)
        initializationStorage.push_back(b.slots[slot].allocation);
    b.slots.resize(residentSlots);
    b.ports.resize(residentPorts);
    auto recipe =
        std::make_shared<rhi::ComputeWorkRecipe>(host, std::move(b.slots), std::move(b.ports), std::move(b.dispatches));
    auto work = recipe->Instantiate();
    if (!work)
        return {};

    std::vector<rhi::ComputeBufferUpdate> initial;
    initial.reserve(b.initial.size());
    const auto &buffers = work->GetBuffers();
    auto initializationBuffers = buffers;
    for (const auto &allocation : initializationStorage) {
        auto buffer = rhi::ComputeBuffer::TryCreatePrivate(host, allocation);
        if (!buffer)
            return {};
        initializationBuffers.push_back(std::move(buffer));
    }
    for (auto &upload : b.initial)
        initial.push_back({initializationBuffers[upload.slot], 0, std::move(upload.data)});
    std::vector<rhi::ComputeDispatchDesc> initialDispatches;
    initialDispatches.reserve(b.initialize.size());
    for (auto &clear : b.initialize)
        initialDispatches.push_back(BindDispatch(std::move(clear), initializationBuffers));
    auto initialize = rhi::PrepareComputeBatch(host, std::move(initial), std::move(initialDispatches));
    WarpContactStream contacts;
    if (contactSlots)
        contacts = {buffers[contactSlots->candidateCount],
                    buffers[contactSlots->counts],
                    buffers[contactSlots->records],
                    response ? buffers[response->status] : nullptr,
                    response ? buffers[response->topology] : nullptr,
                    buffers[contactSlots->narrowphaseStatus]};
    WarpRayBuffers rays;
    std::vector<rhi::ComputeDispatchDesc> rayWork;
    if (raySlots) {
        rays = {buffers[raySlots->inputs], buffers[raySlots->hits]};
        rayWork.reserve(rayDispatches.size());
        for (auto &dispatch : rayDispatches)
            rayWork.push_back(BindDispatch(std::move(dispatch), buffers));
    }
    if (controlSlots) {
        controls.commands = buffers[controlSlots->commands];
        controls.spans = buffers[controlSlots->spans];
        controls.states = buffers[states];
        controls.apply = BindDispatch(std::move(applyCommands), buffers);
        controls.finish = BindDispatch(std::move(finishCommands), buffers);
        for (auto &dispatch : bodyPublication)
            controls.publication.push_back(BindDispatch(std::move(dispatch), buffers));
    }
    ContactReports reports;
    if (response && response->capture)
        reports = {buffers[response->reportRecords],
                   buffers[multiple ? response->status : contactSlots->counts],
                   BindDispatch(std::move(*response->capture), buffers),
                   multiple ? worlds * sizeof(uint32_t) : 0,
                   worldWords(substeps) * sizeof(uint32_t),
                   capacity.contactsPerWorld};
    HingeDrives hinges;
    if (response && response->hingeDrives != NoModelIndex)
        hinges = {buffers[response->hingeDrives], static_cast<uint32_t>(model.initialHingeDrives.size())};
    FixedSubsteps schedule;
    schedule.count = substeps;
    if (multiple) {
        for (auto &dispatch : begin)
            schedule.begin.push_back(BindDispatch(std::move(dispatch), buffers));
        if (drive.kernel)
            schedule.drive = BindDispatch(std::move(drive), buffers);
        if (advance.kernel)
            schedule.advance = BindDispatch(std::move(advance), buffers);
        if (restore.kernel)
            schedule.restore = BindDispatch(std::move(restore), buffers);
        schedule.clock = BindDispatch(std::move(clock), buffers);
    }
    auto step = std::unique_ptr<WarpStep>(new WarpStep(work, BindDispatch(std::move(snapshot), buffers), positions,
                                                       velocities, applied, std::move(contacts), std::move(rays),
                                                       std::move(rayWork), std::move(controls), std::move(reports),
                                                       std::move(schedule), std::move(hinges)));
    return {std::move(step), std::move(initialize)};
}

WarpStep::WarpStep(std::shared_ptr<rhi::ComputeWorkInstance> work, rhi::ComputeDispatchDesc snapshot,
                   uint32_t positions, uint32_t velocities, uint32_t appliedForces, WarpContactStream contacts,
                   WarpRayBuffers rays, std::vector<rhi::ComputeDispatchDesc> rayWork, BodyControls controls,
                   ContactReports reports, FixedSubsteps substeps, HingeDrives hinges)
    : m_work(std::move(work)), m_snapshot(std::move(snapshot)), m_positions(positions), m_velocities(velocities),
      m_appliedForces(appliedForces), m_contacts(std::move(contacts)), m_rays(std::move(rays)),
      m_rayWork(std::move(rayWork)), m_controls(std::move(controls)), m_reports(std::move(reports)),
      m_substeps(std::move(substeps)), m_hinges(std::move(hinges))
{
}

std::shared_ptr<rhi::PreparedComputeBatch> WarpStep::Prepare(bool publish,
                                                             std::vector<rhi::ComputeBufferUpdate> updates) const
{
    if (!publish && m_substeps.count == 1)
        return m_work->PrepareBatch(std::move(updates));
    auto dispatches = m_substeps.begin;
    AppendFixedWork(dispatches, 0, false);
    std::vector<rhi::ComputeBufferRead> reads;
    if (publish) {
        dispatches.push_back(m_snapshot);
        const auto &snapshot = m_snapshot.buffers[4]; // Binding 4 is the snapshot output in both entries.
        reads.push_back({snapshot, 0, snapshot->GetByteSize()});
    }
    auto host = m_work->GetRecipe().GetHost();
    return rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches), std::move(reads));
}

void WarpStep::AppendFixedWork(std::vector<rhi::ComputeDispatchDesc> &dispatches, uint32_t spans,
                               bool captureContacts) const
{
    auto drive = m_substeps.drive;
    const bool driving = spans != 0 && drive.kernel;
    if (driving) {
        drive.groupCountX = (spans - 1) / 64 + 1;
        LinearGrid(drive);
    }
    const bool reporting = captureContacts && m_reports.records;
    const auto width = m_work->GetRecipe().GetDispatches().size() + driving + reporting + (driving || reporting);
    dispatches.reserve(dispatches.size() + m_substeps.count * width + 2);
    for (uint32_t substep = 0; substep < m_substeps.count; ++substep) {
        if (driving)
            dispatches.push_back(drive);
        m_work->AppendDispatches(dispatches);
        if (reporting)
            dispatches.push_back(m_reports.capture);
        if ((driving || reporting) && substep + 1 < m_substeps.count)
            dispatches.push_back(m_substeps.advance);
    }
    if (m_substeps.restore.kernel)
        dispatches.push_back(m_substeps.restore);
    if (m_substeps.clock.kernel)
        dispatches.push_back(m_substeps.clock);
}

std::shared_ptr<rhi::PreparedComputeBatch> WarpStep::PrepareRays(bool publish,
                                                                 std::vector<rhi::ComputeBufferUpdate> updates) const
{
    if (!m_rays.hits)
        throw std::logic_error("Warp step has no ray query layout");
    std::vector<rhi::ComputeBufferRead> reads;
    if (publish)
        reads.push_back({m_rays.hits, 0, m_rays.hits->GetByteSize()});
    auto host = m_work->GetRecipe().GetHost();
    return rhi::PrepareComputeBatch(host, std::move(updates), m_rayWork, std::move(reads));
}

std::shared_ptr<rhi::PreparedComputeBatch> WarpStep::PrepareControlled(std::vector<WarpBodyCommand> commands,
                                                                       WarpBodyPublication publication,
                                                                       std::vector<rhi::ComputeBufferUpdate> updates,
                                                                       bool captureContacts,
                                                                       std::vector<WarpHingeDriveUpdate> drives) const
{
    if (!m_controls.states)
        throw std::logic_error("Warp step has no rigid body control layout");
    m_hinges.Append(updates, std::move(drives));
    if (commands.empty() && publication == WarpBodyPublication::None && !captureContacts)
        return Prepare(false, std::move(updates));
    if (commands.size() > m_controls.capacity)
        throw std::length_error("Warp fixed step body command capacity exceeded");

    auto dispatches = m_substeps.begin;
    const auto key = [](const auto &command) { return std::tie(command.world, command.body); };
    uint32_t spanCount = 0;
    if (!commands.empty()) {
        std::stable_sort(commands.begin(), commands.end(),
                         [&](const auto &a, const auto &b) { return key(a) < key(b); });
        std::vector<ModelWords> spans(1);
        for (uint32_t index = 0; index < commands.size(); ++index) {
            const auto &command = commands[index];
            const auto kind = command.kind;
            const auto motion = m_controls.motion.at(command.body);
            const bool move = kind == WarpBodyCommandKind::MovePosition || kind == WarpBodyCommandKind::MoveRotation;
            const bool dynamic =
                kind >= WarpBodyCommandKind::SetLinearVelocity && kind <= WarpBodyCommandKind::ForceAtPosition;
            if (command.world >= m_controls.worlds || motion == Uncontrolled || (move && motion != KinematicControl) ||
                (dynamic && motion != DynamicControl))
                throw std::invalid_argument("Warp body command does not match its world or motion type");
            if (index == 0 || key(command) != key(commands[index - 1]))
                spans.push_back({command.world, command.body, index, 0});
            ++spans.back()[3];
        }
        spanCount = static_cast<uint32_t>(spans.size() - 1);
        spans[0] = {spanCount, m_controls.timestep, m_substeps.count, 0};
        auto &spansUpload = updates.emplace_back(
            rhi::ComputeBufferUpdate{m_controls.spans, 0, std::vector<uint8_t>(spans.size() * sizeof(ModelWords))});
        std::memcpy(spansUpload.data.data(), spans.data(), spansUpload.data.size());
        auto &commandsUpload = updates.emplace_back(rhi::ComputeBufferUpdate{
            m_controls.commands, 0, std::vector<uint8_t>(commands.size() * sizeof(WarpBodyCommand))});
        std::memcpy(commandsUpload.data.data(), commands.data(), commandsUpload.data.size());
    }

    const auto commandDispatch = [&](rhi::ComputeDispatchDesc command) {
        command.groupCountX = (spanCount - 1) / 64 + 1;
        LinearGrid(command);
        dispatches.push_back(std::move(command));
    };
    if (spanCount != 0)
        commandDispatch(m_controls.apply);
    AppendFixedWork(dispatches, spanCount, captureContacts);
    if (spanCount != 0)
        commandDispatch(m_controls.finish);
    if (publication != WarpBodyPublication::None)
        dispatches.insert(dispatches.end(), m_controls.publication.begin(), m_controls.publication.end());
    std::vector<rhi::ComputeBufferRead> reads;
    if (publication == WarpBodyPublication::Cpu)
        reads.push_back({m_controls.states, 0, m_controls.states->GetByteSize()});
    if (publication == WarpBodyPublication::Status) {
        const auto headers = sizeof(ModelWords) * uint64_t{m_controls.worlds};
        reads.push_back({m_controls.states, m_controls.states->GetByteSize() - headers, headers});
    }
    if (captureContacts && m_reports.records)
        reads.push_back({m_reports.counts, m_reports.countsOffset, m_reports.countsBytes});
    auto host = m_work->GetRecipe().GetHost();
    return rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches), std::move(reads));
}

void WarpStep::HingeDrives::Append(std::vector<rhi::ComputeBufferUpdate> &updates,
                                   std::vector<WarpHingeDriveUpdate> values) const
{
    const auto key = [](const auto &value) { return std::tie(value.world, value.row); };
    std::stable_sort(values.begin(), values.end(), [&](const auto &a, const auto &b) { return key(a) < key(b); });
    rhi::ComputeBufferUpdate *range = nullptr;
    constexpr auto bytes = sizeof(ModelHingeDrive);
    for (const auto &value : values) {
        if (value.row >= stride || value.world >= buffer->GetByteSize() / (bytes * uint64_t{stride}))
            throw std::invalid_argument("Warp drive update does not match its world or hinge");
        const auto offset = (uint64_t{value.world} * stride + value.row) * bytes;
        if (!range || offset > range->offset + range->data.size())
            range = &updates.emplace_back(rhi::ComputeBufferUpdate{buffer, offset, {}});
        const auto local = static_cast<size_t>(offset - range->offset);
        range->data.resize(local + bytes);
        std::memcpy(range->data.data() + local, &value.drive, bytes);
    }
}

std::shared_ptr<rhi::PreparedComputeBatch> WarpStep::PrepareContactReports(const std::vector<uint32_t> &counts) const
{
    std::vector<rhi::ComputeBufferRead> reads;
    for (size_t world = 0; world < counts.size(); ++world) {
        const auto count = counts[world];
        if (count > m_reports.capacity)
            throw std::logic_error("Accepted Warp contact report count exceeds its world capacity");
        if (count != 0) {
            const auto base = world * m_reports.capacity * sizeof(WarpContactReport);
            reads.push_back({m_reports.records, base, count * sizeof(WarpContactReport)});
        }
    }
    if (reads.empty())
        return {};
    auto host = m_work->GetRecipe().GetHost();
    return rhi::PrepareComputeBatch(host, {}, {}, std::move(reads));
}

std::shared_ptr<rhi::ComputeBuffer> WarpStep::BodyStates() const noexcept
{
    return m_controls.states;
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
