// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "InxPhyGpuStep.h"
#include "../../GameObject.h"
#include "../../Rigidbody.h"
#include "InxPhyGpuCollisionModel.h"
#include "InxPhyGpuParticleExport.h"
#include "InxPhyGpuProgramSources.h"
#include "PhysicsRayIndex.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace infernux::physics::inxphygpu
{
namespace
{
using namespace rhi;
using Access = ComputeBufferAccess;
constexpr auto R = Access::Read;
constexpr auto W = Access::Write;
constexpr auto RW = Access::ReadWrite;
static_assert(sizeof(InxPhyGpuRay) == 48);
static_assert(sizeof(InxPhyGpuRayHit) == 32);
static_assert(sizeof(InxPhyGpuShapeCastHit) == 48);
static_assert(sizeof(InxPhyGpuBodyCommand) == 48);
static_assert(sizeof(ModelJointVectors) == 32);
static_assert(sizeof(ModelWords) == 16 && sizeof(ModelVector) == 16);
static_assert(sizeof(InxPhyGpuDistanceSpring) == 8);
static_assert(sizeof(InxPhyGpuBodyState) == PhysicsGpuPoseSource::RowBytes);
static_assert(sizeof(InxPhyGpuContactReport) == 96);
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
    StaticControl,
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
        const auto &[program, entry] = Resolve(module, name);
        ComputeWorkDispatch result;
        const auto width = entry.workgroupSize[0];
        result.groupCountX = groups ? x : (x - 1) / width + 1;
        result.groupCountY = worlds;
        result.bindings.reserve(uses.size());
        for (const auto &use : uses) {
            const auto range = Range(use.slot);
            if (use.complete)
                result.completeWrites.push_back({static_cast<uint32_t>(result.bindings.size()), 0, range.byteSize});
            result.bindings.push_back({range, use.access});
        }
        result.kernel = Kernel(program, entry, uses, [](const Use &use) { return use.binding; });
        return result;
    }

    std::shared_ptr<ComputeKernel> Kernel(std::string_view module, std::string_view name,
                                          std::initializer_list<uint32_t> bindings)
    {
        const auto &[program, entry] = Resolve(module, name);
        return Kernel(program, entry, bindings, [](uint32_t binding) { return binding; });
    }

    std::pair<const ShaderTargetCode &, const ShaderEntryPointCode &> Resolve(std::string_view module,
                                                                              std::string_view name)
    {
        auto foundProgram = programs.find(module);
        if (foundProgram == programs.end())
            foundProgram = programs.emplace(module, LoadInxPhyGpuProgram(module, target)).first;
        const auto &program = foundProgram->second;
        const auto entry = std::find_if(program.entries.begin(), program.entries.end(),
                                        [name](const auto &item) { return item.name == name; });
        if (entry == program.entries.end())
            throw std::logic_error("inxphygpu step program has no entry " + std::string(name));
        return {program, *entry};
    }

    template <class Bindings, class Projection>
    std::shared_ptr<ComputeKernel> Kernel(const ShaderTargetCode &program, const ShaderEntryPointCode &entry,
                                          const Bindings &bindings, Projection bindingOf)
    {
        if (entry.bindings.size() != bindings.size())
            throw std::logic_error("inxphygpu step resource count for " + entry.name + ": shader declares " +
                                   std::to_string(entry.bindings.size()) + ", step binds " +
                                   std::to_string(bindings.size()));
        auto &kernel = kernels[&entry];
        std::vector<ComputeBufferBinding> descriptors;
        for (const auto &use : bindings) {
            const auto binding = bindingOf(use);
            const std::array<uint32_t, 2> key{0, binding};
            if (std::find(entry.bindings.begin(), entry.bindings.end(), key) == entry.bindings.end())
                throw std::logic_error("inxphygpu step resource is absent from " + entry.name);
            if (!kernel) {
                const auto resource =
                    std::find_if(program.resources.begin(), program.resources.end(),
                                 [&](const auto &r) { return r.set == key[0] && r.binding.binding == key[1]; });
                if (resource == program.resources.end())
                    throw std::logic_error("inxphygpu step resource has no layout in " + entry.name);
                descriptors.push_back({binding, resource->binding.type, resource->binding.readOnlyStorage});
            }
        }
        if (!kernel) {
            const ShaderModuleDesc shader{target == ShaderCodeTarget::SpirV ? ShaderSourceLanguage::SpirV
                                                                            : ShaderSourceLanguage::Wgsl,
                                          entry.code.data(), entry.code.size()};
            kernel = std::make_shared<ComputeKernel>(host, shader, entry.name, std::move(descriptors), 0,
                                                     ComputeKernel::PipelineCreation::FirstDispatch);
        }
        return kernel;
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
            throw std::length_error("inxphygpu scratch exceeds the shader address space");
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
    uint32_t pairRules;
    uint32_t materials;
};

struct GeometrySlots
{
    uint32_t geometries, bounds, frames, shapes, data;
};

GeometrySlots BindGeometry(StepBuilder &b, const InxPhyGpuModel &model, const InxPhyGpuCollisionModel &collision,
                           bool rays)
{
    const auto count = model.dimensions.layout[0];
    return {b.Shared("geometries", model.geometries), b.Shared("localBounds", collision.bounds),
            count != 0 ? b.Slot(16 * uint64_t{count} * b.worlds) : b.Seed("geometryFrames", 16, nullptr),
            b.Shared("collisionShapes", collision.shapes),
            rays || !collision.convexData.empty() ? b.Shared("convexModel", collision.convexData) : NoModelIndex};
}

ContactSlots AddCollision(StepBuilder &b, const InxPhyGpuModel &model, const InxPhyGpuCollisionModel &collision,
                          uint32_t dimensions, const GeometrySlots &geometry, InxPhyGpuCollisionCapacity capacity,
                          float minimumRestitutionSpeed)
{
    const auto ng = model.dimensions.layout[0];
    const auto geometryWords = uint64_t{ng} * b.worlds;
    const auto bounds = geometry.bounds, frames = geometry.frames, shapes = geometry.shapes, data = geometry.data;
    const auto rules = b.Shared("collisionPairRules", collision.pairRules);
    const auto projected = b.Slot(28 * geometryWords);
    const uint32_t sorts[] = {b.Slot(2 * geometryWords), b.Slot(2 * geometryWords)};
    constexpr uint32_t sweepTile = 64;
    const auto tiles = (ng - 1) / sweepTile + 1;
    const auto tileWords = uint64_t{tiles} * b.worlds;
    const auto ranges = b.Slot(geometryWords);
    const uint32_t prefixes[] = {b.Slot(tileWords), b.Slot(tileWords)};
    const auto pairCount = b.Slot(1, BufferUsageFlags::TransferSource);
    const auto pairs = b.Zero("collisionCandidates", 5 * uint64_t{capacity.candidates});
    const auto workers = ng;
    constexpr uint32_t boundsFilters = 15;
    constexpr uint32_t parentFilter = 16;
    const auto filters = boundsFilters | (collision.filterConnectedBodies ? parentFilter : 0);
    const ModelWords options{0, capacity.candidates, filters, workers};
    const auto emitSettings = b.Seed("sweepEmission", 4, &options);
    b.Add("broadphase", "sap_project_tiles", ng,
          {{0, dimensions, R}, {2, bounds, R}, {3, frames, R}, {4, projected, W, true}, {6, sorts[0], W, true}});
    std::vector<uint32_t> strides;
    uint32_t currentSort = 0;
    for (uint64_t width = sweepTile; width < ng; width *= 2) {
        const auto stride = static_cast<uint32_t>(width);
        const ModelWords values{stride, stride / sweepTile, 0, 0};
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
           {8, ranges, W, true},
           {10, pairCount, W, true},
           {13, prefixes[0], W, true}});
    uint32_t currentScan = 0;
    for (const auto settings : strides) {
        b.Add("broadphase", "sap_scan_tiles", tiles,
              {{0, dimensions, R},
               {1, settings, R},
               {7, prefixes[currentScan], R},
               {8, prefixes[1 - currentScan], W, true}});
        currentScan = 1 - currentScan;
    }
    b.Add("broadphase", "sap_emit_pairs", workers,
          {{0, dimensions, R},
           {1, emitSettings, R},
           {4, projected, R},
           {5, sorts[currentSort], R},
           {7, ranges, R},
           {9, rules, R},
           {10, pairCount, RW},
           {11, pairs, W},
           {12, prefixes[currentScan], R}});

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
    return {pairCount, counts, contacts, status, rules, materials};
}

struct BodyControlSlots
{
    uint32_t spans, commands, force, saved, prescribed, fixedSteps, activity, pendingWrenches;
};

void AddSceneContinuousCollision(StepBuilder &b, const InxPhyGpuModel &model, const InxPhyGpuCollisionModel &collision,
                                 const GeometrySlots &geometry, const ContactSlots &contacts,
                                 const InxPhyGpuContactSolver &solver, float dt, uint32_t dimensions, uint32_t bodies,
                                 uint32_t start, uint32_t positions, uint32_t velocities, uint32_t status,
                                 const BodyControlSlots &controls)
{
    const ModelWords settings{FloatWord(dt), FloatWord(solver.linearCastThreshold),
                              FloatWord(solver.minimumRestitutionSpeed), collision.filterConnectedBodies};
    const auto options = b.Seed("sceneCcdSettings", 4, &settings);
    const auto rows = b.Shared("sceneCcdBodies", model.sceneContinuousCollision);
    const auto nb = model.dimensions.state[2];
    const auto motion = b.Slot(16 * uint64_t{nb} * b.worlds);
    constexpr uint64_t workers = 8, records = 1360, edges = 512;
    const auto scratch = b.Slot(4 * workers * records * b.worlds);
    const auto horizon = b.Slot(workers * edges * b.worlds);
    b.Add("scene_ccd", "prepare_scene_ccd", nb,
          {{8, dimensions, R},
           {9, bodies, R},
           {10, rows, R},
           {11, positions, R},
           {12, velocities, R},
           {13, start, R},
           {14, motion, W, true},
           {15, options, R},
           {16, status, R},
           {18, controls.prescribed, R},
           {24, controls.fixedSteps, R}});
    b.Add("scene_ccd", "solve_scene_ccd", 1,
          {{2, geometry.shapes, R},
           {5, contacts.materials, R},
           {8, dimensions, R},
           {9, bodies, R},
           {10, rows, R},
           {14, motion, RW},
           {15, options, R},
           {17, geometry.geometries, R},
           {19, contacts.pairRules, R},
           {20, geometry.data, R},
           {21, scratch, W},
           {22, horizon, W}},
          true);
    b.Add("scene_ccd", "publish_scene_ccd", nb,
          {{8, dimensions, R},
           {9, bodies, R},
           {10, rows, R},
           {11, positions, RW},
           {12, velocities, RW},
           {14, motion, R},
           {23, controls.activity, RW}});
}

struct RaySlots
{
    uint32_t inputs, hits, settings;
    uint32_t metadata;
    uint32_t inputSettings = NoModelIndex, outputSettings = NoModelIndex;
    uint32_t closestSettings = NoModelIndex, closestResult = NoModelIndex;
    uint32_t overlapRequest = NoModelIndex, overlapMask = NoModelIndex;
    uint32_t particlePoses, particleShapes, particleCold;
    ComputeWorkDispatch pack, project, traceAll, closest, overlap;
};

constexpr uint32_t SavedMotionWords = 44;
constexpr uint32_t MaxBodyCommands = UINT32_MAX / SavedMotionWords;

BodyControlSlots BindBodyControls(StepBuilder &b, const InxPhyGpuModel &model, uint32_t capacity, bool multiple)
{
    if (capacity > MaxBodyCommands)
        throw std::length_error("inxphygpu body commands exceed the shader address space");
    const auto prescribedRows = uint64_t{model.dimensions.layout[1]} + model.initialStaticPoses.size();
    const ModelWords awake{0, NoModelIndex, 0, 0};
    return {b.Slot(4 * (uint64_t{capacity} + 1), BufferUsageFlags::TransferDestination),
            b.Slot(12 * uint64_t{capacity}, BufferUsageFlags::TransferDestination),
            b.Zero("commandedForce", uint64_t{model.dimensions.state[1]} * b.worlds),
            b.Slot(SavedMotionWords * uint64_t{capacity}),
            b.Seed("prescribedMotion", 12 * std::max(uint64_t{1}, prescribedRows),
                   model.initialKinematicMotion.empty() ? nullptr : model.initialKinematicMotion.data(), b.worlds),
            b.Zero("acceptedFixedSteps", uint64_t{b.worlds} + uint32_t(multiple)),
            b.Seed("bodyActivity", awake.size(), awake.data(), b.worlds * model.dimensions.state[2]),
            b.Zero("pendingWrenches", 6 * uint64_t{b.worlds} * model.dimensions.state[2])};
}

RaySlots AddRayWork(StepBuilder &b, const InxPhyGpuModel &model, const InxPhyGpuCollisionModel &collision,
                    const GeometrySlots &geometry, InxPhyGpuRayBatchLayout layout,
                    std::vector<ComputeWorkDispatch> &work, bool published)
{
    const auto worlds = published ? 1u : b.worlds;
    const auto inputRows = layout.sharedRays ? 1u : worlds;
    const auto rayTotal = uint64_t{layout.raysPerWorld} * worlds;
    if (rayTotal > UINT32_MAX)
        throw std::length_error("inxphygpu ray batch exceeds the shader address space");
    const auto inputs = b.Seed("rays", 12 * uint64_t{layout.raysPerWorld} * inputRows, nullptr);
    // Scene queries attach the actual result allocation when they are prepared.
    const auto hits = b.Slot(8 * (published ? 1u : rayTotal), BufferUsageFlags::TransferSource);
    const auto geometryCount = model.dimensions.layout[0];
    const ModelWords settings{geometryCount, layout.raysPerWorld, inputRows, 0};
    const auto query = b.Seed("raySettings", 4, &settings);
    const auto index = BuildInxPhyGpuRayIndex(model, collision);
    const auto nodes = b.Shared("rayIndex", index.records);
    const auto bounds = b.Zero("rayBounds", 8 * uint64_t{std::max(1u, index.nodeCount)} * worlds);
    const auto metadata = b.Shared("rayGeometry", collision.rayGeometry);
    const auto describeFrames = [&](ComputeWorkDispatch &dispatch) {
        if (!published || geometryCount == 0)
            return;
        // Frozen queries project and refit just one selected world into the prefix.
        // The rest of the numerical scratch may be uninitialized before its first step.
        for (auto &binding : dispatch.bindings)
            if (binding.range.slot == geometry.frames)
                binding.range.byteSize = uint64_t{geometryCount} * 4 * sizeof(ModelVector);
    };
    const auto refit = [&](uint32_t first, uint32_t count, bool complete) {
        const ModelWords options{geometryCount, first, 0, 0};
        const auto slot = b.Seed(first == 0 ? "rayRefitSubtrees" : "rayRefitRoot", 4, &options);
        work.push_back(b.Dispatch(
            "ray_bvh_refit", "refit_ray_bvh", count,
            {{0, slot, R}, {1, geometry.bounds, R}, {2, nodes, R}, {3, bounds, RW, complete}, {4, geometry.frames, R}},
            true));
        describeFrames(work.back());
    };
    if (index.refitSubtrees != 0) {
        refit(0, index.refitSubtrees, !index.refitRoot);
        if (index.refitRoot)
            refit(index.refitSubtrees, 1, true);
    }
    const auto trace = [&](const char *entry, uint32_t count) {
        auto dispatch = b.Dispatch("ray_queries", entry, count,
                                   {{0, query, R},
                                    {1, inputs, R},
                                    {2, geometry.frames, R},
                                    {3, geometry.shapes, R},
                                    {4, metadata, R},
                                    {5, geometry.data, R},
                                    {7, hits, W, true},
                                    {8, nodes, R},
                                    {9, bounds, R}});
        describeFrames(dispatch);
        return dispatch;
    };
    work.push_back(trace("raycast_worlds", layout.raysPerWorld));
    RaySlots result{};
    result.particlePoses = geometry.geometries;
    result.particleShapes = geometry.shapes;
    result.particleCold = geometry.data;
    result.inputs = inputs;
    result.hits = hits;
    result.settings = query;
    result.metadata = metadata;
    if (published) {
        result.traceAll = trace("raycast_world_all", 1);
        result.inputSettings = b.Seed("packedSceneRaySettings", 8, nullptr);
        // Only the selected external views are bound in the prepared packet.
        result.pack =
            b.Dispatch("ray_queries", "pack_scene_rays", layout.raysPerWorld,
                       {{10, result.inputSettings, R}, {11, inputs, R}, {12, inputs, R}, {13, inputs, W, true}});
        result.outputSettings = b.Seed("projectedSceneRaySettings", 8, nullptr);
        result.project = b.Dispatch("ray_queries", "project_scene_hits", layout.raysPerWorld,
                                    {{14, result.outputSettings, R},
                                     {1, inputs, R},
                                     {15, hits, R},
                                     {16, hits, W, true},
                                     {17, hits, W, true},
                                     {18, hits, W, true},
                                     {19, hits, W, true}});
        if (geometryCount != 0) {
            result.closestSettings = b.Slot(8, BufferUsageFlags::TransferDestination);
            result.closestResult = b.Slot(4, BufferUsageFlags::TransferSource);
            result.closest = b.Dispatch("ray_queries", "closest_collider_point", 1,
                                        {{2, geometry.frames, R},
                                         {3, geometry.shapes, R},
                                         {5, geometry.data, R},
                                         {24, result.closestSettings, R},
                                         {25, result.closestResult, W, true}});
            describeFrames(result.closest);
            result.closest.groupCountY = 1;
            result.overlapRequest = b.Slot(16, BufferUsageFlags::TransferDestination);
            result.overlapMask = b.Slot((uint64_t{geometryCount} + 31) / 32, BufferUsageFlags::TransferSource);
            result.overlap = b.Dispatch("ray_queries", "overlap_scene_shapes", geometryCount,
                                        {{2, geometry.frames, R},
                                         {3, geometry.shapes, R},
                                         {4, metadata, R},
                                         {5, geometry.data, R},
                                         {20, geometry.bounds, R},
                                         {26, result.overlapRequest, R},
                                         {27, result.overlapMask, W, true}});
            describeFrames(result.overlap);
            result.overlap.groupCountY = 1;
        }
    }
    return result;
}

struct ContactResponse
{
    uint32_t acceleration;
    uint32_t status;
    uint32_t topology;
    uint32_t reportRecords = NoModelIndex;
    std::optional<ComputeWorkDispatch> capture;
    uint32_t jointParameters = NoModelIndex;
    uint32_t settings;
    ModelWords worldDefaults;
};

struct ReferenceWeights
{
    uint32_t dimensions, dofs, mass, centeredDof, inertiaFrames, subtreeCom, topology;
};

void InitializeWeights(StepBuilder &b, const InxPhyGpuModel &model, const ReferenceWeights &ref)
{
    const auto factorization = BuildInxPhyGpuReferenceMassFactors(model);
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

struct SceneWakeWork
{
    const BodyControlSlots *control;
    uint32_t groups;
    uint32_t applied;
};

void AddSceneWakeWork(StepBuilder &b, const InxPhyGpuModel &model, const ContactDynamics &dyn,
                      const SceneWakeWork &work, const ContactSlots *stream = nullptr, uint32_t contacts = 0,
                      uint32_t settings = NoModelIndex, uint32_t topology = NoModelIndex,
                      uint32_t joints = NoModelIndex)
{
    const auto nb = model.dimensions.state[2];
    const auto state = work.control->activity;
    const auto groups = work.groups;
    b.Add("sleep", "begin_scene_wakes", nb,
          {{34, state, RW}, {2, groups, RW}, {5, work.applied, R}, {27, dyn.motionModel, R}, {28, dyn.bodyModel, R}});
    if (stream) {
        std::vector<Use> edges{{34, state, R},
                               {2, groups, RW},
                               {27, dyn.motionModel, R},
                               {28, dyn.bodyModel, R},
                               {29, work.control->fixedSteps, R},
                               {30, work.control->prescribed, R}};
        auto contactEdges = edges;
        contactEdges.insert(contactEdges.end(),
                            {{9, stream->records, R}, {10, stream->counts, R}, {11, settings, R}, {12, topology, R}});
        b.dispatches.push_back(b.Bind("sleep", "link_scene_contact_wakes", contacts, contactEdges));
        if (joints != NoModelIndex) {
            edges.push_back({31, joints, R});
            b.dispatches.push_back(b.Bind("sleep", "link_scene_joint_wakes", model.sceneJoints[0][0], edges));
        }
    }
    b.Add("sleep", "collect_scene_wakes", nb, {{34, state, R}, {2, groups, RW}, {27, dyn.motionModel, R}});
    b.Add("sleep", "resolve_scene_wakes", nb,
          {{34, state, RW},
           {2, groups, RW},
           {4, dyn.smoothAcceleration, RW},
           {22, dyn.warmstart, RW},
           {27, dyn.motionModel, R},
           {28, dyn.bodyModel, R},
           {40, dyn.smoothForce, RW}});
}

ContactResponse AddContactResponse(StepBuilder &b, const InxPhyGpuModel &model, const ContactSlots &stream,
                                   InxPhyGpuCollisionCapacity capacity, const InxPhyGpuContactSolver &solver,
                                   float timestep, const ContactDynamics &dyn,
                                   std::vector<ComputeWorkDispatch> precondition, bool contactReports,
                                   uint32_t substeps, uint32_t maxStorageBuffers, const SceneWakeWork &wake,
                                   uint32_t sceneJoints,
                                   const std::vector<InxPhyGpuStateHandoff::PoseTarget> &poseTargets)
{
    const auto nv = model.dimensions.state[1];
    const auto nb = model.dimensions.state[2];
    const auto ng = model.dimensions.layout[0];
    if (solver.rowsPerWorld == 0 || solver.columnsPerWorld == 0 || solver.iterations == 0 || solver.lineIterations == 0)
        throw std::invalid_argument("inxphygpu contact response requires nonzero solver limits");

    std::vector<uint32_t> lockedBodies;
    uint64_t rowCapacity = solver.rowsPerWorld;
    uint64_t columnCapacity = solver.columnsPerWorld;
    for (uint32_t row = 1; row < nb; ++row) {
        const auto &body = model.bodies[row];
        const auto mask = body.subtree[3];
        if (!mask || !body.dofs[1])
            continue;
        lockedBodies.push_back(row);
        for (uint32_t axis = 0; axis < 6; ++axis) {
            if (mask & (1u << axis)) {
                ++rowCapacity;
                columnCapacity += body.dofs[1];
            }
        }
    }
    if (rowCapacity > UINT32_MAX || columnCapacity > UINT32_MAX)
        throw std::length_error("inxphygpu constraint capacity exceeds its device index range");
    const auto rows = static_cast<uint32_t>(rowCapacity);
    const auto columns = static_cast<uint32_t>(columnCapacity);
    const auto contacts = capacity.contactsPerWorld;
    constexpr uint32_t ellipticCone = 1;
    const ModelWords worldDefaults{FloatWord(timestep), FloatWord(1.0f), FloatWord(solver.tolerance),
                                   FloatWord(solver.lineTolerance)};
    std::vector<ModelWords> config(b.worlds + 2, worldDefaults);
    config[0] = {ng, contacts, rows, columns};
    config[1] = {nb, nv, contacts, ellipticCone};
    const auto settings = b.Shared("constraintSettings", config);

    std::vector<ModelWords> topology(uint64_t{ng} + nb);
    for (uint32_t geom = 0; geom < ng; ++geom) {
        const auto body = model.geometries[geom].indices[0];
        topology[geom][0] = body;
        const auto &row = model.bodies[body];
        topology[geom][1] =
            row.dofs[2] != 0 ? model.dimensions.layout[1] + row.dofs[2] - 1 : model.bodies[row.subtree[0]].hierarchy[3];
    }
    topology[ng] = {NoModelIndex, 0, 0, nb};
    for (uint32_t body = 1; body < nb; ++body) {
        const auto &record = model.bodies[body];
        const auto lastDof =
            record.dofs[1] != 0 ? record.dofs[0] + record.dofs[1] - 1 : topology[ng + record.hierarchy[0]][0];
        topology[ng + body] = {lastDof, record.subtree[0], 0, record.subtree[1]};
    }
    const auto contactTopology = b.Shared("contactTopology", topology);
    if (wake.control)
        AddSceneWakeWork(b, model, dyn, wake, &stream, contacts, settings, contactTopology, sceneJoints);
    const auto across = [count = b.worlds](uint64_t words) { return words * count; };
    const auto counters = b.Slot(across(2));
    const auto reservations = b.Zero("contactReservations", across(4 * uint64_t{contacts}));
    const auto rowStore = b.Zero("constraintRows", across(12 * uint64_t{rows}));
    const auto columnStore = b.Zero("constraintColumns", across(2 * uint64_t{columns}));
    const auto transposeAddress = b.Zero("constraintTransposeAddress", across(2 * uint64_t{nv} + 1));
    const auto transposeEntries = b.Zero("constraintTransposeEntries", across(2 * uint64_t{columns}));
    const auto constraintState = b.Zero("constraintSolveState", across(4 * uint64_t{rows}));
    const auto acceleration = b.Zero("contactAcceleration", across(nv));

    constexpr uint32_t ContactTileSize = 256;
    const auto orderWords = across(contacts);
    auto order = b.Zero("contactOrder", orderWords);
    b.Add("contact_constraints", "sort_contact_tiles", (contacts - 1) / ContactTileSize + 1,
          {{0, settings, R}, {1, stream.records, R}, {2, stream.counts, R}, {21, order, W}}, true);
    if (contacts > ContactTileSize) {
        auto merged = b.Zero("mergedContactOrder", orderWords);
        for (uint64_t width = ContactTileSize; width < contacts; width *= 2) {
            const auto mergeWidth = static_cast<uint32_t>(width);
            const auto merge = b.Seed("contactMerge" + std::to_string(width), 1, &mergeWidth);
            b.Add("contact_constraints", "merge_contact_tiles", contacts,
                  {{0, settings, R},
                   {1, stream.records, R},
                   {2, stream.counts, R},
                   {20, order, R},
                   {21, merged, W},
                   {22, merge, R}});
            std::swap(order, merged);
        }
    }
    std::vector<Use> reserve{{0, settings, R},        {1, stream.records, R}, {2, stream.counts, R},
                             {3, contactTopology, R}, {4, dyn.dofs, R},       {5, counters, W, true},
                             {6, reservations, W},    {8, columnStore, W},    {20, order, R}};
    if (wake.control)
        reserve.push_back({25, wake.control->activity, R});
    b.dispatches.push_back(b.Bind("contact_constraints",
                                  wake.control ? "reserve_scene_contact_rows" : "reserve_contact_rows", 1, reserve));
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
               {18, dyn.prescribedMotion, R},
               {19, dyn.fixedSteps, R}});
    uint32_t jointParameters = NoModelIndex;
    if (!model.sceneJoints.empty()) {
        const auto jointCount = model.sceneJoints[0][0];
        const auto sixCount = model.sceneJoints[0][2];
        const auto &parameters = model.initialSceneJointParameters;
        const auto parameterWords = sizeof(ModelWords) / sizeof(uint32_t);
        jointParameters =
            b.Seed("sceneJointParameters", parameterWords * parameters.size(), parameters.data(), b.worlds);
        auto &initial = b.initial.back().data;
        for (const auto &target : poseTargets) {
            if (target.world >= b.worlds || target.offset >= parameters.size())
                throw std::invalid_argument("inxphygpu pose target handoff exceeds its scene joint layout");
            const auto offset = (target.world * parameters.size() + target.offset) * sizeof(ModelWords);
            std::memcpy(initial.data() + offset, target.orientation.data(), sizeof(ModelWords));
        }
        if (dyn.prescribedMotion != NoModelIndex)
            b.Add("spatial_constraints", "write_scene_prescribed_velocities", nb,
                  {{4, dyn.subtreeCom, R},
                   {12, dyn.bodyModel, R},
                   {13, dyn.prescribedMotion, R},
                   {14, dyn.fixedSteps, R},
                   {15, dyn.bodyVelocity, W},
                   {16, dyn.motionModel, R}});
        const auto addJoints = [&](std::string_view entry, uint32_t count) {
            if (count == 0)
                return;
            std::vector<Use> bindings{
                {0, settings, R},          {1, sceneJoints, R},     {3, dyn.bodyPose, R},
                {4, dyn.subtreeCom, R},    {5, counters, RW},       {6, dyn.centeredDof, R},
                {7, rowStore, W},          {8, columnStore, W},     {9, dyn.centeredDofDerivative, R},
                {10, dyn.bodyVelocity, R}, {11, dyn.velocities, R}, {17, jointParameters, R}};
            if (wake.control)
                bindings.insert(bindings.end(), {{12, dyn.bodyModel, R}, {18, wake.control->activity, R}});
            b.dispatches.push_back(b.Bind("spatial_constraints", entry, count, bindings));
        };
        addJoints(wake.control ? "emit_controlled_scene_joint_rows" : "emit_scene_joint_rows", jointCount - sixCount);
        addJoints(wake.control ? "emit_controlled_scene_six_dof_rows" : "emit_scene_six_dof_rows", sixCount);
    }
    if (!lockedBodies.empty()) {
        const auto bodyRows = b.Shared("sceneLockedBodies", lockedBodies);
        std::vector<Use> bindings{{0, settings, R},          {3, dyn.bodyPose, R},   {5, counters, RW},
                                  {6, dyn.centeredDof, R},   {7, rowStore, W},       {8, columnStore, W},
                                  {10, dyn.bodyVelocity, R}, {12, dyn.bodyModel, R}, {22, bodyRows, R}};
        if (wake.control)
            bindings.push_back({18, wake.control->activity, R});
        const auto *entry = wake.control ? "emit_controlled_scene_axis_locks" : "emit_scene_axis_locks";
        b.dispatches.push_back(
            b.Bind("spatial_constraints", entry, static_cast<uint32_t>(lockedBodies.size()), bindings));
    }
    b.Add("contact_constraints", "seed_contact_acceleration", nv,
          {{0, settings, R},
           {5, counters, R},
           {12, dyn.smoothAcceleration, R},
           {13, dyn.warmstart, R},
           {14, acceleration, W},
           {23, transposeAddress, W}});
    std::array<ModelWords, 3> options{{{solver.iterations, solver.lineIterations, model.dimensions.layout[3], 0},
                                       {capacity.candidates, substeps, 0, 0},
                                       {}}};
    const auto indirectSlot = b.Slot(7 * 4, BufferUsageFlags::Indirect);
    const auto directionIndirect = b.Slot(4, BufferUsageFlags::Indirect);
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
    const auto massTopology = b.Shared("massAdjacency", model.massAdjacency);
    const auto worldState = b.Slot(across(12));
    const auto gradient = b.Zero("contactGradient", across(nv));
    const auto preconditioned = b.Zero("preconditionedContactGradient", across(nv));
    const auto history = b.Zero("contactSearchHistory", across(4 * uint64_t{nv}));
    const auto products = b.Zero("contactMassProducts", across(2 * uint64_t{nv}));
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
          {{0, settings, R},
           {1, opt, R},
           {16, worldState, R},
           {28, indirectSlot, W, true},
           {29, directionIndirect, W, true}});
    b.dispatches.back().groupCountY = 1;
    b.Add("contact_cg", "count_cg_transpose", rows,
          {{0, settings, R}, {3, rowStore, R}, {4, columnStore, R}, {16, worldState, R}, {23, transposeAddress, RW}});
    indirectLast(Rows);
    b.Add("contact_cg", "prefix_cg_transpose", 1, {{0, settings, R}, {16, worldState, R}, {23, transposeAddress, RW}});
    indirectLast(Worlds);
    b.Add("contact_cg", "scatter_cg_transpose", rows,
          {{0, settings, R},
           {3, rowStore, R},
           {4, columnStore, R},
           {16, worldState, R},
           {23, transposeAddress, RW},
           {24, transposeEntries, W}});
    indirectLast(Rows);
    b.Add("contact_cg", "cg_initialize_constraint_residuals", rows,
          {{0, settings, R},
           {3, rowStore, R},
           {4, columnStore, R},
           {5, stream.records, R},
           {6, constraintState, W},
           {8, acceleration, R},
           {16, worldState, R}});
    indirectLast(Rows);
    b.Add("contact_cg", "cg_initial_mass_product", nv,
          {{0, settings, R},
           {1, opt, R},
           {8, acceleration, R},
           {10, dyn.mass, R},
           {11, massTopology, R},
           {16, worldState, R},
           {17, products, W}});
    indirectLast(Dofs);

    auto rayProduct = b.Dispatch("contact_cg", "cg_ray_constraint_product", rows,
                                 {{0, settings, R},
                                  {3, rowStore, R},
                                  {4, columnStore, R},
                                  {6, constraintState, RW},
                                  {15, history, R},
                                  {16, worldState, R}});
    rayProduct.indirectArguments = grid(Rows);
    constexpr uint32_t FullSearchStorageBindings = 11;
    std::optional<ComputeWorkDispatch> massSearch;
    ComputeWorkDispatch search;
    if (maxStorageBuffers >= FullSearchStorageBindings) {
        search = b.Dispatch("contact_cg", "cg_search_step", 1,
                            {{0, settings, R},
                             {1, opt, R},
                             {3, rowStore, R},
                             {5, stream.records, R},
                             {6, constraintState, R},
                             {9, dyn.smoothForce, R},
                             {10, dyn.mass, R},
                             {11, massTopology, R},
                             {15, history, R},
                             {16, worldState, RW},
                             {17, products, RW}});
    } else {
        massSearch = b.Dispatch("contact_cg", "cg_ray_mass_and_gauss", 1,
                                {{0, settings, R},
                                 {1, opt, R},
                                 {9, dyn.smoothForce, R},
                                 {10, dyn.mass, R},
                                 {11, massTopology, R},
                                 {15, history, R},
                                 {16, worldState, RW},
                                 {17, products, RW}});
        massSearch->indirectArguments = grid(Worlds);
        search = b.Dispatch("contact_cg", "cg_line_search", 1,
                            {{0, settings, R},
                             {1, opt, R},
                             {3, rowStore, R},
                             {5, stream.records, R},
                             {6, constraintState, R},
                             {16, worldState, RW}});
    }
    search.indirectArguments = grid(Worlds);

    // A fixed recipe records the upper bound. Each world's progress controls its device work.
    for (uint32_t iteration = 0; iteration <= solver.iterations; ++iteration) {
        const bool first = iteration == 0;
        b.Add("contact_cg", first ? "cg_project_first_gradient" : "cg_project_gradient", nv,
              {{0, settings, R},
               {6, constraintState, R},
               {9, dyn.smoothForce, R},
               {12, gradient, W},
               {16, worldState, R},
               {17, products, R},
               {23, transposeAddress, R},
               {24, transposeEntries, first ? RW : R}});
        indirectLast(Dofs);
        b.dispatches.insert(b.dispatches.end(), precondition.begin(), precondition.end());
        b.Add("contact_cg", "cg_search_direction", 1,
              {{0, settings, R},
               {1, opt, R},
               {12, gradient, R},
               {13, preconditioned, R},
               {15, history, RW},
               {16, worldState, RW},
               {28, indirectSlot, RW}});
        // This dispatch updates the consumer grids, so its own arguments
        // live in a separate allocation on both Vulkan and WebGPU.
        b.dispatches.back().indirectArguments =
            ComputeWorkRange{directionIndirect, 0, ComputeIndirectArguments::ByteSize};
        if (iteration == solver.iterations)
            break;
        b.dispatches.push_back(rayProduct);
        if (massSearch)
            b.dispatches.push_back(*massSearch);
        b.dispatches.push_back(search);
        b.Add("contact_cg", "cg_apply_step", std::max(nv, rows),
              {{0, settings, R},
               {3, rowStore, R},
               {5, stream.records, R},
               {6, constraintState, RW},
               {8, acceleration, RW},
               {15, history, R},
               {16, worldState, R},
               {17, products, RW}});
        indirectLast(RowsOrDofs);
    }
    b.Add("contact_cg", "finish_contact_step", 1,
          {{0, settings, R},
           {1, opt, R},
           {16, worldState, R},
           {25, stream.candidateCount, R},
           {26, stream.counts, R},
           {27, status, multiple ? RW : W, !multiple},
           {30, stream.narrowphaseStatus, R}});
    ContactResponse response{acceleration, status,          contactTopology, NoModelIndex,
                             {},           jointParameters, settings,        worldDefaults};
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
                              {5, constraintState, R},
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

struct BodyBoundsSlots
{
    uint32_t request = NoModelIndex, mask = NoModelIndex;
    std::vector<uint32_t> rows;
    ComputeWorkDispatch select;
};

BodyBoundsSlots AddPublishedBodyBounds(StepBuilder &b, const GeometrySlots &geometry, uint32_t metadata,
                                       uint32_t bodyCount, const std::vector<ModelGeometry> &geometries,
                                       std::vector<uint32_t> bodyRows)
{
    const auto count = static_cast<uint32_t>(bodyRows.size());
    if (count == 0)
        return {};
    std::vector<uint32_t> indices(bodyCount, NoModelIndex);
    for (uint32_t index = 0; index < count; ++index)
        indices[bodyRows[index]] = index;
    std::vector<std::array<uint32_t, 2>> ranges(count, {NoModelIndex, 0});
    // The scene cooker emits every body's leaves contiguously in author order.
    for (uint32_t row = 0; row < geometries.size(); ++row) {
        const auto index = indices[geometries[row].indices[0]];
        if (index != NoModelIndex) {
            auto &range = ranges[index];
            range[0] = std::min(range[0], row);
            range[1] = row + 1;
        }
    }
    const auto rangeStorage = b.Shared("queryBodyGeometryRanges", ranges);
    const auto request = b.Slot(12, BufferUsageFlags::TransferDestination);
    const auto mask = b.Slot((uint64_t{count} + 31) / 32, BufferUsageFlags::TransferSource);
    auto select = b.Dispatch("ray_queries", "select_body_bounds", count,
                             {{2, geometry.frames, R},
                              {3, geometry.shapes, R},
                              {4, metadata, R},
                              {20, geometry.bounds, R},
                              {21, rangeStorage, R},
                              {22, request, R},
                              {23, mask, W, true}});
    select.bindings.front().range.byteSize = geometries.size() * 4 * sizeof(ModelVector);
    return {request, mask, std::move(bodyRows), std::move(select)};
}
} // namespace

namespace
{
std::shared_ptr<rhi::ComputeBuffer> AllocateParticleWords(rhi::ComputeHost &host, uint64_t bytes)
{
    auto buffer =
        rhi::ComputeBuffer::TryCreatePrivate(host, {bytes / sizeof(uint32_t), rhi::ComputeScalarType::UInt32, 1});
    if (!buffer)
        throw std::runtime_error("inxphygpu could not allocate particle collision storage");
    return buffer;
}

template <class T> std::vector<uint8_t> ParticleBytes(const T *value, size_t count = 1)
{
    std::vector<uint8_t> result(count * sizeof(T));
    std::memcpy(result.data(), value, result.size());
    return result;
}

rhi::ComputeDispatchDesc
ParticleDispatch(std::shared_ptr<rhi::ComputeKernel> kernel, uint32_t groups,
                 std::initializer_list<std::pair<std::shared_ptr<rhi::ComputeBuffer>, Access>> buffers)
{
    rhi::ComputeDispatchDesc result;
    result.kernel = std::move(kernel);
    result.groupCountX = groups;
    result.proof.emplace();
    for (const auto &[buffer, access] : buffers) {
        const auto bind = static_cast<uint32_t>(result.buffers.size());
        result.buffers.push_back(buffer);
        result.bufferAccesses.push_back(access);
        if (access != W)
            result.proof->reads.push_back({bind, 0, buffer->GetByteSize()});
        if (access != R)
            result.proof->writes.push_back({bind, 0, buffer->GetByteSize()});
    }
    return result;
}
} // namespace

InxPhyGpuParticleExport::InxPhyGpuParticleExport(rhi::ComputeHost &host, ShaderCodeTarget target)
{
    StepBuilder builder(host, target, 1);
    m_project = builder.Kernel("particle_collision", "particle_collision_project", {0, 1, 2, 3, 4, 5, 6, 10});
    m_count = builder.Kernel("particle_collision", "particle_collision_count", {0, 6, 7, 8, 10});
    m_scatter = builder.Kernel("particle_collision", "particle_collision_scatter", {0, 6, 7, 8, 9, 10});
}

InxPhyGpuParticleExport::Work
InxPhyGpuParticleExport::Prepare(rhi::ComputeHost &host, const Source &source, const Scene *scene,
                                 const std::shared_ptr<const particle::GpuParticleCollisionDevicePayload> &previous,
                                 uint64_t revision, std::vector<rhi::ComputeBufferUpdate> updates,
                                 std::vector<rhi::ComputeDispatchDesc> initialize) const
{
    auto payload = std::make_shared<particle::GpuParticleCollisionDevicePayload>();
    payload->worldId = source.identity.worldId;
    payload->simulationEpoch = source.identity.simulationEpoch;
    payload->revision = revision;
    payload->retained = source.retained;
    const bool hasHistory =
        previous && previous->worldId == payload->worldId && previous->simulationEpoch == payload->simulationEpoch;
    std::map<particle::GpuParticleCollisionRowKey, uint32_t> history;
    if (hasHistory)
        for (uint32_t row = 0; row < previous->rows.size(); ++row)
            history.emplace(previous->rows[row], row);

    std::array<std::vector<const InxPhyGpuParticleSurface *>, 2> ordered;
    for (const auto &surface : source.surfaces) {
        const auto *owner = scene ? surface.target.GetGameObject() : nullptr;
        if (!scene || (owner && owner->GetScene() == scene))
            ordered[surface.isStatic ? 0 : 1].push_back(&surface);
    }
    payload->staticCount = static_cast<uint32_t>(ordered[0].size());
    const auto size = ordered[0].size() + ordered[1].size();
    std::vector<InxPhyGpuParticleRow> rows;
    rows.reserve(size);
    payload->rows.reserve(size);
    for (const auto &range : ordered)
        for (const auto *surface : range) {
            payload->rows.push_back(surface->identity);
            auto &record = rows.emplace_back(surface->row);
            if (const auto prior = history.find(surface->identity); prior != history.end())
                record.selection[1] = prior->second;
        }
    payload->topologyRevision =
        hasHistory ? previous->topologyRevision + static_cast<uint64_t>(previous->rows != payload->rows ||
                                                                        previous->staticCount != payload->staticCount)
                   : 1u;

    const auto &poses = source.poses;
    const auto &geometry = source.geometry;
    static_assert(sizeof(InxPhyGpuParticleRow) == 96);
    constexpr auto width = PhysicsGpuPoseSource::RowWords;
    constexpr auto rowBytes = PhysicsGpuPoseSource::RowBytes;
    const auto count = static_cast<uint32_t>(rows.size());
    const uint32_t groups = (count + 63u) / 64u;
    if (rows.size() > UINT32_MAX / 512u || groups > host.device.GetCapabilities().limits.maxComputeWorkgroupCount[0] ||
        (count && (!poses.states || poses.firstRow + poses.bodyCount > UINT32_MAX / width)))
        throw std::invalid_argument("inxphygpu particle export source exceeds its indexed ranges");
    const std::array<ModelWords, 2> parameters{
        {{count, payload->staticCount, static_cast<uint32_t>(poses.firstRow), 0},
         {static_cast<uint32_t>(payload->revision), static_cast<uint32_t>(payload->revision >> 32u),
          static_cast<uint32_t>(payload->topologyRevision), static_cast<uint32_t>(payload->topologyRevision >> 32u)}}};
    const auto upload = [&](std::vector<uint8_t> data) {
        auto buffer = AllocateParticleWords(host, data.size());
        updates.push_back({buffer, 0, std::move(data)});
        return buffer;
    };
    const auto settings = upload(ParticleBytes(&parameters));
    const ModelWords initialStatus{0u, 0u, count, 1u};
    const auto status = upload(ParticleBytes(&initialStatus));
    payload->geometry = geometry.cold;
    payload->rowCount = count;
    using Record = particle::GpuParticleColliderRecord;
    payload->records = AllocateParticleWords(host, sizeof(Record) * uint64_t{std::max(count, 1u)});
    payload->header = AllocateParticleWords(host, sizeof(particle::GpuParticleCollisionSceneHeader));
    payload->gridOffsets = AllocateParticleWords(host, 513u * sizeof(uint32_t));
    if (count) {
        const auto inputs = upload(ParticleBytes(rows.data(), rows.size()));
        const auto previousRecords = hasHistory ? previous->records : upload(std::vector<uint8_t>(sizeof(Record)));
        auto project = ParticleDispatch(m_project, groups,
                                        {{settings, R},
                                         {inputs, R},
                                         {previousRecords, R},
                                         {geometry.localPoses, R},
                                         {geometry.shapes, R},
                                         {poses.states, R},
                                         {payload->records, W},
                                         {status, RW}});
        project.proof->reads[5] = {5, poses.firstRow * rowBytes, uint64_t{poses.bodyCount} * rowBytes};
        project.proof->writes.back() = {7, sizeof(uint32_t), sizeof(uint32_t)};
        initialize.push_back(std::move(project));
    } else {
        updates.push_back({payload->records, 0, std::vector<uint8_t>(sizeof(Record))});
    }
    auto countGrid = ParticleDispatch(
        m_count, 1,
        {{settings, R}, {payload->records, R}, {payload->header, W}, {payload->gridOffsets, W}, {status, RW}});
    countGrid.proof->reads.back() = {4, sizeof(uint32_t), sizeof(uint32_t)};
    initialize.push_back(std::move(countGrid));
    Work result;
    result.payload = std::move(payload);
    result.status = status;
    result.packet =
        rhi::PrepareComputeBatch(host, std::move(updates), std::move(initialize), {{status, 0, sizeof(ModelWords)}});
    result.scatter.kernel = m_scatter;
    result.scatter.buffers = {settings, result.payload->records, result.payload->header, result.payload->gridOffsets};
    return result;
}

std::shared_ptr<rhi::PreparedComputeBatch> InxPhyGpuParticleExport::Work::PrepareGrid(rhi::ComputeHost &host)
{
    ModelWords info{};
    packet->Readback(0)->ReadInto(info.data(), sizeof(info));
    const auto [references, error, count, cells] = info;
    if (error)
        return {};
    if (count != payload->rowCount || cells == 0 || cells > 512 || references > uint64_t{count} * 512u)
        throw std::runtime_error("inxphygpu returned invalid particle collision grid metadata");
    payload->gridRefCount = references;
    payload->gridRefs = AllocateParticleWords(host, uint64_t{std::max(1u, references)} * sizeof(uint32_t));
    std::vector<rhi::ComputeBufferUpdate> updates;
    std::vector<rhi::ComputeDispatchDesc> dispatches;
    if (references) {
        auto &b = scatter.buffers;
        auto dispatch =
            ParticleDispatch(scatter.kernel, (cells + 63u) / 64u,
                             {{b[0], R}, {b[1], R}, {b[2], R}, {b[3], R}, {payload->gridRefs, W}, {status, RW}});
        dispatch.proof->reads.back() = {5, sizeof(uint32_t), sizeof(uint32_t)};
        dispatch.proof->writes.back() = {5, sizeof(uint32_t), sizeof(uint32_t)};
        dispatches.push_back(std::move(dispatch));
    } else {
        const uint32_t invalid = NoModelIndex;
        updates.push_back({payload->gridRefs, 0, ParticleBytes(&invalid)});
    }
    packet =
        rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches), {{status, 0, sizeof(ModelWords)}});
    return packet;
}

std::shared_ptr<particle::GpuParticleCollisionDevicePayload> InxPhyGpuParticleExport::Work::Finish()
{
    ModelWords info{};
    packet->Readback(0)->ReadInto(info.data(), sizeof(info));
    if (info[1] || info[0] != payload->gridRefCount)
        throw std::runtime_error("inxphygpu particle collision data could not be published");
    for (auto *buffer :
         {&payload->records, &payload->header, &payload->gridOffsets, &payload->gridRefs, &payload->geometry}) {
        *buffer = (*buffer)->Freeze();
        (*buffer)->PublishStorage();
    }
    payload->geometryLease = payload->geometry;
    payload->geometryWordRanges = {{0u, payload->geometry->GetByteSize() / sizeof(ModelWords)}};
    return std::move(payload);
}

InxPhyGpuParticleGeometry InxPhyGpuPublishedRayQuery::ParticleGeometry() const
{
    return {m_particlePoses, m_particleShapes, m_particleCold};
}

InxPhyGpuParticleGeometry InxPhyGpuStep::PublishedParticleGeometry() const
{
    return m_publishedRays.ParticleGeometry();
}

InxPhyGpuPairQuery::InxPhyGpuPairQuery(rhi::ComputeHost &host, ShaderCodeTarget target)
{
    StepBuilder builder(host, target, 1);
    const auto data = builder.Slot(9 * 4, BufferUsageFlags::TransferDestination);
    // One EPA worker. Two records per vertex, then its faces and separate horizon indices.
    const auto scratch = builder.Slot((2 * 101 + 1158) * 4);
    const auto horizon = builder.Slot(512);
    const auto result = builder.Slot(3 * 4, BufferUsageFlags::TransferSource);
    auto dispatch = builder.Dispatch("pair_query", "compute_penetration", 1,
                                     {{0, data, R}, {1, scratch, W}, {2, horizon, W}, {3, result, W, true}});
    std::vector<std::shared_ptr<rhi::ComputeBuffer>> buffers;
    for (const auto &slot : builder.slots) {
        auto buffer = rhi::ComputeBuffer::TryCreatePrivate(host, slot.allocation);
        if (!buffer)
            throw std::runtime_error("Could not allocate inxphygpu pair query storage");
        buffers.push_back(std::move(buffer));
    }
    m_dispatch = BindDispatch(std::move(dispatch), buffers);
}

std::shared_ptr<rhi::PreparedComputeBatch> InxPhyGpuPairQuery::Prepare(rhi::ComputeHost &host,
                                                                       const std::vector<ModelWords> &data)
{
    const auto bytes = data.size() * sizeof(ModelWords);
    auto &input = m_dispatch.buffers[0];
    if (bytes > input->GetByteSize()) {
        auto desc = input->GetDesc();
        desc.elementCount = bytes / sizeof(uint32_t);
        auto buffer = rhi::ComputeBuffer::TryCreatePrivate(host, desc);
        if (!buffer)
            throw std::runtime_error("Could not allocate inxphygpu pair geometry");
        input = std::move(buffer);
    }
    auto dispatch = m_dispatch;
    dispatch.proof->reads.front().byteSize = bytes;
    std::vector<rhi::ComputeBufferUpdate> updates(1);
    auto &upload = updates.front();
    upload.buffer = input;
    upload.data.resize(bytes);
    std::memcpy(upload.data.data(), data.data(), bytes);
    return rhi::PrepareComputeBatch(host, std::move(updates), {std::move(dispatch)},
                                    {{m_dispatch.buffers[3], 0, 3 * sizeof(ModelVector)}});
}

PreparedInxPhyGpuStep InxPhyGpuStep::Build(rhi::ComputeHost &host, ShaderCodeTarget target, const InxPhyGpuModel &model,
                                           uint32_t worlds, float timestep, const ModelVector &gravity,
                                           const InxPhyGpuCollisionModel *collision,
                                           InxPhyGpuCollisionCapacity capacity, const InxPhyGpuContactSolver *solver,
                                           InxPhyGpuRayBatchLayout rayLayout, uint32_t bodyCommandCapacity,
                                           bool contactReports, uint32_t substeps, const InxPhyGpuStateHandoff &handoff,
                                           InxPhyGpuSleepSettings sleep)
{
    if (worlds == 0)
        throw std::invalid_argument("inxphygpu step requires a world");
    if (substeps == 0 || (substeps > 1 && bodyCommandCapacity == 0))
        throw std::invalid_argument("inxphygpu fixed substeps require a controlled step layout");

    const auto [nq, nv, nb, nj] = model.dimensions.state;
    if (!model.initialVelocities.empty() && model.initialVelocities.size() != nv)
        throw std::invalid_argument("inxphygpu initial velocities must match the model degrees of freedom");
    const auto nm = model.dimensions.layout[1];
    const auto ns = model.initialStaticPoses.size();
    const auto branches = model.dimensions.layout[2];
    const auto packedMass = model.dimensions.layout[3];
    const auto worldWords = [worlds](uint64_t words) { return words * worlds; };
    StepBuilder b(host, target, worlds);
    const auto dimensions = b.Seed("model", 8, &model.dimensions);
    const auto sceneJointDescriptors =
        model.sceneJoints.empty() ? NoModelIndex : b.Shared("sceneJoints", model.sceneJoints);
    const bool multiple = substeps > 1;
    const float substepDt = timestep / static_cast<float>(substeps);
    const auto stepSizes = b.Seed("timestep", 1, &substepDt, worlds);
    const auto fixedSizes = multiple ? b.Seed("fixedTimestep", 1, &timestep, worlds) : stepSizes;
    uint32_t gravitySlot = NoModelIndex;
    const auto clocks = b.Seed("time", 1, nullptr, worlds);
    const auto positions = b.Seed("qpos", nq, model.initialPositions.data(), worlds);
    const auto velocities =
        b.Seed("qvel", nv, model.initialVelocities.empty() ? nullptr : model.initialVelocities.data(), worlds);
    const auto applied = b.Seed("appliedForce", nv, nullptr, worlds);
    const bool sceneDynamics = !model.sceneBodyMotion.empty();
    const auto sceneMotion = sceneDynamics ? b.Shared("sceneBodyMotion", model.sceneBodyMotion) : NoModelIndex;
    const auto states = bodyCommandCapacity != 0
                            ? b.Slot(worldWords(20 * uint64_t{nb} + 4), BufferUsageFlags::TransferSource)
                            : NoModelIndex;
    BodyControls controls;
    std::optional<BodyControlSlots> controlSlots;
    if (bodyCommandCapacity != 0) {
        controls.worlds = worlds;
        controls.motion.resize(nb, Uncontrolled);
        for (uint32_t index = 1; index < nb; ++index) {
            const auto &body = model.bodies[index];
            if (body.hierarchy[0] != 0 ||
                (body.dofs[1] != 0 && (body.hierarchy[2] != 1 || model.joints[body.hierarchy[1]].indices[0] !=
                                                                     static_cast<uint32_t>(ModelJointType::Free))))
                throw std::invalid_argument("inxphygpu body commands require independent rigid body rows");
            if (body.dofs[1] != 0)
                controls.motion[index] = DynamicControl;
            else if (body.hierarchy[3] != NoModelIndex)
                controls.motion[index] = KinematicControl;
            else if (body.dofs[2] != 0)
                controls.motion[index] = StaticControl;
        }
        controlSlots = BindBodyControls(b, model, bodyCommandCapacity, multiple);
    }
    const bool detectContacts = collision && model.dimensions.layout[0] != 0;
    const bool traceRays = rayLayout.raysPerWorld != 0;
    if (traceRays && model.dimensions.layout[0] != 0 && !collision)
        throw std::invalid_argument("inxphygpu rays require the compiled collision geometry");
    std::vector<ComputeWorkDispatch> rayDispatches;
    const bool publishedRaysEnabled = traceRays && controlSlots;
    std::optional<GeometrySlots> particleGeometry;
    ComputeWorkDispatch publishedGeometry;
    BodyBoundsSlots publishedBodyBounds;
    std::optional<RaySlots> raySlots;
    std::optional<ContactSlots> contactSlots;
    std::optional<ContactResponse> response;
    uint32_t warmstart = NoModelIndex;
    std::optional<ReferenceWeights> reference;
    const auto kinematicsPass = b.dispatches.size();
    struct
    {
        uint32_t bodies, joints, leaves, pose, jointPose, mocap, staticPoses;
    } kinematics{};
    if (nv != 0 || detectContacts || controlSlots) {
        auto &[bodies, joints, leaves, pose, jointPose, mocap, staticPoses] = kinematics;
        bodies = b.Shared("bodies", model.bodies);
        joints = b.Shared("joints", model.joints);
        leaves = b.Shared("branches", model.branches);
        mocap = nm == 0 ? b.Seed("kinematicPoses", 8, nullptr)
                        : b.Seed("kinematicPoses", uint64_t{8} * nm, model.initialKinematicPoses.data(), worlds);
        staticPoses = ns == 0 ? b.Seed("staticPoses", 8, nullptr)
                              : b.Seed("staticPoses", uint64_t{8} * ns, model.initialStaticPoses.data(), worlds);
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
               {7, jointPose, W, nj != 0},
               {13, staticPoses, R}});
        if (detectContacts) {
            if (collision->bounds.size() != model.dimensions.layout[0] || capacity.candidates == 0 ||
                capacity.contactsPerWorld == 0)
                throw std::invalid_argument("inxphygpu collision requires geometry rows and nonzero stream capacities");
            const auto geometry = BindGeometry(b, model, *collision, traceRays || controlSlots);
            if (controlSlots)
                particleGeometry = geometry;
            b.Add("position", "geom_local_to_global", model.dimensions.layout[0],
                  {{0, dimensions, R}, {6, pose, R}, {9, geometry.geometries, R}, {10, geometry.frames, W, true}});
            if (traceRays) {
                if (!publishedRaysEnabled) {
                    if (branches != 0)
                        rayDispatches.push_back(b.dispatches[kinematicsPass]);
                    rayDispatches.push_back(b.dispatches.back());
                }
                raySlots = AddRayWork(b, model, *collision, geometry, rayLayout, rayDispatches, publishedRaysEnabled);
                if (publishedRaysEnabled) {
                    publishedGeometry =
                        b.Dispatch("position", "published_geom_local_to_global", model.dimensions.layout[0],
                                   {{9, geometry.geometries, R},
                                    {10, geometry.frames, W, true},
                                    {11, states, R},
                                    {12, raySlots->settings, R}});
                    std::vector<uint32_t> rows;
                    for (uint32_t row = 1; row < nb; ++row)
                        if (controls.motion[row] == DynamicControl || controls.motion[row] == KinematicControl)
                            rows.push_back(row);
                    publishedBodyBounds =
                        AddPublishedBodyBounds(b, geometry, raySlots->metadata, nb, model.geometries, std::move(rows));
                }
            }
            contactSlots = AddCollision(b, model, *collision, dimensions, geometry, capacity,
                                        solver ? solver->minimumRestitutionSpeed : 0.0f);
        }
    }
    if ((traceRays || controlSlots) && !detectContacts) {
        const InxPhyGpuCollisionModel empty;
        const auto geometry = BindGeometry(b, model, empty, true);
        if (traceRays)
            raySlots = AddRayWork(b, model, empty, geometry, rayLayout, rayDispatches, publishedRaysEnabled);
        if (controlSlots)
            particleGeometry = geometry;
    }
    const auto collisionPasses = b.dispatches.size();
    if (nv != 0) {
        const auto &[bodies, joints, leaves, pose, jointPose, mocap, staticPoses] = kinematics;
        const auto dofs = b.Shared("dofs", model.dofs);
        const auto gravities = gravitySlot = b.Seed("gravity", 4, &gravity, worlds);
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
        warmstart = (contactSlots && solver) || controlSlots ? b.Zero("warmstartAcceleration", worldWords(nv))
                                                             : b.Slot(worldWords(nv));

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
        if (sceneDynamics)
            b.Add("acceleration", "damp_scene_free_motion", nb,
                  {{0, dimensions, R},
                   {5, force, RW},
                   {6, bodies, R},
                   {7, pose, R},
                   {8, velocities, R},
                   {9, solved, RW},
                   {10, stepSizes, R},
                   {11, sceneMotion, R}});
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
                                       controlSlots && (nm != 0 || ns != 0) ? controlSlots->prescribed : NoModelIndex,
                                       bodyVelocity};
        const auto groups = controlSlots ? b.Zero("sceneWakeGroups", worldWords(4 * uint64_t{nb})) : NoModelIndex;
        const SceneWakeWork wake{controlSlots ? &*controlSlots : nullptr, groups, applied};
        if (contactSlots && solver) {
            response = AddContactResponse(b, model, *contactSlots, capacity, *solver, substepDt, dynamics,
                                          std::move(massSolves), contactReports, substeps,
                                          host.device.GetCapabilities().limits.maxStorageBuffersPerStage, wake,
                                          sceneJointDescriptors, handoff.poseTargets);
            reference = {dimensions, dofs, mass, centeredDof, inertialFrame, subtreeCom, response->topology};
            const std::array singleWorld{positions,
                                         pose,
                                         jointPose,
                                         inertialFrame,
                                         subtreeCom,
                                         inertia,
                                         centeredDof,
                                         composite,
                                         mass,
                                         nm == 0 ? NoModelIndex : mocap,
                                         ns == 0 ? NoModelIndex : staticPoses};
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
        } else if (controlSlots) {
            AddSceneWakeWork(b, model, dynamics, wake);
        }
        const auto acceleration = response ? response->acceleration : solved;
        if (!sceneDynamics) {
            std::vector<Use> integrateVelocity{{0, dimensions, R},           {2, stepSizes, R},
                                               {3, acceleration, R},         {4, velocities, RW},
                                               {6, warmstart, W, !response}, {8, acceleration, R}};
            if (response)
                integrateVelocity.push_back({15, response->status, R});
            if (controlSlots) {
                integrateVelocity.push_back({9, dofs, R});
                integrateVelocity.push_back({19, controlSlots->activity, R});
            }
            const auto *entry = controlSlots
                                    ? (response ? "advance_controlled_contact_velocity" : "advance_controlled_velocity")
                                    : (response ? "advance_contact_velocity" : "advance_velocity");
            b.dispatches.push_back(b.Bind("euler", entry, nv, integrateVelocity));
        }
        std::vector<Use> integratePosition{{0, dimensions, R},
                                           {1, joints, R},
                                           {2, stepSizes, R},
                                           {4, velocities, sceneDynamics ? RW : R},
                                           {5, positions, RW}};
        if (response)
            integratePosition.push_back({15, response->status, R});
        if (sceneDynamics) {
            integratePosition.push_back({3, acceleration, R});
            integratePosition.push_back({6, warmstart, W, !response});
            integratePosition.push_back({17, bodies, R});
            integratePosition.push_back({18, sceneMotion, R});
        }
        using Entries = std::array<const char *, 2>;
        const std::array<Entries, 4> positionEntries{{
            {"advance_position", "advance_contact_position"},
            {"advance_scene_position", "advance_scene_contact_position"},
            {"advance_controlled_position", "advance_controlled_contact_position"},
            {"advance_controlled_scene_position", "advance_controlled_scene_contact_position"},
        }};
        if (controlSlots)
            integratePosition.push_back({19, controlSlots->activity, R});
        const auto *entry = positionEntries[2 * controlSlots.has_value() + sceneDynamics][response.has_value()];
        if (nj != 0)
            b.dispatches.push_back(b.Bind("euler", entry, nj, integratePosition));
        if (response && controlSlots && !model.sceneContinuousCollision.empty())
            AddSceneContinuousCollision(b, model, *collision, *particleGeometry, *contactSlots, *solver, substepDt,
                                        dimensions, bodies, pose, positions, velocities, response->status,
                                        *controlSlots);
        if (controlSlots) {
            if (!std::isfinite(sleep.quietTime) || sleep.quietTime <= 0.0f || !std::isfinite(sleep.pointVelocity) ||
                sleep.pointVelocity < 0.0f)
                throw std::invalid_argument("inxphygpu sleep requires positive quiet time and nonnegative point speed");
            const float parameters[2]{sleep.quietTime, sleep.pointVelocity};
            const auto settings = b.Seed("sceneSleepSettings", 2, parameters, worlds);
            b.Add("sleep", "measure_scene_quiet", nb,
                  {{2, groups, RW},
                   {3, velocities, R},
                   {5, applied, R},
                   {27, dimensions, R},
                   {28, bodies, R},
                   {34, controlSlots->activity, RW},
                   {35, settings, R},
                   {36, controlSlots->force, R},
                   {37, stepSizes, R},
                   {38, positions, R},
                   {39, joints, R}});
            std::vector<Use> finish{{2, groups, RW},     {3, velocities, RW}, {22, warmstart, RW},
                                    {27, dimensions, R}, {28, bodies, R},     {34, controlSlots->activity, RW},
                                    {35, settings, R}};
            if (response)
                finish.push_back({33, response->status, R});
            b.dispatches.push_back(
                b.Bind("sleep", response ? "finish_scene_contact_sleep" : "finish_scene_sleep", nb, finish));
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
    ComputeWorkDispatch advance, restore, restoreActivity, drive;
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
            throw std::length_error("inxphygpu fixed step checkpoint exceeds the device address range");
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

    if (response && controlSlots) {
        const auto backup = b.Slot(worldWords(4 * uint64_t{nb}));
        const auto activity = controlSlots->activity;
        begin.push_back(b.Dispatch("sleep", "checkpoint_scene_activity", nb,
                                   {{34, activity, R}, {27, dimensions, R}, {32, backup, W, true}}));
        restoreActivity =
            b.Dispatch("sleep", "restore_scene_activity", nb,
                       {{34, activity, RW}, {27, dimensions, R}, {32, backup, R}, {33, response->status, R}});
    }
    ComputeWorkDispatch applyCommands, editCommands, finishCommands, deviceMotion, deviceImpulse, publishPose,
        publishBodies, publishBodiesWithStatus;
    const auto firstBodyPublication = b.initialize.size();
    if (controlSlots) {
        const auto &[spans, commands, force, saved, prescribed, fixedSteps, activity, pending] = *controlSlots;
        const auto &[bodies, joints, leaves, pose, jointPose, mocap, staticPoses] = kinematics;
        std::vector<Use> commandBindings{{0, dimensions, R},  {1, bodies, R},        {2, joints, R},
                                         {3, spans, R},       {4, commands, R},      {5, positions, RW},
                                         {6, velocities, RW}, {7, mocap, RW},        {12, prescribed, RW},
                                         {15, fixedSteps, R}, {16, staticPoses, RW}, {23, activity, RW}};
        if (sceneDynamics)
            commandBindings.push_back({21, sceneMotion, R});
        commandBindings.push_back({24, pending, RW});
        editCommands = b.Bind("body_controls", sceneDynamics ? "edit_scene_body_state" : "edit_body_state",
                              bodyCommandCapacity, commandBindings);
        commandBindings.back().access = R;
        commandBindings.insert(commandBindings.end(), {{8, force, RW}, {11, saved, W}, {22, fixedSizes, R}});
        applyCommands = b.Bind("body_controls", sceneDynamics ? "apply_scene_body_commands" : "apply_body_commands",
                               bodyCommandCapacity, commandBindings);
        deviceImpulse = b.Dispatch("body_controls", "apply_device_body_impulses", 1,
                                   {{0, dimensions, R},
                                    {1, bodies, R},
                                    {2, joints, R},
                                    {5, positions, R},
                                    {6, velocities, RW},
                                    {23, activity, RW},
                                    {29, spans, R},
                                    {30, commands, R}});
        deviceMotion = b.Dispatch("body_controls", "gather_device_body_motion", 1,
                                  {{0, dimensions, R},
                                   {1, bodies, R},
                                   {2, joints, R},
                                   {5, positions, R},
                                   {6, velocities, R},
                                   {23, activity, RW},
                                   {29, spans, R},
                                   {31, saved, W}});
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
                                {15, fixedSteps, R},
                                {16, staticPoses, RW},
                                {22, fixedSizes, R}});
        finishCommands = response ? b.Dispatch("body_controls", "finish_contact_body_commands", bodyCommandCapacity,
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
                                                {12, prescribed, RW},
                                                {16, staticPoses, W},
                                                {24, pending, W}})
                                  : b.Dispatch("body_controls", "finish_body_commands", bodyCommandCapacity,
                                               {{0, dimensions, R},
                                                {1, bodies, R},
                                                {2, joints, R},
                                                {3, spans, R},
                                                {8, force, RW},
                                                {24, pending, W}});
        if (branches != 0) {
            publishPose = b.dispatches[kinematicsPass];
            b.initialize.push_back(publishPose);
        }
        std::vector<Use> stateBindings{{0, dimensions, R},    {1, bodies, R},      {6, velocities, R},
                                       {9, clocks, R},        {12, prescribed, R}, {13, pose, R},
                                       {14, states, W, true}, {15, fixedSteps, R}, {23, activity, R}};
        publishBodies = b.Bind("body_controls", "publish_body_states", nb, stateBindings);
        b.initialize.push_back(publishBodies);
        if (response) {
            stateBindings.insert(stateBindings.begin() + 4, {10, response->status, R});
            publishBodiesWithStatus = b.Bind("body_controls", "publish_contact_body_states", nb, stateBindings);
        }
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
    if (handoff.states) {
        if (!controlSlots || handoff.addresses.size() != worldWords(nb))
            throw std::invalid_argument("inxphygpu topology handoff requires one mapping per controlled body");
        const auto addresses = b.Shared("topologyHandoff", handoff.addresses);
        // The frozen source is attached to this initialization dispatch below.
        b.initialize.insert(b.initialize.begin() + firstBodyPublication,
                            b.Dispatch("body_controls", "import_body_states", nb,
                                       {{0, dimensions, R},
                                        {1, kinematics.bodies, R},
                                        {2, kinematics.joints, R},
                                        {5, positions, RW},
                                        {6, velocities, RW},
                                        {7, kinematics.mocap, RW},
                                        {12, controlSlots->prescribed, RW},
                                        {16, kinematics.staticPoses, RW},
                                        {18, clocks, RW},
                                        {19, controlSlots->fixedSteps, RW},
                                        {20, addresses, R},
                                        {23, controlSlots->activity, RW},
                                        {17, positions, R}}));
        if (handoff.pendingWrenches)
            b.initialize.insert(b.initialize.begin() + firstBodyPublication + 1,
                                b.Dispatch("body_controls", "import_pending_wrenches", nb,
                                           {{0, dimensions, R},
                                            {1, kinematics.bodies, R},
                                            {20, addresses, R},
                                            {24, controlSlots->pendingWrenches, W},
                                            {25, positions, R}}));
    }
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
    for (auto &dispatch : b.initialize)
        initialDispatches.push_back(BindDispatch(std::move(dispatch), initializationBuffers));
    if (handoff.states) {
        auto &importState = initialDispatches[firstBodyPublication];
        importState.buffers.back() = handoff.states;
        importState.proof->reads.back().byteSize = handoff.states->GetByteSize();
        if (handoff.pendingWrenches) {
            auto &importWrenches = initialDispatches[firstBodyPublication + 1];
            importWrenches.buffers.back() = handoff.pendingWrenches;
            importWrenches.proof->reads.back().byteSize = handoff.pendingWrenches->GetByteSize();
        }
    }
    auto initialize = rhi::PrepareComputeBatch(host, std::move(initial), std::move(initialDispatches));
    InxPhyGpuContactStream contacts;
    if (contactSlots)
        contacts = {buffers[contactSlots->candidateCount],
                    buffers[contactSlots->counts],
                    buffers[contactSlots->records],
                    response ? buffers[response->status] : nullptr,
                    response ? buffers[response->topology] : nullptr,
                    buffers[contactSlots->narrowphaseStatus]};
    InxPhyGpuRayBuffers rays;
    std::vector<rhi::ComputeDispatchDesc> rayWork;
    InxPhyGpuPublishedRayQuery publishedRays;
    if (particleGeometry) {
        publishedRays.m_particlePoses = buffers[particleGeometry->geometries];
        publishedRays.m_particleShapes = buffers[particleGeometry->shapes];
        publishedRays.m_particleCold = buffers[particleGeometry->data];
    }
    if (raySlots) {
        rays = {buffers[raySlots->inputs], publishedRaysEnabled ? nullptr : buffers[raySlots->hits]};
        rayWork.reserve(rayDispatches.size());
        for (auto &dispatch : rayDispatches)
            rayWork.push_back(BindDispatch(std::move(dispatch), buffers));
        if (publishedRaysEnabled) {
            publishedRays.m_inputs = rays.inputs;
            publishedRays.m_settings = buffers[raySlots->settings];
            publishedRays.m_inputSettings = buffers[raySlots->inputSettings];
            publishedRays.m_outputSettings = buffers[raySlots->outputSettings];
            publishedRays.m_pack = BindDispatch(std::move(raySlots->pack), buffers);
            publishedRays.m_project = BindDispatch(std::move(raySlots->project), buffers);
            publishedRays.m_traceAll = BindDispatch(std::move(raySlots->traceAll), buffers);
            if (raySlots->closest.kernel) {
                publishedRays.m_closest = BindDispatch(std::move(raySlots->closest), buffers);
                publishedRays.m_closestRequest = buffers[raySlots->closestSettings];
                publishedRays.m_closestResult = buffers[raySlots->closestResult];
                publishedRays.m_overlap = BindDispatch(std::move(raySlots->overlap), buffers);
                publishedRays.m_overlapRequest = buffers[raySlots->overlapRequest];
                publishedRays.m_overlapMask = buffers[raySlots->overlapMask];
            }
            publishedRays.m_geometryCount = model.dimensions.layout[0];
            publishedRays.m_bodyCount = nb;
            publishedRays.m_capacity = rayLayout.raysPerWorld;
            publishedRays.m_rayWork = std::move(rayWork);
            if (publishedGeometry.kernel)
                publishedRays.m_geometry = BindDispatch(std::move(publishedGeometry), buffers);
            if (publishedBodyBounds.select.kernel) {
                auto &bounds = publishedBodyBounds;
                publishedRays.m_selectBounds = BindDispatch(std::move(bounds.select), buffers);
                publishedRays.m_boundsRequest = buffers[bounds.request];
                publishedRays.m_boundsMask = buffers[bounds.mask];
                publishedRays.m_bodyBoundsRows = std::make_shared<const std::vector<uint32_t>>(std::move(bounds.rows));
            }
        }
    }
    if (controlSlots) {
        controls.streams = {buffers[controlSlots->commands], buffers[controlSlots->spans], buffers[controlSlots->saved],
                            bodyCommandCapacity};
        controls.states = buffers[states];
        controls.activity = buffers[controlSlots->activity];
        controls.pendingWrenches = buffers[controlSlots->pendingWrenches];
        controls.apply = BindDispatch(std::move(applyCommands), buffers);
        controls.edit = BindDispatch(std::move(editCommands), buffers);
        controls.finish = BindDispatch(std::move(finishCommands), buffers);
        controls.deviceMotion = BindDispatch(std::move(deviceMotion), buffers);
        controls.deviceImpulse = BindDispatch(std::move(deviceImpulse), buffers);
        if (publishPose.kernel)
            controls.pose = BindDispatch(std::move(publishPose), buffers);
        controls.publish = BindDispatch(std::move(publishBodies), buffers);
        if (publishBodiesWithStatus.kernel)
            controls.publishWithStatus = BindDispatch(std::move(publishBodiesWithStatus), buffers);
    }
    ContactReports reports;
    if (response && response->capture)
        reports = {buffers[response->reportRecords],
                   buffers[multiple ? response->status : contactSlots->counts],
                   BindDispatch(std::move(*response->capture), buffers),
                   multiple ? worlds * sizeof(uint32_t) : 0,
                   worldWords(substeps) * sizeof(uint32_t),
                   capacity.contactsPerWorld};
    SceneJoints sceneJoints;
    if (sceneJointDescriptors != NoModelIndex) {
        sceneJoints.descriptors = buffers[sceneJointDescriptors];
        if (controlSlots)
            sceneJoints.motionKernel = b.Kernel("spatial_constraints", "read_scene_joint_motion", {1, 19, 20, 21});
        if (response) {
            sceneJoints.parameters = buffers[response->jointParameters];
            sceneJoints.stride = static_cast<uint32_t>(model.initialSceneJointParameters.size());
        }
    }
    FixedSubsteps schedule;
    schedule.count = substeps;
    schedule.collisionPasses = collisionPasses;
    for (auto &dispatch : begin)
        schedule.begin.push_back(BindDispatch(std::move(dispatch), buffers));
    if (restoreActivity.kernel)
        schedule.restoreActivity = BindDispatch(std::move(restoreActivity), buffers);
    if (multiple) {
        if (drive.kernel)
            schedule.drive = BindDispatch(std::move(drive), buffers);
        if (advance.kernel)
            schedule.advance = BindDispatch(std::move(advance), buffers);
        if (restore.kernel)
            schedule.restore = BindDispatch(std::move(restore), buffers);
        schedule.clock = BindDispatch(std::move(clock), buffers);
    }
    FixedInputs fixedInputs{
        buffers[fixedSizes], buffers[stepSizes], gravitySlot != NoModelIndex ? buffers[gravitySlot] : nullptr,
        response ? buffers[response->settings] : nullptr, response ? response->worldDefaults : ModelWords{}};
    auto step = std::unique_ptr<InxPhyGpuStep>(new InxPhyGpuStep(
        work, BindDispatch(std::move(snapshot), buffers), positions, velocities, applied, std::move(contacts),
        std::move(rays), std::move(rayWork), std::move(controls), std::move(reports), std::move(schedule),
        std::move(sceneJoints), std::move(publishedRays), std::move(fixedInputs)));
    return {std::move(step), std::move(initialize)};
}

InxPhyGpuStep::InxPhyGpuStep(std::shared_ptr<rhi::ComputeWorkInstance> work, rhi::ComputeDispatchDesc snapshot,
                             uint32_t positions, uint32_t velocities, uint32_t appliedForces,
                             InxPhyGpuContactStream contacts, InxPhyGpuRayBuffers rays,
                             std::vector<rhi::ComputeDispatchDesc> rayWork, BodyControls controls,
                             ContactReports reports, FixedSubsteps substeps, SceneJoints sceneJoints,
                             InxPhyGpuPublishedRayQuery publishedRays, FixedInputs fixedInputs)
    : m_work(std::move(work)), m_snapshot(std::move(snapshot)), m_positions(positions), m_velocities(velocities),
      m_appliedForces(appliedForces), m_contacts(std::move(contacts)), m_rays(std::move(rays)),
      m_rayWork(std::move(rayWork)), m_controls(std::move(controls)), m_reports(std::move(reports)),
      m_substeps(std::move(substeps)), m_sceneJoints(std::move(sceneJoints)), m_publishedRays(std::move(publishedRays)),
      m_fixedInputs(std::move(fixedInputs))
{
}

std::shared_ptr<rhi::PreparedComputeBatch> InxPhyGpuStep::Prepare(bool publish,
                                                                  std::vector<rhi::ComputeBufferUpdate> updates) const
{
    auto dispatches = m_substeps.begin;
    std::vector<rhi::ComputeReplayRange> replays;
    AppendFixedWork(dispatches, replays, 0, false, m_controls.streams);
    std::vector<rhi::ComputeBufferRead> reads;
    if (publish) {
        dispatches.push_back(m_snapshot);
        const auto &snapshot = m_snapshot.buffers[4]; // Binding 4 is the snapshot output in both entries.
        reads.push_back({snapshot, 0, snapshot->GetByteSize()});
    }
    auto host = m_work->GetRecipe().GetHost();
    auto packet = rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches), std::move(reads));
    packet->SetDispatchReplay(std::move(replays));
    return packet;
}

void InxPhyGpuStep::AppendFixedWork(std::vector<rhi::ComputeDispatchDesc> &dispatches,
                                    std::vector<rhi::ComputeReplayRange> &replays, uint32_t spans, bool captureContacts,
                                    const BodyCommandStreams &streams, std::vector<SubstepStages> stages) const
{
    using Stage = PhysicsParticipantStage;
    auto drive = m_substeps.drive;
    const bool driving = spans != 0 && drive.kernel;
    if (driving) {
        BindBodyCommandStreams(drive, streams, 0, spans);
        drive.groupCountX = (spans - 1) / 64 + 1;
        LinearGrid(drive);
    }
    const bool reporting = captureContacts && m_reports.records;
    const auto numericalDispatches = m_work->GetRecipe().GetDispatches().size();
    const auto width = numericalDispatches + driving + reporting + (driving || reporting);
    auto stepWidth = dispatches.size() + m_substeps.count * width + 2;
    for (const auto &substep : stages)
        for (const auto &phase : substep)
            stepWidth += phase.size();
    dispatches.reserve(stepWidth);
    replays.reserve(m_substeps.count * (stages.empty() ? 1 : 2));
    const auto append = [&](uint32_t substep, Stage stage) {
        auto &phase = stages[substep][static_cast<size_t>(stage)];
        dispatches.insert(dispatches.end(), std::make_move_iterator(phase.begin()),
                          std::make_move_iterator(phase.end()));
    };
    for (uint32_t substep = 0; substep < m_substeps.count; ++substep) {
        if (driving)
            dispatches.push_back(drive);
        if (!stages.empty())
            append(substep, Stage::Predict);
        const auto solve = dispatches.size();
        m_work->AppendDispatches(dispatches);
        const auto middle = m_substeps.collisionPasses;
        if (!stages.empty() && (!stages[substep][static_cast<size_t>(Stage::Prepare)].empty() ||
                                !stages[substep][static_cast<size_t>(Stage::Exchange)].empty())) {
            const auto end = dispatches.size();
            append(substep, Stage::Prepare);
            append(substep, Stage::Exchange);
            std::rotate(dispatches.begin() + solve + middle, dispatches.begin() + end, dispatches.end());
            if (middle != 0)
                replays.push_back({solve, middle, m_collisionReplay});
            if (middle != numericalDispatches)
                replays.push_back(
                    {solve + middle + dispatches.size() - end, numericalDispatches - middle, m_dynamicsReplay});
        } else {
            replays.push_back({solve, numericalDispatches, m_fixedReplay});
        }
        if (reporting)
            dispatches.push_back(m_reports.capture);
        if (!stages.empty())
            append(substep, Stage::Accept);
        if ((driving || reporting) && substep + 1 < m_substeps.count)
            dispatches.push_back(m_substeps.advance);
    }
    if (m_substeps.restore.kernel)
        dispatches.push_back(m_substeps.restore);
    if (m_substeps.restoreActivity.kernel)
        dispatches.push_back(m_substeps.restoreActivity);
    if (m_substeps.clock.kernel)
        dispatches.push_back(m_substeps.clock);
}

std::shared_ptr<rhi::PreparedComputeBatch>
InxPhyGpuStep::PrepareRays(bool publish, std::vector<rhi::ComputeBufferUpdate> updates) const
{
    if (!m_rays.hits)
        throw std::logic_error("inxphygpu step has no ray query layout");
    std::vector<rhi::ComputeBufferRead> reads;
    if (publish)
        reads.push_back({m_rays.hits, 0, m_rays.hits->GetByteSize()});
    auto host = m_work->GetRecipe().GetHost();
    return rhi::PrepareComputeBatch(host, std::move(updates), m_rayWork, std::move(reads));
}

InxPhyGpuPreparedRayBatch
InxPhyGpuStep::PreparePublishedRays(const PhysicsGpuPoseSource &source, uint32_t rayCount, bool publish,
                                    std::vector<rhi::ComputeBufferUpdate> updates,
                                    const InxPhyGpuDeviceRays *deviceRays, const PhysicsDeviceRayOutputs *outputs,
                                    uint32_t allHitCapacity, const std::vector<InxPhyGpuRay> *hostRays) const
{
    auto host = m_work->GetRecipe().GetHost();
    return m_publishedRays.Prepare(host, source, rayCount, publish, std::move(updates), deviceRays, outputs,
                                   allHitCapacity, hostRays);
}

void InxPhyGpuPublishedRayQuery::AppendGeometry(std::vector<rhi::ComputeDispatchDesc> &dispatches,
                                                const PhysicsGpuPoseSource &source) const
{
    if (!m_geometry.kernel)
        return;
    constexpr auto wordsPerRow = PhysicsGpuPoseSource::RowWords;
    const uint64_t maxRows = UINT32_MAX / wordsPerRow;
    constexpr uint64_t rowBytes = PhysicsGpuPoseSource::RowBytes;
    if (!source.states || source.bodyCount != m_bodyCount || source.firstRow > maxRows ||
        source.bodyCount > maxRows - source.firstRow ||
        (source.firstRow + source.bodyCount) * rowBytes > source.states->GetByteSize())
        throw std::invalid_argument("inxphygpu query requires a complete published body slot range");
    auto &geometry = dispatches.emplace_back(m_geometry);
    geometry.groupCountY = 1;
    geometry.buffers[2] = source.states;
    geometry.proof->reads[1] = {2, source.firstRow * rowBytes, source.bodyCount * rowBytes};
    geometry.proof->writes.front().byteSize = 4 * sizeof(ModelVector) * m_geometryCount;
}

InxPhyGpuPreparedBodyBounds InxPhyGpuStep::PreparePublishedBodyBounds(const PhysicsGpuPoseSource &source,
                                                                      const ModelVector &minimum,
                                                                      const ModelVector &maximum, uint32_t layers,
                                                                      bool triggers) const
{
    auto host = m_work->GetRecipe().GetHost();
    return m_publishedRays.PrepareBodyBounds(host, source, minimum, maximum, layers, triggers, {});
}

InxPhyGpuPreparedBodyBounds
InxPhyGpuPublishedRayQuery::PrepareBodyBounds(rhi::ComputeHost &host, const PhysicsGpuPoseSource &source,
                                              const ModelVector &minimum, const ModelVector &maximum, uint32_t layers,
                                              bool triggers, std::vector<rhi::ComputeBufferUpdate> updates) const
{
    if (!m_selectBounds.kernel)
        return {};
    const ModelWords settings{m_geometryCount, 0, 0, static_cast<uint32_t>(source.firstRow)};
    auto &frame = updates.emplace_back(rhi::ComputeBufferUpdate{m_settings, 0, std::vector<uint8_t>(sizeof(settings))});
    std::memcpy(frame.data.data(), settings.data(), sizeof(settings));
    const ModelWords filter{static_cast<uint32_t>(m_bodyBoundsRows->size()), layers, static_cast<uint32_t>(triggers),
                            0};
    auto &request = updates.emplace_back(
        rhi::ComputeBufferUpdate{m_boundsRequest, 0, std::vector<uint8_t>(2 * sizeof(ModelVector) + sizeof(filter))});
    std::memcpy(request.data.data(), minimum.data(), sizeof(ModelVector));
    std::memcpy(request.data.data() + sizeof(ModelVector), maximum.data(), sizeof(ModelVector));
    std::memcpy(request.data.data() + 2 * sizeof(ModelVector), filter.data(), sizeof(filter));
    std::vector<rhi::ComputeDispatchDesc> dispatches;
    AppendGeometry(dispatches, source);
    dispatches.push_back(m_selectBounds);
    return {rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches),
                                     {{m_boundsMask, 0, m_boundsMask->GetByteSize()}}),
            m_bodyBoundsRows};
}

std::shared_ptr<rhi::PreparedComputeBatch>
InxPhyGpuStep::PreparePublishedClosestPoint(const PhysicsGpuPoseSource &source, const ModelVector &point,
                                            uint32_t first, uint32_t end, bool closedMesh) const
{
    auto host = m_work->GetRecipe().GetHost();
    return m_publishedRays.PrepareClosestPoint(host, source, point, first, end, closedMesh, {});
}

std::shared_ptr<rhi::PreparedComputeBatch>
InxPhyGpuPublishedRayQuery::PrepareClosestPoint(rhi::ComputeHost &host, const PhysicsGpuPoseSource &source,
                                                const ModelVector &point, uint32_t first, uint32_t end, bool closedMesh,
                                                std::vector<rhi::ComputeBufferUpdate> updates) const
{
    const ModelWords settings{m_geometryCount, 0, 0, static_cast<uint32_t>(source.firstRow)};
    auto &frame = updates.emplace_back(rhi::ComputeBufferUpdate{m_settings, 0, std::vector<uint8_t>(sizeof(settings))});
    std::memcpy(frame.data.data(), settings.data(), sizeof(settings));
    const ModelWords target{first, end, static_cast<uint32_t>(closedMesh), 0};
    auto &request = updates.emplace_back(
        rhi::ComputeBufferUpdate{m_closestRequest, 0, std::vector<uint8_t>(sizeof(point) + sizeof(target))});
    std::memcpy(request.data.data(), point.data(), sizeof(point));
    std::memcpy(request.data.data() + sizeof(point), target.data(), sizeof(target));
    std::vector<rhi::ComputeDispatchDesc> dispatches;
    AppendGeometry(dispatches, source);
    dispatches.push_back(m_closest);
    return rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches),
                                    {{m_closestResult, 0, sizeof(ModelVector)}});
}

std::shared_ptr<rhi::PreparedComputeBatch>
InxPhyGpuPublishedRayQuery::PrepareOverlap(rhi::ComputeHost &host, const PhysicsGpuPoseSource &source,
                                           const InxPhyGpuQueryVolume &query,
                                           std::vector<rhi::ComputeBufferUpdate> updates) const
{
    if (!m_overlap.kernel || !query.layers)
        return {};
    const ModelWords settings{m_geometryCount, 0, 0, static_cast<uint32_t>(source.firstRow)};
    auto &frame = updates.emplace_back(rhi::ComputeBufferUpdate{m_settings, 0, std::vector<uint8_t>(sizeof(settings))});
    std::memcpy(frame.data.data(), settings.data(), sizeof(settings));
    const ModelWords filter{m_geometryCount, query.layers, static_cast<uint32_t>(query.triggers), query.type};
    auto &request = updates.emplace_back(rhi::ComputeBufferUpdate{
        m_overlapRequest, 0, std::vector<uint8_t>(sizeof(query.pose) + sizeof(query.size) + sizeof(filter))});
    std::memcpy(request.data.data(), &query.pose, sizeof(query.pose));
    std::memcpy(request.data.data() + sizeof(query.pose), query.size.data(), sizeof(query.size));
    std::memcpy(request.data.data() + sizeof(query.pose) + sizeof(query.size), filter.data(), sizeof(filter));
    std::vector<rhi::ComputeDispatchDesc> dispatches;
    AppendGeometry(dispatches, source);
    dispatches.push_back(m_overlap);
    return rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches),
                                    {{m_overlapMask, 0, m_overlapMask->GetByteSize()}});
}

std::shared_ptr<rhi::PreparedComputeBatch>
InxPhyGpuStep::PreparePublishedOverlap(const PhysicsGpuPoseSource &source, const InxPhyGpuQueryVolume &volume) const
{
    auto host = m_work->GetRecipe().GetHost();
    return m_publishedRays.PrepareOverlap(host, source, volume, {});
}

std::shared_ptr<rhi::PreparedComputeBatch>
InxPhyGpuStep::PreparePublishedShapeCast(ShaderCodeTarget target, const PhysicsGpuPoseSource &source,
                                         const InxPhyGpuQueryVolume &volume, const ModelVector &direction,
                                         float maximum, const std::vector<ModelWords> &exclusions,
                                         const std::function<uint32_t(uint32_t)> &bodyId) const
{
    auto host = m_work->GetRecipe().GetHost();
    return m_publishedRays.PrepareShapeCast(host, target, source, volume, direction, maximum, exclusions, bodyId, {});
}

std::shared_ptr<rhi::PreparedComputeBatch> InxPhyGpuPublishedRayQuery::PrepareShapeCast(
    rhi::ComputeHost &host, ShaderCodeTarget target, const PhysicsGpuPoseSource &source,
    const InxPhyGpuQueryVolume &volume, const ModelVector &direction, float maximum,
    const std::vector<ModelWords> &exclusions, const std::function<uint32_t(uint32_t)> &bodyId,
    std::vector<rhi::ComputeBufferUpdate> updates) const
{
    struct Parameters
    {
        ModelPose originLimit;
        ModelVector size, direction;
        ModelWords filter;
    };
    if (!m_overlap.kernel || !volume.layers)
        return {};
    Parameters request{volume.pose,
                       volume.size,
                       direction,
                       {m_geometryCount, volume.layers,
                        static_cast<uint32_t>(volume.triggers) | (exclusions.empty() ? 0u : 2u), volume.type}};
    request.originLimit.position[3] = maximum;
    const auto allocate = [&](uint64_t words, BufferUsageFlags usage = BufferUsageFlags::None) {
        auto buffer = ComputeBuffer::TryCreatePrivate(host, {words, ComputeScalarType::UInt32, 1, usage});
        if (!buffer)
            throw std::runtime_error("Could not allocate inxphygpu shape cast storage");
        return buffer;
    };
    // Stable body IDs follow the request and the optional per-row exclusion bits.
    const uint64_t identityOffset = sizeof(Parameters) + sizeof(ModelWords) * ((uint64_t{m_geometryCount} + 127) / 128);
    const uint64_t identityBytes = sizeof(ModelWords) * ((uint64_t{m_geometryCount} + 3) / 4);
    if (!m_shapeCast.kernel) {
        constexpr uint32_t workers = 16, records = 2 * 101 + 1158, horizon = 512;
        const auto result =
            allocate(sizeof(InxPhyGpuShapeCastHit) / sizeof(uint32_t), BufferUsageFlags::TransferSource);
        ComputeDispatchDesc dispatch;
        // The accepted geometry prefix is shared by overlaps and casts.
        dispatch.buffers.assign(m_overlap.buffers.begin(), m_overlap.buffers.begin() + 4);
        dispatch.buffers.emplace_back();
        dispatch.buffers.push_back(result);
        dispatch.buffers.push_back(allocate(workers * records * 4));
        dispatch.buffers.push_back(allocate(workers * horizon));
        dispatch.bufferAccesses = {R, R, R, R, R, W, W, W};
        dispatch.proof.emplace();
        dispatch.proof->reads.assign(m_overlap.proof->reads.begin(), m_overlap.proof->reads.begin() + 4);
        dispatch.proof->reads.push_back({4, identityOffset, identityBytes});
        dispatch.proof->reads.push_back({4, 0, sizeof(Parameters)});
        dispatch.proof->writes.push_back({5, 0, result->GetByteSize()});
        StepBuilder builder(host, target, 1);
        dispatch.kernel = builder.Kernel("shape_cast", "cast_scene_shapes", {0, 1, 2, 3, 4, 5, 6, 7});
        m_shapeCast = std::move(dispatch);
    }
    // Clone worlds share numerical buffers, but each owns different public IDs.
    auto &requestBuffer = m_shapeCastRequests[source.firstRow];
    if (!requestBuffer)
        requestBuffer =
            allocate((identityOffset + identityBytes) / sizeof(uint32_t), BufferUsageFlags::TransferDestination);
    if (!requestBuffer->GetLastWriteSubmission().IsValid()) {
        auto &identity = updates.emplace_back(
            ComputeBufferUpdate{requestBuffer, identityOffset, std::vector<uint8_t>(identityBytes)});
        for (uint32_t row = 0; row < m_geometryCount; ++row) {
            const uint32_t id = bodyId(row);
            std::memcpy(identity.data.data() + row * sizeof(id), &id, sizeof(id));
        }
    }
    auto cast = m_shapeCast;
    cast.buffers[4] = requestBuffer;
    const size_t requestBytes = sizeof(request) + exclusions.size() * sizeof(ModelWords);
    cast.proof->reads.back().byteSize = requestBytes;
    auto &parameters =
        updates.emplace_back(ComputeBufferUpdate{cast.buffers[4], 0, std::vector<uint8_t>(requestBytes)});
    std::memcpy(parameters.data.data(), &request, sizeof(request));
    if (!exclusions.empty())
        std::memcpy(parameters.data.data() + sizeof(request), exclusions.data(),
                    exclusions.size() * sizeof(ModelWords));
    const ModelWords settings{m_geometryCount, 0, 0, static_cast<uint32_t>(source.firstRow)};
    auto &frame = updates.emplace_back(ComputeBufferUpdate{m_settings, 0, std::vector<uint8_t>(sizeof(settings))});
    std::memcpy(frame.data.data(), settings.data(), sizeof(settings));
    std::vector<rhi::ComputeDispatchDesc> dispatches;
    AppendGeometry(dispatches, source);
    const auto result = cast.buffers[5];
    dispatches.push_back(std::move(cast));
    return rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches),
                                    {{result, 0, sizeof(InxPhyGpuShapeCastHit)}});
}

std::vector<uint32_t> ReadInxPhyGpuSelection(const rhi::PreparedComputeBatch &packet, uint32_t count)
{
    std::vector<uint32_t> mask((uint64_t{count} + 31) / 32);
    packet.Readback(0)->ReadInto(mask.data(), mask.size() * sizeof(uint32_t));
    std::vector<uint32_t> matches;
    for (size_t word = 0; word < mask.size(); ++word) {
        auto row = word * 32;
        for (auto bits = mask[word]; bits != 0; bits >>= 1, ++row)
            if (bits & 1u)
                matches.push_back(static_cast<uint32_t>(row));
    }
    return matches;
}

std::vector<uint32_t> InxPhyGpuPreparedBodyBounds::ReadMatchingRows() const
{
    if (!packet)
        return {};
    auto matches = ReadInxPhyGpuSelection(*packet, static_cast<uint32_t>(rows->size()));
    for (auto &index : matches)
        index = (*rows)[index];
    return matches;
}

InxPhyGpuPreparedRayBatch
InxPhyGpuPublishedRayQuery::Prepare(rhi::ComputeHost &host, const PhysicsGpuPoseSource &source, uint32_t rayCount,
                                    bool publish, std::vector<rhi::ComputeBufferUpdate> updates,
                                    const InxPhyGpuDeviceRays *deviceRays, const PhysicsDeviceRayOutputs *outputs,
                                    uint32_t allHitCapacity, const std::vector<InxPhyGpuRay> *hostRays) const
{
    if (!m_settings || rayCount == 0 || (rayCount > m_capacity && !hostRays && !deviceRays))
        throw std::invalid_argument("inxphygpu published ray batch exceeds the scene query layout");
    const auto inputBytes = uint64_t{rayCount} * sizeof(InxPhyGpuRay);
    auto input = m_inputs;
    if (rayCount > m_capacity) {
        if (!m_inputScratch || m_inputScratch->GetByteSize() < inputBytes)
            m_inputScratch = std::make_shared<rhi::ComputeBuffer>(
                host, rhi::ComputeBufferDesc{inputBytes / sizeof(uint32_t), rhi::ComputeScalarType::UInt32, 1,
                                             BufferUsageFlags::Storage | BufferUsageFlags::TransferDestination});
        input = m_inputScratch;
    }
    if (hostRays) {
        auto &upload = updates.emplace_back(rhi::ComputeBufferUpdate{input, 0, std::vector<uint8_t>(inputBytes)});
        std::memcpy(upload.data.data(), hostRays->data(), upload.data.size());
    }
    const ModelWords settings{m_geometryCount, rayCount, allHitCapacity ? allHitCapacity : 1u,
                              static_cast<uint32_t>(source.firstRow)};
    auto &params =
        updates.emplace_back(rhi::ComputeBufferUpdate{m_settings, 0, std::vector<uint8_t>(sizeof(settings))});
    std::memcpy(params.data.data(), settings.data(), params.data.size());

    std::vector<rhi::ComputeDispatchDesc> dispatches;
    dispatches.reserve(m_rayWork.size() + size_t(bool(m_geometry.kernel)) + size_t(deviceRays != nullptr) +
                       size_t(outputs != nullptr));
    const auto scalarOffset = [&host](const PhysicsDeviceBufferRange &view, uint64_t bytes, ComputeScalarType scalar) {
        const auto end = view.buffer ? std::min(view.byteEnd, view.buffer->GetByteSize()) : 0u;
        if (!view.buffer || !host.SharesServicesWith(view.buffer->GetHost()) ||
            view.buffer->GetDesc().scalarType != scalar || view.byteOffset % sizeof(uint32_t) != 0 ||
            view.byteOffset > end || bytes > end - view.byteOffset ||
            (view.byteOffset + bytes) / sizeof(uint32_t) > UINT32_MAX)
            throw std::invalid_argument("Device raycast requires a complete scalar-aligned buffer range");
        return static_cast<uint32_t>(view.byteOffset / sizeof(uint32_t));
    };
    if (deviceRays) {
        const auto &rays = *deviceRays;
        if (!std::isfinite(rays.maxDistance) || rays.maxDistance <= 0)
            throw std::invalid_argument("Device raycast requires a positive finite maximum distance");
        const auto vectorBytes = uint64_t{rayCount} * 3 * sizeof(float);
        std::array<ModelWords, 2> inputSettings{
            ModelWords{rayCount, scalarOffset(rays.origins, vectorBytes, ComputeScalarType::Float32),
                       scalarOffset(rays.directions, vectorBytes, ComputeScalarType::Float32), 0},
            rays.filter};
        std::memcpy(&inputSettings[0][3], &rays.maxDistance, sizeof(float));
        auto &upload = updates.emplace_back(
            rhi::ComputeBufferUpdate{m_inputSettings, 0, std::vector<uint8_t>(sizeof(inputSettings))});
        std::memcpy(upload.data.data(), inputSettings.data(), upload.data.size());
        auto &pack = dispatches.emplace_back(m_pack);
        pack.groupCountX = (rayCount - 1) / 64 + 1;
        pack.buffers[1] = rays.origins.buffer;
        pack.buffers[2] = rays.directions.buffer;
        pack.buffers[3] = input;
        pack.proof->reads[1] = {1, rays.origins.byteOffset, vectorBytes};
        pack.proof->reads[2] = {2, rays.directions.byteOffset, vectorBytes};
        pack.proof->writes.front().byteSize = inputBytes;
    }
    AppendGeometry(dispatches, source);
    dispatches.insert(dispatches.end(), m_rayWork.begin(), m_rayWork.end());
    for (auto &dispatch : dispatches)
        dispatch.groupCountY = 1;

    const auto rows = allHitCapacity ? uint64_t{allHitCapacity} + 1 : uint64_t{rayCount};
    const auto bytes = rows * sizeof(InxPhyGpuRayHit);
    if (!m_output || m_output.use_count() != 1 || m_output->GetByteSize() < bytes)
        m_output = std::make_shared<rhi::ComputeBuffer>(
            host, rhi::ComputeBufferDesc{bytes / sizeof(uint32_t), rhi::ComputeScalarType::UInt32, 1,
                                         rhi::BufferUsageFlags::Storage | rhi::BufferUsageFlags::TransferSource});
    auto &trace = dispatches.back();
    if (allHitCapacity)
        trace = m_traceAll;
    else
        trace.groupCountX = (rayCount - 1) / 64 + 1;
    trace.buffers[1] = input;
    trace.buffers[6] = m_output;
    trace.proof->reads[1].byteSize = inputBytes;
    trace.proof->writes.front().byteSize = bytes;
    if (outputs) {
        const std::array<const PhysicsDeviceBufferRange *, 4> views{&outputs->point, &outputs->normal,
                                                                    &outputs->distance, &outputs->hit};
        constexpr std::array<uint32_t, 4> lanes{3, 3, 1, 1};
        std::array<uint32_t, 4> offsets{};
        auto &project = dispatches.emplace_back(m_project);
        project.groupCountX = (rayCount - 1) / 64 + 1;
        project.groupCountY = 1;
        project.buffers[1] = input;
        project.buffers[2] = m_output;
        project.proof->reads[1].byteSize = inputBytes;
        project.proof->reads[2].byteSize = bytes;
        for (uint32_t field = 0; field < views.size(); ++field) {
            const auto &view = *views[field];
            const auto size = uint64_t{rayCount} * lanes[field] * sizeof(uint32_t);
            const auto scalar = field == 3 ? ComputeScalarType::UInt32 : ComputeScalarType::Float32;
            offsets[field] = scalarOffset(view, size, scalar);
            if (view.buffer->IsFrozen())
                throw std::invalid_argument("Device raycast destinations must be writable");
            for (uint32_t prior = 0; prior < field; ++prior)
                if (view.buffer->GetAllocation() == views[prior]->buffer->GetAllocation() &&
                    offsets[field] < offsets[prior] + rayCount * lanes[prior] &&
                    offsets[prior] < offsets[field] + rayCount * lanes[field])
                    throw std::invalid_argument("Device raycast destinations must not overlap");
            project.buffers[field + 3] = view.buffer;
            project.proof->writes[field] = {field + 3, view.byteOffset, size};
        }
        const std::array<ModelWords, 2> outputSettings{ModelWords{rayCount, offsets[0], offsets[1], offsets[2]},
                                                       ModelWords{offsets[3], 0, 0, 0}};
        auto &upload = updates.emplace_back(
            rhi::ComputeBufferUpdate{m_outputSettings, 0, std::vector<uint8_t>(sizeof(outputSettings))});
        std::memcpy(upload.data.data(), outputSettings.data(), upload.data.size());
    }
    std::vector<rhi::ComputeBufferRead> reads;
    if (publish)
        reads.push_back({m_output, 0, bytes});
    return {rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches), std::move(reads)), m_output,
            rayCount};
}

std::vector<InxPhyGpuStep::SubstepStages>
InxPhyGpuStep::PrepareFixedStages(std::vector<InxPhyGpuFixedStageWork> stages,
                                  std::vector<InxPhyGpuBodyCommand> &commands,
                                  std::vector<rhi::ComputeBufferUpdate> &uploads) const
{
    if (stages.empty())
        return {};

    std::set<std::pair<uint32_t, uint32_t>> savedBodies;
    for (const auto &command : commands)
        savedBodies.emplace(command.world, command.body);
    auto host = m_work->GetRecipe().GetHost();
    const auto portBytes = [&host](const PhysicsDeviceBufferRange &port, uint64_t words) {
        const auto bytes = words * sizeof(float);
        const auto end = port.buffer ? std::min(port.byteEnd, port.buffer->GetByteSize()) : 0u;
        if (!port.buffer || !host.SharesServicesWith(port.buffer->GetHost()) ||
            port.buffer->GetDesc().scalarType != ComputeScalarType::Float32 || port.buffer->GetDesc().lanes != 1 ||
            port.byteOffset % sizeof(float) != 0 || port.byteOffset > end || bytes > end - port.byteOffset ||
            (port.byteOffset + bytes) / sizeof(float) > UINT32_MAX)
            throw std::invalid_argument("inxphygpu rigid exchange requires a scalar f32 port on the physics device");
        return bytes;
    };
    std::vector<SubstepStages> work(m_substeps.count);
    for (auto &stage : stages) {
        auto &phase = work.at(stage.substep).at(static_cast<size_t>(stage.stage));
        uploads.insert(uploads.end(), std::make_move_iterator(stage.updates.begin()),
                       std::make_move_iterator(stage.updates.end()));
        if (!stage.bodies) {
            phase.insert(phase.end(), std::make_move_iterator(stage.dispatches.begin()),
                         std::make_move_iterator(stage.dispatches.end()));
            continue;
        }

        const auto &batch = *stage.bodies;
        if (batch.world >= m_controls.worlds || batch.bodies.empty())
            throw std::invalid_argument("inxphygpu rigid exchange requires enrolled bodies");
        const auto &input = batch.impulses;
        const auto &motion = batch.motion;
        const auto bytes = portBytes(input, batch.bodies.size() * uint64_t{6});
        const auto motionBytes = motion.buffer ? portBytes(motion, batch.bodies.size() * uint64_t{16}) : 0;
        uint32_t first = static_cast<uint32_t>(input.byteOffset / sizeof(float));
        uint32_t firstMotion = static_cast<uint32_t>(motion.byteOffset / sizeof(float));
        std::set<uint32_t> addressed;
        std::vector<ModelWords> rows;
        rows.reserve(batch.bodies.size());
        for (const auto body : batch.bodies) {
            if (body >= m_controls.motion.size() || m_controls.motion[body] != DynamicControl ||
                !addressed.insert(body).second)
                throw std::invalid_argument("inxphygpu rigid exchange requires distinct dynamic bodies");
            rows.push_back({batch.world, body, first, firstMotion});
            first += 6;
            firstMotion += 16;
            // Record the original motion so a rejected step also rolls back impulse-only bodies.
            if (savedBodies.emplace(batch.world, body).second)
                commands.push_back({batch.world, body, InxPhyGpuBodyCommandKind::ApplyPendingWrench});
        }
        using Usage = rhi::BufferUsageFlags;
        auto indices = std::make_shared<rhi::ComputeBuffer>(
            host, rhi::ComputeBufferDesc{rows.size() * 4, ComputeScalarType::UInt32, 1,
                                         Usage::Storage | Usage::TransferDestination});
        rhi::ComputeBufferUpdate upload{indices, 0, std::vector<uint8_t>(indices->GetByteSize())};
        std::memcpy(upload.data.data(), rows.data(), upload.data.size());
        uploads.push_back(std::move(upload));
        const auto groups = (static_cast<uint32_t>(rows.size()) - 1) / 64 + 1;
        if (motionBytes != 0) {
            auto gather = m_controls.deviceMotion;
            gather.buffers[6] = indices;
            gather.buffers[7] = motion.buffer;
            gather.proof->reads[6] = {6, 0, indices->GetByteSize()};
            gather.proof->writes.push_back({7, motion.byteOffset, motionBytes});
            gather.groupCountX = groups;
            LinearGrid(gather);
            phase.push_back(std::move(gather));
        }
        phase.insert(phase.end(), std::make_move_iterator(stage.dispatches.begin()),
                     std::make_move_iterator(stage.dispatches.end()));
        auto dispatch = m_controls.deviceImpulse;
        dispatch.buffers[6] = std::move(indices);
        dispatch.buffers[7] = input.buffer;
        dispatch.proof->reads[6] = {6, 0, dispatch.buffers[6]->GetByteSize()};
        dispatch.proof->reads[7] = {7, input.byteOffset, bytes};
        dispatch.groupCountX = groups;
        LinearGrid(dispatch);
        phase.push_back(std::move(dispatch));
    }
    return work;
}

std::shared_ptr<rhi::PreparedComputeBatch>
InxPhyGpuStep::PrepareControlled(std::vector<InxPhyGpuBodyCommand> commands, InxPhyGpuBodyPublication publication,
                                 std::vector<rhi::ComputeBufferUpdate> updates, bool captureContacts,
                                 std::vector<InxPhyGpuJointParameterUpdate> jointUpdates,
                                 const std::vector<PhysicsFixedInputs> &fixedInputs,
                                 std::vector<InxPhyGpuFixedStageWork> projectStages) const
{
    if (!m_controls.states)
        throw std::logic_error("inxphygpu step has no rigid body control layout");
    if (!fixedInputs.empty()) {
        if (fixedInputs.size() != m_controls.worlds)
            throw std::invalid_argument("inxphygpu fixed inputs require one row per enrolled world");
        m_fixedInputs.Append(updates, fixedInputs, m_substeps.count);
    }
    AppendJointEdits(updates, std::move(jointUpdates));
    auto stageWork = PrepareFixedStages(std::move(projectStages), commands, updates);
    if (commands.empty() && stageWork.empty() && publication == InxPhyGpuBodyPublication::None && !captureContacts)
        return Prepare(false, std::move(updates));
    auto dispatches = m_substeps.begin;
    std::vector<rhi::ComputeReplayRange> replays;
    const auto &streams = BodyCommandStorage(commands.size());
    const auto spanCount = AppendBodyCommands(std::move(commands), true, streams, updates, dispatches);
    AppendFixedWork(dispatches, replays, spanCount, captureContacts, streams, std::move(stageWork));
    if (spanCount != 0) {
        auto finish = m_controls.finish;
        BindBodyCommandStreams(finish, streams, 0, spanCount);
        finish.groupCountX = (spanCount - 1) / 64 + 1;
        LinearGrid(finish);
        dispatches.push_back(std::move(finish));
    }
    if (publication != InxPhyGpuBodyPublication::None)
        AppendBodyPublication(dispatches, true);
    std::vector<rhi::ComputeBufferRead> reads;
    if (publication == InxPhyGpuBodyPublication::Cpu)
        reads.push_back({m_controls.states, 0, m_controls.states->GetByteSize()});
    if (publication == InxPhyGpuBodyPublication::Status) {
        const auto headers = sizeof(ModelWords) * uint64_t{m_controls.worlds};
        reads.push_back({m_controls.states, m_controls.states->GetByteSize() - headers, headers});
    }
    if (captureContacts && m_reports.records)
        reads.push_back({m_reports.counts, m_reports.countsOffset, m_reports.countsBytes});
    auto host = m_work->GetRecipe().GetHost();
    auto packet = rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches), std::move(reads));
    packet->SetDispatchReplay(std::move(replays));
    return packet;
}

std::shared_ptr<rhi::PreparedComputeBatch>
InxPhyGpuStep::PrepareEdits(std::vector<InxPhyGpuBodyCommand> commands,
                            std::vector<InxPhyGpuJointParameterUpdate> jointUpdates) const
{
    if (!m_controls.states)
        throw std::logic_error("inxphygpu step has no rigid body control layout");
    std::vector<rhi::ComputeBufferUpdate> updates;
    AppendJointEdits(updates, std::move(jointUpdates));
    std::vector<rhi::ComputeDispatchDesc> dispatches;
    if (!commands.empty()) {
        const auto &streams = BodyCommandStorage(commands.size());
        AppendBodyCommands(std::move(commands), false, streams, updates, dispatches);
        AppendBodyPublication(dispatches, false);
    }
    auto host = m_work->GetRecipe().GetHost();
    return rhi::PrepareComputeBatch(host, std::move(updates), std::move(dispatches));
}

void InxPhyGpuStep::AppendJointEdits(std::vector<rhi::ComputeBufferUpdate> &updates,
                                     std::vector<InxPhyGpuJointParameterUpdate> jointUpdates) const
{
    std::vector<uint64_t> wakeRows;
    for (const auto &update : jointUpdates) {
        for (const auto body : update.dynamicEndpoints) {
            if (body != NoModelIndex)
                wakeRows.push_back(uint64_t{update.world} * m_controls.motion.size() + body);
        }
    }
    std::sort(wakeRows.begin(), wakeRows.end());
    wakeRows.erase(std::unique(wakeRows.begin(), wakeRows.end()), wakeRows.end());
    m_sceneJoints.Append(updates, std::move(jointUpdates));
    for (const auto row : wakeRows) {
        const uint32_t jointEdited = 1;
        auto &bytes = updates
                          .emplace_back(rhi::ComputeBufferUpdate{m_controls.activity,
                                                                 row * sizeof(ModelWords) + 3 * sizeof(uint32_t),
                                                                 std::vector<uint8_t>(sizeof(jointEdited))})
                          .data;
        std::memcpy(bytes.data(), &jointEdited, sizeof(jointEdited));
    }
}

const InxPhyGpuStep::BodyCommandStreams &InxPhyGpuStep::BodyCommandStorage(size_t count) const
{
    const auto &initial = m_controls.streams;
    const auto &current = m_expandedBodyCommands.capacity ? m_expandedBodyCommands : initial;
    if (count <= current.capacity)
        return current;
    if (count > MaxBodyCommands)
        throw std::length_error("inxphygpu body commands exceed the shader address space");
    const auto capacity = static_cast<uint32_t>(
        std::max(uint64_t{count}, std::min(uint64_t{MaxBodyCommands}, uint64_t{2} * current.capacity)));
    auto host = m_work->GetRecipe().GetHost();
    const auto allocate = [&](const auto &prototype, uint64_t words) {
        auto desc = prototype->GetDesc();
        desc.elementCount = words;
        auto buffer = rhi::ComputeBuffer::TryCreatePrivate(host, desc);
        if (!buffer)
            throw std::runtime_error("inxphygpu command stream could not reserve device storage");
        return buffer;
    };
    m_expandedBodyCommands = {
        allocate(initial.commands, uint64_t{capacity} * sizeof(InxPhyGpuBodyCommand) / sizeof(uint32_t)),
        allocate(initial.spans, (uint64_t{capacity} + 1) * 4),
        allocate(initial.saved, uint64_t{capacity} * SavedMotionWords), capacity};
    return m_expandedBodyCommands;
}

void InxPhyGpuStep::BindBodyCommandStreams(rhi::ComputeDispatchDesc &dispatch, const BodyCommandStreams &streams,
                                           uint32_t commands, uint32_t spans) const
{
    const auto &initial = m_controls.streams;
    const auto bind = [&](const auto &prototype, const auto &storage, uint64_t bytes, bool written = false) {
        const auto found = std::find(dispatch.buffers.begin(), dispatch.buffers.end(), prototype);
        if (found == dispatch.buffers.end())
            return;
        const auto binding = static_cast<uint32_t>(found - dispatch.buffers.begin());
        *found = storage;
        for (auto &range : dispatch.proof->reads)
            if (range.binding == binding)
                range.byteSize = bytes;
        if (written && dispatch.bufferAccesses[binding] == W)
            dispatch.proof->writes.push_back({binding, 0, bytes});
    };
    bind(initial.commands, streams.commands, uint64_t{commands} * sizeof(InxPhyGpuBodyCommand));
    bind(initial.spans, streams.spans, (uint64_t{spans} + 1) * sizeof(ModelWords));
    bind(initial.saved, streams.saved, uint64_t{spans} * SavedMotionWords * sizeof(uint32_t), true);
}

uint32_t InxPhyGpuStep::AppendBodyCommands(std::vector<InxPhyGpuBodyCommand> commands, bool integrate,
                                           const BodyCommandStreams &streams,
                                           std::vector<rhi::ComputeBufferUpdate> &updates,
                                           std::vector<rhi::ComputeDispatchDesc> &dispatches) const
{
    if (commands.empty())
        return 0;
    const auto key = [](const auto &command) { return std::tie(command.world, command.body); };
    std::stable_sort(commands.begin(), commands.end(), [&](const auto &a, const auto &b) { return key(a) < key(b); });
    std::vector<ModelWords> spans(1);
    for (uint32_t index = 0; index < commands.size(); ++index) {
        const auto &command = commands[index];
        const auto kind = command.kind;
        const auto motion = m_controls.motion.at(command.body);
        const bool move =
            kind == InxPhyGpuBodyCommandKind::MovePosition || kind == InxPhyGpuBodyCommandKind::MoveRotation;
        const bool dynamic =
            (kind >= InxPhyGpuBodyCommandKind::Force && kind <= InxPhyGpuBodyCommandKind::ForceAtPosition) ||
            kind == InxPhyGpuBodyCommandKind::ApplyPendingWrench;
        const bool advance = kind == InxPhyGpuBodyCommandKind::AdvanceKinematicVelocity;
        const auto drive =
            move && command.poseFlags == InxPhyGpuPoseCommandFlags::TemporaryStatic ? StaticControl : KinematicControl;
        if (command.world >= m_controls.worlds || motion == Uncontrolled || (move && motion != drive) ||
            (dynamic && motion != DynamicControl) || (advance && motion != KinematicControl) ||
            (motion == StaticControl && !move && kind > InxPhyGpuBodyCommandKind::SetRotation))
            throw std::invalid_argument("inxphygpu body command does not match its world or motion type");
        if (!integrate && (move || advance || kind == InxPhyGpuBodyCommandKind::ApplyPendingWrench))
            throw std::invalid_argument("Kinematic targets and wrench consumption require a fixed step");
        if (move && (!std::isfinite(command.point[0]) || command.point[0] <= 0.0f || !std::isfinite(command.point[1]) ||
                     command.point[1] < 0.0f))
            throw std::invalid_argument("Physics movement requires a positive duration and nonnegative speed limit");
        if (index == 0 || key(command) != key(commands[index - 1]))
            spans.push_back({command.world, command.body, index, 0});
        ++spans.back()[3];
    }

    const auto spanCount = static_cast<uint32_t>(spans.size() - 1);
    spans[0] = {spanCount, 0, m_substeps.count, 0};
    auto &spansUpload = updates.emplace_back(
        rhi::ComputeBufferUpdate{streams.spans, 0, std::vector<uint8_t>(spans.size() * sizeof(ModelWords))});
    std::memcpy(spansUpload.data.data(), spans.data(), spansUpload.data.size());
    auto &commandsUpload = updates.emplace_back(rhi::ComputeBufferUpdate{
        streams.commands, 0, std::vector<uint8_t>(commands.size() * sizeof(InxPhyGpuBodyCommand))});
    std::memcpy(commandsUpload.data.data(), commands.data(), commandsUpload.data.size());

    auto apply = integrate ? m_controls.apply : m_controls.edit;
    BindBodyCommandStreams(apply, streams, static_cast<uint32_t>(commands.size()), spanCount);
    apply.groupCountX = (spanCount - 1) / 64 + 1;
    LinearGrid(apply);
    dispatches.push_back(std::move(apply));
    return spanCount;
}

void InxPhyGpuStep::AppendBodyPublication(std::vector<rhi::ComputeDispatchDesc> &dispatches, bool includeStatus) const
{
    if (m_controls.pose.kernel)
        dispatches.push_back(m_controls.pose);
    dispatches.push_back(includeStatus && m_controls.publishWithStatus.kernel ? m_controls.publishWithStatus
                                                                              : m_controls.publish);
}

void InxPhyGpuStep::SceneJoints::Append(std::vector<rhi::ComputeBufferUpdate> &updates,
                                        std::vector<InxPhyGpuJointParameterUpdate> values) const
{
    const auto key = [](const auto &value) { return std::tie(value.world, value.offset); };
    // Keep the last edit for each field while combining adjacent transfer ranges.
    std::stable_sort(values.begin(), values.end(), [&](const auto &a, const auto &b) { return key(a) < key(b); });
    rhi::ComputeBufferUpdate *range = nullptr;
    constexpr auto recordBytes = sizeof(ModelWords);
    for (size_t index = 0; index < values.size(); ++index) {
        const auto &value = values[index];
        if (index + 1 < values.size() && key(value) == key(values[index + 1]))
            continue;
        const auto record = uint64_t{value.world} * stride + value.offset;
        std::visit(
            [&](const auto &data) {
                if (value.offset >= stride || sizeof(data) > (uint64_t{stride} - value.offset) * recordBytes ||
                    !parameters || record >= parameters->GetByteSize() / recordBytes)
                    throw std::invalid_argument("inxphygpu joint update does not match its enrolled world or record");
                const auto offset = record * recordBytes;
                if (!range || offset > range->offset + range->data.size())
                    range = &updates.emplace_back(rhi::ComputeBufferUpdate{parameters, offset, {}});
                const auto local = static_cast<size_t>(offset - range->offset);
                range->data.resize(local + sizeof(data));
                std::memcpy(range->data.data() + local, &data, sizeof(data));
            },
            value.value);
    }
}

void InxPhyGpuStep::FixedInputs::Append(std::vector<rhi::ComputeBufferUpdate> &updates,
                                        const std::vector<PhysicsFixedInputs> &values, uint32_t substeps) const
{
    const float divisions = static_cast<float>(substeps);
    for (const auto &input : values)
        if (!std::isfinite(input.duration) || input.duration / divisions <= 0.0f || !std::isfinite(input.gravity.x) ||
            !std::isfinite(input.gravity.y) || !std::isfinite(input.gravity.z))
            throw std::invalid_argument("inxphygpu fixed inputs require a positive duration and finite gravity");

    // These buffers already belong to the resident step recipe. Write only their
    // per-world settings, leaving shader bindings, clocks, and motion in place.
    const auto upload = [&](const auto &buffer, uint64_t offset, const auto &pack) {
        using Row = decltype(pack(values.front()));
        auto &data = updates
                         .emplace_back(rhi::ComputeBufferUpdate{buffer, offset,
                                                                std::vector<uint8_t>(values.size() * sizeof(Row))})
                         .data;
        for (size_t world = 0; world < values.size(); ++world) {
            const auto row = pack(values[world]);
            std::memcpy(data.data() + world * sizeof(row), &row, sizeof(row));
        }
    };
    upload(timestep, 0, [](const auto &input) { return input.duration; });
    if (substep != timestep)
        upload(substep, 0, [&](const auto &input) { return input.duration / divisions; });
    if (gravity)
        upload(gravity, 0,
               [](const auto &input) { return ModelVector{input.gravity.x, input.gravity.y, input.gravity.z, 0.0f}; });
    if (constraints)
        upload(constraints, 2 * sizeof(ModelWords), [&](const auto &input) {
            auto row = constraintDefaults;
            row[0] = FloatWord(input.duration / divisions);
            return row;
        });
}

std::shared_ptr<rhi::PreparedComputeBatch>
InxPhyGpuStep::PrepareContactReports(const std::vector<uint32_t> &counts) const
{
    std::vector<rhi::ComputeBufferRead> reads;
    for (size_t world = 0; world < counts.size(); ++world) {
        const auto count = counts[world];
        if (count > m_reports.capacity)
            throw std::logic_error("Accepted inxphygpu contact report count exceeds its world capacity");
        if (count != 0) {
            const auto base = world * m_reports.capacity * sizeof(InxPhyGpuContactReport);
            reads.push_back({m_reports.records, base, count * sizeof(InxPhyGpuContactReport)});
        }
    }
    if (reads.empty())
        return {};
    auto host = m_work->GetRecipe().GetHost();
    return rhi::PrepareComputeBatch(host, {}, {}, std::move(reads));
}

std::shared_ptr<rhi::ComputeBuffer> InxPhyGpuStep::BodyStates() const noexcept
{
    return m_controls.states;
}

PhysicsGpuJointMotionRead InxPhyGpuStep::PrepareJointMotion(const PhysicsGpuPoseSource &source,
                                                            const std::vector<uint32_t> &rows, bool readback) const
{
    if (rows.empty())
        return {};
    constexpr auto bodyBytes = sizeof(InxPhyGpuBodyState);
    if (!m_sceneJoints.motionKernel)
        throw std::invalid_argument("inxphygpu joint motion requires its native scene body publication");
    const auto count = static_cast<uint32_t>(rows.size());
    const auto resultBytes = uint64_t{count} * sizeof(PhysicsGpuJointMotion);
    auto host = m_work->GetRecipe().GetHost();
    using Usage = rhi::BufferUsageFlags;
    auto request = std::make_shared<rhi::ComputeBuffer>(
        host, rhi::ComputeBufferDesc{uint64_t{count} + 2, rhi::ComputeScalarType::UInt32, 1,
                                     Usage::Storage | Usage::TransferDestination});
    auto motion = std::make_shared<rhi::ComputeBuffer>(
        host, rhi::ComputeBufferDesc{resultBytes / sizeof(float), rhi::ComputeScalarType::Float32, 1,
                                     Usage::Storage | Usage::TransferSource});
    const uint32_t header[]{count, static_cast<uint32_t>(source.firstRow)};
    rhi::ComputeBufferUpdate upload{request, 0, std::vector<uint8_t>(request->GetByteSize())};
    std::memcpy(upload.data.data(), header, sizeof(header));
    std::memcpy(upload.data.data() + sizeof(header), rows.data(), uint64_t{count} * sizeof(uint32_t));
    rhi::ComputeDispatchDesc dispatch;
    dispatch.kernel = m_sceneJoints.motionKernel;
    dispatch.buffers = {m_sceneJoints.descriptors, source.states, request, motion};
    dispatch.bufferAccesses = {R, R, R, W};
    dispatch.groupCountX = (count - 1) / 64 + 1;
    LinearGrid(dispatch);
    dispatch.proof = rhi::ComputeDispatchProof{{{0, 0, m_sceneJoints.descriptors->GetByteSize()},
                                                {1, source.firstRow * bodyBytes, source.bodyCount * bodyBytes},
                                                {2, 0, request->GetByteSize()}},
                                               {{3, 0, resultBytes}},
                                               {}};
    std::vector<rhi::ComputeBufferRead> reads;
    if (readback)
        reads.push_back({motion, 0, resultBytes});
    auto packet = rhi::PrepareComputeBatch(host, {std::move(upload)}, {std::move(dispatch)}, std::move(reads));
    return {std::move(packet), std::move(motion), count};
}

const std::shared_ptr<rhi::ComputeBuffer> &InxPhyGpuStep::Positions() const noexcept
{
    return m_work->GetBuffers()[m_positions];
}

const std::shared_ptr<rhi::ComputeBuffer> &InxPhyGpuStep::Velocities() const noexcept
{
    return m_work->GetBuffers()[m_velocities];
}

const std::shared_ptr<rhi::ComputeBuffer> &InxPhyGpuStep::AppliedForces() const noexcept
{
    return m_work->GetBuffers()[m_appliedForces];
}

} // namespace infernux::physics::inxphygpu
