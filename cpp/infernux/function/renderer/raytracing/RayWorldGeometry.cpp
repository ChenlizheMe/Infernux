#include "RayWorldGeometry.h"

#include "RayDeformation.h"
#include "RayDeformationSource.h"
#include "RayEmissive.h"
#include "RayGeometry.h"
#include "RayMaterial.h"
#include "RayShadowChanges.h"

#include <core/error/InxError.h>
#include <core/types/ColorSpace.h>
#include <function/renderer/RenderSurfaceRecords.h>
#include <function/renderer/RenderWorld.h>
#include <function/renderer/RendererParameterBlock.h>
#include <function/renderer/rhi/RhiBuffer.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxMesh/InxMesh.h>
#include <function/resources/InxMesh/MeshArtifactReader.h>
#include <function/resources/ShaderAsset/ShaderAsset.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace infernux::raytracing
{
namespace
{
using MeshRevision = std::array<uint64_t, 5>;
using MeshSlice = std::tuple<uintptr_t, uintptr_t, MeshGeometryView, uint32_t, uint32_t, int32_t>;

MeshSlice GetMeshSlice(const DrawCall &draw)
{
    return {reinterpret_cast<uintptr_t>(draw.meshVertices),
            reinterpret_cast<uintptr_t>(draw.meshIndices),
            draw.meshGeometryView,
            draw.indexStart,
            draw.indexCount,
            draw.vertexStart};
}

struct WorldRevision
{
    MeshRevision meshes;
    uint64_t transform;
    bool operator==(const WorldRevision &other) const noexcept
    {
        return meshes == other.meshes && transform == other.transform;
    }

    bool operator!=(const WorldRevision &other) const noexcept
    {
        return !(*this == other);
    }

    bool SharesDrawSlots(const WorldRevision &other) const noexcept
    {
        // Surface slots can be reused only within the same World, simulation epoch and structure.
        return std::equal(meshes.begin(), meshes.begin() + 3, other.meshes.begin());
    }
};
constexpr uint32_t FullBuild = UINT32_MAX;
constexpr uint32_t InlineWorkWindow = 1024;
constexpr uint64_t UploadWindow = 256 * 1024;
const RenderLightingSource kNoWorldLighting;

WorldRevision GetRevision(const RenderWorldFrame &world)
{
    return {{world.WorldId(), world.SimulationEpoch(), world.StructuralRevision(), world.ContentRevision(),
             world.PoseRevision()},
            world.TransformRevision()};
}

uint64_t AddStorage(uint64_t bytes, uint64_t extra)
{
    const auto maximum = (std::numeric_limits<uint64_t>::max)();
    return bytes > maximum - extra ? maximum : bytes + extra;
}

std::shared_ptr<const rhi::BufferResource> AllocatePage(rhi::Device &device, uint64_t bytes)
{
    if (!device.GetCapabilities().IsStorageBufferRangeValid(0, bytes))
        throw std::length_error("World ray geometry exceeds the device storage range");
    rhi::BufferDesc desc;
    desc.byteSize = bytes;
    desc.usage = rhi::BufferUsageFlags::Storage;
    desc.memory = rhi::BufferMemory::Upload;
    desc.queueAccess = rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute;
    const auto allocation = device.CreateBuffer(desc);
    if (!allocation.IsValid())
        throw std::runtime_error("Could not allocate the world's ray geometry page");
    return std::make_shared<rhi::BufferResource>(device, allocation, bytes);
}

void WritePage(rhi::Device &device, const rhi::BufferResource &buffer, uint64_t offset, const void *data,
               uint64_t bytes)
{
    if (!device.WriteBuffer(buffer.GetBuffer(), offset, data, bytes))
        throw std::runtime_error("Could not publish the world's ray geometry page");
}

struct DrawMeshPlacement
{
    uint32_t surfaceId;
    RayMeshPlacement placement;
    MeshSlice source;
    std::shared_ptr<const void> owner;
    uint64_t version;
    std::shared_ptr<const std::vector<glm::mat4>> palette;
    bool deviceVertices;
    uint32_t triangleStart;
    uint32_t triangleCount;
};

// A selection proxy only: the shader evaluates emission and the current world
// triangle area. Uniform mixing keeps every triangle in the supported domain.
float RayEmissivePower(const InxMaterial &material, const RendererParameterBlock *parameters)
{
    const auto kind = ClassifyRayMaterial(material);
    if (kind == RayMaterialParticipation::Lit) {
        const auto &program = material.GetSelectedShaderArtifact();
        if (program && program->hasRaySurface)
            return program->raySurfaceMayEmit ? 1.0f : 0.0f;
    }
    const bool lit = kind == RayMaterialParticipation::Lit;
    if (!lit && kind != RayMaterialParticipation::Unlit)
        return 0.0f;
    const std::string property = lit ? "emission_color" : "base_color";
    const auto *declared = material.GetProperty(property);
    if (!declared)
        return 0.0f;
    const auto *value = declared;
    if (parameters) {
        const auto found = parameters->properties.find(property);
        if (found != parameters->properties.end() && MaterialPropertyOverrideCompatible(*declared, found->second))
            value = &found->second;
    }
    const auto color = std::get<glm::vec4>(value->value);
    const auto linear = declared->type == MaterialPropertyType::Color ? inx::color::SrgbToLinear(color) : color;
    const auto magnitude = glm::abs(glm::vec3(linear));
    return std::max({magnitude.x, magnitude.y, magnitude.z}) * (lit ? std::abs(color.w) : 1.0f);
}

struct RaySkinTopology
{
    MeshSlice source;
    std::shared_ptr<const void> owner;
    std::shared_ptr<const spatial::BvhTopology> tree;
};

struct RayWorldMeshPage
{
    std::vector<DrawMeshPlacement> draws;
    std::vector<float> referenceAreas;
    std::shared_ptr<const rhi::BufferResource> buffer;
    std::shared_ptr<FrameCpuStorageClaim> storage;
    std::map<uint32_t, RaySkinTopology> skins;
    uint64_t skinStorage = 0;
    RayRefitPlan refit;
    mutable std::shared_ptr<RayDeformationProgram> deformationProgram;
    // The frame owner caches the latest validated TLAS topology for this mesh set.
    mutable std::shared_ptr<const spatial::BvhTopology> topology;
    mutable RayShadowState shadowHistory;
};
} // namespace

class RayWorldGeometryPublication final
{
  public:
    explicit RayWorldGeometryPublication(std::shared_ptr<const RenderWorldFrame> frame)
        : revision(GetRevision(*frame)), visibilityLighting(frame->Lighting() ? frame->Lighting()->generation : 0u),
          world(std::move(frame))
    {
    }

    const WorldRevision revision;
    const uint64_t visibilityLighting;
    std::shared_ptr<const RenderWorldFrame> world;
    std::shared_ptr<const rhi::BufferResource> buffer;
    std::shared_ptr<const RayWorldMeshPage> meshes;
    std::shared_ptr<const RayWorldMeshPage> skinBasis;
    rhi::SubmissionTicket producer;
    uint32_t visibilitySerial = 0;
    RayGeometryStatus status = RayGeometryStatus::Pending;
};

rhi::SubmissionTicket
RayGeometryProducer(const std::shared_ptr<const RayWorldGeometryPublication> &publication) noexcept
{
    return publication ? publication->producer : rhi::SubmissionTicket{};
}

class RayWorldGeometryBuild final
{
  public:
    enum class Step : uint8_t
    {
        Plan,
        Open,
        Decode,
        BuildRecovered,
        BuildResident,
        PackMeshes,
        BuildReferenceAreas,
        CollectInstances,
        BuildInstances,
        UpdateShadows,
        BuildEmissiveTable,
        Upload,
        CompileDeformation,
        Deform,
    };

    bool Plan(uint32_t window);
    void Run(uint32_t window);
    [[nodiscard]] bool Upload(rhi::Device &device, uint64_t bytes);
    [[nodiscard]] FrameWorkResource Resource() const;
    [[nodiscard]] uint64_t UploadBytes(uint64_t remaining) const;
    [[nodiscard]] std::shared_ptr<const RayWorldMeshPage> PublishedMeshes();
    bool NeedsDeformation() const noexcept
    {
        const auto &page = meshCandidate ? meshCandidate : sharedMeshes;
        return page && !page->refit.meshes.empty();
    }
    const RayRefitPlan &DeformationPlan() const noexcept
    {
        return (meshCandidate ? meshCandidate : sharedMeshes)->refit;
    }
    void PrepareDeformation(rhi::Device &device)
    {
        const auto &page = meshCandidate ? meshCandidate : sharedMeshes;
        if (!page->deformationProgram) {
            page->deformationProgram = std::make_shared<RayDeformationProgram>(device, *deformationCompilation);
            deformationCompilation.reset();
        }
        deformation = std::make_shared<RayDeformation>(
            device, page->deformationProgram, page->refit, instancePage, world->DrawCalls().drawCalls,
            RayPageUpload(instancePage, RayGeometryStatus::Ready, shadowHistory.revision, shadowHistory.stamps),
            page->buffer);
    }
    [[nodiscard]] uint32_t VisibilitySerial() const noexcept
    {
        return shadowHistory.revision.serial;
    }
    [[nodiscard]] uint64_t DecodeWorkspace() const
    {
        return reader->WorkspaceBytes();
    }

    Step step = Step::Plan;
    std::weak_ptr<RayWorldGeometryPublication> publication;
    std::shared_ptr<const RenderWorldFrame> world;
    std::shared_ptr<const RayWorldMeshPage> sharedMeshes;
    std::shared_ptr<const RayWorldMeshPage> skinBasis;
    std::shared_ptr<FrameCpuStorageClaim> storage;
    std::shared_ptr<FrameCpuStorageClaim> decodeStorage;
    std::shared_ptr<const rhi::BufferResource> buffer;
    std::exception_ptr failure;
    uint64_t workspace = 3 * sizeof(RayGeometryWord);
    uint64_t elapsed = 0;
    std::shared_ptr<RayDeformation> deformation;
    bool deformationSubmitted = false;
    uint64_t deformationTransferGrant = 0;
    uint64_t topologyUploadBytes = 0;
    ShaderCodeTarget deformationTarget = ShaderCodeTarget::SpirV;
    ShaderSourceModule deformationSource;
    std::optional<ShaderCompilation> deformationCompilation;

  private:
    using SliceKey = std::tuple<size_t, MeshSlice, uintptr_t>;

    struct Slice
    {
        const DrawCall *draw;
        size_t mesh;
    };

    struct Recovery
    {
        MeshArtifactReference artifact;
        std::vector<Slice> slices;
    };

    bool BuildSlices(const std::vector<Slice> &work, uint32_t window);
    void StartUpload(const RayPageUpload &page);

    std::map<SliceKey, size_t> sliceIndices;
    std::map<std::pair<std::string, uint64_t>, size_t> artifactIndices;
    size_t nextDraw = 0;
    std::vector<RayMesh> meshes;
    std::vector<Slice> instances;
    std::vector<Slice> resident;
    std::vector<Recovery> recovery;
    size_t nextArtifact = 0;
    std::unique_ptr<MeshArtifactReader> reader;
    std::shared_ptr<InxMesh> decodedMesh;
    size_t nextSlice = 0;
    std::unique_ptr<RayMeshBuild> meshBuild;
    std::unique_ptr<RayMeshPageBuild> meshPageBuild;
    RayMeshPage meshPage;
    std::vector<std::array<uint32_t, 2>> triangleRanges;
    size_t nextReferenceArea = 0;
    std::shared_ptr<RayWorldMeshPage> meshCandidate;
    std::vector<RayDeformedMesh> deformed;
    bool reusesSources = true;
    std::vector<RayMeshInstance> rays;
    std::vector<RayShadowInstance> shadowInstances;
    size_t surfaceCount = 1;
    size_t nextInstanceDraw = 0;
    size_t nextPlacement = 0;
    std::unique_ptr<RayInstancePageBuild> instanceBuild;
    RayInstancePage instancePage;
    std::unique_ptr<RayShadowChanges> shadowChanges;
    RayShadowState shadowHistory;
    std::vector<RayEmissiveSource> emissiveSources;
    std::unique_ptr<RayEmissiveBuild> emissiveBuild;
    std::optional<RayPageUpload> layout;
    uint64_t uploadedBytes = 0;
    size_t uploadSection = 0;
    uint64_t sectionOffset = 0;
};

bool RayWorldGeometryBuild::Plan(uint32_t window)
{
    // Matches the exact draw ranges of the frozen raster world. Ray queries
    // also need geometry outside the current camera frustum.
    const auto &draws = world->DrawCalls().drawCalls;
    const auto end = nextDraw + std::min<size_t>(window, draws.size() - nextDraw);
    for (; nextDraw < end; ++nextDraw) {
        const auto &draw = draws[nextDraw];
        if (!draw.indexCount)
            continue;
        if (!draw.surfaceId || draw.devicePose.HasValue())
            return false;
        if ((draw.skinBoneMatrices || draw.meshVertexBuffer) &&
            (!draw.material || draw.material->GetVertShaderName() != kStandardVertexShaderId))
            return false;
        if (draw.skinBoneMatrices && draw.meshVertexBuffer)
            return false;
        surfaceCount = std::max(surfaceCount, size_t{draw.surfaceId} + 1);

        size_t artifact = 0;
        // The frozen draw borrows process-lifetime primitive streams or owns a mesh snapshot.
        const bool hasStreams =
            draw.meshVertices && draw.meshIndices && !draw.meshVertices->empty() && !draw.meshIndices->empty();
        if (!hasStreams) {
            if (draw.skinBoneMatrices || draw.meshUsesDetachedStorage || draw.meshAssetGuid.empty())
                return false;
            auto [entry, inserted] =
                artifactIndices.try_emplace({draw.meshAssetGuid, draw.meshRuntimeVersion}, recovery.size() + 1);
            if (inserted) {
                auto reference =
                    AssetRegistry::Instance().CaptureMeshArtifact(draw.meshAssetGuid, draw.meshRuntimeVersion);
                if (!reference)
                    return false;
                recovery.push_back({std::move(*reference), {}});
            }
            artifact = entry->second;
        }
        auto source = GetMeshSlice(draw);
        if (!hasStreams) {
            std::get<0>(source) = 0;
            std::get<1>(source) = 0;
        }
        const SliceKey key{artifact, source,
                           draw.meshVertexBuffer ? draw.surfaceId
                                                 : reinterpret_cast<uintptr_t>(draw.skinBoneMatrices.get())};
        if (skinBasis && reusesSources) {
            const auto index = instances.size();
            reusesSources = index < skinBasis->draws.size();
            if (reusesSources) {
                const auto &old = skinBasis->draws[index];
                reusesSources = old.surfaceId == draw.surfaceId && old.source == GetMeshSlice(draw) &&
                                old.owner == draw.meshDataOwner && old.version == draw.meshRuntimeVersion &&
                                old.palette == draw.skinBoneMatrices &&
                                old.deviceVertices == bool(draw.meshVertexBuffer);
            }
        }
        auto [entry, inserted] = sliceIndices.try_emplace(key, sliceIndices.size());
        const Slice slice{&draw, entry->second};
        instances.push_back(slice);
        workspace = AddStorage(workspace, 512 + sizeof(RayShadowInstance) + sizeof(RayEmissiveSource) +
                                              uint64_t{draw.indexCount / 3u} * sizeof(RayGeometryWord));
        if (inserted) {
            // Covers the compact-index map, BLAS, and its contribution to the final page.
            workspace =
                AddStorage(workspace, uint64_t{draw.indexCount} * 352 + uint64_t{draw.indexCount / 3u} * sizeof(float) +
                                          sizeof(std::array<uint32_t, 2>));
            if (draw.skinBoneMatrices || draw.meshVertexBuffer)
                workspace = AddStorage(workspace, 256 + uint64_t{draw.indexCount} * sizeof(spatial::BvhLink));
            meshes.emplace_back();
            auto &work = hasStreams ? resident : recovery[artifact - 1].slices;
            work.push_back(slice);
        }
    }
    if (nextDraw == draws.size()) {
        workspace = AddStorage(workspace, (sizeof(uint32_t) + sizeof(RayGeometryWord)) * (surfaceCount + 3u));
        if (skinBasis && reusesSources && skinBasis->draws.size() == instances.size()) {
            sharedMeshes = std::move(skinBasis);
            step = Step::CollectInstances;
            return true;
        }
        rays.reserve(instances.size());
        shadowInstances.reserve(instances.size());
        meshCandidate = std::make_shared<RayWorldMeshPage>();
        if (skinBasis)
            meshCandidate->deformationProgram = skinBasis->deformationProgram;
        meshCandidate->draws.reserve(instances.size());
        step = recovery.empty() ? Step::BuildResident : Step::Open;
    }
    return true;
}

FrameWorkResource RayWorldGeometryBuild::Resource() const
{
    switch (step) {
    case Step::Open:
    case Step::Decode:
        return FrameWorkResource::Io;
    case Step::Plan:
    case Step::BuildRecovered:
    case Step::BuildResident:
    case Step::PackMeshes:
    case Step::BuildReferenceAreas:
    case Step::CollectInstances:
    case Step::BuildInstances:
    case Step::UpdateShadows:
    case Step::BuildEmissiveTable:
    case Step::CompileDeformation:
        return FrameWorkResource::Cpu;
    case Step::Upload:
        return FrameWorkResource::Transfer;
    case Step::Deform:
        return FrameWorkResource::Gpu;
    }
    std::terminate();
}

bool RayWorldGeometryBuild::BuildSlices(const std::vector<Slice> &work, uint32_t window)
{
    while (nextSlice < work.size()) {
        const auto &slice = work[nextSlice];
        const auto &draw = *slice.draw;
        if (!meshBuild) {
            const auto *vertices = draw.meshVertices;
            const auto *indices = draw.meshIndices;
            if (decodedMesh) {
                const auto geometry = draw.meshGeometryView == MeshGeometryView::NodeLocal
                                          ? decodedMesh->GetModelSourceGeometry()
                                          : decodedMesh->GetGeometrySnapshot();
                if (!geometry)
                    throw std::runtime_error("The cooked mesh does not contain the frozen geometry view");
                vertices = &geometry->vertices;
                indices = &geometry->indices;
            }
            std::shared_ptr<const spatial::BvhTopology> tree;
            if (draw.skinBoneMatrices && skinBasis) {
                const auto found = skinBasis->skins.find(draw.surfaceId);
                if (found != skinBasis->skins.end() && found->second.owner == draw.meshDataOwner &&
                    found->second.source == GetMeshSlice(draw))
                    tree = found->second.tree;
            }
            meshBuild = std::make_unique<RayMeshBuild>(*vertices, *indices, draw.indexStart, draw.indexCount,
                                                       draw.vertexStart, draw.skinBoneMatrices.get(), std::move(tree),
                                                       bool(draw.meshVertexBuffer));
        }
        if (meshBuild->Advance(window)) {
            auto mesh = meshBuild->Take();
            if (mesh.topology && draw.skinBoneMatrices) {
                meshCandidate->skinStorage += 256 + mesh.topology->links.capacity() * sizeof(spatial::BvhLink) +
                                              mesh.topology->primitiveOrder.capacity() * sizeof(uint32_t);
                meshCandidate->skins.emplace(draw.surfaceId,
                                             RaySkinTopology{GetMeshSlice(draw), draw.meshDataOwner, mesh.topology});
            }
            if (draw.meshVertexBuffer) {
                deformed.push_back({draw.surfaceId, static_cast<uint32_t>(slice.mesh), 0u, 0u,
                                    static_cast<uint32_t>(mesh.nodes.size()), std::move(mesh.sourceVertices),
                                    std::move(mesh.refitLeaves), std::move(mesh.refitLevels)});
            }
            meshes[slice.mesh] = std::move(mesh);
            meshBuild.reset();
            ++nextSlice;
        }
        if (window != FullBuild)
            break;
    }
    if (nextSlice != work.size())
        return false;
    nextSlice = 0;
    return true;
}

void RayWorldGeometryBuild::Run(uint32_t window)
{
    const auto start = std::chrono::steady_clock::now();
    try {
        switch (step) {
        case Step::Plan:
            std::terminate();
        case Step::Open:
            reader = std::make_unique<MeshArtifactReader>(recovery[nextArtifact].artifact);
            step = Step::Decode;
            break;
        case Step::Decode:
            decodedMesh = reader->Read();
            reader.reset();
            step = Step::BuildRecovered;
            break;
        case Step::BuildRecovered:
            if (BuildSlices(recovery[nextArtifact].slices, window)) {
                decodedMesh.reset();
                step = ++nextArtifact == recovery.size() ? Step::BuildResident : Step::Open;
            }
            break;
        case Step::BuildResident:
            if (!BuildSlices(resident, window))
                break;
            skinBasis.reset();
            meshPageBuild = std::make_unique<RayMeshPageBuild>(meshes);
            step = Step::PackMeshes;
            if (window != FullBuild)
                break;
            [[fallthrough]];
        case Step::PackMeshes:
            do {
                if (meshPageBuild->Advance(window)) {
                    meshPage = meshPageBuild->Take();
                    meshPageBuild.reset();
                    // Cooked recovery and resident streams complete in separate groups.
                    // The packed page keeps the original slice order.
                    std::sort(deformed.begin(), deformed.end(),
                              [](const auto &a, const auto &b) { return a.meshIndex < b.meshIndex; });
                    uint32_t vertexBase = 0, triangleBase = 0;
                    size_t dynamic = 0;
                    for (size_t mesh = 0; mesh < meshes.size(); ++mesh) {
                        const auto count = static_cast<uint32_t>(meshes[mesh].triangles.size());
                        triangleRanges.push_back({triangleBase, count});
                        triangleBase += count;
                        if (dynamic < deformed.size() && deformed[dynamic].meshIndex == mesh) {
                            auto &deformationLayout = deformed[dynamic++];
                            deformationLayout.root = meshPage.placements[mesh].root;
                            deformationLayout.vertexBase = vertexBase;
                        }
                        vertexBase += static_cast<uint32_t>(meshes[mesh].positions.size());
                    }
                    meshes.clear();
                    meshCandidate->referenceAreas.resize(meshPage.triangles.size());
                    step = Step::BuildReferenceAreas;
                    break;
                }
            } while (window == FullBuild);
            if (window != FullBuild)
                break;
            [[fallthrough]];
        case Step::BuildReferenceAreas: {
            const auto end =
                nextReferenceArea + std::min<size_t>(window, meshPage.triangles.size() - nextReferenceArea);
            for (; nextReferenceArea != end; ++nextReferenceArea) {
                const auto &triangle = meshPage.triangles[nextReferenceArea];
                const auto point = [&](uint32_t index) {
                    const auto &p = meshPage.positions[index];
                    return glm::vec3(p.x, p.y, p.z);
                };
                meshCandidate->referenceAreas[nextReferenceArea] =
                    0.5f * glm::length(glm::cross(point(triangle.b) - point(triangle.a),
                                                  point(triangle.c) - point(triangle.a)));
            }
            if (nextReferenceArea != meshPage.triangles.size())
                break;
            step = Step::CollectInstances;
            if (window != FullBuild)
                break;
            [[fallthrough]];
        }
        case Step::CollectInstances: {
            const auto &draws = world->DrawCalls().drawCalls;
            const auto &records = world->SurfaceRecords();
            if (!nextInstanceDraw && sharedMeshes) {
                rays.reserve(sharedMeshes->draws.size());
                shadowInstances.reserve(sharedMeshes->draws.size());
            }
            const auto end = nextInstanceDraw + std::min<size_t>(window, draws.size() - nextInstanceDraw);
            for (; nextInstanceDraw != end; ++nextInstanceDraw) {
                const auto &draw = draws[nextInstanceDraw];
                if (!draw.indexCount)
                    continue;
                RayMeshPlacement placement;
                std::array<uint32_t, 2> triangles;
                if (sharedMeshes) {
                    if (nextPlacement == sharedMeshes->draws.size() ||
                        sharedMeshes->draws[nextPlacement].surfaceId != draw.surfaceId)
                        throw std::logic_error("World mesh revision reused with a different frozen draw order");
                    placement = sharedMeshes->draws[nextPlacement].placement;
                    triangles = {sharedMeshes->draws[nextPlacement].triangleStart,
                                 sharedMeshes->draws[nextPlacement].triangleCount};
                } else {
                    const auto mesh = instances[nextPlacement].mesh;
                    placement = meshPage.placements[mesh];
                    triangles = triangleRanges[mesh];
                    meshCandidate->draws.push_back({draw.surfaceId, placement, GetMeshSlice(draw), draw.meshDataOwner,
                                                    draw.meshRuntimeVersion, draw.skinBoneMatrices,
                                                    bool(draw.meshVertexBuffer), triangles[0], triangles[1]});
                }
                ++nextPlacement;
                if (placement.root != kInvalidRayIndex) {
                    surfaceCount = std::max(surfaceCount, size_t{draw.surfaceId} + 1);
                    rays.push_back({placement, draw.worldMatrix, draw.surfaceId, draw.layerMask});
                    const glm::uvec4 lighting = records ? records->Lighting(draw.surfaceId) : glm::uvec4(0u);
                    shadowInstances.push_back(
                        {draw.surfaceId, draw.layerMask, lighting.w != 0u, lighting.z != 0u, draw.worldMatrix, {}});
                    if (draw.material) {
                        const auto power = RayEmissivePower(*draw.material, draw.parameterBlock.get());
                        if (power > 0.0f) {
                            emissiveSources.push_back({draw.surfaceId, triangles[0], triangles[1], power});
                        }
                    }
                }
            }
            if (nextInstanceDraw != draws.size())
                break;
            if (sharedMeshes && nextPlacement != sharedMeshes->draws.size())
                throw std::logic_error("World mesh revision reused with a different frozen draw count");
            instanceBuild =
                std::make_unique<RayInstancePageBuild>(rays, sharedMeshes ? sharedMeshes->topology : nullptr);
            step = Step::BuildInstances;
            if (window != FullBuild)
                break;
            [[fallthrough]];
        }
        case Step::BuildInstances:
            do {
                if (instanceBuild->Advance(window)) {
                    instancePage = instanceBuild->Take();
                    instanceBuild.reset();
                    const auto &lighting = world->Lighting();
                    shadowChanges = std::make_unique<RayShadowChanges>(
                        sharedMeshes ? &sharedMeshes->shadowHistory : nullptr, std::move(shadowInstances), instancePage,
                        lighting ? *lighting : kNoWorldLighting, surfaceCount);
                    step = Step::UpdateShadows;
                    break;
                }
            } while (window == FullBuild);
            if (window != FullBuild)
                break;
            [[fallthrough]];
        case Step::UpdateShadows:
            if (shadowChanges->Advance(window)) {
                shadowHistory = shadowChanges->Take();
                if (NeedsDeformation() || !deformed.empty()) {
                    shadowHistory.revision = {shadowHistory.revision.serial, 0u, shadowHistory.revision.serial};
                    for (auto &stamp : shadowHistory.stamps)
                        stamp.fill(shadowHistory.revision.serial);
                }
                shadowChanges.reset();
                emissiveBuild = std::make_unique<RayEmissiveBuild>(
                    emissiveSources, (meshCandidate ? meshCandidate : sharedMeshes)->referenceAreas, instancePage,
                    surfaceCount);
                step = Step::BuildEmissiveTable;
            }
            break;
        case Step::BuildEmissiveTable:
            if (emissiveBuild->Advance(window)) {
                emissiveBuild.reset();
                if (sharedMeshes && NeedsDeformation()) {
                    step = Step::Deform;
                    break;
                }
                StartUpload(meshCandidate ? RayPageUpload(meshPage)
                                          : RayPageUpload(instancePage, RayGeometryStatus::Ready,
                                                          shadowHistory.revision, shadowHistory.stamps));
                step = Step::Upload;
            }
            break;
        case Step::Upload:
        case Step::Deform:
            std::terminate();
        case Step::CompileDeformation:
            deformationCompilation =
                SlangCompiler::Get().CompileRootProgram(std::move(deformationSource), {}, {deformationTarget}, true);
            deformationSource = {};
            step = Step::Deform;
            break;
        }
    } catch (...) {
        failure = std::current_exception();
    }
    elapsed = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

void RayWorldGeometryBuild::StartUpload(const RayPageUpload &page)
{
    layout = page;
    uploadedBytes = 0;
    uploadSection = 0;
    sectionOffset = 0;
}

uint64_t RayWorldGeometryBuild::UploadBytes(uint64_t remaining) const
{
    const auto capacity = std::max<uint64_t>(sizeof(RayGeometryWord), remaining);
    const auto window = std::min<uint64_t>(capacity, UploadWindow);
    return std::min(layout->bytes - uploadedBytes, window / sizeof(RayGeometryWord) * sizeof(RayGeometryWord));
}

bool RayWorldGeometryBuild::Upload(rhi::Device &device, uint64_t bytes)
{
    if (!buffer)
        buffer = AllocatePage(device, layout->bytes);
    while (bytes) {
        const auto section = layout->Section(uploadSection);
        const auto part = std::min<uint64_t>(bytes, section.bytes - sectionOffset);
        if (part) {
            const auto *data = static_cast<const std::byte *>(section.data) + sectionOffset;
            WritePage(device, *buffer, uploadedBytes, data, part);
            uploadedBytes += part;
            sectionOffset += part;
            bytes -= part;
        }
        if (sectionOffset == section.bytes) {
            ++uploadSection;
            sectionOffset = 0;
        }
    }
    if (uploadedBytes != layout->bytes)
        return false;
    if (meshCandidate && !meshCandidate->buffer) {
        meshCandidate->buffer = std::move(buffer);
        if (!deformed.empty()) {
            meshCandidate->refit = UploadRayRefitPlan(device, std::move(deformed));
            topologyUploadBytes = meshCandidate->refit.buffer->GetByteSize();
        }
        meshPage = {};
        if (!meshCandidate->refit.meshes.empty()) {
            step = meshCandidate->deformationProgram ? Step::Deform : Step::CompileDeformation;
            if (step == Step::CompileDeformation)
                deformationSource = {"inx_ray_deformation", "Infernux/RayDeformation.slang",
                                     std::string(RayDeformationSource())};
            return false;
        }
        StartUpload(
            RayPageUpload(instancePage, RayGeometryStatus::Ready, shadowHistory.revision, shadowHistory.stamps));
        return false;
    }
    return true;
}

std::shared_ptr<const RayWorldMeshPage> RayWorldGeometryBuild::PublishedMeshes()
{
    if (meshCandidate) {
        // A binary TLAS has at most twice as many links as instance records.
        constexpr auto topologyBytesPerDraw = 2 * sizeof(spatial::BvhLink) + sizeof(uint32_t);
        const auto shadowBytes = shadowHistory.instances.capacity() * sizeof(RayShadowInstance) +
                                 shadowHistory.stamps.capacity() * sizeof(RayGeometryWord);
        uint64_t refitBytes = meshCandidate->refit.meshes.capacity() * sizeof(RayRefitMesh);
        for (const auto &mesh : meshCandidate->refit.meshes)
            refitBytes += mesh.levels.capacity() * sizeof(RayRefitRange);
        storage->ReduceTo(meshCandidate->draws.capacity() * (sizeof(DrawMeshPlacement) + topologyBytesPerDraw) +
                          meshCandidate->referenceAreas.capacity() * sizeof(float) + shadowBytes +
                          meshCandidate->skinStorage + refitBytes);
        meshCandidate->storage = std::move(storage);
        sharedMeshes = std::move(meshCandidate);
    }
    sharedMeshes->topology = std::move(instancePage.topology);
    sharedMeshes->shadowHistory = std::move(shadowHistory);
    return std::move(sharedMeshes);
}

RayWorldGeometry::RayWorldGeometry(FrameWorkOwner &owner, rhi::Device &device) : m_owner(owner), m_device(device)
{
    FrameWorkSource source;
    source.observe = [this] { Observe(); };
    source.refresh = source.observe;
    source.describe = [this] { return Describe(); };
    source.advance = [this](bool gpu, uint64_t) { return Advance(gpu); };
    m_source = owner.AddSource(std::move(source));
}

RayWorldGeometry::~RayWorldGeometry()
{
    m_owner.RemoveSource(m_source);
    if (m_job.IsValid()) {
        m_job.Cancel();
        JobSystem::Get().WaitPassiveCompletion(m_job);
    }
}

RayWorldGeometryBinding RayWorldGeometry::Acquire(const std::shared_ptr<const RenderWorldFrame> &world,
                                                  std::shared_ptr<const RayWorldGeometryPublication> &publication)
{
    const auto revision = GetRevision(*world);
    if (!publication || publication->revision != revision) {
        auto meshes = publication && publication->revision.meshes == revision.meshes ? publication->meshes : nullptr;
        auto skinBasis = publication && revision.SharesDrawSlots(publication->revision) ? publication->meshes : nullptr;
        if (skinBasis && skinBasis->skins.empty() && skinBasis->refit.meshes.empty())
            skinBasis.reset();
        publication.reset();
        m_publications.erase(std::remove_if(m_publications.begin(), m_publications.end(),
                                            [](const auto &entry) { return entry.expired(); }),
                             m_publications.end());
        for (const auto &entry : m_publications) {
            auto value = entry.lock();
            if (!value || !revision.SharesDrawSlots(value->revision))
                continue;
            if (!skinBasis && value->meshes && (!value->meshes->skins.empty() || !value->meshes->refit.meshes.empty()))
                skinBasis = value->meshes;
            if (value->revision.meshes != revision.meshes)
                continue;
            if (value->revision == revision) {
                publication = std::move(value);
                break;
            }
            if (!meshes)
                meshes = value->meshes;
        }
        if (!publication) {
            auto value = std::make_shared<RayWorldGeometryPublication>(world);
            value->meshes = std::move(meshes);
            if (!value->meshes)
                value->skinBasis = std::move(skinBasis);
            m_publications.emplace_back(value);
            publication = std::move(value);
            m_owner.Wake(m_source);
        }
    }
    if (publication->buffer)
        return {publication->buffer,           publication->meshes->buffer,     false,
                publication->visibilitySerial, publication->visibilityLighting, publication->producer};
    if (publication->status == RayGeometryStatus::Pending)
        return {{}, {}, true};
    if (!m_unavailableBuffer) {
        const RayPageUpload layout(RayInstancePage{}, RayGeometryStatus::Unavailable);
        auto buffer = AllocatePage(m_device, layout.bytes);
        WritePage(m_device, *buffer, 0, layout.Section(0).data, layout.bytes);
        m_unavailableBuffer = std::move(buffer);
    }
    if (!m_emptyMeshBuffer) {
        const RayPageUpload layout(RayMeshPage{});
        auto buffer = AllocatePage(m_device, layout.bytes);
        WritePage(m_device, *buffer, 0, layout.Section(0).data, layout.bytes);
        m_emptyMeshBuffer = std::move(buffer);
    }
    return {m_unavailableBuffer, m_emptyMeshBuffer, false};
}

std::shared_ptr<RayWorldGeometryPublication> RayWorldGeometry::FindPending() const
{
    for (const auto &entry : m_publications) {
        auto value = entry.lock();
        if (value && value->world)
            return value;
    }
    return {};
}

void RayWorldGeometry::Observe()
{
    if (!m_job.IsValid() || !m_job.IsComplete())
        return;
    m_job = {};
    CollectReceipt();
}

void RayWorldGeometry::CollectReceipt()
{
    m_owner.ObserveResource(m_workResource, m_active->elapsed, m_submittedOpportunity, m_workGrant);
    if (m_active->step != RayWorldGeometryBuild::Step::BuildRecovered)
        m_active->decodeStorage.reset();
}

void RayWorldGeometry::CollectDeformationReceipt()
{
    const auto elapsed = m_active->deformation->Elapsed();
    m_owner.ObserveResource(FrameWorkResource::Gpu, elapsed, m_submittedOpportunity, m_workGrant);
    m_owner.ObserveResource(FrameWorkResource::Transfer, m_active->deformation->TransferBytes(), m_submittedOpportunity,
                            m_active->deformationTransferGrant);
    if (elapsed)
        m_active->DeformationPlan().predictedNanoseconds = elapsed;
}

FrameWorkFacts RayWorldGeometry::Describe()
{
    FrameWorkFacts facts;
    if (m_active) {
        facts.pending = true;
        if (m_active->step == RayWorldGeometryBuild::Step::Deform) {
            if (m_active->publication.expired()) {
                facts.gpuBusy = m_active->deformationSubmitted && !m_active->deformation->IsComplete();
                facts.host.ready = !facts.gpuBusy;
                return facts;
            }
            if (!m_active->deformation)
                facts.host.ready = m_owner.CanAdmitResource(FrameWorkResource::Cpu, 0);
            else if (!m_active->deformationSubmitted)
                facts.gpu.ready =
                    m_owner.CanAdmitResource(FrameWorkResource::Gpu, m_active->DeformationPlan().predictedNanoseconds,
                                             true) &&
                    m_owner.CanAdmitResource(FrameWorkResource::Transfer, m_active->deformation->TransferBytes(), true);
            else {
                facts.host.ready = m_active->deformation->IsComplete();
                facts.gpuBusy = !facts.host.ready;
            }
            return facts;
        }
        facts.workerBusy = m_job.IsValid();
        if (!facts.workerBusy) {
            const auto resource = m_active->Resource();
            const auto prediction = resource == FrameWorkResource::Transfer
                                        ? m_active->UploadBytes(m_owner.ResourceRemaining(resource))
                                        : 0;
            facts.host.ready = m_active->publication.expired() || m_active->failure ||
                               m_owner.CanAdmitResource(resource, prediction, true);
            facts.ioReady = facts.host.ready && resource == FrameWorkResource::Io;
        }
    } else if (FindPending()) {
        facts.pending = true;
        facts.host.ready = m_owner.CanAdmitResource(FrameWorkResource::Cpu, 0);
    }
    return facts;
}

bool RayWorldGeometry::Advance(bool gpu)
{
    if (m_job.IsValid())
        return false;
    const auto publication = m_active ? m_active->publication.lock() : FindPending();
    if (!publication) {
        const bool wasActive = static_cast<bool>(m_active);
        if (m_active && m_active->deformationSubmitted) {
            if (!m_active->deformation->IsComplete())
                return false;
            CollectDeformationReceipt();
        }
        m_active.reset();
        return wasActive;
    }
    try {
        if (!m_active) {
            auto build = std::make_shared<RayWorldGeometryBuild>();
            build->publication = publication;
            build->world = std::move(publication->world);
            build->deformationTarget = m_device.GetShaderSourceLanguage() == rhi::ShaderSourceLanguage::Wgsl
                                           ? ShaderCodeTarget::Wgsl
                                           : ShaderCodeTarget::SpirV;
            build->sharedMeshes = publication->meshes;
            build->skinBasis = std::move(publication->skinBasis);
            if (build->sharedMeshes) {
                build->step = RayWorldGeometryBuild::Step::CollectInstances;
                build->workspace = AddStorage(
                    build->workspace,
                    build->sharedMeshes->draws.size() * (512u + sizeof(RayShadowInstance) + sizeof(RayEmissiveSource)) +
                        build->sharedMeshes->shadowHistory.stamps.size() * 2u * sizeof(RayGeometryWord));
                for (const auto &draw : build->sharedMeshes->draws)
                    build->workspace =
                        AddStorage(build->workspace, uint64_t{draw.triangleCount} * sizeof(RayGeometryWord));
            }
            m_active = std::move(build);
        }
        if (m_active->failure)
            std::rethrow_exception(m_active->failure);
        const auto resource = m_active->Resource();
        const auto bytes =
            resource == FrameWorkResource::Transfer ? m_active->UploadBytes(m_owner.ResourceRemaining(resource)) : 0;
        if (resource != FrameWorkResource::Gpu && !m_owner.CanAdmitResource(resource, bytes, true))
            return false;
        if (resource == FrameWorkResource::Transfer) {
            const bool done = m_active->Upload(m_device, bytes);
            m_owner.RecordAcceptedResource(resource, bytes + std::exchange(m_active->topologyUploadBytes, 0));
            if (done) {
                if (m_active->NeedsDeformation()) {
                    m_active->step = RayWorldGeometryBuild::Step::Deform;
                    return true;
                }
                publication->buffer = std::move(m_active->buffer);
                publication->visibilitySerial = m_active->VisibilitySerial();
                publication->meshes = m_active->PublishedMeshes();
                publication->status = RayGeometryStatus::Ready;
                m_active.reset();
            }
            return true;
        }
        if (resource == FrameWorkResource::Gpu) {
            if (!m_active->deformation) {
                const auto before = std::chrono::steady_clock::now();
                const auto opportunity = m_owner.Opportunity();
                const auto grant = m_owner.RecordAcceptedResource(FrameWorkResource::Cpu, 0);
                m_active->PrepareDeformation(m_device);
                const auto elapsed = std::chrono::steady_clock::now() - before;
                m_owner.ObserveResource(FrameWorkResource::Cpu,
                                        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count(),
                                        opportunity, grant);
                return true;
            }
            if (!m_active->deformationSubmitted) {
                if (!gpu)
                    return false;
                if (!m_active->deformation->Submit())
                    return false;
                m_active->deformationSubmitted = true;
                m_submittedOpportunity = m_owner.Opportunity();
                m_workGrant =
                    m_owner.RecordAcceptedResource(resource, m_active->DeformationPlan().predictedNanoseconds);
                m_active->deformationTransferGrant =
                    m_owner.RecordAcceptedResource(FrameWorkResource::Transfer, m_active->deformation->TransferBytes());
                return true;
            }
            if (!m_active->deformation->IsComplete())
                return false;
            CollectDeformationReceipt();
            publication->buffer = m_active->deformation->Output();
            publication->producer = m_active->deformation->Submission();
            publication->visibilitySerial = m_active->VisibilitySerial();
            publication->meshes = m_active->PublishedMeshes();
            publication->status = RayGeometryStatus::Ready;
            m_active.reset();
            return true;
        }
        auto &jobs = JobSystem::Get();
        if (m_active->step == RayWorldGeometryBuild::Step::Plan) {
            const auto opportunity = m_owner.Opportunity();
            const auto grant = m_owner.RecordAcceptedResource(resource, 0);
            const auto started = std::chrono::steady_clock::now();
            const bool supported = m_active->Plan(jobs.IsInline() ? InlineWorkWindow : FullBuild);
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started);
            m_owner.ObserveResource(resource, static_cast<uint64_t>(elapsed.count()), opportunity, grant);
            if (!supported) {
                publication->status = RayGeometryStatus::Unavailable;
                m_active.reset();
            }
            return true;
        }
        const auto decode = m_active->step == RayWorldGeometryBuild::Step::Decode ? m_active->DecodeWorkspace() : 0;
        if (AddStorage(m_active->workspace, decode) > m_owner.CpuStorageLimit())
            throw std::length_error("Build and mesh decode exceed the CPU storage limit");
        if (!m_active->storage) {
            m_active->storage = m_owner.TryReserveCpuStorage(m_active->workspace);
            if (!m_active->storage)
                return false;
        }
        if (decode) {
            m_active->decodeStorage = m_owner.TryReserveCpuStorage(decode);
            if (!m_active->decodeStorage)
                return false;
        }
        if (!jobs.IsInline())
            m_job = jobs.Schedule([build = m_active] { build->Run(FullBuild); },
                                  resource == FrameWorkResource::Io ? JobDomain::IO : JobDomain::Render,
                                  JobPriority::Low, JobAffinity::WorkerOnly);
        m_workResource = resource;
        m_submittedOpportunity = m_owner.Opportunity();
        m_workGrant = m_owner.RecordAcceptedResource(resource, 0);
        if (jobs.IsInline()) {
            m_active->Run(InlineWorkWindow);
            CollectReceipt();
        }
    } catch (const std::exception &error) {
        INXLOG_ERROR("World ray geometry preparation failed: ", error.what());
        publication->world.reset();
        publication->status = RayGeometryStatus::Unavailable;
        m_active.reset();
    }
    return true;
}
} // namespace infernux::raytracing
