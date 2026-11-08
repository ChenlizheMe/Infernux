#include "Infernux.h"
#include <algorithm>
#include <atomic>
#include <exception>
#include <function/renderer/gui/MeshPreviewPreparation.h>
#include <function/resources/AssetRegistry/AssetBytes.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxMesh/ModelMeshReference.h>
#include <optional>
#include <platform/filesystem/InxPath.h>
#include <stdexcept>

namespace infernux
{
struct MeshPreviewPreparation
{
    enum class Phase
    {
        Source,
        Parse,
        Dependencies,
        Materials,
        Aggregate,
        Ready
    };
    Phase phase = Phase::Source;
    std::string guid;
    std::optional<uint64_t> sourceGeneration;
    bool prefab = false;
    PreviewAssetLoads loads;
    std::shared_ptr<FrameCpuStorageClaim> aggregateStorage;
    PrefabPreviewLayout layout;
    std::vector<PrefabPreviewPart> parts;
    std::shared_ptr<InxMesh> mesh;
    std::vector<std::shared_ptr<InxMaterial>> sourceMaterials;
    std::vector<std::shared_ptr<InxMaterial>> materials;
    std::atomic<bool> workerComplete{false};
    std::exception_ptr failure;
};

Infernux::PreviewStep Infernux::DescribeMeshPreviewRequest(const MeshPreviewRequest &request) const
{
    const auto state = m_meshPreviewStates.find(request.resourceKey);
    if (state == m_meshPreviewStates.end() || state->second.pendingGeneration != request.generation ||
        !request.preparation || request.generation != state->second.generation ||
        request.meshFilePath != state->second.meshFilePath)
        return PreviewStep::Host;
    const auto &prepared = *request.preparation;
    using Phase = MeshPreviewPreparation::Phase;
    const bool gpu = prepared.phase == Phase::Materials || prepared.phase == Phase::Ready;
    if (!gpu && (prepared.phase == Phase::Source || prepared.phase == Phase::Dependencies ||
                 prepared.workerComplete.load(std::memory_order_acquire)))
        return PreviewStep::Host;
    if (GetPreviewDependencyRevision(request.meshFilePath) != state->second.lastDependencyRevision)
        return PreviewStep::Host;
    return gpu ? PreviewStep::Gpu : PreviewStep::Wait;
}

bool Infernux::RefreshMeshPreviewDependency(MeshPreviewState &state) const
{
    const auto revision = GetPreviewDependencyRevision(state.meshFilePath);
    if (revision == state.lastDependencyRevision)
        return false;
    state.lastDependencyRevision = revision;
    ++state.generation;
    return true;
}

void Infernux::ObserveSubmittedPreviews()
{
    std::lock_guard lock(m_previewResultMutex);
    for (auto &[key, state] : m_materialPreviewStates) {
        if ((state.renderTicket || state.upload.version) && RefreshMaterialPreviewDependency(state))
            ScheduleDirtyMaterialPreview(key, state);
    }
    for (auto &[key, state] : m_meshPreviewStates) {
        if (state.renderTicket && RefreshMeshPreviewDependency(state))
            ScheduleDirtyMeshPreview(key, state);
    }
}

bool Infernux::FinishMeshPreviewPreparation(const MeshPreviewRequest &request, std::shared_ptr<PreviewImage> image)
{
    std::lock_guard lock(m_previewResultMutex);
    const auto found = m_meshPreviewStates.find(request.resourceKey);
    if (found == m_meshPreviewStates.end() || found->second.pendingGeneration != request.generation)
        return false;
    auto &state = found->second;
    state.pendingGeneration = 0;
    if (image) {
        state.renderTicket = std::move(image);
        state.renderGeneration = request.generation;
        m_hasPreviewPumpWork.store(true, std::memory_order_release);
    } else {
        state.failedGeneration = request.generation;
    }
    ScheduleDirtyMeshPreview(found->first, state);
    return true;
}

bool Infernux::PrepareMeshPreview(MeshPreviewRequest &request, bool gpu, std::shared_ptr<InxMesh> &mesh,
                                  std::vector<std::shared_ptr<InxMaterial>> &materials, bool &progressed)
{
    auto &registry = AssetRegistry::Instance();
    if (request.preparation &&
        ((request.preparation->sourceGeneration &&
          *request.preparation->sourceGeneration != registry.GetAssetContentGeneration(request.preparation->guid)) ||
         !request.preparation->loads.IsCurrent())) {
        request.geometry.reset();
        request.preparation.reset();
        progressed = true;
    }
    if (!request.preparation) {
        if (gpu)
            return false;
        auto prepared = std::make_shared<MeshPreviewPreparation>();
        const auto *database = registry.GetAssetDatabase();
        prepared->guid =
            database ? database->GetGuidFromPath(SplitModelMeshReference(request.meshFilePath).first) : std::string();
        if (prepared->guid.empty())
            throw std::runtime_error("Mesh preview source is no longer registered: " + request.meshFilePath);
        prepared->prefab = IsPrefabPreviewPath(request.meshFilePath);
        request.preparation = std::move(prepared);
        progressed = true;
    }
    const auto &prepared = request.preparation;
    using Phase = MeshPreviewPreparation::Phase;
    if (prepared->phase == Phase::Source) {
        if (!registry.GetAssetDatabase()->GetMetaByGuid(prepared->guid))
            throw std::runtime_error("Mesh preview source was retired: " + request.meshFilePath);
        if (!prepared->loads.Require(prepared->guid, prepared->prefab ? ResourceType::DefaultText : ResourceType::Mesh,
                                     progressed))
            return false;
        if (prepared->prefab) {
            const auto source = registry.GetAsset<AssetBytes>(prepared->guid);
            if (!source)
                return false;
            prepared->sourceGeneration = registry.GetAssetContentGeneration(prepared->guid);
            prepared->phase = Phase::Parse;
            EnqueuePreviewTask([prepared, source] {
                try {
                    prepared->parts = ParsePrefabPreview(source->View());
                } catch (...) {
                    prepared->failure = std::current_exception();
                }
                prepared->workerComplete.store(true, std::memory_order_release);
            });
            progressed = true;
            return false;
        }
        prepared->mesh = registry.GetAsset<InxMesh>(prepared->guid);
        if (!prepared->mesh)
            return false;
        if (!prepared->mesh->HasCpuGeometry())
            throw std::runtime_error("Mesh preview cannot access released CPU geometry: " + request.meshFilePath);
        if (request.meshFilePath.find(ModelMeshToken) != std::string::npos)
            prepared->mesh = prepared->mesh->CreateModelNodeCopy(SplitModelMeshReference(request.meshFilePath).second);
        prepared->phase = Phase::Dependencies;
        progressed = true;
    }
    if (prepared->phase == Phase::Parse || prepared->phase == Phase::Aggregate) {
        if (!prepared->workerComplete.load(std::memory_order_acquire))
            return false;
        if (prepared->failure)
            std::rethrow_exception(prepared->failure);
        progressed = true;
        if (prepared->phase == Phase::Parse) {
            // Only the parsed parts remain inputs to aggregation. Let pressure
            // reclaim the source document without losing its revision guard.
            prepared->loads.Clear();
            prepared->phase = Phase::Dependencies;
        } else {
            prepared->parts.clear();
            prepared->phase = Phase::Ready;
        }
    }
    if (prepared->phase == Phase::Dependencies) {
        if (!prepared->prefab) {
            if (!TryPrepareModelPreviewMaterials(*prepared->mesh, prepared->mesh->GetSubMeshes(), prepared->loads,
                                                 prepared->sourceMaterials, progressed))
                return false;
        } else {
            bool ready = true;
            for (const auto &part : prepared->parts)
                ready = prepared->loads.Require(part.meshGuid, ResourceType::Mesh, progressed) && ready;
            if (!ready)
                return false;

            const auto *database = registry.GetAssetDatabase();
            std::vector<std::shared_ptr<InxMaterial>> candidates;
            for (auto &part : prepared->parts) {
                const auto imported = registry.GetAsset<InxMesh>(part.meshGuid);
                if (imported) {
                    if (!imported->HasCpuGeometry())
                        throw std::runtime_error("Prefab preview cannot access released mesh CPU geometry: " +
                                                 part.meshGuid);
                    part.geometry = imported->GetGeometrySnapshot();
                }
                if (!part.geometry)
                    continue;
                for (size_t index = 0; index < part.geometry->subMeshes.size(); ++index) {
                    const auto &submesh = part.geometry->subMeshes[index];
                    if (!part.Includes(index, submesh))
                        continue;
                    const auto slot = submesh.materialSlot;
                    std::shared_ptr<InxMaterial> material;
                    if (slot < part.materials.size() && database->GetMetaByGuid(part.materials[slot])) {
                        if (!prepared->loads.Require(part.materials[slot], ResourceType::Material, progressed)) {
                            ready = false;
                            continue;
                        }
                        material = registry.GetAsset<InxMaterial>(part.materials[slot]);
                        if (material && material->IsDeleted())
                            material = registry.GetBuiltinMaterial("ErrorMaterial");
                    }
                    if (!material) {
                        if (imported && slot < imported->GetMaterialSlotData().size() &&
                            !prepared->loads.Require(imported->GetMaterialSlotData()[slot].materialGuid,
                                                     ResourceType::Material, progressed)) {
                            ready = false;
                            continue;
                        }
                        material = GetModelPreviewMaterial(imported.get(), slot);
                    }
                    candidates.push_back(std::move(material));
                }
            }
            if (!ready)
                return false;
            prepared->sourceMaterials = std::move(candidates);
            prepared->layout = MeasurePrefabPreviewGeometry(prepared->parts);
        }
        prepared->phase = Phase::Materials;
        progressed = true;
    }
    if (prepared->phase == Phase::Materials) {
        if (!gpu)
            return false;
        if (!m_renderer->PrepareMeshPreviewMaterials(prepared->sourceMaterials, prepared->materials, progressed))
            return false;
        const bool drawable = std::any_of(prepared->materials.begin(), prepared->materials.end(),
                                          [](const auto &material) { return bool(material); });
        if (!drawable) {
            prepared->mesh.reset();
        } else if (prepared->prefab) {
            const auto bytes = prepared->layout.StorageBytes();
            if (bytes > m_frameWork.CpuStorageLimit())
                throw std::length_error("Prefab preview geometry exceeds host CPU storage capacity");
            prepared->aggregateStorage = m_frameWork.TryReserveCpuStorage(bytes);
            if (!prepared->aggregateStorage)
                return false;
            prepared->phase = Phase::Aggregate;
            prepared->workerComplete.store(false, std::memory_order_relaxed);
            EnqueuePreviewTask([weak = std::weak_ptr(prepared)] {
                if (const auto work = weak.lock()) {
                    try {
                        work->mesh = BuildPrefabPreviewGeometry(work->parts, work->layout);
                    } catch (...) {
                        work->failure = std::current_exception();
                    }
                    work->workerComplete.store(true, std::memory_order_release);
                }
            });
            progressed = true;
            return false;
        }
        prepared->phase = Phase::Ready;
        progressed = true;
    }
    if (!gpu)
        return false;
    if (!prepared->loads.IsCurrent()) {
        request.geometry.reset();
        request.preparation.reset();
        progressed = true;
        return false;
    }
    mesh = prepared->mesh;
    materials = prepared->materials;
    return true;
}
} // namespace infernux
