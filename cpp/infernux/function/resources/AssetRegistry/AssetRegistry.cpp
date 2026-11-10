#include "AssetRegistry.h"

#include <core/log/InxLog.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxMesh/InxMesh.h>
#include <function/resources/InxSkinnedMesh/InxSkinnedMesh.h>
#include <function/resources/InxTexture/InxTexture.h>
#include <function/resources/InxTexture/TextureReadTask.h>
#include <function/resources/PhysicMaterial/PhysicMaterial.h>
#include <function/resources/ShaderAsset/ShaderAsset.h>

#include <platform/filesystem/InxPath.h>
#include <platform/filesystem/IoStep.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <type_traits>
#include <unordered_set>

namespace infernux
{
namespace
{
detail::FileIoWindow PlanWorkerRead(WorkerAssetLoadPlan::Read read, uint64_t allowance, double bytesPerNanosecond,
                                    uint64_t controlNanoseconds)
{
    if (!read.bytes)
        return {0, controlNanoseconds};
    if (read.kind == WorkerAssetLoadPlan::ReadKind::RangedFile)
        return detail::PlanFileIoWindow(read.bytes, allowance, bytesPerNanosecond);
    return {read.bytes, detail::PredictFileIoNanoseconds(read.bytes, bytesPerNanosecond)};
}
} // namespace

// =============================================================================
// Singleton
// =============================================================================

AssetRegistry &AssetRegistry::Instance()
{
    // Intentionally leaked — Shutdown() runs explicitly in Infernux::Cleanup().
    static AssetRegistry *instance = new AssetRegistry();
    return *instance;
}

// =============================================================================
// Lifecycle
// =============================================================================

void AssetRegistry::Initialize(std::unique_ptr<AssetDatabase> adb)
{
    if (m_initialized)
        throw std::logic_error("AssetRegistry::Initialize may only be called once per engine lifetime");
    if (!adb)
        throw std::invalid_argument("AssetRegistry::Initialize requires an AssetDatabase");

    m_assetDb = std::move(adb);
    AssetDependencyGraph::SetCurrent(m_assetDb->GetDependencyGraph());
    m_ownerThread = std::this_thread::get_id();
    m_accessSerial = 0;
    CollectCpuStorage();
    m_cpuBudgetBytes = 512ULL * 1024ULL * 1024ULL;
    m_cpuEvictionCount = 0;
    m_runtimeMeshSerial = 0;
    m_textureReadProfile = std::make_shared<TextureReadProfile>();
    m_initialized = true;
}

void AssetRegistry::Shutdown()
{
    const auto graph = m_assetDb ? m_assetDb->GetDependencyGraph() : nullptr;
    DrainPendingLoads();
    m_loadedAssets.clear();
    CollectCpuStorage();
    m_accessSerial = 0;
    m_cpuEvictionCount = 0;
    m_runtimeMeshSerial = 0;
    m_assetMutationGenerations.clear();
    m_assetContentGenerations.clear();
    m_assetRuntimeVersions.clear();
    m_meshGpuViewResidency.clear();
    m_assetRuntimeTypes.clear();
    m_pendingLoads.clear();
    m_textureReads.clear();
    m_textureReadProfile.reset();
    m_cpuResidency.reset();
    m_frameWork = nullptr;
    // The database's accepted metadata/import jobs retain loader pointers.
    // Its destructor waits for them before the loader registry can retire.
    m_assetDb.reset();
    m_loaders.clear();
    m_builtinMaterials.clear();
    m_runtimeMaterials.clear();
    if (graph) {
        graph->Clear();
        AssetDependencyGraph::SetCurrent(nullptr);
    }
    m_ownerThread = {};
    m_initialized = false;
}

// =============================================================================
// Loader registration
// =============================================================================

void AssetRegistry::RegisterLoader(ResourceType type, std::unique_ptr<IAssetLoader> loader)
{
    if (!loader)
        throw std::invalid_argument("AssetRegistry loader cannot be null");
    if (!m_loaders.emplace(type, std::move(loader)).second)
        throw std::logic_error("AssetRegistry loader already registered for ResourceType");
}

IAssetLoader *AssetRegistry::GetLoader(ResourceType type) const
{
    auto it = m_loaders.find(type);
    return it != m_loaders.end() ? it->second.get() : nullptr;
}

void AssetRegistry::PopulateAssetDatabaseLoaders()
{
    if (!m_assetDb)
        throw std::logic_error("AssetRegistry has no AssetDatabase");
    for (auto &[type, loader] : m_loaders) {
        m_assetDb->SetMetaLoader(type, loader.get());
    }
}

// =============================================================================
// Internal load helper
// =============================================================================

RuntimeAssetPayload AssetRegistry::LoadAssetInternal(const std::string &filePath, const std::string &guid,
                                                     ResourceType type)
{
    if (m_assetDb->IsStagingRuntimeArtifact(guid, type)) {
        if (m_frameWork && m_frameWork->IsActive())
            return {};
        m_assetDb->CompleteRuntimeArtifactPublication();
        if (m_assetDb->GetPathFromGuid(guid).empty())
            return {};
        auto ticket = BeginLoadAsset(guid, type);
        ticket->Wait();
        TryCommitAssetLoad(ticket);
        return ticket->m_payload;
    }
    if (type == ResourceType::Mesh) {
        if (auto pending = GetPendingMeshReload(guid)) {
            if (m_frameWork->IsActive())
                return {};
            pending->Wait();
            TryCommitAssetLoad(pending);
            return pending->m_payload;
        }
    }
    auto loaderIt = m_loaders.find(type);
    if (loaderIt == m_loaders.end()) {
        INXLOG_WARN("AssetRegistry: no loader registered for ResourceType ", static_cast<int>(type));
        return nullptr;
    }

    auto payload = loaderIt->second->Load(filePath, guid, m_assetDb.get());
    if (!payload) {
        INXLOG_WARN("AssetRegistry: loader returned nullptr for '", filePath, "' (GUID: ", guid, ")");
        return nullptr;
    }

    InsertEntry(guid, payload, type);
    loaderIt->second->OnPublished(payload, guid, m_assetDb.get());
    ++m_assetMutationGenerations[guid];
    (void)TrimCpuBudget();
    return payload;
}

uint64_t AssetRegistry::NextRuntimeVersion(const std::string &guid)
{
    m_meshGpuViewResidency.erase(guid);
    uint64_t &version = m_assetRuntimeVersions[guid];
    if (version == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("Asset runtime version overflow for GUID: " + guid);
    return ++version;
}

size_t AssetRegistry::TrackPayloadStorage(ResourceType type, const RuntimeAssetPayload &payload)
{
    if (type == ResourceType::Mesh)
        return TrackMeshStorage(payload.GetStorageOwner(), *payload.Get<InxMesh>());
    const auto bytes = m_loaders.at(type)->EstimateRuntimeBytes(payload);
    if (!bytes)
        throw std::logic_error("Asset loader reported zero bytes for a non-empty runtime payload");
    TrackCpuStorage(payload.GetStorageOwner(), bytes);
    MaybeCollectCpuStorage();
    return bytes;
}

size_t AssetRegistry::TrackMeshStorage(std::weak_ptr<const void> owner, const InxMesh &mesh)
{
    const auto objectBytes = mesh.GetObjectMemoryBytes();
    size_t bytes = objectBytes;
    const auto track = [&](const auto &block) {
        if (block) {
            const auto blockBytes = block->GetRuntimeMemoryBytes();
            TrackCpuStorage(block, blockBytes);
            bytes += blockBytes;
        }
    };
    const auto geometry = mesh.GetGeometrySnapshot();
    const auto source = mesh.GetModelSourceGeometry();
    track(geometry);
    if (source != geometry)
        track(source);
    track(mesh.GetSkinnedData());
    TrackCpuStorage(std::move(owner), objectBytes);
    MaybeCollectCpuStorage();
    return bytes;
}

void AssetRegistry::TrackCpuStorage(std::weak_ptr<const void> owner, size_t bytes)
{
    auto &recorded = m_cpuStorage[std::move(owner)];
    const auto otherBytes = m_cpuStorageBytes - recorded;
    if (bytes > (std::numeric_limits<size_t>::max)() - otherBytes)
        throw std::overflow_error("AssetRegistry CPU residency byte total overflow");
    m_cpuStorageBytes = otherBytes + bytes;
    recorded = bytes;
}

void AssetRegistry::CollectCpuStorage() const noexcept
{
    for (auto it = m_cpuStorage.begin(); it != m_cpuStorage.end();) {
        if (it->first.expired()) {
            m_cpuStorageBytes -= it->second;
            it = m_cpuStorage.erase(it);
        } else {
            ++it;
        }
    }
    m_cpuStorageChanges = 0;
}

void AssetRegistry::MaybeCollectCpuStorage()
{
    // Keep publication cheap when below budget, without retaining a control
    // block for every historical asset generation during hot reload.
    if (++m_cpuStorageChanges >= std::max<size_t>(64, m_cpuStorage.size() / 4))
        CollectCpuStorage();
}

size_t AssetRegistry::GetTotalCpuBytes() const noexcept
{
    CollectCpuStorage();
    return m_cpuStorageBytes;
}

void AssetRegistry::InsertEntry(const std::string &guid, RuntimeAssetPayload payload, ResourceType type)
{
    const auto bytes = TrackPayloadStorage(type, payload);
    const auto [entry, inserted] =
        m_loadedAssets.emplace(guid, AssetEntry{std::move(payload), type, 0, bytes, ++m_accessSerial, 0});
    if (!inserted)
        throw std::logic_error("AssetRegistry attempted to replace a loaded asset without invalidation");
    try {
        auto &runtimeType = m_assetRuntimeTypes[guid];
        entry->second.version = NextRuntimeVersion(guid);
        runtimeType = type;
    } catch (...) {
        m_loadedAssets.erase(entry);
        throw;
    }
}

std::array<std::weak_ptr<const void>, 4> AssetRegistry::CpuStorageOwners(const AssetEntry &entry)
{
    std::array<std::weak_ptr<const void>, 4> owners{entry.payload.GetStorageOwner()};
    if (entry.type == ResourceType::Mesh) {
        const auto mesh = entry.payload.Get<InxMesh>();
        owners[1] = mesh->GetGeometrySnapshot();
        owners[2] = mesh->GetModelSourceGeometry();
        owners[3] = mesh->GetSkinnedData();
    }
    return owners;
}

void AssetRegistry::RemoveEntry(AssetEntryMap::iterator entry)
{
    const auto owners = CpuStorageOwners(entry->second);
    m_meshGpuViewResidency.erase(entry->first);
    m_loadedAssets.erase(entry);
    for (const auto &owner : owners) {
        if (owner.expired()) {
            const auto record = m_cpuStorage.find(owner);
            if (record != m_cpuStorage.end()) {
                m_cpuStorageBytes -= record->second;
                m_cpuStorage.erase(record);
            }
        }
    }
    MaybeCollectCpuStorage();
}

void AssetRegistry::RefreshEntryBytes(AssetEntry &entry)
{
    entry.cpuBytes = TrackPayloadStorage(entry.type, entry.payload);
}

// =============================================================================
// Hot-reload / invalidation
// =============================================================================

bool AssetRegistry::NotifySourceImported(const std::string &guid, ResourceType type,
                                         const std::shared_ptr<AssetLoadTicket> &repairTicket)
{
    const bool repairIsCurrent = repairTicket && IsAssetLoadContentCurrent(repairTicket);
    ++m_assetContentGenerations[guid];
    ++m_assetMutationGenerations[guid];
    const auto entry = m_loadedAssets.find(guid);
    if (type == ResourceType::Mesh) {
        m_meshReloads.erase(guid);
        if (repairIsCurrent) {
            repairTicket->m_expectedContentGeneration = m_assetContentGenerations[guid];
            repairTicket->m_expectedMutationGeneration = m_assetMutationGenerations[guid];
            repairTicket->m_replaceMesh = true;
            m_meshReloads.emplace(guid, repairTicket);
            return true;
        }
        if (entry == m_loadedAssets.end() && !m_assetDb->GetDependencyGraph()->HasRuntimeDependents(guid))
            return false;
        auto ticket = BeginAssetLoad(guid, type, true);
        if (m_frameWork && m_frameWork->IsActive()) {
            m_meshReloads.emplace(guid, std::move(ticket));
        } else {
            ticket->Wait();
            TryCommitAssetLoad(ticket);
        }
        return true;
    }
    if (entry != m_loadedAssets.end() &&
        m_loaders.at(entry->second.type)->GetReloadMode() == AssetReloadMode::Invalidate)
        RemoveEntry(entry);
    return false;
}

std::shared_ptr<AssetLoadTicket> AssetRegistry::GetPendingMeshReload(const std::string &guid) const
{
    const auto found = m_meshReloads.find(guid);
    if (found == m_meshReloads.end() || found->second->m_committed || found->second->m_rejected ||
        !IsAssetLoadContentCurrent(found->second))
        return {};
    return found->second;
}

bool AssetRegistry::ReloadAsset(const std::string &guid)
{
    if (auto pending = GetPendingMeshReload(guid)) {
        if (!m_frameWork->IsActive()) {
            pending->Wait();
            return TryCommitAssetLoad(pending);
        }
        return true;
    }
    auto it = m_loadedAssets.find(guid);
    if (it != m_loadedAssets.end() && it->second.type == ResourceType::Mesh)
        return NotifySourceImported(guid, ResourceType::Mesh);
    ++m_assetContentGenerations[guid];
    ++m_assetMutationGenerations[guid];
    if (it == m_loadedAssets.end())
        return false;

    if (!m_assetDb)
        return false;

    std::string path = m_assetDb->GetPathFromGuid(guid);
    if (path.empty()) {
        INXLOG_WARN("AssetRegistry::ReloadAsset: GUID has no path mapping — ", guid);
        return false;
    }

    auto loaderIt = m_loaders.find(it->second.type);
    if (loaderIt == m_loaders.end())
        return false;

    if (loaderIt->second->GetReloadMode() == AssetReloadMode::Invalidate) {
        RemoveEntry(it);
        return true;
    }
    if (!loaderIt->second->Reload(it->second.payload, path, guid, m_assetDb.get()))
        return false;
    RefreshEntryBytes(it->second);
    it->second.lastAccessSerial = ++m_accessSerial;
    it->second.version = NextRuntimeVersion(guid);
    loaderIt->second->OnPublished(it->second.payload, guid, m_assetDb.get());
    (void)TrimCpuBudget();
    return true;
}

void AssetRegistry::PublishShader(const std::string &guid, std::shared_ptr<ShaderAsset> candidate)
{
    if (!m_initialized || std::this_thread::get_id() != m_ownerThread)
        throw std::logic_error("Shader publication requires the initialized registry owner thread");
    if (guid.empty() || !candidate || !candidate->HasVariant(ShaderCompileTarget::Forward) ||
        (candidate->shaderType != "vertex" && candidate->shaderType != "fragment"))
        throw std::invalid_argument("Shader publication requires a compiled imported stage");
    const auto path = m_assetDb->GetPathFromGuid(guid);
    if (path.empty() || FoldFilesystemPathCase(ResolveFilesystemPath(path)) !=
                            FoldFilesystemPathCase(ResolveFilesystemPath(candidate->filePath)))
        throw std::invalid_argument("Shader candidate does not belong to its imported GUID");
    auto entry = m_loadedAssets.find(guid);
    if (entry != m_loadedAssets.end() && entry->second.type != ResourceType::Shader)
        throw std::invalid_argument("Shader publication conflicts with the resident resource type");
    if (entry == m_loadedAssets.end()) {
        InsertEntry(guid, RuntimeAssetPayload(candidate), ResourceType::Shader);
    } else {
        auto resident = entry->second.payload.Get<ShaderAsset>();
        static_assert(std::is_nothrow_move_assignable_v<ShaderAsset>);
        const auto bytes = candidate->GetRuntimeMemoryBytes();
        TrackCpuStorage(entry->second.payload.GetStorageOwner(), bytes);
        const auto version = NextRuntimeVersion(guid);
        if (resident != candidate)
            *resident = std::move(*candidate);
        entry->second.cpuBytes = bytes;
        entry->second.version = version;
        entry->second.lastAccessSerial = ++m_accessSerial;
    }
    ++m_assetMutationGenerations[guid];
    ++m_assetContentGenerations[guid];
    (void)TrimCpuBudget();
}

void AssetRegistry::UpdateMeshPositions(const std::string &guid, size_t first, const std::vector<glm::vec3> &positions,
                                        const std::optional<std::vector<glm::vec3>> &normals)
{
    if (!m_initialized || std::this_thread::get_id() != m_ownerThread)
        throw std::logic_error("Mesh publication requires the initialized registry owner thread");
    auto entry = m_loadedAssets.find(guid);
    if (entry == m_loadedAssets.end() || entry->second.type != ResourceType::Mesh)
        throw std::invalid_argument("Mesh publication requires a loaded mesh GUID");
    auto mesh = entry->second.payload.Get<InxMesh>();
    mesh->RequireCpuReadable("Mesh.update_vertices");
    const auto &source = mesh->GetVertices();
    if (first > source.size() || positions.size() > source.size() - first)
        throw std::invalid_argument("Mesh position range exceeds vertex count");
    if (normals && normals->size() != positions.size())
        throw std::invalid_argument("Mesh normals must match position count");
    if (positions.empty())
        return;
    std::vector<Vertex> vertices(source.begin() + first, source.begin() + first + positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        if (!std::isfinite(positions[i].x) || !std::isfinite(positions[i].y) || !std::isfinite(positions[i].z))
            throw std::invalid_argument("Mesh positions must be finite");
        vertices[i].pos = positions[i];
        if (normals) {
            const auto &normal = (*normals)[i];
            if (!std::isfinite(normal.x) || !std::isfinite(normal.y) || !std::isfinite(normal.z))
                throw std::invalid_argument("Mesh normals must be finite");
            vertices[i].normal = normal;
        }
    }
    InxMesh candidate = *mesh;
    candidate.UpdateVertexRange(first, vertices);
    PublishMesh(guid, std::move(candidate));
}

std::shared_ptr<InxMesh> AssetRegistry::CreateRuntimeMesh(const std::string &name)
{
    if (!m_initialized || std::this_thread::get_id() != m_ownerThread)
        throw std::logic_error("Runtime Mesh creation requires the initialized registry owner thread");
    if (m_runtimeMeshSerial == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("Runtime Mesh identity serial overflow");

    const std::string guid = "runtime-mesh:" + std::to_string(++m_runtimeMeshSerial);
    auto mesh = std::make_shared<InxMesh>(name.empty() ? "Mesh" : name);
    mesh->SetGuid(guid);
    InsertEntry(guid, RuntimeAssetPayload(mesh), ResourceType::Mesh);
    ++m_assetMutationGenerations[guid];
    (void)TrimCpuBudget();
    return mesh;
}

std::shared_ptr<InxMesh> AssetRegistry::CloneRuntimeMesh(const std::string &guid, const std::string &name)
{
    auto source = GetAsset<InxMesh>(guid);
    if (!source)
        throw std::invalid_argument("Mesh copy requires a loaded Mesh GUID");
    source->RequireCpuReadable("Mesh.copy");
    auto copy = CreateRuntimeMesh(name.empty() ? source->GetName() + " Copy" : name);
    const std::string runtimeGuid = copy->GetGuid();
    InxMesh candidate = *source;
    candidate.SetGuid(runtimeGuid);
    candidate.SetFilePath({});
    candidate.SetName(copy->GetName());
    PublishMesh(runtimeGuid, std::move(candidate));
    return copy;
}

void AssetRegistry::DestroyRuntimeMesh(const std::string &guid)
{
    if (!m_initialized || std::this_thread::get_id() != m_ownerThread)
        throw std::logic_error("Runtime Mesh destruction requires the initialized registry owner thread");
    if (guid.rfind("runtime-mesh:", 0) != 0)
        throw std::invalid_argument("Only transient runtime Mesh resources can be destroyed directly");
    const auto entry = m_loadedAssets.find(guid);
    if (entry == m_loadedAssets.end() || entry->second.type != ResourceType::Mesh)
        throw std::invalid_argument("Runtime Mesh destruction requires a live Mesh identity");

    AssetDependencyGraph::Instance().NotifyEvent(guid, ResourceType::Mesh, AssetEvent::Deleted);
    auto &graph = AssetDependencyGraph::Instance();
    for (const auto &dependentGuid : graph.GetDependents(guid))
        graph.RemoveRuntimeDependency(dependentGuid, guid);
    RemoveEntry(entry);
    ++m_assetMutationGenerations[guid];
    ++m_assetContentGenerations[guid];
    graph.RemoveAsset(guid);
}

void AssetRegistry::PublishMesh(const std::string &guid, InxMesh replacement)
{
    if (!m_initialized || std::this_thread::get_id() != m_ownerThread)
        throw std::logic_error("Mesh publication requires the initialized registry owner thread");
    auto entry = m_loadedAssets.find(guid);
    if (entry == m_loadedAssets.end() || entry->second.type != ResourceType::Mesh)
        throw std::invalid_argument("Mesh publication requires a loaded Mesh GUID");
    auto mesh = entry->second.payload.Get<InxMesh>();
    replacement.SetGuid(mesh->GetGuid());
    replacement.SetFilePath(mesh->GetFilePath());
    const auto bytes = TrackMeshStorage(entry->second.payload.GetStorageOwner(), replacement);
    const auto version = NextRuntimeVersion(guid);
    static_assert(std::is_nothrow_move_assignable_v<InxMesh>);
    *mesh = std::move(replacement);
    entry->second.cpuBytes = bytes;
    entry->second.version = version;
    entry->second.lastAccessSerial = ++m_accessSerial;
    ++m_assetMutationGenerations[guid];
    ++m_assetContentGenerations[guid];
    AssetDependencyGraph::Instance().NotifyEvent(guid, ResourceType::Mesh, AssetEvent::RuntimeModified);
    (void)TrimCpuBudget();
}

void AssetRegistry::InvalidateAsset(const std::string &guid)
{
    m_meshReloads.erase(guid);
    ++m_assetMutationGenerations[guid];
    ++m_assetContentGenerations[guid];
    auto it = m_loadedAssets.find(guid);
    if (it != m_loadedAssets.end()) {
        RemoveEntry(it);
    }
}

void AssetRegistry::RemoveAsset(const std::string &guid)
{
    m_meshReloads.erase(guid);
    ++m_assetMutationGenerations[guid];
    ++m_assetContentGenerations[guid];
    const auto entry = m_loadedAssets.find(guid);
    if (entry != m_loadedAssets.end())
        RemoveEntry(entry);
}

std::shared_ptr<AssetLoadTicket> AssetRegistry::BeginLoadAsset(const std::string &guid, ResourceType type,
                                                               bool preferPublished)
{
    return BeginAssetLoad(guid, type, false, preferPublished);
}

std::shared_ptr<AssetLoadTicket> AssetRegistry::BeginAssetLoad(const std::string &guid, ResourceType type,
                                                               bool replaceMesh, bool preferPublished)
{
    if (!m_initialized || std::this_thread::get_id() != m_ownerThread)
        throw std::logic_error("AssetRegistry::BeginLoadAsset must run on the initialized owner thread");
    if (guid.empty())
        throw std::invalid_argument("AssetRegistry worker load requires a GUID");
    if (const auto metadata = m_assetDb->GetMetaByGuid(guid); metadata && metadata->GetResourceType() != type)
        throw std::invalid_argument("AssetRegistry worker request differs from the catalog resource type: " + guid);
    if (!replaceMesh && type == ResourceType::Mesh && (!preferPublished || !IsLoaded(guid)))
        if (auto pending = GetPendingMeshReload(guid))
            return pending;

    auto ticket = std::make_shared<AssetLoadTicket>();
    ticket->m_registry = this;
    ticket->m_sourceGraph = m_assetDb->GetDependencyGraph();
    ticket->m_guid = guid;
    ticket->m_resourceType = type;
    ticket->m_replaceMesh = replaceMesh;
    ticket->m_ownerThread = m_ownerThread;
    ticket->m_expectedMutationGeneration = m_assetMutationGenerations[guid];
    ticket->m_expectedContentGeneration = m_assetContentGenerations[guid];

    const auto cached = m_loadedAssets.find(guid);
    if (!replaceMesh && cached != m_loadedAssets.end()) {
        if (cached->second.type != type)
            throw std::invalid_argument("AssetRegistry resource type mismatch for GUID: " + guid);
        ticket->m_payload = cached->second.payload;
        cached->second.lastAccessSerial = ++m_accessSerial;
        ticket->m_producerThread = m_ownerThread;
        ticket->m_committed = true;
        return ticket;
    }

    // A model thumbnail, animation inspector and scene consumer can request the
    // same unpublished revision. They share its preparation and publication.
    m_pendingLoads.erase(std::remove_if(m_pendingLoads.begin(), m_pendingLoads.end(),
                                        [](const auto &pending) { return pending.expired(); }),
                         m_pendingLoads.end());
    for (const auto &weak : m_pendingLoads) {
        const auto pending = weak.lock();
        if (pending && pending->m_guid == guid && pending->m_resourceType == type &&
            pending->m_replaceMesh == replaceMesh && !pending->m_committed && !pending->m_rejected &&
            pending->m_expectedMutationGeneration == ticket->m_expectedMutationGeneration &&
            pending->m_expectedContentGeneration == ticket->m_expectedContentGeneration)
            return pending;
    }

    ticket->m_sourcePath = m_assetDb->GetPathFromGuid(guid);
    if (ticket->m_sourcePath.empty())
        throw std::invalid_argument("AssetRegistry worker load GUID has no path: " + guid);
    const auto loader = m_loaders.find(type);
    if (loader == m_loaders.end())
        throw std::invalid_argument("AssetRegistry has no loader for worker request");
    if (!loader->second->SupportsWorkerLoad())
        throw std::invalid_argument("AssetRegistry loader does not support worker loading");
    if (!JobSystem::IsAvailable())
        throw std::logic_error("AssetRegistry worker load requires JobSystem");

    ticket->m_loader = loader->second.get();
    ticket->m_database = m_assetDb.get();
    ticket->m_token = m_nextLoadToken++;
    m_pendingLoads.emplace_back(ticket);
    return ticket;
}

void AssetLoadTicket::Wait()
{
    m_registry->WaitAssetLoad(shared_from_this());
}

void AssetRegistry::WaitAssetLoad(const std::shared_ptr<AssetLoadTicket> &ticket)
{
    do {
        bool progressed = false;
        if (!SubmitAssetLoad(ticket, true, progressed))
            JobSystem::Get().WaitPassive(m_activeLoad->m_job);
        else if (ticket->m_job.IsValid())
            JobSystem::Get().WaitPassive(ticket->m_job);
        ObserveAssetLoad();
    } while (!ticket->IsComplete());
}

bool AssetRegistry::TrySubmitAssetLoad(const std::shared_ptr<AssetLoadTicket> &ticket)
{
    bool progressed = false;
    return SubmitAssetLoad(ticket, true, progressed);
}

void AssetRegistry::RejectSupersededLoad(AssetLoadTicket &ticket)
{
    ticket.m_payload = {};
    ticket.m_workerPlan = {};
    ticket.m_storageClaim.reset();
    ticket.m_failure = std::make_exception_ptr(AssetLoadSuperseded("Asset load was superseded by a newer revision"));
    ticket.m_rejected = true;
}

void AssetRegistry::ObserveAssetLoad()
{
    if (!m_activeLoad || !m_activeLoad->m_job.IsComplete())
        return;
    if (m_frameWork) {
        m_frameWork->ObserveResource(FrameWorkResource::Io, m_activeLoad->m_executionNanoseconds, m_loadFrame,
                                     m_loadGrant);
        if (m_loadGrant && !m_activeLoad->m_failure) {
            if (!m_acceptedRead.bytes)
                m_loadControlNanoseconds[m_activeLoad->m_resourceType] = m_activeLoad->m_executionNanoseconds;
            auto &rate = m_readRates[static_cast<size_t>(m_acceptedRead.kind)];
            const auto minimum =
                m_acceptedRead.kind == WorkerAssetLoadPlan::ReadKind::RangedFile ? detail::kMinimumFileWindow : 0;
            detail::ObserveFileIoWindow(rate, m_acceptedRead.bytes, m_activeLoad->m_executionNanoseconds, minimum);
        }
    }
    auto completed = std::exchange(m_activeLoad, {});
    m_loadFrame = m_loadGrant = 0;
    m_acceptedRead = {};
    if (!completed->m_failure && !IsAssetLoadContentCurrent(completed)) {
        RejectSupersededLoad(*completed);
        return;
    }
    if ((completed->m_workerPlan.advance || completed->m_repairNeeded) && !completed->m_payload &&
        !completed->m_failure && !completed->m_rejected) {
        completed->m_job = {};
        return;
    }
    if (completed->m_payload && !completed->m_failure) {
        try {
            (void)TrackPayloadStorage(completed->m_resourceType, completed->m_payload);
        } catch (...) {
            completed->m_payload = {};
            completed->m_failure = std::current_exception();
        }
    }
    completed->m_workerPlan = {};
    completed->m_storageClaim.reset();
}

bool AssetRegistry::AdvanceMeshArtifactRepair(const std::shared_ptr<AssetLoadTicket> &ticket, bool synchronous,
                                              bool &progressed)
{
    if (ticket->m_repairNeeded) {
        if (synchronous) {
            if (m_assetDb->IsRefreshPending())
                m_assetDb->CompletePendingRefresh();
            if (m_assetDb->IsModelReimportPending())
                m_assetDb->CompletePendingModelReimport();
            if (!IsAssetLoadContentCurrent(ticket))
                throw AssetLoadSuperseded("Mesh load was superseded before artifact repair");
        } else if (!m_assetDb->CanBeginMeshArtifactRepair()) {
            return false;
        }
        m_assetDb->BeginMeshArtifactRepair(ticket);
        ticket->m_repairNeeded = false;
        ticket->m_repairPending = true;
        progressed = true;
    }
    if (!synchronous)
        return false;
    m_assetDb->CompletePendingModelReimport();
    return true;
}

void AssetRegistry::FinishMeshArtifactRepair(AssetLoadTicket &ticket, const AssetMutationResult &result)
{
    ticket.m_repairPending = false;
    if (!result.succeeded) {
        ticket.m_failure = std::make_exception_ptr(std::runtime_error(result.error));
        ticket.m_rejected = true;
    }
}

bool AssetRegistry::SubmitAssetLoad(const std::shared_ptr<AssetLoadTicket> &ticket, bool synchronous, bool &progressed)
{
    if (!m_initialized || std::this_thread::get_id() != m_ownerThread)
        throw std::logic_error("Asset load admission requires the initialized registry owner thread");
    ObserveAssetLoad();
    if (ticket && ticket->m_registry == this && !ticket->m_committed &&
        m_assetDb->IsStagingRuntimeArtifact(ticket->m_guid, ticket->m_resourceType)) {
        if (!synchronous)
            return false;
        if (m_frameWork && m_frameWork->IsActive())
            throw std::logic_error("A synchronous asset load cannot wait for staged runtime artifacts");
        m_assetDb->CompleteRuntimeArtifactPublication();
    }
    if (!IsAssetLoadContentCurrent(ticket)) {
        if (ticket && ticket->m_registry == this && !ticket->m_committed && ticket != m_activeLoad)
            RejectSupersededLoad(*ticket);
        throw AssetLoadSuperseded("Asset load ticket is stale before acceptance");
    }
    if (ticket->IsComplete() || ticket->m_job.IsValid())
        return true;
    try {
        if (ticket->m_repairNeeded || ticket->m_repairPending) {
            if (!AdvanceMeshArtifactRepair(ticket, synchronous, progressed))
                return false;
            if (!IsAssetLoadContentCurrent(ticket))
                throw AssetLoadSuperseded("Mesh load was superseded during artifact repair");
            if (ticket->IsComplete())
                return true;
        }
        ticket->m_expectedMutationGeneration = m_assetMutationGenerations[ticket->m_guid];
        const auto cached = m_loadedAssets.find(ticket->m_guid);
        if (!ticket->m_replaceMesh && cached != m_loadedAssets.end()) {
            if (cached->second.type != ticket->m_resourceType)
                throw std::invalid_argument("Asset load type changed during preparation");
            ticket->m_payload = cached->second.payload;
            ticket->m_workerPlan = {};
            ticket->m_storageClaim.reset();
            ticket->m_producerThread = m_ownerThread;
            ticket->m_committed = true;
            progressed = true;
            return true;
        }
        ObserveAssetLoad();
        if (m_activeLoad)
            return false;
        const bool managed = !synchronous && m_frameWork;
        const auto read = ticket->m_workerPlan.read;
        const auto range = managed ? PlanWorkerRead(read, m_frameWork->ResourceRemaining(FrameWorkResource::Io),
                                                    m_readRates[static_cast<size_t>(read.kind)],
                                                    m_loadControlNanoseconds[ticket->m_resourceType])
                                   : detail::FileIoWindow{read.bytes, 0};
        if (managed && !m_frameWork->CanAdmitResource(FrameWorkResource::Io, range.prediction, true))
            return false;
        auto claim = ticket->m_storageClaim;
        if (!claim && managed && ticket->m_workerPlan.advance) {
            const auto bytes = ticket->m_workerPlan.storageBytes;
            if (bytes > m_frameWork->CpuStorageLimit())
                throw std::length_error("Asset load exceeds host CPU storage capacity: " + ticket->m_sourcePath);
            claim = m_frameWork->TryReserveCpuStorage(bytes);
            if (!claim)
                return false;
        }
        ticket->m_job = JobSystem::Get().Schedule(
            [ticket, managed, bytes = range.bytes] {
                const auto started = std::chrono::steady_clock::now();
                ticket->m_producerThread = std::this_thread::get_id();
                try {
                    if (ticket->m_workerPlan.advance) {
                        ticket->m_payload = ticket->m_workerPlan.advance(bytes, ticket->m_workerPlan.read);
                    } else if (ticket->m_resourceType == ResourceType::Mesh && !ticket->m_replaceMesh &&
                               !ticket->m_database->HasCurrentMeshArtifacts(ticket->m_guid)) {
                        ticket->m_repairNeeded = true;
                    } else {
                        if (managed)
                            ticket->m_workerPlan = ticket->m_loader->PlanWorkerLoad(ticket->m_sourcePath,
                                                                                    ticket->m_guid, ticket->m_database);
                        if (!ticket->m_workerPlan.advance)
                            ticket->m_payload =
                                ticket->m_loader->Load(ticket->m_sourcePath, ticket->m_guid, ticket->m_database);
                    }
                    if (!ticket->m_payload && !ticket->m_workerPlan.advance && !ticket->m_repairNeeded)
                        throw std::runtime_error("AssetRegistry worker loader returned an empty payload");
                } catch (...) {
                    ticket->m_failure = std::current_exception();
                }
                ticket->m_executionNanoseconds = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started)
                        .count());
            },
            JobDomain::IO, JobPriority::Low, detail::IoJobAffinity);
        ticket->m_storageClaim = std::move(claim);
        m_activeLoad = ticket;
        progressed = true;
        if (managed) {
            m_loadFrame = m_frameWork->Opportunity();
            m_loadGrant = m_frameWork->RecordAcceptedResource(FrameWorkResource::Io, range.prediction);
            m_acceptedRead = {range.bytes, read.kind};
        }
        return true;
    } catch (...) {
        ticket->m_workerPlan = {};
        ticket->m_storageClaim.reset();
        ticket->m_failure = std::current_exception();
        ticket->m_rejected = true;
        throw;
    }
}

FrameWorkSource AssetRegistry::GetAssetLoadWorkSource(FrameWorkOwner &owner)
{
    m_frameWork = &owner;
    m_readRates.fill(0);
    m_loadControlNanoseconds.clear();
    m_cpuResidency = std::make_shared<FrameCpuResidencySource>(
        FrameCpuResidencySource{[this] { return GetTotalCpuBytes(); },
                                [this](uint64_t target, bool reclaim) { return FitCpuStorage(target, reclaim); }});
    owner.SetCpuResidencySource(m_cpuResidency);
    FrameWorkSource source;
    source.observe = [this] {
        ObserveAssetLoad();
        m_pendingLoads.erase(std::remove_if(m_pendingLoads.begin(), m_pendingLoads.end(),
                                            [this](const auto &weak) {
                                                const auto ticket = weak.lock();
                                                if (!ticket || ticket->m_committed || ticket->m_rejected)
                                                    return true;
                                                if (!ticket->m_job.IsValid() && ticket->m_workerPlan.advance &&
                                                    !IsAssetLoadContentCurrent(ticket)) {
                                                    RejectSupersededLoad(*ticket);
                                                    return true;
                                                }
                                                return false;
                                            }),
                             m_pendingLoads.end());
        for (auto reload = m_meshReloads.begin(); reload != m_meshReloads.end();) {
            if (reload->second->m_committed || reload->second->m_rejected)
                reload = m_meshReloads.erase(reload);
            else
                ++reload;
        }
    };
    source.refresh = source.observe;
    source.describe = [this] {
        FrameWorkFacts facts;
        const auto allowance = m_frameWork->ResourceRemaining(FrameWorkResource::Io);
        for (const auto &weak : m_pendingLoads) {
            const auto ticket = weak.lock();
            if (!ticket || ticket->m_committed || ticket->m_rejected)
                continue;
            const bool complete = ticket->IsComplete();
            const bool replacementReady = ticket->m_replaceMesh && complete;
            facts.pending |= ticket->m_replaceMesh || !complete;
            facts.workerBusy |= ticket->m_job.IsValid() && !ticket->m_job.IsComplete();
            if (m_assetDb->IsStagingRuntimeArtifact(ticket->m_guid, ticket->m_resourceType))
                continue;
            if (facts.host.ready && ticket->m_token >= facts.host.token)
                continue;
            if (replacementReady && !m_frameWork->IsLate()) {
                facts.host = {ticket->m_token, 0, true, false};
                continue;
            }
            if (ticket->m_job.IsValid() || ticket->m_repairPending)
                continue;
            if (ticket->m_repairNeeded) {
                if (m_assetDb->CanBeginMeshArtifactRepair() || !IsAssetLoadContentCurrent(ticket))
                    facts.host = {ticket->m_token, 0, true, false};
                continue;
            }
            if (m_activeLoad)
                continue;
            const auto read = ticket->m_workerPlan.read;
            const auto range = PlanWorkerRead(read, allowance, m_readRates[static_cast<size_t>(read.kind)],
                                              m_loadControlNanoseconds[ticket->m_resourceType]);
            if (!m_frameWork->CanAdmitResource(FrameWorkResource::Io, range.prediction, true))
                continue;
            const auto bytes = ticket->m_workerPlan.storageBytes;
            if (ticket->m_storageClaim || !bytes || m_frameWork->CanReserveCpuStorage(bytes) ||
                bytes > m_frameWork->CpuStorageLimit() || !IsAssetLoadContentCurrent(ticket) ||
                (!ticket->m_replaceMesh && m_loadedAssets.find(ticket->m_guid) != m_loadedAssets.end()))
                facts.host = {ticket->m_token, 0, true, false};
        }
        facts.workerBusy |= m_activeLoad && !m_activeLoad->m_job.IsComplete();
        facts.pending |= bool(m_activeLoad);
        facts.observe = !m_pendingLoads.empty() || bool(m_activeLoad);
        facts.ioReady = facts.host.ready;
        return facts;
    };
    source.advance = [this](bool, uint64_t token) {
        std::shared_ptr<AssetLoadTicket> selected;
        for (const auto &weak : m_pendingLoads) {
            const auto ticket = weak.lock();
            if (!ticket || ticket->m_token != token)
                continue;
            selected = ticket;
            break;
        }
        if (!selected)
            return false;
        bool progressed = false;
        try {
            if (selected->m_replaceMesh && selected->IsComplete())
                return CommitAssetLoad(selected, false, true);
            (void)SubmitAssetLoad(selected, false, progressed);
        } catch (...) {
            selected->m_failure = std::current_exception();
            selected->m_rejected = !selected->m_committed;
            if (selected->m_replaceMesh) {
                try {
                    std::rethrow_exception(selected->m_failure);
                } catch (const AssetLoadSuperseded &) {
                    // The newer source publication owns the next state.
                } catch (const std::exception &error) {
                    INXLOG_ERROR("Mesh reload failed for '", selected->m_sourcePath, "': ", error.what());
                } catch (...) {
                    INXLOG_ERROR("Mesh reload failed for '", selected->m_sourcePath, "'");
                }
            }
            return true;
        }
        return progressed;
    };
    return source;
}

bool AssetRegistry::IsAssetLoadContentCurrent(const std::shared_ptr<AssetLoadTicket> &ticket) const
{
    if (!m_initialized || std::this_thread::get_id() != m_ownerThread)
        throw std::logic_error("Asset load content observation requires the initialized registry owner thread");
    if (!ticket || ticket->m_registry != this || ticket->m_sourceGraph.lock() != m_assetDb->GetDependencyGraph())
        return false;
    const auto content = m_assetContentGenerations.find(ticket->m_guid);
    return ticket->m_expectedContentGeneration == (content == m_assetContentGenerations.end() ? 0 : content->second);
}

bool AssetRegistry::TryCommitAssetLoad(const std::shared_ptr<AssetLoadTicket> &ticket, bool allowStaleIfUnloaded)
{
    return CommitAssetLoad(ticket, allowStaleIfUnloaded, !m_frameWork || !m_frameWork->IsActive());
}

bool AssetRegistry::CommitAssetLoad(const std::shared_ptr<AssetLoadTicket> &ticket, bool allowStaleIfUnloaded,
                                    bool publishReplacement)
{
    if (!ticket || ticket->m_registry != this)
        throw std::invalid_argument("Asset load ticket belongs to another registry");
    if (std::this_thread::get_id() != m_ownerThread)
        throw std::logic_error("AssetRegistry::TryCommitAssetLoad must run on the owner thread");
    if (!m_initialized || ticket->m_sourceGraph.lock() != m_assetDb->GetDependencyGraph())
        throw AssetLoadSuperseded("Asset load ticket belongs to a retired registry world");
    if (ticket->m_committed)
        return true;
    if (m_assetDb->IsStagingRuntimeArtifact(ticket->m_guid, ticket->m_resourceType))
        return false;
    if (!ticket->IsComplete())
        return false;
    ObserveAssetLoad();
    const bool contentCurrent = IsAssetLoadContentCurrent(ticket);
    if (contentCurrent && ticket->m_failure) {
        ticket->m_rejected = true;
        std::rethrow_exception(ticket->m_failure);
    }
    if (contentCurrent && ticket->m_rejected)
        throw std::logic_error("Asset load ticket was already rejected");
    const bool staleAfterMutation = m_assetMutationGenerations[ticket->m_guid] != ticket->m_expectedMutationGeneration;
    const bool canUseStaleUnloadedPayload = contentCurrent && allowStaleIfUnloaded && !ticket->m_replaceMesh &&
                                            m_loadedAssets.find(ticket->m_guid) == m_loadedAssets.end();
    if (!contentCurrent || (staleAfterMutation && !canUseStaleUnloadedPayload)) {
        RejectSupersededLoad(*ticket);
        std::rethrow_exception(ticket->m_failure);
    }
    if (staleAfterMutation)
        INXLOG_DEBUG("AssetRegistry: accepting stale worker payload for unloaded preview asset ", ticket->m_guid);
    if (!ticket->m_payload) {
        ticket->m_rejected = true;
        throw std::logic_error("Asset load ticket completed without a payload");
    }
    const auto existing = m_loadedAssets.find(ticket->m_guid);
    // Only replacing a resident Mesh changes data already visible to the frame.
    // A repaired first load can publish at its ordinary consumption boundary.
    if (ticket->m_replaceMesh && existing != m_loadedAssets.end() && !publishReplacement)
        return false;

    if (ticket->m_replaceMesh) {
        const auto path = m_assetDb->GetPathFromGuid(ticket->m_guid);
        if (path != ticket->m_sourcePath)
            ticket->m_loader->OnRelocated(ticket->m_payload, path);
    }
    if (existing != m_loadedAssets.end()) {
        if (existing->second.type != ticket->m_resourceType)
            throw std::logic_error("Asset load ticket conflicts with the cached resource type");
        if (ticket->m_replaceMesh) {
            auto candidate = ticket->m_payload.Get<InxMesh>();
            auto &entry = existing->second;
            const auto bytes = TrackMeshStorage(entry.payload.GetStorageOwner(), *candidate);
            const auto version = NextRuntimeVersion(ticket->m_guid);
            entry.payload.Get<InxMesh>()->ReplaceImportedContent(std::move(*candidate));
            entry.cpuBytes = bytes;
            entry.version = version;
            ++m_assetMutationGenerations[ticket->m_guid];
        }
        ticket->m_payload = existing->second.payload;
        existing->second.lastAccessSerial = ++m_accessSerial;
    } else {
        InsertEntry(ticket->m_guid, ticket->m_payload, ticket->m_resourceType);
        m_loaders.at(ticket->m_resourceType)->OnPublished(ticket->m_payload, ticket->m_guid, m_assetDb.get());
        ++m_assetMutationGenerations[ticket->m_guid];
    }
    ticket->m_committed = true;
    (void)TrimCpuBudget();
    if (ticket->m_replaceMesh)
        m_assetDb->GetDependencyGraph()->NotifyEvent(ticket->m_guid, ResourceType::Mesh, AssetEvent::Modified);
    return true;
}

std::shared_ptr<TextureReadTask> AssetRegistry::BeginTextureRead(const std::string &guid)
{
    if (!m_initialized || std::this_thread::get_id() != m_ownerThread)
        throw std::logic_error("texture reads require the initialized AssetRegistry owner");
    const auto asset = m_loadedAssets.find(guid);
    if (asset == m_loadedAssets.end() || asset->second.type != ResourceType::Texture)
        throw std::invalid_argument("texture reading requires a loaded texture asset: " + guid);
    const auto metadata = m_assetDb->GetMetaByGuid(guid);
    if (!metadata)
        throw std::invalid_argument("texture source metadata is missing: " + guid);
    auto task =
        std::make_shared<TextureReadTask>(m_assetDb->GetRuntimeArtifactPath(guid, ResourceType::Texture),
                                          metadata->GetDataAs<std::string>("content_hash"), m_textureReadProfile);
    m_textureReads.push_back(task);
    return task;
}

void AssetRegistry::PollTextureReads()
{
    for (auto task = m_textureReads.begin(); task != m_textureReads.end();) {
        if (m_frameWork)
            (*task)->Poll(*m_frameWork);
        if (task->use_count() != 1) {
            ++task;
            continue;
        }
        (*task)->Cancel();
        if ((*task)->IsRetired()) {
            if (m_frameWork)
                (*task)->Poll(*m_frameWork);
            task = m_textureReads.erase(task);
        } else
            ++task;
    }
}

size_t AssetRegistry::GetTextureReadRetainedBytes() const
{
    size_t bytes = 0;
    for (const auto &task : m_textureReads)
        bytes += task->IsPending() ? task->GetReservationBytes(0) : task->GetRetainedBytes();
    return bytes;
}

bool AssetRegistry::HasPendingTextureReads() const
{
    return std::any_of(m_textureReads.begin(), m_textureReads.end(),
                       [](const auto &task) { return task->IsPending(); });
}

void AssetRegistry::DrainPendingLoads() noexcept
{
    for (const auto &pending : m_pendingLoads) {
        const auto ticket = pending.lock();
        if (!ticket || ticket->m_committed)
            continue;
        if (ticket->m_job.IsValid() && !ticket->m_job.IsComplete()) {
            if (!JobSystem::IsAvailable()) {
                INXLOG_ERROR("AssetRegistry pending load outlived JobSystem: ", ticket->m_guid);
                continue;
            }
            JobSystem::Get().WaitPassiveCompletion(ticket->m_job);
        }
        ticket->m_payload = {};
        ticket->m_workerPlan = {};
        ticket->m_storageClaim.reset();
        ticket->m_rejected = true;
    }
    m_pendingLoads.clear();
    m_meshReloads.clear();
    if (m_activeLoad) {
        JobSystem::Get().WaitPassiveCompletion(m_activeLoad->m_job);
        ObserveAssetLoad();
    }
    for (const auto &task : m_textureReads) {
        task->Cancel();
        task->Wait();
        if (m_frameWork)
            task->Poll(*m_frameWork);
    }
    m_textureReads.clear();
}

void AssetRegistry::UpdateLoadedAssetPath(const std::string &guid, const std::string &newPath)
{
    if (guid.empty() || newPath.empty())
        throw std::invalid_argument("AssetRegistry::UpdateLoadedAssetPath requires GUID and destination path");

    // The staged mesh reads GUID-keyed artifacts. A move changes its final
    // location, so rebind that location at publication while keeping the IO.
    auto pendingReload = GetPendingMeshReload(guid);
    const auto mutation = ++m_assetMutationGenerations[guid];
    const auto content = ++m_assetContentGenerations[guid];
    if (pendingReload) {
        pendingReload->m_expectedMutationGeneration = mutation;
        pendingReload->m_expectedContentGeneration = content;
    }
    auto it = m_loadedAssets.find(guid);
    if (it == m_loadedAssets.end())
        return;

    m_loaders.at(it->second.type)->OnRelocated(it->second.payload, newPath);
    RefreshEntryBytes(it->second);
}

// =============================================================================
// Built-in materials (named, no GUID)
// =============================================================================

void AssetRegistry::RegisterBuiltinMaterial(const std::string &key, std::shared_ptr<InxMaterial> mat)
{
    if (!mat) {
        INXLOG_WARN("AssetRegistry::RegisterBuiltinMaterial: null material for key '", key, "'");
        return;
    }
    mat->BindDependencyGraph(m_assetDb->GetDependencyGraph());
    m_builtinMaterials[key] = std::move(mat);
}

std::shared_ptr<InxMaterial> AssetRegistry::GetBuiltinMaterial(const std::string &key) const
{
    auto it = m_builtinMaterials.find(key);
    return it != m_builtinMaterials.end() ? it->second : nullptr;
}

bool AssetRegistry::LoadBuiltinMaterialFromFile(const std::string &key, const std::string &matFilePath)
{
    if (matFilePath.empty())
        return false;

    std::ifstream file(ToFsPath(matFilePath));
    if (!file.is_open()) {
        INXLOG_WARN("AssetRegistry::LoadBuiltinMaterialFromFile: cannot open '", matFilePath, "'");
        return false;
    }
    std::string jsonStr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    auto material = std::make_shared<InxMaterial>();
    if (!material->Deserialize(jsonStr)) {
        INXLOG_ERROR("AssetRegistry::LoadBuiltinMaterialFromFile: deserialization failed for '", matFilePath, "'");
        return false;
    }

    material->SetFilePath(matFilePath);
    material->SetBuiltin(true);
    RegisterBuiltinMaterial(key, material);
    return true;
}

// =============================================================================
// Built-in material initialization
// =============================================================================

void AssetRegistry::InitializeBuiltinMaterials()
{
    auto registerBuiltin = [this](const std::string &key, std::shared_ptr<InxMaterial> mat) {
        if (mat) {
            mat->SetBuiltin(true);
            RegisterBuiltinMaterial(key, mat);
        }
    };

    registerBuiltin("DefaultLit", InxMaterial::CreateDefaultLit());
    registerBuiltin("DefaultUnlit", InxMaterial::CreateDefaultUnlit());
    registerBuiltin("DefaultLineMaterial", InxMaterial::CreateDefaultLineMaterial());
    registerBuiltin("ParticleSpriteMaterial", InxMaterial::CreateParticleSpriteMaterial());
    registerBuiltin("ParticleSixWaySmokeMaterial", InxMaterial::CreateParticleSixWaySmokeMaterial());
    registerBuiltin("GizmoMaterial", InxMaterial::CreateGizmoMaterial());
    registerBuiltin("GridMaterial", InxMaterial::CreateGridMaterial());
    registerBuiltin("ComponentGizmosMaterial", InxMaterial::CreateComponentGizmosMaterial());
    registerBuiltin("ComponentGizmoIconMaterial", InxMaterial::CreateComponentGizmoIconMaterial());
    registerBuiltin("ComponentGizmoCameraIconMaterial", InxMaterial::CreateComponentGizmoCameraIconMaterial());
    registerBuiltin("ComponentGizmoLightIconMaterial", InxMaterial::CreateComponentGizmoLightIconMaterial());
    registerBuiltin("ComponentGizmoParticleIconMaterial", InxMaterial::CreateComponentGizmoParticleIconMaterial());
    registerBuiltin("EditorToolsMaterial", InxMaterial::CreateEditorToolsMaterial());
    registerBuiltin("SkyboxProcedural", InxMaterial::CreateSkyboxProceduralMaterial());
    registerBuiltin("ErrorMaterial", InxMaterial::CreateErrorMaterial());
}

// =============================================================================
// GetAllMaterials — builtin + loaded assets + live runtime instances
// =============================================================================

void AssetRegistry::RegisterRuntimeMaterial(const std::shared_ptr<InxMaterial> &material)
{
    if (std::this_thread::get_id() != m_ownerThread)
        throw std::logic_error("Runtime material registration requires the AssetRegistry owner thread");
    if (!material)
        throw std::invalid_argument("Runtime material registration requires a material");
    material->BindDependencyGraph(m_assetDb->GetDependencyGraph());
    m_runtimeMaterials[material.get()] = material;
}

std::vector<std::shared_ptr<InxMaterial>> AssetRegistry::GetAllMaterials() const
{
    std::vector<std::shared_ptr<InxMaterial>> result;
    std::unordered_set<InxMaterial *> seen;

    // 1. Built-in materials
    for (const auto &[key, mat] : m_builtinMaterials) {
        if (mat && seen.find(mat.get()) == seen.end()) {
            result.push_back(mat);
            seen.insert(mat.get());
        }
    }

    // 2. User-loaded materials from disk (via AssetRegistry cache)
    for (const auto &[guid, entry] : m_loadedAssets) {
        if (entry.type == ResourceType::Material) {
            auto mat = entry.payload.Get<InxMaterial>();
            if (mat && seen.find(mat.get()) == seen.end()) {
                result.push_back(mat);
                seen.insert(mat.get());
            }
        }
    }

    for (auto it = m_runtimeMaterials.begin(); it != m_runtimeMaterials.end();) {
        auto material = it->second.lock();
        if (!material) {
            it = m_runtimeMaterials.erase(it);
            continue;
        }
        if (seen.insert(material.get()).second)
            result.push_back(std::move(material));
        ++it;
    }
    return result;
}

// =============================================================================
// Queries
// =============================================================================

bool AssetRegistry::IsLoaded(const std::string &guid) const
{
    return m_loadedAssets.count(guid) > 0;
}

ResourceType AssetRegistry::GetAssetType(const std::string &guid) const
{
    auto it = m_loadedAssets.find(guid);
    if (it != m_loadedAssets.end())
        return it->second.type;
    return ResourceType::DefaultBinary;
}

uint64_t AssetRegistry::GetAssetVersion(const std::string &guid) const
{
    const auto found = m_assetRuntimeVersions.find(guid);
    return found != m_assetRuntimeVersions.end() ? found->second : 0;
}

uint64_t AssetRegistry::GetAssetContentGeneration(const std::string &guid) const
{
    const auto found = m_assetContentGenerations.find(guid);
    return found != m_assetContentGenerations.end() ? found->second : 0;
}

void AssetRegistry::MarkMeshGpuViewResident(const std::string &guid, uint64_t runtimeVersion, MeshGeometryView view)
{
    if (std::this_thread::get_id() != m_ownerThread)
        throw std::logic_error("Mesh GPU residency publication requires the AssetRegistry owner thread");
    const auto asset = m_loadedAssets.find(guid);
    if (asset == m_loadedAssets.end() || asset->second.type != ResourceType::Mesh ||
        asset->second.version != runtimeVersion)
        return;
    const uint8_t bit = view == MeshGeometryView::MergedModelSpace ? uint8_t{1} : uint8_t{2};
    auto &residency = m_meshGpuViewResidency[guid];
    if (residency.runtimeVersion != runtimeVersion)
        residency = {runtimeVersion, 0};
    residency.mask = static_cast<uint8_t>(residency.mask | bit);
}

uint8_t AssetRegistry::GetMeshGpuViewResidencyMask(const std::string &guid, uint64_t runtimeVersion) const
{
    const auto found = m_meshGpuViewResidency.find(guid);
    return found != m_meshGpuViewResidency.end() && found->second.runtimeVersion == runtimeVersion ? found->second.mask
                                                                                                   : 0;
}

bool AssetRegistry::IsMeshGpuResidencyRequired(const std::string &guid, uint64_t runtimeVersion) const
{
    const auto found = m_loadedAssets.find(guid);
    if (found == m_loadedAssets.end() || found->second.type != ResourceType::Mesh ||
        found->second.version != runtimeVersion)
        return false;
    const auto mesh = found->second.payload.Get<InxMesh>();
    return mesh && !mesh->HasCpuGeometry();
}

size_t AssetRegistry::ReleaseMeshCpuGeometry(const std::string &guid, uint64_t runtimeVersion)
{
    if (std::this_thread::get_id() != m_ownerThread)
        throw std::logic_error("Mesh CPU geometry release requires the AssetRegistry owner thread");
    auto found = m_loadedAssets.find(guid);
    if (found == m_loadedAssets.end() || found->second.type != ResourceType::Mesh ||
        GetAssetVersion(guid) != runtimeVersion)
        return 0;
    auto mesh = found->second.payload.Get<InxMesh>();
    if (!mesh || mesh->IsCpuReadable())
        return 0;
    const uint8_t requiredMask = mesh->GetModelSourceGeometry() ? uint8_t{3} : uint8_t{1};
    if ((GetMeshGpuViewResidencyMask(guid, runtimeVersion) & requiredMask) != requiredMask)
        return 0;
    const size_t released = mesh->ReleaseCpuGeometry();
    if (released)
        RefreshEntryBytes(found->second);
    return released;
}

std::string AssetRegistry::GetAssetRuntimeTypeName(const std::string &guid) const
{
    const auto found = m_loadedAssets.find(guid);
    return found != m_loadedAssets.end() ? found->second.payload.GetTypeName() : std::string{};
}

std::vector<std::string> AssetRegistry::GetAllLoadedGuids() const
{
    std::vector<std::string> guids;
    guids.reserve(m_loadedAssets.size());
    for (const auto &[guid, entry] : m_loadedAssets)
        guids.push_back(guid);
    return guids;
}

AssetResidencyRecord AssetRegistry::GetAssetResidency(const std::string &guid) const
{
    const auto found = m_loadedAssets.find(guid);
    if (found == m_loadedAssets.end())
        throw std::invalid_argument("AssetRegistry has no loaded residency record for GUID: " + guid);
    const long useCount = found->second.payload.GetUseCount();
    const size_t externalReferences = useCount > 1 ? static_cast<size_t>(useCount - 1) : 0;
    return {guid,
            found->second.type,
            found->second.payload.GetTypeName(),
            found->second.version,
            found->second.cpuBytes,
            found->second.lastAccessSerial,
            found->second.explicitPinCount,
            externalReferences,
            found->second.explicitPinCount == 0 && externalReferences == 0};
}

std::vector<AssetResidencyRecord> AssetRegistry::GetAllAssetResidency() const
{
    std::vector<AssetResidencyRecord> records;
    records.reserve(m_loadedAssets.size());
    for (const auto &loaded : m_loadedAssets)
        records.push_back(GetAssetResidency(loaded.first));
    std::sort(records.begin(), records.end(),
              [](const auto &left, const auto &right) { return left.guid < right.guid; });
    return records;
}

std::vector<PublishedAssetVersion> AssetRegistry::GetAllPublishedAssetVersions() const
{
    std::vector<PublishedAssetVersion> versions;
    versions.reserve(m_assetRuntimeVersions.size());
    for (const auto &[guid, runtimeVersion] : m_assetRuntimeVersions) {
        const auto type = m_assetRuntimeTypes.find(guid);
        if (type == m_assetRuntimeTypes.end())
            throw std::logic_error("Asset runtime version has no resource type: " + guid);
        versions.push_back({guid, type->second, runtimeVersion});
    }
    std::sort(versions.begin(), versions.end(),
              [](const auto &left, const auto &right) { return left.guid < right.guid; });
    return versions;
}

void AssetRegistry::SetCpuBudgetBytes(size_t bytes)
{
    if (bytes == 0)
        throw std::invalid_argument("AssetRegistry CPU budget must be greater than zero");
    m_cpuBudgetBytes = bytes;
    (void)TrimCpuBudget();
}

size_t AssetRegistry::TrimCpuBudget()
{
    if (m_cpuStorageBytes > m_cpuBudgetBytes)
        CollectCpuStorage();
    return TrimCpuStorageTo(m_cpuBudgetBytes);
}

bool AssetRegistry::CanEvict(const AssetEntry &entry) noexcept
{
    return entry.explicitPinCount == 0 && entry.payload.GetUseCount() == 1;
}

bool AssetRegistry::FitCpuStorage(uint64_t target, bool reclaim)
{
    if (m_cpuStorageBytes <= target)
        return true;
    CollectCpuStorage();
    if (m_cpuStorageBytes <= target)
        return true;

    // Count a shared block only when evicting these entries would drop its
    // last strong owner. A live upload snapshot cannot fund new work.
    uint64_t needed = m_cpuStorageBytes - target;
    std::map<std::weak_ptr<const void>, long, std::owner_less<>> sharedOwners;
    for (const auto &entry : m_loadedAssets) {
        if (!CanEvict(entry.second))
            continue;
        for (const auto &owner : CpuStorageOwners(entry.second)) {
            const auto count = owner.use_count();
            if (!count || (count > 1 && ++sharedOwners[owner] != count))
                continue;
            const auto tracked = m_cpuStorage.find(owner);
            if (tracked == m_cpuStorage.end())
                continue;
            if (tracked->second >= needed) {
                if (!reclaim)
                    return true;
                if (TrimCpuStorageTo(target))
                    m_frameWork->ReportProgress();
                return m_cpuStorageBytes <= target;
            }
            needed -= tracked->second;
        }
    }
    return false;
}

size_t AssetRegistry::TrimCpuStorageTo(uint64_t target)
{
    size_t evicted = 0;
    while (m_cpuStorageBytes > target) {
        auto candidate = m_loadedAssets.end();
        for (auto entry = m_loadedAssets.begin(); entry != m_loadedAssets.end(); ++entry) {
            if (!CanEvict(entry->second))
                continue;
            if (candidate == m_loadedAssets.end() ||
                entry->second.lastAccessSerial < candidate->second.lastAccessSerial)
                candidate = entry;
        }
        if (candidate == m_loadedAssets.end())
            break;
        RemoveEntry(candidate);
        ++evicted;
        ++m_cpuEvictionCount;
    }
    return evicted;
}

void AssetRegistry::PinAsset(const std::string &guid)
{
    const auto found = m_loadedAssets.find(guid);
    if (found == m_loadedAssets.end())
        throw std::invalid_argument("cannot pin an unloaded asset: " + guid);
    if (found->second.explicitPinCount == std::numeric_limits<uint32_t>::max())
        throw std::overflow_error("asset pin count overflow: " + guid);
    ++found->second.explicitPinCount;
    found->second.lastAccessSerial = ++m_accessSerial;
}

void AssetRegistry::UnpinAsset(const std::string &guid)
{
    const auto found = m_loadedAssets.find(guid);
    if (found == m_loadedAssets.end())
        throw std::invalid_argument("cannot unpin an unloaded asset: " + guid);
    if (found->second.explicitPinCount == 0)
        throw std::logic_error("asset pin count is already zero: " + guid);
    --found->second.explicitPinCount;
}

} // namespace infernux
