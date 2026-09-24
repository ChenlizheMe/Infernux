#pragma once

#include <core/types/InxFwdType.h>
#include <function/resources/AssetRuntimeApi.h>

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace infernux
{

enum class AssetEvent
{
    Deleted,
    Modified,
    Moved,
    /// In-memory publication: notify runtime users, not source asset imports.
    RuntimeModified,
};

using AssetEventCallback =
    std::function<void(const std::string &dependentGuid, const std::string &dependencyGuid, AssetEvent event)>;

class AssetDependencyGraph;

class AssetDependencySnapshot final
{
  public:
    [[nodiscard]] uint64_t GetGeneration() const noexcept
    {
        return m_generation;
    }
    [[nodiscard]] size_t GetEdgeCount() const noexcept
    {
        return m_edgeCount;
    }
    [[nodiscard]] size_t GetNodeCount() const noexcept
    {
        return m_nodes.size();
    }

  private:
    friend class AssetDependencyGraph;
    uint64_t m_generation = 0;
    size_t m_edgeCount = 0;
    std::unordered_map<std::string, std::unordered_set<std::string>> m_dependencies;
    std::unordered_map<std::string, std::unordered_set<std::string>> m_dependents;
    std::unordered_set<std::string> m_nodes;
};

/**
 * Asset-to-asset edges and runtime-object usage have different lifetimes.
 * Asset edges are published as immutable generations; runtime usage remains a
 * small mutable overlay. Queries and lifecycle notifications observe their
 * union without making refresh publication copy runtime state.
 */
class AssetDependencyGraph
{
  public:
    INFERNUX_ASSET_RUNTIME_API static AssetDependencyGraph &Instance();

    AssetDependencyGraph(const AssetDependencyGraph &) = delete;
    AssetDependencyGraph &operator=(const AssetDependencyGraph &) = delete;

    INFERNUX_ASSET_RUNTIME_API void AddAssetDependency(const std::string &assetGuid,
                                                       const std::string &dependencyGuid);
    INFERNUX_ASSET_RUNTIME_API void RemoveAssetDependency(const std::string &assetGuid,
                                                          const std::string &dependencyGuid);
    INFERNUX_ASSET_RUNTIME_API void ClearAssetDependenciesOf(const std::string &assetGuid);
    INFERNUX_ASSET_RUNTIME_API void
    SetAssetDependencies(const std::string &assetGuid, const std::unordered_set<std::string> &dependencyGuids);

    [[nodiscard]] INFERNUX_ASSET_RUNTIME_API static std::shared_ptr<const AssetDependencySnapshot>
    BuildAssetSnapshot(const std::unordered_map<std::string, std::vector<std::string>> &dependenciesByAsset,
                       uint64_t generation);
    INFERNUX_ASSET_RUNTIME_API void InstallAssetSnapshot(std::shared_ptr<const AssetDependencySnapshot> snapshot);
    [[nodiscard]] INFERNUX_ASSET_RUNTIME_API std::shared_ptr<const AssetDependencySnapshot>
    GetAssetSnapshot() const;
    [[nodiscard]] INFERNUX_ASSET_RUNTIME_API uint64_t GetAssetGeneration() const;

    INFERNUX_ASSET_RUNTIME_API void AddRuntimeDependency(const std::string &objectGuid, const std::string &assetGuid);
    INFERNUX_ASSET_RUNTIME_API void RemoveRuntimeDependency(const std::string &objectGuid,
                                                            const std::string &assetGuid);
    INFERNUX_ASSET_RUNTIME_API void ClearRuntimeDependenciesOf(const std::string &objectGuid);
    /// Preserve all asset subscriptions when a staged component receives its published ID.
    INFERNUX_ASSET_RUNTIME_API void RekeyRuntimeDependencies(const std::string &oldOwner,
                                                             const std::string &newOwner);

    /// Remove dependencies owned by an asset while retaining references to it.
    /// Incoming edges represent serialized missing references and must survive
    /// deletion so a later reimport/Undo can notify and reconnect dependents.
    INFERNUX_ASSET_RUNTIME_API void RemoveAsset(const std::string &guid);

    [[nodiscard]] INFERNUX_ASSET_RUNTIME_API std::unordered_set<std::string>
    GetDependencies(const std::string &guid) const;
    [[nodiscard]] INFERNUX_ASSET_RUNTIME_API std::unordered_map<std::string, std::unordered_set<std::string>>
    GetDependenciesBatch(const std::vector<std::string> &guids) const;
    [[nodiscard]] INFERNUX_ASSET_RUNTIME_API std::unordered_set<std::string>
    GetDependents(const std::string &guid) const;
    [[nodiscard]] INFERNUX_ASSET_RUNTIME_API bool HasDependency(const std::string &userGuid,
                                                               const std::string &dependencyGuid) const;

    INFERNUX_ASSET_RUNTIME_API void RegisterCallback(ResourceType type, AssetEventCallback callback);
    INFERNUX_ASSET_RUNTIME_API void NotifyEvent(const std::string &guid, ResourceType type, AssetEvent event);

    [[nodiscard]] INFERNUX_ASSET_RUNTIME_API size_t GetEdgeCount() const;
    [[nodiscard]] INFERNUX_ASSET_RUNTIME_API size_t GetNodeCount() const;
    INFERNUX_ASSET_RUNTIME_API void Clear();

  private:
    AssetDependencyGraph();
    ~AssetDependencyGraph() = default;

    void PublishAssetMutation(const std::function<void(AssetDependencySnapshot &)> &mutation);
    static void AddEdge(AssetDependencySnapshot &snapshot, const std::string &userGuid,
                        const std::string &dependencyGuid);
    static void RemoveEdge(AssetDependencySnapshot &snapshot, const std::string &userGuid,
                           const std::string &dependencyGuid);
    static void ClearEdgesOf(AssetDependencySnapshot &snapshot, const std::string &userGuid);
    static void RebuildStatistics(AssetDependencySnapshot &snapshot);

    std::shared_ptr<const AssetDependencySnapshot> m_assetSnapshot;
    mutable std::mutex m_assetWriteMutex;

    std::unordered_map<std::string, std::unordered_set<std::string>> m_runtimeDependencies;
    std::unordered_map<std::string, std::unordered_set<std::string>> m_runtimeDependents;
    std::unordered_map<ResourceType, std::vector<AssetEventCallback>> m_callbacks;
    mutable std::mutex m_runtimeMutex;
};

} // namespace infernux
