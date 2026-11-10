#pragma once

#include <core/threading/JobSystem.h>
#include <core/types/InxFwdType.h>
#include <function/resources/AssetDatabase/AssetIndex.h>
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetImporter/ImporterRegistry.h>
#include <function/resources/AssetRegistry/IAssetLoader.h>
#include <function/resources/InxResource/InxResourceMeta.h>
#include <function/scene/NativeLifetime.h>
#include <platform/filesystem/DocumentTransaction.h>
#include <platform/filesystem/InxPath.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace infernux
{

class InxRenderer;
class AssetRegistry;
class AssetLoadTicket;
class FrameWorkOwner;
struct FrameWorkSource;
enum class FrameWorkResource : uint8_t;

enum class AssetMutationErrorCode : uint8_t
{
    None,
    InvalidPath,
    NotFound,
    UnsupportedType,
    ImportFailed,
    RuntimeApplyFailed,
};

struct AssetMutationResult
{
    bool succeeded = false;
    bool databaseCommitted = false;
    bool changed = false;
    std::string operation;
    std::string guid;
    std::string path;
    std::string previousPath;
    ResourceType resourceType = ResourceType::DefaultText;
    AssetMutationErrorCode errorCode = AssetMutationErrorCode::None;
    std::string error;
    uint64_t queryGeneration = 0;
    std::string authoredMetadataGuid;
    std::optional<AtomicFileState> observedMetadata;

    [[nodiscard]] explicit operator bool() const noexcept
    {
        return succeeded;
    }
};

struct AssetCatalogEntry
{
    std::string guid;
    std::string path;
    std::string name;
    std::string sortKey;
    ResourceType resourceType = ResourceType::DefaultText;
    AssetFileFingerprint source;
    std::string importError;
};

class AssetCatalogSnapshot final
{
  public:
    [[nodiscard]] uint64_t GetGeneration() const noexcept
    {
        return m_generation;
    }
    [[nodiscard]] const std::vector<AssetCatalogEntry> &GetDirectory(const std::string &normalizedDirectory) const;
    [[nodiscard]] const std::unordered_map<std::string, std::vector<AssetCatalogEntry>> &GetDirectories() const noexcept
    {
        return m_directories;
    }

  private:
    friend class AssetDatabase;
    uint64_t m_generation = 0;
    std::unordered_map<std::string, std::vector<AssetCatalogEntry>> m_directories;
};

/**
 * @brief Central asset database for the project.
 *
 * Responsibilities:
 * - Import assets and generate .meta files
 * - Maintain GUID <-> path mappings
 * - Provide Asset CRUD operations for editor and file watcher
 * - Own the ImporterRegistry and drive dependency scanning on import
 * - Dispatch AssetEvent notifications via AssetDependencyGraph
 * - Manage resource loaders, metadata cache and
 * compiled resource cache
 *
 * Durable-reference invariant:
 * - Persistent asset dependencies are identified by GUID
 * only.
 * - Paths and path_hint values are mutable authoring/display metadata.
 * - Importers, loaders, runtime code
 * and dependency publication must never
 *   call GetGuidFromPath() to repair or infer a stored dependency.
 * -
 * Path-to-GUID conversion is permitted only at an explicit editor authoring
 *   boundary, before the reference is
 * persisted.
 */
class AssetDatabase : public NativeLifetimeOwner
{
  public:
    AssetDatabase();
    ~AssetDatabase();

    [[nodiscard]] const std::shared_ptr<AssetDependencyGraph> &GetDependencyGraph() const noexcept
    {
        return m_dependencyGraph;
    }

    /// @brief Initialize database with project root path.
    /// Also creates and registers all built-in importers.
    void Initialize(const std::string &projectRoot);

    /// Configure before startup, or on the owner between import transactions.
    /// Empty paths disable .blend authoring; Players never need this tool.
    void ConfigureBlenderImport(const std::string &executable, const std::string &exportScript);

    /// @brief Initialize the immutable Player-side database without editor importers.
    ///
    /// Platform Players consume build-authored RuntimeAssetRecords and never
    /// scan or mutate source assets. Keeping this entry point separate avoids
    /// linking the authoring/import pipeline into constrained platform hosts.
    void InitializeRuntime(const std::string &projectRoot);

    /// @brief Refresh all assets by scanning the Assets folder
    void Refresh();

    /// @brief Restore the last published editor catalog without touching source files.
    ///
    /// This is the interactive-startup path. A caller must follow it with an
    /// asynchronous BeginRefresh/TryCommitRefresh transaction so external
    /// changes made while the editor was closed are still reconciled.
    [[nodiscard]] bool RestoreCachedCatalog();

    /// @brief Schedule filesystem enumeration and fingerprint collection.
    /// The worker never mutates AssetDatabase, metadata, or dependency state.
    void BeginRefresh();
    [[nodiscard]] FrameWorkSource GetCatalogWorkSource(FrameWorkOwner &owner);

    /// @brief Advance scan, worker import, and owner commit without blocking.
    /// @return false while either worker phase is incomplete.
    bool TryCommitRefresh();

    /// @brief Complete the currently pending refresh on the mutation thread.
    ///
    /// Waits on each native worker phase and advances the same commit state
    /// machine used by TryCommitRefresh. A pending refresh is required.
    void CompletePendingRefresh();

    [[nodiscard]] bool IsRefreshPending() const;
    [[nodiscard]] bool IsOwnerThread() const;

    /// @brief Persist the current derived index if runtime CRUD made it dirty.
    /// @brief Persist a dirty derived AssetIndex. With waitForPendingScan set
    /// to false (engine shutdown), a still-running background scan skips the
    /// flush instead of blocking teardown; the next startup revalidates the
    /// cache regardless.
    void FlushDerivedIndex(bool waitForPendingScan = true);

    /// @brief Add an extra directory to scan during Refresh (e.g. Library/Resources).
    void AddScanRoot(const std::string &path);

    /// @brief Add an immutable packaged-resource root. Missing metadata is kept in memory.
    void AddReadOnlyScanRoot(const std::string &path);

    /// @brief Overlay the cooked Player asset identities after the immutable
    /// project payload has been scanned. The catalog replaces editor .meta
    /// sidecars inside Player builds and is never persisted back to disk.
    void InstallRuntimeAssetCatalog(const std::string &catalogPath, bool trustedPackage = false);

    /// Prepare new cooked identities on the asset worker. The Player publishes
    /// the prepared index at the same boundary as its component and scene data.
    void BeginRuntimeAssetCatalog(const std::string &packageRoot, const std::string &contentRoot,
                                  const std::vector<std::string> &allowedRoots);
    [[nodiscard]] bool IsRuntimeAssetCatalogReady();
    [[nodiscard]] const nlohmann::json *GetPreparedRuntimeCatalogDocuments();
    void PublishRuntimeAssetCatalog();
    void DiscardRuntimeAssetCatalog();

    /// @brief Import an asset and create/update its meta.
    /// Runs the appropriate AssetImporter to scan dependencies.
    [[nodiscard]] AssetMutationResult ImportAsset(const std::string &path, const PathWriteScope &writeScope = {});

    /// @brief Re-run metadata/dependency import for an already registered asset.
    /// Preserves the existing GUID and returns false for missing or unregistered paths.
    [[nodiscard]] AssetMutationResult ReimportAsset(const std::string &path, const nlohmann::json &settings = nullptr,
                                                    const PathWriteScope &writeScope = {});
    using WatcherImportReady = std::function<bool(const AssetMutationResult &, bool beforeWrite)>;
    using WatcherImportFinished =
        std::function<void(std::string_view, const AssetMutationResult &, uint64_t, std::optional<float>)>;
    /// Prepare one external revision on the catalog worker and commit at the owner safe point.
    bool BeginWatcherImport(const std::string &path, WatcherImportReady ready, WatcherImportFinished finished,
                            bool metadataOnly = false);
    void CompletePendingWatcherImport();
    [[nodiscard]] bool IsWatcherImportPending() const noexcept;
    using ModelReimportPublished = std::function<void(const AssetMutationResult &, uint64_t, std::optional<float>)>;
    /// Explicit model Apply: cook, persist, then publish from the owner thread.
    void BeginModelReimport(const std::string &path, const nlohmann::json &settings,
                            ModelReimportPublished onPublished = {});
    [[nodiscard]] std::optional<AssetMutationResult> TryCommitModelReimport();
    std::optional<AssetMutationResult> CompletePendingModelReimport();
    bool DiscardModelReimport();
    [[nodiscard]] bool IsModelReimportPending() const noexcept;
    [[nodiscard]] bool IsModelReimportPublishing() const noexcept;
    [[nodiscard]] bool IsDeliveringModelReimport() const noexcept;

    /// @brief Delete asset and meta.
    /// Notifies dependents via AssetDependencyGraph::NotifyEvent(Deleted).
    [[nodiscard]] AssetMutationResult DeleteAsset(const std::string &path, const PathWriteScope &writeScope = {});
    /// Exact source/sidecar/derived paths covered by a delete, including model
    /// subassets. Directory traversal uses the published catalog, not disk IO.
    [[nodiscard]] std::vector<std::string> GetDeletionWritePaths(const std::vector<std::string> &roots) const;

    /// @brief Publish an already completed filesystem relocation, preserving GUID.
    /// The caller moves the asset first; a missing destination is rejected
    /// before metadata or directory mappings change.
    /// Notifies dependents via AssetDependencyGraph::NotifyEvent(Moved).
    [[nodiscard]] AssetMutationResult MoveAsset(const std::string &oldPath, const std::string &newPath,
                                                const PathWriteScope &writeScope = {});

    /// Commit completed filesystem relocations with one catalog publication.
    /// All destinations must exist before any metadata is changed.
    [[nodiscard]] std::vector<AssetMutationResult>
    MoveAssetsBatch(const std::vector<std::pair<std::string, std::string>> &moves,
                    const PathWriteScope &writeScope = {});

    /// @brief Check if a GUID exists
    [[nodiscard]] bool ContainsGuid(const std::string &guid) const;

    /// @brief Check if a path exists in database
    [[nodiscard]] bool ContainsPath(const std::string &path) const;

    /// @brief Get GUID by path (empty if not found).
    ///
    /// This is an authoring/discovery API for imports, file watchers, pickers
    /// and drag/drop. It is forbidden as a fallback for resolving a persistent
    /// asset reference or publishing a dependency edge; those operations must
    /// already carry the authoritative GUID.
    [[nodiscard]] std::string GetGuidFromPath(const std::string &path) const;

    /// @brief Query a canonical catalog path without consulting the filesystem.
    /// Unlike the authoring alias resolver, this reads only the published index.
    [[nodiscard]] std::string GetGuidFromCatalogPath(const std::string &path) const;

    /// @brief Resolve the current mutable location of an authoritative GUID.
    /// Returns empty when the referenced asset is missing.
    [[nodiscard]] std::string GetPathFromGuid(const std::string &guid) const;

    /// @brief Get immutable metadata by GUID. The shared result remains valid
    /// across concurrent index publication and asset deletion.
    [[nodiscard]] std::shared_ptr<const InxResourceMeta> GetMetaByGuid(const std::string &guid) const;

    /// @brief Get immutable metadata by path.
    [[nodiscard]] std::shared_ptr<const InxResourceMeta> GetMetaByPath(const std::string &path) const;

    /// @brief Get all GUIDs
    [[nodiscard]] std::vector<std::string> GetAllGuids() const;

    /// @brief Get all asset paths from one published database generation.
    [[nodiscard]] std::vector<std::string> GetAllAssetPaths() const;

    /// @brief Acquire one immutable path/type/fingerprint catalog generation.
    [[nodiscard]] std::shared_ptr<const AssetCatalogSnapshot> GetCatalogSnapshot() const;
    /// Canonical catalog directory; no filesystem existence or alias resolution.
    [[nodiscard]] std::vector<AssetCatalogEntry> GetDirectoryCatalog(const std::string &directory) const;
    [[nodiscard]] std::string GetImportError(const std::string &guid) const;
    [[nodiscard]] std::vector<AssetCatalogEntry> GetImportFailures() const;
    void ReportAssetLoadFailure(const std::string &guid, const std::string &error);

    /// @brief Published query snapshot generation and asset count.
    [[nodiscard]] uint64_t GetQueryGeneration() const;
    [[nodiscard]] size_t GetAssetCount() const;
    [[nodiscard]] size_t GetLastRefreshReusedCount() const noexcept
    {
        return m_lastRefreshReusedCount;
    }
    [[nodiscard]] size_t GetLastRefreshImportedCount() const noexcept
    {
        return m_lastRefreshImportedCount;
    }
    [[nodiscard]] const std::vector<std::string> &GetLastRefreshImportedPaths() const noexcept
    {
        return m_lastRefreshImportedPaths;
    }
    [[nodiscard]] size_t GetLastRefreshImporterTaskCount() const noexcept
    {
        return m_lastRefreshImporterTaskCount;
    }
    [[nodiscard]] size_t GetLastRefreshMetadataTaskCount() const noexcept
    {
        return m_lastRefreshMetadataTaskCount;
    }
    [[nodiscard]] size_t GetLastRefreshWorkerMetadataCount() const noexcept
    {
        return m_lastRefreshWorkerMetadataCount;
    }
    [[nodiscard]] size_t GetLastRefreshWorkerImporterCount() const noexcept
    {
        return m_lastRefreshWorkerImporterCount;
    }
    [[nodiscard]] size_t GetLastRefreshScannedCount() const noexcept
    {
        return m_lastRefreshScannedCount;
    }
    [[nodiscard]] double GetLastRefreshScanMilliseconds() const noexcept
    {
        return m_lastRefreshScanMilliseconds;
    }
    [[nodiscard]] double GetLastRefreshCommitMilliseconds() const noexcept
    {
        return m_lastRefreshCommitMilliseconds;
    }
    [[nodiscard]] double GetLastModelReimportWorkerMilliseconds() const noexcept
    {
        return m_lastModelReimportWorkerMilliseconds;
    }
    [[nodiscard]] double GetLastModelReimportPrepareMilliseconds() const noexcept
    {
        return m_lastModelReimportPrepareMilliseconds;
    }
    [[nodiscard]] double GetLastModelReimportPersistenceMilliseconds() const noexcept
    {
        return m_lastModelReimportPersistenceMilliseconds;
    }
    [[nodiscard]] double GetLastModelReimportLivePublicationMilliseconds() const noexcept
    {
        return m_lastModelReimportLivePublicationMilliseconds;
    }
    [[nodiscard]] double GetLastRefreshPrepareMilliseconds() const noexcept
    {
        return m_lastRefreshPrepareMilliseconds;
    }
    [[nodiscard]] double GetLastRefreshFinalizeMilliseconds() const noexcept
    {
        return m_lastRefreshFinalizeMilliseconds;
    }
    [[nodiscard]] double GetLastRefreshOwnerMergeMaxSliceMilliseconds() const noexcept
    {
        return m_lastRefreshOwnerMergeMaxSliceMilliseconds;
    }
    [[nodiscard]] size_t GetLastRefreshOwnerMergeSliceCount() const noexcept
    {
        return m_lastRefreshOwnerMergeSliceCount;
    }
    [[nodiscard]] double GetLastRefreshMetadataWriteMilliseconds() const noexcept
    {
        return m_lastRefreshMetadataWriteMilliseconds;
    }
    [[nodiscard]] uint64_t GetLastRefreshStagedBytes() const noexcept
    {
        return m_lastRefreshStagedBytes;
    }
    [[nodiscard]] double GetLastRefreshStagingMilliseconds() const noexcept
    {
        return m_lastRefreshStagingMilliseconds;
    }
    [[nodiscard]] uint64_t GetLastRefreshJournalBytes() const noexcept
    {
        return m_lastRefreshJournalBytes;
    }
    [[nodiscard]] double GetLastRefreshJournalSerializeMilliseconds() const noexcept
    {
        return m_lastRefreshJournalSerializeMilliseconds;
    }
    [[nodiscard]] double GetLastRefreshJournalWriteMilliseconds() const noexcept
    {
        return m_lastRefreshJournalWriteMilliseconds;
    }
    [[nodiscard]] double GetLastRefreshJournalApplyMilliseconds() const noexcept
    {
        return m_lastRefreshJournalApplyMilliseconds;
    }
    [[nodiscard]] bool WasLastRefreshScanOnWorker() const noexcept
    {
        return m_lastRefreshScanOnWorker;
    }
    [[nodiscard]] double GetLastRefreshRestoreMilliseconds() const noexcept
    {
        return m_lastRefreshRestoreMilliseconds;
    }
    [[nodiscard]] double GetLastRefreshImportMilliseconds() const noexcept
    {
        return m_lastRefreshImportMilliseconds;
    }
    [[nodiscard]] double GetLastRefreshIndexBuildMilliseconds() const noexcept
    {
        return m_lastRefreshIndexBuildMilliseconds;
    }
    [[nodiscard]] double GetLastRefreshIndexSaveMilliseconds() const noexcept
    {
        return m_lastRefreshIndexSaveMilliseconds;
    }
    [[nodiscard]] bool WasLastRefreshIndexBuildOnWorker() const noexcept
    {
        return m_lastRefreshIndexBuildOnWorker;
    }
    [[nodiscard]] double GetLastRefreshQueryBuildMilliseconds() const noexcept
    {
        return m_lastRefreshQueryBuildMilliseconds;
    }
    [[nodiscard]] bool WasLastRefreshQueryBuildOnWorker() const noexcept
    {
        return m_lastRefreshQueryBuildOnWorker;
    }
    [[nodiscard]] double GetLastRefreshDependencyBuildMilliseconds() const noexcept
    {
        return m_lastRefreshDependencyBuildMilliseconds;
    }
    [[nodiscard]] bool WasLastRefreshDependencyBuildOnWorker() const noexcept
    {
        return m_lastRefreshDependencyBuildOnWorker;
    }
    [[nodiscard]] double GetLastRefreshPublishMilliseconds() const noexcept
    {
        return m_lastRefreshPublishMilliseconds;
    }
    [[nodiscard]] const std::string &GetAssetIndexPath() const noexcept
    {
        return m_assetIndexPath;
    }

    /// @brief Check if path is within Assets folder
    [[nodiscard]] bool IsAssetPath(const std::string &path) const;

    /// @brief Get project root
    [[nodiscard]] const std::string &GetProjectRoot() const
    {
        return m_projectRoot;
    }

    /// @brief Get assets root
    [[nodiscard]] const std::string &GetAssetsRoot() const
    {
        return m_assetsRoot;
    }

    /// @brief Resolve the deterministic project-local runtime CPU artifact path.
    /// Returns empty for resource types that do not produce a runtime artifact.
    [[nodiscard]] std::string GetRuntimeArtifactPath(const std::string &guid, ResourceType type) const;
    [[nodiscard]] std::string GetSkinnedMeshArtifactPath(const std::string &guid) const;

    [[nodiscard]] bool IsStagingRuntimeArtifact(const std::string &guid, ResourceType type) const noexcept;
    void CompleteRuntimeArtifactPublication();

    /// @brief Access the importer registry
    [[nodiscard]] ImporterRegistry &GetImporterRegistry()
    {
        return m_importerRegistry;
    }

    // ========================================================================
    // Resource management
    // ========================================================================

    /// @brief Read exact source bytes without platform newline translation.
    bool ReadFile(const std::string &filePath, std::vector<char> &content) const;

    /// @brief Find shader file path by shader_id
    [[nodiscard]] std::string FindShaderPathById(const std::string &shaderId, const std::string &shaderType) const;

    /// @brief Get resource type by file extension
    [[nodiscard]] ResourceType GetResourcesType(const std::string &extensionName) const;

    /// @brief Get resource type from a file path
    [[nodiscard]] ResourceType GetResourceTypeForPath(const std::string &filePath) const;

  private:
    friend class AssetRegistry;

    struct AssetScanRoot
    {
        std::string path;
        bool readOnly = false;
    };

    struct AssetScanRequest
    {
        std::vector<AssetScanRoot> roots;
        std::string assetIndexPath;
        std::string normalizedProjectRoot;
        uint64_t expectedQueryGeneration = 0;
    };

    struct AssetScanFile
    {
        std::string path;
        std::string normalizedPath;
        std::string identityKey;
        AssetFileFingerprint source;
        AssetFileFingerprint meta;
        bool readOnly = false;
    };

    struct AssetScanArtifact
    {
        AssetIndex index;
        std::vector<AssetScanFile> files;
        std::vector<std::string> orphanedTempFiles;
        std::vector<std::string> diagnostics;
        std::thread::id producerThread;
        double scanMilliseconds = 0.0;
    };

    struct PendingAssetScan
    {
        explicit PendingAssetScan(uint64_t generation) : expectedQueryGeneration(generation)
        {
        }

        std::optional<AssetScanArtifact> artifact;
        std::exception_ptr failure;
        uint64_t expectedQueryGeneration = 0;
    };

    struct CachedFileState
    {
        AssetFileFingerprint source;
        AssetFileFingerprint meta;
        bool readOnly = false;
    };

    struct ImportResultState
    {
        bool succeeded = true;
        std::string error;
    };

    struct WorkingSet
    {
        std::unordered_map<std::string, std::string> guidToPath;
        std::unordered_map<std::string, std::string> pathToGuid;
        std::unordered_map<std::string, std::shared_ptr<InxResourceMeta>> metas;
        std::unordered_map<std::string, CachedFileState> fileStates;
        std::unordered_map<std::string, ImportResultState> importResults;
        AssetIndex assetIndex;
        bool assetIndexDirty = false;
    };

    struct PendingAsset
    {
        std::string guid;
        std::string path;
        std::string normalizedPath;
        AssetFileFingerprint source;
        AssetFileFingerprint meta;
        bool readOnly = false;
        bool persistMetadata = true;
        std::optional<AtomicFileState> expectedMetadata;
    };

    struct WorkerImport
    {
        size_t assetIndex = 0;
        AssetImporter *importer = nullptr;
        ImportRequest request;
        AssetFileFingerprint expectedSource;
        std::optional<ImportArtifact> artifact;
        std::shared_ptr<FrameCpuStorageClaim> storage;
        std::optional<DocumentTransactionEntry> stagedTexture;
        std::shared_ptr<DocumentWriteTicket> stageWrite;
        std::exception_ptr stageFailure;
        std::string error;
        std::thread::id producerThread;
        std::optional<size_t> skeletonDependency;
        bool scheduled = false;
        bool complete = false;
    };

    struct MetadataPlanCursor;
    struct MetadataReadCursor;

    struct WorkerMetadataPrepare
    {
        PathWriteScope writeScope;
        AssetScanFile file;
        ResourceType resourceType = ResourceType::DefaultText;
        const IAssetLoader *loader = nullptr;
        std::string fallbackGuid;
        std::string createdGuid;
        std::string authoredGuid;
        bool inheritedGuid = false;
        std::shared_ptr<InxResourceMeta> metadata;
        std::optional<AtomicFileState> expectedMetadata;
        std::shared_ptr<MetadataPlanCursor> sidecarPlan;
        std::shared_ptr<MetadataReadCursor> sidecarRead;
        std::unique_ptr<InxResourceMeta> decodedSidecar;
        std::optional<FileInput::Revision> sidecarRevision;
        uint64_t sidecarWorkingBytes = 0;
        std::shared_ptr<const InxResourceMeta> publishedSidecar;
        const std::string *sidecarProjectRoot = nullptr;
        uint64_t sidecarComparisonBytes = 0;
        bool sidecarPlanned = false;
        bool adoptingSidecar = false;
        bool skipSidecarChange = false;
        std::optional<FileInput::Revision> textureRevision;
        uint64_t textureWorkingBytes = 0;
        std::string texturePlanningError;
        std::string error;
        std::thread::id producerThread;

        [[nodiscard]] uint64_t SidecarWorkspace() const noexcept
        {
            return sidecarWorkingBytes + sidecarComparisonBytes;
        }
    };

    struct TextureSourceSnapshot;
    struct PendingWatcherImport;

    struct PendingModelReimport
    {
        enum class Phase
        {
            MetadataPlan,
            TexturePrepare,
            Cook,
            Write,
            Publish,
            Complete
        };

        PathWriteScope writeScope;
        WorkerImport worker;
        WorkerMetadataPrepare metadata;
        std::shared_ptr<TextureSourceSnapshot> textureSource;
        std::string textureGuid;
        nlohmann::json settings;
        AssetMutationResult result;
        ModelReimportPublished onPublished;
        std::shared_ptr<AssetLoadTicket> repairTicket;
        std::unordered_set<std::string> artifactGuids;
        CachedFileState committedFiles;
        FileInput::Revision sourceRevision;
        std::optional<FileInput::Revision> metadataRevision;
        Phase phase = Phase::Cook;
        bool abandoned = false;
        enum : uint8_t
        {
            WorkerFinished = 1,
            Discarded = 2
        };
        std::atomic<uint8_t> lifecycle{0};
        bool IsDiscarded() const
        {
            return (lifecycle.load(std::memory_order_acquire) & Discarded) != 0;
        }
        void Discard()
        {
            const auto previous = lifecycle.fetch_or(Discarded, std::memory_order_acq_rel);
            if ((previous & WorkerFinished) && !(previous & Discarded))
                writeScope.Close();
        }
        void FinishWorker()
        {
            if (lifecycle.fetch_or(WorkerFinished, std::memory_order_acq_rel) & Discarded)
                writeScope.Close();
        }
        double workerMilliseconds = 0.0;
        double prepareMilliseconds = 0.0;
        double persistenceMilliseconds = 0.0;
    };

    struct FinishedModelApply
    {
        AssetMutationResult result;
        bool discarded = false;
        bool abandoned = false;
    };

    struct WorkerMetadataOutput
    {
        size_t assetIndex;
        std::shared_ptr<const InxResourceMeta> metadata;
        uint64_t workspace = 0;
        size_t bytes = 0;
        uint64_t fingerprint = 0;
        std::optional<DocumentTransactionEntry> staged;
        std::shared_ptr<DocumentWriteTicket> write;
        std::exception_ptr failure;
    };

    enum class CatalogIoStage : uint8_t
    {
        Scan,
        MetadataPlan,
        MetadataCapture,
        MetadataDecode,
        Metadata,
        Sidecar,
        SidecarMeasure,
        SidecarStage,
        Cook,
        Persist,
        Index,
        Apply,
        RuntimeCatalog,
        Count
    };

    struct CatalogWork
    {
        using StoragePlan = std::function<uint64_t(uint32_t)>;
        using Factory = std::function<JobSystem::JobFn(uint32_t, std::shared_ptr<FrameCpuStorageClaim>)>;
        struct Receipt
        {
            JobHandle job;
            std::shared_ptr<uint64_t> elapsed;
            uint64_t frame = 0;
            uint64_t grant = 0;
        };
        [[nodiscard]] bool Complete() const noexcept
        {
            return submitted == count && active.empty();
        }
        void DropUnsubmitted() noexcept
        {
            submitted = count;
            make = {};
            selectFileWindow = {};
            waiting.clear();
        }

        CatalogIoStage stage = CatalogIoStage::Scan;
        uint32_t count = 0;
        uint32_t submitted = 0;
        Factory make;
        std::function<uint64_t(bool managed)> selectFileWindow;
        std::vector<std::pair<uint64_t, uint32_t>> waiting;
        std::vector<Receipt> active;
    };

    using RuntimeArtifactRoots = std::unordered_map<std::string, std::string>;

    struct QuerySnapshot
    {
        uint64_t generation = 0;
        std::unordered_map<std::string, std::string> guidToPath;
        std::unordered_map<std::string, std::string> pathToGuid;
        std::unordered_map<std::string, std::shared_ptr<const InxResourceMeta>> metas;
        std::unordered_map<std::string, std::string> shaderPaths;
        std::unordered_map<std::string, std::string> importErrors;
        std::shared_ptr<const RuntimeArtifactRoots> runtimeArtifactRoots;
        std::shared_ptr<const AssetCatalogSnapshot> catalog;
    };

    struct RuntimeCatalogPreparation
    {
        WorkingSet workingSet;
        std::shared_ptr<QuerySnapshot> query;
        uint64_t baseGeneration = 0;
        std::exception_ptr failure;
        nlohmann::json documents;
    };

    struct PendingRefreshCommit
    {
        PathWriteScope writeScope;
        std::vector<PathWriteScope> artifactWriteScopes;
        enum class Phase : uint8_t
        {
            MetadataPlan,
            MetadataPrepare,
            MetadataMerge,
            SidecarCreate,
            SidecarPlan,
            MetadataPublish,
            ImportPrepare,
            Import,
            ImportMerge,
            MetadataOutputPlan,
            MetadataOutputMeasure,
            MetadataOutputStage,
            MetadataWrite,
            FileStateMerge,
            ReadyToBuildIndex,
            IndexBuild,
            ReadyToFinalize,
        };

        AssetScanArtifact scanArtifact;
        std::string projectRoot;
        std::vector<WorkerMetadataPrepare> workerMetadata;
        std::vector<uint32_t> metadataPlans;
        std::vector<PendingAsset> pendingImports;
        std::vector<size_t> metadataPublications;
        std::vector<size_t> metadataAdoptions;
        std::vector<WorkerImport> workerImports;
        std::unordered_map<std::string, std::vector<std::string>> restoredDependencies;
        WorkingSet stagedWorkingSet;
        std::unordered_map<std::string, std::vector<std::string>> committedDependencies;
        std::vector<DocumentTransactionEntry> metadataWrites;
        std::vector<WorkerMetadataOutput> metadataOutputs;
        std::unordered_set<std::string> stagedTextureGuids;
        std::vector<size_t> importWave;
        uint64_t importStorageWaitBytes = 0;
        std::exception_ptr metadataWriteFailure;
        std::exception_ptr indexBuildFailure;
        std::optional<AssetIndex> builtAssetIndex;
        std::shared_ptr<QuerySnapshot> builtQuerySnapshot;
        std::shared_ptr<const AssetDependencySnapshot> builtDependencySnapshot;
        bool indexRebuildRequired = false;
        uint64_t stagedBytes = 0;
        double stagingMilliseconds = 0.0;
        uint64_t journalBytes = 0;
        double journalSerializeMilliseconds = 0.0;
        double journalWriteMilliseconds = 0.0;
        double journalApplyMilliseconds = 0.0;
        double indexBuildMilliseconds = 0.0;
        double indexSaveMilliseconds = 0.0;
        double queryBuildMilliseconds = 0.0;
        double dependencyBuildMilliseconds = 0.0;
        std::thread::id indexProducerThread;
        std::thread::id queryProducerThread;
        std::thread::id dependencyProducerThread;
        uint64_t expectedDependencyGeneration = 0;
        std::unordered_map<std::string, AssetFileFingerprint> committedMetadataFingerprints;
        std::chrono::steady_clock::time_point commitStarted;
        std::chrono::steady_clock::time_point importStarted;
        std::chrono::steady_clock::time_point metadataWriteStarted;
        size_t metadataIdentityCursor = 0;
        std::unordered_map<std::string, std::string> authoredGuidPaths;
        size_t metadataMergeCursor = 0;
        size_t metadataPublicationEraseCursor = 0;
        size_t metadataPublicationMergeCursor = 0;
        size_t importRequestCursor = 0;
        size_t importResultMergeCursor = 0;
        size_t metadataWritePrepareCursor = 0;
        size_t dependencyMergeCursor = 0;
        size_t fileStateMergeCursor = 0;
        bool importMergeInitialized = false;
        double ownerFinalizeMilliseconds = 0.0;
        double ownerMergeMaxSliceMilliseconds = 0.0;
        size_t ownerMergeSliceCount = 0;
        Phase phase = Phase::MetadataPrepare;
    };

    void AssertMutationThread(const char *operation) const;
    void AssertNoPendingCommit(const char *operation) const;
    [[nodiscard]] bool CanReadWorkingSet() const;
    [[nodiscard]] WorkingSet TakeWorkingSet();
    void InstallWorkingSet(WorkingSet workingSet);
    [[nodiscard]] WorkingSet CaptureRuntimeWorkingSet() const;
    static size_t PrepareRuntimeCatalog(RuntimeCatalogPreparation &preparation, const nlohmann::json &document,
                                        const std::string &contentRoot,
                                        const std::unordered_set<std::string> &builtinRoots, bool trustedPackage,
                                        const std::unordered_set<std::string> *newGuids,
                                        const std::function<void(const char *)> &reportPhase = {});
    void InstallPreparedRuntimeCatalog(RuntimeCatalogPreparation &preparation) noexcept;
    [[nodiscard]] std::string RuntimeArtifactRoot(const std::string &guid) const;
    void PublishQuerySnapshot(bool includeCatalog = true);
    void PublishModelSubAssetEvents(const std::shared_ptr<const QuerySnapshot> &previous,
                                    const std::unordered_set<std::string> *changedGuids = nullptr);
    void UpdateModelSubAssets(const std::string &owner, const QuerySnapshot &previous, QuerySnapshot &snapshot,
                              std::unordered_set<std::string> &changedGuids);
    void PublishRefreshAssetEvents(const WorkingSet &previous);
    [[nodiscard]] std::unordered_set<std::string> PublishQuerySnapshotForPaths(const std::vector<std::string> &paths);
    void InstallQuerySnapshot(std::shared_ptr<QuerySnapshot> snapshot) noexcept;
    static void BuildShaderPathIndex(QuerySnapshot &snapshot);
    [[nodiscard]] std::shared_ptr<const QuerySnapshot> LoadQuerySnapshot() const;
    [[nodiscard]] AssetScanRequest CaptureScanRequest() const;
    [[nodiscard]] static AssetScanArtifact BuildScanArtifact(const AssetScanRequest &request);
    static void PlanMetadataRead(WorkerMetadataPrepare &item, uint64_t range = 0);
    void RequireMetadataStorage(WorkerMetadataPrepare &item, const std::shared_ptr<FrameCpuStorageClaim> &storage,
                                uint64_t extraBytes = 0) const;
    [[nodiscard]] std::shared_ptr<FrameCpuStorageClaim> ReserveMetadataRead(WorkerMetadataPrepare &item) const;
    static void PrepareMetadata(WorkerMetadataPrepare &item, TextureSourceSnapshot *textureSource = nullptr,
                                const nlohmann::json *sourceSettings = nullptr);
    static void FinishMetadataAdoption(WorkerMetadataPrepare &item, TextureSourceSnapshot *textureSource);
    static void PublishMissingMetadata(WorkerMetadataPrepare &item, const std::string &projectRoot,
                                       TextureSourceSnapshot *textureSource = nullptr, bool deferAdoption = false);
    [[nodiscard]] static AssetIndex
    BuildDerivedIndexArtifact(const WorkingSet &workingSet,
                              const std::unordered_map<std::string, std::vector<std::string>> &dependenciesByGuid,
                              const std::string &normalizedProjectRoot);
    [[nodiscard]] static std::shared_ptr<QuerySnapshot>
    BuildQuerySnapshotArtifact(const std::unordered_map<std::string, std::string> &guidToPath,
                               const std::unordered_map<std::string, std::string> &pathToGuid,
                               const std::unordered_map<std::string, std::shared_ptr<InxResourceMeta>> &metas,
                               const std::unordered_map<std::string, CachedFileState> &fileStates,
                               const std::unordered_map<std::string, ImportResultState> &importResults,
                               uint64_t generation, bool includeCatalog = true);
    [[nodiscard]] bool CommitScanArtifact(AssetScanArtifact artifact, uint64_t expectedQueryGeneration);
    void QueueRefreshMetadataPlan(const std::shared_ptr<PendingRefreshCommit> &state);
    void QueueRefreshMetadata(const std::shared_ptr<PendingRefreshCommit> &state, bool adopting);
    [[nodiscard]] bool ContinuePendingMetadataMerge(const std::shared_ptr<PendingRefreshCommit> &state);
    [[nodiscard]] bool ContinuePendingMetadataPublish(const std::shared_ptr<PendingRefreshCommit> &state);
    [[nodiscard]] bool ContinuePendingImportPrepare(const std::shared_ptr<PendingRefreshCommit> &state);
    [[nodiscard]] bool BeginNextImportWave(const std::shared_ptr<PendingRefreshCommit> &state, bool managed,
                                           bool drain);
    [[nodiscard]] static bool AreImportWaveWritesComplete(const PendingRefreshCommit &state);
    [[nodiscard]] static bool FinishImportWave(PendingRefreshCommit &state, bool drain);
    [[nodiscard]] bool ContinuePendingImportMerge(const std::shared_ptr<PendingRefreshCommit> &state);
    void QueueRefreshMetadataOutput(const std::shared_ptr<PendingRefreshCommit> &state,
                                    PendingRefreshCommit::Phase phase);
    [[nodiscard]] static bool AreMetadataOutputsComplete(const PendingRefreshCommit &state);
    static void DrainMetadataOutputs(PendingRefreshCommit &state);
    void BeginRefreshMetadataTransaction(const std::shared_ptr<PendingRefreshCommit> &state);
    [[nodiscard]] bool ContinuePendingFileStateMerge(const std::shared_ptr<PendingRefreshCommit> &state);
    void BeginPendingIndexBuild(const std::shared_ptr<PendingRefreshCommit> &state);
    void FinalizePendingRefreshCommit(const std::shared_ptr<PendingRefreshCommit> &state);
    void QueueCatalogWork(CatalogWork &work, CatalogIoStage stage, uint32_t count, CatalogWork::Factory make,
                          CatalogWork::StoragePlan storagePlan = {});
    void QueueCatalogJob(CatalogWork &work, CatalogIoStage stage, JobSystem::JobFn job);
    void QueueMetadataRead(CatalogWork &work, CatalogIoStage stage, WorkerMetadataPrepare &item,
                           std::function<void(std::shared_ptr<FrameCpuStorageClaim>)> read);
    void QueueMetadataPreparationStep(CatalogWork &work, std::shared_ptr<WorkerMetadataPrepare> item);
    void DrainMetadataCapture(CatalogWork &work, const std::shared_ptr<WorkerMetadataPrepare> &item);
    static FrameWorkResource CatalogStageResource(CatalogIoStage stage);
    [[nodiscard]] bool CanSubmitCatalogStage(CatalogIoStage stage, std::optional<uint64_t> prediction = {}) const;
    [[nodiscard]] bool CanSubmitCatalogWork(const CatalogWork &work) const;
    bool SubmitCatalogWork(CatalogWork &work, bool managed, bool drain = false);
    void ObserveCatalogWork(CatalogWork &work) noexcept;
    void WaitForCatalogWork(CatalogWork &work) noexcept;
    void DrainCatalogTextureStage(CatalogWork &work, WorkerImport &item);
    bool AdvanceRefresh(bool managed, bool drain = false);
    [[nodiscard]] bool CanAdvanceWatcherImport() const;
    [[nodiscard]] bool IsWatcherMetadataWaiting() const;
    [[nodiscard]] static bool IsWatcherSourceCurrent(const PendingWatcherImport &pending);
    bool AdvanceWatcherImport(bool managed);
    void CompleteWatcherWriteForExplicitImport(const PathWriteScope &scope);
    void WriteWatcherImport(PendingWatcherImport &pending);
    void PublishWatcherImport(PendingWatcherImport &pending);
    void QueueWatcherMetadataPlan(const std::shared_ptr<PendingWatcherImport> &pending);
    void QueueWatcherCook(const std::shared_ptr<PendingWatcherImport> &pending);
    void QueueModelMetadataPlan(const std::shared_ptr<PendingModelReimport> &pending);
    void QueueTextureApplyPreparation(const std::shared_ptr<PendingModelReimport> &pending);
    void QueueModelCook(const std::shared_ptr<PendingModelReimport> &pending);
    [[nodiscard]] bool CanAdvanceModelReimport() const;
    bool AdvanceModelReimport(bool managed, bool drain = false);
    void WriteModelReimport(PendingModelReimport &pending);
    void PublishModelReimport(PendingModelReimport &pending);
    bool CommitReadyRefresh(bool managed, bool drain);
    void RethrowRefreshFailure();
    [[nodiscard]] static bool IsIgnoredImportPath(const std::filesystem::path &path);
    [[nodiscard]] bool IsMetaFile(const std::filesystem::path &path) const;
    void UpdateMapping(const std::string &guid, const std::string &path);
    void RemoveMappingByGuid(const std::string &guid);
    void RemoveMappingByPath(const std::string &path);
    void UpdateCachedFileState(const std::string &path, bool readOnly);

    /// Cook and publish an asset. The caller decides whether a failure belongs in the catalog.
    std::optional<std::string> RunImporter(const std::string &guid, const std::string &path, bool isReimport,
                                           bool persistMetadata, const InxResourceMeta *candidateMetadata,
                                           const AssetFileFingerprint *expectedSource,
                                           const AtomicFileState *expectedMetadata, const PathWriteScope &writeScope,
                                           TextureSourceSnapshot *textureSource);
    bool PrepareReimportInput(const std::string &path, WorkerMetadataPrepare &candidate, AssetMutationResult &result);
    static bool PrepareReimportMetadata(WorkerMetadataPrepare &candidate, const nlohmann::json &settings,
                                        AssetMutationResult &result, TextureSourceSnapshot *textureSource = nullptr);
    ImportRequest MakeImportRequest(const std::string &guid, const std::string &path, bool isReimport,
                                    const InxResourceMeta &metadata) const;
    ImportRequest MakeImportRequestBase(const std::string &guid, const std::string &path, bool isReimport,
                                        const InxResourceMeta &metadata) const;
    void ValidateImportPublication(const ImportRequest &request, ImportArtifact &artifact, bool creating = false) const;
    static void StageTextureArtifact(WorkerImport &item, std::shared_ptr<FrameCpuStorageClaim> storage);
    static std::vector<DocumentTransactionEntry>
    BuildImportWrites(const ImportRequest &request, ImportArtifact &artifact, const AtomicFileState *expectedMetadata,
                      std::optional<DocumentTransactionEntry> stagedTexture = std::nullopt);
    std::vector<DocumentTransactionFileState> PersistImportWrites(const ImportRequest &request,
                                                                  std::vector<DocumentTransactionEntry> writes,
                                                                  const PathWriteScope &writeScope) const;
    void InstallImportArtifact(const std::string &guid, ImportArtifact artifact);
    void FinishImport(AssetMutationResult &result, bool imported, const CachedFileState *committedFiles = nullptr,
                      bool notify = true);
    void FinishReimport(AssetMutationResult &result, const CachedFileState *committedFiles = nullptr,
                        const std::shared_ptr<AssetLoadTicket> &repairTicket = {}, bool notify = true);
    void StartModelReimport(const std::string &sourcePath, const std::string &textureGuid,
                            const nlohmann::json &settings, ModelReimportPublished onPublished,
                            std::shared_ptr<AssetLoadTicket> repairTicket = {});
    [[nodiscard]] bool HasCurrentMeshArtifacts(const std::string &guid) const;
    [[nodiscard]] bool CanBeginMeshArtifactRepair() const;
    void BeginMeshArtifactRepair(const std::shared_ptr<AssetLoadTicket> &ticket);
    AssetRegistry *BoundRegistry() const;

    std::string CreateOrLoadMetadata(const std::string &filePath, ResourceType type, bool readOnly,
                                     bool persistMetadata, const std::string &identityKey,
                                     std::optional<AtomicFileState> *expectedMetadata = nullptr,
                                     const PathWriteScope &writeScope = {},
                                     TextureSourceSnapshot *textureSource = nullptr);
    void DeleteMetadata(const std::string &filePath);
    void MoveMetadata(const std::string &oldFilePath, const std::string &newFilePath, const PathWriteScope &writeScope,
                      const InxResourceMeta *replacement = nullptr);
    void RebuildDerivedIndex();
    [[nodiscard]] bool IsReadOnlyPath(const std::string &normalizedPath) const;

    std::string m_projectRoot;
    std::string m_blenderExecutable;
    std::string m_blenderExportScript;
    std::shared_ptr<PendingModelReimport> m_pendingModelReimport;
    std::shared_ptr<PendingWatcherImport> m_pendingWatcherImport;
    std::optional<FinishedModelApply> m_finishedModelApply;
    std::string m_assetsRoot;
    std::vector<std::string> m_extraScanRoots;
    std::unordered_set<std::string> m_readOnlyScanRoots;
    std::thread::id m_ownerThread;
    bool m_initialized = false;

    // GUID -> path
    std::shared_ptr<AssetDependencyGraph> m_dependencyGraph = std::make_shared<AssetDependencyGraph>();
    std::unordered_map<std::string, std::string> m_guidToPath;
    // normalized path -> GUID
    std::unordered_map<std::string, std::string> m_pathToGuid;

    // Asset importer registry (populated in Initialize)
    ImporterRegistry m_importerRegistry;

    // Resource loaders (one per ResourceType, used for meta creation/loading)
    // Non-owning pointers — ownership is in AssetRegistry::m_loaders.
    std::unordered_map<ResourceType, IAssetLoader *> m_loaders;
    // GUID -> metadata cache
    std::unordered_map<std::string, std::shared_ptr<InxResourceMeta>> m_metas;

    std::unordered_map<std::string, CachedFileState> m_fileStates;

    // Readers atomically acquire one immutable generation. Only the owner
    // thread mutates the working maps above and publishes completed commits.
    std::shared_ptr<const QuerySnapshot> m_querySnapshot;
    uint64_t m_queryGeneration = 0;

    AssetIndex m_assetIndex;
    std::string m_assetIndexPath;
    // Last committed catalog retained across source/meta mutations. The live
    // AssetIndex is deliberately invalidated while a refresh is pending so a
    // build cannot consume stale data; Editor startup may still restore these
    // identities and validate them asynchronously after the window appears.
    std::string m_assetStartupCachePath;
    std::string m_assetTransactionJournalPath;
    std::unordered_map<std::string, ImportResultState> m_importResults;
    size_t m_lastRefreshReusedCount = 0;
    size_t m_lastRefreshImportedCount = 0;
    std::vector<std::string> m_lastRefreshImportedPaths;
    size_t m_lastRefreshImporterTaskCount = 0;
    size_t m_lastRefreshMetadataTaskCount = 0;
    size_t m_lastRefreshWorkerMetadataCount = 0;
    size_t m_lastRefreshWorkerImporterCount = 0;
    size_t m_lastRefreshScannedCount = 0;
    double m_lastRefreshScanMilliseconds = 0.0;
    double m_lastRefreshCommitMilliseconds = 0.0;
    double m_lastModelReimportWorkerMilliseconds = 0.0;
    double m_lastModelReimportPrepareMilliseconds = 0.0;
    double m_lastModelReimportPersistenceMilliseconds = 0.0;
    double m_lastModelReimportLivePublicationMilliseconds = 0.0;
    double m_lastRefreshPrepareMilliseconds = 0.0;
    double m_lastRefreshFinalizeMilliseconds = 0.0;
    double m_lastRefreshOwnerMergeMaxSliceMilliseconds = 0.0;
    size_t m_lastRefreshOwnerMergeSliceCount = 0;
    double m_lastRefreshMetadataWriteMilliseconds = 0.0;
    uint64_t m_lastRefreshStagedBytes = 0;
    double m_lastRefreshStagingMilliseconds = 0.0;
    uint64_t m_lastRefreshJournalBytes = 0;
    double m_lastRefreshJournalSerializeMilliseconds = 0.0;
    double m_lastRefreshJournalWriteMilliseconds = 0.0;
    double m_lastRefreshJournalApplyMilliseconds = 0.0;
    double m_lastRefreshRestoreMilliseconds = 0.0;
    double m_lastRefreshImportMilliseconds = 0.0;
    double m_lastRefreshIndexBuildMilliseconds = 0.0;
    double m_lastRefreshIndexSaveMilliseconds = 0.0;
    bool m_lastRefreshIndexBuildOnWorker = false;
    double m_lastRefreshQueryBuildMilliseconds = 0.0;
    bool m_lastRefreshQueryBuildOnWorker = false;
    double m_lastRefreshDependencyBuildMilliseconds = 0.0;
    bool m_lastRefreshDependencyBuildOnWorker = false;
    double m_lastRefreshPublishMilliseconds = 0.0;
    bool m_lastRefreshScanOnWorker = false;
    bool m_assetIndexDirty = false;
    std::shared_ptr<PendingAssetScan> m_pendingAssetScan;
    std::shared_ptr<PendingRefreshCommit> m_pendingRefreshCommit;
    FrameWorkOwner *m_frameWork = nullptr;
    CatalogWork m_refreshWork;
    CatalogWork m_applyWork;
    CatalogWork m_watcherWork;
    CatalogWork m_runtimeCatalogWork;
    std::shared_ptr<RuntimeCatalogPreparation> m_runtimeCatalog;
    bool m_cookedRuntimeCatalogInstalled = false;
    std::array<uint64_t, static_cast<size_t>(CatalogIoStage::Count)> m_catalogIoCosts{};
    std::exception_ptr m_refreshFailure;
    bool m_preferApply = false;

  public:
    /// @brief Set a meta-creation loader for a resource type.
    /// Called by AssetRegistry after all loaders are registered.
    void SetMetaLoader(ResourceType type, IAssetLoader *loader)
    {
        AssertMutationThread("SetMetaLoader");
        AssertNoPendingCommit("SetMetaLoader");
        if (!loader)
            throw std::invalid_argument("AssetDatabase meta loader cannot be null");
        m_loaders[type] = loader;
    }
};

} // namespace infernux
