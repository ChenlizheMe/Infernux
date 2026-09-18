#pragma once

#include <core/types/InxFwdType.h>
#include <function/resources/InxResource/InxResourceMeta.h>

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace infernux
{

/**
 * @brief Immutable input captured before importer execution.
 */
struct ImportRequest
{
    std::string sourcePath;
    std::string guid;
    ResourceType resourceType = ResourceType::DefaultText;
    InxResourceMeta metadata;
    bool isReimport = false;
    // Editor-owned tool configuration is captured on the owner thread; worker
    // imports never call Python or read mutable editor preferences.
    std::string projectRoot;
    std::string blenderExecutable;
    std::string blenderExportScript;
    // Native immutable catalog lookup, captured before worker execution.
    // No AssetDatabase mutation or Python callback is permitted here.
    std::function<std::string(const std::string &, bool linear)> resolveTextureGuid;
};

/**
 * @brief Pure CPU result. AssetDatabase is the only owner allowed to publish it.
 */
struct ImportArtifact
{
    explicit ImportArtifact(InxResourceMeta metadataSnapshot) : metadata(std::move(metadataSnapshot))
    {
    }

    InxResourceMeta metadata;
    // Durable dependency identity is GUID-only. Importers must never turn a
    // path, path_hint, filename, or shader search result into an asset edge.
    // A path_hint is non-authoritative editor/diagnostic metadata and may be
    // stale after an asset move. AssetDatabase validates this contract before
    // publishing the dependency graph.
    std::vector<std::string> dependencies;
    bool dependenciesAuthoritative = false;

    // Importers may report source-local paths only as an authoring hand-off.
    // AssetDatabase resolves these paths against the current scan catalog and
    // publishes only the resulting GUIDs. They never enter the durable graph.
    std::vector<std::string> dependencyPathHints;
    // Source-to-GUID bindings resolved by the immutable worker catalog. The
    // publication boundary rejects asset identity changes during the import.
    std::vector<std::pair<std::string, std::string>> resolvedTextureSources;

    enum class RuntimeArtifactKind : uint8_t
    {
        Primary,
        SkinnedMesh,
    };

    struct RuntimeCpuArtifact
    {
        RuntimeArtifactKind kind = RuntimeArtifactKind::Primary;
        ResourceType resourceType = ResourceType::DefaultBinary;
        std::string bytes;
        std::string guid; // Empty for the source itself; set for an owned imported Texture.
    };

    std::vector<RuntimeCpuArtifact> runtimeCpuArtifacts;
};

/**
 * @brief Abstract base for asset importers.
 *
 * ── Architecture Note ──────────────────────────────────────────────
 * The asset pipeline has two distinct layers:
 *
 *   AssetImporter  (this class, in AssetImporter/)
 *     Responsible for the *import strategy*: how a raw source file
 *     (.png, .fbx, .glsl …) is processed, what metadata is generated,
 *     and what import settings are stored in the .meta sidecar.
 *     AssetDatabase drives importers during first-import and reimport.
 *
 *   IAssetLoader  (in AssetRegistry/IAssetLoader.h)
 *     Responsible for *runtime loading*: turning an already-imported
 *     asset into an in-memory object (InxMesh, InxTexture, …).
 *     AssetRegistry delegates Load / Reload / ScanDependencies here.
 *
 * The generic helpers in InxFileLoader/ implement IAssetLoader for
 * text, binary, and script resources.
 *
 * ──────────────────────────────────────────────────────────────────
 *
 * Each concrete importer handles one category of resource
 * (textures, shaders, materials, …).  The ImporterRegistry
 * maps file extensions to their importer, and AssetDatabase
 * calls Import() / Reimport() during the asset pipeline.
 *
 * Importers are thin wrappers that delegate heavy work to
 * the registered IAssetLoader implementations.
 */
class AssetImporter
{
  public:
    virtual ~AssetImporter() = default;

    /// @brief Resource type this importer handles
    [[nodiscard]] virtual ResourceType GetResourceType() const = 0;

    /// @brief File extensions this importer supports (e.g. {".png", ".jpg"})
    [[nodiscard]] virtual std::vector<std::string> GetSupportedExtensions() const = 0;

    /// @brief Build a pure CPU artifact without publishing shared engine state.
    /// External input failures are reported with exceptions.
    [[nodiscard]] virtual ImportArtifact Import(const ImportRequest &request) const = 0;

    [[nodiscard]] virtual ImportArtifact Reimport(const ImportRequest &request) const
    {
        return Import(request);
    }

    /// @brief Called after meta is loaded, before Import. Allows the importer
    ///        to fill default import settings if missing.
    virtual void EnsureDefaultSettings(InxResourceMeta & /*meta*/) const
    {
        // Override in concrete importers to populate import_settings
    }
};

} // namespace infernux
