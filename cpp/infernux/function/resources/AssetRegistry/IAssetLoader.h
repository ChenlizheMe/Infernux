#pragma once

#include "AssetSource.h"
#include <core/types/InxFwdType.h>
#include <function/resources/InxResource/InxResourceMeta.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <utility>

namespace infernux
{

class AssetDatabase;

enum class AssetReloadMode
{
    InPlace,
    Invalidate,
};

class RuntimeAssetPayload final
{
  public:
    RuntimeAssetPayload() = default;
    RuntimeAssetPayload(std::nullptr_t) noexcept
    {
    }

    template <typename T>
    RuntimeAssetPayload(std::shared_ptr<T> instance) : m_instance(std::move(instance)), m_type(typeid(T))
    {
    }

    [[nodiscard]] explicit operator bool() const noexcept
    {
        return static_cast<bool>(m_instance);
    }

    [[nodiscard]] std::type_index GetType() const noexcept
    {
        return m_type;
    }

    [[nodiscard]] const char *GetTypeName() const noexcept
    {
        return m_type.name();
    }

    [[nodiscard]] long GetUseCount() const noexcept
    {
        return m_instance.use_count();
    }

    [[nodiscard]] std::weak_ptr<const void> GetStorageOwner() const noexcept
    {
        return m_instance;
    }

    template <typename T> [[nodiscard]] std::shared_ptr<T> Get() const
    {
        if (!m_instance)
            return {};
        if (m_type != std::type_index(typeid(T)))
            throw std::invalid_argument(std::string("Runtime asset type mismatch: stored '") + m_type.name() +
                                        "', requested '" + typeid(T).name() + "'");
        return std::static_pointer_cast<T>(m_instance);
    }

  private:
    std::shared_ptr<void> m_instance;
    std::type_index m_type{typeid(void)};
};

struct WorkerAssetLoadPlan
{
    enum class ReadKind
    {
        RangedFile,
        GeometryArtifact,
        SkeletonArtifact,
        Count
    };

    struct Read
    {
        uint64_t bytes = 0;
        ReadKind kind = ReadKind::RangedFile;
    };

    uint64_t storageBytes = 0;
    /// The worker receives the admitted byte count and describes its next read
    /// when it returns an incomplete payload.
    std::function<RuntimeAssetPayload(uint64_t, Read &)> advance;
    Read read;
};

/// @brief Interface for type-specific asset loading/reloading in AssetRegistry.
///
/// ── Architecture Note ──────────────────────────────────────────────
/// This is the *runtime loading* layer of the asset pipeline.
/// IAssetLoader turns already-imported assets into live in-memory objects.
///
/// The *import strategy* layer lives in AssetImporter/ (AssetImporter.h).
/// That layer handles source-file processing and .meta generation.
///
/// See AssetImporter.h for the full two-layer pipeline description.
/// ──────────────────────────────────────────────────────────────────
///
/// Each ResourceType registers one IAssetLoader implementation.
/// AssetRegistry delegates Load / Reload / ScanDependencies to the loader.
class IAssetLoader
{
  public:
    virtual ~IAssetLoader() = default;

    /// Prepare a private CPU payload. Global dependency/residency state belongs
    /// to the registry owner and must not change before candidate publication.
    /// @return Type-tagged runtime object. Empty means this asset has no C++ runtime payload.
    virtual RuntimeAssetPayload Load(const std::string &filePath, const std::string &guid, AssetDatabase *adb) = 0;

    /// Worker loading is opt-in. Implementations may only read immutable state.
    [[nodiscard]] virtual bool SupportsWorkerLoad() const noexcept
    {
        return false;
    }

    /// Optional worker-side size discovery before a managed payload allocation.
    /// An accepted plan keeps its claim until its last step retires.
    virtual WorkerAssetLoadPlan PlanWorkerLoad(const std::string &, const std::string &, AssetDatabase *)
    {
        return {};
    }

    /// Mutable runtime objects keep their identity; immutable source revisions
    /// leave the cache and are prepared again through the original shared ticket.
    [[nodiscard]] virtual AssetReloadMode GetReloadMode() const noexcept
    {
        return AssetReloadMode::InPlace;
    }

    /// Synchronous in-place reload for asset types that own that boundary.
    virtual bool Reload(const RuntimeAssetPayload &, const std::string &, const std::string &, AssetDatabase *)
    {
        throw std::logic_error("Asset loader requires registry-managed publication");
    }

    /// Finish owner-dependent preparation on the private candidate. Dependencies
    /// may be loaded here; failure does not publish this candidate. This runs
    /// before residency accounting and publication, never on a worker.
    virtual void PreparePublication(const RuntimeAssetPayload &, const std::string &, AssetDatabase *)
    {
    }

    /// Apply loader-owned global state after the owner accepts the live payload.
    /// Synchronous loading, reloading and worker publication share this boundary.
    virtual void OnPublished(const RuntimeAssetPayload &, const std::string &, AssetDatabase *)
    {
    }

    /// A catalog move preserves the payload and its runtime version. Only
    /// loaders whose live objects retain a location need to update it here.
    virtual void OnRelocated(const RuntimeAssetPayload &, const std::string &)
    {
    }

    /// @brief Estimate bytes exclusively owned by this live CPU payload.
    /// A non-empty payload must never report zero.
    [[nodiscard]] virtual size_t EstimateRuntimeBytes(const RuntimeAssetPayload &payload) const = 0;

    /// @brief Return GUIDs of assets this asset depends on.
    ///
    /// Durable dependency identity is GUID-only. Implementations must never
    /// derive an edge from a path, path_hint, filename, or search result.
    virtual std::set<std::string> ScanDependencies(const std::string &filePath, AssetDatabase *adb) = 0;

    /// Source bytes are materialized only when this metadata parser needs them.
    virtual void CreateMeta(AssetSource &source, InxResourceMeta &metadata) const = 0;
};

} // namespace infernux
