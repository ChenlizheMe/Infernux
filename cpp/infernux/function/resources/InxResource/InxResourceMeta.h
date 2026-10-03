#pragma once
#include <core/types/InxFwdType.h>
#include <function/resources/AssetRuntimeApi.h>

#include <nlohmann/json.hpp>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>

namespace infernux
{
// ----------------------------------
// InxResourceMeta Class
// ----------------------------------

class InxResourceMeta
{
  public:
    // Type definitions
    using MetadataValue = std::variant<std::string, int, bool, size_t, float, ResourceType>;
    using MetadataType = std::pair<std::string, MetadataValue>;
    using MetadataMap = std::unordered_map<std::string, MetadataType>;

    InxResourceMeta() = default;
    ~InxResourceMeta() = default;

    INFERNUX_ASSET_RUNTIME_API void Init(const char *content, size_t contentSize, const std::string &filePath,
                                         ResourceType type);

    // Copy constructor and assignment operator
    InxResourceMeta(const InxResourceMeta &other) = default;
    InxResourceMeta &operator=(const InxResourceMeta &other) = default;

    // Move constructor and assignment operator
    InxResourceMeta(InxResourceMeta &&other) noexcept = default;
    InxResourceMeta &operator=(InxResourceMeta &&other) noexcept = default;

    // Metadata operations
    INFERNUX_ASSET_RUNTIME_API void AddMetadata(const std::string &key, const std::string &value);
    INFERNUX_ASSET_RUNTIME_API void AddMetadata(const std::string &key, const char *value);
    INFERNUX_ASSET_RUNTIME_API void AddMetadata(const std::string &key, int value);
    INFERNUX_ASSET_RUNTIME_API void AddMetadata(const std::string &key, bool value);
    INFERNUX_ASSET_RUNTIME_API void AddMetadata(const std::string &key, size_t value);
    INFERNUX_ASSET_RUNTIME_API void AddMetadata(const std::string &key, float value);
    INFERNUX_ASSET_RUNTIME_API void AddMetadata(const std::string &key, ResourceType value);
    INFERNUX_ASSET_RUNTIME_API void AddMetadata(const std::string &key, const nlohmann::json &value);

    /// Copy one metadata entry without changing its persisted type tag.
    /// Returns false when the source has no such key or this instance already
    /// contains it.
    INFERNUX_ASSET_RUNTIME_API bool CopyMetadataIfMissing(const InxResourceMeta &source, const std::string &key);

    /// @brief Get metadata value with type registry
    /// @tparam T The type to retrieve
    /// @param key type key to retrieve the value for
    /// @return The metadata value of type T
    template <typename T> T GetDataAs(const std::string &key) const;

    // Metadata has one closed, deterministic value representation. Android
    // hosts the asset runtime and Python bindings in separate DSOs, so an
    // open-ended RTTI container is not a valid ABI boundary.
    INFERNUX_ASSET_RUNTIME_API std::string GetStringData(const std::string &key) const;
    INFERNUX_ASSET_RUNTIME_API int GetIntData(const std::string &key) const;
    INFERNUX_ASSET_RUNTIME_API bool GetBoolData(const std::string &key) const;
    INFERNUX_ASSET_RUNTIME_API size_t GetSizeData(const std::string &key) const;
    INFERNUX_ASSET_RUNTIME_API float GetFloatData(const std::string &key) const;
    INFERNUX_ASSET_RUNTIME_API nlohmann::json GetJsonData(const std::string &key) const;

    // Fixed getters
    INFERNUX_ASSET_RUNTIME_API const std::string &GetResourceName() const;
    INFERNUX_ASSET_RUNTIME_API const std::string &GetHashCode() const;
    INFERNUX_ASSET_RUNTIME_API const std::string &GetGuid() const;
    INFERNUX_ASSET_RUNTIME_API const MetadataMap &GetMetadata() const;
    INFERNUX_ASSET_RUNTIME_API const ResourceType &GetResourceType() const;

    /// @brief Check if metadata has a specific key
    INFERNUX_ASSET_RUNTIME_API bool HasKey(const std::string &key) const;

    /// @brief Update file path (for move/rename operations)
    INFERNUX_ASSET_RUNTIME_API void UpdateFilePath(const std::string &newFilePath);

    // Serialization methods (JSON only)
    [[nodiscard]] INFERNUX_ASSET_RUNTIME_API nlohmann::json SerializeDocument() const;
    /// Serialize metadata for a project sidecar without embedding the local
    /// checkout's absolute path.  Runtime/editor memory keeps the resolved
    /// path; only the durable document uses a project-relative path hint.
    [[nodiscard]] INFERNUX_ASSET_RUNTIME_API nlohmann::json
    SerializeDocumentPortable(const std::string &projectRoot) const;
    INFERNUX_ASSET_RUNTIME_API void DeserializeDocument(const nlohmann::json &document);
    INFERNUX_ASSET_RUNTIME_API bool SaveToFile(const std::string &metaFilePath) const;
    INFERNUX_ASSET_RUNTIME_API bool LoadFromFile(const std::string &metaFilePath);

    // Generate metadata file path from resource file path
    INFERNUX_ASSET_RUNTIME_API static std::string GetMetaFilePath(const std::string &resourceFilePath);
    INFERNUX_ASSET_RUNTIME_API static std::string NormalizeFilePath(const std::string &filePath);

  private:
    MetadataMap m_metadata;
};

} // namespace infernux

// Include template implementations
#include "InxResourceMeta.inl"
