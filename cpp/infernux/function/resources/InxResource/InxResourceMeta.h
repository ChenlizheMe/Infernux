#pragma once
#include <core/reflection/InxTypeRegistry.h>
#include <core/types/InxFwdType.h>
#include <function/resources/AssetRuntimeApi.h>

#include <any>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

namespace infernux
{
// ----------------------------------
// InxResourceMeta Class
// ----------------------------------

class InxResourceMeta
{
  public:
    // Type definitions
    using MetadataType = std::pair<std::string, std::any>;
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
    INFERNUX_ASSET_RUNTIME_API void AddMetadata(const std::string &key, const std::any &value);

    /// Copy one metadata entry without changing its persisted type tag.
    /// Returns false when the source has no such key or this instance already
    /// contains it.
    INFERNUX_ASSET_RUNTIME_API bool CopyMetadataIfMissing(const InxResourceMeta &source, const std::string &key);

    /// @brief Get metadata value with type registry
    /// @tparam T The type to retrieve
    /// @param key type key to retrieve the value for
    /// @return The metadata value of type T
    template <typename T> T GetDataAs(const std::string &key) const;

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
