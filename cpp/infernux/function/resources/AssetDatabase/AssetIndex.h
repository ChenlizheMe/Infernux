#pragma once

#include <core/types/InxFwdType.h>
#include <function/resources/AssetRuntimeApi.h>
#include <function/resources/InxResource/InxResourceMeta.h>

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace infernux
{

struct AssetFileFingerprint
{
    uint64_t size = 0;
    int64_t modifiedNs = 0;

    [[nodiscard]] bool operator==(const AssetFileFingerprint &other) const noexcept
    {
        return size == other.size && modifiedNs == other.modifiedNs;
    }
    [[nodiscard]] bool operator!=(const AssetFileFingerprint &other) const noexcept
    {
        return !(*this == other);
    }
};

struct AssetIndexEntry
{
    std::string normalizedPath;
    std::string guid;
    ResourceType resourceType = ResourceType::DefaultText;
    AssetFileFingerprint source;
    AssetFileFingerprint meta;
    std::string contentHash;
    std::vector<std::string> dependencies;
    bool readOnly = false;
    bool importSucceeded = true;
    std::string importError;
    std::string artifactPath;
    InxResourceMeta metadata;
};

class AssetIndex final
{
  public:
    INFERNUX_ASSET_RUNTIME_API void Reset(std::string normalizedProjectRoot);
    [[nodiscard]] INFERNUX_ASSET_RUNTIME_API bool Load(const std::string &path,
                                                       const std::string &normalizedProjectRoot);
    INFERNUX_ASSET_RUNTIME_API void Save(const std::string &path) const;

    [[nodiscard]] INFERNUX_ASSET_RUNTIME_API const AssetIndexEntry *Find(const std::string &normalizedPath) const;
    [[nodiscard]] const std::unordered_map<std::string, AssetIndexEntry> &Entries() const noexcept
    {
        return m_entries;
    }
    INFERNUX_ASSET_RUNTIME_API void Upsert(AssetIndexEntry entry);
    [[nodiscard]] size_t Size() const noexcept
    {
        return m_entries.size();
    }

    [[nodiscard]] INFERNUX_ASSET_RUNTIME_API nlohmann::json SerializeDocument() const;
    INFERNUX_ASSET_RUNTIME_API void DeserializeDocument(const nlohmann::json &document,
                                                        const std::string &normalizedProjectRoot);

  private:
    std::string m_projectRoot;
    std::unordered_map<std::string, AssetIndexEntry> m_entries;
};

} // namespace infernux
