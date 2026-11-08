#pragma once

#include <cstddef>
#include <cstdint>
#include <function/resources/AssetRuntimeApi.h>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace infernux
{
class FileInput;

/// One metadata input. Hash-only consumers stream the original open file;
/// parsers explicitly request its bytes. In-memory authoring borrows its input.
class AssetSource
{
  public:
    enum class ReadMode
    {
        Once,
        Retained
    };

    INFERNUX_ASSET_RUNTIME_API explicit AssetSource(std::string path, ReadMode mode = ReadMode::Once);
    INFERNUX_ASSET_RUNTIME_API AssetSource(std::string path, std::string_view bytes);
    INFERNUX_ASSET_RUNTIME_API ~AssetSource();
    AssetSource(const AssetSource &) = delete;
    AssetSource &operator=(const AssetSource &) = delete;

    const std::string &Path() const
    {
        return m_path;
    }
    uint64_t Size() const
    {
        return m_size;
    }
    INFERNUX_ASSET_RUNTIME_API std::string_view Bytes();
    /// Consume this input, moving owned bytes or copying a borrowed authoring view.
    INFERNUX_ASSET_RUNTIME_API std::string TakeBytes() &&;
    INFERNUX_ASSET_RUNTIME_API const std::string &ContentHash();

  private:
    friend class InxDefaultLoader;
    INFERNUX_ASSET_RUNTIME_API bool ReadNext(size_t maximum);
    void CheckRevision();

    std::string m_path;
    std::unique_ptr<FileInput> m_input;
    uint64_t m_size;
    std::optional<std::pair<int64_t, int64_t>> m_modified;
    std::optional<std::string> m_bytes;
    std::string_view m_view;
    std::optional<std::string> m_hash;
};
} // namespace infernux
