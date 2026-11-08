#pragma once

#include "core/FoundationApi.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

namespace infernux
{

class FrameWorkOwner;

enum class FileWriteMode : uint8_t
{
    Replace,
    Create,
};

enum class FilePublication : uint8_t
{
    Written,
    AlreadyExists,
};

/// One same-volume publication. Existing readers retain the old file object.
bool ReplaceFileAtomically(const std::filesystem::path &source, const std::filesystem::path &target,
                           std::error_code &error);

/// Publish a complete private file without ever replacing an existing name.
/// Written consumes the source name; AlreadyExists leaves it with its owner.
std::optional<FilePublication> PublishFileIfAbsent(const std::filesystem::path &source,
                                                   const std::filesystem::path &target, std::error_code &error);

INFERNUX_FOUNDATION_API std::filesystem::path MakeAtomicTemporaryPath(const std::filesystem::path &target);
INFERNUX_FOUNDATION_API bool IsAtomicTemporaryPath(const std::filesystem::path &path);

#ifdef _WIN32
namespace detail
{
// Extraction keeps its private output handle through verification/publication.
bool ReplaceOpenFileAtomically(void *handle, const std::filesystem::path &target, std::error_code &error);
} // namespace detail
#else
namespace detail
{
// Archive outputs keep their already-open directory throughout publication.
std::optional<FilePublication> PublishFileIfAbsentAt(int directory, const std::filesystem::path &source,
                                                     const std::filesystem::path &target, std::error_code &error);
} // namespace detail
#endif

struct AtomicFileState
{
    bool exists = false;
    uint64_t size = 0;
    int64_t modifiedNs = 0;
    uint64_t contentHash = 0;

    [[nodiscard]] bool operator==(const AtomicFileState &other) const noexcept
    {
        return exists == other.exists && size == other.size && modifiedNs == other.modifiedNs &&
               contentHash == other.contentHash;
    }
};

struct AtomicWriteOptions
{
    FileWriteMode mode = FileWriteMode::Replace;
    std::optional<AtomicFileState> expectedState;
};

struct AtomicWriteReceipt
{
    FilePublication publication = FilePublication::Written;
    // Only a Written receipt owns a durable file revision. An Existing caller
    // must consume the winner before it can revise or use its document.
    std::optional<AtomicFileState> fileState;
};

/// A complete, flushed sibling of a destination. Its journal owns cleanup.
struct AtomicPreparedFile
{
    std::string path;
    AtomicFileState state;
};

/// Retain a private output and any conditional input across IO calls.
class AtomicFileWriter final
{
  public:
    /// With a resolved path, completeBytes allows direct IO for one full Append.
    INFERNUX_FOUNDATION_API explicit AtomicFileWriter(std::string path, AtomicWriteOptions options = {},
                                                      uint64_t completeBytes = 0, bool namedStage = false);
    INFERNUX_FOUNDATION_API AtomicFileWriter(std::string path, AtomicWriteOptions options,
                                             const AtomicPreparedFile &prepared);
    INFERNUX_FOUNDATION_API ~AtomicFileWriter();
    INFERNUX_FOUNDATION_API void Append(std::string_view bytes, const FrameWorkOwner *storageOwner = nullptr);
    [[nodiscard]] uint64_t WrittenBytes() const noexcept;
    void VerifyExpected(uint64_t maximumBytes);
    [[nodiscard]] uint64_t VerifiedBytes() const noexcept;
    /// Consume the output after its expected input has been verified.
    [[nodiscard]] INFERNUX_FOUNDATION_API AtomicWriteReceipt Publish();

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

/// Capture a stable state used by conditional document writes. Missing files
/// are represented explicitly; inspection failures throw with path context.
INFERNUX_FOUNDATION_API AtomicFileState CaptureAtomicFileState(const std::string &path);

struct AtomicTextFileSnapshot
{
    std::string content;
    AtomicFileState state;
};

/// Read once and retain the conditional-write identity of those exact bytes.
/// Parsing/publishing consumers must not replace it with a later disk capture.
INFERNUX_FOUNDATION_API AtomicTextFileSnapshot ReadTextFileSnapshot(const std::string &path);

/// Write complete UTF-8 text through an exclusive same-directory temporary file.
/// The receipt describes this request's actual publication, never a later read
/// of the destination. IO failure returns no receipt and supplies an error.
INFERNUX_FOUNDATION_API std::optional<AtomicWriteReceipt> WriteTextFileAtomically(const std::string &path,
                                                                                  std::string_view content,
                                                                                  std::string &error,
                                                                                  AtomicWriteOptions options = {});

/// Remove a file and persist the directory update where the platform exposes directory fsync.
/// Missing files are treated as already removed.
INFERNUX_FOUNDATION_API bool RemoveFileDurably(const std::string &path, std::string &error);

/// Persist a batch of name changes in this directory.
void FlushFileDirectory(const std::filesystem::path &path);

} // namespace infernux
