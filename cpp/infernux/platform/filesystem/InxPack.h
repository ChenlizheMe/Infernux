#pragma once

#include "core/FoundationApi.h"

#include "IoWork.h"
#include "PathWriteScope.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <variant>
#include <vector>

namespace infernux
{
class FileInput;
class VerifiedFile;
} // namespace infernux

namespace infernux::inxpack
{

// This is the only supported InxPack container layout.
inline constexpr std::array<char, 8> kMagic = {'I', 'N', 'X', 'P', 'K', 'G', '\0', '\0'};
inline constexpr uint32_t kAlignment = 64u;

enum class Codec : uint8_t
{
    Store = 0,
    Zstandard = 1,
};

enum class CompressionProfile : uint8_t
{
    Development = 0,
    Release = 1,
};

inline constexpr int kDevelopmentCompressionLevel = 3;
// Player builds are frequent editor operations. Level 6 retains most of
// Zstandard's size win without turning finalization into an offline encode.
inline constexpr int kReleaseCompressionLevel = 6;
inline constexpr int kMinimumCompressionLevel = 1;
inline constexpr int kMaximumCompressionLevel = 22;

struct WriteOptions
{
    CompressionProfile profile = CompressionProfile::Development;
    // Zero selects the profile default. Non-zero values are validated against
    // the public range above before any source is read or output is created.
    int compressionLevel = 0;
    // A text header adds an INM envelope; the same byte-bounded IO and checksum apply.
    std::shared_ptr<const std::string> textHeader;
    PathWriteScope writeScope;
};

struct MemorySource
{
    const uint8_t *data = nullptr;
    size_t bytes = 0;
};

struct Source
{
    std::string path;
    // Memory is borrowed for the synchronous Write call, just like an open
    // source file. Its owner must keep the bytes unchanged until Write returns.
    std::variant<std::filesystem::path, MemorySource> data;
};

struct Entry
{
    std::string path;
    uint64_t offset = 0;
    uint64_t storedBytes = 0;
    uint64_t rawBytes = 0;
    Codec codec = Codec::Store;
};

struct Manifest
{
    std::shared_ptr<const std::string> textHeader;
    size_t HeaderStorageBytes() const
    {
        return textHeader ? sizeof(std::string) + textHeader->capacity() + 1 : 0;
    }
    uint64_t rawBytes = 0;
    uint64_t storedBytes = 0;
    uint64_t payloadBytes = 0;
    uint64_t archiveBytes = 0;
    std::array<uint8_t, 32> archiveHash{};
    std::vector<Entry> entries;
};

struct WriteEntry
{
    std::string path;
    uint64_t bytes = 0;
    bool store = false;
    // Optional immutable memory owned by the caller for the writer's lifetime.
    const uint8_t *residentData = nullptr;
};

enum class WriteState : uint8_t
{
    Working,
    NeedsInput,
    ReadyToPublish,
    Published,
    Cancelled,
};

struct WriteProgress
{
    uint64_t queuedBytes = 0;
    uint64_t consumedBytes = 0;
    uint64_t writtenBytes = 0;
    uint64_t hashedBytes = 0;
};

/// A single bounded input block, one compression context and one output file.
/// Append copies a bounded block; Advance performs at most one IO/codec step.
/// Only Cancel may overlap execution. Other methods require a finished step.
class Writer
{
  public:
    INFERNUX_FOUNDATION_API Writer(const std::filesystem::path &destination, std::vector<WriteEntry> entries,
                                   const WriteOptions &options = {}, size_t blockBytes = 64u * 1024u);
    INFERNUX_FOUNDATION_API ~Writer();
    INFERNUX_FOUNDATION_API WriteState GetState() const;
    INFERNUX_FOUNDATION_API WriteProgress GetProgress() const;
    INFERNUX_FOUNDATION_API IoWork DescribeWork(size_t maxBytes, bool publish = false) const;
    INFERNUX_FOUNDATION_API IoRecipe DescribeRemainingWork() const;
    const std::string &GetInputPath() const;
    INFERNUX_FOUNDATION_API size_t GetInputBytes() const;
    INFERNUX_FOUNDATION_API void Append(const uint8_t *data, size_t bytes);
    INFERNUX_FOUNDATION_API void Advance(size_t maxBytes);
    // True means publication has been prevented. An accepted IO job must
    // still advance once to close and remove its temporary output.
    INFERNUX_FOUNDATION_API bool Cancel();
    // One atomic replace attempt. Transient failure returns false and leaves
    // the writer ready; cancellation and the actual commit share one gate.
    INFERNUX_FOUNDATION_API bool Publish();
    Manifest GetManifest() const;
    INFERNUX_FOUNDATION_API size_t GetRetainedBytes() const;
    INFERNUX_FOUNDATION_API size_t GetReservationBytes() const;

  private:
    friend INFERNUX_FOUNDATION_API Manifest Write(const std::filesystem::path &, std::vector<Source>,
                                                  const WriteOptions &);
    friend class WriteTask;
    void Discard();
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

/// Return a validated public Zstandard level.
INFERNUX_FOUNDATION_API int ValidateCompressionLevel(int compressionLevel);

/// Resolve the effective level without changing the package format.
INFERNUX_FOUNDATION_API int ResolveCompressionLevel(const WriteOptions &options);

/// Additional serialized-header storage for a writer descriptor, without
/// allocating that header. Actual capacity replaces this estimate on creation.
INFERNUX_FOUNDATION_API size_t TextHeaderReservationBytes(size_t headerBytes);

/// Write one deterministic InxPack from file or contiguous memory sources.
INFERNUX_FOUNDATION_API Manifest Write(const std::filesystem::path &destination, std::vector<Source> sources,
                                       const WriteOptions &options = {});

/// Read and validate the current fixed header, TOC and package identity.
INFERNUX_FOUNDATION_API Manifest ReadManifest(const std::filesystem::path &path);

/// Validate and extract the package.  Empty allowedRoots means no root filter.
/// The destination and its descendants must be plain directories (no links).
/// Ancestors of the destination are resolved before pinning the output tree.
/// Each decoded entry replaces its output atomically; extraction of the whole
/// archive is not transactional, so earlier successful entries can remain on failure.
INFERNUX_FOUNDATION_API Manifest Extract(const std::filesystem::path &path, const std::filesystem::path &destination,
                                         const std::vector<std::string> &allowedRoots = {}, bool parallel = true);

/// Read one validated entry without exposing the container algorithm to Python.
INFERNUX_FOUNDATION_API std::vector<uint8_t> ReadEntry(const std::filesystem::path &path, const std::string &entryPath);

/// Copy immutable content to a cache. Concurrent equal writers reuse the
/// existing complete file without replacing a possibly loaded OS module.
INFERNUX_FOUNDATION_API std::shared_ptr<VerifiedFile> CopyVerifiedFile(const std::filesystem::path &source,
                                                                       const std::filesystem::path &destination,
                                                                       const std::string &expectedSha256);

enum class ReadState : uint8_t
{
    Opening,
    Ready,
    Checking,
    Reading,
    Materializing,
    Closed,
};

struct ReadProgress
{
    uint64_t readBytes = 0;
    uint64_t hashedBytes = 0;
    uint64_t codecInputBytes = 0;
    uint64_t decodedBytes = 0;
    // Raw output handed to the consumer, distinct from compressed input.
    uint64_t consumedBytes = 0;
    uint64_t writtenBytes = 0;
    // Output written into task-owned storage on the IO worker.
    uint64_t bufferedBytes = 0;
    uint64_t bufferWorkBytes = 0;
    uint64_t overlappedReads = 0;
    uint64_t pendingOverlappedReads = 0;
    uint64_t fileCompletionNs = 0;
};

struct EntryReadRange
{
    uint64_t offset = 0;
    uint8_t *destination = nullptr;
    size_t bytes = 0;
};

/// One opened file and bounded, resumable codec state. Construction performs
/// no file IO. Each step reads and produces at most maxBytes and blockBytes.
/// Directory parsing has a serialized metadata cap.
/// Observe ReservationBytes after each step before admitting another step.
/// One caller owns the cursor; the IO task supplies asynchronous serialization.
class ReadCursor
{
  public:
    INFERNUX_FOUNDATION_API ReadCursor(const std::filesystem::path &path, size_t metadataLimit,
                                       size_t blockBytes = 64u * 1024u);
    INFERNUX_FOUNDATION_API ~ReadCursor();
    INFERNUX_FOUNDATION_API void Advance(size_t maxBytes);
    INFERNUX_FOUNDATION_API void BeginChecksum();
    INFERNUX_FOUNDATION_API void BeginRead(const std::string &entryPath);
    /// Decode the next part of the entry into caller-owned storage.
    INFERNUX_FOUNDATION_API size_t ReadInto(uint8_t *destination, size_t bytes, size_t maxBytes);
    /// The async caller fills this range before finishing the same decode step.
    EntryReadRange PrepareReadInto(uint8_t *destination, size_t bytes, size_t maxBytes);
    size_t FinishReadInto(uint8_t *destination, size_t bytes, size_t maxBytes, size_t readBytes);
    FileInput &OpenedFile() const;
    /// Ensure an immutable file by content identity, reusing a verified target.
    /// Begin only describes the operation; Advance owns all filesystem work.
    INFERNUX_FOUNDATION_API void BeginMaterialize(const std::string &entryPath,
                                                  const std::filesystem::path &destination,
                                                  const std::string &expectedSha256);
    std::shared_ptr<VerifiedFile> TakeMaterializedFile();
    INFERNUX_FOUNDATION_API ReadState GetState() const;
    INFERNUX_FOUNDATION_API ReadProgress GetProgress() const;
    INFERNUX_FOUNDATION_API IoWork DescribeWork(size_t maxBytes) const;
    INFERNUX_FOUNDATION_API IoRecipe DescribeRemainingWork() const;
    INFERNUX_FOUNDATION_API IoRecipe DescribeReadWork(const std::string &entryPath) const;
    INFERNUX_FOUNDATION_API bool IsChecksumVerified() const;
    INFERNUX_FOUNDATION_API Manifest GetManifest() const;
    uint64_t GetEntryBytes(const std::string &entryPath) const;
    /// Abandon the current checksum/entry, retaining this open file/directory.
    /// An incomplete Open cannot become Ready through this operation.
    INFERNUX_FOUNDATION_API void DiscardOperation();
    INFERNUX_FOUNDATION_API size_t GetRetainedBytes() const;
    INFERNUX_FOUNDATION_API size_t GetReservationBytes() const;
    bool HasFileResources() const;
    bool HasMaterialization() const;
    INFERNUX_FOUNDATION_API void Close();

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

/// One open file and parsed directory for multi-entry reads and materialization.
class Reader
{
  public:
    INFERNUX_FOUNDATION_API explicit Reader(const std::filesystem::path &path);
    INFERNUX_FOUNDATION_API ~Reader();
    INFERNUX_FOUNDATION_API Manifest GetManifest() const;
    INFERNUX_FOUNDATION_API void VerifyChecksum();
    INFERNUX_FOUNDATION_API std::vector<uint8_t> Read(const std::string &entryPath);
    INFERNUX_FOUNDATION_API void ReadInto(const std::string &entryPath, uint8_t *destination, size_t bytes);
    // Materialize one entry through the existing bounded decoder and atomic
    // output, e.g. an OS-loadable program. Does not copy other entries.
    INFERNUX_FOUNDATION_API std::shared_ptr<VerifiedFile> ExtractEntry(const std::string &entryPath,
                                                                       const std::filesystem::path &destination,
                                                                       const std::string &expectedSha256);
    INFERNUX_FOUNDATION_API void Close();

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    mutable std::mutex m_mutex;
    Impl &State() const;
};

INFERNUX_FOUNDATION_API std::string HashToHex(const std::array<uint8_t, 32> &hash);
/// Hash one open snapshot in bounded blocks, leaving its cursor at EOF.
INFERNUX_FOUNDATION_API std::array<uint8_t, 32> HashFile(FileInput &input);
INFERNUX_FOUNDATION_API const char *CodecName(Codec codec) noexcept;

} // namespace infernux::inxpack
