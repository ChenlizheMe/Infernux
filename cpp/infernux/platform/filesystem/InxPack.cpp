#include "InxPack.h"

#include "AtomicFile.h"
#include "AtomicFileOutput.h"
#include "FileFingerprint.h"
#include "InxPath.h"
#include "VerifiedFile.h"

#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace infernux::inxpack
{
namespace
{

constexpr uint32_t kHeaderBytes = 128;
constexpr uint32_t kTocPrefixBytes = 24;
constexpr uint32_t kEntryBytes = 64;
constexpr size_t kHashChunkBytes = 64u * 1024u;
constexpr size_t kSynchronousWriteBlockBytes = 1024u * 1024u;
constexpr std::array<char, 4> kTocMagic = {'T', 'O', 'C', '0'};
constexpr std::string_view kInmMagic = "INM2\n";

#pragma pack(push, 1)
struct HeaderDisk
{
    char magic[8];
    uint32_t headerBytes;
    uint32_t flags;
    uint32_t entryBytes;
    uint32_t reserved0;
    uint64_t tocOffset;
    uint64_t tocBytes;
    uint64_t payloadOffset;
    uint64_t payloadBytes;
    uint64_t entryCount;
    uint64_t stringBytes;
    uint64_t rawBytes;
    uint8_t packageHash[32];
    uint8_t reserved[16];
};

struct TocPrefixDisk
{
    char magic[4];
    uint32_t flags;
    uint64_t entryCount;
    uint64_t stringBytes;
};

struct EntryDisk
{
    uint64_t pathOffset;
    uint32_t pathBytes;
    uint32_t flags;
    uint64_t payloadOffset;
    uint64_t storedBytes;
    uint64_t rawBytes;
    uint8_t codec;
    uint8_t codecReserved[7];
    uint32_t alignment;
    uint32_t reserved;
    uint8_t tailReserved[8];
};
#pragma pack(pop)

static_assert(sizeof(HeaderDisk) == kHeaderBytes);
static_assert(sizeof(TocPrefixDisk) == kTocPrefixBytes);
static_assert(sizeof(EntryDisk) == kEntryBytes);

constexpr uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    if (value > std::numeric_limits<uint64_t>::max() - (alignment - 1u))
        throw std::overflow_error("InxPack alignment overflow");
    return (value + alignment - 1u) / alignment * alignment;
}

uint64_t CheckedAdd(uint64_t left, uint64_t right, const char *what)
{
    if (left > std::numeric_limits<uint64_t>::max() - right)
        throw std::runtime_error(std::string("InxPack ") + what + " overflows");
    return left + right;
}

uint64_t CheckedMul(uint64_t left, uint64_t right, const char *what)
{
    if (right != 0 && left > std::numeric_limits<uint64_t>::max() / right)
        throw std::runtime_error(std::string("InxPack ") + what + " overflows");
    return left * right;
}

std::string LowerExtension(const std::filesystem::path &path)
{
    auto extension = FromFsPath(path.extension());
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return extension;
}

bool IsTransientReplaceError(const std::error_code &error) noexcept
{
#ifdef _WIN32
    return error.value() == ERROR_ACCESS_DENIED || error.value() == ERROR_SHARING_VIOLATION ||
           error.value() == ERROR_LOCK_VIOLATION;
#else
    return error.value() == EACCES || error.value() == EAGAIN || error.value() == EBUSY;
#endif
}

// Keep every directory in the output path pinned for the entire extraction.
// POSIX writes are relative to directory descriptors; Windows denies deletion
// of the opened ancestors so path-based writes cannot be redirected by rename.
// Reparse points/symlinks are never traversed, including in destination itself.
struct ExtractionDirectory
{
    std::filesystem::path path;
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
    ~ExtractionDirectory()
    {
        if (handle != INVALID_HANDLE_VALUE)
            ::CloseHandle(handle);
    }
#else
    int descriptor = -1;
    ~ExtractionDirectory()
    {
        if (descriptor >= 0)
            ::close(descriptor);
    }
#endif
};

class ExtractionTree
{
  public:
    explicit ExtractionTree(const std::filesystem::path &destination)
    {
        auto absolute = std::filesystem::absolute(destination.empty() ? std::filesystem::path(".") : destination)
                            .lexically_normal();
        while (absolute != absolute.root_path() && absolute.filename().empty())
            absolute = absolute.parent_path();
        // Resolve aliases selected by the caller above the destination (e.g.
        // /var on macOS), without following the destination entry itself.
        if (absolute != absolute.root_path())
            absolute = std::filesystem::weakly_canonical(absolute.parent_path()) / absolute.filename();
        root_ = OpenDirectory(nullptr, absolute.root_path());
        directories_.emplace(root_->path, root_);
        for (const auto &part : absolute.relative_path()) {
            if (!part.empty() && part != ".")
                root_ = Child(root_, part);
        }
    }

    std::shared_ptr<ExtractionDirectory> Root() const
    {
        return root_;
    }

    std::shared_ptr<ExtractionDirectory> Child(const std::shared_ptr<ExtractionDirectory> &parent,
                                               const std::filesystem::path &name)
    {
        const auto path = parent->path / name;
        const auto found = directories_.find(path);
        if (found != directories_.end())
            return found->second;
        auto directory = OpenDirectory(parent, name);
        directories_.emplace(path, directory);
        return directory;
    }

  private:
    static std::shared_ptr<ExtractionDirectory> OpenDirectory(const std::shared_ptr<ExtractionDirectory> &parent,
                                                              const std::filesystem::path &name)
    {
        auto directory = std::make_shared<ExtractionDirectory>();
        directory->path = parent ? parent->path / name : name;
#ifdef _WIN32
        const auto nativePath = NativeFilePath(directory->path);
        if (parent && !::CreateDirectoryW(nativePath.c_str(), nullptr) && ::GetLastError() != ERROR_ALREADY_EXISTS)
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "InxPack cannot create extraction directory");
        directory->handle =
            ::CreateFileW(nativePath.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                          FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
        if (directory->handle == INVALID_HANDLE_VALUE)
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "InxPack cannot pin extraction directory");
        FILE_ATTRIBUTE_TAG_INFO attributes{};
        if (!::GetFileInformationByHandleEx(directory->handle, FileAttributeTagInfo, &attributes, sizeof(attributes)))
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "InxPack cannot inspect extraction directory");
        if (!(attributes.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) ||
            (attributes.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT))
            throw std::runtime_error("InxPack extraction directory is not a plain directory: " +
                                     FromFsPath(directory->path));
#else
        if (parent && ::mkdirat(parent->descriptor, name.c_str(), 0777) != 0 && errno != EEXIST)
            throw std::system_error(errno, std::generic_category(), "InxPack cannot create extraction directory");
#if defined(__linux__) || defined(__ANDROID__)
        // Anchors are only used by *at operations, never for directory reads.
        // Android's /data ancestors are searchable but not readable by apps.
        constexpr int accessMode = O_PATH;
#elif defined(O_SEARCH) && !defined(__EMSCRIPTEN__)
        constexpr int accessMode = O_SEARCH;
#else
        constexpr int accessMode = O_RDONLY;
#endif
        // O_DIRECTORY is essential with O_PATH: O_NOFOLLOW alone can return
        // a descriptor for the symlink itself instead of rejecting it.
        constexpr int directoryFlags = accessMode | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC;
        directory->descriptor =
            parent ? ::openat(parent->descriptor, name.c_str(), directoryFlags) : ::open(name.c_str(), directoryFlags);
        if (directory->descriptor < 0)
            throw std::system_error(errno, std::generic_category(), "InxPack cannot pin extraction directory");
#endif
        return directory;
    }

    std::shared_ptr<ExtractionDirectory> root_;
    std::map<std::filesystem::path, std::shared_ptr<ExtractionDirectory>> directories_;
};

struct ExtractionTarget
{
    std::shared_ptr<ExtractionDirectory> directory;
    std::filesystem::path name;
};

bool CheckExtractionTarget(const ExtractionTarget &target)
{
#ifdef _WIN32
    const DWORD attributes = ::GetFileAttributesW(NativeFilePath(target.directory->path / target.name).c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES) {
        const auto error = ::GetLastError();
        if (error != ERROR_FILE_NOT_FOUND)
            throw std::system_error(static_cast<int>(error), std::system_category(), "InxPack cannot inspect entry");
        return false;
    }
    if (attributes & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY))
        throw std::runtime_error("InxPack extraction target is not a plain file");
#else
    struct stat status{};
    if (::fstatat(target.directory->descriptor, target.name.c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0) {
        if (errno != ENOENT)
            throw std::system_error(errno, std::generic_category(), "InxPack cannot inspect entry");
        return false;
    }
    if (!S_ISREG(status.st_mode))
        throw std::runtime_error("InxPack extraction target is not a plain file");
#endif
    return true;
}

// Stream into an exclusively created sibling, then publish the complete entry.
// Besides preserving an old file on decode failure, replacement avoids writing
// through an existing hard link to a file outside the extraction tree.
class ExtractedFile
{
  public:
    explicit ExtractedFile(const ExtractionTarget &target, const PathWriteScope &parent = {}) : target_(target)
    {
        const auto destination = FromFsPath(target_.directory->path / target_.name);
        writeScope_ = PathWriteScope::BeginWrite({destination}, parent.Covers(destination) ? parent : PathWriteScope{});
        CheckExtractionTarget(target_);
        for (unsigned attempt = 0; attempt < 1024; ++attempt) {
            temporary_ = MakeAtomicTemporaryPath(target_.name);
            temporaryWriteScope_ = PathWriteScope::BeginWrite({FromFsPath(target_.directory->path / temporary_)});
#ifdef _WIN32
            handle_ =
                ::CreateFileW(NativeFilePath(target_.directory->path / temporary_).c_str(), GENERIC_WRITE | DELETE, 0,
                              nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
            if (handle_ != INVALID_HANDLE_VALUE)
                return;
            const auto error = ::GetLastError();
            if (error != ERROR_FILE_EXISTS && error != ERROR_ALREADY_EXISTS)
                throw std::system_error(static_cast<int>(error), std::system_category(), "InxPack cannot create entry");
#else
            descriptor_ = ::openat(target_.directory->descriptor, temporary_.c_str(),
                                   O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0666);
            if (descriptor_ >= 0)
                return;
            if (errno != EEXIST)
                throw std::system_error(errno, std::generic_category(), "InxPack cannot create entry");
#endif
        }
        throw std::runtime_error("InxPack exhausted unique extraction names");
    }

    ~ExtractedFile()
    {
        Close();
        if (!published_) {
#ifdef _WIN32
            ::DeleteFileW(NativeFilePath(target_.directory->path / temporary_).c_str());
#else
            ::unlinkat(target_.directory->descriptor, temporary_.c_str(), 0);
#endif
        }
    }

    void Write(const char *data, size_t size)
    {
        while (size) {
#ifdef _WIN32
            DWORD count = 0;
            if (!::WriteFile(handle_, data, static_cast<DWORD>(std::min<size_t>(size, MAXDWORD)), &count, nullptr))
                throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                        "InxPack cannot write entry");
#else
            const auto count = ::write(descriptor_, data, size);
            if (count < 0) {
                if (errno == EINTR)
                    continue;
                throw std::system_error(errno, std::generic_category(), "InxPack cannot write entry");
            }
#endif
            if (count == 0)
                throw std::runtime_error("InxPack entry write made no progress");
            data += count;
            size -= count;
        }
    }

#ifdef INX_PLATFORM_WINDOWS
    std::pair<uint64_t, uint64_t> Identity() const
    {
        return FileIdentity(handle_);
    }
#elif defined(INX_PLATFORM_LINUX)
    std::shared_ptr<FileInput> RetainInput() const
    {
        return std::make_shared<FileInput>(NativeDescriptorPath(descriptor_));
    }
#endif

    bool Publish(bool replace = true)
    {
        CheckExtractionTarget(target_);
#ifdef _WIN32
        if (!::FlushFileBuffers(handle_))
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "InxPack cannot flush entry");
        if (!replace) {
            // A cache name may be opened immediately by another loader. Close
            // the complete private temp before making that name visible.
            if (!::CloseHandle(std::exchange(handle_, INVALID_HANDLE_VALUE)))
                throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                        "InxPack cannot close immutable entry");
            std::error_code error;
            const auto result = PublishFileIfAbsent(target_.directory->path / temporary_,
                                                    target_.directory->path / target_.name, error);
            if (!result)
                throw std::system_error(error, "InxPack cannot publish immutable entry");
            if (*result == FilePublication::AlreadyExists)
                return false;
            published_ = true;
            return true;
        }
        // Keep the exclusive verified temp handle through the common rename.
        std::error_code error;
        if (!detail::ReplaceOpenFileAtomically(handle_, target_.directory->path / target_.name, error))
            throw std::system_error(error, "InxPack cannot publish entry");
#else
        // write() is unbuffered; report close errors before publishing the file.
        const int descriptor = std::exchange(descriptor_, -1);
        if (::close(descriptor) != 0)
            throw std::system_error(errno, std::generic_category(), "InxPack cannot close entry");
        if (replace) {
            if (::renameat(target_.directory->descriptor, temporary_.c_str(), target_.directory->descriptor,
                           target_.name.c_str()) != 0)
                throw std::system_error(errno, std::generic_category(), "InxPack cannot publish entry");
        } else {
            std::error_code error;
            const auto result =
                detail::PublishFileIfAbsentAt(target_.directory->descriptor, temporary_, target_.name, error);
            if (!result)
                throw std::system_error(error, "InxPack cannot publish immutable entry");
            if (*result == FilePublication::AlreadyExists)
                return false;
        }
#endif
        published_ = true;
        return true;
    }

  private:
    void Close() noexcept
    {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE)
            ::CloseHandle(std::exchange(handle_, INVALID_HANDLE_VALUE));
#else
        if (descriptor_ >= 0)
            ::close(std::exchange(descriptor_, -1));
#endif
    }

    PathWriteScope writeScope_;
    PathWriteScope temporaryWriteScope_;
    ExtractionTarget target_;
    std::filesystem::path temporary_;
    bool published_ = false;
#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int descriptor_ = -1;
#endif
};

class Sha256
{
  public:
    Sha256()
    {
        state_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                  0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    }

    void Update(const uint8_t *data, size_t size)
    {
        while (size != 0) {
            const size_t available = sizeof(buffer_) - bufferSize_;
            const size_t count = std::min(available, size);
            std::memcpy(buffer_.data() + bufferSize_, data, count);
            bufferSize_ += count;
            data += count;
            size -= count;
            bitCount_ += static_cast<uint64_t>(count) * 8u;
            if (bufferSize_ == sizeof(buffer_)) {
                Transform(buffer_.data());
                bufferSize_ = 0;
            }
        }
    }

    std::array<uint8_t, 32> Final()
    {
        const uint64_t originalBits = bitCount_;
        const uint8_t one = 0x80;
        Update(&one, 1);
        const uint8_t zero = 0;
        while (bufferSize_ != 56)
            Update(&zero, 1);
        std::array<uint8_t, 8> length{};
        for (size_t index = 0; index < length.size(); ++index)
            length[length.size() - index - 1] = static_cast<uint8_t>(originalBits >> (index * 8));
        Update(length.data(), length.size());

        std::array<uint8_t, 32> result{};
        for (size_t index = 0; index < state_.size(); ++index) {
            for (size_t byte = 0; byte < sizeof(uint32_t); ++byte)
                result[index * 4 + byte] = static_cast<uint8_t>(state_[index] >> (24 - byte * 8));
        }
        return result;
    }

  private:
    static constexpr std::array<uint32_t, 64> kRoundConstants = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

    static uint32_t RotateRight(uint32_t value, uint32_t count)
    {
        return (value >> count) | (value << (32u - count));
    }

    static uint32_t Ch(uint32_t x, uint32_t y, uint32_t z)
    {
        return (x & y) ^ (~x & z);
    }

    static uint32_t Maj(uint32_t x, uint32_t y, uint32_t z)
    {
        return (x & y) ^ (x & z) ^ (y & z);
    }

    void Transform(const uint8_t *block)
    {
        std::array<uint32_t, 64> words{};
        for (size_t index = 0; index < 16; ++index)
            words[index] =
                (static_cast<uint32_t>(block[index * 4]) << 24) | (static_cast<uint32_t>(block[index * 4 + 1]) << 16) |
                (static_cast<uint32_t>(block[index * 4 + 2]) << 8) | static_cast<uint32_t>(block[index * 4 + 3]);
        for (size_t index = 16; index < words.size(); ++index) {
            const uint32_t s0 =
                RotateRight(words[index - 15], 7) ^ RotateRight(words[index - 15], 18) ^ (words[index - 15] >> 3);
            const uint32_t s1 =
                RotateRight(words[index - 2], 17) ^ RotateRight(words[index - 2], 19) ^ (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }

        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (size_t index = 0; index < words.size(); ++index) {
            const uint32_t sum1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
            const uint32_t choice = Ch(e, f, g);
            const uint32_t temporary1 = h + sum1 + choice + kRoundConstants[index] + words[index];
            const uint32_t sum0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
            const uint32_t majority = Maj(a, b, c);
            const uint32_t temporary2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<uint32_t, 8> state_{};
    std::array<uint8_t, 64> buffer_{};
    size_t bufferSize_ = 0;
    uint64_t bitCount_ = 0;
};

class ArchiveChecksum
{
  public:
    explicit ArchiveChecksum(bool model)
    {
        if (model)
            checksum_.emplace<detail::FileFingerprint>();
    }
    void Update(const uint8_t *data, size_t size)
    {
        if (auto *model = std::get_if<detail::FileFingerprint>(&checksum_))
            model->Append({reinterpret_cast<const char *>(data), size});
        else
            std::get<Sha256>(checksum_).Update(data, size);
    }
    std::array<uint8_t, 32> Final()
    {
        if (const auto *model = std::get_if<detail::FileFingerprint>(&checksum_)) {
            XXH64_canonical_t digest;
            XXH64_canonicalFromHash(&digest, model->Value());
            std::array<uint8_t, 32> result{};
            std::copy(std::begin(digest.digest), std::end(digest.digest), result.begin());
            return result;
        }
        return std::get<Sha256>(checksum_).Final();
    }

  private:
    std::variant<Sha256, detail::FileFingerprint> checksum_;
};

std::array<uint8_t, 32> CopyHash(const uint8_t *data)
{
    std::array<uint8_t, 32> result{};
    std::copy(data, data + result.size(), result.begin());
    return result;
}

// Independent positions over one pinned input, including parallel extraction.
class InputCursor
{
  public:
    explicit InputCursor(FileInput &file) : file_(file)
    {
    }
    void Seek(uint64_t offset)
    {
        position_ = offset;
    }
    uint64_t Size()
    {
        return file_.Size() - origin_;
    }
    FileInput &File()
    {
        return file_;
    }
    uint64_t Origin() const
    {
        return origin_;
    }
    void SetOrigin(uint64_t origin)
    {
        origin_ = origin;
        position_ = 0;
    }
    void Read(void *destination, size_t bytes)
    {
        file_.ReadAt(origin_ + position_, destination, bytes);
        position_ += bytes;
    }

  private:
    FileInput &file_;
    uint64_t position_ = 0, origin_ = 0;
};

class RangeDigest
{
  public:
    RangeDigest(InputCursor &input, uint64_t offset, uint64_t size, size_t blockBytes,
                std::optional<std::pair<uint64_t, uint64_t>> zeroRange = {})
        : RangeDigest(input.File(), offset, size, blockBytes, zeroRange)
    {
        origin_ = input.Origin();
    }
    RangeDigest(FileInput &input, uint64_t offset, uint64_t size, size_t blockBytes,
                std::optional<std::pair<uint64_t, uint64_t>> zeroRange = {}, bool model = false)
        : input_(input), cursor_(offset), remaining_(size), blockBytes_(blockBytes), zeroRange_(zeroRange),
          digest_(model)
    {
    }

    size_t Advance(size_t maximum)
    {
        const auto count = static_cast<size_t>(std::min<uint64_t>(remaining_, std::min(blockBytes_, maximum)));
        if (buffer_.capacity() == 0)
            buffer_.reserve(static_cast<size_t>(std::min<uint64_t>(blockBytes_, remaining_)));
        buffer_.resize(count);
        if (count) {
            input_.ReadAt(origin_ + cursor_, buffer_.data(), count);
            if (zeroRange_) {
                const auto [start, size] = *zeroRange_;
                const auto end = start + size;
                if (cursor_ < end && cursor_ + count > start) {
                    std::fill(buffer_.begin() + static_cast<size_t>(std::max(cursor_, start) - cursor_),
                              buffer_.begin() + static_cast<size_t>(std::min(cursor_ + count, end) - cursor_), 0);
                }
            }
            digest_.Update(buffer_.data(), count);
            cursor_ += count;
            remaining_ -= count;
        }
        return count;
    }

    bool Done() const
    {
        return remaining_ == 0;
    }
    uint64_t RemainingBytes() const
    {
        return remaining_;
    }
    std::array<uint8_t, 32> Finish()
    {
        return digest_.Final();
    }
    size_t RetainedBytes() const
    {
        return sizeof(*this) + buffer_.capacity();
    }
    size_t ReservationBytes() const
    {
        return sizeof(*this) + std::max(buffer_.capacity(), blockBytes_);
    }

  private:
    FileInput &input_;
    uint64_t cursor_, remaining_;
    uint64_t origin_ = 0;
    size_t blockBytes_;
    std::optional<std::pair<uint64_t, uint64_t>> zeroRange_;
    ArchiveChecksum digest_;
    std::vector<uint8_t> buffer_;
};

std::string ToHex(const uint8_t *data, size_t size)
{
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(size * 2);
    for (size_t index = 0; index < size; ++index) {
        result.push_back(digits[data[index] >> 4]);
        result.push_back(digits[data[index] & 0x0f]);
    }
    return result;
}

std::string NormalizePath(const std::string &path)
{
    std::string normalized = NormalizePortablePath(path);
    if (normalized.empty() || normalized.front() == '/' || normalized.find(':') != std::string::npos)
        throw std::invalid_argument("InxPack path must be portable and relative: " + path);
    size_t begin = 0;
    while (begin <= normalized.size()) {
        const size_t end = normalized.find('/', begin);
        const auto part =
            std::string_view(normalized).substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (part.empty() || part == "." || part == ".." || part.back() == '.' || part.back() == ' ' ||
            part.find_first_of("<>\"|?*") != std::string::npos ||
            std::any_of(part.begin(), part.end(), [](unsigned char value) { return value < 32; }))
            throw std::invalid_argument("InxPack path contains an unsafe component: " + path);
        std::string stem(part.substr(0, part.find('.')));
        while (!stem.empty() && stem.back() == ' ')
            stem.pop_back();
        std::transform(stem.begin(), stem.end(), stem.begin(),
                       [](unsigned char value) { return static_cast<char>(std::toupper(value)); });
        const bool devicePrefix = stem.compare(0, 3, "COM") == 0 || stem.compare(0, 3, "LPT") == 0;
        const bool numberedDevice = stem.size() == 4 && devicePrefix && stem[3] >= '1' && stem[3] <= '9';
        const bool superscriptDevice = devicePrefix && (stem.substr(3) == "\xc2\xb9" || stem.substr(3) == "\xc2\xb2" ||
                                                        stem.substr(3) == "\xc2\xb3");
        if (stem == "CON" || stem == "CONIN$" || stem == "CONOUT$" || stem == "PRN" || stem == "AUX" || stem == "NUL" ||
            numberedDevice || superscriptDevice)
            throw std::invalid_argument("InxPack path contains a reserved device name: " + path);
        if (end == std::string::npos)
            break;
        begin = end + 1;
    }
    return normalized;
}

class SourceInput
{
  public:
    explicit SourceInput(const std::filesystem::path &path) : m_file(path), m_size(m_file.Size())
    {
    }
    uint64_t Size() const
    {
        return m_size;
    }
    const uint8_t *ReadView(uint8_t *workspace, size_t size)
    {
        m_file.Read(workspace, size);
        m_position += size;
        return workspace;
    }
    void Finish()
    {
        if (m_position != m_size || m_file.Size() != m_size)
            throw std::runtime_error("InxPack source changed while packaging");
    }

  private:
    FileInput m_file;
    uint64_t m_size = 0, m_position = 0;
};

class EntryDecoder
{
  public:
    EntryDecoder(InputCursor &input, uint64_t payloadOffset, const Entry &entry, size_t blockBytes)
        : input_(input), entry_(entry), offset_(CheckedAdd(payloadOffset, entry.offset, "entry offset")),
          remaining_(entry.storedBytes), blockBytes_(blockBytes), pending_(entry.codec == Codec::Zstandard ? 1 : 0)
    {
    }

    IoWork DescribeWork(size_t maximum) const
    {
        const auto limit = std::min(maximum, blockBytes_);
        if (entry_.codec == Codec::Store)
            return IoByteWork("read.store", remaining_, limit);
        if (!workspaceBytes_)
            return {"read.zstd_header",
                    static_cast<size_t>(
                        std::min<uint64_t>(remaining_, std::min(limit, ZSTD_FRAMEHEADERSIZE_MAX - stored_.size()))),
                    {},
                    0};
        return {!decoder_                                        ? "read.zstd_allocate_decode"
                : inputPosition_ == stored_.size() && remaining_ ? "read.zstd_read_decode"
                                                                 : "read.zstd_decode",
                limit,
                {},
                !decoder_ ? workspaceBytes_ : 0};
    }

    EntryReadRange PrepareRead(size_t maximum, uint8_t *destination, size_t bytes)
    {
        if (!maximum)
            throw std::invalid_argument("InxPack decode step requires a positive byte budget");
        const auto offset = input_.Origin() + offset_;
        if (entry_.codec == Codec::Store) {
            const auto count = static_cast<size_t>(std::min<uint64_t>(remaining_, std::min(maximum, bytes)));
            if (!count)
                throw std::runtime_error("InxPack decoded size mismatch: " + entry_.path);
            return {offset, destination, count};
        }
        const auto limit = std::min(maximum, blockBytes_);
        if (stored_.capacity() == 0)
            stored_.reserve(InputCapacity());
        if (!workspaceBytes_) {
            const auto count = static_cast<size_t>(
                std::min<uint64_t>(remaining_, std::min(limit, ZSTD_FRAMEHEADERSIZE_MAX - stored_.size())));
            const auto begin = stored_.size();
            stored_.resize(begin + count);
            return {offset, stored_.data() + begin, count};
        }
        if (inputPosition_ == stored_.size() && remaining_) {
            const auto count = static_cast<size_t>(std::min<uint64_t>(remaining_, limit));
            stored_.resize(count);
            inputPosition_ = 0;
            return {offset, stored_.data(), count};
        }
        return {};
    }

    void FinishRead(size_t maximum, uint8_t *destination, size_t bytes, size_t readBytes)
    {
        outputBytes_ = 0;
        offset_ += readBytes;
        remaining_ -= readBytes;
        if (entry_.codec == Codec::Store) {
            consumed_ += readBytes;
            Output(readBytes);
            return;
        }
        // Learn the real frame workspace before allocating it on a later step.
        if (!workspaceBytes_) {
            const auto needed = ZSTD_estimateDStreamSize_fromFrame(stored_.data(), stored_.size());
            if (ZSTD_isError(needed)) {
                ZSTD_frameHeader frame{};
                const auto header = ZSTD_getFrameHeader(&frame, stored_.data(), stored_.size());
                if (!ZSTD_isError(header) && header && remaining_)
                    return;
                throw std::runtime_error("InxPack zstd frame header is invalid: " + entry_.path);
            }
            workspaceBytes_ = needed;
            return;
        }
        if (!decoder_) {
            workspace_.resize((workspaceBytes_ + sizeof(uint64_t) - 1) / sizeof(uint64_t));
            decoder_ = ZSTD_initStaticDStream(workspace_.data(), workspace_.size() * sizeof(uint64_t));
            if (!decoder_)
                throw std::runtime_error("InxPack cannot initialize decompression");
        }
        ZSTD_inBuffer in{stored_.data(), stored_.size(), inputPosition_};
        ZSTD_outBuffer out{destination, std::min({maximum, blockBytes_, bytes}), 0};
        pending_ = ZSTD_decompressStream(decoder_, &out, &in);
        if (ZSTD_isError(pending_))
            throw std::runtime_error("InxPack zstd decompression failed: " + entry_.path);
        const auto consumed = in.pos - inputPosition_;
        consumed_ += consumed;
        inputPosition_ = in.pos;
        Output(out.pos);
        if (!consumed && !out.pos && !Done())
            throw std::runtime_error("InxPack truncated or mismatched entry: " + entry_.path);
    }

    bool Done() const
    {
        return remaining_ == 0 && (entry_.codec == Codec::Store || inputPosition_ == stored_.size()) && pending_ == 0 &&
               written_ == entry_.rawBytes;
    }
    size_t OutputBytes() const
    {
        return outputBytes_;
    }
    uint64_t ConsumedInputBytes() const
    {
        return consumed_;
    }
    uint64_t DecodedBytes() const
    {
        return written_;
    }
    const Entry &SourceEntry() const
    {
        return entry_;
    }
    size_t RetainedBytes() const
    {
        return sizeof(*this) + stored_.capacity() + workspace_.capacity() * sizeof(uint64_t);
    }
    size_t ReservationBytes() const
    {
        return sizeof(*this) + std::max(stored_.capacity(), InputCapacity()) +
               std::max(workspace_.capacity() * sizeof(uint64_t),
                        workspaceBytes_ ? workspaceBytes_ + sizeof(uint64_t) - 1 : size_t{0});
    }

  private:
    size_t InputCapacity() const
    {
        return entry_.codec == Codec::Store
                   ? 0
                   : static_cast<size_t>(std::min<uint64_t>(entry_.storedBytes,
                                                            std::max(blockBytes_, size_t{ZSTD_FRAMEHEADERSIZE_MAX})));
    }
    void Output(size_t count)
    {
        if (count > entry_.rawBytes - written_)
            throw std::runtime_error("InxPack decoded size mismatch: " + entry_.path);
        written_ += count;
        outputBytes_ = count;
        if (!remaining_ && (entry_.codec == Codec::Store || inputPosition_ == stored_.size()) && pending_ == 0 &&
            written_ != entry_.rawBytes)
            throw std::runtime_error("InxPack truncated or mismatched entry: " + entry_.path);
    }
    InputCursor &input_;
    const Entry &entry_;
    uint64_t offset_, remaining_, written_ = 0, consumed_ = 0;
    size_t blockBytes_, pending_, inputPosition_ = 0, outputBytes_ = 0, workspaceBytes_ = 0;
    std::vector<uint8_t> stored_;
    std::vector<uint64_t> workspace_;
    ZSTD_DStream *decoder_ = nullptr;
};

template <class Write> void DecodeEntry(InputCursor &input, uint64_t payloadOffset, const Entry &entry, Write &&write)
{
    EntryDecoder decoder(input, payloadOffset, entry, kHashChunkBytes);
    std::array<uint8_t, kHashChunkBytes> block;
    while (!decoder.Done()) {
        const auto range = decoder.PrepareRead(block.size(), block.data(), block.size());
        if (range.bytes)
            input.File().ReadAt(range.offset, range.destination, range.bytes);
        decoder.FinishRead(block.size(), block.data(), block.size(), range.bytes);
        if (decoder.OutputBytes())
            write(block.data(), decoder.OutputBytes());
    }
}

bool IsAllowedRoot(const std::string &path, const std::vector<std::string> &allowedRoots)
{
    if (allowedRoots.empty())
        return true;
    const size_t separator = path.find('/');
    const std::string root = path.substr(0, separator);
    return std::find(allowedRoots.begin(), allowedRoots.end(), root) != allowedRoots.end();
}

// One sink for both cold extraction/copy and the asynchronous read cursor.
// Filesystem handles and directory traversal are platform bookkeeping, not a
// numerical/codec heap or OS residency guarantee. Only the bounded digest
// block is an additional byte payload; decoder output is borrowed directly.
class Materialization
{
  public:
    Materialization(std::filesystem::path destination, uint64_t bytes, std::string expectedSha256, size_t blockBytes)
        : destination_(ToFsPath(NormalizeFilesystemPathLexically(FromFsPath(destination)))),
          writeScope_(PathWriteScope::BeginWrite({FromFsPath(destination_)})), bytes_(bytes),
          expected_(std::move(expectedSha256)), blockBytes_(blockBytes)
    {
    }

    bool NeedsInput() const
    {
        return phase_ == Phase::Input;
    }
    bool Done() const
    {
        return phase_ == Phase::Complete;
    }

    IoWork DescribeWork(size_t maximum) const
    {
        switch (phase_) {
        case Phase::Opening:
            return {"materialize.open", 0, {}, bytes_};
        case Phase::Publishing:
            return {"materialize.publish", 0, {}, bytes_};
        case Phase::Verifying: {
            const auto left = verification_->RemainingBytes();
            return IoByteWork(left <= maximum ? "materialize.verify_finish" : "materialize.verify", left, maximum);
        }
        case Phase::Input:
        case Phase::Complete:
            return {};
        }
        throw std::logic_error("unknown materialization phase");
    }

    IoRecipe DescribeRemainingWork(const Entry &entry, size_t index, const EntryDecoder *decoder,
                                   bool pendingOutput) const
    {
        IoRecipe result;
        auto add = [&](std::string_view kind, std::string_view operation, uint64_t input,
                       std::optional<uint64_t> output = {}, uint64_t parameter = 0) {
            result.push_back({{{kind, operation, index, parameter, input, output, {}, entry.rawBytes}}, 1});
        };
        if (phase_ == Phase::Complete)
            return result;
        if (phase_ == Phase::Verifying) {
            result.push_back({{{"bytes",
                                "materialize.verify",
                                index,
                                0,
                                verification_->RemainingBytes(),
                                {},
                                "materialize.verify_finish"}},
                              1});
            return result;
        }
        if (phase_ == Phase::Opening)
            add("atom", "materialize.open", 0, {}, bytes_);
        if (phase_ == Phase::Opening || phase_ == Phase::Input) {
            if (!decoder || !decoder->Done() || pendingOutput)
                add("worker", "materialize.decode_write",
                    entry.storedBytes - (decoder ? decoder->ConsumedInputBytes() : 0), bytes_ - consumed_,
                    static_cast<uint64_t>(entry.codec));
            add("atom", "materialize.hash_finish", 0);
        }
        add("atom", "materialize.publish", 0, {}, bytes_);
        // A cache hit skips decoding/publication. A miss can lose publication
        // to one immutable winner; that branch verifies at most one full file.
        // This conditional upper extent disappears as soon as the real cursor
        // knows the branch. It is not an extra unconditional physical read.
        result.push_back(
            {{{"upper_bytes", "materialize.verify", index, 0, bytes_, {}, "materialize.verify_finish"}}, 1});
        return result;
    }

    void Advance(size_t maximum, ReadProgress &progress)
    {
        switch (phase_) {
        case Phase::Opening:
            tree_ = std::make_shared<ExtractionTree>(destination_.parent_path());
            destination_ = tree_->Root()->path / destination_.filename();
            if (!BeginVerification()) {
                output_ = std::make_unique<ExtractedFile>(ExtractionTarget{tree_->Root(), destination_.filename()},
                                                          writeScope_);
                phase_ = Phase::Input;
            }
            return;
        case Phase::Publishing: {
#ifdef INX_PLATFORM_WINDOWS
            const auto identity = output_->Identity();
#elif defined(INX_PLATFORM_LINUX)
            auto input = output_->RetainInput();
#endif
            if (output_->Publish(false)) {
#ifdef INX_PLATFORM_LINUX
                existing_ = std::move(input);
#else
                existing_ = std::make_shared<FileInput>(destination_);
#ifdef INX_PLATFORM_WINDOWS
                if (existing_->Identity() != identity)
                    throw std::runtime_error("InxPack published path no longer names its completed file");
#endif
#endif
                Complete();
            } else {
                output_.reset(); // Only this attempt's private sibling, never the cache winner.
                if (!BeginVerification())
                    throw std::runtime_error("InxPack immutable cache disappeared before verification");
            }
            return;
        }
        case Phase::Verifying: {
            const auto count = verification_->Advance(maximum);
            progress.readBytes += count;
            progress.hashedBytes += count;
            if (verification_->Done()) {
                if (HashToHex(verification_->Finish()) != expected_)
                    throw std::runtime_error("InxPack existing immutable entry differs from its content identity");
                Complete();
            }
            return;
        }
        case Phase::Input:
            throw std::logic_error("materialization requires its next source block");
        case Phase::Complete:
            return;
        }
    }

    void Write(const uint8_t *data, size_t bytes, ReadProgress &progress)
    {
        if (!NeedsInput() || bytes > bytes_ - consumed_)
            throw std::logic_error("materialization source exceeds its declared content");
        digest_.Update(data, bytes);
        output_->Write(reinterpret_cast<const char *>(data), bytes);
        consumed_ += bytes;
        progress.hashedBytes += bytes;
        progress.writtenBytes += bytes;
        progress.consumedBytes += bytes;
    }

    void FinishInput()
    {
        if (!NeedsInput() || consumed_ != bytes_)
            throw std::logic_error("materialization source is incomplete");
        if (HashToHex(digest_.Final()) != expected_)
            throw std::runtime_error("InxPack materialized entry differs from its content identity");
        phase_ = Phase::Publishing; // Flush/atomic publication are their own indivisible IO step.
    }

    size_t RetainedBytes() const
    {
        return BaseBytes() + (existing_ ? sizeof(FileInput) : 0) + (input_ ? sizeof(InputCursor) : 0) +
               (output_ ? sizeof(ExtractedFile) : 0) + (verification_ ? verification_->RetainedBytes() : 0);
    }
    size_t ReservationBytes() const
    {
        return BaseBytes() + sizeof(FileInput) + sizeof(InputCursor) + sizeof(ExtractedFile) +
               (verification_ ? verification_->ReservationBytes() : sizeof(RangeDigest) + blockBytes_);
    }

  private:
    enum class Phase
    {
        Opening,
        Input,
        Publishing,
        Verifying,
        Complete
    };
    size_t BaseBytes() const
    {
        return sizeof(*this) + destination_.native().capacity() * sizeof(std::filesystem::path::value_type) +
               expected_.capacity() + 1;
    }
    bool BeginVerification()
    {
        if (!CheckExtractionTarget({tree_->Root(), destination_.filename()}))
            return false;
        existing_ = std::make_shared<FileInput>(destination_);
        if (existing_->Size() != bytes_)
            throw std::runtime_error("InxPack existing immutable entry has a different size");
        input_ = std::make_unique<InputCursor>(*existing_);
        verification_ = std::make_unique<RangeDigest>(*input_, 0, bytes_, blockBytes_);
        phase_ = Phase::Verifying;
        return true;
    }
    void Complete()
    {
        verification_.reset();
        input_.reset();
        output_.reset();
        phase_ = Phase::Complete;
        writeScope_.Close();
    }

  public:
    uint64_t Bytes() const
    {
        return bytes_;
    }
    std::filesystem::path TakePath()
    {
        return std::move(destination_);
    }
    std::string TakeSha256()
    {
        return std::move(expected_);
    }
    std::shared_ptr<FileInput> TakeInput()
    {
        return std::move(existing_);
    }
    std::shared_ptr<void> TakeAnchors()
    {
        return std::move(tree_);
    }

  private:
    std::filesystem::path destination_;
    PathWriteScope writeScope_;
    uint64_t bytes_, consumed_ = 0;
    std::string expected_;
    size_t blockBytes_;
    Phase phase_ = Phase::Opening;
    Sha256 digest_;
    std::shared_ptr<ExtractionTree> tree_;
    std::unique_ptr<ExtractedFile> output_;
    std::shared_ptr<FileInput> existing_;
    std::unique_ptr<InputCursor> input_;
    std::unique_ptr<RangeDigest> verification_;
};

template <class Emit>
Materialization Materialize(const std::filesystem::path &destination, uint64_t bytes, const std::string &expectedSha256,
                            Emit &&emit)
{
    Materialization sink(destination, bytes, expectedSha256, kHashChunkBytes);
    ReadProgress progress;
    while (!sink.NeedsInput() && !sink.Done())
        sink.Advance(kHashChunkBytes, progress);
    if (sink.NeedsInput()) {
        emit([&](const uint8_t *data, size_t size) { sink.Write(data, size, progress); });
        sink.FinishInput();
    }
    while (!sink.Done())
        sink.Advance(kHashChunkBytes, progress);
    return sink;
}

void ValidateHeader(const HeaderDisk &header, uint64_t fileBytes, const std::filesystem::path &path)
{
    if (!std::equal(std::begin(header.magic), std::end(header.magic), kMagic.begin()))
        throw std::runtime_error("unsupported InxPack magic: " + FromFsPath(path));
    if (header.headerBytes != sizeof(HeaderDisk) || header.entryBytes != sizeof(EntryDisk))
        throw std::runtime_error("unsupported InxPack record layout: " + FromFsPath(path));
    if (header.tocOffset != sizeof(HeaderDisk) || header.tocOffset % kAlignment != 0 ||
        header.payloadOffset % kAlignment != 0 || header.entryCount > std::numeric_limits<size_t>::max())
        throw std::runtime_error("invalid InxPack offsets or alignment: " + FromFsPath(path));
    const auto tocEnd = CheckedAdd(header.tocOffset, header.tocBytes, "TOC range");
    const auto payloadEnd = CheckedAdd(header.payloadOffset, header.payloadBytes, "payload range");
    if (header.payloadOffset < tocEnd)
        throw std::runtime_error("invalid InxPack TOC/payload ordering: " + FromFsPath(path));
    if (fileBytes < payloadEnd)
        throw std::runtime_error("truncated InxPack payload: " + FromFsPath(path));
    if (header.tocBytes > std::numeric_limits<size_t>::max())
        throw std::runtime_error("InxPack TOC is too large: " + FromFsPath(path));
}

struct DirectoryLayout
{
    uint64_t entries = 0, stringsOffset = 0, pathBytes = 0, largestPath = 0;
};

DirectoryLayout InspectDirectory(const HeaderDisk &header, const std::vector<uint8_t> &toc,
                                 const std::filesystem::path &path)
{
    if (toc.size() < sizeof(TocPrefixDisk))
        throw std::runtime_error("InxPack TOC is truncated: " + FromFsPath(path));
    TocPrefixDisk prefix{};
    std::memcpy(&prefix, toc.data(), sizeof(prefix));
    if (!std::equal(std::begin(prefix.magic), std::end(prefix.magic), kTocMagic.begin()) ||
        prefix.entryCount != header.entryCount || prefix.stringBytes != header.stringBytes)
        throw std::runtime_error("InxPack TOC header is invalid: " + FromFsPath(path));
    if (prefix.entryCount > (toc.size() - sizeof(TocPrefixDisk)) / sizeof(EntryDisk))
        throw std::runtime_error("InxPack TOC entry table overflows: " + FromFsPath(path));
    DirectoryLayout layout;
    layout.entries = prefix.entryCount;
    layout.stringsOffset =
        CheckedAdd(sizeof(TocPrefixDisk), CheckedMul(prefix.entryCount, sizeof(EntryDisk), "TOC entry table"),
                   "TOC string offset");
    if (layout.stringsOffset > toc.size() || prefix.stringBytes > toc.size() - layout.stringsOffset)
        throw std::runtime_error("InxPack TOC tables are out of range: " + FromFsPath(path));
    // Inspect scalar records before allocating parsed entries. Shared path
    // ranges can represent more string memory than the physical TOC size.
    for (uint64_t index = 0; index < prefix.entryCount; ++index) {
        EntryDisk disk{};
        std::memcpy(&disk, toc.data() + sizeof(TocPrefixDisk) + index * sizeof(EntryDisk), sizeof(disk));
        if (disk.pathOffset > prefix.stringBytes || disk.pathBytes > prefix.stringBytes - disk.pathOffset ||
            disk.payloadOffset % kAlignment != 0 || disk.payloadOffset > header.payloadBytes ||
            disk.storedBytes > header.payloadBytes - disk.payloadOffset ||
            disk.storedBytes > std::numeric_limits<size_t>::max() || disk.rawBytes > std::numeric_limits<size_t>::max())
            throw std::runtime_error("InxPack entry is out of range: " + FromFsPath(path));
        if (disk.codec != static_cast<uint8_t>(Codec::Store) && disk.codec != static_cast<uint8_t>(Codec::Zstandard))
            throw std::runtime_error("InxPack entry uses an unsupported codec: " + FromFsPath(path));
        if (disk.alignment != kAlignment)
            throw std::runtime_error("InxPack entry alignment is invalid: " + FromFsPath(path));
        layout.pathBytes = CheckedAdd(layout.pathBytes, disk.pathBytes, "parsed path bytes");
        layout.largestPath = std::max<uint64_t>(layout.largestPath, disk.pathBytes);
    }
    return layout;
}

Manifest ParseDirectory(const HeaderDisk &header, const std::vector<uint8_t> &toc, const DirectoryLayout &layout,
                        uint64_t fileBytes)
{
    Manifest manifest;
    manifest.rawBytes = header.rawBytes;
    manifest.payloadBytes = header.payloadBytes;
    manifest.archiveBytes = fileBytes;
    std::copy(std::begin(header.packageHash), std::end(header.packageHash), manifest.archiveHash.begin());
    manifest.entries.reserve(static_cast<size_t>(layout.entries));
    for (uint64_t index = 0; index < layout.entries; ++index) {
        EntryDisk disk{};
        std::memcpy(&disk, toc.data() + sizeof(TocPrefixDisk) + index * sizeof(EntryDisk), sizeof(disk));
        const std::string rawPath(reinterpret_cast<const char *>(toc.data() + layout.stringsOffset + disk.pathOffset),
                                  disk.pathBytes);
        auto normalized = NormalizePath(rawPath);
        if (normalized != rawPath)
            throw std::runtime_error("InxPack entry path is not canonical: " + rawPath);
        Entry entry;
        entry.path = std::move(normalized);
        entry.offset = disk.payloadOffset;
        entry.storedBytes = disk.storedBytes;
        entry.rawBytes = disk.rawBytes;
        entry.codec = static_cast<Codec>(disk.codec);
        manifest.storedBytes = CheckedAdd(manifest.storedBytes, entry.storedBytes, "stored byte count");
        manifest.entries.push_back(std::move(entry));
    }
    std::sort(manifest.entries.begin(), manifest.entries.end(),
              [](const Entry &a, const Entry &b) { return a.path < b.path; });
    if (std::adjacent_find(manifest.entries.begin(), manifest.entries.end(),
                           [](const Entry &a, const Entry &b) { return a.path == b.path; }) != manifest.entries.end())
        throw std::runtime_error("InxPack entry path is duplicated");
    return manifest;
}

class ManifestCursor
{
  public:
    ManifestCursor(InputCursor &input, const std::filesystem::path &path, size_t metadataLimit)
        : input_(input), path_(path), fileBytes_(input.Size()), metadataLimit_(metadataLimit)
    {
    }

    IoWork DescribeWork(size_t maximum) const
    {
        switch (phase_) {
        case Phase::Signature:
            return IoByteWork("read.signature", signature_.size() - signatureBytes_, maximum);
        case Phase::Length:
            return {"read.header_length", 1, {}, 0};
        case Phase::Text:
            return IoByteWork(text_.empty() ? "read.text_allocate" : "read.text", textBytes_ - textRead_, maximum,
                              text_.empty() ? textBytes_ : 0);
        case Phase::Delimiter:
            return {"read.delimiter", 1, 1, 0};
        case Phase::Header: {
            const auto left = sizeof(HeaderDisk) - headerBytes_;
            return IoByteWork(left <= maximum ? "read.header_validate" : "read.header", left, maximum);
        }
        case Phase::Toc:
            return IoByteWork(toc_.empty() ? "read.toc_allocate" : "read.toc", header_.tocBytes - tocBytes_, maximum,
                              toc_.empty() ? header_.tocBytes : 0);
        case Phase::Inspect:
            return {"read.inspect_directory", 0, {}, toc_.size()};
        case Phase::Parse:
            return {"read.parse_directory", 0, {}, toc_.size()};
        case Phase::Ready:
            return {};
        }
        throw std::logic_error("unknown InxPack manifest phase");
    }

    size_t Advance(size_t maximum)
    {
        if (!maximum)
            throw std::invalid_argument("InxPack open step requires a positive byte budget");
        switch (phase_) {
        case Phase::Signature: {
            const auto count = std::min(maximum, signature_.size() - signatureBytes_);
            input_.Read(signature_.data() + signatureBytes_, count);
            signatureBytes_ += count;
            if (signatureBytes_ == signature_.size()) {
                if (std::string_view(signature_.data(), signature_.size()) == kInmMagic)
                    phase_ = Phase::Length;
                else {
                    std::memcpy(&header_, signature_.data(), signature_.size());
                    headerBytes_ = signature_.size();
                    phase_ = Phase::Header;
                }
            }
            return count;
        }
        case Phase::Length: {
            char value;
            input_.Read(&value, 1);
            ++lengthBytes_;
            if (value == '\n') {
                const auto envelopeBytes = uint64_t{kInmMagic.size()} + lengthBytes_ + 1;
                if (!textBytes_ || textBytes_ > metadataLimit_ ||
                    textBytes_ > fileBytes_ - std::min(fileBytes_, envelopeBytes))
                    throw std::runtime_error("INM text header exceeds the metadata limit or file size");
                phase_ = Phase::Text;
            } else {
                if (value < '0' || value > '9' || lengthBytes_ > 20 ||
                    textBytes_ > (std::numeric_limits<uint64_t>::max() - (value - '0')) / 10)
                    throw std::runtime_error("INM header length is invalid");
                textBytes_ = textBytes_ * 10 + (value - '0');
            }
            return 1;
        }
        case Phase::Text: {
            if (text_.empty())
                text_.resize(static_cast<size_t>(textBytes_));
            const auto count = std::min(maximum, text_.size() - textRead_);
            input_.Read(text_.data() + textRead_, count);
            textRead_ += count;
            if (textRead_ == text_.size())
                phase_ = Phase::Delimiter;
            return count;
        }
        case Phase::Delimiter: {
            char value;
            input_.Read(&value, 1);
            if (value != '\n')
                throw std::runtime_error("INM text header requires a newline before ckpt");
            input_.SetOrigin(kInmMagic.size() + lengthBytes_ + textBytes_ + 1);
            fileBytes_ = input_.Size();
            phase_ = Phase::Header;
            return 1;
        }
        case Phase::Header: {
            const auto count = std::min(maximum, sizeof(HeaderDisk) - headerBytes_);
            input_.Seek(headerBytes_);
            input_.Read(reinterpret_cast<uint8_t *>(&header_) + headerBytes_, count);
            headerBytes_ += count;
            if (headerBytes_ == sizeof(HeaderDisk)) {
                ValidateHeader(header_, fileBytes_, path_);
                if (header_.tocBytes > metadataLimit_)
                    throw std::runtime_error("InxPack TOC exceeds the managed metadata limit");
                phase_ = Phase::Toc;
            }
            return count;
        }
        case Phase::Toc: {
            if (toc_.empty())
                toc_.resize(static_cast<size_t>(header_.tocBytes));
            const auto count = std::min(maximum, toc_.size() - tocBytes_);
            if (count) {
                input_.Seek(header_.tocOffset + tocBytes_);
                input_.Read(toc_.data() + tocBytes_, count);
                tocBytes_ += count;
            }
            if (tocBytes_ == toc_.size())
                phase_ = Phase::Inspect;
            return count;
        }
        case Phase::Inspect:
            layout_ = InspectDirectory(header_, toc_, path_);
            phase_ = Phase::Parse;
            return 0;
        case Phase::Parse:
            manifest_ = ParseDirectory(header_, toc_, layout_, fileBytes_);
            manifest_.archiveBytes += input_.Origin();
            if (!text_.empty())
                manifest_.textHeader = std::make_shared<const std::string>(std::move(text_));
            std::vector<uint8_t>().swap(toc_);
            phase_ = Phase::Ready;
            return 0;
        case Phase::Ready:
            return 0;
        }
        throw std::logic_error("unknown InxPack manifest phase");
    }

    bool Done() const
    {
        return phase_ == Phase::Ready;
    }
    const HeaderDisk &Header() const
    {
        return header_;
    }
    Manifest TakeManifest()
    {
        return std::move(manifest_);
    }
    size_t RetainedBytes() const
    {
        size_t size = sizeof(*this) + path_.native().capacity() * sizeof(std::filesystem::path::value_type) +
                      toc_.capacity() + text_.capacity() + manifest_.HeaderStorageBytes() +
                      manifest_.entries.capacity() * sizeof(Entry);
        for (const auto &entry : manifest_.entries)
            size += entry.path.capacity() + 1;
        return size;
    }
    size_t ReservationBytes() const
    {
        if (Done())
            return RetainedBytes();
        uint64_t size = RetainedBytes();
        if (phase_ == Phase::Text)
            size = CheckedAdd(size, textBytes_ - text_.size(), "INM header workspace");
        if (phase_ == Phase::Toc)
            size = CheckedAdd(size, header_.tocBytes - toc_.size(), "TOC workspace");
        if (phase_ == Phase::Parse) {
            if (!text_.empty())
                size = CheckedAdd(size, sizeof(std::string) + 1 + std::string{}.capacity(), "INM header owner");
            size = CheckedAdd(size, CheckedMul(layout_.entries, sizeof(Entry) + 16, "directory entries"),
                              "directory workspace");
            size = CheckedAdd(size, layout_.pathBytes, "directory paths");
            size =
                CheckedAdd(size, CheckedMul(layout_.largestPath, 3, "directory path scratch"), "directory workspace");
        }
        if (size > std::numeric_limits<size_t>::max())
            throw std::runtime_error("InxPack metadata memory is too large");
        return static_cast<size_t>(size);
    }

  private:
    enum class Phase
    {
        Signature,
        Length,
        Text,
        Delimiter,
        Header,
        Toc,
        Inspect,
        Parse,
        Ready
    };
    InputCursor &input_;
    std::filesystem::path path_;
    uint64_t fileBytes_;
    size_t metadataLimit_, headerBytes_ = 0, tocBytes_ = 0;
    std::array<char, kInmMagic.size()> signature_{};
    size_t signatureBytes_ = 0, lengthBytes_ = 0, textRead_ = 0;
    uint64_t textBytes_ = 0;
    std::string text_;
    HeaderDisk header_{};
    DirectoryLayout layout_;
    std::vector<uint8_t> toc_;
    Manifest manifest_;
    Phase phase_ = Phase::Signature;
};

Manifest ParseManifest(InputCursor &input, const std::filesystem::path &path)
{
    ManifestCursor cursor(input, path, std::numeric_limits<size_t>::max());
    while (!cursor.Done())
        cursor.Advance(kHashChunkBytes);
    return cursor.TakeManifest();
}

HeaderDisk MakeHeader(const Manifest &manifest, uint64_t tocOffset, uint64_t tocBytes, uint64_t payloadOffset,
                      uint64_t stringBytes, const std::array<uint8_t, 32> &packageHash)
{
    HeaderDisk header{};
    std::copy(kMagic.begin(), kMagic.end(), std::begin(header.magic));
    header.headerBytes = sizeof(HeaderDisk);
    header.entryBytes = sizeof(EntryDisk);
    header.tocOffset = tocOffset;
    header.tocBytes = tocBytes;
    header.payloadOffset = payloadOffset;
    header.payloadBytes = manifest.payloadBytes;
    header.entryCount = manifest.entries.size();
    header.stringBytes = stringBytes;
    header.rawBytes = manifest.rawBytes;
    std::copy(packageHash.begin(), packageHash.end(), std::begin(header.packageHash));
    return header;
}

} // namespace

std::string HashToHex(const std::array<uint8_t, 32> &hash)
{
    return ToHex(hash.data(), hash.size());
}

const char *CodecName(Codec codec) noexcept
{
    return codec == Codec::Zstandard ? "zstd" : "store";
}

int ValidateCompressionLevel(const int compressionLevel)
{
    if (compressionLevel < kMinimumCompressionLevel || compressionLevel > kMaximumCompressionLevel)
        throw std::invalid_argument("InxPack compression level must be between " +
                                    std::to_string(kMinimumCompressionLevel) + " and " +
                                    std::to_string(kMaximumCompressionLevel));
    return compressionLevel;
}

int ResolveCompressionLevel(const WriteOptions &options)
{
    if (options.compressionLevel != 0)
        return ValidateCompressionLevel(options.compressionLevel);

    switch (options.profile) {
    case CompressionProfile::Development:
        return kDevelopmentCompressionLevel;
    case CompressionProfile::Release:
        return kReleaseCompressionLevel;
    default:
        throw std::invalid_argument("InxPack compression profile is invalid");
    }
}

size_t TextHeaderReservationBytes(size_t headerBytes)
{
    if (!headerBytes)
        return 0;
    const auto prelude = std::string(kInmMagic) + std::to_string(headerBytes) + "\n";
    return CheckedAdd(sizeof(std::string) + 1 + prelude.capacity(), std::max(headerBytes, std::string{}.capacity()),
                      "INM header reservation");
}

struct Writer::Impl
{
    enum class Phase
    {
        Open,
        Prelude,
        Text,
        Delimiter,
        Headers,
        Entry,
        Data,
        Padding,
        TocPrefix,
        TocEntries,
        TocStrings,
        Header,
        Hash,
        HashPatch,
        Close,
        Ready,
        Published,
        Cancelled
    };
    enum class Publication
    {
        Open,
        Cancelled,
        Committing,
        Published
    };

    Impl(const std::filesystem::path &target, std::vector<WriteEntry> entries, const WriteOptions &options,
         size_t block)
        : destination(ToFsPath(NormalizeFilesystemPathLexically(FromFsPath(target)))), file(destination),
          digest(static_cast<bool>(options.textHeader)), level(ResolveCompressionLevel(options)), blockBytes(block)
    {
        if (!blockBytes)
            throw std::invalid_argument("InxPack writer requires a nonempty IO block");
        manifest.textHeader = options.textHeader;
        if (manifest.textHeader) {
            if (manifest.textHeader->empty())
                throw std::invalid_argument("INM text header must not be empty");
            prelude = std::string(kInmMagic) + std::to_string(manifest.textHeader->size()) + "\n";
            binaryOffset = prelude.size() + manifest.textHeader->size() + 1;
        }
        const bool model = manifest.textHeader || LowerExtension(destination) == ".inm";
        for (auto &source : entries)
            source.path = NormalizePath(source.path);
        std::sort(entries.begin(), entries.end(),
                  [](const WriteEntry &left, const WriteEntry &right) { return left.path < right.path; });
        manifest.entries.reserve(entries.size());
        const bool resident = std::any_of(entries.begin(), entries.end(),
                                          [](const WriteEntry &entry) { return entry.residentData != nullptr; });
        if (resident)
            residentInputs.reserve(entries.size());
        for (auto &source : entries) {
            Entry entry;
            entry.path = std::move(source.path);
            if (entry.path.size() > std::numeric_limits<uint32_t>::max())
                throw std::invalid_argument("InxPack entry path is too long");
            if (source.residentData && source.bytes > (std::numeric_limits<size_t>::max)())
                throw std::length_error("InxPack resident input exceeds the address space");
            entry.rawBytes = source.bytes;
            const bool tensor = model && LowerExtension(ToFsPath(entry.path)) == ".f32";
            entry.codec = source.store || !source.bytes || tensor ? Codec::Store : Codec::Zstandard;
            manifest.entries.push_back(std::move(entry));
            if (resident)
                residentInputs.push_back(source.residentData);
        }
        std::string previous;
        bool compressed = false;
        for (const auto &entry : manifest.entries) {
            if (entry.path == previous)
                throw std::invalid_argument("duplicate InxPack entry path: " + entry.path);
            previous = entry.path;
            stringBytes = CheckedAdd(stringBytes, entry.path.size(), "path table");
            compressed = compressed || entry.codec == Codec::Zstandard;
        }
        tocBytes = AlignUp(
            CheckedAdd(CheckedAdd(sizeof(TocPrefixDisk),
                                  CheckedMul(manifest.entries.size(), sizeof(EntryDisk), "entry table"), "entry table"),
                       stringBytes, "path table"),
            kAlignment);
        payloadOffset = AlignUp(CheckedAdd(sizeof(HeaderDisk), tocBytes, "payload offset"), kAlignment);
        if (compressed) {
            contextReservation = ZSTD_estimateCStreamSize(level);
            if (ZSTD_isError(contextReservation))
                throw std::runtime_error("InxPack cannot estimate its compression workspace");
        }
        // The Writer already owns future publication even while idle for input.
        // Register the logical target and temporary sibling before any IO.
        const auto temporary = destination.parent_path() / file.Path().filename();
        if (options.writeScope.IsOpen()) {
            // A parent may authorize the public target but not its private,
            // generated sibling. The sibling gets its own ordinary intent.
            writeScope = PathWriteScope::BeginWrite({FromFsPath(destination)}, options.writeScope);
            const auto temporaryPath = FromFsPath(temporary);
            temporaryWriteScope = PathWriteScope::BeginWrite(
                {temporaryPath}, options.writeScope.Covers(temporaryPath) ? options.writeScope : PathWriteScope{});
        } else {
            writeScope = PathWriteScope::BeginWrite({FromFsPath(destination), FromFsPath(temporary)});
        }
    }

    size_t MetadataBytes() const
    {
        size_t bytes = sizeof(Impl) + manifest.entries.capacity() * sizeof(Entry) +
                       residentInputs.capacity() * sizeof(const uint8_t *) + prelude.capacity() +
                       manifest.HeaderStorageBytes();
        for (const auto &entry : manifest.entries)
            bytes += entry.path.capacity() + 1;
        bytes += destination.native().capacity() * sizeof(std::filesystem::path::value_type) + file.PathStorageBytes();
        return bytes;
    }

    const uint8_t *ResidentInput(size_t entry) const
    {
        return residentInputs.empty() ? nullptr : residentInputs[entry];
    }

    WriteState State() const
    {
        if (phase == Phase::Published)
            return WriteState::Published;
        if (phase == Phase::Cancelled)
            return WriteState::Cancelled;
        if (phase == Phase::Ready)
            return WriteState::ReadyToPublish;
        if (phase == Phase::Data && inputPosition == inputSize && entryQueued < manifest.entries[index].rawBytes)
            return WriteState::NeedsInput;
        return WriteState::Working;
    }

    void Abort()
    {
        file.Remove();
        context.reset();
        std::vector<uint8_t>().swap(input);
        std::vector<uint8_t>().swap(output);
        phase = Phase::Cancelled;
        temporaryWriteScope.Close();
        writeScope.Close();
    }

    void Write(const void *data, size_t bytes)
    {
        if (!bytes)
            return;
        file.Write(std::string_view(static_cast<const char *>(data), bytes));
        progress.writtenBytes += bytes;
    }

    void WriteZeros(size_t count)
    {
        static constexpr std::array<uint8_t, 4096> zeros{};
        while (count) {
            const auto bytes = std::min(count, zeros.size());
            Write(zeros.data(), bytes);
            count -= bytes;
        }
    }

    void QueueInput(const uint8_t *data, size_t bytes)
    {
        inputData = data;
        inputPosition = 0;
        inputSize = bytes;
        entryQueued += bytes;
        progress.queuedBytes += bytes;
    }

    bool WritePart(const void *data, size_t bytes, size_t maximum)
    {
        const auto count = static_cast<size_t>(std::min<uint64_t>(maximum, bytes - position));
        Write(static_cast<const uint8_t *>(data) + position, count);
        position += count;
        return position == bytes;
    }

    void Next(Phase next)
    {
        phase = next;
        position = 0;
    }

    IoWork DescribeWork(size_t maximum, bool publish) const
    {
        if (publication.load(std::memory_order_acquire) == Publication::Cancelled)
            return {"write.cleanup", 0, {}, 0};
        if (publish)
            return {"write.publish", 0, {}, 0};
        if (!maximum)
            throw std::invalid_argument("InxPack IO step requires a positive byte allowance");
        maximum = std::min(maximum, blockBytes);
        switch (phase) {
        case Phase::Open:
            return {"write.open", 0, {}, blockBytes};
        case Phase::Prelude:
            return IoByteWork("write.prelude", prelude.size() - position, maximum);
        case Phase::Text:
            return IoByteWork("write.text", manifest.textHeader->size() - position, maximum);
        case Phase::Delimiter:
            return {"write.delimiter", 1, 1, 0};
        case Phase::Headers:
            return IoByteWork("write.headers", payloadOffset - position, maximum);
        case Phase::Entry:
            if (index == manifest.entries.size())
                return {"write.entries_end", 0, {}, 0};
            if (manifest.entries[index].codec == Codec::Zstandard)
                return {context ? "write.zstd_reset" : "write.zstd_create", 0, {}, static_cast<uint64_t>(level)};
            return {"write.entry", 0, {}, 0};
        case Phase::Data: {
            if (State() == WriteState::NeedsInput)
                return {};
            const auto &entry = manifest.entries[index];
            if (entry.codec == Codec::Store)
                return IoByteWork("write.store", inputSize - inputPosition, maximum);
            const auto finishing = entryQueued == entry.rawBytes && inputPosition == inputSize;
            return {finishing ? "write.zstd_finish" : "write.zstd", maximum, {}, static_cast<uint64_t>(level)};
        }
        case Phase::Padding: {
            const auto &entry = manifest.entries[index];
            return IoByteWork("write.padding", AlignUp(entry.storedBytes, kAlignment) - entry.storedBytes - position,
                              maximum);
        }
        case Phase::TocPrefix:
            return IoByteWork("write.toc_prefix", sizeof(prefix) - position, maximum);
        case Phase::TocEntries:
            return index == manifest.entries.size() ? IoWork{"write.toc_entries_end", 0, {}, 0}
                                                    : IoByteWork("write.toc_entry", sizeof(disk) - position, maximum);
        case Phase::TocStrings:
            return index == manifest.entries.size()
                       ? IoWork{"write.toc_strings_end", 0, {}, 0}
                       : IoByteWork("write.toc_path", manifest.entries[index].path.size() - position, maximum);
        case Phase::Header: {
            const auto left = sizeof(header) - position;
            return IoByteWork(left <= maximum ? "write.header_finish" : "write.header", left, maximum);
        }
        case Phase::Hash: {
            const auto left = manifest.archiveBytes - progress.hashedBytes;
            return IoByteWork(left <= maximum ? "write.hash_finish" : "write.hash", left, maximum);
        }
        case Phase::HashPatch:
            return IoByteWork("write.hash_patch", manifest.archiveHash.size() - position, maximum);
        case Phase::Close:
            return {"write.close", 0, {}, blockBytes};
        case Phase::Ready:
        case Phase::Published:
        case Phase::Cancelled:
            return {};
        }
        throw std::logic_error("unknown InxPack writer phase");
    }

    static std::optional<uint64_t> StoredBound(const Entry &entry)
    {
        if (entry.codec == Codec::Store)
            return entry.rawBytes;
        // One successful frame: body <= raw, at most N nonempty blocks with
        // 3-byte headers, plus frame/header/epilogue <=25. This also covers
        // level1..22 splitting; it is not compressBound's one-pass contract.
        if (entry.rawBytes > (std::numeric_limits<uint64_t>::max() - 25) / 4)
            return {};
        return 4 * entry.rawBytes + 25;
    }

    std::optional<uint64_t> ArchiveBound() const
    {
        if (phase >= Phase::TocPrefix)
            return manifest.archiveBytes;
        auto bytes = binaryOffset;
        if (bytes > std::numeric_limits<uint64_t>::max() - payloadOffset)
            return {};
        bytes += payloadOffset;
        for (size_t i = 0; i < manifest.entries.size(); ++i) {
            const auto &entry = manifest.entries[i];
            const bool complete = phase >= Phase::Entry && (i < index || (phase == Phase::Padding && i == index));
            auto stored = complete ? std::optional<uint64_t>{entry.storedBytes} : StoredBound(entry);
            if (!stored || *stored > std::numeric_limits<uint64_t>::max() - (kAlignment - 1))
                return {};
            const auto aligned = AlignUp(*stored, kAlignment);
            if (bytes > std::numeric_limits<uint64_t>::max() - aligned)
                return {};
            bytes += aligned;
        }
        return bytes;
    }

    IoRecipe DescribeRemainingWork() const
    {
        if (phase == Phase::Published || phase == Phase::Cancelled)
            return {};
        IoRecipe result;
        const auto atom = [&](std::string_view operation, uint64_t parameter = 0, std::optional<size_t> entry = {}) {
            result.push_back({{{"atom", operation, entry, parameter, {}, {}, {}}}, 1});
        };
        const auto bytes = [&](std::string_view operation, std::optional<uint64_t> count, std::string_view finish = {},
                               bool upper = false, std::optional<size_t> entry = {}) {
            const bool empty = count && !*count;
            result.push_back({{{empty   ? "atom"
                                : upper ? "upper_bytes"
                                        : "bytes",
                                empty && !finish.empty() ? finish : operation,
                                entry,
                                0,
                                count,
                                {},
                                finish}},
                              1});
        };
        if (publication.load(std::memory_order_acquire) == Publication::Cancelled) {
            atom("write.cleanup");
            return result;
        }
        const auto remaining = [&](Phase step, uint64_t count) { return count - (phase == step ? position : 0); };
        if (phase == Phase::Open)
            atom("write.open", blockBytes);
        if (binaryOffset && phase <= Phase::Prelude)
            bytes("write.prelude", remaining(Phase::Prelude, prelude.size()));
        if (binaryOffset && phase <= Phase::Text)
            bytes("write.text", remaining(Phase::Text, manifest.textHeader->size()));
        if (binaryOffset && phase <= Phase::Delimiter)
            bytes("write.delimiter", 1);
        if (phase <= Phase::Headers)
            bytes("write.headers", remaining(Phase::Headers, payloadOffset));
        if (phase <= Phase::Padding) {
            bool hasContext = static_cast<bool>(context);
            const size_t first = phase >= Phase::Entry ? index : 0;
            for (size_t i = first; i < manifest.entries.size(); ++i) {
                const auto &entry = manifest.entries[i];
                const bool current = phase >= Phase::Entry && i == index;
                const bool data = current && phase == Phase::Data;
                const bool compressed = entry.codec == Codec::Zstandard;
                if (!current || phase == Phase::Entry) {
                    atom(compressed ? hasContext ? "write.zstd_reset" : "write.zstd_create" : "write.entry",
                         compressed ? static_cast<uint64_t>(level) : 0, i);
                    hasContext |= compressed;
                }
                if (!current || phase <= Phase::Data) {
                    const auto codec = [&](uint64_t inputBytes) {
                        return IoSpan{compressed ? "codec_fragment" : "bytes_fragment",
                                      compressed ? "write.zstd" : "write.store",
                                      i,
                                      compressed ? static_cast<uint64_t>(level) : 0,
                                      inputBytes,
                                      {},
                                      {},
                                      entry.rawBytes};
                    };
                    if (data && inputPosition != inputSize)
                        result.push_back({{codec(inputSize - inputPosition)}, 1});
                    uint64_t left = entry.rawBytes - (data ? entryQueued : 0);
                    if (left && ResidentInput(i)) {
                        result.push_back({{codec(left)}, 1});
                        left = 0;
                    }
                    const auto block = [&](uint64_t count, uint64_t repeats) {
                        if (repeats)
                            result.push_back(
                                {{{"input", "write.input", i, 0, count, count, {}}, codec(count)}, repeats});
                    };
                    // Keep first and tail separate even when both are full.
                    // A repeated middle is input/codec interleaving, not an
                    // invented number of future compression callbacks.
                    if (left) {
                        const auto firstBytes = std::min<uint64_t>(left, blockBytes);
                        block(firstBytes, 1);
                        left -= firstBytes;
                        if (left) {
                            const auto middle = (left - 1) / blockBytes;
                            block(blockBytes, middle);
                            block(left - middle * blockBytes, 1);
                        }
                    }
                    if (compressed) {
                        auto output = StoredBound(entry);
                        // Consumed input may still be buffered by the codec.
                        // Deduct only actual stored output from the whole-entry bound.
                        if (output && data)
                            *output -= entry.storedBytes;
                        result.push_back({{{"codec",
                                            "write.zstd_finish",
                                            i,
                                            static_cast<uint64_t>(level),
                                            0,
                                            output,
                                            {},
                                            entry.rawBytes}},
                                          1});
                    }
                }
                if (current && phase == Phase::Padding)
                    bytes("write.padding", AlignUp(entry.storedBytes, kAlignment) - entry.storedBytes - position, {},
                          false, i);
                else if (compressed)
                    bytes("write.padding", kAlignment - 1, {}, true, i);
                else
                    bytes("write.padding", AlignUp(entry.rawBytes, kAlignment) - entry.rawBytes, {}, false, i);
            }
            atom("write.entries_end");
        }
        if (phase <= Phase::TocPrefix)
            bytes("write.toc_prefix", remaining(Phase::TocPrefix, sizeof(prefix)));
        if (phase <= Phase::TocEntries) {
            const auto first = phase == Phase::TocEntries ? index : 0;
            for (size_t i = first; i < manifest.entries.size(); ++i)
                bytes("write.toc_entry", sizeof(disk) - (phase == Phase::TocEntries && i == index ? position : 0), {},
                      false, i);
            atom("write.toc_entries_end");
        }
        if (phase <= Phase::TocStrings) {
            const auto first = phase == Phase::TocStrings ? index : 0;
            for (size_t i = first; i < manifest.entries.size(); ++i)
                bytes("write.toc_path",
                      manifest.entries[i].path.size() - (phase == Phase::TocStrings && i == index ? position : 0), {},
                      false, i);
            atom("write.toc_strings_end");
        }
        if (phase <= Phase::Header)
            bytes("write.header", remaining(Phase::Header, sizeof(header)), "write.header_finish");
        if (phase <= Phase::Hash) {
            auto count = ArchiveBound();
            if (count)
                *count -= progress.hashedBytes;
            bytes("write.hash", count, "write.hash_finish", phase < Phase::TocPrefix);
        }
        if (phase <= Phase::HashPatch)
            bytes("write.hash_patch", remaining(Phase::HashPatch, manifest.archiveHash.size()));
        if (phase <= Phase::Close)
            atom("write.close", blockBytes);
        atom("write.publish");
        return result;
    }

    void Advance(size_t maximum)
    {
        if (!maximum)
            throw std::invalid_argument("InxPack IO step requires a positive byte allowance");
        if (publication.load(std::memory_order_acquire) == Publication::Cancelled) {
            Abort();
            return;
        }
        maximum = std::min(maximum, blockBytes);
        switch (phase) {
        case Phase::Open:
            if (!destination.parent_path().empty())
                std::filesystem::create_directories(NativeFilePath(destination.parent_path()));
            file.Open(detail::AtomicFileOutput::Access::ReadWrite);
            input.resize(blockBytes);
            if (contextReservation)
                output.resize(blockBytes);
            Next(binaryOffset ? Phase::Prelude : Phase::Headers);
            break;
        case Phase::Prelude:
            if (WritePart(prelude.data(), prelude.size(), maximum))
                Next(Phase::Text);
            break;
        case Phase::Text:
            if (WritePart(manifest.textHeader->data(), manifest.textHeader->size(), maximum))
                Next(Phase::Delimiter);
            break;
        case Phase::Delimiter:
            Write("\n", 1);
            Next(Phase::Headers);
            break;
        case Phase::Headers: {
            const auto count = static_cast<size_t>(std::min<uint64_t>(maximum, payloadOffset - position));
            WriteZeros(count);
            position += count;
            if (position == payloadOffset)
                Next(Phase::Entry);
            break;
        }
        case Phase::Entry:
            if (index == manifest.entries.size()) {
                manifest.payloadBytes = payloadCursor;
                manifest.archiveBytes =
                    CheckedAdd(binaryOffset, CheckedAdd(payloadOffset, payloadCursor, "archive size"), "INM size");
                std::copy(kTocMagic.begin(), kTocMagic.end(), std::begin(prefix.magic));
                prefix.entryCount = manifest.entries.size();
                prefix.stringBytes = stringBytes;
                file.Seek(binaryOffset + sizeof(HeaderDisk));
                Next(Phase::TocPrefix);
                break;
            }
            manifest.entries[index].offset = payloadCursor;
            inputData = nullptr;
            inputPosition = inputSize = 0;
            entryQueued = 0;
            if (manifest.entries[index].codec == Codec::Zstandard) {
                if (!context)
                    context.reset(ZSTD_createCCtx());
                if (!context || ZSTD_isError(ZSTD_CCtx_reset(context.get(), ZSTD_reset_session_and_parameters)) ||
                    ZSTD_isError(ZSTD_CCtx_setParameter(context.get(), ZSTD_c_compressionLevel, level)) ||
                    ZSTD_isError(ZSTD_CCtx_setPledgedSrcSize(context.get(), manifest.entries[index].rawBytes)))
                    throw std::runtime_error("InxPack cannot initialize compression");
            }
            Next(manifest.entries[index].rawBytes ? Phase::Data : Phase::Padding);
            if (const auto *resident = ResidentInput(index); resident && phase == Phase::Data)
                QueueInput(resident, static_cast<size_t>(manifest.entries[index].rawBytes));
            break;
        case Phase::Data: {
            if (State() == WriteState::NeedsInput)
                break;
            auto &entry = manifest.entries[index];
            const auto count = std::min(maximum, inputSize - inputPosition);
            if (entry.codec == Codec::Store) {
                Write(inputData + inputPosition, count);
                inputPosition += count;
                progress.consumedBytes += count;
                entry.storedBytes += count;
                if (inputPosition == inputSize && entryQueued == entry.rawBytes)
                    Next(Phase::Padding);
            } else {
                ZSTD_inBuffer in{count ? inputData + inputPosition : nullptr, count, 0};
                ZSTD_outBuffer out{output.data(), maximum, 0};
                // End with an empty input step. Otherwise a large last chunk
                // can select zstd's one-shot fast path while tiny chunks use
                // its buffered path, making IO budgets change asset bytes.
                const bool last = entryQueued == entry.rawBytes && inputPosition == inputSize;
                const auto pending =
                    ZSTD_compressStream2(context.get(), &out, &in, last ? ZSTD_e_end : ZSTD_e_continue);
                if (ZSTD_isError(pending))
                    throw std::runtime_error(std::string("InxPack streaming compression failed: ") +
                                             ZSTD_getErrorName(pending));
                Write(output.data(), out.pos);
                inputPosition += in.pos;
                progress.consumedBytes += in.pos;
                entry.storedBytes += out.pos;
                if (last && inputPosition == inputSize && pending == 0)
                    Next(Phase::Padding);
            }
            break;
        }
        case Phase::Padding: {
            const auto &entry = manifest.entries[index];
            const auto padding = AlignUp(entry.storedBytes, kAlignment) - entry.storedBytes;
            const auto count = static_cast<size_t>(std::min<uint64_t>(maximum, padding - position));
            WriteZeros(count);
            position += count;
            if (position == padding) {
                manifest.rawBytes = CheckedAdd(manifest.rawBytes, entry.rawBytes, "raw bytes");
                manifest.storedBytes = CheckedAdd(manifest.storedBytes, entry.storedBytes, "stored bytes");
                payloadCursor = CheckedAdd(entry.offset, entry.storedBytes + padding, "payload bytes");
                ++index;
                Next(Phase::Entry);
            }
            break;
        }
        case Phase::TocPrefix:
            if (WritePart(&prefix, sizeof(prefix), maximum)) {
                index = 0;
                Next(Phase::TocEntries);
            }
            break;
        case Phase::TocEntries:
            if (index == manifest.entries.size()) {
                index = 0;
                Next(Phase::TocStrings);
                break;
            }
            if (!position) {
                const auto &entry = manifest.entries[index];
                disk = {};
                disk.pathOffset = pathCursor;
                disk.pathBytes = static_cast<uint32_t>(entry.path.size());
                disk.payloadOffset = entry.offset;
                disk.storedBytes = entry.storedBytes;
                disk.rawBytes = entry.rawBytes;
                disk.codec = static_cast<uint8_t>(entry.codec);
                disk.alignment = kAlignment;
            }
            if (WritePart(&disk, sizeof(disk), maximum)) {
                pathCursor += manifest.entries[index].path.size();
                ++index;
                position = 0;
            }
            break;
        case Phase::TocStrings:
            if (index == manifest.entries.size()) {
                header = MakeHeader(manifest, sizeof(HeaderDisk), tocBytes, payloadOffset, stringBytes, {});
                file.Seek(binaryOffset);
                Next(Phase::Header);
                break;
            }
            if (WritePart(manifest.entries[index].path.data(), manifest.entries[index].path.size(), maximum)) {
                ++index;
                position = 0;
            }
            break;
        case Phase::Header:
            if (WritePart(&header, sizeof(header), maximum)) {
                file.Seek(0);
                Next(Phase::Hash);
            }
            break;
        case Phase::Hash: {
            const auto count =
                static_cast<size_t>(std::min<uint64_t>(maximum, manifest.archiveBytes - progress.hashedBytes));
            file.Read(input.data(), count);
            digest.Update(input.data(), count);
            progress.hashedBytes += count;
            if (progress.hashedBytes == manifest.archiveBytes) {
                manifest.archiveHash = digest.Final();
                file.Seek(binaryOffset + offsetof(HeaderDisk, packageHash));
                Next(Phase::HashPatch);
            }
            break;
        }
        case Phase::HashPatch:
            if (WritePart(manifest.archiveHash.data(), manifest.archiveHash.size(), maximum))
                Next(Phase::Close);
            break;
        case Phase::Close:
            file.Finish();
            context.reset();
            std::vector<uint8_t>().swap(input);
            std::vector<uint8_t>().swap(output);
            Next(Phase::Ready);
            break;
        case Phase::Ready:
        case Phase::Published:
        case Phase::Cancelled:
            break;
        }
    }

    std::filesystem::path destination;
    // Destruction must close/remove private IO
    // before releasing the accepted operation's path ownership.
    PathWriteScope writeScope;
    PathWriteScope temporaryWriteScope;
    detail::AtomicFileOutput file;
    Manifest manifest;
    std::vector<const uint8_t *> residentInputs;
    std::string prelude;
    std::vector<uint8_t> input, output;
    const uint8_t *inputData = nullptr;
    std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)> context{nullptr, &ZSTD_freeCCtx};
    ArchiveChecksum digest;
    TocPrefixDisk prefix{};
    EntryDisk disk{};
    HeaderDisk header{};
    WriteProgress progress;
    std::atomic<Publication> publication{Publication::Open};
    Phase phase = Phase::Open;
    int level;
    size_t blockBytes, contextReservation = 0, index = 0, inputPosition = 0, inputSize = 0;
    uint64_t binaryOffset = 0, stringBytes = 0, tocBytes = 0, payloadOffset = 0, payloadCursor = 0;
    uint64_t position = 0, pathCursor = 0, entryQueued = 0;
    uint32_t publishAttempts = 0;
};

Writer::Writer(const std::filesystem::path &destination, std::vector<WriteEntry> entries, const WriteOptions &options,
               size_t blockBytes)
    : m_impl(std::make_unique<Impl>(destination, std::move(entries), options, blockBytes))
{
}
Writer::~Writer() = default;
WriteState Writer::GetState() const
{
    return m_impl->State();
}
WriteProgress Writer::GetProgress() const
{
    return m_impl->progress;
}
IoWork Writer::DescribeWork(size_t maxBytes, bool publish) const
{
    return m_impl->DescribeWork(maxBytes, publish);
}
IoRecipe Writer::DescribeRemainingWork() const
{
    return m_impl->DescribeRemainingWork();
}
const std::string &Writer::GetInputPath() const
{
    if (GetState() != WriteState::NeedsInput)
        throw std::logic_error("InxPack writer is not requesting input");
    return m_impl->manifest.entries[m_impl->index].path;
}
size_t Writer::GetInputBytes() const
{
    if (GetState() != WriteState::NeedsInput)
        return 0;
    return static_cast<size_t>(
        std::min<uint64_t>(m_impl->blockBytes, m_impl->manifest.entries[m_impl->index].rawBytes - m_impl->entryQueued));
}
void Writer::Append(const uint8_t *data, size_t bytes)
{
    if (!bytes || bytes > GetInputBytes())
        throw std::invalid_argument("InxPack input block exceeds the current entry range");
    std::memcpy(m_impl->input.data(), data, bytes);
    m_impl->QueueInput(m_impl->input.data(), bytes);
}
void Writer::Advance(size_t maxBytes)
{
    m_impl->Advance(maxBytes);
}
void Writer::Discard()
{
    m_impl->Abort();
}
bool Writer::Cancel()
{
    auto expected = Impl::Publication::Open;
    return m_impl->publication.compare_exchange_strong(expected, Impl::Publication::Cancelled,
                                                       std::memory_order_acq_rel) ||
           expected == Impl::Publication::Cancelled;
}
bool Writer::Publish()
{
    auto &state = *m_impl;
    if (state.phase == Impl::Phase::Published)
        return true;
    if (state.phase == Impl::Phase::Cancelled)
        return false;
    if (state.phase != Impl::Phase::Ready)
        throw std::logic_error("InxPack output is not ready to publish");
    auto expected = Impl::Publication::Open;
    if (!state.publication.compare_exchange_strong(expected, Impl::Publication::Committing,
                                                   std::memory_order_acq_rel)) {
        if (expected == Impl::Publication::Cancelled)
            state.Abort();
        return false;
    }
    std::error_code error;
    if (ReplaceFileAtomically(state.file.Path(), state.destination, error)) {
        const auto syncError = state.file.Published();
        state.publication.store(Impl::Publication::Published, std::memory_order_release);
        state.phase = Impl::Phase::Published;
        state.temporaryWriteScope.Close();
        state.writeScope.Close();
        if (syncError)
            throw std::system_error(syncError, "InxPack published output but directory flush failed");
        return true;
    }
    if (IsTransientReplaceError(error) && ++state.publishAttempts < 64) {
        state.publication.store(Impl::Publication::Open, std::memory_order_release);
        return false;
    }
    throw std::runtime_error("InxPack cannot publish output: " + error.message());
}
Manifest Writer::GetManifest() const
{
    if (GetState() != WriteState::Published)
        throw std::logic_error("InxPack output is not published");
    return m_impl->manifest;
}
size_t Writer::GetRetainedBytes() const
{
    return m_impl->MetadataBytes() + m_impl->input.capacity() + m_impl->output.capacity() +
           (m_impl->context ? ZSTD_sizeof_CCtx(m_impl->context.get()) : 0);
}
size_t Writer::GetReservationBytes() const
{
    if (GetState() == WriteState::Published || GetState() == WriteState::Cancelled ||
        GetState() == WriteState::ReadyToPublish)
        return GetRetainedBytes();
    return m_impl->MetadataBytes() + m_impl->blockBytes +
           (m_impl->contextReservation ? m_impl->blockBytes + m_impl->contextReservation : 0);
}

Manifest Write(const std::filesystem::path &destination, std::vector<Source> sources, const WriteOptions &options)
{
    // The synchronous caller only feeds the same bounded writer. It has no
    // second small-entry codec, whole-archive hash or publication algorithm.
    std::vector<WriteEntry> entries;
    entries.reserve(sources.size());
    size_t blockBytes = kHashChunkBytes;
    for (auto &source : sources) {
        source.path = NormalizePath(source.path);
        const auto *path = std::get_if<std::filesystem::path>(&source.data);
        const auto extension = LowerExtension(path ? *path : ToFsPath(source.path));
        const auto size =
            path ? std::filesystem::file_size(NativeFilePath(*path)) : std::get<MemorySource>(source.data).bytes;
        blockBytes = std::max(blockBytes, static_cast<size_t>(std::min<uint64_t>(size, kSynchronousWriteBlockBytes)));
        entries.push_back({source.path, size,
                           extension == ".mp3" || extension == ".ogg" || extension == ".flac" || extension == ".inm",
                           path ? nullptr : std::get<MemorySource>(source.data).data});
    }
    std::sort(sources.begin(), sources.end(), [](const Source &a, const Source &b) { return a.path < b.path; });
    Writer writer(destination, std::move(entries), options, blockBytes);
    auto &state = *writer.m_impl;
    std::unique_ptr<SourceInput> input;
    while (writer.GetState() != WriteState::ReadyToPublish) {
        if (writer.GetState() == WriteState::NeedsInput) {
            if (!input) {
                const auto found =
                    std::lower_bound(sources.begin(), sources.end(), writer.GetInputPath(),
                                     [](const Source &source, const std::string &path) { return source.path < path; });
                input = std::make_unique<SourceInput>(std::get<std::filesystem::path>(found->data));
            }
            const auto count = writer.GetInputBytes();
            state.QueueInput(input->ReadView(state.input.data(), count), count);
            if (state.entryQueued == state.manifest.entries[state.index].rawBytes) {
                input->Finish();
                input.reset();
            }
        }
        writer.Advance(blockBytes);
    }
    while (!writer.Publish())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return writer.GetManifest();
}

Manifest ReadManifest(const std::filesystem::path &path)
{
    FileInput file(path);
    InputCursor input(file);
    return ParseManifest(input, path);
}

struct ReadCursor::Impl
{
    Impl(const std::filesystem::path &source, size_t metadata, size_t block)
        : path(std::filesystem::absolute(source)), metadataLimit(metadata), blockBytes(block)
    {
        if (!blockBytes)
            throw std::invalid_argument("InxPack read block must be positive");
    }
    void RequireReady() const
    {
        if (state != ReadState::Ready)
            throw std::logic_error("InxPack cursor is not ready for another operation");
    }
    const Entry &Find(const std::string &name) const
    {
        if (state == ReadState::Opening || state == ReadState::Closed)
            throw std::logic_error("InxPack directory is not open");
        const auto path = NormalizePath(name);
        const auto found =
            std::lower_bound(manifest.entries.begin(), manifest.entries.end(), path,
                             [](const Entry &entry, const std::string &key) { return entry.path < key; });
        if (found == manifest.entries.end() || found->path != path)
            throw std::runtime_error("InxPack entry was not found: " + name);
        return *found;
    }
    size_t BaseBytes() const
    {
        size_t size = sizeof(Impl) + path.native().capacity() * sizeof(std::filesystem::path::value_type) +
                      manifest.HeaderStorageBytes() + manifest.entries.capacity() * sizeof(Entry);
        for (const auto &entry : manifest.entries)
            size += entry.path.capacity() + 1;
        return size + (file ? sizeof(FileInput) : 0) + (input ? sizeof(InputCursor) : 0);
    }
    size_t MaterializeBlockBytes() const
    {
        return static_cast<size_t>(std::min<uint64_t>(blockBytes, materializedEntry->rawBytes));
    }
    size_t AdvanceDecoder(size_t maximum, uint8_t *destination, size_t bytes)
    {
        const auto range = decoder->PrepareRead(maximum, destination, bytes);
        if (range.bytes)
            file->ReadAt(range.offset, range.destination, range.bytes);
        return FinishDecoder(maximum, destination, bytes, range.bytes);
    }
    size_t FinishDecoder(size_t maximum, uint8_t *destination, size_t bytes, size_t readBytes)
    {
        const auto consumed = decoder->ConsumedInputBytes();
        decoder->FinishRead(maximum, destination, bytes, readBytes);
        const auto produced = decoder->OutputBytes();
        progress.readBytes += readBytes;
        progress.codecInputBytes += decoder->ConsumedInputBytes() - consumed;
        progress.decodedBytes += produced;
        return produced;
    }
    std::filesystem::path path;
    size_t metadataLimit, blockBytes;
    std::unique_ptr<FileInput> file;
    std::unique_ptr<InputCursor> input;
    std::unique_ptr<ManifestCursor> opening;
    Manifest manifest;
    uint64_t payloadOffset = 0;
    std::unique_ptr<RangeDigest> digest;
    std::unique_ptr<EntryDecoder> decoder;
    std::unique_ptr<Materialization> materialization;
    const Entry *materializedEntry = nullptr;
    std::unique_ptr<uint8_t[]> materializeBlock;
    std::optional<size_t> sinkOutputOffset;
    std::shared_ptr<VerifiedFile> materializedFile;
    ReadState state = ReadState::Opening;
    ReadProgress progress;
    bool verified = false;
};

ReadCursor::ReadCursor(const std::filesystem::path &path, size_t metadataLimit, size_t blockBytes)
    : m_impl(std::make_unique<Impl>(path, metadataLimit, blockBytes))
{
}
ReadCursor::~ReadCursor() = default;
void ReadCursor::Advance(size_t maxBytes)
{
    if (!maxBytes)
        throw std::invalid_argument("InxPack read step requires a positive byte budget");
    auto &s = *m_impl;
    const auto maximum = std::min(maxBytes, s.blockBytes);
    switch (s.state) {
    case ReadState::Opening:
        if (!s.file) {
            auto file = std::make_unique<FileInput>(s.path);
            auto input = std::make_unique<InputCursor>(*file);
            auto opening = std::make_unique<ManifestCursor>(*input, s.path, s.metadataLimit);
            s.file = std::move(file);
            s.input = std::move(input);
            s.opening = std::move(opening);
            return;
        }
        s.progress.readBytes += s.opening->Advance(maximum);
        if (s.opening->Done()) {
            s.payloadOffset = s.opening->Header().payloadOffset;
            s.manifest = s.opening->TakeManifest();
            s.opening.reset();
            s.state = ReadState::Ready;
        }
        return;
    case ReadState::Checking: {
        s.materializedFile.reset();
        const auto count = s.digest->Advance(maximum);
        s.progress.readBytes += count;
        s.progress.hashedBytes += count;
        if (s.digest->Done()) {
            if (s.digest->Finish() != s.manifest.archiveHash)
                throw std::runtime_error("InxPack payload checksum mismatch");
            s.verified = true;
            s.digest.reset();
            s.state = ReadState::Ready;
        }
        return;
    }
    case ReadState::Reading:
        throw std::logic_error("provide a destination for the InxPack entry");
    case ReadState::Materializing:
        s.materializedFile.reset();
        if (!s.materialization->NeedsInput()) {
            s.materialization->Advance(maximum, s.progress);
            if (s.materialization->Done()) {
                auto &sink = *s.materialization;
                s.materializedFile = std::shared_ptr<VerifiedFile>(new VerifiedFile(
                    sink.TakePath(), sink.Bytes(), sink.TakeSha256(), sink.TakeInput(), sink.TakeAnchors()));
                s.materialization.reset();
                s.decoder.reset();
                s.materializedEntry = nullptr;
                s.state = ReadState::Ready;
            }
            return;
        }
        if (!s.decoder) {
            s.decoder = std::make_unique<EntryDecoder>(*s.input, s.payloadOffset, *s.materializedEntry, s.blockBytes);
            s.materializeBlock.reset(new uint8_t[s.MaterializeBlockBytes()]);
            return; // Reserve the compressed input before allocating it on the next step.
        }
        if (s.sinkOutputOffset) {
            const auto offset = *s.sinkOutputOffset;
            const auto count = std::min(maximum, s.decoder->OutputBytes() - offset);
            s.materialization->Write(s.materializeBlock.get() + offset, count, s.progress);
            if (offset + count == s.decoder->OutputBytes())
                s.sinkOutputOffset.reset();
            else
                s.sinkOutputOffset = offset + count;
            return;
        }
        if (s.decoder->Done()) {
            s.materialization->FinishInput();
            s.decoder.reset();
            s.materializeBlock.reset();
        } else if (s.AdvanceDecoder(maximum, s.materializeBlock.get(), s.MaterializeBlockBytes())) {
            s.sinkOutputOffset = 0;
        }
        return;
    case ReadState::Ready:
        return;
    case ReadState::Closed:
        throw std::logic_error("InxPack cursor is closed");
    }
}
void ReadCursor::BeginChecksum()
{
    auto &s = *m_impl;
    s.RequireReady();
    s.digest = std::make_unique<RangeDigest>(
        *s.file, 0, s.manifest.archiveBytes, s.blockBytes,
        std::make_pair(s.input->Origin() + offsetof(HeaderDisk, packageHash), uint64_t{32}),
        static_cast<bool>(s.manifest.textHeader));
    s.verified = false;
    s.state = ReadState::Checking;
}
void ReadCursor::BeginRead(const std::string &entryPath)
{
    auto &s = *m_impl;
    s.RequireReady();
    s.decoder = std::make_unique<EntryDecoder>(*s.input, s.payloadOffset, s.Find(entryPath), s.blockBytes);
    s.state = ReadState::Reading;
}
size_t ReadCursor::ReadInto(uint8_t *destination, size_t bytes, size_t maxBytes)
{
    const auto range = PrepareReadInto(destination, bytes, maxBytes);
    if (range.bytes)
        OpenedFile().ReadAt(range.offset, range.destination, range.bytes);
    return FinishReadInto(destination, bytes, maxBytes, range.bytes);
}
EntryReadRange ReadCursor::PrepareReadInto(uint8_t *destination, size_t bytes, size_t maxBytes)
{
    if (!maxBytes)
        throw std::invalid_argument("InxPack read step requires a positive byte budget");
    auto &s = *m_impl;
    if (s.state != ReadState::Reading)
        throw std::logic_error("InxPack cursor is not reading an entry");
    return s.decoder->Done() ? EntryReadRange{} : s.decoder->PrepareRead(maxBytes, destination, bytes);
}
size_t ReadCursor::FinishReadInto(uint8_t *destination, size_t bytes, size_t maxBytes, size_t readBytes)
{
    auto &s = *m_impl;
    s.materializedFile.reset();
    const auto produced = s.decoder->Done() ? 0 : s.FinishDecoder(maxBytes, destination, bytes, readBytes);
    s.progress.consumedBytes += produced;
    if (s.decoder->Done()) {
        s.decoder.reset();
        s.state = ReadState::Ready;
    }
    return produced;
}
FileInput &ReadCursor::OpenedFile() const
{
    return *m_impl->file;
}
void ReadCursor::BeginMaterialize(const std::string &entryPath, const std::filesystem::path &destination,
                                  const std::string &expectedSha256)
{
    auto &s = *m_impl;
    s.RequireReady();
    const auto &entry = s.Find(entryPath);
    auto materialization = std::make_unique<Materialization>(destination, entry.rawBytes, expectedSha256, s.blockBytes);
    s.materialization = std::move(materialization);
    s.materializedEntry = &entry;
    s.sinkOutputOffset.reset();
    s.state = ReadState::Materializing;
}
std::shared_ptr<VerifiedFile> ReadCursor::TakeMaterializedFile()
{
    auto &s = *m_impl;
    s.RequireReady();
    if (!s.materializedFile)
        throw std::logic_error("InxPack has no completed file result");
    return std::move(s.materializedFile);
}
ReadState ReadCursor::GetState() const
{
    return m_impl->state;
}
ReadProgress ReadCursor::GetProgress() const
{
    return m_impl->progress;
}
IoWork ReadCursor::DescribeWork(size_t maxBytes) const
{
    if (!maxBytes)
        throw std::invalid_argument("InxPack read step requires a positive byte budget");
    const auto &s = *m_impl;
    const auto maximum = std::min(maxBytes, s.blockBytes);
    switch (s.state) {
    case ReadState::Opening:
        return s.file ? s.opening->DescribeWork(maximum) : IoWork{"read.open", 0, {}, 0};
    case ReadState::Checking: {
        const auto left = s.digest->RemainingBytes();
        return IoByteWork(left <= maximum ? "read.hash_finish" : "read.hash", left, maximum);
    }
    case ReadState::Reading:
        return s.decoder->Done() ? IoWork{"read.entry_end", 0, {}, 0} : s.decoder->DescribeWork(maximum);
    case ReadState::Materializing:
        if (!s.materialization->NeedsInput())
            return s.materialization->DescribeWork(maximum);
        if (!s.decoder)
            return {"materialize.decoder_create", 0, {}, 0};
        if (s.sinkOutputOffset)
            return IoByteWork("materialize.write", s.decoder->OutputBytes() - *s.sinkOutputOffset, maximum);
        if (s.decoder->Done())
            return {"materialize.hash_finish", 0, {}, 0};
        return s.decoder->DescribeWork(maximum);
    case ReadState::Ready:
    case ReadState::Closed:
        return {};
    }
    throw std::logic_error("unknown InxPack read state");
}
IoRecipe ReadCursor::DescribeReadWork(const std::string &entryPath) const
{
    const auto &s = *m_impl;
    const auto &entry = s.Find(entryPath);
    const auto index = static_cast<size_t>(&entry - s.manifest.entries.data());
    if (entry.codec == Codec::Store && !entry.rawBytes)
        return {{{{"atom", "read.entry_end", index, 0, 0, 0, {}}}, 1}};
    return {{{{"codec", entry.codec == Codec::Store ? "read.store" : "read.zstd", index, 0, entry.storedBytes,
               entry.rawBytes, "read.entry_end", entry.rawBytes}},
             1}};
}
IoRecipe ReadCursor::DescribeRemainingWork() const
{
    const auto &s = *m_impl;
    switch (s.state) {
    case ReadState::Opening:
        return {{{{"barrier", "read.directory", {}, 0, {}, {}, {}}}, 1}};
    case ReadState::Checking:
        return {{{{"bytes", "read.hash", {}, 0, s.digest->RemainingBytes(), {}, "read.hash_finish"}}, 1}};
    case ReadState::Reading: {
        const auto &entry = s.decoder->SourceEntry();
        const auto index = static_cast<size_t>(&entry - s.manifest.entries.data());
        IoRecipe result;
        if (s.decoder->Done())
            result.push_back({{{"atom", "read.entry_end", index, 0, {}, {}, {}}}, 1});
        else
            result.push_back({{{"codec", entry.codec == Codec::Store ? "read.store" : "read.zstd", index, 0,
                                entry.storedBytes - s.decoder->ConsumedInputBytes(),
                                entry.rawBytes - s.decoder->DecodedBytes(), "read.entry_end", entry.rawBytes}},
                              1});
        return result;
    }
    case ReadState::Materializing:
        return s.materialization->DescribeRemainingWork(
            *s.materializedEntry, static_cast<size_t>(s.materializedEntry - s.manifest.entries.data()), s.decoder.get(),
            s.sinkOutputOffset.has_value());
    case ReadState::Ready:
    case ReadState::Closed:
        return {};
    }
    throw std::logic_error("unknown InxPack read state");
}
bool ReadCursor::IsChecksumVerified() const
{
    return m_impl->verified;
}
Manifest ReadCursor::GetManifest() const
{
    if (m_impl->state == ReadState::Opening || m_impl->state == ReadState::Closed)
        throw std::logic_error("InxPack directory is not open");
    return m_impl->manifest;
}
uint64_t ReadCursor::GetEntryBytes(const std::string &entryPath) const
{
    return m_impl->Find(entryPath).rawBytes;
}
void ReadCursor::DiscardOperation()
{
    auto &s = *m_impl;
    if (s.state == ReadState::Checking || s.state == ReadState::Reading || s.state == ReadState::Materializing) {
        s.digest.reset();
        s.decoder.reset();
        s.materialization.reset();
        s.materializeBlock.reset();
        s.materializedEntry = nullptr;
        s.sinkOutputOffset.reset();
        s.state = ReadState::Ready;
    }
}
size_t ReadCursor::GetRetainedBytes() const
{
    const auto &s = *m_impl;
    return s.BaseBytes() + (s.opening ? s.opening->RetainedBytes() : 0) + (s.digest ? s.digest->RetainedBytes() : 0) +
           (s.decoder ? s.decoder->RetainedBytes() : 0) + (s.materialization ? s.materialization->RetainedBytes() : 0) +
           (s.materializeBlock ? s.MaterializeBlockBytes() : 0) +
           (s.materializedFile ? s.materializedFile->RetainedBytes() : 0);
}
size_t ReadCursor::GetReservationBytes() const
{
    const auto &s = *m_impl;
    size_t size = s.BaseBytes() + (s.opening ? s.opening->ReservationBytes() : 0) +
                  (s.digest ? s.digest->ReservationBytes() : 0) + (s.decoder ? s.decoder->ReservationBytes() : 0) +
                  (s.materialization ? s.materialization->ReservationBytes() + sizeof(VerifiedFile) : 0) +
                  (s.materializedFile ? s.materializedFile->RetainedBytes() : 0);
    if (s.state == ReadState::Opening && !s.opening)
        size += sizeof(FileInput) + sizeof(InputCursor) + sizeof(ManifestCursor) +
                s.path.native().capacity() * sizeof(std::filesystem::path::value_type) + std::string{}.capacity();
    if (s.state == ReadState::Ready)
        size += std::max(sizeof(RangeDigest), sizeof(EntryDecoder));
    if (s.materialization && s.materialization->NeedsInput()) {
        size += s.MaterializeBlockBytes();
        if (!s.decoder)
            size += sizeof(EntryDecoder);
    }
    return size;
}
bool ReadCursor::HasFileResources() const
{
    return bool(m_impl->file || m_impl->materialization || m_impl->materializedFile);
}
bool ReadCursor::HasMaterialization() const
{
    return bool(m_impl->materialization);
}
void ReadCursor::Close()
{
    auto &s = *m_impl;
    s.decoder.reset();
    s.materialization.reset();
    s.materializeBlock.reset();
    s.materializedFile.reset();
    s.materializedEntry = nullptr;
    s.sinkOutputOffset.reset();
    s.digest.reset();
    s.opening.reset();
    s.input.reset();
    s.file.reset();
    s.manifest = {};
    s.state = ReadState::Closed;
}

struct Reader::Impl
{
    explicit Impl(const std::filesystem::path &path) : cursor(path, std::numeric_limits<size_t>::max())
    {
        Drive();
    }
    void Drive()
    {
        try {
            while (cursor.GetState() != ReadState::Ready)
                cursor.Advance(kHashChunkBytes);
        } catch (...) {
            cursor.DiscardOperation();
            throw;
        }
    }
    void ReadInto(const std::string &entryPath, uint8_t *destination, size_t bytes)
    {
        if (bytes != cursor.GetEntryBytes(entryPath))
            throw std::invalid_argument("InxPack destination size differs from entry: " + entryPath);
        size_t offset = 0;
        cursor.BeginRead(entryPath);
        try {
            while (cursor.GetState() != ReadState::Ready)
                offset += cursor.ReadInto(destination + offset, bytes - offset, kHashChunkBytes);
        } catch (...) {
            cursor.DiscardOperation();
            throw;
        }
    }
    ReadCursor cursor;
};

Reader::Reader(const std::filesystem::path &path) : m_impl(std::make_unique<Impl>(path))
{
}
Reader::~Reader() = default;
Reader::Impl &Reader::State() const
{
    if (!m_impl)
        throw std::runtime_error("InxPack reader is closed");
    return *m_impl;
}
Manifest Reader::GetManifest() const
{
    std::lock_guard lock(m_mutex);
    return State().cursor.GetManifest();
}
void Reader::Close()
{
    std::lock_guard lock(m_mutex);
    m_impl.reset();
}
void Reader::VerifyChecksum()
{
    std::lock_guard lock(m_mutex);
    auto &s = State();
    s.cursor.BeginChecksum();
    s.Drive();
}
void Reader::ReadInto(const std::string &entryPath, uint8_t *destination, size_t bytes)
{
    std::lock_guard lock(m_mutex);
    State().ReadInto(entryPath, destination, bytes);
}
std::vector<uint8_t> Reader::Read(const std::string &entryPath)
{
    std::lock_guard lock(m_mutex);
    auto &s = State();
    std::vector<uint8_t> raw(static_cast<size_t>(s.cursor.GetEntryBytes(entryPath)));
    s.ReadInto(entryPath, raw.data(), raw.size());
    return raw;
}
std::shared_ptr<VerifiedFile> Reader::ExtractEntry(const std::string &entryPath,
                                                   const std::filesystem::path &destination,
                                                   const std::string &expectedSha256)
{
    std::lock_guard lock(m_mutex);
    auto &s = State();
    s.cursor.BeginMaterialize(entryPath, destination, expectedSha256);
    s.Drive();
    return s.cursor.TakeMaterializedFile();
}
std::vector<uint8_t> ReadEntry(const std::filesystem::path &path, const std::string &entryPath)
{
    return Reader(path).Read(entryPath);
}

std::shared_ptr<VerifiedFile> CopyVerifiedFile(const std::filesystem::path &source,
                                               const std::filesystem::path &destination,
                                               const std::string &expectedSha256)
{
    const auto writeScope = PathWriteScope::BeginWrite({FromFsPath(destination)});
    SourceInput input(source);
    auto sink = Materialize(destination, input.Size(), expectedSha256, [&](auto &&write) {
        std::array<uint8_t, kHashChunkBytes> block;
        uint64_t remaining = input.Size();
        while (remaining) {
            const auto count = static_cast<size_t>(std::min<uint64_t>(remaining, block.size()));
            write(input.ReadView(block.data(), count), count);
            remaining -= count;
        }
        input.Finish();
    });
    return std::shared_ptr<VerifiedFile>(
        new VerifiedFile(sink.TakePath(), sink.Bytes(), sink.TakeSha256(), sink.TakeInput(), sink.TakeAnchors()));
}

std::array<uint8_t, 32> HashFile(FileInput &input)
{
    input.Seek(0);
    Sha256 digest;
    std::array<uint8_t, kHashChunkBytes> block;
    uint64_t remaining = input.Size();
    while (remaining) {
        const auto count = static_cast<size_t>(std::min<uint64_t>(remaining, block.size()));
        input.Read(block.data(), count);
        digest.Update(block.data(), count);
        remaining -= count;
    }
    return digest.Final();
}

Manifest Extract(const std::filesystem::path &path, const std::filesystem::path &destination,
                 const std::vector<std::string> &allowedRoots, bool parallel)
{
    FileInput source(path);
    InputCursor manifestInput(source);
    const Manifest manifest = ParseManifest(manifestInput, path);
    HeaderDisk header{};
    manifestInput.Seek(0);
    manifestInput.Read(&header, sizeof(header));

    // Pin the complete directory tree before workers start. Workers only open
    // exclusive temporary files beneath these directories and publish entries
    // after validating the entire decoded payload.
    ExtractionTree tree(destination);
    std::vector<ExtractionTarget> outputs;
    outputs.reserve(manifest.entries.size());
#ifdef _WIN32
    const auto ignoreCaseLess = [](const std::wstring &left, const std::wstring &right) {
        return ::CompareStringOrdinal(left.c_str(), static_cast<int>(left.size()), right.c_str(),
                                      static_cast<int>(right.size()), TRUE) == CSTR_LESS_THAN;
    };
    std::set<std::wstring, decltype(ignoreCaseLess)> outputNames(ignoreCaseLess);
#endif
    for (const auto &entry : manifest.entries) {
        if (!IsAllowedRoot(entry.path, allowedRoots))
            throw std::runtime_error("InxPack entry root is not allowed: " + entry.path);
#ifdef _WIN32
        if (!outputNames.insert(ToFsPath(entry.path).wstring()).second)
            throw std::runtime_error("InxPack entries collide on a case-insensitive filesystem: " + entry.path);
#endif
        auto directory = tree.Root();
        std::filesystem::path name;
        size_t begin = 0;
        while (begin < entry.path.size()) {
            const size_t end = entry.path.find('/', begin);
            const auto part =
                ToFsPath(entry.path.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
            if (end == std::string::npos) {
                name = part;
                break;
            }
            directory = tree.Child(directory, part);
            begin = end + 1;
        }
        outputs.push_back({std::move(directory), std::move(name)});
    }

    const auto extractEntry = [&](InputCursor &input, size_t index) {
        const auto &entry = manifest.entries[index];
        ExtractedFile destinationFile(outputs[index]);
        DecodeEntry(input, header.payloadOffset, entry, [&](const uint8_t *bytes, size_t size) {
            destinationFile.Write(reinterpret_cast<const char *>(bytes), size);
        });
        destinationFile.Publish();
    };

    // Archive entries are independently compressed. Browser builds
    // deliberately remain single-threaded unless a future Player profile opts
    // into wasm threads and the required cross-origin isolation contract.
#if defined(INFERNUX_SINGLE_THREADED_RUNTIME)
    parallel = false;
#endif
    if (!parallel) {
        for (size_t index = 0; index < manifest.entries.size(); ++index)
            extractEntry(manifestInput, index);
        return manifest;
    }
#if !defined(INFERNUX_SINGLE_THREADED_RUNTIME)
    // Native hosts use bounded parallel extraction during a cold Player
    // launch without creating an unbounded number of disk writers on slower
    // machines. Workers own decode cursors and outputs over the same input.
    const size_t hardwareThreads = std::max<size_t>(1, std::thread::hardware_concurrency());
    const size_t workerCount = std::min({manifest.entries.size(), hardwareThreads, size_t{4}});
    std::atomic<size_t> nextEntry{0};
    std::atomic<bool> failed{false};
    std::exception_ptr failure;
    std::mutex failureMutex;
    std::vector<std::thread> workers;
    workers.reserve(workerCount);
    for (size_t worker = 0; worker < workerCount; ++worker) {
        workers.emplace_back([&] {
            try {
                InputCursor input(source);
                input.SetOrigin(manifestInput.Origin());
                while (!failed.load(std::memory_order_acquire)) {
                    const size_t index = nextEntry.fetch_add(1, std::memory_order_relaxed);
                    if (index >= manifest.entries.size())
                        break;
                    extractEntry(input, index);
                }
            } catch (...) {
                failed.store(true, std::memory_order_release);
                std::lock_guard lock(failureMutex);
                if (!failure)
                    failure = std::current_exception();
            }
        });
    }
    for (auto &worker : workers)
        worker.join();
    if (failure)
        std::rethrow_exception(failure);
#endif
    return manifest;
}

} // namespace infernux::inxpack
