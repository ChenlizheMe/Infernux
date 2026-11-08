#pragma once

#include "NativeFilePath.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <system_error>
#include <utility>

#include <core/config/InxPlatform.h>
#ifndef INX_PLATFORM_WINDOWS
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace infernux
{
inline std::string FromFsPath(const std::filesystem::path &p);
#ifdef INX_PLATFORM_WINDOWS
inline std::pair<uint64_t, uint64_t> FileIdentity(HANDLE handle)
{
    BY_HANDLE_FILE_INFORMATION info{};
    if (!::GetFileInformationByHandle(handle, &info))
        throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                "Cannot identify file input");
    return {info.dwVolumeSerialNumber, (uint64_t{info.nFileIndexHigh} << 32) | info.nFileIndexLow};
}
#endif
} // namespace infernux
namespace infernux
{

/**
 * @brief Normalize a portable logical path by using forward slashes.
 */
inline std::string NormalizePortablePath(const std::string &path)
{
    std::string result = path;
    std::replace(result.begin(), result.end(), '\\', '/');
    return result;
}

/**
 * @brief Convert a UTF-8 path string to std::filesystem::path.
 *
 * On Windows the native encoding for narrow strings is the active code-page,
 * NOT UTF-8.  This helper ensures paths that were produced by Python
 * (which always outputs UTF-8) are correctly converted to wide-char
 * internally so that std::ifstream / std::filesystem operations work with
 * non-ASCII characters (e.g. Chinese filenames).
 */
inline std::filesystem::path ToFsPath(const std::string &utf8Path)
{
#ifdef INX_PLATFORM_WINDOWS
    // MultiByteToWideChar: convert UTF-8 → wchar_t
    if (utf8Path.empty())
        return {};
    int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8Path.data(), static_cast<int>(utf8Path.size()), nullptr, 0);
    if (wlen <= 0)
        return std::filesystem::path(utf8Path);
    std::wstring wstr(static_cast<size_t>(wlen), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8Path.data(), static_cast<int>(utf8Path.size()), wstr.data(), wlen);
    return std::filesystem::path(std::move(wstr));
#else
    // On Linux / macOS the native encoding is UTF-8 already.
    return std::filesystem::path(utf8Path);
#endif
}

/**
 * @brief Convert a std::filesystem::path to a UTF-8 std::string.
 *
 * On Windows, std::filesystem::path::string() / generic_string() encode
 * using the active code-page (e.g. GBK), NOT UTF-8.  This helper always
 * produces a UTF-8 string so that the result is compatible with ToFsPath()
 * and with Python (which expects UTF-8).
 */
inline std::string FromFsPath(const std::filesystem::path &p)
{
#ifdef INX_PLATFORM_WINDOWS
    std::wstring ws = p.generic_wstring();
    if (ws.empty())
        return {};
    int len = WideCharToMultiByte(CP_UTF8, 0, ws.data(), static_cast<int>(ws.size()), nullptr, 0, nullptr, nullptr);
    if (len <= 0)
        return p.generic_string();
    std::string result(static_cast<size_t>(len), '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws.data(), static_cast<int>(ws.size()), result.data(), len, nullptr, nullptr);
    return result;
#else
    return p.generic_string();
#endif
}

/**
 * @brief Return an absolute, lexically-normal UTF-8 path without consulting disk.
 *
 * Use this for paths that may
 * no longer exist. It deliberately does not
 * resolve symlinks or Windows 8.3 aliases, so its result never changes
 * merely
 * because a file was created or deleted.
 */
inline std::string NormalizeFilesystemPathLexically(const std::string &path)
{
    if (path.empty())
        return {};

    std::error_code error;
    std::filesystem::path normalized = ToFsPath(path);
    if (normalized.is_relative()) {
        auto absolute = std::filesystem::absolute(normalized, error);
        if (!error)
            normalized = std::move(absolute);
    }
    return FromFsPath(normalized.lexically_normal());
}

/**
 * @brief Resolve an existing path to its stable display/storage spelling.
 *
 * Resolves aliases in the existing prefix and keeps the remaining path
 * lexical. Windows uses the normalized long names of the existing entries.
 */
inline std::string ResolveFilesystemPath(const std::string &path)
{
    std::string lexical = NormalizeFilesystemPathLexically(path);
    if (lexical.empty())
        return {};

    std::error_code error;
    const auto canonical = std::filesystem::weakly_canonical(ToFsPath(lexical), error);
    return error ? lexical : FromFsPath(canonical.lexically_normal());
}

inline std::string PortablePathFilename(const std::string &path)
{
    return FromFsPath(ToFsPath(NormalizePortablePath(path)).filename());
}

inline std::string PortablePathStem(const std::string &path)
{
    return FromFsPath(ToFsPath(NormalizePortablePath(path)).stem());
}

inline bool TryNormalizePortableRelativePath(const std::string &path, std::string &normalized, bool allowRoot = false)
{
    normalized.clear();
    if (path.empty())
        return false;

    const auto portable = NormalizePortablePath(path);
    // Portable asset paths must reject Windows roots on every build host.
    const bool drive = portable.size() >= 2 && portable[1] == ':' &&
                       ((portable[0] >= 'A' && portable[0] <= 'Z') || (portable[0] >= 'a' && portable[0] <= 'z'));
    if (portable.front() == '/' || drive)
        return false;
    const auto relative = ToFsPath(portable).lexically_normal();
    if (relative.empty() || relative.is_absolute())
        return false;
    if (relative == ".") {
        if (!allowRoot)
            return false;
        normalized = ".";
        return true;
    }
    if (*relative.begin() == "..")
        return false;
    normalized = NormalizePortablePath(FromFsPath(relative));
    return true;
}

inline std::string FoldFilesystemPathCase(std::string path)
{
#ifdef INX_PLATFORM_WINDOWS
    std::wstring folded = ToFsPath(path).generic_wstring();
    if (!folded.empty())
        CharLowerBuffW(folded.data(), static_cast<DWORD>(folded.size()));
    return FromFsPath(std::filesystem::path(std::move(folded)));
#else
    return path;
#endif
}

inline std::string FilesystemPathKey(const std::string &path)
{
    return FoldFilesystemPathCase(ResolveFilesystemPath(path));
}

/**
 * @brief Return a stable key for a filesystem-backed asset or virtual subresource.
 *
 * Project assets may append an engine-owned ``::kind:id`` suffix to a real
 * filesystem path. Only the backing path participates in filesystem
 * normalization; the virtual identity remains exact and case-sensitive.
 */
inline std::string AssetPathKey(const std::string &path)
{
    const auto virtualSuffix = path.find("::");
    if (virtualSuffix == std::string::npos)
        return FilesystemPathKey(path);
    return FilesystemPathKey(path.substr(0, virtualSuffix)) + path.substr(virtualSuffix);
}

inline std::string LexicalFilesystemPathKey(const std::string &path)
{
    return FoldFilesystemPathCase(NormalizeFilesystemPathLexically(path));
}

inline bool FilesystemPathsEquivalent(const std::string &left, const std::string &right)
{
    if (left.empty() || right.empty())
        return false;
    std::error_code error;
    if (std::filesystem::exists(ToFsPath(left), error) && !error && std::filesystem::exists(ToFsPath(right), error) &&
        !error) {
        const bool equivalent = std::filesystem::equivalent(ToFsPath(left), ToFsPath(right), error);
        if (!error)
            return equivalent;
    }
    return FilesystemPathKey(left) == FilesystemPathKey(right);
}

/// Pure containment on already normalized keys, with a component boundary.
inline bool IsFilesystemPathKeyWithin(const std::string &path, const std::string &root, bool allowRoot = true)
{
    if (path.empty() || root.empty())
        return false;
    if (path == root)
        return allowRoot;
    return path.size() > root.size() && path.compare(0, root.size(), root) == 0 &&
           (root.back() == '/' || path[root.size()] == '/');
}

inline bool IsFilesystemPathWithin(const std::string &path, const std::string &root, bool allowRoot = true)
{
    if (path.empty() || root.empty())
        return false;

    return IsFilesystemPathKeyWithin(FilesystemPathKey(path), FilesystemPathKey(root), allowRoot);
}

inline bool TryMakeRelativeFilesystemPath(const std::string &path, const std::string &root, std::string &relative,
                                          bool allowRoot = false)
{
    relative.clear();
    if (path.empty() || root.empty())
        return false;

    const auto candidate = ToFsPath(ResolveFilesystemPath(path));
    const auto parent = ToFsPath(ResolveFilesystemPath(root));
    auto candidatePart = candidate.begin();
    for (auto rootPart = parent.begin(); rootPart != parent.end(); ++rootPart, ++candidatePart) {
        if (candidatePart == candidate.end() ||
            FoldFilesystemPathCase(FromFsPath(*candidatePart)) != FoldFilesystemPathCase(FromFsPath(*rootPart)))
            return false;
    }
    if (candidatePart == candidate.end()) {
        if (!allowRoot)
            return false;
        relative = ".";
        return true;
    }
    // The prefix comparison follows host case rules; constructing the suffix
    // preserves the asset's spelling even when the root uses another case.
    std::filesystem::path result;
    for (; candidatePart != candidate.end(); ++candidatePart)
        result /= *candidatePart;
    return TryNormalizePortableRelativePath(FromFsPath(result), relative, allowRoot);
}

#ifdef INX_PLATFORM_LINUX
inline std::filesystem::path NativeDescriptorPath(int descriptor)
{
    return std::filesystem::path("/proc/self/fd") / std::to_string(descriptor);
}
#endif

/// One seekable input snapshot. Readers permit atomic pathname publication;
/// callers choose bounded blocks rather than requiring a whole-file copy.
class FileInput
{
  public:
    enum class Sharing
    {
        ReadWrite,
        Snapshot
    };

    struct State
    {
        uint64_t size;
        std::pair<int64_t, int64_t> modified;
    };

    struct Revision
    {
        State data;
        std::pair<uint64_t, uint64_t> identity;
        std::pair<int64_t, int64_t> changed;

        bool operator==(const Revision &other) const
        {
            return data.size == other.data.size && data.modified == other.data.modified && identity == other.identity &&
                   changed == other.changed;
        }
    };

    explicit FileInput(const std::filesystem::path &path, Sharing sharing = Sharing::ReadWrite)
    {
#ifdef INX_PLATFORM_WINDOWS
        shareAccess_ = FILE_SHARE_READ | FILE_SHARE_DELETE;
        if (sharing == Sharing::ReadWrite)
            shareAccess_ |= FILE_SHARE_WRITE;
        handle_ = ::CreateFileW(NativeFilePath(path).c_str(), GENERIC_READ, shareAccess_, nullptr, OPEN_EXISTING,
                                FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE)
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "Cannot open file input");
#else
        (void)sharing;
        descriptor_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor_ < 0)
            throw std::system_error(errno, std::generic_category(), "Cannot open file input: " + FromFsPath(path));
#endif
    }
#ifdef INX_PLATFORM_LINUX
    int Descriptor() const
    {
        return descriptor_;
    }
#endif
    ~FileInput()
    {
        Close();
    }
    FileInput(const FileInput &) = delete;
    FileInput &operator=(const FileInput &) = delete;

    void Close() noexcept
    {
        std::lock_guard lock(mutex_);
#ifdef INX_PLATFORM_WINDOWS
        if (handle_ != INVALID_HANDLE_VALUE)
            ::CloseHandle(std::exchange(handle_, INVALID_HANDLE_VALUE));
#else
        if (descriptor_ >= 0)
            ::close(std::exchange(descriptor_, -1));
#endif
    }

    uint64_t Size()
    {
        return Inspect().size;
    }

    State Inspect()
    {
        std::lock_guard lock(mutex_);
#ifdef INX_PLATFORM_WINDOWS
        BY_HANDLE_FILE_INFORMATION info{};
        if (!::GetFileInformationByHandle(handle_, &info))
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "Cannot inspect file input");
        return {(uint64_t{info.nFileSizeHigh} << 32) | info.nFileSizeLow,
                {info.ftLastWriteTime.dwHighDateTime, info.ftLastWriteTime.dwLowDateTime}};
#else
        struct stat info{};
        if (::fstat(descriptor_, &info) != 0)
            throw std::system_error(errno, std::generic_category(), "Cannot inspect file input");
#ifdef __APPLE__
        const auto &modified = info.st_mtimespec;
#else
        const auto &modified = info.st_mtim;
#endif
        return {static_cast<uint64_t>(info.st_size), {modified.tv_sec, modified.tv_nsec}};
#endif
    }

    /// Identity of the open object, including in-place changes that restore mtime.
    Revision InspectRevision()
    {
        std::lock_guard lock(mutex_);
#ifdef INX_PLATFORM_WINDOWS
        BY_HANDLE_FILE_INFORMATION info{};
        FILE_BASIC_INFO basic{};
        if (!::GetFileInformationByHandle(handle_, &info) ||
            !::GetFileInformationByHandleEx(handle_, FileBasicInfo, &basic, sizeof(basic)))
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "Cannot inspect file revision");
        return {{(uint64_t{info.nFileSizeHigh} << 32) | info.nFileSizeLow,
                 {info.ftLastWriteTime.dwHighDateTime, info.ftLastWriteTime.dwLowDateTime}},
                {info.dwVolumeSerialNumber, (uint64_t{info.nFileIndexHigh} << 32) | info.nFileIndexLow},
                {basic.ChangeTime.QuadPart, 0}};
#else
        struct stat info{};
        if (::fstat(descriptor_, &info) != 0)
            throw std::system_error(errno, std::generic_category(), "Cannot inspect file revision");
#ifdef __APPLE__
        const auto &modified = info.st_mtimespec;
        const auto &changed = info.st_ctimespec;
#else
        const auto &modified = info.st_mtim;
        const auto &changed = info.st_ctim;
#endif
        return {{static_cast<uint64_t>(info.st_size), {modified.tv_sec, modified.tv_nsec}},
                {static_cast<uint64_t>(info.st_dev), static_cast<uint64_t>(info.st_ino)},
                {changed.tv_sec, changed.tv_nsec}};
#endif
    }

    void Seek(uint64_t offset)
    {
        std::lock_guard lock(mutex_);
        position_ = offset;
    }

#ifdef INX_PLATFORM_WINDOWS
    std::pair<uint64_t, uint64_t> Identity()
    {
        std::lock_guard lock(mutex_);
        return FileIdentity(handle_);
    }

    HANDLE OpenOverlapped()
    {
        std::lock_guard lock(mutex_);
        const auto handle =
            ::ReOpenFile(handle_, GENERIC_READ, shareAccess_, FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN);
        if (handle == INVALID_HANDLE_VALUE)
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "Cannot open asynchronous file input");
        return handle;
    }
#endif

    void Read(void *destination, size_t bytes)
    {
        std::lock_guard lock(mutex_);
        ReadExact(position_, destination, bytes);
        position_ += bytes;
    }

    /// Independent ranges share the same file revision, without moving the
    /// sequential cursor. Decode/consumer work happens outside this IO lock.
    void ReadAt(uint64_t offset, void *destination, size_t bytes)
    {
        std::lock_guard lock(mutex_);
        ReadExact(offset, destination, bytes);
    }

  private:
    void ReadExact(uint64_t offset, void *destination, size_t bytes)
    {
        auto *data = static_cast<unsigned char *>(destination);
        while (bytes) {
            const auto count = (std::min)(bytes, size_t{1} << 30);
#ifdef INX_PLATFORM_WINDOWS
            OVERLAPPED range{};
            range.Offset = static_cast<DWORD>(offset);
            range.OffsetHigh = static_cast<DWORD>(offset >> 32);
            DWORD received = 0;
            // A synchronous handle can read at an explicit offset in one call.
            if (!::ReadFile(handle_, data, static_cast<DWORD>(count), &received, &range)) {
                const auto error = ::GetLastError();
                if (error != ERROR_HANDLE_EOF)
                    throw std::system_error(static_cast<int>(error), std::system_category(), "Cannot read file input");
            }
#else
            const auto received = ::pread(descriptor_, data, count, static_cast<off_t>(offset));
            if (received < 0) {
                if (errno == EINTR)
                    continue;
                throw std::system_error(errno, std::generic_category(), "Cannot read file input");
            }
#endif
            if (!received)
                throw std::runtime_error("File input ended before the requested range");
            data += received;
            bytes -= received;
            offset += received;
        }
    }

    std::mutex mutex_;
    uint64_t position_ = 0;
#ifdef INX_PLATFORM_WINDOWS
    DWORD shareAccess_;
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int descriptor_ = -1;
#endif
};

/**
 * @brief Read a file into a byte vector, supporting Unicode paths on Windows.
 */
inline bool ReadFileBytes(const std::string &filePath, std::vector<unsigned char> &out)
{
    try {
        FileInput file(ToFsPath(filePath));
        out.resize(static_cast<size_t>(file.Size()));
        file.Read(out.data(), out.size());
        return true;
    } catch (const std::runtime_error &) {
        return false;
    }
}

inline std::ifstream OpenInputFile(const std::string &filePath, std::ios_base::openmode mode = std::ios::in)
{
    return std::ifstream(ToFsPath(filePath), mode);
}

inline std::ofstream OpenOutputFile(const std::string &filePath, std::ios_base::openmode mode = std::ios::out)
{
    return std::ofstream(ToFsPath(filePath), mode);
}

inline std::string JoinPath(std::initializer_list<const char *> parts)
{
    std::filesystem::path path;
    for (const auto &part : parts) {
        path /= ToFsPath(part);
    }
    return FromFsPath(path);
}

inline std::string JoinPath(std::initializer_list<std::string> parts)
{
    std::filesystem::path path;
    for (const auto &part : parts) {
        path /= ToFsPath(part);
    }
    return FromFsPath(path);
}
} // namespace infernux
