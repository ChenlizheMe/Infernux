#include "AtomicFile.h"

#include "AtomicFileOutput.h"
#include "FileFingerprint.h"
#include "InxPath.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <system_error>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#ifdef _WIN32
#include "OverlappedFile.h"
#include <condition_variable>
#include <core/threading/FrameWorkOwner.h>
#include <core/threading/JobSystem.h>
#include <mutex>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace infernux
{
#ifdef _WIN32
bool detail::ReplaceOpenFileAtomically(void *handle, const std::filesystem::path &target, std::error_code &error)
{
    const auto absolute = std::filesystem::absolute(target, error);
    if (error)
        return false;
    const auto name = NativeFilePath(absolute).native();
    const auto nameBytes = name.size() * sizeof(wchar_t);
    std::vector<uint8_t> bytes(offsetof(FILE_RENAME_INFO, FileName) + nameBytes + sizeof(wchar_t), 0);
    auto *rename = reinterpret_cast<FILE_RENAME_INFO *>(bytes.data());
    rename->Flags = FILE_RENAME_FLAG_REPLACE_IF_EXISTS | FILE_RENAME_FLAG_POSIX_SEMANTICS;
    rename->RootDirectory = nullptr;
    rename->FileNameLength = static_cast<DWORD>(nameBytes);
    std::memcpy(rename->FileName, name.data(), nameBytes);
    if (::SetFileInformationByHandle(handle, FileRenameInfoEx, rename, static_cast<DWORD>(bytes.size()))) {
        error.clear();
        return true;
    }
    error = std::error_code(static_cast<int>(::GetLastError()), std::system_category());
    return false;
}
#endif

bool ReplaceFileAtomically(const std::filesystem::path &source, const std::filesystem::path &target,
                           std::error_code &error)
{
#ifdef _WIN32
    HANDLE handle =
        ::CreateFileW(NativeFilePath(source).c_str(), DELETE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                      nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        error = std::error_code(static_cast<int>(::GetLastError()), std::system_category());
        return false;
    }
    const auto close = [](void *value) { ::CloseHandle(value); };
    const std::unique_ptr<void, decltype(close)> owned(handle, close);
    return detail::ReplaceOpenFileAtomically(owned.get(), target, error);
#else
    std::filesystem::rename(source, target, error);
    return !error;
#endif
}

std::optional<FilePublication> PublishFileIfAbsent(const std::filesystem::path &source,
                                                   const std::filesystem::path &target, std::error_code &error)
{
#ifdef _WIN32
    if (::MoveFileExW(NativeFilePath(source).c_str(), NativeFilePath(target).c_str(), MOVEFILE_WRITE_THROUGH)) {
        error.clear();
        return FilePublication::Written;
    }
    const auto code = ::GetLastError();
    if (code == ERROR_ALREADY_EXISTS || code == ERROR_FILE_EXISTS) {
        error.clear();
        return FilePublication::AlreadyExists;
    }
    error = std::error_code(static_cast<int>(code), std::system_category());
    return std::nullopt;
#else
    return detail::PublishFileIfAbsentAt(AT_FDCWD, source, target, error);
#endif
}

#ifndef _WIN32
std::optional<FilePublication> detail::PublishFileIfAbsentAt(int directory, const std::filesystem::path &source,
                                                             const std::filesystem::path &target,
                                                             std::error_code &error)
{
    if (::linkat(directory, source.c_str(), directory, target.c_str(), 0) != 0) {
        if (errno == EEXIST) {
            error.clear();
            return FilePublication::AlreadyExists;
        }
        error = std::error_code(errno, std::generic_category());
        return std::nullopt;
    }
    if (::unlinkat(directory, source.c_str(), 0) != 0) {
        error = std::error_code(errno, std::generic_category());
        return std::nullopt;
    }
    error.clear();
    return FilePublication::Written;
}
#endif

std::filesystem::path MakeAtomicTemporaryPath(const std::filesystem::path &target)
{
    static std::atomic<uint64_t> sequence{0};
#ifdef _WIN32
    const auto process = ::GetCurrentProcessId();
#else
    const auto process = ::getpid();
#endif
    auto path = target;
    path += std::filesystem::path(".inx-tmp." + std::to_string(process) + "." +
                                  std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "." +
                                  std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
    return path;
}

bool IsAtomicTemporaryPath(const std::filesystem::path &path)
{
    const std::string name = FromFsPath(path.filename());
    constexpr std::string_view marker = ".inx-tmp.";
    const auto position = name.rfind(marker);
    if (position == std::string::npos)
        return false;
    size_t cursor = position + marker.size();
    for (unsigned part = 0; part < 3; ++part) {
        const size_t begin = cursor;
        while (cursor < name.size() && name[cursor] >= '0' && name[cursor] <= '9')
            ++cursor;
        if (cursor == begin)
            return false;
        if (part == 2)
            return cursor == name.size();
        if (cursor == name.size() || name[cursor++] != '.')
            return false;
    }
    return false;
}

namespace
{

#ifdef _WIN32
constexpr size_t kDirectRangeBytes = 32 * 1024 * 1024;
constexpr size_t kDirectMinimumBytes = kDirectRangeBytes / 2;

bool UseDirectOutput(const std::filesystem::path &target, uint64_t completeBytes)
{
    if (completeBytes < kDirectMinimumBytes)
        return false;
    const auto root = target.root_path();
    return ::GetDriveTypeW(root.c_str()) == DRIVE_FIXED;
}

[[noreturn]] void ThrowOutputError(const char *operation)
{
    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), operation);
}

struct ReleasePages
{
    void operator()(void *pages) const noexcept
    {
        ::VirtualFree(pages, 0, MEM_RELEASE);
    }
};

void WriteDirectOutput(HANDLE handle, OverlappedFile &file, std::string_view content,
                       const FrameWorkOwner *storageOwner)
{
    FILE_STORAGE_INFO storage{};
    FILE_ALIGNMENT_INFO alignment{};
    if (!::GetFileInformationByHandleEx(handle, FileStorageInfo, &storage, sizeof(storage)) ||
        !::GetFileInformationByHandleEx(handle, FileAlignmentInfo, &alignment, sizeof(alignment)))
        ThrowOutputError("cannot inspect direct file output alignment");
    const size_t sector = storage.LogicalBytesPerSector;
    const size_t rangeBytes = std::max(sector, kDirectRangeBytes / sector * sector);
    const auto alignBytes = [sector](size_t bytes) { return (bytes + sector - 1) / sector * sector; };
    const size_t alignmentMask = alignment.AlignmentRequirement;
    const auto paddedSize = alignBytes(content.size());
    FILE_ALLOCATION_INFO allocation{};
    allocation.AllocationSize.QuadPart = paddedSize;
    if (!::SetFileInformationByHandle(handle, FileAllocationInfo, &allocation, sizeof(allocation)))
        ThrowOutputError("cannot reserve direct file output");

    std::shared_ptr<FrameCpuStorageClaim> claim;
    std::unique_ptr<void, ReleasePages> staging;
    std::mutex mutex;
    std::condition_variable completed;

    // Complete a large contiguous range before reusing its input or staging page.
    for (size_t offset = 0; offset < content.size();) {
        const size_t used = std::min(content.size() - offset, rangeBytes);
        const size_t count = alignBytes(used);
        const char *source = content.data() + offset;
        if (used != count || (reinterpret_cast<uintptr_t>(source) & alignmentMask)) {
            if (!staging) {
                SYSTEM_INFO systemInfo{};
                ::GetSystemInfo(&systemInfo);
                const size_t extra = alignmentMask < systemInfo.dwAllocationGranularity ? 0 : alignmentMask;
                const size_t pageSize = systemInfo.dwPageSize;
                const size_t capacity = (count + extra + pageSize - 1) / pageSize * pageSize;
                if (storageOwner)
                    claim = storageOwner->TrackOwnedCpuStorage(capacity);
                staging.reset(::VirtualAlloc(nullptr, capacity, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
                if (!staging)
                    ThrowOutputError("cannot allocate direct file staging pages");
            }
            const auto address = reinterpret_cast<uintptr_t>(staging.get());
            auto *page = reinterpret_cast<char *>((address + alignmentMask) & ~alignmentMask);
            std::memcpy(page, source, used);
            std::memset(page + used, 0, count - used);
            source = page;
        }
        std::optional<FileRangeCompletion> receipt;
        file.Write(offset, source, count, [&mutex, &completed, &receipt](FileRangeCompletion result) {
            std::lock_guard lock(mutex);
            receipt = result;
            completed.notify_one();
        });
        std::unique_lock lock(mutex);
        completed.wait(lock, [&receipt] { return receipt.has_value(); });
        if (receipt->error)
            throw std::system_error(receipt->error, "cannot write direct file range");
        if (receipt->bytes != count)
            throw std::runtime_error("short direct file write");
        offset += used;
    }

    if (paddedSize != content.size()) {
        FILE_END_OF_FILE_INFO end{};
        end.EndOfFile.QuadPart = content.size();
        if (!::SetFileInformationByHandle(handle, FileEndOfFileInfo, &end, sizeof(end)))
            ThrowOutputError("cannot trim direct file output");
    }
}
#endif

#ifndef INX_PLATFORM_WINDOWS
int OpenFileDirectory(const std::filesystem::path &path)
{
    return ::open(path.empty() ? "." : path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
}
#endif

} // namespace

detail::AtomicFileOutput::AtomicFileOutput(const std::filesystem::path &target, uint64_t completeBytes, bool namedStage)
    : m_path(namedStage ? NativeFilePath(target) : MakeAtomicTemporaryPath(NativeFilePath(target)))
#ifndef INX_PLATFORM_WINDOWS
      ,
      m_leaf(m_path.filename())
#endif
{
#ifdef INX_PLATFORM_WINDOWS
    if (UseDirectOutput(target, completeBytes))
        m_directBytes = completeBytes;
#else
    (void)completeBytes;
#endif
}

detail::AtomicFileOutput::~AtomicFileOutput()
{
    Close();
#ifdef INX_PLATFORM_WINDOWS
    if (m_private) {
        std::error_code ignored;
        std::filesystem::remove(m_path, ignored);
    }
#else
    if (m_private)
        ::unlinkat(m_directory, m_leaf.c_str(), 0);
    if (m_directory >= 0)
        ::close(m_directory);
#endif
}

size_t detail::AtomicFileOutput::PathStorageBytes() const noexcept
{
    auto bytes = m_path.native().capacity() * sizeof(std::filesystem::path::value_type);
#ifndef INX_PLATFORM_WINDOWS
    bytes += m_leaf.native().capacity() * sizeof(std::filesystem::path::value_type);
#endif
    return bytes;
}

bool detail::AtomicFileOutput::UsesDirectOutput() const noexcept
{
#ifdef INX_PLATFORM_WINDOWS
    return m_directBytes != 0;
#else
    return false;
#endif
}

void detail::AtomicFileOutput::Open(Access access)
{
    const bool readable = access == Access::ReadWrite;
#ifdef INX_PLATFORM_WINDOWS
    const DWORD flags =
        FILE_ATTRIBUTE_NORMAL | (UsesDirectOutput() ? FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED : 0);
    const DWORD accessMask = GENERIC_WRITE | (readable ? GENERIC_READ : 0);
    m_handle = ::CreateFileW(m_path.c_str(), accessMask, 0, nullptr, CREATE_NEW, flags, nullptr);
    if (m_handle == INVALID_HANDLE_VALUE)
        ThrowOutputError("cannot create private file output");
#else
    m_directory = OpenFileDirectory(m_path.parent_path());
    if (m_directory < 0)
        throw std::system_error(errno, std::generic_category(), "cannot open publication directory");
    m_descriptor = ::openat(m_directory, m_leaf.c_str(), (readable ? O_RDWR : O_WRONLY) | O_CREAT | O_EXCL | O_CLOEXEC,
                            readable ? 0600 : 0666);
    if (m_descriptor < 0)
        throw std::system_error(errno, std::generic_category(), "cannot create private file output");
#endif
    m_private = true;
}

void detail::AtomicFileOutput::Write(std::string_view content, const FrameWorkOwner *storageOwner)
{
#ifdef INX_PLATFORM_WINDOWS
    if (UsesDirectOutput()) {
        if (m_ranges || content.size() != m_directBytes)
            throw std::logic_error("direct file output requires one complete payload");
        m_ranges = std::make_shared<OverlappedFile>(m_handle, OverlappedFile::HandleOwnership::Borrowed);
        WriteDirectOutput(m_handle, *m_ranges, content, storageOwner);
        return;
    }
#else
    (void)storageOwner;
#endif
    while (!content.empty()) {
#ifdef INX_PLATFORM_WINDOWS
        DWORD count = 0;
        if (!::WriteFile(m_handle, content.data(), static_cast<DWORD>(std::min<size_t>(content.size(), MAXDWORD)),
                         &count, nullptr))
            ThrowOutputError("cannot write private file output");
#else
        const auto count = ::write(m_descriptor, content.data(), content.size());
        if (count < 0) {
            if (errno == EINTR)
                continue;
            throw std::system_error(errno, std::generic_category(), "cannot write private file output");
        }
#endif
        if (count == 0)
            throw std::runtime_error("private file write made no progress");
        content.remove_prefix(static_cast<size_t>(count));
    }
}

void detail::AtomicFileOutput::Seek(uint64_t offset)
{
#ifdef INX_PLATFORM_WINDOWS
    LARGE_INTEGER position{};
    position.QuadPart = static_cast<LONGLONG>(offset);
    if (!::SetFilePointerEx(m_handle, position, nullptr, FILE_BEGIN))
        ThrowOutputError("cannot seek private file output");
#else
    if (::lseek(m_descriptor, static_cast<off_t>(offset), SEEK_SET) == -1)
        throw std::system_error(errno, std::generic_category(), "cannot seek private file output");
#endif
}

void detail::AtomicFileOutput::Read(void *destination, size_t bytes)
{
    auto *cursor = static_cast<char *>(destination);
    while (bytes) {
#ifdef INX_PLATFORM_WINDOWS
        DWORD count = 0;
        if (!::ReadFile(m_handle, cursor, static_cast<DWORD>(std::min<size_t>(bytes, MAXDWORD)), &count, nullptr))
            ThrowOutputError("cannot read private file output");
#else
        const auto count = ::read(m_descriptor, cursor, bytes);
        if (count < 0) {
            if (errno == EINTR)
                continue;
            throw std::system_error(errno, std::generic_category(), "cannot read private file output");
        }
#endif
        if (!count)
            throw std::runtime_error("private file output ended before the requested range");
        cursor += count;
        bytes -= count;
    }
}

std::error_code detail::AtomicFileOutput::Close() noexcept
{
#ifdef INX_PLATFORM_WINDOWS
    if (m_handle == INVALID_HANDLE_VALUE)
        return {};
    const bool closed = ::CloseHandle(std::exchange(m_handle, INVALID_HANDLE_VALUE));
    const auto error =
        closed ? std::error_code{} : std::error_code(static_cast<int>(::GetLastError()), std::system_category());
    m_ranges.reset();
    return error;
#else
    if (m_descriptor >= 0 && ::close(std::exchange(m_descriptor, -1)) != 0)
        return {errno, std::generic_category()};
    return {};
#endif
}

void detail::AtomicFileOutput::Finish()
{
#ifdef INX_PLATFORM_WINDOWS
    if (!::FlushFileBuffers(m_handle))
        ThrowOutputError("cannot flush private file output");
#else
    if (::fsync(m_descriptor) != 0)
        throw std::system_error(errno, std::generic_category(), "cannot flush private file output");
#endif
    if (const auto error = Close())
        throw std::system_error(error, "cannot close private file output");
}

void detail::AtomicFileOutput::Remove()
{
    Close();
    if (m_private) {
#ifdef INX_PLATFORM_WINDOWS
        std::filesystem::remove(m_path);
#else
        if (::unlinkat(m_directory, m_leaf.c_str(), 0) != 0 && errno != ENOENT)
            throw std::system_error(errno, std::generic_category(), "cannot remove private file output");
#endif
        m_private = false;
    }
#ifndef INX_PLATFORM_WINDOWS
    if (m_directory >= 0)
        ::close(std::exchange(m_directory, -1));
#endif
}

void detail::AtomicFileOutput::Release() noexcept
{
    m_private = false;
#ifndef INX_PLATFORM_WINDOWS
    ::close(std::exchange(m_directory, -1));
#endif
}

std::error_code detail::AtomicFileOutput::Published() noexcept
{
    std::error_code error;
#ifndef INX_PLATFORM_WINDOWS
    if (::fsync(m_directory) != 0)
        error = std::error_code(errno, std::generic_category());
#endif
    Release();
    return error;
}

namespace
{
void HashFileRange(FileInput &file, uint64_t bytes, detail::FileFingerprint &hash)
{
    char buffer[64 * 1024];
    while (bytes) {
        const auto count = static_cast<size_t>(std::min<uint64_t>(bytes, sizeof(buffer)));
        file.Read(buffer, count);
        hash.Append(std::string_view(buffer, count));
        bytes -= count;
    }
}

AtomicFileState InspectDocumentPath(const std::filesystem::path &target, const std::string &path)
{
    std::error_code error;
    const bool exists = std::filesystem::exists(target, error);
    if (error)
        throw std::runtime_error("failed to inspect '" + path + "': " + error.message());
    if (!exists)
        return {};
    if (!std::filesystem::is_regular_file(target, error) || error)
        throw std::runtime_error("document target is not a regular file: '" + path + "'");
    const auto size = std::filesystem::file_size(target);
    const auto modified = std::filesystem::last_write_time(target);
    return {true, static_cast<uint64_t>(size), static_cast<int64_t>(modified.time_since_epoch().count()), 0};
}

[[noreturn]] void DocumentChanged()
{
    throw std::runtime_error("target changed outside the editor before atomic replace");
}

bool FlushDirectory(const std::filesystem::path &path, std::error_code &error)
{
#ifdef _WIN32
    (void)path;
    error.clear();
    return true;
#else
    const int descriptor = OpenFileDirectory(path);
    if (descriptor < 0) {
        error = std::error_code(errno, std::generic_category());
        return false;
    }
    const bool flushed = ::fsync(descriptor) == 0;
    const int flushError = flushed ? 0 : errno;
    ::close(descriptor);
    if (!flushed) {
        error = std::error_code(flushError, std::generic_category());
        return false;
    }
    error.clear();
    return true;
#endif
}

} // namespace

AtomicFileState CaptureAtomicFileState(const std::string &path)
{
    const std::filesystem::path target = NativeFilePath(ToFsPath(path));
    auto state = InspectDocumentPath(target, path);
    if (state.exists) {
        FileInput file(target);
        detail::FileFingerprint hash;
        state.size = file.Size();
        HashFileRange(file, state.size, hash);
        state.contentHash = hash.Value();
    }
    return state;
}

AtomicTextFileSnapshot ReadTextFileSnapshot(const std::string &path)
{
    const auto target = NativeFilePath(ToFsPath(path));
    FileInput input(target);
    const auto size = input.Size();
    AtomicTextFileSnapshot snapshot;
    snapshot.content.resize(static_cast<size_t>(size));
    input.Read(snapshot.content.data(), snapshot.content.size());
    const auto modified = std::filesystem::last_write_time(target);
    // Even if an atomic replacement happened after opening the stream, the
    // hash/size describe the consumed bytes, never a second read of the path.
    snapshot.state = {true, static_cast<uint64_t>(snapshot.content.size()),
                      static_cast<int64_t>(modified.time_since_epoch().count()),
                      detail::FileFingerprint::Of(snapshot.content)};
    return snapshot;
}

bool IsTransientReplaceError(const std::error_code &error)
{
#ifdef _WIN32
    return error.value() == ERROR_ACCESS_DENIED || error.value() == ERROR_SHARING_VIOLATION ||
           error.value() == ERROR_LOCK_VIOLATION;
#else
    (void)error;
    return false;
#endif
}

struct AtomicFileWriter::Impl
{
    using Output = std::variant<detail::AtomicFileOutput, AtomicPreparedFile>;

    Impl(std::string filename, AtomicWriteOptions writeOptions, const std::filesystem::path &destination,
         uint64_t completeBytes, bool stage)
        : path(std::move(filename)), options(std::move(writeOptions)), target(NativeFilePath(destination)),
          output(std::in_place_type<detail::AtomicFileOutput>, destination, completeBytes, stage), namedStage(stage)
    {
        std::get<detail::AtomicFileOutput>(output).Open();
    }

    Impl(std::string filename, AtomicWriteOptions writeOptions, const AtomicPreparedFile &prepared)
        : path(std::move(filename)), options(std::move(writeOptions)), target(NativeFilePath(ToFsPath(path))),
          output(prepared), written(prepared.state.size)
    {
    }

    std::string path;
    AtomicWriteOptions options;
    std::filesystem::path target;
    Output output;
    uint64_t written = 0;
    bool namedStage = false;
    detail::FileFingerprint writtenHash;
    std::unique_ptr<FileInput> expectedInput;
    FileInput::Revision expectedRevision{};
    uint64_t verified = 0;
#ifndef INX_PLATFORM_WINDOWS
    bool verifiedInSlices = false;
#endif
    detail::FileFingerprint expectedHash;
};

AtomicFileWriter::AtomicFileWriter(std::string path, AtomicWriteOptions options, uint64_t completeBytes,
                                   bool namedStage)
{
    if (options.mode == FileWriteMode::Create && options.expectedState && options.expectedState->exists)
        throw std::invalid_argument("create-only publication cannot revise an existing file state");
    const auto output = ToFsPath(path);
    m_impl = std::make_unique<Impl>(std::move(path), std::move(options), output, completeBytes, namedStage);
}

AtomicFileWriter::AtomicFileWriter(std::string path, AtomicWriteOptions options, const AtomicPreparedFile &prepared)
    : m_impl(std::make_unique<Impl>(std::move(path), std::move(options), prepared))
{
}

AtomicFileWriter::~AtomicFileWriter() = default;

uint64_t AtomicFileWriter::WrittenBytes() const noexcept
{
    return m_impl->written;
}
uint64_t AtomicFileWriter::VerifiedBytes() const noexcept
{
    return m_impl->verified;
}

void AtomicFileWriter::Append(std::string_view bytes, const FrameWorkOwner *storageOwner)
{
    auto &temporary = std::get<detail::AtomicFileOutput>(m_impl->output);
#ifdef _WIN32
    if (storageOwner && temporary.UsesDirectOutput() && JobSystem::IsAvailable() &&
        JobSystem::Get().GetWorkerCount() > 1) {
        auto &jobs = JobSystem::Get();
        auto &hash = m_impl->writtenHash;
        auto fingerprint =
            jobs.Schedule([&hash, bytes] { hash.Append(bytes); }, JobDomain::Default, JobPriority::Normal);
        std::exception_ptr failure;
        try {
            temporary.Write(bytes, storageOwner);
        } catch (...) {
            failure = std::current_exception();
        }
        jobs.WaitDependency(fingerprint);
        if (failure)
            std::rethrow_exception(failure);
    } else
#endif
    {
        temporary.Write(bytes, storageOwner);
        m_impl->writtenHash.Append(bytes);
    }
    m_impl->written += bytes.size();
}

void AtomicFileWriter::VerifyExpected(uint64_t maximumBytes)
{
    auto &output = *m_impl;
    const auto &expected = output.options.expectedState;
    if (output.options.mode != FileWriteMode::Replace || !expected || !expected->exists)
        return;
    if (!output.expectedInput) {
        const auto current = InspectDocumentPath(output.target, output.path);
        if (!current.exists || current.size != expected->size || current.modifiedNs != expected->modifiedNs)
            DocumentChanged();
        output.expectedInput = std::make_unique<FileInput>(output.target);
        output.expectedRevision = output.expectedInput->InspectRevision();
        if (output.expectedRevision.data.size != expected->size)
            DocumentChanged();
    }
    const auto bytes = (std::min)(maximumBytes, expected->size - output.verified);
#ifndef INX_PLATFORM_WINDOWS
    output.verifiedInSlices |= bytes < expected->size - output.verified;
#endif
    HashFileRange(*output.expectedInput, bytes, output.expectedHash);
    output.verified += bytes;
    if (output.verified == expected->size && output.expectedHash.Value() != expected->contentHash)
        DocumentChanged();
}

AtomicWriteReceipt AtomicFileWriter::Publish()
{
    const auto expected = m_impl->options.expectedState;
    const bool revise = m_impl->options.mode == FileWriteMode::Replace && expected && expected->exists;
    if (revise) {
        if (m_impl->verified != expected->size)
            throw std::logic_error("expected document has not been verified");
        VerifyExpected(0); // Empty inputs still need their file revision.
    }
    auto output = std::move(m_impl);
    auto *temporary = std::get_if<detail::AtomicFileOutput>(&output->output);
    const auto &options = output->options;
    const auto &target = output->target;
    AtomicFileState writtenState;
    std::filesystem::path preparedPath;
    if (temporary) {
        temporary->Finish();
        writtenState = {
            true, output->written,
            static_cast<int64_t>(std::filesystem::last_write_time(temporary->Path()).time_since_epoch().count()),
            output->writtenHash.Value()};
        if (output->namedStage) {
            temporary->Release();
            return {FilePublication::Written, writtenState};
        }
    } else {
        const auto &prepared = std::get<AtomicPreparedFile>(output->output);
        preparedPath = NativeFilePath(ToFsPath(prepared.path));
        writtenState = prepared.state;
    }
    const auto &source = temporary ? temporary->Path() : preparedPath;

    std::error_code publishError;
    std::optional<FilePublication> publication;
    constexpr unsigned kPublishAttempts = 8;
    for (unsigned attempt = 0; attempt < kPublishAttempts; ++attempt) {
        std::optional<FileInput> current;
        if (revise) {
            const auto state = InspectDocumentPath(target, output->path);
            if (!state.exists || state.size != expected->size || state.modifiedNs != expected->modifiedNs)
                DocumentChanged();
            current.emplace(target, FileInput::Sharing::Snapshot);
            if (!(current->InspectRevision() == output->expectedRevision))
                DocumentChanged();
#ifndef INX_PLATFORM_WINDOWS
            if (output->verifiedInSlices) {
                // Coarse inode timestamps can be identical across a write and
                // a restored mtime. Reread at publication after a paced check.
                detail::FileFingerprint fingerprint;
                HashFileRange(*current, expected->size, fingerprint);
                if (fingerprint.Value() != expected->contentHash ||
                    !(current->InspectRevision() == output->expectedRevision))
                    DocumentChanged();
            }
#endif
        }
        if (options.mode == FileWriteMode::Create || (expected && !expected->exists)) {
            publication = PublishFileIfAbsent(source, target, publishError);
            if (publication == FilePublication::AlreadyExists && options.mode == FileWriteMode::Replace)
                DocumentChanged();
        } else if (ReplaceFileAtomically(source, target, publishError)) {
            publication = FilePublication::Written;
        }
        if (publication)
            break;
        if (!IsTransientReplaceError(publishError) || attempt + 1 == kPublishAttempts)
            throw std::system_error(publishError, "atomic write failed for '" + output->path + "'");
        std::this_thread::sleep_for(std::chrono::milliseconds(2u << attempt));
    }
    if (*publication == FilePublication::AlreadyExists)
        return {FilePublication::AlreadyExists, std::nullopt};
    if (temporary) {
        if (const auto error = temporary->Published())
            throw std::system_error(error, "file published but parent directory flush failed");
    }
    return {FilePublication::Written, writtenState};
}

std::optional<AtomicWriteReceipt> WriteTextFileAtomically(const std::string &path, std::string_view content,
                                                          std::string &error, AtomicWriteOptions options)
{
    try {
        AtomicFileWriter writer(path, std::move(options));
        writer.Append(content);
        writer.VerifyExpected((std::numeric_limits<uint64_t>::max)());
        auto receipt = writer.Publish();
        error.clear();
        return receipt;
    } catch (const std::exception &exception) {
        error = exception.what();
        return std::nullopt;
    }
}

bool RemoveFileDurably(const std::string &path, std::string &error)
{
    const std::filesystem::path target = NativeFilePath(ToFsPath(path));
    std::error_code removeError;
    const bool removed = std::filesystem::remove(target, removeError);
    if (removeError) {
        error = "failed to remove file: " + removeError.message();
        return false;
    }
    if (!removed) {
        error.clear();
        return true;
    }

    std::error_code directoryFlushError;
    if (!FlushDirectory(target.parent_path(), directoryFlushError)) {
        error = "file removed but parent directory flush failed: " + directoryFlushError.message();
        return false;
    }
    error.clear();
    return true;
}

void FlushFileDirectory(const std::filesystem::path &path)
{
    std::error_code error;
    if (!FlushDirectory(NativeFilePath(path), error))
        throw std::system_error(error, "failed to flush directory '" + FromFsPath(path) + "'");
}

} // namespace infernux
