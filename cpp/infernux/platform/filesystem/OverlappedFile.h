#pragma once

#include <core/config/InxPlatform.h>
#include <cstddef>
#include <cstdint>
#include <system_error>

namespace infernux
{
struct FileRangeCompletion
{
    size_t bytes;
    std::error_code error;
    bool endOfFile;
    uint64_t completionNs;
};
} // namespace infernux

#ifdef INX_PLATFORM_WINDOWS
#include <chrono>
#include <functional>
#include <memory>

namespace infernux
{
class FileInput;

class OverlappedFile;

/// An accepted file range retains its completion closure and the open file.
class OverlappedFileRequest final : private OVERLAPPED
{
  public:
    void Cancel() noexcept;

  private:
    friend class OverlappedFile;
    using Callback = std::function<void(FileRangeCompletion)>;
    OverlappedFileRequest(std::shared_ptr<OverlappedFile> file, uint64_t offset, size_t bytes, bool write,
                          Callback callback);
    static void CALLBACK Complete(PTP_CALLBACK_INSTANCE, void *, void *, ULONG, ULONG_PTR, PTP_IO) noexcept;

    std::shared_ptr<OverlappedFile> m_file;
    std::shared_ptr<OverlappedFileRequest> m_accepted;
    size_t m_bytes;
    bool m_write;
    Callback m_callback;
    std::chrono::steady_clock::time_point m_submitted;
};

/// Uses the Windows completion pool. Reads belong to the opened file object
/// even when the pathname is later atomically replaced.
class OverlappedFile final : public std::enable_shared_from_this<OverlappedFile>
{
  public:
    enum class HandleOwnership
    {
        Borrowed,
        Owned
    };

    explicit OverlappedFile(FileInput &source);
    OverlappedFile(HANDLE file, HandleOwnership ownership);
    ~OverlappedFile();
    OverlappedFile(const OverlappedFile &) = delete;
    OverlappedFile &operator=(const OverlappedFile &) = delete;

    struct Submission
    {
        std::shared_ptr<OverlappedFileRequest> request;
        bool pending;
    };
    Submission Read(uint64_t offset, void *destination, size_t bytes, OverlappedFileRequest::Callback callback);
    Submission Write(uint64_t offset, const void *source, size_t bytes, OverlappedFileRequest::Callback callback);

  private:
    friend class OverlappedFileRequest;
    Submission Submit(uint64_t offset, const void *buffer, size_t bytes, bool write,
                      OverlappedFileRequest::Callback callback);
    HANDLE m_file;
    HandleOwnership m_ownership;
    PTP_IO m_completion;
};
} // namespace infernux
#endif
