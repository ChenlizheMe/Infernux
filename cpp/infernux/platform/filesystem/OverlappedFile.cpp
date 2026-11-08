#include "OverlappedFile.h"

#ifdef INX_PLATFORM_WINDOWS
#include "InxPath.h"
#include <stdexcept>
#include <system_error>
#include <utility>

namespace infernux
{
OverlappedFile::OverlappedFile(FileInput &source) : OverlappedFile(source.OpenOverlapped(), HandleOwnership::Owned)
{
}

OverlappedFile::OverlappedFile(HANDLE file, HandleOwnership ownership)
    : m_file(file), m_ownership(ownership),
      m_completion(::CreateThreadpoolIo(m_file, OverlappedFileRequest::Complete, nullptr, nullptr))
{
    if (!m_completion) {
        const auto error = ::GetLastError();
        if (m_ownership == HandleOwnership::Owned)
            ::CloseHandle(m_file);
        throw std::system_error(static_cast<int>(error), std::system_category(),
                                "Cannot bind file to the completion pool");
    }
}

OverlappedFile::~OverlappedFile()
{
    if (m_ownership == HandleOwnership::Owned)
        ::CloseHandle(m_file);
    ::CloseThreadpoolIo(m_completion);
}

OverlappedFileRequest::OverlappedFileRequest(std::shared_ptr<OverlappedFile> file, uint64_t offset, size_t bytes,
                                             bool write, Callback callback)
    : OVERLAPPED{}, m_file(std::move(file)), m_bytes(bytes), m_write(write), m_callback(std::move(callback))
{
    Offset = static_cast<DWORD>(offset);
    OffsetHigh = static_cast<DWORD>(offset >> 32);
}

void OverlappedFileRequest::Cancel() noexcept
{
    ::CancelIoEx(m_file->m_file, this);
}

void CALLBACK OverlappedFileRequest::Complete(PTP_CALLBACK_INSTANCE, void *, void *overlapped, ULONG error,
                                              ULONG_PTR transferred, PTP_IO) noexcept
{
    auto &request = *static_cast<OverlappedFileRequest *>(static_cast<OVERLAPPED *>(overlapped));
    auto callback = std::move(request.m_callback);
    auto accepted = std::move(request.m_accepted);
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - request.m_submitted)
            .count();
    const bool endOfFile =
        !request.m_write && (error == ERROR_HANDLE_EOF || (error == ERROR_SUCCESS && transferred != request.m_bytes));
    callback({static_cast<size_t>(transferred),
              {static_cast<int>(error), std::system_category()},
              endOfFile,
              static_cast<uint64_t>(elapsed)});
}

OverlappedFile::Submission OverlappedFile::Read(uint64_t offset, void *destination, size_t bytes,
                                                OverlappedFileRequest::Callback callback)
{
    return Submit(offset, destination, bytes, false, std::move(callback));
}

OverlappedFile::Submission OverlappedFile::Write(uint64_t offset, const void *source, size_t bytes,
                                                 OverlappedFileRequest::Callback callback)
{
    return Submit(offset, source, bytes, true, std::move(callback));
}

OverlappedFile::Submission OverlappedFile::Submit(uint64_t offset, const void *buffer, size_t bytes, bool write,
                                                  OverlappedFileRequest::Callback callback)
{
    if (!bytes || bytes > MAXDWORD)
        throw std::invalid_argument("Invalid asynchronous file range size");
    auto request = std::shared_ptr<OverlappedFileRequest>(
        new OverlappedFileRequest(shared_from_this(), offset, bytes, write, std::move(callback)));
    request->m_accepted = request;
    request->m_submitted = std::chrono::steady_clock::now();
    ::StartThreadpoolIo(m_completion);
    const bool submitted =
        write ? ::WriteFile(m_file, buffer, static_cast<DWORD>(bytes), nullptr, request.get())
              : ::ReadFile(m_file, const_cast<void *>(buffer), static_cast<DWORD>(bytes), nullptr, request.get());
    if (submitted)
        return {std::move(request), false}; // Success also posts a completion packet.
    const auto error = ::GetLastError();
    if (error == ERROR_IO_PENDING)
        return {std::move(request), true};
    ::CancelThreadpoolIo(m_completion);
    request->m_accepted.reset();
    if (!write && error == ERROR_HANDLE_EOF)
        throw std::runtime_error("File input ended before the requested range");
    throw std::system_error(static_cast<int>(error), std::system_category(),
                            write ? "Cannot submit asynchronous file output" : "Cannot submit asynchronous file input");
}
} // namespace infernux
#endif
