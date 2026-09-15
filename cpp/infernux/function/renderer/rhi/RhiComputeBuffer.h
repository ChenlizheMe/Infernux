#pragma once

#include "RhiBuffer.h"
#include "RhiComputeHost.h"
#include "RhiSubmission.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace infernux::rhi
{

class ComputeRecordingContext;
class ComputeBuffer;
class ComputeReadback;

struct ComputeBufferRead
{
    std::shared_ptr<ComputeBuffer> buffer;
    uint64_t offset = 0;
    uint64_t byteSize = 0;
};

std::vector<std::vector<uint8_t>> ReadComputeBatch(ComputeHost &host, std::vector<ComputeBufferRead> reads);

enum class ComputeScalarType : uint8_t
{
    Float32,
    Int32,
    UInt32,
};

struct ComputeBufferDesc
{
    uint64_t elementCount = 0;
    ComputeScalarType scalarType = ComputeScalarType::Float32;
    uint8_t lanes = 1;

    [[nodiscard]] uint32_t GetScalarBytes() const noexcept
    {
        return 4;
    }
    [[nodiscard]] uint32_t GetElementStride() const noexcept
    {
        return GetScalarBytes() * lanes;
    }
    [[nodiscard]] uint64_t GetByteSize() const noexcept
    {
        return elementCount * GetElementStride();
    }
};

/// Engine-owned, device-local storage used by public compute buffers. Upload
/// and readback memory are staging details; kernels and renderers consume the
/// single resident buffer handle.
class ComputeBuffer final : public std::enable_shared_from_this<ComputeBuffer>
{
  public:
    ComputeBuffer(ComputeHost &host, ComputeBufferDesc desc);
    ~ComputeBuffer();

    ComputeBuffer(const ComputeBuffer &) = delete;
    ComputeBuffer &operator=(const ComputeBuffer &) = delete;

    [[nodiscard]] uint64_t GetByteSize() const noexcept;
    [[nodiscard]] const ComputeBufferDesc &GetDesc() const noexcept;
    [[nodiscard]] BufferHandle GetBuffer() const noexcept;
    [[nodiscard]] ComputeHost &GetHost() noexcept;
    [[nodiscard]] const ComputeHost &GetHost() const noexcept;
    void SetData(uint64_t offset, const void *data, uint64_t byteSize);
    [[nodiscard]] std::vector<uint8_t> GetData(uint64_t offset, uint64_t byteSize);
    [[nodiscard]] std::shared_ptr<ComputeReadback> GetDataAsync(uint64_t offset, uint64_t byteSize);

    /// Stage one host update so a compute batch can record its transfer and
    /// dispatches into the same queue submission.
    void PrepareUpload(uint64_t offset, const void *data, uint64_t byteSize);
    void RecordPreparedUpload(ComputeRecordingContext &context, uint64_t offset, uint64_t byteSize) const;
    void AttachUpload(SubmissionTicket ticket) noexcept;
    void AttachWrite(SubmissionTicket ticket) noexcept;
    [[nodiscard]] SubmissionTicket GetLastWriteSubmission() const noexcept;

    /// Prepare, record, and finish a synchronous readback owned by an engine
    /// compute batch. These are RHI plumbing methods; public callers use the
    /// typed inx.buffer boundary rather than staging resources directly.
    void PrepareReadback(uint64_t offset, uint64_t byteSize);
    void RecordPreparedReadback(ComputeRecordingContext &context, uint64_t offset, uint64_t byteSize) const;
    [[nodiscard]] std::vector<uint8_t> FinishReadback(uint64_t offset, uint64_t byteSize);

  private:
    [[nodiscard]] BufferResource &EnsureUploadBuffer();
    [[nodiscard]] BufferResource &EnsureReadbackBuffer();
    void ValidateRange(uint64_t offset, uint64_t byteSize) const;

    ComputeHost m_host;
    ComputeBufferDesc m_desc;
    std::unique_ptr<BufferResource> m_storage;
    std::unique_ptr<BufferResource> m_upload;
    std::unique_ptr<BufferResource> m_readback;
    SubmissionTicket m_uploadTicket;
    SubmissionTicket m_lastWriteSubmission;
};

/// One explicit GPU-to-CPU transfer. The staging allocation and source stay
/// alive until its exact compute-queue ticket completes.
class ComputeReadback final
{
  public:
    ComputeReadback(ComputeHost &host, std::shared_ptr<ComputeBuffer> source, uint64_t offset, uint64_t byteSize);
    ~ComputeReadback();

    ComputeReadback(const ComputeReadback &) = delete;
    ComputeReadback &operator=(const ComputeReadback &) = delete;

    [[nodiscard]] bool IsComplete();
    void Wait();
    [[nodiscard]] std::vector<uint8_t> GetData();
    [[nodiscard]] uint64_t GetByteSize() const noexcept
    {
        return m_byteSize;
    }

  private:
    ComputeHost m_host;
    std::shared_ptr<ComputeBuffer> m_source;
    std::unique_ptr<BufferResource> m_staging;
    SubmissionTicket m_ticket;
    uint64_t m_byteSize = 0;
    bool m_complete = false;
};

} // namespace infernux::rhi
