#pragma once

#include "RenderCoreApi.h"
#include "RhiBuffer.h"
#include "RhiComputeHost.h"
#include "RhiSubmission.h"
#include <core/math/Intervals.h>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace infernux::rhi
{

class ComputeRecordingContext;
class ComputeBuffer;
class ComputeReadback;
class ComputeKernel;
class PreparedComputeBatch;
class ComputeBufferState;
class ComputeBufferAllocation;
using ComputeBufferBindings = std::unordered_map<const ComputeBuffer *, std::shared_ptr<ComputeBufferAllocation>>;

/// Buffer storage retained by compute work, not the driver's memory-block size.
struct ComputeStorageBlock
{
    BufferHandle buffer;
    uint64_t byteSize;
};

struct ComputeBufferRead
{
    std::shared_ptr<ComputeBuffer> buffer;
    uint64_t offset = 0;
    uint64_t byteSize = 0;
};

INFERNUX_RENDER_CORE_API std::vector<std::vector<uint8_t>> ReadComputeBatch(ComputeHost &host,
                                                                            std::vector<ComputeBufferRead> reads);

/// One physical allocation and its accepted value facts. Prepared work retains
/// this identity, independently of the logical buffer used to author the work.
class ComputeBufferAllocation final
{
  public:
    INFERNUX_RENDER_CORE_API ComputeBufferAllocation(Device &device, const BufferDesc &desc,
                                                     bool graphWritable = false);
    [[nodiscard]] BufferHandle GetBuffer() const noexcept
    {
        return m_storage.GetBuffer();
    }
    [[nodiscard]] uint64_t GetByteSize() const noexcept
    {
        return m_storage.GetByteSize();
    }
    [[nodiscard]] SubmissionTicket GetLastWriteSubmission() const noexcept
    {
        return m_lastWriteSubmission;
    }
    [[nodiscard]] QueueRole GetLastWriterQueue() const noexcept
    {
        return m_lastWriterQueue;
    }
    [[nodiscard]] bool IsGraphWritable() const noexcept
    {
        return m_graphWritable;
    }
    [[nodiscard]] bool IsReadProtected() const noexcept
    {
        return m_readers != 0;
    }
    [[nodiscard]] const BufferDesc &GetStorageDesc() const noexcept
    {
        return m_desc;
    }
    /// Internal publication after graph prefix validation and real GPU acceptance.
    /// The completion queue may differ from the queue that actually wrote storage.
    INFERNUX_RENDER_CORE_API void PublishGraphValue(SubmissionTicket completion, QueueRole writer,
                                                    intervals::Set initialized, intervals::Set written) noexcept;
    [[nodiscard]] INFERNUX_RENDER_CORE_API bool HasInitializedRange(uint64_t offset, uint64_t byteSize) const;
    INFERNUX_RENDER_CORE_API void AppendRetainedStorage(std::vector<ComputeStorageBlock> &blocks) const;

  private:
    friend class ComputeBuffer;
    friend class ComputeKernel;
    friend class PreparedComputeBatch;
    friend class ComputeBufferState;
    ComputeBufferAllocation(Device &device, BufferHandle buffer, const BufferDesc &desc, bool graphWritable);
    [[nodiscard]] static std::shared_ptr<ComputeBufferAllocation> CreatePrivate(Device &device, const BufferDesc &desc,
                                                                                bool graphWritable, bool managed);
    void RecordUpload(ComputeRecordingContext &context, uint64_t offset, const void *data, uint64_t byteSize) const;
    BufferResource m_storage;
    BufferDesc m_desc;
    SubmissionTicket m_lastWriteSubmission;
    QueueRole m_lastWriterQueue = QueueRole::Compute;
    bool m_graphWritable;
    intervals::Set m_initialized;
    intervals::Set m_written;
    uint32_t m_readers = 0;
};

/// The logical value endpoint shared by recorded work. It outlives public
/// wrappers, so accepting one queued writer also advances later queued work.
class ComputeBufferState final
{
  public:
    INFERNUX_RENDER_CORE_API ComputeBufferState(ComputeBufferDesc desc,
                                                std::shared_ptr<ComputeBufferAllocation> allocation, bool frozen);
    INFERNUX_RENDER_CORE_API ~ComputeBufferState();
    ComputeBufferState(const ComputeBufferState &) = delete;
    ComputeBufferState &operator=(const ComputeBufferState &) = delete;
    [[nodiscard]] const ComputeBufferDesc &GetDesc() const noexcept
    {
        return m_desc;
    }
    [[nodiscard]] std::shared_ptr<ComputeBufferAllocation> GetAllocation() const noexcept
    {
        return m_allocation;
    }
    [[nodiscard]] bool IsFrozen() const noexcept
    {
        return m_frozen;
    }

  private:
    friend class PreparedComputeBatch;
    ComputeBufferDesc m_desc;
    std::shared_ptr<ComputeBufferAllocation> m_allocation;
    bool m_frozen;
};

/// Engine-owned, device-local storage used by public compute buffers. Upload
/// and readback memory are staging details; kernels and renderers consume the
/// single resident buffer handle.
class ComputeBuffer final : public std::enable_shared_from_this<ComputeBuffer>
{
  public:
    INFERNUX_RENDER_CORE_API ComputeBuffer(ComputeHost &host, ComputeBufferDesc desc);
    /// A budgeted target stays submission-owned until its completed value is handed off.
    [[nodiscard]] INFERNUX_RENDER_CORE_API static std::shared_ptr<ComputeBuffer>
    TryCreatePrivate(ComputeHost &host, ComputeBufferDesc desc);
    /// Graph assembly publishes these owners before their first submission.
    [[nodiscard]] INFERNUX_RENDER_CORE_API static std::shared_ptr<ComputeBuffer>
    TryCreateGraphBuffer(ComputeHost &host, ComputeBufferDesc desc);
    INFERNUX_RENDER_CORE_API ~ComputeBuffer();

    ComputeBuffer(const ComputeBuffer &) = delete;
    ComputeBuffer &operator=(const ComputeBuffer &) = delete;

    [[nodiscard]] INFERNUX_RENDER_CORE_API uint64_t GetByteSize() const noexcept;
    [[nodiscard]] INFERNUX_RENDER_CORE_API const ComputeBufferDesc &GetDesc() const noexcept;
    [[nodiscard]] INFERNUX_RENDER_CORE_API BufferHandle GetBuffer() const noexcept;
    [[nodiscard]] INFERNUX_RENDER_CORE_API ComputeHost &GetHost() noexcept;
    [[nodiscard]] INFERNUX_RENDER_CORE_API const ComputeHost &GetHost() const noexcept;
    [[nodiscard]] std::shared_ptr<ComputeBufferAllocation> GetAllocation() const noexcept
    {
        return m_state->GetAllocation();
    }
    [[nodiscard]] INFERNUX_RENDER_CORE_API std::shared_ptr<ComputeBuffer> Freeze();
    [[nodiscard]] INFERNUX_RENDER_CORE_API std::shared_ptr<ComputeBuffer> FreezeScalars();
    INFERNUX_RENDER_CORE_API void PublishStorage();
    [[nodiscard]] bool IsFrozen() const noexcept
    {
        return m_state->IsFrozen();
    }
    INFERNUX_RENDER_CORE_API void AppendRetainedStorage(std::vector<ComputeStorageBlock> &blocks) const;
    INFERNUX_RENDER_CORE_API void SetData(uint64_t offset, const void *data, uint64_t byteSize);
    [[nodiscard]] INFERNUX_RENDER_CORE_API std::vector<uint8_t> GetData(uint64_t offset, uint64_t byteSize);

    [[nodiscard]] INFERNUX_RENDER_CORE_API SubmissionTicket GetLastWriteSubmission() const noexcept;
    /// Accepted compute writes cover this range; consumers must still order
    /// after the source ticket. Coverage is not GPU completion or freshness.
    [[nodiscard]] INFERNUX_RENDER_CORE_API bool HasInitializedRange(uint64_t offset, uint64_t byteSize) const;

  private:
    friend class ComputeKernel;
    friend class ComputeReadback;
    friend class PreparedComputeBatch;
    [[nodiscard]] static std::shared_ptr<ComputeBuffer> TryCreate(ComputeHost &host, ComputeBufferDesc desc,
                                                                  BufferUsageFlags usage, bool graphWritable);
    ComputeBuffer(ComputeHost host, std::shared_ptr<ComputeBufferState> state);
    void ValidateRange(uint64_t offset, uint64_t byteSize) const;

    ComputeHost m_host;
    std::shared_ptr<ComputeBufferState> m_state;
};

/// One prepared tail read. No submission occurs during construction. A batch
/// publishes its exact ticket on acceptance and retains separate GPU storage
/// ownership, so cancelling the consumer never releases an in-flight copy.
class ComputeReadback final
{
  public:
    ~ComputeReadback() = default;

    ComputeReadback(const ComputeReadback &) = delete;
    ComputeReadback &operator=(const ComputeReadback &) = delete;

    [[nodiscard]] INFERNUX_RENDER_CORE_API bool IsComplete();
    [[nodiscard]] bool IsCancelled() const noexcept
    {
        return m_cancelled;
    }
    INFERNUX_RENDER_CORE_API bool Cancel() noexcept;
    INFERNUX_RENDER_CORE_API void Wait();
    /// Materialize once, then release the consumer's staging lease.
    INFERNUX_RENDER_CORE_API void ReadInto(void *destination, uint64_t byteSize);
    [[nodiscard]] INFERNUX_RENDER_CORE_API std::vector<uint8_t> GetData();
    [[nodiscard]] uint64_t GetByteSize() const noexcept
    {
        return m_byteSize;
    }
    INFERNUX_RENDER_CORE_API void AppendRetainedStorage(std::vector<ComputeStorageBlock> &blocks) const;
    [[nodiscard]] SubmissionTicket GetSubmission() const noexcept
    {
        return m_ticket;
    }
    [[nodiscard]] SubmissionTicket GetSourceWriteSubmission() const noexcept
    {
        return m_sourceWriteSubmission;
    }
    [[nodiscard]] BufferHandle GetSourceBuffer() const noexcept
    {
        return m_sourceBuffer;
    }
    [[nodiscard]] uint64_t GetSourceOffset() const noexcept
    {
        return m_offset;
    }
    [[nodiscard]] ComputeHost &GetHost() noexcept
    {
        return m_host;
    }

  private:
    friend class ComputeKernel;
    friend class PreparedComputeBatch;
    ComputeReadback(ComputeHost &host, std::shared_ptr<ComputeBuffer> source, uint64_t offset, uint64_t byteSize);
    ComputeHost m_host;
    struct Storage
    {
        std::shared_ptr<ComputeBufferAllocation> source;
        std::shared_ptr<BufferResource> staging;
        void AppendRetainedStorage(std::vector<ComputeStorageBlock> &blocks) const;
    };
    void Record(ComputeRecordingContext &context) const;
    [[nodiscard]] std::shared_ptr<BufferResource> AllocateStaging(bool managed);
    void Rebind(std::shared_ptr<ComputeBufferAllocation> allocation);
    [[nodiscard]] std::shared_ptr<Storage> Attach(SubmissionTicket ticket) noexcept;
    std::shared_ptr<Storage> m_storage;
    std::shared_ptr<ComputeBufferState> m_source;
    SubmissionTicket m_ticket;
    SubmissionTicket m_sourceWriteSubmission;
    BufferHandle m_sourceBuffer;
    uint64_t m_offset = 0;
    uint64_t m_byteSize = 0;
    bool m_complete = false;
    bool m_cancelled = false;
};

} // namespace infernux::rhi
