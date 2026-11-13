#include "RhiComputeBuffer.h"
#include "RhiComputeKernel.h"

#include <limits>
#include <stdexcept>

namespace infernux::rhi
{
namespace
{
constexpr QueueAccessFlags kComputeBufferQueues =
    QueueAccessFlags::Graphics | QueueAccessFlags::Compute | QueueAccessFlags::Transfer;

BufferHandle CreateComputeBuffer(Device &device, const BufferDesc &desc)
{
    const auto handle = device.CreateBuffer(desc);
    if (!handle.IsValid())
        throw std::runtime_error("Failed to allocate compute buffer storage");
    return handle;
}

BufferDesc ComputeStorageDesc(const ComputeHost &host, ComputeBufferDesc desc, BufferUsageFlags usage)
{
    host.RequireAlive();
    if (desc.elementCount == 0)
        throw std::invalid_argument("Compute buffer element count must be positive");
    if (desc.lanes == 0 || desc.lanes > 4)
        throw std::invalid_argument("Compute buffer lanes must be between one and four");
    if (desc.elementCount > (std::numeric_limits<uint64_t>::max)() / desc.GetElementStride())
        throw std::overflow_error("Compute buffer byte size exceeds the addressable range");
    return {desc.GetByteSize(),
            BufferUsageFlags::Storage | BufferUsageFlags::Uniform | BufferUsageFlags::Vertex |
                BufferUsageFlags::TransferSource | BufferUsageFlags::TransferDestination | desc.usage | usage,
            BufferMemory::DeviceLocal,
            nullptr,
            0,
            kComputeBufferQueues};
}

} // namespace

ComputeBufferAllocation::ComputeBufferAllocation(Device &device, const BufferDesc &desc, bool graphWritable)
    : ComputeBufferAllocation(device, CreateComputeBuffer(device, desc), desc, graphWritable)
{
}

ComputeBufferAllocation::ComputeBufferAllocation(Device &device, BufferHandle buffer, const BufferDesc &desc,
                                                 bool graphWritable)
    : m_storage(device, buffer, desc.byteSize), m_desc(desc), m_graphWritable(graphWritable)
{
    // The descriptor describes future storage; initial payload belongs only to this creation.
    m_desc.initialData = nullptr;
    m_desc.initialDataBytes = 0;
}

std::shared_ptr<ComputeBufferAllocation> ComputeBufferAllocation::CreatePrivate(Device &device, const BufferDesc &desc,
                                                                                bool graphWritable, bool managed)
{
    const auto handle = device.CreatePrivateBuffer(desc, managed);
    if (!handle)
        return {};
    if (!handle->IsValid())
        throw std::runtime_error("Failed to allocate private compute storage");
    std::unique_ptr<ComputeBufferAllocation> allocation;
    try {
        allocation.reset(new ComputeBufferAllocation(device, *handle, desc, graphWritable));
    } catch (...) {
        device.Release(*handle);
        throw;
    }
    return allocation;
}

void ComputeBufferAllocation::PublishGraphValue(SubmissionTicket completion, QueueRole writer,
                                                intervals::Set initialized, intervals::Set written) noexcept
{
    m_lastWriteSubmission = completion;
    m_lastWriterQueue = writer;
    m_initialized = std::move(initialized);
    m_written = std::move(written);
}

ComputeBufferState::ComputeBufferState(ComputeBufferDesc desc, std::shared_ptr<ComputeBufferAllocation> allocation,
                                       bool frozen)
    : m_desc(desc), m_allocation(std::move(allocation)), m_frozen(frozen)
{
    if (m_frozen)
        ++m_allocation->m_readers;
}

ComputeBufferState::~ComputeBufferState()
{
    if (m_frozen)
        --m_allocation->m_readers;
}

ComputeBuffer::ComputeBuffer(ComputeHost &host, ComputeBufferDesc desc)
    : m_host(host), m_state(std::make_shared<ComputeBufferState>(
                        desc,
                        std::make_shared<ComputeBufferAllocation>(
                            host.device, ComputeStorageDesc(host, desc, BufferUsageFlags::None)),
                        false))
{
}

std::shared_ptr<ComputeBuffer> ComputeBuffer::TryCreatePrivate(ComputeHost &host, ComputeBufferDesc desc)
{
    return TryCreate(host, desc, BufferUsageFlags::None, false);
}

std::shared_ptr<ComputeBuffer> ComputeBuffer::TryCreateGraphBuffer(ComputeHost &host, ComputeBufferDesc desc)
{
    return TryCreate(host, desc, BufferUsageFlags::None, true);
}

std::shared_ptr<ComputeBuffer> ComputeBuffer::TryCreate(ComputeHost &host, ComputeBufferDesc desc,
                                                        BufferUsageFlags usage, bool graphWritable)
{
    auto allocation =
        ComputeBufferAllocation::CreatePrivate(host.device, ComputeStorageDesc(host, desc, usage), graphWritable, true);
    if (!allocation)
        return {};
    return std::shared_ptr<ComputeBuffer>(
        new ComputeBuffer(host, std::make_shared<ComputeBufferState>(desc, std::move(allocation), false)));
}

ComputeBuffer::ComputeBuffer(ComputeHost host, std::shared_ptr<ComputeBufferState> state)
    : m_host(std::move(host)), m_state(std::move(state))
{
}

void ComputeBuffer::PublishStorage()
{
    m_host.RequireAlive();
    m_host.device.PublishPrivateBuffer(GetBuffer());
}

std::shared_ptr<ComputeBuffer> ComputeBuffer::Freeze()
{
    if (IsFrozen())
        return shared_from_this();
    return std::shared_ptr<ComputeBuffer>(
        new ComputeBuffer(m_host, std::make_shared<ComputeBufferState>(GetDesc(), GetAllocation(), true)));
}

std::shared_ptr<ComputeBuffer> ComputeBuffer::FreezeScalars()
{
    auto layout = GetDesc();
    layout.elementCount *= layout.lanes;
    layout.lanes = 1;
    return std::shared_ptr<ComputeBuffer>(
        new ComputeBuffer(m_host, std::make_shared<ComputeBufferState>(layout, GetAllocation(), true)));
}

ComputeBuffer::~ComputeBuffer() = default;

uint64_t ComputeBuffer::GetByteSize() const noexcept
{
    return GetAllocation()->GetByteSize();
}

const ComputeBufferDesc &ComputeBuffer::GetDesc() const noexcept
{
    return m_state->GetDesc();
}

BufferHandle ComputeBuffer::GetBuffer() const noexcept
{
    return GetAllocation()->GetBuffer();
}

ComputeHost &ComputeBuffer::GetHost() noexcept
{
    return m_host;
}

const ComputeHost &ComputeBuffer::GetHost() const noexcept
{
    return m_host;
}

void ComputeBuffer::AppendRetainedStorage(std::vector<ComputeStorageBlock> &blocks) const
{
    GetAllocation()->AppendRetainedStorage(blocks);
}

void ComputeBufferAllocation::AppendRetainedStorage(std::vector<ComputeStorageBlock> &blocks) const
{
    blocks.push_back({GetBuffer(), GetByteSize()});
}

void ComputeBuffer::ValidateRange(uint64_t offset, uint64_t byteSize) const
{
    const auto capacity = GetByteSize();
    if (byteSize == 0 || offset % 4 != 0 || byteSize % 4 != 0 || offset > capacity || byteSize > capacity - offset)
        throw std::out_of_range("Compute buffer transfer range is invalid");
}

void ComputeBuffer::SetData(uint64_t offset, const void *data, uint64_t byteSize)
{
    ValidateRange(offset, byteSize);
    if (!data)
        throw std::invalid_argument("Compute buffer source cannot be null");
    const auto *bytes = static_cast<const uint8_t *>(data);
    (void)PrepareComputeBatch(m_host, {{shared_from_this(), offset, {bytes, bytes + byteSize}}}, {})->Submit();
}

void ComputeBufferAllocation::RecordUpload(ComputeRecordingContext &context, uint64_t offset, const void *data,
                                           uint64_t byteSize) const
{
    if (!context.Transfer().UpdateBuffer(GetBuffer(), offset, data, byteSize))
        throw std::runtime_error("Failed to record compute buffer upload");
}

SubmissionTicket ComputeBuffer::GetLastWriteSubmission() const noexcept
{
    return GetAllocation()->GetLastWriteSubmission();
}

bool ComputeBuffer::HasInitializedRange(uint64_t offset, uint64_t byteSize) const
{
    ValidateRange(offset, byteSize);
    return GetAllocation()->HasInitializedRange(offset, byteSize);
}

bool ComputeBufferAllocation::HasInitializedRange(uint64_t offset, uint64_t byteSize) const
{
    return byteSize && offset <= GetByteSize() && byteSize <= GetByteSize() - offset &&
           intervals::Contains(m_initialized, offset, offset + byteSize);
}

std::vector<uint8_t> ComputeBuffer::GetData(uint64_t offset, uint64_t byteSize)
{
    auto result = ReadComputeBatch(m_host, {{shared_from_this(), offset, byteSize}});
    return std::move(result.front());
}

ComputeReadback::ComputeReadback(ComputeHost &host, std::shared_ptr<ComputeBuffer> source, uint64_t offset,
                                 uint64_t byteSize)
    : m_host(host), m_storage(std::make_shared<Storage>()), m_offset(offset), m_byteSize(byteSize)
{
    if (!source || !source->GetHost().SharesServicesWith(host))
        throw std::invalid_argument("Compute readback source must belong to its host");
    source->ValidateRange(offset, byteSize);
    m_source = source->m_state;
    m_storage->source = source->GetAllocation();
    m_sourceBuffer = m_storage->source->GetBuffer();
}

std::shared_ptr<BufferResource> ComputeReadback::AllocateStaging(bool managed)
{
    const BufferDesc desc{
        m_byteSize, BufferUsageFlags::TransferDestination, BufferMemory::Readback, nullptr, 0, kComputeBufferQueues};
    const auto handle = m_host.device.CreatePrivateBuffer(desc, managed);
    if (!handle)
        return {};
    if (!handle->IsValid())
        throw std::runtime_error("Failed to allocate compute readback storage");
    std::shared_ptr<BufferResource> staging;
    try {
        staging = std::make_shared<BufferResource>(m_host.device, *handle, m_byteSize);
    } catch (...) {
        m_host.device.Release(*handle);
        throw;
    }
    m_host.Queue().RecordStagingAllocation();
    return staging;
}

void ComputeReadback::Rebind(std::shared_ptr<ComputeBufferAllocation> allocation)
{
    if (m_cancelled)
        return;
    m_storage->source = std::move(allocation);
    m_sourceBuffer = m_storage->source->GetBuffer();
}

void ComputeReadback::Record(ComputeRecordingContext &context) const
{
    if (!m_cancelled) {
        context.Require(m_storage->source->GetLastWriteSubmission());
        context.Transfer().CopyBuffer(m_sourceBuffer, m_storage->staging->GetBuffer(), {m_offset, 0, m_byteSize});
    }
}

std::shared_ptr<ComputeReadback::Storage> ComputeReadback::Attach(SubmissionTicket ticket) noexcept
{
    if (m_cancelled)
        return {};
    m_ticket = ticket;
    m_sourceWriteSubmission = m_storage->source->GetLastWriteSubmission();
    m_source.reset();
    return m_storage;
}

bool ComputeReadback::Cancel() noexcept
{
    if (m_cancelled || !m_storage)
        return false;
    m_cancelled = true;
    m_storage.reset();
    m_source.reset();
    return true;
}

void ComputeReadback::AppendRetainedStorage(std::vector<ComputeStorageBlock> &blocks) const
{
    if (m_storage)
        m_storage->AppendRetainedStorage(blocks);
}

void ComputeReadback::Storage::AppendRetainedStorage(std::vector<ComputeStorageBlock> &blocks) const
{
    if (source)
        source->AppendRetainedStorage(blocks);
    if (staging)
        blocks.push_back({staging->GetBuffer(), staging->GetByteSize()});
}

bool ComputeReadback::IsComplete()
{
    if (!m_complete && m_ticket.IsValid() && m_host.Queue().IsComplete(m_ticket)) {
        m_complete = true;
        if (m_storage)
            m_storage->source.reset();
        m_source.reset();
    }
    return m_complete;
}

void ComputeReadback::Wait()
{
    if (m_cancelled)
        throw std::logic_error("Compute readback was cancelled");
    if (!m_ticket.IsValid())
        throw std::logic_error("Compute readback has not been accepted");
    if (m_complete)
        return;
    m_host.Queue().Wait(m_ticket);
    m_complete = true;
    m_storage->source.reset();
    m_source.reset();
}

std::vector<uint8_t> ComputeReadback::GetData()
{
    std::vector<uint8_t> bytes(static_cast<size_t>(m_byteSize));
    ReadInto(bytes.data(), m_byteSize);
    return bytes;
}

void ComputeReadback::ReadInto(void *destination, uint64_t byteSize)
{
    if (!destination || byteSize != m_byteSize)
        throw std::invalid_argument("Compute readback destination must match its exact byte size");
    if (m_cancelled)
        throw std::logic_error("Compute readback was cancelled");
    if (!m_storage)
        throw std::logic_error("Compute readback data has already been consumed");
    m_host.RequireAlive();
    Wait();
    if (!m_host.device.ReadBuffer(m_storage->staging->GetBuffer(), 0, destination, m_byteSize))
        throw std::runtime_error("Failed to read completed compute readback storage");
    m_host.Queue().RecordHostMap();
    m_storage.reset();
}

std::vector<std::vector<uint8_t>> ReadComputeBatch(ComputeHost &host, std::vector<ComputeBufferRead> reads)
{
    return SubmitComputeBatchAndRead(host, {}, {}, std::move(reads));
}

} // namespace infernux::rhi
