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

std::unique_ptr<BufferResource> CreateBufferResource(Device &device, const BufferDesc &desc)
{
    const auto handle = device.CreateBuffer(desc);
    if (!handle.IsValid())
        throw std::runtime_error("Failed to allocate compute buffer storage");
    return std::make_unique<BufferResource>(device, handle, desc.byteSize);
}
} // namespace

ComputeBuffer::ComputeBuffer(ComputeHost &host, ComputeBufferDesc desc) : m_host(host.device, host.queue), m_desc(desc)
{
    if (desc.elementCount == 0)
        throw std::invalid_argument("Compute buffer element count must be positive");
    if (desc.lanes == 0 || desc.lanes > 4)
        throw std::invalid_argument("Compute buffer lanes must be between one and four");
    if (desc.elementCount > (std::numeric_limits<uint64_t>::max)() / desc.GetElementStride())
        throw std::overflow_error("Compute buffer byte size exceeds the addressable range");
    const auto byteSize = desc.GetByteSize();

    m_storage = CreateBufferResource(host.device,
                                     {byteSize,
                                      BufferUsageFlags::Storage | BufferUsageFlags::Uniform | BufferUsageFlags::Vertex |
                                          BufferUsageFlags::TransferSource | BufferUsageFlags::TransferDestination,
                                      BufferMemory::DeviceLocal, nullptr, 0, kComputeBufferQueues});
}

ComputeBuffer::~ComputeBuffer()
{
    if (m_uploadTicket.IsValid()) {
        try {
            m_host.queue.Wait(m_uploadTicket);
        } catch (...) {
        }
    }
}

BufferResource &ComputeBuffer::EnsureUploadBuffer()
{
    if (!m_upload) {
        m_upload = CreateBufferResource(m_host.device, {GetByteSize(), BufferUsageFlags::TransferSource,
                                                        BufferMemory::Upload, nullptr, 0, kComputeBufferQueues});
        m_host.queue.RecordStagingAllocation();
    }
    return *m_upload;
}

BufferResource &ComputeBuffer::EnsureReadbackBuffer()
{
    if (!m_readback) {
        m_readback = CreateBufferResource(m_host.device, {GetByteSize(), BufferUsageFlags::TransferDestination,
                                                          BufferMemory::Readback, nullptr, 0, kComputeBufferQueues});
        m_host.queue.RecordStagingAllocation();
    }
    return *m_readback;
}

uint64_t ComputeBuffer::GetByteSize() const noexcept
{
    return m_storage ? m_storage->GetByteSize() : 0;
}

const ComputeBufferDesc &ComputeBuffer::GetDesc() const noexcept
{
    return m_desc;
}

BufferHandle ComputeBuffer::GetBuffer() const noexcept
{
    return m_storage ? m_storage->GetBuffer() : BufferHandle{};
}

ComputeHost &ComputeBuffer::GetHost() noexcept
{
    return m_host;
}

const ComputeHost &ComputeBuffer::GetHost() const noexcept
{
    return m_host;
}

void ComputeBuffer::ValidateRange(uint64_t offset, uint64_t byteSize) const
{
    const auto capacity = GetByteSize();
    if (byteSize == 0 || offset > capacity || byteSize > capacity - offset)
        throw std::out_of_range("Compute buffer transfer range is invalid");
}

void ComputeBuffer::SetData(uint64_t offset, const void *data, uint64_t byteSize)
{
    PrepareUpload(offset, data, byteSize);
    m_uploadTicket = m_host.queue.Submit([this, offset, byteSize](ComputeRecordingContext &context) {
        RecordPreparedUpload(context, offset, byteSize);
        // This submission belongs to Compute, which may have no graphics stages.
        // Graphics consumers acquire visibility through the attached write ticket.
        context.PipelineBarrier(PipelineStage::Transfer, Access::TransferWrite, PipelineStage::ComputeShader,
                                Access::ShaderRead | Access::ShaderWrite);
        return true;
    });
    m_host.queue.RecordUpload(byteSize);
    AttachWrite(m_uploadTicket);
}

void ComputeBuffer::PrepareUpload(uint64_t offset, const void *data, uint64_t byteSize)
{
    ValidateRange(offset, byteSize);
    if (!data)
        throw std::invalid_argument("Compute buffer source cannot be null");
    // The upload allocation is persistently reused. Wait only when the CPU is
    // about to overwrite bytes still consumed by its previous copy; the new
    // upload itself stays asynchronous and is ordered before later kernels by
    // the engine compute queue.
    if (m_uploadTicket.IsValid()) {
        m_host.queue.Wait(m_uploadTicket);
        m_uploadTicket = {};
    }
    auto &uploadResource = EnsureUploadBuffer();
    if (!m_host.device.WriteBuffer(uploadResource.GetBuffer(), offset, data, byteSize))
        throw std::runtime_error("Failed to write compute upload storage");
    m_host.queue.RecordHostMap();
}

void ComputeBuffer::RecordPreparedUpload(ComputeRecordingContext &context, uint64_t offset, uint64_t byteSize) const
{
    ValidateRange(offset, byteSize);
    if (!m_upload)
        throw std::logic_error("Compute buffer upload was not prepared");
    context.Transfer().CopyBuffer(m_upload->GetBuffer(), m_storage->GetBuffer(), {offset, offset, byteSize});
}

void ComputeBuffer::AttachUpload(SubmissionTicket ticket) noexcept
{
    m_uploadTicket = ticket;
    AttachWrite(ticket);
}

void ComputeBuffer::AttachWrite(SubmissionTicket ticket) noexcept
{
    if (ticket.IsValid())
        m_lastWriteSubmission = ticket;
}

SubmissionTicket ComputeBuffer::GetLastWriteSubmission() const noexcept
{
    return m_lastWriteSubmission;
}

void ComputeBuffer::PrepareReadback(uint64_t offset, uint64_t byteSize)
{
    ValidateRange(offset, byteSize);
    (void)EnsureReadbackBuffer();
}

void ComputeBuffer::RecordPreparedReadback(ComputeRecordingContext &context, uint64_t offset, uint64_t byteSize) const
{
    ValidateRange(offset, byteSize);
    if (!m_readback)
        throw std::logic_error("Compute buffer readback was not prepared");
    context.Transfer().CopyBuffer(m_storage->GetBuffer(), m_readback->GetBuffer(), {offset, offset, byteSize});
}

std::vector<uint8_t> ComputeBuffer::FinishReadback(uint64_t offset, uint64_t byteSize)
{
    ValidateRange(offset, byteSize);
    if (!m_readback)
        throw std::logic_error("Compute buffer readback was not prepared");
    m_uploadTicket = {};
    std::vector<uint8_t> bytes(static_cast<size_t>(byteSize));
    if (!m_host.device.ReadBuffer(m_readback->GetBuffer(), offset, bytes.data(), byteSize))
        throw std::runtime_error("Failed to read compute readback storage");
    m_host.queue.RecordHostMap();
    return bytes;
}

std::vector<uint8_t> ComputeBuffer::GetData(uint64_t offset, uint64_t byteSize)
{
    auto result = ReadComputeBatch(m_host, {{shared_from_this(), offset, byteSize}});
    return std::move(result.front());
}

std::shared_ptr<ComputeReadback> ComputeBuffer::GetDataAsync(uint64_t offset, uint64_t byteSize)
{
    ValidateRange(offset, byteSize);
    return std::make_shared<ComputeReadback>(m_host, shared_from_this(), offset, byteSize);
}

ComputeReadback::ComputeReadback(ComputeHost &host, std::shared_ptr<ComputeBuffer> source, uint64_t offset,
                                 uint64_t byteSize)
    : m_host(host.device, host.queue), m_source(std::move(source)), m_byteSize(byteSize)
{
    if (!m_source || !m_source->GetHost().SharesServicesWith(host))
        throw std::invalid_argument("Compute readback source must belong to its host");
    if (byteSize == 0 || offset > m_source->GetByteSize() || byteSize > m_source->GetByteSize() - offset)
        throw std::out_of_range("Compute readback range is invalid");
    m_staging = CreateBufferResource(host.device, {byteSize, BufferUsageFlags::TransferDestination,
                                                   BufferMemory::Readback, nullptr, 0, kComputeBufferQueues});
    host.queue.RecordStagingAllocation();
    m_ticket = host.queue.Submit([this, offset, byteSize](ComputeRecordingContext &context) {
        context.PipelineBarrier(PipelineStage::AllCommands, Access::MemoryWrite, PipelineStage::Transfer,
                                Access::TransferRead);
        context.Transfer().CopyBuffer(m_source->GetBuffer(), m_staging->GetBuffer(), {offset, 0, byteSize});
        context.PipelineBarrier(PipelineStage::Transfer, Access::TransferWrite, PipelineStage::Host, Access::HostRead);
        return true;
    });
    if (!m_ticket.IsValid())
        throw std::runtime_error("Failed to submit compute readback");
    host.queue.RecordReadback(byteSize);
}

ComputeReadback::~ComputeReadback()
{
    try {
        Wait();
    } catch (...) {
    }
}

bool ComputeReadback::IsComplete()
{
    if (!m_complete && m_host.queue.IsComplete(m_ticket)) {
        m_complete = true;
        m_ticket = {};
    }
    return m_complete;
}

void ComputeReadback::Wait()
{
    if (m_complete || !m_ticket.IsValid())
        return;
    m_host.queue.Wait(m_ticket);
    m_complete = true;
    m_ticket = {};
}

std::vector<uint8_t> ComputeReadback::GetData()
{
    Wait();
    std::vector<uint8_t> bytes(static_cast<size_t>(m_byteSize));
    if (!m_host.device.ReadBuffer(m_staging->GetBuffer(), 0, bytes.data(), m_byteSize))
        throw std::runtime_error("Failed to read completed compute readback storage");
    m_host.queue.RecordHostMap();
    return bytes;
}

std::vector<std::vector<uint8_t>> ReadComputeBatch(ComputeHost &host, std::vector<ComputeBufferRead> reads)
{
    return SubmitComputeBatchAndRead(host, {}, {}, std::move(reads));
}

} // namespace infernux::rhi
