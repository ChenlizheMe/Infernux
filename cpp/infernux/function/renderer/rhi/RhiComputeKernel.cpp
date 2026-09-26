#include "RhiComputeKernel.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace infernux::rhi
{

namespace
{
std::vector<uint32_t> SequentialBindings(uint32_t count)
{
    std::vector<uint32_t> result(count);
    for (uint32_t index = 0; index < count; ++index)
        result[index] = index;
    return result;
}

std::vector<ComputeBufferBinding> StorageBindings(std::vector<uint32_t> slots)
{
    std::vector<ComputeBufferBinding> result;
    result.reserve(slots.size());
    for (const auto slot : slots)
        result.push_back({slot, BindingType::StorageBuffer});
    return result;
}
} // namespace

ComputeKernel::ComputeKernel(ComputeHost &host, const uint32_t *spirvWords, size_t wordCount,
                             uint32_t bufferBindingCount, uint32_t pushConstantBytes)
    : ComputeKernel(host, spirvWords, wordCount, SequentialBindings(bufferBindingCount), pushConstantBytes)
{
}

ComputeKernel::ComputeKernel(ComputeHost &host, const uint32_t *spirvWords, size_t wordCount,
                             std::vector<uint32_t> bufferBindings, uint32_t pushConstantBytes)
    : ComputeKernel(host, spirvWords, wordCount, StorageBindings(std::move(bufferBindings)), pushConstantBytes)
{
}

ComputeKernel::ComputeKernel(ComputeHost &host, const uint32_t *spirvWords, size_t wordCount,
                             std::vector<ComputeBufferBinding> bufferBindings, uint32_t pushConstantBytes)
    : m_host(host.device, host.queue), m_pushConstantBytes(pushConstantBytes)
{
    if (!spirvWords || wordCount < 5 || spirvWords[0] != 0x07230203u)
        throw std::invalid_argument("Compute kernel requires valid SPIR-V words");
    if (bufferBindings.size() > BindingLayoutDesc::MaxEntries)
        throw std::invalid_argument("Compute kernel has too many buffer bindings");
    m_bufferBindings.reserve(bufferBindings.size());
    m_bufferBindingTypes.reserve(bufferBindings.size());
    for (size_t index = 0; index < bufferBindings.size(); ++index) {
        const auto binding = bufferBindings[index];
        if (binding.slot >= BindingLayoutDesc::MaxEntries)
            throw std::invalid_argument("Compute kernel buffer binding exceeds the RHI binding range");
        for (size_t previous = 0; previous < index; ++previous)
            if (bufferBindings[previous].slot == binding.slot)
                throw std::invalid_argument("Compute kernel buffer bindings must be unique");
        if (binding.type != BindingType::StorageBuffer && binding.type != BindingType::UniformBuffer)
            throw std::invalid_argument("Compute kernel accepts only storage or uniform buffers");
        m_bufferBindings.push_back(binding.slot);
        m_bufferBindingTypes.push_back(binding.type);
    }

    BindingLayoutDesc layoutDesc;
    for (size_t index = 0; index < m_bufferBindings.size(); ++index)
        layoutDesc.entries[index] = {m_bufferBindings[index], m_bufferBindingTypes[index], ShaderStage::Compute, 1};
    layoutDesc.entryCount = static_cast<uint32_t>(m_bufferBindings.size());
    m_layout = host.device.CreateBindingLayout(layoutDesc);
    if (!m_layout.IsValid())
        throw std::runtime_error("Failed to create compute kernel binding layout");

    const auto shader = host.device.CreateShaderModule(ShaderModuleDesc::FromSpirV(spirvWords, wordCount));
    if (shader.IsValid()) {
        ComputePipelineDesc pipelineDesc;
        pipelineDesc.computeShader = shader;
        pipelineDesc.bindingLayouts[0] = m_layout;
        pipelineDesc.bindingLayoutCount = 1;
        pipelineDesc.pushConstantBytes = pushConstantBytes;
        m_pipeline = host.device.CreateComputePipeline(pipelineDesc);
    }
    host.device.Release(shader);
    if (!m_pipeline.IsValid()) {
        host.device.Release(m_layout);
        m_layout = {};
        throw std::runtime_error("Failed to create compute kernel pipeline");
    }
}

ComputeKernel::~ComputeKernel()
{
    Destroy();
}

uint32_t ComputeKernel::GetBufferBindingCount() const noexcept
{
    return static_cast<uint32_t>(m_bufferBindings.size());
}

const std::vector<uint32_t> &ComputeKernel::GetBufferBindings() const noexcept
{
    return m_bufferBindings;
}

uint32_t ComputeKernel::GetPushConstantBytes() const noexcept
{
    return m_pushConstantBytes;
}

void ComputeKernel::Dispatch(std::vector<std::shared_ptr<ComputeBuffer>> buffers, const void *pushConstants,
                             uint32_t pushConstantBytes, uint32_t groupCountX, uint32_t groupCountY,
                             uint32_t groupCountZ)
{
    if (pushConstantBytes > 0 && !pushConstants)
        throw std::invalid_argument("Compute kernel push constants cannot be null");
    Collect();
    std::vector<uint8_t> constants(pushConstantBytes);
    if (pushConstantBytes > 0)
        std::memcpy(constants.data(), pushConstants, pushConstantBytes);
    std::vector<ComputeBufferAccess> accesses(buffers.size(), ComputeBufferAccess::ReadWrite);
    auto prepared = PrepareDispatch(std::move(buffers), std::move(accesses), std::move(constants), groupCountX,
                                    groupCountY, groupCountZ);
    const auto usedGroup = prepared.group;
    const auto usedBuffers = prepared.buffers;
    const auto ticket = m_host.queue.Submit([prepared = std::move(prepared)](ComputeRecordingContext &context) {
        context.PipelineBarrier(PipelineStage::AllCommands, Access::MemoryWrite, PipelineStage::ComputeShader,
                                Access::ShaderRead | Access::ShaderWrite);
        const auto encoder = context.Compute();
        encoder.BindPipeline(prepared.pipeline);
        encoder.BindGroup(prepared.pipeline, 0, prepared.group);
        if (!prepared.pushConstants.empty())
            encoder.PushConstants(prepared.pipeline, static_cast<uint32_t>(prepared.pushConstants.size()),
                                  prepared.pushConstants.data());
        encoder.Dispatch(prepared.groupCountX, prepared.groupCountY, prepared.groupCountZ);
        context.PipelineBarrier(PipelineStage::ComputeShader, Access::ShaderWrite, PipelineStage::AllCommands,
                                Access::MemoryRead | Access::MemoryWrite);
        return true;
    });
    if (!ticket.IsValid()) {
        throw std::runtime_error("Failed to submit compute kernel");
    }
    m_host.queue.RecordDispatches(1);
    Attach(ticket);
    AttachGroup(usedGroup, ticket, usedBuffers);
    for (const auto &buffer : usedBuffers)
        buffer->AttachWrite(ticket);
}

ComputeKernel::PreparedDispatch ComputeKernel::PrepareDispatch(std::vector<std::shared_ptr<ComputeBuffer>> buffers,
                                                               std::vector<ComputeBufferAccess> bufferAccesses,
                                                               std::vector<uint8_t> pushConstants, uint32_t groupCountX,
                                                               uint32_t groupCountY, uint32_t groupCountZ)
{
    if (!m_pipeline.IsValid())
        throw std::runtime_error("Compute kernel is closed");
    if (buffers.size() != m_bufferBindings.size())
        throw std::invalid_argument("Compute kernel buffer binding count does not match its compiled layout");
    if (bufferAccesses.size() != buffers.size())
        throw std::invalid_argument("Compute kernel buffer access count does not match its resource bindings");
    if (pushConstants.size() != m_pushConstantBytes)
        throw std::invalid_argument("Compute kernel push constants do not match its compiled layout");
    if (groupCountX == 0 || groupCountY == 0 || groupCountZ == 0)
        throw std::invalid_argument("Compute kernel dispatch dimensions must be positive");
    for (const auto &buffer : buffers)
        if (!buffer)
            throw std::invalid_argument("Compute kernel buffer cannot be null");
    return {this,
            m_pipeline,
            ResolveBindGroup(buffers),
            std::move(buffers),
            std::move(bufferAccesses),
            false,
            std::move(pushConstants),
            groupCountX,
            groupCountY,
            groupCountZ};
}

void ComputeKernel::Attach(SubmissionTicket ticket)
{
    for (const auto &pending : m_pending)
        if (pending.ticket.device == ticket.device && pending.ticket.queue == ticket.queue &&
            pending.ticket.serial == ticket.serial)
            return;
    m_pending.push_back({ticket});
}

void ComputeKernel::AttachGroup(BindGroupHandle group, SubmissionTicket ticket,
                                const std::vector<std::shared_ptr<ComputeBuffer>> &buffers)
{
    const auto found = std::find_if(m_groups.begin(), m_groups.end(),
                                    [group](const BindingGroupEntry &entry) { return entry.group == group; });
    if (found == m_groups.end())
        throw std::logic_error("Submitted compute binding group is not owned by its kernel");
    found->inFlightBuffers = buffers;
    found->inFlightTicket = ticket;
}

SubmissionTicket SubmitComputeBatch(ComputeHost &host, std::vector<ComputeDispatchDesc> dispatches)
{
    return SubmitComputeBatch(host, {}, std::move(dispatches));
}

ComputeKernel::PreparedBatch ComputeKernel::PrepareBatch(ComputeHost &host, std::vector<ComputeBufferUpdate> updates,
                                                         std::vector<ComputeDispatchDesc> dispatches)
{
    PreparedBatch result;
    std::unordered_map<ComputeBuffer *, bool> updatedBuffers;
    result.updatedBuffers.reserve(updates.size());
    for (auto &update : updates) {
        if (!update.buffer)
            throw std::invalid_argument("Compute batch update buffer cannot be null");
        if (!update.buffer->GetHost().SharesServicesWith(host))
            throw std::invalid_argument("Compute batch update buffers must belong to its host");
        if (!updatedBuffers.emplace(update.buffer.get(), true).second)
            throw std::invalid_argument("A compute batch can update each buffer only once");
        update.buffer->PrepareUpload(update.offset, update.data.data(), update.data.size());
        result.updatedBuffers.push_back(update.buffer);
    }

    std::unordered_map<ComputeKernel *, bool> uniqueKernels;
    result.kernels.reserve(dispatches.size());
    for (const auto &dispatch : dispatches) {
        if (!dispatch.kernel || !dispatch.kernel->m_host.SharesServicesWith(host))
            throw std::invalid_argument("Compute batch kernels must belong to its host");
        if (uniqueKernels.emplace(dispatch.kernel.get(), true).second) {
            dispatch.kernel->Collect();
            result.kernels.push_back(dispatch.kernel);
        }
    }

    result.dispatches.reserve(dispatches.size());
    std::unordered_map<ComputeBuffer *, uint8_t> pendingAccesses;
    std::unordered_map<ComputeBuffer *, bool> writtenBuffers;
    for (auto &dispatch : dispatches) {
        auto prepared = dispatch.kernel->PrepareDispatch(
            std::move(dispatch.buffers), std::move(dispatch.bufferAccesses), std::move(dispatch.pushConstants),
            dispatch.groupCountX, dispatch.groupCountY, dispatch.groupCountZ);
        std::unordered_map<ComputeBuffer *, uint8_t> currentAccesses;
        for (size_t index = 0; index < prepared.buffers.size(); ++index) {
            const auto bits = static_cast<uint8_t>(prepared.bufferAccesses[index]);
            if (bits == 0 || bits > static_cast<uint8_t>(ComputeBufferAccess::ReadWrite))
                throw std::invalid_argument("Compute buffer access declaration is invalid");
            currentAccesses[prepared.buffers[index].get()] |= bits;
        }
        for (const auto &[buffer, current] : currentAccesses) {
            const auto found = pendingAccesses.find(buffer);
            if (found == pendingAccesses.end())
                continue;
            const auto previous = found->second;
            constexpr auto write = static_cast<uint8_t>(ComputeBufferAccess::Write);
            if ((previous & write) != 0 || (current & write) != 0) {
                prepared.barrierBefore = true;
                break;
            }
        }
        if (prepared.barrierBefore)
            pendingAccesses.clear();
        for (const auto &[buffer, access] : currentAccesses)
            pendingAccesses[buffer] |= access;
        constexpr auto write = static_cast<uint8_t>(ComputeBufferAccess::Write);
        for (size_t index = 0; index < prepared.buffers.size(); ++index) {
            if ((static_cast<uint8_t>(prepared.bufferAccesses[index]) & write) == 0)
                continue;
            if (writtenBuffers.emplace(prepared.buffers[index].get(), true).second)
                result.writtenBuffers.push_back(prepared.buffers[index]);
        }
        result.dispatches.push_back(std::move(prepared));
    }
    result.updates = std::move(updates);
    return result;
}

SubmissionTicket ComputeKernel::SubmitBatch(ComputeHost &host, PreparedBatch batch,
                                            const std::vector<ComputeBufferRead> &reads)
{
    struct UsedGroup
    {
        ComputeKernel *kernel = nullptr;
        BindGroupHandle group;
        std::vector<std::shared_ptr<ComputeBuffer>> buffers;
    };
    std::vector<UsedGroup> usedGroups;
    usedGroups.reserve(batch.dispatches.size());
    for (const auto &dispatch : batch.dispatches)
        usedGroups.push_back({dispatch.owner, dispatch.group, dispatch.buffers});
    const uint64_t dispatchCount = static_cast<uint64_t>(batch.dispatches.size());
    uint64_t uploadBytes = 0;
    for (const auto &update : batch.updates)
        uploadBytes += static_cast<uint64_t>(update.data.size());
    uint64_t readbackBytes = 0;
    for (const auto &read : reads)
        readbackBytes += read.byteSize;

    const auto ticket = host.queue.Submit([updates = std::move(batch.updates), prepared = std::move(batch.dispatches),
                                           reads](ComputeRecordingContext &context) {
        for (const auto &update : updates)
            update.buffer->RecordPreparedUpload(context, update.offset, update.data.size());
        if (!prepared.empty()) {
            if (!updates.empty())
                context.PipelineBarrier(PipelineStage::Transfer, Access::TransferWrite, PipelineStage::ComputeShader,
                                        Access::ShaderRead | Access::ShaderWrite);
            context.PipelineBarrier(PipelineStage::AllCommands, Access::MemoryWrite, PipelineStage::ComputeShader,
                                    Access::ShaderRead | Access::ShaderWrite);
            const auto encoder = context.Compute();
            for (size_t index = 0; index < prepared.size(); ++index) {
                const auto &dispatch = prepared[index];
                if (dispatch.barrierBefore)
                    context.PipelineBarrier(PipelineStage::ComputeShader, Access::ShaderWrite,
                                            PipelineStage::ComputeShader, Access::ShaderRead | Access::ShaderWrite);
                encoder.BindPipeline(dispatch.pipeline);
                encoder.BindGroup(dispatch.pipeline, 0, dispatch.group);
                if (!dispatch.pushConstants.empty())
                    encoder.PushConstants(dispatch.pipeline, static_cast<uint32_t>(dispatch.pushConstants.size()),
                                          dispatch.pushConstants.data());
                encoder.Dispatch(dispatch.groupCountX, dispatch.groupCountY, dispatch.groupCountZ);
            }
            if (reads.empty())
                context.PipelineBarrier(PipelineStage::ComputeShader, Access::ShaderWrite, PipelineStage::AllCommands,
                                        Access::MemoryRead | Access::MemoryWrite);
            else
                context.PipelineBarrier(PipelineStage::ComputeShader, Access::ShaderWrite, PipelineStage::AllCommands,
                                        Access::TransferRead | Access::MemoryRead | Access::MemoryWrite);
        } else if (!updates.empty()) {
            if (reads.empty())
                context.PipelineBarrier(PipelineStage::Transfer, Access::TransferWrite, PipelineStage::AllCommands,
                                        Access::MemoryRead | Access::MemoryWrite);
            else
                context.PipelineBarrier(PipelineStage::Transfer, Access::TransferWrite, PipelineStage::AllCommands,
                                        Access::TransferRead | Access::MemoryRead | Access::MemoryWrite);
        } else if (!reads.empty()) {
            context.PipelineBarrier(PipelineStage::AllCommands, Access::MemoryWrite, PipelineStage::Transfer,
                                    Access::TransferRead);
        }
        if (!reads.empty()) {
            for (const auto &read : reads)
                read.buffer->RecordPreparedReadback(context, read.offset, read.byteSize);
            context.PipelineBarrier(PipelineStage::Transfer, Access::TransferWrite, PipelineStage::Host,
                                    Access::HostRead);
        }
        return true;
    });
    if (!ticket.IsValid())
        throw std::runtime_error("Failed to submit compute batch");
    host.queue.RecordDispatches(dispatchCount);
    if (uploadBytes != 0)
        host.queue.RecordUpload(uploadBytes);
    if (readbackBytes != 0)
        host.queue.RecordReadback(readbackBytes);
    for (const auto &buffer : batch.updatedBuffers)
        buffer->AttachUpload(ticket);
    for (const auto &buffer : batch.writtenBuffers)
        buffer->AttachWrite(ticket);
    for (const auto &kernel : batch.kernels)
        kernel->Attach(ticket);
    for (const auto &used : usedGroups)
        used.kernel->AttachGroup(used.group, ticket, used.buffers);
    return ticket;
}

SubmissionTicket SubmitComputeBatch(ComputeHost &host, std::vector<ComputeBufferUpdate> updates,
                                    std::vector<ComputeDispatchDesc> dispatches)
{
    if (updates.empty() && dispatches.empty())
        throw std::invalid_argument("Compute batch must contain an update or dispatch");
    return ComputeKernel::SubmitBatch(host,
                                      ComputeKernel::PrepareBatch(host, std::move(updates), std::move(dispatches)), {});
}

std::vector<std::vector<uint8_t>> SubmitComputeBatchAndRead(ComputeHost &host, std::vector<ComputeBufferUpdate> updates,
                                                            std::vector<ComputeDispatchDesc> dispatches,
                                                            std::vector<ComputeBufferRead> reads)
{
    if (reads.empty())
        return {};
    for (const auto &read : reads) {
        if (!read.buffer || !read.buffer->GetHost().SharesServicesWith(host))
            throw std::invalid_argument("Compute readback buffers must belong to its host");
        read.buffer->PrepareReadback(read.offset, read.byteSize);
    }
    const auto ticket = ComputeKernel::SubmitBatch(
        host, ComputeKernel::PrepareBatch(host, std::move(updates), std::move(dispatches)), reads);
    host.queue.Wait(ticket);
    std::vector<std::vector<uint8_t>> result;
    result.reserve(reads.size());
    for (const auto &read : reads)
        result.push_back(read.buffer->FinishReadback(read.offset, read.byteSize));
    return result;
}

BindGroupHandle ComputeKernel::ResolveBindGroup(const std::vector<std::shared_ptr<ComputeBuffer>> &buffers)
{
    for (const auto &entry : m_groups) {
        bool matches = buffers.size() == entry.buffers.size();
        for (size_t index = 0; matches && index < buffers.size(); ++index) {
            const auto cached = entry.buffers[index].lock();
            matches = cached && buffers[index].get() == cached.get();
        }
        if (matches)
            return entry.group;
    }

    BindGroupDesc groupDesc;
    groupDesc.layout = m_layout;
    for (size_t index = 0; index < m_bufferBindings.size(); ++index) {
        groupDesc.buffers[index] = {m_bufferBindings[index], m_bufferBindingTypes[index], buffers[index]->GetBuffer(),
                                    0, buffers[index]->GetByteSize()};
    }
    groupDesc.bufferCount = static_cast<uint32_t>(m_bufferBindings.size());
    const auto group = m_host.device.CreateBindGroup(groupDesc);
    if (!group.IsValid())
        throw std::runtime_error("Failed to create compute kernel bind group");
    std::vector<std::weak_ptr<ComputeBuffer>> weakBuffers;
    weakBuffers.reserve(buffers.size());
    for (const auto &buffer : buffers)
        weakBuffers.push_back(buffer);
    m_groups.push_back({group, std::move(weakBuffers), {}, {}});
    return group;
}

void ComputeKernel::CollectBindingGroups()
{
    auto entry = m_groups.begin();
    while (entry != m_groups.end()) {
        if (entry->inFlightTicket.IsValid() && m_host.queue.IsComplete(entry->inFlightTicket)) {
            entry->inFlightBuffers.clear();
            entry->inFlightTicket = {};
        }
        const bool expired = !entry->inFlightTicket.IsValid() &&
                             std::any_of(entry->buffers.begin(), entry->buffers.end(),
                                         [](const std::weak_ptr<ComputeBuffer> &buffer) { return buffer.expired(); });
        if (!expired) {
            ++entry;
            continue;
        }
        m_host.device.Release(entry->group);
        entry = m_groups.erase(entry);
    }
}

void ComputeKernel::Collect()
{
    m_host.queue.Collect();
    auto pending = m_pending.begin();
    while (pending != m_pending.end()) {
        if (!m_host.queue.IsComplete(pending->ticket)) {
            ++pending;
            continue;
        }
        pending = m_pending.erase(pending);
    }
    CollectBindingGroups();
}

void ComputeKernel::Wait()
{
    for (const auto &pending : m_pending)
        m_host.queue.Wait(pending.ticket);
    m_pending.clear();
    for (auto &entry : m_groups) {
        entry.inFlightBuffers.clear();
        entry.inFlightTicket = {};
    }
    CollectBindingGroups();
}

void ComputeKernel::Destroy() noexcept
{
    try {
        Wait();
    } catch (...) {
    }
    for (const auto &entry : m_groups)
        m_host.device.Release(entry.group);
    m_host.device.Release(m_pipeline);
    m_host.device.Release(m_layout);
    m_groups.clear();
    m_pipeline = {};
    m_layout = {};
}

} // namespace infernux::rhi
