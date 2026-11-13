#include "RhiComputeKernel.h"
#include "RhiUpload.h"
#include <core/log/InxLog.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace infernux::rhi
{

namespace
{
void ValidateIndirectRange(const ComputeBufferDesc &desc, uint64_t offset)
{
    if (!HasBufferUsage(desc.usage, BufferUsageFlags::Indirect))
        throw std::invalid_argument("Indirect arguments require an allocation with Indirect usage");
    const auto capacity = desc.GetByteSize();
    if (offset % sizeof(uint32_t) || offset > capacity || ComputeIndirectArguments::ByteSize > capacity - offset)
        throw std::out_of_range("Indirect arguments require twelve aligned bytes within their allocation");
}

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
    : ComputeKernel(host, ShaderModuleDesc::FromSpirV(spirvWords, wordCount), "main", std::move(bufferBindings),
                    pushConstantBytes)
{
}

ComputeKernel::ComputeKernel(ComputeHost &host, const ShaderModuleDesc &shaderDesc, const std::string &entryPoint,
                             std::vector<ComputeBufferBinding> bufferBindings, uint32_t pushConstantBytes)
    : m_host(host), m_pushConstantBytes(pushConstantBytes)
{
    if (!m_host.IsAlive())
        throw std::runtime_error("Compute kernel device is retired");
    if (bufferBindings.size() > BindingLayoutDesc::MaxEntries)
        throw std::invalid_argument("Compute kernel has too many buffer bindings");
    m_bufferBindings.reserve(bufferBindings.size());
    m_bufferBindingTypes.reserve(bufferBindings.size());
    m_defaultBufferAccesses.reserve(bufferBindings.size());
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
        m_defaultBufferAccesses.push_back(binding.type == BindingType::UniformBuffer || binding.readOnlyStorage
                                              ? ComputeBufferAccess::Read
                                              : ComputeBufferAccess::ReadWrite);
    }

    BindingLayoutDesc layoutDesc;
    for (size_t index = 0; index < m_bufferBindings.size(); ++index) {
        layoutDesc.entries[index] = {m_bufferBindings[index], m_bufferBindingTypes[index], ShaderStage::Compute, 1};
        layoutDesc.entries[index].readOnlyStorage = bufferBindings[index].readOnlyStorage;
    }
    layoutDesc.entryCount = static_cast<uint32_t>(m_bufferBindings.size());
    m_layout = host.device.CreateBindingLayout(layoutDesc);
    if (!m_layout.IsValid())
        throw std::runtime_error("Failed to create compute kernel binding layout");

    const auto shader = host.device.CreateShaderModule(shaderDesc);
    if (shader.IsValid()) {
        ComputePipelineDesc pipelineDesc;
        pipelineDesc.computeShader = shader;
        pipelineDesc.entryPoint = entryPoint.c_str();
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
    std::vector<uint8_t> constants(pushConstantBytes);
    if (pushConstantBytes > 0)
        std::memcpy(constants.data(), pushConstants, pushConstantBytes);
    SubmitComputeBatch(m_host, {{shared_from_this(), std::move(buffers), m_defaultBufferAccesses, std::move(constants),
                                 groupCountX, groupCountY, groupCountZ}});
}

ComputeKernel::PreparedDispatch ComputeKernel::PrepareDispatch(
    std::vector<std::shared_ptr<ComputeBuffer>> buffers, std::vector<ComputeBufferAccess> bufferAccesses,
    std::vector<uint8_t> pushConstants, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ,
    std::optional<ComputeDispatchProof> proof, std::optional<ComputeIndirectArguments> indirectArguments)
{
    if (bufferAccesses.size() != buffers.size())
        throw std::invalid_argument("Compute kernel buffer access count does not match its resource bindings");
    ValidateDispatch(bufferAccesses, pushConstants.size(), groupCountX, groupCountY, groupCountZ,
                     indirectArguments.has_value());
    for (const auto &buffer : buffers)
        if (!buffer || !buffer->GetHost().SharesServicesWith(m_host))
            throw std::invalid_argument("Compute kernel buffers must belong to its host");
    if (proof) {
        const auto validate = [&](const auto &ranges, ComputeBufferAccess required) {
            for (const auto &range : ranges) {
                if (range.binding >= buffers.size() ||
                    (static_cast<uint8_t>(bufferAccesses[range.binding]) & static_cast<uint8_t>(required)) == 0)
                    throw std::invalid_argument("Compute proof range must name a binding with matching access");
                buffers[range.binding]->ValidateRange(range.offset, range.byteSize);
            }
        };
        validate(proof->reads, ComputeBufferAccess::Read);
        validate(proof->writes, ComputeBufferAccess::Write);
    }
    PreparedDispatch result;
    result.kernel = shared_from_this();
    result.pipeline = m_pipeline;
    result.allocations.reserve(buffers.size());
    for (size_t index = 0; index < buffers.size(); ++index) {
        const auto &buffer = buffers[index];
        if (buffer->IsFrozen() &&
            (static_cast<uint8_t>(bufferAccesses[index]) & static_cast<uint8_t>(ComputeBufferAccess::Write)))
            throw std::invalid_argument("A frozen compute value cannot be written");
        result.allocations.push_back(buffer->GetAllocation());
        result.states.push_back(buffer->m_state);
    }
    result.group = ResolveBindGroup(result.allocations);
    result.bufferAccesses = std::move(bufferAccesses);
    result.pushConstants = std::move(pushConstants);
    result.groupCountX = groupCountX;
    result.groupCountY = groupCountY;
    result.groupCountZ = groupCountZ;
    result.proof = std::move(proof);
    if (indirectArguments) {
        const auto &arguments = *indirectArguments;
        if (!arguments.buffer || !arguments.buffer->GetHost().SharesServicesWith(m_host))
            throw std::invalid_argument("Indirect arguments must belong to the kernel host");
        ValidateIndirectRange(arguments.buffer->GetDesc(), arguments.offset);
        const auto allocation = arguments.buffer->GetAllocation();
        for (size_t index = 0; index < result.allocations.size(); ++index)
            if (result.allocations[index] == allocation &&
                (m_defaultBufferAccesses[index] != ComputeBufferAccess::Read ||
                 result.bufferAccesses[index] != ComputeBufferAccess::Read))
                throw std::invalid_argument("Indirect arguments cannot alias writable shader storage");
        if (!result.proof)
            result.proof.emplace();
        result.proof->reads.push_back(
            {static_cast<uint32_t>(result.allocations.size()), arguments.offset, ComputeIndirectArguments::ByteSize});
        result.allocations.push_back(allocation);
        result.states.push_back(arguments.buffer->m_state);
        result.bufferAccesses.push_back(ComputeBufferAccess::Read);
        result.indirectOffset = arguments.offset;
    }
    return result;
}

void ComputeKernel::ValidateDispatch(const std::vector<ComputeBufferAccess> &bufferAccesses, size_t pushConstantBytes,
                                     uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ,
                                     bool indirect) const
{
    if (!m_host.IsAlive())
        throw std::runtime_error("Compute kernel device is retired");
    if (!m_pipeline.IsValid())
        throw std::runtime_error("Compute kernel is closed");
    if (bufferAccesses.size() != m_bufferBindings.size())
        throw std::invalid_argument("Compute kernel buffer binding count does not match its compiled layout");
    if (pushConstantBytes != m_pushConstantBytes)
        throw std::invalid_argument("Compute kernel push constants do not match its compiled layout");
    if (!indirect && (groupCountX == 0 || groupCountY == 0 || groupCountZ == 0))
        throw std::invalid_argument("Compute kernel dispatch dimensions must be positive");
    const uint32_t groups[] = {groupCountX, groupCountY, groupCountZ};
    const auto &limits = m_host.device.GetCapabilities().limits;
    for (size_t axis = 0; axis < 3; ++axis) {
        if (groups[axis] > limits.maxComputeWorkgroupCount[axis])
            throw std::invalid_argument("Compute kernel dispatch group count " + std::string(1, "XYZ"[axis]) + "=" +
                                        std::to_string(groups[axis]) + " exceeds device limit " +
                                        std::to_string(limits.maxComputeWorkgroupCount[axis]));
    }
    for (const auto access : bufferAccesses)
        if (access < ComputeBufferAccess::Read || access > ComputeBufferAccess::ReadWrite)
            throw std::invalid_argument("Compute buffer access declaration is invalid");
}

bool ComputeKernel::PreparedDispatch::RecordOne(const ComputeCommandEncoder &encoder) const
{
    if (!kernel || !kernel->GetHost().IsAlive() || !encoder.CanRecord(pipeline, group))
        return false;
    encoder.BindPipeline(pipeline);
    encoder.BindGroup(pipeline, 0, group);
    if (!pushConstants.empty())
        encoder.PushConstants(pipeline, static_cast<uint32_t>(pushConstants.size()), pushConstants.data());
    if (indirectOffset)
        encoder.DispatchIndirect(allocations.back()->GetBuffer(), *indirectOffset);
    else
        encoder.Dispatch(groupCountX, groupCountY, groupCountZ);
    return true;
}

const std::vector<BindingType> &ComputeKernel::PreparedDispatch::GetBufferBindingTypes() const noexcept
{
    return kernel->m_bufferBindingTypes;
}

void ComputeKernel::Attach(SubmissionTicket ticket)
{
    for (const auto &pending : m_pending)
        if (pending.ticket.device == ticket.device && pending.ticket.queue == ticket.queue &&
            pending.ticket.serial == ticket.serial)
            return;
    m_pending.push_back({ticket});
}

SubmissionTicket SubmitComputeBatch(ComputeHost &host, std::vector<ComputeDispatchDesc> dispatches)
{
    return SubmitComputeBatch(host, {}, std::move(dispatches));
}

ComputeKernel::PreparedBatch ComputeKernel::PrepareBatch(ComputeHost &host, std::vector<ComputeBufferUpdate> updates,
                                                         std::vector<ComputeDispatchDesc> dispatches,
                                                         std::vector<ComputeBufferCopy> copies)
{
    host.RequireAlive();
    PreparedBatch result;
    std::unordered_map<ComputeBufferAllocation *, bool> updatedBuffers;
    result.updatedBuffers.reserve(updates.size());
    result.updates.reserve(updates.size());
    for (auto &update : updates) {
        if (!update.buffer)
            throw std::invalid_argument("Compute batch update buffer cannot be null");
        if (!update.buffer->GetHost().SharesServicesWith(host))
            throw std::invalid_argument("Compute batch update buffers must belong to its host");
        if (update.buffer->IsFrozen())
            throw std::invalid_argument("A frozen compute value cannot be updated");
        const auto allocation = update.buffer->GetAllocation();
        if (!updatedBuffers.emplace(allocation.get(), true).second)
            throw std::invalid_argument("A compute batch can update each buffer only once");
        update.buffer->ValidateRange(update.offset, update.data.size());
        result.updatedBuffers.push_back(allocation);
        result.updates.push_back({allocation, update.buffer->m_state, update.offset, std::move(update.data)});
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
    std::unordered_map<ComputeBufferAllocation *, uint8_t> pendingAccesses;
    std::unordered_map<ComputeBufferAllocation *, bool> writtenBuffers;
    for (auto &dispatch : dispatches) {
        auto prepared = dispatch.kernel->PrepareDispatch(
            std::move(dispatch.buffers), std::move(dispatch.bufferAccesses), std::move(dispatch.pushConstants),
            dispatch.groupCountX, dispatch.groupCountY, dispatch.groupCountZ, std::move(dispatch.proof),
            std::move(dispatch.indirectArguments));
        result.hasIndirect |= prepared.IsIndirect();
        std::unordered_map<ComputeBufferAllocation *, uint8_t> currentAccesses;
        for (size_t index = 0; index < prepared.allocations.size(); ++index) {
            const auto bits = static_cast<uint8_t>(prepared.bufferAccesses[index]);
            currentAccesses[prepared.allocations[index].get()] |= bits;
        }
        bool barrierBefore = false;
        for (const auto &[buffer, current] : currentAccesses) {
            const auto found = pendingAccesses.find(buffer);
            if (found == pendingAccesses.end())
                continue;
            const auto previous = found->second;
            constexpr auto write = static_cast<uint8_t>(ComputeBufferAccess::Write);
            if ((previous & write) != 0 || (current & write) != 0) {
                barrierBefore = true;
                break;
            }
        }
        if (barrierBefore)
            pendingAccesses.clear();
        for (const auto &[buffer, access] : currentAccesses)
            pendingAccesses[buffer] |= access;
        constexpr auto write = static_cast<uint8_t>(ComputeBufferAccess::Write);
        for (size_t index = 0; index < prepared.allocations.size(); ++index) {
            if ((static_cast<uint8_t>(prepared.bufferAccesses[index]) & write) == 0)
                continue;
            if (writtenBuffers.emplace(prepared.allocations[index].get(), true).second)
                result.writtenBuffers.push_back(prepared.allocations[index]);
        }
        result.dispatches.push_back({std::move(prepared), barrierBefore});
    }
    for (const auto &copy : copies) {
        if (!copy.source || !copy.destination || !copy.source->GetHost().SharesServicesWith(host) ||
            !copy.destination->GetHost().SharesServicesWith(host))
            throw std::invalid_argument("Compute copy buffers must belong to their host");
        const auto &region = copy.region;
        copy.source->ValidateRange(region.sourceOffset, region.byteSize);
        copy.destination->ValidateRange(region.destinationOffset, region.byteSize);
        if (copy.destination->IsFrozen())
            throw std::invalid_argument("A frozen compute value cannot be a copy destination");
        if (!region.byteSize || region.sourceOffset % 4 || region.destinationOffset % 4 || region.byteSize % 4)
            throw std::invalid_argument("Compute copies require nonempty four-byte-aligned ranges");
        if (copy.source->GetBuffer() == copy.destination->GetBuffer() &&
            region.sourceOffset < region.destinationOffset + region.byteSize &&
            region.destinationOffset < region.sourceOffset + region.byteSize)
            throw std::invalid_argument("Compute copies cannot overlap within one allocation");
        const auto source = copy.source->GetAllocation(), destination = copy.destination->GetAllocation();
        if (writtenBuffers.emplace(destination.get(), true).second)
            result.writtenBuffers.push_back(destination);
        result.copies.push_back({source, destination, copy.source->m_state, copy.destination->m_state, region});
    }
    return result;
}

std::vector<ComputeKernel::BufferValueState> ComputeKernel::PrepareValues(const PreparedBatch &batch)
{
    std::vector<BufferValueState> values;
    const auto include = [&](const auto &allocation) {
        if (std::none_of(values.begin(), values.end(),
                         [&](const auto &value) { return value.allocation == allocation; }))
            values.push_back({allocation, allocation->m_initialized, allocation->m_written});
    };
    for (const auto &buffer : batch.updatedBuffers)
        include(buffer);
    for (const auto &buffer : batch.writtenBuffers)
        include(buffer);
    const auto known = [&](const auto &allocation) -> const intervals::Set & {
        const auto value =
            std::find_if(values.begin(), values.end(), [&](const auto &item) { return item.allocation == allocation; });
        return value == values.end() ? allocation->m_initialized : value->initialized;
    };
    const auto written = [&](const auto &allocation) -> intervals::Set & {
        return std::find_if(values.begin(), values.end(),
                            [&](const auto &value) { return value.allocation == allocation; })
            ->written;
    };
    for (const auto &update : batch.updates) {
        auto &target = std::find_if(values.begin(), values.end(), [&](const auto &value) {
                           return value.allocation == update.allocation;
                       })->initialized;
        intervals::Add(target, update.offset, update.offset + update.data.size());
        intervals::Add(written(update.allocation), update.offset, update.offset + update.data.size());
    }
    for (const auto &dispatch : batch.dispatches) {
        const auto &work = dispatch.work;
        for (size_t index = 0; index < work.allocations.size(); ++index)
            if (static_cast<uint8_t>(work.bufferAccesses[index]) & static_cast<uint8_t>(ComputeBufferAccess::Write))
                intervals::Add(written(work.allocations[index]), 0, work.allocations[index]->GetByteSize());
        if (!work.proof)
            continue; // Ordinary raw work makes no new complete-store claim.
        for (const auto &read : work.proof->reads)
            if (!intervals::Contains(known(work.allocations[read.binding]), read.offset, read.offset + read.byteSize))
                throw std::invalid_argument("Compute input range lacks a prior accepted complete writer");
        for (const auto &write : work.proof->writes) {
            if (work.IsIndirect() && !intervals::Contains(known(work.allocations[write.binding]), write.offset,
                                                          write.offset + write.byteSize))
                throw std::invalid_argument("Indirect work cannot establish an unconditional complete-write range");
            auto &target = std::find_if(values.begin(), values.end(), [&](const auto &value) {
                               return value.allocation == work.allocations[write.binding];
                           })->initialized;
            intervals::Add(target, write.offset, write.offset + write.byteSize);
        }
    }
    for (const auto &copy : batch.copies) {
        const auto &region = copy.region;
        if (!copy.preserveInitialization &&
            !intervals::Contains(known(copy.source), region.sourceOffset, region.sourceOffset + region.byteSize))
            throw std::invalid_argument("Compute copy source lacks a prior accepted complete writer");
        auto &target = std::find_if(values.begin(), values.end(), [&](const auto &value) {
                           return value.allocation == copy.destination;
                       })->initialized;
        if (copy.preserveInitialization) {
            const auto proven = intervals::Intersect(known(copy.source),
                                                     {{region.sourceOffset, region.sourceOffset + region.byteSize}});
            for (const auto &[begin, end] : proven)
                intervals::Add(target, region.destinationOffset + begin - region.sourceOffset,
                               region.destinationOffset + end - region.sourceOffset);
        } else {
            intervals::Add(target, region.destinationOffset, region.destinationOffset + region.byteSize);
        }
        intervals::Add(written(copy.destination), region.destinationOffset, region.destinationOffset + region.byteSize);
    }
    return values;
}

bool ComputeKernel::RecordBatch(ComputeRecordingContext &context, const PreparedBatch &batch,
                                const std::vector<std::shared_ptr<ComputeReadback>> &reads)
{
    const auto &updates = batch.updates;
    const auto &prepared = batch.dispatches;
    const bool hasReads =
        std::any_of(reads.begin(), reads.end(), [](const auto &read) { return !read->IsCancelled(); });
    for (const auto &update : updates)
        context.Require(update.allocation->GetLastWriteSubmission());
    for (const auto &dispatch : prepared)
        for (const auto &allocation : dispatch.work.allocations)
            context.Require(allocation->GetLastWriteSubmission());
    for (const auto &copy : batch.copies) {
        context.Require(copy.source->GetLastWriteSubmission());
        context.Require(copy.destination->GetLastWriteSubmission());
    }
    if (!updates.empty())
        context.PipelineBarrier(PipelineStage::AllCommands, Access::MemoryRead | Access::MemoryWrite,
                                PipelineStage::Transfer, Access::TransferWrite);
    for (const auto &update : updates)
        update.allocation->RecordUpload(context, update.offset, update.data.data(), update.data.size());
    if (!prepared.empty()) {
        // Compiler arguments are uniform buffers; ShaderRead only covers
        // storage/image reads and does not make uploaded uniforms visible.
        auto destinationStage = PipelineStage::ComputeShader;
        auto computeAccess = Access::UniformRead | Access::ShaderRead | Access::ShaderWrite;
        if (batch.hasIndirect) {
            destinationStage = destinationStage | PipelineStage::DrawIndirect;
            computeAccess = computeAccess | Access::IndirectRead;
        }
        if (!updates.empty())
            context.PipelineBarrier(PipelineStage::Transfer, Access::TransferWrite, destinationStage, computeAccess);
        context.PipelineBarrier(PipelineStage::AllCommands, Access::MemoryWrite, destinationStage, computeAccess);
        const auto encoder = context.Compute();
        for (size_t index = 0; index < prepared.size(); ++index) {
            const auto &dispatch = prepared[index];
            if (dispatch.barrierBefore)
                context.PipelineBarrier(PipelineStage::ComputeShader, Access::ShaderWrite, destinationStage,
                                        computeAccess);
            if (!dispatch.work.RecordOne(encoder))
                return false;
        }
        if (!hasReads)
            context.PipelineBarrier(PipelineStage::ComputeShader, Access::ShaderWrite, PipelineStage::AllCommands,
                                    Access::MemoryRead | Access::MemoryWrite);
        else
            context.PipelineBarrier(PipelineStage::ComputeShader, Access::ShaderWrite, PipelineStage::AllCommands,
                                    Access::TransferRead | Access::MemoryRead | Access::MemoryWrite);
    } else if (!updates.empty()) {
        if (!hasReads)
            context.PipelineBarrier(PipelineStage::Transfer, Access::TransferWrite, PipelineStage::AllCommands,
                                    Access::MemoryRead | Access::MemoryWrite);
        else
            context.PipelineBarrier(PipelineStage::Transfer, Access::TransferWrite, PipelineStage::AllCommands,
                                    Access::TransferRead | Access::MemoryRead | Access::MemoryWrite);
    } else if (hasReads) {
        context.PipelineBarrier(PipelineStage::AllCommands, Access::MemoryWrite, PipelineStage::Transfer,
                                Access::TransferRead);
    }
    if (!batch.copies.empty()) {
        context.PipelineBarrier(PipelineStage::AllCommands, Access::MemoryRead | Access::MemoryWrite,
                                PipelineStage::Transfer, Access::TransferRead | Access::TransferWrite);
        const auto encoder = context.Transfer();
        for (size_t index = 0; index < batch.copies.size(); ++index) {
            if (index)
                context.PipelineBarrier(PipelineStage::Transfer, Access::TransferWrite, PipelineStage::Transfer,
                                        Access::TransferRead | Access::TransferWrite);
            const auto &copy = batch.copies[index];
            encoder.CopyBuffer(copy.source->GetBuffer(), copy.destination->GetBuffer(), copy.region);
        }
        context.PipelineBarrier(PipelineStage::Transfer, Access::TransferWrite, PipelineStage::AllCommands,
                                Access::TransferRead | Access::MemoryRead | Access::MemoryWrite);
    }
    if (hasReads) {
        for (const auto &read : reads)
            read->Record(context);
        context.PipelineBarrier(PipelineStage::Transfer, Access::TransferWrite, PipelineStage::Host, Access::HostRead);
    }
    return true;
}

bool ComputeCompletion::IsComplete()
{
    if (m_receipt->complete)
        return true;
    if (!m_host.IsAlive())
        throw std::runtime_error("Compute completion host is retired");
    // IsComplete may retire queue-owned resources. Do not hold the device's
    // lifetime gate across it: resource destructors acquire that same gate.
    (void)m_host.Queue().IsComplete(m_ticket);
    return m_receipt->complete;
}

void ComputeCompletion::Wait()
{
    if (m_receipt->complete)
        return;
    if (!m_host.IsAlive())
        throw std::runtime_error("Compute completion host is retired");
    // Queue retirement releases resources that acquire the lifetime gate.
    m_host.Queue().Wait(m_ticket);
}

PreparedComputeBatch::PreparedComputeBatch(ComputeHost &host, std::vector<ComputeBufferUpdate> updates,
                                           std::vector<ComputeDispatchDesc> dispatches,
                                           std::vector<ComputeBufferRead> reads, std::vector<ComputeBufferCopy> copies)
    : m_host(host)
{
    for (const auto &read : reads) {
        if (!read.buffer || !read.buffer->GetHost().SharesServicesWith(host))
            throw std::invalid_argument("Compute readback sources must belong to their host");
        read.buffer->ValidateRange(read.offset, read.byteSize);
    }
    m_batch = std::make_shared<ComputeKernel::PreparedBatch>(
        ComputeKernel::PrepareBatch(host, std::move(updates), std::move(dispatches), std::move(copies)));
    m_batch->readOwners.resize(reads.size());
    m_readbacks.reserve(reads.size());
    for (auto &read : reads)
        m_readbacks.emplace_back(new ComputeReadback(host, std::move(read.buffer), read.offset, read.byteSize));
    m_dispatchCount = static_cast<uint64_t>(m_batch->dispatches.size());
    m_executionSignature = {m_dispatchCount, m_batch->updates.size(), m_batch->copies.size(), m_readbacks.size()};
    for (const auto &update : m_batch->updates) {
        const auto bytes = static_cast<uint64_t>(update.data.size());
        m_uploadBytes += bytes;
        if (bytes > InlineBufferUpdateBytes)
            m_uploadStagingBytes += bytes;
        m_executionSignature.push_back(bytes);
    }
    m_uploadStagingBytes = UploadStagingClass(m_uploadStagingBytes);
    for (const auto &copy : m_batch->copies) {
        m_copyBytes += copy.region.byteSize;
        m_executionSignature.push_back(copy.region.byteSize);
    }
    for (const auto &dispatch : m_batch->dispatches) {
        const auto &work = dispatch.work;
        m_executionSignature.insert(m_executionSignature.end(),
                                    {work.pipeline.index, work.pipeline.generation, work.groupCountX, work.groupCountY,
                                     work.groupCountZ, work.allocations.size(), work.IsIndirect(),
                                     work.indirectOffset.value_or(0)});
        for (size_t index = 0; index < work.allocations.size(); ++index) {
            m_executionSignature.push_back(work.allocations[index]->GetByteSize());
            m_executionSignature.push_back(static_cast<uint64_t>(work.bufferAccesses[index]));
        }
    }
}

PreparedComputeBatch::~PreparedComputeBatch()
{
    Cancel();
}

std::shared_ptr<PreparedComputeBatch> PrepareComputeBatch(ComputeHost &host, std::vector<ComputeBufferUpdate> updates,
                                                          std::vector<ComputeDispatchDesc> dispatches,
                                                          std::vector<ComputeBufferRead> reads,
                                                          std::vector<ComputeBufferCopy> copies)
{
    if (updates.empty() && dispatches.empty() && reads.empty() && copies.empty())
        throw std::invalid_argument("Compute batch must contain an update, dispatch, copy or readback");
    return std::shared_ptr<PreparedComputeBatch>(
        new PreparedComputeBatch(host, std::move(updates), std::move(dispatches), std::move(reads), std::move(copies)));
}

bool PreparedComputeBatch::HasWork() const noexcept
{
    return m_batch && !m_ticket.IsValid() &&
           (!m_batch->updates.empty() || !m_batch->dispatches.empty() || !m_batch->copies.empty() ||
            std::any_of(m_readbacks.begin(), m_readbacks.end(), [](const auto &read) { return !read->IsCancelled(); }));
}

template <typename Visitor> bool PreparedComputeBatch::VisitWriters(Visitor &&visitor) const
{
    if (!m_batch)
        return false;
    for (const auto &update : m_batch->updates)
        if (visitor(update.state))
            return true;
    for (const auto &dispatch : m_batch->dispatches)
        for (size_t index = 0; index < dispatch.work.states.size(); ++index)
            if ((static_cast<uint8_t>(dispatch.work.bufferAccesses[index]) &
                 static_cast<uint8_t>(ComputeBufferAccess::Write)) &&
                visitor(dispatch.work.states[index]))
                return true;
    return std::any_of(m_batch->copies.begin(), m_batch->copies.end(),
                       [&](const auto &copy) { return visitor(copy.destinationState); });
}

bool PreparedComputeBatch::HasPendingWrite(const ComputeBuffer &buffer) const noexcept
{
    return !m_ticket.IsValid() &&
           VisitWriters([&](const auto &state) { return state.get() == buffer.GetWriterStateIdentity(); });
}

void PreparedComputeBatch::AppendPendingWriterStates(WriterStateSet &states) const
{
    if (m_ticket.IsValid())
        return;
    VisitWriters([&](const auto &state) {
        states.insert(state.get());
        return false;
    });
}

bool PreparedComputeBatch::RequiresPreparation() const
{
    if (!m_batch)
        return false;
    for (const auto &version : m_writeVersions)
        if (version.target->GetAllocation() != version.base)
            return true;
    const auto changed = [&](const auto &state, const auto &allocation, bool write) {
        for (const auto &version : m_writeVersions)
            if (version.target == state)
                return allocation != version.candidate->GetAllocation();
        const auto current = state->GetAllocation();
        return current != allocation || (write && current->m_readers != 0);
    };
    for (const auto &update : m_batch->updates)
        if (changed(update.state, update.allocation, true))
            return true;
    for (const auto &dispatch : m_batch->dispatches)
        for (size_t index = 0; index < dispatch.work.states.size(); ++index)
            if (changed(dispatch.work.states[index], dispatch.work.allocations[index],
                        (static_cast<uint8_t>(dispatch.work.bufferAccesses[index]) &
                         static_cast<uint8_t>(ComputeBufferAccess::Write)) != 0))
                return true;
    for (const auto &copy : m_batch->copies)
        if (changed(copy.sourceState, copy.source, false) || changed(copy.destinationState, copy.destination, true))
            return true;
    for (const auto &read : m_readbacks)
        if (!read->IsCancelled() && changed(read->m_source, read->m_storage->source, false))
            return true;
    return false;
}

bool PreparedComputeBatch::OverwritesBeforeRead(const ComputeBufferState &state) const
{
    intervals::Set covered;
    const auto full = [&] { return intervals::Contains(covered, 0, state.GetDesc().GetByteSize()); };
    for (const auto &update : m_batch->updates)
        if (update.state.get() == &state)
            intervals::Add(covered, update.offset, update.offset + update.data.size());
    if (full())
        return true;
    for (const auto &dispatch : m_batch->dispatches) {
        const auto &work = dispatch.work;
        for (size_t index = 0; index < work.states.size(); ++index)
            if (work.states[index].get() == &state &&
                (static_cast<uint8_t>(work.bufferAccesses[index]) & static_cast<uint8_t>(ComputeBufferAccess::Read)))
                return false;
        if (work.proof && !work.IsIndirect())
            for (const auto &write : work.proof->writes)
                if (work.states[write.binding].get() == &state)
                    intervals::Add(covered, write.offset, write.offset + write.byteSize);
        if (full())
            return true;
    }
    for (const auto &copy : m_batch->copies) {
        if (copy.sourceState.get() == &state &&
            !intervals::Contains(covered, copy.region.sourceOffset, copy.region.sourceOffset + copy.region.byteSize))
            return false;
        if (copy.destinationState.get() == &state)
            intervals::Add(covered, copy.region.destinationOffset,
                           copy.region.destinationOffset + copy.region.byteSize);
        if (full())
            return true;
    }
    return false;
}

std::vector<std::shared_ptr<ComputeBufferState>> PreparedComputeBatch::WriterStates() const
{
    std::vector<std::shared_ptr<ComputeBufferState>> writers;
    VisitWriters([&](const auto &state) {
        if (std::find(writers.begin(), writers.end(), state) == writers.end())
            writers.push_back(state);
        return false;
    });
    return writers;
}

uint64_t PreparedComputeBatch::GetPreparationBytes() const
{
    uint64_t bytes = 0;
    for (const auto &state : WriterStates()) {
        const auto current = state->GetAllocation();
        if (current->m_readers && std::none_of(m_writeVersions.begin(), m_writeVersions.end(),
                                               [&](const auto &version) { return version.target == state; }))
            bytes += current->GetByteSize();
    }
    return bytes;
}

bool PreparedComputeBatch::ResolveStorage(bool managed)
{
    if (!RequiresPreparation())
        return true;
    std::vector<WriteVersion> resolved;
    std::vector<std::shared_ptr<ComputeBufferAllocation>> invalidated;
    for (const auto &state : WriterStates()) {
        const auto base = state->GetAllocation();
        if (!base->m_readers)
            continue;
        const auto previous = std::find_if(m_writeVersions.begin(), m_writeVersions.end(),
                                           [&](const auto &version) { return version.target == state; });
        if (previous != m_writeVersions.end() && previous->base == base) {
            resolved.push_back(*previous);
            continue;
        }
        auto candidate = previous != m_writeVersions.end() ? previous->candidate : nullptr;
        if (!candidate) {
            auto allocation = ComputeBufferAllocation::CreatePrivate(m_host.device, base->GetStorageDesc(),
                                                                     base->IsGraphWritable(), managed);
            if (!allocation)
                return false;
            candidate = std::shared_ptr<ComputeBuffer>(new ComputeBuffer(
                m_host, std::make_shared<ComputeBufferState>(state->GetDesc(), std::move(allocation), false)));
        } else {
            invalidated.push_back(candidate->GetAllocation());
        }
        auto source = std::shared_ptr<ComputeBuffer>(
            new ComputeBuffer(m_host, std::make_shared<ComputeBufferState>(state->GetDesc(), base, true)));
        intervals::Set preserve;
        if (!OverwritesBeforeRead(*state)) {
            preserve = base->m_written;
            intervals::Add(preserve, base->m_initialized);
        }
        const auto offset = preserve.empty() ? 0 : preserve.front().first;
        resolved.push_back({state, base, std::move(source), std::move(candidate), std::move(preserve), 0, offset});
    }
    // Commit the new binding only after all new allocations fit. Keep the
    // current copy and cursor if their source version is still the same.
    if (m_preservation) {
        const auto &old = m_writeVersions[m_preservationVersion];
        const auto current = std::find_if(resolved.begin(), resolved.end(), [&](const auto &version) {
            return version.target == old.target && version.base == old.base;
        });
        if (current == resolved.end()) {
            m_preservation->Cancel();
            m_preservation.reset();
        } else {
            m_preservationVersion = static_cast<size_t>(current - resolved.begin());
        }
    }
    for (const auto &allocation : invalidated) {
        // Any earlier copies into this still private candidate precede the
        // new preservation copies on the same compute queue.
        allocation->m_initialized.clear();
        allocation->m_written.clear();
    }
    m_writeVersions.swap(resolved);
    const auto resolve = [&](const auto &state) {
        for (const auto &version : m_writeVersions)
            if (version.target == state)
                return version.candidate->GetAllocation();
        return state->GetAllocation();
    };
    m_batch->updatedBuffers.clear();
    m_batch->writtenBuffers.clear();
    const auto include = [](auto &values, const auto &allocation) {
        if (std::find(values.begin(), values.end(), allocation) == values.end())
            values.push_back(allocation);
    };
    for (auto &update : m_batch->updates) {
        update.allocation = resolve(update.state);
        include(m_batch->updatedBuffers, update.allocation);
    }
    for (auto &dispatch : m_batch->dispatches) {
        auto &work = dispatch.work;
        for (size_t index = 0; index < work.states.size(); ++index) {
            work.allocations[index] = resolve(work.states[index]);
            if (static_cast<uint8_t>(work.bufferAccesses[index]) & static_cast<uint8_t>(ComputeBufferAccess::Write))
                include(m_batch->writtenBuffers, work.allocations[index]);
        }
        work.group = work.kernel->ResolveBindGroup(work.allocations);
    }
    for (auto &copy : m_batch->copies) {
        copy.source = resolve(copy.sourceState);
        copy.destination = resolve(copy.destinationState);
        include(m_batch->writtenBuffers, copy.destination);
    }
    for (const auto &read : m_readbacks)
        if (!read->IsCancelled())
            read->Rebind(resolve(read->m_source));
    return true;
}

bool PreparedComputeBatch::HasPreservationWork() const noexcept
{
    return std::any_of(m_writeVersions.begin(), m_writeVersions.end(),
                       [](const auto &version) { return version.region < version.preserve.size(); });
}

PreparedComputeBatch::PreservationPreparation PreparedComputeBatch::PreparePreservation(uint64_t maxBytes, bool managed)
{
    m_host.RequireAlive();
    if (!maxBytes)
        throw std::invalid_argument("Preservation needs a positive byte allowance");
    if (!ResolveStorage(managed))
        return {false, {}};
    if (m_preservation) {
        if (!m_preservation->GetSubmission().IsValid())
            return {true, m_preservation};
        auto &version = m_writeVersions[m_preservationVersion];
        version.offset = m_preservationEnd;
        if (version.offset == version.preserve[version.region].second) {
            ++version.region;
            if (version.region < version.preserve.size())
                version.offset = version.preserve[version.region].first;
        }
        m_preservation.reset();
    }
    maxBytes = std::max(uint64_t{4}, maxBytes - maxBytes % 4);
    for (size_t index = 0; index < m_writeVersions.size(); ++index) {
        auto &version = m_writeVersions[index];
        if (version.region == version.preserve.size())
            continue;
        const auto bytes = std::min(maxBytes, version.preserve[version.region].second - version.offset);
        m_preservation = PrepareComputeBatch(
            m_host, {}, {}, {}, {{version.source, version.candidate, {version.offset, version.offset, bytes}}});
        m_preservation->m_batch->copies.front().preserveInitialization = true;
        m_preservationVersion = index;
        m_preservationEnd = version.offset + bytes;
        return {true, m_preservation};
    }
    return {true, {}};
}

void PreparedComputeBatch::AppendRetainedStorage(std::vector<ComputeStorageBlock> &blocks) const
{
    for (const auto &version : m_writeVersions) {
        version.source->AppendRetainedStorage(blocks);
        version.candidate->AppendRetainedStorage(blocks);
    }
    if (m_batch) {
        for (const auto &buffer : m_batch->updatedBuffers)
            buffer->AppendRetainedStorage(blocks);
        for (const auto &dispatch : m_batch->dispatches) {
            for (const auto &buffer : dispatch.work.allocations)
                buffer->AppendRetainedStorage(blocks);
            for (const auto &state : dispatch.work.states)
                state->GetAllocation()->AppendRetainedStorage(blocks);
        }
        for (const auto &update : m_batch->updates)
            update.state->GetAllocation()->AppendRetainedStorage(blocks);
        for (const auto &copy : m_batch->copies) {
            copy.source->AppendRetainedStorage(blocks);
            copy.destination->AppendRetainedStorage(blocks);
            copy.sourceState->GetAllocation()->AppendRetainedStorage(blocks);
            copy.destinationState->GetAllocation()->AppendRetainedStorage(blocks);
        }
        for (const auto &read : m_batch->readOwners)
            if (read)
                read->AppendRetainedStorage(blocks);
    }
    for (const auto &read : m_readbacks)
        read->AppendRetainedStorage(blocks);
}

uint64_t PreparedComputeBatch::GetRetainedPayloadBytes() const noexcept
{
    uint64_t bytes = 0;
    if (m_batch) {
        for (const auto &update : m_batch->updates)
            bytes += update.data.capacity();
        for (const auto &dispatch : m_batch->dispatches)
            bytes += dispatch.work.pushConstants.capacity();
    }
    return bytes;
}

uint64_t PreparedComputeBatch::GetTransferBytes() const noexcept
{
    if (m_ticket.IsValid())
        return m_transferBytes;
    if (!m_batch)
        return 0;
    uint64_t bytes = m_uploadBytes + m_copyBytes;
    for (const auto &read : m_readbacks)
        if (!read->IsCancelled())
            bytes += read->GetByteSize();
    return bytes;
}

bool PreparedComputeBatch::Cancel() noexcept
{
    if (!m_batch || m_ticket.IsValid())
        return false;
    for (const auto &read : m_readbacks)
        read->Cancel();
    if (m_preservation)
        m_preservation->Cancel();
    m_preservation.reset();
    m_writeVersions.clear();
    m_batch.reset();
    return true;
}

std::shared_ptr<ComputeReadback> PreparedComputeBatch::Readback(size_t index) const
{
    return m_readbacks.at(index);
}

SubmissionTicket PreparedComputeBatch::Submit()
{
    m_host.RequireAlive();
    while (auto preservation = PreparePreservation(64 * 1024, false).copy)
        (void)preservation->Submit();
    return *SubmitInternal(true);
}

std::optional<SubmissionTicket> PreparedComputeBatch::TrySubmit()
{
    return SubmitInternal(false);
}

ComputeCompletion PreparedComputeBatch::GetCompletion() const
{
    if (!m_ticket.IsValid())
        throw std::logic_error("Compute completion requires an accepted submission");
    return ComputeCompletion(m_host, m_ticket, m_receipt);
}

uint64_t PreparedComputeBatch::GetPendingStagingBytes() const noexcept
{
    if (!m_batch || m_ticket.IsValid())
        return 0;
    uint64_t bytes = m_uploadStagingBytes;
    for (const auto &read : m_readbacks)
        if (!read->IsCancelled() && !read->m_storage->staging)
            bytes += read->GetByteSize();
    return bytes;
}

bool PreparedComputeBatch::PrepareReadbacks(bool managed)
{
    // Keep tentative storage local until the entire batch can record. A
    // capacity refusal must not pin some of its readbacks across retries.
    std::vector<std::pair<ComputeReadback *, std::shared_ptr<BufferResource>>> staging;
    for (const auto &read : m_readbacks) {
        if (read->IsCancelled() || read->m_storage->staging)
            continue;
        auto buffer = read->AllocateStaging(managed);
        if (!buffer)
            return false;
        staging.emplace_back(read.get(), std::move(buffer));
    }
    for (auto &[read, buffer] : staging)
        read->m_storage->staging = std::move(buffer);
    return true;
}

std::optional<SubmissionTicket> PreparedComputeBatch::SubmitInternal(bool block)
{
    m_host.RequireAlive();
    if (m_ticket.IsValid())
        throw std::logic_error("Compute batch has already been submitted");
    if (!HasWork())
        throw std::logic_error("Compute batch has no remaining work");
    if (RequiresPreparation() || HasPreservationWork())
        throw std::logic_error("Prepare and accept preservation work before submitting the original compute batch");
    std::vector<ComputeKernel::BufferValueState> values;
    const auto record = [&](ComputeRecordingContext &context) {
        // Read the latest accepted prefix only once a slot is available. All
        // allocation and proof checks finish before recording real commands.
        values = ComputeKernel::PrepareValues(*m_batch);
        // Graph writers need the contents, not just the allocation, protected
        // until this accepted read retires. Pure Compute FIFO keeps early release.
        m_batch->contentReadOwners.clear();
        const auto retain = [&](const auto &allocation, const auto &desc) {
            if (allocation->IsGraphWritable() &&
                std::none_of(m_batch->contentReadOwners.begin(), m_batch->contentReadOwners.end(),
                             [&](const auto &value) { return value->GetAllocation() == allocation; }))
                m_batch->contentReadOwners.push_back(std::make_shared<ComputeBufferState>(desc, allocation, true));
        };
        for (const auto &dispatch : m_batch->dispatches)
            for (size_t index = 0; index < dispatch.work.allocations.size(); ++index)
                if (static_cast<uint8_t>(dispatch.work.bufferAccesses[index]) &
                    static_cast<uint8_t>(ComputeBufferAccess::Read))
                    retain(dispatch.work.allocations[index], dispatch.work.states[index]->GetDesc());
        for (const auto &copy : m_batch->copies)
            retain(copy.source, copy.sourceState->GetDesc());
        for (const auto &read : m_readbacks)
            if (!read->IsCancelled())
                retain(read->m_storage->source, read->m_source->GetDesc());
        return ComputeKernel::RecordBatch(context, *m_batch, m_readbacks);
    };
    const ComputeSubmissionResources owners{m_batch, m_receipt, m_uploadStagingBytes};
    if (block)
        (void)PrepareReadbacks(false);
    const auto ticket = block ? std::optional{m_host.Queue().Submit(record, owners)}
                              : m_host.Queue().TrySubmit(record, owners, [&] { return PrepareReadbacks(true); });
    if (!ticket)
        return std::nullopt;
    if (!ticket->IsValid())
        throw std::runtime_error("Failed to submit compute batch");
    // Acceptance is final even if later host bookkeeping reports an error.
    m_ticket = *ticket;
    for (auto &value : values) {
        value.allocation->m_initialized.swap(value.initialized);
        value.allocation->m_written.swap(value.written);
    }
    Accepted(*ticket);
    return ticket;
}

void PreparedComputeBatch::Accepted(SubmissionTicket ticket)
{
    m_transferBytes = m_uploadBytes + m_copyBytes;
    for (const auto &buffer : m_batch->updatedBuffers) {
        buffer->m_lastWriteSubmission = ticket;
        buffer->m_lastWriterQueue = ticket.queue;
    }
    for (const auto &buffer : m_batch->writtenBuffers) {
        buffer->m_lastWriteSubmission = ticket;
        buffer->m_lastWriterQueue = ticket.queue;
    }
    for (auto &version : m_writeVersions) {
        m_host.device.PublishPrivateBuffer(version.candidate->GetBuffer());
        version.target->m_allocation = version.candidate->GetAllocation();
    }
    uint64_t readbackBytes = 0;
    for (size_t index = 0; index < m_readbacks.size(); ++index) {
        // Stabilize each result ticket and its independent queue owner before
        // any kernel bookkeeping can throw after GPU acceptance.
        m_batch->readOwners[index] = m_readbacks[index]->Attach(ticket);
        if (m_batch->readOwners[index])
            readbackBytes += m_readbacks[index]->GetByteSize();
    }
    m_host.Queue().RecordDispatches(m_dispatchCount);
    m_transferBytes += readbackBytes;
    if (m_uploadBytes != 0)
        m_host.Queue().RecordUpload(m_uploadBytes);
    if (readbackBytes != 0)
        m_host.Queue().RecordReadback(readbackBytes);
    for (const auto &copy : m_batch->copies)
        m_host.Queue().RecordCopy(copy.region.byteSize);
    for (const auto &kernel : m_batch->kernels)
        kernel->Attach(ticket);
    // Commands now contain the upload bytes/constants. Keep only GPU owners
    // in the queue's shared bundle, not another full CPU payload until retirement.
    m_batch->updates.clear();
    m_batch->writtenBuffers.clear();
    for (auto &dispatch : m_batch->dispatches) {
        std::vector<uint8_t>().swap(dispatch.work.pushConstants);
        dispatch.work.bufferAccesses.clear();
        dispatch.work.states.clear();
    }
    // Accepted compute reads are ordered before later writes on this queue.
    // Keep actual allocations to the fence, not mutable endpoint slots or an
    // extra content lease after their last unsubmitted read has been recorded.
    for (auto &copy : m_batch->copies) {
        copy.sourceState.reset();
        copy.destinationState.reset();
    }
    m_batch.reset();
    m_preservation.reset();
    m_writeVersions.clear();
}

SubmissionTicket SubmitComputeBatch(ComputeHost &host, std::vector<ComputeBufferUpdate> updates,
                                    std::vector<ComputeDispatchDesc> dispatches)
{
    return PrepareComputeBatch(host, std::move(updates), std::move(dispatches))->Submit();
}

std::vector<std::vector<uint8_t>> SubmitComputeBatchAndRead(ComputeHost &host, std::vector<ComputeBufferUpdate> updates,
                                                            std::vector<ComputeDispatchDesc> dispatches,
                                                            std::vector<ComputeBufferRead> reads)
{
    if (reads.empty())
        return {};
    const auto readCount = reads.size();
    auto batch = PrepareComputeBatch(host, std::move(updates), std::move(dispatches), std::move(reads));
    (void)batch->Submit();
    std::vector<std::vector<uint8_t>> result;
    result.reserve(readCount);
    for (size_t index = 0; index < readCount; ++index)
        result.push_back(batch->Readback(index)->GetData());
    return result;
}

BindGroupHandle ComputeKernel::ResolveBindGroup(const std::vector<std::shared_ptr<ComputeBufferAllocation>> &buffers)
{
    CollectBindingGroups();
    for (const auto &entry : m_groups) {
        bool matches = m_bufferBindings.size() == entry.allocations.size();
        for (size_t index = 0; matches && index < m_bufferBindings.size(); ++index) {
            const auto cached = entry.allocations[index].lock();
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
    std::vector<std::weak_ptr<ComputeBufferAllocation>> weakBuffers;
    weakBuffers.reserve(m_bufferBindings.size());
    for (size_t index = 0; index < m_bufferBindings.size(); ++index)
        weakBuffers.push_back(buffers[index]);
    m_groups.push_back({group, std::move(weakBuffers)});
    return group;
}

void ComputeKernel::CollectBindingGroups()
{
    auto entry = m_groups.begin();
    while (entry != m_groups.end()) {
        const bool expired =
            std::any_of(entry->allocations.begin(), entry->allocations.end(),
                        [](const std::weak_ptr<ComputeBufferAllocation> &buffer) { return buffer.expired(); });
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
    m_host.Queue().Collect();
    auto pending = m_pending.begin();
    while (pending != m_pending.end()) {
        if (!m_host.Queue().IsComplete(pending->ticket)) {
            ++pending;
            continue;
        }
        pending = m_pending.erase(pending);
    }
    CollectBindingGroups();
}

void ComputeKernel::Wait()
{
    m_host.RequireAlive();
    for (const auto &pending : m_pending)
        m_host.Queue().Wait(pending.ticket);
    m_pending.clear();
    CollectBindingGroups();
}

void ComputeKernel::Destroy() noexcept
{
    // Accepted compute batches retain the kernel until completion; frame
    // callers retain PreparedDispatch until their Graphics completion epoch.
    // Destruction must not wait on a queue or call a retired device.
    const auto release = [&] {
        for (const auto &entry : m_groups)
            m_host.device.Release(entry.group);
        m_host.device.Release(m_pipeline);
        m_host.device.Release(m_layout);
    };
    if (const auto &lifetime = m_host.GetLifetime(); lifetime) {
        std::shared_lock lock(lifetime->gate);
        if (lifetime->alive.load(std::memory_order_acquire))
            release();
    } else {
        release();
    }
    m_groups.clear();
    m_pending.clear();
    m_pipeline = {};
    m_layout = {};
}

namespace
{
bool Writes(ComputeBufferAccess access)
{
    return (static_cast<uint8_t>(access) & static_cast<uint8_t>(ComputeBufferAccess::Write)) != 0;
}

bool Reads(ComputeBufferAccess access)
{
    return (static_cast<uint8_t>(access) & static_cast<uint8_t>(ComputeBufferAccess::Read)) != 0;
}

bool SameAllocationLayout(const ComputeBufferDesc &left, const ComputeBufferDesc &right)
{
    return left.elementCount == right.elementCount && left.scalarType == right.scalarType &&
           left.lanes == right.lanes && HasBufferUsage(right.usage, left.usage);
}

bool Overlaps(const ComputeWorkRange &left, const ComputeWorkRange &right)
{
    return left.offset < right.offset + right.byteSize && right.offset < left.offset + left.byteSize;
}

void ValidateWorkAliases(const ComputeWorkDispatch &dispatch, const std::vector<BufferHandle> &identities)
{
    for (size_t index = 0; index < dispatch.bindings.size(); ++index) {
        const auto &left = dispatch.bindings[index];
        if (dispatch.indirectArguments) {
            const auto slot = dispatch.indirectArguments->slot;
            const bool same = left.range.slot == slot || (identities[left.range.slot].IsValid() &&
                                                          identities[left.range.slot] == identities[slot]);
            if (same && Writes(left.access))
                throw std::invalid_argument("Indirect arguments cannot alias writable shader storage");
        }
        for (size_t other = 0; other < index; ++other) {
            const auto &right = dispatch.bindings[other];
            const bool same =
                left.range.slot == right.range.slot ||
                (identities[left.range.slot].IsValid() && identities[left.range.slot] == identities[right.range.slot]);
            if (same && (Writes(left.access) || Writes(right.access)) && Overlaps(left.range, right.range))
                throw std::invalid_argument("Compute work writable bindings overlap");
        }
    }
}

void ValidateWorkIsolation(const std::vector<ComputeWorkSlot> &slots, const std::vector<ComputeWorkPort> &ports,
                           const std::vector<ComputeWorkDispatch> &dispatches,
                           const std::vector<BufferHandle> &identities)
{
    struct AllocationUses
    {
        bool aliases = false, external = false;
        std::vector<ComputeWorkBinding> bindings;
        std::vector<const ComputeWorkRange *> ports;
    };
    // Most instances bind distinct allocations. Compare logical lifetimes only
    // within an allocation that is actually shared, not across the whole graph.
    std::unordered_map<BufferHandle, AllocationUses, HandleHash<BufferTag>> allocations;
    allocations.reserve(slots.size());
    for (size_t index = 0; index < slots.size(); ++index) {
        if (!identities[index].IsValid())
            continue;
        auto [entry, inserted] = allocations.try_emplace(identities[index]);
        auto &uses = entry->second;
        uses.aliases |= !inserted;
        uses.external |= bool(slots[index].external);
    }
    for (const auto &dispatch : dispatches) {
        for (const auto &binding : dispatch.bindings) {
            const auto identity = identities[binding.range.slot];
            if (!identity.IsValid())
                continue;
            auto &uses = allocations.at(identity);
            if (uses.aliases)
                uses.bindings.push_back(binding);
        }
        if (dispatch.indirectArguments) {
            const auto &range = *dispatch.indirectArguments;
            const auto identity = identities[range.slot];
            if (identity.IsValid() && allocations.at(identity).aliases)
                allocations.at(identity).bindings.push_back({range, ComputeBufferAccess::Read});
        }
    }
    for (const auto &port : ports) {
        const auto identity = identities[port.range.slot];
        if (identity.IsValid()) {
            auto &uses = allocations.at(identity);
            if (uses.aliases)
                uses.ports.push_back(&port.range);
        }
    }
    for (const auto &allocation : allocations) {
        const auto &uses = allocation.second;
        for (const auto &binding : uses.bindings) {
            if (!Writes(binding.access))
                continue;
            if (uses.external)
                throw std::invalid_argument("Compute work input binding aliases read-only external storage");
            const auto conflicts = [&](const ComputeWorkRange &other) {
                return other.slot != binding.range.slot && Overlaps(binding.range, other);
            };
            for (const auto &other : uses.bindings)
                if (conflicts(other.range))
                    throw std::invalid_argument("Compute work input bindings alias distinct logical slot lifetimes");
            for (const auto *other : uses.ports)
                if (conflicts(*other))
                    throw std::invalid_argument("Compute work input binding aliases another logical port");
        }
    }
}
} // namespace

ComputeWorkRecipe::ComputeWorkRecipe(ComputeHost &host, std::vector<ComputeWorkSlot> slots,
                                     std::vector<ComputeWorkPort> ports, std::vector<ComputeWorkDispatch> dispatches)
    : m_host(host), m_slots(std::move(slots)), m_ports(std::move(ports)), m_dispatches(std::move(dispatches))
{
    if (!m_host.IsAlive())
        throw std::runtime_error("Compute work device is retired");
    if (m_slots.empty() || m_dispatches.empty())
        throw std::invalid_argument("Compute work requires storage and at least one dispatch");
    std::vector<intervals::Set> initialized(m_slots.size());
    std::vector<BufferHandle> identities(m_slots.size());
    for (size_t index = 0; index < m_slots.size(); ++index) {
        const auto &slot = m_slots[index];
        const auto &desc = slot.allocation;
        if (desc.elementCount == 0 || desc.lanes == 0 || desc.lanes > 4 || desc.scalarType > ComputeScalarType::Int64 ||
            desc.elementCount > (std::numeric_limits<uint64_t>::max)() / desc.GetElementStride())
            throw std::invalid_argument("Compute work allocation layout is invalid");
        if (slot.external) {
            const auto ticket = slot.external->GetLastWriteSubmission();
            if (!slot.external->GetHost().SharesServicesWith(m_host) ||
                !SameAllocationLayout(desc, slot.external->GetDesc()) || !slot.external->GetBuffer().IsValid() ||
                !ticket.IsValid() || ticket.device != host.device.GetDeviceId())
                throw std::invalid_argument("Compute work external storage needs a matching accepted compute producer");
            identities[index] = slot.external->GetBuffer();
        }
    }
    const auto validateRange = [&](const ComputeWorkRange &range) {
        if (range.slot >= m_slots.size() || range.byteSize == 0 || range.offset % 4 || range.byteSize % 4)
            throw std::invalid_argument("Compute work binding range is invalid");
        const auto capacity = m_slots[range.slot].allocation.GetByteSize();
        if (range.offset > capacity || range.byteSize > capacity - range.offset)
            throw std::out_of_range("Compute work binding range exceeds its allocation");
    };
    const auto requireRead = [&](const ComputeWorkRange &range) {
        if (const auto &external = m_slots[range.slot].external; external) {
            if (!external->HasInitializedRange(range.offset, range.byteSize))
                throw std::invalid_argument("Compute work external input lacks an initialized logical range");
            intervals::Add(initialized[range.slot], range.offset, range.offset + range.byteSize);
        }
        if (!intervals::Contains(initialized[range.slot], range.offset, range.offset + range.byteSize))
            throw std::invalid_argument("Compute work reads storage without a prior complete writer or input port");
    };
    m_portIndices.reserve(m_ports.size());
    for (size_t index = 0; index < m_ports.size(); ++index) {
        const auto &port = m_ports[index];
        validateRange(port.range);
        if (port.name.empty() || !m_portIndices.emplace(port.name, index).second)
            throw std::invalid_argument("Compute work port names must be nonempty and unique");
        if (port.input) {
            if (m_slots[port.range.slot].external)
                throw std::invalid_argument("Compute work input ports must name per-instance storage");
            // An explicit caller precondition. This does not initialize an
            // instance; its graph must supply the complete writer before use.
            intervals::Add(initialized[port.range.slot], port.range.offset, port.range.offset + port.range.byteSize);
        }
    }
    for (const auto &dispatch : m_dispatches) {
        if (!dispatch.kernel || !dispatch.kernel->GetHost().SharesServicesWith(m_host))
            throw std::invalid_argument("Compute work kernels must share its host");
        std::vector<ComputeBufferAccess> accesses;
        for (const auto &binding : dispatch.bindings) {
            validateRange(binding.range);
            accesses.push_back(binding.access);
            if (m_slots[binding.range.slot].external && Writes(binding.access))
                throw std::invalid_argument("Compute work cannot write external storage");
        }
        dispatch.kernel->ValidateDispatch(accesses, dispatch.pushConstants.size(), dispatch.groupCountX,
                                          dispatch.groupCountY, dispatch.groupCountZ,
                                          dispatch.indirectArguments.has_value());
        if (dispatch.indirectArguments) {
            const auto &arguments = *dispatch.indirectArguments;
            validateRange(arguments);
            if (arguments.byteSize != ComputeIndirectArguments::ByteSize)
                throw std::invalid_argument("Compute work indirect input must contain exactly three uint32 values");
            ValidateIndirectRange(m_slots[arguments.slot].allocation, arguments.offset);
            requireRead(arguments);
        }
        for (size_t index = 0; index < dispatch.bindings.size(); ++index) {
            const auto &binding = dispatch.bindings[index];
            if (dispatch.kernel->m_bufferBindingTypes[index] == BindingType::UniformBuffer && Writes(binding.access))
                throw std::invalid_argument("Compute work uniform bindings must be read-only");
            if (Reads(binding.access))
                requireRead(binding.range);
        }
        ValidateWorkAliases(dispatch, identities);
        for (const auto &fact : dispatch.completeWrites) {
            const auto index = fact.binding;
            if (index >= dispatch.bindings.size() || !Writes(dispatch.bindings[index].access))
                throw std::invalid_argument("Compute work complete-write facts must name writable bindings");
            const auto &binding = dispatch.bindings[index].range;
            validateRange({binding.slot, fact.offset, fact.byteSize});
            if (fact.offset < binding.offset || fact.offset - binding.offset > binding.byteSize ||
                fact.byteSize > binding.byteSize - (fact.offset - binding.offset))
                throw std::out_of_range("Compute work complete write exceeds its binding range");
            if (dispatch.indirectArguments &&
                !intervals::Contains(initialized[binding.slot], fact.offset, fact.offset + fact.byteSize))
                throw std::invalid_argument("Indirect work cannot establish an unconditional complete-write range");
            intervals::Add(initialized[binding.slot], fact.offset, fact.offset + fact.byteSize);
        }
    }
    for (const auto &port : m_ports)
        if (!port.input && !intervals::Contains(initialized[port.range.slot], port.range.offset,
                                                port.range.offset + port.range.byteSize))
            throw std::invalid_argument("Compute work output lacks a complete writer for its entire port range");
    const auto include = [&](const ComputeWorkRange &range) {
        if (std::none_of(m_requiredInputs.begin(), m_requiredInputs.end(), [&](const ComputeWorkRange &other) {
                return other.slot == range.slot && other.offset == range.offset && other.byteSize == range.byteSize;
            }))
            m_requiredInputs.push_back(range);
    };
    for (const auto &port : m_ports)
        if (port.input)
            include(port.range);
    for (const auto &dispatch : m_dispatches) {
        for (const auto &binding : dispatch.bindings)
            if (m_slots[binding.range.slot].external && Reads(binding.access))
                include(binding.range);
        if (dispatch.indirectArguments && m_slots[dispatch.indirectArguments->slot].external)
            include(*dispatch.indirectArguments);
    }
}

std::shared_ptr<ComputeWorkInstance>
ComputeWorkRecipe::Instantiate(std::vector<ComputeWorkInput> inputs,
                               const std::vector<std::shared_ptr<ComputeBuffer>> &externalInputs) const
{
    return Bind(std::move(inputs), {}, externalInputs);
}

std::shared_ptr<ComputeWorkInstance>
ComputeWorkRecipe::Bind(std::vector<ComputeWorkInput> inputs,
                        const std::vector<std::shared_ptr<ComputeBuffer>> &retained,
                        const std::vector<std::shared_ptr<ComputeBuffer>> &externalInputs) const
{
    if (!m_host.IsAlive())
        throw std::runtime_error("Compute work device is retired");
    std::vector<std::shared_ptr<ComputeBuffer>> buffers;
    buffers.reserve(m_slots.size());
    for (size_t index = 0; index < m_slots.size(); ++index)
        buffers.push_back(externalInputs.empty() ? m_slots[index].external : externalInputs[index]);
    std::vector<bool> provided(m_ports.size());
    for (const auto &input : inputs) {
        const auto found = m_portIndices.find(input.port);
        if (found == m_portIndices.end())
            throw std::invalid_argument("Compute work input does not name a declared port");
        const auto &port = m_ports[found->second];
        if (provided[found->second] || !input.buffer || !input.buffer->GetHost().SharesServicesWith(m_host) ||
            !input.buffer->GetBuffer().IsValid() ||
            !SameAllocationLayout(m_slots[port.range.slot].allocation, input.buffer->GetDesc()))
            throw std::invalid_argument(
                "Compute work input must bind one matching real owner to a declared input port");
        provided[found->second] = true;
        auto &target = buffers[port.range.slot];
        if (target && target->GetBuffer() != input.buffer->GetBuffer())
            throw std::invalid_argument("Compute work ports sharing one slot cannot bind different allocations");
        target = input.buffer;
    }
    for (size_t index = 0; index < retained.size(); ++index)
        if (!buffers[index] && !retained[index]->GetAllocation()->IsReadProtected())
            buffers[index] = retained[index];
    std::vector<BufferHandle> identities;
    for (const auto &buffer : buffers)
        identities.push_back(buffer ? buffer->GetBuffer() : BufferHandle{});
    for (const auto &dispatch : m_dispatches)
        ValidateWorkAliases(dispatch, identities);
    ValidateWorkIsolation(m_slots, m_ports, m_dispatches, identities);
    auto host = m_host;
    for (size_t index = 0; index < buffers.size(); ++index) {
        if (buffers[index])
            continue;
        buffers[index] = ComputeBuffer::TryCreateGraphBuffer(host, m_slots[index].allocation);
        if (!buffers[index])
            return {};
    }
    return std::shared_ptr<ComputeWorkInstance>(new ComputeWorkInstance(shared_from_this(), std::move(buffers)));
}

const std::vector<ComputeKernel::PreparedDispatch> &ComputeWorkInstance::GetDispatches() const
{
    if (!m_dispatches.empty())
        return m_dispatches;
    std::vector<ComputeKernel::PreparedDispatch> prepared;
    prepared.reserve(m_recipe->GetDispatches().size());
    for (const auto &dispatch : m_recipe->GetDispatches()) {
        std::vector<std::shared_ptr<ComputeBuffer>> resources;
        std::vector<ComputeBufferAccess> accesses;
        for (const auto &binding : dispatch.bindings) {
            resources.push_back(m_buffers[binding.range.slot]);
            accesses.push_back(binding.access);
        }
        std::optional<ComputeIndirectArguments> arguments;
        if (dispatch.indirectArguments)
            arguments = ComputeIndirectArguments{m_buffers[dispatch.indirectArguments->slot],
                                                 dispatch.indirectArguments->offset};
        prepared.push_back(dispatch.kernel->PrepareDispatch(
            std::move(resources), std::move(accesses), dispatch.pushConstants, dispatch.groupCountX,
            dispatch.groupCountY, dispatch.groupCountZ, {}, std::move(arguments)));
    }
    m_dispatches = std::move(prepared);
    return m_dispatches;
}

ComputeWorkInstance::ComputeWorkInstance(std::shared_ptr<const ComputeWorkRecipe> recipe,
                                         std::vector<std::shared_ptr<ComputeBuffer>> buffers)
    : m_recipe(std::move(recipe)), m_buffers(std::move(buffers))
{
}

std::shared_ptr<ComputeWorkInstance>
ComputeWorkInstance::Rebind(std::vector<ComputeWorkInput> inputs,
                            const std::vector<std::shared_ptr<ComputeBuffer>> &externalInputs) const
{
    return m_recipe->Bind(std::move(inputs), m_buffers, externalInputs);
}

std::shared_ptr<PreparedComputeBatch> ComputeWorkInstance::PrepareBatch(std::vector<ComputeBufferUpdate> updates) const
{
    std::vector<ComputeDispatchDesc> dispatches;
    dispatches.reserve(m_recipe->GetDispatches().size());
    for (const auto &operation : m_recipe->GetDispatches()) {
        ComputeDispatchDesc dispatch;
        dispatch.kernel = operation.kernel;
        dispatch.pushConstants = operation.pushConstants;
        dispatch.groupCountX = operation.groupCountX;
        dispatch.groupCountY = operation.groupCountY;
        dispatch.groupCountZ = operation.groupCountZ;
        dispatch.proof.emplace();
        dispatch.proof->writes = operation.completeWrites;
        if (operation.indirectArguments)
            dispatch.indirectArguments = ComputeIndirectArguments{m_buffers[operation.indirectArguments->slot],
                                                                  operation.indirectArguments->offset};
        dispatch.buffers.reserve(operation.bindings.size());
        dispatch.bufferAccesses.reserve(operation.bindings.size());
        for (uint32_t index = 0; index < operation.bindings.size(); ++index) {
            const auto &binding = operation.bindings[index];
            dispatch.buffers.push_back(m_buffers[binding.range.slot]);
            dispatch.bufferAccesses.push_back(binding.access);
            if (Reads(binding.access))
                dispatch.proof->reads.push_back({index, binding.range.offset, binding.range.byteSize});
        }
        dispatches.push_back(std::move(dispatch));
    }
    auto host = m_recipe->m_host;
    return PrepareComputeBatch(host, std::move(updates), std::move(dispatches));
}

std::vector<ComputeWorkRange> ComputeWorkRecipe::GetRequiredInputs() const
{
    return m_requiredInputs;
}

std::vector<ComputeWorkRange> ComputeWorkInstance::GetRequiredInputs() const
{
    return m_recipe->GetRequiredInputs();
}

} // namespace infernux::rhi
