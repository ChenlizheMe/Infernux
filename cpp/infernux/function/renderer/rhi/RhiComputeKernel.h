#pragma once

#include "RhiComputeBuffer.h"
#include "RhiComputeHost.h"
#include "RhiDescriptors.h"
#include "RhiSubmission.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace infernux::rhi
{

class ComputeKernel;

struct ComputeBufferBinding
{
    uint32_t slot = 0;
    BindingType type = BindingType::StorageBuffer;
};

enum class ComputeBufferAccess : uint8_t
{
    Read = 1,
    Write = 2,
    ReadWrite = 3,
};

struct ComputeDispatchDesc
{
    std::shared_ptr<ComputeKernel> kernel;
    std::vector<std::shared_ptr<ComputeBuffer>> buffers;
    std::vector<ComputeBufferAccess> bufferAccesses;
    std::vector<uint8_t> pushConstants;
    uint32_t groupCountX = 1;
    uint32_t groupCountY = 1;
    uint32_t groupCountZ = 1;
};

struct ComputeBufferUpdate
{
    std::shared_ptr<ComputeBuffer> buffer;
    uint64_t offset = 0;
    std::vector<uint8_t> data;
};

/// Record several kernel launches into one engine-owned command buffer and
/// submit them once. Kernels may repeat only with the same resident bindings;
/// changing a binding set remains an explicit retirement boundary.
SubmissionTicket SubmitComputeBatch(ComputeHost &host, std::vector<ComputeDispatchDesc> dispatches);
SubmissionTicket SubmitComputeBatch(ComputeHost &host, std::vector<ComputeBufferUpdate> updates,
                                    std::vector<ComputeDispatchDesc> dispatches);
/// Record uploads, dependent compute dispatches, and their requested
/// readbacks into one queue submission, then wait once and return the host
/// bytes. This is the synchronization boundary for compact CPU consumers.
std::vector<std::vector<uint8_t>> SubmitComputeBatchAndRead(ComputeHost &host, std::vector<ComputeBufferUpdate> updates,
                                                            std::vector<ComputeDispatchDesc> dispatches,
                                                            std::vector<ComputeBufferRead> reads);

/// Engine-owned executable compute pipeline. Compiler integrations provide
/// SPIR-V and binding metadata; this object owns device binding, submission,
/// and in-flight resource retirement through the shared engine RHI.
class ComputeKernel final
{
  public:
    ComputeKernel(ComputeHost &host, const uint32_t *spirvWords, size_t wordCount, uint32_t bufferBindingCount,
                  uint32_t pushConstantBytes);
    ComputeKernel(ComputeHost &host, const uint32_t *spirvWords, size_t wordCount, std::vector<uint32_t> bufferBindings,
                  uint32_t pushConstantBytes);
    ComputeKernel(ComputeHost &host, const uint32_t *spirvWords, size_t wordCount,
                  std::vector<ComputeBufferBinding> bufferBindings, uint32_t pushConstantBytes);
    ~ComputeKernel();

    ComputeKernel(const ComputeKernel &) = delete;
    ComputeKernel &operator=(const ComputeKernel &) = delete;

    [[nodiscard]] uint32_t GetBufferBindingCount() const noexcept;
    [[nodiscard]] const std::vector<uint32_t> &GetBufferBindings() const noexcept;
    [[nodiscard]] uint32_t GetPushConstantBytes() const noexcept;
    [[nodiscard]] ComputeHost &GetHost() noexcept
    {
        return m_host;
    }
    [[nodiscard]] const ComputeHost &GetHost() const noexcept
    {
        return m_host;
    }
    [[nodiscard]] size_t GetPendingDispatchCount() const noexcept
    {
        return m_pending.size();
    }
    [[nodiscard]] size_t GetCachedBindingGroupCount() const noexcept
    {
        return m_groups.size();
    }
    void Dispatch(std::vector<std::shared_ptr<ComputeBuffer>> buffers, const void *pushConstants,
                  uint32_t pushConstantBytes, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);
    void Collect();
    void Wait();

  private:
    struct PreparedDispatch
    {
        ComputeKernel *owner = nullptr;
        ComputePipelineHandle pipeline;
        BindGroupHandle group;
        std::vector<std::shared_ptr<ComputeBuffer>> buffers;
        std::vector<ComputeBufferAccess> bufferAccesses;
        bool barrierBefore = false;
        std::vector<uint8_t> pushConstants;
        uint32_t groupCountX = 1;
        uint32_t groupCountY = 1;
        uint32_t groupCountZ = 1;
    };

    struct PreparedBatch
    {
        std::vector<ComputeBufferUpdate> updates;
        std::vector<PreparedDispatch> dispatches;
        std::vector<std::shared_ptr<ComputeBuffer>> updatedBuffers;
        std::vector<std::shared_ptr<ComputeBuffer>> writtenBuffers;
        std::vector<std::shared_ptr<ComputeKernel>> kernels;
    };

    struct PendingDispatch
    {
        SubmissionTicket ticket;
    };

    struct BindingGroupEntry
    {
        BindGroupHandle group;
        std::vector<std::weak_ptr<ComputeBuffer>> buffers;
        std::vector<std::shared_ptr<ComputeBuffer>> inFlightBuffers;
        SubmissionTicket inFlightTicket;
    };

    [[nodiscard]] BindGroupHandle ResolveBindGroup(const std::vector<std::shared_ptr<ComputeBuffer>> &buffers);
    [[nodiscard]] PreparedDispatch PrepareDispatch(std::vector<std::shared_ptr<ComputeBuffer>> buffers,
                                                   std::vector<ComputeBufferAccess> bufferAccesses,
                                                   std::vector<uint8_t> pushConstants, uint32_t groupCountX,
                                                   uint32_t groupCountY, uint32_t groupCountZ);
    [[nodiscard]] static PreparedBatch PrepareBatch(ComputeHost &host, std::vector<ComputeBufferUpdate> updates,
                                                    std::vector<ComputeDispatchDesc> dispatches);
    [[nodiscard]] static SubmissionTicket SubmitBatch(ComputeHost &host, PreparedBatch batch,
                                                      const std::vector<ComputeBufferRead> &reads);
    void Attach(SubmissionTicket ticket);
    void AttachGroup(BindGroupHandle group, SubmissionTicket ticket,
                     const std::vector<std::shared_ptr<ComputeBuffer>> &buffers);
    void CollectBindingGroups();
    void Destroy() noexcept;

    friend SubmissionTicket SubmitComputeBatch(ComputeHost &host, std::vector<ComputeDispatchDesc> dispatches);
    friend SubmissionTicket SubmitComputeBatch(ComputeHost &host, std::vector<ComputeBufferUpdate> updates,
                                               std::vector<ComputeDispatchDesc> dispatches);
    friend std::vector<std::vector<uint8_t>> SubmitComputeBatchAndRead(ComputeHost &host,
                                                                       std::vector<ComputeBufferUpdate> updates,
                                                                       std::vector<ComputeDispatchDesc> dispatches,
                                                                       std::vector<ComputeBufferRead> reads);

    ComputeHost m_host;
    BindingLayoutHandle m_layout;
    ComputePipelineHandle m_pipeline;
    std::vector<BindingGroupEntry> m_groups;
    std::vector<uint32_t> m_bufferBindings;
    std::vector<BindingType> m_bufferBindingTypes;
    uint32_t m_pushConstantBytes = 0;
    std::vector<PendingDispatch> m_pending;
};

} // namespace infernux::rhi
