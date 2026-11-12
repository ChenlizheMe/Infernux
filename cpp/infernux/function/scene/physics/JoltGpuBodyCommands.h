#pragma once

#include "JoltGpuPrograms.h"

namespace infernux
{

class JoltGpuCommandCursor;

/// Prepares immutable, per-body FIFO commands against accepted author metadata.
/// The owner finishes the current logical tick/model transaction first, then
/// drains the cursor before a step, query, model mutation or pose consumer.
/// Independent bodies may execute in a different order; only these owner
/// boundaries establish an observable cross-body ordering.
class JoltGpuBodyCommands final
{
  public:
    JoltGpuBodyCommands(rhi::ComputeHost &host, const ShaderTargetCode &target);
    [[nodiscard]] std::unique_ptr<JoltGpuCommandCursor> Prepare(const JoltGpuStorage &storage,
                                                                std::vector<JPH::GPU::BodyCommand> commands) const;

  private:
    rhi::ComputeHost m_host;
    std::shared_ptr<rhi::ComputeKernel> m_kernel;
    std::vector<uint8_t> m_ports;
};

/// One retained command plan. Pending batches survive Busy unchanged. Progress
/// advances at GPU acceptance, including acceptance followed by a submit error.
/// Shared buffers outlive storage; no host lease or device state readback is held.
class JoltGpuCommandCursor final
{
  public:
    ~JoltGpuCommandCursor();
    JoltGpuCommandCursor(const JoltGpuCommandCursor &) = delete;
    JoltGpuCommandCursor &operator=(const JoltGpuCommandCursor &) = delete;

    /// Budget a prefix of bounded dispatches: <=64 bodies and <=64 commands
    /// per body per dispatch. A pending prefix is returned verbatim. Hosts must
    /// share the original compute services; no accepted commands are resent.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch> PrepareNext(rhi::ComputeHost &host,
                                                                         uint32_t maxDispatches);
    /// Observation/preparation only. Submit through TrySubmit to commit progress.
    [[nodiscard]] const std::shared_ptr<rhi::PreparedComputeBatch> &GetBatch() const noexcept
    {
        return m_batch;
    }
    [[nodiscard]] std::optional<rhi::SubmissionTicket> TrySubmit();
    /// Replan an unaccepted prefix without changing the immutable plan or any
    /// accepted progress. Accepted batches can only advance through TrySubmit.
    void DiscardPreparedPrefix() noexcept;
    [[nodiscard]] bool IsComplete() const noexcept;
    [[nodiscard]] size_t GetAcceptedDispatchCount() const noexcept
    {
        return m_next;
    }
    [[nodiscard]] size_t GetDispatchCount() const noexcept
    {
        return m_dispatches.size();
    }

  private:
    friend class JoltGpuBodyCommands;
    struct Dispatch
    {
        uint32_t firstCommand, commandCount, firstSpan, spanCount, firstBody;
    };
    JoltGpuCommandCursor(const rhi::ComputeHost &host, std::shared_ptr<rhi::ComputeKernel> kernel,
                         std::vector<uint8_t> ports, const JoltGpuStorage &storage);
    void Accepted() noexcept;
    rhi::ComputeHost m_host;
    std::shared_ptr<rhi::ComputeKernel> m_kernel;
    std::vector<uint8_t> m_ports;
    std::shared_ptr<rhi::ComputeBuffer> m_models, m_states, m_forces;
    // Immutable after Prepare returns; no CPU mirror of live state exists.
    std::vector<JPH::GPU::BodyCommand> m_commands;
    std::vector<JPH::GPU::BodyCommandSpan> m_spans;
    std::vector<Dispatch> m_dispatches;
    std::shared_ptr<rhi::PreparedComputeBatch> m_batch;
    size_t m_next = 0, m_pendingEnd = 0;
};

} // namespace infernux
