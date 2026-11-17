#pragma once

#include "JoltGpuPrograms.h"

#include <functional>
#include <variant>

namespace infernux
{

class JoltGpuCommandCursor;

struct JoltGpuPoseWake
{
    struct Work
    {
        uint32_t source, first, count;
        bool capture;
    };

    static constexpr uint32_t BodiesPerDispatch = 2048u;
    uint32_t bodyCount = 0;
    std::function<void(const Work &, std::vector<rhi::ComputeBufferUpdate> &, std::vector<rhi::ComputeDispatchDesc> &)>
        append;
};

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
                                                                std::vector<JPH::GPU::BodyCommand> commands,
                                                                JoltGpuPoseWake poseWake = {}) const;

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

    /// Budget a prefix of bounded dispatches. A pending prefix is returned verbatim. Hosts must
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
        return m_dispatchCount;
    }

  private:
    friend class JoltGpuBodyCommands;
    struct BodyDispatch
    {
        uint32_t firstCommand, commandCount, firstSpan, spanCount, firstBody;
    };
    using Dispatch = std::variant<BodyDispatch, JoltGpuPoseWake::Work>;
    using CommandIterator = std::vector<JPH::GPU::BodyCommand>::const_iterator;
    struct Location
    {
        size_t index = 0;
        uint32_t offset = 0;
    };
    JoltGpuCommandCursor(const rhi::ComputeHost &host, std::shared_ptr<rhi::ComputeKernel> kernel,
                         std::vector<uint8_t> ports, const JoltGpuStorage &storage, JoltGpuPoseWake poseWake);
    void AppendCommands(CommandIterator begin, CommandIterator end);
    void Accepted() noexcept;
    rhi::ComputeHost m_host;
    std::shared_ptr<rhi::ComputeKernel> m_kernel;
    std::vector<uint8_t> m_ports;
    std::shared_ptr<rhi::ComputeBuffer> m_models, m_states, m_forces, m_targets, m_sleeps;
    JoltGpuPoseWake m_poseWake;
    // Immutable after Prepare returns; no CPU mirror of live state exists.
    std::vector<JPH::GPU::BodyCommand> m_commands;
    std::vector<JPH::GPU::BodyCommandSpan> m_spans;
    std::vector<Dispatch> m_dispatches;
    std::shared_ptr<rhi::PreparedComputeBatch> m_batch;
    size_t m_dispatchCount = 0, m_next = 0, m_pendingEnd = 0;
    Location m_location, m_pendingLocation;
};

} // namespace infernux
