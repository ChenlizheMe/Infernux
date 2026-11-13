#pragma once

#include "RenderCoreApi.h"
#include "RhiComputeBuffer.h"
#include "RhiComputeHost.h"
#include "RhiDescriptors.h"
#include "RhiSubmission.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace infernux::rhi
{

class ComputeKernel;
class PreparedComputeBatch;
class ComputeWorkRecipe;
class ComputeWorkInstance;

struct ComputeBufferBinding
{
    uint32_t slot = 0;
    BindingType type = BindingType::StorageBuffer;
    bool readOnlyStorage = false;
};

enum class ComputeBufferAccess : uint8_t
{
    Read = 1,
    Write = 2,
    ReadWrite = 3,
};

struct ComputeBindingRange
{
    uint32_t binding = 0;
    uint64_t offset = 0;
    uint64_t byteSize = 0;
};

struct ComputeDispatchProof
{
    std::vector<ComputeBindingRange> reads;
    std::vector<ComputeBindingRange> writes;
};

/// Device-produced XYZ workgroup counts; not a shader resource binding.
struct ComputeIndirectArguments
{
    static constexpr uint64_t ByteSize = 3 * sizeof(uint32_t);
    std::shared_ptr<ComputeBuffer> buffer;
    uint64_t offset = 0;
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
    std::optional<ComputeDispatchProof> proof;
    /// groupCountXYZ are finite algorithmic upper bounds for indirect work.
    std::optional<ComputeIndirectArguments> indirectArguments;
};

struct ComputeBufferUpdate
{
    std::shared_ptr<ComputeBuffer> buffer;
    uint64_t offset = 0;
    std::vector<uint8_t> data;
};

struct ComputeBufferCopy
{
    std::shared_ptr<ComputeBuffer> source;
    std::shared_ptr<ComputeBuffer> destination;
    BufferCopyRegion region;
};

/// Validate bindings and capture upload bytes once, before queue admission.
INFERNUX_RENDER_CORE_API std::shared_ptr<PreparedComputeBatch>
PrepareComputeBatch(ComputeHost &host, std::vector<ComputeBufferUpdate> updates,
                    std::vector<ComputeDispatchDesc> dispatches, std::vector<ComputeBufferRead> reads = {},
                    std::vector<ComputeBufferCopy> copies = {});
/// Record several kernel launches into one engine-owned command buffer.
INFERNUX_RENDER_CORE_API SubmissionTicket SubmitComputeBatch(ComputeHost &host,
                                                             std::vector<ComputeDispatchDesc> dispatches);
INFERNUX_RENDER_CORE_API SubmissionTicket SubmitComputeBatch(ComputeHost &host,
                                                             std::vector<ComputeBufferUpdate> updates,
                                                             std::vector<ComputeDispatchDesc> dispatches);
/// Record uploads, dependent compute dispatches, and their requested
/// readbacks into one queue submission, then wait once and return the host
/// bytes. This is the synchronization boundary for compact CPU consumers.
INFERNUX_RENDER_CORE_API std::vector<std::vector<uint8_t>>
SubmitComputeBatchAndRead(ComputeHost &host, std::vector<ComputeBufferUpdate> updates,
                          std::vector<ComputeDispatchDesc> dispatches, std::vector<ComputeBufferRead> reads);

/// Engine-owned executable compute pipeline. Compiler integrations provide
/// SPIR-V and binding metadata; this object owns device binding, submission,
/// and in-flight resource retirement through the shared engine RHI.
/// Kernels are shared-owned so submissions can retain them through completion.
class ComputeKernel final : public std::enable_shared_from_this<ComputeKernel>
{
  public:
    INFERNUX_RENDER_CORE_API ComputeKernel(ComputeHost &host, const uint32_t *spirvWords, size_t wordCount,
                                           uint32_t bufferBindingCount, uint32_t pushConstantBytes);
    INFERNUX_RENDER_CORE_API ComputeKernel(ComputeHost &host, const uint32_t *spirvWords, size_t wordCount,
                                           std::vector<uint32_t> bufferBindings, uint32_t pushConstantBytes);
    INFERNUX_RENDER_CORE_API ComputeKernel(ComputeHost &host, const uint32_t *spirvWords, size_t wordCount,
                                           std::vector<ComputeBufferBinding> bufferBindings,
                                           uint32_t pushConstantBytes);
    INFERNUX_RENDER_CORE_API ComputeKernel(ComputeHost &host, const ShaderModuleDesc &shader,
                                           const std::string &entryPoint,
                                           std::vector<ComputeBufferBinding> bufferBindings,
                                           uint32_t pushConstantBytes);
    INFERNUX_RENDER_CORE_API ~ComputeKernel();

    ComputeKernel(const ComputeKernel &) = delete;
    ComputeKernel &operator=(const ComputeKernel &) = delete;

    [[nodiscard]] INFERNUX_RENDER_CORE_API uint32_t GetBufferBindingCount() const noexcept;
    [[nodiscard]] INFERNUX_RENDER_CORE_API const std::vector<uint32_t> &GetBufferBindings() const noexcept;
    [[nodiscard]] INFERNUX_RENDER_CORE_API uint32_t GetPushConstantBytes() const noexcept;
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
    INFERNUX_RENDER_CORE_API void Dispatch(std::vector<std::shared_ptr<ComputeBuffer>> buffers,
                                           const void *pushConstants, uint32_t pushConstantBytes, uint32_t groupCountX,
                                           uint32_t groupCountY, uint32_t groupCountZ);
    INFERNUX_RENDER_CORE_API void Collect();
    INFERNUX_RENDER_CORE_API void Wait();

    /// Validated, immutable dispatch payload. RecordOne neither submits nor
    /// attaches a compute ticket. A graphics caller must retain this owner
    /// through its actual submission completion; Wait() cannot retire it.
    struct PreparedDispatch
    {
      public:
        PreparedDispatch(PreparedDispatch &&) noexcept = default;
        PreparedDispatch &operator=(PreparedDispatch &&) noexcept = default;
        PreparedDispatch(const PreparedDispatch &) = delete;
        PreparedDispatch &operator=(const PreparedDispatch &) = delete;

        [[nodiscard]] INFERNUX_RENDER_CORE_API bool RecordOne(const ComputeCommandEncoder &encoder) const;
        [[nodiscard]] bool IsIndirect() const noexcept
        {
            return indirectOffset.has_value();
        }
        [[nodiscard]] const std::vector<std::shared_ptr<ComputeBufferAllocation>> &GetAllocations() const noexcept
        {
            return allocations;
        }
        [[nodiscard]] const std::vector<ComputeBufferAccess> &GetBufferAccesses() const noexcept
        {
            return bufferAccesses;
        }
        [[nodiscard]] INFERNUX_RENDER_CORE_API const std::vector<BindingType> &GetBufferBindingTypes() const noexcept;

      private:
        PreparedDispatch() = default;
        friend class ComputeKernel;
        friend class PreparedComputeBatch;

        std::shared_ptr<ComputeKernel> kernel;
        ComputePipelineHandle pipeline;
        BindGroupHandle group;
        std::vector<std::shared_ptr<ComputeBufferAllocation>> allocations;
        std::vector<std::shared_ptr<ComputeBufferState>> states;
        std::vector<ComputeBufferAccess> bufferAccesses;
        std::vector<uint8_t> pushConstants;
        uint32_t groupCountX = 1;
        uint32_t groupCountY = 1;
        uint32_t groupCountZ = 1;
        std::optional<ComputeDispatchProof> proof;
        /// The final allocation/state is the indirect read, outside shader bindings.
        std::optional<uint64_t> indirectOffset;
    };

    /// Prepare once for either a background compute batch or a frame graph.
    /// The caller supplies resource barriers, recording and completion ownership.
    [[nodiscard]] INFERNUX_RENDER_CORE_API PreparedDispatch PrepareDispatch(
        std::vector<std::shared_ptr<ComputeBuffer>> buffers, std::vector<ComputeBufferAccess> bufferAccesses,
        std::vector<uint8_t> pushConstants, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ,
        std::optional<ComputeDispatchProof> proof = {}, std::optional<ComputeIndirectArguments> indirectArguments = {});

  private:
    struct BatchDispatch
    {
        PreparedDispatch work;
        bool barrierBefore = false;
    };

    struct PreparedBatch
    {
        struct Update
        {
            std::shared_ptr<ComputeBufferAllocation> allocation;
            std::shared_ptr<ComputeBufferState> state;
            uint64_t offset;
            std::vector<uint8_t> data;
        };
        struct Copy
        {
            std::shared_ptr<ComputeBufferAllocation> source, destination;
            std::shared_ptr<ComputeBufferState> sourceState, destinationState;
            BufferCopyRegion region;
            bool preserveInitialization = false;
        };
        std::vector<Update> updates;
        std::vector<BatchDispatch> dispatches;
        std::vector<Copy> copies;
        std::vector<std::shared_ptr<ComputeBufferAllocation>> updatedBuffers;
        std::vector<std::shared_ptr<ComputeBufferAllocation>> writtenBuffers;
        std::vector<std::shared_ptr<ComputeKernel>> kernels;
        std::vector<std::shared_ptr<ComputeReadback::Storage>> readOwners;
        std::vector<std::shared_ptr<ComputeBufferState>> contentReadOwners;
        bool hasIndirect = false;
    };

    struct PendingDispatch
    {
        SubmissionTicket ticket;
    };

    struct BufferValueState
    {
        std::shared_ptr<ComputeBufferAllocation> allocation;
        intervals::Set initialized;
        intervals::Set written;
    };

    struct BindingGroupEntry
    {
        BindGroupHandle group;
        std::vector<std::weak_ptr<ComputeBufferAllocation>> allocations;
    };

    [[nodiscard]] BindGroupHandle
    ResolveBindGroup(const std::vector<std::shared_ptr<ComputeBufferAllocation>> &allocations);
    void ValidateDispatch(const std::vector<ComputeBufferAccess> &accesses, size_t pushConstantBytes,
                          uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ,
                          bool indirect = false) const;
    [[nodiscard]] static PreparedBatch PrepareBatch(ComputeHost &host, std::vector<ComputeBufferUpdate> updates,
                                                    std::vector<ComputeDispatchDesc> dispatches,
                                                    std::vector<ComputeBufferCopy> copies = {});
    [[nodiscard]] static bool RecordBatch(ComputeRecordingContext &context, const PreparedBatch &batch,
                                          const std::vector<std::shared_ptr<ComputeReadback>> &reads);
    [[nodiscard]] static std::vector<BufferValueState> PrepareValues(const PreparedBatch &batch);
    void Attach(SubmissionTicket ticket);
    void CollectBindingGroups();
    void Destroy() noexcept;

    friend class PreparedComputeBatch;
    friend class ComputeWorkRecipe;

    ComputeHost m_host;
    BindingLayoutHandle m_layout;
    ComputePipelineHandle m_pipeline;
    std::vector<BindingGroupEntry> m_groups;
    std::vector<uint32_t> m_bufferBindings;
    std::vector<BindingType> m_bufferBindingTypes;
    std::vector<ComputeBufferAccess> m_defaultBufferAccesses;
    uint32_t m_pushConstantBytes = 0;
    std::vector<PendingDispatch> m_pending;
};

struct ComputeWorkRange
{
    uint32_t slot = 0;
    uint64_t offset = 0;
    uint64_t byteSize = 0;
};

struct ComputeWorkSlot
{
    ComputeBufferDesc allocation;
    /// Absent for per-instance storage; external storage is always read-only.
    std::shared_ptr<ComputeBuffer> external;
};

struct ComputeWorkPort
{
    std::string name;
    ComputeWorkRange range;
    bool input = false;
};

struct ComputeWorkBinding
{
    ComputeWorkRange range;
    ComputeBufferAccess access = ComputeBufferAccess::Read;
};

struct ComputeWorkDispatch
{
    std::shared_ptr<ComputeKernel> kernel;
    std::vector<ComputeWorkBinding> bindings;
    /// Proven complete writes for this whole-domain execution, associated
    /// with actual binding roles and bounded by their exact logical ranges.
    std::vector<ComputeBindingRange> completeWrites;
    std::vector<uint8_t> pushConstants;
    uint32_t groupCountX = 1;
    uint32_t groupCountY = 1;
    uint32_t groupCountZ = 1;
    /// An exact twelve-byte execution input in a declared storage slot.
    std::optional<ComputeWorkRange> indirectArguments;
};

struct ComputeWorkInput
{
    std::string port;
    std::shared_ptr<ComputeBuffer> buffer;
};

/// A finite, validated resource/program recipe, independent of a graph domain.
/// External producers must already be accepted; their tickets establish only
/// dependencies. Actual external read ranges and private input ports require
/// caller-provided valid contents, not allocation-wide initialization proof.
class ComputeWorkRecipe final : public std::enable_shared_from_this<ComputeWorkRecipe>
{
  public:
    INFERNUX_RENDER_CORE_API ComputeWorkRecipe(ComputeHost &host, std::vector<ComputeWorkSlot> slots,
                                               std::vector<ComputeWorkPort> ports,
                                               std::vector<ComputeWorkDispatch> dispatches);
    [[nodiscard]] const std::vector<ComputeWorkSlot> &GetSlots() const noexcept
    {
        return m_slots;
    }
    [[nodiscard]] const std::vector<ComputeWorkPort> &GetPorts() const noexcept
    {
        return m_ports;
    }
    [[nodiscard]] const std::vector<ComputeWorkDispatch> &GetDispatches() const noexcept
    {
        return m_dispatches;
    }
    [[nodiscard]] INFERNUX_RENDER_CORE_API std::vector<ComputeWorkRange> GetRequiredInputs() const;
    [[nodiscard]] ComputeHost &GetHost() noexcept
    {
        return m_host;
    }
    [[nodiscard]] const ComputeHost &GetHost() const noexcept
    {
        return m_host;
    }
    /// Reuse matching real input owners before allocating remaining private
    /// slots. Returns null when storage cannot be admitted. No GPU work is submitted.
    [[nodiscard]] INFERNUX_RENDER_CORE_API std::shared_ptr<ComputeWorkInstance>
    Instantiate(std::vector<ComputeWorkInput> inputs = {},
                const std::vector<std::shared_ptr<ComputeBuffer>> &externalInputs = {}) const;

  private:
    friend class ComputeWorkInstance;
    [[nodiscard]] std::shared_ptr<ComputeWorkInstance>
    Bind(std::vector<ComputeWorkInput> inputs, const std::vector<std::shared_ptr<ComputeBuffer>> &retained,
         const std::vector<std::shared_ptr<ComputeBuffer>> &externalInputs) const;
    ComputeHost m_host;
    std::vector<ComputeWorkSlot> m_slots;
    std::vector<ComputeWorkPort> m_ports;
    std::unordered_map<std::string, size_t> m_portIndices;
    std::vector<ComputeWorkRange> m_requiredInputs;
    std::vector<ComputeWorkDispatch> m_dispatches;
};

/// Prepared native resources only. The caller must establish complete input
/// writers before recording and retain this instance to actual GPU completion.
/// ComputeKernel::Wait cannot stand in for a Graphics completion epoch.
class ComputeWorkInstance final
{
  public:
    /// Rebuild immutable dispatch bindings while reusing existing private slots.
    [[nodiscard]] INFERNUX_RENDER_CORE_API std::shared_ptr<ComputeWorkInstance>
    Rebind(std::vector<ComputeWorkInput> inputs,
           const std::vector<std::shared_ptr<ComputeBuffer>> &externalInputs = {}) const;
    /// Lower the bound recipe through the same batch preparation and writer
    /// versioning used by individual compute dispatches. Does not submit work.
    [[nodiscard]] INFERNUX_RENDER_CORE_API std::shared_ptr<PreparedComputeBatch>
    PrepareBatch(std::vector<ComputeBufferUpdate> updates = {}) const;
    [[nodiscard]] const std::vector<std::shared_ptr<ComputeBuffer>> &GetBuffers() const noexcept
    {
        return m_buffers;
    }
    /// Graphics recording needs immutable bindings; batch execution prepares
    /// its own writer versions. Materialize these only for the recording path.
    [[nodiscard]] INFERNUX_RENDER_CORE_API const std::vector<ComputeKernel::PreparedDispatch> &GetDispatches() const;
    [[nodiscard]] INFERNUX_RENDER_CORE_API std::vector<ComputeWorkRange> GetRequiredInputs() const;
    [[nodiscard]] const ComputeWorkRecipe &GetRecipe() const noexcept
    {
        return *m_recipe;
    }

  private:
    friend class ComputeWorkRecipe;
    ComputeWorkInstance(std::shared_ptr<const ComputeWorkRecipe> recipe,
                        std::vector<std::shared_ptr<ComputeBuffer>> buffers);
    std::shared_ptr<const ComputeWorkRecipe> m_recipe;
    std::vector<std::shared_ptr<ComputeBuffer>> m_buffers;
    mutable std::vector<ComputeKernel::PreparedDispatch> m_dispatches;
};

/// One accepted compute submission, independent of its originating batch.
/// Only metadata is retained: no service lease or GPU resources. Query on the
/// engine owner thread, serialized with compute submission and device teardown.
class ComputeCompletion final
{
  public:
    /// Poll without waiting. Queue-published completion stays true after teardown;
    /// a retired host cannot resolve work that never actually completed.
    [[nodiscard]] INFERNUX_RENDER_CORE_API bool IsComplete();
    /// Observe retirement already published by the queue without polling it.
    [[nodiscard]] bool HasRetired() const noexcept
    {
        return m_receipt->complete;
    }
    /// Explicit synchronization for this ticket, including shutdown. Frame
    /// continuation uses IsComplete; this never admits unrelated queued work.
    INFERNUX_RENDER_CORE_API void Wait();
    [[nodiscard]] SubmissionTicket GetSubmission() const noexcept
    {
        return m_ticket;
    }
    [[nodiscard]] const ComputeHost &GetHost() const noexcept
    {
        return m_host;
    }
    /// The measured command region, not host waiting or exclusive GPU active time.
    /// Pure metadata: empty until collected, or when timestamps are unavailable.
    [[nodiscard]] std::optional<double> GetDeviceElapsedNanoseconds() const noexcept
    {
        return m_receipt->deviceNanoseconds;
    }

  private:
    friend class PreparedComputeBatch;
    ComputeCompletion(const ComputeHost &host, SubmissionTicket ticket,
                      std::shared_ptr<ComputeExecutionReceipt> receipt)
        : m_host(host), m_ticket(ticket), m_receipt(std::move(receipt))
    {
    }

    ComputeHost m_host;
    SubmissionTicket m_ticket;
    std::shared_ptr<ComputeExecutionReceipt> m_receipt;
};

/// One prepared submission. Busy preserves its payload; acceptance consumes it.
/// The queue retains GPU owners until completion, independently of this handle.
class PreparedComputeBatch final
{
  public:
    INFERNUX_RENDER_CORE_API ~PreparedComputeBatch();
    PreparedComputeBatch(const PreparedComputeBatch &) = delete;
    PreparedComputeBatch &operator=(const PreparedComputeBatch &) = delete;

    [[nodiscard]] INFERNUX_RENDER_CORE_API SubmissionTicket Submit();
    [[nodiscard]] INFERNUX_RENDER_CORE_API std::optional<SubmissionTicket> TrySubmit();
    /// Requires acceptance; the returned ticket does not retain this batch.
    [[nodiscard]] INFERNUX_RENDER_CORE_API ComputeCompletion GetCompletion() const;
    [[nodiscard]] INFERNUX_RENDER_CORE_API bool HasWork() const noexcept;
    /// Declared writers still awaiting acceptance; reads never make a writer.
    [[nodiscard]] INFERNUX_RENDER_CORE_API bool HasPendingWrite(const ComputeBuffer &buffer) const noexcept;
    using WriterStateSet = std::unordered_set<const ComputeBufferState *>;
    /// Borrowed identities for an uninterrupted observation of the existing FIFO.
    INFERNUX_RENDER_CORE_API void AppendPendingWriterStates(WriterStateSet &states) const;
    /// Exact retained buffer capacities and CPU byte-vector payload; callers
    /// deduplicate storage by handle. These queries never collect or submit.
    INFERNUX_RENDER_CORE_API void AppendRetainedStorage(std::vector<ComputeStorageBlock> &blocks) const;
    [[nodiscard]] INFERNUX_RENDER_CORE_API uint64_t GetRetainedPayloadBytes() const noexcept;
    [[nodiscard]] INFERNUX_RENDER_CORE_API uint64_t GetPendingStagingBytes() const noexcept;
    /// Real upload/copy/readback traffic; acceptance fixes this count even if
    /// a readback consumer is cancelled afterwards.
    [[nodiscard]] INFERNUX_RENDER_CORE_API uint64_t GetTransferBytes() const noexcept;
    [[nodiscard]] const std::vector<uint64_t> &GetExecutionSignature() const noexcept
    {
        return m_executionSignature;
    }
    /// Host preparation only: resolve the latest logical endpoints and return
    /// one bounded preservation copy before the original writer may submit.
    [[nodiscard]] INFERNUX_RENDER_CORE_API bool RequiresPreparation() const;
    [[nodiscard]] INFERNUX_RENDER_CORE_API uint64_t GetPreparationBytes() const;
    [[nodiscard]] INFERNUX_RENDER_CORE_API bool HasPreservationWork() const noexcept;
    struct PreservationPreparation
    {
        bool capacityGranted;
        std::shared_ptr<PreparedComputeBatch> copy;
    };
    [[nodiscard]] INFERNUX_RENDER_CORE_API PreservationPreparation PreparePreservation(uint64_t maxBytes, bool managed);
    /// Revoke unaccepted payload and endpoints, even while result handles live.
    /// Accepted GPU work cannot be cancelled and remains queue-owned.
    INFERNUX_RENDER_CORE_API bool Cancel() noexcept;
    [[nodiscard]] INFERNUX_RENDER_CORE_API std::shared_ptr<ComputeReadback> Readback(size_t index) const;
    [[nodiscard]] SubmissionTicket GetSubmission() const noexcept
    {
        return m_ticket;
    }
    [[nodiscard]] ComputeHost &GetHost() noexcept
    {
        return m_host;
    }

  private:
    PreparedComputeBatch(ComputeHost &host, std::vector<ComputeBufferUpdate> updates,
                         std::vector<ComputeDispatchDesc> dispatches, std::vector<ComputeBufferRead> reads,
                         std::vector<ComputeBufferCopy> copies);
    [[nodiscard]] std::optional<SubmissionTicket> SubmitInternal(bool block);
    void Accepted(SubmissionTicket ticket);
    [[nodiscard]] bool ResolveStorage(bool managed);
    [[nodiscard]] bool PrepareReadbacks(bool managed);
    [[nodiscard]] bool OverwritesBeforeRead(const ComputeBufferState &state) const;
    template <typename Visitor> bool VisitWriters(Visitor &&visitor) const;
    [[nodiscard]] std::vector<std::shared_ptr<ComputeBufferState>> WriterStates() const;

    friend INFERNUX_RENDER_CORE_API std::shared_ptr<PreparedComputeBatch>
    PrepareComputeBatch(ComputeHost &host, std::vector<ComputeBufferUpdate> updates,
                        std::vector<ComputeDispatchDesc> dispatches, std::vector<ComputeBufferRead> reads,
                        std::vector<ComputeBufferCopy> copies);

    ComputeHost m_host;
    std::shared_ptr<ComputeKernel::PreparedBatch> m_batch;
    std::vector<std::shared_ptr<ComputeReadback>> m_readbacks;
    std::shared_ptr<ComputeExecutionReceipt> m_receipt = std::make_shared<ComputeExecutionReceipt>();
    SubmissionTicket m_ticket;
    uint64_t m_uploadBytes = 0;
    uint64_t m_uploadStagingBytes = 0;
    uint64_t m_copyBytes = 0;
    uint64_t m_transferBytes = 0;
    uint64_t m_dispatchCount = 0;
    std::vector<uint64_t> m_executionSignature;
    struct WriteVersion
    {
        std::shared_ptr<ComputeBufferState> target;
        std::shared_ptr<ComputeBufferAllocation> base;
        std::shared_ptr<ComputeBuffer> source, candidate;
        intervals::Set preserve;
        size_t region = 0;
        uint64_t offset = 0;
    };
    std::vector<WriteVersion> m_writeVersions;
    std::shared_ptr<PreparedComputeBatch> m_preservation;
    size_t m_preservationVersion = 0;
    uint64_t m_preservationEnd = 0;
};

} // namespace infernux::rhi
