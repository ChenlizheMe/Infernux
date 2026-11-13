#pragma once

#include <Jolt/GPU/PhysicsStages.h>
#include <function/renderer/rhi/RhiComputeKernel.h>
#include <function/renderer/shader/ShaderProgramCode.h>

#include <array>
#include <memory>
#include <optional>
#include <vector>

namespace infernux
{

/// Device records and immutable geometry for the first GPU rigid-body slice.
/// No CPU PhysicsSystem or Jolt Body memory is uploaded through this interface.
struct JoltGpuModel
{
    std::vector<JPH::GPU::BodyModel> bodies;
    std::vector<JPH::GPU::ShapeModel> shapes;
    std::vector<JPH::GPU::WorldRange> worlds;
};

struct JoltGpuBodyModelUpdate
{
    uint32_t index;
    JPH::GPU::BodyModel model;
};

struct JoltGpuShapeModelUpdate
{
    uint32_t index;
    JPH::GPU::ShapeModel model;
};

class JoltGpuModelUpdate;

/// One validated step description, reusing the shared device record. Contact
/// settings and iteration counts are prepared once; only gravity/dt change in
/// place during normal stepping. Zero velocity/position iterations are legal.
class JoltGpuStepParameters
{
  public:
    explicit JoltGpuStepParameters(JPH::GPU::StepParams parameters);
    void SetGravityAndDeltaTime(JPH::GPU::Float4 gravityDt);
    [[nodiscard]] const JPH::GPU::StepParams &Get() const noexcept
    {
        return m_parameters;
    }

  private:
    JPH::GPU::StepParams m_parameters;
};

/// Physics-owned resident values. Compilation, submission, budgets, fences and
/// resource retirement remain services of the shared engine compute runtime.
class JoltGpuStorage
{
  public:
    JoltGpuStorage(rhi::ComputeHost &host, JoltGpuModel model);
    JoltGpuStorage(const JoltGpuStorage &) = delete;
    JoltGpuStorage &operator=(const JoltGpuStorage &) = delete;

    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    PrepareInitialization(const std::vector<JPH::GPU::BodyState> &states,
                          const std::vector<JPH::GPU::BodyForces> &forces);
    /// Updating step parameters never overwrites a device-produced force input.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch> PrepareParameters(const JoltGpuStepParameters &parameters);
    /// One CPU-input packet. A null forces pointer leaves the GPU-produced
    /// force port untouched. Capture these values only after packet acceptance.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    PrepareInputs(const JoltGpuStepParameters &parameters,
                  const std::vector<JPH::GPU::BodyForces> *cpuForces = nullptr);
    /// Explicit CPU input path. A GPU controller can instead write the resident
    /// BodyForces port directly; the normal compute dependency orders its use.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    PrepareForces(const std::vector<JPH::GPU::BodyForces> &forces);
    /// Fixed-capacity author-model transaction; existing GPU state is retained.
    /// Additional uploads/dispatches share its acceptance boundary, for example
    /// a new slot's initial state or an explicit device-side COM mutation.
    /// Mutations may update runtime state only; author-model ports remain owned
    /// by the typed deltas, including when dispatches share this packet.
    /// Only one unaccepted model transaction may exist. Storage must outlive it.
    [[nodiscard]] std::unique_ptr<JoltGpuModelUpdate>
    PrepareModelUpdate(std::vector<JoltGpuBodyModelUpdate> bodyChanges,
                       std::vector<JoltGpuShapeModelUpdate> shapeChanges,
                       std::vector<rhi::ComputeBufferUpdate> additionalUpdates = {},
                       std::vector<rhi::ComputeDispatchDesc> mutations = {});
    // Initialization/step-input preparation returns nullptr for an empty model;
    // the caller completes that phase without preparing or submitting a packet.

    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &Get(JPH::GPU::Resource resource) const;
    /// Accepted author metadata only; never a CPU mirror of dynamic body state.
    [[nodiscard]] const JoltGpuModel &GetModel() const noexcept
    {
        return m_model;
    }
    [[nodiscard]] uint32_t GetBodyCount() const noexcept
    {
        return m_bodyCount;
    }
    [[nodiscard]] uint32_t GetWorldCount() const noexcept
    {
        return m_worldCount;
    }
    [[nodiscard]] uint32_t GetContactCount() const noexcept
    {
        return m_contactCount;
    }
    [[nodiscard]] uint32_t GetMaxBodiesPerWorld() const noexcept
    {
        return m_maxBodiesPerWorld;
    }
    [[nodiscard]] const rhi::ComputeHost &GetHost() const noexcept
    {
        return m_host;
    }

  private:
    friend class JoltGpuPrograms;
    friend class JoltGpuModelUpdate;
    rhi::ComputeHost m_host;
    // Accepted author metadata; dynamic state remains exclusively on the GPU.
    JoltGpuModel m_model;
    bool m_modelUpdatePending = false;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, 9> m_buffers;
    uint32_t m_bodyCount = 0, m_worldCount = 0, m_contactCount = 0, m_maxBodiesPerWorld = 0;
};

/// One model delta and its exact RHI packet. GPU acceptance commits the author
/// metadata, not GPU completion. Destruction revokes an unaccepted packet even
/// when another observer retains GetBatch(). Storage must outlive this owner.
class JoltGpuModelUpdate final
{
  public:
    ~JoltGpuModelUpdate();
    JoltGpuModelUpdate(const JoltGpuModelUpdate &) = delete;
    JoltGpuModelUpdate &operator=(const JoltGpuModelUpdate &) = delete;

    /// Preparation/observation only, including budgets and preservation copies.
    /// Submit the original packet exclusively through this owner's TrySubmit().
    [[nodiscard]] const std::shared_ptr<rhi::PreparedComputeBatch> &GetBatch() const noexcept
    {
        return m_batch;
    }
    [[nodiscard]] std::optional<rhi::SubmissionTicket> TrySubmit();
    void Cancel() noexcept;

  private:
    friend class JoltGpuStorage;
    JoltGpuModelUpdate(JoltGpuStorage &storage, std::shared_ptr<rhi::PreparedComputeBatch> batch,
                       std::vector<JoltGpuBodyModelUpdate> bodyChanges,
                       std::vector<JoltGpuShapeModelUpdate> shapeChanges);
    void Commit() noexcept;

    JoltGpuStorage *m_storage;
    std::shared_ptr<rhi::PreparedComputeBatch> m_batch;
    std::vector<JoltGpuBodyModelUpdate> m_bodyChanges;
    std::vector<JoltGpuShapeModelUpdate> m_shapeChanges;
};

/// The five read-only input values of one logical tick. Capture each port
/// immediately after its final producer is accepted, without waiting for GPU
/// completion. Every stage prefix binds these leases, never the live endpoints.
class JoltGpuStepInputs
{
  public:
    /// Empty models need no input leases. Writable solver ports are not inputs.
    void Capture(const JoltGpuStorage &storage, JPH::GPU::Resource resource);

  private:
    friend class JoltGpuPrograms;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, 5> m_buffers;
};

/// A consumer of the common compiler's typed target artifact. This class does
/// not load sources, own a compiler/cache/device, or submit a second GPU queue.
class JoltGpuPrograms
{
  public:
    JoltGpuPrograms(rhi::ComputeHost &host, const ShaderTargetCode &target);

    /// Prepare one dependency-ordered ready prefix, including repeated solver
    /// iterations. The caller owns its stage cursor and selects a bounded prefix.
    /// Preparation does not submit; accepting the resulting batch does not
    /// publish a physics tick. A single stage is {stage}; an empty list/model
    /// returns nullptr, without constructing a GPU packet.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch> PrepareStages(const JoltGpuStorage &storage,
                                                                           const JoltGpuStepInputs &inputs,
                                                                           const std::vector<JPH::GPU::Stage> &stages);

  private:
    struct Binding
    {
        size_t slot;
        rhi::ComputeBufferAccess access;
        bool read;
        bool completeWrite;
    };
    struct Entry
    {
        JPH::GPU::StageDefinition definition;
        std::shared_ptr<rhi::ComputeKernel> kernel;
        std::vector<Binding> bindings;
    };
    [[nodiscard]] rhi::ComputeDispatchDesc BuildDispatch(const Entry &entry, const JoltGpuStorage &storage,
                                                         const JoltGpuStepInputs &inputs) const;
    static constexpr size_t StageCount = sizeof(JPH::GPU::Stages) / sizeof(JPH::GPU::Stages[0]);
    rhi::ComputeHost m_host;
    std::array<Entry, StageCount> m_entries;
};

} // namespace infernux
