#pragma once

#include <Jolt/GPU/ConstraintTypes.h>
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
    std::vector<JPH::GPU::JointAuthor> joints;
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

struct JoltGpuJointModelUpdate
{
    uint32_t index;
    JPH::GPU::JointAuthor author;
};

/// One bounded unit of the shared stage recipe. Parallel items use global
/// ordinals; island domains use local ordinals and an explicit island-table span.
struct JoltGpuStageWork
{
    JPH::GPU::Stage stage;
    JPH::GPU::StageWork parameters;
};

/// One substep's accepted sparse contact resources. Counts describe logical
/// candidates, never buffer capacity; writable contacts/arena share this lease.
struct JoltGpuContactWorkset
{
    uint32_t patchCount = 0, maxPatchesPerWorld = 0;
    std::shared_ptr<rhi::ComputeBuffer> parameters, worlds, contacts, patches;
};

/// One substep plan built on the GPU with the required CC checks. Only the fixed
/// structural summary is returned to the host for producer installation. Numerical
/// dispatches bind schedule as one transient port and retain its accepted lease.
struct JoltGpuConstraintWorkset
{
    uint32_t islandCount = 0, jointIslandCount = 0, contactIslandCount = 0;
    uint32_t activeJoints = 0, activePatches = 0, maxJointItems = 0, maxPatchItems = 0;
    std::shared_ptr<rhi::ComputeBuffer> schedule;
};

class JoltGpuModelUpdate;

/// Numerical endpoints for one accepted or prepared body-capacity domain.
struct JoltGpuBodyStatePorts
{
    uint32_t count = 0;
    std::shared_ptr<rhi::ComputeBuffer> states, forces, targets;
};

/// Candidate endpoints; author uploads and their installation remain one model transaction.
struct JoltGpuBodyGrowthBuffers
{
    std::shared_ptr<rhi::ComputeBuffer> models, worlds;
    JoltGpuBodyStatePorts bodies;
};

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
    [[nodiscard]] std::unique_ptr<JoltGpuModelUpdate> PrepareModelUpdate(
        std::vector<JoltGpuBodyModelUpdate> bodyChanges, std::vector<JoltGpuShapeModelUpdate> shapeChanges,
        std::vector<rhi::ComputeBufferUpdate> additionalUpdates = {},
        std::vector<rhi::ComputeDispatchDesc> mutations = {}, std::vector<JoltGpuJointModelUpdate> jointChanges = {});
    /// Replace leaf streams and world ranges at one accepted boundary. Body
    /// capacity/world count and the live state/force/target endpoints are retained.
    [[nodiscard]] std::unique_ptr<JoltGpuModelUpdate>
    PrepareModelReplacement(JoltGpuModel replacement, std::vector<rhi::ComputeBufferUpdate> additionalUpdates = {},
                            std::vector<rhi::ComputeDispatchDesc> mutations = {});
    /// Install a prepared larger body domain. Numerical prefix copy and tail
    /// initialization must already be complete; this packet uploads author
    /// tables and commits every endpoint/count together at acceptance.
    [[nodiscard]] std::unique_ptr<JoltGpuModelUpdate> PrepareBodyGrowth(uint32_t capacity,
                                                                        JoltGpuBodyGrowthBuffers buffers);
    /// Replace only joint topology and world joint ranges. The caller prepares
    /// the new stream's author upload and GPU runtime initialization/remap;
    /// this transaction never reads numerical joint state back to the host.
    [[nodiscard]] std::unique_ptr<JoltGpuModelUpdate>
    PrepareJointReplacement(std::vector<JPH::GPU::JointAuthor> joints, std::shared_ptr<rhi::ComputeBuffer> newStream,
                            std::vector<rhi::ComputeBufferUpdate> uploads = {},
                            std::vector<rhi::ComputeDispatchDesc> remapInit = {});
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
    [[nodiscard]] uint32_t GetJointCount() const noexcept
    {
        return static_cast<uint32_t>(m_model.joints.size());
    }
    [[nodiscard]] uint32_t GetShapeCount() const noexcept
    {
        return m_shapeCount;
    }
    [[nodiscard]] uint32_t GetMaxLeavesPerWorld() const noexcept
    {
        return m_maxLeavesPerWorld;
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
    std::array<std::shared_ptr<rhi::ComputeBuffer>, 11> m_buffers;
    uint32_t m_bodyCount = 0, m_worldCount = 0, m_shapeCount = 0, m_maxBodiesPerWorld = 0, m_maxLeavesPerWorld = 0;
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
                       std::vector<JoltGpuShapeModelUpdate> shapeChanges,
                       std::vector<JoltGpuJointModelUpdate> jointChanges);
    JoltGpuModelUpdate(JoltGpuStorage &storage, std::shared_ptr<rhi::PreparedComputeBatch> batch,
                       JoltGpuModel replacement, std::array<std::shared_ptr<rhi::ComputeBuffer>, 11> replacementBuffers,
                       std::array<uint32_t, 3> replacementCounts);
    void Commit() noexcept;

    JoltGpuStorage *m_storage;
    std::shared_ptr<rhi::PreparedComputeBatch> m_batch;
    std::vector<JoltGpuBodyModelUpdate> m_bodyChanges;
    std::vector<JoltGpuShapeModelUpdate> m_shapeChanges;
    std::vector<JoltGpuJointModelUpdate> m_jointChanges;
    std::optional<JoltGpuModel> m_replacement;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, 11> m_replacementBuffers;
    // Shapes, maximum bodies/world, maximum leaves/world.
    std::array<uint32_t, 3> m_replacementCounts{};
};

/// The five read-only input values of one logical tick. Capture each port
/// immediately after its final producer is accepted, without waiting for GPU
/// completion. Every stage prefix binds these leases, never the live endpoints.
class JoltGpuStepInputs
{
  public:
    /// Empty models need no input leases. Writable solver ports are not inputs.
    void Capture(const JoltGpuStorage &storage, JPH::GPU::Resource resource);
    /// Original frozen author inputs, even while contact work overrides bindings.
    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &GetCaptured(JPH::GPU::Resource resource) const;
    void SetContacts(std::shared_ptr<const JoltGpuContactWorkset> contacts);
    void ClearContacts() noexcept;
    [[nodiscard]] const std::shared_ptr<const JoltGpuContactWorkset> &GetContacts() const noexcept
    {
        return m_contacts;
    }

    void SetConstraints(std::shared_ptr<const JoltGpuConstraintWorkset> constraints) noexcept;
    void ClearConstraints() noexcept;
    [[nodiscard]] const std::shared_ptr<const JoltGpuConstraintWorkset> &GetConstraints() const noexcept
    {
        return m_constraints;
    }

  private:
    friend class JoltGpuPrograms;
    struct WorkUniform
    {
        JPH::GPU::StageWork parameters{};
        std::shared_ptr<rhi::ComputeBuffer> buffer;
    };
    std::array<std::shared_ptr<rhi::ComputeBuffer>, 5> m_buffers;
    std::shared_ptr<const JoltGpuContactWorkset> m_contacts;
    std::shared_ptr<const JoltGpuConstraintWorkset> m_constraints;
    // Immutable range values, not additional frozen model/input ports. Capture
    // starts a fresh bounded cache; an unaccepted upload is never a cache hit.
    mutable std::vector<WorkUniform> m_workUniforms;
    mutable size_t m_nextWorkUniform = 0;
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
    /// publish a physics tick. Every work item carries its exact bounded range;
    /// an empty list/model
    /// returns nullptr, without constructing a GPU packet.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch> PrepareStages(const JoltGpuStorage &storage,
                                                                           const JoltGpuStepInputs &inputs,
                                                                           const std::vector<JoltGpuStageWork> &work);

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
                                                         const JoltGpuStepInputs &inputs, const JoltGpuStageWork &work,
                                                         const std::shared_ptr<rhi::ComputeBuffer> &workUniform) const;
    static constexpr size_t StageCount = sizeof(JPH::GPU::Stages) / sizeof(JPH::GPU::Stages[0]);
    rhi::ComputeHost m_host;
    std::array<Entry, StageCount> m_entries;
};

} // namespace infernux
