#pragma once

#include <Jolt/GPU/ConstraintTypes.h>
#include <Jolt/GPU/PhysicsStages.h>
#include <Jolt/GPU/SleepTypes.h>
#include "JoltGpuJointLayout.h"
#include "PhysicsContactTypes.h"
#include <function/renderer/rhi/RhiComputeKernel.h>
#include <function/renderer/shader/ShaderProgramCode.h>

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace infernux
{
class JoltGpuGeometryStore;
class JoltGpuSleep;
class JoltGpuArticulations;
struct JoltGpuArticulationState;
struct JoltGpuGeometryLease;
// One accepted policy prefix, including every endpoint's full ID/generation.
// Old and new generations of one ID may coexist while a Scene is retained.
struct JoltGpuColliderPairs
{
    std::shared_ptr<const PhysicsColliderPairs> identities;
    std::vector<JPH::GPU::ColliderPairKey> keys;
    std::shared_ptr<rhi::ComputeBuffer> buffer;
};
struct JoltGpuColliderPairCapture
{
    std::shared_ptr<const JoltGpuColliderPairs> value;
};

/// Device records and immutable geometry for the first GPU rigid-body slice.
/// No CPU PhysicsSystem or Jolt Body memory is uploaded through this interface.
struct JoltGpuModel
{
    std::vector<JPH::GPU::BodyModel> bodies;
    std::vector<JPH::GPU::ShapeModel> shapes;
    std::vector<JPH::GPU::WorldRange> worlds;
    std::vector<JPH::GPU::JointAuthor> joints;
};

/// The only body/leaf/joint WorldRange builder for accepted packed models.
/// Bodies/shapes/joints are already in world order. Rebind global leaf owners
/// and author world rows while validating complete capacity coverage.
void JoltGpuBuildWorldRanges(JoltGpuModel &model, const std::vector<uint32_t> &capacities);

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
    uint32_t runtimeIntent = 0;
};

/// One bounded unit of the shared stage recipe. Parallel items use global
/// ordinals; island domains use local ordinals and an explicit island-table span.
struct JoltGpuStageWork
{
    JPH::GPU::Stage stage;
    JPH::GPU::StageWork parameters;
    uint32_t islandDispatch = 0;
};

/// The selected substep's dispatch extent. The candidate header and shader
/// parameters own the logical prefix even when the host uses a capacity bound.
struct JoltGpuContactWorkset
{
    enum class Range : uint32_t
    {
        Pairs,
        Setup,
        History,
        RangeCount
    };

    uint32_t patchSlots = 0;
    std::shared_ptr<rhi::ComputeBuffer> parameters, contacts, patches;
    // Per tile, in Range order. The shader writes actual workload sizes when
    // the host has only a capacity for the contact stream.
    std::shared_ptr<rhi::ComputeBuffer> countedArguments;
    uint32_t patchesPerDispatch = 0;

    [[nodiscard]] std::optional<rhi::ComputeIndirectArguments> Dispatch(Range range, uint32_t firstPatch) const
    {
        if (!countedArguments)
            return {};
        const uint64_t record =
            uint64_t(firstPatch / patchesPerDispatch) * uint32_t(Range::RangeCount) + uint32_t(range);
        return rhi::ComputeIndirectArguments{countedArguments, record * sizeof(JPH::GPU::UInt4)};
    }
};

inline uint32_t JoltGpuStageItemLimit(JPH::GPU::Stage stage, JPH::GPU::Domain domain,
                                      const JoltGpuContactWorkset *contacts) noexcept
{
    using namespace JPH::GPU;
    if (contacts && contacts->countedArguments) {
        if (stage == Stage::SetupContacts)
            return contacts->patchesPerDispatch * MaxContactPointsPerPair;
        if (stage == Stage::StoreContactHistory)
            return contacts->patchesPerDispatch;
    }
    return WorkItemLimit(domain);
}

// Sorted by the preceding substep's canonical leaf pair. Only the stored
// prefix is readable when a larger device allocation is reused.
struct JoltGpuContactHistory
{
    std::shared_ptr<rhi::ComputeBuffer> buffer;
    uint32_t patchCount = 0;

    [[nodiscard]] uint64_t ByteSize() const noexcept
    {
        return (1u + uint64_t(patchCount) * JPH::GPU::ContactHistoryWords) * sizeof(JPH::GPU::UInt4);
    }
};

/// Capacity-bounded dispatches whose effective ranges are produced by the GPU.
struct JoltGpuIslandDispatch
{
    JPH::GPU::StageWork parameters;
    uint32_t argumentRecord;
};

/// One accepted substep plan on the tick's queue. Island counts, source
/// ranges and indirect arguments stay on the device between producer and solver.
struct JoltGpuConstraintWorkset
{
    std::shared_ptr<rhi::ComputeBuffer> schedule, solveArguments;
    uint32_t bodyRootFirst = 0, bodyFirst = 0, bodyCount = 0;
    uint32_t jointRefFirst = 0, jointRefCount = 0, patchRefFirst = 0, patchRefCount = 0;
    std::shared_ptr<rhi::ComputeBuffer> rootVotes;
    std::vector<JoltGpuIslandDispatch> jointDispatches, patchDispatches;

    [[nodiscard]] const std::vector<JoltGpuIslandDispatch> &Dispatches(bool joints) const noexcept
    {
        return joints ? jointDispatches : patchDispatches;
    }
};

// One topology, before awake-domain numerical setup/finalization.
// Schedule is a source-exclusive live producer, not an installable frozen plan.
// The opaque execution is the SAME CC producer resumed after setup.
struct JoltGpuConstraintTopology
{
    uint32_t bodyRootFirst = 0, bodyFirst = 0, bodyCount = 0;
    std::shared_ptr<rhi::ComputeBuffer> schedule, rootVotes;
    std::shared_ptr<void> execution;
    // The accepted graph gates all wake consumers. The counted patch command
    // can carry a smaller effective range than its allocated candidate slots.
    std::shared_ptr<rhi::ComputeBuffer> wakeArguments;
    std::optional<rhi::ComputeIndirectArguments> wakePatchDispatch;
};

class JoltGpuModelUpdate;

/// Numerical endpoints for one accepted or prepared body-capacity domain.
struct JoltGpuBodyStatePorts
{
    uint32_t count = 0;
    std::shared_ptr<rhi::ComputeBuffer> states, forces, targets, sleeps;
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
    PrepareInputs(const JoltGpuStepParameters &parameters, const std::vector<JPH::GPU::BodyForces> *cpuForces = nullptr,
                  const std::shared_ptr<rhi::ComputeBuffer> &parameterBuffer = {});
    /// Explicit CPU input path. A GPU controller can instead write the resident
    /// BodyForces port directly. Its consumer owner must place
    /// GetSleep().PrepareForceWake immediately after that genuinely accepted
    /// writer, BEFORE any later author groups/Sleep command, using the existing
    /// control-work source. A no-upload Begin is not a new force producer;
    /// historical nonzero accumulators never imply a new wake event.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    PrepareForces(const std::vector<JPH::GPU::BodyForces> &forces, const JPH::GPU::StepParams &selected);
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
    /// Accept one completed general packed-layout transaction. New numeric
    /// ports already contain their GPU-remapped complete values. Author/model
    /// resources are prepared here and every endpoint commits at one acceptance.
    [[nodiscard]] std::unique_ptr<JoltGpuModelUpdate>
    PreparePackedReplacement(JoltGpuModel replacement, JoltGpuBodyStatePorts bodies,
                             std::shared_ptr<rhi::ComputeBuffer> jointStream);
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
    [[nodiscard]] JoltGpuContactHistory GetContactHistory(uint32_t worldRow) const;
    void PublishContactHistory(uint32_t worldRow, const JoltGpuContactHistory &history);
    void ClearContactHistory(uint32_t worldRow) noexcept;
    /// Accepted author metadata only; never a CPU mirror of dynamic body state.
    [[nodiscard]] const JoltGpuModel &GetModel() const noexcept
    {
        return m_model;
    }
    [[nodiscard]] const std::shared_ptr<const JoltGpuJointLayout> &GetJointLayout() const noexcept
    {
        return m_jointLayout;
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
    [[nodiscard]] bool HasSweepBodies(uint32_t world) const noexcept
    {
        return m_sweepCounts[world].enabled != 0u;
    }
    [[nodiscard]] bool HasDynamicSweeps(uint32_t world) const noexcept
    {
        return m_sweepCounts[world].dynamicPairs != 0u;
    }
    struct DiscretePairLimits
    {
        uint64_t candidates, projections;
        uint32_t activeLeaves, dynamicBodies;
    };
    [[nodiscard]] const DiscretePairLimits &GetDiscretePairLimits(uint32_t world) const;
    [[nodiscard]] uint32_t GetMaxLeavesPerWorld() const noexcept
    {
        return m_maxLeavesPerWorld;
    }
    [[nodiscard]] uint32_t GetMaxBodiesPerWorld() const noexcept
    {
        return m_maxBodiesPerWorld;
    }
    void SetGeometryStore(std::shared_ptr<JoltGpuGeometryStore> store);
    void SetGeometry(std::shared_ptr<const JoltGpuGeometryLease> geometry, bool hasGeneric) noexcept
    {
        m_geometry = std::move(geometry);
        m_hasGenericGeometry = hasGeneric;
    }
    [[nodiscard]] bool HasGenericGeometry() const noexcept
    {
        return m_hasGenericGeometry;
    }
    [[nodiscard]] const std::shared_ptr<JoltGpuGeometryStore> &GetGeometryStore() const noexcept
    {
        return m_geometryStore;
    }
    [[nodiscard]] const std::shared_ptr<const JoltGpuGeometryLease> &GetGeometry() const noexcept
    {
        return m_geometry;
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
    std::shared_ptr<JoltGpuGeometryStore> m_geometryStore;
    std::shared_ptr<const JoltGpuGeometryLease> m_geometry;
    bool m_hasGenericGeometry = false;
    std::shared_ptr<const JoltGpuJointLayout> m_jointLayout;
    bool m_modelUpdatePending = false;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, 12> m_buffers;
    struct SweepCounts
    {
        uint32_t enabled = 0u, dynamicPairs = 0u;
        mutable std::optional<DiscretePairLimits> discrete;
    };
    static std::vector<SweepCounts> CountSweeps(const JoltGpuModel &model);
    std::vector<SweepCounts> m_sweepCounts;
    std::map<uint32_t, JoltGpuContactHistory> m_contactHistory;
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
                       JoltGpuModel replacement, std::array<std::shared_ptr<rhi::ComputeBuffer>, 12> replacementBuffers,
                       std::array<uint32_t, 3> replacementCounts);
    void Commit() noexcept;

    JoltGpuStorage *m_storage;
    std::shared_ptr<const JoltGpuJointLayout> m_jointLayout;
    std::shared_ptr<rhi::PreparedComputeBatch> m_batch;
    std::vector<JoltGpuBodyModelUpdate> m_bodyChanges;
    std::vector<JoltGpuShapeModelUpdate> m_shapeChanges;
    std::vector<JoltGpuJointModelUpdate> m_jointChanges;
    std::optional<JoltGpuModel> m_replacement;
    std::vector<JoltGpuStorage::SweepCounts> m_replacementSweepCounts;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, 12> m_replacementBuffers;
    // Shapes, maximum bodies/world, maximum leaves/world.
    std::array<uint32_t, 3> m_replacementCounts{};
};

/// Immutable parameter/author values and a borrowed live force endpoint for
/// one selected-world tick. The shared packed writer lease excludes later
/// producers until this cursor releases it. Forces never acquire a content
/// reader; same-queue accepted reads precede this tick's final force clear.
class JoltGpuStepInputs
{
  public:
    /// Empty models need no input leases. Writable solver ports are not inputs.
    void Capture(const JoltGpuStorage &storage, JPH::GPU::Resource resource);
    void CaptureParameters(const JoltGpuStorage &storage, const std::shared_ptr<rhi::ComputeBuffer> &buffer);
    void CaptureGeometry(std::shared_ptr<const JoltGpuGeometryLease> geometry, bool generic);
    void SetSelection(const JPH::GPU::StepParams &parameters) noexcept
    {
        m_parameters = parameters;
    }
    [[nodiscard]] const JPH::GPU::StepParams &GetParameters() const noexcept
    {
        return m_parameters;
    }
    [[nodiscard]] const std::shared_ptr<const JoltGpuGeometryLease> &GetGeometry() const noexcept
    {
        return m_geometry;
    }
    [[nodiscard]] bool HasGenericGeometry() const noexcept
    {
        return m_hasGenericGeometry;
    }
    /// Original frozen author inputs, even while contact work overrides bindings.
    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &GetCaptured(JPH::GPU::Resource resource) const;
    void SetColliderPairs(std::shared_ptr<const JoltGpuColliderPairs> pairs)
    {
        m_colliderPairs = std::move(pairs);
    }
    [[nodiscard]] const std::shared_ptr<const JoltGpuColliderPairs> &GetColliderPairs() const
    {
        return m_colliderPairs;
    }
    void SetArticulations(std::shared_ptr<JoltGpuArticulationState> value) noexcept
    {
        m_articulations = std::move(value);
    }
    [[nodiscard]] const std::shared_ptr<JoltGpuArticulationState> &GetArticulations() const noexcept
    {
        return m_articulations;
    }
    void SetContacts(std::shared_ptr<const JoltGpuContactWorkset> contacts);
    void ClearContacts() noexcept;
    [[nodiscard]] const std::shared_ptr<const JoltGpuContactWorkset> &GetContacts() const noexcept
    {
        return m_contacts;
    }
    void SetSweepBodies(std::shared_ptr<rhi::ComputeBuffer> bodies) noexcept
    {
        m_sweepBodies = std::move(bodies);
    }
    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &GetSweepBodies() const noexcept
    {
        return m_sweepBodies;
    }
    void SetSweepChoices(std::shared_ptr<rhi::ComputeBuffer> choices) noexcept
    {
        m_sweepChoices = std::move(choices);
    }
    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &GetSweepChoices() const noexcept
    {
        return m_sweepChoices;
    }
    void SetSweepClock(std::shared_ptr<rhi::ComputeBuffer> clock) noexcept
    {
        m_sweepClock = std::move(clock);
    }
    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &GetSweepClock() const noexcept
    {
        return m_sweepClock;
    }
    void SetSweepContacts(std::shared_ptr<const JoltGpuContactWorkset> contacts) noexcept
    {
        m_sweepContacts = std::move(contacts);
    }
    [[nodiscard]] const std::shared_ptr<const JoltGpuContactWorkset> &GetSweepContacts() const noexcept
    {
        return m_sweepContacts;
    }

    void SetContactHistory(JoltGpuContactHistory previous, JoltGpuContactHistory next) noexcept;
    [[nodiscard]] const JoltGpuContactHistory &PreviousContactHistory() const noexcept
    {
        return m_previousContacts;
    }
    [[nodiscard]] const JoltGpuContactHistory &NextContactHistory() const noexcept
    {
        return m_nextContacts;
    }

    void SetConstraints(std::shared_ptr<const JoltGpuConstraintWorkset> constraints) noexcept;
    void ClearConstraints() noexcept;
    [[nodiscard]] const std::shared_ptr<const JoltGpuConstraintWorkset> &GetConstraints() const noexcept
    {
        return m_constraints;
    }

  private:
    friend class JoltGpuPrograms;
    JPH::GPU::StepParams m_parameters{};
    std::shared_ptr<const JoltGpuGeometryLease> m_geometry;
    bool m_hasGenericGeometry = false;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, 5> m_buffers;
    std::shared_ptr<rhi::ComputeBuffer> m_sweepBodies;
    std::shared_ptr<rhi::ComputeBuffer> m_sweepChoices;
    std::shared_ptr<rhi::ComputeBuffer> m_sweepClock;
    std::shared_ptr<const JoltGpuContactWorkset> m_sweepContacts;
    std::shared_ptr<const JoltGpuContactWorkset> m_contacts;
    JoltGpuContactHistory m_previousContacts, m_nextContacts;
    std::shared_ptr<const JoltGpuColliderPairs> m_colliderPairs;
    std::shared_ptr<const JoltGpuConstraintWorkset> m_constraints;
    std::shared_ptr<JoltGpuArticulationState> m_articulations;
};

/// One ready body unit. The caller retains the member/cursor and clean layout lease.
/// These input pointers are consumed synchronously and never escape preparation.
struct JoltGpuBodyBatchMember
{
    const JoltGpuStepInputs *inputs;
    JoltGpuStageWork work;
};

/// Source/Lane-owned reusable header, selection and tile allocations. Pure
/// Compute FIFO protects accepted reads; reset on pressure/exclusive retirement,
/// while accepted packet owners retain the allocations they actually reference.
struct JoltGpuBodyBatchInputs
{
    std::array<std::shared_ptr<rhi::ComputeBuffer>, 3> buffers;
};

/// A consumer of the common compiler's typed target artifact. This class does
/// not load sources, own a compiler/cache/device, or submit a second GPU queue.
class JoltGpuPrograms
{
  public:
    JoltGpuPrograms(rhi::ComputeHost &host, const ShaderTargetCode &target,
                    const ShaderTargetCode *captureTarget = nullptr, const ShaderTargetCode *dynamicsTarget = nullptr);
    [[nodiscard]] bool TryPrepareStartupPrograms(bool hasLocalContacts);
    [[nodiscard]] JoltGpuSleep &GetSleep() const noexcept
    {
        return *m_sleep;
    }

    [[nodiscard]] JoltGpuArticulations &GetArticulations() const noexcept
    {
        return *m_articulations;
    }

    /// Append one member's units using its own input bindings; no cursor or GPU work is accepted.
    /// On failure discard all unaccepted descriptors. The source prepares one packet after all appends.
    void AppendStages(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                      const std::vector<JoltGpuStageWork> &work, std::vector<rhi::ComputeBufferUpdate> &uploads,
                      std::vector<rhi::ComputeDispatchDesc> &dispatches);

    /// Prepare one dependency-ordered ready prefix, including repeated solver
    /// iterations. The caller owns its stage cursor and selects a bounded prefix.
    /// Preparation does not submit; accepting the resulting batch does not
    /// publish a physics tick. Every work item carries its exact bounded range;
    /// an empty list/model
    /// returns nullptr, without constructing a GPU packet.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch> PrepareStages(const JoltGpuStorage &storage,
                                                                           const JoltGpuStepInputs &inputs,
                                                                           const std::vector<JoltGpuStageWork> &work);

    /// One true body-tile dispatch, with the three immutable tables uploaded in
    /// the same packet. Members must share one of the three body stages and
    /// one accepted layout; total padded work is bounded by WorkItemLimit(Bodies).
    /// nullptr means managed input allocation deferred: retain every cursor and
    /// retry unchanged. This method neither submits nor accepts member work.
    [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch>
    PrepareBodyBatch(const JoltGpuStorage &storage, const std::vector<JoltGpuBodyBatchMember> &members,
                     JoltGpuBodyBatchInputs &inputs);

  private:
    friend class JoltGpuBroadphase;
    // Append only: same Tick producer, packet admission and real receipts.
    void AppendResidentCollide(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                               const std::shared_ptr<const JoltGpuContactWorkset> &contacts, uint32_t capacity,
                               const JPH::GPU::StageWork &work, const std::shared_ptr<rhi::ComputeBuffer> &uniform,
                               const rhi::ComputeIndirectArguments &arguments,
                               std::vector<rhi::ComputeBufferUpdate> &uploads,
                               std::vector<rhi::ComputeDispatchDesc> &dispatches);
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
        std::shared_ptr<rhi::ComputeKernel> directKernel;
    };
    [[nodiscard]] rhi::ComputeDispatchDesc
    BuildDispatch(const Entry &entry, const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                  const JoltGpuStageWork &work, const std::shared_ptr<rhi::ComputeBuffer> &workUniform,
                  const std::shared_ptr<const JoltGpuContactWorkset> &resident = {}) const;
    struct BodyBatchEntry
    {
        std::shared_ptr<rhi::ComputeKernel> kernel;
        std::vector<Binding> bindings;
    };
    struct WorkUniform
    {
        std::array<uint32_t, 4> range;
        std::shared_ptr<rhi::ComputeBuffer> buffer;
    };
    static constexpr size_t StageCount = sizeof(JPH::GPU::Stages) / sizeof(JPH::GPU::Stages[0]);
    rhi::ComputeHost m_host;
    std::shared_ptr<JoltGpuSleep> m_sleep;
    std::shared_ptr<JoltGpuArticulations> m_articulations;
    std::shared_ptr<rhi::ComputeBuffer> m_emptyArticulationControl;
    std::array<Entry, StageCount> m_entries;
    std::array<BodyBatchEntry, 3> m_bodyBatchEntries;
    // Each initialized buffer contains only a fixed dispatch range. The storage
    // and physics inputs may change while that value stays reusable on this device.
    std::vector<WorkUniform> m_workUniforms;
    size_t m_nextWorkUniform = 0;
};

} // namespace infernux
