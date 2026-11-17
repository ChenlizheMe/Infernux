#pragma once

#include "JoltGpuBodyCommands.h"
#include <core/threading/FrameWorkOwner.h>

#include <functional>
#include <memory>
#include <optional>
#include <utility>

namespace infernux
{
class JoltGpuBroadphase;
class JoltGpuNarrowphase;
class JoltGpuConstraints;
struct JoltGpuArticulationCapture;
struct JoltGpuArticulationState;

/// One unaccepted packet and its optional successor.
/// The existing tick source alone submits, accounts and observes every packet.
/// Continuations normally wait for their real receipt. A producer may explicitly
/// permit pure host preparation after acceptance when GPU dependencies suffice.
/// A continuation without a packet retains host preparation under capacity pressure;
/// the owner retries only at a new opportunity or after resource progress.
/// Neither form may submit, wait or reenter. Both absent means chain complete.
struct JoltGpuControlWork
{
    std::shared_ptr<rhi::PreparedComputeBatch> packet;
    // Exact immutable ranges touched by this packet, retained by its real
    // accepted receipt even if Cancel destroys the logical continuation.
    std::shared_ptr<const JoltGpuGeometryLease> geometry;
    std::function<JoltGpuControlWork()> continueWork;
    enum class ResumeAfter
    {
        Completion,
        Acceptance
    };
    ResumeAfter resumeAfter = ResumeAfter::Completion;
    // Optional multi-unit packet contraction before acceptance. It must
    // preserve the accepted producer cursor and return its minimum unit.
    std::function<JoltGpuControlWork()> prepareMinimum;

    JoltGpuControlWork() = default;
    JoltGpuControlWork(std::shared_ptr<rhi::PreparedComputeBatch> value) : packet(std::move(value))
    {
    }
    JoltGpuControlWork(std::shared_ptr<rhi::PreparedComputeBatch> value,
                       std::function<JoltGpuControlWork()> continuation, ResumeAfter boundary = ResumeAfter::Completion)
        : packet(std::move(value)), continueWork(std::move(continuation)), resumeAfter(boundary)
    {
    }
};

/// One ordered author-model boundary, per-body FIFO commands, then optional
/// deferred compute work. Factories run after preceding groups are accepted,
/// so storage never has more than one unaccepted model transaction. A null model result is a
/// metadata-only boundary. The acceptance callback runs once, including for a
/// null result; it must not throw, allocate, perform IO or submit GPU work.
struct JoltGpuControlGroup
{
    std::function<std::unique_ptr<JoltGpuModelUpdate>()> prepareModel;
    // Explicit null-capacity result, distinct from a metadata-only acceptance.
    std::function<bool()> modelDeferred;
    // Only a revoked, unaccepted model transaction may run this cold rebuild.
    std::function<JoltGpuControlWork()> reprepareModel;
    // This cold layout's private joint copies must observe converted histories.
    std::function<void()> retireJointRuntime;
    std::function<void()> modelAccepted;
    std::vector<JPH::GPU::BodyCommand> commands;
    std::function<void(std::vector<JPH::GPU::BodyCommand> &)> materializeCommands;
    /// Prepare once, after this group's model and commands are accepted. Its
    /// packet and completion-dependent successors share this source's admission
    /// and receipts. The group advances only after the chain's final acceptance.
    std::function<JoltGpuControlWork()> prepareWork;
};

/// Compact completed numerical output. Body slots stay global in the author
/// directory; readers validate [bodyBase,bodyBase+bodyCount) then subtract base.
/// Joint rows are local [0,jointCount), preserving stable author identity.
/// worldRow is an address in this accepted layout, never a stable world ID.
struct JoltGpuTickOutput
{
    uint32_t worldRow = 0, bodyBase = 0, bodyCount = 0, leafBase = 0, leafCount = 0, jointBase = 0, jointCount = 0;
    std::shared_ptr<rhi::ComputeBuffer> states, joints;
    std::shared_ptr<const JoltGpuJointLayout> jointLayout;
    std::shared_ptr<JoltGpuGeometryStore> geometryStore;
    std::shared_ptr<const JoltGpuGeometryLease> geometry;
    bool hasGenericGeometry = false;
};

/// Accepted through the tick's existing work source before solver inputs are
/// captured: initialization, then each group's model, commands and deferred work.
enum class JoltGpuContactReportStage
{
    BeforeVelocity,
    AfterVelocity
};

enum class JoltGpuCoupledContactStage
{
    Prepare,
    Velocity,
    Position
};

enum class JoltGpuParticipantStage : uint32_t
{
    Predict,
    Prepare,
    Exchange,
    Accept,
    Initialize
};

struct JoltGpuTickControl
{
    // Native Work factories only: no project Python or independent submissions.
    std::function<JoltGpuControlWork(JoltGpuParticipantStage, const JPH::GPU::StepParams &, float, uint32_t, uint32_t,
                                     std::function<void()>)>
        prepareParticipant;
    // Captured existing settings; elapsed physical dt is filled by Tick for
    // each actual substep. Never Host time or number of admitted packets.
    JPH::GPU::BodySleepParameters sleepParameters{};
    // Accepted author addresses only, not a Host mirror of motion/activity.
    // Runs after control groups. Scene retains stable owner IDs and resolves
    // packed slots here; shader decides whether each source actually moved.
    std::function<std::vector<uint32_t>()> prepareSleepSources;
    // Pure observations explicitly opt out; existing Tick/Control/Work retain snapshots.
    bool needsSnapshot = true;
    std::shared_ptr<const JoltGpuGeometryLease> acceptedGeometry;
    bool acceptedGenericGeometry = false;
    JoltGpuTickControl() = default;
    JoltGpuTickControl(JoltGpuTickControl &&) = default;
    JoltGpuTickControl &operator=(JoltGpuTickControl &&) = default;
    JoltGpuTickControl(const JoltGpuTickControl &) = delete;
    JoltGpuTickControl &operator=(const JoltGpuTickControl &) = delete;
    std::shared_ptr<rhi::PreparedComputeBatch> initialization;
    // Required when any group has commands; must outlive control acceptance.
    JoltGpuBodyCommands *commandProgram = nullptr;
    std::vector<JoltGpuControlGroup> groups;
    std::shared_ptr<JoltGpuColliderPairCapture> colliderPairs;
    std::shared_ptr<JoltGpuArticulationCapture> articulations;
    /// Opt-in producer only, using this tick's existing source and receipts.
    /// Call done exactly once after all phase writers are accepted/completed
    /// as required by their normal queue dependencies. Never submit or wait.
    /// Geometry is captured before warm/solve; lambdas after velocity, before
    /// integration. A missing producer preserves the ordinary stage path.
    std::function<JoltGpuControlWork(const JoltGpuStorage &, const JoltGpuStepInputs &, const JPH::GPU::StepParams &,
                                     uint32_t, JoltGpuContactReportStage, std::function<void()>)>
        prepareContactReport;
    // Prepare before native topology, then exchange after local velocity and position rounds.
    // With no local GPU rows, the caller may advance multiple independent rounds in one submission.
    std::function<JoltGpuControlWork(const JoltGpuStorage &, JoltGpuStepInputs &, const JPH::GPU::StepParams &,
                                     uint32_t, uint32_t, JoltGpuCoupledContactStage, uint32_t,
                                     std::function<void(uint32_t)>)>
        prepareCoupledContacts;
    /// Prepare exactly once after every numerical writer is accepted, including
    /// for a control-only/empty world. The input is the compact selected-world
    /// output; its states are null for a zero-body range. All packets use this source's admission and
    /// receipts; completion waits for the whole chain. Do not submit, wait or
    /// reenter the owner.
    std::function<JoltGpuControlWork(const JoltGpuTickOutput &)> prepareFinalWork;
};

/// One bounded numerical cursor under the engine's existing physics Source.
/// A primary cursor may own a clean sibling; both share one admission/queue driver.
/// Storage/programs/work must outlive the cursors and their accepted receipts.
class JoltGpuTick final
{
  public:
    enum class Status
    {
        Idle,
        Running,
        Complete,
        Cancelled,
        Failed
    };

    JoltGpuTick(FrameWorkOwner &work, JoltGpuStorage &storage, JoltGpuPrograms &programs, JoltGpuBroadphase &broadphase,
                JoltGpuNarrowphase &narrowphase, JoltGpuConstraints &constraints,
                std::function<void()> failureObserver = {});
    /// Explicit owner shutdown boundary: drains only this owner's accepted
    /// receipts before unregistering. Ordinary Stop first calls Cancel, retains
    /// this owner until CanRetire(), and only then destroys it without waiting.
    ~JoltGpuTick();
    JoltGpuTick(const JoltGpuTick &) = delete;
    JoltGpuTick &operator=(const JoltGpuTick &) = delete;

    /// An additional numerical cursor driven only by this Tick's existing Source.
    /// Scene admits it only under the clean packed-layout member exclusion.
    [[nodiscard]] JoltGpuTick &GetSharedMember(std::function<void()> failureObserver);
    void SetSourceFocus(JoltGpuTick *member) noexcept;
    [[nodiscard]] FrameWorkWaitResult WaitForSourceProgress(uint64_t maxWaitNanoseconds);

    /// Requires the packed backend's writer/layout exclusion. Only a Scene-admitted
    /// disjoint clean sibling may share this unchanged layout and Source. worldRow
    /// belongs to the accepted layout; controls may move its ranges before
    /// capture, but may not renumber its reserved row. Arbitrary controller/model/
    /// layout writers remain excluded until all relevant member receipts retire.
    /// Models/worlds and, when cpuForces is absent, the GPU force producer must
    /// already be accepted. Capture those values immediately here. CPU forces
    /// are followed by their genuine-input wake producer. A direct GPU force
    /// controller must have accepted its consumer-owned wake successor before
    /// Begin/control groups, so a later explicit Sleep retains FIFO precedence.
    /// No-input Begin never rescans a historical accumulator as a new write.
    /// CPU forces
    /// are moved into one parameter/force upload packet and captured only when
    /// it is accepted. dt is the logical fixed step; each substep uses dt/N.
    /// Kinematic preparation precedes all substeps and FinishTick follows them
    /// once. Target runtime state is live, outside the five captured inputs.
    /// The owner keeps solver state and the force accumulator exclusive until
    /// completion. Later controller results wait for the next control boundary;
    /// they must not overwrite this accumulator while its tick is running.
    void Begin(uint32_t worldRow, float fixedDeltaTime, const JoltGpuStepParameters &parameters,
               std::optional<std::vector<JPH::GPU::BodyForces>> cpuForces = std::nullopt);
    /// Control packets share admission, preparation, prediction and receipts
    /// with solver work. Commands freeze the new model only after its producer
    /// is accepted. Optional CPU forces explicitly replace the accumulator
    /// after commands as the final force producer. No completion wait separates
    /// these phases; the existing compute dependencies order accepted packets.
    void Begin(uint32_t worldRow, float fixedDeltaTime, const JoltGpuStepParameters &parameters,
               JoltGpuTickControl control, std::optional<std::vector<JPH::GPU::BodyForces>> cpuForces = std::nullopt);
    /// Admit creation/model/command boundaries through this same source without
    /// advancing simulation or uploading solver inputs. Completion publishes
    /// the resulting body states only after all control receipts retire.
    void BeginControl(uint32_t worldRow, JoltGpuTickControl control);

    /// Nonblocking. Unaccepted packets are revoked; accepted receipts remain
    /// observable through the same source. A cancelled/failed partial storage
    /// must be retired or reinitialized by the outer owner, never resumed here.
    void Cancel() noexcept;
    /// The outer owner has settled receipts and will admit only GPU cleanup
    /// of the cancelled partition before exposing any new numerical tick.
    void ReopenForPartitionCleanup();
    using PressureCompletion = std::function<void(uint32_t, std::shared_ptr<JoltGpuArticulationState>)>;
    /// Scene owner hooks; all retirement packets remain in this existing source.
    /// observe reports real retirement debt or caches requiring serial observation.
    void SetPressureOwner(std::function<bool()> observe, std::function<bool()> pending,
                          std::function<JoltGpuControlWork(uint32_t, PressureCompletion)> prepare);
    [[nodiscard]] bool IsPressureActive() const noexcept;

    /// The owning world controls when this source may prepare, submit or wait.
    /// Closing admission retains its cursor and packets; already accepted
    /// receipts can still retire. Reopening wakes the same source.
    void SetAdmissionEnabled(bool enabled) noexcept;
    [[nodiscard]] bool IsAdmissionEnabled() const noexcept;

    [[nodiscard]] FrameWorkOwner::SourceId GetSourceId() const noexcept;
    [[nodiscard]] Status GetStatus() const noexcept;
    [[nodiscard]] bool IsPending() const noexcept;
    [[nodiscard]] bool IsComplete() const noexcept;
    /// All control producers accepted, independently of GPU completion. Solver
    /// inputs and the optional CPU-force packet may still be pending.
    [[nodiscard]] bool AreControlInputsAccepted() const noexcept;
    /// Strict consumers may wait only this source's accepted work. A timeout
    /// means a real bounded queue wait; no receipts means a blocked dependency.
    [[nodiscard]] FrameWorkWaitResult WaitForProgress(uint64_t maxWaitNanoseconds);
    [[nodiscard]] bool HasReceipts() const noexcept;
    [[nodiscard]] bool CanRetire() const noexcept;
    [[nodiscard]] float GetFixedDeltaTime() const noexcept;
    /// Number of accepted bounded stage dispatch units in this logical tick;
    /// each phase may span several item/island units. Acceptance never replays one.
    [[nodiscard]] size_t GetAcceptedStageCount() const noexcept;
    /// Last complete compact world output, retained while a later tick runs.
    /// Its body state buffer is null until a nonempty range completes.
    /// Consumers must retain the output lease and its exact author directory.
    [[nodiscard]] const JoltGpuTickOutput &GetCompletedOutput() const noexcept;

  private:
    friend class JoltGpuScene;
    class Impl;
    explicit JoltGpuTick(JoltGpuTick &owner, std::function<void()> failureObserver, bool observation = false);
    JoltGpuTick &GetObservationMember();
    void ReleaseBodyBatchInputs() noexcept;
    void ReleaseObservation() noexcept;
    void TracePublicationRead(const char *boundary, uint32_t row) const noexcept;
    std::unique_ptr<Impl> m_impl;
    std::unique_ptr<JoltGpuTick> m_sharedMember;
    std::unique_ptr<JoltGpuTick> m_observationMember;
};

} // namespace infernux
