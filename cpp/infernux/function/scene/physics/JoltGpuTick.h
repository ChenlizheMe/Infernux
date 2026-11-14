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
class JoltGpuConstraints;

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
    std::function<JoltGpuControlWork()> continueWork;
    enum class ResumeAfter
    {
        Completion,
        Acceptance
    };
    ResumeAfter resumeAfter = ResumeAfter::Completion;

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
};

/// Accepted through the tick's existing work source before solver inputs are
/// captured: initialization, then each group's model, commands and deferred work.
struct JoltGpuTickControl
{
    JoltGpuTickControl() = default;
    JoltGpuTickControl(JoltGpuTickControl &&) = default;
    JoltGpuTickControl &operator=(JoltGpuTickControl &&) = default;
    JoltGpuTickControl(const JoltGpuTickControl &) = delete;
    JoltGpuTickControl &operator=(const JoltGpuTickControl &) = delete;
    std::shared_ptr<rhi::PreparedComputeBatch> initialization;
    // Required when any group has commands; must outlive control acceptance.
    JoltGpuBodyCommands *commandProgram = nullptr;
    std::vector<JoltGpuControlGroup> groups;
    /// Prepare exactly once after every numerical writer is accepted, including
    /// for a control-only/empty world. The input is the compact selected-world
    /// output; its states are null for a zero-body range. All packets use this source's admission and
    /// receipts; completion waits for the whole chain. Do not submit, wait or
    /// reenter the owner.
    std::function<JoltGpuControlWork(const JoltGpuTickOutput &)> prepareFinalWork;
};

/// One PhysicsWorld-owned bounded stage cursor using the engine's existing work source,
/// compute queue and receipts. Storage/programs/work must outlive this owner.
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
                JoltGpuConstraints &constraints);
    /// Explicit owner shutdown boundary: drains only this owner's accepted
    /// receipts before unregistering. Ordinary Stop first calls Cancel, retains
    /// this owner until CanRetire(), and only then destroys it without waiting.
    ~JoltGpuTick();
    JoltGpuTick(const JoltGpuTick &) = delete;
    JoltGpuTick &operator=(const JoltGpuTick &) = delete;

    /// Requires the shared packed backend's exclusive writer lease. worldRow
    /// belongs to the accepted layout; controls may move its ranges before
    /// capture, but may not renumber its reserved row. No other numerical
    /// writer/controller/layout commit may enter until this cursor retires.
    /// Models/worlds and, when cpuForces is absent, the GPU force producer must
    /// already be accepted. Capture those values immediately here. CPU forces
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
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace infernux
