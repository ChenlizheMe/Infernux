#pragma once

#include "JoltGpuBodyCommands.h"
#include <core/threading/FrameWorkOwner.h>

#include <functional>
#include <memory>
#include <optional>

namespace infernux
{

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
    /// Prepare once, after this group's model and commands are accepted. The
    /// returned unaccepted packet shares this source's admission and receipts.
    /// Null means no GPU work; preparation must not submit the packet itself.
    std::function<std::shared_ptr<rhi::PreparedComputeBatch>()> prepareWork;
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
};

/// One PhysicsWorld-owned stage cursor using the engine's existing work source,
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

    JoltGpuTick(FrameWorkOwner &work, JoltGpuStorage &storage, JoltGpuPrograms &programs);
    /// Explicit owner shutdown boundary: drains only this owner's accepted
    /// receipts before unregistering. Ordinary Stop first calls Cancel, retains
    /// this owner until CanRetire(), and only then destroys it without waiting.
    ~JoltGpuTick();
    JoltGpuTick(const JoltGpuTick &) = delete;
    JoltGpuTick &operator=(const JoltGpuTick &) = delete;

    /// Models/worlds and, when cpuForces is absent, the GPU force producer must
    /// already be accepted. Capture those values immediately here. CPU forces
    /// are moved into one parameter/force upload packet and captured only when
    /// it is accepted. dt is the logical fixed step; each substep uses dt/N.
    /// The owner keeps solver state and the force accumulator exclusive until
    /// completion. Later controller results wait for the next control boundary;
    /// they must not overwrite this accumulator while its tick is running.
    void Begin(float fixedDeltaTime, const JoltGpuStepParameters &parameters,
               std::optional<std::vector<JPH::GPU::BodyForces>> cpuForces = std::nullopt);
    /// Control packets share admission, preparation, prediction and receipts
    /// with solver work. Commands freeze the new model only after its producer
    /// is accepted. Optional CPU forces explicitly replace the accumulator
    /// after commands as the final force producer. No completion wait separates
    /// these phases; the existing compute dependencies order accepted packets.
    void Begin(float fixedDeltaTime, const JoltGpuStepParameters &parameters, JoltGpuTickControl control,
               std::optional<std::vector<JPH::GPU::BodyForces>> cpuForces = std::nullopt);
    /// Admit creation/model/command boundaries through this same source without
    /// advancing simulation or uploading solver inputs. Completion publishes
    /// the resulting body states only after all control receipts retire.
    void BeginControl(JoltGpuTickControl control);

    /// Nonblocking. Unaccepted packets are revoked; accepted receipts remain
    /// observable through the same source. A cancelled/failed partial storage
    /// must be retired or reinitialized by the outer owner, never resumed here.
    void Cancel() noexcept;

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
    [[nodiscard]] size_t GetAcceptedStageCount() const noexcept;
    /// Last fully completed tick, retained while a later tick runs. Null until
    /// the first nonempty tick completes. No partial solver value is exposed.
    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &GetCompletedBodyStates() const noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace infernux
