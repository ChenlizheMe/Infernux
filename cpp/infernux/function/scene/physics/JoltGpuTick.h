#pragma once

#include "JoltGpuPrograms.h"
#include <core/threading/FrameWorkOwner.h>

#include <memory>
#include <optional>

namespace infernux
{

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
    void Begin(float fixedDeltaTime, const JoltGpuStepParameters &parameters,
               std::optional<std::vector<JPH::GPU::BodyForces>> cpuForces = std::nullopt);

    /// Nonblocking. Unaccepted packets are revoked; accepted receipts remain
    /// observable through the same source. A cancelled/failed partial storage
    /// must be retired or reinitialized by the outer owner, never resumed here.
    void Cancel() noexcept;

    [[nodiscard]] FrameWorkOwner::SourceId GetSourceId() const noexcept;
    [[nodiscard]] Status GetStatus() const noexcept;
    [[nodiscard]] bool IsPending() const noexcept;
    [[nodiscard]] bool IsComplete() const noexcept;
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
