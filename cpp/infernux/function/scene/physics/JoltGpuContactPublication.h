#pragma once

#include "JoltGpuScene.h"
#include "PhysicsContactPublication.h"
#include <memory>

namespace infernux
{
class JoltGpuContactReports;

/// Adapts GPU reports to contact events and retains the Fixed snapshot that
/// synchronous GPU queries use during callback delivery.
class JoltGpuContactPublication final
{
  public:
    using Publication = JoltGpuScene::Publication;
    [[nodiscard]] JoltGpuScene::ContactObserver Prepare(JoltGpuContactReports &, const JoltGpuScene &, RuntimeWorld &,
                                                        const std::shared_ptr<const Publication> &, uint32_t substeps,
                                                        PhysicsContactPublication::Policy);
    [[nodiscard]] JoltGpuScene::ContactObserver Prepare(JoltGpuContactReports &,
                                                        const std::shared_ptr<const Publication> &, uint32_t substeps,
                                                        PhysicsContactPublication::Policy);
    void ValidateCompleted(const std::shared_ptr<const Publication> &, uint32_t substeps) const;
    /// Decode into this substep's existing ranges after ValidateCompleted.
    void AppendFrame(uint32_t substep, PhysicsContactPublication::Frame &) const;
    /// Retain the query lease for callbacks only after the common batch publishes successfully.
    void AcceptCompleted(std::shared_ptr<const Publication>) noexcept;
    void Publish(PhysicsContactPublication &, const std::shared_ptr<const Publication> &);
    void CancelPending() noexcept;
    void Close() noexcept;
    [[nodiscard]] const std::shared_ptr<const Publication> &GetPublication() const noexcept
    {
        return m_publication;
    }

  private:
    struct Pending;
    std::shared_ptr<Pending> m_pending;
    std::shared_ptr<const Publication> m_publication;
};
} // namespace infernux
