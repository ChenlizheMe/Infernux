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
