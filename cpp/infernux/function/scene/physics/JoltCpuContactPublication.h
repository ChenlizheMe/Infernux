#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Physics/EPhysicsUpdateError.h>
#include "PhysicsContactPublication.h"

#include <memory>
#include <optional>

namespace infernux
{
class JoltCpuExecution;
class JoltCpuScene;

/// Samples one native Fixed step and hands complete CPU frames to the common publication.
class JoltCpuContactPublication final
{
  public:
    JoltCpuContactPublication();
    ~JoltCpuContactPublication();

    JPH::EPhysicsUpdateError Step(JoltCpuExecution &, JoltCpuScene &, float deltaTime, uint32_t substeps,
                                  ContactStreamIdentity, PhysicsContactPublication::Policy,
                                  std::function<void(uint32_t)> afterSubstep = {});
    /// Observe the previous prepared Fixed. Call Validate before taking frames.
    void ValidateCompleted(ContactStreamIdentity, uint32_t substeps) const;
    [[nodiscard]] PhysicsContactPublication::Frame TakeFrame(uint32_t substep);
    /// Accept history only after the common contact batch publishes successfully.
    void AcceptCompleted() noexcept;
    void CancelPending() noexcept;
    void Publish(PhysicsContactPublication &);
    void Close() noexcept;

  private:
    struct Capture;
    std::unique_ptr<Capture> m_capture;
    std::optional<ContactStreamIdentity> m_ready;
    std::optional<ContactStreamIdentity> m_published;
    bool m_observing = false;
};
} // namespace infernux
