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
                                  ContactStreamIdentity, PhysicsContactPublication::Policy);
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
