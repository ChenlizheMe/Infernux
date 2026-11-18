#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Physics/EPhysicsUpdateError.h>
#include <Jolt/Physics/PhysicsSystem.h>
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
    void ConfigureCoupledFixed(JoltCpuExecution &, JoltCpuScene &, float deltaTime, uint32_t substeps,
                               ContactStreamIdentity, PhysicsContactPublication::Policy);
    void BeginCoupledSubstep(uint32_t substep, const std::vector<JPH::BodyID> &endpoints,
                             const std::vector<JPH::BodyID> &sleepEndpoints,
                             const std::vector<JPH::BodyID> &wakeEndpoints);
    // Each returns true when its phase finishes on this substep.
    bool AdvanceCoupledVelocity(const std::vector<JPH::PhysicsSystem::CoupledBodyImpulse> &impulses = {});
    void PrepareCoupledIntegration(bool retainMotion);
    [[nodiscard]] std::optional<float> CoupledImpactTime() const;
    [[nodiscard]] bool HasCoupledImpact(JPH::BodyID body) const;
    [[nodiscard]] JPH::Vec3 CoupledDisplacement(JPH::BodyID body) const;
    void AdvanceCoupledSweep(float fraction, float remaining,
                             const std::vector<JPH::PhysicsSystem::CoupledBodyImpulse> &impulses);
    void AdvanceCoupledIntegration();
    bool AdvanceCoupledPosition(const std::vector<JPH::PhysicsSystem::CoupledBodyCorrection> &, uint32_t rounds);
    [[nodiscard]] JPH::PhysicsUpdateContext::CoupledFreeVelocity CoupledFreeVelocity(JPH::BodyID body) const;
    [[nodiscard]] bool CoupledHasLocalVelocityRows() const noexcept;
    [[nodiscard]] JPH::PhysicsSystem::CoupledSleepIsland CoupledSleepIsland(JPH::BodyID endpoint) const;
    void FinishCoupledSleep(const std::vector<uint32_t> &sleepingIslands);
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
    struct CoupledStep;
    void PrepareObservation(JoltCpuScene &, uint32_t substeps, ContactStreamIdentity, PhysicsContactPublication::Policy,
                            std::function<void(uint32_t)>);
    void FinishObservation(JoltCpuScene &, uint32_t substeps, ContactStreamIdentity);
    bool AcceptCoupledRound(std::optional<JPH::EPhysicsUpdateError>, JPH::PhysicsUpdateContext::ECoupledStage);
    void CancelCoupled() noexcept;
    std::unique_ptr<Capture> m_capture;
    std::unique_ptr<CoupledStep> m_coupled;
    std::optional<ContactStreamIdentity> m_ready;
    std::optional<ContactStreamIdentity> m_published;
    bool m_observing = false;
};
} // namespace infernux
