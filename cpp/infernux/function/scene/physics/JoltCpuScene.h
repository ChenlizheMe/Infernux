#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Core/JobSystemSingleThreaded.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include "InfernuxJoltJobSystemAdapter.h"

#include <cstdint>
#include <unordered_map>

namespace infernux
{
struct EngineConfig;

/// Shared scratch and worker access, not another pool for each RuntimeWorld.
/// The owner advances CPU worlds serially; Jolt may parallelize a world's step
/// on the engine workers. No GPU device or host-frame budget is involved.
class JoltCpuExecution final
{
  public:
    explicit JoltCpuExecution(const EngineConfig &config);
    JPH::EPhysicsUpdateError Step(JPH::PhysicsSystem &system, float deltaTime, int collisionSteps);

  private:
    JPH::TempAllocatorImpl m_scratch;
    JPH::JobSystemSingleThreaded m_inlineJobs;
    InfernuxJoltJobSystemAdapter m_workerJobs;
};

/// Native state owned by one logical World. Public identities are supplied by
/// the common actor store; Jolt indices/sequence numbers never escape this
/// backend. Layer filters and any installed listeners outlive the scene.
class JoltCpuScene final
{
  public:
    JoltCpuScene(const EngineConfig &config, const JPH::BroadPhaseLayerInterface &layers,
                 const JPH::ObjectVsBroadPhaseLayerFilter &broadphaseFilter,
                 const JPH::ObjectLayerPairFilter &pairFilter);

    /// Creation does not enter broadphase. The common actor lifecycle publishes
    /// the body through Jolt's normal single/batch add operations.
    JPH::BodyID CreateBody(uint32_t bodyId, const JPH::BodyCreationSettings &settings);
    /// Caller has detached this body's constraints before destruction.
    void DestroyBody(uint32_t bodyId);
    [[nodiscard]] JPH::BodyID NativeBody(uint32_t bodyId) const;
    [[nodiscard]] size_t GetBodyCount() const noexcept
    {
        return m_bodies.size();
    }
    [[nodiscard]] JPH::PhysicsSystem &System() noexcept
    {
        return m_system;
    }
    [[nodiscard]] const JPH::PhysicsSystem &System() const noexcept
    {
        return m_system;
    }

  private:
    JPH::PhysicsSystem m_system;
    std::unordered_map<uint32_t, JPH::BodyID> m_bodies;
};
} // namespace infernux
