#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Core/JobSystemSingleThreaded.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/Collision/ContactListener.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include "InfernuxJoltJobSystemAdapter.h"
#include "JoltCpuActor.h"
#include "PhysicsContactTypes.h"

#include <cstdint>
#include <unordered_map>

namespace JPH
{
class CompoundShape;
}

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
/// backend. The shared layer filters outlive the scene; contact policy is owned here.
class JoltCpuScene final : private JPH::ContactListener
{
  public:
    JoltCpuScene(const EngineConfig &config, const JPH::BroadPhaseLayerInterface &layers,
                 const JPH::ObjectVsBroadPhaseLayerFilter &broadphaseFilter,
                 const JPH::ObjectLayerPairFilter &pairFilter);
    ~JoltCpuScene() override;

    /// Creation does not enter broadphase. The common actor lifecycle publishes
    /// the body through Jolt's normal single/batch add operations.
    JPH::BodyID CreateBody(uint32_t bodyId, const PreparedCpuActor &actor);
    /// Caller has detached this body's constraints before destruction.
    void DestroyBody(uint32_t bodyId);
    [[nodiscard]] JPH::BodyID NativeBody(uint32_t bodyId) const;
    /// Immutable, sorted canonical pairs from the common world policy.
    void SetIgnoredPairs(std::shared_ptr<const PhysicsColliderPairs> pairs);

    // Captured author values are immutable throughout Update. Jolt workers do
    // not call live Components or resolve their targets while solving contacts.
    using Surface = PreparedCpuActor::Surface;
    [[nodiscard]] const Surface &GetSurface(uint32_t bodyId, JPH::SubShapeID subshape) const;
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
    struct BodyRecord
    {
        JPH::BodyID native;
        JPH::RefConst<JPH::Shape> shape;
        const JPH::CompoundShape *memberCompound = nullptr;
        std::vector<Surface> surfaces;
        int collisionDetectionMode = 0;
    };
    JPH::BodyID CreateBody(uint32_t bodyId, const JPH::BodyCreationSettings &settings, BodyRecord record);
    JPH::ValidateResult OnContactValidate(const JPH::Body &a, const JPH::Body &b, JPH::RVec3Arg,
                                          const JPH::CollideShapeResult &hit) override;
    JPH::ValidateResult OnCCDContactValidate(const JPH::Body &a, const JPH::Body &b, JPH::RVec3Arg origin,
                                             const JPH::CollideShapeResult &hit) override;
    void OnContactAdded(const JPH::Body &a, const JPH::Body &b, const JPH::ContactManifold &manifold,
                        JPH::ContactSettings &settings) override;
    void OnContactPersisted(const JPH::Body &a, const JPH::Body &b, const JPH::ContactManifold &manifold,
                            JPH::ContactSettings &settings) override;
    JPH::PhysicsSystem m_system;
    std::unordered_map<uint32_t, BodyRecord> m_bodies;
    std::shared_ptr<const PhysicsColliderPairs> m_ignoredPairs;
};
} // namespace infernux
