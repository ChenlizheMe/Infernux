#pragma once

#include "JoltGpuRenderBindings.h"
#include "JoltGpuRenderPoses.h"
#include "PhysicsWorld.h"

namespace infernux
{
class RuntimeWorld;

/// RuntimeWorld owns this logical cursor. All physical services remain shared.
/// RuntimeWorld retirement retains this object until shared-source callbacks settle.
struct PhysicsDomain final
{
    explicit PhysicsDomain(RuntimeWorld &world);
    RuntimeWorld &owner;
    glm::vec3 gravity;
    bool retiring = false;
    bool gpuCancelled = false;
    std::vector<PhysicsActorHandle> recreateActors;
    uint64_t surfaceRevision = 0;
    bool stepPending = false, stepCompleted = false, solverStarted = false;
    enum class FixedGpuPhase
    {
        Idle,
        RetireControl,
        RetryCreation,
        Solver,
        Failed
    };
    FixedGpuPhase fixedGpuPhase = FixedGpuPhase::Idle;
    bool waitingForSlot = false;
    bool controlPending = false, controlNumericalChange = false;
    float admittedDeltaTime = 0;
    std::optional<JoltGpuStepParameters> admittedParameters;
    uint64_t stepEpoch = 0, stepIndex = 0;
    bool flushingAuthoring = false, synchronousRead = false;
    std::atomic<uint64_t> queryGeneration{1};
    std::shared_ptr<const JoltGpuScene::Publication> lastPublication;
    struct CachedMotionState
    {
        uint64_t generation = 0;
        PhysicsBodyMotionState state;
        PhysicsStateFields fields = PhysicsStateFields::None;
    };
    std::unordered_map<uint32_t, CachedMotionState> requestedMotion;
    std::unordered_map<const JPH::Shape *, std::weak_ptr<const PhysicsShapeGeometry>> shapeGeometry;
    bool contactEventStreamEnabled = false, contactImpulseStreamEnabled = false;
    struct RenderState
    {
        using Keys = std::array<uint64_t, 3>;
        struct Job
        {
            JoltGpuRenderBindings::SnapshotPtr snapshot;
            Keys keys{};
            uint64_t numericalGeneration = 0;
            std::optional<JoltGpuRenderBindings::Prepared> bindings;
            JoltGpuRenderPoseBatch output;
            const JoltGpuScene::Publication *publication = nullptr;
        };
        JoltGpuRenderBindings bindings;
        std::shared_ptr<Job> pending, ready;
        std::optional<Keys> checkedKeys;
    };
    RenderState rendering;
};
} // namespace infernux
