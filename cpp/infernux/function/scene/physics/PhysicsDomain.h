#pragma once

#include "JoltGpuRenderBindings.h"
#include "JoltGpuRenderPoses.h"
#include "PhysicsContactPublication.h"
#include "PhysicsParticipant.h"
#include "PhysicsWorld.h"
#include <map>
#include <set>

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
    std::vector<std::weak_ptr<PhysicsParticipant>> participants;
    std::weak_ptr<PhysicsParticipant> participantPreparing;
    bool participantOutputPending = false;
    decltype(JoltGpuTickControl::prepareParticipant) admittedParticipants;
    bool retiring = false;
    bool gpuCancelled = false;
    Ragdoll *pendingRagdollHead = nullptr, *pendingRagdollTail = nullptr;
    bool reconcilingRagdolls = false;
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
    struct ColliderPairPolicy
    {
        bool user = false;
        std::set<std::array<uint64_t, 2>> owners;
    };
    std::map<PhysicsColliderPair, ColliderPairPolicy> colliderPairs;
    std::shared_ptr<const PhysicsColliderPairs> colliderPairSnapshot, queuedColliderPairs;
    PhysicsColliderPairs initialColliderPairs;
    bool contactEventStreamEnabled = false, contactImpulseStreamEnabled = false, contactIncludeTriggers = true;
    PhysicsContactPublication contacts;
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
    struct PresentationPoseReads
    {
        std::shared_ptr<const JoltGpuScene::Publication> publication;
        uint64_t generation = 0;
        PhysicsRigidbodyStateBatch batch;
        std::shared_ptr<JoltGpuQueryResult> motion, shapes;
        std::exception_ptr error;
    };
    std::shared_ptr<PresentationPoseReads> presentationPoseReads;
    uint32_t presentationPoseReadDepth = 0;
    uint64_t presentationPoseReadGeneration = 0;
};
} // namespace infernux
