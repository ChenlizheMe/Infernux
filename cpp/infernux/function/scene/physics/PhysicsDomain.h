#pragma once

#include <Jolt/Jolt.h>

#include <Jolt/Physics/PhysicsSettings.h>
#include "JoltCpuContactPublication.h"
#include "JoltGpuContactPublication.h"
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
class JoltCpuScene;
class JoltCpuControl;
class JoltMixedContactStep;

/// RuntimeWorld owns its logical cursor and isolated CPU numerical state.
/// Worker services and GPU storage/submission remain shared between worlds.
/// RuntimeWorld retirement retains this object until shared-source callbacks settle.
struct PhysicsDomain final
{
    explicit PhysicsDomain(RuntimeWorld &world);
    ~PhysicsDomain();
    RuntimeWorld &owner;
    glm::vec3 gravity;
    std::unique_ptr<JoltCpuScene> cpuScene;
    std::unique_ptr<JoltCpuControl> cpuControl;
    JoltCpuContactPublication cpuContacts;
    ContactStreamIdentity cpuPublication;
    struct CpuPose
    {
        PhysicsBodyPoseUpdate pose;
        bool sleeping, fixed;
    };
    std::vector<CpuPose> cpuPoses;
    std::unordered_set<uint32_t> cpuInterpolated;
    bool cpuFixedAdmitted = false;
    bool gpuRegistered = false;
    std::vector<std::weak_ptr<PhysicsParticipant>> participants;
    std::weak_ptr<PhysicsParticipant> participantPreparing;
    bool participantOutputPending = false;
    decltype(JoltGpuTickControl::prepareParticipant) admittedParticipants;
    bool retiring = false;
    bool gpuCancelled = false;
    Ragdoll *pendingRagdollHead = nullptr, *pendingRagdollTail = nullptr;
    bool reconcilingRagdolls = false;
    std::unordered_set<uint32_t> bodyIds;
    std::unordered_set<uint32_t> pendingStaticCpu, pendingStaticGpu;
    struct StaticMove
    {
        glm::vec3 position;
        glm::quat rotation;
        std::weak_ptr<PhysicsStaticBoundaryModel> model;
        uint64_t positionRevision = 0, rotationRevision = 0;
        bool cpuQueued = false, gpuQueued = false;
    };
    std::unordered_map<uint32_t, StaticMove> pendingStaticMoves;
    std::vector<std::pair<uint32_t, StaticMove>> admittedStaticMoves;
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
    std::optional<JPH::PhysicsSettings> admittedCpuSettings;
    PhysicsContactPublication::Policy admittedContactPolicy;
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
    JoltGpuContactPublication gpuContacts;
    std::unique_ptr<JoltMixedContactStep> mixedContacts;
    struct RenderState
    {
        using Keys = std::array<uint64_t, 3>;
        struct Job
        {
            struct PoseInput
            {
                uint32_t bodyId;
                bool cpu;
                std::optional<JoltGpuRenderBindings::PhysicsPose> pose;
            };
            JoltGpuRenderBindings::SnapshotPtr snapshot;
            Keys keys{};
            uint64_t numericalGeneration = 0;
            std::optional<JoltGpuRenderBindings::Prepared> bindings;
            JoltGpuRenderPoseBatch output;
            std::shared_ptr<const JoltGpuScene::Publication> publication;
            float alpha = 1.0f;
            std::vector<PoseInput> inputs;
        };
        JoltGpuRenderBindings bindings;
        std::shared_ptr<Job> pending, ready;
        std::optional<Keys> checkedKeys;
        JoltGpuRenderBindings::SnapshotPtr mixedHierarchy;
        std::optional<Keys> hierarchyKeys;
        std::shared_ptr<Job> mixedPending;
        uint64_t mixedGeneration = 0;
        float mixedAlpha = 1.0f;
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
