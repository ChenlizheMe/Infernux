#pragma once

#include <Jolt/Jolt.h>

#include <Jolt/Physics/PhysicsSettings.h>
#include "../Scene.h"
#include "JoltCpuContactPublication.h"
#include "JoltGpuContactPublication.h"
#include "PhysicsBodyCommand.h"
#include "PhysicsContactPublication.h"
#include "PhysicsFixedInputs.h"
#include "PhysicsGpuRenderBindings.h"
#include "PhysicsGpuRenderPoses.h"
#include "PhysicsParticipant.h"
#include "PhysicsWorld.h"
#include <map>
#include <mutex>
#include <set>
#include <shared_mutex>

namespace infernux
{
class RuntimeWorld;
class JoltCpuScene;
class JoltCpuControl;
class PhysicsConstraintOwner;
struct PhysicsSceneRayPlan;
struct PhysicsGpuPublicationRequest;

/// RuntimeWorld owns its logical cursor and isolated CPU numerical state.
/// Worker services and GPU storage/submission remain shared between worlds.
/// RuntimeWorld retirement retains this object until shared-source callbacks settle.
struct PhysicsDomain final
{
    explicit PhysicsDomain(RuntimeWorld &world);
    ~PhysicsDomain();
    void MarkGpuPoseDiscontinuity(uint32_t bodyId);
    RuntimeWorld &owner;
    PhysicsBackend backend = PhysicsBackend::CPU;
    [[nodiscard]] bool UsesGpu() const noexcept;
    glm::vec3 gravity;
    std::shared_ptr<JoltCpuScene> cpuScene;
    std::unique_ptr<JoltCpuControl> cpuControl;
    std::optional<std::unique_lock<std::shared_mutex>> cpuFixedQueryWrite;
    JoltCpuContactPublication cpuContacts;
    ContactStreamIdentity cpuPublication;
    struct CpuPose
    {
        PhysicsBodyPoseUpdate pose;
        bool sleeping, fixed;
    };
    std::vector<CpuPose> cpuPoses;
    std::unordered_set<uint32_t> cpuInterpolated;
    std::optional<float> publishedCpuInterpolationAlpha;
    bool cpuFixedAdmitted = false;
    std::vector<PhysicsSceneBodyCommand> gpuBodyCommands;
    // Shared by the queued authored moves and their retained component guards.
    std::shared_ptr<PhysicsTargetBatch> gpuTargetAuthors;
    bool gpuRegistered = false;
    size_t gpuBodyOwners = 0;
    struct StartupPrograms
    {
        bool gpuBodies = false;
        bool gpuContacts = false;
    };
    std::optional<StartupPrograms> startupPrograms;
    std::vector<std::weak_ptr<PhysicsParticipant>> participants;
    std::shared_ptr<PhysicsGpuPublicationRequest> participantInitialization;
    std::shared_ptr<PhysicsGpuPublicationRequest> participantOutput;
    JoltGpuParticipantStages admittedParticipants;
    bool retiring = false;
    Ragdoll *pendingRagdollHead = nullptr, *pendingRagdollTail = nullptr;
    bool reconcilingRagdolls = false;
    std::unordered_set<uint32_t> bodyIds;
    std::unordered_set<uint32_t> pendingStatic;
    struct StaticMove
    {
        glm::vec3 position;
        glm::quat rotation;
        std::weak_ptr<PhysicsStaticBoundaryModel> model;
        uint64_t positionRevision = 0, rotationRevision = 0;
        bool queued = false;
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
    uint32_t admittedSubsteps = 1;
    std::optional<JoltGpuStepParameters> admittedParameters;
    JPH::PhysicsSettings admittedCpuSettings;
    JPH::Vec3 admittedGravity = JPH::Vec3::sZero();
    [[nodiscard]] PhysicsFixedInputs AdmittedFixedInputs() const
    {
        return {admittedDeltaTime, {admittedGravity.GetX(), admittedGravity.GetY(), admittedGravity.GetZ()}};
    }
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
    struct RenderState
    {
        using Keys = std::array<uint64_t, 3>;
        struct Job
        {
            PhysicsGpuRenderBindings::SnapshotPtr snapshot;
            Keys keys{};
            uint64_t numericalGeneration = 0;
            std::optional<PhysicsGpuRenderBindings::Export> output;
            // Compare the accepted receipt by identity, including control
            // publications that share the same fixed index or device buffer.
            std::shared_ptr<const void> receipt;
        };
        PhysicsGpuRenderBindings bindings;
        std::shared_ptr<Job> pending, ready;
        std::optional<Keys> checkedKeys;
    };
    RenderState rendering;
    struct PresentationPoseReads
    {
        std::shared_ptr<const JoltGpuScene::Publication> publication;
        uint64_t generation = 0;
        PhysicsGpuMotionRead read;
        std::shared_ptr<PhysicsGpuPublicationRequest> request;
    };
    std::shared_ptr<PresentationPoseReads> presentationPoseReads;
    uint32_t presentationPoseReadDepth = 0;
    uint64_t presentationPoseReadGeneration = 0;
    // A collision graph can consume static GPU boundaries without a Rigidbody.
    std::weak_ptr<const void> particleCollisionConsumer;
    std::unique_ptr<PhysicsSceneRayPlan> sceneRays;
};
} // namespace infernux
