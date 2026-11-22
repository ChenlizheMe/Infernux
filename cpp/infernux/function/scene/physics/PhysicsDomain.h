#pragma once

#include <Jolt/Jolt.h>

#include <Jolt/Physics/PhysicsSettings.h>
#include "../Scene.h"
#include "JoltCpuContactPublication.h"
#include "PhysicsBodyCommand.h"
#include "PhysicsContactPublication.h"
#include "PhysicsFixedInputs.h"
#include "PhysicsGpuRenderBindings.h"
#include "PhysicsGpuRenderPoses.h"
#include "PhysicsParticipant.h"
#include "PhysicsWorld.h"
#include "inxphygpu/InxPhyGpuSceneModel.h"
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
struct PhysicsGpuPublicationRequest;

struct PhysicsSceneJointAuthor
{
    uint32_t ownerBody, connectedBody;
    glm::vec3 ownerLocalAnchor;
    bool enableCollision;
    physics::inxphygpu::InxPhyGpuSceneJointSettings settings;
    PhysicsConstraintOwner *owner = nullptr;
    std::optional<glm::quat> poseTarget{};
    uint8_t pendingEdits = 0;
};

enum class PhysicsSceneJointEdit : uint8_t
{
    Limits = 1,
    Drive = 2,
    Pose = 4,
};

struct PhysicsGpuRenderJob
{
    PhysicsGpuRenderBindings::SnapshotPtr snapshot;
    std::array<uint64_t, 3> keys{};
    uint64_t numericalGeneration = 0;
    std::optional<PhysicsGpuRenderBindings::Export> output;
    std::shared_ptr<const void> receipt;
};

/// RuntimeWorld owns its logical cursor and isolated CPU numerical state.
/// Worker services and GPU storage/submission remain shared between worlds.
/// RuntimeWorld retirement retains this object until shared-source callbacks settle.
struct PhysicsDomain final
{
    explicit PhysicsDomain(RuntimeWorld &world);
    ~PhysicsDomain();
    void MarkGpuPoseDiscontinuity(uint32_t bodyId);
    void RetireGpuJoint(uint64_t id) noexcept;
    void MarkGpuJointEdit(uint64_t id, PhysicsSceneJointEdit edit);
    void SetGpuJointPose(uint64_t id, std::optional<glm::quat> relative);
    void SetGpuJointCollision(PhysicsSceneJointAuthor &joint, bool enabled) noexcept;
    [[nodiscard]] std::vector<physics::inxphygpu::InxPhyGpuSceneJointUpdate> CaptureGpuJointEdits() const;
    void AcceptGpuJointEdits() noexcept;
    void BeginInxPhyGpuFixed(ContactStreamIdentity, PhysicsFixedInputs input);
    [[nodiscard]] bool PublishInxPhyGpuFixed();
    void RetireInxPhyGpuFixed() noexcept;
    void RequestInxPhyGpuAuthorPresentation() const;
    void MarkInxPhyGpuTopology() noexcept;
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
    std::map<uint64_t, PhysicsSceneJointAuthor> gpuJointAuthors;
    std::vector<uint64_t> gpuJointEdits;
    bool gpuJointLayoutDirty = false;
    // Shared by the queued authored moves and their retained component guards.
    std::shared_ptr<PhysicsTargetBatch> gpuTargetAuthors;
    std::vector<std::weak_ptr<PhysicsParticipant>> participants;
    std::shared_ptr<PhysicsGpuPublicationRequest> participantInitialization;
    std::shared_ptr<PhysicsGpuPublicationRequest> participantOutput;
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
        Solver,
        Failed
    };
    FixedGpuPhase fixedGpuPhase = FixedGpuPhase::Idle;
    float admittedDeltaTime = 0;
    uint32_t admittedSubsteps = 1;
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
    struct CachedMotionState
    {
        uint64_t generation = 0;
        PhysicsBodyMotionState state;
        PhysicsStateFields fields = PhysicsStateFields::None;
    };
    std::unordered_map<uint32_t, CachedMotionState> requestedMotion;
    struct ColliderPairPolicy
    {
        bool user = false;
        std::set<std::array<uint64_t, 2>> owners;
    };
    std::map<PhysicsColliderPair, ColliderPairPolicy> colliderPairs;
    std::shared_ptr<const PhysicsColliderPairs> colliderPairSnapshot;
    PhysicsColliderPairs initialColliderPairs;
    bool contactEventStreamEnabled = false, contactImpulseStreamEnabled = false, contactIncludeTriggers = true;
    PhysicsContactPublication contacts;
    struct InxPhyGpuFixedAdmission
    {
        ContactStreamIdentity identity;
        PhysicsContactPublication::Policy contacts;
        std::shared_ptr<PhysicsGpuRenderJob> renderJob;
    };
    std::optional<InxPhyGpuFixedAdmission> nativeFixedAdmission;
    std::shared_ptr<const physics::inxphygpu::InxPhyGpuFixedPublication> nativeFixedPublication;
    std::unique_ptr<physics::inxphygpu::InxPhyGpuSceneRuntime> nativeRuntime;
    std::unique_ptr<physics::inxphygpu::InxPhyGpuSceneRuntime> nativeTopology;
    uint64_t nativeLayoutEpoch = 0;
    std::vector<std::pair<uint32_t, uint64_t>> nativeTopologyActors;
    bool nativeBodyLayoutDirty = false;
    struct InxPhyGpuEdit
    {
        std::shared_ptr<PhysicsGpuRenderJob> renderJob;
        bool numericalChange;
    };
    std::optional<InxPhyGpuEdit> nativeEdit;
    struct RenderState
    {
        using Keys = std::array<uint64_t, 3>;
        PhysicsGpuRenderBindings bindings;
        std::shared_ptr<PhysicsGpuRenderJob> ready;
        std::optional<Keys> checkedKeys;
    };
    RenderState rendering;
    struct PresentationPoseReads
    {
        std::shared_ptr<const physics::inxphygpu::InxPhyGpuFixedPublication> nativePublication;
        uint64_t generation = 0;
        PhysicsGpuMotionRead read;
        std::shared_ptr<PhysicsGpuPublicationRequest> request;
    };
    std::shared_ptr<PresentationPoseReads> presentationPoseReads;
    uint32_t presentationPoseReadDepth = 0;
    uint64_t presentationPoseReadGeneration = 0;
};
} // namespace infernux
