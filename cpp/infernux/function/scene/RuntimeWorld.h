#pragma once

#include <core/math/FrameTimeSmoothing.h>
#include <core/math/Random.h>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <platform/input/InputManager.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace infernux
{

class Scene;
class MeshRenderer;
class SkinnedMeshRenderer;
class Transform;
class SkeletonInstance;
class Light;
struct PhysicsDomain;

/// Stable management identity. Survives graph/domain retirement while scripts retain it.
/// Reserved handles contain no Scene, PhysicsDomain, GPU allocation or Python graph.
struct RuntimeWorldHandle final
{
    const uint64_t id;
    std::string state = "preparing";
    std::string error;
    std::string retirementOutcome = "destroyed";
    bool displayInGame = true;
    bool pendingPrimary = false;
    bool destroyRequested = false;
    uint64_t managementSession = 0;
    explicit RuntimeWorldHandle(uint64_t identity) : id(identity)
    {
    }
};

/// Scene ownership and simulation state; device and submission services stay shared.
class RuntimeWorld final
{
  public:
    struct FrameTime
    {
        double time = 0.0;
        double unscaledTime = 0.0;
        float deltaTime = 0.0f;
        float unscaledDeltaTime = 0.0f;
        float smoothDeltaTime = 0.0f;
        uint64_t frameCount = 0;
    };

    enum class FixedPhase
    {
        Idle,
        SceneScripts,
        Control,
        Integration,
        Failed,
        Publication,
        PostNotification,
        PostStep
    };

    enum class FramePhase
    {
        Idle,
        Input,
        Fixed,
        Update,
        Late,
        Ready,
        Complete,
        Publish
    };

    struct FrameAdmission
    {
        float unscaledDeltaTime = 0.0f;
        float deltaTime = 0.0f;
        InputFrameLease input;
    };

    struct FrameState
    {
        std::deque<FrameAdmission> pending;
        InputFrameLease input;
        FramePhase phase = FramePhase::Idle;
        uint64_t generation = 0;
        uint64_t fixedTarget = 0;
        std::vector<uint64_t> phaseScenes;
        size_t sceneCursor = 0;
        bool updateStarted = false;
        bool animationStarted = false;
        bool lateStarted = false;
        bool hadRuntimeLifecycleScope = false;
        uint64_t stepLogicalFrameTarget = 0;
        InputFrameLease stepInput;
    };

    struct SceneCollection
    {
        SceneCollection();
        ~SceneCollection();
        void Clear();

        std::vector<std::unique_ptr<Scene>> loaded;
        std::unordered_set<Scene *> loadedSet;
        Scene *active = nullptr;
        std::unique_ptr<Scene> persistent;
        std::vector<uint64_t> pendingPersistentRootIds;
        std::unordered_set<uint64_t> pendingPersistentRootIdSet;
    };

    struct SimulationState
    {
        float fixedTimeStep = 1.0f / 50.0f; // 50 Hz default (Unity default)
        float fixedTimeAccumulator = 0.0f;
        float maxFixedDeltaTime = 1.0f / 3.0f; // Unity Maximum Allowed Timestep default
        float timeScale = 1.0f;
        FrameTime frameTime;
        timing::FrameTimeSmoother frameTimeSmoother;
        bool resetDeltaTimeOnNextFrame = true;
        uint64_t deltaResetHostOpportunity = 0;
        double fixedTime = 0.0;
        double fixedUnscaledTime = 0.0;
        uint64_t simulationEpoch = 0;
        uint64_t fixedStepIndex = 0;
        uint64_t completedFixedStepIndex = 0;
        uint64_t publishedFixedStepIndex = 0;
        FixedPhase fixedPhase = FixedPhase::Idle;
        float admittedFixedDeltaTime = 0.0f;
        bool fixedHasPhysics = false;
        bool fixedExecuting = false;
        std::vector<uint64_t> fixedSceneWorlds;
        size_t fixedSceneCursor = 0;
        bool isPlaying = false;
        bool isPaused = false;
        bool isStepping = false;
        bool pausedStepPending = false;
        bool stepFrameOpen = false;
        float pausedStepUnscaledDeltaTime = 0.0f;
        bool startupPending = false;
        uint64_t runtimeFrameCount = 0;
    };

    struct LifecycleState
    {
        bool runtimeLifecycleWorkAvailable = false;
        bool runtimeResultConsumersPresent = false;
        bool runtimeLifecycleFrameOpen = false;
        uint64_t runtimeLifecyclePlanRevision = 0;
        size_t runtimeLifecycleFixedUpdateCount = 0;
        size_t runtimeLifecycleUpdateCount = 0;
        size_t runtimeLifecycleAnimationUpdateCount = 0;
        size_t runtimeLifecycleLateUpdateCount = 0;
    };

    struct RenderMembership
    {
        uint64_t renderTransformRevision = 1;
        uint64_t renderContentRevision = 1;

        // MeshRenderer component registry — populated by MeshRenderer OnEnable/OnDisable.
        // Avoids per-frame GetAllObjects() + dynamic_cast in CollectRenderables.
        std::vector<MeshRenderer *> activeMeshRenderers;
        std::unordered_set<MeshRenderer *> activeMeshRendererSet; // O(1) duplicate check
        std::unordered_map<MeshRenderer *, SkinnedMeshRenderer *> pendingSkinPoseHistoryCommits;
        std::unordered_map<Transform *, SkeletonInstance *> skeletonBoneOwners;
        std::vector<SkeletonInstance *> pendingSkeletonPoses;
        uint64_t meshRendererVersion = 0;
        uint32_t rendererRegistryTransactionDepth = 0;
        bool rendererRegistryTransactionDirty = false;

        // Light component registry — populated by Light OnEnable/OnDisable.
        // Avoids per-frame GetAllObjects() + GetComponent<Light>() in CollectLights/ComputeShadowVP.
        std::vector<Light *> activeLights;
    };

    static std::shared_ptr<RuntimeWorldHandle> ReserveHandle();
    explicit RuntimeWorld(std::shared_ptr<RuntimeWorldHandle> handle = nullptr);
    ~RuntimeWorld();
    RuntimeWorld(const RuntimeWorld &) = delete;
    RuntimeWorld &operator=(const RuntimeWorld &) = delete;
    RuntimeWorld(RuntimeWorld &&) = delete;
    RuntimeWorld &operator=(RuntimeWorld &&) = delete;

    const std::shared_ptr<RuntimeWorldHandle> handle;
    const uint64_t id;
    std::shared_ptr<Random::Storage> random = std::make_shared<Random::Storage>();
    size_t executionScopes = 0;
    bool retirementScenesComplete = false;
    std::unique_ptr<PhysicsDomain> physics;
    SimulationState simulation;
    LifecycleState lifecycle;
    FrameState frame;
    RenderMembership render;
    SceneCollection scenes;
};

/// Restore the caller's World after callbacks, including exceptional returns.
class WorldExecutionScope final
{
  public:
    explicit WorldExecutionScope(RuntimeWorld *world) noexcept;
    ~WorldExecutionScope();
    WorldExecutionScope(const WorldExecutionScope &) = delete;
    WorldExecutionScope &operator=(const WorldExecutionScope &) = delete;

    [[nodiscard]] static RuntimeWorld *Current() noexcept;

  private:
    RuntimeWorld *m_previous;
    RuntimeWorld *m_world;
    std::optional<InputReadScope> m_inputScope;
    std::optional<Random::StorageScope> m_randomScope;
};

} // namespace infernux
