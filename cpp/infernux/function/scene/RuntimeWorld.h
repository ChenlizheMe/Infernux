#pragma once

#include <algorithm>
#include <cmath>
#include <core/math/FrameTimeSmoothing.h>
#include <core/math/Random.h>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
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
enum class PhysicsBackend : uint8_t;
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

    struct TimedInputSpan
    {
        float unscaledDeltaTime = 0, deltaTime = 0;
        InputFrameLease input;
        std::optional<double> sampleOffset;
    };

    struct FrameAdmission
    {
        float unscaledDeltaTime = 0.0f;
        float deltaTime = 0.0f;
        InputFrameLease input;
        bool automaticHost = false;
        float sampledTimeScale = 1.0f;
        float sampledMaximumDeltaTime = 0.0f;
        float sampledFixedTimeStep = 0.0f;
        // Each unadmitted automatic record is one continuous input-state
        // segment. A real boundary seals its previous clock debt.
        double automaticUnscaledDebt = 0.0;
        double automaticScaledDebt = 0.0;
        bool automaticInputPending = false;
        bool coalescedHostTime = false;
        std::vector<TimedInputSpan> inputSpans;
        double automaticSampleDelay = 0;
        bool automaticSamplePending = true;

        void InitializeAutomatic(bool inputBoundary)
        {
            automaticUnscaledDebt = unscaledDeltaTime;
            automaticScaledDebt = deltaTime;
            automaticInputPending = inputBoundary;
            automaticSampleDelay = deltaTime;
        }

        [[nodiscard]] bool HasAutomaticDemand() const noexcept
        {
            return automaticUnscaledDebt > 0.0 || automaticScaledDebt > 0.0 || automaticInputPending;
        }

        // Group queued Host arrivals by the amount of Fixed work that recently
        // fit in a frame. A single original Host sample keeps its own delta.
        [[nodiscard]] double ScaledCatchUpLimit(double fixedWaitSeconds) const noexcept
        {
            if (fixedWaitSeconds <= 0.0 || sampledTimeScale <= 0.0f)
                return sampledMaximumDeltaTime;
            const double fixed = sampledFixedTimeStep;
            const double interval = (std::max)(double(unscaledDeltaTime), fixed / sampledTimeScale);
            const double steps = (std::max)(1.0, std::floor(interval / fixedWaitSeconds));
            return (std::min)(double(sampledMaximumDeltaTime), steps * fixed);
        }

        FrameAdmission TakeNextAutomatic(double unscaledLimit = std::numeric_limits<double>::infinity(),
                                         double scaledLimit = std::numeric_limits<double>::infinity())
        {
            // Keep an individual Host sample intact and distribute queued time
            // across the budgets supplied by the world.
            const double frameLimit = sampledMaximumDeltaTime;
            const double unscaledQuantum =
                (std::min)(unscaledLimit, (std::max)(static_cast<double>(unscaledDeltaTime), frameLimit));
            const double scaledQuantum =
                (std::min)(scaledLimit, (std::max)(static_cast<double>(deltaTime), frameLimit));
            double fraction = 1.0;
            if (automaticUnscaledDebt > 0.0)
                fraction = (std::min)(fraction, unscaledQuantum / automaticUnscaledDebt);
            if (automaticScaledDebt > 0.0)
                fraction = (std::min)(fraction, scaledQuantum / automaticScaledDebt);
            // Subtract the exact published float duration, keeping any smaller
            // residual in double debt. Never throw away accumulated Host time.
            const auto take = [fraction](double &debt) {
                float amount = static_cast<float>(debt * fraction);
                if (static_cast<double>(amount) > debt)
                    amount = std::nextafter(amount, 0.0f);
                debt -= static_cast<double>(amount);
                return amount;
            };
            FrameAdmission admitted;
            admitted.unscaledDeltaTime = take(automaticUnscaledDebt);
            admitted.deltaTime = take(automaticScaledDebt);
            admitted.input = input;
            admitted.automaticHost = true;
            admitted.sampledTimeScale = sampledTimeScale;
            admitted.sampledMaximumDeltaTime = sampledMaximumDeltaTime;
            admitted.sampledFixedTimeStep = sampledFixedTimeStep;
            admitted.automaticUnscaledDebt = admitted.unscaledDeltaTime;
            admitted.automaticScaledDebt = admitted.deltaTime;
            std::optional<double> sampleOffset;
            if (automaticSamplePending && automaticSampleDelay <= admitted.deltaTime) {
                sampleOffset = automaticSampleDelay;
                automaticSamplePending = false;
            } else if (automaticSamplePending) {
                automaticSampleDelay -= admitted.deltaTime;
            }
            admitted.inputSpans.push_back({admitted.unscaledDeltaTime, admitted.deltaTime, input, sampleOffset});
            automaticInputPending = false;
            return admitted;
        }

        bool TryAppendAutomatic(FrameAdmission &next)
        {
            if (!automaticHost || !next.automaticHost || sampledTimeScale != next.sampledTimeScale ||
                sampledMaximumDeltaTime != next.sampledMaximumDeltaTime ||
                sampledFixedTimeStep != next.sampledFixedTimeStep ||
                (input->sampleIdentity != next.input->sampleIdentity &&
                 !InputManager::CanCoalesceFrames(*input, *next.input)))
                return false;
            automaticUnscaledDebt += next.automaticUnscaledDebt;
            automaticScaledDebt += next.automaticScaledDebt;
            coalescedHostTime = true;
            // Preserve a single large arrival. Coalesced debt is admitted in
            // quanta bounded by the existing maximum frame delta or a larger
            // real single arrival, never as one giant Fixed catch-up frame.
            unscaledDeltaTime = (std::max)(unscaledDeltaTime, next.unscaledDeltaTime);
            deltaTime = (std::max)(deltaTime, next.deltaTime);
            automaticInputPending = automaticInputPending || next.automaticInputPending;
            // Keep the first equivalent source and its original arrival time.
            // Empty observations do not create later occurrences of its edges.
            return true;
        }
    };

    struct FrameState
    {
        std::deque<FrameAdmission> pending;
        InputFrameLease input;
        FramePhase phase = FramePhase::Idle;
        uint64_t generation = 0;
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

    // Explicit authored edits own a display request, independently of logical F.
    struct PausedAuthorPresentation
    {
        uint64_t requested = 0;
        uint64_t published = 0;
        [[nodiscard]] bool Pending() const noexcept
        {
            return requested != published;
        }
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
    [[nodiscard]] PhysicsBackend GetPhysicsBackend() const noexcept;
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
    PausedAuthorPresentation pausedAuthor;
    RenderMembership render;
    SceneCollection scenes;

    struct InputTimeline
    {
        struct Point
        {
            double time;
            InputFrameLease input;
            bool deliverEvents = true;
        };
        std::deque<Point> points;
        double acceptedTime = 0, fixedTime = 0;
        double currentFixedWaitSeconds = 0, recentFixedWaitSeconds = 0;
        uint64_t lastInitialSource = 0, lastUpdateSource = 0, lastFixedSource = 0;
        InputFrameLease held, latest, fixedInput, updateInput;
        bool zeroScaledFrame = false;
        bool automaticFrame = false;
    };
    std::unique_ptr<InputTimeline> inputTimeline;
};

/// A NoView UI query may suppress implicit authoring/primary selection.
/// Explicit execution scopes remain authoritative, including management work.
class WorldQueryScope final
{
  public:
    explicit WorldQueryScope(bool allowDefaultWorld) noexcept;
    ~WorldQueryScope();
    WorldQueryScope(const WorldQueryScope &) = delete;
    WorldQueryScope &operator=(const WorldQueryScope &) = delete;

    [[nodiscard]] static bool AllowsDefaultWorld() noexcept;

  private:
    bool m_previous;
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
