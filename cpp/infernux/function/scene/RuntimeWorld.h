#pragma once

#include <core/math/FrameTimeSmoothing.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

namespace infernux
{

class Scene;

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

    struct SceneCollection
    {
        SceneCollection();
        ~SceneCollection();

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
        double fixedTime = 0.0;
        double fixedUnscaledTime = 0.0;
        uint64_t simulationEpoch = 0;
        uint64_t fixedStepIndex = 0;
        uint64_t completedFixedStepIndex = 0;
        uint64_t publishedFixedStepIndex = 0;
        FixedPhase fixedPhase = FixedPhase::Idle;
        float admittedFixedDeltaTime = 0.0f;
        bool fixedAccumulatorDebit = false;
        bool fixedHasPhysics = false;
        bool fixedExecuting = false;
        std::vector<uint64_t> fixedSceneWorlds;
        size_t fixedSceneCursor = 0;
        bool isPlaying = false;
        bool isPaused = false;
        bool isStepping = false;
        bool pausedStepPending = false;
        bool stepFrameOpen = false;
        float pausedStepDeltaTime = 0.0f;
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

    RuntimeWorld();
    ~RuntimeWorld();
    RuntimeWorld(const RuntimeWorld &) = delete;
    RuntimeWorld &operator=(const RuntimeWorld &) = delete;
    RuntimeWorld(RuntimeWorld &&) = delete;
    RuntimeWorld &operator=(RuntimeWorld &&) = delete;

    const uint64_t id;
    SimulationState simulation;
    LifecycleState lifecycle;
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
};

} // namespace infernux
