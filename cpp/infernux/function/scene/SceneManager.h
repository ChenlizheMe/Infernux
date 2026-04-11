#pragma once

#include "EditorCameraController.h"
#include "Scene.h"
#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace infernux
{

// Forward declaration for MeshRenderer registry
class MeshRenderer;
// Forward declaration for Light registry
class Light;

/**
 * @brief SceneManager - singleton that manages all scenes.
 *
 * Handles scene loading, switching, and provides access to the active scene.
 * In editor mode, it also manages the editor scene camera.
 */
class SceneManager
{
  public:
    struct FrameProfile
    {
        double editorCameraMs = 0.0;
        double editorUpdateMs = 0.0;
        double pendingStartsMs = 0.0;
        double syncExternalMovesMs = 0.0;
        double syncCollidersMs = 0.0;
        double fixedUpdateMs = 0.0;
        double physicsStepMs = 0.0;
        double physicsEventsMs = 0.0;
        double syncRigidbodiesMs = 0.0;
        double interpolationMs = 0.0;
        double gameplayUpdateMs = 0.0;
        double lateUpdateMs = 0.0;
        double audioMs = 0.0;
        double endFrameMs = 0.0;
        double fixedSteps = 0.0;
    };

    // Singleton access
    static SceneManager &Instance();

    // Prevent copying
    SceneManager(const SceneManager &) = delete;
    SceneManager &operator=(const SceneManager &) = delete;

    // ========================================================================
    // Scene management
    // ========================================================================

    /// @brief Create a new empty scene
    Scene *CreateScene(const std::string &name);

    /// @brief Set the active scene
    void SetActiveScene(Scene *scene);

    /// @brief Get the currently active scene
    [[nodiscard]] Scene *GetActiveScene() const
    {
        return m_activeScene;
    }

    /// @brief Unload a scene
    void UnloadScene(Scene *scene);

    /// @brief Unload all scenes
    void UnloadAllScenes();

    /// @brief Get a scene by name
    [[nodiscard]] Scene *GetScene(const std::string &name) const;

    /// @brief Get all loaded scenes
    [[nodiscard]] const std::vector<std::unique_ptr<Scene>> &GetAllScenes() const
    {
        return m_scenes;
    }

    // ========================================================================
    // Frame update
    // ========================================================================

    /// @brief Call at the start of the game (after first scene loads)
    void Start();

    /// @brief Call every frame
    void Update(float deltaTime);

    /// @brief Called at a fixed time step (physics / deterministic logic)
    void FixedUpdate();

    /// @brief Call every frame after Update
    void LateUpdate(float deltaTime);

    /// @brief Process pending destroys at end of frame
    void EndFrame();

    /// @brief Manually flush Transform changes to the physics engine.
    /// Unity equivalent: Physics.SyncTransforms().
    /// Useful when script code modifies transforms in Update and needs
    /// immediate physics queries (raycast, overlap) against the new positions.
    /// Normally called automatically before each physics step; calling it
    /// explicitly is only needed for same-frame queries after transform edits.
    void SyncTransforms();

    [[nodiscard]] const FrameProfile &GetLastFrameProfile() const
    {
        return m_lastFrameProfile;
    }

    // ========================================================================
    // DontDestroyOnLoad
    // ========================================================================

    /// @brief Mark a root GameObject so it survives scene switches.
    /// The object is moved to an internal persistent list owned by SceneManager.
    /// Unity: Object.DontDestroyOnLoad(gameObject)
    void DontDestroyOnLoad(GameObject *gameObject);

    /// @brief Get all persistent (DontDestroyOnLoad) objects
    [[nodiscard]] const std::vector<std::unique_ptr<GameObject>> &GetPersistentObjects() const
    {
        return m_persistentObjects;
    }

    // ========================================================================
    // Editor support
    // ========================================================================

    /// @brief Get the editor camera controller
    [[nodiscard]] EditorCameraController &GetEditorCameraController()
    {
        return m_editorCamera;
    }

    /// @brief Is the scene in play mode?
    [[nodiscard]] bool IsPlaying() const
    {
        return m_isPlaying;
    }

    /// @brief Enter play mode
    void Play();

    /// @brief Stop play mode
    void Stop();

    /// @brief Pause play mode
    void Pause();

    /// @brief Step exactly one frame while paused (Update + LateUpdate + EndFrame).
    /// Does nothing if not currently paused and playing.
    void Step(float deltaTime);

    [[nodiscard]] bool IsPaused() const
    {
        return m_isPaused;
    }

    /// @brief Get the fixed physics timestep in seconds.
    [[nodiscard]] float GetFixedTimeStep() const
    {
        return m_fixedTimeStep;
    }

    /// @brief Set the fixed physics timestep in seconds.
    void SetFixedTimeStep(float value)
    {
        m_fixedTimeStep = std::max(0.001f, value);
        m_maxFixedDeltaTime = std::max(m_maxFixedDeltaTime, m_fixedTimeStep);
    }

    /// @brief Get the max clamped frame delta used by the fixed-step accumulator.
    [[nodiscard]] float GetMaxFixedDeltaTime() const
    {
        return m_maxFixedDeltaTime;
    }

    /// @brief Set the max clamped frame delta used by the fixed-step accumulator.
    void SetMaxFixedDeltaTime(float value)
    {
        m_maxFixedDeltaTime = std::max(m_fixedTimeStep, value);
    }

    // ========================================================================
    // Callbacks
    // ========================================================================

    using SceneCallback = std::function<void(Scene *)>;

    void OnSceneLoaded(SceneCallback callback)
    {
        m_onSceneLoaded = callback;
    }
    void OnSceneUnloaded(SceneCallback callback)
    {
        m_onSceneUnloaded = callback;
    }

    // ========================================================================
    // Component registries
    // ========================================================================

    /// Clear MeshRenderer registry (called on scene unload / deserialize).
    void ClearComponentRegistries();

    /// Pre-allocate MeshRenderer registry storage for bulk creation.
    void ReserveRendererCapacity(size_t count);

    /// Register a MeshRenderer so rendering can iterate it directly.
    void RegisterMeshRenderer(MeshRenderer *renderer);

    /// Unregister a MeshRenderer (e.g. OnDisable / destruction).
    void UnregisterMeshRenderer(MeshRenderer *renderer);

    /// Bump the renderable cache version after a registered MeshRenderer
    /// changes mesh/material state without leaving the registry.
    void NotifyMeshRendererChanged(MeshRenderer *renderer);

    /// Read-only access to the active mesh renderers registry.
    [[nodiscard]] const std::vector<MeshRenderer *> &GetActiveMeshRenderers() const
    {
        return m_activeMeshRenderers;
    }

    /// Monotonic counter bumped when a MeshRenderer is registered/unregistered.
    [[nodiscard]] uint64_t GetMeshRendererVersion() const
    {
        return m_meshRendererVersion;
    }

    /// Mark all MeshRenderers referencing a given mesh GUID as buffer-dirty.
    void MarkMeshRenderersDirtyForAsset(const std::string &meshGuid);

    /// Register a Light so lighting can iterate it directly.
    void RegisterLight(Light *light);

    /// Unregister a Light (e.g. OnDisable / destruction).
    void UnregisterLight(Light *light);

    /// Read-only access to the active lights registry.
    [[nodiscard]] const std::vector<Light *> &GetActiveLights() const
    {
        return m_activeLights;
    }

  private:
    SceneManager();
    ~SceneManager() = default;

    /// Walk all colliders in the active scene and sync transforms to Jolt.
    /// Uses a global transform serial to skip entirely when no transforms changed.
    void SyncCollidersToPhysics();

    /// Flush pending broadphase additions (batched from Collider::AddToBroadphase).
    /// Also rebuilds the BVH tree when new bodies were added.
    void FlushPendingBroadphase();

    /// Force-sync ALL collider body positions to their current Transform,
    /// including dynamic bodies (which SyncCollidersToPhysics normally skips).
    /// Called once at the start of play to fix stale editor-mode positions.
    void ForceAllBodiesToCurrentTransform();

    /// Walk all Rigidbodies and write Jolt position/rotation back to Transform.
    void SyncRigidbodiesToTransform();

    /// Apply presentation interpolation for dynamic rigidbodies.
    void ApplyInterpolatedRigidbodies(float alpha);

    /// Detect user-driven Transform changes on dynamic Rigidbodies and teleport
    /// their Jolt bodies before the physics step.
    void SyncExternalRigidbodyMoves();

    /// Detach persistent (DontDestroyOnLoad) root objects from a scene
    /// into m_persistentObjects, keeping them alive across scene switches.
    void ExtractPersistentObjects(Scene *scene);

    std::vector<std::unique_ptr<Scene>> m_scenes;
    Scene *m_activeScene = nullptr;

    // Editor camera (exists even when no scene is loaded)
    std::unique_ptr<GameObject> m_editorCameraObject;
    Camera *m_editorCameraComponent = nullptr;
    EditorCameraController m_editorCamera;

    // Persistent objects (DontDestroyOnLoad)
    std::vector<std::unique_ptr<GameObject>> m_persistentObjects;

    // Fixed-update timing
    float m_fixedTimeStep = 1.0f / 50.0f; // 50 Hz default (Unity default)
    float m_fixedTimeAccumulator = 0.0f;
    float m_maxFixedDeltaTime = 0.1f; // cap to avoid spiral-of-death

    // Play mode state
    bool m_isPlaying = false;
    bool m_isPaused = false;

    // Callbacks
    SceneCallback m_onSceneLoaded;
    SceneCallback m_onSceneUnloaded;

    // MeshRenderer component registry — populated by MeshRenderer OnEnable/OnDisable.
    // Avoids per-frame GetAllObjects() + dynamic_cast in CollectRenderables.
    std::vector<MeshRenderer *> m_activeMeshRenderers;
    std::unordered_set<MeshRenderer *> m_activeMeshRendererSet; // O(1) duplicate check
    uint64_t m_meshRendererVersion = 0;

    // Light component registry — populated by Light OnEnable/OnDisable.
    // Avoids per-frame GetAllObjects() + GetComponent<Light>() in CollectLights/ComputeShadowVP.
    std::vector<Light *> m_activeLights;

    // ── Physics sync state (Unity-style deferred transform sync) ─────
    /// Cached TransformECSStore global serial at last SyncCollidersToPhysics.
    /// When the store serial hasn't changed, the entire sync is skipped.
    uint64_t m_lastPhysicsSyncTransformSerial = 0;

    FrameProfile m_lastFrameProfile;
};

} // namespace infernux
