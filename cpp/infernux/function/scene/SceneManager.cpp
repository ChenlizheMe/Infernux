// Jolt types hidden behind opaque headers — no Jolt include needed
#include "SceneManager.h"
#include "Collider.h"
#include "EditorCameraController.h"
#include "GameObject.h"
#include "MeshRenderer.h"
#include "Rigidbody.h"
#include "physics/PhysicsECSStore.h"
#include "physics/PhysicsWorld.h"
#include <algorithm>
#include <function/audio/AudioEngine.h>

namespace
{
using ProfileClock = std::chrono::high_resolution_clock;

double ProfileMsSince(ProfileClock::time_point start)
{
    return std::chrono::duration<double, std::milli>(ProfileClock::now() - start).count();
}
} // namespace

namespace infernux
{

SceneManager &SceneManager::Instance()
{
    static SceneManager instance;
    return instance;
}

SceneManager::SceneManager()
{
    // Create editor camera
    m_editorCameraObject = std::make_unique<GameObject>("Editor Camera");
    m_editorCameraComponent = m_editorCameraObject->AddComponent<Camera>();
    m_editorCamera.SetCamera(m_editorCameraComponent);
    m_editorCamera.Reset(); // Set default position
}

Scene *SceneManager::CreateScene(const std::string &name)
{
    auto scene = std::make_unique<Scene>(name);
    Scene *ptr = scene.get();
    m_scenes.push_back(std::move(scene));

    // If no active scene, make this one active
    if (!m_activeScene) {
        SetActiveScene(ptr);
    }

    if (m_onSceneLoaded) {
        m_onSceneLoaded(ptr);
    }

    return ptr;
}

void SceneManager::SetActiveScene(Scene *scene)
{
    m_activeScene = scene;
    // Note: We do NOT auto-assign the editor camera as mainCamera.
    // mainCamera == nullptr means "no game camera assigned" — the Game View
    // will show a placeholder. Scene View always uses the editor camera
    // via SceneRenderBridge / EditorCameraController, independent of mainCamera.
}

void SceneManager::UnloadScene(Scene *scene)
{
    if (!scene)
        return;

    if (m_onSceneUnloaded) {
        m_onSceneUnloaded(scene);
    }

    // If this was the active scene, clear registry and pointer
    if (m_activeScene == scene) {
        ClearComponentRegistries();
        m_activeScene = nullptr;
    }

    auto it = std::find_if(m_scenes.begin(), m_scenes.end(),
                           [scene](const std::unique_ptr<Scene> &s) { return s.get() == scene; });

    if (it != m_scenes.end()) {
        m_scenes.erase(it);
    }

    // Set a new active scene if available
    if (!m_activeScene && !m_scenes.empty()) {
        m_activeScene = m_scenes[0].get();
    }
}

void SceneManager::UnloadAllScenes()
{
    ClearComponentRegistries();

    for (auto &scene : m_scenes) {
        if (m_onSceneUnloaded) {
            m_onSceneUnloaded(scene.get());
        }
    }

    m_scenes.clear();
    m_activeScene = nullptr;
}

Scene *SceneManager::GetScene(const std::string &name) const
{
    for (const auto &scene : m_scenes) {
        if (scene->GetName() == name) {
            return scene.get();
        }
    }
    return nullptr;
}

void SceneManager::Start()
{
    if (m_activeScene) {
        m_activeScene->Start();
    }
}

void SceneManager::Update(float deltaTime)
{
    m_lastFrameProfile = {};

    // Always update editor camera (for editor viewport navigation)
    auto t0 = ProfileClock::now();
    m_editorCamera.Update(deltaTime);
    m_lastFrameProfile.editorCameraMs += ProfileMsSince(t0);

    if (!m_isPlaying && m_activeScene) {
        t0 = ProfileClock::now();
        m_activeScene->EditorUpdate(deltaTime);
        m_lastFrameProfile.editorUpdateMs += ProfileMsSince(t0);
    }

    // Update active scene if playing
    if (m_isPlaying && !m_isPaused && m_activeScene) {
        t0 = ProfileClock::now();
        m_activeScene->ProcessPendingStarts();
        m_lastFrameProfile.pendingStartsMs += ProfileMsSince(t0);

        // ---- Fixed-update accumulator (Unity-style) ----
        float dt = std::min(deltaTime, m_maxFixedDeltaTime);
        m_fixedTimeAccumulator += dt;
        while (m_fixedTimeAccumulator >= m_fixedTimeStep) {
            m_lastFrameProfile.fixedSteps += 1.0;

            // Detect user-driven Transform changes on dynamic Rigidbodies
            // and teleport their Jolt bodies before the physics step.
            t0 = ProfileClock::now();
            SyncExternalRigidbodyMoves();
            m_lastFrameProfile.syncExternalMovesMs += ProfileMsSince(t0);

            // Sync collider transforms before physics step
            t0 = ProfileClock::now();
            SyncCollidersToPhysics();
            m_lastFrameProfile.syncCollidersMs += ProfileMsSince(t0);

            t0 = ProfileClock::now();
            m_activeScene->FixedUpdate(m_fixedTimeStep);
            m_lastFrameProfile.fixedUpdateMs += ProfileMsSince(t0);

            // Step Jolt physics world
            t0 = ProfileClock::now();
            PhysicsWorld::Instance().Step(m_fixedTimeStep);
            m_lastFrameProfile.physicsStepMs += ProfileMsSince(t0);

            // Dispatch collision/trigger callbacks to components (Unity-style)
            t0 = ProfileClock::now();
            PhysicsWorld::Instance().DispatchContactEvents();
            m_lastFrameProfile.physicsEventsMs += ProfileMsSince(t0);

            // Write physics results back to Transforms (dynamic Rigidbodies)
            t0 = ProfileClock::now();
            SyncRigidbodiesToTransform();
            m_lastFrameProfile.syncRigidbodiesMs += ProfileMsSince(t0);

            m_fixedTimeAccumulator -= m_fixedTimeStep;
        }

        t0 = ProfileClock::now();
        ApplyInterpolatedRigidbodies(m_fixedTimeAccumulator / m_fixedTimeStep);
        m_lastFrameProfile.interpolationMs += ProfileMsSince(t0);

        t0 = ProfileClock::now();
        m_activeScene->Update(deltaTime);
        m_lastFrameProfile.gameplayUpdateMs += ProfileMsSince(t0);
    }
}

void SceneManager::FixedUpdate()
{
    // Intentionally empty — fixed update is driven by the accumulator inside
    // Update() for correct time-step handling.  Exposed in the header so
    // external code *could* call it manually if needed, but normally it is
    // not called directly.
}

void SceneManager::LateUpdate(float deltaTime)
{
    if (m_isPlaying && !m_isPaused && m_activeScene) {
        auto t0 = ProfileClock::now();
        m_activeScene->ProcessPendingStarts();
        m_lastFrameProfile.pendingStartsMs += ProfileMsSince(t0);

        t0 = ProfileClock::now();
        m_activeScene->LateUpdate(deltaTime);
        m_lastFrameProfile.lateUpdateMs += ProfileMsSince(t0);
    }

    // Update spatial audio (runs even when paused so listener position stays synced)
    auto t0 = ProfileClock::now();
    AudioEngine::Instance().Update(deltaTime);
    m_lastFrameProfile.audioMs += ProfileMsSince(t0);
}

void SceneManager::EndFrame()
{
    if (m_activeScene) {
        auto t0 = ProfileClock::now();
        m_activeScene->ProcessPendingDestroys();
        m_lastFrameProfile.endFrameMs += ProfileMsSince(t0);
    }
}

void SceneManager::Play()
{
    // Only reset accumulator on initial play, not on resume-from-pause
    if (!m_isPlaying) {
        m_fixedTimeAccumulator = 0.0f;
    }

    m_isPlaying = true;
    m_isPaused = false;

    if (m_activeScene) {
        m_activeScene->SetPlaying(true);
        m_activeScene->Start();

        // Force-sync ALL body positions to current Transform.
        // Editor-mode bodies (created by EnsureSceneBodiesRegistered for picking)
        // may have stale positions if the user moved objects before pressing Play.
        ForceAllBodiesToCurrentTransform();

        // Ensure broad-phase tree is rebuilt after all Awake() calls
        // registered collider bodies, so raycasts work from the first frame.
        PhysicsWorld::Instance().OptimizeBroadPhase();
    }
}

void SceneManager::Stop()
{
    m_isPlaying = false;
    m_isPaused = false;
    m_fixedTimeAccumulator = 0.0f;

    if (m_activeScene) {
        m_activeScene->SetPlaying(false);
    }

    // Scene state restore is handled by Python PlayModeManager
    // (serialize on Play, deserialize on Stop)
}

void SceneManager::Pause()
{
    m_isPaused = !m_isPaused;
}

void SceneManager::Step(float deltaTime)
{
    if (!m_isPaused || !m_isPlaying || !m_activeScene)
        return;

    m_activeScene->ProcessPendingStarts();

    // Detect external moves before stepping physics
    SyncExternalRigidbodyMoves();
    SyncCollidersToPhysics();
    m_activeScene->FixedUpdate(m_fixedTimeStep);
    PhysicsWorld::Instance().Step(m_fixedTimeStep);
    PhysicsWorld::Instance().DispatchContactEvents();
    SyncRigidbodiesToTransform();
    ApplyInterpolatedRigidbodies(1.0f);
    m_activeScene->Update(deltaTime);
    m_activeScene->ProcessPendingStarts();
    m_activeScene->LateUpdate(deltaTime);
    m_activeScene->ProcessPendingDestroys();
}

void SceneManager::DontDestroyOnLoad(GameObject *gameObject)
{
    if (!gameObject)
        return;

    // Must be a root object (no parent) for DontDestroyOnLoad
    if (gameObject->GetParent() != nullptr) {
        // Walk up to root
        GameObject *root = gameObject;
        while (root->GetParent()) {
            root = root->GetParent();
        }
        gameObject = root;
    }

    // Detach from current scene
    Scene *scene = gameObject->GetScene();
    if (!scene)
        return;

    auto owned = scene->DetachRootObject(gameObject);
    if (!owned)
        return;

    // Move to persistent list
    owned->SetScene(nullptr); // No longer belongs to any scene
    m_persistentObjects.push_back(std::move(owned));
}

void SceneManager::SyncCollidersToPhysics()
{
    auto handles = PhysicsECSStore::Instance().GetAliveColliderHandles();
    for (auto handle : handles) {
        auto &data = PhysicsECSStore::Instance().GetCollider(handle);
        auto *col = data.owner;
        if (!col || !col->IsEnabled())
            continue;
        auto *go = col->GetGameObject();
        if (!go || go->GetScene() != m_activeScene)
            continue;
        col->SyncTransformToPhysics();
    }
}

void SceneManager::ForceAllBodiesToCurrentTransform()
{
    auto &pw = PhysicsWorld::Instance();
    if (!pw.IsInitialized())
        return;

    auto handles = PhysicsECSStore::Instance().GetAliveColliderHandles();
    for (auto handle : handles) {
        auto &data = PhysicsECSStore::Instance().GetCollider(handle);
        auto *col = data.owner;
        if (!col || col->GetBodyId() == 0xFFFFFFFF)
            continue;

        auto *go = col->GetGameObject();
        if (!go)
            continue;

        Transform *tf = go->GetTransform();
        if (!tf)
            continue;

        glm::quat rot = tf->GetWorldRotation();
        glm::vec3 pos = tf->GetPosition();
        pw.SetBodyPosition(col->GetBodyId(), pos, rot);
    }
}

void SceneManager::SyncRigidbodiesToTransform()
{
    auto handles = PhysicsECSStore::Instance().GetAliveRigidbodyHandles();
    for (auto handle : handles) {
        auto &data = PhysicsECSStore::Instance().GetRigidbody(handle);
        auto *rb = data.owner;
        if (!rb || !rb->IsEnabled())
            continue;
        auto *go = rb->GetGameObject();
        if (!go || go->GetScene() != m_activeScene)
            continue;
        rb->SyncPhysicsToTransform();
    }
}

void SceneManager::ApplyInterpolatedRigidbodies(float alpha)
{
    if (!m_activeScene)
        return;

    auto handles = PhysicsECSStore::Instance().GetAliveRigidbodyHandles();
    for (auto handle : handles) {
        auto &data = PhysicsECSStore::Instance().GetRigidbody(handle);
        auto *rb = data.owner;
        if (!rb || !rb->IsEnabled())
            continue;
        auto *go = rb->GetGameObject();
        if (!go || go->GetScene() != m_activeScene)
            continue;
        rb->ApplyInterpolatedTransform(alpha);
    }
}

void SceneManager::SyncExternalRigidbodyMoves()
{
    auto handles = PhysicsECSStore::Instance().GetAliveRigidbodyHandles();
    for (auto handle : handles) {
        auto &data = PhysicsECSStore::Instance().GetRigidbody(handle);
        auto *rb = data.owner;
        if (!rb || !rb->IsEnabled())
            continue;
        auto *go = rb->GetGameObject();
        if (!go || go->GetScene() != m_activeScene)
            continue;
        rb->SyncExternalMovesToPhysics();
    }
}

// ============================================================================
// Component registries
// ============================================================================

void SceneManager::ClearComponentRegistries()
{
    m_activeMeshRenderers.clear();
    m_activeMeshRendererSet.clear();
    m_activeLights.clear();
}

// ========================================================================
// MeshRenderer component registry
// ========================================================================

void SceneManager::RegisterMeshRenderer(MeshRenderer *renderer)
{
    if (!renderer)
        return;
    if (m_activeMeshRendererSet.insert(renderer).second) {
        m_activeMeshRenderers.push_back(renderer);
    }
}

void SceneManager::UnregisterMeshRenderer(MeshRenderer *renderer)
{
    if (!m_activeMeshRendererSet.erase(renderer))
        return;
    // Swap-and-pop for O(1) removal from vector
    for (size_t i = 0; i < m_activeMeshRenderers.size(); ++i) {
        if (m_activeMeshRenderers[i] == renderer) {
            m_activeMeshRenderers[i] = m_activeMeshRenderers.back();
            m_activeMeshRenderers.pop_back();
            return;
        }
    }
}

void SceneManager::MarkMeshRenderersDirtyForAsset(const std::string &meshGuid)
{
    if (meshGuid.empty())
        return;
    for (auto *renderer : m_activeMeshRenderers) {
        if (renderer && renderer->HasMeshAsset() && renderer->GetMeshAssetGuid() == meshGuid) {
            renderer->MarkMeshBufferDirty();
            // Update local bounds from the reloaded mesh
            auto mesh = renderer->GetMeshAssetRef().Get();
            if (mesh)
                renderer->SetLocalBounds(mesh->GetBoundsMin(), mesh->GetBoundsMax());
        }
    }
}

// ========================================================================
// Light component registry
// ========================================================================

void SceneManager::RegisterLight(Light *light)
{
    if (!light)
        return;
    for (auto *l : m_activeLights) {
        if (l == light)
            return;
    }
    m_activeLights.push_back(light);
}

void SceneManager::UnregisterLight(Light *light)
{
    for (size_t i = 0; i < m_activeLights.size(); ++i) {
        if (m_activeLights[i] == light) {
            m_activeLights[i] = m_activeLights.back();
            m_activeLights.pop_back();
            return;
        }
    }
}

} // namespace infernux
