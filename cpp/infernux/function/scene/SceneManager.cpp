// Jolt types hidden behind opaque headers — no Jolt include needed
#include "SceneManager.h"
#include "Collider.h"
#include "EditorCameraController.h"
#include "GameObject.h"
#include "Light.h"
#include "MeshRenderer.h"
#include "Rigidbody.h"
#include "Transform.h"
#include "TransformECSStore.h"
#include "physics/PhysicsECSStore.h"
#include "physics/PhysicsWorld.h"
#include <InxLog.h>
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

    // ── Migrate DontDestroyOnLoad objects to the new scene ──
    if (scene && !m_persistentObjects.empty()) {
        for (auto &obj : m_persistentObjects) {
            if (!obj)
                continue;
            GameObject *raw = obj.get();
            scene->AttachRootObject(std::move(obj)); // sets scene ptr on tree

            // Re-register MeshRenderers and Lights that were cleared
            // when the old scene was unloaded (ClearComponentRegistries).
            for (auto *mr : raw->GetComponentsInChildren<MeshRenderer>()) {
                if (mr && mr->IsEnabled())
                    RegisterMeshRenderer(mr);
            }
            for (auto *lt : raw->GetComponentsInChildren<Light>()) {
                if (lt && lt->IsEnabled())
                    RegisterLight(lt);
            }
        }
        m_persistentObjects.clear();
    }

    // Note: We do NOT auto-assign the editor camera as mainCamera.
    // mainCamera == nullptr means "no game camera assigned" — the Game View
    // will show a placeholder. Scene View always uses the editor camera
    // via SceneRenderBridge / EditorCameraController, independent of mainCamera.
}

void SceneManager::UnloadScene(Scene *scene)
{
    if (!scene)
        return;

    // ── Extract persistent (DontDestroyOnLoad) root objects before unload ──
    ExtractPersistentObjects(scene);

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
    // ── Extract persistent objects from all scenes before unload ──
    for (auto &scene : m_scenes) {
        ExtractPersistentObjects(scene.get());
    }

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

            // Flush deferred broadphase additions (Unity-style batch add)
            FlushPendingBroadphase();

            // Sync collider transforms before physics step (serial-skip when nothing moved)
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

        auto tStart = ProfileClock::now();
        m_activeScene->Start();
        double startMs = ProfileMsSince(tStart);

        // Force-sync ALL body positions to current Transform.
        ForceAllBodiesToCurrentTransform();

        // Flush any deferred broadphase additions from Awake/OnEnable, then
        // rebuild broad-phase tree so raycasts work from the first frame.
        auto tFlush = ProfileClock::now();
        FlushPendingBroadphase();
        PhysicsWorld::Instance().OptimizeBroadPhase();
        double flushMs = ProfileMsSince(tFlush);

        if (startMs + flushMs > 500.0) {
            INXLOG_INFO("[Perf] Play(): Start=", static_cast<int>(startMs), "ms, Flush=", static_cast<int>(flushMs),
                        "ms");
        }

        // Reset physics sync serial so the first fixed step does a full sync.
        m_lastPhysicsSyncTransformSerial = 0;
    }
}

void SceneManager::Stop()
{
    m_isPlaying = false;
    m_isPaused = false;
    m_fixedTimeAccumulator = 0.0f;

    // Discard any persistent objects — play session is over.
    m_persistentObjects.clear();

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
    FlushPendingBroadphase();
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

    // Walk up to root if called on a child
    if (gameObject->GetParent() != nullptr) {
        GameObject *root = gameObject;
        while (root->GetParent()) {
            root = root->GetParent();
        }
        gameObject = root;
    }

    // Just mark as persistent — the object stays in its scene normally.
    // When a scene is unloaded, persistent roots are migrated to the new scene.
    gameObject->SetPersistent(true);
}

void SceneManager::ExtractPersistentObjects(Scene *scene)
{
    if (!scene)
        return;

    // Collect persistent roots — iterate by index since DetachRootObject
    // modifies the vector.
    std::vector<GameObject *> toExtract;
    for (const auto &root : scene->GetRootObjects()) {
        if (root && root->IsPersistent())
            toExtract.push_back(root.get());
    }

    for (GameObject *go : toExtract) {
        auto owned = scene->DetachRootObject(go);
        if (owned) {
            owned->SetScene(nullptr);
            m_persistentObjects.push_back(std::move(owned));
        }
    }
}

void SceneManager::SyncCollidersToPhysics()
{
    // ── Unity-style deferred transform sync ──
    // Skip the entire collider walk when no transform has been invalidated
    // since the last sync.  This is the single biggest win for static scenes
    // (thousands of colliders, none of which moved).
    auto &tStore = TransformECSStore::Instance();
    uint64_t currentSerial = tStore.GetGlobalTransformSerial();
    if (currentSerial == m_lastPhysicsSyncTransformSerial) {
        return; // nothing moved — zero work
    }
    m_lastPhysicsSyncTransformSerial = currentSerial;

    // Something moved — walk all colliders.  Each collider's
    // SyncTransformToPhysics() has its own lastSyncedPos/Rot early-out
    // so only colliders that actually moved pay for a Jolt call.
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

void SceneManager::FlushPendingBroadphase()
{
    auto &store = PhysicsECSStore::Instance();
    auto &pw = PhysicsWorld::Instance();
    if (!pw.IsInitialized())
        return;

    // ── Phase 1: Create deferred Jolt bodies ──
    auto pendingBodies = store.ConsumePendingBodyCreations();
    if (pendingBodies.empty() && !store.HasPendingBroadphaseAdds())
        return;

    auto t0 = ProfileClock::now();
    const size_t bodyCount = pendingBodies.size();

    for (auto handle : pendingBodies) {
        if (!store.IsValid(handle))
            continue;
        auto &data = store.GetCollider(handle);
        auto *col = data.owner;
        if (!col || !col->IsEnabled() || data.bodyId != 0xFFFFFFFF)
            continue;

        // Actually create the Jolt body (deferred from Awake)
        col->RegisterBody();

        // Queue broadphase add (consumed in Phase 2 below)
        if (data.bodyId != 0xFFFFFFFF) {
            col->AddToBroadphase();
        }
    }

    double phase1Ms = ProfileMsSince(t0);

    // ── Phase 2: Batch add to broadphase ──
    auto t1 = ProfileClock::now();
    auto pending = store.ConsumePendingBroadphaseAdds();
    if (pending.empty())
        return;

    // Use Jolt batch API (AddBodiesPrepare/Finalize) for large batches,
    // which is significantly faster than individual AddBody calls.
    pw.AddBodiesBatch(pending);

    double phase2Ms = ProfileMsSince(t1);

    // Rebuild broad-phase tree once after batch-adding all new bodies.
    auto t2 = ProfileClock::now();
    pw.OptimizeBroadPhase();
    double optimizeMs = ProfileMsSince(t2);

    // if (bodyCount >= 100) {
    //     INXLOG_INFO("[Perf] FlushPendingBroadphase: ", bodyCount, " bodies — "
    //                 "CreateBody: ", static_cast<int>(phase1Ms), "ms, "
    //                 "AddBatch: ", static_cast<int>(phase2Ms), "ms, "
    //                 "Optimize: ", static_cast<int>(optimizeMs), "ms");
    // }
}

void SceneManager::SyncTransforms()
{
    // Flush any pending broadphase additions first
    FlushPendingBroadphase();
    // Force a full collider sync regardless of serial
    m_lastPhysicsSyncTransformSerial = 0; // invalidate cache
    SyncCollidersToPhysics();
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
    ++m_meshRendererVersion;
}

// ========================================================================
// MeshRenderer component registry
// ========================================================================

void SceneManager::ReserveRendererCapacity(size_t count)
{
    m_activeMeshRenderers.reserve(m_activeMeshRenderers.size() + count);
    m_activeMeshRendererSet.reserve(m_activeMeshRendererSet.size() + count);
}

void SceneManager::RegisterMeshRenderer(MeshRenderer *renderer)
{
    if (!renderer)
        return;
    if (m_activeMeshRendererSet.insert(renderer).second) {
        m_activeMeshRenderers.push_back(renderer);
        ++m_meshRendererVersion;
    }
}

void SceneManager::UnregisterMeshRenderer(MeshRenderer *renderer)
{
    if (!m_activeMeshRendererSet.erase(renderer))
        return;
    ++m_meshRendererVersion;
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
