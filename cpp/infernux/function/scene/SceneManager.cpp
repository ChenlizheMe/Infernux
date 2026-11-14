// Jolt types hidden behind opaque headers — no Jolt include needed
#include "SceneManager.h"
#include "Collider.h"
#if !defined(INFERNUX_RUNTIME_MINIMAL_HOST)
#include "EditorCameraController.h"
#endif
#include "GameObject.h"
#include "Light.h"
#include "MeshCollider.h"
#include "MeshRenderer.h"
#include "Rigidbody.h"
#include "SkeletonInstance.h"
#include "SkinnedMeshRenderer.h"
#include "Transform.h"
#include "TransformECSStore.h"
#include "physics/PhysicsDomain.h"
#include "physics/PhysicsECSStore.h"
#include "physics/PhysicsWorld.h"
#if !defined(INFERNUX_RUNTIME_MINIMAL_HOST)
#include <SDL3/SDL.h>
#endif
#include <algorithm>
#include <core/log/InxLog.h>
#include <core/threading/FrameWorkOwner.h>
#include <optional>
#if !defined(INFERNUX_RUNTIME_MINIMAL_HOST)
#include <function/audio/AudioEngine.h>
#endif
#include <platform/input/InputManager.h>

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

namespace
{
#if !defined(INFERNUX_RUNTIME_MINIMAL_HOST)
void UpdateCapturedEditorCamera(EditorCameraController &controller, float deltaTime)
{
    InputManager &input = InputManager::Instance();
    const bool captured = input.IsEditorMouseCaptureActive();
    const bool rightDown = captured && input.GetMouseButton(1);
    const bool middleDown = captured && input.GetMouseButton(2);

    if (!captured)
        return;
    const auto [deltaX, deltaY] = input.ConsumeEditorMouseDelta();
    const auto key = [&input, rightDown](SDL_Scancode scancode) {
        return rightDown && input.GetKey(static_cast<int>(scancode));
    };
    controller.ProcessInput(deltaTime, rightDown, middleDown, deltaX, deltaY, key(SDL_SCANCODE_W), key(SDL_SCANCODE_A),
                            key(SDL_SCANCODE_S), key(SDL_SCANCODE_D), key(SDL_SCANCODE_Q), key(SDL_SCANCODE_E),
                            key(SDL_SCANCODE_LSHIFT) || key(SDL_SCANCODE_RSHIFT));
}
#endif
} // namespace

SceneManager &SceneManager::Instance()
{
    // Intentionally leaked: scene teardown is driven explicitly by
    // Infernux::Cleanup() -> Shutdown(); never by C++ static destruction,
    // whose cross-TU ordering vs the ECS stores is undefined (was the root
    // cause of the shutdown heap corruption / DLL-unload access violations).
    static SceneManager *instance = new SceneManager();
    return *instance;
}

RuntimeWorld &SceneManager::CurrentWorld() noexcept
{
    auto *world = WorldExecutionScope::Current();
    return world ? *world : m_runtimeWorlds.empty() ? *m_authoringWorld : *m_runtimeWorlds.front();
}

const RuntimeWorld &SceneManager::CurrentWorld() const noexcept
{
    auto *world = WorldExecutionScope::Current();
    return world ? *world : m_runtimeWorlds.empty() ? *m_authoringWorld : *m_runtimeWorlds.front();
}

uint64_t SceneManager::GetCurrentRuntimeWorldId() const noexcept
{
    const auto *world = WorldExecutionScope::Current();
    if (!world)
        world = m_runtimeWorlds.empty() ? m_authoringWorld.get() : m_runtimeWorlds.front().get();
    return world ? world->id : 0;
}

SceneManager::SceneManager() : m_authoringWorld(std::make_unique<RuntimeWorld>())
{
    m_authoringWorld->random = Random::ToolStorage();
    RefreshThreadDefaultRandomStorage();
    TransformECSStore::Instance().SetInvalidationObserver([this](Transform *transform, const Transform *physicsSource,
                                                                 const Transform *authoredSource,
                                                                 bool rotationAuthored) {
        auto *gameObject = transform ? transform->GetGameObject() : nullptr;
        if (!gameObject)
            return;
        PhysicsECSStore::Instance().MarkGameObjectDirty(gameObject, physicsSource, authoredSource, rotationAuthored);
        Scene *scene = gameObject->GetScene();
        if (!scene || !scene->GetRuntimeWorld())
            return;
        const WorldExecutionScope worldScope(scene->GetRuntimeWorld());
        if (const auto bone = CurrentWorld().render.skeletonBoneOwners.find(transform);
            bone != CurrentWorld().render.skeletonBoneOwners.end())
            bone->second->MarkPoseDirty();
        // A moving camera, light, audio source, or UI element must not force
        // the renderer to rescan every geometry proxy. Invalidate the shared
        // RenderWorld only for objects that actually contribute geometry.
        // Transform invalidation walks descendants, so moving a plain parent
        // still reaches and invalidates every child MeshRenderer correctly.
        if (scene && IsRuntimeScene(scene) && gameObject->GetComponent<MeshRenderer>()) {
            ++CurrentWorld().render.renderTransformRevision;
            if (CurrentWorld().render.renderTransformRevision == 0)
                CurrentWorld().render.renderTransformRevision = 1;
        }
    });

#if !defined(INFERNUX_RUNTIME_MINIMAL_HOST)
    // Create editor camera
    m_editorCameraObject = std::make_unique<GameObject>("Editor Camera");
    m_editorCameraComponent = m_editorCameraObject->AddComponent<Camera>();
    m_editorCamera.SetCamera(m_editorCameraComponent);
    m_editorCamera.Reset(); // Set default position
#endif
}

bool SceneManager::IsPreparingRuntimeWorld(const RuntimeWorld &world) const noexcept
{
    return std::any_of(m_preparingWorlds.begin(), m_preparingWorlds.end(),
                       [&world](const auto &candidate) { return candidate.get() == &world; });
}

RuntimeWorld *SceneManager::GetPrimaryRuntimeWorld() const noexcept
{
    return m_runtimeWorlds.empty() ? nullptr : m_runtimeWorlds.front().get();
}

std::vector<uint64_t> SceneManager::GetRuntimeWorldIds() const
{
    std::vector<uint64_t> ids;
    ids.reserve(m_runtimeWorlds.size());
    for (const auto &world : m_runtimeWorlds)
        ids.push_back(world->id);
    return ids;
}

RuntimeWorld *SceneManager::FindRuntimeWorld(uint64_t id) const noexcept
{
    if (m_authoringWorld && m_authoringWorld->id == id)
        return m_authoringWorld.get();
    for (const auto *worlds : {&m_runtimeWorlds, &m_preparingWorlds, &m_retiringWorlds})
        for (const auto &world : *worlds)
            if (world->id == id)
                return world.get();
    return nullptr;
}

void SceneManager::RefreshThreadDefaultRandomStorage() const
{
    const auto *world = m_runtimeWorlds.empty() ? m_authoringWorld.get() : m_runtimeWorlds.front().get();
    Random::SetThreadDefaultStorage(world ? world->random : nullptr);
}

SceneManager::RuntimeWorldInitialSettings SceneManager::CaptureAuthoringWorldSettings() const
{
    return {m_authoringWorld->simulation.fixedTimeStep, m_authoringWorld->simulation.maxFixedDeltaTime,
            m_authoringWorld->simulation.timeScale, m_authoringWorld->physics->gravity,
            Random::CaptureStorage(*m_authoringWorld->random)};
}

uint64_t SceneManager::BeginRuntimeWorldManagementSession()
{
    if (m_activeManagementSession)
        throw std::logic_error("a RuntimeWorld management session is already active");
    m_initialRuntimeWorldSettings = CaptureAuthoringWorldSettings();
    m_terminalWorldHandles.clear();
    m_activeManagementSession = ++m_managementSessionSequence;
    return m_activeManagementSession;
}

void SceneManager::EndRuntimeWorldManagementSession()
{
    m_terminalWorldHandles.clear();
    m_activeManagementSession = 0;
    m_initialRuntimeWorldSettings.reset();
    RefreshThreadDefaultRandomStorage();
}

std::shared_ptr<RuntimeWorldHandle> SceneManager::ReserveRuntimeWorld()
{
    // Safe even inside OnDestroy: reserve metadata only, without world/ECS mutation.
    auto handle = RuntimeWorld::ReserveHandle();
    handle->managementSession = m_activeManagementSession;
    m_reservedWorlds.push_back(handle);
    return handle;
}

RuntimeWorld &SceneManager::BeginRuntimeWorld()
{
    return BeginReservedRuntimeWorld(ReserveRuntimeWorld());
}

RuntimeWorld &SceneManager::BeginReservedRuntimeWorld(const std::shared_ptr<RuntimeWorldHandle> &handle)
{
    Scene::RequireNoWorldRetirement();
    const auto it = std::find(m_reservedWorlds.begin(), m_reservedWorlds.end(), handle);
    if (it == m_reservedWorlds.end() || handle->destroyRequested || handle->state != "preparing")
        throw std::invalid_argument("RuntimeWorld reservation is not available");
    if (handle->managementSession != m_activeManagementSession)
        throw std::invalid_argument("RuntimeWorld reservation belongs to a different management session");
    // A managed clone always starts from this Play's frozen initial settings.
    // Standalone native fixtures retain their existing authoring-copy contract.
    const auto initial =
        handle->managementSession ? m_initialRuntimeWorldSettings.value() : CaptureAuthoringWorldSettings();
    auto world = std::make_unique<RuntimeWorld>(handle);
    auto &result = *world;
    result.simulation.fixedTimeStep = initial.fixedTimeStep;
    result.simulation.maxFixedDeltaTime = initial.maxFixedDeltaTime;
    result.simulation.timeScale = initial.timeScale;
    result.physics->gravity = initial.gravity;
    Random::RestoreStorage(*result.random, initial.random);
    m_preparingWorlds.push_back(std::move(world));
    m_reservedWorlds.erase(it);
    return result;
}

std::shared_ptr<RuntimeWorldHandle> SceneManager::GetRuntimeWorldHandle(uint64_t id) const
{
    for (const auto &handle : m_reservedWorlds)
        if (handle->id == id)
            return handle;
    auto *world = FindRuntimeWorld(id);
    if (!world || world == m_authoringWorld.get())
        throw std::invalid_argument("RuntimeWorld identity is not managed");
    return world->handle;
}

std::vector<std::shared_ptr<RuntimeWorldHandle>> SceneManager::GetManagedRuntimeWorldHandles() const
{
    std::vector<std::shared_ptr<RuntimeWorldHandle>> handles;
    for (const auto *worlds : {&m_runtimeWorlds, &m_preparingWorlds, &m_retiringWorlds})
        for (const auto &world : *worlds)
            if (world->handle->managementSession == m_activeManagementSession)
                handles.push_back(world->handle);
    for (const auto &handle : m_reservedWorlds)
        if (handle->managementSession == m_activeManagementSession)
            handles.push_back(handle);
    handles.insert(handles.end(), m_terminalWorldHandles.begin(), m_terminalWorldHandles.end());
    return handles;
}

void SceneManager::CancelReservedRuntimeWorld(const std::shared_ptr<RuntimeWorldHandle> &handle,
                                              const std::string &state, const std::string &reason)
{
    if (state != "cancelled" && state != "failed")
        throw std::invalid_argument("invalid reservation terminal state");
    const auto it = std::find(m_reservedWorlds.begin(), m_reservedWorlds.end(), handle);
    if (it == m_reservedWorlds.end())
        throw std::invalid_argument("RuntimeWorld reservation has already materialized");
    handle->state = state;
    handle->error = reason;
    handle->pendingPrimary = false;
    if (m_activeManagementSession && handle->managementSession == m_activeManagementSession)
        m_terminalWorldHandles.push_back(handle);
    m_reservedWorlds.erase(it);
}

void SceneManager::RequestRuntimeWorldPrimary(const std::shared_ptr<RuntimeWorldHandle> &handle)
{
    if (!handle || handle->destroyRequested || (handle->state != "ready" && handle->state != "preparing"))
        throw std::invalid_argument("primary target must be preparing or ready");
    if (GetRuntimeWorldHandle(handle->id) != handle)
        throw std::invalid_argument("primary target is not in this runtime collection");
    for (const auto &candidate : GetManagedRuntimeWorldHandles())
        candidate->pendingPrimary = false;
    handle->pendingPrimary = true;
}

Scene *SceneManager::CreatePreparingPersistentScene()
{
    Scene::RequireNoWorldRetirement();
    if (!IsPreparingRuntimeWorld(CurrentWorld()))
        throw std::logic_error("persistent template restore requires a preparing world");
    auto &world = CurrentWorld();
    if (!world.scenes.persistent) {
        world.scenes.persistent = std::make_unique<Scene>("DontDestroyOnLoad");
        world.scenes.persistent->m_runtimeWorld = &world;
        world.scenes.persistent->SetRuntimeLifecycleSchedulerEnabled(m_runtimeLifecycleSchedulerEnabled);
    }
    return world.scenes.persistent.get();
}

void SceneManager::RemapPreparingWorldReferences(uint64_t id, const std::unordered_map<uint64_t, uint64_t> &components)
{
    auto *world = FindRuntimeWorld(id);
    if (!world || !IsPreparingRuntimeWorld(*world))
        throw std::invalid_argument("reference remap requires a preparing RuntimeWorld");
    const WorldExecutionScope scope(world);
    std::vector<Scene *> initialScenes;
    for (const auto &scene : world->scenes.loaded)
        initialScenes.push_back(scene.get());
    if (world->scenes.persistent)
        initialScenes.push_back(world->scenes.persistent.get());
    for (auto *scene : initialScenes)
        for (auto *object : scene->GetAllObjects()) {
            object->GetTransform()->RemapComponentReferences(components);
            for (const auto &component : object->GetAllComponents())
                component->RemapComponentReferences(components);
        }
}

bool SceneManager::PublishRuntimeWorld(uint64_t id)
{
    Scene::RequireNoWorldRetirement();
    const auto it = std::find_if(m_preparingWorlds.begin(), m_preparingWorlds.end(),
                                 [id](const auto &world) { return world->id == id; });
    if (it == m_preparingWorlds.end())
        throw std::invalid_argument("RuntimeWorld is not awaiting graph publication");
    for (const auto &scene : (*it)->scenes.loaded)
        if (scene->HasPendingPyComponents())
            throw std::logic_error("RuntimeWorld Python graph must be attached before publication");
    if ((*it)->scenes.persistent && (*it)->scenes.persistent->HasPendingPyComponents())
        throw std::logic_error("RuntimeWorld persistent Python graph must be attached before publication");
    auto &world = **it;
    m_runtimeWorlds.push_back(std::move(*it));
    m_preparingWorlds.erase(it);
    RefreshThreadDefaultRandomStorage();
    const WorldExecutionScope scope(&world);
    const auto admittedState = GetSimulationState();
    const auto current = [&] {
        return GetSimulationState() == admittedState &&
               std::any_of(m_runtimeWorlds.begin(), m_runtimeWorlds.end(),
                           [id](const auto &candidate) { return candidate->id == id; });
    };
    world.handle->state = "ready";
    std::vector<std::pair<uint64_t, std::vector<ObjectHandle>>> published;
    std::vector<Scene *> initialScenes;
    for (const auto &scene : world.scenes.loaded)
        initialScenes.push_back(scene.get());
    if (world.scenes.persistent)
        initialScenes.push_back(world.scenes.persistent.get());
    for (auto *scene : initialScenes) {
        auto &roots = published.emplace_back(scene->GetWorldId(), std::vector<ObjectHandle>{}).second;
        for (const auto &root : scene->GetRootObjects())
            roots.push_back(root->GetHandle());
    }
    for (const auto &[sceneId, roots] : published)
        for (auto handle : roots) {
            if (!current())
                return false;
            if (auto *scene = GetSceneByWorldId(sceneId))
                scene->AwakeObject(scene->ResolveGameObject(handle));
        }
    return current();
}

void SceneManager::SetPrimaryRuntimeWorld(uint64_t id)
{
    Scene::RequireNoWorldRetirement();
    const auto it = std::find_if(m_runtimeWorlds.begin(), m_runtimeWorlds.end(),
                                 [id](const auto &world) { return world->id == id; });
    if (it == m_runtimeWorlds.end())
        throw std::invalid_argument("primary RuntimeWorld must be published");
    std::rotate(m_runtimeWorlds.begin(), it, std::next(it));
    (*m_runtimeWorlds.begin())->handle->pendingPrimary = false;
    RefreshThreadDefaultRandomStorage();
}

void SceneManager::BeginRuntimeWorldRetirement(uint64_t id)
{
    Scene::RequireNoWorldRetirement();
    if (std::any_of(m_retiringWorlds.begin(), m_retiringWorlds.end(),
                    [id](const auto &world) { return world->id == id; }))
        return;
    const bool wasPlaying = std::any_of(m_runtimeWorlds.begin(), m_runtimeWorlds.end(),
                                        [](const auto &world) { return world->simulation.isPlaying; });
    for (auto *worlds : {&m_runtimeWorlds, &m_preparingWorlds}) {
        const auto it =
            std::find_if(worlds->begin(), worlds->end(), [id](const auto &world) { return world->id == id; });
        if (it == worlds->end())
            continue;
        const WorldExecutionScope scope(it->get());
        auto &world = **it;
        world.handle->state = "retiring";
        world.handle->pendingPrimary = false;
        PhysicsWorld::Instance().BeginRetireDomain(world);
        world.frame.pending.clear();
        CloseRuntimeFrame(false);
        world.simulation.isPlaying = false;
        world.simulation.isPaused = false;
        world.simulation.pausedStepPending = false;
        ++world.simulation.simulationEpoch;
        if (m_runtimeWorldBeginRetirementCallback)
            m_runtimeWorldBeginRetirementCallback(id);
        m_retiringWorlds.push_back(std::move(*it));
        worlds->erase(it);
        RefreshThreadDefaultRandomStorage();
        const bool playing = std::any_of(m_runtimeWorlds.begin(), m_runtimeWorlds.end(),
                                         [](const auto &candidate) { return candidate->simulation.isPlaying; });
        if (playing != wasPlaying && m_onPlayStateChanged)
            m_onPlayStateChanged(playing);
        return;
    }
    throw std::invalid_argument("RuntimeWorld is not preparing or published");
}

bool SceneManager::RetireRuntimeWorldScenes(uint64_t id)
{
    Scene::RequireNoWorldRetirement();
    const auto it = std::find_if(m_retiringWorlds.begin(), m_retiringWorlds.end(),
                                 [id](const auto &world) { return world->id == id; });
    if (it == m_retiringWorlds.end())
        throw std::invalid_argument("RuntimeWorld has not begun retirement");
    auto &world = **it;
    if (world.retirementScenesComplete)
        return true;
    if (world.executionScopes || world.lifecycle.runtimeLifecycleFrameOpen)
        return false;
    {
        const WorldExecutionScope scope(&world);
        const InputReadScope hostInput({}, 0);
        if (m_runtimeFrameRetirement)
            m_runtimeFrameRetirement(world.id);
        // Scene destruction is the existing synchronous lifecycle quantum. Recheck
        // the shared Host allowance between Scenes, never halfway through a destructor.
        const auto available = [this] {
            return !m_runtimeFrameWork || !m_runtimeFrameWork->IsActive() || m_runtimeFrameWork->Remaining() > 0;
        };
        if (world.scenes.persistent) {
            if (!available())
                return false;
            ClearRuntimePersistentScene();
        }
        while (!world.scenes.loaded.empty()) {
            if (!available())
                return false;
            UnloadScene(world.scenes.loaded.back().get());
        }
        world.retirementScenesComplete = true;
    }
    return true;
}

void SceneManager::SetRuntimeWorldRetirementCallbacks(std::function<void(uint64_t)> begin,
                                                      std::function<void(uint64_t)> finish)
{
    m_runtimeWorldBeginRetirementCallback = std::move(begin);
    m_runtimeWorldRetirementCallback = std::move(finish);
}

bool SceneManager::FinishRuntimeWorldRetirement(uint64_t id)
{
    const auto it = std::find_if(m_retiringWorlds.begin(), m_retiringWorlds.end(),
                                 [id](const auto &world) { return world->id == id; });
    if (it == m_retiringWorlds.end())
        throw std::invalid_argument("RuntimeWorld has not begun retirement");
    const auto &world = **it;
    if (world.executionScopes || world.lifecycle.runtimeLifecycleFrameOpen || !world.scenes.loaded.empty() ||
        world.scenes.persistent)
        throw std::logic_error("RuntimeWorld callbacks and scenes must retire before its domain");
    if (!world.retirementScenesComplete)
        throw std::logic_error("RuntimeWorld scene retirement has not completed");
    if (!PhysicsWorld::Instance().CanRetireDomain(world))
        return false;
    {
        const WorldExecutionScope scope(it->get());
        if (m_runtimeWorldRetirementCallback)
            m_runtimeWorldRetirementCallback(id);
    }
    world.handle->state = world.handle->retirementOutcome;
    if (m_activeManagementSession && world.handle->managementSession == m_activeManagementSession)
        m_terminalWorldHandles.push_back(world.handle);
    m_retiringWorlds.erase(it);
    return true;
}

bool SceneManager::CollectRetiredRuntimeWorlds()
{
    std::vector<uint64_t> ids;
    for (const auto &world : m_retiringWorlds)
        ids.push_back(world->id);
    for (uint64_t id : ids) {
        if (!RetireRuntimeWorldScenes(id))
            continue;
        if (m_runtimeFrameWork && m_runtimeFrameWork->IsActive() && m_runtimeFrameWork->Remaining() <= 0)
            break;
        FinishRuntimeWorldRetirement(id);
    }
    return m_retiringWorlds.empty();
}

void SceneManager::EnsureAuthoringWorld()
{
    if (m_authoringWorld) {
        RefreshThreadDefaultRandomStorage();
        return;
    }
    m_authoringWorld = std::make_unique<RuntimeWorld>();
    m_authoringWorld->random = Random::ToolStorage();
    RefreshThreadDefaultRandomStorage();
#if !defined(INFERNUX_RUNTIME_MINIMAL_HOST)
    m_editorCameraObject = std::make_unique<GameObject>("Editor Camera");
    m_editorCameraComponent = m_editorCameraObject->AddComponent<Camera>();
    m_editorCamera.SetCamera(m_editorCameraComponent);
    m_editorCamera.Reset();
#endif
}

void SceneManager::CompleteShutdown()
{
    // PhysicsWorld::Shutdown has settled its shared service, not a domain wait loop.
    if (!CollectRetiredRuntimeWorlds() ||
        (m_authoringWorld && !PhysicsWorld::Instance().CanRetireDomain(*m_authoringWorld)))
        throw std::logic_error("shared physics shutdown retained an unfinished RuntimeWorld");
    if (m_authoringWorld && m_runtimeWorldRetirementCallback) {
        const WorldExecutionScope scope(m_authoringWorld.get());
        m_runtimeWorldRetirementCallback(m_authoringWorld->id);
    }
    ClearRuntimeLifecycleCallbacks();
    m_authoringWorld.reset();
    RefreshThreadDefaultRandomStorage();
    m_runtimeFramePublication = nullptr;
    m_runtimeFrameRetirement = nullptr;
    m_runtimeWorldBeginRetirementCallback = nullptr;
    m_runtimeWorldRetirementCallback = nullptr;
    UnbindRuntimeFrameWork();
    m_onSceneLoaded = nullptr;
    m_onSceneUnloaded = nullptr;
    m_onPlayStateChanged = nullptr;
}

uint64_t SceneManager::GetGlobalTransformSerial() const
{
    return TransformECSStore::Instance().GetGlobalTransformSerial();
}

void SceneManager::PublishPhysicsTransformsToRenderer() noexcept
{
    PhysicsECSStore::Instance().NotifyCollisionSceneChanged();
    ++CurrentWorld().render.renderTransformRevision;
    if (CurrentWorld().render.renderTransformRevision == 0)
        CurrentWorld().render.renderTransformRevision = 1;
}

void SceneManager::PublishPhysicsStateToRenderer()
{
    if (!PhysicsWorld::Instance().PublishRenderPoses())
        return;
    InvalidatePhysicsSkeletonPoses();
    PublishPhysicsTransformsToRenderer();
}

Scene *SceneManager::CreateScene(const std::string &name)
{
    const WorldExecutionScope worldScope(&CurrentWorld());
    auto &world = CurrentWorld();
    auto scene = std::make_unique<Scene>(name);
    scene->m_runtimeWorld = &world;
    scene->SetRuntimeLifecycleSchedulerEnabled(m_runtimeLifecycleSchedulerEnabled);
    Scene *ptr = scene.get();
    world.scenes.loaded.push_back(std::move(scene));
    world.scenes.loadedSet.insert(ptr);

    // If no active scene, make this one active
    if (!world.scenes.active) {
        SetActiveScene(ptr);
    }

    if (world.simulation.isPlaying) {
        ptr->SetPlaying(true);
        ptr->Start();
    }

    if (auto callback = m_onSceneLoaded) {
        callback(ptr);
    }

    return ptr;
}

Scene *SceneManager::CreatePreviewScene(const std::string &name)
{
    auto scene = std::make_unique<Scene>(name, true);
    // Python records still need the binding bridge; execution is excluded by
    // preview membership, not by disabling authored component flags.
    scene->SetRuntimeLifecycleSchedulerEnabled(true);
    auto *result = scene.get();
    m_previewScenes.push_back(std::move(scene));
    return result;
}

void SceneManager::ClosePreviewScene(Scene *scene)
{
    Scene::RequireNoWorldRetirement();
    const auto it = std::find_if(m_previewScenes.begin(), m_previewScenes.end(),
                                 [scene](const auto &item) { return item.get() == scene; });
    if (it == m_previewScenes.end())
        throw std::invalid_argument("Scene is not an open preview Scene");
    auto retiring = std::move(*it);
    m_previewScenes.erase(it);
}

void SceneManager::SetActiveScene(Scene *scene)
{
    const WorldExecutionScope worldScope(scene ? scene->GetRuntimeWorld() : nullptr);
    auto &world = CurrentWorld();
    if (scene && world.scenes.loadedSet.find(scene) == world.scenes.loadedSet.end())
        throw std::invalid_argument("active Scene must be loaded in this SceneManager");
    const bool activeSceneChanged = scene != world.scenes.active;
    if (activeSceneChanged && world.simulation.isPlaying)
        FlushPersistentPromotions();

    world.scenes.active = scene;
    if (world.scenes.active) {
        world.scenes.active->SetRuntimeLifecycleSchedulerEnabled(m_runtimeLifecycleSchedulerEnabled);
        world.scenes.active->SetPlaying(world.simulation.isPlaying);
    }

    // Note: We do NOT auto-assign the editor camera as mainCamera.
    // mainCamera == nullptr means "no game camera assigned" — the Game View
    // will show a placeholder. Scene View always uses the editor camera
    // via SceneRenderBridge / EditorCameraController, independent of mainCamera.
}

float SceneManager::ConsumeFrameDeltaTime(float deltaTime) noexcept
{
    auto &world = CurrentWorld();
    if (!world.simulation.resetDeltaTimeOnNextFrame)
        return deltaTime;
    // A transition inside this Host may spend time loading after the Host's
    // delta was sampled. Discard that interval only for the affected World.
    if (m_runtimeFrameWork && m_runtimeFrameWork->IsActive() &&
        world.simulation.deltaResetHostOpportunity == m_runtimeFrameWork->Opportunity())
        return 0.0f;
    world.simulation.resetDeltaTimeOnNextFrame = false;
    world.simulation.deltaResetHostOpportunity = 0;
    return 0.0f;
}

void SceneManager::StartRuntimeClock()
{
    auto &world = CurrentWorld();
    m_startupTime = std::chrono::steady_clock::now();
    world.simulation.timeScale = 1.0f;
    ResetFrameTime();
}

void SceneManager::ResetFrameTime()
{
    auto &world = CurrentWorld();
    world.frame.pending.clear();
    CloseRuntimeFrame(false);
    world.frame.stepInput.reset();
    world.frame.stepTarget = 0;
    world.simulation.frameTime = {};
    world.simulation.frameTimeSmoother.Reset();
}

void SceneManager::ResetFrameDelta()
{
    auto &world = CurrentWorld();
    world.simulation.resetDeltaTimeOnNextFrame = true;
    world.simulation.deltaResetHostOpportunity =
        m_runtimeFrameWork && m_runtimeFrameWork->IsActive() ? m_runtimeFrameWork->Opportunity() : 0;
    world.simulation.frameTime.deltaTime = 0.0f;
    world.simulation.frameTime.unscaledDeltaTime = 0.0f;
    world.simulation.frameTime.smoothDeltaTime = 0.0f;
    world.simulation.frameTimeSmoother.Reset();
}

void SceneManager::AdvanceFrameTime(float unscaledDeltaTime, float scaledDeltaTime)
{
    auto &world = CurrentWorld();
    world.simulation.frameTime.deltaTime = scaledDeltaTime;
    world.simulation.frameTime.unscaledDeltaTime = unscaledDeltaTime;
    world.simulation.frameTime.smoothDeltaTime = world.simulation.frameTimeSmoother.Advance(scaledDeltaTime);
    world.simulation.frameTime.time += static_cast<double>(scaledDeltaTime);
    world.simulation.frameTime.unscaledTime += static_cast<double>(unscaledDeltaTime);
    ++world.simulation.frameTime.frameCount;
}

void SceneManager::UnloadScene(Scene *scene)
{
    const WorldExecutionScope worldScope(scene ? scene->GetRuntimeWorld() : nullptr);
    auto &world = CurrentWorld();
    Scene::RequireNoWorldRetirement();
    if (!scene || world.scenes.loadedSet.find(scene) == world.scenes.loadedSet.end())
        return;

    if (world.simulation.isPlaying)
        FlushPersistentPromotions();

    const bool wasActive = world.scenes.active == scene;
    Scene *replacement = nullptr;
    if (wasActive) {
        for (const auto &candidate : world.scenes.loaded) {
            if (candidate && candidate.get() != scene) {
                replacement = candidate.get();
                break;
            }
        }
        // Every active-scene transition goes through the same publication
        // path. Never leave registries and lifecycle flags behind by assigning
        // world.scenes.active directly.
        SetActiveScene(replacement);
    }

    if (auto callback = m_onSceneUnloaded) {
        callback(scene);
    }

    world.scenes.loadedSet.erase(scene);

    auto it = std::find_if(world.scenes.loaded.begin(), world.scenes.loaded.end(),
                           [scene](const std::unique_ptr<Scene> &s) { return s.get() == scene; });

    if (it != world.scenes.loaded.end()) {
        auto retiring = std::move(*it);
        world.scenes.loaded.erase(it);
    }
}

bool SceneManager::MoveSceneAdjacent(uint64_t draggedWorldId, uint64_t targetWorldId, bool after)
{
    auto &world = CurrentWorld();
    if (draggedWorldId == 0 || targetWorldId == 0 || draggedWorldId == targetWorldId)
        return false;

    auto draggedIt =
        std::find_if(world.scenes.loaded.begin(), world.scenes.loaded.end(), [draggedWorldId](const auto &candidate) {
            return candidate && candidate->GetWorldId() == draggedWorldId;
        });
    auto targetIt =
        std::find_if(world.scenes.loaded.begin(), world.scenes.loaded.end(), [targetWorldId](const auto &candidate) {
            return candidate && candidate->GetWorldId() == targetWorldId;
        });
    if (draggedIt == world.scenes.loaded.end() || targetIt == world.scenes.loaded.end())
        return false;

    std::unique_ptr<Scene> dragged = std::move(*draggedIt);
    world.scenes.loaded.erase(draggedIt);
    targetIt =
        std::find_if(world.scenes.loaded.begin(), world.scenes.loaded.end(), [targetWorldId](const auto &candidate) {
            return candidate && candidate->GetWorldId() == targetWorldId;
        });
    if (after)
        ++targetIt;
    world.scenes.loaded.insert(targetIt, std::move(dragged));
    return true;
}

void SceneManager::Shutdown()
{
    const WorldExecutionScope worldScope(m_authoringWorld.get());
    auto &world = CurrentWorld();
    Scene::RequireNoWorldRetirement();
    UnbindRuntimeFrameWork();
    auto ids = GetRuntimeWorldIds();
    for (const auto &candidate : m_preparingWorlds)
        ids.push_back(candidate->id);
    for (uint64_t id : ids)
        BeginRuntimeWorldRetirement(id);
    ids.clear();
    for (const auto &candidate : m_retiringWorlds)
        ids.push_back(candidate->id);
    for (uint64_t id : ids) {
        RetireRuntimeWorldScenes(id);
    }
    world.frame.pending.clear();
    CloseRuntimeFrame(false);
    const InputReadScope hostInput({}, 0);
    PhysicsWorld::Instance().BeginRetireDomain(world);

    // No play-mode restore logic here — this is irreversible engine teardown.
    world.simulation.isPlaying = false;
    world.simulation.isPaused = false;
    ResetFrameTime();
    world.simulation.startupPending = false;
    world.simulation.fixedPhase = RuntimeWorld::FixedPhase::Idle;
    world.simulation.pausedStepPending = false;
    world.simulation.isStepping = false;
    world.simulation.stepFrameOpen = false;
    world.simulation.fixedSceneWorlds.clear();
    world.simulation.fixedSceneCursor = 0;

    // Destroy the runtime-only persistent Scene first so its colliders release
    // Jolt bodies while PhysicsWorld is still initialized.
    ClearRuntimePersistentScene();

    // Destroy all scenes (GameObjects → Components → Colliders → bodies).
    UnloadAllScenes();
    if (m_runtimeFrameRetirement)
        m_runtimeFrameRetirement(world.id);
    std::vector<std::unique_ptr<Scene>> retiringPreviews;
    retiringPreviews.swap(m_previewScenes);
    retiringPreviews.clear();
    // Drain cooking and release cached Jolt shapes while the physics factory is alive.
    MeshCollider::ClearCookingCache();
    PhysicsECSStore::Instance().ClearPendingQueues();

#if !defined(INFERNUX_RUNTIME_MINIMAL_HOST)
    // Destroy the editor camera object (its Camera component must leave the
    // component registry before any later teardown step).
    m_editorCamera.SetCamera(nullptr);
    m_editorCameraComponent = nullptr;
    m_editorCameraObject.reset();
#endif

    // Drop callbacks so nothing external fires into a dead engine.
}

void SceneManager::UnloadAllScenes()
{
    const WorldExecutionScope worldScope(&CurrentWorld());
    auto &world = CurrentWorld();
    Scene::RequireNoWorldRetirement();
    if (world.simulation.isPlaying)
        FlushPersistentPromotions();

    SetActiveScene(nullptr);

    for (auto &scene : world.scenes.loaded) {
        if (auto callback = m_onSceneUnloaded) {
            callback(scene.get());
        }
    }

    std::vector<std::unique_ptr<Scene>> retiring;
    retiring.swap(world.scenes.loaded);
    world.scenes.loadedSet.clear();
    retiring.clear();
}

Scene *SceneManager::GetScene(const std::string &name) const
{
    auto &world = CurrentWorld();
    for (const auto &scene : world.scenes.loaded) {
        if (scene->GetName() == name) {
            return scene.get();
        }
    }
    return nullptr;
}

void SceneManager::Start()
{
    const WorldExecutionScope worldScope(&CurrentWorld());
    auto &world = CurrentWorld();
    for (const auto &scene : world.scenes.loaded)
        if (scene)
            scene->Start();
    if (world.scenes.persistent)
        world.scenes.persistent->Start();
}

void SceneManager::SetRuntimeLifecycleCallbacks(RuntimeLifecycleBeginCallback beginFrame,
                                                RuntimeLifecyclePhaseCallback fixedUpdate,
                                                RuntimeLifecyclePhaseCallback update,
                                                RuntimeLifecyclePhaseCallback lateUpdate,
                                                RuntimeLifecyclePhaseCallback editorUpdate,
                                                RuntimeLifecycleEndCallback endFrame)
{
    auto &world = CurrentWorld();
    m_runtimeLifecycleBegin = std::move(beginFrame);
    m_runtimeLifecycleFixedUpdate = std::move(fixedUpdate);
    m_runtimeLifecycleUpdate = std::move(update);
    m_runtimeLifecycleLateUpdate = std::move(lateUpdate);
    m_runtimeLifecycleEditorUpdate = std::move(editorUpdate);
    m_runtimeLifecycleEnd = std::move(endFrame);
    m_runtimeLifecycleSchedulerEnabled =
        static_cast<bool>(m_runtimeLifecycleBegin) && static_cast<bool>(m_runtimeLifecycleFixedUpdate) &&
        static_cast<bool>(m_runtimeLifecycleUpdate) && static_cast<bool>(m_runtimeLifecycleLateUpdate) &&
        static_cast<bool>(m_runtimeLifecycleEditorUpdate) && static_cast<bool>(m_runtimeLifecycleEnd);
    for (const auto &scene : world.scenes.loaded) {
        if (scene)
            scene->SetRuntimeLifecycleSchedulerEnabled(m_runtimeLifecycleSchedulerEnabled);
    }
    if (world.scenes.persistent)
        world.scenes.persistent->SetRuntimeLifecycleSchedulerEnabled(m_runtimeLifecycleSchedulerEnabled);
}

void SceneManager::SetRuntimeLifecycleWorkAvailable(bool available) noexcept
{
    auto &world = CurrentWorld();
    if (world.lifecycle.runtimeLifecycleWorkAvailable == available)
        return;

    world.lifecycle.runtimeLifecycleWorkAvailable = available;
}

void SceneManager::SetRuntimeLifecyclePlan(uint64_t revision, size_t fixedUpdateCount, size_t updateCount,
                                           size_t animationUpdateCount, size_t lateUpdateCount) noexcept
{
    auto &world = CurrentWorld();
    if (revision < world.lifecycle.runtimeLifecyclePlanRevision)
        return;
    world.lifecycle.runtimeLifecyclePlanRevision = revision;
    world.lifecycle.runtimeLifecycleFixedUpdateCount = fixedUpdateCount;
    world.lifecycle.runtimeLifecycleUpdateCount = updateCount;
    world.lifecycle.runtimeLifecycleAnimationUpdateCount = animationUpdateCount;
    world.lifecycle.runtimeLifecycleLateUpdateCount = lateUpdateCount;
}

void SceneManager::SetRuntimeFrameBarrierCallback(RuntimeFrameBarrierCallback callback)
{
    m_runtimeFrameBarrier = std::move(callback);
}

void SceneManager::SetRuntimeResultConsumerCallback(RuntimeResultConsumerCallback callback)
{
    m_runtimeResultConsumer = std::move(callback);
}

bool SceneManager::RuntimeResultsReady(RuntimeFrameBarrier boundary) const
{
    if (auto callback = m_runtimeResultConsumer)
        return callback(boundary);
    return true;
}

void SceneManager::EmitRuntimeFrameBarrier(RuntimeFrameBarrier barrier) const
{
    // Frame barriers are authoritative engine-phase notifications, not an
    // implementation detail of script lifecycle dispatch.  Edit Mode and
    // script-free scenes still cross RenderExtraction/RenderGraph and need
    // render-facing services (component gizmos, compute pose bindings, UI)
    // to publish against those exact phases.  The Python scheduler already
    // ignores lifecycle consumption when no native execution frame is open.
    if (auto callback = m_runtimeFrameBarrier)
        callback(barrier);
}

void SceneManager::ClearRuntimeLifecycleCallbacks()
{
    const auto clear = [this](RuntimeWorld &world) {
        const WorldExecutionScope scope(&world);
        if (world.lifecycle.runtimeLifecycleFrameOpen) {
            world.lifecycle.runtimeLifecycleFrameOpen = false;
            if (auto callback = m_runtimeLifecycleEnd)
                callback();
        }
        world.lifecycle.runtimeLifecycleWorkAvailable = false;
        world.lifecycle.runtimeResultConsumersPresent = false;
        world.lifecycle.runtimeLifecycleFrameOpen = false;
        world.lifecycle.runtimeLifecyclePlanRevision = 0;
        world.lifecycle.runtimeLifecycleFixedUpdateCount = 0;
        world.lifecycle.runtimeLifecycleUpdateCount = 0;
        world.lifecycle.runtimeLifecycleAnimationUpdateCount = 0;
        world.lifecycle.runtimeLifecycleLateUpdateCount = 0;
        for (const auto &scene : world.scenes.loaded) {
            if (scene)
                scene->SetRuntimeLifecycleSchedulerEnabled(false);
        }
        if (world.scenes.persistent)
            world.scenes.persistent->SetRuntimeLifecycleSchedulerEnabled(false);
    };
    if (m_authoringWorld)
        clear(*m_authoringWorld);
    for (const auto *worlds : {&m_runtimeWorlds, &m_preparingWorlds, &m_retiringWorlds})
        for (const auto &world : *worlds)
            clear(*world);
    m_runtimeInput = nullptr;
    m_runtimeLifecycleBegin = nullptr;
    m_runtimeLifecycleFixedUpdate = nullptr;
    m_runtimeLifecycleUpdate = nullptr;
    m_runtimeLifecycleLateUpdate = nullptr;
    m_runtimeLifecycleEditorUpdate = nullptr;
    m_runtimeLifecycleEnd = nullptr;
    m_runtimeFrameBarrier = nullptr;
    m_runtimeResultConsumer = nullptr;
    m_runtimeLifecycleSchedulerEnabled = false;
}

SceneManager::SimulationState SceneManager::GetSimulationState() const noexcept
{
    const auto &world = CurrentWorld();
    return {world.simulation.simulationEpoch, world.simulation.isPlaying, world.simulation.isPaused,
            world.frame.generation, !m_runtimeFrameWork || m_runtimeFrameWork->IsAccepting()};
}

bool SceneManager::CanContinueRuntimeFrame() const noexcept
{
    return !m_runtimeFrameWork || (m_runtimeFrameWork->IsAccepting() &&
                                   (!CurrentWorld().simulation.isPlaying || !m_runtimeFrameWork->IsActive() ||
                                    m_runtimeFrameWork->ServiceRemaining() > 0));
}

void SceneManager::ReportRuntimeFrameProgress() noexcept
{
    if (m_runtimeFrameWork && m_runtimeFrameWork->IsActive())
        m_runtimeFrameWork->ReportProgress();
}

void SceneManager::BindRuntimeFrameWork(FrameWorkOwner &work)
{
    if (m_runtimeFrameWork == &work && m_runtimeFrameSource)
        return;
    UnbindRuntimeFrameWork();
    m_runtimeFrameWork = &work;
    FrameWorkSource source;
    source.describe = [this] { return DescribeRuntimeFrames(); };
    source.advance = [this](bool gpu, uint64_t worldId) { return AdvanceRuntimeFrame(gpu, worldId); };
    m_runtimeFrameSource = work.AddSource(std::move(source));
}

void SceneManager::UnbindRuntimeFrameWork() noexcept
{
    if (m_runtimeFrameWork && m_runtimeFrameSource)
        m_runtimeFrameWork->RemoveSource(m_runtimeFrameSource);
    m_runtimeFrameSource = 0;
    m_runtimeFrameWork = nullptr;
}

FrameWorkFacts SceneManager::DescribeRuntimeFrames()
{
    FrameWorkFacts facts;
    auto ids = GetRuntimeWorldIds();
    const auto first = std::find(ids.begin(), ids.end(), m_nextRuntimeWorldId);
    if (first != ids.end())
        std::rotate(ids.begin(), first, ids.end());
    for (uint64_t id : ids) {
        const auto *world = FindRuntimeWorld(id);
        if (!world || !world->simulation.isPlaying)
            continue;
        const bool step = world->simulation.isPaused && world->simulation.pausedStepPending;
        const bool frame = !world->simulation.isPaused &&
                           (world->frame.phase != RuntimeWorld::FramePhase::Idle || !world->frame.pending.empty());
        if (!step && !frame)
            continue;
        facts.pending = true;
        facts.host = {id, 0, m_runtimeFrameWork && m_runtimeFrameWork->IsLate(), false};
        break;
    }
    return facts;
}

bool SceneManager::AdvanceRuntimeFrame(bool gpu, uint64_t worldId)
{
    auto *work = m_runtimeFrameWork;
    if (gpu || !work || !work->IsActive() || !work->IsLate() || !work->IsAccepting() || work->ServiceRemaining() <= 0)
        return false;
    // A blocked World must not hide ready peers behind this aggregate source.
    // Probe this stable published-ID snapshot at most once; return immediately
    // after the first real quantum, never after cursor rotation alone.
    auto ids = GetRuntimeWorldIds();
    auto first = std::find(ids.begin(), ids.end(), worldId);
    if (first == ids.end())
        first = std::find(ids.begin(), ids.end(), m_nextRuntimeWorldId);
    if (first != ids.end())
        std::rotate(ids.begin(), first, ids.end());
    for (auto selected = ids.begin(); selected != ids.end(); ++selected) {
        if (m_runtimeFrameWork != work || !work->IsActive() || !work->IsLate() || !work->IsAccepting() ||
            work->ServiceRemaining() <= 0)
            return false;
        const auto live = std::find_if(m_runtimeWorlds.begin(), m_runtimeWorlds.end(),
                                       [id = *selected](const auto &candidate) { return candidate->id == id; });
        if (live == m_runtimeWorlds.end())
            continue;
        auto *world = live->get();
        if (!world->simulation.isPlaying)
            continue;
        const bool step = world->simulation.isPaused && world->simulation.pausedStepPending;
        const bool frame = !world->simulation.isPaused &&
                           (world->frame.phase != RuntimeWorld::FramePhase::Idle || !world->frame.pending.empty());
        if (!step && !frame)
            continue;
        const auto next = std::next(selected);
        m_nextRuntimeWorldId = next == ids.end() ? ids.front() : *next;
        const WorldExecutionScope scope(world);
        const auto cursor = [&] {
            return std::make_tuple(world->frame.phase, world->frame.generation, world->frame.sceneCursor,
                                   world->frame.pending.size(), world->simulation.fixedPhase,
                                   world->simulation.fixedStepIndex, world->simulation.completedFixedStepIndex,
                                   world->simulation.publishedFixedStepIndex, world->simulation.runtimeFrameCount,
                                   world->simulation.pausedStepPending, world->physics->fixedGpuPhase,
                                   world->physics->controlPending, world->physics->stepPending,
                                   world->physics->stepCompleted, world->physics->solverStarted);
        };
        const auto before = cursor();
        const auto progress = work->ProgressSerial();
        const bool published = ServiceCurrentWorldFrame();
        const bool accepted = published || work->ProgressSerial() != progress;
        // Callbacks may Stop/retire this owner or detach the source. Do not
        // inspect the old World cursor after it leaves the published collection.
        const bool stillPublished = std::any_of(m_runtimeWorlds.begin(), m_runtimeWorlds.end(),
                                                [world](const auto &candidate) { return candidate.get() == world; });
        if (!stillPublished || m_runtimeFrameWork != work || !work->IsAccepting())
            return accepted || !stillPublished;
        if (accepted || cursor() != before)
            return true;
        // No retained change: try the next World while the same credit remains.
        // Every pointer/cursor reference above expires with this real scope.
    }
    return false;
}

void SceneManager::SetRuntimeInputCallback(std::function<void()> callback)
{
    m_runtimeInput = std::move(callback);
}

void SceneManager::SetRuntimeFramePublicationCallbacks(std::function<bool(RuntimeWorld &)> publish,
                                                       std::function<void(uint64_t)> retire)
{
    m_runtimeFramePublication = std::move(publish);
    m_runtimeFrameRetirement = std::move(retire);
}

bool SceneManager::TryPublishRuntimeFrame()
{
    auto &world = CurrentWorld();
    if (world.frame.phase != RuntimeWorld::FramePhase::Publish || !CanContinueRuntimeFrame())
        return false;
    const auto admittedState = GetSimulationState();
    const auto publish = m_runtimeFramePublication;
    if (publish && !publish(world))
        return false;
    if (GetSimulationState() != admittedState)
        return false;
    CloseRuntimeFrame(true);
    if (publish)
        EmitRuntimeFrameBarrier(RuntimeFrameBarrier::SnapshotPublication);
    return GetSimulationState() == admittedState;
}

void SceneManager::FinishStepPresentation(bool completeStep) noexcept
{
    if (completeStep && !CurrentWorld().simulation.pausedStepPending &&
        CurrentWorld().frame.phase == RuntimeWorld::FramePhase::Idle)
        CurrentWorld().simulation.stepFrameOpen = false;
}

void SceneManager::RecordHostFrame(float deltaTime)
{
    auto &world = CurrentWorld();
    if (!world.simulation.isPlaying || world.simulation.isPaused ||
        (world.scenes.loaded.empty() && !world.scenes.persistent))
        return;
    auto input = InputManager::Instance().CaptureFrame();
    // Repeated zero-time service of one input sample creates no new logical demand.
    const uint64_t lastSource = !world.frame.pending.empty() ? world.frame.pending.back().input->sampleIdentity
                                : world.frame.input          ? world.frame.input->sourceSampleIdentity
                                                             : 0;
    if (deltaTime == 0.0f && input->sampleIdentity == lastSource)
        return;
    world.frame.pending.push_back(
        {deltaTime, std::clamp(deltaTime * world.simulation.timeScale, 0.0f, world.simulation.maxFixedDeltaTime),
         std::move(input)});
}

void SceneManager::AdmitRuntimeFrame(RuntimeWorld::FrameAdmission admission, bool stepping)
{
    auto &world = CurrentWorld();
    ++world.frame.generation;
    world.frame.phase = RuntimeWorld::FramePhase::Input;
    world.frame.phaseScenes.clear();
    world.frame.sceneCursor = 0;
    world.frame.updateStarted = false;
    world.frame.animationStarted = false;
    world.frame.lateStarted = false;
    world.frame.hadRuntimeLifecycleScope = false;
    world.frame.debitFixedAccumulator = !stepping;
    AdvanceFrameTime(admission.unscaledDeltaTime, admission.deltaTime);
    world.frame.input = InputManager::Instance().BeginWorldFrameInput(world.id, std::move(admission.input),
                                                                      world.simulation.frameTime.frameCount);
    if (stepping) {
        world.frame.fixedTarget = world.simulation.fixedStepIndex + static_cast<uint64_t>(!HasPendingFixedStep());
    } else {
        world.simulation.fixedTimeAccumulator += admission.deltaTime;
        const auto due = static_cast<uint64_t>(world.simulation.fixedTimeAccumulator / world.simulation.fixedTimeStep);
        world.frame.fixedTarget = world.simulation.fixedStepIndex + due;
    }
}

void SceneManager::BeginRuntimePhaseScenes()
{
    auto &world = CurrentWorld();
    world.frame.phaseScenes.clear();
    for (const auto &scene : world.scenes.loaded)
        if (scene)
            world.frame.phaseScenes.push_back(scene->GetWorldId());
    if (world.scenes.persistent)
        world.frame.phaseScenes.push_back(world.scenes.persistent->GetWorldId());
    world.frame.sceneCursor = 0;
}

void SceneManager::BeginRuntimeLifecycleFrame(bool useScheduler)
{
    auto &world = CurrentWorld();
    useScheduler = useScheduler || (m_runtimeLifecycleSchedulerEnabled && world.frame.hadRuntimeLifecycleScope);
    if (useScheduler && !world.lifecycle.runtimeLifecycleFrameOpen) {
        world.frame.hadRuntimeLifecycleScope = true;
        world.lifecycle.runtimeLifecycleFrameOpen = true;
        if (auto callback = m_runtimeLifecycleBegin)
            callback();
    }
}

void SceneManager::CloseRuntimeFrame(bool completed)
{
    auto &world = CurrentWorld();
    if (completed)
        ++world.simulation.runtimeFrameCount;
    else
        ++world.frame.generation;
    auto &frame = world.frame;
    frame.phase = RuntimeWorld::FramePhase::Idle;
    frame.input.reset();
    frame.phaseScenes.clear();
    frame.sceneCursor = 0;
    frame.hadRuntimeLifecycleScope = false;
}

void SceneManager::Update(float deltaTime)
{
    UpdateHostFrame(deltaTime);
    UpdateCurrentWorld(deltaTime);
}

void SceneManager::UpdateHostFrame(float deltaTime)
{
    const InputReadScope hostInput({}, 0);
    m_lastFrameProfile = {};

#if !defined(INFERNUX_RUNTIME_MINIMAL_HOST)
    auto t0 = ProfileClock::now();
    UpdateCapturedEditorCamera(m_editorCamera, deltaTime);
    m_lastFrameProfile.editorCameraMs += ProfileMsSince(t0);
#else
    auto t0 = ProfileClock::now();
#endif
}

void SceneManager::UpdateCurrentWorld(float deltaTime)
{
    const WorldExecutionScope worldScope(&CurrentWorld());
    auto &world = CurrentWorld();
    const InputReadScope hostInput({}, 0);
    const auto admittedState = GetSimulationState();
    auto t0 = ProfileClock::now();
    if (world.simulation.pausedStepPending && world.simulation.isPlaying && world.simulation.isPaused) {
        Step(world.simulation.pausedStepDeltaTime);
        return;
    }

    if (!world.simulation.isPlaying && !world.scenes.loaded.empty()) {
        const bool useRuntimeScheduler =
            m_runtimeLifecycleSchedulerEnabled && world.lifecycle.runtimeLifecycleWorkAvailable;
        if (useRuntimeScheduler) {
            world.lifecycle.runtimeLifecycleFrameOpen = true;
            if (auto callback = m_runtimeLifecycleBegin)
                callback();
        }

        t0 = ProfileClock::now();
        for (const auto &scene : world.scenes.loaded) {
            if (GetSimulationState() != admittedState)
                return;
            if (scene)
                scene->EditorUpdate(deltaTime);
        }
        m_lastFrameProfile.editorUpdateMs += ProfileMsSince(t0);

        if (useRuntimeScheduler && (world.lifecycle.runtimeLifecycleUpdateCount > 0 ||
                                    world.lifecycle.runtimeLifecycleAnimationUpdateCount > 0 ||
                                    world.lifecycle.runtimeLifecycleLateUpdateCount > 0)) {
            if (auto callback = m_runtimeLifecycleEditorUpdate)
                callback(deltaTime);
        }
        if (GetSimulationState() != admittedState)
            return;
    }

    RecordHostFrame(deltaTime);
    ContinueRuntimeUpdate();
}

void SceneManager::ContinueRuntimeUpdate()
{
    auto &world = CurrentWorld();
    if (!world.simulation.isPlaying || (world.simulation.isPaused && !world.simulation.isStepping) ||
        !CanContinueRuntimeFrame())
        return;
    if (world.frame.phase == RuntimeWorld::FramePhase::Idle) {
        if (world.frame.pending.empty() || !CanContinueRuntimeFrame())
            return;
        auto admission = std::move(world.frame.pending.front());
        world.frame.pending.pop_front();
        AdmitRuntimeFrame(std::move(admission), false);
    }
    if (world.frame.phase == RuntimeWorld::FramePhase::Ready ||
        world.frame.phase == RuntimeWorld::FramePhase::Complete ||
        world.frame.phase == RuntimeWorld::FramePhase::Publish || world.frame.phase == RuntimeWorld::FramePhase::Late)
        return;
    const InputReadScope inputScope(world.frame.input, world.id);
    const auto admittedState = GetSimulationState();
    if (world.frame.phase == RuntimeWorld::FramePhase::Input) {
        world.frame.phase = RuntimeWorld::FramePhase::Fixed;
        if (auto callback = m_runtimeInput)
            callback();
        if (GetSimulationState() != admittedState || !CanContinueRuntimeFrame())
            return;
    }
    for (const auto &scene : world.scenes.loaded) {
        if (!CanContinueRuntimeFrame())
            return;
        if (scene)
            scene->ProcessPendingStarts();
        if (GetSimulationState() != admittedState)
            return;
    }
    if (world.scenes.persistent)
        world.scenes.persistent->ProcessPendingStarts();
    if (GetSimulationState() != admittedState || !CanContinueRuntimeFrame())
        return;
    const bool useRuntimeScheduler =
        m_runtimeLifecycleSchedulerEnabled && (world.lifecycle.runtimeLifecycleWorkAvailable || HasPendingFixedStep() ||
                                               world.lifecycle.runtimeResultConsumersPresent);
    BeginRuntimeLifecycleFrame(useRuntimeScheduler);
    if (GetSimulationState() != admittedState)
        return;
    if (world.frame.phase == RuntimeWorld::FramePhase::Fixed) {
        while (HasPendingFixedStep() || (world.simulation.fixedStepIndex < world.frame.fixedTarget &&
                                         (!world.frame.debitFixedAccumulator ||
                                          world.simulation.fixedTimeAccumulator >= world.simulation.fixedTimeStep))) {
            if (!CanContinueRuntimeFrame() || !RunFixedSimulationStep(useRuntimeScheduler, admittedState))
                return;
        }
        world.frame.phase = RuntimeWorld::FramePhase::Update;
        BeginRuntimePhaseScenes();
        EmitRuntimeFrameBarrier(RuntimeFrameBarrier::TransformResolve);
        if (GetSimulationState() != admittedState)
            return;
    }
    if (world.frame.phase != RuntimeWorld::FramePhase::Update || !CanContinueRuntimeFrame())
        return;
    const auto phaseStart = ProfileClock::now();
    if (!world.frame.updateStarted) {
        world.frame.updateStarted = true;
        if (useRuntimeScheduler && world.lifecycle.runtimeLifecycleUpdateCount > 0)
            if (auto callback = m_runtimeLifecycleUpdate)
                callback(world.simulation.frameTime.deltaTime);
    }
    if (GetSimulationState() != admittedState || !RuntimeResultsReady(RuntimeFrameBarrier::UpdateScript))
        return;
    while (world.frame.sceneCursor < world.frame.phaseScenes.size()) {
        if (!CanContinueRuntimeFrame())
            return;
        if (auto *scene = GetSceneByWorldId(world.frame.phaseScenes[world.frame.sceneCursor]))
            scene->Update(world.simulation.frameTime.deltaTime);
        if (GetSimulationState() != admittedState || !CanContinueRuntimeFrame())
            return;
        ++world.frame.sceneCursor;
    }
    if (!TransformECSStore::Instance().IsFrameCacheActive())
        FlushPersistentPromotions();
    m_lastFrameProfile.gameplayUpdateMs += ProfileMsSince(phaseStart);
    world.frame.phase = RuntimeWorld::FramePhase::Late;
    BeginRuntimePhaseScenes();
}

void SceneManager::EnsurePhysicsQueriesCurrent()
{
    auto &world = CurrentWorld();
    if (HasPendingFixedStep() && !world.simulation.fixedExecuting) {
        PhysicsWorld::Instance().CompletePendingStep();
    }
    if (world.scenes.loaded.empty() && !world.scenes.persistent)
        return;
    FlushPendingBroadphase();
    // Outside play mode nothing steps the simulation, so moved bodies must be
    // teleported (dt = 0): the kinematic-velocity paths would otherwise leave
    // bodies with a velocity that is never integrated nor settled.
    SyncCollidersToPhysics(world.simulation.isPlaying ? GetFixedTimeStep() : 0.0f);
}

void SceneManager::ContinueRuntimeLateUpdate()
{
    auto &world = CurrentWorld();
    if (!world.simulation.isPlaying || (world.simulation.isPaused && !world.simulation.isStepping) ||
        world.frame.phase != RuntimeWorld::FramePhase::Late || HasPendingFixedStep() || !CanContinueRuntimeFrame())
        return;
    const InputReadScope inputScope(world.frame.input, world.id);
    const auto admittedState = GetSimulationState();
    for (const auto &scene : world.scenes.loaded) {
        if (!CanContinueRuntimeFrame())
            return;
        if (scene)
            scene->ProcessPendingStarts();
        if (GetSimulationState() != admittedState)
            return;
    }
    if (world.scenes.persistent)
        world.scenes.persistent->ProcessPendingStarts();
    if (GetSimulationState() != admittedState || !CanContinueRuntimeFrame())
        return;
    const bool useRuntimeScheduler =
        m_runtimeLifecycleSchedulerEnabled &&
        (world.lifecycle.runtimeLifecycleWorkAvailable || world.lifecycle.runtimeResultConsumersPresent);
    BeginRuntimeLifecycleFrame(useRuntimeScheduler);
    if (GetSimulationState() != admittedState)
        return;
    const auto phaseStart = ProfileClock::now();
    if (!world.frame.animationStarted) {
        world.frame.animationStarted = true;
        EmitRuntimeFrameBarrier(RuntimeFrameBarrier::AnimationEvaluation);
    }
    if (GetSimulationState() != admittedState || !RuntimeResultsReady(RuntimeFrameBarrier::AnimationEvaluation))
        return;
    if (!world.frame.lateStarted) {
        world.frame.lateStarted = true;
        if (useRuntimeScheduler && world.lifecycle.runtimeLifecycleLateUpdateCount > 0)
            if (auto callback = m_runtimeLifecycleLateUpdate)
                callback(world.simulation.frameTime.deltaTime);
    }
    if (GetSimulationState() != admittedState || !RuntimeResultsReady(RuntimeFrameBarrier::LateScript))
        return;
    while (world.frame.sceneCursor < world.frame.phaseScenes.size()) {
        if (!CanContinueRuntimeFrame())
            return;
        if (auto *scene = GetSceneByWorldId(world.frame.phaseScenes[world.frame.sceneCursor]))
            scene->LateUpdate(world.simulation.frameTime.deltaTime);
        if (GetSimulationState() != admittedState || !CanContinueRuntimeFrame())
            return;
        ++world.frame.sceneCursor;
    }
    if (!TransformECSStore::Instance().IsFrameCacheActive())
        FlushPersistentPromotions();
    m_lastFrameProfile.lateUpdateMs += ProfileMsSince(phaseStart);
    world.frame.phase = RuntimeWorld::FramePhase::Ready;
}

void SceneManager::LateUpdate(float deltaTime)
{
    const WorldExecutionScope worldScope(&CurrentWorld());
    ContinueRuntimeLateUpdate();
    if (!HasPendingFixedStep())
        PublishSkeletonPoses();
#if !defined(INFERNUX_RUNTIME_MINIMAL_HOST)
    const auto start = ProfileClock::now();
    AudioEngine::Instance().Update(deltaTime);
    m_lastFrameProfile.audioMs += ProfileMsSince(start);
#else
    (void)deltaTime;
#endif
}

void SceneManager::FinalizeRuntimeTransforms(bool completed)
{
    auto &world = CurrentWorld();
    const auto admittedState = GetSimulationState();
    if (completed)
        PublishSkeletonPoses();
    if (TransformECSStore::Instance().IsFrameCacheActive()) {
        if (TransformECSStore::Instance().EndFrameCache())
            PublishPhysicsTransformsToRenderer();
    } else if (completed) {
        FlushPersistentPromotions();
        for (const auto &scene : world.scenes.loaded)
            if (scene)
                TransformECSStore::Instance().SyncSceneWorldMatrices(scene.get());
        if (world.scenes.persistent)
            TransformECSStore::Instance().SyncSceneWorldMatrices(world.scenes.persistent.get());
    }
    PublishAuthoredTransformsToPhysics();
    if (completed || !world.simulation.isPlaying) {
        EmitRuntimeFrameBarrier(RuntimeFrameBarrier::FinalTransformResolve);
        if (GetSimulationState() == admittedState)
            EmitRuntimeFrameBarrier(RuntimeFrameBarrier::AnimationTimeline);
    }
}

bool SceneManager::ServiceCurrentWorldFrame()
{
    auto &world = CurrentWorld();
    TransformECSStore::Instance().BeginFrameCache();
    if (world.simulation.pausedStepPending && world.simulation.isPaused) {
        const auto completed = world.simulation.runtimeFrameCount;
        Step(world.simulation.pausedStepDeltaTime);
        if (&world != GetPrimaryRuntimeWorld())
            FinishStepPresentation(true);
        return world.simulation.runtimeFrameCount != completed;
    }
    ContinueRuntimeUpdate();
    ContinueRuntimeLateUpdate();
    const bool published = EndRuntimeFrame(false);
    if (&world != GetPrimaryRuntimeWorld())
        FinishStepPresentation(true);
    return published;
}

void SceneManager::ServiceRuntimeFrames(FrameWorkOwner &work, float hostDeltaTime, bool explicitDelta)
{
    BindRuntimeFrameWork(work);
    if (!work.IsAccepting())
        return;
    FrameWorkOwner::Scope accounting(work);
    CollectRetiredRuntimeWorlds();
    if (!work.IsAccepting())
        return;
    UpdateHostFrame(hostDeltaTime);
    const auto consumeDelta = [this, hostDeltaTime, explicitDelta] {
        if (!explicitDelta)
            return ConsumeFrameDeltaTime(hostDeltaTime);
        auto &simulation = CurrentWorld().simulation;
        simulation.resetDeltaTimeOnNextFrame = false;
        simulation.deltaResetHostOpportunity = 0;
        return hostDeltaTime;
    };
#if !defined(INFERNUX_RUNTIME_MINIMAL_HOST)
    const auto audioStart = ProfileClock::now();
    AudioEngine::Instance().Update(hostDeltaTime);
    m_lastFrameProfile.audioMs += ProfileMsSince(audioStart);
#endif
    if (m_runtimeWorlds.empty()) {
        if (m_activeManagementSession)
            return; // An empty Play collection still owns its live preparation source.
        const WorldExecutionScope authorScope(m_authoringWorld.get());
        TransformECSStore::Instance().BeginFrameCache();
        UpdateCurrentWorld(consumeDelta());
        EndRuntimeFrame(false);
        return;
    }
    // Admission stays at the original Host/input boundary. Actual runtime work
    // competes with preparation/physics through the same late owner chooser.
    for (uint64_t id : GetRuntimeWorldIds()) {
        const WorldExecutionScope scope(FindRuntimeWorld(id));
        RecordHostFrame(consumeDelta());
    }
    work.Wake(m_runtimeFrameSource);
}

void SceneManager::CompleteRuntimeFrame()
{
    auto &world = CurrentWorld();
    const auto admittedState = GetSimulationState();
    if (world.frame.phase == RuntimeWorld::FramePhase::Ready ||
        world.frame.phase == RuntimeWorld::FramePhase::Complete ||
        world.frame.phase == RuntimeWorld::FramePhase::Publish)
        BeginRuntimeLifecycleFrame(m_runtimeLifecycleSchedulerEnabled &&
                                   (world.lifecycle.runtimeLifecycleWorkAvailable ||
                                    world.lifecycle.runtimeResultConsumersPresent || HasPendingFixedStep()));
    if (GetSimulationState() != admittedState)
        return;
    if (world.frame.phase == RuntimeWorld::FramePhase::Ready && CanContinueRuntimeFrame() &&
        RuntimeResultsReady(RuntimeFrameBarrier::RenderExtraction) && GetSimulationState() == admittedState)
        world.frame.phase = RuntimeWorld::FramePhase::Complete;
}

void SceneManager::EndFrame(bool completeStep)
{
    EndRuntimeFrame(completeStep);
}

bool SceneManager::EndRuntimeFrame(bool completeStep)
{
    const WorldExecutionScope worldScope(&CurrentWorld());
    auto &world = CurrentWorld();
    const InputReadScope inputScope(world.frame.input, world.id);
    const auto admittedState = GetSimulationState();
    // Publication and every required consumer run inside this Host's native
    // lifecycle scope. Even a pending publication closes that scope below.
    struct LifecycleScopeEnd
    {
        RuntimeWorld &world;
        std::function<void()> callback;
        ~LifecycleScopeEnd()
        {
            // Exit or a scene replacement can return before normal publication.
            // Commit only the CPU author cache; do not start another GPU consumer.
            if (TransformECSStore::Instance().IsFrameCacheActive())
                TransformECSStore::Instance().EndFrameCache();
            if (world.lifecycle.runtimeLifecycleFrameOpen) {
                world.lifecycle.runtimeLifecycleFrameOpen = false;
                if (callback)
                    callback();
            }
        }
    } lifecycleEnd{world, m_runtimeLifecycleEnd};
    if (m_runtimeFrameWork && !m_runtimeFrameWork->IsAccepting())
        return false;
    CompleteRuntimeFrame();
    if (GetSimulationState() != admittedState)
        return false;
    const bool completed = world.frame.phase == RuntimeWorld::FramePhase::Complete;
    if (completed)
        world.frame.phase = RuntimeWorld::FramePhase::Publish;
    if (world.frame.phase != RuntimeWorld::FramePhase::Idle || !world.simulation.isPlaying)
        FinalizeRuntimeTransforms(completed);
    if (GetSimulationState() != admittedState)
        return false;
    EmitRuntimeFrameBarrier(RuntimeFrameBarrier::PendingDestroy);
    if (GetSimulationState() != admittedState)
        return false;
    if (!world.scenes.loaded.empty() || world.scenes.persistent) {
        const auto t0 = ProfileClock::now();
        // Callbacks can synchronously replace the scene graph.
        std::vector<uint64_t> scenes;
        for (const auto &scene : world.scenes.loaded)
            if (scene)
                scenes.push_back(scene->GetWorldId());
        if (world.scenes.persistent)
            scenes.push_back(world.scenes.persistent->GetWorldId());
        for (uint64_t id : scenes) {
            if (auto *scene = GetSceneByWorldId(id))
                scene->ProcessPendingDestroys();
            if (GetSimulationState() != admittedState)
                return false;
        }
        FlushPersistentPromotions();
        m_lastFrameProfile.endFrameMs += ProfileMsSince(t0);
    }
    const bool published = TryPublishRuntimeFrame();
    FinishStepPresentation(completeStep);
    return published;
}

void SceneManager::Play()
{
    if (m_runtimeWorlds.empty())
        throw std::logic_error("Play requires a prepared runtime scene graph");
    const auto *target = WorldExecutionScope::Current();
    const auto ids =
        target && target != m_authoringWorld.get() ? std::vector<uint64_t>{target->id} : GetRuntimeWorldIds();
    for (uint64_t id : ids) {
        auto *world = FindRuntimeWorld(id);
        const WorldExecutionScope scope(world);
        PlayCurrentWorld();
    }
    if (auto callback = m_onPlayStateChanged)
        callback(std::any_of(m_runtimeWorlds.begin(), m_runtimeWorlds.end(),
                             [](const auto &world) { return world->simulation.isPlaying; }));
}

void SceneManager::PlayCurrentWorld()
{
    const WorldExecutionScope worldScope(&CurrentWorld());
    auto &world = CurrentWorld();
    if (world.simulation.fixedPhase == RuntimeWorld::FixedPhase::Failed)
        throw std::runtime_error("the physical step failed; stop Play before restarting the simulation");
    Scene::RequireNoWorldRetirement();
    // Only reset accumulator on initial play, not on resume-from-pause.
    const bool fresh = !world.simulation.isPlaying;
    if (fresh) {
        if (m_runtimeFrameRetirement)
            m_runtimeFrameRetirement(world.id);
        world.simulation.fixedTimeAccumulator = 0.0f;
        world.simulation.fixedTime = 0.0;
        world.simulation.fixedUnscaledTime = 0.0;
        ResetFrameTime();
        world.simulation.runtimeFrameCount = 0;
    }

    world.simulation.isPlaying = true;
    world.simulation.isPaused = false;
    world.simulation.pausedStepPending = false;
    world.frame.stepInput.reset();
    world.frame.stepTarget = 0;
    world.simulation.stepFrameOpen = false;
    if (fresh) {
        ++world.simulation.simulationEpoch;
        world.simulation.startupPending = true;
        world.simulation.fixedStepIndex = 0;
        world.simulation.completedFixedStepIndex = 0;
        world.simulation.publishedFixedStepIndex = 0;
        world.simulation.fixedPhase = RuntimeWorld::FixedPhase::Idle;
    }
#if !defined(INFERNUX_RUNTIME_MINIMAL_HOST)
    AudioEngine::Instance().RefreshWorldState(world);
#endif

    // Notify renderer to exit idle mode immediately.

    ResetFrameDelta();
    if (world.simulation.startupPending)
        StartActiveSceneForPlay();
}

void SceneManager::StartActiveSceneForPlay()
{
    const WorldExecutionScope worldScope(&CurrentWorld());
    auto &world = CurrentWorld();
    if (world.scenes.loaded.empty() && !world.scenes.persistent) {
        world.simulation.startupPending = false;
        return;
    }
    const auto admittedState = GetSimulationState();
    std::vector<Scene *> initialScenes;
    for (const auto &scene : world.scenes.loaded)
        initialScenes.push_back(scene.get());
    if (world.scenes.persistent)
        initialScenes.push_back(world.scenes.persistent.get());

    // Scene-relative frame numbering is the useful contract for deterministic
    // capture and diagnostics. A transactional scene replacement therefore
    // starts a fresh sequence even though the play session itself continues.
    world.simulation.runtimeFrameCount = 0;

    const auto transitionStart = ProfileClock::now();
    for (auto *scene : initialScenes)
        if (scene)
            scene->SetPlaying(true);

    // A transactional runtime load publishes freshly-deserialized Transform
    // rows immediately before this call. Their world caches can still contain
    // values from recycled ECS slots. Jolt shapes consume world scale during
    // body creation, so synchronize the graph before Collider::RegisterBody.
    auto phaseStart = ProfileClock::now();
    for (auto *scene : initialScenes)
        if (scene)
            TransformECSStore::Instance().SyncSceneWorldMatrices(scene);
    const double initialTransformMs = ProfileMsSince(phaseStart);

    phaseStart = ProfileClock::now();
    for (auto *scene : initialScenes) {
        if (GetSimulationState() != admittedState)
            return;
        if (scene)
            scene->Start();
    }
    if (GetSimulationState() != admittedState)
        return;
    FlushPersistentPromotions();
    const double startMs = ProfileMsSince(phaseStart);

    // Start callbacks may author transforms. Publish those edits before shape
    // creation as well, matching the transform state visible to gameplay.
    phaseStart = ProfileClock::now();
    for (auto *scene : initialScenes)
        if (scene)
            TransformECSStore::Instance().SyncSceneWorldMatrices(scene);
    const double postStartTransformMs = ProfileMsSince(phaseStart);

    // Start callbacks can author transforms. The transform observer already
    // records exactly which physics actors changed, so avoid rewriting every
    // resident body when entering Play Mode.
    phaseStart = ProfileClock::now();
    SyncCollidersToPhysics(world.simulation.fixedTimeStep);
    const double colliderSyncMs = ProfileMsSince(phaseStart);

    phaseStart = ProfileClock::now();
    FlushPendingBroadphase();
    // Play is a cold publication boundary. Finish the shapes requested above
    // before any link in a connected body assembly can integrate on its own.
    MeshCollider::FlushCompletedCooking(true);
    FlushPendingBroadphase();
    const double flushMs = ProfileMsSince(phaseStart);

    world.simulation.startupPending = false;

    const double totalMs = ProfileMsSince(transitionStart);
    if (totalMs > 25.0) {
        // INXLOG_INFO("[Perf] StartActiveSceneForPlay: total=", totalMs, "ms transform=", initialTransformMs,
        //             "ms lifecycle=", startMs, "ms postTransform=", postStartTransformMs,
        //             "ms colliderSync=", colliderSyncMs, "ms broadphase=", flushMs,
        //             "ms");
    }
}

void SceneManager::StartSceneForPlay(Scene *scene)
{
    const WorldExecutionScope worldScope(scene ? scene->GetRuntimeWorld() : nullptr);
    auto &world = CurrentWorld();
    if (!scene || world.scenes.loadedSet.find(scene) == world.scenes.loadedSet.end())
        throw std::invalid_argument("additive Scene must be loaded before entering Play");
    if (!world.simulation.isPlaying)
        throw std::logic_error("additive Scene can enter Play only during a play session");

    scene->SetRuntimeLifecycleSchedulerEnabled(m_runtimeLifecycleSchedulerEnabled);
    scene->SetPlaying(true);
    TransformECSStore::Instance().SyncSceneWorldMatrices(scene);
    const auto admittedState = GetSimulationState();
    scene->Start();
    if (GetSimulationState() != admittedState)
        return;
    TransformECSStore::Instance().SyncSceneWorldMatrices(scene);
    SyncCollidersToPhysics(world.simulation.fixedTimeStep);
    FlushPendingBroadphase();
    MeshCollider::FlushCompletedCooking(true);
    FlushPendingBroadphase();
}

void SceneManager::Stop()
{
    const bool wasPlaying = std::any_of(m_runtimeWorlds.begin(), m_runtimeWorlds.end(),
                                        [](const auto &world) { return world->simulation.isPlaying; });
    const auto *target = WorldExecutionScope::Current();
    const auto ids =
        target && target != m_authoringWorld.get() ? std::vector<uint64_t>{target->id} : GetRuntimeWorldIds();
    for (uint64_t id : ids) {
        const WorldExecutionScope scope(FindRuntimeWorld(id));
        StopCurrentWorld();
    }
    for (uint64_t id : ids)
        BeginRuntimeWorldRetirement(id);
    if (wasPlaying && m_onPlayStateChanged)
        m_onPlayStateChanged(std::any_of(m_runtimeWorlds.begin(), m_runtimeWorlds.end(),
                                         [](const auto &world) { return world->simulation.isPlaying; }));
}

void SceneManager::StopCurrentWorld()
{
    const WorldExecutionScope worldScope(&CurrentWorld());
    auto &world = CurrentWorld();
    Scene::RequireNoWorldRetirement();
    world.frame.pending.clear();
    CloseRuntimeFrame(false);
    const InputReadScope hostInput({}, 0);
    PhysicsWorld::Instance().CancelPendingStep();
    const bool wasPlaying = world.simulation.isPlaying;
    world.simulation.isPlaying = false;
    world.simulation.isPaused = false;
    if (wasPlaying) {
        ++world.simulation.simulationEpoch;
        world.simulation.fixedStepIndex = 0;
        world.simulation.completedFixedStepIndex = 0;
        world.simulation.publishedFixedStepIndex = 0;
    }
    // CancelPendingStep retires bound constraints before the Play epoch ends.
    // Drop those callbacks' pending nodes before editor/runtime graph restore.
    PhysicsECSStore::Instance().ClearPendingJoints(world);
    world.simulation.fixedPhase = RuntimeWorld::FixedPhase::Idle;
    world.simulation.pausedStepPending = false;
    world.simulation.stepFrameOpen = false;
#if !defined(INFERNUX_RUNTIME_MINIMAL_HOST)
    AudioEngine::Instance().RefreshWorldState(world);
#endif
    world.simulation.fixedTimeAccumulator = 0.0f;
    world.simulation.fixedTime = 0.0;
    world.simulation.fixedUnscaledTime = 0.0;
    ResetFrameTime();
    world.simulation.runtimeFrameCount = 0;

    // Notify renderer that play stopped.

    // Runtime persistence ends exactly at the Play boundary. Python restores
    // the authored scene snapshot after this native graph has been destroyed.
    ClearRuntimePersistentScene();
    UpdateRuntimeScenePlayingState(false);

    // Scene state restore is handled by Python PlayModeManager
    // (serialize on Play, deserialize on Stop)
}

void SceneManager::Pause()
{
    const auto *target = WorldExecutionScope::Current();
    const auto ids =
        target && target != m_authoringWorld.get() ? std::vector<uint64_t>{target->id} : GetRuntimeWorldIds();
    for (uint64_t id : ids) {
        const WorldExecutionScope scope(FindRuntimeWorld(id));
        PauseCurrentWorld();
    }
}

void SceneManager::PauseCurrentWorld()
{
    const WorldExecutionScope worldScope(&CurrentWorld());
    auto &world = CurrentWorld();
    if (!world.simulation.isPlaying || world.simulation.isPaused)
        return;
    world.simulation.isPaused = true;
    // Explicit Pause ends observations which were never admitted to a WorldFrame.
    // The open F and accepted physics/NN requests keep their own retained owners.
    world.frame.pending.clear();
#if !defined(INFERNUX_RUNTIME_MINIMAL_HOST)
    AudioEngine::Instance().RefreshWorldState(world);
#endif
}

void SceneManager::Step(float requestedDeltaTime)
{
    const WorldExecutionScope worldScope(&CurrentWorld());
    auto &world = CurrentWorld();
    std::optional<FrameWorkOwner::Scope> accounting;
    if (m_runtimeFrameWork && m_runtimeFrameWork->IsActive())
        accounting.emplace(*m_runtimeFrameWork);
    if (world.simulation.fixedPhase == RuntimeWorld::FixedPhase::Failed)
        throw std::runtime_error("the physical step failed; stop Play before restarting the simulation");
    if (!world.simulation.isPaused || !world.simulation.isPlaying ||
        (world.scenes.loaded.empty() && !world.scenes.persistent))
        return;
    if (world.simulation.isStepping || (world.simulation.stepFrameOpen && !world.simulation.pausedStepPending))
        throw std::logic_error("a paused simulation step cannot be reentered");
    const bool beginningFrame = !world.simulation.pausedStepPending;
    world.simulation.isStepping = true;
    world.simulation.pausedStepPending = true;
    struct StepScope
    {
        bool &active;
        ~StepScope()
        {
            active = false;
        }
    } stepScope{world.simulation.isStepping};
    if (beginningFrame) {
        world.simulation.pausedStepDeltaTime = requestedDeltaTime;
        world.frame.stepTarget = world.simulation.completedFixedStepIndex + 1;
        world.frame.stepInput = InputManager::Instance().CaptureFrame();
        world.simulation.stepFrameOpen = true;
        if (world.frame.phase == RuntimeWorld::FramePhase::Input ||
            world.frame.phase == RuntimeWorld::FramePhase::Fixed)
            world.frame.fixedTarget = std::min(world.frame.fixedTarget, world.frame.stepTarget);
    }
    const bool managedCache = TransformECSStore::Instance().IsFrameCacheActive();
    for (;;) {
        if (!CanContinueRuntimeFrame())
            return;
        if (world.frame.phase == RuntimeWorld::FramePhase::Idle) {
            const float delta = world.simulation.pausedStepDeltaTime;
            AdmitRuntimeFrame({world.simulation.timeScale > 0.0f ? delta / world.simulation.timeScale : 0.0f, delta,
                               world.frame.stepInput},
                              true);
            world.frame.fixedTarget = world.frame.stepTarget;
        }
        const auto admittedState = GetSimulationState();
        const InputReadScope inputScope(world.frame.input, world.id);
        if (world.simulation.startupPending)
            StartActiveSceneForPlay();
        if (GetSimulationState() == admittedState)
            ContinueRuntimeUpdate();
        if (GetSimulationState() == admittedState)
            ContinueRuntimeLateUpdate();
        const bool current = GetSimulationState() == admittedState;
        const bool published = EndRuntimeFrame(false);
        if (!current || !published || GetSimulationState() != admittedState)
            return;
        if (world.simulation.completedFixedStepIndex >= world.frame.stepTarget) {
            world.simulation.pausedStepPending = false;
            world.frame.stepInput.reset();
            return;
        }
        // The old F had already passed Fixed. Its tail has now completed;
        // the one new Step F uses the request-time sample and fixed target.
        if (managedCache)
            TransformECSStore::Instance().BeginFrameCache();
    }
}

void SceneManager::DontDestroyOnLoad(GameObject *gameObject)
{
    const WorldExecutionScope worldScope(
        gameObject && gameObject->GetScene() ? gameObject->GetScene()->GetRuntimeWorld() : nullptr);
    auto &world = CurrentWorld();
    // DontDestroyOnLoad is runtime residency, not an authored flag. Ignoring
    // Edit Mode calls prevents a script/tool invocation from leaking an object
    // across project or play-session boundaries.
    if (!world.simulation.isPlaying || !gameObject)
        return;

    while (gameObject->GetParent())
        gameObject = gameObject->GetParent();
    if (gameObject->GetScene() == world.scenes.persistent.get())
        return;
    if (!gameObject->GetScene())
        return;

    gameObject->SetPersistent(true);
    if (world.scenes.pendingPersistentRootIdSet.insert(gameObject->GetID()).second)
        world.scenes.pendingPersistentRootIds.push_back(gameObject->GetID());
}

Scene *SceneManager::EnsureRuntimePersistentScene()
{
    auto &world = CurrentWorld();
    if (!world.simulation.isPlaying)
        return nullptr;
    if (!world.scenes.persistent) {
        world.scenes.persistent = std::make_unique<Scene>("DontDestroyOnLoad");
        world.scenes.persistent->m_runtimeWorld = &world;
        world.scenes.persistent->SetRuntimeLifecycleSchedulerEnabled(m_runtimeLifecycleSchedulerEnabled);
        world.scenes.persistent->SetPlaying(true);
        // Start the empty Scene once. Trees transferred into it already own
        // their lifecycle state and must never replay Awake/Start/OnEnable.
        world.scenes.persistent->Start();
    }
    return world.scenes.persistent.get();
}

GameObject *SceneManager::FindRuntimeObjectByID(uint64_t id) const
{
    auto &world = CurrentWorld();
    if (id == 0)
        return nullptr;
    if (world.scenes.active) {
        if (GameObject *object = world.scenes.active->FindByID(id))
            return object;
    }
    for (const auto &scene : world.scenes.loaded) {
        if (!scene || scene.get() == world.scenes.active)
            continue;
        if (GameObject *object = scene->FindByID(id))
            return object;
    }
    if (world.scenes.persistent) {
        if (GameObject *object = world.scenes.persistent->FindByID(id))
            return object;
    }
    return nullptr;
}

Scene *SceneManager::GetSceneByWorldId(uint64_t worldId) const noexcept
{
    auto &world = CurrentWorld();
    if (worldId == 0)
        return nullptr;
    for (const auto &scene : world.scenes.loaded) {
        if (scene && scene->GetWorldId() == worldId)
            return scene.get();
    }
    if (world.scenes.persistent && world.scenes.persistent->GetWorldId() == worldId)
        return world.scenes.persistent.get();
    for (const auto &scene : m_previewScenes)
        if (scene->GetWorldId() == worldId)
            return scene.get();
    return nullptr;
}

GameObject *SceneManager::FindRuntimeObject(const std::string &name) const
{
    auto &world = CurrentWorld();
    if (world.scenes.active) {
        if (GameObject *object = world.scenes.active->Find(name))
            return object;
    }
    for (const auto &scene : world.scenes.loaded) {
        if (!scene || scene.get() == world.scenes.active)
            continue;
        if (GameObject *object = scene->Find(name))
            return object;
    }
    return world.scenes.persistent ? world.scenes.persistent->Find(name) : nullptr;
}

GameObject *SceneManager::FindRuntimeObjectWithTag(const std::string &tag) const
{
    auto &world = CurrentWorld();
    if (world.scenes.active) {
        if (GameObject *object = world.scenes.active->FindWithTag(tag))
            return object;
    }
    for (const auto &scene : world.scenes.loaded) {
        if (!scene || scene.get() == world.scenes.active)
            continue;
        if (GameObject *object = scene->FindWithTag(tag))
            return object;
    }
    return world.scenes.persistent ? world.scenes.persistent->FindWithTag(tag) : nullptr;
}

std::vector<GameObject *> SceneManager::FindRuntimeObjectsWithTag(const std::string &tag) const
{
    auto &world = CurrentWorld();
    std::vector<GameObject *> objects;
    const auto append = [&](const Scene *scene) {
        if (!scene)
            return;
        auto found = scene->FindGameObjectsWithTag(tag);
        objects.insert(objects.end(), found.begin(), found.end());
    };
    append(world.scenes.active);
    for (const auto &scene : world.scenes.loaded) {
        if (scene && scene.get() != world.scenes.active)
            append(scene.get());
    }
    append(world.scenes.persistent.get());
    return objects;
}

std::vector<GameObject *> SceneManager::FindRuntimeObjectsInLayer(int layer) const
{
    auto &world = CurrentWorld();
    std::vector<GameObject *> objects;
    const auto append = [&](const Scene *scene) {
        if (!scene)
            return;
        auto found = scene->FindGameObjectsInLayer(layer);
        objects.insert(objects.end(), found.begin(), found.end());
    };
    append(world.scenes.active);
    for (const auto &scene : world.scenes.loaded) {
        if (scene && scene.get() != world.scenes.active)
            append(scene.get());
    }
    append(world.scenes.persistent.get());
    return objects;
}

void SceneManager::MoveGameObjectToScene(GameObject *gameObject, Scene *destination)
{
    const WorldExecutionScope worldScope(
        gameObject && gameObject->GetScene() ? gameObject->GetScene()->GetRuntimeWorld() : nullptr);
    auto &world = CurrentWorld();
    if (!gameObject || !destination)
        throw std::invalid_argument("game object and destination Scene are required");
    if (world.scenes.loadedSet.find(destination) == world.scenes.loadedSet.end())
        throw std::invalid_argument("destination Scene must be loaded");
    Scene *source = gameObject->GetScene();
    if (!source || !IsRuntimeScene(source))
        throw std::invalid_argument("game object must belong to a loaded Scene");
    if (source == destination)
        return;
    if (gameObject->GetParent())
        throw std::invalid_argument("only a root GameObject can move between Scenes");
    if (!source->TransferRootObjectTo(gameObject, *destination))
        throw std::runtime_error("GameObject hierarchy could not move between Scenes");
}

void SceneManager::FlushPersistentPromotions()
{
    auto &world = CurrentWorld();
    if (!world.simulation.isPlaying || world.scenes.pendingPersistentRootIds.empty())
        return;

    std::vector<uint64_t> requests;
    requests.swap(world.scenes.pendingPersistentRootIds);
    world.scenes.pendingPersistentRootIdSet.clear();
    Scene *persistentScene = EnsureRuntimePersistentScene();
    if (!persistentScene)
        return;

    for (uint64_t id : requests) {
        GameObject *root = FindRuntimeObjectByID(id);
        if (!root)
            continue;
        while (root->GetParent())
            root = root->GetParent();
        Scene *source = root->GetScene();
        if (!source || source == persistentScene)
            continue;
        if (!source->TransferRootObjectTo(root, *persistentScene)) {
            root->SetPersistent(false);
            INXLOG_WARN("DontDestroyOnLoad could not promote root '", root->GetName(), "' (id=", root->GetID(), ")");
        }
    }
}

void SceneManager::PrepareActiveSceneReplacement()
{
    auto &world = CurrentWorld();
    // Finish accepted work against the old graph before retiring its actors.
    // This does not run FixedUpdate or admit another solver step, and leaves
    // additive/persistent actors on the same compute owner.
    auto &physics = PhysicsWorld::Instance();
    const bool retainedStep = physics.HasPendingStep();
    const bool needsPublication =
        world.simulation.fixedHasPhysics &&
        (world.simulation.fixedPhase == RuntimeWorld::FixedPhase::Publication ||
         (retainedStep && world.simulation.fixedPhase == RuntimeWorld::FixedPhase::Integration));
    physics.CompletePendingStep();
    if (retainedStep && world.simulation.fixedPhase == RuntimeWorld::FixedPhase::Integration) {
        if (!physics.Step(world.simulation.admittedFixedDeltaTime))
            throw std::logic_error("Completed physical step did not retire at the scene replacement boundary");
    }
    if (needsPublication) {
        PublishPhysicsStateToRenderer();
    }
    // SceneDocumentTransaction replaces the active graph in place, so the
    // Scene pointer does not change and SetActiveScene cannot publish this
    // boundary. Discard both wall-clock loading time and old-scene fixed-step
    // remainder before the new graph is allowed to tick.
    world.simulation.fixedTimeAccumulator = 0.0f;
    world.simulation.fixedPhase = RuntimeWorld::FixedPhase::Idle;
    world.simulation.pausedStepPending = false;
    world.simulation.stepFrameOpen = false;
    world.frame.pending.clear();
    CloseRuntimeFrame(false);
    world.frame.stepInput.reset();
    world.frame.stepTarget = 0;
    ResetFrameDelta();
    if (world.simulation.isPlaying) {
        FlushPersistentPromotions();
        // Publish promoted bodies at this explicit Single-replacement boundary.
        // Scene commits retain other Worlds' pending physics work.
        FlushPendingBroadphase();
    }
}

void SceneManager::UpdateRuntimeScenePlayingState(bool playing)
{
    auto &world = CurrentWorld();
    for (const auto &scene : world.scenes.loaded)
        if (scene)
            scene->SetPlaying(playing);
    if (world.scenes.persistent)
        world.scenes.persistent->SetPlaying(playing);
}

void SceneManager::ClearRuntimePersistentScene()
{
    auto &world = CurrentWorld();
    Scene::RequireNoWorldRetirement();
    for (uint64_t id : world.scenes.pendingPersistentRootIds) {
        if (GameObject *root = FindRuntimeObjectByID(id))
            root->SetPersistent(false);
    }
    world.scenes.pendingPersistentRootIds.clear();
    world.scenes.pendingPersistentRootIdSet.clear();
    if (world.scenes.persistent)
        world.scenes.persistent->SetPlaying(false);
    world.scenes.persistent.reset();
}

void SceneManager::RestoreResidentComponentRegistries(Scene *sceneBeingRebuilt)
{
    const WorldExecutionScope worldScope(sceneBeingRebuilt ? sceneBeingRebuilt->GetRuntimeWorld() : nullptr);
    auto &world = CurrentWorld();
    auto restore = [this](Scene *scene) {
        if (!scene)
            return;
        for (GameObject *object : scene->GetAllObjects()) {
            if (!object || !object->IsActiveInHierarchy())
                continue;
            for (MeshRenderer *renderer : object->GetComponents<MeshRenderer>()) {
                if (renderer && renderer->IsEnabled())
                    RegisterMeshRenderer(renderer);
            }
            for (Light *light : object->GetComponents<Light>()) {
                if (light && light->IsEnabled())
                    RegisterLight(light);
            }
        }
    };
    for (const auto &scene : world.scenes.loaded)
        if (scene && scene.get() != sceneBeingRebuilt)
            restore(scene.get());
    restore(world.scenes.persistent.get());
}

void SceneManager::SyncCollidersToPhysics(float fixedDeltaTime)
{
    auto &world = CurrentWorld();
    TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    // FixedUpdate may have authored cached poses. They must reach physics
    // before the solver, rather than waiting for the end-of-frame commit.
    TransformECSStore::Instance().VisitAuthoredPoseWrites([](Transform *transform, bool rotationAuthored) {
        PhysicsECSStore::Instance().MarkAuthoredHierarchyDirty(transform->GetGameObject(), rotationAuthored);
    });
    auto &store = PhysicsECSStore::Instance();
    const auto &dirtyColliders = store.ConsumeDirtyActors(world);
    m_lastFrameProfile.colliderSyncCandidates += static_cast<double>(dirtyColliders.size());
    static thread_local std::vector<PhysicsBodyPoseUpdate> staticPoseBatch;
    staticPoseBatch.clear();
    staticPoseBatch.reserve(dirtyColliders.size());

    for (const auto handle : dirtyColliders) {
        if (!store.IsValid(handle))
            continue;
        const auto &actor = store.GetActor(handle);
        if (!actor.owner || !IsRuntimeScene(actor.owner->GetScene()))
            continue;
        const bool profilesRigidbodySync = actor.rigidbody && actor.rigidbody->IsEnabled();
        const auto syncStart = profilesRigidbodySync ? ProfileClock::now() : ProfileClock::time_point{};
        store.SyncActorTransform(handle, fixedDeltaTime, &staticPoseBatch);
        if (profilesRigidbodySync)
            m_lastFrameProfile.syncExternalMovesMs += ProfileMsSince(syncStart);
    }
    PhysicsWorld::Instance().SetBodyPositionsBatch(staticPoseBatch);
}

void SceneManager::PublishAuthoredTransformsToPhysics()
{
    auto &world = CurrentWorld();
    if (HasPendingFixedStep())
        return; // The retained step owns its original synchronization boundary and dt.
    // Runtime persistence is committed only after the Transform frame cache
    // has flushed. Moving a root while that cache still names the authored
    // Scene would otherwise make EndFrameCache skip its dirty slots.
    FlushPersistentPromotions();
    if (world.scenes.persistent)
        TransformECSStore::Instance().SyncSceneWorldMatrices(world.scenes.persistent.get());

    auto &store = PhysicsECSStore::Instance();
    if (world.scenes.loaded.empty() && !world.scenes.persistent)
        return;

    // Transform writes performed by Update/LateUpdate are collected by the
    // frame cache. Publish them once, after the cache commits, instead of
    // synchronizing on every property setter. This preserves high-FPS editor
    // performance while preventing Rigidbody interpolation from restoring an
    // older pose on the following frame.
    if (store.HasDirtyColliders(world)) {
        FlushPendingBroadphase();
        SyncCollidersToPhysics(0.0f);
    }
    // Ordinary child meshes can change without dirtying a physical actor.
    // Admit their export after authored local channels have been committed.
    PhysicsWorld::Instance().RefreshRenderPoses();
}

bool SceneManager::RunFixedSimulationStep(bool useRuntimeScheduler, const SimulationState &admittedState)
{
    const WorldExecutionScope worldScope(&CurrentWorld());
    auto &world = CurrentWorld();
    if (world.simulation.fixedPhase == RuntimeWorld::FixedPhase::Failed)
        throw std::runtime_error("the physical step failed; stop Play before restarting the simulation");
    if (world.simulation.fixedExecuting)
        throw std::logic_error("a fixed simulation step cannot be reentered");
    world.simulation.fixedExecuting = true;
    struct ExecutionScope
    {
        bool &active;
        ~ExecutionScope()
        {
            active = false;
        }
    } executionScope{world.simulation.fixedExecuting};
    const auto validWorld = [&] {
        return world.simulation.simulationEpoch == std::get<0>(admittedState) && world.simulation.isPlaying &&
               world.simulation.fixedPhase != RuntimeWorld::FixedPhase::Idle;
    };
    const auto current = [&] { return validWorld() && GetSimulationState() == admittedState; };
    if (world.simulation.fixedPhase == RuntimeWorld::FixedPhase::Idle) {
        world.simulation.admittedFixedDeltaTime = world.simulation.fixedTimeStep;
        world.simulation.fixedAccumulatorDebit = world.frame.debitFixedAccumulator;
        world.simulation.fixedHasPhysics = false;
        ++world.simulation.fixedStepIndex;
        world.simulation.fixedTime += static_cast<double>(world.simulation.admittedFixedDeltaTime);
        if (world.simulation.timeScale > 0.0f)
            world.simulation.fixedUnscaledTime +=
                static_cast<double>(world.simulation.admittedFixedDeltaTime / world.simulation.timeScale);
        world.simulation.fixedSceneWorlds.clear();
        for (const auto &scene : world.scenes.loaded)
            if (scene)
                world.simulation.fixedSceneWorlds.push_back(scene->GetWorldId());
        if (world.scenes.persistent)
            world.simulation.fixedSceneWorlds.push_back(world.scenes.persistent->GetWorldId());
        world.simulation.fixedSceneCursor = 0;
        world.simulation.fixedPhase = RuntimeWorld::FixedPhase::SceneScripts;
        FlushPendingBroadphase();
        if (!current())
            return false;
        const auto phaseStart = ProfileClock::now();
        if (useRuntimeScheduler && world.lifecycle.runtimeLifecycleFixedUpdateCount > 0)
            if (auto callback = m_runtimeLifecycleFixedUpdate)
                callback(world.simulation.admittedFixedDeltaTime);
        m_lastFrameProfile.fixedUpdateMs += ProfileMsSince(phaseStart);
        if (!current())
            return false;
    }
    if (world.simulation.fixedPhase == RuntimeWorld::FixedPhase::SceneScripts) {
        if (!RuntimeResultsReady(RuntimeFrameBarrier::FixedScript) || !current())
            return false;
        const auto phaseStart = ProfileClock::now();
        while (world.simulation.fixedSceneCursor < world.simulation.fixedSceneWorlds.size()) {
            if (!CanContinueRuntimeFrame())
                return false;
            if (auto *scene = GetSceneByWorldId(world.simulation.fixedSceneWorlds[world.simulation.fixedSceneCursor]))
                scene->FixedUpdate(world.simulation.admittedFixedDeltaTime);
            if (!current() || !CanContinueRuntimeFrame())
                return false;
            ++world.simulation.fixedSceneCursor;
        }
        if (!TransformECSStore::Instance().IsFrameCacheActive())
            FlushPersistentPromotions();
        m_lastFrameProfile.fixedUpdateMs += ProfileMsSince(phaseStart);
        world.simulation.fixedPhase = RuntimeWorld::FixedPhase::Control;
        EmitRuntimeFrameBarrier(RuntimeFrameBarrier::TransformToPhysics);
        if (!current())
            return false;
    }
    if (world.simulation.fixedPhase == RuntimeWorld::FixedPhase::Control) {
        if (!RuntimeResultsReady(RuntimeFrameBarrier::PhysicsSimulation) || !current())
            return false;
        // Applying a result may author Transform, body, or joint data. The
        // admitted dt belongs to this step even when settings changed while waiting.
        FlushPendingBroadphase();
        SyncCollidersToPhysics(world.simulation.admittedFixedDeltaTime);
        world.simulation.fixedPhase = RuntimeWorld::FixedPhase::Integration;
        EmitRuntimeFrameBarrier(RuntimeFrameBarrier::PhysicsSimulation);
        if (!current())
            return false;
    }
    if (world.simulation.fixedPhase == RuntimeWorld::FixedPhase::Integration) {
        if (PhysicsWorld::Instance().HasPendingStep() || PhysicsWorld::Instance().HasPendingControl() ||
            PhysicsECSStore::Instance().GetAliveRigidbodyCount(world) > 0) {
            world.simulation.fixedHasPhysics = true;
            const auto phaseStart = ProfileClock::now();
            PhysicsStepProfile profile;
            bool completed = false;
            try {
                completed = PhysicsWorld::Instance().Step(world.simulation.admittedFixedDeltaTime, &profile);
            } catch (...) {
                // Accepted GPU work may have changed its retained storage. Preserve the
                // last published world and never replay this admitted step.
                world.simulation.fixedPhase = RuntimeWorld::FixedPhase::Failed;
                world.simulation.pausedStepPending = false;
                Pause();
                throw;
            }
            m_lastFrameProfile.physicsStepMs += ProfileMsSince(phaseStart);
            m_lastFrameProfile.physicsPrepareMs += profile.prepareNs * 1e-6;
            m_lastFrameProfile.physicsCCDMs += profile.ccdNs * 1e-6;
            m_lastFrameProfile.physicsJoltMs += profile.joltNs * 1e-6;
            m_lastFrameProfile.physicsFinishMs += profile.finishNs * 1e-6;
            m_lastFrameProfile.dynamicCCDSplits +=
                static_cast<double>(PhysicsWorld::Instance().GetLastDynamicCCDSplitCount());
            if (!completed)
                return false;
        }
        world.simulation.completedFixedStepIndex = world.simulation.fixedStepIndex;
        m_lastFrameProfile.fixedSteps += 1.0;
        if (world.simulation.fixedAccumulatorDebit)
            world.simulation.fixedTimeAccumulator -= world.simulation.admittedFixedDeltaTime;
        // Mark before user collision callbacks: a Pause or delayed publication
        // must never repeat the solver or dispatch this batch a second time.
        world.simulation.fixedPhase = RuntimeWorld::FixedPhase::Publication;
        if (world.simulation.fixedHasPhysics) {
            const auto phaseStart = ProfileClock::now();
            m_lastFrameProfile.contactEvents += static_cast<double>(PhysicsWorld::Instance().DispatchContactEvents());
            m_lastFrameProfile.physicsEventsMs += ProfileMsSince(phaseStart);
        }
        if (!validWorld())
            return false;
    }
    if (world.simulation.fixedPhase == RuntimeWorld::FixedPhase::Publication) {
        if (!RuntimeResultsReady(RuntimeFrameBarrier::PhysicsToTransform) || !validWorld())
            return false;
        if (world.simulation.fixedHasPhysics) {
            const auto phaseStart = ProfileClock::now();
            // The world published one completed GPU generation. Invalidating
            // consumers here does not read every body back into Transform.
            PublishPhysicsStateToRenderer();
            m_lastFrameProfile.syncRigidbodiesMs += ProfileMsSince(phaseStart);
        }
        world.simulation.publishedFixedStepIndex = world.simulation.completedFixedStepIndex;
        world.simulation.fixedPhase = RuntimeWorld::FixedPhase::PostNotification;
        if (!current())
            return false;
    }
    if (world.simulation.fixedPhase == RuntimeWorld::FixedPhase::PostNotification) {
        world.simulation.fixedPhase = RuntimeWorld::FixedPhase::PostStep;
        EmitRuntimeFrameBarrier(RuntimeFrameBarrier::PhysicsToTransform);
        if (!current())
            return false;
    }
    if (!RuntimeResultsReady(RuntimeFrameBarrier::PhysicsPostScript) || !current())
        return false;
    world.simulation.fixedPhase = RuntimeWorld::FixedPhase::Idle;
    return true;
}

void SceneManager::FlushPendingBroadphase()
{
    auto &world = CurrentWorld();
    MeshCollider::FlushCompletedCooking();
    auto &store = PhysicsECSStore::Instance();
    auto &pw = PhysicsWorld::Instance();
    if (!pw.IsInitialized())
        return;

    // ── Create deferred Jolt bodies ──
    auto pendingBodies = store.ConsumePendingBodyCreations(world);
    if (pendingBodies.empty() && !store.HasPendingBroadphaseAdds(world) && !store.HasPendingBroadphaseRemoves(world)) {
        store.DrainPendingJoints(world);
        return;
    }

    auto t0 = ProfileClock::now();
    const size_t bodyCount = pendingBodies.size();

    for (auto handle : pendingBodies) {
        if (store.IsValid(handle) && store.GetActor(handle).bodyId == 0xFFFFFFFF)
            store.RegisterActor(handle);
    }

    double createBodiesMs = ProfileMsSince(t0);

    // ── Remove before add ──
    // Gameplay Update/FixedUpdate may toggle Collider.enabled. Submit those
    // mutations only at this fixed-step boundary so Jolt never observes a
    // broadphase change from inside a component callback.
    for (const uint32_t bodyId : store.ConsumePendingBroadphaseRemoves(world)) {
        pw.RemoveBodyFromBroadphase(bodyId);
    }

    // ── Batch add to broadphase ──
    auto t1 = ProfileClock::now();
    auto pending = store.ConsumePendingBroadphaseAdds(world);
    if (pending.empty()) {
        store.DrainPendingJoints(world);
        return;
    }

    // Use Jolt batch API (AddBodiesPrepare/Finalize) for large batches,
    // which is significantly faster than individual AddBody calls.
    pw.AddBodiesBatch(pending);
    // Both endpoints' actor initialization and residency now precede the
    // Joint topology group in the same ordered author command source.
    store.DrainPendingJoints(world);

    double addBodiesMs = ProfileMsSince(t1);

    // Jolt incrementally maintains its broad phase when bodies are added.
    // Rebuilding the complete tree for one or two runtime spawns creates a
    // periodic main-thread spike (particularly visible in continuous physics
    // piles) without improving the query structure. Reserve the explicit
    // optimization for genuinely large scene-import batches.
    double optimizeMs = 0.0;
    constexpr size_t kBroadphaseRebuildBatchThreshold = 128;
    if (pending.size() >= kBroadphaseRebuildBatchThreshold) {
        auto t2 = ProfileClock::now();
        pw.OptimizeBroadPhase();
        optimizeMs = ProfileMsSince(t2);
    }

    // if (bodyCount >= 100) {
    //     INXLOG_INFO("[Perf] FlushPendingBroadphase: ", bodyCount, " bodies — "
    //                 "CreateBody: ", static_cast<int>(createBodiesMs), "ms, "
    //                 "AddBatch: ", static_cast<int>(addBodiesMs), "ms, "
    //                 "Optimize: ", static_cast<int>(optimizeMs), "ms");
    // }
}

void SceneManager::SyncTransforms(float motionDeltaTime)
{
    // First pass starts any deferred mesh cooking. Explicit SyncTransforms is
    // a barrier: wait for those immutable CPU jobs, commit them on this main
    // thread, then create/rebuild bodies before returning.
    FlushPendingBroadphase();
    PhysicsECSStore::Instance().MarkAllCollidersDirty(CurrentWorld());
    // The ordinary barrier publishes immediately. Editor drag explicitly
    // supplies a step when its movement must impart contact velocity.
    SyncCollidersToPhysics(motionDeltaTime);
    // Scale changes discovered above can start mesh cooking too. Include
    // those jobs in this explicit barrier rather than waiting before discovery.
    MeshCollider::FlushCompletedCooking(true);
    FlushPendingBroadphase();
    SyncCollidersToPhysics(motionDeltaTime);
}

void SceneManager::ForceAllBodiesToCurrentTransform()
{
    TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    auto &pw = PhysicsWorld::Instance();
    if (!pw.IsInitialized())
        return;

    PhysicsECSStore::Instance().ForEachAliveCollider([&pw](ColliderECSData &data) {
        auto *col = data.owner;
        if (!col || col->GetBodyId() == 0xFFFFFFFF)
            return;

        auto *go = col->GetGameObject();
        if (!go)
            return;

        Transform *tf = go->GetTransform();
        if (!tf)
            return;

        glm::quat rot = tf->GetWorldRotation();
        glm::vec3 pos = tf->GetPosition();
        pw.SetBodyPosition(col->GetBodyId(), pos, rot);
    });
}

// ============================================================================
// Component registries
// ============================================================================

void SceneManager::ClearComponentRegistries(Scene *sceneBeingRebuilt)
{
    if (sceneBeingRebuilt && !sceneBeingRebuilt->GetRuntimeWorld())
        return;
    const WorldExecutionScope worldScope(sceneBeingRebuilt ? sceneBeingRebuilt->GetRuntimeWorld() : nullptr);
    // Renderer-facing registries: MeshRenderer/Light keep raw component
    // pointers, so we must drop them before the owning GameObjects die during
    // a Scene::DeserializeDocument() commit (see the Scene Rebuild Contract).
    CurrentWorld().render.activeMeshRenderers.clear();
    CurrentWorld().render.activeMeshRendererSet.clear();
    CurrentWorld().render.pendingSkinPoseHistoryCommits.clear();
    CurrentWorld().render.activeLights.clear();
    ++CurrentWorld().render.meshRendererVersion;

    // Physics queues belong to individual actors across all resident Worlds.
    // Collider/actor handles include their generation, and body destruction
    // cancels its own pending broadphase commands. Clearing those queues here
    // would discard live Worlds' creation, enabled-state and transform edits.

    // Scene document replacement clears process-wide registries. Re-publish
    // every other loaded Scene plus the runtime-persistent Scene; the caller
    // publishes the rebuilt Scene after its transaction commits.
    RestoreResidentComponentRegistries(sceneBeingRebuilt);
}

// ========================================================================
// MeshRenderer component registry
// ========================================================================

void SceneManager::ReserveRendererCapacity(size_t count)
{
    CurrentWorld().render.activeMeshRenderers.reserve(CurrentWorld().render.activeMeshRenderers.size() + count);
    CurrentWorld().render.activeMeshRendererSet.reserve(CurrentWorld().render.activeMeshRendererSet.size() + count);
}

void SceneManager::BeginRendererRegistryTransaction()
{
    ++CurrentWorld().render.rendererRegistryTransactionDepth;
}

void SceneManager::EndRendererRegistryTransaction()
{
    if (CurrentWorld().render.rendererRegistryTransactionDepth == 0)
        throw std::logic_error("renderer registry transaction underflow");
    --CurrentWorld().render.rendererRegistryTransactionDepth;
    if (CurrentWorld().render.rendererRegistryTransactionDepth == 0 &&
        CurrentWorld().render.rendererRegistryTransactionDirty) {
        CurrentWorld().render.rendererRegistryTransactionDirty = false;
        ++CurrentWorld().render.meshRendererVersion;
    }
}

namespace
{
void MarkRendererRegistryChanged(uint32_t transactionDepth, bool &transactionDirty, uint64_t &version)
{
    if (transactionDepth != 0)
        transactionDirty = true;
    else
        ++version;
}
} // namespace

void SceneManager::RegisterMeshRenderer(MeshRenderer *renderer)
{
    auto *ownerScene = renderer && renderer->GetGameObject() ? renderer->GetGameObject()->GetScene() : nullptr;
    if (!ownerScene || !ownerScene->GetRuntimeWorld())
        return;
    const WorldExecutionScope ownerScope(ownerScene->GetRuntimeWorld());
    if (!renderer)
        return;
    if (CurrentWorld().render.activeMeshRendererSet.insert(renderer).second) {
        CurrentWorld().render.activeMeshRenderers.push_back(renderer);
        if (auto *skinned = dynamic_cast<SkinnedMeshRenderer *>(renderer))
            QueueSkinPoseHistoryCommit(skinned);
        MarkRendererRegistryChanged(CurrentWorld().render.rendererRegistryTransactionDepth,
                                    CurrentWorld().render.rendererRegistryTransactionDirty,
                                    CurrentWorld().render.meshRendererVersion);
    }
}

void SceneManager::UnregisterMeshRenderer(MeshRenderer *renderer)
{
    auto *ownerScene = renderer && renderer->GetGameObject() ? renderer->GetGameObject()->GetScene() : nullptr;
    if (!ownerScene || !ownerScene->GetRuntimeWorld())
        return;
    const WorldExecutionScope ownerScope(ownerScene->GetRuntimeWorld());
    CurrentWorld().render.pendingSkinPoseHistoryCommits.erase(renderer);
    if (!CurrentWorld().render.activeMeshRendererSet.erase(renderer))
        return;
    MarkRendererRegistryChanged(CurrentWorld().render.rendererRegistryTransactionDepth,
                                CurrentWorld().render.rendererRegistryTransactionDirty,
                                CurrentWorld().render.meshRendererVersion);
    // Swap-and-pop for O(1) removal from vector
    for (size_t i = 0; i < CurrentWorld().render.activeMeshRenderers.size(); ++i) {
        if (CurrentWorld().render.activeMeshRenderers[i] == renderer) {
            CurrentWorld().render.activeMeshRenderers[i] = CurrentWorld().render.activeMeshRenderers.back();
            CurrentWorld().render.activeMeshRenderers.pop_back();
            return;
        }
    }
}

void SceneManager::NotifyMeshRendererChanged(MeshRenderer *renderer)
{
    auto *ownerScene = renderer && renderer->GetGameObject() ? renderer->GetGameObject()->GetScene() : nullptr;
    if (!ownerScene || !ownerScene->GetRuntimeWorld())
        return;
    const WorldExecutionScope ownerScope(ownerScene->GetRuntimeWorld());
    if (!renderer)
        return;
    if (CurrentWorld().render.activeMeshRendererSet.find(renderer) != CurrentWorld().render.activeMeshRendererSet.end())
        MarkRendererRegistryChanged(CurrentWorld().render.rendererRegistryTransactionDepth,
                                    CurrentWorld().render.rendererRegistryTransactionDirty,
                                    CurrentWorld().render.meshRendererVersion);
}

void SceneManager::NotifyMeshRendererContentChanged(MeshRenderer *renderer)
{
    auto *ownerScene = renderer && renderer->GetGameObject() ? renderer->GetGameObject()->GetScene() : nullptr;
    if (!ownerScene || !ownerScene->GetRuntimeWorld())
        return;
    const WorldExecutionScope ownerScope(ownerScene->GetRuntimeWorld());
    if (!renderer ||
        CurrentWorld().render.activeMeshRendererSet.find(renderer) == CurrentWorld().render.activeMeshRendererSet.end())
        return;
    ++CurrentWorld().render.renderContentRevision;
    if (CurrentWorld().render.renderContentRevision == 0)
        CurrentWorld().render.renderContentRevision = 1;
}

void SceneManager::QueueSkinPoseHistoryCommit(SkinnedMeshRenderer *renderer)
{
    auto *ownerScene = renderer && renderer->GetGameObject() ? renderer->GetGameObject()->GetScene() : nullptr;
    if (!ownerScene || !ownerScene->GetRuntimeWorld())
        return;
    const WorldExecutionScope ownerScope(ownerScene->GetRuntimeWorld());
    if (renderer &&
        CurrentWorld().render.activeMeshRendererSet.find(renderer) != CurrentWorld().render.activeMeshRendererSet.end())
        CurrentWorld().render.pendingSkinPoseHistoryCommits.emplace(renderer, renderer);
}

void SceneManager::RegisterSkeletonBone(Transform *bone, SkeletonInstance *skeleton)
{
    auto *ownerScene = bone && bone->GetGameObject() ? bone->GetGameObject()->GetScene() : nullptr;
    if (!ownerScene || !ownerScene->GetRuntimeWorld())
        return;
    const WorldExecutionScope ownerScope(ownerScene->GetRuntimeWorld());
    CurrentWorld().render.skeletonBoneOwners[bone] = skeleton;
}

void SceneManager::ForgetSkeletonBone(Transform *bone)
{
    auto *ownerScene = bone && bone->GetGameObject() ? bone->GetGameObject()->GetScene() : nullptr;
    if (!ownerScene || !ownerScene->GetRuntimeWorld())
        return;
    const WorldExecutionScope ownerScope(ownerScene->GetRuntimeWorld());
    if (const auto found = CurrentWorld().render.skeletonBoneOwners.find(bone);
        found != CurrentWorld().render.skeletonBoneOwners.end()) {
        auto *skeleton = found->second;
        CurrentWorld().render.skeletonBoneOwners.erase(found);
        skeleton->MarkPoseDirty();
    }
}

void SceneManager::InvalidatePhysicsSkeletonPoses()
{
    auto &physics = PhysicsECSStore::Instance();
    const auto physicalRoot = [&](GameObject *object) {
        for (; object; object = object->GetParent()) {
            const auto handle = physics.FindActor(object);
            if (!physics.IsValid(handle))
                continue;
            const auto &actor = physics.GetActor(handle);
            if (actor.rigidbody && actor.rigidbody->IsEnabled() && !actor.sceneSuspended && actor.bodyId != UINT32_MAX)
                return actor.bodyId;
        }
        return UINT32_MAX;
    };
    // A palette needs refresh when a bone has an independently moving physical
    // ancestor. A common rigid root cancels from root-space bone matrices.
    for (const auto &[bone, skeleton] : CurrentWorld().render.skeletonBoneOwners) {
        const auto body = physicalRoot(bone->GetGameObject());
        if (body != UINT32_MAX && body != physicalRoot(skeleton->GetGameObject()))
            skeleton->MarkPoseDirty();
    }
}

void SceneManager::NotifySkeletonBoneChanged(Transform *bone)
{
    auto *ownerScene = bone && bone->GetGameObject() ? bone->GetGameObject()->GetScene() : nullptr;
    if (!ownerScene || !ownerScene->GetRuntimeWorld())
        return;
    const WorldExecutionScope ownerScope(ownerScene->GetRuntimeWorld());
    if (const auto found = CurrentWorld().render.skeletonBoneOwners.find(bone);
        found != CurrentWorld().render.skeletonBoneOwners.end())
        found->second->MarkPoseDirty();
}

void SceneManager::ReleaseSkeletonBones(SkeletonInstance *skeleton, const std::vector<Transform *> &bones)
{
    auto *ownerScene = skeleton && skeleton->GetGameObject() ? skeleton->GetGameObject()->GetScene() : nullptr;
    if (!ownerScene || !ownerScene->GetRuntimeWorld())
        return;
    const WorldExecutionScope ownerScope(ownerScene->GetRuntimeWorld());
    for (auto *bone : bones)
        if (const auto found = CurrentWorld().render.skeletonBoneOwners.find(bone);
            found != CurrentWorld().render.skeletonBoneOwners.end() && found->second == skeleton)
            CurrentWorld().render.skeletonBoneOwners.erase(found);
    CurrentWorld().render.pendingSkeletonPoses.erase(std::remove(CurrentWorld().render.pendingSkeletonPoses.begin(),
                                                                 CurrentWorld().render.pendingSkeletonPoses.end(),
                                                                 skeleton),
                                                     CurrentWorld().render.pendingSkeletonPoses.end());
}

void SceneManager::QueueSkeletonPose(SkeletonInstance *skeleton)
{
    auto *ownerScene = skeleton && skeleton->GetGameObject() ? skeleton->GetGameObject()->GetScene() : nullptr;
    if (!ownerScene || !ownerScene->GetRuntimeWorld())
        return;
    const WorldExecutionScope ownerScope(ownerScene->GetRuntimeWorld());
    CurrentWorld().render.pendingSkeletonPoses.push_back(skeleton);
}

void SceneManager::PublishSkeletonPoses()
{
    while (!CurrentWorld().render.pendingSkeletonPoses.empty()) {
        auto *skeleton = CurrentWorld().render.pendingSkeletonPoses.back();
        CurrentWorld().render.pendingSkeletonPoses.pop_back();
        skeleton->PublishRenderPoses();
    }
}

void SceneManager::CommitSkinPoseHistories()
{
    // Unregistration removes queued entries before component destruction.
    // Static scenes take the empty path without scanning all mesh renderers.
    for (const auto &[identity, renderer] : CurrentWorld().render.pendingSkinPoseHistoryCommits)
        renderer->CommitRuntimeSkinPoseHistory();
    CurrentWorld().render.pendingSkinPoseHistoryCommits.clear();
}

void SceneManager::NotifyMeshRendererGeometryChanged(MeshRenderer *renderer)
{
    auto *ownerScene = renderer && renderer->GetGameObject() ? renderer->GetGameObject()->GetScene() : nullptr;
    if (!ownerScene || !ownerScene->GetRuntimeWorld())
        return;
    const WorldExecutionScope ownerScope(ownerScene->GetRuntimeWorld());
    if (!renderer ||
        CurrentWorld().render.activeMeshRendererSet.find(renderer) == CurrentWorld().render.activeMeshRendererSet.end())
        return;
    ++CurrentWorld().render.renderContentRevision;
    if (CurrentWorld().render.renderContentRevision == 0)
        CurrentWorld().render.renderContentRevision = 1;
    // Camera visibility caches use this revision for every bounds-affecting
    // change, including procedural geometry that moves in world space.
    ++CurrentWorld().render.renderTransformRevision;
    if (CurrentWorld().render.renderTransformRevision == 0)
        CurrentWorld().render.renderTransformRevision = 1;
}

// ========================================================================
// Light component registry
// ========================================================================

void SceneManager::RegisterLight(Light *light)
{
    auto *ownerScene = light && light->GetGameObject() ? light->GetGameObject()->GetScene() : nullptr;
    if (!ownerScene || !ownerScene->GetRuntimeWorld())
        return;
    const WorldExecutionScope ownerScope(ownerScene->GetRuntimeWorld());
    if (!light)
        return;
    for (auto *l : CurrentWorld().render.activeLights) {
        if (l == light)
            return;
    }
    CurrentWorld().render.activeLights.push_back(light);
}

void SceneManager::UnregisterLight(Light *light)
{
    auto *ownerScene = light && light->GetGameObject() ? light->GetGameObject()->GetScene() : nullptr;
    if (!ownerScene || !ownerScene->GetRuntimeWorld())
        return;
    const WorldExecutionScope ownerScope(ownerScene->GetRuntimeWorld());
    for (size_t i = 0; i < CurrentWorld().render.activeLights.size(); ++i) {
        if (CurrentWorld().render.activeLights[i] == light) {
            CurrentWorld().render.activeLights[i] = CurrentWorld().render.activeLights.back();
            CurrentWorld().render.activeLights.pop_back();
            return;
        }
    }
}

} // namespace infernux
