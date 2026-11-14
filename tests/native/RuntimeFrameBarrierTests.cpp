#include <Infernux.h>
#include <function/scene/GameObject.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <function/scene/Transform.h>
#include <function/scene/TransformECSStore.h>
#include <function/scene/physics/PhysicsECSStore.h>
#include <function/scene/physics/PhysicsWorld.h>

#include "BroadphaseQueuePublicationTests.h"
#include "CompoundColliderPublicationTests.h"
#include "PhysicsQueryDefaultMaskTests.h"
#include "TransformHierarchyPublicationTests.h"
#include "TransformWorldBatchTests.h"
#include <cassert>
#include <chrono>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

using infernux::GameObject;
using infernux::PhysicsECSStore;
using infernux::Scene;
using infernux::SceneManager;
using infernux::Transform;
using infernux::TransformECSStore;

namespace
{
// Physics services belong to the engine. Each publication run uses the public
// headless startup/cleanup lifecycle, including the query-mask reinitializations.
class PhysicsPublicationGpuFixture
{
  public:
    PhysicsPublicationGpuFixture()
    {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        m_projectPath =
            std::filesystem::temp_directory_path() / ("infernux-physics-publication-" + std::to_string(stamp));
        if (!std::filesystem::create_directory(m_projectPath))
            throw std::runtime_error("Could not create isolated physics publication project");
    }

    ~PhysicsPublicationGpuFixture()
    {
        ShutdownPhysics();
        infernux::JobSystem::Shutdown();
        std::error_code ignored;
        std::filesystem::remove_all(m_projectPath, ignored);
    }

    void InitializePhysics()
    {
        m_engine = std::make_unique<infernux::Infernux>("", infernux::RuntimeMode::Headless);
        m_engine->InitHeadless(infernux::FromFsPath(m_projectPath), "", {}, true);
    }

    void ShutdownPhysics()
    {
        if (!m_engine)
            return;
        m_engine->Exit();
        m_engine->Cleanup();
        m_engine.reset();
    }

  private:
    std::filesystem::path m_projectPath;
    std::unique_ptr<infernux::Infernux> m_engine;
};
} // namespace

int main()
{
    SceneManager &manager = SceneManager::Instance();
    using Barrier = SceneManager::RuntimeFrameBarrier;

    std::vector<Barrier> observed;
    int beginCount = 0;
    int fixedUpdateCount = 0;
    int updateCount = 0;
    int lateUpdateCount = 0;
    int editorUpdateCount = 0;
    int endCount = 0;
    manager.SetRuntimeFrameBarrierCallback([&observed](Barrier barrier) { observed.push_back(barrier); });

    // Engine-phase notifications are observable even when no script
    // lifecycle frame is open. Render-facing services use these barriers in
    // Edit Mode and in scenes without script components.
    manager.EmitRuntimeFrameBarrier(Barrier::TransformToPhysics);
    assert((observed == std::vector<Barrier>{Barrier::TransformToPhysics}));
    observed.clear();

    manager.CreateScene("RuntimeFrameBarrierTests");
    assert(manager.ConsumeFrameDeltaTime(0.25f) == 0.0f);
    assert(manager.ConsumeFrameDeltaTime(0.25f) == 0.25f);
    manager.PrepareActiveSceneReplacement();
    assert(manager.ConsumeFrameDeltaTime(0.25f) == 0.0f);
    assert(manager.ConsumeFrameDeltaTime(0.25f) == 0.25f);
    manager.SetRuntimeLifecycleCallbacks(
        [&beginCount] { ++beginCount; }, [&fixedUpdateCount](float) { ++fixedUpdateCount; },
        [&updateCount](float) { ++updateCount; }, [&lateUpdateCount](float) { ++lateUpdateCount; },
        [&editorUpdateCount](float) { ++editorUpdateCount; }, [&endCount] { ++endCount; });

    // Installing the lifecycle bridge alone must not dispatch script phases
    // in a scene with no script components, while engine barriers remain
    // authoritative and observable.
    manager.Update(0.0f);
    manager.EndFrame();
    manager.EmitRuntimeFrameBarrier(Barrier::RenderExtraction);
    assert(beginCount == 0);
    assert(editorUpdateCount == 0);
    assert(endCount == 0);
    assert((observed == std::vector<Barrier>{Barrier::FinalTransformResolve, Barrier::AnimationTimeline,
                                             Barrier::PendingDestroy, Barrier::RenderExtraction}));
    observed.clear();

    manager.SetRuntimeLifecycleWorkAvailable(true);
    // Production begin_frame publishes this immutable summary before the
    // native driver dispatches any phase callback.
    manager.SetRuntimeLifecyclePlan(1, 1, 1, 0, 1);
    manager.Update(0.0f);
    assert(beginCount == 1);
    assert(editorUpdateCount == 1);
    assert(endCount == 0);

    manager.EndFrame();
    manager.EmitRuntimeFrameBarrier(Barrier::RenderExtraction);
    manager.EmitRuntimeFrameBarrier(Barrier::RenderGraph);
    manager.EmitRuntimeFrameBarrier(Barrier::SnapshotPublication);
    assert((observed == std::vector<Barrier>{Barrier::FinalTransformResolve, Barrier::AnimationTimeline,
                                             Barrier::PendingDestroy, Barrier::RenderExtraction, Barrier::RenderGraph,
                                             Barrier::SnapshotPublication}));
    assert(endCount == 1);

    // Exercise the empty-scene simulation and publisher completion protocol.
    // This data-only publisher accepts a complete F; it does not pretend to
    // record a renderer graph or generate its separate Host notifications.
    observed.clear();
    manager.SetRuntimeFramePublicationCallbacks([](infernux::RuntimeWorld &) { return true; }, {});
    const auto playRuntimeScene = [&manager] {
        auto &world = manager.BeginRuntimeWorld();
        {
            const infernux::WorldExecutionScope scope(&world);
            manager.CreateScene("RuntimeFrameBarrierTestsRuntime");
            const bool published = manager.PublishRuntimeWorld(world.id);
            assert(published);
        }
        manager.SetRuntimeLifecycleWorkAvailable(true);
        manager.SetRuntimeLifecyclePlan(1, 1, 1, 0, 1);
        manager.Play();
    };
    playRuntimeScene();
    assert(manager.GetRuntimeFrameCount() == 0);
    assert(manager.ConsumeFrameDeltaTime(0.25f) == 0.0f);
    assert(manager.ConsumeFrameDeltaTime(0.25f) == 0.25f);
    manager.Update(manager.GetFixedTimeStep());
    assert(manager.GetRuntimeFrameCount() == 0);
    manager.LateUpdate(manager.GetFixedTimeStep());
    manager.EndFrame();
    assert(manager.GetRuntimeFrameCount() == 1);
    assert(fixedUpdateCount == 1);
    assert(updateCount == 1);
    assert(lateUpdateCount == 1);
    assert((observed == std::vector<Barrier>{
                            Barrier::TransformToPhysics,
                            Barrier::PhysicsSimulation,
                            Barrier::PhysicsToTransform,
                            Barrier::TransformResolve,
                            Barrier::AnimationEvaluation,
                            Barrier::FinalTransformResolve,
                            Barrier::AnimationTimeline,
                            Barrier::PendingDestroy,
                            Barrier::SnapshotPublication,
                        }));
    assert(endCount == 2);
    manager.Stop();
    const bool retired1 = manager.CollectRetiredRuntimeWorlds();
    assert(retired1);
    assert(manager.GetRuntimeFrameCount() == 0);

    // The production accumulator may run zero or multiple fixed steps in one
    // rendered frame. A long frame is clamped once, time scale changes the
    // real-time frequency. Paused Step admits one new logical frame through
    // that same accumulator, including zero or multiple fixed steps.
    observed.clear();
    beginCount = 0;
    fixedUpdateCount = 0;
    updateCount = 0;
    lateUpdateCount = 0;
    endCount = 0;
    manager.SetFixedTimeStep(0.02f);
    manager.SetMaxFixedDeltaTime(0.05f);
    manager.SetTimeScale(1.0f);
    const auto runFrame = [&manager](float deltaTime) {
        manager.Update(deltaTime);
        manager.LateUpdate(deltaTime);
        manager.EndFrame();
    };

    playRuntimeScene();
    runFrame(0.01f);
    assert(fixedUpdateCount == 0);
    runFrame(0.01f);
    assert(fixedUpdateCount == 1);
    runFrame(1.0f);
    assert(fixedUpdateCount == 3);
    assert(manager.GetFixedTime() > 0.059 && manager.GetFixedTime() < 0.061);
    manager.Stop();
    const bool retired2 = manager.CollectRetiredRuntimeWorlds();
    assert(retired2);

    fixedUpdateCount = 0;
    manager.SetTimeScale(0.0f);
    playRuntimeScene();
    runFrame(1.0f);
    assert(fixedUpdateCount == 0);
    assert(manager.GetFixedTime() == 0.0);

    manager.SetTimeScale(2.0f);
    runFrame(0.01f);
    assert(fixedUpdateCount == 1);
    assert(manager.GetFixedTime() > 0.019 && manager.GetFixedTime() < 0.021);
    assert(manager.GetFixedUnscaledTime() > 0.009 && manager.GetFixedUnscaledTime() < 0.011);
    manager.Pause();
    runFrame(1.0f);
    assert(fixedUpdateCount == 1);
    manager.Step(0.016f);
    assert(fixedUpdateCount == 1); // Admission does not execute callbacks.
    runFrame(0.0f);
    assert(fixedUpdateCount == 2);
    assert(manager.GetFixedTime() > 0.039 && manager.GetFixedTime() < 0.041);
    assert(manager.GetFixedUnscaledTime() > 0.019 && manager.GetFixedUnscaledTime() < 0.021);
    const auto steppedFrame = manager.GetRuntimeFrameCount();
    manager.Step(0.0f);
    runFrame(0.0f);
    assert(manager.GetRuntimeFrameCount() == steppedFrame + 1);
    assert(fixedUpdateCount == 2);
    manager.Step(0.016f);
    runFrame(0.0f);
    assert(manager.GetRuntimeFrameCount() == steppedFrame + 2);
    assert(fixedUpdateCount == 4);
    assert(manager.GetFixedTime() > 0.079 && manager.GetFixedTime() < 0.081);
    assert(manager.GetFixedUnscaledTime() > 0.039 && manager.GetFixedUnscaledTime() < 0.041);
    manager.Stop();
    const bool retired3 = manager.CollectRetiredRuntimeWorlds();
    assert(retired3);
    manager.SetTimeScale(1.0f);
    manager.SetMaxFixedDeltaTime(1.0f / 3.0f);
    manager.SetRuntimeFramePublicationCallbacks({}, {});
    observed.clear();

    // Frame-cache commits retain mutation origin. A physics-authored root pose
    // must not feed the same Rigidbody back into transform-to-physics sync,
    // while descendants still observe their inherited world-space change.
    Scene *scene = manager.GetActiveScene();
    assert(scene);
    GameObject *root = scene->CreateGameObject("PhysicsPoseRoot");
    GameObject *child = scene->CreateGameObject("PhysicsPoseChild");
    child->SetParent(root, true);
    auto &transforms = TransformECSStore::Instance();
    std::vector<Transform *> invalidated;
    bool authoredRotationInvalidated = false;
    transforms.SetInvalidationObserver([&](Transform *transform, const Transform *physicsSource,
                                           const Transform *authoredSource, bool rotationAuthored) {
        invalidated.push_back(transform);
        authoredRotationInvalidated |= !physicsSource && authoredSource && rotationAuthored;
    });
    transforms.SyncSceneWorldMatrices(scene);
    transforms.BeginFrameCache();
    transforms.SetCachedWorldPoseFromPhysics(root->GetTransform()->GetECSHandle().index, glm::vec3(1.0f, 2.0f, 3.0f),
                                             glm::quat(1.0f, 0.0f, 0.0f, 0.0f), true);
    const uint64_t revisionBeforePhysicsPose = manager.GetRenderTransformRevision();
    const bool publishedPhysicsPose = transforms.EndFrameCache();
    assert(publishedPhysicsPose);
    manager.PublishPhysicsTransformsToRenderer();
    assert(manager.GetRenderTransformRevision() != revisionBeforePhysicsPose);
    assert((invalidated == std::vector<Transform *>{child->GetTransform()}));

    invalidated.clear();
    transforms.BeginFrameCache();
    root->GetTransform()->SetPosition(glm::vec3(2.0f, 3.0f, 4.0f));
    assert(!transforms.EndFrameCache());
    assert((invalidated == std::vector<Transform *>{root->GetTransform(), child->GetTransform()}));

    // A physical rotation alongside an authored position clears rotation
    // caches without granting an authored rotation to children.
    invalidated.clear();
    transforms.BeginFrameCache();
    root->GetTransform()->SetPosition(glm::vec3(3.0f, 4.0f, 5.0f));
    transforms.SetCachedWorldPoseFromPhysics(root->GetTransform()->GetECSHandle().index, glm::vec3(3.0f, 4.0f, 5.0f),
                                             glm::angleAxis(0.3f, glm::vec3(0, 1, 0)), true);
    assert(transforms.EndFrameCache());
    assert(!authoredRotationInvalidated);
    assert((invalidated == std::vector<Transform *>{root->GetTransform(), child->GetTransform()}));

    // Runtime authoring and bulk Instantiate may allocate transforms after
    // BeginFrameCache(). Every frame-cache array must grow in lockstep so the
    // new slot can be written and committed in the same frame.
    invalidated.clear();
    transforms.BeginFrameCache();
    GameObject *runtimeCreated = scene->CreateGameObject("CreatedDuringFrameCache");
    runtimeCreated->GetTransform()->SetPosition(glm::vec3(7.0f, 8.0f, 9.0f));
    assert(!transforms.EndFrameCache());
    assert(runtimeCreated->GetTransform()->GetPosition() == glm::vec3(7.0f, 8.0f, 9.0f));
    assert((invalidated == std::vector<Transform *>{runtimeCreated->GetTransform()}));

    transforms.SetInvalidationObserver([](Transform *transform, const Transform *physicsSource,
                                          const Transform *authoredSource, bool rotationAuthored) {
        auto *gameObject = transform ? transform->GetGameObject() : nullptr;
        if (gameObject)
            PhysicsECSStore::Instance().MarkGameObjectDirty(gameObject, physicsSource, authoredSource,
                                                            rotationAuthored);
    });

    TestTransformHierarchyPublication(*scene);
    TestTransformWorldBatchPublication(*scene);

    manager.ClearRuntimeLifecycleCallbacks();
    manager.EmitRuntimeFrameBarrier(Barrier::SnapshotPublication);
    assert(observed.empty());

    int selfClearingBarriers = 0;
    manager.SetRuntimeFrameBarrierCallback([&](Barrier) {
        ++selfClearingBarriers;
        manager.SetRuntimeFrameBarrierCallback({});
    });
    manager.EmitRuntimeFrameBarrier(Barrier::RenderGraph);
    manager.EmitRuntimeFrameBarrier(Barrier::SnapshotPublication);
    assert(selfClearingBarriers == 1);
    int selfClearingEnds = 0;
    manager.SetRuntimeLifecycleCallbacks([] {}, [](float) {}, [](float) {}, [](float) {}, [](float) {},
                                         [&] {
                                             ++selfClearingEnds;
                                             manager.ClearRuntimeLifecycleCallbacks();
                                         });
    manager.SetRuntimeLifecycleWorkAvailable(true);
    manager.SetRuntimeLifecyclePlan(0, 0, 0, 0, 0);
    manager.Update(0.0f);
    manager.EndFrame();
    manager.EndFrame();
    assert(selfClearingEnds == 1);
    manager.UnloadAllScenes();
    try {
        TestBroadphaseQueuePublication();
        PhysicsPublicationGpuFixture physics;
        const auto initializePhysics = [&] { physics.InitializePhysics(); };
        const auto shutdownPhysics = [&] { physics.ShutdownPhysics(); };
        TestCompoundColliderPublication(initializePhysics, shutdownPhysics);
        TestPhysicsQueryDefaultMask(initializePhysics, shutdownPhysics);
    } catch (const std::exception &error) {
        std::cerr << "Physics publication test failed: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
