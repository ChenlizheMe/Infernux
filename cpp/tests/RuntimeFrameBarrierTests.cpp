#include <function/scene/GameObject.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <function/scene/Transform.h>
#include <function/scene/TransformECSStore.h>
#include <function/scene/physics/PhysicsECSStore.h>

#include <cassert>
#include <vector>

using infernux::GameObject;
using infernux::PhysicsECSStore;
using infernux::Scene;
using infernux::SceneManager;
using infernux::Transform;
using infernux::TransformECSStore;

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
    manager.EmitRuntimeFrameBarrier(Barrier::RenderExtraction);
    manager.EndFrame();
    assert(beginCount == 0);
    assert(editorUpdateCount == 0);
    assert(endCount == 0);
    assert((observed == std::vector<Barrier>{Barrier::RenderExtraction, Barrier::PendingDestroy}));
    observed.clear();

    manager.SetRuntimeLifecycleWorkAvailable(true);
    // Production begin_frame publishes this immutable summary before the
    // native driver dispatches any phase callback.
    manager.SetRuntimeLifecyclePlan(1, 1, 1, 1);
    manager.Update(0.0f);
    assert(beginCount == 1);
    assert(editorUpdateCount == 1);
    assert(endCount == 0);

    manager.EmitRuntimeFrameBarrier(Barrier::RenderExtraction);
    manager.EmitRuntimeFrameBarrier(Barrier::RenderGraph);
    manager.EmitRuntimeFrameBarrier(Barrier::SnapshotPublication);
    manager.EndFrame();
    assert((observed == std::vector<Barrier>{Barrier::RenderExtraction, Barrier::RenderGraph,
                                             Barrier::SnapshotPublication, Barrier::PendingDestroy}));
    assert(endCount == 1);

    // Exercise the real empty-scene production flow. Physics work is skipped,
    // but its fixed-step boundaries remain observable and no extra transform
    // synchronization is introduced merely to emit a barrier.
    observed.clear();
    manager.Play();
    assert(manager.GetRuntimeFrameCount() == 0);
    manager.Update(manager.GetFixedTimeStep());
    assert(manager.GetRuntimeFrameCount() == 1);
    manager.LateUpdate(manager.GetFixedTimeStep());
    manager.EmitRuntimeFrameBarrier(Barrier::FinalTransformResolve);
    manager.EmitRuntimeFrameBarrier(Barrier::AnimationTimeline);
    manager.EmitRuntimeFrameBarrier(Barrier::RenderExtraction);
    manager.EmitRuntimeFrameBarrier(Barrier::RenderGraph);
    manager.EmitRuntimeFrameBarrier(Barrier::SnapshotPublication);
    manager.EndFrame();
    assert(fixedUpdateCount == 1);
    assert(updateCount == 1);
    assert(lateUpdateCount == 1);
    assert((observed == std::vector<Barrier>{
                            Barrier::TransformToPhysics,
                            Barrier::PhysicsSimulation,
                            Barrier::PhysicsToTransform,
                            Barrier::TransformResolve,
                            Barrier::FinalTransformResolve,
                            Barrier::AnimationTimeline,
                            Barrier::RenderExtraction,
                            Barrier::RenderGraph,
                            Barrier::SnapshotPublication,
                            Barrier::PendingDestroy,
                        }));
    assert(endCount == 2);
    manager.Stop();
    assert(manager.GetRuntimeFrameCount() == 0);

    // The production accumulator may run zero or multiple fixed steps in one
    // rendered frame. A long frame is clamped once, time scale changes the
    // real-time frequency, and paused Step advances exactly one fixed step.
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

    manager.Play();
    runFrame(0.01f);
    assert(fixedUpdateCount == 0);
    runFrame(0.01f);
    assert(fixedUpdateCount == 1);
    runFrame(1.0f);
    assert(fixedUpdateCount == 3);
    assert(manager.GetFixedTime() > 0.059 && manager.GetFixedTime() < 0.061);
    manager.Stop();

    fixedUpdateCount = 0;
    manager.SetTimeScale(0.0f);
    manager.Play();
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
    manager.EndFrame();
    assert(fixedUpdateCount == 2);
    assert(manager.GetFixedTime() > 0.039 && manager.GetFixedTime() < 0.041);
    assert(manager.GetFixedUnscaledTime() > 0.019 && manager.GetFixedUnscaledTime() < 0.021);
    manager.Stop();
    manager.SetTimeScale(1.0f);
    manager.SetMaxFixedDeltaTime(1.0f / 3.0f);
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
    transforms.SetInvalidationObserver([&invalidated](Transform *transform) { invalidated.push_back(transform); });
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

    transforms.SetInvalidationObserver([](Transform *transform) {
        auto *gameObject = transform ? transform->GetGameObject() : nullptr;
        if (gameObject)
            PhysicsECSStore::Instance().MarkGameObjectDirty(gameObject);
    });

    manager.ClearRuntimeLifecycleCallbacks();
    manager.EmitRuntimeFrameBarrier(Barrier::SnapshotPublication);
    assert(observed.empty());
    manager.UnloadAllScenes();
    return 0;
}
