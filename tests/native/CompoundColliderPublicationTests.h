#pragma once

#include <core/threading/JobSystem.h>
#include <function/scene/BoxCollider.h>
#include <function/scene/CapsuleCollider.h>
#include <function/scene/SceneManager.h>
#include <function/scene/SphereCollider.h>
#include <function/scene/physics/PhysicsWorld.h>
#include <iostream>
#include <stdexcept>

inline void TestCompoundColliderPublication()
{
    using namespace infernux;
    auto &world = PhysicsWorld::Instance();
    auto &manager = SceneManager::Instance();
    JobSystem::Initialize(2);
    world.Initialize();
    Scene *scene = manager.CreateScene("Compound publication native stress");
    const auto expect = [&](float x, bool triggers, Collider *expected) {
        manager.SyncTransforms();
        RaycastHit hit;
        const bool found = world.Raycast(glm::vec3(x, 3, 0), glm::vec3(0, -1, 0), 8, hit, 0xFFFFFFFF, triggers);
        if (found != (expected != nullptr) || (found && hit.collider != expected))
            throw std::runtime_error("compound shape/sensor/member identity publication mismatch");
    };
    constexpr int cycles = 256;
    for (int cycle = 0; cycle < cycles; ++cycle) {
        auto *owner = scene->CreateGameObject("Transient compound");
        auto *solid = owner->AddComponent<BoxCollider>();
        auto *sensor = owner->AddComponent<SphereCollider>();
        auto *third = owner->AddComponent<CapsuleCollider>();
        sensor->SetCenter(glm::vec3(6, 0, 0));
        third->SetCenter(glm::vec3(12, 0, 0));
        sensor->SetIsTrigger(true);
        solid->SetEnabled(false);
        third->SetEnabled(false);
        expect(6, true, sensor);
        expect(6, false, nullptr);
        solid->SetEnabled(true);
        expect(0, false, solid);
        solid->SetEnabled(false);
        third->SetEnabled(true);
        expect(12, false, third);
        if (!owner->RemoveComponent(third))
            throw std::runtime_error("compound member removal failed");
        sensor->SetIsTrigger(false);
        expect(6, false, sensor);
        sensor->SetEnabled(false);
        expect(6, true, nullptr);
        solid->SetEnabled(true);
        expect(0, false, solid);
        if (!owner->RemoveComponent(solid))
            throw std::runtime_error("last active compound member removal failed");
        expect(0, true, nullptr);
        if ((cycle % 2) != 0) {
            sensor->SetEnabled(true);
            owner->AddComponent<BoxCollider>();
            expect(6, false, sensor);
        }
        scene->DestroyGameObject(owner);
        scene->ProcessPendingDestroys();
        if (world.GetBodyCount() != 0)
            throw std::runtime_error("compound teardown left an allocated physics body");
    }
    manager.UnloadAllScenes();
    world.Shutdown();
    JobSystem::Shutdown();
    std::cout << "COMPOUND_PUBLICATION cycles=" << cycles << " identities=sensor+solid teardown=passed\n";
}
