#pragma once

#include <core/config/EngineConfig.h>
#include <core/threading/JobSystem.h>
#include <function/scene/BoxCollider.h>
#include <function/scene/GameObject.h>
#include <function/scene/Rigidbody.h>
#include <function/scene/SceneManager.h>
#include <function/scene/Transform.h>
#include <function/scene/physics/PhysicsWorld.h>
#include <array>
#include <iostream>
#include <stdexcept>

inline void TestPhysicsQueryDefaultMask()
{
    using namespace infernux;
    auto &world = PhysicsWorld::Instance();
    auto &manager = SceneManager::Instance();
    auto &config = EngineConfig::Get();
    const uint32_t original = config.defaultQueryLayerMask;
    JobSystem::Initialize(2);
    const auto clean = [&] {
        manager.UnloadAllScenes();
        world.Shutdown();
    };
    try {
        for (uint32_t mask : {0u, 4u, 0x80000000u, 0xFFFFFFFBu}) {
            config.defaultQueryLayerMask = mask;
            world.Initialize();
            auto *scene = manager.CreateScene("Configured native query masks");
            std::array<Collider *, 3> colliders{};
            const std::array<int, 3> layers{0, 2, 31};
            size_t expectedCount = 0;
            Collider *expectedClosest = nullptr;
            for (size_t i = 0; i < layers.size(); ++i) {
                auto *owner = scene->CreateGameObject("Layer actor");
                owner->SetLayer(layers[i]);
                owner->GetTransform()->SetPosition(glm::vec3(0, float(i) * 2, 0));
                colliders[i] = owner->AddComponent<BoxCollider>();
                owner->AddComponent<Rigidbody>()->SetUseGravity(false);
                if (mask & (1u << layers[i])) {
                    ++expectedCount;
                    expectedClosest = colliders[i];
                }
            }
            manager.SyncTransforms();
            const glm::vec3 origin(0, 8, 0), down(0, -1, 0), center(0, 2, 0), half(2, 6, 2);
            const glm::quat identity(1, 0, 0, 0);
            RaycastHit hit;
            const auto checkHit = [&](bool found) {
                if (found != (expectedClosest != nullptr) || (found && hit.collider != expectedClosest))
                    throw std::runtime_error("native default mask selected the wrong collider");
            };
            const auto checkCount = [&](size_t count) {
                if (count != expectedCount)
                    throw std::runtime_error("native default mask selected the wrong body count");
            };
            checkHit(world.Raycast(origin, down, 12, hit));
            checkHit(world.SphereCast(origin, .25f, down, 12, hit));
            checkHit(world.BoxCast(origin, glm::vec3(.25f), down, identity, 12, hit));
            checkHit(world.CapsuleCast(glm::vec3(0, 7.75f, 0), glm::vec3(0, 8.25f, 0), .25f, down, 12, hit));
            checkCount(world.RaycastAll(origin, down, 12).size());
            checkCount(world.OverlapSphere(center, 6).size());
            checkCount(world.OverlapBox(center, half).size());
            checkCount(world.OverlapCapsule(glm::vec3(0), glm::vec3(0, 4, 0), 2).size());
            checkCount(world.QueryRigidbodiesInBounds(glm::vec3(-2), glm::vec3(2, 6, 2)).size());
            const float starts[]{0, 8, 0}, directions[]{0, -1, 0};
            uint8_t found = 0;
            world.RaycastBatch(starts, directions, 1, 12, &hit, &found);
            checkHit(found != 0);
            world.RaycastBatchPublished(starts, directions, 1, 12, &hit, &found);
            checkHit(found != 0);
            if (world.Raycast(origin, down, 12, hit, 0u) ||
                !world.Raycast(origin, down, 12, hit, 4u) || hit.collider != colliders[1])
                throw std::runtime_error("explicit native mask was replaced by the default");
            clean();
        }
    } catch (...) {
        clean();
        config.defaultQueryLayerMask = original;
        JobSystem::Shutdown();
        throw;
    }
    config.defaultQueryLayerMask = original;
    JobSystem::Shutdown();
    std::cout << "QUERY_DEFAULT_MASK configurations=4 all-query-families=passed\n";
}
