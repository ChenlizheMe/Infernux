#pragma once

#include <function/scene/GameObject.h>
#include <function/scene/Scene.h>
#include <function/scene/Transform.h>
#include <function/scene/TransformECSStore.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

inline void TestTransformHierarchyPublication(infernux::Scene &scene)
{
    using namespace infernux;
    auto &store = TransformECSStore::Instance();
    std::array<GameObject *, 3> objects{};
    for (size_t i = 0; i < objects.size(); ++i)
        objects[i] = scene.CreateGameObject("CommitOrder" + std::to_string(i));
    objects[1]->SetParent(objects[0], false);
    auto *spacer = scene.CreateGameObject("CommitIntermediary");
    spacer->SetParent(objects[1], false);
    spacer->GetTransform()->SetLocalPosition(glm::vec3(0.0f, 2.0f, 0.0f));
    objects[2]->SetParent(spacer, false);
    auto *follower = scene.CreateGameObject("CommitFollower");
    follower->SetParent(objects[0], false);
    follower->GetTransform()->SetLocalPosition(glm::vec3(1.0f, 2.0f, 3.0f));

    const std::array<glm::vec3, 3> targets{glm::vec3(2.0f, 3.0f, 4.0f), glm::vec3(-8.0f, 5.0f, 3.0f),
                                           glm::vec3(13.0f, 7.0f, -9.0f)};
    const std::array<glm::quat, 3> rotations{glm::angleAxis(0.7f, glm::vec3(0.0f, 0.0f, 1.0f)),
                                             glm::angleAxis(0.4f, glm::vec3(0.0f, 1.0f, 0.0f)),
                                             glm::angleAxis(-0.3f, glm::vec3(1.0f, 0.0f, 0.0f))};
    size_t cases = 0;
    for (const bool authored : {false, true}) {
        std::array<int, 3> order{0, 1, 2};
        do {
            for (auto *object : objects) {
                object->GetTransform()->SetLocalPosition(glm::vec3(0.0f));
                object->GetTransform()->SetLocalRotation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
            }
            store.BeginFrameCache();
            for (const int i : order) {
                auto *transform = objects[i]->GetTransform();
                if (authored) {
                    transform->SetWorldPosition(targets[i]);
                    transform->SetWorldRotation(rotations[i]);
                } else {
                    store.SetCachedWorldPoseFromPhysics(transform->GetECSHandle().index, targets[i], rotations[i],
                                                        true);
                }
            }
            if (store.EndFrameCache() == authored)
                throw std::runtime_error("frame publication lost mutation origin");
            for (size_t i = 0; i < objects.size(); ++i) {
                const auto *transform = objects[i]->GetTransform();
                if (glm::length(transform->GetWorldPosition() - targets[i]) > 2e-5f ||
                    std::abs(glm::dot(transform->GetWorldRotation(), rotations[i])) < 1.0f - 2e-5f)
                    throw std::runtime_error("world pose depends on frame-cache write order");
            }
            const glm::vec3 expectedFollower = objects[0]->GetTransform()->TransformPoint(glm::vec3(1, 2, 3));
            if (glm::length(follower->GetTransform()->GetWorldPosition() - expectedFollower) > 2e-5f)
                throw std::runtime_error("ordinary child no longer follows its parent");
            ++cases;
        } while (std::next_permutation(order.begin(), order.end()));
    }
    std::cout << "HIERARCHY_COMMIT permutations=" << cases << " intermediary=passed follower=passed\n";
}
