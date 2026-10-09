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

    // Physics publishes world channels before Update/Timeline writes local
    // channels. The latest write wins per channel, both now and after commit.
    auto *localOwner = scene.CreateGameObject("LocalAfterPhysics");
    auto *localParent = scene.CreateGameObject("LocalAfterPhysicsParent");
    localParent->GetTransform()->SetLocalTRS(glm::vec3(10, 4, -3), glm::vec3(0, 25, 0), glm::vec3(2));
    auto *tf = localOwner->GetTransform();
    const glm::vec3 solverPosition(7, 8, 9), localPosition(1, 2, 3), localEuler(0, 37, 0);
    const glm::quat solverRotation = glm::angleAxis(glm::radians(22.0f), glm::vec3(0, 1, 0));
    const glm::quat authoredRotation = glm::angleAxis(glm::radians(localEuler.y), glm::vec3(0, 1, 0));
    const auto checkPose = [&](const glm::vec3 &position, const glm::quat &rotation) {
        const glm::quat actual = tf->GetWorldRotation();
        const glm::quat aligned = glm::dot(actual, rotation) < 0 ? -rotation : rotation;
        const glm::quat delta = actual - aligned;
        if (glm::length(tf->GetPosition() - position) > 2e-5f || glm::dot(delta, delta) > 1e-10f)
            throw std::runtime_error("local authoring was overwritten by a cached physics pose");
        const auto &matrix = tf->GetWorldMatrix();
        if (glm::length(glm::vec3(matrix[3]) - position) > 2e-5f)
            throw std::runtime_error("mixed local/world pose matrix has a stale position");
        const glm::vec3 forward = glm::normalize(glm::vec3(matrix[2]));
        if (glm::length(forward - rotation * glm::vec3(0, 0, 1)) > 2e-5f)
            throw std::runtime_error("mixed local/world pose matrix has a stale rotation");
    };
    for (const bool nested : {false, true}) {
        localOwner->SetParent(nested ? localParent : nullptr, false);
        const auto *parent = tf->GetParent();
        for (int method = 0; method != 13; ++method) {
            tf->SetLocalTRS(glm::vec3(0), glm::vec3(0), glm::vec3(1));
            store.BeginFrameCache();
            store.SetCachedWorldPoseFromPhysics(tf->GetECSHandle().index, solverPosition, solverRotation, true);
            const glm::vec3 expectedLocal = parent
                ? glm::vec3(glm::inverse(parent->GetWorldMatrix()) * glm::vec4(solverPosition, 1)) : solverPosition;
            const glm::quat expectedLocalRotation = parent
                ? glm::inverse(parent->GetWorldRotation()) * solverRotation : solverRotation;
            if (glm::length(tf->GetLocalPosition() - expectedLocal) > 2e-5f ||
                glm::length(tf->GetLocalRotation() * glm::vec3(0, 0, 1) - expectedLocalRotation * glm::vec3(0, 0, 1)) > 2e-5f)
                throw std::runtime_error("local reads did not observe the current physics pose");
            Transform *batch[] = {tf};
            float gathered[4]{};
            store.GatherLocalPositions(batch, gathered, 1);
            if (glm::length(glm::vec3(gathered[0], gathered[1], gathered[2]) - expectedLocal) > 2e-5f)
                throw std::runtime_error("batch local position read did not observe physics");
            store.GatherLocalRotations(batch, gathered, 1);
            if (glm::length(glm::quat(gathered[3], gathered[0], gathered[1], gathered[2]) * glm::vec3(0, 0, 1)
                            - expectedLocalRotation * glm::vec3(0, 0, 1)) > 2e-5f)
                throw std::runtime_error("batch local rotation read did not observe physics");
            glm::vec3 expectedPosition = solverPosition;
            glm::quat expectedRotation = solverRotation;
            if (method == 0 || method == 3) {
                expectedPosition = parent ? parent->TransformPoint(localPosition) : localPosition;
                if (method == 0)
                    tf->SetLocalPosition(localPosition);
            }
            if (method >= 1 && method <= 3) {
                expectedRotation = (parent ? parent->GetWorldRotation() : glm::quat(1, 0, 0, 0)) * authoredRotation;
                if (method == 1) tf->SetLocalEulerAngles(localEuler);
                if (method == 2) tf->SetLocalRotation(authoredRotation);
                if (method == 3) tf->SetLocalTRS(localPosition, localEuler, glm::vec3(1));
            }
            if (method == 4 || method == 5) {
                expectedRotation = solverRotation * authoredRotation;
                if (method == 4) tf->Rotate(localEuler);
                else tf->Rotate(glm::vec3(0, 1, 0), 37.0f);
            }
            if (method == 6 || method == 10) {
                expectedPosition = parent ? parent->TransformPoint(localPosition) : localPosition;
                if (method == 6) store.ScatterLocalPositions(batch, &localPosition.x, 1);
                else store.SetCachedLocalPosition(tf->GetECSHandle().index, localPosition);
            }
            if (method == 7 || method == 8 || method == 11 || method == 12) {
                const auto localRotation = authoredRotation;
                expectedRotation = (parent ? parent->GetWorldRotation() : glm::quat(1, 0, 0, 0)) * localRotation;
                const float values[] = {localRotation.x, localRotation.y, localRotation.z, localRotation.w};
                if (method == 7) store.ScatterLocalRotations(batch, values, 1);
                if (method == 8) store.ScatterLocalEulerAngles(batch, &localEuler.x, 1);
                if (method == 11) store.SetCachedLocalRotation(tf->GetECSHandle().index, localRotation);
                if (method == 12) store.SetCachedLocalEulerAngles(tf->GetECSHandle().index, localEuler);
            }
            if (method == 9) {
                expectedPosition = glm::vec3(-3, 11, 7);
                store.ScatterWorldPositions(batch, &expectedPosition.x, 1);
            }
            checkPose(expectedPosition, expectedRotation);
            if (!store.EndFrameCache())
                throw std::runtime_error("authored override lost physics publication provenance");
            checkPose(expectedPosition, expectedRotation);
        }
    }
    std::cout << "LOCAL_AFTER_PHYSICS root/nested setters=passed reads=passed\n";
}
