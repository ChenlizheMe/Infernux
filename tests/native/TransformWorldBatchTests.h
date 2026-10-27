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
#include <string>

inline void TestTransformWorldBatchPublication(infernux::Scene &scene)
{
    using namespace infernux;
    auto &store = TransformECSStore::Instance();
    const auto rotation = [](const glm::vec3 &euler) {
        return glm::angleAxis(glm::radians(euler.y), glm::vec3(0, 1, 0)) *
               glm::angleAxis(glm::radians(euler.x), glm::vec3(1, 0, 0)) *
               glm::angleAxis(glm::radians(euler.z), glm::vec3(0, 0, 1));
    };
    const auto sameRotation = [](const glm::quat &actual, const glm::quat &expected) {
        return std::abs(glm::dot(glm::normalize(actual), glm::normalize(expected))) > 1.0f - 2e-5f;
    };
    std::array<GameObject *, 3> objects{};
    for (size_t i = 0; i < objects.size(); ++i)
        objects[i] = scene.CreateGameObject("WorldBatchTarget" + std::to_string(i));
    auto *spacer = scene.CreateGameObject("WorldBatchUntargetedAncestor");
    spacer->SetParent(objects[0], false);
    spacer->GetTransform()->SetLocalTRS(glm::vec3(1, 2, -1), glm::vec3(13, 24, 8), glm::vec3(1.25f));
    objects[1]->SetParent(spacer, false);
    objects[2]->SetParent(objects[1], false);
    auto *follower = scene.CreateGameObject("WorldBatchUntargetedFollower");
    follower->SetParent(spacer, false);
    follower->GetTransform()->SetLocalTRS(glm::vec3(3, -2, 5), glm::vec3(4, 18, 32), glm::vec3(1));
    const glm::vec3 followerLocalPosition = follower->GetTransform()->GetLocalPosition();

    const std::array<glm::vec3, 3> positions{glm::vec3(3, 8, -2), glm::vec3(-5, 7, 13), glm::vec3(19, -3, 11)};
    // Include a non-canonical Euler representation to check that authored
    // world Euler values survive the deferred parent/child commit as well.
    const std::array<glm::vec3, 3> eulers{glm::vec3(120, 330, 15), glm::vec3(-25, 58, 29), glm::vec3(31, -40, 74)};
    const std::array<glm::quat, 3> rotations{rotation(eulers[0]), rotation(eulers[1]), rotation(eulers[2])};
    size_t cases = 0;
    for (const bool cached : {false, true}) {
        for (int property = 0; property != 3; ++property) {
            std::array<int, 3> order{0, 1, 2};
            do {
                for (size_t i = 0; i < objects.size(); ++i)
                    objects[i]->GetTransform()->SetLocalTRS(glm::vec3(float(i + 1), 2, 3), glm::vec3(5, 10, 15),
                                                            glm::vec3(1));
                // Prime both matrix and exact Euler caches on targets and
                // non-target ancestors, before entering the frame cache.
                for (GameObject *object : {objects[0], objects[1], objects[2], spacer, follower}) {
                    auto *transform = object->GetTransform();
                    transform->SetWorldEulerAngles(transform->GetWorldEulerAngles());
                    (void)transform->GetWorldMatrix();
                }
                const glm::quat spacerLocalRotation = spacer->GetTransform()->GetLocalRotation();
                const glm::quat followerLocalRotation = follower->GetTransform()->GetLocalRotation();

                // Duplicate a child before its final row and a root after its
                // earlier row. Sorting by hierarchy must preserve last wins.
                std::array<Transform *, 5> batch{objects[2]->GetTransform(), objects[order[0]]->GetTransform(),
                                                 objects[order[1]]->GetTransform(), objects[order[2]]->GetTransform(),
                                                 objects[0]->GetTransform()};
                const std::array<int, 5> indices{2, order[0], order[1], order[2], 0};
                std::array<float, 20> values{};
                const size_t columns = property == 1 ? 4 : 3;
                for (size_t row = 0; row < batch.size(); ++row) {
                    const int i = indices[row];
                    const bool superseded = row == 0 || (row != 4 && i == 0);
                    if (property == 1) {
                        const glm::quat value = superseded ? rotation(glm::vec3(9, 27, 35)) : rotations[i];
                        values[row * columns] = value.x;
                        values[row * columns + 1] = value.y;
                        values[row * columns + 2] = value.z;
                        values[row * columns + 3] = value.w;
                    } else {
                        const glm::vec3 value =
                            superseded ? glm::vec3(9, 27, 35) : (property == 0 ? positions[i] : eulers[i]);
                        for (size_t lane = 0; lane != 3; ++lane)
                            values[row * columns + lane] = value[lane];
                    }
                }
                const auto original = values;
                if (cached)
                    store.BeginFrameCache();
                const auto serialBeforeWrite = store.GetGlobalTransformSerial();
                if (property == 0)
                    store.ScatterWorldPositions(batch.data(), values.data(), batch.size());
                else if (property == 1)
                    store.ScatterWorldRotations(batch.data(), values.data(), batch.size());
                else
                    store.ScatterWorldEulerAngles(batch.data(), values.data(), batch.size());
                if (cached && store.GetGlobalTransformSerial() != serialBeforeWrite)
                    throw std::runtime_error("staged world batch advanced the committed serial before publication");

                const auto check = [&] {
                    for (size_t i = 0; i < objects.size(); ++i) {
                        const auto *transform = objects[i]->GetTransform();
                        if (property == 0 && glm::length(transform->GetWorldPosition() - positions[i]) > 1e-4f)
                            throw std::runtime_error("world position batch lost a final target value");
                        if (property != 0 && !sameRotation(transform->GetWorldRotation(), rotations[i]))
                            throw std::runtime_error("world rotation batch depends on target order");
                        if (property != 0 && !sameRotation(rotation(transform->GetWorldEulerAngles()), rotations[i]))
                            throw std::runtime_error("world batch read returned a stale exact Euler cache");
                        if (property == 2 &&
                            glm::length(transform->GetWorldEulerAngles() - glm::mod(eulers[i], glm::vec3(360))) > 1e-4f)
                            throw std::runtime_error("world Euler batch lost its explicit representation");
                    }
                    const auto *parent = spacer->GetTransform();
                    if (glm::length(follower->GetTransform()->GetWorldPosition() -
                                    parent->TransformPoint(followerLocalPosition)) > 1e-4f ||
                        !sameRotation(parent->GetWorldRotation(),
                                      objects[0]->GetTransform()->GetWorldRotation() * spacerLocalRotation) ||
                        !sameRotation(rotation(parent->GetWorldEulerAngles()), parent->GetWorldRotation()) ||
                        !sameRotation(follower->GetTransform()->GetWorldRotation(),
                                      parent->GetWorldRotation() * followerLocalRotation))
                        throw std::runtime_error("untargeted hierarchy did not follow the final batch parents");
                };
                check();
                if (cached) {
                    if (store.GetGlobalTransformSerial() != serialBeforeWrite)
                        throw std::runtime_error("reading staged world batch matrices advanced the committed serial");
                    if (store.EndFrameCache())
                        throw std::runtime_error("authored world batch acquired physics provenance");
                    if (store.GetGlobalTransformSerial() == serialBeforeWrite)
                        throw std::runtime_error("changed world batch did not publish the committed serial");
                    check();
                }
                if (values != original)
                    throw std::runtime_error("world batch modified its input array");
                ++cases;
            } while (std::next_permutation(order.begin(), order.end()));
        }
    }

    // Provisional cache dirtiness is separate from a committed mutation. In
    // particular a no-op ancestor can invalidate a child's matrix before the
    // child changes, and the child must publish whether or not a read cleared
    // that provisional dirtiness. A no-op frame must not hide a later local
    // write outside the cache either.
    auto *serialParent = scene.CreateGameObject("BatchSerialParent");
    auto *serialChild = scene.CreateGameObject("BatchSerialChild");
    serialChild->SetParent(serialParent, false);
    for (int property = 0; property != 3; ++property) {
        for (const bool readBeforeCommit : {false, true}) {
            const auto reset = [&] {
                serialParent->GetTransform()->SetLocalTRS(glm::vec3(0), glm::vec3(0), glm::vec3(1));
                serialChild->GetTransform()->SetLocalTRS(glm::vec3(2, 0, 0), glm::vec3(0), glm::vec3(1));
                store.SyncSceneWorldMatrices(&scene);
            };
            const auto scatter = [&](Transform *transform, bool changed) {
                Transform *batch[] = {transform};
                if (property == 0) {
                    const glm::vec3 value = changed ? glm::vec3(4, 5, 6) : glm::vec3(0);
                    store.ScatterWorldPositions(batch, &value.x, 1);
                } else if (property == 1) {
                    const glm::quat value = changed ? rotation(glm::vec3(0, 45, 0)) : glm::quat(1, 0, 0, 0);
                    const float values[] = {value.x, value.y, value.z, value.w};
                    store.ScatterWorldRotations(batch, values, 1);
                } else {
                    const glm::vec3 value = changed ? glm::vec3(0, 45, 0) : glm::vec3(0);
                    store.ScatterWorldEulerAngles(batch, &value.x, 1);
                }
            };
            reset();
            store.BeginFrameCache();
            const auto unchangedSerial = store.GetGlobalTransformSerial();
            scatter(serialParent->GetTransform(), false);
            if (readBeforeCommit)
                (void)serialChild->GetTransform()->GetWorldMatrix();
            if (store.GetGlobalTransformSerial() != unchangedSerial)
                throw std::runtime_error("no-op staged ancestor published a Transform change");
            (void)store.EndFrameCache();
            if (store.GetGlobalTransformSerial() != unchangedSerial)
                throw std::runtime_error("no-op world batch advanced the serial at commit");
            // Do not resolve matrices between EndFrame and this real write.
            serialChild->GetTransform()->SetLocalPosition(glm::vec3(7, 0, 0));
            if (store.GetGlobalTransformSerial() == unchangedSerial)
                throw std::runtime_error("provisional dirtiness hid a later local write outside the frame");

            reset();
            store.BeginFrameCache();
            const auto changedSerial = store.GetGlobalTransformSerial();
            scatter(serialParent->GetTransform(), false);
            scatter(serialChild->GetTransform(), true);
            if (readBeforeCommit)
                (void)serialChild->GetTransform()->GetWorldMatrix();
            if (store.GetGlobalTransformSerial() != changedSerial)
                throw std::runtime_error("staged child world batch published before EndFrame");
            (void)store.EndFrameCache();
            if (store.GetGlobalTransformSerial() == changedSerial)
                throw std::runtime_error("provisional descendant dirtiness hid its real world commit");
        }
    }
    // Local ancestor writes must not invalidate a descendant's still-explicit
    // world rotation or its authored Euler representation, including gimbal
    // representations for which quaternion round-tripping loses information.
    auto *localParent = scene.CreateGameObject("BatchMixedLocalParent");
    auto *worldChild = scene.CreateGameObject("BatchMixedWorldChild");
    worldChild->SetParent(localParent, false);
    for (int method = 0; method != 3; ++method) {
        for (const bool worldFirst : {false, true}) {
            for (const glm::vec3 target : {glm::vec3(90, 30, 40), glm::vec3(120, 330, 15)}) {
                localParent->GetTransform()->SetLocalTRS(glm::vec3(0), glm::vec3(0), glm::vec3(1));
                worldChild->GetTransform()->SetLocalTRS(glm::vec3(1, 2, 3), glm::vec3(0), glm::vec3(1));
                store.BeginFrameCache();
                const auto writeWorld = [&] { worldChild->GetTransform()->SetWorldEulerAngles(target); };
                const auto writeLocal = [&] {
                    auto *transform = localParent->GetTransform();
                    const glm::vec3 localEuler(30, 50, 20);
                    if (method == 0)
                        transform->SetLocalEulerAngles(localEuler);
                    else if (method == 1)
                        transform->SetLocalRotation(rotation(localEuler));
                    else
                        transform->SetLocalTRS(glm::vec3(3, -2, 4), localEuler, glm::vec3(1.25f));
                };
                if (worldFirst)
                    writeWorld();
                writeLocal();
                if (!worldFirst)
                    writeWorld();
                const auto checkMixed = [&] {
                    if (!sameRotation(worldChild->GetTransform()->GetWorldRotation(), rotation(target)) ||
                        glm::length(worldChild->GetTransform()->GetWorldEulerAngles() - target) > 1e-4f)
                        throw std::runtime_error("local ancestor write invalidated an explicit world Euler target");
                };
                checkMixed();
                (void)store.EndFrameCache();
                checkMixed();
            }
        }
    }
    auto *eulerRoot = scene.CreateGameObject("BatchRootEulerRepresentation");
    for (const bool cached : {false, true}) {
        eulerRoot->GetTransform()->SetLocalEulerAngles(glm::vec3(5, 10, 15));
        if (cached)
            store.BeginFrameCache();
        eulerRoot->GetTransform()->SetWorldEulerAngles(eulers[0]);
        const auto checkRootEuler = [&] {
            if (glm::length(eulerRoot->GetTransform()->GetLocalEulerAngles() - eulers[0]) > 1e-4f ||
                glm::length(eulerRoot->GetTransform()->GetWorldEulerAngles() - eulers[0]) > 1e-4f)
                throw std::runtime_error("root world/local Euler representation differs inside a frame");
        };
        checkRootEuler();
        if (cached) {
            (void)store.EndFrameCache();
            checkRootEuler();
        }
    }

    // A no-op quaternion override still pins the current world rotation, but
    // must preserve both the target's and inherited descendants' authored
    // Euler representations. Half-turn quaternions normalize bit-exactly.
    auto *noopParent = scene.CreateGameObject("BatchNoopEulerParent");
    auto *noopChild = scene.CreateGameObject("BatchNoopEulerChild");
    noopChild->SetParent(noopParent, false);
    noopParent->GetTransform()->SetWorldEulerAngles(glm::vec3(180, 0, 0));
    noopChild->GetTransform()->SetWorldEulerAngles(glm::vec3(90, 30, 40));
    store.SyncSceneWorldMatrices(&scene);
    store.BeginFrameCache();
    const auto noopSerial = store.GetGlobalTransformSerial();
    noopParent->GetTransform()->SetWorldRotation(noopParent->GetTransform()->GetWorldRotation());
    const auto checkNoopEuler = [&] {
        if (glm::length(noopParent->GetTransform()->GetWorldEulerAngles() - glm::vec3(180, 0, 0)) > 1e-4f ||
            glm::length(noopChild->GetTransform()->GetWorldEulerAngles() - glm::vec3(90, 30, 40)) > 1e-4f ||
            store.GetGlobalTransformSerial() != noopSerial)
            throw std::runtime_error("no-op quaternion write changed an authored Euler representation or serial");
    };
    checkNoopEuler();
    (void)store.EndFrameCache();
    checkNoopEuler();

    // A Scene move changes batch identity without allocating or freeing any
    // Transform storage. Descendant ownership must advance the same revision.
    Scene destination("WorldBatchIdentityDestination");
    auto *moving = scene.CreateGameObject("WorldBatchIdentityRoot");
    auto *child = scene.CreateGameObject("WorldBatchIdentityChild");
    child->SetParent(moving, false);
    const auto handle = child->GetTransform()->GetECSHandle();
    const auto world = child->GetHandle().worldId;
    const auto revision = store.GetStructuralVersion();
    if (!scene.TransferRootObjectTo(moving, destination) || store.GetStructuralVersion() <= revision ||
        child->GetHandle().worldId == world || child->GetTransform()->GetECSHandle() != handle)
        throw std::runtime_error("Scene transfer did not invalidate cached Transform batch identity");
    std::cout << "WORLD_BATCH cases=" << cases << " duplicates=passed intermediate=passed identity=passed\n";
}
