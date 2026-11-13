#pragma once

#include <function/scene/RuntimeWorld.h>
#include <function/scene/physics/PhysicsECSStore.h>
#include <iostream>
#include <stdexcept>
#include <vector>

inline void TestBroadphaseQueuePublication()
{
    auto &store = infernux::PhysicsECSStore::Instance();
    infernux::RuntimeWorld world;
    infernux::RuntimeWorld other;
    store.QueueBroadphaseAdd(other, 200);
    store.QueueBroadphaseRemove(other, 201);
    if (!store.ConsumePendingBroadphaseAdds(world).empty() || !store.ConsumePendingBroadphaseRemoves(world).empty())
        throw std::runtime_error("unexpected pending bodies before queue test");
    for (int cycle = 0; cycle < 128; ++cycle) {
        for (int repeat = 0; repeat < 32; ++repeat) {
            store.QueueBroadphaseAdd(world, 100);
            store.QueueBroadphaseAdd(world, 101);
            if (!store.CancelBroadphaseAdd(world, 100))
                throw std::runtime_error("add request was lost");
            store.QueueBroadphaseAdd(world, 100);
            store.QueueBroadphaseRemove(world, 100);
            store.QueueBroadphaseRemove(world, 101);
            if (!store.CancelBroadphaseRemove(world, 100))
                throw std::runtime_error("remove request was lost");
            store.QueueBroadphaseRemove(world, 100);
        }
        store.QueueBroadphaseAdd(world, 102);
        store.CancelBroadphaseAdd(world, 102);
        store.QueueBroadphaseRemove(world, 102);
        store.CancelBroadphaseRemove(world, 102);
        const std::vector<uint32_t> expected{101, 100};
        if (store.ConsumePendingBroadphaseAdds(world) != expected ||
            store.ConsumePendingBroadphaseRemoves(world) != expected)
            throw std::runtime_error("canceled requests resurfaced or final request order changed");
        if (store.HasPendingBroadphaseAdds(world) || store.HasPendingBroadphaseRemoves(world) ||
            !store.ConsumePendingBroadphaseAdds(world).empty() || !store.ConsumePendingBroadphaseRemoves(world).empty())
            throw std::runtime_error("consumed queue retained a request");
    }
    store.ClearWorldPendingQueues(world);
    if (store.ConsumePendingBroadphaseAdds(other) != std::vector<uint32_t>{200} ||
        store.ConsumePendingBroadphaseRemoves(other) != std::vector<uint32_t>{201})
        throw std::runtime_error("consuming one World changed another World's requests");
    std::cout << "BROADPHASE_QUEUE cycles=128 repeats=32 final-order=passed\n";
}
