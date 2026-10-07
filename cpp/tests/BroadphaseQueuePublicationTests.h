#pragma once

#include <function/scene/physics/PhysicsECSStore.h>
#include <iostream>
#include <stdexcept>
#include <vector>

inline void TestBroadphaseQueuePublication()
{
    auto &store = infernux::PhysicsECSStore::Instance();
    if (!store.ConsumePendingBroadphaseAdds().empty() || !store.ConsumePendingBroadphaseRemoves().empty())
        throw std::runtime_error("unexpected pending bodies before queue test");
    for (int cycle = 0; cycle < 128; ++cycle) {
        for (int repeat = 0; repeat < 32; ++repeat) {
            store.QueueBroadphaseAdd(100);
            store.QueueBroadphaseAdd(101);
            if (!store.CancelBroadphaseAdd(100))
                throw std::runtime_error("add request was lost");
            store.QueueBroadphaseAdd(100);
            store.QueueBroadphaseRemove(100);
            store.QueueBroadphaseRemove(101);
            if (!store.CancelBroadphaseRemove(100))
                throw std::runtime_error("remove request was lost");
            store.QueueBroadphaseRemove(100);
        }
        store.QueueBroadphaseAdd(102);
        store.CancelBroadphaseAdd(102);
        store.QueueBroadphaseRemove(102);
        store.CancelBroadphaseRemove(102);
        const std::vector<uint32_t> expected{101, 100};
        if (store.ConsumePendingBroadphaseAdds() != expected ||
            store.ConsumePendingBroadphaseRemoves() != expected)
            throw std::runtime_error("canceled requests resurfaced or final request order changed");
        if (store.HasPendingBroadphaseAdds() || store.HasPendingBroadphaseRemoves() ||
            !store.ConsumePendingBroadphaseAdds().empty() || !store.ConsumePendingBroadphaseRemoves().empty())
            throw std::runtime_error("consumed queue retained a request");
    }
    std::cout << "BROADPHASE_QUEUE cycles=128 repeats=32 final-order=passed\n";
}
