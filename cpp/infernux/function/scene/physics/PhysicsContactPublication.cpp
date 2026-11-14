#include "PhysicsContactPublication.h"
#include <algorithm>
#include <cmath>
#include <function/scene/Collider.h>
#include <function/scene/Component.h>
#include <function/scene/GameObject.h>
#include <function/scene/RuntimeWorld.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <stdexcept>

namespace infernux
{
namespace
{
using State = PhysicsContactPublication;
using namespace JPH::GPU;
glm::vec3 V(Float4 v)
{
    return {v.x, v.y, v.z};
}
bool Finite(Float4 v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && std::isfinite(v.w);
}
uint8_t Mask(const Component &c)
{
    return uint8_t((c.WantsCollisionEnterCallbacks() ? 1 : 0) | (c.WantsCollisionStayCallbacks() ? 2 : 0) |
                   (c.WantsCollisionExitCallbacks() ? 4 : 0) | (c.WantsTriggerEnterCallbacks() ? 8 : 0) |
                   (c.WantsTriggerStayCallbacks() ? 16 : 0) | (c.WantsTriggerExitCallbacks() ? 32 : 0));
}
uint32_t Category(const State::Policy &p, State::Surface surface)
{
    uint8_t mask = 0;
    const auto i = p.receivers.find(surface);
    if (i != p.receivers.end())
        for (const auto &r : i->second)
            mask |= r.mask;
    return ((mask & 7u) ? 1u : 0u) | ((mask & 56u) ? 2u : 0u);
}
bool Observed(const State::Policy &p, const State::Pair &pair)
{
    const auto interest = Category(p, pair.a.SurfaceKey()) | Category(p, pair.b.SurfaceKey());
    return pair.trigger ? (p.events && p.triggers) || (interest & 2u) : p.events || p.impulses || (interest & 1u);
}
State::Endpoint Decode(const JoltGpuScene::Publication &p, uint32_t slot, uint32_t leaf)
{
    if (slot < p.bodyBase || slot - p.bodyBase >= p.bodyCount || leaf < p.leafBase ||
        leaf - p.leafBase >= p.leafCount || !p.directory || slot >= p.directory->actors.size() ||
        leaf >= p.directory->leaves.size())
        throw std::runtime_error("Contact endpoint outside retained publication range");
    const auto &actor = p.directory->actors[slot];
    const auto address = p.directory->leaves[leaf];
    if (!actor || address.bodySlot != slot || address.actorLeaf >= actor->leaves.size() ||
        slot >= p.directory->residentBodies.size() || !p.directory->residentBodies[slot])
        throw std::runtime_error("Contact endpoint differs from retained actor/leaf directory");
    const auto &surface = actor->leaves[address.actorLeaf];
    State::Endpoint e;
    e.body = actor->bodyId;
    e.subshape = surface.subShapeId;
    e.collider = surface.colliderId;
    e.generation = surface.colliderGeneration;
    e.target = surface.surfaceTarget;
    e.bodyTarget = actor->bodyTarget;
    return e;
}
State::Pair Decode(const JoltGpuScene::Publication &p, const ContactReportPatch &r, uint32_t substep)
{
    if (r.meta.x != substep || !r.meta.z || r.meta.z > 4 ||
        (r.meta.y & ~(PatchActive | PatchSensor | PatchOverlapping)))
        throw std::runtime_error("Invalid GPU contact report metadata");
    State::Pair pair;
    pair.a = Decode(p, r.source.z, r.source.x);
    pair.b = Decode(p, r.source.w, r.source.y);
    if (pair.a.body == pair.b.body)
        throw std::runtime_error("GPU contact report contains a self pair");
    pair.trigger = (r.meta.y & PatchSensor) != 0;
    pair.record = r;
    if (pair.trigger ? !(r.meta.y & PatchOverlapping) : !(r.meta.y & PatchActive))
        throw std::runtime_error("GPU contact kind/active mismatch");
    for (uint32_t i = 0; i < r.meta.z; ++i) {
        const auto &v = r.points[i];
        const auto n = V(v.normal);
        if (v.feature.y != 1 || v.feature.z > 1 || v.feature.w || !Finite(v.pointDepth) || !Finite(v.normal) ||
            !Finite(v.relativeVelocity) || !Finite(v.impulse) || std::abs(glm::dot(n, n) - 1.0f) > 1.0e-3f ||
            (pair.trigger && (v.impulse.x || v.impulse.y || v.impulse.z)))
            throw std::runtime_error("GPU contact report has invalid numerical point");
    }
    if (pair.b.Key() < pair.a.Key()) {
        std::swap(pair.a, pair.b);
        std::swap(pair.record.source.x, pair.record.source.y);
        std::swap(pair.record.source.z, pair.record.source.w);
        for (auto &v : pair.record.points) {
            v.normal.x = -v.normal.x;
            v.normal.y = -v.normal.y;
            v.normal.z = -v.normal.z;
            v.relativeVelocity.x = -v.relativeVelocity.x;
            v.relativeVelocity.y = -v.relativeVelocity.y;
            v.relativeVelocity.z = -v.relativeVelocity.z;
            v.impulse.x = -v.impulse.x;
            v.impulse.y = -v.impulse.y;
            v.impulse.z = -v.impulse.z;
        }
    }
    return pair;
}
template <class T> void Identity(T &e, const State::Pair &pair, const JoltGpuScene::Publication &p, uint32_t substep)
{
    e.worldId = p.worldId;
    e.simulationEpoch = p.simulationEpoch;
    e.fixedStepIndex = p.fixedStepIndex;
    e.substep = substep;
    e.bodyIdA = pair.a.body;
    e.bodyIdB = pair.b.body;
    e.colliderIdA = pair.a.collider;
    e.colliderIdB = pair.b.collider;
    e.colliderGenerationA = pair.a.generation;
    e.colliderGenerationB = pair.b.generation;
    e.subShapeIdA = pair.a.subshape;
    e.subShapeIdB = pair.b.subshape;
}
void Receivers(State::Batch &batch, const State::Policy &policy, const State::Pair &pair, ContactEventType type)
{
    for (bool sideB : {false, true}) {
        const auto &endpoint = sideB ? pair.b : pair.a;
        const auto found = policy.receivers.find(endpoint.SurfaceKey());
        if (found == policy.receivers.end())
            continue;
        for (const auto &receiver : found->second)
            if (receiver.mask & ContactEventInterestBit(type))
                batch.deliveries.push_back({receiver, pair, type, sideB});
    }
}
ContactEventType Type(bool trigger, uint32_t phase)
{
    return static_cast<ContactEventType>((trigger ? 3 : 0) + phase);
}
} // namespace
State::Policy State::Capture(RuntimeWorld &world, const JoltGpuScene &scene, bool events, bool triggers, bool impulses)
{
    Policy policy;
    policy.events = events;
    policy.triggers = triggers;
    policy.impulses = impulses;
    auto &store = PhysicsECSStore::Instance();
    scene.VisitActorHandles(world, [&](PhysicsActorHandle handle) {
        if (!store.IsValid(handle))
            return;
        const auto &actor = store.GetActor(handle);
        if (actor.runtimeWorld != &world || !actor.owner || actor.sceneSuspended || !actor.owner->IsActiveInHierarchy())
            return;
        for (auto *collider : actor.colliders) {
            if (!collider || !collider->GetGameObject() || !collider->GetGameObject()->IsActiveInHierarchy())
                continue;
            std::vector<Receiver> receivers;
            std::set<std::pair<uint64_t, uint64_t>> seen;
            for (auto *object : {actor.owner, collider->GetGameObject()})
                for (const auto &c : object->GetAllComponents()) {
                    if (!c || c->IsDestroyed() || c->IsBeingDestroyed())
                        continue;
                    const auto mask = Mask(*c);
                    const auto identity = std::make_pair(c->GetComponentID(), c->GetLifetimeGeneration());
                    if (mask && seen.insert(identity).second)
                        receivers.push_back({PhysicsTargetReference(c.get()), mask});
                }
            if (!receivers.empty())
                policy.receivers[{collider->GetComponentID(), collider->GetLifetimeGeneration()}] =
                    std::move(receivers);
        }
    });
    return policy;
}
std::vector<uint32_t> State::Policy::LeafInterest(const JoltGpuScene::Publication &p) const
{
    if (!p.directory)
        throw std::logic_error("Contact interest needs accepted directory");
    std::vector<uint32_t> result(p.leafCount);
    for (uint32_t i = 0; i < p.leafCount; ++i) {
        const auto &address = p.directory->leaves.at(p.leafBase + i);
        const auto &actor = p.directory->actors.at(address.bodySlot);
        if (!actor || address.actorLeaf >= actor->leaves.size())
            throw std::logic_error("Invalid accepted interest directory");
        const auto &leaf = actor->leaves[address.actorLeaf];
        result[i] = Category(*this, {leaf.colliderId, leaf.colliderGeneration});
    }
    return result;
}
void State::Publish(const std::shared_ptr<const JoltGpuScene::Publication> &p)
{
    if (!pending) {
        auto empty = std::make_shared<Batch>();
        empty->publication = p;
        pairs.clear();
        invalidated.clear();
        batch = std::move(empty);
        receiverCursor = 0;
        return;
    }
    if (pending->report->publication.get() != p.get())
        throw std::logic_error("Contact capture publication mismatch");
    auto captured = std::move(pending);
    const auto &report = *captured->report;
    const auto &policy = captured->policy;
    if (!report.complete || report.frames.size() != report.expectedSubsteps)
        throw std::logic_error("Cannot publish partial contact Fixed");
    if (batch && batch->publication->worldId == p->worldId &&
        batch->publication->simulationEpoch == p->simulationEpoch &&
        batch->publication->fixedStepIndex >= p->fixedStepIndex)
        throw std::logic_error("Repeated/reordered contact publication");
    auto previous = (batch && batch->publication->worldId == p->worldId &&
                     batch->publication->simulationEpoch == p->simulationEpoch)
                        ? pairs
                        : std::map<PairKey, Pair>{};
    auto output = std::make_shared<Batch>();
    output->publication = p;
    for (auto it = previous.begin(); it != previous.end();) {
        if (!Observed(policy, it->second))
            it = previous.erase(it);
        else
            ++it;
    }
    using CallbackKey = CallbackPairKey;
    std::set<CallbackKey> stayed;
    auto stream = [&](const Pair &pair, uint32_t phase, uint32_t substep) {
        if (!policy.events || (pair.trigger && !policy.triggers))
            return;
        ContactEvent e{};
        Identity(e, pair, *p, substep);
        e.type = Type(pair.trigger, phase);
        e.contactPoint = V(pair.record.points[0].pointDepth);
        e.contactNormal = V(pair.record.points[0].normal);
        e.relativeVelocity = V(pair.record.points[0].relativeVelocity);
        e.feature = pair.record.points[0].feature.x;
        output->events.push_back(e);
    };
    auto callbacks = [&](const std::map<PairKey, Pair> &before, const std::map<PairKey, Pair> &after,
                         bool actualFrame) {
        std::map<CallbackKey, const Pair *> a, b;
        for (const auto &[key, pair] : before)
            a.emplace(pair.CallbackKey(), &pair);
        for (const auto &[key, pair] : after)
            b.emplace(pair.CallbackKey(), &pair);
        for (const auto &[key, pair] : a)
            if (!b.count(key))
                Receivers(*output, policy, *pair, Type(pair->trigger, 2));
        for (const auto &[key, pair] : b) {
            if (!a.count(key)) {
                Receivers(*output, policy, *pair, Type(pair->trigger, 0));
                stayed.insert(key);
            } else if (actualFrame && stayed.insert(key).second)
                Receivers(*output, policy, *pair, Type(pair->trigger, 1));
        }
    };
    std::map<PairKey, Pair> retained = previous;
    for (auto it = retained.begin(); it != retained.end();) {
        if (invalidated.count(it->second.a.body) || invalidated.count(it->second.b.body)) {
            stream(it->second, 2, 0);
            it = retained.erase(it);
        } else
            ++it;
    }
    callbacks(previous, retained, false);
    previous = std::move(retained);
    std::set<PairKey> rawStayed;
    for (uint32_t substep = 0; substep < report.expectedSubsteps; ++substep) {
        const auto &frame = report.frames[substep];
        if (frame.substep != substep || !frame.finished || !frame.seeded)
            throw std::logic_error("Contact frame sequence mismatch");
        std::map<PairKey, Pair> current;
        for (const auto &record : frame.records) {
            if (record.meta.w >= frame.capacity)
                throw std::runtime_error("Contact source ordinal exceeds retained workset");
            auto pair = Decode(*p, record, substep);
            if (!Observed(policy, pair) || !current.emplace(pair.Key(), pair).second)
                throw std::runtime_error("Duplicate/unrequested GPU contact pair");
        }
        for (const auto &[key, pair] : current) {
            if (policy.impulses && !pair.trigger)
                for (uint32_t i = 0; i < pair.record.meta.z; ++i) {
                    ContactImpulse impulse;
                    Identity(impulse, pair, *p, substep);
                    const auto &point = pair.record.points[i];
                    impulse.feature = point.feature.x;
                    impulse.pointIndex = i;
                    impulse.contactPoint = V(point.pointDepth);
                    impulse.contactNormal = V(point.normal);
                    impulse.impulse = V(point.impulse);
                    output->impulses.push_back(impulse);
                }
        }
        for (const auto &[key, pair] : previous)
            if (!current.count(key))
                stream(pair, 2, substep);
        for (const auto &[key, pair] : current) {
            if (!previous.count(key)) {
                stream(pair, 0, substep);
                rawStayed.insert(key);
            } else if (rawStayed.insert(key).second)
                stream(pair, 1, substep);
        }
        callbacks(previous, current, true);
        previous = std::move(current);
    }
    pairs = std::move(previous);
    invalidated.clear();
    batch = std::move(output);
    receiverCursor = 0;
}
size_t State::Dispatch(RuntimeWorld &world)
{
    if (dispatching)
        throw std::logic_error("Contact callbacks cannot reenter delivery");
    const auto retained = batch;
    if (!retained)
        return 0;
    const WorldExecutionScope ownerScope(&world);
    const auto epoch = retained->publication->simulationEpoch;
    const auto admittedState = SceneManager::Instance().GetSimulationState();
    if (world.id != retained->publication->worldId || world.simulation.simulationEpoch != epoch) {
        Close();
        return 0;
    }
    struct Guard
    {
        bool &v;
        explicit Guard(bool &x) : v(x)
        {
            v = true;
        }
        ~Guard()
        {
            v = false;
        }
    } guard(dispatching);
    size_t delivered = 0;
    while (batch == retained && receiverCursor < retained->deliveries.size() &&
           world.simulation.simulationEpoch == epoch && world.simulation.isPlaying &&
           SceneManager::Instance().GetSimulationState() == admittedState) {
        const auto &d =
            retained->deliveries[receiverCursor++]; // Consume before user code, including throwing callbacks.
        auto *component = d.receiver.target.GetComponent();
        auto *object = component ? component->GetGameObject() : nullptr;
        if (!object || component->IsDestroyed() || component->IsBeingDestroyed() || !object->IsActiveInHierarchy() ||
            !object->GetScene() || object->GetScene()->GetRuntimeWorld() != &world ||
            object->GetScene()->IsPendingDestroy(object))
            continue;
        const auto &other = d.sideB ? d.pair.a : d.pair.b;
        CollisionInfo info;
        info.target = other.target;
        info.bodyTarget = other.bodyTarget;
        info.collider = other.target.GetCollider();
        info.gameObject = other.target.GetGameObject();
        info.contactPoint = V(d.pair.record.points[0].pointDepth);
        info.contactNormal = (d.sideB ? 1.0f : -1.0f) * V(d.pair.record.points[0].normal);
        info.relativeVelocity = (d.sideB ? -1.0f : 1.0f) * V(d.pair.record.points[0].relativeVelocity);
        ++delivered;
        switch (d.type) {
        case ContactEventType::CollisionEnter:
            component->OnCollisionEnter(info);
            break;
        case ContactEventType::CollisionStay:
            component->OnCollisionStay(info);
            break;
        case ContactEventType::CollisionExit:
            component->OnCollisionExit(info);
            break;
        case ContactEventType::TriggerEnter:
            component->OnTriggerEnter(info.collider);
            break;
        case ContactEventType::TriggerStay:
            component->OnTriggerStay(info.collider);
            break;
        case ContactEventType::TriggerExit:
            component->OnTriggerExit(info.collider);
            break;
        }
    }
    return delivered;
}
void State::Close() noexcept
{
    pending.reset();
    batch.reset();
    pairs.clear();
    invalidated.clear();
    receiverCursor = 0;
}
const std::vector<ContactEvent> &State::Events() const
{
    static const std::vector<ContactEvent> empty;
    return batch ? batch->events : empty;
}
const std::vector<ContactImpulse> &State::Impulses() const
{
    static const std::vector<ContactImpulse> empty;
    return batch ? batch->impulses : empty;
}
} // namespace infernux
