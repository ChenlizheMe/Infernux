#include "PhysicsContactPublication.h"
#include "PhysicsECSStore.h"
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
constexpr uint8_t CollisionInterest = CollisionEnterInterest | CollisionStayInterest | CollisionExitInterest;
constexpr uint8_t TriggerInterest = TriggerEnterInterest | TriggerStayInterest | TriggerExitInterest;

uint8_t Mask(const Component &c)
{
    return uint8_t((c.WantsCollisionEnterCallbacks() ? CollisionEnterInterest : 0) |
                   (c.WantsCollisionStayCallbacks() ? CollisionStayInterest : 0) |
                   (c.WantsCollisionExitCallbacks() ? CollisionExitInterest : 0) |
                   (c.WantsTriggerEnterCallbacks() ? TriggerEnterInterest : 0) |
                   (c.WantsTriggerStayCallbacks() ? TriggerStayInterest : 0) |
                   (c.WantsTriggerExitCallbacks() ? TriggerExitInterest : 0));
}

State::PairKey CanonicalKey(const State::Manifold &manifold)
{
    const auto a = manifold.a.Key(), b = manifold.b.Key();
    return {std::min(a, b), std::max(a, b), manifold.trigger};
}

template <class T> void Identity(T &e, const State::Pair &pair, ContactStreamIdentity identity, uint32_t substep)
{
    e.worldId = identity.worldId;
    e.simulationEpoch = identity.simulationEpoch;
    e.fixedStepIndex = identity.fixedStepIndex;
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
        const auto &other = sideB ? pair.a : pair.b;
        const auto found = policy.receivers.find(endpoint.SurfaceKey());
        if (found == policy.receivers.end())
            continue;
        for (const auto &receiver : found->second)
            if (receiver.mask & ContactEventInterestBit(type))
                batch.deliveries.push_back({receiver, other, pair.position, (sideB ? 1.0f : -1.0f) * pair.normal,
                                            (sideB ? -1.0f : 1.0f) * pair.relativeVelocity, type});
    }
}
ContactEventType Type(bool trigger, uint32_t phase)
{
    return static_cast<ContactEventType>((trigger ? 3 : 0) + phase);
}
} // namespace
void State::Policy::CaptureActor(const RuntimeWorld &world, const PhysicsActorData &actor)
{
    if (actor.runtimeWorld != &world || !actor.owner || actor.sceneSuspended || !actor.owner->IsActiveInHierarchy())
        return;
    for (auto *collider : actor.colliders) {
        if (!collider || !collider->GetGameObject() || !collider->GetGameObject()->IsActiveInHierarchy())
            continue;
        std::vector<Receiver> interested;
        std::set<Surface> seen;
        for (auto *object : {actor.owner, collider->GetGameObject()})
            for (const auto &c : object->GetAllComponents()) {
                if (!c || c->IsDestroyed() || c->IsBeingDestroyed())
                    continue;
                const auto mask = Mask(*c);
                const auto identity = std::make_pair(c->GetComponentID(), c->GetLifetimeGeneration());
                if (mask && seen.insert(identity).second)
                    interested.push_back({PhysicsTargetReference(c.get()), mask});
            }
        if (!interested.empty())
            receivers[{collider->GetComponentID(), collider->GetLifetimeGeneration()}] = std::move(interested);
    }
}
uint8_t State::Policy::Interest(Surface surface) const
{
    uint8_t mask = 0;
    const auto found = receivers.find(surface);
    if (found != receivers.end())
        for (const auto &receiver : found->second)
            mask |= receiver.mask;
    return mask;
}
bool State::Policy::WantsTriggers() const
{
    return (events && triggers) || std::any_of(receivers.begin(), receivers.end(), [](const auto &entry) {
               return std::any_of(entry.second.begin(), entry.second.end(),
                                  [](const Receiver &receiver) { return (receiver.mask & TriggerInterest) != 0; });
           });
}
bool State::Policy::Observes(Surface a, Surface b, bool trigger) const
{
    if (trigger ? events && triggers : events || impulses)
        return true;
    const auto mask = trigger ? TriggerInterest : CollisionInterest;
    return ((Interest(a) | Interest(b)) & mask) != 0;
}
void State::PublishEmpty(ContactStreamIdentity identity)
{
    auto empty = std::make_shared<Batch>();
    empty->identity = identity;
    pairs.clear();
    batch = std::move(empty);
    receiverCursor = 0;
}
void State::Publish(ContactStreamIdentity identity, const Policy &policy, uint32_t substeps, const ReadFrame &readFrame)
{
    const bool sameEpoch = batch && batch->identity.worldId == identity.worldId &&
                           batch->identity.simulationEpoch == identity.simulationEpoch;
    if (sameEpoch && batch->identity.fixedStepIndex >= identity.fixedStepIndex)
        throw std::logic_error("Repeated/reordered contact publication");
    auto previous = sameEpoch ? pairs : std::map<PairKey, Pair>{};
    auto output = std::make_shared<Batch>();
    output->identity = identity;
    const auto observed = [&](const Pair &pair) {
        return policy.Observes(pair.a.SurfaceKey(), pair.b.SurfaceKey(), pair.trigger);
    };
    for (auto it = previous.begin(); it != previous.end();) {
        if (!observed(it->second))
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
        Identity(e, pair, identity, substep);
        e.type = Type(pair.trigger, phase);
        e.contactPoint = pair.position;
        e.contactNormal = pair.normal;
        e.relativeVelocity = pair.relativeVelocity;
        e.feature = pair.feature;
        if (phase != 2)
            e.contactSeparation = pair.separation;
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
    std::set<PairKey> rawStayed;
    for (uint32_t substep = 0; substep < substeps; ++substep) {
        auto frame = readFrame(substep);
        // Read impulse samples in the same canonical pair order as stream events.
        std::stable_sort(frame.manifolds.begin(), frame.manifolds.end(),
                         [](const Manifold &a, const Manifold &b) { return CanonicalKey(a) < CanonicalKey(b); });
        std::map<PairKey, Pair> current;
        uint32_t nextPointIndex = 0;
        for (const auto &manifold : frame.manifolds) {
            const auto offset = manifold.pointOffset;
            if (manifold.a.body == manifold.b.body || !manifold.pointCount || offset > frame.points.size() ||
                manifold.pointCount > frame.points.size() - offset)
                throw std::runtime_error("Invalid contact pair");
            const auto key = CanonicalKey(manifold);
            if (!policy.Observes(manifold.a.SurfaceKey(), manifold.b.SurfaceKey(), manifold.trigger))
                throw std::runtime_error("Unrequested contact pair");
            // One substep can revisit a pair. Its lifecycle uses the first manifold;
            // each later solved point keeps its own direction, impulse and order.
            const bool firstManifold = current.empty() || current.rbegin()->first != key;
            if (firstManifold)
                nextPointIndex = 0;

            const bool reverse = manifold.a.Key() != std::get<0>(key);
            const auto orient = [reverse](const glm::vec3 &value) { return reverse ? -value : value; };
            const auto &first = frame.points[offset];
            const Pair pair{reverse ? manifold.b : manifold.a,
                            reverse ? manifold.a : manifold.b,
                            manifold.trigger,
                            first.position,
                            orient(first.normal),
                            orient(first.relativeVelocity),
                            first.feature,
                            first.separation};
            if (policy.impulses && !manifold.trigger)
                for (uint32_t i = 0; i < manifold.pointCount; ++i) {
                    const auto &point = frame.points[offset + i];
                    if (!point.hasSolvedImpulse)
                        continue;
                    ContactImpulse impulse;
                    Identity(impulse, pair, identity, substep);
                    impulse.feature = point.feature;
                    impulse.pointIndex = nextPointIndex + i;
                    impulse.contactPoint = point.position;
                    impulse.contactNormal = orient(point.normal);
                    impulse.impulse = orient(point.impulse);
                    output->impulses.push_back(impulse);
                }
            nextPointIndex += manifold.pointCount;
            if (firstManifold)
                current.emplace_hint(current.end(), key, pair);
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
    const auto epoch = retained->identity.simulationEpoch;
    const auto admittedState = SceneManager::Instance().GetSimulationState();
    if (world.id != retained->identity.worldId || world.simulation.simulationEpoch != epoch) {
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
        CollisionInfo info;
        info.target = d.other.target;
        info.bodyTarget = d.other.bodyTarget;
        info.collider = d.other.target.GetCollider();
        info.gameObject = d.other.target.GetGameObject();
        info.contactPoint = d.position;
        info.contactNormal = d.normal;
        info.relativeVelocity = d.relativeVelocity;
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
    batch.reset();
    pairs.clear();
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
