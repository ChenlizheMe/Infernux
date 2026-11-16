#pragma once
#include "PhysicsContactTypes.h"
#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <utility>

namespace infernux
{
struct PhysicsActorData;
class RuntimeWorld;
/// Publishes events, impulses and callbacks from a completed Fixed step.
class PhysicsContactPublication final
{
  public:
    using Surface = std::pair<uint64_t, uint64_t>; // Collider ID + lifetime generation.
    using EndpointKey = std::tuple<uint32_t, uint64_t, uint64_t, uint32_t>;
    using PairKey = std::tuple<EndpointKey, EndpointKey, bool>;
    using CallbackPairKey = std::tuple<Surface, Surface, bool>;
    struct Receiver
    {
        PhysicsTargetReference target;
        uint8_t mask = 0;
    };
    struct Policy
    {
        bool events = false, triggers = true, impulses = false;
        std::map<Surface, std::vector<Receiver>> receivers;
        [[nodiscard]] bool Requested() const
        {
            return events || impulses || !receivers.empty();
        }
        void CaptureActor(const RuntimeWorld &, const PhysicsActorData &);
        [[nodiscard]] uint8_t Interest(Surface) const;
        [[nodiscard]] bool Observes(Surface a, Surface b, bool trigger) const;
    };
    struct Endpoint
    {
        uint32_t body = UINT32_MAX, subshape = 0;
        uint64_t collider = 0, generation = 0;
        PhysicsTargetReference target, bodyTarget;
        EndpointKey Key() const
        {
            return std::make_tuple(body, collider, generation, subshape);
        }
        Surface SurfaceKey() const
        {
            return {collider, generation};
        }
    };
    struct Point
    {
        // Normal points from A to B, relative velocity is B minus A, and
        // impulse is the one applied to B.
        glm::vec3 position{0.0f}, normal{0.0f}, relativeVelocity{0.0f}, impulse{0.0f};
        uint32_t feature = 0;
        bool hasSolvedImpulse = true;
    };
    struct Manifold
    {
        Endpoint a, b;
        bool trigger = false;
        size_t pointOffset = 0;
        uint32_t pointCount = 0;
    };
    struct Frame
    {
        std::vector<Manifold> manifolds;
        // Each manifold's range stores its points in solver order.
        std::vector<Point> points;
    };
    using ReadFrame = std::function<Frame(uint32_t substep)>;
    struct Pair
    {
        Endpoint a, b;
        bool trigger = false;
        // The first solver point supplies stream events and callbacks.
        glm::vec3 position, normal, relativeVelocity;
        uint32_t feature;
        CallbackPairKey CallbackKey() const
        {
            return std::make_tuple(std::min(a.SurfaceKey(), b.SurfaceKey()), std::max(a.SurfaceKey(), b.SurfaceKey()),
                                   trigger);
        }
    };
    struct Delivery
    {
        Receiver receiver;
        Endpoint other;
        glm::vec3 position, normal, relativeVelocity;
        ContactEventType type;
    };
    struct Batch
    {
        ContactStreamIdentity identity;
        std::vector<ContactEvent> events;
        std::vector<ContactImpulse> impulses;
        std::vector<Delivery> deliveries;
    };
    std::shared_ptr<const Batch> batch;
    bool dispatching = false;
    /// Reads each completed substep synchronously in order. The result is
    /// visible only after every frame is accepted.
    void Publish(ContactStreamIdentity, const Policy &, uint32_t substeps, const ReadFrame &);
    void PublishEmpty(ContactStreamIdentity);
    [[nodiscard]] size_t Dispatch(RuntimeWorld &);
    void Invalidate(uint32_t body)
    {
        invalidated.insert(body);
    }
    void Close() noexcept;
    [[nodiscard]] const std::vector<ContactEvent> &Events() const;
    [[nodiscard]] const std::vector<ContactImpulse> &Impulses() const;

  private:
    std::map<PairKey, Pair> pairs;
    std::set<uint32_t> invalidated;
    size_t receiverCursor = 0;
};
} // namespace infernux
