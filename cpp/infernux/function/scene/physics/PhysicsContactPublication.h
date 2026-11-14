#pragma once
#include "JoltGpuContactReports.h"
#include "PhysicsContactTypes.h"
#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <utility>

namespace infernux
{
class RuntimeWorld;
/// CPU notification state only. Geometry, overlap and impulses always originate
/// in the GPU report; this class neither tests overlap nor solves contacts.
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
        [[nodiscard]] uint32_t Flags() const
        {
            return (events ? 1u : 0u) | (events && triggers ? 2u : 0u) | (impulses ? 4u : 0u);
        }
        [[nodiscard]] std::vector<uint32_t> LeafInterest(const JoltGpuScene::Publication &) const;
    };
    struct Pending
    {
        Policy policy;
        std::shared_ptr<JoltGpuContactReportResult> report;
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
    struct Pair
    {
        Endpoint a, b;
        bool trigger = false;
        JPH::GPU::ContactReportPatch record{};
        PairKey Key() const
        {
            return std::make_tuple(a.Key(), b.Key(), trigger);
        }
        CallbackPairKey CallbackKey() const
        {
            return std::make_tuple(std::min(a.SurfaceKey(), b.SurfaceKey()), std::max(a.SurfaceKey(), b.SurfaceKey()),
                                   trigger);
        }
    };
    struct Delivery
    {
        Receiver receiver;
        Pair pair;
        ContactEventType type;
        bool sideB = false;
    };
    struct Batch
    {
        std::shared_ptr<const JoltGpuScene::Publication> publication;
        std::vector<ContactEvent> events;
        std::vector<ContactImpulse> impulses;
        std::vector<Delivery> deliveries;
    };
    std::shared_ptr<Pending> pending;
    std::shared_ptr<const Batch> batch;
    bool dispatching = false;
    [[nodiscard]] static Policy Capture(RuntimeWorld &, const JoltGpuScene &, bool events, bool triggers,
                                        bool impulses);
    void Publish(const std::shared_ptr<const JoltGpuScene::Publication> &);
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
