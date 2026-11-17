#include "Bvh.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace infernux::spatial
{

void Bounds::Include(const Bounds &other) noexcept
{
    minimum = glm::min(minimum, other.minimum);
    maximum = glm::max(maximum, other.maximum);
}

glm::vec3 Bounds::Center() const noexcept
{
    return minimum * 0.5f + maximum * 0.5f;
}

double Bounds::Area() const noexcept
{
    const glm::dvec3 extent = glm::dvec3(maximum) - glm::dvec3(minimum);
    return 2.0 * (extent.x * extent.y + extent.x * extent.z + extent.y * extent.z);
}

namespace
{
constexpr uint32_t BucketCount = 12;

struct Bucket
{
    Bounds bounds;
    uint32_t count = 0;
};

struct Split
{
    double cost = std::numeric_limits<double>::infinity();
    double minimum = 0.0;
    double bucketsPerUnit = 0.0;
    uint32_t leftCount = 0;
    uint32_t lastLeftBucket = 0;
    unsigned axis = 0;

    [[nodiscard]] uint32_t BucketOf(const glm::vec3 &point) const noexcept
    {
        const auto bucket = static_cast<uint32_t>((point[axis] - minimum) * bucketsPerUnit);
        return std::min(bucket, BucketCount - 1);
    }
};

enum class Phase : uint8_t
{
    Initialize,
    ScanBounds,
    BucketAxis,
    Partition,
    Commit,
    Complete,
};

struct NodeRange
{
    uint32_t first;
    uint32_t end;
    uint32_t depth;
    uint32_t parent;
    bool right;
};
} // namespace

struct BvhBuild::State
{
    explicit State(const std::vector<Bounds> &source) : primitiveBounds(source)
    {
        if (source.size() > kBvhLeafBit / 2u)
            throw std::length_error("BVH exceeds its node index range");
        if (source.empty()) {
            phase = Phase::Complete;
            return;
        }

        // Reserve raw storage once. Growing any of these vectors during an
        // advance could relocate an entire previously processed range.
        centers.reserve(source.size());
        scratch.reserve(source.size());
        result.primitiveOrder.reserve(source.size());
        result.nodes.reserve(source.size() * 2 - 1);
        range.end = static_cast<uint32_t>(source.size());
    }

    [[nodiscard]] bool Advance(uint32_t primitiveBudget)
    {
        while (primitiveBudget && phase != Phase::Complete) {
            const auto count = std::min(primitiveBudget, range.end - cursor);
            const uint32_t stop = cursor + count;
            switch (phase) {
            case Phase::Initialize:
                Initialize(stop);
                break;
            case Phase::ScanBounds:
                ScanBounds(stop);
                break;
            case Phase::BucketAxis:
                BucketAxis(stop);
                break;
            case Phase::Partition:
                Partition(stop);
                break;
            case Phase::Commit:
                Commit(stop);
                break;
            case Phase::Complete:
                break;
            }
            primitiveBudget -= count;
        }
        return phase == Phase::Complete;
    }

    void Initialize(uint32_t stop)
    {
        for (; cursor < stop; ++cursor) {
            const auto &bound = primitiveBounds[cursor];
            for (unsigned component = 0; component < 3; ++component)
                if (!std::isfinite(bound.minimum[component]) || !std::isfinite(bound.maximum[component]) ||
                    bound.minimum[component] > bound.maximum[component])
                    throw std::invalid_argument("BVH requires finite, ordered primitive bounds");
            centers.push_back(bound.Center());
            scratch.push_back(0);
            result.primitiveOrder.push_back(cursor);
        }
        if (cursor == range.end) {
            cursor = 0;
            phase = Phase::ScanBounds;
        }
    }

    void ScanBounds(uint32_t stop)
    {
        for (; cursor < stop; ++cursor) {
            const auto primitive = result.primitiveOrder[cursor];
            bounds.Include(primitiveBounds[primitive]);
            const auto center = centers[primitive];
            centerBounds.Include({center, center});
        }
        if (cursor == range.end)
            EmitNode();
    }

    void EmitNode()
    {
        node = static_cast<uint32_t>(result.nodes.size());
        const uint32_t count = range.end - range.first;
        result.nodes.push_back({{bounds.minimum.x, bounds.minimum.y, bounds.minimum.z},
                                range.first,
                                {bounds.maximum.x, bounds.maximum.y, bounds.maximum.z},
                                kBvhLeafBit | count});
        if (range.depth != 0) {
            auto &parent = result.nodes[range.parent];
            (range.right ? parent.second : parent.first) = node;
        }
        if (count <= kBvhLeafCapacity) {
            NextNode();
            return;
        }
        best = {};
        axis = 0;
        NextAxis();
    }

    void NextAxis()
    {
        for (; axis < 3; ++axis) {
            const double extent = static_cast<double>(centerBounds.maximum[axis]) - centerBounds.minimum[axis];
            if (extent == 0.0)
                continue;
            axisSplit.minimum = centerBounds.minimum[axis];
            axisSplit.bucketsPerUnit = static_cast<double>(BucketCount) / extent;
            axisSplit.axis = axis;
            buckets = {};
            cursor = range.first;
            phase = Phase::BucketAxis;
            return;
        }

        if (best.leftCount == 0) {
            ScheduleChildren(range.first + (range.end - range.first) / 2);
            return;
        }
        cursor = left = range.first;
        right = range.first + best.leftCount;
        phase = Phase::Partition;
    }

    void BucketAxis(uint32_t stop)
    {
        for (; cursor < stop; ++cursor) {
            const auto primitive = result.primitiveOrder[cursor];
            auto &bucket = buckets[axisSplit.BucketOf(centers[primitive])];
            bucket.bounds.Include(primitiveBounds[primitive]);
            ++bucket.count;
        }
        if (cursor == range.end) {
            ScoreAxis();
            ++axis;
            NextAxis();
        }
    }

    void ScoreAxis()
    {
        // Bucket reductions have a fixed size regardless of the input range.
        std::array<double, BucketCount - 1> rightCost{};
        Bucket rightBucket;
        for (uint32_t index = BucketCount - 1; index > 0; --index) {
            rightBucket.bounds.Include(buckets[index].bounds);
            rightBucket.count += buckets[index].count;
            rightCost[index - 1] = rightBucket.count ? rightBucket.count * rightBucket.bounds.Area() : 0.0;
        }

        const uint64_t childCapacity = uint64_t{kBvhLeafCapacity} << (kBvhMaximumDepth - range.depth - 1);
        Bucket leftBucket;
        for (uint32_t index = 0; index < BucketCount - 1; ++index) {
            leftBucket.bounds.Include(buckets[index].bounds);
            leftBucket.count += buckets[index].count;
            const uint32_t rightCount = range.end - range.first - leftBucket.count;
            if (leftBucket.count == 0 || rightCount == 0 || leftBucket.count > childCapacity ||
                rightCount > childCapacity)
                continue;
            const double cost = leftBucket.count * leftBucket.bounds.Area() + rightCost[index];
            if (cost < best.cost) {
                best = axisSplit;
                best.cost = cost;
                best.leftCount = leftBucket.count;
                best.lastLeftBucket = index;
            }
        }
    }

    void Partition(uint32_t stop)
    {
        for (; cursor < stop; ++cursor) {
            const auto primitive = result.primitiveOrder[cursor];
            const bool goesLeft = best.BucketOf(centers[primitive]) <= best.lastLeftBucket;
            scratch[goesLeft ? left++ : right++] = primitive;
        }
        if (cursor == range.end) {
            cursor = range.first;
            phase = Phase::Commit;
        }
    }

    void Commit(uint32_t stop)
    {
        std::copy(scratch.begin() + cursor, scratch.begin() + stop, result.primitiveOrder.begin() + cursor);
        cursor = stop;
        if (cursor == range.end)
            ScheduleChildren(range.first + best.leftCount);
    }

    void ScheduleChildren(uint32_t middle)
    {
        const uint32_t depth = range.depth + 1;
        pending[pendingCount++] = {middle, range.end, depth, node, true};
        pending[pendingCount++] = {range.first, middle, depth, node, false};
        NextNode();
    }

    void NextNode()
    {
        if (pendingCount == 0) {
            phase = Phase::Complete;
            return;
        }
        range = pending[--pendingCount];
        cursor = range.first;
        bounds = {};
        centerBounds = {};
        phase = Phase::ScanBounds;
    }

    const std::vector<Bounds> &primitiveBounds;
    std::vector<glm::vec3> centers;
    std::vector<uint32_t> scratch;
    Bvh result;
    std::array<NodeRange, kBvhMaximumDepth + 1> pending;
    std::array<Bucket, BucketCount> buckets;
    NodeRange range{};
    Bounds bounds;
    Bounds centerBounds;
    Split best;
    Split axisSplit;
    uint32_t pendingCount = 0;
    uint32_t cursor = 0;
    uint32_t node = 0;
    uint32_t left = 0;
    uint32_t right = 0;
    unsigned axis = 0;
    Phase phase = Phase::Initialize;
};

BvhBuild::BvhBuild(const std::vector<Bounds> &primitiveBounds) : m_state(std::make_unique<State>(primitiveBounds))
{
}

BvhBuild::~BvhBuild() = default;
BvhBuild::BvhBuild(BvhBuild &&) noexcept = default;
BvhBuild &BvhBuild::operator=(BvhBuild &&) noexcept = default;

bool BvhBuild::Advance(uint32_t primitiveBudget)
{
    return m_state->Advance(primitiveBudget);
}

Bvh BvhBuild::Take()
{
    if (m_state->phase != Phase::Complete)
        throw std::logic_error("BVH build is incomplete");
    return std::move(m_state->result);
}

Bvh BuildBvh(const std::vector<Bounds> &primitiveBounds)
{
    BvhBuild build(primitiveBounds);
    while (!build.Advance(std::numeric_limits<uint32_t>::max())) {
    }
    return build.Take();
}

namespace
{
Bounds NodeBounds(const BvhNode &node) noexcept
{
    return {{node.minimum[0], node.minimum[1], node.minimum[2]}, {node.maximum[0], node.maximum[1], node.maximum[2]}};
}

template <typename NodeAt>
BvhNode FitNode(uint32_t first, uint32_t second, const std::vector<uint32_t> &order,
                const std::vector<Bounds> &primitives, NodeAt child)
{
    Bounds bounds;
    if (second & kBvhLeafBit) {
        const uint32_t count = second & ~kBvhLeafBit;
        for (uint32_t i = 0; i < count; ++i)
            bounds.Include(primitives[order[first + i]]);
    } else {
        bounds = NodeBounds(child(first));
        bounds.Include(NodeBounds(child(second)));
    }
    return {{bounds.minimum.x, bounds.minimum.y, bounds.minimum.z},
            first,
            {bounds.maximum.x, bounds.maximum.y, bounds.maximum.z},
            second};
}
} // namespace

double BvhNodeCost(const BvhNode &node) noexcept
{
    return NodeBounds(node).Area() * ((node.second & kBvhLeafBit) ? (node.second & ~kBvhLeafBit) : 1u);
}

double NormalizeBvhCost(double cost, const std::vector<BvhNode> &nodes) noexcept
{
    const auto area = nodes.empty() ? 0.0 : NodeBounds(nodes.front()).Area();
    return area > 0.0 ? cost / area : 0.0;
}

BvhRefit::BvhRefit(const BvhTopology &topology, const std::vector<Bounds> &primitiveBounds)
    : m_topology(topology), m_bounds(primitiveBounds), m_remaining(topology.links.size())
{
    if (topology.primitiveOrder.size() != primitiveBounds.size())
        throw std::invalid_argument("BVH refit requires the same primitive count");
    m_nodes.reserve(m_remaining);
}

bool BvhRefit::Advance(uint32_t nodeBudget)
{
    const size_t count = m_topology.links.size();
    while (m_remaining && nodeBudget) {
        const auto &[first, second] = m_topology.links[--m_remaining];
        const auto node = FitNode(first, second, m_topology.primitiveOrder, m_bounds,
                                  [&](size_t child) -> const BvhNode & { return m_nodes[count - 1 - child]; });
        m_cost += BvhNodeCost(node);
        m_nodes.push_back(node);
        --nodeBudget;
    }
    if (m_remaining)
        return false;
    const auto end = m_flip + std::min<size_t>(nodeBudget, count / 2 - m_flip);
    for (; m_flip < end; ++m_flip)
        std::swap(m_nodes[m_flip], m_nodes[count - 1 - m_flip]);
    return m_flip == count / 2;
}

double BvhRefit::Cost() const noexcept
{
    return NormalizeBvhCost(m_cost, m_nodes);
}

std::vector<BvhNode> BvhRefit::Take()
{
    return std::move(m_nodes);
}

void RefitBvh(Bvh &tree, const std::vector<Bounds> &primitiveBounds)
{
    if (tree.primitiveOrder.size() != primitiveBounds.size())
        throw std::invalid_argument("BVH refit requires the same primitive count");
    for (auto it = tree.nodes.rbegin(); it != tree.nodes.rend(); ++it)
        *it = FitNode(it->first, it->second, tree.primitiveOrder, primitiveBounds,
                      [&](size_t child) -> const BvhNode & { return tree.nodes[child]; });
}

} // namespace infernux::spatial
