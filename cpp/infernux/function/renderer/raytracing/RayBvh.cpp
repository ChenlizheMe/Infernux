#include "RayBvh.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace infernux::raytracing
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

class Builder
{
  public:
    explicit Builder(const std::vector<Bounds> &bounds)
        : m_bounds(bounds), m_centers(bounds.size()), m_scratch(bounds.size())
    {
        for (size_t index = 0; index < bounds.size(); ++index) {
            const auto &bound = bounds[index];
            for (unsigned axis = 0; axis < 3; ++axis)
                if (!std::isfinite(bound.minimum[axis]) || !std::isfinite(bound.maximum[axis]) ||
                    bound.minimum[axis] > bound.maximum[axis])
                    throw std::invalid_argument("Ray acceleration requires finite primitive bounds");
            m_centers[index] = bound.Center();
        }
        m_result.primitiveOrder.resize(bounds.size());
        std::iota(m_result.primitiveOrder.begin(), m_result.primitiveOrder.end(), 0u);
        m_result.nodes.reserve((bounds.size() + kBvhLeafCapacity - 1) / kBvhLeafCapacity * 2);
    }

    [[nodiscard]] Bvh Build()
    {
        Emit(0, static_cast<uint32_t>(m_bounds.size()), 0);
        return std::move(m_result);
    }

  private:
    [[nodiscard]] Split FindSplit(uint32_t first, uint32_t end, const Bounds &centers, uint32_t depth) const
    {
        Split best;
        const uint64_t childCapacity = uint64_t{kBvhLeafCapacity} << (kBvhMaximumDepth - depth - 1);
        for (unsigned axis = 0; axis < 3; ++axis) {
            const double extent = static_cast<double>(centers.maximum[axis]) - centers.minimum[axis];
            if (extent == 0.0)
                continue;

            Split split;
            split.minimum = centers.minimum[axis];
            split.bucketsPerUnit = static_cast<double>(BucketCount) / extent;
            split.axis = axis;
            std::array<Bucket, BucketCount> buckets{};
            for (uint32_t index = first; index < end; ++index) {
                const auto primitive = m_result.primitiveOrder[index];
                auto &bucket = buckets[split.BucketOf(m_centers[primitive])];
                bucket.bounds.Include(m_bounds[primitive]);
                ++bucket.count;
            }

            std::array<double, BucketCount - 1> rightCost{};
            Bucket right;
            for (uint32_t index = BucketCount - 1; index > 0; --index) {
                right.bounds.Include(buckets[index].bounds);
                right.count += buckets[index].count;
                rightCost[index - 1] = right.count ? right.count * right.bounds.Area() : 0.0;
            }

            Bucket left;
            for (uint32_t index = 0; index < BucketCount - 1; ++index) {
                left.bounds.Include(buckets[index].bounds);
                left.count += buckets[index].count;
                const uint32_t rightCount = end - first - left.count;
                if (left.count == 0 || rightCount == 0 || left.count > childCapacity || rightCount > childCapacity)
                    continue;
                const double cost = left.count * left.bounds.Area() + rightCost[index];
                if (cost < best.cost) {
                    best = split;
                    best.cost = cost;
                    best.leftCount = left.count;
                    best.lastLeftBucket = index;
                }
            }
        }
        return best;
    }

    [[nodiscard]] uint32_t Partition(uint32_t first, uint32_t end, const Split &split)
    {
        if (split.leftCount == 0)
            return first + (end - first) / 2;
        const uint32_t middle = first + split.leftCount;
        uint32_t left = first, right = middle;
        for (uint32_t index = first; index < end; ++index) {
            const auto primitive = m_result.primitiveOrder[index];
            const bool goesLeft = split.BucketOf(m_centers[primitive]) <= split.lastLeftBucket;
            m_scratch[goesLeft ? left++ : right++] = primitive;
        }
        std::copy(m_scratch.begin() + first, m_scratch.begin() + end, m_result.primitiveOrder.begin() + first);
        return middle;
    }

    uint32_t Emit(uint32_t first, uint32_t end, uint32_t depth)
    {
        Bounds bounds, centers;
        for (uint32_t index = first; index < end; ++index) {
            const auto primitive = m_result.primitiveOrder[index];
            bounds.Include(m_bounds[primitive]);
            const auto center = m_centers[primitive];
            centers.Include({center, center});
        }

        const uint32_t node = static_cast<uint32_t>(m_result.nodes.size());
        const uint32_t count = end - first;
        m_result.nodes.push_back({{bounds.minimum.x, bounds.minimum.y, bounds.minimum.z},
                                  first,
                                  {bounds.maximum.x, bounds.maximum.y, bounds.maximum.z},
                                  kBvhLeafBit | count});
        if (count > kBvhLeafCapacity) {
            const auto middle = Partition(first, end, FindSplit(first, end, centers, depth));
            const auto left = Emit(first, middle, depth + 1);
            const auto right = Emit(middle, end, depth + 1);
            m_result.nodes[node].first = left;
            m_result.nodes[node].second = right;
        }
        return node;
    }

    const std::vector<Bounds> &m_bounds;
    std::vector<glm::vec3> m_centers;
    std::vector<uint32_t> m_scratch;
    Bvh m_result;
};
} // namespace

Bvh BuildBvh(const std::vector<Bounds> &primitiveBounds)
{
    if (primitiveBounds.empty())
        return {};
    if (primitiveBounds.size() > (kBvhLeafBit / 2u))
        throw std::length_error("Ray acceleration structure exceeds its node index range");
    return Builder(primitiveBounds).Build();
}

} // namespace infernux::raytracing
