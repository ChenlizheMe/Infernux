#pragma once

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace infernux
{

struct WorldUIOcclusionPolicy
{
    uint64_t ignoredOccluderId = 0;
    bool alwaysOnTop = false;
};

struct WorldUIOcclusionRun
{
    uint32_t firstOrdinal = 0;
    uint32_t endOrdinal = 0;
    uint64_t ignoredOccluderId = 0;
    bool alwaysOnTop = false;

    [[nodiscard]] bool operator==(const WorldUIOcclusionRun &other) const noexcept
    {
        return firstOrdinal == other.firstOrdinal && endOrdinal == other.endOrdinal &&
               ignoredOccluderId == other.ignoredOccluderId && alwaysOnTop == other.alwaysOnTop;
    }
};

struct WorldUIOcclusionPlan
{
    std::vector<WorldUIOcclusionRun> runs;
    std::vector<uint64_t> uniqueIgnoredObjectIds;
};

[[nodiscard]] inline WorldUIOcclusionPlan
BuildWorldUIOcclusionPlan(const std::vector<WorldUIOcclusionPolicy> &sortedPolicies)
{
    WorldUIOcclusionPlan plan;
    std::unordered_set<uint64_t> seen;
    for (uint32_t ordinal = 0; ordinal < sortedPolicies.size(); ++ordinal) {
        const auto &policy = sortedPolicies[ordinal];
        const uint64_t ignoredId = policy.alwaysOnTop ? 0 : policy.ignoredOccluderId;
        if (ignoredId && seen.insert(ignoredId).second)
            plan.uniqueIgnoredObjectIds.push_back(ignoredId);
        if (!plan.runs.empty() && plan.runs.back().ignoredOccluderId == ignoredId &&
            plan.runs.back().alwaysOnTop == policy.alwaysOnTop) {
            plan.runs.back().endOrdinal = ordinal + 1;
        } else {
            plan.runs.push_back({ordinal, ordinal + 1, ignoredId, policy.alwaysOnTop});
        }
    }
    return plan;
}

} // namespace infernux
