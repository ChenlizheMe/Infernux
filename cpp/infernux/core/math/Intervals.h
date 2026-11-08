#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace infernux::intervals
{

// Sorted, disjoint half-open ranges. Callers validate their own byte/pixel
// bounds; the same algebra serves resident buffers and graph prefix proofs.
using Set = std::vector<std::pair<uint64_t, uint64_t>>;

inline void Add(Set &target, uint64_t begin, uint64_t end)
{
    if (begin == end)
        return;
    auto first = std::lower_bound(target.begin(), target.end(), begin,
                                  [](const auto &range, uint64_t value) { return range.second < value; });
    if (first == target.end() || end < first->first) {
        target.insert(first, {begin, end});
        return;
    }
    if (first->first <= begin && end <= first->second)
        return;
    first->first = std::min(first->first, begin);
    first->second = std::max(first->second, end);
    auto next = first + 1;
    while (next != target.end() && next->first <= first->second) {
        first->second = std::max(first->second, next->second);
        ++next;
    }
    target.erase(first + 1, next);
}

inline void Add(Set &target, const Set &added)
{
    for (const auto &range : added)
        Add(target, range.first, range.second);
}

inline bool Contains(const Set &known, uint64_t begin, uint64_t end)
{
    const auto found = std::upper_bound(known.begin(), known.end(), begin,
                                        [](uint64_t value, const auto &range) { return value < range.first; });
    return found != known.begin() && begin >= (found - 1)->first && end <= (found - 1)->second;
}

inline Set Intersect(const Set &known, const Set &region)
{
    Set result;
    std::size_t left = 0, right = 0;
    while (left < known.size() && right < region.size()) {
        const auto begin = std::max(known[left].first, region[right].first);
        const auto end = std::min(known[left].second, region[right].second);
        if (begin < end)
            result.emplace_back(begin, end);
        if (known[left].second < region[right].second)
            ++left;
        else
            ++right;
    }
    return result;
}

inline void Remove(Set &known, const Set &region)
{
    Set retained;
    std::size_t removed = 0;
    for (const auto &range : known) {
        auto cursor = range.first;
        while (removed < region.size() && region[removed].second <= cursor)
            ++removed;
        auto index = removed;
        while (index < region.size() && region[index].first < range.second) {
            if (cursor < region[index].first)
                retained.emplace_back(cursor, region[index].first);
            cursor = std::max(cursor, region[index].second);
            if (cursor >= range.second)
                break;
            ++index;
        }
        if (cursor < range.second)
            retained.emplace_back(cursor, range.second);
        removed = index;
    }
    known = std::move(retained);
}

} // namespace infernux::intervals
