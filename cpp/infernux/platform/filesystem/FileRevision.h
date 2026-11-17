#pragma once

#include <cstdint>
#include <utility>

namespace infernux
{

/// Identity and observed data state of one open file.
struct FileRevision
{
    struct State
    {
        uint64_t size;
        std::pair<int64_t, int64_t> modified;
    };

    State data;
    std::pair<uint64_t, uint64_t> identity;
    std::pair<int64_t, int64_t> changed;

    bool operator==(const FileRevision &other) const
    {
        return data.size == other.data.size && data.modified == other.data.modified && identity == other.identity &&
               changed == other.changed;
    }
};

} // namespace infernux
