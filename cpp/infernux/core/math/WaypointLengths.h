#pragma once

#include "Float32Array.h"
#include <stdexcept>

namespace infernux::motion
{
class MissingWaypointLengths : public std::logic_error
{
  public:
    MissingWaypointLengths() : std::logic_error("path waypoint-length array is null")
    {
    }
};

inline int WaypointIndex(const Float32Array *lengths, int pointCount, float percentage, float length, bool forward)
{
    if (percentage <= 0)
        return 0;
    if (percentage >= 1)
        return pointCount - 1;
    if (!lengths)
        throw MissingWaypointLengths();
    const float distance = length * percentage;
    float cumulative = 0;
    const auto count = lengths->values.size();
    for (size_t index = 0; index + 1 < count; ++index) {
        cumulative += lengths->values[index];
        if (cumulative == distance)
            return static_cast<int>(index);
        if (cumulative > distance)
            return static_cast<int>(index) - (forward ? 1 : 0);
    }
    return count > 1 ? static_cast<int>(count) - (forward ? 2 : 1) : 0;
}

inline float WaypointFraction(const Float32Array *lengths, int index, float length)
{
    if (!lengths)
        throw MissingWaypointLengths();
    float cumulative = 0;
    for (int point = 0; point <= index; ++point)
        cumulative += lengths->values.at(static_cast<size_t>(point));
    return cumulative / length;
}
} // namespace infernux::motion
