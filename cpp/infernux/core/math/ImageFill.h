#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace infernux::imagefill
{
struct Point
{
    float x, y;
};
struct Polygon
{
    std::array<Point, 8> points{};
    size_t count = 0;
};
struct Geometry
{
    std::array<Polygon, 4> polygons{};
    size_t count = 0;
};

// Independent convex half-plane clipping. Coordinates are bottom-left unit
// coordinates; texture and screen transforms are applied only by the consumer.
inline Polygon Clip(const Polygon &input, Point pivot, Point direction, float sign)
{
    Polygon result;
    if (!input.count)
        return result;
    const auto distance = [&](Point p) {
        return sign * (direction.x * (p.y - pivot.y) - direction.y * (p.x - pivot.x));
    };
    Point previous = input.points[input.count - 1];
    float previousDistance = distance(previous);
    for (size_t i = 0; i < input.count; ++i) {
        const Point current = input.points[i];
        const float currentDistance = distance(current);
        if ((previousDistance >= 0) != (currentDistance >= 0)) {
            const float t = previousDistance / (previousDistance - currentDistance);
            result.points[result.count++] = {previous.x + (current.x - previous.x) * t,
                                             previous.y + (current.y - previous.y) * t};
        }
        if (currentDistance >= 0)
            result.points[result.count++] = current;
        previous = current;
        previousDistance = currentDistance;
    }
    return result;
}

inline Point Direction(float degrees)
{
    const int quarter = static_cast<int>(std::round(degrees / 90.0f));
    if (std::fabs(degrees - quarter * 90.0f) < 0.00001f) {
        constexpr std::array<Point, 4> axes{{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}};
        return axes[(quarter % 4 + 4) % 4];
    }
    const float angle = degrees * 0.017453292519943295f;
    return {std::cos(angle), std::sin(angle)};
}

inline Polygon Rectangle(float x0 = 0, float y0 = 0, float x1 = 1, float y1 = 1)
{
    Polygon polygon;
    polygon.count = 4;
    polygon.points[0] = {x0, y0};
    polygon.points[1] = {x1, y0};
    polygon.points[2] = {x1, y1};
    polygon.points[3] = {x0, y1};
    return polygon;
}

inline Geometry Build(float amount, int method, int origin, bool clockwise)
{
    if (!std::isfinite(amount))
        throw std::invalid_argument("Image fill amount must be finite");
    if (method < 0 || method > 4 || origin < 0 || origin > (method < 2 ? 1 : 3))
        throw std::invalid_argument("Invalid image fill method or origin");
    amount = std::clamp(amount, 0.0f, 1.0f);
    Geometry result;
    if (amount < 0.001f)
        return result;
    if (method < 2 || amount >= 1) {
        float x0 = 0, y0 = 0, x1 = 1, y1 = 1;
        if (method == 0) {
            if (origin)
                x0 = 1 - amount;
            else
                x1 = amount;
        }
        if (method == 1) {
            if (origin)
                y0 = 1 - amount;
            else
                y1 = amount;
        }
        result.polygons[result.count++] = Rectangle(x0, y0, x1, y1);
        return result;
    }
    const int pieces = method == 2 ? 1 : method == 3 ? 2 : 4;
    Point pivot{0.5f, 0.5f};
    float start = -90.0f + origin * 90.0f;
    if (method == 2) {
        constexpr std::array<Point, 4> corners{{{0, 0}, {0, 1}, {1, 1}, {1, 0}}};
        pivot = corners[origin];
        start = -origin * 90.0f + (clockwise ? 90.0f : 0.0f);
    } else if (method == 3) {
        constexpr std::array<Point, 4> edges{{{0.5f, 0}, {0, 0.5f}, {0.5f, 1}, {1, 0.5f}}};
        pivot = edges[origin];
        start = -origin * 90.0f + (clockwise ? 180.0f : 0.0f);
    }
    const float sign = clockwise ? -1.0f : 1.0f;
    for (int i = 0; i < pieces; ++i) {
        float fraction = std::clamp(amount * pieces - i, 0.0f, 1.0f);
        if (fraction < 0.001f)
            continue;
        const float begin = start + sign * i * 90.0f;
        const Point middle = Direction(begin + sign * 45.0f);
        float x0 = 0, y0 = 0, x1 = 1, y1 = 1;
        // Each radial quadrant has its own normalized chart. This keeps the
        // sweep stable on non-square images and on the two half-image charts.
        if (method == 4 || (method == 3 && origin % 2 == 0)) {
            if (middle.x < 0)
                x1 = 0.5f;
            else
                x0 = 0.5f;
        }
        if (method == 4 || (method == 3 && origin % 2 != 0)) {
            if (middle.y < 0)
                y1 = 0.5f;
            else
                y0 = 0.5f;
        }
        const Point localPivot{(pivot.x - x0) / (x1 - x0), (pivot.y - y0) / (y1 - y0)};
        // Measured endpoint snapping differs for alternate pivot corners and
        // sweep directions. Keep it in the normalized chart, before clipping.
        const int corner = localPivot.x < 0.5f ? (localPivot.y < 0.5f ? 0 : 1) : (localPivot.y < 0.5f ? 3 : 2);
        if (fraction > 0.999f && clockwise == bool(corner % 2))
            fraction = 1;
        const float end = begin + sign * fraction * 90.0f;
        auto polygon = Clip(Rectangle(), localPivot, Direction(begin), sign);
        polygon = Clip(polygon, localPivot, Direction(end), -sign);
        if (polygon.count < 3)
            continue;
        for (size_t v = 0; v < polygon.count; ++v) {
            polygon.points[v] = {x0 + (x1 - x0) * polygon.points[v].x, y0 + (y1 - y0) * polygon.points[v].y};
        }
        result.polygons[result.count++] = polygon;
    }
    return result;
}
} // namespace infernux::imagefill
