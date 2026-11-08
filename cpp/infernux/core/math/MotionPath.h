#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

// Table values include the reference's separate multiply/add rounding. GCC's
// default fast contraction can otherwise fuse across scalar statements.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

namespace infernux::motion
{
using Point = std::array<float, 3>;
class MissingPathControls : public std::logic_error
{
  public:
    MissingPathControls() : std::logic_error("path control array is null")
    {
    }
};

inline double SquaredLength(const Point &vector)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    return double(vector[0]) * vector[0] + double(vector[1]) * vector[1] + double(vector[2]) * vector[2];
}

inline Point ClampLength(Point vector, float maximum)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const float squared = static_cast<float>(SquaredLength(vector));
    if (squared > maximum * maximum) {
        const float magnitude = std::sqrt(squared);
        for (float &value : vector)
            value = (value / magnitude) * maximum;
    }
    return vector;
}

inline std::array<Point, 2> BezierClosingHandles(const Point &start, const Point &end, const Point &firstHandle,
                                                 const Point &lastHandle)
{
    Point gap, incoming, outgoing;
    for (int axis = 0; axis < 3; ++axis) {
        gap[axis] = start[axis] - end[axis];
        incoming[axis] = start[axis] - firstHandle[axis];
        outgoing[axis] = end[axis] - lastHandle[axis];
    }
    // Vector magnitude returns a rounded sqrt of its wider sum; ClampMagnitude
    // separately stores squared magnitude as float before normalizing.
    const float maximum = static_cast<float>(std::sqrt(SquaredLength(gap)));
    incoming = ClampLength(incoming, maximum);
    outgoing = ClampLength(outgoing, maximum);
    for (int axis = 0; axis < 3; ++axis) {
        incoming[axis] += start[axis];
        outgoing[axis] += end[axis];
    }
    return {outgoing, incoming};
}

inline Point BlendBezier(const Point &p0, const Point &p1, const Point &p2, const Point &p3, float first,
                         float middleStart, float middleEnd, float last)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    Point result;
    for (int axis = 0; axis < 3; ++axis)
        result[axis] = ((p0[axis] * first + p1[axis] * middleStart) + p2[axis] * middleEnd) + p3[axis] * last;
    return result;
}

inline Point EvaluateCubicBezier(const Point &p0, const Point &p1, const Point &p2, const Point &p3, float t)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const float complement = 1.0f - t;
    const float squaredT = t * t, squaredComplement = complement * complement;
    const float first = squaredComplement * complement, last = squaredT * t;
    const float leadingStart = 3.0f * squaredComplement, leadingEnd = 3.0f * complement;
    return BlendBezier(p0, p1, p2, p3, first, leadingStart * t, leadingEnd * squaredT, last);
}

// The segment kernel owns no Python state and allocates no intermediate vectors.
inline Point EvaluateSegment(int kind, const Point &p0, const Point &p1, const Point &p2, const Point &p3, float t)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    Point result;
    if (kind == 2) {
        const float complement = 1.0f - t;
        const float squaredT = t * t, squaredComplement = complement * complement;
        const float cubicT = squaredT * t, cubicComplement = squaredComplement * complement;
        const float middleStart = static_cast<float>(3.0 * squaredComplement * t);
        const float middleEnd = static_cast<float>(3.0 * complement * squaredT);
        return BlendBezier(p0, p1, p2, p3, cubicComplement, middleStart, middleEnd, cubicT);
    }
    const double u = t;
    for (int axis = 0; axis < 3; ++axis) {
        if (kind == 0) {
            result[axis] = p1[axis] + (p2[axis] - p1[axis]) * t;
        } else {
            const float a = ((p1[axis] * 3 - p0[axis]) - p2[axis] * 3) + p3[axis];
            const float b = ((p0[axis] * 2 - p1[axis] * 5) + p2[axis] * 4) - p3[axis];
            result[axis] =
                (((a * static_cast<float>(u * u * u) + b * static_cast<float>(u * u)) + (p2[axis] - p0[axis]) * t) +
                 p1[axis] * 2) *
                .5f;
        }
    }
    return result;
}

template <typename Storage = Point> class PathDataSnapshot
{
    struct Tables
    {
        std::vector<float> parameters, distances, waypointLengths;
    };

  public:
    PathDataSnapshot(std::vector<Storage> points, int kind, int resolution, bool closed)
        : points_(std::make_shared<std::vector<Storage>>(std::move(points))),
          controls_(kind == 0 ? nullptr : std::make_shared<std::vector<Storage>>()), kind_(kind), closed_(closed)
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
        if (kind < 0 || kind > 2 || resolution < 1 || points_->size() < 2 || (kind == 2 && (points_->size() - 1) % 3))
            throw std::invalid_argument("invalid motion path recipe");
        for (const auto &point : *points_)
            for (int axis = 0; axis < 3; ++axis)
                if (!std::isfinite(point[axis]))
                    throw std::invalid_argument("motion path points must be finite");
        segments_ = static_cast<int>((points_->size() - 1) / (kind == 2 ? 3 : 1));
        if (kind == 2) {
            auto endpoints = std::make_shared<std::vector<Storage>>();
            endpoints->reserve(segments_ + 1);
            endpoints->push_back((*points_)[0]);
            controls_->reserve(2 * segments_);
            for (int segment = 0; segment < segments_; ++segment) {
                const auto index = 1 + 3 * segment;
                endpoints->push_back((*points_)[index]);
                controls_->push_back((*points_)[index + 1]);
                controls_->push_back((*points_)[index + 2]);
            }
            points_ = std::move(endpoints);
        } else if (kind == 1) {
            controls_->resize(4, Storage{0, 0, 0});
            (*controls_)[0] = closed ? (*points_)[points_->size() - 2] : (*points_)[1];
            if (closed)
                (*controls_)[2] = (*points_)[1];
            else
                for (int axis = 0; axis < 3; ++axis) {
                    const float last = points_->back()[axis];
                    (*controls_)[2][axis] = last + (last - (*points_)[points_->size() - 2][axis]);
                }
        }
        auto tables = std::make_shared<Tables>();
        tables_ = tables;
        auto &parameters_ = tables->parameters;
        auto &distances_ = tables->distances;
        auto &waypointLengths_ = tables->waypointLengths;
        const size_t count = kind == 0 ? segments_ : (static_cast<size_t>(segments_) + 1) * resolution;
        parameters_.reserve(count + 1);
        distances_.reserve(count + 1);
        Point previous = Evaluate(0);
        float distance = 0;
        const float interval = 1.0f / static_cast<float>(count);
        for (size_t index = 0; index <= count; ++index) {
            const float percentage = interval * static_cast<float>(index);
            const double parameter = kind == 0 ? static_cast<double>(index) : double(percentage) * segments_;
            const Point point = kind == 0 ? PointAt(index) : Evaluate(parameter);
            if (index)
                distance += Distance(point, previous);
            parameters_.push_back(kind == 0 ? static_cast<float>(index) : percentage);
            distances_.push_back(distance);
            previous = point;
        }
        if (kind == 0)
            for (size_t index = 1; index < parameters_.size(); ++index)
                parameters_[index] = distances_[index] / distance;
        waypointLengths_.reserve(segments_ + 1);
        waypointLengths_.push_back(0);
        const int steps = kind == 0 ? 1 : resolution;
        const float partialInterval = 1.0f / steps;
        for (int segment = 0; segment < segments_; ++segment) {
            previous = Evaluate(segment);
            float partial = 0;
            for (int index = 1; index <= steps; ++index) {
                const float percentage = partialInterval * index;
                const Point point = kind == 0 ? PointAt(segment + 1) : EvaluateOnSegment(segment, percentage);
                partial += Distance(point, previous);
                previous = point;
            }
            waypointLengths_.push_back(partial);
        }
    }

    PathDataSnapshot(const PathDataSnapshot &source)
        : points_(std::make_shared<std::vector<Storage>>(*source.points_)),
          controls_(source.controls_ ? std::make_shared<std::vector<Storage>>(*source.controls_) : nullptr),
          kind_(source.kind_), segments_(source.segments_), closed_(source.closed_), tables_(source.tables_)
    {
    }
    PathDataSnapshot(PathDataSnapshot &&) = default;

    Point Evaluate(double parameter) const
    {
        if (std::isnan(parameter)) {
            const float value = std::numeric_limits<float>::quiet_NaN();
            return {value, value, value};
        }
        const int segments = Segments();
        const int segment = parameter < 0           ? (kind_ == 0 ? 0 : -1)
                            : parameter >= segments ? segments - 1
                                                    : static_cast<int>(parameter);
        const float t = static_cast<float>(parameter - segment);
        return EvaluateOnSegment(segment, t);
    }

    double ParameterAt(double progress) const
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
        const auto &parameters_ = tables_->parameters;
        const auto &distances_ = tables_->distances;
        if (kind_ != 0) {
            const float percentage = std::clamp(static_cast<float>(progress), 0.0f, 1.0f);
            if (percentage == 0 || percentage == 1 || Length() <= 0)
                return double(percentage) * Segments();
            const float distance = static_cast<float>(Length()) * percentage;
            const auto upper = std::upper_bound(distances_.begin(), distances_.end(), distance);
            const size_t index =
                std::min(distances_.size() - 2,
                         static_cast<size_t>(std::max<std::ptrdiff_t>(0, upper - distances_.begin() - 1)));
            const double from = parameters_[index], to = parameters_[index + 1];
            const double span = double(distances_[index + 1]) - distances_[index];
            const double fraction = span ? (distance - distances_[index]) / span : 0;
            const float normalized = static_cast<float>(from + fraction * (to - from));
            return double(normalized) * Segments();
        }
        const double distance = std::clamp(progress, 0.0, 1.0) * Length();
        const auto upper = std::upper_bound(distances_.begin(), distances_.end(), distance);
        const size_t index = std::min(distances_.size() - 2,
                                      static_cast<size_t>(std::max<std::ptrdiff_t>(0, upper - distances_.begin() - 1)));
        const double span = double(distances_[index + 1]) - distances_[index];
        const double fraction = span ? (distance - distances_[index]) / span : 0;
        return static_cast<double>(index) + fraction;
    }

    Point Sample(double progress) const
    {
        return kind_ == 0 ? SampleLinear(progress) : Evaluate(ParameterAt(progress));
    }
    double Length() const
    {
        return tables_->distances.back();
    }
    int Segments() const
    {
        return static_cast<int>(points_->size()) - 1;
    }
    std::vector<double> Parameters() const
    {
        std::vector<double> result;
        result.reserve(tables_->parameters.size());
        for (size_t index = 0; index < tables_->parameters.size(); ++index)
            result.push_back(kind_ == 0 ? static_cast<double>(index) : double(tables_->parameters[index]) * Segments());
        return result;
    }
    const std::vector<float> &Distances() const
    {
        return tables_->distances;
    }
    const std::vector<float> &WaypointLengths() const
    {
        return tables_->waypointLengths;
    }
    std::vector<float> WaypointDistances() const
    {
        std::vector<float> result;
        result.reserve(tables_->waypointLengths.size() - 1);
        float cumulative = 0;
        for (size_t index = 1; index < tables_->waypointLengths.size(); ++index) {
            cumulative += tables_->waypointLengths[index];
            result.push_back(cumulative);
        }
        return result;
    }

    // Moving a finalized path preserves its arc tables. Only the small control
    // geometry is copied; existing immutable samplers keep their original pose.
    PathDataSnapshot Translated(const Point &offset) const
    {
        return PathDataSnapshot(*this).TranslateOwned(offset);
    }
    // The caller transfers a private frozen copy; no second geometry copy is
    // needed before translating it outside the interpreter lock.
    PathDataSnapshot TranslateOwned(const Point &offset) &&
    {
        for (auto &point : *points_)
            for (int axis = 0; axis < 3; ++axis)
                point[axis] += offset[axis];
        if (controls_)
            for (auto &point : *controls_)
                for (int axis = 0; axis < 3; ++axis)
                    point[axis] += offset[axis];
        return std::move(*this);
    }
    size_t PointCount() const
    {
        return points_->size();
    }
    std::vector<Storage> &Points()
    {
        return *points_;
    }
    const std::shared_ptr<std::vector<Storage>> &PointStorage() const
    {
        return points_;
    }
    const std::shared_ptr<std::vector<Storage>> &ControlStorage() const
    {
        return controls_;
    }
    PathDataSnapshot WithWaypoints(std::shared_ptr<std::vector<Storage>> points) const
    {
        return PathDataSnapshot(std::move(points), controls_, kind_, segments_, closed_, tables_);
    }
    PathDataSnapshot WithControls(std::shared_ptr<std::vector<Storage>> controls) const
    {
        return PathDataSnapshot(points_, std::move(controls), kind_, segments_, closed_, tables_);
    }
    size_t RecipeSize() const
    {
        return kind_ == 2 && !points_->empty() ? (points_->size() - 1) * 3 + 1 : points_->size();
    }
    Storage &RecipePoint(size_t index)
    {
        if (kind_ != 2 || index == 0)
            return points_->at(index);
        const size_t group = (index - 1) / 3, component = (index - 1) % 3;
        return component == 0 ? points_->at(group + 1) : ControlValue(2 * group + component - 1);
    }

  private:
    PathDataSnapshot(std::shared_ptr<std::vector<Storage>> points, std::shared_ptr<std::vector<Storage>> controls,
                     int kind, int segments, bool closed, std::shared_ptr<const Tables> tables)
        : points_(std::move(points)), controls_(std::move(controls)), kind_(kind), segments_(segments), closed_(closed),
          tables_(std::move(tables))
    {
    }

    Point SampleLinear(double progress) const
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
        const float percentage = static_cast<float>(progress);
        if (percentage <= 0)
            return PointAt(0);
        const auto &times = tables_->parameters;
        const auto upper = std::lower_bound(times.begin() + 1, times.end(), percentage,
                                            [](float time, float value) { return !(time >= value); });
        if (upper == times.end())
            return PointAt(0);
        const size_t end = static_cast<size_t>(upper - times.begin()), start = end - 1;
        const float partialPercentage = percentage - times[start];
        const float distance = static_cast<float>(Length() * partialPercentage);
        Point delta;
        const Point from = PointAt(start), to = PointAt(end);
        for (int axis = 0; axis < 3; ++axis)
            delta[axis] = to[axis] - from[axis];
        delta = ClampLength(delta, distance);
        for (int axis = 0; axis < 3; ++axis)
            delta[axis] += from[axis];
        return delta;
    }

    Point EvaluateOnSegment(int segment, float t) const
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
        if (kind_ == 2)
            return EvaluateSegment(kind_, PointAt(segment), ControlAt(2 * segment), ControlAt(2 * segment + 1),
                                   PointAt(segment + 1), t);
        const Point p1 = PointAt(segment), p2 = PointAt(segment + 1);
        if (kind_ == 0)
            return EvaluateSegment(0, p1, p1, p2, p2, t);
        const Point p0 = segment ? PointAt(segment - 1) : ControlAt(0);
        const Point p3 = segment + 2 < static_cast<int>(points_->size()) ? PointAt(segment + 2) : ControlAt(2);
        return EvaluateSegment(kind_, p0, p1, p2, p3, t);
    }

    Point PointAt(size_t index) const
    {
        const auto &value = points_->at(index);
        return {value[0], value[1], value[2]};
    }
    Point ControlAt(size_t index) const
    {
        const auto &value = ControlValue(index);
        return {value[0], value[1], value[2]};
    }
    Storage &ControlValue(size_t index) const
    {
        if (!controls_)
            throw MissingPathControls();
        return controls_->at(index);
    }

    static float Distance(const Point &a, const Point &b)
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
        const Point delta{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
        return static_cast<float>(std::sqrt(SquaredLength(delta)));
    }
    std::shared_ptr<std::vector<Storage>> points_, controls_;
    int kind_, segments_;
    bool closed_;
    std::shared_ptr<const Tables> tables_;
};
using PathSnapshot = PathDataSnapshot<>;
} // namespace infernux::motion

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
