#pragma once

#include <array>
#include <cmath>

namespace infernux::ui
{
// Canvas-space geometry is shared by drawing, pointer mapping and editor tools.
// Keep logical pixels in double, like the existing layout; the renderer owns
// its final float vertex conversion. This snapshot never owns another pose.
class ScreenRect
{
  public:
    using Point = std::array<double, 2>;
    using Rect = std::array<double, 4>;

    ScreenRect(Rect rect, double rotation, Point pivot = {.5, .5}) : rect_(rect), pivot_(pivot)
    {
        rotation_ = std::fmod(rotation, 360.0);
        if (rotation_ < 0)
            rotation_ += 360.0;
        if (rotation_ == 0)
            rotation_ = 0;
        const double radians = rotation_ * 0.017453292519943295;
        sine_ = std::sin(radians);
        cosine_ = std::cos(radians);
    }

    const Rect &GetRect() const
    {
        return rect_;
    }
    Point PivotPosition() const
    {
        return {rect_[0] + rect_[2] * pivot_[0], rect_[1] + rect_[3] * pivot_[1]};
    }
    Point Center() const
    {
        return ToCanvas({rect_[2] * .5, rect_[3] * .5});
    }
    Rect RenderRect() const
    {
        if (rotation_ == 0 || (pivot_[0] == .5 && pivot_[1] == .5))
            return rect_;
        const auto center = Center();
        return {center[0] - rect_[2] * .5, center[1] - rect_[3] * .5, rect_[2], rect_[3]};
    }

    Point ToLocal(Point point) const
    {
        const auto origin = PivotPosition();
        const double x = point[0] - origin[0], y = point[1] - origin[1];
        return {x * cosine_ + y * sine_ + rect_[2] * pivot_[0], -x * sine_ + y * cosine_ + rect_[3] * pivot_[1]};
    }

    Point ToCanvas(Point point) const
    {
        const auto origin = PivotPosition();
        const double x = point[0] - rect_[2] * pivot_[0], y = point[1] - rect_[3] * pivot_[1];
        return {origin[0] + x * cosine_ - y * sine_, origin[1] + x * sine_ + y * cosine_};
    }

    std::array<Point, 4> Corners() const
    {
        return {{ToCanvas({0, 0}), ToCanvas({rect_[2], 0}), ToCanvas({rect_[2], rect_[3]}), ToCanvas({0, rect_[3]})}};
    }

    Rect VisualRect() const
    {
        if (std::fabs(rotation_) < .001)
            return rect_;
        const double width = rect_[2] * std::fabs(cosine_) + rect_[3] * std::fabs(sine_);
        const double height = rect_[2] * std::fabs(sine_) + rect_[3] * std::fabs(cosine_);
        const auto center = Center();
        return {center[0] - width * .5, center[1] - height * .5, width, height};
    }

    bool Contains(Point point, double tolerance = 0) const
    {
        if (tolerance < 0)
            tolerance = 0;
        if (std::fabs(rotation_) < .001)
            return point[0] >= rect_[0] - tolerance && point[0] <= rect_[0] + rect_[2] + tolerance &&
                   point[1] >= rect_[1] - tolerance && point[1] <= rect_[1] + rect_[3] + tolerance;
        const auto local = ToLocal(point);
        return local[0] >= -tolerance && local[0] <= rect_[2] + tolerance && local[1] >= -tolerance &&
               local[1] <= rect_[3] + tolerance;
    }

  private:
    Rect rect_;
    Point pivot_;
    double rotation_, sine_, cosine_;
};
} // namespace infernux::ui
