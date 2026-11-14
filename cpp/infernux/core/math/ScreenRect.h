#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

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

    ScreenRect(Rect rect, double rotation, Point pivot = {.5, .5}, Point scale = {1, 1})
        : rect_(rect), pivot_(pivot), scale_(scale)
    {
        if (!std::isfinite(scale_[0]) || !std::isfinite(scale_[1]))
            throw std::invalid_argument("Screen UI local scale must be finite");
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
        // Drawing records unscaled vertices. Its native per-element transform
        // applies local scale after recording; do not bake it into layout too.
        const auto origin = PivotPosition();
        const double x = rect_[2] * (.5 - pivot_[0]), y = rect_[3] * (.5 - pivot_[1]);
        const Point center{origin[0] + x * cosine_ - y * sine_, origin[1] + x * sine_ + y * cosine_};
        return {center[0] - rect_[2] * .5, center[1] - rect_[3] * .5, rect_[2], rect_[3]};
    }

    Point ToLocal(Point point) const
    {
        if (scale_[0] == 0 || scale_[1] == 0) {
            const auto nan = std::numeric_limits<double>::quiet_NaN();
            return {nan, nan};
        }
        const auto origin = PivotPosition();
        const double x = point[0] - origin[0], y = point[1] - origin[1];
        return {(x * cosine_ + y * sine_) / scale_[0] + rect_[2] * pivot_[0],
                (-x * sine_ + y * cosine_) / scale_[1] + rect_[3] * pivot_[1]};
    }

    Point ToCanvas(Point point) const
    {
        const auto origin = PivotPosition();
        const double x = (point[0] - rect_[2] * pivot_[0]) * scale_[0],
                     y = (point[1] - rect_[3] * pivot_[1]) * scale_[1];
        return {origin[0] + x * cosine_ - y * sine_, origin[1] + x * sine_ + y * cosine_};
    }

    std::array<Point, 4> Corners() const
    {
        return {{ToCanvas({0, 0}), ToCanvas({rect_[2], 0}), ToCanvas({rect_[2], rect_[3]}), ToCanvas({0, rect_[3]})}};
    }

    Rect VisualRect() const
    {
        const double width = std::fabs(rect_[2] * scale_[0] * cosine_) + std::fabs(rect_[3] * scale_[1] * sine_);
        const double height = std::fabs(rect_[2] * scale_[0] * sine_) + std::fabs(rect_[3] * scale_[1] * cosine_);
        const auto center = Center();
        return {center[0] - width * .5, center[1] - height * .5, width, height};
    }

    bool Contains(Point point, double tolerance = 0) const
    {
        if (tolerance < 0)
            tolerance = 0;
        if (scale_[0] == 0 || scale_[1] == 0 || rect_[2] <= 0 || rect_[3] <= 0)
            return false;
        const auto local = ToLocal(point);
        const double toleranceX = tolerance / std::fabs(scale_[0]), toleranceY = tolerance / std::fabs(scale_[1]);
        return local[0] >= -toleranceX && local[0] <= rect_[2] + toleranceX && local[1] >= -toleranceY &&
               local[1] <= rect_[3] + toleranceY;
    }

  private:
    Rect rect_;
    Point pivot_;
    Point scale_;
    double rotation_, sine_, cosine_;
};
} // namespace infernux::ui
