#pragma once

#include "QuaternionEuler.h"
#include "QuaternionLookRotation.h"
#include "TweenNumerics.h"

namespace infernux::tween
{
template <typename Vector> Point EulerChange(const Vector &start, const Vector &end, int mode, bool relative)
{
    Point result;
    for (size_t i = 0; i < result.size(); ++i) {
        if (mode >= 2 && !relative)
            result[i] = end[i];
        else {
            result[i] = (mode == 0 && !relative ? std::fmod(end[i], 360.f) : end[i]) - start[i];
            if (mode == 0 && !relative) {
                if (result[i] > 180.f)
                    result[i] -= 360.f;
                else if (result[i] < -180.f)
                    result[i] += 360.f;
            }
        }
    }
    return result;
}

template <typename Vector>
void PrepareEulerLookAt(const Vector &start, Vector &end, Vector &change, const Vector &point, const Vector &up,
                        const Vector &position, int mode, int constraint, bool relative)
{
    Point forward{point[0] - position[0], point[1] - position[1], point[2] - position[2]};
    if (constraint == 2)
        forward[0] = 0;
    else if (constraint == 4)
        forward[1] = 0;
    else if (constraint == 8)
        forward[2] = 0;
    const auto q = rotation::LookRotation(forward, {up[0], up[1], up[2]});
    const auto angles = rotation::ToPublicEulerYXZ(q[0], q[1], q[2], q[3]);
    for (size_t i = 0; i < angles.size(); ++i)
        end[i] = angles[i];
    const auto delta = EulerChange(start, end, mode, relative);
    for (size_t i = 0; i < delta.size(); ++i)
        change[i] = delta[i];
}

class EulerValuesSnapshot
{
  public:
    using Value = std::array<float, 4>;
    using Incremental = double;
    EulerValuesSnapshot(Point start, Point change, int mode) : start_(start), change_(change), mode_(mode)
    {
        if (mode_ >= 2)
            initial_ = rotation::FromEulerYXZ(start_[0], start_[1], start_[2]);
    }
    int Mode() const
    {
        return mode_;
    }
    Value Interpolate(double factor, double own = 0, double multiplier = 1, double parent = 0) const
    {
        if (mode_ >= 2) {
            Point step;
            for (size_t i = 0; i < step.size(); ++i)
                step[i] = change_[i] * static_cast<float>(factor);
            const auto offset = rotation::FromEulerYXZ(step[0], step[1], step[2]);
            return mode_ == 2 ? rotation::MultiplyPublic(offset, initial_) : rotation::MultiplyPublic(initial_, offset);
        }
        const auto angles = Vector3Snapshot(start_, change_).Interpolate(factor, own, multiplier, parent);
        return rotation::FromEulerYXZ(angles[0], angles[1], angles[2]);
    }

  private:
    Point start_, change_;
    Value initial_{};
    int mode_;
};

template <typename Vector> class LiveEulerValues
{
  public:
    using Value = EulerValuesSnapshot::Value;
    using Incremental = double;
    LiveEulerValues(const Vector &start, const Vector &change, int mode) : start_(&start), change_(&change), mode_(mode)
    {
    }
    int Mode() const
    {
        return mode_;
    }
    EulerValuesSnapshot Frozen() const
    {
        return {{(*start_)[0], (*start_)[1], (*start_)[2]}, {(*change_)[0], (*change_)[1], (*change_)[2]}, mode_};
    }
    Value Interpolate(double factor, double own = 0, double multiplier = 1, double parent = 0) const
    {
        return Frozen().Interpolate(factor, own, multiplier, parent);
    }

  private:
    const Vector *start_, *change_;
    int mode_;
};
} // namespace infernux::tween
