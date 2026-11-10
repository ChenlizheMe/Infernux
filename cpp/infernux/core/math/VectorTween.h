#pragma once

#include "TweenNumerics.h"

namespace infernux::tween
{
// Single-frame sampling observes the owner's vectors directly. Bulk callers
// freeze once before releasing the interpreter lock, retaining a stable batch.
template <typename Vector, size_t Channels> class LiveVectorValues
{
  public:
    using FrozenValues = VectorSnapshot<Channels>;
    using Value = typename FrozenValues::Value;
    LiveVectorValues(const Vector &start, const Vector &change, bool snapping)
        : start_(&start), change_(&change), snapping_(snapping)
    {
    }
    FrozenValues Frozen() const
    {
        Value start, change;
        for (size_t axis = 0; axis < Channels; ++axis) {
            start[axis] = (*start_)[axis];
            change[axis] = (*change_)[axis];
        }
        return FrozenValues(start, change, snapping_);
    }
    Value Interpolate(double factor, double own = 0, double multiplier = 1, double parent = 0) const
    {
        return Frozen().Interpolate(factor, own, multiplier, parent);
    }

  private:
    const Vector *start_, *change_;
    bool snapping_;
};
} // namespace infernux::tween
