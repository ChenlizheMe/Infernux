#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include "WeightedBezier.h"

namespace infernux::curve
{
struct HermitePolynomial
{
    float cubic, quadratic, linear, constant;

    template <typename Key> static HermitePolynomial Compile(const Key &left, const Key &right)
    {
        const float duration = std::max(right.time - left.time, .0001f);
        const float difference = right.value - left.value;
        const float inverse = 1.f / duration;
        const float inverse2 = inverse * inverse;
        // Slopes and the Horner products have separate float32 stores. Keep
        // those rounding boundaries on processors with fused multiply-add.
        const volatile float first = left.outTangent * duration;
        const volatile float second = right.inTangent * duration;
        const float cubic = (((first + second) - difference) - difference) * inverse2 * inverse;
        const float quadratic = ((((difference + difference) + difference) - first) - first - second) * inverse2;
        return {cubic, quadratic, left.outTangent, left.value};
    }

    float Evaluate(float offset) const
    {
        const volatile float first = cubic * offset;
        const volatile float second = (first + quadratic) * offset;
        const volatile float third = (second + linear) * offset;
        return third + constant;
    }
};

// Key supplies time/value, in/outTangent, in/outWeight and weightedMode bits.
// CPU bindings, renderer consumers and GPU curve data share this key contract.
template <typename Key> float EvaluateSegment(const Key &left, const Key &right, float time)
{
    const float duration = right.time - left.time;
    if (duration <= 0.0)
        return right.value;
    if (time >= right.time)
        return right.value;
    if (std::isinf(left.outTangent) || std::isinf(right.inTangent)) {
        const float infinity = std::numeric_limits<float>::infinity();
        return left.outTangent == infinity || right.inTangent == infinity ? left.value : right.value;
    }
    if (!(left.weightedMode & 2) && !(right.weightedMode & 1)) {
        return HermitePolynomial::Compile(left, right).Evaluate(std::max(time - left.time, 0.f));
    }
    const float t = (time - left.time) / duration;
    const float w0 = left.weightedMode & 2 ? left.outWeight : 1.f / 3.f;
    const float w1 = right.weightedMode & 1 ? right.inWeight : 1.f / 3.f;
    if (!std::isfinite(w0) || !std::isfinite(w1))
        return std::numeric_limits<float>::quiet_NaN();
    const float parameter = BezierTimeParameter(t, w0, 1.f - w1);
    return WeightedBezierValue(parameter, left.value, duration * left.outTangent, w0, right.value,
                               duration * right.inTangent, w1);
}
} // namespace infernux::curve
