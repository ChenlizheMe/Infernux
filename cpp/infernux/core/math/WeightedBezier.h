#pragma once

#include <cmath>

namespace infernux::curve
{
inline float RoundedProduct(float left, float right)
{
    const volatile float product = left * right;
    return product;
}

inline float SignedCubeRoot(float value)
{
    // Preserve the float logarithm before the wider exponent. Replacing this
    // expression with pow/cbrt changes the inverse near cancelling roots.
    const float logarithm = std::log(std::abs(value));
    const float magnitude = static_cast<float>(std::exp(static_cast<double>(logarithm) / 3.0));
    return value < 0.f ? -magnitude : magnitude;
}

inline bool IsBezierParameter(float value)
{
    return value >= 0.f && value <= 1.f;
}

inline float BezierTimeParameter(float time, float first, float second)
{
    const float linear = RoundedProduct(first, 3.f);
    const float scaledSecond = RoundedProduct(second, 3.f);
    const float cubic = (linear - scaledSecond) + 1.f;
    const float quadratic = scaledSecond - RoundedProduct(first, 6.f);
    constexpr float epsilon = .001f;
    float parameter;
    if (std::abs(cubic) > epsilon) {
        const float denominator = cubic * 3.f;
        const float shift = -quadratic / denominator;
        const float shift2 = RoundedProduct(shift, shift);
        const float q = linear / denominator - shift2;
        const float numerator = RoundedProduct(linear, quadratic) - RoundedProduct(denominator, -time);
        const float r = numerator / ((cubic * 6.f) * cubic) + RoundedProduct(shift2, shift);
        const float q3 = RoundedProduct(q * q, q);
        const float r2 = RoundedProduct(r, r);
        const float discriminant = q3 + r2;
        if (discriminant < 0.f) {
            const float radius = SignedCubeRoot(std::sqrt(r2 - discriminant)) * 2.f;
            const float angle = std::atan2(std::sqrt(-discriminant), r) / 3.f;
            const auto root = [radius, shift](float phase) { return RoundedProduct(radius, std::cos(phase)) + shift; };
            parameter = root(angle);
            if (IsBezierParameter(parameter))
                return parameter;
            constexpr float angleStep = 2.094395160675048828125f;
            parameter = root(angle + angleStep);
            if (IsBezierParameter(parameter))
                return parameter;
            parameter = root(angle - angleStep);
        } else {
            const float radical = std::sqrt(discriminant);
            const float firstRoot = SignedCubeRoot(radical + r);
            const float secondRoot = SignedCubeRoot(r - radical);
            parameter = (secondRoot + firstRoot) + shift;
        }
    } else if (std::abs(quadratic) > epsilon) {
        const float radical = std::sqrt(RoundedProduct(linear, linear) - RoundedProduct(quadratic * 4.f, -time));
        const float denominator = quadratic + quadratic;
        parameter = (-linear - radical) / denominator;
        if (IsBezierParameter(parameter))
            return parameter;
        parameter = (radical - linear) / denominator;
    } else if (std::abs(linear) > epsilon) {
        return time / linear;
    } else {
        return 0.f;
    }
    // A rounded analytic root can leave the segment. Endpoint selection is
    // part of interpolation, including nonmonotonic horizontal handles.
    return IsBezierParameter(parameter) ? parameter : (time < .5f ? 0.f : 1.f);
}

inline float WeightedBezierValue(float parameter, float first, float firstSlope, float firstWeight, float last,
                                 float lastSlope, float lastWeight)
{
    const float remaining = 1.f - parameter;
    const float square = parameter * parameter;
    const float remainingSquare = remaining * remaining;
    const float firstBasis = (parameter * 3.f) * remainingSquare;
    const float lastBasis = (square * 3.f) * remaining;
    const float firstControl = RoundedProduct(firstSlope, firstWeight) + first;
    const float lastControl = last - RoundedProduct(lastSlope, lastWeight);
    const float firstTerm = RoundedProduct(firstControl, firstBasis);
    const float startTerm = RoundedProduct(remainingSquare * remaining, first);
    const float lastTerm = RoundedProduct(lastControl, lastBasis);
    const float endTerm = RoundedProduct(square * parameter, last);
    return ((firstTerm + startTerm) + lastTerm) + endTerm;
}
} // namespace infernux::curve
