#pragma once

#include "core/FoundationApi.h"

#include <array>

namespace infernux::rotation
{
INFERNUX_FOUNDATION_API std::array<float, 4> FromEulerYXZ(float x, float y, float z);

// Public normalized-copy contract, separate from Transform's native rotation
// storage and the normalization used internally by Euler conversion.
INFERNUX_FOUNDATION_API std::array<float, 4> NormalizePublic(float x, float y, float z, float w);

// Managed Quaternion * Vector3: float products, wider expression evaluation.
// This does not normalize either input and is separate from native direction
// conversion, whose operation ordering has a different measured contract.
INFERNUX_FOUNDATION_API std::array<float, 3> RotatePublic(const std::array<float, 4> &rotation,
                                                          const std::array<float, 3> &point);

// Public products retain wider intermediate expressions, then store their
// components as float. Native Transform composition remains a separate path.
INFERNUX_FOUNDATION_API std::array<float, 4> MultiplyPublic(const std::array<float, 4> &left,
                                                            const std::array<float, 4> &right);

// Transform's native normalization has two cyclic pairwise reductions. It is
// observably different from the managed Quaternion.normalized property.
INFERNUX_FOUNDATION_API std::array<float, 4> NormalizeTransform(float x, float y, float z, float w);

// Native Transform composition uses a different addition order from the
// public managed quaternion operator. Inputs/outputs use x,y,z,w order.
INFERNUX_FOUNDATION_API std::array<float, 4> MultiplyTransform(const std::array<float, 4> &left,
                                                               const std::array<float, 4> &right);

// YXZ intrinsic / ZXY extrinsic rotation. The float stores and addition order
// reproduce the measured UnityEngine quaternion conversion, including its
// singular branch. Keep these stages separate from algebraic simplifications.
INFERNUX_FOUNDATION_API std::array<float, 3> ToEulerYXZ(float x, float y, float z, float w);

inline float PublicAngleDegrees(float angle)
{
    // Unity preserves tiny signed angles around zero instead of snapping them.
    constexpr float negativeFlip = -0.005729578f;
    constexpr float positiveFlip = 360.0f + negativeFlip;
    if (angle < negativeFlip)
        angle += 360.0f;
    else if (angle > positiveFlip)
        angle -= 360.0f;
    return angle;
}

inline std::array<float, 3> ToPublicEulerYXZ(float x, float y, float z, float w)
{
    auto result = ToEulerYXZ(x, y, z, w);
    for (float &angle : result)
        angle = PublicAngleDegrees(angle);
    return result;
}
} // namespace infernux::rotation
