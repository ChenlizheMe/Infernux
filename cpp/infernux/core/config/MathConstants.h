#pragma once

namespace infernux
{

/// General-purpose float epsilon for zero-checks, normalization guards, and approximate equality.
constexpr float kEpsilon = 1e-6f;

/// Maximum value of an 8-bit color channel (used in float ↔ byte conversions).
constexpr float kColorByteMax = 255.0f;

} // namespace infernux
