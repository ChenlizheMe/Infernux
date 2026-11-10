#pragma once

#include "core/FoundationApi.h"

#include <array>

namespace infernux::rotation
{
// Public quaternion components in x/y/z/w order. The construction branch and
// degenerate-direction convention are observable independently of orientation.
INFERNUX_FOUNDATION_API std::array<float, 4> LookRotation(const std::array<float, 3> &forward,
                                                          const std::array<float, 3> &up);
} // namespace infernux::rotation
