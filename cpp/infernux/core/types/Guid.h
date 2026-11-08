#pragma once

#include "core/FoundationApi.h"

#include <string>

namespace infernux
{
/// Allocate a persistent author identity, independently of runtime counters.
[[nodiscard]] INFERNUX_FOUNDATION_API std::string GenerateGuid();
} // namespace infernux
