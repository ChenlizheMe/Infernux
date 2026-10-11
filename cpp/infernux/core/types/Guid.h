#pragma once

#include <string>

namespace infernux
{
/// Allocate a persistent author identity, independently of runtime counters.
[[nodiscard]] std::string GenerateGuid();
} // namespace infernux
