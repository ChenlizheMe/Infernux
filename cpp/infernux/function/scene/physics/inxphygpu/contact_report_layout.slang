// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef __cplusplus
#include <cstdint>
namespace infernux::physics::inxphygpu
{
inline constexpr uint32_t ContactReportUnknownSeparation = 1;
inline constexpr uint32_t SceneCcdImpactCapacityExceeded = 32;
inline constexpr uint32_t SceneCcdUnresolvedSweep = 64;
}
#else
static const uint ContactReportUnknownSeparation = 1;
static const uint SceneCcdImpactCapacityExceeded = 32;
static const uint SceneCcdUnresolvedSweep = 64;

struct ContactReport
{
    float4 pointSeparation;
    float4 normal;
    float4 relativeVelocity;
    float4 impulse;
    uint4 endpoints; // Geometry A/B and their respective subshapes.
    uint4 identity; // Feature, trigger, solved impulse, flags.
};
#endif
