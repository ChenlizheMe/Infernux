// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef __cplusplus
#include <cstdint>
namespace infernux::physics::inxphygpu
{
inline constexpr uint32_t SceneCcdMaxImpacts = 32;
inline constexpr uint32_t SceneCcdReportFlag = 4;
inline constexpr uint32_t ContactReportUnknownSeparation = 1;
}
#else
static const uint SceneCcdMaxImpacts = 32;
static const uint SceneCcdReportFlag = 4;
static const uint ContactReportUnknownSeparation = 1;

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
