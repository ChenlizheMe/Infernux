// SPDX-License-Identifier: MIT
#ifndef JPH_GPU_CONTINUOUS_TYPES_H
#define JPH_GPU_CONTINUOUS_TYPES_H
#include "PhysicsTypes.h"

#ifdef __cplusplus
namespace JPH
{
namespace GPU
{
#define JPH_CCD_ALIGN alignas(16)
#define JPH_CCD_UINT uint32_t
#define JPH_CCD_FLOAT4 Float4
#else
#define JPH_CCD_ALIGN
#define JPH_CCD_UINT uint
#define JPH_CCD_FLOAT4 float4
#endif

struct JPH_CCD_ALIGN ContinuousCpuBody
{
    BodyModel model;
    BodyState state;
    JPH_CCD_FLOAT4 displacement;
};
struct JPH_CCD_ALIGN ContinuousSample
{
    BodyState state;
    float dx, dy, dz;
    JPH_CCD_UINT islandAndImpact; // World-local event island in the high bits, earlier impact in bit zero.
};
struct JPH_CCD_ALIGN ContinuousMixedHit
{
    JPH_CCD_UINT cpuRow, gpuSlot;
    float fraction, minRestitutionSpeed;
    JPH_CCD_FLOAT4 normalFriction;
    JPH_CCD_FLOAT4 pointRestitution;
};
struct JPH_CCD_ALIGN ContinuousReceipt
{
    JPH_CCD_FLOAT4 linear; // Impulse on CPU xyz.
    JPH_CCD_FLOAT4 angular;
    JPH_CCD_FLOAT4 relative; // Incoming GPU point velocity minus CPU point velocity.
};
#ifdef __cplusplus
static_assert(sizeof(ContinuousCpuBody) == 160);
static_assert(sizeof(ContinuousSample) == 80);
static_assert(sizeof(ContinuousMixedHit) == 48);
static_assert(sizeof(ContinuousReceipt) == 48);
}
}
#endif
#undef JPH_CCD_ALIGN
#undef JPH_CCD_UINT
#undef JPH_CCD_FLOAT4
#endif
