#pragma once
#include "PhysicsTypes.h"
#ifdef __cplusplus
namespace JPH
{
namespace GPU
{
#define CR_ALIGN alignas(16)
#define CR_U4 UInt4
#define CR_F4 Float4
#else
#define CR_ALIGN
#define CR_U4 uint4
#define CR_F4 float4
#endif
struct CR_ALIGN ContactReportParams
{
    CR_U4 range;     // Window first/count, logical patch count, substep.
    CR_U4 selection; // World row, global body base/count, global leaf base.
    CR_U4 requests;  // Leaf count, collision/trigger/impulse stream bits, swept impacts, zero.
};
struct CR_ALIGN ContactReportPoint
{
    CR_U4 feature; // Primitive feature, present, solver active, reserved.
    CR_F4 pointDepth;
    CR_F4 normal;
    CR_F4 relativeVelocity;
    CR_F4 impulse;
};
struct CR_ALIGN ContactReportPatch
{
    CR_U4 source; // Global leaf A/B, global body A/B in retained publication.
    CR_U4 meta;   // Substep, patch flags, actual manifold point count, source ordinal.
    ContactReportPoint points[4];
};
#ifdef __cplusplus
static_assert(sizeof(ContactReportParams) == 48);
static_assert(sizeof(ContactReportPoint) == 80);
static_assert(sizeof(ContactReportPatch) == 352);
static_assert(std::is_trivially_copyable_v<ContactReportPatch>);
}
}
#endif
#undef CR_ALIGN
#undef CR_U4
#undef CR_F4
