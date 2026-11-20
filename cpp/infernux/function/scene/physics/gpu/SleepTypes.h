// Pointer-free C++ / Slang sleep history; body pose/mobility ABI remains unchanged.
#ifndef JPH_GPU_SLEEP_TYPES_H
#define JPH_GPU_SLEEP_TYPES_H
#include "PhysicsTypes.h"
#ifdef __cplusplus
namespace JPH
{
namespace GPU
{
#define S_ALIGN alignas(16)
#define S_F4 Float4
#define S_U4 UInt4
#else
#define S_ALIGN
#define S_F4 float4
#define S_U4 uint4
#endif

// CPU author updates touch only these48bytes, from the accepted native root
// shape bounds. They never upload a copy of the device-owned runtime suffix.
struct S_ALIGN BodySleepAuthor
{
    S_F4 rootHalfExtent;   // Final root shape COM-local GetLocalBounds().GetExtent(), w0.
    S_F4 rootBoundsCenter; // Actual bounds center in the same COM-local frame, w0.
    S_U4 policy;           // x bit0 allow automatic sleep, bit1 body sensor; yzw0.
};
// The native float-profile three Sphere records, timer, and explicit activity.
struct S_ALIGN BodySleepRuntime
{
    S_F4 sphere0;
    S_F4 sphere1;
    S_F4 sphere2;
    S_U4 timing; // Quiet seconds (FP32 bits), contact anchor discontinuity, 0, 0.
    S_U4 state;  // active01, initialized01, canSleep01, prescribedStaticActive01.
};
// One additional SSBO, without widening BodyState/BodyModel or borrowing padding.
struct S_ALIGN BodySleepRecord
{
    BodySleepAuthor author;
    BodySleepRuntime runtime;
};
struct S_ALIGN BodySleepParameters
{
    S_F4 timing; // elapsed simulated interval, configured quiet seconds,
                 // configured point velocity threshold, reserved0.
};
// One completion record for the fixed step. Its device words are atomic
// because separate dispatches can contribute failures before publication.
#ifdef __cplusplus
struct PhysicsStepStatus
{
    std::uint32_t sleep;
    std::uint32_t topology;
    std::uint32_t broadphase;
};
static_assert(sizeof(PhysicsStepStatus) == 12);
#else
static const uint StepSleepStatus = 0u;
static const uint StepTopologyStatus = 1u;
static const uint StepBroadphaseStatus = 2u;
#endif

// Metadata for the existing source's bounded sleep control dispatches. All
// addresses are body slots or uint4 words, never byte offsets.
struct S_ALIGN SleepWorkParameters
{
    S_F4 timing;    // substep simulated dt, configured quiet seconds/point velocity,0.
    S_U4 selected;  // world row, body base, body count, total body slots.
    S_U4 layout;    // schedule local body-root word base, zero,
                    // source first, source count (joint/patch/query domain).
                    // For batched bounds wakes, zw locate/count sorted source–target exclusions.
    S_U4 operation; // source body slot or UINT_MAX, old/new AABB bits, finalFixed01, queryCount.
};
#ifdef __cplusplus
static_assert(sizeof(BodySleepAuthor) == 48);
static_assert(sizeof(BodySleepRuntime) == 80);
static_assert(sizeof(BodySleepRecord) == 128);
static_assert(offsetof(BodySleepRecord, runtime) == 48);
static_assert(offsetof(BodySleepRuntime, timing) == 48);
static_assert(offsetof(BodySleepRuntime, state) == 64);
static_assert(sizeof(BodySleepParameters) == 16);
static_assert(sizeof(SleepWorkParameters) == 64);
static_assert(std::is_trivially_copyable_v<BodySleepRecord>);
}
}
#endif
#undef S_ALIGN
#undef S_F4
#undef S_U4
#endif
