// Internal lowering control. Byte offsets address ONE allocation; no pose mirror.
#ifndef JPH_GPU_ARTICULATION_CONTROL_TYPES_H
#define JPH_GPU_ARTICULATION_CONTROL_TYPES_H
#include "PhysicsTypes.h"
#ifdef __cplusplus
namespace JPH
{
namespace GPU
{
#define AC_U UInt4
#define AC_ALIGN alignas(16)
#define AC_CONSTANT(name, value) inline constexpr std::uint32_t name = value
#else
#define AC_U uint4
#define AC_ALIGN
#define AC_CONSTANT(name, value) static const uint name = value
#endif
AC_CONSTANT(ArtControlInvalid, 0xffffffffu);
AC_CONSTANT(ArtControlHeaderBytes, 48u);
AC_CONSTANT(ArtControlEligible, 1u);
AC_CONSTANT(ArtControlAdmitted, 2u);
AC_CONSTANT(ArtRefusedBody, 1u);
AC_CONSTANT(ArtRefusedSleeping, 2u);
AC_CONSTANT(ArtRefusedPrescribed, 4u);
AC_CONSTANT(ArtRefusedContact, 8u);
AC_CONSTANT(ArtRefusedExternalJoint, 16u);
AC_CONSTANT(ArtRefusedRuntimeJoint, 32u);
struct AC_ALIGN ArticulationControlHeader
{
    AC_U selected; // bodyBase/count, jointBase/count (native global ordinals).
    AC_U maps;     // treeCount, bodyMapByte, jointMapByte, flagsByte.
    AC_U storage;  // reasonsByte, preRotorByte, numericScratchByte, totalBytes.
};
struct AC_ALIGN ArticulationClassifyWork
{
    AC_U range;      // first/count, accepted World row, mode (initial refusal or reference kind).
    AC_U references; // jointRefFirst/count, patchRefFirst/count; bases are uint4 words.
};
// preRotor is one Float4 per selected joint: physical speed, cached armature,
// valid01, zero. Fixed/Slider records remain explicit zero; no zero-armature divide.
#ifdef __cplusplus
static_assert(sizeof(ArticulationControlHeader) == 48);
static_assert(sizeof(ArticulationClassifyWork) == 32);
static_assert(offsetof(ArticulationControlHeader, storage) == 32);
static_assert(std::is_trivially_copyable_v<ArticulationControlHeader>);
}
}
#endif
#undef AC_U
#undef AC_ALIGN
#undef AC_CONSTANT
#endif
