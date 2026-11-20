// Private POD contract; not included by production. Units are SI.
#ifndef JPH_GPU_PRIVATE_ARTICULATION_TYPES_H
#define JPH_GPU_PRIVATE_ARTICULATION_TYPES_H
#include "PhysicsTypes.h"
#ifdef __cplusplus
namespace JPH
{
namespace GPU
{
#define ART_U UInt4
#define ART_F Float4
#define ART_ALIGN alignas(16)
#define ART_CONSTANT(name, value) inline constexpr std::uint32_t name = value
#else
#define ART_U uint4
#define ART_F float4
#define ART_ALIGN
#define ART_CONSTANT(name, value) static const uint name = value
#endif
ART_CONSTANT(ArtNoParent, 0xffffffffu);
ART_CONSTANT(ArtScratchWords, 32u);
ART_CONSTANT(ArtOutputWords, 4u);
ART_CONSTANT(ArtPacketMaxUnits, 8u);
ART_CONSTANT(ArtPending, 0u);
ART_CONSTANT(ArtSuccess, 1u);
ART_CONSTANT(ArtInvalid, 2u);
ART_CONSTANT(ArtNumericalFailure, 3u);
ART_CONSTANT(ArtInitialize, 0u);
ART_CONSTANT(ArtBackward, 1u);
ART_CONSTANT(ArtRoot, 2u);
ART_CONSTANT(ArtForward, 3u);
ART_CONSTANT(ArtDone, 4u);
struct ART_ALIGN ArticulationLink
{
    // body row, parent local index, accepted joint directory row, JointFixed/Hinge/Slider.
    ART_U topology;
    // stable joint ID low/high, accepted World row, zero; root ID is zero.
    ART_U identity;
};
struct ART_ALIGN ArticulationImpulse
{
    // World angular impulse at COM; w is incoming one-DoF generalized impulse (N*m*s or N*s).
    ART_F angularJoint;
    ART_F linear;
};
struct ART_ALIGN ArticulationTree
{
    // first link, count, scratch-first word, output-first record.
    ART_U range;
    ART_U rootMode; // x fixed-root flag, yzw zero.
    // status, phase, cursor local link, reason. Device-owned after begin.
    ART_U progress;
};
struct ART_ALIGN ArticulationOutput
{
    ART_F angularJoint; // delta omega xyz; incoming one-DoF delta qdot in w (rad/s or m/s).
    ART_F linear;
    ART_F supportAngular; // fixed root only, otherwise zero.
    ART_F supportLinear;
};
struct ART_ALIGN ArticulationWork
{
    ART_U range; // first tree, tree count, requested1..8 real work units, zero.
};
#ifdef __cplusplus
static_assert(sizeof(ArticulationLink) == 32 && sizeof(ArticulationImpulse) == 32);
static_assert(sizeof(ArticulationTree) == 48 && sizeof(ArticulationOutput) == 64 && sizeof(ArticulationWork) == 16);
static_assert(std::is_trivially_copyable_v<ArticulationLink> && std::is_trivially_copyable_v<ArticulationTree>);
}
}
#endif
#undef ART_U
#undef ART_F
#undef ART_ALIGN
#undef ART_CONSTANT
#endif
