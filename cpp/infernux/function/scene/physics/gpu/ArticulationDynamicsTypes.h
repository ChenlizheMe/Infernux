#ifndef JPH_GPU_PRIVATE_ARTICULATION_DYNAMICS_TYPES_H
#define JPH_GPU_PRIVATE_ARTICULATION_DYNAMICS_TYPES_H
#include "ArticulationTypes.h"
#ifdef __cplusplus
namespace JPH
{
namespace GPU
{
#define AD_U UInt4
#define AD_F Float4
#define AD_ALIGN alignas(16)
#define AD_CONST(name, value) inline constexpr std::uint32_t name = value
#else
#define AD_U uint4
#define AD_F float4
#define AD_ALIGN
#define AD_CONST(name, value) static const uint name = value
#endif
// ArticulationTree48 is reused. Dynamics range=(linkFirst,count,scratchFirst,
// coordinateFirst); rootMode=(fixedStaticRoot,rootRecord,World,rootBodyRow).
// Existing impulse-response interpretation/entry points are unchanged.
AD_CONST(ArtDynamicsUnrepresentable, 4u);
AD_CONST(ArtDynPreserve, 0u);
AD_CONST(ArtDynAwaitForces, 1u);
AD_CONST(ArtDynInitialize, 2u);
AD_CONST(ArtDynBackward, 3u);
AD_CONST(ArtDynRoot, 4u);
AD_CONST(ArtDynForward, 5u);
AD_CONST(ArtDynCandidate, 6u);
AD_CONST(ArtDynReady, 7u);
AD_CONST(ArtDynCommitVelocity, 8u);
AD_CONST(ArtDynAwaitIntegration, 9u);
AD_CONST(ArtDynCommitPose, 10u);
AD_CONST(ArtDynDone, 11u);
struct AD_ALIGN ArticulationDynamicsWork
{
    AD_U range;       // first tree, tree count, requested1..8 real units, zero.
    AD_F integration; // dt, position tolerance, rotation tolerance, twist tolerance.
};
#ifdef __cplusplus
static_assert(sizeof(ArticulationDynamicsWork) == 32);
static_assert(std::is_trivially_copyable_v<ArticulationDynamicsWork>);
}
}
#endif
#undef AD_U
#undef AD_F
#undef AD_ALIGN
#undef AD_CONST
#endif
