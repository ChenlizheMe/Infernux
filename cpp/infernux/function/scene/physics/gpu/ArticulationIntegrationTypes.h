#ifndef JPH_GPU_ARTICULATION_INTEGRATION_TYPES_H
#define JPH_GPU_ARTICULATION_INTEGRATION_TYPES_H
#include "ArticulationControlTypes.h"
#ifdef __cplusplus
namespace JPH
{
namespace GPU
{
#define AI_CONSTANT(name, value) inline constexpr std::uint32_t name = value
#else
#define AI_CONSTANT(name, value) static const uint name = value
#endif
AI_CONSTANT(ArtControlDirty, 4u);
AI_CONSTANT(ArtControlPreviousReduced, 8u);
AI_CONSTANT(ArtControlCaptured, 16u);
AI_CONSTANT(ArtControlCandidate, 32u);
AI_CONSTANT(ArtRefusedVelocityDisabled, 64u);
AI_CONSTANT(ArtNormalSeed, 3u);
AI_CONSTANT(ArtNormalPromote, 4u);
AI_CONSTANT(ArtNormalReaction, 12u);
AI_CONSTANT(ArtNormalHistory, 13u);
#ifdef __cplusplus
}
}
#endif
#undef AI_CONSTANT
#endif
