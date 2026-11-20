#ifndef JPH_GPU_PRIVATE_STATE_FK_TYPES_H
#define JPH_GPU_PRIVATE_STATE_FK_TYPES_H
#include "ArticulationTypes.h"
#ifdef __cplusplus
namespace JPH
{
namespace GPU
{
#define SF_U UInt4
#define SF_F Float4
#define SF_ALIGN alignas(16)
#define SF_CONST(name, value) inline constexpr std::uint32_t name = value
#else
#define SF_U uint4
#define SF_F float4
#define SF_ALIGN
#define SF_CONST(name, value) static const uint name = value
#endif
SF_CONST(StateFkPending, 0u);
SF_CONST(StateFkSuccess, 1u);
SF_CONST(StateFkInvalid, 2u);
SF_CONST(StateFkUnrepresentable, 3u);
SF_CONST(StateFkNumericalFailure, 4u);
SF_CONST(StateFkCapturePhase, 0u);
SF_CONST(StateFkForwardPhase, 1u);
SF_CONST(StateFkDone, 2u);
struct SF_ALIGN StateFkTree
{
    SF_U range;    // first link, link count, first coordinate, first output.
    SF_U owner;    // accepted World row, root body row, zero, zero.
    SF_U progress; // status, phase, next local link, reason; GPU-owned after begin.
};
struct SF_ALIGN StateFkWork
{
    SF_U range;     // first tree, tree count, requested real units1..8, zero.
    SF_F tolerance; // absolute position(m), angle(rad), linear(m/s), angular(rad/s).
};
#ifdef __cplusplus
static_assert(sizeof(StateFkTree) == 48 && sizeof(StateFkWork) == 32);
static_assert(std::is_trivially_copyable_v<StateFkTree>);
}
}
#endif
#undef SF_U
#undef SF_F
#undef SF_ALIGN
#undef SF_CONST
#endif
