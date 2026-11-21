// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#ifndef INFERNUX_GPU_RENDER_POSE_LAYOUT
#define INFERNUX_GPU_RENDER_POSE_LAYOUT

#ifdef __cplusplus
#include <cstdint>
namespace infernux::physics::gpu_render
{
struct Float4 { float x, y, z, w; };
struct UInt4 { uint32_t x, y, z, w; };
inline constexpr uint32_t PoseNoParent = UINT32_MAX;
#define INX_POSE_ALIGN alignas(16)
#define INX_POSE_FLOAT4 Float4
#define INX_POSE_UINT4 UInt4
#else
static const uint PoseNoParent = 0xffffffffu;
#define INX_POSE_ALIGN
#define INX_POSE_FLOAT4 float4
#define INX_POSE_UINT4 uint4
#endif

// Immutable render hierarchy nodes address one frozen BodyState generation.
// Parent chains are acyclic. The host schedules topological levels separately
// from physical body capacity; a node consumes its completed parent's affine.
// Ordinary parent chains retain both their affine and their authored TRS inverse:
// a collapsed scale uses zero reciprocal, rather than a generic matrix inverse.
struct INX_POSE_ALIGN PoseNode
{
    // Body-state slot, parent PoseNode index or PoseNoParent, then zero padding.
    INX_POSE_UINT4 bodyParent;
    // Accepted body-origin-to-COM vector in the body frame; w is zero padding.
    INX_POSE_FLOAT4 rootCOM;
    // This Transform's authored signed local scale; w is zero padding.
    INX_POSE_FLOAT4 localScale;
    // Quaternion xyzw of the ordinary chain from the parent body root to this
    // Transform's direct parent. With PoseNoParent, the direct parent's world
    // quaternion instead. Independent of the affine matrix; do not decompose it.
    INX_POSE_FLOAT4 parentRotation;
    // Affine columns mapping this body's direct parent into its ancestor body
    // root, or into world space for PoseNoParent. w = 0, 0, 0, 1 respectively.
    INX_POSE_FLOAT4 parentLocalToAncestor0;
    INX_POSE_FLOAT4 parentLocalToAncestor1;
    INX_POSE_FLOAT4 parentLocalToAncestor2;
    INX_POSE_FLOAT4 parentLocalToAncestor3;
    // Reverse map composed from each ordinary Transform's authored inverse TRS,
    // in reverse order. abs(localScale) < 1e-9 has zero reciprocal per axis.
    // This is not the generic inverse or pseudoinverse of the forward matrix.
    // Columns have w = 0, 0, 0, 1 respectively.
    INX_POSE_FLOAT4 parentAncestorToLocal0;
    INX_POSE_FLOAT4 parentAncestorToLocal1;
    INX_POSE_FLOAT4 parentAncestorToLocal2;
    INX_POSE_FLOAT4 parentAncestorToLocal3;
};

// One renderer's ordinary local chain from its nearest exported body root.
// Mesh draw and existing local bounds deliberately use separate pivot paths.
struct INX_POSE_ALIGN PoseBinding
{
    // PoseNode index in x; yzw are zero padding.
    INX_POSE_UINT4 nodeIndex;
    // Affine object-local-to-body-root columns; w = 0, 0, 0, 1 respectively.
    // The chain contains no other exported body node.
    INX_POSE_FLOAT4 objectLocal0;
    INX_POSE_FLOAT4 objectLocal1;
    INX_POSE_FLOAT4 objectLocal2;
    INX_POSE_FLOAT4 objectLocal3;
    // Selected-submesh draw pivot, or zero for an unshifted draw. w is zero.
    INX_POSE_FLOAT4 drawPivot;
    // Existing object-local AABB center and nonnegative half extent. These
    // bounds already contain any selected-submesh pivot. Both w are zero.
    INX_POSE_FLOAT4 boundsCenter;
    INX_POSE_FLOAT4 boundsExtent;
};

struct INX_POSE_ALIGN PoseParams
{
    // Total node count, renderer binding count, current level first/count.
    INX_POSE_UINT4 counts;
    // First float4 of the world's body rows, float4 stride, origin/WXYZ flag, padding.
    INX_POSE_UINT4 stateLayout;
};

#undef INX_POSE_ALIGN
#undef INX_POSE_FLOAT4
#undef INX_POSE_UINT4

#ifdef __cplusplus
static_assert(sizeof(PoseNode) == 192);
static_assert(sizeof(PoseBinding) == 128);
static_assert(sizeof(PoseParams) == 32);
}
#endif

#endif
