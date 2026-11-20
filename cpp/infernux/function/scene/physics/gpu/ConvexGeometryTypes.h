// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT
#ifndef JPH_GPU_CONVEX_GEOMETRY_TYPES_H
#define JPH_GPU_CONVEX_GEOMETRY_TYPES_H
#include "PhysicsTypes.h"
#ifdef __cplusplus
namespace JPH
{
namespace GPU
{
#define JPH_CONVEX_CONSTANT(name, value) inline constexpr std::uint32_t name = value
#else
#define JPH_CONVEX_CONSTANT(name, value) static const uint name = value
#endif

// All offsets/lengths address UInt4 (16-byte) words, never scalar uints/bytes.
// Shape/explicit operand geometryReference=(absolute first word,length,0,0).
// A live record is immutable for the lifetime of every captured range lease.
JPH_CONVEX_CONSTANT(ConvexGeometryHeaderWords, 5);

// Header[0]=(ShapeConvexHull,totalWords,vertexCount,faceCount).
// Header[1]=(fullVertices,coreVertices,planes,faceDirectory), relative words.
// Header[2]=(loopIndicesWord,loopIndexCount,meshRayVertices,0); indices are packed uint4.
// Mesh triangles append three original mesh-local Float4(xyz,0) vertices at
// meshRayVertices. Other hulls have a zero offset and no ray vertex region.
// Header[3/4]=Float4(COM-local full bounds min/max.xyz,0).
// Full/core points are Float4(xyz,0). Planes use native n.dot(x)+constant=0.
// Face directory=(first loop INDEX,index count,plane INDEX,0).
// Zero-margin hull aliases its core vertex words to its full vertex words.

JPH_CONVEX_CONSTANT(ConvexJobSlotWords, 2048);
JPH_CONVEX_CONSTANT(ConvexJobGroupSize, 8);
JPH_CONVEX_CONSTANT(ConvexJobAdvanceQuantum, 32);
JPH_CONVEX_CONSTANT(ConvexJobHeaderWords, 64);
JPH_CONVEX_CONSTANT(ConvexJobSimplexFirst, 64);
JPH_CONVEX_CONSTANT(ConvexJobPointsFirst, 76);
JPH_CONVEX_CONSTANT(ConvexJobFacesFirst, 460);
JPH_CONVEX_CONSTANT(ConvexJobHorizonFirst, 1740);
JPH_CONVEX_CONSTANT(ConvexJobTraversalFirst, 1868);
JPH_CONVEX_CONSTANT(ConvexJobNewFacesFirst, 1996);
JPH_CONVEX_CONSTANT(ConvexJobMasksFirst, 2028);
JPH_CONVEX_CONSTANT(ConvexJobReservedFirst, 2044);

#ifdef __cplusplus
static_assert(ConvexJobSimplexFirst + 4 * 3 == ConvexJobPointsFirst);
static_assert(ConvexJobPointsFirst + 128 * 3 == ConvexJobFacesFirst);
static_assert(ConvexJobFacesFirst + 256 * 5 == ConvexJobHorizonFirst);
static_assert(ConvexJobHorizonFirst + 128 == ConvexJobTraversalFirst);
static_assert(ConvexJobTraversalFirst + 128 == ConvexJobNewFacesFirst);
static_assert(ConvexJobNewFacesFirst + 32 == ConvexJobMasksFirst);
static_assert(ConvexJobMasksFirst + 16 == ConvexJobReservedFirst);
static_assert(ConvexJobReservedFirst + 4 == ConvexJobSlotWords);
}
}
#endif
#undef JPH_CONVEX_CONSTANT
#endif
