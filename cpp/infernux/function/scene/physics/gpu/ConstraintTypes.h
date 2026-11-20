// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT
#ifndef JPH_GPU_CONSTRAINT_TYPES_H
#define JPH_GPU_CONSTRAINT_TYPES_H

#include "PhysicsTypes.h"

#ifdef __cplusplus
#include <algorithm>
namespace JPH
{
namespace GPU
{
#define CS_ALIGN alignas(16)
#define CS_U4 UInt4
#define CS_UINT std::uint32_t
#define CS_MIN std::min
#define CS_INLINE inline constexpr
#define CS_CONST(name, value) inline constexpr std::uint32_t name = value
#else
#define CS_ALIGN
#define CS_U4 uint4
#define CS_UINT uint
#define CS_MIN min
#define CS_INLINE
#define CS_CONST(name, value) static const uint name = value
#endif

CS_CONST(ConstraintBlockSize, 64u);
CS_CONST(ConstraintItemLimit, 2048u);
CS_CONST(ConstraintBlockLimit, 32u);
CS_CONST(ConstraintLocalBodies, 8u * ConstraintBlockSize);
// The compact directory scans at most one workgroup-width of slots per lane.
CS_CONST(ConstraintLocalBodyScanLimit, ConstraintBlockSize *ConstraintBlockSize);
CS_CONST(ConstraintLocalSortItems, 16u * ConstraintBlockSize);
CS_CONST(ConstraintLocalRefs, 3u * ConstraintLocalSortItems);
CS_CONST(ConstraintLocalJointDispatch, ConstraintLocalRefs / ConstraintBlockSize);
CS_CONST(ConstraintLocalPatchDispatch, ConstraintLocalJointDispatch + 1u);
CS_CONST(ConstraintLocalWakePatchDispatch, ConstraintLocalPatchDispatch + 1u);
CS_CONST(ConstraintLocalArgumentWords, ConstraintLocalWakePatchDispatch + 1u);
CS_CONST(ConstraintScheduleHeaderWords, 4u);
CS_CONST(IslandJointQuantum, 64u);
CS_CONST(IslandPatchQuantum, 256u);
CS_CONST(IslandPatchRangeRows, 4u * IslandPatchQuantum);
CS_CONST(IslandRangeGroups, 64u);
// Count visits across rounds separately from the shared tile size. A counted
// dispatch processes independent islands with at most this many visits per group.
CS_CONST(IslandPatchRoundVisits, 32u * IslandPatchQuantum);
CS_CONST(ContactSolverLanes, 64u);
CS_CONST(ContactParallelMinimum, ContactSolverLanes / 2u);
CS_CONST(ConstraintInvalid, 0xffffffffu);
CS_CONST(ConstraintWholeIslandTable, 0xffffffffu);
CS_CONST(ConstraintJoint, 0u);
CS_CONST(ConstraintPatch, 1u);
CS_CONST(ConstraintBadModel, 1u);
CS_CONST(ConstraintBadJoint, 2u);
CS_CONST(ConstraintBadManifold, 4u);
CS_CONST(ConstraintBadLabel, 8u);
CS_CONST(ConstraintNotConverged, 16u);
CS_CONST(ConstraintOverflow, 32u);
CS_CONST(ConstraintBadSchedule, 64u);
CS_CONST(ConstraintNotReady, 0x80000000u);

// An island in the tail keeps the numerical tile size used by its canonical
// group, even when the execution directory groups islands by their row counts.
// Only cooperative contact solves use this width (max rows >= 32).
CS_INLINE CS_UINT ConstraintContactTileWidth(CS_UINT island, CS_UINT islandCount)
{
    const CS_UINT groupFirst = (island / ConstraintBlockSize) * ConstraintBlockSize;
    return IslandPatchQuantum / CS_MIN(ConstraintBlockSize, islandCount - groupFirst);
}

struct CS_ALIGN ConstraintScheduleWord
{
    CS_U4 value;
};

// All addresses are UInt4 word offsets in ConstraintSchedule, except labels
// (independent scalar uint32 allocations, stride 4) and source ordinals.
// All ranges / products / dispatch dimensions must be checked in host uint64.
// Checked constraints: 2*(J+P), 4*P, 12*J, B*2, totalWords all fit uint32;
// every nonempty read/write range fits storage/device limits and is disjoint
// from concurrent writers. B and J span the selected world. P bounds the
// candidate prefix, whose logical count comes from the collision header.
// Tombstones are nonresident models.
struct CS_ALIGN ConstraintParams
{
    CS_U4 counts; // B local slots, J local joints, P candidate slots, packed JointStream directory size.
    CS_U4 limits; // total schedule words, local shape count, contact rows=4P, body-root word base.
    CS_U4 layout; // header base=0, connectivity control, body-check base, ref-check base.
    // Joint and patch reference bases (2 words per item), then the final joint and patch sort banks.
    CS_U4 references;
    CS_U4 sources;    // joint source word base, patch source word base, joint ID-table base, patch ID-table base.
    CS_U4 islands;    // 2B-word range base, B root-row base, B exclusive-prefix base, B range-summary-row base.
    CS_U4 scan;       // input base, exclusive-prefix base, block-total base, input length.
    CS_U4 scanParent; // scanned parent-prefix base, combine mode (0 or 1), 0, 0.
    CS_U4 merge;      // source bank, target bank, length (or Invalid for live patches), run width power-of-two<=2^31.
    CS_U4 select;     // kind (Joint=0/Patch=1), sort seed bank, awake refs, wide reduction workspace.
    CS_U4 summary;    // Wide: root scan, range reduction, reference and body checks.
                      // Compact local: writer directory base, writer count, 0, 0. Direct local: zero.
    CS_U4 selection;  // accepted packed world row and body, leaf, joint bases.
};

// Frozen numerical consumer header (four words, base=0):
// h0=(islandCount,jointIslandCount,contactIslandCount,status)
// h1=(activeJ,activeP,maxJointItems,maxPatchItems)
// h2=(rangeWordBase,jointTableWordBase,contactTableWordBase,jointSourceWordBase)
// h3=(patchSourceWordBase,wordCount,bodyRootWordBase,liveCandidateCount)
// status==0 is the ONLY installable state. Initialization sets NotReady.
// range[i].word0=(world,root,jointBegin,jointCount)
// range[i].word1=(patchBegin,patchCount,0,0)
// begin/count index the respective source-word stream (not absolute words).
// Tables contain (islandID,canonicalTableIndex,0,0). Joint sources contain
// (sourceOrdinal,0,0,0). The solver may order disjoint islands by row count.
// Patch sources contain (sourceOrdinal,writerA,writerB,0). Readonly endpoints
// use ConstraintInvalid so static and kinematic state can be read in parallel.
// Joint ordinals address the unchanged accepted JointStream. Patch ordinals
// address the unchanged candidate bank and four original Contact records.
// Within each range all source ordinals strictly increase, independently for
// Joint and patch phases. No numeric state is copied, reordered or solved here.
// Island IDs follow increasing root BODY SLOT (world order need not match).
//
// Scratch ref: word0=(world,writerA,writerB,sourceOrdinal),
// word1=(kind,active01,status,0). Readonly writer endpoints use Invalid.
// Sort key=(world,root,sourceOrdinal,kind). Inactive keys are all Invalid;
// stable merge places all active keys before initialized inactive tails.
// Root flag scalar (reuses the sleep vote bank after wake):
// bit0=has Joint, bit1=has patch. Only active refs mark roots.
// Labels and body-root rows are local to the selected world.
// Parents and sleep votes each use B scalars.
// References and island ranges retain packed body slots.
// Scan mode0 record=(sum0,sum1,sum2,ORstatus), checked u32 sums.
// Scan mode1 record=(sumJ,sumP,maxJ,maxP); sums bounded by checked J/P.
// Ref validation total (mode0)=(activeJ,activeP,0,status).
// Connectivity control=(0,0,status,validated01). Edges join by descending to
// lower labels. Root compression then produces a stable minimum-slot root
// for every writer. Validation writes the fixed step's topology status and
// gates subsequent wake dispatches with argument records indexed by their
// workgroup count. The local path also emits the counted wake patch command.
// Both atomic banks have scalar uint32 layout, with dispatch dependencies
// between reset, linking, flattening, and publication.
// Reflected storage access remains read_write even in load-only label entries.
// Do not bind an atomic label array as WGSL storage read-only or use a UInt4
// field as its backing storage. CPU POD remains scalar uint32, not a C++ atomic.

#ifdef __cplusplus
inline constexpr bool ConstraintLocalTopologyFits(std::uint32_t bodies, std::uint32_t joints, std::uint32_t patches)
{
    return bodies <= ConstraintLocalBodies && joints <= ConstraintLocalRefs && patches <= ConstraintLocalRefs - joints;
}

inline constexpr bool ConstraintDeviceScheduleFits(std::uint32_t bodies, std::uint32_t joints, std::uint32_t patches)
{
    return joints <= IslandJointQuantum && ConstraintLocalTopologyFits(bodies, joints, patches);
}

static_assert(sizeof(ConstraintScheduleWord) == 16);
static_assert(alignof(ConstraintScheduleWord) == 16);
static_assert(std::is_trivially_copyable_v<ConstraintScheduleWord>);
static_assert(sizeof(ConstraintParams) == 12 * 16);
static_assert(alignof(ConstraintParams) == 16);
}
}
#endif
#undef CS_ALIGN
#undef CS_U4
#undef CS_UINT
#undef CS_MIN
#undef CS_INLINE
#undef CS_CONST
#endif
