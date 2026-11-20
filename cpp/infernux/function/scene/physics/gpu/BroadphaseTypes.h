// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

// Shared C++ / Slang POD; the host owns scheduling and allocation.
#ifndef JPH_GPU_BROADPHASE_TYPES_H
#define JPH_GPU_BROADPHASE_TYPES_H
#include "PhysicsTypes.h"
#ifdef __cplusplus
namespace JPH
{
namespace GPU
{
#define BP_ALIGN alignas(16)
#define BP_U4 UInt4
#else
#define BP_ALIGN
#define BP_U4 uint4
#endif

#ifdef __cplusplus
inline constexpr std::uint32_t BroadphaseBlockSize = 64u;
inline constexpr std::uint32_t BroadphaseItemLimit = 2048u;
inline constexpr std::uint32_t BroadphaseBlockLimit = BroadphaseItemLimit / BroadphaseBlockSize;
inline constexpr std::uint32_t BroadphaseSweepJobs = BroadphaseItemLimit * 128u;
inline constexpr std::uint32_t BroadphaseSmallSetLimit = 2u * BroadphaseBlockSize;
inline constexpr std::uint32_t BroadphaseSmallJointLimit = 16u;
#else
static const uint BroadphaseBlockSize = 64u;
static const uint BroadphaseItemLimit = 2048u;
static const uint BroadphaseBlockLimit = BroadphaseItemLimit / BroadphaseBlockSize;
static const uint BroadphaseSweepJobs = BroadphaseItemLimit * 128u;
static const uint BroadphaseSmallSetLimit = 2u * BroadphaseBlockSize;
static const uint BroadphaseSmallJointLimit = 16u;
#endif

// All addresses below are ContactPatch / uint4 WORD addresses, not bytes,
// Contact ordinals, candidate ordinals, leaf ordinals or body slots.
struct BP_ALIGN BroadphaseParams
{
    BP_U4 counts;    // accepted leaf N, joint J, world W, body B.
    BP_U4 workspace; // arena word capacity, header address, final SAP bank, final suppression bank.
    BP_U4 round;     // Project active pairs, swept pairs, device-owned count, single-group pairs.
    // candidates.xy address the separate resident_patches allocation;
    // all other arena addresses refer to the scratch contact_patches port.
    BP_U4 candidates; // candidate bank0 base, bank1 base, capacity K, zero; actual C is scratch header.x.
    BP_U4 merge;      // source base, target base, valid length, run width [1,2^31].
    BP_U4 mergeKey;   // 0=SAP(world,minX,leaf), 1=suppression(world,bodyA,bodyB,joint), 2=pair(world,leafA,leafB); yzw
                      // zero.
    BP_U4 scan;       // input base, exclusive-prefix base, block-total base, valid input length.
    BP_U4 scanParent; // globally scanned parent prefix base for add_scan_offsets; yzw zero.
    BP_U4 summary;    // Reduction record, projection rows, active rank directory, zero.
    // One accepted layout address, never a stable RuntimeWorld identity.
    // selection=(world row, body base, leaf base, joint base).
    // selectedCounts=(body capacity, leaf count, joint count, Collider-pair ignore count).
    // Counts above remain TOTAL array extents for global address validation.
    BP_U4 selection;
    BP_U4 selectedCounts;
};

// resident_patches[0] = (candidateBankWordBase, actualC, sweptStatus, 0).
// A swept consumer resumes its producer when sweptStatus is nonzero.
// Scratch header = (totalLo, totalHi, statusBits, selected-world pair count).
// Bit 7 is a distinct capacity-needed result: no suffix work consumes C>K.
// Bit 8 indicates more projected pairs than the admitted sweep windows. The
// following scratch word records the exact 64-bit projection count.
// The exclusive projection scan maps every pair ordinal to a sorted leaf.
// StageWork for prepare_indirect_arguments = (first candidate, record count, 0, 0).
// StageWork for prepare_sweep_arguments = (first job lo, record count, first job hi, last-window limit).
// StageWork for collect_sap_pairs = (first job lo, maximum jobs, first job hi, 0).
// Indirect records live in a separate allocation: (groupsX, 1, 1, 0), with
// exactly the first 12 bytes consumed as indirect arguments.
// StageWork for publish_candidates = (pair chunk width, tiles, records per tile, 0).
// A counted tile publishes consecutive pair, setup and history records.
// All scan/reduce records = (sumLo,sumHi,statusBits,0). The associative
// operator performs checked uint2 sum and bitwise OR. Sum overflow sets bit 1.
// Pair words = (canonicalLeafA,canonicalLeafB,0,world) before narrowphase.
// Narrowphase overwrites patch flags and clears w. Keeping world in the sort
// key preserves runtime world segments even if author leaf ranges are not
// ordered by world index. Within each world the key is exactly (leafA,leafB).

#ifdef __cplusplus
static_assert(sizeof(BroadphaseParams) == 11 * 16);
static_assert(alignof(BroadphaseParams) == 16);
static_assert(sizeof(WorldRange) == 32);
}
}
#endif
#undef BP_ALIGN
#undef BP_U4
#endif
