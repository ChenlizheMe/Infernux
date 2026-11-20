// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

// Host validates all word ranges, nonoverlap, domains and frozen bindings.
#include "../BroadphaseTypes.h"
#include "../SleepTypes.h"
#include "key_sort.slang"

ConstantBuffer<BroadphaseParams> broadphase;
ConstantBuffer<StageWork> stage_work;
StructuredBuffer<BodyModel> body_models;
StructuredBuffer<ShapeModel> shape_models;
StructuredBuffer<WorldRange> author_worlds;
StructuredBuffer<Aabb> bounds;
StructuredBuffer<QueryRecordWord> joint_stream;
StructuredBuffer<ColliderPairKey> collider_pair_ignores;
RWStructuredBuffer<ContactPatch> contact_patches;
// Separate from the writable arena: indirect consumers never bind this as
// writable storage. Each direct producer completely writes its used records.
RWStructuredBuffer<QueryRecordWord> indirect_arguments;
RWStructuredBuffer<QueryRecordWord> resident_step;
RWStructuredBuffer<Contact> resident_contacts;
RWStructuredBuffer<ContactPatch> resident_patches;
RWStructuredBuffer<Atomic<uint>> pair_count;
RWStructuredBuffer<Atomic<uint>> step_status;

static const uint BP_BAD_BOUNDS = 1u;
static const uint BP_SUM_OVERFLOW = 2u;
static const uint BP_BAD_MODEL = 4u;
static const uint BP_BAD_BANK = 16u;
static const uint BP_INDEX_OVERFLOW = 32u;
static const uint BP_CAPACITY_NEEDED = 128u;
static const uint BP_SWEEP_PENDING = 256u;
static const uint BP_LIMIT_FLAGS = BP_CAPACITY_NEEDED | BP_SWEEP_PENDING;

uint4 bp_load(uint p) { return contact_patches[p].leaves; }
void bp_store(uint p, uint4 v) { contact_patches[p].leaves = v; }
uint bp_header() { return broadphase.workspace.y; }

bool bp_suffix_valid()
{
    uint4 header = bp_load(bp_header());
    return header.z == 0u && header.y == 0u && header.x <= broadphase.candidates.z;
}

uint2 bp_add64(uint2 a, uint2 b, out bool overflow)
{
    uint lo = a.x + b.x;
    uint carry = lo < a.x ? 1u : 0u;
    uint hi0 = a.y + b.y;
    uint hi = hi0 + carry;
    overflow = hi0 < a.y || hi < hi0;
    return uint2(lo, hi);
}

bool bp_less64(uint2 a, uint2 b)
{
    return a.y < b.y || (a.y == b.y && a.x < b.x);
}

uint2 bp_subtract64(uint2 a, uint2 b)
{
    return uint2(a.x - b.x, a.y - b.y - (a.x < b.x ? 1u : 0u));
}

uint4 bp_combine(uint4 a, uint4 b)
{
    bool overflow;
    uint2 s = bp_add64(a.xy, b.xy, overflow);
    return uint4(s.x, s.y, a.z | b.z | (overflow ? BP_SUM_OVERFLOW : 0u), 0u);
}

bool bp_finite3(float3 v)
{
    return all((asuint(v) & uint3(0x7fffffffu)) < uint3(0x7f800000u));
}

bool bp_valid_bounds(Aabb a)
{
    return bp_finite3(a.minimum.xyz) && bp_finite3(a.maximum.xyz)
        && all(a.minimum.xyz <= a.maximum.xyz);
}

uint bp_ordered(float v)
{
    uint u = v == 0.0f ? 0u : asuint(v); // -0 and +0 share one key.
    return (u & 0x80000000u) != 0u ? ~u : (u ^ 0x80000000u);
}

bool bp_in_range(uint index, uint first, uint count)
{
    return index >= first && index - first < count;
}

bool bp_world_selected(uint world)
{
    return world == broadphase.selection.x;
}

bool bp_item(uint lane, uint length, out uint item)
{
    item = 0u;
    if (lane >= stage_work.range.y || stage_work.range.x >= length)
        return false;
    if (lane >= length - stage_work.range.x)
        return false;
    item = stage_work.range.x + lane;
    return true;
}

// Only for independent globally selected word domains. The host initializes
// every subsequently read header/scratch interval through exact finite chunks;
// an unwritten spare capacity/tail is never an initialized logical prefix.
[shader("compute")]
[numthreads(64, 1, 1)]
void init_words(uint3 tid : SV_DispatchThreadID)
{
    uint word;
    if (bp_item(tid.x, broadphase.workspace.x, word)) bp_store(word, uint4(0u));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void init_contacts(uint3 tid : SV_DispatchThreadID)
{
    uint item;
    uint points = broadphase.round.y != 0u ? 1u : MaxContactPointsPerPair;
    if (!bp_item(tid.x, max(1u, broadphase.candidates.z * points), item)) return;
    Contact empty;
    empty.meta = uint4(0u);
    empty.normalDepth = empty.localAnchorA = empty.localAnchorB = empty.impulse = float4(0.0f);
    resident_contacts[item] = empty;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void init_candidates(uint3 tid : SV_DispatchThreadID)
{
    uint word;
    if (bp_item(tid.x, 1u + 2u * broadphase.candidates.z, word))
        resident_patches[word].leaves = uint4(0u);
}

uint4 bp_merge_load(uint word)
{
    return broadphase.mergeKey.x == 2u ? resident_patches[word].leaves : bp_load(word);
}

void bp_merge_store(uint word, uint4 value)
{
    if (broadphase.mergeKey.x == 2u) resident_patches[word].leaves = value;
    else bp_store(word, value);
}

uint bp_blocks(uint count)
{
    return count / BroadphaseBlockSize + (count % BroadphaseBlockSize != 0u ? 1u : 0u);
}

uint4 bp_dispatch_dimensions(uint length, uint first, uint limit)
{
    uint count = first < length ? min(limit, length - first) : 0u;
    return uint4(bp_blocks(count), 1u, 1u, 0u);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void seed_sweep(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    uint status = bp_load(bp_header()).z & ~BP_LIMIT_FLAGS;
    bp_store(bp_header(), uint4(0u, 0u, status, 0u));
    pair_count[0].store(0u);
    pair_count[1].store(0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void prepare_indirect_arguments(uint3 tid : SV_DispatchThreadID)
{
    uint record = tid.x;
    if (record >= stage_work.range.y) return;
    uint length = bp_suffix_valid() ? bp_load(bp_header()).x : 0u;
    uint first = stage_work.range.x + record * BroadphaseItemLimit;
    indirect_arguments[record].value = bp_dispatch_dimensions(length, first, BroadphaseItemLimit);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void prepare_sweep_arguments(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint4 projection = bp_load(broadphase.summary.x);
    uint status = projection.z | (bp_load(bp_header()).z & ~BP_LIMIT_FLAGS) | pair_count[1].load();
    bool overflow;
    uint2 first = bp_add64(stage_work.range.xz, uint2(tid.x * BroadphaseSweepJobs, 0u), overflow);
    uint limit = tid.x + 1u == stage_work.range.y ? stage_work.range.w : BroadphaseSweepJobs;
    uint jobs = 0u;
    if (!overflow && status == 0u && bp_less64(first, projection.xy))
    {
        uint2 remaining = bp_subtract64(projection.xy, first);
        jobs = remaining.y == 0u ? min(remaining.x, limit) : limit;
    }
    indirect_arguments[tid.x].value = uint4(bp_blocks(jobs), 1u, 1u, 0u);
}

uint4 bp_sort_record(uint leaf, Aabb box)
{
    ShapeModel shape = shape_models[leaf];
    uint status = 0u;
    if (shape.surface.x >= 32u) status |= BP_BAD_MODEL;
    uint world = 0xffffffffu;
    if (shape.meta.y >= broadphase.counts.w) status |= BP_BAD_MODEL;
    else
    {
        BodyModel body = body_models[shape.meta.y];
        world = body.meta.w;
        if (world >= broadphase.counts.z || !bp_world_selected(world)) status |= BP_BAD_MODEL;
        else
        {
            WorldRange owner = author_worlds[world];
            if (!bp_in_range(leaf, owner.leaves.x, owner.leaves.y)
                || !bp_in_range(shape.meta.y, owner.bodies.x, owner.bodies.y)
                || !bp_in_range(leaf, body.meta.x, body.meta.y)) status |= BP_BAD_MODEL;
        }
    }
    if (!bp_valid_bounds(box)) status |= BP_BAD_BOUNDS;
    // Invalid records still get an initialized, unique key and an explicit
    // failure. They never silently become a successful zero-candidate result.
    uint min_x = (status & BP_BAD_BOUNDS) == 0u ? bp_ordered(box.minimum.x) : 0u;
    return uint4(world, min_x, leaf, status);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void build_sort_records(uint3 tid : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    uint item;
    bool valid = bp_item(tid.x, broadphase.selectedCounts.y, item);
    uint4 record = uint4(0xffffffffu);
    if (valid)
    {
        uint leaf = broadphase.selection.z + item;
        record = bp_sort_record(leaf, bounds[leaf]);
    }
    record = jph_sort_tile(lane, record);
    if (valid) bp_store(broadphase.merge.x + item, record);
}

uint4 bp_suppression_record(uint joint, out uint status)
{
    uint p = 1u + JointDirectoryWords * joint;
    uint4 endpoints = joint_stream[p].value;
    uint world = joint_stream[p + 1u].value.z;
    status = 0u;
    bool suppress = (endpoints.w & (JointEnabled | JointCollisionEnabled)) == JointEnabled;
    // World endpoints cannot match any two real bodies, exactly as before.
    suppress = suppress && endpoints.x != InvalidQueryBodySlot;
    if (world >= broadphase.counts.z || !bp_world_selected(world) || endpoints.y >= broadphase.counts.w
        || (endpoints.x != InvalidQueryBodySlot && endpoints.x >= broadphase.counts.w))
        status |= BP_BAD_MODEL;
    else
    {
        WorldRange owner = author_worlds[world];
        if (!bp_in_range(joint, owner.leaves.z, owner.leaves.w)
            || body_models[endpoints.y].meta.w != world
            || (endpoints.x != InvalidQueryBodySlot && body_models[endpoints.x].meta.w != world))
            status |= BP_BAD_MODEL;
    }
    if (suppress && status == 0u)
        return uint4(world, min(endpoints.x, endpoints.y), max(endpoints.x, endpoints.y), joint);
    return uint4(0xffffffffu, 0xffffffffu, 0xffffffffu, joint);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void build_suppression(uint3 tid : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    uint item;
    bool valid = bp_item(tid.x, broadphase.selectedCounts.z, item);
    uint4 key = uint4(0xffffffffu);
    if (valid)
    {
        uint status;
        key = bp_suppression_record(broadphase.selection.w + item, status);
        bp_store(broadphase.scan.x + item, uint4(0u, 0u, status, 0u));
    }
    key = jph_sort_tile(lane, key);
    if (valid) bp_store(broadphase.merge.x + item, key);
}

bool bp_less(uint4 a, uint4 b)
{
    return broadphase.mergeKey.x == 2u ? jph_key_less(a.wxyz, b.wxyz) : jph_key_less(a, b);
}

// Filled pairs are sorted by (world, first leaf, second leaf). The input and
// output of a block are the same disjoint range of the accepted candidate bank.
[shader("compute")]
[numthreads(64, 1, 1)]
void sort_candidate_tiles(uint3 tid : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    uint item;
    uint length = bp_suffix_valid() ? bp_load(bp_header()).x : 0u;
    bool valid = bp_item(tid.x, length, item);
    uint4 key = uint4(0xffffffffu);
    if (valid) key = resident_patches[broadphase.merge.x + item].leaves.wxyz;
    key = jph_sort_tile(lane, key);
    if (valid) resident_patches[broadphase.merge.x + item].leaves = key.yzwx;
}

// One stable merge destination per source item. Left items use lower_bound,
// right items upper_bound, so even equal keys cannot collide in the scatter.
[shader("compute")]
[numthreads(64, 1, 1)]
void merge_items(uint3 tid : SV_DispatchThreadID)
{
    uint length = broadphase.mergeKey.x == 2u ? (bp_suffix_valid() ? bp_load(bp_header()).x : 0u) : broadphase.merge.z;
    uint item;
    if (!bp_item(tid.x, length, item)) return;
    uint width = broadphase.merge.w; // Host validates power-of-two and <= 2^31.
    uint run = item / width;
    uint own_begin = run * width;
    uint own_end = own_begin + min(width, length - own_begin);
    bool right = (run & 1u) != 0u;
    uint other_begin = right ? own_begin - width : own_end;
    uint other_end = right ? own_begin : other_begin + min(width, length - other_begin);
    uint merged_begin = right ? other_begin : own_begin;
    uint4 key = bp_merge_load(broadphase.merge.x + item);
    uint low = other_begin, high = other_end;
    for (uint step = 0u; step < 32u && low < high; ++step)
    {
        uint middle = low + (high - low) / 2u;
        uint4 other = bp_merge_load(broadphase.merge.x + middle);
        bool before = right ? !bp_less(key, other) : bp_less(other, key);
        if (before) low = middle + 1u; else high = middle;
    }
    uint destination = merged_begin + (item - own_begin) + (low - other_begin);
    bp_merge_store(broadphase.merge.y + destination, key);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void find_sap_ends(uint3 tid : SV_DispatchThreadID)
{
    uint rank;
    if (!bp_item(tid.x, broadphase.selectedCounts.y, rank)) return;
    uint4 key = bp_load(broadphase.workspace.z + rank);
    uint status = key.w;
    uint end = rank + 1u;
    if (key.x >= broadphase.counts.z || key.z >= broadphase.counts.x) status |= BP_BAD_MODEL;
    if (status == 0u)
    {
        const float max_x = bounds[key.z].maximum.x;
        uint low = rank + 1u, high = broadphase.selectedCounts.y;
        for (uint step = 0u; step < 32u && low < high; ++step)
        {
            uint middle = low + (high - low) / 2u;
            uint4 other = bp_load(broadphase.workspace.z + middle);
            if (other.x == key.x && bounds[other.z].minimum.x <= max_x) low = middle + 1u;
            else high = middle;
        }
        end = low;
    }
    if (broadphase.round.x != 0u)
    {
        bool active = false;
        if (status == 0u)
        {
            ShapeModel shape = shape_models[key.z];
            BodyModel body = body_models[shape.meta.y];
            bool resident = (body.meta.z & MotionNotResident) == 0u;
            bool sensor = (shape.meta.w & ShapeSensor) != 0u;
            if (broadphase.round.y != 0u)
            {
                // A swept pair has an initiating leaf or an endpoint driven
                // by an impact earlier in this interval. The other leaf can
                // be moving, sleeping or static, and stays in the SAP bank.
                Aabb box = bounds[key.z];
                active = resident && !sensor && (box.maximum.w != 0.0f || box.minimum.w > 0.0f ||
                    (box.minimum.w != 0.0f && body.invInertiaDiagonal.w < 0.0f));
            }
            else
                active = resident && (body.meta.z == MotionDynamic || sensor);
        }
        bp_store(broadphase.summary.y + rank, uint4(end, uint(active), status, 0u));
        bp_store(broadphase.scan.x + rank, uint4(uint(active), 0u, status, 0u));
    }
    else
        bp_store(broadphase.scan.x + rank, uint4(end - (rank + 1u), 0u, status, 0u));
}

// A dynamic or sensor leaf owns the entire right interval. Other leaves
// project only against active positions in that interval. The geometrical
// filters and canonical candidate order stay shared by both domains.
[shader("compute")]
[numthreads(64, 1, 1)]
void project_active_pairs(uint3 tid : SV_DispatchThreadID)
{
    uint rank;
    if (!bp_item(tid.x, broadphase.selectedCounts.y, rank)) return;
    uint4 row = bp_load(broadphase.summary.y + rank);
    const uint active_before = bp_load(broadphase.scan.y + rank).x;
    uint start = rank + 1u;
    uint count = row.x - start;
    if (row.y != 0u)
        bp_store(broadphase.summary.z + active_before, uint4(rank, 0u, 0u, 0u));
    else
    {
        const uint active_end = row.x < broadphase.selectedCounts.y
            ? bp_load(broadphase.scan.y + row.x).x
            : bp_load(broadphase.summary.x).x;
        start = active_before;
        count = active_end - active_before;
    }
    bp_store(broadphase.summary.y + rank, uint4(start, row.y, 0u, 0u));
    bp_store(broadphase.scan.x + rank, uint4(count, 0u, row.z, 0u));
}

bool bp_suppressed(uint world, uint body_a, uint body_b)
{
    uint a = min(body_a, body_b), b = max(body_a, body_b);
    uint low = 0u, high = broadphase.selectedCounts.z;
    for (uint step = 0u; step < 32u && low < high; ++step)
    {
        uint middle = low + (high - low) / 2u;
        uint4 key = bp_load(broadphase.workspace.w + middle);
        bool before = key.x < world || (key.x == world &&
            (key.y < a || (key.y == a && key.z < b)));
        if (before) low = middle + 1u; else high = middle;
    }
    if (low == broadphase.selectedCounts.z) return false;
    uint4 found = bp_load(broadphase.workspace.w + low);
    // Duplicates retain sourceJointOrdinal in the full sort key. Any matching
    // suppressing joint wins, preserving the original multi-joint OR rule.
    return found.x == world && found.y == a && found.z == b;
}

bool bp_collider_suppressed(uint4 a, uint4 b)
{
    ColliderPairKey key = jph_collider_pair_key(a, b);
    uint low = 0u, high = broadphase.selectedCounts.w;
    for (uint step = 0u; step < 32u && low < high; ++step) {
        uint middle = low + (high - low) / 2u;
        if (jph_collider_pair_less(collider_pair_ignores[middle], key)) low = middle + 1u;
        else high = middle;
    }
    if (low >= broadphase.selectedCounts.w) return false;
    return jph_collider_pair_equal(collider_pair_ignores[low], key);
}

groupshared Aabb bp_small_bounds[BroadphaseSmallSetLimit];
groupshared uint bp_small_leaf_status[BroadphaseSmallSetLimit];
groupshared uint4 bp_small_joint_keys[BroadphaseSmallJointLimit];
groupshared uint4 bp_small_pair_masks[BroadphaseSmallSetLimit];
groupshared uint bp_small_pair_offsets[BroadphaseSmallSetLimit];

bool bp_small_suppressed(uint world, uint a, uint b)
{
    uint2 pair = uint2(min(a, b), max(a, b));
    for (uint i = 0u; i < broadphase.selectedCounts.z; ++i)
    {
        uint4 key = bp_small_joint_keys[i];
        if (key.x == world && all(key.yz == pair)) return true;
    }
    return false;
}

// AABBs already include the existing half speculative-band expansion.
bool bp_eligible_leaves(uint first, uint second, uint world, Aabb ba, Aabb bb, bool small)
{
    if (any(ba.minimum.xyz > bb.maximum.xyz) || any(bb.minimum.xyz > ba.maximum.xyz)) return false;
    ShapeModel a = shape_models[first], b = shape_models[second];
    if (a.meta.y == b.meta.y) return false;
    BodyModel ma = body_models[a.meta.y], mb = body_models[b.meta.y];
    if ((ma.meta.z & MotionNotResident) != 0u || (mb.meta.z & MotionNotResident) != 0u) return false;
    bool sensor = ((a.meta.w | b.meta.w) & ShapeSensor) != 0u;
    if (broadphase.round.y != 0u)
    {
        if (sensor) return false;
        bool twoDynamic = ba.minimum.w != 0.0f && bb.minimum.w != 0.0f;
        bool continuedImpact = ba.maximum.w != 0.0f || bb.maximum.w != 0.0f;
        if (twoDynamic)
        {
            if (ma.invInertiaDiagonal.w >= 0.0f && mb.invInertiaDiagonal.w >= 0.0f && !continuedImpact) return false;
        }
        else if (ba.minimum.w <= 0.0f && bb.minimum.w <= 0.0f && !continuedImpact) return false;
    }
    if (!sensor && (ma.meta.z & ~MotionNotResident) != MotionDynamic
        && (mb.meta.z & ~MotionNotResident) != MotionDynamic) return false;
    if ((a.surface.y & (1u << b.surface.x)) == 0u
        || (b.surface.y & (1u << a.surface.x)) == 0u) return false;
    if (small)
    {
        if (bp_small_suppressed(world, a.meta.y, b.meta.y)) return false;
    }
    else if (bp_suppressed(world, a.meta.y, b.meta.y)) return false;
    return !bp_collider_suppressed(a.source, b.source);
}

// Cache the leaf bounds for the group and compact by source leaf ordinal.
// The resulting (leafA, leafB) sequence matches the global sort.
[shader("compute")]
[numthreads(64, 1, 1)]
void build_small_pairs(uint3 local : SV_GroupThreadID)
{
    uint lane = local.x;
    uint first = broadphase.selection.z;
    uint leaves = broadphase.selectedCounts.y;
    uint words = (leaves + 31u) >> 5u;
    uint joints = broadphase.selectedCounts.z;
    uint world = broadphase.selection.x;
    for (uint row = lane; row < leaves; row += BroadphaseBlockSize)
    {
        Aabb box = bounds[first + row];
        bp_small_bounds[row] = box;
        bp_small_leaf_status[row] = bp_sort_record(first + row, box).w;
    }
    if (lane < joints)
    {
        uint status;
        uint4 key = bp_suppression_record(broadphase.selection.w + lane, status);
        key.w = status;
        bp_small_joint_keys[lane] = key;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint row = lane; row < leaves; row += BroadphaseBlockSize)
    {
        uint4 mask = uint4(0u);
        if (bp_small_leaf_status[row] == 0u)
        {
            Aabb box = bp_small_bounds[row];
            for (uint other = row + 1u; other < leaves; ++other)
            {
                if (bp_small_leaf_status[other] != 0u) continue;
                if (bp_eligible_leaves(first + row, first + other, world, box, bp_small_bounds[other], true))
                    mask[other >> 5u] |= 1u << (other & 31u);
            }
        }
        bp_small_pair_masks[row] = mask;
        uint4 bits = countbits(mask);
        bp_small_pair_offsets[row] = bits.x + bits.y + bits.z + bits.w;
    }
    GroupMemoryBarrierWithGroupSync();

    if (lane == 0u)
    {
        uint count = 0u, status = 0u;
        for (uint i = 0u; i < leaves; ++i)
        {
            uint row_count = bp_small_pair_offsets[i];
            bp_small_pair_offsets[i] = count;
            count += row_count;
            status |= bp_small_leaf_status[i];
        }
        for (uint i = 0u; i < joints; ++i)
            status |= bp_small_joint_keys[i].w;
        bp_store(bp_header(), uint4(count, 0u, status, count));
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint row = lane; row < leaves; row += BroadphaseBlockSize)
    {
        uint offset = bp_small_pair_offsets[row];
        uint4 mask = bp_small_pair_masks[row];
        for (uint word = row >> 5u; word < words; ++word)
        {
            uint bits = mask[word];
            while (bits != 0u && offset < broadphase.candidates.z)
            {
                uint other = word * 32u + uint(firstbitlow(bits));
                resident_patches[broadphase.candidates.x + offset].leaves = uint4(first + row, first + other, 0u, world);
                ++offset;
                bits &= bits - 1u;
            }
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void collect_sap_pairs(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    bool overflow;
    uint2 job = bp_add64(stage_work.range.xz, uint2(tid.x, 0u), overflow);
    if (overflow || !bp_less64(job, bp_load(broadphase.summary.x).xy)) return;

    // Find the last exclusive prefix <= job, including runs of empty spans.
    // The next prefix (or the total for the final rank) is strictly larger.
    uint rank = 0u, high = broadphase.selectedCounts.y;
    while (high - rank > 1u)
    {
        uint middle = rank + (high - rank) / 2u;
        if (bp_less64(job, bp_load(broadphase.scan.y + middle).xy)) high = middle;
        else rank = middle;
    }
    // The rank's projected span is at most the number of leaves, so only
    // the low word of the difference is needed across a 32-bit carry.
    const uint local_ordinal = job.x - bp_load(broadphase.scan.y + rank).x;
    uint other = rank + 1u + local_ordinal;
    if (broadphase.round.x != 0u)
    {
        const uint2 row = bp_load(broadphase.summary.y + rank).xy;
        other = row.x + local_ordinal;
        if (row.y == 0u)
            other = bp_load(broadphase.summary.z + other).x;
    }
    uint4 key_a = bp_load(broadphase.workspace.z + rank);
    uint4 key_b = bp_load(broadphase.workspace.z + other);
    if (!bp_eligible_leaves(key_a.z, key_b.z, key_a.x, bounds[key_a.z], bounds[key_b.z], false)) return;
    // Atomic append balances the filtered output. Canonical sorting orders
    // the accepted candidate stream before any dependent physics stage.
    uint pair = pair_count[0].add(1u);
    if (pair >= 0xffffffffu / MaxContactPointsPerPair) pair_count[1].or(BP_INDEX_OVERFLOW);
    if (pair < broadphase.candidates.z)
        resident_patches[broadphase.candidates.x + pair].leaves =
            uint4(min(key_a.z, key_b.z), max(key_a.z, key_b.z), 0u, key_a.x);
}

groupshared uint4 bp_shared[BroadphaseBlockSize];

// For these two entries only, StageWork item first/count means BLOCKS.
// Every group participates in all barriers, including the masked final tail.
[shader("compute")]
[numthreads(64, 1, 1)]
void scan_blocks(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint length = broadphase.scan.w;
    uint block_count = bp_blocks(length);
    uint block;
    // Mask tail lanes without changing workgroup barrier participation.
    bool active_block = bp_item(gid.x, block_count, block);
    uint first = block * BroadphaseBlockSize;
    bool valid = active_block && lane < length - first;
    uint4 input = uint4(0u);
    if (valid) input = bp_load(broadphase.scan.x + first + lane);
    bp_shared[lane] = input;
    GroupMemoryBarrierWithGroupSync();
    for (uint distance = 1u; distance < BroadphaseBlockSize; distance *= 2u)
    {
        uint4 value = bp_shared[lane];
        uint4 previous = uint4(0u);
        if (lane >= distance) previous = bp_shared[lane - distance];
        GroupMemoryBarrierWithGroupSync();
        bp_shared[lane] = bp_combine(previous, value);
        GroupMemoryBarrierWithGroupSync();
    }
    if (valid)
    {
        uint4 exclusive = uint4(0u);
        if (lane != 0u) exclusive = bp_shared[lane - 1u];
        bp_store(broadphase.scan.y + first + lane, exclusive);
    }
    if (active_block && lane == BroadphaseBlockSize - 1u)
        bp_store(broadphase.scan.z + block, bp_shared[BroadphaseBlockSize - 1u]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void reduce_blocks(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint length = broadphase.scan.w;
    uint block_count = bp_blocks(length);
    uint block;
    bool active_block = bp_item(gid.x, block_count, block);
    uint first = block * BroadphaseBlockSize;
    bool valid = active_block && lane < length - first;
    uint4 input = uint4(0u);
    if (valid) input = bp_load(broadphase.scan.x + first + lane);
    bp_shared[lane] = input;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = BroadphaseBlockSize / 2u; stride != 0u; stride /= 2u)
    {
        if (lane < stride) bp_shared[lane] = bp_combine(bp_shared[lane], bp_shared[lane + stride]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (active_block && lane == 0u) bp_store(broadphase.scan.z + block, bp_shared[0u]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void add_scan_offsets(uint3 tid : SV_DispatchThreadID)
{
    uint item;
    uint length = broadphase.scan.w;
    if (!bp_item(tid.x, length, item)) return;
    uint4 parent = bp_load(broadphase.scanParent.x + item / BroadphaseBlockSize);
    bp_store(broadphase.scan.y + item, bp_combine(parent, bp_load(broadphase.scan.y + item)));
}

// Dedicated fixed-one-record publication entries. Dispatch exactly one group;
// only lane zero accesses the records. summary.x points to a fully reduced
// record, or to a zero initialized record for an empty domain.
[shader("compute")]
[numthreads(1, 1, 1)]
void accumulate_status_summary(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    uint4 header = bp_load(bp_header());
    header.z |= bp_load(broadphase.summary.x).z;
    bp_store(bp_header(), header);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void publish_count_summary(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    uint count = pair_count[0].load();
    uint4 projection = bp_load(broadphase.summary.x);
    uint status = (bp_load(bp_header()).z & ~BP_LIMIT_FLAGS) | projection.z | pair_count[1].load();
    if (bp_less64(stage_work.range.xz, projection.xy)) status |= BP_SWEEP_PENDING;
    bp_store(bp_header(), uint4(count, 0u, status, count));
    bp_store(bp_header() + 1u, projection);
}

groupshared uint bp_published_count;

// The canonical bank, logical length and narrowphase dispatch arguments all
// come from the same counted candidate stream.
[shader("compute")]
[numthreads(64, 1, 1)]
void publish_candidates(uint lane : SV_GroupIndex)
{
    if (lane == 0u)
    {
        uint4 header = bp_load(bp_header());
        if (header.y != 0u) header.z |= BP_BAD_BANK;
        if (header.x > broadphase.candidates.z) header.z |= BP_CAPACITY_NEEDED;
        bp_store(bp_header(), header);
        bool swept = broadphase.round.y != 0u;
        uint faults = swept ? header.z & ~BP_LIMIT_FLAGS : header.z;
        if (broadphase.round.z != 0u && faults != 0u)
            step_status[StepBroadphaseStatus].or(faults);
        bp_published_count = header.z == 0u ? header.x : 0u;
        resident_patches[0u].leaves = uint4(broadphase.merge.x, bp_published_count, swept ? header.z : 0u, 0u);
        uint4 solve = resident_step[1u].value;
        uint points = swept ? 1u : MaxContactPointsPerPair;
        solve.zw = uint2(bp_published_count * points, bp_published_count);
        resident_step[1u].value = solve;
    }
    GroupMemoryBarrierWithGroupSync();

    const uint chunk = stage_work.range.x;
    const uint stride = stage_work.range.z;
    for (uint tile = lane; tile < stage_work.range.y; tile += BroadphaseBlockSize)
    {
        const uint first = tile * chunk;
        const uint record = tile * stride;
        const uint4 pairs = bp_dispatch_dimensions(bp_published_count, first, chunk);
        indirect_arguments[record].value = pairs;
        if (stride != 1u)
        {
            indirect_arguments[record + 1u].value =
                bp_dispatch_dimensions(bp_published_count * MaxContactPointsPerPair,
                                       first * MaxContactPointsPerPair, chunk * MaxContactPointsPerPair);
            // History always replaces its header in the first tile.
            indirect_arguments[record + 2u].value = uint4(max(uint(tile == 0u), pairs.x), 1u, 1u, 0u);
        }
    }
}
