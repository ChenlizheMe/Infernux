// Jolt Physics Library; SPDX-License-Identifier: MIT
#include "rigid_math.slang"
#include "../ContactReportTypes.h"
ConstantBuffer<ContactReportParams> contact_report_params;
StructuredBuffer<ContactPatch> contact_patches;
StructuredBuffer<Contact> contacts;
StructuredBuffer<BodyModel> body_models;
StructuredBuffer<BodyState> body_states;
StructuredBuffer<KinematicTarget> kinematic_targets;
StructuredBuffer<uint> contact_report_interest;
// count, status, then source-patch -> compact row. Entire map initialized once.
RWStructuredBuffer<Atomic<uint>> contact_report_indices;
RWStructuredBuffer<QueryRecordWord> contact_report_words;

bool cr_range(uint x, uint first, uint count) { return x >= first && x - first < count; }
void cr_word(uint row, uint word, uint4 v)
{
    QueryRecordWord q; q.value = v; contact_report_words[row * 22u + word] = q;
}
uint4 cr_read(uint row, uint word) { return contact_report_words[row * 22u + word].value; }
void cr_bad() { contact_report_indices[1].or(1u); }

[shader("compute")]
[numthreads(64, 1, 1)]
void contact_report_initialize(uint3 id : SV_DispatchThreadID)
{
    ContactReportParams p = contact_report_params;
    if (id.x >= p.range.y) return;
    uint ordinal = p.range.x + id.x;
    if (ordinal >= p.range.z) return;
    contact_report_indices[2u + ordinal].store(0xffffffffu);
    for (uint word = 0u; word < 22u; ++word) cr_word(ordinal, word, uint4(0u));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void contact_report_seed(uint3 id : SV_DispatchThreadID)
{
    ContactReportParams p = contact_report_params;
    if (id.x >= p.range.y) return;
    uint ordinal = p.range.x + id.x;
    if (ordinal >= p.range.z) { cr_bad(); return; }
    // Broadphase keeps its candidate bank offset in the immutable header.
    uint4 header = contact_patches[0].leaves;
    if (ordinal >= header.y) return;
    ContactPatch patch = contact_patches[header.x + ordinal];
    if (patch.leaves.w == 0u) return;
    if (patch.leaves.w > 4u || !cr_range(patch.leaves.x, p.selection.w, p.requests.x)
        || !cr_range(patch.leaves.y, p.selection.w, p.requests.x)) { cr_bad(); return; }
    bool sensor = (patch.leaves.z & PatchSensor) != 0u;
    uint interest = contact_report_interest[patch.leaves.x - p.selection.w]
                  | contact_report_interest[patch.leaves.y - p.selection.w];
    bool wanted = sensor ? ((patch.leaves.z & PatchOverlapping) != 0u
        && ((p.requests.y & 2u) != 0u || (interest & 2u) != 0u))
        : ((patch.leaves.z & PatchActive) != 0u
        && ((p.requests.y & 5u) != 0u || (interest & 1u) != 0u));
    if (!wanted) return;
    bool swept = p.requests.z != 0u;
    Contact first = contacts[(swept ? 1u : 4u) * ordinal];
    if (!cr_range(first.meta.x, p.selection.y, p.selection.z)
        || !cr_range(first.meta.y, p.selection.y, p.selection.z)
        || body_models[first.meta.x].meta.w != p.selection.x
        || body_models[first.meta.y].meta.w != p.selection.x) { cr_bad(); return; }
    uint row = contact_report_indices[0].add(1u);
    if (row >= p.range.z) { cr_bad(); return; }
    contact_report_indices[2u + ordinal].store(row);
    cr_word(row, 0u, uint4(patch.leaves.xy, first.meta.xy));
    cr_word(row, 1u, uint4(p.range.w, patch.leaves.z, patch.leaves.w, ordinal));
    if (swept) {
        cr_word(row, 2u, uint4(first.meta.w, 1u, first.meta.z, 0u));
        cr_word(row, 3u, asuint(first.localAnchorA));
        cr_word(row, 4u, asuint(first.normalDepth));
        cr_word(row, 5u, asuint(first.localAnchorB));
        cr_word(row, 6u, asuint(first.impulse));
        return;
    }
    BodyModel ma = jph_effective_model(body_models[first.meta.x], kinematic_targets[first.meta.x]);
    BodyModel mb = jph_effective_model(body_models[first.meta.y], kinematic_targets[first.meta.y]);
    BodyState sa = body_states[first.meta.x], sb = body_states[first.meta.y];
    for (uint i = 0u; i < 4u; ++i) {
        uint base = 2u + 5u * i;
        for (uint k = 0u; k < 5u; ++k) cr_word(row, base + k, uint4(0u));
        if (i >= patch.leaves.w) continue;
        Contact c = contacts[4u * ordinal + i];
        if (any(c.meta.xy != first.meta.xy)) { cr_bad(); continue; }
        float3 pa, pb, ra, rb; jph_contact_points(c, sa, sb, pa, pb, ra, rb);
        cr_word(row, base, uint4(c.meta.w, 1u, c.meta.z, 0u));
        cr_word(row, base + 1u, asuint(float4(0.5f * (pa + pb), c.normalDepth.w)));
        cr_word(row, base + 2u, asuint(float4(c.normalDepth.xyz, 0.0f)));
        cr_word(row, base + 3u, asuint(float4(jph_point_velocity(mb, sb, rb) - jph_point_velocity(ma, sa, ra), 0.0f)));
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void contact_report_finish(uint3 id : SV_DispatchThreadID)
{
    ContactReportParams p = contact_report_params;
    if (id.x >= p.range.y) return;
    uint ordinal = p.range.x + id.x;
    if (ordinal >= p.range.z) return;
    uint row = contact_report_indices[2u + ordinal].load();
    if (row == 0xffffffffu) return;
    if (row >= p.range.z) { cr_bad(); return; }
    uint4 meta = cr_read(row, 1u);
    if ((meta.y & PatchSensor) != 0u) return;
    if (meta.z == 0u || meta.z > 4u || meta.x != p.range.w || meta.w != ordinal) { cr_bad(); return; }
    Contact first = contacts[4u * ordinal];
    float3 t1, t2; jph_tangents(asfloat(cr_read(row, 4u)).xyz, t1, t2);
    float3 share = (t1 * first.impulse.y + t2 * first.impulse.z) / float(meta.z);
    for (uint i = 0u; i < meta.z; ++i) {
        float3 n = asfloat(cr_read(row, 4u + 5u * i)).xyz;
        cr_word(row, 6u + 5u * i, asuint(float4(n * contacts[4u * ordinal + i].impulse.x + share, 0.0f)));
    }
}
