#include "rigid_sleep_math.slang"
#include "rigid_joints.slang"
#include "../ContinuousTypes.h"

struct CouplingParams
{
    uint4 range; // First group or wake slot, count, rounds, clear accumulated CPU impulses.
    float4 gravityDt;
    float4 contact; // Warm-start scale, penetration slop, max correction, Baumgarte.
};
struct CpuBoundary
{
    BodyModel model;
    float4 center, rotation;
};
struct Velocity
{
    float4 linear, angular;
};
struct Pose
{
    float4 center, rotation;
};
struct PositionAnchors
{
    float4 localA[4], localB[4];
};
struct MixedPatch
{
    uint4 endpoints; // CPU address, GPU slot, point count, previous manifold + 1 or joint status.
    float4 normalFriction;
    float4 points[4]; // Contact midpoint/bias, or two COM arms and two frames for a fixed joint (zero points).
    float4 normalLambda, frictionLambda;
};

[[vk::binding(0, 0)]] ConstantBuffer<CouplingParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<BodyModel> gpu_models;
[[vk::binding(2, 0)]] RWStructuredBuffer<BodyState> gpu_states;
[[vk::binding(3, 0)]] StructuredBuffer<CpuBoundary> cpu_models;
[[vk::binding(4, 0)]] RWStructuredBuffer<Velocity> cpu_velocities;
[[vk::binding(5, 0)]] StructuredBuffer<uint4> groups;
[[vk::binding(6, 0)]] RWStructuredBuffer<MixedPatch> patches;
[[vk::binding(7, 0)]] RWStructuredBuffer<Velocity> cpu_impulses;
[[vk::binding(8, 0)]] RWStructuredBuffer<BodySleepRecord> gpu_sleep;
[[vk::binding(9, 0)]] StructuredBuffer<BodyForces> gpu_forces;
[[vk::binding(10, 0)]] StructuredBuffer<uint> wake_slots;
[[vk::binding(11, 0)]] RWStructuredBuffer<Pose> cpu_poses;
[[vk::binding(12, 0)]] StructuredBuffer<PositionAnchors> anchors;
[[vk::binding(13, 0)]] StructuredBuffer<MixedPatch> previous_patches;
[[vk::binding(14, 0)]] StructuredBuffer<uint> continuous_slots;
[[vk::binding(15, 0)]] RWStructuredBuffer<SweepBody> continuous_motion;
[[vk::binding(16, 0)]] StructuredBuffer<uint> continuous_choices;
[[vk::binding(17, 0)]] RWStructuredBuffer<ContinuousSample> continuous_samples;
[[vk::binding(18, 0)]] StructuredBuffer<ContinuousRootPolicy> continuous_policies;
[[vk::binding(19, 0)]] StructuredBuffer<ContinuousCpuBody> continuous_cpu;
[[vk::binding(20, 0)]] StructuredBuffer<ContinuousMixedHit> continuous_hits;
[[vk::binding(21, 0)]] RWStructuredBuffer<ContinuousReceipt> continuous_receipts;
[[vk::binding(22, 0)]] RWStructuredBuffer<ContactPatch> continuous_patches;
[[vk::binding(23, 0)]] StructuredBuffer<Contact> continuous_contacts;

void store_velocity_result(uint a, uint b, bool dynamic_a, bool dynamic_b,
                           BodyState sa, BodyState sb, float3 impulse, float3 torqueA)
{
    if (dynamic_b) gpu_states[b] = sb;
    if (!dynamic_a) return;
    cpu_velocities[a] = {sa.linearVelocity, sa.angularVelocity};
    Velocity accumulated = cpu_impulses[a];
    accumulated.linear.xyz -= impulse;
    accumulated.angular.xyz -= torqueA;
    cpu_impulses[a] = accumulated;
}

void warm_patch(uint index)
{
    MixedPatch patch = patches[index];
    if (patch.endpoints.z == 0u || patch.endpoints.w == 0u) return;
    const MixedPatch previous = previous_patches[patch.endpoints.w - 1u];
    const uint a = patch.endpoints.x, b = patch.endpoints.y, count = patch.endpoints.z;
    const float3 normal = patch.normalFriction.xyz, oldNormal = previous.normalFriction.xyz;
    const float normalAlignment = dot(normal, oldNormal);
    const float scale = params.contact.x;
    float4 normalLambda = float4(0);
    float3 center = float3(0);
    for (uint point = 0; point < count; ++point)
    {
        // On this first dispatch the host packed previous point indices + 1.
        const uint previousPoint = uint(patch.normalLambda[point]);
        if (previousPoint != 0u)
            normalLambda[point] = previous.normalLambda[previousPoint - 1u] * (scale * normalAlignment);
        center += patch.points[point].xyz;
    }
    center /= float(count);

    float3 oldT1, oldT2, t1, t2;
    jph_tangents(oldNormal, oldT1, oldT2);
    jph_tangents(normal, t1, t2);
    const float normalSum = dot(normalLambda, float4(1));
    const float3 oldFriction = (oldT1 * previous.frictionLambda.x + oldT2 * previous.frictionLambda.y) * scale;
    const float frictionLimit = patch.normalFriction.w * normalSum;
    float2 tangent = float2(dot(oldFriction, t1), dot(oldFriction, t2));
    const float tangentSquared = dot(tangent, tangent);
    if (tangentSquared > frictionLimit * frictionLimit)
        tangent *= frictionLimit / sqrt(tangentSquared);

    const CpuBoundary boundary = cpu_models[a];
    const Velocity incoming = cpu_velocities[a];
    const BodyModel ma = boundary.model, mb = gpu_models[b];
    BodyState sa = {boundary.center, boundary.rotation, incoming.linear, incoming.angular};
    BodyState sb = gpu_states[b];
    float3 torqueA = float3(0), torqueB = float3(0);
    float weightedRadius = 0.0f;
    for (uint point = 0; point < count; ++point)
    {
        const float3 midpoint = patch.points[point].xyz;
        const float3 offset = midpoint - center;
        weightedRadius += length(offset - normal * dot(offset, normal)) * normalLambda[point];
        const float3 impulse = normal * normalLambda[point];
        torqueA += cross(midpoint - sa.positionCOM.xyz, impulse);
        torqueB += cross(midpoint - sb.positionCOM.xyz, impulse);
    }
    const float spinLimit = patch.normalFriction.w * weightedRadius;
    const float spin = clamp(previous.frictionLambda.z * (scale * normalAlignment), -spinLimit, spinLimit);
    const float3 friction = t1 * tangent.x + t2 * tangent.y;
    torqueA += cross(center - sa.positionCOM.xyz, friction) + normal * spin;
    torqueB += cross(center - sb.positionCOM.xyz, friction) + normal * spin;
    const float3 impulse = normal * normalSum + friction;
    sa.linearVelocity.xyz -= jph_inverse_linear_mass(ma, impulse);
    sa.angularVelocity.xyz -= jph_inverse_inertia(ma, sa, torqueA);
    sb.linearVelocity.xyz += jph_inverse_linear_mass(mb, impulse);
    sb.angularVelocity.xyz += jph_inverse_inertia(mb, sb, torqueB);
    store_velocity_result(a, b, jph_dynamic(ma), jph_dynamic(mb), sa, sb, impulse, torqueA);
    patch.normalLambda = normalLambda;
    patch.frictionLambda = float4(tangent, spin, 0);
    patches[index] = patch;
}

JointFixedState fixed_state(MixedPatch patch)
{
    JointFixedState state = (JointFixedState)0;
    state.flags = uint4(1u, patch.endpoints.w, 0u, 0u);
    state.anchor1 = patch.points[0];
    state.anchor2 = patch.points[1];
    state.frame1 = patch.points[2];
    state.frame2 = patch.points[3];
    return state;
}

JointFixedAuthor fixed_author()
{
    JointFixedAuthor author = (JointFixedAuthor)0;
    author.bodies.w = JointEnabled;
    // The host already expresses both anchors relative to their current COM.
    author.scale1 = float4(1);
    author.scale2 = float4(1);
    return author;
}

// A thread owns every CPU and GPU endpoint of one connected cross-domain
// group. Local constraint solvers have completed their current velocity round.
void solve_patch(uint index)
{
    MixedPatch patch = patches[index];
    uint a = patch.endpoints.x;
    uint b = patch.endpoints.y;
    uint count = patch.endpoints.z;
    CpuBoundary boundary = cpu_models[a];
    Velocity incoming = cpu_velocities[a];
    BodyModel ma = boundary.model, mb = gpu_models[b];
    BodyState sa = {boundary.center, boundary.rotation, incoming.linear, incoming.angular};
    BodyState sb = gpu_states[b];
    if (count == 0u)
    {
        JointFixedState fixed = fixed_state(patch);
        const JointFixedAuthor author = fixed_author();
        jph_joint_solve_velocity(author, fixed, ma, sa, mb, sb, params.gravityDt.w);
        if (fixed.flags.y != JointStatusSuccess)
        {
            patch.endpoints.w = fixed.flags.y;
            patches[index] = patch;
            return;
        }
        const float3 j = fixed.pointImpulse.xyz;
        const float3 torque = fixed.angularImpulse.xyz + cross(jph_rotate(sa.rotation, fixed.anchor1.xyz), j);
        store_velocity_result(a, b, jph_dynamic(ma), jph_dynamic(mb), sa, sb, j, torque);
        return;
    }
    float3 normal = patch.normalFriction.xyz;
    float3 center = float3(0.0f);
    float3 ja[4], jb[4];
    float4 bias = float4(0.0f);
    for (uint i = 0; i < count; ++i)
    {
        float3 p = patch.points[i].xyz;
        center += p;
        ja[i] = cross(p - sa.positionCOM.xyz, normal);
        jb[i] = cross(p - sb.positionCOM.xyz, normal);
        bias[i] = patch.points[i].w;
    }
    for (uint i = count; i < 4u; ++i)
    {
        ja[i] = float3(0.0f);
        jb[i] = float3(0.0f);
    }
    center /= float(count);
    const float3 ra = center - sa.positionCOM.xyz;
    const float3 rb = center - sb.positionCOM.xyz;
    const float friction = patch.normalFriction.w;

    float2 tangent = patch.frictionLambda.xy;
    jph_solve_friction(ma, sa, ra, mb, sb, rb, normal,
                       friction * dot(patch.normalLambda, float4(1.0f)), tangent);
    float3 t1, t2;
    jph_tangents(normal, t1, t2);
    const float3 jt = t1 * (tangent.x - patch.frictionLambda.x) + t2 * (tangent.y - patch.frictionLambda.y);
    float angular = patch.frictionLambda.z;
    if (count > 1u)
    {
        float weighted_radius = 0.0f;
        for (uint i = 0; i < count; ++i)
        {
            float3 offset = patch.points[i].xyz - center;
            weighted_radius += length(offset - normal * dot(normal, offset)) * patch.normalLambda[i];
        }
        jph_solve_angular_friction(ma, sa, mb, sb, normal, friction * weighted_radius, angular);
    }
    const float3 jt_angular = cross(ra, jt) + normal * (angular - patch.frictionLambda.z);
    const float4 next = jph_solve_normal_block(ma, sa, mb, sb, normal, ja, jb, patch.normalLambda, bias, count);
    const float4 delta = next - patch.normalLambda;
    const float3 jn = normal * dot(delta, float4(1.0f));
    float3 jna = float3(0.0f), jnb = float3(0.0f);
    for (uint i = 0; i < count; ++i)
    {
        jna += ja[i] * delta[i];
        jnb += jb[i] * delta[i];
    }
    sa.linearVelocity.xyz -= jph_inverse_linear_mass(ma, jn);
    sa.angularVelocity.xyz -= jph_inverse_inertia(ma, sa, jna);
    sb.linearVelocity.xyz += jph_inverse_linear_mass(mb, jn);
    sb.angularVelocity.xyz += jph_inverse_inertia(mb, sb, jnb);
    store_velocity_result(a, b, jph_dynamic(ma), jph_dynamic(mb), sa, sb, jt + jn, jt_angular + jna);
    patch.normalLambda = next;
    patch.frictionLambda.xyz = float3(tangent, angular);
    patches[index] = patch;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void exchange_cross_domain_velocity(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= params.range.y) return;
    uint4 group = groups[params.range.x + id.x];
    if (params.range.w != 0u)
    {
        const Velocity zero = {float4(0), float4(0)};
        for (uint i = group.z; i < group.z + group.w; ++i)
            cpu_impulses[i] = zero;
    }
    if (params.contact.x > 0.0f)
        for (uint i = group.x; i < group.x + group.y; ++i)
            warm_patch(i);
    for (uint round = 0; round < params.range.z; ++round)
        for (uint i = group.x; i < group.x + group.y; ++i)
            solve_patch(i);
}

void project_patch(uint index)
{
    MixedPatch patch = patches[index];
    const uint a = patch.endpoints.x, b = patch.endpoints.y;
    const Pose pose = cpu_poses[a];
    BodyState sa = {pose.center, pose.rotation, float4(0), float4(0)};
    BodyState sb = gpu_states[b];
    const BodyModel ma = cpu_models[a].model, mb = gpu_models[b];
    const float4 linear = sb.linearVelocity, angular = sb.angularVelocity;
    if (patch.endpoints.z == 0u)
    {
        JointFixedState fixed = fixed_state(patch);
        jph_joint_solve_position(fixed_author(), fixed, ma, sa, mb, sb, params.contact.w);
        if (fixed.flags.y != JointStatusSuccess)
        {
            patch.endpoints.w = fixed.flags.y;
            patches[index] = patch;
            return;
        }
    }
    else
    {
        const PositionAnchors geometry = anchors[index];
        Contact contact = (Contact)0;
        contact.normalDepth = patch.normalFriction;
        for (uint i = 0; i < patch.endpoints.z; ++i)
        {
            contact.localAnchorA = geometry.localA[i];
            contact.localAnchorB = geometry.localB[i];
            jph_solve_position(ma, sa, mb, sb, contact, params.contact.y, params.contact.z, params.contact.w);
        }
    }
    if (jph_dynamic(ma)) cpu_poses[a] = {sa.positionCOM, sa.rotation};
    if (jph_dynamic(mb))
    {
        sb.linearVelocity = linear;
        sb.angularVelocity = angular;
        gpu_states[b] = sb;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void exchange_cross_domain_position(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= params.range.y) return;
    const uint4 group = groups[params.range.x + id.x];
    for (uint round = 0; round < params.range.z; ++round)
        for (uint i = group.x; i < group.x + group.y; ++i)
            project_patch(i);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void wake_cross_domain_bodies(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= params.range.y) return;
    uint slot = wake_slots[params.range.x + id.x];
    BodySleepRecord record = gpu_sleep[slot];
    if (record.runtime.state.x != 0u) return;
    BodyState state = gpu_states[slot];
    float3 p0, p1, p2;
    sleep_test_points(state, record.author.rootHalfExtent.xyz, p0, p1, p2);
    sleep_reset(record.runtime, p0, p1, p2);
    record.runtime.state.x = 1u;
    gpu_sleep[slot] = record;
    jph_apply_body_inputs(gpu_models[slot], state, gpu_forces[slot], params.gravityDt);
    gpu_states[slot] = state;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gather_continuous_boundaries(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.range.y) return;
    uint row = params.range.x + tid.x;
    uint slot = continuous_slots[row];
    uint local = slot - params.range.z;
    uint root = continuous_choices[local];
    uint ordinal = continuous_choices[2u * params.range.w + root];
    SweepBody motion = continuous_motion[local];
    uint first = 0xffffffffu;
    if (ordinal != 0xffffffffu)
        first = asuint(continuous_contacts[ordinal].impulse.w);
    ContinuousSample sample = {jph_sweep_state(motion), motion.displacement, uint4(slot, root, first, 0u)};
    continuous_samples[row] = sample;
}

uint continuous_policy(uint root)
{
    uint lo = 0u, hi = params.range.w;
    while (lo < hi)
    {
        uint mid = lo + (hi - lo) / 2u;
        if (continuous_policies[mid].decision.x < root) lo = mid + 1u;
        else hi = mid;
    }
    return lo < params.range.w && continuous_policies[lo].decision.x == root ? lo : 0xffffffffu;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void restrict_continuous_motion(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.range.y) return;
    uint local = params.range.x + tid.x;
    uint policy = continuous_policy(continuous_choices[local]);
    if (policy == 0xffffffffu) return;
    SweepBody motion = continuous_motion[local];
    if (motion.displacement.w == 0.0f) return; // Prescribed boundaries keep their full motion.
    motion.linearResponse = float4(0.0f, 0.0f, 0.0f,
        min(motion.linearResponse.w, asfloat(continuous_policies[policy].decision.y)));
    motion.angularResponse = float4(0.0f);
    continuous_motion[local] = motion;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void restrict_continuous_reports(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.range.y) return;
    uint ordinal = params.range.x + tid.x;
    uint address = continuous_patches[0].leaves.x + ordinal;
    ContactPatch patch = continuous_patches[address];
    if ((patch.leaves.z & PatchActive) == 0u) return;
    Contact hit = continuous_contacts[ordinal];
    uint policy = continuous_policy(continuous_choices[hit.meta.x - params.range.z]);
    if (policy != 0xffffffffu)
    {
        patch.leaves.zw = uint2(0u);
        continuous_patches[address] = patch;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void apply_cross_domain_sweep(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.range.y) return;
    uint row = params.range.x + tid.x;
    ContinuousMixedHit hit = continuous_hits[row];
    ContinuousCpuBody a = continuous_cpu[hit.bodies.x];
    uint slot = hit.bodies.y, local = slot - params.range.z;
    SweepBody b = continuous_motion[local];
    BodyModel ma = a.model, mb = gpu_models[slot];
    BodyState sa = a.state, sb = jph_sweep_state(b);
    if (!jph_dynamic(mb)) mb.meta.z = MotionKinematic;
    float3 ra = hit.pointRestitution.xyz - (sa.positionCOM.xyz + hit.time.x * a.displacement.xyz);
    float3 rb = hit.pointRestitution.xyz - (sb.positionCOM.xyz + hit.time.x * b.displacement.xyz);
    float3 incoming;
    float3 impulse = jph_solve_sweep_impact(ma, sa, ra, mb, sb, rb, hit.normalFriction.xyz,
        hit.normalFriction.w, hit.pointRestitution.w, hit.time.z, incoming);
    if (jph_dynamic(mb))
    {
        continuous_motion[local].linearResponse = float4(
            jph_clamp_speed(sb.linearVelocity.xyz, mb.damping.z) - b.linearVelocity.xyz, hit.time.y);
        continuous_motion[local].angularResponse = float4(
            jph_clamp_speed(sb.angularVelocity.xyz, mb.damping.w) - b.angularVelocity.xyz, 1.0f);
    }
    ContinuousReceipt receipt;
    receipt.linear = float4(-impulse, 0.0f);
    receipt.angular = float4(-cross(ra, impulse), 0.0f);
    receipt.relative = float4(-incoming, 0.0f);
    continuous_receipts[row] = receipt;
}
