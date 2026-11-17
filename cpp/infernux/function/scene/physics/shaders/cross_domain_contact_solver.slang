#include "rigid_sleep_math.slang"

struct CouplingParams
{
    uint4 range; // First group or wake slot, count, and rounds.
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
    uint4 endpoints; // CPU address, GPU slot, point count, previous manifold + 1.
    float4 normalFriction;
    float4 points[4]; // World midpoint and material/speculative bias.
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
    if (patch.endpoints.w == 0u) return;
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
    const Velocity zero = {float4(0), float4(0)};
    for (uint i = group.z; i < group.z + group.w; ++i)
        cpu_impulses[i] = zero;
    if (params.contact.x > 0.0f)
        for (uint i = group.x; i < group.x + group.y; ++i)
            warm_patch(i);
    for (uint round = 0; round < params.range.z; ++round)
        for (uint i = group.x; i < group.x + group.y; ++i)
            solve_patch(i);
}

void project_patch(uint index)
{
    const MixedPatch patch = patches[index];
    const PositionAnchors geometry = anchors[index];
    const uint a = patch.endpoints.x, b = patch.endpoints.y;
    const Pose pose = cpu_poses[a];
    BodyState sa = {pose.center, pose.rotation, float4(0), float4(0)};
    BodyState sb = gpu_states[b];
    const BodyModel ma = cpu_models[a].model, mb = gpu_models[b];
    const float4 linear = sb.linearVelocity, angular = sb.angularVelocity;
    Contact contact = (Contact)0;
    contact.normalDepth = patch.normalFriction;
    for (uint i = 0; i < patch.endpoints.z; ++i)
    {
        contact.localAnchorA = geometry.localA[i];
        contact.localAnchorB = geometry.localB[i];
        jph_solve_position(ma, sa, mb, sb, contact, params.contact.y, params.contact.z, params.contact.w);
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
