#include "rigid_sleep_math.slang"

struct CouplingParams
{
    uint4 range; // First group or wake slot, count, and rounds.
    float4 gravityDt;
    float4 contact;
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
    uint4 endpoints; // CPU address, GPU slot, point count, reserved.
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
    gpu_states[b] = sb;
    cpu_velocities[a] = {sa.linearVelocity, sa.angularVelocity};
    Velocity impulses = cpu_impulses[a];
    impulses.linear.xyz -= jt + jn;
    impulses.angular.xyz -= jt_angular + jna;
    cpu_impulses[a] = impulses;
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
    cpu_poses[a] = {sa.positionCOM, sa.rotation};
    sb.linearVelocity = linear;
    sb.angularVelocity = angular;
    gpu_states[b] = sb;
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
