#include "rigid_math.slang"

struct CrossDomainParams
{
    uint4 range; // GPU leaf first/count, observation count address, substep dt bits.
};

[[vk::binding(0, 0)]] ConstantBuffer<CrossDomainParams> cross_domain;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> cpu_tree;
[[vk::binding(2, 0)]] StructuredBuffer<Aabb> gpu_bounds;
[[vk::binding(3, 0)]] StructuredBuffer<ShapeModel> gpu_shapes;
[[vk::binding(4, 0)]] StructuredBuffer<BodyModel> gpu_models;
[[vk::binding(5, 0)]] StructuredBuffer<BodyState> gpu_states;
[[vk::binding(6, 0)]] RWStructuredBuffer<Atomic<uint>> candidate_count;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint4> candidates;

bool inside_range(float3 a_min, float3 a_max, float3 b_min, float3 b_max)
{
    return all(a_min <= b_max) && all(b_min <= a_max);
}

bool has_proxy(float3 lo, float3 hi, uint flags)
{
    uint4 layout = cpu_tree[0]; // node count, escape base, ordered bounds base, proxy count.
    uint node = 0u;
    while (node < layout.x) {
        uint position = 1u + node * 2u;
        uint4 lower = cpu_tree[position], upper = cpu_tree[position + 1u];
        uint escape = cpu_tree[layout.y + node / 4u][node % 4u];
        if (!inside_range(lo, hi, asfloat(lower.xyz), asfloat(upper.xyz))) {
            node = escape;
            continue;
        }
        if ((upper.w & 0x80000000u) == 0u) {
            node = lower.w;
            continue;
        }
        uint end = lower.w + (upper.w & 0x7fffffffu);
        for (uint i = lower.w; i < end; ++i) {
            uint address = layout.z + 2u * i;
            uint4 proxy_min = cpu_tree[address], proxy_max = cpu_tree[address + 1u];
            bool sensor = ((flags | proxy_max.w) & ShapeSensor) != 0u;
            if (sensor && inside_range(lo, hi, asfloat(proxy_min.xyz), asfloat(proxy_max.xyz)))
                return true;
        }
        node = escape;
    }
    return false;
}

// Copy each candidate leaf pose once, even when it overlaps several CPU proxies.
// CPU narrowphase runs after the numerical stage queue has finished.
[shader("compute")]
[numthreads(64, 1, 1)]
void gather_cross_domain_candidates(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= cross_domain.range.y) return;
    uint leaf = cross_domain.range.x + id.x;
    ShapeModel shape = gpu_shapes[leaf];
    uint motion = gpu_models[shape.meta.y].meta.z;
    if (motion != MotionDynamic && motion != MotionKinematic) return;
    BodyState state = gpu_states[shape.meta.y];
    Aabb current = gpu_bounds[leaf];
    float dt = asfloat(cross_domain.range.w);
    float3 translation = dt * state.linearVelocity.xyz;
    float radius = length(max(abs(current.minimum.xyz - state.positionCOM.xyz),
                              abs(current.maximum.xyz - state.positionCOM.xyz)));
    float angular_sweep = radius * min(2.0f, dt * length(state.angularVelocity.xyz));
    float3 lo = current.minimum.xyz + min(translation, 0.0f) - angular_sweep;
    float3 hi = current.maximum.xyz + max(translation, 0.0f) + angular_sweep;
    if (!has_proxy(lo, hi, shape.meta.w)) return;

    uint output = candidate_count[cross_domain.range.z].add(1u) * 7u;
    float3 arm = jph_rotate(state.rotation, shape.localPositionCOM.xyz);
    candidates[output] = uint4(leaf, shape.surface.xy, 0u);
    candidates[output + 1u] = asuint(float4(state.positionCOM.xyz + arm, 0.0f));
    candidates[output + 2u] = asuint(jph_quat_mul(state.rotation, shape.localRotation));
    candidates[output + 3u] = asuint(float4(state.linearVelocity.xyz + cross(state.angularVelocity.xyz, arm), 0.0f));
    candidates[output + 4u] = asuint(state.angularVelocity);
    candidates[output + 5u] = asuint(float4(lo, 0.0f));
    candidates[output + 6u] = asuint(float4(hi, 0.0f));
}
