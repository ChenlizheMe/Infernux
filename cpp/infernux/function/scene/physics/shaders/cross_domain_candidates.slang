#include "rigid_sleep_math.slang"

struct CrossDomainParams
{
    uint4 range; // GPU leaf first/count, observation count address, substep dt bits.
    uint4 policy; // Trigger and response masks.
};

[[vk::binding(0, 0)]] ConstantBuffer<CrossDomainParams> cross_domain;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> cpu_tree;
[[vk::binding(2, 0)]] StructuredBuffer<Aabb> gpu_bounds;
[[vk::binding(3, 0)]] StructuredBuffer<ShapeModel> gpu_shapes;
[[vk::binding(4, 0)]] StructuredBuffer<BodyModel> gpu_models;
[[vk::binding(5, 0)]] StructuredBuffer<BodyState> gpu_states;
[[vk::binding(6, 0)]] RWStructuredBuffer<Atomic<uint>> candidate_count;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint4> candidates;
[[vk::binding(8, 0)]] StructuredBuffer<BodySleepRecord> gpu_sleep;

static const uint ResponseProxy = 2u;
static const uint DynamicProxy = 4u;
static const uint ContinuousProxy = 16u;
static const uint DynamicContinuousProxy = 32u;

bool inside_range(float3 a_min, float3 a_max, float3 b_min, float3 b_max)
{
    return all(a_min <= b_max) && all(b_min <= a_max);
}

bool accepts_continuous_pair(uint cpuFlags, uint gpuMotion, float gpuRadius)
{
    bool cpuDynamic = (cpuFlags & DynamicProxy) != 0u;
    if (cpuDynamic && gpuMotion == MotionDynamic)
        return (cpuFlags & DynamicContinuousProxy) != 0u || gpuRadius < 0.0f;
    return cpuDynamic ? (cpuFlags & ContinuousProxy) != 0u : gpuRadius != 0.0f;
}

bool has_proxy(float3 lo, float3 hi, uint flags, uint motion, bool checkCcdMode, float radius)
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
            bool trigger = ((flags | proxy_max.w) & ShapeSensor) != 0u;
            bool observes = (cross_domain.policy.x & 1u) != 0u && trigger;
            bool responds = (cross_domain.policy.x & 2u) != 0u && (flags & ShapeSensor) == 0u
                && (proxy_max.w & ResponseProxy) != 0u
                && (motion == MotionDynamic || (proxy_max.w & DynamicProxy) != 0u)
                && (!checkCcdMode || accepts_continuous_pair(proxy_max.w, motion, radius));
            if ((observes || responds) && inside_range(lo, hi, asfloat(proxy_min.xyz), asfloat(proxy_max.xyz)))
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
    if (!has_proxy(lo, hi, shape.meta.w, motion, false, 0.0f)) return;

    uint output = candidate_count[cross_domain.range.z].add(1u) * 8u;
    float3 arm = jph_rotate(state.rotation, shape.localPositionCOM.xyz);
    uint active = gpu_sleep[shape.meta.y].runtime.state.x != 0u ? 8u : 0u;
    candidates[output] = uint4(leaf, shape.surface.xy, motion | active);
    candidates[output + 1u] = asuint(float4(state.positionCOM.xyz + arm, 0.0f));
    candidates[output + 2u] = asuint(jph_quat_mul(state.rotation, shape.localRotation));
    candidates[output + 3u] = asuint(float4(state.linearVelocity.xyz + cross(state.angularVelocity.xyz, arm), 0.0f));
    candidates[output + 4u] = asuint(state.angularVelocity);
    candidates[output + 5u] = asuint(float4(lo, 0.0f));
    candidates[output + 6u] = asuint(float4(hi, 0.0f));
    candidates[output + 7u] = uint4(asuint(shape.material.xy), shape.surface.zw);
}

// Continuous discovery uses the already prepared/rotated trajectories and
// transfers only leaf identities. Motion is gathered once per related body.
[shader("compute")]
[numthreads(64, 1, 1)]
void gather_cross_domain_sweep_candidates(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= cross_domain.range.y) return;
    uint leaf = cross_domain.range.x + id.x;
    ShapeModel shape = gpu_shapes[leaf];
    BodyModel body = gpu_models[shape.meta.y];
    uint motion = body.meta.z;
    if (motion != MotionDynamic && motion != MotionKinematic) return;
    Aabb swept = gpu_bounds[leaf];
    // A prior impulse makes this trajectory a CCD source for the rest of this substep.
    bool checkCcdMode = swept.maximum.w == 0.0f;
    if (!has_proxy(swept.minimum.xyz, swept.maximum.xyz, shape.meta.w, motion, checkCcdMode, body.invInertiaDiagonal.w)) return;
    uint output = candidate_count[cross_domain.range.z].add(1u);
    candidates[output] = uint4(leaf, shape.surface.xy, motion);
}
