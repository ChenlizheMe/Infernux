struct Counters
{
    uint free_count;
    uint visible_count;
    uint dropped_count;
    uint reserved_count;
};

struct ParticleInstance
{
    float4 position_size;
    float4 color;
    float4 rotation_custom;
    float4 scale_custom;
    uint4 ribbon_data;
    float4 custom_data;
    float4 previous_position_history;
};

struct DrawArguments
{
    uint vertex_count;
    uint instance_count;
    uint first_vertex;
    uint first_instance;
};

struct DispatchValues
{
    uint capacity;
    uint invocation_count;
    uint spawn_base_id;
    uint spawn_generation;
    uint system_seed;
    uint simulation_step;
    float delta_time;
    uint diagnostic_flags;
    uint alive_read_slot;
    uint alive_write_slot;
    uint use_alive_list;
    uint reserved;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint4> states;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> free_slots;
[[vk::binding(2, 0)]] RWStructuredBuffer<Counters> counters;
[[vk::binding(3, 0)]] RWStructuredBuffer<ParticleInstance> instances;
[[vk::binding(4, 0)]] RWStructuredBuffer<DrawArguments> indirect_args;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> render_indices;
#ifdef INX_FIXTURE_ALIVE_DISPATCH
[[vk::binding(9, 0)]] RWStructuredBuffer<uint4> alive_dispatch_args;
#endif
[[vk::push_constant]] ConstantBuffer<DispatchValues> dispatch;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x != 0u)
        return;

#ifdef INX_FIXTURE_ALIVE_DISPATCH
    // Bootstrap and export reset publish one bounded workgroup for this
    // single-particle fixture; Update/Rendering only consume the arguments.
    alive_dispatch_args[0] = alive_dispatch_args[1] = uint4(1u, 1u, 1u, 0u);
#endif

    states[0] = uint4(1u, dispatch.simulation_step, dispatch.system_seed, dispatch.capacity);
    free_slots[0] = 0u;
    counters[0].visible_count = 1u;
    instances[0] = {
        float4(0.0, 0.0, 0.0, 0.65),
        float4(0.9, 0.2, 0.1, 1.0),
        float4(0.0),
        float4(1.0, 1.0, 1.0, 0.0),
        uint4(7u, dispatch.simulation_step, 0u, dispatch.spawn_base_id),
        float4(0.0),
        float4(0.0, 0.0, 0.0, asfloat(dispatch.spawn_generation)),
    };
    render_indices[0] = 0u;
    indirect_args[0] = {6u, 1u, 0u, 0u};
}
