import inx_ray_layout;

// The CPU owns immutable topology. Every invocation reads the exact canonical
// Vertex allocation retained by the raster World and writes a new ray page.
StructuredBuffer<uint> vertex_source;
StructuredBuffer<uint> deformation_plan;
StructuredBuffer<uint4> mesh_page;
RWStructuredBuffer<uint4> geometry_page;

struct RefitParameters
{
    uint4 work; // mode, count, compact-to-source offset, node-list offset
    uint4 destination; // nodes, positions, attributes, mesh vertex base
    uint4 mesh; // static BLAS root, TLAS node offset, TLAS instance offset, reserved
};

INX_ROOT_CONSTANTS ConstantBuffer<RefitParameters> pc;

float3 position_at(uint vertex)
{
    return asfloat(geometry_page[pc.destination.y + vertex - pc.destination.w].xyz);
}

void write_bounds(uint offset, float3 minimum, float3 maximum)
{
    uint4 low = geometry_page[offset];
    uint4 high = geometry_page[offset + 1u];
    geometry_page[offset] = uint4(asuint(minimum), low.w);
    geometry_page[offset + 1u] = uint4(asuint(maximum), high.w);
}

void merge_children(uint nodes, uint index, bool instance_tree)
{
    uint offset = nodes + 2u * index;
    uint a, b;
    if (instance_tree)
    {
        a = geometry_page[offset].w;
        b = geometry_page[offset + 1u].w;
    }
    else
    {
        uint source = mesh_page[0].x + 2u * (pc.mesh.x + index);
        a = mesh_page[source].w - pc.mesh.x;
        b = mesh_page[source + 1u].w - pc.mesh.x;
        geometry_page[offset].w = a;
        geometry_page[offset + 1u].w = b;
    }
    float3 low = min(asfloat(geometry_page[nodes + 2u * a].xyz),
                     asfloat(geometry_page[nodes + 2u * b].xyz));
    float3 high = max(asfloat(geometry_page[nodes + 2u * a + 1u].xyz),
                      asfloat(geometry_page[nodes + 2u * b + 1u].xyz));
    write_bounds(offset, low, high);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void refit_geometry(uint3 thread : SV_DispatchThreadID)
{
    uint row = thread.x + pc.mesh.w;
    if (row >= pc.work.y)
        return;
    uint mode = pc.work.x;
    if (mode == 0u)
    {
        uint source = 25u * deformation_plan[pc.work.z + row];
        float3 position = asfloat(uint3(vertex_source[source], vertex_source[source + 1u], vertex_source[source + 2u]));
        geometry_page[pc.destination.y + row] = uint4(asuint(position), 0u);
        uint attributes = pc.destination.z + 4u * row;
        geometry_page[attributes] = uint4(vertex_source[source + 13u], vertex_source[source + 14u],
                                         vertex_source[source + 15u], vertex_source[source + 16u]);
        geometry_page[attributes + 1u] = uint4(vertex_source[source + 3u], vertex_source[source + 4u],
                                              vertex_source[source + 5u], 0u);
        geometry_page[attributes + 2u] = uint4(vertex_source[source + 6u], vertex_source[source + 7u],
                                              vertex_source[source + 8u], vertex_source[source + 9u]);
        geometry_page[attributes + 3u] = uint4(vertex_source[source + 10u], vertex_source[source + 11u],
                                              vertex_source[source + 12u], asuint(1.0f));
        return;
    }

    uint node = deformation_plan[pc.work.w + row];
    if (mode == 1u)
    {
        uint static_nodes = mesh_page[0].x;
        uint4 low = mesh_page[static_nodes + 2u * (pc.mesh.x + node)];
        uint4 high = mesh_page[static_nodes + 2u * (pc.mesh.x + node) + 1u];
        geometry_page[pc.destination.x + 2u * node] = low;
        geometry_page[pc.destination.x + 2u * node + 1u] = high;
        uint triangles = mesh_page[0].z;
        float3 minimum = float3(3.402823e38f);
        float3 maximum = -minimum;
        for (uint index = 0u; index < (high.w & 0x7fffffffu); ++index)
        {
            uint3 corners = mesh_page[triangles + low.w + index].xyz;
            float3 a = position_at(corners.x);
            float3 b = position_at(corners.y);
            float3 c = position_at(corners.z);
            minimum = min(minimum, min(a, min(b, c)));
            maximum = max(maximum, max(a, max(b, c)));
        }
        write_bounds(pc.destination.x + 2u * node, minimum, maximum);
        return;
    }
    if (mode == 2u)
    {
        merge_children(pc.destination.x, node, false);
        return;
    }
    if (mode == 4u)
    {
        merge_children(pc.mesh.y, node, true);
        return;
    }

    // Rebuild TLAS leaf bounds from the computed BLAS roots. Authored CPU
    // envelope bounds do not constrain the GPU positions used by ray queries.
    uint offset = pc.mesh.y + 2u * node;
    uint first = geometry_page[offset].w;
    uint count = geometry_page[offset + 1u].w & 0x7fffffffu;
    float3 minimum = float3(3.402823e38f);
    float3 maximum = -minimum;
    for (uint index = first; index < first + count; ++index)
    {
        uint instance = pc.mesh.z + ray_instance_words * index;
        float4 r0 = asfloat(geometry_page[instance + 4u]);
        float4 r1 = asfloat(geometry_page[instance + 5u]);
        float4 r2 = asfloat(geometry_page[instance + 6u]);
        uint root = geometry_page[instance + 3u].x;
        float3 low, high;
        if ((root & 0x80000000u) != 0u)
        {
            uint nodes = geometry_page[root & 0x7fffffffu].x;
            low = asfloat(geometry_page[nodes].xyz);
            high = asfloat(geometry_page[nodes + 1u].xyz);
        }
        else
        {
            uint nodes = mesh_page[0].x + 2u * root;
            low = asfloat(mesh_page[nodes].xyz);
            high = asfloat(mesh_page[nodes + 1u].xyz);
        }
        float3 c0 = float3(r0.x, r1.x, r2.x);
        float3 c1 = float3(r0.y, r1.y, r2.y);
        float3 c2 = float3(r0.z, r1.z, r2.z);
        float3 translation = float3(r0.w, r1.w, r2.w);
        float3 a0 = c0 * low.x, b0 = c0 * high.x;
        float3 a1 = c1 * low.y, b1 = c1 * high.y;
        float3 a2 = c2 * low.z, b2 = c2 * high.z;
        float3 world_low = translation + min(a0, b0) + min(a1, b1) + min(a2, b2);
        float3 world_high = translation + max(a0, b0) + max(a1, b1) + max(a2, b2);
        float3 rounding = 1.0e-5f + 1.0e-6f * (abs(translation) + max(abs(a0), abs(b0)) +
                                                  max(abs(a1), abs(b1)) + max(abs(a2), abs(b2)));
        minimum = min(minimum, world_low - rounding);
        maximum = max(maximum, world_high + rounding);
    }
    write_bounds(offset, minimum, maximum);
}
