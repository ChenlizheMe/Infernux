// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "collision_geometry.slang"

struct ParticleExportParameters
{
    uint4 counts; // Output rows, static rows, first body row.
    uint4 revision;
};

struct ParticleExportRow
{
    uint4 selection; // Geometry row, previous record, collider generation low/high.
    uint4 identity; // Collider and owner IDs.
    uint4 attributes; // Layer mask, trigger flag, reserved.
    float4 localBounds; // Half extents.
    float4 boundsCenter;
    float4 material;
};

struct ParticleWord { uint4 value; };

[[vk::binding(0, 0)]] ConstantBuffer<ParticleExportParameters> particle_export_params;
[[vk::binding(1, 0)]] StructuredBuffer<ParticleExportRow> particle_export_rows;
[[vk::binding(2, 0)]] StructuredBuffer<ParticleWord> particle_export_previous;
[[vk::binding(3, 0)]] StructuredBuffer<GeomModel> particle_export_geoms;
[[vk::binding(4, 0)]] StructuredBuffer<CollisionShape> particle_export_shapes;
[[vk::binding(5, 0)]] StructuredBuffer<float4> particle_export_bodies;
[[vk::binding(6, 0)]] RWStructuredBuffer<ParticleWord> particle_export_records;
[[vk::binding(7, 0)]] RWStructuredBuffer<ParticleWord> particle_export_header;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> particle_export_offsets;
[[vk::binding(9, 0)]] RWStructuredBuffer<uint> particle_export_indices;
[[vk::binding(10, 0)]] RWStructuredBuffer<Atomic<uint>> particle_export_status;

void particle_export_failure()
{
    particle_export_status[1].max(1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void particle_collision_project(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= particle_export_params.counts.x)
        return;
    const ParticleExportRow mapping = particle_export_rows[id.x];
    const uint row = mapping.selection.x;
    const GeomModel geom = particle_export_geoms[row];
    const CollisionShape geometry = particle_export_shapes[row];
    const uint body = 5u * (particle_export_params.counts.z + geom.indices.x);
    const float3 position = particle_export_bodies[body].xyz;
    const float4 rotation = particle_export_bodies[body + 1u];
    const float3 center = position + rotateVector(geom.position.xyz, rotation);
    const CartesianFrame frame = cartesianFrame(center, mulQuat(rotation, geom.rotation));
    float3 x = frameAxis(frame, 0), y = frameAxis(frame, 1), z = frameAxis(frame, 2);
    float3 extent = abs(x) * mapping.localBounds.x + abs(y) * mapping.localBounds.y + abs(z) * mapping.localBounds.z;
    float4 size = float4(0.0);
    uint type = 0;
    uint4 reference = uint4(0u);
    const uint kind = geometry.type.x;

    if (kind == GeomBox)
    {
        const float3 core = max(geometry.size.xyz - geometry.size.w, float3(0.0));
        extent = abs(x) * core.x + abs(y) * core.y + abs(z) * core.z + geometry.size.w;
        size = float4(2.0 * geometry.size.xyz, geometry.size.w);
    }
    else if (kind == GeomSphere)
    {
        size.x = geometry.size.x;
        extent = float3(size.x);
        type = 1u;
    }
    else if (kind == GeomCapsule || kind == GeomCylinder)
    {
        const float radius = geometry.size.x, half = geometry.size.y;
        const float3 axis = z;
        z = -y;
        y = axis;
        if (kind == GeomCapsule)
        {
            extent = abs(axis) * half + radius;
            size = float4(radius, 2.0 * (half + radius), 1.0, 0.0);
            type = 2u;
        }
        else
        {
            extent = abs(axis) * half + radius * sqrt(max(float3(0.0), 1.0 - axis * axis));
            size = geometry.size;
            type = 5u;
        }
    }
    else if (kind == GeomMesh)
    {
        type = 6u;
        reference = uint4(row, geometry.type.y, geometry.type.z, 0u);
    }
    else if (kind == GeomTriangleMesh)
    {
        type = 7u;
        reference.x = geometry.type.y;
    }
    else
        particle_export_failure();

    const float4 inverse0 = float4(x, -dot(x, center));
    const float4 inverse1 = float4(y, -dot(y, center));
    const float4 inverse2 = float4(z, -dot(z, center));
    const float3 boundsCenter = center + rotatePoint(frame, mapping.boundsCenter.xyz);
    const float4 low = float4(boundsCenter - extent, 0.0);
    const float4 high = float4(boundsCenter + extent, 0.0);
    float4 previous0 = inverse0, previous1 = inverse1, previous2 = inverse2, previousLow = low, previousHigh = high;
    if (mapping.selection.y != NoIndex)
    {
        const uint previous = mapping.selection.y * 22u;
        previous0 = asfloat(particle_export_previous[previous + 3u].value);
        previous1 = asfloat(particle_export_previous[previous + 4u].value);
        previous2 = asfloat(particle_export_previous[previous + 5u].value);
        previousLow = asfloat(particle_export_previous[previous + 15u].value);
        previousHigh = asfloat(particle_export_previous[previous + 16u].value);
    }
    const uint flags = mapping.attributes.y | (id.x >= particle_export_params.counts.y ? 2u : 0u) |
                       (kind == GeomTriangleMesh ? 0u : 4u);
    const uint destination = id.x * 22u;
    particle_export_records[destination].value = asuint(float4(x.x, y.x, z.x, center.x));
    particle_export_records[destination + 1u].value = asuint(float4(x.y, y.y, z.y, center.y));
    particle_export_records[destination + 2u].value = asuint(float4(x.z, y.z, z.z, center.z));
    particle_export_records[destination + 3u].value = asuint(inverse0);
    particle_export_records[destination + 4u].value = asuint(inverse1);
    particle_export_records[destination + 5u].value = asuint(inverse2);
    particle_export_records[destination + 6u].value = asuint(previous0);
    particle_export_records[destination + 7u].value = asuint(previous1);
    particle_export_records[destination + 8u].value = asuint(previous2);
    particle_export_records[destination + 9u].value = asuint(size);
    particle_export_records[destination + 10u].value = reference;
    particle_export_records[destination + 11u].value = asuint(float4(particle_export_bodies[body + 3u].xyz, 0.0));
    particle_export_records[destination + 12u].value = asuint(float4(particle_export_bodies[body + 4u].xyz, 0.0));
    const float3 com = particle_export_bodies[body + 2u].xyz;
    particle_export_records[destination + 13u].value = asuint(float4(com, 0.0));
    particle_export_records[destination + 14u].value = asuint(mapping.material);
    particle_export_records[destination + 15u].value = asuint(low);
    particle_export_records[destination + 16u].value = asuint(high);
    particle_export_records[destination + 17u].value = asuint(previousLow);
    particle_export_records[destination + 18u].value = asuint(previousHigh);
    particle_export_records[destination + 19u].value = uint4(type, mapping.attributes.x, flags, 0u);
    particle_export_records[destination + 20u].value = mapping.identity;
    particle_export_records[destination + 21u].value = uint4(mapping.selection.zw, 0u, 0u);
    for (uint word = 0u; word < 19u; ++word)
        if (word != 10u && !all(isfinite(asfloat(particle_export_records[destination + word].value))))
            particle_export_failure();
}

bool particle_collision_range(uint row, float4 grid, uint3 dimensions, out uint3 low, out uint3 high)
{
    uint base = row * 22u;
    float3 a = min(asfloat(particle_export_records[base + 15u].value).xyz,
                   asfloat(particle_export_records[base + 17u].value).xyz);
    float3 b = max(asfloat(particle_export_records[base + 16u].value).xyz,
                   asfloat(particle_export_records[base + 18u].value).xyz);
    a = (a - grid.xyz) * grid.w;
    b = (b - grid.xyz) * grid.w;
    if (!all(isfinite(a)) || !all(isfinite(b)))
    {
        low = high = uint3(0u);
        return false;
    }
    low = uint3(clamp(floor(a), float3(0.0f), float3(dimensions - 1u)));
    high = uint3(clamp(floor(b), float3(0.0f), float3(dimensions - 1u)));
    return true;
}

// Count and prefix-scan are one bounded owner stage. The initial correctness
// path uses one lane, sharing exactly the range predicate used by scatter.
[shader("compute")]
[numthreads(64, 1, 1)]
void particle_collision_count(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u)
        return;
    uint rows = particle_export_params.counts.x;
    float4 grid = float4(0, 0, 0, 1);
    uint3 dimensions = uint3(1u);
    bool valid = particle_export_status[1].load() == 0u;
    if (rows != 0u && valid)
    {
        float3 low = float3(3.402823466e+38f), high = -low;
        for (uint row = 0u; row < rows; ++row)
        {
            uint base = row * 22u;
            low = min(low, min(asfloat(particle_export_records[base + 15u].value).xyz,
                               asfloat(particle_export_records[base + 17u].value).xyz));
            high = max(high, max(asfloat(particle_export_records[base + 16u].value).xyz,
                                 asfloat(particle_export_records[base + 18u].value).xyz));
        }
        float3 extent = high - low;
        float maximum = max(max(extent.x, extent.y), max(extent.z, 1.0e-3f));
        float cell = maximum / 8.0f;
        low -= cell * 0.001f;
        float3 axes = ceil(max(high - low, float3(0.0f)) / cell);
        grid = float4(low, 1.0f / cell);
        valid = all(isfinite(extent)) && all(isfinite(grid)) && grid.w > 0.0f && all(isfinite(axes));
        if (valid)
            dimensions = uint3(clamp(axes, float3(1.0f), float3(8.0f)));
        else
            particle_export_failure();
    }
    uint cells = dimensions.x * dimensions.y * dimensions.z;
    particle_export_header[0].value = uint4(rows, particle_export_params.counts.y, particle_export_params.revision.xy);
    particle_export_header[1].value = asuint(grid);
    particle_export_header[2].value = uint4(dimensions, cells);
    particle_export_header[3].value = uint4(0u, 0u, 0u, particle_export_params.revision.z);
    uint prefix = 0u;
    for (uint cell = 0u; cell < 512u; ++cell)
    {
        particle_export_offsets[cell] = prefix;
        if (cell >= cells || !valid)
            continue;
        uint3 coordinate = uint3(cell % dimensions.x, (cell / dimensions.x) % dimensions.y,
                                 cell / (dimensions.x * dimensions.y));
        for (uint row = 0u; row < rows; ++row)
        {
            uint3 a, b;
            if (!particle_collision_range(row, grid, dimensions, a, b))
            {
                particle_export_failure();
                valid = false;
                break;
            }
            if (all(coordinate >= a) && all(coordinate <= b))
                ++prefix;
        }
    }
    particle_export_offsets[512] = prefix;
    particle_export_status[0].store(prefix);
    particle_export_status[2].store(rows);
    particle_export_status[3].store(cells);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void particle_collision_scatter(uint3 tid : SV_DispatchThreadID)
{
    uint cell = tid.x;
    uint4 dimensions = particle_export_header[2].value;
    if (cell >= dimensions.w)
        return;
    uint begin = particle_export_offsets[cell], end = particle_export_offsets[cell + 1u];
    uint cursor = begin;
    float4 grid = asfloat(particle_export_header[1].value);
    uint3 coordinate = uint3(cell % dimensions.x, (cell / dimensions.x) % dimensions.y,
                             cell / (dimensions.x * dimensions.y));
    for (uint row = 0u; row < particle_export_params.counts.x; ++row)
    {
        uint3 a, b;
        bool valid = particle_collision_range(row, grid, dimensions.xyz, a, b);
        if (!valid)
            particle_export_failure();
        if (valid && all(coordinate >= a) && all(coordinate <= b))
        {
            if (cursor < end)
                particle_export_indices[cursor++] = row;
            else
                particle_export_failure();
        }
    }
    if (cursor != end)
        particle_export_failure();
    // Even a failed segment fully initializes its declared prefix domain;
    // the explicit status prevents publishing it as a successful scene.
    for (; cursor < end; ++cursor)
        particle_export_indices[cursor] = NoIndex;
}
