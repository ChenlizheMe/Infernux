// Engine-owned particle ABI projection. Included by the common query artifact;
// QueryRecordWord, QueryShapeState and rigid quaternion helpers are in scope.
struct ParticleExportParameters
{
    uint4 counts; // Output rows, static rows, source rows, previous rows.
    uint4 revision;
};
struct ParticleExportRow
{
    uint4 selection;
    uint4 identity;
};
ConstantBuffer<ParticleExportParameters> particle_export_params;
StructuredBuffer<ParticleExportRow> particle_export_rows;
StructuredBuffer<QueryRecordWord> particle_export_previous;
RWStructuredBuffer<QueryRecordWord> particle_export_records;
RWStructuredBuffer<QueryRecordWord> particle_export_header;
RWStructuredBuffer<uint> particle_export_offsets;
RWStructuredBuffer<uint> particle_export_indices;
RWStructuredBuffer<Atomic<uint>> particle_export_status;

void particle_export_failure()
{
    particle_export_status[1].max(QueryNumericalFailure);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void particle_collision_project(uint3 tid : SV_DispatchThreadID)
{
    uint row = tid.x;
    if (row >= particle_export_params.counts.x)
        return;
    ParticleExportRow mapping = particle_export_rows[row];
    uint source = mapping.selection.x * 9u;
    uint destination = row * 21u;
    uint4 meta = query_hits[source].value;
    uint4 surface = query_hits[source + 1u].value;
    float4 geometry = asfloat(query_hits[source + 2u].value);
    float3 center = asfloat(query_hits[source + 3u].value).xyz;
    float4 q = asfloat(query_hits[source + 4u].value);
    float3 x = jph_rotate(q, float3(1, 0, 0));
    float3 y = jph_rotate(q, float3(0, 1, 0));
    float3 z = jph_rotate(q, float3(0, 0, 1));
    float4 inverse0 = float4(x, -dot(x, center));
    float4 inverse1 = float4(y, -dot(y, center));
    float4 inverse2 = float4(z, -dot(z, center));
    float4 shape = float4(0.0f);
    float3 extent = float3(0.0f);
    uint type = 0u;
    if (surface.y == ShapeBox)
    {
        float3 core = max(geometry.xyz - geometry.w, float3(0.0f));
        extent = abs(x) * core.x + abs(y) * core.y + abs(z) * core.z + geometry.w;
        shape = float4(2.0f * geometry.xyz, geometry.w);
        type = 0u;
    }
    else if (surface.y == ShapeSphere)
    {
        extent = float3(geometry.w);
        shape.x = geometry.w;
        type = 1u;
    }
    else if (surface.y == ShapeCapsule)
    {
        extent = abs(y) * geometry.y + geometry.w;
        shape = float4(geometry.w, 2.0f * (geometry.y + geometry.w), 1.0f, 0.0f);
        type = 2u;
    }
    else
        particle_export_failure();
    float4 low = float4(center - extent, 0.0f);
    float4 high = float4(center + extent, 0.0f);
    float4 previous0 = inverse0, previous1 = inverse1, previous2 = inverse2;
    float4 previousLow = low, previousHigh = high;
    if (mapping.selection.y != InvalidQueryBodySlot)
    {
        uint previous = mapping.selection.y * 21u;
        previous0 = asfloat(particle_export_previous[previous + 3u].value);
        previous1 = asfloat(particle_export_previous[previous + 4u].value);
        previous2 = asfloat(particle_export_previous[previous + 5u].value);
        previousLow = asfloat(particle_export_previous[previous + 15u].value);
        previousHigh = asfloat(particle_export_previous[previous + 16u].value);
    }
    float4 material = asfloat(query_hits[source + 8u].value);
    material.zw = float2(surface.zw);
    uint flags = ((meta.w & QueryShapeSensor) != 0u ? 1u : 0u) |
                 ((meta.w & QueryShapeMoving) != 0u ? 2u : 0u) | 4u;
    particle_export_records[destination].value = asuint(float4(x.x, y.x, z.x, center.x));
    particle_export_records[destination + 1u].value = asuint(float4(x.y, y.y, z.y, center.y));
    particle_export_records[destination + 2u].value = asuint(float4(x.z, y.z, z.z, center.z));
    particle_export_records[destination + 3u].value = asuint(inverse0);
    particle_export_records[destination + 4u].value = asuint(inverse1);
    particle_export_records[destination + 5u].value = asuint(inverse2);
    particle_export_records[destination + 6u].value = asuint(previous0);
    particle_export_records[destination + 7u].value = asuint(previous1);
    particle_export_records[destination + 8u].value = asuint(previous2);
    particle_export_records[destination + 9u].value = asuint(shape);
    particle_export_records[destination + 10u].value = uint4(0u);
    particle_export_records[destination + 11u].value = query_hits[source + 6u].value;
    particle_export_records[destination + 12u].value = query_hits[source + 7u].value;
    particle_export_records[destination + 13u].value = query_hits[source + 5u].value;
    particle_export_records[destination + 14u].value = asuint(material);
    particle_export_records[destination + 15u].value = asuint(low);
    particle_export_records[destination + 16u].value = asuint(high);
    particle_export_records[destination + 17u].value = asuint(previousLow);
    particle_export_records[destination + 18u].value = asuint(previousHigh);
    particle_export_records[destination + 19u].value = uint4(type, surface.x, flags, 0u);
    particle_export_records[destination + 20u].value = mapping.identity;
    bool moving = (meta.w & QueryShapeMoving) != 0u;
    if (meta.y != 1u || moving != (row >= particle_export_params.counts.y))
        particle_export_failure();
    for (uint word = 0u; word < 19u; ++word)
        if (word != 10u && !all(isfinite(asfloat(particle_export_records[destination + word].value))))
            particle_export_failure();
}

bool particle_collision_range(uint row, float4 grid, uint3 dimensions, out uint3 low, out uint3 high)
{
    uint base = row * 21u;
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
    bool valid = particle_export_status[1].load() == QuerySuccess;
    if (rows != 0u && valid)
    {
        float3 low = float3(3.402823466e+38f), high = -low;
        for (uint row = 0u; row < rows; ++row)
        {
            uint base = row * 21u;
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
        particle_export_indices[cursor] = InvalidQueryBodySlot;
}
