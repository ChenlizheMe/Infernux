// Particle collision records projected from the accepted body and leaf buffers.
// Included by the query artifact for its binding and rigid shape layout.
struct ParticleExportParameters
{
    uint4 counts; // Output rows, static rows, first body slot, previous rows.
    uint4 revision;
};
struct ParticleExportRow
{
    uint4 selection;
    uint4 identity;
    float4 localBoundsMin;
    float4 localBoundsMax;
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
    uint destination = row * 22u;
    const ShapeModel leaf = shape_models[mapping.selection.x];
    BodyState state = body_states[leaf.meta.y - particle_export_params.counts.z];
    if (!query_valid_quaternion(state.rotation))
    {
        particle_export_failure();
        return;
    }
    state.rotation = normalize(state.rotation);
    float3 center;
    float4 q;
    query_leaf_pose(leaf, state, center, q);
    q = normalize(q);
    const float4 geometry = leaf.geometry;
    const uint kind = leaf.meta.x;
    float3 x = jph_rotate(q, float3(1, 0, 0));
    float3 y = jph_rotate(q, float3(0, 1, 0));
    float3 z = jph_rotate(q, float3(0, 0, 1));
    float4 inverse0 = float4(x, -dot(x, center));
    float4 inverse1 = float4(y, -dot(y, center));
    float4 inverse2 = float4(z, -dot(z, center));
    float4 shape = float4(0.0f);
    float3 extent = float3(0.0f);
    uint type = 0u;
    if (kind == ShapeBox)
    {
        float3 core = max(geometry.xyz - geometry.w, float3(0.0f));
        extent = abs(x) * core.x + abs(y) * core.y + abs(z) * core.z + geometry.w;
        shape = float4(2.0f * geometry.xyz, geometry.w);
        type = 0u;
    }
    else if (kind == ShapeSphere)
    {
        extent = float3(geometry.w);
        shape.x = geometry.w;
        type = 1u;
    }
    else if (kind == ShapeCapsule)
    {
        extent = abs(y) * geometry.y + geometry.w;
        shape = float4(geometry.w, 2.0f * (geometry.y + geometry.w), 1.0f, 0.0f);
        type = 2u;
    }
    else if (kind == ShapeCylinder)
    {
        extent = abs(y) * geometry.y + geometry.x * sqrt(max(float3(0), 1.0f - y * y));
        shape = geometry;
        type = 5u;
    }
    else if (kind == ShapeConvexHull)
    {
        float3 halfExtent = 0.5f * (mapping.localBoundsMax.xyz - mapping.localBoundsMin.xyz);
        extent = abs(x) * halfExtent.x + abs(y) * halfExtent.y + abs(z) * halfExtent.z;
        type = 6u;
    }
    else
        particle_export_failure();
    float3 boundsCenter = center;
    if (kind == ShapeConvexHull)
        boundsCenter += jph_rotate(q, 0.5f * (mapping.localBoundsMin.xyz + mapping.localBoundsMax.xyz));
    float4 low = float4(boundsCenter - extent, 0.0f);
    float4 high = float4(boundsCenter + extent, 0.0f);
    float4 previous0 = inverse0, previous1 = inverse1, previous2 = inverse2;
    float4 previousLow = low, previousHigh = high;
    if (mapping.selection.y != InvalidQueryBodySlot)
    {
        uint previous = mapping.selection.y * 22u;
        previous0 = asfloat(particle_export_previous[previous + 3u].value);
        previous1 = asfloat(particle_export_previous[previous + 4u].value);
        previous2 = asfloat(particle_export_previous[previous + 5u].value);
        previousLow = asfloat(particle_export_previous[previous + 15u].value);
        previousHigh = asfloat(particle_export_previous[previous + 16u].value);
    }
    const float4 material = float4(leaf.material.xy, float2(leaf.surface.zw));
    const uint flags = (jph_sensor(leaf) ? 1u : 0u) | (row >= particle_export_params.counts.y ? 2u : 0u) | 4u;
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
    particle_export_records[destination + 10u].value = kind == ShapeConvexHull ? leaf.geometryReference : uint4(0u);
    particle_export_records[destination + 11u].value = asuint(float4(state.linearVelocity.xyz, 0.0f));
    particle_export_records[destination + 12u].value = asuint(float4(state.angularVelocity.xyz, 0.0f));
    particle_export_records[destination + 13u].value = asuint(float4(state.positionCOM.xyz, 0.0f));
    particle_export_records[destination + 14u].value = asuint(material);
    particle_export_records[destination + 15u].value = asuint(low);
    particle_export_records[destination + 16u].value = asuint(high);
    particle_export_records[destination + 17u].value = asuint(previousLow);
    particle_export_records[destination + 18u].value = asuint(previousHigh);
    particle_export_records[destination + 19u].value = uint4(type, 1u << leaf.surface.x, flags, 0u);
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
    bool valid = particle_export_status[1].load() == QuerySuccess;
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
        particle_export_indices[cursor] = InvalidQueryBodySlot;
}
