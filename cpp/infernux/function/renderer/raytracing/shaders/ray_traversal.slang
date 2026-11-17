// Both buffers use uint4 word offsets. TLAS nodes and instances live in the
// geometry page. Mesh nodes, positions, triangles and attributes live in the
// mesh page, so instance roots index the mesh node array.
static const uint ray_leaf_bit = 0x80000000u;
static const uint ray_invalid_index = 0xffffffffu;
static const uint ray_stack_capacity = 48u;
static const uint ray_page_ready = 1u;

StructuredBuffer<uint4> world_ray_geometry;
StructuredBuffer<uint4> world_ray_meshes;

struct RayPage
{
    uint root;
    uint top_nodes;
    uint instances;
    uint status;
    uint mesh_nodes;
    uint positions;
    uint triangles;
    uint attributes;
};

RayPage ray_page()
{
    uint4 geometry = world_ray_geometry[0];
    uint4 meshes = world_ray_meshes[0];
    return {geometry.x, geometry.y, geometry.z, geometry.w, meshes.x, meshes.y, meshes.z, meshes.w};
}

struct RayNode
{
    float3 minimum;
    uint first;
    float3 maximum;
    uint second;
};

struct RayInstance
{
    float4 world_to_object_x;
    float4 world_to_object_y;
    float4 world_to_object_z;
    uint root;
    uint surface_id;
    uint layer_mask;
    float orientation;
};

RayNode ray_node<let MeshNodes : bool>(uint nodes, uint index)
{
    uint offset = nodes + 2u * index;
    uint4 low, high;
    if (MeshNodes)
    {
        low = world_ray_meshes[offset];
        high = world_ray_meshes[offset + 1u];
    }
    else
    {
        low = world_ray_geometry[offset];
        high = world_ray_geometry[offset + 1u];
    }
    return {asfloat(low.xyz), low.w, asfloat(high.xyz), high.w};
}

RayInstance ray_instance(RayPage page, uint index)
{
    uint offset = page.instances + 4u * index;
    uint4 record = world_ray_geometry[offset + 3u];
    return {asfloat(world_ray_geometry[offset]), asfloat(world_ray_geometry[offset + 1u]),
            asfloat(world_ray_geometry[offset + 2u]), record.x, record.y, record.z, asfloat(record.w)};
}

struct TraceRay
{
    float3 origin;
    float minimum;
    float3 direction;
    float maximum;
};

struct RayHit
{
    float distance;
    float2 barycentrics; // Weights of b and c. The weight of a is 1 - b - c.
    uint instance_index; // TLAS instance order. A BLAS can have many different world transforms.
    uint triangle_index;
    uint primitive;
    uint surface_id;
};

interface RayHitPolicy
{
    bool accepts(RayPage page, TraceRay ray, RayHit candidate);
}

struct RayCursor
{
    uint2 pending[ray_stack_capacity]; // Node index and its conservative entry distance.
    uint count;
    uint2 current;
};

bool ray_box_entry(TraceRay ray, RayNode node, float maximum, out float entry)
{
    const float rounding = 3.57628e-7f;
    bool3 parallel = ray.direction == float3(0.0f);
    float3 direction = select(parallel, float3(1.0f), ray.direction);
    float3 a = (node.minimum - ray.origin) / direction;
    float3 b = (node.maximum - ray.origin) / direction;
    float3 low = max(min(a, b), ray.minimum);
    float3 high = min(max(a, b), maximum);
    bool3 outside = (ray.origin < node.minimum) | (ray.origin > node.maximum);
    bool3 disjoint = (low > maximum) | (high < ray.minimum);
    bool3 missed = select(parallel, outside, disjoint);
    float3 near_distance = select(parallel, float3(ray.minimum), low - abs(low) * rounding);
    float3 far_distance = select(parallel, float3(maximum), high + abs(high) * rounding);
    entry = max(ray.minimum, max(near_distance.x, max(near_distance.y, near_distance.z)));
    return !any(missed) && entry <= min(maximum, min(far_distance.x, min(far_distance.y, far_distance.z)));
}

void ray_begin<let MeshNodes : bool>(uint nodes, out RayCursor cursor, TraceRay ray, uint root, float maximum)
{
    cursor.count = 0u;
    float entry = ray.minimum;
    cursor.current = uint2(ray_invalid_index, asuint(entry));
    if (root != ray_invalid_index && ray_box_entry(ray, ray_node<MeshNodes>(nodes, root), maximum, entry))
        cursor.current = uint2(root, asuint(entry));
}

uint2 ray_continue(inout RayCursor cursor)
{
    if (cursor.count == 0u)
        return uint2(ray_invalid_index, 0u);
    cursor.count -= 1u;
    return cursor.pending[cursor.count];
}

uint2 ray_next_leaf<let MeshNodes : bool>(uint nodes, inout RayCursor cursor, TraceRay ray, float maximum)
{
    while (cursor.current.x != ray_invalid_index)
    {
        if (asfloat(cursor.current.y) <= maximum)
        {
            RayNode node = ray_node<MeshNodes>(nodes, cursor.current.x);
            if ((node.second & ray_leaf_bit) != 0u)
            {
                cursor.current = ray_continue(cursor);
                return uint2(node.first, node.second & ~ray_leaf_bit);
            }

            float left_entry, right_entry;
            bool left = ray_box_entry(ray, ray_node<MeshNodes>(nodes, node.first), maximum, left_entry);
            bool right = ray_box_entry(ray, ray_node<MeshNodes>(nodes, node.second), maximum, right_entry);
            uint2 left_record = uint2(node.first, asuint(left_entry));
            uint2 right_record = uint2(node.second, asuint(right_entry));
            if (left && right)
            {
                bool left_first = left_entry <= right_entry;
                cursor.pending[cursor.count++] = left_first ? right_record : left_record;
                cursor.current = left_first ? left_record : right_record;
                continue;
            }
            if (left || right)
            {
                cursor.current = left ? left_record : right_record;
                continue;
            }
        }
        cursor.current = ray_continue(cursor);
    }
    return uint2(0u);
}

struct RayProjection
{
    uint3 axes;
    float2 shear;
    float inverse_depth;
};

RayProjection ray_projection(float3 direction)
{
    float3 magnitude = abs(direction);
    uint z = magnitude.x > magnitude.y ? 0u : 1u;
    z = magnitude[z] > magnitude.z ? z : 2u;
    uint x = (z + 1u) % 3u;
    uint y = (x + 1u) % 3u;
    RayProjection projection;
    projection.axes = uint3(x, y, z);
    projection.shear = float2(direction[x], direction[y]) / direction[z];
    projection.inverse_depth = 1.0f / direction[z];
    return projection;
}

float3 ray_project_point(float3 point, RayProjection projection)
{
    uint3 axes = projection.axes;
    float2 xy = float2(point[axes.x], point[axes.y]) - projection.shear * point[axes.z];
    return float3(xy, point[axes.z] * projection.inverse_depth);
}

// Canonical evaluation makes opposite orders of a shared edge return opposite
// signs even on targets that contract floating point operations.
float ray_edge(float2 a, float2 b)
{
    bool ascending = a.x < b.x || (a.x == b.x && a.y < b.y);
    float2 start = ascending ? a : b;
    float2 end = ascending ? b : a;
    float edge = start.x * end.y - start.y * end.x;
    return all(a == b) ? 0.0f : ascending ? edge : -edge;
}

bool ray_intersect_triangle(RayPage page, TraceRay ray, RayProjection projection, uint3 corners, float maximum,
                            out float distance, out float2 weights)
{
    float3 a = ray_project_point(asfloat(world_ray_meshes[page.positions + corners.x].xyz) - ray.origin, projection);
    float3 b = ray_project_point(asfloat(world_ray_meshes[page.positions + corners.y].xyz) - ray.origin, projection);
    float3 c = ray_project_point(asfloat(world_ray_meshes[page.positions + corners.z].xyz) - ray.origin, projection);
    float3 edge = float3(ray_edge(b.xy, c.xy), ray_edge(c.xy, a.xy), ray_edge(a.xy, b.xy));
    float det = edge.x + edge.y + edge.z;
    distance = maximum;
    weights = float2(0.0f);
    if ((any(edge < 0.0f) && any(edge > 0.0f)) || det == 0.0f)
        return false;

    float inverse_det = 1.0f / det;
    distance = dot(edge, float3(a.z, b.z, c.z)) * inverse_det;
    weights = edge.yz * inverse_det;
    return distance >= ray.minimum && distance <= maximum;
}

TraceRay ray_to_object(TraceRay ray, RayInstance instance)
{
    float4 point = float4(ray.origin, 1.0f);
    float3 direction = ray.direction;
    ray.origin = float3(dot(point, instance.world_to_object_x), dot(point, instance.world_to_object_y),
                        dot(point, instance.world_to_object_z));
    ray.direction = float3(dot(direction, instance.world_to_object_x.xyz), dot(direction, instance.world_to_object_y.xyz),
                           dot(direction, instance.world_to_object_z.xyz));
    return ray;
}

RayHit trace_ray<Policy : RayHitPolicy>(RayPage page, TraceRay world_ray, uint layer_mask, bool stop_on_first_hit,
                                       Policy policy)
{
    RayHit hit;
    hit.distance = world_ray.maximum;
    hit.barycentrics = float2(0.0f);
    hit.instance_index = ray_invalid_index;
    hit.triangle_index = ray_invalid_index;
    hit.primitive = ray_invalid_index;
    hit.surface_id = ray_invalid_index;

    RayCursor world;
    ray_begin<false>(page.top_nodes, world, world_ray, page.root, hit.distance);
    for (;;)
    {
        uint2 leaf = ray_next_leaf<false>(page.top_nodes, world, world_ray, hit.distance);
        if (leaf.y == 0u)
            return hit;
        for (uint index = leaf.x; index < leaf.x + leaf.y; ++index)
        {
            RayInstance instance = ray_instance(page, index);
            if ((instance.layer_mask & layer_mask) == 0u)
                continue;

            TraceRay local_ray = ray_to_object(world_ray, instance);
            RayProjection projection = ray_projection(local_ray.direction);
            RayCursor local;
            ray_begin<true>(page.mesh_nodes, local, local_ray, instance.root, hit.distance);
            for (;;)
            {
                uint2 span = ray_next_leaf<true>(page.mesh_nodes, local, local_ray, hit.distance);
                if (span.y == 0u)
                    break;
                for (uint primitive = span.x; primitive < span.x + span.y; ++primitive)
                {
                    uint4 face = world_ray_meshes[page.triangles + primitive];
                    float distance;
                    float2 weights;
                    if (!ray_intersect_triangle(page, local_ray, projection, face.xyz, hit.distance, distance, weights))
                        continue;
                    bool earlier = instance.surface_id < hit.surface_id ||
                                   (instance.surface_id == hit.surface_id && face.w < hit.primitive);
                    if (distance < hit.distance || earlier)
                    {
                        RayHit candidate = {distance, weights, index, primitive, face.w, instance.surface_id};
                        // Only an accepted candidate can shorten traversal or block visibility.
                        if (!policy.accepts(page, world_ray, candidate))
                            continue;
                        hit = candidate;
                        if (stop_on_first_hit)
                            return hit;
                    }
                }
            }
        }
    }
}

RayHit trace_closest<Policy : RayHitPolicy>(RayPage page, TraceRay ray, uint layer_mask, Policy policy)
{
    return trace_ray(page, ray, layer_mask, false, policy);
}

bool trace_visibility<Policy : RayHitPolicy>(RayPage page, TraceRay ray, uint layer_mask, Policy policy)
{
    return page.status == ray_page_ready &&
           trace_ray(page, ray, layer_mask, true, policy).triangle_index == ray_invalid_index;
}

struct OpaqueRayHitPolicy : RayHitPolicy
{
    bool accepts(RayPage page, TraceRay ray, RayHit candidate)
    {
        return true;
    }
};

RayHit trace_closest(RayPage page, TraceRay ray, uint layer_mask)
{
    return trace_closest(page, ray, layer_mask, OpaqueRayHitPolicy());
}

bool trace_visibility(RayPage page, TraceRay ray, uint layer_mask)
{
    return trace_visibility(page, ray, layer_mask, OpaqueRayHitPolicy());
}

struct RaySurfaceHit
{
    float3 position_ws;
    float3 geometric_normal_ws; // Triangle winding before any face-forward rule.
    float3 normal_ws;
    float4 tangent_ws;
    float4 color;
    float2 uv0;
    float2 uv1;
    float3 edge_u_ws;
    float3 edge_v_ws;
    float2 edge_u_uv1;
    float2 edge_v_uv1;
    bool front_face;
};

float4 ray_interpolate_attribute(RayPage page, uint3 vertices, uint field, float3 weights)
{
    uint3 offset = page.attributes + 4u * vertices + field;
    return asfloat(world_ray_meshes[offset.x]) * weights.x +
           asfloat(world_ray_meshes[offset.y]) * weights.y +
           asfloat(world_ray_meshes[offset.z]) * weights.z;
}

float3 ray_world_normal(float3 local_normal, RayInstance instance)
{
    return local_normal.x * instance.world_to_object_x.xyz +
           local_normal.y * instance.world_to_object_y.xyz +
           local_normal.z * instance.world_to_object_z.xyz;
}

float3 ray_surface_normalize(float3 v, float3 fallback)
{
    float length_squared = dot(v, v);
    return length_squared > 0.0f ? v * rsqrt(length_squared) : fallback;
}

float3 ray_transform_columns(float3 v, float3 column0, float3 column1, float3 column2)
{
    return v.x * column0 + v.y * column1 + v.z * column2;
}

float3 ray_corner_direction(uint offset, float3 r0, float3 r1, float3 r2, float3 fallback)
{
    float3 v = asfloat(world_ray_meshes[offset].xyz);
    return ray_surface_normalize(ray_transform_columns(v, r0, r1, r2), fallback);
}

float3 ray_interpolate_direction(uint3 offsets, float3 weights, float3 r0, float3 r1, float3 r2, float3 fallback)
{
    float3 a = ray_corner_direction(offsets.x, r0, r1, r2, fallback);
    float3 b = ray_corner_direction(offsets.y, r0, r1, r2, fallback);
    float3 c = ray_corner_direction(offsets.z, r0, r1, r2, fallback);
    return ray_surface_normalize(a * weights.x + b * weights.y + c * weights.z, fallback);
}

// Only called on a valid hit.
RaySurfaceHit ray_surface_at_hit(RayPage page, TraceRay ray, RayHit hit)
{
    RayInstance instance = ray_instance(page, hit.instance_index);
    uint3 vertices = world_ray_meshes[page.triangles + hit.triangle_index].xyz;
    float3 weights = float3(1.0f - hit.barycentrics.x - hit.barycentrics.y, hit.barycentrics);
    float3 a = asfloat(world_ray_meshes[page.positions + vertices.x].xyz);
    float3 b = asfloat(world_ray_meshes[page.positions + vertices.y].xyz);
    float3 c = asfloat(world_ray_meshes[page.positions + vertices.z].xyz);
    float3 geometric = normalize(ray_world_normal(cross(b - a, c - a), instance) * instance.orientation);
    float3 r0 = instance.world_to_object_x.xyz;
    float3 r1 = instance.world_to_object_y.xyz;
    float3 r2 = instance.world_to_object_z.xyz;
    uint3 attributes = page.attributes + 4u * vertices;
    // Raster transforms and normalizes each corner before interpolation.
    float3 normal = ray_interpolate_direction(attributes + 1u, weights, r0, r1, r2, geometric);

    // Cofactors of the inverse transform give the forward tangent direction.
    // Normalization cancels the magnitude of the determinant, but not its sign.
    float3 c0 = cross(r1, r2);
    float3 c1 = cross(r2, r0);
    float3 c2 = cross(r0, r1);
    float3 axis = abs(normal.x) < 0.9f ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
    float3 fallback = normalize(cross(normal, axis));
    float3 tangent_ws = ray_interpolate_direction(attributes + 2u, weights,
        c0 * instance.orientation, c1 * instance.orientation, c2 * instance.orientation, fallback);
    uint3 tangent_offsets = attributes + 2u;
    float3 signs = float3(asfloat(world_ray_meshes[tangent_offsets.x].w),
                          asfloat(world_ray_meshes[tangent_offsets.y].w),
                          asfloat(world_ray_meshes[tangent_offsets.z].w));
    float sign = dot(weights, signs) * instance.orientation;
    float4 uv_a = asfloat(world_ray_meshes[attributes.x]);
    float4 uv_b = asfloat(world_ray_meshes[attributes.y]);
    float4 uv_c = asfloat(world_ray_meshes[attributes.z]);
    float4 uv = uv_a * weights.x + uv_b * weights.y + uv_c * weights.z;
    float inverse_det = 1.0f / dot(r0, c0);

    RaySurfaceHit surface;
    surface.position_ws = ray.origin + hit.distance * ray.direction;
    surface.geometric_normal_ws = geometric;
    surface.normal_ws = normal;
    surface.tangent_ws = float4(tangent_ws, sign);
    surface.color = ray_interpolate_attribute(page, vertices, 3u, weights);
    surface.uv0 = uv.xy;
    surface.uv1 = uv.zw;
    surface.edge_u_ws = ray_transform_columns(b - a, c0, c1, c2) * inverse_det;
    surface.edge_v_ws = ray_transform_columns(c - a, c0, c1, c2) * inverse_det;
    surface.edge_u_uv1 = uv_b.zw - uv_a.zw;
    surface.edge_v_uv1 = uv_c.zw - uv_a.zw;
    surface.front_face = dot(ray.direction, geometric) < 0.0f;
    return surface;
}
