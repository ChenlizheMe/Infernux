import inx_ray_layout;

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
    uint vertex_base;
};

RayPage ray_page()
{
    uint4 geometry = world_ray_geometry[0];
    uint4 meshes = world_ray_meshes[0];
    return {geometry.x, geometry.y, geometry.z, geometry.w, meshes.x, meshes.y, meshes.z, meshes.w, 0u};
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

struct RayInstanceTransform
{
    float4 x;
    float4 y;
    float4 z;
};

RayNode ray_node<let MeshNodes : bool, let Dynamic : bool>(uint nodes, uint index)
{
    uint offset = nodes + 2u * index;
    uint4 low, high;
    if (MeshNodes && !Dynamic)
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
    uint offset = page.instances + ray_instance_words * index;
    uint4 record = world_ray_geometry[offset + 3u];
    return {asfloat(world_ray_geometry[offset]), asfloat(world_ray_geometry[offset + 1u]),
            asfloat(world_ray_geometry[offset + 2u]), record.x, record.y, record.z, asfloat(record.w)};
}

RayInstanceTransform ray_instance_transform(RayPage page, uint index)
{
    uint offset = page.instances + ray_instance_words * index;
    return {asfloat(world_ray_geometry[offset + 4u]), asfloat(world_ray_geometry[offset + 5u]),
            asfloat(world_ray_geometry[offset + 6u])};
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

// Precompute each axis once in the instance's coordinate system. Keep exactly
// parallel axes explicit so rays in a slab face work the same on WGSL and SPIR-V.
struct RayBoundsQuery
{
    float3 origin;
    float minimum;
    float3 inverse_direction;
    bool3 parallel;
};

RayBoundsQuery ray_bounds_query(TraceRay ray)
{
    // Components below the binary32 normal range cannot reliably be inverted
    // on all targets. Open those slabs conservatively and test the actual triangle.
    bool3 invertible = abs(ray.direction) >= asfloat(0x00800000u);
    float3 inverse = 1.0f / select(invertible, ray.direction, float3(1.0f));
    return {ray.origin, ray.minimum, select(invertible, inverse, float3(0.0f)), ray.direction == float3(0.0f)};
}

bool ray_box_entry(RayBoundsQuery ray, RayNode node, float maximum, out float entry)
{
    const float rounding = 3.57628e-7f;
    float3 a = (node.minimum - ray.origin) * ray.inverse_direction;
    float3 b = (node.maximum - ray.origin) * ray.inverse_direction;
    bool3 open_slab = ray.inverse_direction == float3(0.0f);
    float3 low = select(open_slab, float3(ray.minimum), min(a, b));
    float3 high = select(open_slab, float3(maximum), max(a, b));
    float near_distance = max(ray.minimum, max(low.x, max(low.y, low.z)));
    float far_distance = min(maximum, min(high.x, min(high.y, high.z)));
    entry = max(ray.minimum, near_distance * (near_distance < 0.0f ? 1.0f + rounding : 1.0f - rounding));
    bool3 outside = (ray.origin < node.minimum) | (ray.origin > node.maximum);
    float exit = far_distance * (far_distance < 0.0f ? 1.0f - rounding : 1.0f + rounding);
    return !any(ray.parallel & outside) && entry <= min(maximum, exit);
}

void ray_begin<let MeshNodes : bool, let Dynamic : bool>(uint nodes, out RayCursor cursor, RayBoundsQuery ray,
                                                        uint root, float maximum)
{
    cursor.count = 0u;
    float entry = ray.minimum;
    cursor.current = uint2(ray_invalid_index, asuint(entry));
    if (root != ray_invalid_index && ray_box_entry(ray, ray_node<MeshNodes, Dynamic>(nodes, root), maximum, entry))
        cursor.current = uint2(root, asuint(entry));
}

uint2 ray_continue(inout RayCursor cursor)
{
    if (cursor.count == 0u)
        return uint2(ray_invalid_index, 0u);
    cursor.count -= 1u;
    return cursor.pending[cursor.count];
}

uint2 ray_next_leaf<let MeshNodes : bool, let Dynamic : bool>(uint nodes, inout RayCursor cursor, RayBoundsQuery ray,
                                                             float maximum)
{
    while (cursor.current.x != ray_invalid_index)
    {
        if (asfloat(cursor.current.y) <= maximum)
        {
            RayNode node = ray_node<MeshNodes, Dynamic>(nodes, cursor.current.x);
            if ((node.second & ray_leaf_bit) != 0u)
            {
                cursor.current = ray_continue(cursor);
                return uint2(node.first, node.second & ~ray_leaf_bit);
            }

            float left_entry, right_entry;
            bool left = ray_box_entry(ray, ray_node<MeshNodes, Dynamic>(nodes, node.first), maximum, left_entry);
            bool right = ray_box_entry(ray, ray_node<MeshNodes, Dynamic>(nodes, node.second), maximum, right_entry);
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

// Forward-error factor for binary32 arithmetic. Bounds also cover a contracted multiply/add.
float ray_gamma(float operations)
{
    float accumulated = operations * 5.9604644775390625e-8f;
    return accumulated / (1.0f - accumulated);
}

uint4 ray_mesh_word<let Dynamic : bool>(uint offset)
{
    if (Dynamic)
        return world_ray_geometry[offset];
    else
        return world_ray_meshes[offset];
}

bool ray_intersect_triangle<let Dynamic : bool>(RayPage page, TraceRay ray, RayProjection projection, uint3 corners, float maximum,
                            out float distance, out float2 weights)
{
    float3 vertex_a = asfloat(ray_mesh_word<Dynamic>(page.positions + corners.x - page.vertex_base).xyz);
    float3 vertex_b = asfloat(ray_mesh_word<Dynamic>(page.positions + corners.y - page.vertex_base).xyz);
    float3 vertex_c = asfloat(ray_mesh_word<Dynamic>(page.positions + corners.z - page.vertex_base).xyz);
    float3 local_a = vertex_a - ray.origin;
    float3 a = ray_project_point(local_a, projection);
    float3 b = ray_project_point(vertex_b - ray.origin, projection);
    float3 c = ray_project_point(vertex_c - ray.origin, projection);
    float3 edge = float3(ray_edge(b.xy, c.xy), ray_edge(c.xy, a.xy), ray_edge(a.xy, b.xy));
    float det = edge.x + edge.y + edge.z;
    distance = maximum;
    weights = float2(0.0f);
    if ((any(edge < 0.0f) && any(edge > 0.0f)) || det == 0.0f)
        return false;
    weights = edge.yz / det;

    // The coverage test remains watertight. Compute t from the original plane,
    // avoiding cancellation of large sheared vertex depths on long triangles.
    float3 e1 = vertex_b - vertex_a;
    float3 e2 = vertex_c - vertex_a;
    float3 plane = cross(e1, e2);
    float3 plane_error = ray_gamma(5.0f) * (abs(e1.yzx * e2.zxy) + abs(e1.zxy * e2.yzx));
    float numerator = dot(plane, local_a);
    float denominator = dot(plane, ray.direction);
    float numerator_error = ray_gamma(5.0f) * dot(abs(plane), abs(local_a)) +
                              (1.0f + ray_gamma(3.0f)) * dot(plane_error, abs(local_a));
    float denominator_error = ray_gamma(4.0f) * dot(abs(plane), abs(ray.direction)) +
                                (1.0f + ray_gamma(3.0f)) * dot(plane_error, abs(ray.direction));
    distance = numerator / denominator;
    return distance > 0.0f && distance >= ray.minimum && distance <= maximum &&
           abs(numerator) > numerator_error && abs(denominator) > denominator_error;
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

RayPage ray_deformed_page(RayPage page, uint tagged_root)
{
    uint4 layout = world_ray_geometry[tagged_root & 0x7fffffffu];
    page.mesh_nodes = layout.x;
    page.positions = layout.y;
    page.attributes = layout.z;
    page.vertex_base = layout.w;
    return page;
}

void trace_instance<let Dynamic : bool, Policy : RayHitPolicy>(
    RayPage page, TraceRay world_ray, uint instance_index, RayInstance instance,
    bool stop_on_first_hit, Policy policy, inout RayHit hit)
{
    if (Dynamic)
        page = ray_deformed_page(page, instance.root);
    TraceRay local_ray = ray_to_object(world_ray, instance);
    RayProjection projection = ray_projection(local_ray.direction);
    RayCursor local;
    RayBoundsQuery local_bounds = ray_bounds_query(local_ray);
    ray_begin<true, Dynamic>(page.mesh_nodes, local, local_bounds, Dynamic ? 0u : instance.root, hit.distance);
    for (;;)
    {
        uint2 span = ray_next_leaf<true, Dynamic>(page.mesh_nodes, local, local_bounds, hit.distance);
        if (span.y == 0u)
            return;
        for (uint primitive = span.x; primitive < span.x + span.y; ++primitive)
        {
            uint4 face = world_ray_meshes[page.triangles + primitive];
            float distance;
            float2 weights;
            if (!ray_intersect_triangle<Dynamic>(page, local_ray, projection, face.xyz, hit.distance, distance, weights))
                continue;
            bool earlier = instance.surface_id < hit.surface_id ||
                           (instance.surface_id == hit.surface_id && face.w < hit.primitive);
            if (distance < hit.distance || earlier)
            {
                RayHit candidate = {distance, weights, instance_index, primitive, face.w, instance.surface_id};
                if (!policy.accepts(page, world_ray, candidate))
                    continue;
                hit = candidate;
                if (stop_on_first_hit)
                    return;
            }
        }
    }
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
    RayBoundsQuery world_bounds = ray_bounds_query(world_ray);
    ray_begin<false, false>(page.top_nodes, world, world_bounds, page.root, hit.distance);
    for (;;)
    {
        uint2 leaf = ray_next_leaf<false, false>(page.top_nodes, world, world_bounds, hit.distance);
        if (leaf.y == 0u)
            return hit;
        for (uint index = leaf.x; index < leaf.x + leaf.y; ++index)
        {
            RayInstance instance = ray_instance(page, index);
            if ((instance.layer_mask & layer_mask) == 0u)
                continue;

            if ((instance.root & 0x80000000u) != 0u)
                trace_instance<true>(page, world_ray, index, instance, stop_on_first_hit, policy, hit);
            else
                trace_instance<false>(page, world_ray, index, instance, stop_on_first_hit, policy, hit);
            if (stop_on_first_hit && hit.triangle_index != ray_invalid_index)
                return hit;
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
    float3 position_error_ws;
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

float4 ray_interpolate_attribute<let Dynamic : bool>(RayPage page, uint3 vertices, uint field, float3 weights)
{
    uint3 offset = page.attributes + 4u * (vertices - page.vertex_base) + field;
    return asfloat(ray_mesh_word<Dynamic>(offset.x)) * weights.x +
           asfloat(ray_mesh_word<Dynamic>(offset.y)) * weights.y +
           asfloat(ray_mesh_word<Dynamic>(offset.z)) * weights.z;
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

float3 ray_corner_direction<let Dynamic : bool>(uint offset, float3 r0, float3 r1, float3 r2, float3 fallback)
{
    float3 v = asfloat(ray_mesh_word<Dynamic>(offset).xyz);
    return ray_surface_normalize(ray_transform_columns(v, r0, r1, r2), fallback);
}

float3 ray_interpolate_direction<let Dynamic : bool>(uint3 offsets, float3 weights, float3 r0, float3 r1, float3 r2, float3 fallback)
{
    float3 a = ray_corner_direction<Dynamic>(offsets.x, r0, r1, r2, fallback);
    float3 b = ray_corner_direction<Dynamic>(offsets.y, r0, r1, r2, fallback);
    float3 c = ray_corner_direction<Dynamic>(offsets.z, r0, r1, r2, fallback);
    return ray_surface_normalize(a * weights.x + b * weights.y + c * weights.z, fallback);
}

// Only called on a valid hit.
RaySurfaceHit ray_surface_at_hit_impl<let Dynamic : bool>(RayPage page, TraceRay ray, RayHit hit)
{
    RayInstance instance = ray_instance(page, hit.instance_index);
    if (Dynamic)
        page = ray_deformed_page(page, instance.root);
    uint3 vertices = world_ray_meshes[page.triangles + hit.triangle_index].xyz;
    float3 weights = float3(1.0f - hit.barycentrics.x - hit.barycentrics.y, hit.barycentrics);
    float3 a = asfloat(ray_mesh_word<Dynamic>(page.positions + vertices.x - page.vertex_base).xyz);
    float3 b = asfloat(ray_mesh_word<Dynamic>(page.positions + vertices.y - page.vertex_base).xyz);
    float3 c = asfloat(ray_mesh_word<Dynamic>(page.positions + vertices.z - page.vertex_base).xyz);
    float3 local_geometric = cross(b - a, c - a);
    float3 world_geometric = ray_world_normal(local_geometric, instance);
    float inverse_normal_length = rsqrt(dot(world_geometric, world_geometric));
    float3 geometric = world_geometric * (inverse_normal_length * instance.orientation);
    float3 r0 = instance.world_to_object_x.xyz;
    float3 r1 = instance.world_to_object_y.xyz;
    float3 r2 = instance.world_to_object_z.xyz;
    uint3 attributes = page.attributes + 4u * (vertices - page.vertex_base);
    // Raster transforms and normalizes each corner before interpolation.
    float3 normal = ray_interpolate_direction<Dynamic>(attributes + 1u, weights, r0, r1, r2, geometric);

    // Use the original affine rows, rather than numerically inverting the inverse again.
    RayInstanceTransform model = ray_instance_transform(page, hit.instance_index);
    float4 forward_x = model.x;
    float4 forward_y = model.y;
    float4 forward_z = model.z;
    float3 c0 = float3(forward_x.x, forward_y.x, forward_z.x);
    float3 c1 = float3(forward_x.y, forward_y.y, forward_z.y);
    float3 c2 = float3(forward_x.z, forward_y.z, forward_z.z);
    float3 axis = abs(normal.x) < 0.9f ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
    float3 fallback = normalize(cross(normal, axis));
    float3 tangent_ws = ray_interpolate_direction<Dynamic>(attributes + 2u, weights,
        c0, c1, c2, fallback);
    uint3 tangent_offsets = attributes + 2u;
    float3 signs = float3(asfloat(ray_mesh_word<Dynamic>(tangent_offsets.x).w),
                          asfloat(ray_mesh_word<Dynamic>(tangent_offsets.y).w),
                          asfloat(ray_mesh_word<Dynamic>(tangent_offsets.z).w));
    float sign = dot(weights, signs) * instance.orientation;
    float4 uv_a = asfloat(ray_mesh_word<Dynamic>(attributes.x));
    float4 uv_b = asfloat(ray_mesh_word<Dynamic>(attributes.y));
    float4 uv_c = asfloat(ray_mesh_word<Dynamic>(attributes.z));
    float4 uv = uv_a * weights.x + uv_b * weights.y + uv_c * weights.z;
    RaySurfaceHit surface;
    float3 local_position = a + hit.barycentrics.x * (b - a) + hit.barycentrics.y * (c - a);
    float3 edge_u = b - a;
    float3 edge_v = c - a;
    float3 local_error = ray_gamma(5.0f) * (abs(a) + abs(hit.barycentrics.x * edge_u) +
                                            abs(hit.barycentrics.y * edge_v)) +
                         ray_gamma(2.0f) * (abs(hit.barycentrics.x) * (abs(a) + abs(b)) +
                                            abs(hit.barycentrics.y) * (abs(a) + abs(c)));
    float4 homogeneous = float4(local_position, 1.0f);
    surface.position_ws = float3(dot(forward_x, homogeneous), dot(forward_y, homogeneous), dot(forward_z, homogeneous));
    float3 propagated = abs(c0) * local_error.x + abs(c1) * local_error.y + abs(c2) * local_error.z;
    float3 forward_magnitude = abs(c0) * abs(local_position.x) + abs(c1) * abs(local_position.y) +
                                abs(c2) * abs(local_position.z) + abs(float3(forward_x.w, forward_y.w, forward_z.w));
    float3 world_error = (1.0f + ray_gamma(5.0f)) * propagated + ray_gamma(5.0f) * forward_magnitude;
    // The stored forward/inverse matrices are independently rounded binary32
    // values. Their actual round trip and its arithmetic bound cover that
    // mismatch before a secondary ray is transformed back to object space.
    float4 world_position = float4(surface.position_ws, 1.0f);
    float3 round_trip = float3(dot(instance.world_to_object_x, world_position),
                                dot(instance.world_to_object_y, world_position),
                                dot(instance.world_to_object_z, world_position));
    float3 inverse_magnitude = float3(dot(abs(instance.world_to_object_x), abs(world_position)),
                                       dot(abs(instance.world_to_object_y), abs(world_position)),
                                       dot(abs(instance.world_to_object_z), abs(world_position)));
    float3 inverse_error = (1.0f + ray_gamma(1.0f)) * abs(round_trip - local_position) +
                            ray_gamma(5.0f) * inverse_magnitude;
    float inverse_plane_error = dot(abs(local_geometric), inverse_error) * inverse_normal_length;
    surface.position_error_ws = world_error + abs(geometric) * inverse_plane_error;
    surface.geometric_normal_ws = geometric;
    surface.normal_ws = normal;
    surface.tangent_ws = float4(tangent_ws, sign);
    surface.color = ray_interpolate_attribute<Dynamic>(page, vertices, 3u, weights);
    surface.uv0 = uv.xy;
    surface.uv1 = uv.zw;
    surface.edge_u_ws = ray_transform_columns(b - a, c0, c1, c2);
    surface.edge_v_ws = ray_transform_columns(c - a, c0, c1, c2);
    surface.edge_u_uv1 = uv_b.zw - uv_a.zw;
    surface.edge_v_uv1 = uv_c.zw - uv_a.zw;
    surface.front_face = dot(ray.direction, geometric) < 0.0f;
    return surface;
}

RaySurfaceHit ray_surface_at_hit(RayPage page, TraceRay ray, RayHit hit)
{
    if ((ray_instance(page, hit.instance_index).root & 0x80000000u) != 0u)
        return ray_surface_at_hit_impl<true>(page, ray, hit);
    return ray_surface_at_hit_impl<false>(page, ray, hit);
}
