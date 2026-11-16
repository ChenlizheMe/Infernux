// Storage layout follows RayBvh.h and RayGeometry.h on both graphics backends.
static const uint ray_leaf_bit = 0x80000000u;
static const uint ray_invalid_index = 0xffffffffu;
static const uint ray_stack_capacity = 48u;

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
    uint reserved;
};

StructuredBuffer<RayNode> ray_nodes;
StructuredBuffer<float4> ray_positions;
StructuredBuffer<uint4> ray_triangles;
StructuredBuffer<RayInstance> ray_instances;

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
    uint triangle_index;
    uint primitive;
    uint surface_id;
};

struct RayCursor
{
    uint2 pending[ray_stack_capacity]; // Node index and its conservative entry distance.
    uint count;
    uint2 current;
};

bool ray_box_entry(TraceRay ray, RayNode node, float maximum, out float entry)
{
    entry = ray.minimum;
    float exit_distance = maximum;
    for (uint axis = 0u; axis < 3u; ++axis)
    {
        float origin = ray.origin[axis];
        float direction = ray.direction[axis];
        if (direction == 0.0f)
        {
            if (origin < node.minimum[axis] || origin > node.maximum[axis])
                return false;
        }
        else
        {
            float a = (node.minimum[axis] - origin) / direction;
            float b = (node.maximum[axis] - origin) / direction;
            float low = max(min(a, b), ray.minimum);
            float high = min(max(a, b), maximum);
            if (low > maximum || high < ray.minimum)
                return false;
            const float rounding = 3.57628e-7f;
            entry = max(entry, low - abs(low) * rounding);
            exit_distance = min(exit_distance, high + abs(high) * rounding);
            if (entry > exit_distance)
                return false;
        }
    }
    return true;
}

void ray_begin(out RayCursor cursor, TraceRay ray, uint root, float maximum)
{
    cursor.count = 0u;
    float entry = ray.minimum;
    cursor.current = uint2(ray_invalid_index, asuint(entry));
    if (root != ray_invalid_index && ray_box_entry(ray, ray_nodes[root], maximum, entry))
        cursor.current = uint2(root, asuint(entry));
}

uint2 ray_continue(inout RayCursor cursor)
{
    if (cursor.count == 0u)
        return uint2(ray_invalid_index, 0u);
    cursor.count -= 1u;
    return cursor.pending[cursor.count];
}

uint2 ray_next_leaf(inout RayCursor cursor, TraceRay ray, float maximum)
{
    while (cursor.current.x != ray_invalid_index)
    {
        if (asfloat(cursor.current.y) <= maximum)
        {
            RayNode node = ray_nodes[cursor.current.x];
            if ((node.second & ray_leaf_bit) != 0u)
            {
                cursor.current = ray_continue(cursor);
                return uint2(node.first, node.second & ~ray_leaf_bit);
            }

            float left_entry, right_entry;
            bool left = ray_box_entry(ray, ray_nodes[node.first], maximum, left_entry);
            bool right = ray_box_entry(ray, ray_nodes[node.second], maximum, right_entry);
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

bool ray_intersect_triangle(TraceRay ray, RayProjection projection, uint3 corners, float maximum,
                            out float distance, out float2 weights)
{
    float3 a = ray_project_point(ray_positions[corners.x].xyz - ray.origin, projection);
    float3 b = ray_project_point(ray_positions[corners.y].xyz - ray.origin, projection);
    float3 c = ray_project_point(ray_positions[corners.z].xyz - ray.origin, projection);
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

RayHit trace_ray(TraceRay world_ray, uint root, uint layer_mask, bool stop_on_first_hit)
{
    RayHit hit;
    hit.distance = world_ray.maximum;
    hit.barycentrics = float2(0.0f);
    hit.triangle_index = ray_invalid_index;
    hit.primitive = ray_invalid_index;
    hit.surface_id = ray_invalid_index;

    RayCursor world;
    ray_begin(world, world_ray, root, hit.distance);
    for (;;)
    {
        uint2 leaf = ray_next_leaf(world, world_ray, hit.distance);
        if (leaf.y == 0u)
            return hit;
        for (uint index = leaf.x; index < leaf.x + leaf.y; ++index)
        {
            RayInstance instance = ray_instances[index];
            if ((instance.layer_mask & layer_mask) == 0u)
                continue;

            TraceRay local_ray = ray_to_object(world_ray, instance);
            RayProjection projection = ray_projection(local_ray.direction);
            RayCursor local;
            ray_begin(local, local_ray, instance.root, hit.distance);
            for (;;)
            {
                uint2 span = ray_next_leaf(local, local_ray, hit.distance);
                if (span.y == 0u)
                    break;
                for (uint primitive = span.x; primitive < span.x + span.y; ++primitive)
                {
                    uint4 face = ray_triangles[primitive];
                    float distance;
                    float2 weights;
                    if (!ray_intersect_triangle(local_ray, projection, face.xyz, hit.distance, distance, weights))
                        continue;
                    bool earlier = instance.surface_id < hit.surface_id ||
                                   (instance.surface_id == hit.surface_id && face.w < hit.primitive);
                    if (distance < hit.distance || earlier)
                    {
                        hit.distance = distance;
                        hit.barycentrics = weights;
                        hit.triangle_index = primitive;
                        hit.primitive = face.w;
                        hit.surface_id = instance.surface_id;
                        if (stop_on_first_hit)
                            return hit;
                    }
                }
            }
        }
    }
}

RayHit trace_closest(TraceRay ray, uint root, uint layer_mask)
{
    return trace_ray(ray, root, layer_mask, false);
}

bool trace_visibility(TraceRay ray, uint root, uint layer_mask)
{
    return trace_ray(ray, root, layer_mask, true).triangle_index == ray_invalid_index;
}
