// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#include "rigid_math.slang"
#include "rigid_convex.slang"

ConstantBuffer<QueryParams> query_params;
StructuredBuffer<QueryRecordWord> query_inputs;
// Compact selected-world identities. Model/shape slots and hit addresses stay
// global; only this identity port subtracts the retained world's body/leaf bases.
StructuredBuffer<QueryBodyIdentity> query_body_ids;
StructuredBuffer<BodyModel> body_models;
StructuredBuffer<ShapeModel> shape_models;
StructuredBuffer<BodyState> body_states;
RWStructuredBuffer<QueryRecordWord> query_candidates;
RWStructuredBuffer<uint4> query_control;
StructuredBuffer<uint4> query_geometry;
RWStructuredBuffer<QuerySummary> query_summary;
RWStructuredBuffer<QueryRecordWord> query_hits;
ConstantBuffer<QueryCastParams> query_cast_params;

// One immutable request stream: Input144 records precede Primitive80 suffix. Identity records likewise share one resource.
QueryInput query_load_input(uint index)
{
    uint base = index * 9u;
    QueryInput request;
    request.meta = query_inputs[base].value;
    request.filter = query_inputs[base + 1u].value;
    request.source = query_inputs[base + 2u].value;
    request.positionMaximum = asfloat(query_inputs[base + 3u].value);
    request.direction = asfloat(query_inputs[base + 4u].value);
    request.rotation = asfloat(query_inputs[base + 5u].value);
    request.geometry = asfloat(query_inputs[base + 6u].value);
    request.lifetime = query_inputs[base + 7u].value;
    request.geometryReference = query_inputs[base + 8u].value;
    return request;
}

QueryPrimitive query_load_primitive(uint index)
{
    uint base = query_params.counts.x * 9u + index * 5u;
    QueryPrimitive primitive;
    primitive.meta = query_inputs[base].value;
    primitive.geometry = asfloat(query_inputs[base + 1u].value);
    primitive.position = asfloat(query_inputs[base + 2u].value);
    primitive.rotation = asfloat(query_inputs[base + 3u].value);
    primitive.geometryReference = query_inputs[base + 4u].value;
    return primitive;
}

// The first request-count UInt4 words retain the public QuerySpan prefix.
// Four CA words per request follow it, independently of candidate record width.
QuerySpan query_load_span(uint index)
{
    QuerySpan span;
    span.range = query_control[index];
    return span;
}

void query_store_span(uint index, QuerySpan span)
{
    query_control[index] = span.range;
}

uint query_cast_state_word(uint index)
{
    return query_params.counts.x + index * 4u;
}

QueryCastState query_load_cast_state(uint index)
{
    uint base = query_cast_state_word(index);
    QueryCastState state;
    state.cursor = query_control[base];
    state.progress = asfloat(query_control[base + 1u]);
    state.normal = asfloat(query_control[base + 2u]);
    state.iteration = query_control[base + 3u];
    return state;
}

void query_store_cast_state(uint index, QueryCastState state)
{
    uint base = query_cast_state_word(index);
    query_control[base] = state.cursor;
    query_control[base + 1u] = asuint(state.progress);
    query_control[base + 2u] = asuint(state.normal);
    query_control[base + 3u] = state.iteration;
}

QueryHit query_load_hit(uint record)
{
    uint base = record * 4u;
    QueryHit hit;
    hit.meta = query_candidates[base].value;
    hit.positionDistance = asfloat(query_candidates[base + 1u].value);
    hit.normalFraction = asfloat(query_candidates[base + 2u].value);
    hit.pointADepth = asfloat(query_candidates[base + 3u].value);
    return hit;
}

void query_store_hit(uint record, QueryHit hit)
{
    uint base = record * 4u;
    query_candidates[base].value = hit.meta;
    query_candidates[base + 1u].value = asuint(hit.positionDistance);
    query_candidates[base + 2u].value = asuint(hit.normalFraction);
    query_candidates[base + 3u].value = asuint(hit.pointADepth);
}

void query_clear_candidate(uint record)
{
    const uint words = 4u;
    for (uint word = 0u; word < words; ++word)
        query_candidates[record * words + word].value = word == 0u ? uint4(InvalidQueryBodySlot, 0u, InvalidQueryBodySlot, 0u) : uint4(0u);
}

float3 query_box_face_normal(float3 point, float3 half_extent)
{
    // BoxShape::GetSurfaceNormal: nearest outer face, lowest axis on ties.
    float3 distance = abs(abs(point) - half_extent);
    uint axis = distance.y < distance.x ? 1u : 0u;
    if (distance.z < distance[axis])
        axis = 2u;
    float3 normal = float3(0.0f);
    normal[axis] = point[axis] > 0.0f ? 1.0f : -1.0f;
    return normal;
}

float query_entering_distance(float b, float a, float root, float c, float scale)
{
    // The same quadratic has two conditioned forms. The projection minus
    // chord preserves distant hits; c/q preserves a small near-surface gap.
    return root <= -0.5f * b ? (-b / a) * scale - (root / a) * scale
                             : (c / (-b + root)) * scale;
}

float query_squared_radius_difference(float3 point, float radius, float scale)
{
    uint axis = abs(point.y) > abs(point.x) ? 1u : 0u;
    if (abs(point.z) > abs(point[axis]))
        axis = 2u;
    float coordinate = abs(point[axis]);
    float3 other = point / scale;
    other[axis] = 0.0f;
    return ((coordinate - radius) / scale) * (coordinate / scale + radius / scale) + dot(other, other);
}

bool query_sphere(float3 origin, float3 direction, float radius, float maximum,
                  out float distance, out float3 normal)
{
    distance = 0.0f;
    normal = float3(0.0f, 1.0f, 0.0f);
    // Scaling the quadratic avoids overflow when finite origins are large.
    float scale = max(max(abs(origin.x), max(abs(origin.y), abs(origin.z))), radius);
    float3 scaled = origin / scale;
    float scaled_radius = radius / scale;
    float c = query_squared_radius_difference(origin, radius, scale);
    if (c > 0.0f)
    {
        float b = dot(scaled, direction);
        float a = dot(direction, direction);
        float3 perpendicular = scaled - direction * (b / a);
        float discriminant = a * (scaled_radius * scaled_radius - dot(perpendicular, perpendicular));
        if (b >= 0.0f || discriminant < 0.0f)
            return false;
        float root = sqrt(discriminant);
        distance = query_entering_distance(b, a, root, c, scale);
        if (distance > maximum)
            return false;
        // Use the small entry residual, not origin + distance*direction:
        // distant origins can erase the entire radius in that subtraction.
        float3 entry = perpendicular - direction * (root / a);
        float normal_scale = max(abs(entry.x), max(abs(entry.y), abs(entry.z)));
        normal = normal_scale > 0.0f ? normalize(entry / normal_scale) : -normalize(direction);
        return true;
    }
    float3 point = origin + direction * distance;
    float point_scale = max(abs(point.x), max(abs(point.y), abs(point.z)));
    if (point_scale > 0.0f)
        normal = normalize(point / point_scale);
    return true;
}

bool query_capsule(float3 origin, float3 direction, float half_height, float radius, float maximum,
                   out float distance, out float3 normal)
{
    float3 core = float3(0.0f, clamp(origin.y, -half_height, half_height), 0.0f);
    float3 radial = origin - core;
    float radial_scale = max(max(abs(radial.x), max(abs(radial.y), abs(radial.z))), radius);
    distance = 0.0f;
    normal = float3(1.0f, 0.0f, 0.0f);
    if (query_squared_radius_difference(radial, radius, radial_scale) <= 0.0f)
    {
        float normal_scale = max(abs(radial.x), max(abs(radial.y), abs(radial.z)));
        if (normal_scale > 0.0f)
            normal = normalize(radial / normal_scale);
        return true;
    }

    bool found = false;
    float closest = maximum;
    // A capsule is the union of its finite cylinder and endpoint spheres.
    // Starting outside that union, its first entering component is the true
    // surface; internal sphere hemispheres cannot become an earlier hit.
    for (uint end = 0u; end < 2u; ++end)
    {
        float center_y = end == 0u ? -half_height : half_height;
        float candidate;
        float3 candidate_normal;
        if (query_sphere(origin - float3(0.0f, center_y, 0.0f), direction, radius,
                         closest, candidate, candidate_normal))
        {
            closest = candidate;
            normal = candidate_normal;
            found = true;
        }
    }

    float2 planar_origin = origin.xz;
    float2 planar_direction = direction.xz;
    float velocity_scale = max(abs(planar_direction.x), abs(planar_direction.y));
    if (velocity_scale > 0.0f)
    {
        // Normalize direction and radial coordinates separately so a nearly
        // axial ray does not disappear when its squared planar speed underflows.
        float position_scale = max(max(abs(planar_origin.x), abs(planar_origin.y)), radius);
        float2 offset = planar_origin / position_scale;
        float2 velocity = planar_direction / velocity_scale;
        float r = radius / position_scale;
        float a = dot(velocity, velocity);
        float b = dot(offset, velocity);
        uint axis = abs(planar_origin.y) > abs(planar_origin.x) ? 1u : 0u;
        float coordinate = abs(planar_origin[axis]);
        float other = offset[1u - axis];
        float c = ((coordinate - radius) / position_scale) * (coordinate / position_scale + r) + other * other;
        if (c > 0.0f && b < 0.0f)
        {
            float2 perpendicular = offset - velocity * (b / a);
            float discriminant = a * (r * r - dot(perpendicular, perpendicular));
            if (discriminant >= 0.0f)
            {
                float root = sqrt(discriminant);
                float candidate = query_entering_distance(b, a, root, c, position_scale) / velocity_scale;
                float y = origin.y + direction.y * candidate;
                if (candidate <= closest && y >= -half_height && y <= half_height)
                {
                    float2 surface = perpendicular - velocity * (root / a);
                    float surface_scale = max(abs(surface.x), abs(surface.y));
                    float2 outward = surface_scale > 0.0f ? surface / surface_scale : -velocity;
                    normal = normalize(float3(outward.x, 0.0f, outward.y));
                    closest = candidate;
                    found = true;
                }
            }
        }
    }
    distance = closest;
    return found;
}

bool query_rounded_box(float3 origin, float3 direction, float4 geometry, float maximum,
                       out float distance, out float3 normal)
{
    // Keep finite author coordinates out of squared-distance overflow. A ray's
    // entry is computed in these scaled units; only the final distance leaves
    // them. A tiny shape and large far plane may overflow the division, but
    // no entering surface requires a normalized endpoint beyond float range.
    float scale = max(max(abs(origin.x), max(abs(origin.y), abs(origin.z))),
                      max(geometry.x, max(geometry.y, geometry.z)));
    origin /= scale;
    geometry /= scale;
    maximum = min(maximum / scale, 3.402823466e+38f);
    distance = 0.0f;
    normal = query_box_face_normal(origin, geometry.xyz);
    float radius = geometry.w;
    float3 core = max(geometry.xyz - radius, float3(0.0f));
    float3 delta = origin - clamp(origin, -core, core);
    float delta_scale = max(max(abs(delta.x), max(abs(delta.y), abs(delta.z))), radius);
    if (delta_scale == 0.0f || dot(delta / delta_scale, delta / delta_scale) <= (radius / delta_scale) * (radius / delta_scale))
        return true; // Solid convex volume, including its boundary.

    // Squared distance to an axis-aligned core is piecewise quadratic. Its
    // active outside axes change only at these six core-plane crossings.
    // Solve at most seven intervals analytically: no ray marching/tolerance,
    // and cylindrical edges/spherical corners retain the convex radius.
    float points[8];
    points[0] = 0.0f;
    points[1] = maximum;
    uint count = 2u;
    for (uint axis = 0u; axis < 3u; ++axis)
    {
        if (direction[axis] == 0.0f)
            continue;
        for (uint side = 0u; side < 2u; ++side)
        {
            float plane = side == 0u ? -core[axis] : core[axis];
            float t = (plane - origin[axis]) / direction[axis];
            if (t > 0.0f && t < maximum)
                points[count++] = t;
        }
    }
    for (uint i = 1u; i < count; ++i)
    {
        float value = points[i];
        uint j = i;
        for (uint move = 0u; move < 7u && j > 0u; ++move)
        {
            if (points[j - 1u] <= value)
                break;
            points[j] = points[j - 1u];
            --j;
        }
        points[j] = value;
    }
    for (uint interval = 0u; interval + 1u < count; ++interval)
    {
        float start = points[interval];
        float width = points[interval + 1u] - start;
        if (width <= 0.0f)
            continue;
        float3 at_start = origin + start * direction;
        float3 middle = origin + (start + width * 0.5f) * direction;
        float3 offset = float3(0.0f);
        float3 velocity = float3(0.0f);
        for (uint axis = 0u; axis < 3u; ++axis)
        {
            if (middle[axis] < -core[axis] || middle[axis] > core[axis])
            {
                offset[axis] = at_start[axis] - (middle[axis] < 0.0f ? -core[axis] : core[axis]);
                velocity[axis] = direction[axis];
            }
        }
        float root = 0.0f;
        // Normalize this interval again: a large aspect ratio can otherwise
        // underflow a small rounded edge after the outer coordinate scaling.
        float offset_scale = max(max(abs(offset.x), max(abs(offset.y), abs(offset.z))), radius);
        if (offset_scale > 0.0f)
        {
            float3 scaled_offset = offset / offset_scale;
            float scaled_radius = radius / offset_scale;
            float c = dot(scaled_offset, scaled_offset) - scaled_radius * scaled_radius;
            if (c > 0.0f)
            {
                float velocity_scale = max(abs(velocity.x), max(abs(velocity.y), abs(velocity.z)));
                if (velocity_scale == 0.0f)
                    continue;
                float3 scaled_velocity = velocity / velocity_scale;
                float a = dot(scaled_velocity, scaled_velocity);
                float b = dot(scaled_offset, scaled_velocity);
                if (b >= 0.0f)
                    continue;
                // Closest-approach residual avoids subtracting two large
                // nearly equal quadratic terms at an edge/corner tangent.
                float3 perpendicular = scaled_offset - scaled_velocity * (b / a);
                float discriminant = a * (scaled_radius * scaled_radius - dot(perpendicular, perpendicular));
                if (discriminant < 0.0f)
                    continue;
                root = ((c / (-b + sqrt(discriminant))) * offset_scale) / velocity_scale;
                if (root > width)
                    continue;
            }
        }
        float scaled_distance = start + root;
        float3 point = origin + scaled_distance * direction;
        float3 outward = point - clamp(point, -core, core);
        float normal_scale = max(abs(outward.x), max(abs(outward.y), abs(outward.z)));
        normal = normal_scale > 0.0f ? normalize(outward / normal_scale) : query_box_face_normal(point, geometry.xyz);
        distance = scaled_distance * scale;
        return true;
    }
    return false;
}

QueryHit query_miss()
{
    QueryHit hit;
    hit.meta = uint4(InvalidQueryBodySlot, 0u, InvalidQueryBodySlot, 0u);
    hit.positionDistance = float4(0.0f);
    hit.normalFraction = float4(0.0f);
    hit.pointADepth = float4(0.0f);
    return hit;
}

bool query_accepts(QueryInput request, ShapeModel shape, uint slot)
{
    if ((request.meta.w & QueryExcludeBody) != 0u && slot == request.source.x)
        return false;
    if ((request.meta.w & QueryTargetCollider) != 0u)
        return all(shape.source.xy == request.filter.zw);
    return (request.filter.x & (1u << shape.surface.x)) != 0u &&
           (request.filter.y != 0u || (shape.meta.w & ShapeSensor) == 0u);
}

void query_leaf_pose(ShapeModel shape, BodyState state, out float3 center, out float4 rotation)
{
    center = state.positionCOM.xyz + jph_rotate(state.rotation, shape.localPositionCOM.xyz);
    rotation = jph_quat_mul(state.rotation, shape.localRotation);
}

void query_leaf_bounds(ShapeModel shape, BodyState state, out float3 low, out float3 high)
{
    float3 center;
    float4 rotation;
    query_leaf_pose(shape, state, center, rotation);
    float3 extent = float3(shape.geometry.w);
    if (shape.meta.x == ShapeBox)
    {
        float3 core = max(shape.geometry.xyz - shape.geometry.w, float3(0.0f));
        extent += abs(jph_rotate(rotation, float3(1.0f, 0.0f, 0.0f))) * core.x +
                  abs(jph_rotate(rotation, float3(0.0f, 1.0f, 0.0f))) * core.y +
                  abs(jph_rotate(rotation, float3(0.0f, 0.0f, 1.0f))) * core.z;
    }
    else if (shape.meta.x == ShapeCapsule)
        extent += abs(jph_rotate(rotation, float3(0.0f, 1.0f, 0.0f))) * shape.geometry.y;
    else if (shape.meta.x == ShapeCylinder)
    {
        float3 axis = jph_rotate(rotation, float3(0, 1, 0));
        extent = abs(axis) * shape.geometry.y + shape.geometry.x * sqrt(max(float3(0), 1.0f - axis * axis));
    }
    else if (shape.meta.x == ShapeConvexHull)
    {
        uint4 ref = shape.geometryReference;
        float3 a = asfloat(query_geometry[ref.x + 3u]).xyz, b = asfloat(query_geometry[ref.x + 4u]).xyz;
        center += jph_rotate(rotation, 0.5f * (a + b));
        float3 h = 0.5f * (b - a);
        extent = abs(jph_rotate(rotation, float3(h.x, 0, 0))) +
                 abs(jph_rotate(rotation, float3(0, h.y, 0))) + abs(jph_rotate(rotation, float3(0, 0, h.z)));
    }
    low = center - extent;
    high = center + extent;
}

uint query_metric_status(JphPairMetric metric)
{
    return metric.status == 1u ? QuerySuccess :
           metric.status == 0u ? QueryUnsupported : QueryNumericalFailure;
}

QueryHit query_leaf(QueryInput request, ShapeModel shape, BodyState state, uint leaf, out uint status)
{
    status = QuerySuccess;
    QueryHit hit = query_miss();
    float3 center;
    float4 rotation;
    query_leaf_pose(shape, state, center, rotation);
    if (!all(isfinite(center)) || !all(isfinite(rotation)))
    {
        status = QueryNumericalFailure;
        return hit;
    }
    uint query_type = request.meta.x == QueryClosestPoint ? ShapeSphere : request.meta.z;
    float4 query_geometry = request.meta.x == QueryClosestPoint ? float4(0.0f) : request.geometry;
    JphPairMetric metric = jph_pair_metric(query_type, request.positionMaximum.xyz, request.rotation, query_geometry,
                                          shape.meta.x, center, rotation, shape.geometry);
    status = query_metric_status(metric);
    if (status != QuerySuccess)
        return hit;
    if (request.meta.x == QueryOverlap && metric.separation > 0.0f)
        return hit;
    hit.meta = uint4(shape.meta.y, 1u, leaf, metric.feature);
    float3 point = metric.pointB;
    if (request.meta.x == QueryClosestPoint && metric.separation <= 0.0f)
        point = request.positionMaximum.xyz;
    hit.positionDistance = float4(point, max(metric.separation, 0.0f));
    hit.normalFraction = float4(-metric.normal, 0.0f);
    hit.pointADepth = float4(metric.pointA, max(-metric.separation, 0.0f));
    return hit;
}

bool query_less(QueryInput request, QueryHit a, QueryHit b)
{
    if (request.meta.y == QueryUniqueCollider)
    {
        uint2 ca = shape_models[a.meta.z].source.xy;
        uint2 cb = shape_models[b.meta.z].source.xy;
        return ca.y != cb.y ? ca.y < cb.y : ca.x < cb.x;
    }
    if (request.meta.y == QueryUniqueBody)
    {
        uint2 a_id = query_body_ids[a.meta.x - query_params.selection.y].ids.zw;
        uint2 b_id = query_body_ids[b.meta.x - query_params.selection.y].ids.zw;
        if (a_id.y != b_id.y || a_id.x != b_id.x)
            return a_id.y != b_id.y ? a_id.y < b_id.y : a_id.x < b_id.x;
    }
    if (request.meta.y != QueryUniqueBody && a.positionDistance.w != b.positionDistance.w)
        return a.positionDistance.w < b.positionDistance.w;
    uint bodyA = query_body_ids[a.meta.x - query_params.selection.y].ids.x;
    uint bodyB = query_body_ids[b.meta.x - query_params.selection.y].ids.x;
    if (bodyA != bodyB)
        return bodyA < bodyB;
    uint subA = shape_models[a.meta.z].meta.z;
    uint subB = shape_models[b.meta.z].meta.z;
    return subA != subB ? subA < subB : a.meta.z < b.meta.z;
}

void query_collect(QueryInput request, QueryHit hit, uint begin, inout uint count)
{
    if (hit.meta.y == 0u)
        return;
    if (request.meta.y == QueryClosest)
    {
        QueryHit previous = query_load_hit(begin);
        // Closest rays preserve the existing strict-distance slot/leaf tie.
        if (previous.meta.y == 0u || hit.positionDistance.w < previous.positionDistance.w ||
            (request.meta.x == QueryShapeCast && hit.positionDistance.w == previous.positionDistance.w &&
             query_less(request, hit, previous)))
            query_store_hit(begin, hit);
        return;
    }
    if (request.meta.y == QueryUniqueCollider)
        for (uint old = 0u; old < count; ++old)
            if (all(shape_models[query_load_hit(begin + old).meta.z].source.xy == shape_models[hit.meta.z].source.xy))
                return;
    if (request.meta.y == QueryUniqueBody)
        for (uint old = 0u; old < count; ++old)
            if (query_load_hit(begin + old).meta.x == hit.meta.x)
                return;
    uint position = count;
    while (position > 0u && query_less(request, hit, query_load_hit(begin + position - 1u)))
    {
        query_store_hit(begin + position, query_load_hit(begin + position - 1u));
        --position;
    }
    query_store_hit(begin + position, hit);
    ++count;
}


bool query_valid_quaternion(float4 q)
{
    float length = dot(q, q);
    return all(isfinite(q)) && isfinite(length) && length > 0.0f;
}

float3 query_cast_support(uint type, float4 rotation, float4 geometry, float3 n)
{
    float3 core = float3(0.0f);
    if (type == ShapeCapsule)
    {
        float3 axis = jph_rotate(rotation, float3(0, 1, 0));
        core = (dot(axis, n) >= 0.0f ? geometry.y : -geometry.y) * axis;
    }
    else if (type == ShapeBox)
    {
        float3 local = jph_inverse_rotate(rotation, n);
        float3 half = max(geometry.xyz - geometry.w, float3(0.0f));
        core = jph_rotate(rotation, float3(local.x >= 0.0f ? half.x : -half.x,
                                          local.y >= 0.0f ? half.y : -half.y,
                                          local.z >= 0.0f ? half.z : -half.z));
    }
    return core + geometry.w * n;
}

bool query_cast_accept(QueryInput request, float3 normal, float3 direction)
{
    float closing = dot(normal, direction);
    if ((request.meta.w & QuerySweepPolicy) != 0u)
        return closing > 1.0e-3f;
    return closing >= 0.0f;
}

bool query_cast_valid_hit(QueryHit hit)
{
    return all(isfinite(hit.positionDistance)) && all(isfinite(hit.normalFraction)) &&
           all(isfinite(hit.pointADepth)) && hit.positionDistance.w >= 0.0f;
}

// An exact primitive reduction, not another executor: a Sphere versus a
// rounded primitive is a ray against the same core with the radius sum.
bool query_cast_sphere_root(QueryInput request, ShapeModel target, float3 centerRelative, float4 rotation,
    float3 direction, float maximum, out QueryHit hit, out uint status)
{
    hit = query_miss();
    status = QuerySuccess;
    bool reverse = request.meta.z != ShapeSphere;
    uint type = reverse ? request.meta.z : target.meta.x;
    float4 q = reverse ? request.rotation : rotation;
    float4 geometry = reverse ? request.geometry : target.geometry;
    float radius = reverse ? target.geometry.w : request.geometry.w;
    float3 relative = reverse ? centerRelative : -centerRelative;
    float3 local = jph_inverse_rotate(q, relative);
    float3 velocity = jph_inverse_rotate(q, reverse ? -direction : direction);
    float inflated = geometry.w + radius;
    if (!all(isfinite(local)) || !all(isfinite(velocity)) || !isfinite(inflated))
    {
        status = QueryNumericalFailure;
        return false;
    }
    float distance;
    float3 outward;
    bool found = false;
    if (type == ShapeSphere)
        found = query_sphere(local, velocity, inflated, maximum, distance, outward);
    else if (type == ShapeCapsule)
        found = query_capsule(local, velocity, geometry.y, inflated, maximum, distance, outward);
    else if (type == ShapeBox)
    {
        float4 expanded = float4(geometry.xyz + radius, inflated);
        if (!all(isfinite(expanded)))
        {
            status = QueryNumericalFailure;
            return false;
        }
        found = query_rounded_box(local, velocity, expanded, maximum, distance, outward);
    }
    else
    {
        status = QueryUnsupported;
        return false;
    }
    if (!found)
        return false;
    if (!isfinite(distance) || distance < 0.0f || !all(isfinite(outward)) ||
        jph_capsule_max_abs(outward) == 0.0f)
    {
        status = QueryNumericalFailure;
        return false;
    }
    float3 normal = jph_rotate(q, jph_capsule_unit(outward));
    float3 n = reverse ? normal : -normal;
    if (!query_cast_accept(request, n, direction))
        return false;
    float3 point;
    if (reverse || target.meta.x == ShapeSphere)
        point = centerRelative - target.geometry.w * n;
    else
    {
        float3 at = local + velocity * distance;
        float3 core = float3(0.0f);
        if (type == ShapeCapsule)
        {
            core.y = clamp(at.y, -geometry.y, geometry.y);
            if (outward.y != 0.0f)
                core.y = outward.y > 0.0f ? geometry.y : -geometry.y;
        }
        else
        {
            float3 half = max(geometry.xyz - geometry.w, float3(0.0f));
            core = clamp(at, -half, half);
            for (uint axis = 0u; axis < 3u; ++axis)
                if (outward[axis] != 0.0f)
                    core[axis] = outward[axis] > 0.0f ? half[axis] : -half[axis];
        }
        point = centerRelative + jph_rotate(rotation, core + geometry.w * jph_capsule_unit(outward));
    }
    point += request.positionMaximum.xyz;
    hit.meta = uint4(target.meta.y, 1u, 0u, 0u);
    hit.positionDistance = float4(point, distance);
    hit.normalFraction = float4(-n, request.positionMaximum.w > 0.0f ? distance / request.positionMaximum.w : 0.0f);
    hit.pointADepth = float4(point, 0.0f);
    if (!query_cast_valid_hit(hit))
    {
        status = QueryNumericalFailure;
        return false;
    }
    return true;
}

// One real CA step. true ends the current pair; false retains its exact travel
// and restart state for this request's next quantum. No iteration cap is miss.
bool query_cast_pair_step(QueryInput request, ShapeModel target, BodyState body,
    inout QueryCastState cursor, float3 direction, out QueryHit hit, out uint status)
{
    hit = query_miss();
    status = QuerySuccess;
    float3 relative;
    float4 rotation;
    body.positionCOM.xyz -= request.positionMaximum.xyz;
    query_leaf_pose(target, body, relative, rotation);
    if (!all(isfinite(relative)) || !query_valid_quaternion(rotation) ||
        target.meta.x > ShapeCapsule)
    {
        status = target.meta.x > ShapeCapsule ? QueryUnsupported : QueryNumericalFailure;
        return true;
    }
    rotation = normalize(rotation);
    float travel = cursor.progress.x;
    float3 remainingCenter = relative - direction * travel;
    float scale = max(jph_capsule_max_abs(remainingCenter),
        max(max(jph_capsule_max_abs(request.geometry.xyz), request.geometry.w),
            max(jph_capsule_max_abs(target.geometry.xyz), target.geometry.w)));
    if (!all(isfinite(relative)) || !all(isfinite(remainingCenter)) || !isfinite(scale) || scale <= 0.0f)
    {
        status = QueryNumericalFailure;
        return true;
    }
    float3 b = remainingCenter / scale;
    float4 ga = request.geometry / scale, gb = target.geometry / scale;
    float3 origin = cursor.iteration.z == 0u ? float3(0.0f) : -0.5f * b;
    JphPairMetric metric = jph_pair_metric(request.meta.z, origin, request.rotation, ga,
        target.meta.x, b + origin, rotation, gb);
    metric.pointA -= origin;
    metric.pointB -= origin;
    status = query_metric_status(metric);
    if (status != QuerySuccess)
        return true;
    float normalSize = jph_capsule_max_abs(metric.normal);
    if (normalSize == 0.0f || !isfinite(normalSize))
    {
        status = QueryNumericalFailure;
        return true;
    }
    metric.normal = jph_capsule_unit(metric.normal);
    float separation = metric.separation * scale;
    if (!isfinite(separation))
    {
        status = QueryNumericalFailure;
        return true;
    }
    bool initial = cursor.iteration.y == 0u;
    ++cursor.iteration.y;
    float limit = min(request.positionMaximum.w, cursor.progress.w);
    if (initial && separation > 0.0f &&
        (request.meta.z == ShapeSphere || target.meta.x == ShapeSphere))
    {
        query_cast_sphere_root(request, target, relative, rotation, direction, limit, hit, status);
        return true;
    }
    if (separation > 0.0f)
    {
        float3 frontA = query_cast_support(request.meta.z, request.rotation, ga, metric.normal);
        float3 backB = b + query_cast_support(target.meta.x, rotation, gb, -metric.normal);
        float gap = dot(metric.normal, backB - frontA) * scale;
        uint result = cvx_cast_progress(separation, gap, dot(metric.normal, direction), limit,
                                        cursor.progress.x, cursor.iteration.z);
        if (result == CvxCastError)
        {
            status = QueryNumericalFailure;
            return true;
        }
        if (result == CvxCastMiss) return true;
        if (result == CvxCastPending) return false;
    }
    if (!query_cast_accept(request, metric.normal, direction))
        return true;
    float3 pointA = request.positionMaximum.xyz + (direction * travel + metric.pointA * scale);
    float3 pointB = request.positionMaximum.xyz + (relative + (metric.pointB - b) * scale);
    hit.meta = uint4(target.meta.y, 1u, cursor.cursor.x, metric.feature);
    hit.positionDistance = float4(pointB, travel);
    hit.normalFraction = float4(-metric.normal, request.positionMaximum.w > 0.0f ? travel / request.positionMaximum.w : 0.0f);
    hit.pointADepth = float4(pointA, max(-separation, 0.0f));
    if (!query_cast_valid_hit(hit))
        status = QueryNumericalFailure;
    return true;
}

ColliderPairKey query_collider_pair(uint index)
{
    uint base = query_cast_params.controls.z + index * 2u;
    ColliderPairKey key;
    key.a = query_body_ids[base].ids;
    key.b = query_body_ids[base + 1u].ids;
    return key;
}

bool query_collider_suppressed(uint4 a, uint4 b)
{
    ColliderPairKey key = jph_collider_pair_key(a, b);
    uint low = 0u, high = query_cast_params.controls.w;
    for (uint step = 0u; step < 32u && low < high; ++step)
    {
        uint middle = low + (high - low) / 2u;
        ColliderPairKey value = query_collider_pair(middle);
        if (jph_collider_pair_less(value, key)) low = middle + 1u;
        else high = middle;
    }
    if (low >= query_cast_params.controls.w) return false;
    return jph_collider_pair_equal(query_collider_pair(low), key);
}

void query_cast_advance(QueryInput request, uint begin, inout QueryCastState state)
{
    uint quantum = max(query_cast_params.controls.x, 1u);
    float3 direction = jph_capsule_unit(request.direction.xyz);
    for (uint unit = 0u; unit < quantum && state.cursor.y != 2u; ++unit)
    {
        if (state.iteration.x == 0xffffffffu)
        {
            state.cursor.w = QueryNumericalFailure;
            state.cursor.y = 2u;
            break;
        }
        ++state.iteration.x;
        if (state.cursor.x - query_params.selection.z >= query_params.selectedCounts.y)
        {
            state.cursor.w = QuerySuccess;
            state.cursor.y = 2u;
            break;
        }
        uint leaf = state.cursor.x;
        ShapeModel shape = shape_models[leaf];
        uint slot = shape.meta.y;
        if (state.cursor.y == 0u)
        {
            bool admitted = slot >= query_params.counts.y && slot - query_params.counts.y < query_params.counts.z;
            if (admitted)
            {
                BodyModel model = body_models[slot];
                admitted = jph_resident(model) && leaf >= model.meta.x && leaf - model.meta.x < model.meta.y &&
                           query_accepts(request, shape, slot);
            }
            if (admitted && (request.meta.w & QuerySweepPolicy) != 0u)
                admitted = !all(query_body_ids[query_cast_params.controls.y + (leaf - query_params.selection.z)].ids.xy == request.source.zw);
            if (admitted && (request.meta.w & QuerySweepPairPolicy) != 0u)
                admitted = !query_collider_suppressed(uint4(request.filter.zw, request.lifetime.xy), shape.source);
            if (!admitted)
            {
                ++state.cursor.x;
                continue;
            }
            state.cursor.y = 1u;
            state.progress.xyz = float3(0.0f, 3.402823466e+38f, 0.0f);
            state.normal = float4(0.0f);
            state.iteration.yzw = uint3(0u);
            continue; // A leaf visit consumes a whole request work unit.
        }
        QueryHit hit;
        uint status;
        bool finished = query_cast_pair_step(request, shape, body_states[slot - query_params.selection.y], state, direction, hit, status);
        if (status != QuerySuccess)
        {
            state.cursor.w = status;
            state.cursor.y = 2u;
            break;
        }
        if (!finished)
            continue;
        if (hit.meta.y != 0u)
        {
            hit.meta.z = leaf;
            query_collect(request, hit, begin, state.cursor.z);
            if (request.meta.y == QueryClosest)
                state.progress.w = query_load_hit(begin).positionDistance.w;
            if (request.meta.y == QueryAny)
            {
                state.cursor.w = QuerySuccess;
                state.cursor.y = 2u;
                break;
            }
        }
        ++state.cursor.x;
        state.cursor.y = 0u;
    }
    if (state.cursor.y != 2u)
        state.cursor.w = QueryPending;
}

#include "rigid_convex_queries.slang"

[shader("compute")]
[numthreads(64, 1, 1)]
void query_evaluate(uint3 tid : SV_DispatchThreadID)
{
    uint index = tid.x;
    if (index >= query_params.counts.x)
        return;
    QueryInput request = query_load_input(index);
    uint begin = index * query_params.sizes.y;
    for (uint row = 0u; row < query_params.sizes.y; ++row)
        query_clear_candidate(begin + row);
    uint count = request.meta.y == QueryClosest ? 1u : 0u;
    uint status = QuerySuccess;
    QueryCastState cast;
    cast.cursor = uint4(query_params.selection.z, 2u, count, QuerySuccess);
    cast.progress = float4(0.0f, 3.402823466e+38f, 0.0f, request.positionMaximum.w);
    cast.normal = float4(0.0f);
    cast.iteration = uint4(0u);
    if (cvx_query_slots() != 0u && cvx_query_deferred(request))
    {
        cast.cursor.y = 0u;
        cast.cursor.w = status = QueryPending;
    }
    else if (request.meta.x == QueryShapeCast)
    {
        cast.cursor.y = 0u;
        query_cast_advance(request, begin, cast);
        count = cast.cursor.z;
        status = cast.cursor.w;
    }
    else if (request.meta.x == QueryPenetration)
    {
        QueryPrimitive other = query_load_primitive(request.source.y);
        JphPairMetric metric = jph_pair_metric(request.meta.z, request.positionMaximum.xyz, request.rotation, request.geometry,
                                              other.meta.x, other.position.xyz, other.rotation, other.geometry);
        status = query_metric_status(metric);
        if (status == QuerySuccess && metric.separation < 0.0f)
        {
            QueryHit hit = query_miss();
            hit.meta.yw = uint2(1u, metric.feature);
            hit.positionDistance = float4(metric.pointB, 0.0f);
            hit.normalFraction = float4(-metric.normal, 0.0f);
            hit.pointADepth = float4(metric.pointA, -metric.separation);
            query_store_hit(begin, hit);
            count = 1u;
        }
    }
    else
    {
        for (uint local = 0u; local < query_params.counts.z && status == QuerySuccess; ++local)
        {
            uint slot = query_params.counts.y + local;
            BodyModel model = body_models[slot];
            if (!jph_resident(model) || model.meta.y == 0u)
                continue;
            if (request.meta.x == QueryBodyBounds && query_body_ids[slot - query_params.selection.y].ids.y == 0u)
                continue;
            BodyState state = body_states[slot - query_params.selection.y];
            bool accepted = false;
            float3 low = float3(3.402823466e+38f);
            float3 high = -low;
            uint first = model.meta.x;
            for (uint ordinal = 0u; ordinal < model.meta.y; ++ordinal)
            {
                uint leaf = model.meta.x + ordinal;
            if (leaf < query_params.selection.z || leaf - query_params.selection.z >= query_params.selectedCounts.y)
            {
                status = QueryNumericalFailure;
                break;
            }
                ShapeModel shape = shape_models[leaf];
                if (request.meta.x == QueryBodyBounds)
                {
                    float3 a, b;
                    query_leaf_bounds(shape, state, a, b);
                    if (!all(isfinite(a)) || !all(isfinite(b)))
                    {
                        status = QueryNumericalFailure;
                        break;
                    }
                    low = min(low, a);
                    high = max(high, b);
                    accepted = accepted || query_accepts(request, shape, slot);
                    continue;
                }
                if (!query_accepts(request, shape, slot))
                    continue;
                uint leaf_status;
                QueryHit hit = query_leaf(request, shape, state, leaf, leaf_status);
                if (leaf_status != QuerySuccess)
                {
                    status = leaf_status;
                    break;
                }
                query_collect(request, hit, begin, count);
                if (request.meta.y == QueryAny && count != 0u)
                    break;
            }
            if (request.meta.x == QueryBodyBounds && accepted &&
                all(low <= request.positionMaximum.xyz + request.direction.xyz) &&
                all(high >= request.positionMaximum.xyz - request.direction.xyz))
            {
                QueryHit hit = query_miss();
                hit.meta = uint4(slot, 1u, first, 0u);
                query_collect(request, hit, begin, count);
            }
            if (request.meta.y == QueryAny && count != 0u)
                break;
        }
    }
    query_store_cast_state(index, cast);
    QuerySpan span;
    span.range = uint4(0u, count, status, 0u);
    query_store_span(index, span);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void query_resume_casts(uint3 tid : SV_DispatchThreadID)
{
    uint index = tid.x;
    if (index >= query_params.counts.x || query_load_span(index).range.z != QueryPending)
        return;
    QueryInput request = query_load_input(index);
    if (request.meta.x != QueryShapeCast)
        return;
    QueryCastState cast = query_load_cast_state(index);
    query_cast_advance(request, index * query_params.sizes.y, cast);
    query_store_cast_state(index, cast);
    QuerySpan span;
    span.range = uint4(0u, cast.cursor.z, cast.cursor.w, 0u);
    query_store_span(index, span);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void query_scan(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u)
        return;
    uint count = 0u;
    uint status = QuerySuccess;
    bool pending = false;
    for (uint index = 0u; index < query_params.counts.x; ++index)
    {
        QuerySpan span = query_load_span(index);
        span.range.x = count;
        count += span.range.y;
        if (span.range.z == QueryPending)
            pending = true;
        else
            status = max(status, span.range.z);
        query_store_span(index, span);
    }
    QuerySummary summary;
    summary.counts = uint4(count, status != QuerySuccess ? status : pending ? QueryPending : QuerySuccess, query_params.counts.x, 0u);
    query_summary[0] = summary;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void query_compact(uint3 tid : SV_DispatchThreadID)
{
    uint index = tid.x;
    if (index >= query_params.sizes.z)
        return;
    uint source_record = InvalidQueryBodySlot;
    if (index < query_summary[0].counts.x)
    {
        uint low = 0u;
        uint high = query_params.counts.x;
        while (low < high)
        {
            uint middle = low + (high - low) / 2u;
            QuerySpan span = query_load_span(middle);
            if (span.range.x + span.range.y <= index)
                low = middle + 1u;
            else
                high = middle;
        }
        QuerySpan source = query_load_span(low);
        uint begin = low * query_params.sizes.y;
        source_record = begin + (index - source.range.x);
    }
    // One writer per final index, including the unused tail. A sparse scatter
    // plus concurrent clearing would race and would not prove initialization.
    const uint words = 4u;
    for (uint word = 0u; word < words; ++word)
        query_hits[index * words + word].value = source_record != InvalidQueryBodySlot
            ? query_candidates[source_record * words + word].value
            : word == 0u ? uint4(InvalidQueryBodySlot, 0u, InvalidQueryBodySlot, 0u) : uint4(0u);
}
