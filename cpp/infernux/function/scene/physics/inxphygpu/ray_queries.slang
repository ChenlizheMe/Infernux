// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from MuJoCo Warp _src/ray.py.

#include "collision_geometry.slang"
#include "convex_model.slang"
#include "heightfield_geometry.slang"
#include "ray_bvh_layout.slang"
#include "triangle_mesh_geometry.slang"

static const uint RayIncludeStatic = 1;
static const uint RayIncludeTriggers = 2;
static const uint RayHitBackFaces = 4;

struct PhysicsRay
{
    float4 originLimit;
    float4 direction;
    uint4 filter; // Layer mask, excluded body, flags, unused.
};

struct PhysicsRayHit
{
    float4 normalDistance;
    uint4 geometry; // Geometry row, body row, subshape ID, asset face index.
};

struct RaySurface
{
    float3 normal;
    float distance;
};

static const RaySurface RayMiss = {float3(0.0), -1.0};
static const uint RayGroupSize = 64;

[[vk::binding(0, 0)]] StructuredBuffer<uint4> raySettings; // geometry count, rays per world, input world rows, unused
[[vk::binding(1, 0)]] StructuredBuffer<PhysicsRay> rays;
[[vk::binding(2, 0)]] StructuredBuffer<CartesianFrame> rayFrames;
[[vk::binding(3, 0)]] StructuredBuffer<CollisionShape> rayShapes;
[[vk::binding(4, 0)]] StructuredBuffer<uint4> rayGeometry; // Body row, layer bits, flags, subshape ID.
[[vk::binding(5, 0)]] StructuredBuffer<uint4> rayModel;
[[vk::binding(7, 0)]] RWStructuredBuffer<PhysicsRayHit> rayHits;
[[vk::binding(8, 0)]] StructuredBuffer<uint4> rayIndex;
[[vk::binding(9, 0)]] StructuredBuffer<RayBvhBounds> rayBounds;
[[vk::binding(10, 0)]] StructuredBuffer<uint4> packedRaySettings; // Count, origin and direction scalar offsets, limit; filter.
[[vk::binding(11, 0)]] StructuredBuffer<float> packedRayOrigins;
[[vk::binding(12, 0)]] StructuredBuffer<float> packedRayDirections;
[[vk::binding(13, 0)]] RWStructuredBuffer<PhysicsRay> packedRays;

[shader("compute")]
[numthreads(RayGroupSize, 1, 1)]
void pack_scene_rays(uint3 id : SV_DispatchThreadID)
{
    const uint4 settings = packedRaySettings[0];
    if (id.x >= settings.x)
        return;
    const uint first = 3u * id.x;
    const uint o = settings.y + first, d = settings.z + first;
    const float3 origin = float3(packedRayOrigins[o], packedRayOrigins[o + 1], packedRayOrigins[o + 2]);
    const float3 direction = float3(packedRayDirections[d], packedRayDirections[d + 1], packedRayDirections[d + 2]);
    const float scale = max(max(abs(direction.x), abs(direction.y)), abs(direction.z));
    const bool valid = all(isfinite(origin)) && all(isfinite(direction)) && scale > 0.0;
    PhysicsRay ray = {float4(0.0), float4(0.0), packedRaySettings[1]};
    if (valid)
    {
        ray.originLimit = float4(origin, asfloat(settings.w));
        ray.direction.xyz = normalize(direction / scale);
    }
    packedRays[id.x] = ray;
}

float rayDot(float3 a, float3 b)
{
    return mad(a.z, b.z, mad(a.y, b.y, a.x * b.x));
}

float3 rayLocalVector(CartesianFrame frame, float3 vector)
{
    return float3(rayDot(frameAxis(frame, 0), vector), rayDot(frameAxis(frame, 1), vector),
                  rayDot(frameAxis(frame, 2), vector));
}

float2 rayRoots(float a, float b, float c)
{
    const float determinant = mad(-a, c, b * b);
    if (determinant < MinValue)
        return float2(-1.0);
    const float root = sqrt(determinant);
    return (float2(-root, root) - b) * (1.0 / a);
}

float nearestRayRoot(float2 roots)
{
    return roots.x >= 0.0 ? roots.x : roots.y;
}

bool nearerRaySurface(float distance, RaySurface best)
{
    return distance >= 0.0 && (best.distance < 0.0 || distance < best.distance);
}

float2 raySphereRoots(float3 origin, float3 direction, float radiusSquared)
{
    return rayRoots(rayDot(direction, direction), rayDot(direction, origin), rayDot(origin, origin) - radiusSquared);
}

RaySurface rayBox(float3 origin, float3 direction, float3 size, out float faces[6])
{
    RaySurface best = RayMiss;
    for (uint axis = 0; axis < 3; ++axis)
    {
        const uint a = (axis + 1) % 3;
        const uint b = (axis + 2) % 3;
        for (uint side = 0; side < 2; ++side)
        {
            const uint face = 2 * axis + side;
            const float sign = side == 0 ? -1.0 : 1.0;
            faces[face] = -1.0;
            if (abs(direction[axis]) <= MinValue)
                continue;
            const float distance = (sign * size[axis] - origin[axis]) / direction[axis];
            const float3 point = origin + distance * direction;
            if (distance < 0.0 || abs(point[a]) > size[a] || abs(point[b]) > size[b])
                continue;
            faces[face] = distance;
            if (nearerRaySurface(distance, best))
            {
                best = {float3(0.0), distance};
                best.normal[axis] = sign;
            }
        }
    }
    return best;
}

RaySurface rayRoundCylinder(float3 origin, float3 direction, float2 size, bool capsule)
{
    const float radius2 = size.x * size.x;
    const float bound2 = capsule ? (size.x + size.y) * (size.x + size.y) : radius2 + size.y * size.y;
    if (nearestRayRoot(raySphereRoots(origin, direction, bound2)) < 0.0)
        return RayMiss;

    RaySurface best = RayMiss;
    for (int side = -1; side <= 1; side += 2)
    {
        const float height = float(side) * size.y;
        if (capsule)
        {
            const float3 center = float3(0.0, 0.0, height);
            const float2 roots = raySphereRoots(origin - center, direction, radius2);
            for (uint i = 0; i < 2; ++i)
            {
                const float3 point = origin + roots[i] * direction;
                const float3 offset = point - center;
                if (nearerRaySurface(roots[i], best) && float(side) * offset.z >= 0.0)
                    best = {normalize(offset), roots[i]};
            }
        }
        else if (abs(direction.z) > MinValue)
        {
            const float distance = (height - origin.z) / direction.z;
            const float2 point = origin.xy + distance * direction.xy;
            if (nearerRaySurface(distance, best) && dot(point, point) <= radius2)
                best = {float3(0.0, 0.0, float(side)), distance};
        }
    }

    const float2 roots = rayRoots(dot(direction.xy, direction.xy), dot(direction.xy, origin.xy),
                                  dot(origin.xy, origin.xy) - radius2);
    const float distance = nearestRayRoot(roots);
    const float3 point = origin + distance * direction;
    if (nearerRaySurface(distance, best) && abs(point.z) <= size.y)
        best = {normalize(float3(point.xy, 0.0)), distance};
    return best;
}

void rayTriangleBasis(float3 direction, out float3 b0, out float3 b1)
{
    const float sign = direction.z >= 0.0 ? 1.0 : -1.0;
    const float a = -1.0 / (sign + direction.z);
    const float b = direction.x * direction.y * a;
    b0 = float3(1.0 + sign * direction.x * direction.x * a, sign * b, -sign * direction.x);
    b1 = float3(b, sign + direction.y * direction.y * a, -direction.y);
}

RaySurface rayTriangle(float3 origin, float3 direction, float3 b0, float3 b1, float3 a, float3 b, float3 c)
{
    const float3 da = a - origin;
    const float3 db = b - origin;
    const float3 dc = c - origin;
    const float2 pa = float2(dot(da, b0), dot(da, b1));
    const float2 pb = float2(dot(db, b0), dot(db, b1));
    const float2 pc = float2(dot(dc, b0), dot(dc, b1));
    if (any(min(pa, min(pb, pc)) > 0.0) || any(max(pa, max(pb, pc)) < 0.0))
        return RayMiss;

    const float2 ac = pa - pc;
    const float2 bc = pb - pc;
    const float determinant = ac.x * bc.y - ac.y * bc.x;
    if (abs(determinant) < MinValue)
        return RayMiss;
    const float2 uv = float2(bc.x * pc.y - bc.y * pc.x, ac.y * pc.x - ac.x * pc.y) / determinant;
    if (any(uv < 0.0) || uv.x + uv.y > 1.0)
        return RayMiss;

    const float3 normal = cross(a - c, b - c);
    const float denominator = dot(direction, normal);
    if (abs(denominator) < MinValue)
        return RayMiss;
    return {normalize(normal), dot(dc, normal) / denominator};
}

RaySurface rayHeightfield(float3 origin, float3 direction, uint field)
{
    const HeightfieldGrid grid = heightfieldGrid(rayModel, field);
    const float4 size = grid.size;
    const float bottom = 0.5 * size.w, top = 0.5 * size.z;
    float faces[6];
    RaySurface best = rayBox(origin + float3(0.0, 0.0, bottom), direction, float3(size.xy, bottom), faces);
    const RaySurface upper = rayBox(origin - float3(0.0, 0.0, top), direction, float3(size.xy, top), faces);
    if (upper.distance < 0.0)
        return best;

    float2 segment = float2(0.0, upper.distance);
    for (uint i = 0; i < 6; ++i)
        if (faces[i] > segment.y)
            segment = float2(upper.distance, faces[i]);

    const uint2 intervals = grid.layout.yx - 1u;
    const float2 step = 2.0 * size.xy / float2(intervals);
    const float2 begin = (origin.xy + segment.x * direction.xy + size.xy) / step;
    const float2 end = (origin.xy + segment.y * direction.xy + size.xy) / step;
    const int2 lo = max(0, int2(floor(min(begin, end))) - 1);
    const int2 hi = min(int2(intervals), int2(ceil(max(begin, end))) + 1);
    float3 b0, b1;
    rayTriangleBasis(direction, b0, b1);
    for (int row = lo.y; row < hi.y; ++row)
    {
        for (int col = lo.x; col < hi.x; ++col)
        {
            const float3 a = heightfieldVertex(rayModel, grid, uint2(col, row), step, 0.0);
            const float3 b = heightfieldVertex(rayModel, grid, uint2(col + 1, row), step, 0.0);
            const float3 c = heightfieldVertex(rayModel, grid, uint2(col + 1, row + 1), step, 0.0);
            const float3 d = heightfieldVertex(rayModel, grid, uint2(col, row + 1), step, 0.0);
            const RaySurface lower = rayTriangle(origin, direction, b0, b1, a, b, c);
            const RaySurface higher = rayTriangle(origin, direction, b0, b1, a, c, d);
            if (nearerRaySurface(lower.distance, best))
                best = lower;
            if (nearerRaySurface(higher.distance, best))
                best = higher;
        }
    }

    for (uint face = 0; face < 4; ++face)
    {
        const float distance = faces[face];
        if (!nearerRaySurface(distance, best))
            continue;
        const float3 point = origin + distance * direction;
        const uint axis = face / 2;
        const uint across = 1 - axis;
        const float coordinate = (point[across] + size[across]) / step[across];
        const uint cell = uint(clamp(floor(coordinate), 0.0, float(intervals[across] - 1u)));
        uint2 edge;
        edge[axis] = face % 2 == 0 ? 0 : intervals[axis];
        edge[across] = cell;
        const float a = heightfieldElevation(rayModel, grid, edge);
        ++edge[across];
        const float b = heightfieldElevation(rayModel, grid, edge);
        if (point.z / size.z < lerp(a, b, coordinate - float(cell)))
        {
            best = {float3(0.0), distance};
            best.normal[axis] = face % 2 == 0 ? -1.0 : 1.0;
        }
    }
    return best;
}

RaySurface rayMesh(float3 origin, float3 direction, CollisionShape shape, float maximum, bool backFaces,
                   out uint subShape, out uint assetFace)
{
    const TriangleMeshTree tree = triangleMeshTree(rayModel, shape);
    float3 b0, b1;
    rayTriangleBasis(direction, b0, b1);
    const RaySlabs slabs = raySlabs(origin, direction);
    RaySurface best = {float3(0.0), maximum};
    subShape = NoIndex;
    assetFace = NoIndex;
    bool found = false;
    uint node = 0;
    while (node < tree.nodeCount)
    {
        const uint address = tree.nodeBase + 2 * node;
        const uint4 low = rayModel[address];
        const uint4 high = rayModel[address + 1];
        const bool overlaps = rayBoundsOverlap(slabs, asfloat(low.xyz), asfloat(high.xyz), best.distance);
        if ((high.w & BvhLeaf) == 0)
        {
            node = overlaps ? low.w : high.w;
            continue;
        }
        if (overlaps)
        {
            const uint end = low.w + (high.w & ~BvhLeaf);
            for (uint i = low.w; i < end; ++i)
            {
                const uint4 triangle = rayModel[tree.faceBase + i];
                const RaySurface hit = rayTriangle(origin, direction, b0, b1, convexVertex(rayModel, triangle.x),
                    convexVertex(rayModel, triangle.y), convexVertex(rayModel, triangle.z));
                if (hit.distance < 0.0 || hit.distance > best.distance ||
                    (!backFaces && dot(hit.normal, direction) >= 0.0))
                    continue;
                const uint id = triangleMeshSubShape(rayModel, tree, triangle.w);
                if (!found || hit.distance < best.distance || id < subShape)
                {
                    best = hit;
                    subShape = id;
                    assetFace = triangleMeshFeature(rayModel, tree, triangle.w, 2);
                    found = true;
                }
            }
        }
        ++node;
    }
    return found ? best : RayMiss;
}

RaySurface rayHull(float3 origin, float3 direction, CollisionShape shape, uint geometry)
{
    const uint2 faces = convexTopology(rayModel, geometry).xy;
    float earliest = 0.0;
    float latest = 3.402823e38;
    float depth = -latest;
    float3 front = float3(0.0);
    float3 nearest = float3(0.0);
    for (uint i = faces.x; i < faces.y; ++i)
    {
        const uint4 face = convexTopology(rayModel, i);
        const float3 normal = convexVertex(rayModel, face.z);
        const float3 point = convexVertex(rayModel, shape.type.y + convexLink(rayModel, face.x));
        const float gap = dot(origin - point, normal);
        if (gap > depth)
        {
            depth = gap;
            nearest = normal;
        }
        const float speed = dot(direction, normal);
        if (abs(speed) <= MinValue)
        {
            if (gap > 0.0)
                return RayMiss;
        }
        else if (speed < 0.0)
        {
            const float time = -gap / speed;
            if (time > earliest)
            {
                earliest = time;
                front = normal;
            }
        }
        else
        {
            latest = min(latest, -gap / speed);
        }
        if (earliest > latest)
            return RayMiss;
    }
    return {depth <= 0.0 ? nearest : front, earliest};
}

RaySurface rayGeometrySurface(float3 origin, float3 direction, CollisionShape shape, uint geometry, float maximum,
                              bool backFaces, inout uint subShape, out uint assetFace)
{
    assetFace = NoIndex;
    const float3 size = shape.size.xyz;
    switch (shape.type.x)
    {
    case GeomPlane:
        if (direction.z <= -MinValue)
        {
            const float distance = -origin.z / direction.z;
            const float2 point = origin.xy + distance * direction.xy;
            if (distance >= 0.0 && all((size.xy <= 0.0) | (abs(point) <= size.xy)))
                return {float3(0.0, 0.0, 1.0), distance};
        }
        return RayMiss;
    case GeomCapsule:
        const float3 spine = float3(0.0, 0.0, clamp(origin.z, -size.y, size.y));
        const float3 radial = origin - spine;
        if (dot(radial, radial) <= size.x * size.x)
            return {dot(radial, radial) > MinValue ? normalize(radial) : -direction, 0.0};
        return rayRoundCylinder(origin, direction, size.xy, true);
    case GeomCylinder:
        const float radius = length(origin.xy);
        if (radius <= size.x && abs(origin.z) <= size.y)
        {
            if (size.x - radius >= size.y - abs(origin.z))
                return {float3(0.0, 0.0, origin.z < 0.0 ? -1.0 : 1.0), 0.0};
            return {radius > MinValue ? float3(origin.xy / radius, 0.0) : float3(1.0, 0.0, 0.0), 0.0};
        }
        return rayRoundCylinder(origin, direction, size.xy, false);
    case GeomEllipsoid:
        const float3 inverse2 = 1.0 / (size * size);
        const float3 scaled = inverse2 * direction;
        if (dot(inverse2 * origin, origin) <= 1.0)
            return {dot(origin, origin) > MinValue ? normalize(inverse2 * origin) : -direction, 0.0};
        const float ellipsoid = nearestRayRoot(rayRoots(dot(scaled, direction), dot(scaled, origin),
            dot(inverse2 * origin, origin) - 1.0));
        return ellipsoid < 0.0 ? RayMiss : RaySurface(normalize(inverse2 * (origin + ellipsoid * direction)), ellipsoid);
    case GeomBox:
        const float3 gap = size - abs(origin);
        if (all(gap >= 0.0))
        {
            const uint axis = gap.x <= gap.y && gap.x <= gap.z ? 0 : (gap.y <= gap.z ? 1 : 2);
            float3 normal = float3(0.0);
            normal[axis] = origin[axis] < 0.0 ? -1.0 : 1.0;
            return {normal, 0.0};
        }
        float faces[6];
        return rayBox(origin, direction, size, faces);
    case GeomHeightfield:
        return rayHeightfield(origin, direction, shape.type.y);
    case GeomMesh:
        return rayHull(origin, direction, shape, geometry);
    case GeomTriangleMesh:
        return rayMesh(origin, direction, shape, maximum, backFaces, subShape, assetFace);
    default:
        return RayMiss;
    }
}

RaySurface rayWorldSurface(CartesianFrame frame, PhysicsRay ray, CollisionShape shape, uint geometry,
                           inout uint subShape, out uint assetFace)
{
    const float3 offset = ray.originLimit.xyz - frame.position.xyz;
    if (shape.type.x == GeomSphere)
    {
        assetFace = NoIndex;
        const float radius = shape.size.x;
        const float squared = dot(offset, offset);
        if (squared <= radius * radius)
            return {squared > MinValue ? normalize(offset) : -normalize(ray.direction.xyz), 0.0};
        const float distance = nearestRayRoot(raySphereRoots(offset, ray.direction.xyz, radius * radius));
        if (distance < 0.0)
            return RayMiss;
        const float3 point = ray.originLimit.xyz + ray.direction.xyz * distance;
        return {normalize(point - frame.position.xyz), distance};
    }
    const RaySurface local = rayGeometrySurface(rayLocalVector(frame, offset), rayLocalVector(frame, ray.direction.xyz),
        shape, geometry, ray.originLimit.w, (ray.filter.z & RayHitBackFaces) != 0, subShape, assetFace);
    return {rotatePoint(frame, local.normal), local.distance};
}

void testRayGeometry(PhysicsRay ray, uint base, uint geometry, inout RaySurface best, inout uint4 winner)
{
    const uint4 metadata = rayGeometry[geometry];
    const uint surfaceFlags = RayIncludeStatic | RayIncludeTriggers;
    if ((metadata.y & ray.filter.x) == 0 || metadata.x == ray.filter.y ||
        (metadata.z & surfaceFlags & ~ray.filter.z) != 0)
        return;
    ray.originLimit.w = best.distance;
    uint subShape = metadata.w;
    uint assetFace;
    const RaySurface surface = rayWorldSurface(rayFrames[base + geometry], ray, rayShapes[geometry], geometry,
                                               subShape, assetFace);
    if (surface.distance >= 0.0 && (surface.distance < best.distance ||
        (surface.distance == best.distance && geometry < winner.x)))
    {
        best = surface;
        winner = uint4(geometry, metadata.x, subShape, assetFace);
    }
}

[shader("compute")]
[numthreads(RayGroupSize, 1, 1)]
void raycast_worlds(uint3 id : SV_DispatchThreadID)
{
    const uint4 settings = raySettings[0];
    if (id.x >= settings.y)
        return;
    const PhysicsRay ray = rays[(id.y % settings.z) * settings.y + id.x];
    const uint output = id.y * settings.y + id.x;
    if (ray.originLimit.w <= 0.0)
    {
        rayHits[output] = {float4(0.0, 0.0, 0.0, -1.0), uint4(NoIndex)};
        return;
    }
    RaySurface best = {float3(0.0), ray.originLimit.w};
    uint4 winner = uint4(NoIndex);
    const uint base = id.y * settings.x;
    const uint4 index = rayIndex[0];
    for (uint plane = 0; plane < index.w; ++plane)
        testRayGeometry(ray, base, rayIndexAt(rayIndex, index.z, plane), best, winner);

    const uint nodeCount = rayIndex[1].x;
    const uint boundsBase = id.y * nodeCount;
    const RaySlabs slabs = raySlabs(ray.originLimit.xyz, ray.direction.xyz);
    uint node = 0;
    while (node < nodeCount)
    {
        const uint4 links = rayIndex[index.x + node];
        const RayBvhBounds bounds = rayBounds[boundsBase + node];
        const bool overlaps = rayBoundsOverlap(slabs, bounds.low.xyz, bounds.high.xyz, best.distance);
        if (overlaps && (links.y & BvhLeaf) == 0)
        {
            node = links.x;
            continue;
        }
        if (overlaps)
        {
            const uint end = links.x + (links.y & ~BvhLeaf);
            for (uint i = links.x; i < end; ++i)
                testRayGeometry(ray, base, rayIndexAt(rayIndex, index.y, i), best, winner);
        }
        node = links.z;
    }

    if (winner.x == NoIndex)
        rayHits[output] = {float4(0.0, 0.0, 0.0, -1.0), winner};
    else
        rayHits[output] = {float4(best.normal, best.distance), winner};
}
