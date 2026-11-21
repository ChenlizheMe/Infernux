// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from MuJoCo Warp _src/collision_convex.py and _src/collision_gjk.py.

#include "convex_penetration.slang"
#include "contact_publish.slang"
#include "heightfield_geometry.slang"
#include "linear_dispatch.slang"

static const uint HeightfieldCapacityExceeded = 6;
static const float HeightfieldContactSpacing = 1.0e-3;

[[vk::binding(8, 0)]] StructuredBuffer<uint4> convexModel;
[[vk::binding(9, 0)]] RWStructuredBuffer<uint4> convexScratch;
[[vk::binding(10, 0)]] RWStructuredBuffer<uint> convexHorizon;
[[vk::binding(11, 0)]] RWStructuredBuffer<Atomic<uint>> convexStatus;

CartesianFrame heightfieldLocalFrame(CartesianFrame terrain, CartesianFrame geometry)
{
    const float3 x = inverseRotatePoint(terrain, frameAxis(geometry, 0));
    const float3 y = inverseRotatePoint(terrain, frameAxis(geometry, 1));
    const float3 z = inverseRotatePoint(terrain, frameAxis(geometry, 2));
    return {float4(inverseRotatePoint(terrain, geometry.position.xyz - terrain.position.xyz), 0.0),
            float4(x.x, y.x, z.x, 0.0), float4(x.y, y.y, z.y, 0.0), float4(x.z, y.z, z.z, 0.0)};
}

ConvexGeometry heightfieldPrism(float3 a, float3 b, float3 c, float baseDepth)
{
    const float3 center = (float3(a.xy, -baseDepth) + float3(b.xy, -baseDepth) +
                           float3(c.xy, -baseDepth) + a + b + c) * (1.0 / 6.0);
    const CartesianFrame frame = {float4(center, 0.0), float4(a, 0.0), float4(b, 0.0), float4(c, 0.0)};
    const CollisionShape shape = {float4(0.0, 0.0, 0.0, baseDepth), uint4(GeomHeightfield, 0u, 0u, 0u)};
    return {frame, shape, 0.0, -1};
}

ConvexPenetration heightfieldSurfaceWitness(ConvexPenetration result, ConvexGeometry prism,
                                           ConvexGeometry shape, float radius)
{
    const float3 a = prism.frame.row0.xyz;
    const float3 b = prism.frame.row1.xyz;
    const float3 c = prism.frame.row2.xyz;
    result.second = convexSupport(shape, result.second, convexModel).point - float3(0.0, 0.0, radius);
    float3 weights;
    convexTriangleCoordinates(a, b, c, result.second, weights);
    if (all(weights > 0.0))
        result.first = a * weights.x + b * weights.y + c * weights.z;
    else
        result.first = float3(result.second.xy, weights.x > 0.0 ? a.z : (weights.y > 0.0 ? b.z : c.z));
    result.distance = -length(result.first - result.second);
    return result;
}

ConvexPenetration heightfieldPrismContact(ConvexGeometry prism, ConvexGeometry shape, uint4 options,
                                         ConvexPolytope workspace)
{
    const float tolerance = asfloat(options.x);
    ConvexGeometry core = shape;
    const float radius = shrinkConvexGeometry(core, false);
    if (radius > 0.0)
    {
        const ConvexDistance seed = gjkDistance(prism, core, tolerance, radius, options.y, convexModel);
        if (seed.distance >= ConvexFarDistance)
            return {seed.first, seed.second, seed.distance, NoIndex, 0u};
        if (seed.distance > tolerance)
        {
            ConvexPenetration result = {seed.first, seed.second, seed.distance, NoIndex, 0u};
            for (uint i = 1; i < seed.count; ++i)
            {
                if (seed.simplex[i].indices.x != seed.simplex[0].indices.x)
                    return heightfieldSurfaceWitness(result, prism, core, radius);
            }
            const float3 normal = normalize(seed.second - seed.first);
            result.second -= radius * normal;
            result.distance -= radius;
            return result;
        }
    }

    const ConvexDistance seed = gjkDistance(prism, shape, tolerance, 0.0, options.y, convexModel);
    if (seed.distance > tolerance || seed.count < 2 || seed.separated)
        return {seed.first, seed.second, seed.distance, NoIndex, 0u};

    ConvexPenetration result = expandConvexPenetration(seed, prism, shape, tolerance, options.z, workspace,
        convexModel, convexScratch, convexHorizon);
    if (result.face != NoIndex)
    {
        const uint3 ids = unpackEpaFace(convexScratch[workspace.faceBase + result.face].w);
        const int a = loadEpaVertex(workspace, ids.x, convexScratch).indices.x;
        const int b = loadEpaVertex(workspace, ids.y, convexScratch).indices.x;
        const int c = loadEpaVertex(workspace, ids.z, convexScratch).indices.x;
        if (a != b || a != c)
        {
            if (radius <= 0.0)
                result.second = normalize(result.second);
            return heightfieldSurfaceWitness(result, prism, core, radius);
        }
    }
    return result;
}

float4 loadHeightfieldPoint(uint base, uint index)
{
    return asfloat(convexScratch[base + 2u * index]);
}

void publishHeightfieldPoint(uint world, uint4 pair, uint ordinal, uint base, uint index,
                             ContactMaterial material, uint capacity)
{
    const float4 point = loadHeightfieldPoint(base, index);
    const float3 normal = asfloat(convexScratch[base + 2u * index + 1u].xyz);
    writeContact(world, pair, ordinal, point.w, point.xyz, makeBasis(normal), material, capacity);
}

void publishHeightfieldManifold(uint world, uint4 pair, uint count, uint deepest, uint base,
                                ContactMaterial material, uint capacity)
{
    publishHeightfieldPoint(world, pair, 0, base, deepest, material, capacity);
    uint selected[4] = {deepest, NoIndex, NoIndex, NoIndex};
    const float3 start = loadHeightfieldPoint(base, deepest).xyz;
    const float3 normal = asfloat(convexScratch[base + 2u * deepest + 1u].xyz);
    float3 end = start;
    float3 third = start;
    for (uint point = 1; point < 4 && point < count; ++point)
    {
        uint best = NoIndex;
        float span = -1.0;
        const float3 edge = cross(normal, start - end);
        const float3 firstTriangleEdge = cross(normal, start - third);
        const float3 secondTriangleEdge = cross(normal, end - third);
        for (uint i = 0; i < count; ++i)
        {
            if (i == selected[0] || i == selected[1] || i == selected[2])
                continue;
            const float3 candidate = loadHeightfieldPoint(base, i).xyz;
            float score;
            if (point == 1)
                score = length(candidate - start);
            else if (point == 2)
                score = abs(dot(candidate - start, edge));
            else
                score = abs(dot(candidate - start, firstTriangleEdge)) + abs(dot(end - candidate, secondTriangleEdge));
            if (score > span)
            {
                best = i;
                span = score;
            }
        }
        if (best == NoIndex || (span > 0.0 && span < HeightfieldContactSpacing))
            break;
        selected[point] = best;
        const float3 chosen = loadHeightfieldPoint(base, best).xyz;
        if (point == 1)
            end = chosen;
        else if (point == 2)
            third = chosen;
        publishHeightfieldPoint(world, pair, point, base, best, material, capacity);
    }
}

void collideHeightfieldPair(uint world, uint4 pair, CollisionShape terrain, CollisionShape other, uint4 settings,
                            uint4 options, ConvexPolytope workspace, uint candidateBase, uint candidateCapacity)
{
    const uint base = world * settings.x;
    const CartesianFrame terrainFrame = geometryFrames[base + pair.x];
    const CartesianFrame local = heightfieldLocalFrame(terrainFrame, geometryFrames[base + pair.y]);
    ConvexGeometry shape = {local, other, 0.0, -1};
    const HeightfieldGrid grid = heightfieldGrid(convexModel, terrain.type.y);
    const float4 size = grid.size;

    const float3 x = float3(1.0, 0.0, 0.0), y = float3(0.0, 1.0, 0.0), z = float3(0.0, 0.0, 1.0);
    const float3 upper = float3(convexSupport(shape, x, convexModel).point.x,
                               convexSupport(shape, y, convexModel).point.y,
                               convexSupport(shape, z, convexModel).point.z);
    const float3 lower = float3(convexSupport(shape, -x, convexModel).point.x,
                               convexSupport(shape, -y, convexModel).point.y,
                               convexSupport(shape, -z, convexModel).point.z);
    const float filterMargin = materials[pair.x].parameters.x + materials[pair.y].parameters.x;
    if (any(lower - filterMargin > size.xyz) || any(upper + filterMargin < -float3(size.xy, size.w)))
        return;

    const float2 intervals = float2(grid.layout.yx - 1u);
    const float2 gridScale = 0.5 * intervals / size.xy;
    const int2 begin = max(int2(0), int2(floor((lower.xy + size.xy) * gridScale)));
    const int2 end = min(int2(intervals), int2(ceil((upper.xy + size.xy) * gridScale)));
    if (any(begin >= end))
        return;
    const float2 step = 2.0 * size.xy / intervals;
    const ContactMaterial material = resolveMaterial(pair, settings.x);
    const float margin = material.parameters.x;
    shape.margin = margin;
    float minimum = ConvexFarDistance;
    uint deepest = 0;
    uint count = 0;
    for (int row = begin.y; row < end.y && count < candidateCapacity; ++row)
    {
        float3 a = heightfieldVertex(convexModel, grid, uint2(begin.x, row + 1), step, margin);
        float3 b = heightfieldVertex(convexModel, grid, uint2(begin.x, row), step, margin);
        for (int col = begin.x + 1; col <= end.x && count < candidateCapacity; ++col)
        {
            for (uint corner = 0; corner < 2 && count < candidateCapacity; ++corner)
            {
                const float3 c = heightfieldVertex(convexModel, grid, uint2(col, row + 1 - int(corner)), step, margin);
                const ConvexGeometry prism = heightfieldPrism(a, b, c, size.w);
                const float height = max(a.z, max(b.z, c.z));
                a = b;
                b = c;
                if (height < lower.z)
                    continue;

                const ConvexPenetration contact = heightfieldPrismContact(prism, shape, options, workspace);
                if (contact.status >= EpaCapacity)
                    convexStatus[world].or(1u << contact.status);
                if (contact.distance >= margin + material.parameters.y)
                    continue;
                const float3 point = terrainFrame.position.xyz + rotatePoint(terrainFrame, 0.5 * (contact.first + contact.second));
                const float3 direction = contact.first - contact.second;
                const float3 normal = rotatePoint(terrainFrame,
                    dot(direction, direction) > 0.0 ? normalize(direction) : float3(0.0, 0.0, 1.0));
                convexScratch[candidateBase + 2u * count] = asuint(float4(point, contact.distance));
                convexScratch[candidateBase + 2u * count + 1u] = asuint(float4(normal, 0.0));
                if (contact.distance < minimum)
                {
                    minimum = contact.distance;
                    deepest = count;
                }
                ++count;
                if (count == candidateCapacity && (corner == 0 || col < end.x || row + 1 < end.y))
                    convexStatus[world].or(1u << HeightfieldCapacityExceeded);
            }
        }
    }
    if (count != 0)
        publishHeightfieldManifold(world, pair, count, deepest, candidateBase, material, settings.z);
}

[shader("compute")]
[numthreads(ConvexCollisionThreads, 1, 1)]
void collide_heightfield(uint3 id : SV_DispatchThreadID)
{
    const uint4 settings = contactSettings[0];
    const uint pairCount = collisionWorkCount(settings);
    const uint worker = linearWorkIndex<ConvexCollisionThreads>(id);
    if (worker >= min(pairCount, settings.w))
        return;
    const uint4 layout = convexModel[0];
    const uint4 options = convexModel[1];
    const uint candidateCapacity = convexModel[3].z;
    const uint base = worker * (2u * layout.y + layout.z + 2u * candidateCapacity);
    const ConvexPolytope workspace = {base, base + 2u * layout.y, worker * layout.w, layout.y, layout.z, layout.w,
                                     0, 0, float3(0.0)};
    for (uint item = worker; item < pairCount; item += settings.w)
    {
        const CollisionCandidate candidate = collisionPair[item];
        const uint4 pair = candidate.pair();
        const CollisionShape terrain = shapes[pair.x];
        const CollisionShape other = shapes[pair.y];
        if (terrain.type.x == GeomHeightfield && other.type.x >= GeomSphere && other.type.x <= GeomMesh)
            collideHeightfieldPair(candidate.world, pair, terrain, other, settings, options, workspace,
                workspace.faceBase + layout.z, candidateCapacity);
    }
}
