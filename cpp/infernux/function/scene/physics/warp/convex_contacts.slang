// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from MuJoCo Warp _src/collision_convex.py, convex pair evaluation and publication.

#include "convex_manifold.slang"
#include "contact_publish.slang"
#include "linear_dispatch.slang"

[[vk::binding(8, 0)]] StructuredBuffer<uint4> convexModel;
[[vk::binding(9, 0)]] RWStructuredBuffer<uint4> convexScratch;
[[vk::binding(10, 0)]] RWStructuredBuffer<uint> convexHorizon;
[[vk::binding(11, 0)]] RWStructuredBuffer<Atomic<uint>> convexStatus;

bool convexPair(uint a, uint b, uint flags)
{
    if (a < GeomSphere || b > GeomMesh)
        return false;
    if (a == GeomEllipsoid || b == GeomEllipsoid || b == GeomMesh)
        return true;
    if (a == GeomCapsule && b == GeomCylinder)
        return true;
    if (a == GeomCylinder && (b == GeomCylinder || b == GeomBox))
        return true;
    return a == GeomBox && b == GeomBox && (flags & UseConvexBoxes) != 0;
}

bool convexPairHasManifold(uint a, uint b, uint flags)
{
    if (a == GeomBox && b == GeomBox)
        return true;
    if ((flags & UseConvexMulticontact) == 0)
        return false;
    const bool faceA = convexFaceGeometry(a);
    const bool faceB = convexFaceGeometry(b);
    return (faceA && (faceB || b == GeomCapsule)) || (faceB && a == GeomCapsule);
}

void collidePlaneHull(uint world, uint4 pair, CartesianFrame plane, ConvexGeometry hull,
                      ContactMaterial material, uint capacity, uint clipBase, uint clipCapacity)
{
    const ContactBasis basis = makeBasis(frameAxis(plane, 2));
    const float3 normal = basis.normal;
    const float3 local = inverseRotatePoint(hull.frame, normal);
    const ConvexSupportPoint support = hullSupport(hull.shape.type, -1, -local, convexModel);
    const float3 first = hull.frame.position.xyz + rotatePoint(hull.frame, support.point);
    const float distance = dot(normal, first - plane.position.xyz);
    writeContact(world, pair, 0, distance, first - 0.5 * distance * normal, basis, material, capacity);

    const uint2 mesh = convexTopology(convexModel, pair.y).xy;
    if (mesh.x == NoIndex || distance > material.parameters.x + material.parameters.y)
        return;
    const uint2 adjacent = convexTopology(convexModel, mesh.y + uint(support.vertex)).xy;
    float alignment = 1.0;
    uint face = NoIndex;
    for (uint i = 0; i < adjacent.y; ++i)
    {
        const uint candidate = convexLink(convexModel, adjacent.x + i);
        const float value = dot(convexVertex(convexModel, convexTopology(convexModel, mesh.x + candidate).z), local);
        if (value < alignment)
        {
            alignment = value;
            face = candidate;
        }
    }
    if (face == NoIndex)
        return;
    const uint2 corners = convexTopology(convexModel, mesh.x + face).xy;
    uint firstCorner = 0;
    for (uint i = 0; i < corners.y; ++i)
    {
        if (convexLink(convexModel, corners.x + i) == uint(support.vertex))
        {
            firstCorner = i;
            break;
        }
    }
    uint4 chosen = (firstCorner + uint4(0,1,2,3)) % corners.y;
    if (corners.y > 4)
    {
        if (corners.y > clipCapacity)
        {
            convexStatus[world].or(1u << ConvexClipCapacity);
            return;
        }
        for (uint i = 0; i < corners.y; ++i)
        {
            const uint vertex = convexLink(convexModel, corners.x + i);
            storeConvexClipPoint(convexScratch, clipBase + i, convexVertex(convexModel, hull.shape.type.y + vertex));
        }
        const ConvexPolygon polygon = {clipBase, corners.y};
        uint4 indices = chosen;
        float area = convexQuadArea(clipBase, chosen, convexScratch);
        optimizeConvexQuad(polygon, indices, chosen, area, convexScratch);
    }
    uint ordinal = 1;
    for (uint i = 1; i < min(corners.y, 4); ++i)
    {
        const float3 point = convexMeshVertex(hull, convexLink(convexModel, corners.x + chosen[i]), convexModel);
        if (dot(normal, point - hull.frame.position.xyz) > 0.0)
            continue;
        const float separation = dot(normal, point - plane.position.xyz);
        writeContact(world, pair, ordinal++, separation, point - 0.5 * separation * normal, basis, material, capacity);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void clear_convex_status(uint3 id : SV_DispatchThreadID)
{
    convexStatus[id.y].store(0);
}

[shader("compute")]
[numthreads(ConvexCollisionThreads, 1, 1)]
void collide_convex(uint3 id : SV_DispatchThreadID)
{
    const uint4 settings = contactSettings[0];
    const uint pairCount = collisionWorkCount(settings);
    const uint worker = linearWorkIndex<ConvexCollisionThreads>(id);
    if (worker >= min(pairCount, settings.w))
        return;
    const uint4 layout = convexModel[0]; // EPA vertex, face and horizon capacities in yzw
    const uint flags = contactSettings[1].x;
    const uint4 options = convexModel[1]; // tolerance bits, GJK iterations, EPA iterations, max polygon sides
    const float tolerance = asfloat(options.x);
    const uint base = worker * (2 * layout.y + layout.z + 5 * options.w);
    const ConvexPolytope workspace = {base, base + 2 * layout.y, worker * layout.w, layout.y, layout.z, layout.w,
                                      0, 0, float3(0.0)};
    const uint clipBase = workspace.faceBase + layout.z;
    for (uint item = worker; item < pairCount; item += settings.w)
    {
        const CollisionCandidate candidate = collisionPair[item];
        const uint4 pair = candidate.pair();
        const CollisionShape shapeA = shapes[pair.x];
        const CollisionShape shapeB = shapes[pair.y];
        const uint typeA = shapeA.type.x;
        const uint typeB = shapeB.type.x;
        const bool planeHull = typeA == GeomPlane && typeB == GeomMesh;
        if (!planeHull && !convexPair(typeA, typeB, flags))
            continue;

        const ContactMaterial material = resolveMaterial(pair, settings.x);
        const uint geometryBase = candidate.world * settings.x;
        const float margin = material.parameters.x;
        const float gap = material.parameters.y;
        const bool sensor = asint(pair.w) >= 0;
        const float cutoff = sensor ? ConvexFarDistance : gap;
        ConvexGeometry a = {geometryFrames[geometryBase + pair.x], shapeA, margin, -1};
        ConvexGeometry b = {geometryFrames[geometryBase + pair.y], shapeB, margin, -1};
        if (planeHull)
        {
            collidePlaneHull(candidate.world, pair, a.frame, b, material, settings.z, clipBase, options.w);
            continue;
        }
        const float3 origin = a.frame.position.xyz;
        b.frame.position.xyz -= origin;
        a.frame.position.xyz = float3(0.0);
        bool needsPenetration;
        const ConvexDistance seed = convexDistancePhase(a, b, tolerance, cutoff, options.y, convexModel, needsPenetration);
        ConvexPenetration result = {seed.first, seed.second, seed.distance, NoIndex, 0};
        if (needsPenetration)
            result = expandConvexPenetration(seed, a, b, tolerance, options.z, workspace, convexModel,
                                             convexScratch, convexHorizon);
        if (result.status >= EpaCapacity)
            convexStatus[candidate.world].or(1u << result.status);
        if (result.distance >= ConvexFarDistance || (!sensor && result.distance >= gap))
            continue;

        const float distance = result.distance + margin;
        ConvexManifold manifold;
        manifold.points[0] = float4(0.5 * (result.first + result.second), result.distance);
        manifold.direction = result.first - result.second;
        manifold.count = 1;
        manifold.status = 0;
        if (convexPairHasManifold(typeA, typeB, flags) &&
            (result.face != NoIndex || (!needsPenetration && result.distance < 0.0 && seed.count > 0)))
        {
            const uint2 meshA = typeA == GeomMesh ? convexTopology(convexModel, pair.x).xy : uint2(NoIndex);
            const uint2 meshB = typeB == GeomMesh ? convexTopology(convexModel, pair.y).xy : uint2(NoIndex);
            if ((typeA != GeomMesh || meshA.x != NoIndex) && (typeB != GeomMesh || meshB.x != NoIndex))
            {
                ConvexSimplexVertex support[3];
                if (result.face != NoIndex)
                {
                    const uint3 face = unpackEpaFace(convexScratch[workspace.faceBase + result.face].w);
                    for (uint i = 0; i < 3; ++i)
                        support[i] = loadEpaVertex(workspace, face[i], convexScratch);
                }
                else
                {
                    support[0] = seed.simplex[0];
                    support[1] = seed.simplex[seed.count > 1 ? 1 : 0];
                    support[2] = seed.simplex[seed.count > 2 ? 2 : 0];
                }
                buildConvexManifold(manifold, support, a, b, result.first, result.second, meshA, meshB, clipBase,
                                    options.w, convexModel, convexScratch);
            }
        }
        if (manifold.status != 0)
        {
            convexStatus[candidate.world].or(1u << manifold.status);
            continue;
        }
        float3 normal = result.distance <= 0.0 ? manifold.direction : -manifold.direction;
        if (dot(normal, normal) == 0.0)
        {
            normal = b.frame.position.xyz;
            if (dot(normal, normal) == 0.0)
                normal = float3(1.0, 0.0, 0.0);
        }
        const ContactBasis basis = makeBasis(normal);
        for (uint i = 0; i < manifold.count; ++i)
        {
            const float4 point = manifold.points[i];
            const float separation = manifold.count > 1 ? point.w + margin : distance;
            writeContact(candidate.world, pair, i, separation, origin + point.xyz, basis, material, settings.z);
        }
    }
}
