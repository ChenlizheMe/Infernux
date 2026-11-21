// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "contact_publish.slang"
#include "convex_manifold.slang"
#include "linear_dispatch.slang"
#include "triangle_mesh_geometry.slang"

static const uint MeshBvhLeaf = 1u << 31;

[[vk::binding(8, 0)]] StructuredBuffer<uint4> convexModel;
[[vk::binding(9, 0)]] RWStructuredBuffer<uint4> convexScratch;
[[vk::binding(10, 0)]] RWStructuredBuffer<uint> convexHorizon;
[[vk::binding(11, 0)]] RWStructuredBuffer<Atomic<uint>> convexStatus;

struct MeshContact
{
    float3 point;
    float distance;
    float3 normal;
    uint subshape;
};

struct MeshManifold
{
    MeshContact points[4];
    uint count;
};

float meshContactSpacing(MeshContact a, MeshContact b)
{
    const float3 offset = a.point - b.point;
    const float3 normal = a.normal - b.normal;
    return dot(offset, offset) + dot(normal, normal);
}

void includeMeshContact(inout MeshManifold manifold, MeshContact next)
{
    for (uint i = 0; i < manifold.count; ++i)
    {
        if (meshContactSpacing(manifold.points[i], next) <= 1.0e-8)
        {
            if (next.distance < manifold.points[i].distance ||
                (next.distance == manifold.points[i].distance && next.subshape < manifold.points[i].subshape))
                manifold.points[i] = next;
            return;
        }
    }
    if (manifold.count < 4)
    {
        manifold.points[manifold.count++] = next;
        return;
    }

    MeshContact candidates[5] = {manifold.points[0], manifold.points[1], manifold.points[2],
                                 manifold.points[3], next};
    uint deepest = 0;
    for (uint i = 1; i < 5; ++i)
        if (candidates[i].distance < candidates[deepest].distance)
            deepest = i;

    uint drop = deepest == 4 ? 0 : 4;
    float best = -1.0;
    for (uint removed = 0; removed < 5; ++removed)
    {
        if (removed == deepest)
            continue;
        float coverage = ConvexFarDistance;
        for (uint a = 0; a < 4; ++a)
            for (uint b = a + 1; b < 5; ++b)
                if (a != removed && b != removed)
                    coverage = min(coverage, meshContactSpacing(candidates[a], candidates[b]));
        if (coverage > best || (coverage == best && candidates[removed].distance > candidates[drop].distance))
        {
            best = coverage;
            drop = removed;
        }
    }
    if (drop < 4)
        manifold.points[drop] = next;
}

float3 meshContactNormal(float3 a, float3 b, float3 c, float3 witness, float3 normal,
                         float3 face, uint activeEdges)
{
    if (activeEdges == 7 || dot(normal, face) > 0.99985)
        return normal;
    float3 weights;
    convexTriangleCoordinates(a, b, c, witness, weights);
    const uint incident = (weights.z <= 1.0e-4 ? 1u : 0u) |
                          (weights.x <= 1.0e-4 ? 2u : 0u) |
                          (weights.y <= 1.0e-4 ? 4u : 0u);
    return (incident & activeEdges) != 0 ? normal : face;
}

void collideMeshTriangle(uint world, ConvexGeometry shape, uint2 hullTopology, float3 a, float3 b, float3 c,
                          uint edges, uint identity, ContactMaterial material, uint4 options,
                          ConvexPolytope workspace, inout MeshManifold contacts)
{
    const float3 face = normalize(cross(b - a, c - a));
    if (dot(face, shape.frame.position.xyz - a) < 0.0)
        return;
    const float allowedDistance = material.parameters.x + material.parameters.y;
    if (shape.shape.type.x == GeomSphere)
    {
        const float3 center = shape.frame.position.xyz;
        float3 vertices[3] = {a - center, b - center, c - center};
        const float3 weights = projectConvexTriangle(vertices);
        const float3 witness = weights.x * a + weights.y * b + weights.z * c;
        const float3 delta = center - witness;
        const float length2 = dot(delta, delta);
        const float radius = shape.shape.size.x;
        const float distance = sqrt(length2) - radius;
        if (distance >= allowedDistance)
            return;
        const float3 normal = meshContactNormal(a, b, c, witness, length2 > 0.0 ? normalize(delta) : face,
                                                face, edges);
        const float3 point = 0.5 * (witness + center - radius * normal);
        includeMeshContact(contacts, {point, dot(delta, normal) - radius, normal, identity});
        return;
    }

    const float3 origin = a;
    const CartesianFrame triangleFrame = {float4((b + c - 2.0 * a) / 3.0, 0.0), float4(0.0),
                                          float4(b - a, 0.0), float4(c - a, 0.0)};
    const CollisionShape triangleShape = {float4(0.0), uint4(GeomTriangle, 0, 0, 0)};
    const float margin = material.parameters.x;
    ConvexGeometry triangle = {triangleFrame, triangleShape, margin, -1};
    shape.frame.position.xyz -= origin;
    shape.margin = margin;
    const float tolerance = asfloat(options.x);
    bool overlap;
    const ConvexDistance seed = convexDistancePhase(triangle, shape, tolerance, material.parameters.y, options.y,
                                                    convexModel, overlap);
    ConvexPenetration result = {seed.first, seed.second, seed.distance, NoIndex, 0};
    if (overlap)
        result = expandConvexPenetration(seed, triangle, shape, tolerance, options.z, workspace, convexModel,
                                         convexScratch, convexHorizon);
    if (result.status >= EpaCapacity)
        convexStatus[world].or(1u << result.status);
    if (result.distance >= material.parameters.y)
        return;

    const float direction = result.distance <= 0.0 ? 1.0 : -1.0;
    const float3 delta = direction * (result.first - result.second);
    const float3 axis = dot(delta, delta) > 0.0 ? normalize(delta) : face;
    const float3 normal = meshContactNormal(a, b, c, result.first + origin, axis, face, edges);
    if (dot(normal, face) < -tolerance)
        return;

    ConvexManifold clipped;
    clipped.points[0] = float4(0.5 * (result.first + result.second), result.distance);
    clipped.direction = result.first - result.second;
    clipped.count = 1;
    clipped.status = 0;
    if (result.face != NoIndex && (convexFaceGeometry(shape.shape.type.x) || shape.shape.type.x == GeomCapsule))
    {
        const uint3 vertices = unpackEpaFace(convexScratch[workspace.faceBase + result.face].w);
        ConvexSimplexVertex support[3];
        for (uint j = 0; j < 3; ++j)
            support[j] = loadEpaVertex(workspace, vertices[j], convexScratch);
        buildConvexManifold(clipped, support, triangle, shape, result.first, result.second, uint2(NoIndex),
                            hullTopology, workspace.faceBase + workspace.faceCapacity, options.w,
                            convexModel, convexScratch);
    }
    if (clipped.status != 0)
        convexStatus[world].or(1u << clipped.status);
    for (uint j = 0; j < clipped.count; ++j)
    {
        const float4 point = clipped.points[j];
        const float signedDistance = point.w * max(dot(axis, normal), 0.0) + margin;
        if (signedDistance < allowedDistance)
            includeMeshContact(contacts, {point.xyz + origin, signedDistance, normal, identity});
    }
}

void collideMeshPair(CollisionCandidate candidate, CollisionShape other, CollisionShape mesh, uint4 settings,
                     uint4 options, ConvexPolytope workspace)
{
    const uint4 pair = candidate.pair();
    const uint world = candidate.world;
    const uint base = world * settings.x;
    const CartesianFrame frame = geometryFrames[base + pair.y];
    const ConvexGeometry shape = {relativeCollisionFrame(frame, geometryFrames[base + pair.x]), other, 0.0, -1};
    const ContactMaterial material = resolveMaterial(pair, settings.x);
    const float inflate = material.parameters.x + material.parameters.y;
    const float3 x = float3(1, 0, 0), y = float3(0, 1, 0), z = float3(0, 0, 1);
    const float3 upper = float3(convexSupport(shape, x, convexModel).point.x,
                               convexSupport(shape, y, convexModel).point.y,
                               convexSupport(shape, z, convexModel).point.z) + inflate;
    const float3 lower = float3(convexSupport(shape, -x, convexModel).point.x,
                               convexSupport(shape, -y, convexModel).point.y,
                               convexSupport(shape, -z, convexModel).point.z) - inflate;
    const uint2 hullTopology = other.type.x == GeomMesh ? convexTopology(convexModel, pair.x).xy : uint2(NoIndex);
    const TriangleMeshTree tree = triangleMeshTree(convexModel, mesh);

    MeshManifold contacts;
    contacts.count = 0;
    uint node = 0;
    while (node < tree.nodeCount)
    {
        const uint address = tree.nodeBase + 2 * node;
        const uint4 lo = convexModel[address];
        const uint4 hi = convexModel[address + 1];
        const bool overlaps = all(upper >= asfloat(lo.xyz)) && all(lower <= asfloat(hi.xyz));
        if ((hi.w & MeshBvhLeaf) == 0)
        {
            node = overlaps ? lo.w : hi.w;
            continue;
        }
        if (overlaps)
        {
            const uint end = lo.w + (hi.w & ~MeshBvhLeaf);
            for (uint i = lo.w; i < end; ++i)
            {
                const uint4 triangle = convexModel[tree.faceBase + i];
                const float3 a = convexVertex(convexModel, triangle.x);
                const float3 b = convexVertex(convexModel, triangle.y);
                const float3 c = convexVertex(convexModel, triangle.z);
                if (any(upper < min(a, min(b, c))) || any(lower > max(a, max(b, c))))
                    continue;
                const uint subshape = triangleMeshSubShape(convexModel, tree, triangle.w);
                const uint edges = triangleMeshFeature(convexModel, tree, triangle.w, 1);
                collideMeshTriangle(world, shape, hullTopology, a, b, c, edges, subshape,
                                    material, options, workspace, contacts);
            }
        }
        ++node;
    }
    for (uint i = 0; i < contacts.count; ++i)
    {
        const MeshContact contact = contacts.points[i];
        const float3 point = frame.position.xyz + rotatePoint(frame, contact.point);
        const ContactBasis basis = makeBasis(-rotatePoint(frame, contact.normal));
        writeContact(world, pair, i, contact.distance, point, basis, material, settings.z, contact.subshape);
    }
}

[shader("compute")]
[numthreads(ConvexCollisionThreads, 1, 1)]
void collide_triangle_mesh(uint3 id : SV_DispatchThreadID)
{
    const uint4 settings = contactSettings[0];
    const uint pairCount = collisionWorkCount(settings);
    const uint worker = linearWorkIndex<ConvexCollisionThreads>(id);
    if (worker >= min(pairCount, settings.w))
        return;
    const uint4 layout = convexModel[0];
    const uint4 options = convexModel[1];
    const uint base = worker * (2u * layout.y + layout.z + 5u * options.w);
    const ConvexPolytope workspace = {base, base + 2u * layout.y, worker * layout.w, layout.y, layout.z, layout.w,
                                     0, 0, float3(0.0)};
    for (uint item = worker; item < pairCount; item += settings.w)
    {
        const CollisionCandidate candidate = collisionPair[item];
        const CollisionShape other = shapes[candidate.firstGeometry];
        const CollisionShape mesh = shapes[candidate.secondGeometry];
        if (mesh.type.x == GeomTriangleMesh && other.type.x >= GeomSphere && other.type.x <= GeomMesh)
            collideMeshPair(candidate, other, mesh, settings, options, workspace);
    }
}
