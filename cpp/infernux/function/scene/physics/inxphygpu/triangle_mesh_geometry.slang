// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

struct TriangleMeshTree
{
    uint nodeBase;
    uint faceBase;
    uint nodeCount;
    uint featureBase;
    uint path;
    uint pathBits;
};

TriangleMeshTree triangleMeshTree(StructuredBuffer<uint4> model, CollisionShape mesh)
{
    const uint base = model[4].x;
    const uint4 layout = model[base + mesh.type.y];
    const uint bits = mesh.type.w;
    return {base + layout.x, base + layout.y, layout.z, base + layout.w, mesh.type.z & ((1u << bits) - 1u), bits};
}

uint triangleMeshFeature(StructuredBuffer<uint4> model, TriangleMeshTree tree, uint face, uint field)
{
    const uint index = 3 * face + field;
    return model[tree.featureBase + index / 4][index % 4];
}

uint triangleMeshSubShape(StructuredBuffer<uint4> model, TriangleMeshTree tree, uint face)
{
    return tree.path | (triangleMeshFeature(model, tree, face, 0) << tree.pathBits);
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
