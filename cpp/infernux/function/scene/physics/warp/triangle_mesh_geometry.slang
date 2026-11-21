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
