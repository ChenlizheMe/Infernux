// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

// Model records 0 and 1 are the dispatch settings, record 2 indexes the
// immutable sections. The two scalar arrays use four words per record.
float3 convexVertex(StructuredBuffer<uint4> model, uint index)
{
    return asfloat(model[model[2].x + index].xyz);
}

int convexAdjacency(StructuredBuffer<uint4> model, uint index)
{
    return asint(model[model[2].y + index / 4][index % 4]);
}

uint4 convexTopology(StructuredBuffer<uint4> model, uint index)
{
    return model[model[2].z + index];
}

uint convexLink(StructuredBuffer<uint4> model, uint index)
{
    return model[model[2].w + index / 4][index % 4];
}
