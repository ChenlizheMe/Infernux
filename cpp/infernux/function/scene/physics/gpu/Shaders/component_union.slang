// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

// Each successful link points to a smaller slot. A caller can resume an edge
// from the returned ancestors without spinning on another invocation.
uint2 jph_join_components(RWStructuredBuffer<Atomic<uint>> parents, uint2 edge, uint quantum)
{
    for (uint n = 0u; n < quantum && edge.x != edge.y; ++n)
    {
        uint hi = max(edge.x, edge.y), lo = min(edge.x, edge.y);
        uint parent = parents[hi].load();
        if (parent == hi)
        {
            uint prior = parents[hi].compareExchange(hi, lo);
            parent = prior == hi ? lo : prior;
        }
        else
        {
            // Shortcuts follow established links. A concurrent shortcut can
            // only lower the parent further and cannot undo a completed link.
            const uint ancestor = parents[parent].load();
            if (ancestor < parent) parents[hi].min(ancestor);
            parent = ancestor;
        }
        edge = uint2(parent, lo);
    }
    return edge;
}
