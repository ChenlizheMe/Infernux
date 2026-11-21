// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

struct TransposedConstraint
{
    uint row;
    float value;
};

// Per-world offsets, including the end, followed by one counting/scatter cursor per DOF.
[[vk::binding(23, 0)]] RWStructuredBuffer<Atomic<uint>> transposeAddress;
[[vk::binding(24, 0)]] RWStructuredBuffer<TransposedConstraint> transposeEntries;

uint transposeOffsetBase(uint world, uint dofs)
{
    return world * (2 * dofs + 1);
}

uint transposeCursorBase(uint world, uint dofs)
{
    return transposeOffsetBase(world, dofs) + dofs + 1;
}
