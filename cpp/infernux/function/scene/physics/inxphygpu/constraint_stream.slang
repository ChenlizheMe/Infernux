// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

[[vk::binding(0, 0)]] StructuredBuffer<uint4> settings;
[[vk::binding(5, 0)]] RWStructuredBuffer<Atomic<uint>> rowCounts;
[[vk::binding(7, 0)]] RWStructuredBuffer<ConstraintRow> rows;
[[vk::binding(8, 0)]] RWStructuredBuffer<ConstraintColumn> columns;

uint2 reserveConstraintBlock(uint world, uint rowCount, uint width)
{
    const uint firstRow = rowCounts[2 * world].add(rowCount);
    const uint firstColumn = rowCounts[2 * world + 1].add(rowCount * width);
    const uint4 limits = settings[0];
    return uint2(firstRow + rowCount <= limits.z && firstColumn + rowCount * width <= limits.w ? firstRow : NoIndex,
                 firstColumn);
}
