// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/smooth.py factor_m and solve_m.

#include "model.slang"

static const uint CompactMassBlock = 0xfffffffeu;
static const uint TileWidth = 64;

struct DenseMassBlock
{
    uint firstDof;
    uint dofCount;
    uint factorOffset;
    uint gatherOffset;
};

struct SparseMassBlock
{
    uint firstLevel;
    uint levelCount;
    uint firstDof;
    uint dofCount;
};

struct SparseMassRow
{
    uint dof;
    uint firstContribution;
    uint contributionCount;
};

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> factorLayout; // factor row size, dense region size, small block count
[[vk::binding(2, 0)]] StructuredBuffer<DenseMassBlock> smallBlocks;
[[vk::binding(3, 0)]] StructuredBuffer<DenseMassBlock> tileBlocks;
[[vk::binding(4, 0)]] StructuredBuffer<uint> tileElements;
[[vk::binding(5, 0)]] StructuredBuffer<uint> sparseTopology;
[[vk::binding(9, 0)]] StructuredBuffer<DofModel> dofs;
[[vk::binding(10, 0)]] StructuredBuffer<float> massMatrix;
[[vk::binding(11, 0)]] RWStructuredBuffer<float> factors;
[[vk::binding(12, 0)]] RWStructuredBuffer<float> inverseDiagonal;
[[vk::binding(13, 0)]] StructuredBuffer<float> rhs;
[[vk::binding(14, 0)]] RWStructuredBuffer<float> solution;
[[vk::binding(15, 0)]] StructuredBuffer<uint> symmetricAdjacency;
[[vk::binding(16, 0)]] StructuredBuffer<float> massVectorInput;
[[vk::binding(17, 0)]] RWStructuredBuffer<float> massVectorOutput;

[shader("compute")]
[numthreads(64, 1, 1)]
void multiply_packed_mass(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions dimensions = model[0];
    const uint dofCount = dimensions.state.y;
    if (id.x >= dofCount)
        return;
    const uint vectorBase = id.y * dofCount;
    const uint matrixBase = id.y * dimensions.layout.w;
    float product = 0.0;
    for (uint i = symmetricAdjacency[id.x]; i < symmetricAdjacency[id.x + 1]; ++i)
    {
        const uint offset = dofCount + 1 + 2 * i;
        const uint column = symmetricAdjacency[offset];
        const uint massIndex = symmetricAdjacency[offset + 1];
        product += massMatrix[matrixBase + massIndex] * massVectorInput[vectorBase + column];
    }
    massVectorOutput[vectorBase + id.x] = product;
}

groupshared float massTile[TileWidth * (TileWidth + 1) / 2];
groupshared float massVector[TileWidth];
groupshared uint massLoopCount;

uint massUniformCount(uint lane, uint count)
{
    __target_switch
    {
    case wgsl:
        if (lane == 0)
            massLoopCount = count;
        return workgroupUniformLoad(massLoopCount);
    default:
        return count;
    }
}

void syncMassStorage()
{
    __target_switch
    {
    case wgsl: __intrinsic_asm "storageBarrier()";
    default: DeviceMemoryBarrierWithGroupSync();
    }
}

uint tileEntry(uint row, uint column)
{
    return column * (column + 1) / 2 + row;
}

// The header holds the word offsets of the level, row, and contribution sections.
// Blocks begin immediately after the header.
SparseMassBlock sparseBlock(uint index)
{
    const uint word = 4 + index * 4;
    return {sparseTopology[word], sparseTopology[word + 1], sparseTopology[word + 2], sparseTopology[word + 3]};
}

uint2 sparseLevel(uint index)
{
    const uint word = sparseTopology[0] + index * 2;
    return uint2(sparseTopology[word], sparseTopology[word + 1]);
}

SparseMassRow sparseRow(uint index)
{
    const uint word = sparseTopology[1] + index * 3;
    return {sparseTopology[word], sparseTopology[word + 1], sparseTopology[word + 2]};
}

uint2 sparseContribution(uint index)
{
    const uint word = sparseTopology[2] + index * 2;
    return uint2(sparseTopology[word], sparseTopology[word + 1]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void factor_small_mass(uint3 id : SV_DispatchThreadID)
{
    const uint4 layout = factorLayout[0];
    if (id.x >= layout.z)
        return;
    const DenseMassBlock block = smallBlocks[id.x];
    const uint n = block.dofCount;
    const uint matrix = id.y * model[0].layout.w + dofs[block.firstDof].indices.z;
    if (block.factorOffset == CompactMassBlock)
    {
        const uint row = id.y * model[0].state.y + block.firstDof;
        for (uint i = 0; i < n; ++i)
            inverseDiagonal[row + i] = 1.0 / massMatrix[matrix + i];
        return;
    }

    const uint factor = id.y * layout.x + block.factorOffset;
    for (uint i = 0; i < n; ++i)
    {
        float diagonal = massMatrix[matrix + i * (i + 1) / 2 + i];
        for (uint k = 0; k < i; ++k)
        {
            const float value = factors[factor + k * n + i];
            diagonal -= value * value;
        }
        const float root = sqrt(diagonal);
        factors[factor + i * n + i] = root;
        const float reciprocal = 1.0 / root;
        for (uint j = i + 1; j < n; ++j)
        {
            float value = massMatrix[matrix + j * (j + 1) / 2 + i];
            for (uint k = 0; k < i; ++k)
                value -= factors[factor + k * n + i] * factors[factor + k * n + j];
            factors[factor + i * n + j] = value * reciprocal;
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void factor_tiled_mass(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    const DenseMassBlock block = tileBlocks[group.x];
    const uint n = massUniformCount(lane, block.dofCount);
    const uint matrix = group.y * model[0].layout.w;
    const uint factor = group.y * factorLayout[0].x + block.factorOffset;
    for (uint row = 0; row <= lane && lane < n; ++row)
    {
        const uint address = tileElements[block.gatherOffset + row * n + lane];
        massTile[tileEntry(row, lane)] = address == NoIndex ? 0.0 : massMatrix[matrix + address];
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint i = 0; i < n; ++i)
    {
        const uint diagonal = tileEntry(i, i);
        if (lane == i)
            massTile[diagonal] = sqrt(massTile[diagonal]);
        GroupMemoryBarrierWithGroupSync();
        if (lane > i && lane < n)
            massTile[tileEntry(i, lane)] /= massTile[diagonal];
        GroupMemoryBarrierWithGroupSync();
        for (uint row = i + 1; row <= lane && lane < n; ++row)
            massTile[tileEntry(row, lane)] -= massTile[tileEntry(i, row)] * massTile[tileEntry(i, lane)];
        GroupMemoryBarrierWithGroupSync();
    }
    for (uint row = 0; row <= lane && lane < n; ++row)
        factors[factor + row * n + lane] = massTile[tileEntry(row, lane)];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void factor_sparse_mass(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    const SparseMassBlock block = sparseBlock(group.x);
    const uint4 layout = factorLayout[0];
    const uint matrix = group.y * model[0].layout.w;
    const uint factor = group.y * layout.x + layout.y;
    const uint dofBase = group.y * model[0].state.y;
    const uint levelCount = massUniformCount(lane, block.levelCount);
    for (uint i = lane; i < block.dofCount; i += TileWidth)
    {
        const uint4 row = dofs[block.firstDof + i].indices;
        for (uint j = 0; j < row.w; ++j)
            factors[factor + row.z + j] = massMatrix[matrix + row.z + j];
    }
    syncMassStorage();

    // Each destination row owns its reduction. A descendant contributes at most once at a given level.
    for (uint level = levelCount; level != 0; --level)
    {
        const uint2 range = sparseLevel(block.firstLevel + level - 1);
        for (uint i = lane; i < range.y; i += TileWidth)
        {
            const SparseMassRow rowInfo = sparseRow(range.x + i);
            const uint4 row = dofs[rowInfo.dof].indices;
            for (uint j = 0; j < rowInfo.contributionCount; ++j)
            {
                const uint2 update = sparseContribution(rowInfo.firstContribution + j);
                const uint4 source = dofs[update.x].indices;
                const float ratio = factors[factor + update.y] / factors[factor + source.z + source.w - 1];
                for (uint entry = 0; entry < row.w; ++entry)
                    factors[factor + row.z + entry] -= factors[factor + source.z + entry] * ratio;
                factors[factor + update.y] = ratio;
            }
            inverseDiagonal[dofBase + rowInfo.dof] = 1.0 / factors[factor + row.z + row.w - 1];
        }
        syncMassStorage();
    }
}

void solveDenseMass(DenseMassBlock block, uint world)
{
    const uint n = block.dofCount;
    const uint row = world * model[0].state.y + block.firstDof;
    if (block.factorOffset == CompactMassBlock)
    {
        for (uint i = 0; i < n; ++i)
            solution[row + i] = rhs[row + i] * inverseDiagonal[row + i];
        return;
    }
    const uint factor = world * factorLayout[0].x + block.factorOffset;
    for (uint i = 0; i < n; ++i)
    {
        float value = rhs[row + i];
        for (uint k = 0; k < i; ++k)
            value -= factors[factor + k * n + i] * solution[row + k];
        solution[row + i] = value / factors[factor + i * n + i];
    }
    for (uint i = n; i != 0; --i)
    {
        const uint column = i - 1;
        float value = solution[row + column];
        for (uint k = i; k < n; ++k)
            value -= factors[factor + column * n + k] * solution[row + k];
        solution[row + column] = value / factors[factor + column * n + column];
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void solve_small_mass(uint3 id : SV_DispatchThreadID)
{
    if (id.x < factorLayout[0].z)
        solveDenseMass(smallBlocks[id.x], id.y);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void solve_tiled_mass(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    const DenseMassBlock block = tileBlocks[group.x];
    const uint n = massUniformCount(lane, block.dofCount);
    const uint row = group.y * model[0].state.y + block.firstDof;
    const uint factor = group.y * factorLayout[0].x + block.factorOffset;
    if (lane < n)
        massVector[lane] = rhs[row + lane];
    GroupMemoryBarrierWithGroupSync();
    for (uint i = 0; i < n; ++i)
    {
        if (lane == i)
            massVector[i] /= factors[factor + i * n + i];
        GroupMemoryBarrierWithGroupSync();
        if (lane > i && lane < n)
            massVector[lane] -= factors[factor + i * n + lane] * massVector[i];
        GroupMemoryBarrierWithGroupSync();
    }
    for (uint i = n; i != 0; --i)
    {
        const uint column = i - 1;
        if (lane == column)
            massVector[column] /= factors[factor + column * n + column];
        GroupMemoryBarrierWithGroupSync();
        if (lane < column)
            massVector[lane] -= factors[factor + lane * n + column] * massVector[column];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane < n)
        solution[row + lane] = massVector[lane];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void solve_sparse_mass(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    const SparseMassBlock block = sparseBlock(group.x);
    const uint4 layout = factorLayout[0];
    const uint factor = group.y * layout.x + layout.y;
    const uint base = group.y * model[0].state.y;
    const uint levelCount = massUniformCount(lane, block.levelCount);
    for (uint level = levelCount; level != 0; --level)
    {
        const uint2 range = sparseLevel(block.firstLevel + level - 1);
        for (uint i = lane; i < range.y; i += TileWidth)
        {
            const SparseMassRow row = sparseRow(range.x + i);
            float value = rhs[base + row.dof];
            for (uint j = 0; j < row.contributionCount; ++j)
            {
                const uint2 update = sparseContribution(row.firstContribution + j);
                value -= factors[factor + update.y] * solution[base + update.x];
            }
            solution[base + row.dof] = value;
        }
        syncMassStorage();
    }
    for (uint level = 0; level < levelCount; ++level)
    {
        const uint2 range = sparseLevel(block.firstLevel + level);
        for (uint i = lane; i < range.y; i += TileWidth)
        {
            const uint row = sparseRow(range.x + i).dof;
            const uint4 dof = dofs[row].indices;
            uint parent = dof.y;
            float value = solution[base + row] * inverseDiagonal[base + row];
            for (uint entry = dof.w - 1; entry != 0; --entry)
            {
                value -= factors[factor + dof.z + entry - 1] * solution[base + parent];
                parent = dofs[parent].indices.y;
            }
            solution[base + row] = value;
        }
        syncMassStorage();
    }
}
