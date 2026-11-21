// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/solver.py sparse rigid contact Newton solve.

#include "contact_solve_common.slang"

static const uint NewtonTile = 8;
static const uint NewtonTileArea = NewtonTile * NewtonTile;

struct ContactMetric
{
    float4 low;
    float2 high;
    uint count;
    uint firstRow;
};

[[vk::binding(20, 0)]] RWStructuredBuffer<Atomic<uint>> hessianPattern;
[[vk::binding(22, 0)]] RWStructuredBuffer<Atomic<uint>> hessianValues;
[[vk::binding(23, 0)]] RWStructuredBuffer<ContactMetric> contactMetric;

uint tileCount()
{
    return (settings[1].y + NewtonTile - 1) / NewtonTile;
}

uint patternWords(uint count)
{
    return (count + 31u) / 32u;
}

uint patternBase(uint world, uint col, uint count, uint words)
{
    return world * (count * words + 5 * count + 3) + col * words;
}

uint componentOffsetBase(uint world, uint count, uint words)
{
    return patternBase(world, count, count, words) + count + 1;
}

uint factorErrorAddress(uint world, uint count, uint words)
{
    return componentOffsetBase(world, count, words) + 4 * count + 1;
}

uint inclusiveWordMask(uint bit)
{
    return 0xffffffffu >> (31u - (bit & 31u));
}

void syncNewtonStorage()
{
    __target_switch
    {
    case wgsl: __intrinsic_asm "storageBarrier()";
    default: DeviceMemoryBarrierWithGroupSync();
    }
}

void markTile(uint base, uint words, uint first, uint second)
{
    const uint col = max(first, second);
    const uint row = min(first, second);
    hessianPattern[base + col * words + row / 32u].or(1u << (row & 31u));
}

bool hasTile(uint world, uint col, uint row, uint count, uint words)
{
    const uint word = hessianPattern[patternBase(world, col, count, words) + row / 32u].load();
    return (word & (1u << (row & 31u))) != 0u;
}

uint tileAddress(uint world, uint col, uint row, uint count, uint words)
{
    const uint base = patternBase(world, col, count, words);
    const uint end = row / 32u;
    uint rank = hessianPattern[patternBase(world, count, count, words) + col].load();
    for (uint i = 0; i < end; ++i)
        rank += countbits(hessianPattern[base + i].load());
    rank += countbits(hessianPattern[base + end].load() & ((1u << (row & 31u)) - 1u));
    return (world * cgOptions[0].w + rank) * NewtonTileArea;
}

uint valueAddress(uint world, uint first, uint second, uint count, uint words)
{
    const uint col = max(first, second);
    const uint row = min(first, second);
    return tileAddress(world, col / NewtonTile, row / NewtonTile, count, words) +
           (row % NewtonTile) * NewtonTile + col % NewtonTile;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void clear_newton_pattern(uint3 id : SV_DispatchThreadID)
{
    id = solverDispatchId(id);
    const uint count = tileCount();
    const uint entries = count * patternWords(count);
    if (id.x < entries && worlds[id.y].progress.y == Running)
        hessianPattern[patternBase(id.y, 0, count, patternWords(count)) + id.x].store(0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void seed_newton_mass_pattern(uint3 id : SV_DispatchThreadID)
{
    id = solverDispatchId(id);
    const uint nv = settings[1].y;
    if (id.x >= nv || worlds[id.y].progress.y != Running)
        return;
    const uint count = tileCount();
    const uint words = patternWords(count);
    const uint base = patternBase(id.y, 0, count, words);
    const uint tile = id.x / NewtonTile;
    for (uint i = massTopology[id.x]; i < massTopology[id.x + 1]; ++i)
    {
        const uint other = massTopology[nv + 1 + 2 * i];
        if (other >= id.x)
            markTile(base, words, tile, other / NewtonTile);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void seed_newton_contact_pattern(uint3 id : SV_DispatchThreadID)
{
    id = solverDispatchId(id);
    const uint4 progress = worlds[id.y].progress;
    if (id.x >= progress.z || progress.y != Running)
        return;
    const uint4 limits = settings[0];
    const ConstraintRow row = rows[id.y * limits.z + id.x];
    if (row.layout.w != 0)
        return;

    const uint base = id.y * limits.w + row.layout.x;
    const uint count = tileCount();
    const uint words = patternWords(count);
    const uint pattern = patternBase(id.y, 0, count, words);
    // All rows emitted for a constraint block share their DOF support.
    uint previous = 0xffffffffu;
    for (uint i = 0; i < row.layout.y; ++i)
    {
        const uint first = columns[base + i].dof / NewtonTile;
        if (first == previous)
            continue;
        previous = first;
        uint last = 0xffffffffu;
        for (uint j = i; j < row.layout.y; ++j)
        {
            const uint second = columns[base + j].dof / NewtonTile;
            if (second != last)
                markTile(pattern, words, first, second);
            last = second;
        }
    }
}

void packNewtonComponents(uint world, uint count, uint words)
{
    const uint offsets = componentOffsetBase(world, count, words);
    const uint members = offsets + count + 1;
    const uint roots = members + count;
    const uint cursors = roots + count;
    for (uint col = 0; col <= count; ++col)
        hessianPattern[offsets + col].store(0);
    for (uint col = 0; col < count; ++col)
    {
        // Cholesky filled the earlier neighbors into a clique. The nearest
        // predecessor in the elimination tree therefore represents all of them.
        uint root = col;
        const uint mask = patternBase(world, col, count, words);
        for (uint w = col / 32u + 1; w > 0; --w)
        {
            const uint current = w - 1;
            uint bits = hessianPattern[mask + current].load();
            if (current == col / 32u)
                bits &= (1u << (col & 31u)) - 1u;
            if (bits != 0)
            {
                const uint parent = current * 32u + uint(firstbithigh(bits));
                root = hessianPattern[roots + parent].load();
                break;
            }
        }
        hessianPattern[roots + col].store(root);
        hessianPattern[offsets + root + 1].add(1);
    }

    uint start = 0;
    for (uint root = 0; root < count; ++root)
    {
        hessianPattern[cursors + root].store(start);
        start += hessianPattern[offsets + root + 1].load();
        hessianPattern[offsets + root + 1].store(start);
    }
    for (uint col = 0; col < count; ++col)
    {
        const uint root = hessianPattern[roots + col].load();
        const uint index = hessianPattern[cursors + root].add(1);
        hessianPattern[members + index].store(col);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void seal_newton_pattern(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    group = solverDispatchId(group);
    const uint world = group.y;
    CgWorld state = worlds[world];
    if (broadcastCg(state.progress.y, lane) != Running)
        return;

    const uint count = broadcastCg(tileCount(), lane);
    const uint words = patternWords(count);
    // Reverse symbolic elimination. Earlier neighbors of each pivot become a clique.
    for (uint p = count; p > 0; --p)
    {
        const uint pivot = p - 1;
        const uint parent = patternBase(world, pivot, count, words);
        for (uint col = lane; col < pivot; col += CgWidth)
        {
            if (!hasTile(world, pivot, col, count, words))
                continue;
            const uint target = patternBase(world, col, count, words);
            const uint end = col / 32u;
            for (uint word = 0; word < end; ++word)
            {
                const uint bits = hessianPattern[parent + word].load();
                hessianPattern[target + word].or(bits);
            }
            const uint bits = hessianPattern[parent + end].load() & inclusiveWordMask(col);
            hessianPattern[target + end].or(bits);
        }
        syncNewtonStorage();
    }

    if (lane == 0)
    {
        const uint base = patternBase(world, count, count, words);
        uint used = 0;
        for (uint col = 0; col < count; ++col)
        {
            hessianPattern[base + col].store(used);
            const uint start = patternBase(world, col, count, words);
            for (uint word = 0; word <= col / 32u; ++word)
                used += countbits(hessianPattern[start + word].load());
        }
        hessianPattern[base + count].store(used);
        if (used > cgOptions[0].w)
        {
            state.progress.y = CapacityExceeded;
            worlds[world] = state;
        }
        else
            packNewtonComponents(world, count, words);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void clear_newton_matrix(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    group = solverDispatchId(group);
    const uint world = group.y;
    if (worlds[world].progress.y != Running)
        return;
    const uint count = tileCount();
    const uint words = patternWords(count);
    const uint used = hessianPattern[patternBase(world, count, count, words) + count].load();
    const uint base = world * cgOptions[0].w * NewtonTileArea;
    if (lane == 0)
        hessianPattern[factorErrorAddress(world, count, words)].store(0);
    for (uint i = lane; i < used * NewtonTileArea; i += CgWidth)
        hessianValues[base + i].store(0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void add_newton_mass(uint3 id : SV_DispatchThreadID)
{
    id = solverDispatchId(id);
    const uint count = tileCount();
    if (id.x >= count * NewtonTile || worlds[id.y].progress.y != Running)
        return;
    const uint nv = settings[1].y;
    const uint words = patternWords(count);
    if (id.x >= nv)
    {
        hessianValues[valueAddress(id.y, id.x, id.x, count, words)].store(asuint(1.0));
        return;
    }
    const uint massBase = id.y * cgOptions[0].z;
    for (uint i = massTopology[id.x]; i < massTopology[id.x + 1]; ++i)
    {
        const uint index = nv + 1 + 2 * i;
        const uint col = massTopology[index];
        if (col >= id.x)
        {
            const uint address = valueAddress(id.y, id.x, col, count, words);
            hessianValues[address].store(asuint(massMatrix[massBase + massTopology[index + 1]]));
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void build_newton_contact_metric(uint3 id : SV_DispatchThreadID)
{
    id = solverDispatchId(id);
    const uint4 progress = worlds[id.y].progress;
    if (id.x >= progress.z || progress.y != Running)
        return;
    const uint4 limits = settings[0];
    const uint index = id.y * limits.z + id.x;
    const ConstraintRow row = rows[index];
    const uint status = evaluation[index].state;
    ContactMetric metric;
    metric.low = float4(status == QuadraticConstraint ? row.diagonal : 0.0, 0.0, 0.0, 0.0);
    metric.high = float2(0.0);
    metric.firstRow = index;
    metric.count = status == QuadraticConstraint ? 1u : 0u;
    if (status == EllipticMiddleState)
    {
        const Contact contact = contacts[id.y * limits.y + row.layout.z];
        const uint ordinal = row.layout.w;
        const uint first = index - ordinal;
        const uint dimension = contact.geometry.z;
        const float mu = contact.friction.x * asfloat(settings[2 + id.y].y);
        const float mu2 = mu * mu;
        const float dm = rows[first].diagonal / max(CgMinValue, mu2 * (1.0 + mu2));
        float norm = 0.0;
        for (uint i = 1; i < dimension; ++i)
        {
            const float value = contactFriction(contact, i - 1) * residuals[first + i];
            norm += value * value;
        }
        const float inverseLength = 1.0 / max(CgMinValue, sqrt(norm));
        const float scaledNormal = mu2 * residuals[first] * inverseLength;
        const float firstScale = ordinal == 0 ? mu : contactFriction(contact, ordinal - 1);
        const float projectedFirst = ordinal == 0 ? 0.0 :
                                     firstScale * firstScale * residuals[index] * inverseLength;
        const float normalFirst = ordinal == 0 ? firstScale : 0.0;
        for (uint i = 0; i < dimension; ++i)
        {
            const float secondScale = i == 0 ? mu : contactFriction(contact, i - 1);
            const float projectedSecond = i == 0 ? 0.0 :
                                          secondScale * secondScale * residuals[first + i] * inverseLength;
            const float normalSecond = i == 0 ? secondScale : 0.0;
            const float diagonal = i == ordinal && i != 0 ? firstScale * firstScale : 0.0;
            const float value = dm * (normalFirst * normalSecond -
                                      mu * (normalFirst * projectedSecond + normalSecond * projectedFirst) +
                                      scaledNormal * projectedFirst * projectedSecond +
                                      (mu2 - scaledNormal) * diagonal);
            if (i < 4)
                metric.low[i] = value;
            else
                metric.high[i - 4] = value;
        }
        metric.firstRow = first;
        metric.count = dimension;
    }
    contactMetric[index] = metric;
}

void addHessianValue(uint address, float value)
{
    if (value == 0.0)
        return;
    uint previous = hessianValues[address].load();
    for (;;)
    {
        const uint next = asuint(asfloat(previous) + value);
        const uint observed = hessianValues[address].compareExchange(previous, next);
        if (observed == previous)
            return;
        previous = observed;
    }
}

void addNewtonContact(uint world, uint rowIndex, uint lane)
{
    const uint4 limits = settings[0];
    const uint index = world * limits.z + rowIndex;
    const ContactMetric metric = contactMetric[index];
    if (metric.count == 0)
        return;
    const ConstraintRow row = rows[index];
    const uint base = world * limits.w;
    const uint colBase = base + row.layout.x;
    const uint count = tileCount();
    const uint words = patternWords(count);
    const uint entries = row.layout.y * (row.layout.y + 1) / 2;
    for (uint pair = lane; pair < entries; pair += CgWidth)
    {
        uint j = uint((sqrt(8.0 * float(pair) + 1.0) - 1.0) * 0.5);
        if (j * (j + 1) / 2 > pair)
            --j;
        if ((j + 1) * (j + 2) / 2 <= pair)
            ++j;
        const uint i = pair - j * (j + 1) / 2;
        const ConstraintColumn left = columns[colBase + i];
        const ConstraintColumn right = columns[colBase + j];
        float contracted = metric.low.x * right.value;
        if (metric.count > 1)
        {
            contracted = 0.0;
            const uint first = base + rows[metric.firstRow].layout.x;
            for (uint k = 0; k < metric.count; ++k)
            {
                const float coefficient = k < 4 ? metric.low[k] : metric.high[k - 4];
                contracted += coefficient * columns[first + k * row.layout.y + j].value;
            }
        }
        const uint address = valueAddress(world, left.dof, right.dof, count, words);
        addHessianValue(address, left.value * contracted);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void add_newton_contacts(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    group = solverDispatchId(group);
    const uint4 progress = worlds[group.y].progress;
    if (progress.y != Running)
        return;
    const uint groups = cgOptions[1].x;
    for (uint row = group.x; row < progress.z; row += groups)
        addNewtonContact(group.y, row, lane);
}

groupshared float diagonalTile[NewtonTileArea];
groupshared float tileVector[NewtonTile];
groupshared uint invalidDiagonal;

float loadHessian(uint address)
{
    return asfloat(hessianValues[address].load());
}

void storeHessian(uint address, float value)
{
    hessianValues[address].store(asuint(value));
}

// Reverse block Cholesky maintains the upper factor R of H = R Rᵀ.
[shader("compute")]
[numthreads(64, 1, 1)]
void factor_newton_matrix(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    group = solverDispatchId(group);
    const uint world = group.y;
    const CgWorld state = worlds[world];
    if (broadcastCg(state.progress.y, lane) != Running)
        return;
    const uint count = broadcastCg(tileCount(), lane);
    const uint words = patternWords(count);
    const uint offsets = componentOffsetBase(world, count, words);
    const uint begin = broadcastCg(hessianPattern[offsets + group.x].load(), lane);
    const uint end = broadcastCg(hessianPattern[offsets + group.x + 1].load(), lane);
    const uint members = offsets + count + 1;
    const uint x = lane / NewtonTile;
    const uint y = lane % NewtonTile;
    const uint length = end - begin;
    for (uint reverse = 0; reverse < length; ++reverse)
    {
        const uint earlier = length - 1 - reverse;
        const uint pivot = hessianPattern[members + begin + earlier].load();
        const uint diagonal = tileAddress(world, pivot, pivot, count, words);
        diagonalTile[lane] = loadHessian(diagonal + lane);
        GroupMemoryBarrierWithGroupSync();
        for (uint k = NewtonTile; k > 0; --k)
        {
            const uint inner = k - 1;
            const uint entry = inner * NewtonTile + inner;
            if (lane == 0)
            {
                const float value = diagonalTile[entry];
                invalidDiagonal = value > 0.0 && isfinite(value) ? 0u : 1u;
                diagonalTile[entry] = sqrt(value);
            }
            GroupMemoryBarrierWithGroupSync();
            if (workgroupUniformLoad(invalidDiagonal) != 0)
            {
                if (lane == 0)
                    hessianPattern[factorErrorAddress(world, count, words)].store(1);
                return;
            }
            if (y == inner && x < inner)
                diagonalTile[lane] /= diagonalTile[entry];
            GroupMemoryBarrierWithGroupSync();
            if (x <= y && y < inner)
                diagonalTile[lane] -= diagonalTile[x * NewtonTile + inner] * diagonalTile[y * NewtonTile + inner];
            GroupMemoryBarrierWithGroupSync();
        }
        if (x <= y)
            storeHessian(diagonal + lane, diagonalTile[lane]);

        for (uint i = lane; i < earlier * NewtonTile; i += CgWidth)
        {
            const uint row = hessianPattern[members + begin + i / NewtonTile].load();
            if (!hasTile(world, pivot, row, count, words))
                continue;
            const uint base = tileAddress(world, pivot, row, count, words) + (i % NewtonTile) * NewtonTile;
            for (uint k = NewtonTile; k > 0; --k)
            {
                const uint j = k - 1;
                float value = loadHessian(base + j);
                for (uint t = k; t < NewtonTile; ++t)
                    value -= loadHessian(base + t) * diagonalTile[j * NewtonTile + t];
                storeHessian(base + j, value / diagonalTile[j * NewtonTile + j]);
            }
        }
        syncNewtonStorage();

        const uint mask = patternBase(world, pivot, count, words);
        for (uint wc = 0; wc <= pivot / 32u; ++wc)
        {
            uint cols = hessianPattern[mask + wc].load();
            if (wc == pivot / 32u)
                cols &= (1u << (pivot & 31u)) - 1u;
            while (cols != 0)
            {
                const uint col = wc * 32u + uint(firstbitlow(cols));
                cols &= cols - 1u;
                const uint right = tileAddress(world, pivot, col, count, words) + y * NewtonTile;
                for (uint wr = 0; wr <= col / 32u; ++wr)
                {
                    uint entries = hessianPattern[mask + wr].load();
                    if (wr == col / 32u)
                        entries &= inclusiveWordMask(col);
                    while (entries != 0)
                    {
                        const uint row = wr * 32u + uint(firstbitlow(entries));
                        entries &= entries - 1u;
                        if (row == col && x > y)
                            continue;
                        const uint left = tileAddress(world, pivot, row, count, words) + x * NewtonTile;
                        const uint target = tileAddress(world, col, row, count, words) + lane;
                        float value = loadHessian(target);
                        for (uint k = 0; k < NewtonTile; ++k)
                            value -= loadHessian(left + k) * loadHessian(right + k);
                        storeHessian(target, value);
                    }
                }
            }
        }
        syncNewtonStorage();
    }
}

float directionValue(uint base, uint dof, uint nv)
{
    return dof < nv ? directionHistory[base + dof].z : 0.0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void solve_newton_direction(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    group = solverDispatchId(group);
    const uint world = group.y;
    const CgWorld state = worlds[world];
    if (broadcastCg(state.progress.y, lane) != Running)
        return;
    const uint nv = settings[1].y;
    const uint count = broadcastCg(tileCount(), lane);
    const uint words = patternWords(count);
    if (broadcastCg(hessianPattern[factorErrorAddress(world, count, words)].load(), lane) != 0)
        return;
    const uint offsets = componentOffsetBase(world, count, words);
    const uint begin = broadcastCg(hessianPattern[offsets + group.x].load(), lane);
    const uint length = broadcastCg(hessianPattern[offsets + group.x + 1].load() - begin, lane);
    if (length == 0)
        return;
    const uint members = offsets + count + 1;
    const uint base = world * nv;
    for (uint i = lane; i < length * NewtonTile; i += CgWidth)
    {
        const uint tile = hessianPattern[members + begin + i / NewtonTile].load();
        const uint dof = tile * NewtonTile + i % NewtonTile;
        if (dof < nv)
            directionHistory[base + dof] = float4(0.0, 0.0, -gradient[base + dof], 0.0);
    }
    syncNewtonStorage();

    for (uint reverse = 0; reverse < length; ++reverse)
    {
        const uint earlier = length - 1 - reverse;
        const uint col = hessianPattern[members + begin + earlier].load();
        const uint diagonal = tileAddress(world, col, col, count, words);
        if (lane == 0)
        {
            for (uint k = NewtonTile; k > 0; --k)
            {
                const uint i = k - 1;
                const uint dof = col * NewtonTile + i;
                float value = directionValue(base, dof, nv);
                for (uint j = k; j < NewtonTile; ++j)
                    value -= loadHessian(diagonal + i * NewtonTile + j) * tileVector[j];
                tileVector[i] = value / loadHessian(diagonal + i * NewtonTile + i);
                if (dof < nv)
                    directionHistory[base + dof] = float4(0.0, 0.0, tileVector[i], 0.0);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint i = lane; i < earlier * NewtonTile; i += CgWidth)
        {
            const uint row = hessianPattern[members + begin + i / NewtonTile].load();
            if (!hasTile(world, col, row, count, words))
                continue;
            const uint address = tileAddress(world, col, row, count, words) + (i % NewtonTile) * NewtonTile;
            const uint dof = row * NewtonTile + i % NewtonTile;
            float value = directionHistory[base + dof].z;
            for (uint j = 0; j < NewtonTile; ++j)
                value -= loadHessian(address + j) * tileVector[j];
            directionHistory[base + dof] = float4(0.0, 0.0, value, 0.0);
        }
        syncNewtonStorage();
    }

    for (uint i = 0; i < length; ++i)
    {
        const uint col = hessianPattern[members + begin + i].load();
        const uint diagonal = tileAddress(world, col, col, count, words);
        if (lane < NewtonTile)
        {
            const uint dof = col * NewtonTile + lane;
            float value = directionValue(base, dof, nv);
            const uint mask = patternBase(world, col, count, words);
            for (uint word = 0; word <= col / 32u; ++word)
            {
                uint entries = hessianPattern[mask + word].load();
                if (word == col / 32u)
                    entries &= (1u << (col & 31u)) - 1u;
                while (entries != 0)
                {
                    const uint row = word * 32u + uint(firstbitlow(entries));
                    entries &= entries - 1u;
                    const uint tile = tileAddress(world, col, row, count, words);
                    for (uint i = 0; i < NewtonTile; ++i)
                        value -= loadHessian(tile + i * NewtonTile + lane) *
                                 directionHistory[base + row * NewtonTile + i].z;
                }
            }
            tileVector[lane] = value;
        }
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0)
        {
            for (uint i = 0; i < NewtonTile; ++i)
            {
                float value = tileVector[i];
                for (uint j = 0; j < i; ++j)
                    value -= loadHessian(diagonal + j * NewtonTile + i) * tileVector[j];
                const uint dof = col * NewtonTile + i;
                tileVector[i] = value / loadHessian(diagonal + i * NewtonTile + i);
                if (dof < nv)
                    directionHistory[base + dof] = float4(0.0, 0.0, tileVector[i], 0.0);
            }
        }
        syncNewtonStorage();
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void finalize_newton_direction(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    group = solverDispatchId(group);
    const uint world = group.y;
    CgWorld state = worlds[world];
    if (broadcastCg(state.progress.y, lane) != Running)
        return;
    const uint count = broadcastCg(tileCount(), lane);
    if (broadcastCg(hessianPattern[factorErrorAddress(world, count, patternWords(count))].load(), lane) != 0)
    {
        if (lane == 0)
        {
            state.progress.y = InvalidFactor;
            worlds[world] = state;
        }
        return;
    }
    const uint nv = settings[1].y;
    const uint base = world * nv;
    float4 partial = float4(0.0);
    for (uint i = lane; i < nv; i += CgWidth)
    {
        const float g = gradient[base + i];
        const float direction = directionHistory[base + i].z;
        partial.xyz += float3(direction * direction, g * g, -g * direction);
    }
    partial = reduceCg(partial, lane);
    if (lane == 0)
    {
        const float scale = state.search.w;
        const float tolerance = asfloat(settings[2 + world].z);
        const bool first = state.progress.x == 0;
        const bool done = scale * 0.5 * partial.z < tolerance || scale * sqrt(partial.y) < tolerance ||
                          (!first && (state.search.x == 0.0 ||
                                     (state.search.y > 0.0 && scale * state.search.y < tolerance)));
        if (done)
            state.progress.y = Converged;
        else if (state.progress.x >= cgOptions[0].x)
            state.progress.y = IterationLimit;
        state.search.z = partial.x;
        worlds[world] = state;
    }
}
