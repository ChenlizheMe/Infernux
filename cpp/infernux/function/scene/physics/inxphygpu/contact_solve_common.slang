// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/solver.py rigid contact solver data and workgroup reductions.

#include "contact.slang"
#include "constraint_data.slang"
#include "solver_worlds.slang"

static const uint CgWidth = 64;
static const float CgMinValue = 1.0e-15;
static const float AlphaNoise = 9.5367431640625e-7;
static const uint Running = 0;
static const uint Converged = 1;
static const uint IterationLimit = 2;
static const uint CapacityExceeded = 3;
static const uint InvalidFactor = 4;
static const uint LineSearchLimit = 1;

struct CgWorld
{
    uint4 progress; // accepted/attempted iterations, status, active rows, warning bits
    float4 search; // alpha, improvement, squared direction norm, inverse inertia trace
    float2 gauss; // smooth cost slope, curvature
};

[[vk::binding(0, 0)]] StructuredBuffer<uint4> settings;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> cgOptions; // iteration limit, line limit, packed mass length, Hessian tile budget
[[vk::binding(2, 0)]] RWStructuredBuffer<Atomic<uint>> rowCounts;
[[vk::binding(3, 0)]] StructuredBuffer<ConstraintRow> rows;
[[vk::binding(4, 0)]] StructuredBuffer<ConstraintColumn> columns;
[[vk::binding(5, 0)]] StructuredBuffer<Contact> contacts;
[[vk::binding(6, 0)]] RWStructuredBuffer<ConstraintSolveState> constraintState;
[[vk::binding(8, 0)]] RWStructuredBuffer<float> acceleration;
[[vk::binding(9, 0)]] StructuredBuffer<float> smoothForce;
[[vk::binding(10, 0)]] StructuredBuffer<float> massMatrix;
// DOF row offsets followed by pairs of (DOF, packed matrix offset), diagonal first.
[[vk::binding(11, 0)]] StructuredBuffer<uint> massTopology;
[[vk::binding(12, 0)]] RWStructuredBuffer<float> gradient;
[[vk::binding(13, 0)]] StructuredBuffer<float> preconditionedGradient;
[[vk::binding(15, 0)]] RWStructuredBuffer<float4> directionHistory; // previous gradient, preconditioned, direction
[[vk::binding(16, 0)]] RWStructuredBuffer<CgWorld> worlds;
[[vk::binding(17, 0)]] RWStructuredBuffer<float2> massProducts; // M * acceleration, M * direction

groupshared float4 cgReduction[CgWidth];
groupshared uint uniformStatus;

float4 reduceCg(float4 partial, uint lane)
{
    cgReduction[lane] = partial;
    GroupMemoryBarrierWithGroupSync();
    for (uint width = CgWidth / 2; width > 0; width >>= 1)
    {
        if (lane < width)
            cgReduction[lane] += cgReduction[lane + width];
        GroupMemoryBarrierWithGroupSync();
    }
    const float4 sum = cgReduction[0];
    GroupMemoryBarrierWithGroupSync();
    return sum;
}

uint broadcastCg(uint value, uint lane)
{
    if (lane == 0)
        uniformStatus = value;
    GroupMemoryBarrierWithGroupSync();
    const uint result = workgroupUniformLoad(uniformStatus);
    GroupMemoryBarrierWithGroupSync();
    return result;
}
