// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from MuJoCo Warp _src/derivative.py, _src/forward.py, and _src/smooth.py.

#include "model.slang"

struct RneDerivative
{
    SpatialMotion acceleration;
    SpatialMotion force;
};

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] StructuredBuffer<BodyModel> bodies;
[[vk::binding(2, 0)]] StructuredBuffer<JointModel> joints;
[[vk::binding(3, 0)]] StructuredBuffer<DofModel> dofs;
[[vk::binding(4, 0)]] StructuredBuffer<float> qvel;
[[vk::binding(5, 0)]] StructuredBuffer<SpatialMotion> centeredDof;
[[vk::binding(6, 0)]] StructuredBuffer<SpatialMotion> centeredDofDot;
[[vk::binding(7, 0)]] StructuredBuffer<SpatialMotion> bodyVelocity;
[[vk::binding(8, 0)]] StructuredBuffer<SpatialInertia> centeredInertia;
// Header: full matrix size, derivative count, row table start, entry table start.
// Row: first matrix entry, diagonal entry, end entry, derivative range for this column.
// Entry: column, transpose entry, packed mass index, derivative index.
[[vk::binding(9, 0)]] StructuredBuffer<uint4> implicitTopology;
[[vk::binding(10, 0)]] RWStructuredBuffer<RneDerivative> rneDerivatives;
[[vk::binding(11, 0)]] StructuredBuffer<float> timestep;
[[vk::binding(12, 0)]] StructuredBuffer<float> packedEffectiveMass;
[[vk::binding(13, 0)]] RWStructuredBuffer<float> effectiveMatrix;
[[vk::binding(14, 0)]] StructuredBuffer<float> rhs;
[[vk::binding(15, 0)]] RWStructuredBuffer<float> acceleration;

[shader("compute")]
[numthreads(64, 1, 1)]
void differentiate_rne(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions dimensions = model[0];
    if (id.x >= dimensions.state.y)
        return;

    const uint4 layout = implicitTopology[0];
    const uint derivativeBase = id.y * layout.y + implicitTopology[layout.z + id.x].w;
    const uint dofBase = id.y * dimensions.state.y;
    const uint bodyBase = id.y * dimensions.state.z;
    const uint owner = dofs[id.x].indices.x;
    const uint end = owner + bodies[owner].subtree.y;
    const SpatialMotion velocityDerivative = centeredDof[dofBase + id.x];

    // A generalized velocity changes only its owner's subtree. The parent part of
    // the acceleration derivative is ready when visiting each descendant.
    for (uint index = owner; index < end; ++index)
    {
        const BodyModel body = bodies[index];
        const uint slot = derivativeBase + index - owner;
        SpatialMotion derivative = spatial(float3(0.0), float3(0.0));
        if (index != owner)
            derivative = rneDerivatives[derivativeBase + body.hierarchy.x - owner].acceleration;

        bool preceding = index != owner;
        for (uint local = 0; local < body.hierarchy.z; ++local)
        {
            const JointModel joint = joints[body.hierarchy.y + local];
            uint first = joint.indices.z;
            const uint type = joint.indices.x;
            if (type == JointFree)
            {
                // Free translation has constant motion axes.
                preceding = preceding || (id.x >= first && id.x < first + 3);
                first += 3;
            }
            const uint count = type == JointFree || type == JointBall ? 3 : 1;
            for (uint axis = 0; axis < count; ++axis)
            {
                const uint coordinate = first + axis;
                const uint dof = dofBase + coordinate;
                if (coordinate == id.x)
                    addMotion(derivative, centeredDofDot[dof], 1.0);
                if (preceding)
                    addMotion(derivative, motionCross(velocityDerivative, centeredDof[dof]), qvel[dof]);
            }
            preceding = preceding || (id.x >= first && id.x < first + count);
        }

        const SpatialMotion velocity = bodyVelocity[bodyBase + index];
        const SpatialInertia inertia = centeredInertia[bodyBase + index];
        SpatialMotion force = inertiaTimesMotion(inertia, derivative);
        addMotion(force, motionCrossForce(velocityDerivative, inertiaTimesMotion(inertia, velocity)), 1.0);
        addMotion(force, motionCrossForce(velocity, inertiaTimesMotion(inertia, velocityDerivative)), 1.0);
        rneDerivatives[slot] = {derivative, force};
    }

    for (uint child = end - 1; child > owner; --child)
    {
        const uint parent = derivativeBase + bodies[child].hierarchy.x - owner;
        SpatialMotion force = rneDerivatives[parent].force;
        addMotion(force, rneDerivatives[derivativeBase + child - owner].force, 1.0);
        rneDerivatives[parent].force = force;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void assemble_implicit_matrix(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions dimensions = model[0];
    if (id.x >= dimensions.state.y)
        return;
    const uint4 layout = implicitTopology[0];
    const uint4 row = implicitTopology[layout.z + id.x];
    const uint matrixBase = id.y * layout.x;
    const uint massBase = id.y * dimensions.layout.w;
    const uint derivativeBase = id.y * layout.y;
    const SpatialMotion axis = centeredDof[id.y * dimensions.state.y + id.x];
    const float dt = timestep[id.y];

    for (uint i = row.x; i < row.z; ++i)
    {
        const uint4 entry = implicitTopology[layout.w + i];
        float value = 0.0;
        if (entry.z != NoIndex)
            value = packedEffectiveMass[massBase + entry.z];
        if (entry.w != NoIndex)
            value += dt * spatialDot(axis, rneDerivatives[derivativeBase + entry.w].force);
        effectiveMatrix[matrixBase + i] = value;
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void factor_solve_implicit(uint3 id : SV_DispatchThreadID)
{
    const uint dofCount = model[0].state.y;
    uint rhsCount, rhsStride;
    rhs.GetDimensions(rhsCount, rhsStride);
    if (id.x * dofCount >= rhsCount)
        return;

    const uint4 layout = implicitTopology[0];
    const uint matrixBase = id.x * layout.x;
    const uint vectorBase = id.x * dofCount;

    // Reverse sparse LU stores a unit upper triangle. Its affected upper entries
    // are the transposes of the pivot row's lower entries.
    for (uint i = dofCount; i > 0; --i)
    {
        const uint4 pivot = implicitTopology[layout.z + i - 1];
        const float diagonal = effectiveMatrix[matrixBase + pivot.y];
        for (uint k = pivot.y; k > pivot.x; --k)
        {
            const uint2 neighbor = implicitTopology[layout.w + k - 1].xy;
            const uint upper = neighbor.y;
            const float scale = effectiveMatrix[matrixBase + upper] / diagonal;
            effectiveMatrix[matrixBase + upper] = scale;

            uint source = pivot.x;
            uint target = implicitTopology[layout.z + neighbor.x].x;
            while (source < pivot.y && target < upper)
            {
                const uint sourceColumn = implicitTopology[layout.w + source].x;
                const uint targetColumn = implicitTopology[layout.w + target].x;
                if (sourceColumn == targetColumn)
                    effectiveMatrix[matrixBase + target] =
                        fma(-scale, effectiveMatrix[matrixBase + source], effectiveMatrix[matrixBase + target]);
                if (sourceColumn <= targetColumn)
                    ++source;
                if (targetColumn <= sourceColumn)
                    ++target;
            }
        }
    }

    for (uint i = dofCount; i > 0; --i)
    {
        const uint dof = i - 1;
        const uint4 row = implicitTopology[layout.z + dof];
        float value = rhs[vectorBase + dof];
        for (uint j = row.y + 1; j < row.z; ++j)
            value = fma(-effectiveMatrix[matrixBase + j],
                        acceleration[vectorBase + implicitTopology[layout.w + j].x], value);
        acceleration[vectorBase + dof] = value;
    }

    for (uint dof = 0; dof < dofCount; ++dof)
    {
        const uint4 row = implicitTopology[layout.z + dof];
        float value = acceleration[vectorBase + dof];
        for (uint j = row.x; j < row.y; ++j)
            value = fma(-effectiveMatrix[matrixBase + j],
                        acceleration[vectorBase + implicitTopology[layout.w + j].x], value);
        acceleration[vectorBase + dof] = value / effectiveMatrix[matrixBase + row.y];
    }
}
