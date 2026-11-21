// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// Slang adaptation of MuJoCo Warp _src/constraint.py connect and weld rows.

#include "model.slang"
#include "constraint_data.slang"
#include "constraint_parameters.slang"
#include "constraint_stream.slang"

static const uint SpatialEqualityWords = 9;

[[vk::binding(1, 0)]] StructuredBuffer<uint4> spatialModel;
[[vk::binding(2, 0)]] StructuredBuffer<uint> equalityActive;
[[vk::binding(3, 0)]] StructuredBuffer<Pose> bodyPose;
[[vk::binding(4, 0)]] StructuredBuffer<float4> subtreeCom;
[[vk::binding(6, 0)]] StructuredBuffer<SpatialMotion> centeredDof;
[[vk::binding(9, 0)]] StructuredBuffer<SpatialMotion> centeredDofDerivative;
[[vk::binding(10, 0)]] StructuredBuffer<SpatialMotion> bodyVelocity;
[[vk::binding(11, 0)]] StructuredBuffer<float> qvel;

struct SpatialEndpoint
{
    float3 point;
    float3 offset;
    float3 pointVelocity;
    float3 omega;
    float4 orientation;
};

float4 conjugateQuaternion(float4 q)
{
    return float4(q.x, -q.yzw);
}

SpatialEndpoint equalityEndpoint(uint base, uint body, uint welded, uint root, float3 localPoint, float4 localFrame)
{
    const Pose pose = bodyPose[base + body];
    const SpatialMotion weldedMotion = bodyVelocity[base + welded];
    const float3 point = pose.position.xyz + rotateVector(localPoint, pose.rotation);
    const float3 offset = point - subtreeCom[base + root].xyz;
    return {point, offset, weldedMotion.linear.xyz + cross(weldedMotion.angular.xyz, offset),
            bodyVelocity[base + body].angular.xyz, mulQuat(pose.rotation, localFrame)};
}

float3 pointJacobian(SpatialMotion motion, SpatialEndpoint endpoint)
{
    return motion.linear.xyz + cross(motion.angular.xyz, endpoint.offset);
}

float3 pointJacobianDerivative(SpatialMotion motion, SpatialMotion derivative, SpatialEndpoint endpoint)
{
    return pointJacobian(derivative, endpoint) + cross(motion.angular.xyz, endpoint.pointVelocity);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void emit_spatial_equality_rows(uint3 id : SV_DispatchThreadID)
{
    const uint4 model = spatialModel[0]; // descriptor count, equality count
    if (id.x >= model.x)
        return;
    const uint record = 1 + SpatialEqualityWords * id.x;
    const uint4 source = spatialModel[record]; // equality id, first body, second body, Jacobian width
    if (equalityActive[id.y * model.y + source.x] == 0)
        return;
    const uint4 span = spatialModel[record + 1]; // first welded body, second welded body, support offset, height
    const uint2 block = reserveConstraintBlock(id.y, span.w, source.w);
    if (block.x == NoIndex)
        return;

    const uint4 limits = settings[0];
    const uint4 config = settings[1];
    const uint bodyBase = id.y * config.x;
    const uint dofBase = id.y * config.y;
    const uint columnBase = id.y * limits.w + block.y;
    const bool weld = span.w == 6;
    const uint4 reference = spatialModel[record + 8]; // roots, inverse weights as float bits
    const float4 anchorA = asfloat(spatialModel[record + 4]);
    const float3 anchorB = asfloat(spatialModel[record + 5].xyz);
    const SpatialEndpoint a = equalityEndpoint(bodyBase, source.y, span.x, reference.x, anchorA.xyz,
                                               asfloat(spatialModel[record + 6]));
    const SpatialEndpoint b = equalityEndpoint(bodyBase, source.z, span.y, reference.y, anchorB,
                                               asfloat(spatialModel[record + 7]));
    const float4 orientationBInverse = conjugateQuaternion(b.orientation);
    const float torqueScale = anchorA.w;
    float3 linearVelocity = float3(0.0);
    float3 angularVelocity = float3(0.0);
    float3 linearShift = float3(0.0);
    float3 angularShift = float3(0.0);
    for (uint j = 0; j < source.w; ++j)
    {
        // A DOF shared by the endpoints contributes to both point Jacobians.
        const uint4 support = spatialModel[span.z + j]; // DOF, endpoint mask, quaternion owner or NoIndex
        const bool inA = (support.y & 1u) != 0;
        const bool inB = (support.y & 2u) != 0;
        const uint index = dofBase + support.x;
        const SpatialMotion motion = centeredDof[index];
        const SpatialMotion derivative = support.z == NoIndex ? centeredDofDerivative[index] :
            motionCross(bodyVelocity[bodyBase + support.z], motion);
        const float speed = qvel[index];
        const float3 linear = (inA ? pointJacobian(motion, a) : float3(0.0)) -
                              (inB ? pointJacobian(motion, b) : float3(0.0));
        const float3 linearDot = (inA ? pointJacobianDerivative(motion, derivative, a) : float3(0.0)) -
                                 (inB ? pointJacobianDerivative(motion, derivative, b) : float3(0.0));
        for (uint axis = 0; axis < 3; ++axis)
            columns[columnBase + axis * source.w + j] = {support.x, linear[axis]};
        linearVelocity += linear * speed;
        linearShift += linearDot * speed;

        if (weld)
        {
            const float direction = float(inA) - float(inB);
            const float4 relativeAxis = mulQuat(mulQuat(orientationBInverse,
                float4(0.0, direction * motion.angular.xyz)), a.orientation);
            const float3 angular = (0.5 * torqueScale) * relativeAxis.yzw;
            for (uint axis = 0; axis < 3; ++axis)
                columns[columnBase + (axis + 3) * source.w + j] = {support.x, angular[axis]};
            angularVelocity += angular * speed;
            angularShift += direction * derivative.angular.xyz * speed;
        }
    }

    const float3 linearError = a.point - b.point;
    float3 angularError = float3(0.0);
    if (weld)
    {
        angularError = torqueScale * mulQuat(orientationBInverse, a.orientation).yzw;
        const float4 delta = float4(0.0, a.omega - b.omega);
        const float4 aDot = 0.5 * mulQuat(float4(0.0, a.omega), a.orientation);
        const float4 bDotInv = conjugateQuaternion(0.5 * mulQuat(float4(0.0, b.omega), b.orientation));
        const float4 shift = mulQuat(mulQuat(bDotInv, delta), a.orientation) +
            mulQuat(mulQuat(orientationBInverse, float4(0.0, angularShift)), a.orientation) +
            mulQuat(mulQuat(orientationBInverse, delta), aDot);
        angularShift = (0.5 * torqueScale) * shift.yzw;
    }

    const ConstraintParameters parameters = {asfloat(spatialModel[record + 2]), asfloat(spatialModel[record + 3])};
    const float impedancePosition = sqrt(dot(linearError, linearError) + dot(angularError, angularError));
    const float timestep = asfloat(settings[2 + id.y].x);
    const bool unclamped = (config.w & UnclampedReference) != 0;
    const float3 kbi = impedanceParameters(parameters, impedancePosition, timestep, unclamped);
    const uint rowBase = id.y * limits.z + block.x;
    for (uint axis = 0; axis < 3; ++axis)
    {
        rows[rowBase + axis] = makeConstraintRow(uint4(block.y + axis * source.w, source.w, source.x, axis),
            EqualityRow, kbi, linearError[axis], 0.0, linearVelocity[axis], asfloat(reference.z), 0.0, linearShift[axis]);
        if (weld)
        {
            const uint ordinal = axis + 3;
            rows[rowBase + ordinal] = makeConstraintRow(uint4(block.y + ordinal * source.w, source.w, source.x, ordinal),
                EqualityRow, kbi, angularError[axis], 0.0, angularVelocity[axis], asfloat(reference.w), 0.0,
                angularShift[axis]);
        }
    }
}
