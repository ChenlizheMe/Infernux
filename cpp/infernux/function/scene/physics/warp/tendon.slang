// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/smooth.py and _src/passive.py tendon dynamics.

#include "model.slang"
#include "tendon_data.slang"

static const uint TendonWidth = 64;
static const float TendonMinLength = 1.0e-15;

struct TendonSegment
{
    uint4 bodies; // body A, body B, root A, root B
    float4 anchorA; // local anchor and inverse pulley divisor
    float4 anchorB;
};

struct TendonSegmentState
{
    float4 offsetA; // relative to tree COM, segment length after pulley scaling
    float4 offsetB; // relative to tree COM, inverse pulley divisor
    float4 direction;
    float4 directionDerivative;
    float4 velocityA;
    float4 velocityB;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint4> tendonDimensions;
[[vk::binding(1, 0)]] StructuredBuffer<TendonModel> tendons;
[[vk::binding(2, 0)]] StructuredBuffer<TendonSegment> segments;
[[vk::binding(3, 0)]] StructuredBuffer<uint4> tendonColumns; // DOF, segment offset/qpos, segment count/coefficient, quaternion owner
[[vk::binding(4, 0)]] StructuredBuffer<uint2> columnSegments; // segment index, endpoint mask
[[vk::binding(5, 0)]] StructuredBuffer<Pose> bodyPose;
[[vk::binding(6, 0)]] StructuredBuffer<float4> subtreeCom;
[[vk::binding(7, 0)]] StructuredBuffer<SpatialMotion> bodyVelocity;
[[vk::binding(8, 0)]] RWStructuredBuffer<TendonSegmentState> segmentState;
[[vk::binding(9, 0)]] StructuredBuffer<float> qpos;
[[vk::binding(10, 0)]] StructuredBuffer<float> qvel;
[[vk::binding(11, 0)]] StructuredBuffer<SpatialMotion> centeredDof;
[[vk::binding(12, 0)]] StructuredBuffer<SpatialMotion> centeredDofDot;
[[vk::binding(13, 0)]] RWStructuredBuffer<float> tendonJacobian;
[[vk::binding(14, 0)]] RWStructuredBuffer<TendonState> tendonState;

groupshared float3 tendonReduction[TendonWidth];

[shader("compute")]
[numthreads(64, 1, 1)]
void evaluate_tendon_segments(uint3 id : SV_DispatchThreadID)
{
    const uint segmentCount = tendonDimensions[0].y;
    if (id.x >= segmentCount)
        return;
    const TendonSegment segment = segments[id.x];
    const uint base = id.y * tendonDimensions[1].z;
    const uint bodyA = base + segment.bodies.x;
    const uint bodyB = base + segment.bodies.y;
    const Pose a = bodyPose[bodyA];
    const Pose b = bodyPose[bodyB];
    const float3 pointA = a.position.xyz + rotateVector(segment.anchorA.xyz, a.rotation);
    const float3 pointB = b.position.xyz + rotateVector(segment.anchorB.xyz, b.rotation);
    const float3 offsetA = pointA - subtreeCom[base + segment.bodies.z].xyz;
    const float3 offsetB = pointB - subtreeCom[base + segment.bodies.w].xyz;
    const SpatialMotion motionA = bodyVelocity[bodyA];
    const SpatialMotion motionB = bodyVelocity[bodyB];
    const float3 velocityA = motionA.linear.xyz + cross(motionA.angular.xyz, offsetA);
    const float3 velocityB = motionB.linear.xyz + cross(motionB.angular.xyz, offsetB);
    const float3 difference = pointB - pointA;
    const float distance = length(difference);
    const float scale = segment.anchorA.w;
    const float3 direction = distance < TendonMinLength ? float3(1.0, 0.0, 0.0) : difference / distance;
    const float3 relativeVelocity = velocityB - velocityA;
    const float3 derivative = distance > TendonMinLength ?
        (relativeVelocity - direction * dot(direction, relativeVelocity)) / distance : float3(0.0);
    const TendonSegmentState state = {float4(offsetA, scale * distance), float4(offsetB, scale),
        float4(direction, 0.0), float4(derivative, 0.0), float4(velocityA, 0.0), float4(velocityB, 0.0)};
    segmentState[id.y * segmentCount + id.x] = state;
}

float3 tendonPointMotion(SpatialMotion coordinate, float3 offset)
{
    return coordinate.linear.xyz + cross(coordinate.angular.xyz, offset);
}

float2 tendonColumn(uint world, uint4 entry, uint dof, bool useDerivative)
{
    const uint segmentBase = world * tendonDimensions[0].y;
    const SpatialMotion motion = centeredDof[dof];
    SpatialMotion derivative = spatial(float3(0.0), float3(0.0));
    if (useDerivative)
        derivative = entry.w == NoIndex ? centeredDofDot[dof] :
            motionCross(bodyVelocity[world * tendonDimensions[1].z + entry.w], motion);
    float2 result = float2(0.0);
    for (uint i = entry.y; i < entry.y + entry.z; ++i)
    {
        const uint2 influence = columnSegments[i];
        const TendonSegmentState segment = segmentState[segmentBase + influence.x];
        float3 jacobian = float3(0.0);
        float3 jacobianDot = float3(0.0);
        if ((influence.y & 1u) != 0)
        {
            jacobian -= tendonPointMotion(motion, segment.offsetA.xyz);
            if (useDerivative)
                jacobianDot -= tendonPointMotion(derivative, segment.offsetA.xyz) +
                               cross(motion.angular.xyz, segment.velocityA.xyz);
        }
        if ((influence.y & 2u) != 0)
        {
            jacobian += tendonPointMotion(motion, segment.offsetB.xyz);
            if (useDerivative)
                jacobianDot += tendonPointMotion(derivative, segment.offsetB.xyz) +
                               cross(motion.angular.xyz, segment.velocityB.xyz);
        }
        result.x += segment.offsetB.w * dot(jacobian, segment.direction.xyz);
        if (useDerivative)
            result.y += segment.offsetB.w * (dot(jacobianDot, segment.direction.xyz) +
                                            dot(jacobian, segment.directionDerivative.xyz));
    }
    return result;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void assemble_tendon_state(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    const uint4 dimensions = tendonDimensions[0]; // tendon count, segment count, Jacobian column count
    const uint4 stateDimensions = tendonDimensions[1]; // nq, nv, body count, packed mass length
    const TendonModel tendon = tendons[group.x];
    const uint4 span = tendon.layout;
    const bool spatial = span.y != 0;
    const uint qposBase = group.y * stateDimensions.x;
    const uint dofBase = group.y * stateDimensions.y;
    const uint segmentBase = group.y * dimensions.y + span.x;
    const uint columnBase = group.y * dimensions.z + span.z;
    float3 sum = float3(0.0); // length, J*qvel, Jdot*qvel
    for (uint i = lane; i < span.y; i += TendonWidth)
        sum.x += segmentState[segmentBase + i].offsetA.w;
    for (uint i = lane; i < span.w; i += TendonWidth)
    {
        const uint4 entry = tendonColumns[span.z + i];
        const uint dof = dofBase + entry.x;
        float2 j;
        if (spatial)
            j = tendonColumn(group.y, entry, dof, tendon.springArmature.z != 0.0);
        else
        {
            j = float2(asfloat(entry.z), 0.0);
            sum.x += j.x * qpos[qposBase + entry.y];
        }
        tendonJacobian[columnBase + i] = j.x;
        sum.yz += qvel[dof] * j;
    }
    tendonReduction[lane] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint width = TendonWidth / 2; width > 0; width >>= 1)
    {
        if (lane < width)
            tendonReduction[lane] += tendonReduction[lane + width];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0)
    {
        sum = tendonReduction[0];
        const float x = sum.x - clamp(sum.x, tendon.springArmature.x, tendon.springArmature.y);
        const float v = sum.y;
        const float spring = dot(tendon.stiffness.xyz, float3(1.0, x, x * x));
        const float damper = dot(tendon.damping.xyz, float3(1.0, abs(v), v * v));
        tendonState[group.y * dimensions.x + group.x] = {sum.x, v, -spring * x - damper * v,
                                                       tendon.springArmature.z * sum.z};
    }
}
