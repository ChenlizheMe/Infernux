// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/smooth.py and _src/passive.py tendon dynamics.

#include "model.slang"
#include "tendon_data.slang"

static const uint TendonWidth = 64;
static const float TendonMinLength = 1.0e-15;

#include "tendon_wrap.slang"

struct TendonSegment
{
    uint4 bodies; // body A, body B, root A, root B
    float4 anchorA; // local anchor and inverse pulley divisor
    float4 anchorB;
    uint4 outputSpan; // first resolved segment, wrap geometry or NoIndex
};

struct TendonWrapGeometry
{
    uint4 site; // body, root, cylinder flag, side site body or NoIndex
    float4 centerRadius;
    float4 rotation;
    float4 sideAnchor;
};

struct TendonWrapState
{
    float4 entry; // world point and unscaled arc length, negative when clear
    float4 exit;
};

struct TendonSegmentState
{
    float4 offsetA; // relative to tree COM, segment length after pulley scaling
    float4 offsetB; // relative to tree COM, inverse pulley divisor
    float4 direction; // unit direction and active wrap flag
    float4 directionDerivative;
    float4 velocityA;
    float4 velocityB;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint4> tendonDimensions;
[[vk::binding(1, 0)]] StructuredBuffer<TendonModel> tendons;
[[vk::binding(2, 0)]] StructuredBuffer<TendonSegment> segments;
[[vk::binding(3, 0)]] StructuredBuffer<uint4> tendonColumns; // DOF, segment offset/qpos, segment count/coefficient, quaternion owner
[[vk::binding(4, 0)]] StructuredBuffer<uint2> columnSegments; // segment index, clear mask, wrap mask, derivative flag
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
[[vk::binding(15, 0)]] StructuredBuffer<TendonWrapGeometry> wrapGeometry;
[[vk::binding(16, 0)]] RWStructuredBuffer<TendonWrapState> wrapState;

groupshared float3 tendonReduction[TendonWidth];

float3 tendonAttachment(uint body, float3 offset)
{
    const Pose pose = bodyPose[body];
    return pose.position.xyz + rotateVector(offset, pose.rotation);
}

TendonSegmentState tendonLine(uint4 bodies, float3 pointA, float3 pointB, float scale, float arc, bool wrapped)
{
    const float3 offsetA = pointA - subtreeCom[bodies.z].xyz;
    const float3 offsetB = pointB - subtreeCom[bodies.w].xyz;
    const SpatialMotion motionA = bodyVelocity[bodies.x];
    const SpatialMotion motionB = bodyVelocity[bodies.y];
    const float3 velocityA = motionA.linear.xyz + cross(motionA.angular.xyz, offsetA);
    const float3 velocityB = motionB.linear.xyz + cross(motionB.angular.xyz, offsetB);
    const float3 difference = pointB - pointA;
    const float distance = length(difference);
    const float3 direction = distance < TendonMinLength ? float3(1.0, 0.0, 0.0) : difference / distance;
    const float3 relativeVelocity = velocityB - velocityA;
    const float3 derivative = !wrapped && distance > TendonMinLength ?
        (relativeVelocity - direction * dot(direction, relativeVelocity)) / distance : float3(0.0);
    return {float4(offsetA, scale * (distance + arc)), float4(offsetB, scale), float4(direction, float(wrapped)),
            float4(derivative, 0.0), float4(velocityA, 0.0), float4(velocityB, 0.0)};
}

[shader("compute")]
[numthreads(64, 1, 1)]
void evaluate_tendon_segments(uint3 id : SV_DispatchThreadID)
{
    const uint4 dims = tendonDimensions[0];
    if (id.x >= dims.y)
        return;
    const TendonSegment segment = segments[id.x];
    const uint base = id.y * tendonDimensions[1].z;
    const uint4 bodies = base + segment.bodies;
    const float3 pointA = tendonAttachment(bodies.x, segment.anchorA.xyz);
    const float3 pointB = tendonAttachment(bodies.y, segment.anchorB.xyz);
    const float scale = segment.anchorA.w;
    const uint slot = id.y * dims.w + segment.outputSpan.x;
    const uint geometryIndex = segment.outputSpan.y;
    if (geometryIndex != NoIndex)
    {
        const TendonWrapGeometry geometry = wrapGeometry[geometryIndex];
        const uint body = base + geometry.site.x;
        const uint root = base + geometry.site.y;
        const Pose geomPose = bodyPose[body];
        const float3 center = geomPose.position.xyz + rotateVector(geometry.centerRadius.xyz, geomPose.rotation);
        const float4 rotation = mulQuat(geomPose.rotation, geometry.rotation);
        const bool guided = geometry.site.w != NoIndex;
        float3 guide = center;
        if (guided)
            guide = tendonAttachment(base + geometry.site.w, geometry.sideAnchor.xyz);
        const TendonWrapSolution solution = solveTendonWrap(pointA, pointB, center, rotation,
            geometry.centerRadius.w, geometry.site.z != 0, guided, guide);
        const TendonWrapState wrap = {float4(solution.pointA, solution.arcLength), float4(solution.pointB, 0.0)};
        wrapState[id.y * tendonDimensions[2].x + geometryIndex] = wrap;
        if (solution.arcLength >= 0.0)
        {
            segmentState[slot] = tendonLine(uint4(bodies.x, body, bodies.z, root),
                pointA, solution.pointA, scale, solution.arcLength, true);
            segmentState[slot + 1] = tendonLine(uint4(body, bodies.y, root, bodies.w),
                solution.pointB, pointB, scale, 0.0, true);
            return;
        }
        const TendonSegmentState empty = {};
        segmentState[slot + 1] = empty;
    }
    segmentState[slot] = tendonLine(bodies, pointA, pointB, scale, 0.0, false);
}

float3 tendonPointMotion(SpatialMotion coordinate, float3 offset)
{
    return coordinate.linear.xyz + cross(coordinate.angular.xyz, offset);
}

float2 tendonColumn(uint world, uint4 entry, uint dof, bool useDerivative)
{
    const uint segmentBase = world * tendonDimensions[0].w;
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
        const uint mask = (segment.direction.w != 0.0 ? influence.y >> 2u : influence.y) & 3u;
        const bool differentiate = useDerivative && (influence.y & 16u) != 0;
        float3 jacobian = float3(0.0);
        float3 jacobianDot = float3(0.0);
        if ((mask & 1u) != 0)
        {
            jacobian -= tendonPointMotion(motion, segment.offsetA.xyz);
            if (differentiate)
                jacobianDot -= tendonPointMotion(derivative, segment.offsetA.xyz) +
                               cross(motion.angular.xyz, segment.velocityA.xyz);
        }
        if ((mask & 2u) != 0)
        {
            jacobian += tendonPointMotion(motion, segment.offsetB.xyz);
            if (differentiate)
                jacobianDot += tendonPointMotion(derivative, segment.offsetB.xyz) +
                               cross(motion.angular.xyz, segment.velocityB.xyz);
        }
        result.x += segment.offsetB.w * dot(jacobian, segment.direction.xyz);
        if (differentiate)
            result.y += segment.offsetB.w * (dot(jacobianDot, segment.direction.xyz) +
                                            dot(jacobian, segment.directionDerivative.xyz));
    }
    return result;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void assemble_tendon_state(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    const uint4 dimensions = tendonDimensions[0]; // tendon count, input segments, Jacobian columns, resolved segments
    const uint4 stateDimensions = tendonDimensions[1]; // nq, nv, body count, packed mass length
    const TendonModel tendon = tendons[group.x];
    const uint4 span = tendon.layout;
    const bool spatial = span.y != 0;
    const uint qposBase = group.y * stateDimensions.x;
    const uint dofBase = group.y * stateDimensions.y;
    const uint segmentBase = group.y * dimensions.w + span.x;
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
