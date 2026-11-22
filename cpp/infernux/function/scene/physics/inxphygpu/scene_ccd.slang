// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "convex_sweep.slang"
#include "box_contacts.slang"
#include "collision_pair_rules.slang"
#include "contact_publish.slang"
#include "prescribed_motion.slang"
#include "scene_motion.slang"

static const uint CcdLanes = 8;
static const uint MaxImpacts = 32;

struct SweepMotion
{
    float4 center; // World COM. w records whether an impulse changed the body.
    float4 rotation;
    float4 linear;
    float4 angular;
};

[[vk::binding(8, 0)]] StructuredBuffer<ModelDimensions> ccdModel;
[[vk::binding(9, 0)]] StructuredBuffer<BodyModel> ccdBodies;
[[vk::binding(10, 0)]] StructuredBuffer<uint4> ccdRows; // qpos, qvel, mode, inner radius.
[[vk::binding(11, 0)]] RWStructuredBuffer<float> ccdPositions;
[[vk::binding(12, 0)]] RWStructuredBuffer<float> ccdVelocities;
[[vk::binding(13, 0)]] StructuredBuffer<Pose> ccdStart;
[[vk::binding(14, 0)]] RWStructuredBuffer<SweepMotion> ccdMotion;
[[vk::binding(15, 0)]] StructuredBuffer<uint4> ccdSettings; // dt, threshold, minimum rebound speed, filter parents.
[[vk::binding(16, 0)]] StructuredBuffer<uint> ccdContactStatus;
[[vk::binding(17, 0)]] StructuredBuffer<GeomModel> ccdGeoms;
[[vk::binding(18, 0)]] StructuredBuffer<PrescribedMotion> ccdPrescribed;
[[vk::binding(19, 0)]] StructuredBuffer<uint4> ccdRules;
[[vk::binding(20, 0)]] StructuredBuffer<uint4> ccdGeometry;
[[vk::binding(21, 0)]] RWStructuredBuffer<uint4> ccdScratch;
[[vk::binding(22, 0)]] RWStructuredBuffer<uint> ccdHorizon;
[[vk::binding(23, 0)]] RWStructuredBuffer<SceneBodyActivity> ccdActivity;
[[vk::binding(24, 0)]] StructuredBuffer<uint> ccdFixedSteps;

groupshared ShapeCastHit ccdHits[CcdLanes];
groupshared uint ccdSelection;

void syncSweepMotion()
{
    __target_switch
    {
    case wgsl: __intrinsic_asm "storageBarrier()";
    default: DeviceMemoryBarrierWithGroupSync();
    }
}

float4 inverseRotation(float4 rotation)
{
    return float4(rotation.x, -rotation.yzw);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void prepare_scene_ccd(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions size = ccdModel[0];
    if (id.x >= size.state.z)
        return;
    const uint index = id.y * size.state.z + id.x;
    const BodyModel body = ccdBodies[id.x];
    const Pose old = ccdStart[index];
    const float dt = asfloat(ccdSettings[0].x);
    SweepMotion motion = {float4(old.position.xyz + rotateVector(body.inertialPosition.xyz, old.rotation), 0.0),
                          old.rotation, float4(0.0), float4(0.0)};
    const uint4 row = ccdRows[id.x];
    if (row.x != NoIndex)
    {
        const uint p = id.y * size.state.x + row.x;
        const uint v = id.y * size.state.y + row.y;
        motion.rotation = float4(ccdPositions[p + 3], ccdPositions[p + 4], ccdPositions[p + 5], ccdPositions[p + 6]);
        motion.angular.xyz =
            rotateVector(float3(ccdVelocities[v + 3], ccdVelocities[v + 4], ccdVelocities[v + 5]), motion.rotation);
        const float3 end = float3(ccdPositions[p], ccdPositions[p + 1], ccdPositions[p + 2]) +
                           rotateVector(body.inertialPosition.xyz, motion.rotation);
        motion.linear.xyz = (end - motion.center.xyz) / dt;
    }
    else
    {
        const uint support = prescribedRow(body, size.layout.y);
        if (support != NoIndex)
        {
            const PrescribedMotion drive = ccdPrescribed[id.y * prescribedStride(size, ccdBodies[0]) + support];
            if (drive.step == ccdFixedSteps[id.y] + 1u)
            {
                // The command phase already placed this body at the substep end.
                // Sweep over its commanded translation with the new orientation.
                motion.center.xyz = drive.center - dt * drive.linear.xyz;
                motion.linear = drive.linear;
                motion.angular = drive.angular;
            }
        }
    }
    if (id.x == 0)
        motion.center.w = ccdContactStatus[id.y] != 0u ? 2.0 : 0.0;
    ccdMotion[index] = motion;
}

CartesianFrame sweepFrame(uint geometry, SweepMotion motion)
{
    const GeomModel shape = ccdGeoms[geometry];
    const float3 offset = shape.position.xyz - ccdBodies[shape.indices.x].inertialPosition.xyz;
    return cartesianFrame(motion.center.xyz + rotateVector(offset, motion.rotation),
                          mulQuat(motion.rotation, shape.rotation));
}

ShapeCastHit earliestLaneImpact(uint world, uint lane, float duration, bool threshold)
{
    const ModelDimensions size = ccdModel[0];
    const uint base = world * size.state.z;
    const uint4 settings = ccdSettings[0];
    ShapeCastHit result = {float4(0, 0, 0, duration), uint4(NoIndex), float4(0.0)};
    const ConvexPolytope workspace = castWorkspace(world * CcdLanes + lane);
    for (uint first = lane; first < size.layout.x; first += CcdLanes)
    {
        const uint4 firstGeometry = ccdGeoms[first].indices;
        const uint a = firstGeometry.x;
        const uint4 source = ccdRows[a];
        if (source.z == 0 || (ccdRules[first].z & 2u) != 0)
            continue;
        const SweepMotion motion = ccdMotion[base + a];
        const CartesianFrame movingFrame = sweepFrame(first, motion);
        ConvexGeometry moving = queryConvexGeometry(movingFrame, shapes[first]);
        moving.frame.position.xyz = float3(0.0);
        float3 minimum, maximum;
        castBounds(moving, ccdGeometry, minimum, maximum);
        const float3 extent = max(abs(minimum), abs(maximum));
        for (uint second = 0; second < size.layout.x; ++second)
        {
            const uint4 secondGeometry = ccdGeoms[second].indices;
            const uint b = secondGeometry.x;
            if (a == b || (ccdRules[second].z & 2u) != 0)
                continue;
            const uint4 target = ccdRows[b];
            if (target.x != NoIndex && max(source.z, target.z) < 2u)
                continue;
            const CollisionPairRule rule = collisionRule(
                first, second, firstGeometry, secondGeometry, settings.w != 0, ccdRules);
            if (rule.pair < -1 || rule.sensor >= 0)
                continue;
            const SweepMotion other = ccdMotion[base + b];
            const float3 relative = motion.linear.xyz - other.linear.xyz;
            const float speed = length(relative);
            if (speed <= 1.0e-5 || (threshold && speed * duration < asfloat(settings.y) * asfloat(source.w)))
                continue;
            const float3 direction = relative / speed;
            const CollisionShape shape = shapes[second];
            CartesianFrame frame = sweepFrame(second, other);
            frame.position.xyz -= movingFrame.position.xyz;
            const uint4 identity = uint4(first, second, 0u, NoIndex);
            ShapeCastHit hit = {float4(0, 0, 0, result.normalDistance.w * speed), uint4(NoIndex), float4(0.0)};
            if (shape.type.x == GeomTriangleMesh)
                castMesh(moving, frame, shape, extent, direction, 1.0e-5, identity, workspace,
                         ccdGeometry, ccdScratch, ccdHorizon, hit);
            else
            {
                ConvexGeometry stationary = queryConvexGeometry(frame, shape);
                float3 low, high;
                castBounds(stationary, ccdGeometry, low, high);
                const RaySlabs ray = raySlabs(float3(0.0), direction);
                if (rayBoundsOverlap(ray, low - extent, high + extent, hit.normalDistance.w))
                    castConvexPair(moving, stationary, direction, 1.0e-5, float3(0.0), 0u, identity, workspace,
                                   ccdGeometry, ccdScratch, ccdHorizon, hit);
            }
            if (hit.identity.x == NoIndex)
                continue;
            hit.identity.w = asuint(rule.pair);
            hit.normalDistance.w /= speed;
            hit.point.xyz += movingFrame.position.xyz + other.linear.xyz * hit.normalDistance.w;
            if (castPrecedes(hit, result))
                result = hit;
        }
    }
    return result;
}

float3 inverseSpin(BodyModel body, float4 rotation, float3 torque)
{
    const uint axes = body.subtree.w >> 3u;
    const float4 inertia = mulQuat(rotation, body.inertialRotation);
    const float3 local = rotateVector(projectSceneAxes(torque, axes), inverseRotation(inertia));
    return projectSceneAxes(rotateVector(local / body.inertiaMass.xyz, inertia), axes);
}

SpatialMotion impulseResponse(uint row, SweepMotion motion, float3 arm, float3 axis)
{
    if (ccdRows[row].x == NoIndex)
        return spatial(float3(0.0), float3(0.0));
    const BodyModel body = ccdBodies[row];
    return spatial(inverseSpin(body, motion.rotation, cross(arm, axis)),
                   projectSceneAxes(axis, body.subtree.w) / body.inertiaMass.w);
}

float3 pointVelocity(SweepMotion motion, float3 arm)
{
    return motion.linear.xyz + cross(motion.angular.xyz, arm);
}

float axisImpulse(uint2 bodies, float3 r1, float3 r2, float3 axis, float targetSpeed,
                  float limit, inout SweepMotion first, inout SweepMotion second)
{
    const SpatialMotion response1 = impulseResponse(bodies.x, first, r1, axis);
    const SpatialMotion response2 = impulseResponse(bodies.y, second, r2, axis);
    const float effective = dot(response1.linear.xyz + response2.linear.xyz +
                                cross(response1.angular.xyz, r1) + cross(response2.angular.xyz, r2), axis);
    const float speed = dot(pointVelocity(first, r1) - pointVelocity(second, r2), axis);
    const float impulse = effective > 0.0 ? clamp((targetSpeed - speed) / effective, 0.0, limit) : 0.0;
    first.linear.xyz += response1.linear.xyz * impulse;
    first.angular.xyz += response1.angular.xyz * impulse;
    second.linear.xyz -= response2.linear.xyz * impulse;
    second.angular.xyz -= response2.angular.xyz * impulse;
    return impulse;
}

float3 impactPoint(ShapeCastHit hit, SweepMotion first, SweepMotion second)
{
    const CollisionShape a = shapes[hit.identity.x], b = shapes[hit.identity.y];
    if (a.type.x != GeomBox || b.type.x != GeomBox)
        return hit.point.xyz;
    const BoxManifold manifold = boxBoxContact(sweepFrame(hit.identity.x, first), a.size.xyz,
                                               sweepFrame(hit.identity.y, second), b.size.xyz, 2.0e-5);
    if (manifold.count == 0)
        return hit.point.xyz;
    float3 point = float3(0.0);
    for (uint i = 0; i < manifold.count; ++i)
        point += manifold.points[i].xyz;
    return point / float(manifold.count);
}

void solveImpact(uint world, ShapeCastHit hit)
{
    const ModelDimensions size = ccdModel[0];
    const uint base = world * size.state.z;
    const uint a = ccdGeoms[hit.identity.x].indices.x, b = ccdGeoms[hit.identity.y].indices.x;
    SweepMotion first = ccdMotion[base + a], second = ccdMotion[base + b];
    const float3 point = impactPoint(hit, first, second);
    const float3 r1 = point - first.center.xyz, r2 = point - second.center.xyz;
    const float3 normal = hit.normalDistance.xyz;
    const float speed = dot(pointVelocity(first, r1) - pointVelocity(second, r2), normal);
    const ContactMaterial material = resolveMaterial(uint4(hit.identity.xy, hit.identity.w, NoIndex), size.layout.x);
    const float rebound = speed < -asfloat(ccdSettings[0].z) ? -speed * material.extra.z : 0.0;
    const uint2 pair = uint2(a, b);
    const float impulse = axisImpulse(pair, r1, r2, normal, rebound, 1.0e30, first, second);
    const float3 relative = pointVelocity(first, r1) - pointVelocity(second, r2);
    const float3 tangent = relative - normal * dot(relative, normal);
    const float slip = length(tangent);
    if (slip > 1.0e-5 && material.friction.x > 0.0)
        axisImpulse(pair, r1, r2, -tangent / slip, 0.0, impulse * material.friction.x, first, second);
    first.center.w = 1.0;
    second.center.w = 1.0;
    ccdMotion[base + a] = first;
    ccdMotion[base + b] = second;
}

[shader("compute")]
[numthreads(CcdLanes, 1, 1)]
void solve_scene_ccd(uint lane : SV_GroupIndex, uint3 group : SV_GroupID)
{
    const uint world = group.y;
    const uint bodies = ccdModel[0].state.z;
    const uint base = world * bodies;
    if (lane == 0)
        ccdSelection = ccdMotion[base].center.w != 0.0 ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    if (workgroupUniformLoad(ccdSelection) != 0u)
        return;
    float duration = asfloat(ccdSettings[0].x);
    for (uint impact = 0; impact < MaxImpacts; ++impact)
    {
        ccdHits[lane] = earliestLaneImpact(world, lane, duration, impact == 0);
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0)
        {
            uint best = 0;
            for (uint i = 1; i < CcdLanes; ++i)
                if (castPrecedes(ccdHits[i], ccdHits[best]))
                    best = i;
            ccdSelection = ccdHits[best].identity.x != NoIndex ? best + 1u : 0u;
        }
        GroupMemoryBarrierWithGroupSync();
        const uint selection = workgroupUniformLoad(ccdSelection);
        const bool found = selection != 0u;
        if (!found && impact == 0)
            break;
        const ShapeCastHit earliest = ccdHits[found ? selection - 1u : 0u];
        const float advance = found ? earliest.normalDistance.w : duration;
        for (uint body = lane; body < bodies; body += CcdLanes)
        {
            const uint index = base + body;
            SweepMotion motion = ccdMotion[index];
            motion.center.xyz += motion.linear.xyz * advance;
            ccdMotion[index] = motion;
        }
        if (!found)
            break;
        syncSweepMotion();
        if (lane == 0 && found)
        {
            solveImpact(world, earliest);
            ccdMotion[base].center.w = 1.0;
        }
        syncSweepMotion();
        GroupMemoryBarrierWithGroupSync();
        duration -= advance;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void publish_scene_ccd(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions size = ccdModel[0];
    if (id.x >= size.state.z || ccdMotion[id.y * size.state.z].center.w != 1.0)
        return;
    const uint4 row = ccdRows[id.x];
    if (row.x == NoIndex)
        return;
    const uint index = id.y * size.state.z + id.x;
    const SweepMotion motion = ccdMotion[index];
    const BodyModel body = ccdBodies[id.x];
    const float3 offset = rotateVector(body.inertialPosition.xyz, motion.rotation);
    const float3 position = motion.center.xyz - offset;
    const uint p = id.y * size.state.x + row.x;
    for (uint axis = 0; axis < 3; ++axis)
        ccdPositions[p + axis] = position[axis];
    if (motion.center.w == 0.0)
        return;
    const float3 velocity = motion.linear.xyz - cross(motion.angular.xyz, offset);
    const float3 spin = rotateVector(motion.angular.xyz, inverseRotation(motion.rotation));
    const uint v = id.y * size.state.y + row.y;
    for (uint axis = 0; axis < 3; ++axis)
    {
        ccdVelocities[v + axis] = velocity[axis];
        ccdVelocities[v + 3 + axis] = spin[axis];
    }
    ccdActivity[index].group = NoIndex;
    ccdActivity[index].marker = 0u;
}
