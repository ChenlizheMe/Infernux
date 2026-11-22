// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "model.slang"
#include "prescribed_motion.slang"
#include "linear_dispatch.slang"
#include "scene_motion.slang"

static const uint SetPosition = 0;
static const uint SetRotation = 1;
static const uint SetLinearVelocity = 2;
static const uint SetAngularVelocity = 3;
static const uint AddForce = 4;
static const uint AddTorque = 5;
static const uint AddForceAtPosition = 6;
static const uint MovePosition = 7;
static const uint MoveRotation = 8;
static const uint ResetMotion = 9;
static const uint Sleep = 10;
static const uint Wake = 11;
static const uint ApplyPendingWrench = 12;
static const uint PreserveDrive = 1;
// Shares the snapshot lane used by dynamic sleep time. Negative zero is a
// finite marker distinct from the dynamic timer's nonnegative bit patterns.
static const uint PersistentKinematicVelocity = 0x80000000u;

struct BodyCommand
{
    uint4 fields; // world, body, operation, force mode or pose flags
    float4 value;
    float4 point;
};

struct SavedMotion
{
    Pose pose;
    float4 linear;
    float4 angular;
    float3 center;
    uint step;
    Pose start;
    float4 centerTravel;
    float4 angularTravel;
    float4 worldForce;
    float4 worldTorque;
};

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] StructuredBuffer<BodyModel> bodies;
[[vk::binding(2, 0)]] StructuredBuffer<JointModel> joints;
[[vk::binding(3, 0)]] StructuredBuffer<uint4> spans; // count/unused/substeps, then world/body/start/count
[[vk::binding(4, 0)]] StructuredBuffer<BodyCommand> commands;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> qpos;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> qvel;
[[vk::binding(7, 0)]] RWStructuredBuffer<Pose> mocap;
[[vk::binding(8, 0)]] RWStructuredBuffer<float> commandedForce;
[[vk::binding(9, 0)]] StructuredBuffer<float> clocks;
[[vk::binding(10, 0)]] StructuredBuffer<uint> status;
[[vk::binding(11, 0)]] RWStructuredBuffer<SavedMotion> saved;
[[vk::binding(12, 0)]] RWStructuredBuffer<PrescribedMotion> prescribed;
[[vk::binding(13, 0)]] StructuredBuffer<Pose> bodyPoses;
[[vk::binding(14, 0)]] RWStructuredBuffer<uint4> bodyStates;
[[vk::binding(15, 0)]] StructuredBuffer<uint> fixedSteps;
[[vk::binding(16, 0)]] RWStructuredBuffer<Pose> staticPoses;
[[vk::binding(17, 0)]] StructuredBuffer<uint4> previousStates;
[[vk::binding(18, 0)]] RWStructuredBuffer<float> importedClocks;
[[vk::binding(19, 0)]] RWStructuredBuffer<uint> importedSteps;
[[vk::binding(20, 0)]] StructuredBuffer<uint> previousAddresses;
[[vk::binding(21, 0)]] StructuredBuffer<SceneRigidMotion> sceneMotion;
[[vk::binding(22, 0)]] StructuredBuffer<float> fixedTimeSteps;
[[vk::binding(23, 0)]] RWStructuredBuffer<SceneBodyActivity> bodyActivity;
[[vk::binding(24, 0)]] RWStructuredBuffer<float> pendingWrenches;
[[vk::binding(25, 0)]] StructuredBuffer<float> previousWrenches;
[[vk::binding(26, 0)]] StructuredBuffer<uint4> motionSeedRows;
[[vk::binding(27, 0)]] StructuredBuffer<uint4> motionSeedActivity;
[[vk::binding(28, 0)]] RWStructuredBuffer<uint4> motionSeedOutput;

float4 conjugate(float4 rotation)
{
    return float4(rotation.x, -rotation.yzw);
}

float3 load3(RWStructuredBuffer<float> source, uint offset)
{
    return float3(source[offset], source[offset + 1], source[offset + 2]);
}

void store3(RWStructuredBuffer<float> destination, uint offset, float3 value)
{
    destination[offset] = value.x;
    destination[offset + 1] = value.y;
    destination[offset + 2] = value.z;
}

Pose loadOrigin(uint index)
{
    return {float4(load3(qpos, index), 0.0), float4(qpos[index + 3], load3(qpos, index + 4))};
}

void storeOrigin(uint index, Pose pose)
{
    store3(qpos, index, pose.position.xyz);
    qpos[index + 3] = pose.rotation.x;
    store3(qpos, index + 4, pose.rotation.yzw);
}

uint2 freeAddresses(ModelDimensions size, BodyModel body, uint world)
{
    return world * size.state.xy + joints[body.hierarchy.y].indices.yz;
}

float3 centerOffset(BodyModel body, Pose pose)
{
    return rotateVector(body.inertialPosition.xyz, pose.rotation);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pack_accepted_motion(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= model[0].state.z)
        return;
    const uint output = 5 * id.x;
    const uint4 row = motionSeedRows[id.x];
    uint4 position = uint4(0), rotation = uint4(0), center = uint4(0), linear = uint4(0), angular = uint4(0);
    if (id.x == 0)
    {
        position = row;
        position.w = 0;
    }
    else if (row.w != NoIndex)
    {
        const uint input = 4 * row.w;
        const float4 q = asfloat(previousStates[input + 1]).wxyz;
        const float3 com = asfloat(previousStates[input].xyz);
        const float3 origin = com - rotateVector(asfloat(row.xyz), q);
        const bool sleeping = motionSeedActivity[row.w - motionSeedRows[0].w].x == 0;
        position = asuint(float4(origin, 0.0));
        rotation = asuint(q);
        center = asuint(float4(com, sleeping ? 1.0 : 0.0));
        linear = uint4(previousStates[input + 2].xyz, 0);
        angular = uint4(previousStates[input + 3].xyz, 0);
    }
    motionSeedOutput[output] = position;
    motionSeedOutput[output + 1] = rotation;
    motionSeedOutput[output + 2] = center;
    motionSeedOutput[output + 3] = linear;
    motionSeedOutput[output + 4] = angular;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void import_body_states(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions size = model[0];
    if (id.x >= size.state.z)
        return;
    const uint first = id.y * size.state.z;
    const uint address = previousAddresses[first + id.x];
    if (address == NoIndex)
        return;
    if (id.x == 0)
    {
        const uint4 header = previousStates[address];
        importedClocks[id.y] = asfloat(header.x);
        importedSteps[id.y] = header.z;
        return;
    }
    const BodyModel body = bodies[id.x];
    const Pose pose = {asfloat(previousStates[address]), asfloat(previousStates[address + 1])};
    const float3 linear = projectSceneAxes(asfloat(previousStates[address + 3].xyz), body.subtree.w);
    const float3 spin = projectSceneAxes(asfloat(previousStates[address + 4].xyz), body.subtree.w >> 3u);
    const float3 shift = centerOffset(body, pose);
    if (body.dofs.y != 0)
    {
        const uint2 offset = freeAddresses(size, body, id.y);
        storeOrigin(offset.x, pose);
        store3(qvel, offset.y, linear - cross(spin, shift));
        store3(qvel, offset.y + 3, rotateVector(spin, conjugate(pose.rotation)));
        bodyActivity[first + id.x] = {previousStates[address + 3].w,
            asfloat(previousStates[address + 2].w) != 0.0 ? id.x : NoIndex, previousStates[address + 4].w, 0u};
    }
    else if (body.hierarchy.w != NoIndex)
    {
        const uint row = id.y * size.layout.y + body.hierarchy.w;
        mocap[row] = pose;
        const uint accepted = previousStates[previousAddresses[first]].z;
        const uint motionRow = id.y * prescribedStride(size, bodies[0]) + body.hierarchy.w;
        const float mode = float(previousStates[address + 3].w == PersistentKinematicVelocity);
        prescribed[motionRow] = {float4(linear, mode), float4(spin, 0.0),
                                 pose.position.xyz + shift, accepted};
    }
    else if (body.dofs.z != 0)
    {
        staticPoses[id.y * bodies[0].dofs.w + body.dofs.z - 1] = pose;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void import_pending_wrenches(uint3 id : SV_DispatchThreadID)
{
    const uint width = model[0].state.z;
    if (id.x == 0 || id.x >= width || bodies[id.x].dofs.y == 0)
        return;
    const uint row = id.y * width + id.x;
    const uint address = previousAddresses[row];
    if (address == NoIndex)
        return;
    const uint source = (address / 5) * 6;
    store3(pendingWrenches, row * 6, float3(previousWrenches[source], previousWrenches[source + 1],
                                             previousWrenches[source + 2]));
    store3(pendingWrenches, row * 6 + 3, float3(previousWrenches[source + 3], previousWrenches[source + 4],
                                                 previousWrenches[source + 5]));
}

void applyDynamic<let Integrate : bool, let SceneLimits : bool>(uint index, uint4 span, ModelDimensions size,
                                                               BodyModel body)
{
    const uint2 offset = freeAddresses(size, body, span.x);
    Pose pose = loadOrigin(offset.x);
    const float3 originSpeed = load3(qvel, offset.y);
    const float3 localSpin = load3(qvel, offset.y + 3);
    const Pose undo = pose;
    float3 shift = centerOffset(body, pose);
    float3 spin = rotateVector(localSpin, pose.rotation);
    float3 speed = originSpeed + cross(spin, shift);
    const uint activityRow = span.x * size.state.z + span.y;
    const uint wrenchRow = activityRow * 6;
    float3 force = load3(pendingWrenches, wrenchRow);
    float3 torque = load3(pendingWrenches, wrenchRow + 3);
    SceneBodyActivity activity = bodyActivity[activityRow];
    uint sleep = activity.group;
    for (uint i = span.z; i < span.z + span.w; ++i)
    {
        const BodyCommand command = commands[i];
        const uint operation = command.fields.z;
        if (operation == Sleep)
        {
            speed = spin = force = torque = float3(0.0);
            sleep = span.y;
        }
        else if (operation == Wake)
        {
            sleep = NoIndex;
            activity.marker = 0;
        }
        else if (operation == SetPosition)
        {
            if (any(pose.position.xyz != command.value.xyz))
            {
                sleep = NoIndex;
                activity.marker = 0;
            }
            pose.position.xyz = command.value.xyz;
        }
        else if (operation == SetRotation)
        {
            const float4 target = normalize(command.value);
            if (!all(pose.rotation == target) && !all(pose.rotation == -target))
            {
                sleep = NoIndex;
                activity.marker = 0;
            }
            pose.rotation = target;
            shift = centerOffset(body, pose);
        }
        else if (operation == SetLinearVelocity)
        {
            speed = command.value.xyz;
        }
        else if (operation == SetAngularVelocity)
        {
            spin = command.value.xyz;
        }
        else if (operation == ResetMotion)
        {
            spin = speed = force = torque = float3(0.0);
            sleep = NoIndex;
            activity.marker = 0;
        }
        else if (operation != ApplyPendingWrench)
        {
            const uint mode = command.fields.w;
            const bool massIndependent = (mode & 1u) != 0;
            const bool continuous = mode < 2u;
            const float4 axes = mulQuat(pose.rotation, body.inertialRotation);
            const float4 inverseAxes = conjugate(axes);
            float3 linear = float3(0.0);
            float3 angular = float3(0.0);
            if (operation == AddTorque)
            {
                angular = command.value.xyz;
                if (massIndependent)
                    angular = rotateVector(body.inertiaMass.xyz * rotateVector(angular, inverseAxes), axes);
            }
            else
            {
                linear = command.value.xyz * (massIndependent ? body.inertiaMass.w : 1.0);
                if (operation == AddForceAtPosition)
                    angular = cross(command.point.xyz - pose.position.xyz - shift, linear);
            }
            linear = projectSceneAxes(linear, body.subtree.w);
            angular = projectSceneAxes(angular, body.subtree.w >> 3u);
            if (continuous)
            {
                force += linear;
                torque += angular;
            }
            else
            {
                speed += linear / body.inertiaMass.w;
                spin += rotateVector(rotateVector(angular, inverseAxes) / body.inertiaMass.xyz, axes);
            }
        }
        speed = projectSceneAxes(speed, body.subtree.w);
        spin = projectSceneAxes(spin, body.subtree.w >> 3u);
        if (SceneLimits)
        {
            const float2 limits = sceneMotion[span.y].velocityLimit;
            speed = clampSceneSpeed(speed, limits.x);
            spin = clampSceneSpeed(spin, limits.y);
        }
        if (any(speed != 0.0) || any(spin != 0.0) || any(force != 0.0) || any(torque != 0.0))
            sleep = NoIndex;
    }
    if (activity.group != sleep || any(speed != 0.0) || any(spin != 0.0) ||
        any(force != 0.0) || any(torque != 0.0))
        activity.marker = 0;
    activity.group = sleep;
    bodyActivity[activityRow] = activity;
    const float4 inverse = conjugate(pose.rotation);
    storeOrigin(offset.x, pose);
    store3(qvel, offset.y, speed - cross(spin, shift));
    store3(qvel, offset.y + 3, rotateVector(spin, inverse));
    if (!Integrate)
    {
        store3(pendingWrenches, wrenchRow, force);
        store3(pendingWrenches, wrenchRow + 3, torque);
        return;
    }
    saved[index] = {undo, float4(originSpeed, 0.0), float4(localSpin, 0.0), float3(0.0), 0u,
                    undo, float4(0.0), float4(0.0),
                    float4(force, 0.0), float4(torque, 0.0)};
    if (spans[0].z == 1u)
    {
        store3(commandedForce, offset.y, force);
        store3(commandedForce, offset.y + 3, rotateVector(torque + cross(shift, force), inverse));
    }
}

Pose loadPrescribedPose(uint world, BodyModel body, ModelDimensions size)
{
    if (body.dofs.z != 0)
        return staticPoses[world * bodies[0].dofs.w + body.dofs.z - 1];
    return mocap[world * size.layout.y + body.hierarchy.w];
}

void storePrescribedPose(uint world, BodyModel body, ModelDimensions size, Pose pose)
{
    if (body.dofs.z != 0)
        staticPoses[world * bodies[0].dofs.w + body.dofs.z - 1] = pose;
    else
        mocap[world * size.layout.y + body.hierarchy.w] = pose;
}

uint prescribedAddress(uint world, BodyModel body, ModelDimensions size)
{
    return world * prescribedStride(size, bodies[0]) + prescribedRow(body, size.layout.y);
}

void writePrescribedStep(uint world, BodyModel body, ModelDimensions size, Pose start, float3 travel, float3 path,
                         float fraction, float timestep, float velocityMode)
{
    const float3 center = start.position.xyz + centerOffset(body, start) + travel * fraction;
    const float4 rotation = integrateQuat(start.rotation, path, fraction);
    Pose pose = {float4(0.0), rotation};
    pose.position.xyz = center - centerOffset(body, pose);
    const float3 spin = rotateVector(path, start.rotation) / timestep;
    storePrescribedPose(world, body, size, pose);
    prescribed[prescribedAddress(world, body, size)] = {
        float4(travel / timestep, velocityMode), float4(spin, 0.0), center, fixedSteps[world] + 1u};
}

void applyPrescribed<let Integrate : bool, let SceneLimits : bool>(uint index, uint4 span, ModelDimensions size,
                                                                   BodyModel body)
{
    const uint row = prescribedAddress(span.x, body, size);
    Pose origin = loadPrescribedPose(span.x, body, size);
    Pose target = origin;
    const Pose undo = origin;
    bool moving = false;
    float2 timing = float2(0.0);
    const PrescribedMotion previous = prescribed[row];
    bool velocityMode = previous.linear.w != 0.0;
    const bool hasMotion = velocityMode || previous.step == fixedSteps[span.x];
    float3 linear = hasMotion ? previous.linear.xyz : float3(0.0);
    float3 angular = hasMotion ? previous.angular.xyz : float3(0.0);
    for (uint i = span.z; i < span.z + span.w; ++i)
    {
        const BodyCommand command = commands[i];
        switch (command.fields.z)
        {
        case MovePosition:
            if (!moving)
                target = origin;
            target.position.xyz = command.value.xyz;
            timing = command.point.xy;
            moving = true;
            break;
        case MoveRotation:
            if (!moving)
                target = origin;
            target.rotation = normalize(command.value);
            timing = command.point.xy;
            moving = true;
            break;
        case SetPosition:
            origin.position.xyz = command.value.xyz;
            if ((command.fields.w & PreserveDrive) == 0)
            {
                moving = false;
                if (!velocityMode)
                    linear = angular = float3(0.0);
            }
            break;
        case SetRotation:
            origin.rotation = normalize(command.value);
            if ((command.fields.w & PreserveDrive) == 0)
            {
                moving = false;
                if (!velocityMode)
                    linear = angular = float3(0.0);
            }
            break;
        case SetLinearVelocity:
            linear = projectSceneAxes(command.value.xyz, body.subtree.w);
            if (SceneLimits)
                linear = clampSceneSpeed(linear, sceneMotion[span.y].velocityLimit.x);
            velocityMode = true;
            moving = false;
            break;
        case SetAngularVelocity:
            angular = projectSceneAxes(command.value.xyz, body.subtree.w >> 3u);
            if (SceneLimits)
                angular = clampSceneSpeed(angular, sceneMotion[span.y].velocityLimit.y);
            velocityMode = true;
            moving = false;
            break;
        case Sleep:
        case ResetMotion:
            linear = angular = float3(0.0);
            velocityMode = false;
            moving = false;
            break;
        }
    }
    if (any(origin.position.xyz != undo.position.xyz) ||
        (!all(origin.rotation == undo.rotation) && !all(origin.rotation == -undo.rotation)))
        bodyActivity[span.x * size.state.z + span.y].marker = fixedSteps[span.x] + 1u;
    if (!Integrate)
    {
        storePrescribedPose(span.x, body, size, origin);
        prescribed[row] = {float4(linear, float(velocityMode)), float4(angular, 0.0),
                           origin.position.xyz + centerOffset(body, origin), fixedSteps[span.x]};
        return;
    }
    const float timestep = fixedTimeSteps[span.x];
    float3 travel = float3(0.0);
    float3 path = float3(0.0);
    if (moving)
    {
        velocityMode = false;
        const float3 delta = target.position.xyz - origin.position.xyz;
        const float distance = length(delta);
        const float limit = timing.x * timing.y;
        if (timing.y > 0.0 && distance > limit)
        {
            origin.position.xyz = target.position.xyz - delta * (limit / distance);
            origin.rotation = target.rotation;
        }
        const float3 startCenter = origin.position.xyz + centerOffset(body, origin);
        const float3 endCenter = target.position.xyz + centerOffset(body, target);
        const float fraction = timestep / timing.x;
        travel = projectSceneAxes(endCenter - startCenter, body.subtree.w) * fraction;
        path = projectSceneSpin(rotationVector(mulQuat(conjugate(origin.rotation), target.rotation)), origin.rotation,
                                body.subtree.w >> 3u) * fraction;
    }
    else if (velocityMode)
    {
        travel = linear * timestep;
        path = rotateVector(angular, conjugate(origin.rotation)) * timestep;
    }
    saved[index] = {undo, previous.linear, previous.angular, previous.center, previous.step, origin,
                    float4(travel, float(velocityMode)), float4(path, 0.0),
                    float4(0.0), float4(0.0)};
    if (spans[0].z > 1u)
    {
        storePrescribedPose(span.x, body, size, origin);
        return;
    }
    writePrescribedStep(span.x, body, size, origin, travel, path, 1.0, timestep, float(velocityMode));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void advance_commanded_substep(uint3 id : SV_DispatchThreadID)
{
    const uint index = linearWorkIndex<64>(id);
    if (index >= spans[0].x)
        return;
    const uint4 span = spans[index + 1];
    const BodyModel body = bodies[span.y];
    const SavedMotion motion = saved[index];
    if (body.dofs.z == 0 && body.hierarchy.w == NoIndex)
    {
        const uint2 address = freeAddresses(model[0], body, span.x);
        const Pose pose = loadOrigin(address.x);
        const float3 force = motion.worldForce.xyz;
        const float3 torque = motion.worldTorque.xyz + cross(centerOffset(body, pose), force);
        store3(commandedForce, address.y, force);
        store3(commandedForce, address.y + 3, rotateVector(torque, conjugate(pose.rotation)));
        return;
    }

    uint count, stride;
    fixedSteps.GetDimensions(count, stride);
    const uint next = fixedSteps[count - 1u] + 1u;
    const uint4 control = spans[0];
    const float alpha = float(next) / float(control.z);
    const float dt = fixedTimeSteps[span.x];
    writePrescribedStep(span.x, body, model[0], motion.start, motion.centerTravel.xyz, motion.angularTravel.xyz,
                        alpha, dt, motion.centerTravel.w);
}

void applyBodyCommands<let Integrate : bool, let SceneLimits : bool>(uint index)
{
    if (index >= spans[0].x)
        return;
    const uint4 span = spans[index + 1];
    const BodyModel body = bodies[span.y];
    if (body.dofs.z == 0 && body.hierarchy.w == NoIndex)
        applyDynamic<Integrate, SceneLimits>(index, span, model[0], body);
    else
        applyPrescribed<Integrate, SceneLimits>(index, span, model[0], body);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void apply_body_commands(uint3 id : SV_DispatchThreadID)
{
    applyBodyCommands<true, false>(linearWorkIndex<64>(id));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void edit_body_state(uint3 id : SV_DispatchThreadID)
{
    applyBodyCommands<false, false>(linearWorkIndex<64>(id));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void apply_scene_body_commands(uint3 id : SV_DispatchThreadID)
{
    applyBodyCommands<true, true>(linearWorkIndex<64>(id));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void edit_scene_body_state(uint3 id : SV_DispatchThreadID)
{
    applyBodyCommands<false, true>(linearWorkIndex<64>(id));
}

void finishCommands<let HasStatus : bool>(uint index)
{
    if (index >= spans[0].x)
        return;
    const uint4 span = spans[index + 1];
    const ModelDimensions size = model[0];
    const BodyModel body = bodies[span.y];
    const bool rejected = HasStatus && status[span.x] != 0;
    if (body.dofs.z == 0 && body.hierarchy.w == NoIndex)
    {
        const uint2 address = freeAddresses(size, body, span.x);
        store3(commandedForce, address.y, float3(0.0));
        store3(commandedForce, address.y + 3, float3(0.0));
        if (rejected)
        {
            const SavedMotion undo = saved[index];
            storeOrigin(address.x, undo.pose);
            store3(qvel, address.y, undo.linear.xyz);
            store3(qvel, address.y + 3, undo.angular.xyz);
        }
        else
        {
            const uint wrenchRow = (span.x * size.state.z + span.y) * 6;
            store3(pendingWrenches, wrenchRow, float3(0.0));
            store3(pendingWrenches, wrenchRow + 3, float3(0.0));
        }
    }
    else if (rejected)
    {
        const SavedMotion undo = saved[index];
        const uint address = prescribedAddress(span.x, body, size);
        storePrescribedPose(span.x, body, size, undo.pose);
        prescribed[address] = {undo.linear, undo.angular, undo.center, undo.step};
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void finish_body_commands(uint3 id : SV_DispatchThreadID)
{
    finishCommands<false>(linearWorkIndex<64>(id));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void finish_contact_body_commands(uint3 id : SV_DispatchThreadID)
{
    finishCommands<true>(linearWorkIndex<64>(id));
}

void publishBodies<let HasStatus : bool>(uint3 id)
{
    const ModelDimensions size = model[0];
    if (id.x >= size.state.z)
        return;
    const BodyModel body = bodies[id.x];
    const uint row = id.y * size.state.z + id.x;
    const Pose pose = bodyPoses[row];
    const float3 offset = centerOffset(body, pose);
    float3 speed = float3(0.0);
    float3 spin = float3(0.0);
    float velocityMode = 0.0;
    const float time = clocks[id.y];
    if (body.dofs.y != 0)
    {
        const uint address = id.y * size.state.y + body.dofs.x;
        spin = rotateVector(load3(qvel, address + 3), pose.rotation);
        speed = load3(qvel, address) + cross(spin, offset);
    }
    else if (body.dofs.z != 0 || body.hierarchy.w != NoIndex)
    {
        const PrescribedMotion motion = prescribed[prescribedAddress(id.y, body, size)];
        if (motion.step == fixedSteps[id.y])
        {
            speed = motion.linear.xyz;
            spin = motion.angular.xyz;
        }
        velocityMode = motion.linear.w;
    }
    const uint base = 5 * row;
    bodyStates[base] = asuint(pose.position);
    bodyStates[base + 1] = asuint(pose.rotation);
    const SceneBodyActivity activity = bodyActivity[row];
    const bool dynamic = body.dofs.y != 0;
    const bool sleeping = dynamic && activity.group != NoIndex;
    bodyStates[base + 2] = asuint(float4(pose.position.xyz + offset, sleeping ? 1.0 : 0.0));
    bodyStates[base + 3] = uint4(asuint(speed), velocityMode != 0.0 ? PersistentKinematicVelocity : 0u);
    bodyStates[base + 4] = asuint(float4(spin, 0.0));
    if (dynamic)
    {
        bodyStates[base + 3].w = activity.marker;
        bodyStates[base + 4].w = activity.supports;
    }
    if (id.x == 0)
    {
        uint count, stride;
        clocks.GetDimensions(count, stride);
        const uint value = HasStatus ? status[id.y] : 0u;
        bodyStates[5 * count * size.state.z + id.y] = uint4(asuint(time), value, fixedSteps[id.y], 0);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void publish_body_states(uint3 id : SV_DispatchThreadID)
{
    publishBodies<false>(id);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void publish_contact_body_states(uint3 id : SV_DispatchThreadID)
{
    publishBodies<true>(id);
}
