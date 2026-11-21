// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#include "model.slang"
#include "prescribed_motion.slang"
#include "linear_dispatch.slang"

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
static const uint PreserveDrive = 1;

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
    Pose target;
    float4 worldForce;
    float4 worldTorque;
};

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] StructuredBuffer<BodyModel> bodies;
[[vk::binding(2, 0)]] StructuredBuffer<JointModel> joints;
[[vk::binding(3, 0)]] StructuredBuffer<uint4> spans; // count/fixed dt/substeps, then world/body/start/count
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

void applyDynamic<let Integrate : bool>(uint index, uint4 span, ModelDimensions size, BodyModel body)
{
    const uint2 offset = freeAddresses(size, body, span.x);
    Pose pose = loadOrigin(offset.x);
    const float3 originSpeed = load3(qvel, offset.y);
    const float3 localSpin = load3(qvel, offset.y + 3);
    const Pose undo = pose;
    float3 shift = centerOffset(body, pose);
    float3 spin = rotateVector(localSpin, pose.rotation);
    float3 speed = originSpeed + cross(spin, shift);
    float3 force = float3(0.0);
    float3 torque = float3(0.0);
    for (uint i = span.z; i < span.z + span.w; ++i)
    {
        const BodyCommand command = commands[i];
        const uint operation = command.fields.z;
        if (operation == SetPosition)
        {
            pose.position.xyz = command.value.xyz;
        }
        else if (operation == SetRotation)
        {
            pose.rotation = normalize(command.value);
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
        }
        else
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
    }
    const float4 inverse = conjugate(pose.rotation);
    storeOrigin(offset.x, pose);
    store3(qvel, offset.y, speed - cross(spin, shift));
    store3(qvel, offset.y + 3, rotateVector(spin, inverse));
    if (!Integrate)
        return;
    saved[index] = {undo, float4(originSpeed, 0.0), float4(localSpin, 0.0), float3(0.0), 0u, undo, undo,
                    float4(force, 0.0), float4(torque, 0.0)};
    if (spans[0].z == 1u)
    {
        store3(commandedForce, offset.y, force);
        store3(commandedForce, offset.y + 3, rotateVector(torque + cross(shift, force), inverse));
    }
}

void applyKinematic<let Integrate : bool>(uint index, uint4 span, ModelDimensions size, BodyModel body)
{
    const uint row = span.x * size.layout.y + body.hierarchy.w;
    Pose origin = mocap[row];
    Pose target = origin;
    const Pose undo = origin;
    bool cancelDrive = false;
    for (uint i = span.z; i < span.z + span.w; ++i)
    {
        const BodyCommand command = commands[i];
        switch (command.fields.z)
        {
        case MovePosition:
            target.position.xyz = command.value.xyz;
            break;
        case MoveRotation:
            target.rotation = normalize(command.value);
            break;
        case SetPosition:
            origin.position.xyz = command.value.xyz;
            target = origin;
            cancelDrive = cancelDrive || (command.fields.w & PreserveDrive) == 0;
            break;
        case SetRotation:
            origin.rotation = normalize(command.value);
            target = origin;
            cancelDrive = cancelDrive || (command.fields.w & PreserveDrive) == 0;
            break;
        case ResetMotion:
            target = origin;
            cancelDrive = true;
            break;
        }
    }
    if (!Integrate)
    {
        mocap[row] = origin;
        PrescribedMotion motion = prescribed[row];
        if (cancelDrive)
            motion = {float4(0.0), float4(0.0), float3(0.0), 0u};
        motion.center = origin.position.xyz + centerOffset(body, origin);
        prescribed[row] = motion;
        return;
    }
    const PrescribedMotion previous = prescribed[row];
    saved[index] = {undo, previous.linear, previous.angular, previous.center, previous.step, origin, target,
                    float4(0.0), float4(0.0)};
    if (spans[0].z > 1u)
    {
        mocap[row] = origin;
        return;
    }
    const float timestep = asfloat(spans[0].y);
    const float3 startCenter = origin.position.xyz + centerOffset(body, origin);
    const float3 endCenter = target.position.xyz + centerOffset(body, target);
    const float3 linear = (endCenter - startCenter) / timestep;
    const float3 angular = rotationVector(mulQuat(target.rotation, conjugate(origin.rotation))) / timestep;
    prescribed[row] = {float4(linear, 0.0), float4(angular, 0.0), endCenter, fixedSteps[span.x] + 1u};
    mocap[row] = target;
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
    if (body.dofs.z != 0)
        return;
    const SavedMotion motion = saved[index];
    if (body.hierarchy.w == NoIndex)
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
    const float dt = asfloat(control.y);
    const uint row = span.x * model[0].layout.y + body.hierarchy.w;
    const Pose previous = mocap[row];
    const float3 path = rotationVector(mulQuat(conjugate(motion.start.rotation), motion.target.rotation));
    Pose target = motion.target;
    if (next != control.z)
    {
        target.position = float4(lerp(motion.start.position.xyz, motion.target.position.xyz, alpha), 0.0);
        target.rotation = integrateQuat(motion.start.rotation, path, alpha);
    }
    const float3 before = previous.position.xyz + centerOffset(body, previous);
    const float3 center = target.position.xyz + centerOffset(body, target);
    const float3 linear = (center - before) * (float(control.z) / dt);
    const float3 angular = rotateVector(path, motion.start.rotation) / dt;
    mocap[row] = target;
    prescribed[row] = {float4(linear, 0.0), float4(angular, 0.0), center, fixedSteps[span.x] + 1u};
}

void applyBodyCommands<let Integrate : bool>(uint index)
{
    if (index >= spans[0].x)
        return;
    const uint4 span = spans[index + 1];
    const BodyModel body = bodies[span.y];
    if (body.dofs.z != 0)
    {
        const uint row = span.x * bodies[0].dofs.w + body.dofs.z - 1;
        Pose pose = staticPoses[row];
        if (Integrate)
            saved[index].pose = pose;
        for (uint i = span.z; i < span.z + span.w; ++i)
        {
            const BodyCommand command = commands[i];
            if (command.fields.z == SetPosition)
                pose.position.xyz = command.value.xyz;
            else
                pose.rotation = normalize(command.value);
        }
        staticPoses[row] = pose;
    }
    else if (body.hierarchy.w == NoIndex)
        applyDynamic<Integrate>(index, span, model[0], body);
    else
        applyKinematic<Integrate>(index, span, model[0], body);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void apply_body_commands(uint3 id : SV_DispatchThreadID)
{
    applyBodyCommands<true>(linearWorkIndex<64>(id));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void edit_body_state(uint3 id : SV_DispatchThreadID)
{
    applyBodyCommands<false>(linearWorkIndex<64>(id));
}

void finishCommands<let HasStatus : bool>(uint index)
{
    if (index >= spans[0].x)
        return;
    const uint4 span = spans[index + 1];
    const ModelDimensions size = model[0];
    const BodyModel body = bodies[span.y];
    const bool rejected = HasStatus && status[span.x] != 0;
    if (body.dofs.z != 0)
    {
        if (rejected)
        {
            const uint row = span.x * bodies[0].dofs.w + body.dofs.z - 1;
            staticPoses[row] = saved[index].pose;
        }
        return;
    }
    if (body.hierarchy.w == NoIndex)
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
    }
    else if (rejected)
    {
        const SavedMotion undo = saved[index];
        const uint address = span.x * size.layout.y + body.hierarchy.w;
        mocap[address] = undo.pose;
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
    const float time = clocks[id.y];
    if (body.dofs.y != 0)
    {
        const uint address = id.y * size.state.y + body.dofs.x;
        spin = rotateVector(load3(qvel, address + 3), pose.rotation);
        speed = load3(qvel, address) + cross(spin, offset);
    }
    else if (body.hierarchy.w != NoIndex)
    {
        const PrescribedMotion motion = prescribed[id.y * size.layout.y + body.hierarchy.w];
        if (motion.step == fixedSteps[id.y])
        {
            speed = motion.linear.xyz;
            spin = motion.angular.xyz;
        }
    }
    const uint base = 5 * row;
    bodyStates[base] = asuint(pose.position);
    bodyStates[base + 1] = asuint(pose.rotation);
    bodyStates[base + 2] = asuint(float4(pose.position.xyz + offset, 0.0));
    bodyStates[base + 3] = asuint(float4(speed, 0.0));
    bodyStates[base + 4] = asuint(float4(spin, 0.0));
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
