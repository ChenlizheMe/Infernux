// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/smooth.py joint transmission and
// _src/forward.py actuation and activation advancement.

#include "model.slang"

static const uint ActivationIntegrator = 1;
static const uint ActivationFilter = 2;
static const uint ActivationFilterExact = 3;
static const uint FixedGain = 0;
static const uint AffineBias = 1;

static const uint ParentFrame = 1;
static const uint LimitControl = 2;
static const uint LimitActivation = 4;
static const uint LimitForce = 8;
static const uint EarlyActivation = 16;
static const float Pi = 3.141592653589793;

struct JointActuatorModel
{
    uint4 indices; // joint, control, activation, flags
    uint4 modes; // activation dynamics, gain, bias
    float4 gearFirst;
    float4 gearLast;
    float4 gain;
    float4 bias;
    float4 controlActivationRange;
    float4 forceRangeAndFilterTime;
};

struct JointActuatorStep
{
    float4 firstMoment;
    float4 lastMoment;
    float4 values; // length, velocity, force, activation derivative
};

struct JointActuatorDof
{
    uint4 entries; // first adjacency, count, force limit enabled
    float4 forceRange;
};

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> actuatorDimensions; // actuator count, control count, activation count
[[vk::binding(2, 0)]] StructuredBuffer<JointModel> joints;
[[vk::binding(3, 0)]] StructuredBuffer<JointActuatorModel> actuators;
[[vk::binding(4, 0)]] StructuredBuffer<float> timestep;
[[vk::binding(5, 0)]] StructuredBuffer<float> qpos;
[[vk::binding(6, 0)]] StructuredBuffer<float> qvel;
[[vk::binding(7, 0)]] StructuredBuffer<float> control;
[[vk::binding(8, 0)]] RWStructuredBuffer<float> activation;
[[vk::binding(9, 0)]] RWStructuredBuffer<JointActuatorStep> actuatorStep;
[[vk::binding(10, 0)]] StructuredBuffer<JointActuatorDof> actuatorDofs;
[[vk::binding(11, 0)]] StructuredBuffer<uint2> actuatorAdjacency; // actuator id, joint-relative dof
[[vk::binding(12, 0)]] RWStructuredBuffer<float> qfrcActuator;

float3 rotationVector(float4 rotation)
{
    const float sine = length(rotation.yzw);
    if (sine == 0.0)
        return float3(0.0);
    float angle = 2.0 * atan2(sine, rotation.x);
    if (angle > Pi)
        angle -= 2.0 * Pi;
    return rotation.yzw * (angle / sine);
}

float4 jointRotation(uint address)
{
    return normalize(float4(qpos[address], qpos[address + 1], qpos[address + 2], qpos[address + 3]));
}

float3 parentGear(float3 gear, float4 rotation)
{
    return rotateVector(gear, float4(rotation.x, -rotation.yzw));
}

float servoPeriod(JointActuatorModel actuator)
{
    if (joints[actuator.indices.x].indices.x == JointBall && actuator.modes.y == FixedGain &&
        actuator.modes.z == AffineBias && actuator.gain.x == -actuator.bias.y)
        return 2.0 * Pi * length(actuator.gearFirst.xyz);
    return 0.0;
}

float nearestSetpoint(float value, float reference, float period)
{
    if (period > 0.0)
    {
        const float turns = (value - reference) / period;
        value -= period * sign(turns) * floor(abs(turns) + 0.5);
    }
    return value;
}

float nextActivation(JointActuatorModel actuator, float previous, float derivative, float dt)
{
    if (actuator.modes.x == ActivationFilterExact)
    {
        const float tau = max(1.0e-15, actuator.forceRangeAndFilterTime.z);
        dt = tau * (1.0 - exp(-dt / tau));
    }
    const float value = previous + derivative * dt;
    const float2 range = actuator.controlActivationRange.zw;
    return (actuator.indices.w & LimitActivation) != 0 ? clamp(value, range.x, range.y) : value;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void evaluate_joint_actuators(uint3 id : SV_DispatchThreadID)
{
    const uint4 sizes = actuatorDimensions[0];
    if (id.x >= sizes.x)
        return;
    const JointActuatorModel actuator = actuators[id.x];
    const uint4 joint = joints[actuator.indices.x].indices;
    const uint flags = actuator.indices.w;
    const uint q = id.y * model[0].state.x + joint.y;
    const uint v = id.y * model[0].state.y + joint.z;
    float3 firstMoment = actuator.gearFirst.xyz;
    float3 lastMoment = float3(0.0);
    float coordinate = 0.0;
    float velocity = 0.0;
    if (joint.x == JointFree)
    {
        lastMoment = actuator.gearLast.xyz;
        if ((flags & ParentFrame) != 0)
            lastMoment = parentGear(lastMoment, jointRotation(q + 3));
        velocity = dot(firstMoment, float3(qvel[v], qvel[v + 1], qvel[v + 2])) +
                   dot(lastMoment, float3(qvel[v + 3], qvel[v + 4], qvel[v + 5]));
    }
    else if (joint.x == JointBall)
    {
        const float4 rotation = jointRotation(q);
        if ((flags & ParentFrame) != 0)
            firstMoment = parentGear(firstMoment, rotation);
        coordinate = dot(rotationVector(rotation), firstMoment);
        velocity = dot(firstMoment, float3(qvel[v], qvel[v + 1], qvel[v + 2]));
    }
    else
    {
        coordinate = qpos[q] * firstMoment.x;
        velocity = qvel[v] * firstMoment.x;
    }

    float command = control[id.y * sizes.y + actuator.indices.y];
    if ((flags & LimitControl) != 0)
        command = clamp(command, actuator.controlActivationRange.x, actuator.controlActivationRange.y);

    const float period = servoPeriod(actuator);
    float derivative = 0.0;
    if (actuator.indices.z != NoIndex)
    {
        const float current = activation[id.y * sizes.z + actuator.indices.z];
        if (actuator.modes.x == ActivationIntegrator)
            derivative = command;
        if (actuator.modes.x == ActivationFilter || actuator.modes.x == ActivationFilterExact)
        {
            const float target = nearestSetpoint(command, current, period);
            derivative = (target - current) / max(1.0e-15, actuator.forceRangeAndFilterTime.z);
        }
        command = (flags & EarlyActivation) != 0 ? nextActivation(actuator, current, derivative, timestep[id.y]) : current;
    }

    const float3 terms = float3(1.0, coordinate, velocity);
    const float gain = actuator.modes.y == FixedGain ? actuator.gain.x : dot(actuator.gain.xyz, terms);
    const float bias = actuator.modes.z == AffineBias ? dot(actuator.bias.xyz, terms) : 0.0;
    float force = gain * nearestSetpoint(command, coordinate, period) + bias;
    if ((flags & LimitForce) != 0)
        force = clamp(force, actuator.forceRangeAndFilterTime.x, actuator.forceRangeAndFilterTime.y);

    const JointActuatorStep result = {float4(firstMoment, 0.0), float4(lastMoment, 0.0),
                                      float4(coordinate, velocity, force, derivative)};
    actuatorStep[id.y * sizes.x + id.x] = result;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_joint_actuator_force(uint3 id : SV_DispatchThreadID)
{
    const uint nv = model[0].state.y;
    if (id.x >= nv)
        return;
    const JointActuatorDof dof = actuatorDofs[id.x];
    const uint base = id.y * actuatorDimensions[0].x;
    float force = 0.0;
    for (uint i = 0; i < dof.entries.y; ++i)
    {
        const uint2 edge = actuatorAdjacency[dof.entries.x + i];
        const JointActuatorStep step = actuatorStep[base + edge.x];
        const float moment = edge.y < 3 ? step.firstMoment[edge.y] : step.lastMoment[edge.y - 3];
        force += moment * step.values.z;
    }
    if (dof.entries.z != 0)
        force = clamp(force, dof.forceRange.x, dof.forceRange.y);
    qfrcActuator[id.y * nv + id.x] = force;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void advance_joint_activation(uint3 id : SV_DispatchThreadID)
{
    const uint4 sizes = actuatorDimensions[0];
    if (id.x >= sizes.x)
        return;
    const JointActuatorModel actuator = actuators[id.x];
    if (actuator.indices.z == NoIndex)
        return;
    const uint index = id.y * sizes.z + actuator.indices.z;
    const float4 values = actuatorStep[id.y * sizes.x + id.x].values;
    const float next = nextActivation(actuator, activation[index], values.w, timestep[id.y]);
    activation[index] = nearestSetpoint(next, values.x, servoPeriod(actuator));
}
