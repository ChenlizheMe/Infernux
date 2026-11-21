// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/smooth.py actuator transmission and
// _src/forward.py actuation and activation advancement.

#include "model.slang"
#include "tendon_data.slang"

static const uint ActivationIntegrator = 1;
static const uint ActivationFilter = 2;
static const uint ActivationFilterExact = 3;
static const uint ActivationMuscle = 4;
static const uint ActivationDcMotor = 5;
static const uint FixedGain = 0;
static const uint AffineGain = 1;
static const uint MuscleGain = 2;
static const uint DcMotorGain = 3;
static const uint AffineBias = 1;
static const uint MuscleBias = 2;
static const uint DcMotorBias = 3;
static const float MinActuatorValue = 1.0e-15;

static const uint ParentFrame = 1;
static const uint LimitControl = 2;
static const uint LimitActivation = 4;
static const uint LimitForce = 8;
static const uint EarlyActivation = 16;
static const uint ReferenceSite = 32;
static const uint JointTransmission = 0;
static const uint TendonTransmission = 1;
static const uint SiteTransmission = 2;
static const uint BodyTransmission = 3;

struct ActuatorModel
{
    uint4 indices; // transmission target, control, activation, flags
    uint4 modes; // activation dynamics, gain, bias, transmission
    float4 gearFirst;
    float4 gearLast;
    float4 gain;
    float4 bias;
    float4 controlActivationRange;
    float2 forceRange;
    float filterTime;
    uint dynamicsIndex;
};

struct ActuatorStep
{
    float4 firstMoment;
    float4 lastMoment;
    float4 values; // length, velocity, force, mechanical force for tendon limiting
};

struct ActivationState
{
    float value;
    float derivative;
};

struct ActuatorDof
{
    uint4 entries; // first adjacency, count, force limit enabled
    float4 forceRange;
};

struct TendonActuatorGroup
{
    uint4 entries; // first actuator in the list, count, force limit enabled
    float4 forceRange;
};

[[vk::binding(0, 0)]] StructuredBuffer<ModelDimensions> model;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> actuatorDimensions; // counts and sparse strides, see README
[[vk::binding(2, 0)]] StructuredBuffer<JointModel> joints;
[[vk::binding(3, 0)]] StructuredBuffer<ActuatorModel> actuators;
[[vk::binding(4, 0)]] StructuredBuffer<float> timestep;
[[vk::binding(5, 0)]] StructuredBuffer<float> qpos;
[[vk::binding(6, 0)]] StructuredBuffer<float> qvel;
[[vk::binding(7, 0)]] StructuredBuffer<float> control;
[[vk::binding(8, 0)]] RWStructuredBuffer<ActivationState> activation;
[[vk::binding(9, 0)]] RWStructuredBuffer<ActuatorStep> actuatorStep;
[[vk::binding(10, 0)]] StructuredBuffer<ActuatorDof> actuatorDofs;
[[vk::binding(11, 0)]] StructuredBuffer<uint2> actuatorAdjacency; // actuator id, joint-relative DOF or sparse moment index
[[vk::binding(12, 0)]] RWStructuredBuffer<float> qfrcActuator;
[[vk::binding(13, 0)]] StructuredBuffer<TendonState> tendonState;
[[vk::binding(15, 0)]] StructuredBuffer<float> tendonJacobian;
[[vk::binding(16, 0)]] StructuredBuffer<TendonActuatorGroup> tendonActuatorGroups;
[[vk::binding(17, 0)]] StructuredBuffer<uint> tendonActuatorList;
[[vk::binding(21, 0)]] StructuredBuffer<float4> transmissionTreeCom;
[[vk::binding(22, 0)]] StructuredBuffer<SpatialMotion> transmissionCenteredDof;
[[vk::binding(23, 0)]] RWStructuredBuffer<float> actuatorMoments;
[[vk::binding(24, 0)]] StructuredBuffer<DofModel> actuationDofs;
[[vk::binding(25, 0)]] StructuredBuffer<float> qfrcGravityCompensation;

#include "site_transmission.slang"
#include "body_transmission.slang"
#include "actuator_dynamics.slang"

float4 jointRotation(uint address)
{
    return normalize(float4(qpos[address], qpos[address + 1], qpos[address + 2], qpos[address + 3]));
}

float3 parentGear(float3 gear, float4 rotation)
{
    return rotateVector(gear, float4(rotation.x, -rotation.yzw));
}

float servoPeriod(ActuatorModel actuator)
{
    if (actuator.modes.x > ActivationFilterExact || actuator.modes.y != FixedGain ||
        actuator.modes.z != AffineBias || actuator.gain.x != -actuator.bias.y)
        return 0.0;

    if (actuator.modes.w == SiteTransmission && (actuator.indices.w & ReferenceSite) != 0 &&
        all(actuator.gearFirst.xyz == float3(0.0)))
        return 2.0 * PhysicsPi * length(actuator.gearLast.xyz);
    if (actuator.modes.w == JointTransmission)
    {
        if (joints[actuator.indices.x].indices.x == JointBall)
            return 2.0 * PhysicsPi * length(actuator.gearFirst.xyz);
    }
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

float nextActivation(ActuatorModel actuator, float previous, float derivative, float dt)
{
    if (actuator.modes.x == ActivationFilterExact)
    {
        const float tau = max(MinActuatorValue, actuator.filterTime);
        dt = tau * (1.0 - exp(-dt / tau));
    }
    const float value = previous + derivative * dt;
    const float2 range = actuator.controlActivationRange.zw;
    return (actuator.indices.w & LimitActivation) != 0 ? clamp(value, range.x, range.y) : value;
}

float2 jointActuatorMotion(ActuatorModel actuator, uint world, inout float3 firstMoment, out float3 lastMoment)
{
    const uint4 joint = joints[actuator.indices.x].indices;
    const uint q = world * model[0].state.x + joint.y;
    const uint v = world * model[0].state.y + joint.z;
    lastMoment = float3(0.0);
    if (joint.x == JointFree)
    {
        lastMoment = actuator.gearLast.xyz;
        if ((actuator.indices.w & ParentFrame) != 0)
            lastMoment = parentGear(lastMoment, jointRotation(q + 3));
        const float velocity = dot(firstMoment, float3(qvel[v], qvel[v + 1], qvel[v + 2])) +
                               dot(lastMoment, float3(qvel[v + 3], qvel[v + 4], qvel[v + 5]));
        return float2(0.0, velocity);
    }
    if (joint.x == JointBall)
    {
        const float4 rotation = jointRotation(q);
        if ((actuator.indices.w & ParentFrame) != 0)
            firstMoment = parentGear(firstMoment, rotation);
        return float2(dot(rotationVector(rotation), firstMoment),
                      dot(firstMoment, float3(qvel[v], qvel[v + 1], qvel[v + 2])));
    }
    return firstMoment.x * float2(qpos[q], qvel[v]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void evaluate_actuators(uint3 id : SV_DispatchThreadID)
{
    const uint4 sizes = actuatorDimensions[0];
    if (id.x >= sizes.x)
        return;
    const ActuatorModel actuator = actuators[id.x];
    const uint flags = actuator.indices.w;
    const uint transmission = actuator.modes.w;
    const uint index = id.y * sizes.x + id.x;
    float3 firstMoment = actuator.gearFirst.xyz;
    float3 lastMoment = float3(0.0);
    float2 motion;
    if (transmission == TendonTransmission)
    {
        const TendonState state = tendonState[id.y * actuatorDimensions[1].y + actuator.indices.x];
        motion = firstMoment.x * float2(state.length, state.velocity);
    }
    else if (transmission == SiteTransmission || transmission == BodyTransmission)
        motion = actuatorStep[index].values.xy;
    else
        motion = jointActuatorMotion(actuator, id.y, firstMoment, lastMoment);
    const float coordinate = motion.x;
    const float velocity = motion.y;

    float command = 0.0;
    if (actuator.indices.y != NoIndex)
    {
        command = control[id.y * sizes.y + actuator.indices.y];
        if ((flags & LimitControl) != 0)
            command = clamp(command, actuator.controlActivationRange.x, actuator.controlActivationRange.y);
    }

    const float period = servoPeriod(actuator);
    const uint activationBase = id.y * sizes.z;
    const uint controlBase = id.y * sizes.y;
    float forceCommand = command;
    if (actuator.modes.x != ActivationDcMotor && actuator.indices.z != NoIndex)
    {
        const uint state = activationBase + actuator.indices.z;
        const float current = activation[state].value;
        float derivative = 0.0;
        if (actuator.modes.x == ActivationIntegrator)
            derivative = command;
        else if (actuator.modes.x == ActivationFilter || actuator.modes.x == ActivationFilterExact)
            derivative = (nearestSetpoint(command, current, period) - current) /
                         max(MinActuatorValue, actuator.filterTime);
        else if (actuator.modes.x == ActivationMuscle)
            derivative = muscleActivationRate(command, current, actuatorDynamics[actuator.dynamicsIndex].parametersFirst.xyz);
        activation[state].derivative = derivative;
        forceCommand = (flags & EarlyActivation) != 0 ? nextActivation(actuator, current, derivative, timestep[id.y]) : current;
    }

    const float3 terms = float3(1.0, coordinate, velocity);
    float gain = 0.0;
    if (actuator.modes.y == FixedGain)
        gain = actuator.gain.x;
    else if (actuator.modes.y == AffineGain)
        gain = dot(actuator.gain.xyz, terms);
    else if (actuator.modes.y == MuscleGain)
        gain = muscleActiveForce(actuator, actuatorDynamics[actuator.dynamicsIndex], motion);
    float bias = actuator.modes.z == AffineBias ? dot(actuator.bias.xyz, terms) : 0.0;
    if (actuator.modes.z == MuscleBias)
        bias = musclePassiveForce(actuator, actuatorDynamics[actuator.dynamicsIndex], coordinate);

    float mechanicalForce = 0.0;
    if (actuator.modes.x == ActivationDcMotor || actuator.modes.y == DcMotorGain || actuator.modes.z == DcMotorBias)
    {
        const ActuatorDynamicsModel dynamics = actuatorDynamics[actuator.dynamicsIndex];
        const DcMotorResponse motor = evaluateDcMotor(actuator, dynamics, activationBase, controlBase,
                                                     command, motion, timestep[id.y]);
        if (actuator.modes.x == ActivationDcMotor)
            forceCommand = motor.controlActivation;
        if (actuator.modes.y == DcMotorGain)
        {
            gain = actuator.gain.y;
            if (dynamics.parametersFirst.x <= 0.0)
            {
                gain /= motor.resistance;
                forceCommand = motor.voltage;
            }
        }
        if (actuator.modes.z == DcMotorBias && dynamics.parametersFirst.x <= 0.0)
            bias -= gain * actuator.gain.y * velocity;
        mechanicalForce = motor.mechanicalForce;
    }

    float force = gain * nearestSetpoint(forceCommand, coordinate, period) + bias;
    // Tendon actuators apply the group limit before their own limit.
    if (transmission != TendonTransmission)
    {
        if ((flags & LimitForce) != 0)
            force = clamp(force, actuator.forceRange.x, actuator.forceRange.y);
        force += mechanicalForce;
    }

    const ActuatorStep result = {float4(firstMoment, 0.0), float4(lastMoment, 0.0),
                                 float4(coordinate, velocity, force, mechanicalForce)};
    actuatorStep[index] = result;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void limit_tendon_actuator_forces(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= actuatorDimensions[1].y)
        return;
    const TendonActuatorGroup tendon = tendonActuatorGroups[id.x];
    const uint begin = tendon.entries.x;
    const uint end = begin + tendon.entries.y;
    const uint base = id.y * actuatorDimensions[0].x;
    float scale = 1.0;
    if (tendon.entries.z != 0)
    {
        float sum = 0.0;
        for (uint i = begin; i < end; ++i)
            sum += actuatorStep[base + tendonActuatorList[i]].values.z;
        if (sum != 0.0)
            scale = clamp(sum, tendon.forceRange.x, tendon.forceRange.y) / sum;
    }
    for (uint i = begin; i < end; ++i)
    {
        const uint actuatorId = tendonActuatorList[i];
        const ActuatorModel actuator = actuators[actuatorId];
        const float2 range = actuator.forceRange;
        float force = scale * actuatorStep[base + actuatorId].values.z;
        if ((actuator.indices.w & LimitForce) != 0)
            force = clamp(force, range.x, range.y);
        actuatorStep[base + actuatorId].values.z = force + actuatorStep[base + actuatorId].values.w;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_actuator_force(uint3 id : SV_DispatchThreadID)
{
    const uint nv = model[0].state.y;
    if (id.x >= nv)
        return;
    const ActuatorDof dof = actuatorDofs[id.x];
    const uint base = id.y * actuatorDimensions[0].x;
    const uint index = id.y * nv + id.x;
    float force = 0.0;
    for (uint i = 0; i < dof.entries.y; ++i)
    {
        const uint2 edge = actuatorAdjacency[dof.entries.x + i];
        const ActuatorStep step = actuatorStep[base + edge.x];
        float moment;
        const uint transmission = actuators[edge.x].modes.w;
        if (transmission == TendonTransmission)
            moment = step.firstMoment.x * tendonJacobian[id.y * actuatorDimensions[1].z + edge.y];
        else if (transmission == SiteTransmission || transmission == BodyTransmission)
            moment = actuatorMoments[id.y * actuatorDimensions[0].w + edge.y];
        else
            moment = edge.y < 3 ? step.firstMoment[edge.y] : step.lastMoment[edge.y - 3];
        force += moment * step.values.z;
    }
    if ((actuationDofs[id.x].flags & GravityCompensatedByActuator) != 0)
        force += qfrcGravityCompensation[index];
    if (dof.entries.z != 0)
        force = clamp(force, dof.forceRange.x, dof.forceRange.y);
    qfrcActuator[index] = force;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void advance_activation(uint3 id : SV_DispatchThreadID)
{
    const uint4 sizes = actuatorDimensions[0];
    if (id.x >= sizes.x)
        return;
    const ActuatorModel actuator = actuators[id.x];
    if (actuator.indices.z == NoIndex)
        return;
    const uint index = id.y * sizes.z + actuator.indices.z;
    const float4 values = actuatorStep[id.y * sizes.x + id.x].values;
    const float dt = timestep[id.y];
    if (actuator.modes.x == ActivationDcMotor)
    {
        advanceDcMotor(actuator, actuatorDynamics[actuator.dynamicsIndex], index, values.y, dt);
        return;
    }
    const ActivationState state = activation[index];
    const float next = nextActivation(actuator, state.value, state.derivative, dt);
    activation[index].value = nearestSetpoint(next, values.x, servoPeriod(actuator));
}
