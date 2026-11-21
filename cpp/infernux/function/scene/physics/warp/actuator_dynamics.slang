// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/util_misc.py and _src/forward.py actuator dynamics.

struct ActuatorDynamicsModel
{
    float4 parametersFirst; // dynamics parameters 0 through 3
    float4 parametersSecond; // dynamics parameters 4 through 7
    float4 parametersLast; // integral limit in word 0
    float4 gainSecond; // gain parameters 4 through 7
    float4 gainLast; // gain parameters 8 and 9, muscle length range
    float4 biasSecond; // bias parameters 4 through 7
    float4 biasLast; // bias parameters 8 and 9, reference acceleration
    uint4 layout; // control input mask, motor state mask
};

[[vk::binding(14, 0)]] StructuredBuffer<ActuatorDynamicsModel> actuatorDynamics;

float muscleActivationRate(float controlValue, float activationValue, float3 parameters)
{
    const float change = saturate(controlValue) - activationValue;
    const float scale = 0.5 + 1.5 * saturate(activationValue);
    const float rising = parameters.x * scale;
    const float falling = parameters.y / scale;
    float time;
    if (parameters.z < MinActuatorValue)
        time = change > 0.0 ? rising : falling;
    else
    {
        const float x = saturate(change / parameters.z + 0.5);
        const float sigmoid = x * x * x * (3.0 * x * (2.0 * x - 5.0) + 10.0);
        time = lerp(falling, rising, sigmoid);
    }
    return change / max(MinActuatorValue, time);
}

float muscleLengthGain(float value, float lower, float upper)
{
    if (value < lower || value > upper)
        return 0.0;
    const float lowerMid = 0.5 * (lower + 1.0);
    const float upperMid = 0.5 * (upper + 1.0);
    if (value <= lowerMid)
    {
        const float ratio = (value - lower) / max(MinActuatorValue, lowerMid - lower);
        return 0.5 * ratio * ratio;
    }
    if (value <= 1.0)
    {
        const float ratio = (1.0 - value) / max(MinActuatorValue, 1.0 - lowerMid);
        return 1.0 - 0.5 * ratio * ratio;
    }
    if (value <= upperMid)
    {
        const float ratio = (value - 1.0) / max(MinActuatorValue, upperMid - 1.0);
        return 1.0 - 0.5 * ratio * ratio;
    }
    const float ratio = (upper - value) / max(MinActuatorValue, upper - upperMid);
    return 0.5 * ratio * ratio;
}

float muscleOptimalLength(float2 lengthRange, float2 normalizedRange)
{
    return (lengthRange.y - lengthRange.x) / max(MinActuatorValue, normalizedRange.y - normalizedRange.x);
}

float muscleActiveForce(ActuatorModel actuator, ActuatorDynamicsModel dynamics, float2 motion)
{
    const float2 lengthRange = dynamics.gainLast.zw;
    const float optimal = muscleOptimalLength(lengthRange, actuator.gain.xy);
    const float normalizedLength = actuator.gain.x + (motion.x - lengthRange.x) / max(MinActuatorValue, optimal);
    const float normalizedVelocity = motion.y / max(MinActuatorValue, optimal * dynamics.gainSecond.z);
    const float lengthGain = muscleLengthGain(normalizedLength, dynamics.gainSecond.x, dynamics.gainSecond.y);
    const float maximum = dynamics.gainLast.x;
    const float shoulder = maximum - 1.0;
    float velocityGain;
    if (normalizedVelocity <= -1.0)
        velocityGain = 0.0;
    else if (normalizedVelocity <= 0.0)
    {
        const float value = normalizedVelocity + 1.0;
        velocityGain = value * value;
    }
    else if (normalizedVelocity <= shoulder)
    {
        const float value = shoulder - normalizedVelocity;
        velocityGain = maximum - value * value / max(MinActuatorValue, shoulder);
    }
    else
        velocityGain = maximum;
    const float force = actuator.gain.z < 0.0 ? actuator.gain.w / max(MinActuatorValue, dynamics.biasLast.z) : actuator.gain.z;
    return -force * lengthGain * velocityGain;
}

float musclePassiveForce(ActuatorModel actuator, ActuatorDynamicsModel dynamics, float coordinate)
{
    const float2 lengthRange = dynamics.gainLast.zw;
    const float optimal = muscleOptimalLength(lengthRange, actuator.bias.xy);
    const float normalizedLength = actuator.bias.x + (coordinate - lengthRange.x) / max(MinActuatorValue, optimal);
    if (normalizedLength <= 1.0)
        return 0.0;
    const float middle = 0.5 * (1.0 + dynamics.biasSecond.y);
    const float width = max(MinActuatorValue, middle - 1.0);
    const float force = actuator.bias.z < 0.0 ? actuator.bias.w / max(MinActuatorValue, dynamics.biasLast.z) : actuator.bias.z;
    const float value = (normalizedLength - 1.0) / width;
    const float passive = normalizedLength <= middle ? 0.5 * value * value : 0.5 + (normalizedLength - middle) / width;
    return -force * dynamics.biasSecond.w * passive;
}

static const uint MotorSlew = 1;
static const uint MotorIntegral = 2;
static const uint MotorTemperature = 4;
static const uint MotorBristle = 8;
static const uint MotorCurrent = 16;

uint motorStateIndex(uint base, uint mask, uint state)
{
    return base + countbits(mask & (state - 1u));
}

float motorBristleCoefficient(ActuatorModel actuator, ActuatorDynamicsModel dynamics, float speed)
{
    const float ratio = speed / max(MinActuatorValue, dynamics.biasSecond.y);
    const float scale = actuator.bias.w + (dynamics.biasSecond.x - actuator.bias.w) * exp(-ratio * ratio);
    return -dynamics.parametersSecond.y * abs(speed) / max(MinActuatorValue, scale);
}

float motorCurrentAtEnd(ActuatorDynamicsModel dynamics, ActivationState state, float dt)
{
    const float time = max(MinActuatorValue, dynamics.parametersFirst.x);
    return state.value + state.derivative * time * (1.0 - exp(-dt / time));
}

float motorVoltage(ActuatorModel actuator, ActuatorDynamicsModel dynamics, uint controlBase,
                   float command, float2 motion, float integral)
{
    if (actuator.indices.y == NoIndex)
        return 0.0;
    const uint inputs = dynamics.layout.x;
    if ((inputs & 7u) == 0)
        return command;

    float4 setpoint = float4(0.0);
    uint inputOffset = 0;
    for (uint slot = 0; slot < 4; ++slot)
    {
        if ((inputs & (1u << slot)) == 0)
            continue;
        setpoint[slot] = inputOffset == 0 ? command : control[controlBase + actuator.indices.y + inputOffset];
        ++inputOffset;
    }
    const float4 controller = dynamics.gainSecond; // kp, ki, kd, voltage limit
    const float torque = controller.x * (setpoint.x - motion.x) + controller.y * integral +
                         controller.z * (setpoint.y - motion.y) + setpoint.z;
    float voltage = (actuator.gain.x / actuator.gain.y) * torque + actuator.gain.y * motion.y;
    if (controller.w > 0.0)
        voltage = clamp(voltage, -controller.w, controller.w);
    return voltage + setpoint.w;
}

float advanceMotorSlot(ActuatorModel actuator, ActuatorDynamicsModel dynamics, uint slot,
                       ActivationState state, float speed, float dt)
{
    if (slot == MotorCurrent)
        return motorCurrentAtEnd(dynamics, state, dt);
    if (slot == MotorBristle)
    {
        const float coefficient = motorBristleCoefficient(actuator, dynamics, speed);
        const float decay = exp(coefficient * dt);
        const float integral = abs(coefficient) > MinActuatorValue ? (decay - 1.0) / coefficient : dt;
        return decay * state.value + integral * speed;
    }
    const float next = state.value + state.derivative * dt;
    const float limit = dynamics.parametersLast.x;
    return slot == MotorIntegral && limit > 0.0 ? clamp(next, -limit, limit) : next;
}

struct DcMotorResponse
{
    float controlActivation;
    float voltage;
    float resistance;
    float mechanicalForce;
};

DcMotorResponse evaluateDcMotor(ActuatorModel actuator, ActuatorDynamicsModel dynamics, uint activationBase,
                               uint controlBase, float command, float2 motion, float dt)
{
    const uint mask = dynamics.layout.y;
    const uint base = activationBase + actuator.indices.z;
    if ((mask & MotorSlew) != 0)
    {
        const float previous = activation[base].value;
        const float travel = dynamics.parametersSecond.w * dt;
        command = clamp(command, previous - travel, previous + travel);
        activation[base].derivative = (command - previous) / dt;
    }
    float integral = 0.0;
    if ((mask & MotorIntegral) != 0)
    {
        const uint index = motorStateIndex(base, mask, MotorIntegral);
        integral = activation[index].value;
        float rate = command - motion.x;
        const float limit = dynamics.parametersLast.x;
        if (limit > 0.0)
        {
            if (integral >= limit)
                rate = min(rate, 0.0);
            else if (integral <= -limit)
                rate = max(rate, 0.0);
        }
        activation[index].derivative = rate;
    }

    const float voltage = motorVoltage(actuator, dynamics, controlBase, command, motion, integral);
    const float motorConstant = actuator.gain.y;
    const float backEmf = motorConstant * motion.y;
    float resistance = actuator.gain.x;
    uint temperatureIndex = 0;
    float temperature = 0.0;
    if ((mask & MotorTemperature) != 0)
    {
        temperatureIndex = motorStateIndex(base, mask, MotorTemperature);
        temperature = activation[temperatureIndex].value;
        resistance *= 1.0 + actuator.gain.z * (temperature + dynamics.parametersSecond.x - actuator.gain.w);
    }
    float current = 0.0;
    if ((mask & MotorCurrent) != 0)
    {
        const uint index = motorStateIndex(base, mask, MotorCurrent);
        current = activation[index].value;
        float rate = (voltage / resistance - backEmf / resistance - current) /
                     max(MinActuatorValue, dynamics.parametersFirst.x);
        const float limit = dynamics.parametersFirst.y;
        if (limit > 0.0)
            rate = clamp(rate, -limit, limit);
        activation[index].derivative = rate;
    }
    if ((mask & MotorTemperature) != 0)
    {
        if ((mask & MotorCurrent) == 0)
            current = (voltage - backEmf) / resistance;
        const float cooling = temperature / dynamics.parametersFirst.z;
        activation[temperatureIndex].derivative = (resistance * current * current - cooling) / dynamics.parametersFirst.w;
    }

    const bool mechanicalBias = actuator.modes.z == DcMotorBias;
    float mechanical = mechanicalBias ? actuator.bias.x * sin(actuator.bias.y * motion.x + actuator.bias.z) : 0.0;
    if ((mask & MotorBristle) != 0)
    {
        const uint index = motorStateIndex(base, mask, MotorBristle);
        const float displacement = activation[index].value;
        const float rate = motorBristleCoefficient(actuator, dynamics, motion.y) * displacement + motion.y;
        activation[index].derivative = rate;
        if (mechanicalBias)
            mechanical -= dynamics.parametersSecond.y * displacement + dynamics.parametersSecond.z * rate;
    }

    float controlActivation = command;
    if (mask != 0)
    {
        const uint last = 1u << uint(firstbithigh(mask));
        const ActivationState state = activation[motorStateIndex(base, mask, last)];
        controlActivation = (actuator.indices.w & EarlyActivation) != 0 ?
            advanceMotorSlot(actuator, dynamics, last, state, motion.y, dt) : state.value;
    }
    return {controlActivation, voltage, max(MinActuatorValue, resistance), mechanical};
}

void advanceDcMotor(ActuatorModel actuator, ActuatorDynamicsModel dynamics, uint base, float speed, float dt)
{
    const uint mask = dynamics.layout.y;
    uint index = base;
    for (uint slot = MotorSlew; slot <= MotorCurrent; slot <<= 1u)
    {
        if ((mask & slot) == 0)
            continue;
        const ActivationState state = activation[index];
        activation[index].value = advanceMotorSlot(actuator, dynamics, slot, state, speed, dt);
        ++index;
    }
}
