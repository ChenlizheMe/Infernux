// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

struct SceneRigidMotion
{
    float2 damping;       // Linear and angular rates, in inverse seconds.
    float2 velocityLimit; // COM metres/s and radians/s.
};

struct SceneBodyActivity
{
    uint marker; // Quiet seconds for dynamic bodies, next fixed index for edited supports.
    uint group;  // Sleeping island root, or NoIndex when awake.
    uint supports;
    uint jointEdited;
};

float3 clampSceneSpeed(float3 velocity, float limit)
{
    const float squared = dot(velocity, velocity);
    return squared > limit * limit ? velocity * (limit * rsqrt(squared)) : velocity;
}

float3 projectSceneAxes(float3 value, uint frozenAxes)
{
    return float3((frozenAxes & 1u) != 0u ? 0.0 : value.x,
                  (frozenAxes & 2u) != 0u ? 0.0 : value.y,
                  (frozenAxes & 4u) != 0u ? 0.0 : value.z);
}

float3 projectSceneSpin(float3 localSpin, float4 rotation, uint frozenAxes)
{
    if (frozenAxes == 0u)
        return localSpin;
    const float3 worldSpin = projectSceneAxes(rotateVector(localSpin, rotation), frozenAxes);
    return rotateVector(worldSpin, float4(rotation.x, -rotation.yzw));
}

float3 centerTransport(float3 spin, float3 offset)
{
    return cross(spin, cross(spin, offset));
}
