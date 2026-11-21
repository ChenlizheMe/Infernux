// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

struct SceneRigidMotion
{
    float2 damping;       // Linear and angular rates, in inverse seconds.
    float2 velocityLimit; // COM metres/s and radians/s.
};

float3 clampSceneSpeed(float3 velocity, float limit)
{
    const float squared = dot(velocity, velocity);
    return squared > limit * limit ? velocity * (limit * rsqrt(squared)) : velocity;
}

float3 centerTransport(float3 spin, float3 offset)
{
    return cross(spin, cross(spin, offset));
}
