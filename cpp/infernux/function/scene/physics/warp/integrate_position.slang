// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from MuJoCo Warp _src/forward.py, _next_position.

void integrateJointPosition(uint jointType, uint pos, float3 linear, float3 angular, float dt,
                            RWStructuredBuffer<float> source, RWStructuredBuffer<float> destination)
{
    if (jointType == JointFree)
    {
        destination[pos] = source[pos] + dt * linear.x;
        destination[pos + 1] = source[pos + 1] + dt * linear.y;
        destination[pos + 2] = source[pos + 2] + dt * linear.z;
        pos += 3;
    }
    if (jointType == JointFree || jointType == JointBall)
    {
        const float4 orientation = float4(source[pos], source[pos + 1], source[pos + 2], source[pos + 3]);
        const float4 next = integrateQuat(orientation, angular, dt);
        destination[pos] = next.x;
        destination[pos + 1] = next.y;
        destination[pos + 2] = next.z;
        destination[pos + 3] = next.w;
    }
    else
        destination[pos] = source[pos] + dt * linear.x;
}
