// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#include "rigid_math.slang"

// Accepted shape and body rows used by the particle collision export.
StructuredBuffer<ShapeModel> shape_models;
StructuredBuffer<BodyState> body_states;

void query_leaf_pose(ShapeModel shape, BodyState state, out float3 center, out float4 rotation)
{
    center = state.positionCOM.xyz + jph_rotate(state.rotation, shape.localPositionCOM.xyz);
    rotation = jph_quat_mul(state.rotation, shape.localRotation);
}

bool query_valid_quaternion(float4 q)
{
    const float length = dot(q, q);
    return all(isfinite(q)) && isfinite(length) && length > 0.0f;
}
