// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021-2026 Jorrit Rouwe
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#include "render_pose_layout.slang"

ConstantBuffer<PoseParams> pose_params;
StructuredBuffer<PoseNode> pose_nodes;
StructuredBuffer<PoseBinding> pose_bindings;
StructuredBuffer<float4> body_states;
// Separate entry-point ports share one buffer in the prepared packet. The node
// producer writes eight columns per node: world affine followed by its authored
// inverse. Earlier topological levels are read by later levels in this buffer.
RWStructuredBuffer<float4> pose_node_models;
StructuredBuffer<float4> pose_node_models_read;
// Four float4 columns per renderer, matching column-major InxModelMatrix.
RWStructuredBuffer<float4> pose_models;
// Two float4 records per renderer: final world-space mesh AABB min and max.
RWStructuredBuffer<float4> pose_bounds;

float4 pose_mul_quat(float4 a, float4 b)
{
    return float4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
                  a.w * b.w - dot(a.xyz, b.xyz));
}

float3 pose_rotate(float4 q, float3 v)
{
    const float3 t = 2.0f * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

struct PoseAffine
{
    float3 column0;
    float3 column1;
    float3 column2;
    float3 column3;
};

float3 pose_vector(PoseAffine matrix, float3 value)
{
    return matrix.column0 * value.x + matrix.column1 * value.y + matrix.column2 * value.z;
}

float3 pose_point(PoseAffine matrix, float3 value)
{
    return pose_vector(matrix, value) + matrix.column3;
}

PoseAffine pose_affine(float4 column0, float4 column1, float4 column2, float4 column3)
{
    PoseAffine result;
    result.column0 = column0.xyz;
    result.column1 = column1.xyz;
    result.column2 = column2.xyz;
    result.column3 = column3.xyz;
    return result;
}

PoseAffine pose_compose(PoseAffine left, PoseAffine right)
{
    PoseAffine result;
    result.column0 = pose_vector(left, right.column0);
    result.column1 = pose_vector(left, right.column1);
    result.column2 = pose_vector(left, right.column2);
    result.column3 = pose_point(left, right.column3);
    return result;
}

PoseAffine pose_trs(float3 position, float4 rotation, float3 scale)
{
    PoseAffine result;
    result.column0 = pose_rotate(rotation, float3(scale.x, 0.0f, 0.0f));
    result.column1 = pose_rotate(rotation, float3(0.0f, scale.y, 0.0f));
    result.column2 = pose_rotate(rotation, float3(0.0f, 0.0f, scale.z));
    result.column3 = position;
    return result;
}

// Same authored inverse-TRS convention as TransformMatrix::InverseTRS and
// Transform::InverseTransformPoint. A collapsed axis contributes an all-zero
// row, including translation; a generic inverse cannot express this contract.
PoseAffine pose_inverse_trs(float3 position, float4 rotation, float3 scale)
{
    float3 reciprocal = float3(abs(scale.x) < 1e-9f ? 0.0f : 1.0f / scale.x,
                               abs(scale.y) < 1e-9f ? 0.0f : 1.0f / scale.y,
                               abs(scale.z) < 1e-9f ? 0.0f : 1.0f / scale.z);
    float3 row0 = pose_rotate(rotation, float3(1.0f, 0.0f, 0.0f)) * reciprocal.x;
    float3 row1 = pose_rotate(rotation, float3(0.0f, 1.0f, 0.0f)) * reciprocal.y;
    float3 row2 = pose_rotate(rotation, float3(0.0f, 0.0f, 1.0f)) * reciprocal.z;
    PoseAffine result;
    result.column0 = float3(row0.x, row1.x, row2.x);
    result.column1 = float3(row0.y, row1.y, row2.y);
    result.column2 = float3(row0.z, row1.z, row2.z);
    result.column3 = -float3(dot(row0, position), dot(row1, position), dot(row2, position));
    return result;
}

// The host validates immutable author data once: finite affine columns/scales/
// normalized xyzw parent rotations, in-range body/node indices, zero
// padding and an acyclic graph ordered into consecutive topological levels.
// Read one frozen completed publication.
uint pose_state_base(uint body)
{
    return 5u * (pose_params.stateRows.x + body);
}

float4 pose_state_rotation(uint base)
{
    return body_states[base + 1u].yzwx;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void export_body_pose_nodes(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= pose_params.counts.w)
        return;
    uint index = pose_params.counts.z + tid.x;
    PoseNode node = pose_nodes[index];
    uint body = pose_state_base(node.bodyParent.x);
    float4 rotation = pose_state_rotation(body);
    PoseAffine parent = pose_affine(node.parentLocalToAncestor0, node.parentLocalToAncestor1,
                                           node.parentLocalToAncestor2, node.parentLocalToAncestor3);
    PoseAffine parent_inverse = pose_affine(node.parentAncestorToLocal0, node.parentAncestorToLocal1,
                                                   node.parentAncestorToLocal2, node.parentAncestorToLocal3);
    float4 parent_rotation = node.parentRotation;
    if (node.bodyParent.y != PoseNoParent)
    {
        uint ancestor = 8u * node.bodyParent.y;
        PoseAffine world = pose_affine(pose_node_models[ancestor], pose_node_models[ancestor + 1u],
                                               pose_node_models[ancestor + 2u], pose_node_models[ancestor + 3u]);
        PoseAffine world_inverse = pose_affine(pose_node_models[ancestor + 4u], pose_node_models[ancestor + 5u],
                                                       pose_node_models[ancestor + 6u], pose_node_models[ancestor + 7u]);
        parent = pose_compose(world, parent);
        parent_inverse = pose_compose(parent_inverse, world_inverse);
        float4 ancestor_rotation = pose_state_rotation(pose_state_base(pose_nodes[node.bodyParent.y].bodyParent.x));
        parent_rotation = pose_mul_quat(ancestor_rotation, parent_rotation);
    }

    const float3 desired_origin = body_states[body].xyz;
    float3 local_position = pose_point(parent_inverse, desired_origin);
    float4 local_rotation = pose_mul_quat(float4(-parent_rotation.xyz, parent_rotation.w), rotation);
    PoseAffine world = pose_compose(parent, pose_trs(local_position, local_rotation, node.localScale.xyz));
    PoseAffine world_inverse = pose_compose(
        pose_inverse_trs(local_position, local_rotation, node.localScale.xyz), parent_inverse);

    // Keep the rebuilt origin: under collapsed parent axes it is the reachable
    // projection of desired_origin, exactly as CPU ApplyWorldPoseFromPhysics.
    uint begin = 8u * index;
    pose_node_models[begin] = float4(world.column0, 0.0f);
    pose_node_models[begin + 1u] = float4(world.column1, 0.0f);
    pose_node_models[begin + 2u] = float4(world.column2, 0.0f);
    pose_node_models[begin + 3u] = float4(world.column3, 1.0f);
    pose_node_models[begin + 4u] = float4(world_inverse.column0, 0.0f);
    pose_node_models[begin + 5u] = float4(world_inverse.column1, 0.0f);
    pose_node_models[begin + 6u] = float4(world_inverse.column2, 0.0f);
    pose_node_models[begin + 7u] = float4(world_inverse.column3, 1.0f);
}

// Author mesh bindings have finite affine columns (w = 0, 0, 0, 1),
// nonnegative local extents and in-range node indices. The object chain contains
// only ordinary child transforms and stops at the nearest exported body node.
// Capacities cover 8N node columns, 4M model columns and 2M bounds records;
// nodeCount <= UINT_MAX / 8 and bindingCount <= UINT_MAX / 4 for uint indexing.
[shader("compute")]
[numthreads(64, 1, 1)]
void export_render_poses(uint3 tid : SV_DispatchThreadID)
{
    uint index = tid.x;
    if (index >= pose_params.counts.y)
        return;

    PoseBinding binding = pose_bindings[index];
    uint root_begin = 8u * binding.nodeIndex.x;
    PoseAffine root = pose_affine(pose_node_models_read[root_begin], pose_node_models_read[root_begin + 1u],
                                        pose_node_models_read[root_begin + 2u], pose_node_models_read[root_begin + 3u]);
    PoseAffine object = pose_compose(root, pose_affine(binding.objectLocal0, binding.objectLocal1,
                                                                binding.objectLocal2, binding.objectLocal3));

    // Selected-submesh drawing applies its pivot after the object transform.
    // Existing local bounds already include that pivot, so bounds use object.
    uint model_begin = 4u * index;
    pose_models[model_begin] = float4(object.column0, 0.0f);
    pose_models[model_begin + 1u] = float4(object.column1, 0.0f);
    pose_models[model_begin + 2u] = float4(object.column2, 0.0f);
    pose_models[model_begin + 3u] = float4(pose_point(object, binding.drawPivot.xyz), 1.0f);

    float3 center = pose_point(object, binding.boundsCenter.xyz);
    float3 extent = binding.boundsExtent.xyz;
    float3 world_extent = abs(object.column0) * extent.x + abs(object.column1) * extent.y + abs(object.column2) * extent.z;
    uint bounds_begin = 2u * index;
    pose_bounds[bounds_begin] = float4(center - world_extent, 0.0f);
    pose_bounds[bounds_begin + 1u] = float4(center + world_extent, 0.0f);
}

struct MotionExportParams
{
    uint4 source; // First body row and request stride.
    uint4 target; // First request, count, destination scalar offset, field.
    uint4 output; // Values per row, scatter flag, reserved.
};

ConstantBuffer<MotionExportParams> motion_params;
StructuredBuffer<uint> motion_rows;
RWStructuredBuffer<float> motion_values;

float3 motion_allowed_axes(uint mask)
{
    return float3((mask & 1u) == 0u, (mask & 2u) == 0u, (mask & 4u) == 0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void export_body_motion(uint3 id : SV_DispatchThreadID)
{
    const uint index = id.x;
    if (index >= motion_params.target.y)
        return;
    const uint row = (motion_params.target.x + index) * motion_params.source.y;
    const uint body = 5u * (motion_params.source.x + motion_rows[row]);
    const uint outputRow = motion_params.output.y != 0u ? motion_rows[row + 1] : index;
    const uint start = motion_params.target.z + outputRow * motion_params.output.x;
    const uint field = motion_params.target.w;
    if (field == 1u)
    {
        const float4 rotation = normalize(pose_state_rotation(body));
        for (uint lane = 0; lane < 4u; ++lane)
            motion_values[start + lane] = rotation[lane];
        return;
    }
    if (field == 6u)
    {
        const float4 rotation = normalize(pose_state_rotation(body));
        const float4 frame = asfloat(uint4(motion_rows[row + 8], motion_rows[row + 9],
                                           motion_rows[row + 10], motion_rows[row + 11]));
        const float4 toWorld = normalize(pose_mul_quat(rotation, frame));
        const float4 toLocal = float4(-toWorld.xyz, toWorld.w);
        const float3 inertia = asfloat(uint3(motion_rows[row + 4], motion_rows[row + 5], motion_rows[row + 6]));
        const float3 allowed = motion_allowed_axes(motion_rows[row + 7]);
        for (uint column = 0; column < 3u; ++column)
        {
            float3 direction = float3(0.0);
            direction[column] = allowed[column];
            const float3 response = allowed * pose_rotate(toWorld, inertia * pose_rotate(toLocal, direction));
            for (uint component = 0; component < 3u; ++component)
                motion_values[start + 3u * component + column] = response[component];
        }
        return;
    }

    float3 value = float3(0.0);
    switch (field)
    {
    case 0u: value = body_states[body].xyz; break;
    case 2u: value = body_states[body + 2u].xyz; break;
    case 3u: value = body_states[body + 3u].xyz; break;
    case 4u: value = body_states[body + 4u].xyz; break;
    case 5u: value = asfloat(motion_rows[row + 3]) * motion_allowed_axes(motion_rows[row + 2]); break;
    }
    for (uint lane = 0; lane < 3u; ++lane)
        motion_values[start + lane] = value[lane];
}

RWStructuredBuffer<uint> box_values;

[shader("compute")]
[numthreads(64, 1, 1)]
void export_body_boxes(uint3 id : SV_DispatchThreadID)
{
    const uint index = id.x;
    if (index >= motion_params.target.y)
        return;
    const uint row = (motion_params.target.x + index) * motion_params.source.y;
    const uint outputRow = motion_params.output.y != 0u ? motion_rows[row + 1] : index;
    const uint start = motion_params.target.z + outputRow * motion_params.output.x;
    const uint field = motion_params.target.w;
    if (field == 7u || field >= 11u)
    {
        const uint scalar = field == 7u ? 2u : (field == 11u ? 7u : 15u);
        box_values[start] = motion_rows[row + scalar];
        return;
    }
    if (field == 9u)
    {
        for (uint axis = 0; axis < 3u; ++axis)
            box_values[start + axis] = motion_rows[row + 12u + axis];
        return;
    }
    const uint body = 5u * (motion_params.source.x + motion_rows[row]);
    const float4 rotation = normalize(pose_state_rotation(body));
    if (field == 8u)
    {
        const float3 local = asfloat(uint3(motion_rows[row + 4], motion_rows[row + 5], motion_rows[row + 6]));
        const float3 center = body_states[body + 2u].xyz + pose_rotate(rotation, local);
        for (uint axis = 0; axis < 3u; ++axis)
            box_values[start + axis] = asuint(center[axis]);
    }
    else
    {
        const float4 local = asfloat(uint4(motion_rows[row + 8], motion_rows[row + 9],
                                          motion_rows[row + 10], motion_rows[row + 11]));
        const float4 world = normalize(pose_mul_quat(rotation, local));
        for (uint axis = 0; axis < 4u; ++axis)
            box_values[start + axis] = asuint(world[axis]);
    }
}
