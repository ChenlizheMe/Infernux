// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#include "rigid_math.slang"

ConstantBuffer<PoseParams> pose_params;
StructuredBuffer<PoseNode> pose_nodes;
StructuredBuffer<PoseBinding> pose_bindings;
StructuredBuffer<BodyState> body_states;
// Separate entry-point ports share one buffer in the prepared packet. The node
// producer writes eight columns per node: world affine followed by its authored
// inverse. Earlier topological levels are read by later levels in this buffer.
RWStructuredBuffer<float4> pose_node_models;
StructuredBuffer<float4> pose_node_models_read;
// Four float4 columns per renderer, matching column-major InxModelMatrix.
RWStructuredBuffer<float4> pose_models;
// Two float4 records per renderer: final world-space mesh AABB min and max.
RWStructuredBuffer<float4> pose_bounds;

struct JphPoseAffine
{
    float3 column0;
    float3 column1;
    float3 column2;
    float3 column3;
};

float3 jph_pose_vector(JphPoseAffine matrix, float3 value)
{
    return matrix.column0 * value.x + matrix.column1 * value.y + matrix.column2 * value.z;
}

float3 jph_pose_point(JphPoseAffine matrix, float3 value)
{
    return jph_pose_vector(matrix, value) + matrix.column3;
}

JphPoseAffine jph_pose_affine(float4 column0, float4 column1, float4 column2, float4 column3)
{
    JphPoseAffine result;
    result.column0 = column0.xyz;
    result.column1 = column1.xyz;
    result.column2 = column2.xyz;
    result.column3 = column3.xyz;
    return result;
}

JphPoseAffine jph_pose_compose(JphPoseAffine left, JphPoseAffine right)
{
    JphPoseAffine result;
    result.column0 = jph_pose_vector(left, right.column0);
    result.column1 = jph_pose_vector(left, right.column1);
    result.column2 = jph_pose_vector(left, right.column2);
    result.column3 = jph_pose_point(left, right.column3);
    return result;
}

JphPoseAffine jph_pose_trs(float3 position, float4 rotation, float3 scale)
{
    JphPoseAffine result;
    result.column0 = jph_rotate(rotation, float3(scale.x, 0.0f, 0.0f));
    result.column1 = jph_rotate(rotation, float3(0.0f, scale.y, 0.0f));
    result.column2 = jph_rotate(rotation, float3(0.0f, 0.0f, scale.z));
    result.column3 = position;
    return result;
}

// Same authored inverse-TRS convention as TransformMatrix::InverseTRS and
// Transform::InverseTransformPoint. A collapsed axis contributes an all-zero
// row, including translation; a generic inverse cannot express this contract.
JphPoseAffine jph_pose_inverse_trs(float3 position, float4 rotation, float3 scale)
{
    float3 reciprocal = float3(abs(scale.x) < 1e-9f ? 0.0f : 1.0f / scale.x,
                               abs(scale.y) < 1e-9f ? 0.0f : 1.0f / scale.y,
                               abs(scale.z) < 1e-9f ? 0.0f : 1.0f / scale.z);
    float3 row0 = jph_rotate(rotation, float3(1.0f, 0.0f, 0.0f)) * reciprocal.x;
    float3 row1 = jph_rotate(rotation, float3(0.0f, 1.0f, 0.0f)) * reciprocal.y;
    float3 row2 = jph_rotate(rotation, float3(0.0f, 0.0f, 1.0f)) * reciprocal.z;
    JphPoseAffine result;
    result.column0 = float3(row0.x, row1.x, row2.x);
    result.column1 = float3(row0.y, row1.y, row2.y);
    result.column2 = float3(row0.z, row1.z, row2.z);
    result.column3 = -float3(dot(row0, position), dot(row1, position), dot(row2, position));
    return result;
}

// The host validates immutable author data once: finite affine columns/scales/
// COM, normalized xyzw parent rotations, in-range body/node indices, zero
// padding and an acyclic graph ordered into consecutive topological levels.
// BodyState is the frozen completed publication; no intermediate solve is read.
[shader("compute")]
[numthreads(64, 1, 1)]
void export_body_pose_nodes(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= pose_params.counts.w)
        return;
    uint index = pose_params.counts.z + tid.x;
    PoseNode node = pose_nodes[index];
    BodyState state = body_states[node.bodyParent.x];
    JphPoseAffine parent = jph_pose_affine(node.parentLocalToAncestor0, node.parentLocalToAncestor1,
                                           node.parentLocalToAncestor2, node.parentLocalToAncestor3);
    JphPoseAffine parent_inverse = jph_pose_affine(node.parentAncestorToLocal0, node.parentAncestorToLocal1,
                                                   node.parentAncestorToLocal2, node.parentAncestorToLocal3);
    float4 parent_rotation = node.parentRotation;
    if (node.bodyParent.y != PoseNoParent)
    {
        uint ancestor = 8u * node.bodyParent.y;
        JphPoseAffine world = jph_pose_affine(pose_node_models[ancestor], pose_node_models[ancestor + 1u],
                                               pose_node_models[ancestor + 2u], pose_node_models[ancestor + 3u]);
        JphPoseAffine world_inverse = jph_pose_affine(pose_node_models[ancestor + 4u], pose_node_models[ancestor + 5u],
                                                       pose_node_models[ancestor + 6u], pose_node_models[ancestor + 7u]);
        parent = jph_pose_compose(world, parent);
        parent_inverse = jph_pose_compose(parent_inverse, world_inverse);
        float4 ancestor_rotation = body_states[pose_nodes[node.bodyParent.y].bodyParent.x].rotation;
        parent_rotation = jph_quat_mul(ancestor_rotation, parent_rotation);
    }

    float3 desired_origin = state.positionCOM.xyz - jph_rotate(state.rotation, node.rootCOM.xyz);
    float3 local_position = jph_pose_point(parent_inverse, desired_origin);
    float4 local_rotation = jph_quat_mul(float4(-parent_rotation.xyz, parent_rotation.w), state.rotation);
    JphPoseAffine world = jph_pose_compose(parent, jph_pose_trs(local_position, local_rotation, node.localScale.xyz));
    JphPoseAffine world_inverse = jph_pose_compose(
        jph_pose_inverse_trs(local_position, local_rotation, node.localScale.xyz), parent_inverse);

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
    JphPoseAffine root = jph_pose_affine(pose_node_models_read[root_begin], pose_node_models_read[root_begin + 1u],
                                        pose_node_models_read[root_begin + 2u], pose_node_models_read[root_begin + 3u]);
    JphPoseAffine object = jph_pose_compose(root, jph_pose_affine(binding.objectLocal0, binding.objectLocal1,
                                                                binding.objectLocal2, binding.objectLocal3));

    // Selected-submesh drawing applies its pivot after the object transform.
    // Existing local bounds already include that pivot, so bounds use object.
    uint model_begin = 4u * index;
    pose_models[model_begin] = float4(object.column0, 0.0f);
    pose_models[model_begin + 1u] = float4(object.column1, 0.0f);
    pose_models[model_begin + 2u] = float4(object.column2, 0.0f);
    pose_models[model_begin + 3u] = float4(jph_pose_point(object, binding.drawPivot.xyz), 1.0f);

    float3 center = jph_pose_point(object, binding.boundsCenter.xyz);
    float3 extent = binding.boundsExtent.xyz;
    float3 world_extent = abs(object.column0) * extent.x + abs(object.column1) * extent.y + abs(object.column2) * extent.z;
    uint bounds_begin = 2u * index;
    pose_bounds[bounds_begin] = float4(center - world_extent, 0.0f);
    pose_bounds[bounds_begin + 1u] = float4(center + world_extent, 0.0f);
}
