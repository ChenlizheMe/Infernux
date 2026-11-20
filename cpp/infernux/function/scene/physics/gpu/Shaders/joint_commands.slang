// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#include "rigid_joints.slang"

ConstantBuffer<JointParams> joint_params;
StructuredBuffer<uint> joint_remap;
StructuredBuffer<QueryRecordWord> old_joint_stream;
StructuredBuffer<BodyState> body_states;
StructuredBuffer<uint> joint_query_rows;
StructuredBuffer<JointPoseDrive> joint_pose_drives;
RWStructuredBuffer<JointReadback> joint_motion;

void jph_update_joint_state(JointFixedAuthor a, inout JointFixedState state, uint flags)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess) return;
    jph_joint_clear_impulses(state);
}

void jph_update_joint_state(JointHingeAuthor a, inout JointHingeState state, uint flags)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess) return;
    if ((flags & JointResetPoseDrive) != 0u)
    {
        state.flags.w &= ~JointPoseDriveActive;
        state.poseDriveRelative = float4(0, 0, 0, 1);
    }
    // Cache physical rotor speed before clearing its warm reaction. Previous
    // dt now lives in flags.z and must survive every author/mobility reset.
    float old_armature = state.rotor.w;
    float speed = old_armature > 0.0f ? state.rotor.x - state.rotor.y / old_armature : state.rotor.x;
    if ((flags & JointResetMobility) != 0u)
    {
        jph_joint_clear_impulses(state);
        state.rotor.xz = float2(speed);
        if (!isfinite(speed)) state.flags.y = JointStatusNumericalFailure;
        return;
    }
    if ((flags & JointResetRotor) != 0u)
    {
        BodyState s1 = jph_joint_world_state();
        if (a.bodies.x != InvalidQueryBodySlot) s1 = body_states[a.bodies.x];
        BodyState s2 = body_states[a.bodies.y];
        speed = dot(jph_joint_axis(s1, state), s2.angularVelocity.xyz - s1.angularVelocity.xyz);
    }
    jph_joint_clear_impulses(state);
    state.rotor = float4(speed, 0.0f, speed, a.passive.z);
    if (!isfinite(speed)) state.flags.y = JointStatusNumericalFailure;
}

void jph_update_joint_state(JointSliderAuthor a, inout JointSliderState state, uint flags)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess) return;
    jph_joint_clear_impulses(state);
}

void jph_update_joint_state(JointDistanceAuthor a, inout JointDistanceState state, uint flags)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess) return;
    jph_joint_clear_impulses(state);
}

void jph_update_joint_state(JointSwingAuthor a, inout JointSwingState state, uint flags)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess) return;
    if ((flags & JointResetPoseDrive) != 0u)
    {
        state.flags.w &= ~JointPoseDriveActive;
        state.poseDriveRelative = float4(0, 0, 0, 1);
    }
    if ((flags & JointResetMobility) != 0u)
    {
        jph_joint_clear_impulses(state);
        return;
    }
    if ((flags & JointResetSwingLimits) != 0u) state.angularLimitImpulse = float4(0.0f);
    if ((flags & JointResetSwingMotor) != 0u) state.angularMotorImpulse.yz = float2(0.0f);
    if ((flags & JointResetTwistMotor) != 0u) state.angularMotorImpulse.x = 0.0f;
}

void jph_update_joint_state(JointSixAuthor a, inout JointSixState state, uint flags)
{
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess) return;
    if ((flags & JointResetPoseDrive) != 0u)
    {
        state.flags.w &= ~JointPoseDriveActive;
        state.poseDriveRelative = float4(0, 0, 0, 1);
    }
    if ((flags & (JointResetMobility | JointResetSixClassification)) != 0u)
    { jph_joint_clear_impulses(state); return; }
    if ((flags & JointResetSwingLimits) != 0u)
        state.angularLimitImpulse = float4(0.0f);
    for (uint i=0u;i<3u;++i)
    {
        if ((flags & (1u << (JointResetSixLimitShift+i))) != 0u) state.sixLimitImpulse[i]=0.0f;
        if ((flags & (1u << (JointResetSixLinearMotorShift+i))) != 0u) state.sixMotorImpulse[i]=0.0f;
        if ((flags & (1u << (JointResetSixAngularMotorShift+i))) != 0u) state.angularMotorImpulse[i]=0.0f;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void update_joint_state(uint3 tid : SV_DispatchThreadID)
{
    uint request = tid.x;
    if (request >= joint_params.counts.z) return;
    uint index = joint_query_rows[2u * request];
    uint flags = joint_query_rows[2u * request + 1u];
    JointDirectory directory = jph_joint_load_directory(index);
    switch (directory.bodies.z)
    {
    case JointFixed:
    {
        JointFixedAuthor author; jph_joint_load_author(directory, author);
        JointFixedState runtime; jph_joint_load_state(directory, runtime);
        jph_update_joint_state(author, runtime, flags);
        jph_joint_store_state(directory, runtime);
        break;
    }
    case JointHinge:
    {
        JointHingeAuthor author; jph_joint_load_author(directory, author);
        JointHingeState runtime; jph_joint_load_state(directory, runtime);
        jph_update_joint_state(author, runtime, flags);
        jph_joint_store_state(directory, runtime);
        break;
    }
    case JointSlider:
    {
        JointSliderAuthor author; jph_joint_load_author(directory, author);
        JointSliderState runtime; jph_joint_load_state(directory, runtime);
        jph_update_joint_state(author, runtime, flags);
        jph_joint_store_state(directory, runtime);
        break;
    }
    case JointDistance:
    {
        JointDistanceAuthor author; jph_joint_load_author(directory, author);
        JointDistanceState runtime; jph_joint_load_state(directory, runtime);
        jph_update_joint_state(author, runtime, flags);
        jph_joint_store_state(directory, runtime);
        break;
    }
    case JointSwingTwist:
    {
        JointSwingAuthor author; jph_joint_load_author(directory, author);
        JointSwingState runtime; jph_joint_load_state(directory, runtime);
        jph_update_joint_state(author, runtime, flags);
        jph_joint_store_state(directory, runtime);
        break;
    }
    case JointSixDOF:
    {
        JointSixAuthor author; jph_joint_load_author(directory, author);
        JointSixState runtime; jph_joint_load_state(directory, runtime);
        jph_update_joint_state(author, runtime, flags);
        jph_joint_store_state(directory, runtime);
        break;
    }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void initialize_joints(uint3 tid : SV_DispatchThreadID)
{
    uint index = tid.x;
    if (index >= joint_params.counts.x) return;
    JointDirectory directory = jph_joint_load_directory(index);
    uint old_index = joint_remap[index];
    if (old_index != InvalidQueryBodySlot && old_index < joint_params.counts.y)
    {
        uint old_directory = 1u + JointDirectoryWords * old_index;
        uint4 old_bodies = old_joint_stream[old_directory].value;
        uint4 old_identity = old_joint_stream[old_directory + 1u].value;
        uint old_state = old_joint_stream[old_directory + 2u].value.y;
        if (old_bodies.z == directory.bodies.z && all(old_identity.xy == directory.identity.xy))
        {
            uint words = jph_joint_state_words(directory.bodies.z);
            // Exact kind payload only. Frames/history and the new dt/armature
            // lanes remain GPU-owned; no unused wide suffix exists anymore.
            for (uint word = 0u; word < words; ++word)
                joint_stream[directory.layout.y + word] = old_joint_stream[old_state + word];
            return;
        }
    }
    BodyState s1 = jph_joint_world_state();
    if (directory.bodies.x != InvalidQueryBodySlot) s1 = body_states[directory.bodies.x];
    BodyState s2 = body_states[directory.bodies.y];
    switch (directory.bodies.z)
    {
    case JointFixed:
    {
        JointFixedAuthor author; jph_joint_load_author(directory, author);
        JointFixedState runtime = jph_joint_initialize(author, s1, s2);
        if (old_index != InvalidQueryBodySlot) runtime.flags.y = JointStatusInvalid;
        jph_joint_store_state(directory, runtime);
        break;
    }
    case JointHinge:
    {
        JointHingeAuthor author; jph_joint_load_author(directory, author);
        JointHingeState runtime = jph_joint_initialize(author, s1, s2);
        if (old_index != InvalidQueryBodySlot) runtime.flags.y = JointStatusInvalid;
        jph_joint_store_state(directory, runtime);
        break;
    }
    case JointSlider:
    {
        JointSliderAuthor author; jph_joint_load_author(directory, author);
        JointSliderState runtime = jph_joint_initialize(author, s1, s2);
        if (old_index != InvalidQueryBodySlot) runtime.flags.y = JointStatusInvalid;
        jph_joint_store_state(directory, runtime);
        break;
    }
    case JointDistance:
    {
        JointDistanceAuthor author; jph_joint_load_author(directory, author);
        JointDistanceState runtime = jph_joint_initialize(author, s1, s2);
        if (old_index != InvalidQueryBodySlot) runtime.flags.y = JointStatusInvalid;
        jph_joint_store_state(directory, runtime);
        break;
    }
    case JointSwingTwist:
    {
        JointSwingAuthor author; jph_joint_load_author(directory, author);
        JointSwingState runtime = jph_joint_initialize(author, s1, s2);
        if (old_index != InvalidQueryBodySlot) runtime.flags.y = JointStatusInvalid;
        jph_joint_store_state(directory, runtime);
        break;
    }
    case JointSixDOF:
    {
        JointSixAuthor author; jph_joint_load_author(directory, author);
        JointSixState runtime = jph_joint_initialize(author, s1, s2);
        if (old_index != InvalidQueryBodySlot) runtime.flags.y = JointStatusInvalid;
        jph_joint_store_state(directory, runtime);
        break;
    }
    }
}

void jph_read_joint_motion(JointHingeAuthor a, JointHingeState state, inout JointReadback result)
{
    if (state.flags.x == 0u) return;
    result.status.x = state.flags.y;
    if (state.flags.y != JointStatusSuccess) return;
    BodyState s1 = jph_joint_world_state();
    if (a.bodies.x != InvalidQueryBodySlot) s1 = body_states[a.bodies.x - joint_params.counts.w];
    BodyState s2 = body_states[a.bodies.y - joint_params.counts.w];
    float coordinate, velocity;
    coordinate = jph_joint_angle(s1, s2, state);
    float3 owner_axis = jph_rotate(jph_quat_mul(s2.rotation, state.frame2), float3(0.0f, 0.0f, 1.0f));
    velocity = dot(owner_axis, s2.angularVelocity.xyz - s1.angularVelocity.xyz);
    if (isfinite(coordinate) && isfinite(velocity)) result.values.xy = float2(coordinate, velocity);
    else result.status.x = JointStatusNumericalFailure;
}

void jph_read_joint_motion(JointSliderAuthor a, JointSliderState state, inout JointReadback result)
{
    if (state.flags.x == 0u) return;
    result.status.x = state.flags.y;
    if (state.flags.y != JointStatusSuccess) return;
    BodyState s1 = jph_joint_world_state();
    if (a.bodies.x != InvalidQueryBodySlot) s1 = body_states[a.bodies.x - joint_params.counts.w];
    BodyState s2 = body_states[a.bodies.y - joint_params.counts.w];
    float coordinate, velocity;
    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);
    float3 u = s2.positionCOM.xyz - s1.positionCOM.xyz + r2 - r1;
    float3 axis, n0, n1;
    jph_slider_basis(s1, state, axis, n0, n1);
    coordinate = dot(axis, u);
    velocity = dot(axis, s2.linearVelocity.xyz - s1.linearVelocity.xyz)
        + dot(cross(r2, axis), s2.angularVelocity.xyz)
        - dot(cross(r1 + u, axis), s1.angularVelocity.xyz);
    if (isfinite(coordinate) && isfinite(velocity)) result.values.xy = float2(coordinate, velocity);
    else result.status.x = JointStatusNumericalFailure;
}

void jph_read_joint_motion(JointDistanceAuthor a, JointDistanceState state, inout JointReadback result)
{
    if (state.flags.x == 0u) return;
    result.status.x = state.flags.y;
    if (state.flags.y != JointStatusSuccess) return;
    BodyState s1 = jph_joint_world_state();
    if (a.bodies.x != InvalidQueryBodySlot) s1 = body_states[a.bodies.x - joint_params.counts.w];
    BodyState s2 = body_states[a.bodies.y - joint_params.counts.w];
    float coordinate, velocity;
    float3 r1, r2;
    jph_joint_arms(a, state, s1, s2, r1, r2);
    float3 u = s2.positionCOM.xyz - s1.positionCOM.xyz + r2 - r1;
    coordinate = length(u);
    float3 normal = coordinate > 0.0f ? u / coordinate : state.distanceReference.xyz;
    velocity = dot(normal, s2.linearVelocity.xyz - s1.linearVelocity.xyz)
        + dot(cross(r2, normal), s2.angularVelocity.xyz)
        - dot(cross(r1 + u, normal), s1.angularVelocity.xyz);
    if (isfinite(coordinate) && isfinite(velocity)) result.values.xy = float2(coordinate, velocity);
    else result.status.x = JointStatusNumericalFailure;
}

void jph_read_joint_motion(JointSwingAuthor a, JointSwingState state, inout JointReadback result)
{
    if (state.flags.x == 0u) return;
    result.status.x = state.flags.y;
    if (state.flags.y != JointStatusSuccess) return;
    BodyState s1 = jph_joint_world_state();
    if (a.bodies.x != InvalidQueryBodySlot) s1 = body_states[a.bodies.x - joint_params.counts.w];
    BodyState s2 = body_states[a.bodies.y - joint_params.counts.w];
    result.orientation = jph_swing_orientation(s1, s2, state);
    if (!all(isfinite(result.orientation))) result.status.x = JointStatusNumericalFailure;
}

void jph_read_joint_motion(JointSixAuthor a, JointSixState state, inout JointReadback result)
{
    if (state.flags.x == 0u) return;
    result.status.x = state.flags.y;
    if (state.flags.y != JointStatusSuccess) return;
    BodyState s1 = jph_joint_world_state();
    if (a.bodies.x != InvalidQueryBodySlot) s1 = body_states[a.bodies.x - joint_params.counts.w];
    BodyState s2 = body_states[a.bodies.y - joint_params.counts.w];
    float3 r1,r2; jph_joint_arms(a,state,s1,s2,r1,r2);
    float4 world1 = jph_quat_mul(s1.rotation,state.frame1);
    result.values = float4(jph_inverse_rotate(world1,s2.positionCOM.xyz-s1.positionCOM.xyz+r2-r1),0.0f);
    result.orientation = jph_six_orientation(s1,s2,state);
    if (!all(isfinite(result.values)) || !all(isfinite(result.orientation))) result.status.x=JointStatusNumericalFailure;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void read_joint_motion(uint3 tid : SV_DispatchThreadID)
{
    uint row = tid.x;
    if (row >= joint_params.counts.z) return;
    JointReadback result;
    result.values = float4(0.0f);
    result.orientation = float4(0.0f);
    result.status = uint4(JointStatusInvalid, 0u, 0u, 0u);
    uint index = joint_query_rows[row];
    if (index < joint_stream[0].value.x)
    {
        JointDirectory directory = jph_joint_load_directory(index);
        if (directory.bodies.z == joint_params.counts.y)
        {
            switch (directory.bodies.z)
            {
            case JointHinge:
            {
                JointHingeAuthor author; jph_joint_load_author(directory, author);
                JointHingeState runtime; jph_joint_load_state(directory, runtime);
                jph_read_joint_motion(author, runtime, result);
                break;
            }
            case JointSlider:
            {
                JointSliderAuthor author; jph_joint_load_author(directory, author);
                JointSliderState runtime; jph_joint_load_state(directory, runtime);
                jph_read_joint_motion(author, runtime, result);
                break;
            }
            case JointDistance:
            {
                JointDistanceAuthor author; jph_joint_load_author(directory, author);
                JointDistanceState runtime; jph_joint_load_state(directory, runtime);
                jph_read_joint_motion(author, runtime, result);
                break;
            }
            case JointSwingTwist:
            {
                JointSwingAuthor author; jph_joint_load_author(directory, author);
                JointSwingState runtime; jph_joint_load_state(directory, runtime);
                jph_read_joint_motion(author, runtime, result);
                break;
            }
            case JointSixDOF:
            {
                JointSixAuthor author; jph_joint_load_author(directory, author);
                JointSixState runtime; jph_joint_load_state(directory, runtime);
                jph_read_joint_motion(author, runtime, result);
                break;
            }
            }
        }
    }
    joint_motion[row] = result;
}

groupshared uint joint_status_values[64];
groupshared uint joint_status_failures[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void read_joint_status(uint3 tid : SV_DispatchThreadID, uint3 lane : SV_GroupThreadID, uint3 group : SV_GroupID)
{
    uint status = JointStatusSuccess;
    uint first_failure = InvalidQueryBodySlot;
    // Status work: source first/count, destination first, reduction level.
    // Every lane reaches the barriers, including the last group's padding.
    if (tid.x < joint_params.counts.y)
    {
        uint index = joint_params.counts.x + tid.x;
        if (joint_params.counts.w == 0u)
        {
            uint runtime_word = joint_stream[1u + JointDirectoryWords * index + 2u].value.y;
            status = joint_stream[runtime_word].value.y;
            if (status != JointStatusSuccess)
                first_failure = index;
        }
        else
        {
            uint4 previous = joint_motion[index].status;
            status = previous.x;
            first_failure = previous.y;
        }
    }
    joint_status_values[lane.x] = status;
    joint_status_failures[lane.x] = first_failure;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 32u; stride != 0u; stride >>= 1u)
    {
        if (lane.x < stride)
        {
            joint_status_values[lane.x] = max(joint_status_values[lane.x], joint_status_values[lane.x + stride]);
            joint_status_failures[lane.x] = min(joint_status_failures[lane.x], joint_status_failures[lane.x + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane.x == 0u)
    {
        JointReadback result;
        result.values = float4(0.0f);
    result.orientation = float4(0.0f);
        result.status = uint4(joint_status_values[0], joint_status_failures[0], 0u, 0u);
        joint_motion[joint_params.counts.z + group.x] = result;
    }
}


void jph_pose_drive(inout JointHingeState state, JointPoseDrive request, bool identityMatches)
{
    if (!identityMatches) { state.flags.y = JointStatusInvalid; return; }
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess) return;
    if (request.meta.w == JointPoseDriveClear)
    {
        state.flags.w &= ~JointPoseDriveActive;
        state.poseDriveRelative = float4(0, 0, 0, 1);
        return;
    }
    float4 target = normalize(request.relative);
    if (request.meta.w != JointPoseDriveSet || !all(isfinite(target)))
    { state.flags.y = JointStatusNumericalFailure; return; }
    state.poseDriveRelative = target;
    state.flags.w |= JointPoseDriveActive;
}

void jph_pose_drive(inout JointSwingState state, JointPoseDrive request, bool identityMatches)
{
    if (!identityMatches) { state.flags.y = JointStatusInvalid; return; }
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess) return;
    if (request.meta.w == JointPoseDriveClear)
    {
        state.flags.w &= ~JointPoseDriveActive;
        state.poseDriveRelative = float4(0, 0, 0, 1);
        return;
    }
    float4 target = normalize(request.relative);
    if (request.meta.w != JointPoseDriveSet || !all(isfinite(target)))
    { state.flags.y = JointStatusNumericalFailure; return; }
    state.poseDriveRelative = target;
    state.flags.w |= JointPoseDriveActive;
}

void jph_pose_drive(inout JointSixState state, JointPoseDrive request, bool identityMatches)
{
    if (!identityMatches) { state.flags.y = JointStatusInvalid; return; }
    if (state.flags.x == 0u || state.flags.y != JointStatusSuccess) return;
    if (request.meta.w == JointPoseDriveClear)
    {
        state.flags.w &= ~JointPoseDriveActive;
        state.poseDriveRelative = float4(0, 0, 0, 1);
        return;
    }
    float4 target = normalize(request.relative);
    if (request.meta.w != JointPoseDriveSet || !all(isfinite(target)))
    { state.flags.y = JointStatusNumericalFailure; return; }
    state.poseDriveRelative = target;
    state.flags.w |= JointPoseDriveActive;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void drive_joint_pose(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= joint_params.counts.z) return;
    JointPoseDrive request = joint_pose_drives[tid.x];
    if (request.meta.x >= joint_params.counts.x) return; // Host validates the complete captured range.
    JointDirectory directory = jph_joint_load_directory(request.meta.x);
    bool identityMatches = all(directory.identity.xy == request.meta.yz);
    switch (directory.bodies.z)
    {
    case JointHinge:
    {
        JointHingeState state; jph_joint_load_state(directory, state);
        jph_pose_drive(state, request, identityMatches);
        jph_joint_store_state(directory, state);
        break;
    }
    case JointSwingTwist:
    {
        JointSwingState state; jph_joint_load_state(directory, state);
        jph_pose_drive(state, request, identityMatches);
        jph_joint_store_state(directory, state);
        break;
    }
    case JointSixDOF:
    {
        JointSixState state; jph_joint_load_state(directory, state);
        jph_pose_drive(state, request, identityMatches);
        jph_joint_store_state(directory, state);
        break;
    }
    }
}
