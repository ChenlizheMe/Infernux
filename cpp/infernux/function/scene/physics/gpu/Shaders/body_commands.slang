// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#include "rigid_math.slang"
#include "rigid_sleep_math.slang"

ConstantBuffer<BodyCommandParams> command_params;
StructuredBuffer<BodyCommand> body_commands;
StructuredBuffer<BodyCommandSpan> command_spans;
StructuredBuffer<BodyModel> body_models;
RWStructuredBuffer<BodyState> body_states;
RWStructuredBuffer<BodyForces> body_forces;
RWStructuredBuffer<KinematicTarget> kinematic_targets;
RWStructuredBuffer<BodySleepRecord> body_sleep;
StructuredBuffer<uint2> participant_rows;
RWStructuredBuffer<float> participant_motion;
StructuredBuffer<float> participant_impulses;

void command_reset_sleep(BodyState state,float3 extent,inout BodySleepRuntime sleep)
{
    float3 p0,p1,p2;sleep_test_points(state,extent,p0,p1,p2);
    sleep_reset(sleep,p0,p1,p2);
}
void command_wake(BodyModel model,BodyState state,float3 extent,inout BodySleepRuntime sleep,bool resetActive)
{
    if(jph_nonstatic(model) && (resetActive || sleep.state.x==0u)) {
        command_reset_sleep(state,extent,sleep);
        sleep.state.x=1u;
    }
}

// Scene resolves these sparse addresses once against this Fixed's admitted
// author prefix. CPU rows in the project arrays are deliberately untouched.
[shader("compute")]
[numthreads(64, 1, 1)]
void prepare_participant_bodies(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= command_params.spans.y) return;
    uint2 row = participant_rows[command_params.spans.x + tid.x];
    BodyModel model = body_models[row.x];
    BodyState state = body_states[row.x];
    BodySleepRuntime sleep = body_sleep[row.x].runtime;
    command_wake(model, state, body_sleep[row.x].author.rootHalfExtent.xyz, sleep, false);
    body_sleep[row.x].runtime = sleep;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gather_participant_motion(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= command_params.spans.y) return;
    uint2 row = participant_rows[command_params.spans.x + tid.x];
    BodyState state = body_states[row.x];
    uint first = row.y * 16u;
    for (uint lane = 0u; lane < 4u; ++lane) {
        participant_motion[first + lane] = lane == 3u ? 0.0f : state.positionCOM[lane];
        participant_motion[first + 4u + lane] = state.rotation[lane];
        participant_motion[first + 8u + lane] = lane == 3u ? 0.0f : state.linearVelocity[lane];
        participant_motion[first + 12u + lane] = lane == 3u ? 0.0f : state.angularVelocity[lane];
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void apply_participant_impulses(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= command_params.spans.y) return;
    uint2 row = participant_rows[command_params.spans.x + tid.x];
    uint first = row.y * 6u;
    float3 linear = float3(participant_impulses[first], participant_impulses[first + 1u], participant_impulses[first + 2u]);
    float3 angular = float3(participant_impulses[first + 3u], participant_impulses[first + 4u], participant_impulses[first + 5u]);
    // A zero increment does not restart an already active body's sleep history.
    if (all(linear == float3(0.0f)) && all(angular == float3(0.0f))) return;
    BodyModel model = body_models[row.x];
    BodyState state = body_states[row.x];
    state.linearVelocity.xyz += jph_inverse_linear_mass(model, linear);
    state.angularVelocity.xyz += jph_inverse_inertia(model, state, angular);
    jph_project_motion(model, state);
    // As with native contact rows, let the connected constraints distribute
    // momentum before the ordinary drift clamps the physical endpoint speeds.
    BodySleepRuntime sleep = body_sleep[row.x].runtime;
    command_wake(model, state, body_sleep[row.x].author.rootHalfExtent.xyz, sleep, false);
    body_states[row.x] = state;
    body_sleep[row.x].runtime = sleep;
}

float3 command_world_inertia(BodyModel model, BodyState state, float3 value)
{
    float4 frame = jph_quat_mul(state.rotation, model.inertiaRotation);
    return jph_rotate(frame, jph_inverse_rotate(frame, value) / model.invInertiaDiagonal.xyz);
}

void command_cancel_move(inout BodyState state, inout KinematicTarget target)
{
    if (jph_has_kinematic_move(target))
    {
        state.linearVelocity = float4(0.0f);
        state.angularVelocity = float4(0.0f);
    }
    target.state = uint4(0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void apply_body_commands(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= command_params.spans.y)
        return;
    BodyCommandSpan span = command_spans[command_params.spans.x + tid.x];
    uint body = span.range.x;
    BodyModel model = body_models[body];
    BodyState state = body_states[body];
    BodyForces forces = body_forces[body];
    KinematicTarget target = kinematic_targets[body];
    BodySleepRuntime sleep = body_sleep[body].runtime;
    float3 sleep_extent = body_sleep[body].author.rootHalfExtent.xyz;
    // The validated immutable plan bounds every span to 64. Each dispatch has
    // at most one span for any body, so no atomics or global command scan occur.
    for (uint offset = 0; offset < span.range.z; ++offset)
    {
        BodyCommand command = body_commands[span.range.y + offset];
        uint kind = command.meta.y;
        uint mode = command.meta.z;
        if (kind == CommandSleep)
        {
            command_cancel_move(state,target);
            sleep.state.zw=uint2(0u);
            if (jph_nonstatic(model)) {
                state.linearVelocity = state.angularVelocity = float4(0);
                sleep.state.x=0u;
            }
        }
        else if (kind == CommandWake)
            command_wake(model,state,sleep_extent,sleep,true);
        else if (kind <= CommandSetPose)
        {
            BodyState previous=state;
            if ((mode & CommandPreserveKinematicMove) == 0u)
            {
                command_cancel_move(state, target);
                sleep.state.w=0u;
            }
            if (kind == CommandSetPosition)
                state.positionCOM.xyz = command.value.xyz + jph_rotate(state.rotation, command.rootCOM.xyz);
            else if (kind == CommandSetRotation)
            {
                float3 origin = state.positionCOM.xyz - jph_rotate(state.rotation, command.rootCOM.xyz);
                state.rotation = command.auxiliary;
                state.positionCOM.xyz = origin + jph_rotate(state.rotation, command.rootCOM.xyz);
            }
            else
            {
                state.rotation = command.auxiliary;
                state.positionCOM.xyz = command.value.xyz + jph_rotate(state.rotation, command.rootCOM.xyz);
            }
            // A pose command changes the principal frame before later FIFO Jv.
            jph_project_motion(model, state);
            if(any(previous.positionCOM.xyz!=state.positionCOM.xyz) ||
               !(all(previous.rotation==state.rotation) || all(previous.rotation==-state.rotation)))
            {
                ++sleep.timing.y;
                command_wake(model,state,sleep_extent,sleep,true);
            }
        }
        else if (kind == CommandResetMotion)
        {
            state.linearVelocity = float4(0.0f);
            state.angularVelocity = float4(0.0f);
            forces.force = float4(0.0f);
            forces.torque = float4(0.0f);
            target.state = uint4(0u);
            command_reset_sleep(state,sleep_extent,sleep);
            ++sleep.timing.y;
            // This is the existing explicit full motion-state/episode reset,
            // not Sleep or a retained-preserve boundary.
            sleep.state.x=jph_nonstatic(model)?1u:0u;
        }
        else if (kind >= CommandMovePosition && kind <= CommandMovePose)
        {
            uint motion = model.meta.z & ~MotionNotResident;
            bool temporary_static = motion == MotionStatic && (mode & CommandTemporaryStatic) != 0u;
            if (motion == MotionKinematic || temporary_static)
            {
                // Only the first pending partial move samples the actual
                // device pose. Later partial moves compose that queued target.
                if ((target.state.x & TargetPending) == 0u)
                {
                    float3 origin = state.positionCOM.xyz - jph_rotate(state.rotation, command.rootCOM.xyz);
                    target.positionSpeed = float4(origin, 0.0f);
                    target.rotation = state.rotation;
                }
                if (kind != CommandMoveRotation)
                    target.positionSpeed.xyz = command.value.xyz;
                if (kind != CommandMovePosition)
                    target.rotation = command.auxiliary;
                target.positionSpeed.w = command.rootCOM.w;
                target.rootCOMDt = float4(command.rootCOM.xyz, command.value.w);
                target.state = uint4(TargetPending | (temporary_static ? TargetTemporaryStatic : 0u), 0u, 0u, 0u);
            }
        }
        else if (kind == CommandMotionBoundary)
        {
            uint action = mode & MotionBoundaryActionMask;
            bool had_move = jph_has_kinematic_move(target);
            if (action == MotionBoundaryProjectMobility)
            {
                // New model is accepted first. Keep forces and target identity.
                jph_project_motion(model, state);
                command_reset_sleep(state,sleep_extent,sleep);
                continue;
            }
            if (action == MotionBoundaryTypeChange)
            {
                // Carry protected author channels from the device target,
                // never from a newer CPU pose. COM mutation rebased rootCOM.
                uint carry = mode & (MotionBoundaryCarryAuthoredPosition | MotionBoundaryCarryAuthoredRotation);
                if (had_move && carry != 0u)
                {
                    float3 origin = state.positionCOM.xyz - jph_rotate(state.rotation, target.rootCOMDt.xyz);
                    if ((carry & MotionBoundaryCarryAuthoredPosition) != 0u)
                        origin = target.positionSpeed.xyz;
                    if ((carry & MotionBoundaryCarryAuthoredRotation) != 0u)
                        state.rotation = target.rotation;
                    state.positionCOM.xyz = origin + jph_rotate(state.rotation, target.rootCOMDt.xyz);
                }
                uint motion = model.meta.z & ~MotionNotResident;
                if (motion == MotionStatic)
                {
                    state.linearVelocity = float4(0.0f);
                    state.angularVelocity = float4(0.0f);
                }
                if (motion != MotionDynamic)
                {
                    forces.force = float4(0.0f);
                    forces.torque = float4(0.0f);
                }
            }
            else if (action == MotionBoundarySuspendReset || had_move)
            {
                // Suspension stops target-owned motion. Explicit manual
                // velocity survives preserve mode; force accumulation survives
                // either suspension mode, matching ordinary body removal.
                state.linearVelocity = float4(0.0f);
                state.angularVelocity = float4(0.0f);
            }
            target.state = uint4(0u);
            jph_project_motion(model, state);
            ++sleep.timing.y;
            if (action != MotionBoundarySuspendPreserve)
                command_reset_sleep(state,sleep_extent,sleep);
            else
                sleep.state.w=0u; // Preserve the hidden episode's three spheres/timer/activity.
            if (action == MotionBoundaryTypeChange)
                sleep.state.x=jph_nonstatic(model)?1u:0u;
            else if (action == MotionBoundarySuspendReset)
                sleep.state.x=1u;
        }
        else if (kind == CommandSetLinearVelocity || kind == CommandSetAngularVelocity)
        {
            if (jph_nonstatic(model))
            {
                // Explicit velocity supersedes the whole arrival stop while
                // preserving the other channel's actual current velocity.
                target.state = uint4(0u);
                sleep.state.w=0u;
                if (kind == CommandSetLinearVelocity)
                    state.linearVelocity.xyz = jph_clamp_speed(jph_project_linear(model, command.value.xyz), model.damping.z);
                else
                    state.angularVelocity.xyz = jph_clamp_speed(jph_project_angular(model, state, command.value.xyz), model.damping.w);
                if (any(command.value.xyz != float3(0)))
                    command_wake(model,state,sleep_extent,sleep,false);
            }
        }
        else if (jph_dynamic(model))
        {
            if (any(command.value.xyz != float3(0)))
                command_wake(model,state,sleep_extent,sleep,mode == CommandForce || mode == CommandAcceleration);
            if (kind == CommandAddTorque)
            {
                float3 torque = command.value.xyz;
                // Acceleration converts to a world torque NOW. A later pose
                // command must not change the inertia used for this conversion.
                if (mode == CommandAcceleration || mode == CommandVelocityChange)
                    torque = command_world_inertia(model, state, torque);
                if (mode == CommandForce || mode == CommandAcceleration)
                    forces.torque.xyz += torque;
                else
                    state.angularVelocity.xyz = jph_clamp_speed(
                        state.angularVelocity.xyz + jph_inverse_inertia(model, state, torque), model.damping.w);
            }
            else
            {
                float3 force = command.value.xyz;
                if (mode == CommandAcceleration || mode == CommandVelocityChange)
                    force /= model.mass.x;
                float3 torque = kind == CommandAddForceAtPosition
                    ? cross(command.auxiliary.xyz - state.positionCOM.xyz, force) : float3(0.0f);
                if (mode == CommandForce || mode == CommandAcceleration)
                {
                    forces.force.xyz += force;
                    forces.torque.xyz += torque;
                }
                else
                {
                    state.linearVelocity.xyz = jph_clamp_speed(
                        state.linearVelocity.xyz + jph_inverse_linear_mass(model, force), model.damping.z);
                    if (kind == CommandAddForceAtPosition)
                        state.angularVelocity.xyz = jph_clamp_speed(
                            state.angularVelocity.xyz + jph_inverse_inertia(model, state, torque), model.damping.w);
                }
            }
        }
    }
    // Every selected span writes its complete slot values, including when
    // individual kinds have no effect for the accepted author motion type.
    body_states[body] = state;
    body_forces[body] = forces;
    kinematic_targets[body] = target;
    body_sleep[body].runtime = sleep;
}
