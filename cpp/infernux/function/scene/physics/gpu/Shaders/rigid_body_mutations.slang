// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#include "rigid_math.slang"
#include "rigid_sleep_math.slang"

ConstantBuffer<BodyMutationParams> mutation_params;
ConstantBuffer<StageWork> stage_work;
StructuredBuffer<BodyInitialization> body_initializations;
StructuredBuffer<CenterOfMassUpdate> center_of_mass_updates;
RWStructuredBuffer<BodyState> body_states;
RWStructuredBuffer<BodyForces> body_forces;
RWStructuredBuffer<KinematicTarget> kinematic_targets;
RWStructuredBuffer<BodySleepRecord> body_sleep;

[shader("compute")]
[numthreads(64, 1, 1)]
void initialize_capacity(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y)
        return;
    uint index = stage_work.range.x + tid.x;
    BodyState state;
    state.positionCOM = float4(0.0f);
    state.rotation = float4(0.0f, 0.0f, 0.0f, 1.0f);
    state.linearVelocity = float4(0.0f);
    state.angularVelocity = float4(0.0f);
    BodyForces forces;
    forces.force = float4(0.0f);
    forces.torque = float4(0.0f);
    body_states[index] = state;
    body_forces[index] = forces;
    kinematic_targets[index] = jph_empty_kinematic_target();
    BodySleepRecord sleep;
    sleep.author.rootHalfExtent = sleep.author.rootBoundsCenter = float4(0);
    sleep.author.policy = uint4(0);
    sleep.runtime.sphere0 = sleep.runtime.sphere1 = sleep.runtime.sphere2 = float4(0);
    sleep.runtime.timing = uint4(0);
    sleep.runtime.state = uint4(0);
    body_sleep[index] = sleep;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void initialize_bodies(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= mutation_params.counts.x)
        return;
    BodyInitialization initial = body_initializations[mutation_params.counts.z + tid.x];
    body_states[initial.bodyIndex.x] = initial.state;
    body_forces[initial.bodyIndex.x] = initial.forces;
    kinematic_targets[initial.bodyIndex.x] = jph_empty_kinematic_target();
    BodySleepRuntime sleep = body_sleep[initial.bodyIndex.x].runtime;
    float3 p0,p1,p2;
    sleep_test_points(initial.state, body_sleep[initial.bodyIndex.x].author.rootHalfExtent.xyz, p0,p1,p2);
    sleep_reset(sleep,p0,p1,p2);
    ++sleep.timing.y;
    sleep.state.x=1u;
    body_sleep[initial.bodyIndex.x].runtime=sleep;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void update_centers_of_mass(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= mutation_params.counts.y)
        return;
    CenterOfMassUpdate change = center_of_mass_updates[tid.x];
    BodyState state = body_states[change.bodyIndex.x];
    // Body::UpdateCenterOfMassInternal preserves the world body origin and
    // does not modify velocities, accumulated forces or the body rotation.
    state.positionCOM.xyz += jph_rotate(state.rotation, change.deltaLocalCOM.xyz);
    body_states[change.bodyIndex.x] = state;
    KinematicTarget target = kinematic_targets[change.bodyIndex.x];
    if (jph_has_kinematic_move(target))
        target.rootCOMDt.xyz += change.deltaLocalCOM.xyz;
    kinematic_targets[change.bodyIndex.x] = target;
    if (any(change.deltaLocalCOM.xyz != float3(0))) {
        BodySleepRuntime sleep = body_sleep[change.bodyIndex.x].runtime;
        float3 p0,p1,p2;
        sleep_test_points(state, body_sleep[change.bodyIndex.x].author.rootHalfExtent.xyz,p0,p1,p2);
        sleep_reset(sleep,p0,p1,p2);
        ++sleep.timing.y;
        body_sleep[change.bodyIndex.x].runtime=sleep;
    }
}
