// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021-2026 Jorrit Rouwe
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT

#include "rigid_joints.slang"
#include "rigid_convex.slang"
#include "rigid_sleep_math.slang"

ConstantBuffer<StepParams> step_params;
ConstantBuffer<StageWork> stage_work;
StructuredBuffer<BodyModel> body_models;
StructuredBuffer<ShapeModel> shape_models;
RWStructuredBuffer<BodyForces> body_forces;
// Subsequent contact levels read the body states written by other lanes.
globallycoherent RWStructuredBuffer<BodyState> body_states;
RWStructuredBuffer<Aabb> bounds;
RWStructuredBuffer<Contact> contacts;
RWStructuredBuffer<KinematicTarget> kinematic_targets;
RWStructuredBuffer<ContactPatch> contact_patches;
StructuredBuffer<uint4> convex_geometry;
RWStructuredBuffer<BodySleepRecord> body_sleep;
RWStructuredBuffer<QueryRecordWord> contact_history;
RWStructuredBuffer<SweepBody> sweep_bodies;
ConstantBuffer<BodyBatchHeader> body_batch_header;
StructuredBuffer<StepParams> body_batch_selections;
StructuredBuffer<BodyBatchTile> body_batch_tiles;

bool jph_sleep_awake_dynamic(uint body)
{
    if (body == InvalidQueryBodySlot) return false;
    return jph_dynamic(body_models[body]) && body_sleep[body].runtime.state.x != 0u;
}
bool jph_sleep_pair_awake(uint a,uint b)
{
    return jph_sleep_awake_dynamic(a) || jph_sleep_awake_dynamic(b);
}
bool jph_sleep_dormant_joint(JointDirectory directory)
{
    // Preserve the existing nonresident warm/rotor cleanup path. Sleep is
    // activity, not residency, and never replaces that separate boundary.
    bool resident = jph_resident(body_models[directory.bodies.y]);
    if(directory.bodies.x != InvalidQueryBodySlot)
        resident = resident && jph_resident(body_models[directory.bodies.x]);
    return resident && !jph_sleep_pair_awake(directory.bodies.x,directory.bodies.y);
}

bool jph_selected_range(uint index, uint base, uint count)
{
    return index >= base && index - base < count;
}
bool jph_selected_body_slot(uint index, StepParams parameters)
{
    return index < parameters.counts.x && jph_selected_range(index, parameters.selection.y, parameters.selectedCounts.x);
}
bool jph_selected_body(uint index, StepParams parameters)
{
    if (!jph_selected_body_slot(index, parameters)) return false;
    return body_models[index].meta.w == parameters.selection.x;
}
bool jph_selected_body_slot(uint index)
{
    return jph_selected_body_slot(index, step_params);
}
bool jph_selected_body(uint index)
{
    return jph_selected_body(index, step_params);
}
bool jph_selected_joint(uint index, JointDirectory author)
{
    return jph_selected_range(index, step_params.selection.w, step_params.selectedCounts.z)
        && author.identity.z == step_params.selection.x && jph_selected_body_slot(author.bodies.y)
        && (author.bodies.x == InvalidQueryBodySlot || jph_selected_body_slot(author.bodies.x));
}

#include "island_owner_schedule.slang"
#include "rigid_articulations.slang"

void jph_joint_models(JointDirectory author, out BodyModel m1, out BodyModel m2)
{
    m1 = jph_joint_world_model();
    if (author.bodies.x != InvalidQueryBodySlot)
        m1 = jph_effective_model(body_models[author.bodies.x], kinematic_targets[author.bodies.x]);
    m2 = jph_effective_model(body_models[author.bodies.y], kinematic_targets[author.bodies.y]);
}

void jph_joint_states(JointDirectory author, out BodyState s1, out BodyState s2)
{
    s1 = jph_joint_world_state();
    if (author.bodies.x != InvalidQueryBodySlot)
        s1 = body_states[author.bodies.x];
    s2 = body_states[author.bodies.y];
}

void jph_joint_bodies(JointDirectory author, out BodyModel m1, out BodyState s1,
    out BodyModel m2, out BodyState s2)
{
    jph_joint_models(author, m1, m2);
    jph_joint_states(author, s1, s2);
}

// One island lane alone owns every Dynamic endpoint of its numerical rows.
// All-stream SetupJoints/FinishJoints retain independent runtime maintenance.
void jph_step_joint_typed(uint phase, JointDirectory directory, JointFixedAuthor author, JointFixedState runtime)
{
    // Inner solver early returns must never be followed by stale full-state
    // stores from an inactive joint assigned outside another island's owner.
    // SetupJoints still checks the complete accepted stream before graph build;
    // a failed or uninitialized row cannot be accepted as a successful empty row.
    if (runtime.flags.x == 0u || runtime.flags.y != JointStatusSuccess
        || (author.bodies.w & JointEnabled) == 0u)
        return;
    BodyModel m1, m2;
    jph_joint_models(directory, m1, m2);
    if (!jph_resident(m1) || !jph_resident(m2))
    {
        jph_joint_clear_impulses(runtime);
        jph_joint_store_state(directory, runtime);
        return;
    }
    if (!jph_dynamic(m1) && !jph_dynamic(m2)) return;
    BodyState s1, s2;
    jph_joint_states(directory, s1, s2);
    BodyState old1 = s1, old2 = s2;
    if (phase == 1u)
        jph_joint_solve_velocity(author, runtime, m1, s1, m2, s2, step_params.gravityDt.w);
    else if (phase == 2u)
        jph_joint_solve_position(author, runtime, m1, s1, m2, s2, step_params.contact.w);
    else
        jph_joint_warm_start(author, runtime, m1, s1, m2, s2);
    if (runtime.flags.y != JointStatusSuccess || !jph_joint_finite(s1) || !jph_joint_finite(s2)
        || !jph_joint_state_finite(runtime))
    {
        if (runtime.flags.y == JointStatusSuccess)
            runtime.flags.y = JointStatusNumericalFailure;
        s1 = old1;
        s2 = old2;
    }
    if (jph_dynamic(m1)) body_states[author.bodies.x] = s1;
    if (jph_dynamic(m2)) body_states[author.bodies.y] = s2;
    jph_joint_store_state(directory, runtime);
}

void jph_step_joint_typed(uint phase, JointDirectory directory, JointHingeAuthor author, JointHingeState runtime)
{
    // Inner solver early returns must never be followed by stale full-state
    // stores from an inactive joint assigned outside another island's owner.
    // SetupJoints still checks the complete accepted stream before graph build;
    // a failed or uninitialized row cannot be accepted as a successful empty row.
    if (runtime.flags.x == 0u || runtime.flags.y != JointStatusSuccess
        || (author.bodies.w & JointEnabled) == 0u)
        return;
    BodyModel m1, m2;
    jph_joint_models(directory, m1, m2);
    if (!jph_resident(m1) || !jph_resident(m2))
    {
        jph_joint_clear_impulses(runtime);
        jph_joint_store_state(directory, runtime);
        return;
    }
    if (!jph_dynamic(m1) && !jph_dynamic(m2)) return;
    BodyState s1, s2;
    jph_joint_states(directory, s1, s2);
    BodyState old1 = s1, old2 = s2;
    if (phase == 1u)
        jph_joint_solve_velocity(author, runtime, m1, s1, m2, s2, step_params.gravityDt.w);
    else if (phase == 2u)
        jph_joint_solve_position(author, runtime, m1, s1, m2, s2, step_params.contact.w);
    else
        jph_joint_warm_start(author, runtime, m1, s1, m2, s2);
    if (runtime.flags.y != JointStatusSuccess || !jph_joint_finite(s1) || !jph_joint_finite(s2)
        || !jph_joint_state_finite(runtime))
    {
        if (runtime.flags.y == JointStatusSuccess)
            runtime.flags.y = JointStatusNumericalFailure;
        s1 = old1;
        s2 = old2;
    }
    if (jph_dynamic(m1)) body_states[author.bodies.x] = s1;
    if (jph_dynamic(m2)) body_states[author.bodies.y] = s2;
    jph_joint_store_state(directory, runtime);
}

void jph_step_joint_typed(uint phase, JointDirectory directory, JointSliderAuthor author, JointSliderState runtime)
{
    // Inner solver early returns must never be followed by stale full-state
    // stores from an inactive joint assigned outside another island's owner.
    // SetupJoints still checks the complete accepted stream before graph build;
    // a failed or uninitialized row cannot be accepted as a successful empty row.
    if (runtime.flags.x == 0u || runtime.flags.y != JointStatusSuccess
        || (author.bodies.w & JointEnabled) == 0u)
        return;
    BodyModel m1, m2;
    jph_joint_models(directory, m1, m2);
    if (!jph_resident(m1) || !jph_resident(m2))
    {
        jph_joint_clear_impulses(runtime);
        jph_joint_store_state(directory, runtime);
        return;
    }
    if (!jph_dynamic(m1) && !jph_dynamic(m2)) return;
    BodyState s1, s2;
    jph_joint_states(directory, s1, s2);
    BodyState old1 = s1, old2 = s2;
    if (phase == 1u)
        jph_joint_solve_velocity(author, runtime, m1, s1, m2, s2, step_params.gravityDt.w);
    else if (phase == 2u)
        jph_joint_solve_position(author, runtime, m1, s1, m2, s2, step_params.contact.w);
    else
        jph_joint_warm_start(author, runtime, m1, s1, m2, s2);
    if (runtime.flags.y != JointStatusSuccess || !jph_joint_finite(s1) || !jph_joint_finite(s2)
        || !jph_joint_state_finite(runtime))
    {
        if (runtime.flags.y == JointStatusSuccess)
            runtime.flags.y = JointStatusNumericalFailure;
        s1 = old1;
        s2 = old2;
    }
    if (jph_dynamic(m1)) body_states[author.bodies.x] = s1;
    if (jph_dynamic(m2)) body_states[author.bodies.y] = s2;
    jph_joint_store_state(directory, runtime);
}

void jph_step_joint_typed(uint phase, JointDirectory directory, JointDistanceAuthor author, JointDistanceState runtime)
{
    // Inner solver early returns must never be followed by stale full-state
    // stores from an inactive joint assigned outside another island's owner.
    // SetupJoints still checks the complete accepted stream before graph build;
    // a failed or uninitialized row cannot be accepted as a successful empty row.
    if (runtime.flags.x == 0u || runtime.flags.y != JointStatusSuccess
        || (author.bodies.w & JointEnabled) == 0u)
        return;
    BodyModel m1, m2;
    jph_joint_models(directory, m1, m2);
    if (!jph_resident(m1) || !jph_resident(m2))
    {
        jph_joint_clear_impulses(runtime);
        jph_joint_store_state(directory, runtime);
        return;
    }
    if (!jph_dynamic(m1) && !jph_dynamic(m2)) return;
    BodyState s1, s2;
    jph_joint_states(directory, s1, s2);
    BodyState old1 = s1, old2 = s2;
    if (phase == 1u)
        jph_joint_solve_velocity(author, runtime, m1, s1, m2, s2, step_params.gravityDt.w);
    else if (phase == 2u)
        jph_joint_solve_position(author, runtime, m1, s1, m2, s2, step_params.contact.w);
    else
        jph_joint_warm_start(author, runtime, m1, s1, m2, s2);
    if (runtime.flags.y != JointStatusSuccess || !jph_joint_finite(s1) || !jph_joint_finite(s2)
        || !jph_joint_state_finite(runtime))
    {
        if (runtime.flags.y == JointStatusSuccess)
            runtime.flags.y = JointStatusNumericalFailure;
        s1 = old1;
        s2 = old2;
    }
    if (jph_dynamic(m1)) body_states[author.bodies.x] = s1;
    if (jph_dynamic(m2)) body_states[author.bodies.y] = s2;
    jph_joint_store_state(directory, runtime);
}

void jph_step_joint_typed(uint phase, JointDirectory directory, JointSwingAuthor author, JointSwingState runtime)
{
    // Inner solver early returns must never be followed by stale full-state
    // stores from an inactive joint assigned outside another island's owner.
    // SetupJoints still checks the complete accepted stream before graph build;
    // a failed or uninitialized row cannot be accepted as a successful empty row.
    if (runtime.flags.x == 0u || runtime.flags.y != JointStatusSuccess
        || (author.bodies.w & JointEnabled) == 0u)
        return;
    BodyModel m1, m2;
    jph_joint_models(directory, m1, m2);
    if (!jph_resident(m1) || !jph_resident(m2))
    {
        jph_joint_clear_impulses(runtime);
        jph_joint_store_state(directory, runtime);
        return;
    }
    if (!jph_dynamic(m1) && !jph_dynamic(m2)) return;
    BodyState s1, s2;
    jph_joint_states(directory, s1, s2);
    BodyState old1 = s1, old2 = s2;
    if (phase == 1u)
        jph_joint_solve_velocity(author, runtime, m1, s1, m2, s2, step_params.gravityDt.w);
    else if (phase == 2u)
        jph_joint_solve_position(author, runtime, m1, s1, m2, s2, step_params.contact.w);
    else
        jph_joint_warm_start(author, runtime, m1, s1, m2, s2);
    if (runtime.flags.y != JointStatusSuccess || !jph_joint_finite(s1) || !jph_joint_finite(s2)
        || !jph_joint_state_finite(runtime))
    {
        if (runtime.flags.y == JointStatusSuccess)
            runtime.flags.y = JointStatusNumericalFailure;
        s1 = old1;
        s2 = old2;
    }
    if (jph_dynamic(m1)) body_states[author.bodies.x] = s1;
    if (jph_dynamic(m2)) body_states[author.bodies.y] = s2;
    jph_joint_store_state(directory, runtime);
}

void jph_step_joint_typed(uint phase, JointDirectory directory, JointSixAuthor author, JointSixState runtime)
{
    // Inner solver early returns must never be followed by stale full-state
    // stores from an inactive joint assigned outside another island's owner.
    // SetupJoints still checks the complete accepted stream before graph build;
    // a failed or uninitialized row cannot be accepted as a successful empty row.
    if (runtime.flags.x == 0u || runtime.flags.y != JointStatusSuccess
        || (author.bodies.w & JointEnabled) == 0u)
        return;
    BodyModel m1, m2;
    jph_joint_models(directory, m1, m2);
    if (!jph_resident(m1) || !jph_resident(m2))
    {
        jph_joint_clear_impulses(runtime);
        jph_joint_store_state(directory, runtime);
        return;
    }
    if (!jph_dynamic(m1) && !jph_dynamic(m2)) return;
    BodyState s1, s2;
    jph_joint_states(directory, s1, s2);
    BodyState old1 = s1, old2 = s2;
    if (phase == 1u)
        jph_joint_solve_velocity(author, runtime, m1, s1, m2, s2, step_params.gravityDt.w);
    else if (phase == 2u)
        jph_joint_solve_position(author, runtime, m1, s1, m2, s2, step_params.contact.w);
    else
        jph_joint_warm_start(author, runtime, m1, s1, m2, s2);
    if (runtime.flags.y != JointStatusSuccess || !jph_joint_finite(s1) || !jph_joint_finite(s2)
        || !jph_joint_state_finite(runtime))
    {
        if (runtime.flags.y == JointStatusSuccess)
            runtime.flags.y = JointStatusNumericalFailure;
        s1 = old1;
        s2 = old2;
    }
    if (jph_dynamic(m1)) body_states[author.bodies.x] = s1;
    if (jph_dynamic(m2)) body_states[author.bodies.y] = s2;
    jph_joint_store_state(directory, runtime);
}

void jph_step_joint(uint index, uint phase)
{
    if (jph_articulation_admitted(index, true)) return;
    JointDirectory directory = jph_joint_load_directory(index);
    if (!jph_selected_joint(index, directory) || !jph_selected_body(directory.bodies.y)
        || (directory.bodies.x != InvalidQueryBodySlot && !jph_selected_body(directory.bodies.x))) return;
    if (jph_sleep_dormant_joint(directory)) return;
    switch (directory.bodies.z)
    {
    case JointFixed:
    {
        JointFixedAuthor author; jph_joint_load_author(directory, author);
        JointFixedState runtime; jph_joint_load_state(directory, runtime);
        jph_step_joint_typed(phase, directory, author, runtime);
        break;
    }
    case JointHinge:
    {
        JointHingeAuthor author; jph_joint_load_author(directory, author);
        JointHingeState runtime; jph_joint_load_state(directory, runtime);
        jph_step_joint_typed(phase, directory, author, runtime);
        break;
    }
    case JointSlider:
    {
        JointSliderAuthor author; jph_joint_load_author(directory, author);
        JointSliderState runtime; jph_joint_load_state(directory, runtime);
        jph_step_joint_typed(phase, directory, author, runtime);
        break;
    }
    case JointDistance:
    {
        JointDistanceAuthor author; jph_joint_load_author(directory, author);
        JointDistanceState runtime; jph_joint_load_state(directory, runtime);
        jph_step_joint_typed(phase, directory, author, runtime);
        break;
    }
    case JointSwingTwist:
    {
        JointSwingAuthor author; jph_joint_load_author(directory, author);
        JointSwingState runtime; jph_joint_load_state(directory, runtime);
        jph_step_joint_typed(phase, directory, author, runtime);
        break;
    }
    case JointSixDOF:
    {
        JointSixAuthor author; jph_joint_load_author(directory, author);
        JointSixState runtime; jph_joint_load_state(directory, runtime);
        jph_step_joint_typed(phase, directory, author, runtime);
        break;
    }
    }
}

void jph_step_island_joints(uint island_lane, uint phase)
{
    uint begin, first, end;
    if (!jph_island_joint_range(island_lane, begin, first, end)) return;
    for (uint local = first; local < end; ++local)
        jph_step_joint(jph_island_joint_source(begin, local), phase);
}

void jph_setup_joint_typed(JointDirectory directory, JointFixedAuthor author, JointFixedState runtime)
{
    BodyModel m1, m2;
    BodyState s1, s2;
    jph_joint_bodies(directory, m1, s1, m2, s2);
    if (!jph_resident(m1) || !jph_resident(m2))
        jph_joint_clear_impulses(runtime);
    else
    {
        // This stage stores runtime only. All global setup units precede any
        // warm unit, so shared endpoints are read before warm impulses change w.
        jph_joint_setup(author, runtime, m1, s1, m2, s2,
            step_params.gravityDt.w, step_params.counts.w != 0u);
        if (runtime.flags.y == JointStatusSuccess
            && (!jph_joint_finite(s1) || !jph_joint_finite(s2) || !jph_joint_state_finite(runtime)))
            runtime.flags.y = JointStatusNumericalFailure;
    }
    jph_joint_store_state(directory, runtime);
}

void jph_setup_joint_typed(JointDirectory directory, JointHingeAuthor author, JointHingeState runtime)
{
    BodyModel m1, m2;
    BodyState s1, s2;
    jph_joint_bodies(directory, m1, s1, m2, s2);
    if (!jph_resident(m1) || !jph_resident(m2))
    {
        // Suspension clears warm reactions, not the physical rotor speed.
        float old_armature = runtime.rotor.w;
        float speed = old_armature > 0.0f ? runtime.rotor.x - runtime.rotor.y / old_armature : runtime.rotor.x;
        jph_joint_clear_impulses(runtime);
        runtime.rotor.xz = float2(speed);
        if (!isfinite(speed)) runtime.flags.y = JointStatusNumericalFailure;
    }
    else
    {
        // This stage stores runtime only. All global setup units precede any
        // warm unit, so shared endpoints are read before warm impulses change w.
        jph_joint_setup(author, runtime, m1, s1, m2, s2,
            step_params.gravityDt.w, step_params.counts.w != 0u);
        if (runtime.flags.y == JointStatusSuccess
            && (!jph_joint_finite(s1) || !jph_joint_finite(s2) || !jph_joint_state_finite(runtime)))
            runtime.flags.y = JointStatusNumericalFailure;
    }
    jph_joint_store_state(directory, runtime);
}

void jph_setup_joint_typed(JointDirectory directory, JointSliderAuthor author, JointSliderState runtime)
{
    BodyModel m1, m2;
    BodyState s1, s2;
    jph_joint_bodies(directory, m1, s1, m2, s2);
    if (!jph_resident(m1) || !jph_resident(m2))
        jph_joint_clear_impulses(runtime);
    else
    {
        // This stage stores runtime only. All global setup units precede any
        // warm unit, so shared endpoints are read before warm impulses change w.
        jph_joint_setup(author, runtime, m1, s1, m2, s2,
            step_params.gravityDt.w, step_params.counts.w != 0u);
        if (runtime.flags.y == JointStatusSuccess
            && (!jph_joint_finite(s1) || !jph_joint_finite(s2) || !jph_joint_state_finite(runtime)))
            runtime.flags.y = JointStatusNumericalFailure;
    }
    jph_joint_store_state(directory, runtime);
}

void jph_setup_joint_typed(JointDirectory directory, JointDistanceAuthor author, JointDistanceState runtime)
{
    BodyModel m1, m2;
    BodyState s1, s2;
    jph_joint_bodies(directory, m1, s1, m2, s2);
    if (!jph_resident(m1) || !jph_resident(m2))
        jph_joint_clear_impulses(runtime);
    else
    {
        // This stage stores runtime only. All global setup units precede any
        // warm unit, so shared endpoints are read before warm impulses change w.
        jph_joint_setup(author, runtime, m1, s1, m2, s2,
            step_params.gravityDt.w, step_params.counts.w != 0u);
        if (runtime.flags.y == JointStatusSuccess
            && (!jph_joint_finite(s1) || !jph_joint_finite(s2) || !jph_joint_state_finite(runtime)))
            runtime.flags.y = JointStatusNumericalFailure;
    }
    jph_joint_store_state(directory, runtime);
}

void jph_setup_joint_typed(JointDirectory directory, JointSwingAuthor author, JointSwingState runtime)
{
    BodyModel m1, m2;
    BodyState s1, s2;
    jph_joint_bodies(directory, m1, s1, m2, s2);
    if (!jph_resident(m1) || !jph_resident(m2))
        jph_joint_clear_impulses(runtime);
    else
    {
        // This stage stores runtime only. All global setup units precede any
        // warm unit, so shared endpoints are read before warm impulses change w.
        jph_joint_setup(author, runtime, m1, s1, m2, s2,
            step_params.gravityDt.w, step_params.counts.w != 0u);
        if (runtime.flags.y == JointStatusSuccess
            && (!jph_joint_finite(s1) || !jph_joint_finite(s2) || !jph_joint_state_finite(runtime)))
            runtime.flags.y = JointStatusNumericalFailure;
    }
    jph_joint_store_state(directory, runtime);
}

void jph_setup_joint_typed(JointDirectory directory, JointSixAuthor author, JointSixState runtime)
{
    BodyModel m1, m2;
    BodyState s1, s2;
    jph_joint_bodies(directory, m1, s1, m2, s2);
    if (!jph_resident(m1) || !jph_resident(m2))
        jph_joint_clear_impulses(runtime);
    else
    {
        // This stage stores runtime only. All global setup units precede any
        // warm unit, so shared endpoints are read before warm impulses change w.
        jph_joint_setup(author, runtime, m1, s1, m2, s2,
            step_params.gravityDt.w, step_params.counts.w != 0u);
        if (runtime.flags.y == JointStatusSuccess
            && (!jph_joint_finite(s1) || !jph_joint_finite(s2) || !jph_joint_state_finite(runtime)))
            runtime.flags.y = JointStatusNumericalFailure;
    }
    jph_joint_store_state(directory, runtime);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void setup_joints(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint index = stage_work.range.x + tid.x;
    if (index >= joint_stream[0].value.x) return;
    JointDirectory directory = jph_joint_load_directory(index);
    if (!jph_selected_joint(index, directory) || !jph_selected_body(directory.bodies.y)
        || (directory.bodies.x != InvalidQueryBodySlot && !jph_selected_body(directory.bodies.x))) return;
    if (jph_sleep_dormant_joint(directory)) return;
    switch (directory.bodies.z)
    {
    case JointFixed:
    {
        JointFixedAuthor author; jph_joint_load_author(directory, author);
        JointFixedState runtime; jph_joint_load_state(directory, runtime);
        jph_setup_joint_typed(directory, author, runtime);
        break;
    }
    case JointHinge:
    {
        JointHingeAuthor author; jph_joint_load_author(directory, author);
        JointHingeState runtime; jph_joint_load_state(directory, runtime);
        jph_setup_joint_typed(directory, author, runtime);
        break;
    }
    case JointSlider:
    {
        JointSliderAuthor author; jph_joint_load_author(directory, author);
        JointSliderState runtime; jph_joint_load_state(directory, runtime);
        jph_setup_joint_typed(directory, author, runtime);
        break;
    }
    case JointDistance:
    {
        JointDistanceAuthor author; jph_joint_load_author(directory, author);
        JointDistanceState runtime; jph_joint_load_state(directory, runtime);
        jph_setup_joint_typed(directory, author, runtime);
        break;
    }
    case JointSwingTwist:
    {
        JointSwingAuthor author; jph_joint_load_author(directory, author);
        JointSwingState runtime; jph_joint_load_state(directory, runtime);
        jph_setup_joint_typed(directory, author, runtime);
        break;
    }
    case JointSixDOF:
    {
        JointSixAuthor author; jph_joint_load_author(directory, author);
        JointSixState runtime; jph_joint_load_state(directory, runtime);
        jph_setup_joint_typed(directory, author, runtime);
        break;
    }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void warm_joints(uint3 tid : SV_DispatchThreadID)
{
    jph_step_island_joints(tid.x, 3u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void solve_joint_velocity(uint3 tid : SV_DispatchThreadID)
{
    jph_step_island_joints(tid.x, 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void solve_joint_position(uint3 tid : SV_DispatchThreadID)
{
    jph_step_island_joints(tid.x, 2u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void finish_joints(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint index = stage_work.range.x + tid.x;
    if (index >= joint_stream[0].value.x) return;
    JointDirectory directory = jph_joint_load_directory(index);
    if (!jph_selected_joint(index, directory)) return;
    if (jph_sleep_dormant_joint(directory)) return;
    // All actual payloads share the first flags word. Previous dt belongs to
    // flags.z; Hinge rotor.w retains its cached physical armature.
    uint4 flags = joint_stream[directory.layout.y].value;
    flags.z = asuint(step_params.gravityDt.w);
    joint_stream[directory.layout.y].value = flags;
}

// Only the primitive geometry uses this local pose. Body state, principal
// inertia and contact anchors remain in the original body COM frame.
float3 jph_leaf_position(ShapeModel shape, BodyState state)
{
    return state.positionCOM.xyz + jph_rotate(state.rotation, shape.localPositionCOM.xyz);
}

float4 jph_leaf_rotation(ShapeModel shape, BodyState state)
{
    return jph_quat_mul(state.rotation, shape.localRotation);
}

void jph_body_prepare_kinematics(uint index, StepParams parameters)
{
    if (!jph_selected_body(index, parameters))
        return;
    BodyModel model = body_models[index];
    BodySleepRuntime sleep = body_sleep[index].runtime;
    // Seed a newly published row from the tick's original pose, before applying a drive.
    if (sleep.state.y == 0u) {
        float3 p0, p1, p2;
        sleep_test_points(body_states[index], body_sleep[index].author.rootHalfExtent.xyz, p0, p1, p2);
        sleep_reset(sleep, p0, p1, p2);
        sleep.state.x = jph_nonstatic(model) ? 1u : 0u;
    }
    // Overwrite the derived activity on every selected row, including early
    // exits for residency/motion/target stop. World endpoints have no slot.
    sleep.state.w = 0u;
    body_sleep[index].runtime = sleep;
    if (!jph_resident(model))
        return;
    KinematicTarget target = kinematic_targets[index];
    uint motion = model.meta.z & ~MotionNotResident;
    if (motion != MotionKinematic &&
        !(motion == MotionStatic && (target.state.x & TargetTemporaryStatic) != 0u))
        return;
    BodyState state = body_states[index];
    BodyState previous_state = state;
    if ((target.state.x & TargetPending) != 0u)
    {
        // Capture this logical tick's start once. Requested time belongs to
        // the command; only later Integrate dispatches use substep time h.
        float requested_dt = target.rootCOMDt.w;
        float3 root_com = target.rootCOMDt.xyz;
        // Capture one reachable endpoint at the actual tick start. The COM
        // conversion and prescribed angular motion must use this same pose.
        float3 angular_velocity = jph_project_angular(model, state,
            jph_kinematic_angular_velocity(target.rotation, state.rotation, requested_dt));
        float4 reachable_rotation = model.mass.z == 0.0f ? target.rotation
            : jph_kinematic_rotation_step(state.rotation, angular_velocity * requested_dt);
        float3 origin = state.positionCOM.xyz - jph_rotate(state.rotation, root_com);
        float3 delta = target.positionSpeed.xyz - origin;
        float maximum_distance = target.positionSpeed.w * requested_dt;
        float distance = length(delta);
        if (target.positionSpeed.w > 0.0f && distance > maximum_distance)
        {
            // Preserve the old origin-based transform-drive cap exactly:
            // excess translation AND rotation teleport before the last stretch.
            origin = target.positionSpeed.xyz - delta * (maximum_distance / distance);
            float3 capped_com = origin + jph_rotate(reachable_rotation, root_com);
            if (model.mass.y == 0.0f) state.positionCOM.xyz = capped_com;
            else state.positionCOM.xyz += jph_project_linear(model, capped_com - state.positionCOM.xyz);
            state.rotation = reachable_rotation;
            angular_velocity = float3(0.0f); // Rotation used the existing cap teleport.
            jph_project_motion(model, state);
        }
        float3 target_com = target.positionSpeed.xyz + jph_rotate(reachable_rotation, root_com);
        state.linearVelocity = float4(jph_project_linear(model, (target_com - state.positionCOM.xyz) / requested_dt), 0.0f);
        state.angularVelocity = float4(angular_velocity, 0.0f);
        // Preparation runs once per Fixed. A new target replaces this drive; otherwise
        // the next preparation clears its velocity after all current substeps.
        target.state.x = (target.state.x & TargetTemporaryStatic) | TargetDriven;
    }
    else if ((target.state.x & TargetDriven) != 0u)
    {
        state.linearVelocity = float4(0.0f);
        state.angularVelocity = float4(0.0f);
        target.state = uint4(0u);
    }
    body_states[index] = state;
    kinematic_targets[index] = target;
    bool moving = any(state.linearVelocity.xyz != float3(0)) || any(state.angularVelocity.xyz != float3(0));
    bool changed = any(state.positionCOM.xyz != previous_state.positionCOM.xyz)
        || !(all(state.rotation == previous_state.rotation) || all(state.rotation == -previous_state.rotation));
    if (motion == MotionKinematic && (moving || changed) && sleep.state.x == 0u) {
        float3 p0,p1,p2;
        sleep_test_points(state,body_sleep[index].author.rootHalfExtent.xyz,p0,p1,p2);
        sleep_reset(sleep,p0,p1,p2);sleep.state.x=1u;
    }
    BodyModel effective = jph_effective_model(model,target);
    sleep.state.w = motion == MotionStatic && (effective.meta.z & ~MotionNotResident) == MotionKinematic
        && (moving || changed) ? 1u : 0u;
    body_sleep[index].runtime = sleep;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void prepare_kinematics(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint index = stage_work.range.x + tid.x;
    jph_body_prepare_kinematics(index, step_params);
}

void jph_body_apply_forces(uint index, StepParams parameters)
{
    if (!jph_selected_body(index, parameters))
        return;
    BodyModel model = body_models[index];
    if (!jph_dynamic(model) || !jph_resident(model) || body_sleep[index].runtime.state.x == 0u)
        return;
    BodyState state = body_states[index];
    jph_apply_body_inputs(model, state, body_forces[index], parameters.gravityDt);
    body_states[index] = state;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void apply_forces(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint index = stage_work.range.x + tid.x;
    jph_body_apply_forces(index, step_params);
}

void jph_build_leaf_bounds(uint index, StepParams parameters)
{
    if (index >= parameters.counts.z || !jph_selected_range(index, parameters.selection.z, parameters.selectedCounts.y))
        return;
    ShapeModel shape = shape_models[index];
    if (!jph_selected_body_slot(shape.meta.y, parameters)) return;
    CvxOperand operand = cvx_leaf_operand(shape, body_states[shape.meta.y]);
    // The two half bands add up to the configured speculative distance.
    bounds[index] = cvx_leaf_bounds(convex_geometry, operand, 0.5f * parameters.contact.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void build_bounds(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x < stage_work.range.y)
        jph_build_leaf_bounds(stage_work.range.x + tid.x, step_params);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void narrowphase_candidates(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y || stage_work.range.x >= step_params.solve.w) return;
    if (tid.x >= step_params.solve.w - stage_work.range.x) return;
    uint candidate_ordinal = stage_work.range.x + tid.x;
    uint candidate_base = contact_patches[0].leaves.x;
    uint patch_index = candidate_base + candidate_ordinal; // WORD address.
    uint slot = candidate_ordinal * MaxContactPointsPerPair; // CONTACT ordinal.
    ContactPatch patch = contact_patches[patch_index];
    if (patch.leaves.w != step_params.selection.x
        || !jph_selected_range(patch.leaves.x, step_params.selection.z, step_params.selectedCounts.y)
        || !jph_selected_range(patch.leaves.y, step_params.selection.z, step_params.selectedCounts.y))
    {
        Contact empty;
        empty.meta = uint4(0u);
        empty.normalDepth = empty.localAnchorA = empty.localAnchorB = empty.impulse = float4(0.0f);
        for (uint point = 0u; point < MaxContactPointsPerPair; ++point) contacts[slot + point] = empty;
        return;
    }
    ShapeModel shape_a = shape_models[patch.leaves.x];
    ShapeModel shape_b = shape_models[patch.leaves.y];
    uint a = shape_a.meta.y;
    uint b = shape_b.meta.y;
    bool sensor = jph_sensor(shape_a) || jph_sensor(shape_b);
    // Eligibility was established in count/fill against the same frozen model
    // and bounds. There is no third filter that could change that contract.
    patch.leaves.z = sensor ? PatchSensor : 0u;
    patch.leaves.w = 0u;
    contact_patches[patch_index] = patch;
    Contact result;
    result.meta = uint4(a, b, 0u, 0u);
    result.normalDepth = float4(0.0f);
    result.localAnchorA = float4(0.0f);
    result.localAnchorB = float4(0.0f);
    result.impulse = float4(0.0f);
    // Every accepted candidate is a complete four-Contact writer, including
    // no-manifold and sensor candidates. Other capacity is not read/written.
    for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
        contacts[slot + point] = result;

    if (!jph_selected_body_slot(a) || !jph_selected_body_slot(b)) return;
    BodyState sa = body_states[a];
    BodyState sb = body_states[b];
    float4 ga = shape_a.geometry;
    float4 gb = shape_b.geometry;
    float3 ca = jph_leaf_position(shape_a, sa);
    float3 cb = jph_leaf_position(shape_b, sb);
    float4 qa = jph_leaf_rotation(shape_a, sa);
    float4 qb = jph_leaf_rotation(shape_b, sb);
    float speculative_distance = sensor ? 0.0f : step_params.contact.x;
    // The producer's generic jobs own these candidates. SetupContacts cannot
    // consume this initialized prefix until every persistent slot completes.
    if (shape_a.meta.x > ShapeCapsule || shape_b.meta.x > ShapeCapsule)
        return;
    bool box_pair = shape_a.meta.x == ShapeBox && shape_b.meta.x == ShapeBox;
    if (box_pair || shape_a.meta.x == ShapeCapsule || shape_b.meta.x == ShapeCapsule)
    {
        JphBoxManifold manifold;
        bool flip = shape_b.meta.x == ShapeCapsule && shape_a.meta.x != ShapeCapsule;
        if (box_pair)
            manifold = jph_box_box(ca, qa, ga, cb, qb, gb, speculative_distance);
        else if (shape_a.meta.x == ShapeCapsule && shape_b.meta.x == ShapeCapsule)
            manifold = jph_capsule_capsule(ca, qa, ga.y, ga.w, cb, qb, gb.y, gb.w, speculative_distance);
        else
        {
            // Mixed manifolds are generated capsule-first and published in original A/B order.
            float3 capsule_center = flip ? cb : ca;
            float4 capsule_rotation = flip ? qb : qa;
            float4 capsule_geometry = flip ? gb : ga;
            float3 other_center = flip ? ca : cb;
            float4 other_geometry = flip ? ga : gb;
            if (shape_a.meta.x == ShapeSphere || shape_b.meta.x == ShapeSphere)
                manifold = jph_capsule_sphere(capsule_center, capsule_rotation, capsule_geometry.y, capsule_geometry.w,
                                              other_center, other_geometry.w, speculative_distance);
            else
                manifold = jph_capsule_box(capsule_center, capsule_rotation, capsule_geometry.y, capsule_geometry.w,
                                           other_center, flip ? qa : qb, other_geometry, speculative_distance);
        }
        if (manifold.count == 0u)
            return;
        if (flip)
            manifold.normal = -manifold.normal;
        patch.leaves.z |= (sensor ? 0u : PatchActive) |
                          (manifold.penetration >= 0.0f ? PatchOverlapping : 0u);
        patch.leaves.w = manifold.count;
        contact_patches[patch_index] = patch;
        for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
            if (point < manifold.count)
            {
                JphBoxPoint contact = manifold.points[point];
                if (flip)
                {
                    float3 point_a = contact.pointA;
                    contact.pointA = contact.pointB;
                    contact.pointB = point_a;
                }
                result.meta = uint4(a, b, sensor ? 0u : 1u, contact.feature);
                result.normalDepth = float4(manifold.normal, contact.depth);
                result.localAnchorA = float4(jph_inverse_rotate(sa.rotation, contact.pointA - sa.positionCOM.xyz), 0.0f);
                result.localAnchorB = float4(jph_inverse_rotate(sb.rotation, contact.pointB - sb.positionCOM.xyz), 0.0f);
                contacts[slot + point] = result;
            }
        return;
    }

    float3 normal, pa, pb;
    float penetration;
    uint feature = 0u;
    if (shape_a.meta.x == ShapeSphere && shape_b.meta.x == ShapeSphere)
        jph_sphere_sphere(ca, ga.w, cb, gb.w, normal, penetration, pa, pb);
    else if (shape_a.meta.x == ShapeSphere && shape_b.meta.x == ShapeBox)
        jph_sphere_box(ca, ga.w, cb, qb, gb, normal, penetration, pa, pb, feature);
    else if (shape_a.meta.x == ShapeBox && shape_b.meta.x == ShapeSphere)
    {
        jph_sphere_box(cb, gb.w, ca, qa, ga, normal, penetration, pb, pa, feature);
        normal = -normal;
    }
    else
        return;
    if (penetration < -speculative_distance)
        return;
    patch.leaves.z |= (sensor ? 0u : PatchActive) | (penetration >= 0.0f ? PatchOverlapping : 0u);
    patch.leaves.w = 1u;
    contact_patches[patch_index] = patch;
    result.meta.z = sensor ? 0u : 1u;
    result.meta.w = feature;
    result.normalDepth = float4(normal, penetration);
    result.localAnchorA = float4(jph_inverse_rotate(sa.rotation, pa - sa.positionCOM.xyz), 0.0f);
    result.localAnchorB = float4(jph_inverse_rotate(sb.rotation, pb - sb.positionCOM.xyz), 0.0f);
    contacts[slot] = result;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void setup_contacts(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint slot = stage_work.range.x + tid.x;
    if (slot >= step_params.solve.z)
        return;
    Contact contact = contacts[slot];
    if (contact.meta.z == 0u)
        return;
    ContactPatch patch = contact_patches[contact_patches[0].leaves.x + slot / MaxContactPointsPerPair];
    ShapeModel shape_a = shape_models[patch.leaves.x];
    ShapeModel shape_b = shape_models[patch.leaves.y];
    uint a = contact.meta.x;
    uint b = contact.meta.y;
    if (!jph_selected_body(a) || !jph_selected_body(b)) return;
    if (!jph_sleep_pair_awake(a,b)) return;
    BodyModel ma = jph_effective_model(body_models[a], kinematic_targets[a]);
    BodyModel mb = jph_effective_model(body_models[b], kinematic_targets[b]);
    BodyState sa = body_states[a];
    BodyState sb = body_states[b];
    float3 pa, pb, ra, rb;
    jph_contact_points(contact, sa, sb, pa, pb, ra, rb);
    float3 normal = contact.normalDepth.xyz;
    float normal_velocity = dot(jph_point_velocity(mb, sb, rb) - jph_point_velocity(ma, sa, ra), normal);
    float dt = step_params.gravityDt.w;
    float speculative_bias = max(0.0f, -contact.normalDepth.w / dt);
    float restitution = jph_combine_material(shape_a.material.y, shape_a.surface.w,
                                               shape_b.material.y, shape_b.surface.w);
    float bias = speculative_bias;
    if (restitution > 0.0f && normal_velocity < -step_params.restitution.x
        && normal_velocity < -speculative_bias)
    {
        // ContactConstraintManager compensates the force/gravity velocity
        // already added at the start of this substep before restitution.
        float3 relative_acceleration = jph_acceleration(mb, body_forces[b], step_params.gravityDt.xyz)
                                     - jph_acceleration(ma, body_forces[a], step_params.gravityDt.xyz);
        float force_delta_velocity = min(0.0f, dot(relative_acceleration, normal) * dt);
        bias = restitution * (normal_velocity - force_delta_velocity);
    }
    contact.impulse = float4(0.0f, 0.0f, 0.0f, bias);
    contact.localAnchorA.w = 0.0f;
    if (slot % MaxContactPointsPerPair == 0u)
        contact.localAnchorB.w = jph_combine_material(shape_a.material.x, shape_a.surface.z,
                                                        shape_b.material.x, shape_b.surface.z);
    contacts[slot] = contact;
}

uint jph_history_word(uint patch)
{
    return 1u + patch * ContactHistoryWords;
}

uint jph_previous_contact(uint2 pair, uint count)
{
    uint first = 0u, end = count;
    while (first < end)
    {
        uint middle = first + (end - first) / 2u;
        uint2 key = contact_history[jph_history_word(middle)].value.xy;
        if (key.x < pair.x || (key.x == pair.x && key.y < pair.y)) first = middle + 1u;
        else end = middle;
    }
    return first;
}

void jph_warm_contact(uint ordinal, uint4 header)
{
    uint4 patch = contact_patches[contact_patches[0].leaves.x + ordinal].leaves;
    uint slot = ordinal * MaxContactPointsPerPair;
    Contact contact = contacts[slot];
    uint a = contact.meta.x, b = contact.meta.y;
    uint previous = jph_previous_contact(patch.xy, header.x);
    if (previous == header.x) return;
    uint base = jph_history_word(previous);
    uint4 key = contact_history[base].value;
    if (any(key.xy != patch.xy) || key.z != body_sleep[a].runtime.timing.y || key.w != body_sleep[b].runtime.timing.y)
        return;
    float4 old_normal = asfloat(contact_history[base + 1u].value);
    uint old_count = uint(old_normal.w);
    float3 normal = contact.normalDepth.xyz;
    if (old_count == 0u || dot(normal, old_normal.xyz) < 0.95f) return;

    float dt_ratio = step_params.gravityDt.w / asfloat(header.y);
    BodyModel ma = jph_effective_model(body_models[a], kinematic_targets[a]);
    BodyModel mb = jph_effective_model(body_models[b], kinematic_targets[b]);
    BodyState sa = body_states[a], sb = body_states[b];
    float3 center = float3(0.0f), midpoints[4];
    float4 lambdas = float4(0.0f);
    uint matched = 0u;
    for (uint i = 0u; i < patch.w; ++i)
    {
        Contact current = contacts[slot + i];
        // Jolt's mContactPointPreserveLambdaMaxDistSq on both local anchors.
        constexpr float max_distance_sq = 0.0001f;
        float distance = 2.0f * max_distance_sq;
        uint selected = old_count;
        for (uint j = 0u; j < old_count; ++j)
        {
            if ((matched & (1u << j)) != 0u) continue;
            float3 delta_a = current.localAnchorA.xyz - asfloat(contact_history[base + 3u + 2u * j].value.xyz);
            float3 delta_b = current.localAnchorB.xyz - asfloat(contact_history[base + 4u + 2u * j].value.xyz);
            float da = dot(delta_a, delta_a), db = dot(delta_b, delta_b);
            if (da <= max_distance_sq && db <= max_distance_sq && da + db < distance)
            {
                selected = j;
                distance = da + db;
            }
        }
        float3 pa, pb, ra, rb;
        jph_contact_points(current, sa, sb, pa, pb, ra, rb);
        midpoints[i] = 0.5f * (pa + pb);
        center += midpoints[i];
        if (selected < old_count)
        {
            matched |= 1u << selected;
            float lambda = dt_ratio * asfloat(contact_history[base + 3u + 2u * selected].value.w);
            jph_apply_impulse(ma, sa, ra, mb, sb, rb, normal, lambda);
            lambdas[i] = lambda;
            contacts[slot + i].impulse.x = lambda;
        }
    }
    float sum = dot(lambdas, float4(1.0f));
    if (sum == 0.0f) return;

    center /= float(patch.w);
    float radius = 0.0f;
    for (uint i = 0u; i < patch.w; ++i)
    {
        float3 offset = midpoints[i] - center;
        radius += length(offset - normal * dot(offset, normal)) * lambdas[i];
    }
    float friction = contact.localAnchorB.w;
    float4 prior_friction = asfloat(contact_history[base + 2u].value) * dt_ratio;
    float3 t1, t2;
    jph_tangents(normal, t1, t2);
    float2 tangent = float2(dot(prior_friction.xyz, t1), dot(prior_friction.xyz, t2));
    float bound = friction * sum;
    float tangent_sq = dot(tangent, tangent);
    if (tangent_sq > bound * bound) tangent *= bound * rsqrt(tangent_sq);
    jph_apply_impulse(ma, sa, center - sa.positionCOM.xyz, mb, sb, center - sb.positionCOM.xyz,
                      tangent.x * t1 + tangent.y * t2, 1.0f);
    float twist = clamp(prior_friction.w, -friction * radius, friction * radius);
    contacts[slot].impulse.yz = tangent;
    contacts[slot].localAnchorA.w = twist;
    if (jph_dynamic(ma))
    {
        sa.angularVelocity.xyz -= jph_inverse_inertia(ma, sa, normal * twist);
        body_states[a] = sa;
    }
    if (jph_dynamic(mb))
    {
        sb.angularVelocity.xyz += jph_inverse_inertia(mb, sb, normal * twist);
        body_states[b] = sb;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void store_contact_history(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint ordinal = stage_work.range.x + tid.x;
    if (ordinal == 0u)
    {
        uint count = constraint_schedule[1].value.y != 0u ? step_params.solve.w : 0u;
        contact_history[0].value = uint4(count, asuint(step_params.gravityDt.w), 0u, 0u);
    }
    if (ordinal >= step_params.solve.w)
    {
        uint base = jph_history_word(ordinal);
        for (uint word = 0u; word < ContactHistoryWords; ++word)
            contact_history[base + word].value = uint4(0u);
        return;
    }
    uint4 patch = contact_patches[contact_patches[0].leaves.x + ordinal].leaves;
    uint slot = ordinal * MaxContactPointsPerPair;
    Contact contact = contacts[slot];
    uint a = contact.meta.x, b = contact.meta.y;
    bool active = (patch.z & PatchActive) != 0u && jph_sleep_pair_awake(a, b);
    uint base = jph_history_word(ordinal);
    uint count = active ? patch.w : 0u;
    contact_history[base].value = uint4(patch.xy, active ? uint2(body_sleep[a].runtime.timing.y,
                                                                body_sleep[b].runtime.timing.y) : uint2(0u));
    float3 normal = contact.normalDepth.xyz;
    contact_history[base + 1u].value = asuint(float4(normal, float(count)));
    float4 friction = float4(0.0f);
    if (active)
    {
        float3 t1, t2;
        jph_tangents(normal, t1, t2);
        friction = float4(contact.impulse.y * t1 + contact.impulse.z * t2, contact.localAnchorA.w);
    }
    contact_history[base + 2u].value = asuint(friction);
    for (uint i = 0u; i < MaxContactPointsPerPair; ++i)
    {
        Contact point = contacts[slot + i];
        contact_history[base + 3u + 2u * i].value = asuint(float4(point.localAnchorA.xyz, point.impulse.x));
        contact_history[base + 4u + 2u * i].value = asuint(point.localAnchorB);
    }
}

// One complete original patch: linear friction, angular friction, four-point normal block.
void jph_solve_patch_velocity(uint patch_index)
{
    ContactPatch patch = contact_patches[contact_patches[0].leaves.x + patch_index];
    if ((patch.leaves.z & PatchActive) == 0u)
        return;
    uint slot = patch_index * MaxContactPointsPerPair;
    // Narrowphase packs active points from the first record of a candidate.
    Contact contact = contacts[slot];
    uint a = contact.meta.x;
    uint b = contact.meta.y;
    if (!jph_selected_body(a) || !jph_selected_body(b)) return;
    if (!jph_sleep_pair_awake(a,b)) return;
    BodyModel ma = jph_effective_model(body_models[a], kinematic_targets[a]);
    BodyModel mb = jph_effective_model(body_models[b], kinematic_targets[b]);
    BodyState sa = body_states[a];
    BodyState sb = body_states[b];
    // ContactConstraintManager uses one linear-friction center per patch,
    // plus normal-axis angular friction for multiple contact points. Both
    // precede all non-penetration constraints in this PGS iteration.
    float3 normal = contact.normalDepth.xyz;
    float3 center = float3(0.0f);
    float normal_impulse = 0.0f;
    uint point_count = 0u;
    float3 ja[4];
    float3 jb[4];
    float4 old_normal_impulse = float4(0.0f);
    float4 normal_bias = float4(0.0f);
    for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
    {
        Contact current = contacts[slot + point];
        if (current.meta.z == 0u)
            continue;
        float3 pa, pb, ra, rb;
        jph_contact_points(current, sa, sb, pa, pb, ra, rb);
        ja[point] = cross(ra, normal);
        jb[point] = cross(rb, normal);
        old_normal_impulse[point] = current.impulse.x;
        normal_bias[point] = current.impulse.w;
        center += 0.5f * (pa + pb);
        normal_impulse += current.impulse.x;
        ++point_count;
    }
    center /= float(point_count);
    float friction = contact.localAnchorB.w;
    float2 linear_impulse = contact.impulse.yz;
    jph_solve_friction(ma, sa, center - sa.positionCOM.xyz, mb, sb, center - sb.positionCOM.xyz,
                       normal, friction * normal_impulse, linear_impulse);
    contact.impulse.yz = linear_impulse;

    float angular_impulse = 0.0f;
    if (point_count > 1u)
    {
        // Read the four point records again instead of holding a private
        // Contact array. Radius is measured only in the contact plane.
        float weighted_radius = 0.0f;
        for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
        {
            Contact current = contacts[slot + point];
            if (current.meta.z == 0u)
                continue;
            float3 pa, pb, ra, rb;
            jph_contact_points(current, sa, sb, pa, pb, ra, rb);
            float3 delta = 0.5f * (pa + pb) - center;
            delta -= normal * dot(delta, normal);
            weighted_radius += length(delta) * current.impulse.x;
        }
        angular_impulse = contact.localAnchorA.w;
        jph_solve_angular_friction(ma, sa, mb, sb, normal, friction * weighted_radius, angular_impulse);
    }
    contact.localAnchorA.w = angular_impulse;
    contacts[slot] = contact;

    // Solve the pair's unilateral normal QP as one block. Coupled normal
    // impulses must not leave the last scalar point's rocking residual.
    float4 next_normal_impulse = jph_solve_normal_block(ma, sa, mb, sb, normal, ja, jb,
                                                        old_normal_impulse, normal_bias, point_count);
    float delta_linear = 0.0f;
    float3 delta_angular_a = float3(0.0f);
    float3 delta_angular_b = float3(0.0f);
    for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
    {
        Contact current = contacts[slot + point];
        if (current.meta.z == 0u)
            continue;
        float delta = next_normal_impulse[point] - old_normal_impulse[point];
        delta_linear += delta;
        delta_angular_a += ja[point] * delta;
        delta_angular_b += jb[point] * delta;
        current.impulse.x = next_normal_impulse[point];
        contacts[slot + point] = current;
    }
    if (jph_dynamic(ma))
    {
        sa.linearVelocity.xyz -= delta_linear * jph_inverse_linear_mass(ma, normal);
        sa.angularVelocity.xyz -= jph_inverse_inertia(ma, sa, delta_angular_a);
        body_states[a] = sa;
    }
    if (jph_dynamic(mb))
    {
        sb.linearVelocity.xyz += delta_linear * jph_inverse_linear_mass(mb, normal);
        sb.angularVelocity.xyz += jph_inverse_inertia(mb, sb, delta_angular_b);
        body_states[b] = sb;
    }
}

bool jph_can_drift(uint index, BodyModel model)
{
    return jph_nonstatic(model) && jph_resident(model)
        && ((body_models[index].meta.z & ~MotionNotResident) == MotionStatic || body_sleep[index].runtime.state.x != 0u);
}

void jph_body_integrate(uint index, StepParams parameters)
{
    if (!jph_selected_body(index, parameters))
        return;
    BodyModel model = jph_effective_model(body_models[index], kinematic_targets[index]);
    if (!jph_can_drift(index, model)) return;
    BodyState state = body_states[index];
    float dt = parameters.gravityDt.w;
    jph_prepare_drift(model, state, dt);
    state.positionCOM.xyz += dt * state.linearVelocity.xyz;
    // Angular DOF locks remain fixed in world space as the body rotates.
    jph_project_motion(model, state);
    body_states[index] = state;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void integrate(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint index = stage_work.range.x + tid.x;
    if (jph_articulation_admitted(index, false)) return;
    jph_body_integrate(index, step_params);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void capture_sweep_motion(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint index = stage_work.range.x + tid.x;
    BodyModel model = jph_effective_model(body_models[index], kinematic_targets[index]);
    BodyState state = body_states[index];
    if (jph_can_drift(index, model))
        jph_prepare_drift(model, state, step_params.gravityDt.w);
    else
        state.linearVelocity = state.angularVelocity = float4(0.0f);
    float3 displacement = state.linearVelocity.xyz * step_params.gravityDt.w;
    SweepBody sweep;
    sweep.positionCOM = float4(state.positionCOM.xyz, 0.0f);
    sweep.rotation = state.rotation;
    sweep.linearVelocity = state.linearVelocity;
    sweep.angularVelocity = state.angularVelocity;
    sweep.displacement = float4(displacement, jph_sweep_role(model, displacement, step_params.restitution.y));
    sweep.linearResponse = float4(0.0f, 0.0f, 0.0f, 1.0f);
    sweep.angularResponse = float4(0.0f);
    sweep_bodies[index - step_params.selection.y] = sweep;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void build_swept_bounds(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint index = stage_work.range.x + tid.x;
    ShapeModel shape = shape_models[index];
    SweepBody motion = sweep_bodies[shape.meta.y - step_params.selection.y];
    CvxOperand operand = cvx_leaf_operand(shape, jph_sweep_state(motion));
    Aabb swept = cvx_leaf_bounds(convex_geometry, operand, 0.5f * step_params.contact.x);
    swept.minimum.xyz += min(float3(0.0f), motion.displacement.xyz);
    swept.maximum.xyz += max(float3(0.0f), motion.displacement.xyz);
    swept.minimum.w = motion.displacement.w;
    swept.maximum.w = motion.positionCOM.w;
    bounds[index] = swept;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void integrate_swept(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint index = stage_work.range.x + tid.x;
    if (jph_articulation_admitted(index, false)) return;
    BodyModel model = jph_effective_model(body_models[index], kinematic_targets[index]);
    SweepBody sweep = sweep_bodies[index - step_params.selection.y];
    bool impact = sweep.angularResponse.w != 0.0f || sweep.positionCOM.w != 0.0f;
    if (!impact && !jph_can_drift(index, model)) return;
    BodyState state = jph_sweep_state(sweep);
    state.positionCOM.xyz += sweep.linearResponse.w * sweep.displacement.xyz;
    state.linearVelocity.xyz += sweep.linearResponse.xyz;
    state.angularVelocity.xyz += sweep.angularResponse.xyz;
    jph_project_motion(model, state);
    body_states[index] = state;
    if (impact && body_sleep[index].runtime.state.x == 0u)
    {
        BodySleepRuntime sleep = body_sleep[index].runtime;
        float3 p0, p1, p2;
        sleep_test_points(state, body_sleep[index].author.rootHalfExtent.xyz, p0, p1, p2);
        sleep_reset(sleep, p0, p1, p2);
        sleep.state.x = 1u;
        body_sleep[index].runtime = sleep;
    }
}

void jph_solve_patch_position(uint patch_index)
{
    uint slot = patch_index * MaxContactPointsPerPair;
    Contact contact = contacts[slot];
    if (contact.meta.z == 0u)
        return;
    uint a = contact.meta.x;
    uint b = contact.meta.y;
    if (!jph_selected_body(a) || !jph_selected_body(b)) return;
    if (!jph_sleep_pair_awake(a,b)) return;
    BodyModel ma = body_models[a];
    BodyModel mb = body_models[b];
    BodyState sa = body_states[a];
    BodyState sb = body_states[b];
    // ContactConstraintManager::sSolvePositionConstraint captures both
    // transforms once for the patch. Re-linearizing the next witness after
    // each point correction gives a symmetric resting patch an artificial
    // rocking correction. Arms and inverse inertia still use the updated
    // bodies, as in the original Jolt position iteration.
    float3 position_a = sa.positionCOM.xyz, position_b = sb.positionCOM.xyz;
    float4 rotation_a = sa.rotation, rotation_b = sb.rotation;
    for (uint point = 0u; point < MaxContactPointsPerPair; ++point)
    {
        Contact current = contacts[slot + point];
        if (current.meta.z == 0u)
            continue;
        float3 pa = position_a + jph_rotate(rotation_a, current.localAnchorA.xyz);
        float3 pb = position_b + jph_rotate(rotation_b, current.localAnchorB.xyz);
        jph_solve_position_points(ma, sa, mb, sb, current.normalDepth.xyz, pa, pb,
            step_params.contact.y, step_params.contact.z, step_params.contact.w);
    }
    if (jph_dynamic(ma))
        body_states[a] = sa;
    if (jph_dynamic(mb))
        body_states[b] = sb;
}

interface ContactSolver
{
    void solve(uint source);
}

struct WarmContactSolver : ContactSolver
{
    uint4 history;
    void solve(uint source) { jph_warm_contact(source, history); }
}

struct VelocityContactSolver : ContactSolver
{
    void solve(uint source) { jph_solve_patch_velocity(source); }
}

struct PositionContactSolver : ContactSolver
{
    void solve(uint source)
    {
        jph_solve_patch_position(source);
    }
}

groupshared uint4 contact_tile;
groupshared uint4 contact_rows[IslandPatchQuantum];
groupshared uint2 contact_writer_scratch[IslandPatchQuantum];
groupshared uint contact_levels[IslandPatchQuantum];
groupshared uint contact_level_ends[IslandPatchQuantum];
groupshared uint contact_order[IslandPatchQuantum];
groupshared uint contact_level_count;

uint2 jph_contact_predecessors(uint row)
{
    const uint2 writers = contact_rows[row].yz;
    uint2 previous = uint2(0u);
    bool find_a = writers.x != ConstraintInvalid;
    bool find_b = writers.y != ConstraintInvalid;
    for (uint i = row; i > 0u && (find_a || find_b); --i)
    {
        const uint2 endpoints = contact_rows[i - 1u].yz;
        if (find_a && any(endpoints == writers.x)) { previous.x = i; find_a = false; }
        if (find_b && any(endpoints == writers.y)) { previous.y = i; find_b = false; }
    }
    return previous;
}

// Lane zero visits patches in stable order. A link points to the latest
// preceding row writing the same body; after visiting that row its two links
// hold the colors already used by each of its endpoints.
uint jph_previous_colors(uint previous, uint writer)
{
    if (previous == 0u) return 0u;
    uint row = previous - 1u;
    return contact_rows[row].y == writer ? contact_writer_scratch[row].x : contact_writer_scratch[row].y;
}

uint jph_plan_contact_tile(uint2 tile, uint lane)
{
    for (uint row = lane; row < tile.y; row += ContactSolverLanes)
    {
        contact_rows[row] = jph_island_patch_record(tile.x + row);
        contact_level_ends[row] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint row = lane; row < tile.y; row += ContactSolverLanes)
        contact_writer_scratch[row] = jph_contact_predecessors(row);
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u)
    {
        uint level_count = 0u;
        for (uint row = 0u; row < tile.y; ++row)
        {
            const uint2 previous = contact_writer_scratch[row];
            const uint2 writers = contact_rows[row].yz;
            const uint2 used = uint2(jph_previous_colors(previous.x, writers.x),
                                     jph_previous_colors(previous.y, writers.y));
            const uint available = ~(used.x | used.y);
            // A saturated writer gets an exclusive level after the 32 mask bits.
            uint level = level_count;
            uint bit = 0u;
            if (available != 0u)
            {
                level = uint(firstbitlow(available));
                bit = 1u << level;
            }
            contact_writer_scratch[row] = used | bit;
            contact_levels[row] = level;
            ++contact_level_ends[level];
            level_count = max(level_count, level + 1u);
        }
        contact_level_count = 0u;
        // A patch group dominated by one writer has little work to share between barriers.
        if (level_count <= tile.y / 2u)
        {
            uint offset = 0u;
            for (uint level = 0u; level < level_count; ++level)
            {
                const uint count = contact_level_ends[level];
                contact_level_ends[level] = offset;
                offset += count;
            }
            for (uint row = 0u; row < tile.y; ++row)
            {
                const uint index = contact_level_ends[contact_levels[row]]++;
                contact_order[index] = contact_rows[row].x;
            }
            contact_level_count = level_count;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    return workgroupUniformLoad(contact_level_count);
}

void jph_finish_contact_level()
{
    __target_switch
    {
    case wgsl: __intrinsic_asm "storageBarrier()";
    default: DeviceMemoryBarrierWithGroupSync();
    }
}

// A lane owns one complete island range. Small work windows need no shared
// memory, row planning, or barriers; overlapping writers stay in solver order.
void jph_solve_contact_range<Solver : ContactSolver>(Solver solver, uint island)
{
    uint begin, first, end;
    if (jph_island_patch_range(island, begin, first, end))
        for (uint row = first; row < end; ++row)
            solver.solve(jph_island_patch_record(begin + row).x);
}

uint jph_prepare_contact_tile(uint2 tile, uint lane)
{
    return tile.y >= ContactParallelMinimum ? jph_plan_contact_tile(tile, lane) : 0u;
}

void jph_solve_planned_contact_tile<Solver : ContactSolver>(Solver solver, uint2 tile, uint lane, uint levels)
{
    const uint first_row = tile.x;
    const uint row_count = lane == 0u ? tile.y : 0u;

    // One solver body serves both contiguous rows and dependency-ordered rows.
    // A level contains disjoint writers; later levels see all its updates.
    uint level_begin = 0u;
    for (uint level = 0u; level < max(levels, 1u); ++level)
    {
        uint row = first_row;
        uint end = first_row + row_count;
        uint stride = 1u;
        if (levels != 0u)
        {
            row = level_begin + lane;
            end = contact_level_ends[level];
            stride = ContactSolverLanes;
            level_begin = end;
        }
        for (; row < end; row += stride)
        {
            uint source;
            if (levels != 0u)
                source = contact_order[row];
            else
                source = jph_island_patch_record(row).x;
            solver.solve(source);
        }
        if (levels != 0u)
            jph_finish_contact_level();
    }
}

void jph_solve_contact_tile<Solver : ContactSolver>(Solver solver, uint2 tile, uint lane)
{
    jph_solve_planned_contact_tile(solver, tile, lane, jph_prepare_contact_tile(tile, lane));
}

uint jph_contact_tile_boundary(uint row, uint width)
{
    return row + (width - row % width) % width;
}

// (first source, row count, slice width, one island per invocation).
uint4 jph_contact_group_tile(uint group, uint lane)
{
    if (lane == 0u)
    {
        bool whole_table = stage_work.range.z == ConstraintWholeIslandTable;
        if (constraint_schedule[1].value.w < ContactParallelMinimum)
            contact_tile = uint4(0u, 0u, 0u, 1u);
        else
        {
            uint4 jr, pr;
            uint canonical;
            contact_tile = uint4(0u);
            if (jph_island_unit(group, false, jr, pr, canonical))
            {
                const uint width = ConstraintContactTileWidth(canonical, constraint_schedule[0].value.z);
                // The two consecutive windows use the same tile boundary.
                // Packing changes dispatches, keeping the writer order within each tile.
                const uint first = whole_table ? 0u : jph_contact_tile_boundary(stage_work.range.x, width);
                const uint end = whole_table ? pr.y : min(pr.y,
                    jph_contact_tile_boundary(stage_work.range.x + stage_work.range.y, width));
                if (first < end) contact_tile = uint4(pr.x + first, end - first, width, 0u);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    return workgroupUniformLoad(contact_tile);
}

void jph_solve_contact_tiles<Solver : ContactSolver>(Solver solver, uint4 work, uint lane)
{
    for (uint offset = 0u; offset < work.y; offset += work.z)
    {
        const uint size = min(work.z, work.y - offset);
        jph_solve_contact_tile(solver, uint2(work.x + offset, size), lane);
        if (offset + size < work.y) jph_finish_contact_level();
    }
}

void jph_solve_contact_group<Solver : ContactSolver>(Solver solver, uint group, uint lane)
{
    const uint4 work = jph_contact_group_tile(group, lane);
    if (work.w != 0u)
        jph_solve_contact_range(solver, group * ContactSolverLanes + lane);
    else
        jph_solve_contact_tiles(solver, work, lane);
}

[shader("compute")]
[numthreads(ContactSolverLanes, 1, 1)]
void warm_contacts(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    WarmContactSolver solver = {contact_history[0].value};
    jph_solve_contact_group(solver, group.x, lane);
}

[shader("compute")]
[numthreads(ContactSolverLanes, 1, 1)]
void solve_velocity(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    VelocityContactSolver solver;
    jph_solve_contact_group(solver, group.x, lane);
}

[shader("compute")]
[numthreads(ContactSolverLanes, 1, 1)]
void solve_position(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    PositionContactSolver solver;
    jph_solve_contact_group(solver, group.x, lane);
}

[shader("compute")]
[numthreads(ContactSolverLanes, 1, 1)]
void warm_contacts_direct(uint3 tid : SV_DispatchThreadID)
{
    WarmContactSolver solver = {contact_history[0].value};
    jph_solve_contact_range(solver, tid.x);
}

[shader("compute")]
[numthreads(ContactSolverLanes, 1, 1)]
void solve_velocity_direct(uint3 tid : SV_DispatchThreadID)
{
    VelocityContactSolver solver;
    jph_solve_contact_range(solver, tid.x);
}

void jph_solve_contact_rounds<Solver : ContactSolver>(Solver solver, uint group, uint lane, uint configured_rounds)
{
    const bool whole_table = stage_work.range.z == ConstraintWholeIslandTable;
    const uint rounds = whole_table ? stage_work.range.x : configured_rounds;
    if (whole_table && stage_work.range.y >= ContactParallelMinimum)
    {
        const uint4 work = jph_contact_group_tile(group, lane);
        if (work.w == 0u)
        {
            if (work.y == 0u) return;
            if (work.y <= work.z)
            {
                // Endpoints and writer order come from the immutable island
                // directory. All admitted rounds reuse that same tile plan.
                const uint2 tile = work.xy;
                const uint levels = jph_prepare_contact_tile(tile, lane);
                for (uint iteration = 0u; iteration < rounds; ++iteration)
                {
                    jph_solve_planned_contact_tile(solver, tile, lane, levels);
                    if (levels == 0u && iteration + 1u < rounds) jph_finish_contact_level();
                }
            }
            else
            {
                for (uint iteration = 0u; iteration < rounds; ++iteration)
                {
                    jph_solve_contact_tiles(solver, work, lane);
                    if (iteration + 1u < rounds) jph_finish_contact_level();
                }
            }
            return;
        }
    }
    uint begin, first, end;
    if (!jph_island_patch_range(group * ContactSolverLanes + lane, begin, first, end)) return;

    // This lane has all the dynamic writers of its island. Each round sees
    // the prior round's impulses in the same canonical patch order.
    for (uint iteration = 0u; iteration < rounds; ++iteration)
        for (uint row = first; row < end; ++row)
            solver.solve(jph_island_patch_record(begin + row).x);
}

[shader("compute")]
[numthreads(ContactSolverLanes, 1, 1)]
void solve_velocity_rounds(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    VelocityContactSolver solver;
    jph_solve_contact_rounds(solver, group.x, lane, step_params.counts.w);
}

[shader("compute")]
[numthreads(ContactSolverLanes, 1, 1)]
void solve_position_rounds(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    PositionContactSolver solver;
    jph_solve_contact_rounds(solver, group.x, lane, step_params.solve.x);
}

[shader("compute")]
[numthreads(ContactSolverLanes, 1, 1)]
void solve_position_direct(uint3 tid : SV_DispatchThreadID)
{
    PositionContactSolver solver;
    jph_solve_contact_range(solver, tid.x);
}

#include "rigid_sleep.slang"


// The owner admits only no-joint/no-articulation/no-participant members and
// validates disjoint tiles against the same accepted layout/allocation generation.
// Guard subtraction before addition so malformed intervals cannot wrap a slot.
bool jph_body_batch_slot(uint group, uint lane, out uint index, out StepParams parameters)
{
    index = 0u;
    parameters = (StepParams)0;
    if (group >= body_batch_header.counts.y) return false;
    BodyBatchTile tile = body_batch_tiles[group];
    if (tile.selectionIndex >= body_batch_header.counts.x || tile.bodyCount == 0u
        || tile.bodyCount > 64u || lane >= tile.bodyCount) return false;
    parameters = body_batch_selections[tile.selectionIndex];
    if (parameters.counts.x != body_batch_header.counts.z || parameters.counts.y != body_batch_header.counts.w
        || parameters.selection.x >= parameters.counts.y || parameters.selectedCounts.z != 0u) return false;
    if (!jph_selected_body_slot(tile.bodyFirst, parameters)) return false;
    uint selectedOffset = tile.bodyFirst - parameters.selection.y;
    if (tile.bodyCount > parameters.selectedCounts.x - selectedOffset
        || tile.bodyCount > parameters.counts.x - tile.bodyFirst) return false;
    index = tile.bodyFirst + lane;
    return jph_selected_body(index, parameters);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void body_batch_prepare_kinematics(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    uint index;
    StepParams parameters;
    if (jph_body_batch_slot(group.x, lane.x, index, parameters))
        jph_body_prepare_kinematics(index, parameters);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void body_batch_apply_forces(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    uint index;
    StepParams parameters;
    if (jph_body_batch_slot(group.x, lane.x, index, parameters))
        jph_body_apply_forces(index, parameters);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void body_batch_build_bounds(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    if (group.x >= body_batch_header.counts.y) return;
    BodyBatchTile tile = body_batch_tiles[group.x];
    if (tile.selectionIndex < body_batch_header.counts.x && lane < tile.bodyCount)
        jph_build_leaf_bounds(tile.bodyFirst + lane, body_batch_selections[tile.selectionIndex]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void body_batch_integrate(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    uint index;
    StepParams parameters;
    if (jph_body_batch_slot(group.x, lane.x, index, parameters))
        jph_body_integrate(index, parameters);
}
